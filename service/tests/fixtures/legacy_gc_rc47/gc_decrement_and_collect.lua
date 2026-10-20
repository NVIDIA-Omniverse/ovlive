-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
--
-- ── VENDORED TEST FIXTURE (b.1jt) — never loaded by the service ──────────
--
-- A verbatim copy of `src/storage/scripts/gc_decrement_and_collect.lua` at commit bf32b533
-- (0.2.0-rc.47, the last build before b.yuo's record-aware GC: a record-blind
-- GC with an unmarked leader lock).
-- Every earlier release runs the same code: 0.1.0 and 0.1.1 differ from it
-- only in headers and comments. `tests/integration/storage/b_1jt_rollout_tests.rs`
-- runs it against the store exactly as an older build's replica would, to
-- prove this build's state is safe in a mixed-build store. Only this block was
-- added; everything below it is the older build's file, unchanged.

-- gc_decrement_and_collect.lua — atomic external-value GC step (b.kf1 C-5).
--
-- Collapses what was previously two unguarded round-trips:
--
--     HINCRBY ext_meta ref_count -1               -- step (2)
--     if new_ref <= 0 then                        -- step (3)
--       MULTI; DEL ext_meta; DEL ext_data; EXEC
--     end
--
-- into a single atomic Lua call. Because Valkey runs Lua under its single
-- server thread, no other command — including a concurrent writer's
-- `HINCRBY ext_meta ref_count +1` from `store_values`, or a parallel
-- GC sweep on a duplicate pending entry — can interleave between the
-- decrement and the conditional delete. That is the exact crash window
-- the b.kf1 C-5 finding called out: a process death between (2) and (3)
-- left ref_count decremented but keys alive, and the startup drain then
-- re-queued the hash, causing a second decrement on retry that double-
-- subtracted the reference and silently deleted blobs another live
-- layer still pointed at.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  ext_meta:{hash_hex}   (the hash whose ref_count we're decrementing)
--   [2]  ext_data:{hash_hex}   (deleted only if ref_count drops to <= 0)
--
-- ARGV: (none)
--
-- Both keys MUST share the `{hash_hex}` Cluster slot (see KeySchema::ext_meta /
-- ext_data). The caller — `ValkeyExternalValueStore::process_pending` — looks
-- both keys up via the same `hash_hex` so this is guaranteed by construction.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Two-element table:
--   [1] action — "deleted" if both keys were unlinked with a non-negative
--                final ref_count, "deleted_underflow" if both keys were
--                unlinked AND the post-decrement ref_count went below zero
--                (a synthetic double-decrement / corruption / replay signal
--                surfaced for operator alerting via b.j9j item 4), or
--                "decremented" if the ref_count remains positive.
--   [2] new_ref_count — the integer ref_count after the decrement (negative
--                values are possible after a crash-replay double-decrement;
--                see the "retry-idempotent" paragraph below).
--
-- Note on `deleted_underflow`: the script *still* DELs both keys when the
-- ref_count goes negative — defensive cleanup matches the pre-b.j9j contract,
-- under which a negative count was deleted exactly like zero. The distinct
-- action label exists purely so the caller can fire a
-- `gc_underflow_total` counter and a `redis.log` WARNING; data-plane
-- behaviour is unchanged from the previous `"deleted"` path.
--
-- ── Retry-idempotent (the b.kf1 C-5 invariant) ─────────────────────────────
--
-- A Valkey transient I/O blip on the EVALSHA round-trip causes the caller's
-- `process_pending` loop to leave the hash in `ext:inflight`. The startup
-- drain (`drain_inflight_to_pending`) re-queues the hash on the next process
-- start, and the next sweep tries this script again. There are two cases:
--
--   1. The earlier EVAL completed server-side but the reply got lost on the
--      wire. The keys may already be deleted. The retry's HINCRBY on a
--      missing meta key creates `ref_count = -1`, the `<= 0` branch runs
--      `DEL` on already-absent keys (no-op), and we return ("deleted", -1).
--      The caller SREMs `ext:inflight`, exactly as it would have if the
--      first EVAL's reply had been delivered.
--
--   2. The earlier EVAL never reached the server (TCP died before the
--      command was sent). The keys are still live. The retry decrements
--      cleanly and behaves identically to a fresh sweep.
--
-- Either case converges on the correct end state with a bounded number of
-- retries. Lua atomicity guarantees no other writer sees the partially-
-- mutated state during the script's execution.
--
-- ── Invariant: this script must never touch `last_store_ms` (b.rej) ────────
--
-- `ext_store_value.lua` stamps `last_store_ms` on every pre-flight store, and
-- the reference-count reconciler (`storage::ext_reconcile`) will only judge a
-- meta whose stamp is a full grace old. A decrement that re-stamped would
-- refresh the field on every GC pass over a busy hash, so the reconciler would
-- find it perpetually too fresh and the surplus it exists to drain would stand
-- forever. The `DEL` below removes the field along with the rest of the meta at
-- zero; that deletion is the ONLY interaction this script may have with it.

local meta_key = KEYS[1]
local data_key = KEYS[2]

-- HINCRBY auto-creates a missing key with field=0 then increments. With a
-- delta of -1 and a missing key that yields ref_count=-1, which the
-- `<= 0` branch correctly classifies as DELETE. This is the retry-
-- idempotency mechanism described in the header.
local new_ref = redis.call("HINCRBY", meta_key, "ref_count", -1)

if new_ref <= 0 then
  -- Atomic within the script: both DELs run before the next external
  -- command. UNLINK would be slightly cheaper but the ext_meta hash is
  -- tiny (two fields: `ref_count` and `last_store_ms`) and the data blob
  -- may be large; UNLINK semantics
  -- here would defer the data free into a background thread, which is
  -- harmless but offers no atomicity benefit because we're already
  -- inside the Lua atomic block. Use DEL for parity with the previous
  -- MULTI/EXEC formulation so wire-level monitors / replication see the
  -- same command stream.
  redis.call("DEL", meta_key, data_key)
  if new_ref < 0 then
    -- b.j9j item 4: surface negative ref_count to operators. The post-
    -- decrement value going below zero means either (a) a crash-replay
    -- double-decrement landed on a single-ref blob whose meta was
    -- already DEL'd (the documented retry-idempotency case — benign),
    -- or (b) two GC paths concurrently decremented a hash that had only
    -- been ext-released once (the b.j9j root cause — corruption). The
    -- script cannot tell the two apart from inside Lua, so we surface
    -- the signal unconditionally and let the operator dashboard
    -- (gc_underflow_total) reveal a sustained underflow rate as the
    -- corruption case.
    redis.log(redis.LOG_WARNING,
      "gc_decrement_and_collect: ref_count underflow (post-decrement = " ..
      tostring(new_ref) .. ") for meta_key=" .. tostring(meta_key))
    return { "deleted_underflow", new_ref }
  end
  return { "deleted", new_ref }
end

return { "decremented", new_ref }
