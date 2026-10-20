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
-- ── VENDORED TEST FIXTURE (b.pa1) — never loaded by the service ──────────
--
-- A verbatim copy of `src/storage/scripts/gc_decrement_and_collect.lua` at commit f956c371,
-- the last build before b.pa1 added the out-of-memory opener (`ZREM applied
-- ":oom-gate"` before each branch's first write) and the third reply element
-- (`freed_bytes`). `tests/integration/storage/oom_collection_test.rs` runs it
-- beside the current script over identical planted states and requires the
-- same `(action, n)` and the same stored keys from both: the opener changes
-- no result. Only this block was added; everything below it is the older
-- build's file, unchanged.

-- gc_decrement_and_collect.lua — atomic external-value GC step (b.kf1 C-5),
-- applied at most once per release member (b.yuo).
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
-- decrement and the conditional delete.
--
-- ── At most one decrement per release member (b.yuo) ──────────────────────
--
-- A release member (`{hash_hex}:{object_id}:{event}`) owes exactly one
-- `ref_count` decrement. The call that runs this script is NOT executed exactly
-- once, and before b.yuo this script was wrongly described as safe to re-run:
--
--   * fred re-sends a command whose connection dropped after the server ran it
--     but before the reply arrived (the maintenance pool keeps fred's retries);
--   * a member whose completion failed, or whose sweeper died after the
--     decrement, stays in `inflight`, and a later pass re-claims it in place
--     (`gc_reclaim_stale.lua`) and runs the decrement again.
--
-- A second run against a hash still referenced elsewhere decremented a live
-- reference — and at zero deleted a blob a layer still pointed at: attachment
-- data loss, not the benign `-1` on a missing meta the old header argued from.
--
-- Two mechanisms close it, both checked here, on the hash's own slot, at
-- execution time:
--
--   1. **The replay record** (KEYS[3], `le:ext:applied:{hash_hex}`, ZSET:
--      member → claim time). Every applied decrement writes it in the same
--      atomic step. A run that finds its member there answers `replayed` and
--      decrements nothing. It is a separate key on the `{hash_hex}` slot — not
--      a field of the meta, which is deleted at zero and would forget the
--      record, letting a replay decrement a blob re-stored in between.
--   2. **The claim-age fence** (ARGV[2], ARGV[3]). The caller passes the time
--      its claim was taken (`gc_claim_pending.lua`'s or
--      `gc_reclaim_stale.lua`'s reply, the unencoded claim time) and the
--      fence F.
--      A run that executes more than F after that claim, by this primary's
--      clock, answers `stale` and writes nothing. However late a copy is
--      written — by fred after its caller gave up, or by a sweeper paused
--      after its claim — it cannot act after the member was re-claimed and
--      its record torn down: the record is kept until the member has been
--      settled for the replay-record retention U, and U > F + clock skew.
--
-- The record is torn down by `gc_unmark.lua`, which the GC loop runs once the
-- member has left both `pending` and `inflight` (it is "settled") and U has
-- passed. Leak, never over-release: `stale` leaves the member in `inflight`
-- for a later pass to re-claim, `replayed` writes nothing, and a record the
-- teardown cannot prove spent is kept.
--
-- Clock assumption (cluster only; standalone runs one clock): the fence
-- compares this `{hash_hex}` primary's `TIME` against a claim time from the
-- `{gc}` primary. A primary running at least F ahead of the `{gc}` primary
-- refuses its decrements as `stale` (a leak, visible on
-- `live_edit_gc_decrement_total{outcome="stale"}`); the replay guarantee needs
-- no primary to run behind the `{gc}` primary by U − F or more.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS (all three share the `{hash_hex}` Cluster slot — see
-- `KeySchema::ext_meta` / `ext_data` / `ext_applied`; the caller builds all
-- three from the same `hash_hex`, so this holds by construction):
--   [1]  ext_meta:{hash_hex}      (the hash whose ref_count is decremented)
--   [2]  ext_data:{hash_hex}      (deleted only if ref_count drops to <= 0)
--   [3]  ext:applied:{hash_hex}   (the replay record, ZSET)
--
-- ARGV:
--   [1]  member     (the release member being applied)
--   [2]  claim_ms   (decimal ms; the claim's time, from the `{gc}` primary)
--   [3]  fence_ms   (decimal ms; F)
--
-- In order:
--   1. Record present for the member: raise its score to
--      max(score, claim_ms) and return `replayed` with the current ref_count
--      (an absent meta reads 0). Nothing else is written; the meta is never
--      created.
--   2. Else, if now − claim_ms > fence_ms: return `stale`, no write.
--   3. Else: `HINCRBY ref_count -1` and `ZADD applied claim_ms member`. At
--      <= 0, `DEL meta data` (never the applied key) and return `deleted`
--      (or `deleted_underflow` below zero); otherwise `decremented`.
--
-- The record read comes first, so a wrong-typed applied key raises before any
-- write; the first write is the `HINCRBY`, so a store write refusal leaves the
-- keyspace untouched.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Two-element table `{ action, n }`:
--   "decremented"        n = the new ref_count (> 0); keys retained.
--   "deleted"            n = the new ref_count (0); meta and data deleted.
--   "deleted_underflow"  n = the new ref_count (< 0); meta and data deleted.
--                        A decrement with no increment behind it: an older
--                        build's replay, a legacy bare member, or a bug.
--                        Replays now answer `replayed`, so this has a zero
--                        baseline; the caller ticks `gc_underflow_total`.
--   "replayed"           n = the current ref_count; nothing written.
--   "stale"              n = now − claim_ms, the claim's age in ms; nothing
--                        written.
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

local meta_key    = KEYS[1]
local data_key    = KEYS[2]
local applied_key = KEYS[3]
local member      = ARGV[1]
local claim_ms    = tonumber(ARGV[2])
local fence_ms    = tonumber(ARGV[3])

if member == nil or claim_ms == nil or fence_ms == nil then
  return redis.error_reply("ENCODING: expected ARGV member, claim_ms, fence_ms")
end

-- 1. The replay record.
local recorded = redis.call("ZSCORE", applied_key, member)
if recorded then
  if claim_ms > tonumber(recorded) then
    redis.call("ZADD", applied_key, claim_ms, member)
  end
  local rc = tonumber(redis.call("HGET", meta_key, "ref_count")) or 0
  return { "replayed", rc }
end

-- 2. The claim-age fence, on this primary's clock at execution time.
local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local claim_age = now_ms - claim_ms
if claim_age > fence_ms then
  return { "stale", claim_age }
end

-- 3. Apply, and record it in the same atomic step.
local new_ref = redis.call("HINCRBY", meta_key, "ref_count", -1)
redis.call("ZADD", applied_key, claim_ms, member)

if new_ref <= 0 then
  -- Both DELs run before the next external command. The applied key is NOT
  -- deleted: it must outlive the meta so a replay after a re-store still
  -- answers `replayed`.
  redis.call("DEL", meta_key, data_key)
  if new_ref < 0 then
    redis.log(redis.LOG_WARNING,
      "gc_decrement_and_collect: ref_count underflow (post-decrement = " ..
      tostring(new_ref) .. ") for meta_key=" .. tostring(meta_key))
    return { "deleted_underflow", new_ref }
  end
  return { "deleted", new_ref }
end

return { "decremented", new_ref }
