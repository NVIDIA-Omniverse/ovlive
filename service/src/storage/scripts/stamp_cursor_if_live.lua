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

-- stamp_cursor_if_live.lua — conditional `obj:hdr` checkpoint-cursor stamp
-- (b.vn6).
--
-- ── The race this closes ──────────────────────────────────────────────────
--
-- Every producer of a durable layer generation finishes by stamping the
-- cursor columns onto `le:{object_id}:obj:hdr` — `last_checkpoint_seq`,
-- `last_checkpoint_ms`, and (on the two producers that retire a restore
-- correction) `restore_seq_bump`. A bare `HSET` is unconditional: on a key
-- that no longer exists it CREATES it, and what it creates is a header
-- carrying only cursor columns — no `seq`, no `created_by`, none of the
-- twelve fields the materialization entry points (`create_layer` →
-- `write_object`, and the DR restore's `write_restored_object`) write in one
-- MULTI/EXEC.
--
-- `DeleteLayer` is a sequence of independent commands inside a retry helper,
-- not a transaction, so there is no transaction for a stamp site to join: the
-- layer mapping is deleted first and the header is UNLINKed part-way through
-- the cascade. A generation whose PUT was in flight across that window
-- therefore stamps over a grave and resurrects it as a phantom. Observed in
-- dev (b.vn6): eight layers created and deleted inside ~1 s each left a
-- cursor-only `obj:hdr`, which then made the checkpoint loop WARN
-- "missing/unparseable obj:hdr seq" every tick and made `diff_read` default
-- fields, for the ~5 minutes until the Valkey orphan sweep reaped it.
--
-- Only a server-side atomic conditional write closes that. This is it.
--
-- ── Why `HEXISTS ... seq` rather than `EXISTS` ────────────────────────────
--
-- `create_layer` writes all twelve canonical header fields — `seq` among them
-- — in a single MULTI/EXEC `HSET` through `write_object`, and
-- `write_restored_object` (the restore path) does the same inside its WATCHed
-- transaction, so `seq` is never transiently absent on a live header. Against a
-- live layer the two predicates are therefore equivalent; against a header
-- that is ALREADY a phantom (a pre-fix residual still sitting in a deployed
-- keyspace) `HEXISTS seq` is strictly stronger — it refuses to stamp, so the
-- fix is self-healing over the garbage the bug already produced rather than
-- perpetuating it. It is also exactly the predicate the checkpoint loop
-- already uses to refuse a layer whose header carries no `seq`.
--
-- ── Why Lua rather than WATCH/MULTI/EXEC ──────────────────────────────────
--
-- This is a single-key conditional write. The transactional form would cost
-- three-plus round-trips, a slot-pinned `acquire_txn_for` checkout with its
-- DISCARD/UNWATCH hygiene probe, and one of the bounded `txn_pool` slots —
-- all to express what one EVALSHA expresses atomically. `alloc_object_id.lua`
-- is the repo precedent for exactly this trade.
--
-- Cluster-safe by construction: one key, therefore one slot.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  header_key   (`le:{object_id}:obj:hdr`)
--
-- ARGV, in one of two shapes told apart by length parity:
--
--   EVEN length — the plain stamp:
--     [1..] flat field/value pairs to stamp — `field, value, field, value, …`.
--           What the checkpoint loop's three arms send
--           (`storage::lua::stamp_obj_header_cursor`), unchanged.
--
--   ODD length — the stamp with a cursor-regression floor:
--     [1]   `refuse_at_or_above` — a decimal target sequence.
--     [2..] the flat field/value pairs, as above.
--           What the base bootstrap sends
--           (`storage::lua::stamp_obj_header_cursor_below`), with its target
--           `snap_seq`.
--
-- The pair list is non-empty either way; the Rust helpers build it from the
-- producer's own cursor-column list, which is two or three pairs, so an even
-- ARGV is always at least four elements and an odd one at least five.
--
-- ── The optional floor: guard (b)'s seq half, made atomic ────────────────
--
-- The base bootstrap must not stamp `last_checkpoint_seq` when a generation
-- at or above its own target has already been stamped (its guard (b),
-- `checkpoint::base_bootstrap::raw_cursor_blocks_stamp`). Checked from Rust,
-- that is a read and then this write — two commands — so two bootstrap runs
-- for ONE object (the run of the call that created it, and the re-spawn of a
-- client re-send carrying the same operation id) can both pass the read
-- before either stamps, and both report a persisted generation. Checking it
-- here, in the same atomic step as the write, closes that window: the second
-- run finds the first run's `last_checkpoint_seq` and is refused.
--
-- The predicate is the Rust one's: an ABSENT `last_checkpoint_seq` permits the
-- stamp; a present one refuses when it is at or above the target, and also
-- when it is not a plain decimal — present-but-unreadable is the conservative
-- direction there too (declining to stamp costs one redundant re-snapshot,
-- stamping over a landed generation costs a regressed cursor).
--
-- The even shape never reaches that code: the checkpoint loop's arms keep
-- their own equality semantics (a chain-collapse re-base re-stamps the very
-- `to_seq` its capture already stamped), which this floor would break.
--
-- ── Return value ─────────────────────────────────────────────────────────
--
--   1  the header was live (`seq` present), no floor refused, and every pair
--      was written.
--   0  the header carries no `seq` — deleted, never created, or already a
--      phantom — and NOTHING was written. The caller must treat this as
--      "the layer died under me": it may not report a stamped cursor, and at
--      the creation-time bootstrap it owes the just-written blob a reap.
--      Checked first, on both shapes.
--   2  odd shape only: the header was live but its `last_checkpoint_seq` is
--      at or above the floor (or unreadable), and NOTHING was written. A
--      generation that covers this one was already stamped.
if redis.call("HEXISTS", KEYS[1], "seq") == 0 then
  return 0
end

local first_pair = 1
if #ARGV % 2 == 1 then
  first_pair = 2
  local floor = tonumber(ARGV[1])
  if floor == nil then
    return redis.error_reply("ENCODING: refuse_at_or_above is not a number")
  end
  local observed = redis.call("HGET", KEYS[1], "last_checkpoint_seq")
  if observed then
    if not string.match(observed, "^%d+$") or tonumber(observed) >= floor then
      return 2
    end
  end
end

redis.call("HSET", KEYS[1], unpack(ARGV, first_pair))
return 1
