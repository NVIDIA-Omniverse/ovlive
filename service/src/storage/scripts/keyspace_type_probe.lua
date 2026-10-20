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

-- keyspace_type_probe.lua — pre-MULTI keyspace type probe for the
-- `update_layer` WATCH/MULTI/EXEC slow path (b.459) and for
-- both object-materialization MULTIs (b.zvd): `LayerStore::write_object`'s
-- CreateLayer one and `LayerStore::write_restored_object`'s WATCHed DR-restore
-- one.
--
-- ── Why the slow path needs this at all ───────────────────────────────
--
-- The Lua fast path TYPE-probes every key it will write in a validation
-- pre-pass before its first mutation (`update_layer_fast.lua`, "read-only
-- keyspace-shape probe"), so an out-of-band writer that left a string on a
-- field key cannot make it half-commit. The slow path could: a WRONGTYPE
-- raised by a queued command inside `EXEC` does NOT abort the transaction
-- and does NOT roll back its siblings — every other queued command still
-- runs — and fred surfaces only the first error frame, so the caller never
-- learns which siblings landed. The only place that commit can still fail
-- whole is before `MULTI`, which is where this runs.
--
-- ── Why `TYPE` and why no `pcall` ─────────────────────────────────────
--
-- `TYPE` replies with a status reply, surfaced to Lua as a table carrying
-- an `ok` field. It is read-only and it never errors, on any key of any
-- type, so — unlike `stream_collision_probe.lua`'s `XRANGE` — there is no
-- error reply to trap and no allowlist to apply. Identical reasoning to
-- the fast path's `wrong_type` helper, which this mirrors deliberately so
-- the two backends refuse the same keyspace with the same tag and the same
-- sentence shape (the fast path additionally appends a locating qualifier
-- such as `(mutation record 3)`, which this probe has no notion of).
--
-- Being read-only also keeps the script clear of every server-wide
-- first-write gate (`-OOM`, `-READONLY`, `-MISCONF`, `-NOREPLICAS`): those
-- are checked against write commands, so nothing transient can be
-- mistaken here for a keyspace anomaly.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS:  ONE CHUNK of the keys this commit will write, deduped by the
--        caller. All carry the `{object_id}` hash tag, so they share one
--        Cluster slot with each other and with the keys the caller
--        WATCHes.
-- ARGV:  parallel array of expected type names — `"hash"`, `"set"` or
--        `"zset"` — one per KEYS entry.
--
-- Returns `nil` when every key is either absent (the commit creates it) or
-- already the expected type. Otherwise a `WRONGTYPE_KEY:`-tagged error
-- reply naming the first offending key, its actual type and the expected
-- one — the same tag and the same sentence shape the fast path raises, so
-- both backends land on one Rust arm and one message prefix.
--
-- ── A chunk, not the whole commit (b.zvd) ─────────────────────────────
--
-- `LayerStore::probe_keyspace_types` splits the caller's deduped key set
-- at `KEYSPACE_PROBE_CHUNK_SIZE` and invokes this script once per chunk,
-- with at most `KEYSPACE_PROBE_CONCURRENCY` (4) invocations in flight,
-- stopping at the first refusal. The `update_layer` caller's key set is
-- bounded by one delta, but the two materialization callers' sets are bounded
-- only by the object — up to `MAX_DELTA_ENTRIES` (99,999) fields per upload on
-- `write_object` (the CreateLayer entry point), and NO per-call cap at all on
-- `write_restored_object` (the DR-restore one), whose state is the whole
-- accumulated object. As one command that
-- overshoots the same per-command argument ceiling `UNLINK_CHUNK_SIZE`
-- exists for, and holds Valkey's single Lua thread for the whole
-- `for i = 1, #KEYS` loop.
--
-- Concurrency is bounded rather than serial because on the `update_layer`
-- path the caller runs this inside its header `WATCH`: elapsed probe time
-- is OCC window there, so a serial chunk loop would buy `EXEC`
-- invalidations. It is bounded rather than unlimited because each
-- invocation holds the single Lua thread for its own loop — an unbounded
-- fan-out would trade one long stall for many overlapping ones.
--
-- This script needs no knowledge of the split and has none: it is
-- read-only, so a chunked probe answers exactly what an un-chunked one
-- would, and every chunk completes before the caller opens its `MULTI`.
-- The caller awaits each wave whole, so the <= 3 read-only siblings already
-- in flight when a refusal is found complete harmlessly; no later wave is
-- issued. The `WRONGTYPE_KEY:` message names the offending KEY, never its
-- ordinal within the commit, and the caller scans a wave's results IN
-- ORDER, so the key a refusal reports is the one an un-chunked probe would
-- have reported — not whichever chunk happened to answer first.
--
-- ── Who builds KEYS, and what each caller leaves out ──────────────────
--
-- `mutation_probe_keys` / `collect_probe_keys` for the `update_layer`
-- slow path; `create_probe_keys` for BOTH materialization entry points
-- (`write_object` and `write_restored_object` build their key set through
-- it). All live in src/storage/layer.rs and each argues its own exclusions
-- there:
--
--   * update — keys the commit only `DEL`s (`DEL` is type-agnostic), and
--     `SetChildren`'s list key (`DEL`'d before its `RPUSH`).
--   * materialization — the per-parent `obj:ch:{parent_id}` lists (same
--     `DEL`-before-`RPUSH` heal), and the stream key. The stream-key
--     exclusion holds on both entry points, for two different reasons:
--     `write_object` covers it with its own pre-`MULTI` `EXISTS` guard, and
--     `write_restored_object` never writes the stream at all (the restore
--     path adjudicates it separately in src/checkpoint/recovery.rs), so
--     there is nothing there for a WRONGTYPE to half-commit.
--
-- The stream key is likewise out of scope on the update path, where
-- `stream_collision_probe.lua` covers it. That is why `ProbedType` has
-- neither a `List` nor a `Stream` variant.

if #KEYS ~= #ARGV then
  -- Rust-side encoder bug: the parallel arrays diverged. Fail loudly
  -- rather than probe a prefix and report clean.
  return redis.error_reply(
    "ENCODING: keyspace_type_probe got " .. #KEYS .. " keys and " .. #ARGV ..
    " expected types")
end

for i = 1, #KEYS do
  local actual = redis.call("TYPE", KEYS[i])["ok"]
  if actual ~= ARGV[i] and actual ~= "none" then
    return redis.error_reply(
      "WRONGTYPE_KEY: " .. KEYS[i] .. " holds a " .. actual ..
      ", expected " .. ARGV[i])
  end
end

return nil
