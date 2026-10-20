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

-- stream_collision_probe.lua — pre-MULTI stream-ID collision guard for the
-- `update_layer` WATCH/MULTI/EXEC slow path (b.ph9).
--
-- ── Why the slow path needs a probe and not a reordering ───────────────
--
-- The Lua fast path fixes the same hazard by issuing its `XADD` first, so
-- a rejected stream write leaves the keyspace untouched. That move is
-- inert inside a MULTI: Valkey's transactions are atomic in *isolation and
-- queueing*, not in execution. A command that fails at execution time does
-- not abort the transaction and does not roll back its siblings — every
-- queued command still runs, and fred merely surfaces the first nested
-- error frame from the EXEC array as `Err`. Ordering the `XADD` first
-- inside the MULTI would therefore change nothing at all.
--
-- The only place a slow-path commit can still fail whole is *before*
-- `MULTI`. Hence this probe, paired with a `WATCH` on the same stream key
-- so that anything touching the stream between the probe and the `EXEC`
-- cancels the transaction into the existing Aborted → retry machinery.
--
-- ── Why XRANGE and not XINFO / XREVRANGE ──────────────────────────────
--
-- `XINFO STREAM` has no metadata-only form: the plain form embeds
-- `first-entry` and `last-entry` in full and the `FULL` form embeds
-- `COUNT` entries. Per-object layer-stream entries carry the writer's raw
-- inbound OMPF (b.gze), bounded only by `LIVE_EDIT_MAX_LAYER_SIZE`
-- (100 MB by default), so that shape is far too expensive to run on a
-- commit path — see the header of `stream_high_water_probe.lua`, which
-- accepts that cost precisely once per DR recovery and warns against
-- reuse on a hot path.
--
-- `XREVRANGE`'s tail ID is also the wrong value: it drops back after an
-- `XDEL` of the top entry, where `last-generated-id` holds
-- (`src/checkpoint/recovery.rs` records the case).
--
-- `XRANGE <new_seq>-0 + COUNT 1` answers exactly the question that
-- matters — "does an entry already exist at or above the ID we are about
-- to write?" — touching at most one entry, and this script projects the
-- reply down to that entry's ID server-side so the payload never crosses
-- the wire.
--
-- ── Exactness caveat ──────────────────────────────────────────────────
--
-- This detects "an entry exists at or above `new_seq`", which is not the
-- same predicate as "`last-generated-id` >= `new_seq`". The two diverge
-- only after an `XDEL` of the top entry or an `XSETID`, and no code path
-- in this service performs either — the only trimming is `XADD MAXLEN ~`
-- and the relay byte budget's `XTRIM MINID ~` (whose cut never reaches the
-- top entry), both of which trim from the head, never the tail. So within
-- this codebase the two predicates coincide, and the cheap one is the right
-- trade.
--
-- ── Why the header is re-read here, atomically ────────────────────────
--
-- "The stream holds an entry at or above `<new_seq>-0`" is on its own a
-- USELESS predicate, because the ordinary optimistic-concurrency loss
-- satisfies it. Two writers WATCH at `seq = N` and both plan `N+1`; the
-- winner commits and appends `N+1-0`; the loser's probe then sees an entry
-- at its intended ID and — under a naive check — would report a
-- catastrophic keyspace anomaly for what is in fact the single most
-- routine event on this code path. (Measured: the 50-concurrent-writer
-- regression test turns ~30 of 50 writes into false collisions.)
--
-- The pathology is not "an entry exists at `<new_seq>-0`". It is "the
-- stream's top has run AHEAD OF THE HEADER" — the invariant a live object
-- must satisfy is that its stream top is exactly `{header.seq}-0`. So the
-- caller passes the header sequence it WATCHed, and this script re-reads
-- the header inside its own atomic block:
--
--   * header still at `<new_seq> - 1`, and an entry exists at or above
--     `<new_seq>-0`  →  genuine anomaly, observed atomically. Fail closed.
--   * header has moved  →  someone committed after the caller's WATCH.
--     Report clean and let the caller proceed: its WATCH on the header is
--     already invalidated, so the `EXEC` cancels and the ordinary
--     `Aborted` → retry path re-runs everything against fresh state. This
--     script deliberately does NOT invent a second retry signal for a
--     condition the transaction already handles.
--
-- The header read must be in the same script as the `XRANGE`, not a
-- separate round-trip, or the two observations could straddle a commit
-- and resurrect the false positive from the other side.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  per-object stream key — `le:{object_id}:obj:stream`
--        (see KeySchema::layer_stream).
--   [2]  object header hash — `le:{object_id}:obj:hdr`
--        (see KeySchema::obj_header).
--
--   Both carry the `{object_id}` hash tag, so they share one Cluster slot
--   with each other and with the keys the caller WATCHes.
--
-- ARGV:
--   [1]  the stream ID the caller intends to write, `"<new_seq>-0"`.
--   [2]  the header `seq` the caller read under its WATCH, decimal. The
--        probe is meaningful only while the header still holds this value.
--
-- ── Return value ──────────────────────────────────────────────────────
--
-- `nil`  — proceed. Either no entry exists at or above ARGV[1] (a missing
--          stream key answers this way too: `XRANGE` on a missing key is
--          an empty range, not an error), or the header has moved since
--          the caller read it, which the caller's WATCH already covers.
--
-- Otherwise a two-element array `{id, kind}`:
--   [1]  the ID of the lowest colliding entry, verbatim.
--   [2]  `"deleted"` when that entry is the layer-retirement tombstone
--        (`payload` field exactly `deleted`, written by `delete_layer` and by
--        the `CreateLayer`-overwrite path), otherwise the empty string.
--
-- The caller fails the commit closed before opening its MULTI — unless [2]
-- says the entry is a tombstone, in which case the "collision" is an artifact
-- of a layer being retired underneath an in-flight commit and the caller
-- surfaces the ordinary deleted / superseded shape instead (b.ph9 review).
-- Reading [2] is free: `redis.pcall('XRANGE', ...)` has already materialized
-- the whole entry inside the Lua VM, so the projection below is about what
-- crosses the WIRE, not about what the server touches.
--
-- On a keyspace-level error (`WRONGTYPE` — something other than a stream
-- occupies the key), the script returns a tagged error reply rather than
-- `nil`. That case is not "clean": the `XADD` inside the MULTI would fail
-- the same way and half-commit, which is the whole condition this probe
-- exists to prevent, so it must fail closed too.
--
-- ── Why the WRONGTYPE tag is an allowlist here too ────────────────────
--
-- The fast path's `XADD` is a *write*, and being its script's first write
-- it absorbs every server-wide first-write gate — `-OOM`, `-READONLY`,
-- `-MISCONF`, `-NOREPLICAS` — so tagging its rejections by exclusion
-- mis-reported transient server conditions as keyspace anomalies
-- (post-completion review of b.ph9). This probe is NOT exposed to that
-- class: both of its commands (`HGET`, `XRANGE`) are read-only, carrying
-- neither `CMD_WRITE` nor `CMD_DENYOOM`, so `scriptVerifyWriteCommandAllow`
-- returns early on the non-write check and `scriptVerifyOOM` on the
-- missing deny-oom flag (valkey `src/script.c`). Verified, not assumed.
--
-- What can still reach the `pcall` branch is the residue —
-- `scriptVerifyACL` (`-NOPERM`), `scriptVerifyClusterState` (`-MOVED` /
-- `-CROSSSLOT`), `scriptVerifyAllowStale`. None of those is a verdict
-- about this stream key, so the branch below matches `WRONGTYPE`
-- explicitly and returns everything else verbatim and untagged, mirroring
-- the fast path's allowlist. Untagged still fails closed: the caller's
-- `Err` arm routes it through `map_snapshot_load_err`, so the commit never
-- reaches its `MULTI` — it just retries or reports as itself instead of
-- ticking the collision counter and telling an operator to delete a
-- healthy stream.

-- Numeric, not string, comparison: `tonumber` on both sides so a header
-- value that is decimal-equal but not byte-equal still counts as
-- unchanged. Both commit paths write `seq` in canonical decimal, so this
-- is belt-and-braces — but the failure mode of getting it wrong is a
-- silently skipped guard, which is precisely what must not happen here.
-- A missing header/field yields `false`, whose `tonumber` is `nil` and
-- therefore never equal to the caller's value: report clean and let the
-- caller's own empty-header handling speak.
local observed_seq = tonumber(redis.call('HGET', KEYS[2], 'seq'))
if observed_seq == nil or observed_seq ~= tonumber(ARGV[2]) then
  return nil
end

-- `pcall` so a WRONGTYPE key becomes a value we can tag rather than a raw
-- script abort. Matches the idiom in `stream_high_water_probe.lua`.
local entries = redis.pcall('XRANGE', KEYS[1], ARGV[1], '+', 'COUNT', 1)

if type(entries) == 'table' and entries.err then
  -- Allowlist, per the note above: only `WRONGTYPE` is a verdict about this
  -- stream key. Anything else is a verdict about the server and goes back
  -- verbatim and untagged, to be classified as itself.
  if string.sub(entries.err, 1, 9) ~= "WRONGTYPE" then
    return redis.error_reply(entries.err)
  end
  return redis.error_reply("STREAM_ID_COLLISION: stream probe failed: " .. tostring(entries.err))
end

if type(entries) ~= 'table' or #entries == 0 then
  return nil
end

-- Each entry is `[id, [field, value, ...]]`. Project to the ID plus a
-- one-word classification of the entry so the (potentially multi-megabyte)
-- `ompf` field never leaves the server.
--
-- The classification exists to separate a real keyspace anomaly from the
-- layer-retirement race: `delete_layer` and the `CreateLayer`-overwrite path
-- both `XADD *` a `deleted` tombstone, whose auto-generated wall-clock ID
-- sits far above any `{new_seq}-0`, so an in-flight commit that resolved the
-- layer mapping just before the retirement would otherwise be told its stream
-- had been corrupted by an out-of-band writer.
--
-- Only the `payload` field is compared, and only against the exact sentinel;
-- an `ompf` field is never touched, so a large entry costs nothing extra
-- beyond the materialization `XRANGE` already paid.
local kind = ""
local pairs_list = entries[1][2]
if type(pairs_list) == 'table' then
  for i = 1, #pairs_list - 1, 2 do
    if pairs_list[i] == 'payload' and pairs_list[i + 1] == 'deleted' then
      kind = "deleted"
      break
    end
  end
end

return { entries[1][1], kind }
