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

-- stream_high_water_probe.lua — server-side scalar extraction for the
-- checkpoint recovery hook's sequence high-water probe (b.8y5).
--
-- ── Why this is a script and not a client-side XINFO STREAM ────────────
--
-- The probe needs exactly one scalar out of `XINFO STREAM`:
-- `last-generated-id`. No `XINFO STREAM` form returns metadata alone —
-- the non-FULL form embeds `first-entry` AND `last-entry` in full, and the
-- FULL form embeds `COUNT` entries. Entries on a per-object layer stream
-- carry the writer's raw inbound OMPF (b.gze), bounded only by
-- `LIVE_EDIT_MAX_LAYER_SIZE` (100 MB by default), so a client-side probe
-- could pull ~200 MB across the wire to read a ~15-byte ID.
--
-- That cost lands in the worst possible place: `apply_checkpoint` runs
-- inline on an RPC-triggered recovery, under the hook's per-`object_id`
-- K=1 admission mutex, and a transfer that size would blow
-- `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` and fold the probe to `nil` —
-- silently disabling wedge-prevention on exactly the large layers where
-- the post-restore XADD wedge is most likely and most expensive.
--
-- ── Why FULL COUNT 1, and its honest cost ─────────────────────────────
--
-- `FULL COUNT 1` is the cheapest form that still carries
-- `last-generated-id`: it materializes one entry rather than the non-FULL
-- form's two (`first-entry` + `last-entry`). Measured on Valkey 7.2.4
-- with two 32 MB entries, the whole EVAL round-trip went 96.8 ms →
-- 47.5 ms — the expected ~2×.
--
-- The reply this script returns is O(1) in layer size, but the *server-side*
-- work is not: Valkey still materializes one entry payload inside its
-- single-threaded Lua VM before this script filters it away, so a 100 MB
-- entry costs a ~150 ms stall of the whole server. That is accepted here
-- and nowhere else: this runs once per layer on a DR recovery, which is
-- rare, already expensive, and strictly better than the alternative of
-- moving those bytes over the network too. Do NOT reuse this script on a
-- hot path.
--
-- (`XINFO STREAM` has no metadata-only form to prefer instead; `XLEN` and
-- friends do not expose `last-generated-id`, and the `XREVRANGE` tail ID
-- is not the same value — it drops back after an `XDEL` of the top entry
-- where `last-generated-id` holds, so it would under-report the floor in
-- exactly the case the probe exists to catch.)
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  per-object stream key — `le:{object_id}:obj:stream`
--        (see KeySchema::layer_stream). Single-key, and the `{object_id}`
--        hash tag places it on one slot, so this is Cluster-safe.
--
-- ARGV: none. The plausibility gate that consumes this runs on the Rust
--       side; the script makes no policy decision at all.
--
-- ── Return value ──────────────────────────────────────────────────────
--
-- `nil` when the probe cannot produce an answer — missing key, wrong
-- type, or a reply shape without `last-generated-id`. This script cannot
-- tell those apart and does not try; since b.4et the caller un-conflates
-- them Rust-side with one `TYPE` round-trip on the same key, and the three
-- answers take three different actions: `none` folds the restore floor
-- back to `snap_seq` and proceeds (the full-keyspace-loss shape), `stream`
-- means the stream is present and its top unreadable so the restore is
-- REFUSED, and any other type is reported as a foreign key — proceed at
-- the `snap_seq` floor, and never delete a value this service did not
-- write. No `nil` arm takes any action on the stream.
--
-- Otherwise `last-generated-id` verbatim (a `<ms>-<seq>` string). The
-- caller parses the leading component as the sequence number and applies
-- its own plausibility bound. Note this value is NOT necessarily a
-- sequence number: the two tombstone sites in `LayerStore`
-- (`delete_layer` and the CreateLayer-overwrite cascade) append their
-- `deleted` sentinel with the auto-ID `*`, which mints a wall-clock
-- millisecond ID. Distinguishing those is the caller's plausibility
-- bound, not this script's job — an auto-ID timestamp is ~1.787e12 and
-- rising, three orders of magnitude above the bound.

-- `pcall` rather than `call`: a missing key answers `ERR no such key`,
-- which is the ordinary case here (the stream did not survive the state
-- loss) and must not abort the script. A WRONGTYPE key lands here too.
local info = redis.pcall('XINFO', 'STREAM', KEYS[1], 'FULL', 'COUNT', 1)
if type(info) ~= 'table' or info.err then
  return nil
end

-- Scripts run in RESP2 conversion mode unless a script calls
-- `redis.setresp(3)`, so the map arrives as a flat [name, value, ...]
-- array regardless of the connection's negotiated protocol.
for i = 1, #info - 1, 2 do
  if info[i] == 'last-generated-id' then
    return info[i + 1]
  end
end

return nil
