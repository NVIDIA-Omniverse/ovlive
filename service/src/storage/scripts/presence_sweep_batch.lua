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

-- presence_sweep_batch.lua — remove the stale connections of one chunk of a
-- session's members, and announce each removal, in one atomic step.
--
-- A connection is stale when its lease has expired (the key is absent) while
-- it is still in the member set. For each chunk cid this script re-checks both
-- conditions itself and, only when both hold, removes the member, deletes its
-- info hash and publishes its `left` event. So:
--   - a connection whose keeper renewed or re-registered it after the caller
--     read the member page has a lease again and is left alone;
--   - a connection that is no longer a member — removed by another replica's
--     sweep of the same chunk, by its own teardown, or by an earlier chunk
--     after the member page repeated it — is neither removed again nor
--     announced again.
-- Every removal is therefore announced exactly once, by the execution that
-- made it.
--
-- Publication (path A): `left` is published inside this execution for each
-- cid it removed, and for no other. A lost reply cannot separate a removal
-- from its announcement, and every membership event of a session is emitted
-- by the node owning `{sid}` in the order the membership changed.
-- `PUBLISH` needs no key and no slot; the channel travels in ARGV.
--
-- Removal delay: for each removed cid the script reports how long after the
-- lease's recorded expiry the removal happened, `now_ms − lease_expires_at_ms`
-- from this node's `TIME` and the `lease_expires_at_ms` field that
-- `presence_register.lua` and `presence_refresh.lua` write from the same
-- clock. The field is read only here. An info hash written before the field
-- existed, or a test seed, lacks it; that removal reports -1, as does a field
-- whose value is negative or non-numeric (only corrupted data, since both
-- writers render it with `%d`). A delay that would come out negative (the
-- lease key was deleted before its recorded expiry, for example by a test) is
-- reported as 0, so -1 always means "no usable field".
--
-- The lease is only checked, never written: a stale connection's lease is
-- already gone. The teardown marker is neither read nor written: a member
-- with a marker has already been removed by its teardown, so the membership
-- check alone skips it.
--
-- Server clock: one `TIME` sample per execution, the same
-- `redis.call("TIME")` discipline as `presence_register.lua` (safe under
-- effects replication, which Valkey 7+ scripts use by default).
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS (all carry the `{sid}` hash tag, so they share one slot), for a chunk
-- of n ≥ 1 connections:
--   [1]       members — `le:{sid}:presence`              (KeySchema::presence_members)
--   [2i]      lease_i — `le:{sid}:presence:{cid_i}:lease` (KeySchema::presence_lease)
--   [2i + 1]  info_i  — `le:{sid}:presence:{cid_i}`       (KeySchema::presence_user)
--   for i = 1 … n.
--
-- ARGV:
--   [1]       channel        — `le:presence:events:<sid>`
--                              (KeySchema::presence_events_channel).
--   [2i]      cid_i          — the i-th connection id of the chunk.
--   [2i + 1]  left_payload_i — the encoded `left` event for cid_i, built by
--                              the caller.
--   for i = 1 … n.
--
-- The caller bounds n (`storage::presence::SWEEP_MEMBER_BATCH`); the script
-- checks only that KEYS and ARGV describe the same n ≥ 1 cids.
--
-- No other key is read or written. The session index
-- (`le:presence:active_sessions`) is in another slot; the caller runs the
-- index-removal protocol itself when `remaining` is 0. No lease and no
-- teardown marker (`le:{sid}:presence:{cid}:gone`) is written.
--
-- ── Return value ──────────────────────────────────────────────────────
--
-- Array `{removed, remaining}`:
--   removed   — a flat array `{cid_a, delay_a, cid_b, delay_b, …}` with one
--               pair per cid this execution removed (and announced), in chunk
--               order. `delay` is the removal delay in whole milliseconds
--               (≥ 0), or -1 when the info hash's `lease_expires_at_ms` was
--               absent, negative or non-numeric. Empty when nothing was
--               removed.
--   remaining — `SCARD` of the member set after the removals.
--
-- KEYS and ARGV that do not describe the same n ≥ 1 cids are an error reply,
-- raised before any read or write.

local nkeys = #KEYS
local nargs = #ARGV
local n = (nkeys - 1) / 2
if nkeys < 3 or n ~= math.floor(n) or nargs ~= 2 * n + 1 then
  return redis.error_reply(
    "ERR presence_sweep_batch: expected KEYS = 1 + 2n and ARGV = 1 + 2n for n >= 1, got "
      .. nkeys .. " KEYS and " .. nargs .. " ARGV"
  )
end

local members_key = KEYS[1]
local channel = ARGV[1]

-- Collapse TIME's {seconds, microseconds} to integer milliseconds.
local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local removed = {}
for i = 1, n do
  local lease_key = KEYS[2 * i]
  local info_key = KEYS[2 * i + 1]
  local cid = ARGV[2 * i]
  local left_payload = ARGV[2 * i + 1]

  if redis.call("EXISTS", lease_key) == 0 and redis.call("SISMEMBER", members_key, cid) == 1 then
    -- `HGET` of an absent field is `false`, which `tonumber` maps to nil.
    local expires_at_ms = tonumber(redis.call("HGET", info_key, "lease_expires_at_ms"))
    redis.call("SREM", members_key, cid)
    redis.call("DEL", info_key)
    redis.call("PUBLISH", channel, left_payload)

    -- Lua 5.1's `tonumber` is `strtod`, so it also accepts "-inf", "nan" and
    -- "-1e300". Only a value >= 0 is used (NaN fails the comparison), which
    -- caps the delay at `now_ms` and keeps the reply integer in range; any
    -- other value reads as absent.
    local delay = -1
    if expires_at_ms ~= nil and expires_at_ms >= 0 then
      delay = math.max(0, now_ms - expires_at_ms)
    end
    removed[#removed + 1] = cid
    -- An integer reply; `string.format("%d")` would make it a bulk string.
    removed[#removed + 1] = delay
  end
end

return { removed, redis.call("SCARD", members_key) }
