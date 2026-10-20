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

-- presence_unregister.lua — tear one connection out of its session's
-- presence state, fence it against late writes, and announce the removal iff
-- this call made it.
--
-- One script so that the removal, the fence and the `left` announcement land
-- together. The teardown marker is armed first, in the same execution, so any
-- `presence_register.lua` or `presence_refresh.lua` for this connection that
-- the store executes afterwards — a register or refresh that timed out on the
-- client but still ran later on another pooled connection, or that fred
-- re-sent after a reconnect — writes and publishes nothing.
--
-- The marker is the first part of a two-part fence and lives one lease TTL.
-- After it expires, a late register is refused by its own store-clock
-- deadline (`not_after_ms` in `presence_register.lua`). That deadline falls
-- before the marker's expiry only because the register's caller is obliged to
-- read its store time while the connection's teardown has not yet started, so
-- that read precedes this script (see
-- `storage::presence::register_connection`). With no register able to
-- recreate the member, a late refresh finds it gone (Evicted) and writes
-- nothing either.
--
-- Publication (path A): `left` is published iff the `SREM` removed the
-- member, in this execution. A connection the sweeper already removed was
-- announced by the sweep that removed it, so a teardown after it publishes
-- nothing, and peers see exactly one `left` per removal.
--
-- Idempotent: a repeated call re-arms the marker (restarting its TTL),
-- removes nothing and publishes nothing.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS (all carry the `{sid}` hash tag, so they share one slot):
--   [1]  members — `le:{sid}:presence`                (KeySchema::presence_members)
--   [2]  info    — `le:{sid}:presence:{cid}`          (KeySchema::presence_user)
--   [3]  lease   — `le:{sid}:presence:{cid}:lease`    (KeySchema::presence_lease)
--   [4]  gone    — `le:{sid}:presence:{cid}:gone`     (KeySchema::presence_gone)
--
-- ARGV:
--   [1]  cid          — the connection id.
--   [2]  ttl_secs     — marker TTL in whole seconds (the configured lease TTL);
--                       a positive integer.
--   [3]  channel      — `le:presence:events:<sid>`
--                       (KeySchema::presence_events_channel).
--   [4]  left_payload — the encoded `left` event, built by the caller.
--
-- No other key is read or written. The session index
-- (`le:presence:active_sessions`) is in another slot; the caller runs the
-- index-removal protocol itself when `remaining` is 0.
--
-- ── Return value ──────────────────────────────────────────────────────
--
-- Array `{removed, remaining}`:
--   removed   — `1` if the `SREM` removed `cid` (and `left` was published),
--               `0` if it was not a member.
--   remaining — `SCARD` of the member set after the removal.
--
-- A non-positive or non-integer `ttl_secs` is an error reply, raised before
-- any write.

local members_key = KEYS[1]
local info_key = KEYS[2]
local lease_key = KEYS[3]
local gone_key = KEYS[4]

local cid = ARGV[1]
local ttl_arg = ARGV[2]
local channel = ARGV[3]
local left_payload = ARGV[4]

local ttl = tonumber(ttl_arg)
if ttl == nil or ttl <= 0 or ttl ~= math.floor(ttl) then
  return redis.error_reply("ERR presence_unregister: ttl_secs must be a positive integer, got " .. tostring(ttl_arg))
end
local ttl_secs = string.format("%d", ttl)

redis.call("SET", gone_key, "1", "EX", ttl_secs)
local removed = redis.call("SREM", members_key, cid)
redis.call("DEL", info_key)
redis.call("DEL", lease_key)
if removed == 1 then
  redis.call("PUBLISH", channel, left_payload)
end

return { removed, redis.call("SCARD", members_key) }
