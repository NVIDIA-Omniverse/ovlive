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

-- presence_refresh.lua — renew one live connection's presence lease, but
-- only while the connection is still fully registered.
--
-- Replaces an unconditional `SET lease "1" EX ttl`. That form re-created the
-- lease of a connection the sweeper had already removed, leaving a lease with
-- no member behind it and hiding the eviction from the connection's keeper,
-- and it could re-create the lease of a connection already torn down. This
-- script writes only when the connection is still a member with an intact
-- info hash, and otherwise reports why it did not, so the keeper can
-- re-register (Evicted, InfoMissing) or stop (Gone).
--
-- It never creates the info hash: a member whose `user_name` is missing is
-- reported, not patched, because the repair belongs to
-- `presence_register.lua`, which writes the whole hash.
--
-- Server clock: `lease_expires_at_ms` is `TIME` + ttl on this node, as in
-- `presence_register.lua`.
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
--   [1]  cid      — the connection id.
--   [2]  ttl_secs — lease TTL in whole seconds; a positive integer.
--
-- No other key is read or written.
--
-- ── Return value ──────────────────────────────────────────────────────
--
-- Integer, checked in this order; every outcome but Renewed writes nothing:
--   `2` — Gone: the teardown marker exists.
--   `0` — Evicted: `cid` is not in the member set.
--   `3` — InfoMissing: `cid` is a member but its info hash has no
--         `user_name` (the hash is gone or incomplete).
--   `1` — Renewed: the lease was set to `ttl_secs` and
--         `lease_expires_at_ms` rewritten.
--
-- A non-positive or non-integer `ttl_secs` is an error reply, raised before
-- any read or write.

local members_key = KEYS[1]
local info_key = KEYS[2]
local lease_key = KEYS[3]
local gone_key = KEYS[4]

local cid = ARGV[1]
local ttl_arg = ARGV[2]

local ttl = tonumber(ttl_arg)
if ttl == nil or ttl <= 0 or ttl ~= math.floor(ttl) then
  return redis.error_reply("ERR presence_refresh: ttl_secs must be a positive integer, got " .. tostring(ttl_arg))
end
local ttl_secs = string.format("%d", ttl)

if redis.call("EXISTS", gone_key) == 1 then
  return 2
end
if redis.call("SISMEMBER", members_key, cid) == 0 then
  return 0
end
if redis.call("HEXISTS", info_key, "user_name") == 0 then
  return 3
end

-- Integer milliseconds; see `presence_register.lua` for the formatting.
local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local expires_at_ms = string.format("%d", now_ms + ttl * 1000)

redis.call("SET", lease_key, "1", "EX", ttl_secs)
redis.call("HSET", info_key, "lease_expires_at_ms", expires_at_ms)

return 1
