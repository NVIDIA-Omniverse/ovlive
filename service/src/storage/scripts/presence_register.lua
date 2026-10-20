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

-- presence_register.lua — register one JoinSession connection in its
-- session's presence state, and announce it iff this call made it a member.
--
-- One script so that the lease, the info hash, the membership and the
-- `joined` announcement change together. Written as separate round-trips,
-- a connection could be a member without a lease (and be swept), or be
-- announced without being a member, or be re-created by a late write after
-- its teardown. The same script serves the initial registration, a
-- self-rejoin after eviction and an info-hash repair.
--
-- Late-write fence, in two parts, checked in every mode because an initial
-- register whose reply was lost can itself be the late write:
--   1. the teardown marker (`presence_unregister.lua` arms it for one lease
--      TTL) refuses the register while it exists (Gone);
--   2. the caller's store-clock deadline `not_after_ms` refuses it once this
--      node's `TIME` has passed it (Expired). The caller sets the deadline to
--      a store-time read taken before it registers, plus one lease TTL, and
--      must take that read while the connection's teardown has not yet
--      started (the rustdoc of `storage::presence::register_connection` states
--      this obligation). Then the read precedes teardown, the deadline falls
--      before the marker expires, and a register reaching the store after
--      teardown meets one part or the other however late it arrives. A read
--      taken after teardown began would put the deadline after the marker's
--      expiry and reopen the gap. Without the second part, a register
--      arriving after the marker expired could recreate the member and info
--      hash of a session's last connection after teardown had taken the
--      session out of the index, where no sweep would ever visit it.
-- Both parts compare against the clock of the node owning `{sid}`: the
-- caller pins its store-time read to this slot.
--
-- Publication (path A): `joined` is published iff the `SADD` added the
-- member, in this execution, so every membership addition is announced
-- exactly once and nothing else is announced — a repair of a missing info
-- hash, or a late duplicate register for a connection that is still a
-- member, publishes nothing. `PUBLISH` needs no key and no slot; the channel
-- travels in ARGV. Because every membership script for a session runs on the
-- node that owns `{sid}`, a subscriber sees one session's `joined` / `left`
-- events in the order the membership changed.
--
-- Server clock: `lease_expires_at_ms` is `TIME` + ttl on this node, never a
-- caller-supplied clock, so every replica's registrations share one clock
-- (the same `redis.call("TIME")` discipline as `gc_claim_pending.lua`; safe
-- under effects replication, which Valkey 7+ scripts use by default).
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
--   [1]  cid              — the connection id; the member written to KEYS[1].
--   [2]  user_name        — written to the info hash.
--   [3]  user_application — written to the info hash.
--   [4]  ttl_secs         — lease TTL in whole seconds; a positive integer.
--   [5]  channel          — `le:presence:events:<sid>`
--                           (KeySchema::presence_events_channel).
--   [6]  joined_payload   — the encoded `joined` event, built by the caller.
--   [7]  not_after_ms     — register deadline in milliseconds on this node's
--                           `TIME`; a non-negative integer. The register is
--                           refused once `TIME` is past it.
--
-- No other key is read or written. The session index
-- (`le:presence:active_sessions`) is untagged, lives in another slot, and is
-- maintained by the caller outside this script.
--
-- ── Return value ──────────────────────────────────────────────────────
--
-- Array `{status, added}`, checked in this order:
--   `{0, 0}` — Gone: the teardown marker exists. Nothing was written and
--              nothing was published.
--   `{3, 0}` — Expired: this node's `TIME` is past `not_after_ms`. Nothing
--              was written and nothing was published. (Status 2 is left
--              unused; it is reserved for a `Pending` refusal this design
--              does not build.)
--   `{1, 1}` — Registered, and the `SADD` added the member; `joined` was
--              published.
--   `{1, 0}` — Registered; the connection was already a member, so the
--              lease and info hash were rewritten and nothing was published.
--
-- A non-positive or non-integer `ttl_secs`, or a negative or non-integer
-- `not_after_ms`, is an error reply, raised before any read or write.

local members_key = KEYS[1]
local info_key = KEYS[2]
local lease_key = KEYS[3]
local gone_key = KEYS[4]

local cid = ARGV[1]
local user_name = ARGV[2]
local user_application = ARGV[3]
local ttl_arg = ARGV[4]
local channel = ARGV[5]
local joined_payload = ARGV[6]
local not_after_arg = ARGV[7]

local ttl = tonumber(ttl_arg)
if ttl == nil or ttl <= 0 or ttl ~= math.floor(ttl) then
  return redis.error_reply("ERR presence_register: ttl_secs must be a positive integer, got " .. tostring(ttl_arg))
end
-- Re-rendered so `SET … EX` gets a plain integer even for an ARGV like "60.0".
local ttl_secs = string.format("%d", ttl)

-- Lua numbers are doubles, so a deadline near u64::MAX loses precision here;
-- that only matters for deadlines far beyond any real `TIME`, which stay
-- "not yet passed" either way.
local not_after_ms = tonumber(not_after_arg)
if not_after_ms == nil or not_after_ms < 0 or not_after_ms ~= math.floor(not_after_ms) then
  return redis.error_reply("ERR presence_register: not_after_ms must be a non-negative integer, got " .. tostring(not_after_arg))
end

if redis.call("EXISTS", gone_key) == 1 then
  return { 0, 0 }
end

-- Collapse TIME's {seconds, microseconds} to integer milliseconds. The same
-- sample serves the deadline gate and `lease_expires_at_ms`.
-- `string.format("%d")` pins an integer rendering; Lua 5.1's `tostring` uses
-- `%.14g`, which is integral only while the value has at most 14 digits.
local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
if now_ms > not_after_ms then
  return { 3, 0 }
end
local expires_at_ms = string.format("%d", now_ms + ttl * 1000)

redis.call("SET", lease_key, "1", "EX", ttl_secs)
redis.call(
  "HSET", info_key,
  "user_name", user_name,
  "user_application", user_application,
  "lease_expires_at_ms", expires_at_ms
)
local added = redis.call("SADD", members_key, cid)
if added == 1 then
  redis.call("PUBLISH", channel, joined_payload)
end

return { 1, added }
