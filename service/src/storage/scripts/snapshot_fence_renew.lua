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

-- snapshot_fence_renew.lua — extend one fenced reader's lease, only while it
-- is still held (Spec SDD SR-5.3).
--
-- The lease is extended only if `r:<token>` still exists AND its stored
-- deadline has not passed on this node's `TIME`. A lease that has already
-- expired is never revived: writers judge expiry against the same clock and
-- may have resumed, so reviving it would let the read go on to "prove" a cut
-- writers have already moved past. Such a call answers LOST and writes
-- nothing; the reader stops and its release answers BROKEN.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  fence — `le:{oid}:obj:snapfence` (KeySchema::obj_snapshot_fence)
--
-- ARGV:
--   [1]      token — the reader's token, as granted.
--   [2]      est_e — the caller's size estimate (items), used only when the
--            key holds no stored size.
--   [3]      est_b — the same, reply bytes.
--   [4..12]  the cost model's coefficient block (see
--            `snapshot_fence_acquire.lua`).
--
-- On renewal: `r:<token>` = max(its deadline, now + lease(L)), with `L` the
-- stored size when present; and the key's expiry is raised, never lowered, to
-- that deadline + key_linger(L). Every earlier writing script set the expiry to
-- max(latest live deadline, gap_until) + key_linger, so keeping the larger of
-- the two keeps that invariant without reading every field.
--
-- ── Return value ──────────────────────────────────────────────────────
--
--   {"RENEWED", <new deadline ms>}
--   {"LOST"}        — the field is gone or its deadline has passed; nothing
--                     was written.
--
-- A malformed argument is an error reply raised before any read or write.

local fence_key = KEYS[1]

local token = ARGV[1]
if token == nil or token == "" then
  return redis.error_reply("ERR snapshot_fence_renew: empty token")
end

local function count_arg(i, name)
  local v = tonumber(ARGV[i])
  if v == nil or v < 0 or v ~= math.floor(v) or v == math.huge then
    return nil, "ERR snapshot_fence_renew: " .. name .. " must be a non-negative integer, got " .. tostring(ARGV[i])
  end
  return v
end

local est_e, err_e = count_arg(2, "est_e")
if est_e == nil then return redis.error_reply(err_e) end
local est_b, err_b = count_arg(3, "est_b")
if est_b == nil then return redis.error_reply(err_b) end

-- The cost model. Same expressions, same order, as
-- `FenceCoefficients::timing` (src/storage/snapshot_fence/cost.rs).
local COEFFICIENTS = {"f0", "f1", "f2", "s1", "s0", "ceiling_read", "rtt99", "w_wave", "s_min_items"}
local c = {}
for i, name in ipairs(COEFFICIENTS) do
  local v = tonumber(ARGV[3 + i])
  if v == nil or v ~= v or v == math.huge or v == -math.huge then
    return redis.error_reply("ERR snapshot_fence_renew: coefficient " .. name .. " must be a finite number, got " .. tostring(ARGV[3 + i]))
  end
  c[name] = v
end

local function timing(e, b)
  local t = {}
  t.hold = c.f0 + c.f1 * e + c.f2 * b
  local rtt_q = c.rtt99 + (c.w_wave + c.ceiling_read)
  t.lease = math.max(4 * rtt_q, t.hold / 4)
  t.join_window = 2 * rtt_q
  t.poll = t.join_window
  t.min_gap = c.ceiling_read
  -- S's structural term no lower than s_min_items (below it the fitted
  -- chord is an extrapolation, and the term can be negative there). S
  -- itself can then fall below zero for a layer whose hold is short;
  -- it is used only here, through intent_ttl's max with the floor it shares
  -- with the lease.
  t.intent_ttl = math.max(4 * rtt_q, 1.5 * (t.hold + c.s1 * math.max(e, c.s_min_items) + c.s0))
  t.key_linger = 20 * (t.hold + math.max(t.hold, t.min_gap) + t.intent_ttl)
  return t
end

local function ms(x)
  return string.format("%d", x)
end

local field = "r:" .. token
local tm = redis.call("TIME")
local now = tonumber(tm[1]) * 1000 + math.floor(tonumber(tm[2]) / 1000)

local current = tonumber(redis.call("HGET", fence_key, field))
if current == nil or current <= now then
  return {"LOST"}
end

local stored = redis.call("HMGET", fence_key, "size_e", "size_b")
local size_e, size_b = tonumber(stored[1]), tonumber(stored[2])
if size_e == nil or size_b == nil then
  size_e, size_b = est_e, est_b
end
local t = timing(size_e, size_b)

local deadline = math.max(current, now + math.ceil(t.lease))
redis.call("HSET", fence_key, field, ms(deadline))
local expire_ms = deadline + math.ceil(t.key_linger) - now
if redis.call("PTTL", fence_key) < expire_ms then
  redis.call("PEXPIRE", fence_key, ms(expire_ms))
end

return {"RENEWED", deadline}
