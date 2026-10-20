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

-- snapshot_fence_release.lua — prove a fenced read and release its lease, in
-- one atomic step (Spec SDD SR-5.3).
--
-- The check and the release share one script, so nothing can commit between
-- "the lease was still held and the header still at the cut" and the lease
-- going away.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS (both carry the `{oid}` hash tag):
--   [1]  fence  — `le:{oid}:obj:snapfence`  (KeySchema::obj_snapshot_fence)
--   [2]  header — `le:{oid}:obj:hdr`        (KeySchema::obj_header)
--
-- ARGV:
--   [1]      token      — the reader's token, as granted.
--   [2]      cut        — the header `seq` the grant returned, verbatim.
--   [3]      measured_e — the layer's size as read (items); a read that did
--                         not complete passes the size it was granted under.
--   [4]      measured_b — the same, reply bytes.
--   [5..13]  the cost model's coefficient block (see
--            `snapshot_fence_acquire.lua`).
--
-- ── Steps ─────────────────────────────────────────────────────────────
--
--   1. Header absent, or present without a numeric `seq` → NOT_FOUND, writing
--      nothing.
--   2. `r:<token>` absent → BROKEN, writing nothing. Steps 1 and 2 are what
--      keep a release from recreating a fence key on an object that cleanup
--      or a restore wipe has UNLINKed.
--   3. VERIFIED when `r:<token>` is still live (deadline > now) and the header
--      `seq` equals `cut` byte for byte; BROKEN otherwise.
--   4. Delete `r:<token>`.
--   5. If no live reader remains and `epoch_start` is still set, the epoch
--      ends here: `gap_until` = now + max(min_gap, now − epoch_start), the
--      measured size is recorded (`size_e`, `size_b`, `bpi`) and `epoch_start`
--      is cleared. If another reader is still live, the epoch goes on for it.
--      If `epoch_start` is already gone, an earlier release closed this
--      epoch — this reader's lease had expired, so that release did not count
--      it — and nothing more is written: recomputing the gap now, with no
--      epoch start to measure from, would shrink the gap that release set
--      below the epoch's length (SR-5.5), and this reader's size would
--      overwrite the size the epoch's last live reader measured.
--   6. PEXPIRE the key to max(latest live deadline, gap_until) +
--      key_linger(L), with `L` the size now stored; left as it is when there
--      is neither (no live reader and no recorded gap).
--
-- Steps 4–6 run on BROKEN too: a broken read still gives its lease up, and
-- the gap after the epoch is owed to writers either way.
--
-- ── Return value ──────────────────────────────────────────────────────
--
--   {"VERIFIED"} | {"BROKEN"} | {"NOT_FOUND"}
--
-- A malformed argument is an error reply raised before any read or write.

local fence_key = KEYS[1]
local header_key = KEYS[2]

local token = ARGV[1]
if token == nil or token == "" then
  return redis.error_reply("ERR snapshot_fence_release: empty token")
end
local cut = ARGV[2]
if cut == nil or not string.match(cut, "^%d+$") then
  return redis.error_reply("ERR snapshot_fence_release: cut must be a decimal sequence, got " .. tostring(cut))
end

local function count_arg(i, name)
  local v = tonumber(ARGV[i])
  if v == nil or v < 0 or v ~= math.floor(v) or v == math.huge then
    return nil, "ERR snapshot_fence_release: " .. name .. " must be a non-negative integer, got " .. tostring(ARGV[i])
  end
  return v
end

local measured_e, err_e = count_arg(3, "measured_e")
if measured_e == nil then return redis.error_reply(err_e) end
local measured_b, err_b = count_arg(4, "measured_b")
if measured_b == nil then return redis.error_reply(err_b) end

-- The cost model. Same expressions, same order, as
-- `FenceCoefficients::timing` (src/storage/snapshot_fence/cost.rs).
local COEFFICIENTS = {"f0", "f1", "f2", "s1", "s0", "ceiling_read", "rtt99", "w_wave", "s_min_items"}
local c = {}
for i, name in ipairs(COEFFICIENTS) do
  local v = tonumber(ARGV[4 + i])
  if v == nil or v ~= v or v == math.huge or v == -math.huge then
    return redis.error_reply("ERR snapshot_fence_release: coefficient " .. name .. " must be a finite number, got " .. tostring(ARGV[4 + i]))
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

-- 1. The header.
local seq = redis.call("HGET", header_key, "seq")
if not seq or not string.match(seq, "^%d+$") then
  return {"NOT_FOUND"}
end

-- 2. This reader's field.
local own_field = "r:" .. token
local raw = redis.call("HGETALL", fence_key)
local own = nil
local fields = {}
local tm = redis.call("TIME")
local now = tonumber(tm[1]) * 1000 + math.floor(tonumber(tm[2]) / 1000)
local latest_other = nil
for i = 1, #raw, 2 do
  local name, value = raw[i], raw[i + 1]
  if name == own_field then
    own = value
  elseif string.sub(name, 1, 2) == "r:" then
    local deadline = tonumber(value)
    if deadline ~= nil and deadline > now and (latest_other == nil or deadline > latest_other) then
      latest_other = deadline
    end
  else
    fields[name] = value
  end
end
if own == nil then
  return {"BROKEN"}
end

-- 3. The proof.
local own_deadline = tonumber(own)
local verified = own_deadline ~= nil and own_deadline > now and seq == cut

-- 4. Give the lease up.
redis.call("HDEL", fence_key, own_field)

-- 5. The epoch ends with its last live reader — once. A release that finds
-- `epoch_start` already cleared belongs to an epoch an earlier release closed
-- (its own lease had expired, so that release did not count it), and writes
-- nothing beyond step 4.
local gap_until = tonumber(fields["gap_until"])
local epoch_start = tonumber(fields["epoch_start"])
local size_e, size_b
if latest_other == nil and epoch_start ~= nil then
  size_e, size_b = measured_e, measured_b
  local t = timing(size_e, size_b)
  gap_until = now + math.ceil(math.max(t.min_gap, now - epoch_start))
  redis.call("HSET", fence_key, "gap_until", ms(gap_until), "size_e", ms(size_e), "size_b", ms(size_b))
  if size_e > 0 then
    redis.call("HSET", fence_key, "bpi", string.format("%.17g", size_b / size_e))
  end
  redis.call("HDEL", fence_key, "epoch_start")
else
  size_e, size_b = tonumber(fields["size_e"]), tonumber(fields["size_b"])
  if size_e == nil or size_b == nil then
    size_e, size_b = measured_e, measured_b
  end
end

-- 6. The key's expiry, from the latest live deadline and the recorded gap.
-- With neither, nothing here bounds the key's life any better than the expiry
-- the writing script before this one set, so it is left as it is.
local horizon = latest_other
if gap_until ~= nil and (horizon == nil or gap_until > horizon) then horizon = gap_until end
if horizon ~= nil then
  local t = timing(size_e, size_b)
  local expire_ms = horizon + math.ceil(t.key_linger) - now
  redis.call("PEXPIRE", fence_key, ms(math.max(1, expire_ms)))
end

if verified then
  return {"VERIFIED"}
end
return {"BROKEN"}
