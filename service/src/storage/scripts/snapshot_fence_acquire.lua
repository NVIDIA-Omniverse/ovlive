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

-- snapshot_fence_acquire.lua — grant a fenced snapshot reader a lease on one
-- object's fence, or tell it how long to wait (Spec SDD SR-5.2).
--
-- WRITES NOTHING ON ANY BRANCH THAT DOES NOT GRANT. WATCH-path writers
-- (structural or reset commits on the WATCH path, and every commit under
-- `txn`) WATCH the fence key, so a write by a refused or waiting reader would
-- abort their EXEC; only a real epoch may do that.
--
-- Atomic with every commit: every Lua commit (either mode of
-- update_layer_fast.lua) before this script is fully in the header `seq` it
-- returns (the cut), and every one after it is refused at its step 1d. Every
-- WATCH-path EXEC after a grant fails its WATCH on the fence key.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS (all carry the `{oid}` hash tag, so they share one slot):
--   [1]  fence   — `le:{oid}:obj:snapfence`   (KeySchema::obj_snapshot_fence)
--   [2]  intent  — `le:{oid}:obj:snapintent`  (KeySchema::obj_snapshot_intent)
--   [3]  header  — `le:{oid}:obj:hdr`         (KeySchema::obj_header)
--
-- ARGV:
--   [1]      token — this read's random 128-bit hex token.
--   [2]      est_e — the caller's estimate of the layer's items.
--   [3]      est_b — the caller's estimate of the layer's reply bytes.
--   [4..12]  the cost model's coefficient block, in the order
--            `FenceCoefficients::script_argv` emits it: f0, f1, f2, s1, s0,
--            ceiling_read, rtt99, w_wave, s_min_items (ms-based, see
--            `src/storage/snapshot_fence/cost.rs`).
--
-- The layer size `L` every limit is computed from is the fence key's stored
-- `size_e` / `size_b` when present — the stored size always wins, so every
-- reader of one epoch derives the same deadlines — and the caller's estimate
-- otherwise. All times are this node's `TIME`, in integer milliseconds.
--
-- ── Decision (read-only until the grant) ──────────────────────────────
--
--   1. header absent, or present without a numeric `seq` → NOT_FOUND.
--   2. live readers (`r:*` with a deadline > now):
--      - inside the join window (now − epoch_start ≤ join_window) → grant,
--        joining the epoch;
--      - otherwise → RETRY_AFTER min(latest live deadline − now, poll), epoch.
--   3. no live readers:
--      - the effective gap is the stored `gap_until`; when `epoch_start` is
--        still set the epoch ended by lease expiry (a crashed or stalled
--        holder, not a release), and the gap is also at least
--        ended + max(min_gap, ended − epoch_start), where `ended` is the
--        latest expired `r:*` deadline, so a crash never skips the writers'
--        gap. now < effective gap → RETRY_AFTER (effective gap − now), gap:
--        exact, so every waiting reader wakes at the same instant;
--      - live intents (`w:*` with a deadline > now) while
--        now < effective gap + intent_ttl(L) → RETRY_AFTER
--        min(latest intent deadline − now, poll), intent. With no gap recorded
--        there is no writers' window to extend, and intents are ignored, as
--        they are past that limit;
--      - otherwise → grant, starting an epoch.
--
-- On a grant, and only then: prune every expired `r:*`; write
-- `r:<token>` = now + lease(L); on a started epoch write `epoch_start` = now
-- (replacing a stale one); clear `gap_until` when it has passed; on a key
-- with no stored size write `size_e`, `size_b` and `bpi` from the estimate;
-- and PEXPIRE the key to max(latest live deadline, gap_until) + key_linger(L).
--
-- ── Return value ──────────────────────────────────────────────────────
--
--   {"NOT_FOUND"}
--   {"RETRY_AFTER", <hint ms ≥ 1>, "epoch" | "gap" | "intent", size_e, size_b}
--   {"OK", <seq: the cut, as stored>, "started_epoch" | "joined_epoch",
--    size_e, size_b}
--
-- `size_e` / `size_b` are the `L` this call used, so the caller derives every
-- other limit (its renewal period, its wait cap) from the same size.
--
-- A malformed argument is an error reply raised before any read or write.

local fence_key = KEYS[1]
local intent_key = KEYS[2]
local header_key = KEYS[3]

local token = ARGV[1]
if token == nil or token == "" then
  return redis.error_reply("ERR snapshot_fence_acquire: empty token")
end

local function count_arg(i, name)
  local v = tonumber(ARGV[i])
  if v == nil or v < 0 or v ~= math.floor(v) or v == math.huge then
    return nil, "ERR snapshot_fence_acquire: " .. name .. " must be a non-negative integer, got " .. tostring(ARGV[i])
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
    return redis.error_reply("ERR snapshot_fence_acquire: coefficient " .. name .. " must be a finite number, got " .. tostring(ARGV[3 + i]))
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

-- 1. The header and the cut.
local seq = redis.call("HGET", header_key, "seq")
if not seq or not string.match(seq, "^%d+$") then
  return {"NOT_FOUND"}
end

local tm = redis.call("TIME")
local now = tonumber(tm[1]) * 1000 + math.floor(tonumber(tm[2]) / 1000)

-- The fence key, read only.
local fields = {}
local live_count = 0
local latest_live = nil
local max_expired = nil
local expired = {}
local raw = redis.call("HGETALL", fence_key)
for i = 1, #raw, 2 do
  local name, value = raw[i], raw[i + 1]
  if string.sub(name, 1, 2) == "r:" then
    local deadline = tonumber(value)
    if deadline ~= nil and deadline > now then
      live_count = live_count + 1
      if latest_live == nil or deadline > latest_live then latest_live = deadline end
    else
      expired[#expired + 1] = name
      if deadline ~= nil and (max_expired == nil or deadline > max_expired) then
        max_expired = deadline
      end
    end
  else
    fields[name] = value
  end
end

local stored_e = tonumber(fields["size_e"])
local stored_b = tonumber(fields["size_b"])
local fresh = stored_e == nil or stored_b == nil
local size_e, size_b = est_e, est_b
if not fresh then
  size_e, size_b = stored_e, stored_b
end
local t = timing(size_e, size_b)

local epoch_start = tonumber(fields["epoch_start"])
local gap_until = tonumber(fields["gap_until"])

local function retry(hint, reason)
  return {"RETRY_AFTER", math.max(1, math.ceil(hint)), reason, size_e, size_b}
end

local kind
if live_count > 0 then
  -- 2. A live epoch: join it inside its window, or poll for its end.
  if epoch_start ~= nil and now - epoch_start <= t.join_window then
    kind = "joined_epoch"
  else
    return retry(math.min(latest_live - now, t.poll), "epoch")
  end
else
  -- 3. No live reader. The effective gap, with the crashed-holder rule.
  local effective_gap = gap_until
  if epoch_start ~= nil then
    local ended = max_expired or epoch_start
    local crash_gap = ended + math.max(t.min_gap, ended - epoch_start)
    if effective_gap == nil or crash_gap > effective_gap then
      effective_gap = crash_gap
    end
  end
  if effective_gap ~= nil and now < effective_gap then
    return retry(effective_gap - now, "gap")
  end
  if effective_gap ~= nil and now < effective_gap + t.intent_ttl then
    local latest_intent = nil
    local intents = redis.call("HGETALL", intent_key)
    for i = 1, #intents, 2 do
      if string.sub(intents[i], 1, 2) == "w:" then
        local deadline = tonumber(intents[i + 1])
        if deadline ~= nil and deadline > now and (latest_intent == nil or deadline > latest_intent) then
          latest_intent = deadline
        end
      end
    end
    if latest_intent ~= nil then
      return retry(math.min(latest_intent - now, t.poll), "intent")
    end
  end
  kind = "started_epoch"
end

-- The grant: the only branch that writes.
for _, name in ipairs(expired) do
  redis.call("HDEL", fence_key, name)
end
local deadline = now + math.ceil(t.lease)
redis.call("HSET", fence_key, "r:" .. token, ms(deadline))
if kind == "started_epoch" then
  redis.call("HSET", fence_key, "epoch_start", ms(now))
end
if gap_until ~= nil and gap_until <= now then
  redis.call("HDEL", fence_key, "gap_until")
  gap_until = nil
end
if fresh then
  redis.call("HSET", fence_key, "size_e", ms(size_e), "size_b", ms(size_b))
  if size_e > 0 then
    redis.call("HSET", fence_key, "bpi", string.format("%.17g", size_b / size_e))
  end
end
local horizon = deadline
if latest_live ~= nil and latest_live > horizon then horizon = latest_live end
if gap_until ~= nil and gap_until > horizon then horizon = gap_until end
redis.call("PEXPIRE", fence_key, ms(horizon + math.ceil(t.key_linger) - now))

return {"OK", seq, kind, size_e, size_b}
