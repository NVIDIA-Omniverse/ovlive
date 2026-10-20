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

-- snapshot_intent_register.lua — register a WATCH-path writer's intent on an
-- above-ceiling layer, unless a reader holds the fence (Spec SDD SR-5.4).
--
-- An intent delays the next fence epoch while the attempt runs, so a
-- WATCH-path attempt that reads a snapshot and starts in the writers' gap can
-- finish (the cached structural route registers none: its commit script is
-- refused at step 1d instead). It lives in its own key: other WATCH-path
-- writers WATCH the fence key, and writing there would abort their EXEC.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS (all carry the `{oid}` hash tag):
--   [1]  intent — `le:{oid}:obj:snapintent`  (KeySchema::obj_snapshot_intent)
--   [2]  fence  — `le:{oid}:obj:snapfence`   (KeySchema::obj_snapshot_fence)
--   [3]  header — `le:{oid}:obj:hdr`         (KeySchema::obj_header)
--
-- ARGV:
--   [1]      token — this attempt's random 128-bit hex token.
--   [2]      est_e — the caller's size estimate (items), used only when the
--            fence key stores no size.
--   [3]      est_b — the same, reply bytes.
--   [4..12]  the cost model's coefficient block (see
--            `snapshot_fence_acquire.lua`).
--
-- ── Steps ─────────────────────────────────────────────────────────────
--
--   0. The header is absent (a deleted, retired or wiped object) →
--      NOT_FOUND, writing nothing: an intent key is never recreated on a
--      removed object, and nothing relies on its expiry to clean one up
--      (Spec SDD SR-5.4). Absent means no key at all — a header without a
--      numeric `seq` is a present, corrupt object, which the commit path
--      refuses on its own terms, so it is not this script's to answer.
--   0a. The fence or the intent key holds a type other than a hash →
--      WRONGTYPE_KEY, fence first, writing nothing — the same reply, in the
--      same order, as `update_layer_fast.lua` step 1d and
--      `snapshot_fence_probe.lua` (SDD §7.5 backend parity).
--   1. A live reader (`r:*` on the fence with a deadline > now) → FENCED,
--      writing nothing.
--   2. Otherwise: write `w:<token>` = now + intent_ttl(L) — `L` the fence
--      key's stored size when present, the estimate otherwise;
--      intent_ttl(L) = max(4·(rtt99 + Q), 1.5·S(e, b)), the layer's
--      structural-attempt cost floored like the lease (S can be below zero
--      for a layer whose hold is short; the floor then decides) — then prune
--      every other expired `w:*`, and PEXPIRE the intent key to its latest
--      deadline. The own field is written before the prune so that no command
--      names an emptied key during a cluster slot migration (see the
--      ordering invariant at that code, b.cyu).
--
-- ── Return value ──────────────────────────────────────────────────────
--
--   {"NOT_FOUND"}
--   {"FENCED", <latest live reader deadline − now, ms ≥ 1>}
--   {"REGISTERED", <intent_ttl(L), whole ms>} — the caller renews every
--                  quarter of it, passing it back to
--                  `snapshot_intent_renew.lua`.
--   `redis.error_reply("WRONGTYPE_KEY: <key> holds a <type>, expected hash")`
--
-- A malformed argument is an error reply raised before any read or write.

local intent_key = KEYS[1]
local fence_key = KEYS[2]
local header_key = KEYS[3]

local token = ARGV[1]
if token == nil or token == "" then
  return redis.error_reply("ERR snapshot_intent_register: empty token")
end

local function count_arg(i, name)
  local v = tonumber(ARGV[i])
  if v == nil or v < 0 or v ~= math.floor(v) or v == math.huge then
    return nil, "ERR snapshot_intent_register: " .. name .. " must be a non-negative integer, got " .. tostring(ARGV[i])
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
    return redis.error_reply("ERR snapshot_intent_register: coefficient " .. name .. " must be a finite number, got " .. tostring(ARGV[3 + i]))
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

-- 0. A removed object takes no intent.
if redis.call("EXISTS", header_key) == 0 then
  return {"NOT_FOUND"}
end

-- 0a. Both keys this script reads or writes must be hashes (or absent).
for _, key in ipairs({fence_key, intent_key}) do
  local actual = redis.call("TYPE", key)["ok"]
  if actual ~= "hash" and actual ~= "none" then
    return redis.error_reply("WRONGTYPE_KEY: " .. key .. " holds a " .. actual .. ", expected hash")
  end
end

local tm = redis.call("TIME")
local now = tonumber(tm[1]) * 1000 + math.floor(tonumber(tm[2]) / 1000)

-- 1. A live reader refuses the intent.
local fence = redis.call("HGETALL", fence_key)
local latest_reader = nil
local fields = {}
for i = 1, #fence, 2 do
  local name, value = fence[i], fence[i + 1]
  if string.sub(name, 1, 2) == "r:" then
    local deadline = tonumber(value)
    if deadline ~= nil and deadline > now and (latest_reader == nil or deadline > latest_reader) then
      latest_reader = deadline
    end
  else
    fields[name] = value
  end
end
if latest_reader ~= nil then
  return {"FENCED", math.max(1, latest_reader - now)}
end

-- 2. Register.
local size_e, size_b = tonumber(fields["size_e"]), tonumber(fields["size_b"])
if size_e == nil or size_b == nil then
  size_e, size_b = est_e, est_b
end
local t = timing(size_e, size_b)
local ttl = math.ceil(t.intent_ttl)

local own_field = "w:" .. token
local latest = now + ttl
local intents = redis.call("HGETALL", intent_key)
-- Ordering invariant (b.cyu): write the own field BEFORE any prune. A prune
-- that empties the key before the write would make the write name an absent
-- key, which a migrating cluster slot refuses mid-script, tearing the run
-- (the expired intents gone, no intent registered). Written first, the own
-- field keeps the key non-empty for every prune and the `PEXPIRE` after it.
-- The stored result is the same as pruning first: the loop skips the own
-- field, and `latest` starts at its deadline.
redis.call("HSET", intent_key, own_field, ms(now + ttl))
for i = 1, #intents, 2 do
  local name, value = intents[i], intents[i + 1]
  if name ~= own_field and string.sub(name, 1, 2) == "w:" then
    local deadline = tonumber(value)
    if deadline == nil or deadline <= now then
      redis.call("HDEL", intent_key, name)
    elseif deadline > latest then
      latest = deadline
    end
  end
end
redis.call("PEXPIRE", intent_key, ms(math.max(1, latest - now)))

return {"REGISTERED", ttl}
