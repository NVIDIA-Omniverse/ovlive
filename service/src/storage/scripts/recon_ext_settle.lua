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

-- recon_ext_settle.lua — settle one batch of the checkpoint collect on the
-- `{recon}` slot: retire what the batch finished with, keep what a writer
-- re-armed, date what was deleted, and report the kept hashes an earlier
-- delete left a `:collected` record for (b.g65, SDD §22.17.5).
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:ckpt:{recon}:tomb
--   [2]  le:ckpt:{recon}:rearm
--   [3]  le:ckpt:{recon}:collected
--
-- ARGV:
--   [1]  threshold_ms   the batch's snapshot threshold, t0 − grace.
--   [2]  n_spared       then, in this order, the five groups' sizes:
--   [3]  n_deleted
--   [4]  n_not_live
--   [5]  n_healed
--   [6]  n_kept
--   [7..] the hashes, group after group:
--         spared    in the collect's live set: `ZREM` from `:tomb` and
--                   `:rearm` — unless the hash has a `:collected` entry and
--                   is not in this run's `healed` group. Then an earlier
--                   delete of its backup copy is unsettled and a retained
--                   backup names it: its tombstone stays, so a later claim
--                   serves it again and the caller retries the heal
--                   (heal-on-keep) until it lands. The caller passes it
--                   here again with `healed` once healed.
--         deleted   their backup copy was deleted this batch: `ZREM` from
--                   `:tomb` unless a writer re-armed the hash after the
--                   threshold (then the tombstone stays and the next pass
--                   finds the new sidecar in its live set).
--         not_live  deleted, and the post-delete step found the value gone
--                   from Valkey: the `:collected` entry is re-dated to this
--                   run's time (`ZADD XX`), so a writer that started before
--                   the delete settled refuses.
--         healed    deleted, and the value was re-put from Valkey: `ZREM`
--                   from `:collected`.
--         kept      not deleted (spared, or deferred because the value is live
--                   in Valkey or a writer re-armed it): only read.
--
-- Returns the `kept` hashes that have a `:collected` entry: an earlier delete
-- of their backup copy was never settled (a crash, or a heal that found no
-- budget or failed), so the caller re-puts them from Valkey (heal-on-keep) and
-- settles them as `healed`.
--
-- ── Store memory (§7.5.1) ─────────────────────────────────────────────────
--
-- Opens with a memory-neutral write, `ZREM :tomb ":oom-gate"` (no member
-- starting with `:` is ever written), so on a store over `maxmemory` the
-- re-date of a `:collected` entry still lands after a delete.
--
-- ── Slot migrations (b.cyu) ───────────────────────────────────────────────
--
-- The opener and every read of `:rearm` and `:collected` come before the
-- first real write, so on a migrating source an absent key refuses the run
-- before it changes anything. Each key is then written by at most one `ZREM` that can
-- empty it, and nothing names that key afterwards: `:tomb` and `:rearm` once
-- each, and `:collected` by an `XX` re-date (which cannot empty it) before its
-- one `ZREM`.

local tomb_key      = KEYS[1]
local rearm_key     = KEYS[2]
local collected_key = KEYS[3]
local threshold_ms  = tonumber(ARGV[1])
local sizes = {
  tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4]), tonumber(ARGV[5]), tonumber(ARGV[6])
}

if threshold_ms == nil then
  return redis.error_reply("ENCODING: expected ARGV threshold_ms")
end
local total = 0
for g = 1, 5 do
  if sizes[g] == nil or sizes[g] < 0 then
    return redis.error_reply("ENCODING: expected five group sizes")
  end
  total = total + sizes[g]
end
if #ARGV ~= 6 + total then
  return redis.error_reply("ENCODING: the group sizes do not match the hashes given")
end

local groups = {}
local at = 7
for g = 1, 5 do
  local members = {}
  for _ = 1, sizes[g] do
    members[#members + 1] = ARGV[at]
    at = at + 1
  end
  groups[g] = members
end
local spared, deleted, not_live, healed, kept =
  groups[1], groups[2], groups[3], groups[4], groups[5]

-- The opener (see "Store memory" above). `:oom-gate` is never written.
redis.call("ZREM", tomb_key, ":oom-gate")

-- Reads.
local deleted_rearms = {}
if #deleted > 0 then
  deleted_rearms = redis.call("ZMSCORE", rearm_key, unpack(deleted))
else
  redis.call("ZCARD", rearm_key)
end
local kept_collected = {}
if #kept > 0 then
  kept_collected = redis.call("ZMSCORE", collected_key, unpack(kept))
else
  redis.call("ZCARD", collected_key)
end
local spared_collected = {}
if #spared > 0 then
  spared_collected = redis.call("ZMSCORE", collected_key, unpack(spared))
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

-- A spared hash is retired unless an unsettled delete record stands for it
-- (see `spared` in the contract above).
local healed_now = {}
for _, hash in ipairs(healed) do
  healed_now[hash] = true
end
local spared_retired = {}
for i, hash in ipairs(spared) do
  if spared_collected[i] == false or healed_now[hash] then
    spared_retired[#spared_retired + 1] = hash
  end
end

-- Writes: `:tomb`, one `ZREM`.
local retire = {}
for _, hash in ipairs(spared_retired) do
  retire[#retire + 1] = hash
end
for i, hash in ipairs(deleted) do
  local rearm = deleted_rearms[i]
  if rearm == false or tonumber(rearm) <= threshold_ms then
    retire[#retire + 1] = hash
  end
end
if #retire > 0 then
  redis.call("ZREM", tomb_key, unpack(retire))
end

-- Writes: `:rearm`, one `ZREM`.
if #spared_retired > 0 then
  redis.call("ZREM", rearm_key, unpack(spared_retired))
end

-- Writes: `:collected`, the re-date, then one `ZREM`.
if #not_live > 0 then
  local args = {}
  for _, hash in ipairs(not_live) do
    args[#args + 1] = now_ms
    args[#args + 1] = hash
  end
  redis.call("ZADD", collected_key, "XX", unpack(args))
end
if #healed > 0 then
  redis.call("ZREM", collected_key, unpack(healed))
end

local reply = {}
for i, hash in ipairs(kept) do
  if kept_collected[i] then
    reply[#reply + 1] = hash
  end
end
return reply
