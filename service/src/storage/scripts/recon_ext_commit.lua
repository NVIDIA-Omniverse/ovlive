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

-- recon_ext_commit.lua — the checkpoint collect's delete-time fence: license,
-- atomically on the `{recon}` slot, the backup-store delete of each claimed
-- ext hash that neither the collect's live set nor Valkey still holds
-- (b.g65, SDD §22.17.5).
--
-- ── Why a fence at delete time ────────────────────────────────────────────
--
-- The collect reads its live set once and then deletes for up to the whole
-- collect budget. A base naming a hash can publish after that read, and a hash
-- can be used again after its tombstone aged. The collect therefore anchors
-- every delete to the clock it read just before its live-set walk (`t0`): a
-- hash may be deleted only when its tombstone and its latest writer re-arm are
-- both at or before `threshold = t0 − grace`. A re-arm at or before the
-- threshold means the writer's sidecar was visible before `t0` (a re-arm lands
-- at most one publish worst case before its sidecar, and the grace is a
-- multiple of that), so the snapshot already reflects it; a later re-arm means
-- it may not, and the hash is not deleted this pass.
--
-- That argument needs every re-arm newer than the threshold to still be on
-- `:rearm` when this run reads it. Every run trims `:rearm` entries older than
-- two grace windows — a peer runner's run included — so it holds only while
-- this run's clock is at most one grace past `t0`: then the trim's cutoff,
-- `now − 2 × grace`, is at or before `t0 − grace`. The bound is checked here,
-- against this run's own clock, rather than trusted to the caller's deadline:
-- a run later than that licenses nothing.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:ckpt:{recon}:tomb
--   [2]  le:ckpt:{recon}:rearm
--   [3]  le:ckpt:{recon}:collected
--
-- ARGV:
--   [1]  t0_ms         the collect's snapshot clock, read just before its
--                      live-set walk, in Valkey ms; the threshold is
--                      t0_ms − grace_ms.
--   [2]  grace_ms      the reconciliation grace; `:rearm` keeps two of them.
--   [3..] hashes       the candidates: claimed, absent from the live set and
--                      not live in Valkey.
--
-- Returns one status per candidate, in order:
--   "licensed"   recorded on `:collected` at this run's time; the caller may
--                delete the backup copy.
--   "rearmed"    a writer re-armed the hash after the threshold; its
--                tombstone is re-dated to now (`ZADD XX GT`) so the claim
--                serves the members behind it; not deleted.
--   "too_young"  the tombstone is newer than the threshold (or no longer a
--                member, retired by a duplicate runner), or this run is more
--                than one grace past `t0` (every candidate, then); nothing
--                written. The claim serves oldest first, so the caller's drain
--                ends here.
--
-- ── What it writes ────────────────────────────────────────────────────────
--
-- * `ZADD :collected now hash` for every licensed hash, BEFORE the caller's
--   DELETE: a writer that re-arms after this run sees the entry and refuses
--   its PUT (`recon_ext_rearm.lua`). Then the trim of `:collected` entries
--   older than 24 h and `PEXPIRE :collected` 24 h.
-- * `ZADD :tomb XX GT now hash` for every re-armed hash. `XX` never creates
--   a member and `GT` never lowers a score.
-- * The trim of `:rearm` entries older than two grace windows: older re-arms
--   are at or before any later collect's threshold, so they can no longer
--   refuse anything.
--
-- No out-of-memory opener: on a store over `maxmemory` the first `ZADD` is
-- refused, nothing is licensed, and the collect deletes nothing (fail closed).
--
-- ── Slot migrations (b.cyu) ───────────────────────────────────────────────
--
-- Every read — `:tomb`, `:rearm` and `:collected` — comes before the first
-- write, so on a migrating source an absent key refuses the run before it
-- writes. Every write then names a key the run proved present, and the one
-- `PEXPIRE` follows the `ZADD` that just filled `:collected`; the trims can
-- empty `:rearm` or `:collected`, and nothing names either after its trim. A
-- run past the `t0` bound returns after its reads and writes nothing.

local tomb_key      = KEYS[1]
local rearm_key     = KEYS[2]
local collected_key = KEYS[3]
local t0_ms         = tonumber(ARGV[1])
local grace_ms      = tonumber(ARGV[2])

if t0_ms == nil or grace_ms == nil then
  return redis.error_reply("ENCODING: expected ARGV t0_ms, grace_ms")
end
local threshold_ms = t0_ms - grace_ms

local hashes = {}
for i = 3, #ARGV do
  hashes[#hashes + 1] = ARGV[i]
end
if #hashes == 0 then
  return {}
end

-- Reads.
local tomb_scores = redis.call("ZMSCORE", tomb_key, unpack(hashes))
local rearm_scores = redis.call("ZMSCORE", rearm_key, unpack(hashes))
-- Read only to prove `:collected` present on a migrating source before the
-- first write (see "Slot migrations" above); the count is not used.
redis.call("ZCARD", collected_key)

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local statuses = {}

-- More than one grace past `t0`: a peer's trim may already have removed a
-- re-arm newer than the threshold, so nothing is licensed (see "Why a fence
-- at delete time" above).
if now_ms - t0_ms > grace_ms then
  for i = 1, #hashes do
    statuses[i] = "too_young"
  end
  return statuses
end

local licensed = 0
for i, hash in ipairs(hashes) do
  local tomb = tomb_scores[i]
  local rearm = rearm_scores[i]
  if tomb == false or tonumber(tomb) > threshold_ms then
    statuses[i] = "too_young"
  elseif rearm ~= false and tonumber(rearm) > threshold_ms then
    redis.call("ZADD", tomb_key, "XX", "GT", now_ms, hash)
    statuses[i] = "rearmed"
  else
    redis.call("ZADD", collected_key, now_ms, hash)
    licensed = licensed + 1
    statuses[i] = "licensed"
  end
end

if licensed > 0 then
  redis.call("ZREMRANGEBYSCORE", collected_key, "-inf",
    "(" .. string.format("%d", now_ms - 86400000))
  redis.call("PEXPIRE", collected_key, 86400000)
end
redis.call("ZREMRANGEBYSCORE", rearm_key, "-inf",
  "(" .. string.format("%d", now_ms - 2 * grace_ms))

return statuses
