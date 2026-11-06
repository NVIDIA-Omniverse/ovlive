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

-- recon_orphan_commit.lua — license the checkpoint orphan reap of one claimed
-- stem, atomically against a publish of the same stem, on the `{recon}` slot
-- (b.g65, SDD §22.17.5).
--
-- ── The race it closes ────────────────────────────────────────────────────
--
-- A publish that failed part-way leaves a stem with no generation artifact,
-- which the orphan phase registers and, a grace later, reaps. A writer can
-- re-publish that same stem (the same sequence) in the meantime. Without a
-- fence, the re-publish could finish while the reap deletes the stem's
-- artifacts, losing the newest backup. Every publish runs
-- `recon_ext_rearm.lua` before its PUT, which removes the stem's member from
-- `:orphans` and refuses while the stem is announced on `:deleting`. This
-- script is the reap's half:
--
-- * member still in `:orphans` → no publish has healed it since it was
--   claimed: announce the stem on `:deleting` at this run's time and license
--   the reap. A publish that re-arms afterwards sees the announcement and
--   refuses until a grace has passed.
-- * member gone from `:orphans` → a publish healed it: not licensed, and the
--   caller deletes nothing (the claim's retire of the member is then a no-op).
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:ckpt:{recon}:orphans
--   [2]  le:ckpt:{recon}:deleting
--
-- ARGV:
--   [1]  member     the claimed stem, in the `:orphans` spelling.
--   [2]  grace_ms   the reconciliation grace; `:deleting` keeps two of them.
--
-- Returns 1 when licensed (the announcement written), 0 otherwise.
--
-- The announcement is trimmed to entries younger than two grace windows, and
-- `:deleting` carries `PEXPIRE` 24 h. No out-of-memory opener: on a store over
-- `maxmemory` the announcement is refused and the reap deletes nothing.
--
-- ── Slot migrations (b.cyu) ───────────────────────────────────────────────
--
-- The read of `:orphans` comes first and the `ZADD` of `:deleting` is the
-- first write, so on a migrating source either absent key refuses the run
-- before it writes. The trim and `PEXPIRE` then name `:deleting`, which the
-- `ZADD` just filled with a member the trim cannot remove.

local orphans_key  = KEYS[1]
local deleting_key = KEYS[2]
local member       = ARGV[1]
local grace_ms     = tonumber(ARGV[2])

if member == nil or member == "" or grace_ms == nil then
  return redis.error_reply("ENCODING: expected ARGV member, grace_ms")
end

if redis.call("ZSCORE", orphans_key, member) == false then
  return 0
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

redis.call("ZADD", deleting_key, now_ms, member)
redis.call("ZREMRANGEBYSCORE", deleting_key, "-inf",
  "(" .. string.format("%d", now_ms - 2 * grace_ms))
redis.call("PEXPIRE", deleting_key, 86400000)
return 1
