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

-- recon_tomb_record.lua — record the ext hashes a base generation's sidecar
-- names as tombstoned, and announce the generation's stem as being deleted,
-- in one run on the `{recon}` slot (b.9qd, b.g65; SDD §22.17.3, §22.17.5).
--
-- ── Who calls it ──────────────────────────────────────────────────────────
--
-- Every production deleter of a base generation, after reading its ext-refs
-- sidecar and BEFORE deleting any of its artifacts
-- (`checkpoint::checkpointer::ExtTombstoneSink::record`): the retention prune,
-- the dead-object reap and the base bootstrap's dead-base reap. (The orphan
-- reap announces through `recon_orphan_commit.lua` before it reads, and then
-- records its hashes here too.)
--
-- ── Why both in one run ───────────────────────────────────────────────────
--
-- * The tombstone is what makes the hashes collectable at all: a generation
--   deleted before its hashes are recorded strands every blob it named.
-- * The announcement is what lets the collect explain a sidecar its live-set
--   walk listed and then read absent: a stem announced here was read and
--   tombstoned by a deleter that decided it is not retained, so skipping its
--   sidecar drops no reference a retained backup needs. Writing both in one
--   run means no deleter can have announced without tombstoning, or the
--   reverse.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:ckpt:{recon}:tomb
--   [2]  le:ckpt:{recon}:deleting
--
-- ARGV:
--   [1]  grace_ms   the reconciliation grace; `:deleting` keeps two of them.
--   [2]  n_stems    how many stems follow.
--   [3..] stems     (n_stems of them) in the `:orphans` spelling, then the
--                   hashes.
--
-- Writes `ZADD NX :tomb now hash` for each hash — `NX` pins the first
-- recording, which is what the claim's grace is measured from — and
-- `ZADD :deleting now stem` for each stem, then trims `:deleting` to two grace
-- windows and sets its `PEXPIRE` 24 h. `:tomb` carries no TTL: it is the only
-- record of owed reclamation. Returns the number of hashes newly tombstoned.
--
-- No out-of-memory opener: on a store over `maxmemory` the run is refused,
-- nothing is recorded, and the caller deletes nothing (it leaks rather than
-- strands).
--
-- ── Slot migrations (b.cyu) ───────────────────────────────────────────────
--
-- Both keys are read (`ZCARD`) before the first write, so on a migrating
-- source an absent key refuses the run before it writes. Every write then
-- names a key the run proved present, and a `ZADD` cannot empty its key.

local tomb_key     = KEYS[1]
local deleting_key = KEYS[2]
local grace_ms     = tonumber(ARGV[1])
local n_stems      = tonumber(ARGV[2])

if grace_ms == nil or n_stems == nil or n_stems < 0 or #ARGV < 2 + n_stems then
  return redis.error_reply("ENCODING: expected ARGV grace_ms, n_stems, stems, hashes")
end

-- Reads, only to prove both keys present on a migrating source before the
-- first write; the counts are not used.
redis.call("ZCARD", tomb_key)
redis.call("ZCARD", deleting_key)

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local added = 0
local tomb_args = {}
for i = 3 + n_stems, #ARGV do
  tomb_args[#tomb_args + 1] = now_ms
  tomb_args[#tomb_args + 1] = ARGV[i]
end
if #tomb_args > 0 then
  added = redis.call("ZADD", tomb_key, "NX", unpack(tomb_args))
end

if n_stems > 0 then
  local stem_args = {}
  for i = 3, 2 + n_stems do
    stem_args[#stem_args + 1] = now_ms
    stem_args[#stem_args + 1] = ARGV[i]
  end
  redis.call("ZADD", deleting_key, unpack(stem_args))
  redis.call("ZREMRANGEBYSCORE", deleting_key, "-inf",
    "(" .. string.format("%d", now_ms - 2 * grace_ms))
  redis.call("PEXPIRE", deleting_key, 86400000)
end

return added
