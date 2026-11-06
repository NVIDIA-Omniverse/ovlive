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

-- session_index_release.lua — release one session id's two index members at
-- a base path: its `path_index` member, then its names member, in one atomic
-- step on the `{hex(base_path)}` slot.
--
-- Run as one `EVALSHA` on the data pool by every remover of a session's index
-- members other than `DeleteSession`:
--   - the `ListSessions` ghost walk (`SessionStore::retire_ghost_member`),
--     after `session_meta_veto.lua` answered `VETOED` or `DELETED`;
--   - `CreateSession`'s clean-ups (`SessionStore`): the compensation of a
--     commit that ran without its `WATCH`, the release of an attempt whose
--     `EXEC` outcome could not be read, and the release after a metadata
--     write the walk or the settle vetoed.
-- One script rather than two `SREM`s, so a release never leaves a
-- `path_index` member without its names member.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (both keys carry the `{hex(base_path)}` hash tag — one Cluster slot):
--   [1] path index          (le:path:{hex(base_path)})          SET
--   [2] path names index    (le:path:{hex(base_path)}:names)    SET
--
-- ARGV:
--   [1] session_id          (the `path_index` member)
--   [2] names member        (`name\0session_id`)
--
-- Return value: {<path_index members removed>, <names members removed>}, each
-- 0 or 1. Releasing members already gone is a no-op, so the script is
-- idempotent.
--   `redis.error_reply("WRONGTYPE_KEY: <key> holds a <type>, expected set
--       (<what>)")` — a key holds the wrong Valkey type. Wrote NOTHING.
--
-- ── Why it cannot tear ───────────────────────────────────────────────────
--
-- Every refusable read comes before the first write, and an `SREM` runs only
-- on a key its `TYPE` found present, so every write names a key present then
-- (SDD §18.1, A1). The `path_index` member goes first: were the run ever cut
-- between the two writes, the leftover would be the names member alone, the
-- same state `DeleteSession`'s own order passes through, and the next
-- `ListSessions` walk would find it.

local index_key = KEYS[1]
local names_key = KEYS[2]

-- `TYPE` replies with a status table carrying `ok`, and never errors.
local index_type = redis.call("TYPE", index_key)["ok"]
local names_type = redis.call("TYPE", names_key)["ok"]

-- ── 1. Type checks, before any write. ────────────────────────────────────
if index_type ~= "set" and index_type ~= "none" then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. index_key .. " holds a " .. index_type ..
    ", expected set (path index)")
end
if names_type ~= "set" and names_type ~= "none" then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. names_key .. " holds a " .. names_type ..
    ", expected set (path names index)")
end

-- ── 2. Release: `path_index` first, then the names member; an absent key
-- is skipped, never named. ─────────────────────────────────────────────────
local index_removed = 0
local names_removed = 0
if index_type == "set" then
  index_removed = redis.call("SREM", index_key, ARGV[1])
end
if names_type == "set" then
  names_removed = redis.call("SREM", names_key, ARGV[2])
end
return { index_removed, names_removed }
