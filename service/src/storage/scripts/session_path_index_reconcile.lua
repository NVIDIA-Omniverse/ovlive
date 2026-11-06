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

-- session_path_index_reconcile.lua — remove every `path_index` member at a
-- base path that has no names member, in one atomic step on the
-- `{hex(base_path)}` slot.
--
-- Run as one `EVALSHA` on the data pool by `SessionStore::reconcile_path_index`,
-- which the `ListSessions` handler calls when `SessionStore::list` found a
-- `path_index` member its names read did not list.
--
-- Safe at any time: every writer adds a session's two members atomically (the
-- `CreateSession` commit, the session checkpoint applier's Step A, the DR
-- restore's Step A), and every remover takes the `path_index` member first or
-- both at once. So a `path_index` member no names member carries is always
-- garbage: what an older build's two-command clean-up left when its second
-- command failed. Nothing reads `path_index` for correctness.
--
-- A names member is `name\0session_id`, split on its FIRST NUL, the rule
-- `KeySchema::split_path_name_member` follows; a member with no NUL names no
-- session id.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (both keys carry the `{hex(base_path)}` hash tag — one Cluster slot):
--   [1] path index          (le:path:{hex(base_path)})          SET
--   [2] path names index    (le:path:{hex(base_path)}:names)    SET
--
-- ARGV: none.
--
-- Return value: the number of `path_index` members removed (0 when none was
-- stale).
--   `redis.error_reply("WRONGTYPE_KEY: <key> holds a <type>, expected set
--       (<what>)")` — a key holds the wrong Valkey type. Wrote NOTHING.
--
-- ── Why it cannot tear ───────────────────────────────────────────────────
--
-- Every read comes before the first write, and the only writes are `SREM`s on
-- `path_index`, which its `TYPE` found present. The removals are chunked under
-- Lua's `unpack` limit; only the last chunk can empty the set, so every
-- `SREM` names a key present then (SDD §18.1, A1).

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

if index_type == "none" then
  return 0
end

-- ── 2. The session ids the names index carries. ──────────────────────────
local listed = {}
if names_type == "set" then
  for _, member in ipairs(redis.call("SMEMBERS", names_key)) do
    local sep = string.find(member, "\0", 1, true)
    if sep then
      listed[string.sub(member, sep + 1)] = true
    end
  end
end

-- ── 3. The `path_index` members no names member carries. ─────────────────
local stale = {}
for _, sid in ipairs(redis.call("SMEMBERS", index_key)) do
  if not listed[sid] then
    stale[#stale + 1] = sid
  end
end

-- ── 4. Remove them, 500 per `SREM` (`unpack` over a large table overflows
-- the Lua C stack). ─────────────────────────────────────────────────────────
local removed = 0
local i = 1
while i <= #stale do
  local last = math.min(i + 499, #stale)
  removed = removed + redis.call("SREM", index_key, unpack(stale, i, last))
  i = last + 1
end
return removed
