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

-- snapshot_intent_renew.lua — extend one WATCH-path writer's intent, only
-- while it is still live (Spec SDD SR-5.4).
--
-- An intent that has lapsed is never revived: once its deadline passed on
-- this node's `TIME`, readers were free to start an epoch, and a revived
-- intent would delay one that may already be running. Such a call answers
-- LOST and writes nothing. Readers in any case honour intents only until
-- `gap_until + intent_ttl(L)` (snapshot_fence_acquire.lua), so renewal cannot
-- starve them.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  intent — `le:{oid}:obj:snapintent` (KeySchema::obj_snapshot_intent)
--
-- ARGV:
--   [1]  token  — the attempt's token, as registered.
--   [2]  ttl_ms — the intent_ttl(L) `snapshot_intent_register.lua` returned;
--                 a positive integer.
--
-- On renewal: `w:<token>` = max(its deadline, now + ttl_ms), and the key's
-- expiry is raised, never lowered, to that deadline. Every writing script sets
-- the expiry to the latest live deadline, so keeping the larger of the two
-- keeps that invariant without reading every field.
--
-- ── Return value ──────────────────────────────────────────────────────
--
--   {"RENEWED", <new deadline ms>}
--   {"LOST"}  — the field is gone or its deadline has passed; nothing was
--               written.
--
-- A malformed argument is an error reply raised before any read or write.

local intent_key = KEYS[1]

local token = ARGV[1]
if token == nil or token == "" then
  return redis.error_reply("ERR snapshot_intent_renew: empty token")
end
local ttl = tonumber(ARGV[2])
if ttl == nil or ttl <= 0 or ttl ~= math.floor(ttl) or ttl == math.huge then
  return redis.error_reply("ERR snapshot_intent_renew: ttl_ms must be a positive integer, got " .. tostring(ARGV[2]))
end

local field = "w:" .. token
local tm = redis.call("TIME")
local now = tonumber(tm[1]) * 1000 + math.floor(tonumber(tm[2]) / 1000)

local current = tonumber(redis.call("HGET", intent_key, field))
if current == nil or current <= now then
  return {"LOST"}
end

local deadline = math.max(current, now + ttl)
redis.call("HSET", intent_key, field, string.format("%d", deadline))
local expire_ms = deadline - now
if redis.call("PTTL", intent_key) < expire_ms then
  redis.call("PEXPIRE", intent_key, string.format("%d", expire_ms))
end

return {"RENEWED", deadline}
