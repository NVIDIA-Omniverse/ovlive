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

-- recon_redate.lua — move existing members of one `{recon}` work list later,
-- to the Valkey clock (b.g65, SDD §22.17.5).
--
-- The reconciliation claims serve the oldest members first. A member the pass
-- defers for a reason of its own — a dead object whose header still stands
-- with `superseded=1`, two listings that disagreed about it, a sidecar read or
-- a delete that failed, a collect candidate whose value is still live in
-- Valkey — would otherwise keep its old score and come back at the head of
-- every claim, so enough of them fill the claim window and block every newer
-- member behind them. Re-dating such a member to now moves it behind those.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  the work list: le:ckpt:{recon}:dead, :orphans or :tomb
--
-- ARGV:
--   [1..] members   the members to re-date.
--
-- One `ZADD XX GT now member...`: `XX` never creates a member (a member
-- retired meanwhile stays retired) and `GT` never lowers a score, so a score
-- only ever moves later. A later score reads to every reader, an older build's
-- included, as a later first sighting, which only delays an action. Returns
-- the number of members re-dated.
--
-- A failed re-date costs only head-of-line order, so the caller treats it as
-- non-fatal, and the script needs no out-of-memory opener.
--
-- ── Slot migrations (b.cyu) ───────────────────────────────────────────────
--
-- One write command, on its one key: on a migrating source an absent key
-- refuses it before anything is written.

local key = KEYS[1]
if #ARGV == 0 then
  return 0
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local args = {}
for i = 1, #ARGV do
  args[#args + 1] = now_ms
  args[#args + 1] = ARGV[i]
end
return redis.call("ZADD", key, "XX", "GT", "CH", unpack(args))
