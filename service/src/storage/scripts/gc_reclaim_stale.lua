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

-- gc_reclaim_stale.lua — re-claim one of this build's own stale `inflight`
-- claims in place (b.1jt).
--
-- A GC claim this build made stays in `le:ext:{gc}:inflight` when its sweep
-- crashed, a step failed, or its decrement was refused `stale`. Once its
-- claim is older than the re-claim age D (`GcPolicy::reclaim_after`, 30 s),
-- a GC pass re-claims it HERE instead of moving it back to `pending`: only
-- while its score is still exactly the one the pass's selection read
-- (`gc_select.lua`), it is re-scored with a fresh encoded claim time and the
-- pass then runs the usual decrement (`gc_decrement_and_collect.lua`) and
-- completion (`gc_complete.lua`).
--
-- Why in place. A member this build may already have decremented must never
-- go back to `pending`: an older build that later inherits `pending` (a
-- rollback, or one whose GC sweep is still running when this build takes the
-- lock) decrements every member there, and 0.1.0 through 0.2.0-rc.47 do so
-- record-blind, a second time. In place,
-- the member stays an `inflight` claim with an encoded score, which no older
-- build's drain or sweep selects. The decrement stays at most once: a member
-- whose decrement already applied is answered `replayed` by its replay
-- record, and the old claim's late copies were refused `stale` long ago,
-- because the decrement fence F (15 s) is below D.
--
-- Two passes that re-claim the same claim at once (two leaders, briefly) are
-- decided here: the first re-scores it, and the second finds the score
-- changed and answers 0.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:ext:{gc}:inflight  (ZSET)
--
-- ARGV:
--   [1]  member
--   [2]  expected_score  — the score `gc_select.lua` read, as the store
--                          reported it (compared numerically)
--   [3]  offset          — the claim-score offset
--                          (`GC_CLAIM_SCORE_OFFSET_MS`, a decimal)
--
-- Only a claim scored at or above the offset — this build's — is re-claimed;
-- an older build's claim is never re-claimed (it is dropped elsewhere,
-- `gc_drop_foreign_claims.lua`).
--
-- The new score is handed to `ZADD` as a Lua number (full precision; Lua's own
-- string conversion keeps only 14 significant digits).
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Integer: the new claim time in ms (unencoded; the decrement fences on it),
-- or 0 when the member is no longer in `inflight`, its score changed since it
-- was read, or it is not this build's claim. 0 writes nothing.

local inflight_key = KEYS[1]
local member = ARGV[1]
local expected = tonumber(ARGV[2])
local offset = tonumber(ARGV[3])
if member == nil or expected == nil or offset == nil then
  return redis.error_reply("ENCODING: gc_reclaim_stale expects ARGV member, expected score, offset")
end

local current = tonumber(redis.call("ZSCORE", inflight_key, member))
if current == nil or current ~= expected or current < offset then
  return 0
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call("ZADD", inflight_key, "XX", now_ms + offset, member)
return now_ms
