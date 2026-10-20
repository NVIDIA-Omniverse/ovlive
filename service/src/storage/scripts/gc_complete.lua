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

-- gc_complete.lua — release a GC claim once its decrement has been decided,
-- and settle the member in the release ledger (b.yuo).
--
-- Replaces the bare `ZREM le:ext:{gc}:inflight <member>` the sweep issued after
-- `gc_decrement_and_collect.lua`. The caller runs it after a decrement that
-- answered `decremented`, `deleted`, `deleted_underflow` or `replayed`; never
-- after `stale` or an error, which leave the member in `inflight` for a later
-- pass to re-claim in place (`gc_reclaim_stale.lua`).
--
-- Only if `ZREM inflight member` returned 1 AND the member is not in `pending`:
--   * read its `pending:bytes` field and `HDEL` it; only if the `HDEL` returned
--     1, `DECRBY pending:total` by that value — so a member's bytes leave the
--     total exactly once, at its first settle;
--   * `ZADD settled <now_ms> member` (a plain `ZADD`: a re-settle overwrites
--     the score). The GC loop tears down the member's replay record
--     (`le:ext:applied:{hash_hex}`) once that entry is the replay-record
--     retention old.
-- Otherwise nothing more is written. A member back in `pending` (re-enqueued
-- while this sweep held it — only an older build's bare `SADD` does that; this
-- build's enqueue refuses a claimed member) keeps its bytes for that pending
-- copy, and settles when that copy completes. A `ZREM` that removed nothing
-- means another sweeper already completed the member (or, for an older
-- build's claim, dropped it).
--
-- `now_ms` is `redis.call("TIME")` on the `{gc}` primary — the same clock the
-- claim times in `inflight` and the replay records' scores come from, which is
-- what lets `gc_unmark.lua` compare a record's claim time with this settle time.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS (all `{gc}`-tagged, so the script is single-slot on a Cluster):
--   [1]  le:ext:{gc}:inflight       (ZSET)
--   [2]  le:ext:{gc}:pending        (SET)
--   [3]  le:ext:{gc}:pending:bytes  (HASH)
--   [4]  le:ext:{gc}:pending:total  (STRING)
--   [5]  le:ext:{gc}:settled        (ZSET)
--
-- ARGV:
--   [1]  member
--
-- Safe under two sweepers and under re-execution (a fred re-send): every write
-- is gated on the `ZREM` / `HDEL` that only one run can win.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Integer: 1 when this run settled the member, 0 otherwise.

local inflight_key = KEYS[1]
local pending_key  = KEYS[2]
local bytes_key    = KEYS[3]
local total_key    = KEYS[4]
local settled_key  = KEYS[5]
local member       = ARGV[1]

if redis.call("ZREM", inflight_key, member) ~= 1 then
  return 0
end
if redis.call("SISMEMBER", pending_key, member) == 1 then
  return 0
end

local held = redis.call("HGET", bytes_key, member)
if redis.call("HDEL", bytes_key, member) == 1 then
  local b = tonumber(held)
  if b and b > 0 then
    redis.call("DECRBY", total_key, b)
  end
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call("ZADD", settled_key, now_ms, member)
return 1
