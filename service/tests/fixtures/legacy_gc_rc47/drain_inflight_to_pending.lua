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
--
-- ── VENDORED TEST FIXTURE (b.1jt) — never loaded by the service ──────────
--
-- A verbatim copy of `src/storage/scripts/drain_inflight_to_pending.lua` at commit bf32b533
-- (0.2.0-rc.47, the last build before b.yuo's record-aware GC: a record-blind
-- GC with an unmarked leader lock).
-- Every earlier release runs the same code: 0.1.0 and 0.1.1 differ from it
-- only in headers and comments. `tests/integration/storage/b_1jt_rollout_tests.rs`
-- runs it against the store exactly as an older build's replica would, to
-- prove this build's state is safe in a mixed-build store. Only this block was
-- added; everything below it is the older build's file, unchanged.

-- drain_inflight_to_pending.lua — atomic Valkey-TIME-based drain of
-- stale `ext_inflight` claims back onto `ext_pending` (b.u96 item 7,
-- carrying forward b.j9j item 2).
--
-- Replaces the pre-b.u96 three-step Rust caller:
--   1. compute `cutoff_ms` from local `SystemTime::now()` minus
--      `LEADER_LOCK_TTL_MS`
--   2. ZRANGEBYSCORE inflight 0 cutoff_ms — fetch stale members
--   3. SADD pending <stale> ; ZREM inflight <stale>
--
-- Pre-b.u96 the local SystemTime exposed the drain to NTP skew: a
-- replica running clock-behind would compute a cutoff_ms ahead of
-- itself, treating fresh in-flight claims by peer replicas as stale
-- and re-queueing them — regressing the b.j9j double-decrement
-- scenario the ZSET timestamp guard was designed to prevent. The
-- Valkey-side `redis.call("TIME")` is consistent across all replicas
-- reading the same Valkey instance, eliminating that failure mode
-- entirely.
--
-- The script also collapses the SADD + ZREM pair into a single
-- atomic block. Pre-b.u96 these were two pipelined commands: a
-- replica crashing between them would leave the hash in BOTH sets,
-- which the next `gc_claim_pending.lua` SREM-then-ZADD would
-- harmlessly clean up — but the atomic shape here removes the
-- transient duplicate-state window altogether. Lua scripts run under
-- Valkey's single server thread so the ZRANGEBYSCORE + SADD + ZREM
-- chain is observed as one operation.
--
-- The SADD / ZREM fan-out is chunked through `unpack(stale, i, j)`
-- slices of `CHUNK_SIZE` to defend against Lua 5.1's MAXARG cliff
-- (~7990 in stock builds; lower in some custom builds). A fleet-wide
-- crash that leaves thousands of stale claims for one leader to drain
-- in a single tick would otherwise hit the unpack limit. Each chunked
-- call is still atomic against external clients (the whole script
-- runs under Valkey's single server thread), and the
-- `ZRANGEBYSCORE` snapshot is fixed at the top of the script, so the
-- chunked loop is semantics-preserving. See the b.w29 post-completion
-- review of the b.u96 sweep for the observation that motivated this.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  ext_inflight    (le:ext:{gc}:inflight — ZSET; score = claim ms)
--   [2]  ext_pending     (le:ext:{gc}:pending — SET)
--
-- ARGV:
--   [1]  ttl_ms          (the leader-lock TTL in milliseconds; claims
--                         whose score is older than `now_ms - ttl_ms`
--                         are treated as crash victims)
--
-- Both KEYS share the `{gc}` Cluster slot so the script is slot-safe
-- on a Valkey Cluster deployment.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Two-element array:
--   [1]  count_drained   (integer — number of stale entries moved back
--                         to pending; zero if every entry is fresh or
--                         inflight is empty)
--   [2]  inflight_total  (integer — current cardinality of inflight
--                         AFTER the drain; lets the caller log
--                         debug-level "still N fresh in inflight" when
--                         drained=0 but the set is non-empty)

local inflight_key = KEYS[1]
local pending_key  = KEYS[2]
local ttl_ms       = tonumber(ARGV[1])

-- Chunk size for the SADD / ZREM fan-out. Stays well below Lua 5.1's
-- MAXARG cliff (~7990 in stock builds) with comfortable headroom for
-- custom builds that lower the ceiling. See the header comment.
local CHUNK_SIZE = 1024

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

-- Saturating subtract — a wildly-misconfigured Valkey clock (e.g.
-- now_ms < ttl_ms during early epoch) should not underflow into a
-- huge positive cutoff that drains everything.
local cutoff_ms
if now_ms < ttl_ms then
  cutoff_ms = 0
else
  cutoff_ms = now_ms - ttl_ms
end

local stale = redis.call("ZRANGEBYSCORE", inflight_key, 0, cutoff_ms)
if #stale == 0 then
  return { 0, redis.call("ZCARD", inflight_key) }
end

-- SADD + ZREM in one script-atomic block. Lua scripts run under
-- Valkey's single server thread, so even though these are two
-- separate commands, no external observer sees the intermediate
-- state where the hash is in BOTH sets. (Compare to the pre-b.u96
-- Rust caller, which pipelined the two commands and accepted a
-- transient duplicate-state window covered by gc_claim_pending's
-- SREM-then-ZADD idempotency.)
--
-- The two calls fan out across `CHUNK_SIZE` slices of `stale` to
-- stay below Lua 5.1's MAXARG cliff (see header). Atomicity is
-- preserved — the entire script still runs under Valkey's single
-- server thread, so external clients observe one consistent
-- transition from the inflight snapshot to (inflight − stale,
-- pending ∪ stale).
local n = #stale
for i = 1, n, CHUNK_SIZE do
  local j = i + CHUNK_SIZE - 1
  if j > n then j = n end
  redis.call("SADD", pending_key, unpack(stale, i, j))
  redis.call("ZREM", inflight_key, unpack(stale, i, j))
end

return { n, redis.call("ZCARD", inflight_key) }
