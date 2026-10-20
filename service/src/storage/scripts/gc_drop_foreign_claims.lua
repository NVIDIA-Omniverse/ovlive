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

-- gc_drop_foreign_claims.lua — drop an older build's stranded `inflight`
-- claims without a decrement (b.1jt).
--
-- Every older build (0.1.0 through 0.2.0-rc.48) scores its GC claims with the
-- bare claim time; this build adds `offset` (`GC_CLAIM_SCORE_OFFSET_MS`,
-- 10^13). So a claim scored below the offset was made by an older build, whose
-- decrement may or may not have applied: 0.1.0 through 0.2.0-rc.47 decrement
-- record-blind, so nothing on the store says which, and this build treats a
-- 0.2.0-rc.48 claim the same way. This build never decrements such a claim:
-- a second decrement of a still-referenced hash deletes a blob a layer still
-- points at. Once the claim is older than the decrement fence F (by then the
-- older build's own sweep is past its claim-to-decrement window), the GC pass
-- drops it here: if the older decrement applied, nothing more is owed; if it
-- did not, the hash keeps one reference too many, which the reference-count
-- reconciler repairs (unless it is turned off,
-- `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0`, when the leak is permanent).
-- Leak, never over-release. Dropping at F also beats an older build's drain
-- (at startup, or per pass in 0.2.0-rc.48), which re-queues a claim only once
-- it is 30 s old.
--
-- Per `(member, expected_score)` pair, only while the claim's score is still
-- exactly the one the pass's selection read, is below the offset, and is at
-- least F old by this script's clock:
--   * `ZREM inflight member`;
--   * if the member is not in `pending` (an older build may have re-enqueued
--     it with a bare `SADD` while it was claimed; that copy keeps its own
--     ledger entries and settles when it completes, as `gc_complete.lua`
--     leaves it): `ZREM pending:age member`; read and `HDEL pending:bytes
--     member`, and only if the `HDEL` removed it, `DECRBY pending:total` by
--     the bytes it held; `ZADD settled <now_ms> member`, so any replay record
--     of the member is torn down after the retention like any other.
-- Any other pair writes nothing.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS (all `{gc}`-tagged, so the script is single-slot on a Cluster):
--   [1]  le:ext:{gc}:inflight       (ZSET)
--   [2]  le:ext:{gc}:pending        (SET)
--   [3]  le:ext:{gc}:pending:age    (ZSET)
--   [4]  le:ext:{gc}:pending:bytes  (HASH)
--   [5]  le:ext:{gc}:pending:total  (STRING)
--   [6]  le:ext:{gc}:settled        (ZSET)
--   [7]  le:{gc}:migration-sentinel (never written; see below)
--
-- Exactly seven KEYS, and `KEYS[7]` must be the migration sentinel of
-- `KEYS[1]`'s hash tag (`KeySchema::migration_sentinel("gc")`), else the
-- script refuses `ENCODING` before any read or write. Nothing here names it.
--
-- ── Migration sentinel (b.cyu) ────────────────────────────────────────────
--
-- During a legacy cluster slot migration every in-script command is checked
-- against the slot's state again, and on the migrating source one that names
-- a key absent at that moment is refused, aborting the script with its
-- earlier writes kept. This loop can empty `pending:age` (one member's `ZREM`)
-- and then name it again for the next member, after that member's `ZREM
-- inflight` landed: the member would be left out of `inflight` and `settled`
-- with its bytes still held. Declaring the never-written sentinel makes the
-- store refuse the whole script before it starts while the `{gc}` slot is
-- moving (`-TRYAGAIN`, or `-ASK`, which the service never follows for a
-- script; both transient), and costs nothing on a stable slot: the GC pass
-- just ends early and retries on its next tick.
--
-- ARGV:
--   [1]     offset            — the claim-score offset, a decimal
--   [2]     foreign_after_ms  — F, a decimal
--   [3..]   pairs `member, expected_score`, the score as `gc_select.lua`
--           read it (compared numerically)
--
-- Idempotent under a re-run (a fred re-send): a dropped claim is gone, so its
-- pair writes nothing the second time.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- The members dropped, in argument order (possibly empty).

local KEY_MIGRATION_SENTINEL = 7

-- The migration sentinel check, before any read or write (see "Migration
-- sentinel").
do
  local function hash_tag(key)
    local open = string.find(key, "{", 1, true)
    if not open then return nil end
    local close = string.find(key, "}", open + 1, true)
    if not close or close == open + 1 then return nil end
    return string.sub(key, open + 1, close - 1)
  end
  local sentinel = KEYS[KEY_MIGRATION_SENTINEL]
  local suffix = ":migration-sentinel"
  if #KEYS ~= KEY_MIGRATION_SENTINEL
      or type(sentinel) ~= "string"
      or string.sub(sentinel, -string.len(suffix)) ~= suffix
      or hash_tag(sentinel) == nil
      or hash_tag(sentinel) ~= hash_tag(KEYS[1]) then
    return redis.error_reply(
      "ENCODING: gc_drop_foreign_claims expects 7 KEYS, KEYS[7] the migration sentinel of " ..
      "KEYS[1]'s hash tag")
  end
end

local inflight_key = KEYS[1]
local pending_key  = KEYS[2]
local age_key      = KEYS[3]
local bytes_key    = KEYS[4]
local total_key    = KEYS[5]
local settled_key  = KEYS[6]

local offset = tonumber(ARGV[1])
local foreign_after_ms = tonumber(ARGV[2])
if offset == nil or foreign_after_ms == nil or (#ARGV - 2) % 2 ~= 0 then
  return redis.error_reply("ENCODING: gc_drop_foreign_claims expects ARGV offset, fence, then (member, score) pairs")
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local dropped = {}
for i = 3, #ARGV, 2 do
  local member = ARGV[i]
  local expected = tonumber(ARGV[i + 1])
  local current = tonumber(redis.call("ZSCORE", inflight_key, member))
  if current ~= nil and expected ~= nil and current == expected and current < offset
      and now_ms - current >= foreign_after_ms then
    redis.call("ZREM", inflight_key, member)
    if redis.call("SISMEMBER", pending_key, member) == 0 then
      redis.call("ZREM", age_key, member)
      local held = redis.call("HGET", bytes_key, member)
      if redis.call("HDEL", bytes_key, member) == 1 then
        local b = tonumber(held)
        if b and b > 0 then
          redis.call("DECRBY", total_key, b)
        end
      end
      redis.call("ZADD", settled_key, now_ms, member)
    end
    dropped[#dropped + 1] = member
  end
end
return dropped
