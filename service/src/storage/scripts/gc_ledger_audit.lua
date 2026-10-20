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

-- gc_ledger_audit.lua — one page of the release-ledger audit (b.yuo).
--
-- The backstop that settles a ledger entry whose member left both `pending`
-- and `inflight` without passing through `gc_complete.lua` or the claim's
-- self-heal. In practice that is a member an older build completed with a
-- bare `ZREM inflight` while both builds ran (a rolling upgrade), or one an
-- older replica's drain moved around. Without the audit such a member would
-- keep its bytes in the budget total forever, and — if this build decremented
-- it — its replay record would never be torn down.
--
-- The claim's `HSETNX pending:bytes <member> 0` guarantees every member this
-- build may have decremented has a ledger field until it settles, so walking
-- `pending:bytes` finds every such member.
--
-- For each field of one `HSCAN pending:bytes <cursor> COUNT 256` page whose
-- member is in neither `pending` nor `inflight`:
--   * `ZREM pending:age member`;
--   * `HDEL pending:bytes member`, and only if that returned 1, `DECRBY
--     pending:total` by the value it held — bytes leave the total once;
--   * `ZADD settled <now_ms> member`, so the member's replay record (if any)
--     is torn down after the retention like any other.
-- A member in either set is live and untouched. Atomic per page, so a member
-- cannot move between the membership check and the repair.
--
-- The GC loop starts a cycle every few minutes (and at the first pass of a
-- new leadership) and runs one page per pass until the cursor wraps to "0".
-- It logs a WARN when a page repairs anything: expected only around a rolling
-- upgrade.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS (all `{gc}`-tagged, so the script is single-slot on a Cluster):
--   [1]  le:ext:{gc}:pending        (SET)
--   [2]  le:ext:{gc}:inflight       (ZSET)
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
-- earlier writes kept. This page can empty `pending:age` while repairing one
-- member and then name it again for the next, leaving a partial page.
-- Declaring the never-written sentinel makes the store refuse the whole
-- script before it starts while the `{gc}` slot is moving (`-TRYAGAIN`, or
-- `-ASK`, which the service never follows for a script; both transient),
-- and costs nothing on a stable slot: the audit pass just ends early and
-- re-runs the page later.
--
-- ARGV:
--   [1]  cursor   (the HSCAN cursor; "0" starts a cycle)
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Two-element array `{ next_cursor, repaired }`: the HSCAN cursor to pass
-- next ("0" when the cycle is complete) and the number of members settled.

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
      "ENCODING: gc_ledger_audit expects 7 KEYS, KEYS[7] the migration sentinel of " ..
      "KEYS[1]'s hash tag")
  end
end

local pending_key  = KEYS[1]
local inflight_key = KEYS[2]
local age_key      = KEYS[3]
local bytes_key    = KEYS[4]
local total_key    = KEYS[5]
local settled_key  = KEYS[6]
local cursor       = ARGV[1]
if cursor == nil or string.find(cursor, "^%d+$") == nil then
  return redis.error_reply("ENCODING: expected a decimal HSCAN cursor in ARGV[1]")
end

local page = redis.call("HSCAN", bytes_key, cursor, "COUNT", 256)
local next_cursor = page[1]
local entries = page[2]

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local repaired = 0
for i = 1, #entries, 2 do
  local member = entries[i]
  if redis.call("SISMEMBER", pending_key, member) == 0
      and not redis.call("ZSCORE", inflight_key, member) then
    redis.call("ZREM", age_key, member)
    if redis.call("HDEL", bytes_key, member) == 1 then
      local b = tonumber(entries[i + 1])
      if b and b > 0 then
        redis.call("DECRBY", total_key, b)
      end
    end
    redis.call("ZADD", settled_key, now_ms, member)
    repaired = repaired + 1
  end
end

return { next_cursor, repaired }
