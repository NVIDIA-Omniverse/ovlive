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

-- gc_enqueue_attempt_audit.lua — retire the `{gc}` records of retirement
-- enqueue attempts nothing needs any more (b.yuo Part R).
--
-- ── What the records are ──────────────────────────────────────────────────
--
-- `le:ext:{gc}:enqueue_attempt` maps an attempt token to `landed:<ms>:<oid>`
-- (written by the attempt-mode `ext_release_enqueue.lua`) or `void:<ms>:<oid>`
-- (written by `ext_enqueue_attempt_resolve.lua`). A record is never changed
-- once written, and it is the only guard against a SEND under its token: an
-- attempt-mode enqueue that finds its token absent enqueues again. So a record
-- may go only once no send under its token can still arrive:
--
--   * F1 — a `landed` record is removed at once only by its token's own
--     sender (the retirement, or the orphan sweep's pay), and only when its
--     single send under the token replied cleanly and its carrier was then
--     cleared. That removal is not this script's.
--   * F2 — every other record is removed only here, once older than V
--     (`ENQUEUE_ATTEMPT_VOID_RETENTION`, 24 h): a `void` record always; a
--     `landed` record only while its object's `le:{oid}:obj:extret:attempts`
--     no longer holds its token. This script cannot see that `{object_id}`
--     key, so it RETURNS such records and the GC loop checks each with
--     `HEXISTS`, then calls this script again in `forget` mode for the ones no
--     carrier holds.
--   * F3 — no service send is decided more than Y
--     (`RETIRE_INTENT_SEND_WINDOW`, 30 s) after the command that recorded its
--     token was issued.
--
-- What is left is transit only: a send whose bytes reach the store more than
-- V − Y after the service decided it can still enqueue once more (an accepted
-- bound, policy 14).
--
-- ── The walk ──────────────────────────────────────────────────────────────
--
-- One explicit-cursor `HSCAN` page per call: the caller passes the cursor the
-- previous page returned, starting from "0", and the walk is complete only when
-- a page returns "0" (SDD §11.3.2: a walk is complete only on its terminal
-- page). Nothing is inferred from completeness — every page acts only on the
-- records it read — so a page the caller never sends costs only latency.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:ext:{gc}:enqueue_attempt  (HASH)
--
-- One `{gc}`-tagged key: single-slot, and keyed for the b.289 fence.
--
-- ARGV, `page` mode:
--   [1]  "page"
--   [2]  cursor        — the `HSCAN` cursor, "0" to start
--   [3]  retention_ms  — V, for both kinds of record
--   [4]  count         — the `HSCAN` COUNT hint
--
-- ARGV, `forget` mode:
--   [1]     "forget"
--   [2..]   pairs `token, value` — each `HDEL`ed only while its recorded value
--           is still exactly `value`.
--
-- Ages are measured on the `{gc}` primary's `TIME`, the clock every record was
-- stamped with.
--
-- ── Return value ─────────────────────────────────────────────────────────
--
-- `page`:   `{ next_cursor, void_removed, landed }`, where `landed` is a flat
--           list of triples `token, oid, value` — the landed records older
--           than V on this page. Every younger record is left untouched.
-- `forget`: the number of records removed.

local attempts = KEYS[1]
local mode = ARGV[1]

local function kind_ms_oid(value)
  local kind, ms, oid = string.match(value, "^(%a+):(%d+):(.*)$")
  return kind, tonumber(ms), oid
end

if mode == "forget" then
  if (#ARGV - 1) % 2 ~= 0 then
    return redis.error_reply("ENCODING: forget mode takes token, value pairs")
  end
  local removed = 0
  for i = 2, #ARGV, 2 do
    if redis.call("HGET", attempts, ARGV[i]) == ARGV[i + 1] then
      removed = removed + redis.call("HDEL", attempts, ARGV[i])
    end
  end
  return removed
end

if mode ~= "page" then
  return redis.error_reply("ENCODING: ARGV[1] must be \"page\" or \"forget\"")
end

local cursor = ARGV[2]
local retention_ms = tonumber(ARGV[3])
local count = tonumber(ARGV[4])
if cursor == nil or retention_ms == nil or count == nil then
  return redis.error_reply("ENCODING: page mode takes cursor, retention_ms, count")
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local page = redis.call("HSCAN", attempts, cursor, "COUNT", count)
local next_cursor = page[1]
local entries = page[2]

local void_removed = 0
local landed = {}
for i = 1, #entries, 2 do
  local token = entries[i]
  local value = entries[i + 1]
  local kind, ms, oid = kind_ms_oid(value)
  if ms ~= nil then
    local age = now_ms - ms
    if kind == "void" and age > retention_ms then
      void_removed = void_removed + redis.call("HDEL", attempts, token)
    elseif kind == "landed" and age > retention_ms then
      landed[#landed + 1] = token
      landed[#landed + 1] = oid
      landed[#landed + 1] = value
    end
  end
end

return { next_cursor, void_removed, landed }
