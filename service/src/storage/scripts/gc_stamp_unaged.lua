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

-- gc_stamp_unaged.lua — stamp an age on un-aged pending release members
-- (b.1jt).
--
-- Every enqueue this build makes (`ext_release_enqueue.lua`) records a
-- member's first-enqueue time in `le:ext:{gc}:pending:age` together with the
-- member, and the GC selection (`gc_select.lua`) collects only members with an
-- age: past the age floor, and clipped below a live read hold. A member in
-- `pending` with no age was enqueued by an older build's bare `SADD`. Its
-- enqueue time is unknown, so collecting it at once could take a blob a
-- reader's catch-up still needs; the selection instead samples such members
-- every pass and the pass stamps them here, with the `{gc}` primary's current
-- time. A stamp is never earlier than the true enqueue, so the age floor and
-- the hold clip, measured from the stamp, are at least as conservative as for
-- a member this build enqueued. Its bytes are not accounted: an older build
-- did not record them, so it counts 0 toward the release budget.
--
-- For each member: only if it is still in `pending`, `ZADD pending:age NX
-- <now_ms> member`. A member claimed or settled since the selection is
-- skipped, and one that gained an age keeps it (`NX`).
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS (both `{gc}`-tagged, so the script is single-slot on a Cluster):
--   [1]  le:ext:{gc}:pending      (SET)
--   [2]  le:ext:{gc}:pending:age  (ZSET)
--
-- ARGV: one or more members.
--
-- Idempotent under a re-run: a stamped member has an age, so `NX` writes
-- nothing the second time.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Integer: the number of members this call stamped.

local pending_key = KEYS[1]
local age_key     = KEYS[2]

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local stamped = 0
for i = 1, #ARGV do
  local member = ARGV[i]
  if redis.call("SISMEMBER", pending_key, member) == 1
      and redis.call("ZADD", age_key, "NX", now_ms, member) == 1 then
    stamped = stamped + 1
  end
end
return stamped
