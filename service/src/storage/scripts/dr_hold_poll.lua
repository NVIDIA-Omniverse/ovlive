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

-- dr_hold_poll.lua — the one read behind the hold poll (SDD SR-2.2, every
-- `LIVE_EDIT_DR_HOLD_POLL_INTERVAL_MS`) and the startup hold evaluation
-- (`storage::dr_hold::read_hold_marker`, `crate::dr::hold`).
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS[1]  the hold marker, `{prefix}dr:held:{<counter key>}`
--          (`KeySchema::dr_hold_marker`). Its hash tag is the whole
--          object-id counter key, so the script runs on the counter's slot.
-- ARGV     none.
--
-- Returns a six-element array:
--   [1] 1 when the marker exists, else 0;
--   [2] `incident`, [3] `reason`, [4] `evidence`, [5] `detected_at_ms`,
--   [6] `detected_by` — each the field's value, or nil when the field is
--       absent (always nil when [1] is 0).
--
-- The hold predicate in this release is "the marker exists": [1] decides it,
-- and a marker missing some fields still holds. A marker key of the wrong
-- type makes `HMGET` fail the script; the caller treats any error as "no
-- verdict" (SR-2.7), never as "not held".
--
-- Read-only. The full predicate (SR-2.1) belongs in this same script, on this
-- same slot: the counter and counter-slot sentinel reads, and the `HSETNX` of
-- a fresh incident (SR-4.3), are not here yet.

if #KEYS ~= 1 then
  return redis.error_reply(
    "ENCODING: dr_hold_poll expects 1 key, got " .. #KEYS)
end

local marker = KEYS[1]
local present = redis.call("EXISTS", marker)
local f = redis.call(
  "HMGET", marker, "incident", "reason", "evidence", "detected_at_ms", "detected_by")
return { present, f[1], f[2], f[3], f[4], f[5] }
