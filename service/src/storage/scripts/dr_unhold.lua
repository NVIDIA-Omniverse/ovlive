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

-- dr_unhold.lua — the un-hold: the restore or accept-loss Job's last step,
-- and the hold marker's only deleter (SDD SR-4.2, SR-9.9;
-- `storage::dr_hold::unhold`).
--
-- In one atomic step on the object-id counter's slot it checks that the
-- calling Job still holds its lease and that the store is still held under
-- the Job's incident, raises the object-id counter to the Job's floor, and
-- deletes the marker. Every read comes before any write, so a refusal writes
-- nothing.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS[1]  the hold marker (`KeySchema::dr_hold_marker`);
-- KEYS[2]  the Job lease (`KeySchema::dr_job_lease`);
-- KEYS[3]  the object-id counter (`KeySchema::counter_object_id`).
--          All three hash to the counter's slot: the marker and the lease are
--          tagged with the whole counter key.
-- ARGV[1]  the Job's incident;
-- ARGV[2]  the Job's lease value (`LeaderIdentity::lock_value`);
-- ARGV[3]  the counter floor, a decimal integer (0: no floor).
--
-- Returns a three-element array `{ outcome, a, b }`:
--   `lease_lost`      the lease is not the caller's: `a` is its value ("" when
--                     absent). Nothing written.
--   `counter_unreadable` the counter holds a string that is not a decimal
--                     integer: `a` is that string. Nothing written.
--   `other_incident`  the marker exists under another incident: `a` is that
--                     incident ("" when the field is absent). Nothing written.
--   `unheld`          the counter is now `b` (it was `a`) and the marker is
--                     deleted.
--   `already_unheld`  the marker was already absent; the counter is now `b`
--                     (it was `a`). Usually the caller's own earlier attempt
--                     whose reply was lost (a re-send of this call); a marker
--                     someone removed by hand answers the same.
-- `a` and `b` of the last two are decimal texts; an absent counter reads "0".
-- A key of the wrong type (the lease, the counter or the marker) fails the
-- script with the server's `WRONGTYPE` reply, before any write.
--
-- The counter is only ever raised: it is set to the floor when the floor is
-- higher than its value, and left alone otherwise.
--
-- ── Slot migrations (b.cyu) ───────────────────────────────────────────────
--
-- Every key is read before the first write, so on a migrating source a key
-- that moved is refused before anything is written. The writes are the
-- counter's `SET` and then the marker's `DEL`, which names a key the opening
-- `EXISTS` proved present.

if #KEYS ~= 3 then
  return redis.error_reply("ENCODING: dr_unhold expects 3 keys, got " .. #KEYS)
end
local floor = ARGV[3]
if type(floor) ~= "string" or not string.match(floor, "^[0-9]+$") then
  return redis.error_reply(
    "ENCODING: dr_unhold expects a decimal floor, got " .. tostring(floor))
end

local marker_key = KEYS[1]
local lease_key = KEYS[2]
local counter_key = KEYS[3]

local holder = redis.call("GET", lease_key)
if holder ~= ARGV[2] then
  return { "lease_lost", holder or "", "" }
end

local current = redis.call("GET", counter_key)
if not current then
  current = "0"
end
if not string.match(current, "^[0-9]+$") then
  return { "counter_unreadable", current, "" }
end

local held = redis.call("EXISTS", marker_key) == 1
if held then
  local incident = redis.call("HGET", marker_key, "incident")
  if incident ~= ARGV[1] then
    return { "other_incident", incident or "", "" }
  end
end

-- Decimal texts compared by length and then lexically, after leading zeros,
-- so a value past a Lua double's exact range still compares exactly.
local function above(a, b)
  a = string.gsub(a, "^0+", "")
  b = string.gsub(b, "^0+", "")
  if string.len(a) ~= string.len(b) then
    return string.len(a) > string.len(b)
  end
  return a > b
end

local new = current
if above(floor, current) then
  redis.call("SET", counter_key, floor)
  new = floor
end
if held then
  redis.call("DEL", marker_key)
  return { "unheld", current, new }
end
return { "already_unheld", current, new }
