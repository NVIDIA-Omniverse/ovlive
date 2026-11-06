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

-- ext_ref_count_set.lua — set one external value's `ref_count` to the
-- restore Job's authoritative recount (SDD SR-9.8;
-- `storage::ext_reconcile::recount_authoritative`).
--
-- The recount runs only while the fleet is held and after the Job's quiet
-- period and last write, so nothing else writes a count while it runs. It
-- walks every object header, retirement carrier and owed release, and hands
-- this script, per hash, the exact count those name: references plus owed.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS[1]  the value's meta, `le:ext:meta:{hash_hex}` (`KeySchema::ext_meta`).
--          The data key is never named: the script reads and writes the meta
--          alone.
-- ARGV[1]  the target count, a decimal integer of at least 1. Anything else
--          is refused with an error before any command runs.
--
-- Returns a two-element array `{ outcome, previous }`:
--   `created`    the meta was absent: it is created with `ref_count` = target
--                and nothing else. The value's bytes are then in no Valkey key
--                (a meta and its data share the `{hash_hex}` slot and are
--                written and deleted together, so an absent meta means absent
--                data): the meta-without-data shape `storage::ext_reconcile`'s
--                module header audits. `previous` is "".
--   `raised`     `ref_count` was a count below the target; set to it.
--   `lowered`    `ref_count` was a count above the target; set to it.
--   `unchanged`  `ref_count` already equalled the target; nothing written.
--   `repaired`   the meta had no `ref_count`, or one that is not a decimal
--                integer; set to the target. `previous` is what it held ("" when
--                absent).
--   `wrong_type` the meta key is not a hash; nothing written. `previous` is the
--                key's type. The caller treats it as a corrupt record.
-- For the other outcomes `previous` is the `ref_count` text before the call.
--
-- It never deletes a meta, and never touches `last_store_ms`, `backed_up_in`
-- or `backup_cleared_ms`: only `ref_count` is written.
--
-- ── Slot migrations (b.cyu) ───────────────────────────────────────────────
--
-- It names one key. The opening `TYPE` names it, so a meta that moved is
-- refused before any write; the one write, an `HSET`, is the last command.

if #KEYS ~= 1 then
  return redis.error_reply(
    "ENCODING: ext_ref_count_set expects 1 key, got " .. #KEYS)
end
local target = ARGV[1]
if type(target) ~= "string" or not string.match(target, "^[1-9][0-9]*$") then
  return redis.error_reply(
    "ENCODING: ext_ref_count_set expects a decimal target of at least 1, got "
      .. tostring(target))
end

local meta_key = KEYS[1]

-- `TYPE` is a status reply, which reaches Lua as `{ ok = "<type>" }`.
local kind = redis.call("TYPE", meta_key).ok
if kind == "none" then
  redis.call("HSET", meta_key, "ref_count", target)
  return { "created", "" }
end
if kind ~= "hash" then
  return { "wrong_type", kind }
end

local current = redis.call("HGET", meta_key, "ref_count")
if not current then
  redis.call("HSET", meta_key, "ref_count", target)
  return { "repaired", "" }
end
-- A count is a decimal integer: an optional sign, then digits. Compared as
-- text by length and then lexically, after leading zeros, so a count past a
-- Lua double's exact range still compares exactly.
local sign, digits = string.match(current, "^(%-?)([0-9]+)$")
if not digits then
  redis.call("HSET", meta_key, "ref_count", target)
  return { "repaired", current }
end
digits = string.gsub(digits, "^0+", "")
local outcome
if sign == "-" and digits ~= "" then
  outcome = "raised"
elseif digits == target then
  return { "unchanged", current }
elseif string.len(digits) < string.len(target)
    or (string.len(digits) == string.len(target) and digits < target) then
  outcome = "raised"
else
  outcome = "lowered"
end
redis.call("HSET", meta_key, "ref_count", target)
return { outcome, current }
