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

-- session_record_restore.lua — on-access session recovery's record step: write
-- a session's layer mappings, layers index and metadata record from its
-- backup, but only while neither the record nor the session's deletion marker
-- exists.
--
-- Run by `checkpoint::session::apply_session_checkpoint` in its on-access
-- mode (`SessionApplySteps::PathIndexThenRecordIfAbsent`), on the
-- `{session_id}` slot, as one `EVALSHA` on the data pool, after Step A. The
-- disaster-recovery restore Job keeps its unconditional `MULTI`/`EXEC` record
-- step instead: the fleet is held, so it is the only writer.
--
-- Why conditional: a recovery must never overwrite a write another call made
-- to a session that already exists. The record is the completeness marker. A
-- re-send of this script after a run that landed (the data pool re-sends a
-- command whose reply was lost), or a second replica restoring the same
-- session, finds the record and writes nothing, so a `DeleteLayer` or
-- `CreateLayer` that landed in between keeps its change. A session deleted in
-- the meantime carries the deletion marker (`SessionStore::delete` writes it
-- atomically with the record's `DEL`), so it is not brought back.
--
-- It does not read the metadata veto (`le:{sid}:sess:veto`): a recovery still
-- completes a vetoed id.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (every key carries the `{session_id}` hash tag — one Cluster slot):
--   [1]    session metadata record (le:{sid}:sess)                 HASH
--   [2]    deletion marker         (le:{sid}:sess:deleted)         any type
--          Only its existence is read.
--   [3]    layers index            (le:{sid}:layers)               SET
--          Declared even when the backup names no layer.
--   [4..]  one layer mapping per layer the backup maps
--          (le:{sid}:layer:{hex(name)})                            STRING
--
-- ARGV:
--   [1..8]  the record's four field / value pairs, in order, as `HSET` takes
--           them (`name`, `base_path`, `created_by`, `created_at`; the field
--           list stays in the caller, beside `SessionStore::finish_create`'s)
--   [9 .. 8+M]  the object id of each mapping, in `KEYS[4..]`'s order
--           (M = #KEYS - 3)
--   [9+M ..]    the layers index's members, hex-encoded layer names
--
-- Return value (flat array of bulk strings):
--   {"PRESENT"} — the record exists: the session is already there. Wrote
--       NOTHING.
--   {"DELETED"} — the record is absent and the deletion marker exists: the
--       session was deleted. Wrote NOTHING.
--   {"APPLIED"} — every mapping, every layers-index member and the record are
--       written.
--   `redis.error_reply("BAD_ARGS: ...")` — the `ARGV` is shorter than the
--       record's pairs plus one object id per mapping key. Wrote NOTHING.
--
-- ── Write order ──────────────────────────────────────────────────────────
--
-- Mappings, then the layers index, then the record `HSET` LAST. A run cut
-- part-way (a write refused out of memory, a wrong-typed key, a slot
-- migration) leaves no record, so the `PRESENT` branch never trusts an
-- incomplete restore, and a later recovery writes it all again: every write is
-- idempotent per key. On a migrating source the script runs only with every
-- declared key present, the record included, and then answers `PRESENT`
-- without writing.

local record_key = KEYS[1]
local marker_key = KEYS[2]
local layers_index_key = KEYS[3]
local FIRST_MAPPING_KEY = 4
local RECORD_ARGC = 8
local mapping_count = #KEYS - (FIRST_MAPPING_KEY - 1)

if #ARGV < RECORD_ARGC + mapping_count then
  return redis.error_reply(
    "BAD_ARGS: session_record_restore needs " .. (RECORD_ARGC + mapping_count) ..
    " ARGV at least, got " .. #ARGV)
end

-- ── 1. The session is there: write nothing. ──────────────────────────────
if redis.call("EXISTS", record_key) == 1 then
  return { "PRESENT" }
end

-- ── 2. The session was deleted: write nothing. ───────────────────────────
if redis.call("EXISTS", marker_key) == 1 then
  return { "DELETED" }
end

-- ── 3. The layer mappings. ───────────────────────────────────────────────
for i = 1, mapping_count do
  redis.call("SET", KEYS[FIRST_MAPPING_KEY - 1 + i], ARGV[RECORD_ARGC + i])
end

-- ── 4. The layers index, one member per call (no `unpack` limit). ────────
for i = RECORD_ARGC + mapping_count + 1, #ARGV do
  redis.call("SADD", layers_index_key, ARGV[i])
end

-- ── 5. The record, last: its presence means the restore is complete. ─────
redis.call("HSET", record_key,
  ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5], ARGV[6], ARGV[7], ARGV[8])
return { "APPLIED" }
