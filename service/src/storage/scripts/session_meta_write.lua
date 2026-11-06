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

-- session_meta_write.lua — every `CreateSession` metadata write: write the
-- session's metadata hash unless the id carries the deletion marker or a veto.
--
-- Run by `SessionStore::finish_create`, on the `{session_id}` slot, as one
-- `EVALSHA` on the data pool, after the attempt's names commit is known to
-- have run once. `session_meta_veto.lua` writes the veto this script reads,
-- in the same slot, before the `ListSessions` ghost walk removes a names
-- member whose metadata it read absent, and when `CreateSession` settles a
-- write whose reply it did not get. Both scripts run atomically on one slot,
-- so exactly one of the two wins: this write ran first (the veto finds the
-- metadata and answers `PRESENT`), or the veto ran first (this script writes
-- nothing). A late-landing run of this script for a vetoed id — a write fred
-- had sent before its command timeout fired — therefore writes nothing.
--
-- The deletion marker (`le:{sid}:sess:deleted`) is checked first. The id is
-- minted fresh for this call, and `DeleteSession` writes the marker only
-- after reading the record present, so a marker here proves that an earlier
-- send of this same write landed (the data pool re-sends a command whose
-- reply was lost) and another call then deleted the session. Writing now
-- would bring the deleted session back; the caller instead answers the
-- create `OK` with the id and enqueues no checkpoint.
--
-- The field list stays in the caller (`SessionStore::finish_create`), beside
-- the session checkpoint applier's record step, which writes the same four
-- fields.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (every key carries the `{session_id}` hash tag — one Cluster slot):
--   [1] session metadata    (le:{sid}:sess)                     HASH
--   [2] metadata veto       (le:{sid}:sess:veto)                any type
--       Only its existence is read.
--   [3] deletion marker     (le:{sid}:sess:deleted)             any type
--       Only its existence is read.
--
-- ARGV: the metadata's field / value pairs, in order, as `HSET` takes them.
--
-- Return value (flat array of bulk strings):
--   {"DELETED"} — the deletion marker exists: this write landed earlier and
--                 the session was deleted since. Wrote NOTHING.
--   {"VETOED"}  — the veto exists: the id was given up. Wrote NOTHING.
--   {"OK"}      — the metadata is written.
-- The `HSET` is the one write, last, so a refused run (a wrong-typed
-- metadata key, a store write refusal) writes nothing either. No opener, by
-- decision: a full store refuses the `HSET` out of memory, so no session is
-- created past `maxmemory` (`session_meta_veto.lua`, which has one, settles
-- the refused write).

local meta_key = KEYS[1]
local veto_key = KEYS[2]
local deleted_marker_key = KEYS[3]

-- ── 1. A deleted session is never brought back. ──────────────────────────
if redis.call("EXISTS", deleted_marker_key) == 1 then
  return { "DELETED" }
end

-- ── 2. A vetoed id never gets metadata. ──────────────────────────────────
if redis.call("EXISTS", veto_key) == 1 then
  return { "VETOED" }
end

-- ── 3. Write the metadata. ───────────────────────────────────────────────
redis.call("HSET", meta_key, unpack(ARGV))
return { "OK" }
