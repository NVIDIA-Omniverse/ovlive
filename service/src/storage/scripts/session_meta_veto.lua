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

-- session_meta_veto.lua — the veto on a session id whose metadata is absent:
-- make it impossible for any `CreateSession` metadata write to complete the
-- id from now on, unless the metadata is already there.
--
-- Run on the `{session_id}` slot, as one `EVALSHA` on the data pool, by:
--   - the `ListSessions` ghost walk (`SessionStore::retire_ghost_member`),
--     before it removes a names member whose metadata it read absent (it
--     removes the member on `VETOED` and on `DELETED`);
--   - `CreateSession` (`SessionStore::settle_failed_meta_write`), to settle a
--     metadata write whose reply it did not get (on `DELETED` the write
--     landed and the session was deleted since: the create answers `OK`).
-- Every `CreateSession` metadata write goes through `session_meta_write.lua`,
-- which reads the veto this script writes, in the same slot, and refuses with
-- `VETOED`. Because both scripts run atomically on one slot, exactly one of
-- the two wins whatever the replicas' clocks say: the write ran first (this
-- script finds the metadata and answers `PRESENT`), or this script ran first
-- (the write finds the veto and writes nothing). So a walk removes the index
-- members only of an id no create can complete, and a create's late write
-- can no longer land on an id a walk or its own settle has given up.
--
-- With the metadata absent and the session's deletion marker
-- (`le:{sid}:sess:deleted`) present, it writes no veto and answers `DELETED`:
-- the session existed and was deleted (`DeleteSession` marks only an id whose
-- metadata it read present). No metadata write can complete such an id
-- anyway — `session_meta_write.lua` refuses a marked id — so a walk releases
-- its members as on `VETOED`, and a `CreateSession` settle learns that its
-- write landed and the session was deleted since.
--
-- The veto is inert to session recovery and the DR restore: their Step B
-- writes the metadata without reading it (on-access recovery in
-- `session_record_restore.lua`, the restore Job in a transaction).
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (every key carries the `{session_id}` hash tag — one Cluster slot):
--   [1] session metadata    (le:{sid}:sess)                     HASH
--   [2] metadata veto       (le:{sid}:sess:veto)                STRING
--   [3] deletion marker     (le:{sid}:sess:deleted)             any type
--       Only its existence is read.
--
-- ARGV:
--   [1] the veto's TTL in milliseconds
--
-- Return value (flat array of bulk strings). Every reply but `VETOED` wrote
-- NOTHING:
--   `redis.error_reply("WRONGTYPE_KEY: <key> holds a <type>, expected <type>
--       (<what>)")` — the metadata or the veto holds the wrong Valkey type.
--   {"PRESENT"} — the metadata exists: a write for the id landed, or the
--       session was recovered. The caller keeps the id.
--   {"DELETED"} — the metadata is absent and the deletion marker exists: the
--       session was deleted. No veto is written; no metadata write for the
--       id can land (the write script refuses a marked id).
--   {"VETOED"}  — the veto `1` is written (or rewritten, refreshing its TTL)
--       with the TTL: no `session_meta_write.lua` run for the id writes from
--       now on.
--
-- The veto is removed by its TTL only, matches no `SCAN` pattern, and is not
-- checkpointed.
--
-- ── Why it runs on a full store ──────────────────────────────────────────
--
-- The veto must land on a full store. A `CreateSession` whose metadata write
-- the store refused out of memory settles with this veto, and only a `VETOED`
-- lets it release its index members, so without the veto every such create
-- would leave its name held by a ghost; and the `ListSessions` walk could
-- never retire a ghost there either. So step 4 follows the opener rule (SDD
-- §7.5.1): it opens with `PEXPIRE` on the veto key, a write that cannot grow
-- memory, and the deny-oom `SET` then runs. The opener is a no-op: on an
-- absent veto it changes nothing, and on a present one (a repeated veto) it
-- sets the TTL the `SET` that follows sets anyway, as `gc_leader_acquire.lua`'s
-- opener does. Where the rule is measured and where it is unverified: SDD
-- §21.5. The overshoot is one veto key (under 100 bytes) per vetoed id, kept
-- for its TTL. The opener is placed there and not first: at the top it would
-- write on the `PRESENT` path, which writes nothing. On a migrating source an
-- absent veto key is a missing declared key, so the `EVALSHA` is redirected
-- before the script runs (or, inside a run, its first `TYPE` is refused); a
-- present one is still present for the `SET`. The metadata
-- write (`session_meta_write.lua`) has no opener, deliberately, so a full
-- store never gains a session. A store that does not follow the rule refuses
-- the veto out of memory: the walk then keeps the members for a later pass,
-- and the create answers its failure without retrying.

local meta_key = KEYS[1]
local veto_key = KEYS[2]
local deleted_marker_key = KEYS[3]

-- `TYPE` replies with a status table carrying `ok`, and never errors.
local meta_type = redis.call("TYPE", meta_key)["ok"]
local veto_type = redis.call("TYPE", veto_key)["ok"]

-- ── 1. Type checks, before the write. ────────────────────────────────────
if meta_type ~= "hash" and meta_type ~= "none" then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. meta_key .. " holds a " .. meta_type ..
    ", expected hash (session metadata)")
end
if veto_type ~= "string" and veto_type ~= "none" then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. veto_key .. " holds a " .. veto_type ..
    ", expected string (session metadata veto)")
end

-- ── 2. The metadata is there: the id is a session. Write nothing. ────────
if meta_type == "hash" then
  return { "PRESENT" }
end

-- ── 3. The session was deleted: write nothing. ───────────────────────────
if redis.call("EXISTS", deleted_marker_key) == 1 then
  return { "DELETED" }
end

-- ── 4. Veto: no metadata write for this id can land from now on. ─────────
-- The opener: a no-op `PEXPIRE` on the veto key (absent, or given the same
-- TTL the `SET` sets), so the deny-oom `SET` after it runs on a full store
-- (see the header).
redis.call("PEXPIRE", veto_key, ARGV[1])
redis.call("SET", veto_key, "1", "PX", ARGV[1])
return { "VETOED" }
