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

-- create_layer_commit.lua — the commit point of every `CreateLayer`: refuse a
-- session whose deletion marker exists, then — only if the new object's commit
-- claim is absent and the session's layer mapping still holds the value the
-- caller planned against — flip the mapping to the freshly materialized
-- object, add the layer to the session's layers index, claim the object as
-- `committed`, and, when the call carried a client `operation_id`, record that
-- operation, all in one atomic step.
--
-- Run by `LayerStore::create_layer_with_operation` as one `EVALSHA` on a
-- `txn_pool` connection pinned to the `{session_id}` slot; with the same
-- KEYS and ARGV by that call's in-process finisher; and by the orphan sweep's
-- completion of a stranded overwrite, which builds the same KEYS and ARGV
-- from the successor object's header — the operation record key and value
-- included when the call carried an operation id (an object an older build
-- materialized carries none, and its completion passes no record).
-- That pool sends each command exactly once (`max_command_attempts = 1`), so
-- fred never replays this script; a failed or lost reply surfaces as a
-- transient error, and the caller reads the mapping, the claim and (with an
-- operation id) the operation record to learn
-- whether it landed.
--
-- ── The commit claim ─────────────────────────────────────────────────────
--
-- `le:{sid}:createclaim:{new_object_id}` (KEYS[4]) decides the new object's
-- fate once, first writer wins, and is never overwritten:
--
--   `committed` — written HERE, atomically with the flip, by every run that
--       answers `COMMITTED`. No later run maps the object again: it answers
--       `COMMITTED_EARLIER` with the mapping as it is now, whoever moved it
--       since (another create, a `DeleteLayer`). So no object is ever mapped
--       twice, and a caller whose run landed learns it even after the
--       mapping has moved on.
--   `vetoed`    — written by `create_layer_veto.lua` (the orphan sweep) only
--       when no claim exists, before it retires an unmapped object. No run
--       maps the object from then on: `VETOED`.
--
-- The claim is read before the first write, so on a migrating slot (where it
-- is normally absent) the script is refused before it writes anything.
--
-- ── Why one script ───────────────────────────────────────────────────────
--
-- The operation record must be CHECKED and WRITTEN atomically with the mapping
-- flip. A check outside the flip (the caller's `MGET` before allocation) leaves
-- a window in which two calls with the same `operation_id` both see no record
-- and both flip the mapping, so the layer is replaced by its own re-send — the
-- exact outcome the id exists to prevent. Here the second call finds the first
-- call's record and flips nothing.
--
-- The mapping comparison must be atomic with the flip for the same reason: the
-- caller marked the object it planned to replace `superseded` before running
-- this script, and a flip from any other value would replace an object nobody
-- marked — an object a concurrent `UpdateLayer` could still commit to, and
-- whose subscribers nothing would wake. A moved mapping is answered, not
-- overwritten, and the caller re-plans against it. The claim is checked
-- before the comparison, so a caller re-plans only when no run of it landed.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (every key carries the `{session_id}` hash tag — one Cluster slot):
--   [1] layer mapping       (le:{sid}:layer:{hex(layer_name)})    STRING
--   [2] layers index        (le:{sid}:layers)                     SET
--   [3] session deletion marker (le:{sid}:sess:deleted)           any type
--       Only its existence is read.
--   [4] commit claim        (le:{sid}:createclaim:{new_object_id}) STRING
--   [5] operation record    (le:{sid}:createop:{operation_id})    STRING
--       Present only when the call carried an `operation_id`.
--
-- ARGV:
--   [1] new object_id       (decimal; the mapping's new value)
--   [2] layers-index member (hex(layer_name))
--   [3] expected mapping    — the mapping value the caller planned against,
--       encoded so an absent mapping and any stored value cannot be confused:
--       `absent` for no mapping, `value:<stored value>` otherwise.
--   [4] the `committed` claim's TTL in milliseconds
--   [5] operation record value — `<object_id>:<previous_object_id>:<digest>`
--       (only read when KEYS[5] is present)
--   [6] operation record TTL in milliseconds (only read when KEYS[5] is present)
--
-- Return value (flat array of bulk strings). Checked in this order, and every
-- reply but `COMMITTED` wrote NOTHING:
--   {"SESSION_DELETED"}                   — a `DeleteSession` has run on the
--       session (its deletion marker exists). The caller answers `NOT_FOUND
--       "session was deleted"` and leaves its own freshly materialized object
--       as an orphan for the orphan sweep. Checked first, so a deleted session
--       refuses the commit whatever else the keys hold.
--   `redis.error_reply("WRONGTYPE_KEY: <key> holds a <type>, expected <type>
--       (<what>)")` — the mapping, the layers index, the claim or the
--       operation record holds the wrong Valkey type. Raised before any
--       write; the caller refuses the create with the keyspace-type refusal.
--   {"ALREADY_COMMITTED", <record value>} — a call with the same
--       `operation_id` already committed. When the record's object id is the
--       caller's own new object, an earlier run of this same call landed and
--       the caller treats it as committed; otherwise the caller answers that
--       call's object and leaves its own as an orphan for the orphan sweep.
--   {"VETOED"}                            — the claim reads `vetoed` (the
--       orphan sweep is retiring, or has retired, the object as an orphan),
--       or holds a value neither writer writes: the object must never become
--       the layer.
--   {"COMMITTED_EARLIER", <current>}      — the claim reads `committed`: a
--       run of this commit already mapped the object. <current> is the
--       mapping now, encoded as ARGV[3] is; it may name the object, another
--       object, or nothing, since other writers may have moved it.
--   {"MAPPING_CHANGED", <current>}        — no run has mapped the object, and
--       the mapping no longer holds the expected value; <current> is its
--       value, encoded as ARGV[3] is. The caller re-plans against it.
--   {"COMMITTED"}                         — this run flipped the mapping and
--       wrote the `committed` claim.
--
-- The layers-index `SADD` runs on overwrite as well as on a new layer. On an
-- overwrite the member is normally already there and the `SADD` is a no-op.

local mapping_key = KEYS[1]
local layers_key  = KEYS[2]
local deleted_key = KEYS[3]
local claim_key   = KEYS[4]
local record_key  = KEYS[5]

-- `TYPE` replies with a status table carrying `ok`, and never errors.
local function wrong_type(key, expected)
  local actual = redis.call("TYPE", key)["ok"]
  if actual == expected or actual == "none" then
    return nil
  end
  return actual
end

-- ── 0. A deleted session takes no new layer. ─────────────────────────────
-- `DeleteSession` writes the marker in the transaction that removes the
-- session, so a commit that runs after it finds the marker and writes nothing.
if redis.call("EXISTS", deleted_key) == 1 then
  return { "SESSION_DELETED" }
end

-- ── 1. Type checks, before any write. ────────────────────────────────────
local actual = wrong_type(mapping_key, "string")
if actual ~= nil then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. mapping_key .. " holds a " .. actual ..
    ", expected string (layer mapping)")
end
actual = wrong_type(layers_key, "set")
if actual ~= nil then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. layers_key .. " holds a " .. actual ..
    ", expected set (layers index)")
end
actual = wrong_type(claim_key, "string")
if actual ~= nil then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. claim_key .. " holds a " .. actual ..
    ", expected string (create-layer commit claim)")
end
if record_key ~= nil then
  actual = wrong_type(record_key, "string")
  if actual ~= nil then
    return redis.error_reply(
      "WRONGTYPE_KEY: " .. record_key .. " holds a " .. actual ..
      ", expected string (create-layer operation record)")
  end

  -- ── 2. Already committed by a call with this id? Answer it. ───────────
  local recorded = redis.call("GET", record_key)
  if recorded then
    return { "ALREADY_COMMITTED", recorded }
  end
end

-- The mapping as it is now, encoded as ARGV[3] is.
local stored = redis.call("GET", mapping_key)
local current = "absent"
if stored then
  current = "value:" .. stored
end

-- ── 3. Claimed? A committed object is never mapped again, a vetoed one
-- never at all. ─────────────────────────────────────────────────────────────
local claim = redis.call("GET", claim_key)
if claim == "committed" then
  return { "COMMITTED_EARLIER", current }
end
if claim then
  return { "VETOED" }
end

-- ── 4. The mapping must still hold the value the caller planned against. ─
if current ~= ARGV[3] then
  return { "MAPPING_CHANGED", current }
end

-- ── 5. Commit: flip the mapping, list the layer, claim the object, record
-- the operation. ────────────────────────────────────────────────────────────
redis.call("SET", mapping_key, ARGV[1])
redis.call("SADD", layers_key, ARGV[2])
redis.call("SET", claim_key, "committed", "PX", ARGV[4])
if record_key ~= nil then
  redis.call("SET", record_key, ARGV[5], "PX", ARGV[6])
end
return { "COMMITTED" }
