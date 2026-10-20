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

-- create_layer_veto.lua — the orphan sweep's veto on an unmapped object: make
-- it impossible for any `CreateLayer` commit to map a layer to the object
-- from now on, so the sweep can retire the object without the layer ever
-- naming a deleted object.
--
-- Run by the orphan sweep (`storage::orphan_sweep`), on the `{session_id}`
-- slot the object's header names, before it retires an object that no layer
-- mapping names and no overwrite fenced. A `CreateLayer` whose commit point
-- has not run yet may still be about to map the layer to that object; its
-- commit script (`create_layer_commit.lua`) reads the object's commit claim
-- this script writes, in the same slot, and refuses with `VETOED`. Because
-- both scripts run atomically on one slot, exactly one of the two wins
-- whatever the replicas' clocks say: the commit ran first (it claimed the
-- object `committed`, and this script finds the mapping naming the object or
-- keeps that claim), or this script ran first (the commit finds `vetoed` and
-- writes nothing).
--
-- The claim is first-writer-wins and never overwritten: this script writes
-- `vetoed` only when no claim exists. An existing claim — `committed` (a
-- commit mapped the object, and the mapping has since moved away from it) or
-- `vetoed` (an earlier pass) — is kept, and the sweep's retirement continues:
-- with either value no commit can map the object again.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (every key carries the `{session_id}` hash tag — one Cluster slot):
--   [1] layer mapping       (le:{sid}:layer:{hex(layer_name)})   STRING
--   [2] session deletion marker (le:{sid}:sess:deleted)          any type
--       Only its existence is read.
--   [3] commit claim        (le:{sid}:createclaim:{object_id})   STRING
--
-- ARGV:
--   [1] object_id           (decimal; the object to veto)
--   [2] the `vetoed` claim's TTL in milliseconds
--
-- Return value (flat array of bulk strings). Every reply but `VETOED` wrote
-- NOTHING:
--   {"SESSION_DELETED"}        — a `DeleteSession` has run on the session
--       (its deletion marker exists). No commit can map a layer of a deleted
--       session, so the sweep may retire the object.
--   `redis.error_reply("WRONGTYPE_KEY: <key> holds a <type>, expected string
--       (<what>)")` — the mapping or the claim holds the wrong Valkey type.
--       The sweep leaves the object for a later pass.
--   {"MAPPED"}                 — the mapping names the object: its commit
--       landed. The object is live; the sweep leaves it alone.
--   {"CLAIMED", <claim value>} — the object already has a claim, kept as it
--       is: no commit can map the object again, and the sweep may retire it.
--   {"VETOED"}                 — the claim `vetoed` is written with the TTL:
--       no commit of the object can land from now on, and the sweep may
--       retire it.
--
-- The claim is removed by its TTL only, and matches no `SCAN` pattern.
--
-- ── Why it runs on a full store ──────────────────────────────────────────
--
-- The veto must land on a full store, or the sweep could never retire an
-- unmapped object there, and deleting objects is what frees a full store. So
-- step 4 follows the opener rule (SDD §7.5.1): it opens with `PEXPIRE` on the
-- claim key, a write that cannot grow memory and a no-op here, because steps
-- 2 and 3 proved the key absent in this same atomic run, and the deny-oom
-- `SET` then runs. Where the rule is measured and where it is unverified:
-- SDD §21.5. The overshoot is one claim key (under 100 bytes) per vetoed
-- object, kept for its TTL, and the object's deletion follows at once. The
-- opener is placed there and not first: at the top it would rewrite the TTL
-- of an existing claim, which the `CLAIMED` path keeps as it is. On a
-- migrating slot the opener names the absent claim key and is refused before
-- any write. A store that does not follow the rule refuses the veto out of
-- memory; the sweep leaves the object for a later pass.

local mapping_key = KEYS[1]
local deleted_key = KEYS[2]
local claim_key   = KEYS[3]

-- `TYPE` replies with a status table carrying `ok`, and never errors.
local function wrong_type(key, expected)
  local actual = redis.call("TYPE", key)["ok"]
  if actual == expected or actual == "none" then
    return nil
  end
  return actual
end

-- ── 0. A deleted session takes no new layer: no veto is needed. ──────────
if redis.call("EXISTS", deleted_key) == 1 then
  return { "SESSION_DELETED" }
end

-- ── 1. Type checks, before the write. ────────────────────────────────────
local actual = wrong_type(mapping_key, "string")
if actual ~= nil then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. mapping_key .. " holds a " .. actual ..
    ", expected string (layer mapping)")
end
actual = wrong_type(claim_key, "string")
if actual ~= nil then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. claim_key .. " holds a " .. actual ..
    ", expected string (create-layer commit claim)")
end

-- ── 2. Already mapped? Then the commit won; write nothing. ───────────────
if redis.call("GET", mapping_key) == ARGV[1] then
  return { "MAPPED" }
end

-- ── 3. Already claimed? Keep the claim; write nothing. ───────────────────
local claim = redis.call("GET", claim_key)
if claim then
  return { "CLAIMED", claim }
end

-- ── 4. Veto: no commit of this object can land from now on. ──────────────
-- The opener: a no-op `PEXPIRE` on the claim key, proven absent just above,
-- so the deny-oom `SET` after it runs on a full store (see the header).
redis.call("PEXPIRE", claim_key, ARGV[2])
redis.call("SET", claim_key, "vetoed", "PX", ARGV[2])
return { "VETOED" }
