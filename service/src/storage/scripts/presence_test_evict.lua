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

-- presence_test_evict.lua — TEST-ONLY: evict one connection the way a sweep
-- does, in one atomic step.
--
-- Loaded only by `storage::presence::evict_connection_for_test`, which is
-- compiled under `cfg(any(test, feature = "test-support"))`; production code
-- never sends it. It gives tests a deterministic eviction that removes the
-- member, deletes its info hash and publishes `left` together, as a sweep
-- script does, instead of racing the live sweeper for it.
--
-- It does NOT check or touch the lease: the caller decides separately whether
-- the lease is absent (typically by deleting it first), so a test can also
-- build the state "evicted while its lease still exists".
--
-- Like a sweep, it acts only on a member: the info hash is deleted and
-- `left` is published (path A) iff the `SREM` removed the member, in this
-- execution. A call for a non-member changes nothing.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS (both carry the `{sid}` hash tag, so they share one slot):
--   [1]  members — `le:{sid}:presence`                (KeySchema::presence_members)
--   [2]  info    — `le:{sid}:presence:{cid}`          (KeySchema::presence_user)
--
-- ARGV:
--   [1]  cid          — the connection id to evict.
--   [2]  channel      — `le:presence:events:<sid>`
--                       (KeySchema::presence_events_channel).
--   [3]  left_payload — the encoded `left` event, built by the caller.
--
-- No other key is read or written.
--
-- ── Return value ──────────────────────────────────────────────────────
--
-- Array `{removed, remaining}`:
--   removed   — `1` if the `SREM` removed `cid` (its info hash was deleted
--               and `left` published), `0` if it was not a member (nothing
--               was written or published).
--   remaining — `SCARD` of the member set after the removal.

local members_key = KEYS[1]
local info_key = KEYS[2]

local cid = ARGV[1]
local channel = ARGV[2]
local left_payload = ARGV[3]

local removed = redis.call("SREM", members_key, cid)
if removed == 1 then
  redis.call("DEL", info_key)
  redis.call("PUBLISH", channel, left_payload)
end

return { removed, redis.call("SCARD", members_key) }
