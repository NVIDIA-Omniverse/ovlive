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

-- leader_release_if_held.lua — atomic compare-and-delete of a lease this
-- process holds.
--
-- The third member of the plain lease family, beside
-- `leader_acquire_or_refresh.lua` (claim or extend) and
-- `leader_refresh_if_held.lua` (extend only). Its one caller is the disaster
-- recovery Job's lease (`KeySchema::dr_job_lease`), released when the Job
-- exits — on every exit after the lease was taken: success, a refusal, a
-- failure or SIGTERM — so a replacement Job acquires at once rather than one
-- lease TTL later.
--
-- ── Why compare-and-delete ────────────────────────────────────────────
--
-- A bare `DEL` would remove a lease another process now holds: after this
-- process's lease lapsed (a refresh that failed for a TTL), a second Job may
-- have claimed the key, and deleting it would let a third Job start beside
-- the second. The `GET` and the `DEL` run in one script, so nothing can claim
-- the key between them.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  lease key — `KeySchema::dr_job_lease`.
--
-- ARGV:
--   [1]  lock value — the caller's `LeaderIdentity::lock_value()`; the key is
--                     deleted only when it holds exactly this value.
--
-- ── Return value ──────────────────────────────────────────────────────
--
-- Integer: `1` if the key held the caller's value and was deleted, `0` if it
--          is unset or holds another value. `0` NEVER mutates the key.

if redis.call("GET", KEYS[1]) == ARGV[1] then
  redis.call("DEL", KEYS[1])
  return 1
end
return 0
