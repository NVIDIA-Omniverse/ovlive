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

-- leader_refresh_if_held.lua — atomic REFRESH-ONLY extension of a leader
-- lock this replica already holds (b.s5y).
--
-- Sibling of `leader_acquire_or_refresh.lua`, and deliberately NOT the same
-- script. The acquire variant claims an unset key; this one never does.
--
-- ── Why refresh must never re-acquire ─────────────────────────────────
--
-- The caller is a long-running tick that is ALREADY mutating shared
-- durable state under the lock (the checkpoint loop's per-layer walk).
-- It calls this mid-walk to prove it still holds the lease. If the key
-- has vanished — the TTL lapsed while the tick was inside a slow adapter
-- call — then during that gap another replica may have acquired the lock
-- and may right now be ticking the same layers. Claiming the key here
-- would return "still held" and let this replica keep mutating, i.e. it
-- would masquerade a lapse as continuity and hide exactly the
-- duplicate-leadership window the caller is checking for. The truthful
-- answer to "am I still the same uninterrupted leader?" after a lapse is
-- `0`, and the caller's contract is to abort the walk on `0`.
--
-- A fresh acquisition is a separate, legitimate act — it happens at the
-- top of the NEXT tick via `leader_acquire_or_refresh.lua`, where the
-- caller has no in-flight mutation state to carry across the gap.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  lock key — e.g. `le:checkpoint:leader`
--        (see `KeySchema::checkpoint_leader_lock`).
--
-- ARGV:
--   [1]  lock value — the caller's `LeaderIdentity::lock_value()`,
--                     `service_label:instance`, unique per process; must
--                     equal the stored value for the refresh to happen.
--   [2]  ttl_ms     — new lock lifetime in milliseconds, chosen by the
--                     caller (production passes `LEADER_LOCK_TTL_MS`;
--                     tests pass shorter TTLs).
--
-- ── Return value ──────────────────────────────────────────────────────
--
-- Integer: `1` if the lock was held by the caller and its TTL was
--          extended, `0` if the key is unset (lapsed) or owned by
--          another process. `0` NEVER mutates the key.

local current = redis.call("GET", KEYS[1])
if current == ARGV[1] then
  -- Still ours: extend in place. PEXPIRE rather than re-SET so the value
  -- (the lock value) is provably preserved — this script can never author a
  -- lock value, only prolong one.
  redis.call("PEXPIRE", KEYS[1], tonumber(ARGV[2]))
  return 1
else
  -- Unset (lapsed) or owned by someone else. Both are "not continuously
  -- held" and both return 0 without touching the key. Collapsing the two
  -- cases is intentional: from the caller's perspective they are the same
  -- verdict — it can no longer prove exclusivity — and distinguishing
  -- them would invite a caller to treat the lapsed case as recoverable.
  return 0
end
