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

-- leader_acquire_or_refresh.lua — atomic CAS for every leader lock
-- (b.kf1 H-stor-4 follow-up).
--
-- Replaces the original `SET ... NX EX` + `GET` + `SET ... EX` (refresh)
-- pair with a single atomic EVAL. The pre-Lua sequence had a clobber
-- window: between the GET that observed our own lock value and the SET
-- that refreshed the TTL, another replica's `NX` claim could land if our
-- TTL expired in that gap, and our refresh would silently overwrite it.
-- Sub-30 s race, low probability, but eliminating it makes the leader
-- lock a correctness primitive on its own rather than relying on the
-- per-hash atomicity of the C-5 GC script to compensate.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  lock key — `le:checkpoint:leader`
--        (KeySchema::checkpoint_leader_lock) or `le:presence:sweeper:leader`
--        (KeySchema::presence_sweeper_leader_lock). Builds before the GC
--        protocol marker (up to and including 0.2.0-rc.48) also took
--        `le:gc:leader` with this script; this build takes that lock only
--        through `gc_leader_acquire.lua`, which writes a marked value this
--        script's equality check never matches, so an older build only ever
--        claims `le:gc:leader` when it is absent.
--
-- ARGV:
--   [1]  lock value — the caller's `LeaderIdentity::lock_value()`,
--                     `service_label:instance`, unique per process. Stored
--                     as the lock value so the script can recognize "we
--                     already hold it"; a bare service label would let two
--                     processes sharing a label both recognize the lock.
--   [2]  ttl_ms     — lock lifetime in milliseconds, chosen by the caller
--                     (production passes `LEADER_LOCK_TTL_MS` = 30_000,
--                     matching the previous `EX 30` behavior; tests pass
--                     shorter TTLs).
--
-- ── Return value ──────────────────────────────────────────────────────
--
-- Integer: `1` if the caller is now (or remained) the leader,
--          `0` if another process owns the lock.

local current = redis.call("GET", KEYS[1])
if current == false then
  -- Lock is unset: claim it. PX (millisecond TTL) so the caller can pass
  -- whichever resolution is convenient; SDD §11.3 leader semantics are
  -- written against a 30 s lock TTL but the unit is irrelevant to the
  -- script.
  redis.call("SET", KEYS[1], ARGV[1], "PX", tonumber(ARGV[2]))
  return 1
elseif current == ARGV[1] then
  -- We're already the leader: extend the TTL in place. PEXPIRE rather
  -- than re-SET so the value (the lock value) is provably preserved — no
  -- chance of an empty or different value being written if the script
  -- is amended later.
  redis.call("PEXPIRE", KEYS[1], tonumber(ARGV[2]))
  return 1
else
  -- Another process owns it. Do not touch the lock.
  return 0
end
