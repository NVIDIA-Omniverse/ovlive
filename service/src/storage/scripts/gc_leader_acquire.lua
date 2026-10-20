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

-- gc_leader_acquire.lua — acquire, refresh or take over the protocol-marked
-- GC leader lock `le:gc:leader` (b.1jt).
--
-- `le:gc:leader` is the one lease the GC sweep, the orphan sweep and the
-- reference-count reconciler share. This build writes a MARKED value,
-- `le-gc/<v>|<service_label>:<instance>` (`LeaderIdentity::gc_lock_value`);
-- every build before the marker (0.1.0 through 0.2.0-rc.48) writes a bare
-- value and runs a GC that acts on this build's state in shapes this build no
-- longer uses: it drains bare-scored claims back to `pending` and claims SET
-- retirement carriers, and 0.1.0 through 0.2.0-rc.47 also decrement
-- record-blind (0.2.0-rc.48 has the replay record). So a lock with no marker,
-- or a lower version, is an older build's, and this script takes it over; an
-- older build's own acquire (`leader_acquire_or_refresh.lua`) only claims an
-- absent key, and its refresh is an equality CAS that never matches a marked
-- value, so no older GC, orphan sweep or reconciler starts a pass while the
-- key holds a marked value.
--
-- The key never lapses while any process of this build with a running GC
-- loop that is making progress can reach the store. Every such process calls
-- this once a second (`GC_PASS_INTERVAL`), leader or not, from a refresh
-- ticker that runs beside its GC loop independent of any pass, stops with
-- the loop, and pauses while the loop has made no progress for
-- `GC_PROGRESS_STALL_AFTER` (20 s; `spawn_gc_task`), and again before each
-- GC, orphan-sweep, reconciler and backup-repair pass. So the leader refreshes the key
-- mid-pass, however long a progressing pass runs, and a non-leader takes a
-- same-version peer's lock over once its TTL is at or below
-- `takeover_below_ms` (a third of the TTL: the holder has missed refreshes
-- for two thirds of it — it stopped, crashed, lost the store, or its GC loop
-- died or stalled), or a marked lock of any version that has no TTL at all.
-- The progress gate lives in the process; this script and the value it
-- writes are unchanged by it. The only lapse is every such process losing
-- the store for the TTL (or every one's loop dying or stalling); an older
-- build, unmarked or a lower marked version, can then acquire the key and
-- start a pass, until a process of this build reaches the store again and
-- takes it back. And an older pass already running when this build took the
-- key over runs on until it ends on its own. Safety in both windows rests on
-- what this build leaves in the store, not on the lock: `pending` holds only
-- obligations no decrement was attempted for, the enqueue refuses a replay of
-- a member already claimed, settled or accounted, and retirement carriers are
-- HASHes an older carrier claim cannot read. Their exposure is an older pass
-- that ignores this build's read holds (a catch-up can fail `INTERNAL`),
-- never a double decrement.
--
-- Two marked leaders can briefly overlap after an expiring takeover (a holder
-- whose refreshes were only stalled); every GC, orphan-sweep, reconciler and
-- backup-repair step is safe under two leaders by construction, so the lock only sheds load
-- between marked builds. A higher version is never taken over while it has a
-- TTL, so this build does not lead across a newer GC protocol while the newer
-- holder keeps refreshing; a newer holder that stops refreshing — every newer
-- process stopped or cut off from the store — lapses within the TTL, and this
-- build then acquires the absent key, exactly as an unmarked build can
-- acquire this build's. So at every version bump the lapse window's safety
-- rests on store state (as above), not on the lock.
--
-- `le:checkpoint:leader` and `le:presence:sweeper:leader` do NOT use this
-- script: they stay on the plain CAS, because the checkpoint lock is a
-- correctness lock a takeover would break and the presence sweeper renews
-- mid-pass through `LeaderLease` (SDD §11.3).
--
-- ── Verdicts, in the order they are checked ──────────────────────────────
--
--   * `acquired`           — the key was absent: SET value PX ttl.
--   * `refreshed`          — the value is ours: PEXPIRE ttl.
--   * `took_over_older`    — the value has no marker (it does not start with
--                            the prefix followed by a decimal version and `|`),
--                            or a version below ours: SET value PX ttl.
--   * `took_over_expiring` — a marked peer with no TTL, whatever its version;
--                            or a same-version peer whose PTTL is at or below
--                            `takeover_below_ms`: SET value PX ttl.
--   * `held`               — a higher-version peer with a TTL, whatever its
--                            PTTL, or a same-version peer whose PTTL is above
--                            `takeover_below_ms`: nothing written.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:gc:leader  (KeySchema::gc_leader_lock; deliberately not
--                       hash-tagged)
--
-- ARGV:
--   [1]  value              — this process's marked lock value
--   [2]  ttl_ms             — lock lifetime, ms (production `LEADER_LOCK_TTL_MS`)
--   [3]  marker_prefix      — the literal opening every marked value
--                             (`GC_LEADER_MARKER_PREFIX`, "le-gc/")
--   [4]  version            — this build's protocol version
--                             (`GC_LEADER_PROTOCOL_VERSION`), a decimal
--   [5]  takeover_below_ms  — a same-version peer at or below this PTTL is
--                             taken over (`GC_LEADER_TAKEOVER_BELOW_MS`)
--
-- A re-run (a fred re-send) after `acquired` or a takeover finds its own value
-- and answers `refreshed`: harmless.
--
-- ── The out-of-memory opener (b.pa1) ──────────────────────────────────────
--
-- `le:gc:leader` is untagged, so it sits on its own slot, and on a store over
-- `maxmemory` Valkey refuses a script's command that can grow memory (`SET`)
-- only while the script has written nothing yet; once any write command has
-- run, every later one runs (`gc_decrement_and_collect.lua`, "The
-- out-of-memory opener"). The refresh's `PEXPIRE` cannot grow memory, so it
-- always ran; an acquisition or a takeover, whose first write was `SET`, was
-- refused, so with the leader gone no replica could lead the GC until an
-- operator freed memory. So every `SET … PX` below is preceded by
-- `PEXPIRE key ttl_ms`, the opener: it cannot grow memory, changes nothing on
-- an absent key, and on a present key only sets a TTL the `SET` that follows
-- replaces with the value. The verdicts and the stored result are what they
-- were without it. What a run can add past `maxmemory` is this one key.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- `{ verdict, displaced }`: `displaced` is the value found in the key for
-- `took_over_older`, `took_over_expiring` and `held`, and "" otherwise.

local key = KEYS[1]
local value = ARGV[1]
local ttl_ms = tonumber(ARGV[2])
local prefix = ARGV[3]
local version = tonumber(ARGV[4])
local takeover_below_ms = tonumber(ARGV[5])
if value == nil or value == "" or ttl_ms == nil or prefix == nil or prefix == ""
    or version == nil or takeover_below_ms == nil then
  return redis.error_reply("ENCODING: gc_leader_acquire expects value, ttl_ms, marker prefix, version, takeover_below_ms")
end

-- Write this process's value, after the opener (see "The out-of-memory
-- opener" above), so the `SET` runs on a store over `maxmemory`. Called after
-- every read the verdict needs, the `PTTL` included.
local function claim()
  redis.call("PEXPIRE", key, ttl_ms)
  redis.call("SET", key, value, "PX", ttl_ms)
end

local current = redis.call("GET", key)
if current == false then
  claim()
  return { "acquired", "" }
end
if current == value then
  redis.call("PEXPIRE", key, ttl_ms)
  return { "refreshed", "" }
end

-- The marker: the prefix, then a decimal version, then `|`. Compared with
-- `string.sub` so the prefix needs no pattern escaping.
local their_version = nil
if string.sub(current, 1, #prefix) == prefix then
  local digits = string.match(string.sub(current, #prefix + 1), "^(%d+)|")
  if digits then their_version = tonumber(digits) end
end
if their_version == nil or their_version < version then
  claim()
  return { "took_over_older", current }
end

-- A marked value with no TTL would never lapse, so it is taken over whatever
-- its version. A higher version is otherwise left alone at any TTL, so this
-- build does not take the lock from a newer GC protocol's live holder; a newer
-- holder that stops refreshing lapses within its TTL, so nothing deadlocks
-- (and this build can then acquire the absent key — see the header).
local pttl = redis.call("PTTL", key)
if pttl == -1 then
  claim()
  return { "took_over_expiring", current }
end
if their_version > version then
  return { "held", current }
end
if pttl >= 0 and pttl <= takeover_below_ms then
  claim()
  return { "took_over_expiring", current }
end
return { "held", current }
