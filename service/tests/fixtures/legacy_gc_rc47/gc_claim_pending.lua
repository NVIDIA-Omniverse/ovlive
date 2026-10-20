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
--
-- ── VENDORED TEST FIXTURE (b.1jt) — never loaded by the service ──────────
--
-- A verbatim copy of `src/storage/scripts/gc_claim_pending.lua` at commit bf32b533
-- (0.2.0-rc.47, the last build before b.yuo's record-aware GC: a record-blind
-- GC with an unmarked leader lock).
-- Every earlier release runs the same code: 0.1.0 and 0.1.1 differ from it
-- only in headers and comments. `tests/integration/storage/b_1jt_rollout_tests.rs`
-- runs it against the store exactly as an older build's replica would, to
-- prove this build's state is safe in a mixed-build store. Only this block was
-- added; everything below it is the older build's file, unchanged.

-- gc_claim_pending.lua — atomic SREM-from-pending + ZADD-to-inflight (b.j9j item 2).
--
-- Replaces the pre-b.j9j `SMOVE ext_pending ext_inflight <hash>` with an
-- equivalent atomic transition that also captures the *time* at which
-- ownership was claimed. The timestamp lets the startup drain
-- (`drain_inflight_to_pending`) distinguish a fresh-still-in-flight
-- sweep (running on another replica right now) from a true crash
-- victim that has been pinned for longer than the leader-lock TTL.
--
-- ── b.j9j item 2 ──────────────────────────────────────────────────────────
--
-- Pre-b.j9j the inflight set was a plain SET. Replica B booting while
-- replica A was mid-`process_pending` for hash H would observe H in
-- inflight, SADD H back to pending via the unconditional drain, and
-- the next leader sweep would SMOVE H to inflight a second time —
-- decrementing the same ref_count twice. The startup drain was not
-- leader-gated and could not be safely leader-gated alone because a
-- newly-booting replica routinely loses the leader race for the
-- first 30 s.
--
-- The fix: inflight becomes a sorted set keyed by claim timestamp,
-- and the drain only re-queues entries whose claim is OLDER than the
-- leader-lock TTL (`LEADER_LOCK_TTL_MS = 30_000`). A fresh claim by a
-- still-alive sweep is younger than the TTL and is left alone; a
-- crash-victim claim from a previous boot is older than the TTL and
-- gets re-queued. The leader lock's own TTL is the natural watermark:
-- if a sweep was running when its replica died, the leader lock has
-- expired by now too, so any in-flight entry older than the lock TTL
-- is unambiguously a crash victim.
--
-- ── b.u96 item 7 — Valkey-side TIME ───────────────────────────────────────
--
-- Pre-b.u96 the millisecond timestamp was caller-provided (passed
-- through ARGV[2] from `SystemTime::now()` in the Rust caller). That
-- exposed the protocol to NTP skew: a replica running
-- clock-behind would stamp claims its peers' drain treated as stale,
-- triggering the very double-decrement scenario the ZSET timestamp
-- was meant to prevent. b.u96 item 7 migrated the score to
-- `redis.call("TIME")` so the entire fleet agrees on the watermark
-- regardless of local-machine clock skew. Valkey's TIME is sampled
-- once per script call and is consistent across replicas reading the
-- same Valkey instance.
--
-- Replication note: since Valkey 7.0 (and Redis 7.0) all Lua scripts
-- default to *effects replication* (a.k.a. `redis.replicate_commands`),
-- so `redis.call("TIME")` is safely deterministic from the replica's
-- point of view — the master records the resulting ZADD with the
-- master-side timestamp, and the replica applies that ZADD verbatim
-- without re-running the TIME sample. Verified against Valkey 9.0.3
-- (the deployed image).
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  ext_pending     (le:ext:{gc}:pending  — SET of release members)
--   [2]  ext_inflight    (le:ext:{gc}:inflight — ZSET of release members
--                         with score = millisecond Valkey timestamp at claim)
--
-- ARGV:
--   [1]  member          (the candidate release member to claim)
--
-- b.qpf: a member is `{hash_hex}:{object_id}:{event}` (or a legacy bare
-- `{hash_hex}` from a pre-b.qpf replica). This script is deliberately
-- MEMBER-OPAQUE — it moves the string it is given and never parses it — which
-- is why the event-identity change needed no edit here. Only the Rust caller
-- (`external_values::pending_member_hash_hex`) splits the hash back out, to
-- name the `ext_meta` / `ext_data` keys the decrement acts on.
--
-- Both KEYS share the `{gc}` Cluster slot so the script is slot-safe
-- on a Valkey Cluster deployment.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Integer:
--   1  — the hash was in `pending`, has been SREM'd and ZADD'd to
--        `inflight` with score = Valkey-side now_ms. The caller owns
--        the hash.
--   0  — the hash was NOT in `pending` (already claimed by another
--        worker, or already drained). The caller skips this candidate.

local pending_key  = KEYS[1]
local inflight_key = KEYS[2]
local member       = ARGV[1]

local removed = redis.call("SREM", pending_key, member)
if removed == 0 then
  return 0
end

-- b.u96 item 7: sample the Valkey-side clock instead of trusting
-- caller-supplied ARGV. `TIME` returns `{seconds, microseconds}` —
-- collapse to millis with integer arithmetic so the ZSET score stays
-- a clean monotonic integer.
local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

-- Atomic within the script: SREM landed, so this ZADD pins ownership
-- with the timestamp captured by Valkey. The startup drain reads the
-- score back to decide whether to re-queue.
redis.call("ZADD", inflight_key, now_ms, member)
return 1
