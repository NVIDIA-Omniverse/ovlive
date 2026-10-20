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

-- gc_claim_pending.lua — atomic SREM-from-pending + ZADD-to-inflight (b.j9j item 2).
--
-- Replaces the pre-b.j9j `SMOVE ext_pending ext_inflight <hash>` with an
-- equivalent atomic transition that also captures the *time* at which
-- ownership was claimed. The claim time is what lets a GC pass tell a claim a
-- sweep is still working on (younger than the re-claim age D) from one a
-- crash, a failed step or a `stale` decrement left behind (older than D),
-- which the pass then re-claims in place (`gc_reclaim_stale.lua`).
--
-- ── The encoded score (b.1jt) ─────────────────────────────────────────────
--
-- The `inflight` score is the claim time PLUS `CLAIM_SCORE_OFFSET_MS`
-- (10^13, below). Every older build scored its claims
-- with the bare claim time and runs a drain that moves claims scored in
-- `[0, now − 30 s]` back to `pending`; an encoded score is never in that
-- range, so no older build re-queues a member this build claimed — and
-- perhaps already decremented — for its GC (record-blind in 0.1.0 through
-- 0.2.0-rc.47) to decrement again.
-- An older reconciler `ZSCAN`s `inflight` without reading the score, so it
-- still counts the claim as owed. The offset is positive so an encoded claim
-- sorts after every bare one, and an older build's oldest-claim read
-- (`ZRANGE inflight 0 0`) still finds its own. A score below the offset is an
-- older build's claim: this build never decrements one (it drops it past the
-- fence, `gc_drop_foreign_claims.lua`).
--
-- The offset is a literal here, not an argument, so the script still reads
-- only ARGV[1] (b.u96 item 7, below). It MUST equal the Rust constant
-- `external_values::GC_CLAIM_SCORE_OFFSET_MS`, which every reader of the score
-- (`gc_select.lua`, `gc_reclaim_stale.lua`, `gc_drop_foreign_claims.lua`)
-- receives as an argument: a mismatch would misread this build's claims as an
-- older build's, or the reverse.
--
-- The score is handed to `ZADD` as a Lua number: Valkey converts it with full
-- precision, while Lua's own string conversion keeps only 14 significant
-- digits. The REPLY is the unencoded claim time, which the decrement's
-- claim-age fence checks and its replay record keeps.
--
-- A late copy of a decrement under an old claim (fred re-sends a command after
-- its caller gave up, and a sweeper can pause between its claim and its
-- decrement) is what `gc_decrement_and_collect.lua`'s claim-age fence and
-- replay record answer: a late copy is refused `stale`, and a re-run of a
-- member whose decrement already applied answers `replayed` without
-- decrementing. The fence is shorter than D, so by the time a claim is
-- re-claimed no copy under it can still act.
--
-- ── b.u96 item 7 — Valkey-side TIME ───────────────────────────────────────
--
-- Pre-b.u96 the millisecond timestamp was caller-provided (passed
-- from `SystemTime::now()` in the Rust caller). That
-- exposed the protocol to NTP skew: a replica running
-- clock-behind would stamp claims its peers treated as stale. b.u96 item 7
-- migrated the score to
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
-- ── b.yuo — the release ledger, and the claim time as the reply ───────────
--
-- The claim also maintains the release ledger beside the pending Set
-- (`ext_release_enqueue.lua` carries its full description):
--
--   * on a claim (SREM returned 1): `ZREM pending:age member` (the member is no
--     longer waiting), and `HSETNX pending:bytes member 0`. The `HSETNX` is the
--     ledger trail: every member this build may go on to decrement has a
--     ledger field until it settles, even one an older build enqueued with a
--     bare `SADD`. That is what lets `gc_ledger_audit.lua` find — and settle —
--     a member an older build later completed with a bare `ZREM`, so its
--     replay record still gets torn down. A field already present keeps its
--     bytes;
--   * the reply is the claim time in ms (the `{gc}` primary's `TIME`) instead
--     of `1`. The caller hands it to `gc_decrement_and_collect.lua`, whose
--     claim-age fence refuses a decrement executing more than the fence after
--     this instant, and which records it as the member's replay-record score.
--
-- **Self-heal (SREM returned 0 and the member is not in `inflight`).** The
-- member has left both sets, yet a caller selected it — its age entry or its
-- ledger field outlived it (an older build's claim does not `ZREM` the age, and
-- its completion is a bare `ZREM inflight` that leaves the ledger field). Clean
-- up after it: `ZREM pending:age member`; and if `HDEL pending:bytes member`
-- returned 1, `DECRBY pending:total` by the value it held. If either removed
-- anything, `ZADD settled <now_ms> member`, so its replay record (if
-- this build ever wrote one) is torn down after the retention like any other.
-- Reply 0: there is nothing to claim. A member that is still in `inflight` is a
-- peer's live claim and is left alone.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS (all `{gc}`-tagged, so the script is single-slot on a Cluster):
--   [1]  ext_pending     (le:ext:{gc}:pending  — SET of release members)
--   [2]  ext_inflight    (le:ext:{gc}:inflight — ZSET of release members;
--                         this build's score = claim ms +
--                         `CLAIM_SCORE_OFFSET_MS`)
--   [3]  pending:age     (le:ext:{gc}:pending:age   — ZSET)
--   [4]  pending:bytes   (le:ext:{gc}:pending:bytes — HASH)
--   [5]  pending:total   (le:ext:{gc}:pending:total — STRING)
--   [6]  settled         (le:ext:{gc}:settled       — ZSET)
--
-- ARGV:
--   [1]  member          (the candidate release member to claim)
--
-- b.qpf: a member is `{hash_hex}:{object_id}:{event}` (or a legacy bare
-- `{hash_hex}` from a pre-b.qpf replica). This script is deliberately
-- MEMBER-OPAQUE — it moves the string it is given and never parses it. Only
-- the Rust caller (`external_values::pending_member_hash_hex`) splits the hash
-- back out, to name the `ext_meta` / `ext_data` keys the decrement acts on.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Integer:
--   > 0  — the claim time in ms (unencoded): the member was in `pending`, has
--          been SREM'd and ZADD'd to `inflight` with the encoded score. The
--          caller owns it.
--   0    — the member was NOT in `pending` (already claimed by another
--          worker, or already settled). The caller skips it.

local pending_key  = KEYS[1]
local inflight_key = KEYS[2]
local age_key      = KEYS[3]
local bytes_key    = KEYS[4]
local total_key    = KEYS[5]
local settled_key  = KEYS[6]
local member       = ARGV[1]

-- Must equal `external_values::GC_CLAIM_SCORE_OFFSET_MS` (see the header).
local CLAIM_SCORE_OFFSET_MS = 10000000000000

-- b.u96 item 7: sample the Valkey-side clock instead of trusting
-- caller-supplied ARGV. `TIME` returns `{seconds, microseconds}` —
-- collapse to millis with integer arithmetic so the ZSET score stays
-- a clean monotonic integer.
local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local removed = redis.call("SREM", pending_key, member)
if removed == 1 then
  -- Atomic within the script: SREM landed, so this ZADD pins ownership
  -- with the encoded claim time. A later pass reads the score back to decide
  -- whether to re-claim; the decrement fences on the unencoded reply.
  redis.call("ZADD", inflight_key, now_ms + CLAIM_SCORE_OFFSET_MS, member)
  redis.call("ZREM", age_key, member)
  redis.call("HSETNX", bytes_key, member, 0)
  return now_ms
end

-- Self-heal: only for a member in neither set (see the header).
if redis.call("ZSCORE", inflight_key, member) then
  return 0
end
local healed = redis.call("ZREM", age_key, member)
local held = redis.call("HGET", bytes_key, member)
if redis.call("HDEL", bytes_key, member) == 1 then
  healed = 1
  local b = tonumber(held)
  if b and b > 0 then
    redis.call("DECRBY", total_key, b)
  end
end
if healed > 0 then
  redis.call("ZADD", settled_key, now_ms, member)
end
return 0
