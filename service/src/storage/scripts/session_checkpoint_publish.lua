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

-- session_checkpoint_publish.lua — monotone publish watermark for
-- session-checkpoint adapter operations (b.443).
--
-- ── The race this closes ──────────────────────────────────────────────
--
-- `put_session_checkpoint` is last-writer-wins to ONE fixed path per
-- session (the LocalDisk adapter overwrites in place; StorageApi PUTs the
-- same address), and the dispatch tasks that call it are independent.
-- So a build that started earlier, stalled, and PUT later durably
-- overwrites a fresher payload — typically one that names a layer the
-- stale payload predates. Nothing re-triggers a repair until the next
-- lifecycle event on that session, and if none ever comes, the loss is
-- permanent and silent.
--
-- The adapter cannot arbitrate this itself: `CheckpointMeta` is discarded
-- by LocalDisk and ignored by StorageApi, SR-1.5 forbids parsing the
-- body, and there is no CAS on the object surface. And the racing writers
-- are usually on DIFFERENT replicas — there is no session affinity — so
-- an in-process lock cannot order them either. Valkey is the one thing
-- every replica shares, so the ordering is anchored there.
--
-- ── The protocol ──────────────────────────────────────────────────────
--
-- Every session-checkpoint build mints a monotone stamp atomically with
-- the state read it stamps (`session_checkpoint_read.lua`). After the
-- adapter call SUCCEEDS, the writer publishes that stamp here:
--
--   * `pub < stamp`  → this operation is the newest one to land. Record
--                      it as the watermark and answer nil (accepted).
--   * `pub >= stamp` → a strictly fresher operation already landed, and
--                      this operation just clobbered it. Answer the
--                      fresher operation's KIND so the caller can repair:
--                      `"write"` → rebuild (mints a higher stamp, so the
--                      repair itself wins), `"delete"` → re-issue the
--                      DELETE.
--
-- Because the stamp is minted atomically with the read, a higher stamp
-- always implies a payload that reflects at least as much committed
-- state. So "highest stamp wins" is not a tiebreak heuristic — it is a
-- freshness order. The watermark makes the highest-stamped operation the
-- one that stays durable, which subsumes the torn-read case as well: a
-- torn build can only have been produced by a mutation that enqueued its
-- own later build.
--
-- This is a repair protocol, not a lock. The clobber still happens; what
-- the watermark guarantees is that it is DETECTED by the clobberer and
-- undone, rather than persisting unnoticed.
--
-- ── The incarnation gate ──────────────────────────────────────────────
--
-- Both `seq` and `pub` live on `le:{sid}:ckpt`, so losing that key resets
-- the whole ordering — and a build minted BEFORE the loss is then holding
-- a stamp drawn from a counter that no longer exists. Its publish would
-- outrank the fresh watermark and durably reinstate pre-loss bytes. (Key
-- loss is not exotic: FLUSHALL, a failover onto a lagging replica, an
-- out-of-band DEL, or the 24 h TTL expiring on an idle session that then
-- sees one more lifecycle event.)
--
-- So each publish carries the incarnation token its build read
-- (`session_checkpoint_read.lua` ARGV[3] / reply element 7), and this
-- script refuses outright when the token on the key is absent or
-- different. The refusal writes NOTHING — not the watermark, not even the
-- TTL: a mismatched incarnation means we no longer own this key, and a
-- bare `PEXPIRE` would resurrect a key the live incarnation never asked
-- for (an `HSETNX`-less, `seq`-less husk that the next mint would then
-- adopt as its own incarnation).
--
-- `seq >= stamp` would ALMOST do the same job without a token, but it
-- leaves the case where the post-loss counter has climbed back to exactly
-- the stale build's stamp: the equal-stamp/same-kind arm below would then
-- accept the stale operation, believing it to be its own retry. The token
-- makes the two distinguishable.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS:
--   [1] `le:{sid}:ckpt` — session-checkpoint coordination hash
--                         (KeySchema::session_checkpoint_state). Single
--                         key, so Cluster-safe with nothing to reason
--                         about.
--
-- ARGV:
--   [1] stamp   — the build stamp of the operation that just landed.
--   [2] kind    — `"write"` or `"delete"`.
--   [3] ttl_ms  — TTL refreshed on KEYS[1] on every invocation whose
--                 incarnation matches, accepted and refused alike. The key
--                 must outlive the session it names (a repair reads it
--                 after a DELETE has landed), so it is reaped by TTL rather
--                 than by any lifecycle path. NOT refreshed on the
--                 incarnation-mismatch arm — see the gate above.
--   [4] inc     — the incarnation token this operation's build read from
--                 `session_checkpoint_read.lua` (its reply element 7).
--
-- Returns:
--   * nil                  — accepted; this operation is the newest to land.
--   * `"stale_incarnation"` — refused; the key this operation's stamp was
--                            drawn from is gone or has been re-created, so
--                            the stamp is not comparable to the live
--                            watermark at all. Nothing was written.
--   * any other string     — refused; it is the `pub_kind` of the fresher
--                            operation that this one clobbered (`"write"`
--                            or `"delete"`, the closed vocabulary shared
--                            with the caller's `PendingOp::kind`).
--
-- `"stale_incarnation"` is deliberately outside the `pub_kind` vocabulary
-- so a caller that only knows the two kinds cannot silently mistake it for
-- one; the Rust decoder matches it by name before consulting `PendingOp`.

local ckpt_key = KEYS[1]
local stamp    = tonumber(ARGV[1])
local kind     = ARGV[2]
local ttl_ms   = ARGV[3]
local inc      = ARGV[4]

-- Incarnation gate — BEFORE any write. A missing `inc` means the key does
-- not exist (or predates the field on a key nothing has minted against
-- since); a different `inc` means the key was lost and re-created under a
-- new lifetime. Either way this operation's stamp was drawn from a counter
-- that no longer governs anything here, so it must not touch the
-- watermark — and must not PEXPIRE a key it does not own.
local live_inc = redis.call('HGET', ckpt_key, 'inc')
if (not live_inc) or live_inc ~= inc then
  return 'stale_incarnation'
end

local raw_pub = redis.call('HGET', ckpt_key, 'pub')
local pub = 0
if raw_pub then
  pub = tonumber(raw_pub) or 0
end

if pub < stamp then
  redis.call('HSET', ckpt_key, 'pub', stamp, 'pub_kind', kind)
  redis.call('PEXPIRE', ckpt_key, ttl_ms)
  return nil
end

redis.call('PEXPIRE', ckpt_key, ttl_ms)

local pub_kind = redis.call('HGET', ckpt_key, 'pub_kind')

-- Equal-stamp arm — load-bearing, not merely defensive. Distinct
-- operations never share a stamp (each mints its own), so this can only
-- be ONE operation publishing twice, which is exactly what the writer's
-- publish-retry ladder does: once an adapter call has landed, the
-- operation is never re-run, and only this publish is retried, carrying
-- the same stamp forward. The common trigger is a timeout on a call the
-- server actually applied — the retry then finds its own watermark
-- already in place. Accept it, rather than reporting a self-supersession
-- that would send the writer into a pointless repair against itself.
if pub == stamp and pub_kind == kind then
  return nil
end

-- `pub` is set but `pub_kind` is not: unreachable, since the two are
-- written by one HSET. Answer "write" so the caller repairs by rebuilding
-- from live state, which is the arm that cannot destroy information.
if not pub_kind then
  return 'write'
end

return pub_kind
