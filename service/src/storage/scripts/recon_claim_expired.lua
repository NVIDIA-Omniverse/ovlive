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

-- recon_claim_expired.lua — register reconciliation candidates and claim the
-- ones that have outlived the grace window (b.9qd).
--
-- The generic half of the checkpoint-store reconciliation protocol. Both of its
-- ZSETs — `le:ckpt:{recon}:orphans` (members are orphan-candidate stems) and
-- `le:ckpt:{recon}:tomb` (members are ext-blob `hash_hex` strings) — need the
-- same two-phase operation, so one script serves both and the caller supplies
-- the key.
--
-- ── Why a script at all ───────────────────────────────────────────────────
--
-- The register and the claim have to see the SAME clock sample. If the caller
-- registered with one timestamp and then computed the expiry cutoff from
-- another, a candidate could be admitted and expired inside one pass — which
-- would destroy the entire safety argument (see below). Sampling `TIME` once,
-- Valkey-side, inside one atomic script is what guarantees "registered now"
-- and "expired now" are answered against one instant.
--
-- ── The safety argument this implements ──────────────────────────────────
--
-- The reconciliation pass has no trustworthy artifact timestamps to work from:
-- a blob store's `last_modified` is not when the publish started, and a
-- filesystem `mtime` survives neither a restore nor a copy. So the protocol
-- dates its own OBSERVATIONS instead of the artifacts. A candidate observed
-- twice, at least one grace window apart, with nothing having healed it in
-- between, cannot be a publish still in flight — the grace is a multiple of the
-- adapter's own `publish_worst_case`. `ZADD NX` is what makes that true: it
-- pins the FIRST observation's timestamp and every later pass leaves it alone,
-- so the score really is "when we first saw this", not "when we last saw it".
-- An `XX`/plain ZADD would refresh the score every pass and nothing would ever
-- expire.
--
-- ── Valkey-side TIME ──────────────────────────────────────────────────────
--
-- Same discipline, and the same reason, as `gc_claim_pending.lua` (b.u96 item
-- 7): a caller-supplied `SystemTime::now()` would expose the protocol to NTP
-- skew, and a replica running clock-ahead would expire candidates its peers had
-- only just registered — deleting artifacts whose publish was still in flight.
-- `redis.call("TIME")` makes the whole fleet agree on the watermark regardless
-- of local-machine clock skew.
--
-- Replication note: since Valkey 7.0 (and Redis 7.0) all Lua scripts default to
-- *effects replication*, so `redis.call("TIME")` is safely deterministic from
-- the replica's point of view — the master records the resulting ZADDs with the
-- master-side timestamp and the replica applies them verbatim without
-- re-sampling. Verified against Valkey 9.0.3 (the deployed image).
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  zset_key    (le:ckpt:{recon}:orphans or le:ckpt:{recon}:tomb)
--
-- ARGV:
--   [1]  grace_ms    (milliseconds a candidate must have been registered for
--                     before it may be claimed; derived from the adapter's
--                     publish_worst_case, never operator-configured)
--   [2]  max_return  (cap on the number of expired members returned. **0 means
--                     register-only** — the caller is batching registrations
--                     and will fetch in a separate call, so nothing is
--                     returned.)
--   [3..] members    (candidates to register with ZADD NX; may be empty, in
--                     which case the call is a pure claim)
--
-- ── Return value ─────────────────────────────────────────────────────────
--
-- Array of member strings whose FIRST-observed score is at or below
-- `now_ms - grace_ms`, oldest first, at most `max_return` of them. Empty when
-- `max_return` is 0, when nothing has aged out, or when the set is empty.
--
-- The script does NOT remove what it returns. Removal is the caller's
-- acknowledgement that it finished acting on the member (deleted the orphan,
-- collected or spared the blob) and is a plain `ZREM` — a member left behind by
-- a failed action stays claimable on the next pass, which is the retry
-- behaviour the protocol wants. Contrast `gc_claim_pending.lua`, which moves
-- ownership atomically because its worker must not share a hash with a peer;
-- here duplicate action is idempotent (both deletes are idempotent-on-missing)
-- so exclusive claim buys nothing worth a second key.

local zset_key   = KEYS[1]
local grace_ms   = tonumber(ARGV[1])
local max_return = tonumber(ARGV[2])

-- One clock sample for both phases. See the header.
local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

-- Phase 1 — register. `NX` pins the first observation and is load-bearing.
for i = 3, #ARGV do
  redis.call("ZADD", zset_key, "NX", now_ms, ARGV[i])
end

if max_return <= 0 then
  return {}
end

-- Phase 2 — claim. A member registered by THIS call scores `now_ms`, which is
-- strictly greater than the cutoff for any positive grace, so it can never be
-- returned by the same call that registered it.
local cutoff = now_ms - grace_ms
if cutoff < 0 then
  cutoff = 0
end

return redis.call("ZRANGEBYSCORE", zset_key, 0, cutoff, "LIMIT", 0, max_return)
