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

-- session_checkpoint_read.lua — atomic session-state read + build-stamp mint
-- for the session-checkpoint writer (b.443).
--
-- ── What this replaces ────────────────────────────────────────────────
--
-- `build_session_checkpoint` used to issue an uncut sequence of
-- round-trips against the data pool:
--
--   HGETALL le:{sid}:sess
--   SMEMBERS le:{sid}:layers
--   GET le:{sid}:layer:{hex(name)}   × N, fanned out
--
-- Nothing pinned that sequence, so a CreateLayer / DeleteLayer landing
-- mid-read produced a torn payload — most damagingly one that names a
-- layer set the session did not have, or omits one it did. The session
-- checkpoint is the ROOT OF REACHABILITY for DR: `apply_checkpoint`'s
-- SR-2.4 live-mapping guard refuses to restore a layer whose mapping is
-- absent, and the session applier's SET loop is the only thing that
-- re-creates mappings after state loss. A payload missing one mapping
-- therefore makes that layer's own checkpoint permanently unrestorable.
--
-- Running the whole read inside one script makes the tear impossible by
-- construction: Valkey executes the script atomically, so the SMEMBERS
-- and the per-member GETs observe one instant.
--
-- ── What the "cheaper" claim does and does not mean ───────────────────
--
-- The win is ROUND-TRIPS, not server occupancy: 1 instead of 2 + N. The
-- per-member GET work is the same work the fan-out did — but it now runs
-- in one uninterruptible block on Valkey's single command thread instead
-- of interleaving with other clients' commands. So this trades a small
-- amount of fairness for atomicity, and the trade is only obviously good
-- while sessions hold few layers.
--
-- Measured on the sandbox Valkey (valkey 9.0.3, EVALSHA wall time
-- including a ~0.06 ms loopback RTT), marginal cost ~1.9 µs per layer:
--
--   layers      10     100    1 000    5 000   20 000
--   median   0.09ms  0.31ms   1.88ms   9.72ms  43.32ms
--
-- Nothing caps layers per session — not config, not the schema — so that
-- last column is reachable in principle, and 43 ms of stop-the-world on
-- every lifecycle event would be a serious latency incident for every
-- other client on the node.
--
-- The assumed regime is therefore: tens of layers per session (a live USD
-- stage's sublayer set), where the atomic read costs microseconds and the
-- fairness question does not arise. If sessions with thousands of layers
-- become real, this script is the wrong shape for them — the fix would be
-- chunking the member walk across several smaller scripts under a
-- version-guard, NOT reverting to the racy fan-out. Treat a rising
-- `layers` count on a session as the signal to revisit.
--
-- ── The stamp, and why it is minted here ──────────────────────────────
--
-- `HINCRBY le:{sid}:ckpt seq 1` mints a monotone build stamp in the same
-- atomic block as the read. That co-location is the whole soundness
-- argument for the publish watermark (`session_checkpoint_publish.lua`):
--
--   * the payload reflects exactly the state committed before the stamp
--     was minted (same instant, by atomicity);
--   * every mutation that could tear a build enqueues its own, later,
--     strictly-higher-stamped build;
--   * therefore the highest-stamped build dominates every lower-stamped
--     one, and a lower-stamped PUT landing later is *known* to be stale
--     rather than merely suspected.
--
-- The stamp lives in Valkey rather than in the process because there is
-- no session affinity — every replica serves every RPC, so the racing
-- builds are typically on DIFFERENT replicas and an in-process counter
-- could not order them.
--
-- ── The incarnation token, and why the stamp alone is not enough ───────
--
-- `seq` is only monotone for as long as `le:{sid}:ckpt` SURVIVES. Lose
-- the key — FLUSHALL, a failover onto a lagging replica, an out-of-band
-- DEL, or the 24 h TTL expiring on an idle session that then sees one
-- more lifecycle event — and the next mint restarts at 1 while a build
-- minted before the loss may still be in flight with a stamp of 60. That
-- pre-loss build's adapter PUT can land AFTER the post-loss build's, and
-- its publish of 60 beats the fresh watermark of 1: the durable session
-- blob is silently replaced by pre-loss layer mappings, and `pub = 60`
-- then poisons the next 59 legitimate builds.
--
-- So every mint also carries an INCARNATION: a caller-minted random token
-- (ARGV[3]) installed with `HSETNX` on the same key. The token identifies
-- the key's LIFETIME, not the build — every build against a surviving key
-- observes the same token, and a key that was lost and re-created gets a
-- brand-new one. The publish script (`session_checkpoint_publish.lua`)
-- refuses any operation whose incarnation is not the one currently on the
-- key, so a build minted against a dead incarnation can never advance the
-- watermark of the live one.
--
-- `HSETNX` (rather than `HSET`) is what makes the token key-lifetime
-- scoped: the first mint after the key comes into existence wins and every
-- later mint reads that value back. A pre-existing key from before this
-- field existed is adopted on its first touch, so no migration step is
-- needed.
--
-- Note the alternative that is NOT sufficient: comparing `seq >= stamp` at
-- publish time. It closes the 60-beats-1 case but not the symmetric one
-- where the pre-loss build's stamp happens to equal the post-loss stamp —
-- the publish script's equal-stamp/same-kind arm would then accept the
-- stale operation silently, believing it to be its own retry. The
-- incarnation closes that too, at the cost of one `HSETNX` inside a script
-- that was already running.
--
-- The mint is unconditional — it happens even when the session is gone
-- (`present == 0`). Two reasons: the DELETE dispatch path calls this
-- same script purely to obtain an ordered stamp (its session is, by
-- construction, already deleted, so the read half short-circuits on an
-- empty HGETALL and costs nothing), and a uniform mint keeps the
-- "stamp minted ⇒ atomic with whatever state was observed" contract
-- free of special cases. A stamp minted for a session that no longer
-- exists is simply never published; the key's TTL reaps it.
--
-- ── Undeclared keys: deliberate, and Cluster-safe ─────────────────────
--
-- The per-layer mapping keys are NOT in KEYS. They cannot be: the set of
-- mapping keys is unknowable before the atomic read, and that
-- unknowability IS the race this script exists to remove. Declaring them
-- would require an SMEMBERS round-trip first, reintroducing the very
-- window we are closing.
--
-- This is safe in Cluster mode because every key this script touches —
-- `le:{sid}:ckpt`, `le:{sid}:sess`, `le:{sid}:layers`, and every
-- `le:{sid}:layer:{hex(name)}` derived from ARGV[2] + a set member —
-- carries the SAME `{session_id}` hash tag, so all of them hash to one
-- slot on one node: the slot the declared KEYS already pinned the script
-- to. The engine's cluster check on an undeclared key access is
-- "is this key local, and in the script's slot" — both hold here.
--
-- The deviation from the repo's declare-all-keys convention is therefore
-- deliberate and bounded to keys derived from ARGV[2], which the caller
-- builds with `KeySchema::layer_mapping`'s own prefix. Do not extend
-- this script to touch a key outside the `{session_id}` tag.
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS:
--   [1] `le:{sid}:ckpt`   — session-checkpoint coordination hash
--                           (KeySchema::session_checkpoint_state)
--   [2] `le:{sid}:sess`   — session metadata hash (KeySchema::session_meta)
--   [3] `le:{sid}:layers` — layers index set (KeySchema::layers_index)
--
-- ARGV:
--   [1] TTL in milliseconds applied to KEYS[1] on every invocation.
--   [2] layer-mapping key prefix — `le:{sid}:layer:`. The layers-index
--       members ARE the hex-encoded layer names (that is what
--       `KeySchema::layer_mapping` appends), so a mapping key is exactly
--       `ARGV[2] .. member`; no decoding happens server-side.
--   [3] candidate incarnation token — a caller-minted random string,
--       installed with `HSETNX` and read back. Only the FIRST mint after
--       the key came into existence installs its candidate; every later
--       mint gets the incumbent back. Never compared server-side here —
--       the comparison is `session_checkpoint_publish.lua`'s job.
--
-- Returns a 7-element array:
--   [1] stamp        — integer, the freshly minted build stamp.
--   [2] present      — 1 when the session exists, 0 when `le:{sid}:sess`
--                      is empty. On 0, elements 3..6 are empty/zero and
--                      the caller maps the reply to `Ok(None)`.
--   [3] sess_fields  — flat [field, value, ...] array from HGETALL.
--   [4] hex_names    — layers-index members that HAVE a live mapping.
--   [5] mapping_vals — mapping values, positionally aligned with [4].
--   [6] dropped      — count of index members with no mapping key.
--   [7] incarnation  — the EFFECTIVE incarnation token on the key, which
--                      is ARGV[3] only when this mint created the field.
--                      Returned on the `present == 0` arm too: the DELETE
--                      dispatch path mints against an already-deleted
--                      session and still has to publish under the right
--                      incarnation.
--
-- Members without a mapping key are filtered out server-side rather than
-- returned as nils: a Lua-to-RESP array truncates at the first nil, so a
-- sparse array would silently drop every member after the first hole.
-- Filtering keeps [4] and [5] aligned by construction and preserves the
-- pre-existing skip-the-member behaviour, which is still reachable — NOT
-- as a torn read (atomicity rules that out) but as a genuine keyspace
-- state: `LayerStore::delete_layer` DELs the mapping several steps before
-- it SREMs the index member, so a concurrent build legitimately observes
-- an index member whose mapping is already gone. The count in [6] lets
-- the caller log that without guessing.

local ckpt_key   = KEYS[1]
local sess_key   = KEYS[2]
local layers_key = KEYS[3]
local ttl_ms     = ARGV[1]
local map_prefix = ARGV[2]
local inc_token  = ARGV[3]

-- Claim the incarnation BEFORE the first stamp is minted, so the
-- invariant "a key carrying `seq` also carries `inc`" holds for every key
-- this script ever creates. `HSETNX` makes the claim key-lifetime scoped:
-- only the mint that brings the key into existence installs its own
-- token; every later mint reads the incumbent back and publishes under
-- it. See the header for why the stamp alone cannot survive key loss.
redis.call('HSETNX', ckpt_key, 'inc', inc_token)
local incarnation = redis.call('HGET', ckpt_key, 'inc')

-- Mint second, unconditionally. See the header for why this is not gated
-- on the session existing.
local stamp = redis.call('HINCRBY', ckpt_key, 'seq', 1)
redis.call('PEXPIRE', ckpt_key, ttl_ms)

local sess = redis.call('HGETALL', sess_key)
if #sess == 0 then
  -- Session gone (or never existed). The caller maps this to `Ok(None)`
  -- and exits silently WITHOUT deleting anything: an absent session-meta
  -- hash is also what selective key loss looks like, and turning that
  -- into a checkpoint DELETE would destroy the recovery artifact that
  -- exists precisely to repair it.
  return { stamp, 0, {}, {}, {}, 0, incarnation }
end

local members     = redis.call('SMEMBERS', layers_key)
local hex_names   = {}
local mapping_val = {}
local dropped     = 0

for i = 1, #members do
  local member = members[i]
  local value = redis.call('GET', map_prefix .. member)
  if value then
    hex_names[#hex_names + 1]     = member
    mapping_val[#mapping_val + 1] = value
  else
    dropped = dropped + 1
  end
end

return { stamp, 1, sess, hex_names, mapping_val, dropped, incarnation }
