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

-- recon_ext_rearm.lua — a backup writer's re-arm on the `{recon}` slot: record
-- that it is about to publish a reference to tombstoned ext hashes, learn
-- whether the collect deleted any of them inside the writer's reliance window,
-- and settle the writer's own stem against the orphan and deletion lists
-- (b.g65, SDD §22.17.3, §22.17.5).
--
-- ── Who calls it, and when ────────────────────────────────────────────────
--
-- * Every layer publish, through `checkpoint::put_guard`, after its liveness
--   probe and immediately BEFORE the adapter PUT, with the hashes its ext-refs
--   sidecar names (none for a tail) and its stem. Before, because the sidecar
--   is the first artifact the adapter publishes: a re-arm after the PUT would
--   leave a collect free to delete when its snapshot predates the sidecar and
--   its fence predates the re-arm.
-- * The backup repair's re-send and the ext-mirror worker's upload, with the
--   one hash they are about to PUT and no stem.
-- * The collect, with no hash and no stem, to read the `{recon}` clock (the
--   snapshot clock `t0` its delete threshold is anchored to).
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:ckpt:{recon}:rearm
--   [2]  le:ckpt:{recon}:tomb
--   [3]  le:ckpt:{recon}:collected
--   [4]  le:ckpt:{recon}:orphans
--   [5]  le:ckpt:{recon}:deleting
--
-- ARGV:
--   [1]  lookback_ms   the writer's reliance window: a `:collected` entry at
--                      or after now − lookback_ms refuses the hash.
--   [2]  grace_ms      the reconciliation grace: a `:deleting` announcement of
--                      the stem younger than this refuses the publish.
--   [3]  stem          the writer's stem in the `:orphans` spelling, or the
--                      empty string for none.
--   [4..] hashes       the ext hashes the writer is about to publish a
--                      reference to (may be empty).
--
-- Returns `{ now_ms, stem_refused, collected... }`:
--   now_ms        the Valkey clock of this run, in ms;
--   stem_refused  1 when the stem is announced on `:deleting` within the
--                 grace, else 0; nothing was written for the stem when this
--                 is 1 or any hash is collected;
--   collected...  the hashes with a `:collected` entry inside the lookback.
--
-- ── What it writes ────────────────────────────────────────────────────────
--
-- * `ZADD GT` on `:rearm` at `now_ms` for every hash that is a member of
--   `:tomb` (an untombstoned hash cannot be collected, so its re-arm would be
--   noise), then `PEXPIRE :rearm` 24 h. `GT` never lowers a score. It never
--   re-scores `:tomb`: an active layer publishes about once a minute, so
--   re-scoring would keep its tombstoned hashes unclaimable forever.
--   The re-arm is written whether or not the run refuses: it only ever makes
--   a collect refuse more.
-- * When a stem is given and the run refuses nothing (the stem is not being
--   deleted and no hash was collected inside the lookback): `ZREM` of the
--   stem from `:orphans` (a publish heals its own orphan candidate) and from
--   `:deleting` (a stale announcement must not explain away a later absence
--   of this new copy). A refused publish writes nothing for its stem: it does
--   not PUT, so there is nothing to heal.
--
-- ── Store memory (§7.5.1) ─────────────────────────────────────────────────
--
-- The first command is the out-of-memory opener, `ZREM :rearm ":oom-gate"`:
-- a write command that cannot grow memory and removes nothing (no member
-- starting with `:` is ever written). On a store over `maxmemory` Valkey
-- refuses a script's memory-growing command only while the script has written
-- nothing, so after the opener the re-arm lands on a full store too and a
-- backup is not refused for want of a few tens of bytes per tombstoned hash.
--
-- ── Slot migrations (b.cyu) ───────────────────────────────────────────────
--
-- The opener and every read — `:tomb`, `:collected`, `:deleting`, `:orphans`
-- — come before the first real write. On a migrating source a command naming
-- an absent key is refused; here that can only be the opener or a read, so
-- the run is refused before anything changes. Every later write names a key
-- the run proved present: `:rearm` by the opener, `:orphans` and `:deleting`
-- by their reads; and a `PEXPIRE` follows only the `ZADD` that just filled
-- its key. The caller reads a refusal as a transient Valkey error and defers
-- its PUT (fail closed).

local rearm_key     = KEYS[1]
local tomb_key      = KEYS[2]
local collected_key = KEYS[3]
local orphans_key   = KEYS[4]
local deleting_key  = KEYS[5]
local lookback_ms   = tonumber(ARGV[1])
local grace_ms      = tonumber(ARGV[2])
local stem          = ARGV[3]

if lookback_ms == nil or grace_ms == nil or stem == nil then
  return redis.error_reply("ENCODING: expected ARGV lookback_ms, grace_ms, stem")
end

-- The opener (see "Store memory" above). `:oom-gate` is never written.
redis.call("ZREM", rearm_key, ":oom-gate")

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local hashes = {}
for i = 4, #ARGV do
  hashes[#hashes + 1] = ARGV[i]
end

-- Reads: tombstone membership and collected entries of the named hashes.
local tomb_scores = {}
local collected_scores = {}
if #hashes > 0 then
  tomb_scores = redis.call("ZMSCORE", tomb_key, unpack(hashes))
  collected_scores = redis.call("ZMSCORE", collected_key, unpack(hashes))
end

-- Reads: the stem's announcement and orphan membership.
local stem_refused = 0
local in_orphans = false
local in_deleting = false
if stem ~= "" then
  local announced = redis.call("ZSCORE", deleting_key, stem)
  in_orphans = redis.call("ZSCORE", orphans_key, stem) ~= false
  if announced ~= false then
    in_deleting = true
    if tonumber(announced) > now_ms - grace_ms then
      stem_refused = 1
    end
  end
end

-- Writes: the re-arm of every tombstoned hash.
local rearmed = 0
for i, hash in ipairs(hashes) do
  if tomb_scores[i] then
    redis.call("ZADD", rearm_key, "GT", now_ms, hash)
    rearmed = rearmed + 1
  end
end
if rearmed > 0 then
  redis.call("PEXPIRE", rearm_key, 86400000)
end

local reply = { now_ms, stem_refused }
local cutoff = now_ms - lookback_ms
for i, hash in ipairs(hashes) do
  local score = collected_scores[i]
  if score and tonumber(score) >= cutoff then
    reply[#reply + 1] = hash
  end
end

-- Writes: the stem's own entries, only when the run refuses nothing — the
-- stem is not being deleted and no hash was collected inside the lookback. A
-- refused publish does not PUT, so it heals nothing.
if stem ~= "" and stem_refused == 0 and #reply == 2 then
  if in_orphans then
    redis.call("ZREM", orphans_key, stem)
  end
  if in_deleting then
    redis.call("ZREM", deleting_key, stem)
  end
end

return reply
