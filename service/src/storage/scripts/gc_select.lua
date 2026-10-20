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

-- gc_select.lua — one read-only snapshot of the external-value release ledger
-- for one GC loop pass (b.yuo; stale claims and un-aged members since b.1jt).
--
-- The GC loop (`external_values::process_pending`) starts every pass with
-- this one round trip. It reads, on the `{gc}` primary's clock:
--
--   * the ledger's levels — the byte total, `SCARD pending`, `ZCARD inflight`,
--     `ZCARD settled` — which the leader publishes as gauges;
--   * the oldest age in `pending:age`;
--   * the read hold: the earliest start among the live holders' fields, and
--     how many fields are live (the leader publishes that count as a gauge);
--   * up to `settled_limit` settled `(member, score)` pairs whose score is at
--     least the replay-record retention U old and whose member is in NEITHER
--     `pending` nor `inflight` (a member back in a set is skipped — its next
--     settle re-scores it). The loop tears down each one's replay record
--     (`gc_unmark.lua`) and then retires the entry (`gc_settled_retire.lua`);
--   * up to `candidate_limit` members to claim, oldest first, by
--     `ZRANGEBYSCORE pending:age -inf <cutoff>` (under pressure, with each
--     examined member's `pending:bytes`, to size the excess over T);
--   * up to `unaged_sample` sampled un-aged members (below);
--   * the stale claims in `inflight` (below), up to `stale_claim_limit` of
--     each kind.
--
-- The cutoff:
--   * normally `now − min_age`, so no member is collected younger than the
--     age floor;
--   * under pressure — the pressure threshold T is above 0 and the byte
--     total is at or above it — the pass collects only the excess over T,
--     oldest first: every member past the age floor as above, whatever the
--     excess, then younger members, no younger than `now − pressure_min_age`,
--     each added only while the `pending:bytes` of the members selected so
--     far (those past the age floor included) do not exceed `total − T`. The
--     younger member that takes that sum past `total − T` is the last one
--     added, so once the selected members complete the total is below T
--     (unless the batch limit ended the selection first). An absent, 0 or
--     non-numeric bytes field counts 0. (`GcPolicy::PRODUCTION`'s pressure
--     floor is 0, so under pressure its excess is collected at any age.)
--     When the pressure floor is not younger than the age floor, the cutoff
--     is simply `now − pressure_min_age` and the excess plays no part;
--   * while any read hold field is live and hold clipping is on,
--     additionally clipped below `hold_start − hold_margin_ms`, where
--     `hold_start` is the EARLIEST live field's start: no member enqueued
--     within the margin of that start, or later, is selected. A reader that
--     missed a blob takes a hold and re-reads; any blob its fresh read
--     references is released, if at all, by a commit after its hold began.
--     A field is live while its `expires_ms` is above `now`; an expired one
--     (a holder that died, or one whose renewal came too late) is ignored.
--
-- Only a member with a `pending:age` entry can be a candidate, so no member is
-- ever collected un-aged.
--
-- **Un-aged members** — in `pending` with no `pending:age` entry: an older
-- build's bare `SADD`, since every enqueue of this build records the age with
-- the member — are looked for on every pass while `pending` is non-empty, with
-- `SRANDMEMBER pending <unaged_sample>`, whether or not a read hold is live,
-- and returned separately. The pass stamps each with the current time
-- (`gc_stamp_unaged.lua`), and from then on it is selected like any other
-- member: past the age floor from its stamp, and clipped by a hold measured
-- from its stamp, which is never earlier than its true enqueue, so the clip
-- stays sound. (The sample is not gated on `SCARD pending > ZCARD
-- pending:age`: stale age entries of members no longer pending would mask
-- un-aged ones.)
--
-- **Stale claims.** `inflight` scores are claim times; this build encodes its
-- own as claim ms + `offset` (`GC_CLAIM_SCORE_OFFSET_MS`, 10^13), and every
-- older build scores its with the bare claim time, always below the offset:
--   * own claims — score at or above `offset` — whose claim time is at least
--     `reclaim_after_ms` (D) old: the pass re-claims each in place
--     (`gc_reclaim_stale.lua`) and decides its decrement once more through
--     the replay record; it never goes back to `pending`;
--   * an older build's claims — score below `offset` — at least
--     `foreign_after_ms` (the decrement fence F) old: the pass drops each
--     without a decrement (`gc_drop_foreign_claims.lua`).
-- Each comes back as `(member, score)`, the score as the store reported it,
-- which is what those scripts compare before they act.
--
-- Read-only: every call is a read, so the script is safe to run anywhere,
-- any number of times. A wrong-typed key raises, which ends that pass only.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS (all `{gc}`-tagged, so the script is single-slot on a Cluster):
--   [1]  le:ext:{gc}:pending        (SET)
--   [2]  le:ext:{gc}:inflight       (ZSET)
--   [3]  le:ext:{gc}:pending:age    (ZSET)
--   [4]  le:ext:{gc}:pending:total  (STRING)
--   [5]  le:ext:{gc}:settled        (ZSET)
--   [6]  le:ext:{gc}:read_hold      (HASH: `<object_id>:<token>` →
--                                    `<start_ms>:<expires_ms>`, one field per
--                                    holder, written by `ext_read_hold.lua`; a
--                                    key of another type fails the pass)
--   [7]  le:ext:{gc}:pending:bytes  (HASH: member → the bytes it accounts
--                                    for; read only under pressure, to size
--                                    the excess)
--
-- ARGV (all canonical non-negative decimals except [5]):
--   [1]  min_age_ms
--   [2]  pressure_min_age_ms
--   [3]  pressure_threshold_bytes   (T; 0 = never under pressure)
--   [4]  replay_retention_ms        (U)
--   [5]  hold_clip                  ("1" = clip below a live hold, "0" = not)
--   [6]  hold_margin_ms
--   [7]  candidate_limit
--   [8]  settled_limit
--   [9]  unaged_sample              (SRANDMEMBER count for un-aged members)
--   [10] claim_score_offset_ms      (the own-claim encoding offset)
--   [11] reclaim_after_ms           (D)
--   [12] foreign_after_ms           (F)
--   [13] stale_claim_limit          (the most stale claims of each kind)
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- A flat array:
--   [1]  now_ms
--   [2]  hold_start_ms              (the earliest live field's start; 0 when
--                                    no field is live)
--   [3]  holds_live                 (live read hold fields; read whether or
--                                    not hold clipping is on)
--   [4]  pending_total_bytes        (0 when absent or not a number)
--   [5]  SCARD pending
--   [6]  ZCARD inflight
--   [7]  oldest age, ms             (-1 when pending:age is empty)
--   [8]  ZCARD settled
--   [9]  under_pressure             (1 | 0)
--   [10] S, the number of settled pairs that follow
--   [11] C, the number of candidates that follow
--   [12] A, the number of un-aged members that follow
--   [13] R, the number of own stale claims that follow
--   [14] X, the number of an older build's stale claims that follow
--   then 2S strings `member, score` (settled), C member strings
--   (candidates), A member strings (un-aged), 2R strings `member, score`
--   (own stale claims) and 2X strings `member, score` (an older build's).

local pending_key  = KEYS[1]
local inflight_key = KEYS[2]
local age_key      = KEYS[3]
local total_key    = KEYS[4]
local settled_key  = KEYS[5]
local hold_key     = KEYS[6]
local bytes_key    = KEYS[7]

local min_age_ms          = tonumber(ARGV[1])
local pressure_min_age_ms = tonumber(ARGV[2])
local pressure_threshold  = tonumber(ARGV[3])
local retention_ms        = tonumber(ARGV[4])
local hold_clip           = ARGV[5] == "1"
local hold_margin_ms      = tonumber(ARGV[6])
local candidate_limit     = tonumber(ARGV[7])
local settled_limit       = tonumber(ARGV[8])
local unaged_sample       = tonumber(ARGV[9])
local claim_offset        = tonumber(ARGV[10])
local reclaim_after_ms    = tonumber(ARGV[11])
local foreign_after_ms    = tonumber(ARGV[12])
local stale_claim_limit   = tonumber(ARGV[13])
if min_age_ms == nil or pressure_min_age_ms == nil or pressure_threshold == nil
    or retention_ms == nil or hold_margin_ms == nil or candidate_limit == nil
    or settled_limit == nil or unaged_sample == nil or claim_offset == nil
    or reclaim_after_ms == nil or foreign_after_ms == nil
    or stale_claim_limit == nil then
  return redis.error_reply("ENCODING: gc_select expects thirteen ARGV slots, all decimal but the hold-clip flag")
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local total = tonumber(redis.call("GET", total_key)) or 0
local pending_count = redis.call("SCARD", pending_key)
local inflight_count = redis.call("ZCARD", inflight_key)
local settled_count = redis.call("ZCARD", settled_key)

local oldest_age = -1
local oldest = redis.call("ZRANGE", age_key, 0, 0, "WITHSCORES")
if #oldest == 2 then oldest_age = tonumber(oldest[2]) end

local hold_start = 0
local holds_live = 0
local hold_fields = redis.call("HGETALL", hold_key)
for i = 2, #hold_fields, 2 do
  local s, e = string.match(hold_fields[i], "^(%d+):(%d+)$")
  if s and tonumber(e) > now_ms then
    holds_live = holds_live + 1
    local start = tonumber(s)
    if hold_start == 0 or start < hold_start then hold_start = start end
  end
end
local hold_live = hold_clip and holds_live > 0

-- Settled replay records that are due for teardown.
local settled_out = {}
local settled_n = 0
if settled_limit > 0 then
  local raw = redis.call("ZRANGEBYSCORE", settled_key, "-inf", now_ms - retention_ms,
    "WITHSCORES", "LIMIT", 0, settled_limit * 4)
  for i = 1, #raw, 2 do
    if settled_n >= settled_limit then break end
    local m = raw[i]
    if redis.call("SISMEMBER", pending_key, m) == 0
        and not redis.call("ZSCORE", inflight_key, m) then
      settled_out[#settled_out + 1] = m
      settled_out[#settled_out + 1] = raw[i + 1]
      settled_n = settled_n + 1
    end
  end
end

-- Candidates, oldest first. `cutoff` bounds the members selected whatever
-- the total; under pressure, `excess_cutoff` (when the pressure floor is the
-- younger one) bounds the younger members selected only while the bytes
-- selected so far do not exceed `total − T`.
local under_pressure = 0
local cutoff = now_ms - min_age_ms
local excess_cutoff = nil
if pressure_threshold > 0 and total >= pressure_threshold then
  under_pressure = 1
  local floor_cutoff = now_ms - pressure_min_age_ms
  if floor_cutoff > cutoff then
    excess_cutoff = floor_cutoff
  else
    cutoff = floor_cutoff
  end
end
if hold_live then
  local clip = hold_start - hold_margin_ms - 1
  if cutoff > clip then cutoff = clip end
  if excess_cutoff and excess_cutoff > clip then excess_cutoff = clip end
end

local candidates = {}
if candidate_limit > 0 then
  if excess_cutoff and excess_cutoff > cutoff then
    local excess = total - pressure_threshold
    local covered = 0
    local raw = redis.call("ZRANGEBYSCORE", age_key, "-inf", excess_cutoff,
      "WITHSCORES", "LIMIT", 0, candidate_limit)
    for i = 1, #raw, 2 do
      local m = raw[i]
      if tonumber(raw[i + 1]) > cutoff and covered > excess then break end
      candidates[#candidates + 1] = m
      local bytes = tonumber(redis.call("HGET", bytes_key, m))
      if bytes and bytes > 0 then covered = covered + bytes end
    end
  else
    candidates = redis.call("ZRANGEBYSCORE", age_key, "-inf", cutoff,
      "LIMIT", 0, candidate_limit)
  end
end

-- Un-aged members: sampled every pass while `pending` is non-empty, hold or
-- no hold. Returned for stamping, never as candidates.
local unaged = {}
if unaged_sample > 0 and pending_count > 0 then
  local seen = {}
  local sample = redis.call("SRANDMEMBER", pending_key, unaged_sample)
  for _, m in ipairs(sample) do
    if not seen[m] and not redis.call("ZSCORE", age_key, m) then
      unaged[#unaged + 1] = m
      seen[m] = true
    end
  end
end

-- Stale claims. Own: encoded scores in [offset, offset + now − D]. An older
-- build's: bare scores in [-inf, now − F], always below the offset (a bare
-- claim time is far below it).
local own_stale = {}
local foreign = {}
if stale_claim_limit > 0 then
  own_stale = redis.call("ZRANGEBYSCORE", inflight_key, claim_offset,
    claim_offset + now_ms - reclaim_after_ms, "WITHSCORES", "LIMIT", 0, stale_claim_limit)
  local foreign_cutoff = now_ms - foreign_after_ms
  if foreign_cutoff >= claim_offset then foreign_cutoff = claim_offset - 1 end
  foreign = redis.call("ZRANGEBYSCORE", inflight_key, "-inf", foreign_cutoff,
    "WITHSCORES", "LIMIT", 0, stale_claim_limit)
end

local out = {
  now_ms, hold_start, holds_live, total, pending_count, inflight_count,
  oldest_age, settled_count, under_pressure,
  settled_n, #candidates, #unaged, #own_stale / 2, #foreign / 2,
}
for _, v in ipairs(settled_out) do out[#out + 1] = v end
for _, m in ipairs(candidates) do out[#out + 1] = m end
for _, m in ipairs(unaged) do out[#out + 1] = m end
for _, v in ipairs(own_stale) do out[#out + 1] = v end
for _, v in ipairs(foreign) do out[#out + 1] = v end
return out
