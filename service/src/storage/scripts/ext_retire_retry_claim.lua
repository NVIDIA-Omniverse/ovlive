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

-- ext_retire_retry_claim.lua — the orphan sweep's re-token of a retiring
-- object's carrier: take up its recorded release obligation under a fresh
-- attempt token, without removing anything (b.yda; rewritten by b.yuo Part R).
--
-- ── What this is for ──────────────────────────────────────────────────────
--
-- A retirement writes its owed release members to `le:{oid}:obj:extret` and
-- the token of each enqueue attempt to `le:{oid}:obj:extret:attempts` BEFORE
-- the attempt is sent (`ext_retire_begin.lua`). When the retirement could not
-- confirm any attempt landed, the carrier is still there, and the orphan
-- sweep's retire-retry phase (`storage::orphan_sweep::drain_retire_carrier`)
-- settles it. This script is that drain's first step.
--
-- It is NON-DESTRUCTIVE, and that is the point. Before Part R the claim
-- `SMEMBERS`ed and `DEL`ed the carrier in one step and the caller put the
-- members back if its enqueue failed; a lost claim reply lost the obligation,
-- and a lost enqueue reply re-presented members that may already have been
-- paid. Now the drain only ADDS a token — the one its own re-drive will carry —
-- and the carrier is deleted only later, by `ext_retire_clear.lua` once some
-- attempt is known to have landed (or, for an older build's carrier, to drop
-- it), or by `ext_retire_settle.lua` when every recorded attempt is decided
-- void and the id is live again. A lost reply here leaves every member on the
-- carrier.
--
-- ── The carrier's type (b.1jt) ────────────────────────────────────────────
--
-- This build writes the carrier as a HASH (member → 1). An older build wrote
-- a SET at the same key, and its own carrier claim reads `SCARD` / `SMEMBERS`,
-- so it fails `WRONGTYPE` on a HASH before writing anything. This script reads
-- either: a HASH by `HLEN` / `HKEYS` (`kind` "hash"), a SET by `SCARD` /
-- `SMEMBERS` (`kind` "set", an older build's carrier). The drain never pays a
-- SET carrier — most older builds record no attempt token at all, and no
-- recorded token is not proof that nothing ran — it clears one if a recorded
-- attempt landed and otherwise drops it. A key of any other type is refused
-- `WRONGTYPE` with nothing written.
--
-- ── Steps ─────────────────────────────────────────────────────────────────
--
--   1. Empty carrier → reply `empty`, write NOTHING. The obligation was
--      settled (or never written). A token HASH beside no carrier is left
--      alone: new-build code deletes the HASH only in the same step as an
--      existing carrier, so a lone HASH comes only from an older build's
--      destructive claim, and it may be the only `{object_id}` evidence that
--      this incarnation's obligation was already paid. Only the incarnation
--      boundary may remove it, and nothing in this build does: b.9ud sub-task
--      3 plans the only safe remover (the restore's residue wipe, extended to
--      void the tokens and delete the HASH — not built).
--   2. The newest token was added less than `in_progress_ms` ago by the
--      object's primary clock → reply `in_progress`, write nothing. A
--      retirement still running between its begin and its clear owns that
--      carrier, as does a drain that re-tokened it moments ago; the work-list
--      entry is kept and the next tick looks again. This bound decides how
--      soon a sweep may take a carrier from a sender that is still live; it
--      never causes an over-release (a drain that overtakes a live retirement
--      is still decided by the `{gc}` resolve, and no `{gc}` record is removed
--      while a send under its token can still be decided), and its only cost
--      is how often a still-live retirement is cancelled (a leak).
--   3. Otherwise `HSETNX attempts <new_token> <now_ms>` and reply `claimed`
--      with the carrier's members, the tokens that were there before
--      (`prior`) and the carrier's kind. The caller resolves every prior token
--      on the `{gc}` slot before it sends anything under the new one.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (both `{object_id}`-tagged: single-slot, and keyed for the b.289
-- fence):
--   [1]  carrier   le:{oid}:obj:extret
--   [2]  attempts  le:{oid}:obj:extret:attempts
--
-- ARGV:
--   [1]  new_token       — 32 lowercase hex characters
--   [2]  in_progress_ms  — non-negative decimal (`RETIRE_INTENT_IN_PROGRESS_AGE`)
--
-- An untokened carrier from an older build has no token HASH at all: it is
-- never "in progress" and comes back `claimed` (kind "set") with an empty
-- `prior`.
--
-- ── Idempotence ───────────────────────────────────────────────────────────
--
-- A re-run of the same command (a fred re-send) either answers `in_progress`,
-- because its own first run's token is younger than `in_progress_ms`, or
-- re-adds the same token (`HSETNX`, so its time is kept) and answers `claimed`
-- with that token left out of `prior`. Either way it removes nothing.
--
-- ── Return value ─────────────────────────────────────────────────────────
--
-- `{ verdict, members, prior, kind }`, always four elements:
--   verdict  `"claimed"`, `"in_progress"` or `"empty"`;
--   members  the carrier's members on `claimed`, verbatim; empty otherwise;
--   prior    the tokens recorded before this call on `claimed`; empty
--            otherwise;
--   kind     the carrier's type, `"hash"` (this build's) or `"set"` (an older
--            build's); `""` on `empty`.

local carrier  = KEYS[1]
local attempts = KEYS[2]

local new_token = ARGV[1]
local in_progress_ms = tonumber(ARGV[2])
if new_token == nil or new_token == "" or in_progress_ms == nil or in_progress_ms < 0 then
  return redis.error_reply("ENCODING: ARGV must be a new attempt token and a non-negative in-progress age in ms")
end

local kind = redis.call("TYPE", carrier)["ok"]
if kind == "none" then
  return { "empty", {}, {}, "" }
end
if kind ~= "hash" and kind ~= "set" then
  return redis.error_reply("WRONGTYPE_KEY: " .. carrier .. " holds a " .. kind .. ", expected a hash or a set")
end
local count
if kind == "hash" then
  count = redis.call("HLEN", carrier)
else
  count = redis.call("SCARD", carrier)
end
if count == 0 then
  return { "empty", {}, {}, "" }
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local recorded = redis.call("HGETALL", attempts)
local prior = {}
local newest = nil
for i = 1, #recorded, 2 do
  local token = recorded[i]
  local added_ms = tonumber(recorded[i + 1])
  if added_ms ~= nil and (newest == nil or added_ms > newest) then
    newest = added_ms
  end
  if token ~= new_token then
    prior[#prior + 1] = token
  end
end

if newest ~= nil and now_ms - newest < in_progress_ms then
  return { "in_progress", {}, {}, "" }
end

redis.call("HSETNX", attempts, new_token, now_ms)

local members
if kind == "hash" then
  members = redis.call("HKEYS", carrier)
else
  members = redis.call("SMEMBERS", carrier)
end
return { "claimed", members, prior, kind }
