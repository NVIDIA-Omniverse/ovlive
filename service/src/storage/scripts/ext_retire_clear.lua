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

-- ext_retire_clear.lua — delete a retiring object's carrier and its attempt
-- tokens once one of its attempts is known to have LANDED (b.yuo Part R).
--
-- ── Where this sits ───────────────────────────────────────────────────────
--
-- The carrier (`le:{oid}:obj:extret`) records a retirement's owed release
-- members, and `le:{oid}:obj:extret:attempts` the token of every attempt to
-- enqueue them, each written before its attempt is sent. Once an attempt is
-- known to have landed on the `{gc}` slot — the retirement's own attempt-mode
-- enqueue answered `enqueued` or `already`, or the orphan sweep's resolve found
-- a prior token `landed`, or the sweep's own re-drive landed — the obligation
-- is paid and the record can go. This script removes it, under the token of
-- the attempt that proved it.
--
-- The orphan sweep also calls it under its OWN new token (the one its re-token
-- just added, never sent) to DROP an older build's SET carrier none of whose
-- recorded attempts landed (b.1jt): this build never pays a SET carrier, so it
-- deletes the carrier and its tokens with nothing enqueued, leaving
-- `ext_gc_scheduled` as it is — a leak, which the reference-count reconciler
-- repairs. The checks below apply unchanged: a drain that re-tokened since
-- makes it `superseded`.
--
-- The script works on a carrier of either type (it only tests `EXISTS` and
-- `DEL`s); this build's carriers are HASHes, an older build's SETs.
--
--   * token absent from the HASH → `gone`, nothing written. The carrier and
--     its tokens are already gone — removed by a drain, by an older build's
--     claim, or by an earlier copy of this same clear (the verdict does not
--     say which).
--   * carrier absent → `gone`, nothing written. Only an older build's
--     destructive claim (of an older build's SET carrier; it cannot read
--     this build's HASH) removes a carrier and leaves its token HASH behind,
--     and that lone HASH may then be the only `{object_id}` evidence that this
--     incarnation's obligation was paid — deleting it here would let a
--     re-derivation of the same incarnation (its UNLINK failed, the header
--     survived) pay a second time. Only the incarnation boundary may remove
--     it, and nothing in this build does: b.9ud sub-task 3 plans the only safe
--     remover (the restore's residue wipe, extended to void the tokens and
--     delete the HASH — not built).
--   * a token NEWER than this one is recorded → `superseded`, nothing written.
--     A drain has re-tokened the carrier since this attempt's token was added,
--     and is mid-decision; it resolves this attempt's token itself (finding it
--     `landed`) and clears then. Clearing under it here would not be unsafe —
--     its settle re-checks the token set and stops — but it would pull the
--     record out from under a drain that owns it.
--   * otherwise `DEL carrier attempts` → `cleared`. The token HASH is deleted
--     only here and in a settle that cancels, both in the same step as a
--     carrier that exists.
--
-- "Newer" is decided without clocks: a token is newer than this one when it is
-- neither this token nor one of the `older` tokens the caller passes — the
-- tokens it knows were recorded before this one. A retirement passes none: its
-- begin recorded its token into an empty HASH, so every other token is a
-- drain's later re-token. A drain passes the `prior` its re-token returned.
--
-- It never touches `ext_gc_scheduled`. A paid obligation keeps the flag even if
-- the object id is live again (policy 12): clearing it would re-arm a
-- re-derivation that pays the same members a second time. A dropped older
-- carrier keeps it for the same reason: its obligation may have been paid.
--
-- `cleared` alone never licenses removing the token's `landed` record from
-- `le:ext:{gc}:enqueue_attempt`: the record is also the only guard against a
-- later SEND under the token (a retirement's second attempt, or a copy fred
-- writes late). The caller removes it at once only when it is the token's own
-- sender, its single send under the token replied cleanly, and this script
-- then answered `cleared` (F1); every other record is left to the GC loop's
-- attempt audit, which removes it after V (F2).
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (both `{object_id}`-tagged: single-slot, and keyed for the b.289
-- fence):
--   [1]  carrier   le:{oid}:obj:extret
--   [2]  attempts  le:{oid}:obj:extret:attempts
--
-- ARGV:
--   [1]     token  — the attempt known to have landed
--   [2..]   older  — tokens the caller knows were recorded before it (may be
--           none)
--
-- ── Idempotence ───────────────────────────────────────────────────────────
--
-- A re-run after `cleared` answers `gone` and writes nothing.
--
-- ── Return value ─────────────────────────────────────────────────────────
--
-- The verdict string: `"cleared"`, `"superseded"` or `"gone"`.

local carrier  = KEYS[1]
local attempts = KEYS[2]

local token = ARGV[1]
if token == nil or token == "" then
  return redis.error_reply("ENCODING: ARGV[1] must be the landed attempt's token")
end

if redis.call("HEXISTS", attempts, token) == 0 then
  return "gone"
end

if redis.call("EXISTS", carrier) == 0 then
  return "gone"
end

local known = { [token] = true }
for i = 2, #ARGV do
  known[ARGV[i]] = true
end

local recorded = redis.call("HKEYS", attempts)
for i = 1, #recorded do
  if not known[recorded[i]] then
    return "superseded"
  end
end

redis.call("DEL", carrier, attempts)
return "cleared"
