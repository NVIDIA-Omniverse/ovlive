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

-- ext_retire_settle.lua — decide whether a retiring object's recorded release
-- obligation is paid by a re-drive or CANCELLED, once every RECORDED earlier
-- enqueue attempt resolved void (b.yuo Part R; the liveness gate is b.yda's).
-- Only this build's carriers — HASHes, every attempt of which records its
-- token before it is sent — are settled here (b.1jt). An older build's SET
-- carrier, which may record none of its in-doubt attempts, is never paid: the
-- drain clears it if a recorded attempt landed and otherwise drops it, and a
-- settle that finds a carrier of any type but HASH answers `changed`.
--
-- ── Where this sits ───────────────────────────────────────────────────────
--
-- The orphan sweep's drain (`storage::orphan_sweep::drain_retire_carrier`)
-- re-tokens a carrier (`ext_retire_retry_claim.lua`, which adds `new_token` and
-- returns the tokens that were there before, `prior`), resolves every prior
-- token on the `{gc}` slot (`ext_enqueue_attempt_resolve.lua`), and calls this
-- only for this build's HASH carrier, and only when every recorded attempt
-- resolved `void` — no attempt ran, and none ever will, because this build
-- records every attempt's token before sending it. A carrier with any
-- `landed` attempt is never settled here: it was paid, and the drain only
-- clears it.
--
-- ── The invariant the gate enforces ───────────────────────────────────────
--
-- **A recorded retirement obligation is owed only while its `object_id` is
-- dead.** An id that is live again has INHERITED the standing, unreleased `+1`
-- per hash the carrier's members represent, so the obligation is cancelled,
-- never paid.
--
-- The inheritance is not hypothetical. A base-only checkpoint restore
-- (`RestorePayload::Ompf`) pre-stores NOTHING — `checkpoint::apply` binds
-- `ext_values = Vec::new()` on that arm — while `decode_checkpoint_ompf` still
-- rebuilds `FieldValue::External` rows from the base blob's hash references, so
-- the re-materialised object at the SAME `object_id` references exactly the
-- hashes the dead incarnation did and holds exactly its `+1` (which is why the
-- DR-restore materialiser — the `SurplusPolicy::Suppress` entry point in
-- `storage::layer` — emits no `surplus_acquire`). Paying the carrier against it
-- would be an UNMATCHED decrement: `ext_meta` + `ext_data` deleted under a live
-- reference. A retirement whose structural UNLINK failed permanently reaches the
-- same shape without a restore, through b.1ah's re-adoption.
--
-- Cancelling instead LEAKS in the worst case, and a leak is drained by the
-- reference-count reconciler (`storage::ext_reconcile`). An over-release
-- deletes data and is drained by nothing. The gate is biased that way.
--
-- **Why `obj:ext` is in the gate as well as `obj:hdr`.** A restore's residue
-- wipe (`checkpoint::residue`) UNLINKs `obj:hdr` but WITHHOLDS `obj:ext`,
-- which the DR-restore materialiser reads as its `before` image. In that
-- wipe → materialise window the header is absent while the index stands, and a
-- settle landing there must still cancel. On a genuinely dead object both keys
-- go in ONE `UNLINK`, the teardown's first
-- (`KeySchema::object_teardown_first_keys`), so `obj:ext` without a header
-- never means "dead".
--
-- ── Why the cancel clears `ext_gc_scheduled` ──────────────────────────────
--
-- In the surviving-header shape the header carries `ext_gc_scheduled=1`, written
-- by the retirement's begin. Left in place, the object's eventual REAL
-- retirement would short-circuit on it and release nothing — turning the
-- over-release this gate prevents into a standing leak. Clearing it hands that
-- retirement its re-derivation back. It is safe because the cancel happens only
-- after every attempt this carrier recorded resolved `void`, and this build
-- records every attempt: no attempt of this obligation was enqueued, so a
-- re-derivation pays it once.
--
-- A slow retirement that is still running when a drain overtakes it (past
-- `RETIRE_INTENT_IN_PROGRESS_AGE`) can be cancelled here while its object is
-- still live: its own enqueue then finds its token `void` and writes nothing,
-- and if its teardown completes afterwards the references leak. Leak
-- direction, and the reconciler's to drain.
--
-- ── Why the token set is re-checked ──────────────────────────────────────
--
-- Between the re-token and this call the carrier can change hands: a
-- retirement can clear it (its attempt landed), another drain can re-token it
-- or clear it. Any of those changes the token set, so this script settles only
-- if the set is still exactly `prior ∪ {new_token}`; otherwise it answers
-- `changed`, writes nothing, and the drain looks again next tick.
--
-- It also answers `changed`, writing nothing, when the carrier itself is gone
-- while the token set is not. Only an older build's destructive claim does
-- that, and only to an older build's SET carrier (this build never settles
-- one, and an older claim cannot read a HASH) — it took the carrier and
-- enqueues the members itself, or cancelled them — so paying here would be a
-- second enqueue; and
-- the lone token HASH it leaves may be the only evidence that this
-- incarnation's obligation was paid, so it is not deleted either. Only the
-- incarnation boundary may remove it, and nothing in this build does: b.9ud
-- sub-task 3 plans the only safe remover (the restore's residue wipe, extended
-- to void the tokens and delete the HASH — not built).
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (all `{object_id}`-tagged: single-slot, and keyed for the b.289 fence):
--   [1]  carrier   le:{oid}:obj:extret
--   [2]  attempts  le:{oid}:obj:extret:attempts
--   [3]  header    le:{oid}:obj:hdr
--   [4]  index     le:{oid}:obj:ext
--
-- ARGV:
--   [1]     new_token — the token the drain's re-token added
--   [2..]   prior     — the tokens the re-token returned (may be none)
--
-- ── Idempotence ───────────────────────────────────────────────────────────
--
-- A re-run after a `cancelled` finds the token set gone and answers `changed`;
-- a re-run after `pay` answers `pay` again. Neither writes anything new.
--
-- ── Return value ─────────────────────────────────────────────────────────
--
-- `{ verdict, count }`:
--   `"changed"`   the token set moved, or the carrier is gone or not a HASH;
--                 nothing written (count 0);
--   `"cancelled"` the id is live: carrier and tokens deleted, the flag cleared
--                 off a surviving header; count = the members dropped;
--   `"pay"`       the id is dead: nothing written; count = the members owed.
--                 The caller enqueues them under `new_token`.

local carrier  = KEYS[1]
local attempts = KEYS[2]
local header   = KEYS[3]
local index    = KEYS[4]

if #ARGV < 1 or ARGV[1] == "" then
  return redis.error_reply("ENCODING: ARGV[1] must be the drain's new attempt token")
end

local expected = {}
local expected_n = 0
for i = 1, #ARGV do
  if not expected[ARGV[i]] then
    expected[ARGV[i]] = true
    expected_n = expected_n + 1
  end
end

local current = redis.call("HKEYS", attempts)
if #current ~= expected_n then
  return { "changed", 0 }
end
for i = 1, #current do
  if not expected[current[i]] then
    return { "changed", 0 }
  end
end

-- Only this build's HASH carrier is ever settled: an older build's SET
-- carrier is never paid (the drain clears or drops it instead), so a carrier
-- that is absent or not a HASH answers `changed`.
if redis.call("TYPE", carrier)["ok"] ~= "hash" then
  return { "changed", 0 }
end

local count = redis.call("HLEN", carrier)

local header_live = redis.call("EXISTS", header) == 1
local index_live  = redis.call("EXISTS", index) == 1
if header_live or index_live then
  redis.call("DEL", carrier, attempts)
  if header_live then
    redis.call("HDEL", header, "ext_gc_scheduled")
  end
  return { "cancelled", count }
end

return { "pay", count }
