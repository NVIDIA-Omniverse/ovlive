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
-- A verbatim copy of `src/storage/scripts/ext_retire_retry_claim.lua` at commit bf32b533
-- (0.2.0-rc.47, the last build before b.yuo's record-aware GC: a record-blind
-- GC with an unmarked leader lock).
-- Every earlier release runs the same code: 0.1.0 and 0.1.1 differ from it
-- only in headers and comments. `tests/integration/storage/b_1jt_rollout_tests.rs`
-- runs it against the store exactly as an older build's replica would, to
-- prove this build's state is safe in a mixed-build store. Only this block was
-- added; everything below it is the older build's file, unchanged.

-- ext_retire_retry_claim.lua — atomically claim a retiring object's deferred
-- external-value release obligations off its own carrier, or CANCEL them
-- because the object id is live again (b.yda).
--
-- ── What this is for ──────────────────────────────────────────────────────
--
-- When `schedule_old_object_ext_gc` cannot enqueue a retiring object's owed
-- `ref_count` decrements onto `le:ext:{gc}:pending` — the natural Cluster shape
-- being "the object-slot reads succeeded, the {gc}-slot SADD failed" — it writes
-- the formatted pending members to `le:{oid}:obj:extret` on the OBJECT's slot
-- and lets the structural teardown proceed. The orphan sweep's retire-retry
-- phase later re-drives them. This script is that phase's claim step.
--
-- ── The invariant this script enforces ────────────────────────────────────
--
-- **A deferred retirement obligation is owed only while its `object_id` is
-- dead.** An id that is live again at claim time has INHERITED the standing,
-- unreleased `+1` per hash that those members represent, so the obligation is
-- CANCELLED, never paid.
--
-- The inheritance is not a hypothetical. A base-only checkpoint restore
-- (`RestorePayload::Ompf`) pre-stores NOTHING — `checkpoint::apply` binds
-- `ext_values = Vec::new()` on that arm and the pre-store block is gated on
-- `!ext_values.is_empty()` — while `decode_checkpoint_ompf` still rebuilds
-- `FieldValue::External` rows from the base blob's hash references. So the
-- re-materialised object at the SAME `object_id` references exactly the hashes
-- the dead incarnation did, holding exactly the `+1` the dead incarnation held
-- and never acquiring a new one (that is why the DR-restore materialiser — the
-- `SurplusPolicy::Suppress` entry point in `storage::layer` — emits no
-- `surplus_acquire`). Draining the old
-- carrier against it would be an UNMATCHED decrement: `ext_meta` + `ext_data`
-- deleted under a live reference. The same over-release is reachable without a
-- restore at all — a deferral whose caller's structural UNLINK then failed
-- permanently leaves `obj:hdr` (carrying `ext_gc_scheduled=1`), the value rows
-- and `obj:ext` standing, and b.1ah's re-adoption puts that id back in service.
--
-- Cancelling instead LEAKS the references in the worst case (nothing decrements
-- a `ref_count` that is genuinely owed), and a leak is drained by the
-- reference-count reconciler (`storage::ext_reconcile`), which enumerates live
-- objects by `obj:hdr` and repairs the surplus. An over-release deletes data and
-- is drained by nothing. The gate is deliberately biased that way.
--
-- ── Why `obj:ext` is in the gate as well as `obj:hdr` ─────────────────────
--
-- A restore's residue wipe (`checkpoint::residue`) UNLINKs `obj:hdr` but
-- deliberately WITHHOLDS `obj:ext` from the inventory, because the DR-restore
-- materialiser (the `SurplusPolicy::Suppress` entry point in `storage::layer`)
-- reads that index as the `before` image for the `rewrite` releases it owes. In
-- the wipe → materialise window the header is therefore ABSENT while
-- `obj:ext` still stands, and a claim landing there must still cancel: the
-- standing `+1` is inherited by that materialiser's rewrite pass either way — a
-- hash the restored state drops is released exactly once there, and a hash it
-- keeps is inherited by the live incarnation.
--
-- `obj:ext` standing without a header never means "dead". On a genuinely dead
-- object both keys go in ONE `UNLINK`: they are elements 0 and 6 of
-- `KeySchema::object_cleanup_keys`, and `UNLINK_CHUNK_SIZE` is 500, so they are
-- always in the same chunk however large the object is — for all three
-- retirement/wipe routes, which share that inventory. Either both are gone or
-- the teardown never ran.
--
-- ── Why the `HDEL` is load-bearing, not tidiness ──────────────────────────
--
-- In the surviving-header shape above, the header carries `ext_gc_scheduled=1`.
-- Left in place, the object's eventual REAL retirement short-circuits on it
-- (`storage::layer::schedule_old_object_ext_gc`) and releases nothing — turning
-- the over-release this gate prevents into a standing leak instead, one only the
-- reference-count reconciler's hourly pass would drain. Clearing
-- the flag hands the next retirement its re-derivation back: it re-reads the
-- rows and `obj:ext` and enqueues for real.
--
-- **The clear happens only when a deferred obligation is actually cancelled** —
-- it is gated on the carrier having had members, not merely on the header being
-- live. A work-list entry can name a carrier that is already gone: every
-- `recon_retire` failure arm in `storage::orphan_sweep::drain_retire_carrier`
-- leaves the entry registered after the carrier was drained or cancelled, and
-- the next tick re-visits it. On such an entry there is nothing to hand back,
-- while the `ext_gc_scheduled=1` the live header carries may well belong to a
-- retirement whose enqueue SUCCEEDED. Clearing it would re-arm
-- `schedule_old_object_ext_gc` to re-derive those same members and enqueue them
-- a second time — a second decrement against a reference already given up, once
-- the GC has drained the first. That is the exact over-release this script
-- exists to refuse, reached from the direction the gate does not look at.
-- No carrier ⇒ no deferral ⇒ nothing to cancel ⇒ the flag is not ours to touch.
--
-- That is also what makes the cancel safe against an in-flight retirement. A
-- deferral that has written the carrier and flagged a still-live header, but
-- whose structural UNLINK has not run yet, can be cancelled by a concurrent
-- sweep — and the flag clear is what lets the cascade's re-run (or the next
-- orphan-sweep visit) re-derive the same members and pay them properly. Only if
-- the teardown completes between the cancel and any re-derivation is anything
-- lost, and that loss is a leak.
--
-- So `ext_gc_scheduled` is a CLEARABLE field, and its contract reads "GC has
-- been considered *for this incarnation*". There are three writers: this
-- script's `HDEL`, the success-path `HSET` in `schedule_old_object_ext_gc`, and
-- the deferral `HSET` in `ext_retire_defer.lua`. Every reader treats an absent
-- field as "not scheduled", so a clear can only cause re-derivation, never a
-- skipped one.
--
-- ── The one residual this gate does NOT close ─────────────────────────────
--
-- `claim (header AND index both absent) → enqueue onto pending → a restore
-- lands at that id → the GC drains the member and decrements`. The claim was
-- correct when it was taken and the restore is what made it stale. It is not
-- closable from inside the `{object_id}`-slot discipline, because the enqueue
-- lands on the `{gc}` slot and no single atomic step spans both. It is the
-- pre-existing "a checkpoint restore materialises external references without
-- acquiring them" defect, filed as its own checkpoint-lane Issue; this script
-- narrows the window to the flight time of one enqueue but does not remove it.
--
-- ── Why the claim must be atomic ──────────────────────────────────────────
--
-- Two drainers that each `SMEMBERS` the carrier and each `SADD pending` are
-- safe only while the members are still sitting in `pending`: SADD collapses
-- equal members, so the second copy costs nothing. They stop being safe the
-- moment the GC has drained the first copy — the second copy is then a fresh
-- member owing a SECOND decrement against a reference that was already given
-- up, which is an over-release and, with the GC behind it, a blob deleted under
-- a live reference. Reading and removing in one atomic step is what makes the
-- claim exclusive: exactly one caller ever sees a given member set.
--
-- The `DEL` is deliberately inside the same script rather than issued by the
-- caller after a successful `SADD pending`. Doing it afterwards would leave the
-- window above open for the whole round-trip. The caller's failure recovery
-- runs the other way instead: if its `SADD pending` fails it writes the claimed
-- members BACK onto the carrier, which leaks toward retry (the obligation is
-- re-claimable next tick) rather than toward loss.
--
-- The liveness gate is inside the same step for the same reason: an `EXISTS`
-- pair issued by the caller could observe a dead object and have a restore land
-- before the `SMEMBERS`, which is precisely the interleaving this exists to
-- refuse.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  carrier  (le:{object_id}:obj:extret)
--   [2]  header   (le:{object_id}:obj:hdr)
--   [3]  index    (le:{object_id}:obj:ext)
--
-- ARGV: none.
--
-- All three keys carry the same `{object_id}` hash tag, so this is a
-- single-slot script; having keys at all is also what satisfies the b.289 fence
-- against zero-key EVALSHA, structurally rather than by convention.
--
-- ── Return value ─────────────────────────────────────────────────────────
--
-- `{ verdict, count, members }`, always three elements:
--
--   verdict  "claimed" — the id is dead, the obligation is the caller's to
--            re-enqueue. "cancelled" — the id is live again (or a restore is
--            mid-flight, or a teardown never completed) and NOTHING may be
--            enqueued; when the carrier held members they have been removed and
--            `ext_gc_scheduled` cleared, and when it held none NOTHING was
--            written at all (see the `HDEL` section above).
--   count    how many members were on the carrier. On "cancelled" this is the
--            whole payload: the caller reports it and drops it. Zero there means
--            the work-list entry outlived its carrier — a no-op visit, not a
--            cancellation.
--   members  the claimed member set on "claimed"; ALWAYS EMPTY on "cancelled",
--            so a caller that ignores the verdict cannot enqueue a cancelled
--            obligation by accident.
--
-- The verdict is what distinguishes a cancel from "the carrier was already
-- gone" — before b.yda's revision both answered with an empty array and the
-- caller could not tell them apart. A zero `count` under "claimed" still means
-- exactly that: another drainer got there first, or the work-list entry names a
-- carrier whose write never landed.
--
-- Members are `le:ext:{gc}:pending` member strings
-- (`{hash_hex}:{object_id}:{event}`), returned verbatim. This script does not
-- interpret them and must not: the caller re-enqueues the same bytes, and
-- byte-identity is what makes a duplicate drain collapse in the pending Set.

local carrier = KEYS[1]
local header  = KEYS[2]
local index   = KEYS[3]

local header_live = redis.call("EXISTS", header) == 1
local index_live  = redis.call("EXISTS", index) == 1

if header_live or index_live then
  -- The id is live again. Drop the obligation whole; do not hand the caller
  -- anything it could enqueue.
  local count = redis.call("SCARD", carrier)
  if count > 0 then
    redis.call("DEL", carrier)
    if header_live then
      -- BOTH gates are load-bearing, and the `count > 0` one is the outer of
      -- the two on purpose. `HDEL`ing the flag is how a CANCELLED obligation
      -- hands the object's next retirement its re-derivation back, so it may
      -- only fire when there was a deferred obligation to cancel. With no
      -- carrier there was none: the work-list entry is a leftover (any of
      -- `recon_retire`'s failure arms in `storage::orphan_sweep` leaves one
      -- behind after the carrier is already drained or cancelled), and the
      -- `ext_gc_scheduled=1` on this live header belongs to a retirement whose
      -- enqueue SUCCEEDED. Clearing it there re-arms `schedule_old_object_ext_gc`
      -- to re-derive the same members and enqueue them a SECOND time — a second
      -- decrement once the GC has drained the first, which is the over-release
      -- this whole script exists to refuse.
      --
      -- The inner `header_live` gate is the milder one: the header-absent arm
      -- is the restore's wipe window, where there is no flag to clear, and an
      -- `HDEL` there would only write a replicated no-op.
      redis.call("HDEL", header, "ext_gc_scheduled")
    end
  end
  return { "cancelled", count, {} }
end

local members = redis.call("SMEMBERS", carrier)

-- Guard the DEL rather than issuing it unconditionally: an unconditional DEL on
-- an absent key is harmless but writes a replicated no-op command for every
-- work-list entry whose carrier is already gone, which is the common case once
-- a backlog drains.
if #members > 0 then
  redis.call("DEL", carrier)
end

return { "claimed", #members, members }
