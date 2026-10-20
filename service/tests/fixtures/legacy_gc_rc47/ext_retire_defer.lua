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
-- A verbatim copy of `src/storage/scripts/ext_retire_defer.lua` at commit bf32b533
-- (0.2.0-rc.47, the last build before b.yuo's record-aware GC: a record-blind
-- GC with an unmarked leader lock).
-- Every earlier release runs the same code: 0.1.0 and 0.1.1 differ from it
-- only in headers and comments. `tests/integration/storage/b_1jt_rollout_tests.rs`
-- runs it against the store exactly as an older build's replica would, to
-- prove this build's state is safe in a mixed-build store. Only this block was
-- added; everything below it is the older build's file, unchanged.

-- ext_retire_defer.lua — park a retiring object's owed external-value release
-- members on its own carrier AND mark the object ext-GC-considered, atomically
-- (b.yda).
--
-- ── What this is for ──────────────────────────────────────────────────────
--
-- When `schedule_old_object_ext_gc` cannot enqueue a retiring object's owed
-- `ref_count` decrements onto `le:ext:{gc}:pending` — the natural Cluster shape
-- being "the object-slot reads succeeded, the {gc}-slot SADD failed" — it defers
-- them onto `le:{oid}:obj:extret`, on the object's OWN slot, and lets the
-- structural teardown proceed. The orphan sweep's retire-retry phase re-drives
-- them later. This script is that deferral.
--
-- ── Why the two writes must be one step ───────────────────────────────────
--
-- The carrier alone is not enough, because the deferral arm returns `Err` and
-- `delete_layer`'s cascade can re-run the whole retirement (an INDEPENDENT
-- transient in step 5 or step 6 drives `retry_aborted_with_jitter`). A re-run
-- re-reads the object's rows and `obj:ext`, re-derives the SAME members, and —
-- if the `{gc}` slot has recovered by then — enqueues them onto `pending` for
-- real. The sweep later drains the carrier and enqueues them a SECOND time,
-- against a hash whose reference was already given up. That is an over-release:
-- the loss direction, with the GC behind it deleting a blob under a live
-- reference.
--
-- `ext_gc_scheduled` is what stops the re-derivation. Its contract is "GC has
-- been CONSIDERED for this INCARNATION of this object" (b.j9j item 1, narrowed
-- by b.yda), not "GC has been enqueued", and a deferral is exactly that
-- consideration: the obligation is recorded, on a durable key, and owned by the
-- sweep. So the flag is written here, and every later
-- `schedule_old_object_ext_gc` on this object — the delete cascade's re-run, and
-- the orphan sweep's revisit of a surviving header — short-circuits on it and
-- derives nothing. The carrier becomes the single record of the debt.
--
-- The "for this incarnation" half is why the field is CLEARABLE. There is a
-- third writer: `ext_retire_retry_claim.lua` `HDEL`s it when it CANCELS a
-- deferred obligation — one that is really there, on a carrier holding members —
-- because the `object_id` turned out to be live again. A header that outlived
-- its retirement must not go on suppressing the next one: the clear is what
-- hands `schedule_old_object_ext_gc` its re-derivation back, and without it a
-- cancel would convert an over-release into a standing leak, drained only by the
-- reference-count reconciler's hourly pass.
--
-- That third writer clears NOTHING when the carrier is empty, and the asymmetry
-- is deliberate. Handing the re-derivation back is only safe against a debt that
-- was cancelled; a flag standing over an empty carrier belongs to a retirement
-- that ENQUEUED, and re-arming it there buys a second enqueue of members the GC
-- may already have drained — the over-release, from the other direction.
-- Every reader treats an absent field as "not scheduled", so a clear can only
-- cause a re-derivation, never a skipped one.
--
-- Writing the two with two round trips would leave the torn state this script
-- exists to prevent: a carrier written with no flag is a debt recorded on the
-- carrier that a cascade re-run is still free to re-derive and pay a second
-- time. Both keys carry the `{object_id}` hash tag, so they are one slot and one
-- atomic step is available for free.
--
-- ── The header-existence gate, and the residual it leaves ────────────────
--
-- The `HSET` is gated on the header still existing, matching the bare-`HSET`
-- site this script replaces on the success path (`EXISTS` then `HSET`). An
-- ungated `HSET` on an already-torn-down object would CREATE a header holding
-- nothing but `ext_gc_scheduled=1` — a phantom the orphan sweep would then
-- discover as an orphan, and the re-adoption hazard b.1ah is about.
--
-- The residual that leaves is pre-existing and unchanged by b.yda: if a teardown
-- removed `obj:hdr` (element 0 of `KeySchema::object_cleanup_keys`) but a later
-- UNLINK chunk failed and left `obj:ext` standing, a re-run finds no flag to
-- short-circuit on and can re-derive from the surviving index. That hole belongs
-- to the b.j9j guard itself — its `HSET` has always been `EXISTS`-gated — and is
-- not widened here: the deferral records no more than the successful path does.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  carrier  (le:{object_id}:obj:extret)
--   [2]  header   (le:{object_id}:obj:hdr)
--
-- Both keys carry the same `{object_id}` hash tag, so this is a single-slot
-- script; two keys also satisfies the b.289 fence against zero-key EVALSHA
-- structurally rather than by convention.
--
-- ARGV:
--   [1..] members  (the `le:ext:{gc}:pending` member strings this retirement
--                   owed — `{hash_hex}:{object_id}:{event}`, verbatim. At least
--                   one; the caller does not call this with an empty release
--                   set.)
--
-- The members are stored and returned uninterpreted. Byte-identity is what makes
-- a duplicate drain collapse in the pending Set rather than owe a second
-- decrement, so nothing here may normalise them.
--
-- ── Return value ─────────────────────────────────────────────────────────
--
-- `{added, flagged}`:
--   added    number of members this call newly added to the carrier (SADD's own
--            return). Zero is normal on a retry of the same deferral.
--   flagged  1 if `ext_gc_scheduled=1` was written, 0 if the header was already
--            gone. Reported rather than swallowed so the caller can log a
--            deferral that landed without its guard.

local carrier = KEYS[1]
local header  = KEYS[2]

if #ARGV == 0 then
  return { 0, 0 }
end

-- Chunked rather than one `unpack(ARGV)`: the member count is the number of
-- distinct external hashes one object holds, which is unbounded in principle,
-- and `unpack` over a large table overflows the Lua C stack. The chunking is
-- invisible to the caller — `SADD` is idempotent per member, so a partial
-- application followed by a retry converges on the same Set.
local added = 0
local i = 1
while i <= #ARGV do
  local last = math.min(i + 499, #ARGV)
  added = added + redis.call("SADD", carrier, unpack(ARGV, i, last))
  i = last + 1
end

local flagged = 0
if redis.call("EXISTS", header) == 1 then
  redis.call("HSET", header, "ext_gc_scheduled", "1")
  flagged = 1
end

return { added, flagged }
