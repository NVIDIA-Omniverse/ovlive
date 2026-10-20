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

-- ext_retire_begin.lua — record a retiring object's owed external-value
-- release members, the token of the attempt about to enqueue them, and
-- `ext_gc_scheduled`, in one step, BEFORE the enqueue is sent (b.yuo Part R).
--
-- ── What this is for ──────────────────────────────────────────────────────
--
-- `schedule_old_object_ext_gc` (`storage::layer`) owes one `ref_count`
-- decrement per hash a retiring object references. It enqueues them onto the
-- `{gc}` slot, and an enqueue whose reply is lost has an unknown outcome. Before
-- Part R the members were written to the object's carrier only AFTER both
-- enqueue attempts failed, and the flag that stops a re-derivation only AFTER a
-- successful one — so a landed-but-unacknowledged enqueue, a crash between the
-- enqueue and the flag, or two concurrent retirements could each pay the same
-- members twice, once their replay records were torn down.
--
-- This script writes the intent first. In one atomic step, on the object's own
-- slot:
--
--   1. `HSET header ext_gc_scheduled 1` — every later
--      `schedule_old_object_ext_gc` on this incarnation (a `delete_layer`
--      cascade re-run, the orphan sweep's revisit of a surviving header, a
--      concurrent second retirement) short-circuits and derives nothing;
--   2. `HSET carrier <member> 1 …` — the record of the obligation, which
--      outlives the teardown that follows. The carrier is a HASH (member → 1,
--      b.1jt) at the key where an older build writes a SET: an older build's
--      carrier claim (`SCARD` / `SMEMBERS`) and reconciler fail `WRONGTYPE` on
--      it before writing anything, so no older build can pay or delete an
--      obligation this build recorded, and a SET found there is an older
--      build's carrier, which this build never pays;
--   3. `HSET attempts <token> <now_ms>` — the token every attempt-mode enqueue
--      of this retirement carries, so the orphan sweep's drain can ask the
--      `{gc}` slot whether it landed before it re-drives anything.
--
-- With no members (an object that references no external value) only step 1
-- runs: the flag's contract is "GC has been considered for this incarnation",
-- not "something was enqueued", and no token is recorded because no attempt
-- will be sent.
--
-- ── Verdicts, in the order they are checked ──────────────────────────────
--
--   * `gone`    — the header is absent, so nothing is written. The object was
--     torn down after the caller read it; whichever retirement removed it began
--     first and owns the obligation. (An ungated `HSET` here would also create a
--     header holding nothing but the flag — a phantom the orphan sweep would
--     rediscover, the b.1ah re-adoption hazard.)
--   * `begun`   — the flag is already set AND the attempt HASH holds this
--     call's token: a re-run of this same command (a fred re-send). Nothing is
--     written again.
--   * `already` — the flag is already set by an earlier begin (or by an older
--     build's success path or deferral). Nothing is written; that retirement
--     owns the obligation, and enqueuing here would pay it twice.
--   * `busy`    — the flag is NOT set, but a carrier or attempt tokens from an
--     earlier lifecycle of this `object_id` are still outstanding (a carrier
--     that outlived a teardown when a restore re-materialised the same id; an
--     older build's untokened carrier). Nothing is written: merging this
--     retirement's members into that carrier would tie them to attempts this
--     retirement never made. The caller counts the references as a failed
--     enqueue; the orphan sweep settles the earlier carrier on its own.
--
--     That includes a token HASH with NO carrier. Only an older build creates
--     one: its destructive claim deletes the carrier, and its cancel also
--     clears the flag, but neither touches the HASH. The HASH may then be the
--     only evidence that this incarnation's obligation was already paid (an
--     attempt of it landed), so begin must neither delete it nor begin past it
--     — a re-derivation of the same incarnation would pay a second time. The
--     cost is that every later retirement of this id answers `busy` (counted,
--     leak direction). Only the incarnation boundary may remove the HASH, and
--     nothing in this build does: b.9ud sub-task 3 plans the only safe remover
--     (the restore's residue wipe, extended to void the tokens and delete the
--     HASH — not built).
--
--     Begin's idempotence holds within one incarnation. A copy of a begin
--     still in transit across a DR restore of the same object id can land on
--     the restored header (no flag, no HASH) and answer `begun` there — a
--     residual b.9ud plans to close with an incarnation stamp; not built.
--   * `begun`   — otherwise, after the writes above.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (all `{object_id}`-tagged, so the script is single-slot and keyed — the
-- b.289 fence against zero-key EVALSHA holds structurally):
--   [1]  header    le:{oid}:obj:hdr
--   [2]  carrier   le:{oid}:obj:extret
--   [3]  attempts  le:{oid}:obj:extret:attempts
--
-- ARGV:
--   [1]     token — the attempt token, 32 lowercase hex characters.
--   [2..]   members — the `le:ext:{gc}:pending` member strings this retirement
--           owes, `{hash_hex}:{object_id}:ret`, verbatim. May be empty.
--
-- The members are stored uninterpreted: the drain re-enqueues the same bytes.
--
-- `now_ms` is `redis.call("TIME")` on the object's primary, the same clock the
-- drain's re-token reads when it asks whether the newest token is still young
-- enough to belong to a retirement in progress.
--
-- ── Idempotence ───────────────────────────────────────────────────────────
--
-- A re-send answers `begun` for its own token and writes nothing more.
--
-- ── Return value ─────────────────────────────────────────────────────────
--
-- The verdict string: `"begun"`, `"already"`, `"busy"` or `"gone"`.

local header   = KEYS[1]
local carrier  = KEYS[2]
local attempts = KEYS[3]

local token = ARGV[1]
if token == nil or token == "" then
  return redis.error_reply("ENCODING: ARGV[1] must be the attempt token")
end

if redis.call("EXISTS", header) == 0 then
  return "gone"
end

if redis.call("HGET", header, "ext_gc_scheduled") == "1" then
  if redis.call("HEXISTS", attempts, token) == 1 then
    return "begun"
  end
  return "already"
end

if redis.call("HLEN", attempts) > 0 or redis.call("EXISTS", carrier) == 1 then
  return "busy"
end

redis.call("HSET", header, "ext_gc_scheduled", "1")

if #ARGV > 1 then
  -- Chunked rather than one `HSET` of every field: the member count is the
  -- number of distinct external hashes one object holds, which is unbounded
  -- in principle, and `unpack` over a large table overflows the Lua C stack.
  -- 250 members per call is 500 field/value arguments.
  local i = 2
  while i <= #ARGV do
    local last = math.min(i + 249, #ARGV)
    local fields = {}
    for j = i, last do
      fields[#fields + 1] = ARGV[j]
      fields[#fields + 1] = "1"
    end
    redis.call("HSET", carrier, unpack(fields))
    i = last + 1
  end
  local t = redis.call("TIME")
  local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
  redis.call("HSET", attempts, token, now_ms)
end

return "begun"
