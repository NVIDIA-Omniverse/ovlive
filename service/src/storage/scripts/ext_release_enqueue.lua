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

-- ext_release_enqueue.lua — enqueue owed external-value releases onto the GC
-- pending set and the release ledger, in one atomic step (b.yuo), refusing
-- replays (b.1jt).
--
-- The single production enqueue of a release member, and the only writer of
-- this build that adds a member to `le:ext:{gc}:pending`. It replaces the bare
-- `SADD le:ext:{gc}:pending` that the `external_values` enqueue issued before
-- the release ledger existed. The pending Set is unchanged —
-- still the membership and idempotence authority, with the same member grammar
-- (`{hash_hex}:{object_id}:{event}`, see `external_values::ExtRelease`) — and
-- this script maintains the three companion keys beside it:
--
--   * `pending:age`   (ZSET)   member → first-enqueue time, Valkey ms;
--   * `pending:bytes` (HASH)   member → the bytes the member accounts for;
--   * `pending:total` (STRING) the sum of `pending:bytes` over pending ∪
--                              inflight — the quantity the release budget
--                              (`LIVE_EDIT_EXT_RELEASE_MAX_BYTES`) bounds.
--
-- Per (member, bytes) pair, in order:
--   0. **Replay refusal (b.1jt).** If the member is NOT in `pending` but is in
--      `inflight` (claimed), in `settled` (its decrement decided, within the
--      replay window), or holds a `pending:bytes` field (accounted and not yet
--      settled), write nothing for it and count it `refused`. Such a member
--      was enqueued once already and its decrement was or is being decided;
--      enqueuing it again could hand it to a GC that cannot recognise a replay
--      — an older build's record-blind sweep — for a second decrement. The
--      case this closes: an older build's sweep, still running when this
--      build took the GC lock, collects a member of this build's and a fred
--      re-send of that member's enqueue then arrives. The refusal lasts from
--      the claim until the member's `settled` entry is retired (at least the
--      replay-record retention U after its settle), the window the replay
--      record covers on the decrement side.
--   1. `SADD pending member`; only when it ADDED the member,
--      `ZADD age NX <now_ms> member`. A replay of an enqueue that is still
--      pending keeps its original age.
--   2. Only when `HEXISTS bytes member` is false: `HSET bytes member <bytes>`
--      and `INCRBY total <bytes>`. A member already accounted — still pending,
--      or claimed (the claim writes `HSETNX bytes member 0`) — is never counted
--      twice; its bytes leave the total once, when it settles
--      (`gc_complete.lua`, the claim's self-heal, an older build's claim drop,
--      or `gc_ledger_audit.lua`).
--
-- `now_ms` is `redis.call("TIME")`, the `{gc}` primary's clock, for the same
-- reason `gc_claim_pending.lua` samples it: every replica must agree on the age
-- of a member regardless of local clock skew.
--
-- ── Attempt mode (b.yuo Part R) ───────────────────────────────────────────
--
-- A retirement's releases, and the orphan sweep's re-drive of them, are sent
-- in ATTEMPT MODE: ARGV starting `"@attempt", K, oid` and a seventh key, the
-- `{gc}` attempt record `le:ext:{gc}:enqueue_attempt`, where K is the
-- attempt's 32-hex token (already recorded on the object's
-- `le:{oid}:obj:extret:attempts` before this was sent) and oid the retiring
-- object. Then, before anything else is written:
--
--   * K recorded `void:…`  → reply `{ "void", 0, 0 }`, write NOTHING. The
--     orphan sweep's resolve (`ext_enqueue_attempt_resolve.lua`) decided this
--     attempt never ran, and a re-drive under a newer token owns the members
--     now, so this copy — a slow retirement, or a command fred writes late —
--     must not enqueue them a second time.
--   * K recorded `landed:…` → reply `{ "already", 0, 0 }`, write nothing. This
--     attempt already enqueued (a fred re-send of the same command).
--   * K absent → run the body below unchanged (refusals included), then
--     `HSET K "landed:<now_ms>:<oid>"` → reply `{ "enqueued", added, refused }`.
--     The attempt is recorded `landed` even when every member was refused: it
--     ran, and a refused member was already owed through its earlier enqueue.
--
-- That check-then-set and the resolve's `HSETNX K void` run on one slot, so
-- each attempt is decided exactly once: it landed before the resolve, or it
-- never lands. Any other recorded value (a newer build's spelling) is refused
-- `ENCODING` with nothing written — the safe reading of an unknown verdict is
-- "do not enqueue".
--
-- Attempt mode is chosen by the first argument, never by the key count: a
-- call whose ARGV[1] is `"@attempt"` must pass the attempt record as KEYS[7],
-- and a seven-key call must start its ARGV with `"@attempt"`; either mismatch
-- is refused `ENCODING` with nothing written, so an attempt can never be read
-- as a member by a caller that forgot the record key. Without attempt mode the
-- commit paths, the reference-count reconciler's repairs and the proptests
-- call it with six keys.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS (all `{gc}`-tagged, so the script is single-slot on a Cluster):
--   [1]  le:ext:{gc}:pending          (SET)
--   [2]  le:ext:{gc}:pending:age      (ZSET)
--   [3]  le:ext:{gc}:pending:bytes    (HASH)
--   [4]  le:ext:{gc}:pending:total    (STRING, integer)
--   [5]  le:ext:{gc}:inflight         (ZSET; read only, for the refusal)
--   [6]  le:ext:{gc}:settled          (ZSET; read only, for the refusal)
--   [7]  le:ext:{gc}:enqueue_attempt  (HASH; attempt mode only)
--
-- ARGV: one or more pairs `member, bytes`, where `bytes` is a canonical
--   non-negative decimal. The Rust caller derives it from `STRLEN
--   le:ext:data:{hash}` (0 for a commit's `surplus_acquire` release, whose
--   decrement cannot free the blob, and for a hash whose `STRLEN` failed).
--   In attempt mode the pairs follow `"@attempt", K, oid`.
--
-- Every pair is validated before the first write: an odd pair count or a
-- `bytes` that is not a canonical non-negative decimal (digits only, no
-- leading zero — the spelling `INCRBY` takes) is refused `ENCODING` with
-- nothing written. (A canonical value past `INCRBY`'s signed 64-bit range is
-- not checked here; the Rust caller's `STRLEN`-derived sizes never reach it.)
-- The first
-- write is the `SADD`, so a store write refusal (`-OOM`) also leaves the
-- keyspace untouched, exactly as the bare `SADD` it replaces did; in attempt
-- mode the record is written after the body, in the same atomic step.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Without attempt mode: `{ added, refused }` — the members this call added to
-- `pending`, and the members it refused as replays. In attempt mode:
-- `{ verdict, added, refused }`, verdict one of `"enqueued"`, `"already"`,
-- `"void"` (see above).

local pending_key  = KEYS[1]
local age_key      = KEYS[2]
local bytes_key    = KEYS[3]
local total_key    = KEYS[4]
local inflight_key = KEYS[5]
local settled_key  = KEYS[6]

local ATTEMPT_MARK = "@attempt"

if #KEYS ~= 6 and #KEYS ~= 7 then
  return redis.error_reply("ENCODING: ext_release_enqueue expects six keys, or seven in attempt mode")
end
local attempt_mode = ARGV[1] == ATTEMPT_MARK
local first = 1
local attempt_key, attempt, attempt_oid
if attempt_mode then
  if #KEYS ~= 7 then
    return redis.error_reply("ENCODING: an attempt-mode enqueue needs the attempt record as KEYS[7]")
  end
  if ARGV[2] == nil or ARGV[3] == nil then
    return redis.error_reply("ENCODING: attempt mode needs ARGV \"@attempt\", token, object_id")
  end
  attempt_key = KEYS[7]
  attempt     = ARGV[2]
  attempt_oid = ARGV[3]
  first = 4
elseif #KEYS == 7 then
  return redis.error_reply("ENCODING: a seven-key enqueue must be in attempt mode (ARGV[1] \"@attempt\")")
end

local n = #ARGV
local pairs_n = n - first + 1
if pairs_n <= 0 or pairs_n % 2 ~= 0 then
  return redis.error_reply("ENCODING: ARGV must be one or more (member, bytes) pairs")
end
-- Canonical decimals only (`^0$` or `^[1-9]%d*$`; a Lua pattern has no `|`):
-- `INCRBY` refuses a leading zero, and it would do so only after this pair's
-- `SADD` had run.
for i = first + 1, n, 2 do
  local b = ARGV[i]
  if b ~= "0" and string.find(b, "^[1-9]%d*$") == nil then
    return redis.error_reply("ENCODING: bytes for a release member must be a canonical non-negative decimal")
  end
end

if attempt_mode then
  local recorded = redis.call("HGET", attempt_key, attempt)
  if recorded then
    if string.sub(recorded, 1, 5) == "void:" then
      return { "void", 0, 0 }
    end
    if string.sub(recorded, 1, 7) == "landed:" then
      return { "already", 0, 0 }
    end
    return redis.error_reply("ENCODING: unrecognised enqueue attempt record for this token")
  end
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local added = 0
local refused = 0
for i = first, n, 2 do
  local member = ARGV[i]
  local bytes  = ARGV[i + 1]
  if redis.call("SISMEMBER", pending_key, member) == 0
      and (redis.call("ZSCORE", inflight_key, member)
        or redis.call("ZSCORE", settled_key, member)
        or redis.call("HEXISTS", bytes_key, member) == 1) then
    refused = refused + 1
  else
    if redis.call("SADD", pending_key, member) == 1 then
      added = added + 1
      redis.call("ZADD", age_key, "NX", now_ms, member)
    end
    if redis.call("HEXISTS", bytes_key, member) == 0 then
      redis.call("HSET", bytes_key, member, bytes)
      if bytes ~= "0" then
        redis.call("INCRBY", total_key, bytes)
      end
    end
  end
end

if attempt_mode then
  redis.call("HSET", attempt_key, attempt, "landed:" .. now_ms .. ":" .. attempt_oid)
  return { "enqueued", added, refused }
end

return { added, refused }
