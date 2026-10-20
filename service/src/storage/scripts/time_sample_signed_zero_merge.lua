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

-- time_sample_signed_zero_merge.lua — atomic merge of one object's legacy
-- `-0.0` time-sample rows into the `+0.0` key.
--
-- Rust side: `storage::time_sample_merge`. One caller: the reference-count
-- reconciler's object walk (`storage::ext_reconcile`), for an object whose
-- `obj:tss` lists a member ending `:8000000000000000`.
--
-- ── What it establishes ──────────────────────────────────────────────────
--
-- `-0.0` and `0.0` are one USD time code, but a time-sample row key spells
-- the time by its IEEE-754 bits, so `-0.0` (`8000000000000000`) and `0.0`
-- (`0000000000000000`) are two stored rows. This build never writes a `-0.0`
-- row (`oodf_ffi::parse_delta` canonicalizes every time); an older build did.
-- For each section the caller names, this script leaves one row, at `+0.0`:
--
--   keep (`K`)  the `+0.0` row survives: `DEL` the `-0.0` row and `ZREM` its
--               member;
--   take (`T`)  the `-0.0` row survives, or is the only one readers see (no
--               `+0.0` key, or one `obj:tss` does not list): `RENAME` it over
--               the `+0.0` key, `ZADD` the `+0.0` member at the `-0.0`
--               member's score (its version), then `ZREM` the `-0.0` member.
--
-- The caller decides each row in Rust, from both rows decoded by the full
-- readers' decoder and resolved by the readers' own rule
-- (`domain::types::resolve_signed_zero_pair`), so the merge changes nothing a
-- reader serves. This script only checks that the object is the one the
-- caller read, and applies.
--
-- **Idempotent.** A row whose `-0.0` key is already gone is skipped: another
-- run (a second leader, say) merged it. With the `seq` guard below that makes
-- any number of runs over the same read land on the same result. A keep whose
-- `+0.0` row is gone, or whose `+0.0` key has no `obj:tss` member, is skipped
-- too: a stale keep never drops a section's only row at this time, and never
-- drops the `-0.0` row readers serve when the index does not list the `+0.0`
-- one. The caller plans an unlisted `+0.0` row as absent, so it decides take
-- there and never keep; this skip is the backstop for a member that went
-- between the caller's read and this script, and the next pass re-plans the
-- section from a fresh read.
--
-- ── It never releases anything ───────────────────────────────────────────
--
-- The row it drops may hold an external value. It does not touch `obj:ext`
-- and enqueues no release: losing a row can only leave the object's index
-- counting one row too many, which is the leak direction. The same reconciler
-- pass's index audit then recounts the object, and the reconciler drains any
-- `ref_count` surplus through its usual two-observation protocol.
--
-- ── Guards, in order, before any write ───────────────────────────────────
--
--  1. The migration sentinel (KEYS[4]) is this object's (below).
--  2. Liveness and the `seq` guard: the header exists, its `seq` is numeric
--     and equals ARGV[3], the `seq` the caller read before its rows, and it
--     carries neither `superseded` nor `ext_gc_scheduled`. A missing or
--     retiring header answers `NOT_LIVE`, a moved `seq` `CHANGED`; both write
--     nothing. The guard is essential: this rewrite does not advance `seq`,
--     so without it a stale "take" could overwrite a newer `+0.0` write.
--  3. No live snapshot fence: any `r:*` lease on `obj:snapfence` whose
--     deadline is past this node's `TIME` answers `{'FENCED', ms}` and
--     writes nothing — the commit script's check (`update_layer_fast.lua`
--     step 1d), since this rewrite must not land under a fenced reader any
--     more than a commit may.
--  4. Every declared key is TYPE-probed, so a wrong-typed key refuses before
--     any write.
--
-- ── The header write, and why it is first ────────────────────────────────
--
-- When at least one row needs work, the script first stamps
-- `time_samples_merged_ms` on the header (the Valkey-side time in ms):
--
--  1. It cancels every in-flight WATCHED txn commit on this object, as
--     `ext_index_recount.lua`'s `ext_index_repaired_ms` does: such a commit
--     planned from the pre-merge rows cannot `EXEC` after this script.
--  2. As the script's first write it absorbs the store's write gates: a
--     script without a shebang is refused under `maxmemory` (and on
--     `-READONLY` / `-MISCONF` / `-NOREPLICAS`) only at its FIRST write, and
--     `HSET` is deny-oom, so on a full store the script is refused with
--     nothing written. It is never needed for correctness, so it waits for
--     memory like any ordinary write, and never writes past `maxmemory`.
--
-- The field has no code reader. It is removed with the header on every
-- retirement route and by the restore residue wipe, and the snapshot decoder
-- ignores it, so it never reaches a checkpoint. It does NOT advance `seq`.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- No `#!lua` shebang, like every script in this directory: compat mode is what
-- the first-write gate argument above relies on.
--
-- KEYS (all carry the `{oid}` hash tag, so one slot):
--   [1]  the object header      `{prefix}{oid}:obj:hdr` (pins the slot)
--   [2]  the time-sample index  `{prefix}{oid}:obj:tss`
--   [3]  the snapshot fence     `{prefix}{oid}:obj:snapfence`
--   [4]  the migration sentinel `{prefix}{oid}:migration-sentinel` (never
--        written, read or named by a command here; see "Migration sentinel")
--   then per row, in ARGV order: the `+0.0` row key, then the `-0.0` row key
--   (`{prefix}{oid}:obj:ts:{section_id}:0000000000000000` and
--   `...:8000000000000000`).
--
-- ARGV:
--   [1]  key prefix (`KeySchema::prefix`)
--   [2]  object id
--   [3]  the header `seq` the caller read before its rows
--   then per row: the section id, then its action, `K` (keep) or `T` (take).
--
-- At most MAX_ROWS rows per call; the caller splits a longer list.
--
-- Replies — a flat array of strings (RESP2-safe: no nils):
--   {'NOT_LIVE'}              not live (guard 2). Nothing written.
--   {'CHANGED', seq}          `seq` moved since the caller's read. Nothing
--                             written.
--   {'FENCED', retry_after_ms}
--                             a live snapshot fence. Nothing written.
--   {'MERGED', merged, rekeyed, skipped}
--                             `merged`: rows whose `+0.0` key existed;
--                             `rekeyed`: `take` rows with no `+0.0` key;
--                             `skipped`: rows left as they are — the `-0.0`
--                             key already gone, its member absent, or a keep
--                             whose `+0.0` key or member is gone. The header
--                             was stamped iff `merged + rekeyed > 0`.
-- Errors (nothing written in any of them):
--   MERGE_ARGS: <reason>        a caller bug
--   WRONGTYPE_KEY: <key> holds a <type>, expected <type>
--   anything else is a script-level failure (OOM or another write gate on the
--   first write, NOSCRIPT, -BUSY, a slot migration's `-TRYAGAIN` / `-ASK`),
--   which writes nothing because the first write is the header's, or because
--   the script never started.
--
-- ── Migration sentinel ───────────────────────────────────────────────────
--
-- During a legacy cluster slot migration every in-script command is checked
-- against the slot's state again, and on the migrating source one that names
-- a key absent at that moment is refused, aborting the script with its earlier
-- writes kept: a `RENAME` could land and the `ZADD` that follows be refused,
-- leaving a row its index does not list. Declaring the never-written KEYS[4]
-- makes the store refuse the whole script before it starts while the `{oid}`
-- slot is moving (`-TRYAGAIN`, or `-ASK`, which the service never follows for
-- a script; both transient), so the "nothing written" contract above holds; on
-- a stable slot the sentinel is never looked up. The caller retries on the
-- next reconciler pass.

local MAX_ROWS = 256
local POSITIVE_ZERO_HEX = '0000000000000000'
local NEGATIVE_ZERO_HEX = '8000000000000000'

local hdr_key = KEYS[1]
local tss_key = KEYS[2]
local fence_key = KEYS[3]
if ARGV[1] == nil or ARGV[2] == nil then
  return redis.error_reply('MERGE_ARGS: missing prefix or object id')
end
-- The migration sentinel (see "Migration sentinel"): declared, never named by
-- any command below.
if #KEYS < 4 or KEYS[4] ~= ARGV[1] .. '{' .. ARGV[2] .. '}:migration-sentinel' then
  return redis.error_reply('MERGE_ARGS: KEYS[4] is not the migration sentinel of ' ..
    ARGV[1] .. '{' .. ARGV[2] .. '}')
end
local base = ARGV[1] .. '{' .. ARGV[2] .. '}:obj:'
if hdr_key ~= base .. 'hdr' or tss_key ~= base .. 'tss' or fence_key ~= base .. 'snapfence' then
  return redis.error_reply('MERGE_ARGS: KEYS[1..3] are not the hdr/tss/snapfence keys of ' ..
    base)
end
local expected_seq = ARGV[3]
if expected_seq == nil or not string.find(expected_seq, '^%d+$') then
  return redis.error_reply('MERGE_ARGS: missing or non-decimal expected seq')
end
local rows = (#ARGV - 3) / 2
if rows < 1 or rows ~= math.floor(rows) or rows > MAX_ROWS then
  return redis.error_reply('MERGE_ARGS: the row arguments are not 1..' .. MAX_ROWS ..
    ' (section id, action) pairs')
end
if #KEYS ~= 4 + 2 * rows then
  return redis.error_reply('MERGE_ARGS: KEYS does not carry two row keys per row')
end
-- Each row's keys are exactly the section's two zero-time row keys, so no key
-- this script touches can fall outside the object's `{oid}` tag.
for i = 1, rows do
  local sid = ARGV[3 + 2 * i - 1]
  local action = ARGV[3 + 2 * i]
  if not string.find(sid, '^%d+$') then
    return redis.error_reply('MERGE_ARGS: row ' .. i .. ' has a non-decimal section id')
  end
  if action ~= 'K' and action ~= 'T' then
    return redis.error_reply('MERGE_ARGS: row ' .. i .. ' has an action other than K or T')
  end
  if KEYS[4 + 2 * i - 1] ~= base .. 'ts:' .. sid .. ':' .. POSITIVE_ZERO_HEX
      or KEYS[4 + 2 * i] ~= base .. 'ts:' .. sid .. ':' .. NEGATIVE_ZERO_HEX then
    return redis.error_reply('MERGE_ARGS: row ' .. i .. "'s keys are not section " .. sid ..
      "'s zero-time row keys")
  end
end

local function wrong_type(key, actual, expected)
  return redis.error_reply('WRONGTYPE_KEY: ' .. key .. ' holds a ' .. actual ..
    ', expected ' .. expected)
end

-- ── 2. Liveness and the `seq` guard (read-only). ─────────────────────────
local hdr_type = redis.call('TYPE', hdr_key)['ok']
if hdr_type == 'none' then
  return {'NOT_LIVE'}
end
if hdr_type ~= 'hash' then
  return wrong_type(hdr_key, hdr_type, 'hash (object header)')
end
local hdr = redis.call('HMGET', hdr_key, 'seq', 'superseded', 'ext_gc_scheduled')
if not tonumber(hdr[1]) or hdr[2] or hdr[3] then
  return {'NOT_LIVE'}
end
if hdr[1] ~= expected_seq then
  return {'CHANGED', hdr[1]}
end

-- ── 3. The snapshot fence (`update_layer_fast.lua` step 1d). ─────────────
local fence_type = redis.call('TYPE', fence_key)['ok']
if fence_type ~= 'hash' and fence_type ~= 'none' then
  return wrong_type(fence_key, fence_type, 'hash (snapshot fence)')
end
if fence_type == 'hash' then
  local fence = redis.call('HGETALL', fence_key)
  local tm = redis.call('TIME')
  local now = tonumber(tm[1]) * 1000 + math.floor(tonumber(tm[2]) / 1000)
  local latest_live = nil
  for i = 1, #fence, 2 do
    if string.sub(fence[i], 1, 2) == 'r:' then
      local deadline = tonumber(fence[i + 1])
      if deadline ~= nil and deadline > now and (latest_live == nil or deadline > latest_live) then
        latest_live = deadline
      end
    end
  end
  if latest_live ~= nil then
    return {'FENCED', string.format('%d', math.max(1, latest_live - now))}
  end
end

-- ── 4. Type probes. ──────────────────────────────────────────────────────
local tss_type = redis.call('TYPE', tss_key)['ok']
if tss_type ~= 'zset' and tss_type ~= 'none' then
  return wrong_type(tss_key, tss_type, 'zset (time-sample index)')
end
-- Per row: whether each zero-time row exists, the `-0.0` member's score, and,
-- for a keep, whether the `+0.0` member is listed.
local plan = {}
for i = 1, rows do
  local sid = ARGV[3 + 2 * i - 1]
  local pos_key = KEYS[4 + 2 * i - 1]
  local neg_key = KEYS[4 + 2 * i]
  local pos_type = redis.call('TYPE', pos_key)['ok']
  if pos_type ~= 'hash' and pos_type ~= 'none' then
    return wrong_type(pos_key, pos_type, 'hash (time-sample row)')
  end
  local neg_type = redis.call('TYPE', neg_key)['ok']
  if neg_type ~= 'hash' and neg_type ~= 'none' then
    return wrong_type(neg_key, neg_type, 'hash (time-sample row)')
  end
  local action = ARGV[3 + 2 * i]
  local pos_member = sid .. ':' .. POSITIVE_ZERO_HEX
  local neg_member = sid .. ':' .. NEGATIVE_ZERO_HEX
  local score = false
  if neg_type == 'hash' then
    score = redis.call('ZSCORE', tss_key, neg_member) or false
  end
  -- A keep's `+0.0` row counts only when its member is listed too: the same
  -- key-and-member test the `-0.0` row gets above.
  local pos_listed = false
  if action == 'K' and pos_type == 'hash' then
    pos_listed = redis.call('ZSCORE', tss_key, pos_member) ~= false
  end
  plan[i] = {
    action = action,
    pos_key = pos_key,
    neg_key = neg_key,
    pos_member = pos_member,
    neg_member = neg_member,
    pos_exists = pos_type == 'hash',
    -- Only a row whose `-0.0` key and member both exist is acted on: a key
    -- already gone was merged by another run, and a key with no member is
    -- not one the readers see, so it is left as it is. A keep needs the
    -- `+0.0` row it keeps, key and member both: without the key, dropping
    -- the `-0.0` row would drop the section's only sample at this time; with
    -- the key but no member, readers do not see the `+0.0` row and serve the
    -- `-0.0` one, so dropping it would change what they serve. Either way the
    -- row is left instead.
    acts = neg_type == 'hash' and score ~= false and (action == 'T' or pos_listed),
    score = score,
  }
end

local acting = 0
for i = 1, rows do
  if plan[i].acts then
    acting = acting + 1
  end
end
if acting == 0 then
  return {'MERGED', '0', '0', string.format('%d', rows)}
end

-- ── 5. Merge. The header stamp is the FIRST write (see the header). ──────
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('HSET', hdr_key, 'time_samples_merged_ms', string.format('%d', now_ms))

local merged, rekeyed, skipped = 0, 0, 0
for i = 1, rows do
  local r = plan[i]
  if not r.acts then
    skipped = skipped + 1
  elseif r.action == 'K' then
    redis.call('DEL', r.neg_key)
    redis.call('ZREM', tss_key, r.neg_member)
    merged = merged + 1
  else
    redis.call('RENAME', r.neg_key, r.pos_key)
    redis.call('ZADD', tss_key, r.score, r.pos_member)
    redis.call('ZREM', tss_key, r.neg_member)
    if r.pos_exists then
      merged = merged + 1
    else
      rekeyed = rekeyed + 1
    end
  end
end

return {'MERGED', string.format('%d', merged), string.format('%d', rekeyed),
  string.format('%d', skipped)}
