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

-- ext_index_recount.lua — atomic per-object recount and repair of the
-- external-value reference index `le:{oid}:obj:ext`.
--
-- Rust side: `storage::ext_index_audit::recount_ext_index`. Two callers: the
-- reference-count reconciler's object walk, when an object's index disagrees
-- with its rows (`site="reconcile"`), and the txn commit path after a torn
-- `EXEC` (`site="torn_commit"`).
--
-- ── What it establishes ──────────────────────────────────────────────────
--
-- `obj:ext` holds, per external hash h, the number of the object's value rows
-- (fields and time samples) that hold h, and holds no field for a hash no row
-- holds. This script recounts that from the rows themselves, in one atomic
-- unit, and rewrites every field that disagrees with an absolute `HSET` (or an
-- `HDEL` when the true count is 0) — every RAISE, at least; lowerings are
-- withheld while any row's `hash` column is malformed (below). Nothing can
-- commit to the object between the count and the write, so what it writes is
-- exact at the instant it lands.
--
-- **The counting rule (rule 1).** A value row holds external hash h iff it has
-- NO `value` column and its `hash` column is h. It is the decoder's rule
-- (`storage::diff_read::decode_field_value` reads `value` first), and every
-- per-object reader and writer of the index counts by it — the materializers,
-- both commit backends, retirement and this script. A legacy dual-shape row
-- (`value` plus a stale `hash`) therefore holds no hash.
--
-- A `hash` column that is not 64 lowercase hex digits is not counted: no
-- writer produces one (every writer hex-encodes in lowercase), so it is
-- keyspace corruption, and the script does not copy it into the index. It is
-- reported in the reply (the `malformed` count) instead. Likewise an existing
-- index field whose name is not 64 lowercase hex is left exactly as it is:
-- this script never writes or removes a field it could not have counted.
--
-- **Raises only while any row is malformed.** Not counting a malformed row is
-- not the same as that row holding no hash: the Rust decoder accepts
-- uppercase hex (`hash_from_hex`), so the decoder, the materializers and the
-- reconciler's walk can all count a row this script skips as holding h, and
-- that row keeps serving h. Lowering or removing h's field on this script's
-- count would then be an under-count against what the object serves — the
-- loss direction. So when `malformed > 0` the repair applies only RAISES (a
-- count increased, or a missing field added) and withholds every lowering and
-- every `HDEL`, reporting how many it withheld (the `WITHHELD` reply). Every
-- reply reports the malformed count, and the reconciler treats ANY reply with
-- `malformed > 0` as not authoritative — even a `CLEAN` or `REPAIRED` one, since
-- the counts it matched or raised to still omit the rows the decoder counts —
-- and holds the object's mismatched hashes from surplus drain, exactly as for
-- an object it could not recount. With no malformed row, a lowering is a
-- lowering against every reader's count, and is applied.
--
-- **Row-key derivation is a mirror.** Row keys are built from the index
-- members exactly as `KeySchema::obj_field_from_member` and
-- `KeySchema::obj_ts_from_member` build them (`{prefix}{oid}:obj:fld:{member}`
-- and `{prefix}{oid}:obj:ts:{member}`). Any new value-row family, or any change
-- to that derivation, must be mirrored here, or this script would count a
-- subset of the rows and lower counts that are correct.
--
-- ── It never releases anything ───────────────────────────────────────────
--
-- Lowering or removing an over-counted field hands back nothing: the global
-- `ref_count` counts live objects, so a field removed because no row holds the
-- hash any more leaves a `+1` the object no longer justifies, which is left
-- for the reference-count reconciler to drain as ordinary surplus through its
-- two-observation protocol (a leak until then, and for good while the
-- reconciler is disabled). Releasing in-script would be unmatched for objects
-- whose surplus a restore suppressed (`SurplusPolicy::Suppress`). Raising a
-- count changes no global count at all. Either way this script leaves the
-- global side to the reconciler, and with the raises-only rule above its own
-- index writes err in the leak direction, never the loss direction, even on
-- an object carrying a hash column it cannot count. (It cannot undo an
-- over-release a commit path already emitted from a drifted count.)
--
-- ── The header write, and why it is first ────────────────────────────────
--
-- A repair first stamps `ext_index_repaired_ms` on the header. Two reasons:
--
--  1. It cancels every in-flight WATCHED txn commit on this object. The txn
--     backend WATCHes the header, plans absolute `obj:ext` counts from the
--     index it read before `MULTI`, and EXECs them. Without the header write,
--     such a commit could EXEC after this repair and overwrite it with counts
--     planned from the pre-repair index. With it, the EXEC is cancelled and the
--     retry plans from the repaired index. A commit whose WATCH was void (its
--     connection was replaced) is not cancelled: it commits unconditionally,
--     can overwrite this repair, and returns an all-OK EXEC, so no torn-commit
--     recount follows; that drift waits for the reconciler's next audit.
--  2. As the script's first write it is the one that absorbs the OOM gate: a
--     script without a shebang is refused under `maxmemory` at its FIRST
--     deny-oom write, so an OOM refusal leaves nothing written.
--
-- The field has no code reader. It is removed with the header on every
-- retirement route and by the restore residue wipe, and the snapshot decoder
-- ignores it, so it never reaches a checkpoint. It does NOT advance `seq`, so
-- no cache keyed on `seq` (or on a commit chain) may hold `obj:ext` counts.
--
-- ── Liveness, checked read-only ──────────────────────────────────────────
--
-- The script writes only to a LIVE object: a header that exists, whose `seq`
-- is numeric, and that carries neither `superseded` (a CreateLayer overwrite
-- has retired it) nor `ext_gc_scheduled` (its retirement releases are being
-- scheduled). Anything else answers `NOT_LIVE` and writes nothing, so the
-- script never creates a phantom header (the `stamp_cursor_if_live.lua`
-- precedent) and never interferes with a retirement or with a restore's
-- residue-wipe → rebuild window.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- No `#!lua` shebang, like every script in this directory: compat mode is what
-- the first-write OOM argument above relies on.
--
-- KEYS (all carry the `{oid}` hash tag, so one slot):
--   [1]  the object header      `{prefix}{oid}:obj:hdr` (pins the slot)
--   [2]  the reference index    `{prefix}{oid}:obj:ext`
--   [3]  the fields index       `{prefix}{oid}:obj:flds`
--   [4]  the time-sample index  `{prefix}{oid}:obj:tss`
--   [5]  the migration sentinel `{prefix}{oid}:migration-sentinel` (never
--        written, read or named by a command here; see "Migration sentinel")
--   The value rows are undeclared same-slot keys built from ARGV[1..2] — the
--   capability the atomic snapshot read's startup self-test already gates. The
--   script refuses to run unless KEYS is exactly these five keys of the
--   object ARGV[1..2] names, so no key it touches can fall outside the `{oid}`
--   tag.
--
-- ARGV:
--   [1]  key prefix (`KeySchema::prefix`)
--   [2]  object id
--   [3]  row ceiling: the most value rows (fields + time samples) one call
--        will read. Two reads per row, so the caller passes half the atomic
--        snapshot read's item limit (`LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS`)
--        — a conservative proxy for the same script-duration budget, not a
--        measurement.
--
-- Replies — a flat array of strings (RESP2-safe: no nils):
--   {'NOT_LIVE'}
--       The object is not live (see above). Nothing written.
--   {'TOO_LARGE', rows}
--       The object has more rows than the ceiling. Nothing read past the
--       cardinalities, nothing written.
--   {'CLEAN', malformed}
--       Every countable field already matches. Nothing written.
--   {'REPAIRED', under, over, malformed, h1, stored1, counted1, h2, ...}
--       `under` / `over` count the index FIELDS corrected in each direction
--       (`under`: the stored count was below the rows, an absent field reading
--       as 0; `over`: above, including a field present for a hash no row
--       holds). A non-numeric stored value reads as 0 for the direction. The
--       triples are a bounded exemplar list (at most EXEMPLAR_LIMIT): the
--       hash, its stored value before the repair ('' when absent; truncated),
--       and the count written. Only when `malformed` is 0, or when every fix
--       was a raise.
--   {'WITHHELD', under, withheld, malformed, h1, stored1, counted1, h2, ...}
--       `malformed > 0` and at least one fix was a lowering or removal, which
--       the raises-only rule withheld (`withheld` counts those fields). `under`
--       and the exemplar triples cover the raises applied, if any; the header
--       stamp was written iff `under > 0`. The index still disagrees with the
--       rows on the withheld fields, so the caller must treat the object as
--       not repaired.
--   `malformed` is the number of rows with no `value` column whose `hash`
--   column was not 64 lowercase hex (not counted, see above).
-- Errors (nothing written in any of them):
--   RECOUNT_ARGS: <reason>      a caller bug
--   WRONGTYPE_KEY: <key> holds a <type>, expected <type>
--                               a declared key or a value row of the wrong type
--   anything else is a script-level failure (OOM on the first write, NOSCRIPT,
--   -BUSY, a slot migration's `-TRYAGAIN` / `-ASK`, ...), which writes nothing
--   because the first write is the header's, or because the script never
--   started. The one refusal that could otherwise land after the header stamp,
--   a cluster node's in-script refusal during a slot migration, cannot happen:
--   the migration sentinel keeps the script from starting while its slot is
--   moving (below).
--
-- ── Migration sentinel (b.cyu) ───────────────────────────────────────────
--
-- During a legacy cluster slot migration every in-script command is checked
-- against the slot's state again, and on the migrating source one that names
-- a key absent at that moment is refused, aborting the script with its
-- earlier writes kept. Step 6 applies the stored-field fixes in `HGETALL`
-- order, then the missing fields: a lowering's `HDEL ext` can empty `obj:ext`
-- before a later raise's `HSET ext`, after the header stamp landed, leaving
-- the index part-fixed. Declaring the never-written `KEYS[5]` makes the store
-- refuse the whole script before it starts while the `{oid}` slot is moving
-- (`-TRYAGAIN`, or `-ASK`, which the service never follows for a script; both
-- transient), so the "nothing written" contract above holds; on a stable slot
-- the sentinel is never looked up. The callers already retry a failed
-- recount: the reconciler on its next pass, and the torn-commit site leaves
-- the drift to it.

local EXEMPLAR_LIMIT = 8
local STORED_EXEMPLAR_MAX = 32

local hdr_key = KEYS[1]
local ext_key = KEYS[2]
local flds_key = KEYS[3]
local tss_key = KEYS[4]
if ARGV[1] == nil or ARGV[2] == nil then
  return redis.error_reply('RECOUNT_ARGS: missing prefix or object id')
end
local base = ARGV[1] .. '{' .. ARGV[2] .. '}:obj:'
local row_ceiling = tonumber(ARGV[3])
if not row_ceiling then
  return redis.error_reply('RECOUNT_ARGS: missing or non-numeric row ceiling')
end
if hdr_key ~= base .. 'hdr' or ext_key ~= base .. 'ext' or flds_key ~= base .. 'flds'
    or tss_key ~= base .. 'tss' then
  return redis.error_reply('RECOUNT_ARGS: KEYS[1..4] are not the hdr/ext/flds/tss keys of ' ..
    base)
end
-- The migration sentinel (see "Migration sentinel"): declared, never named by
-- any command below.
if #KEYS ~= 5 or KEYS[5] ~= ARGV[1] .. '{' .. ARGV[2] .. '}:migration-sentinel' then
  return redis.error_reply('RECOUNT_ARGS: KEYS[5] is not the migration sentinel of ' ..
    ARGV[1] .. '{' .. ARGV[2] .. '}')
end

local function wrong_type(key, actual, expected)
  return redis.error_reply('WRONGTYPE_KEY: ' .. key .. ' holds a ' .. actual ..
    ', expected ' .. expected)
end

-- ── 1. Liveness (read-only). ─────────────────────────────────────────────
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

-- ── 2. Type probes on the declared data keys, then the size cap. ─────────
local probes = {
  {ext_key, 'hash', 'hash (external-value reference index)'},
  {flds_key, 'zset', 'zset (fields index)'},
  {tss_key, 'zset', 'zset (time-sample index)'},
}
for _, p in ipairs(probes) do
  local t = redis.call('TYPE', p[1])['ok']
  if t ~= p[2] and t ~= 'none' then
    return wrong_type(p[1], t, p[3])
  end
end

local rows = redis.call('ZCARD', flds_key) + redis.call('ZCARD', tss_key)
if rows > row_ceiling then
  return {'TOO_LARGE', string.format('%d', rows)}
end

-- ── 3. Count by rule 1. ──────────────────────────────────────────────────
local function is_hash_hex(s)
  return #s == 64 and not string.find(s, '[^0-9a-f]')
end

local counted = {}       -- hash -> rows holding it
local counted_order = {} -- first-seen order, so the reply is deterministic
local malformed = 0

local function count_rows(index_key, sub)
  local members = redis.call('ZRANGE', index_key, 0, -1)
  for i = 1, #members do
    local row_key = base .. sub .. members[i]
    -- `HEXISTS` rather than reading `value`, so inline bytes are never copied
    -- into the script. A wrong-typed row refuses before any write.
    local has_value = redis.pcall('HEXISTS', row_key, 'value')
    if type(has_value) == 'table' and has_value.err then
      local t = redis.call('TYPE', row_key)['ok']
      return wrong_type(row_key, t, 'hash (value row)')
    end
    if has_value == 0 then
      local h = redis.call('HGET', row_key, 'hash')
      if h then
        if is_hash_hex(h) then
          if counted[h] == nil then
            counted[h] = 0
            counted_order[#counted_order + 1] = h
          end
          counted[h] = counted[h] + 1
        else
          malformed = malformed + 1
        end
      end
    end
  end
  return nil
end

local refusal = count_rows(flds_key, 'fld:')
if refusal then return refusal end
refusal = count_rows(tss_key, 'ts:')
if refusal then return refusal end

-- ── 4. Compare against the stored index. ─────────────────────────────────
local function parse_count(v)
  if string.find(v, '^%d+$') then
    return tonumber(v)
  end
  return nil
end

local stored = redis.call('HGETALL', ext_key)
local stored_map = {}
local fixes = {} -- each {hash, stored_value_or_false, count}
for i = 1, #stored, 2 do
  local field = stored[i]
  local value = stored[i + 1]
  stored_map[field] = value
  if is_hash_hex(field) then
    local want = counted[field] or 0
    local have = parse_count(value)
    -- "present ⟺ count > 0": a present field is wrong when the rows hold the
    -- hash nowhere, whatever it stores.
    if want == 0 or have ~= want then
      fixes[#fixes + 1] = {field, value, want}
    end
  end
end
for _, h in ipairs(counted_order) do
  if stored_map[h] == nil then
    fixes[#fixes + 1] = {h, false, counted[h]}
  end
end

if #fixes == 0 then
  return {'CLEAN', string.format('%d', malformed)}
end

-- ── 5. Split the fixes. With any malformed row, lowerings are withheld (see
--       "Raises only while any row is malformed" above). A raise is a count
--       written above what is stored, a non-numeric stored value reading as 0
--       (the commit paths read it as 0 too); everything else, `HDEL`
--       included, is a lowering. ───────────────────────────────────────────
local apply = {} -- each {hash, stored_value_or_false, count, is_raise}
local withheld = 0
for _, f in ipairs(fixes) do
  local stored_value, want = f[2], f[3]
  local have = (stored_value and parse_count(stored_value)) or 0
  local is_raise = want > have
  if is_raise or malformed == 0 then
    apply[#apply + 1] = {f[1], stored_value, want, is_raise}
  else
    withheld = withheld + 1
  end
end

-- ── 6. Repair. The header stamp is the FIRST write (see the header); when
--       every fix was withheld there is no write at all. ───────────────────
if #apply > 0 then
  local now = redis.call('TIME')
  local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
  redis.call('HSET', hdr_key, 'ext_index_repaired_ms', string.format('%d', now_ms))
end

local under, over = 0, 0
local exemplars = {}
for _, f in ipairs(apply) do
  local h, stored_value, want, is_raise = f[1], f[2], f[3], f[4]
  if is_raise then
    under = under + 1
  else
    over = over + 1
  end
  if want > 0 then
    redis.call('HSET', ext_key, h, string.format('%d', want))
  else
    redis.call('HDEL', ext_key, h)
  end
  if #exemplars < EXEMPLAR_LIMIT * 3 then
    exemplars[#exemplars + 1] = h
    exemplars[#exemplars + 1] = stored_value and string.sub(stored_value, 1, STORED_EXEMPLAR_MAX)
      or ''
    exemplars[#exemplars + 1] = string.format('%d', want)
  end
end

local reply
if withheld > 0 then
  -- No lowering was applied (`over` is 0 by construction), so the third
  -- element carries the withheld count instead.
  reply = {'WITHHELD', string.format('%d', under), string.format('%d', withheld),
    string.format('%d', malformed)}
else
  reply = {'REPAIRED', string.format('%d', under), string.format('%d', over),
    string.format('%d', malformed)}
end
for _, e in ipairs(exemplars) do
  reply[#reply + 1] = e
end
return reply
