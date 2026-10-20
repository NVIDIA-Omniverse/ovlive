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

-- object_snapshot_read.lua — the atomic object-snapshot read (SDD SR-1).
--
-- Returns the whole materialized state of one object in one EVALSHA. A script
-- runs atomically, so no commit lands between its first read and its last:
-- the state it returns is exactly the state at the header `seq` it returns
-- (the cut). Rust side: `storage::diff_read::read_object_snapshot`.
--
-- Read-only: no command below writes. No `#!lua` shebang, like every script
-- in this directory, so it runs in compat mode (SDD §7.5.1): the mode the
-- write scripts' OOM classification relies on, and one in which this read
-- keeps running on a store over `maxmemory`.
--
-- KEYS[1]   the object header, `{prefix}{oid}:obj:hdr`. The only declared
--           key; it pins the slot. Every other key is built below from
--           ARGV[1..2] and carries the same `{oid}` hash tag, so it is
--           undeclared but same-slot. The script refuses to run unless
--           KEYS[1] is exactly the header ARGV[1..2] name, so no key it
--           touches can fall outside the `{oid}` tag.
-- ARGV[1]   key prefix (`KeySchema::prefix`)      ARGV[2]  object id
-- ARGV[3]   mode: 'read' | 'selftest'
-- ARGV[4]   item limit  (LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS)
-- ARGV[5]   byte limit  (LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES)
--           Every read passes the configured limits, except the raised-ceiling
--           read (no production caller; the measurement tool's), which passes
--           multiples of them
--           (`STRUCTURAL_COLD_START_ITEMS_FACTOR` / `_BYTES_FACTOR`).
-- ARGV[6]   section hash width        ARGV[7]  tombstone hash width
-- ARGV[8]   field hash width          ARGV[9]  time-sample hash width
--           (the `SNAPSHOT_*_HASH_WIDTH` constants in `storage::diff_read`,
--           which cite the writers; passed in so the bound has one source)
-- ARGV[10]  the children-index completeness column and ARGV[11] its marked
--           value (`storage::layer::HEADER_FIELD_CHS_COMPLETE` /
--           `CHS_COMPLETE_MARKED`)
-- ARGV[12]  optional `return_dangling` flag: '1' sets it; absent or '0'
--           leaves it unset. Only the structural cache's cold-start and audit
--           reads set it (`SlimSnapshotReadConsumer::StructuralCache` /
--           `StructuralCacheAudit`); every reader passes eleven arguments,
--           and a slim read that does not set it passes '0' before ARGV[13].
--           Unset, every reply is the one listed below without `dangling`,
--           byte for byte: the flag is additive, and a change to this script
--           must keep the unset replies unchanged for the consumers that never
--           set it. Set, it issues no extra command (the dangling members
--           come from the `obj:chs` SMEMBERS already read) and adds one
--           element to an `OK` reply only.
-- ARGV[13]  optional `slim` value-rows flag: '1' sets it; absent or '0'
--           leaves it unset. Only the slim reads of the structural
--           write-planning consumers set it (`storage::diff_read`'s
--           `SlimSnapshotReadConsumer`), which pass ARGV[12] too. Additive
--           like `return_dangling`: unset, every reply is byte for byte the
--           one listed below. Set, it issues no extra command and changes no
--           tally: each field and time-sample row is still read with one
--           `HGETALL` and tallied in full, so the item, byte and entry
--           tallies, and every `TOO_LARGE` decision, are a full read's. Only
--           the per-row element of `flds` and `tss` changes: instead of the
--           row's columns it carries the row's shape, so no inline value is
--           sent:
--             {member, '-'}         the row's hash is absent (empty)
--             {member, 'i'}         inline: a `value` column, or neither
--                                   value column
--             {member, 'b'}         inline, with a `hash` column beside the
--                                   `value` (the reader decodes `value`,
--                                   and reports the conflict)
--             {member, 'x', hash}   external: the raw `hash` column, which
--                                   the reader validates
--           (the readers' rule: `value` is checked first). Sections,
--           tombstones and children are unchanged.
--
-- Replies. RESP2-safe: a Lua array truncates at its first nil on a RESP2
-- connection, so no array below ever holds one; an absent hash is an empty
-- array, and every per-entry element is {id, flat_hash_or_list} (with
-- `slim` set, a field or time-sample row's element is its shape above).
--   {fmt, 'OK', hdr, secs, dels, flds, tss, children, {items, bytes, entries}}
--   {fmt, 'OK', hdr, secs, dels, flds, tss, children, {items, bytes, entries},
--    dangling}
--       the second form only with `return_dangling` set: `dangling` is always
--       an array, the `obj:chs` members the read left out because their
--       parent is neither the root nor a live section (the "lists under
--       dangling parents stay excluded" filter below), in `SMEMBERS` order.
--       `{}` when there are none, and on an unmarked object, whose `obj:chs`
--       may be partial and so cannot enumerate them (the caller reads the
--       marker from `hdr`).
--   {fmt, 'NOT_FOUND'}
--   {fmt, 'TOO_LARGE', which, cards, bound, {items, bytes, entries}}
--       which = 'bound' (the early-out: nothing past step 2 was read),
--               'items' or 'bytes' (the running tally crossed that limit)
--       cards = {secs, dels, flds, tss, chs (-1 when unmarked), marked}
--   {fmt, 'SELFTEST', exists}
-- Errors:
--   SNAPSHOT_KEY_REFUSED: <family> <key> <raw error>   a per-key failure
--       (`redis.pcall`), <family> a `SnapshotKeyFamily` label
--   SNAPSHOT_SELFTEST_REFUSED: <key> <raw error>       the self-test's read
--   SNAPSHOT_READ_ARGS: <reason>                       a caller bug
-- Anything else is a script-level failure (NOSCRIPT, -BUSY, -LOADING, ...).

local FORMAT = 'object-snapshot-read-v1'

local hdr_key = KEYS[1]
local base = ARGV[1] .. '{' .. ARGV[2] .. '}:obj:'
local mode = ARGV[3]

if hdr_key ~= base .. 'hdr' then
  return redis.error_reply('SNAPSHOT_READ_ARGS: KEYS[1] ' .. tostring(hdr_key) ..
    ' is not the header ' .. base .. 'hdr')
end

-- SR-1.8 startup self-test: read one undeclared same-slot key of an object
-- that is never written. A store that refuses undeclared keys inside a script
-- refuses here, at startup, instead of on every snapshot read at runtime.
if mode == 'selftest' then
  local probe = base .. 'secs'
  local r = redis.pcall('EXISTS', probe)
  if type(r) == 'table' and r.err then
    return redis.error_reply('SNAPSHOT_SELFTEST_REFUSED: ' .. probe .. ' ' .. r.err)
  end
  return {FORMAT, 'SELFTEST', r}
end
if mode ~= 'read' then
  return redis.error_reply('SNAPSHOT_READ_ARGS: unknown mode ' .. tostring(mode))
end

local max_items = tonumber(ARGV[4])
local max_bytes = tonumber(ARGV[5])
local w_sec = tonumber(ARGV[6])
local w_del = tonumber(ARGV[7])
local w_fld = tonumber(ARGV[8])
local w_ts = tonumber(ARGV[9])
local marker_col = ARGV[10]
local marker_val = ARGV[11]
if not (max_items and max_bytes and w_sec and w_del and w_fld and w_ts and marker_col
    and marker_val) then
  return redis.error_reply('SNAPSHOT_READ_ARGS: missing or non-numeric limit or width')
end
-- Nil-tolerant: a caller that passes eleven arguments leaves the flag unset.
local flag_arg = ARGV[12]
if flag_arg ~= nil and flag_arg ~= '0' and flag_arg ~= '1' then
  return redis.error_reply('SNAPSHOT_READ_ARGS: return_dangling flag ' .. tostring(flag_arg) ..
    ' is not 0 or 1')
end
local return_dangling = flag_arg == '1'
local slim_arg = ARGV[13]
if slim_arg ~= nil and slim_arg ~= '0' and slim_arg ~= '1' then
  return redis.error_reply('SNAPSHOT_READ_ARGS: slim flag ' .. tostring(slim_arg) ..
    ' is not 0 or 1')
end
local slim = slim_arg == '1'

-- SR-1.4: every read goes through `redis.pcall`, and a failure becomes one
-- tagged error naming the key family and the key. `error(REFUSAL)` unwinds
-- out of `read` to the `pcall` at the bottom, which turns it into the reply;
-- any other error is rethrown untouched, so it surfaces as the script-level
-- failure it is.
local REFUSAL = {}
local function call(family, cmd, key, ...)
  local r = redis.pcall(cmd, key, ...)
  if type(r) == 'table' and r.err then
    REFUSAL.msg = 'SNAPSHOT_KEY_REFUSED: ' .. family .. ' ' .. key .. ' ' .. r.err
    error(REFUSAL)
  end
  return r
end

local function read()
  -- 1. Header. Absent means no object.
  local hdr = call('header', 'HGETALL', hdr_key)
  if #hdr == 0 then return {FORMAT, 'NOT_FOUND'} end
  local marked = 0
  for i = 1, #hdr, 2 do
    if hdr[i] == marker_col and hdr[i + 1] == marker_val then marked = 1 end
  end

  -- 2. Cardinalities and the item bound (SR-1.2 step 2): an upper bound on the
  -- items steps 3-4 tally, from the schema's fixed hash widths. Per live
  -- section: its index member, its hash, and at most one children-list
  -- element (each live section sits in exactly one parent's list). Per
  -- tombstone, field and time sample: its index member and its hash. When
  -- marked, the `obj:chs` members. Over the limit, answer before reading any
  -- entry: the cheap early-out.
  local n_secs = call('sections_index', 'SCARD', base .. 'secs')
  local n_dels = call('deleted_sections_index', 'SCARD', base .. 'dels')
  local n_flds = call('fields_index', 'ZCARD', base .. 'flds')
  local n_tss = call('time_samples_index', 'ZCARD', base .. 'tss')
  local n_chs = -1
  local bound = n_secs * (1 + w_sec + 1) + n_dels * (1 + w_del) + n_flds * (1 + w_fld)
    + n_tss * (1 + w_ts)
  if marked == 1 then
    n_chs = call('children_index', 'SCARD', base .. 'chs')
    bound = bound + n_chs
  end
  local cards = {n_secs, n_dels, n_flds, n_tss, n_chs, marked}
  if bound > max_items then
    return {FORMAT, 'TOO_LARGE', 'bound', cards, bound, {0, 0, 0}}
  end

  -- 3-4. The full read, with running tallies of items (hash fields, set and
  -- sorted-set members, list elements) and bytes (the length of every string
  -- read), checked after each command. No per-entry pre-check: one would add
  -- a command per entry and move the measured cost basis the limits were
  -- derived from, so a mid-read stop overshoots by at most the one entry that
  -- crossed the limit. On a well-formed object the item tally never crosses:
  -- step 2's bound covers it, and a crossing means a broken width or tree
  -- invariant (the anomaly the Rust side WARN-logs).
  local items, bytes, entries = 0, 0, 0
  local function add_list(t)
    items = items + #t
    for i = 1, #t do bytes = bytes + #t[i] end
  end
  local function add_hash(t)
    items = items + #t / 2
    for i = 1, #t do bytes = bytes + #t[i] end
  end
  local function over()
    if items > max_items then return 'items' end
    if bytes > max_bytes then return 'bytes' end
    return nil
  end
  local function too_large(which)
    return {FORMAT, 'TOO_LARGE', which, cards, bound, {items, bytes, entries}}
  end
  local w

  local secs = call('sections_index', 'SMEMBERS', base .. 'secs')
  add_list(secs); w = over(); if w then return too_large(w) end
  local dels = call('deleted_sections_index', 'SMEMBERS', base .. 'dels')
  add_list(dels); w = over(); if w then return too_large(w) end
  local flds = call('fields_index', 'ZRANGE', base .. 'flds', 0, -1)
  add_list(flds); w = over(); if w then return too_large(w) end
  local tss = call('time_samples_index', 'ZRANGE', base .. 'tss', 0, -1)
  add_list(tss); w = over(); if w then return too_large(w) end

  -- Index members name their hash keys directly: a section or tombstone id,
  -- a field member `{sid}:{hex(key_name)}`, a time-sample member
  -- `{sid}:{hex_f64(time)}` (`KeySchema::obj_field_from_member` /
  -- `obj_ts_from_member`).
  -- A value row's slim element (the `slim` flag, ARGV[13]): its shape, and
  -- the raw `hash` column of an external row.
  local function slim_row(id, h)
    if #h == 0 then return {id, '-'} end
    local has_value, hash = false, nil
    for i = 1, #h, 2 do
      if h[i] == 'value' then
        has_value = true
      elseif h[i] == 'hash' then
        hash = h[i + 1]
      end
    end
    if has_value then
      if hash then return {id, 'b'} end
      return {id, 'i'}
    end
    if hash then return {id, 'x', hash} end
    return {id, 'i'}
  end

  -- `value_rows`: the family holds value rows, whose element is slim when
  -- the flag is set. The read and the tallies are the same either way.
  local function read_hashes(family, ids, sub, value_rows)
    local out = {}
    local shape = slim and value_rows
    for i = 1, #ids do
      local h = call(family, 'HGETALL', base .. sub .. ids[i])
      add_hash(h)
      entries = entries + 1
      if shape then out[i] = slim_row(ids[i], h) else out[i] = {ids[i], h} end
      w = over(); if w then return nil end
    end
    return out
  end

  local out_secs = read_hashes('section', secs, 'sec:', false)
  if not out_secs then return too_large(w) end
  local out_dels = read_hashes('deleted_section', dels, 'del:', false)
  if not out_dels then return too_large(w) end
  local out_flds = read_hashes('field', flds, 'fld:', true)
  if not out_flds then return too_large(w) end
  local out_tss = read_hashes('time_sample', tss, 'ts:', true)
  if not out_tss then return too_large(w) end

  local out_ch = {}
  local function read_list(p)
    local l = call('children', 'LRANGE', base .. 'ch:' .. p, 0, -1)
    add_list(l)
    entries = entries + 1
    if #l > 0 then out_ch[#out_ch + 1] = {p, l} end
    return over()
  end
  -- Children (SR-1.2 step 4). Either way the parent set is the root plus the
  -- live sections, exactly the bounded fan-out's in `storage::diff_read`:
  -- lists under dangling parents stay excluded.
  -- The parents whose lists were excluded, for `return_dangling`; filled on
  -- a marked object only.
  local dangling = {}
  if marked == 1 then
    -- Marked: `obj:chs` is complete (every list write maintains it in the same
    -- transaction), so read only the lists it names whose parent is live.
    local chs = call('children_index', 'SMEMBERS', base .. 'chs')
    add_list(chs); w = over(); if w then return too_large(w) end
    local live = {['0'] = true}
    for i = 1, #secs do live[secs[i]] = true end
    for i = 1, #chs do
      if live[chs[i]] then
        w = read_list(chs[i]); if w then return too_large(w) end
      elseif return_dangling then
        dangling[#dangling + 1] = chs[i]
      end
    end
  else
    -- Unmarked (materialized before the marker existed; its `obj:chs` may be
    -- partial): every live section plus the root.
    for i = 1, #secs do
      w = read_list(secs[i]); if w then return too_large(w) end
    end
    w = read_list('0'); if w then return too_large(w) end
  end

  if return_dangling then
    return {FORMAT, 'OK', hdr, out_secs, out_dels, out_flds, out_tss, out_ch,
      {items, bytes, entries}, dangling}
  end
  return {FORMAT, 'OK', hdr, out_secs, out_dels, out_flds, out_tss, out_ch,
    {items, bytes, entries}}
end

local ok, reply = pcall(read)
if ok then return reply end
if reply == REFUSAL then return redis.error_reply(REFUSAL.msg) end
error(reply)
