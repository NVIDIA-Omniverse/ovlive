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

-- structural_cache_sync.lua — the structural cache's repair read (Spec SDD
-- SR-4.1).
--
-- One read-only call that tells a replica's cached decoded state what it
-- missed: the object header's facts, this call's commit-token record, and,
-- outside probe mode, a bounded chunk of the layer-stream entries after the
-- cached `seq` — or, in probe mode, the object's five index cardinalities.
-- It reports; it decides nothing. The Rust side
-- (`storage::structural_cache_sync`) decodes the reply, and the caller maps
-- the header facts, checks the token and replays the entries. Beyond needing
-- a range to read (NO_RANGE below), the chunk ignores the header facts: a
-- `superseded` header, or a header `seq` below the cached one (an empty OK
-- chunk), is reported beside it for the caller to act on.
--
-- Read-only: no command below writes. No `#!lua` shebang, like every script
-- in this directory, so it runs in compat mode (SDD §7.5.1) and keeps running
-- on a store over `maxmemory`.
--
-- KEYS (all declared, all carrying the object's `{oid}` hash tag, so the
-- script runs in that one slot and touches no undeclared key):
--   KEYS[1]  the object header        `{prefix}{oid}:obj:hdr`
--   KEYS[2]  the layer stream         `{prefix}{oid}:obj:stream`
--   KEYS[3]  the commit-token record  `{prefix}{oid}:obj:wtok`
-- In probe mode only, five more (a repair call declares exactly the three
-- above):
--   KEYS[4]  the sections index       `{prefix}{oid}:obj:secs`
--   KEYS[5]  the tombstones index     `{prefix}{oid}:obj:dels`
--   KEYS[6]  the fields index         `{prefix}{oid}:obj:flds`
--   KEYS[7]  the time-samples index   `{prefix}{oid}:obj:tss`
--   KEYS[8]  the children-list index  `{prefix}{oid}:obj:chs`
--
-- ARGV (every position always present; '' means "not given"):
--   ARGV[1]  the cached entry's `seq`, or '' for PROBE MODE: header, token
--            and the index cardinalities (KEYS[4..8]); the stream is not read
--   ARGV[2]  entry bound: at most this many stream entries per call
--            (`STRUCTURAL_REPAIR_MAX_ENTRIES`)
--   ARGV[3]  byte bound: at most this many bytes of stream entries per call
--            (`STRUCTURAL_REPAIR_MAX_BYTES`)
--   ARGV[4]  the call's `client_id`, the commit-token record's field
--   ARGV[5]  the target `seq` that caps the range, or '' for the header `seq`
--   ARGV[6]  page size: entries per `XRANGE` (`STRUCTURAL_REPAIR_PAGE_ENTRIES`)
--
-- Replies. RESP2-safe: a Lua array truncates at its first nil on a RESP2
-- connection, so no array below ever holds one. An optional value is an
-- array: `{}` when absent, `{value}` when present.
--
--   {fmt, header, token, chunk}
--
--   fmt = 'structural-cache-sync-v2' (`FORMAT` below). The Rust decoder
--   refuses any other tag as `Internal`, so a change to any reply shape here
--   changes the tag with it. v2 added the probe chunk's cardinalities.
--
--   header = {exists, seq, commit_chain, superseded, chs_complete}
--     exists        1 when the header key holds any field, else 0 (and the
--                   other four are their empty forms)
--     seq           {decimal} when the `seq` column is a decimal of at most
--                   2^53 - 1 (the ceiling both commit paths enforce); {} when
--                   it is absent or anything else. Stricter than the fast
--                   path's partial-header test, `tonumber(hdr["seq"])`: a value
--                   such as `7.0` passes that test and reads {} here. Both
--                   commit paths write `seq` as a plain decimal, so only a
--                   corrupted header tells the two apart.
--     commit_chain  the `commit_chain` column, '' when absent
--     superseded    the `superseded` column, optional
--     chs_complete  the `chs_complete` column, optional
--
--   token = {type, record}
--     type          `TYPE` of KEYS[3]: 'none', 'hash', or a foreign type
--     record        `HGET KEYS[3] <client_id>`, optional; read only when the
--                   type is 'hash' or 'none', so a foreign type is reported
--                   as data instead of failing the script. Returned verbatim
--                   in either record shape — `<token>:<seq>:<server_ts>`, or
--                   `op.<id>:<seq>:<server_ts>:<token>:<request_digest>` for
--                   a call with a client operation id — and matched in Rust
--                   (`structural_cache_repair::check_commit_token_record`)
--
--   chunk, one of:
--     {'PROBE', cards}                   probe mode: the stream was not read;
--                                        cards = {secs, dels, flds, tss, chs},
--                                        each optional: {n}, the `SCARD` /
--                                        `ZCARD` of its KEYS[4..8] key (0 for
--                                        an absent key), or {} when the key
--                                        holds a foreign type. Like the
--                                        commit-token record, a foreign type
--                                        is reported as data (its `TYPE` is
--                                        read first, and the count only for
--                                        'none' or the index's own type), so
--                                        it never fails the probe's other
--                                        answers. Read whatever the header
--                                        says — `chs` is reported on an
--                                        unmarked object too, and the caller
--                                        decides what it means. They are the
--                                        cardinalities
--                                        `object_snapshot_read.lua`'s step 2
--                                        reads, so the caller can compute that
--                                        script's item bound before any full
--                                        read (the cold start's route,
--                                        `storage::structural_cache_cold_start`)
--     {'NO_RANGE'}                       not probe mode, but the header is
--                                        absent or `seq` is not a decimal, so
--                                        there is no range to read
--     {'OK', end, entries, reached, bytes}
--     {'TOO_LARGE', end, id, entry_bytes}
--   end       the range's last `seq` (decimal): the header `seq`, or the
--             target `seq` when that is lower
--   entries   the stream entries `{from+1}-0 .. {end}-0`, in id order, each
--             verbatim as `XRANGE` returns it: {id, {field, value, ...}}
--   reached   1 when the chunk covers the range through `end`: the last
--             entry is `{end}-0`, or the range ran out of entries (including
--             a range that is empty because `from` >= `end`). 0 when the
--             chunk stopped at the entry bound or the byte bound first
--   bytes     the chunk's byte tally
--   id, entry_bytes   the first pending entry, alone larger than the byte
--             bound, and its size (no entry is returned with it)
--
-- The progress rule. An entry's size is its id plus every field name and
-- value. Entries are read a page at a time (`XRANGE ... COUNT <n>`, `n` the
-- page size or the entries left under the entry bound, whichever is fewer;
-- each page after the first starts exclusively after the last id read,
-- `(id`), never as one range truncated afterwards, so one call copies at most
-- the byte bound plus one page into Lua (at a one-entry page, the bound plus
-- one entry). An entry that would take the tally past the byte bound ends the
-- chunk before it; if that entry is the first, the answer is TOO_LARGE. The
-- caller continues from the last returned entry.
--
-- Errors:
--   STRUCTURAL_SYNC_KEY_REFUSED: <what> <key> <raw error>   a read of one key
--       failed (`redis.pcall`); <what> is 'header', 'commit_token', 'stream'
--       or, in probe mode, an index read: 'sections_index',
--       'deleted_sections_index', 'fields_index', 'time_samples_index' or
--       'children_index' (a wrong-typed index is data, not this error; it
--       can only come from the store refusing the `TYPE` or the count)
--   STRUCTURAL_SYNC_ARGS: <reason>                           a caller bug
-- Anything else is a script-level failure (NOSCRIPT, -BUSY, -LOADING, ...).

local FORMAT = 'structural-cache-sync-v2'

-- 2^53 - 1: the highest `seq` either commit path writes, and the highest
-- integer a Lua number holds exactly.
local MAX_SEQ = 9007199254740991

-- A decimal `seq` within the ceiling, as a number; nil for anything else.
local function decimal(s)
  if type(s) ~= 'string' or not string.match(s, '^%d+$') then return nil end
  local n = tonumber(s)
  if n == nil or n > MAX_SEQ then return nil end
  return n
end

if #ARGV ~= 6 then
  return redis.error_reply('STRUCTURAL_SYNC_ARGS: expected 6 arguments, got ' .. #ARGV)
end
local from_arg = ARGV[1]
local probe = from_arg == ''

-- Probe mode declares the five index keys as well; a repair call does not.
local key_count = 3
if probe then key_count = 8 end
if #KEYS ~= key_count then
  return redis.error_reply('STRUCTURAL_SYNC_ARGS: expected ' .. key_count .. ' keys, got ' ..
    #KEYS)
end
local hdr_key = KEYS[1]
local stream_key = KEYS[2]
local wtok_key = KEYS[3]
-- nil outside probe mode, which reads none of them.
local secs_key = KEYS[4]
local dels_key = KEYS[5]
local flds_key = KEYS[6]
local tss_key = KEYS[7]
local chs_key = KEYS[8]

local max_entries = tonumber(ARGV[2])
local max_bytes = tonumber(ARGV[3])
local client_id = ARGV[4]
local target_arg = ARGV[5]
local page = tonumber(ARGV[6])
if not (max_entries and max_bytes and page) or max_entries < 1 or max_bytes < 1
    or page < 1 then
  return redis.error_reply('STRUCTURAL_SYNC_ARGS: entry bound, byte bound and page size ' ..
    'must be positive numbers')
end
local from = nil
if not probe then
  from = decimal(from_arg)
  if from == nil then
    return redis.error_reply('STRUCTURAL_SYNC_ARGS: entry seq ' .. from_arg ..
      ' is not a decimal seq')
  end
end
local target = nil
if target_arg ~= '' then
  target = decimal(target_arg)
  if target == nil then
    return redis.error_reply('STRUCTURAL_SYNC_ARGS: target seq ' .. target_arg ..
      ' is not a decimal seq')
  end
end

-- Every read goes through `redis.pcall`, and a failure becomes one tagged
-- error naming what was read and the key. `error(REFUSAL)` unwinds out of
-- `read` to the `pcall` at the bottom, which turns it into the reply; any
-- other error is rethrown untouched, so it surfaces as the script-level
-- failure it is.
local REFUSAL = {}
local function call(what, cmd, key, ...)
  local r = redis.pcall(cmd, key, ...)
  if type(r) == 'table' and r.err then
    REFUSAL.msg = 'STRUCTURAL_SYNC_KEY_REFUSED: ' .. what .. ' ' .. key .. ' ' .. r.err
    error(REFUSAL)
  end
  return r
end

local function opt(v)
  if v then return {v} end
  return {}
end

local function read()
  -- 1. Header facts.
  local hdr = call('header', 'HGETALL', hdr_key)
  local cols = {}
  for i = 1, #hdr, 2 do cols[hdr[i]] = hdr[i + 1] end
  local exists = 0
  if #hdr > 0 then exists = 1 end
  local header_seq = decimal(cols['seq'])
  local seq_opt = {}
  if header_seq then seq_opt = {cols['seq']} end
  local header = {exists, seq_opt, cols['commit_chain'] or '', opt(cols['superseded']),
    opt(cols['chs_complete'])}

  -- 2. This call's commit-token record, as data. `TYPE` answers for a key of
  -- any type; `HGET` raises WRONGTYPE on a foreign one, so it is skipped
  -- there and the type alone is reported.
  local wtok_type = call('commit_token', 'TYPE', wtok_key)['ok']
  local record = {}
  if wtok_type == 'hash' or wtok_type == 'none' then
    record = opt(call('commit_token', 'HGET', wtok_key, client_id))
  end
  local token = {wtok_type, record}

  -- Probe mode ends here, with the index cardinalities: O(1) each, the same
  -- five `object_snapshot_read.lua`'s step 2 reads. Each is data, as the
  -- commit-token record is: `TYPE` answers for a key of any type, and the
  -- count, which raises WRONGTYPE on a foreign one, runs only when the key is
  -- absent or of the index's own type. So a corrupt index never takes the
  -- header facts and the token record down with it.
  if probe then
    -- {n} for an absent key or one of type `want`; {} for a foreign type.
    -- `count` runs only in the first case.
    local function card(key_type, want, count)
      if key_type == 'none' or key_type == want then return {count()} end
      return {}
    end
    local cards = {
      card(call('sections_index', 'TYPE', secs_key)['ok'], 'set',
        function() return call('sections_index', 'SCARD', secs_key) end),
      card(call('deleted_sections_index', 'TYPE', dels_key)['ok'], 'set',
        function() return call('deleted_sections_index', 'SCARD', dels_key) end),
      card(call('fields_index', 'TYPE', flds_key)['ok'], 'zset',
        function() return call('fields_index', 'ZCARD', flds_key) end),
      card(call('time_samples_index', 'TYPE', tss_key)['ok'], 'zset',
        function() return call('time_samples_index', 'ZCARD', tss_key) end),
      card(call('children_index', 'TYPE', chs_key)['ok'], 'set',
        function() return call('children_index', 'SCARD', chs_key) end),
    }
    return {FORMAT, header, token, {'PROBE', cards}}
  end
  if header_seq == nil then return {FORMAT, header, token, {'NO_RANGE'}} end

  -- 3. The stream chunk `{from+1}-0 .. {end}-0`.
  local last_seq = header_seq
  if target and target < last_seq then last_seq = target end
  local end_str = string.format('%d', last_seq)
  local end_id = end_str .. '-0'
  local entries, bytes, reached = {}, 0, 0
  if from >= last_seq then
    return {FORMAT, header, token, {'OK', end_str, entries, 1, bytes}}
  end
  local cursor = string.format('%d', from + 1) .. '-0'
  while true do
    local want = page
    if max_entries - #entries < want then want = max_entries - #entries end
    if want < 1 then break end
    local got = call('stream', 'XRANGE', stream_key, cursor, end_id, 'COUNT', want)
    for i = 1, #got do
      local e = got[i]
      local n = #e[1]
      local f = e[2]
      for j = 1, #f do n = n + #f[j] end
      if bytes + n > max_bytes then
        if #entries == 0 then
          return {FORMAT, header, token, {'TOO_LARGE', end_str, e[1], n}}
        end
        return {FORMAT, header, token, {'OK', end_str, entries, 0, bytes}}
      end
      bytes = bytes + n
      entries[#entries + 1] = e
    end
    if #got < want then reached = 1; break end
    local last = got[#got][1]
    if last == end_id then reached = 1; break end
    cursor = '(' .. last
  end
  return {FORMAT, header, token, {'OK', end_str, entries, reached, bytes}}
end

local ok, reply = pcall(read)
if ok then return reply end
if reply == REFUSAL then return redis.error_reply(REFUSAL.msg) end
error(reply)
