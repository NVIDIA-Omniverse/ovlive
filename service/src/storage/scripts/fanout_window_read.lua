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

-- fanout_window_read.lua — one window of the structural WATCH read (SDD §10.5).
--
-- The structural `UpdateLayer` WATCH path reads its object through the
-- unfenced bounded fan-out (`storage::diff_read`, `FanoutReplies::Bounded`).
-- Its rounds 2 and 3 are issued in windows, and each window is one EVALSHA of
-- this script: the window's value rows, section and tombstone hashes,
-- children-list lengths and ranged children lists, read in one call. Round 1
-- (the header, the index counts and the index pages) stays plain commands,
-- and two kinds of round-2/3 window do not come here either: a lone children
-- list whose modelled shard time alone exceeds the budget below, issued as
-- one plain LRANGE, and every window after the read's script fallback.
-- Rust side: `storage::diff_read`'s `issue_script_window` and
-- `decode_fanout_window_reply`.
--
-- Not the atomic snapshot read (`object_snapshot_read.lua`): this script reads
-- one window, never the whole object, and the window is sized before it is
-- sent so that its modelled shard time stays within the per-window budget
-- (`config::FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS`). Nothing fences the read
-- between windows; the caller proves its state by re-reading the header.
--
-- It transports raw per-command answers and decides nothing: every command a
-- plain window would issue for an entry is issued here, with the same
-- arguments, and its answer returned as it came. A value row is read without
-- its value (`HSTRLEN` of `value`, `HMGET` of the other columns, and the
-- follow-ups below), so the reply stays bounded whatever the row holds. One
-- window's commands run atomically, so the commands of one row cannot be torn
-- across another client's write.
--
-- Read-only: no command below writes. No `#!lua` shebang, like every script
-- in this directory, so it runs in compat mode (SDD §7.5.1) and keeps running
-- on a store over `maxmemory`.
--
-- KEYS[1]   the object header, `{prefix}{oid}:obj:hdr`. The only declared
--           key; it pins the slot. Every key the script reads is built below
--           from ARGV[1..2] and an entry's id, and carries the same `{oid}`
--           hash tag, so it is undeclared but same-slot. The script refuses
--           to run unless KEYS[1] is exactly the header ARGV[1..2] name.
-- ARGV[1]   key prefix (`KeySchema::prefix`)      ARGV[2]  object id
-- ARGV[3]   a field row's columns, space-separated, `hash` first
-- ARGV[4]   a time-sample row's columns, space-separated, `hash` first
--           (`storage::diff_read::value_row_columns`; passed in so the
--           column lists have one source)
-- ARGV[5..] the window's entries, in window order: an op, then an id; an
--           `r` entry also a length.
--             f <member>        field row        base .. 'fld:' .. member
--             t <member>        time-sample row  base .. 'ts:' .. member
--             s <section id>    HGETALL          base .. 'sec:' .. id
--             d <section id>    HGETALL          base .. 'del:' .. id
--             l <parent id>     LLEN             base .. 'ch:' .. id
--             r <parent id> n   LRANGE 0 n-1     base .. 'ch:' .. id, n >= 1
--           where base = ARGV[1] .. '{' .. ARGV[2] .. '}:obj:'.
--
-- A value row (`f`, `t`): `HSTRLEN key value`, then `HMGET key <columns>`;
-- then `HEXISTS key value` only when the value length is 0 and the row has
-- its `hash` column or no column at all; then `EXISTS key` only when
-- `HEXISTS` answered 0 and the row has no column at all. That is the plain
-- value-row read's rule (`ValueRowReply::asks_value_exists` /
-- `asks_row_exists`); the Rust decode rejects a reply whose follow-ups differ
-- from it.
--
-- Replies. RESP2-safe: a Lua array truncates at its first nil on a RESP2
-- connection, so no array below ever holds a nil or a false.
--   {fmt, 'OK', e_1, ..., e_N}   one element per entry, in entry order:
--     f, t   {value_len, mask, value_exists, row_exists, c_1, ..., c_k}
--            bit i-1 of `mask` set: column i is present and c_i is its
--            value; unset: the column is absent and c_i is ''.
--            `value_exists` / `row_exists`: -1 when not asked, else the
--            integer answer (0 or 1; EXISTS counts the key).
--     s, d   the flat HGETALL array, {} for an absent hash
--     l      the LLEN integer
--     r      the LRANGE array
-- Errors:
--   FANOUT_WINDOW_REFUSED: <entry> <command> <raw error>
--       a command refused (`redis.pcall`): <entry> is the entry's 1-based
--       index, <command> one of HSTRLEN, HMGET, HEXISTS, EXISTS, HGETALL,
--       LLEN, LRANGE
--   FANOUT_WINDOW_ARGS: <reason>   a caller bug: KEYS[1] mismatch, an unknown
--       op, a bad length, a missing id or a bad column list
-- Anything else is a script-level failure (NOSCRIPT, -BUSY, -LOADING, ...).

local FORMAT = 'fanout-window-read-v1'
local FIRST_ENTRY = 5

local hdr_key = KEYS[1]
local base = ARGV[1] .. '{' .. ARGV[2] .. '}:obj:'

local function args_error(reason)
  return redis.error_reply('FANOUT_WINDOW_ARGS: ' .. reason)
end

if hdr_key ~= base .. 'hdr' then
  return args_error('KEYS[1] ' .. tostring(hdr_key) .. ' is not the header ' .. base .. 'hdr')
end

-- A row's column list: space-separated, `hash` first (the row's shape reads
-- it, and the follow-up rule asks for it by position). At most 30 columns, so
-- the mask stays an exact integer.
local function columns(arg)
  local cols = {}
  for col in string.gmatch(arg or '', '%S+') do cols[#cols + 1] = col end
  if #cols == 0 or #cols > 30 or cols[1] ~= 'hash' then return nil end
  return cols
end
local field_cols = columns(ARGV[3])
local sample_cols = columns(ARGV[4])
if not field_cols or not sample_cols then
  return args_error('a column list is empty, too long or does not start with hash')
end

-- Validate every entry before any command runs, so a malformed request reads
-- nothing.
do
  local i, n, entries = FIRST_ENTRY, #ARGV, 0
  while i <= n do
    entries = entries + 1
    local op, id = ARGV[i], ARGV[i + 1]
    if id == nil or id == '' then
      return args_error('entry ' .. entries .. ' has no id')
    end
    i = i + 2
    if op == 'r' then
      local len = tonumber(ARGV[i])
      if not len or len < 1 or len ~= math.floor(len) then
        return args_error('entry ' .. entries .. ' has a bad length ' .. tostring(ARGV[i]))
      end
      i = i + 1
    elseif op ~= 'f' and op ~= 't' and op ~= 's' and op ~= 'd' and op ~= 'l' then
      return args_error('entry ' .. entries .. ' has an unknown op ' .. tostring(op))
    end
  end
  if entries == 0 then
    return args_error('the window has no entries')
  end
end

-- Every command goes through `redis.pcall`, and a refusal becomes one tagged
-- error naming the entry and the command. `error(REFUSAL)` unwinds out of
-- `read` to the `pcall` at the bottom, which turns it into the reply; any
-- other error is rethrown untouched, so it surfaces as the script-level
-- failure it is.
local REFUSAL = {}
local entry = 0
local function call(cmd, key, ...)
  local r = redis.pcall(cmd, key, ...)
  if type(r) == 'table' and r.err then
    REFUSAL.msg = 'FANOUT_WINDOW_REFUSED: ' .. entry .. ' ' .. cmd .. ' ' .. r.err
    error(REFUSAL)
  end
  return r
end

local function value_row(key, cols)
  local value_len = call('HSTRLEN', key, 'value')
  local got = call('HMGET', key, unpack(cols))
  local out = {value_len, 0, -1, -1}
  local mask, bit, any = 0, 1, false
  for j = 1, #cols do
    local v = got[j]
    if v then
      mask = mask + bit
      any = true
      out[4 + j] = v
    else
      out[4 + j] = ''
    end
    bit = bit * 2
  end
  out[2] = mask
  -- A row with its `hash`, or with no column at all, and no value length may
  -- still hold an empty `value`; then a row with no column and no `value` is
  -- absent or holds only columns no writer writes.
  if value_len == 0 and (got[1] or not any) then
    out[3] = call('HEXISTS', key, 'value')
    if out[3] == 0 and not any then
      out[4] = call('EXISTS', key)
    end
  end
  return out
end

local function read()
  local out = {FORMAT, 'OK'}
  local i, n = FIRST_ENTRY, #ARGV
  while i <= n do
    entry = entry + 1
    local op, id = ARGV[i], ARGV[i + 1]
    i = i + 2
    local e
    if op == 'f' then
      e = value_row(base .. 'fld:' .. id, field_cols)
    elseif op == 't' then
      e = value_row(base .. 'ts:' .. id, sample_cols)
    elseif op == 's' then
      e = call('HGETALL', base .. 'sec:' .. id)
    elseif op == 'd' then
      e = call('HGETALL', base .. 'del:' .. id)
    elseif op == 'l' then
      e = call('LLEN', base .. 'ch:' .. id)
    else
      -- 'r', validated above.
      local len = tonumber(ARGV[i])
      i = i + 1
      e = call('LRANGE', base .. 'ch:' .. id, 0, len - 1)
    end
    out[#out + 1] = e
  end
  return out
end

local ok, reply = pcall(read)
if ok then return reply end
if reply == REFUSAL then return redis.error_reply(REFUSAL.msg) end
error(reply)
