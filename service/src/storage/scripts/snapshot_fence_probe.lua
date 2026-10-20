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

-- snapshot_fence_probe.lua — read-only look at one object's snapshot fence
-- (Spec SDD SR-5.4).
--
-- A WATCH-path writer runs it on its WATCHed connection beside the header
-- HGETALL (EVALSHA does not disturb the connection's WATCH) to learn, in one
-- round trip and on this node's `TIME` rather than any replica's clock:
--   - whether a reader holds the fence now, and for how long;
--   - whether the key exists at all (it exists only for layers fenced within
--     the last key_linger(L)), and the layer size it stores, so the attempt is
--     classified above or below the ceiling without another round trip;
--   - the key's epoch marker (`epoch_start`, `gap_until` and the number of
--     `r:*` fields), which every fence write after a probe that found no live
--     lease changes.
-- After a WATCH-cancelled EXEC the writer runs it once more, on the data pool,
-- and compares the answer with its attempt's own probe: a live lease now, or
-- a fence key that changed since that probe (an epoch began, or began and
-- ended, after the WATCH), is an uncounted fence wait; an unchanged key is an
-- ordinary conflict.
--
-- READ-ONLY: it never writes, so it can run on a WATCHed connection and
-- inside any number of retries without disturbing anything.
--
-- It also `TYPE`-probes the fence and the writer-intent key, fence first, so
-- the WATCH path refuses a wrong-typed one with the same `WRONGTYPE_KEY`
-- reply, at the same point in its guard order, as `update_layer_fast.lua`
-- step 1d in either mode (SDD §7.5 parity).
--
-- ── Contract ──────────────────────────────────────────────────────────
--
-- KEYS (both carry the `{oid}` hash tag):
--   [1]  fence  — `le:{oid}:obj:snapfence`   (KeySchema::obj_snapshot_fence)
--   [2]  intent — `le:{oid}:obj:snapintent`  (KeySchema::obj_snapshot_intent);
--                 only its type is read
--
-- ARGV: none.
--
-- ── Return value ──────────────────────────────────────────────────────
--
--   {exists, live, retry_after_ms, size_e, size_b, bpi,
--    epoch_start, gap_until, readers}
--
--   exists          1 when the key exists, else 0 (and every later element is
--                   its "absent" value).
--   live            1 when some `r:*` has a deadline > now, else 0.
--   retry_after_ms  the latest live deadline − now when live, else 0.
--   size_e, size_b  the stored size, or -1 when the key stores none.
--   bpi             the stored bytes per item, verbatim, or "" when absent.
--   epoch_start     the stored `epoch_start` (ms), or -1 when absent. A grant
--                   that starts an epoch writes it; the release that ends the
--                   epoch clears it.
--   gap_until       the stored `gap_until` (ms), or -1 when absent. The release
--                   that ends an epoch sets it later; a grant after it passed
--                   clears it.
--   readers         the number of `r:*` fields, live or expired, or 0 when
--                   the key is absent. A release removes its own field.
--
-- The last three are the epoch marker the writer compares across its two
-- probes. They are read from fields the writing scripts already keep: no
-- script writes anything new for them. Between a probe that saw no live lease
-- and a later one, every fence write changes the marker or the key's
-- existence: a grant then starts an epoch (a new `epoch_start`); a release
-- that ends an epoch clears `epoch_start` and sets a later `gap_until`; any
-- other release lowers `readers`; expiry or UNLINK removes the key; joins and
-- renewals happen only inside an epoch that began after the first probe.
--
-- A fence or intent key holding a type other than a hash answers
-- `redis.error_reply("WRONGTYPE_KEY: <key> holds a <type>, expected hash")`,
-- the fast path's own sentence, byte for byte.

local fence_key = KEYS[1]
local intent_key = KEYS[2]

for _, key in ipairs({fence_key, intent_key}) do
  local actual = redis.call("TYPE", key)["ok"]
  if actual ~= "hash" and actual ~= "none" then
    return redis.error_reply("WRONGTYPE_KEY: " .. key .. " holds a " .. actual .. ", expected hash")
  end
end

local raw = redis.call("HGETALL", fence_key)
if #raw == 0 then
  return {0, 0, 0, -1, -1, "", -1, -1, 0}
end

local tm = redis.call("TIME")
local now = tonumber(tm[1]) * 1000 + math.floor(tonumber(tm[2]) / 1000)

local latest_live = nil
local readers = 0
local fields = {}
for i = 1, #raw, 2 do
  local name, value = raw[i], raw[i + 1]
  if string.sub(name, 1, 2) == "r:" then
    readers = readers + 1
    local deadline = tonumber(value)
    if deadline ~= nil and deadline > now and (latest_live == nil or deadline > latest_live) then
      latest_live = deadline
    end
  else
    fields[name] = value
  end
end

local size_e = tonumber(fields["size_e"]) or -1
local size_b = tonumber(fields["size_b"]) or -1
local bpi = fields["bpi"] or ""
local epoch_start = tonumber(fields["epoch_start"]) or -1
local gap_until = tonumber(fields["gap_until"]) or -1

if latest_live == nil then
  return {1, 0, 0, size_e, size_b, bpi, epoch_start, gap_until, readers}
end
return {1, 1, latest_live - now, size_e, size_b, bpi, epoch_start, gap_until, readers}
