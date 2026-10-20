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

-- ext_read_hold.lua — take, or renew, one holder's external-value GC read hold
-- (b.yuo, the catch-up miss protocol).
--
-- A catch-up whose build found an external-value blob missing takes a hold
-- before it reads the layer's state again, renews it every few seconds while
-- the re-read and rebuild run, and releases it (a plain `HDEL` of its field)
-- when the rebuild ends. While any holder's field is live, `gc_select.lua`
-- selects no release member enqueued within its margin of the EARLIEST live
-- start, or later. So a blob the fresh read references cannot be collected
-- under the rebuild: a release of it enqueued after the hold began is not
-- selected, and a release enqueued before frees nothing the fresh read holds,
-- because a reference written since re-stored the blob with its own `+1`.
--
-- The key is a HASH with one field per holder, so holders never extend each
-- other's holds and a released hold stops clipping at once:
--
--   field  `<object_id>:<token>`  (the token: random, one per protocol run)
--   value  `<start_ms>:<expires_ms>`, both on this primary's `TIME`
--
-- A field is live while `expires_ms > now`, the rule `gc_select.lua` clips by,
-- so a field no GC pass would still honour is never extended.
--
-- Modes (ARGV[1]):
--
--   * `take`  — if THIS field is live, extend it and keep its start (so a
--     re-sent `take` is harmless); otherwise start a new hold at now. Writes
--     the field, sets the key's expiry to `ttl_ms` from now (every field
--     shares one TTL, so the key outlives every live field and goes away after
--     the last write's TTL), then prunes every other expired field, and
--     returns the start. The write comes before the prune so that no command
--     names an emptied key (see the ordering invariant at the take code).
--   * `renew` — if THIS field is present and live, extend it, refresh
--     the key's expiry and return its start. Otherwise delete it and return 0
--     (LAPSED), writing nothing else: a renewal that runs after its hold
--     lapsed (a late write, or a field lost to a failover) never revives it,
--     and the holder learns the protection may have had a gap.
--
-- A field whose holder died stops clipping at its `expires_ms`; the next
-- `take` prunes it, and the key expires on its own.
--
-- ── The out-of-memory opener (b.pa1) ──────────────────────────────────────
--
-- A take and a live renew open their writes with `HDEL hold_key ":oom-gate"`,
-- the opener, before the `HSET`. On a store over `maxmemory` Valkey refuses a
-- script's command that can grow memory (`HSET`) only while the script has
-- written nothing yet; once any write command has run, every later one runs
-- (`gc_decrement_and_collect.lua`, "The out-of-memory opener"). The opener
-- cannot grow memory and removes nothing: fields are `<object_id>:<token>`,
-- so no field starts with `:`, and the key is a hash or absent (the read
-- before it raises on any other type). So a catch-up can take and renew its
-- hold while the `{gc}` shard is full, which the GC's out-of-memory
-- collection, collecting at any age, relies on for readers (SDD §11.3). What
-- a run can add past `maxmemory` is one field of under 100 B per holder,
-- living at most one TTL; the stored result is what it was without the
-- opener. A renew of a lapsed field writes only its `HDEL`, which cannot grow
-- memory either.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS (one, `{gc}`-tagged, so the script routes to the `{gc}` primary):
--   [1]  le:ext:{gc}:read_hold   (HASH)
--
-- ARGV:
--   [1]  mode                    ("take" | "renew")
--   [2]  field                   (`<object_id>:<token>`, non-empty)
--   [3]  ttl_ms                  (the hold's lifetime from this call; > 0)
--
-- Returns the hold's start in ms (an integer), or 0 from `renew` for a lapsed
-- or missing field. A key that is not a hash raises `WRONGTYPE` from its first
-- read, and nothing is written: the caller treats any error as a hold not
-- taken (or not renewed). Such a key also fails every GC pass's `gc_select.lua`
-- until it is removed.

local hold_key = KEYS[1]
local mode = ARGV[1]
local field = ARGV[2]
local ttl_ms = tonumber(ARGV[3])
if (mode ~= "take" and mode ~= "renew") or field == nil or field == ""
    or ttl_ms == nil or ttl_ms <= 0 then
  return redis.error_reply(
    "ENCODING: ext_read_hold expects ARGV mode (take|renew), a field and a positive ttl_ms")
end

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

-- `<start>:<expires>` → start, expires (nil, nil for anything else).
local function parse(value)
  if not value then return nil, nil end
  local s, e = string.match(value, "^(%d+):(%d+)$")
  if not s then return nil, nil end
  return tonumber(s), tonumber(e)
end

local function write(start)
  -- The opener (see "The out-of-memory opener" above): never a field name.
  redis.call("HDEL", hold_key, ":oom-gate")
  redis.call("HSET", hold_key, field,
    string.format("%d:%d", start, now_ms + ttl_ms))
  redis.call("PEXPIRE", hold_key, ttl_ms)
end

if mode == "renew" then
  local start, expires = parse(redis.call("HGET", hold_key, field))
  if start == nil or expires <= now_ms then
    redis.call("HDEL", hold_key, field)
    return 0
  end
  write(start)
  return start
end

-- take: start or extend this field, then prune every other expired field.
--
-- Ordering invariant (b.cyu): write the own field BEFORE any prune. A prune
-- that empties the key before the write would make the write name an absent
-- key, which a migrating cluster slot refuses mid-script, tearing the run
-- (the expired fields gone, no hold taken). Written first, the own field keeps
-- the key non-empty for every prune after it. The stored result is the same
-- as pruning first: the own field is never pruned (it was just written live),
-- and every other expired or unparseable field is. The opener `write` starts
-- with removes nothing, so it never empties the key, and it runs before any
-- other write: a refusal of it, on an absent key in a migrating slot, tears
-- nothing.
local all = redis.call("HGETALL", hold_key)
local own_start = nil
for i = 1, #all, 2 do
  if all[i] == field then
    local start, expires = parse(all[i + 1])
    if start ~= nil and expires > now_ms then
      own_start = start
    end
  end
end
local start = own_start or now_ms
write(start)
for i = 1, #all, 2 do
  if all[i] ~= field then
    local s, expires = parse(all[i + 1])
    if s == nil or expires <= now_ms then
      redis.call("HDEL", hold_key, all[i])
    end
  end
end
return start
