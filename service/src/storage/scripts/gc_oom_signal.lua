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

-- gc_oom_signal.lua — set one of the GC's out-of-memory signal keys on the
-- `{gc}` slot, on a store over `maxmemory` too (b.pa1).
--
-- Two `{gc}`-tagged string keys carry the GC's out-of-memory collection
-- (`storage::external_values`, SDD §11.3), and both are written only here:
--
--   * `le:ext:{gc}:oom:request` (`KeySchema::ext_oom_request`): a replica that
--     saw a store write refused `-OOM` asks the GC leader for a collection.
--     Value: the requesting replica's label; `PX` 10 s, so a request nobody
--     takes goes away on its own. Taken by the leader's `GETDEL` at the start
--     of its next pass.
--   * `le:ext:{gc}:oom:upload_hold` (`KeySchema::ext_oom_upload_hold`): set by
--     the leader for the length of one collection, so every replica's upload
--     gate holds uploads carrying external values until it ends. Value: a
--     random token; `PX` 15 s, so a leader that dies mid-collection releases
--     it; deleted (`DEL`) by the leader when the collection ends.
--
-- Either may need writing exactly when the `{gc}` shard is the full one, which
-- is why this is a script and not a plain `SET`. On a store over `maxmemory`
-- Valkey refuses a script's command that can grow memory (`SET`) only while
-- the script has written nothing yet; once any write command has run, every
-- later one runs (`gc_decrement_and_collect.lua`, "The out-of-memory opener").
-- So the `SET` is preceded by `PEXPIRE key px`, the opener: a write command
-- that cannot grow memory, changes nothing on an absent key, and on a present
-- key only sets a TTL the `SET` then replaces. The stored result is the
-- `SET`'s alone. What a run can add past `maxmemory` is this one small key,
-- living at most `px`.
--
-- Not shebang-flagged, so it runs in compatibility mode like every other
-- script of this service (SDD §21.5).
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS (one, `{gc}`-tagged, so the script routes to the `{gc}` primary):
--   [1]  the signal key (a string)
--
-- ARGV:
--   [1]  value   (non-empty)
--   [2]  px      (the key's lifetime in ms; > 0)
--
-- Returns 1. `-MISCONF`, `-READONLY` and `-NOREPLICAS` refuse the opener, the
-- first write, with nothing written.

local key = KEYS[1]
local value = ARGV[1]
local px = tonumber(ARGV[2])
if key == nil or value == nil or value == "" or px == nil or px <= 0 then
  return redis.error_reply(
    "ENCODING: gc_oom_signal expects one key, a non-empty value and a positive px")
end

-- The opener: cannot grow memory, so the `SET` after it runs on a full store.
redis.call("PEXPIRE", key, px)
redis.call("SET", key, value, "PX", px)
return 1
