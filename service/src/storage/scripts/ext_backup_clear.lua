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

-- ext_backup_clear.lua — the checkpoint collect's post-delete step on an
-- external value's own meta: withdraw its backup confirmation and date the
-- delete, in one run on the value's slot (b.g65, SDD §22.17).
--
-- The collect runs it right after the backup store answered that it deleted
-- the value's copy (`checkpoint::checkpointer`, the collect's delete).
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  ext_meta:{hash_hex}
--
-- Returns 0 when the meta is absent ("not live": Valkey no longer holds the
-- value, and nothing is written), 1 when it exists ("live": it wrote).
--
-- On a live value it writes, in this order:
-- * `HDEL backed_up_in`: the copy it confirmed is gone, so the value reads as
--   owed and the backup repair re-sends it. This is also the out-of-memory
--   opener (§7.5.1): a write command that cannot grow memory, so on a store
--   over `maxmemory` the next command still lands.
-- * `HSET backup_cleared_ms <Valkey ms>`: when the collect deleted the copy.
--   `ext_backup_confirm.lua` declines a confirmation whose caller started
--   relying on the copy (its upload or existence check) at or before this
--   time, so a confirmation that raced the delete cannot write
--   `backed_up_in` after this clear. A later clear overwrites it, and the
--   GC's `DEL meta data` removes it with the meta.
--
-- It never creates a meta: the `EXISTS` check comes first for that reason.
-- (A meta can exist without its data: the DR restore Job's recount creates
-- one for a value whose bytes only the backup store holds. This script acts
-- on such a meta as on any other.)
--
-- ── Slot migrations (b.cyu) ───────────────────────────────────────────────
--
-- It names one key, and the opening `EXISTS` names it, so a meta that moved
-- is refused before any write; with the meta present every later command names
-- a present key, and the `HDEL` cannot empty a hash that keeps `ref_count`.

local meta_key = KEYS[1]

if redis.call("EXISTS", meta_key) == 0 then
    return 0
end

redis.call("HDEL", meta_key, "backed_up_in")

-- Valkey's own clock, folded to milliseconds and formatted with `%d`, as
-- `ext_store_value.lua` stamps `last_store_ms`.
local t = redis.call("TIME")
local ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call("HSET", meta_key, "backup_cleared_ms", string.format("%d", ms))
return 1
