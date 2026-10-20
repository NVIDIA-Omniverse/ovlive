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

-- ext_backup_confirm.lua — record that the backup store holds an external
-- value's bytes (SDD §22.17).
--
-- Writes the `backed_up_in` field of `le:ext:meta:{hash_hex}` as
-- `<store id>:<ms>`: the backup store whose identity is `<store id>`
-- (`Config::checkpoint_store_identity`) held the bytes stored under this hash
-- at Valkey time `<ms>`. The caller runs it only after the backup store
-- answered so: the ext-mirror worker after a successful upload, the backup
-- repair after a successful upload or an existence check that found the
-- bytes. An absent field is the "owed" state the repair acts on, so a copy
-- that never arrived (a dropped send, a failed upload, a discard at shutdown,
-- a crash) needs no record of its own: it is simply never confirmed.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  ext_meta:{hash_hex}
--
-- ARGV:
--   [1]  the backup store's identity, 16 lowercase hex characters.
--   [2]  the `last_store_ms` the caller read with the value, exactly as
--        stored: the store script's returned stamp (the mirror worker) or
--        the field as the repair's scan read it. The empty string stands for
--        "the meta carried no `last_store_ms`" (a meta an older build or a
--        recount wrote).
--
-- Returns 1 when it wrote the field, 0 when it wrote nothing.
--
-- ── It writes only when the meta exists and its stamp is the caller's ────
--
-- * **It never creates a meta.** A meta never exists without its data (SDD
--   §11.1), and the GC deletes both together when a value is collected. A
--   confirmation that created a meta for a collected value would revive it as
--   a meta with no data. The `EXISTS` check comes first for that reason: an
--   absent meta answers `HGET` nil, which would otherwise match an empty
--   ARGV[2].
-- * **A store after the copy declines it.** Every store re-stamps
--   `last_store_ms` (`ext_store_value.lua`). When the stamp is no longer the
--   one the caller read with the bytes, a store has happened since; the bytes
--   are the same (content-addressed), but the confirmation is left to that
--   store's own copy or to the next repair pass, so no confirmation is ever
--   written on the strength of a read older than the meta's last store.
--
-- The GC's `DEL meta data` (`gc_decrement_and_collect.lua`) removes the field
-- with the meta, and the checkpoint collect clears it after it deletes the
-- value's backup copy (`storage::ext_backup::clear_ext_backup`). Nothing else
-- writes or removes it.
--
-- ── Store memory ──────────────────────────────────────────────────────────
--
-- On a store out of memory the `HSET` is refused `-OOM` and nothing is
-- written: the value stays owed, and the next repair pass checks it again.
--
-- ── Slot migrations (b.cyu) ───────────────────────────────────────────────
--
-- It names one key. During a legacy cluster slot migration a command naming a
-- key absent at that moment is refused on the migrating source, aborting the
-- script; here that can only be the first command, before any write. With the
-- meta present every command names a present key. So the script either runs
-- whole or is refused before it writes.

local meta_key = KEYS[1]

if redis.call("EXISTS", meta_key) == 0 then
    return 0
end

local stamp = redis.call("HGET", meta_key, "last_store_ms")
if stamp == false then
    stamp = ""
end
if stamp ~= ARGV[2] then
    return 0
end

-- Valkey's own clock, folded to milliseconds and formatted with `%d`, as
-- `ext_store_value.lua` stamps `last_store_ms` (no exponent, no decimal
-- point).
local t = redis.call("TIME")
local ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call("HSET", meta_key, "backed_up_in", ARGV[1] .. ":" .. string.format("%d", ms))
return 1
