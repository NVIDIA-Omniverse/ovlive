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
--   [3]  the caller's reliance start, in Valkey milliseconds: when the
--        caller began relying on the backup store's copy — the mirror
--        worker's item stamp (`last_store_ms`), the repair pass's Valkey
--        read, the collect's heal read. The empty string, or no ARGV[3] at
--        all, stands for an older caller and skips that check.
--
-- Returns 1 when it wrote the field, 0 when it wrote nothing.
--
-- ── It writes only when the meta exists and its stamp is the caller's ────
--
-- * **It never creates a meta.** The GC deletes meta and data together when
--   a value is collected, and a confirmation that created a meta for a
--   collected value would revive it as a meta with no data, a count nothing
--   backs. The `EXISTS` check comes first for that reason: an absent meta
--   answers `HGET` nil, which would otherwise match an empty ARGV[2].
-- * **A store after the copy declines it.** Every store re-stamps
--   `last_store_ms` (`ext_store_value.lua`). When the stamp is no longer the
--   one the caller read with the bytes, a store has happened since; the bytes
--   are the same (content-addressed), but the confirmation is left to that
--   store's own copy or to the next repair pass, so no confirmation is ever
--   written on the strength of a read older than the meta's last store.
-- * **A delete after the caller started declines it (b.g65).** When the
--   checkpoint collect deletes the value's backup copy it clears
--   `backed_up_in` and stamps `backup_cleared_ms` in the same run
--   (`ext_backup_clear.lua`). A caller whose upload or existence check began
--   at or before that stamp may have seen a copy the collect then deleted,
--   so when `backup_cleared_ms` ≥ ARGV[3] the confirmation is declined and
--   the value stays owed; the next repair pass checks it again. Both scripts
--   run on the meta's own slot, so the clear and this check are ordered.
--
-- The GC's `DEL meta data` (`gc_decrement_and_collect.lua`) removes the field
-- with the meta, and the checkpoint collect's post-delete step clears it
-- (`ext_backup_clear.lua`). Nothing else writes or removes it.
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
-- meta present every command names a present key, and both `HGET`s read
-- before the one `HSET`. So the script either runs whole or is refused before
-- it writes.

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

local start = ARGV[3]
if start ~= nil and start ~= "" then
    local cleared = redis.call("HGET", meta_key, "backup_cleared_ms")
    if cleared ~= false and tonumber(cleared) ~= nil
        and tonumber(start) ~= nil and tonumber(cleared) >= tonumber(start) then
        return 0
    end
end

-- Valkey's own clock, folded to milliseconds and formatted with `%d`, as
-- `ext_store_value.lua` stamps `last_store_ms` (no exponent, no decimal
-- point).
local t = redis.call("TIME")
local ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call("HSET", meta_key, "backed_up_in", ARGV[1] .. ":" .. string.format("%d", ms))
return 1
