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

-- layer_unmap.lua — `DeleteLayer`'s conditional unmap: remove a layer's
-- mapping and its layers-index member only while the mapping still names the
-- object the delete resolved, in one atomic step.
--
-- Run by `LayerStore::delete_layer_target` (step 2 of `DeleteLayer`, SDD
-- §7.11) as one `EVALSHA` on the data pool, on the `{session_id}` slot,
-- written to the wire at most once (`lua::LAYER_UNMAP_MAX_ATTEMPTS`): fred
-- never re-sends it after a lost connection, so a reply that never came is
-- answered to the caller as a transient error, never replayed behind its
-- back. The caller never retargets on a send that follows such an error, and
-- sends none once one settled.
--
-- ── Why conditional ──────────────────────────────────────────────────────
--
-- A plain `DEL` of the mapping, re-sent or re-run after its reply was lost or
-- after a later step of the delete failed, removes whatever the mapping names
-- at that moment — including the mapping a concurrent `CreateLayer` of the
-- same name committed after the first `DEL` landed (`create_layer_commit.lua`
-- maps from "absent"). That client was answered OK and its layer is gone. A
-- layers-index `SREM` sent after the object's teardown removes that
-- replacement's member the same way, dropping the layer from session
-- checkpoints and from `DeleteSession`'s cascade. This script compares before
-- it writes, and writes both keys in one step, before the teardown.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS (both carry the `{session_id}` hash tag — one Cluster slot):
--   [1] layer mapping   (le:{sid}:layer:{hex(layer_name)})   STRING
--   [2] layers index    (le:{sid}:layers)                    SET
--
-- ARGV:
--   [1] the mapping value expected, compared exactly: the object the delete
--       resolved, as its canonical decimal — or, on the caller's one re-send,
--       the non-canonical spelling of that object (`0123`) an earlier `OTHER`
--       reply showed the mapping holds
--   [2] hex(layer_name), the layer's member in the layers index
--
-- Return value (flat array of bulk strings):
--   {"UNMAPPED"}    — the mapping held ARGV[1]: the member was removed
--       and the mapping deleted.
--   {"ABSENT"}      — no mapping: the member was removed. A `CreateLayer`
--       commit writes the mapping and the member together, so with the
--       mapping absent no live layer owns the member.
--   {"OTHER", <id>} — the mapping holds a value other than ARGV[1], `<id>`
--       as stored: another object, or this same object spelled otherwise
--       (`0123` for `123`), which the caller tells apart. Nothing was
--       written.
-- A wrong-typed mapping fails the `GET`, and a wrong-typed index fails the
-- `SREM`, the script's first write: either error leaves both keys as found.
--
-- ── Why it runs on a full store ──────────────────────────────────────────
--
-- Deleting a layer is what frees a full store, so the unmap must not be
-- refused out of memory. Its only writes, `SREM` and `DEL`, cannot grow
-- memory and are not refused on a full store, and its one read, `GET`, is
-- not either. The script must keep no `#!lua` shebang: a shebang script
-- without `allow-oom` gets an up-front deny-oom check that refuses it.
--
-- ── Why it cannot tear during a slot migration (SDD §18.1) ───────────────
--
-- On a migrating source every command naming a key absent at that moment is
-- refused, the script aborting with its earlier writes kept. The `GET` names
-- the mapping first, so a moved or absent mapping is refused before any
-- write (`ABSENT` therefore runs only on a slot that is not migrating). The
-- `SREM` is the first write: a moved index refuses it before anything is
-- written. The `DEL` that follows names the mapping the `GET` proved present
-- in this same atomic run. The `SREM` may empty the index; the index is not
-- named again.

local mapping_key = KEYS[1]
local layers_key  = KEYS[2]

local stored = redis.call("GET", mapping_key)

if not stored then
  redis.call("SREM", layers_key, ARGV[2])
  return { "ABSENT" }
end

if stored ~= ARGV[1] then
  return { "OTHER", stored }
end

redis.call("SREM", layers_key, ARGV[2])
redis.call("DEL", mapping_key)
return { "UNMAPPED" }
