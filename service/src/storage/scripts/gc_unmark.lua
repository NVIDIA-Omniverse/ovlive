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

-- gc_unmark.lua — tear down one settled member's GC decrement replay record
-- (b.yuo).
--
-- `gc_decrement_and_collect.lua` records every applied decrement in
-- `le:ext:applied:{hash_hex}` (member → claim time) so a re-run of the member
-- answers `replayed`. The record must live until no copy of any claim of the
-- member can still act, and no longer. The GC loop runs this script for each
-- member `gc_select.lua` reports settled — out of both `pending` and
-- `inflight` — for at least the replay-record retention U, on the member's
-- own `{hash_hex}` slot.
--
-- It removes the member's record only while the record's score (the claim time
-- of the latest claim whose decrement reached this hash) is at or below the
-- settle time read from `le:ext:{gc}:settled`. Every claim of the lifecycle
-- that settled precedes the settle, and a copy of a claim can act only until
-- claim time + the decrement fence F, so with U > F + clock skew the teardown
-- comes after any such copy's last chance. A record scored above the settle
-- time belongs to a newer lifecycle of the same member (re-enqueued and
-- claimed after it settled) and is kept for that lifecycle's own settle.
--
-- Both times come from the `{gc}` primary's clock (the claim's and the
-- settle's `TIME`), so the comparison is on one clock.
--
-- Idempotent and safe under two sweepers: a second run finds nothing to remove,
-- or a newer record it must keep. The key vanishes when its last member goes.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:ext:applied:{hash_hex}   (ZSET; the member's hash's own slot)
--
-- ARGV:
--   [1]  member
--   [2]  settled_ms   (the member's score in `le:ext:{gc}:settled`, as read)
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Integer: 1 when the record was removed, 0 when there was none or it is newer.

local applied_key = KEYS[1]
local member      = ARGV[1]
local settled_ms  = tonumber(ARGV[2])
if member == nil or settled_ms == nil then
  return redis.error_reply("ENCODING: expected ARGV member, settled_ms")
end

local recorded = redis.call("ZSCORE", applied_key, member)
if recorded and tonumber(recorded) <= settled_ms then
  redis.call("ZREM", applied_key, member)
  return 1
end
return 0
