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

-- gc_settled_retire.lua — retire settle-carrier entries whose replay records
-- were torn down (b.yuo).
--
-- The second half of the replay-record teardown. After the GC loop ran
-- `gc_unmark.lua` for a settled member on its hash's slot, it removes the
-- member's `le:ext:{gc}:settled` entry here — but only while the entry's score
-- still equals the one `gc_select.lua` read. A member that settled again in
-- between (a newer lifecycle completed) carries a newer score; its entry stays,
-- and its newer record is torn down on a later pass.
--
-- A pair whose unmark failed is not passed here, so its entry survives for the
-- next pass to retry. Idempotent and safe under two sweepers.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:ext:{gc}:settled   (ZSET)
--
-- ARGV: one or more pairs `member, score`, the score as `gc_select.lua`
--   returned it.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- Integer: the number of entries removed.

local settled_key = KEYS[1]
local n = #ARGV
if n % 2 ~= 0 then
  return redis.error_reply("ENCODING: ARGV must be (member, score) pairs")
end

local removed = 0
for i = 1, n, 2 do
  local member = ARGV[i]
  local expected = tonumber(ARGV[i + 1])
  local current = redis.call("ZSCORE", settled_key, member)
  if current and expected and tonumber(current) == expected then
    removed = removed + redis.call("ZREM", settled_key, member)
  end
end
return removed
