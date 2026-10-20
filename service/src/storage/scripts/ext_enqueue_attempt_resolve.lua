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

-- ext_enqueue_attempt_resolve.lua — decide, once and for good, whether each of
-- a retirement intent's earlier enqueue attempts ran (b.yuo Part R).
--
-- ── What this is for ──────────────────────────────────────────────────────
--
-- A retirement records its owed release members on its own carrier
-- (`le:{oid}:obj:extret`) and the token of every attempt to enqueue them on
-- `le:{oid}:obj:extret:attempts`, BEFORE the attempt is sent. When an attempt's
-- reply is lost, its outcome is unknown: the enqueue may have landed. The
-- orphan sweep's drain must not send the members again until it knows that no
-- earlier attempt did — a landed attempt plus a re-drive is two decrements
-- against one reference, and the GC behind them deletes a blob under a live
-- reference.
--
-- Only the `{gc}` slot can answer, and only if the answer is recorded when the
-- attempt runs and can never change afterwards. So for each token this script
-- runs `HSETNX token "void:<now_ms>:<oid>"` and returns what is recorded:
--
--   * `landed` — the attempt-mode enqueue (`ext_release_enqueue.lua`) ran
--     with this token before the resolve, and recorded it;
--   * `void`   — it had not, and now never will: a copy of that enqueue that
--     arrives later (a slow retirement, or a command fred writes late) finds
--     `void` and writes nothing.
--
-- The enqueue's check-then-set and this `HSETNX` run on one slot, so every
-- attempt is decided exactly once, with no timing assumption — for as long as
-- the record stands. A record is removed only by its token's own sender after
-- a single clean send and a cleared carrier (F1), or by the GC loop's attempt
-- audit after V (F2), and no service send is decided more than Y after its
-- token was recorded (F3); see `gc_enqueue_attempt_audit.lua`.
--
-- ── Idempotence ───────────────────────────────────────────────────────────
--
-- A re-run (a fred re-send of this command) returns the recorded verdicts
-- unchanged: `HSETNX` writes nothing over a token already decided.
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1]  le:ext:{gc}:enqueue_attempt  (HASH: token → landed|void:<ms>:<oid>)
--
-- One `{gc}`-tagged key: single-slot, and keyed, which satisfies the b.289
-- fence against zero-key EVALSHA.
--
-- ARGV:
--   [1]     oid — the object whose carrier holds the tokens; written into each
--           `void` record so the GC loop's attempt audit can name it.
--   [2..]   the tokens to resolve (at least one).
--
-- `now_ms` is `redis.call("TIME")` on the `{gc}` primary, the same clock the
-- enqueue stamps its `landed` records with, so the audit compares like with
-- like.
--
-- ── Return value ─────────────────────────────────────────────────────────
--
-- An array with one entry per token, in ARGV order: `"landed"` or `"void"`.
-- A recorded value of any other spelling (a newer build) is returned verbatim,
-- and the caller treats it as undecided: it re-drives nothing.

if #ARGV < 2 then
  return redis.error_reply("ENCODING: ARGV must be an object id and at least one attempt token")
end

local attempts = KEYS[1]
local oid = ARGV[1]

local t = redis.call("TIME")
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local void_record = "void:" .. now_ms .. ":" .. oid

local verdicts = {}
for i = 2, #ARGV do
  local token = ARGV[i]
  redis.call("HSETNX", attempts, token, void_record)
  local recorded = redis.call("HGET", attempts, token)
  if string.sub(recorded, 1, 7) == "landed:" then
    verdicts[#verdicts + 1] = "landed"
  elseif string.sub(recorded, 1, 5) == "void:" then
    verdicts[#verdicts + 1] = "void"
  else
    verdicts[#verdicts + 1] = recorded
  end
end

return verdicts
