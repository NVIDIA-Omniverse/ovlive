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

-- commit_record_retract.lua — compare-and-delete of one writer's field in an
-- object's `UpdateLayer` commit-token record.
--
-- Run by the `update_layer` slow path after a TORN commit (an `EXEC` that came
-- back with an error frame) of a call that carried a client `operation_id`.
-- The torn `MULTI` may still have written that call's record, and the client's
-- re-send of the operation would read it as an earlier call's landed commit.
-- Removing the field makes the re-send apply instead.
--
-- It removes the field ONLY while it still holds exactly the torn call's
-- record, so a later commit by the same `client_id` — which overwrote the
-- field — is never touched. `HGET` + `HDEL` under Lua atomicity is the
-- compare-and-delete; nothing can interleave between them. Idempotent, so a
-- transparent replay of this script is harmless.
--
-- KEYS:
--   [1] commit-token record (le:{oid}:obj:wtok)  HASH
-- ARGV:
--   [1] the writer's `client_id` (the field)
--   [2] the torn call's exact record value
--
-- Returns 1 when the field was removed, 0 when it did not hold that record
-- (absent, or already overwritten). `HGET` on a non-hash raises WRONGTYPE,
-- which surfaces to the caller as an error: the retract failed.

if redis.call("HGET", KEYS[1], ARGV[1]) == ARGV[2] then
  return redis.call("HDEL", KEYS[1], ARGV[1])
end
return 0
