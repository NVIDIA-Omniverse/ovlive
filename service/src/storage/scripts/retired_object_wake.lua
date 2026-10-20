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

-- retired_object_wake.lua — the one wake-up of every retirement route: tell
-- the subscribers still tailing a retired object's stream that the object is
-- gone, and make sure the stream expires.
--
-- Run through `layer::wake_retired_object_stream`, the one helper shared by the
-- three routes that retire an object: the `CreateLayer` overwrite tail (the
-- call's, its finisher's and the orphan sweep's), the orphan sweep's
-- retirement of any other orphan, and `DeleteLayer` (its step 3). Each route
-- removes the object only after this script ran, so no retired object loses
-- its header before its stream carries `deleted` and an expiry.
--
-- ── Commands, in order ───────────────────────────────────────────────────
--
--   1. `PEXPIRE stream <ttl>` — the opener: the stream expires even if the
--      script went no further, and the script has now run a write.
--   2. `XADD stream * payload deleted` — the wake-up a subscriber reads,
--      re-reads the layer mapping on, and ends or moves on. Uncapped: the
--      stream is about to expire. It creates the stream if it is gone, so a
--      subscriber still blocked on a stream that was already removed is
--      woken too.
--   3. `PEXPIRE stream <ttl>` — the expiry of the stream as it is now: the
--      first one was a no-op on an absent stream, and the `XADD` that just
--      re-created it set none.
--
-- ── Why it runs on a full store ──────────────────────────────────────────
--
-- The opener follows the opener rule (SDD §7.5.1): a `PEXPIRE`, which cannot
-- grow memory, is the script's first write, so the deny-oom `XADD` after it
-- runs on a full store. The script must keep no `#!lua` shebang: a shebang
-- script without `allow-oom` gets an up-front deny-oom check that refuses it
-- before its opener runs. A full store must still retire objects, because
-- deleting them is what frees it: a retirement writes one entry (about 100
-- bytes) past `maxmemory`, and the object's deletion follows at once. Because
-- the opener runs first, even a crash right after it leaves the stream with an
-- expiry. Where the rule is measured and where it is unverified: SDD §21.5. A
-- store that does not follow it may still refuse the script out of memory;
-- the callers handle that refusal (see the helper).
--
-- ── Contract ─────────────────────────────────────────────────────────────
--
-- KEYS:
--   [1] the retired object's stream (le:{oid}:obj:stream)   STREAM
--
-- ARGV:
--   [1] the stream's expiry in milliseconds
--
-- Return value: the `deleted` entry's id.
--
-- On a migrating slot the opener names a key that is absent on the source
-- whenever the stream has moved, and is refused there before any write;
-- otherwise every command names a key present at that moment.

local stream_key = KEYS[1]

redis.call("PEXPIRE", stream_key, ARGV[1])
local id = redis.call("XADD", stream_key, "*", "payload", "deleted")
redis.call("PEXPIRE", stream_key, ARGV[1])
return id
