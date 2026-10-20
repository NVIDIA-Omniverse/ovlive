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

-- update_layer_fast.lua — atomic server-side commit for `update_layer`, in two
-- modes (b.67u; structural mode: Spec SDD SR-6).
--
-- Collapses the legacy WATCH/HGETALL/MULTI/EXEC client-side OCC loop to a
-- single `EVALSHA` round-trip. Because Lua runs atomically under the Valkey
-- server's single thread, there is no possibility of interleaving with
-- other writers on the same object: no attempt loses a conflict, and the
-- retry loop compresses to retrying only on transient connection errors
-- (and, in structural mode, a missed `seq` fence or a diverged cache). The
-- one refusal the caller waits out is `FENCED` (step 1d): a fenced snapshot
-- reader holds the object, and the caller waits for it and re-runs the
-- script, uncounted against its retry budget (Spec SDD SR-5.4). (The writer
-- guards' `OVERTAKEN` and `PREDECESSOR` are success replies too, but end the
-- call.)
--
-- ── Two modes, one guard skeleton ─────────────────────────────────────────
--
-- The mode slot (`ARGV[SHARED_ARGV + 1]`, see the contract below) selects:
--
--   * `fast` — the field / time-sample fast path. `SF` / `ST` records, with
--     the sequence-derived values computed in-script from the header (the
--     TOCTOU rule below). `update_layer_once_lua` dispatches it only when
--     `lua::is_fast_path(&delta)` holds.
--   * `structural` — a structural plan computed in Rust on a cached state
--     (`storage::structural::commit_structural`). The records carry Rust's
--     absolute values, which is sound only because this mode is FENCED: it
--     refuses (`STALE`) unless the header's `seq` and `commit_chain` still
--     equal the cut the plan was made at, and it refuses (`CACHE_DIVERGED`)
--     unless every key the plan rewrites still holds what the plan believed.
--     Under the `lua` backend `update_layer` dispatches every reset and
--     structural-command delta to it (`LayerStore::update_layer_once_structural`,
--     planning on the per-replica structural cache); an object the cache
--     cannot hold, and the `lua-watch-structural` and `txn` backends, take
--     the WATCH/MULTI/EXEC path instead.
--
-- Both modes run the same code for every guard they share, the commit-token
-- check and record, the `pcall`ed `XADD`, the external-value fold and the
-- settle's release arms, the `commit_chain` fold and the reply. Only the
-- record vocabulary, its pre-pass and its apply pass, the settle's `obj:ext`
-- count rule (see "External-value reference accounting") and the header
-- columns differ by mode. The guard order (Spec SDD SR-6.2), in this script's
-- step numbers:
--
--   guard  step  check                                         reply
--          0     `KEYS[10]` is not the object's migration      ENCODING
--                sentinel (a caller bug; see "Migration
--                sentinel")
--   1      1     header absent                                 NOT_FOUND
--   2      1a    `seq` missing / malformed                     PARTIAL_HEADER
--   3      1b    a commit under this call's match key recorded ALREADY_LANDED
--                (unchained operation ids) the start record    OVERTAKEN
--                has moved
--   4      1c    `superseded`                                  SUPERSEDED
--          PRED  (chained calls) the record does not name the  {PREDECESSOR,
--                predecessor (a chain's first write: a record  <record>}
--                exists)
--   5      1d    a wrong-typed fence or intent key, then a     WRONGTYPE_KEY /
--                live snapshot fence (both modes)              {FENCED, <ms>}
--          1e    a wrong-typed commit release record (b.yuo,   WRONGTYPE_KEY
--                both modes)
--   6      2b    (structural) the fence on (seq, commit_chain) STALE <seq>
--   7      3, 4  `seq` / `set_order` ceilings                  SEQ_OVERFLOW /
--                                                              SET_ORDER_OVERFLOW
--   8      4     the pre-pass: input shape, then `TYPE`        ENCODING /
--                probes, then (structural) the digests         WRONGTYPE_KEY /
--                                                              CACHE_DIVERGED
--   9      5     `XADD {seq+1}-0`                              STREAM_ID_COLLISION
--          5a    byte trim `XTRIM MINID ~` (cannot fail)
--   10     6     apply (cannot fail)
--   11     6a/6b external-value fold and settle
--   12     7     header HSET (folds `commit_chain` and
--                `relay_gens`)
--   13     7a    commit-token record HSET (and, when the commit owes
--                external-value releases, the release ring write and
--                its PEXPIRE)
--   14     8     reply
--
-- Every refusal returns before step 5's `XADD`, the first write, so a refused
-- commit writes nothing.
--
-- In fast mode the set_order ceiling is folded into the pre-pass's record walk
-- (step 4), interleaved per record with its input-shape and `TYPE` checks, as it
-- always has been. In structural mode the pre-pass runs three separate walks —
-- parse (input shape and the set_order ceilings), probe, digest — so every
-- ceiling is decided before any probe; within the parse walk a ceiling is decided
-- when its record is parsed, so an earlier malformed record answers `ENCODING`
-- first (its successors cannot be located).
--
-- ── The TOCTOU rule (why seq/next_set_order/version live HERE, not Rust)
--
-- An earlier v1 pre-computed `new_seq`, per-field `version`, and per-field
-- `set_order` in a Rust-side planner fed by an HMGET issued before the
-- EVAL. That shape has a fatal TOCTOU race: two concurrent writers each
-- HMGET `seq=N`, each compute `new_seq=N+1`, each call EVAL with the same
-- `new_seq`, and the second `XADD <new_seq>-0` collides on stream IDs
-- ("ERR The ID specified in XADD is equal or smaller than the target
-- stream top item"). Lua atomicity protects the script body from inter-
-- leaving, but it does NOT freshen a seq that was read outside the atomic
-- window. Fix: every sequence-derived value (new_seq, version per field /
-- time sample, stored_set_order, next_set_order) is computed INSIDE this
-- script from the header read within the same atomic block, using the
-- OODF delta-level `set_order` offsets (which are TOCTOU-free) carried
-- through ARGV. See b.67u test-writer regression notes for the failure
-- mode.
--
-- That rule is fast mode's. Structural mode carries Rust-computed absolute
-- values instead, and is safe from the same race by its fence: a plan made at
-- a `seq` the header no longer holds is refused (`STALE`) before any write, so
-- two plans made at one `seq` cannot both commit.
--
-- ── Contract ──────────────────────────────────────────────────────────────
--
-- Both modes share a fixed KEYS prefix of SHARED_KEYS keys and a fixed ARGV
-- prefix of SHARED_ARGV slots, parsed by the same code; everything
-- mode-specific follows them and is located from those two constants (and
-- their Rust twins `lua::UPDATE_SCRIPT_SHARED_KEYS` /
-- `lua::UPDATE_SCRIPT_SHARED_ARGV`). A key or slot both modes need is added to
-- the prefix — plan b.y3m's snapshot-fence and writer-intent keys joined it as
-- `KEYS[7]` / `KEYS[8]`, the commit release ring (b.yuo) as `KEYS[9]`, and
-- the migration sentinel (b.qtk) as `KEYS[10]`;
-- plan b.v6d's record suffix and start record as `ARGV[9]` / `ARGV[10]` (the
-- slot is now the writer guard, which b.pav's predecessor guard shares), the
-- relay byte budget's four parameters as `ARGV[11..14]`, the call token
-- as `ARGV[15]` and the commit release ring's TTL as `ARGV[16]` — by naming
-- it (step 0) and bumping the length on both
-- sides, and every mode-specific position moves with it. `storage/lua.rs` reads every slot name and both
-- lengths (step 0) out of this file at compile time and refuses to build when
-- they disagree with the Rust twins.
--
-- Shared KEYS (`KEYS[1..SHARED_KEYS]`, step 0's `KEY_*`):
--   [1]      — object header hash    (le:{oid}:obj:hdr)
--   [2]      — fields index sset     (le:{oid}:obj:flds)
--   [3]      — time samples index    (le:{oid}:obj:tss)
--   [4]      — per-object stream     (le:{oid}:obj:stream)
--   [5]      — per-object ext index  (le:{oid}:obj:ext)   (b.qpf)
--   [6]      — commit-token record   (le:{oid}:obj:wtok)
--   [7]      — snapshot fence        (le:{oid}:obj:snapfence)   (SR-5.4)
--   [8]      — writer intents        (le:{oid}:obj:snapintent)  (SR-5.4;
--              read only by its `TYPE` probe, see step 1d)
--   [9]      — commit release ring   (le:{oid}:obj:wrel)  (see "Commit
--              release ring"; its TTL is ARGV[16])
--   [10]     — migration sentinel    (le:{oid}:migration-sentinel) (b.qtk;
--              see "Migration sentinel". Never written, read or named by
--              any command here: step 0 checks its name and nothing else
--              uses it.)
--
-- Fast-mode KEYS after the prefix:
--   [SHARED_KEYS+1 .. SHARED_KEYS+N] — per-mutation key, one per record
--              (field or time-sample key), appended in the same order as the
--              mutation records below.
--
-- Structural-mode KEYS after the prefix (the object-level indexes the records
-- write, once each, then one key per record):
--   [SHARED_KEYS+1]  — sections index set          (le:{oid}:obj:secs)
--   [SHARED_KEYS+2]  — deleted-sections index set  (le:{oid}:obj:dels)
--   [SHARED_KEYS+3]  — children-list index set     (le:{oid}:obj:chs)
--   [SHARED_KEYS+4 .. SHARED_KEYS+3+N] — per-record key, one per record, in
--              record order (see "Structural-mode records").
--
-- In both modes KEYS ends at the last record's key: a call whose KEYS count is
-- not exactly the mode's fixed keys plus N (N = `num_mutations`, ARGV[6]) is
-- refused `ENCODING` by the mode's pre-pass, before any record key is used. A
-- key added or dropped mid-list would otherwise shift every later record onto
-- its neighbour's key, which a record without a digest cannot detect.
--
-- Shared ARGV (`ARGV[1..SHARED_ARGV]`, step 0's `ARG_*`), then the mode slot,
-- then the mode's slots and its variable number of mutation records:
--   [1]  server_ts_us             (string, u64)
--   [2]  client_id                (string)
--   [3]  client_ts                (string, u64)
--   [4]  stream_maxlen            (string, u64; "0" = no MAXLEN trim — the
--                                  entry cap, kept on every XADD whatever
--                                  the byte budget below says)
--   [5]  object_id                (string, u64; appears in the stream payload)
--   [6]  num_mutations            (string, N)
--   [7]  relay_ompf               (binary; the writer's raw inbound OMPF. An
--                                  empty string means "no relay bytes" — the
--                                  XADD then carries only the `payload` field,
--                                  identical to the pre-b.gze entry shape, which
--                                  marks the commit empty: the subscriber sends
--                                  the empty delta at its sequence with no read.)
--   [8]  commit match key         (string; what the commit-token record is
--                                  matched on — the calling call's per-call
--                                  commit token, 32 lowercase hex digits, or
--                                  `op.<operation_id>` when the call carried a
--                                  client operation id. Never empty — see
--                                  "Commit token".)
--   [9]  commit record suffix     (string; appended verbatim to the record
--                                  step 7a writes. `""` without an operation
--                                  id; `:<call_token>:<request_digest>` with
--                                  one.)
--   [10] writer guard             (string; `""` without an operation id. With
--                                  one, a tag and its value — see "Writer
--                                  guard": for an unchained call its start
--                                  record, `"="` when `wtok[client_id]` was
--                                  absent when the call started, else `"="`
--                                  followed by the value it held; for a
--                                  chained call its predecessor, `"^"` for a
--                                  chain's first write, else `"^op.<previous>"`.
--                                  Any other tag is `ENCODING`. Paired with
--                                  ARGV[9]: both empty or both non-empty,
--                                  else `ENCODING`.)
--   [11] relay_budget_bytes       (B: canonical decimal, at most 2^53 - 1; the
--                                  per-object byte budget for retained relay
--                                  bytes. "0" turns it off: `relay_gens` is
--                                  neither read nor written and no byte trim
--                                  is issued. See "Relay byte budget".)
--   [12] relay_generation_bytes   (T = max(1, ceil(B/K)): canonical decimal,
--                                  at least 1 when B > 0)
--   [13] relay_generations        (K: canonical decimal, at least 1 when B > 0)
--   [14] relay_min_entries        (MIN: canonical decimal, at least 1 when
--                                  B > 0; the fewest entries a byte trim
--                                  leaves)
--                                 Slots 11-14 come from Rust
--                                 (`storage::relay_retention::RelayRetention`),
--                                 so the script derives no constant of its
--                                 own; a malformed one is refused `ENCODING`.
--   [15] call token               (string; the calling call's commit token in
--                                  its recorded form — 32 lowercase hex
--                                  digits in production — with or without an
--                                  operation id. The key of the call's entry
--                                  in the commit release ring, written at
--                                  step 7a when the commit owes releases; see
--                                  "Commit release ring". Never empty: an
--                                  empty one is refused `ENCODING` before the
--                                  record is read. Nothing else about it is
--                                  checked.)
--   [16] ring_ttl_ms              (the commit release ring's TTL in ms: 1 to
--                                  12 decimal digits with no leading zero,
--                                  from Rust
--                                  (`storage::keys::commit_release_ring_ttl`,
--                                  the deadline + command timeout + 60 s; the
--                                  script derives no constant of its own).
--                                  Step 7a `PEXPIRE`s the ring with it after
--                                  every ring write. Anything else is refused
--                                  `ENCODING` on every run that reaches step
--                                  1b, whether or not the commit owes
--                                  releases, before the commit-token record
--                                  is read; see "Commit release ring".)
--   [SHARED_ARGV+1]  mode        ("fast" | "structural"; anything else is
--                                  refused `ENCODING` before any guard that
--                                  depends on the mode)
--
-- Fast mode, after the mode slot:
--   [SHARED_ARGV+2 ..] N mutation records, each a variable-length block:
--          tag                    ("SF" | "ST")
--          per-tag arguments (see below)
--
-- Structural mode, after the mode slot:
--   [SHARED_ARGV+2]  expected_seq           (decimal; the header `seq` the
--                                            plan was made at)
--   [SHARED_ARGV+3]  expected_commit_chain  (the header `commit_chain` at
--                                            that `seq`; "" for an object
--                                            whose header has none)
--   [SHARED_ARGV+4]  K                      (decimal; the number of entries
--                                            in the untouched-row count block)
--   [SHARED_ARGV+5 .. SHARED_ARGV+4+2K]     K pairs `hash_hex, n` (see
--                                            "Untouched-row counts")
--   [SHARED_ARGV+5+2K ..] N structural records (see "Structural-mode
--                                            records"), running to the end of
--                                            ARGV.
--
-- Fast-mode record payloads:
--   "SF"  num_value_hfields,
--         hf1_name, hf1_val, ..., hfK_name, hfK_val,    -- value / size /
--                                                        -- hash pairs
--         delta_set_order,                              -- OODF delta
--                                                        -- offset (u64)
--         zadd_member                                   -- fields-index
--                                                        -- sorted-set
--                                                        -- member
--         The script HSETs the field key with the value hash fields PLUS
--         computed `set_order` + `version` fields, then ZADDs the fields
--         index with score = new_seq.
--
--   "ST"  num_value_hfields,
--         hf1_name, hf1_val, ..., hfK_name, hfK_val,
--         zadd_member
--         The script HSETs the time-sample key with the value hash fields
--         PLUS a computed `version` field, then ZADDs the ts index with
--         score = new_seq.
--
-- Field / time-sample deletions never appear in fast mode: they only
-- arise from `apply_reset` or `apply_delete_section` in the Rust planner,
-- neither of which is fast-path (see `lua::is_fast_path`). The fast-path
-- encoder (`lua::encode_delta`) emits only "SF" and "ST" records, and fast
-- mode refuses every other tag. Structural mode's "DF" / "DT" carry them.
--
-- ── Structural-mode records (SR-6.3) ─────────────────────────────────────
--
-- The wire contract `lua::encode_mutations` produces and structural mode
-- consumes. Structural mode accepts only these tags, and fast mode none of
-- them.
--
-- One record per `StorageMutation` of the plan, 1:1 and in the plan's
-- emission order (`apply_delta` emits `SetObjectHeader` last), each with
-- exactly ONE per-record KEYS entry, in record order: the key the record
-- writes, named under each tag below. The object-level keys the records also
-- write (`secs`, `dels`, `chs`, `flds`, `tss`, `obj:ext`) are NOT per-record
-- keys; each travels once among the mode's fixed KEYS. Every id is a decimal
-- string. Every column value is the exact string the WATCH path's
-- `execute_mutation` writes for the same mutation, so the script writes the
-- slots verbatim and does no arithmetic on them; in particular `SFA` / `STA`
-- carry Rust's absolute `set_order` / `authoring_set_order` / `version`.
--
-- A plan carries exactly one "HDR" record; a missing or second one, a KEYS
-- count other than the mode's fixed keys plus N, a record
-- cut short by the end of ARGV, a slot left over after the N-th record, an
-- unknown digest token, or a non-decimal value in a slot the script does
-- arithmetic on or scores an index with, is refused `ENCODING`.
--
-- Untouched-row counts. The block ahead of the records carries, for each
-- external hash in a set the plan's value rows name, the cache's pre-plan
-- count of the value rows the plan does NOT touch that hold it
-- (`storage::structural::PreImageDigests`): `K`, then K pairs of a 64
-- lowercase-hex hash and a decimal count, hashes strictly ascending. A
-- non-decimal `K`, a block of `2K` slots running past the end of ARGV, a hash
-- that is not 64 lowercase hex, a non-decimal count (or one past the f64
-- integer ceiling), or a hash not strictly greater than the one before it (a
-- duplicate included) is refused `ENCODING`. Settle pass 6b reads the counts
-- (see "External-value reference accounting"); nothing else does.
--
-- Pre-image digests. Every digest-checked record ENDS with a digest, the
-- cache's belief about its key BEFORE the plan, which the pre-pass compares
-- with the store before any write; a mismatch answers `CACHE_DIVERGED`. A key
-- the plan touches more than once is checked once, on its first
-- digest-checked record; the later ones carry "none". The digest tokens:
--   "none"     no check for this record;
--   "absent"   the key must be absent;
--   "row"      SEC / DSEC only: six slots follow, the section row's stored
--              parent_id, name, type, order, first_ver, last_ver, and the
--              stored row's six columns must equal them;
--   "exists"   DF / DT only: the row must exist;
--   <40 hex>   CHL / DCHL only: the stored list's digest must equal it.
-- A list's digest is `redis.sha1hex` of its ids joined by "," — the same
-- bytes `lua::children_list_digest` hashes — and a list that is absent has
-- the digest "absent" (an empty list has no stored form). A list carries
-- "none" when its parent was neither the root nor a live section before the
-- plan: the cache deliberately does not hold such a list, so the script
-- neither type-probes nor reads it.
--
-- Type probes. Structural mode `TYPE`-probes exactly the keys the WATCH path's
-- pre-`MULTI` probe names for the same mutations (`layer::mutation_probe_keys`
-- / `collect_probe_keys`, SDD §7.5.4's coverage table) — every record key it
-- `HSET`s, every index it writes, `obj:ext` when any value row is touched, and
-- the header — plus every non-list key the digest pass reads (a digest-checked
-- `DSEC` / `DF` / `DT` key, whose write is a type-agnostic `DEL`). A wrong type
-- answers `WRONGTYPE_KEY`. Children-list keys are never type-refused: the
-- record's `DEL` + `RPUSH` replaces whatever the key held, as on the WATCH
-- path. A digest-checked list key that holds a non-list therefore answers
-- `CACHE_DIVERGED` (the store does not hold what the cache believed), an
-- absent one matches only "absent", and a "none" list is not touched before
-- the apply pass at all.
--
--   "SEC"   KEYS: sec:{sid}       SetSection
--           sid, parent_id, name, type, order, first_ver, last_ver,
--           digest ("none" | "absent" | "row" + 6 slots)
--           Writes the six columns to sec:{sid}; SADD secs sid.
--
--   "DSEC"  KEYS: sec:{sid}       DeleteSection
--           sid,
--           digest ("none" | "absent" | "row" + 6 slots)
--           DEL sec:{sid}; SREM secs sid.
--
--   "TOMB"  KEYS: del:{sid}       CreateTombstone
--           sid, first_ver, last_ver
--           No digest (a TYPE probe only). HSET del:{sid} first_ver last_ver;
--           SADD dels sid.
--
--   "CHL"   KEYS: ch:{p}          SetChildren
--           p, n, child_1, ..., child_n,
--           digest ("none" | "absent" | <40 hex>)
--           DEL ch:{p}; n > 0: RPUSH ch:{p} child_1..child_n and SADD chs p;
--           n == 0: SREM chs p.
--
--   "DCHL"  KEYS: ch:{p}          DeleteChildrenList
--           p,
--           digest ("none" | "absent" | <40 hex>)
--           DEL ch:{p}; SREM chs p.
--
--   "SFA"   KEYS: fld row         SetField
--           num_value_hfields, hf1_name, hf1_val, ..., hfK_name, hfK_val,
--           set_order, authoring_set_order, version, zadd_member
--           No digest. The fast arm's HGET hash (ext pre-image) + HDEL value
--           hash size + HSET of the value pairs plus `set_order`,
--           `authoring_set_order` and `version` as carried; ZADD flds with
--           score = version.
--
--   "STA"   KEYS: ts row          SetTimeSample
--           num_value_hfields, hf1_name, hf1_val, ..., hfK_name, hfK_val,
--           version, zadd_member
--           No digest. As "SFA", writing `version` only; ZADD tss with
--           score = version.
--
--   "DF"    KEYS: fld row         DeleteField
--           zrem_member,
--           digest ("none" | "exists" | "absent")
--           HGET hash (ext pre-image; only when the row is a hash, since a
--           "none" record's row is not probed) → DEL the row; ZREM flds
--           zrem_member.
--
--   "DT"    KEYS: ts row          DeleteTimeSample
--           zrem_member,
--           digest ("none" | "exists" | "absent")
--           As "DF", on tss.
--
--   "HDR"   KEYS: obj:hdr         SetObjectHeader
--           seq, first_ver, next_set_order, has_src_fmt ("1" | "0"),
--           [src_fmt, only when has_src_fmt is "1"]
--           No digest (the fence covers the header). `seq` is the planned
--           sequence number, which must equal expected_seq + 1;
--           `next_set_order` is the plan's post-apply cursor. Written in the
--           header HSET with the timestamps, `last_client_*` and
--           `commit_chain`; `src_fmt` only when present (it may be "").
--
-- Header HSET runs after all mutations; it writes every header field that
-- can change across an `update_layer` call (seq, next_set_order,
-- modified_at, last_client_id, last_client_ts, last_server_ts,
-- commit_chain, and `relay_gens` when the byte budget is on). In fast mode
-- `first_ver` and `src_fmt` are NOT rewritten —
-- fast mode never runs `apply_reset`, so their current values are preserved
-- in place. In structural mode the "HDR" record also carries `first_ver` and,
-- when present, `src_fmt`, written in `execute_mutation`'s header order.
--
-- ── commit_chain: every commit folds it ───────────────────────────────────
--
-- Every commit, an empty one (no mutation records) included, sets
--   commit_chain = redis.sha1hex(prev .. "\n" .. payload)
-- in the step-7 header HSET, where `prev` is the header's `commit_chain` as
-- step 1's HGETALL read it (`""` when the field is absent — an object
-- materialized before the column existed) and `payload` is the exact
-- `update|…` string step 5 XADDs. Byte-identical to the Rust fold
-- (`storage::commit_chain::commit_chain_fold`) the txn backend writes in its
-- `MULTI`. Every refusal returns before step 7, so a refused commit leaves
-- the column, like the rest of the keyspace, untouched.
--
-- ── Relay byte budget: `relay_gens` and the byte trim ─────────────────────
--
-- Every XADD carries the writer's whole OMPF, so retention costs bytes, while
-- `MAXLEN ~ stream_maxlen` bounds only entries. When B (ARGV[11]) is above 0,
-- every commit, an empty one included, also folds the header's `relay_gens`
-- column — at most K+1 `start:bytes` pairs, oldest first, joined by `,`;
-- `bytes` is the summed length of the `ompf` fields of the commits from
-- `start` to the next pair's `start` — and writes the result in the step-7
-- header HSET that advances `seq`. With `s` = #relay_ompf:
--   1. Parse the column as step 1's HGETALL read it. Read it as fresh,
--      `{(new_seq, 0)}`, when it is absent or malformed (anything but
--      `%d+:%d+` pairs joined by `,`, or a number above 2^53 - 1), when any
--      `start` is above `new_seq`, or when the starts are not strictly
--      increasing.
--   2. Add `s` to the newest generation's bytes.
--   3. While more than one generation remains and their bytes sum above B,
--      drop the oldest.
--   4. If the newest generation holds at least T bytes, open
--      `(new_seq + 1, 0)`.
--   5. While more than K+1 generations remain, merge the two oldest.
--   6. Write the result, every number `%d`, in the step-7 HSET.
--   7. If step 3 dropped a generation, `XTRIM stream MINID ~ {cut}-0` right
--      after this commit's XADD succeeded (step 5a), with
--      `cut = min(oldest kept start, max(1, new_seq - MIN + 1))`.
-- Every sum is clamped to 2^53 - 1, so every value stays exact in a Lua
-- number. The fold is pure Lua over the step-1 header and ARGV, computed
-- before the XADD, and is byte-identical to the WATCH path's
-- (`storage::relay_retention::RelayRetention::fold`), which writes the same
-- column in its `MULTI`: the `commit_chain` pattern. Trimming stays head-only
-- and never reaches this commit's own entry, because `cut <= new_seq`.
-- `MAXLEN ~ stream_maxlen` stays on the XADD as the entry cap. With B = 0 the
-- column is neither read nor written and no byte trim is issued.
--
-- ── Commit token: one call commits at most once ───────────────────────────
--
-- `LayerStore::update_layer_with_operation` mints ONE random 128-bit token per
-- call and passes the same match key (ARGV[8]) and record suffix (ARGV[9]) to
-- every attempt of that call, whichever backend runs it. The match key is the
-- token itself, or `op.<operation_id>` when the client sent an operation id. A
-- commit records it in `le:{oid}:obj:wtok` (KEYS[6]): field = `client_id`,
-- value = `<match_key>:<new_seq>:<server_ts><suffix>` — so
-- `<token>:<seq>:<ts>` without an id, and
-- `op.<id>:<seq>:<ts>:<token>:<request_digest>` with one — written by one HSET
-- in step 7a, inside this script's atomic unit, so the record exists exactly
-- when its commit does.
--
-- Every run checks that record (step 1b) BEFORE the superseded guard, the
-- structural fence, the sequence ceiling, the validation pre-pass and any
-- write. If `wtok[client_id]` begins with `<match_key>:`, a commit under this
-- key has already landed and the script answers it (the recorded value, see
-- "Return value") and writes nothing. Without an operation id that commit is
-- an earlier attempt of the same call — its reply was lost (the Rust caller
-- sends this script exactly once per attempt, `max_attempts = 1`, so fred
-- never replays it; a lost reply surfaces as a transient error and the next
-- attempt of the call arrives here instead). With one it may also be an
-- earlier CALL with the same operation id: a client re-send. The script does
-- not tell the two apart; the Rust caller does, from the call token and
-- request digest the record carries (own call → report it; another call with
-- the same digest → `ALREADY_COMMITTED`; another call with a different digest
-- → `ALREADY_EXISTS`). Running the delta again would apply it twice; worse, a
-- re-run of the pre-pass against state that already contains the commit can
-- REFUSE it (`SET_ORDER_OVERFLOW` at the ceiling, or in structural mode
-- `STALE`, since the landed commit moved the `seq` the plan was fenced on) for
-- an edit that landed. A landed commit is reported as landed even if a later
-- CreateLayer overwrite has since set `superseded=1`, matching the slow path's
-- in-doubt reconcile.
--
-- The check (step 1b, with the writer-guard checks below) and the record
-- write (step 7a) are ONE piece of code that both modes run, with no per-mode
-- copy, so the record and its answer are the same whichever mode committed; a
-- change to the match key, the record or the writer guard is made once and
-- holds for both.
--
-- The check reads the record by `client_id`, so it holds while a client keeps
-- at most one write in flight per object and `client_id` — or chains its
-- writes, each guarded by its predecessor (see "Writer guard"). Two concurrent
-- calls sharing one overwrite each other's field; that fails OPEN (each may
-- re-apply), never to a false match, because a random token cannot match
-- another call's commit and an `op.` key only matches the same operation id.
-- Neither key can contain `:`, and an `op.` key can never equal a hex token,
-- so the `<match_key>:` prefix match is exact. The match key is never empty:
-- an empty ARGV[8] would match every record written with an empty key, so it
-- is refused as an encoder bug (`ENCODING`) before the record is read.
--
-- The txn backend writes the identical record in its `MULTI` and runs the
-- identical check on its WATCHed connection, so the record value and the
-- replay answer are the same on both backends (SDD §7.5 parity).
--
-- ── Commit release ring ───────────────────────────────────────────────────
--
-- A commit's owed external-value releases ride out in its reply (step 8) and
-- the Rust caller enqueues them. When that reply is lost they would be lost
-- with it, stranding the released blobs until the hourly reference-count
-- reconciler. So step 7a also records them, in the same atomic unit as the
-- commit, when the commit owes at least one release: an ENTRY
--   `|t|<call_token>` → `"<new_seq> <token> <token> ..."`
-- in `le:{oid}:obj:wrel`, keyed by the call token (ARGV[15]), where each token
-- is a reply release `"<reason>:<hash_hex>"`, space-separated (neither part
-- can contain a space), every release the commit owes. A commit that owes
-- nothing writes nothing here. The txn backend writes the identical entry in
-- its `MULTI`.
--
-- The key is an object-wide ring of RING slots (`COMMIT_RELEASE_RING_SLOTS`,
-- step 0): `|s|<k>` holds the token whose entry starts at slot k, or `""` for
-- a slot a wider entry covers, and `|n|` the next slot. An entry owing R
-- releases takes u = min(RING, ceil(R / M)) slots from `|n|`
-- (`COMMIT_RELEASE_RING_SLOT_RELEASES`); writing it evicts (`HDEL |t|<old>`)
-- every other token whose start slot it takes, then one `HSET` writes the
-- entry, its slots and the advanced `|n|`. So the key holds at most
-- 2·RING + 1 fields, an entry survives until later releasing commits on the
-- object wrap the ring onto its start slot, and an eviction only ever leaks
-- (the reconciler repairs it), never releases. `|` cannot occur in a
-- `client_id`, so these fields never collide with the per-`client_id` field
-- an older build wrote; this script neither reads nor writes that field.
-- `storage::layer::plan_release_ring_write` is the one definition of the
-- rule; step 7a transcribes it.
--
-- Every ring write ends with `PEXPIRE <ring> <ARGV[16]>`, in the same atomic
-- unit (the txn backend queues the same `PEXPIRE` after its ring `HSET`), so
-- the key lives while the layer is being edited and expires ARGV[16] ms after
-- its last releasing commit; a commit that owes nothing neither writes nor
-- refreshes it. The TTL (`storage::keys::commit_release_ring_ttl`, the RPC
-- deadline + the command timeout + 60 s) outlives every reader: an entry is
-- read only by its own call, before that call's deadline. Expiry only ever
-- leaks: an expired ring reads as absent — the lookup enqueues nothing, a
-- missing `|n|` reads as slot 0, a missing slot evicts nothing — and the
-- reference-count reconciler, which never reads the ring, repairs what was
-- owed. A plain `PEXPIRE`, not `PEXPIRE … GT`, which would never set the
-- first TTL on a key without one (an older build's, or one a torn `EXEC`
-- left). An older build's script does not set the TTL: its write into a key
-- with one keeps it, and its write into an absent key leaves it persistent
-- until this script's next ring write on the object.
--
-- The entry is read by one call only, by its token: the call that wrote it.
-- An attempt answered `ALREADY_LANDED` for this call's own commit enqueues
-- it, and so does a call that ends without proof of its outcome — retry
-- exhaustion, or, after a counted retry, `OVERTAKEN`, `PREDECESSOR` or a
-- refusal answered `UNAVAILABLE` outcome unknown — which is how a chained
-- call whose successor committed before its next attempt still recovers its
-- lost reply's releases: that attempt meets the successor's record and is
-- answered `PREDECESSOR`, but its own entry is keyed by its token, which no
-- successor's write replaces.
--
-- Step 1e, right after the fence check (1d), TYPE-probes the key and refuses
-- a non-hash `WRONGTYPE_KEY` before the pre-pass and any write, which is what
-- keeps the step-7a ring write un-rejectable. It sits after the own-key match,
-- the superseded guard, PREDECESSOR and FENCED, and before the `seq` /
-- `set_order` ceilings, the pre-pass and the `XADD`. The WATCH path checks
-- the same key at the same point (its step 2c, after its predecessor check
-- and fence wait and before its snapshot read, its ceilings and its
-- stream-id collision probe) and refuses with this reply, byte for byte. The
-- cached structural route plans in Rust before it sends this script (its
-- read has already answered the own-key match, the superseded guard and
-- PREDECESSOR), and a plan that meets a ceiling — an
-- object at one, or a delta whose own `set_order` would reach it, on any
-- object — sends the call to the WATCH path rather than refusing it there.
-- So on every route a wrong-typed record is refused `WRONGTYPE_KEY` after
-- those first guards and ahead of the ceilings and the stream-id collision.
-- (A delta the structural route's plan cannot apply to the cached state at
-- all, such as a duplicate section, is refused by that plan before the fence
-- check (1d) and this step: the script is never sent.)
--
-- ── Writer guard: no attempt applies out of this client's order ───────────
--
-- The record holds only `client_id`'s LATEST commit on this object. An attempt
-- of call X that executes after the same `client_id`'s next commit Y — a late
-- retry, or an `EVALSHA` sent once that the server runs late — finds `op.Y`,
-- matches nothing, and would apply X again, reordered after Y. With an
-- operation id the caller therefore passes a writer guard as ARGV[10], a tag
-- and its value, which every attempt checks after the own-key match:
--
--   * `=` — an unchained call's START RECORD: the caller reads
--     `wtok[client_id]` ONCE when the call starts and passes `"="` followed
--     by it (`"="` alone when absent). Every attempt refuses itself when the
--     field no longer holds that value and is not this call's own key, in
--     step 1b, before the superseded guard. The answer is `{"OVERTAKEN"}` with
--     nothing written by this run: the Rust caller ends the call
--     `UNAVAILABLE`. On the call's first attempt it hands back the call's
--     pre-flight increments; after a counted retry it hands back nothing and
--     looks up the call's own ring entry instead. (An earlier attempt of the
--     call may have committed; once the field has moved, a re-send cannot be
--     answered from it.)
--   * `^` — a chained call's PREDECESSOR: `"^op.<previous>"`, or `"^"` for a
--     chain's first write. The call reads no start record. Every attempt
--     refuses itself unless the field names that predecessor (`op.<previous>:…`)
--     or, for a chain's first write, is absent — in the PREDECESSOR step,
--     after the superseded guard (1c) and before FENCED (1d). The answer is
--     `{"PREDECESSOR", <record or "">}` with nothing written by this run: the
--     Rust caller refuses the call `ABORTED` `PREDECESSOR_NOT_COMMITTED` on
--     its first attempt, handing back the call's pre-flight increments, or
--     answers it a plain `UNAVAILABLE`, outcome unknown, after a counted
--     retry, handing back nothing and looking up the call's own ring entry
--     instead. So a chained commit lands only directly after its predecessor,
--     however late it runs. A chained call is never answered `OVERTAKEN`.
--
-- Without an operation id ARGV[10] is `""` and nothing is checked. Any other
-- tag, or a guard without ARGV[9]'s operation suffix (or the reverse), is
-- refused `ENCODING` before the record is read.
--
-- ── Step-1b order (canonical across commit modes) ─────────────────────────
--
-- own-key match (`ALREADY_LANDED`) → start record (`OVERTAKEN`) → superseded
-- (`SUPERSEDED`) → predecessor (`PREDECESSOR`) → FENCED → a wrong-typed
-- commit release record (`WRONGTYPE_KEY`) → STALE / CACHE_DIVERGED, in both
-- modes: steps 1b, 1c, PREDECESSOR, 1d, 1e, 2b and the pre-pass's digest walk
-- (the guard table above). The own-key match comes first so a re-send, a
-- conflict, or this call's own landed attempt is answered before anything can
-- refuse it, and a landed commit is reported under a live fence rather than
-- made to wait. A call carries one guard, so at most one of `OVERTAKEN` and
-- `PREDECESSOR` can answer it.
--
-- ── Validate-then-write: why the XADD goes FIRST (b.ph9) ──────────────────
--
-- Lua on Valkey is atomic *in isolation*, not transactional: an error
-- raised by `redis.call` aborts the script mid-body and every write that
-- already ran STAYS COMMITTED. There is no rollback. So the ordering of
-- mutations inside this script is load-bearing after all — not for what
-- other clients can observe mid-script (nothing can), but for what
-- survives when the script dies partway through.
--
-- The structural rule this script obeys is therefore: **no MUTATION runs
-- until every abort condition has been evaluated.** Read-only
-- `redis.call`s (the header `HGETALL`, the `TYPE` probes below) are free
-- to run in the validation phase — a read that fails aborts a script that
-- has written nothing.
--
-- The pre-b.ph9 shape wrote fields/sections, HSET the header, and XADDed
-- last, with no type validation at all. Two families of call could reject
-- after state had already been committed:
--
--   * `XADD <new_seq>-0` — stream-ID monotonicity (the stream's top entry
--     is already at or above `<new_seq>-0`), a WRONGTYPE stream key, or
--     an OOM refusal (see the OOM note at the `pcall` site below).
--   * any per-record `HSET` / index `ZADD` — a WRONGTYPE on a field key,
--     a time-sample key, or either index ZSet. This one is worse than it
--     sounds *after* the reordering: it would abort the apply pass with
--     the stream entry for `{new_seq}-0` already committed, publishing a
--     commit — verbatim, to every live subscriber — that never
--     materialized, with the header left un-advanced so every later write
--     is refused at the XADD forever after.
--
-- The fix is structural rather than a "check the stream first" bolt-on:
--
--  1. A validation pre-pass walks the ARGV mutation records issuing no
--     MUTATING call whatsoever. Every rejectable *input* condition — an
--     unknown tag, a malformed record, the `set_order` fold crossing the
--     2^53 ceiling — is decided there from ARGV alone, and every
--     rejectable *keyspace-shape* condition — a wrong-typed per-record
--     key or index ZSet — is decided there from read-only `TYPE` probes,
--     before anything is written. (Pre-b.ph9 the input errors were raised
--     mid-loop and half-committed the records that preceded them;
--     SET_ORDER_OVERFLOW in particular left field versions above the
--     header seq, corrupting the version-filtered-diff invariant.) The
--     pre-pass also does the `set_order` arithmetic ONCE and stashes the
--     per-record results, so the apply pass consumes them rather than
--     recomputing — the two walks cannot drift.
--  2. The XADD is then issued FIRST among the mutations, through
--     `redis.pcall` so a rejection returns an error table instead of
--     aborting the script. Because it is the first write, the early return
--     leaves the keyspace bit-for-bit untouched: the commit fails whole.
--  3. Only then does the apply pass (HSET/ZADD per record) run, followed
--     by the header HSET. Both are now proven un-rejectable: the types
--     were probed in step 1 and cannot have changed (nothing else runs
--     inside the atomic window), and the OOM gate — which fires only on a
--     script's FIRST deny-oom write — was already spent on the XADD.
--
-- Structural mode holds to the same rule. Its pre-pass adds the fence
-- (`STALE`) and the pre-image digests (`CACHE_DIVERGED`), both decided from
-- read-only calls (`HMGET`, `EXISTS`, `TYPE`, `LRANGE`), and its apply pass
-- writes only keys the pre-pass probed or `DEL`s first: the section and
-- tombstone rows and every index were probed, a children list is `DEL`'d
-- before its `RPUSH`, and a deleted row is read only after a `TYPE` says it is
-- a hash. So it, too, cannot reject after the `XADD`.
--
-- The reordering is externally invisible. Lua runs atomically, so no
-- client can observe the intermediate state where the stream entry exists
-- but the fields do not; subscribers blocked in `XREAD` are woken only
-- after the script completes.
--
-- ── Migration sentinel: no in-script cluster refusal (b.qtk) ──────────────
--
-- The rule above covers every refusal a command can meet on its own key
-- (WRONGTYPE, OOM). One more kind exists on a cluster: during a legacy slot
-- migration (`CLUSTER SETSLOT … MIGRATING` plus key-by-key `MIGRATE`), every
-- in-script command is checked against the slot's state again, and on the
-- migrating source one that names a key absent at that moment is refused
-- ("non local key" / "undeclared keys … rehashing"). The script then aborts
-- with its earlier writes kept. This run can empty a key and then name it:
-- the settle's `HDEL obj:ext <old>` can empty `obj:ext` before the next
-- `HGET obj:ext <new>` (both modes), and a children-list rebuild `DEL`s the
-- list before its `RPUSH` (structural mode). Either would land after the
-- `XADD`, leaving a published entry, a header that never advanced, and every
-- later commit refused `STREAM_ID_COLLISION`.
--
-- Rather than reorder those writes, the script declares one more key, its
-- object's migration sentinel (`KEYS[10]`), which is never written. The
-- store's pre-execution check refuses a multi-key command naming an absent
-- key in a migrating or importing slot (`-TRYAGAIN`, or `-ASK`, which the
-- service never follows for a script), so while the object's slot is
-- moving the script never starts; on a stable slot the sentinel is never
-- looked up. Step 0 checks that the slot holds the sentinel of `KEYS[1]`'s
-- own hash tag, so a caller that drops it is refused `ENCODING` everywhere,
-- standalone stores included. No command here names it.
--
-- ── External-value reference accounting (b.qpf) ───────────────────────────
--
-- `le:ext:meta:{hash_hex}`'s `ref_count` counts LIVE OBJECTS referencing the
-- hash — one per object per distinct hash — and the caller has already paid a
-- pre-flight `+1` for every distinct external hash this delta references
-- (`store_external_values`, before the EVAL). This script owns the other half:
-- deciding, for each hash, whether that `+1` established a NEW reference or was
-- surplus, and whether a row it overwrote gave up the object's LAST reference
-- to some other hash.
--
-- It can decide both locally because `le:{oid}:obj:ext` (KEYS[5]) records, per
-- hash, how many of this object's value rows currently hold it. The apply pass
-- reads each row's OLD hash immediately before the b.zvd `HDEL` that
-- clears it, folds `-1` for the old hash and `+1` for the new one into a
-- per-hash delta table, and then rewrites exactly the touched fields of that
-- index. A hash whose count reaches zero is `HDEL`ed and its reference is owed
-- back ("rewrite"); a hash this commit references that the object ALREADY held
-- means the pre-flight `+1` bought nothing and is owed back
-- ("surplus_acquire"). The two are mutually exclusive per hash: a referenced
-- hash ends at count >= 1.
--
-- A row holds hash h iff it has NO `value` column and its `hash` column is h.
-- That is the decoder's rule (`storage::diff_read::decode_field_value` reads
-- `value` first), and every per-object reader and writer of the index counts
-- by it — the materializers, both commit backends, retirement and the
-- per-object recount (`ext_index_recount.lua`). So a legacy dual-shape row
-- (`value` plus a stale `hash`, written before b.zvd) holds no hash, and its
-- stale `hash` column is never released as the row's OLD hash.
--
-- Structural mode runs the same recording and fold, and the same settle
-- release arms, generalised to deletes (its settle's count rule differs; see
-- the next paragraph): an "SFA" / "STA" records its row like "SF" / "ST", and
-- a "DF" / "DT" reads its row's `hash` before the `DEL` and records the row as
-- ending with no hash. A hash a delete drops to zero is therefore owed back as
-- "rewrite" — the reason the WATCH path's `plan_ext_index` gives it — and no
-- reason outside `EXT_REF_RELEASE_REASONS` is ever emitted (the Rust decoder
-- drops an unknown one as undecodable, leaking the reference).
--
-- Structural mode settles a touched hash's count differently from fast mode,
-- and only the count. Fast mode writes `after = before + delta` when
-- `delta ~= 0`, trusting the stored count. Structural mode writes the
-- raise-only floor
--   after = max(before + delta, untouched[h] + ends[h])
-- whenever the stored field is not exactly that value (`HDEL` at 0), where
-- `untouched[h]` is the cache's count of the rows the plan does not touch
-- (the block ahead of the records, 0 when the block carries no entry for
-- `h`) and `ends[h]` the touched rows whose FINAL write holds `h` (settle
-- pass 6a, from this script's own records). So an index that drifted below
-- the rows is repaired upward on any structural commit that touches the hash,
-- and a commit that would drive such a count to 0 while a row still holds the
-- hash neither `HDEL`s it nor emits "rewrite". The floor can only raise:
-- untouched rows carry no pre-image digest, so the cache's recount is never
-- allowed to lower a count or cause a release — a cache wrong about one
-- untouched row leaks a reference at worst. An index above the rows is kept
-- at `before + delta`, a leak until the object retires. The release arms are
-- the same code in both modes, over each mode's `after`, and still decide
-- from store-side evidence alone: `before`, `ext_held`, and an `after` of 0,
-- which in structural mode also requires `before + delta <= 0`. Policy:
-- `KeySchema::obj_ext_index`'s `# Drift`.
--
-- "A hash this commit references" means, precisely, one that some row this
-- commit writes ENDS holding — the folded final-write set of settle pass 6a
-- below, which is a SUBSET of the pre-flight `+1` set, not equal to it. Subset
-- is the sound direction: every `surplus_acquire` still has a paid `+1` behind
-- it, and the pre-stored hashes that fold out only leak (see the deliberate
-- imprecision below). Widening it back to the pre-flight set would hand back an
-- increment for a hash the object may never have held — over-release.
--
-- ── Per row: ORIGINAL pre-image vs FINAL write (b.yda) ────────────────────
--
-- More precisely than the paragraph above (which predates b.yda) states it: the
-- apply pass RECORDS, per value row, two hashes — the row's ORIGINAL pre-image,
-- captured on the row's FIRST touch and never overwritten, and the FINAL hash
-- the commit leaves the row holding. Nothing is folded during the apply pass.
-- Settle pass 6a then folds ONE `-original +final` step per touched
-- ROW, and settle pass 6b turns the totals into index writes and releases.
--
-- **That is the same algorithm, expressed the same way, as the txn backend's
-- `storage::layer::fold_ext_row_transitions`**, which collapses duplicate row
-- keys to the last write and folds each row's pre-image against it. The two
-- backends must agree by construction, not by testing, because they share one
-- `ref_count` and disagreement in the "already held" direction deletes a blob
-- under a live row.
--
-- Folding per RECORD instead was the b.yda round-3 defect. A delta may carry
-- two records for the same `(section_id, key_name)` — the encoder does not
-- dedup, `is_fast_path` admits it and `validate_delta_bounds` does not reject
-- it — and the second record's `HGET` then reads what the FIRST record of this
-- same commit just wrote. Per-record, that reads as "the row already held this
-- hash" for a hash the object never held before the commit, and hands back a
-- `+1` that bought a real, new reference: over-release, the loss direction.
--
-- "Already held" has TWO independent proofs, and either suffices:
--   * `before > 0` — the index says so. The general case.
--   * `ext_held[h]` — some row's ORIGINAL pre-image AND its FINAL write are
--     both `h`. Row-local, and the only proof available on an object
--     materialized before b.qpf, which carries no index at all: there `before`
--     reads 0 for a hash the object genuinely holds, and the delta of a
--     same-hash rewrite nets to zero so the index is never seeded either. That
--     shape leaked its pre-flight `+1` on every commit for the object's whole
--     life. The `held` proof is deliberately NOT a count clamp — it says the
--     object held the hash, never how many rows do — so it hands back exactly
--     the one increment this commit paid and cannot over-release.
--
-- One imprecision is deliberate and shared with the txn backend. A hash written
-- only as an INTERMEDIATE state of a doubly-written row (`A → B → C`: the
-- caller pre-flight-incremented `B`, but no row ends holding it) is folded out
-- of existence here, so its `+1` is neither matched by a reference nor handed
-- back — a leak, drained by the reference-count reconciler
-- (`storage::ext_reconcile`). Releasing it would need this script to distinguish
-- "the caller paid for it" from "some other object holds it", which it cannot
-- do from inside the `{object_id}` slot. Leak direction over loss direction,
-- the same trade the retire-retry settle gate takes.
--
-- **Absolute `HSET`/`HDEL`, never `HINCRBY`.** `HINCRBY` errors on a field
-- whose stored value is not an integer, and this bookkeeping runs AFTER the
-- XADD — inside the region the b.ph9 proof says cannot reject. An `HGET` +
-- `HSET` pair is equivalent under Lua atomicity (nothing interleaves) and
-- cannot fail on a key the pre-pass proved is a hash.
--
-- The owed decrements are NOT enqueued here: `le:ext:{gc}:pending` lives on the
-- `{gc}` slot and `ext_meta` on `{hash_hex}`, so neither is reachable from this
-- object's atomic unit (CROSSSLOT). They ride out in the return value and the
-- Rust caller SADDs them immediately after.
--
-- ── Return value ──────────────────────────────────────────────────────────
--
-- The same reply shapes in both modes, plus three refusals only structural
-- mode can return (see the end of this section).
--
-- On success: a flat array of bulk strings.
--   [1]   the new sequence number, decimal (`string.format("%d", ...)`).
--   [2]   structural mode only: `"drift:<under>:<over>"`, the number of
--         touched hashes whose settle found `before + delta` below / above
--         the recount (step 6b), each a decimal.
--   then  zero or more owed external-value releases, each
--         `"<reason>:<hash_hex>"` with reason `rewrite` | `surplus_acquire`.
-- Deliberately an all-string flat array rather than a mixed
-- `{integer, {array}}`: it decodes as `Vec<String>` under one unambiguous fred
-- conversion, with no nested-array or tuple-arity assumptions.
-- When the start record no longer matches (step 1b): `{"OVERTAKEN"}`, and
--   nothing is written.
-- When a chained call's predecessor is not the record (the PREDECESSOR step):
--   a two-element flat array `{"PREDECESSOR", "<record>"}` — the tag, then
--   the commit-token record verbatim, `""` when absent — and nothing is
--   written. Element 1 never follows a decimal element 0, so it cannot be
--   read as a commit; the Rust caller reads it as the refusal's
--   `last_record`.
-- When a commit under this call's match key already landed (step 1b): a
--   two-element flat array `{"ALREADY_LANDED", "<record>"}` — the tag, then
--   the commit-token record verbatim. Element 1 is deliberately not a decimal,
--   so it can never be read as a fresh commit's sequence; the Rust caller
--   parses the record with the same parser the txn backend uses for its own
--   read, so both backends reach the identical verdict and report the
--   identical `(seq, server_ts)`. No commit releases ride on it: the releases
--   the landed commit owed were in its own lost reply, or were enqueued by the
--   earlier call that made it. For the first case the landed commit recorded
--   them in its entry of the commit release ring `le:{oid}:obj:wrel` (KEYS[9])
--   in its own atomic unit (step 7a), and the Rust caller, on an own-call
--   verdict, reads the entry under this call's token and enqueues them when
--   its sequence matches the landed commit's — see "Commit release ring",
--   which also says how a call that never sees this reply (a chained call
--   whose successor committed first) still recovers them. Nothing is
--   written. (When the record belongs to another call, the Rust caller hands
--   back THIS call's pre-flight increments itself and never reads the ring.)
-- When a fenced snapshot reader holds the object (step 1d), in either mode: a
--   two-element flat array `{"FENCED", "<retry_after_ms>"}` — the tag, then the
--   latest live reader deadline minus this node's `TIME`, whole ms, at least
--   1. A success reply, not an error: the caller waits and re-runs the script,
--   and the wait is not a failure. Element 1 never follows a decimal element
--   0, so it cannot be read as a commit. Nothing is written. Decoded by one
--   Rust decoder for both modes (`layer::decode_fenced_reply`).
-- On a missing header: `redis.error_reply("NOT_FOUND: object header missing")`.
--   The Rust caller maps this to `ServiceError::NotFound` — same mapping as
--   the legacy path's empty-HGETALL branch (b.f1v), which is what routes the
--   gRPC handler into its single checkpoint-recovery arm.
-- On a header that EXISTS but whose `seq` is missing / non-numeric:
--   `redis.error_reply("PARTIAL_HEADER: object header partial (seq missing or
--   malformed)")`. Deliberately a distinct TAG rather than a second `NOT_FOUND`
--   prose variant (b.f1v): the Rust caller maps it to
--   `ServiceError::FailedPrecondition`, matching the legacy path's `seq_ok`
--   guard. It must not be routed into checkpoint recovery, because a
--   client-visible precondition failure on the COMMIT path must not be silently
--   converted into a restore — the caller is told its write did not apply, not
--   that its layer was lost. The recovery arm is keyed on the `NotFound`
--   VARIANT, so `FailedPrecondition` structurally cannot reach it. (Since b.f4t
--   a partial header IS restorable when recovery is reached through
--   `NotFound`: the presence predicate is a structural `HGETALL` on `seq` +
--   `created_by`, not the bare `EXISTS` it once was.)
-- On a stream write rejected *because of this stream key*:
--   `redis.error_reply("STREAM_ID_COLLISION: <underlying error>")`. The Rust
--   caller maps this to a non-retriable `ServiceError::Internal` prefixed
--   `update_layer stream-id collision:` (b.ph9). No writes were performed.
--   The tag is applied from an ALLOWLIST of exactly two XADD rejections —
--   the ID-monotonicity refusal (matched on the fragment `equal or smaller`)
--   and `WRONGTYPE` — never as a catch-all. A WRONGTYPE on the stream key is
--   deliberately folded into the tag: the operator-facing condition ("this
--   layer's stream key is unusable, writes fail closed until it is cleared")
--   and the remediation are identical, and the slow path's probe classifies
--   it the same way (backend parity).
--   The tag is a *verdict about the stream*, not a final client status: the
--   Rust caller re-reads the header and layer mapping first
--   (`LayerStore::disambiguate_stream_collision`), because a layer being
--   deleted or overwritten underneath an in-flight commit appends a
--   wall-clock-ID `deleted` tombstone that refuses this XADD for reasons that
--   have nothing to do with a corrupt stream.
-- On a wrong-typed per-record key or index ZSet, a wrong-typed commit-token
--   record (step 1b), a wrong-typed snapshot fence or writer-intent key (step
--   1d), or a wrong-typed commit release record (step 1e; the latter four
--   ahead of the pre-pass, in both modes):
--   `redis.error_reply("WRONGTYPE_KEY: <key> holds a <type>, expected
--   <type>")`, raised before any write. The Rust caller maps
--   it to a non-retriable `ServiceError::Internal`. Kept OUT of the
--   `STREAM_ID_COLLISION` family on purpose: nothing is wrong with the
--   stream, so the collision counter must not tick and the "clear the stream
--   key" remediation would be actively misleading.
-- On any OTHER refusal of the XADD — `-OOM`, `-READONLY`, `-MISCONF`,
--   `-NOREPLICAS`, stale-replica, `-NOPERM`, `-MOVED` / `-CROSSSLOT`, or a
--   gate a future server version adds: the server's own reply, verbatim,
--   untagged, so the Rust caller classifies it as itself — the four store
--   write refusals (`-OOM`, `-MISCONF`, `-READONLY`, `-NOREPLICAS`) as a
--   nothing-applied `UNAVAILABLE` (`storage::client::store_write_refusal`),
--   anything else through the ordinary transient/permanent mapping. No
--   writes were performed here either — this XADD is the script's first
--   write. See the allowlist note at the `pcall` site.
--
-- Structural mode only — each an error reply, raised before any write, and
-- each answered to `commit_structural`'s caller as a `CommitOutcome` rather
-- than a `ServiceError` (SR-6.4); every other reply above maps exactly as in
-- fast mode, through the same Rust mapping (the `FENCED` table included,
-- which `commit_structural` answers as `CommitOutcome::Fenced`):
--   `STALE <seq>` — the fence missed (step 2b): the header's `seq` is not
--     `expected_seq`, or its `commit_chain` (`""` when absent) is not
--     `expected_commit_chain`. `<seq>` is the header's `seq` as stored.
--   `CACHE_DIVERGED: <record and key>` — a pre-image digest mismatched (step
--     4): the store does not hold what the plan's cache believed.
--
-- ── Invariants ────────────────────────────────────────────────────────────
--
--  1. Atomic: no other Valkey command runs between the header HGETALL and
--     step 7a's last write (the commit-token HSET, or the commit release ring
--     write and its PEXPIRE after it). No WATCH / retry needed.
--  1a. At most once per call, and per operation id: a run whose match key is
--     already recorded for its `client_id` writes nothing (step 1b), so a
--     retried attempt of one call — or a client re-send carrying the same
--     operation id — can never commit a second time. With an operation id,
--     a run whose writer guard fails writes nothing either: an unchained
--     run whose start record no longer matches (`OVERTAKEN`), so no attempt
--     applies after that `client_id`'s next commit on this object, and a
--     chained run whose predecessor is not the record (`PREDECESSOR`), so a
--     chained commit lands only directly after its predecessor.
--  1b. Never under a live fence: a run that finds a live `r:*` lease on the
--     snapshot fence writes nothing (step 1d), in either mode. Because the
--     script is atomic, every commit before a fence grant is in the reader's
--     cut and none lands after it until the lease is released or expires
--     (Spec SDD SR-5.2).
--  2. Byte-for-byte parity with `execute_mutation` + legacy XADD: the
--     fast-path HSET field set matches what the legacy path would write
--     given the same current header and the same delta. The legacy path
--     produces `set_order`/`version` via `apply_delta`; this script
--     produces them via the same arithmetic applied to the header it
--     reads inside its atomic window. Structural mode writes, per record,
--     the commands `execute_mutation` issues for the same mutation, with the
--     same values (carried verbatim from `apply_delta`'s plan).
--  3. The stream entry id is `"<new_seq>-0"` and its `payload` field is
--     `"update|<object_id>|<new_seq>|<client_id>|<client_ts>|<server_ts>"`,
--     matching the legacy XADD exactly (SDD §7.5, §12.2) — the string
--     `storage::commit_chain::update_stream_payload` builds, and the bytes the
--     `commit_chain` fold hashes on both backends. When `relay_ompf`
--     is non-empty a second binary `ompf` field carries the writer's raw
--     inbound OMPF (b.gze), matching the slow path's two-field XADD.
--  4. All-or-nothing (b.ph9): every error this script can return is raised
--     before any MUTATION. The validation phase may issue read-only calls
--     (the header `HGETALL`, the commit-token `TYPE` + `HGET`, the fence and
--     intent `TYPE`s, the fence `HGETALL` and `TIME`, the commit release
--     record's `TYPE`, the per-key `TYPE` probes, structural mode's digest
--     reads) — a failure there aborts
--     a script that has written nothing — but the first mutating
--     call, the `pcall`ed `XADD`, runs only once every abort condition has
--     been evaluated, and nothing after it can reject (types were probed —
--     the commit release ring's included, so its step-7a `HGET` / `HMGET` /
--     `HDEL` / `HSET` cannot fail, nor its `PEXPIRE`, which is not deny-oom
--     and takes an `ENCODING`-checked TTL; the OOM gate fires only on a
--     script's first deny-oom write, which the XADD already absorbed). The
--     one refusal no probe can rule out, a cluster node's in-script refusal
--     during a slot migration, cannot happen either: the migration sentinel
--     keeps the script from starting while its slot is moving (see
--     "Migration sentinel"). A returned error therefore guarantees the
--     object's keyspace is exactly as it was when the script started.

-- ── 0. Layout constants. ─────────────────────────────────────────────────
-- The shared KEYS / ARGV prefix (see "Contract"): every slot of it is read
-- through one of these names, never a bare index, and every mode-specific
-- position below is computed from the two lengths, so growing the shared
-- prefix is a change here plus its Rust twins (`lua::UPDATE_SCRIPT_KEY_*`,
-- `lua::UPDATE_SCRIPT_ARG_*`, `lua::UPDATE_SCRIPT_SHARED_KEYS` /
-- `UPDATE_SCRIPT_SHARED_ARGV`). `storage/lua.rs` reads every
-- `local KEY_<X> = <n>` / `local ARG_<X> = <n>` and both lengths out of this
-- file at compile time and refuses to build on any disagreement, or on a
-- name either side has and the other lacks.
local KEY_HEADER = 1
local KEY_FIELDS_INDEX = 2
local KEY_TS_INDEX = 3
local KEY_STREAM = 4
local KEY_EXT_INDEX = 5
local KEY_COMMIT_TOKEN = 6
local KEY_FENCE = 7
local KEY_INTENT = 8
local KEY_COMMIT_RELEASE = 9
local KEY_MIGRATION_SENTINEL = 10
local SHARED_KEYS = 10

local ARG_SERVER_TS = 1
local ARG_CLIENT_ID = 2
local ARG_CLIENT_TS = 3
local ARG_STREAM_MAXLEN = 4
local ARG_OBJECT_ID = 5
local ARG_NUM_MUTATIONS = 6
local ARG_RELAY_OMPF = 7
local ARG_COMMIT_KEY = 8
local ARG_RECORD_SUFFIX = 9
local ARG_WRITER_GUARD = 10
local ARG_RELAY_BUDGET_BYTES = 11
local ARG_RELAY_GENERATION_BYTES = 12
local ARG_RELAY_GENERATIONS = 13
local ARG_RELAY_MIN_ENTRIES = 14
local ARG_CALL_TOKEN = 15
local ARG_RING_TTL_MS = 16
local SHARED_ARGV = 16

-- The commit release ring's shape (see "Commit release ring"): RING slots,
-- and the releases one slot accounts for, M. The Rust twins are
-- `storage::layer::COMMIT_RELEASE_RING_SLOTS` /
-- `COMMIT_RELEASE_RING_SLOT_RELEASES`, which carry the sizing rationale;
-- `storage/lua.rs` checks both at compile time. RING is at most 1000, so the
-- `|n|` field is at most three digits.
local COMMIT_RELEASE_RING_SLOTS = 256
local COMMIT_RELEASE_RING_SLOT_RELEASES = 8

-- The mode slot's two values (`lua::UpdateScriptMode`).
local MODE_FAST       = "fast"
local MODE_STRUCTURAL = "structural"

-- The largest integer a Lua 5.1 number (an IEEE 754 f64) holds exactly:
-- 2^53 - 1. Every sequence / set_order ceiling in this script, in both modes,
-- is this one value — and `apply_delta` enforces the same one on the WATCH
-- path (`delta.rs`), so the backends refuse at the same boundary.
local F64_INT_CEILING = 9007199254740991  -- 2^53 - 1

-- The migration sentinel check (see "Migration sentinel"), before any read or
-- write: `KEYS[KEY_MIGRATION_SENTINEL]` must be `<prefix>{<tag>}:migration-sentinel`
-- for `KEYS[1]`'s own hash tag (the text between the first `{` and the next
-- `}`, as the store computes it). Declaring it is what keeps the script from
-- starting during a slot migration, so a call without it is a caller bug and
-- is refused here on every store. Nothing below names the slot.
do
  local function hash_tag(key)
    local open = string.find(key, "{", 1, true)
    if not open then return nil end
    local close = string.find(key, "}", open + 1, true)
    if not close or close == open + 1 then return nil end
    return string.sub(key, open + 1, close - 1)
  end
  local sentinel = KEYS[KEY_MIGRATION_SENTINEL]
  local suffix = ":migration-sentinel"
  if type(sentinel) ~= "string"
      or string.sub(sentinel, -string.len(suffix)) ~= suffix
      or hash_tag(sentinel) == nil
      or hash_tag(sentinel) ~= hash_tag(KEYS[KEY_HEADER]) then
    return redis.error_reply(
      "ENCODING: KEYS[" .. KEY_MIGRATION_SENTINEL .. "] is not the migration sentinel of " ..
      "KEYS[" .. KEY_HEADER .. "]'s hash tag")
  end
end

-- ── 1. Load current header. HGETALL on the single object header hash is
--       cheap (≈ 10 small fields) and is required anyway for seq /
--       next_set_order. Reads the same fields `header_to_object_state`
--       would pull in the legacy path's fast-path branch. ────────────────
local header_key = KEYS[KEY_HEADER]
local header = redis.call("HGETALL", header_key)
if #header == 0 then
  return redis.error_reply("NOT_FOUND: object header missing")
end

-- Flatten Valkey's array-of-alternating-KV reply into a lookup table.
local hdr = {}
for i = 1, #header, 2 do
  hdr[header[i]] = header[i + 1]
end

-- ── 1a. Partial-header guard (b.k1r medium). ─────────────────────────────
-- The HGETALL-empty check above catches a completely-deleted header but
-- not a header that exists with `seq` somehow missing / non-numeric —
-- which would let `tonumber(hdr["seq"]) or 0` fall through to
-- `current_seq = 0` and `new_seq = 1`, colliding with the initial `1-0`
-- stream entry that CreateLayer emitted. Treat a missing / malformed
-- `seq` the same as a missing header for ADMISSION purposes — refuse the
-- commit — but report it under its own tag (b.f1v). It is NOT the same
-- CONDITION: the object is present and corrupt, not absent, so the Rust
-- caller returns `FailedPrecondition` (the legacy path's `seq_ok` shape)
-- rather than the `NotFound` that routes the gRPC handler into checkpoint
-- recovery. Converting a client-visible precondition failure on the commit
-- path into a restore would replace the live object with checkpoint state on
-- an RPC that never asked for one; the recovery arm matches the VARIANT, so
-- `FailedPrecondition` structurally cannot land there. Non-retriable either
-- way — no retry on the fast path can make a bad header good.
if hdr["seq"] == nil or tonumber(hdr["seq"]) == nil then
  return redis.error_reply("PARTIAL_HEADER: object header partial (seq missing or malformed)")
end

-- ── 1b. Commit-token check: has THIS call (or operation) already committed,
--       or has this client's next commit overtaken it? ─────────────────────
-- See "Commit token" and "Writer guard" in the contract above. Placed after
-- the two guards that say the object is not there to have committed to, and
-- before every guard or validation that a re-run against already-committed
-- state could trip — the superseded flag, the sequence ceiling, the pre-pass —
-- and before any write. Read-only, so returning from here leaves the keyspace
-- untouched. Order, per the canonical step-1b order in the contract: own-key
-- match, then the start record (the `=` guard), then (1c) the superseded
-- guard, then the predecessor (the `^` guard, the PREDECESSOR step), then
-- (1d) FENCED, then (1e) the commit release record's type; STALE /
-- CACHE_DIVERGED (2b and the pre-pass, structural mode) come after that. One
-- piece of code for both modes.
--
-- The record key is TYPE-probed first: `HGET` on a non-hash raises, which
-- would abort the script with the server's untagged WRONGTYPE instead of the
-- `WRONGTYPE_KEY` tag both backends refuse a wrong-typed commit key with.
-- `TYPE` never errors (see `wrong_type` below for the reply shape).
local wtok_key      = KEYS[KEY_COMMIT_TOKEN]
local commit_key    = ARGV[ARG_COMMIT_KEY]
local record_suffix = ARGV[ARG_RECORD_SUFFIX]
local writer_guard  = ARGV[ARG_WRITER_GUARD]
if commit_key == nil or commit_key == "" then
  return redis.error_reply(
    "ENCODING: missing commit match key (ARGV[" .. ARG_COMMIT_KEY .. "])")
end
if record_suffix == nil then
  return redis.error_reply(
    "ENCODING: missing commit record suffix (ARGV[" .. ARG_RECORD_SUFFIX .. "])")
end
if writer_guard == nil then
  return redis.error_reply(
    "ENCODING: missing writer guard (ARGV[" .. ARG_WRITER_GUARD .. "])")
end
-- The record suffix and the writer guard are both set by an operation id and
-- both empty without one; a mismatched pair is an encoder bug, and guessing
-- which half is right would either skip the writer guard or apply it to a
-- call it cannot cover.
if record_suffix ~= "" and writer_guard == "" then
  return redis.error_reply(
    "ENCODING: operation record suffix (ARGV[" .. ARG_RECORD_SUFFIX ..
    "]) without a writer guard (ARGV[" .. ARG_WRITER_GUARD .. "])")
end
if record_suffix == "" and writer_guard ~= "" then
  return redis.error_reply(
    "ENCODING: writer guard (ARGV[" .. ARG_WRITER_GUARD ..
    "]) without an operation record suffix (ARGV[" .. ARG_RECORD_SUFFIX .. "])")
end
-- The guard's tag (see "Writer guard"): `=` an unchained call's start record,
-- `^` a chained call's predecessor. Any other first byte is an encoder bug;
-- running the attempt unguarded or under a guessed guard could reorder it.
local guard_tag   = string.sub(writer_guard, 1, 1)
local guard_value = string.sub(writer_guard, 2)
if writer_guard ~= "" and guard_tag ~= "=" and guard_tag ~= "^" then
  return redis.error_reply(
    "ENCODING: unknown writer guard tag (ARGV[" .. ARG_WRITER_GUARD .. "])")
end
-- The call token keys this call's entry in the commit release ring (step 7a,
-- "Commit release ring"). An empty one would key every token-less call's
-- entry alike, and the ring's reader could then enqueue another call's
-- releases, so it is refused as an encoder bug before any record is read.
-- Nothing else about it is checked: its recorded form is the caller's.
local call_token = ARGV[ARG_CALL_TOKEN]
if call_token == nil or call_token == "" then
  return redis.error_reply(
    "ENCODING: missing call token (ARGV[" .. ARG_CALL_TOKEN .. "])")
end
-- The commit release ring's TTL (step 7a, "Commit release ring"): 1 to 12
-- decimal digits with no leading zero, so the step-7a `PEXPIRE` after the
-- `XADD` can never be refused for its argument (a positive integer far below
-- the overflow bound) and never expires the ring at once. Checked on every
-- run that reaches step 1b (NOT_FOUND and PARTIAL_HEADER answer first),
-- whether or not the commit owes releases: the verdict does not depend on
-- the delta. Kept as the string the caller sent; the server parses
-- it.
local ring_ttl = ARGV[ARG_RING_TTL_MS]
if ring_ttl == nil or #ring_ttl > 12 or string.find(ring_ttl, "^[1-9]%d*$") == nil then
  return redis.error_reply(
    "ENCODING: bad commit release ring TTL (ARGV[" .. ARG_RING_TTL_MS .. "])")
end
local wtok_type = redis.call("TYPE", wtok_key)["ok"]
if wtok_type ~= "hash" and wtok_type ~= "none" then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. wtok_key .. " holds a " .. wtok_type ..
    ", expected hash (commit-token record)")
end
-- `false` when the key or the field is absent.
local recorded = false
if wtok_type == "hash" then
  recorded = redis.call("HGET", wtok_key, ARGV[ARG_CLIENT_ID])
  local prefix = commit_key .. ":"
  if recorded and string.sub(recorded, 1, #prefix) == prefix then
    return { "ALREADY_LANDED", recorded }
  end
end
-- Start record (the `=` tag: unchained operation ids only): the field must
-- still hold what it held when the call started (`""` standing for absent on
-- both sides). Anything else is another commit of this `client_id` since the
-- call started (this attempt would apply after it) or a checkpoint restore
-- that wiped the record. A chained call (the `^` tag) is never overtaken: its
-- guard is the PREDECESSOR step below, after the superseded guard.
if guard_tag == "=" then
  if guard_value ~= (recorded or "") then
    return { "OVERTAKEN" }
  end
end

-- ── 1c. Superseded-flag guard (b.qw6 item 1). ────────────────────────────
-- CreateLayer overwrite atomically HSETs `superseded=1` on the OLD
-- object's header BEFORE flipping the session→layer mapping. An
-- UpdateLayer that resolved the layer mapping just before the flip
-- arrives here against the old object_id; if `superseded=1` is set, the
-- write would commit to a doomed object whose subscribers have already
-- been re-bootstrapped onto the new stream. Surface SUPERSEDED so the
-- Rust caller maps to `FailedPrecondition` (matches the slow path's
-- corresponding WATCH-window check). Non-retriable — no retry can
-- un-supersede an object — the caller's safe move is to re-resolve the
-- mapping and re-issue against the new object_id.
--
-- The check sits AFTER `hdr["seq"] == nil` because a partial header is
-- categorically worse than a superseded one (the latter is a normal
-- consequence of overwrite, the former is corruption); reporting the
-- categorical error first keeps operator triage clean.
if hdr["superseded"] then
  return redis.error_reply("SUPERSEDED: object superseded by a concurrent CreateLayer overwrite")
end

-- ── PREDECESSOR. A chained call's predecessor guard (the `^` tag). ────────
-- See "Writer guard" in the contract above. A chained call may commit only
-- directly after its predecessor: the field must name it (`op.<previous>:…`,
-- `guard_value` being `op.<previous>`), or, for a chain's first write
-- (`guard_value` `""`), be absent. Otherwise the reply is
-- `{"PREDECESSOR", <record or "">}` — the field as read, `""` when absent —
-- and nothing is written; the Rust caller refuses the call
-- `PREDECESSOR_NOT_COMMITTED` and hands back its pre-flight increments.
-- After a counted retry it instead answers the call a plain `UNAVAILABLE`,
-- outcome unknown, and looks up its own commit release ring entry (see
-- "Writer guard").
--
-- Position: after the own-key match (1b), so a re-send of this very operation
-- is answered `ALREADY_LANDED`, and after the superseded guard (1c), so an
-- object that can never take this commit is refused SUPERSEDED; before FENCED
-- (1d), so a call that may not commit yet is not made to wait for a reader,
-- and before the commit release record's type (1e), the ceilings and the
-- pre-pass. It reuses step 1b's `HGET` (`recorded`), so it costs no round
-- trip and no command of its own. The WATCH path makes the same check on its
-- step-2a read, after its superseded guard, and the cached structural route's
-- read side after its own superseded check, before planning (SDD §7.5
-- parity).
if guard_tag == "^" then
  local pending
  if guard_value == "" then
    pending = recorded ~= false
  else
    local prefix = guard_value .. ":"
    pending = not (recorded and string.sub(recorded, 1, #prefix) == prefix)
  end
  if pending then
    return { "PREDECESSOR", recorded or "" }
  end
end

-- Read-only keyspace-shape probe, shared by step 1d and the validation
-- pre-pass (step 4). Returns the offending type name when `key` exists and
-- is NOT `expected`, or `nil` when the key is absent ("none" — the apply
-- pass creates it, or, for the fence and intent keys, nothing is there) or
-- already the right type.
--
-- `TYPE` replies with a status reply, which the Lua client surfaces as a
-- table carrying an `ok` field. It never errors, on any key, so no `pcall`
-- is needed here.
local function wrong_type(key, expected)
  local actual = redis.call("TYPE", key)["ok"]
  if actual == expected or actual == "none" then
    return nil
  end
  return actual
end

-- ── 1d. Snapshot fence (guard 5, Spec SDD SR-5.4 / structural-edits SR-8.1),
--       both modes. ───────────────────────────────────────────────────────
-- A fenced snapshot reader (`snapshot_fence_acquire.lua`) proves its read by
-- holding a lease, `r:<token>` = deadline on this node's `TIME`, during which
-- no commit may land. Any live lease refuses this commit with the reply table
-- `{"FENCED", "<retry_after_ms>"}` (the latest live deadline − now, whole ms,
-- at least 1), writing nothing; the caller sleeps and re-runs the script
-- without spending a retry attempt. One piece of code for both modes.
--
-- Position: after guard 1c and the PREDECESSOR step, before the mode parse,
-- every later guard and any write — the canonical order is own-key match
-- (1b) → OVERTAKEN (1b) → SUPERSEDED (1c) → PREDECESSOR → FENCED (here) →
-- the commit release record's type (1e) → everything that can only refuse a
-- fresh apply (ENCODING, STALE, the ceilings, the pre-pass's WRONGTYPE_KEY
-- and CACHE_DIVERGED). A commit this call already landed is still reported
-- under a fence, and an object that can never take this commit is refused
-- SUPERSEDED — a chained call whose predecessor is not the record, PREDECESSOR
-- — rather than made to wait for a reader first.
--
-- The writer-intent key is not read here — neither mode registers an intent
-- (only the WATCH path does, `LayerStore::update_layer_once_watch`) — but
-- both keys are TYPE-probed, fence first, so a wrong-typed fence or intent
-- key is refused with the same `WRONGTYPE_KEY` reply in both modes and on
-- the WATCH path, whose fence probe (`snapshot_fence_probe.lua`) checks the
-- same two keys in the same order at the same point in its guard sequence
-- (SDD §7.5 parity). Unqualified on purpose: the WATCH path's reply carries
-- no locating qualifier for these keys, so neither does this one.
--
-- Cost on an unfenced layer: two `TYPE`s (fence, intent) inside the same
-- `EVALSHA` and nothing else — the fence key exists only for layers fenced
-- within the last `key_linger(L)`, and the `HGETALL` and `TIME` run only when
-- it does.
local fence_key  = KEYS[KEY_FENCE]
local intent_key = KEYS[KEY_INTENT]
-- The fence key's `TYPE` is kept, not folded through `wrong_type`: "none" is
-- what lets an unfenced commit skip the `HGETALL` and `TIME` below.
local fence_type = redis.call("TYPE", fence_key)["ok"]
if fence_type ~= "hash" and fence_type ~= "none" then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. fence_key .. " holds a " .. fence_type .. ", expected hash")
end
local intent_wrong_type = wrong_type(intent_key, "hash")
if intent_wrong_type ~= nil then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. intent_key .. " holds a " .. intent_wrong_type .. ", expected hash")
end
if fence_type == "hash" then
  local fence = redis.call("HGETALL", fence_key)
  local tm = redis.call("TIME")
  local now = tonumber(tm[1]) * 1000 + math.floor(tonumber(tm[2]) / 1000)
  local latest_live = nil
  for i = 1, #fence, 2 do
    if string.sub(fence[i], 1, 2) == "r:" then
      local deadline = tonumber(fence[i + 1])
      if deadline ~= nil and deadline > now and (latest_live == nil or deadline > latest_live) then
        latest_live = deadline
      end
    end
  end
  if latest_live ~= nil then
    return { "FENCED", string.format("%d", math.max(1, latest_live - now)) }
  end
end

-- ── 1e. Commit release record type (b.yuo), both modes. ──────────────────
-- Step 7a writes the commit release ring when the commit owes releases, so
-- the key is TYPE-probed here, before the mode parse, the pre-pass and any
-- write, which keeps that write un-rejectable after the `XADD`. Checked on
-- every run, whether or not this commit will owe releases: the verdict does
-- not depend on the delta.
--
-- Position: after FENCED (1d), not beside the commit-token probe at 1b. The
-- WATCH path checks this key's type at its step 2c, after its own-key match,
-- superseded guard, predecessor check and fence wait and before its snapshot
-- read, ceilings and stream-id collision probe, so a commit this call already
-- landed is reported `ALREADY_LANDED`, a superseded object is refused
-- `SUPERSEDED`, a chained call whose predecessor is not the record is refused
-- `PREDECESSOR` and a live fence is waited out before a wrong-typed release
-- record is refused, and a wrong-typed record is refused ahead of the
-- ceilings, the pre-pass and
-- the stream-id collision — on the WATCH path as here, and on the cached
-- structural route, whose Rust plan sends a call that meets a ceiling (an
-- object at one, or a delta whose own `set_order` would reach it) to the
-- WATCH path instead of refusing it before this script runs (SDD §7.5
-- parity). The refusal text is shared byte for byte
-- (`COMMIT_RELEASE_RECORD_TYPE_QUALIFIER` in `storage::layer`). Only a delta
-- that route's plan cannot apply to the cached state at all is refused by the
-- plan first, before the fence check (1d) and this step; that route's read
-- has already answered the own-key match, the superseded guard and the
-- predecessor check.
local wrel_key  = KEYS[KEY_COMMIT_RELEASE]
local wrel_type = redis.call("TYPE", wrel_key)["ok"]
if wrel_type ~= "hash" and wrel_type ~= "none" then
  return redis.error_reply(
    "WRONGTYPE_KEY: " .. wrel_key .. " holds a " .. wrel_type ..
    ", expected hash (commit release record)")
end

-- ── 2. Parse fixed-prefix ARGV. ────────────────────────────────────────
local server_ts     = ARGV[ARG_SERVER_TS]
local client_id     = ARGV[ARG_CLIENT_ID]
local client_ts     = ARGV[ARG_CLIENT_TS]
local stream_maxlen = tonumber(ARGV[ARG_STREAM_MAXLEN])
local object_id     = ARGV[ARG_OBJECT_ID]
local num_mutations = tonumber(ARGV[ARG_NUM_MUTATIONS])
-- b.gze: the writer's raw inbound OMPF, relayed verbatim to steady-state
-- subscribers (set_order + burst fields cannot be reconstructed from
-- materialized state — SDD §12.2, §12.8). An empty string means "no relay bytes".
local relay_ompf    = ARGV[ARG_RELAY_OMPF]

-- The relay byte budget's parameters (ARGV[11..14], see "Relay byte budget").
-- Each must be a canonical decimal at or below 2^53 - 1, and T, K and MIN at
-- least 1 when the budget is on; anything else is an encoder bug, refused
-- before any write.
local function budget_param(slot)
  local raw = ARGV[slot]
  if raw == nil or string.find(raw, "^%d+$") == nil then
    return nil
  end
  local n = tonumber(raw)
  if n > F64_INT_CEILING then
    return nil
  end
  return n
end
local relay_budget_bytes     = budget_param(ARG_RELAY_BUDGET_BYTES)
local relay_generation_bytes = budget_param(ARG_RELAY_GENERATION_BYTES)
local relay_generations      = budget_param(ARG_RELAY_GENERATIONS)
local relay_min_entries      = budget_param(ARG_RELAY_MIN_ENTRIES)
if relay_budget_bytes == nil or relay_generation_bytes == nil
    or relay_generations == nil or relay_min_entries == nil then
  return redis.error_reply("ENCODING: malformed relay byte budget (ARGV[11..14])")
end
if relay_budget_bytes > 0
    and (relay_generation_bytes < 1 or relay_generations < 1 or relay_min_entries < 1) then
  return redis.error_reply(
    "ENCODING: relay byte budget on with a zero generation size, count or minimum (ARGV[12..14])")
end

-- The mode slot, then the mode's own slots. `records_argv_start` is the ARGV
-- index of the first fast-mode record (fast mode only). In structural mode
-- `structural_argv_start` is the ARGV index of the untouched-row count block,
-- and the parse walk (step 4s) locates the first record past it.
local mode = ARGV[SHARED_ARGV + 1]
local records_argv_start
local structural_argv_start
local expected_seq_str
local expected_commit_chain
if mode == MODE_FAST then
  records_argv_start = SHARED_ARGV + 2
elseif mode == MODE_STRUCTURAL then
  expected_seq_str      = ARGV[SHARED_ARGV + 2]
  expected_commit_chain = ARGV[SHARED_ARGV + 3]
  -- A canonical decimal, because the fence below compares it with the
  -- header's `seq` as stored, byte for byte (every writer stores `seq` in
  -- canonical decimal), and `tonumber` would round a value past 2^53.
  if expected_seq_str == nil or string.find(expected_seq_str, "^%d+$") == nil then
    return redis.error_reply(
      "ENCODING: malformed expected_seq (ARGV[" .. (SHARED_ARGV + 2) .. "])")
  end
  if expected_commit_chain == nil then
    return redis.error_reply(
      "ENCODING: missing expected_commit_chain (ARGV[" .. (SHARED_ARGV + 3) .. "])")
  end
  structural_argv_start = SHARED_ARGV + 4
else
  return redis.error_reply(
    "ENCODING: unknown mode " .. tostring(mode) .. " (ARGV[" .. (SHARED_ARGV + 1) .. "])")
end

-- ── 2b. Structural fence (guard 6, structural mode only, SR-6.4). ────────
-- The plan was made at `(expected_seq, expected_commit_chain)`; its absolute
-- values are right only against exactly that state. A header at any other
-- `seq`, or at the same `seq` with a different `commit_chain` (a
-- re-materialization, or a backing-store rollback re-advanced by other
-- commits, SR-4.5), refuses the commit as `STALE <seq>`, carrying the
-- header's `seq` as stored. A header without the column reads as `""`, as the
-- `commit_chain` fold reads it. Read-only; nothing is written.
if mode == MODE_STRUCTURAL then
  if hdr["seq"] ~= expected_seq_str
      or (hdr["commit_chain"] or "") ~= expected_commit_chain then
    return redis.error_reply("STALE " .. hdr["seq"])
  end
end

-- ── 3. Derive the sequence + set-order cursors from the just-read
--       header. Mirrors `apply_delta`'s bookkeeping (SDD §9.1 steps 1
--       and 5a-5b): new_seq = seq + 1; next_set_order cursor carries the
--       header's current value and is advanced per SetField as we walk
--       the mutation records. In structural mode the header's `seq` is
--       `expected_seq` (step 2b), so `new_seq` is `expected_seq + 1`; the
--       `next_set_order` cursor is fast mode's alone (structural mode's
--       "HDR" record carries the plan's post-apply value). ─────────────────
local current_seq       = tonumber(hdr["seq"])
local next_set_order    = tonumber(hdr["next_set_order"]) or 0

-- b.k1r H-stor-9: guard against the 2^53 Lua double-precision ceiling
-- on `current_seq + 1`. Lua 5.1 `number` is IEEE 754 f64 and integer
-- precision drops past 2^53 — `2^53 + 1` evaluates to `2^53`, which
-- would silently allocate a duplicate sequence number. 2^53 writes is
-- astronomically large (≈ 9e15 commits), but the ceiling is real, the
-- check is one instruction on the hot path, and the alternative is an
-- eventual undebuggable corruption when a long-lived object finally
-- crosses the line. Matching check lives on the Rust txn path so the
-- two backends fail identically at the ceiling. Both modes (guard 7): in
-- structural mode `current_seq` is `expected_seq`, so this refuses exactly
-- when `expected_seq >= 2^53 - 1`.
if current_seq >= F64_INT_CEILING then
  return redis.error_reply("SEQ_OVERFLOW: object sequence number would exceed Lua f64 ceiling (2^53 - 1)")
end

local new_seq           = current_seq + 1
-- b.qw6 item 2 (proptest follow-up): Lua's `tostring(N)` uses the
-- `%.14g` default format, which switches to scientific notation when
-- an integer carries more than 14 significant digits — e.g.
-- `tostring(9007199254740976)` returns `"9.007199254741e+15"`. That
-- silently corrupts every numeric field stored as a hash value
-- (`seq`, `next_set_order`, `stored_set_order`) once the running
-- counter passes ~10^14, even though the underlying f64 still holds
-- the value exactly (the ceiling for that is 2^53 ≈ 9×10^15).
-- `string.format("%d", N)` preserves integer representation up to
-- 2^53 — the same ceiling we enforce elsewhere — so it's the right
-- shape for every hash value we store. The bare `tostring` lives on
-- only for non-numeric tags (`tag` in the encoding-error path).
local new_seq_str       = string.format("%d", new_seq)


-- ── 4. Validation pre-pass (b.ph9). Walks the ARGV mutation records
--       issuing no MUTATING call, deciding every failure this script can
--       reach before the keyspace is written. See the
--       "Validate-then-write" block at the top of this file for why the
--       split exists; the short version is that a mid-loop
--       `redis.error_reply` used to leave the records that preceded it
--       committed, with no way to roll them back.
--
--       Two classes of failure are decided here. The input-shaped ones
--       (KEYS count, unknown tag, malformed record, `set_order` ceiling)
--       need no server state and read only the call's KEYS and ARGV. The
--       keyspace-shape one — a
--       per-record key or index ZSet holding the wrong type — needs a
--       read-only `TYPE` probe per key. Those probes are `redis.call`s,
--       which is fine: the invariant is "no MUTATION until every abort
--       condition has been evaluated", not "no call at all". A `TYPE`
--       cannot itself fail on a live keyspace, and if the script died on
--       one it would die having written nothing.
--
--       This step is fast mode's walk; structural mode's pre-pass is step 4s
--       below. Each mode's walks visit only that mode's records: the fast
--       walks run `num_fast_records` times, which is zero in structural mode,
--       and the structural walks visit `structural_records`, which is empty
--       in fast mode.
--
--       KEYS cursor starts at SHARED_KEYS + 1 (after the shared KEYS
--       prefix, step 0's `KEY_*` slots). ARGV cursor starts
--       at `records_argv_start` (after the shared prefix and the mode slot).
--       This pass only walks ARGV —
--       the KEYS cursor is advanced by the apply pass below, whose record walk is
--       identical by construction (both are driven by the same
--       `num_value_hfields` counts read from the same ARGV slots).
--
--       The `set_order` fold happens HERE and only here: each SetField's
--       computed `stored_set_order` is stashed by record index and the
--       running `next_set_order` cursor is left holding the value the
--       header HSET will persist. The apply pass consumes the stash
--       rather than recomputing, so the two walks cannot drift. ────────
local fields_index_key = KEYS[KEY_FIELDS_INDEX]
local ts_index_key     = KEYS[KEY_TS_INDEX]
-- b.qpf: per-object external-value reference index. Written by the apply pass
-- below, so it is type-probed here with the per-record keys.
local ext_index_key    = KEYS[KEY_EXT_INDEX]

-- Per-record stash, indexed by record ordinal. SetField records get a
-- number; time-sample records leave the slot `nil` (never read).
local stored_set_orders = {}

-- (`wrong_type`, the read-only keyspace-shape probe this pass uses, is
-- defined above step 1d, which shares it.)

-- Whether the fields / time-sample index ZSets are actually touched by
-- this commit. An index is probed only when a record of its class exists:
-- an all-time-sample delta must not be refused because the *fields* index
-- is corrupt, and an empty delta touches neither.
local saw_field = false
local saw_ts    = false

-- The number of fast-mode records the fast walks visit: every record in fast
-- mode, none in structural mode.
--
-- Fast mode's KEYS shape is decided here, before the walk reads a record key:
-- each record consumes exactly one key, so KEYS holds exactly the shared
-- prefix plus `num_mutations` keys, and any other count is an encoder bug
-- (see "Contract"). Checked whole rather than per record, because a key added
-- mid-list shifts every later record onto its neighbour's key while leaving
-- none of them keyless, so no per-record presence check can see it.
local num_fast_records = 0
if mode == MODE_FAST then
  if num_mutations == nil then
    return redis.error_reply(
      "ENCODING: malformed num_mutations (ARGV[" .. ARG_NUM_MUTATIONS .. "])")
  end
  if #KEYS ~= SHARED_KEYS + num_mutations then
    return redis.error_reply(
      "ENCODING: " .. #KEYS .. " KEYS for " .. ARGV[ARG_NUM_MUTATIONS] ..
      " mutation record(s); expected " .. (SHARED_KEYS + num_mutations))
  end
  num_fast_records = num_mutations
end

local scan_cursor = records_argv_start
for i = 1, num_fast_records do
  local tag = ARGV[scan_cursor]
  scan_cursor = scan_cursor + 1

  if tag == "SF" or tag == "ST" then
    -- The apply pass consumes exactly one KEYS entry per record, starting
    -- at KEYS[SHARED_KEYS + 1], so record `i` maps to KEYS[i + SHARED_KEYS].
    -- The KEYS count check above guarantees the entry is present.
    local record_key = KEYS[i + SHARED_KEYS]
    local actual = wrong_type(record_key, "hash")
    if actual ~= nil then
      return redis.error_reply(
        "WRONGTYPE_KEY: " .. record_key .. " holds a " .. actual ..
        ", expected hash (mutation record " .. i .. ")")
    end

    local num_value_hfields = tonumber(ARGV[scan_cursor])
    if num_value_hfields == nil then
      -- Malformed record — a Rust-side encoder bug. Fail before any write.
      return redis.error_reply(
        "ENCODING: malformed value-hfield count in mutation record " .. i)
    end
    -- Skip the count itself plus the name/value pairs it introduces.
    scan_cursor = scan_cursor + 1 + (num_value_hfields * 2)

    if tag == "SF" then
      saw_field = true
      local delta_set_order = tonumber(ARGV[scan_cursor])
      if delta_set_order == nil then
        return redis.error_reply(
          "ENCODING: malformed delta_set_order in mutation record " .. i)
      end
      scan_cursor = scan_cursor + 2  -- delta_set_order + zadd_member

      -- `stored_set_order = next_set_order + delta_set_order` and
      -- `next_set_order = max(next_set_order, stored_set_order + 1)`.
      -- Exact arithmetic from `apply_set_field` (src/domain/delta.rs).
      --
      -- b.qw6 item 2: guard against the 2^53 Lua double-precision ceiling
      -- on the addition. Lua 5.1 `number` is IEEE 754 f64 — integer
      -- precision drops past 2^53 so `next_set_order + delta_set_order`
      -- silently rounds when the sum crosses the line, and the resulting
      -- `set_order` field stored on Valkey is no longer monotonically
      -- distinct from neighboring writes. The slow path's
      -- `apply_set_field` uses `checked_add` on u64 (capping at u64::MAX),
      -- but the actual reachable ceiling on the fast path is 2^53, so we
      -- pin both backends at 2^53 here for parity and add a matching
      -- check on the slow path (see `apply_set_field` in
      -- `src/domain/delta.rs`). Mirrors the SEQ_OVERFLOW reply shape so
      -- the Rust caller can detect it via the same `details.starts_with(...)`
      -- pattern; non-retriable (no retry can shrink the cursor).
      -- `delta_set_order` upper-bound at delta validation
      -- (`validate_delta_bounds`) keeps a single malicious delta from
      -- driving past the ceiling in one shot, but objects that
      -- legitimately accumulate set_order over time can still reach the
      -- ceiling and must fail loudly here instead of silently corrupting.
      --
      -- b.qw6 PM follow-up: the comparison is `>=`, not `>`. The slow
      -- path's belt-and-braces guard in `apply_set_field` rejects when
      -- `stored_set_order >= SET_ORDER_CEILING` (i.e. sum equals the
      -- ceiling) — because `next_set_order` after the update is
      -- `stored + 1 = 2^53`, which the NEXT call could not advance past
      -- the same precision boundary. Pre-fix, `>` here let the Lua path
      -- commit at `sum == 2^53 - 1` while the Txn path rejected the
      -- identical input, violating AC #2's byte-identical-behavior
      -- contract. Pin both backends at "reject when sum >= 2^53 - 1".
      --
      -- b.ph9: raised from the apply loop into this pre-pass. Pre-fix
      -- this fired mid-loop, so records 1..i-1 had already been HSET —
      -- and their `version = new_seq` sat above a header whose `seq` was
      -- never advanced, breaking the version-filtered-diff invariant
      -- that catch-up reconstruction depends on.
      if next_set_order + delta_set_order >= F64_INT_CEILING then
        return redis.error_reply("SET_ORDER_OVERFLOW: next_set_order + delta_set_order would exceed Lua f64 ceiling (2^53 - 1)")
      end
      local stored_set_order = next_set_order + delta_set_order
      stored_set_orders[i] = stored_set_order
      if stored_set_order + 1 > next_set_order then
        next_set_order = stored_set_order + 1
      end
    else
      saw_ts = true
      scan_cursor = scan_cursor + 1  -- zadd_member
    end
  else
    -- Unknown tag — a Rust-side encoder bug. Fail loudly so the bad
    -- encoding is caught in tests rather than silently committing a
    -- partial update. b.ph9: raised here rather than mid-apply, so a
    -- malformed record N no longer commits records 1..N-1.
    return redis.error_reply("ENCODING: unknown mutation tag " .. tostring(tag))
  end
end

-- Index ZSets, probed once each and only when this commit will ZADD them.
-- Same fail-closed reasoning as the per-record keys: a WRONGTYPE here would
-- otherwise abort the apply pass with the stream entry already published.
if saw_field then
  local actual = wrong_type(fields_index_key, "zset")
  if actual ~= nil then
    return redis.error_reply(
      "WRONGTYPE_KEY: " .. fields_index_key .. " holds a " .. actual ..
      ", expected zset (fields index)")
  end
end
if saw_ts then
  local actual = wrong_type(ts_index_key, "zset")
  if actual ~= nil then
    return redis.error_reply(
      "WRONGTYPE_KEY: " .. ts_index_key .. " holds a " .. actual ..
      ", expected zset (time-samples index)")
  end
end
-- b.qpf: the external-value reference index. Probed whenever this commit
-- carries ANY value record, not only records with external values: a record
-- carrying an inline value can still OVERWRITE a row that held an external one,
-- which decrements the index just the same. Same fail-closed reasoning as the
-- two index ZSets — a WRONGTYPE discovered during the apply pass would abort it
-- with the stream entry already published.
if saw_field or saw_ts then
  local actual = wrong_type(ext_index_key, "hash")
  if actual ~= nil then
    return redis.error_reply(
      "WRONGTYPE_KEY: " .. ext_index_key .. " holds a " .. actual ..
      ", expected hash (external-value reference index)")
  end
end

-- ── 4s. Structural pre-pass (structural mode only; SR-6.2 guards 7–8,
--        SR-6.3). Three walks over the records, each finished before the
--        next starts and none of them writing:
--
--          1. parse (ARGV only): the untouched-row count block, then the
--             records; every input-shaped refusal → `ENCODING`, and the
--             set_order ceilings → `SET_ORDER_OVERFLOW`;
--          2. probe: the `TYPE` probes (see "Type probes" in the contract)
--             → `WRONGTYPE_KEY`;
--          3. digests: each record's pre-image digest against the store
--             → `CACHE_DIVERGED`.
--
--        The parsed records are kept, so the apply pass consumes them rather
--        than re-reading ARGV, and the walks cannot drift. ─────────────────

-- Record `i`'s key is `KEYS[STRUCTURAL_RECORD_KEYS_BASE + i]`, after the
-- shared prefix and the three object-level indexes.
local sections_index_key         = KEYS[SHARED_KEYS + 1]
local deleted_sections_index_key = KEYS[SHARED_KEYS + 2]
local children_index_key         = KEYS[SHARED_KEYS + 3]
local STRUCTURAL_RECORD_KEYS_BASE = SHARED_KEYS + 3

-- A section row's six stored columns, in `layer::section_hash_fields` order:
-- the order a "SEC" record carries them in, a "row" digest's six slots are
-- in, and the section `HSET` writes them in.
local SECTION_COLUMNS = { "parent_id", "name", "type", "order", "first_ver", "last_ver" }

-- The parsed structural records, in record order, and the one "HDR" record.
-- Empty / nil in fast mode.
local structural_records = {}
local structural_hdr = nil

-- The untouched-row count block (see "Untouched-row counts" in the contract),
-- hash → count, as walk 1 parsed it. Read only by settle pass 6b. Empty in
-- fast mode.
local ext_untouched = {}

local function is_decimal(s)
  return s ~= nil and string.find(s, "^%d+$") ~= nil
end

-- Whether 64-hex hash `a` sorts strictly before `b`, byte by byte — the order
-- Rust sorts the block's `[u8; 32]` hashes in. Compared bytewise rather than
-- with Lua's `<`, which compares strings with the server's `strcoll` locale.
local function hash_before(a, b)
  for j = 1, 64 do
    local x, y = string.byte(a, j), string.byte(b, j)
    if x ~= y then
      return x < y
    end
  end
  return false
end

-- Walk 1: parse. Returns nil, or the refusal to return. Parses the
-- untouched-row count block into `ext_untouched`, then the records: each
-- record becomes a table holding its tag, its key and its slots by name.
local function parse_structural_records()
  if num_mutations == nil then
    return redis.error_reply(
      "ENCODING: malformed num_mutations (ARGV[" .. ARG_NUM_MUTATIONS .. "])")
  end
  -- The KEYS shape: every record consumes exactly one key, after the shared
  -- prefix and the three object-level indexes, so any other count is an
  -- encoder bug (see "Contract"). Checked whole, before any record key is
  -- read, because a key added or dropped mid-list shifts every later record
  -- onto its neighbour's key, which a record without a digest cannot detect.
  if #KEYS ~= STRUCTURAL_RECORD_KEYS_BASE + num_mutations then
    return redis.error_reply(
      "ENCODING: " .. #KEYS .. " KEYS for " .. ARGV[ARG_NUM_MUTATIONS] ..
      " mutation record(s); expected " .. (STRUCTURAL_RECORD_KEYS_BASE + num_mutations))
  end
  local cursor = structural_argv_start
  -- Set by `take` when a record runs past the end of ARGV. Checked once per
  -- record, after its slots are taken, so a short record never reaches the
  -- per-tag validation (or the apply pass) with a nil slot.
  local short = false
  local function take()
    local v = ARGV[cursor]
    cursor = cursor + 1
    if v == nil then
      short = true
    end
    return v
  end
  -- The ARGV slots not yet taken: bounds every count a record carries, so a
  -- corrupt count cannot spin the walk.
  local function remaining()
    return #ARGV - cursor + 1
  end
  local function encoding(i, what)
    return redis.error_reply("ENCODING: " .. what .. " in mutation record " .. i)
  end

  -- The untouched-row count block, ahead of the records. `K` is bounded by
  -- the slots left, so every pair below is present.
  local k = take()
  if not is_decimal(k) then
    return redis.error_reply(
      "ENCODING: malformed untouched-row count block size (ARGV[" .. structural_argv_start .. "])")
  end
  if tonumber(k) * 2 > remaining() then
    return redis.error_reply(
      "ENCODING: untouched-row count block of " .. k .. " entries runs past the end of ARGV")
  end
  local previous_hash = nil
  for j = 1, tonumber(k) do
    local hash = take()
    local n = take()
    if #hash ~= 64 or string.find(hash, "^[0-9a-f]+$") == nil then
      return redis.error_reply("ENCODING: malformed hash in untouched-row count entry " .. j)
    end
    -- The ceiling keeps the count exact in a Lua number, so settle pass 6b's
    -- `string.format("%d", ...)` cannot misrender it after the `XADD`.
    if not is_decimal(n) or tonumber(n) > F64_INT_CEILING then
      return redis.error_reply("ENCODING: malformed count in untouched-row count entry " .. j)
    end
    if previous_hash ~= nil and not hash_before(previous_hash, hash) then
      return redis.error_reply(
        "ENCODING: hash not strictly ascending in untouched-row count entry " .. j)
    end
    ext_untouched[hash] = tonumber(n)
    previous_hash = hash
  end

  for i = 1, num_mutations do
    -- Present: the KEYS count was checked above.
    local key = KEYS[STRUCTURAL_RECORD_KEYS_BASE + i]
    local tag = take()
    local r = { tag = tag, key = key }

    if tag == "SEC" or tag == "DSEC" then
      r.sid = take()
      if tag == "SEC" then
        r.columns = {}
        for j = 1, 6 do
          r.columns[j] = take()
        end
      end
      r.digest = take()
      if r.digest == "row" then
        r.pre = {}
        for j = 1, 6 do
          r.pre[j] = take()
        end
      end
    elseif tag == "TOMB" then
      r.sid = take()
      r.first_ver = take()
      r.last_ver = take()
    elseif tag == "CHL" or tag == "DCHL" then
      r.parent = take()
      if tag == "CHL" then
        local n = take()
        if not is_decimal(n) or tonumber(n) > remaining() then
          return encoding(i, "malformed children count")
        end
        r.children = {}
        for j = 1, tonumber(n) do
          r.children[j] = take()
        end
      end
      r.digest = take()
    elseif tag == "SFA" or tag == "STA" then
      local count = take()
      if not is_decimal(count) or tonumber(count) * 2 > remaining() then
        return encoding(i, "malformed value-hfield count")
      end
      -- The value pairs, verbatim. The presence of a `hash` pair is the
      -- external / inline discriminator, as in fast mode.
      r.pairs = {}
      for j = 1, tonumber(count) * 2 do
        r.pairs[j] = take()
      end
      for j = 1, #r.pairs, 2 do
        if r.pairs[j] == "hash" then
          r.new_hash = r.pairs[j + 1]
        end
      end
      if tag == "SFA" then
        r.set_order = take()
        r.authoring_set_order = take()
      end
      r.version = take()
      r.member = take()
    elseif tag == "DF" or tag == "DT" then
      r.member = take()
      r.digest = take()
    elseif tag == "HDR" then
      r.seq = take()
      r.first_ver = take()
      r.next_set_order = take()
      local has_src_fmt = take()
      if has_src_fmt == "1" then
        r.src_fmt = take()
      elseif has_src_fmt ~= "0" and has_src_fmt ~= nil then
        return encoding(i, "malformed has_src_fmt " .. has_src_fmt)
      end
    else
      return redis.error_reply("ENCODING: unknown mutation tag " .. tostring(tag))
    end

    if short then
      return encoding(i, "truncated " .. tag .. " record")
    end

    -- Per-tag slot validation: the digest tokens, and every slot the script
    -- does arithmetic on or scores an index with. The ceilings (guard 7) are
    -- decided here, per record, as each is parsed.
    if tag == "SEC" or tag == "DSEC" then
      if r.digest ~= "none" and r.digest ~= "absent" and r.digest ~= "row" then
        return encoding(i, "unknown section digest " .. r.digest)
      end
    elseif tag == "CHL" or tag == "DCHL" then
      if r.digest ~= "none" and r.digest ~= "absent"
          and not (#r.digest == 40 and string.find(r.digest, "^[0-9a-f]+$")) then
        return encoding(i, "unknown children-list digest " .. r.digest)
      end
    elseif tag == "SFA" or tag == "STA" then
      if tag == "SFA" then
        if not is_decimal(r.set_order) then
          return encoding(i, "malformed set_order")
        end
        -- Fast mode's `>=` boundary (its step-4 `SET_ORDER_OVERFLOW`, and
        -- `apply_set_field` on the WATCH path): a stored `set_order` of
        -- 2^53 - 1 leaves the cursor at 2^53, past the exact range.
        if tonumber(r.set_order) >= F64_INT_CEILING then
          return redis.error_reply(
            "SET_ORDER_OVERFLOW: set_order " .. r.set_order .. " in mutation record " .. i ..
            " would exceed Lua f64 ceiling (2^53 - 1)")
        end
      end
      if not is_decimal(r.version) then
        return encoding(i, "malformed version")
      end
    elseif tag == "DF" or tag == "DT" then
      if r.digest ~= "none" and r.digest ~= "exists" and r.digest ~= "absent" then
        return encoding(i, "unknown row digest " .. r.digest)
      end
    elseif tag == "HDR" then
      if structural_hdr ~= nil then
        return encoding(i, "second HDR record")
      end
      if key ~= header_key then
        return encoding(i, "HDR key " .. key .. " is not the header key")
      end
      -- The plan was made at `expected_seq`, so it must land at the next
      -- `seq` — the one this script allocates (`new_seq`, step 3).
      if r.seq ~= new_seq_str then
        return encoding(i, "HDR seq " .. r.seq .. " is not expected_seq + 1 (" .. new_seq_str .. ")")
      end
      if not is_decimal(r.next_set_order) then
        return encoding(i, "malformed HDR next_set_order")
      end
      -- `>`, not `>=`: `next_set_order` is the post-apply cursor, and 2^53 - 1
      -- is a legal cursor (the value a stored `set_order` of 2^53 - 2 leaves).
      if tonumber(r.next_set_order) > F64_INT_CEILING then
        return redis.error_reply(
          "SET_ORDER_OVERFLOW: HDR next_set_order " .. r.next_set_order ..
          " would exceed Lua f64 ceiling (2^53 - 1)")
      end
      structural_hdr = r
    end

    structural_records[i] = r
  end

  if cursor ~= #ARGV + 1 then
    return redis.error_reply(
      "ENCODING: " .. (#ARGV + 1 - cursor) .. " ARGV slot(s) past the last mutation record")
  end
  if structural_hdr == nil then
    return redis.error_reply("ENCODING: no HDR record")
  end
  return nil
end

-- Walk 2: the `TYPE` probes. Returns nil, or the refusal to return. Each key
-- is probed once.
local function probe_structural_records()
  local probed = {}
  local function probe(key, expected, what)
    if probed[key] then
      return nil
    end
    probed[key] = true
    local actual = wrong_type(key, expected)
    if actual ~= nil then
      return redis.error_reply(
        "WRONGTYPE_KEY: " .. key .. " holds a " .. actual ..
        ", expected " .. expected .. " (" .. what .. ")")
    end
    return nil
  end

  local saw_section, saw_tombstone, saw_list = false, false, false
  local saw_field_row, saw_ts_row = false, false
  for i, r in ipairs(structural_records) do
    local tag = r.tag
    -- The record's own key: probed when the apply pass `HSET`s it (the
    -- WATCH path's coverage) or the digest walk reads it; never for a
    -- children list (see "Type probes").
    local probe_key = false
    if tag == "SEC" or tag == "TOMB" or tag == "SFA" or tag == "STA" or tag == "HDR" then
      probe_key = true
    elseif tag == "DSEC" or tag == "DF" or tag == "DT" then
      probe_key = r.digest ~= "none"
    end
    if probe_key then
      local err = probe(r.key, "hash", "mutation record " .. i)
      if err then
        return err
      end
    end

    if tag == "SEC" or tag == "DSEC" then
      saw_section = true
    elseif tag == "TOMB" then
      saw_tombstone = true
    elseif tag == "CHL" or tag == "DCHL" then
      saw_list = true
    elseif tag == "SFA" or tag == "DF" then
      saw_field_row = true
    elseif tag == "STA" or tag == "DT" then
      saw_ts_row = true
    end
  end

  -- The object-level indexes, each probed only when a record writes it, as
  -- in fast mode.
  local indexes = {
    { saw_section, sections_index_key, "set", "sections index" },
    { saw_tombstone, deleted_sections_index_key, "set", "deleted-sections index" },
    { saw_list, children_index_key, "set", "children-list index" },
    { saw_field_row, fields_index_key, "zset", "fields index" },
    { saw_ts_row, ts_index_key, "zset", "time-samples index" },
    { saw_field_row or saw_ts_row, ext_index_key, "hash", "external-value reference index" },
  }
  for _, index in ipairs(indexes) do
    if index[1] then
      local err = probe(index[2], index[3], index[4])
      if err then
        return err
      end
    end
  end
  return nil
end

-- Walk 3: the pre-image digests. Returns nil, or the `CACHE_DIVERGED`
-- refusal naming the first record whose key does not hold what the cache
-- believed. Every key read here was probed by walk 2, except a children list,
-- which is classified by its `TYPE` and read only when it is a list.
local function check_structural_digests()
  for i, r in ipairs(structural_records) do
    local digest = r.digest
    local tag = r.tag
    local diverged = nil
    if digest == nil or digest == "none" then
      diverged = nil
    elseif tag == "SEC" or tag == "DSEC" then
      if digest == "absent" then
        if redis.call("EXISTS", r.key) == 1 then
          diverged = "the cache holds no such section, and the row exists"
        end
      else
        -- "row": `HMGET` answers a missing field (and every field of an absent
        -- row) as Lua `false`, which equals no carried column string.
        local stored = redis.call("HMGET", r.key, unpack(SECTION_COLUMNS))
        for j = 1, 6 do
          if stored[j] ~= r.pre[j] then
            diverged = "stored " .. SECTION_COLUMNS[j] .. " differs from the cache's row"
            break
          end
        end
      end
    elseif tag == "DF" or tag == "DT" then
      local exists = redis.call("EXISTS", r.key) == 1
      if digest == "exists" and not exists then
        diverged = "the cache holds the row, and it is absent"
      elseif digest == "absent" and exists then
        diverged = "the cache holds no such row, and it exists"
      end
    elseif tag == "CHL" or tag == "DCHL" then
      local actual = redis.call("TYPE", r.key)["ok"]
      if actual == "none" then
        if digest ~= "absent" then
          diverged = "the cache holds the list, and it is absent"
        end
      elseif actual ~= "list" then
        diverged = "the key holds a " .. actual .. ", not a list"
      elseif digest == "absent" then
        diverged = "the cache holds no such list, and it exists"
      elseif redis.sha1hex(table.concat(redis.call("LRANGE", r.key, 0, -1), ",")) ~= digest then
        diverged = "the stored list differs from the cache's"
      end
    end
    if diverged ~= nil then
      return redis.error_reply(
        "CACHE_DIVERGED: mutation record " .. i .. " (" .. tag .. ") " .. r.key ..
        ": " .. diverged)
    end
  end
  return nil
end

if mode == MODE_STRUCTURAL then
  local err = parse_structural_records()
  if err == nil then
    err = probe_structural_records()
  end
  if err == nil then
    err = check_structural_digests()
  end
  if err ~= nil then
    return err
  end
end

-- ── 4r. Fold the relay byte accounting (see "Relay byte budget"). Pure Lua
--        over the step-1 header and ARGV — no call, so nothing here can
--        refuse — and the statement-for-statement twin of
--        `storage::relay_retention::RelayRetention::fold`. `relay_gens_value`
--        rides the step-7 header HSET; `relay_trim_cut` (nil = no trim) the
--        step-5a XTRIM. Both stay nil with the budget off. ─────────────────
local relay_gens_value = nil
local relay_trim_cut   = nil

-- `a + b` clamped to 2^53 - 1. Both inputs are at or below it, so the true
-- sum is at most 2^54 - 2, and an over-ceiling sum, however a double rounds
-- it, still compares above the ceiling.
local function clamped_sum(a, b)
  return math.min(a + b, F64_INT_CEILING)
end

-- The column as `{ {start=, bytes=}, ... }`, oldest first, or nil when it
-- must be read as fresh (fold step 1).
local function parse_relay_gens(value)
  if not value then
    return nil
  end
  local gens = {}
  local pos = 1
  local len = #value
  while true do
    local s, e, start_digits, bytes_digits = string.find(value, "^(%d+):(%d+)", pos)
    if s == nil then
      return nil
    end
    local start = tonumber(start_digits)
    local bytes = tonumber(bytes_digits)
    if start > F64_INT_CEILING or bytes > F64_INT_CEILING then
      return nil
    end
    if start > new_seq then
      return nil
    end
    if #gens > 0 and start <= gens[#gens].start then
      return nil
    end
    gens[#gens + 1] = { start = start, bytes = bytes }
    pos = e + 1
    if pos > len then
      break
    end
    if string.sub(value, pos, pos) ~= "," then
      return nil
    end
    pos = pos + 1
  end
  return gens
end

if relay_budget_bytes > 0 then
  -- Step 1.
  local gens = parse_relay_gens(hdr["relay_gens"]) or { { start = new_seq, bytes = 0 } }
  -- Step 2.
  gens[#gens].bytes = clamped_sum(gens[#gens].bytes, #relay_ompf)
  -- Step 3: keep the longest run of newest generations whose bytes sum to at
  -- most B, and always the newest one.
  local first_kept = #gens
  local kept_bytes = gens[first_kept].bytes
  while first_kept > 1 do
    local candidate = gens[first_kept - 1].bytes
    if kept_bytes + candidate > relay_budget_bytes then
      break
    end
    kept_bytes = kept_bytes + candidate
    first_kept = first_kept - 1
  end
  local dropped = first_kept > 1
  local kept = {}
  for i = first_kept, #gens do
    kept[#kept + 1] = gens[i]
  end
  gens = kept
  -- Step 4.
  if gens[#gens].bytes >= relay_generation_bytes then
    gens[#gens + 1] = { start = new_seq + 1, bytes = 0 }
  end
  -- Step 5.
  while #gens > relay_generations + 1 do
    gens[1].bytes = clamped_sum(gens[1].bytes, gens[2].bytes)
    table.remove(gens, 2)
  end
  -- Step 6.
  local pairs_out = {}
  for i = 1, #gens do
    pairs_out[i] = string.format("%d", gens[i].start) .. ":" .. string.format("%d", gens[i].bytes)
  end
  relay_gens_value = table.concat(pairs_out, ",")
  -- Step 7.
  if dropped then
    relay_trim_cut = math.min(gens[1].start, math.max(1, new_seq - relay_min_entries + 1))
  end
end

-- ── 5. XADD the live-update entry — the FIRST mutating call in this
--       script (b.ph9). ID is "<new_seq>-0"; payload matches the legacy
--       `update|...` format exactly (SDD §7.5). ─────────────────────────
local stream_key = KEYS[KEY_STREAM]
local payload = "update|" .. object_id
  .. "|" .. new_seq_str
  .. "|" .. client_id
  .. "|" .. client_ts
  .. "|" .. server_ts
local stream_id = new_seq_str .. "-0"

-- b.gze: when the writer supplied a non-empty inbound OMPF, attach it as a
-- second binary `ompf` field so steady-state subscribers can relay it
-- verbatim. The bytes ride inside the same XADD, sharing the entry's
-- retention — the `MAXLEN ~` entry cap here, and the byte trim of step 5a —
-- and this script's atomicity with the state mutation. An empty string keeps
-- the pre-b.gze single-field shape.
--
-- b.ph9: `redis.pcall`, not `redis.call`. After the pre-pass this is the
-- only *mutating* call the keyspace can still reject — and, being the
-- script's FIRST write, it is also the call that absorbs every server-wide
-- first-write gate. Two very different classes of rejection arrive here and
-- must NOT be conflated:
--
--   A. Something is wrong with THIS STREAM KEY. Non-retriable, operator
--      must intervene, tagged STREAM_ID_COLLISION. Exactly two members:
--
--        1. Stream-ID monotonicity — the stream's top entry is already at
--           or above `<new_seq>-0`. Valkey answers
--           `ERR The ID specified in XADD is equal or smaller than the
--           target stream top item` (`streamAppendItem` → `EDOM`, valkey
--           `src/t_stream.c`); matched below on the stable fragment
--           `equal or smaller`. The genuine collision.
--        2. WRONGTYPE — something other than a stream occupies the key.
--           Folded into the same tag on purpose: same fail-closed operator
--           condition, same "clear the stream key" remediation, and the
--           slow path's probe classifies it identically (backend parity).
--
--   B. Something is wrong with THE SERVER, and this stream is an innocent
--      bystander. `scriptCall` (valkey `src/script.c`) runs a fixed
--      verification chain in front of every command a script issues, and
--      two of its members refuse only the script's FIRST write
--      (`!(flags & SCRIPT_WRITE_DIRTY)`) — which, post-b.ph9, is precisely
--      this XADD:
--        * `scriptVerifyWriteCommandAllow` → `-READONLY` on a read-only
--          replica (a managed failover or a mis-routed connection),
--          `-MISCONF` when RDB/AOF persistence is failing, `-NOREPLICAS`
--          under `min-replicas-to-write`;
--        * `scriptVerifyOOM` → `-OOM` under `maxmemory`.
--      Two more refuse regardless of write-dirty state and so can also
--      land on this call: `scriptVerifyAllowStale` (stale replica) and
--      `scriptVerifyACL` (`-NOPERM`), as can `scriptVerifyClusterState`
--      (`-MOVED` / `-CROSSSLOT`). Plus whatever gate a future server
--      version adds. Every one of these is passed through VERBATIM so it
--      reaches the Rust caller and is classified as itself: the four
--      first-write refusals above become a nothing-applied `UNAVAILABLE`
--      (`storage::client::store_write_refusal`, never retried in-process),
--      and the rest reach the shared fred classifier — several of those
--      are transient and therefore retriable.
--
-- The classification below is therefore an ALLOWLIST, not a denylist: only
-- the two class-A shapes get the tag, everything else passes through. The
-- inverse — "tag everything that isn't OOM" — is what this replaces
-- (post-completion review of b.ph9), and it mis-reported every class-B
-- refusal as a keyspace anomaly: non-retriable `INTERNAL` to a client for a
-- transient condition, a tick on a zero-baseline pageable counter, and a
-- WARN telling the operator to delete a perfectly healthy stream. A
-- denylist here is unmaintainable by construction, because the set of
-- first-write gates belongs to the server, not to us.
--
-- Read this before "simplifying" the class-B pass-through away. It is
-- tempting to assume legacy `EVALSHA` is deny-oom at the command level and
-- that an OOM can therefore never reach this call. It is not, and it does.
-- In `evalGetCommandFlags` (valkey `src/eval.c`) a script with no shebang
-- carries `SCRIPT_FLAG_EVAL_COMPAT_MODE` and returns the *command's own*
-- flags unchanged — and `EVAL`/`EVALSHA` in `commands.def` carry neither
-- `CMD_DENYOOM` nor `CMD_WRITE` — so `processCommand`'s pre-execution OOM
-- and deny-write gates do not fire; `scriptFlagsToCmdFlags`, which would
-- have added `CMD_DENYOOM`, is reached only by shebang scripts and
-- functions. `scriptPrepareForRun` likewise skips its OOM / READONLY /
-- MISCONF / NOREPLICAS block in compat mode (that block sits under the
-- declared-flags branch) and does NOT set `SCRIPT_ALLOW_OOM`. What remains
-- is the per-command chain listed in (B) above, whose first-write members
-- therefore all fire on this `XADD` — `scriptVerifyOOM` rejecting with
-- `-OOM command not allowed when used memory > 'maxmemory'.`, and
-- `scriptVerifyWriteCommandAllow` with `shared.roreplicaerr` /
-- the disk-error MISCONF text / `shared.noreplicaserr`.
--
-- (That same rule is what makes the apply pass un-rejectable once this
-- call succeeds: the OOM and deny-write gates are spent, so the HSETs and
-- ZADDs below cannot be refused for memory or writability even under
-- `noeviction`.)
--
-- `pcall` returns the error as a table instead of aborting the script, so
-- we can classify it and return with the keyspace untouched. The
-- `type(x) == 'table' and x.err` idiom matches
-- `scripts/stream_high_water_probe.lua`.
local xadd_result
if stream_maxlen > 0 then
  -- b.qw6 item 2 (proptest follow-up): match the `string.format("%d", N)`
  -- shape used for `seq` and `next_set_order` so MAXLEN never picks up
  -- scientific notation either. Defense in depth — the Rust config
  -- ceiling check (item 3) already rejects values past 2^53 at
  -- startup, so a misconfigured MAXLEN can't reach this branch
  -- anymore, but %.14g's behavior past 10^14 is enough of a foot-gun
  -- that we use the same exact-integer format everywhere.
  if relay_ompf ~= "" then
    xadd_result = redis.pcall(
      "XADD", stream_key,
      "MAXLEN", "~", string.format("%d", stream_maxlen),
      stream_id,
      "payload", payload,
      "ompf", relay_ompf
    )
  else
    xadd_result = redis.pcall(
      "XADD", stream_key,
      "MAXLEN", "~", string.format("%d", stream_maxlen),
      stream_id,
      "payload", payload
    )
  end
else
  if relay_ompf ~= "" then
    xadd_result = redis.pcall(
      "XADD", stream_key,
      stream_id,
      "payload", payload,
      "ompf", relay_ompf
    )
  else
    xadd_result = redis.pcall(
      "XADD", stream_key,
      stream_id,
      "payload", payload
    )
  end
end

if type(xadd_result) == "table" and xadd_result.err then
  local xadd_err = xadd_result.err
  -- ALLOWLIST (class A above). Only these two shapes are verdicts about
  -- this stream key; everything else is a verdict about the server.
  --
  --   * `equal or smaller` — a stable fragment of valkey's
  --     `The ID specified in XADD is equal or smaller than the target
  --     stream top item` (`src/t_stream.c`, the `EDOM` arm of
  --     `streamAppendItem`). Matched with `plain = true` so the fragment is
  --     never read as a Lua pattern. Deliberately a fragment and not the
  --     full sentence: the sentence is wrapped across two string literals
  --     in the server source, so the exact whitespace is a refactor away
  --     from changing, while these three words have been byte-stable since
  --     streams shipped. The *other* XADD ID error — `The ID specified in
  --     XADD must be greater than 0-0` — is unreachable here (`new_seq` is
  --     `current_seq + 1 >= 1`) and would not be a collision anyway, so it
  --     correctly falls through to the pass-through below.
  --   * `WRONGTYPE` — the stream key holds a non-stream. Prefix match; the
  --     server always emits this code first in the message.
  --
  -- Anything else — every class-B server-state refusal, and every gate a
  -- future server version adds — is returned VERBATIM, exactly as OOM
  -- alone was pre-review. It carries no STREAM_ID_COLLISION tag, so the
  -- Rust caller's collision arm does not fire, the collision counter does
  -- not tick, no "delete the stream key" WARN is emitted, and the Rust
  -- caller classifies the reply as itself: a store write refusal (`-OOM`,
  -- `-MISCONF`, `-READONLY`, `-NOREPLICAS`) as a nothing-applied
  -- `UNAVAILABLE`, anything else through the shared fred classifier —
  -- including back to a retriable `Aborted` for the transient members.
  local is_id_collision = string.find(xadd_err, "equal or smaller", 1, true) ~= nil
  local is_wrongtype = string.sub(xadd_err, 1, 9) == "WRONGTYPE"
  if not (is_id_collision or is_wrongtype) then
    return redis.error_reply(xadd_err)
  end
  -- Nothing has been written yet, so returning here fails the commit
  -- whole. The Rust caller maps the `STREAM_ID_COLLISION` tag to a
  -- non-retriable `ServiceError::Internal` — no retry can lower a stream
  -- top, so the caller must not spin on it (b.ph9).
  -- The attempted ID and the header sequence it was derived from ride in
  -- the message text: this script owns sequence allocation, so the Rust
  -- caller has no other way to name them in its WARN. The *observed*
  -- colliding ID is deliberately NOT read back here — the only way to get
  -- it is an `XRANGE`/`XINFO` that materializes a stream entry inside the
  -- Lua VM, and entries carry up to `LIVE_EDIT_MAX_LAYER_SIZE` of OMPF
  -- (see `stream_high_water_probe.lua`'s cost note). Paying a ~150 ms
  -- server stall per rejected write, on a condition that repeats until an
  -- operator intervenes, is not worth one diagnostic field; the slow
  -- path's probe reports the colliding ID for free.
  return redis.error_reply(
    "STREAM_ID_COLLISION: XADD " .. stream_id .. " (header seq was " ..
    string.format("%d", current_seq) .. ") rejected: " .. tostring(xadd_err))
end

-- ── 5a. Byte trim (see "Relay byte budget"): only when step 4r dropped a
--        generation. Right after the XADD succeeded, so it cannot be refused:
--        the key was just proven a stream, `XTRIM` carries `write` but not
--        `denyoom`, and the first-write gates were spent on the XADD (step 6's
--        proof). `MINID ~` trims whole stream nodes below the cut, head-only;
--        the cut is at most `new_seq`, so this commit's entry always stays. ──
if relay_trim_cut ~= nil then
  redis.call("XTRIM", stream_key, "MINID", "~", string.format("%d", relay_trim_cut) .. "-0")
end

-- ── 6. Apply pass. Re-walks the same mutation records the pre-pass
--       validated, consuming the stashed `stored_set_order` values.
--       From here to the end of the script no call can reject, and that
--       is a proof rather than an aspiration:
--         * input shape — decided by the pre-pass from ARGV alone;
--         * key types — probed by the pre-pass, and nothing else can run
--           inside this atomic window to change one;
--         * memory — `scriptVerifyOOM` only refuses a script's FIRST
--           deny-oom write, which the `XADD` above already absorbed;
--         * the header `HSET` — the header's type was proven by the
--           `HGETALL` in step 1, which precedes every write here; its
--           `relay_gens` value was computed in step 4r without a call.
--         * step 5a's `XTRIM MINID ~` — on the key the `XADD` just proved is
--           a stream, with a cut step 4r formatted as `<n>-0`; `XTRIM` is not
--           a deny-oom command, and the first-write gates are spent.
--         * the per-record `HDEL`s below (b.zvd) — the pre-pass's
--           `wrong_type(record_key, "hash")` already proved every record
--           key is a hash or absent, and `HDEL` frees memory rather than
--           allocating, so it cannot re-arm the deny-OOM gate the `XADD`
--           above already spent. `HDEL` on an absent key is a no-op `0`.
--         * the b.qpf `HGET` / `HSET` / `HDEL` on `le:{oid}:obj:ext` — the
--           pre-pass's `wrong_type(ext_index_key, "hash")` proved that key is
--           a hash or absent, and an absolute `HSET` (deliberately not an
--           `HINCRBY`, which errors on a non-integer field value) cannot be
--           refused on one.
--         * the step-7a `HSET` on the commit-token record `le:{oid}:obj:wtok`
--           and the commit release ring write on `le:{oid}:obj:wrel` (its
--           `HGET`, `HMGET`, `HDEL` and `HSET`) — steps 1b and 1e TYPE-probed
--           both keys as a hash or absent before any write, and none of those
--           commands can be refused on either (an `HDEL` frees memory).
--         * the ring's `PEXPIRE` after that `HSET` — `PEXPIRE` is `WRITE,
--           FAST`, not deny-oom (Valkey 8.0 `src/commands/pexpire.json`), the
--           key exists (the `HSET` just wrote it), and its TTL passed step
--           1b's `ENCODING` check as a positive integer of at most 12 digits,
--           which cannot overflow the expiry arithmetic.
--         * structural mode's writes — every key they `HSET`, `SADD`,
--           `SREM`, `ZADD` or `ZREM` was probed by step 4s's walk 2; a
--           children list is `DEL`'d before its `RPUSH`, so the `RPUSH`
--           lands on an absent key whatever the list key held, and is
--           issued in chunks so no `unpack` can overflow the Lua stack;
--           `DEL` is type-agnostic; and a "DF" / "DT" row, which walk 2
--           probes only when it carries a digest, is read with `HGET` only
--           after a `TYPE` says it is a hash.
--       So a partially-applied commit is unreachable.
--
--       The fast walk below runs only in fast mode (`num_fast_records`); the
--       structural walk after it visits `structural_records`, empty in fast
--       mode. ──────────────────────────────────────────────────────────────
local key_cursor = SHARED_KEYS + 1
local argv_cursor = records_argv_start

-- b.qpf/b.yda accumulators, shared by both record arms.
--
-- The apply pass below RECORDS per value row and folds NOTHING; the fold runs
-- in settle pass 6a, over the recordings. That split is b.yda and it is what
-- makes this arithmetic identical BY CONSTRUCTION to the txn backend's
-- `storage::layer::fold_ext_row_transitions` — see the "Per row: ORIGINAL
-- pre-image vs FINAL write" block in the contract at the top of this file.
--
-- Row recordings (settle pass 6a's input):
--   `ext_row_original[k]` the hash row `k` held BEFORE this commit — `""` when
--                         the row was absent or held an inline value. Written
--                         ONCE, on the row's first touch, so a later record for
--                         the same row cannot overwrite it with what an earlier
--                         record of this same commit just wrote.
--   `ext_row_final[k]`    the hash row `k` ENDS this commit holding (`""` when
--                         inline or deleted). Overwritten by every later record
--                         for `k`.
--   `ext_row_keys`        insertion-ordered row keys, so settle pass 6a is
--                         deterministic (`pairs()` order is not).
--
-- Folded totals (settle pass 6a's output, settle pass 6b's input):
--   `ext_delta[h]`      net change in the number of this object's rows holding
--                       `h`, folded across every row this commit touched.
--   `ext_referenced[h]` true when some row this commit touched ENDS holding
--                       `h` — the hashes whose pre-flight `+1` this commit's
--                       final state can account for.
--   `ext_held[h]`       true when some row's ORIGINAL pre-image and FINAL write
--                       are both `h` (b.yda). Row-local proof that the object
--                       held the hash before this commit, independent of what
--                       `le:{oid}:obj:ext` records.
--   `ext_touched`       insertion-ordered list of the distinct hashes named by
--                       `ext_delta`, so settle pass 6b is deterministic too.
--   `ext_ends[h]`       the number of touched rows whose FINAL write holds `h`
--                       — structural mode's `ends[h]` in settle pass 6b's
--                       floor. Fast mode computes it and never reads it.
local ext_row_original = {}
local ext_row_final    = {}
local ext_row_keys     = {}

local ext_delta      = {}
local ext_referenced = {}
local ext_held       = {}
local ext_touched    = {}
local ext_ends       = {}

local function ext_note(hash)
  if ext_delta[hash] == nil then
    ext_delta[hash] = 0
    ext_touched[#ext_touched + 1] = hash
  end
end

-- Record one value row's transition. `old_hash` is the row's `hash` column as
-- this record read it, immediately before the `HDEL` (or, for a structural
-- "DF" / "DT", the `DEL`) that clears it (nil / false when the row was absent
-- or held an inline value); `new_hash` is what this record writes (nil when
-- inline, or when the record deletes the row).
--
-- The FIRST record for a row is the only one whose `old_hash` is a genuine
-- pre-image: from the second record on, the `HGET` reads what an earlier record
-- of THIS SAME commit wrote. Keeping only the first is the whole point (b.yda).
local function ext_note_row(row_key, old_hash, new_hash)
  if ext_row_original[row_key] == nil then
    ext_row_original[row_key] = (old_hash and old_hash ~= "") and old_hash or ""
    ext_row_keys[#ext_row_keys + 1] = row_key
  end
  ext_row_final[row_key] = (new_hash and new_hash ~= "") and new_hash or ""
end

for i = 1, num_fast_records do
  local tag = ARGV[argv_cursor]
  argv_cursor = argv_cursor + 1

  if tag == "SF" then
    -- Set field. Reconstruct the HSET arg list from the value hash fields
    -- carried by ARGV, then append the pre-pass-computed `set_order` +
    -- `version` pairs. ZADD score = new_seq (matches the planner's
    -- `field.version = new_seq`).
    local field_key = KEYS[key_cursor]
    key_cursor = key_cursor + 1

    local num_value_hfields = tonumber(ARGV[argv_cursor])
    argv_cursor = argv_cursor + 1

    local hset_args = { field_key }
    -- b.qpf: the new value's hash column, if this record writes one. The
    -- encoder emits `hash` + `size` for an external value and `value` for an
    -- inline one (`storage::lua::value_hash_fields`), so the presence of a
    -- `hash` pair IS the external/inline discriminator.
    local new_hash = nil
    for _ = 1, num_value_hfields do
      local hf_name = ARGV[argv_cursor]
      argv_cursor = argv_cursor + 1
      local hf_value = ARGV[argv_cursor]
      argv_cursor = argv_cursor + 1
      if hf_name == "hash" then
        new_hash = hf_value
      end
      hset_args[#hset_args + 1] = hf_name
      hset_args[#hset_args + 1] = hf_value
    end

    local delta_set_order = tonumber(ARGV[argv_cursor])
    argv_cursor = argv_cursor + 1
    local zadd_member = ARGV[argv_cursor]
    argv_cursor = argv_cursor + 1

    hset_args[#hset_args + 1] = "set_order"
    -- b.qw6 item 2 (proptest follow-up): `string.format("%d", N)` instead
    -- of `tostring(N)` — preserves integer representation up to the
    -- 2^53 ceiling enforced in the pre-pass. `tostring` would silently
    -- produce scientific notation past ~10^14 even when arithmetic
    -- precision still holds.
    hset_args[#hset_args + 1] = string.format("%d", stored_set_orders[i])
    -- t3.msb.35.ey.9u: persist the writer's TRUE raw per-delta authoring
    -- rank alongside the cumulative `set_order`. `string.format("%d", ...)`
    -- matches the slow path's `to_string()` decimal byte-for-byte (and
    -- preserves integer representation up to the 2^53 ceiling above). Uses
    -- the raw `delta_set_order`, NOT the folded `stored_set_order`.
    hset_args[#hset_args + 1] = "authoring_set_order"
    hset_args[#hset_args + 1] = string.format("%d", delta_set_order)
    hset_args[#hset_args + 1] = "version"
    hset_args[#hset_args + 1] = new_seq_str

    -- b.zvd: a value row holds exactly ONE shape. `HSET` merges, so a field
    -- first written inline (`value`) and later rewritten as an external value
    -- (`hash` + `size`) kept the stale `value` column — and the reader
    -- (`storage::diff_read::decode_field_value`) checks `value` first, so the
    -- row decoded as the OLD inline bytes forever. Clearing all three columns
    -- unconditionally needs no knowledge here of which shape is about to be
    -- written and no ARGV protocol change; the delete set is a strict superset
    -- of what the `HSET` writes, so the ORDER is load-bearing. Mirrors the slow
    -- path's `storage::layer::VALUE_HFIELDS` `HDEL`.
    --
    -- b.qpf: read the superseded hash IMMEDIATELY BEFORE the `HDEL` that
    -- destroys it — this is the only moment it is still observable. The row
    -- holds a hash only when it has no `value` column (the decoder's rule, see
    -- the reference-accounting contract above): a legacy dual-shape row
    -- decodes as its inline bytes, so its stale `hash` column is not a
    -- reference and must not be released here. `HEXISTS` rather than `HMGET`
    -- so inline value bytes are never copied into the script. `HEXISTS` and
    -- `HGET` are read-only on a key the pre-pass proved is a hash-or-absent, so
    -- they do not weaken the post-XADD no-rejection proof above. This is a
    -- *rewrite* site: the row's hash column is unconditionally cleared and
    -- re-written, so a superseded external value is genuinely released here.
    -- (Fast mode has no tombstone-that-retains-its-hash case: it emits only
    -- SF / ST records, and field / time-sample deletion never reaches it —
    -- see the "Field / time-sample deletions never appear in fast mode"
    -- note in the contract block above. Structural mode's "DF" / "DT" `DEL`
    -- the row outright, so they retain nothing either.)
    local old_hash = false
    if redis.call("HEXISTS", field_key, "value") == 0 then
      old_hash = redis.call("HGET", field_key, "hash")
    end
    ext_note_row(field_key, old_hash, new_hash)
    redis.call("HDEL", field_key, "value", "hash", "size")
    redis.call("HSET", unpack(hset_args))
    redis.call("ZADD", fields_index_key, new_seq_str, zadd_member)

  else
    -- "ST" — set time sample. The pre-pass already rejected every other
    -- tag, so this branch is unconditionally the time-sample case.
    -- HSET value fields + computed `version`, ZADD ts index with
    -- score = new_seq.
    local ts_key = KEYS[key_cursor]
    key_cursor = key_cursor + 1

    local num_value_hfields = tonumber(ARGV[argv_cursor])
    argv_cursor = argv_cursor + 1

    local hset_args = { ts_key }
    -- b.qpf: same `hash`-pair discriminator as the "SF" arm — time samples
    -- share the value encoder.
    local new_hash = nil
    for _ = 1, num_value_hfields do
      local hf_name = ARGV[argv_cursor]
      argv_cursor = argv_cursor + 1
      local hf_value = ARGV[argv_cursor]
      argv_cursor = argv_cursor + 1
      if hf_name == "hash" then
        new_hash = hf_value
      end
      hset_args[#hset_args + 1] = hf_name
      hset_args[#hset_args + 1] = hf_value
    end

    local zadd_member = ARGV[argv_cursor]
    argv_cursor = argv_cursor + 1

    hset_args[#hset_args + 1] = "version"
    hset_args[#hset_args + 1] = new_seq_str

    -- b.zvd: same one-shape-per-row clear as the "SF" arm above. Time samples
    -- share the value encoder and decoder with fields, so they inherit the same
    -- stale-shape hazard.
    -- b.qpf: same pre-image read as the "SF" arm, by the same rule (no
    -- `value` column, then `hash`), for the same reason and in the same
    -- position — before the `HDEL` that clears the column. Also a rewrite
    -- site, never a retaining tombstone.
    local old_hash = false
    if redis.call("HEXISTS", ts_key, "value") == 0 then
      old_hash = redis.call("HGET", ts_key, "hash")
    end
    ext_note_row(ts_key, old_hash, new_hash)
    redis.call("HDEL", ts_key, "value", "hash", "size")
    redis.call("HSET", unpack(hset_args))
    redis.call("ZADD", ts_index_key, new_seq_str, zadd_member)
  end
end

-- ── 6s. Structural apply pass (structural mode only). Each record issues
--        the commands `execute_mutation` issues for the same mutation, with
--        the values the record carries; the "HDR" record is written by step 7.
--        Value rows record their external-value transition through the same
--        `ext_note_row` as fast mode, so step 6a folds both modes alike and
--        step 6b differs only in its `obj:ext` count rule. ─────────────────────

-- Children per `RPUSH`: far below the Lua C stack's `unpack` limit (~8000),
-- so a large list cannot make the `RPUSH` raise after the `XADD`.
local RPUSH_CHUNK = 1000

-- A deleted row's `hash` column, read before its `DEL`, or nil. The row is
-- probed only when its record carries a digest (as on the WATCH path, a `DEL`
-- needs no type), so `HGET` — which raises on a non-hash — is issued only
-- after `TYPE` says it is a hash. A non-hash row holds no external value, and
-- neither does a row with a `value` column (the counting rule above).
local function deleted_row_hash(key)
  if redis.call("TYPE", key)["ok"] == "hash"
    and redis.call("HEXISTS", key, "value") == 0 then
    return redis.call("HGET", key, "hash")
  end
  return nil
end

for _, r in ipairs(structural_records) do
  local tag = r.tag
  if tag == "SEC" then
    local c = r.columns
    redis.call(
      "HSET", r.key,
      SECTION_COLUMNS[1], c[1], SECTION_COLUMNS[2], c[2], SECTION_COLUMNS[3], c[3],
      SECTION_COLUMNS[4], c[4], SECTION_COLUMNS[5], c[5], SECTION_COLUMNS[6], c[6])
    redis.call("SADD", sections_index_key, r.sid)
  elseif tag == "DSEC" then
    redis.call("DEL", r.key)
    redis.call("SREM", sections_index_key, r.sid)
  elseif tag == "TOMB" then
    redis.call("HSET", r.key, "first_ver", r.first_ver, "last_ver", r.last_ver)
    redis.call("SADD", deleted_sections_index_key, r.sid)
  elseif tag == "CHL" then
    -- Rebuild: `DEL` then `RPUSH` in order, and the index follows the key's
    -- existence (b.jdr) — `SADD` when a list remains, `SREM` when the plan
    -- empties it, exactly as `execute_mutation`'s `SetChildren` arm.
    redis.call("DEL", r.key)
    local n = #r.children
    if n > 0 then
      for first = 1, n, RPUSH_CHUNK do
        redis.call("RPUSH", r.key, unpack(r.children, first, math.min(first + RPUSH_CHUNK - 1, n)))
      end
      redis.call("SADD", children_index_key, r.parent)
    else
      redis.call("SREM", children_index_key, r.parent)
    end
  elseif tag == "DCHL" then
    redis.call("DEL", r.key)
    redis.call("SREM", children_index_key, r.parent)
  elseif tag == "SFA" or tag == "STA" then
    -- Fast mode's value-row write — pre-image `HGET`, the b.zvd value-shape
    -- `HDEL`, then the `HSET` — with the record's absolute columns in
    -- `layer::field_value_hash_fields` / `time_sample_hash_fields` order; the
    -- index scored by the record's `version`.
    local hset_args = { r.key }
    for j = 1, #r.pairs do
      hset_args[#hset_args + 1] = r.pairs[j]
    end
    local index_key = ts_index_key
    if tag == "SFA" then
      hset_args[#hset_args + 1] = "set_order"
      hset_args[#hset_args + 1] = r.set_order
      hset_args[#hset_args + 1] = "authoring_set_order"
      hset_args[#hset_args + 1] = r.authoring_set_order
      index_key = fields_index_key
    end
    hset_args[#hset_args + 1] = "version"
    hset_args[#hset_args + 1] = r.version
    -- Pre-image by the counting rule: a row with a `value` column holds no
    -- hash.
    local old_hash = false
    if redis.call("HEXISTS", r.key, "value") == 0 then
      old_hash = redis.call("HGET", r.key, "hash")
    end
    ext_note_row(r.key, old_hash, r.new_hash)
    redis.call("HDEL", r.key, "value", "hash", "size")
    redis.call("HSET", unpack(hset_args))
    redis.call("ZADD", index_key, r.version, r.member)
  elseif tag == "DF" or tag == "DT" then
    -- The row's pre-image hash, then the `DEL`: the row ends holding no
    -- hash, so settle pass 6b owes back a hash this drops to zero as
    -- "rewrite".
    ext_note_row(r.key, deleted_row_hash(r.key), nil)
    redis.call("DEL", r.key)
    local index_key = ts_index_key
    if tag == "DF" then
      index_key = fields_index_key
    end
    redis.call("ZREM", index_key, r.member)
  end
end

-- ── 6a. Fold the recorded rows into the per-hash totals (b.yda).
--
--       ONE fold per touched ROW, from its ORIGINAL pre-image to its FINAL
--       write — never one per record. This is a statement-for-statement mirror
--       of `storage::layer::fold_ext_row_transitions`, which folds the same two
--       values per row after collapsing duplicate row keys to the last write.
--
--       Equivalence of the `ext_delta` arithmetic to the pre-b.yda per-record
--       fold (which read `old_hash` live and so self-corrected): for a row
--       written once the two are trivially the same, and for a row written
--       R times the per-record fold contributes
--       `-h0 +h1, -h1 +h2, ..., -h(R-1) +hR`, whose intermediate terms cancel
--       pairwise — because record i+1's live `HGET` reads exactly what record i
--       wrote — leaving `-h0 +hR`, which is this loop's single step. (An
--       intermediate inline write contributes no term on either side and
--       cancels the same way.) So `ext_delta` is unchanged by the restructure.
--       `ext_referenced` and `ext_held` are NOT, deliberately: see the
--       contract block at the top of this file.
--
--       `ext_ends` counts the rows per final hash, for structural mode's
--       floor in 6b; `fold_ext_row_transitions` has no twin of it, because
--       the txn backend has no structural mode. ───────────────────────────
for _, row_key in ipairs(ext_row_keys) do
  local original = ext_row_original[row_key]
  local final    = ext_row_final[row_key]
  if original ~= "" then
    ext_note(original)
    ext_delta[original] = ext_delta[original] - 1
  end
  if final ~= "" then
    ext_note(final)
    ext_delta[final] = ext_delta[final] + 1
    ext_referenced[final] = true
    ext_ends[final] = (ext_ends[final] or 0) + 1
  end
  if original ~= "" and original == final then
    ext_held[original] = true
  end
end

-- ── 6b. Settle the external-value reference index (b.qpf). Runs after the
--       whole apply pass so a hash that left one row and arrived at another
--       inside the SAME commit nets out to zero change and owes nothing.
--
--       Absolute `HSET` / `HDEL`, never `HINCRBY` — see the b.qpf block in
--       the contract at the top of this file. Neither can reject on a key the
--       pre-pass proved is a hash, so the post-XADD no-rejection proof holds.
--
--       `tonumber(...) or 0` rather than a validity check: a non-numeric
--       field value is out-of-band corruption, and treating it as 0 rebuilds
--       the field from this commit's own evidence instead of aborting a
--       commit whose stream entry is already published.
--
--       The two modes compute `after` and its write differently (see
--       "External-value reference accounting" in the contract) and share the
--       release arms:
--         * fast — `after = before + delta`, written only when `delta ~= 0`;
--         * structural — the raise-only floor
--           `after = max(before + delta, untouched + ends)`, written whenever
--           the stored field is not exactly `after` (`HDEL` at 0, if the field
--           exists). `untouched` is the block's count, 0 for a hash the block
--           does not carry: that happens only when the cache was wrong about a
--           touched row, and the floor keeps it safe, since it can only
--           raise. On a drift-free index with a correct cache the floor equals
--           `before + delta`, and the writes are fast mode's. ─────────────────
local releases = {}
-- Structural mode only: the touched hashes whose `before + delta` fell below
-- (under) or above (over) the recount `untouched + ends` — drift the floor
-- raised, or kept for the reconciler's audit to lower. Reported in the reply
-- (step 8); the caller counts them on
-- `live_edit_ext_index_drift_total{site="structural_settle"}`.
local settle_drift_under = 0
local settle_drift_over = 0
for _, hash in ipairs(ext_touched) do
  local delta = ext_delta[hash]
  local stored = redis.call("HGET", ext_index_key, hash)
  local before = tonumber(stored) or 0
  local after
  if mode == MODE_STRUCTURAL then
    local recount = (ext_untouched[hash] or 0) + (ext_ends[hash] or 0)
    if before + delta < recount then
      settle_drift_under = settle_drift_under + 1
    elseif before + delta > recount then
      settle_drift_over = settle_drift_over + 1
    end
    after = math.max(before + delta, recount)
    if after > 0 then
      local after_str = string.format("%d", after)
      if stored ~= after_str then
        redis.call("HSET", ext_index_key, hash, after_str)
      end
    elseif stored then
      redis.call("HDEL", ext_index_key, hash)
    end
  else
    after = before + delta
    if delta ~= 0 then
      if after > 0 then
        redis.call("HSET", ext_index_key, hash, string.format("%d", after))
      else
        redis.call("HDEL", ext_index_key, hash)
      end
    end
  end
  if ext_referenced[hash] then
    -- Some row this commit touched ENDS holding this hash, so `after >= 1` on
    -- an index that matches the rows (structural mode's floor guarantees it
    -- outright, `ends >= 1`), and the row-count arm below does not also fire
    -- for it. `before > 0` means the object
    -- ALREADY referenced it, so the caller's pre-flight `+1` bought nothing.
    -- `ext_held[hash]` is the second, index-independent proof of the same thing
    -- (b.yda) — a row whose ORIGINAL pre-image and FINAL write are both this
    -- hash, which is why it is computed in settle pass 6a and not from
    -- any single record. Note fast mode's seeding arm above is deliberately
    -- NOT extended to match: seeding an absolute count from a commit that saw
    -- only some of the object's rows would under-count, which is the
    -- count-clamp hazard `KeySchema::obj_ext_index`'s `# Drift` section
    -- refuses. Structural mode's floor is not that clamp: it counts every row
    -- of the object — the untouched rows from the cache, the touched rows
    -- from this commit's records — and only ever raises the stored count.
    if before > 0 or ext_held[hash] then
      releases[#releases + 1] = "surplus_acquire:" .. hash
    end
  elseif before > 0 and after <= 0 then
    -- The object's last row holding this hash gave it up.
    releases[#releases + 1] = "rewrite:" .. hash
  end
end

-- ── 7. HSET the object header. In fast mode, writes every field that can
--       change on the fast path: `first_ver` and `src_fmt` are preserved in
--       place (only `apply_reset`, never fast-path, modifies them), and
--       `next_set_order` carries the value the pre-pass folded. In structural
--       mode, writes `execute_mutation`'s header columns in its order: `seq`
--       (the "HDR" record's, which step 4s proved is `new_seq`), the record's
--       `first_ver` and post-apply `next_set_order` verbatim, the same
--       timestamps and `last_client_*`, and `src_fmt` when the record carries
--       one. Both modes advance `commit_chain` by this commit's fold, with
--       one line (see "commit_chain" above), and write step 4r's
--       `relay_gens` when the byte budget is on (see "Relay byte budget").
--       `redis.sha1hex` is pure and cannot fail, so computing it here keeps
--       the post-XADD no-rejection proof intact.
local commit_chain = redis.sha1hex((hdr["commit_chain"] or "") .. "\n" .. payload)
if mode == MODE_FAST then
  local header_args = {
    header_key,
    "seq",             new_seq_str,
    -- b.qw6 item 2 (proptest follow-up): `string.format("%d", N)` keeps
    -- the stored `next_set_order` as an exact decimal integer all the
    -- way to the 2^53 ceiling. `tostring` would have switched to
    -- scientific notation past ~10^14 — visible regression on the b.qw6
    -- parity proptest because the slow path stores u64 decimal verbatim.
    "next_set_order",  string.format("%d", next_set_order),
    "modified_at",     server_ts,
    "last_client_id",  client_id,
    "last_client_ts",  client_ts,
    "last_server_ts",  server_ts,
    "commit_chain",    commit_chain,
  }
  if relay_gens_value ~= nil then
    header_args[#header_args + 1] = "relay_gens"
    header_args[#header_args + 1] = relay_gens_value
  end
  redis.call("HSET", unpack(header_args))
else
  local header_args = {
    header_key,
    "seq",             new_seq_str,
    "first_ver",       structural_hdr.first_ver,
    "next_set_order",  structural_hdr.next_set_order,
    "modified_at",     server_ts,
    "last_client_id",  client_id,
    "last_client_ts",  client_ts,
    "last_server_ts",  server_ts,
    "commit_chain",    commit_chain,
  }
  if structural_hdr.src_fmt ~= nil then
    header_args[#header_args + 1] = "src_fmt"
    header_args[#header_args + 1] = structural_hdr.src_fmt
  end
  if relay_gens_value ~= nil then
    header_args[#header_args + 1] = "relay_gens"
    header_args[#header_args + 1] = relay_gens_value
  end
  redis.call("HSET", unpack(header_args))
end

-- ── 7a. Record this call's commit (see "Commit token" above). Same atomic
--       unit as the commit it describes, so a later attempt of this call
--       (after a lost reply), or a client re-send carrying the same operation
--       id, finds it in step 1b instead of applying the delta again. The
--       value `<match_key>:<seq>:<server_ts><suffix>` is shared byte-for-byte
--       with the txn backend's MULTI. One write for both modes. Un-rejectable:
--       see step 6's proof.
redis.call(
  "HSET", wtok_key,
  client_id, commit_key .. ":" .. new_seq_str .. ":" .. server_ts .. record_suffix
)
-- The commit release ring write ("Commit release ring" above): only when this
-- commit owes releases. The transcription of `storage::layer::
-- plan_release_ring_write`, which the WATCH path's `MULTI` runs; edit the two
-- together. Un-rejectable for the same reason: step 1e proved the key a hash
-- or absent, and only `HGET` / `HMGET` / `HDEL` / `HSET` touch it, then the
-- `PEXPIRE` that sets (or refreshes) its TTL to ARGV[16] (see step 6's proof).
if #releases > 0 then
  -- The next slot: `|n|` when it is one to three digits below RING, else 0.
  local next_raw = redis.call("HGET", wrel_key, "|n|")
  local start = 0
  if next_raw and string.match(next_raw, "^%d%d?%d?$") then
    local parsed = tonumber(next_raw)
    if parsed < COMMIT_RELEASE_RING_SLOTS then
      start = parsed
    end
  end
  -- u = min(RING, ceil(R / M)) slots, from `start`, wrapping.
  local slots = math.min(
    COMMIT_RELEASE_RING_SLOTS,
    math.ceil(#releases / COMMIT_RELEASE_RING_SLOT_RELEASES))
  local slot_fields = {}
  for i = 0, slots - 1 do
    slot_fields[i + 1] = "|s|" .. string.format("%d", (start + i) % COMMIT_RELEASE_RING_SLOTS)
  end
  -- Evict the entry of every other token whose start slot this write takes;
  -- an entry it only covers (`""`) has already been evicted with its start.
  local old = redis.call("HMGET", wrel_key, unpack(slot_fields))
  local evicted = {}
  for i = 1, slots do
    local token = old[i]
    if token and token ~= "" and token ~= call_token then
      evicted[#evicted + 1] = "|t|" .. token
    end
  end
  if #evicted > 0 then
    redis.call("HDEL", wrel_key, unpack(evicted))
  end
  local ring_args = {
    wrel_key,
    "|t|" .. call_token, new_seq_str .. " " .. table.concat(releases, " "),
    slot_fields[1],      call_token,
  }
  for i = 2, slots do
    ring_args[#ring_args + 1] = slot_fields[i]
    ring_args[#ring_args + 1] = ""
  end
  ring_args[#ring_args + 1] = "|n|"
  ring_args[#ring_args + 1] = string.format("%d", (start + slots) % COMMIT_RELEASE_RING_SLOTS)
  redis.call("HSET", unpack(ring_args))
  redis.call("PEXPIRE", wrel_key, ring_ttl)
end

-- ── 8. Return `{new_seq, "<reason>:<hash_hex>"...}` as bulk strings. The
--       Rust caller decodes `Vec<String>`, parses element 1 as u64, and SADDs
--       the owed decrements onto `le:ext:{gc}:pending` — which lives on
--       another slot and therefore cannot be written from here (b.qpf).
--       Structural mode puts `"drift:<under>:<over>"` (the settle's counts,
--       step 6b) at element 2, ahead of the releases; fast mode's reply has
--       no such element. ──────────────────────────────────────────────────────
local reply = { new_seq_str }
if mode == MODE_STRUCTURAL then
  reply[2] = "drift:" .. settle_drift_under .. ":" .. settle_drift_over
end
for _, r in ipairs(releases) do
  reply[#reply + 1] = r
end
return reply
