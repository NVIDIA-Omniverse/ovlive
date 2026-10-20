// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// b.1qo — a command, pipeline or fred `Transaction` handed to fred is never
// written after its caller gave up (timeout, error or a dropped future), and a
// batch cut after its `MULTI` leaves no `MULTI` open (the vendored fred's
// change 9). Driven through a gated proxy that cuts a connection and refuses
// its reconnects, or pauses it after a `MULTI`; no feature gate beyond
// `test-support`.
mod abandoned_write_test;
// b.1jt — a store shared with an older build across the GC protocol marker:
// the marked leader lock and its takeovers, an older build's drain, sweep,
// carrier claim and lock CAS (vendored from 0.2.0-rc.47 under
// `tests/fixtures/legacy_gc_rc47/`) run against this build's claims, carriers
// and enqueues, un-aged stamping, foreign-claim drops, in-place re-claims,
// the enqueue's replay refusal, the ledger after an older leader, and the
// source guard on who writes `pending` and `inflight`. Its claim-planting
// helpers are imported by `b_kf1_tests`, `b_j9j_tests`, `b_yuo_gc_tests` and
// `external_value_test`.
mod b_1jt_rollout_tests;
// b.9fn — the root children-ordering list `le:{oid}:obj:ch:0` leaked on every
// object retirement (parent id 0 is in neither `obj:secs` nor `obj:dels`, so
// the index-walking UNLINK enumeration never produced it). Covers both
// `cleanup_old_object` entry points; the orphan-sweep route is covered in
// `orphan_sweep_test.rs`.
mod b_9fn_tests;
// b.459 — the WATCH/MULTI/EXEC commit path must refuse a wrong-typed mutated
// key BEFORE it opens the `MULTI`. A `WRONGTYPE` raised inside `EXEC` neither
// aborts the transaction nor rolls back its siblings, so pre-fix an out-of-band
// string on one field hash half-committed a structural UpdateLayer — the `XADD`
// published a commit whose `HSET` had been refused. Every test plants a real
// string on a real key and drives a real commit; the per-key-family coverage,
// the two deliberate exclusions (`DEL` is type-agnostic; the children list is
// `DEL`'d before its `RPUSH`), backend parity and the reachability boundary are
// tabulated in the module header. No feature gate: the probe is production
// code, not a seam. The per-variant key inventory is pinned as unit tests beside
// `mutation_probe_keys` in `src/storage/layer.rs`, and the source-shape fence
// keeping it in step with `execute_mutation` lives in `mapper_routing_test.rs`.
mod b_459_tests;
// b.62m — a transient Valkey failure on a unary RPC's round trip answered a
// hand-wrapped `INTERNAL` (and logged at ERROR) instead of a retriable class.
// The four sites reached first on their path — DeleteSession's first read,
// ListSessions' names read, `get_session_meta`, `load_values` — are each driven
// by a real data-pool stall and must answer `Unavailable`, counting exactly one
// data-pool command timeout. The inner `undo` module drives CreateSession's
// three post-commit failures (the metadata `HSET`, the in-doubt `SISMEMBER`,
// and the in-doubt re-scan `SMEMBERS`, reached by letting the `SISMEMBER`
// through a single stalled connection). The call retries only once the undo
// has released the attempt's names member, otherwise it ends (see
// `undo_create_attempt` / SDD §7.10). Each is driven to both undo outcomes, and
// the failed-undo cases pin the not-released WARN; the metadata `HSET` is also
// driven through a partial undo (names `SREM` let through alone, told apart by
// its wire size). Two more pin the per-attempt `session_id`: the partial undo's
// timed-out commands landing after `Ok`, and a real in-doubt `EXEC` landing
// after its `SISMEMBER` read "absent", must each leave the call's result
// intact. The module needs
// `--features test-fault-injection` (the pre-dispatch hook and the verdict
// override) and is silently compiled out without it. The converted sites this
// file does not drive are guarded per site by `mapper_routing_test.rs`, which
// also pins the three undo call sites.
mod b_62m_tests;
// b.75i — one `UpdateLayer` call commits at most once. Each call mints one
// random commit token that its commit records in `le:{oid}:obj:wtok` and that
// every attempt checks first, so a commit whose reply was lost (a fast-path
// `EVALSHA` timeout, a slow-path in-doubt `EXEC` whose reconcile read failed)
// is reported by the retry instead of applied a second time — including when
// another writer lands in between and after an overwrite set `superseded=1`.
// Two seam-free tests pin that byte-identical calls still commit twice and
// that a wrong-typed record is refused identically on both backends. The
// lost-reply scenarios live in an inner
// `#[cfg(feature = "test-fault-injection")] mod lost_reply` and are compiled
// out without that feature. The record's byte parity across backends is in
// `update_layer_lua_test`, its cleanup in `b_9fn_tests` / `orphan_sweep_test`
// / `checkpoint::residue`.
mod b_75i_tests;
// b.v6d — a client re-send carrying the same `operation_id` is exactly-once
// across calls. `UpdateLayer`: a re-send of a landed operation is answered
// `ALREADY_COMMITTED` with the original commit on both backends (a lost-reply
// original included), its pre-flight external-value increment is handed back,
// a different request under the same id is `ALREADY_EXISTS`, the id is scoped
// to (object, client_id), concurrent duplicates commit once, and a torn
// slow-path commit retracts its record. `CreateLayer`: a re-send allocates
// nothing and answers the recorded `object_id`, including a duplicate that
// loses the race at the commit script. Revision 3: the start-record guard
// refuses, as overtaken, an attempt whose writer has committed since the call
// started (a late re-send, a parked retry); a failed start-record read ends
// the call before any attempt; and the commit `EVALSHA` is written to the wire
// once, so fred never replays a dropped caller's commit after a reconnect.
// Amendment R4-1: an overwrite whose commit script failed reads the mapping,
// its commit claim and (with an operation id) its operation record, and tries
// the commit once more while no run landed and the mapping still names the
// old object. The
// seam-driven scenarios live in inner
// `#[cfg(feature = "test-fault-injection")]` modules (`lost_reply`,
// `overwrite_second_try`).
mod b_v6d_tests;
// b.v6d revision 4 — an `-ASK` a one-attempt command cannot follow leaves the
// pooled client's fred router running (the vendored fred patch), and a router
// that ends any other way is restarted by `storage::router_supervisor`. The
// `-ASK` cases run against a clustered facade over `ScriptReplyProxy`, which
// also delivers the real `-ASK` / `-MOVED` an overwrite's commit meets before
// its Amendment R4-1 second try.
mod b_v6d_router_test;
// b.ua9 — a `CreateLayer` overwrite's post-commit failures converge without a
// client retry: the commit and veto scripts and their claim, the wake script,
// the overwrite tail's retried wake-up and out-of-memory fallback, the
// in-process finisher, the orphan sweep's three arms (completing a stranded
// overwrite, retiring an overwrite-fenced orphan, vetoing then retiring any
// other orphan) with their grace, the re-plan of a moved mapping, and a landed
// commit answered as committed whatever the mapping holds now. Its
// `sweep_one_pass` is shared with `grpc::b_ua9_subscriber_test`. The
// seam-driven scenarios live in an inner
// `#[cfg(feature = "test-fault-injection")] mod seams`.
pub(crate) mod b_ua9_tests;
mod b_vn4_ask_follow_test;
mod b_vn4_tryagain_test;
// b.cyu — a fred `Transaction` a cluster node redirects ("Max redirections
// exceeded") is transient at every txn-pool call site (`DeleteSession`, both
// redirect shapes; the `CreateLayer` materialization, including its
// lost-reply replay; the session checkpoint applier), staged through
// `ScriptReplyProxy` in front of the standalone store.
mod b_cyu_txn_redirect_test;
// b.cyu — the migration sentinel's runtime fence on the two GC ledger
// scripts and the external-value index recount (the commit script's is in
// `update_layer_lua_test`), and the write-first order of the three decision-9
// scripts, read off `MONITOR`.
mod b_cyu_script_sentinel_test;
// b.cyu — the same rule on a real cluster: each sentinel script refused before
// any write while its slot migrates, each decision-9 script landing whole,
// `DeleteSession` / `CreateLayer` answering `UNAVAILABLE` and the txn-pool
// routing re-sync. Every cluster test is `#[ignore]`d and gated on
// `LIVE_EDIT_TEST_CLUSTER_URL`; CI's `test-cluster-disruptive` job selects
// them by `slot_migration_disruptive`. The module also holds standalone checks
// of the two commit regressions' staging premise, which run in the ordinary
// suite. Compiled only with `test-fault-injection`, the feature the shared
// `SlotMigration` helper is built under.
#[cfg(feature = "test-fault-injection")]
mod b_cyu_slot_migration_test;
// The shared Lua-source scanners (`compiled_scripts`, `lua_code`, the command
// scan and the pinned-hash normalization) behind `b_1jt_rollout_tests`,
// `ext_blob_reader_guard_test` and `script_migration_guard_test`.
pub(crate) mod lua_source_scan;
// b.cyu — the guard behind SDD §18.1's rule that no writing script that can
// tear runs while its slot migrates: the script inventory, the closed-world
// command scan, the sentinel and allowlist (with pinned hashes), the
// sentinel's wiring, and the SDD's naming of every script. Pinned in
// scripts/guard-tests.sh `MODULES`. Its Rust-source helpers are shared with
// `txn_exec_routing_guard_test`.
mod script_migration_guard_test;
// b.cyu — the guard that every fred `Transaction`'s `EXEC` goes through
// `ValkeyClient::exec_txn`, which re-syncs the checkout's routing after a
// redirect: no raw `exec` outside it (a `WatchedTxn`'s is told apart), every
// `.multi()` site calls it, and each call passes the checkout its transaction
// was opened on. Pinned in scripts/guard-tests.sh `MODULES`.
mod txn_exec_routing_guard_test;
// b.atj — the read-side twin of the keyspace type-error counter. b.zvd made a
// wrong-typed key on any of the snapshot reads fail every snapshot
// consumer closed, but nothing ticked: `keyspace_type_error_total` is written
// only by the two commit-side pre-MULTI probes, so an operator alerting on it
// saw a flat line through the wedge. The three positive stagings (the fenced
// read's own wrapper on a permanent refusal — one attempt, one tick, no retry —
// the structural-commit funnel, and the checkpoint assembler's header read) each
// plant one string and assert exactly one `key_family` cell of
// `live_edit_snapshot_read_type_error_total` moved and the other ten did not;
// the negative controls and the two non-WRONGTYPE arms assert none moved. The
// per-family labels are staged on the atomic read and the unfenced fan-out (the
// fenced pass runs the same fan-out), in `snapshot_read_atomic_test` and
// `snapshot_read_unfenced_test`. The tick itself needs no feature gate — it is production code, not a seam — but the
// two non-WRONGTYPE gate arms live in an inner
// `#[cfg(feature = "test-fault-injection")] mod non_wrongtype_snapshot_read_errors`
// (only an injected error can be classified-but-not-`WRONGTYPE`) and are silently
// compiled out without `--features test-fault-injection`. The refusal contract
// itself belongs to `b_zvd_tests` / `b_459_tests`, the enum↔label-set agreement
// to a unit test in `src/storage/diff_read.rs`, and the zero-seeded bounded
// label set to `observability_metrics_test.rs`.
mod b_atj_tests;
// b.muu — the three holes b.atj left. (1) A snapshot read refused PERMANENTLY by
// anything other than a wrong-typed key moved nothing at all, so a standing
// backing-store failure of every consumer of an object's state was invisible on
// every counter; the fix is a sibling counter,
// `live_edit_snapshot_read_refused_total{kind}`, ticked from the arm `WRONGTYPE`
// does not claim. (2) The checkpoint assembler's four direct `obj:hdr` /
// `obj:dels` / `obj:del` reads bypassed the counting helper. (3) The orphan
// sweep's two `obj:hdr` touches did too — the worst of the three, because the
// sweep is a background task whose only other evidence is a log line, and its
// discovery read `?`-propagates, so one bad header suspends orphan reaping for
// the whole keyspace. Together (2) and (3) are why `key_family="header"` was
// race-only on every consumer. The per-`kind` arms need
// `--features test-fault-injection` (no keyspace can hold a `-NOAUTH` or a
// `-MISCONF` still) and are silently compiled out without it; the sweep and
// assembler arms plant real strings on real keys and need no feature. The
// `key_family` fixture, the family asserters and the injector staging helper are
// imported from `b_atj_tests` rather than copied, and that file imports this
// one's two refusal-counter readers back — one definition each. The kind
// vocabulary's own consistency is unit tested in `src/storage/diff_read.rs`, the
// twelve cells' zero-seeding in `observability_metrics_test.rs`.
mod b_muu_tests;
// b.ot3 — the persistence tier of the bounded structural planner. Small legal
// deltas used to retain N(N+1)/2 child ids and execute N DEL/RPUSH rebuilds per
// parent; children writes are now coalesced into one per dirty parent, emitted
// at the close of the structural phase, and the cascade's field/time-sample
// purge is one deferred pass. Most tests here drive the same command sequence
// twice — once as one commit, once as one commit per command — and compare the
// whole persisted surface (`obj:sec:*`, the ordered `obj:ch:*` lists,
// `obj:chs`, fields, time samples, tombstones, `obj:ext`), because one command
// per commit is one command per plan and therefore the pre-b.ot3 answer. Two
// tests pin children KEYS that must not survive: a reset's (the synthetic
// root) and a cascade's (a ghost parent's, under which no commit writes a list
// since b.a4k). The counters, the emission ordering and the cache-parity
// shapes live inline in `src/domain/delta.rs`; the duplicate-name tie-break's
// property coverage is in `tests/proptests.rs`. The external-value readers are
// imported from `b_qpf_tests` rather than copied.
mod b_ot3_tests;
// b.a4k — no children list under a parent with no row (SDD §10.4). Each of the
// ticket's replays (an orphan adopted when its parent is born, in the same
// delta as the commands that build the new list; an orphan moved away,
// deleted, reset over or joined by a second orphan before its parent is born;
// a create-then-delete of the parent; a cascade from an ancestor) runs through
// `update_layer` on both backends, and after every commit the fresh atomic
// snapshot, the structural fan-out, the in-memory reference state and the
// faithful snapshot's sibling order agree on every parent's child order. Also
// the lists earlier builds stored under a then-missing parent: a commit
// creating that parent deletes them, on marked and unmarked objects.
mod b_a4k_tests;
// b.289 — cluster-mode follow-ups from b.ji8. Sub-task 4: the state-proof
// recycle streak and the repair-ineffective breaker it feeds are now keyed by
// cluster node, so a Clean checkout on one primary can no longer re-arm the
// breaker on another — the gap that let a single misbehaving primary recycle
// without bound. Sub-task 1: the keyless-command audit's two source-shape
// fences (no zero-key EVALSHA, no bare SCAN outside an `is_clustered`
// dispatch) plus a counting fence over the `.multi()` inventory that
// `watch_guard.rs`'s header and SDD §15.3 state in prose. The fences and the
// single-endpoint bucket tests run everywhere; the per-node latching test
// needs ≥2 primaries and self-gates on `LIVE_EDIT_TEST_CLUSTER_URL`, printing
// a SKIP line when it is unset. Everything that drives a verdict also needs
// `test-fault-injection` (the probe seam) and is gated accordingly — that
// second condition is the silent one: without the feature those tests are
// compiled out rather than red.
mod b_289_tests;
// b.bhb — a watched transaction must commit only on the connection holding its
// WATCH. fred replayed a `Transaction` onto a fresh connection after a
// reconnect (and a fresh node after a MOVED), where the EXEC ran unwatched:
// CreateSession silently lost its duplicate-name check and UpdateLayer tore
// between materialized state and the live stream. Every test of the three
// defence layers stages a real `CLIENT KILL ID` on the exact socket carrying
// the WATCH rather than a synthetic error; the in-doubt reconciliation block
// additionally substitutes the EXEC verdict, because a lost reply needs the
// socket to die with the pipeline in flight and no seam can hold a call site
// there (the module's own docs carry the full reasoning). Either way the module
// needs `test-fault-injection` for the hooks and the override seam. A third
// staging mechanism covers the store that simply refuses `CLIENT ID` — a real
// Valkey ACL user denied the `client` command family, proving both identity
// layers disarm rather than fail closed. Three of its tests additionally
// self-gate on `LIVE_EDIT_TEST_CLUSTER_URL`, and the ACL pair self-gates on the
// store supporting ACL users at all; each prints a SKIP line when unavailable.
#[cfg(feature = "test-fault-injection")]
mod b_bhb_tests;
mod b_j9j_tests;
// b.ji8 — `WatchGuard` teardown routing. UNWATCH carries no key, so fred sent
// it to an arbitrary cluster node and the WATCH survived on the node holding
// it; the next borrower of the pooled txn connection paid for the residue.
// Two of the tests self-gate on `LIVE_EDIT_TEST_CLUSTER_URL` (a single
// connection cannot reproduce a routing defect) and print a SKIP line when it
// is unset; the rest — including the source-shape fence — run everywhere.
mod b_ji8_tests;
mod b_k1r_tests;
mod b_kf1_tests;
// b.nvf — per-object `obj:ext` counts drifting from the rows. Two producers
// pinned at the commit sites: the pre-image read of a rewritten row must count
// a dual-shape row (an inline `value` beside a stale `hash`) as holding no hash,
// on both backends and both row kinds; and a torn txn `EXEC` is followed by a
// recount of its object, staged with the b.bhb kill-and-collide mechanism and
// therefore gated on `test-fault-injection` inside the module. The reconciler's
// per-object audit, its drain gate and the recount script are tested in-crate
// (`src/storage/ext_reconcile.rs`, `src/storage/ext_index_audit.rs`), where
// their private entry points are reachable. Helpers are imported from
// `b_qpf_tests` and `b_bhb_tests` rather than copied.
mod b_nvf_tests;
// b.nxz — the txn backend's external-value pre-image read
// (`read_ext_pre_image`) used to queue every written row's `HEXISTS` + `HGET`
// in one data-pool pipeline; it now sends bounded chunks, a bounded number in
// flight. One commit whose rows span two waves, with every pre-image kind and
// external rows across every chunk boundary, asserts the bound through a
// test-only pipeline observer and the committed `obj:ext` and releases. The
// observer is gated on `test-fault-injection`, so the module is too. The pure
// plan and per-chunk decoder are unit tests in `src/storage/layer.rs`; helpers
// are imported from `b_qpf_tests`. b.26e Stage 2 adds the count phase's tests:
// one rewrite on a 20,000-hash index reads only its own hashes' counts, and
// the count phase's waves, one-hash decode and corrupt-count handling.
#[cfg(feature = "test-fault-injection")]
mod b_nxz_tests;
// b.qpf — an external value's `ext_meta` `ref_count` was released only at
// object retirement, and then only for the object's CURRENT values, so every
// superseded external value leaked its reference permanently. The fix counts
// `ref_count` as "live objects referencing this hash", tracks per-object row
// counts in `le:{object_id}:obj:ext` inside each commit's atomic unit on both
// backends, and gives every pending-set member the identity of the event that
// owes it (`{hash_hex}:{object_id}:{event}`) so two independently-owed
// decrements can no longer collapse into one Set member. Every behavioural arm
// a fast-path delta can reach runs on both backends; the two that are
// structurally single-backend say so in their own doc comments. Two
// cluster-gated siblings self-gate on `LIVE_EDIT_TEST_CLUSTER_URL` and print a
// SKIP line when it is unset. One arm needs `test-fault-injection` for the
// b.bhb watch-window hook and is gated accordingly.
mod b_qpf_tests;
mod b_qw6_tests;
// b.rej — the external-value pre-flight store moved off the txn pool: the
// per-hash `MULTI { SET ext_data; HINCRBY ext_meta ref_count 1 } EXEC` is now
// one `EVALSHA` of `scripts/ext_store_value.lua` on the multiplexed data pool,
// and the increment gained a server-side `last_store_ms` stamp (the timestamp
// `storage::ext_reconcile` needs to tell a standing surplus from an upload whose
// increment has not reached its commit yet). Every arm drives the real
// `ValkeyExternalValueStore` and checks the "same semantics, different
// transport" bill line by line: the count, the binary payload, the stamp's shape
// and value, the GC boundary (a collect takes the stamp, a decrement never
// writes it), and — the regression test for the move itself — that a store
// completes while another task holds every txn-pool slot. Two arms pin the
// unverified set-if-absent data write (b.682): a `store_values` over an
// existing blob, or over a wrong-typed data key, leaves it standing and still
// pays its one increment and stamp. Four arms pin the verified heal: an
// `enforce` upload replaces a blob of the wrong length (shorter or longer) or a
// wrong-typed data key, an
// `observe` upload replaces neither, and a same-length wrong blob is never
// replaced. One arm deliberately
// PINS a defect rather than fixing it: a wrong-typed meta key leaves the data
// write standing (b.5yz). The reader side of the stamp lives in
// `src/storage/ext_reconcile.rs`'s in-crate module, the `ref_count` semantics in
// `b_qpf_tests` / `b_yda_tests`, and the two source-shape fences the new script
// had to satisfy in `b_289_tests`.
mod b_rej_tests;
// b.yda W1 — the `held` rule. b.qpf's `surplus_acquire` release fired on one
// proof only (`before > 0`, read from `le:{oid}:obj:ext`), which an object
// materialized before b.qpf structurally cannot satisfy: it carries no index, so
// `before` reads 0 for a hash it genuinely holds, AND a same-hash rewrite nets
// `delta == 0` so the commit never seeds the index either. Such an object leaked
// its pre-flight `+1` on every commit for its whole life (residual (c)). b.yda
// adds the second, row-local proof — a row whose ORIGINAL pre-image equals its
// FINAL write — and that pairing is the whole subtlety: a per-RECORD fold reads
// the first record's write back as the second record's pre-image and hands back
// an increment that bought a brand-new reference, which is a blob deleted under
// a live row. Every arm runs on both commit backends (two transcriptions, one
// shared `ref_count`) and asserts the pending member set and `obj:ext` as whole
// values so the two answers are comparable literally. Two arms deliberately PIN
// a documented leak rather than fixing it; see the module header.
mod b_yda_tests;
// b.yuo — the external-value release ledger, the age-gated GC loop and the
// at-most-once decrement (replay record, claim-age fence, settle-carrier
// teardown, per-pass drain, ledger audit), against a real Valkey. The
// fault-injected arms are gated on `test-fault-injection` inside the module.
mod b_yuo_gc_tests;
// b.yuo decider ruling 4 — the release budget read from the store's own
// `maxmemory` (`storage::store_memory`): the test store as it is, replies
// staged through `ScriptReplyProxy` (a limit, 0, absent, malformed, refused),
// the refresh's effect on the gate and the GC threshold, and the cluster path
// when `LIVE_EDIT_TEST_CLUSTER_URL` is set.
mod b_yuo_store_memory_tests;
// b.yuo part E — the commit release ring `obj:wrel`: its entry grammar on
// every route, its type check, and that a lost reply's own-call verdict — or,
// for a chained call whose successor committed first, its lost-reply lookup —
// enqueues the landed commit's releases exactly once (the lost-reply arms are
// gated on `test-fault-injection` inside the module).
mod b_yuo_wrel_tests;
// b.pgs — the commit release ring through real commits on the fast,
// structural and WATCH routes: the lost-reply lookup on every ending that
// runs it (overtaken, a pending predecessor, exhaustion — at attempt 1 too —
// and a refusal-detail answer after a lost reply) with each of its outcomes
// (`enqueued`, `absent`, `failed`), the endings that never run it (`NOT_FOUND`,
// `FAILED_PRECONDITION`, `INTERNAL`), the own-call read's sequence check, the
// ring's write rule (eviction, weighted slots, the bound under writer churn,
// the three routes' parity, an empty `client_id`, a legacy per-`client_id`
// field left untouched), and the first-attempt hand-back of a writer-guard
// refusal, a fence wait and a recharge included. The module's shared fixture
// (the lost-reply staging, pinned commit tokens, the hooks) uses
// `test-fault-injection` seams, so the whole module is gated on it, the few
// cases that need none of them (the churn bound) included.
#[cfg(feature = "test-fault-injection")]
mod b_pgs_tests;
// b.zvd — three silent-corruption defects with one theme: a write or a read
// that reported success while losing data. (1) The snapshot read's per-parent
// children `LRANGE` swallowed every failure as "no children", so a transient
// blip let a structural commit DEL a real children list under a valid WATCH;
// (2) `HSET` merged the two value shapes, so an inline→external field rewrite
// left the stale `value` column and the row decoded as the old bytes forever
// (and leaked the external blob); (3) `write_object` had no pre-MULTI type
// probe and no DEL before its children RPUSHes, so a CreateLayer or DR restore
// over a surviving keyspace half-committed or appended. Most tests run
// unfeatured against real Valkey; the two arms a live server cannot stage — a
// transient snapshot-read failure (the children `LRANGE`, and the structural
// commit path's read) and a transient create-site probe failure — are
// feature-gated on `test-fault-injection` behind their own module headers.
mod b_zvd_tests;
// b.zt3 F-C — fred's command-queue counter forced high or drifted no longer
// holds a frame back: a GET at idle with the counter at 1 is answered (T6),
// the drift is reset at the next receive and logged once (T7), and a frame
// left fed with nothing behind it is flushed before the router waits (T18).
mod b_zt3_command_queue_test;
// b.zt3 — a pooled connection silent while its shard answers others: through
// a marker-stalling `ScriptReplyProxy`, the fast unresponsive verdict closes
// it in time for the stalled `GET`'s re-send to be answered inside the
// command timeout (a one-attempt `GET` fails with the verdict instead); and a
// data member with two fast verdicts is replaced by a fresh client — new id,
// supervision, hooks and the readers' routing view moved, the old client
// retired without a router restart and refusing sends at once, `quit()`
// leaving no member watch.
mod b_zt3_dead_connection_test;
// b.zt3 — a GC decrement whose reply is lost with its connection (held, then
// the connection closed, by `ScriptReplyProxy`) takes exactly one reference
// and leaves the data; the invariant only, not the at-most-once mechanism.
mod b_zt3_gc_resend_test;
// b.zt3 — what the unresponsive verdict leaves alone and what it records: a
// store answering nobody (a proxy hold), a reply still trickling in, and every
// connection silent at once (slow verdict, no replacement); `verdict_of`; the
// verdict WARN's fields; and the guard on expired connections (T-X): a fred
// `Transaction` whose `EXEC` read expired is replayed without its `DISCARD`
// waiting again on the failed connection.
mod b_zt3_liveness_test;
// b.zt3 — the unresponsive policy on every client (one peer registry, the
// readers without the fast rule) and its constants; every metric cell at zero;
// each way a member replacement ends (fresh connect failed, `quit()` before
// and after activation, spare cells used up, a verdict in the cooldown, a
// retired client's release after an aborted router); and `end_router_for_test`
// reaching a replaced member's fresh client. Postcomp-1: members due while a
// replacement holds the pool replaced at its swap in index order, slot
// ownership, the per-replica retiring cap (which ends a `maintenance`
// retirement in its grace like any other, never passing four), and a
// write killed by the cap reaching its caller as an unknown outcome.
mod b_zt3_member_replacement_test;
// b.zt3 T17 — the vendored fred `Pool` arena: every pool operation reaches a
// replaced member's fresh client, a `&Client` borrowed before stays valid, and
// `replace_member` refuses once the spare cells are used up.
mod b_zt3_pool_arena_test;
// b.zt3 T-RESP2 — a pubsub-shaped reply on a RESP2 data connection is matched
// to its own command (HMGET of message/ch/x, a subscribe-shaped and a nested
// message-shaped script reply), and the subscriber still gets a real PUBLISH.
mod b_zt3_resp2_test;
// b.zt3 postcomp-1 T-W0 — a real fred SCAN stream whose router is aborted with
// a page in flight ends with no error, and `orphan_sweep::complete_pages`
// ends the same walk with exactly one error.
mod b_zt3_scan_walk_test;
// b.zt3 T20 — a `txn` slot's client replaced under the slot's guard: never
// while checked out; slot id, watch and hooks re-keyed; permits unchanged; a
// racing router restart skipped; `end_router_for_test` reaching the fresh
// client; the `txn` budget.
mod b_zt3_txn_replacement_test;
// b.26e Stage 2 — the snapshot lane's routers on the client's `snapshot-io`
// thread after connect, a supervisor restart and a member replacement (their
// credential-refresh tasks beside them), no other pool's there, and both
// fan-out threads stopped once the client's last owner is dropped, quit or
// not, on a service worker, on a thread whose runtime has ended, or on
// `fanout-read` itself.
mod b26e_fanout_thread_test;
// Shared harness of the `b_zt3_` suites: the proxied client, the marker
// stall, peer reads, the metric names and the fred-log bridge.
pub(crate) mod b_zt3_support;
mod checkpoint_apply_test;
// b.y3m Epic 2 Task 2 — the children-index completeness marker (`chs_complete`,
// SR-1.7). On a marked object `obj:chs` must name exactly the parents owning an
// `obj:ch:*` key — checked against a keyspace SCAN, not the index or the
// snapshot loader — after every arm that writes a children list: CreateLayer, a
// DR restore over a wiped keyspace, `SetChildren` (non-empty and empty),
// `DeleteChildrenList`, and a reset. Also pins the marker's lifecycle: stamped
// by both `queue_object_state` entry points, kept by structural, reset and Lua
// fast-path commits, never added to an unmarked object. The restore fixture
// builder is imported from `checkpoint_apply_test` rather than copied.
mod chs_complete_tests;
mod client_quit_timeout_test;
mod client_test;
// b.yxt — cluster-gated regression coverage for the live-relay subscriber path
// (slot-bound stream readers, Timeout→force-reconnect self-heal,
// header-derived cursor, the b.c7a terminal-error contract and its routing
// carve-out). Every test in here self-gates on `LIVE_EDIT_TEST_CLUSTER_URL`
// and prints a SKIP line when it is unset, so the module compiles and passes
// everywhere; CI's `test-cluster:` job is what actually runs the bodies.
mod cluster_subscriber_test;
// b.dkt Epic 2 — the `commit_chain` header column (SR-4.5). Every
// materialization writes a fresh value: CreateLayer, an overwrite, a DR restore,
// and each of two restores of one checkpoint generation (same `seq` and
// `created_at`). The column is not structural: a header without it stays
// complete and readable. The Rust fold agrees with Valkey's own
// `redis.sha1hex` through a keyless `EVAL`. Every commit folds it, on the Lua
// fast path and the WATCH path alike: after a mixed history — sequential, and
// with txn and Lua writers racing until a WATCH retry is observed — the header
// equals the fold over the stream's `XRANGE` payloads (each the shared
// `update_stream_payload` string), a pre-feature object folds from `""` (on
// the `lua` backend's structural route too), and a
// refused commit leaves it unchanged. b.dkt Epic 5 Task 3 extends the
// sequential history to structural commits (`commit_structural`) interleaved
// with fast and WATCH commits on one object, every ordered switch between the
// three modes included, from the materializer's value and from `""`. The
// restore fixture builder is imported from `checkpoint_apply_test`.
mod commit_chain_tests;
// b.26e Stage 2 — a structural `UpdateLayer`'s commit charge through the
// storage seam (`storage::commit_charge`): the charge points' needs (WA0
// direct), the release-and-recharge outer loop (per-object permits and the
// txn checkout given back while it waits), the overflow-slot deadlock it
// prevents, the cold start's seam-A charge, the WATCH read's allowance stop,
// the bounded plan, grow-only recharges and the handler hint. The probes at
// the wait point read the per-object permits, so the module needs
// `test-fault-injection`.
#[cfg(feature = "test-fault-injection")]
mod commit_charge_test;
mod compute_diff_test;
// Harness shared by the stream-multiplexer suites and the gRPC suites that
// inject reader faults. `pub(crate)` so the `grpc` lane can reach it.
mod diff_read_test;
pub(crate) mod mux_test_helpers;
// The SR-5.5 durability-guard validator, without ever changing the server's
// `appendonly`. Each branch test connects through a `ScriptReplyProxy` that
// stages the reply to the probe `CONFIG GET appendonly` (`yes`, `no`, an empty
// reply as Azure Managed Redis gives, an error reply as MemoryDB / ElastiCache
// give, a delay past the probe timeout) and counts on the wire whether the
// probe was sent: once with checkpointing off, never with it on (case 2's
// Disabled-adapter refusal and case 1a's real adapters). One read-only test
// runs the validator against the real server and asserts only that its probe
// got a definite answer (`Ok` without the WARN, or the SR-5.5 refusal), which
// holds whatever another tenant does to the shared server's `appendonly`.
mod durability_guard_test;
// b.bnn — source guard over every production site under `src/` that can read
// external-value blob bytes (the `KeySchema::ext_data` key builder, the
// `ExternalValueStore` read methods, the key's literal spelling, and the Lua
// scripts handed a blob key): each is allowlisted with the reason it is safe
// under the 5 s GC floor (SDD §11.2, §11.3). Static: reads source files, needs
// no Valkey and no feature.
mod ext_blob_reader_guard_test;
mod external_value_test;
// b.gbt, b.bo2 — a write fred fed but did not flush never waits through a
// cluster client's inline connect (a reconnect or cluster-sync attempt, an
// `-ASK` target's on-demand connect) nor through the router's wait on another
// connection's I/O (a write that can reach a silent socket, an `-ASK`
// target's write, the flush before the router waits), and every such flush
// costs one liveness bound however many connections are silent:
// deterministic tests on a four-node facade over `ScriptReplyProxy`, one per
// hook and per term of the check before a write, and one `#[ignore]`d
// real-cluster test (`fed_frame_flush_disruptive_*`, run by CI's
// `test-cluster-disruptive:` job) that holds the reconnect with `CLIENT PAUSE`.
mod fed_frame_flush_test;
// G3 (SDD SR-5.3) — an external value's hash is exactly 32 bytes. Write side:
// a wrong-length hash is refused `InvalidArgument` on both backends and leaves
// the object's keyspace byte-identical (the handler's no-blob half is in
// `grpc/update_layer_test.rs` / `grpc/create_layer_test.rs`). Read side: a
// malformed stored `hash` column fails the test-support reader
// `read_object_state_for_test` and both `read_object_snapshot` routes
// `Internal`, never panicking or decoding a
// padded hash. The atomic-read fixture builders are imported from
// `snapshot_read_atomic_test`.
mod g3_external_hash_tests;
mod gc_test;
mod layer_test;
// b.v9c Epic 3 — SDD SR-2.5: two processes sharing a `service_label` (one label,
// two `LeaderIdentity` instances) never both hold a leader lock — through the
// public acquire / refresh-if-held wrappers, across a real short-TTL lapse,
// through `LeaderLease::ensure_held`, and under a barrier-synchronised acquire
// race. No feature gate beyond `test-support`.
mod leader_identity_test;
// b.fd9 follow-up — site-level routing guard. Greps the b.fd9-scope
// source files for direct `ServiceError::Internal(format!` wraps of fred
// errors that should route through `map_transient_fred_err` /
// `map_fred_err_for_grpc` instead. Compile-time-cheap, runtime-cheap
// (single integration unit test).
mod mapper_routing_test;
mod orphan_sweep_test;
// b.qpf post-completion (review finding 2) — the orphan sweep's retirement
// fence. `cleanup_orphaned_object` now writes `HSET obj:hdr superseded 1` before
// reading the object's external-value references, so a straggler `UpdateLayer`
// cannot land between that read and the UNLINK and release the same reference
// twice — which, with a second live object holding the hash, deletes a live blob
// rather than merely leaking one. Sits beside `orphan_sweep_test` (the sweep's
// structural semantics) and `b_kf1_tests` (that the sweep schedules ext-GC at
// all) rather than inside either: this is the third question, whether the
// scheduling read is SAFE. The b.qpf keyspace readers are imported from
// `b_qpf_tests` rather than copied, so `ref_count` / `pending_members` /
// `obj_ext_sorted` have one definition across both files. Most tests run
// unfeatured; the two WATCH-window arms need `test-fault-injection` for the
// commit-window hooks and live in an inner `#[cfg]` module that is compiled out
// — not skipped — without it.
mod postcomp_orphan_sweep_fence_tests;
// b.v9c Task 2 — the atomic presence store operations in `storage::presence`:
// register / refresh / unregister, the teardown marker, `lease_expires_at_ms`,
// the SR-3.8 index-removal helper and the test-only eviction helper. Events are
// observed on an external subscriber and proven exhaustive with a published
// fence rather than a sleep. No feature gate beyond `test-support`; the cluster
// variant self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` and prints a SKIP line
// when it is unset.
mod presence_ops_test;
mod pubsub_reconnect_test;
// b.kgb — two source fences over production code under `src/` (comments,
// string literals and test-only `cfg` items removed): a streaming RPC takes its
// shutdown token only from `GracefulShutdown::enroll_stream`, never a bare
// child of the service shutdown token; and no file but `src/error.rs` builds a
// gRPC status detail (`ErrorInfo` / `RetryInfo` / `ErrorDetails`, or a
// hand-written `grpc-status-details-bin`). Static: reads source files, needs no
// Valkey and no feature.
mod refusal_and_shutdown_guards_test;
mod session_test;
// b.xc9 — source fence for the rule that no test against the shared
// `TEST_VALKEY_URL` server changes server-wide configuration: `CONFIG SET` /
// `REWRITE` / `RESETSTAT` (fred's `config_set` family, or a `CONFIG`
// `CustomCommand` with one of those subcommands) may appear only in an
// allowlisted file whose tests run against their own server
// (`LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL`), at that file's exact site count.
// Comments that mention the command do not count. Static: reads source files,
// needs no Valkey and no feature.
mod shared_valkey_config_fence_test;
// b.y3m Epic 3 Task 2 — the per-object snapshot fence's storage layer
// (`storage::snapshot_fence`): every verdict and no-write rule of the six fence
// and intent scripts through their wrappers, time moved by staging fields
// against the store's `TIME` (no timing assertions), the Lua copies of the cost
// model against `FenceCoefficients::timing`, argument errors, and the WATCH
// path's type probe on both keys. Needs only Valkey. The cluster-gated twin runs
// every script for an object on each shard, self-gates on
// `LIVE_EDIT_TEST_CLUSTER_URL` and prints a SKIP line when it is unset.
mod snapshot_fence_cluster_test;
mod snapshot_fence_test;
// b.y3m Epic 3 Task 3 / plan b.dkt Task 11 (W1) — writers honour the snapshot
// fence (SR-5.4, structural-edits SR-8.1): every `update_layer` path (the
// update script's fast mode, its structural mode on a cached object, a cold
// start and a reset, and the WATCH path under `lua-watch-structural`, `txn`
// and as the uncacheable fallback) against a fence granted by the Task 2
// wrapper. A fenced commit waits — uncounted, on a one-attempt budget —
// writes nothing and commits after the release; a killed reader's lease ends;
// the fence-wait cap fails `UNAVAILABLE`; writer intents are registered only
// by WATCH-path attempts that read a snapshot (only on layers known to be
// above the ceiling), refused while a reader is live and cleared on every
// exit; step 1d's unfenced cost in both script modes; a fenced txn writer
// holds no txn slot while it waits; and every path refuses a wrong-typed
// fence or intent key with one refusal. A background writer's attempts are
// observed on a `MONITOR` feed, time is never asserted, and every test runs
// under one 90 s deadline. The seam-driven arms (a fence granted between
// `WATCH` and `EXEC`, an abort, a dropped attempt, intent renewal, the tallies
// classification, an already-landed commit under a fence) sit in an inner
// `#[cfg(feature = "test-fault-injection")] mod seams`, compiled out without
// that feature.
mod snapshot_fence_writer_test;
// b.y3m Epic 3 Task 10 (Subtask f2) — the fence under streams of fenced reads,
// end to end through `LiveEditService`, with counts and no clock: structural
// writers keep committing under closed-loop joiners (no retry exhaustion,
// fence waits, intents); two service instances on one store serve a paced join
// stream verified, one epoch per join, the cut advancing between every two
// epochs; a replica whose memory is held misses a join window and is still
// served in its own epoch; a crashed reader's lapsed epoch is followed by the
// writers' gap. Needs only Valkey; about 25 s for the longest test.
mod snapshot_fence_stream_test;
// b.y3m Epic 2 Task 4 — the atomic snapshot read (`read_object_snapshot` over
// `object_snapshot_read.lua`). Equality with the reference reader
// (`test_support::read_object_state_for_test`) on every
// state the object's own writes produce (marked and unmarked), the
// unmarked-partial-`obj:chs` regression and its marked twin, RESP2 decoding,
// the cut and tallies, never-tears under a concurrent writer, the item / byte
// ceilings, a WRONGTYPE refusal per key family (the b.atj fixture and
// asserters, imported), the read metrics, the key set from a `MONITOR`
// record, width pinning across every row writer, and the startup self-test's
// verdicts through a reply-substituting proxy (the F1 regression). The
// fault-seam module needs `test-fault-injection` and is compiled out without
// it. Instruments live in `snapshot_read_support`; the cluster-gated twin
// self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` and prints a SKIP line when it is
// unset.
mod snapshot_read_atomic_test;
mod snapshot_read_cluster_test;
// b.y3m Epic 3 Task 5 — the fenced snapshot read
// (`storage::snapshot_fence::read::read_object_snapshot_fenced`). Equality
// with the atomic read on every quiescent shape through a below-ceiling
// client, a commit refused under the lease and landing after it, the release
// proof (a pass at another `seq`, a header moved before release), restore and
// cleanup under the read (the `NotFound`-needs-the-release's-confirmation
// arms, a failed release), `FENCE_READ_ATTEMPTS` exhaustion, the hook's event
// order, attempt outcomes and refusals, `CappedFenceWait`, the `path="fenced"`
// and fence metrics, and cancellation at each step leaving no live lease. The
// renewal-timer, lease-lost-in-a-hook, failed-fan-out and mid-read
// cancellation arms use the fault seam's delays, injected errors and renewal
// stall in an inner `#[cfg(feature = "test-fault-injection")] mod seams`,
// compiled out without that feature.
mod snapshot_read_fenced_test;
// b.mr6 — the structural consumers' slim reads
// (`read_object_snapshot_slim`, and the fenced read for a slim consumer):
// on every path the compact projection of the full read of that path, or its
// failure, over the stored row shapes no commit writes (legacy, malformed,
// empty, absent, foreign-only, wrong-typed, bad base64); and the slim atomic
// reply's received bytes against the full one's on a rig-shaped layer. The
// generated histories are `tests/proptests.rs`'s.
mod snapshot_read_slim_test;
pub(crate) mod snapshot_read_support;
// b.y3m Epic 3 Task 6 — the shared fenced read, `storage::snapshot_flights`
// (the flight registry's fenced arm), driven directly: coalescing (waiters
// attached while `Queued` and between grant and release share one read; a
// newcomer after the release creates a successor that reads after the
// predecessor's `done`), the drop-order regression (a cancelled predecessor's
// slot, permit and grant are free when its `done` wakes the successor), FIFO
// build turns and extra builds, the clamp, two waves and the put-back of a
// displaced predecessor, ownership, the progress rule, per-waiter charging and
// the reader cap, replica shutdown, isolation, failure and panic, and
// cancellation at every await. The arms that hold a read open, break it or
// fail it need the fault seam and sit in an inner
// `#[cfg(feature = "test-fault-injection")] mod seams`.
mod snapshot_flights_test;
// b.y3m Epic 2 Task 5 — the unfenced bounded fan-out (`read_object_snapshot`
// for the structural `UpdateLayer` and ext-GC consumers). Equality with the
// atomic read on every quiescent shape (marked and unmarked), the read at a
// one-command one-byte window, a `MONITOR` record that neither consumer ever
// runs the atomic script (directly, and through a WATCH-path structural commit
// and a `DeleteLayer` above the ceiling), the SR-1.5 header-without-`seq` rules, a
// WRONGTYPE refusal per key family on the b.atj fixture, and the
// `path="unfenced"` metrics. Shares `snapshot_read_atomic_test`'s fixture
// builders. The fault-seam module needs `test-fault-injection` and is
// compiled out without it. The windows' mechanics and the structural path's
// torn-read proof are inline unit tests in `src/storage/diff_read.rs` and
// `src/storage/layer.rs`.
mod snapshot_read_unfenced_test;
mod stream_consumer_test;
mod stream_mux_cluster_test;
#[cfg(feature = "test-fault-injection")]
mod stream_mux_fault_test;
// b.xtb stage 4 — the reader's head check on a reply whose first entry jumps
// past its read position: a restore band is `no_gap`, a recreated stream ends
// the subscription behind it `stream_gap`, and a failed check loses nothing and
// is charged to the transient budget. The fault cases sit in an inner
// `#[cfg(feature = "test-fault-injection")]` module, compiled out without it.
mod stream_mux_gap_test;
// b.xtb stage 5 — a stream reader's checks of its own connection: the
// idle→active probe (a hang reconnects under `probe`; failures are charged to
// the transient budget one each; a permanent error ends the reader's
// subscriptions and heals), the idle keepalive (only with nothing to read,
// reservations included; a failed tick reconnects under `keepalive`; the budget
// is neither charged nor reset) and the liveness wake. The whole module needs
// `test-fault-injection` (fault sites and knobs) and is compiled out without
// it. One test self-gates on `LIVE_EDIT_TEST_CLUSTER_URL`; the disruptive one
// is `#[ignore]`d and self-gates on `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL`.
#[cfg(feature = "test-fault-injection")]
mod stream_mux_health_test;
// b.azh — stream readers connect lazily, and on a cluster to their bound
// slot's primary alone: no reader connection at startup, the connected gauge,
// quit with connected readers, hooks armed once per client; and, through a
// two-node cluster facade over `ScriptReplyProxy` (runs against the standalone
// store), the slot-owner connection, the retarget on a rebind and on a real
// `-MOVED`, supervision and hooks across retargets, node affinity, the
// clustered reader's `CLIENT UNBLOCK` interrupt, the clustered reader's own
// credential re-AUTH (b.19i), fred's cluster-wide `AUTH` broadcast behind a
// blocked read failing at the slow bound (b.zt3), another node's replies
// counting as no evidence for a silent connection (b.zt3 T3), which replies
// count as evidence for a stalled command — a slot another node owns, one
// node serving two ranges, and the positive control (b.zt3 T-K2a, T-K2b,
// T-K3, T-Kpos) — a connection whose flush expired being neither flushed nor
// fed again (the b.zt3 guard on expired connections), and the readers'
// routing after data member 0 is replaced (b.zt3 T14). Everything that drives the
// multiplexer's fault sites or knobs needs `test-fault-injection` and is
// compiled out without it: the keepalive case, the inner `faults` module
// (every failed, stalled or held reader connect, staged with the `Connect`
// fault site — transient, permanent, the credential hint, the connect bound,
// past the budget, quit mid-connect, a subscription leaving during a held
// connect and the reader gauges returning to 0 (b.9y2), a failed retarget and
// its retry, the failed-retarget bind — forced reconnects that retarget,
// retire or rebuild,
// `-MOVED` on the probe and head check and the consecutive-`-MOVED` charge,
// the stalled-`QUIT` retargets), and the zero-budget knob of the `-MOVED`
// XREAD retarget test — that test still runs without the feature, but no
// longer shows the redirect went uncharged.
mod stream_mux_reader_connect_test;
// b.19i — a stream reader re-AUTHs its own connection between its reads:
// fred schedules no refresh on a reader client; a due credential wakes an
// in-flight read, and the `AUTH` goes out with no read in flight. Through a
// counting proxy and a scripted (or real, rotation-gated) provider. The fault
// and knob cases sit in an inner `#[cfg(feature = "test-fault-injection")]`
// module, compiled out without it.
mod stream_mux_reauth_test;
mod stream_mux_test;
// b.fsz MR-D4 — external-value clean-up of a retiring object, bounded in
// memory (`storage::retirement`). The reference walk against an independent
// readers'-rule oracle over every row shape, the signed-zero pair, a header
// without a numeric `seq`, and its `ext_gc` snapshot-read metrics; the
// retirement allowance `R_ref` on every route (the bound, FIFO, every exit
// giving it back, the clamp, the top-up, the delete-vs-replace retarget on one
// replica and across two, wake-up before reservation, the sweep one at a time,
// a duplicate tail paying once, the size read's foreign-key and transient
// arms, the per-instance metric sink, the reservation sized per object and
// its widening past an index that grew after the size read); and the paged
// teardown's first-`UNLINK` failure, `DeleteLayer`'s re-run, the overwrite
// and sweep routes' in-place page retry, and a failed teardown's report,
// the same on every route. The allowance and teardown modules drive the
// fault-injection seams and are compiled out without `test-fault-injection`.
#[cfg(feature = "test-fault-injection")]
pub(crate) mod retirement_budget_test;
#[cfg(feature = "test-fault-injection")]
mod retirement_teardown_test;
mod retirement_walk_test;
// b.dkt Epic 4 Task 2 — object retirement (`DeleteLayer`, and the old object
// of a `CreateLayer` overwrite, on success and on their error exits) invalidates
// the store's structural-planning cache, so a put-back by a checkout taken
// before it is refused. The cache's own epoch rules are inline in
// `src/storage/structural_cache.rs`.
mod structural_cache_retirement_test;
// b.dkt Epic 4 Task 3 — the structural cache's cold start
// (`storage::structural_cache_cold_start::cold_start`, Spec SDD SR-4.3; and,
// Epic 6 Task 11, the probed route `cold_start_probed` and C1's fenced cold
// start, `storage::structural_cache_fenced`): every route (`atomic`, `fenced`
// on items and on bytes, and `uncacheable` — over weight, unmarked), the
// fenced entry against the atomic entry at the same cut, the probe's item
// bound, the route predictor (one read when predicted over bytes, a counted
// miss), the pre-read weight check on both paths, superseded / absent /
// disabled answers, the fenced read's permit pair, `P_f` and memory grant
// (order, what is held during a `RETRY_AFTER` sleep, release on cancel and on
// failure, the grant's size and its drop before admission), the waiting cap,
// the two per-replica plain permits, a `MONITOR` record that every snapshot
// script runs at the plain ceilings and every fan-out command sits under a
// lease, the script's `return_dangling` flag (`ARGV[12]`), a `DeleteLayer`
// during the read answering `Invalidated` with no token read, Task 2's
// deferred side-store and negative-TTL items end to end, and the cold-start
// metrics. The cache's ticket rules are inline in
// `src/storage/structural_cache.rs`.
mod structural_cache_cold_start_test;
// b.dkt Epic 4 Task 5 — the structural cache's read side and stream repair
// (`storage::structural_cache_read::read_for_structural_edit` over
// `storage::structural_cache_repair`, Spec SDD SR-4.1–SR-4.3, SR-3.2, SR-3.4,
// SR-2.4, SR-5.4), on objects committed through `update_layer` with relay
// bytes: repaired and cold-started entries equal the fan-out's normalized
// state across every commit shape; four 256-entry chunks for 1,024 pending,
// `too_far` for 1,025; one test per reload reason, each followed by an
// accepted put-back and a hit (`dangling_adopted` on a real orphan adopted by
// another store's `update_layer`); dangling-parent upkeep during replay; the
// header mapping on a hit, a miss and after the probe; token-before-plan on
// every route, between two repair reads included, and `update_layer`'s
// wrong-type refusal byte for byte; the `NoChsComplete` negative entry
// (`MONITOR`) and an over-weight cold start; G4 replay; `repair_to_target`
// ticking nothing and its out-of-range targets; `plan_adopts_dangling_parent`.
// The guard's settling rules are inline in `src/storage/structural_cache.rs`.
mod structural_cache_read_test;
// b.dkt Epic 4 Task 6 — the structural cache's sampled audit
// (`storage::structural_cache_audit`, Spec SDD SR-3.6, SR-2.4): `match` on an
// unchanged object, the same cut with commits landing between the read and the
// repair (a reply-holding proxy) and a cut behind the entry `skipped`;
// `diverged` after each `rewrite_out_of_band` change, dropping the entry,
// refusing an earlier checkout's put-back, writing nothing (`DUMP`) and logging
// one WARN; every `skipped` reason, store errors on either read and a delete
// between the read and the repair included; the above-ceiling switch and,
// above the plain ceilings, the fenced read under the audit's own consumer
// (`MONITOR`: plain ceilings only, the fan-out under the lease), its pacing,
// its `busy` skips that never wait for capacity, its memory grant freed
// before the repair, a live lease waited out holding nothing (none of it for
// the plain read, which takes no cold-start permit); metric isolation; one
// audit at a time across store clones, through a panic; shutdown cancellation
// before the read and mid fenced read.
// `first_difference`'s naming and walk order are inline in
// `src/storage/structural_cache_audit.rs`.
mod structural_cache_audit_test;
// b.dkt Epic 4 Task 4 — the structural cache's repair read
// (`storage::structural_cache_sync` over `structural_cache_sync.lua`, Spec SDD
// SR-4.1), on objects seeded through `create_layer` and fast-path and
// WATCH-path `update_layer` commits. The script writes nothing (a `DUMP` /
// `PTTL` keyspace witness across every reply shape); probe mode reads only the
// header and the commit-token record (`MONITOR`); 300 pending entries chunk as
// 256 then 44; the byte bound's exact fit, stop and `TOO_LARGE`; one-entry
// `XRANGE` pages that stop with the chunk (`MONITOR`), and exact bounds and
// range ends at a wider page; the target `seq` cap; the header facts as
// stored; a wrong-typed commit-token key reported as data; and the refusals —
// a wrong-typed header or stream, and a `seq` argument above 2^53 − 1, fail
// `Internal`. The source scans and decoder shape tests are inline in
// `src/storage/structural_cache_sync.rs`.
mod structural_cache_sync_test;
// b.dkt Epic 5 Task 2 — the structural commit (`storage::structural::
// commit_structural` over `update_layer_fast.lua`'s structural mode, SR-6),
// planned by the test-support planner on `snapshot_read_atomic_test`'s
// `every_shape_state`: each record tag's writes, the header, the stream entry
// and the successor `commit_chain`; `STALE`, and `CACHE_DIVERGED` per digest
// kind and digest-carrying record tag (a section row per column), each
// writing nothing; a key touched twice in one delta still commits;
// the Rust list digest against the script's `redis.sha1hex`; every parse-pass
// `ENCODING` refusal, through a raw structural-mode invocation builder
// (`StructuralInvocation`, `pub(crate)` for reuse); `AlreadyLanded`
// from a fast-mode or structural token record; the observer's calls; source
// shape (one shared already-landed decode and error mapping, no `.await` before
// the landing calls, no `txn_pool` / `WATCH` / `MULTI`); and the test-support
// keyspace dump. The fault-seam module (release-enqueue order, the Lua fault
// queues, the fixed-token comparison with the fast path) needs
// `test-fault-injection` and is compiled out without it. The digest rules and
// the record encoder are unit tests in `structural.rs` and `lua.rs`; the C3
// cache API's tests are in `src/storage/structural_cache.rs` (Epic 4's file).
mod structural_commit_test;
// b.dkt Epic 5 Task 3 — structural mode keeps every write-safety guard the
// fast path keeps (Spec SDD SR-10.4, script side), each test the structural
// counterpart of a named fast-path test: superseded and the token check ahead
// of it, stream-top fail-closed, the sequence / set_order ceilings,
// `obj:chs` ⟺ lists with `chs_complete` unchanged, `obj:ext` counting with
// `rewrite` releases, the `obj:wtok` record and a replayed `EVALSHA`, partial
// vs absent header, `WRONGTYPE_KEY` per written key family (and the header,
// children-list heal and digest-checked list cases), one value shape per row,
// `STALE` under a racing fast-path commit, and a delete over an old
// tombstone. Shares `structural_commit_test`'s fixture.
mod structural_guard_parity_test;
// b.dkt Epic 6 Task 6 — the cached structural attempt end to end, through
// `update_layer` on the `lua` backend: same-replica serialization; `STALE`
// under a racing fast writer (repaired, committed once, one start record per
// call) and staged by the structural pre-dispatch hook (on `structural_full`
// and on `reset`, the retry repairing one entry); a re-plan carrying the
// call's start record (overtaken); the lost reply answered by the retry's read
// side; the shared own-commit result, from the read side and from the commit
// script; the commit script's `Recorded`, `OVERTAKEN`, `NOT_FOUND` and
// `SUPERSEDED`, staged by the same hook; the op-id re-send regression;
// `CACHE_DIVERGED` (on a hit and a reset) and divergence never looping (three
// planted shapes, then a WATCH-path commit); the identity fence across two DR
// restores; uncacheable routing and both halves of the C3 over-cap put-back;
// the put-back by every outcome; own adoption; the audit trigger; and the
// `FENCED` wait and its caps (one long fence, several short ones together, the
// remaining deadline) through the scripted-reply hook. Needs
// `test-fault-injection` (the fault queues and hooks) and is compiled out
// without it.
#[cfg(feature = "test-fault-injection")]
mod structural_dispatch_test;
// Structural-edits C1 — the structural cache's fenced cold start through
// `update_layer` on the `lua` backend, on a client whose plain item ceiling
// puts the object above it: a fenced read that fails for its call (three
// broken attempts staged by the snapshot-read delay seam and a raw header
// `seq` change, the waiting cap, a capacity wait timing out) commits through
// the WATCH path for that call only, with no negative entry, and the next call
// caches; closed capacity answers the shutdown error, never WATCH; the
// read-permit wait under `consumer="structural_cache"`; no writer intent on
// the structural route with a lingering fence key (`MONITOR`); and the WATCH
// fallback classified from the read side's size, registering its intent
// before `WATCH`. Only the broken-attempts case needs `test-fault-injection`
// (the delay seam); it is gated inside the module, and the rest run without
// the feature.
mod structural_fenced_cold_start_dispatch_test;
// Structural-edits C1 — the structural route's size gate
// (`storage::structural_size_gate`) through `update_layer` on the `lua`
// backend, at a small per-store limit (the test-support override): a plan of
// exactly the limit commits on the structural route and one unit over
// commits through the WATCH path (`over_work_limit`, no structural
// `EVALSHA`, the entry put back); the children term counting a large
// parent's held list; the pre-read reset gate sending a cold reset estimated
// over the limit to the WATCH path with no cold-start read
// (`reset_over_work_limit`, still cacheable) and its boundary cold-starting
// as before; and the slow-commit WARN, staged by a reply-delaying proxy.
mod structural_size_gate_test;
// b.mr6 PD1 and b.8jc — a time-sample row's identity on the structural write
// path. An object holding `0.0` and legacy `-0.0` rows of one external hash, on
// the cached route and the WATCH path: an unrelated structural commit keeps
// both counted in `obj:ext` and releases nothing, a rewrite of the `0.0` row
// moves only its count, a reset and a cascade each delete both, and on the
// cached route a rewrite of the `0.0` row over an `obj:ext` drifted to count
// the hash once releases nothing. Rule E (a reset or cascade deletes time 0 at
// both spellings) against a cache entry holding fewer spellings than the
// store: an older build's `-0.0` commit replayed at `+0.0`, a lone one under a
// this-build write, a merge under a cached pair (each `CACHE_DIVERGED` once,
// then no row left), a cold start over a legacy row (first-attempt commit),
// and the WATCH path's added delete. And a repair over a this-build record
// whose bytes carry `-0.0` equals a fresh read.
mod structural_signed_zero_test;
// b.dkt Epic 5 Task 4 — structural mode's `obj:ext` accounting on a drifted
// index beside the WATCH path's (Spec SDD SR-10.3, SDD §7.5.6): the raise-only
// floor repairs an under-count (where the WATCH path does, and also where the
// commit nets no change), keeps an over-count or absent count where the WATCH
// path releases, and cannot release on a cache wrong about an untouched row
// or the touched row; every other key is equal. Keyspace equality itself is
// the proptest in `tests/proptests.rs` (`structural_watch_equivalence`).
// Shares `structural_commit_test`'s fixture.
mod structural_watch_equivalence_test;
mod test_valkey_test;
// b.8jc — the reference-count reconciler's signed-zero merge, through its
// public API: rule B's decision on planted inline and external pairs and a
// lone `-0.0` row, the readers' full read unchanged, no `obj:ext` write and no
// release, idempotency, the script's per-row skips (a stale plan's `-0.0` key
// already gone, a keep whose `+0.0` row or member is gone, a `-0.0` key with
// no member), a `+0.0` key with no member taken over, `CHANGED`, `FENCED`,
// `NOT_LIVE`, two merges at once, more sections than one call takes, and
// (under `test-fault-injection`) the header stamp aborting a watched txn
// commit. The pass-level arms, a failed merge among them, are in
// `storage::ext_reconcile`'s own tests.
mod time_sample_merge_test;
mod update_layer_admission_test;
// Spec SDD SR-2.1–SR-2.3 — the `UpdateLayer` commit metrics' `path` label,
// conflict counter and commit-duration histogram. Needs `test-fault-injection`
// (fault injectors and the WATCH-window hook) and is compiled out without it.
// `pub(crate)` because `update_layer_commit_count_test` imports its metric
// readers and `Aborted` injector.
#[cfg(feature = "test-fault-injection")]
pub(crate) mod update_layer_commit_metrics_test;
// Spec SDD SR-2.6 / SR-1.2 — every landed `UpdateLayer` commit is counted and
// logged exactly once, at its landing: each of the eight landing routes (the
// structural route's commit and its read-side already-landed answer included) run to
// completion and dropped right after the landing (the commit-landed hook), plus
// the commit line's attempt / path / per-kind conflict / records / bytes fields
// checked against the store's `MONITOR` feed. `pub(crate)` because
// `grpc::rpc_failure_log_test` reuses its `MONITOR` byte counters. Needs
// `test-fault-injection` (the hook and the fault seams) and is compiled out
// without it.
#[cfg(feature = "test-fault-injection")]
pub(crate) mod update_layer_commit_count_test;
mod update_layer_lua_test;
// Gated behind `test-fault-injection` — the tests here install the
// `LayerStore::with_fault_injector_for_test` and
// `with_fred_error_injector_for_test` seams, which only exist when the
// feature is enabled. Without the feature this module is omitted and the
// rest of the integration binary still compiles.
//
// Run via: `cargo test --workspace --features test-fault-injection`.
// See Cargo.toml `[features]` and CLAUDE.md "Build & Test Commands".
#[cfg(feature = "test-fault-injection")]
mod session_fault_injection_test;
// b.fd9 — Sweep storage/grpc Internal → Unavailable for transient Valkey
// errors. Drives the new `map_fred_err_for_grpc` classifier and the
// retry-wrapped `delete_layer` body via test-only seams.
#[cfg(feature = "test-fault-injection")]
mod transient_classification_test;
#[cfg(feature = "test-fault-injection")]
mod update_layer_retry_test;
// b.7td — `txn_pool` checkout hygiene. A borrower that dies mid-transaction
// leaves an open MULTI on the pooled connection; fred can neither detect nor
// repair it, so every checkout now probes (DISCARD then UNWATCH) and recycles a
// dirty slot before handing it over. Single-node by design — the residue lives
// on one connection — see the module header for why no cluster variant exists.
mod txn_checkout_test;

// b.tbw part one — the two whole-write transactions (the WATCH-route
// `UpdateLayer` commit, `CreateLayer`'s object write) wait a timeout scaled to
// the commands they queue: a reply that outlives a low command timeout but not
// the scaled one commits on its first attempt, with no orphan; the residuals
// (a forced short object-write timeout, a client deadline shorter than a
// reset) are pinned. Needs the pre-dispatch hook and the object-write timeout
// seam, both under `test-fault-injection`.
#[cfg(feature = "test-fault-injection")]
mod txn_timeout_test;
// b.dkt Epic 6 Task 6 — `update_layer`'s relay precondition of the
// `ompf`-presence invariant (SDD §12.2) through the public storage API: a
// delta without non-empty relay OMPF is refused `Internal` before any store
// I/O on every backend, and before the admission wait on `lua` (that test
// needs `test-fault-injection`); `(None, Some(empty))` is the plain empty
// commit with no `ompf` field, and `(None, Some(non-empty))` writes those
// bytes as `ompf`, on every backend; and the `next_set_order` regression the
// refusal closes (a relay-less fast commit, then a structural commit on
// `lua`).
mod update_layer_relay_precondition_test;
// b.yvh — the per-object relay byte budget (`LIVE_EDIT_LAYER_STREAM_MAX_BYTES`)
// on the fast path, the WATCH path and structural mode: the header's
// `relay_gens` equals the Rust fold after every commit, the stream keeps what
// the fold's cuts keep (B + 2 × the largest entry at most, never trimming
// history the budget had room for), the entry cap still applies, B = 0 writes
// nothing, foreign columns fold identically on every route, and a new object
// starts fresh. Its lost-reply cases need `test-fault-injection` and are
// compiled out without it.
mod relay_retention_test;
// b.26e f3c — report only: the wall time per command of one clustered general
// client under the three test sweep modes, interleaved, and the
// connection-buffer sweep's own time per pass, as `SWEEP-COST:` lines.
// `#[ignore]`d and cluster-gated: it runs only in the `test-cluster-disruptive:`
// job and prints a `SKIP:` line without `LIVE_EDIT_TEST_CLUSTER_URL`.
mod retention_sweep_cost_test;
// b.bo2 PD2 — what fred's check before a write (`live_edit::before_socket_write`)
// costs per write on a cluster, over pipelined, single and burst shapes:
// asserted at most 1,000 ns per check in a release build, and each shape's
// flushes before a write equal to the design's prediction in any build, with
// `WRITE-CHECK-COST:` lines. `#[ignore]`d and cluster-gated: it runs only in
// the `test-cluster-disruptive:` job's release-build sweep-cost step; without
// `LIVE_EDIT_TEST_CLUSTER_URL` it prints a `SKIP:` line outside CI and fails
// under `GITLAB_CI`.
mod write_check_cost_test;
// b.yvh F5 — an upload write the backing store refuses before applying
// anything (`-OOM`, `-MISCONF`, `-READONLY`, `-NOREPLICAS`) is answered
// `UNAVAILABLE` once, and counted on
// `live_edit_store_write_refused_total{stage, reason}`: with the retry hint
// and "nothing was applied, retry" at the commit script in both modes (plain,
// outcome unknown, after a counted retry), the WATCH path's discarded
// transaction and the external-value pre-store; plain at CreateLayer's four
// writes. Each staged through `ScriptReplyProxy`. Its handler-level cases need `test-fault-injection`; its
// real-`-OOM` case is `#[ignore]`d and runs only in the
// `test-cluster-disruptive:` job against that job's own Valkey. b.xd1 adds a
// discarded transaction classified from its cause (CreateSession and the
// WATCH path: a store refusal answered plain `UNAVAILABLE` once, `-LOADING` /
// `-BUSY` retried) and a store refusal at a mapper site (DeleteLayer's mapping
// `DEL`, JoinSession's presence-index `SADD`): plain `UNAVAILABLE`, uncounted.
// b.pa1 adds that a store write refused `-OOM`, at an upload site or a mapper
// site, requests the GC's out-of-memory collection.
mod store_write_refusal_test;
// The disruptive store (`LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL`) and the
// `maxmemory` guard shared by the `store_write_refusal_disruptive_*` tests of
// `store_write_refusal_test` and `oom_collection_test`.
pub(crate) mod disruptive_valkey;
// The poll for the GC's out-of-memory collection request a store write refused
// `-OOM` writes (b.pa1), shared by `store_write_refusal_test` and
// `b_ua9_tests`.
pub(crate) mod oom_request_support;
// b.pa1 — the GC's out-of-memory collection: the decrement's opener changes
// no result (a differential against the pre-change script), the leader takes
// one collection per request and backs off after a futile one, the upload
// hold stands for the collection and is cleared on every exit, a collection
// honours read holds and never deletes a value still referenced, a refused
// collection is told apart (`refused`, WARN once), and a pass skips a
// re-claim refused `-OOM`. The hold's error exit, `refused` and the skipped
// re-claim need `test-fault-injection` and are compiled out without it.
// Its real-`-OOM` cases are `#[ignore]`d `store_write_refusal_disruptive_*`
// tests that run only in the `test-cluster-disruptive:` job against that
// job's own Valkey.
mod oom_collection_test;
mod update_layer_transaction_test;
mod watch_leak_test;
// Shortens BLOCK through the stream-mux test knobs.
#[cfg(feature = "test-fault-injection")]
mod xread_block_idle_timeout_test;

// Task t2.fgj.ya.9j — checkpoint loop integration tests.
mod checkpointer_end_to_end_test;
// b.yut — checkpoint cadence regression. Drives the gate off REAL commits
// (no injected timestamps) to pin the µs commit clock the loop reads, and
// pins the interval arm as both a liveness guarantee and a write-volume
// bound for a layer that is never idle.
mod checkpointer_interval_cadence_test;
mod checkpointer_leader_race_test;
mod checkpointer_metrics_test;
// b.dfh Epic `t1.dfh.57` Task 7 — PRD AC-1 driven tail cadence, the Epic's
// flag-off zero-movement criterion, and Task 4's four pinned outcome→metric /
// cursor-stamp mappings, all through the real periodic loop. No feature gate:
// the tail knobs are ordinary `CheckpointLoopConfig` fields, not an injection
// seam. Wall-clock-bound; every test holds ONE deadline and the module header
// tabulates the per-test worst case against nextest's 120 s hang-killer.
mod checkpointer_retention_test;
mod checkpointer_shutdown_test;
mod checkpointer_skip_gates_test;
mod checkpointer_tail_cadence_test;
// b.dfh Epic 3 Task 7 — PRD AC-2, the per-outcome tail-capture matrix. Drives
// `capture_layer_tail` directly against hand-planted Valkey streams (the loop's
// capture fallback is unreachable for the terminal blocker arms), asserting
// `outcome_label()` + adapter state. Wall-clock-free.
mod checkpointer_tail_outcomes_test;
// b.dfh Epic 3 Task 7 — SDD SR-9.2, the duplicate-leader race. Invokes
// `capture_layer_tail` twice CONCURRENTLY (two loops would serialize on
// `le:checkpoint:leader`, so the window is inexpressible with loops), gating the
// listing seam through a recording store to make both shapes deterministic:
// identical-S writes byte-identical envelopes to one key, divergent-S forks into
// two keys whose head is the newer tail. No feature gate; no wall-clock waits
// beyond two bounded rendezvous.
mod checkpointer_tail_duplicate_leader_test;
// b.dfh Epic 5 Task 3 — SDD SR-9.2 for the chain-collapse RE-BASE, the twin of
// the tail file above. Invokes `rebase_layer_lineage` twice CONCURRENTLY with
// distinct `replica_id`s (two loops would serialize on `le:checkpoint:leader`)
// and RECORDS the puts rather than reading the survivor, reusing that file's
// three structural arguments by reference. Identical-S asserts BYTE equality on
// the base OMPF and the ext-refs sidecar, with the faithful-column comparison
// living only inside the failure path as a classifier; divergent-S forks into
// two bases at distinct keys and pins that neither gather consulted a
// generation outside its own walked chain. No Valkey (the unit takes a
// `&dyn CheckpointStore` and nothing else) and no sleeps; both listing gates
// are bounded.
mod checkpointer_rebase_duplicate_leader_test;

// Task t2.fgj.zc.56 Subtask .eg — retention-prune ext-blob cascade (SR-3.4)
// integration coverage. End-to-end the spawned checkpoint loop with a
// `RecordingCheckpointStore` wrapper that captures `delete_ext_blob` calls.
mod checkpointer_ext_cascade_test;

// Task t2.fgj.zc.56 Subtask .1n — fault-injection integration test covering
// AC #5: the cascade is recoverable from a transient
// `list_referenced_ext_hashes` failure. Gated on
// `test-checkpoint-fault-injection` because it installs the InMemory
// adapter's `with_fault_injector` seam, which only exists under that
// feature.
#[cfg(feature = "test-checkpoint-fault-injection")]
mod checkpointer_ext_cascade_fault_injection_test;

// Task t2.fgj.mh.fc Subtask .h7 — integration coverage for the SR-5.1
// periodic counters-sidecar write in the checkpoint leader loop
// (Epic `t1.fgj.mh`, Bee `b.fgj`), and for its SR-2.4 rule that a
// counters tick never lowers the stored sidecar (Plan b.tv3). Eleven
// scenarios share one `CountingCheckpointStore` boilerplate so the
// harness is single-instance per the Subtask body's bundling decision.
// Gated on `test-checkpoint-fault-injection` because scenarios #4 and #10
// install the `with_fault_injector` seam on `InMemoryCheckpointStore` to
// fail a `put_counters_sidecar` and a `get_counters_sidecar` once.
#[cfg(feature = "test-checkpoint-fault-injection")]
mod checkpointer_counters_sync_test;
