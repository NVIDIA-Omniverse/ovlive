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

//! The serving-process guard of the SCAN-family walks
//! (`live_edit::storage::scan_guard`, b.gns: the periodic reference-count
//! reconcile pass's since b.f3t, which gave the disaster-recovery Jobs'
//! walks completeness proofs instead) against a real store, through
//! the walk seam (`arm_walk_fault_for_test`, `test-fault-injection`): a
//! forged run_id stands in for a change of serving process, a forged
//! `CLUSTER NODES` reply for a node that does not serve its primary's slots,
//! a pinned node for a routing table that lists another, a refused pin
//! check for a table that dropped the pin, a forged end witness
//! (`EndWitness`) for a process change caught only by the end fingerprint,
//! and, for the coverage walk, a forged routing-table comparison for an
//! out-of-date table and a forged `CLUSTER SLOTS` read for a changed map.
//!
//! - **The cursor pager** (test 1): a `ZSCAN` walk of a table-encoded sorted
//!   set by small steps ends at the step whose witness names another
//!   process, having handed on only the steps before it; a step that fails
//!   transiently with the witness unchanged is retried in place (R1), and one
//!   that fails on every attempt ends the walk; a step that fails with a
//!   deterministic error (`WRONGTYPE`) ends the walk on its first attempt,
//!   with no witness re-read; a placeholder run_id or none is a missing
//!   witness.
//! - **A step's pipeline** (b.qgz): each guarded step is sent in one
//!   pipeline with its witnesses. A keyed walk's first pipeline that failed
//!   as a whole is resent whole and lands, or ends the walk once its
//!   attempts are spent, its witnesses read as reads 0 and 1; a first step
//!   whose own reply alone failed transiently (`StepReplyError`) keeps the
//!   start witness its pipeline led with, lands on its next attempt, and
//!   ends the walk when its R1 re-read names another process; an
//!   `INFO server` that answers an ACL refusal ends
//!   the walk with that witness's `Read` error, not the step's; and, with
//!   the walk's connection held in a server-side `BLPOP` between two steps,
//!   a later keyed step's `ZSCAN` and a guarded keyspace step's `SCAN` each
//!   reach the server with their `INFO server`, byte for byte, while a
//!   coverage walk's `SCAN` reaches it alone. A DR Job's keyspace walk
//!   (`KeyspaceWalk::start_job`, b.f3t) sends its first step alone too,
//!   with `COUNT 1000` where a coverage walk sends `COUNT 100` (T7). On a
//!   cluster (cluster-gated), an agreeing pair of Job walks returns every
//!   shard's keys once and reads no witness, and a Job walk fails on an open
//!   slot at its start or end, a pin whose myself line misses a slot at its
//!   start or end (`NotOwner`), a pin to an unconnected node, and an end
//!   slot map that differs from the start's (`SlotMap`); a pin that resolves
//!   to no node (`UnresolvedPin`) fails a guarded walk, a Job walk and an
//!   agreeing pair at the start (`NotOwner`), where a coverage walk leaves
//!   that primary unwalked and answers the same `NotOwner` after its keys.
//! - **The keyspace walk, standalone** (test 3): a clean walk returns every
//!   key; a change between two pages fails the walk; an end witness that
//!   names another process fails it at its end.
//! - **On a cluster** (cluster-gated, CI's `test-cluster:` job): a clean
//!   walk across the three shards returns every key (test 3); an open slot
//!   forged at the end fails it (test 3); a pinned node whose myself line
//!   misses one of its primary's slots is `NotOwner` at the start, before
//!   any step, and at the end, after the steps (test 15); a primary pinned
//!   to a node the client holds no connection for fails the walk at once,
//!   with no fallback to another node (test 17); a pin refused during an R1
//!   re-read, before a primary's start `CLUSTER NODES` or before a step's
//!   first send ends the walk with the refusal (test 21); and the third
//!   primary's end witness naming another process fails the walk at its end.
//!
//! - **The coverage walk** (b.hyw, `KeyspaceWalk::start_coverage`, the
//!   service's non-Job keyspace walks): standalone, it reads no witness, so a
//!   store whose `INFO server` shows no run_id, which fails a guarded walk,
//!   still gets every key (T3), and a step that fails transiently is sent
//!   again and lands; on a cluster (cluster-gated, CI's `test-cluster:`
//!   job), a forged myself line on the first primary is walked past and
//!   answered after the last key when it only leaves ownership unproven (an
//!   open slot, an unusable line), and leaves that primary unwalked,
//!   re-syncing the routing table, when it proves a non-owner (T7); a
//!   non-owner found by the end check is `NotOwner` at the end; a store
//!   fault after a recorded open slot is answered in its place; and a pin
//!   refused at the start fails the walk before any step. b.87a added the
//!   table check: with a pin's ownership unproven (no myself line), a
//!   routing-table comparison forged to differ (`TablePartitionDiffers`) at
//!   the start or the end is `StaleTable` there, after a re-sync, and loses
//!   coverage, while a walk whose pins were all proven (an open slot) makes
//!   no comparison; a walk with an unproven pin whose table matches still
//!   re-syncs it at its end, keeping its coverage, while a walk whose every
//!   pin is proven re-syncs nothing; on a client whose ACL user is denied
//!   `CLUSTER NODES`, every pin is unproven and the same table check holds;
//!   and the end `CLUSTER SLOTS` map, read with two primaries' names
//!   exchanged (`SlotMapPrimariesSwapped`), differs from the start's and is
//!   `SlotMap`, losing coverage.
//!
//! A real slot moved during a guarded walk is `scan_guard_disruptive_test`'s,
//! and a stale routing table, under the guarded walk and under the coverage
//! walk with its pins proven and unproven, `scan_guard_stale_table_test`'s.
//! The pure decisions (the parses, the coverage verdict, the table
//! comparison, the routes' options, the R1 verdict, and the reading of a
//! step's pipeline: one reply per command judged in command order, any
//! other count failing closed) and a coverage walk's answer given what it
//! recorded are `storage::scan_guard`'s unit tests; each consumer's exit is
//! its own module's (the DR Jobs' completeness proofs are
//! `storage::ext_reconcile`'s, `dr::job::execute::discard`'s and the DR Job
//! tests').
//!
//! Requires Valkey (`TEST_VALKEY_URL`); the cluster tests also
//! `LIVE_EDIT_TEST_CLUSTER_URL` and skip without it. The whole module needs
//! `test-fault-injection`, the seam's feature, and is compiled out without
//! it.
//!
//! # Fail-on-old
//!
//! Four standalone tests were proven red by in-place perturbations of
//! `storage::scan_guard` and `storage::retirement`, then green once
//! restored: with the guard's comparisons removed (`changed_between`
//! answering no change, the cursor pager's run_id comparison never true),
//! the pager test's crossed walk returns its members and the keyspace
//! test's second page answers keys; with the step sender (then `send_step`,
//! now `send_guarded_step`) given one attempt, the R1 test's walk ends on its
//! first injected failure; with the end witness comparison removed from
//! `check_end`, the end-witness test's walk completes. The pipeline tests
//! (b.qgz) name their own perturbations, each run in place against
//! `send_guarded_step`. The run_id parse the
//! missing-witness test rests on is proven in `storage::scan_guard`'s unit
//! tests. The cluster tests' perturbations, named on each, are not run in
//! this repository's sandbox, which has no cluster.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{ClientLike, ClusterInterface, KeysInterface, SortedSetsInterface};
use fred::types::config::Server;
use fred::types::{InfoKind, Value};

use live_edit::storage::client::ValkeyClient;
use live_edit::storage::scan_guard::{
    arm_walk_fault_for_test, collect_keyspace, collect_keyspace_agreed,
    forge_process_change_for_test, KeyspaceWalk, MyselfLineSeen, RoutingRefresh, RunIdSeen,
    ServingChange, SlotSet, WalkEnd, WalkError, WalkFault, WitnessSeen, DR_KEYSPACE_SCAN_COUNT,
    KEYSPACE_SCAN_COUNT, TEST_RUN_ID_A as RUN_A, TEST_RUN_ID_B as RUN_B, WALK_STEP_ATTEMPTS,
};
use live_edit::storage::test_support::{
    connect_test_client_or_skip, object_encoding_for_test, release_all, stall_pool,
    test_cluster_url, zscan_members_by_cursor_for_test, ConnStall, TestValkey, CONN_STALL_WAIT,
    TEST_CLUSTER_URL_VAR,
};

use crate::storage::acl_user_fixture::{
    acl_command, acl_setuser_args, acl_target_nodes, credentialed_url, AclUserGuard,
};

/// A walked sorted set's members: past the compact encoding's 128, so the
/// store keeps it as a table and a step answers about `COUNT` members.
const MEMBERS: usize = 200;

/// Each `ZSCAN` step's `COUNT`. Shared with `scan_guard_disruptive_test`.
pub(crate) const COUNT: u64 = 10;

/// The keyspace tests' keys: enough that a walk of the shared server by
/// `COUNT 100` steps takes more than one step.
const KEYSPACE_KEYS: usize = 300;

/// A bound on one whole walk, far above a healthy walk's milliseconds.
const WALK_BUDGET: Duration = Duration::from_secs(30);

/// `change` is a change of serving process on the `walk` walk of `target`,
/// from [`RUN_A`] to [`RUN_B`], and its text names both.
fn assert_process_change(change: &ServingChange, walk: &str, target: &str) {
    match change {
        ServingChange::Process {
            walk: w,
            target: t,
            before,
            after,
            ..
        } => assert_eq!(
            (*w, &**t, before.as_str(), after.as_str()),
            (walk, target, RUN_A, RUN_B)
        ),
        other => panic!("a change of serving process, got {other:?}"),
    }
    let text = change.to_string();
    for part in [target, RUN_A, RUN_B] {
        assert!(text.contains(part), "the change names {part:?}: {text}");
    }
}

/// Seed `key` with [`MEMBERS`] members, table-encoded. Shared with
/// `scan_guard_disruptive_test`.
pub(crate) async fn seed_sorted_set(tv: &TestValkey, key: &str) {
    let members: Vec<(f64, String)> = (0..MEMBERS)
        .map(|i| (i as f64, format!("m{i:03}")))
        .collect();
    let _: i64 = tv
        .client()
        .pool
        .zadd(key, None, None, false, false, members)
        .await
        .expect("seeding the sorted set must succeed");
    assert_eq!(
        object_encoding_for_test(tv.client(), key).await,
        "skiplist",
        "fixture: the set is past the compact encoding"
    );
}

/// The guarded `ZSCAN` walk of `key` by [`COUNT`] steps: what it returned,
/// and how many steps it handed on.
async fn walk_counting(tv: &TestValkey, key: &str) -> (Result<Vec<String>, WalkError>, usize) {
    let steps = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&steps);
    let walked = tokio::time::timeout(
        WALK_BUDGET,
        zscan_members_by_cursor_for_test(tv.client(), key, COUNT, move |_members| {
            counter.fetch_add(1, Ordering::SeqCst);
            async {}
        }),
    )
    .await
    .expect("the walk ends within its budget");
    (walked, steps.load(Ordering::SeqCst))
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 1: the cursor pager
// ═════════════════════════════════════════════════════════════════════════════

/// A `ZSCAN` walk whose witness names another process after its third step
/// ends there with the change, naming the key and both run_ids, having
/// handed on only its first two steps; the same walk unarmed hands on every
/// step and returns every member. The pager hands a step's members on only
/// once that step's witness matched the walk's first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cursor_pager_ends_a_walk_at_the_step_whose_witness_names_another_process() {
    let tv = TestValkey::setup().await;
    let key = format!("{}scan-guard:pager", tv.prefix());
    seed_sorted_set(&tv, &key).await;

    let (clean, clean_steps) = walk_counting(&tv, &key).await;
    let clean: BTreeSet<String> = clean
        .expect("control: the unarmed walk completes")
        .into_iter()
        .collect();
    assert_eq!(clean.len(), MEMBERS, "control: every member is returned");
    assert!(
        clean_steps > 3,
        "fixture: the walk takes more than three steps"
    );

    // Reads 0 (the start), 1 and 2 (after steps 0 and 1) alike; read 3,
    // after step 2, names another process.
    let forged = forge_process_change_for_test(&key, 3);
    let (crossed, steps) = walk_counting(&tv, &key).await;
    drop(forged);

    match crossed {
        Err(WalkError::Changed(change)) => assert_process_change(&change, "ZSCAN", &key),
        other => panic!("the walk ends with the change, got {other:?}"),
    }
    assert_eq!(
        steps, 2,
        "only the two steps before the change are handed on"
    );
}

/// R1: a step that fails transiently (the seam's injected `IO` error) with
/// the witness unchanged is sent again at the same cursor. Failing twice, it lands on its third attempt and the walk
/// returns every member; failing three times, the walk ends with the step's
/// own error, the last attempt's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cursor_pager_retries_a_failed_step_on_an_unchanged_witness_up_to_three_attempts() {
    let tv = TestValkey::setup().await;
    let key = format!("{}scan-guard:retry", tv.prefix());
    seed_sorted_set(&tv, &key).await;

    let armed = arm_walk_fault_for_test(&key, WalkFault::StepError { step: 1, times: 2 });
    let (retried, _) = walk_counting(&tv, &key).await;
    drop(armed);
    let retried: BTreeSet<String> = retried
        .expect("two failed attempts and an unchanged witness: the step lands on its third")
        .into_iter()
        .collect();
    assert_eq!(retried.len(), MEMBERS, "the walk returns every member");

    let armed = arm_walk_fault_for_test(&key, WalkFault::StepError { step: 1, times: 3 });
    let (exhausted, steps) = walk_counting(&tv, &key).await;
    drop(armed);
    match exhausted {
        Err(WalkError::Step {
            walk,
            target,
            source,
        }) => {
            assert_eq!((walk, &*target), ("ZSCAN", key.as_str()));
            assert!(
                source
                    .details()
                    .contains("test seam: step 1 of a ZSCAN walk failed before it was sent"),
                "the last attempt's error: {source}"
            );
        }
        other => panic!("three failed attempts end the walk, got {other:?}"),
    }
    assert_eq!(steps, 1, "the step before the failing one was handed on");
}

/// R1 retries only a transient step error. A `ZSCAN` walk of a key that
/// holds a string answers `WRONGTYPE`, a deterministic error: the walk ends
/// on its first attempt with that `Step` error, reading no witness again.
/// A change of serving process is forged at read 1, the read an R1 retry's
/// re-read would be; the walk ends with the step's error, not the change,
/// so no re-read was made. (The step's after-step witness rode in its
/// pipeline (b.qgz); that it is left unread once the step answered an
/// error is not visible here, since reading it would leave the same `Step`
/// error. `storage::scan_guard`'s unit test
/// `read_step_pipeline_judges_each_reply_as_its_own_commands` pins it.)
///
/// Fail-on-old (in-place perturbation, red then green once restored): with
/// `step_error_is_transient` answering true for every error, the failed step
/// is followed by its R1 re-read, which reads the forged change, and the
/// walk ends with `Changed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cursor_pager_ends_a_walk_on_its_first_attempt_when_a_step_answers_wrongtype() {
    let tv = TestValkey::setup().await;
    let key = format!("{}scan-guard:wrongtype", tv.prefix());
    let _: () = tv
        .client()
        .pool
        .set(&key, "not-a-sorted-set", None, None, false)
        .await
        .expect("SET must succeed");

    // Read 0 (the start) reports A; read 1, an R1 re-read's, would report B.
    let forged = forge_process_change_for_test(&key, 1);
    let (walked, steps) = walk_counting(&tv, &key).await;
    drop(forged);

    match walked {
        Err(WalkError::Step {
            walk,
            target,
            source,
        }) => {
            assert_eq!((walk, &*target), ("ZSCAN", key.as_str()));
            assert!(
                source.details().contains("WRONGTYPE"),
                "the step's own error: {source}"
            );
        }
        other => panic!(
            "a deterministic error ends the walk on its first attempt, with no re-read, got \
             {other:?}"
        ),
    }
    assert_eq!(steps, 0, "no step was handed on");
}

/// R3: a walk whose `INFO server` reports a placeholder run_id (`"0"`), an
/// empty or 39-character one, or no `run_id` line, at its start or after a
/// step, ends with a missing witness that says what was seen, and hands on
/// no step from then on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cursor_pager_ends_a_walk_whose_witness_reports_no_usable_run_id() {
    let tv = TestValkey::setup().await;
    let key = format!("{}scan-guard:missing", tv.prefix());
    seed_sorted_set(&tv, &key).await;
    let short = "a".repeat(39);

    for (read, run_id, seen, handed_on) in [
        (0, Some("0"), RunIdSeen::Value("0".to_owned()), 0),
        (0, None, RunIdSeen::NoLine, 0),
        (1, Some(""), RunIdSeen::Value(String::new()), 0),
        (2, Some(short.as_str()), RunIdSeen::Value(short.clone()), 1),
    ] {
        let armed = arm_walk_fault_for_test(
            &key,
            WalkFault::RunId {
                read,
                run_id: run_id.map(str::to_owned),
            },
        );
        let (walked, steps) = walk_counting(&tv, &key).await;
        drop(armed);
        match walked {
            Err(WalkError::WitnessMissing(missing)) => {
                assert_eq!(
                    missing.seen,
                    WitnessSeen::RunId(seen.clone()),
                    "read {read}"
                );
                assert_eq!((missing.walk, &*missing.target), ("ZSCAN", key.as_str()));
                assert!(missing.to_string().contains("`INFO server`"));
            }
            other => panic!("read {read} ({seen}): a missing witness, got {other:?}"),
        }
        assert_eq!(steps, handed_on, "read {read}: steps handed on");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// A step's pipeline (b.qgz)
// ═════════════════════════════════════════════════════════════════════════════

/// A keyed walk's first pipeline, `[INFO server; ZSCAN; INFO server]`, that
/// fails as a whole (the seam's `StepError` on step 0: nothing sent, the
/// shape a lost connection gives) read no start witness, so the next
/// attempt resends it whole, its start witness included. Failing once, or
/// on every attempt but its last ([`WALK_STEP_ATTEMPTS`] − 1), the walk
/// returns every member; failing on all [`WALK_STEP_ATTEMPTS`], it ends with
/// the step's own error, the last attempt's, having handed on no step. The
/// failed pipeline reads no witness, so the resent pipeline's witnesses are
/// reads 0 and 1 (the `WalkFault::RunId` numbering): failing once with a
/// change forged at read 1, the walk ends with the change from the resent
/// start witness (read 0, [`RUN_A`]) to its after-step witness (read 1,
/// [`RUN_B`]), having handed on no step.
///
/// Fail-on-old (in-place perturbations of `scan_guard::send_guarded_step`,
/// red then green once restored): with only the first attempt leading with
/// the start witness (a resent first step sent as `[ZSCAN; INFO server]`),
/// the resent step lands with no start witness and the walk fails closed
/// with a `Protocol` `Step` error, not every member. With a witness read
/// alone after the failed pipeline, the resent pipeline's witnesses are
/// reads 1 and 2, and the forged row ends with a change from [`RUN_B`] to
/// the server's own run_id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cursor_pager_resends_a_first_pipeline_that_failed_whole_with_its_start_witness() {
    let tv = TestValkey::setup().await;
    let key = format!("{}scan-guard:resend", tv.prefix());
    seed_sorted_set(&tv, &key).await;

    for times in [1, WALK_STEP_ATTEMPTS - 1] {
        let armed = arm_walk_fault_for_test(&key, WalkFault::StepError { step: 0, times });
        let (resent, _) = walk_counting(&tv, &key).await;
        drop(armed);
        let resent: BTreeSet<String> = resent
            .unwrap_or_else(|e| {
                panic!("{times} failed first pipeline(s): it is resent whole and lands, got {e:?}")
            })
            .into_iter()
            .collect();
        assert_eq!(
            resent.len(),
            MEMBERS,
            "{times} failed first pipeline(s): every member"
        );
    }

    let armed = arm_walk_fault_for_test(
        &key,
        WalkFault::StepError {
            step: 0,
            times: WALK_STEP_ATTEMPTS,
        },
    );
    let (exhausted, steps) = walk_counting(&tv, &key).await;
    drop(armed);
    match exhausted {
        Err(WalkError::Step {
            walk,
            target,
            source,
        }) => {
            assert_eq!((walk, &*target), ("ZSCAN", key.as_str()));
            assert!(
                source
                    .details()
                    .contains("test seam: step 0 of a ZSCAN walk failed before it was sent"),
                "the last attempt's error: {source}"
            );
        }
        other => panic!("every attempt failed: the walk ends with the step's error, got {other:?}"),
    }
    assert_eq!(steps, 0, "no step was handed on");

    let forged = forge_process_change_for_test(&key, 1);
    let failing = arm_walk_fault_for_test(&key, WalkFault::StepError { step: 0, times: 1 });
    let (crossed, steps) = walk_counting(&tv, &key).await;
    drop((forged, failing));
    match crossed {
        Err(WalkError::Changed(change)) => assert_process_change(&change, "ZSCAN", &key),
        other => panic!(
            "the resent pipeline's witnesses are reads 0 and 1: the change ends the walk, got \
             {other:?}"
        ),
    }
    assert_eq!(steps, 0, "the forged row hands on no step");
}

/// A keyed walk's first step whose own reply alone failed transiently in
/// its pipeline (the seam's `StepReplyError`: the step answered a `LOADING`
/// reply, both witnesses answered) keeps the start witness its pipeline led
/// with (read 0), re-reads the witness alone against it (R1, read 1) and
/// sends its next attempt as `[ZSCAN; INFO server]`, returning the kept
/// start witness with the landed step: failing once, the walk returns every
/// member. With a change forged at read 1, the R1 re-read names another
/// process than the kept start witness and the walk ends with the change,
/// from [`RUN_A`] to [`RUN_B`], having handed on no step. With the change
/// forged at read 2 instead, the R1 re-read (read 1) is unchanged and the
/// next attempt, `[ZSCAN; INFO server]` with no start witness of its own,
/// reads its after-step witness as read 2: the walk ends with the change
/// from the kept start witness, [`RUN_A`], to [`RUN_B`], having handed on no
/// step.
///
/// Fail-on-old (in-place perturbations of `scan_guard::send_guarded_step`,
/// red then green once restored): with the landed step returning only the
/// start witness its own pipeline led with (`lead: read`, not
/// `read.or(lead)`), the clean row's resent step lands with no start witness
/// and fails closed as a `Protocol` `Step` error. With the kept start
/// witness dropped (a failed attempt's lead not kept), no R1 re-read is
/// made, the next attempt leads again (its witnesses reads 1 and 2), and the
/// read-1 row's change runs from [`RUN_B`] to the server's own run_id. With
/// the next attempt leading again although the start was kept (`leads`
/// without its `lead.is_none()` term), the read-2 row's next attempt leads
/// with read 2 ([`RUN_B`]) and its after-step witness is read 3, so its
/// change runs from [`RUN_B`] to the server's own run_id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cursor_pager_keeps_the_start_witness_of_a_first_step_whose_reply_alone_failed() {
    let tv = TestValkey::setup().await;
    let key = format!("{}scan-guard:reply-failed", tv.prefix());
    seed_sorted_set(&tv, &key).await;

    let failing = arm_walk_fault_for_test(&key, WalkFault::StepReplyError { step: 0, times: 1 });
    let (retried, _) = walk_counting(&tv, &key).await;
    drop(failing);
    let retried: BTreeSet<String> = retried
        .unwrap_or_else(|e| {
            panic!("a first step whose reply failed once lands on its next attempt, got {e:?}")
        })
        .into_iter()
        .collect();
    assert_eq!(retried.len(), MEMBERS, "the walk returns every member");

    let forged = forge_process_change_for_test(&key, 1);
    let failing = arm_walk_fault_for_test(&key, WalkFault::StepReplyError { step: 0, times: 1 });
    let (crossed, steps) = walk_counting(&tv, &key).await;
    drop((forged, failing));
    match crossed {
        Err(WalkError::Changed(change)) => assert_process_change(&change, "ZSCAN", &key),
        other => panic!(
            "the R1 re-read (read 1) against the kept start witness (read 0) ends the walk, got \
             {other:?}"
        ),
    }
    assert_eq!(steps, 0, "no step was handed on");

    // The next attempt's shape: reads 0 (the kept start) and 1 (the R1
    // re-read) report A; read 2, the next attempt's after-step witness,
    // reports B.
    let forged = forge_process_change_for_test(&key, 2);
    let failing = arm_walk_fault_for_test(&key, WalkFault::StepReplyError { step: 0, times: 1 });
    let (crossed, steps) = walk_counting(&tv, &key).await;
    drop((forged, failing));
    match crossed {
        Err(WalkError::Changed(change)) => assert_process_change(&change, "ZSCAN", &key),
        other => panic!(
            "the next attempt, [ZSCAN; INFO server], reads its after-step witness as read 2, \
             compared with the kept start witness (read 0): the change ends the walk, got \
             {other:?}"
        ),
    }
    assert_eq!(steps, 0, "the next attempt's members are not handed on");
}

/// The one command an ACL-denied fixture's user may not run
/// ([`acl_denied_fixture`]).
#[derive(Clone, Copy)]
enum AclDenied {
    /// `INFO` (`-info`): a guarded walk's witness.
    Info,
    /// `CLUSTER NODES` (`-cluster|nodes`): a keyspace walk's myself line.
    ClusterNodes,
}

impl AclDenied {
    /// The `ACL SETUSER` rule that denies it.
    fn rule(self) -> &'static str {
        match self {
            Self::Info => "-info",
            Self::ClusterNodes => "-cluster|nodes",
        }
    }

    /// The command, as the fixture's messages name it.
    fn command(self) -> &'static str {
        match self {
            Self::Info => "`INFO server`",
            Self::ClusterNodes => "`CLUSTER NODES`",
        }
    }

    /// Send the denied command on `client`: the store's answer, as text.
    async fn probe(self, client: &fred::clients::Client) -> Result<String, fred::error::Error> {
        match self {
            Self::Info => client.info::<String>(Some(InfoKind::Server)).await,
            Self::ClusterNodes => client.cluster_nodes::<String>().await,
        }
    }
}

/// A fixture on `url` (named by the variable `url_var`, which must be a bare
/// `scheme` URL) whose every connection authenticates as an ACL user that
/// can run everything but `denied` (`+@all` less its rule), paired with the
/// guard that deletes the user from every primary on every exit, a panic
/// included (`crate::storage::acl_user_fixture`, shared with
/// `b_bhb_tests`). The user, named `user_prefix` and a UUID, is created on
/// every primary through a default-user fixture; the default user is never
/// changed. With `cluster`, the URL must name a cluster (asserted).
///
/// Bind the pair as `let Some((_guard, restricted)) = …`, so the restricted
/// fixture (and its key clean-up) drops before the guard: `ACL DELUSER`
/// disconnects every client authenticated as the user.
///
/// `None`, after a `SKIP:` line, when the URL already carries credentials
/// or is not a bare `scheme` URL (an injected credential would change what
/// the URL means), or when the store refuses `ACL SETUSER`. A store that
/// still answers `denied` to the user fails the test: it would pass
/// vacuously.
async fn acl_denied_fixture(
    url: String,
    url_var: &str,
    scheme: &str,
    cluster: bool,
    user_prefix: &str,
    denied: AclDenied,
) -> Option<(AclUserGuard, TestValkey)> {
    let user = format!("{user_prefix}_{}", uuid::Uuid::new_v4().simple());
    let Some(restricted_url) = credentialed_url(&url, scheme, &user) else {
        eprintln!(
            "SKIP: {url_var} is not a bare `{scheme}host:port`, so an ACL user's credential \
             cannot be injected into it"
        );
        return None;
    };

    let control = TestValkey::setup_with_url(url, 1).await;
    if cluster {
        assert!(
            control.client().pool.next().is_clustered(),
            "{url_var} names a cluster"
        );
    }
    let nodes = acl_target_nodes(control.client().pool.next());
    let mut refused = None;
    for node in &nodes {
        if let Err(e) = acl_command(
            control.client().pool.next(),
            node.as_ref(),
            acl_setuser_args(&user, denied.rule()),
        )
        .await
        {
            refused = Some(e);
            break;
        }
    }
    // Before the refusal check: a user created on one node and refused on
    // another must still be deleted.
    let guard = AclUserGuard {
        control,
        nodes,
        user,
    };
    if let Some(e) = refused {
        eprintln!("SKIP: `ACL SETUSER` was refused ({e}), so no ACL user can be created");
        return None;
    }
    let restricted = TestValkey::setup_with_url(restricted_url, 1).await;
    let command = denied.command();
    match denied.probe(restricted.client().pool.next()).await {
        Err(e) => {
            let text = e.to_string();
            assert!(
                text.contains("NOPERM") || text.contains("no permissions"),
                "fixture: {command} is refused for an ACL reason, got {text}"
            );
        }
        Ok(answer) => panic!(
            "fixture: the store answered {command} to a user denied it, so the test would pass \
             vacuously: {answer}"
        ),
    }
    Some((guard, restricted))
}

/// [`acl_denied_fixture`] on `TEST_VALKEY_URL` (a bare `redis://` URL),
/// denied `INFO`.
async fn info_denied_fixture() -> Option<(AclUserGuard, TestValkey)> {
    let url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned());
    acl_denied_fixture(
        url,
        "TEST_VALKEY_URL",
        "redis://",
        false,
        "scan_guard_noinfo",
        AclDenied::Info,
    )
    .await
}

/// Per-command attribution (b.qgz). A keyed walk on a client whose ACL user
/// is denied `INFO` sends its first step as `[INFO server; ZSCAN;
/// INFO server]`; the server answers both witnesses with an error reply and
/// the step with its members. Read by command index (fred's `try_all`), the
/// start witness's error is that witness's: the walk ends with
/// `WalkError::Read` of `INFO server`, carrying the ACL refusal, not with a
/// `Step` error, and hands on no step.
///
/// Fail-on-old (in-place perturbation of `scan_guard::send_guarded_step`,
/// red then green once restored): with the pipeline read by fred's `all()`,
/// which answers the first error reply for the whole pipeline, the refusal
/// is read as the pipeline's whole failure and the walk ends with a `Step`
/// error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cursor_pager_ends_a_walk_with_the_witness_read_error_when_info_server_is_refused() {
    let Some((_guard, restricted)) = info_denied_fixture().await else {
        return;
    };
    let key = format!("{}scan-guard:info-denied", restricted.prefix());
    seed_sorted_set(&restricted, &key).await;

    let (walked, steps) = walk_counting(&restricted, &key).await;
    match walked {
        Err(WalkError::Read { op, source }) => {
            assert_eq!(op, "INFO server");
            let text = source.to_string();
            assert!(
                text.contains("NOPERM") || text.contains("no permissions"),
                "the witness's own ACL refusal: {text}"
            );
        }
        other => panic!(
            "a refused witness ends the walk with its Read error, not the step's, got {other:?}"
        ),
    }
    assert_eq!(steps, 0, "no step was handed on");
}

/// How long the pipeline byte tests watch a stalled connection, once the
/// bytes they wait for have arrived, for any more.
const QUIET_WINDOW: Duration = Duration::from_millis(300);

/// The command timeout of the client the pipeline byte tests walk on
/// ([`walker_config`]): far past [`CONN_STALL_WAIT`] plus [`QUIET_WINDOW`],
/// the longest a test holds the walk's connection, so a stalled step never
/// times out. A step that timed out would be followed by its R1 re-read, an
/// `INFO server` of its own, which would stand in for the witness these
/// tests look for in the step's pipeline.
const WALKER_COMMAND_TIMEOUT_MS: u64 = 60_000;

/// The config of the client the pipeline byte tests walk on: one data-pool
/// connection, the one they stall, with [`WALKER_COMMAND_TIMEOUT_MS`].
fn walker_config() -> live_edit::config::Config {
    let mut cfg = crate::common::test_config();
    cfg.valkey_pool_size = 1;
    cfg.valkey_command_timeout_ms = WALKER_COMMAND_TIMEOUT_MS;
    cfg
}

/// One command as the server reads it: a RESP array of bulk strings.
fn frame_len(args: &[&str]) -> u64 {
    let header = format!("*{}\r\n", args.len()).len();
    let body: usize = args
        .iter()
        .map(|arg| format!("${}\r\n{arg}\r\n", arg.len()).len())
        .sum();
    (header + body) as u64
}

/// `INFO server` as the server reads it: a guarded step's witness.
fn info_server_frame_len() -> u64 {
    frame_len(&["INFO", "server"])
}

/// Every length `step`, a step command given its cursor, can have for a
/// cursor of 1 to 20 decimal digits (a cursor is a `u64`), plus `extra`.
/// The step's cursor after the first is the store's, so it is not known in
/// advance; its length changes the frame by at most 20 bytes, under the 26
/// of an `INFO server`, so the lengths with `extra` = 0 and with `extra` =
/// one `INFO server` never meet (asserted).
fn step_lengths(step: impl Fn(&str) -> u64, extra: u64) -> Vec<u64> {
    let lengths: Vec<u64> = (1..=20).map(|digits| step(&"9".repeat(digits))).collect();
    assert!(
        lengths[19] < lengths[0] + info_server_frame_len(),
        "fixture: a step with its witness is longer than any step alone"
    );
    lengths.into_iter().map(|length| length + extra).collect()
}

/// `SCAN <cursor> MATCH <pattern> COUNT 100`, a keyspace walk's step.
fn scan_step_len(pattern: &str) -> impl Fn(&str) -> u64 + '_ {
    let count = KEYSPACE_SCAN_COUNT.to_string();
    move |cursor| frame_len(&["SCAN", cursor, "MATCH", pattern, "COUNT", count.as_str()])
}

/// Wait (bounded by `CONN_STALL_WAIT`) for at least `at_least` bytes past
/// `baseline` to reach the server on the stalled connection, then, once
/// they did, [`QUIET_WINDOW`] more; release every stall; return the bytes
/// that arrived. Released before the caller asserts, so a failed
/// assertion leaves no connection stalled.
async fn bytes_while_stalled(stalls: Vec<ConnStall>, baseline: u64, at_least: u64) -> u64 {
    let deadline = tokio::time::Instant::now() + CONN_STALL_WAIT;
    let arrived = loop {
        let now = stalls[0].net_in().await.expect("the stalled connection") - baseline;
        if now >= at_least || tokio::time::Instant::now() >= deadline {
            break now;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    if arrived >= at_least {
        tokio::time::sleep(QUIET_WINDOW).await;
    }
    let arrived = stalls[0].net_in().await.expect("the stalled connection") - baseline;
    release_all(stalls).await;
    arrived
}

/// Stall `walker`'s one data-pool connection, observed from the fixture's
/// data pool, on a key under the fixture's prefix: the stalls, and the bytes
/// the server had read from the connection when it blocked.
async fn stall_walk_connection(tv: &TestValkey, walker: &ValkeyClient) -> (Vec<ConnStall>, u64) {
    let observer = tv.client().pool.next().clone();
    let prefix = format!("{}conn-stall:walker:", tv.prefix());
    let stalls = stall_pool(&walker.pool, &observer, &prefix).await;
    assert_eq!(stalls.len(), 1, "fixture: one walker connection");
    let baseline = stalls[0].net_in().await.expect("the stalled connection");
    (stalls, baseline)
}

/// A later keyed step's witness is written with the step (b.qgz). A `ZSCAN`
/// walk by the cursor pager on a client of one connection
/// ([`walker_config`]), held in a server-side `BLPOP` from its first step's
/// `after_step` hook: the pager's next step, `[ZSCAN; INFO server]`, reaches
/// the server whole — exactly the `ZSCAN` and its `INFO server`, byte for
/// byte — and nothing more for [`QUIET_WINDOW`]. Released, the walk returns
/// every member.
///
/// Fail-on-old (in-place perturbation of `scan_guard::send_guarded_step`,
/// red then green once restored): with each of a step's commands sent as
/// its own round trip, as before b.qgz, only the `ZSCAN` reaches the
/// stalled server, and the step's `INFO server` never does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cursor_pager_sends_a_later_steps_witness_in_one_pipeline_with_the_step() {
    let tv = TestValkey::setup().await;
    let Some(walker) = connect_test_client_or_skip(&walker_config()).await else {
        return;
    };
    let key = format!("{}scan-guard:pipelined", tv.prefix());
    seed_sorted_set(&tv, &key).await;
    let count = COUNT.to_string();
    let with_witness = step_lengths(
        |cursor| frame_len(&["ZSCAN", key.as_str(), cursor, "COUNT", count.as_str()]),
        info_server_frame_len(),
    );

    // The first `after_step` call stalls the walk's connection and hands
    // the stall to the watcher; later calls do nothing.
    let (stalled_tx, stalled_rx) = tokio::sync::oneshot::channel();
    let stalled_tx = std::sync::Mutex::new(Some(stalled_tx));
    let (tv_ref, walker_ref) = (&tv, &walker);
    let walk = zscan_members_by_cursor_for_test(&walker, &key, COUNT, |_members| {
        let first = stalled_tx.lock().expect("not poisoned").take();
        async move {
            if let Some(stalled_tx) = first {
                let stalled = stall_walk_connection(tv_ref, walker_ref).await;
                stalled_tx
                    .send(stalled)
                    .unwrap_or_else(|_| panic!("the watcher is waiting"));
            }
        }
    });
    let watch = async {
        let (stalls, baseline) = stalled_rx
            .await
            .expect("the walk stalled its connection after its first step");
        bytes_while_stalled(stalls, baseline, with_witness[0]).await
    };
    let (walked, sent) = tokio::time::timeout(WALK_BUDGET, async { tokio::join!(walk, watch) })
        .await
        .expect("the walk ends within its budget");

    assert!(
        with_witness.contains(&sent),
        "the next step's ZSCAN and its INFO server reach the server together, and nothing more: \
         {sent} B, not one of {with_witness:?}"
    );
    assert_eq!(
        walked.expect("released, the walk completes").len(),
        MEMBERS,
        "released, the walk returns every member"
    );
}

/// A guarded keyspace step's witness is written with the step (b.qgz). A
/// standalone keyspace walk on a client of one connection
/// ([`walker_config`]), held in a server-side `BLPOP` after its first page:
/// its next step, `[SCAN; INFO server]`, reaches the server whole — exactly
/// the `SCAN` and its `INFO server`, byte for byte — and nothing more for
/// [`QUIET_WINDOW`]. Released, the page answers.
///
/// Fail-on-old (in-place perturbation of `scan_guard::send_guarded_step`,
/// red then green once restored): with each of a step's commands sent as
/// its own round trip, as before b.qgz, only the `SCAN` reaches the stalled
/// server.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keyspace_walk_sends_a_steps_witness_in_one_pipeline_with_the_step() {
    let tv = TestValkey::setup().await;
    let Some(walker) = connect_test_client_or_skip(&walker_config()).await else {
        return;
    };
    let prefix = format!("{}scan-guard:keyspace-pipelined:", tv.prefix());
    let pattern = format!("{prefix}*");
    seed_keyspace(&tv, &prefix).await;
    let with_witness = step_lengths(scan_step_len(&pattern), info_server_frame_len());

    let mut walk = KeyspaceWalk::start(walker.pool.next(), &pattern)
        .await
        .expect("the walk starts");
    let first = walk.next_page().await.expect("the first page");
    assert!(first.is_some(), "fixture: the walk has a first page");
    let (stalls, baseline) = stall_walk_connection(&tv, &walker).await;
    let (second, sent) = tokio::time::timeout(WALK_BUDGET, async {
        tokio::join!(
            walk.next_page(),
            bytes_while_stalled(stalls, baseline, with_witness[0])
        )
    })
    .await
    .expect("the step ends within its budget");

    assert!(
        with_witness.contains(&sent),
        "the step's SCAN and its INFO server reach the server together, and nothing more: {sent} \
         B, not one of {with_witness:?}"
    );
    assert!(
        matches!(second, Ok(Some(_))),
        "released, the second page answers: {second:?}"
    );
}

/// A coverage walk sends no witness (b.qgz): its step is sent alone. A
/// standalone coverage walk on a client of one connection
/// ([`walker_config`]), held in a server-side `BLPOP` after its first page:
/// its next step reaches the server as exactly one `SCAN`, byte for byte,
/// with no `INFO server`, and nothing more for [`QUIET_WINDOW`]. Released,
/// the page answers.
///
/// Fail-on-old (in-place perturbation of `scan_guard::send_guarded_step`,
/// red then green once restored): with a coverage step sent as a keyed
/// walk's first (`WitnessMode::None` queueing the witnesses `Lead` does),
/// the `SCAN` reaches the server between two `INFO server`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn coverage_walk_sends_its_step_alone_with_no_info_server() {
    let tv = TestValkey::setup().await;
    let Some(walker) = connect_test_client_or_skip(&walker_config()).await else {
        return;
    };
    let prefix = format!("{}scan-guard:coverage-alone:", tv.prefix());
    let pattern = format!("{prefix}*");
    seed_keyspace(&tv, &prefix).await;
    let alone = step_lengths(scan_step_len(&pattern), 0);

    let mut walk = KeyspaceWalk::start_coverage(walker.pool.next(), &pattern)
        .await
        .expect("the coverage walk starts");
    let first = walk.next_page().await.expect("the first page");
    assert!(first.is_some(), "fixture: the walk has a first page");
    let (stalls, baseline) = stall_walk_connection(&tv, &walker).await;
    let (second, sent) = tokio::time::timeout(WALK_BUDGET, async {
        tokio::join!(
            walk.next_page(),
            bytes_while_stalled(stalls, baseline, alone[0])
        )
    })
    .await
    .expect("the step ends within its budget");

    assert!(
        alone.contains(&sent),
        "the coverage step's SCAN reaches the server alone, with no INFO server: {sent} B, not \
         one of {alone:?}"
    );
    assert!(
        matches!(second, Ok(Some(_))),
        "released, the second page answers: {second:?}"
    );
}

/// T7 (b.f3t): a DR Job's keyspace walk (`KeyspaceWalk::start_job`) sends
/// `SCAN … COUNT 1000` (`DR_KEYSPACE_SCAN_COUNT`), alone, with no witness,
/// where a coverage walk sends `COUNT 100` (`KEYSPACE_SCAN_COUNT`, as a
/// guarded walk does: `keyspace_walk_sends_a_steps_witness_in_one_pipeline_with_the_step`).
/// Each standalone walk on a client of one connection ([`walker_config`]),
/// started and then held in a server-side `BLPOP` before its first step:
/// its first step reaches the server as exactly `SCAN 0 MATCH <pattern>
/// COUNT <n>`, byte for byte, and nothing more for [`QUIET_WINDOW`] — so a
/// Job walk reads nothing before its first step, as a coverage walk reads
/// nothing. Released, the page answers.
///
/// Fail-on-old (in-place perturbations, red then green once restored): with
/// a Job walk's steps sent with `KEYSPACE_SCAN_COUNT` (in
/// `KeyspaceGuard::scan_count`), its first step is one byte short (123 B,
/// not 124); with `DR_KEYSPACE_SCAN_COUNT` itself set to 100, the constant
/// pin fails first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn job_walk_sends_its_step_alone_with_count_1000_where_a_coverage_walk_sends_100() {
    assert_eq!(
        (DR_KEYSPACE_SCAN_COUNT, KEYSPACE_SCAN_COUNT),
        (1_000, 100),
        "the two COUNTs this test pins on the wire"
    );
    let tv = TestValkey::setup().await;
    let Some(walker) = connect_test_client_or_skip(&walker_config()).await else {
        return;
    };
    let prefix = format!("{}scan-guard:job-count:", tv.prefix());
    let pattern = format!("{prefix}*");
    seed_keyspace(&tv, &prefix).await;

    for (case, count) in [("a Job walk", "1000"), ("a coverage walk", "100")] {
        let first_step = frame_len(&["SCAN", "0", "MATCH", &pattern, "COUNT", count]);
        let mut walk = if count == "1000" {
            KeyspaceWalk::start_job(walker.pool.next(), &pattern).await
        } else {
            KeyspaceWalk::start_coverage(walker.pool.next(), &pattern).await
        }
        .unwrap_or_else(|e| panic!("{case}: the walk starts, got {e}"));
        let (stalls, baseline) = stall_walk_connection(&tv, &walker).await;
        let (first, sent) = tokio::time::timeout(WALK_BUDGET, async {
            tokio::join!(
                walk.next_page(),
                bytes_while_stalled(stalls, baseline, first_step)
            )
        })
        .await
        .expect("the step ends within its budget");

        assert_eq!(
            sent, first_step,
            "{case}: its first step reaches the server as exactly SCAN 0 MATCH <pattern> COUNT \
             {count}, alone"
        );
        assert!(
            matches!(first, Ok(Some(_))),
            "{case}: released, the first page answers: {first:?}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 3: the keyspace walk, standalone
// ═════════════════════════════════════════════════════════════════════════════

/// Seed [`KEYSPACE_KEYS`] keys under `prefix`; return them.
async fn seed_keyspace(tv: &TestValkey, prefix: &str) -> BTreeSet<String> {
    let keys: BTreeSet<String> = (0..KEYSPACE_KEYS)
        .map(|i| format!("{prefix}{i:04}"))
        .collect();
    for key in &keys {
        let _: () = tv
            .client()
            .pool
            .set(key, "1", None, None, false)
            .await
            .expect("SET must succeed");
    }
    keys
}

/// Every key a page holds, as text.
fn as_strings(keys: Vec<fred::types::Key>) -> Vec<String> {
    keys.into_iter()
        .map(|key| key.as_str().expect("a UTF-8 key").to_owned())
        .collect()
}

/// A keyspace walk on a standalone store returns every key its pattern
/// matches, over more than one step; a change of serving process forged
/// between its first and second pages ends it at the second, naming the
/// pattern and both run_ids.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keyspace_walk_returns_every_key_and_ends_at_a_change_between_two_pages() {
    let tv = TestValkey::setup().await;
    let prefix = format!("{}scan-guard:keyspace:", tv.prefix());
    let pattern = format!("{prefix}*");
    let written = seed_keyspace(&tv, &prefix).await;
    let client = tv.client().pool.next();

    let mut walk = KeyspaceWalk::start(client, &pattern)
        .await
        .expect("control: the walk starts");
    let mut found = BTreeSet::new();
    let mut pages = 0;
    while let Some(page) = tokio::time::timeout(WALK_BUDGET, walk.next_page())
        .await
        .expect("a page within the budget")
        .expect("control: every page")
    {
        pages += 1;
        found.extend(as_strings(page));
    }
    assert_eq!(found, written, "control: every key is returned");
    assert!(pages > 1, "fixture: the walk takes more than one step");

    // Reads 0 (the start) and 1 (after step 0) alike; read 2 names another.
    let forged = forge_process_change_for_test(&pattern, 2);
    let mut walk = KeyspaceWalk::start(client, &pattern)
        .await
        .expect("the start reads its forged witness");
    let first = walk.next_page().await;
    let second = walk.next_page().await;
    drop(forged);
    assert!(
        matches!(first, Ok(Some(_))),
        "the first page matched its witness: {first:?}"
    );
    match second {
        Err(WalkError::Changed(change)) => assert_process_change(&change, "SCAN", &pattern),
        other => panic!("the second page ends the walk with the change, got {other:?}"),
    }
}

/// The end fingerprint's witness comparison (`check_end`). A keyspace walk
/// whose every step matched its start witness, but whose end witness names
/// another process (the seam's `EndWitness`, forged as [`RUN_B`]), answers
/// every page and then fails at its end with the change: before it, the
/// server's own run_id; after it, [`RUN_B`].
///
/// Fail-on-old (in-place perturbation, red then green once restored): with
/// the end witness comparison removed from `check_end`, the walk completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keyspace_walk_fails_at_its_end_when_its_end_witness_names_another_process() {
    let tv = TestValkey::setup().await;
    let prefix = format!("{}scan-guard:end-witness:", tv.prefix());
    let pattern = format!("{prefix}*");
    seed_keyspace(&tv, &prefix).await;
    let client = tv.client().pool.next();

    let _forged = arm_walk_fault_for_test(
        &pattern,
        WalkFault::EndWitness {
            primary: 0,
            run_id: Some(RUN_B.to_owned()),
        },
    );
    let mut walk = KeyspaceWalk::start(client, &pattern)
        .await
        .expect("the start reads the server's own witness");
    let mut pages = 0;
    let ended = tokio::time::timeout(WALK_BUDGET, async {
        while walk.next_page().await?.is_some() {
            pages += 1;
        }
        Ok::<(), WalkError>(())
    })
    .await
    .expect("the walk ends within its budget");

    assert!(pages > 1, "every step matched its witness and was answered");
    match ended {
        Err(WalkError::Changed(change)) => match &*change {
            ServingChange::Process {
                walk,
                target,
                before,
                after,
                ..
            } => {
                assert_eq!((*walk, &**target), ("SCAN", pattern.as_str()));
                assert_eq!(after, RUN_B, "the end witness");
                assert_ne!(before, RUN_B, "the start witness, the server's own");
            }
            other => panic!("a change of serving process at the end, got {other:?}"),
        },
        other => panic!("the end witness fails the walk, got {other:?}"),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The coverage walk (b.hyw)
// ═════════════════════════════════════════════════════════════════════════════

/// A coverage walk of `pattern` on `client`, driven to its end: every key it
/// handed on, the reason it answered after its last key (`None` when it
/// completed), and whether it walked every primary. The reason is answered
/// once: the walk's next page after it is `None`. The whole walk is bounded
/// by [`WALK_BUDGET`].
async fn drive_coverage_walk(
    client: &fred::clients::Client,
    pattern: &str,
) -> (BTreeSet<String>, Option<WalkError>, bool) {
    let (found, reason, every_primary_walked, _) =
        drive_coverage_walk_resyncing(client, pattern).await;
    (found, reason, every_primary_walked)
}

/// [`drive_coverage_walk`], also telling whether the walk re-synced the
/// client's routing table (`KeyspaceWalk::resynced_for_test`).
async fn drive_coverage_walk_resyncing(
    client: &fred::clients::Client,
    pattern: &str,
) -> (BTreeSet<String>, Option<WalkError>, bool, bool) {
    tokio::time::timeout(WALK_BUDGET, async {
        let mut walk = KeyspaceWalk::start_coverage(client, pattern)
            .await
            .expect("a coverage walk starts on anything but a store fault");
        let mut found = BTreeSet::new();
        let reason = loop {
            match walk.next_page().await {
                Ok(Some(page)) => found.extend(as_strings(page)),
                Ok(None) => break None,
                Err(reason) => break Some(reason),
            }
        };
        if reason.is_some() {
            assert!(
                matches!(walk.next_page().await, Ok(None)),
                "the reason is answered once, after the last key"
            );
        }
        let every_primary_walked = walk.walked_every_primary();
        (
            found,
            reason,
            every_primary_walked,
            walk.resynced_for_test(),
        )
    })
    .await
    .expect("the coverage walk ends within its budget")
}

/// T3 (b.hyw): the coverage walk reads no serving-process witness. With
/// every walk of the pattern seeing an `INFO server` with no run_id line
/// (the seam's `RunId` at read 0), the guarded walk fails `WitnessMissing`
/// at its start, and the coverage walk returns every key with no error, so
/// the service's non-Job walks do not depend on a store reporting a run_id.
///
/// Fail-on-old: before b.hyw `scan_keyspace` walked through fred's
/// `scan` / `scan_cluster` streams, not `KeyspaceWalk`, so this test does
/// not apply to it; the guarded walk's failure here is its non-vacuity (the
/// fault is armed and would fail a walk that read the witness).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn coverage_walk_reads_no_witness_so_a_store_with_no_run_id_still_gets_every_key() {
    let tv = TestValkey::setup().await;
    let prefix = format!("{}scan-guard:coverage:", tv.prefix());
    let pattern = format!("{prefix}*");
    let written = seed_keyspace(&tv, &prefix).await;
    let client = tv.client().pool.next();

    let _armed = arm_walk_fault_for_test(
        &pattern,
        WalkFault::RunId {
            read: 0,
            run_id: None,
        },
    );
    match tokio::time::timeout(WALK_BUDGET, collect_keyspace(client, &pattern))
        .await
        .expect("the guarded walk ends within its budget")
    {
        Err(WalkError::WitnessMissing(missing)) => {
            assert_eq!(missing.seen, WitnessSeen::RunId(RunIdSeen::NoLine));
        }
        other => panic!(
            "non-vacuity: the guarded walk fails on the missing run_id, got {:?}",
            other.map(|keys| keys.len())
        ),
    }

    let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
    assert!(
        reason.is_none(),
        "the coverage walk reads no witness, so it ends with no error: {reason:?}"
    );
    assert_eq!(found, written, "every key is returned");
    assert!(every_primary_walked);
}

/// TR4 (d) (b.hyw): a coverage walk's step that fails transiently (the
/// seam's injected `IO` error) is sent again at the same cursor, with no
/// witness to re-read: failing once, or on every attempt but its last
/// ([`WALK_STEP_ATTEMPTS`] − 1), it lands, and the walk returns every key
/// with no error. Failing on every attempt ends the walk with the step's
/// error (`orphan_sweep`'s T4).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn coverage_walk_resends_a_transiently_failed_step_and_returns_every_key() {
    let tv = TestValkey::setup().await;
    let prefix = format!("{}scan-guard:coverage-retry:", tv.prefix());
    let pattern = format!("{prefix}*");
    let written = seed_keyspace(&tv, &prefix).await;
    let client = tv.client().pool.next();

    for times in [1, WALK_STEP_ATTEMPTS - 1] {
        let armed = arm_walk_fault_for_test(&pattern, WalkFault::StepError { step: 0, times });
        let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
        drop(armed);
        assert!(
            reason.is_none(),
            "{times} failed attempt(s): the step lands on a later one, so no error: {reason:?}"
        );
        assert_eq!(found, written, "{times} failed attempt(s): every key");
        assert!(every_primary_walked);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// On a cluster (cluster-gated)
// ═════════════════════════════════════════════════════════════════════════════

/// The fixture on `LIVE_EDIT_TEST_CLUSTER_URL`, or `None` after a `SKIP:`
/// line. One data-pool client, so every walk and every read runs on it.
/// Shared with `scan_guard_disruptive_test`.
pub(crate) async fn cluster_fixture() -> Option<TestValkey> {
    let Some(url) = test_cluster_url() else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
        return None;
    };
    let tv = TestValkey::setup_with_url(url, 1).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL names a cluster"
    );
    Some(tv)
}

/// One primary of the walk's map.
pub(crate) struct MapPrimary {
    /// As `CLUSTER SLOTS` names it: `host:port`.
    pub(crate) name: String,
    /// Its lowest slot, the slot its pin is resolved by.
    pub(crate) pin_slot: u16,
    pub(crate) slots: SlotSet,
}

/// The cluster's primaries as a keyspace walk visits them: by their lowest
/// slot, each with its whole slot set.
pub(crate) async fn map_primaries(tv: &TestValkey) -> Vec<MapPrimary> {
    let reply: Value = tv
        .client()
        .pool
        .next()
        .cluster_slots()
        .await
        .expect("CLUSTER SLOTS must succeed");
    let mut ranges: Vec<(u16, u16, String)> = reply
        .into_array()
        .into_iter()
        .map(|entry| {
            let mut fields = entry.into_array().into_iter();
            let mut slot = || {
                fields
                    .next()
                    .and_then(|v| v.as_i64())
                    .and_then(|n| u16::try_from(n).ok())
                    .expect("a slot number")
            };
            let (start, end) = (slot(), slot());
            let mut node = fields.next().expect("a primary").into_array().into_iter();
            let host = node.next().and_then(|h| h.as_string()).unwrap_or_default();
            let port = node.next().and_then(|p| p.as_i64()).expect("a port");
            (start, end, format!("{host}:{port}"))
        })
        .collect();
    ranges.sort_unstable();
    let mut primaries: Vec<MapPrimary> = Vec::new();
    for (start, end, name) in ranges {
        match primaries.iter_mut().find(|p| p.name == name) {
            Some(primary) => {
                primary.slots = SlotSet::from_ranges(
                    primary.slots.ranges().iter().copied().chain([(start, end)]),
                );
            }
            None => primaries.push(MapPrimary {
                name,
                pin_slot: start,
                slots: SlotSet::from_ranges([(start, end)]),
            }),
        }
    }
    primaries
}

/// A `CLUSTER NODES` reply whose myself line lists `owned`, then `extra`
/// fields (an open-slot marker, say).
pub(crate) fn myself_reply(owned: &SlotSet, extra: &str) -> String {
    let slots: Vec<String> = owned
        .ranges()
        .iter()
        .map(|&(start, end)| format!("{start}-{end}"))
        .collect();
    format!(
        "0000000000000000000000000000000000000001 127.0.0.1:7000@17000 myself,master - 0 0 1 \
         connected {} {extra}\n",
        slots.join(" ")
    )
    .replace(" \n", "\n")
}

/// `owned` less its highest slot.
pub(crate) fn less_its_last_slot(owned: &SlotSet) -> SlotSet {
    let mut ranges = owned.ranges().to_vec();
    let (start, end) = ranges.pop().expect("a primary serves a slot");
    if start < end {
        ranges.push((start, end - 1));
    }
    SlotSet::from_ranges(ranges)
}

/// Write one key under each of 60 hash tags, so every shard holds some;
/// return them.
async fn seed_every_shard(tv: &TestValkey) -> BTreeSet<String> {
    let keys: BTreeSet<String> = (0..60)
        .map(|i| format!("{}{{t{i}}}:k", tv.prefix()))
        .collect();
    for key in &keys {
        let _: () = tv
            .client()
            .pool
            .set(key, "1", None, None, false)
            .await
            .expect("SET must succeed");
    }
    keys
}

/// The node the client's routing table names for `slot`. Shared with
/// `scan_guard_disruptive_test`.
pub(crate) fn table_node(tv: &TestValkey, slot: u16) -> Server {
    tv.client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected cluster client has a routing table")
        .get_server(slot)
        .cloned()
        .expect("the table names the slot's owner")
}

/// Test 3 on a cluster. A clean keyspace walk across the three shards
/// returns every key written on them, passing both fingerprints; a myself
/// line forged at the end with an open slot (`[slot->-id]`, migrating) on
/// the second primary fails it with `OpenSlot`, naming the marker.
///
/// Fail-on-old (not run in this sandbox): with the open-slot check in
/// `read_primary` removed, the forged walk returns every key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_keyspace_walk_returns_every_shards_keys_and_fails_on_an_open_slot_at_its_end() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    let written = seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    assert_eq!(primaries.len(), 3, "fixture: a three-shard cluster");
    let holding: BTreeSet<Server> = written
        .iter()
        .map(|key| table_node(&tv, fred::util::redis_keyslot(key.as_bytes())))
        .collect();
    assert_eq!(holding.len(), 3, "fixture: every shard holds a key");
    let client = tv.client().pool.next();

    let found: BTreeSet<String> = as_strings(
        tokio::time::timeout(WALK_BUDGET, collect_keyspace(client, &pattern))
            .await
            .expect("the walk ends within its budget")
            .expect("a clean walk completes"),
    )
    .into_iter()
    .collect();
    assert_eq!(found, written, "every shard's keys are returned");

    // `CLUSTER NODES` reads: 0..3 at the start, one per primary; 3..6 at
    // the end. Read 4 is the second primary's at the end.
    let marker = "[42->-0000000000000000000000000000000000000002]";
    let _forged = arm_walk_fault_for_test(
        &pattern,
        WalkFault::ClusterNodesReply {
            read: 4,
            reply: myself_reply(&primaries[1].slots, marker),
        },
    );
    match tokio::time::timeout(WALK_BUDGET, collect_keyspace(client, &pattern))
        .await
        .expect("the walk ends within its budget")
    {
        Err(WalkError::Changed(change)) => match *change {
            ServingChange::OpenSlot { ref slots, .. } => assert_eq!(slots, marker),
            other => panic!("an open slot at the end, got {other:?}"),
        },
        other => panic!("an open slot at the end fails the walk, got {other:?}"),
    }
    println!("CLUSTER-GATED RAN: cluster_keyspace_walk_returns_every_shards_keys_and_fails_on_an_open_slot_at_its_end");
}

/// Test 15. A pinned node whose myself line lists one slot fewer than the
/// walk's map gives the first primary is `NotOwner`: forged at the start, the
/// walk fails before any step (`KeyspaceWalk::start`), naming the pattern,
/// the primary, the node and both slot sets, after re-syncing the client's
/// routing table; forged at the end, it fails after its steps, at the end.
///
/// Fail-on-old (not run in this sandbox): with the slot-set comparison
/// removed from `read_primary` (which `check_end` calls too), both walks
/// return `Ok`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_keyspace_walk_whose_pin_misses_a_primary_slot_fails_not_owner() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    let first = &primaries[0];
    let served = less_its_last_slot(&first.slots);
    let node = table_node(&tv, first.pin_slot).to_string();
    let client = tv.client().pool.next();
    let assert_not_owner = |result: Result<(), WalkError>, at: WalkEnd| match result {
        Err(WalkError::Changed(change)) => {
            match &*change {
                ServingChange::NotOwner {
                    target,
                    primary,
                    node: n,
                    expected,
                    served: s,
                    at: a,
                    ..
                } => {
                    assert_eq!(&**target, pattern.as_str());
                    assert_eq!(primary, &first.name);
                    assert_eq!(n.as_deref(), Some(node.as_str()));
                    assert_eq!((expected, s.as_ref()), (&first.slots, Some(&served)));
                    assert_eq!(*a, at);
                }
                other => panic!("{at:?}: NotOwner, got {other:?}"),
            }
            let text = change.to_string();
            for part in [
                pattern.clone(),
                first.name.clone(),
                node.clone(),
                first.slots.to_string(),
                served.to_string(),
            ] {
                assert!(
                    text.contains(&part),
                    "{at:?}: the line names {part:?}: {text}"
                );
            }
        }
        other => panic!("{at:?}: the walk fails NotOwner, got {other:?}"),
    };

    let start = arm_walk_fault_for_test(
        &pattern,
        WalkFault::ClusterNodesReply {
            read: 0,
            reply: myself_reply(&served, ""),
        },
    );
    let started = tokio::time::timeout(WALK_BUDGET, KeyspaceWalk::start(client, &pattern))
        .await
        .expect("the start ends within its budget");
    drop(start);
    assert_not_owner(started.map(|_| ()), WalkEnd::Start);

    // `CLUSTER NODES` read 3 is the first primary's at the end.
    let end = arm_walk_fault_for_test(
        &pattern,
        WalkFault::ClusterNodesReply {
            read: 3,
            reply: myself_reply(&served, ""),
        },
    );
    let mut walk = KeyspaceWalk::start(client, &pattern)
        .await
        .expect("the start fingerprint is the real one");
    let mut pages = 0;
    let ended = tokio::time::timeout(WALK_BUDGET, async {
        while walk.next_page().await?.is_some() {
            pages += 1;
        }
        Ok::<(), WalkError>(())
    })
    .await
    .expect("the walk ends within its budget");
    drop(end);
    assert!(pages >= 3, "the walk took its steps on every primary first");
    assert_not_owner(ended, WalkEnd::End);
    println!(
        "CLUSTER-GATED RAN: cluster_keyspace_walk_whose_pin_misses_a_primary_slot_fails_not_owner"
    );
}

/// Test 17. A primary pinned to a node the client holds no connection for
/// (`127.0.0.1:1`, through the seam) fails the walk at once, bounded in
/// time, with the refusal naming that node and nothing returned: a node pin
/// has no fallback to another connection, and a pin the routing table does
/// not list is refused before anything is sent. The refusal is a `Routing`
/// error, never a `Timeout`, so no caller counts it on
/// `valkey_errors_total`.
///
/// Fail-on-old (not run in this sandbox): with the walk's commands sent by
/// the primary's pin slot instead of its node (`Route::Slot` in
/// `KeyspaceWalk::start`), the same walk returns `Ok`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_keyspace_walk_pinned_to_an_unconnected_node_fails_with_no_fallback() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let _pinned = arm_walk_fault_for_test(
        &pattern,
        WalkFault::PinNode {
            primary: 0,
            server: Server::new("127.0.0.1", 1),
        },
    );
    let walked = tokio::time::timeout(
        WALK_BUDGET,
        collect_keyspace(tv.client().pool.next(), &pattern),
    )
    .await
    .expect("the walk ends within its budget");
    match walked {
        Err(WalkError::Step { walk, source, .. }) => {
            assert_eq!(walk, "SCAN");
            assert_eq!(*source.kind(), fred::error::ErrorKind::Routing);
            assert!(
                source.details().contains("127.0.0.1:1"),
                "the refusal names the node: {source}"
            );
        }
        other => panic!(
            "a pin to an unconnected node fails the walk and returns no key, got {:?}",
            other.map(|keys| keys.len())
        ),
    }
    println!("CLUSTER-GATED RAN: cluster_keyspace_walk_pinned_to_an_unconnected_node_fails_with_no_fallback");
}

/// Test 21 (R1-a). A pin refused during a step's R1 re-read ends the walk
/// with the refusal, sending nothing more. Node-route pin checks are
/// numbered from 0, one before every separate witness read (`read_witness`:
/// the start and end fingerprints' and an R1 re-read), every
/// `CLUSTER NODES` (`read_primary`) and every step attempt's pipeline, the
/// step with its witness (`send_guarded_step`; b.qgz): with three primaries
/// the start is 0–5 (each primary's witness, then its `CLUSTER NODES`). With
/// step 0 failing, its first send is 6, its first re-read 7, then send 8,
/// re-read 9, send 10, re-read 11. With no step failing, 6 is step 0's
/// pipeline and 7 step 1's: the witness riding with a step that lands has
/// no check of its own.
///
/// (a) Step 0 fails once and the check of its first re-read (7) is refused:
/// the walk ends with the refusal's `Step` (a `Routing` error naming the
/// first primary's node) and returns no key. (b) Step 0 fails three times
/// and the check of its third re-read (11) is refused: the error is the
/// refusal, not the seam's `IO` "failed before it was sent". With no step
/// failing: (c) the check before the first primary's start `CLUSTER NODES`
/// (1) refused ends the walk with the same refusal and no key; (d) the check
/// before step 1's pipeline (7) refused ends it after exactly one page, step
/// 0's, with a `Routing` refusal (naming whichever primary step 1 belongs
/// to).
///
/// Fail-on-old (not run in this sandbox): with `after_failed_attempt`'s
/// positive `Read` arm turned back into the catch-all `Err(_) => Ok(())`,
/// (a) sends step 0 again and returns every key, and (b) ends with the
/// seam's `IO` error. With the check before `CLUSTER NODES` removed from
/// `read_primary`, check 1 is the second primary's start witness, so (c)'s
/// refusal names that primary's node, not the first's. With the check
/// before a pipeline removed from `send_guarded_step`, check 7 is a later
/// check (the end fingerprint's or none), so (d) hands on more than one
/// page before any refusal; with a landed step's witness checked again on
/// its own, as before b.qgz, check 7 is step 0's witness and (d) hands on
/// no page.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_keyspace_walk_refused_its_pin_on_an_r1_reread_ends_with_the_refusal() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    assert_eq!(
        primaries.len(),
        3,
        "fixture: the check numbering assumes three"
    );
    let node = table_node(&tv, primaries[0].pin_slot).to_string();
    let client = tv.client().pool.next();

    for (times, check) in [(Some(1), 7), (Some(3), 11), (None, 1)] {
        let mut armed = vec![arm_walk_fault_for_test(
            &pattern,
            WalkFault::RefuseRoute { check },
        )];
        if let Some(times) = times {
            armed.push(arm_walk_fault_for_test(
                &pattern,
                WalkFault::StepError { step: 0, times },
            ));
        }
        let walked = tokio::time::timeout(WALK_BUDGET, collect_keyspace(client, &pattern))
            .await
            .expect("the walk ends within its budget");
        drop(armed);
        match walked {
            Err(WalkError::Step { walk, source, .. }) => {
                assert_eq!(walk, "SCAN");
                assert_eq!(
                    *source.kind(),
                    fred::error::ErrorKind::Routing,
                    "check {check}: the refusal, not the seam's IO error: {source}"
                );
                assert!(
                    source.details().contains(&format!("pinned node {node}")),
                    "check {check}: the refusal names the first primary's node: {source}"
                );
            }
            other => panic!(
                "check {check}: the refusal ends the walk, got {:?}",
                other.map(|keys| keys.len())
            ),
        }
    }

    // (d): the check before step 1's pipeline, with no step failing.
    let armed = arm_walk_fault_for_test(&pattern, WalkFault::RefuseRoute { check: 7 });
    let mut walk = KeyspaceWalk::start(client, &pattern)
        .await
        .expect("the start's checks (0-5) pass");
    let mut pages = 0;
    let ended = tokio::time::timeout(WALK_BUDGET, async {
        while walk.next_page().await?.is_some() {
            pages += 1;
        }
        Ok::<(), WalkError>(())
    })
    .await
    .expect("the walk ends within its budget");
    drop(armed);
    match ended {
        Err(WalkError::Step { walk, source, .. }) => {
            assert_eq!(walk, "SCAN");
            assert_eq!(*source.kind(), fred::error::ErrorKind::Routing, "{source}");
            assert!(
                source.details().contains("pinned node"),
                "check 7: the refusal names its node: {source}"
            );
        }
        other => panic!("check 7: the refusal ends the walk, got {other:?}"),
    }
    assert_eq!(
        pages, 1,
        "check 7 is step 1's pipeline: only step 0's page is handed on before the refusal"
    );
    println!("CLUSTER-GATED RAN: cluster_keyspace_walk_refused_its_pin_on_an_r1_reread_ends_with_the_refusal");
}

/// The end fingerprint's witness comparison on a cluster, for a primary other
/// than the first: a walk whose third primary's end witness names another
/// process (the seam's `EndWitness`, forged as [`RUN_B`]) fails at its end
/// with the change, on that primary's node.
///
/// Fail-on-old: with the end witness comparison removed from `check_end`,
/// the walk completes (proven on the standalone walk,
/// `keyspace_walk_fails_at_its_end_when_its_end_witness_names_another_process`;
/// not run here on a cluster, which this sandbox lacks).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_keyspace_walk_fails_at_its_end_when_a_later_primarys_end_witness_changes() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    assert_eq!(primaries.len(), 3, "fixture: a three-shard cluster");
    let node = table_node(&tv, primaries[2].pin_slot).to_string();

    let _forged = arm_walk_fault_for_test(
        &pattern,
        WalkFault::EndWitness {
            primary: 2,
            run_id: Some(RUN_B.to_owned()),
        },
    );
    let walked = tokio::time::timeout(
        WALK_BUDGET,
        collect_keyspace(tv.client().pool.next(), &pattern),
    )
    .await
    .expect("the walk ends within its budget");
    match walked {
        Err(WalkError::Changed(change)) => match &*change {
            ServingChange::Process {
                node: n,
                before,
                after,
                ..
            } => {
                assert_eq!(
                    n.as_deref(),
                    Some(node.as_str()),
                    "the third primary's node"
                );
                assert_eq!(after, RUN_B, "the end witness");
                assert_ne!(before, RUN_B, "the start witness, the server's own");
            }
            other => panic!("a change of serving process at the end, got {other:?}"),
        },
        other => panic!(
            "the third primary's end witness fails the walk, got {:?}",
            other.map(|keys| keys.len())
        ),
    }
    println!("CLUSTER-GATED RAN: cluster_keyspace_walk_fails_at_its_end_when_a_later_primarys_end_witness_changes");
}

// ── A DR Job's keyspace walk on a cluster (b.f3t) ─────────────────────────
//
// `KeyspaceWalk::start_job` keeps every check of the guarded walk but the
// witness: the node pins, each pin's myself line at the start and the end
// (`check_owner`), the slot map unchanged from start to end
// (`check_job_end`), no open slot, no refused pin. Its `CLUSTER NODES`
// reads are numbered as a guarded walk's: one per primary at the start
// (0–2), one per primary at the end (3–5). A primary whose pin resolves to
// no node at all (the `begin` arm for a missing pin), which the routing
// table of a healthy cluster never gives, is forged by the seam's
// `UnresolvedPin`. All cluster-gated (CI's `test-cluster:` job) and compiled
// only with `test-fault-injection`; not run in this sandbox, which has no
// cluster, so their fail-on-old perturbations are stated, not run.

/// Walk `pattern` once with a DR Job's walk on `client`: every key and the
/// pages that held them, or the walk's error.
async fn drive_job_walk(
    client: &fred::clients::Client,
    pattern: &str,
) -> Result<(BTreeSet<String>, usize), WalkError> {
    let mut walk = KeyspaceWalk::start_job(client, pattern).await?;
    let (mut keys, mut pages) = (BTreeSet::new(), 0);
    while let Some(page) = walk.next_page().await? {
        pages += 1;
        keys.extend(as_strings(page));
    }
    Ok((keys, pages))
}

/// b.f3t on a cluster, the clean walk and an open slot. An agreeing pair of
/// Job walks (`collect_keyspace_agreed`) across the three shards returns
/// every key written on them, each once, and reads no witness: a
/// placeholder run_id forged on its first witness read changes nothing.
/// A myself line forged with an open slot (`[slot->-id]`, migrating) fails a
/// Job walk with `OpenSlot`, naming the marker: on the first primary's
/// start read (0), before any step; on the second primary's end read (4),
/// after its steps.
///
/// Fail-on-old (stated, not run in this sandbox): with the open-slot check
/// removed from `check_owner`, both forged walks return their keys.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_job_walk_returns_every_shards_keys_and_fails_on_an_open_slot() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    let written = seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    assert_eq!(primaries.len(), 3, "fixture: a three-shard cluster");
    let client = tv.client().pool.next();

    let no_witness = arm_walk_fault_for_test(
        &pattern,
        WalkFault::RunId {
            read: 0,
            run_id: Some("0".to_owned()),
        },
    );
    let agreed = tokio::time::timeout(WALK_BUDGET, collect_keyspace_agreed(client, &pattern))
        .await
        .expect("the walks end within their budget")
        .expect("a clean agreeing walk completes, reading no witness");
    drop(no_witness);
    let found: Vec<String> = as_strings(agreed);
    assert_eq!(found.len(), written.len(), "each key once");
    assert_eq!(
        found.into_iter().collect::<BTreeSet<_>>(),
        written,
        "every shard's keys are returned"
    );

    let marker = "[42->-0000000000000000000000000000000000000002]";
    for (at, read, primary) in [(WalkEnd::Start, 0, 0), (WalkEnd::End, 4, 1)] {
        let forged = arm_walk_fault_for_test(
            &pattern,
            WalkFault::ClusterNodesReply {
                read,
                reply: myself_reply(&primaries[primary].slots, marker),
            },
        );
        let walked = tokio::time::timeout(WALK_BUDGET, drive_job_walk(client, &pattern))
            .await
            .expect("the walk ends within its budget");
        drop(forged);
        match walked {
            Err(WalkError::Changed(change)) => match *change {
                ServingChange::OpenSlot { ref slots, .. } => assert_eq!(slots, marker, "{at:?}"),
                other => panic!("{at:?}: an open slot, got {other:?}"),
            },
            other => panic!(
                "{at:?}: an open slot fails the walk, got {:?}",
                other.map(|(keys, _)| keys.len())
            ),
        }
    }
    println!(
        "CLUSTER-GATED RAN: cluster_job_walk_returns_every_shards_keys_and_fails_on_an_open_slot"
    );
}

/// b.f3t on a cluster, ownership. A pinned node whose myself line lists one
/// slot fewer than the walk's map gives the first primary is `NotOwner`:
/// forged on its start read (0), the Job walk fails before any step
/// (`KeyspaceWalk::start_job`); forged on its end read (3), it fails after
/// its steps, at the end (`check_job_end`). Each names the pattern, the
/// primary, the node and both slot sets.
///
/// Fail-on-old (stated, not run in this sandbox): with the slot-set
/// comparison removed from `check_owner`, both walks return `Ok`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_job_walk_whose_pin_misses_a_primary_slot_fails_not_owner() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    let first = &primaries[0];
    let served = less_its_last_slot(&first.slots);
    let node = table_node(&tv, first.pin_slot).to_string();
    let client = tv.client().pool.next();

    let start = arm_walk_fault_for_test(
        &pattern,
        WalkFault::ClusterNodesReply {
            read: 0,
            reply: myself_reply(&served, ""),
        },
    );
    let started = tokio::time::timeout(WALK_BUDGET, KeyspaceWalk::start_job(client, &pattern))
        .await
        .expect("the start ends within its budget");
    drop(start);

    let end = arm_walk_fault_for_test(
        &pattern,
        WalkFault::ClusterNodesReply {
            read: 3,
            reply: myself_reply(&served, ""),
        },
    );
    let ended = tokio::time::timeout(WALK_BUDGET, drive_job_walk(client, &pattern))
        .await
        .expect("the walk ends within its budget");
    drop(end);

    for (at, result) in [
        (WalkEnd::Start, started.map(|_| ())),
        (WalkEnd::End, ended.map(|_| ())),
    ] {
        match result {
            Err(WalkError::Changed(change)) => {
                match &*change {
                    ServingChange::NotOwner {
                        target,
                        primary,
                        node: n,
                        expected,
                        served: s,
                        at: a,
                        ..
                    } => {
                        assert_eq!(&**target, pattern.as_str(), "{at:?}");
                        assert_eq!(primary, &first.name, "{at:?}");
                        assert_eq!(n.as_deref(), Some(node.as_str()), "{at:?}");
                        assert_eq!(
                            (expected, s.as_ref()),
                            (&first.slots, Some(&served)),
                            "{at:?}"
                        );
                        assert_eq!(*a, at);
                    }
                    other => panic!("{at:?}: NotOwner, got {other:?}"),
                }
                let text = change.to_string();
                for part in [
                    pattern.clone(),
                    first.name.clone(),
                    node.clone(),
                    first.slots.to_string(),
                    served.to_string(),
                ] {
                    assert!(
                        text.contains(&part),
                        "{at:?}: the line names {part:?}: {text}"
                    );
                }
            }
            other => panic!("{at:?}: the walk fails NotOwner, got {other:?}"),
        }
    }
    println!("CLUSTER-GATED RAN: cluster_job_walk_whose_pin_misses_a_primary_slot_fails_not_owner");
}

/// b.f3t on a cluster, a refused pin. A Job walk whose first primary is
/// pinned to a node the client holds no connection for (`127.0.0.1:1`,
/// through the seam's `PinNode`) fails at once, before its start
/// `CLUSTER NODES` is sent, with a `Routing` refusal naming that node: a
/// node pin has no fallback to another connection.
///
/// Fail-on-old (stated, not run in this sandbox): with the walk's commands
/// sent by the primary's pin slot instead of its node (`Route::Slot` in
/// `begin`'s Job arm), the walk returns `Ok`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_job_walk_pinned_to_an_unconnected_node_fails_with_no_fallback() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let _pinned = arm_walk_fault_for_test(
        &pattern,
        WalkFault::PinNode {
            primary: 0,
            server: Server::new("127.0.0.1", 1),
        },
    );
    let walked = tokio::time::timeout(
        WALK_BUDGET,
        drive_job_walk(tv.client().pool.next(), &pattern),
    )
    .await
    .expect("the walk ends within its budget");
    match walked {
        Err(WalkError::Step { walk, source, .. }) => {
            assert_eq!(walk, "SCAN");
            assert_eq!(*source.kind(), fred::error::ErrorKind::Routing);
            assert!(
                source.details().contains("127.0.0.1:1"),
                "the refusal names the node: {source}"
            );
        }
        other => panic!(
            "a pin to an unconnected node fails the walk and returns no key, got {:?}",
            other.map(|(keys, _)| keys.len())
        ),
    }
    println!(
        "CLUSTER-GATED RAN: cluster_job_walk_pinned_to_an_unconnected_node_fails_with_no_fallback"
    );
}

/// b.f3t on a cluster, the end map (`check_job_end`). The walk seam reads
/// the end `CLUSTER SLOTS` map (read 1) with its first two primaries' names
/// exchanged (`SlotMapPrimariesSwapped`): every range keeps its slots, so
/// every pin is checked as before, but the end map differs from the
/// start's, so the Job walk fails `SlotMap` at its end, naming the pattern
/// and both maps. Control: the same exchange on both reads (0 and 1) makes
/// the maps agree, and the walk returns every key.
///
/// Fail-on-old (stated, not run in this sandbox): with the map comparison
/// removed from `check_job_end`, the first walk returns every key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_job_walk_whose_end_map_differs_from_its_start_fails_slot_map() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    let written = seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let client = tv.client().pool.next();

    let armed = arm_walk_fault_for_test(&pattern, WalkFault::SlotMapPrimariesSwapped { read: 1 });
    let walked = tokio::time::timeout(WALK_BUDGET, drive_job_walk(client, &pattern))
        .await
        .expect("the walk ends within its budget");
    drop(armed);
    match walked {
        Err(WalkError::Changed(change)) => match *change {
            ServingChange::SlotMap {
                ref target,
                ref before,
                ref after,
            } => {
                assert_eq!(&**target, pattern.as_str());
                assert_ne!(before, after, "the two maps are named, and differ");
            }
            other => panic!("SlotMap, got {other:?}"),
        },
        other => panic!(
            "a map that changed during the walk fails it, got {:?}",
            other.map(|(keys, _)| keys.len())
        ),
    }

    let armed = [
        arm_walk_fault_for_test(&pattern, WalkFault::SlotMapPrimariesSwapped { read: 0 }),
        arm_walk_fault_for_test(&pattern, WalkFault::SlotMapPrimariesSwapped { read: 1 }),
    ];
    let walked = tokio::time::timeout(WALK_BUDGET, drive_job_walk(client, &pattern))
        .await
        .expect("the walk ends within its budget");
    drop(armed);
    let (found, _) = walked.expect("control: the same exchange at both ends is no change");
    assert_eq!(found, written, "control: every key");
    println!(
        "CLUSTER-GATED RAN: cluster_job_walk_whose_end_map_differs_from_its_start_fails_slot_map"
    );
}

/// b.f3t on a cluster, an unresolved pin (`begin`'s arm for a primary whose
/// pin resolves to no node). With the seam's `UnresolvedPin` on the first
/// primary, as if the client's routing table named no node for its pin
/// slot, a guarded walk (`KeyspaceWalk::start`) and a Job walk
/// (`KeyspaceWalk::start_job`) each fail at their start, before any step,
/// with `NotOwner` naming the pattern and the primary, with no node and no
/// served slots, after a routing re-sync that refreshed the table; an
/// agreeing pair of Job walks (`collect_keyspace_agreed`) fails the same
/// way and returns no key. Contrast: a coverage walk leaves that primary
/// unwalked, returns every other primary's keys, then answers the same
/// `NotOwner` and does not count every primary walked.
///
/// A Job walk must fail here, not skip: two agreeing walks that both
/// skipped the primary would agree on a set missing its keys.
///
/// Fail-on-old (stated, not run in this sandbox, which has no cluster):
/// with `KeyspaceGuard::Job` moved from `begin`'s failing `None` arm into
/// the coverage walk's (`coverage_not_owner`, then `continue`), `start_job`
/// returns `Ok`; with the Job's `None` arm a bare `continue`, `start_job`
/// returns `Ok` too and the agreeing pair returns every key but the first
/// primary's with no error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_job_walk_whose_pin_resolves_to_no_node_fails_not_owner_at_its_start() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    let written = seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    let first = &primaries[0];
    let off_first: BTreeSet<String> = written
        .iter()
        .filter(|key| {
            let slot = fred::util::redis_keyslot(key.as_bytes());
            !first
                .slots
                .ranges()
                .iter()
                .any(|&(start, end)| (start..=end).contains(&slot))
        })
        .cloned()
        .collect();
    assert!(
        !off_first.is_empty() && off_first.len() < written.len(),
        "fixture: the first primary and another hold keys"
    );
    let client = tv.client().pool.next();
    let assert_not_owner_at_start = |what: &str, reason: WalkError| match reason {
        WalkError::Changed(change) => match &*change {
            ServingChange::NotOwner {
                target,
                primary,
                node,
                expected,
                served,
                at,
                refresh,
            } => {
                assert_eq!(&**target, pattern.as_str(), "{what}");
                assert_eq!(primary, &first.name, "{what}");
                assert_eq!(*node, None, "{what}: no node");
                assert_eq!(*served, None, "{what}: no served slots");
                assert_eq!(expected, &first.slots, "{what}");
                assert_eq!(*at, WalkEnd::Start, "{what}");
                assert_eq!(refresh, &RoutingRefresh::Refreshed, "{what}: re-synced");
                let text = change.to_string();
                for part in [
                    pattern.clone(),
                    first.name.clone(),
                    "names no node for it at the walk's start".to_owned(),
                ] {
                    assert!(
                        text.contains(&part),
                        "{what}: the line names {part:?}: {text}"
                    );
                }
            }
            other => panic!("{what}: NotOwner, got {other:?}"),
        },
        other => panic!("{what}: NotOwner, got {other:?}"),
    };

    let armed = arm_walk_fault_for_test(&pattern, WalkFault::UnresolvedPin { primary: 0 });
    let guarded = tokio::time::timeout(WALK_BUDGET, KeyspaceWalk::start(client, &pattern))
        .await
        .expect("the start ends within its budget");
    let job = tokio::time::timeout(WALK_BUDGET, KeyspaceWalk::start_job(client, &pattern))
        .await
        .expect("the start ends within its budget");
    let agreed = tokio::time::timeout(WALK_BUDGET, collect_keyspace_agreed(client, &pattern))
        .await
        .expect("the walks end within their budget");
    let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
    drop(armed);

    for (what, started) in [("guarded walk", guarded), ("Job walk", job)] {
        match started {
            Err(reason) => assert_not_owner_at_start(what, reason),
            Ok(_) => panic!("{what}: an unresolved pin fails the start"),
        }
    }
    match agreed {
        Err(reason) => assert_not_owner_at_start("agreeing pair", reason),
        Ok(keys) => panic!(
            "agreeing pair: an unresolved pin fails the walks, got {} keys",
            keys.len()
        ),
    }

    assert_eq!(
        found, off_first,
        "coverage walk: every key but the first primary's"
    );
    assert_not_owner_at_start(
        "coverage walk",
        reason.expect("coverage walk: the NotOwner is answered after its last key"),
    );
    assert!(
        !every_primary_walked,
        "coverage walk: not every primary walked"
    );
    println!(
        "CLUSTER-GATED RAN: cluster_job_walk_whose_pin_resolves_to_no_node_fails_not_owner_at_its_start"
    );
}

/// T7 (b.hyw). The coverage walk on a cluster, with the first primary's start
/// `CLUSTER NODES` reply forged (read 0):
///
/// - an open slot on its own line (`OpenSlot`): the primary is walked, so
///   every key is returned, then the open slot is answered, naming the
///   marker; every primary walked;
/// - one slot fewer (`NotOwner`): the primary is left unwalked, so every key
///   but its own is returned, then `NotOwner` at the start, naming both slot
///   sets, after a routing re-sync that refreshed the table; not every
///   primary walked;
/// - no myself line (`WitnessMissing`): walked, every key, then the missing
///   line; every primary walked.
///
/// Each check that does not hold is answered after the last key, never
/// before any. `orphan_sweep`'s cluster-gated
/// `scan_keyspace_on_a_cluster_ends_a_forged_verdict_with_one_error_after_its_keys`
/// pins the same three through `scan_keyspace`'s error item.
///
/// Fail-on-old (not run in this sandbox, which has no cluster): with
/// `CoverageVerdict::walks` answering `true` for a `NotOwner`, the second
/// walk returns the first primary's keys too (the forged line lies; the pin
/// is its real owner); with it answering `false` for an open slot, the first
/// walk misses that primary's keys.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_coverage_walk_walks_past_an_unproven_owner_and_skips_only_a_proven_non_owner() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    let written = seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    let first = &primaries[0];
    let off_first: BTreeSet<String> = written
        .iter()
        .filter(|key| {
            let slot = fred::util::redis_keyslot(key.as_bytes());
            !first
                .slots
                .ranges()
                .iter()
                .any(|&(start, end)| (start..=end).contains(&slot))
        })
        .cloned()
        .collect();
    assert!(
        !off_first.is_empty() && off_first.len() < written.len(),
        "fixture: the first primary and another hold keys"
    );
    let served = less_its_last_slot(&first.slots);
    let client = tv.client().pool.next();
    let marker = "[42->-0000000000000000000000000000000000000002]";
    let forged = |reply: String| {
        arm_walk_fault_for_test(&pattern, WalkFault::ClusterNodesReply { read: 0, reply })
    };

    // An open slot: walked, answered after the last key.
    let armed = forged(myself_reply(&first.slots, marker));
    let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
    drop(armed);
    assert_eq!(found, written, "open slot: the primary is walked");
    match reason {
        Some(WalkError::Changed(change)) => match *change {
            ServingChange::OpenSlot { ref slots, .. } => assert_eq!(slots, marker),
            other => panic!("open slot: OpenSlot, got {other:?}"),
        },
        other => panic!("open slot: answered after the last key, got {other:?}"),
    }
    assert!(every_primary_walked, "open slot: every primary walked");

    // One slot fewer: a proven non-owner, left unwalked.
    let armed = forged(myself_reply(&served, ""));
    let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
    drop(armed);
    assert_eq!(
        found, off_first,
        "NotOwner: every key but the first primary's"
    );
    match reason {
        Some(WalkError::Changed(change)) => match *change {
            ServingChange::NotOwner {
                ref primary,
                ref expected,
                served: ref s,
                at,
                ref refresh,
                ..
            } => {
                assert_eq!(primary, &first.name);
                assert_eq!((expected, s.as_ref()), (&first.slots, Some(&served)));
                assert_eq!(at, WalkEnd::Start);
                assert_eq!(refresh, &RoutingRefresh::Refreshed);
            }
            other => panic!("NotOwner: NotOwner, got {other:?}"),
        },
        other => panic!("NotOwner: answered after the last key, got {other:?}"),
    }
    assert!(!every_primary_walked, "NotOwner: not every primary walked");

    // No myself line: walked, answered after the last key.
    let armed = forged(String::new());
    let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
    drop(armed);
    assert_eq!(found, written, "unusable line: the primary is walked");
    match reason {
        Some(WalkError::WitnessMissing(missing)) => assert_eq!(
            missing.seen,
            WitnessSeen::MyselfLine(MyselfLineSeen::NoLine)
        ),
        other => panic!("unusable line: WitnessMissing, got {other:?}"),
    }
    assert!(every_primary_walked, "unusable line: every primary walked");
    println!(
        "CLUSTER-GATED RAN: cluster_coverage_walk_walks_past_an_unproven_owner_and_skips_only_a_proven_non_owner"
    );
}

/// TR4 (b) (b.hyw). A coverage walk whose end check finds the first
/// primary's node serving other slots (its end `CLUSTER NODES`, the first
/// end read, forged one slot short) walked every primary, so every key is
/// returned, but then answers `NotOwner` at its end, after a routing re-sync
/// that refreshed the table, and does not count every primary walked: its
/// keys of that primary are unproven to be the primary's.
///
/// Fail-on-old (not run in this sandbox, which has no cluster): with the
/// end verdicts skipped in `check_coverage_end`, the walk completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_coverage_walk_whose_end_check_finds_a_non_owner_ends_not_owner_at_its_end() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    let written = seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    let first = &primaries[0];
    let served = less_its_last_slot(&first.slots);
    let client = tv.client().pool.next();

    // `CLUSTER NODES` reads: one per primary at the start, then the end's,
    // first primary first.
    let armed = arm_walk_fault_for_test(
        &pattern,
        WalkFault::ClusterNodesReply {
            read: primaries.len(),
            reply: myself_reply(&served, ""),
        },
    );
    let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
    drop(armed);
    assert_eq!(
        found, written,
        "every primary was walked before the end check"
    );
    match reason {
        Some(WalkError::Changed(change)) => match *change {
            ServingChange::NotOwner {
                ref primary,
                ref expected,
                served: ref s,
                at,
                ref refresh,
                ..
            } => {
                assert_eq!(primary, &first.name);
                assert_eq!((expected, s.as_ref()), (&first.slots, Some(&served)));
                assert_eq!(at, WalkEnd::End);
                assert_eq!(refresh, &RoutingRefresh::Refreshed);
            }
            other => panic!("NotOwner at the end, got {other:?}"),
        },
        other => panic!("the end check's NotOwner is answered, got {other:?}"),
    }
    assert!(
        !every_primary_walked,
        "a NotOwner at the end loses coverage"
    );
    println!(
        "CLUSTER-GATED RAN: cluster_coverage_walk_whose_end_check_finds_a_non_owner_ends_not_owner_at_its_end"
    );
}

/// TR4 (c) (b.hyw). A coverage walk that recorded an open slot on its first
/// primary (the start `CLUSTER NODES`, read 0, forged) and then meets a
/// store fault (its step 1 failing on every attempt) answers the fault —
/// the step's own `IO` error — not the recorded open slot, and does not
/// count every primary walked: a store fault ends the walk at once and
/// drops the recorded reason.
///
/// Fail-on-old (not run in this sandbox): with a step's error recorded like
/// a check and the walk walked on, the walk answers `OpenSlot`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_coverage_walk_answers_a_store_fault_not_the_reason_it_recorded() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    let client = tv.client().pool.next();

    let armed = [
        arm_walk_fault_for_test(
            &pattern,
            WalkFault::ClusterNodesReply {
                read: 0,
                reply: myself_reply(
                    &primaries[0].slots,
                    "[42->-0000000000000000000000000000000000000002]",
                ),
            },
        ),
        // Three primaries take at least three steps, so step 1 is reached.
        arm_walk_fault_for_test(
            &pattern,
            WalkFault::StepError {
                step: 1,
                times: u32::MAX,
            },
        ),
    ];
    let mut walk = KeyspaceWalk::start_coverage(client, &pattern)
        .await
        .expect("the start's checks only record the open slot");
    let ended = tokio::time::timeout(WALK_BUDGET, async {
        loop {
            match walk.next_page().await {
                Ok(Some(_)) => {}
                Ok(None) => break None,
                Err(e) => break Some(e),
            }
        }
    })
    .await
    .expect("the walk ends within its budget");
    drop(armed);
    match ended {
        Some(WalkError::Step {
            walk: w, source, ..
        }) => {
            assert_eq!(w, "SCAN");
            assert_eq!(*source.kind(), fred::error::ErrorKind::IO, "{source}");
            assert!(
                source
                    .details()
                    .contains("test seam: step 1 of a SCAN walk failed before it was sent"),
                "step 1's own error: {source}"
            );
        }
        other => panic!("the store fault is answered, not the open slot, got {other:?}"),
    }
    assert!(
        !walk.walked_every_primary(),
        "a walk a store fault ended did not walk every primary"
    );
    println!(
        "CLUSTER-GATED RAN: cluster_coverage_walk_answers_a_store_fault_not_the_reason_it_recorded"
    );
}

/// TR4 (f) (b.hyw). A coverage walk whose first pin check (check 0, before
/// the first primary's start `CLUSTER NODES`; a coverage walk reads no
/// witness) is refused, as if the client's routing table no longer listed
/// the pinned node, fails at its start with the refusal — a store fault, a
/// `Routing` error naming that node — before any step.
///
/// Fail-on-old (not run in this sandbox): with the pin check removed from
/// `check_coverage`, check 0 is step 0's send, and the walk starts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_coverage_walk_refused_a_pin_at_its_start_fails_before_any_step() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    let node = table_node(&tv, primaries[0].pin_slot).to_string();

    let armed = arm_walk_fault_for_test(&pattern, WalkFault::RefuseRoute { check: 0 });
    let started = tokio::time::timeout(
        WALK_BUDGET,
        KeyspaceWalk::start_coverage(tv.client().pool.next(), &pattern),
    )
    .await
    .expect("the start ends within its budget");
    drop(armed);
    match started {
        Err(WalkError::Step { walk, source, .. }) => {
            assert_eq!(walk, "SCAN");
            assert_eq!(*source.kind(), fred::error::ErrorKind::Routing, "{source}");
            assert!(
                source.details().contains(&format!("pinned node {node}")),
                "the refusal names the first primary's node: {source}"
            );
        }
        Err(other) => panic!("the refusal fails the start, got {other:?}"),
        Ok(_) => panic!("a refused pin fails the coverage walk's start"),
    }
    println!(
        "CLUSTER-GATED RAN: cluster_coverage_walk_refused_a_pin_at_its_start_fails_before_any_step"
    );
}

/// b.87a, the table check, on a cluster. One of the first primary's
/// `CLUSTER NODES` reads is forged, and the walk seam forges one
/// routing-table comparison to report that the table differs from the map
/// (`TablePartitionDiffers`). Comparisons are numbered only when made: a
/// walk with an unproven start verdict compares at the start (0) and at the
/// end (1); a walk whose first unproven verdict comes at the end compares
/// only there (0).
///
/// - no myself line at the start (read 0), comparison 0 (the start's)
///   differs: every key (the pins are the real owners), then `StaleTable` at
///   the start, after a re-sync that refreshed the table; not every primary
///   walked;
/// - no myself line at the start, comparison 1 (the end's: the start's
///   matched) differs: every key, then `StaleTable` at the end, refreshed;
///   not every primary walked;
/// - no myself line only at the end (the first end read), comparison 0 (the
///   end's, the start having made none) differs: every key, then
///   `StaleTable` at the end, refreshed; not every primary walked;
/// - an open slot on the first primary's own line at the start, comparison
///   0 armed: an open slot proves the node's slots, so no pin is unproven
///   and no comparison is made; every key, then `OpenSlot`, every primary
///   walked.
///
/// Fail-on-old (in-place perturbations of `storage::scan_guard`, stated, not
/// run in this sandbox, which has no cluster): with `check_table`'s
/// `coverage_lost = true` dropped, the first row answers `StaleTable` with
/// every primary walked; with its `if !self.ownership_unproven` gate
/// removed, the fourth row's armed comparison is made, and it answers
/// `StaleTable`, not every primary walked. With the end `check_table` moved
/// above the end verdicts in `check_coverage_end`, or made only when a start
/// verdict was unproven, the third row's walk makes no comparison and
/// answers its missing myself line with every primary walked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_coverage_walk_whose_unproven_pin_came_from_a_table_that_differs_from_its_map_loses_coverage(
) {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    let written = seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let primaries = map_primaries(&tv).await;
    let client = tv.client().pool.next();
    let start_reply = |reply: String| WalkFault::ClusterNodesReply { read: 0, reply };
    let open_slot = myself_reply(
        &primaries[0].slots,
        "[42->-0000000000000000000000000000000000000002]",
    );

    for (case, faults, stale_at) in [
        (
            "no myself line, the start's comparison differs",
            vec![
                start_reply(String::new()),
                WalkFault::TablePartitionDiffers { check: 0 },
            ],
            Some(WalkEnd::Start),
        ),
        (
            "no myself line, the end's comparison differs",
            vec![
                start_reply(String::new()),
                WalkFault::TablePartitionDiffers { check: 1 },
            ],
            Some(WalkEnd::End),
        ),
        (
            "no myself line only at the end, the end's comparison differs",
            vec![
                // The first end read: one read per primary at the start.
                WalkFault::ClusterNodesReply {
                    read: primaries.len(),
                    reply: String::new(),
                },
                WalkFault::TablePartitionDiffers { check: 0 },
            ],
            Some(WalkEnd::End),
        ),
        (
            "an open slot, a comparison armed",
            vec![
                start_reply(open_slot),
                WalkFault::TablePartitionDiffers { check: 0 },
            ],
            None,
        ),
    ] {
        let armed: Vec<_> = faults
            .into_iter()
            .map(|fault| arm_walk_fault_for_test(&pattern, fault))
            .collect();
        let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
        drop(armed);
        assert_eq!(
            found, written,
            "{case}: every key, the pins being the owners"
        );
        match (stale_at, reason) {
            (Some(expected_at), Some(WalkError::Changed(change))) => match *change {
                ServingChange::StaleTable {
                    ref target,
                    at,
                    ref map,
                    ref table,
                    ref refresh,
                } => {
                    assert_eq!(&**target, pattern.as_str(), "{case}");
                    assert_eq!(at, expected_at, "{case}");
                    assert!(
                        !map.is_empty() && table != "no routing table",
                        "{case}: both groupings named: {map} / {table}"
                    );
                    assert_eq!(refresh, &RoutingRefresh::Refreshed, "{case}");
                }
                other => panic!("{case}: StaleTable, got {other:?}"),
            },
            (None, Some(WalkError::Changed(change))) => {
                assert!(
                    matches!(*change, ServingChange::OpenSlot { .. }),
                    "{case}: the open slot, no comparison made, got {change:?}"
                );
            }
            (_, other) => panic!("{case}: answered after the last key, got {other:?}"),
        }
        assert_eq!(
            every_primary_walked,
            stale_at.is_none(),
            "{case}: every primary walked"
        );
    }
    println!(
        "CLUSTER-GATED RAN: cluster_coverage_walk_whose_unproven_pin_came_from_a_table_that_differs_from_its_map_loses_coverage"
    );
}

/// b.87a, the end map check, on a cluster. The walk seam reads the end
/// `CLUSTER SLOTS` map (read 1) with its first two primaries' names
/// exchanged (`SlotMapPrimariesSwapped`): every range keeps its slots, so
/// every pin is checked as before and every key is returned, but the end
/// map differs from the start's, so the walk answers `SlotMap`, naming both
/// maps, and does not count every primary walked. Control: the same
/// exchange on both reads (0 and 1) makes the two maps agree, and the walk
/// completes with no reason and every primary walked, so the answer comes
/// from the comparison, not from the exchanged names.
///
/// Fail-on-old (stated, not run in this sandbox, which has no cluster):
/// with the map comparison removed from `check_coverage_end`, the walk
/// completes with no reason and every primary walked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_coverage_walk_whose_end_map_differs_from_its_start_ends_slot_map() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    let written = seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let client = tv.client().pool.next();

    let armed = arm_walk_fault_for_test(&pattern, WalkFault::SlotMapPrimariesSwapped { read: 1 });
    let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
    drop(armed);
    assert_eq!(found, written, "every key, before the end map is read");
    match reason {
        Some(WalkError::Changed(change)) => match *change {
            ServingChange::SlotMap {
                ref target,
                ref before,
                ref after,
            } => {
                assert_eq!(&**target, pattern.as_str());
                assert_ne!(before, after, "the two maps are named, and differ");
            }
            other => panic!("SlotMap, got {other:?}"),
        },
        other => panic!("the changed map is answered after the last key, got {other:?}"),
    }
    assert!(
        !every_primary_walked,
        "a map that changed during the walk loses coverage"
    );

    let armed = [
        arm_walk_fault_for_test(&pattern, WalkFault::SlotMapPrimariesSwapped { read: 0 }),
        arm_walk_fault_for_test(&pattern, WalkFault::SlotMapPrimariesSwapped { read: 1 }),
    ];
    let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
    drop(armed);
    assert_eq!(found, written, "control: every key");
    assert!(
        reason.is_none(),
        "control: the same exchange at both ends is no change: {reason:?}"
    );
    assert!(every_primary_walked, "control: every primary walked");
    println!(
        "CLUSTER-GATED RAN: cluster_coverage_walk_whose_end_map_differs_from_its_start_ends_slot_map"
    );
}

/// b.87a, the end re-sync, on a cluster. A coverage walk whose first
/// primary's start `CLUSTER NODES` (read 0) has no myself line leaves that
/// pin's ownership unproven; its tables match its map at both ends, so it
/// records nothing beyond the missing line, but at its end it re-syncs the
/// client's routing table anyway (the comparison cannot see a change of
/// primary that keeps every slot set): every key, `WitnessMissing`, every
/// primary walked, and `resynced_for_test()` true. Control: the same walk
/// with no fault proves every pin from its own myself line, makes no
/// comparison and re-syncs nothing: every key, no reason, every primary
/// walked, `resynced_for_test()` false.
///
/// Fail-on-old (stated, not run in this sandbox, which has no cluster): with
/// the end re-sync after a matching table removed from `check_table`, the
/// first walk's `resynced_for_test()` is `false`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_coverage_walk_with_an_unproven_pin_re_syncs_its_matching_table_at_its_end() {
    let Some(tv) = cluster_fixture().await else {
        return;
    };
    let written = seed_every_shard(&tv).await;
    let pattern = format!("{}*", tv.prefix());
    let client = tv.client().pool.next();

    let armed = arm_walk_fault_for_test(
        &pattern,
        WalkFault::ClusterNodesReply {
            read: 0,
            reply: String::new(),
        },
    );
    let (found, reason, every_primary_walked, resynced) =
        drive_coverage_walk_resyncing(client, &pattern).await;
    drop(armed);
    assert_eq!(found, written, "unproven pin: every key");
    match reason {
        Some(WalkError::WitnessMissing(missing)) => assert_eq!(
            missing.seen,
            WitnessSeen::MyselfLine(MyselfLineSeen::NoLine)
        ),
        other => panic!("unproven pin: the missing myself line, got {other:?}"),
    }
    assert!(
        every_primary_walked,
        "unproven pin: the end re-sync keeps the coverage"
    );
    assert!(
        resynced,
        "unproven pin: the matching table is re-synced at the walk's end"
    );

    let (found, reason, every_primary_walked, resynced) =
        drive_coverage_walk_resyncing(client, &pattern).await;
    assert_eq!(found, written, "control: every key");
    assert!(reason.is_none(), "control: no reason, got {reason:?}");
    assert!(every_primary_walked, "control: every primary walked");
    assert!(
        !resynced,
        "control: a walk whose every pin is proven makes no comparison and no re-sync"
    );
    println!(
        "CLUSTER-GATED RAN: cluster_coverage_walk_with_an_unproven_pin_re_syncs_its_matching_table_at_its_end"
    );
}

/// [`acl_denied_fixture`] on the cluster (a bare `redis-cluster://` URL),
/// denied `CLUSTER NODES` (`+@all -cluster|nodes`: `CLUSTER SLOTS`, `SCAN`
/// and the keys' reads and writes stay allowed); `None`, after a `SKIP:`
/// line, also when there is no cluster.
async fn cluster_nodes_denied_fixture() -> Option<(AclUserGuard, TestValkey)> {
    let Some(url) = test_cluster_url() else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
        return None;
    };
    acl_denied_fixture(
        url,
        TEST_CLUSTER_URL_VAR,
        "redis-cluster://",
        true,
        "scan_guard_nonodes",
        AclDenied::ClusterNodes,
    )
    .await
}

/// b.87a on a store that denies `CLUSTER NODES` by ACL, the store class the
/// table check exists for: the walk's client authenticates as a user denied
/// `cluster|nodes` (the default user untouched), so every pin's
/// `CLUSTER NODES` read fails and every pin's ownership is unproven.
///
/// - Control: with the table matching the map, the walk returns every key,
///   answers the failed read (`WalkError::Read`, `CLUSTER NODES`, an ACL
///   refusal), counts every primary walked, and re-syncs its table at its
///   end (`resynced_for_test()` true).
/// - With the start comparison (check 0) forged to differ
///   (`TablePartitionDiffers`): every key (the pins are the real owners),
///   then `StaleTable` at the start, refreshed, and not every primary walked.
///
/// The ACL user is created on every primary and deleted again on every exit,
/// a panic included (`AclUserGuard`).
///
/// Fail-on-old (stated, not run in this sandbox, which has no cluster): with
/// `leaves_ownership_unproven` answering `false` for a failed read, no
/// comparison is made, and the forged row answers the failed read with
/// every primary walked; with the end re-sync after a matching table
/// removed, the control's `resynced_for_test()` is `false`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_coverage_walk_on_a_client_denied_cluster_nodes_checks_its_table() {
    let Some((_guard, restricted)) = cluster_nodes_denied_fixture().await else {
        return;
    };
    let written = seed_every_shard(&restricted).await;
    let pattern = format!("{}*", restricted.prefix());
    let client = restricted.client().pool.next();

    let (found, reason, every_primary_walked, resynced) =
        drive_coverage_walk_resyncing(client, &pattern).await;
    assert_eq!(found, written, "control: every key");
    match reason {
        Some(WalkError::Read { op, source }) => {
            assert_eq!(op, "CLUSTER NODES");
            let text = source.to_string();
            assert!(
                text.contains("NOPERM") || text.contains("no permissions"),
                "control: the read was refused by the ACL: {text}"
            );
        }
        other => panic!("control: the failed CLUSTER NODES read, got {other:?}"),
    }
    assert!(
        every_primary_walked,
        "control: the table matched, so every primary counts as walked"
    );
    assert!(
        resynced,
        "control: an unproven pin's matching table is re-synced at the walk's end"
    );

    let armed = arm_walk_fault_for_test(&pattern, WalkFault::TablePartitionDiffers { check: 0 });
    let (found, reason, every_primary_walked) = drive_coverage_walk(client, &pattern).await;
    drop(armed);
    assert_eq!(
        found, written,
        "stale table: every key, the pins being the owners"
    );
    match reason {
        Some(WalkError::Changed(change)) => match *change {
            ServingChange::StaleTable {
                at, ref refresh, ..
            } => {
                assert_eq!(at, WalkEnd::Start);
                assert_eq!(refresh, &RoutingRefresh::Refreshed);
            }
            other => panic!("stale table: StaleTable at the start, got {other:?}"),
        },
        other => panic!("stale table: StaleTable, got {other:?}"),
    }
    assert!(
        !every_primary_walked,
        "stale table: an unproven pin from an out-of-date table loses coverage"
    );
    println!(
        "CLUSTER-GATED RAN: cluster_coverage_walk_on_a_client_denied_cluster_nodes_checks_its_table"
    );
}
