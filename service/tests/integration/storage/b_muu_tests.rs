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

//! b.muu — a snapshot read refused PERMANENTLY by something other than a
//! wrong-typed key must move a metric too, and the header cell must stop being
//! race-only.
//!
//! # The gap
//!
//! b.atj gave the snapshot loader's eleven reads (`load_object_snapshot`,
//! since deleted) one counter,
//! `live_edit_snapshot_read_type_error_total{key_family}`, ticked when
//! `storage::client::is_server_wrongtype_error` matches. That left three holes,
//! and b.muu closes all three:
//!
//! 1. **Every other permanent refusal was silent.** A `-NOAUTH`, a protocol
//!    violation, a `-MISCONF` store that cannot persist to disk — each fails
//!    every consumer of an object's state closed, indefinitely, and moved
//!    nothing.
//!    `snapshot_read_type_error_total` must NOT count them (its zero baseline is
//!    "no foreign value on the `le:` keyspace", and the shipped recipe is
//!    `increase(...) > 0`), so the fix is a sibling counter:
//!    `live_edit_snapshot_read_refused_total{kind}`, twelve `kind` cells, ticked
//!    from the same helper on the arm `WRONGTYPE` does not claim.
//! 2. **The checkpoint assembler's four direct reads bypassed the helper.**
//!    `read_header_seq` (`HGET obj:hdr seq`), `read_header_identity` (`HGETALL
//!    obj:hdr`) and `read_tombstone`'s pair called `map_snapshot_load_err`
//!    straight, so the CHECKPOINT consumer counted nothing at all.
//! 3. **The orphan sweep's `obj:hdr` touches were uncounted**, which is the
//!    worst of the three for an operator: the sweep is a background task whose
//!    only other evidence is a log line nobody is watching.
//!
//! Together (2) and (3) are why the `header` cell was effectively **race-only on
//! every consumer** before b.muu: the read that meets a wrong-typed `obj:hdr`
//! first is the assembler's pin read on the checkpoint path and the sweep's
//! discovery read on the maintenance path, and neither counted; only a header
//! that turned bad *between* two reads on that loader's path could
//! reach the counted touch.
//!
//! # Why metric VALUES are asserted here at all
//!
//! Same carve-out `b_atj_tests`' header states, for the same reason: the
//! counter IS the deliverable of the Issue, and a permanent refusal that ticks
//! nothing is precisely the defect. See that file for the full argument; it is
//! not restated here.
//!
//! # What lives here
//!
//! | Test | What only it covers |
//! |---|---|
//! | [`refusal_kinds::snapshot_read_refused_by_a_protocol_error_ticks_kind_protocol`] | the plain kind→cell mapping, on a kind that is a client-library bug |
//! | [`refusal_kinds::snapshot_read_refused_by_an_auth_error_ticks_kind_auth`] | the credential cell, and that `detail` carrying a server auth message is not a reason to suppress the count |
//! | [`refusal_kinds::snapshot_read_refused_by_a_parse_error_ticks_kind_parse`] | a third plain arm, so a `snapshot_refusal_kind_label` that returned a constant fails three tests rather than one |
//! | [`refusal_kinds::snapshot_read_refused_by_a_misconf_reply_ticks_kind_backend_unavailable`] | one of the two arms that are not a plain kind mapping: `Unknown` + a `-MISCONF` details prefix, split out of `unknown` by `is_backend_unavailable_reply`; a store write refusal, so the caller gets a plain `Unavailable` |
//! | [`refusal_kinds::snapshot_read_refused_by_an_oom_reply_ticks_kind_backend_out_of_memory`] | the other: `Unknown` + an `-OOM` details prefix, split out of `unknown` by the SEPARATE `is_backend_out_of_memory_reply`, and deliberately NOT folded into `backend_unavailable` |
//! | [`refusal_kinds::snapshot_read_refused_by_an_unrecognised_server_reply_ticks_kind_unknown`] | the catch-all's other side: `Unknown` with details that match none of the five unavailable tokens and do not start with `OOM` stays `unknown` |
//! | [`refusal_kinds::backend_unavailable_logs_at_warn_while_other_kinds_including_backend_out_of_memory_log_at_error`] | the level split — the deliberate operability decision that keeps a condition with an operator-free path back from training operators to filter the logger out, AND that the gate is that one cell by name rather than "came from the backing store" |
//! | [`refusal_kinds::the_atomic_reads_evalsha_refused_loading_or_busy_is_retriable_and_counts_nothing`] | b.xd1: the store's not-serving `-LOADING` / `-BUSY` on the atomic read are transient — `Aborted`, no cell ticked, no refusal line — while a `-MISCONF` on the same read still is counted |
//! | [`refusal_kinds::a_fan_out_read_refused_loading_or_busy_is_retriable_and_counts_nothing`] | the same on one plain command of the unfenced fan-out |
//! | [`orphan_sweep_discovery_over_a_wrongtype_object_header_ticks_header_exactly_once`] | hole (3): the maintenance consumer, counted through the `metrics` PARAMETER rather than the client hooks, counted ONCE, and pinned to the DISCOVERY read by the refusal line's `op` field — the fence write is unreachable past the discovery abort |
//!
//! Hole (2)'s header half lives in `b_atj_tests` as
//! `checkpoint_assembly_over_a_wrongtype_header_ticks_header_exactly_once`,
//! which b.muu FLIPPED in place from the negative scope pin it used to be.
//! Keeping it there rather than moving it here is deliberate: it is the file
//! that owns `key_family`, and its ten sibling family tests are what its
//! ten-other-cells-at-zero assertion is calibrated against.
//!
//! Every permanent-refusal arm in [`refusal_kinds`] (the six `…_ticks_kind_…`
//! tests) asserts **four** things — one `kind` cell at exactly 1, the other
//! eleven at zero, all eleven `key_family` cells at zero, and
//! `keyspace_type_error_total` flat — plus that the returned `ServiceError` is
//! the mapper's own verdict (`Internal`, or for a store write refusal a plain
//! `Unavailable`), never a retried `Aborted`. The eleven-other-kinds half is
//! what makes the `kind` label discriminating rather than decorative; the
//! `key_family` half is the I4 guard that b.atj's counter kept its meaning,
//! i.e. that the two arms of `count_object_key_refusal` PARTITION the error
//! space instead of overlapping on it.
//!
//! The two b.xd1 arms assert the opposite for the store's not-serving replies:
//! each `-LOADING` / `-BUSY` reaches the caller as a retried `Aborted` and ticks
//! no cell of either counter; only their closing `-MISCONF` control ticks one
//! (`backend_unavailable`, once), which proves the zeros are not vacuous.
//!
//! # What is deliberately NOT here
//!
//! * **The kind vocabulary's own consistency.** That every emittable label is a
//!   registered constant, and that `is_backend_unavailable_reply` /
//!   `is_backend_out_of_memory_reply` are kind-gated and prefix-anchored, are
//!   in-crate unit tests beside the code
//!   (`storage::diff_read::tests::snapshot_read_refusal_kind_consts_are_all_registered`,
//!   `…::snapshot_refusal_kind_label_emits_exactly_the_registered_kind_set`,
//!   `…::is_backend_unavailable_reply_matches_the_five_prefixes_under_unknown_only`,
//!   `…::is_backend_out_of_memory_reply_matches_the_oom_prefix_under_unknown_only`).
//!   They are compile-cheap, need no Valkey, and reach items an integration test
//!   cannot name — the per-kind `const`s are `pub(crate)`, which is why every
//!   label below is a restated literal.
//! * **Zero-seeding of the twelve cells, and that twelve increments open exactly
//!   twelve series.**
//!   `observability_metrics_test::snapshot_read_refused_total_pre_initialises_its_bounded_kind_set`.
//! * **The orphan sweep's retirement-fence arm.** `cleanup_orphaned_object`'s
//!   `HSET obj:hdr superseded 1` failure path is counted by the same helper, and
//!   it is **unstageable from a test**: the sweep's own discovery `HGETALL` on
//!   that exact key runs first and `?`-propagates, so a header bad enough to
//!   fail the fence has already ended the pass, and no fault seam exists on the
//!   fence write (`client.pool.hset` is called directly, with no injector in
//!   front of it). The sweep test below pins the *consequence* of that ordering
//!   — one tick, not two — which is the observable half. Giving the fence its
//!   own seam is a source change and belongs to whoever next needs it.
//! * **The assembler's former `read_tombstone` reads.** b.muu routed them
//!   through the counting helper alongside the header reads, and they were
//!   unreachable from any staging: a wrong-typed tombstone key was always met by
//!   a state read first. They are gone now — every read the assembler runs
//!   proves its cut (the atomic read at or below the atomic-read limits, the
//!   fenced read above them; Spec SDD SR-4.1, SR-11.5), so there is no raced
//!   section to reconcile and no tombstone re-read — and the assembler's direct
//!   reads are the two header reads above.
//! * **The remaining uncounted `obj:hdr` touches.** Five in `storage::layer`
//!   (three header pre-reads, `stat_layer_for_object_id`'s blind read, and the
//!   CreateLayer-overwrite retirement fence) and the whole checkpoint RESTORE
//!   lane are out of b.muu's scope and owned by follow-up Issue **b.iy5**. Until
//!   it lands, `key_family="header"` reading zero does not mean no wrong-typed
//!   header was met — it means none was met on a counted touch.
//! * **The classification of the store's own replies.** b.xd1 made `-LOADING`
//!   and `-BUSY` transient (`storage::client::is_store_not_serving_reply`), so
//!   they are retried and never reach this counter: the two b.xd1 arms in
//!   [`refusal_kinds`] pin that from the counter's side, and the
//!   classification itself is pinned in `transient_classification_test` and
//!   `client.rs`'s unit block. The write-gated three (`-MISCONF`, `-READONLY`,
//!   `-NOREPLICAS`) and `-OOM` stay non-transient, so they are still counted
//!   here, and the mapper answers them a plain `Unavailable` rather than
//!   `Internal` (`storage::client::store_write_refusal`). The write-gated
//!   three refuse writes only, so a real store never answers the atomic read
//!   with one; the injected `-MISCONF` arm below stands in for the orphan
//!   sweep's header fence write, the one counted touch that can draw them
//!   and has no seam (see the fence bullet above).
//!
//! # Activation conditions
//!
//! Everything in [`refusal_kinds`] needs `--features test-fault-injection`: no
//! keyspace can hold a `-NOAUTH` or a `-MISCONF` still, so the only way to reach
//! those arms is `ValkeyClient::install_object_snapshot_read_fault_injector_for_test`.
//! That gate is the SILENT one — without the feature the module is compiled out
//! entirely and the run reports green having covered none of it. The sweep test
//! below plants a real string on a real key and needs no feature.

use std::io::Write;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use fred::interfaces::KeysInterface;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::fmt::MakeWriter;

use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{Metrics, SNAPSHOT_READ_REFUSAL_KINDS};
use live_edit::storage::external_values::{ExternalValueStore, MockExternalValueStore};
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use live_edit::storage::test_support::TestValkey;

// (`assert_no_family_ticked` and `Fixture` are imported inside `refusal_kinds`
// instead: their only callers here are the injected-refusal arms, and that
// module is compiled out without `test-fault-injection`.)
use crate::storage::b_atj_tests::{
    assert_keyspace_type_errors_flat, assert_only_family_ticked, snapshot_read_type_errors,
};

// ═════════════════════════════════════════════════════════════════════════════
// Metric readers — the b.muu half of the pair
// ═════════════════════════════════════════════════════════════════════════════

/// `live_edit_snapshot_read_refused_total{kind}`.
///
/// **Panics on an absent series rather than defaulting to zero**, for the reason
/// `b_atj_tests::snapshot_read_type_errors` gives: all twelve cells are
/// zero-seeded in `Metrics::new`, so absence means either the counter does not
/// exist (the pre-b.muu tree — the intended pre-fix failure) or this `Metrics`
/// handle never reached the code under test, and defaulting to zero would turn
/// both into a green run.
///
/// `pub(crate)` because `b_atj_tests` asserts this counter flat on the arms
/// where `WRONGTYPE` claims the refusal first.
pub(crate) fn snapshot_read_refusals(metrics: &Metrics, kind: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_snapshot_read_refused_total",
        &[("kind", kind)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_snapshot_read_refused_total{{kind=\"{kind}\"}} is absent from the \
             exposition. Either the counter does not exist (pre-b.muu: a permanent non-WRONGTYPE \
             refusal of an object-state key moved nothing, which is the defect this file covers), \
             or this `Metrics` handle never reached `count_object_key_refusal` — on the read paths \
             that means `install_observability_hooks` was not called, and on the orphan sweep it \
             means the handle was not passed to `spawn_orphan_sweep`; either way the tick is \
             silently skipped and every count below reads zero for the wrong reason"
        )
    }) as u64
}

/// Assert that exactly one `kind` cell moved, to exactly `expected_ticks`, and
/// that the other eleven are still at their seeded zero.
///
/// The eleven-zeros half is what makes the label discriminating. A
/// `snapshot_refusal_kind_label` that fell through to `unknown` for everything —
/// the single likeliest way for this mapping to rot, since its `_` arm is
/// deliberately a catch-all rather than an exhaustiveness error — ticks the
/// right *counter* and the wrong *cell*, and is invisible to a test that only
/// checks the expected one.
///
/// Not feature-gated, though most call sites are: a refusal that ticks a
/// *specific* kind has to be staged (no keyspace can hold a `-NOAUTH` still),
/// and this module's `refusal_kinds` arms and `b_atj_tests`'
/// `non_wrongtype_snapshot_read_errors` arms stage it through the
/// `test-fault-injection` seam. `snapshot_read_unfenced_test`'s window-script
/// classification test stages it without the seam — a proxy answers the
/// window script's `EVALSHA` with the error — so the helper has a caller in
/// every feature shape.
pub(crate) fn assert_only_refusal_kind_ticked(
    metrics: &Metrics,
    expected: &str,
    expected_ticks: u64,
) {
    assert!(
        SNAPSHOT_READ_REFUSAL_KINDS.contains(&expected),
        "{expected:?} is not a registered refusal kind — the caller is asserting on a cell that \
         cannot exist, so this test would panic in `snapshot_read_refusals` for a reason that has \
         nothing to do with the behaviour under test"
    );
    assert_eq!(
        snapshot_read_refusals(metrics, expected),
        expected_ticks,
        "live_edit_snapshot_read_refused_total{{kind=\"{expected}\"}} must read {expected_ticks} \
         after the staged refusal. This counter IS the b.muu deliverable: pre-fix a permanent \
         failure of every consumer of an object's state moved nothing at all, so an operator \
         watching the read-side counters saw a flat line through a standing outage"
    );
    for &kind in SNAPSHOT_READ_REFUSAL_KINDS {
        if kind == expected {
            continue;
        }
        assert_eq!(
            snapshot_read_refusals(metrics, kind),
            0,
            "one error was staged, so only kind=\"{expected}\" may move — kind=\"{kind}\" moved \
             too. Either a second refusal was classified (the staging is not the one this test \
             intends) or `snapshot_refusal_kind_label` does not track the error it was handed, \
             which routes an operator to the wrong remediation entirely: a credential fault and a \
             store that cannot persist to disk have nothing in common but the counter they land on"
        );
    }
}

/// Assert every `kind` cell is still at its seeded zero.
pub(crate) fn assert_no_refusal_kind_ticked(metrics: &Metrics) {
    for &kind in SNAPSHOT_READ_REFUSAL_KINDS {
        assert_eq!(
            snapshot_read_refusals(metrics, kind),
            0,
            "no permanent non-WRONGTYPE refusal was staged, so no cell of \
             live_edit_snapshot_read_refused_total may move — kind=\"{kind}\" did. The two arms of \
             `count_object_key_refusal` are ordered and mutually exclusive, so a tick here \
             alongside a `key_family` tick means one refusal was counted twice and both counters' \
             baselines are gone"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Log capture — shared by the level-split arms and the orphan-sweep test
// ═════════════════════════════════════════════════════════════════════════════

/// Capturing `MakeWriter`, mirroring the inline pattern `durability_guard_test`
/// / `txn_checkout_test` / `b_zvd_tests` each use. The duplication across files
/// is the existing project convention: every consumer customizes the level
/// filter and the helper is ~20 lines.
///
/// File-scope rather than inside [`refusal_kinds`] because it has callers on
/// both sides of the `test-fault-injection` gate: the level-split arm inside
/// that module, and
/// [`orphan_sweep_discovery_over_a_wrongtype_object_header_ticks_header_exactly_once`],
/// which stages a real wrong-typed key and needs no feature.
///
/// The two consumers attach it **differently**, and the difference is not
/// stylistic. [`refusal_kinds::capture_warn_and_error`] wraps a future with
/// `WithSubscriber`, which is correct there and only there: those arms await the
/// read inline, so the dispatcher travels with the future across worker threads.
/// The orphan sweep runs in a task `spawn_orphan_sweep` hands to `tokio::spawn`
/// with no `.instrument()` / `.with_subscriber()` of its own, and a spawned task
/// inherits neither the parent's dispatcher nor its span — so a `WithSubscriber`
/// around the test body would capture nothing from it. That test therefore uses
/// a thread-local `set_default` on a CURRENT-THREAD runtime, where the spawned
/// task is polled on the same thread the guard is installed on (the pattern
/// `b_ji8_tests` uses for its `Drop`-spawned teardown).
#[derive(Clone)]
struct CapturingWriter(Arc<StdMutex<Vec<u8>>>);

impl CapturingWriter {
    fn new() -> (Self, Arc<StdMutex<Vec<u8>>>) {
        let buf = Arc::new(StdMutex::new(Vec::new()));
        (Self(buf.clone()), buf)
    }
}

impl<'a> MakeWriter<'a> for CapturingWriter {
    type Writer = CapturingWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The kind mapping, one arm per non-trivially-mapped kind
// ═════════════════════════════════════════════════════════════════════════════

/// `snapshot_refusal_kind_label`, driven through the real
/// `classify_snapshot_read_err` call chain rather than called directly.
///
/// # Why a seam, and what the integration tier adds over the unit tests
///
/// The mapping function itself is unit tested beside the code, exhaustively,
/// over a hand-built `fred::error::Error` per kind. What those tests cannot see
/// is the **wiring**: that the mapper is consulted on the way to the counter at
/// all, that the counter it feeds is the one `Metrics::new` seeded, that the
/// `key_family` counter stays out of it, and that the `ServiceError` the caller
/// receives is the mapper's own verdict, untouched by the counting. Those are
/// four properties of the composed path, and each of them is exactly what a
/// refactor breaks.
///
/// No keyspace can hold a `-NOAUTH` or a protocol violation still, so these arms
/// cannot be staged against a live server.
/// `ValkeyClient::install_object_snapshot_read_fault_injector_for_test` pops one
/// canned `fred::error::Error` in place of the command its target names and
/// feeds it into the **same** `classify_snapshot_read_err` call a real failure
/// of that command takes, so an injected error and a real one provably share
/// one path.
///
/// The permanent-refusal arms and the level-split arm all fail the atomic
/// read's `EVALSHA` (a `Catchup` read), through the staging helper
/// `refuse_snapshot_read_with`, imported from `b_atj_tests` rather than copied,
/// so both files' arms are provably about the same call site. The two b.xd1
/// arms stage through the local `read_refused_with`, which takes the target and
/// the read: the atomic one fails that same `EVALSHA` (`next_read`,
/// `Catchup`), the fan-out one the first command of the unfenced fan-out
/// (`nth_command(UnfencedFanout, 0)`, the header `HGETALL`, on the structural
/// `UpdateLayer` WATCH read, `SlimSnapshotReadConsumer::StructuralUpdate`,
/// which always takes the fan-out).
///
/// The seam is `#[cfg(any(test, feature = "test-fault-injection"))]`, so this
/// module is feature-gated to match — and that gate is the *silent* one: without
/// `--features test-fault-injection` every test below is compiled out entirely
/// and the run reports green having covered none of them.
#[cfg(feature = "test-fault-injection")]
mod refusal_kinds {
    use super::*;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use live_edit::error::ServiceError;
    use live_edit::storage::client::STORE_NOT_SERVING_SHAPES;
    use live_edit::storage::diff_read::{
        read_object_snapshot, read_object_snapshot_slim, ObjectSnapshotReadFaultInjector,
        ObjectSnapshotReadFaultTarget, ObjectSnapshotReader, SlimSnapshotReadConsumer,
        SnapshotReadConsumer,
    };
    use tracing::instrument::WithSubscriber;

    use crate::storage::b_atj_tests::non_wrongtype_snapshot_read_errors::{
        refuse_snapshot_read_with, INJECTED_READ_OP_LABEL,
    };
    use crate::storage::b_atj_tests::{assert_no_family_ticked, Fixture};

    /// Valkey's `-MISCONF` reply (`shared.bgsaveerr`), the `backend_unavailable`
    /// fixture: a write-gated refusal, which a real store sends only to a write
    /// — on this counter's touch set, the orphan sweep's header fence write —
    /// so the read arms below inject it.
    const MISCONF_REPLY: &str = "MISCONF Valkey is configured to save RDB snapshots, but it's \
                                 currently unable to persist to disk.";

    /// The `ServiceError` an injected refusal must reach the caller as:
    /// `map_snapshot_load_err`'s own verdict, which the counting never changes.
    #[derive(Clone, Copy, Debug)]
    enum Verdict {
        /// `ServiceError::Internal` — every permanent refusal but a store write
        /// refusal.
        Internal,
        /// A plain `ServiceError::Unavailable` naming the read and `condition`,
        /// ending `; retry` — a store write refusal (b.xd1). Never `Aborted`: no
        /// retry loop retries it.
        Unavailable { condition: &'static str },
    }

    /// Stage one permanent refusal and assert the whole b.muu contract for one
    /// `kind`.
    ///
    /// Five claims, each separately load-bearing:
    ///
    /// 1. **The classification is the mapper's** — `verdict`, still naming the
    ///    read that failed, and never `Aborted`. `count_object_key_refusal`
    ///    returns `()` precisely so nothing downstream can branch on it; an
    ///    `Aborted` here would mean the counting helper started steering, which
    ///    would turn a standing failure into one the update-layer retry loop
    ///    spins on.
    /// 2. **Exactly the expected `kind` ticked, once.**
    /// 3. **Every other `kind` is still zero.**
    /// 4. **Every `key_family` cell is still zero** — the I4 guard. b.atj's
    ///    counter means "a foreign value sits on the `le:` keyspace", and it
    ///    keeps that meaning only if the two arms partition rather than overlap.
    /// 5. **The commit-side `keyspace_type_error_total` is still flat**, for the
    ///    reason `b_atj_tests::assert_keyspace_type_errors_flat` documents.
    async fn assert_permanent_refusal_ticks_only(
        fixture_name: &str,
        injected: FredError,
        expected_kind: &str,
        verdict: Verdict,
    ) {
        let fx = Fixture::empty(fixture_name).await;

        let err = refuse_snapshot_read_with(&fx, injected).await;

        let msg = match (&err, verdict) {
            (ServiceError::Internal(msg), Verdict::Internal) => msg,
            (ServiceError::Unavailable(msg), Verdict::Unavailable { condition }) => {
                assert_eq!(
                    msg,
                    &format!(
                        "{INJECTED_READ_OP_LABEL}: the backing store refused a write this request \
                         needed because {condition}; retry"
                    ),
                    "a store write refusal is answered a plain UNAVAILABLE naming the read and \
                     the condition (b.xd1)"
                );
                msg
            }
            _ => panic!(
                "a permanent refusal must reach the caller as {verdict:?} across \
                 `classify_snapshot_read_err`. b.muu added counting and nothing else: the helper \
                 borrows the error, returns `()`, and `map_snapshot_load_err` alone produces the \
                 verdict, so an `Aborted` here means the counting helper started classifying and \
                 the update-layer retry loop will now spin against a standing condition; got \
                 {err:?}"
            ),
        };
        assert!(
            msg.contains(INJECTED_READ_OP_LABEL),
            "the refusal must be the injected one, naming the read that failed — any \
             other message means some unrelated read failed first and the counter assertions \
             below describe a different staging entirely; got {msg:?}"
        );

        assert_only_refusal_kind_ticked(&fx.metrics, expected_kind, 1);
        assert_no_family_ticked(&fx.metrics);
        assert_keyspace_type_errors_flat(&fx.metrics);
    }

    /// `protocol` — a RESP framing violation.
    ///
    /// The remediation this cell carries is "a client-library bug, not a
    /// keyspace fault", which is the sharpest possible contrast with the
    /// `key_family` counter's "go and find the out-of-band writer". Sending an
    /// operator the second message for this error is the concrete harm the
    /// partition exists to prevent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_read_refused_by_a_protocol_error_ticks_kind_protocol() {
        assert_permanent_refusal_ticks_only(
            "muu-protocol",
            FredError::new(FredErrorKind::Protocol, "injected RESP framing violation"),
            "protocol",
            Verdict::Internal,
        )
        .await;
    }

    /// `auth` — the backing store rejecting this service's credential.
    ///
    /// Also the arm that pins the deliberate NON-redaction of `detail`. Valkey
    /// answers `-NOAUTH` / `-WRONGPASS` with a fixed sentence and echoes no
    /// credential, and no credential transits this path in the first place, so
    /// the refusal log prints fred's message verbatim exactly as the `WRONGTYPE`
    /// arm does. A redaction on one arm and not the other would make the two
    /// lines disagree about what an operator may see.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_read_refused_by_an_auth_error_ticks_kind_auth() {
        assert_permanent_refusal_ticks_only(
            "muu-auth",
            FredError::new(
                FredErrorKind::Auth,
                "NOAUTH Authentication required (injected)",
            ),
            "auth",
            Verdict::Internal,
        )
        .await;
    }

    /// `parse` — a reply this service could not decode.
    ///
    /// A third plain kind→cell arm. Three of them is the point: a mapping that
    /// collapsed to one constant, or a `Metrics::inc_snapshot_read_refused` that
    /// ignored its argument, fails three tests instead of one, and the failure
    /// reads as "the label does not track the error" rather than as a single
    /// puzzling cell.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_read_refused_by_a_parse_error_ticks_kind_parse() {
        assert_permanent_refusal_ticks_only(
            "muu-parse",
            FredError::new(FredErrorKind::Parse, "injected reply parse failure"),
            "parse",
            Verdict::Internal,
        )
        .await;
    }

    /// `backend_unavailable` — one of the two arms that are not a plain kind
    /// mapping.
    ///
    /// fred's `pretty_error` recognises a fixed set of leading RESP error tokens
    /// and maps everything else to `ErrorKind::Unknown`, preserving the whole
    /// reply in `details`. `-MISCONF` is not in that set, so a store that cannot
    /// persist to disk arrives here indistinguishable *by kind* from a service
    /// bug — which is why `is_backend_unavailable_reply` splits it out on a
    /// kind-gated, prefix-anchored match. That split keeps a condition that can
    /// clear without an operator (the next background save succeeding) out of
    /// `unknown`, whose `increase(...) > 0` alert it would otherwise trip.
    ///
    /// `-MISCONF` is a store write refusal, so the caller gets the mapper's
    /// plain `Unavailable` (b.xd1), not `Internal` and never a retried
    /// `Aborted`. Until b.xd1 this arm was staged with `-LOADING`, which is now
    /// transient and ticks nothing (the two b.xd1 arms at the end of this
    /// module).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_read_refused_by_a_misconf_reply_ticks_kind_backend_unavailable() {
        assert_permanent_refusal_ticks_only(
            "muu-misconf",
            FredError::new(FredErrorKind::Unknown, MISCONF_REPLY),
            "backend_unavailable",
            Verdict::Unavailable {
                condition: "it cannot persist data to disk",
            },
        )
        .await;
    }

    /// `backend_out_of_memory` — the SECOND arm that is not a plain kind
    /// mapping, and the one that proves the `Unknown` split is three-way.
    ///
    /// Valkey answers `-OOM command not allowed when used memory >
    /// 'maxmemory'.` (`valkey/src/server.c`, `shared.oomerr`) for any `denyoom`
    /// command once `used_memory` exceeds `maxmemory`. Like the five
    /// backend-unavailable tokens it is not in fred's recognised set, so it
    /// arrives as `ErrorKind::Unknown` with the whole reply in `details` —
    /// indistinguishable *by kind* from a service bug, and from a `-MISCONF`.
    /// It is a store write refusal, so the caller gets the mapper's plain
    /// `Unavailable` naming the read and the condition (b.xd1).
    ///
    /// # Why it is not simply in the `backend_unavailable` family
    ///
    /// This arm is the one that would go green under the obvious wrong edit, so
    /// it is worth being explicit about what it pins. Folding `-OOM` into
    /// `is_backend_unavailable_reply` as a sixth token is wrong on the property
    /// that family's contract rests on: every member of it has a *path back that
    /// can close without an operator* — the next background save succeeds, the
    /// replica is promoted, the replicas come back — which is what licenses
    /// that cell's WARN level and its sustained-rate alert rather than an
    /// `increase(...) > 0` page. Under the `maxmemory-policy: noeviction` this
    /// project mandates (CLAUDE.md § Deployment) `-OOM` has no such path at
    /// all: nothing is evicted and every `denyoom` command is refused until an
    /// operator raises `maxmemory` or removes data. Folded in, an
    /// unconditionally-operator-action condition routes to the no-page arm and
    /// inherits a WARN.
    ///
    /// Asserting the separate cell here is therefore not a vocabulary
    /// formality — it is the metric half of that routing decision, and
    /// [`backend_unavailable_logs_at_warn_while_other_kinds_including_backend_out_of_memory_log_at_error`]
    /// is the level half.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_read_refused_by_an_oom_reply_ticks_kind_backend_out_of_memory() {
        assert_permanent_refusal_ticks_only(
            "muu-oom",
            FredError::new(
                FredErrorKind::Unknown,
                // Valkey's `shared.oomerr`, verbatim.
                "OOM command not allowed when used memory > 'maxmemory'.",
            ),
            "backend_out_of_memory",
            Verdict::Unavailable {
                condition: "it is out of memory",
            },
        )
        .await;
    }

    /// `unknown` — the catch-all, from the other side.
    ///
    /// The pair to the two arms above: same `ErrorKind::Unknown`, details that
    /// match none of the five backend-unavailable prefixes and do not start with
    /// `OOM`. Without it, either predicate widened to a bare kind check — the
    /// obvious wrong "simplification", since every one of those replies arrives
    /// as `Unknown` — would pass its own arm and be invisible. It would also
    /// quietly empty the `unknown` cell, which is the one an operator reads when
    /// a fred bump adds a kind nothing here maps yet.
    ///
    /// The injected details start with `ERR`, which is deliberate: it is a
    /// prefix of nothing in either predicate's list, so this arm fails the
    /// moment either one stops anchoring.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_read_refused_by_an_unrecognised_server_reply_ticks_kind_unknown() {
        assert_permanent_refusal_ticks_only(
            "muu-unknown",
            FredError::new(FredErrorKind::Unknown, "ERR injected unrecognised reply"),
            "unknown",
            Verdict::Internal,
        )
        .await;
    }

    // ── The level split ─────────────────────────────────────────────────────

    /// Run `future` under a capturing subscriber admitting WARN **and** ERROR,
    /// and hand back everything it emitted.
    ///
    /// `WithSubscriber` rather than `set_default`: the fixtures below run on the
    /// multi-thread runtime and await Valkey round trips, so a thread-local
    /// dispatcher would not survive the task resuming on another worker.
    async fn capture_warn_and_error<F, T>(future: F) -> (T, String)
    where
        F: std::future::Future<Output = T>,
    {
        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            // WARN is the *least* severe level admitted, so ERROR passes too —
            // which is the whole point: this test distinguishes the two, so it
            // must be able to see both.
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        let value = future
            .with_subscriber(tracing::Dispatch::new(subscriber))
            .await;
        let buf = captured.lock().unwrap();
        (value, String::from_utf8_lossy(&buf).into_owned())
    }

    /// The refusal event fires at WARN for `backend_unavailable` and at ERROR
    /// for every other kind — one event, two levels, same field set — and the
    /// gate is that ONE cell by name, not "the backing store answered".
    ///
    /// # Why this is pinned rather than left as observability detail
    ///
    /// `docs-internal/test-writing-guide.md` § What NOT to Test carves out "when
    /// the log line IS the deliverable", and here the *level* is. A store that
    /// cannot persist to disk answers `-MISCONF` to the orphan sweep's fence
    /// write on every pass until a background save succeeds, so emitting ERRORs
    /// for a condition that clears on its own is how an operator learns to
    /// filter this logger out — and the other eleven cells' alert recipes
    /// depend on that logger being read, because the counter says only *that* a
    /// refusal happened and the line says *which key* and *what the server
    /// actually replied*. A blanket downgrade to WARN is the opposite failure
    /// and just as silent, which is why both directions are asserted in one
    /// capture: a level test that only checked the WARN arm would pass against
    /// a helper that had stopped emitting ERROR at all.
    ///
    /// # Why a THIRD fixture, and why it is the `-OOM` one
    ///
    /// `backend_out_of_memory` is the counter-example that keeps "logs at WARN"
    /// from being read as "the backing store answered, so it is not our bug".
    /// An `-OOM` reply is every bit as much the store answering as a `-MISCONF`
    /// is — same `ErrorKind::Unknown`, same RESP error shape, both store write
    /// refusals, same operator asking "is Valkey unhappy?" — and it still
    /// belongs at ERROR, because under the mandated `noeviction` policy only
    /// the GC's out-of-memory collection frees anything (released external
    /// values), a store full of data in use stays refused until an operator
    /// acts, and even one that clears was sized too small: it wants the page.
    /// With only the `-MISCONF` and `protocol` fixtures, a gate rewritten as
    /// "came from the backing store" (the natural-language reading of the WARN
    /// arm, and the likeliest way for this `if` to rot) passes both. This
    /// fixture is the one that fails it.
    ///
    /// Three fixtures inside one capture, because the claim is comparative. The
    /// `kind=` field is asserted on all three lines too — a dashboard or
    /// log-based alert keying on it has to find the same keys whichever level
    /// fired, and the two `tracing` macro call sites (WARN and ERROR) could
    /// drift in their field sets.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backend_unavailable_logs_at_warn_while_other_kinds_including_backend_out_of_memory_log_at_error(
    ) {
        let ((), output) = capture_warn_and_error(async {
            let misconf = Fixture::empty("muu-lvl-misconf").await;
            let _ = refuse_snapshot_read_with(
                &misconf,
                FredError::new(FredErrorKind::Unknown, MISCONF_REPLY),
            )
            .await;

            let protocol = Fixture::empty("muu-lvl-protocol").await;
            let _ = refuse_snapshot_read_with(
                &protocol,
                FredError::new(FredErrorKind::Protocol, "injected RESP framing violation"),
            )
            .await;

            let oom = Fixture::empty("muu-lvl-oom").await;
            let _ = refuse_snapshot_read_with(
                &oom,
                FredError::new(
                    FredErrorKind::Unknown,
                    "OOM command not allowed when used memory > 'maxmemory'.",
                ),
            )
            .await;
        })
        .await;

        let lines: Vec<&str> = output
            .lines()
            .filter(|l| l.contains("object-state touch refused by a permanent backend failure"))
            .collect();
        assert_eq!(
            lines.len(),
            3,
            "exactly the three staged refusals must have emitted the permanent-failure event, one \
             per fixture. Fewer means an injection never reached \
             `count_object_key_refusal`'s second arm and the level assertions below would be \
             vacuous; more means something else in the fixture is refusing too. Captured:\n{output}"
        );

        let warn_line = lines
            .iter()
            .find(|l| l.contains("kind=backend_unavailable"))
            .unwrap_or_else(|| {
                panic!(
                    "no refusal line carried kind=backend_unavailable — either the `-MISCONF` \
                     reply was not split out of `unknown`, or the `kind` field was dropped from \
                     the event, which breaks every log-based alert grouping on it. \
                     Captured:\n{output}"
                )
            });
        assert!(
            warn_line.contains("WARN") && !warn_line.contains("ERROR"),
            "the backend_unavailable refusal must log at WARN. Its members are store conditions \
             that can clear without an operator — a -MISCONF ends when the next background save \
             succeeds — and ERRORs for a condition that clears on its own are how an operator \
             learns to filter this logger out, which costs the other eleven cells the evidence \
             line their alert depends on; got {warn_line:?}"
        );

        let error_line = lines
            .iter()
            .find(|l| l.contains("kind=protocol"))
            .unwrap_or_else(|| {
                panic!(
                    "no refusal line carried kind=protocol — the ERROR half of the split is not \
                     observable, so this test cannot tell a level SPLIT from a blanket downgrade. \
                     Captured:\n{output}"
                )
            });
        assert!(
            error_line.contains("ERROR"),
            "every kind other than backend_unavailable must log at ERROR: those eleven cells \
             should never move at all in a healthy deployment, so each line is a page-worthy \
             event. A blanket downgrade to WARN would be as silent a regression as the missing \
             counter b.muu exists to add; got {error_line:?}"
        );

        let oom_line = lines
            .iter()
            .find(|l| l.contains("kind=backend_out_of_memory"))
            .unwrap_or_else(|| {
                panic!(
                    "no refusal line carried kind=backend_out_of_memory — either the `-OOM` reply \
                     was not split out of `unknown` by `is_backend_out_of_memory_reply`, or it was \
                     folded into the backend_unavailable family, which is the exact edit this \
                     fixture exists to fail. Captured:\n{output}"
                )
            });
        assert!(
            oom_line.contains("ERROR"),
            "the backend_out_of_memory refusal must log at ERROR, even though it IS the backing \
             store answering. That is the whole point of this third fixture: the WARN gate is one \
             named cell, not \"the store replied\". Under the `maxmemory-policy: noeviction` this \
             project mandates nothing is evicted and every denyoom command is refused until \
             memory is freed; the GC's out-of-memory collection frees only released external \
             values, so a store full of data in use stays refused until an operator raises \
             maxmemory or frees memory, and even a store that clears was sized too small — so \
             it wants the page that backend_unavailable deliberately does not get; got \
             {oom_line:?}"
        );
    }

    // ── The store's not-serving replies are transient (b.xd1) ────────────────

    /// The message both levels of the refusal event share.
    const REFUSAL_EVENT: &str = "object-state touch refused by a permanent backend failure";

    /// Which read an arm stages its refusal on: the atomic read of a
    /// consumer that takes it, or the unfenced fan-out, whose one reader is
    /// the structural `UpdateLayer` WATCH path's
    /// (`SlimSnapshotReadConsumer::StructuralUpdate`).
    #[derive(Debug, Clone, Copy)]
    enum ReadUnder {
        Atomic(SnapshotReadConsumer),
        UnfencedFanout,
    }

    /// Arm a one-shot injector answering `injected` in place of the command
    /// `target` names, run one read `under`, and hand back the
    /// `ServiceError` it refused with. Asserts the read refused and the
    /// injected error was consumed, so the caller's claims are about it.
    async fn read_refused_with(
        fx: &Fixture,
        target: ObjectSnapshotReadFaultTarget,
        under: ReadUnder,
        injected: FredError,
    ) -> ServiceError {
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        injector.push_at(target, injected);
        fx.tv
            .client()
            .install_object_snapshot_read_fault_injector_for_test(injector.clone());
        let read = match under {
            ReadUnder::Atomic(consumer) => {
                read_object_snapshot(fx.tv.client(), fx.tv.key_schema(), &fx.oid(), consumer)
                    .await
                    .map(|_| ())
            }
            ReadUnder::UnfencedFanout => read_object_snapshot_slim(
                fx.tv.client(),
                fx.tv.key_schema(),
                &fx.oid(),
                SlimSnapshotReadConsumer::StructuralUpdate,
            )
            .await
            .map(|_| ()),
        };
        let err = match read {
            Err(e) => e,
            Ok(()) => panic!(
                "the injected failure must surface — an `Ok` means the seam never fired, so the \
                 assertions that follow describe an ordinary healthy read"
            ),
        };
        assert_eq!(
            injector.remaining(),
            0,
            "the canned error must have been consumed by this read"
        );
        err
    }

    /// Every [`STORE_NOT_SERVING_SHAPES`] row refusing the command `target`
    /// names: the read answers the retriable `Aborted`, no cell of either
    /// counter ticks, and the refusal event never fires. Then, on the same
    /// fixture and the same command, a `-MISCONF` control ticks
    /// `backend_unavailable` once and logs the event once — proof that this
    /// read path counts and that the capture sees its events, so the zeros
    /// before it are about the classification.
    async fn assert_not_serving_replies_are_retriable_and_uncounted(
        fixture_name: &str,
        target: ObjectSnapshotReadFaultTarget,
        under: ReadUnder,
    ) {
        let fx = Fixture::empty(fixture_name).await;

        let ((), output) = capture_warn_and_error(async {
            for reply in STORE_NOT_SERVING_SHAPES {
                let err = read_refused_with(
                    &fx,
                    target,
                    under,
                    FredError::new(FredErrorKind::Unknown, reply),
                )
                .await;
                let ServiceError::Aborted(msg) = &err else {
                    panic!(
                        "{reply:?} is the store not serving for a while (a node replaying its \
                         data, or a long script): the read must answer the retriable `Aborted` \
                         a retry loop absorbs. Pre-b.xd1 it was `Internal`, which ends the call; \
                         got {err:?}"
                    );
                };
                assert!(
                    msg.starts_with("transient I/O:") && msg.contains(reply),
                    "the `Aborted` must carry the `transient I/O:` retry prefix and the \
                     injected reply; got {msg:?}"
                );
            }
            assert_no_refusal_kind_ticked(&fx.metrics);
            assert_no_family_ticked(&fx.metrics);
            assert_keyspace_type_errors_flat(&fx.metrics);

            let control = read_refused_with(
                &fx,
                target,
                under,
                FredError::new(FredErrorKind::Unknown, MISCONF_REPLY),
            )
            .await;
            assert!(
                matches!(control, ServiceError::Unavailable(_)),
                "the -MISCONF control is a store write refusal: plain UNAVAILABLE; got \
                 {control:?}"
            );
        })
        .await;

        assert_only_refusal_kind_ticked(&fx.metrics, "backend_unavailable", 1);
        let lines: Vec<&str> = output
            .lines()
            .filter(|l| l.contains(REFUSAL_EVENT))
            .collect();
        assert_eq!(
            lines.len(),
            1,
            "only the -MISCONF control may emit the refusal event: a -LOADING / -BUSY is retried \
             and is not a permanent backend failure. Zero lines means the capture saw nothing \
             and the absence above proves nothing. Captured:\n{output}"
        );
        assert!(
            lines[0].contains("MISCONF"),
            "the one refusal line must be the control's; got {:?}",
            lines[0]
        );
    }

    /// b.xd1 on the atomic read: `-LOADING` / `-BUSY` answering its one
    /// `EVALSHA` are retriable and counted nowhere.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_atomic_reads_evalsha_refused_loading_or_busy_is_retriable_and_counts_nothing() {
        assert_not_serving_replies_are_retriable_and_uncounted(
            "muu-xd1-atomic",
            ObjectSnapshotReadFaultTarget::next_read(),
            ReadUnder::Atomic(SnapshotReadConsumer::Catchup),
        )
        .await;
    }

    /// b.xd1 on one plain command of the unfenced fan-out (its first, round
    /// 1's header `HGETALL`; the structural `UpdateLayer` WATCH read always
    /// takes the fan-out).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_fan_out_read_refused_loading_or_busy_is_retriable_and_counts_nothing() {
        assert_not_serving_replies_are_retriable_and_uncounted(
            "muu-xd1-fanout",
            ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::UnfencedFanout, 0),
            ReadUnder::UnfencedFanout,
        )
        .await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Hole (3): the orphan sweep
// ═════════════════════════════════════════════════════════════════════════════

/// How long one sweep pass has to reach the planted header. Generous: the
/// healthy path ticks in milliseconds and pays none of this, so the budget is
/// sized to be unreachable under ordinary CI-runner load rather than to be
/// tight.
const SWEEP_BUDGET: Duration = Duration::from_secs(10);

/// Poll interval inside [`SWEEP_BUDGET`] — not an additional wait.
const SWEEP_POLL: Duration = Duration::from_millis(100);

/// Budget for the sweep task to exit after cancellation.
const SWEEP_JOIN_BUDGET: Duration = Duration::from_secs(5);

/// A synthetic object id, planted directly rather than created: the sweep
/// discovers objects by SCANning key NAMES, so a bare string at the header key
/// is enough to be discovered, and no layer is needed.
const PLANTED_OBJECT_ID: &str = "424242";

/// The `op` field the sweep's DISCOVERY read labels its refusal with, restated
/// as a literal because the source constant is an inline string in
/// `storage::orphan_sweep`.
///
/// This is the only thing in the emitted evidence that distinguishes the sweep's
/// two counted `obj:hdr` touches: both carry `key_family=header`, the same
/// `object_id` and the same `key`, so an edit that made the *fence* answer
/// instead would produce a byte-identical line but for this field.
const SWEEP_DISCOVERY_OP: &str = "HGETALL obj:hdr (orphan discovery)";

/// The `op=` prefix the sweep's RETIREMENT FENCE would label its refusal with
/// (`HSET obj:hdr superseded (retirement fence)`), truncated to the shortest
/// token no other part of the line can carry.
///
/// Asserted absent alongside [`SWEEP_DISCOVERY_OP`] being present, because the
/// positive half alone is drift-fragile: the refusal line also carries
/// `SNAPSHOT_READ_TYPE_ERROR_REMEDIATION`, whose prose already names the
/// discovery read, and it spells the field `When op is HGETALL obj:hdr (orphan
/// discovery)` — one `=` away from satisfying the positive `contains` on its
/// own, at which point the whole discrimination goes vacuous. Neither
/// remediation constant in `storage::diff_read` contains `HSET` in any form, so
/// this token can only come from the fence's own `op` field.
const SWEEP_FENCE_OP_PREFIX: &str = "op=HSET";

/// The orphan sweep's discovery `HGETALL obj:hdr` over a wrong-typed header
/// ticks `key_family="header"` — exactly once, through the `metrics` PARAMETER.
///
/// # What was broken
///
/// Pre-b.muu this touch moved nothing. That is the worst of the three holes for
/// an operator: the sweep is a background task on a 300 s cadence whose only
/// other evidence is a `warn!` nobody is watching, so a wrong-typed `obj:hdr`
/// stopped orphan reaping outright — every subsequent pass aborts at the same
/// key — with no signal anywhere. Worse, the sweep's `?` is what makes this the
/// *whole pass*, not one object: a single bad header suspends reaping for the
/// entire keyspace.
///
/// # Why no observability hooks are installed, and why that is the sharp part
///
/// Every other test in this file and in `b_atj_tests` wires `Metrics` through
/// `ValkeyClient::install_observability_hooks`, because
/// `classify_snapshot_read_err` reads the handle from
/// `client.metrics_snapshot()`. The sweep does **not**: it threads
/// `Option<&SharedMetrics>` down as a function parameter from
/// `spawn_orphan_sweep`, and this test deliberately passes the handle ONLY
/// there. A `count_object_key_refusal` keyed on `client.metrics_snapshot()`
/// instead of an explicit argument would compile, would pass every other test in
/// this file, and would count nothing here — which is exactly the trap the
/// helper's signature was shaped to avoid, so this is the test that proves the
/// shape.
///
/// # Why EXACTLY once
///
/// The sweep has two counted `obj:hdr` touches: this discovery read and
/// `cleanup_orphaned_object`'s retirement fence `HSET obj:hdr superseded 1`.
/// They cannot both fire on one object. `sweep_orphaned_objects` is the fence's
/// only caller and reaches it only past this read, which `?`-propagates — so a
/// header bad enough to fail the fence has already ended the pass here. `== 1`
/// is what would catch a future edit that moved the fence ahead of the
/// discovery read, or added a retry around it, and started paging an operator
/// twice for one key. It is also why the fence arm has no test of its own: see
/// the module header.
///
/// The 300 s sweep interval means exactly one pass runs inside the budget, so
/// the count cannot be inflated by a second tick either.
///
/// # Why the emitted `op` is asserted, and not just the count
///
/// "Exactly one tick" says one of the two touches answered; it does not say
/// WHICH. Both carry `key_family=header` on the same key of the same object, so
/// the counter cannot tell them apart, and the fence is unstageable (see the
/// module header) — there is no negative test to bound it from the other side.
/// The refusal line's `op` field is the only evidence that distinguishes them,
/// so it is asserted here: [`SWEEP_DISCOVERY_OP`], not the fence's `HSET
/// obj:hdr superseded (retirement fence)`. Without it, an edit that reordered
/// the fence ahead of the discovery read — the exact regression the `== 1` half
/// is aimed at — could keep the count at one and go unnoticed.
///
/// # Why this test runs on a CURRENT-THREAD runtime
///
/// `spawn_orphan_sweep` hands its loop to a bare `tokio::spawn`, with no
/// `.instrument()` and no `.with_subscriber()`, and a spawned task inherits
/// neither the spawning task's dispatcher nor its span. So the `WithSubscriber`
/// wrapper [`refusal_kinds`] uses — correct there, where the read is awaited
/// inline — would capture nothing at all here, and the `op` assertion would be
/// vacuous rather than red. On a current-thread runtime the spawned task is
/// polled on the same thread that installed the thread-local `set_default`
/// guard, so it is in scope for the sweep's own events; `b_ji8_tests` uses the
/// same construction for its `Drop`-spawned teardown. Nothing here needs
/// parallelism: the test awaits a background task, it does not race one.
#[tokio::test]
async fn orphan_sweep_discovery_over_a_wrongtype_object_header_ticks_header_exactly_once() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-b-muu-sweep"));
    // Deliberately NO `install_observability_hooks` — see the doc-comment.

    let header_key = tv.key_schema().obj_header(PLANTED_OBJECT_ID);
    let _: () = tv
        .client()
        .pool
        .set(
            &header_key,
            "planted-by-an-out-of-band-writer",
            None,
            None,
            false,
        )
        .await
        .unwrap_or_else(|e| panic!("planting a string at {header_key} must succeed: {e}"));

    assert_eq!(
        snapshot_read_type_errors(&metrics, "header"),
        0,
        "the fixture must start from the seeded zero, or the exactly-once claim below is not a \
         claim about this sweep"
    );

    // ERROR is the level the WRONGTYPE arm emits at; the fmt default of INFO
    // would admit it anyway, but pinning the filter keeps the capture from
    // depending on that default. The guard is thread-local and the runtime is
    // current-thread, so the task spawned below is polled under it — see the
    // doc-comment.
    let (writer, captured) = CapturingWriter::new();
    let capture = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::ERROR)
            .with_ansi(false)
            .finish(),
    );

    let cancel = CancellationToken::new();
    let ext_store: Arc<dyn ExternalValueStore> = Arc::new(MockExternalValueStore::new());
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        tv.key_schema().clone(),
        ext_store,
        Arc::new(live_edit::storage::layer::LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            0,
        )),
        crate::common::test_leader("b-muu-replica"),
        cancel.clone(),
        Some(Arc::clone(&metrics)),
    );

    // One deadline for the whole wait (test-writing guide § Synchronization),
    // polling the observable the sweep produces rather than sleeping a fixed
    // amount: the tick IS the completion signal here, because the pass aborts
    // at the planted key and writes nothing else.
    let deadline = tokio::time::Instant::now() + SWEEP_BUDGET;
    while snapshot_read_type_errors(&metrics, "header") == 0 {
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(SWEEP_POLL).await;
    }

    cancel.cancel();
    tokio::time::timeout(SWEEP_JOIN_BUDGET, handle)
        .await
        .expect("the orphan sweep must exit within the join budget of cancellation")
        .expect("the orphan sweep must not panic");

    drop(capture);
    let output = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();

    assert_only_family_ticked(&metrics, "header", 1);
    // A `WRONGTYPE` is claimed by the helper's first arm and returns, so the
    // permanent/transient split is never consulted and the b.muu counter must
    // stay flat. The sweep is the one consumer where both counters are wired
    // through the same parameter, so this is also the place a double-tick would
    // be easiest to introduce.
    assert_no_refusal_kind_ticked(&metrics);
    assert_keyspace_type_errors_flat(&metrics);

    // Which of the sweep's two counted `obj:hdr` touches answered. See the
    // doc-comment: the counter cannot tell them apart and the fence has no test
    // of its own, so this field is the whole discrimination.
    let refusal_lines: Vec<&str> = output
        .lines()
        .filter(|l| l.contains("object-state touch refused: the named key holds the wrong Valkey"))
        .collect();
    assert_eq!(
        refusal_lines.len(),
        1,
        "the sweep must have emitted exactly one wrong-type refusal line, matching the single \
         tick asserted above. Zero means the capture never saw the sweep's events — most likely \
         the task stopped being polled on this thread, which would make the `op` assertion below \
         vacuous rather than red — and more than one means a second touch refused. Captured:\n\
         {output}"
    );
    assert!(
        refusal_lines[0].contains(&format!("op={SWEEP_DISCOVERY_OP}")),
        "the refusal must come from the sweep's DISCOVERY read, labelled {SWEEP_DISCOVERY_OP:?}. \
         Its retirement fence (`HSET obj:hdr superseded (retirement fence)`) counts the same \
         `key_family=header` cell on the same key of the same object, so the exactly-once count \
         above cannot tell the two apart; this field can. A fence label here means the fence now \
         runs BEFORE the discovery read `?`-propagates — the reordering the count was aimed at, \
         landing on an object whose header nothing has successfully read; got \
         {line:?}",
        line = refusal_lines[0]
    );
    // Negative half: keeps the positive `contains` above from going vacuous if
    // the remediation prose on the same line ever spells the field `op=` (see
    // `SWEEP_FENCE_OP_PREFIX`).
    assert!(
        !refusal_lines[0].contains(SWEEP_FENCE_OP_PREFIX),
        "the refusal line must NOT carry the retirement fence's `op` (it begins \
         {SWEEP_FENCE_OP_PREFIX:?}). Together with the positive assertion above this pins the \
         discovery-vs-fence discrimination from both sides, so neither a fence-labelled refusal \
         nor a drift in the remediation prose that made the positive match satisfiable by \
         something other than the `op` field can pass; got {line:?}",
        line = refusal_lines[0]
    );

    // The planted key must still be there: the pass aborted before any UNLINK,
    // which is the fail-closed behaviour b.muu left untouched. An absent key
    // would mean the sweep reaped an object it could not read, and the tick
    // above would be describing a different code path.
    let still_present: bool = tv
        .client()
        .pool
        .exists(&header_key)
        .await
        .expect("EXISTS must succeed");
    assert!(
        still_present,
        "the wrong-typed header must survive the pass — `sweep_orphaned_objects` `?`-propagates \
         the refusal, so nothing downstream of the discovery read runs. A reaped key here means \
         the sweep proceeded on a header it never read, which is a data-loss shape and not a \
         counting question"
    );
}
