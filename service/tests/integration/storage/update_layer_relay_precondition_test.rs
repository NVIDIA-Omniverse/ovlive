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

//! `update_layer`'s relay precondition of the `ompf`-presence invariant (SDD
//! §12.2), through the public storage API
//! (`LayerStore::update_layer_with_summary` and its wrappers): every
//! delta-carrying commit must carry non-empty relay OMPF bytes, on every
//! backend.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`update_layer_delta_without_relay_bytes_is_refused_internal_before_any_store_io_on_every_backend`] | a fast-path and a structural delta, each with a `None` relay and with zero-length relay bytes, on `lua`, `lua-watch-structural` and `txn`: refused `Internal` naming the precondition, with no summary and one `error` commit-duration sample; the store's `MONITOR` feed shows no command naming the object, and its keyspace is unchanged |
//! | [`before_admission::update_layer_lua_delta_without_relay_bytes_is_refused_before_the_admission_wait`] (`test-fault-injection`) | the refusal comes before admission, which is in-memory and so invisible to `MONITOR` and keyspace checks: on `lua` with one writer per object, while a structural call parked before its commit holds the object's `structural` permit and its only `writers` permit, a relay-less call of either class is refused `Internal` at once rather than queuing behind it |
//! | [`update_layer_no_delta_with_zero_length_relay_bytes_is_the_plain_empty_commit_on_every_backend`] | `(None, Some(empty))` is normalized to the empty commit `(None, None)`: one stream entry with a `payload` and no `ompf` field on every backend — the WATCH route used to write a zero-length `ompf` field |
//! | [`update_layer_no_delta_with_non_empty_relay_bytes_writes_them_as_ompf_on_every_backend`] | `(None, Some(non-empty))` — a container with empty delta bytes and one external part — is accepted, and its entry carries exactly those bytes as `ompf` on every backend |
//! | [`update_layer_lua_fast_commit_then_structural_commit_never_moves_next_set_order_backwards`] | the defect the precondition closes: a fast-path field commit sent without relay bytes, then a structural commit on the same replica, must not move `obj:hdr next_set_order` backwards; and with relay bytes the cached entry tracks the cursor the fast commit raised |
//!
//! The relay bytes a delta-carrying call sends are the delta's own OMPF
//! (`storage::test_support::relay::relay_ompf`), as the `UpdateLayer` handler
//! derives them from the request body; the no-delta test builds its container
//! directly.
//!
//! # Wait budget
//!
//! Every call is a few round trips. The `MONITOR` drain waits at most
//! [`MONITOR_DRAIN_BUDGET`] (20 s) per refused call, twelve calls each in
//! their own fixture — but a drain returns as soon as the fence arrives,
//! milliseconds after the call. The admission test waits at most
//! [`before_admission::REFUSAL_BUDGET`] (5 s) per refused call, half a second
//! for its control call to show it queues, and [`MONITOR_DRAIN_BUDGET`] for
//! its holder to park and to commit. The healthy path takes about a second.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};
use oodf_ffi::OwnedDelta;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::render;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::relay::relay_ompf;
use live_edit::storage::test_support::structural_commit::{dump_object_keyspace, CommitTokens};
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::delta_with_set_field_inline;
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, every_shape_state, structural, ROOT, WORLD,
};
use crate::storage::snapshot_read_support::{
    monitor_fence, test_valkey_addr, MonitorTap, MONITOR_DRAIN_BUDGET,
};

const BACKENDS: [UpdateLayerBackend; 3] = [
    UpdateLayerBackend::Lua,
    UpdateLayerBackend::LuaWatchStructural,
    UpdateLayerBackend::Txn,
];
const WRITER: &str = "relay-precondition-writer";
/// The start of the refusal's message: the call names the precondition.
const REFUSAL: &str = "update_layer called with a delta and no non-empty relay OMPF";
/// A section id the seeded tree does not use.
const NEW_SECTION: u64 = 70;

/// One object on a store with `backend`, its registry hooked onto the store
/// and its client.
struct Fx {
    tv: TestValkey,
    metrics: Arc<Metrics>,
    ls: Arc<LayerStore>,
    session_id: String,
    layer: &'static str,
    oid: u64,
}

impl Fx {
    async fn new(layer: &'static str, backend: UpdateLayerBackend) -> Self {
        Self::configured(layer, |ls| ls.with_update_layer_backend(backend)).await
    }

    /// [`Self::new`] on a store `configure` builds from a plain one.
    async fn configured(
        layer: &'static str,
        configure: impl FnOnce(LayerStore) -> LayerStore,
    ) -> Self {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let ls = Arc::new(
            configure(LayerStore::new(
                tv.client().clone(),
                tv.key_schema().clone(),
                4096,
            ))
            .with_metrics(Arc::clone(&metrics)),
        );
        let session_id = create_session(&tv, layer).await;
        let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
        Self {
            tv,
            metrics,
            ls,
            session_id,
            layer,
            oid,
        }
    }

    /// The stream's entries, field names and values as bytes (a relay
    /// `ompf` field is binary).
    async fn stream(&self) -> Vec<(String, HashMap<String, Bytes>)> {
        self.tv
            .client()
            .pool
            .xrange_values(self.tv.key_schema().layer_stream(self.oid), "-", "+", None)
            .await
            .expect("XRANGE over the layer stream must succeed")
    }

    async fn next_set_order(&self) -> u64 {
        let value: Option<String> = self
            .tv
            .client()
            .pool
            .hget(
                self.tv.key_schema().obj_header(&self.oid.to_string()),
                "next_set_order",
            )
            .await
            .expect("HGET obj:hdr next_set_order must succeed");
        value
            .expect("the header carries next_set_order")
            .parse()
            .expect("next_set_order is decimal")
    }

    /// `live_edit_update_layer_commit_duration_seconds_count` summed over
    /// every `path`, for `outcome`.
    fn duration_samples(&self, outcome: &str) -> u64 {
        let needle = format!("outcome=\"{outcome}\"");
        render(&self.metrics)
            .lines()
            .filter(|l| {
                l.starts_with("live_edit_update_layer_commit_duration_seconds_count{")
                    && l.contains(&needle)
            })
            .filter_map(|l| l.rsplit_once(' ')?.1.parse::<f64>().ok())
            .sum::<f64>() as u64
    }
}

/// A delta of each class the dispatch tells apart: a field write (the fast
/// path on the Lua backends) and a `CreateSection` (the structural route on
/// `lua`, the WATCH path on the others).
fn deltas() -> [(&'static str, OwnedDelta); 2] {
    [
        (
            "field write",
            delta_with_set_field_inline(WORLD, "relay", vec![0x52], 1),
        ),
        (
            "CreateSection",
            structural(vec![create_cmd(ROOT, NEW_SECTION, "NoRelay")]),
        ),
    ]
}

/// A delta with no relay bytes, or with zero-length ones, is refused
/// `Internal` before admission and before any store I/O (`update_layer`'s
/// "`relay_ompf` parameter contract"), on every backend and for both delta
/// classes: the answer names the precondition and carries no summary (no
/// attempt started); exactly one `error` commit-duration sample is recorded
/// and no `ok` one; the node's `MONITOR` feed, from before the call to a
/// fence issued after it, holds no command naming the test's keys; and the
/// object's whole keyspace — header, stream, every row — is unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_delta_without_relay_bytes_is_refused_internal_before_any_store_io_on_every_backend(
) {
    for backend in BACKENDS {
        for (what, delta) in deltas() {
            for (relay_case, relay) in [("no relay", None), ("empty relay", Some(Bytes::new()))] {
                let case = format!("{backend:?}, {what}, {relay_case}");
                let fx = Fx::new("relay-refused", backend).await;
                let before = dump_object_keyspace(
                    fx.tv.client(),
                    fx.tv.key_schema(),
                    fx.oid,
                    CommitTokens::Verbatim,
                )
                .await;
                let mut tap = MonitorTap::start(&test_valkey_addr()).await;

                let (result, summary) = tokio::time::timeout(
                    MONITOR_DRAIN_BUDGET,
                    fx.ls.update_layer_with_summary(
                        &fx.session_id,
                        fx.layer,
                        fx.oid,
                        std::sync::Arc::new(Some(delta.clone())),
                        relay,
                        WRITER,
                        1,
                        None,
                        &mut live_edit::storage::commit_charge::NoCommitCharge,
                    ),
                )
                .await
                .unwrap_or_else(|_| panic!("{case}: the call must settle"));

                let fence = monitor_fence("relay-precondition:");
                let _: i64 = fx
                    .tv
                    .client()
                    .pool
                    .exists(&fence)
                    .await
                    .expect("EXISTS the fence must succeed");
                let seen = tap.drain_until(&fence).await;
                match &result {
                    Err(ServiceError::Internal(message)) => assert!(
                        message.starts_with(REFUSAL),
                        "{case}: the refusal names the precondition; got {message:?}"
                    ),
                    other => panic!(
                        "{case}: a delta without relay bytes is refused Internal; got {other:?}"
                    ),
                }
                assert!(
                    summary.is_none(),
                    "{case}: no attempt started, so there is no call summary; got {summary:?}"
                );
                assert_eq!(
                    (fx.duration_samples("error"), fx.duration_samples("ok")),
                    (1, 0),
                    "{case}: the refused call records one error commit-duration sample"
                );
                let touched: Vec<_> = seen
                    .iter()
                    .filter(|c| c.args.iter().any(|a| a.contains(fx.tv.prefix())))
                    .collect();
                assert!(
                    touched.is_empty(),
                    "{case}: the refusal comes before any store I/O; the call issued {touched:?}"
                );
                assert_eq!(
                    dump_object_keyspace(
                        fx.tv.client(),
                        fx.tv.key_schema(),
                        fx.oid,
                        CommitTokens::Verbatim
                    )
                    .await,
                    before,
                    "{case}: nothing was written"
                );
            }
        }
    }
}

/// The refusal's place before admission. The pre-dispatch hook it parks a
/// holder on exists only under `test-fault-injection`, so without that
/// feature this module compiles to nothing.
#[cfg(feature = "test-fault-injection")]
mod before_admission {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use futures::FutureExt;
    use live_edit::storage::watched_txn::WatchWindowHook;
    use tokio::sync::Notify;

    use super::*;

    /// How long a refused call may take while another call holds the
    /// object's admission permits. A refusal returns in microseconds; a call
    /// that waited on admission instead would wait until the holder is
    /// released, which the test does only after every refusal.
    pub(super) const REFUSAL_BUDGET: Duration = Duration::from_secs(5);

    /// How long the fixture's control call is given to show it queues behind
    /// the holder.
    const QUEUED_PROBE: Duration = Duration::from_millis(500);

    /// The refusal comes before the admission wait (`update_layer`'s
    /// "`relay_ompf` parameter contract"). Admission is in-memory, so the
    /// `MONITOR` and keyspace checks above cannot see where the refusal sits;
    /// a held permit can.
    ///
    /// On `lua` with one writer per object, a structural call is parked by
    /// the pre-dispatch hook after it planned its commit and before its
    /// `EVALSHA`: it holds the object's `structural` permit and its only
    /// `writers` permit. A relayed call queues behind it (the fixture's
    /// control). A relay-less call of each class — a field write, which needs
    /// a `writers` permit, and a `CreateSection`, which needs the
    /// `structural` one first — with no relay and with zero-length relay
    /// bytes, is refused `Internal` naming the precondition within
    /// [`REFUSAL_BUDGET`], with no call summary, while the holder is still
    /// parked. Released, the holder commits.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_lua_delta_without_relay_bytes_is_refused_before_the_admission_wait() {
        let parked = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let fired = Arc::new(AtomicUsize::new(0));
        let hook: WatchWindowHook = {
            let (parked, release, fired) = (
                Arc::clone(&parked),
                Arc::clone(&release),
                Arc::clone(&fired),
            );
            Arc::new(move |_eval_client: &fred::clients::Client| {
                let first = fired.fetch_add(1, Ordering::SeqCst) == 0;
                let (parked, release) = (Arc::clone(&parked), Arc::clone(&release));
                async move {
                    if first {
                        parked.notify_one();
                        release.notified().await;
                    }
                }
                .boxed()
            })
        };
        let fx = Fx::configured("relay-before-admission", |ls| {
            ls.with_update_layer_backend(UpdateLayerBackend::Lua)
                .with_update_layer_per_object_concurrency(1)
                .with_structural_commit_pre_dispatch_hook_for_test(hook)
        })
        .await;

        let holder = {
            let (ls, session_id, layer, oid) =
                (Arc::clone(&fx.ls), fx.session_id.clone(), fx.layer, fx.oid);
            let delta = structural(vec![create_cmd(ROOT, NEW_SECTION, "Holder")]);
            let relay = Some(relay_ompf(&delta));
            tokio::spawn(async move {
                ls.update_layer(&session_id, layer, oid, Some(delta), relay, "holder", 1)
                    .await
            })
        };
        tokio::time::timeout(MONITOR_DRAIN_BUDGET, parked.notified())
            .await
            .expect("fixture: the holder must park before its commit");

        let field = delta_with_set_field_inline(WORLD, "queued", vec![0x51], 1);
        let relay = Some(relay_ompf(&field));
        let queued = tokio::time::timeout(
            QUEUED_PROBE,
            fx.ls.update_layer(
                &fx.session_id,
                fx.layer,
                fx.oid,
                Some(field),
                relay,
                WRITER,
                1,
            ),
        )
        .await;
        assert!(
            queued.is_err(),
            "fixture: a relayed call must queue behind the parked holder, so the holder holds \
             the object's only writers permit; got {queued:?}"
        );

        for (what, delta) in deltas() {
            for (relay_case, relay) in [("no relay", None), ("empty relay", Some(Bytes::new()))] {
                let case = format!("{what}, {relay_case}");
                let (result, summary) = tokio::time::timeout(
                    REFUSAL_BUDGET,
                    fx.ls.update_layer_with_summary(
                        &fx.session_id,
                        fx.layer,
                        fx.oid,
                        std::sync::Arc::new(Some(delta.clone())),
                        relay,
                        WRITER,
                        1,
                        None,
                        &mut live_edit::storage::commit_charge::NoCommitCharge,
                    ),
                )
                .await
                .unwrap_or_else(|_| {
                    panic!(
                        "{case}: a delta without relay bytes must be refused before the \
                         admission wait; it waited {REFUSAL_BUDGET:?} behind the parked holder"
                    )
                });
                match &result {
                    Err(ServiceError::Internal(message)) => assert!(
                        message.starts_with(REFUSAL),
                        "{case}: the refusal names the precondition; got {message:?}"
                    ),
                    other => panic!(
                        "{case}: a delta without relay bytes is refused Internal; got {other:?}"
                    ),
                }
                assert!(
                    summary.is_none(),
                    "{case}: no attempt started, so there is no call summary; got {summary:?}"
                );
            }
        }
        assert!(
            !holder.is_finished(),
            "fixture: the holder must still be parked after every refusal"
        );

        release.notify_one();
        tokio::time::timeout(MONITOR_DRAIN_BUDGET, holder)
            .await
            .expect("the released holder must finish")
            .expect("the holder's task must not panic")
            .expect("the released holder must commit");
        assert_eq!(
            fired.load(Ordering::SeqCst),
            1,
            "only the holder reached the structural commit"
        );
    }
}

/// `(None, Some(empty))`, a call with no delta and zero-length relay bytes,
/// is the plain empty commit `(None, None)` on every backend: it commits one
/// stream entry whose only field is the `payload` — no `ompf` field, the one
/// stored form every reader treats as the empty commit. Before the
/// normalization the WATCH route wrote a zero-length `ompf` field.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_no_delta_with_zero_length_relay_bytes_is_the_plain_empty_commit_on_every_backend(
) {
    for backend in BACKENDS {
        let fx = Fx::new("relay-empty-commit", backend).await;
        let entries_before = fx.stream().await.len();

        let (seq, _) = fx
            .ls
            .update_layer(
                &fx.session_id,
                fx.layer,
                fx.oid,
                None,
                Some(Bytes::new()),
                WRITER,
                1,
            )
            .await
            .unwrap_or_else(|e| panic!("{backend:?}: the empty commit must land; got {e:?}"));

        let entries = fx.stream().await;
        assert_eq!(entries.len(), entries_before + 1, "{backend:?}: one entry");
        let (id, fields) = entries.last().expect("the commit's entry");
        assert_eq!(id, &format!("{seq}-0"), "{backend:?}: the commit's entry");
        assert_eq!(
            fields.keys().map(String::as_str).collect::<BTreeSet<_>>(),
            BTreeSet::from(["payload"]),
            "{backend:?}: the empty commit's entry carries no `ompf` field; got {fields:?}"
        );
    }
}

/// `(None, Some(non-empty))`, a call with no delta and an OMPF container
/// whose delta bytes are empty, is accepted (`update_layer`'s "`relay_ompf`
/// parameter contract"), and its stream entry carries exactly those bytes as
/// its `ompf` field, on every backend. The container here holds one external
/// part, so it is non-empty though its delta is. The zero-length filter that
/// normalizes `(None, Some(empty))` must not drop these bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_no_delta_with_non_empty_relay_bytes_writes_them_as_ompf_on_every_backend() {
    let external_part: &[u8] = b"relay-precondition external part";
    let relay = Bytes::from(
        OmpfContainer::serialize(&[], &[external_part])
            .expect("a container with one external part serializes"),
    );
    assert!(
        !relay.is_empty(),
        "fixture: a container with an external part is non-empty"
    );
    for backend in BACKENDS {
        let fx = Fx::new("relay-no-delta", backend).await;
        let entries_before = fx.stream().await.len();

        let (seq, _) = fx
            .ls
            .update_layer(
                &fx.session_id,
                fx.layer,
                fx.oid,
                None,
                Some(relay.clone()),
                WRITER,
                1,
            )
            .await
            .unwrap_or_else(|e| {
                panic!("{backend:?}: a no-delta call with relay bytes must land; got {e:?}")
            });

        let entries = fx.stream().await;
        assert_eq!(entries.len(), entries_before + 1, "{backend:?}: one entry");
        let (id, fields) = entries.last().expect("the commit's entry");
        assert_eq!(id, &format!("{seq}-0"), "{backend:?}: the commit's entry");
        assert!(
            fields.contains_key("payload"),
            "{backend:?}: the entry carries its payload; got {fields:?}"
        );
        assert_eq!(
            fields.get("ompf"),
            Some(&relay),
            "{backend:?}: the entry's `ompf` field is exactly the relay bytes"
        );
    }
}

/// The defect the precondition closes (the shape the atomic-read proptest
/// found), through the public storage API on `lua`, on one replica's store.
///
/// A structural commit caches the object. A fast-path field commit with a
/// high `set_order` then raises the header's `next_set_order`; a structural
/// commit follows, planned on the cached entry repaired over the field
/// commit's stream entry.
///
/// - Sent without relay bytes, as a caller could before the precondition,
///   the field commit is refused, so nothing moves the cursor and the
///   structural commit leaves it where it was. Without the refusal the field
///   commit lands an entry with no `ompf` field, which the repair replays as
///   an empty commit: the cached entry keeps the old cursor, and the
///   structural commit writes it over the header's, moving `next_set_order`
///   backwards.
/// - Sent with its relay bytes, the field commit lands, the repair replays
///   its delta, and the structural commit keeps the raised cursor.
///
/// After every commit `next_set_order` is asserted never to have decreased.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_fast_commit_then_structural_commit_never_moves_next_set_order_backwards()
{
    const HIGH_SET_ORDER: u64 = 1_000;
    let fx = Fx::new("relay-next-set-order", UpdateLayerBackend::Lua).await;
    let mut cursor = fx.next_set_order().await;
    let mut check = |now: u64, step: &str| {
        assert!(
            now >= cursor,
            "{step}: obj:hdr next_set_order moved backwards, from {cursor} to {now}"
        );
        cursor = now;
    };
    let commit_structural = |section_id: u64| {
        let delta = structural(vec![create_cmd(
            ROOT,
            section_id,
            &format!("S{section_id}"),
        )]);
        let relay = Some(relay_ompf(&delta));
        (Some(delta), relay)
    };

    let (delta, relay) = commit_structural(NEW_SECTION);
    fx.ls
        .update_layer(&fx.session_id, fx.layer, fx.oid, delta, relay, WRITER, 1)
        .await
        .expect("the caching structural commit must land");
    check(
        fx.next_set_order().await,
        "after the caching structural commit",
    );

    for (round, with_relay) in [(0u64, false), (1, true)] {
        let field =
            delta_with_set_field_inline(WORLD, "cursor", vec![0x53], HIGH_SET_ORDER + round);
        let relay = with_relay.then(|| relay_ompf(&field));
        let landed = fx
            .ls
            .update_layer(
                &fx.session_id,
                fx.layer,
                fx.oid,
                Some(field),
                relay,
                WRITER,
                2,
            )
            .await;
        if with_relay {
            landed.as_ref().expect("the relayed field commit must land");
            assert!(
                fx.next_set_order().await > HIGH_SET_ORDER,
                "fixture: the field commit raised next_set_order past its set_order"
            );
        }
        check(
            fx.next_set_order().await,
            &format!("round {round}: after the field commit"),
        );

        let (delta, relay) = commit_structural(NEW_SECTION + 1 + round);
        fx.ls
            .update_layer(&fx.session_id, fx.layer, fx.oid, delta, relay, WRITER, 3)
            .await
            .expect("the structural commit must land");
        check(
            fx.next_set_order().await,
            &format!("round {round} (relay {with_relay}): after the structural commit"),
        );
        // Asserted only now, so that without the refusal the cursor check
        // above is what fails: it is the defect, the refusal its guard.
        if !with_relay {
            assert!(
                matches!(&landed, Err(ServiceError::Internal(m)) if m.starts_with(REFUSAL)),
                "a field commit without relay bytes is refused; got {landed:?}"
            );
        }
    }
}
