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

//! Fixtures for the fenced read's progress top-ups, shared by the unit tests
//! of both real [`FencedReadHook`]s — the structural cache's
//! (`storage::structural_cache_fenced`) and the catch-up flight's
//! (`storage::snapshot_flights`).
//!
//! - [`posed_rig_layer`]: a byte-dense layer shaped like posed rigs — few
//!   items, each section carrying one large inline array — above a lowered
//!   item ceiling, so the up-front estimate (bound × a fitted bytes per item)
//!   is far below what the read takes in.
//! - [`ObservedHook`]: wraps a real hook, forwarding every event, and records
//!   what the hook holds — its grant and its one top-up — after each progress
//!   report and when the fan-out ends, before the release: the moment the
//!   post-read adjust has not yet run, so only the progress top-ups can have
//!   covered the read's bytes.
//! - [`assert_accounted_while_reading`]: the checks both hooks' tests make on
//!   it.

use std::sync::{Arc, Mutex as StdMutex};

use super::TestValkey;
use crate::domain::types::{Field, FieldValue, ObjectState, Section};
use crate::error::ServiceError;
use crate::storage::client::ValkeyClient;
use crate::storage::diff_read::{
    read_object_snapshot, AtomicSnapshotOf, SnapshotConsumer, SnapshotRead, SnapshotReadConsumer,
    SnapshotTooLarge,
};
use crate::storage::external_values::ValkeyExternalValueStore;
use crate::storage::layer::LayerStore;
use crate::storage::memory_budget::{kib_for_bytes, MemoryBudget, SNAPSHOT_MEMORY_FACTOR};
use crate::storage::session::SessionStore;
use crate::storage::snapshot_fence::read::{
    read_object_snapshot_fenced, FenceAttempt, FenceGrant, FenceReleased, FenceRetry,
    FencedReadHook,
};
use crate::storage::snapshot_fence::LayerSize;

/// The rig's sections, each a child of the root carrying one inline array.
pub(crate) const RIG_SECTIONS: u64 = 32;
/// Bytes of each section's inline array: a posed joint array. With 14 bound
/// items per section, about 585 bytes per bound item, the order of the posed
/// rigs' 337 and far past the 18.9 at which the cold start's estimate stops
/// covering a read.
pub(crate) const RIG_VALUE_BYTES: usize = 8 * 1024;
/// The lowered atomic-read item ceiling: far below the rig's item bound, so
/// the layer is above the ceiling, its atomic read answers `TOO_LARGE`, and
/// the fenced read's windows (sized from this ceiling) are short.
pub(crate) const LOW_ITEMS: u64 = 64;
/// The lowered atomic-read byte ceiling: narrows the fenced fan-out's byte
/// window, so the read takes in its bytes over many windows, each reported.
pub(crate) const LOW_BYTES: u64 = 64 * 1024;

/// `tv`'s client with the atomic-read ceilings lowered to [`LOW_ITEMS`] /
/// [`LOW_BYTES`] and a catch-up memory budget of its own: `budget_kib` KiB,
/// or `tv`'s `T` when `None`. Its own budget, so everything it accounts is
/// the read's under test.
pub(crate) fn lowered_client(tv: &TestValkey, budget_kib: Option<u64>) -> ValkeyClient {
    let mut client = tv.client().clone();
    client.snapshot_atomic_read_max_items = LOW_ITEMS;
    client.snapshot_atomic_read_max_bytes = LOW_BYTES;
    let kib = budget_kib.unwrap_or(tv.client().catchup_build_budget.total_bytes() / 1024);
    client.catchup_build_budget = Arc::new(MemoryBudget::new(kib, Arc::new(StdMutex::new(None))));
    client
}

/// Create the posed-rig layer `layer` through `client` and return its object
/// id and the `TOO_LARGE` answer `client`'s atomic read gives it — what a
/// catch-up or a cold start sizes its fenced read from.
pub(crate) async fn posed_rig_layer(
    tv: &TestValkey,
    client: &ValkeyClient,
    layer: &str,
) -> (u64, SnapshotTooLarge) {
    let keys = tv.key_schema().clone();
    // Each store future boxed (the b.ae5 stack rule): this helper holds them
    // inside its own future, which the tests hold inside theirs.
    let session_id = Box::pin(SessionStore::new(client.clone(), keys.clone()).create(
        layer,
        &format!("/{layer}"),
        "rig-author",
    ))
    .await
    .expect("session creation must succeed");
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        next_set_order: 1,
        ..Default::default()
    };
    let joints: Vec<u64> = (1..=RIG_SECTIONS).collect();
    for (order, &joint) in joints.iter().enumerate() {
        state.sections.insert(
            joint,
            Section {
                section_id: joint,
                parent_id: 0,
                name: format!("Joint{joint}"),
                section_type: 1,
                section_order: u32::try_from(order).expect("few joints"),
                first_version: 1,
                last_version: 1,
            },
        );
        // A distinct pattern per joint, so no two values are equal.
        let value: Vec<u8> = (0..RIG_VALUE_BYTES)
            .map(|i| (i as u64 ^ joint) as u8)
            .collect();
        state.fields.insert(
            (joint, "xformOp:transform".to_owned()),
            Field {
                section_id: joint,
                key_name: "xformOp:transform".to_owned(),
                value: FieldValue::Inline(value),
                set_order: joint,
                authoring_set_order: joint,
                version: 1,
            },
        );
    }
    state.children.insert(0, joints);
    let ext = Arc::new(ValkeyExternalValueStore::new(client.clone(), keys.clone()));
    let oid = Box::pin(
        LayerStore::new(client.clone(), keys.clone(), 4096).create_layer(
            &session_id,
            layer,
            "rig-author",
            &state,
            ext,
        ),
    )
    .await
    .expect("create_layer must succeed");
    let read = Box::pin(read_object_snapshot(
        client,
        &keys,
        &oid.to_string(),
        SnapshotReadConsumer::Catchup,
    ))
    .await
    .expect("the atomic read must answer");
    let SnapshotRead::TooLarge(answer) = read else {
        panic!("fixture: the rig must be above the lowered atomic-read ceilings");
    };
    assert!(
        answer.item_bound > LOW_ITEMS,
        "fixture: the rig's item bound {} must be above the lowered ceiling {LOW_ITEMS}",
        answer.item_bound
    );
    (oid, answer)
}

/// What a hook held at one moment of its read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Held {
    /// The hook's grant and its one top-up together, in bytes.
    pub(crate) hook: u64,
    /// What the hook's budget accounts in all, `reserved + overcommit`: the
    /// read's, since nothing else holds that budget.
    pub(crate) budget: u64,
}

/// One progress report, as [`ObservedHook`] saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Report {
    /// The reply bytes the fan-out had read so far.
    pub(crate) bytes_read: u64,
    /// What was held once the hook had taken the report.
    pub(crate) held: Held,
}

/// A real hook, `inner`, with every event forwarded to it and what it holds
/// recorded (module docs).
pub(crate) struct ObservedHook<H> {
    pub(crate) inner: H,
    /// Reads the bytes `inner` holds: its grant and its one top-up.
    hook_bytes: fn(&H) -> u64,
    budget: Arc<MemoryBudget>,
    /// What was held once acquire granted the lease, before the fan-out's
    /// first command: the up-front grant alone.
    pub(crate) at_grant: Option<Held>,
    /// Every progress report, in order.
    pub(crate) reports: Vec<Report>,
    /// What was held once the fan-out ended, before the release and so
    /// before the post-read adjust.
    pub(crate) at_release: Option<Held>,
    /// The size the read measured, as the release reported it.
    pub(crate) measured: Option<LayerSize>,
    /// Attempts started.
    pub(crate) attempts: u32,
}

impl<H> ObservedHook<H> {
    /// Observe `inner`, whose grant and top-up `hook_bytes` reads, against
    /// `budget`, the budget its memory comes from.
    pub(crate) fn new(inner: H, hook_bytes: fn(&H) -> u64, budget: Arc<MemoryBudget>) -> Self {
        Self {
            inner,
            hook_bytes,
            budget,
            at_grant: None,
            reports: Vec::new(),
            at_release: None,
            measured: None,
            attempts: 0,
        }
    }

    /// What is held now.
    pub(crate) fn held(&self) -> Held {
        Held {
            hook: (self.hook_bytes)(&self.inner),
            budget: self.budget.reserved_bytes() + self.budget.overcommit_bytes(),
        }
    }
}

impl<H: FencedReadHook> FencedReadHook for ObservedHook<H> {
    async fn before_attempt(&mut self, attempt: FenceAttempt) -> Result<(), ServiceError> {
        self.attempts = attempt.attempt;
        self.inner.before_attempt(attempt).await
    }

    async fn on_retry_after(&mut self, retry: FenceRetry) -> Result<(), ServiceError> {
        self.inner.on_retry_after(retry).await
    }

    async fn on_grant(&mut self, grant: FenceGrant) {
        self.inner.on_grant(grant).await;
        self.at_grant = Some(self.held());
    }

    fn on_read_progress(&mut self, bytes_read: u64) {
        self.inner.on_read_progress(bytes_read);
        let held = self.held();
        self.reports.push(Report { bytes_read, held });
    }

    async fn before_release(&mut self) {
        self.inner.before_release().await;
        self.at_release = Some(self.held());
    }

    async fn after_release(&mut self, released: FenceReleased) {
        self.measured = released.measured;
        self.inner.after_release(released).await;
    }
}

/// Run `object_id`'s fenced read for `consumer` from `estimate` under
/// `hook`, which must verify it. Boxed (the b.ae5 stack rule): the fenced
/// read's future is large, and the tests hold it inside their own.
pub(crate) async fn read_fenced<C: SnapshotConsumer, H: FencedReadHook>(
    client: &ValkeyClient,
    tv: &TestValkey,
    object_id: u64,
    consumer: C,
    estimate: LayerSize,
    hook: &mut H,
) -> AtomicSnapshotOf<C::State> {
    Box::pin(read_object_snapshot_fenced(
        client,
        tv.key_schema(),
        object_id,
        consumer,
        estimate,
        hook,
    ))
    .await
    .expect("the fenced read of a quiet layer verifies")
}

/// `bytes` as a budget of `total` bytes holds it: whole KiB, clamped to the
/// total.
pub(crate) fn as_held(bytes: u64, total: u64) -> u64 {
    (kib_for_bytes(bytes) * 1024).min(total)
}

/// The checks both hooks' tests make on one verified read observed by
/// `observed`, from a budget of `total` bytes (R1, R3b):
///
/// - the read was one attempt, reported its progress over at least three
///   reports (round 1 and two windows), and its last report carries exactly
///   the bytes it measured;
/// - the up-front grant alone was below `SNAPSHOT_MEMORY_FACTOR` × the
///   measured bytes (clamped), so the check is not vacuous;
/// - when the fan-out ended, before the release and so before the post-read
///   adjust, the read held at least `SNAPSHOT_MEMORY_FACTOR` × the measured
///   bytes (clamped), and the budget accounted exactly what the hook held;
/// - after every report, the hook held exactly the larger of its up-front
///   grant and `SNAPSHOT_MEMORY_FACTOR` × the bytes read so far (clamped to
///   `total`): topped up as it read, never past `total`, never shrunk.
///
/// Returns what the hook held at the end of the read.
pub(crate) fn assert_accounted_while_reading<H>(observed: &ObservedHook<H>, total: u64) -> u64 {
    let measured = observed.measured.expect("the read measured the layer");
    let reports = &observed.reports;
    assert_eq!(observed.attempts, 1, "a quiet layer reads in one attempt");
    assert!(
        reports.len() >= 3,
        "the read reports after round 1 and after each window: {} reports",
        reports.len()
    );
    assert_eq!(
        reports.last().map(|r| r.bytes_read),
        Some(measured.bytes),
        "the last report carries the read's measured reply bytes"
    );
    let grant = observed.at_grant.expect("the read was granted").hook;
    let needed = as_held(measured.bytes * SNAPSHOT_MEMORY_FACTOR, total);
    assert!(
        grant < needed,
        "fixture: the up-front grant ({grant} B) must be below {SNAPSHOT_MEMORY_FACTOR} × the \
         measured {} B ({needed} B held), or the read never needed a top-up",
        measured.bytes
    );
    let end = observed.at_release.expect("the fan-out ended");
    assert!(
        end.hook >= needed,
        "at the end of the read, before the adjust, the read holds {} B: under \
         {SNAPSHOT_MEMORY_FACTOR} × its measured {} B ({needed} B), clamped to T = {total} B",
        end.hook,
        measured.bytes
    );
    assert_eq!(end.budget, end.hook, "all of it accounted in the budget");
    let mut previous = 0;
    for (i, report) in reports.iter().enumerate() {
        assert!(
            report.bytes_read >= previous,
            "report {i}: the bytes read are cumulative"
        );
        previous = report.bytes_read;
        let expected = grant.max(as_held(report.bytes_read * SNAPSHOT_MEMORY_FACTOR, total));
        assert_eq!(
            report.held.hook, expected,
            "report {i} ({} B read): the grant topped up to {SNAPSHOT_MEMORY_FACTOR} × the bytes \
             read so far, clamped to T = {total} B",
            report.bytes_read
        );
        assert!(report.held.hook <= total, "report {i}: never past T");
        assert_eq!(
            report.held.budget, report.held.hook,
            "report {i}: the budget accounts exactly the grant and its one top-up"
        );
    }
    end.hook
}
