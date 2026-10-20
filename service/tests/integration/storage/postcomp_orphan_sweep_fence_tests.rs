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

//! b.qpf post-completion (review finding 2) — the orphan sweep must FENCE the
//! object it is retiring before it reads the object's external-value references.
//!
//! # The defect
//!
//! `orphan_sweep::cleanup_orphaned_object` retires an object in three steps:
//! read the release set (`schedule_old_object_ext_gc`: `obj:ext` plus the
//! hashes the value rows name — since b.fsz MR-D4 by its paged reference walk,
//! before that by a snapshot read), enqueue it, then UNLINK the object's
//! keys. Pre-fix
//! nothing stopped an `UpdateLayer` from committing to that object BETWEEN the
//! read and the UNLINK — and an orphan is precisely the object where such a
//! straggler is likely, because the classic way to mint one is a `delete_layer`
//! that died after `DEL mapping`, leaving a writer that resolved the mapping a
//! moment earlier still able to commit.
//!
//! That straggler releases the hashes IT supersedes (`ExtRelease::commit`),
//! while the sweep's pre-commit read releases the same hashes again at
//! `event = ret`. Two decrements for ONE reference. With another live object
//! pointing at the hash, `ref_count` falls straight past that object's own
//! reference and the GC deletes `ext:meta` + `ext:data` underneath it — data
//! loss, not a leak. `the_fence_refuses_a_straggler_and_the_shared_blob_survives`
//! is that scenario end-to-end and is the test that could not pass pre-fix.
//!
//! # The fix, and why one flag is the whole fence
//!
//! `HSET obj:hdr superseded 1` on the data pool, immediately before the
//! retirement read. Both commit backends already refuse a superseded header —
//! the Lua script's §1b presence test (atomic `EVAL`, so a straggler is wholly
//! before or wholly refused) and the slow path's post-`HGETALL` guard, which is
//! additionally protected by the `WATCH` this same key already carries. The
//! other two retirement routes (`delete_layer`'s `XADD deleted`, the
//! CreateLayer-overwrite's own `HSET superseded`) were already fenced; the sweep
//! was the one exception.
//!
//! # What each test pins
//!
//! | test | window |
//! |---|---|
//! | `..._refuses_a_straggler_and_the_shared_blob_survives_{lua,txn}` | straggler starts AFTER the fence, driven from inside `enqueue_retirement_attempt` — the real post-fence rendezvous, plus the GC tick that turns the double-release into an observable blob deletion |
//! | `..._inside_the_watch_window_refuses_the_straggler_before_dispatch` | fence lands after the straggler's `WATCH` but before its header read |
//! | `..._after_the_watched_read_cancels_the_exec_and_the_retry_refuses` | fence lands after the straggler's header read — the `WATCH`-invalidation window, where the abort→retry loop is what converts it into a refusal |
//! | `..._skipping_the_ext_gc_schedule_still_completes_the_structural_unlink` | the skip-scheduling posture the fence's own failure mode inherits |
//! | `..._leaves_no_ghost_header_when_the_object_is_unlinked_under_it` | the fence's write must not outlive the UNLINK that follows it |
//!
//! The two window tests issue the bare `HSET` from a second connection rather
//! than racing a live sweep: it is byte-identical to the fence
//! `cleanup_orphaned_object` writes (one field, one value, on the data pool),
//! and the rendezvous tests above already pin that the sweep issues exactly that
//! write at exactly that point. Racing a real sweep into a two-command window on
//! another connection is not stageable; decomposing it is.
//!
//! Both window tests are txn-path only, and not because of a seam: the Lua fast
//! path opens no `WATCH` and takes no second command, so neither window exists
//! there. The rendezvous test covers Lua.
//!
//! # Feature gating
//!
//! The two window tests need `test-fault-injection` for the `LayerStore`
//! commit-window hooks and live in an inner `#[cfg]` module. Without the feature
//! they are compiled out rather than red — the silent activation condition
//! `docs-internal/test-writing-guide.md` warns about — so the module is named
//! and documented for it. Everything else here runs unfeatured.
//!
//! # Wait budgets
//!
//! `.config/nextest.toml` marks a test slow at 40 s and hard-terminates it at
//! 120 s, and a SIGKILLed test prints nothing — so every diagnostic panic
//! message below would be lost exactly when it is needed. Every wait here is a
//! named module constant held over a whole operation rather than a per-step
//! budget that multiplies, and the per-test worst case is the sum of its waits
//! at full budget:
//!
//! | Test | Waits | Worst case |
//! |---|---|---|
//! | `..._refuses_a_straggler_and_the_shared_blob_survives_{lua,txn}` | 1 × [`SWEEP_BUDGET`] + 1 × [`SWEEP_JOIN_BUDGET`] | **15 s** |
//! | `..._inside_the_watch_window_refuses_the_straggler_before_dispatch` | 1 × `STRAGGLER_BUDGET` | **20 s** |
//! | `..._after_the_watched_read_cancels_the_exec_and_the_retry_refuses` | 1 × `STRAGGLER_BUDGET` | **20 s** |
//! | `..._skipping_the_ext_gc_schedule_still_completes_the_structural_unlink` | 1 × [`SWEEP_BUDGET`] + 1 × [`SWEEP_JOIN_BUDGET`] | **15 s** |
//! | `..._leaves_no_ghost_header_when_the_object_is_unlinked_under_it` | 1 × [`SWEEP_BUDGET`] + 1 × [`SWEEP_JOIN_BUDGET`] | **15 s** |
//!
//! The two window tests run no sweep, so they pay no sweep budget; the
//! rendezvous drives their counterpart's straggler from inside the sweep and is
//! therefore already covered by [`SWEEP_BUDGET`]. [`SWEEP_POLL`] is the poll
//! interval inside [`SWEEP_BUDGET`], not an additional wait. `STRAGGLER_BUDGET`
//! is written unlinked because it is compiled only under `test-fault-injection`,
//! alongside the two rows that spend it. Every row is under the 40 s slow mark,
//! and so is the 35 s a test paying all three would cost; the healthy path
//! finishes in tens of milliseconds and pays none of it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::interfaces::{
    HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
};
use futures::FutureExt;
use tokio_util::sync::CancellationToken;

use live_edit::config::UpdateLayerBackend;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::{
    process_pending_gc, ExtRelease, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{delta_with_set_field_external, delta_with_set_field_inline};
use crate::storage::b_qpf_tests::{
    assert_blob_alive, create_session, empty_state, ext_hash, ext_hash_hex, ext_hash_vec,
    ext_store, layer_store, obj_ext_sorted, pending_members, pre_store, ref_count,
    retirement_member, EXT_SIZE, FIELD_A, SECTION_ID,
};

// ═════════════════════════════════════════════════════════════════════════════
// Contract constants
// ═════════════════════════════════════════════════════════════════════════════

/// The one external hash both fixture objects reference.
///
/// One hash held by TWO objects is the whole point: a double release on a hash
/// only one object holds is a `ref_count` that reaches zero one decrement early,
/// which looks exactly like a correct reclamation. It is the surviving second
/// referrer that turns the second decrement into observable data loss.
const SHARED_HASH: u8 = 0xD1;

/// Enough connections for the straggler's exclusive txn slot, the sweep's
/// maintenance slot and the multiplexed data traffic to coexist. The rendezvous
/// tests drive a full `update_layer` from INSIDE the sweep's `enqueue_retirement_attempt`,
/// so the two are genuinely concurrent rather than interleaved by the runtime.
const POOL_SIZE: u32 = 8;

/// Bound on "the sweep reached its UNLINK". A local sweep over a handful of
/// keys is tens of milliseconds; this is a deadline, not a sleep.
const SWEEP_BUDGET: Duration = Duration::from_secs(10);

/// Poll interval while waiting on [`SWEEP_BUDGET`].
const SWEEP_POLL: Duration = Duration::from_millis(100);

/// Bound on the sweep task exiting after its cancellation token fires.
const SWEEP_JOIN_BUDGET: Duration = Duration::from_secs(5);

/// One deadline over a whole straggler `update_layer` — every `WATCH`, header
/// read, hook body, aborted `EXEC` and retry in it.
///
/// The two window tests park on that single call while a second connection
/// writes the fence from inside a commit-window hook, and the refusal they
/// assert is the ONLY thing that ends the wait. A fence that stops refusing —
/// or a retry loop that never re-reads the flag — turns the await into a hang,
/// and an unbudgeted hang is collected by nextest's 120 s SIGKILL, which prints
/// nothing: every diagnostic panic message below would be lost at exactly the
/// moment a real failure needs reading.
///
/// One deadline over the whole call rather than a per-attempt one, per the
/// whole-drain rule — the txn path is a retry loop, so a per-attempt bound would
/// multiply by an attempt count the test does not control. 20 s is unreachable
/// on the healthy path (a handful of local round trips, tens of milliseconds)
/// and keeps every per-test worst case in this module under the 40 s slow mark:
/// 20 s alone here, and 20 s + [`SWEEP_BUDGET`] + [`SWEEP_JOIN_BUDGET`] = 35 s
/// even for a future test that paid all three.
///
/// Gated with the only module that uses it: the window tests are compiled out
/// without `test-fault-injection`, and an ungated const would then be dead code
/// under the release-shape clippy run.
#[cfg(feature = "test-fault-injection")]
const STRAGGLER_BUDGET: Duration = Duration::from_secs(20);

/// The FULL refusal message both backends must produce for a fenced header,
/// restated verbatim from `storage::layer::SUPERSEDED_SENTINEL`.
///
/// Restated rather than imported: the sentinel is a pinned backend-parity
/// string, and a test that read it from the same `const` the emitter reads could
/// not fail when the two backends drifted apart — both sides of the comparison
/// would move together.
///
/// Asserted by EQUALITY, not by substring, and the full text is restated for
/// exactly that reason. A substring check on `"superseded"` cannot detect the
/// drift this const exists to catch: it is one-sided drift that matters — one
/// backend rewording its message while the other keeps the pinned wording — and
/// every plausible rewording still contains the word `superseded`, so a
/// substring match stays green through it and the parity contract (SDD §7.5)
/// silently stops being asserted. Equality against the restated text fails on
/// the first byte either producer changes.
const SUPERSEDED_MARKER: &str = "object superseded by a concurrent CreateLayer overwrite";

// ═════════════════════════════════════════════════════════════════════════════
// The rendezvous external-value store
// ═════════════════════════════════════════════════════════════════════════════

/// A body run once, from inside `enqueue_retirement_attempt`.
type Rendezvous = Arc<dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync>;

/// Where a rendezvous-driven `update_layer` parks its result for the test body
/// to inspect after the sweep has been joined. The straggler runs inside the
/// sweep task, so its `Result` cannot simply be returned.
type StragglerOutcome = Arc<Mutex<Option<Result<(u64, u64), ServiceError>>>>;

/// An [`ExternalValueStore`] that records every retirement batch it is handed
/// and can run a caller-supplied body at the exact moment the sweep publishes
/// one.
///
/// This is the seam-free way to stand in the post-fence window. The sweep calls
/// `enqueue_retirement_attempt` AFTER writing the fence and AFTER building the release set,
/// and BEFORE the structural UNLINK — which is precisely the window a straggler
/// commit had to land in to double-release. Injecting the store rather than
/// adding a production hook keeps the window defined by production's own call
/// order (b.kf1's `spawn_orphan_sweep` injection point), so a future edit that
/// moves the fence relative to the read moves this window with it.
///
/// Everything other than `enqueue_retirement_attempt` delegates to the real Valkey store,
/// so `ref_count`, `ext:pending` and the GC tick the tests assert on are the
/// production keys rather than a mock's in-memory maps.
struct RendezvousExtStore {
    inner: Arc<ValkeyExternalValueStore>,
    /// One entry per `enqueue_retirement_attempt` call, holding that call's batch.
    batches: Mutex<Vec<Vec<ExtRelease>>>,
    /// Run on the FIRST `enqueue_retirement_attempt` only. `None` makes this a pure
    /// recording wrapper.
    rendezvous: Option<Rendezvous>,
    fired: AtomicUsize,
}

impl RendezvousExtStore {
    fn new(inner: Arc<ValkeyExternalValueStore>, rendezvous: Option<Rendezvous>) -> Self {
        Self {
            inner,
            batches: Mutex::new(Vec::new()),
            rendezvous,
            fired: AtomicUsize::new(0),
        }
    }

    /// Every batch handed to `enqueue_retirement_attempt`, in call order.
    fn batches(&self) -> Vec<Vec<ExtRelease>> {
        self.batches.lock().expect("never poisoned").clone()
    }
}

#[async_trait::async_trait]
impl ExternalValueStore for RendezvousExtStore {
    async fn enqueue_retirement_attempt(
        &self,
        object_id: &str,
        attempt: &str,
        releases: &[ExtRelease],
        send_by: std::time::Instant,
    ) -> Result<live_edit::storage::external_values::ReleaseAttemptOutcome, ServiceError> {
        self.batches
            .lock()
            .expect("never poisoned")
            .push(releases.to_vec());
        if self.fired.fetch_add(1, Ordering::SeqCst) == 0 {
            if let Some(body) = self.rendezvous.as_ref() {
                body().await;
            }
        }
        self.inner
            .enqueue_retirement_attempt(object_id, attempt, releases, send_by)
            .await
    }

    async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        self.inner.store_values(values).await
    }

    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.inner.store_verified_values(values).await
    }

    fn ext_value_verify(&self) -> live_edit::storage::external_values::ExtValueVerify<'_> {
        self.inner.ext_value_verify()
    }

    async fn load_values(
        &self,
        hashes: &[&[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
        self.inner.load_values(hashes).await
    }

    async fn read_batch(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.inner.read_batch(hashes).await
    }

    async fn probe_values(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<live_edit::storage::external_values::ExtValueProbe, ServiceError> {
        self.inner.probe_values(hashes).await
    }

    async fn process_pending(
        &self,
        policy: &live_edit::storage::external_values::GcPolicy,
        state: &mut live_edit::storage::external_values::GcPassState,
    ) -> Result<live_edit::storage::external_values::GcBatchResult, ServiceError> {
        self.inner.process_pending(policy, state).await
    }

    fn ext_release_gate(
        &self,
    ) -> Option<&std::sync::Arc<live_edit::storage::ext_release_gate::ExtReleaseGate>> {
        self.inner.ext_release_gate()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture helpers
// ═════════════════════════════════════════════════════════════════════════════

/// Two live objects sharing one external hash, one of them orphaned.
struct SharedHashFixture {
    session_id: String,
    /// The orphan: header intact, layer mapping deleted. This is what the sweep
    /// retires.
    doomed: u64,
    doomed_layer: String,
    /// The live object that also references the shared hash — the victim an
    /// over-release would take down.
    survivor: u64,
}

/// Build [`SharedHashFixture`]: create two layers, give each a value row holding
/// the SAME external hash, then orphan one by deleting its layer mapping.
///
/// Deleting the mapping while keeping the header is not a synthetic shape — it
/// is exactly what a `delete_layer` that died after its step 2 leaves behind,
/// and it is the state `sweep_orphaned_objects` classifies as an orphan (header
/// names a `(session, layer)` whose mapping no longer resolves to it).
async fn shared_hash_fixture(
    tv: &TestValkey,
    ext: &Arc<ValkeyExternalValueStore>,
    metrics: &Arc<Metrics>,
    tag: &str,
) -> SharedHashFixture {
    let session_id = create_session(tv, &format!("postcomp-fence-{tag}")).await;
    let ls = layer_store(tv, UpdateLayerBackend::Lua, metrics);
    let h = ext_hash(SHARED_HASH);

    let mut oids = Vec::new();
    for role in ["doomed", "survivor"] {
        let layer = format!("postcomp-{tag}-{role}");
        let oid = ls
            .create_layer(
                &session_id,
                &layer,
                "postcomp-user",
                &empty_state(),
                ext.clone(),
            )
            .await
            .expect("create_layer must succeed");
        pre_store(ext, &[h]).await;
        ls.update_layer_relayed(
            &session_id,
            &layer,
            oid,
            Some(delta_with_set_field_external(
                SECTION_ID,
                FIELD_A,
                ext_hash_vec(SHARED_HASH),
                EXT_SIZE,
                1,
                0,
            )),
            "postcomp-writer",
            1,
        )
        .await
        .expect("the external commit must land");
        oids.push((layer, oid));
    }
    let (doomed_layer, doomed) = oids[0].clone();
    let (_, survivor) = oids[1].clone();

    assert_eq!(
        ref_count(tv, h).await,
        Some(2),
        "fixture precondition: two live objects must reference the shared hash, or a double \
         release would look like an ordinary reclamation instead of taking a blob out from under \
         a survivor"
    );
    assert!(
        pending_members(tv).await.is_empty(),
        "fixture precondition: neither commit superseded anything, so nothing may be owed yet — \
         a member here would make every `exactly one member` assertion below ambiguous"
    );

    // Orphan the doomed object.
    let mapping_key = tv.key_schema().layer_mapping(&session_id, &doomed_layer);
    let removed: i64 = tv
        .client()
        .pool
        .del(&mapping_key)
        .await
        .expect("DEL layer mapping must succeed");
    assert_eq!(
        removed, 1,
        "fixture precondition: the mapping must have existed and been removed — with it still \
         present the sweep would not classify the object as an orphan and would clean nothing"
    );

    SharedHashFixture {
        session_id,
        doomed,
        doomed_layer,
        survivor,
    }
}

/// The straggler commit: rewrite the doomed object's only external row to an
/// INLINE value.
///
/// The shape matters. This delta supersedes the shared hash, so the commit owes
/// a `rewrite` release for it — the second decrement. A delta that merely
/// re-wrote the same hash would owe a `surplus_acquire` against its own
/// pre-flight `+1` and net out, and the over-release would be invisible.
fn straggler_delta() -> oodf_ffi::OwnedDelta {
    delta_with_set_field_inline(SECTION_ID, FIELD_A, vec![0x77, 0x88], 1)
}

/// `EXISTS le:{object_id}:obj:hdr`.
async fn header_exists(tv: &TestValkey, object_id: u64) -> bool {
    tv.client()
        .pool
        .exists(tv.key_schema().obj_header(&object_id.to_string()))
        .await
        .expect("EXISTS obj:hdr must succeed")
}

/// Run one sweep iteration: spawn, wait for `key` to disappear (the sweep's
/// last act on an orphan is the header-first UNLINK, so the header vanishing is
/// the observable "this object has been fully processed"), then cancel and join.
///
/// One deadline for the wait and one for the join, per the test-writing guide's
/// whole-drain rule — neither multiplies by anything.
async fn sweep_until_key_gone(
    tv: &TestValkey,
    store: Arc<dyn ExternalValueStore>,
    metrics: &Arc<Metrics>,
    key: &str,
) {
    // The fixture's objects were created a moment ago: without a zero grace
    // the sweep would leave the orphan to a CreateLayer that might still be
    // committing it.
    tv.client().set_orphan_sweep_grace_for_test(Duration::ZERO);
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        tv.key_schema().clone(),
        store,
        Arc::new(live_edit::storage::layer::LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            0,
        )),
        crate::common::test_leader("postcomp-fence-replica"),
        cancel.clone(),
        Some(metrics.clone()),
    );

    let deadline = tokio::time::Instant::now() + SWEEP_BUDGET;
    loop {
        let exists: bool = tv
            .client()
            .pool
            .exists(key)
            .await
            .expect("EXISTS must succeed");
        if !exists || tokio::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(SWEEP_POLL).await;
    }

    cancel.cancel();
    tokio::time::timeout(SWEEP_JOIN_BUDGET, handle)
        .await
        .expect("the orphan sweep must exit within the join budget of cancellation")
        .expect("the orphan sweep must not panic");
}

/// Assert a straggler's `update_layer` was refused as superseded.
fn assert_refused_as_superseded(outcome: &Result<(u64, u64), ServiceError>, ctx: &str) {
    match outcome {
        Err(ServiceError::FailedPrecondition(msg)) => assert_eq!(
            msg, SUPERSEDED_MARKER,
            "{ctx}: the refusal must carry the superseded sentinel VERBATIM — that is the verdict \
             both backends map a fenced header to, and the one whose remedy (re-resolve the \
             mapping, which for a swept orphan answers NOT_FOUND) is correct here. Equality, not \
             containment: a one-sided reword would keep the word `superseded` and slip past a \
             substring check while the backends had already drifted apart; got {msg:?}"
        ),
        Ok((seq, _)) => panic!(
            "{ctx}: the straggler COMMITTED at sequence {seq}. This is the b.qpf post-completion \
             defect itself: its own `rewrite` release and the sweep's retirement release are two \
             decrements for one reference, and the shared blob is deleted under the object that \
             still points at it"
        ),
        Err(other) => panic!(
            "{ctx}: the straggler must be refused as superseded, not by some other guard — a \
             NotFound or an Aborted here means the fence is not what stopped it and the test \
             proves nothing about the fence; got {other:?}"
        ),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. The post-fence rendezvous + the end-to-end value canary
// ═════════════════════════════════════════════════════════════════════════════

/// A straggler that starts inside the sweep's retirement window is refused, and
/// the hash the doomed object shares with a live object survives the GC tick.
///
/// This is the headline regression. The rendezvous fires from inside
/// `enqueue_retirement_attempt`, which production reaches only after the fence has been
/// written and the release set built, and before the structural UNLINK — the
/// exact window the review found unguarded. Pre-fix the straggler commits, and
/// the assertions fail in three independent places: the commit succeeds, a
/// second member appears on `ext:pending`, and the GC tick deletes a blob a live
/// object still references.
///
/// The blob assertion is the one that makes this a data-LOSS test rather than a
/// bookkeeping one, and it is why the fixture carries a survivor: with one
/// referrer, a double decrement merely reclaims the blob one step early and
/// every keyspace assertion still passes.
///
/// Run on both backends. The fence is one header flag and the argument that it
/// covers both is a claim about two different mechanisms — the Lua script's
/// pre-mutation presence test and the slow path's post-`HGETALL` guard — so a
/// single-backend test would leave half the argument unexercised.
async fn the_fence_refuses_a_straggler_and_the_shared_blob_survives(backend: UpdateLayerBackend) {
    let tag = match backend {
        UpdateLayerBackend::Lua => "lua",
        UpdateLayerBackend::LuaWatchStructural => "lua-watch-structural",
        UpdateLayerBackend::Txn => "txn",
    };
    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-postcomp-fence"));
    let ext = ext_store(&tv);
    let fx = shared_hash_fixture(&tv, &ext, &metrics, tag).await;
    let h = ext_hash(SHARED_HASH);

    let straggler = Arc::new(layer_store(&tv, backend, &metrics));
    let outcome: StragglerOutcome = Arc::new(Mutex::new(None));

    let rendezvous: Rendezvous = {
        let straggler = straggler.clone();
        let outcome = outcome.clone();
        let session_id = fx.session_id.clone();
        let layer = fx.doomed_layer.clone();
        let doomed = fx.doomed;
        Arc::new(move || {
            let straggler = straggler.clone();
            let outcome = outcome.clone();
            let session_id = session_id.clone();
            let layer = layer.clone();
            async move {
                let result = straggler
                    .update_layer_relayed(
                        &session_id,
                        &layer,
                        doomed,
                        Some(straggler_delta()),
                        "postcomp-straggler",
                        2,
                    )
                    .await;
                *outcome.lock().expect("never poisoned") = Some(result);
            }
            .boxed()
        })
    };

    let store = Arc::new(RendezvousExtStore::new(ext.clone(), Some(rendezvous)));
    let doomed_header = tv.key_schema().obj_header(&fx.doomed.to_string());
    sweep_until_key_gone(
        &tv,
        store.clone() as Arc<dyn ExternalValueStore>,
        &metrics,
        &doomed_header,
    )
    .await;

    // ── The rendezvous ran, in the window it was supposed to ────────────────
    let batches = store.batches();
    assert_eq!(
        batches,
        vec![vec![ExtRelease::retirement(h, fx.doomed)]],
        "the sweep must have published exactly one retirement batch, owing exactly the shared \
         hash on behalf of the doomed object. An empty list means the rendezvous never fired and \
         every assertion below is vacuous; a different member means the release set is not what \
         the fence is protecting"
    );
    let outcome = outcome
        .lock()
        .expect("never poisoned")
        .take()
        .expect("the rendezvous must have driven the straggler's update_layer to completion");
    assert_refused_as_superseded(&outcome, tag);

    // ── One owed decrement, not two ─────────────────────────────────────────
    assert_eq!(
        pending_members(&tv).await,
        vec![retirement_member(h, fx.doomed)],
        "exactly ONE member may be owed for the shared hash: the retirement. A second member — \
         the straggler's own `rewrite` release — is the over-release, and because pending members \
         carry the event that owes them (b.qpf), the two do NOT collapse into one SADD member: \
         the GC would decrement twice"
    );
    assert!(
        !header_exists(&tv, fx.doomed).await,
        "the orphan's header must be gone. The fence writes to that key, so a surviving `obj:hdr` \
         would mean the fence outlived the UNLINK and left a single-field ghost the next sweep \
         re-discovers forever"
    );

    // ── The GC tick: the decrement lands once and the blob survives ─────────
    assert_eq!(
        ref_count(&tv, h).await,
        Some(2),
        "a release is an ENQUEUE — nothing may have decremented before the GC runs"
    );
    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(
        stats.processed, 1,
        "one owed decrement, one claim. Two would mean the straggler enqueued one too"
    );
    assert_eq!(
        stats.deleted, 0,
        "`ref_count` must land on 1, not 0 — the survivor still holds the hash"
    );
    assert_eq!(
        ref_count(&tv, h).await,
        Some(1),
        "exactly one reference must have been given up: the doomed object's"
    );
    assert_blob_alive(&tv, h, "the hash shared with the surviving object").await;

    // ── The survivor is untouched ───────────────────────────────────────────
    assert_eq!(
        obj_ext_sorted(&tv, fx.survivor).await,
        vec![(ext_hash_hex(SHARED_HASH), 1)],
        "the surviving object's own reference index must be unchanged — it neither committed nor \
         retired, and its `obj:ext` is what a future retirement of ITS own will read"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn orphan_sweep_fence_lua_refuses_a_straggler_and_the_shared_blob_survives() {
    the_fence_refuses_a_straggler_and_the_shared_blob_survives(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn orphan_sweep_fence_txn_refuses_a_straggler_and_the_shared_blob_survives() {
    the_fence_refuses_a_straggler_and_the_shared_blob_survives(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. The two WATCH-relative windows (txn path only)
// ═════════════════════════════════════════════════════════════════════════════

/// Gated on `test-fault-injection` for the `LayerStore` commit-window hooks.
///
/// **The gate is silent.** Without the feature these two tests are not skipped
/// with a notice, they are compiled out, and the run reports green having
/// covered neither window. Any hand-rolled local invocation must pass
/// `--features test-support,test-fault-injection`; CI's `test:` job and the
/// canonical **Full test** command both do.
#[cfg(feature = "test-fault-injection")]
mod watch_windows {
    use super::*;

    use fred::clients::Client;
    use live_edit::storage::watched_txn::WatchWindowHook;

    use crate::storage::b_qpf_tests::{header_seq, watch_voided};

    /// A hook that writes the fence once and counts how many times the commit
    /// path reached it.
    ///
    /// The count is an assertion, not diagnostics: it is what distinguishes
    /// "the first attempt dispatched and was aborted, and a SECOND attempt was
    /// refused at its header read" from "the first attempt never dispatched at
    /// all". Writing the flag only on the first invocation is load-bearing for
    /// the same reason `b_bhb_tests::once_only` exists — a hook that re-staged
    /// its disturbance on every retry would keep re-arming a window the retry is
    /// supposed to escape.
    fn fencing_hook(
        pool: fred::clients::Pool,
        header_key: String,
    ) -> (WatchWindowHook, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let hook: WatchWindowHook = Arc::new(move |_client: &Client| {
            let pool = pool.clone();
            let header_key = header_key.clone();
            let counter = counter.clone();
            async move {
                if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                    // Byte-identical to the fence `cleanup_orphaned_object`
                    // writes: one field, one value, on the data pool, from a
                    // connection that is not the straggler's.
                    let _: () = pool
                        .hset(&header_key, [("superseded", "1")])
                        .await
                        .expect("the fence write must succeed");
                }
            }
            .boxed()
        });
        (hook, calls)
    }

    /// The fence landing between the straggler's `WATCH` and its header read is
    /// caught by the read itself — no commit, nothing owed.
    ///
    /// The earlier of the two slow-path windows. The straggler is already
    /// holding a `WATCH` on the header when the sweep fences it, so it is past
    /// the point where an ordinary "resolve the mapping first" check could have
    /// saved it; what refuses it is the `superseded` guard on the very next
    /// command.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_fence_inside_the_watch_window_refuses_the_straggler_before_dispatch() {
        let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test-postcomp-fence"));
        let ext = ext_store(&tv);
        let fx = shared_hash_fixture(&tv, &ext, &metrics, "watchwindow").await;
        let h = ext_hash(SHARED_HASH);

        let header_key = tv.key_schema().obj_header(&fx.doomed.to_string());
        let seq_before = header_seq(&tv, fx.doomed)
            .await
            .expect("the doomed header must carry a seq");
        let unwatched_before = watch_voided(&metrics, "update_layer", "committed_unwatched");
        let (hook, calls) = fencing_hook(tv.client().pool.clone(), header_key.clone());

        let straggler = layer_store(&tv, UpdateLayerBackend::Txn, &metrics)
            .with_watch_window_hook_for_test(hook);
        let outcome = tokio::time::timeout(
            STRAGGLER_BUDGET,
            straggler.update_layer_relayed(
                &fx.session_id,
                &fx.doomed_layer,
                fx.doomed,
                Some(straggler_delta()),
                "postcomp-straggler",
                2,
            ),
        )
        .await
        .expect(
            "the straggler's update_layer must settle within the straggler budget in the \
             WATCH-window arm (fence written between the WATCH and the header read): the guard \
             refuses before any dispatch, so a call still running here is a hang, not slow work",
        );

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the WATCH window must have been entered exactly once. Zero means the fence was never \
             written and the refusal below came from somewhere else; more than one means the \
             attempt retried, which this window does not produce — the guard fires before any \
             dispatch"
        );
        assert_refused_as_superseded(&outcome, "watch-window fence");
        assert_eq!(
            header_seq(&tv, fx.doomed).await,
            Some(seq_before),
            "nothing may have committed — the guard runs before the MULTI, so the header must not \
             advance"
        );
        assert_eq!(
            watch_voided(&metrics, "update_layer", "committed_unwatched"),
            unwatched_before,
            "the commit must have been refused, not settled as a landed-but-unwatched one. A tick \
             here would mean an EXEC ran, which changes what the refusal means: the keys would be \
             written and a release owed"
        );
        assert!(
            pending_members(&tv).await.is_empty(),
            "a refused commit supersedes nothing, so it may owe nothing. A member here is the \
             second decrement the fence exists to prevent"
        );
        assert_eq!(
            ref_count(&tv, h).await,
            Some(2),
            "both objects still hold the shared hash"
        );
    }

    /// The fence landing AFTER the straggler's header read cancels its `EXEC`,
    /// and the retry is what turns the cancellation into a refusal.
    ///
    /// The later window, and the one the fence's own argument rests on: by the
    /// time the flag is written the straggler has already read a clean header,
    /// so nothing it has seen says `superseded`. What stops it is that the same
    /// key is under its `WATCH` — the `EXEC` returns nil, the existing `Aborted`
    /// path retries, and the retry's `HGETALL` sees the flag.
    ///
    /// The hook-invocation count is the proof that both halves happened: exactly
    /// one dispatch (so an `EXEC` really was attempted and cancelled) followed by
    /// a refusal (so the retry did not reach dispatch again).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_fence_after_the_watched_read_cancels_the_exec_and_the_retry_refuses() {
        let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test-postcomp-fence"));
        let ext = ext_store(&tv);
        let fx = shared_hash_fixture(&tv, &ext, &metrics, "execwindow").await;
        let h = ext_hash(SHARED_HASH);

        let header_key = tv.key_schema().obj_header(&fx.doomed.to_string());
        let seq_before = header_seq(&tv, fx.doomed)
            .await
            .expect("the doomed header must carry a seq");
        let ext_before = obj_ext_sorted(&tv, fx.doomed).await;
        let unwatched_before = watch_voided(&metrics, "update_layer", "committed_unwatched");
        let (hook, calls) = fencing_hook(tv.client().pool.clone(), header_key.clone());

        let straggler = layer_store(&tv, UpdateLayerBackend::Txn, &metrics)
            .with_pre_dispatch_hook_for_test(hook);
        let outcome = tokio::time::timeout(
            STRAGGLER_BUDGET,
            straggler.update_layer_relayed(
                &fx.session_id,
                &fx.doomed_layer,
                fx.doomed,
                Some(straggler_delta()),
                "postcomp-straggler",
                2,
            ),
        )
        .await
        .expect(
            "the straggler's update_layer must settle within the straggler budget in the \
             post-read / cancelled-EXEC arm (fence written after the header read): the abort \
             feeds a retry loop, so a call still running here is that loop failing to terminate",
        );

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "exactly ONE attempt may have reached dispatch. Zero would mean the commit never got \
             past its header read, which is the OTHER window and not this one; two would mean the \
             retry dispatched again — i.e. it did not see the flag the first attempt's abort was \
             caused by, which is the failure this test exists to catch"
        );
        assert_refused_as_superseded(&outcome, "post-read fence");
        assert_eq!(
            header_seq(&tv, fx.doomed).await,
            Some(seq_before),
            "the cancelled EXEC must have written NOTHING — a WATCH that fires cancels the whole \
             transaction, so an advanced seq would mean the commit landed anyway and the release \
             it owes is unenqueued and unowned"
        );
        assert_eq!(
            obj_ext_sorted(&tv, fx.doomed).await,
            ext_before,
            "the per-object reference index must be untouched, for the same reason: it is written \
             inside the cancelled MULTI, and the retirement read that follows the fence reads it"
        );
        assert_eq!(
            watch_voided(&metrics, "update_layer", "committed_unwatched"),
            unwatched_before,
            "the EXEC must have been CANCELLED, not have landed on a connection that lost its \
             WATCH. That second shape also ends in FailedPrecondition, but with the write already \
             in the keyspace — so without this the two are indistinguishable from the outcome"
        );
        assert!(
            pending_members(&tv).await.is_empty(),
            "nothing committed, so nothing may be owed"
        );
        assert_eq!(
            ref_count(&tv, h).await,
            Some(2),
            "both objects still hold the shared hash"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. The skip-scheduling posture, and the ghost header
// ═════════════════════════════════════════════════════════════════════════════

/// An orphan whose ext-GC scheduling is SKIPPED still gets its structural keys
/// UNLINKed.
///
/// The fence's own failure mode (failure mode 4: the `HSET` fails, so the sweep
/// skips the retirement read rather than performing it unfenced) inherits this
/// posture wholesale from the two failure modes that predate it — log, skip the
/// scheduling, proceed to the UNLINK, because permanent orphan retention is a
/// worse outcome than a `ref_count` leak. Mode 4 itself is not reachable from a
/// test today (b.1ah owns the missing fence-write fault seam; making an `HSET`
/// fail while the sweep's own `HGETALL` on the same key succeeds needs an
/// injection seam that does not exist), so this pins the reachable twin: the
/// non-numeric-`object_id` mode, which takes the same skip-then-UNLINK branch.
///
/// The stub records every `enqueue_retirement_attempt`, so "skipped" is asserted as zero
/// invocations rather than inferred from an unchanged pending set — an enqueue
/// that ran and failed silently would leave the pending set unchanged too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_sweep_skipping_the_ext_gc_schedule_still_completes_the_structural_unlink() {
    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-postcomp-fence"));
    let ext = ext_store(&tv);
    let ks = tv.key_schema();
    let pool = tv.client().pool.clone();

    // A non-numeric object_id: `cleanup_orphaned_object` cannot parse it, so it
    // skips the ext-GC scheduling entirely and falls through to the UNLINK.
    let oid = "postcomp-not-a-number";
    let h = ext_hash(0xD2);
    let h_hex = sha256_to_hex(&h);
    ext.store_values(&[(&h, b"postcomp-skip-blob".as_slice())])
        .await
        .expect("the pre-flight store must succeed");

    let header_key = ks.obj_header(oid);
    pool.hset::<(), _, _>(
        &header_key,
        [
            ("seq", "1"),
            ("first_ver", "1"),
            ("src_fmt", "usd"),
            ("session_id", "postcomp-dead-session"),
            ("layer_name", "postcomp-ghost-layer"),
        ],
    )
    .await
    .expect("HSET header");
    pool.sadd::<(), _, _>(ks.obj_sections_index(oid), "1")
        .await
        .expect("SADD sections index");
    pool.hset::<(), _, _>(
        ks.obj_section(oid, "1"),
        [
            ("parent_id", "0"),
            ("name", "root"),
            ("type", "1"),
            ("section_order", "0"),
            ("first_version", "1"),
            ("last_version", "1"),
        ],
    )
    .await
    .expect("HSET section");
    pool.rpush::<(), _, _>(ks.obj_children(oid, "0"), "1")
        .await
        .expect("RPUSH children");
    let key_name = "tex";
    let field_member = live_edit::storage::keys::KeySchema::field_member("1", key_name);
    let field_key = ks.obj_field(oid, "1", key_name);
    pool.zadd::<(), _, _>(
        ks.obj_fields_index(oid),
        None,
        None,
        false,
        false,
        (1.0, field_member.as_str()),
    )
    .await
    .expect("ZADD fields index");
    pool.hset::<(), _, _>(
        &field_key,
        [
            ("hash", h_hex.as_str()),
            ("size", "1024"),
            ("set_order", "0"),
            ("version", "1"),
        ],
    )
    .await
    .expect("HSET field");

    let store = Arc::new(RendezvousExtStore::new(ext.clone(), None));
    sweep_until_key_gone(
        &tv,
        store.clone() as Arc<dyn ExternalValueStore>,
        &metrics,
        &header_key,
    )
    .await;

    assert!(
        store.batches().is_empty(),
        "the ext-GC scheduling must have been SKIPPED, not attempted — the whole point of the \
         skip posture is that an unusable release set is never published. Got {:?}",
        store.batches()
    );
    assert!(
        !pool
            .exists::<bool, _>(&header_key)
            .await
            .expect("EXISTS must succeed"),
        "the structural cleanup must complete regardless: permanent orphan retention is a strictly \
         worse failure than a ref_count leak, because only another sweep can clear it and the next \
         sweep hits the same skip"
    );
    assert!(
        !pool
            .exists::<bool, _>(&field_key)
            .await
            .expect("EXISTS must succeed"),
        "the per-field hash must be UNLINKed too — a sweep that removed only the header would \
         strand every value row with no `obj:hdr` for a later sweep to rediscover them by"
    );
    assert_eq!(
        ref_count(&tv, h).await,
        Some(1),
        "the leak is the accepted cost of the skip and must be exactly that — one un-released \
         reference, in the recoverable direction. A drop to 0 here would mean something released \
         a set that was never built"
    );
}

/// An object UNLINKed underneath the sweep leaves no `obj:hdr` behind.
///
/// The fence writes to `obj:hdr`, which is a key the sweep is about to delete.
/// Ordering is therefore load-bearing in a way it was not before the fence
/// existed — but only past one specific point, and this test pins exactly that
/// point and no more.
///
/// `obj:hdr` is element 0 of `KeySchema::object_cleanup_keys`
/// (`src/storage/keys.rs`), and that builder's dedupe pass preserves
/// first-occurrence order, so the header is always in the UNLINK set no matter
/// what the enumerations turned up. A fence moved to AFTER the enumeration but still BEFORE the UNLINK
/// is therefore removed by that same UNLINK, and this test stays green through
/// it — the header it re-creates is deleted a moment later. The hazard actually
/// pinned here is only the post-UNLINK placement: a fence written after the
/// UNLINK pass has run re-creates the header as a single-field ghost that no
/// mapping names, that carries no `seq`, and that every later sweep re-discovers
/// and re-ghosts forever.
///
/// The rendezvous stands in for a racing retirement by UNLINKing the header from
/// a second connection at the one moment a test can reach inside the sweep. The
/// end state is what is asserted: whatever raced, no `obj:hdr` may survive the
/// sweep that processed the object.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn orphan_sweep_leaves_no_ghost_header_when_the_object_is_unlinked_under_it() {
    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-postcomp-fence"));
    let ext = ext_store(&tv);
    let fx = shared_hash_fixture(&tv, &ext, &metrics, "ghost").await;
    let header_key = tv.key_schema().obj_header(&fx.doomed.to_string());

    let rendezvous: Rendezvous = {
        let pool = tv.client().pool.clone();
        let header_key = header_key.clone();
        Arc::new(move || {
            let pool = pool.clone();
            let header_key = header_key.clone();
            async move {
                let _: () = pool
                    .unlink(&header_key)
                    .await
                    .expect("the racing retirement's UNLINK must succeed");
            }
            .boxed()
        })
    };

    let store = Arc::new(RendezvousExtStore::new(ext.clone(), Some(rendezvous)));
    // As in `sweep_until_key_gone`: the orphan is fresh, so the grace is 0.
    tv.client().set_orphan_sweep_grace_for_test(Duration::ZERO);
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        tv.key_schema().clone(),
        store.clone() as Arc<dyn ExternalValueStore>,
        Arc::new(live_edit::storage::layer::LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            0,
        )),
        crate::common::test_leader("postcomp-ghost-replica"),
        cancel.clone(),
        Some(metrics.clone()),
    );
    // The header is deleted by the rendezvous, so "header gone" is not evidence
    // the sweep finished. Wait on the batch the sweep publishes just before its
    // UNLINK pass, then let the pass complete under the cancellation join.
    let deadline = tokio::time::Instant::now() + SWEEP_BUDGET;
    while store.batches().is_empty() && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(SWEEP_POLL).await;
    }
    cancel.cancel();
    tokio::time::timeout(SWEEP_JOIN_BUDGET, handle)
        .await
        .expect("the orphan sweep must exit within the join budget of cancellation")
        .expect("the orphan sweep must not panic");

    assert_eq!(
        store.batches().len(),
        1,
        "the sweep must have reached its retirement publish; without it the rendezvous never ran \
         and nothing raced"
    );
    assert!(
        !header_exists(&tv, fx.doomed).await,
        "no `obj:hdr` may survive. The fence writes that key, so a survivor means the write \
         outlived the UNLINK — a ghost header with no seq, no mapping and no way to be reclaimed \
         except by a sweep that will re-create it"
    );
}
