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

//! b.vn6 — the **producer** side of `le:ckpt:{recon}:dead`: the delete paths
//! record positive evidence that an `object_id` will never be live again, so the
//! reconciliation pass can reap the checkpoint generations it leaves behind.
//!
//! ## What is under test
//!
//! `LayerStore`'s three death sites, and the gate that turns them on:
//!
//! - `delete_layer` — the standalone `DeleteLayer` RPC path.
//! - `delete_layer_during_session_delete` — the `DeleteSession` cascade.
//! - the CreateLayer-overwrite cleanup, for the SUPERSEDED `old_oid`, which
//!   runs on a spawned task rather than inline.
//! - `with_dead_object_registration(false)` — the default, and the state of
//!   every deployment with checkpointing off.
//!
//! The consumer side — the reconciliation pass's dead-object drain, its liveness
//! re-check, its sidecar-before-delete ordering, and the three `dead_*` outcome
//! cells — is in-crate in `src/checkpoint/checkpointer.rs`'s `reconcile_tests`,
//! because `run_reconcile_dead_objects` is private and its grace window is a
//! derived quantity with a 60 s production floor. This file owns only the claim
//! that the members ever appear.
//!
//! ## Why the leak is general and not a race remedy
//!
//! Nothing in `delete_layer_inner` or the overwrite cleanup touched the
//! checkpoint store before b.vn6, and neither existing reclamation path covers a
//! dead layer: the retention prune only ever runs for a layer that is still
//! being checkpointed, and the reconciliation orphan sweep only reaps
//! *incomplete* stems. So a deleted layer's COMPLETE generations survived it
//! forever. The creation-time bootstrap race the ticket was filed for is one
//! especially fast producer of that leak, not the whole of it — which is why
//! these registrations are unconditional on the delete paths rather than
//! conditional on anything having raced.
//!
//! ## The safety rule these tests are the executable form of
//!
//! A member exists ONLY because a code path that just deleted that layer put it
//! there. Nothing infers membership from absence, and the DR-safety case in
//! `reconcile_tests` pins the other half: an object whose header and mapping are
//! both gone but which nobody registered is left strictly alone, because after a
//! keyspace wipe every surviving base matches that description until a client
//! next touches its session.
//!
//! Sandbox env note: these tests use `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox that variable is preset; see
//! CLAUDE.md.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{KeysInterface, SortedSetsInterface};

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::tail_capture::{capture_layer_tail, TailCaptureDecision};
use live_edit::checkpoint::{CheckpointError, CheckpointStore, TailCaptureConfig};
use live_edit::domain::types::{ObjectState, Section};
use live_edit::storage::external_values::{ExternalValueStore, ValkeyExternalValueStore};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

/// Stream maxlen for every `LayerStore` here — the production default.
const STREAM_MAXLEN: u64 = 4096;

/// The one root section [`DeadRegistrationFixture::create_layer_at`] plants and
/// [`DeadRegistrationFixture::commit`] writes fields into. They must agree, or
/// the delta sets a field on a section that does not exist.
const COMMIT_SECTION_ID: u64 = 1;

/// Header sequence a probe-case layer is created at, and therefore the
/// `snap_seq` of the base its lineage is rooted at: the tail capture addresses
/// `S0` from the lineage head, and its read window opens at `{S0 + 1}-0`.
///
/// Any value `>= 1` works (see `create_layer_at`); 40 is far enough above the
/// `1-0` creation seed entry that a fixture bug which captured the seed instead
/// of the commits could not pass a sequence assertion by coincidence.
const PROBE_CASE_SEED_SEQ: u64 = 40;

/// How many real commits a probe case lands above its base. Two, not one: a
/// single record cannot distinguish a capture that replayed the window from one
/// that stopped at the first entry.
const PROBE_CASE_COMMITS: u64 = 2;

/// Bound on the overwrite-cleanup drain. The cleanup runs on `LayerStore`'s own
/// tracker, so this is a *deadline*, not a sleep: the healthy path returns in
/// milliseconds and a hang here is a cleanup task that never completes.
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Everything a registration case needs, owned together so `TestValkey`'s
/// prefix-scoped Drop cleanup outlives the assertions.
struct DeadRegistrationFixture {
    tv: TestValkey,
    session_id: String,
    layers: LayerStore,
    ext_values: Arc<ValkeyExternalValueStore>,
}

/// Wire a `LayerStore` with dead-object registration resolved to `enabled`.
///
/// `enabled` is a parameter rather than a hardcoded `true` because the OFF case
/// is a real deployment shape — `main.rs` wires this to
/// `checkpoint_enabled && adapter != Disabled`, and with checkpointing off the
/// ZSET has no consumer at all, so registering would grow it by one member per
/// DeleteLayer forever against a `noeviction` Valkey.
async fn fixture(enabled: bool) -> DeadRegistrationFixture {
    let tv = TestValkey::setup().await;

    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(
            "dead-object-registration-session",
            "/dead-object-registration",
            "dead-object-test-user",
        )
        .await
        .expect("session creation must succeed against a healthy Valkey");

    let layers = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), STREAM_MAXLEN)
        .with_dead_object_registration(enabled);
    let ext_values = Arc::new(ValkeyExternalValueStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    DeadRegistrationFixture {
        tv,
        session_id,
        layers,
        ext_values,
    }
}

impl DeadRegistrationFixture {
    async fn create_layer(&self, layer_name: &str) -> u64 {
        self.layers
            .create_layer(
                &self.session_id,
                layer_name,
                "dead-object-test-user",
                &ObjectState::default(),
                Arc::clone(&self.ext_values) as Arc<dyn ExternalValueStore>,
            )
            .await
            .expect("create_layer must succeed against a healthy Valkey")
    }

    /// A layer whose header sequence starts at `sequence_num`, carrying one root
    /// section so a field-setting delta has somewhere to land.
    ///
    /// [`Self::create_layer`]'s `ObjectState::default()` stamps `seq = 0` while
    /// `CreateLayer` seeds the stream at `1-0`, so the first `UpdateLayer` on such
    /// a layer would mint `1-0` a second time and be refused by the stream-top
    /// guard. Any `sequence_num >= 1` avoids that; the commits then land above
    /// both the seed entry and the lineage base.
    async fn create_layer_at(&self, layer_name: &str, sequence_num: u64) -> u64 {
        let mut state = ObjectState {
            sequence_num,
            first_version: 1,
            source_format: "usda".to_string(),
            next_set_order: 0,
            ..Default::default()
        };
        state.sections.insert(
            COMMIT_SECTION_ID,
            Section {
                section_id: COMMIT_SECTION_ID,
                parent_id: 0,
                name: format!("{layer_name}-sec"),
                section_type: 1,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
        state.children.insert(0, vec![COMMIT_SECTION_ID]);

        self.layers
            .create_layer(
                &self.session_id,
                layer_name,
                "dead-object-test-user",
                &state,
                Arc::clone(&self.ext_values) as Arc<dyn ExternalValueStore>,
            )
            .await
            .expect("create_layer must succeed against a healthy Valkey")
    }

    /// One real `UpdateLayer` commit carrying a REAL `relay_ompf`.
    ///
    /// Both halves matter: a field-only delta rides the Lua fast path, and an
    /// `ompf`-less update entry is classified as an EMPTY commit by invariant, so
    /// a fixture built that way would record real commits as no-ops and the tail
    /// capture would refuse before ever reaching the PUT this file is about
    /// (shape borrowed from `restore_rebase_test.rs::commit`).
    async fn commit(&self, layer_name: &str, object_id: u64, nonce: u64) -> u64 {
        let delta = oodf_ffi::OwnedDelta {
            is_diff: true,
            base_version: 1,
            set_fields: vec![oodf_ffi::SetField {
                section_id: COMMIT_SECTION_ID,
                key_name: "dead-object-field".to_string(),
                value_or_ext_hash: nonce.to_le_bytes().to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: nonce,
            }],
            ..Default::default()
        };
        let ompf = bytes::Bytes::from(crate::common::owned_delta::delta_to_ompf_bytes(&delta, &[]));
        let (sequence_num, _) = self
            .layers
            .update_layer(
                &self.session_id,
                layer_name,
                object_id,
                Some(delta),
                Some(ompf),
                "dead-object-test-writer",
                0,
            )
            .await
            .expect("update_layer must succeed against a healthy Valkey");
        sequence_num
    }

    fn dead_key(&self) -> String {
        self.tv.key_schema().checkpoint_reconcile_dead()
    }

    /// Every member of `le:ckpt:{recon}:dead`, sorted.
    async fn dead_members(&self) -> Vec<String> {
        let mut members: Vec<String> = self
            .tv
            .client()
            .pool
            .zrange(&self.dead_key(), 0, -1, None, false, None, false)
            .await
            .expect("ZRANGE over le:ckpt:{recon}:dead must succeed");
        members.sort();
        members
    }

    /// The `ZADD NX` first-observation score of one member.
    async fn dead_score(&self, object_id: u64) -> Option<f64> {
        self.tv
            .client()
            .pool
            .zscore(&self.dead_key(), object_id.to_string())
            .await
            .expect("ZSCORE must succeed")
    }

    /// Valkey's own millisecond clock, read through `TIME`.
    ///
    /// The scores on every `{recon}` ZSET are sampled **server-side**, inside
    /// the claim script, precisely so a fleet with NTP skew still agrees on the
    /// watermark (`recon_claim_expired.lua`'s "Valkey-side TIME" note). A score
    /// assertion measured against the local `SystemTime` would therefore be
    /// testing a different clock from the one the protocol runs on — usually
    /// close enough to pass, and wrong in exactly the deployments the
    /// server-side sample exists for.
    async fn valkey_now_ms(&self) -> f64 {
        use fred::interfaces::ClientLike;
        let parts: Vec<String> = self
            .tv
            .client()
            .pool
            .custom::<Vec<String>, String>(
                fred::types::CustomCommand::new_static(
                    "TIME",
                    fred::types::ClusterHash::Random,
                    false,
                ),
                Vec::<String>::new(),
            )
            .await
            .expect("TIME must succeed");
        let secs: f64 = parts[0].parse().expect("TIME seconds must parse");
        let micros: f64 = parts[1].parse().expect("TIME microseconds must parse");
        secs * 1000.0 + micros / 1000.0
    }

    async fn ext_store(&self) -> Arc<dyn ExternalValueStore> {
        Arc::clone(&self.ext_values) as Arc<dyn ExternalValueStore>
    }
}

// =============================================================================
// The three death sites.
// =============================================================================

/// `DeleteLayer` registers the retired `object_id` as dead.
///
/// The registration sits at `delete_layer_inner`'s success-only exit, beside the
/// ext-GC obligation the same path already discharges, and for the same reason:
/// that is the moment — and the only moment — at which the service knows for
/// certain this `object_id` will never be live again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_registers_the_retired_object_on_the_dead_zset() {
    let fx = fixture(true).await;
    let oid = fx.create_layer("dead-registration-delete-layer").await;

    assert!(
        fx.dead_members().await.is_empty(),
        "precondition: creating a layer must register nothing — only a DEATH may mint a member, \
         and a member minted anywhere else would be the inferential input this ZSET's whole \
         safety argument excludes"
    );

    let before_ms = fx.valkey_now_ms().await;
    fx.layers
        .delete_layer(
            &fx.session_id,
            "dead-registration-delete-layer",
            fx.ext_store().await,
        )
        .await
        .expect("delete_layer must succeed against a healthy Valkey");
    let after_ms = fx.valkey_now_ms().await;

    assert_eq!(
        fx.dead_members().await,
        vec![oid.to_string()],
        "the deleted object must be on le:ckpt:{{recon}}:dead. Without it nothing in the system \
         ever revisits its checkpoint generations: the retention prune runs only for layers still \
         being checkpointed, and the orphan sweep reaps only incomplete stems"
    );

    // ── The score is a FRESH Valkey-side timestamp ───────────────────────────
    //
    // The member's presence is only half the contract. Its score is the
    // first-observation timestamp the reconciliation pass's claim compares
    // against `now − grace`, so the two failure directions are opposite and both
    // silent:
    //
    // - a score in the FUTURE (or absurdly large) is never older than any grace,
    //   so the member is claimed on no pass and the generations leak forever —
    //   the exact leak this ZSET exists to close, reached through its own
    //   bookkeeping;
    // - a score at or near ZERO is instantly claimable, which collapses the
    //   window the drain's rustdoc calls load-bearing: a checkpoint tick can
    //   already be in flight when the layer dies, and claiming before its PUT
    //   becomes enumerable makes the object read clean, retires the member, and
    //   orphans the generation that lands a moment later.
    //
    // Bracketed by two `TIME` reads around the delete, so the assertion is
    // against the same server-side clock the script samples rather than this
    // process's `SystemTime`.
    let score = fx
        .dead_score(oid)
        .await
        .expect("the member was just asserted present, so it carries a score");
    assert!(
        score >= before_ms && score <= after_ms,
        "the first-observation score must be Valkey's own clock, sampled during the delete: got \
         {score}, expected within [{before_ms}, {after_ms}]. A zero would make the member \
         claimable before an in-flight checkpoint PUT could become enumerable; a future value \
         would make it claimable on no pass at all"
    );
    assert!(
        score > 0.0,
        "and explicitly non-zero — the epoch is the value a test fixture pins, never one the \
         producer path can mint"
    );
}

/// The `DeleteSession` cascade registers too — the second entry point onto the
/// same inner path.
///
/// Asserted separately rather than folded into the case above because the two
/// entry points differ in a way that could plausibly have been made to matter:
/// `delete_layer_during_session_delete` suppresses the per-layer
/// session-checkpoint rewrite. A registration placed inside that suppressed
/// branch would leak every layer of every deleted session and pass the
/// standalone case unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_during_session_delete_registers_the_retired_object_too() {
    let fx = fixture(true).await;
    let oid = fx.create_layer("dead-registration-session-cascade").await;

    fx.layers
        .delete_layer_during_session_delete(
            &fx.session_id,
            "dead-registration-session-cascade",
            fx.ext_store().await,
        )
        .await
        .expect("delete_layer_during_session_delete must succeed");

    assert_eq!(
        fx.dead_members().await,
        vec![oid.to_string()],
        "a layer removed by the DeleteSession cascade is as dead as one removed by DeleteLayer, \
         and carries the same reclamation obligation — the suppressed session-checkpoint rewrite \
         is about the SESSION blob, not about the layer's own generations"
    );
}

/// A CreateLayer overwrite registers the SUPERSEDED `old_oid`.
///
/// The superseded object is as dead as a `DeleteLayer`'d one — its header is
/// UNLINKed by the same cleanup and no mapping names it any more — but it dies
/// on a *spawned* task that owns no `&self`, which is a separate code path from
/// the inline one above and the reason this case exists.
///
/// `wait_for_cleanup` rather than a poll or a sleep: the cleanup is tracked, so
/// the drain is deterministic and a hang is a real bug rather than a slow
/// runner.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_registers_the_superseded_object_on_the_dead_zset() {
    let fx = fixture(true).await;
    let layer_name = "dead-registration-overwrite";

    let old_oid = fx.create_layer(layer_name).await;
    let new_oid = fx.create_layer(layer_name).await;
    assert_ne!(
        old_oid, new_oid,
        "precondition: an overwrite must allocate a NEW object_id — the two-level versioning \
         contract, and what makes the old one retirable at all"
    );

    fx.layers.wait_for_cleanup(CLEANUP_TIMEOUT).await;

    assert_eq!(
        fx.dead_members().await,
        vec![old_oid.to_string()],
        "exactly the superseded object, and not the live one. Registering {new_oid} would hand \
         the reconciliation pass a live layer to reap — the liveness re-check would spare it, but \
         relying on that is relying on a backstop to cover a producer bug"
    );
}

// =============================================================================
// `ZADD NX` semantics, and the gate.
// =============================================================================

/// A registration over a member that is ALREADY on the ZSET does not move its
/// score.
///
/// The score is the *first-observation* timestamp, and the reconciliation pass
/// claims only members whose score is older than the grace. A score refreshed on
/// every write would keep a member permanently younger than the grace, so the
/// pass would enumerate the same work forever and reclaim none of it — the exact
/// failure the two sibling `{recon}` ZSETs pin `ZADD NX` against, and the reason
/// this key inherits the same rule.
///
/// The prior observation is planted at the epoch rather than produced by a first
/// real death, because the honest production history is precisely "an earlier
/// pass has not drained this member yet": one `object_id` cannot die twice, so
/// the only way a second `ZADD` lands on an existing member is a delete whose
/// registration already happened — a retried cascade, or a member the pass left
/// registered after a failed reap. Score `0` also makes the assertion sharp: a
/// refreshed score would be the current Valkey clock, which is not a value a
/// tolerance could confuse with the epoch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_registration_over_an_already_registered_dead_object_does_not_refresh_its_score() {
    let fx = fixture(true).await;
    let layer_name = "dead-registration-nx";

    let oid = fx.create_layer(layer_name).await;

    // The earlier observation the pass has not drained yet.
    let _: () = fx
        .tv
        .client()
        .pool
        .zadd(
            fx.dead_key(),
            None,
            None,
            false,
            false,
            (0.0, oid.to_string()),
        )
        .await
        .expect("planting the prior first-observation must succeed");

    fx.layers
        .delete_layer(&fx.session_id, layer_name, fx.ext_store().await)
        .await
        .expect("delete_layer must succeed");

    assert_eq!(
        fx.dead_score(oid).await,
        Some(0.0),
        "the first writer pins the score and no later registration refreshes it. A refreshed \
         score here would read as the current Valkey clock, which is never older than the grace, \
         so the dead-object phase would claim this member every pass and reap it on none of them"
    );
    assert_eq!(
        fx.dead_members().await,
        vec![oid.to_string()],
        "and the member set is unchanged — a ZSET member is its id, so a repeat cannot duplicate"
    );
}

/// With registration off — the default, and every deployment running without
/// checkpointing — the delete paths write nothing at all.
///
/// This is not a cosmetic gate. `le:ckpt:{recon}:dead` has exactly one consumer,
/// the checkpoint loop's reconciliation tick, and that loop is spawned only when
/// `LIVE_EDIT_CHECKPOINT_ENABLED=true` against a real adapter. Registering
/// without it would add one member per DeleteLayer, forever, to a ZSET that is
/// deliberately TTL-less, against a Valkey configured `noeviction` — an
/// unbounded leak introduced by a leak fix.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_layer_store_without_dead_object_registration_writes_no_member() {
    let fx = fixture(false).await;
    let layer_name = "dead-registration-gate-off";

    let old_oid = fx.create_layer(layer_name).await;
    // Both producers, so a gate honoured on one path and dropped on the other
    // cannot pass: the overwrite cleanup captures the flag by value into a
    // spawned task, which is a genuinely separate read of it.
    let new_oid = fx.create_layer(layer_name).await;
    fx.layers.wait_for_cleanup(CLEANUP_TIMEOUT).await;
    fx.layers
        .delete_layer(&fx.session_id, layer_name, fx.ext_store().await)
        .await
        .expect("delete_layer must succeed");

    assert!(
        fx.dead_members().await.is_empty(),
        "neither the superseded {old_oid} nor the deleted {new_oid} may be registered when the \
         gate is off: with no checkpoint loop there is nothing to drain the ZSET, so every member \
         written is a permanent one"
    );
}

// =============================================================================
// b.hka — a dead object costs the checkpoint store NOTHING
// =============================================================================
//
// The ZSET above is the BACKSTOP: it reclaims artifacts that were written for an
// object that had already died. b.hka added the complementary front half — a
// pre-PUT `HEXISTS obj:hdr seq` liveness probe (`checkpoint::put_guard`) routed
// through all four durable-generation producers — whose value is the write that
// never happens: a multi-MB PUT, a retention slot, and a round trip through this
// very reclamation path, all avoided for the common case.
//
// ## Two different claims, and which case pins which
//
// "Reaching the producer" and "reaching the pre-PUT probe" are NOT the same
// claim, and conflating them is easy: three of the four producers ask
// `read_header_identity` first, so a header that is simply GONE is refused there
// and `put_layer_checkpoint_if_live` is never consulted about it. A case that
// deletes `obj:hdr` before the call therefore pins the IDENTITY refusal — a real
// claim, and the one that says nothing is written — while saying nothing at all
// about whether the producer wired the probe in. Both halves are covered below,
// by different cases.
//
// ## Where each of the four producers is pinned
//
// The probe is `pub(crate)`, so an integration test reaches it only through a
// producer.
//
// - **tail capture** — the one producer whose pre-PUT probe an integration test
//   can reach, so BOTH halves live here:
//   - the identity refusals — `a_tail_capture_for_a_dead_object_…` (header
//     deleted) and `a_tail_capture_for_an_identity_empty_header_…` (an identity
//     column emptied), which return `NotFound` at step 3, before any slice read;
//   - the PROBE — `a_tail_capture_whose_header_is_a_cursorless_phantom_…`, which
//     drives the same public `capture_layer_tail` over a lineage carrying REAL
//     commits, so the call runs all the way to step 6 and the verdict under test
//     is the probe's own.
// - **assembler** — `assembler_test.rs`'s
//   `assembler_writes_nothing_when_the_object_header_is_gone` pins its IDENTITY
//   refusal, and deliberately not its probe: `assemble_inner` pins
//   `obj:hdr seq` in step 1 of the stable cut and returns `NotFound` when that
//   column is absent, so the phantom fixture the tail case uses cannot reach its
//   step 7 — and it constructs its `ValkeySnapshotLoader` internally, so there is
//   no seam to kill the header mid-flight either. Its probe call site is one line
//   whose behaviour is pinned in `src/checkpoint/put_guard.rs`'s `mod tests`.
// - **creation-time base bootstrap** — its own `SkippedHeaderAbsent` outcome
//   cell, driven by `first_base_bootstrap_test.rs` case 16 and asserted on the
//   cell in `bootstrap_observability_test.rs`. That coverage predates b.hka and
//   is unchanged by it: the bootstrap's inline probe was folded INTO the shared
//   helper rather than replaced, so case 16 is a probe case, not an identity one.
// - **re-base** — not reachable. Its public `rebase_layer_lineage` passes
//   `None` for the probe by construction (it holds no Valkey client); only the
//   checkpoint loop's `pub(crate)` `rebase_layer_lineage_with_probe` carries
//   one. The shared helper's own three verdicts are pinned in-crate in
//   `src/checkpoint/put_guard.rs`'s `mod tests`, which is what makes this
//   producer's wiring a one-line reading rather than an untested path.
//
// ## What these cases can and cannot show
//
// None of them races. The identity cases degrade `obj:hdr` BEFORE the call; the
// probe case presents a header state in which the two predicates simply
// disagree (see that case's own doc). The real race — a `DeleteLayer` landing
// while the payload is being built — has a window of microseconds and no seam to
// force it deterministically, and a test that raced for it would be flaky in
// exactly the direction that hides a regression. The deterministic forms assert
// the same observable and the one that matters: the object is dead, so nothing
// is written for it.
//
// ## The identity-empty sibling
//
// A third case below drives the SAME producer against a header that exists but
// has lost an identity column. It belongs here rather than beside the assembler's
// own identity cases because the claim is this file's — "for DR purposes the
// object does not exist, so it costs the checkpoint store nothing" — and because
// since b.hka review round 2 the two states arrive at the tail-capture unit
// through one answer: `read_header_identity` reports both as `Ok(None)`.

/// A tail capture against a layer whose `obj:hdr` is gone writes NOTHING.
///
/// `NotFound` is the verdict, not a refusal decision and not an error the loop
/// counts: it is the arm the checkpoint loop already drops silently at `debug!`
/// with no metric tick and no cursor stamp, which is exactly the truth — the
/// object will never be enumerated again, so there is nothing to retry and
/// nothing to persist, and counting it would feed the adapter-health alert from
/// an adapter that was healthy.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tail_capture_for_a_dead_object_writes_no_generation() {
    let fx = fixture(true).await;
    let layer_name = "dead-object-tail-capture";
    let oid = fx.create_layer(layer_name).await;

    // A base has to exist in the store or the capture refuses at step 1 with
    // `NoBase` — a lineage-addressing verdict taken off the LISTING, before any
    // Valkey read at all, which would make this case pass without ever asking
    // about the header.
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    super::helpers::put_base_generation(&store, oid, &fx.session_id, layer_name, 1).await;

    let header_key = fx.tv.key_schema().obj_header(&oid.to_string());
    let deleted: u64 = fx
        .tv
        .client()
        .pool
        .del(&header_key)
        .await
        .expect("DEL obj:hdr must succeed");
    assert_eq!(
        deleted, 1,
        "fixture precondition: the header must have been there to remove, or the capture would \
         be refusing a layer that never existed and the case would prove nothing"
    );

    let err = capture_layer_tail(
        fx.tv.client(),
        fx.tv.key_schema(),
        oid,
        store.as_ref(),
        "dead-object-test-replica",
        &TailCaptureConfig::default(),
    )
    .await
    .expect_err("a capture for an object with no header is a refusal, not a decision");

    assert!(
        matches!(err, CheckpointError::NotFound),
        "`NotFound` specifically — the loop's benign-and-terminal arm. Got {err:?}"
    );
    let listed = store
        .list_layer_checkpoints(oid)
        .await
        .expect("listing an in-memory store cannot fail");
    assert_eq!(
        listed.len(),
        1,
        "the store must still hold ONLY the seeded base — no tail generation may have been \
         written for a dead object. An artifact here would be durable garbage nobody ever \
         reads, occupying a retention slot until the dead-object reconcile phase above \
         reclaimed it. Got {listed:?}"
    );
    assert_eq!(listed[0].kind, live_edit::checkpoint::CheckpointKind::Base);
}

/// b.hka review round 2 — a tail capture against a layer whose `obj:hdr` EXISTS
/// but carries no `session_id` writes NOTHING, on the same `NotFound` arm.
///
/// # Why this producer had no coverage before, and needed a fix
///
/// The capture unit's step 3 refused only an ABSENT header. A degenerate one —
/// present, `seq` intact, an identity column lost to a partial write or a
/// hand-repair — was admitted, the bounded slice read and the envelope encode
/// were paid for, and the meta's four identity columns were populated from it.
/// `validate_generation_identity` then refuses that meta at every adapter with a
/// `Permanent`, so the layer ticked `writes_total{result="error"}` on every tick,
/// forever, from a perfectly healthy adapter. Round 2 moved the decision into
/// `read_header_identity`, which now answers `Ok(None)` for both header states,
/// so this unit's pre-existing `NotFound` arm covers it with no new branch.
///
/// Pre-fix the call below returned `Err(Permanent)` from the adapter's identity
/// gate — a different variant, on a different line, after a wasted PUT — so both
/// assertions discriminate.
///
/// The `HDEL` leaves `seq` alone deliberately: without it this would be the
/// vanished-header case above wearing a different name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tail_capture_for_an_identity_empty_header_writes_no_generation() {
    use fred::interfaces::HashesInterface;

    let fx = fixture(true).await;
    let layer_name = "identity-empty-tail-capture";
    let oid = fx.create_layer(layer_name).await;

    // As above: a base must exist in the store or the capture refuses at step 1
    // with `NoBase`, before any Valkey read at all.
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    super::helpers::put_base_generation(&store, oid, &fx.session_id, layer_name, 1).await;

    let header_key = fx.tv.key_schema().obj_header(&oid.to_string());
    let removed: u64 = fx
        .tv
        .client()
        .pool
        .hdel(&header_key, "session_id")
        .await
        .expect("HDEL obj:hdr session_id must succeed");
    assert_eq!(
        removed, 1,
        "fixture precondition: the column must have been there to remove, or the case would pass \
         vacuously against a header that never carried it"
    );
    let seq_survives: bool = fx
        .tv
        .client()
        .pool
        .hexists(&header_key, "seq")
        .await
        .expect("HEXISTS obj:hdr seq must succeed");
    assert!(
        seq_survives,
        "fixture precondition: `seq` must survive, or this is the vanished-header case above"
    );

    let err = capture_layer_tail(
        fx.tv.client(),
        fx.tv.key_schema(),
        oid,
        store.as_ref(),
        "dead-object-test-replica",
        &TailCaptureConfig::default(),
    )
    .await
    .expect_err("a layer with no identity cannot produce a restorable generation");

    assert!(
        matches!(err, CheckpointError::NotFound),
        "`NotFound` specifically — the loop's benign-and-terminal arm, dropped at `debug!` with \
         no metric tick. Pre-fix this was the adapter's identity `Permanent`, which the loop \
         counts on `writes_total{{result=\"error\"}}` and which therefore alerted on a healthy \
         adapter. Got {err:?}"
    );
    let listed = store
        .list_layer_checkpoints(oid)
        .await
        .expect("listing an in-memory store cannot fail");
    assert_eq!(
        listed.len(),
        1,
        "the store must still hold ONLY the seeded base: a generation whose identity columns are \
         empty is refused by every adapter and unrestorable by the apply path, so the whole \
         encode-and-PUT round trip is waste. Got {listed:?}"
    );
    assert_eq!(listed[0].kind, live_edit::checkpoint::CheckpointKind::Base);
}

/// The negative control for the case above.
///
/// Without it, a `capture_layer_tail` that refused EVERY layer — or an
/// `InMemoryCheckpointStore` whose listing never reported anything — would
/// satisfy every assertion there. This drives the same call against the same
/// fixture with the header left alone, and requires the capture to reach a real
/// decision rather than the `NotFound` refusal.
///
/// It deliberately asserts only "not the dead-object refusal" and not a
/// specific captured shape: a freshly created layer has no commits above its
/// base to capture, so the honest decision here is one of the capture unit's own
/// refusals (which are `Ok(_)`, not `Err`). Pinning which one would duplicate
/// `checkpointer_tail_cadence_test`'s subject and couple this case to it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tail_capture_for_a_live_object_is_not_refused_as_dead() {
    let fx = fixture(true).await;
    let layer_name = "live-object-tail-capture";
    let oid = fx.create_layer(layer_name).await;

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    super::helpers::put_base_generation(&store, oid, &fx.session_id, layer_name, 1).await;

    let decision = capture_layer_tail(
        fx.tv.client(),
        fx.tv.key_schema(),
        oid,
        store.as_ref(),
        "dead-object-test-replica",
        &TailCaptureConfig::default(),
    )
    .await;

    assert!(
        decision.is_ok(),
        "a LIVE layer must never reach the dead-object arm — if it did, the case above would \
         pass on every input and the guard would be silently dropping real generations. Every \
         other outcome the capture unit can reach past step 3 is an `Ok(_)` decision. Got \
         {decision:?}"
    );
}

/// **The pre-PUT probe itself**, reached through the production entry point: a
/// capture that gets all the way to step 6 with everything built, and writes
/// nothing because the object is provably dead.
///
/// # Why this case exists when two others already assert "writes nothing"
///
/// Those two degrade `obj:hdr` before the call, so `capture_layer_tail` refuses
/// at step 3's `read_header_identity` and never consults
/// `put_layer_checkpoint_if_live` at all. They pin the identity refusal. Deleting
/// the probe from the producer — or wiring it with `None` — leaves both of them
/// green. This one goes red, because it is the only case here in which the probe
/// is the thing that decides.
///
/// # The fixture: a cursor-less phantom, not a race
///
/// `HDEL obj:hdr seq` leaves a header the two predicates disagree about, which
/// is exactly the disagreement `checkpoint::put_guard`'s module header describes:
///
/// - `read_header_identity` (`HGETALL`) sees a non-empty hash with both identity
///   columns intact, so step 3 reports the layer as LIVE and the capture proceeds
///   through the slice read and the envelope encode;
/// - `HEXISTS obj:hdr seq` — the probe's predicate, and the same one
///   `stamp_cursor_if_live.lua` uses — answers `false`, so step 6 skips the PUT.
///
/// That is a real production state (a pre-b.vn6 cursor-only phantom, or a header
/// mid-repair), and it presents the code under test with the identical
/// observable a mid-flight `DeleteLayer` would — the probe answers `Dead` with a
/// fully built generation in hand — with no sleep, no retry loop and no timing
/// assumption. The genuine race has a microsecond window and no seam; a test that
/// chased it would be flaky in the direction that hides regressions.
///
/// # What "writes nothing" is read off
///
/// The store listing, which still holds ONLY the seeded base. Pre-fix the capture
/// would have PUT a tail here — a multi-MB durable artifact nobody will ever
/// read, holding a retention slot until the dead-object reconcile phase above
/// reclaims it, which is the whole cost this guard exists to avoid.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tail_capture_whose_header_is_a_cursorless_phantom_writes_no_generation() {
    use fred::interfaces::HashesInterface;

    let fx = fixture(true).await;
    let layer_name = "phantom-header-tail-capture";
    let (oid, store) = seed_capturable_lineage(&fx, layer_name).await;

    let header_key = fx.tv.key_schema().obj_header(&oid.to_string());
    let removed: u64 = fx
        .tv
        .client()
        .pool
        .hdel(&header_key, "seq")
        .await
        .expect("HDEL obj:hdr seq must succeed");
    assert_eq!(
        removed, 1,
        "fixture precondition: `seq` must have been there to remove, or the probe would be \
         answering about a header that never had a cursor column"
    );
    for column in ["session_id", "layer_name"] {
        let survives: bool = fx
            .tv
            .client()
            .pool
            .hexists(&header_key, column)
            .await
            .expect("HEXISTS obj:hdr must succeed");
        assert!(
            survives,
            "fixture precondition: `{column}` must survive, or step 3 refuses on identity and \
             this is the identity case above wearing a different name"
        );
    }

    let err = capture_layer_tail(
        fx.tv.client(),
        fx.tv.key_schema(),
        oid,
        store.as_ref(),
        "dead-object-test-replica",
        &TailCaptureConfig::default(),
    )
    .await
    .expect_err("a capture whose object is provably dead at PUT time is a refusal, not a write");

    assert!(
        matches!(err, CheckpointError::NotFound),
        "`NotFound` specifically — the loop's benign-and-terminal arm, which is what the skipped \
         PUT's `Ok(None)` folds onto. A `Permanent` or a `Transient` here would tick \
         adapter-health for an adapter that was never even called. Got {err:?}"
    );
    let listed = store
        .list_layer_checkpoints(oid)
        .await
        .expect("listing an in-memory store cannot fail");
    assert_eq!(
        listed.len(),
        1,
        "ZERO puts: the store must still hold only the seeded base. The sibling control below \
         proves this same fixture DOES reach the PUT when the header is left alone, so a `1` \
         here is the probe's decision and not a capture that refused earlier. Got {listed:?}"
    );
    assert_eq!(listed[0].kind, live_edit::checkpoint::CheckpointKind::Base);
}

/// The control that makes the case above non-vacuous: the SAME fixture, with the
/// header left alone, captures and persists a tail.
///
/// Without it, a fixture that refused at any earlier arm — no commits above the
/// base, an unclassifiable slice, a chain bound already spent — would satisfy
/// every assertion there while never reaching the probe at all, and the case
/// would be pinning nothing. This is also the only place in this file that
/// asserts the capture path runs to completion, so it is what makes "the probe
/// is what decided" a claim rather than a hope.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tail_capture_over_the_probe_fixture_persists_a_tail_when_the_header_is_intact() {
    let fx = fixture(true).await;
    let layer_name = "phantom-header-control";
    let (oid, store) = seed_capturable_lineage(&fx, layer_name).await;

    let decision = capture_layer_tail(
        fx.tv.client(),
        fx.tv.key_schema(),
        oid,
        store.as_ref(),
        "dead-object-test-replica",
        &TailCaptureConfig::default(),
    )
    .await
    .expect("a live layer with commits above its base must capture");

    match decision {
        TailCaptureDecision::Captured {
            from_seq,
            to_seq,
            record_count,
            ..
        } => {
            assert_eq!(
                (from_seq, to_seq),
                (
                    PROBE_CASE_SEED_SEQ,
                    PROBE_CASE_SEED_SEQ + PROBE_CASE_COMMITS
                ),
                "the capture must cover exactly the commits landed above the base"
            );
            assert_eq!(record_count, PROBE_CASE_COMMITS as usize);
        }
        other => panic!(
            "the probe fixture must reach the PUT, or the phantom case above refuses for a \
             reason that has nothing to do with the probe. Got {other:?}"
        ),
    }

    let listed = store
        .list_layer_checkpoints(oid)
        .await
        .expect("listing an in-memory store cannot fail");
    assert_eq!(
        listed.len(),
        2,
        "base + the tail this capture wrote — the artifact the phantom case must NOT produce. \
         Got {listed:?}"
    );
    assert!(
        listed
            .iter()
            .any(|h| h.kind == live_edit::checkpoint::CheckpointKind::Tail),
        "and one of them is a Tail. Got {listed:?}"
    );
}

/// Shared setup for the probe case and its control: a real layer at
/// [`PROBE_CASE_SEED_SEQ`], a lineage base rooted at that same sequence, and
/// [`PROBE_CASE_COMMITS`] real `UpdateLayer` commits above it.
///
/// The base sequence and the layer's creation sequence are deliberately equal:
/// the capture addresses `S0` off the lineage head, so this makes the read window
/// open at the first commit and nothing else.
async fn seed_capturable_lineage(
    fx: &DeadRegistrationFixture,
    layer_name: &str,
) -> (u64, Arc<dyn CheckpointStore>) {
    let oid = fx.create_layer_at(layer_name, PROBE_CASE_SEED_SEQ).await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    super::helpers::put_base_generation(
        &store,
        oid,
        &fx.session_id,
        layer_name,
        PROBE_CASE_SEED_SEQ,
    )
    .await;

    for i in 0..PROBE_CASE_COMMITS {
        let seq = fx.commit(layer_name, oid, 7_000 + i).await;
        assert_eq!(
            seq,
            PROBE_CASE_SEED_SEQ + i + 1,
            "the commits must land contiguously above the base, or the classifier refuses on a \
             sequence gap before the capture ever reaches the PUT"
        );
    }

    (oid, store)
}
