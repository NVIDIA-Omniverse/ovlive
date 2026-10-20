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

//! SR-1.3 `CheckpointStore` conformance harness.
//!
//! This file is the parametrize-over-closure scaffold — every per-case body
//! (`case_01_*` … `case_43_*`) is a fully populated `pub async fn` that the
//! three per-adapter wrapper binaries drive against their adapter:
//! `adapter_in_memory_test.rs`, `adapter_local_disk_test.rs`, and
//! `adapter_storage_api_test.rs`, all alongside this file under
//! `tests/integration/checkpoint/`.
//!
//! The harness is parameterized by a setup closure of shape
//! `Fn() -> impl Future<Output = (Arc<dyn CheckpointStore>, Guard)>`. The
//! returned `Guard` carries any per-test cleanup state (the `tempfile::TempDir`
//! for LocalDisk; an empty struct for InMemory) and is held to end-of-test
//! scope so cleanup runs when the test future drops.
//!
//! Cases #2, #6, #15 were skip-gated under Epic 1; Task `t2.fgj.7v.cq` landed
//! live session-checkpoint impls in both adapters and Subtask `t3.fgj.7v.cq.7y`
//! removed the gates. The [`skip_if_not_implemented`] helper and macro remain
//! in place as the established pattern if a future adapter ever needs to gate
//! against a new `NotImplemented`-shaped contract.
//!
//! ## Error-classification cases (#19, #20, #22, and Group J's #35, #36)
//!
//! These cases use the [`CheckpointFaultInjector`] seam exposed under the
//! `test-checkpoint-fault-injection` cargo feature; their per-case signatures
//! take a second factory that accepts an `Arc<dyn CheckpointFaultInjector>` and
//! threads it through the adapter's `with_fault_injector(...)` builder method.
//! When the feature is off the seam is compiled out entirely and the case
//! bodies become a no-op (callers won't invoke them — the adapter wrapper
//! modules only register them under the same feature gate). Group J's #35 and
//! #36 extend the same surface to `get_layer_checkpoint` and to a **tail** put,
//! and are gated identically.
//!
//! ## Group J — tail generations (#28–#37)
//!
//! Epic 2 extends the trait with a generation *kind*, so the harness grows a
//! block for it. (Group `I` is raw artifact enumeration, #27 — the tail block
//! takes the next free letter rather than sharing one.) The contracts these
//! cases pin, all cross-adapter:
//!
//! - A tail is put through the **same** `put_layer_checkpoint`, with the kind
//!   read off `meta.kind` and the envelope's `to_seq` passed as `snap_seq`; the
//!   returned handle's `kind` MUST match.
//! - `list_layer_checkpoints` interleaves both kinds in one strict descending
//!   order over the sequence, with the pinned **Base-before-Tail** tie-break at
//!   an equal sequence.
//! - Publish is **meta-first**, so a listed tail always has its meta; on the
//!   adapters that name artifacts the tail's stem is `{oid}.{to_seq}.tail`
//!   alongside `{oid}.{to_seq}.tail.meta.json`, disjoint from the base triplet
//!   at the same sequence.
//! - A tail carries **no ext-refs sidecar** — a `Tail` meta with
//!   `sidecar_bytes: Some` is a `Permanent` rejection, not a silently-dropped
//!   field.
//!
//! Tail bytes are **opaque** here: the fixtures are synthetic
//! ([`sample_tail_envelope`]), and this file has no dependency on
//! `src/checkpoint/tail_codec.rs` and must not acquire one — adapters are
//! contractually blind to the envelope's structure, so a conformance case that
//! knew it would be testing the wrong layer.
//!
//! No Group J case is `skip_if_not_implemented!`-gated: all three shipped
//! adapters store tails, so a `NotImplemented` out of one is a defect to fail
//! on, not a capability to skip.
//!
//! ## Group K — lineage-aware retention (#38)
//!
//! Epic 5 makes retention operate on **lineage units** (a base plus the tails
//! rooted on it), so the pruner's use of the trait changes shape: an ordered
//! *run* of deletes per condemned unit — tails newest-first, base last — issued
//! over handles captured from an earlier listing, and re-issuable in whole or in
//! part by a second leader. Group K pins what that run may do at the adapter:
//! answer `Ok(())` at every position however many times it is issued, and leave
//! every generation outside the condemned lineage — including a sibling lineage
//! under the *same* `object_id`, and the ext-hash live set — untouched.
//!
//! ## Group M — the DR restore Job's read surface (#40–#42)
//!
//! `list_session_checkpoint_ids` and `ext_blob_exists` have one consumer, the
//! DR restore Job, and it acts on a *negative* answer: a session missing from
//! the listing is not restored, and a value whose bytes are reported absent is
//! stripped. So #40 and #41 pin both halves of each answer — what is stored is
//! reported, what is not is not — and #42 pins that a failing backend is an
//! error, never the negative answer. (`L` is the StorageApi wrapper's own
//! redirect block, so this group takes `M`.)
//!
//! ## Group N — the live-set walk's own evidence (#43)
//!
//! The reconciliation collect deletes every aged tombstoned blob the live set
//! does not name, and an EMPTY live set licenses that only when the walk's
//! own listing named a layer artifact (`list_live_ext_set`'s count). #43 pins
//! the count's two answers across adapters — 0 for an empty store, every layer
//! artifact the walk listed otherwise — and that `list_referenced_ext_hashes`
//! answers the same set.

use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;

use live_edit::checkpoint::{
    CheckpointError, CheckpointHandle, CheckpointKind, CheckpointMeta, CheckpointStore,
};

#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::adapters::CheckpointFaultInjector;

// ---------------------------------------------------------------------------
// Factory plumbing
// ---------------------------------------------------------------------------

/// Per-test cleanup token returned alongside the constructed
/// `Arc<dyn CheckpointStore>`. The harness holds it to end-of-test scope so
/// adapter-specific resources (e.g. the LocalDisk `tempfile::TempDir`) are
/// released after the case body finishes.
///
/// `Send + Sync + 'static` so the guard can cross task boundaries inside a
/// multi-task case body (concurrency cases spawn writers — see cases #17 and
/// #18). The harness never inspects the guard's contents; this trait is
/// intentionally empty and exists solely to give the per-adapter wrappers a
/// place to attach drop-on-cleanup state.
pub trait CheckpointStoreGuard: Send + Sync + 'static {}

/// Zero-state guard for adapters that need no per-test cleanup (InMemory).
pub struct NoopGuard;

impl CheckpointStoreGuard for NoopGuard {}

/// Pair returned by every per-test factory closure.
pub type CheckpointStorePair = (Arc<dyn CheckpointStore>, Box<dyn CheckpointStoreGuard>);

/// Pair returned by every fault-injector-aware factory closure (cases #19,
/// #20, #22).
#[cfg(feature = "test-checkpoint-fault-injection")]
pub type CheckpointStoreInjectedPair = (Arc<dyn CheckpointStore>, Box<dyn CheckpointStoreGuard>);

// ---------------------------------------------------------------------------
// Payload builders
// ---------------------------------------------------------------------------

/// Build a small synthetic OMPF-shaped byte slice for a layer-checkpoint put.
///
/// The conformance harness does NOT parse the bytes — per SR-1.5 the adapter
/// treats every put as opaque — so the only requirement is that the buffer be
/// deterministic and round-trippable. We prefix it with the OMPF magic so a
/// regression that accidentally parses inside an adapter trips a recognizable
/// signal rather than a generic byte mismatch.
pub fn sample_layer_ompf(tag: u64) -> Bytes {
    let mut buf = Vec::with_capacity(16);
    // OMPF magic ("OMPF" little-endian).
    buf.extend_from_slice(&0x46504d4f_u32.to_le_bytes());
    // Discriminator so distinct calls produce distinct bytes.
    buf.extend_from_slice(&tag.to_le_bytes());
    Bytes::from(buf)
}

/// Synthetic ext-blob bytes keyed by the supplied tag.
pub fn sample_ext_bytes(tag: u64) -> Bytes {
    let mut buf = Vec::with_capacity(8);
    buf.extend_from_slice(&tag.to_le_bytes());
    Bytes::from(buf)
}

/// Synthetic session-checkpoint body bytes.
pub fn sample_session_body(tag: u64) -> Bytes {
    let mut buf = Vec::with_capacity(8);
    buf.extend_from_slice(&tag.to_le_bytes());
    Bytes::from(buf)
}

/// Synthetic counters-sidecar body bytes.
pub fn sample_counters_body(tag: u64) -> Bytes {
    let mut buf = Vec::with_capacity(8);
    buf.extend_from_slice(&tag.to_le_bytes());
    Bytes::from(buf)
}

/// Build a deterministic SHA-256-shaped hex hash string for cases that need a
/// content-address key. The harness does not actually hash anything — the
/// adapter treats the hex string as an opaque key.
pub fn sample_hash_hex(tag: u64) -> String {
    // 64 lowercase-hex chars matches the production content-address length so
    // any path-length-sensitive adapter (LocalDisk's `<hash>.blob`) is
    // exercised under realistic-sized filenames.
    format!("{:064x}", tag)
}

/// Distinctive non-zero `observed_commit_span` carried by every
/// [`sample_meta`] (b.vnw).
///
/// Non-zero on purpose. `0` is the field's "no measurement available"
/// sentinel and is also what `#[serde(default)]` produces for a blob that
/// never carried the field at all, so an adapter that silently DROPPED the
/// span on the way to storage would round-trip `0 == 0` and pass a
/// preservation assertion vacuously. A value no other meta field or case
/// argument uses makes the round-trip claim discriminating for all three
/// adapters at once.
///
/// The stakes for a drop are asymmetric: an adapter that loses the span
/// degrades every restore of a hot layer back to the configured margin — a
/// *smaller* guard band, i.e. the b.8y5 sequence-reuse direction — and does it
/// silently, because the restore still succeeds.
pub const SAMPLE_OBSERVED_COMMIT_SPAN: u64 = 314_159;

/// Build a `CheckpointMeta` with deterministic provenance fields. Tests that
/// care about specific fields override them after construction.
pub fn sample_meta(snap_seq: u64) -> CheckpointMeta {
    CheckpointMeta {
        creator_replica_id: "conformance-harness".to_string(),
        written_at_unix_ms: 1_700_000_000_000,
        snap_seq: Some(snap_seq),
        session_id: "sess-conformance".to_string(),
        layer_name: "layer-conformance".to_string(),
        created_by: "harness".to_string(),
        server_ts: 1_700_000_000_000,
        observed_commit_span: SAMPLE_OBSERVED_COMMIT_SPAN,
        sidecar_bytes: None,
        ..CheckpointMeta::default()
    }
}

/// Build a deterministic synthetic **tail-envelope** payload for a
/// [`CheckpointKind::Tail`] put.
///
/// **This is deliberately NOT a real tail envelope.** The harness has no
/// dependency on `src/checkpoint/tail_codec.rs` and must not grow one: SR-1.5
/// makes a tail's bytes as opaque to an adapter as a base's OMPF is, so
/// conformance can only assert byte-identical round-trip, never structure. A
/// real envelope here would test the codec, and would silently couple every
/// adapter case to a format change that adapters are contractually blind to.
///
/// The leading four bytes are a `"TAIL"` discriminator chosen to differ from
/// [`sample_layer_ompf`]'s OMPF magic (`0x46504d4f`), so an adapter regression
/// that routes a tail into the base/OMPF read path — or hands back the base
/// sharing the tail's sequence — trips a recognizable byte mismatch rather than
/// an anonymous one.
pub fn sample_tail_envelope(tag: u64) -> Bytes {
    let mut buf = Vec::with_capacity(16);
    // "TAIL" little-endian — distinct from the OMPF magic on the first byte.
    buf.extend_from_slice(&0x4c49_4154_u32.to_le_bytes());
    // Discriminator so distinct calls produce distinct bytes.
    buf.extend_from_slice(&tag.to_le_bytes());
    Bytes::from(buf)
}

/// Build a complete [`CheckpointKind::Tail`] meta over [`sample_meta`]'s
/// provenance fields.
///
/// `to_seq` — the envelope's inclusive upper commit bound — is what lands in
/// `snap_seq`, and is also what the caller passes as `put_layer_checkpoint`'s
/// `snap_seq` argument; the two kinds share one ordering number so they address
/// and sort identically (see `CheckpointHandle::snap_seq`). Both lineage fields
/// are `Some`: a `Tail` meta missing either one is a `Permanent` rejection, not
/// a shape an adapter repairs by guessing.
///
/// `sidecar_bytes` stays `None` (inherited from [`sample_meta`]) and MUST stay
/// `None` for every tail put this harness makes: a tail carrying `Some` is a
/// `Permanent` rejection too, which is a sibling Group J case's subject rather
/// than an incidental behaviour to trip over here.
pub fn sample_tail_meta(base_snap_seq: u64, from_seq: u64, to_seq: u64) -> CheckpointMeta {
    CheckpointMeta {
        kind: CheckpointKind::Tail,
        base_snap_seq: Some(base_snap_seq),
        from_seq: Some(from_seq),
        ..sample_meta(to_seq)
    }
}

/// Collapse a `list_layer_checkpoints` result into an order-preserving
/// `(sequence, kind)` vector.
///
/// The whole observable content of a mixed-kind listing is which generations it
/// reports and in what order, so comparing one `Vec` against an expected literal
/// states the ordering AND the membership in a single assertion — including the
/// pinned Base-before-Tail tie-break at an equal sequence, which is invisible to
/// a sequence-only projection. `adapter_key` is deliberately dropped: it is
/// adapter-specific, so no cross-adapter case can name its expected value.
pub fn kind_seq_pairs(handles: &[CheckpointHandle]) -> Vec<(u64, CheckpointKind)> {
    handles.iter().map(|h| (h.seq(), h.kind)).collect()
}

// ---------------------------------------------------------------------------
// NotImplemented-aware skip helper
// ---------------------------------------------------------------------------

/// Short-circuit the calling case body when the adapter responds with
/// [`CheckpointError::NotImplemented`]. Originally used by cases #2, #6, #15
/// while the Epic-1 session-checkpoint stubs surfaced `NotImplemented`; those
/// gates came out at Task `t2.fgj.7v.cq` once both adapters wired real impls.
/// Kept in place (alongside the [`skip_if_not_implemented!`] macro below) as
/// the established pattern if a future adapter ever needs to gate against a
/// new `NotImplemented`-shaped contract.
///
/// Emits a `tracing::warn!` so the skip is observable in the test log without
/// failing the run. Returns `true` when the case body should bail out.
#[allow(dead_code)] // retained for future NotImplemented-gated cases (see doc above).
pub fn skip_if_not_implemented<T>(
    case_name: &'static str,
    method: &'static str,
    result: &Result<T, CheckpointError>,
) -> bool {
    match result {
        Err(CheckpointError::NotImplemented { method: m }) => {
            tracing::warn!(
                case = case_name,
                method = method,
                adapter_method = *m,
                "conformance case skipped: adapter returned NotImplemented (Epic 2 path)"
            );
            true
        }
        _ => false,
    }
}

/// Convenience macro mirroring [`skip_if_not_implemented`] for the common
/// "call method, bail on NotImplemented, otherwise unwrap" pattern. Use the
/// helper directly when finer control is needed.
#[macro_export]
macro_rules! skip_if_not_implemented {
    ($case:expr, $method:expr, $result:expr $(,)?) => {{
        let result = $result;
        if $crate::checkpoint::conformance::skip_if_not_implemented($case, $method, &result) {
            return;
        }
        result
    }};
}

// ---------------------------------------------------------------------------
// Per-case stubs
// ---------------------------------------------------------------------------
//
// Each `pub async fn case_XX_<name><F, Fut>(factory: F)` follows the same
// generic shape:
//
//     F: Fn() -> Fut,
//     Fut: Future<Output = CheckpointStorePair>,
//
// Per-group Subtasks fill the `todo!()` bodies. The factory is taken by
// reference inside the body via `let (store, _guard) = factory().await;` —
// each invocation MUST produce a fresh, empty adapter (the harness never
// shares state across cases).

// ----- Group A: round-trip puts and gets (#1–#4) ----------------------------

/// Case #1 — `put_layer_checkpoint` followed by `get_latest_layer_checkpoint`
/// round-trips the OMPF bytes and meta verbatim.
pub async fn case_01_put_layer_then_get_latest_round_trips_bytes<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    let object_id: u64 = 42;
    let snap_seq: u64 = 7;
    let ompf = sample_layer_ompf(0xC0_FFEE);
    let meta = sample_meta(snap_seq);

    let handle = store
        .put_layer_checkpoint(object_id, snap_seq, ompf.clone(), meta.clone())
        .await
        .expect("put_layer_checkpoint must succeed on a fresh adapter");

    // The trait contract: the handle's object_id + snap_seq match the args;
    // adapter_key is adapter-specific opaque storage state (not asserted here).
    assert_eq!(
        handle.object_id, object_id,
        "handle.object_id must match the argument"
    );
    assert_eq!(
        handle.snap_seq, snap_seq,
        "handle.snap_seq must match the argument"
    );

    let blob = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest_layer_checkpoint must succeed")
        .expect("get_latest_layer_checkpoint must return Some after a put");

    assert_eq!(blob.handle.object_id, object_id);
    assert_eq!(blob.handle.snap_seq, snap_seq);
    assert_eq!(
        blob.bytes, ompf,
        "OMPF bytes must round-trip verbatim (SR-1.5: opaque payload)"
    );

    // Meta provenance fields must round-trip verbatim — the apply / recovery
    // path consumes them to rebuild `obj:hdr`, so any drop is observable.
    assert_eq!(blob.meta.creator_replica_id, meta.creator_replica_id);
    assert_eq!(blob.meta.written_at_unix_ms, meta.written_at_unix_ms);
    assert_eq!(blob.meta.snap_seq, meta.snap_seq);
    assert_eq!(blob.meta.session_id, meta.session_id);
    assert_eq!(blob.meta.layer_name, meta.layer_name);
    assert_eq!(blob.meta.created_by, meta.created_by);
    assert_eq!(blob.meta.server_ts, meta.server_ts);
    // b.vnw — `observed_commit_span` is the fifth meta field the apply path
    // consumes, and the only one that is a *number the restore does
    // arithmetic with* rather than identity it copies through. An adapter
    // that drops it does not corrupt anything visible: the blob still
    // restores, just with the configured margin instead of the measured band,
    // which is a smaller guard band and therefore the sequence-reuse
    // direction. Pinning it here makes the claim per-adapter (InMemory /
    // LocalDisk / StorageApi all run this case) rather than resting on the
    // one adapter whose serialization happens to be unit-tested.
    assert_eq!(
        blob.meta.observed_commit_span, meta.observed_commit_span,
        "observed_commit_span must round-trip verbatim ({}); a 0 here means \
         the adapter dropped the measured DR guard band and every restore of \
         a hot layer silently falls back to the configured margin",
        meta.observed_commit_span
    );
}

/// Case #2 — session-checkpoint round-trip (Epic 2; skipped on Epic 1 via
/// `NotImplemented`).
pub async fn case_02_put_session_then_get_session_round_trips_bytes<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    let session_id = "sess-conformance-case-02";
    let body = sample_session_body(0xABCD_0001);
    let meta = sample_meta(0);

    let put_handle = store
        .put_session_checkpoint(session_id, body.clone(), meta.clone())
        .await
        .expect("put_session_checkpoint must succeed on a fresh adapter");

    let blob = store
        .get_session_checkpoint(session_id)
        .await
        .expect("get_session_checkpoint must succeed")
        .expect("get_session_checkpoint must return Some after a put");

    assert_eq!(blob.handle, put_handle);
    assert_eq!(
        blob.bytes, body,
        "session-checkpoint bytes must round-trip verbatim"
    );
    // Meta-field persistence is adapter-specific. The trait contract on
    // `put_session_checkpoint` (see `src/checkpoint/store.rs` doc on the
    // method) covers error classification and bytes round-trip; it does NOT
    // mandate that the supplied `CheckpointMeta` is persisted alongside the
    // body. Per `src/checkpoint/adapters/storage_api.rs:825-835`, the
    // storage-api adapter intentionally skips meta persistence because the
    // SR-2.7 session-checkpoint payload carries its own provenance inside
    // the body envelope — duplicating fields in a sidecar would diverge
    // from the canonical source on `get`. Adapters that DO persist meta
    // (a future InMemory / LocalDisk wiring once Epic 2 lands) MUST round
    // trip `creator_replica_id` and `written_at_unix_ms`; adapters that
    // DON'T are within their rights to surface `CheckpointMeta::default()`.
    // The case body therefore asserts only the equality-when-non-default
    // shape: when the adapter surfaces a non-empty creator the round trip
    // must hold; when it surfaces the default placeholder the case
    // tolerates the divergence.
    if !blob.meta.creator_replica_id.is_empty() {
        assert_eq!(
            blob.meta.creator_replica_id, meta.creator_replica_id,
            "if the adapter persists creator_replica_id, the round trip must hold",
        );
    }
    if blob.meta.written_at_unix_ms != 0 {
        assert_eq!(
            blob.meta.written_at_unix_ms, meta.written_at_unix_ms,
            "if the adapter persists written_at_unix_ms, the round trip must hold",
        );
    }
}

/// Case #3 — `put_ext_blob` followed by `get_ext_blob` round-trips the bytes.
pub async fn case_03_put_ext_blob_then_get_ext_blob_round_trips_bytes<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    let hash_hex = sample_hash_hex(0xDEAD_BEEF_F00D_5EA1);
    let bytes = sample_ext_bytes(0xDEAD_BEEF_F00D_5EA1);

    store
        .put_ext_blob(&hash_hex, bytes.clone())
        .await
        .expect("put_ext_blob must succeed on a fresh adapter");

    let fetched = store
        .get_ext_blob(&hash_hex)
        .await
        .expect("get_ext_blob must succeed")
        .expect("get_ext_blob must return Some after a put");

    assert_eq!(
        fetched, bytes,
        "ext-blob bytes must round-trip verbatim under their content-address key"
    );
}

/// Case #4 — `put_counters_sidecar` followed by `get_counters_sidecar`
/// round-trips the bytes (last-writer-wins).
pub async fn case_04_put_counters_sidecar_then_get_round_trips_bytes<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    // First write — establishes the singleton sidecar.
    let body_v1 = sample_counters_body(0x0000_0001);
    store
        .put_counters_sidecar(body_v1.clone())
        .await
        .expect("put_counters_sidecar must succeed on a fresh adapter");

    let fetched_v1 = store
        .get_counters_sidecar()
        .await
        .expect("get_counters_sidecar must succeed")
        .expect("get_counters_sidecar must return Some after a put");
    assert_eq!(
        fetched_v1, body_v1,
        "counters sidecar bytes must round-trip verbatim"
    );

    // Second write to the same singleton key — last-writer-wins per the
    // trait contract on `put_counters_sidecar` ("There is exactly one
    // sidecar per checkpoint-store instance. The adapter stores `body` as
    // given, replacing any earlier sidecar; monotonicity is the caller's
    // rule.").
    let body_v2 = sample_counters_body(0x0000_0002);
    store
        .put_counters_sidecar(body_v2.clone())
        .await
        .expect("second put_counters_sidecar must succeed (last-writer-wins)");

    let fetched_v2 = store
        .get_counters_sidecar()
        .await
        .expect("get_counters_sidecar must succeed after rewrite")
        .expect("get_counters_sidecar must return Some after rewrite");
    assert_eq!(
        fetched_v2, body_v2,
        "counters sidecar must reflect the most recent put (last-writer-wins)"
    );
    assert_ne!(
        fetched_v2, body_v1,
        "last-writer-wins must replace the prior bytes, not merge them"
    );
}

// ----- Group B: absent-key behaviour (#5–#8) --------------------------------

/// Case #5 — `get_latest_layer_checkpoint` on an unknown `object_id` returns
/// `Ok(None)`, NOT `NotFound`.
pub async fn case_05_get_latest_layer_checkpoint_unknown_object_returns_none<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    // No `put_layer_checkpoint` call — the adapter is fresh. The trait
    // contract pins this as `Ok(None)`, not `Err(NotFound)`: "MUST NOT return
    // `NotFound` for an absent object; that is an `Ok(None)`."
    let unknown_object_id: u64 = 0xFFFF_FFFF_FFFF_FFFE;
    let result = store.get_latest_layer_checkpoint(unknown_object_id).await;

    match result {
        Ok(None) => {} // contract-satisfying outcome
        Ok(Some(blob)) => panic!(
            "expected Ok(None) for unknown object_id, got Some(handle={:?})",
            blob.handle
        ),
        Err(CheckpointError::NotFound) => panic!(
            "get_latest_layer_checkpoint must return Ok(None) for an absent object, NOT \
             CheckpointError::NotFound (trait-level contract; the absent-object path stays infallible)"
        ),
        Err(err) => panic!(
            "expected Ok(None) for unknown object_id; got unexpected error: {:?}",
            err
        ),
    }
}

/// Case #6 — `get_session_checkpoint` on an unknown `session_id` returns
/// `Ok(None)` (Epic 2; skipped on Epic 1 via `NotImplemented`).
pub async fn case_06_get_session_checkpoint_unknown_session_returns_none<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    let unknown_session_id = "sess-conformance-case-06-never-written";
    let value = store.get_session_checkpoint(unknown_session_id).await;

    // The absent-session path mirrors the absent-layer path: `Ok(None)`,
    // never `NotFound`.
    match value {
        Ok(None) => {} // contract-satisfying outcome
        Ok(Some(blob)) => panic!(
            "expected Ok(None) for unknown session_id, got Some(handle={:?})",
            blob.handle
        ),
        Err(CheckpointError::NotFound) => panic!(
            "get_session_checkpoint must return Ok(None) for an absent session, NOT \
             CheckpointError::NotFound (trait-level contract)"
        ),
        Err(err) => panic!(
            "expected Ok(None) for unknown session_id; got unexpected error: {:?}",
            err
        ),
    }
}

/// Case #7 — `get_ext_blob` on an unknown hash returns `Ok(None)`.
pub async fn case_07_get_ext_blob_unknown_hash_returns_none<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    // No `put_ext_blob` call — the adapter has never seen this hash.
    let unknown_hash = sample_hash_hex(0xDEAD_DEAD_DEAD_DEAD);
    let result = store.get_ext_blob(&unknown_hash).await;

    match result {
        Ok(None) => {} // contract-satisfying outcome
        Ok(Some(bytes)) => panic!(
            "expected Ok(None) for unknown hash, got Some({} bytes)",
            bytes.len()
        ),
        Err(CheckpointError::NotFound) => panic!(
            "get_ext_blob must return Ok(None) for an absent hash, NOT \
             CheckpointError::NotFound (trait-level contract)"
        ),
        Err(err) => panic!(
            "expected Ok(None) for unknown hash; got unexpected error: {:?}",
            err
        ),
    }
}

/// Case #8 — `get_counters_sidecar` on a fresh adapter (no put yet) returns
/// `Ok(None)`.
pub async fn case_08_get_counters_sidecar_fresh_adapter_returns_none<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    // The factory contract requires a fresh, empty adapter — so the singleton
    // sidecar has never been written. The pre-first-checkpoint recovery path
    // depends on `Ok(None)` here.
    let result = store.get_counters_sidecar().await;

    match result {
        Ok(None) => {} // contract-satisfying outcome
        Ok(Some(bytes)) => panic!(
            "expected Ok(None) for fresh-adapter counters sidecar, got Some({} bytes)",
            bytes.len()
        ),
        Err(CheckpointError::NotFound) => panic!(
            "get_counters_sidecar must return Ok(None) when no sidecar has been written, NOT \
             CheckpointError::NotFound (trait-level contract)"
        ),
        Err(err) => panic!(
            "expected Ok(None) for fresh-adapter counters sidecar; got unexpected error: {:?}",
            err
        ),
    }
}

// ----- Group C: idempotent puts (#9–#10) ------------------------------------

/// Case #9 — two `put_layer_checkpoint`s on the same `(object_id, snap_seq)`
/// collapse to a single generation (last-writer-wins on the bytes).
///
/// Trait-surface assertion: after two puts on the identical
/// `(object_id, snap_seq)` tuple, `list_layer_checkpoints` returns exactly ONE
/// handle and `get_latest_layer_checkpoint` round-trips the SECOND writer's
/// bytes. Per-adapter unit tests under `src/checkpoint/adapters/*` cover the
/// stronger storage-cost properties (inner `BTreeMap::len == 1` for InMemory,
/// on-disk file-count for LocalDisk); this case pins the trait-observable
/// surface that holds across every adapter.
pub async fn case_09_put_layer_idempotent_single_generation<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;
    let object_id: u64 = 9_001;
    let snap_seq: u64 = 42;

    // First put.
    let first_bytes = sample_layer_ompf(1);
    let h1 = store
        .put_layer_checkpoint(
            object_id,
            snap_seq,
            first_bytes.clone(),
            sample_meta(snap_seq),
        )
        .await
        .expect("first put_layer_checkpoint succeeds");
    assert_eq!(h1.object_id, object_id);
    assert_eq!(h1.snap_seq, snap_seq);

    // Second put on the same (object_id, snap_seq) with distinct bytes.
    let second_bytes = sample_layer_ompf(2);
    assert_ne!(
        first_bytes, second_bytes,
        "test precondition: the two puts must carry distinct bytes so last-writer-wins is observable",
    );
    let h2 = store
        .put_layer_checkpoint(
            object_id,
            snap_seq,
            second_bytes.clone(),
            sample_meta(snap_seq),
        )
        .await
        .expect("second put_layer_checkpoint on the same key succeeds (idempotent on slot)");
    assert_eq!(h2.object_id, object_id);
    assert_eq!(h2.snap_seq, snap_seq);

    // Exactly one generation remains — duplicate (object_id, snap_seq) puts
    // collapse to a single slot per the SR-1.3 #9 contract.
    let handles = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints succeeds");
    assert_eq!(
        handles.len(),
        1,
        "duplicate (object_id, snap_seq) puts must collapse to exactly one generation; got {handles:?}",
    );
    assert_eq!(handles[0].object_id, object_id);
    assert_eq!(handles[0].snap_seq, snap_seq);

    // Last-writer-wins on the bytes — the round-trip must return the SECOND
    // payload, not the first.
    let latest = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest_layer_checkpoint succeeds")
        .expect("at least one checkpoint exists after two puts");
    assert_eq!(latest.handle.snap_seq, snap_seq);
    assert_eq!(
        latest.bytes, second_bytes,
        "last-writer-wins must return the second put's bytes",
    );
}

/// Case #10 — two `put_ext_blob`s under the same `hash_hex` do not duplicate
/// storage (content-addressed idempotency).
///
/// Trait-surface assertion: re-putting under the same `hash_hex` with
/// identical bytes returns `Ok(())` and `get_ext_blob` returns the bytes
/// verbatim. The "no duplicate storage" / identity-preserving property is
/// adapter-internal — per the trait doc on `put_ext_blob`, "the adapter MAY
/// skip the write entirely" — and is pinned at the unit-test level
/// (`put_ext_blob_same_hash_twice_does_not_replace_stored_bytes` for InMemory
/// pointer identity; LocalDisk uses a metadata-then-skip path). This case
/// verifies the cross-adapter observable contract.
pub async fn case_10_put_ext_blob_idempotent_no_duplicate_storage<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;
    let hash = sample_hash_hex(0x00C0_FFEE_DEAD_BEEF);
    let payload = sample_ext_bytes(0x00C0_FFEE_DEAD_BEEF);

    // Fresh adapter — get must report None before any put.
    let pre = store
        .get_ext_blob(&hash)
        .await
        .expect("get_ext_blob on a fresh adapter must be Ok, not Err");
    assert!(
        pre.is_none(),
        "get_ext_blob on an unknown hash must be Ok(None); got {pre:?}",
    );

    // First put.
    store
        .put_ext_blob(&hash, payload.clone())
        .await
        .expect("first put_ext_blob succeeds");

    // Second put under the same hash with byte-identical content — MUST
    // succeed as a no-op per the trait contract (content-addressed; re-put
    // under a duplicate hash implies identical bytes).
    store
        .put_ext_blob(&hash, payload.clone())
        .await
        .expect("second put_ext_blob with identical bytes succeeds (idempotent on hash)");

    // The stored bytes round-trip verbatim — the second put has not corrupted
    // the value behind the hash.
    let got = store
        .get_ext_blob(&hash)
        .await
        .expect("get_ext_blob succeeds")
        .expect("hash present after put");
    assert_eq!(
        got, payload,
        "get_ext_blob must round-trip the content-addressed bytes verbatim",
    );
}

// ----- Group D: list ordering (#11–#12) -------------------------------------

/// Case #11 — `list_layer_checkpoints` returns generations newest-first
/// (descending `snap_seq`), independent of insertion order.
///
/// Inserts three snap_seqs in scrambled order (200, 100, 300) and asserts the
/// resulting list is `[300, 200, 100]`. Calls `list_layer_checkpoints` a
/// second time and asserts ordering is stable across re-calls (no
/// hidden round-robin or randomization on the read path).
pub async fn case_11_list_layer_checkpoints_newest_first_ordering<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;
    let object_id: u64 = 11_011;

    // Insert in a deliberately scrambled order. The trait's ordering contract
    // depends on `snap_seq` (higher = newer), NOT on insertion sequence.
    let scrambled_insertion_order: [u64; 3] = [200, 100, 300];
    for snap_seq in scrambled_insertion_order {
        store
            .put_layer_checkpoint(
                object_id,
                snap_seq,
                sample_layer_ompf(snap_seq),
                sample_meta(snap_seq),
            )
            .await
            .unwrap_or_else(|err| panic!("put snap_seq={snap_seq} succeeds, got {err:?}"));
    }

    let handles = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints succeeds");
    let seqs: Vec<u64> = handles.iter().map(|h| h.snap_seq).collect();
    assert_eq!(
        seqs,
        vec![300, 200, 100],
        "list_layer_checkpoints must return generations newest-first by descending snap_seq",
    );

    // Companion: `get_latest_layer_checkpoint` MUST agree with the head of the
    // list per the trait's `list_layer_checkpoints` doc.
    let latest = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest_layer_checkpoint succeeds")
        .expect("at least one checkpoint exists");
    assert_eq!(
        latest.handle.snap_seq, 300,
        "get_latest_layer_checkpoint must agree with list_layer_checkpoints.first()",
    );

    // Stability: ordering MUST NOT change between successive read-only
    // `list_layer_checkpoints` calls.
    let handles_again = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("repeat list_layer_checkpoints succeeds");
    let seqs_again: Vec<u64> = handles_again.iter().map(|h| h.snap_seq).collect();
    assert_eq!(
        seqs_again, seqs,
        "list_layer_checkpoints ordering must be stable across re-calls",
    );
}

/// Case #12 — `list_layer_checkpoints` on an unknown `object_id` returns an
/// empty vector, NOT an error.
///
/// Trait contract: "Returns an empty vector when no checkpoints exist;
/// absence is not an error." Pins that the absent-object branch stays
/// infallible across every adapter so the orphan-sweep enumeration path
/// cannot trip on a synthetic `NotFound`.
pub async fn case_12_list_layer_checkpoints_unknown_object_returns_empty<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    // Pick an object_id far from any value the harness or its helpers stamp
    // so a future test that share-stamps the same `object_id` cannot
    // accidentally populate this slot.
    let unknown_object_id: u64 = 0xFFFF_FFFF_FFFF_FFFD;
    let result = store.list_layer_checkpoints(unknown_object_id).await;

    match result {
        Ok(handles) => assert!(
            handles.is_empty(),
            "list_layer_checkpoints on a fresh, unknown object_id must return an empty vec; got {handles:?}",
        ),
        Err(CheckpointError::NotFound) => panic!(
            "list_layer_checkpoints must return Ok(empty) for an unknown object_id, NOT \
             CheckpointError::NotFound (trait-level contract)"
        ),
        Err(err) => panic!(
            "expected Ok(empty) for unknown object_id; got unexpected error: {err:?}",
        ),
    }
}

// ----- Group E: delete semantics (#13–#16) ----------------------------------

/// Case #13 — after deleting the newest of N generations,
/// `get_latest_layer_checkpoint` returns the next-newest (or `None` when N=1).
///
/// Walks the retention-pruner sequence end-to-end: start with three
/// generations, delete the newest and assert get_latest reveals the
/// second-newest, delete again and assert the oldest is now latest, delete
/// the last and assert `Ok(None)`. Pins the contract that a delete updates
/// the list-head atomically (no stale-latest dangling reference).
pub async fn case_13_delete_layer_then_get_returns_next_newest_or_none<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;
    let object_id: u64 = 13_013;

    // Three generations — bytes distinct per snap_seq so we can identify
    // each generation on read-back.
    let seqs: [u64; 3] = [10, 20, 30];
    let mut handles: Vec<live_edit::checkpoint::CheckpointHandle> = Vec::new();
    for seq in seqs {
        let h = store
            .put_layer_checkpoint(object_id, seq, sample_layer_ompf(seq), sample_meta(seq))
            .await
            .unwrap_or_else(|err| panic!("put snap_seq={seq} succeeds, got {err:?}"));
        handles.push(h);
    }

    // Sanity: latest is the newest (snap_seq=30).
    let latest = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest succeeds")
        .expect("at least one checkpoint after three puts");
    assert_eq!(latest.handle.snap_seq, 30);
    assert_eq!(latest.bytes, sample_layer_ompf(30));

    // Delete the newest (snap_seq=30). The next get_latest must surface
    // snap_seq=20.
    let newest_handle = handles
        .iter()
        .find(|h| h.snap_seq == 30)
        .expect("handle for snap_seq=30 was recorded")
        .clone();
    store
        .delete_layer_checkpoint(newest_handle)
        .await
        .expect("delete newest succeeds");
    let latest_after_first_delete = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest after delete succeeds")
        .expect("two checkpoints still remain after deleting the newest");
    assert_eq!(
        latest_after_first_delete.handle.snap_seq, 20,
        "after deleting snap_seq=30, get_latest must return snap_seq=20 (the next-newest)",
    );
    assert_eq!(latest_after_first_delete.bytes, sample_layer_ompf(20));

    // Delete the middle generation (snap_seq=20). Now snap_seq=10 must be
    // the latest.
    let mid_handle = handles
        .iter()
        .find(|h| h.snap_seq == 20)
        .expect("handle for snap_seq=20 was recorded")
        .clone();
    store
        .delete_layer_checkpoint(mid_handle)
        .await
        .expect("delete middle succeeds");
    let latest_after_second_delete = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest after second delete succeeds")
        .expect("one checkpoint still remains after deleting the newest two");
    assert_eq!(
        latest_after_second_delete.handle.snap_seq, 10,
        "after deleting snap_seqs 30 and 20, get_latest must return snap_seq=10",
    );
    assert_eq!(latest_after_second_delete.bytes, sample_layer_ompf(10));

    // Delete the last remaining generation. Now get_latest must return
    // `Ok(None)` — N=0 after this delete.
    let oldest_handle = handles
        .iter()
        .find(|h| h.snap_seq == 10)
        .expect("handle for snap_seq=10 was recorded")
        .clone();
    store
        .delete_layer_checkpoint(oldest_handle)
        .await
        .expect("delete oldest succeeds");
    let after_all_deletes = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest after deleting all generations succeeds");
    assert!(
        after_all_deletes.is_none(),
        "after deleting every generation, get_latest must return Ok(None); got {after_all_deletes:?}",
    );

    // And the list view must agree — empty after deleting everything.
    let handles_after = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints succeeds after all deletes");
    assert!(
        handles_after.is_empty(),
        "list_layer_checkpoints must be empty after deleting every generation; got {handles_after:?}",
    );
}

/// Case #14 — `delete_layer_checkpoint` on a missing handle is `Ok(())`,
/// NOT `NotFound` (retention-prune idempotency contract).
///
/// Two flavours of "missing":
///
/// 1. A fabricated handle whose `object_id` has never been written.
/// 2. A real `object_id` that exists but with a synthetic `snap_seq` that
///    does not.
///
/// Both must return `Ok(())` per the trait doc on `delete_layer_checkpoint`:
/// "Idempotent on missing handles — if the handle has already been pruned (or
/// never existed), returns `Ok(())`, NEVER [`CheckpointError::NotFound`]."
pub async fn case_14_delete_layer_nonexistent_handle_is_idempotent<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    // Flavour 1: fabricate a handle for an object_id that has never been
    // written. The adapter_key string is intentionally adapter-agnostic
    // — adapters that key off `(object_id, snap_seq)` (InMemory, LocalDisk)
    // ignore it on delete.
    let phantom_handle = live_edit::checkpoint::CheckpointHandle::base(
        0xFFFF_FFFF_FFFF_FFFC,
        1,
        "phantom-conformance-case-14".to_string(),
    );
    match store.delete_layer_checkpoint(phantom_handle).await {
        Ok(()) => {} // contract-satisfying outcome
        Err(CheckpointError::NotFound) => panic!(
            "delete_layer_checkpoint on a fabricated handle must return Ok(()), NOT \
             CheckpointError::NotFound (retention-prune idempotency contract)"
        ),
        Err(err) => {
            panic!("expected Ok(()) for delete on a missing handle; got unexpected error: {err:?}",)
        }
    }

    // Flavour 2: put a real generation, then delete a synthetic
    // (real_object_id, missing_snap_seq) handle. The real generation MUST
    // survive — the spurious delete must not collateral-damage existing
    // entries.
    let object_id: u64 = 14_014;
    let real_seq: u64 = 7;
    let real_bytes = sample_layer_ompf(real_seq);
    store
        .put_layer_checkpoint(
            object_id,
            real_seq,
            real_bytes.clone(),
            sample_meta(real_seq),
        )
        .await
        .expect("put real generation succeeds");

    let synthetic_handle = live_edit::checkpoint::CheckpointHandle::base(
        object_id,
        999_999,
        format!("phantom:{}:999999", object_id),
    );
    store
        .delete_layer_checkpoint(synthetic_handle)
        .await
        .expect("delete on a missing snap_seq within a real object must be Ok, not NotFound");

    // The real entry MUST still exist after the spurious delete.
    let latest = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest after spurious delete succeeds")
        .expect("real generation must survive a spurious delete");
    assert_eq!(latest.handle.snap_seq, real_seq);
    assert_eq!(
        latest.bytes, real_bytes,
        "spurious delete must not corrupt unrelated generations",
    );
}

/// Case #15 — `delete_session_checkpoint` then `get_session_checkpoint`
/// returns `Ok(None)`.
///
/// The put-delete-get sequence MUST collapse to `Ok(None)` per the trait
/// contract — the post-delete read mirrors the absent-key path, never
/// `NotFound`.
pub async fn case_15_delete_session_then_get_returns_none<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;
    let session_id = "sess-conformance-case-15";

    // Seed a session checkpoint so the subsequent delete has something to
    // remove.
    let _put_handle = store
        .put_session_checkpoint(session_id, sample_session_body(15), sample_meta(0))
        .await
        .expect("put_session_checkpoint must succeed on a fresh adapter");

    store
        .delete_session_checkpoint(session_id)
        .await
        .expect("delete_session_checkpoint must succeed for a previously-written session");

    let value = store.get_session_checkpoint(session_id).await;

    // The post-delete read MUST be `Ok(None)`, not `NotFound`.
    match value {
        Ok(None) => {} // contract-satisfying outcome
        Ok(Some(blob)) => panic!(
            "expected Ok(None) after delete_session_checkpoint; got Some(handle={:?})",
            blob.handle
        ),
        Err(CheckpointError::NotFound) => panic!(
            "get_session_checkpoint after delete must return Ok(None), NOT \
             CheckpointError::NotFound (trait-level contract)"
        ),
        Err(err) => panic!(
            "expected Ok(None) after delete_session_checkpoint; got unexpected error: {err:?}",
        ),
    }
}

/// Case #16 — `delete_ext_blob` then `get_ext_blob` returns `Ok(None)`.
///
/// Pins the orphan-sweep contract: a hash that was once stored MUST surface
/// as `Ok(None)` (NOT `NotFound`) once `delete_ext_blob` has run. Also
/// confirms that the delete is idempotent on the now-absent hash — a second
/// delete on the same hash MUST return `Ok(())`, matching the trait doc on
/// `delete_ext_blob`: "Idempotent on missing hashes — deleting an absent blob
/// returns `Ok(())`, never [`CheckpointError::NotFound`]. The orphan-sweep
/// loop depends on this so multiple sweepers do not race on identical
/// hashes."
pub async fn case_16_delete_ext_blob_then_get_returns_none<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;
    let hash = sample_hash_hex(0x0BAD_F00D_F0E5_0123);
    let payload = sample_ext_bytes(0x0BAD_F00D_F0E5_0123);

    // Put then read back to confirm the round-trip baseline.
    store
        .put_ext_blob(&hash, payload.clone())
        .await
        .expect("put_ext_blob succeeds");
    let pre_delete = store
        .get_ext_blob(&hash)
        .await
        .expect("get_ext_blob after put succeeds")
        .expect("hash present after put");
    assert_eq!(pre_delete, payload);

    // Delete the hash.
    store
        .delete_ext_blob(&hash)
        .await
        .expect("delete_ext_blob succeeds");

    // Post-delete get MUST be `Ok(None)`, not `NotFound`.
    let post_delete = store
        .get_ext_blob(&hash)
        .await
        .expect("get_ext_blob after delete must be Ok, not Err");
    assert!(
        post_delete.is_none(),
        "get_ext_blob after delete must return Ok(None); got {post_delete:?}",
    );

    // Second delete on the now-absent hash MUST be `Ok(())` — orphan-sweep
    // idempotency contract.
    match store.delete_ext_blob(&hash).await {
        Ok(()) => {} // contract-satisfying outcome
        Err(CheckpointError::NotFound) => panic!(
            "delete_ext_blob on an already-deleted hash must return Ok(()), NOT \
             CheckpointError::NotFound (orphan-sweep idempotency contract)"
        ),
        Err(err) => {
            panic!("expected Ok(()) for delete on a missing hash; got unexpected error: {err:?}",)
        }
    }

    // And the second post-delete get is still `Ok(None)`.
    let post_second_delete = store
        .get_ext_blob(&hash)
        .await
        .expect("get_ext_blob after second delete must be Ok, not Err");
    assert!(post_second_delete.is_none());
}

// ----- Group F: concurrency safety (#17–#18) --------------------------------

/// Case #17 — concurrent puts and deletes on unrelated keys must not
/// interfere (no cross-key state corruption).
///
/// **Design.** N=8 tasks fan out from a `tokio::sync::Barrier` so every task
/// hits the adapter at the same instant; half put on a unique `object_id`,
/// half delete a previously-seeded `object_id`. The same pattern repeats for
/// ext-blobs (unique `hash_hex`). After the storm we assert that every
/// "putter" `object_id` (and ext-blob hash) is readable, and every
/// "deleter" `object_id` (and ext-blob hash) is absent. The pre-seed step
/// is what gives deletes something to remove; without it a concurrent
/// delete on an absent key is trivially `Ok(())` and the test would not
/// exercise actual cross-key interference.
///
/// The whole body is wrapped in `tokio::time::timeout(BUDGET)` so a hung
/// adapter fails rather than wedges CI. Each spawned task is joined inside
/// its own `tokio::time::timeout` so the outer assert reflects per-task
/// completion, not a futures-pending stall.
pub async fn case_17_concurrent_put_delete_unrelated_keys_no_interference<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    use std::time::Duration;
    use tokio::sync::Barrier;

    /// Per-test wall-clock cap. The adapters under test are in-memory or
    /// tempdir-backed; 30 s is two orders of magnitude over the expected
    /// finish so any hang is unambiguous.
    const BUDGET: Duration = Duration::from_secs(30);
    /// Per-side worker count (half put, half delete). N=8 puts + N=8
    /// deletes = 16 concurrent tasks, well over the typical adapter's
    /// internal mutex contention floor.
    const N: usize = 8;

    let body = async move {
        let (store, _guard) = factory().await;

        // Pre-seed the "to-be-deleted" object_ids and ext-blob hashes so
        // the concurrent delete-side actually has something to remove.
        // We separate the put-side and delete-side ID ranges to keep the
        // "unrelated keys" invariant explicit: put-side oids are
        // [100, 100+N), delete-side oids are [200, 200+N).
        let put_oids: Vec<u64> = (0..N as u64).map(|i| 100 + i).collect();
        let del_oids: Vec<u64> = (0..N as u64).map(|i| 200 + i).collect();
        let put_hashes: Vec<String> = (0..N as u64).map(|i| sample_hash_hex(100 + i)).collect();
        let del_hashes: Vec<String> = (0..N as u64).map(|i| sample_hash_hex(200 + i)).collect();

        // Seed the deletion targets — every del_oids[i] gets one layer
        // checkpoint, every del_hashes[i] gets one ext-blob.
        for (i, oid) in del_oids.iter().enumerate() {
            store
                .put_layer_checkpoint(*oid, 1, sample_layer_ompf(200 + i as u64), sample_meta(1))
                .await
                .expect("seed put_layer_checkpoint for delete target");
        }
        for (i, hash) in del_hashes.iter().enumerate() {
            store
                .put_ext_blob(hash, sample_ext_bytes(200 + i as u64))
                .await
                .expect("seed put_ext_blob for delete target");
        }

        // Snapshot the seeded delete-side handles so the concurrent delete
        // tasks can hand them to `delete_layer_checkpoint`.
        let mut del_handles: Vec<CheckpointHandle> = Vec::with_capacity(N);
        for oid in &del_oids {
            let handle = store
                .list_layer_checkpoints(*oid)
                .await
                .expect("list seeded layer checkpoints")
                .into_iter()
                .next()
                .expect("seeded layer checkpoint must be listable");
            del_handles.push(handle);
        }

        // 2 sides × N tasks each for layers, plus 2 sides × N for ext-blobs.
        // Every task waits on the same barrier so the storm starts at once.
        let barrier = Arc::new(Barrier::new(N * 4));
        let mut joins = Vec::with_capacity(N * 4);

        // --- Layer-checkpoint put side --------------------------------------
        for (i, oid) in put_oids.iter().copied().enumerate() {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            let bytes = sample_layer_ompf(100 + i as u64);
            joins.push(tokio::spawn(async move {
                barrier.wait().await;
                store
                    .put_layer_checkpoint(oid, 1, bytes, sample_meta(1))
                    .await
                    .expect("concurrent put_layer_checkpoint must succeed on its own oid");
            }));
        }

        // --- Layer-checkpoint delete side -----------------------------------
        for handle in del_handles.iter().cloned() {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            joins.push(tokio::spawn(async move {
                barrier.wait().await;
                store
                    .delete_layer_checkpoint(handle)
                    .await
                    .expect("concurrent delete_layer_checkpoint must succeed");
            }));
        }

        // --- Ext-blob put side ----------------------------------------------
        for (i, hash) in put_hashes.iter().cloned().enumerate() {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            let bytes = sample_ext_bytes(100 + i as u64);
            joins.push(tokio::spawn(async move {
                barrier.wait().await;
                store
                    .put_ext_blob(&hash, bytes)
                    .await
                    .expect("concurrent put_ext_blob must succeed on its own hash");
            }));
        }

        // --- Ext-blob delete side -------------------------------------------
        for hash in &del_hashes {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            // Hand the spawned task an owned `String` so the future is
            // `'static`; `del_hashes` is reused after the loop for the
            // post-storm "ext hash absent" assertion.
            let hash = hash.clone();
            joins.push(tokio::spawn(async move {
                barrier.wait().await;
                store
                    .delete_ext_blob(&hash)
                    .await
                    .expect("concurrent delete_ext_blob must succeed");
            }));
        }

        // Bounded join — every spawned task wrapped in its own
        // `tokio::time::timeout` so a hung adapter fails on the offending
        // task rather than wedging the whole gather.
        for join in joins {
            tokio::time::timeout(BUDGET, join)
                .await
                .expect("per-task timeout: spawned worker hung")
                .expect("spawned worker panicked");
        }

        // --- Post-storm invariants -----------------------------------------
        // Every put-side oid landed exactly one checkpoint and is readable.
        for (i, oid) in put_oids.iter().copied().enumerate() {
            let blob = store
                .get_latest_layer_checkpoint(oid)
                .await
                .expect("get_latest after concurrent put must not error")
                .unwrap_or_else(|| panic!("put-side oid {} must be readable after storm", oid));
            assert_eq!(
                blob.bytes,
                sample_layer_ompf(100 + i as u64),
                "put-side oid {} bytes must round-trip verbatim",
                oid
            );
            assert_eq!(blob.handle.object_id, oid);
            assert_eq!(blob.handle.snap_seq, 1);
        }
        // Every delete-side oid is gone.
        for oid in &del_oids {
            assert!(
                store
                    .get_latest_layer_checkpoint(*oid)
                    .await
                    .expect("get_latest after concurrent delete must not error")
                    .is_none(),
                "delete-side oid {} must be absent after storm",
                oid
            );
            assert!(
                store
                    .list_layer_checkpoints(*oid)
                    .await
                    .expect("list after concurrent delete must not error")
                    .is_empty(),
                "delete-side oid {} must have empty list after storm",
                oid
            );
        }
        // Every put-side ext-blob is present with the expected bytes.
        for (i, hash) in put_hashes.iter().enumerate() {
            let got = store
                .get_ext_blob(hash)
                .await
                .expect("get_ext_blob after concurrent put must not error")
                .unwrap_or_else(|| panic!("put-side ext hash {} must be readable", hash));
            assert_eq!(
                got,
                sample_ext_bytes(100 + i as u64),
                "put-side ext hash {} bytes must round-trip verbatim",
                hash
            );
        }
        // Every delete-side ext-blob is gone.
        for hash in &del_hashes {
            assert!(
                store
                    .get_ext_blob(hash)
                    .await
                    .expect("get_ext_blob after concurrent delete must not error")
                    .is_none(),
                "delete-side ext hash {} must be absent after storm",
                hash
            );
        }
    };

    tokio::time::timeout(BUDGET, body)
        .await
        .expect("case_17 body timed out — adapter likely hung under concurrency");
}

/// Case #18 — concurrent `put`/`get` on the same key never observes a
/// torn read (the get either sees the full prior write or the full new write,
/// never a partial blend).
///
/// **Design.** One writer task loops `put_layer_checkpoint` on a single
/// `(object_id, snap_seq=1)` slot, alternating between two distinct OMPF
/// payloads `A` and `B`. M=4 reader tasks loop `get_latest_layer_checkpoint`
/// on the same `object_id`. Because the test pre-seeds the slot with
/// payload A before the barrier release, every reader observation MUST be
/// `Ok(Some(blob))` where `blob.bytes` is byte-exact `A` or byte-exact `B`
/// — never a partial blend, and never `Ok(None)`. Both adapters under
/// test (InMemory, LocalDisk) hold the slot continuously across the
/// put-then-put overwrite — they swap rather than briefly clear — so an
/// observed `None` here would itself be a torn-read regression and trips
/// an explicit panic.
///
/// A `tokio::sync::Barrier` synchronizes the writer + reader start so
/// they actually race rather than serialize. Both the writer and the
/// readers run for a bounded `ITERATIONS` count rather than wall-clock —
/// keeps the test deterministic and bounded. The whole body is wrapped
/// in `tokio::time::timeout` so a hung adapter fails rather than wedges
/// CI.
pub async fn case_18_concurrent_put_get_same_key_no_torn_read<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    use std::time::Duration;
    use tokio::sync::Barrier;

    /// Wall-clock cap. Per-iteration adapter work is on the order of
    /// microseconds (in-memory) to milliseconds (tempdir LocalDisk); 60 s
    /// leaves room for the LocalDisk path under CI contention without
    /// being lenient enough to mask a real hang.
    const BUDGET: Duration = Duration::from_secs(60);
    /// Writer iterations. Higher values widen the race window; 256 is
    /// enough to interleave many writer/reader transitions while keeping
    /// the LocalDisk variant under the budget.
    const ITERATIONS: usize = 256;
    /// Reader fan-out. Four readers + one writer means every reader has
    /// a distinct cadence so we exercise more interleavings than a
    /// single-reader test would.
    const READERS: usize = 4;

    let body = async move {
        let (store, _guard) = factory().await;
        let object_id: u64 = 4242;
        let payload_a = sample_layer_ompf(0xAAAA_AAAA);
        let payload_b = sample_layer_ompf(0xBBBB_BBBB);
        assert_ne!(
            payload_a, payload_b,
            "harness invariant: payload A and B must differ so a torn read is observable"
        );

        // Pre-seed payload A so the readers observe a non-`None` blob
        // immediately after the barrier. Without this, every reader's
        // first iteration would trivially see `None` and the assert
        // would have to allow `None` for the entire run.
        store
            .put_layer_checkpoint(object_id, 1, payload_a.clone(), sample_meta(1))
            .await
            .expect("seed put_layer_checkpoint");

        // 1 writer + READERS = total parties on the barrier.
        let barrier = Arc::new(Barrier::new(1 + READERS));
        let mut joins = Vec::with_capacity(1 + READERS);

        // --- Writer ---------------------------------------------------------
        {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            let payload_a = payload_a.clone();
            let payload_b = payload_b.clone();
            joins.push(tokio::spawn(async move {
                barrier.wait().await;
                for i in 0..ITERATIONS {
                    // Alternate A/B so any torn read would surface as a
                    // mismatch against BOTH payloads.
                    let payload = if i % 2 == 0 {
                        payload_a.clone()
                    } else {
                        payload_b.clone()
                    };
                    store
                        .put_layer_checkpoint(object_id, 1, payload, sample_meta(1))
                        .await
                        .expect("writer put_layer_checkpoint must succeed");
                }
            }));
        }

        // --- Readers --------------------------------------------------------
        for _ in 0..READERS {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            let payload_a = payload_a.clone();
            let payload_b = payload_b.clone();
            joins.push(tokio::spawn(async move {
                barrier.wait().await;
                for _ in 0..ITERATIONS {
                    let observed = store
                        .get_latest_layer_checkpoint(object_id)
                        .await
                        .expect("reader get_latest_layer_checkpoint must not error");
                    match observed {
                        // After the pre-seed put of payload A, the
                        // `(object_id, snap_seq=1)` slot is held
                        // continuously across every subsequent overwrite:
                        // InMemory and LocalDisk both swap the slot
                        // atomically rather than briefly clearing it.
                        // Observing `Ok(None)` here would indicate a
                        // regression in which the adapter momentarily
                        // exposed an empty state mid-write — a torn-read
                        // bug just as severe as a partial-blend payload —
                        // so we panic rather than tolerate it.
                        None => panic!(
                            "torn-read regression: observed Ok(None) after pre-seed; \
                             adapter must hold A or B continuously across overwrites"
                        ),
                        Some(blob) => {
                            // The torn-read assertion: bytes MUST equal
                            // one of the two payloads exactly. Any other
                            // byte sequence (including a prefix, a
                            // suffix, or a blend) fails here.
                            assert!(
                                blob.bytes == payload_a || blob.bytes == payload_b,
                                "torn read detected: observed {} bytes that match neither A ({} bytes) nor B ({} bytes)",
                                blob.bytes.len(),
                                payload_a.len(),
                                payload_b.len()
                            );
                            assert_eq!(blob.handle.object_id, object_id);
                            assert_eq!(blob.handle.snap_seq, 1);
                        }
                    }
                }
            }));
        }

        for join in joins {
            tokio::time::timeout(BUDGET, join)
                .await
                .expect("per-task timeout: spawned worker hung")
                .expect("spawned worker panicked");
        }

        // Final consistency check — the post-storm latest is either A or B
        // (whichever the writer wrote last, which is deterministic in
        // serial program order but not deterministic across the race).
        let final_blob = store
            .get_latest_layer_checkpoint(object_id)
            .await
            .expect("post-storm get_latest must not error")
            .expect("post-storm slot must hold the last write");
        assert!(
            final_blob.bytes == payload_a || final_blob.bytes == payload_b,
            "post-storm latest must equal A or B exactly"
        );
    };

    tokio::time::timeout(BUDGET, body)
        .await
        .expect("case_18 body timed out — adapter likely hung under concurrency");
}

// ----- Group G: error classification (#19–#22) ------------------------------
//
// Cases #19, #20, #22 use the fault-injection seam; they are gated on
// `test-checkpoint-fault-injection`. Case #21 needs no injector — it asserts
// the natural `NotFound` classification surfaced by the production paths.

/// Op-targeted fault injector used by cases #19, #20, #22.
///
/// When the calling adapter method's `op` label matches `target_op`, the
/// injector returns the prebuilt `error_factory`-produced error; on every
/// other op label it returns `Ok(())` so unrelated adapter operations (e.g.
/// the test's own setup puts) run unimpeded.
///
/// `error_factory` is a closure rather than a stored `CheckpointError` because
/// `CheckpointError::Transient` / `Permanent` carry a boxed source that is not
/// `Clone`; the closure rebuilds a fresh error each time `maybe_inject` fires.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct OpTargetedFaultInjector {
    target_op: &'static str,
    error_factory: Box<dyn Fn() -> CheckpointError + Send + Sync>,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl OpTargetedFaultInjector {
    fn new<EF>(target_op: &'static str, error_factory: EF) -> Self
    where
        EF: Fn() -> CheckpointError + Send + Sync + 'static,
    {
        Self {
            target_op,
            error_factory: Box::new(error_factory),
        }
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl CheckpointFaultInjector for OpTargetedFaultInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        if op == self.target_op {
            Err((self.error_factory)())
        } else {
            Ok(())
        }
    }
}

/// Synthetic backend-error source used to populate the boxed `source` field on
/// `Transient` and `Permanent` injections. Carries a stable string so the
/// `Debug` / `Display` impls are observable in test-failure output.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[derive(Debug)]
struct SyntheticBackendError(&'static str);

#[cfg(feature = "test-checkpoint-fault-injection")]
impl std::fmt::Display for SyntheticBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "synthetic backend error: {}", self.0)
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl std::error::Error for SyntheticBackendError {}

/// Case #19 — an injected transient failure surfaces as
/// [`CheckpointError::Transient`] (`is_transient() == true`).
///
/// **Design.** Install an injector that returns
/// `CheckpointError::Transient { source: SyntheticBackendError(..) }` for the
/// `"put_layer_checkpoint"` op label. Drive `put_layer_checkpoint`, assert
/// the returned error classifies as `Transient` (`is_transient()`), and that
/// the other classifier helpers (`is_permanent`, `is_not_found`,
/// `is_conflict`) all return `false` — pinning the mutually-exclusive
/// classification surface the recovery / retry path branches on.
#[cfg(feature = "test-checkpoint-fault-injection")]
pub async fn case_19_transient_error_classification<F, Fut>(factory: F)
where
    F: Fn(Arc<dyn CheckpointFaultInjector>) -> Fut,
    Fut: Future<Output = CheckpointStoreInjectedPair>,
{
    let injector: Arc<dyn CheckpointFaultInjector> =
        Arc::new(OpTargetedFaultInjector::new("put_layer_checkpoint", || {
            CheckpointError::Transient {
                source: Box::new(SyntheticBackendError("injected transient failure")),
            }
        }));
    let (store, _guard) = factory(injector).await;

    let result = store
        .put_layer_checkpoint(7_777, 1, sample_layer_ompf(0x19_19_19_19), sample_meta(1))
        .await;

    let err = result.expect_err(
        "put_layer_checkpoint must surface the injected Transient error; got Ok(_) — the \
         adapter is not consulting its fault-injection seam at the top of the method body",
    );
    assert!(
        err.is_transient(),
        "injected Transient must classify as Transient via is_transient(); got {err:?}",
    );
    assert!(
        matches!(err, CheckpointError::Transient { .. }),
        "the variant returned must literally be CheckpointError::Transient {{ .. }}; got {err:?}",
    );
    // Mutually-exclusive classification: the recovery / retry path branches
    // on these classifiers, so a single error must not simultaneously be both
    // transient and permanent (or transient and conflict, etc.).
    assert!(
        !err.is_permanent(),
        "Transient error must not classify as permanent: {err:?}",
    );
    assert!(
        !err.is_not_found(),
        "Transient error must not classify as not-found: {err:?}",
    );
    assert!(
        !err.is_conflict(),
        "Transient error must not classify as conflict: {err:?}",
    );
}

/// Case #20 — an injected permanent failure surfaces as
/// [`CheckpointError::Permanent`] (`is_permanent() == true`).
///
/// **Design.** Mirrors case #19 with the `Permanent` variant, targeting the
/// `"get_latest_layer_checkpoint"` op label so we exercise the read side of
/// the injection seam in addition to case #19's write side. An unknown
/// `object_id` would normally return `Ok(None)` (per case #5), so any
/// `Permanent` here can only have come from the injection seam, not from a
/// natural absent-object code path. Asserts the classification is exclusively
/// `Permanent` — the recovery path treats this variant as non-retryable, so a
/// misclassification would silently flip a retry storm into immediate
/// operator escalation (or vice versa).
#[cfg(feature = "test-checkpoint-fault-injection")]
pub async fn case_20_permanent_error_classification<F, Fut>(factory: F)
where
    F: Fn(Arc<dyn CheckpointFaultInjector>) -> Fut,
    Fut: Future<Output = CheckpointStoreInjectedPair>,
{
    let injector: Arc<dyn CheckpointFaultInjector> = Arc::new(OpTargetedFaultInjector::new(
        "get_latest_layer_checkpoint",
        || CheckpointError::Permanent {
            source: Box::new(SyntheticBackendError("injected permanent failure")),
        },
    ));
    let (store, _guard) = factory(injector).await;

    let result = store.get_latest_layer_checkpoint(20_020).await;

    let err = result.expect_err(
        "get_latest_layer_checkpoint must surface the injected Permanent error; got Ok(_) — \
         the adapter is not consulting its fault-injection seam at the top of the method body",
    );
    assert!(
        err.is_permanent(),
        "injected Permanent must classify as Permanent via is_permanent(); got {err:?}",
    );
    assert!(
        matches!(err, CheckpointError::Permanent { .. }),
        "the variant returned must literally be CheckpointError::Permanent {{ .. }}; got {err:?}",
    );
    // Mutually-exclusive classification.
    assert!(
        !err.is_transient(),
        "Permanent error must not classify as transient: {err:?}",
    );
    assert!(
        !err.is_not_found(),
        "Permanent error must not classify as not-found: {err:?}",
    );
    assert!(
        !err.is_conflict(),
        "Permanent error must not classify as conflict: {err:?}",
    );
}

/// Case #21 — natural `NotFound` classification surface.
///
/// **Why this case is not feature-gated.** Per the `CheckpointStore` trait
/// contract, `NotFound` is NEVER returned from a `get_*`, `list_*`, or
/// `delete_*` call on the Epic-1 trait surface: absent reads return
/// `Ok(None)`, absent lists return `Ok(empty)`, and deletes on missing keys
/// return `Ok(())`. The error variant exists in the enum so future adapters
/// or richer methods can surface it, but no Epic-1 production code path trips
/// it natively. Cases #5–#8, #12, #14, and #16 already pin each individual
/// absent-path return on its own contract; this case crystallizes the
/// cross-method invariant by walking every `get_*` / `list_*` / `delete_*`
/// shape that *could* misclassify, in one body, with the explicit
/// `!err.is_not_found()` assertion shape so a future refactor that flips an
/// adapter to surface `NotFound` here trips a clear, classified failure
/// rather than a generic byte-mismatch panic.
///
/// This is the meta-classification assertion the orphan-sweep / retention
/// pruner / recovery path depends on: as long as every adapter keeps the
/// natural absent-path return contract, callers MUST NOT have to special-case
/// `NotFound`.
pub async fn case_21_not_found_classification<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    // --- get_* on absent keys must be Ok(None), never NotFound ---------------

    let unknown_object_id: u64 = 0xFFFF_FFFF_FFFF_F021;
    match store.get_latest_layer_checkpoint(unknown_object_id).await {
        Ok(None) => {}
        Ok(Some(blob)) => panic!(
            "case_21: get_latest_layer_checkpoint on a fresh adapter must be Ok(None); got \
             Some(handle={:?})",
            blob.handle
        ),
        Err(err) => {
            assert!(
                !err.is_not_found(),
                "case_21: get_latest_layer_checkpoint must NEVER surface NotFound on an absent \
                 object; trait contract pins it as Ok(None). Got: {err:?}",
            );
            panic!(
                "case_21: get_latest_layer_checkpoint on a fresh adapter must succeed with \
                 Ok(None); got unexpected error: {err:?}",
            );
        }
    }

    let unknown_hash = sample_hash_hex(0xFFFF_FFFF_FFFF_F021);
    match store.get_ext_blob(&unknown_hash).await {
        Ok(None) => {}
        Ok(Some(bytes)) => panic!(
            "case_21: get_ext_blob on an unknown hash must be Ok(None); got Some({} bytes)",
            bytes.len()
        ),
        Err(err) => {
            assert!(
                !err.is_not_found(),
                "case_21: get_ext_blob must NEVER surface NotFound on an absent hash; trait \
                 contract pins it as Ok(None). Got: {err:?}",
            );
            panic!(
                "case_21: get_ext_blob on a fresh adapter must succeed with Ok(None); got \
                 unexpected error: {err:?}",
            );
        }
    }

    match store.get_counters_sidecar().await {
        Ok(None) => {}
        Ok(Some(bytes)) => panic!(
            "case_21: get_counters_sidecar on a fresh adapter must be Ok(None); got Some({} bytes)",
            bytes.len()
        ),
        Err(err) => {
            assert!(
                !err.is_not_found(),
                "case_21: get_counters_sidecar must NEVER surface NotFound when no sidecar has \
                 been written; trait contract pins it as Ok(None). Got: {err:?}",
            );
            panic!(
                "case_21: get_counters_sidecar on a fresh adapter must succeed with Ok(None); \
                 got unexpected error: {err:?}",
            );
        }
    }

    // --- list_* on absent objects must be Ok(empty), never NotFound ---------

    match store.list_layer_checkpoints(unknown_object_id).await {
        Ok(handles) => assert!(
            handles.is_empty(),
            "case_21: list_layer_checkpoints on an unknown object must be Ok(empty); got {handles:?}",
        ),
        Err(err) => {
            assert!(
                !err.is_not_found(),
                "case_21: list_layer_checkpoints must NEVER surface NotFound on an absent \
                 object; trait contract pins it as Ok(empty). Got: {err:?}",
            );
            panic!(
                "case_21: list_layer_checkpoints on a fresh adapter must succeed with Ok(empty); \
                 got unexpected error: {err:?}",
            );
        }
    }

    // --- delete_* on missing keys must be Ok(()), never NotFound ------------

    let phantom_handle = live_edit::checkpoint::CheckpointHandle::base(
        0xFFFF_FFFF_FFFF_F022,
        1,
        "phantom-conformance-case-21".to_string(),
    );
    match store.delete_layer_checkpoint(phantom_handle).await {
        Ok(()) => {}
        Err(err) => {
            assert!(
                !err.is_not_found(),
                "case_21: delete_layer_checkpoint must NEVER surface NotFound on a missing \
                 handle; trait contract pins it as Ok(()) (retention-prune idempotency). Got: \
                 {err:?}",
            );
            panic!(
                "case_21: delete_layer_checkpoint on a missing handle must be Ok(()); got \
                 unexpected error: {err:?}",
            );
        }
    }

    match store.delete_ext_blob(&unknown_hash).await {
        Ok(()) => {}
        Err(err) => {
            assert!(
                !err.is_not_found(),
                "case_21: delete_ext_blob must NEVER surface NotFound on a missing hash; trait \
                 contract pins it as Ok(()) (orphan-sweep idempotency). Got: {err:?}",
            );
            panic!(
                "case_21: delete_ext_blob on a missing hash must be Ok(()); got unexpected \
                 error: {err:?}",
            );
        }
    }
}

/// Case #22 — an injected optimistic-concurrency conflict surfaces as
/// [`CheckpointError::Conflict`] (`is_conflict() == true`).
///
/// **Design.** Mirrors cases #19 / #20 with the `Conflict` variant, targeting
/// the `"put_layer_checkpoint"` op label (conflicts are a write-side concern;
/// the trait doc on `put_layer_checkpoint` calls out `Conflict` as the
/// classified outcome of a competing-writer race). Asserts the classification
/// is exclusively `Conflict` — the periodic checkpoint loop branches on
/// `is_conflict()` to decide whether to bump `snap_seq` and retry vs.
/// surface a permanent failure.
#[cfg(feature = "test-checkpoint-fault-injection")]
pub async fn case_22_conflict_error_classification<F, Fut>(factory: F)
where
    F: Fn(Arc<dyn CheckpointFaultInjector>) -> Fut,
    Fut: Future<Output = CheckpointStoreInjectedPair>,
{
    let injector: Arc<dyn CheckpointFaultInjector> =
        Arc::new(OpTargetedFaultInjector::new("put_layer_checkpoint", || {
            CheckpointError::Conflict
        }));
    let (store, _guard) = factory(injector).await;

    let result = store
        .put_layer_checkpoint(22_022, 1, sample_layer_ompf(0x22_22_22_22), sample_meta(1))
        .await;

    let err = result.expect_err(
        "put_layer_checkpoint must surface the injected Conflict error; got Ok(_) — the \
         adapter is not consulting its fault-injection seam at the top of the method body",
    );
    assert!(
        err.is_conflict(),
        "injected Conflict must classify as Conflict via is_conflict(); got {err:?}",
    );
    assert!(
        matches!(err, CheckpointError::Conflict),
        "the variant returned must literally be CheckpointError::Conflict; got {err:?}",
    );
    // Mutually-exclusive classification.
    assert!(
        !err.is_transient(),
        "Conflict error must not classify as transient: {err:?}",
    );
    assert!(
        !err.is_permanent(),
        "Conflict error must not classify as permanent: {err:?}",
    );
    assert!(
        !err.is_not_found(),
        "Conflict error must not classify as not-found: {err:?}",
    );
}

// ----- Group H: cross-cutting enumeration + size/retention (#23–#25) --------

/// Build an ext-refs sidecar JSON body for [`case_23_list_referenced_ext_hashes_union_across_layers`].
///
/// Both Epic-1 adapters parse the same shape — `{ "ext_hashes": [...] }` — out
/// of `sidecar_bytes` inside `list_referenced_ext_hashes`. The harness emits
/// the JSON directly so we don't pull `serde_json` into the public sample-
/// payload helpers (kept here, local to the only case that needs it).
pub fn ext_refs_sidecar_for(hashes: &[&str]) -> Bytes {
    let json = serde_json::json!({ "ext_hashes": hashes });
    Bytes::from(serde_json::to_vec(&json).expect("ext-refs sidecar JSON encodes"))
}

/// Case #23 — `list_referenced_ext_hashes` returns the union of every layer
/// checkpoint's referenced hashes across `object_id`s (no duplicates).
///
/// **Design.** Writes three layer checkpoints across three distinct
/// `object_id`s with partially-overlapping ext-hash sets:
///
/// - object_id 1, snap_seq 1 → `{h1, h2}`
/// - object_id 2, snap_seq 1 → `{h2, h3}` (overlaps on `h2`)
/// - object_id 3, snap_seq 1 → `{h3, h4}` (overlaps on `h3`)
///
/// Pairs each OMPF put with its companion ext-refs sidecar via
/// `meta.sidecar_bytes`. Both Epic-1 adapters honor this field at
/// `put_layer_checkpoint` time — InMemory was updated in Task 78 `.wi` and
/// LocalDisk does the atomic OMPF-then-sidecar write — so the harness exercises
/// the same trait-surface call across both adapters without falling back to
/// the adapter-specific `set_layer_sidecar` primitive.
///
/// Asserts the call returns the exact union `{h1, h2, h3, h4}` (no duplicates
/// on the overlapping `h2` / `h3` entries, no missing hashes). The default
/// trait impl returns an empty set; both Epic-1 adapters override it to walk
/// the per-layer sidecars and union the parsed `ext_hashes` arrays, so this
/// case is the cross-adapter assertion that the override semantics agree.
pub async fn case_23_list_referenced_ext_hashes_union_across_layers<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    use std::collections::HashSet;

    let (store, _guard) = factory().await;

    // Three layer checkpoints, three distinct object_ids, partially
    // overlapping ext-hash sets. The string content is deliberately short
    // (a few chars each) — `list_referenced_ext_hashes` treats the values
    // as opaque keys (no SHA-256 well-formedness check at parse time) so we
    // don't need the full 64-hex production shape here.
    let cases: [(u64, &[&str]); 3] = [(1, &["h1", "h2"]), (2, &["h2", "h3"]), (3, &["h3", "h4"])];

    for (object_id, hashes) in cases {
        let mut meta = sample_meta(1);
        // Thread the sidecar through the trait surface — both adapters honor
        // `meta.sidecar_bytes` and persist it atomically alongside the OMPF.
        meta.sidecar_bytes = Some(ext_refs_sidecar_for(hashes));
        store
            .put_layer_checkpoint(object_id, 1, sample_layer_ompf(object_id), meta)
            .await
            .unwrap_or_else(|err| {
                panic!("put_layer_checkpoint for object_id={object_id} succeeds; got {err:?}")
            });
    }

    let refs = store
        .list_referenced_ext_hashes()
        .await
        .expect("list_referenced_ext_hashes succeeds");
    let expected: HashSet<String> = ["h1", "h2", "h3", "h4"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        refs, expected,
        "list_referenced_ext_hashes must return the exact union across layer-checkpoints \
         with overlapping ext-hash sets (no duplicates, no missing entries)",
    );
}

/// Case #24 — the adapter accepts a large-but-not-pathologically-large
/// layer-checkpoint put without rejecting it.
///
/// **Size choice.** `LIVE_EDIT_MAX_LAYER_SIZE` defaults to 100 MiB at the
/// service edge, but the `CheckpointStore` trait itself imposes no cap — the
/// adapter contract is "opaque bytes" (SR-1.5) and a future operator might
/// lift the service-edge cap to e.g. 256 MiB without revisiting adapters. A
/// pathological-size test (literal 100 MiB) would allocate ~100 MiB inside
/// every CI runner for every adapter variant; on shared CI workers running
/// the full `--workspace` matrix that risks OOM and noisy-neighbour
/// contention. 1 MiB is two orders of magnitude over the typical fixture
/// (~16 bytes from [`sample_layer_ompf`]) and is enough to exercise any
/// internal "small buffer optimization" cliffs an adapter might have while
/// keeping the test under the OOM-safe budget — the trait contract is "no
/// size cap", which 1 MiB exercises just as well as 100 MiB.
///
/// **Round-trip.** Puts the blob, then reads it back via
/// `get_latest_layer_checkpoint` and asserts byte-for-byte equality. A torn
/// or truncated read would surface as a byte-length mismatch first; an
/// adapter that silently rejected the put would error out of the put itself.
pub async fn case_24_size_bound_sanity_accepts_large_blob<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    /// 1 MiB — large enough to exceed any reasonable per-adapter inline / small-
    /// buffer threshold without risking CI-runner OOM. See the case doc-comment
    /// above for the reasoning behind picking 1 MiB over the 100 MiB
    /// `LIVE_EDIT_MAX_LAYER_SIZE` default.
    const LARGE_BLOB_SIZE: usize = 1024 * 1024;

    let (store, _guard) = factory().await;
    let object_id: u64 = 24_024;
    let snap_seq: u64 = 1;

    // Deterministic, non-zero payload — `i as u8` cycles 0..=255 so the bytes
    // are not all-zero (which some optimizations might short-circuit on) and
    // are reproducible across runs / adapters.
    let payload: Bytes = (0..LARGE_BLOB_SIZE)
        .map(|i| i as u8)
        .collect::<Vec<u8>>()
        .into();
    assert_eq!(
        payload.len(),
        LARGE_BLOB_SIZE,
        "harness invariant: payload length must equal LARGE_BLOB_SIZE",
    );

    let handle = store
        .put_layer_checkpoint(object_id, snap_seq, payload.clone(), sample_meta(snap_seq))
        .await
        .expect(
            "put_layer_checkpoint must accept a 1 MiB blob; the trait imposes no size cap (SR-1.5)",
        );
    assert_eq!(handle.object_id, object_id);
    assert_eq!(handle.snap_seq, snap_seq);

    let blob = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest_layer_checkpoint succeeds")
        .expect("get_latest_layer_checkpoint must return Some after a large-blob put");
    assert_eq!(
        blob.bytes.len(),
        LARGE_BLOB_SIZE,
        "round-tripped blob length must equal the put length (no truncation)",
    );
    assert_eq!(
        blob.bytes, payload,
        "round-tripped blob bytes must equal the put payload verbatim (no corruption)",
    );
}

/// Case #25 — the adapter retains every generation it is told to keep; only
/// explicit `delete_layer_checkpoint` calls prune.
///
/// **Design.** The retention loop lives above the trait (in the checkpointer
/// loop); the adapter's job is the simpler "store what you're told, delete
/// what you're told" contract. This case pins that:
///
/// 1. Writes `RETAIN_GENERATIONS + 1 == 3` checkpoints at snap_seqs
///    `[100, 200, 300]` for one `object_id`.
/// 2. Asserts `list_layer_checkpoints` returns three handles (the adapter
///    did NOT silently prune to the default retention limit).
/// 3. Calls `delete_layer_checkpoint` on the OLDEST (snap_seq=100) — what
///    the retention loop would do to enforce `RETAIN_GENERATIONS=2`.
/// 4. Asserts the post-delete list has exactly two handles matching
///    `RETAIN_GENERATIONS=2`, in newest-first order `[300, 200]`.
///
/// This protects the trait-surface contract that pruning is the loop's
/// responsibility, not the adapter's — a future adapter that "helpfully"
/// auto-pruned to `RETAIN_GENERATIONS` would silently break the recovery
/// path that depends on `list_layer_checkpoints` reflecting durable on-disk
/// state without policy reinterpretation.
pub async fn case_25_retention_bound_sanity_after_prune<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    /// `LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS` default (see SDD §SR-1.x and
    /// the env-var table in CLAUDE.md). Hard-coded here so the case is
    /// self-contained and not coupled to a runtime-config crate import.
    const RETAIN_GENERATIONS: usize = 2;

    let (store, _guard) = factory().await;
    let object_id: u64 = 25_025;

    // Write RETAIN_GENERATIONS + 1 == 3 checkpoints in ascending snap_seq
    // order. Bytes are distinct per snap_seq so post-delete identity is
    // observable.
    let snap_seqs: [u64; 3] = [100, 200, 300];
    for seq in snap_seqs {
        store
            .put_layer_checkpoint(object_id, seq, sample_layer_ompf(seq), sample_meta(seq))
            .await
            .unwrap_or_else(|err| {
                panic!("put_layer_checkpoint snap_seq={seq} succeeds; got {err:?}")
            });
    }

    // Pre-prune: the adapter MUST retain all three generations. The trait
    // contract is "adapter retains every generation until told to delete" —
    // any silent auto-prune to RETAIN_GENERATIONS would surface here as
    // `handles.len() == 2` instead of 3.
    let pre_prune = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints succeeds pre-prune");
    assert_eq!(
        pre_prune.len(),
        snap_seqs.len(),
        "adapter must retain every generation until explicitly deleted; \
         the retention loop is the trait's responsibility, NOT the adapter's",
    );
    let pre_prune_seqs: Vec<u64> = pre_prune.iter().map(|h| h.snap_seq).collect();
    assert_eq!(
        pre_prune_seqs,
        vec![300, 200, 100],
        "list_layer_checkpoints must return newest-first ordering pre-prune",
    );

    // Prune the oldest generation (snap_seq=100). The retention loop would
    // hand us this handle to enforce RETAIN_GENERATIONS=2.
    let oldest = pre_prune
        .iter()
        .find(|h| h.snap_seq == 100)
        .expect("snap_seq=100 handle present in pre-prune list")
        .clone();
    store
        .delete_layer_checkpoint(oldest)
        .await
        .expect("delete_layer_checkpoint on the oldest generation succeeds");

    // Post-prune: exactly RETAIN_GENERATIONS handles, in newest-first order.
    let post_prune = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints succeeds post-prune");
    assert_eq!(
        post_prune.len(),
        RETAIN_GENERATIONS,
        "post-prune list length must equal RETAIN_GENERATIONS; got {post_prune:?}",
    );
    let post_prune_seqs: Vec<u64> = post_prune.iter().map(|h| h.snap_seq).collect();
    assert_eq!(
        post_prune_seqs,
        vec![300, 200],
        "post-prune list must contain the two newest generations in newest-first order",
    );

    // Companion: get_latest_layer_checkpoint must agree with list head per
    // the trait's ordering contract.
    let latest = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest_layer_checkpoint succeeds post-prune")
        .expect("at least one checkpoint remains post-prune");
    assert_eq!(
        latest.handle.snap_seq, 300,
        "get_latest_layer_checkpoint must agree with list_layer_checkpoints.first() post-prune",
    );
}

/// Case #26 — `read_ext_refs_sidecar(&handle)` returns the per-generation
/// referenced ext-hashes parsed from the sidecar stored alongside that
/// specific handle.
///
/// **Design.** Where case #23 pins the *union* contract of
/// `list_referenced_ext_hashes`, this case pins the *per-handle*
/// decomposition: the cascade caller needs to know exactly which hashes a
/// single generation references so it can drop the per-generation tombstone
/// from the GC accounting without touching others. The case exercises five
/// distinct contract branches against a single shared store, then pins the
/// flat_map cross-consistency invariant across the full layout:
///
/// 1. **Happy path** — two layer checkpoints under one `object_id` at
///    distinct `snap_seq`s with disjoint per-generation ext-hash sets; the
///    per-handle reads return the parsed `HashSet` matching the
///    per-generation expected set verbatim (no blur across generations).
/// 2. **Sidecar-absent path** — a third checkpoint with `sidecar_bytes =
///    None` returns `Ok(HashSet::new())`; absence is not an error per the
///    trait contract, matching both adapters' tolerance for legacy
///    generations written before the assembler emitted sidecars.
/// 3. **Cross-consistency** — the union of every per-handle read equals
///    `list_referenced_ext_hashes()`. Pins the contract that
///    `list_referenced_ext_hashes` is logically
///    `flat_map(handles, read_ext_refs_sidecar)` — the empty-sidecar handle
///    contributes nothing to either side, so the equality holds without
///    special-casing it.
/// 4. **Large-sidecar** — a generation with 100+ ext-hashes still parses
///    correctly. Exercises adapter buffering and JSON-array growth at a
///    scale closer to a real assembler-emitted sidecar (a layer with
///    hundreds of distinct external values is realistic per SR-2.3) than
///    the 2-hash micro-sets the other branches use.
/// 5. **Duplicate-hash-across-snaps** — the same ext-hash appearing in two
///    different generations' sidecars surfaces in BOTH per-handle reads
///    independently. Pins the contract that `read_ext_refs_sidecar` is
///    *per-generation* — it does NOT deduplicate across generations
///    (deduplication only happens at the union layer, inside
///    `list_referenced_ext_hashes`). Load-bearing for the retention
///    cascade: when generation N is pruned, the cascade subtracts only its
///    sidecar's hashes from the live set, and a duplicate that's still
///    referenced by surviving generation N+1 must remain live.
///
/// Layout (six checkpoints across four `object_id`s):
///
/// - `(object_id=1, snap_seq=1)` → `{h1, h2}`            (branch 1)
/// - `(object_id=1, snap_seq=2)` → `{h3, h4}`            (branch 1)
/// - `(object_id=2, snap_seq=1)` → no sidecar            (branch 2)
/// - `(object_id=3, snap_seq=1)` → 100 distinct hashes   (branch 4)
/// - `(object_id=4, snap_seq=1)` → `{dup, k1}`           (branch 5)
/// - `(object_id=4, snap_seq=2)` → `{dup, k2}`           (branch 5)
///
/// The cross-consistency assertion (branch 3) is evaluated against the
/// full layout, not a subset — it pins the flat_map contract across the
/// large-sidecar and duplicate-hash generations too.
pub async fn case_26_read_ext_refs_sidecar_per_handle<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    use std::collections::HashSet;

    let (store, _guard) = factory().await;

    // Two layer-checkpoint generations under one object_id with disjoint
    // ext-hash sets — the per-handle reads must return them separately
    // rather than blurring across generations.
    let object_id_a: u64 = 1;
    let gen_a1: (u64, &[&str]) = (1, &["h1", "h2"]);
    let gen_a2: (u64, &[&str]) = (2, &["h3", "h4"]);

    for (snap_seq, hashes) in [gen_a1, gen_a2] {
        let mut meta = sample_meta(snap_seq);
        meta.sidecar_bytes = Some(ext_refs_sidecar_for(hashes));
        store
            .put_layer_checkpoint(object_id_a, snap_seq, sample_layer_ompf(snap_seq), meta)
            .await
            .unwrap_or_else(|err| {
                panic!(
                    "put_layer_checkpoint for object_id={object_id_a}, snap_seq={snap_seq} \
                     succeeds; got {err:?}"
                )
            });
    }

    // A third generation under a *different* object_id with NO sidecar —
    // exercises the absent-sidecar tolerance contract (Ok(empty), not an
    // error).
    let object_id_b: u64 = 2;
    let snap_seq_b: u64 = 1;
    let meta_b = sample_meta(snap_seq_b); // sample_meta defaults sidecar_bytes to None
    store
        .put_layer_checkpoint(
            object_id_b,
            snap_seq_b,
            sample_layer_ompf(snap_seq_b),
            meta_b,
        )
        .await
        .unwrap_or_else(|err| {
            panic!(
                "put_layer_checkpoint for object_id={object_id_b} (no sidecar) succeeds; \
                 got {err:?}"
            )
        });

    // Retrieve real handles via `list_layer_checkpoints` — the cascade
    // caller works off `list_layer_checkpoints` handles, so the harness
    // does too rather than synthesizing handles directly.
    let handles_a = store
        .list_layer_checkpoints(object_id_a)
        .await
        .expect("list_layer_checkpoints succeeds for object_id_a");
    assert_eq!(
        handles_a.len(),
        2,
        "list_layer_checkpoints must return both generations under object_id_a; \
         got {handles_a:?}",
    );
    let handle_a1 = handles_a
        .iter()
        .find(|h| h.snap_seq == gen_a1.0)
        .expect("snap_seq=1 handle present under object_id_a")
        .clone();
    let handle_a2 = handles_a
        .iter()
        .find(|h| h.snap_seq == gen_a2.0)
        .expect("snap_seq=2 handle present under object_id_a")
        .clone();

    let handles_b = store
        .list_layer_checkpoints(object_id_b)
        .await
        .expect("list_layer_checkpoints succeeds for object_id_b");
    assert_eq!(
        handles_b.len(),
        1,
        "list_layer_checkpoints must return the single (no-sidecar) generation \
         under object_id_b; got {handles_b:?}",
    );
    let handle_b = handles_b[0].clone();

    // Per-handle reads — each generation's parsed sidecar must equal the
    // expected set for that generation, not the union or any cross-handle
    // blur.
    let expected_a1: HashSet<String> = gen_a1.1.iter().map(|s| s.to_string()).collect();
    let expected_a2: HashSet<String> = gen_a2.1.iter().map(|s| s.to_string()).collect();

    let read_a1 = store
        .read_ext_refs_sidecar(&handle_a1)
        .await
        .expect("read_ext_refs_sidecar succeeds on object_id_a snap_seq=1 handle");
    assert_eq!(
        read_a1, expected_a1,
        "read_ext_refs_sidecar must return the parsed per-generation HashSet verbatim \
         for object_id_a snap_seq=1",
    );

    let read_a2 = store
        .read_ext_refs_sidecar(&handle_a2)
        .await
        .expect("read_ext_refs_sidecar succeeds on object_id_a snap_seq=2 handle");
    assert_eq!(
        read_a2, expected_a2,
        "read_ext_refs_sidecar must return the parsed per-generation HashSet verbatim \
         for object_id_a snap_seq=2",
    );

    // Absent-sidecar handle — `Ok(HashSet::new())` per the trait contract.
    // Absence is not an error.
    let read_b = store
        .read_ext_refs_sidecar(&handle_b)
        .await
        .expect("read_ext_refs_sidecar on a handle with no sidecar must NOT error");
    assert!(
        read_b.is_empty(),
        "read_ext_refs_sidecar on a handle with no sidecar must return Ok(empty); \
         got {read_b:?}",
    );

    // ---- Branch 4: large-sidecar ----------------------------------------
    //
    // Write a generation with 100 distinct ext-hashes (each a realistic
    // 64-char lowercase-hex string via `sample_hash_hex`) and verify the
    // per-handle read returns the full set verbatim. The case-23 helper
    // accepts only `&[&str]`, so we materialize a `Vec<String>` first and
    // then borrow into a `Vec<&str>` to pass through the existing helper
    // without changing its signature.
    let object_id_c: u64 = 3;
    let snap_seq_c: u64 = 1;
    let large_hash_count: u64 = 100;
    let large_hashes_owned: Vec<String> = (0..large_hash_count)
        // Offset the tag range well beyond any other branch's hashes so the
        // large sidecar's contents are disjoint from branches 1 and 5. This
        // lets the cross-consistency union below be checked by simple
        // composition without subtle dedup interactions.
        .map(|i| sample_hash_hex(1_000 + i))
        .collect();
    let large_hashes_borrow: Vec<&str> = large_hashes_owned.iter().map(String::as_str).collect();
    let expected_c: HashSet<String> = large_hashes_owned.iter().cloned().collect();
    assert_eq!(
        expected_c.len(),
        large_hash_count as usize,
        "large-sidecar fixture must contain {large_hash_count} distinct hashes; \
         got {} — sample_hash_hex collision indicates a harness bug",
        expected_c.len(),
    );
    let mut meta_c = sample_meta(snap_seq_c);
    meta_c.sidecar_bytes = Some(ext_refs_sidecar_for(&large_hashes_borrow));
    store
        .put_layer_checkpoint(
            object_id_c,
            snap_seq_c,
            sample_layer_ompf(snap_seq_c),
            meta_c,
        )
        .await
        .unwrap_or_else(|err| {
            panic!(
                "put_layer_checkpoint for object_id={object_id_c} (large sidecar) succeeds; \
                 got {err:?}"
            )
        });

    let handles_c = store
        .list_layer_checkpoints(object_id_c)
        .await
        .expect("list_layer_checkpoints succeeds for object_id_c");
    assert_eq!(
        handles_c.len(),
        1,
        "list_layer_checkpoints must return the single large-sidecar generation \
         under object_id_c; got {handles_c:?}",
    );
    let handle_c = handles_c[0].clone();
    let read_c = store
        .read_ext_refs_sidecar(&handle_c)
        .await
        .expect("read_ext_refs_sidecar succeeds on the large-sidecar handle");
    assert_eq!(
        read_c, expected_c,
        "read_ext_refs_sidecar must round-trip a {large_hash_count}-entry sidecar \
         verbatim — any drift (truncation, dedup, sort-induced loss) violates the \
         per-generation contract",
    );

    // ---- Branch 5: duplicate-hash-across-snaps --------------------------
    //
    // Two generations under a fresh `object_id` share one ext-hash (`dup`)
    // and carry one disjoint hash each (`k1`, `k2`). The per-handle reads
    // MUST surface `dup` in both — the per-handle contract is
    // *per-generation*, not deduplicated across generations.
    //
    // Load-bearing for the retention cascade: pruning generation N must
    // subtract only N's sidecar, and a hash that's still referenced by a
    // surviving sibling generation must remain live in the union.
    let object_id_d: u64 = 4;
    let dup_hash = sample_hash_hex(2_000);
    let k1_hash = sample_hash_hex(2_001);
    let k2_hash = sample_hash_hex(2_002);

    let gen_d1_hashes: Vec<&str> = vec![dup_hash.as_str(), k1_hash.as_str()];
    let gen_d2_hashes: Vec<&str> = vec![dup_hash.as_str(), k2_hash.as_str()];

    for (snap_seq, hashes) in [(1_u64, &gen_d1_hashes), (2_u64, &gen_d2_hashes)] {
        let mut meta = sample_meta(snap_seq);
        meta.sidecar_bytes = Some(ext_refs_sidecar_for(hashes));
        store
            .put_layer_checkpoint(object_id_d, snap_seq, sample_layer_ompf(snap_seq), meta)
            .await
            .unwrap_or_else(|err| {
                panic!(
                    "put_layer_checkpoint for object_id={object_id_d}, snap_seq={snap_seq} \
                     (duplicate-hash branch) succeeds; got {err:?}"
                )
            });
    }

    let handles_d = store
        .list_layer_checkpoints(object_id_d)
        .await
        .expect("list_layer_checkpoints succeeds for object_id_d");
    assert_eq!(
        handles_d.len(),
        2,
        "list_layer_checkpoints must return both duplicate-hash generations under \
         object_id_d; got {handles_d:?}",
    );
    let handle_d1 = handles_d
        .iter()
        .find(|h| h.snap_seq == 1)
        .expect("snap_seq=1 handle present under object_id_d")
        .clone();
    let handle_d2 = handles_d
        .iter()
        .find(|h| h.snap_seq == 2)
        .expect("snap_seq=2 handle present under object_id_d")
        .clone();

    let expected_d1: HashSet<String> = [dup_hash.clone(), k1_hash.clone()].into_iter().collect();
    let expected_d2: HashSet<String> = [dup_hash.clone(), k2_hash.clone()].into_iter().collect();

    let read_d1 = store
        .read_ext_refs_sidecar(&handle_d1)
        .await
        .expect("read_ext_refs_sidecar succeeds on object_id_d snap_seq=1 handle");
    assert_eq!(
        read_d1, expected_d1,
        "read_ext_refs_sidecar must return the per-generation HashSet verbatim — the \
         shared `dup` hash MUST appear in object_id_d snap_seq=1's read",
    );
    let read_d2 = store
        .read_ext_refs_sidecar(&handle_d2)
        .await
        .expect("read_ext_refs_sidecar succeeds on object_id_d snap_seq=2 handle");
    assert_eq!(
        read_d2, expected_d2,
        "read_ext_refs_sidecar must return the per-generation HashSet verbatim — the \
         shared `dup` hash MUST appear in object_id_d snap_seq=2's read \
         (per-generation, NOT deduplicated across generations)",
    );
    assert!(
        read_d1.contains(&dup_hash) && read_d2.contains(&dup_hash),
        "duplicate-hash contract: a hash referenced by two generations must surface \
         in BOTH per-handle reads independently — got read_d1={read_d1:?}, \
         read_d2={read_d2:?}",
    );

    // ---- Branch 3: cross-consistency (over the full layout) -------------
    //
    // `list_referenced_ext_hashes` MUST equal the union of per-handle reads
    // over ALL six handles. The absent-sidecar handle contributes nothing
    // to either side, and the duplicate `dup` hash collapses to a single
    // entry on the union side (set semantics) — both sides hit the same
    // shape, so the equality holds without special-casing either.
    let listed = store
        .list_referenced_ext_hashes()
        .await
        .expect("list_referenced_ext_hashes succeeds");
    let per_handle_union: HashSet<String> = read_a1
        .iter()
        .chain(read_a2.iter())
        .chain(read_b.iter())
        .chain(read_c.iter())
        .chain(read_d1.iter())
        .chain(read_d2.iter())
        .cloned()
        .collect();
    assert_eq!(
        listed, per_handle_union,
        "list_referenced_ext_hashes must equal the union of read_ext_refs_sidecar over \
         every layer-checkpoint handle (flat_map decomposition contract), including the \
         large-sidecar and duplicate-hash generations",
    );
}

// ----- Group I: raw artifact enumeration (#27) ------------------------------

/// Boxed future returned by the per-adapter planting closures of case #27.
pub type PlantFuture = std::pin::Pin<Box<dyn Future<Output = ()> + Send>>;

/// Per-adapter fixture for case #27.
///
/// The case needs something no other case does: the ability to put the adapter
/// into a state its own trait surface **cannot reach** — a stem whose
/// companions exist without an OMPF, and a leaked write-temp. That is the
/// whole point of `list_layer_artifacts`, so the harness cannot construct it
/// through `put_layer_checkpoint`; each adapter wrapper supplies its own
/// planter instead.
///
/// A `None` planter is a **positive claim**, not a skip: it says the adapter
/// structurally cannot hold that state, and the case asserts the corresponding
/// honest shape (every stem carries an OMPF; `delete_layer_temp_artifacts`
/// returns `Ok(0)`). That keeps the InMemory run from passing vacuously.
pub struct ArtifactFixture {
    pub store: Arc<dyn CheckpointStore>,
    pub guard: Box<dyn CheckpointStoreGuard>,
    /// Plant an ext-refs sidecar + meta companion at `(object_id, snap_seq)`
    /// with **no** OMPF — the residual a `put_layer_checkpoint` that died
    /// mid-publish leaves behind. `None` ⇒ the adapter inserts a generation
    /// as one unit and cannot represent a partial publish.
    pub plant_orphan: Option<Box<dyn Fn(u64, u64) -> PlantFuture + Send + Sync>>,
    /// Plant a `<final-name>.<uuid>.tmp` leftover for `(object_id, snap_seq)`.
    /// `None` ⇒ the adapter has no temp-then-rename publish step.
    pub plant_temp: Option<Box<dyn Fn(u64, u64) -> PlantFuture + Send + Sync>>,
}

/// The ext-hash the orphan planted by case #27 references.
pub const CASE_27_ORPHAN_HASH: &str = "orphan-hash";

/// Case #27 — `list_layer_artifacts` surfaces what every OMPF-driven
/// enumeration cannot: orphaned partial-publish stems and leaked write-temps
/// (b.9qd).
///
/// **Why this case exists.** `list_layer_checkpoints`, `list_all_object_ids`
/// and `get_latest_layer_checkpoint` are all OMPF-driven, and
/// `list_referenced_ext_hashes` is sidecar-driven. Between them, an artifact
/// whose stem never acquired an OMPF is invisible to every *handle-producing*
/// path in the system and therefore unreachable from `delete_layer_checkpoint`
/// — forever. That is not merely untidy: an orphaned `.ext-refs.json` keeps
/// counting toward the retention cascade's live set, so `pruned − live`
/// degrades toward the empty set for every layer whose hashes it names
/// (SDD §22.17.3). `list_layer_artifacts` is the only enumeration that can see
/// it, which makes "the raw listing really is raw" a trait-level contract
/// rather than an adapter detail.
///
/// **Branches.**
///
/// 1. **Fresh adapter** → empty enumeration, not an error.
/// 2. **Complete generation** → all three durable kinds at one stem, and the
///    stem is simultaneously visible to `list_layer_checkpoints`.
/// 3. **Orphan** (adapters that can hold one) → the stem appears with an
///    `ExtRefs` artifact and **no** `Ompf`, while `list_layer_checkpoints` and
///    `list_all_object_ids` cannot see it at all. The contrast is the case.
/// 4. **No orphan possible** (InMemory) → every enumerated stem carries an
///    `Ompf`, and the `Ompf` stems are exactly the generations
///    `list_layer_checkpoints` reports. The adapter's honest shape, asserted
///    rather than skipped.
/// 5. **Temp** → a `Temp` artifact appears beside a live generation without
///    fabricating a second generation, and `delete_layer_temp_artifacts`
///    reclaims it while leaving that generation intact. On adapters with no
///    temp step: no `Temp` artifact is ever emitted and the default
///    `Ok(0)` contract holds.
/// 6. **Idempotent temp delete** → `Ok(0)` on a stem with no temps, on every
///    adapter. Never `NotFound`.
pub async fn case_27_list_layer_artifacts_enumerates_orphans_and_temps<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = ArtifactFixture>,
{
    use std::collections::HashSet;

    use live_edit::checkpoint::{LayerArtifact, LayerArtifactKind};

    /// Kinds present at one `(object_id, snap_seq)` stem.
    fn kinds_at(
        artifacts: &[LayerArtifact],
        object_id: u64,
        snap_seq: u64,
    ) -> HashSet<LayerArtifactKind> {
        artifacts
            .iter()
            .filter(|a| a.object_id == object_id && a.snap_seq == snap_seq)
            .map(|a| a.kind)
            .collect()
    }

    // ---- Branch 1: fresh adapter ----------------------------------------
    {
        let fixture = factory().await;
        let artifacts = fixture
            .store
            .list_layer_artifacts()
            .await
            .expect("list_layer_artifacts on a fresh adapter must be Ok, not Err");
        assert!(
            artifacts.is_empty(),
            "a fresh adapter has no artifacts; got {artifacts:?}"
        );
        drop(fixture.guard);
    }

    let fixture = factory().await;
    let store = Arc::clone(&fixture.store);

    // ---- Branch 2: a complete generation --------------------------------
    let live_object: u64 = 1;
    let live_snap: u64 = 1;
    let mut meta = sample_meta(live_snap);
    meta.sidecar_bytes = Some(ext_refs_sidecar_for(&["live-hash"]));
    store
        .put_layer_checkpoint(live_object, live_snap, sample_layer_ompf(live_snap), meta)
        .await
        .expect("put_layer_checkpoint must succeed");

    let artifacts = store
        .list_layer_artifacts()
        .await
        .expect("list_layer_artifacts must succeed");
    assert_eq!(
        kinds_at(&artifacts, live_object, live_snap),
        [
            LayerArtifactKind::Ompf,
            LayerArtifactKind::Meta,
            LayerArtifactKind::ExtRefs
        ]
        .into_iter()
        .collect::<HashSet<_>>(),
        "a completely published generation must surface all three durable kinds; got \
         {artifacts:?}"
    );

    // ---- Branches 3 / 4: the orphan, or the claim that there can't be one -
    let orphan_object: u64 = 2;
    let orphan_snap: u64 = 9;
    if let Some(plant_orphan) = fixture.plant_orphan.as_ref() {
        plant_orphan(orphan_object, orphan_snap).await;

        let artifacts = store
            .list_layer_artifacts()
            .await
            .expect("list_layer_artifacts must succeed");
        let kinds = kinds_at(&artifacts, orphan_object, orphan_snap);
        assert!(
            kinds.contains(&LayerArtifactKind::ExtRefs),
            "the orphan's ext-refs sidecar MUST be enumerable — it is the artifact that pins \
             this snapshot's whole ext hash set in the retention cascade's live set, and the \
             raw listing is the only thing that can ever offer it for reclamation. Got {kinds:?}"
        );
        assert!(
            !kinds.contains(&LayerArtifactKind::Ompf),
            "the orphan must have NO Ompf artifact — that absence is what makes the stem an \
             orphan candidate rather than a generation. Got {kinds:?}"
        );

        // The contrast that is the whole point of the method.
        assert!(
            store
                .list_layer_checkpoints(orphan_object)
                .await
                .expect("list_layer_checkpoints must succeed")
                .is_empty(),
            "list_layer_checkpoints is OMPF-driven, so it cannot see the orphan — which is \
             exactly why the orphan is unreachable from delete_layer_checkpoint"
        );
        assert!(
            !store
                .list_all_object_ids()
                .await
                .expect("list_all_object_ids must succeed")
                .contains(&orphan_object),
            "and neither can list_all_object_ids"
        );
        assert!(
            store
                .list_referenced_ext_hashes()
                .await
                .expect("list_referenced_ext_hashes must succeed")
                .contains(CASE_27_ORPHAN_HASH),
            "meanwhile the sidecar-driven live set DOES count the orphan's hashes — the \
             asymmetry between these two enumerations is the leak b.9qd exists to close"
        );
    } else {
        // The honest shape for an adapter that inserts a generation as one
        // unit under a mutex: no stem can lack an OMPF.
        let artifacts = store
            .list_layer_artifacts()
            .await
            .expect("list_layer_artifacts must succeed");
        let ompf_stems: HashSet<(u64, u64)> = artifacts
            .iter()
            .filter(|a| a.kind == LayerArtifactKind::Ompf)
            .map(|a| (a.object_id, a.snap_seq))
            .collect();
        let all_stems: HashSet<(u64, u64)> = artifacts
            .iter()
            .map(|a| (a.object_id, a.snap_seq))
            .collect();
        assert_eq!(
            ompf_stems, all_stems,
            "this adapter reports no planting seam, i.e. it claims it structurally cannot hold \
             a partial publish. Then every enumerated stem must carry an Ompf — a stem without \
             one would be an orphan the adapter says it cannot produce. Got {artifacts:?}"
        );
        let mut listed: HashSet<(u64, u64)> = HashSet::new();
        for object_id in store
            .list_all_object_ids()
            .await
            .expect("list_all_object_ids must succeed")
        {
            for handle in store
                .list_layer_checkpoints(object_id)
                .await
                .expect("list_layer_checkpoints must succeed")
            {
                listed.insert((handle.object_id, handle.snap_seq));
            }
        }
        assert_eq!(
            ompf_stems, listed,
            "and the raw enumeration's Ompf stems must be exactly the generations the \
             OMPF-driven listings report — for this adapter the two views genuinely coincide, \
             which is the claim, not an accident"
        );
    }

    // ---- Branch 5: the temp -------------------------------------------—-
    if let Some(plant_temp) = fixture.plant_temp.as_ref() {
        plant_temp(live_object, live_snap).await;

        let artifacts = store
            .list_layer_artifacts()
            .await
            .expect("list_layer_artifacts must succeed");
        assert!(
            kinds_at(&artifacts, live_object, live_snap).contains(&LayerArtifactKind::Temp),
            "a leaked write-temp must be enumerable — it is inert to every reader, so nothing \
             else will ever reclaim it. Got {artifacts:?}"
        );
        assert_eq!(
            store
                .list_layer_checkpoints(live_object)
                .await
                .expect("list_layer_checkpoints must succeed")
                .len(),
            1,
            "and it must NOT fabricate a second generation"
        );

        let removed = store
            .delete_layer_temp_artifacts(live_object, live_snap)
            .await
            .expect("delete_layer_temp_artifacts must succeed");
        assert!(
            removed >= 1,
            "the temp must be reclaimed; got removed={removed}"
        );
        let artifacts = store
            .list_layer_artifacts()
            .await
            .expect("list_layer_artifacts must succeed");
        assert!(
            !kinds_at(&artifacts, live_object, live_snap).contains(&LayerArtifactKind::Temp),
            "no Temp artifact may remain after the reclamation; got {artifacts:?}"
        );
        assert!(
            store
                .get_latest_layer_checkpoint(live_object)
                .await
                .expect("get_latest_layer_checkpoint must succeed")
                .is_some(),
            "and the stem's LIVE generation must be untouched — a temp can be garbage while \
             its generation is perfectly durable, so reclaiming one must never reach for the \
             other"
        );
    } else {
        let artifacts = store
            .list_layer_artifacts()
            .await
            .expect("list_layer_artifacts must succeed");
        assert!(
            artifacts.iter().all(|a| a.kind != LayerArtifactKind::Temp),
            "this adapter has no temp-then-rename step, so it must never emit a Temp artifact; \
             got {artifacts:?}"
        );
        assert_eq!(
            store
                .delete_layer_temp_artifacts(live_object, live_snap)
                .await
                .expect("delete_layer_temp_artifacts must succeed"),
            0,
            "and `Ok(0)` must hold rather than NotImplemented or NotFound — whether the adapter \
             gets there via the trait default or via an override that enumerated and found \
             nothing to delete. An adapter that can INHERIT a foreign tool's temp belongs in \
             the branch above with a planter, not here: `Ok(0)` from the no-op default would \
             pass this assertion while leaving the reconciliation pass a candidate it can \
             never retire"
        );
    }

    // ---- Branch 6: idempotent on an unknown stem ------------------------
    assert_eq!(
        store
            .delete_layer_temp_artifacts(999_999, 1)
            .await
            .expect("delete_layer_temp_artifacts on an unknown stem must be Ok, never NotFound"),
        0,
    );

    drop(fixture.guard);
}

// ----- Group J: tail generations (#28–#37) ----------------------------------
//
// Group letter: `I` was already spent on raw artifact enumeration (#27, b.9qd),
// so the tail block is `J`. Two groups sharing a letter would make "the Group I
// cases" ambiguous in exactly the place this harness is read from — a wrapper
// section header.
//
// No case in this group is `skip_if_not_implemented!`-gated, and none may be.
// All three shipped adapters store tails as of Epic 2 Tasks 3 and 4, so a
// `NotImplemented` surfacing out of a tail put or a get-by-handle is a defect in
// the adapter (or a decorator that forgot to forward the method), not a
// capability the harness should tolerate. Gating would convert that defect into
// a silent pass.

/// Case #28 — a [`CheckpointKind::Tail`] put round-trips through
/// `get_layer_checkpoint` and `list_layer_checkpoints` (SR-2.3 / SR-2.4).
///
/// **The foundational tail case.** Everything later in Group J assumes a tail
/// can be written and read back at all; this is where that is established, on a
/// store holding nothing else, so a failure here localizes to the put/get/list
/// triangle rather than to any interaction with a base.
///
/// The assertions, and why each is load-bearing:
///
/// - **The returned handle is a tail at `to_seq`.** `put_layer_checkpoint`
///   learns the kind from `meta.kind` — there is no `put_tail_checkpoint`
///   sibling — so the returned handle's `kind` is the only put-time evidence the
///   adapter understood which kind it was asked for. A `Base` handle back from a
///   `Tail` meta means the caller now addresses the generation with the wrong
///   kind forever after.
/// - **Bytes byte-identical.** Opaque payload (SR-1.5). The fixture's `"TAIL"`
///   discriminator makes a base/OMPF-path mix-up recognizable.
/// - **Lineage round-trips.** `kind`, `base_snap_seq`, `from_seq` and
///   `snap_seq == to_seq` all come back, so a caller can read a generation's
///   lineage off the fetched blob without re-listing — which is exactly what
///   Epic 4's gather does.
/// - **Provenance round-trips.** A tail's meta is NOT identity-empty: it carries
///   the same `session_id` / `layer_name` identity as the base it is rooted at,
///   and the apply-path guard refuses a blob whose identity fields are blank.
/// - **The listing reports exactly one `(to_seq, Tail)` generation**, and that
///   listed handle is itself usable for a second get. List → get composability
///   is the whole reason `get_layer_checkpoint` takes a handle instead of an
///   `object_id`, and Epic 4's lineage gather is its first real consumer.
/// - **Re-put idempotency (SR-9.2).** Two replicas capturing the same immutable
///   stream slice land the same `(object_id, to_seq)`. The put must therefore be
///   a blind overwrite of one slot — identical re-put leaves one generation, and
///   a same-key put with *different* bytes still leaves one generation holding
///   the later bytes. A read-modify-write or an append would turn duplicate
///   leadership into either a `Conflict` storm or a duplicated generation.
pub async fn case_28_tail_put_then_get_by_handle_and_list_round_trips<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    let object_id: u64 = 28_028;
    // A first tail in a lineage: `from_seq == base_snap_seq`, and the span
    // `(from_seq, to_seq]` covers the commits captured after the base.
    let base_snap_seq: u64 = 12;
    let from_seq: u64 = 12;
    let to_seq: u64 = 19;

    let envelope = sample_tail_envelope(0x7A11);
    let meta = sample_tail_meta(base_snap_seq, from_seq, to_seq);

    let handle = store
        .put_layer_checkpoint(object_id, to_seq, envelope.clone(), meta.clone())
        .await
        .expect(
            "put_layer_checkpoint with a Tail meta must succeed on a fresh adapter — a \
             NotImplemented or an error here is an adapter defect, NOT a capability to skip past",
        );

    assert_eq!(
        handle.kind,
        CheckpointKind::Tail,
        "the returned handle's kind MUST equal meta.kind — it is the only put-time evidence the \
         adapter read the kind off the meta, and every later address of this generation \
         (get / delete / read_ext_refs_sidecar) matches on it",
    );
    assert_eq!(handle.object_id, object_id);
    assert_eq!(
        handle.snap_seq, to_seq,
        "a tail's to_seq is what lands in snap_seq — one ordering number for both kinds",
    );
    assert_eq!(
        handle.seq(),
        to_seq,
        "seq() is the kind-neutral reader of that same number",
    );

    // ---- get by the returned handle -------------------------------------
    let blob = store
        .get_layer_checkpoint(&handle)
        .await
        .expect("get_layer_checkpoint on a just-written tail must succeed")
        .expect("get_layer_checkpoint must return Some for a generation that was just put");

    assert_eq!(blob.handle.object_id, object_id);
    assert_eq!(blob.handle.snap_seq, to_seq);
    assert_eq!(
        blob.handle.kind,
        CheckpointKind::Tail,
        "blob.handle's (object_id, snap_seq, kind) MUST equal the addressed handle's",
    );
    assert_eq!(
        blob.bytes, envelope,
        "tail bytes are opaque (SR-1.5) and MUST round-trip verbatim; a mismatch whose first \
         bytes read as the OMPF magic means the adapter served the base read path",
    );

    // Lineage — the half of the meta that only a tail carries.
    assert_eq!(
        blob.meta.kind,
        CheckpointKind::Tail,
        "meta.kind MUST round-trip: a tail that reads back as a base is a full-state snapshot in \
         the eyes of every restore path, and replaying it as one loses the base beneath it",
    );
    assert_eq!(
        blob.meta.base_snap_seq,
        Some(base_snap_seq),
        "base_snap_seq names the base this envelope replays onto; a lost or invented one layers \
         the tail happily but wrongly",
    );
    assert_eq!(
        blob.meta.from_seq,
        Some(from_seq),
        "from_seq is the exclusive lower bound of the span this envelope carries",
    );
    assert_eq!(
        blob.meta.snap_seq,
        Some(to_seq),
        "and the upper bound lives in snap_seq — there is deliberately no separate to_seq field",
    );

    // Provenance — a tail's meta is NOT identity-empty (the apply-path guard
    // refuses a blob with a blank session_id / layer_name outright).
    assert_eq!(blob.meta.creator_replica_id, meta.creator_replica_id);
    assert_eq!(blob.meta.written_at_unix_ms, meta.written_at_unix_ms);
    assert_eq!(
        blob.meta.session_id, meta.session_id,
        "a tail carries the same session identity as the base it is rooted at; blank fields make \
         the composed restore refuse the blob at the live-mapping guard",
    );
    assert_eq!(blob.meta.layer_name, meta.layer_name);
    assert_eq!(blob.meta.created_by, meta.created_by);
    assert_eq!(blob.meta.server_ts, meta.server_ts);

    // ---- the listing ----------------------------------------------------
    let listed = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints succeeds");
    assert_eq!(
        kind_seq_pairs(&listed),
        vec![(to_seq, CheckpointKind::Tail)],
        "one tail put must surface exactly one generation, at to_seq, labelled Tail; got \
         {listed:?}",
    );

    // ---- list → get composability (Epic-4 gather depends on it) ----------
    let via_listed = store
        .get_layer_checkpoint(&listed[0])
        .await
        .expect("get_layer_checkpoint on a listed handle succeeds")
        .expect("a handle the adapter itself minted MUST resolve");
    assert_eq!(
        via_listed.bytes, envelope,
        "a handle taken straight from list_layer_checkpoints must fetch the identical blob — \
         list-then-get is how lineage gather reaches a generation that is not the newest",
    );
    assert_eq!(via_listed.meta.kind, CheckpointKind::Tail);
    assert_eq!(via_listed.meta.base_snap_seq, Some(base_snap_seq));
    assert_eq!(via_listed.meta.from_seq, Some(from_seq));

    // ---- SR-9.2: re-put of the identical slice --------------------------
    //
    // Duplicate-leader safety. Two replicas capturing the same immutable stream
    // slice produce byte-identical artifacts at the same `(object_id, to_seq)`,
    // so the second put must be an ordinary blind overwrite: no Conflict, no
    // second generation.
    let handle_again = store
        .put_layer_checkpoint(
            object_id,
            to_seq,
            envelope.clone(),
            sample_tail_meta(base_snap_seq, from_seq, to_seq),
        )
        .await
        .expect(
            "re-putting the identical tail at the same (object_id, to_seq) must succeed — two \
             replicas capturing the same immutable stream slice do exactly this (SR-9.2)",
        );
    assert_eq!(
        handle_again, handle,
        "and it must mint the identical handle: the adapter_key is derived from \
         (object_id, seq, kind), so a re-put cannot address a different artifact",
    );

    let listed = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints after the identical re-put succeeds");
    assert_eq!(
        kind_seq_pairs(&listed),
        vec![(to_seq, CheckpointKind::Tail)],
        "an identical re-put must still leave exactly ONE generation; got {listed:?}",
    );
    assert_eq!(
        store
            .get_layer_checkpoint(&handle)
            .await
            .expect("get after the identical re-put succeeds")
            .expect("the generation is still there")
            .bytes,
        envelope,
        "and it must still round-trip the same bytes",
    );

    // ---- SR-9.2: same key, different bytes ------------------------------
    //
    // The overwrite has to be *blind* — no read-modify-write, no
    // append-a-second-generation. A capture that re-runs over a slightly
    // different byte encoding of the same range is the realistic source of this
    // shape, and it must not fork the lineage.
    let rewritten = sample_tail_envelope(0x7A12);
    assert_ne!(
        rewritten, envelope,
        "test precondition: the overwrite must carry distinct bytes for last-writer-wins to be \
         observable",
    );
    store
        .put_layer_checkpoint(
            object_id,
            to_seq,
            rewritten.clone(),
            sample_tail_meta(base_snap_seq, from_seq, to_seq),
        )
        .await
        .expect("a same-key tail put with different bytes must succeed (blind overwrite)");

    let listed = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints after the overwrite succeeds");
    assert_eq!(
        kind_seq_pairs(&listed),
        vec![(to_seq, CheckpointKind::Tail)],
        "the overwrite must not fork the lineage into two generations; got {listed:?}",
    );
    assert_eq!(
        store
            .get_layer_checkpoint(&handle)
            .await
            .expect("get after the overwrite succeeds")
            .expect("the generation survives the overwrite")
            .bytes,
        rewritten,
        "and the surviving generation holds the LATER bytes (last-writer-wins)",
    );
}

/// Case #29 — `get_layer_checkpoint`'s addressing contract, exercised across
/// **both** kinds on one store (SR-2.3).
///
/// `get_latest_layer_checkpoint` structurally cannot express what this method is
/// for: a lineage gather picks the newest *base*, which is by construction not
/// the newest *generation*. So the handle — not the `object_id` — is the
/// addressing argument, and this case pins every branch of that addressing on a
/// store holding `base@3` and `tail@8` with distinct bytes per generation.
///
/// **Branches.**
///
/// 1. **Base-kind parity with `get_latest`** — run BEFORE the tail put, and
///    deliberately so. Once a tail sits above the base, the object's newest
///    generation is the tail, and the assertion would silently stop being about
///    a *base* at all; the mixed-state parity question belongs to the sibling
///    listing case that owns mixed-kind ordering.
/// 2. **Fetch-specific.** Each handle fetches its own generation's bytes and
///    lineage. Neither returns the other's payload.
/// 3. **Kind discriminates at an equal sequence.** A base handle at the tail's
///    sequence, and a tail handle at the base's, both return `Ok(None)` — the
///    kind is part of the address, not a label on the result. Without this a
///    caller asking for a tail can be handed a full snapshot to replay as a
///    delta.
/// 4. **Absent generation and absent object → `Ok(None)`, never `NotFound`.**
///    Asserted with the same match shape cases #12 / #14 / #21 use, because the
///    concurrent-prune race (list observed the handle, retention removed it
///    before the get landed) makes absence an *ordinary* outcome for this
///    method's real caller.
/// 5. **Junk `adapter_key` tolerance.** Adapters MUST address off
///    `(object_id, snap_seq, kind)` and MAY ignore `adapter_key`, so a handle a
///    caller reconstructed without ever seeing the adapter's key still resolves
///    — and the blob comes back carrying the adapter's OWN canonical key, not an
///    echo of the junk. That echo is what would make a fabricated handle
///    round-trip into an unusable one.
pub async fn case_29_get_layer_checkpoint_by_handle_contract_both_kinds<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    /// `Ok(None)`, never `NotFound`, never a blob — the shared absent-handle
    /// assertion, spelled with cases #12 / #14 / #21's match shape.
    async fn assert_absent(store: &Arc<dyn CheckpointStore>, handle: &CheckpointHandle, why: &str) {
        match store.get_layer_checkpoint(handle).await {
            Ok(None) => {}
            Ok(Some(blob)) => panic!(
                "case_29: get_layer_checkpoint({handle:?}) must be Ok(None) — {why}; got \
                 Some(handle={:?}, {} bytes)",
                blob.handle,
                blob.bytes.len(),
            ),
            Err(err) => {
                assert!(
                    !err.is_not_found(),
                    "case_29: get_layer_checkpoint must NEVER surface NotFound for an absent \
                     generation — {why}. Absence is Ok(None) per the trait's get-path rule, and \
                     the concurrent-prune race makes it an ordinary outcome. Got: {err:?}",
                );
                panic!(
                    "case_29: get_layer_checkpoint on an absent generation must be Ok(None) — \
                     {why}; got unexpected error: {err:?}",
                );
            }
        }
    }

    let (store, _guard) = factory().await;

    let object_id: u64 = 29_029;
    let base_seq: u64 = 3;
    let to_seq: u64 = 8;

    let base_bytes = sample_layer_ompf(0x2901);
    let tail_bytes = sample_tail_envelope(0x2902);
    assert_ne!(
        base_bytes, tail_bytes,
        "test precondition: the two generations must carry distinct bytes so \"neither returns \
         the other's payload\" is observable",
    );

    let base_meta = sample_meta(base_seq);
    let base_handle = store
        .put_layer_checkpoint(object_id, base_seq, base_bytes.clone(), base_meta.clone())
        .await
        .expect("put_layer_checkpoint (base) succeeds");
    assert_eq!(
        base_handle.kind,
        CheckpointKind::Base,
        "a meta with the default kind must produce a Base handle",
    );

    // ---- Branch 1: base-kind parity with get_latest, BEFORE the tail -----
    {
        let listed = store
            .list_layer_checkpoints(object_id)
            .await
            .expect("list_layer_checkpoints succeeds");
        assert_eq!(
            kind_seq_pairs(&listed),
            vec![(base_seq, CheckpointKind::Base)],
            "precondition for the parity claim: at this point the object holds the base alone",
        );
        let head = listed.first().expect("the listing is non-empty");
        let by_handle = store
            .get_layer_checkpoint(head)
            .await
            .expect("get_layer_checkpoint on the listing head succeeds")
            .expect("the listing head resolves");
        let latest = store
            .get_latest_layer_checkpoint(object_id)
            .await
            .expect("get_latest_layer_checkpoint succeeds")
            .expect("a checkpoint exists");
        assert_eq!(
            by_handle.bytes, latest.bytes,
            "get_layer_checkpoint(list.first()) and get_latest_layer_checkpoint must agree \
             byte-for-byte on a base-only object — the new addressing path is a generalization \
             of the old one, not a second, subtly different reader",
        );
        assert_eq!(
            (
                by_handle.handle.object_id,
                by_handle.handle.snap_seq,
                by_handle.handle.kind
            ),
            (
                latest.handle.object_id,
                latest.handle.snap_seq,
                latest.handle.kind
            ),
            "and they must agree on the addressed generation's identity",
        );
        assert_eq!(by_handle.meta.kind, CheckpointKind::Base);
    }

    // ---- the tail lands on top -------------------------------------------
    let tail_meta = sample_tail_meta(base_seq, base_seq, to_seq);
    let tail_handle = store
        .put_layer_checkpoint(object_id, to_seq, tail_bytes.clone(), tail_meta.clone())
        .await
        .expect("put_layer_checkpoint (tail) succeeds");
    assert_eq!(tail_handle.kind, CheckpointKind::Tail);

    // ---- Branch 2: each handle fetches its own generation ----------------
    let got_base = store
        .get_layer_checkpoint(&base_handle)
        .await
        .expect("get_layer_checkpoint(base handle) succeeds")
        .expect("the base is still addressable with a tail above it");
    assert_eq!(
        got_base.bytes, base_bytes,
        "the base handle must fetch the BASE's bytes — a lineage gather asks for the base \
         precisely because it is not the newest generation",
    );
    assert_eq!(got_base.meta.kind, CheckpointKind::Base);
    assert_eq!(
        (got_base.meta.base_snap_seq, got_base.meta.from_seq),
        (None, None),
        "a base carries no lineage: both fields MUST be None (the invariant's Base half)",
    );

    let got_tail = store
        .get_layer_checkpoint(&tail_handle)
        .await
        .expect("get_layer_checkpoint(tail handle) succeeds")
        .expect("the tail is addressable");
    assert_eq!(
        got_tail.bytes, tail_bytes,
        "and the tail handle must fetch the TAIL's bytes",
    );
    assert_eq!(got_tail.meta.kind, CheckpointKind::Tail);
    assert_eq!(
        (got_tail.meta.base_snap_seq, got_tail.meta.from_seq),
        (Some(base_seq), Some(base_seq)),
        "a tail carries both lineage fields (the invariant's Tail half)",
    );

    // ---- Branch 3: kind discriminates at an equal sequence ---------------
    assert_absent(
        &store,
        &CheckpointHandle::base(object_id, to_seq, "kind-probe".to_string()),
        "a BASE handle at the tail's sequence names a generation that does not exist; resolving \
         it to the tail would hand a caller a delta envelope to restore as a full snapshot",
    )
    .await;
    assert_absent(
        &store,
        &CheckpointHandle::tail(object_id, base_seq, "kind-probe".to_string()),
        "a TAIL handle at the base's sequence names a generation that does not exist; resolving \
         it to the base would hand a caller a full snapshot to replay as a delta",
    )
    .await;

    // ---- Branch 4: absent generation, both kinds; and absent object ------
    assert_absent(
        &store,
        &CheckpointHandle::tail(
            object_id,
            999_999,
            "phantom-conformance-case-29".to_string(),
        ),
        "no tail was ever written at that sequence",
    )
    .await;
    assert_absent(
        &store,
        &CheckpointHandle::base(
            object_id,
            999_999,
            "phantom-conformance-case-29".to_string(),
        ),
        "no base was ever written at that sequence",
    )
    .await;

    let unknown_object_id: u64 = 0xFFFF_FFFF_FFFF_F029;
    assert_absent(
        &store,
        &CheckpointHandle::base(unknown_object_id, base_seq, "phantom-object".to_string()),
        "the object itself was never written (base handle)",
    )
    .await;
    assert_absent(
        &store,
        &CheckpointHandle::tail(unknown_object_id, to_seq, "phantom-object".to_string()),
        "the object itself was never written (tail handle)",
    )
    .await;

    // ---- Branch 5: junk adapter_key tolerance ---------------------------
    let fabricated = CheckpointHandle::tail(object_id, to_seq, "junk-adapter-key".to_string());
    let via_junk = store
        .get_layer_checkpoint(&fabricated)
        .await
        .expect("get_layer_checkpoint with a junk adapter_key succeeds")
        .expect(
            "a handle carrying a junk adapter_key MUST still resolve — adapters address off \
             (object_id, snap_seq, kind) and MAY ignore the key, which is what lets a caller \
             that never saw the adapter's key reconstruct a handle",
        );
    assert_eq!(
        via_junk.bytes, tail_bytes,
        "and it must resolve to the tail, not to the base sharing the object",
    );

    let listed = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints succeeds");
    let listed_tail = listed
        .iter()
        .find(|h| h.kind == CheckpointKind::Tail)
        .expect("the listing reports the tail");
    assert_eq!(
        via_junk.handle.adapter_key, listed_tail.adapter_key,
        "blob.handle.adapter_key MUST be the adapter's own canonical key — the identical string \
         list_layer_checkpoints mints — so a fabricated handle round-trips into a USABLE one",
    );
    assert_ne!(
        via_junk.handle.adapter_key, "junk-adapter-key",
        "it must NOT be an echo of the caller's key: echoing turns the returned handle into \
         another unusable fabrication and hides the substitution from the caller",
    );
}

/// Case #30 — `list_layer_checkpoints` interleaves both kinds in ONE strict
/// descending order over the sequence, with **Base before Tail** at an equal
/// sequence (SR-2.4).
///
/// The single source of truth is `generation_sort_key(sequence, kind) ->
/// (Reverse<u64>, u8)` in `src/checkpoint/adapters/mod.rs`. Each adapter
/// realizes it differently — InMemory structurally (a per-sequence
/// `GenerationSlot` emitting `base` then `tail`), LocalDisk and StorageApi by
/// parsing two stem families out of an arbitrarily-ordered enumeration and
/// sorting — and collapsing that divergence onto one observable order is
/// precisely what a conformance case is for.
///
/// **The tie is not hypothetical.** SR-4.1 re-base composes `base@S0 + tails`
/// into a fresh **base at S**, where `S` is the newest consumed tail's `to_seq`,
/// so `base@S` and `tail@to_seq == S` genuinely coexist until retention removes
/// one. The fixture below is that shape: `tail@5` covered `(2, 5]` over an older
/// `base@2`, and the re-base that consumed it minted `base@5`.
///
/// Insertion is deliberately scrambled (`tail@5`, `base@7`, `tail@9`, `base@5`),
/// mirroring case #11's design — the ordering contract keys on the generation's
/// sequence and kind, never on write order, and putting `base@5` **last** means
/// an adapter that accidentally preserved insertion order would emit the tie
/// backwards.
///
/// **What the assertions pin.**
///
/// 1. The full `(sequence, kind)` order, written out as a literal rather than
///    computed by calling the production sort — the case must fail if
///    `generation_sort_key` itself is changed, which it cannot do if it derives
///    its expectation from the thing under test.
/// 2. Every listed handle's `adapter_key` is non-empty and the four are pairwise
///    distinct. A shared key between `base@5` and `tail@5` is exactly the
///    collision the kind-qualified stem exists to prevent, and it would make the
///    two generations alias each other on every later address.
/// 3. Each listed handle round-trips through `get_layer_checkpoint` to the bytes
///    originally put for that `(sequence, kind)` — the ordering claim is worth
///    nothing if the handles it orders resolve to the wrong artifacts.
/// 4. Ordering is stable across successive read-only calls (no enumeration
///    order leaking through, no round-robin on the read path).
/// 5. `get_latest_layer_checkpoint` agrees with `list(..).first()` **including
///    when that head is a Tail**. `get_latest` is retained as `list().first()`
///    and MUST be able to return a tail; an adapter that "helpfully" filters it
///    to bases fails here, and that is a defect in the adapter rather than a
///    reason to relax the case. Restore-side "newest base wins" is Task 5's
///    recovery hook (kind-aware selection over the listing), implemented once
///    there rather than three times adapter-side.
/// 6. The tie-break's practical consequence, on a second fresh store holding
///    only `base@5` and `tail@5`: `get_latest` returns the **base** — the
///    artifact that restores on its own — which is the whole reason Base wins
///    the tie for every kind-blind caller.
pub async fn case_30_list_layer_checkpoints_mixed_kind_newest_first_base_before_tail<F, Fut>(
    factory: F,
) where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    let (store, _guard) = factory().await;

    let object_id: u64 = 30_030;

    // Four generations on one object, in scrambled insertion order. The tuple is
    // `(sequence, kind, bytes, meta)`; bytes are distinct per generation so
    // "each handle fetches its OWN artifact" is observable rather than vacuous.
    let generations: Vec<(u64, CheckpointKind, Bytes, CheckpointMeta)> = vec![
        // tail@5 — rooted at the pre-re-base `base@2`, covering (2, 5].
        (
            5,
            CheckpointKind::Tail,
            sample_tail_envelope(0x3005),
            sample_tail_meta(2, 2, 5),
        ),
        // base@7 — an ordinary full snapshot above it.
        (
            7,
            CheckpointKind::Base,
            sample_layer_ompf(0x3007),
            sample_meta(7),
        ),
        // tail@9 — rooted at base@7, covering (7, 9].
        (
            9,
            CheckpointKind::Tail,
            sample_tail_envelope(0x3009),
            sample_tail_meta(7, 7, 9),
        ),
        // base@5 — the SR-4.1 re-base product that consumed tail@5, written
        // LAST on purpose: insertion order must not be mistakable for the
        // reason Base sorts ahead of the tail sharing its sequence.
        (
            5,
            CheckpointKind::Base,
            sample_layer_ompf(0x3105),
            sample_meta(5),
        ),
    ];

    // Precondition for assertion 3: all four payloads differ pairwise.
    for (i, (seq_a, kind_a, bytes_a, _)) in generations.iter().enumerate() {
        for (seq_b, kind_b, bytes_b, _) in generations.iter().skip(i + 1) {
            assert_ne!(
                bytes_a, bytes_b,
                "test precondition: every generation must carry distinct bytes so a mis-addressed \
                 fetch is observable; ({seq_a}, {kind_a:?}) and ({seq_b}, {kind_b:?}) collide",
            );
        }
    }

    for (sequence, kind, bytes, meta) in &generations {
        let handle = store
            .put_layer_checkpoint(object_id, *sequence, bytes.clone(), meta.clone())
            .await
            .unwrap_or_else(|err| {
                panic!("put_layer_checkpoint({sequence}, {kind:?}) must succeed, got {err:?}")
            });
        assert_eq!(
            (handle.seq(), handle.kind),
            (*sequence, *kind),
            "the returned handle must name the generation that was put",
        );
    }

    // ---- Assertion 1: the pinned mixed-kind order ------------------------
    //
    // Spelled as a literal on purpose. Deriving it by calling
    // `generation_sort_key` would make the case agree with whatever that
    // function does, including after someone changes the tie-break.
    let expected_order: Vec<(u64, CheckpointKind)> = vec![
        (9, CheckpointKind::Tail),
        (7, CheckpointKind::Base),
        (5, CheckpointKind::Base),
        (5, CheckpointKind::Tail),
    ];

    let listed = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints succeeds");
    assert_eq!(
        kind_seq_pairs(&listed),
        expected_order,
        "list_layer_checkpoints must emit ONE strict descending order over the sequence with \
         Base before Tail at an equal sequence — `generation_sort_key(sequence, kind) -> \
         (Reverse(sequence), 0 for Base / 1 for Tail)` in src/checkpoint/adapters/mod.rs is the \
         single source of truth, and every adapter MUST realize exactly it. Two segregated runs \
         (all bases, then all tails), or Tail-before-Base at sequence 5, are the two failure \
         shapes this pins. Got {listed:?}",
    );

    // ---- Assertion 2: adapter keys are present and pairwise distinct -----
    for (i, handle) in listed.iter().enumerate() {
        assert!(
            !handle.adapter_key.is_empty(),
            "every listed handle must carry a non-empty adapter_key; ({}, {:?}) has none",
            handle.seq(),
            handle.kind,
        );
        for other in listed.iter().skip(i + 1) {
            assert_ne!(
                handle.adapter_key,
                other.adapter_key,
                "listed handles must have pairwise-distinct adapter_keys: ({}, {:?}) and \
                 ({}, {:?}) share {:?}. A shared key between a base and a tail at the same \
                 sequence is precisely the collision the kind-qualified stem \
                 (`{{oid}}.{{to_seq}}.tail` alongside the base stem) exists to prevent — aliased \
                 keys make the two generations overwrite and delete each other",
                handle.seq(),
                handle.kind,
                other.seq(),
                other.kind,
                handle.adapter_key,
            );
        }
    }

    // ---- Assertion 3: every listed handle fetches its own artifact -------
    for handle in &listed {
        let (_, _, expected_bytes, _) = generations
            .iter()
            .find(|(sequence, kind, _, _)| *sequence == handle.seq() && *kind == handle.kind)
            .unwrap_or_else(|| {
                panic!("the listing reported a generation that was never put: {handle:?}")
            });

        let blob = store
            .get_layer_checkpoint(handle)
            .await
            .unwrap_or_else(|err| {
                panic!(
                    "get_layer_checkpoint on a listed handle ({}, {:?}) must succeed, got {err:?}",
                    handle.seq(),
                    handle.kind,
                )
            })
            .unwrap_or_else(|| {
                panic!(
                    "a handle the adapter itself minted MUST resolve: ({}, {:?})",
                    handle.seq(),
                    handle.kind,
                )
            });

        assert_eq!(
            blob.bytes,
            *expected_bytes,
            "the ({}, {:?}) handle must fetch the bytes originally put for that (sequence, kind) \
             — an ordering contract over handles that resolve to the wrong artifacts is worthless",
            handle.seq(),
            handle.kind,
        );
        assert_eq!(
            (blob.handle.seq(), blob.handle.kind),
            (handle.seq(), handle.kind),
            "and the returned blob must name the addressed generation",
        );
        assert_eq!(
            blob.meta.kind, handle.kind,
            "meta.kind must agree with the handle's kind; a tail that reads back as a base is a \
             full-state snapshot in the eyes of every restore path",
        );
    }

    // ---- Assertion 4: ordering is stable across re-calls -----------------
    let listed_again = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("repeat list_layer_checkpoints succeeds");
    assert_eq!(
        kind_seq_pairs(&listed_again),
        expected_order,
        "list_layer_checkpoints ordering must be stable across successive read-only calls — a \
         backend enumeration order leaking through would make the head of the list, and \
         therefore get_latest_layer_checkpoint, non-deterministic. Got {listed_again:?}",
    );

    // ---- Assertion 5: get_latest agrees with list().first(), tail included
    let latest = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest_layer_checkpoint succeeds")
        .expect("the object holds four generations");
    let (_, _, tail_9_bytes, _) = generations
        .iter()
        .find(|(sequence, kind, _, _)| (*sequence, *kind) == (9, CheckpointKind::Tail))
        .expect("tail@9 is in the fixture");
    assert_eq!(
        (latest.handle.seq(), latest.handle.kind),
        (9, CheckpointKind::Tail),
        "get_latest_layer_checkpoint is retained as list_layer_checkpoints().first() and MUST be \
         able to return a Tail. Filtering it to bases adapter-side is the tempting and wrong \
         change: restore-side \"newest base wins\" is Task 5's recovery hook (list -> newest Base \
         handle -> get_layer_checkpoint), implemented once there rather than three times in the \
         adapters where it can only be tested three times. A Base at 7 here means an adapter \
         filtered the tail out",
    );
    assert_eq!(
        latest.bytes, *tail_9_bytes,
        "and it must carry the tail@9 envelope's bytes verbatim",
    );
    assert_eq!(
        (latest.handle.seq(), latest.handle.kind),
        (listed[0].seq(), listed[0].kind),
        "get_latest_layer_checkpoint must name the identical generation as list(..).first()",
    );

    // ---- Assertion 6: the tie-break's practical consequence --------------
    //
    // A second, fresh store holding ONLY the same-sequence pair. Here the
    // Base-before-Tail tie-break is the entire content of the answer: whichever
    // generation the adapter puts first is what every kind-blind caller of
    // `get_latest_layer_checkpoint` restores.
    let (tie_store, _tie_guard) = factory().await;

    let tie_object_id: u64 = 30_130;
    let tie_seq: u64 = 5;
    let tie_tail_bytes = sample_tail_envelope(0x3055);
    let tie_base_bytes = sample_layer_ompf(0x3155);

    // Tail first, so a first-write-wins adapter would answer with the tail.
    tie_store
        .put_layer_checkpoint(
            tie_object_id,
            tie_seq,
            tie_tail_bytes.clone(),
            sample_tail_meta(2, 2, tie_seq),
        )
        .await
        .expect("put_layer_checkpoint (tail@5) on the second store succeeds");
    tie_store
        .put_layer_checkpoint(
            tie_object_id,
            tie_seq,
            tie_base_bytes.clone(),
            sample_meta(tie_seq),
        )
        .await
        .expect("put_layer_checkpoint (base@5) on the second store succeeds");

    let tie_listed = tie_store
        .list_layer_checkpoints(tie_object_id)
        .await
        .expect("list_layer_checkpoints on the second store succeeds");
    assert_eq!(
        kind_seq_pairs(&tie_listed),
        vec![
            (tie_seq, CheckpointKind::Base),
            (tie_seq, CheckpointKind::Tail)
        ],
        "a store holding only base@{tie_seq} and tail@{tie_seq} must list Base first — the \
         tie-break in isolation, with no sequence component to carry the order; got {tie_listed:?}",
    );

    let tie_latest = tie_store
        .get_latest_layer_checkpoint(tie_object_id)
        .await
        .expect("get_latest_layer_checkpoint on the second store succeeds")
        .expect("the second store holds two generations");
    assert_eq!(
        tie_latest.handle.kind,
        CheckpointKind::Base,
        "at an equal sequence get_latest_layer_checkpoint MUST return the Base: it is the \
         artifact that restores on its own, which is why Base wins the tie for every kind-blind \
         caller. Returning the tail hands such a caller a delta envelope to restore as a full \
         snapshot",
    );
    assert_eq!(
        tie_latest.bytes, tie_base_bytes,
        "and it must carry the base's OMPF bytes, not the tail envelope's",
    );
    assert_eq!(
        (tie_latest.handle.seq(), tie_latest.handle.kind),
        (tie_listed[0].seq(), tie_listed[0].kind),
        "get_latest_layer_checkpoint must agree with list(..).first() on the same-sequence store \
         too",
    );
}

/// Case #31 — deleting a tail is **idempotent on absent handles** and
/// **kind-isolated** at a shared sequence (SR-2.4).
///
/// Two failure shapes, one case, because they are the two halves of the same
/// delete contract and share a fixture.
///
/// 1. **Idempotency.** `delete_layer_checkpoint` on a missing handle is
///    `Ok(())`, NEVER [`CheckpointError::NotFound`] — case #14's retention-prune
///    contract, which tails inherit rather than re-negotiate. The pruner
///    re-issues deletes over a prune set it computed from an earlier listing, so
///    "already gone" is its ordinary case, not its error case.
/// 2. **Kind isolation.** Before the kind branch existed in the delete paths, a
///    tail handle at `to_seq == N` derived the *kind-blind* addresses
///    (`ompf_path` / `layer_ompf_url` and companions) and would have unlinked the
///    BASE triplet at `snap_seq == N`. That is the stranded-generation data-loss
///    shape SR-2.4 forbids: the surviving tail names a base that no longer
///    exists, so the lineage restores to nothing.
///
/// **The collision is real, not hypothetical.** SR-4.1 re-base mints a fresh
/// `base@S` where `S` is the newest consumed tail's `to_seq`, so `base@S` and
/// `tail@to_seq == S` genuinely coexist until retention removes one — the same
/// fixture shape case #30 orders. That makes this the load-bearing case of the
/// delete group, and Phase 2 covers **both** directions: a tail delete must
/// spare the base, and a base delete must spare the tail. Only checking the
/// first direction would pass on an adapter that deletes the base's triplet
/// *and* the tail's pair on either handle.
///
/// **Delete order is the mirror of meta-first publish** (artifact, then meta),
/// so a partial failure strands a meta companion rather than an artifact without
/// one. That is invisible at the trait surface *unless* it resurrects: the
/// assertions therefore re-read the listing after the last delete, since an
/// orphaned `.tail.meta.json` that the enumeration still parses would
/// re-materialize a phantom handle whose `get` then finds no bytes — a
/// generation that exists to a lineage gather and to the retention pruner but
/// restores to nothing.
///
/// Phase 3 pins the neighbour-sequence claim the Epic-5 retention prune is built
/// on: deleting one generation leaves every *other* sequence's artifacts intact
/// and byte-identical, so a prune pass that walks a list of handles cannot
/// collateral-damage the generations it chose to keep.
pub async fn case_31_delete_tail_is_idempotent_and_kind_isolated<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    /// `Ok(())`, never `NotFound` — case #14's delete-path match shape, applied
    /// to every delete this case issues (present or absent alike, since a
    /// present-handle delete that answers `NotFound` is the same defect).
    async fn assert_delete_ok(
        store: &Arc<dyn CheckpointStore>,
        handle: CheckpointHandle,
        why: &str,
    ) {
        let addressed = format!(
            "object_id={}, seq={}, kind={:?}",
            handle.object_id,
            handle.seq(),
            handle.kind,
        );
        match store.delete_layer_checkpoint(handle).await {
            Ok(()) => {} // contract-satisfying outcome
            Err(CheckpointError::NotFound) => panic!(
                "case_31: delete_layer_checkpoint({addressed}) must return Ok(()), NOT \
                 CheckpointError::NotFound — {why}. The trait's idempotent-on-missing contract \
                 exists because the retention pruner deletes over a prune set computed from an \
                 earlier listing, so 'already gone' is its ordinary case",
            ),
            Err(err) => panic!(
                "case_31: delete_layer_checkpoint({addressed}) must succeed — {why}; got \
                 unexpected error: {err:?}",
            ),
        }
    }

    /// `Ok(None)`, never `NotFound`, never a blob — cases #12 / #14 / #21 / #29's
    /// absent-handle match shape on the get path.
    async fn assert_absent(store: &Arc<dyn CheckpointStore>, handle: &CheckpointHandle, why: &str) {
        match store.get_layer_checkpoint(handle).await {
            Ok(None) => {}
            Ok(Some(blob)) => panic!(
                "case_31: get_layer_checkpoint(object_id={}, seq={}, kind={:?}) must be \
                 Ok(None) — {why}; got Some(handle={:?}, {} bytes)",
                handle.object_id,
                handle.seq(),
                handle.kind,
                blob.handle,
                blob.bytes.len(),
            ),
            Err(err) => {
                assert!(
                    !err.is_not_found(),
                    "case_31: get_layer_checkpoint must NEVER surface NotFound for a deleted \
                     generation — {why}. Absence is Ok(None) per the trait's get-path rule, and \
                     a concurrent prune makes it an ordinary outcome. Got: {err:?}",
                );
                panic!(
                    "case_31: get_layer_checkpoint on a deleted generation must be Ok(None) — \
                     {why}; got unexpected error: {err:?}",
                );
            }
        }
    }

    /// Fetch a generation that MUST still be there, failing with the survival
    /// claim rather than an anonymous unwrap panic.
    async fn fetch_survivor(
        store: &Arc<dyn CheckpointStore>,
        handle: &CheckpointHandle,
        why: &str,
    ) -> live_edit::checkpoint::CheckpointBlob {
        match store.get_layer_checkpoint(handle).await {
            Ok(Some(blob)) => blob,
            Ok(None) => panic!(
                "case_31: get_layer_checkpoint(object_id={}, seq={}, kind={:?}) must return \
                 Some — {why}. Ok(None) here means the delete took out a generation it does not \
                 address: the stranded-generation data-loss shape SR-2.4 forbids",
                handle.object_id,
                handle.seq(),
                handle.kind,
            ),
            Err(err) => panic!(
                "case_31: get_layer_checkpoint(object_id={}, seq={}, kind={:?}) must succeed — \
                 {why}; got {err:?}",
                handle.object_id,
                handle.seq(),
                handle.kind,
            ),
        }
    }

    /// Seed `base@seq` and `tail@seq` on one fresh `object_id`, returning
    /// `((base_handle, base_bytes), (tail_handle, tail_bytes))`.
    ///
    /// The tail is rooted at an OLDER base (`base_snap_seq == from_seq ==
    /// [`TAIL_ROOT_SEQ`]`), which is what the SR-4.1 re-base fixture looks like:
    /// the tail covering `(4, 11]` was consumed into a fresh `base@11`, and the
    /// two coexist until retention removes one.
    async fn seed_same_sequence_pair(
        store: &Arc<dyn CheckpointStore>,
        object_id: u64,
        seq: u64,
        tag: u64,
    ) -> ((CheckpointHandle, Bytes), (CheckpointHandle, Bytes)) {
        /// The pre-re-base base the tail replays onto — deliberately BELOW the
        /// shared sequence, so a tail whose lineage read back as `Some(seq)`
        /// would be visibly wrong.
        const TAIL_ROOT_SEQ: u64 = 4;

        let base_bytes = sample_layer_ompf(tag);
        let tail_bytes = sample_tail_envelope(tag);
        assert_ne!(
            base_bytes, tail_bytes,
            "test precondition: the two generations sharing sequence {seq} must carry distinct \
             bytes, or 'the survivor is the OTHER kind's artifact' passes vacuously",
        );

        let base_handle = store
            .put_layer_checkpoint(object_id, seq, base_bytes.clone(), sample_meta(seq))
            .await
            .unwrap_or_else(|err| panic!("put base@{seq} on object {object_id} succeeds: {err:?}"));
        assert_eq!(base_handle.kind, CheckpointKind::Base);

        let tail_handle = store
            .put_layer_checkpoint(
                object_id,
                seq,
                tail_bytes.clone(),
                sample_tail_meta(TAIL_ROOT_SEQ, TAIL_ROOT_SEQ, seq),
            )
            .await
            .unwrap_or_else(|err| panic!("put tail@{seq} on object {object_id} succeeds: {err:?}"));
        assert_eq!(tail_handle.kind, CheckpointKind::Tail);
        assert_ne!(
            base_handle.adapter_key, tail_handle.adapter_key,
            "test precondition: the two generations must address distinct artifacts — a shared \
             adapter_key makes them alias each other and every isolation claim below vacuous",
        );

        let listed = store
            .list_layer_checkpoints(object_id)
            .await
            .expect("list_layer_checkpoints on the seeded pair succeeds");
        assert_eq!(
            kind_seq_pairs(&listed),
            vec![(seq, CheckpointKind::Base), (seq, CheckpointKind::Tail)],
            "test precondition: both generations must be present at sequence {seq} before a \
             delete can be shown to remove exactly one; got {listed:?}",
        );

        ((base_handle, base_bytes), (tail_handle, tail_bytes))
    }

    let (store, _guard) = factory().await;

    // ---- Phase 1: idempotency on absent tail handles ---------------------
    //
    // Tails inherit case #14's contract unchanged. The three flavours of
    // "missing" are: an object that was never written, a live object at a
    // sequence that was never written, and a generation that WAS written and
    // has already been deleted (the pruner's actual re-issue case).

    // Flavour 1 — an object_id that has never been written. The adapter_key is
    // deliberately junk: adapters address off (object_id, seq, kind).
    assert_delete_ok(
        &store,
        CheckpointHandle::tail(
            0xFFFF_FFFF_FFFF_FFFB,
            1,
            "phantom-conformance-case-31".to_string(),
        ),
        "no generation of any kind was ever written for that object_id",
    )
    .await;

    // Flavour 2 — a real object at a sequence that holds nothing. The real
    // generation must survive: a spurious delete may not widen into a
    // neighbouring sequence.
    let phase1_object_id: u64 = 31_031;
    let phase1_to_seq: u64 = 7;
    let phase1_bytes = sample_tail_envelope(0x3107);
    let phase1_tail = store
        .put_layer_checkpoint(
            phase1_object_id,
            phase1_to_seq,
            phase1_bytes.clone(),
            sample_tail_meta(3, 3, phase1_to_seq),
        )
        .await
        .expect("put tail@7 succeeds");

    assert_delete_ok(
        &store,
        CheckpointHandle::tail(
            phase1_object_id,
            999_999,
            format!("phantom:{phase1_object_id}:999999:tail"),
        ),
        "the object exists but holds no generation at that sequence",
    )
    .await;

    let survivor = fetch_survivor(
        &store,
        &phase1_tail,
        "a delete addressed at an absent sequence must not collateral-damage the object's real \
         generations",
    )
    .await;
    assert_eq!(
        survivor.bytes, phase1_bytes,
        "and the surviving tail's envelope must be byte-identical — a spurious delete that \
         truncated or re-wrote it is data loss the listing alone would not reveal",
    );
    assert_eq!(survivor.meta.kind, CheckpointKind::Tail);

    // Flavour 3 — the pruner's real re-issue case: delete a generation that
    // exists, then delete the identical handle again.
    assert_delete_ok(
        &store,
        phase1_tail.clone(),
        "deleting a tail that exists must succeed",
    )
    .await;
    assert_delete_ok(
        &store,
        phase1_tail.clone(),
        "re-deleting the same tail handle is the retention pruner's ordinary repeat — it deletes \
         over a prune set computed from an earlier listing, so a second pass re-issues deletes \
         for generations a concurrent pruner already removed",
    )
    .await;
    assert_absent(
        &store,
        &phase1_tail,
        "the tail was deleted, so the get path reports absence",
    )
    .await;
    let phase1_listed = store
        .list_layer_checkpoints(phase1_object_id)
        .await
        .expect("list_layer_checkpoints after deleting the object's only tail succeeds");
    assert!(
        phase1_listed.is_empty(),
        "an object whose only generation was deleted must list empty; a non-empty listing here \
         is a residual companion (the orphaned `.tail.meta.json` the artifact-then-meta delete \
         order can strand) being parsed back into a phantom generation. Got {phase1_listed:?}",
    );

    // ---- Phase 2: mixed-kind isolation at ONE shared sequence -------------
    //
    // The load-bearing phase. `base@11` and `tail@11` coexist (SR-4.1 re-base),
    // and each delete must remove exactly the addressed kind. Both directions
    // are covered on two fresh object_ids, because an adapter that deletes both
    // families on EITHER handle passes a one-direction check.
    let phase2_seq: u64 = 11;

    // ---- Direction A: delete the tail, the base survives ------------------
    let tail_first_object_id: u64 = 31_111;
    // Only the SURVIVOR's bytes are named here — the deleted generation's are
    // asserted through its absence, and the seed helper has already pinned that
    // the two payloads differ.
    let ((base_a, base_a_bytes), (tail_a, _)) =
        seed_same_sequence_pair(&store, tail_first_object_id, phase2_seq, 0x3111).await;

    assert_delete_ok(
        &store,
        tail_a.clone(),
        "deleting the tail at a sequence shared with a base must succeed",
    )
    .await;

    let surviving_base = fetch_survivor(
        &store,
        &base_a,
        "THE property of this phase: the tail's delete addresses `{oid}.{to_seq}.tail` and its \
         meta companion ONLY. An adapter that fell through to the kind-blind base addresses \
         would have unlinked this base's triplet, leaving a lineage that names a base which no \
         longer exists",
    )
    .await;
    assert_eq!(
        surviving_base.bytes, base_a_bytes,
        "the surviving base must hold its OWN OMPF bytes verbatim; a payload whose first bytes \
         read as the TAIL discriminator means the delete crossed the kinds and the read served \
         the wrong artifact",
    );
    assert_eq!(
        surviving_base.meta.kind,
        CheckpointKind::Base,
        "and it must still read back as a Base — a survivor relabelled Tail is replayed as a \
         delta envelope by every restore path",
    );
    assert_eq!(
        (
            surviving_base.meta.base_snap_seq,
            surviving_base.meta.from_seq
        ),
        (None, None),
        "a base carries no lineage; the tail's delete must not have leaked the tail's lineage \
         fields onto it",
    );

    assert_absent(
        &store,
        &tail_a,
        "the tail at the shared sequence was the deleted generation",
    )
    .await;
    let listed_a = store
        .list_layer_checkpoints(tail_first_object_id)
        .await
        .expect("list_layer_checkpoints after the tail delete succeeds");
    assert_eq!(
        kind_seq_pairs(&listed_a),
        vec![(phase2_seq, CheckpointKind::Base)],
        "exactly one generation must remain at the shared sequence, and it must be the Base; \
         got {listed_a:?}",
    );
    let latest_a = store
        .get_latest_layer_checkpoint(tail_first_object_id)
        .await
        .expect("get_latest_layer_checkpoint after the tail delete succeeds")
        .expect("the base survives the tail's delete");
    assert_eq!(
        (latest_a.handle.seq(), latest_a.handle.kind),
        (phase2_seq, CheckpointKind::Base),
    );
    assert_eq!(
        latest_a.bytes, base_a_bytes,
        "get_latest must serve the surviving base's bytes — the kind-blind restore path reaches \
         this generation through exactly this call",
    );

    // ---- Direction B: delete the base, the tail survives ------------------
    let base_first_object_id: u64 = 31_112;
    let ((base_b, _), (tail_b, tail_b_bytes)) =
        seed_same_sequence_pair(&store, base_first_object_id, phase2_seq, 0x3112).await;

    assert_delete_ok(
        &store,
        base_b.clone(),
        "deleting the base at a sequence shared with a tail must succeed",
    )
    .await;

    let surviving_tail = fetch_survivor(
        &store,
        &tail_b,
        "the mirror claim: a base's delete touches only the base triplet, so the tail sharing \
         the sequence survives. An adapter that swept the whole sequence would lose the newest \
         captured commits with no error anywhere",
    )
    .await;
    assert_eq!(
        surviving_tail.bytes, tail_b_bytes,
        "the surviving tail's envelope must be byte-identical (SR-1.5: opaque payload)",
    );
    assert_eq!(
        surviving_tail.meta.kind,
        CheckpointKind::Tail,
        "and it must still read back as a Tail",
    );
    assert_eq!(
        (
            surviving_tail.meta.base_snap_seq,
            surviving_tail.meta.from_seq,
            surviving_tail.meta.snap_seq,
        ),
        (Some(4), Some(4), Some(phase2_seq)),
        "with its lineage intact: the base delete must not have disturbed the surviving tail's \
         meta companion, which is the only record of which base this envelope replays onto",
    );

    assert_absent(
        &store,
        &base_b,
        "the base at the shared sequence was the deleted generation",
    )
    .await;
    let listed_b = store
        .list_layer_checkpoints(base_first_object_id)
        .await
        .expect("list_layer_checkpoints after the base delete succeeds");
    assert_eq!(
        kind_seq_pairs(&listed_b),
        vec![(phase2_seq, CheckpointKind::Tail)],
        "exactly one generation must remain at the shared sequence, and it must be the Tail; \
         got {listed_b:?}",
    );
    let latest_b = store
        .get_latest_layer_checkpoint(base_first_object_id)
        .await
        .expect("get_latest_layer_checkpoint after the base delete succeeds")
        .expect("the tail survives the base's delete");
    assert_eq!(
        (latest_b.handle.seq(), latest_b.handle.kind),
        (phase2_seq, CheckpointKind::Tail),
        "get_latest is list(..).first() and MUST be able to name a tail — with the base gone the \
         tail is the only generation there is",
    );
    assert_eq!(latest_b.bytes, tail_b_bytes);

    // ---- Phase 2 close-out: delete each survivor, assert no residue -------
    //
    // Meta-first publish makes an orphaned meta companion the realistic
    // residual of a partial delete, and a listing that parses one back into a
    // handle is worse than a leaked file: the phantom generation is offered to
    // the lineage gather and to the pruner, and resolves to no bytes. Re-read
    // the listing after the last delete on each object to catch exactly that.
    for (object_id, survivor_handle, other_handle, label) in [
        (
            tail_first_object_id,
            base_a.clone(),
            tail_a.clone(),
            "the base that survived the tail's delete",
        ),
        (
            base_first_object_id,
            tail_b.clone(),
            base_b.clone(),
            "the tail that survived the base's delete",
        ),
    ] {
        assert_delete_ok(
            &store,
            survivor_handle.clone(),
            &format!("deleting {label} must succeed"),
        )
        .await;

        assert_absent(&store, &survivor_handle, "it was just deleted").await;
        assert_absent(
            &store,
            &other_handle,
            "and the generation deleted earlier must stay absent — a second delete at the same \
             sequence must not resurrect it",
        )
        .await;

        let listed = store
            .list_layer_checkpoints(object_id)
            .await
            .expect("list_layer_checkpoints after deleting both generations succeeds");
        assert!(
            listed.is_empty(),
            "no stranded half-generation may remain once both kinds at sequence {phase2_seq} are \
             deleted: an orphaned `.tail.meta.json` (or `.meta.json`) that the enumeration still \
             parses re-materializes a phantom handle whose get finds no bytes. Got {listed:?}",
        );

        let latest = store
            .get_latest_layer_checkpoint(object_id)
            .await
            .expect("get_latest_layer_checkpoint on the emptied object succeeds");
        assert!(
            latest.is_none(),
            "and get_latest_layer_checkpoint must be Ok(None) on an object whose every \
             generation is gone; got {latest:?}",
        );
    }

    // ---- Phase 3: a delete does not disturb neighbouring sequences --------
    //
    // The shape Epic 5's retention prune is built on: it walks a listing and
    // deletes a subset, so a delete that widened past its own sequence would
    // destroy the generations the prune deliberately kept.
    let phase3_object_id: u64 = 31_140;
    let base_20_bytes = sample_layer_ompf(0x3120);
    let tail_25_bytes = sample_tail_envelope(0x3125);
    let base_30_bytes = sample_layer_ompf(0x3130);

    let base_20 = store
        .put_layer_checkpoint(phase3_object_id, 20, base_20_bytes.clone(), sample_meta(20))
        .await
        .expect("put base@20 succeeds");
    let tail_25 = store
        .put_layer_checkpoint(
            phase3_object_id,
            25,
            tail_25_bytes.clone(),
            // Rooted at base@20, covering (20, 25].
            sample_tail_meta(20, 20, 25),
        )
        .await
        .expect("put tail@25 succeeds");
    let base_30 = store
        .put_layer_checkpoint(phase3_object_id, 30, base_30_bytes.clone(), sample_meta(30))
        .await
        .expect("put base@30 succeeds");

    assert_delete_ok(
        &store,
        tail_25.clone(),
        "deleting the middle generation must succeed",
    )
    .await;

    let listed_phase3 = store
        .list_layer_checkpoints(phase3_object_id)
        .await
        .expect("list_layer_checkpoints after deleting the middle generation succeeds");
    assert_eq!(
        kind_seq_pairs(&listed_phase3),
        vec![(30, CheckpointKind::Base), (20, CheckpointKind::Base)],
        "deleting tail@25 must remove exactly that generation, leaving the neighbouring \
         sequences in the same newest-first order; got {listed_phase3:?}",
    );
    assert_absent(&store, &tail_25, "tail@25 was the deleted generation").await;

    for (handle, expected_bytes, label) in [
        (&base_30, &base_30_bytes, "base@30 (above the deleted tail)"),
        (&base_20, &base_20_bytes, "base@20 (below the deleted tail)"),
    ] {
        let blob = fetch_survivor(
            &store,
            handle,
            &format!("{label} is a neighbouring sequence the delete does not address"),
        )
        .await;
        assert_eq!(
            blob.bytes, *expected_bytes,
            "{label} must round-trip byte-identically after the neighbouring delete — a prune \
             that leaves a generation enumerable but corrupt is worse than one that removes it",
        );
        assert_eq!(blob.meta.kind, CheckpointKind::Base);
    }
}

/// Case #32 — every `validate_tail_put` rejection is `Permanent` AND writes
/// nothing, identically across all three adapters.
///
/// **Where the rule lives.** Once, in `validate_tail_put`
/// (`src/checkpoint/adapters/mod.rs`); all three adapters call it as the first
/// act of `put_layer_checkpoint`, ahead of any address derivation, file write,
/// or outbound request. That single-copy structure is exactly why the pinning
/// belongs here rather than in three per-adapter unit tests: those can only
/// assert that each adapter calls *something*, while a conformance case asserts
/// the three agree on the observable outcome.
///
/// **The three rejected shapes** (all `CheckpointKind::Tail`):
///
/// 1. `sidecar_bytes: Some(_)` — SR-2.4 / b.s5y make "a tail carries no ext-refs
///    sidecar" MUST-level.
/// 2. `base_snap_seq: None`.
/// 3. `from_seq: None`.
///
/// Shapes 2 and 3 are **presence-only** validation. The range/ordering
/// invariants (`base_snap_seq <= from_seq <= to_seq`) are deliberately NOT
/// re-checked adapter-side — they belong to `tail_codec`, which validates them
/// against the envelope's own header. An adapter treats the envelope as opaque
/// bytes (SR-1.5) and so cannot cross-check meta against it; re-deriving the
/// rule here would put a second, weaker copy in four places. This case
/// therefore asserts nothing about ranges.
///
/// **Why the rejection is loud rather than a silent field drop.** Every
/// deletion path in LocalDisk and StorageApi is artifact-family-driven per
/// generation stem, and a tail's stem family has no sidecar member, so a
/// tail-authored `.ext-refs.json` is permanently unreapable on every adapter.
/// On LocalDisk — whose live-set walk is suffix-driven rather than
/// stem-driven — it would also be unioned into `list_referenced_ext_hashes`
/// forever, pinning that snapshot's whole hash set and degrading the retention
/// cascade toward deleting nothing. A caller that passes a tail sidecar has a
/// bug that would otherwise never surface. Hence the load-bearing half of this
/// case is not the error variant but the **absence of any write**; the
/// per-adapter divergence is spelled out at the `list_referenced_ext_hashes`
/// assertion below.
///
/// **Fresh store per shape.** Each of the three rejections runs against its own
/// `factory()` invocation, so "nothing was written" cannot be confounded by an
/// earlier shape's residue — with a shared store, a partial write from shape 1
/// would be indistinguishable from shape 2's clean rejection.
///
/// **Negative control** (fourth fresh store). Without it the case would pass on
/// an adapter that rejected *every* tail put, or that ignored `sidecar_bytes`
/// wholesale: the control asserts a `Base` put carrying `sidecar_bytes: Some`
/// still succeeds and its hashes DO reach `list_referenced_ext_hashes`, and that
/// a lineage-complete tail put on that same store succeeds and resolves.
pub async fn case_32_tail_put_validation_rejections_are_permanent_and_write_nothing<F, Fut>(
    factory: F,
) where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    use std::collections::HashSet;

    /// Ext hashes offered on the rejected shape-1 tail sidecar. Distinct from
    /// the negative control's set so neither half can borrow the other's
    /// evidence.
    const REJECTED_TAIL_HASHES: [&str; 2] = ["c32-tail-h1", "c32-tail-h2"];
    /// Ext hashes on the negative control's *base* sidecar — the set that MUST
    /// reach the live set.
    const CONTROL_BASE_HASHES: [&str; 2] = ["c32-base-h1", "c32-base-h2"];

    /// The lineage the tail fixtures name, and the sequence they cover.
    const BASE_SEQ: u64 = 4;
    const TO_SEQ: u64 = 9;

    /// The full four-way exclusive classification of a `validate_tail_put`
    /// rejection, spelled with cases #19 / #20 / #22's assertion shape.
    ///
    /// `Permanent` and not `Conflict` / `Transient`: the call is a caller bug,
    /// and re-issuing the identical call reproduces it exactly, so a
    /// misclassification as `Transient` would turn a deterministic bug into a
    /// retry storm and a misclassification as `Conflict` would send the caller
    /// off to bump `snap_seq` and try again forever.
    fn assert_permanent_rejection(err: &CheckpointError, shape: &str) {
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "case_32 [{shape}]: the variant returned must literally be \
             CheckpointError::Permanent {{ .. }} — validate_tail_put (the single centralized gate \
             in src/checkpoint/adapters/mod.rs that every adapter calls) returns exactly that; \
             got {err:?}",
        );
        assert!(
            err.is_permanent(),
            "case_32 [{shape}]: validate_tail_put's rejection must classify as permanent via \
             is_permanent(); the rule is centralized in src/checkpoint/adapters/mod.rs, so a \
             disagreement here is an adapter re-wrapping the shared error. Got {err:?}",
        );
        assert!(
            !err.is_transient(),
            "case_32 [{shape}]: a validate_tail_put rejection must NOT classify as transient — \
             retrying the identical call reproduces it exactly, so a transient classification \
             turns a caller bug into a retry storm. The rule is centralized in \
             src/checkpoint/adapters/mod.rs. Got {err:?}",
        );
        assert!(
            !err.is_not_found(),
            "case_32 [{shape}]: a validate_tail_put rejection must NOT classify as not-found — \
             nothing was looked up. The rule is centralized in src/checkpoint/adapters/mod.rs. \
             Got {err:?}",
        );
        assert!(
            !err.is_conflict(),
            "case_32 [{shape}]: a validate_tail_put rejection must NOT classify as conflict — no \
             competing writer is involved, and a conflict verdict would send the caller off to \
             bump snap_seq and retry a call that can never succeed. The rule is centralized in \
             src/checkpoint/adapters/mod.rs. Got {err:?}",
        );
    }

    /// Every trait-surface read that could reveal a partial write for the
    /// rejected `(object_id, TO_SEQ, Tail)` generation, on a store whose ONLY
    /// put was the rejected one.
    async fn assert_nothing_written(
        store: &Arc<dyn CheckpointStore>,
        object_id: u64,
        forbidden_hashes: &[&str],
        shape: &str,
    ) {
        let listed = store
            .list_layer_checkpoints(object_id)
            .await
            .unwrap_or_else(|err| {
                panic!("case_32 [{shape}]: list_layer_checkpoints must succeed; got {err:?}")
            });
        assert!(
            listed.is_empty(),
            "case_32 [{shape}]: a validate_tail_put rejection MUST leave the object with no \
             generations — the gate runs before any artifact is published, so a listed \
             generation means the adapter wrote first and validated second. Got {listed:?}",
        );

        let tail_handle =
            CheckpointHandle::tail(object_id, TO_SEQ, "case-32-rejected-tail".to_string());
        match store.get_layer_checkpoint(&tail_handle).await {
            Ok(None) => {}
            Ok(Some(blob)) => panic!(
                "case_32 [{shape}]: get_layer_checkpoint on the rejected tail's address must be \
                 Ok(None); got Some({} bytes, handle={:?}) — the rejected put reached storage",
                blob.bytes.len(),
                blob.handle,
            ),
            Err(err) => panic!(
                "case_32 [{shape}]: get_layer_checkpoint on the rejected tail's address must be \
                 Ok(None); got {err:?}",
            ),
        }

        match store.get_latest_layer_checkpoint(object_id).await {
            Ok(None) => {}
            Ok(Some(blob)) => panic!(
                "case_32 [{shape}]: get_latest_layer_checkpoint must be Ok(None) on an object \
                 whose only put was rejected; got Some(handle={:?})",
                blob.handle,
            ),
            Err(err) => panic!(
                "case_32 [{shape}]: get_latest_layer_checkpoint must be Ok(None); got {err:?}",
            ),
        }

        let object_ids = store.list_all_object_ids().await.unwrap_or_else(|err| {
            panic!("case_32 [{shape}]: list_all_object_ids must succeed; got {err:?}")
        });
        assert!(
            !object_ids.contains(&object_id),
            "case_32 [{shape}]: list_all_object_ids must NOT report an object whose only put was \
             rejected by validate_tail_put; got {object_ids:?}",
        );

        let refs = store
            .list_referenced_ext_hashes()
            .await
            .unwrap_or_else(|err| {
                panic!("case_32 [{shape}]: list_referenced_ext_hashes must succeed; got {err:?}")
            });
        // LOAD-BEARING — do not simplify away. This is the whole reason
        // validate_tail_put rejects loudly instead of silently dropping the
        // field. The deletion paths are artifact-family-driven per generation
        // stem and a tail's family has no sidecar member, so a sidecar honoured
        // under a tail stem is UNREAPABLE on every adapter. What it then does
        // to the live set diverges — and both branches are bad:
        //
        // - LocalDisk's walk is suffix-driven (every `*.ext-refs.json` under
        //   `layers/`), so it WOULD union the unreapable sidecar: that
        //   snapshot's whole hash set is pinned live forever, GC can never
        //   collect those blobs, and the retention cascade degrades toward
        //   deleting nothing.
        // - InMemory (base slot only) and StorageApi (`parse_ext_refs_filename`
        //   rejects the extra `.tail` stem component) would NOT see it at all,
        //   leaving a permanent invisible artifact and a silent cross-adapter
        //   divergence in what the live set means.
        //
        // Asserting absence is what pins the rejection to the write path on
        // every adapter at once: a rejection that still wrote the sidecar looks
        // correct on every other assertion in this case.
        for hash in forbidden_hashes {
            assert!(
                !refs.contains(*hash),
                "case_32 [{shape}]: list_referenced_ext_hashes must NOT contain {hash:?} — the \
                 rejected tail's sidecar must never have been written. An ext-refs sidecar under \
                 a tail stem is never reaped (the deletion paths are artifact-family-driven and a \
                 tail's family has no sidecar member) yet is still unioned into the live set, so \
                 it pins those hashes forever and degrades the retention cascade toward deleting \
                 nothing. Got {refs:?}",
            );
        }
        assert!(
            refs.is_empty(),
            "case_32 [{shape}]: and the live ext-hash set must be entirely empty — this store's \
             only put was the rejected one, so ANY referenced hash is residue from a write that \
             validate_tail_put was supposed to have prevented. Got {refs:?}",
        );
    }

    // ---- Shape 1: a Tail carrying an ext-refs sidecar ----------------------
    {
        let (store, _guard) = factory().await;
        let object_id: u64 = 32_001;

        let mut meta = sample_tail_meta(BASE_SEQ, BASE_SEQ, TO_SEQ);
        meta.sidecar_bytes = Some(ext_refs_sidecar_for(&REJECTED_TAIL_HASHES));

        let err = store
            .put_layer_checkpoint(object_id, TO_SEQ, sample_tail_envelope(0x3201), meta)
            .await
            .expect_err(
                "case_32 [tail with sidecar_bytes]: put_layer_checkpoint must be rejected by \
                 validate_tail_put — the no-sidecar-on-a-tail rule is MUST-level (SR-2.4) and is \
                 centralized in src/checkpoint/adapters/mod.rs, so an Ok here means this adapter \
                 either skipped the shared gate or silently dropped the field",
            );
        assert_permanent_rejection(&err, "tail with sidecar_bytes");
        assert_nothing_written(
            &store,
            object_id,
            &REJECTED_TAIL_HASHES,
            "tail with sidecar_bytes",
        )
        .await;
    }

    // ---- Shape 2: a Tail missing base_snap_seq -----------------------------
    {
        let (store, _guard) = factory().await;
        let object_id: u64 = 32_002;

        let mut meta = sample_tail_meta(BASE_SEQ, BASE_SEQ, TO_SEQ);
        meta.base_snap_seq = None;

        let err = store
            .put_layer_checkpoint(object_id, TO_SEQ, sample_tail_envelope(0x3202), meta)
            .await
            .expect_err(
                "case_32 [tail missing base_snap_seq]: put_layer_checkpoint must be rejected by \
                 validate_tail_put — a tail that names no base would force replay to INFER a \
                 lineage, silently producing a wrong restore instead of a loud failure. The rule \
                 is centralized in src/checkpoint/adapters/mod.rs",
            );
        assert_permanent_rejection(&err, "tail missing base_snap_seq");
        assert_nothing_written(&store, object_id, &[], "tail missing base_snap_seq").await;
    }

    // ---- Shape 3: a Tail missing from_seq ----------------------------------
    {
        let (store, _guard) = factory().await;
        let object_id: u64 = 32_003;

        let mut meta = sample_tail_meta(BASE_SEQ, BASE_SEQ, TO_SEQ);
        meta.from_seq = None;

        let err = store
            .put_layer_checkpoint(object_id, TO_SEQ, sample_tail_envelope(0x3203), meta)
            .await
            .expect_err(
                "case_32 [tail missing from_seq]: put_layer_checkpoint must be rejected by \
                 validate_tail_put — presence of BOTH lineage fields is the tail put's \
                 precondition, and the rule is centralized in src/checkpoint/adapters/mod.rs",
            );
        assert_permanent_rejection(&err, "tail missing from_seq");
        assert_nothing_written(&store, object_id, &[], "tail missing from_seq").await;
    }

    // ---- Negative control --------------------------------------------------
    //
    // Proves the three rejections above are specific to validate_tail_put's
    // gate rather than an adapter that rejects tail puts (or sidecars) in
    // general. Without this, an adapter that returned Permanent from EVERY
    // put_layer_checkpoint would pass every assertion above.
    {
        let (store, _guard) = factory().await;
        let object_id: u64 = 32_004;

        // A BASE carrying a sidecar — the shape validate_tail_put returns
        // Ok(()) for immediately, and case #23's fixture shape.
        let mut base_meta = sample_meta(BASE_SEQ);
        base_meta.sidecar_bytes = Some(ext_refs_sidecar_for(&CONTROL_BASE_HASHES));
        let base_bytes = sample_layer_ompf(0x3204);
        let base_handle = store
            .put_layer_checkpoint(object_id, BASE_SEQ, base_bytes.clone(), base_meta)
            .await
            .expect(
                "case_32 [negative control]: a BASE put carrying sidecar_bytes MUST still \
                 succeed — validate_tail_put returns Ok(()) for a Base meta unconditionally, so \
                 a failure here means the sidecar rejection was written too broadly and the \
                 ext-refs path is now broken for the kind that owns it",
            );
        assert_eq!(
            base_handle.kind,
            CheckpointKind::Base,
            "case_32 [negative control]: the control's base put must produce a Base handle",
        );

        let refs = store
            .list_referenced_ext_hashes()
            .await
            .expect("case_32 [negative control]: list_referenced_ext_hashes must succeed");
        let expected: HashSet<String> = CONTROL_BASE_HASHES.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            refs, expected,
            "case_32 [negative control]: a BASE's ext-refs sidecar MUST reach the live set — \
             this is the assertion that keeps the three rejections above from passing vacuously \
             on an adapter that ignores sidecar_bytes entirely",
        );

        // And a lineage-complete tail on the SAME store still succeeds: the
        // gate rejects three specific shapes, not the tail path.
        let tail_bytes = sample_tail_envelope(0x3205);
        let tail_handle = store
            .put_layer_checkpoint(
                object_id,
                TO_SEQ,
                tail_bytes.clone(),
                sample_tail_meta(BASE_SEQ, BASE_SEQ, TO_SEQ),
            )
            .await
            .expect(
                "case_32 [negative control]: a lineage-complete tail carrying no sidecar is \
                 exactly the shape validate_tail_put admits, so the put MUST succeed — a failure \
                 here means the gate rejects every tail and the three rejections above proved \
                 nothing",
            );
        assert_eq!(
            tail_handle.kind,
            CheckpointKind::Tail,
            "case_32 [negative control]: the admitted tail put must produce a Tail handle",
        );

        let got_tail = store
            .get_layer_checkpoint(&tail_handle)
            .await
            .expect("case_32 [negative control]: get_layer_checkpoint(tail) must succeed")
            .expect(
                "case_32 [negative control]: the admitted tail must be addressable — an Ok handle \
                 that resolves to nothing would make the control's success claim hollow",
            );
        assert_eq!(
            got_tail.bytes, tail_bytes,
            "case_32 [negative control]: and it must round-trip the tail's own bytes, not the \
             base's",
        );

        let refs_after = store
            .list_referenced_ext_hashes()
            .await
            .expect("case_32 [negative control]: list_referenced_ext_hashes must succeed");
        assert_eq!(
            refs_after, expected,
            "case_32 [negative control]: the admitted tail contributes NO ext hashes — a tail \
             carries no sidecar by construction, so the live set must be unchanged by it",
        );
    }
}

/// Pick the handle naming `(sequence, kind)` out of a `list_layer_checkpoints`
/// result, failing with the fixture's own claim rather than an anonymous
/// `Option::unwrap` panic.
///
/// Shared by the three cross-cutting Group J cases (#33, #34, #37), all of which
/// address generations through adapter-minted handles rather than synthesized
/// ones — the retention cascade and the lineage gather both work off a listing,
/// so the harness does too. A synthesized `CheckpointHandle::tail(..)` would
/// carry a fabricated `adapter_key` and would not exercise the same addressing
/// path.
fn pick_handle(
    handles: &[CheckpointHandle],
    sequence: u64,
    kind: CheckpointKind,
    context: &str,
) -> CheckpointHandle {
    handles
        .iter()
        .find(|h| h.seq() == sequence && h.kind == kind)
        .unwrap_or_else(|| {
            panic!(
                "{context}: the listing must report a ({sequence}, {kind:?}) generation — the \
                 fixture put one, so its absence is a put/list defect that invalidates every \
                 assertion downstream of it. Got {:?}",
                kind_seq_pairs(handles),
            )
        })
        .clone()
}

/// Case #33 — a tail is invisible to **every** ext-refs surface, asserted over a
/// mixed-kind store rather than over a tail in isolation (SR-2.4 / SR-5.3,
/// SDD §22.17.3).
///
/// **What breaks if this drifts, in both directions.**
/// [`CheckpointStore::list_referenced_ext_hashes`] is the *safety input* to every
/// ext-blob deletion in the system — the retention cascade deletes
/// `pruned − live`. A tail that contributes to `live` inflates the subtrahend, so
/// the cascade under-deletes and retention degrades toward reclaiming nothing;
/// that is b.s5y's retention-degradation shape and it is merely expensive. A tail
/// that contributes to a *per-handle* read points the other way and is not:
/// [`CheckpointStore::read_ext_refs_sidecar`] addresses by
/// `(object_id, snap_seq)` and the adapters' addressing helpers are **kind-blind**,
/// so a tail handle that falls through to the ordinary sidecar lookup reports the
/// BASE's hashes as the tail's. Pruning that tail then folds the base's hashes
/// into `pruned` while the base is still live, and `pruned − live` deletes blobs a
/// surviving generation references — a checkpoint that restores with dangling
/// ext-refs, the one silently-unrecoverable outcome in this subsystem.
///
/// **This is final behaviour, not an unfinished arm.** `put_layer_checkpoint`
/// rejects a tail carrying `sidecar_bytes` outright (case #32), so "absent" is the
/// only truthful answer a tail can ever have — on every adapter, forever. A later
/// Epic that "completes" the tail branch here has re-opened the window this case
/// exists to keep closed.
///
/// **Fixture** — one store, three objects, covering the three shapes a kind-blind
/// implementation fails differently on:
///
/// - `object 1`: `base@1` with `{h1, h2}`, plus `tail@4`. The ordinary
///   distinct-sequence tail: a kind-blind read answers `Ok(empty)` here *by
///   accident*, because nothing is stored at `(1, 4)`. That is exactly why this
///   shape is not sufficient on its own.
/// - `object 2`: `base@6` with `{h2, h3}` (overlapping the first object on `h2`),
///   plus `tail@6` at the **same sequence** — the SR-4.1 re-base coexistence,
///   where `base@S` is minted at the newest consumed tail's `to_seq`. This is the
///   discriminating shape: a kind-blind read returns `{h2, h3}` for the tail.
/// - `object 3`: `tail@9` and no base at all. Nothing anchors this object in a
///   sidecar-driven walk, so an adapter that enumerated tails into the live set
///   would surface an object owning no hashes whatsoever.
///
/// **Assertion 4 re-pins case #26's flat_map invariant over the mixed-kind
/// layout** rather than editing case #26, whose all-base fixture cannot express
/// it: `list_referenced_ext_hashes()` must still equal the union of
/// `read_ext_refs_sidecar` over *every* handle the listings report, tails
/// included. The tails contribute the empty set, so the equality holds without
/// special-casing them — and pinning that it *stays* true is the point. An
/// adapter that made tails visible to one of the two surfaces but not the other
/// breaks this equality even when both surfaces are individually plausible, which
/// is the drift a per-surface assertion cannot catch.
pub async fn case_33_tails_are_invisible_to_the_ext_refs_surface<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    use std::collections::HashSet;

    let (store, _guard) = factory().await;

    // Short opaque keys, as in case #23: `list_referenced_ext_hashes` treats
    // them as opaque strings with no SHA-256 well-formedness check, and short
    // names keep the overlap structure (`h2` shared between the two bases)
    // legible in a failure message.
    const H1: &str = "h1";
    const H2: &str = "h2";
    const H3: &str = "h3";

    // ---- object 1: base@1 (+ sidecar) and a distinct-sequence tail@4 -----
    let obj_distinct: u64 = 1;
    let mut base_1_meta = sample_meta(1);
    base_1_meta.sidecar_bytes = Some(ext_refs_sidecar_for(&[H1, H2]));
    store
        .put_layer_checkpoint(obj_distinct, 1, sample_layer_ompf(0x3301), base_1_meta)
        .await
        .expect("case_33: put base@1 on the distinct-sequence object must succeed");
    store
        .put_layer_checkpoint(
            obj_distinct,
            4,
            sample_tail_envelope(0x3304),
            sample_tail_meta(1, 1, 4),
        )
        .await
        .expect("case_33: put tail@4 on the distinct-sequence object must succeed");

    // ---- object 2: base@6 (+ overlapping sidecar) and tail@6 at the SAME
    // sequence — the re-base coexistence, and the shape that discriminates.
    let obj_same_seq: u64 = 2;
    let mut base_6_meta = sample_meta(6);
    base_6_meta.sidecar_bytes = Some(ext_refs_sidecar_for(&[H2, H3]));
    store
        .put_layer_checkpoint(obj_same_seq, 6, sample_layer_ompf(0x3306), base_6_meta)
        .await
        .expect("case_33: put base@6 on the same-sequence object must succeed");
    store
        .put_layer_checkpoint(
            obj_same_seq,
            6,
            sample_tail_envelope(0x3316),
            // Rooted at the pre-re-base base@3, covering (3, 6] — the lineage
            // SR-4.1 consumes to mint base@6.
            sample_tail_meta(3, 3, 6),
        )
        .await
        .expect("case_33: put tail@6 on the same-sequence object must succeed");

    // ---- object 3: tail@9 only, no base anywhere under the object --------
    let obj_tail_only: u64 = 3;
    store
        .put_layer_checkpoint(
            obj_tail_only,
            9,
            sample_tail_envelope(0x3309),
            sample_tail_meta(5, 5, 9),
        )
        .await
        .expect("case_33: put tail@9 on the tail-only object must succeed");

    // Adapter-minted handles, per the cascade's own access pattern.
    let listed_distinct = store
        .list_layer_checkpoints(obj_distinct)
        .await
        .expect("case_33: list_layer_checkpoints(distinct-sequence object) must succeed");
    let listed_same_seq = store
        .list_layer_checkpoints(obj_same_seq)
        .await
        .expect("case_33: list_layer_checkpoints(same-sequence object) must succeed");
    let listed_tail_only = store
        .list_layer_checkpoints(obj_tail_only)
        .await
        .expect("case_33: list_layer_checkpoints(tail-only object) must succeed");

    let base_1 = pick_handle(
        &listed_distinct,
        1,
        CheckpointKind::Base,
        "case_33 [distinct-sequence object]",
    );
    let tail_4 = pick_handle(
        &listed_distinct,
        4,
        CheckpointKind::Tail,
        "case_33 [distinct-sequence object]",
    );
    let base_6 = pick_handle(
        &listed_same_seq,
        6,
        CheckpointKind::Base,
        "case_33 [same-sequence object]",
    );
    let tail_6 = pick_handle(
        &listed_same_seq,
        6,
        CheckpointKind::Tail,
        "case_33 [same-sequence object]",
    );
    let tail_9 = pick_handle(
        &listed_tail_only,
        9,
        CheckpointKind::Tail,
        "case_33 [tail-only object]",
    );

    // ---- Assertion 1: every tail handle reads back the empty set ---------
    let tail_4_refs = store
        .read_ext_refs_sidecar(&tail_4)
        .await
        .expect("case_33: read_ext_refs_sidecar on tail@4 must be Ok — absence is not an error");
    assert!(
        tail_4_refs.is_empty(),
        "case_33: read_ext_refs_sidecar on a Tail handle MUST return Ok(empty) (SR-2.4, final \
         behaviour — put_layer_checkpoint forbids a tail from ever carrying an ext-refs sidecar, \
         so absent is the only truthful answer). Got {tail_4_refs:?}",
    );

    let tail_6_refs = store.read_ext_refs_sidecar(&tail_6).await.expect(
        "case_33: read_ext_refs_sidecar on the same-sequence tail@6 must be Ok, not an error",
    );
    assert!(
        tail_6_refs.is_empty(),
        "case_33: read_ext_refs_sidecar on tail@6 MUST return Ok(empty) even though a BASE at the \
         same sequence owns the sidecar {{h2, h3}}. This is the kind-blind-addressing hazard \
         stated exactly: `(object_id, snap_seq)` addresses the base, so an implementation that \
         does not short-circuit on `handle.kind` falls through to the base's sidecar and reports \
         the base's hashes as the tail's. The retention cascade would then fold {{h2, h3}} into \
         `pruned_hashes` when it prunes the tail, while base@6 is still live — and \
         `pruned − live` deletes ext blobs a surviving generation references. Got {tail_6_refs:?}",
    );

    let tail_9_refs = store
        .read_ext_refs_sidecar(&tail_9)
        .await
        .expect("case_33: read_ext_refs_sidecar on the tail-only object's tail@9 must be Ok");
    assert!(
        tail_9_refs.is_empty(),
        "case_33: read_ext_refs_sidecar on a tail whose object holds no base at all MUST return \
         Ok(empty) — and MUST NOT error. An object with nothing but a tail is an ordinary \
         steady-state shape between two base checkpoints, not a corrupt store. Got {tail_9_refs:?}",
    );

    // ---- Assertion 2: the bases still read back their own exact sets -----
    //
    // The negative control for assertion 1: without it, an adapter that
    // returned the empty set for EVERY handle would pass everything above.
    let base_1_refs = store
        .read_ext_refs_sidecar(&base_1)
        .await
        .expect("case_33: read_ext_refs_sidecar on base@1 must succeed");
    let expected_base_1: HashSet<String> = [H1, H2].iter().map(|s| s.to_string()).collect();
    assert_eq!(
        base_1_refs, expected_base_1,
        "case_33: the base's per-generation set must still come back verbatim — the tail rules \
         above constrain the Tail kind only, and an adapter that satisfied them by blanking the \
         ext-refs surface entirely would silently empty the retention cascade's per-handle input",
    );

    let base_6_refs = store
        .read_ext_refs_sidecar(&base_6)
        .await
        .expect("case_33: read_ext_refs_sidecar on base@6 must succeed");
    let expected_base_6: HashSet<String> = [H2, H3].iter().map(|s| s.to_string()).collect();
    assert_eq!(
        base_6_refs, expected_base_6,
        "case_33: the base SHARING ITS SEQUENCE with a tail must still read back its own exact \
         set — the kind short-circuit must fire on the tail handle only, never on the base one. \
         An adapter that keyed the short-circuit off 'a tail exists at this sequence' rather than \
         off `handle.kind` would blank the base here",
    );

    // ---- Assertion 3: the live set is exactly the bases' union -----------
    let live = store
        .list_referenced_ext_hashes()
        .await
        .expect("case_33: list_referenced_ext_hashes must succeed");
    let expected_live: HashSet<String> = [H1, H2, H3].iter().map(|s| s.to_string()).collect();
    assert_eq!(
        live, expected_live,
        "case_33: list_referenced_ext_hashes over a mixed-kind store MUST equal exactly the union \
         of the BASE sidecars — {{h1, h2, h3}}, with `h2` collapsing to one entry. No tail may \
         contribute, and the tail-only object must contribute nothing at all. The obligation is \
         satisfied by the enumeration's SHAPE, not by a kind predicate: LocalDisk and StorageApi \
         enumerate `*.ext-refs.json` directly and InMemory walks base slots only, so a tail is \
         structurally unreachable from all three. An adapter that re-derived this set from a \
         generation walk would put tails back in scope and reopen b.s5y",
    );

    // ---- Assertion 4: case #26's flat_map invariant, over mixed kinds ----
    //
    // Re-pinned here rather than by editing case #26, whose all-base fixture
    // cannot express it. Tails contribute the empty set, so the equality holds
    // with no special-casing — pinning that it STAYS true is the point: an
    // adapter that made tails visible to exactly one of the two surfaces breaks
    // this equality even when each surface looks plausible on its own.
    let mut per_handle_union: HashSet<String> = HashSet::new();
    let mut handles_visited = 0_usize;
    let mut tails_visited = 0_usize;
    for handle in listed_distinct
        .iter()
        .chain(listed_same_seq.iter())
        .chain(listed_tail_only.iter())
    {
        let refs = store
            .read_ext_refs_sidecar(handle)
            .await
            .unwrap_or_else(|err| {
                panic!(
                    "case_33: read_ext_refs_sidecar on listed handle ({}, {:?}) must succeed; got \
                     {err:?}",
                    handle.seq(),
                    handle.kind,
                )
            });
        handles_visited += 1;
        if handle.kind == CheckpointKind::Tail {
            tails_visited += 1;
        }
        per_handle_union.extend(refs);
    }
    assert_eq!(
        (handles_visited, tails_visited),
        (5, 3),
        "case_33 [precondition]: the three listings must together report all five generations the \
         fixture put, three of them tails. A short listing would make the union below agree with \
         the live set vacuously",
    );
    assert_eq!(
        live, per_handle_union,
        "case_33: list_referenced_ext_hashes must equal the union of read_ext_refs_sidecar over \
         EVERY listed handle across all three objects, tails included — case #26's flat_map \
         decomposition contract, re-pinned over a mixed-kind layout. The tails hold this equality \
         up by contributing the empty set, which is precisely what makes the whole-store surface \
         and the per-handle surface agree without either needing to know about the other",
    );
}

/// Case #34 — `list_all_object_ids` enumerates a **tail-only** object, and stops
/// enumerating an object once its last generation is deleted.
///
/// **Why this case exists.** [`CheckpointStore::list_all_object_ids`] is a
/// *required* trait method — no default impl, so every adapter must write one —
/// and until this case it had **zero** cross-adapter coverage. Its three
/// implementations (an in-memory key walk, a directory walk, a paginated LIST)
/// landed simultaneously in Epic 2 Tasks 3 and 4, which is exactly the
/// circumstance the harness exists for: three divergent realizations of one
/// obligation, authored in parallel, with nothing pinning them to the same
/// observable answer.
///
/// It is deliberately **not** justified by a production caller — there is none at
/// HEAD. b.s5y made the ext live set sidecar-driven and b.9qd routed orphan
/// reclamation through [`CheckpointStore::list_layer_artifacts`], both precisely
/// to get *off* an object-id-then-generation walk. The method stays on the trait
/// as the honest answer to "which objects does this store hold a restorable
/// payload for", and Epic 4's lineage gather and Epic 5's retention build on that
/// answer. Coverage now is what keeps the three implementations from drifting
/// apart in the interval before those consumers arrive — drift discovered by a
/// gather is drift discovered late.
///
/// **The tail-only object is the load-bearing half.** All three adapters are
/// generation-stem-driven, recognizing `<oid>.<to_seq>.tail` alongside
/// `<oid>.<snap_seq>.ompf`; the tempting narrowing — filter to `.ompf`, since "a
/// tail is not restorable on its own" — hides an object from the recovery hook's
/// working set entirely. That is a silent-data-loss shape, not a conservative
/// default: a caller has to *see* the object in order to conclude anything about
/// it, and the InMemory adapter gets the behaviour for free from its shared
/// per-sequence slot map while the other two must recognize the stem explicitly.
/// A green InMemory run is therefore no evidence about the other two, which is
/// the whole argument for pinning it here.
///
/// **The empty-object prune is asserted in both directions**, because only the
/// pair is meaningful. Deleting the tail-only object's sole tail must retire the
/// id — an id that lingers with nothing under it fans the recovery hook out over
/// a layer that has no payload. Deleting the *tail* of an object that also holds
/// a base must NOT retire it — an over-eager prune keyed on "a tail was deleted"
/// rather than on "nothing remains" would hide a perfectly restorable base.
pub async fn case_34_list_all_object_ids_includes_tail_only_objects<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    use std::collections::HashSet;

    /// Read `list_all_object_ids` as a set — the trait leaves the order
    /// unspecified, so every assertion in this case compares sets.
    async fn object_ids(store: &Arc<dyn CheckpointStore>, context: &str) -> HashSet<u64> {
        let ids = store.list_all_object_ids().await.unwrap_or_else(|err| {
            panic!("case_34 [{context}]: list_all_object_ids must succeed; got {err:?}")
        });
        ids.into_iter().collect()
    }

    let (store, _guard) = factory().await;

    let base_only: u64 = 101;
    let tail_only: u64 = 102;
    let both_kinds: u64 = 103;

    store
        .put_layer_checkpoint(base_only, 1, sample_layer_ompf(0x3401), sample_meta(1))
        .await
        .expect("case_34: put base@1 on the base-only object must succeed");
    store
        .put_layer_checkpoint(
            tail_only,
            7,
            sample_tail_envelope(0x3407),
            sample_tail_meta(3, 3, 7),
        )
        .await
        .expect("case_34: put tail@7 on the tail-only object must succeed");
    store
        .put_layer_checkpoint(both_kinds, 2, sample_layer_ompf(0x3402), sample_meta(2))
        .await
        .expect("case_34: put base@2 on the mixed-kind object must succeed");
    store
        .put_layer_checkpoint(
            both_kinds,
            5,
            sample_tail_envelope(0x3405),
            sample_tail_meta(2, 2, 5),
        )
        .await
        .expect("case_34: put tail@5 on the mixed-kind object must succeed");

    // ---- Assertion 1: all three objects enumerate, each exactly once -----
    let ids = object_ids(&store, "after seeding").await;
    assert_eq!(
        ids,
        [base_only, tail_only, both_kinds].into_iter().collect(),
        "case_34: list_all_object_ids must report every object holding at least one generation of \
         EITHER kind, and each exactly once. The tail-only object ({tail_only}) is the \
         load-bearing member: an adapter that filtered the walk to `.ompf` stems would drop it, \
         making it invisible to the recovery hook's working set — a silent leak, not a \
         conservative default. The mixed-kind object ({both_kinds}) holds two generations and \
         must still appear once, since the result is a set over object ids",
    );

    // ---- Assertion 2: deleting the sole tail retires the object ----------
    let tail_only_handles = store
        .list_layer_checkpoints(tail_only)
        .await
        .expect("case_34: list_layer_checkpoints(tail-only object) must succeed");
    let sole_tail = pick_handle(
        &tail_only_handles,
        7,
        CheckpointKind::Tail,
        "case_34 [tail-only object]",
    );
    store
        .delete_layer_checkpoint(sole_tail)
        .await
        .expect("case_34: deleting the tail-only object's sole tail must be Ok(())");

    let ids = object_ids(&store, "after deleting the tail-only object's tail").await;
    assert_eq!(
        ids,
        [base_only, both_kinds].into_iter().collect(),
        "case_34: an object whose LAST generation was deleted must stop enumerating — the \
         empty-object prune, at the trait surface. An id that lingers with nothing under it fans \
         every consumer (the recovery hook's working set, a retention walk) out over a layer that \
         has no restorable payload, and each of them can only discover the emptiness by a second \
         round trip. Note the two survivors: the prune must also not reach past the object it was \
         asked about",
    );

    // ---- Assertion 3: deleting only a tail spares an object with a base --
    let mixed_handles = store
        .list_layer_checkpoints(both_kinds)
        .await
        .expect("case_34: list_layer_checkpoints(mixed-kind object) must succeed");
    let mixed_tail = pick_handle(
        &mixed_handles,
        5,
        CheckpointKind::Tail,
        "case_34 [mixed-kind object]",
    );
    store
        .delete_layer_checkpoint(mixed_tail)
        .await
        .expect("case_34: deleting the mixed-kind object's tail must be Ok(())");

    let ids = object_ids(&store, "after deleting the mixed-kind object's tail").await;
    assert_eq!(
        ids,
        [base_only, both_kinds].into_iter().collect(),
        "case_34: deleting a TAIL from an object that still holds a base must leave the object \
         enumerated — the other direction of the empty-object prune, and the one that costs data \
         if it goes wrong. A prune keyed on 'a generation was deleted' rather than on 'nothing \
         remains under this object' would retire {both_kinds} here and hide a perfectly \
         restorable base@2 from every consumer of this enumeration",
    );
    assert!(
        store
            .get_latest_layer_checkpoint(both_kinds)
            .await
            .expect("case_34: get_latest_layer_checkpoint on the mixed-kind object must succeed")
            .is_some(),
        "case_34: and that base@2 must genuinely still be there — otherwise the enumeration above \
         is reporting an object whose payload the tail delete collaterally removed, which would \
         make assertion 3 pass for the wrong reason",
    );
}

/// Case #37 — `list_layer_artifacts` reports **both generation kinds**: over one
/// mixed-kind store the raw enumeration surfaces every base family AND every tail
/// family, keyed by kind, identically across all three adapters.
///
/// **The contract.** [`LayerArtifactKind`](live_edit::checkpoint::LayerArtifactKind)
/// spans both families — `Ompf` / `Meta` / `ExtRefs` on a base stem,
/// `TailEnvelope` / `TailMeta` on a tail stem — and
/// [`generation_kind`](live_edit::checkpoint::LayerArtifactKind::generation_kind)
/// is what separates them. The reconciliation pass groups by
/// `(object_id, snap_seq, generation kind)`, so the kind is part of the key, not a
/// label on the value: a base at `S` and a tail whose `to_seq` is `S` share the
/// numeric pair, and a kind-blind grouping would let one family's generation
/// artifact vouch for the other's torn stem. This case is the cross-adapter pin
/// that all three enumerations carry that key.
///
/// **Why it must be cross-adapter.** The three adapters arrive at the same set by
/// genuinely different routes: LocalDisk and StorageApi run the shared filename
/// parser over a flat namespace, so both families fall out of the naming grammar,
/// while InMemory has no filenames at all and must emit the tail members from its
/// kind-slotted map. A per-adapter test cannot see the two routes agreeing.
///
/// **Scope.** Everything here goes through the factory store's own trait surface,
/// so the fixture holds only *complete* generations. Torn residuals (an orphaned
/// `.tail.meta.json`, a leaked `.tail.<uuid>.tmp`) are unreachable from any trait
/// call — that is case #27's per-adapter planter, and on the two adapters that can
/// hold such a state they are pinned by
/// `local_disk.rs`'s `list_layer_artifacts_surfaces_orphans_and_temps_and_skips_strays`
/// and `storage_api.rs`'s `list_layer_artifacts_emits_temp_for_a_foreign_leftover`.
/// InMemory structurally cannot reach them (meta is held inline on every entry;
/// there is no temp-then-rename step).
///
/// **Fixture** — the same three objects the Epic-2 posture used, kept because the
/// shape is exactly the discriminating one:
///
/// - `object 371`: `base@1` **with** a sidecar, plus `tail@4` at a *distinct*
///   sequence. Expected: `Ompf`, `Meta`, `ExtRefs` at `(371, 1)` and
///   `TailEnvelope`, `TailMeta` at `(371, 4)`.
/// - `object 372`: `base@6` **without** a sidecar, plus `tail@6` at the *same*
///   sequence. Expected: `Ompf`, `Meta` at `(372, 6)` and `TailEnvelope`,
///   `TailMeta` at `(372, 6)` — four members at one numeric stem, split two/two
///   across the families. The `ExtRefs` absence here is load-bearing twice over:
///   an adapter emitting it unconditionally fails, and so does one that let the
///   tail borrow a sidecar it can never own (SR-2.4).
/// - `object 373`: `tail@9` only — the sharpest discriminator, because the object
///   owns no base stem for its tail to hide behind. Under the retired posture it
///   contributed nothing at all; it must now contribute its own two members, and
///   no base-family member.
///
/// **The same-sequence stem is now directly observable, which is the whole point
/// of putting the kind in the projection.** Assertion 4 reads `(372, 6)` and
/// requires two non-empty, disjoint kind-keyed families there. The retired posture
/// could only reach that stem by deleting the base and inferring from what was
/// left; with the kind present, the collision is a thing the primary enumeration
/// states outright.
///
/// **Phase 2 is kept, and repurposed.** It is no longer a reach-around for an
/// unobservable stem — it is the artifact-surface reading of case #31's
/// kind-isolation contract. #31 pins that a base delete spares the same-sequence
/// tail through the *handle* surface (`get_layer_checkpoint` still resolves it);
/// this asserts the raw enumeration agrees, i.e. that the delete removed exactly
/// the base family's two members at `(372, 6)` and left the tail family's two.
/// Those are different surfaces on different code paths, and it is the artifact
/// one the reconciliation pass actually consumes.
pub async fn case_37_list_layer_artifacts_reports_both_generation_kinds<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    use std::collections::HashSet;

    use live_edit::checkpoint::{LayerArtifact, LayerArtifactKind};

    let (store, _guard) = factory().await;

    let obj_sidecar: u64 = 371;
    let obj_same_seq: u64 = 372;
    let obj_tail_only: u64 = 373;

    // ---- object 371: base@1 with a sidecar + a distinct-sequence tail@4 --
    let mut base_1_meta = sample_meta(1);
    base_1_meta.sidecar_bytes = Some(ext_refs_sidecar_for(&["h-371"]));
    store
        .put_layer_checkpoint(obj_sidecar, 1, sample_layer_ompf(0x3701), base_1_meta)
        .await
        .expect("case_37: put base@1 (with sidecar) must succeed");
    store
        .put_layer_checkpoint(
            obj_sidecar,
            4,
            sample_tail_envelope(0x3704),
            sample_tail_meta(1, 1, 4),
        )
        .await
        .expect("case_37: put tail@4 must succeed");

    // ---- object 372: base@6 with NO sidecar + tail@6 at the same sequence
    store
        .put_layer_checkpoint(obj_same_seq, 6, sample_layer_ompf(0x3706), sample_meta(6))
        .await
        .expect("case_37: put base@6 (no sidecar) must succeed");
    store
        .put_layer_checkpoint(
            obj_same_seq,
            6,
            sample_tail_envelope(0x3716),
            sample_tail_meta(3, 3, 6),
        )
        .await
        .expect("case_37: put tail@6 must succeed");

    // ---- object 373: tail@9 only -----------------------------------------
    store
        .put_layer_checkpoint(
            obj_tail_only,
            9,
            sample_tail_envelope(0x3709),
            sample_tail_meta(5, 5, 9),
        )
        .await
        .expect("case_37: put tail@9 on the tail-only object must succeed");

    /// The kinds reported at one `(object_id, sequence)` stem, split by
    /// generation family — the exact projection the reconciliation pass keys on.
    fn family_kinds_at(
        artifacts: &[LayerArtifact],
        object_id: u64,
        sequence: u64,
        family: CheckpointKind,
    ) -> HashSet<LayerArtifactKind> {
        artifacts
            .iter()
            .filter(|a| {
                a.object_id == object_id
                    && a.snap_seq == sequence
                    && a.kind.generation_kind() == Some(family)
            })
            .map(|a| a.kind)
            .collect()
    }

    let base_family = |object_id: u64, snap_seq: u64, with_sidecar: bool| {
        let mut kinds = vec![LayerArtifactKind::Ompf, LayerArtifactKind::Meta];
        if with_sidecar {
            kinds.push(LayerArtifactKind::ExtRefs);
        }
        kinds.into_iter().map(move |kind| LayerArtifact {
            object_id,
            snap_seq,
            kind,
        })
    };
    let tail_family = |object_id: u64, to_seq: u64| {
        [LayerArtifactKind::TailEnvelope, LayerArtifactKind::TailMeta]
            .into_iter()
            .map(move |kind| LayerArtifact {
                object_id,
                snap_seq: to_seq,
                kind,
            })
    };

    let expected: HashSet<LayerArtifact> = base_family(obj_sidecar, 1, true)
        .chain(tail_family(obj_sidecar, 4))
        .chain(base_family(obj_same_seq, 6, false))
        .chain(tail_family(obj_same_seq, 6))
        .chain(tail_family(obj_tail_only, 9))
        .collect();

    let artifacts = store
        .list_layer_artifacts()
        .await
        .expect("case_37: list_layer_artifacts must succeed on a mixed-kind store");

    // ---- Assertion 1: no artifact escapes the expected families ----------
    //
    // Checked over the RAW vector rather than the deduplicated set, because the
    // trait permits duplicates and this claim must survive them: a foreign
    // artifact reported twice is still a foreign artifact.
    for artifact in &artifacts {
        assert!(
            expected.contains(artifact),
            "case_37: every artifact list_layer_artifacts reports must belong to a generation \
             the fixture put, in the family that generation's kind names. {artifact:?} does \
             not. The reconciliation pass groups by (object_id, snap_seq, generation kind) and \
             treats the family's generation artifact (`Ompf` for a base, `TailEnvelope` for a \
             tail) as 'this stem is live', so a stray member is not a cosmetic extra: it is a \
             phantom generation, or — at a stem where the other family is already present — a \
             member vouching for a stem it does not belong to. Full enumeration: {artifacts:?}",
        );
    }

    // ---- Assertion 2: and it reports exactly those, no fewer -------------
    let distinct: HashSet<LayerArtifact> = artifacts.iter().copied().collect();
    assert_eq!(
        distinct, expected,
        "case_37: the distinct artifacts must be exactly the five generations' families — \
         {{Ompf, Meta, ExtRefs}} at ({obj_sidecar}, 1), {{TailEnvelope, TailMeta}} at \
         ({obj_sidecar}, 4), {{Ompf, Meta}} + {{TailEnvelope, TailMeta}} at ({obj_same_seq}, 6), \
         and {{TailEnvelope, TailMeta}} at ({obj_tail_only}, 9): eleven in all. Assertion 1 \
         rules out extras; this rules out an adapter that satisfied it by under-reporting — \
         including one that still reports base families only — and pins the count exactly. \
         `ExtRefs` appears for the sidecar-bearing base ALONE, so an adapter that emitted the \
         kind unconditionally, or that let a tail carry one it can never own (SR-2.4), fails \
         here too",
    );

    // ---- Assertion 3: the tail-only object contributes its own family ----
    //
    // Implied by assertion 2, stated separately because it is the sharpest
    // discriminator in the fixture: this object owns no base stem, so nothing
    // else in the store can account for its members and nothing can mask their
    // absence. Under the retired base-only posture it contributed nothing at
    // all.
    assert_eq!(
        family_kinds_at(&artifacts, obj_tail_only, 9, CheckpointKind::Tail),
        [LayerArtifactKind::TailEnvelope, LayerArtifactKind::TailMeta]
            .into_iter()
            .collect::<HashSet<_>>(),
        "case_37: an object holding NOTHING but a tail must still contribute that tail's two \
         members — `{obj_tail_only}.9.tail` → TailEnvelope and \
         `{obj_tail_only}.9.tail.meta.json` → TailMeta, on every adapter. This is the whole \
         reachability claim: with no base stem in the object, these are the only artifacts that \
         can ever make it visible to the reconciliation pass. Got {artifacts:?}",
    );
    assert!(
        family_kinds_at(&artifacts, obj_tail_only, 9, CheckpointKind::Base).is_empty(),
        "case_37: ...and it must contribute no BASE-family member, which would fabricate a \
         full-state generation out of a delta envelope. Got {artifacts:?}",
    );

    // ---- Assertion 4: the same-sequence stem, read directly --------------
    //
    // The kind is in the projection, so the base/tail collision at (372, 6) is
    // something the primary enumeration states outright rather than something a
    // later delete has to tease apart. Two families, both complete, at one
    // numeric stem.
    let same_seq_base = family_kinds_at(&artifacts, obj_same_seq, 6, CheckpointKind::Base);
    let same_seq_tail = family_kinds_at(&artifacts, obj_same_seq, 6, CheckpointKind::Tail);
    assert_eq!(
        same_seq_base,
        [LayerArtifactKind::Ompf, LayerArtifactKind::Meta]
            .into_iter()
            .collect::<HashSet<_>>(),
        "case_37: at the shared stem ({obj_same_seq}, 6) the BASE family must report its own two \
         members and no more — no `ExtRefs` (this base was put without a sidecar) and nothing \
         borrowed from the tail sharing the sequence. Got {artifacts:?}",
    );
    assert_eq!(
        same_seq_tail,
        [LayerArtifactKind::TailEnvelope, LayerArtifactKind::TailMeta]
            .into_iter()
            .collect::<HashSet<_>>(),
        "case_37: and the TAIL family at that same stem must report its own two, distinctly. \
         Together with the assertion above this is the collision made observable: four members \
         at one (object_id, sequence), partitioned two/two by generation_kind. An adapter that \
         reported the numeric stem kind-blindly would satisfy neither half, and the \
         reconciliation pass reading such an enumeration would let the base's `Ompf` vouch for \
         the tail's stem (or the reverse) and reap a live generation's companion. \
         Got {artifacts:?}",
    );
    assert!(
        same_seq_base.is_disjoint(&same_seq_tail),
        "case_37 [invariant]: the two families' vocabularies must not overlap — a kind belongs \
         to exactly one generation_kind. Base {same_seq_base:?}, tail {same_seq_tail:?}",
    );

    // ---- Phase 2: kind-isolated delete, read on the artifact surface -----
    //
    // Case #31 pins that deleting the base at a shared stem spares the tail —
    // through the HANDLE surface. This asserts the raw enumeration agrees: the
    // delete removes exactly the base family's two members at (372, 6) and
    // leaves the tail family's two. Different surface, different code path, and
    // it is this one the reconciliation pass consumes.
    let same_seq_handles = store
        .list_layer_checkpoints(obj_same_seq)
        .await
        .expect("case_37: list_layer_checkpoints(same-sequence object) must succeed");
    let base_6 = pick_handle(
        &same_seq_handles,
        6,
        CheckpointKind::Base,
        "case_37 [same-sequence object]",
    );
    let tail_6 = pick_handle(
        &same_seq_handles,
        6,
        CheckpointKind::Tail,
        "case_37 [same-sequence object]",
    );
    store
        .delete_layer_checkpoint(base_6)
        .await
        .expect("case_37: deleting base@6 must be Ok(())");

    assert!(
        store
            .get_layer_checkpoint(&tail_6)
            .await
            .expect("case_37: get_layer_checkpoint(tail@6) must succeed")
            .is_some(),
        "case_37: the base delete must have spared the same-sequence tail on the HANDLE surface \
         too (case #31's kind-isolation contract). Paired with the artifact-surface assertions \
         below, this is what makes the two surfaces agree: an adapter whose delete dropped the \
         tail's payload while leaving its artifact names behind — or the reverse — satisfies one \
         and not the other",
    );

    let after = store
        .list_layer_artifacts()
        .await
        .expect("case_37: list_layer_artifacts after the base delete must succeed");
    assert!(
        family_kinds_at(&after, obj_same_seq, 6, CheckpointKind::Base).is_empty(),
        "case_37: the base delete must clear the BASE family at ({obj_same_seq}, 6) entirely — \
         a surviving `Meta` or `ExtRefs` there is precisely the orphaned companion the \
         reconciliation pass exists to find, manufactured by the delete itself. Got {after:?}",
    );
    assert_eq!(
        family_kinds_at(&after, obj_same_seq, 6, CheckpointKind::Tail),
        [LayerArtifactKind::TailEnvelope, LayerArtifactKind::TailMeta]
            .into_iter()
            .collect::<HashSet<_>>(),
        "case_37: ...and it must leave the TAIL family at that stem untouched — both members, \
         not just the envelope. This is case #31's kind-isolation contract read on the surface \
         the reconciliation pass consumes: a delete that reached across the family boundary \
         would strand a `TailEnvelope` with no meta (unreadable) or a `TailMeta` with no \
         envelope (an orphan candidate), from a call that named the base. Got {after:?}",
    );
    // Everything else is the whole of what may remain, taken from the phase-1
    // expected set minus exactly the two members the delete was addressed at, so
    // the two cannot drift apart. Set-exact rather than a count — arbitrary
    // artifacts of the right cardinality would satisfy a count.
    let deleted_family: HashSet<LayerArtifact> = base_family(obj_same_seq, 6, false).collect();
    let expected_after: HashSet<LayerArtifact> =
        expected.difference(&deleted_family).copied().collect();
    let distinct_after: HashSet<LayerArtifact> = after.iter().copied().collect();
    assert_eq!(
        distinct_after, expected_after,
        "case_37: and globally, what survives must be exactly the nine members the delete did \
         not name — both of ({obj_sidecar}, 1)'s and ({obj_sidecar}, 4)'s families, the tail at \
         the shared stem, and the tail-only object's. The delete is addressed at one stem in \
         one family, so it must neither reach across objects nor across the family boundary, \
         and nothing may be re-admitted. Got {after:?}",
    );
}

// ----- Group K: lineage-aware retention (#38) --------------------------------
//
// Epic 5 makes retention operate on **lineage units** — a base plus the tails
// rooted on it — rather than on individual generations, so the pruner's
// interaction with the store trait changes shape: it issues an ordered *run* of
// deletes per condemned unit (tails newest-first, base last) over handles
// captured from an earlier listing, and it may issue that whole run more than
// once (a second leader, or a retried tick). Group K pins what that run is
// allowed to do at the adapter, which is exactly two things: remove the
// generations it names, and nothing else.

/// Case #38 — deleting a whole lineage is idempotent at **every** position in
/// the run, and touches nothing outside that lineage (SR-5.1 / SR-5.2 / SR-5.3).
///
/// **The production sequence this mirrors.** `checkpointer::prune_old_generations`
/// condemns a superseded lineage unit and deletes it tails-newest-first, base
/// LAST, each delete addressed through the whole [`CheckpointHandle`] —
/// `(object_id, sequence, kind)`. Two properties of that loop are trait-level
/// obligations rather than pruner-local details, and this case is where they are
/// pinned cross-adapter:
///
/// 1. **Idempotent at every position.** The pruner deletes over a plan computed
///    from an *earlier* listing, and the checkpoint lease admits a second leader
///    after a lease handover; so "this handle is already gone" is the ordinary
///    case at any point in the run, not an exceptional one. A `NotFound` at any
///    position turns a duplicate-leader prune into a logged failure and (per the
///    landed loop's log-and-continue posture) leaves the unit half-deleted for
///    the next pass to finish. The trait's MUST is `store.rs` §Idempotency:
///    "deleting an already-absent object returns `Ok(())`, never `NotFound`".
/// 2. **Lineage-isolated.** The run is the only destructive act the retention
///    tick performs, and its blast radius is asserted here artifact-for-artifact
///    against a *sibling* lineage under the **same** `object_id`. That is the
///    realistic neighbour: an object's superseded unit and its live unit coexist
///    for a whole retention window, and the sibling is what every restore after
///    the prune depends on.
///
/// **Why case #31 does not already cover this.** #31 deletes *single*
/// generations and asserts kind-isolation at one shared sequence on objects that
/// hold nothing else. It cannot express (a) an ordered multi-delete run, (b) a
/// second lineage under the same object surviving that run, (c) the ext-refs
/// live set across it, or (d) the pruner's stale-handle re-issue. This case
/// extends #31's discipline — its `Ok(())`-never-`NotFound` delete shape and its
/// `Ok(None)`-never-`NotFound` get shape are reused verbatim — onto the
/// lineage-shaped sequence. It does not restate #31's absent-handle flavours.
///
/// **Fixture** — one store, two objects, three lineages:
///
/// ```text
///   object 38_038   TARGET   base@10 {shared, target-only}
///                            tail@14  (10,14]
///                            tail@17  (14,17]
///                            tail@20  (17,20]      <-- the re-base tie
///                   SIBLING  base@20 {shared, sibling-only}
///                            tail@23  (20,23]
///                            tail@26  (23,26]
///   object 38_039   BYSTANDER base@5 {bystander}
///                            tail@20  (5,20]       <-- cross-object tie
/// ```
///
/// Three collisions are deliberate, and each one is a distinct way a delete can
/// be mis-keyed:
///
/// - **`tail@20` beside `base@20`, same object.** The SR-4.1 re-base shape: the
///   tail covering `(17, 20]` was consumed into a fresh `base@20`, which roots
///   the sibling lineage, and the two coexist until retention removes the older.
///   The pruner deletes `tail@20` while `base@20` must survive — so an adapter
///   that keyed the delete `(object_id, sequence)` and dropped the kind destroys
///   the *newest* lineage's root while condemning the oldest.
/// - **`tail@20` on the bystander object.** Same sequence, same kind, different
///   object: an adapter whose delete addressing lost the `object_id` (a prefix /
///   URL-derivation slip on the two artifact-addressed adapters) takes this one
///   out too.
/// - **`shared` on both bases' ext-refs sidecars.** Deleting the target base
///   retires `target-only` from the live set but MUST NOT retire `shared`, which
///   the surviving sibling base still references. The live set is the *safety
///   input* to every ext-blob deletion (`pruned − live`), so a `shared` that
///   vanishes here is the retention cascade deleting a blob a live generation
///   points at — the one silently-unrecoverable outcome in this subsystem.
///
/// **Invariant battery, re-run after every delete** rather than once at the end,
/// so a failure names the position in the run that broke it: every survivor is
/// re-fetched and compared byte-for-byte with its lineage meta fields and its
/// per-handle ext-refs answer; every already-deleted handle must read `Ok(None)`
/// (no resurrection); both objects' listings must equal their expected
/// remainder; `get_latest_layer_checkpoint` must agree with
/// `list_layer_checkpoints().first()` on both objects; and the live ext-hash set
/// must equal its exact expected value — unchanged across all three tail
/// deletes (tails carry no sidecar, SR-5.3) and losing exactly `target-only` at
/// the base delete.
///
/// **Handles come from the listing, not from the puts.** The pruner plans over
/// `list_layer_checkpoints` output and then deletes, so the plan is captured
/// once from the *initial* listing and every delete — first issue, immediate
/// repeat, and the whole-run re-issue at the end — goes through those same
/// handles. Re-picking a fresh handle before each delete would quietly test a
/// sequence the pruner never executes.
///
/// **Fixtures are this file's synthetic ones** ([`sample_layer_ompf`],
/// [`sample_tail_envelope`], [`sample_tail_meta`]), not the integration
/// `helpers.rs` generation seeders. Those encode real `tail_codec` envelopes,
/// and the harness must not acquire a dependency on that module: SR-1.5 makes a
/// tail's bytes as opaque to an adapter as a base's OMPF is, so a conformance
/// case that knew the envelope's structure would be testing the wrong layer (see
/// this file's module doc).
pub async fn case_38_lineage_prune_is_idempotent_and_lineage_isolated<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    use std::collections::HashSet;

    /// The object holding the condemned lineage AND its surviving sibling.
    const PRUNED_OBJECT: u64 = 38_038;
    /// A second object, never addressed by any delete this case issues.
    const BYSTANDER_OBJECT: u64 = 38_039;

    /// Referenced by BOTH bases in `PRUNED_OBJECT`. Must survive the prune: the
    /// sibling base still points at it.
    const H_SHARED: &str = "c38-shared";
    /// Referenced only by the condemned base — the one hash the prune retires.
    const H_TARGET_ONLY: &str = "c38-target-only";
    /// Referenced only by the surviving sibling base.
    const H_SIBLING_ONLY: &str = "c38-sibling-only";
    /// Referenced only by the bystander object's base.
    const H_BYSTANDER: &str = "c38-bystander";

    /// One seeded generation plus everything a survivor check needs to state its
    /// claim without re-deriving it at the assertion site.
    struct Seeded {
        /// The put-minted handle. Reads address through it; deletes deliberately
        /// do not (see the case doc's handle-provenance paragraph).
        handle: CheckpointHandle,
        /// The exact payload the adapter was handed, for the byte-identity claim.
        bytes: Bytes,
        /// `(base_snap_seq, from_seq, snap_seq)` as `get_layer_checkpoint` must
        /// read them back. A base is `(None, None, Some(seq))`; a tail names the
        /// base it replays onto.
        lineage: (Option<u64>, Option<u64>, Option<u64>),
        /// Exactly what `read_ext_refs_sidecar` must answer for this handle —
        /// empty for every tail (SR-2.4: a tail structurally has no sidecar).
        ext_hashes: HashSet<String>,
        /// Human-readable identity for failure messages.
        label: String,
    }

    fn hash_set(hashes: &[&str]) -> HashSet<String> {
        hashes.iter().map(|h| (*h).to_string()).collect()
    }

    /// Per-artifact payload discriminator. Folds in the `object_id` so the two
    /// `tail@20`s on different objects carry different bytes — otherwise the
    /// cross-object isolation claim would be satisfiable by serving the wrong
    /// object's artifact.
    fn base_tag(object_id: u64, seq: u64) -> u64 {
        (object_id << 16) | 0xB000 | seq
    }

    fn tail_tag(object_id: u64, seq: u64) -> u64 {
        (object_id << 16) | 0x7A00 | seq
    }

    async fn put_base(
        store: &Arc<dyn CheckpointStore>,
        object_id: u64,
        seq: u64,
        ext_hashes: &[&str],
        label: &str,
    ) -> Seeded {
        let bytes = sample_layer_ompf(base_tag(object_id, seq));
        let mut meta = sample_meta(seq);
        meta.sidecar_bytes = Some(ext_refs_sidecar_for(ext_hashes));
        let handle = store
            .put_layer_checkpoint(object_id, seq, bytes.clone(), meta)
            .await
            .unwrap_or_else(|err| {
                panic!("case_38: seeding {label} (base@{seq} on object {object_id}) must succeed; got {err:?}")
            });
        assert_eq!(
            handle.kind,
            CheckpointKind::Base,
            "case_38: {label} was put with a Base meta, so its handle must name a Base",
        );
        Seeded {
            handle,
            bytes,
            lineage: (None, None, Some(seq)),
            ext_hashes: hash_set(ext_hashes),
            label: label.to_string(),
        }
    }

    async fn put_tail(
        store: &Arc<dyn CheckpointStore>,
        object_id: u64,
        base_snap_seq: u64,
        from_seq: u64,
        to_seq: u64,
        label: &str,
    ) -> Seeded {
        let bytes = sample_tail_envelope(tail_tag(object_id, to_seq));
        let handle = store
            .put_layer_checkpoint(
                object_id,
                to_seq,
                bytes.clone(),
                sample_tail_meta(base_snap_seq, from_seq, to_seq),
            )
            .await
            .unwrap_or_else(|err| {
                panic!("case_38: seeding {label} (tail ({from_seq}, {to_seq}] on object {object_id}) must succeed; got {err:?}")
            });
        assert_eq!(
            handle.kind,
            CheckpointKind::Tail,
            "case_38: {label} was put with a Tail meta, so its handle must name a Tail",
        );
        Seeded {
            handle,
            bytes,
            lineage: (Some(base_snap_seq), Some(from_seq), Some(to_seq)),
            // SR-2.4: a tail carries no ext-refs sidecar, and `read_ext_refs_sidecar`
            // MUST answer the empty set on a tail handle rather than falling
            // through to the kind-blind address and reporting the BASE's hashes.
            ext_hashes: HashSet::new(),
            label: label.to_string(),
        }
    }

    /// `Ok(())`, never `NotFound` — case #31's delete-path shape, applied to
    /// every delete in the run (present or absent alike).
    async fn assert_delete_ok(
        store: &Arc<dyn CheckpointStore>,
        handle: &CheckpointHandle,
        why: &str,
    ) {
        let addressed = format!(
            "object_id={}, seq={}, kind={:?}",
            handle.object_id,
            handle.seq(),
            handle.kind,
        );
        match store.delete_layer_checkpoint(handle.clone()).await {
            Ok(()) => {}
            Err(CheckpointError::NotFound) => panic!(
                "case_38: delete_layer_checkpoint({addressed}) must return Ok(()), NOT \
                 CheckpointError::NotFound — {why}. The retention pruner deletes a lineage over a \
                 plan captured from an earlier listing and a second leader may execute the same \
                 plan, so 'already gone' is the ordinary outcome at every position in the run",
            ),
            Err(err) => panic!(
                "case_38: delete_layer_checkpoint({addressed}) must succeed — {why}; got \
                 unexpected error: {err:?}",
            ),
        }
    }

    /// `Ok(None)`, never `NotFound`, never a blob — case #31's absent-handle
    /// shape on the get path.
    async fn assert_absent(store: &Arc<dyn CheckpointStore>, handle: &CheckpointHandle, why: &str) {
        match store.get_layer_checkpoint(handle).await {
            Ok(None) => {}
            Ok(Some(blob)) => panic!(
                "case_38: get_layer_checkpoint(object_id={}, seq={}, kind={:?}) must be Ok(None) \
                 — {why}; got Some({} bytes). A deleted generation that reads back is either a \
                 delete that did not happen or a resurrection by a later delete in the run",
                handle.object_id,
                handle.seq(),
                handle.kind,
                blob.bytes.len(),
            ),
            Err(err) => {
                assert!(
                    !err.is_not_found(),
                    "case_38: get_layer_checkpoint must NEVER surface NotFound for a deleted \
                     generation — {why}. Absence is Ok(None) per the trait's get-path rule, and a \
                     concurrent prune makes it an ordinary outcome. Got: {err:?}",
                );
                panic!(
                    "case_38: get_layer_checkpoint on a deleted generation must be Ok(None) — \
                     {why}; got unexpected error: {err:?}",
                );
            }
        }
    }

    /// Every trait-surface claim about a generation the run must NOT have
    /// touched: it resolves, its bytes are verbatim, its meta still names its
    /// own kind and lineage, and its per-handle ext-refs answer is unchanged.
    async fn assert_intact(store: &Arc<dyn CheckpointStore>, generation: &Seeded, when: &str) {
        let label = &generation.label;
        let blob = match store.get_layer_checkpoint(&generation.handle).await {
            Ok(Some(blob)) => blob,
            Ok(None) => panic!(
                "case_38 [{when}]: {label} must still resolve — it belongs to a lineage the run \
                 never addressed. Ok(None) here is the stranded-generation data-loss shape SR-2.4 \
                 forbids: a delete that widened past the lineage it condemned",
            ),
            Err(err) => panic!(
                "case_38 [{when}]: get_layer_checkpoint on {label} must succeed; got {err:?}",
            ),
        };
        assert_eq!(
            (blob.handle.seq(), blob.handle.kind),
            (generation.handle.seq(), generation.handle.kind),
            "case_38 [{when}]: {label} must read back under its own (sequence, kind)",
        );
        assert_eq!(
            blob.bytes, generation.bytes,
            "case_38 [{when}]: {label} must be byte-identical. Different bytes mean the read \
             served a neighbouring artifact (the two same-sequence collisions in this fixture are \
             exactly where that happens) or the run rewrote one it was only supposed to leave \
             alone — a generation that stays enumerable but corrupt is worse than one removed",
        );
        assert_eq!(
            blob.meta.kind, generation.handle.kind,
            "case_38 [{when}]: {label} must still read back as its own kind — a survivor \
             relabelled Tail is replayed as a delta envelope by every restore path, and one \
             relabelled Base is offered to a restore as a standalone snapshot it is not",
        );
        assert_eq!(
            (
                blob.meta.base_snap_seq,
                blob.meta.from_seq,
                blob.meta.snap_seq
            ),
            generation.lineage,
            "case_38 [{when}]: {label}'s lineage fields must be untouched. The meta companion is \
             the only record of which base a tail replays onto, so a run that perturbed a \
             surviving neighbour's lineage would break the next restore's chain walk without \
             losing a single artifact",
        );
        let refs = store
            .read_ext_refs_sidecar(&generation.handle)
            .await
            .unwrap_or_else(|err| {
                panic!("case_38 [{when}]: read_ext_refs_sidecar({label}) must succeed; got {err:?}")
            });
        assert_eq!(
            refs, generation.ext_hashes,
            "case_38 [{when}]: {label}'s per-handle ext-refs answer must be unchanged by the run. \
             This is the per-generation half of the cascade's arithmetic: a survivor whose \
             sidecar the run collaterally removed reports the empty set, and its blobs then fall \
             out of `live` on the next sweep",
        );
    }

    /// `get_latest_layer_checkpoint` == `list_layer_checkpoints().first()`, both
    /// in identity and in bytes, plus the listing's own expected remainder.
    ///
    /// The identity is a trait MUST that survives the tail era deliberately
    /// (`store.rs` §Ordering): it is the only thing keeping an adapter's two
    /// enumeration paths honest with each other, and a prune is precisely when
    /// they can drift — the listing walks stems while `get_latest` may cache or
    /// short-circuit.
    async fn assert_listing_and_latest_agree(
        store: &Arc<dyn CheckpointStore>,
        object_id: u64,
        expected_listing: &[(u64, CheckpointKind)],
        survivors: &[Seeded],
        when: &str,
    ) {
        let listed = store
            .list_layer_checkpoints(object_id)
            .await
            .unwrap_or_else(|err| {
                panic!(
                    "case_38 [{when}]: list_layer_checkpoints({object_id}) must succeed; got \
                     {err:?}"
                )
            });
        assert_eq!(
            kind_seq_pairs(&listed),
            expected_listing.to_vec(),
            "case_38 [{when}]: object {object_id} must list exactly its expected remainder, in \
             one strict descending order with Base before Tail at a tie. A missing entry is a \
             delete that widened; an extra one is a delete that did not land, or a residual meta \
             companion parsed back into a phantom generation. Got {listed:?}",
        );

        let latest = store
            .get_latest_layer_checkpoint(object_id)
            .await
            .unwrap_or_else(|err| {
                panic!(
                    "case_38 [{when}]: get_latest_layer_checkpoint({object_id}) must succeed; got \
                     {err:?}"
                )
            });
        match (listed.first(), latest) {
            (None, None) => {}
            (None, Some(blob)) => panic!(
                "case_38 [{when}]: object {object_id} lists nothing, so get_latest must be \
                 Ok(None); got a blob at seq={} kind={:?}",
                blob.handle.seq(),
                blob.handle.kind,
            ),
            (Some(head), None) => panic!(
                "case_38 [{when}]: object {object_id} lists ({}, {:?}) at its head, so get_latest \
                 must return that generation, not Ok(None)",
                head.seq(),
                head.kind,
            ),
            (Some(head), Some(blob)) => {
                assert_eq!(
                    (blob.handle.seq(), blob.handle.kind),
                    (head.seq(), head.kind),
                    "case_38 [{when}]: get_latest_layer_checkpoint({object_id}) must name the \
                     identical generation as list(..).first() — the two enumerations are one \
                     contract, and a prune is exactly when an adapter's cached / short-circuited \
                     latest can drift from what its listing walk reports",
                );
                let expected = survivors
                    .iter()
                    .find(|generation| {
                        generation.handle.object_id == object_id
                            && generation.handle.seq() == head.seq()
                            && generation.handle.kind == head.kind
                    })
                    .unwrap_or_else(|| {
                        panic!(
                            "case_38 [{when}]: the listing head ({}, {:?}) on object {object_id} \
                             must be one of the seeded survivors — a head naming anything else is \
                             a generation the fixture never wrote",
                            head.seq(),
                            head.kind,
                        )
                    });
                assert_eq!(
                    blob.bytes, expected.bytes,
                    "case_38 [{when}]: get_latest must serve {}'s bytes verbatim — the kind-blind \
                     readers of this method reach the generation through exactly this call",
                    expected.label,
                );
            }
        }
    }

    /// The full battery, re-run after every delete so a failure names the
    /// position in the run rather than only the end state.
    async fn assert_invariants(
        store: &Arc<dyn CheckpointStore>,
        survivors: &[Seeded],
        deleted: &[&CheckpointHandle],
        expected_pruned_listing: &[(u64, CheckpointKind)],
        expected_live: &HashSet<String>,
        when: &str,
    ) {
        for generation in survivors {
            assert_intact(store, generation, when).await;
        }
        for handle in deleted {
            assert_absent(
                store,
                handle,
                &format!(
                    "[{when}] it was deleted earlier in the run, and no later delete may \
                     resurrect it"
                ),
            )
            .await;
        }
        assert_listing_and_latest_agree(
            store,
            PRUNED_OBJECT,
            expected_pruned_listing,
            survivors,
            when,
        )
        .await;
        assert_listing_and_latest_agree(
            store,
            BYSTANDER_OBJECT,
            &[(20, CheckpointKind::Tail), (5, CheckpointKind::Base)],
            survivors,
            when,
        )
        .await;

        let live = store
            .list_referenced_ext_hashes()
            .await
            .unwrap_or_else(|err| {
                panic!("case_38 [{when}]: list_referenced_ext_hashes must succeed; got {err:?}")
            });
        assert_eq!(
            &live, expected_live,
            "case_38 [{when}]: the live ext-hash set must be exactly this. It is the safety input \
             to every ext-blob deletion (`pruned − live`), so both directions are defects: a hash \
             that lingers under-deletes and degrades the cascade toward reclaiming nothing, while \
             a hash that vanishes early — `{H_SHARED}` is the one at risk, referenced by BOTH \
             bases — lets the cascade delete a blob a surviving generation still points at",
        );
    }

    let (store, _guard) = factory().await;

    // ---- Seed: the SIBLING lineage first ---------------------------------
    //
    // Seeding the survivor before the condemned lineage means the target's own
    // puts must already coexist with it — an adapter that clobbered the sibling
    // at the shared sequence 20 on the *put* path would fail the precondition
    // battery below rather than being mistaken for a delete-side defect.
    let sibling = vec![
        put_base(
            &store,
            PRUNED_OBJECT,
            20,
            &[H_SHARED, H_SIBLING_ONLY],
            "sibling base@20",
        )
        .await,
        put_tail(&store, PRUNED_OBJECT, 20, 20, 23, "sibling tail (20, 23]").await,
        put_tail(&store, PRUNED_OBJECT, 20, 23, 26, "sibling tail (23, 26]").await,
    ];

    // ---- Seed: the TARGET lineage — a base and three chained tails --------
    //
    // The newest tail's `to_seq` is 20, the sibling base's sequence: the SR-4.1
    // re-base coexistence, where the base minted from a consumed tail lands at
    // that tail's upper bound. `base@20` and `tail@20` are two generations, and
    // the prune deletes exactly one of them.
    let target_base = put_base(
        &store,
        PRUNED_OBJECT,
        10,
        &[H_SHARED, H_TARGET_ONLY],
        "target base@10",
    )
    .await;
    let target_tails = [
        put_tail(&store, PRUNED_OBJECT, 10, 10, 14, "target tail (10, 14]").await,
        put_tail(&store, PRUNED_OBJECT, 10, 14, 17, "target tail (14, 17]").await,
        put_tail(&store, PRUNED_OBJECT, 10, 17, 20, "target tail (17, 20]").await,
    ];

    // ---- Seed: the BYSTANDER object --------------------------------------
    //
    // Its tail sits at sequence 20 too — same sequence AND same kind as the
    // target's newest tail, differing only in `object_id`.
    let bystander = vec![
        put_base(
            &store,
            BYSTANDER_OBJECT,
            5,
            &[H_BYSTANDER],
            "bystander base@5",
        )
        .await,
        put_tail(&store, BYSTANDER_OBJECT, 5, 5, 20, "bystander tail (5, 20]").await,
    ];

    let survivors: Vec<Seeded> = sibling.into_iter().chain(bystander).collect();

    // Every payload in the fixture must be distinct, or "the survivor holds its
    // OWN bytes" is satisfiable by serving a neighbour's artifact.
    let distinct_payloads: HashSet<&[u8]> = survivors
        .iter()
        .chain(std::iter::once(&target_base))
        .chain(target_tails.iter())
        .map(|generation| generation.bytes.as_ref())
        .collect();
    assert_eq!(
        distinct_payloads.len(),
        survivors.len() + 1 + target_tails.len(),
        "test precondition: all eight seeded payloads must be pairwise distinct, or every \
         byte-identity claim below can be satisfied by the wrong artifact",
    );

    // ---- Preconditions ---------------------------------------------------
    let initial_listing = store
        .list_layer_checkpoints(PRUNED_OBJECT)
        .await
        .expect("case_38: list_layer_checkpoints on the seeded object must succeed");
    let full_order = vec![
        (26, CheckpointKind::Tail),
        (23, CheckpointKind::Tail),
        (20, CheckpointKind::Base),
        (20, CheckpointKind::Tail),
        (17, CheckpointKind::Tail),
        (14, CheckpointKind::Tail),
        (10, CheckpointKind::Base),
    ];
    assert_eq!(
        kind_seq_pairs(&initial_listing),
        full_order,
        "test precondition: both lineages must be present and interleaved newest-first with the \
         pinned Base-before-Tail tie-break at sequence 20 — the tie IS the fixture, so a listing \
         that reports only one generation there makes the kind-keying claim vacuous. Got \
         {initial_listing:?}",
    );

    let live_before = hash_set(&[H_SHARED, H_TARGET_ONLY, H_SIBLING_ONLY, H_BYSTANDER]);
    // The condemned base's `H_TARGET_ONLY` is the ONLY hash the run retires.
    let live_after = hash_set(&[H_SHARED, H_SIBLING_ONLY, H_BYSTANDER]);

    assert_invariants(
        &store,
        &survivors,
        &[],
        &full_order,
        &live_before,
        "before the run",
    )
    .await;
    for generation in std::iter::once(&target_base).chain(target_tails.iter()) {
        assert_intact(&store, generation, "before the run").await;
    }

    // ---- The prune plan: captured ONCE, from the initial listing ----------
    //
    // Tails newest-first, then the base LAST — the order
    // `checkpointer::prune_old_generations` issues for a condemned unit. The
    // forbidden direction is base-first: it would leave rootless tails behind
    // if the run were interrupted, which is the FR-6 shape the production loop's
    // ordering exists to avoid. Nothing in this case *observes* interruption —
    // that is the in-crate crash-matrix test's job — but the sequence a
    // conformance case exercises should be the one production issues, or it is
    // pinning a code path nothing takes.
    let plan: Vec<CheckpointHandle> = [
        (20, CheckpointKind::Tail),
        (17, CheckpointKind::Tail),
        (14, CheckpointKind::Tail),
        (10, CheckpointKind::Base),
    ]
    .into_iter()
    .map(|(sequence, kind)| {
        pick_handle(
            &initial_listing,
            sequence,
            kind,
            "case_38: building the prune plan",
        )
    })
    .collect();

    // What must hold after one position of the run: `PRUNED_OBJECT`'s expected
    // listing remainder, and the live ext-hash set.
    type StepExpectation<'a> = (Vec<(u64, CheckpointKind)>, &'a HashSet<String>);

    // The remainder after each step, and the live set that must hold at it.
    // The live set is unchanged across all three tail deletes: a tail carries no
    // ext-refs sidecar (SR-2.4), so pruning one can move no hash. Only the base
    // delete retires anything.
    let expected_after_step: [StepExpectation<'_>; 4] = [
        (
            vec![
                (26, CheckpointKind::Tail),
                (23, CheckpointKind::Tail),
                (20, CheckpointKind::Base),
                (17, CheckpointKind::Tail),
                (14, CheckpointKind::Tail),
                (10, CheckpointKind::Base),
            ],
            &live_before,
        ),
        (
            vec![
                (26, CheckpointKind::Tail),
                (23, CheckpointKind::Tail),
                (20, CheckpointKind::Base),
                (14, CheckpointKind::Tail),
                (10, CheckpointKind::Base),
            ],
            &live_before,
        ),
        (
            vec![
                (26, CheckpointKind::Tail),
                (23, CheckpointKind::Tail),
                (20, CheckpointKind::Base),
                (10, CheckpointKind::Base),
            ],
            &live_before,
        ),
        (
            vec![
                (26, CheckpointKind::Tail),
                (23, CheckpointKind::Tail),
                (20, CheckpointKind::Base),
            ],
            &live_after,
        ),
    ];

    // ---- Phase 1: run the plan, repeating EVERY delete in place -----------
    for (position, (handle, (expected_listing, expected_live))) in
        plan.iter().zip(expected_after_step.iter()).enumerate()
    {
        let addressed = format!("seq={} kind={:?}", handle.seq(), handle.kind);
        assert_delete_ok(
            &store,
            handle,
            &format!(
                "position {position} of the lineage run deletes {addressed}, a generation that \
                 exists"
            ),
        )
        .await;
        // The repeat is the whole point of the position-by-position structure:
        // a duplicate leader re-executes the plan, and it may reach any position
        // with the earlier ones already applied.
        assert_delete_ok(
            &store,
            handle,
            &format!(
                "re-issuing position {position} ({addressed}) immediately must also be Ok — this \
                 is the duplicate-leader case, not an error"
            ),
        )
        .await;

        assert_invariants(
            &store,
            &survivors,
            &plan[..=position].iter().collect::<Vec<_>>(),
            expected_listing,
            expected_live,
            &format!("after position {position} ({addressed}) and its repeat"),
        )
        .await;
    }

    // ---- Phase 2: re-issue the WHOLE plan over a lineage that is gone -----
    //
    // The pruner's other repeat shape: a second leader executes the entire plan
    // from scratch against a store where every handle in it has already been
    // deleted. Every position must still answer `Ok(())`, and the pass must be
    // a no-op on everything else.
    for (position, handle) in plan.iter().enumerate() {
        assert_delete_ok(
            &store,
            handle,
            &format!(
                "position {position} of a whole-plan re-issue over an already-pruned lineage — \
                 the shape a second leader executes after the first one finished"
            ),
        )
        .await;
    }

    assert_invariants(
        &store,
        &survivors,
        &plan.iter().collect::<Vec<_>>(),
        &[
            (26, CheckpointKind::Tail),
            (23, CheckpointKind::Tail),
            (20, CheckpointKind::Base),
        ],
        &live_after,
        "after the whole-plan re-issue",
    )
    .await;

    // ---- Phase 3: the surviving lineage is still a lineage ----------------
    //
    // Membership and bytes are asserted above; this closes the loop on the
    // sibling's own internal consistency — its tails still name `base@20`, which
    // is the artifact the prune came closest to destroying (it shares sequence
    // 20 with the newest generation the run deleted).
    let surviving_base = survivors
        .iter()
        .find(|generation| {
            generation.handle.object_id == PRUNED_OBJECT
                && generation.handle.kind == CheckpointKind::Base
        })
        .expect("case_38: the sibling base is a seeded survivor");
    assert_eq!(
        surviving_base.handle.seq(),
        20,
        "case_38: the surviving base is the one at the shared sequence — the whole point of the \
         tie",
    );
    for generation in survivors
        .iter()
        .filter(|generation| generation.handle.object_id == PRUNED_OBJECT)
        .filter(|generation| generation.handle.kind == CheckpointKind::Tail)
    {
        let blob = store
            .get_layer_checkpoint(&generation.handle)
            .await
            .expect("case_38: get_layer_checkpoint on a surviving sibling tail must succeed")
            .expect("case_38: the sibling tails survive the run");
        assert_eq!(
            blob.meta.base_snap_seq,
            Some(20),
            "case_38: {} must still name base@20 as its root. A restore after the prune walks \
             this field to find the base it replays onto; a tail whose root pointer the run \
             disturbed is unrestorable even though every artifact is present",
            generation.label,
        );
    }
}

// ----- Group J, injected error classification (#35–#36) ---------------------
//
// Group G (#19 / #20 / #22) pins the Transient / Permanent / Conflict surface
// for the Epic-1 methods. The tail generations add one new trait method
// (`get_layer_checkpoint`, op label `"get_layer_checkpoint"`) and one new path
// through an existing one (a tail put through `"put_layer_checkpoint"`); these
// two cases extend the same classification claim to both. They live in Group J
// rather than G because what they pin is the tail surface, and they are gated
// on `test-checkpoint-fault-injection` exactly as their Group G ancestors are.
//
// **Op labels are never kind-qualified.** The Epic-wide policy is one label per
// trait *method*, so there is no `"put_tail_checkpoint"` and no
// `"get_tail_checkpoint"` — a case that needs to fail a tail put but not a base
// put keys on injector-side state, not on a second label. StorageApi makes the
// reason concrete: it has no `CheckpointFaultInjector` seam at all, and its
// wrapper translates the verdict into a route-level wiremock override, so both
// kinds share one HTTP route regardless of what the label were called.
//
// **Consequence both case bodies are designed around:** an injection case must
// not require a *successful* call on the injected op's route earlier in its own
// body, because the StorageApi override is route-level rather than call-ordered
// and would fail the setup call too. Case #35 satisfies this by addressing a
// never-written generation; case #36 by making the tail put the body's only
// write (`validate_tail_put` is presence-only, so no pre-existing base is
// needed).

/// Which variant a sub-shape of case #35 / #36 expects back.
///
/// Named rather than passed as a bare bool triple so the call sites read as the
/// claim they make; the helper below turns it into both the literal-variant
/// `matches!` and the four-way exclusive classifier assertion.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedClass {
    Transient,
    Permanent,
    Conflict,
}

/// Assert an injected error is *literally* the expected variant AND that all
/// four classifiers agree exclusively — the assertion shape cases #19 / #20 /
/// #22 spell out inline, factored here because #35 and #36 make five such
/// claims between them across three different variants.
///
/// Both halves are load-bearing and neither implies the other: the classifiers
/// are `CheckpointError` inherent methods that a future variant could answer
/// `true` from without being the named variant, and a caller that matches the
/// variant directly (the recovery loop does, for `Conflict`) would not notice a
/// classifier drifting. `NotFound` is asserted `false` in every case: none of
/// these paths looks anything up in a way that could legitimately produce it.
#[cfg(feature = "test-checkpoint-fault-injection")]
fn assert_exclusively_classified(err: &CheckpointError, expected: ExpectedClass, shape: &str) {
    let literal_match = match expected {
        ExpectedClass::Transient => matches!(err, CheckpointError::Transient { .. }),
        ExpectedClass::Permanent => matches!(err, CheckpointError::Permanent { .. }),
        ExpectedClass::Conflict => matches!(err, CheckpointError::Conflict),
    };
    assert!(
        literal_match,
        "{shape}: the variant returned must literally be CheckpointError::{expected:?}; got \
         {err:?}",
    );

    let (want_transient, want_permanent, want_conflict) = match expected {
        ExpectedClass::Transient => (true, false, false),
        ExpectedClass::Permanent => (false, true, false),
        ExpectedClass::Conflict => (false, false, true),
    };
    assert_eq!(
        err.is_transient(),
        want_transient,
        "{shape}: is_transient() must be {want_transient} for an injected {expected:?} — the \
         retry path branches on exactly this, so a disagreement either turns a terminal failure \
         into a retry storm or abandons a retryable one. Got {err:?}",
    );
    assert_eq!(
        err.is_permanent(),
        want_permanent,
        "{shape}: is_permanent() must be {want_permanent} for an injected {expected:?}; got \
         {err:?}",
    );
    assert_eq!(
        err.is_conflict(),
        want_conflict,
        "{shape}: is_conflict() must be {want_conflict} for an injected {expected:?} — a false \
         positive here sends the checkpoint loop off to bump snap_seq and retry. Got {err:?}",
    );
    assert!(
        !err.is_not_found(),
        "{shape}: an injected {expected:?} must NOT classify as not-found — nothing on this path \
         resolved to an absent key, and absence is Ok(None) on this trait anyway. Got {err:?}",
    );
}

/// Case #35 — `get_layer_checkpoint` surfaces a backend failure with its
/// classification intact, on both kinds of handle.
///
/// **Why the new method needs its own case.** Group G's read-side coverage
/// (#20) goes through `get_latest_layer_checkpoint`, which is a different method
/// with a different first outbound request — a LIST, where get-by-handle issues
/// none. Nothing about #20 constrains `get_layer_checkpoint`: an adapter could
/// forget its `maybe_inject("get_layer_checkpoint")` call entirely, or route the
/// failure at the wrong layer, and #20 would stay green. Epic 4's lineage gather
/// reads *every* generation it replays through this method, so a
/// misclassification here is what decides whether a transient backend blip
/// aborts a restore or is retried.
///
/// **Both sub-shapes address a generation that was never written.** There is no
/// setup put, deliberately, for two independent reasons: the StorageApi
/// wrapper's override is route-level, so a seeding GET on the same route would
/// fail too; and an un-injected `get_layer_checkpoint` on an absent generation
/// returns `Ok(None)` (case #29), so the error the assertions consume cannot
/// have come from a natural absent-path result. It can only have come from the
/// seam.
///
/// **Reading a failure.** An `Ok(_)` here means the adapter is not consulting
/// the seam — for InMemory / LocalDisk a missing
/// `maybe_inject("get_layer_checkpoint")` at the top of the method body; for
/// StorageApi a mis-routed override (that wrapper mounts the failure on the
/// by-address GET, since get-by-handle never enumerates).
///
/// Sub-shape 1 drives a `Base` handle with an injected `Transient`; sub-shape 2
/// drives a `Tail` handle with an injected `Permanent`. The kind is varied
/// across the two so the case also witnesses that classification does not depend
/// on which artifact family the handle resolves to — the label is kind-blind and
/// the behaviour must be too.
#[cfg(feature = "test-checkpoint-fault-injection")]
pub async fn case_35_get_layer_checkpoint_error_classification<F, Fut>(factory: F)
where
    F: Fn(Arc<dyn CheckpointFaultInjector>) -> Fut,
    Fut: Future<Output = CheckpointStoreInjectedPair>,
{
    // ---- Sub-shape 1: Transient, on a Base handle ------------------------
    {
        let injector: Arc<dyn CheckpointFaultInjector> =
            Arc::new(OpTargetedFaultInjector::new("get_layer_checkpoint", || {
                CheckpointError::Transient {
                    source: Box::new(SyntheticBackendError(
                        "injected transient failure on get_layer_checkpoint",
                    )),
                }
            }));
        let (store, _guard) = factory(injector).await;

        let handle = CheckpointHandle::base(35_035, 1, "case-35-never-written-base".to_string());
        let err = store.get_layer_checkpoint(&handle).await.expect_err(
            "case_35 [transient/base]: get_layer_checkpoint must surface the injected Transient \
             error; got Ok(_) — the adapter is not consulting its fault-injection seam \
             (InMemory / LocalDisk: a missing maybe_inject(\"get_layer_checkpoint\") at the top \
             of the method body; StorageApi: an override mounted on a route this op never \
             requests). The generation was never written, so an un-injected call would have \
             returned Ok(None) and this assertion cannot be satisfied by the absent path",
        );
        assert_exclusively_classified(&err, ExpectedClass::Transient, "case_35 [transient/base]");
    }

    // ---- Sub-shape 2: Permanent, on a Tail handle ------------------------
    {
        let injector: Arc<dyn CheckpointFaultInjector> =
            Arc::new(OpTargetedFaultInjector::new("get_layer_checkpoint", || {
                CheckpointError::Permanent {
                    source: Box::new(SyntheticBackendError(
                        "injected permanent failure on get_layer_checkpoint",
                    )),
                }
            }));
        let (store, _guard) = factory(injector).await;

        let handle = CheckpointHandle::tail(35_035, 9, "case-35-never-written-tail".to_string());
        let err = store.get_layer_checkpoint(&handle).await.expect_err(
            "case_35 [permanent/tail]: get_layer_checkpoint must surface the injected Permanent \
             error on a tail handle too; got Ok(_) — the op label is kind-blind, so a tail \
             get that bypasses the seam (or, on StorageApi, a tail read routed somewhere the \
             override does not cover) is the defect this asserts against. The tail was never \
             written, so an un-injected call would have returned Ok(None)",
        );
        assert_exclusively_classified(&err, ExpectedClass::Permanent, "case_35 [permanent/tail]");
    }
}

/// Case #36 — a failing **tail** put surfaces its classification intact and
/// leaves the object with nothing listed.
///
/// **Why the tail path needs its own case when #19 / #22 already cover
/// `put_layer_checkpoint`.** Those drive the *base* branch. The tail branch is a
/// different sequence of artifact writes (meta-first, then the envelope; no
/// ext-refs sidecar at all), so an adapter could classify a base put's failure
/// correctly and still mishandle a tail's — most consequentially by leaving the
/// `.tail.meta.json` published after the envelope write failed, which is a
/// listed generation whose payload does not exist. The `list_layer_checkpoints`
/// assertion below is the trait-level shadow of that meta-first publish
/// obligation: whatever the adapter did internally, a failed tail put must not
/// be visible to a lineage gather as a generation it can fetch.
///
/// **Label policy.** The injector targets `"put_layer_checkpoint"` — the same
/// label the base cases use, because op labels name the method and never the
/// artifact kind. The injector therefore fires for base and tail puts alike,
/// which is exactly why the body's *only* write is the tail put: that is a
/// design constraint of the case, not an incidental shape. Seeding a base first
/// would trip the same injector (and, on StorageApi, the same route-level
/// override). `validate_tail_put` is presence-only — it never looks for the
/// named base — so a lineage-complete tail meta is accepted on an empty store
/// and no seeding is required.
///
/// **No Permanent sub-shape, deliberately.** `Permanent` on a tail put is
/// already pinned by case #32 through the production `validate_tail_put` path —
/// a real rejection from real code, which is strictly stronger evidence than an
/// injected one. Adding an injected Permanent here would assert the injector
/// works, not that the adapter does.
#[cfg(feature = "test-checkpoint-fault-injection")]
pub async fn case_36_tail_put_error_classification<F, Fut>(factory: F)
where
    F: Fn(Arc<dyn CheckpointFaultInjector>) -> Fut,
    Fut: Future<Output = CheckpointStoreInjectedPair>,
{
    /// The lineage the tail fixtures name, and the sequence they cover. The
    /// base at `BASE_SEQ` is never written — `validate_tail_put` is
    /// presence-only, and writing one would trip the very injector under test.
    const BASE_SEQ: u64 = 2;
    const FROM_SEQ: u64 = 3;
    const TO_SEQ: u64 = 7;
    const OBJECT_ID: u64 = 36_036;

    /// A failed tail put must leave nothing a lineage gather could list. Runs
    /// on a store whose only write attempt was the injected tail put, so a
    /// non-empty listing can only be that put's residue.
    async fn assert_nothing_listed(store: &Arc<dyn CheckpointStore>, shape: &str) {
        let listed = store
            .list_layer_checkpoints(OBJECT_ID)
            .await
            .unwrap_or_else(|err| {
                panic!(
                    "case_36 [{shape}]: list_layer_checkpoints must succeed — the injector \
                        targets put_layer_checkpoint only, so the listing path is unaffected. \
                        Got {err:?}"
                )
            });
        assert!(
            listed.is_empty(),
            "case_36 [{shape}]: a failed tail put MUST leave the object with no listed \
             generations. A listed tail here is a half-published one — its meta durable, its \
             envelope not — which a lineage gather would select and then fail to fetch. Got \
             {listed:?}",
        );
    }

    // ---- Sub-shape 1: Transient ------------------------------------------
    {
        let injector: Arc<dyn CheckpointFaultInjector> =
            Arc::new(OpTargetedFaultInjector::new("put_layer_checkpoint", || {
                CheckpointError::Transient {
                    source: Box::new(SyntheticBackendError(
                        "injected transient failure on a tail put",
                    )),
                }
            }));
        let (store, _guard) = factory(injector).await;

        let err = store
            .put_layer_checkpoint(
                OBJECT_ID,
                TO_SEQ,
                sample_tail_envelope(0x3601),
                sample_tail_meta(BASE_SEQ, FROM_SEQ, TO_SEQ),
            )
            .await
            .expect_err(
                "case_36 [transient]: a fully-valid tail put must surface the injected Transient \
                 error; got Ok(_) — the adapter is not consulting its fault-injection seam on the \
                 tail branch of put_layer_checkpoint (the seam is consulted at the top of the \
                 method, before any kind branching, so a tail put that succeeds under an active \
                 put_layer_checkpoint injection means the tail branch runs ahead of it)",
            );
        assert_exclusively_classified(&err, ExpectedClass::Transient, "case_36 [transient]");
        assert_nothing_listed(&store, "transient").await;
    }

    // ---- Sub-shape 2: Conflict -------------------------------------------
    {
        let injector: Arc<dyn CheckpointFaultInjector> =
            Arc::new(OpTargetedFaultInjector::new("put_layer_checkpoint", || {
                CheckpointError::Conflict
            }));
        let (store, _guard) = factory(injector).await;

        let err = store
            .put_layer_checkpoint(
                OBJECT_ID,
                TO_SEQ,
                sample_tail_envelope(0x3602),
                sample_tail_meta(BASE_SEQ, FROM_SEQ, TO_SEQ),
            )
            .await
            .expect_err(
                "case_36 [conflict]: a fully-valid tail put must surface the injected Conflict \
                 error; got Ok(_) — the adapter is not consulting its fault-injection seam on the \
                 tail branch of put_layer_checkpoint",
            );
        assert_exclusively_classified(&err, ExpectedClass::Conflict, "case_36 [conflict]");
        assert_nothing_listed(&store, "conflict").await;
    }
}

/// Case #39 — `validate_generation_identity` (b.hka (b)): a generation whose
/// `session_id` or `layer_name` is empty is rejected `Permanent` on **every**
/// adapter and on **both** kinds, and nothing is written.
///
/// # Why the store is the enforcement point
///
/// The producer-side refusal lives in `checkpoint::assembler`, which declines
/// to assemble a generation whose `obj:hdr` vanished (or carries empty identity
/// columns) mid-assembly. That is where the *decision* belongs. This gate is
/// where the *invariant* is enforced: it holds for every producer, on every
/// backend, including a call site that has not been written yet.
///
/// The invariant is worth a cross-adapter case because an identity-empty
/// generation is **permanently unrestorable, silently**. The restore path's
/// identity guard (`checkpoint::apply::live_mapping_resolves_blob_identity`)
/// refuses a blob whose `session_id` / `layer_name` it cannot verify — it
/// cannot rebuild `obj:hdr` without inventing session metadata, and inventing
/// it is how a restore lands a layer under the wrong session. So the write
/// costs a PUT, occupies a retention slot a restorable generation could have
/// held, and yields an artifact whose only possible future is to fail a
/// restore.
///
/// # Both kinds
///
/// Unlike case #32's gate, which returns early for a base, this one is
/// kind-independent. A tail is not exempt on the grounds that "the base carries
/// the identity": `tail_capture` sources a tail meta's identity columns from
/// the same `obj:hdr` read the assembler uses, and the reconciliation and
/// lineage surfaces read a tail's meta directly.
///
/// # What this case can and cannot see
///
/// It asserts the classification and that nothing became visible on any read
/// surface. The stronger **zero backend requests** claim is adapter-specific —
/// only StorageApi has a request log — and is pinned there, beside the
/// `validate_tail_put` equivalent, in
/// `storage_api::tests::layer_checkpoint_method_tests::put_layer_checkpoint_with_an_empty_*`.
/// A `Permanent` raised *after* a partial publish would satisfy every assertion
/// below on a local adapter and leave a durable orphan on a real deployment,
/// which is exactly why that half is not left to this case.
pub async fn case_39_identity_empty_puts_are_permanent_and_write_nothing<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    const BASE_SEQ: u64 = 4;
    const TO_SEQ: u64 = 9;

    /// The same four-way exclusive classification case #32 pins for the sibling
    /// gate. `Permanent` and not `Conflict` / `Transient`: this is a caller
    /// bug, and re-issuing the identical call reproduces it exactly.
    fn assert_permanent_rejection(err: &CheckpointError, shape: &str) {
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "case_39 [{shape}]: the variant must literally be CheckpointError::Permanent {{ .. }} \
             — validate_generation_identity (the single centralized gate in \
             src/checkpoint/adapters/mod.rs that every adapter calls) returns exactly that; \
             got {err:?}",
        );
        assert!(err.is_permanent(), "case_39 [{shape}]: got {err:?}");
        assert!(
            !err.is_transient(),
            "case_39 [{shape}]: a transient classification would turn a deterministic caller bug \
             into a retry storm. Got {err:?}",
        );
        assert!(
            !err.is_conflict(),
            "case_39 [{shape}]: nothing raced — a conflict verdict would invite a retry that can \
             only fail again. Got {err:?}",
        );
        assert!(
            !err.is_not_found(),
            "case_39 [{shape}]: nothing was looked up. Got {err:?}",
        );
    }

    /// Every read surface that could reveal a partial write, on a store whose
    /// only put was the rejected one.
    async fn assert_nothing_written(store: &Arc<dyn CheckpointStore>, object_id: u64, shape: &str) {
        let listed = store
            .list_layer_checkpoints(object_id)
            .await
            .unwrap_or_else(|err| {
                panic!("case_39 [{shape}]: list_layer_checkpoints must succeed; got {err:?}")
            });
        assert!(
            listed.is_empty(),
            "case_39 [{shape}]: the gate runs before any artifact is published, so a listed \
             generation means the adapter wrote first and validated second. Got {listed:?}",
        );

        match store.get_latest_layer_checkpoint(object_id).await {
            Ok(None) => {}
            Ok(Some(blob)) => panic!(
                "case_39 [{shape}]: get_latest_layer_checkpoint must be Ok(None) on an object \
                 whose only put was rejected; got Some(handle={:?})",
                blob.handle,
            ),
            Err(err) => panic!(
                "case_39 [{shape}]: get_latest_layer_checkpoint must be Ok(None); got {err:?}",
            ),
        }

        let object_ids = store.list_all_object_ids().await.unwrap_or_else(|err| {
            panic!("case_39 [{shape}]: list_all_object_ids must succeed; got {err:?}")
        });
        assert!(
            !object_ids.contains(&object_id),
            "case_39 [{shape}]: an object whose only put was rejected must not be enumerable — \
             an enumerable stem with no restorable generation is precisely what the \
             reconciliation orphan sweep then has to reason about. Got {object_ids:?}",
        );
    }

    // ---- Shape 1: a BASE with an empty session_id -------------------------
    {
        let (store, _guard) = factory().await;
        let object_id: u64 = 39_001;
        let meta = CheckpointMeta {
            session_id: String::new(),
            ..sample_meta(BASE_SEQ)
        };

        let err = store
            .put_layer_checkpoint(object_id, BASE_SEQ, sample_layer_ompf(0x3901), meta)
            .await
            .expect_err(
                "case_39 [base, empty session_id]: the apply path refuses a blob it cannot \
                 identify, so persisting one produces a permanently unrestorable artifact — an \
                 Ok here means this adapter skipped the shared gate",
            );
        assert_permanent_rejection(&err, "base, empty session_id");
        assert_nothing_written(&store, object_id, "base, empty session_id").await;
    }

    // ---- Shape 2: a BASE with an empty layer_name -------------------------
    {
        let (store, _guard) = factory().await;
        let object_id: u64 = 39_002;
        let meta = CheckpointMeta {
            layer_name: String::new(),
            ..sample_meta(BASE_SEQ)
        };

        let err = store
            .put_layer_checkpoint(object_id, BASE_SEQ, sample_layer_ompf(0x3902), meta)
            .await
            .expect_err(
                "case_39 [base, empty layer_name]: both identity columns are required — a \
                 restore that knows the session but not which layer cannot rebuild obj:hdr \
                 either",
            );
        assert_permanent_rejection(&err, "base, empty layer_name");
        assert_nothing_written(&store, object_id, "base, empty layer_name").await;
    }

    // ---- Shape 3: a TAIL with an empty session_id -------------------------
    {
        let (store, _guard) = factory().await;
        let object_id: u64 = 39_003;
        let meta = CheckpointMeta {
            session_id: String::new(),
            ..sample_tail_meta(BASE_SEQ, BASE_SEQ, TO_SEQ)
        };

        let err = store
            .put_layer_checkpoint(object_id, TO_SEQ, sample_tail_envelope(0x3903), meta)
            .await
            .expect_err(
                "case_39 [tail, empty session_id]: tails are NOT exempt — unlike \
                 validate_tail_put, this gate does not return early for a kind. A tail's \
                 identity is sourced from the same obj:hdr read the assembler uses, and the \
                 lineage and reconciliation surfaces read a tail's meta directly",
            );
        assert_permanent_rejection(&err, "tail, empty session_id");
        assert_nothing_written(&store, object_id, "tail, empty session_id").await;
    }

    // ---- Shape 4: a TAIL with an empty layer_name -------------------------
    {
        let (store, _guard) = factory().await;
        let object_id: u64 = 39_004;
        let meta = CheckpointMeta {
            layer_name: String::new(),
            ..sample_tail_meta(BASE_SEQ, BASE_SEQ, TO_SEQ)
        };

        let err = store
            .put_layer_checkpoint(object_id, TO_SEQ, sample_tail_envelope(0x3904), meta)
            .await
            .expect_err("case_39 [tail, empty layer_name]: the same rule, the other column");
        assert_permanent_rejection(&err, "tail, empty layer_name");
        assert_nothing_written(&store, object_id, "tail, empty layer_name").await;
    }

    // ---- Negative control --------------------------------------------------
    //
    // Without this, an adapter that returned Permanent from EVERY
    // put_layer_checkpoint would pass all four shapes above. It also pins that
    // the gate reads the two columns and nothing standing in for them: the
    // control's meta differs from shapes 1-4 in exactly those fields.
    {
        let (store, _guard) = factory().await;
        let object_id: u64 = 39_005;

        let base_handle = store
            .put_layer_checkpoint(
                object_id,
                BASE_SEQ,
                sample_layer_ompf(0x3905),
                sample_meta(BASE_SEQ),
            )
            .await
            .expect(
                "case_39 [negative control]: an identity-carrying base is the ordinary shape and \
                 MUST succeed — a failure here means the gate was written too broadly and the \
                 four rejections above proved nothing",
            );
        assert_eq!(base_handle.kind, CheckpointKind::Base);

        let tail_handle = store
            .put_layer_checkpoint(
                object_id,
                TO_SEQ,
                sample_tail_envelope(0x3906),
                sample_tail_meta(BASE_SEQ, BASE_SEQ, TO_SEQ),
            )
            .await
            .expect(
                "case_39 [negative control]: and an identity-carrying tail on the same store — \
                 the kind-independent gate must admit both kinds, not merely fail to reject \
                 bases",
            );
        assert_eq!(tail_handle.kind, CheckpointKind::Tail);

        let listed = store
            .list_layer_checkpoints(object_id)
            .await
            .expect("case_39 [negative control]: list_layer_checkpoints must succeed");
        assert_eq!(
            listed.len(),
            2,
            "case_39 [negative control]: both admitted generations must be durable; got \
             {listed:?}",
        );
    }
}

// ----- Group M: the DR restore Job's read surface (#40–#42) -----------------

/// Per-adapter fixture for case #40.
///
/// Like [`ArtifactFixture`], it lets an adapter wrapper put the store into a
/// state the trait cannot reach: a file in the session-backup namespace that
/// is not a session backup.
pub struct SessionListingFixture {
    pub store: Arc<dyn CheckpointStore>,
    pub guard: Box<dyn CheckpointStoreGuard>,
    /// Plant a file named `filename` directly in the adapter's session-backup
    /// namespace (`<root>/.live-edit/sessions/`), bypassing the trait — what a
    /// write that died before its rename, or a foreign tool, leaves behind.
    /// `None` is a positive claim, not a skip: the adapter keeps nothing but
    /// committed session backups in that namespace, so there is nothing a
    /// listing could wrongly report.
    pub plant_in_sessions: Option<Box<dyn Fn(String) -> PlantFuture + Send + Sync>>,
}

/// The session id of case #40's planted write-temp. It never has a committed
/// backup, so it must never be listed.
pub const CASE_40_GHOST_SESSION: &str = "sess-ghost";

/// The UUID disambiguator in case #40's planted write-temp — the
/// `<session_id>.json.<uuid>.tmp` shape LocalDisk's write-temp-then-rename
/// produces.
pub const CASE_40_TEMP_UUID: &str = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9";

/// Case #40 — `list_session_checkpoint_ids` returns **exactly** the ids of the
/// stored session backups: never a write-temp, a layer artifact, an ext blob,
/// the counters sidecar or a stray file, and `Ok(vec![])` on an empty store.
///
/// **Branches.**
///
/// 1. **Empty store** → `Ok(vec![])`, not an error. (On the StorageApi fake
///    this passes through the 404 the fake answers for an empty prefix, as
///    the real service does.)
/// 2. **Three backups among every other artifact kind** — a base with an
///    ext-refs sidecar, a tail, an ext blob, the counters sidecar, and, on an
///    adapter with a session namespace, a planted
///    `<CASE_40_GHOST_SESSION>.json.<uuid>.tmp` and a stray `README.txt` →
///    exactly the three ids, each once. A dotted id (`sess.v2`) keeps its
///    inner dots: only the trailing `.json` is stripped. Every listed id
///    resolves through `get_session_checkpoint`, which is what "exactly the
///    ids that have a stored backup" means operationally.
/// 3. **Overwrite and delete** → re-putting a backup does not list it twice,
///    and a deleted session drops out.
/// 4. **Every backup deleted, strays still present** → `Ok(vec![])` again.
///    The strays are what make this branch discriminating: a listing that
///    counted every file in the namespace would be non-empty here.
pub async fn case_40_list_session_checkpoint_ids_returns_exactly_the_stored_session_ids<F, Fut>(
    factory: F,
) where
    F: Fn() -> Fut,
    Fut: Future<Output = SessionListingFixture>,
{
    use std::collections::BTreeSet;

    /// The three session ids branch 2 stores.
    const SESSIONS: [&str; 3] = ["sess-alpha", "sess-beta", "sess.v2"];
    /// The object the neighbouring layer generations are stored under.
    const OBJECT_ID: u64 = 40_040;

    /// List, asserting the call succeeds and names no session twice.
    async fn listed(store: &Arc<dyn CheckpointStore>, shape: &str) -> BTreeSet<String> {
        let ids = store
            .list_session_checkpoint_ids()
            .await
            .unwrap_or_else(|err| {
                panic!("case_40 [{shape}]: list_session_checkpoint_ids must succeed; got {err:?}")
            });
        let count = ids.len();
        let set: BTreeSet<String> = ids.into_iter().collect();
        assert_eq!(
            set.len(),
            count,
            "case_40 [{shape}]: each session id must be listed once; got {count} entries for {} \
             distinct ids: {set:?}",
            set.len(),
        );
        set
    }

    fn ids(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    // ---- Branch 1: empty store -------------------------------------------
    {
        let fixture = factory().await;
        assert_eq!(
            listed(&fixture.store, "empty store").await,
            BTreeSet::new(),
            "case_40 [empty store]: a store with no session backups answers Ok(vec![]) — absence \
             is not an error, and the DR restore Job reads this as 'nothing to restore'",
        );
        drop(fixture.guard);
    }

    let fixture = factory().await;
    let store = Arc::clone(&fixture.store);

    // ---- Branch 2: three backups among every other artifact kind ---------
    for (i, session_id) in SESSIONS.iter().enumerate() {
        store
            .put_session_checkpoint(
                session_id,
                sample_session_body(0x4000 + i as u64),
                sample_meta(0),
            )
            .await
            .unwrap_or_else(|err| {
                panic!("case_40: put_session_checkpoint({session_id}) must succeed; got {err:?}")
            });
    }
    let mut base_meta = sample_meta(4);
    base_meta.sidecar_bytes = Some(ext_refs_sidecar_for(&["case-40-referenced-hash"]));
    store
        .put_layer_checkpoint(OBJECT_ID, 4, sample_layer_ompf(4), base_meta)
        .await
        .expect("case_40: the neighbouring base put must succeed");
    store
        .put_layer_checkpoint(
            OBJECT_ID,
            9,
            sample_tail_envelope(9),
            sample_tail_meta(4, 4, 9),
        )
        .await
        .expect("case_40: the neighbouring tail put must succeed");
    store
        .put_ext_blob(&sample_hash_hex(0x40), sample_ext_bytes(0x40))
        .await
        .expect("case_40: the neighbouring ext-blob put must succeed");
    store
        .put_counters_sidecar(sample_counters_body(0x40))
        .await
        .expect("case_40: the counters-sidecar put must succeed");
    if let Some(plant) = fixture.plant_in_sessions.as_ref() {
        plant(format!(
            "{CASE_40_GHOST_SESSION}.json.{CASE_40_TEMP_UUID}.tmp"
        ))
        .await;
        plant("README.txt".to_owned()).await;
    }

    let after_puts = listed(&store, "three backups").await;
    assert_eq!(
        after_puts,
        ids(&SESSIONS),
        "case_40 [three backups]: the listing must be exactly the stored session ids — no \
         write-temp (`{CASE_40_GHOST_SESSION}` has no committed backup), no layer artifact, ext \
         blob, counters sidecar or stray file, and a dotted id with only its trailing `.json` \
         stripped",
    );
    for session_id in &after_puts {
        assert!(
            store
                .get_session_checkpoint(session_id)
                .await
                .unwrap_or_else(|err| panic!(
                    "case_40: get_session_checkpoint({session_id}) must succeed; got {err:?}"
                ))
                .is_some(),
            "case_40: every listed id must have a stored backup; `{session_id}` does not",
        );
    }
    if fixture.plant_in_sessions.is_some() {
        assert!(
            store
                .get_session_checkpoint(CASE_40_GHOST_SESSION)
                .await
                .expect("case_40: get_session_checkpoint on the ghost id must succeed")
                .is_none(),
            "case_40: the planted write-temp is not a backup, so its session has none — which is \
             exactly why the listing must not name it",
        );
    }

    // ---- Branch 3: overwrite and delete ----------------------------------
    store
        .put_session_checkpoint("sess-alpha", sample_session_body(0x4100), sample_meta(0))
        .await
        .expect("case_40: re-putting sess-alpha must succeed (last-writer-wins)");
    store
        .delete_session_checkpoint("sess-beta")
        .await
        .expect("case_40: delete_session_checkpoint(sess-beta) must succeed");
    assert_eq!(
        listed(&store, "after overwrite and delete").await,
        ids(&["sess-alpha", "sess.v2"]),
        "case_40 [after overwrite and delete]: an overwritten backup is still one backup, and a \
         deleted one is gone — the DR restore Job must not try to restore a session whose backup \
         DeleteSession removed",
    );

    // ---- Branch 4: every backup deleted, strays still present ------------
    for session_id in ["sess-alpha", "sess.v2"] {
        store
            .delete_session_checkpoint(session_id)
            .await
            .unwrap_or_else(|err| {
                panic!("case_40: delete_session_checkpoint({session_id}) must succeed; got {err:?}")
            });
    }
    assert_eq!(
        listed(&store, "every backup deleted").await,
        BTreeSet::new(),
        "case_40 [every backup deleted]: with no backups left the listing is empty again, \
         whatever else the session namespace still holds",
    );

    drop(fixture.guard);
}

/// Case #41 — `ext_blob_exists` answers `Ok(true)` only for a hash whose bytes
/// are stored and `Ok(false)` for one whose bytes are not, never
/// `Err(NotFound)`.
///
/// **Why both halves.** The DR restore Job strips a value it is told is
/// absent, so a false `Ok(false)` loses data, and a false `Ok(true)` leaves a
/// value whose bytes are gone. Each half is discriminating on its own: the
/// `never_put` probe beside a stored blob fails an adapter that answers `true`
/// for everything, and the stored probe fails one that answers `false` for
/// everything — on StorageApi, a fake without a HEAD route does exactly that,
/// because wiremock answers an unmatched HEAD 404.
///
/// **Bytes, not references.** A hash a layer's ext-refs sidecar names but
/// whose bytes were never put is absent: the Job's question is whether the
/// bytes can be read back, and a reference proves nothing about that.
pub async fn case_41_ext_blob_exists_is_true_only_for_stored_bytes<F, Fut>(factory: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    async fn assert_answer(
        store: &Arc<dyn CheckpointStore>,
        hash_hex: &str,
        expected: bool,
        shape: &str,
    ) {
        match store.ext_blob_exists(hash_hex).await {
            Ok(answer) => assert_eq!(
                answer, expected,
                "case_41 [{shape}]: ext_blob_exists must answer Ok({expected}); got Ok({answer})",
            ),
            Err(CheckpointError::NotFound) => panic!(
                "case_41 [{shape}]: an absent blob is Ok(false), never CheckpointError::NotFound \
                 (trait contract)"
            ),
            Err(err) => panic!("case_41 [{shape}]: ext_blob_exists must succeed; got {err:?}"),
        }
    }

    let (store, _guard) = factory().await;

    let stored = sample_hash_hex(0x4101);
    let never_put = sample_hash_hex(0x4102);
    let referenced_only = sample_hash_hex(0x4103);

    assert_answer(&store, &stored, false, "fresh store").await;

    store
        .put_ext_blob(&stored, sample_ext_bytes(0x4101))
        .await
        .expect("case_41: put_ext_blob must succeed");
    assert_answer(&store, &stored, true, "after put").await;
    assert_answer(
        &store,
        &never_put,
        false,
        "a hash never put, beside a stored one",
    )
    .await;

    let mut meta = sample_meta(1);
    meta.sidecar_bytes = Some(ext_refs_sidecar_for(&[referenced_only.as_str()]));
    store
        .put_layer_checkpoint(41_041, 1, sample_layer_ompf(1), meta)
        .await
        .expect("case_41: the referencing base put must succeed");
    assert_answer(
        &store,
        &referenced_only,
        false,
        "referenced by a sidecar, bytes never put",
    )
    .await;

    store
        .delete_ext_blob(&stored)
        .await
        .expect("case_41: delete_ext_blob must succeed");
    assert_answer(&store, &stored, false, "after delete").await;
}

/// Case #42 — a backend failure on either DR restore Job read surfaces as its
/// classified error, never as the negative answer (`Ok(vec![])` from the
/// listing, `Ok(false)` from the existence probe).
///
/// The Job counts an error as "unknown" and a negative answer as a fact it
/// acts on, so folding a failure into the negative answer is the one
/// classification bug with data loss behind it.
///
/// **No setup writes, deliberately.** Each sub-shape runs on an empty store,
/// where an un-injected call answers `Ok(vec![])` / `Ok(false)` — the exact
/// answers this case forbids — so an `Ok` can only mean the call bypassed the
/// seam. On StorageApi the override is route-level (the `sessions/` LIST, or
/// the by-address HEAD), so a setup write on the same route would fail too.
///
/// Sub-shapes: each method × `Transient` and `Permanent`, the two classes an
/// adapter produces for a read (a read never conflicts).
#[cfg(feature = "test-checkpoint-fault-injection")]
pub async fn case_42_dr_restore_read_methods_surface_backend_failures_as_errors<F, Fut>(factory: F)
where
    F: Fn(Arc<dyn CheckpointFaultInjector>) -> Fut,
    Fut: Future<Output = CheckpointStoreInjectedPair>,
{
    for (op, expected) in [
        ("list_session_checkpoint_ids", ExpectedClass::Transient),
        ("list_session_checkpoint_ids", ExpectedClass::Permanent),
        ("ext_blob_exists", ExpectedClass::Transient),
        ("ext_blob_exists", ExpectedClass::Permanent),
    ] {
        let shape = format!("case_42 [{op}, {expected:?}]");
        let injector: Arc<dyn CheckpointFaultInjector> =
            Arc::new(OpTargetedFaultInjector::new(op, move || match expected {
                ExpectedClass::Transient => CheckpointError::Transient {
                    source: Box::new(SyntheticBackendError(
                        "injected transient failure on a DR restore read",
                    )),
                },
                ExpectedClass::Permanent => CheckpointError::Permanent {
                    source: Box::new(SyntheticBackendError(
                        "injected permanent failure on a DR restore read",
                    )),
                },
                ExpectedClass::Conflict => unreachable!("case_42 injects no Conflict"),
            }));
        let (store, _guard) = factory(injector).await;

        let err = if op == "list_session_checkpoint_ids" {
            match store.list_session_checkpoint_ids().await {
                Err(err) => err,
                Ok(ids) => panic!(
                    "{shape}: list_session_checkpoint_ids must surface the injected failure; got \
                     Ok({ids:?}) — the adapter is not consulting its fault-injection seam, or it \
                     folded the failure into 'no session backups'"
                ),
            }
        } else {
            match store.ext_blob_exists(&sample_hash_hex(0x4201)).await {
                Err(err) => err,
                Ok(answer) => panic!(
                    "{shape}: ext_blob_exists must surface the injected failure; got \
                     Ok({answer}) — the adapter is not consulting its fault-injection seam, or it \
                     folded the failure into an answer about the bytes"
                ),
            }
        };
        assert_exclusively_classified(&err, expected, &shape);
    }
}

/// Case #43 — `list_live_ext_set` reports, beside the live hash set, how many
/// layer artifacts its OWN enumeration named: 0 on an empty store, and every
/// layer artifact the walk listed otherwise; and `list_referenced_ext_hashes`
/// answers exactly its hash set.
///
/// **Why the count matters.** The reconciliation collect deletes an aged
/// tombstoned blob whenever the live set does not name it, so an EMPTY set
/// would license deleting every one — and an empty set is also what a store
/// at the wrong address, or a listing that answers a stale not-found, returns.
/// The collect therefore deletes against an empty set only when this count is
/// non-zero. An adapter that reported a non-zero count for an empty store would
/// re-open that hole; one that reported 0 beside a listing that named artifacts
/// would leak every unreferenced blob on a store whose bases name none.
///
/// **Design.** Three shapes on one store, in order:
///
/// 1. **Empty store** → `(∅, 0)`.
/// 2. **Base 43 001 whose sidecar names `h1`, plus base 43 002 with NO
///    sidecar** → hashes `{h1}`, and a count equal to
///    `list_layer_artifacts().len()` (the same store, read again), which is
///    non-zero. The sidecar-less base is the load-bearing half: it contributes
///    artifacts to the count and nothing to the set, so a count taken from the
///    sidecars alone would fall short.
/// 3. **Parity** — `list_referenced_ext_hashes` answers the same set: it is a
///    projection of the same walk.
pub async fn case_43_list_live_ext_set_counts_its_own_listing_and_projects_to_the_live_set<F, Fut>(
    factory: F,
) where
    F: Fn() -> Fut,
    Fut: Future<Output = CheckpointStorePair>,
{
    use std::collections::HashSet;

    let (store, _guard) = factory().await;

    let empty = store
        .list_live_ext_set()
        .await
        .expect("case_43: list_live_ext_set on an empty store must succeed");
    assert!(
        empty.hashes.is_empty() && empty.listed_layer_artifacts == 0,
        "case_43 [empty store]: must answer (∅, 0) — an empty set that vouches for no listing; \
         got {empty:?}"
    );

    let mut with_sidecar = sample_meta(1);
    with_sidecar.sidecar_bytes = Some(ext_refs_sidecar_for(&["h1"]));
    store
        .put_layer_checkpoint(43_001, 1, sample_layer_ompf(43_001), with_sidecar)
        .await
        .expect("case_43: the base with a sidecar must put");
    store
        .put_layer_checkpoint(43_002, 1, sample_layer_ompf(43_002), sample_meta(1))
        .await
        .expect("case_43: the base with no sidecar must put");

    let live = store
        .list_live_ext_set()
        .await
        .expect("case_43: list_live_ext_set must succeed");
    let artifacts = store
        .list_layer_artifacts()
        .await
        .expect("case_43: list_layer_artifacts must succeed");
    let expected: HashSet<String> = ["h1".to_string()].into_iter().collect();
    assert_eq!(
        live.hashes, expected,
        "case_43 [two bases]: the live set is the one sidecar's hashes"
    );
    assert_eq!(
        live.listed_layer_artifacts,
        artifacts.len(),
        "case_43 [two bases]: the count is every layer artifact the walk's listing named — what \
         list_layer_artifacts reports from the same store — the sidecar-less base included"
    );
    assert!(
        live.listed_layer_artifacts > 0,
        "case_43 [two bases]: a store holding two bases must vouch for its listing"
    );

    let projected = store
        .list_referenced_ext_hashes()
        .await
        .expect("case_43: list_referenced_ext_hashes must succeed");
    assert_eq!(
        projected, live.hashes,
        "case_43 [parity]: list_referenced_ext_hashes is list_live_ext_set's hash set"
    );
}
