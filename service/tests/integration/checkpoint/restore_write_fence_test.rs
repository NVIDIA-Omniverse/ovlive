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

//! b.h4m — **two replicas restoring the same layer concurrently may not merge
//! their rebuilds.**
//!
//! # The defect this file fences
//!
//! Layer restore is lazy and per-layer: two clients asking two replicas for a
//! layer whose Valkey state is gone drive two concurrent restores of the SAME
//! `object_id`. The admission lock the recovery hook holds is per-process, so
//! nothing held replica A off while replica B was writing the layer back.
//!
//! Before b.h4m both replicas materialized through the same unfenced
//! `LayerStore::write_object`, which `HSET`s / `SADD`s / `ZADD`s / `RPUSH`es its
//! members *into* whatever is there. A's rebuild landing during B's was
//! therefore **added** to B's rather than replacing it, and the layer came back
//! as a MIXTURE of two backup generations — internally plausible, no error
//! anywhere, no counter moving, and (because a total-loss restore has no residue
//! for the step-3b wipe to notice) widest on the commonest DR case of all.
//!
//! The fix is `LayerStore::write_restored_object`: a `WATCH` on
//! `le:{oid}:obj:hdr` armed before the `MULTI`, requiring the header to be
//! **absent**. Valkey serializes the two `EXEC`s on the `{object_id}` shard, so
//! whichever runs first creates the header and marks every other watcher dirty;
//! the second writes nothing and reports
//! `RestoreWriteOutcome::PeerLandedHeader`, which `apply_restore` maps to
//! `ApplyOutcome::AlreadyPresent`. Neither ordering can produce a union.
//!
//! # Why these tests need the seams, and why there are two of them
//!
//! The window is a handful of round trips wide and the losing replica must be
//! *inside* it when the winner's header lands. A racing second restore would
//! land somewhere arbitrary and reach the verdict by luck, so every arm here
//! drives the race deterministically through one of the two b.h4m hooks. They
//! differ only in which side of the fence's header read they hold the writer on,
//! and that difference is the whole point — the fence has TWO peer-landed arms
//! and each hook reaches exactly one:
//!
//! * `LayerStore::with_restore_write_watch_window_hook_for_test` fires after the
//!   `WATCH` is armed and BEFORE the pre-`MULTI` header read. A peer header
//!   landed from there is seen by that read, so the attempt exits at the
//!   pre-`MULTI` `PeerLandedHeader` arm: no `MULTI`, no `EXEC`, and the `WATCH`
//!   is never load-bearing. Arms 1–4.
//! * `LayerStore::with_restore_write_before_exec_hook_for_test` fires after that
//!   read passed EMPTY, after the queueing, after the L2 pre-dispatch check, and
//!   immediately before `txn.exec()`. A peer header landed from there lands in
//!   the read→`EXEC` window, so Valkey's `WATCH` is what notices it: the `EXEC`
//!   trips, and the attempt takes the `WatchedExec::Conflict` arm — the
//!   post-conflict positive-evidence re-read, its `PeerLandedHeader` /
//!   `Unavailable` split, and that read's own
//!   `classify_restore_header_read_error` call site. Arms 6–8, and nothing else
//!   in the repo reaches any of it.
//!
//! In both cases the peer's writes go out on the DATA pool — a different
//! connection, as a peer replica's would be, and the reason the second hook's
//! writes actually trip this writer's `WATCH` (Valkey marks every *other*
//! watcher of a modified key dirty) — while the restore under test still holds
//! its `txn_pool` slot. The peer's writes assert their own success inside the
//! hook, so a peer write that silently failed cannot masquerade as a
//! `Materialized` verdict with no signal: the panic unwinds through the restore
//! and the `WatchGuard`'s `Drop` still `UNWATCH`es.
//!
//! # The arms, and which fence layer each one proves
//!
//! | # | seam | what the peer lands inside the window | verdict |
//! |---|---|---|---|
//! | 1 | before the read | a structurally complete `obj:hdr` (the 12 canonical fields — a real peer's is a superset, see `peer_header_fields`) + its own section, over an EMPTY keyspace | pre-`MULTI` fence: `AlreadyPresent`, StatLayer `Ok` on the peer's header |
//! | 2 | before the read | the same, but the loser's step-3b residue wipe ran first | fenced identically; the wipe's own counter moved, and so did the fence's |
//! | 3 | before the read | `HSET superseded=1` alone — a racing `CreateLayer` overwrite | refused `Unavailable` ⇒ `UNAVAILABLE`; nothing written |
//! | 4 | before the read | `SET obj:hdr "…"` — a foreign Valkey type | refused `Permanent` ⇒ `INTERNAL`; nothing written |
//! | 5 | neither (planted before the apply) | a complete header already in place at step 1 | `stage="presence_probe"`; the materialization's window never opens |
//! | 6 | before `EXEC` | the same structurally complete 12-canonical-field `obj:hdr` + its own section | `EXEC` TRIPS ⇒ `Conflict` ⇒ re-read finds it complete ⇒ `AlreadyPresent` |
//! | 7 | before `EXEC` | `HSET superseded=1` alone | `EXEC` trips ⇒ re-read finds a PARTIAL header ⇒ `Unavailable` ⇒ `UNAVAILABLE` |
//! | 8 | before `EXEC` | `SET obj:hdr "…"` | `EXEC` trips ⇒ re-read is `WRONGTYPE` ⇒ `Permanent` ⇒ `INTERNAL` |
//!
//! Read down the `stage` column and the three fence layers separate cleanly.
//! Arm 5 is the OUTERMOST layer, `apply_restore`'s step-1 presence probe — a
//! point-in-time read, not a fence at all, and the one every restore runs. Arms
//! 1–4 are the materialization's pre-`MULTI` header read, which is a
//! *point-in-time read taken on the watched connection*: it catches a peer that
//! landed before it, and the `WATCH` contributes nothing to those verdicts. Arms
//! 6–8 are the only ones where the `WATCH` itself is the mechanism — the module
//! claim that "Valkey serializes the two `EXEC`s on the `{object_id}` shard, so
//! the second writes nothing" is exercised THERE and nowhere else. Arm 2 adds
//! the middle layer, the residue wipe's own `WATCH`, as a committed predecessor
//! rather than as the fence that fires.
//!
//! Arms 1, 2 and 6 are the regression proper: all three FAIL without the fence —
//! the loser's section appears alongside the winner's in `obj:secs`, the loser's
//! `created_by` overwrites the winner's, and
//! `live_edit_checkpoint_restore_fenced_total{stage="object_write"}` never
//! moves. They are separate arms and not one parameterised body because they
//! reach the same verdict down three different paths: arm 1 is the widest window
//! and the ONLY stage a keyspace with no residue can reach; arm 2 proves the
//! fence still holds *after* the step-3b wipe has already committed a
//! destructive `UNLINK`, the state in which a merge would be least recoverable;
//! and arm 6 is the one that proves the `WATCH` is real rather than decorative,
//! since a peer that lands after the read can be stopped by nothing else.
//!
//! Arms 3–4 and 7–8 are the fence's non-peer exits, once per header read,
//! because both reads own a copy of the decision and only the pre-`MULTI` one
//! classified `WRONGTYPE` before b.h4m. They are here because both shapes are
//! ones a fence that merely asked "is the header there" would get wrong in the
//! dangerous direction: burying a `superseded=1` under a restored header leaves
//! the layer permanently unwritable on both commit backends, and treating a
//! wrong-typed header as retriable re-drives, forever, a restore that cannot
//! succeed.
//!
//! # Wait budgets vs. nextest's hang-killer
//!
//! `.config/nextest.toml` marks a test slow at 40 s and hard-kills at 120 s, and
//! a SIGKILLed test prints nothing — so a diagnostic panic would be lost exactly
//! when it is needed. Every wait here is explicitly bounded:
//! [`CHECKPOINT_DEADLINE_SECS`] on the loop's first generation, [`RPC_BUDGET`]
//! on the driving RPC, [`LOOP_DRAIN_BUDGET`] on the ONE loop drain — which
//! happens inside [`build_fence_fixture`], before the wipe, not at the end of
//! the arm (see [`release_harness`]). Arms 1–8 sleep on no real interval and
//! none has a worst case beyond ~25 s. Arm 9 polls `CLIENT LIST` and sleeps one
//! [`PIPELINE_SETTLE`], with every wait of its own bounded by
//! [`STALL_STAGE_BUDGET`], which keeps it well inside the 120 s hard kill.
//!
//! # Arm 9 (b.21r)
//!
//! Arm 9 rides the before-`EXEC` hook with no peer at all: it stalls the
//! materialization's own `EXEC` reply to reach its in-doubt verdict, and pins
//! that a timed-out `EXEC` counts exactly one
//! `live_edit_valkey_errors_total{pool="txn", kind="command_timeout"}` while a
//! connection-loss in-doubt counts none. Its section comment covers the
//! mechanism.
//!
//! # Activation
//!
//! The whole module is `#[cfg(feature = "test-fault-injection")]`: the seam it
//! drives exists only under that feature, and a missing feature prints NOTHING
//! (unlike a missing env var, which prints a `SKIP:` line). A hand-rolled local
//! invocation must therefore pass
//! `--features test-support,test-fault-injection,test-checkpoint-fault-injection`
//! or this file reports green having covered nothing.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; see CLAUDE.md.

#![cfg(feature = "test-fault-injection")]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use fred::clients::Client;
use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use futures::FutureExt;
use tonic::Request;

use live_edit::grpc::layer::{handle_create_layer, handle_stat_layer};
use live_edit::grpc::proto::{StatLayerRequest, StatLayerResponse};
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::{command_timeout_row, ConnStall};
use live_edit::storage::watched_txn::WatchWindowHook;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::common::recovery_metric;

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and fixture constants
// ═════════════════════════════════════════════════════════════════════════════

/// How long the loop gets to land its first generation for the fixture layer.
/// The harness runs with `interval = quiesce = 0`, so the only latency to absorb
/// is one `tick = 1 s`.
const CHECKPOINT_DEADLINE_SECS: u64 = 10;

/// One bound over the driving RPC. Every arm's whole race — the restore, the
/// hook's peer writes, and the fence's verdict — happens inside this single
/// call, so one budget covers it and nothing multiplies.
const RPC_BUDGET: Duration = Duration::from_secs(10);

/// Budget for the ONE drain of the checkpoint loop, which
/// [`build_fence_fixture`] performs after the fixture's first generation lands and
/// before the wipe. Nothing later in an arm has a live loop to drain — see
/// [`release_harness`].
const LOOP_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// `created_by` on the fixture's SESSION. Deliberately not asserted on: the
/// layer's own `created_by` — the value a LOSING restore would write back into
/// `obj:hdr` — comes from the userinfo client the harness wires, not from here,
/// so each arm captures it from the live header instead
/// ([`FenceFixture::loser_created_by`]) rather than assuming the two agree.
const CREATED_BY: &str = "restore-write-fence-user";

/// `created_by` on the header the peer replica lands inside the window.
const PEER_CREATED_BY: &str = "restore-write-fence-peer-replica";

/// The section the fixture layer carries, and therefore the one the loser's
/// restore would materialize. Its presence in `obj:secs` after a fenced restore
/// IS the merge, expressed as a keyspace fact.
const LOSER_SECTION_ID: u64 = 42;

/// The section the peer's landed restore carries. Deliberately disjoint from
/// [`LOSER_SECTION_ID`] so a union is visible as two members rather than hiding
/// inside one.
const PEER_SECTION_ID: &str = "7777";

/// Section id of the header-less residue the second arm plants. Disjoint from
/// both generations for the same reason.
const RESIDUE_SECTION_ID: &str = "9999";

/// Hash the residue arm plants in `obj:ext`, standing in for a value row the
/// dead generation held. Content is irrelevant — only its presence in the
/// pre-image the fence reads is.
fn residue_ext_hash_hex() -> String {
    "ab".repeat(32)
}

// ═════════════════════════════════════════════════════════════════════════════
// The peer replica
// ═════════════════════════════════════════════════════════════════════════════

/// What the peer replica does when the fence's window opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PeerAction {
    /// A peer replica's restore COMPLETING: its own section, then all twelve
    /// canonical `obj:hdr` fields.
    ///
    /// Header LAST is deliberate and mirrors the production ordering it stands
    /// in for — a real peer's materialization is one `MULTI`, so no observer can
    /// see its header before its state. Writing the header first here would let
    /// the fence read a complete header over an empty keyspace, and the
    /// anti-union assertion would then be about a generation that does not
    /// exist.
    LandCompleteRestore,
    /// `HSET superseded=1` and nothing else. The most concrete real producer of
    /// a single-field header write: a racing `CreateLayer` overwrite sets
    /// exactly that flag, and burying it under a restored header leaves the
    /// layer permanently unwritable on both commit backends.
    LandSupersededFlagOnly,
    /// `SET obj:hdr "…"` — a foreign Valkey type where a hash belongs. Not a
    /// peer replica at all; the out-of-band writer the `WRONGTYPE_KEY` family
    /// exists for. Reachable ONLY from inside the window: the pre-`MULTI`
    /// keyspace type probe runs before the `WATCH` is armed and would otherwise
    /// have caught it, which is precisely why the fence needs a classifier of
    /// its own.
    RetypeHeaderToString,
}

/// Which of the fence's two windows an arm holds the losing writer in.
///
/// The pair is not a stylistic choice: it decides which of the fence's two
/// peer-landed arms the arm under test reaches, and therefore whether the
/// `WATCH` is load-bearing at all. See the module header's seam section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeamPosition {
    /// After the `WATCH` is armed, BEFORE the pre-`MULTI` header read. The read
    /// sees whatever the peer landed, so the attempt exits at the pre-`MULTI`
    /// `PeerLandedHeader` / refusal arms; no `MULTI` is opened and no `EXEC` is
    /// sent.
    BeforeHeaderRead,
    /// After the pre-`MULTI` read passed EMPTY, after the queueing, after the L2
    /// pre-dispatch check, immediately before `txn.exec()`. The peer's write
    /// therefore lands in the read→`EXEC` window and trips this writer's
    /// `WATCH`, which is the ONLY route to `WatchedExec::Conflict` and the
    /// post-conflict re-read behind it.
    BeforeExec,
}

/// Where the peer writes, resolved after the harness and the layer both exist.
struct PeerTarget {
    /// The DATA pool — a different connection from the guarded one the restore
    /// holds, which is what makes this a peer replica rather than the same
    /// session writing to itself.
    client: ValkeyClient,
    keys: KeySchema,
    object_id: u64,
}

/// The peer replica the fence's window hands control to.
///
/// The target is an [`OnceLock`] rather than a constructor argument because of
/// an ordering constraint that cannot be removed: the hook must be installed on
/// the ONE `LayerStore` the harness builds (see
/// `CheckpointHarnessOptions::restore_write_watch_window_hook`), which happens
/// before a `TestValkey` handle is returned to the caller and long before any
/// layer exists to have an id. The latch is filled the moment CreateLayer
/// returns; it is read only from inside the window, which is strictly later.
struct PeerLander {
    target: OnceLock<PeerTarget>,
    action: PeerAction,
    fired: AtomicUsize,
}

impl PeerLander {
    fn new(action: PeerAction) -> Self {
        Self {
            target: OnceLock::new(),
            action,
            fired: AtomicUsize::new(0),
        }
    }

    fn arm(&self, client: ValkeyClient, keys: KeySchema, object_id: u64) {
        if self
            .target
            .set(PeerTarget {
                client,
                keys,
                object_id,
            })
            .is_err()
        {
            panic!("the peer lander must be armed exactly once");
        }
    }

    fn fire_count(&self) -> usize {
        self.fired.load(Ordering::Acquire)
    }

    /// Land the peer's writes. Once only: the window is entered once per restore
    /// attempt, and a second firing would mean the restore re-drove internally —
    /// which would make every "the peer landed exactly this" state assertion
    /// below ambiguous about which attempt it is describing.
    ///
    /// **Every write asserts its own success, and the landing is read back
    /// before the hook returns.** A hook is the one place in this file where a
    /// failure has no natural surface: the restore continues, its `EXEC`
    /// commits over an unraced keyspace, and the arm reports a `Materialized`
    /// verdict that looks exactly like a fence that did not hold — with no
    /// signal saying the race was never staged. Panicking here instead is safe
    /// by construction: the unwind passes through `write_restored_object`, and
    /// the `WatchGuard`'s `Drop` still routes the `UNWATCH` back to the pinned
    /// slot (b.jov, b.ji8), so no connection is left watching.
    ///
    /// The read-back is on the SAME data-pool connection the write went out on,
    /// which is what makes it a statement about the keyspace rather than about
    /// fred's reply decoding. It matters most on the [`SeamPosition::BeforeExec`]
    /// arms, where the whole verdict rests on the peer's write having become
    /// visible to the server before this writer's `EXEC` reaches it.
    async fn fire(&self) {
        if self.fired.fetch_add(1, Ordering::AcqRel) != 0 {
            return;
        }
        let target = self
            .target
            .get()
            .expect("the peer lander fired before it was armed with a target");
        let oid_str = target.object_id.to_string();
        let header_key = target.keys.obj_header(&oid_str);

        match self.action {
            PeerAction::LandCompleteRestore => {
                let _: () = target
                    .client
                    .pool
                    .hset(
                        target.keys.obj_section(&oid_str, PEER_SECTION_ID),
                        [("name", "peer-root"), ("parent_id", "0")],
                    )
                    .await
                    .expect("the peer's section record must land inside the window");
                let _: () = target
                    .client
                    .pool
                    .sadd(target.keys.obj_sections_index(&oid_str), PEER_SECTION_ID)
                    .await
                    .expect("the peer's section index member must land inside the window");
                let _: () = target
                    .client
                    .pool
                    .hset(&header_key, peer_header_fields())
                    .await
                    .expect("the peer's complete obj:hdr must land inside the window");

                let landed: u64 = target
                    .client
                    .pool
                    .hlen(&header_key)
                    .await
                    .expect("reading the peer's header back must succeed");
                assert_eq!(
                    landed as usize,
                    peer_header_fields().len(),
                    "the peer's header must be OBSERVABLE before the hook returns. A short header \
                     here means the restore is about to be raced by a peer that does not exist \
                     yet, and its EXEC would commit over an unraced keyspace — a merge the arm \
                     would report as a fence that failed, with nothing saying why"
                );
                let members: Vec<String> = target
                    .client
                    .pool
                    .smembers(target.keys.obj_sections_index(&oid_str))
                    .await
                    .expect("reading the peer's section index back must succeed");
                assert_eq!(
                    members,
                    vec![PEER_SECTION_ID.to_string()],
                    "the peer's section index must hold exactly its own member before the hook \
                     returns, or the anti-union assertion downstream is about a generation that \
                     was never written"
                );
            }
            PeerAction::LandSupersededFlagOnly => {
                let _: () = target
                    .client
                    .pool
                    .hset(&header_key, [("superseded", "1")])
                    .await
                    .expect("the single-field header write must land inside the window");

                let landed: Option<String> = target
                    .client
                    .pool
                    .hget(&header_key, "superseded")
                    .await
                    .expect("reading the injected flag back must succeed");
                assert_eq!(
                    landed.as_deref(),
                    Some("1"),
                    "the `superseded=1` flag must be OBSERVABLE before the hook returns; without \
                     it the restore has nothing to be refused by"
                );
            }
            PeerAction::RetypeHeaderToString => {
                let _: () = target
                    .client
                    .pool
                    .set(&header_key, "not-a-hash", None, None, false)
                    .await
                    .expect("the wrong-typed header write must land inside the window");

                // A successful `GET` is itself the type assertion: `GET` against
                // a hash is `WRONGTYPE`, so an `Ok` carrying the injected value
                // proves the key is the string the fence's `HGETALL` will trip
                // over.
                let landed: Option<String> = target
                    .client
                    .pool
                    .get(&header_key)
                    .await
                    .expect("GET must succeed against the string the injector wrote");
                assert_eq!(
                    landed.as_deref(),
                    Some("not-a-hash"),
                    "obj:hdr must actually be a string before the hook returns, or the WRONGTYPE \
                     the arm is about never happens"
                );
            }
        }
    }
}

/// The twelve canonical `obj:hdr` fields — deliberately only those — standing in
/// for the header a peer replica's completed restore lands.
///
/// All twelve rather than the two `header_fields_are_structurally_complete`
/// reads: seeding only the two the predicate happens to read would pass whatever
/// that predicate's field list became, which is part of what is under test. No
/// more than the twelve, because the arms need the header only to look
/// structurally complete. A real peer's completed restore writes a superset: the
/// children-index marker `chs_complete`, the fence identity `commit_chain` and
/// the three checkpoint-cursor columns (`last_checkpoint_seq`,
/// `last_checkpoint_ms`, `restore_seq_bump`) as well.
///
/// So this fixture header is UNMARKED. Once the atomic snapshot read trusts
/// `obj:chs` only on marked objects (b.y3m Epic 2, Task 4), a read of an object
/// carrying this header takes the unmarked fallback path, where a real peer's
/// restored object would be marked.
fn peer_header_fields() -> Vec<(&'static str, String)> {
    vec![
        ("seq", "500".to_owned()),
        ("first_ver", "0".to_owned()),
        ("src_fmt", "usd".to_owned()),
        ("next_set_order", "0".to_owned()),
        ("session_id", "peer-session".to_owned()),
        ("layer_name", "peer-layer".to_owned()),
        ("created_by", PEER_CREATED_BY.to_owned()),
        ("created_at", "1700000000000".to_owned()),
        ("modified_at", "1700000000000".to_owned()),
        ("last_client_id", String::new()),
        ("last_client_ts", "0".to_owned()),
        ("last_server_ts", "1700000000000".to_owned()),
    ]
}

/// Wrap a [`PeerLander`] as the hook the harness installs.
fn peer_window_hook(lander: Arc<PeerLander>) -> WatchWindowHook {
    Arc::new(move |_guarded: &Client| {
        let lander = Arc::clone(&lander);
        // The guarded client is deliberately unused. This file is not about a
        // connection dying under the `WATCH` — that is b.bhb's subject, and the
        // L2 / L3 layers it added still cover it — but about a SECOND replica
        // committing on a connection of its own while this one is fenced.
        async move { lander.fire().await }.boxed()
    })
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// Everything one arm needs to drive and assert.
struct FenceFixture {
    session_id: String,
    layer_name: String,
    object_id: u64,
    /// `created_by` as it stood on the live `obj:hdr` before the wipe — i.e. the
    /// value the checkpoint blob carries and therefore the one THIS replica's
    /// restore would write back.
    ///
    /// Captured rather than assumed. The layer's `created_by` is minted by the
    /// harness's userinfo client, not by the session's own creator, so hardcoding
    /// it here would silently drift the moment that stub's display name changed —
    /// and the assertion that matters ("the answer is the PEER's, not ours")
    /// would degrade into "the answer is the peer's, and the alternative we
    /// checked against was never reachable".
    loser_created_by: String,
}

/// The `obj:*` keys one call to [`plant_header_less_residue`] created.
struct PlantedResidue {
    index: String,
    record: String,
}

impl PlantedResidue {
    fn keys(&self) -> Vec<String> {
        vec![self.index.clone(), self.record.clone()]
    }
}

/// CreateLayer payload: one section plus one inline field, so the checkpointed
/// generation carries real state.
///
/// Non-empty is load-bearing rather than cosmetic. `obj:secs` is where a merge
/// becomes visible, and a layer with no sections would make the sharpest
/// assertion in this file — "the index holds ONLY the peer's member" —
/// trivially true whether the fence held or not.
fn populated_initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: LOSER_SECTION_ID,
            section_name: "restore-write-fence-root".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: LOSER_SECTION_ID,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAA, 0xBB, 0xCC],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    }
}

/// Build the harness with `lander`'s hook installed at `seam`, create a
/// populated layer, wait for the loop's first generation, **fence the loop**,
/// then wipe the object's Valkey state and arm the lander.
///
/// `seam` picks WHICH of the two b.h4m hooks the lander rides, and that choice
/// is the difference between exercising the fence's pre-`MULTI` header read and
/// exercising the `WATCH` itself (see [`SeamPosition`]). Exactly one is
/// installed per arm: arming both would make the pre-`MULTI` read catch the peer
/// first on every arm, and the before-`EXEC` hook would then fire into a window
/// no restore ever reaches.
///
/// Three parts of this are load-bearing rather than boilerplate:
///
/// 1. **The hook rides the harness OPTIONS.** The `LayerStore` is built inside
///    `setup_checkpoint_harness_with_options` and handed by `Arc` to the
///    service, the recovery hook, the counter-repair hook and the
///    session-checkpoint writer. A store rebuilt afterwards would be a SECOND
///    store, the hook would ride an object no restore runs through, and every
///    assertion downstream would read a fence that never fired.
/// 2. **The loop is cancelled once a generation exists.** With
///    `interval = quiesce = 0` a live loop can land a fresh generation at any
///    moment — including over the peer's header after the fence has fired —
///    which would make the end-state assertions race a background writer.
/// 3. **The [`wire_client_metrics`] call in [`build_fence_fixture`] is not
///    optional.**
///    `setup_checkpoint_harness_with_options` wires `SharedMetrics` into the
///    service, the loop and the hooks, but does NOT install them on the
///    `ValkeyClient`. The fenced counter is ticked through
///    `ValkeyClient::metrics_snapshot()`, which is `None` until they are.
///    Without that call every
///    `live_edit_checkpoint_restore_fenced_total` assertion would read a
///    permanent zero and pass whatever the restore did. That the handle is LIVE
///    is proven inside the arms rather than assumed: the two fenced arms assert
///    a `+1` on it, and the residue arm additionally asserts a `+1` on the
///    wipe's own counter.
async fn build_fence_arm(
    name: &str,
    action: PeerAction,
    seam: SeamPosition,
) -> (CheckpointHarness, FenceFixture, Arc<PeerLander>) {
    let lander = Arc::new(PeerLander::new(action));

    let mut opts = CheckpointHarnessOptions::new(1);
    match seam {
        SeamPosition::BeforeHeaderRead => {
            opts.restore_write_watch_window_hook = Some(peer_window_hook(Arc::clone(&lander)));
        }
        SeamPosition::BeforeExec => {
            opts.restore_write_before_exec_hook = Some(peer_window_hook(Arc::clone(&lander)));
        }
    }
    let (harness, fixture) = Box::pin(build_fence_fixture(name, opts)).await;

    lander.arm(
        harness.tv.client().clone(),
        harness.tv.key_schema().clone(),
        fixture.object_id,
    );

    (harness, fixture, lander)
}

/// [`build_fence_arm`]'s body for a caller that installs its own hook in
/// `opts`: build the harness, create the populated layer, wait for the first
/// generation, fence the loop, wipe the object's Valkey state and install the
/// client metrics. Every load-bearing part [`build_fence_arm`]'s rustdoc names
/// lives here. Arming whatever the hook reads is left to the caller, because
/// the object id exists only once this returns.
async fn build_fence_fixture(
    name: &str,
    opts: CheckpointHarnessOptions,
) -> (CheckpointHarness, FenceFixture) {
    let mut harness = setup_checkpoint_harness_with_options(opts).await;

    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(&format!("{name}-session"), "/stage", CREATED_BY)
        .await
        .expect("session creation must succeed");
    let layer_name = format!("{name}-layer");

    let ompf = delta_to_ompf_bytes(&populated_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, &layer_name),
        make_create_data(ompf),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed while building the fence fixture")
        .into_inner()
        .object_id;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );
    assert_eq!(
        section_index_members(&harness, object_id).await,
        vec![LOSER_SECTION_ID.to_string()],
        "fixture precondition: the created layer must carry EXACTLY the loser's section. An empty \
         index would make the anti-union assertion in every arm below trivially true"
    );

    let _handle = wait_for_checkpoint(&harness.store, object_id, CHECKPOINT_DEADLINE_SECS).await;

    let loser_created_by = read_obj_header(&harness, object_id)
        .await
        .get("created_by")
        .cloned()
        .expect("fixture precondition: the live header must carry `created_by` before the wipe");
    assert_ne!(
        loser_created_by, PEER_CREATED_BY,
        "fixture precondition: this replica's generation and the peer's must be DISTINGUISHABLE, \
         or the arms below cannot tell a fence from a merge"
    );

    // ── Fence the loop ───────────────────────────────────────────────────
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, loop_task).await;

    wipe_layer_state(&harness.tv, object_id).await;
    assert!(
        read_obj_header(&harness, object_id).await.is_empty(),
        "wipe precondition: obj:hdr must be GONE, or apply_restore's step-1 presence probe \
         short-circuits and the innermost fence is never reached"
    );

    wire_client_metrics(&harness).await;

    (
        harness,
        FenceFixture {
            session_id,
            layer_name,
            object_id,
            loser_created_by,
        },
    )
}

/// Plant the b.kf1 orphan shape the step-3b wipe exists for: one section index
/// member plus its per-record hash, one field index member plus its per-record
/// hash, and an `obj:ext` pre-image — all under an object with **no header**.
///
/// Header-less is load-bearing in both directions. It is what a real b.f4t
/// residue looks like (the header died, the records did not), and it is what
/// keeps `apply_restore`'s step-1 structural-presence gate from short-circuiting
/// the attempt before the wipe — and therefore before the fence — is ever
/// reached.
///
/// `obj:ext` is planted deliberately: the wipe WITHHOLDS it from its `UNLINK`
/// inventory precisely so the materializer that follows can read it as
/// `ext_before`, so it is the one residue key expected to survive into the
/// fence's own pre-work. That makes it the sharpest place to observe that a
/// FENCED attempt owed — and enqueued — nothing.
async fn plant_header_less_residue(harness: &CheckpointHarness, object_id: u64) -> PlantedResidue {
    let oid_str = object_id.to_string();
    let keys = harness.tv.key_schema();
    let planted = PlantedResidue {
        index: keys.obj_sections_index(&oid_str),
        record: keys.obj_section(&oid_str, RESIDUE_SECTION_ID),
    };

    let _: () = harness
        .tv
        .client()
        .pool
        .sadd(&planted.index, RESIDUE_SECTION_ID)
        .await
        .expect("SADD residue index member");
    let _: () = harness
        .tv
        .client()
        .pool
        .hset(&planted.record, [("name", "stale"), ("parent_id", "0")])
        .await
        .expect("HSET residue record");
    let _: () = harness
        .tv
        .client()
        .pool
        .hset(
            keys.obj_ext_index(&oid_str),
            [(residue_ext_hash_hex(), "1")],
        )
        .await
        .expect("HSET residue obj:ext pre-image");

    planted
}

// ═════════════════════════════════════════════════════════════════════════════
// Keyspace readers
// ═════════════════════════════════════════════════════════════════════════════

/// Read the whole live `obj:hdr` hash.
async fn read_obj_header(harness: &CheckpointHarness, object_id: u64) -> HashMap<String, String> {
    harness
        .tv
        .client()
        .pool
        .hgetall(harness.tv.key_schema().obj_header(&object_id.to_string()))
        .await
        .expect("HGETALL obj:hdr must succeed")
}

/// `SMEMBERS le:{oid}:obj:secs`, sorted.
async fn section_index_members(harness: &CheckpointHarness, object_id: u64) -> Vec<String> {
    let mut members: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers(
            harness
                .tv
                .key_schema()
                .obj_sections_index(&object_id.to_string()),
        )
        .await
        .expect("SMEMBERS obj:secs must succeed");
    members.sort();
    members
}

/// `HGETALL le:{oid}:obj:ext`, sorted into `(hash_hex, count)` pairs.
async fn ext_index_rows(harness: &CheckpointHarness, object_id: u64) -> Vec<(String, String)> {
    let raw: HashMap<String, String> = harness
        .tv
        .client()
        .pool
        .hgetall(
            harness
                .tv
                .key_schema()
                .obj_ext_index(&object_id.to_string()),
        )
        .await
        .expect("HGETALL obj:ext must succeed");
    let mut rows: Vec<(String, String)> = raw.into_iter().collect();
    rows.sort();
    rows
}

/// `SMEMBERS le:ext:{gc}:pending`, sorted — the owed-decrement queue.
async fn ext_pending_members(harness: &CheckpointHarness) -> Vec<String> {
    let mut members: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers(harness.tv.key_schema().ext_pending())
        .await
        .expect("SMEMBERS ext pending must succeed");
    members.sort();
    members
}

// ═════════════════════════════════════════════════════════════════════════════
// Metric readers
// ═════════════════════════════════════════════════════════════════════════════
//
// The restore-fenced family's readers — the per-cell `restore_fenced_metric`,
// the whole-vocabulary `restore_fenced_cells` snapshot and the
// `assert_restore_fenced_deltas` comparison — live in `helpers.rs`, shared with
// `restore_fault_injection_matrix_test.rs` (which owns the third stage,
// `residue_wipe`, that nothing in this file can reach) and
// `e2e_object_id_preservation_test.rs`. One reader per cell is what keeps their
// absent-series discipline from drifting apart.

/// Live value of one
/// `live_edit_storage_watch_voided_total{path="restore_write_object", disposition}`
/// cell.
fn restore_write_voided_metric(harness: &CheckpointHarness, disposition: &str) -> u64 {
    sample_value(
        &render(&harness.metrics),
        "live_edit_storage_watch_voided_total",
        &[
            ("path", "restore_write_object"),
            ("disposition", disposition),
        ],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_storage_watch_voided_total{{path=\"restore_write_object\", \
             disposition=\"{disposition}\"}} must be pre-initialised — b.h4m appends \
             `restore_write_object` to WATCH_VOIDED_PATHS and the cross product is zero-seeded"
        )
    }) as u64
}

/// The two dispositions the restore-write fence's own code can tick on a store
/// that answers `CLIENT ID` — i.e. the two whose movement is a finding rather
/// than an environment report.
///
/// `witness_unavailable` / `fingerprint_unavailable` are deliberately excluded:
/// both describe a backing store that would not give the fence a stable
/// in-`MULTI` `CLIENT ID` or a connection id, which is a property of the Valkey
/// under the test rather than of the arm, and pinning them would fail these arms
/// on a proxied managed store for a reason they do not test. `compensated` has
/// no producer on this path at all — a landed materialization cannot be
/// un-merged.
const RESTORE_WRITE_DISPOSITIONS_UNDER_TEST: [&str; 2] =
    ["aborted_pre_exec", "committed_unwatched"];

fn restore_write_dispositions(harness: &CheckpointHarness) -> Vec<(&'static str, u64)> {
    RESTORE_WRITE_DISPOSITIONS_UNDER_TEST
        .iter()
        .map(|disposition| {
            (
                *disposition,
                restore_write_voided_metric(harness, disposition),
            )
        })
        .collect()
}

/// Assert neither disposition moved.
///
/// `aborted_pre_exec` is the L2 pre-dispatch `Refuse` arm and
/// `committed_unwatched` is the L3 witness contradiction on a COMMITTED `EXEC`.
/// No arm in this file is either: each reaches its verdict by READING the
/// header — before the `MULTI` (arms 1–4) or after a tripped `EXEC` (arms 6–8) —
/// and neither of those ticks anything.
///
/// The tripped-`EXEC` arms are the sharper statement of the two. A trip is
/// Valkey's L1 fence doing exactly its job, so both cells staying flat is what
/// says so: `aborted_pre_exec` would mean the attempt refused *itself* before
/// dispatch (a different incident, on a replaced connection), and
/// `committed_unwatched` would mean the `EXEC` COMMITTED somewhere the `WATCH`
/// never held — i.e. the trip never happened and the merge did.
///
/// `committed_unwatched` in particular is the cell the counter's own rustdoc
/// names as the correctness failure behind
/// `live_edit_checkpoint_restore_fenced_total`: the mixture-of-two-backups
/// hazard actually happening, on a store that replayed the transaction onto a
/// connection that never held the `WATCH`. A tick there would be this file's
/// subject failing, not a bookkeeping surprise.
fn assert_restore_write_dispositions_flat(
    harness: &CheckpointHarness,
    before: &[(&'static str, u64)],
    context: &str,
) {
    for (disposition, pre) in before {
        assert_eq!(
            restore_write_voided_metric(harness, disposition),
            *pre,
            "{context}: storage_watch_voided_total{{path=\"restore_write_object\", \
             disposition=\"{disposition}\"}} MUST NOT move"
        );
    }
}

/// The three layer-scope recovery-attempt cells, for a before/after comparison.
fn recovery_cells(harness: &CheckpointHarness) -> Vec<(&'static str, u64)> {
    ["restored", "not_recoverable", "error"]
        .iter()
        .map(|result| (*result, recovery_metric(&harness.metrics, result)))
        .collect()
}

/// Assert exactly the named recovery results moved by the given deltas.
fn assert_recovery_deltas(
    harness: &CheckpointHarness,
    before: &[(&'static str, u64)],
    expected: &[(&str, u64)],
    context: &str,
) {
    for (result, pre) in before {
        let want = expected
            .iter()
            .find(|(name, _)| name == result)
            .map(|(_, delta)| *delta)
            .unwrap_or(0);
        assert_eq!(
            recovery_metric(&harness.metrics, result) - *pre,
            want,
            "{context}: checkpoint_recovery_attempts_total{{result=\"{result}\", scope=\"layer\"}} \
             must move by {want}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Driving
// ═════════════════════════════════════════════════════════════════════════════

fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Drive `StatLayer` under [`RPC_BUDGET`] and hand back whatever it produced.
async fn drive_stat_layer(
    harness: &CheckpointHarness,
    fixture: &FenceFixture,
) -> Result<StatLayerResponse, tonic::Status> {
    tokio::time::timeout(
        RPC_BUDGET,
        handle_stat_layer(
            &harness.svc,
            stat_request(&fixture.session_id, &fixture.layer_name),
        ),
    )
    .await
    .expect(
        "StatLayer did not settle within the RPC budget — the restore is wedged, most likely \
         inside the fence's WATCH window (the hook awaits inline on the restore's own task)",
    )
    .map(tonic::Response::into_inner)
}

/// Drop the harness at the end of an arm, re-cancelling its token first.
///
/// **This is NOT the loop drain, and does not pretend to be one.**
/// [`build_fence_fixture`] cancels the token and joins the real `loop_handle` under
/// [`LOOP_DRAIN_BUDGET`] long before any arm body runs — it has to, because the
/// wipe and the race that follow must not race a live generation writer — and it
/// leaves a `tokio::spawn(async {})` no-op in the field it took the handle from.
/// So the join below is over that no-op and completes immediately.
///
/// It is still worth a named call rather than an implicit end-of-scope drop.
/// Cancelling the token is idempotent and keeps the "no writer outlives the
/// assertions" claim true of the harness as a whole rather than of one field,
/// and the bounded join is what would surface a real handle if a future refactor
/// ever put one back — silently, if this were a bare `drop(harness)`.
async fn release_harness(harness: CheckpointHarness) {
    harness.cancel.cancel();
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, harness.loop_handle).await;
}

/// The assertions both fenced arms share: the RPC served the PEER's generation,
/// and the loser's contributed nothing at all.
///
/// Factored because the two arms differ only in the keyspace they start from —
/// the claim about the OUTCOME is identical, and stating it once is what keeps
/// the residue arm from quietly weakening it.
async fn assert_peer_generation_serves_alone(
    harness: &CheckpointHarness,
    fixture: &FenceFixture,
    stat: &StatLayerResponse,
    context: &str,
) {
    assert_eq!(
        stat.created_by_user_name, PEER_CREATED_BY,
        "{context}: StatLayer MUST serve the PEER's header. `{}` — this replica's own \
         generation — here means its materialization overwrote a restore that had already \
         landed: the merge, seen from the client",
        fixture.loser_created_by
    );

    let members = section_index_members(harness, fixture.object_id).await;
    assert_eq!(
        members,
        vec![PEER_SECTION_ID.to_string()],
        "{context}: obj:secs must hold EXACTLY the peer's member. This is the sharpest statement \
         of the whole defect: the materializer's `SADD` unions rather than replaces, so \
         `{LOSER_SECTION_ID}` appearing beside `{PEER_SECTION_ID}` is the mixture of two backup \
         generations — a layer that is internally plausible, raises no error, and serves a \
         section the peer's checkpoint never contained. Got {members:?}"
    );
    assert_eq!(
        count_existing(
            harness,
            vec![harness.tv.key_schema().obj_section(
                &fixture.object_id.to_string(),
                &LOSER_SECTION_ID.to_string()
            )]
        )
        .await,
        0,
        "{context}: the loser's per-record section hash must not exist either. An index that \
         excludes it while the record survives is the b.9fn orphan shape — unreachable from any \
         index, and enumerable by no retirement path"
    );

    let header = read_obj_header(harness, fixture.object_id).await;
    assert_eq!(
        header.get("created_by").map(String::as_str),
        Some(PEER_CREATED_BY),
        "{context}: the live header must still be the peer's, field for field; got {header:?}"
    );
    assert!(
        !header.contains_key("last_checkpoint_seq"),
        "{context}: `last_checkpoint_seq` is stamped ONLY by the materialization's own \
         MULTI/EXEC, so its presence proves step 5 wrote after the fence said the layer was \
         already present. This is the assertion a fence that merely returned early — without \
         actually withholding the write — cannot fake. Got {header:?}"
    );
}

/// The three before-readings every refusal arm takes, captured in one call.
///
/// Bundled rather than three loose locals for the same anti-drift reason as
/// [`assert_refusal_wrote_nothing`] below: the four refusal arms must pin the
/// same counters, and a snapshot one of them forgot to take is a counter that
/// silently stops being asserted on.
struct RefusalCounters {
    fenced: Vec<(&'static str, u64)>,
    recovery: Vec<(&'static str, u64)>,
    dispositions: Vec<(&'static str, u64)>,
}

impl RefusalCounters {
    fn capture(harness: &CheckpointHarness) -> Self {
        Self {
            fenced: restore_fenced_cells(harness),
            recovery: recovery_cells(harness),
            dispositions: restore_write_dispositions(harness),
        }
    }
}

/// The assertions all FOUR refusal arms share: nothing of the restored state
/// reached the keyspace, no fence was claimed, the attempt counted as an `error`,
/// and neither watch-voided disposition moved.
///
/// Factored for the same reason as [`assert_peer_generation_serves_alone`]: arms
/// 3, 4, 7 and 8 differ in the SHAPE the peer landed (`superseded=1` vs a
/// wrong-typed key) and in WHICH of the fence's two header reads decided the
/// refusal — but the claim about what a refusal must have left behind is one
/// claim, and it was written out in four spellings. Four spellings of one claim
/// is four places for it to be weakened one arm at a time; the arms that would
/// then stop asserting it are exactly the ones whose whole subject is that
/// nothing was written.
///
/// Each arm keeps what genuinely distinguishes it: its gRPC status code (the
/// transient/permanent classification) and its own header end-state read-back —
/// `HGETALL` against the injected `{superseded: 1}` hash, or `GET` against the
/// string the retyping injector left. `context` must name the arm, and is
/// threaded into every message below so a failure still says which of the four
/// produced it.
async fn assert_refusal_wrote_nothing(
    harness: &CheckpointHarness,
    fixture: &FenceFixture,
    before: &RefusalCounters,
    context: &str,
) {
    let members = section_index_members(harness, fixture.object_id).await;
    assert!(
        members.is_empty(),
        "{context}: obj:secs must be EMPTY. Not one member of the restored state may have \
         survived — a refusal decided before the MULTI opens never queues one, and a refusal \
         decided by the post-conflict re-read had its queued SADD discarded by the trip. A member \
         here is the materialization landing over a header it was supposed to refuse. Got \
         {members:?}"
    );

    assert_restore_fenced_deltas(
        harness,
        &before.fenced,
        &[],
        &format!(
            "{context} — a refusal is not a fence: \
             `live_edit_checkpoint_restore_fenced_total` counts attempts that found a peer's \
             COMPLETED rebuild, and neither a lone `superseded` flag nor a wrong-typed key is one"
        ),
    );
    assert_recovery_deltas(
        harness,
        &before.recovery,
        &[("error", 1)],
        &format!(
            "{context} — `restored` ticking here is the metric dishonesty these arms rule out: a \
             restore that never ran, reported as a recovery, while the caller got its own \
             NotFound straight back. `not_recoverable` is the SR-2.4 mapping-guard verdict, which \
             a structural keyspace refusal is not either"
        ),
    );
    assert_restore_write_dispositions_flat(harness, &before.dispositions, context);
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 1 — total loss: the widest window, and the only stage it can reach
// ═════════════════════════════════════════════════════════════════════════════

/// A peer replica's restore landing inside `write_restored_object`'s `WATCH`
/// window, over a keyspace with NO residue, is a fence: this replica writes
/// nothing, `apply_restore` reports `AlreadyPresent`, and `StatLayer` answers
/// `Ok` from the peer's header.
///
/// # Why this arm is the important one
///
/// Total loss is the commonest DR shape and the one the two OUTER fences cannot
/// see. `apply_restore`'s step-1 presence probe is a point-in-time read that
/// happened before the peer's header existed, and the step-3b residue wipe
/// short-circuits `Clean` over an empty keyspace and never arms its `WATCH` at
/// all. So `object_write` is the ONLY stage reachable here, and before b.h4m
/// there was no fence at that position — which is exactly why the pre-fix defect
/// was widest precisely where the keyspace was emptiest.
///
/// # What each assertion adds
///
/// - **The seam fired exactly once.** A zero would make every assertion below
///   vacuous: it would mean the restore never reached the window, so nothing was
///   ever raced.
/// - **`StatLayer` is `Ok` with the peer's `created_by`.** The fence held AND
///   the client still got a working layer. A fence that failed the RPC would be
///   safe and useless.
/// - **`obj:secs` holds only the peer's member**, and the loser's per-record
///   hash does not exist. The anti-union claim, stated over both the index and
///   the record.
/// - **No `last_checkpoint_seq`.** Positive evidence that step 5's `HSET` never
///   ran, which no early return can fake.
/// - **`fenced{object_write}` +1, the other two stages flat.** The counter says
///   how late the peer was noticed, and "innermost" is the answer this arm
///   claims.
/// - **`restored` +1, `not_recoverable` / `error` flat.** `AlreadyPresent` folds
///   into `restored` — the layer IS restored, by the other replica — which keeps
///   `sum by (result) == attempts` true. The fence is emphatically not an error.
/// - **Neither watch-voided disposition moved.** A fence reached by reading the
///   header is neither an L2 pre-dispatch refusal nor an unfenced commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_landing_a_complete_restore_inside_the_write_window_fences_a_total_loss_restore() {
    let (harness, fixture, lander) = Box::pin(build_fence_arm(
        "fence-total-loss",
        PeerAction::LandCompleteRestore,
        SeamPosition::BeforeHeaderRead,
    ))
    .await;

    let fenced_pre = restore_fenced_cells(&harness);
    let recovery_pre = recovery_cells(&harness);
    let dispositions_pre = restore_write_dispositions(&harness);
    let wiped_pre = residue_wiped_metric(&harness);

    let stat = drive_stat_layer(&harness, &fixture)
        .await
        .unwrap_or_else(|status| {
            panic!(
                "a peer-landed restore is a FENCE, not an error: apply reports AlreadyPresent, \
                 the hook reports Restored, and StatLayer's retried header read finds the peer's \
                 header. Got {:?}: {}",
                status.code(),
                status.message()
            )
        });

    assert_eq!(
        lander.fire_count(),
        1,
        "the restore-write window seam MUST have fired exactly once. Zero means the restore never \
         armed its WATCH — it short-circuited earlier, or the hook rode a LayerStore no restore \
         runs through — and every assertion below would be vacuous"
    );

    assert_peer_generation_serves_alone(&harness, &fixture, &stat, "total-loss restore").await;

    assert_restore_fenced_deltas(
        &harness,
        &fenced_pre,
        &[("object_write", 1)],
        "a total-loss restore can reach no other fence: step 1's probe already ran and the wipe \
         short-circuits Clean over an empty keyspace",
    );
    assert_recovery_deltas(
        &harness,
        &recovery_pre,
        &[("restored", 1)],
        "AlreadyPresent folds into `restored` — the layer IS there — so the fence must not show \
         up as an error, and must not double-count",
    );
    assert_eq!(
        residue_wiped_metric(&harness),
        wiped_pre,
        "there was no residue, so the wipe's counter must not move: a tick here would mean the \
         wipe took its destructive path over a keyspace the fixture emptied"
    );
    assert_restore_write_dispositions_flat(
        &harness,
        &dispositions_pre,
        "a fence reached by reading the header",
    );

    release_harness(harness).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 2 — residue present: the fence still holds after the wipe committed
// ═════════════════════════════════════════════════════════════════════════════

/// The same race, but this replica's step-3b residue wipe ran and COMMITTED
/// first. The fence must still hold, and the fenced attempt must owe nothing.
///
/// # Why this is a separate arm
///
/// The wipe is a destructive `UNLINK` that has already landed by the time the
/// window opens, so this is the state in which a merge would be least
/// recoverable: the loser has removed the old generation's records and is about
/// to write its own over a keyspace the peer has meanwhile rebuilt. The
/// total-loss arm cannot reach that ordering, because with nothing to wipe the
/// step never takes its `WATCH` path.
///
/// It also puts the two additive counters on the same timeline: the wipe's
/// `+1` and the fence's `+1` describe two different events inside ONE restore
/// attempt, and their coexistence is what the operator reads as "this replica
/// did real work and was still fenced".
///
/// # The external-value half
///
/// `obj:ext` is the one residue key the wipe deliberately WITHHOLDS, so the
/// fence's own pre-work reads it as `ext_before` and plans `rewrite` releases
/// from it — here, a release of the planted hash, since the restored state does
/// not hold it. Those releases are enqueued on the COMMITTED arm only. So a
/// fenced attempt must leave both the index and the pending queue untouched:
/// an enqueue here would be an unmatched decrement against a blob nothing gave
/// up, and with GC behind it, `ext:meta` + `ext:data` deleted under a live
/// reference — data loss rather than a leak.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_landing_a_complete_restore_inside_the_write_window_fences_after_the_residue_wipe() {
    let (harness, fixture, lander) = Box::pin(build_fence_arm(
        "fence-over-residue",
        PeerAction::LandCompleteRestore,
        SeamPosition::BeforeHeaderRead,
    ))
    .await;

    let residue = plant_header_less_residue(&harness, fixture.object_id).await;
    assert_eq!(
        count_existing(&harness, residue.keys()).await,
        2,
        "fixture precondition: both planted residue keys must exist, or the wipe finds nothing \
         and this arm degenerates into the total-loss one"
    );
    let planted_ext = ext_index_rows(&harness, fixture.object_id).await;
    assert_eq!(
        planted_ext,
        vec![(residue_ext_hash_hex(), "1".to_owned())],
        "fixture precondition: the obj:ext pre-image must be the planted row alone"
    );

    let fenced_pre = restore_fenced_cells(&harness);
    let recovery_pre = recovery_cells(&harness);
    let dispositions_pre = restore_write_dispositions(&harness);
    let wiped_pre = residue_wiped_metric(&harness);

    let stat = drive_stat_layer(&harness, &fixture)
        .await
        .unwrap_or_else(|status| {
            panic!(
                "a peer-landed restore is a FENCE even after this replica's wipe committed. Got \
                 {:?}: {}",
                status.code(),
                status.message()
            )
        });

    assert_eq!(
        lander.fire_count(),
        1,
        "the restore-write window seam MUST have fired exactly once"
    );

    assert_peer_generation_serves_alone(&harness, &fixture, &stat, "restore over residue").await;
    assert_eq!(
        count_existing(&harness, vec![residue.record.clone()]).await,
        0,
        "the planted per-record hash MUST be gone: this arm's whole premise is that the wipe ran \
         and committed BEFORE the window opened. A survivor means the wipe was fenced instead, \
         and the `residue_wipe` stage assertion below would be the one describing this run"
    );

    assert_restore_fenced_deltas(
        &harness,
        &fenced_pre,
        &[("object_write", 1)],
        "the wipe committed before the peer landed, so the middle fence was NOT reached — the \
         innermost one was",
    );
    assert_eq!(
        residue_wiped_metric(&harness) - wiped_pre,
        1,
        "the wipe committed exactly once. This is also what proves the flat readings elsewhere in \
         this arm were taken through a LIVE metrics handle rather than an uninstalled one"
    );
    assert_recovery_deltas(
        &harness,
        &recovery_pre,
        &[("restored", 1)],
        "a fenced attempt that did real work is still `restored`, not an error",
    );
    assert_restore_write_dispositions_flat(
        &harness,
        &dispositions_pre,
        "a fence reached by reading the header, after a committed wipe",
    );

    // ── The external-value ledger ────────────────────────────────────────
    assert_eq!(
        ext_index_rows(&harness, fixture.object_id).await,
        planted_ext,
        "obj:ext must be untouched. The fenced attempt queued its absolute HSET/HDEL rebuild \
         inside the transaction that never EXECed, so a changed index means the materialization \
         landed after all"
    );
    assert!(
        ext_pending_members(&harness).await.is_empty(),
        "no release may be enqueued. The plan DID compute one — the pre-image holds a hash the \
         restored state does not — but `enqueue_ext_releases` runs on the COMMITTED arm alone. An \
         entry here is an unmatched decrement against a blob nothing gave up, and with the GC \
         behind it, ext:meta + ext:data deleted under a live reference"
    );

    release_harness(harness).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 3 — a dirty header is REFUSED, not written over
// ═════════════════════════════════════════════════════════════════════════════

/// A single-field `HSET superseded=1` inside the window is NOT evidence a peer
/// landed anything: the restore is refused `Unavailable` (⇒ `UNAVAILABLE`),
/// writes nothing, and counts as an `error`.
///
/// # The defect this fences
///
/// Two opposite mistakes are both wrong here, and the arm rules out both.
///
/// Treating a present-but-incomplete header as "a peer landed it" would report
/// `AlreadyPresent`, tick
/// `checkpoint_recovery_attempts_total{result="restored"}` for a restore that
/// never ran, and hand the caller its own `NotFound` straight back while the
/// metric claimed a recovery.
///
/// Treating it as "not a real header, write over it" is worse. The concrete
/// producer of this shape is a racing `CreateLayer` overwrite, which sets
/// exactly that one flag — and burying `superseded=1` under a freshly restored
/// complete header leaves the layer PERMANENTLY unwritable, because both
/// commit backends fail closed on the flag. Refusing is what lets the next
/// attempt's residue wipe remove the partial header (it is element 0 of the
/// shared cleanup inventory) so the retry converges.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_partial_header_landing_inside_the_write_window_is_refused_and_writes_nothing() {
    let (harness, fixture, lander) = Box::pin(build_fence_arm(
        "fence-dirty-header",
        PeerAction::LandSupersededFlagOnly,
        SeamPosition::BeforeHeaderRead,
    ))
    .await;

    let before = RefusalCounters::capture(&harness);

    let status = drive_stat_layer(&harness, &fixture)
        .await
        .expect_err("a refused restore MUST surface an error, never Ok");
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "the handler maps a Transient RecoveryOutcome::Error onto UNAVAILABLE (retriable) and \
         every other CheckpointError onto INTERNAL, so the code IS the classification. A partial \
         header is transient by construction — the next attempt wipes it and converges. Got \
         {:?}: {}",
        status.code(),
        status.message()
    );

    assert_eq!(
        lander.fire_count(),
        1,
        "the seam MUST have fired exactly once, or the UNAVAILABLE above came from somewhere else"
    );

    assert_eq!(
        read_obj_header(&harness, fixture.object_id).await,
        HashMap::from([("superseded".to_owned(), "1".to_owned())]),
        "obj:hdr must carry EXACTLY the injected single field. MORE fields mean the \
         materialization ran over a header it was supposed to refuse — burying the flag and \
         leaving the layer permanently unwritable; FEWER mean the injected write never landed \
         inside the window and this arm observed nothing"
    );

    assert_refusal_wrote_nothing(
        &harness,
        &fixture,
        &before,
        "arm 3 — a lone `superseded` flag refused by the pre-MULTI header read",
    )
    .await;

    release_harness(harness).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 4 — a wrong-typed header is PERMANENT
// ═════════════════════════════════════════════════════════════════════════════

/// `SET obj:hdr "…"` inside the window makes the fence's `HGETALL` fail
/// `WRONGTYPE`, which must surface as `INTERNAL` — permanent — and write
/// nothing.
///
/// # Why permanence is the assertion
///
/// No retry can `HGETALL` a key that is not a hash. A retriable verdict would
/// re-drive a restore that cannot succeed, forever, ticking
/// `checkpoint_recovery_attempts_total{result="error"}` on every pass while an
/// operator sees a layer that never comes back and a counter that never explains
/// why. The remediation is human — remove the key — so the code has to say so.
///
/// # Why this shape is only reachable from inside the window
///
/// `write_restored_object` runs a pre-`MULTI` keyspace type probe over the same
/// key set BEFORE it acquires the transaction slot, so a header that was already
/// wrong-typed is refused there, by the b.zvd path with its own message and its
/// own metric cell (`b_zvd_tests.rs` owns that case). Retyping the key from
/// inside the window is what routes the condition through the fence's own
/// `classify_restore_header_read_error` instead — the classifier b.h4m added so
/// that BOTH of the fence's header reads reach the same verdict. The unit tests
/// beside it in `src/storage/layer.rs` pin the predicate; this arm pins the
/// consequence an operator actually sees.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrong_typed_header_landing_inside_the_write_window_is_permanent() {
    let (harness, fixture, lander) = Box::pin(build_fence_arm(
        "fence-wrongtype-header",
        PeerAction::RetypeHeaderToString,
        SeamPosition::BeforeHeaderRead,
    ))
    .await;

    let before = RefusalCounters::capture(&harness);

    let status = drive_stat_layer(&harness, &fixture)
        .await
        .expect_err("a wrong-typed obj:hdr MUST surface an error, never Ok");
    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a WRONGTYPE header is PERMANENT, and the handler spells permanence as INTERNAL. \
         UNAVAILABLE here would invite the client to retry a condition only an operator can \
         clear. Got {:?}: {}",
        status.code(),
        status.message()
    );

    assert_eq!(
        lander.fire_count(),
        1,
        "the seam MUST have fired exactly once, or the INTERNAL above came from somewhere else"
    );

    // `GET`, not `HGETALL`: reading it back as the STRING it still is proves two
    // things at once — the key was not repaired, and it was not overwritten by
    // the header hash a materialization would have written (an `HSET`
    // against a string key errors, so a merge here could not even have been
    // silent).
    let header_value: Option<String> = harness
        .tv
        .client()
        .pool
        .get(
            harness
                .tv
                .key_schema()
                .obj_header(&fixture.object_id.to_string()),
        )
        .await
        .expect("GET obj:hdr must succeed against the string the injector left");
    assert_eq!(
        header_value.as_deref(),
        Some("not-a-hash"),
        "the wrong-typed key must survive untouched — the refusal writes nothing, including no \
         repair. Removing it is the operator's call, and a service that silently retyped a key it \
         did not create would destroy the evidence"
    );

    assert_refusal_wrote_nothing(
        &harness,
        &fixture,
        &before,
        "arm 4 — a wrong-typed header refused by the pre-MULTI header read",
    )
    .await;

    release_harness(harness).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 5 — the OUTERMOST stage, so "the other two stayed flat" means something
// ═════════════════════════════════════════════════════════════════════════════

/// A peer's complete header already in place when `apply_restore` starts ticks
/// `stage="presence_probe"` — the outermost of the three fences.
///
/// # Why this arm exists at all
///
/// Every arm above asserts that `presence_probe` and `residue_wipe` stayed FLAT.
/// That assertion is only worth making if those cells CAN move: a stage that no
/// test ever ticks is indistinguishable from a stage whose increment was never
/// wired, and "it stayed at zero" would then be true for the wrong reason.
/// This arm is the non-vacuity proof for the outermost cell, exactly as the
/// residue arm's `residue_wiped_metric` `+1` is for that counter's handle.
///
/// # Why it drives `apply_checkpoint` rather than the RPC
///
/// `RecoveryHook` has two short-circuits of its own — the post-admission
/// `is_layer_present` on entry and the post-adjudication re-check — that return
/// `Restored` WITHOUT ever reaching `apply_restore`, so a complete header
/// planted before a `StatLayer` call would be caught by one of those and step 1
/// would never run. That is the counter's own documented caveat: `presence_probe`
/// is a *partial* count of pre-work discoveries, not a complete one. Driving the
/// apply directly is what puts step 1 on the path.
///
/// Deliberately invalid OMPF bytes (`Bytes::new()`) are the second half of the
/// assertion: if anything past step 1 ran, the decoder would surface
/// `Permanent`. `Ok(())` proves the short-circuit fired — the same technique
/// `apply.rs`'s own unit test uses, with the metrics handle this file needs
/// wired on top.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_header_present_before_the_apply_starts_fences_at_the_presence_probe() {
    let (harness, fixture, lander) = Box::pin(build_fence_arm(
        "fence-presence-probe",
        PeerAction::LandCompleteRestore,
        SeamPosition::BeforeHeaderRead,
    ))
    .await;

    // The peer landed BEFORE this attempt did any work, so its header is simply
    // there when step 1 reads it. The window hook never fires on this arm.
    let _: () = harness
        .tv
        .client()
        .pool
        .hset(
            harness
                .tv
                .key_schema()
                .obj_header(&fixture.object_id.to_string()),
            peer_header_fields(),
        )
        .await
        .expect("planting the peer's complete header must succeed");

    let fenced_pre = restore_fenced_cells(&harness);
    let dispositions_pre = restore_write_dispositions(&harness);

    let result = live_edit::checkpoint::apply_checkpoint(
        &harness.svc.layer_store,
        fixture.object_id,
        bytes::Bytes::new(),
        0,
        &live_edit::checkpoint::CheckpointMeta::default(),
        None,
        &harness.config,
    )
    .await;
    assert!(
        result.is_ok(),
        "the structurally-complete-header arm must short-circuit Ok(()) without decoding; got \
         {:?}",
        result.err()
    );

    assert_eq!(
        lander.fire_count(),
        0,
        "the materialization's own window must never have opened: step 1 returned before the \
         residue wipe, the ext pre-store, or write_restored_object was reached"
    );
    assert_restore_fenced_deltas(
        &harness,
        &fenced_pre,
        &[("presence_probe", 1)],
        "the peer was already complete before any work, which is the OUTERMOST fence and the \
         cheapest — one HGETALL",
    );
    assert_restore_write_dispositions_flat(
        &harness,
        &dispositions_pre,
        "the materialization never armed a WATCH",
    );
    assert_eq!(
        read_obj_header(&harness, fixture.object_id)
            .await
            .get("created_by")
            .map(String::as_str),
        Some(PEER_CREATED_BY),
        "the short-circuit must not have clobbered the peer's header"
    );

    release_harness(harness).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 6 — the read→EXEC window: the only arm where the `WATCH` is the mechanism
// ═════════════════════════════════════════════════════════════════════════════

/// A peer replica's restore landing AFTER the fence's header read passed empty
/// and BEFORE its `EXEC` trips this writer's `WATCH`: the `EXEC` returns nil,
/// nothing of the loser's lands, the post-conflict re-read finds the peer's
/// complete header, and the attempt reports `AlreadyPresent`.
///
/// # Why this arm is the one that proves the fix
///
/// Arms 1–4 install the seam BEFORE the pre-`MULTI` header read, so every one of
/// them exits on that read — a point-in-time `HGETALL` that happens to be issued
/// on the watched connection. Delete the `WATCH` entirely and all four still
/// pass. They are worth having (the read is a real fence layer, and its two
/// refusal shapes have to be pinned) but none of them exercises the sentence the
/// module header opens with: *Valkey serializes the two `EXEC`s on the
/// `{object_id}` shard, so the second writes nothing.*
///
/// This arm does, and it is the only thing that can. The peer's header lands in
/// the read→`EXEC` window — after the read has already answered "absent" and
/// after every mutation is queued — so no read this replica performs can catch
/// it and no amount of re-checking would. The `WATCH` is the entire defence, and
/// what makes the window enterable at all is
/// `with_restore_write_before_exec_hook_for_test`: a peer write *spawned* from
/// the sibling seam is a race that in practice lands before the read, which is
/// why this arm needed a second seam rather than a second `tokio::spawn`.
///
/// It is also the shape a real cross-replica race most often takes. The read is
/// one round trip; the queueing of a whole object's state is many. The
/// read→`EXEC` window is the WIDE part of the fenced region, so a peer landing
/// there is the likely case, not the exotic one.
///
/// # What each assertion adds beyond arm 1
///
/// The outcome assertions are deliberately arm 1's, via the shared
/// [`assert_peer_generation_serves_alone`] — the CLAIM is identical and stating
/// it twice in two spellings is how the two quietly drift. What differs is the
/// path, and these three pin it:
///
/// - **The seam fired exactly once**, on the before-`EXEC` hook. Zero means the
///   restore never reached the dispatch — it exited at the pre-`MULTI` read or
///   earlier — and the `Conflict` arm was not what produced the verdict below.
/// - **`fenced{object_write}` +1.** The post-conflict re-read reports the same
///   `RestoreWriteOutcome::PeerLandedHeader` as the pre-`MULTI` arm, so the
///   operator's counter reads identically whichever of the two fired. That
///   sameness is the design: the stage says WHERE the peer was noticed, and both
///   are the materialization.
/// - **Both watch-voided dispositions flat.** A tripped `EXEC` is Valkey's L1
///   fence working; it is neither an L2 pre-dispatch refusal
///   (`aborted_pre_exec`) nor a commit that ran unwatched
///   (`committed_unwatched` — which here would mean the trip never happened and
///   the mixture landed).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_landing_a_complete_restore_between_the_read_and_the_exec_trips_the_watch() {
    let (harness, fixture, lander) = Box::pin(build_fence_arm(
        "fence-exec-trip",
        PeerAction::LandCompleteRestore,
        SeamPosition::BeforeExec,
    ))
    .await;

    let fenced_pre = restore_fenced_cells(&harness);
    let recovery_pre = recovery_cells(&harness);
    let dispositions_pre = restore_write_dispositions(&harness);
    let wiped_pre = residue_wiped_metric(&harness);

    let stat = drive_stat_layer(&harness, &fixture)
        .await
        .unwrap_or_else(|status| {
            panic!(
                "a tripped EXEC whose post-conflict re-read finds a COMPLETE header is a FENCE, \
                 not an error: the outcome is PeerLandedHeader, apply reports AlreadyPresent, and \
                 StatLayer serves the peer's header. Got {:?}: {}",
                status.code(),
                status.message()
            )
        });

    assert_eq!(
        lander.fire_count(),
        1,
        "the read→EXEC seam MUST have fired exactly once. Zero means the attempt never reached \
         dispatch — it exited at the pre-MULTI read, the L2 check, or a queueing failure — so the \
         WatchedExec::Conflict arm is not what produced the verdict above, and this arm covers \
         nothing the four before it did not"
    );

    assert_peer_generation_serves_alone(&harness, &fixture, &stat, "EXEC-trip restore").await;

    assert_restore_fenced_deltas(
        &harness,
        &fenced_pre,
        &[("object_write", 1)],
        "the post-conflict re-read reports the same PeerLandedHeader outcome as the pre-MULTI \
         read, so the stage is the materialization's either way — an operator must not have to \
         know which of the fence's two reads noticed the peer",
    );
    assert_recovery_deltas(
        &harness,
        &recovery_pre,
        &[("restored", 1)],
        "positive evidence, not the trip alone: the re-read FOUND a complete header, so the layer \
         genuinely is restored (by the peer) and `restored` is honest. Claiming it off the trip \
         itself is what the post-conflict re-read exists to avoid",
    );
    assert_eq!(
        residue_wiped_metric(&harness),
        wiped_pre,
        "there was no residue, so the wipe's counter must not move"
    );
    assert_restore_write_dispositions_flat(
        &harness,
        &dispositions_pre,
        "a fence reached by a TRIPPED EXEC",
    );

    release_harness(harness).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 7 — a tripped EXEC is not evidence a peer landed anything
// ═════════════════════════════════════════════════════════════════════════════

/// `HSET superseded=1` alone in the read→`EXEC` window trips the `WATCH` just as
/// a complete restore would — and must NOT be reported as a peer's landed
/// rebuild. The re-read finds a partial header, so the verdict is `Unavailable`
/// ⇒ `UNAVAILABLE`, and nothing was written.
///
/// # Why the trip cannot be the verdict
///
/// A `WATCH` trips on any modification of the watched key. A single-field `HSET`
/// trips it; so does a `DEL`, an expiry, or a `CreateLayer` overwrite's
/// `superseded` flag. If the `Conflict` arm inferred "a peer landed the restore"
/// from the trip alone it would tick
/// `checkpoint_recovery_attempts_total{result="restored"}` for a restore that
/// never ran and hand the caller its own `NotFound` back while the metric
/// claimed a recovery — the same dishonesty arm 3 rules out one read earlier,
/// but reached down a path where the temptation is stronger, because the trip
/// LOOKS like evidence and is free to read.
///
/// So this arm's real subject is the post-conflict re-read: positive evidence,
/// same structural predicate as the pre-`MULTI` read, and a refusal when the
/// evidence is not there.
///
/// The end state is also the pointed one for the `superseded` flag specifically.
/// The header must be EXACTLY `{superseded: 1}` afterwards: the queued
/// header `HSET` was already sitting in the pipeline when the trip
/// happened, so a header carrying more than that one field is the flag buried
/// under a restored header — and both commit backends fail closed on it, leaving
/// the layer permanently unwritable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_partial_header_landing_between_the_read_and_the_exec_is_refused_not_claimed_as_a_peer() {
    let (harness, fixture, lander) = Box::pin(build_fence_arm(
        "fence-exec-trip-partial",
        PeerAction::LandSupersededFlagOnly,
        SeamPosition::BeforeExec,
    ))
    .await;

    let before = RefusalCounters::capture(&harness);

    let status = drive_stat_layer(&harness, &fixture)
        .await
        .expect_err("a tripped EXEC with no peer behind it MUST surface an error, never Ok");
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "the trip is transient by construction — the next attempt's residue wipe removes the \
         partial header (element 0 of the shared cleanup inventory) and converges — so the \
         handler's Transient ⇒ UNAVAILABLE mapping is the honest one. Got {:?}: {}",
        status.code(),
        status.message()
    );

    assert_eq!(
        lander.fire_count(),
        1,
        "the read→EXEC seam MUST have fired exactly once, or the UNAVAILABLE above came from the \
         pre-MULTI read (arm 3's subject) rather than from the post-conflict one"
    );

    assert_eq!(
        read_obj_header(&harness, fixture.object_id).await,
        HashMap::from([("superseded".to_owned(), "1".to_owned())]),
        "obj:hdr must carry EXACTLY the injected single field. This is the sharpest assertion in \
         the arm: the materialization's header HSET was ALREADY QUEUED when the trip \
         happened, so any extra field means the EXEC ran anyway — burying the CreateLayer \
         overwrite's flag and leaving the layer permanently unwritable on both commit backends"
    );

    assert_refusal_wrote_nothing(
        &harness,
        &fixture,
        &before,
        "arm 7 — a lone `superseded` flag refused by the post-conflict re-read, where the trip \
         itself LOOKS like evidence a peer landed",
    )
    .await;

    release_harness(harness).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 8 — the post-conflict re-read's own WRONGTYPE classification
// ═════════════════════════════════════════════════════════════════════════════

/// `SET obj:hdr "…"` in the read→`EXEC` window trips the `WATCH`, and the
/// post-conflict re-read then fails `WRONGTYPE` — which must surface as
/// `INTERNAL` (permanent), exactly as the pre-`MULTI` read's does.
///
/// # Why this arm exists as well as arm 4
///
/// The fence reads `obj:hdr` at TWO sites and each owns a copy of the decision.
/// Before b.h4m only the pre-`MULTI` read classified `WRONGTYPE`; the
/// post-conflict read folded the same condition into an untagged `Internal` and
/// thence into one wasted retriable `UNAVAILABLE`, so a restore that could never
/// succeed was re-driven forever while
/// `checkpoint_recovery_attempts_total{result="error"}` ticked on every pass.
/// `classify_restore_header_read_error` is the shared predicate that fixed it,
/// and a shared predicate is only shared if BOTH call sites are exercised —
/// otherwise the second site can quietly stop calling it and arm 4 still passes.
///
/// This is the arm that would catch that, and it is the only test in the repo on
/// the second call site: unlike arm 4's shape, this one is unreachable from
/// outside the read→`EXEC` window twice over — the pre-`MULTI` keyspace type
/// probe (b.zvd) catches a header that was already wrong-typed before the
/// attempt started, and the pre-`MULTI` read catches one retyped before it.
///
/// The `GET` read-back is the same technique arm 4 uses and proves the same two
/// things at once: the key was not repaired, and it was not overwritten by the
/// header hash the queued materialization was holding.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrong_typed_header_landing_between_the_read_and_the_exec_is_permanent() {
    let (harness, fixture, lander) = Box::pin(build_fence_arm(
        "fence-exec-trip-wrongtype",
        PeerAction::RetypeHeaderToString,
        SeamPosition::BeforeExec,
    ))
    .await;

    let before = RefusalCounters::capture(&harness);

    let status = drive_stat_layer(&harness, &fixture)
        .await
        .expect_err("a wrong-typed obj:hdr MUST surface an error, never Ok");
    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "the POST-CONFLICT re-read must reach the same permanent verdict the pre-MULTI read does \
         (arm 4). UNAVAILABLE here is the pre-b.h4m behaviour of this exact call site: an \
         untagged Internal folded to Transient, re-driving forever a restore that no retry can \
         complete. Got {:?}: {}",
        status.code(),
        status.message()
    );

    assert_eq!(
        lander.fire_count(),
        1,
        "the read→EXEC seam MUST have fired exactly once, or the INTERNAL above came from the \
         pre-MULTI read's classifier call (arm 4's subject) rather than the post-conflict one"
    );

    let header_value: Option<String> = harness
        .tv
        .client()
        .pool
        .get(
            harness
                .tv
                .key_schema()
                .obj_header(&fixture.object_id.to_string()),
        )
        .await
        .expect("GET obj:hdr must succeed against the string the injector left");
    assert_eq!(
        header_value.as_deref(),
        Some("not-a-hash"),
        "the wrong-typed key must survive untouched — the refusal writes nothing, including no \
         repair. Removing it is the operator's call, and a service that silently retyped a key it \
         did not create would destroy the evidence"
    );

    assert_refusal_wrote_nothing(
        &harness,
        &fixture,
        &before,
        "arm 8 — a wrong-typed header refused by the post-conflict re-read, the fence's SECOND \
         classify_restore_header_read_error call site",
    )
    .await;

    release_harness(harness).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 9 — an EXEC whose reply never arrives (b.21r)
// ═════════════════════════════════════════════════════════════════════════════
//
// `WatchedTxn::exec` counts an in-doubt verdict carrying fred's `Timeout` once,
// on the pool its caller passes — txn, for the materialization — so a
// timed-out restore `EXEC` ticks
// `live_edit_valkey_errors_total{pool="txn", kind="command_timeout"}` exactly
// once. Nothing else may count it: the materialization's in-doubt arm hands the
// caller an `Unavailable` carrying the fred error as TEXT, so nothing
// downstream sees a fred error to count.
//
// No seam substitutes this `EXEC`'s verdict (`WatchedExecOverride` is consumed
// by the commit paths only), so the arm stalls the REAL reply instead. The before-`EXEC` hook parks a `BLPOP` on the watched connection;
// the pipeline `txn.exec()` sends next queues behind it server-side, and its
// reply is held until the test releases the `BLPOP`. What ends the wait decides
// the error the in-doubt arm receives:
//
// * held past the fixture's 5 s `default_command_timeout` ⇒ fred's own
//   `Timeout` — the case that must count;
// * the connection killed once the pipeline has reached the server ⇒ a
//   connection-loss error — the control, which must count nothing.
//
// The release in the timeout case is keyed to the teardown `UNWATCH` reaching
// the server rather than to the RPC returning. The `WatchGuard`'s `Drop` sends
// that `UNWATCH` on the same stalled connection with a 650 ms per-attempt
// bound, and an `UNWATCH` that expired would count a SECOND txn timeout of its
// own — so the stall must end the moment it arrives, not whenever the
// handler's error path happens to finish.

/// Per-stage bound on every wait arm 9 performs itself: the hook firing, the
/// pipeline arriving, the `UNWATCH` arriving, and the teardown draining.
/// `ConnStall` bounds its own waits (the `BLPOP` blocking and settling).
const STALL_STAGE_BUDGET: Duration = Duration::from_secs(10);

/// How long the pipeline's bytes get to finish arriving before the stalled
/// connection's `tot-net-in` is taken as "pipeline received". fred writes the
/// pipeline in one flush, so this is slack, not a race: the next bytes on that
/// connection are the `UNWATCH`, which cannot be sent until the 5 s `EXEC`
/// timeout has fired.
const PIPELINE_SETTLE: Duration = Duration::from_secs(1);

/// Poll interval for `CLIENT LIST ID` against the stalled connection.
const STALL_POLL: Duration = Duration::from_millis(5);

/// What the stall hook needs, resolved after the layer exists.
struct StallTarget {
    /// Carries the `CLIENT LIST` / `LPUSH` / `CLIENT KILL` calls on the DATA
    /// pool — never on the connection being stalled.
    client: ValkeyClient,
    /// `obj:hdr` of the object being restored. The stall key is derived from it
    /// so the `BLPOP` shares its `{object_id}` hash slot, and `ConnStall` pins
    /// the `BLPOP` to that slot, i.e. to the connection holding the `WATCH`.
    header_key: String,
}

/// Parks a [`ConnStall`] on the restore's watched connection immediately
/// before `txn.exec()`, so the `EXEC` reply is held server-side until the test
/// ends the stall.
struct ExecStaller {
    target: OnceLock<StallTarget>,
    /// The stall, once the hook has parked it and the server reports the
    /// connection blocked. Taken by whichever ending the case applies.
    stall: tokio::sync::Mutex<Option<ConnStall>>,
    fired: AtomicUsize,
}

impl ExecStaller {
    fn new() -> Self {
        Self {
            target: OnceLock::new(),
            stall: tokio::sync::Mutex::new(None),
            fired: AtomicUsize::new(0),
        }
    }

    fn arm(&self, client: ValkeyClient, header_key: String) {
        if self.target.set(StallTarget { client, header_key }).is_err() {
            panic!("the exec staller must be armed exactly once");
        }
    }

    fn fire_count(&self) -> usize {
        self.fired.load(Ordering::Acquire)
    }

    fn target(&self) -> &StallTarget {
        self.target
            .get()
            .expect("the exec staller was used before it was armed")
    }

    /// Park the stall and return only once the server reports the connection
    /// blocked in it — so the pipeline `txn.exec()` sends next is queued behind
    /// it by construction, not by timing.
    async fn fire(&self, guarded: &Client) {
        if self.fired.fetch_add(1, Ordering::AcqRel) != 0 {
            return;
        }
        let target = self.target();
        let observer = target.client.pool.next().clone();
        let stall_key = format!("{}:b21r-exec-stall", target.header_key);
        let stall = ConnStall::park(guarded, &observer, stall_key).await;
        *self.stall.lock().await = Some(stall);
    }

    /// Take the stall to end it (`release` or `kill`).
    async fn take(&self) -> ConnStall {
        self.stall
            .lock()
            .await
            .take()
            .expect("the stall hook has not fired, so no connection is stalled")
    }
}

/// Wrap an [`ExecStaller`] as the hook the harness installs.
fn exec_stall_hook(staller: Arc<ExecStaller>) -> WatchWindowHook {
    Arc::new(move |guarded: &Client| {
        let staller = Arc::clone(&staller);
        async move { staller.fire(guarded).await }.boxed()
    })
}

/// `tot-net-in` of the stalled connection; it must still be open.
async fn stalled_net_in(stall: &ConnStall) -> u64 {
    stall
        .net_in()
        .await
        .expect("the stalled connection must still be open")
}

/// Poll `stall`'s `tot-net-in` until it exceeds `above`, within
/// [`STALL_STAGE_BUDGET`].
async fn wait_for_net_in_above(stall: &ConnStall, above: u64, what: &str) {
    let deadline = tokio::time::Instant::now() + STALL_STAGE_BUDGET;
    while stalled_net_in(stall).await <= above {
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out after {STALL_STAGE_BUDGET:?} waiting for {what} on connection {}",
            stall.conn_id()
        );
        tokio::time::sleep(STALL_POLL).await;
    }
}

/// Wait until the stall hook has fired and the restore's pipeline has reached
/// the server, still queued behind the `BLPOP`. Returns the connection's
/// `tot-net-in` with the whole pipeline received.
async fn await_pipeline_queued(staller: &ExecStaller) -> u64 {
    let deadline = tokio::time::Instant::now() + STALL_STAGE_BUDGET;
    loop {
        if staller.stall.lock().await.is_some() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the before-EXEC stall hook never fired: the restore did not reach txn.exec()"
        );
        tokio::time::sleep(STALL_POLL).await;
    }
    let held = staller.stall.lock().await;
    let stall = held.as_ref().expect("checked just above");
    let blocked = stalled_net_in(stall).await;
    wait_for_net_in_above(stall, blocked, "the EXEC pipeline to reach the server").await;
    tokio::time::sleep(PIPELINE_SETTLE).await;
    let queued = stalled_net_in(stall).await;
    assert!(
        stall.is_blocked().await,
        "the connection must still be blocked in the BLPOP with the pipeline queued behind it"
    );
    queued
}

/// The [`live_edit::storage::test_support::command_timeout_row`] row `before`,
/// plus `n` on `pool`.
fn plus_on(before: &[(&'static str, u64)], pool: &str, n: u64) -> Vec<(&'static str, u64)> {
    before
        .iter()
        .map(|(p, v)| (*p, if *p == pool { v + n } else { *v }))
        .collect()
}

/// Wait for the `Drop`-spawned teardown `UNWATCH` to finish, so any count it
/// contributes is in the row before it is read.
async fn drain_unwatch_teardown(harness: &CheckpointHarness) {
    let tracker = harness.tv.client().watch_unwatch_tracker();
    tracker.close();
    tokio::time::timeout(STALL_STAGE_BUDGET, tracker.wait())
        .await
        .expect("the Drop-spawned UNWATCH must finish within the stall budget");
}

/// Build the arm-9 fixture with an [`ExecStaller`] on the before-`EXEC` seam.
async fn build_stall_arm(name: &str) -> (CheckpointHarness, FenceFixture, Arc<ExecStaller>) {
    let staller = Arc::new(ExecStaller::new());
    let mut opts = CheckpointHarnessOptions::new(1);
    opts.restore_write_before_exec_hook = Some(exec_stall_hook(Arc::clone(&staller)));
    let (harness, fixture) = Box::pin(build_fence_fixture(name, opts)).await;
    staller.arm(
        harness.tv.client().clone(),
        harness
            .tv
            .key_schema()
            .obj_header(&fixture.object_id.to_string()),
    );
    (harness, fixture, staller)
}

/// Every in-doubt verdict reaches the client as the same retriable
/// `UNAVAILABLE`, whatever ended the wait, and the message names the in-doubt
/// arm rather than one of the materialization's other `Unavailable` exits.
fn assert_in_doubt_unavailable(status: &tonic::Status, context: &str) {
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "{context}: an in-doubt restore EXEC must surface as UNAVAILABLE — the next attempt \
         re-probes and converges. Got {:?}: {}",
        status.code(),
        status.message()
    );
    assert!(
        status
            .message()
            .contains("was left in doubt (the EXEC reply was lost)"),
        "{context}: the status must come from write_restored_object's in-doubt arm, not from \
         another Unavailable exit. Got: {}",
        status.message()
    );
}

/// b.21r: a restore `EXEC` whose reply is held past fred's command timeout
/// counts exactly one `live_edit_valkey_errors_total{pool="txn",
/// kind="command_timeout"}`, and the client still gets the unchanged in-doubt
/// `UNAVAILABLE`.
///
/// Fails if the materialization's `EXEC` stops passing through the counted
/// `WatchedTxn::exec` (or passes it the wrong pool): the timeout reaches no
/// other counting site, so the row stays flat. The whole row is
/// compared, so a count on the wrong pool fails too, as does a second count
/// from the teardown `UNWATCH` (the release keyed to its arrival is what keeps
/// that attempt from expiring — see the section comment).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restore_exec_whose_reply_times_out_counts_one_txn_command_timeout() {
    let (harness, fixture, staller) = Box::pin(build_stall_arm("fence-exec-timeout")).await;
    let timeouts_before = command_timeout_row(&harness.metrics);
    let recovery_before = recovery_cells(&harness);

    // `ConnStall::release` asserts its `LPUSH` landed on an empty stall list.
    let release_on_unwatch = async {
        let queued = await_pipeline_queued(&staller).await;
        let stall = staller.take().await;
        wait_for_net_in_above(
            &stall,
            queued,
            "the teardown UNWATCH to reach the server after the EXEC timed out",
        )
        .await;
        stall.release().await
    };
    let (outcome, released) =
        tokio::join!(drive_stat_layer(&harness, &fixture), release_on_unwatch);

    let status = outcome.expect_err("an EXEC whose reply timed out MUST surface an error");
    assert_in_doubt_unavailable(&status, "arm 9, timed-out EXEC");
    assert!(
        status.message().to_ascii_lowercase().contains("timed out"),
        "the in-doubt error must be fred's command timeout, or this arm staged the wrong \
         failure. Got: {}",
        status.message()
    );
    assert_eq!(
        staller.fire_count(),
        1,
        "the before-EXEC stall must fire exactly once"
    );
    let released =
        released.expect("the stalling BLPOP must be ended by the release LPUSH, not by an error");
    assert!(
        !released.is_null(),
        "the BLPOP must have been served by the release, i.e. the stall held until the \
         UNWATCH arrived; a nil reply means its own server timeout ended it"
    );

    drain_unwatch_teardown(&harness).await;
    assert_eq!(
        command_timeout_row(&harness.metrics),
        plus_on(&timeouts_before, "txn", 1),
        "one timed-out restore EXEC must count exactly once, on the txn pool"
    );
    assert_recovery_deltas(
        &harness,
        &recovery_before,
        &[("error", 1)],
        "arm 9, timed-out EXEC — counting the timeout changes nothing about the attempt's verdict",
    );

    release_harness(harness).await;
}

/// b.21r control: a restore `EXEC` in doubt because its connection was killed
/// with the pipeline queued reaches the same in-doubt arm with a non-timeout
/// fred error, and counts NO `command_timeout` on any pool. Pairs with the
/// arm above to show the count keys on the error's kind, not on the arm.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restore_exec_lost_with_its_connection_counts_no_command_timeout() {
    let (harness, fixture, staller) = Box::pin(build_stall_arm("fence-exec-conn-lost")).await;
    let timeouts_before = command_timeout_row(&harness.metrics);

    // `ConnStall::kill` asserts `CLIENT KILL` closed exactly the stalled
    // connection.
    let kill_once_queued = async {
        await_pipeline_queued(&staller).await;
        staller.take().await.kill().await
    };
    let (outcome, blpop_after_kill) =
        tokio::join!(drive_stat_layer(&harness, &fixture), kill_once_queued);

    let status = outcome.expect_err("an EXEC lost with its connection MUST surface an error");
    assert_in_doubt_unavailable(&status, "arm 9 control, connection killed");
    assert!(
        !status.message().to_ascii_lowercase().contains("timed out"),
        "the control must reach the in-doubt arm with a non-timeout error. Got: {}",
        status.message()
    );
    assert_eq!(
        staller.fire_count(),
        1,
        "the before-EXEC stall must fire exactly once"
    );
    assert!(
        blpop_after_kill.is_err(),
        "the stalling BLPOP must have died with the killed connection"
    );

    drain_unwatch_teardown(&harness).await;
    assert_eq!(
        command_timeout_row(&harness.metrics),
        timeouts_before,
        "an in-doubt EXEC whose error is not a timeout must count no command_timeout"
    );

    release_harness(harness).await;
}
