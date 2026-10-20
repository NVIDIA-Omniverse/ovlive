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

//! b.xje, b.ze1 — every subscriber ends with the server's sibling order,
//! however it joined, in both catch-up modes.
//!
//! Each subscriber builds its scene from exactly what the stream sends it,
//! with an independent client reader model (`tests/common/sibling_model.rs`;
//! never the server's engine): the Kit client's reader, which takes a catch-up
//! `sectionOrder` as a key and moves a sibling already holding it to the end,
//! and slot-fills a live reorder; and the C++ client library's reader, which
//! takes it as an index into the parent's children and renumbers, and orders
//! a live reorder by USD's `SdfApplyListOrdering` per child list. Each model's
//! USD child lists are compared with the server's; how the lists interleave is
//! invisible to USD. Sections are typed as Kit types them (`section_type` is
//! an `SdfSpecType`).
//!
//! # Scenario 1 (b.xje): appends, moves and reorders on gap-free keys
//!
//! 1. CreateLayer seeds P and Q under the root; A(0), p1(1), B(2) and p2(3)
//!    under P by `DiffChangeSection` — the shape the Kit client's and the C++
//!    client library's uploads have — with A and B prims and p1 and p2
//!    attributes; and the prim S under Q.
//! 2. Writer A: a bare `CreateSection` of the prim C under P.
//! 3. Writer A: `MoveSection` S Q → P.
//! 4. Writer B repeats that move, still carrying Q: a stale carried parent.
//! 5. Writer A: a Kit-shaped `ReorderChildren` naming all of P's prims,
//!    `[C, A, S, B]`. The prims are reordered among the positions they hold
//!    and the attributes stay where they are: `[C, p1, A, p2, S, B]`.
//! 6. Writer A deletes B.
//! 7. Writer A: a bare `CreateSection` of the prim D under P.
//! 8. Writer B's reorder of the prims it knew, `[S, B, A, C, S]`: it names B,
//!    which writer A deleted, repeats S, and does not name D, whose create
//!    committed first. B and the repeat are ignored and USD's ordering moves D
//!    along with S, the named prim it followed: `[S, p1, D, p2, A, C]`.
//!
//! P's keys stay `0..n-1` throughout commits 2–8, so a catch-up's position and
//! a client's key agree here; scenario 2 is where they part.
//!
//! An **early** subscriber that joins before commit 2 and applies every commit
//! live (the C++ client library's reader), a **new joiner** of each reader
//! that subscribes after commit 8, and an **in-window reconnector** of each
//! reader that follows through commit 2, disconnects, and reconnects after
//! commit 8 from version 2, must all hold the server's children, in order, in
//! every child list — `[S, D, A, C]` and `[p1, p2]` under P.
//!
//! What that proves differs by commit. Commits 2–7 follow rules every client
//! shares (appends, moves, a reorder naming a whole child list, a delete), so
//! there the test shows the server matches any client. Commit 8 names only
//! part of P's prims (not D), and the early subscriber orders it by the SDD
//! §7.7 rule — `SdfApplyListOrdering` per child list, which the C++ client
//! library applies. A live Kit client does not: Kit v1 refills the named
//! prims' slots, leaving D last, and Kit v2 applies only a complete list, so
//! no live Kit subscriber is checked here. A Kit joiner and a Kit reconnector
//! are: they rebuild from what the service sends, not from the reorder.
//!
//! ## Fails on the pre-b.xje code
//!
//! Before b.xje the server stored commit 2's C at order 0 and placed it after
//! A, while the early subscriber appended; it rejected commit 8 for naming the
//! deleted B; and it installed each reorder's list as P's whole list, dropping
//! the attributes. Measured by perturbation on that code (both modes):
//!
//! * `apply_create_section` storing order 0 again — the server-list assertion
//!   fails, reading `[S, A, C, p1, D, p2]` under P;
//! * `child_list` returning the fallback list for every pair, so a reorder
//!   orders P's whole list as one — it fails there too, reading
//!   `[S, p2, D, A, p1, C]`;
//! * an early return whenever a reorder names an id that is not a listed
//!   child (the old rejection's shape, without the error) — it reads
//!   `[C, p1, A, p2, S, D]`, commit 8 having changed nothing.
//!
//! # Scenario 2 (b.ze1): reconnectors over gapped keys
//!
//! A client never renumbers its keys on a delete, so a reconnector's keys can
//! have gaps where the server's positions have none. CreateLayer seeds three
//! prims under the root, each with the shape of one case:
//!
//! * **P**, interleaved: prims A, G, B, F, H, D and attributes x, y, z, listed
//!   `A x G B y F H D z` at keys 0–8;
//! * **Q**, the delete-then-create repro: prims QA, QB, QC, QX at 0–3;
//! * **R**, a partial reorder: prims RA, RX, RB, RC, RY, RE at 0–5.
//!
//! Commits 2–6 delete G, F, H, RX and RY. Every subscriber but the joiners
//! follows them live, so a reconnector of either reader keys P's children A0
//! x1 B3 y4 D7 z8 and R's RA0 RB2 RC3 RE5. The reconnectors then disconnect
//! at version 6, and while they are away:
//!
//! 7. QB is deleted, and 8. QD created: Q is `[QA, QC, QX, QD]`, QD at key 4.
//! 9. A prim C is created under P: P's prims are `[A, B, D, C]`.
//! 10. RD is created under R (key 6), and 11. writer B, which has not seen
//!     RD, reorders R's prims `[RC, RA, RB, RE]`: R is
//!     `[RC, RA, RB, RE, RD]`, with only RC, RA and RB re-keyed.
//!
//! The reconnectors reconnect from version 6, and a joiner of each reader
//! subscribes. Live subscribers of both readers, both joiners and both
//! reconnectors must hold the server's order in every child list. (Commit
//! 11's unnamed RD is already last, so the Kit client's slot fill agrees with
//! USD's ordering here and the live Kit subscriber is checked too.)
//!
//! In snapshot mode a reconnector receives one coalesced diff, which restates
//! every listed child of each parent with a new or changed child at its
//! position: both readers rebuild the server's order from it, whatever keys
//! they held. A diff restating only the changed children — QD at 3, C at 6;
//! RD at 4 and RC, RA, RB at 0–2 — leaves the Kit reconnector with
//! `[QA, QC, QD, QX]`, P's prims `[A, B, C, D]` and `[RC, RA, RB, RD, RE]`.
//!
//! In rollback mode a reconnector replays one diff per commit, each emitting
//! the changed sections at their stored keys. The Kit reader holds exactly
//! those keys and rebuilds every list. The C++ client library's reader
//! renumbered its keys when it placed the earlier diffs, so it reads commit
//! 11's stored keys 0, 2 and 3 as indexes among renumbered siblings and
//! misplaces R's prims (measured: `[RC, RE, RA, RB, RD]`); rollback mode
//! keeps that legacy emission unchanged by design (SDD §12.10), so for that
//! reader R is left out of the rollback comparison and its other lists are
//! checked.
//!
//! # Wait budget
//!
//! Each test runs its whole scenario — CreateLayer, every commit and every
//! subscriber's reads (scenario 1: at most 33 messages — early 2 + 7, each
//! reconnector's first session 2 + 1 and second at most 6 + 1, each joiner 2;
//! scenario 2: at most 54 — each live subscriber 2 + 10, each reconnector's
//! first session 2 + 5 and second at most 5 + 1, each joiner 2) — under one
//! [`SCENARIO_BUDGET`] (100 s) deadline, so its worst case is
//! 100 s whatever the message count, under nextest's 120 s hang-killer, and a
//! stuck run fails with its own message rather than being killed silently. No
//! read carries a deadline of its own: one slow read on a loaded runner spends
//! the scenario's slack rather than failing the run. The healthy path finishes
//! in well under a second.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::{Stream, StreamExt};
use tonic::{Request, Status};

use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::ObjectState;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, update_layer_request, SubscribeReadLayerHeader,
    SubscribeReadLayerRequest, SubscribeReadLayerResponse, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::diff_read::snapshot_to_object_state;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::{read_object_state_for_test, TestValkey};
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection,
    CommandReorderChildren, OwnedDelta, StructCommand,
};

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::common::sibling_model::{
    server_child_lists, ChildListKey, Reader, SiblingModel, READERS,
};

use super::helpers::{
    build_service, make_create_data, make_create_header, make_request, make_update_data,
    streaming_create, streaming_update, AllowAllPermissions,
};

/// The one deadline over a whole scenario (module header, § Wait budget).
const SCENARIO_BUDGET: Duration = Duration::from_secs(100);

const LAYER: &str = "sibling-order-layer";
const WRITER_A: &str = "sibling-order-writer-a";
const WRITER_B: &str = "sibling-order-writer-b";

/// `SdfSpecType` values (OpenUSD `pxr/usd/sdf/types.h`), as Kit writes them
/// into `section_type`.
const PRIM: u8 = 6;
const ATTRIBUTE: u8 = 1;

const P: u64 = 1;
const Q: u64 = 2;
const A: u64 = 10;
const B: u64 = 11;
const C: u64 = 12;
const D: u64 = 13;
const S: u64 = 20;
const P1: u64 = 30;
const P2: u64 = 31;

/// The version the reconnector holds when it disconnects: after commit 2.
const RECONNECT_HAVE: u64 = 2;

/// The last commit's sequence number.
const LAST_SEQ: u64 = 8;

type SubStream = Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

fn change(parent_id: u64, section_id: u64, name: &str, ty: u8, order: u32) -> StructCommand {
    StructCommand::DiffChangeSection(CommandDiffChangeSection {
        parent_id,
        section_id,
        section_name: name.to_owned(),
        section_type: ty,
        section_order: order,
    })
}

fn structural(cmd: StructCommand) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![cmd],
        ..Default::default()
    }
}

/// A bare `CreateSection` of a prim.
fn create_prim(parent_id: u64, section_id: u64, name: &str) -> OwnedDelta {
    structural(StructCommand::CreateSection(CommandCreateSection {
        parent_id,
        section_id,
        section_name: name.to_owned(),
        section_type: PRIM,
    }))
}

fn move_section(section_id: u64, old_parent_id: u64, new_parent_id: u64) -> OwnedDelta {
    structural(StructCommand::MoveSection(CommandMoveSection {
        old_parent_id,
        new_parent_id,
        section_id,
        new_name: "S".to_owned(),
    }))
}

fn delete(parent_id: u64, section_id: u64) -> OwnedDelta {
    structural(StructCommand::DeleteSection(CommandDeleteSection {
        parent_id,
        section_id,
    }))
}

/// A `ReorderChildren` of `parent`'s prims, with the child-list id Kit sends
/// for prims (0; the server ignores it).
fn reorder_prims(parent: u64, children_list: Vec<u64>) -> OwnedDelta {
    structural(StructCommand::ReorderChildren(CommandReorderChildren {
        section_id: parent,
        children_list_id: 0,
        children_list,
    }))
}

/// The CreateLayer content: explicit-order siblings, as uploads send them.
fn seed() -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![
            change(0, P, "P", PRIM, 0),
            change(0, Q, "Q", PRIM, 1),
            change(P, A, "A", PRIM, 0),
            change(P, P1, "p1", ATTRIBUTE, 1),
            change(P, B, "B", PRIM, 2),
            change(P, P2, "p2", ATTRIBUTE, 3),
            change(Q, S, "S", PRIM, 0),
        ],
        ..Default::default()
    }
}

async fn subscribe(
    svc: &LiveEditService,
    session_id: &str,
    object_id: u64,
    sequence_num: u64,
    client_id: &str,
) -> SubStream {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER.to_owned(),
        object_id,
        sequence_num,
        client_id: client_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    handle_subscribe_read_layer(svc, request)
        .await
        .unwrap_or_else(|s| panic!("[{client_id}] the subscribe must be admitted: {s:?}"))
        .into_inner()
}

/// One message's header and reassembled body. Bounded only by the scenario's
/// [`SCENARIO_BUDGET`].
async fn next_message(stream: &mut SubStream, who: &str) -> (SubscribeReadLayerHeader, Vec<u8>) {
    let header = match stream.next().await {
        Some(Ok(SubscribeReadLayerResponse {
            response_type: Some(subscribe_read_layer_response::ResponseType::Header(h)),
        })) => h,
        other => panic!("[{who}] expected a header, got {other:?}"),
    };
    let mut body = Vec::new();
    while (body.len() as u64) < header.total_size {
        match stream.next().await {
            Some(Ok(SubscribeReadLayerResponse {
                response_type: Some(subscribe_read_layer_response::ResponseType::Chunk(c)),
            })) => body.extend_from_slice(&c.data),
            other => panic!("[{who}] expected a body chunk, got {other:?}"),
        }
    }
    (header, body)
}

/// Apply one delivered OMPF body to `model`.
fn apply_body(model: &mut SiblingModel, body: &[u8], who: &str) {
    let container =
        OmpfContainer::parse(body).unwrap_or_else(|e| panic!("[{who}] parse OMPF: {e:?}"));
    if container.delta_bytes.is_empty() {
        return;
    }
    let delta = oodf_ffi::parse_delta(container.delta_bytes)
        .unwrap_or_else(|e| panic!("[{who}] parse delta: {e:?}"));
    model.apply(&delta);
}

/// Apply every catch-up message up to the caught-up ack (`total_size == 0`)
/// and return the ack's sequence number.
async fn catch_up(stream: &mut SubStream, model: &mut SiblingModel, who: &str) -> u64 {
    loop {
        let (header, body) = next_message(stream, who).await;
        if header.total_size == 0 {
            return header.sequence_num;
        }
        apply_body(model, &body, who);
    }
}

/// Apply the next live commit, which must be `seq`.
async fn follow_live(stream: &mut SubStream, model: &mut SiblingModel, seq: u64, who: &str) {
    let (header, body) = next_message(stream, who).await;
    assert_eq!(
        header.sequence_num, seq,
        "[{who}] the live commits arrive in order"
    );
    apply_body(model, &body, who);
}

/// A subscriber: its stream (while connected) and the scene it built.
struct Subscriber {
    who: String,
    stream: Option<SubStream>,
    model: SiblingModel,
}

impl Subscriber {
    /// Subscribe from empty and apply the catch-up, which must end at `seq`.
    async fn join(
        svc: &LiveEditService,
        session_id: &str,
        reader: Reader,
        role: &str,
        seq: u64,
    ) -> Self {
        let who = format!("{role} {reader:?}");
        let mut stream = subscribe(svc, session_id, 0, 0, &who).await;
        let mut model = SiblingModel::new(reader);
        assert_eq!(
            catch_up(&mut stream, &mut model, &who).await,
            seq,
            "[{who}] the join catches up to the head"
        );
        Self {
            who,
            stream: Some(stream),
            model,
        }
    }

    async fn follow(&mut self, seq: u64) {
        let stream = self.stream.as_mut().expect("a connected subscriber");
        follow_live(stream, &mut self.model, seq, &self.who).await;
    }

    /// Reconnect in-window from `have` and apply the catch-up, which must end
    /// at `seq`.
    async fn reconnect(
        &mut self,
        svc: &LiveEditService,
        session_id: &str,
        object_id: u64,
        have: u64,
        seq: u64,
    ) {
        let mut stream = subscribe(svc, session_id, object_id, have, &self.who).await;
        assert_eq!(
            catch_up(&mut stream, &mut self.model, &self.who).await,
            seq,
            "[{}] the reconnector catches up to the last commit",
            self.who
        );
        self.stream = Some(stream);
    }
}

async fn commit(
    svc: &LiveEditService,
    session_id: &str,
    object_id: u64,
    client_id: &str,
    delta: &OwnedDelta,
) {
    let header = UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: LAYER.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: 1,
                client_id: client_id.to_owned(),
                operation_id: String::new(),
            },
        )),
    };
    let data = make_update_data(delta_to_ompf_bytes(delta, &[]));
    handle_update_layer(svc, make_request(streaming_update(vec![header, data])))
        .await
        .unwrap_or_else(|e| panic!("[{client_id}] the UpdateLayer must succeed: {e}"));
}

/// The server's stored state.
async fn server_state(tv: &TestValkey, object_id: u64) -> ObjectState {
    snapshot_to_object_state(
        &read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
            .await
            .expect("the server state must load"),
    )
}

/// Every non-empty children list the server stores, by parent.
fn server_lists(state: &ObjectState) -> BTreeMap<u64, Vec<u64>> {
    state
        .children
        .iter()
        .filter(|(_, l)| !l.is_empty())
        .map(|(p, l)| (*p, l.clone()))
        .collect()
}

/// Create the session and the layer from `content`; returns
/// `(session_id, object_id)`.
async fn create_layer(
    tv: &TestValkey,
    svc: &LiveEditService,
    content: &OwnedDelta,
) -> (String, u64) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("sibling-order-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let object_id = handle_create_layer(
        svc,
        make_request(streaming_create(vec![
            make_create_header(&session_id, LAYER),
            make_create_data(delta_to_ompf_bytes(content, &[])),
        ])),
    )
    .await
    .expect("CreateLayer must succeed")
    .into_inner()
    .object_id;
    (session_id, object_id)
}

/// Require `model`'s order in every child list, optionally leaving out the
/// lists under `except_parent`, to equal `server`'s.
fn assert_child_lists(
    case: &str,
    subscriber: &Subscriber,
    server: &BTreeMap<ChildListKey, Vec<u64>>,
    except_parent: Option<u64>,
) {
    let keep = |(key, _): &(&ChildListKey, &Vec<u64>)| Some(key.0) != except_parent;
    let held = subscriber.model.child_lists();
    let held: BTreeMap<&ChildListKey, &Vec<u64>> = held.iter().filter(keep).collect();
    let expected: BTreeMap<&ChildListKey, &Vec<u64>> = server.iter().filter(keep).collect();
    assert_eq!(
        held, expected,
        "[{case} / {}] the subscriber must hold the server's children, in order, in every \
         child list",
        subscriber.who
    );
}

async fn every_subscriber_ends_with_the_servers_sibling_order(catchup_snapshot_enabled: bool) {
    let case = if catchup_snapshot_enabled {
        "snapshot mode"
    } else {
        "rollback mode"
    };
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    svc.catchup_snapshot_enabled = catchup_snapshot_enabled;
    let (session_id, object_id) = create_layer(&tv, &svc, &seed()).await;

    // The early client and both reconnectors join at the seeded layer.
    let mut early = Subscriber::join(&svc, &session_id, Reader::Ovlive, "early", 1).await;
    let mut reconnectors = Vec::new();
    for reader in READERS {
        reconnectors.push(Subscriber::join(&svc, &session_id, reader, "reconnector", 1).await);
    }

    // Commit 2: a bare create under the seeded siblings. The reconnectors
    // follow it, then disconnect.
    commit(
        &svc,
        &session_id,
        object_id,
        WRITER_A,
        &create_prim(P, C, "C"),
    )
    .await;
    early.follow(2).await;
    for reconnector in &mut reconnectors {
        reconnector.follow(2).await;
        reconnector.stream = None;
    }

    let rest = [
        (3, WRITER_A, move_section(S, Q, P)),
        // Writer B saw S under Q: its carried parent is stale.
        (4, WRITER_B, move_section(S, Q, P)),
        // Kit's shape: every prim of P, and no attribute.
        (5, WRITER_A, reorder_prims(P, vec![C, A, S, B])),
        (6, WRITER_A, delete(P, B)),
        (7, WRITER_A, create_prim(P, D, "D")),
        // Writer B has not seen commits 6 and 7: it names the deleted B, not
        // D, and repeats S.
        (8, WRITER_B, reorder_prims(P, vec![S, B, A, C, S])),
    ];
    for (seq, writer, delta) in &rest {
        commit(&svc, &session_id, object_id, writer, delta).await;
        early.follow(*seq).await;
    }

    let mut joiners = Vec::new();
    for reader in READERS {
        joiners.push(Subscriber::join(&svc, &session_id, reader, "new joiner", LAST_SEQ).await);
    }
    for reconnector in &mut reconnectors {
        reconnector
            .reconnect(&svc, &session_id, object_id, RECONNECT_HAVE, LAST_SEQ)
            .await;
    }

    let state = server_state(&tv, object_id).await;
    assert_eq!(
        server_lists(&state).get(&P),
        Some(&vec![S, P1, D, P2, A, C]),
        "[{case}] the server stores the order every client builds: appends last, the stale \
         repeated move in place, and each reorder ordering P's prims by USD's rule among the \
         prims' own positions, ignoring the deleted B and the repeated S"
    );
    let server = server_child_lists(&state);
    for subscriber in std::iter::once(&early).chain(&joiners).chain(&reconnectors) {
        assert_child_lists(case, subscriber, &server, None);
    }
}

// ── Scenario 2 (b.ze1): reconnectors over gapped keys ───────────────────────

/// P's children, in seeded order: prims and attributes interleaved.
const GP_A: u64 = 100;
const GP_X: u64 = 101;
const GP_G: u64 = 102;
const GP_B: u64 = 103;
const GP_Y: u64 = 104;
const GP_F: u64 = 105;
const GP_H: u64 = 106;
const GP_D: u64 = 107;
const GP_Z: u64 = 108;
/// The prim created under P while the reconnectors are away.
const GP_C: u64 = 109;

/// Q's prims (the delete-then-create repro).
const GQ_A: u64 = 200;
const GQ_B: u64 = 201;
const GQ_C: u64 = 202;
const GQ_X: u64 = 203;
const GQ_D: u64 = 204;

/// R's prims (the partial reorder).
const GR_A: u64 = 300;
const GR_X: u64 = 301;
const GR_B: u64 = 302;
const GR_C: u64 = 303;
const GR_Y: u64 = 304;
const GR_E: u64 = 305;
const GR_D: u64 = 306;

const GAP_P: u64 = 1;
const GAP_Q: u64 = 2;
const GAP_R: u64 = 3;

/// The version the reconnectors hold when they disconnect: after the deletes.
const GAP_RECONNECT_HAVE: u64 = 6;

/// The last commit's sequence number.
const GAP_LAST_SEQ: u64 = 11;

/// The scenario 2 CreateLayer content (module header).
fn gapped_seed() -> OwnedDelta {
    let mut struct_commands = vec![
        change(0, GAP_P, "P", PRIM, 0),
        change(0, GAP_Q, "Q", PRIM, 1),
        change(0, GAP_R, "R", PRIM, 2),
    ];
    let p = [
        (GP_A, "A", PRIM),
        (GP_X, "x", ATTRIBUTE),
        (GP_G, "G", PRIM),
        (GP_B, "B", PRIM),
        (GP_Y, "y", ATTRIBUTE),
        (GP_F, "F", PRIM),
        (GP_H, "H", PRIM),
        (GP_D, "D", PRIM),
        (GP_Z, "z", ATTRIBUTE),
    ];
    let q = [(GQ_A, "QA"), (GQ_B, "QB"), (GQ_C, "QC"), (GQ_X, "QX")];
    let r = [
        (GR_A, "RA"),
        (GR_X, "RX"),
        (GR_B, "RB"),
        (GR_C, "RC"),
        (GR_Y, "RY"),
        (GR_E, "RE"),
    ];
    for (key, (id, name, ty)) in p.into_iter().enumerate() {
        struct_commands.push(change(GAP_P, id, name, ty, key as u32));
    }
    for (key, (id, name)) in q.into_iter().enumerate() {
        struct_commands.push(change(GAP_Q, id, name, PRIM, key as u32));
    }
    for (key, (id, name)) in r.into_iter().enumerate() {
        struct_commands.push(change(GAP_R, id, name, PRIM, key as u32));
    }
    OwnedDelta {
        struct_commands,
        ..Default::default()
    }
}

async fn reconnectors_over_gapped_keys_end_with_the_servers_order(catchup_snapshot_enabled: bool) {
    let case = if catchup_snapshot_enabled {
        "snapshot mode"
    } else {
        "rollback mode"
    };
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    svc.catchup_snapshot_enabled = catchup_snapshot_enabled;
    let (session_id, object_id) = create_layer(&tv, &svc, &gapped_seed()).await;

    let mut live = Vec::new();
    let mut reconnectors = Vec::new();
    for reader in READERS {
        live.push(Subscriber::join(&svc, &session_id, reader, "live", 1).await);
        reconnectors.push(Subscriber::join(&svc, &session_id, reader, "reconnector", 1).await);
    }

    // Commits 2–6: deletes every subscriber follows live, which leave gaps in
    // its keys under P and R.
    let deletes = [
        (2, delete(GAP_P, GP_G)),
        (3, delete(GAP_P, GP_F)),
        (4, delete(GAP_P, GP_H)),
        (5, delete(GAP_R, GR_X)),
        (6, delete(GAP_R, GR_Y)),
    ];
    for (seq, delta) in &deletes {
        commit(&svc, &session_id, object_id, WRITER_A, delta).await;
        for subscriber in live.iter_mut().chain(&mut reconnectors) {
            subscriber.follow(*seq).await;
        }
    }
    for reconnector in &mut reconnectors {
        reconnector.stream = None;
    }

    let window = [
        (7, WRITER_A, delete(GAP_Q, GQ_B)),
        (8, WRITER_A, create_prim(GAP_Q, GQ_D, "QD")),
        (9, WRITER_A, create_prim(GAP_P, GP_C, "C")),
        (10, WRITER_A, create_prim(GAP_R, GR_D, "RD")),
        // Writer B has not seen RD: every prim of R it knows.
        (
            11,
            WRITER_B,
            reorder_prims(GAP_R, vec![GR_C, GR_A, GR_B, GR_E]),
        ),
    ];
    for (seq, writer, delta) in &window {
        commit(&svc, &session_id, object_id, writer, delta).await;
        for subscriber in &mut live {
            subscriber.follow(*seq).await;
        }
    }

    let mut joiners = Vec::new();
    for reader in READERS {
        joiners.push(Subscriber::join(&svc, &session_id, reader, "new joiner", GAP_LAST_SEQ).await);
    }
    for reconnector in &mut reconnectors {
        reconnector
            .reconnect(
                &svc,
                &session_id,
                object_id,
                GAP_RECONNECT_HAVE,
                GAP_LAST_SEQ,
            )
            .await;
    }

    let state = server_state(&tv, object_id).await;
    let stored = server_lists(&state);
    assert_eq!(
        (stored.get(&GAP_P), stored.get(&GAP_Q), stored.get(&GAP_R)),
        (
            Some(&vec![GP_A, GP_X, GP_B, GP_Y, GP_D, GP_Z, GP_C]),
            Some(&vec![GQ_A, GQ_C, GQ_X, GQ_D]),
            Some(&vec![GR_C, GR_A, GR_B, GR_E, GR_D]),
        ),
        "[{case}] precondition: the server's lists after the window"
    );
    let server = server_child_lists(&state);
    for subscriber in live.iter().chain(&joiners) {
        assert_child_lists(case, subscriber, &server, None);
    }
    for reconnector in &reconnectors {
        // Rollback mode's per-commit stored keys are not indexes the C++
        // client library's renumbered siblings can take (module header).
        let except = (!catchup_snapshot_enabled && reconnector.model.reader() == Reader::Ovlive)
            .then_some(GAP_R);
        assert_child_lists(case, reconnector, &server, except);
    }
}

/// Run `scenario` in one catch-up mode under [`SCENARIO_BUDGET`].
async fn within_budget<F: std::future::Future<Output = ()>>(scenario: F) {
    tokio::time::timeout(SCENARIO_BUDGET, Box::pin(scenario))
        .await
        .unwrap_or_else(|_| panic!("the scenario did not finish within {SCENARIO_BUDGET:?}"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_subscriber_ends_with_the_servers_sibling_order_snapshot_mode() {
    within_budget(every_subscriber_ends_with_the_servers_sibling_order(true)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_subscriber_ends_with_the_servers_sibling_order_rollback_mode() {
    within_budget(every_subscriber_ends_with_the_servers_sibling_order(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnectors_over_gapped_keys_end_with_the_servers_order_snapshot_mode() {
    within_budget(reconnectors_over_gapped_keys_end_with_the_servers_order(
        true,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnectors_over_gapped_keys_end_with_the_servers_order_rollback_mode() {
    within_budget(reconnectors_over_gapped_keys_end_with_the_servers_order(
        false,
    ))
    .await;
}
