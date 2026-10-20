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

// Integration tests for the SubscribeReadLayer handler: input validation
// and catch-up Cases 1/2/3.
//
// Tests call `handle_subscribe_read_layer` directly with a LiveEditService
// backed by a real Valkey (TestValkey fixture), mock auth clients, and
// hand-built `SubscribeReadLayerRequest` messages.

use std::sync::Arc;

use futures::StreamExt;
use tonic::Request;

use live_edit::grpc::proto::{subscribe_read_layer_response, SubscribeReadLayerRequest};
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// ── Subscribe-specific helpers ───────────────────────────────────────────────

fn make_subscribe_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    sequence_num: u64,
    client_id: &str,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        object_id,
        sequence_num,
        client_id: client_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═══════════════════════════════════════════════════════════════════════════════
// Validation tests (t3.1it.oi.3m.8z)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Empty session_id → INVALID_ARGUMENT ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_empty_session_id_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let request = make_subscribe_request("", "some-layer", 0, 0, "client-1");

    let result = handle_subscribe_read_layer(&svc, request).await;
    let status = result.err().expect("empty session_id must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "empty session_id must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
    assert!(
        status.message().contains("session_id"),
        "error message should mention session_id, got: {}",
        status.message()
    );
}

// ── Empty layer_name → INVALID_ARGUMENT ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_empty_layer_name_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // No session needed: `validate_layer_name` runs with the other header
    // validators, before `get_base_path`.
    let request = make_subscribe_request("some-session", "", 0, 0, "client-1");

    let result = handle_subscribe_read_layer(&svc, request).await;
    let status = result.err().expect("empty layer_name must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "empty layer_name must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
    // The shared `validate_layer_name` wording — the same text CreateLayer /
    // UpdateLayer / StatLayer / DeleteLayer answer for a blank name.
    assert_eq!(status.message(), "layer name must not be blank");
}

// ── Invalid layer_name → INVALID_ARGUMENT before the stream opens ───────────

/// `SubscribeReadLayer` refuses every layer name `CreateLayer` would refuse,
/// at setup, with the handler returning `Err` (Trailers-Only) rather than a
/// stream that later ends `NOT_FOUND`.
///
/// Discriminating against the pre-change handler, which checked only for
/// emptiness: each name below is non-empty, so it would have reached the
/// catch-up, failed the mapping lookup, and ended mid-stream `NOT_FOUND` with
/// the unvalidated name echoed back in the status message. The setup rejection
/// is also what makes the teardown line's `status_message` safe to log, since
/// that `NOT_FOUND` message quotes the name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_invalid_layer_name_is_rejected_at_setup_like_create_layer() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "p1-session", "p1-real-layer").await;

    // One byte over the shared 255-byte name cap.
    let oversized = "x".repeat(256);
    let cases: [(&str, &str); 5] = [
        ("   ", "layer name must not be blank"),
        (
            "bad\u{1b}[31mlayer",
            "layer name contains control characters",
        ),
        ("a/../b", "layer name contains path traversal sequence '..'"),
        ("//unc/share", "layer name starts with UNC path prefix"),
        (oversized.as_str(), "layer name exceeds"),
    ];
    for (layer_name, want) in cases {
        let request = make_subscribe_request(&session_id, layer_name, 0, 0, "client-1");
        let status = match handle_subscribe_read_layer(&svc, request).await {
            Err(status) => status,
            Ok(_) => panic!(
                "layer_name {layer_name:?} opened a stream; it must be refused at setup, before \
                 any response head, like CreateLayer refuses it"
            ),
        };
        assert_eq!(
            status.code(),
            tonic::Code::InvalidArgument,
            "layer_name {layer_name:?}: {status:?}"
        );
        assert!(
            status.message().starts_with(want),
            "layer_name {layer_name:?}: expected the shared validator's message `{want}`, got \
             `{}`",
            status.message()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Catch-up integration tests (t3.1it.oi.3m.pv)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Case 1: Full content (object_id=0) ──────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_case1_full_content_with_object_id_zero() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "case1-session", "case1-layer").await;

    // Subscribe with object_id=0 → Case 1 (full content).
    let request = make_subscribe_request(&session_id, "case1-layer", 0, 0, "subscriber-1");

    let response = handle_subscribe_read_layer(&svc, request)
        .await
        .expect("Case 1 subscribe must succeed");

    let mut stream = response.into_inner();

    // Collect the first message — should be a header with total_size > 0.
    let first = stream
        .next()
        .await
        .expect("stream must yield at least one message")
        .expect("first message must be Ok");

    match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(header)) => {
            assert!(
                header.total_size > 0,
                "Case 1 full content must have total_size > 0, got: {}",
                header.total_size
            );
            assert!(header.object_id > 0, "header must report a valid object_id");
            assert_eq!(
                header.sequence_num, 1,
                "freshly created layer should be at sequence_num=1"
            );
        }
        other => panic!("expected Header message, got: {other:?}"),
    }

    // If total_size > 0, there should be at least one data chunk following.
    let second = stream
        .next()
        .await
        .expect("stream must yield a data chunk after header")
        .expect("data chunk must be Ok");

    match second.response_type {
        Some(subscribe_read_layer_response::ResponseType::Chunk(chunk)) => {
            assert!(!chunk.data.is_empty(), "data chunk must contain OMPF bytes");
        }
        other => panic!("expected Chunk message, got: {other:?}"),
    }

    // b.rdr: the brand-new (Case 1) full-state catch-up must terminate with a
    // total_size==0 caught-up ack header carrying the delivered sequence_num.
    // This is the `OkLatest` marker the Kit client gates stage-apply on; the
    // populated arm pre-fix dropped it. Drain remaining chunks (if any) then
    // assert the trailing ack.
    loop {
        // Time-bound the read so a missing caught-up ack fails fast (10s)
        // instead of hanging `cargo test` forever (b.rdr).
        let next = tokio::time::timeout(std::time::Duration::from_secs(10), stream.next())
            .await
            .expect("stream timed out (10s) waiting for the caught-up ack after the content chunks")
            .expect("stream must yield the caught-up ack after the content chunks")
            .expect("message must be Ok");
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(chunk)) => {
                assert!(
                    !chunk.data.is_empty(),
                    "trailing data chunk must be non-empty"
                );
            }
            Some(subscribe_read_layer_response::ResponseType::Header(ack)) => {
                assert_eq!(
                    ack.total_size, 0,
                    "brand-new catch-up trailer must be a total_size==0 caught-up ack, got {}",
                    ack.total_size
                );
                assert_eq!(
                    ack.sequence_num, 1,
                    "caught-up ack must echo the delivered sequence_num (1 for a fresh layer)"
                );
                break;
            }
            other => panic!("expected Chunk or caught-up Header, got: {other:?}"),
        }
    }
}

// ── Case 3: Up-to-date (matching object_id and current seq) ────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_case3_up_to_date_returns_ack_header() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "case3-session", "case3-layer").await;

    // Subscribe with matching object_id and current seq (1 for freshly created).
    let request = make_subscribe_request(&session_id, "case3-layer", object_id, 1, "subscriber-1");

    let response = handle_subscribe_read_layer(&svc, request)
        .await
        .expect("Case 3 subscribe must succeed");

    let mut stream = response.into_inner();

    // Should receive exactly one ack header with total_size=0.
    let first = stream
        .next()
        .await
        .expect("stream must yield the ack header")
        .expect("ack header must be Ok");

    match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(header)) => {
            assert_eq!(
                header.total_size, 0,
                "Case 3 ack must have total_size=0, got: {}",
                header.total_size
            );
            assert_eq!(
                header.object_id, object_id,
                "ack header must echo the object_id"
            );
            assert_eq!(
                header.sequence_num, 1,
                "ack header must echo the current sequence_num"
            );
        }
        other => panic!("expected ack Header with total_size=0, got: {other:?}"),
    }
}

// ── Non-existent layer → NOT_FOUND ──────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_nonexistent_layer_returns_not_found() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Create a session but no layer.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("no-layer-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let request = make_subscribe_request(&session_id, "nonexistent-layer", 0, 0, "subscriber-1");

    // Post-Epic-3 rewrite: the layer-mapping lookup runs inside the spawned
    // subscription task, so NOT_FOUND is delivered as the first item on the
    // stream rather than as a synchronous handler error.
    let response = handle_subscribe_read_layer(&svc, request)
        .await
        .expect("handler must return Ok — NOT_FOUND is delivered on the stream");

    let mut stream = response.into_inner();
    let first = stream
        .next()
        .await
        .expect("stream must yield the NOT_FOUND status");
    let status = first.expect_err("first stream item must be an error");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "non-existent layer must return NOT_FOUND, got: {}",
        status.code()
    );
}

// ── A genuine `deleted` sentinel → NOT_FOUND *claiming a deletion* ──────────
//
// b.hxk narrowed the "was deleted" wording to the one cause that actually
// observed a deletion. This is the positive half of that narrowing: the arm
// that MUST keep saying it.
//
// The negative half — a state read that answered NotFound, where recovery could
// not or would not restore, and the terminal status must NOT claim a deletion —
// is `subscribe_read_layer_recovery_disabled_returns_not_found_directly` in
// `tests/integration/checkpoint/recovery_hook_test.rs`, which asserts the
// `state was lost and could not be recovered` wording on the same
// `handle_deleted_and_rebootstrap` terminal arm. The two are what make the
// distinction a contract rather than a phrasing preference: identical code
// path, identical status code, opposite claims about durable data, selected
// solely by `RebootstrapCause`.
//
// Without this half, the fix could regress by making BOTH causes say "state was
// lost" — which would be just as wrong in the other direction, telling an
// operator that a layer their own `DeleteLayer` removed had suffered state loss.

// ── Both in-window catch-up builders, not just the harness default ──────────
//
// The sentinel is detected at TWO independent sites, selected by
// `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED`:
//
// - `false` (the legacy rollback path, and the `build_service` harness default)
//   reaches it from `catch_up_in_window`'s up-front scan over the entries its
//   paged in-window replay parsed, which runs before that replay opens its
//   consumer, computes any per-commit diff or sends anything;
// - `true` (the SHIPPED PRODUCTION DEFAULT — see the `LIVE_EDIT_CATCHUP_
//   SNAPSHOT_ENABLED` row in SDD §16 and in
//   `docs/service/docs/configuration.md`) reaches it from the in-window point
//   read's last-entry check (a `deleted` sentinel can only be the stream's last entry), which
//   `catch_up_in_window_snapshot` acts on up front and which reconstructs no
//   per-commit diffs.
//
// Two call sites, two chances to regress independently — and the production one
// is the one an operator actually runs. Both arms below therefore assert the
// same terminal message, at the same position; a fix applied to only one site
// fails exactly one of them. `svc.catchup_snapshot_enabled` is set directly on
// the built service, matching `e2e_subscribe_read_layer_recovery_test.rs`'s
// parameterized reconnect arms, because the harness builder hardcodes `false`.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_deleted_sentinel_terminal_status_says_the_layer_was_deleted_legacy_catchup() {
    assert_deleted_sentinel_terminal_status_says_the_layer_was_deleted(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_deleted_sentinel_terminal_status_says_the_layer_was_deleted_snapshot_catchup() {
    assert_deleted_sentinel_terminal_status_says_the_layer_was_deleted(true).await;
}

/// Shared body of the two arms above. `catchup_snapshot_enabled` selects which
/// in-window catch-up builder — and therefore which of the two sentinel
/// detection sites — the subscriber runs through; every assertion is identical
/// on both, which is the point.
async fn assert_deleted_sentinel_terminal_status_says_the_layer_was_deleted(
    catchup_snapshot_enabled: bool,
) {
    use fred::interfaces::StreamsInterface;
    use live_edit::storage::notifications::{DELETED_STREAM_PAYLOAD, NOTIFICATION_PAYLOAD_FIELD};
    use std::time::Duration;

    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    svc.catchup_snapshot_enabled = catchup_snapshot_enabled;

    // Distinct per-arm session names so the two arms cannot collide on the
    // shared Valkey when nextest runs them concurrently.
    let session_name = if catchup_snapshot_enabled {
        "deleted-sentinel-session-snapshot"
    } else {
        "deleted-sentinel-session-legacy"
    };
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, session_name, "deleted-sentinel-layer").await;

    // Arrange: append the retirement tombstone exactly as `delete_layer` and the
    // CreateLayer-overwrite path do — `XADD * payload deleted`. The mapping is
    // deliberately LEFT INTACT, which selects the `Some(_same)` terminal arm:
    // the one whose pre-b.hxk WARN called this shape defensive-and-unexpected,
    // and the one a `StateGone` subscriber used to land on routinely and
    // wrongly.
    let _: String = tv
        .client()
        .pool
        .xadd(
            tv.key_schema().layer_stream(object_id),
            false,
            None,
            "*",
            vec![(NOTIFICATION_PAYLOAD_FIELD, DELETED_STREAM_PAYLOAD)],
        )
        .await
        .expect("planting the retirement tombstone must succeed");

    // Act: reconnect at version 0. The create seed sits at `1-0`, which is
    // exactly the in-window start id for `have_version == 0`, so the retained
    // window the catch-up reads holds the seed AND, as its last entry, the
    // tombstone — no blocking read, no timing dependence.
    let request = make_subscribe_request(
        &session_id,
        "deleted-sentinel-layer",
        object_id,
        0,
        "subscriber-1",
    );
    let response = handle_subscribe_read_layer(&svc, request)
        .await
        .expect("handler must return Ok — the terminal status is delivered on the stream");

    // Assert: the stream terminates with the deletion status. The retained
    // window holds the seed commit AHEAD of the tombstone, and both catch-up
    // builders act on the sentinel up front — the rollback replay from its
    // scan of the parsed entries, the snapshot builder from the point read's
    // last-entry check — and return before sending anything at all. The seed
    // commit is superseded by the deletion, and sending it first would put
    // content ahead of the ending. Drain a small
    // bounded number of Ok items looking for the terminal status, so that a
    // regression delivers its count to the position assertion below rather
    // than failing here on an unexpected message.
    //
    // Each poll is bounded (finding 3): a regression where the sentinel stops
    // terminating the stream leaves the subscriber sitting in steady-state
    // `XREAD BLOCK` with nothing more to deliver, which without a timeout hangs
    // to nextest's 120 s hard-kill and reports as a hang rather than as this
    // assertion. Ten seconds is far above anything this path legitimately takes
    // (the whole retained window is already in the stream before the first
    // poll).
    const MAX_DRAINED_MESSAGES: usize = 4;
    let mut stream = response.into_inner();
    let mut terminal = None;
    let mut drained_ok = 0usize;
    for _ in 0..MAX_DRAINED_MESSAGES {
        let next = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect(
                "the subscription stream neither delivered nor terminated within 10 s — the \
                 `deleted` sentinel is already the retained window's last entry, so acting on it needs \
                 no blocking read and cannot legitimately take this long",
            );
        match next {
            Some(Err(status)) => {
                terminal = Some(status);
                break;
            }
            Some(Ok(_)) => {
                drained_ok += 1;
                continue;
            }
            None => panic!(
                "the stream ended CLEANLY. A retired object must end the subscription with a \
                 status, or a gRPC client reads a normal stream end and never re-subscribes"
            ),
        }
    }
    let status = terminal.unwrap_or_else(|| {
        panic!(
            "the stream delivered {MAX_DRAINED_MESSAGES} messages without reaching the terminal \
             status — the `deleted` sentinel in the retained window was not acted on"
        )
    });

    // WHICH detection site fired. Without this, either arm is a coverage
    // illusion: delete the up-front scan and the subscriber still terminates
    // with the same status — it delivers its catch-up content (the coalesced
    // diff, or the seed commit's per-commit replay), hands over to steady
    // state, and the steady-state loop meets the very same tombstone one entry
    // later and acts on it there. The message assertion below cannot tell those
    // two apart, so it would keep passing against an arm with no sentinel
    // handling at all.
    //
    // Position is what separates them. The up-front scan runs BEFORE the
    // consumer is opened and before any content is sent, so a working arm
    // terminates on message #1 with nothing delivered ahead of it; a
    // steady-state rescue necessarily sits behind at least the catch-up content
    // and its caught-up ack. Pinning zero drained messages therefore pins the
    // catch-up site specifically — on the legacy arm too, whose replay no longer
    // sends the seed commit (sequence 1, new to a `have_version=0` subscriber)
    // ahead of the tombstone.
    let site = if catchup_snapshot_enabled {
        "catch_up_in_window_snapshot"
    } else {
        "catch_up_in_window"
    };
    assert_eq!(
        drained_ok, 0,
        "`{site}` detects the sentinel in its up-front scan, which returns before opening a \
         consumer or sending anything — so the terminal status must be the FIRST item on the \
         stream. Having drained {drained_ok} message(s) first means the scan did not fire and \
         the subscriber was rescued downstream (steady state meeting the same tombstone), or \
         the replay sent content ahead of the ending, which leaves this catch-up site untested"
    );

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "a retired object has nothing left to serve, got: {}",
        status.code()
    );
    assert_eq!(
        status.message(),
        "layer 'deleted-sentinel-layer' was deleted",
        "a `deleted` sentinel READ OFF THE STREAM is the affirmative evidence of removal, and \
         the only cause licensed to say so (b.hxk). Softening this to the state-loss wording \
         would tell an operator that a layer their own DeleteLayer removed had suffered state \
         loss"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.c7a — a server-driven consumer tear-down must NOT end the RPC with a clean OK
// ═══════════════════════════════════════════════════════════════════════════════
//
// The incident's client-visible signature. The layer-stream consumer returns
// `()`, so its only channel to this handler is the `terminal_error` slot:
// `run_steady_state` sees `consumer.recv() == None`, drains the slot, and
// returns `Err(status)` if something is there or `Ok(())` if nothing is. An
// empty slot is therefore an affirmative claim that the subscription ended
// NORMALLY — and a gRPC client reading a normal stream end does not
// re-subscribe. Every consumer exit that leaves it empty silently stops
// delivering live updates to a client that believes it is fine.
//
// Pre-b.c7a both permanent-error exits returned bare. On a deployed service a
// post-token-rotation routing gap, misclassified as permanent, therefore closed
// live subscriptions with `OK` and Kit went quiet with nothing to retry on.
//
// The test below drives that path end-to-end through the REAL handler. It
// injects nothing: it points the layer-stream key at a string, so the stream
// reader's XREAD gets a genuine `WRONGTYPE` from Valkey. `WRONGTYPE`
// is chosen over the routing message that motivated b.c7a on purpose — it is
// unarguably permanent, so this pins the contract for the whole permanent CLASS
// rather than for whichever message fred happens to emit. Reclassifying one
// message would have left `Auth`, `Protocol`, `Parse`, and every unrecognised
// fred error still closing cleanly.
//
// Scope note: the complementary half — a subscription that RIDES OUT a
// transient connection-level failure and goes on relaying — is covered in two
// other places.
//
// At the reader level, `tests/integration/storage/stream_mux_fault_test.rs`
// drives it through the stream multiplexer's fault-injection seam
// (`transient_failures_are_ridden_out_without_losing_entries`, and
// `a_router_miss_or_a_spent_redirect_budget_on_the_xread_is_ridden_out_without_a_reconnect`
// for the routing shapes). A handler test could reach that seam too, since it
// lives on the `ValkeyClient`, but the retry is decided in the reader, so it is
// pinned there.
//
// At the RPC level — a real `SubscribeReadLayer` stream staying open and
// relaying the next commit across a genuine connection loss — it is covered by
// `cluster_subscribe_rpc_relays_across_connection_loss` in
// `tests/integration/storage/cluster_subscriber_test.rs` (b.yxt). A real
// cluster provides the seam this file lacks: the disturbance is a server-side
// `CLIENT KILL` on the exact socket the RPC's consumer is parked on, so no
// injection point is needed. That test is `LIVE_EDIT_TEST_CLUSTER_URL`-gated
// and runs in CI's `test-cluster:` job.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_permanent_consumer_error_ends_the_stream_with_unavailable() {
    use fred::interfaces::KeysInterface;
    use std::time::Duration;

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "c7a-session", "c7a-layer").await;

    // Arrange: replace the layer-stream key with a string. Every stream command
    // against it now fails `WRONGTYPE` — permanent, and not swept up by b.c7a's
    // message-matched routing carve-out.
    let stream_key = tv.key_schema().layer_stream(object_id);
    let _: u64 = tv
        .client()
        .pool
        .del(stream_key.clone())
        .await
        .expect("DEL of the layer stream must succeed");
    let _: () = tv
        .client()
        .pool
        .set(stream_key, "not-a-stream", None, None, false)
        .await
        .expect("SET must succeed");

    // Act: subscribe in the up-to-date regime (matching object_id, current seq).
    // That regime reads the layer mapping and the object header, opens the
    // consumer at `{seq}-0`, and hands straight over to steady state — it never
    // touches the stream key itself, so the WRONGTYPE lands exactly where this
    // test aims it: the stream reader's XREAD, which isolates the bad key.
    let request = make_subscribe_request(&session_id, "c7a-layer", object_id, 1, "subscriber-1");
    let response = handle_subscribe_read_layer(&svc, request)
        .await
        .expect("handler must return Ok — the tear-down is delivered on the stream");
    let mut stream = response.into_inner();

    // Assert: the RPC terminates with an ERROR. The caught-up ack for the
    // up-to-date regime may or may not have been emitted before the consumer's
    // probe failed, so drain a small bounded number of Ok items looking for the
    // terminal status rather than pinning the interleaving.
    const MAX_DRAINED_ACKS: usize = 4;
    let mut status = None;
    for _ in 0..MAX_DRAINED_ACKS {
        let next = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect(
                "the subscription stream neither delivered nor terminated within 10 s — the \
                 reader's XREAD should fail on the WRONGTYPE key immediately",
            );
        match next {
            Some(Err(s)) => {
                status = Some(s);
                break;
            }
            Some(Ok(msg)) => {
                // Only the zero-body caught-up ack is legitimate here; a
                // content message would mean the regime routing changed.
                match msg.response_type {
                    Some(subscribe_read_layer_response::ResponseType::Header(h)) => assert_eq!(
                        h.total_size, 0,
                        "the up-to-date regime emits only a zero-body caught-up ack before \
                         steady state"
                    ),
                    other => panic!("unexpected pre-termination message: {other:?}"),
                }
            }
            None => panic!(
                "the subscription stream CLOSED CLEANLY after its consumer hit a permanent \
                 error. That is the b.c7a defect verbatim: a clean gRPC OK is reserved for \
                 cancellation, receiver-gone and a Deleted-driven end, and a client reading a \
                 normal stream end never re-subscribes — so the subscriber silently stops \
                 receiving updates while believing the stream simply finished"
            ),
        }
    }

    let status = status.expect("stream must terminate with a status, not keep emitting acks");
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "a server-driven tear-down must surface as Code::Unavailable (14) so the client's \
         standard retry policy re-subscribes; got: {} ({})",
        status.code(),
        status.message()
    );
    // Provenance: `ServiceError::Unavailable`'s message reaches the client
    // verbatim, so this pins that the status came from the reader's poison-key
    // isolation rather than from some other Unavailable on the subscribe path
    // (a state read, a recovery attempt). Without it the test could pass for
    // the wrong reason.
    assert!(
        status.message().contains("not readable as a stream"),
        "the status must carry the reader's terminal message naming the unreadable stream key; \
         got: {}",
        status.message()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.27b — a transient catch-up snapshot read must end the stream UNAVAILABLE
// ═══════════════════════════════════════════════════════════════════════════════

/// A transient Valkey failure on the catch-up's materialized-state read must
/// end the subscription with a status the Kit client retries.
///
/// At or below the atomic-read limits the catch-up reads state through
/// `read_object_snapshot`'s atomic read, whose error mapper is shared with
/// UpdateLayer's commit path and so classifies a transient fred error as the
/// in-process retry signal `ServiceError::Aborted`. No retry loop wraps the
/// catch-up, so that `Aborted` escapes the handler. It must reach the client as
/// `UNAVAILABLE`: the Kit client's retry policy treats `ABORTED` as
/// terminal, so an `ABORTED` here left a viewer's layer blank until a manual
/// rejoin.
///
/// The fault goes in through
/// `ValkeyClient::install_object_snapshot_read_fault_injector_for_test`, which
/// hands a canned fred `Timeout` to the atomic read in place of its `EVALSHA`
/// result, through the same classifier a real error takes. Every assertion is
/// keyed so the test cannot pass for the wrong reason:
///
/// * the status message names `EVALSHA object_snapshot_read` — the header read
///   and `open_consumer` already answer `UNAVAILABLE` on their own transient
///   failures, so a bare code check could be satisfied by one of them without
///   the snapshot read ever failing;
/// * no `total_size == 0` caught-up ack precedes the status — the failure
///   ended the catch-up, it did not ride through it;
/// * the injector queue is drained — the seam really fired.
///
/// The in-window test covers a reconnect regime under the default
/// `catchup_snapshot_enabled`; out-of-window takes the same proven read, so it
/// gets no separate test. No checkpoint loop runs on these fixtures, so the
/// one queued error can only be consumed by the catch-up's read.
///
/// Feature-gated: without `--features test-fault-injection` this module is
/// compiled out and the run reports green having covered nothing.
#[cfg(feature = "test-fault-injection")]
mod transient_snapshot_read {
    use super::*;

    use std::collections::HashMap;
    use std::time::Duration;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use fred::interfaces::StreamsInterface;
    use futures::Stream;
    use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};
    use tonic::Status;

    use live_edit::grpc::layer::handle_update_layer;
    use live_edit::grpc::proto::SubscribeReadLayerResponse;
    use live_edit::storage::diff_read::ObjectSnapshotReadFaultInjector;

    /// One deadline for the whole drain, however many messages arrive before
    /// the terminal status. The healthy failure path ends in milliseconds.
    const DRAIN_DEADLINE: Duration = Duration::from_secs(30);

    /// The op label the atomic snapshot read's script-level failure carries:
    /// it is what proves the failure came from the catch-up's state read.
    const SNAPSHOT_READ_LABEL: &str = "EVALSHA object_snapshot_read";

    /// Arm the atomic snapshot-read seam with one transient fred `Timeout`.
    /// Installed on the fixture's client, which shares the slot with the
    /// service's clone.
    fn arm_one_transient_snapshot_read_failure(
        tv: &TestValkey,
    ) -> Arc<ObjectSnapshotReadFaultInjector> {
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        injector.push(FredError::new(
            FredErrorKind::Timeout,
            "injected catch-up atomic snapshot read timeout",
        ));
        tv.client()
            .install_object_snapshot_read_fault_injector_for_test(injector.clone());
        injector
    }

    /// Drain the stream to its terminal status and confirm the stream then
    /// ends. Panics on a bare end-of-stream: that is gRPC `OK`, which the Kit
    /// client does not retry either.
    async fn drain_to_terminal_status<S>(
        stream: &mut S,
    ) -> (Vec<SubscribeReadLayerResponse>, Status)
    where
        S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
    {
        tokio::time::timeout(DRAIN_DEADLINE, async {
            let mut delivered = Vec::new();
            loop {
                match stream.next().await {
                    Some(Ok(msg)) => delivered.push(msg),
                    Some(Err(status)) => {
                        assert!(
                            stream.next().await.is_none(),
                            "the error status must be the stream's terminal item"
                        );
                        return (delivered, status);
                    }
                    None => panic!(
                        "the stream ended with no status after {} message(s): a bare EOS is \
                         gRPC OK, which the Kit client does not retry",
                        delivered.len()
                    ),
                }
            }
        })
        .await
        .expect("the subscription neither terminated nor errored within the drain deadline")
    }

    fn assert_ends_unavailable_from_the_snapshot_read(
        regime: &str,
        delivered: &[SubscribeReadLayerResponse],
        status: &Status,
        injector: &ObjectSnapshotReadFaultInjector,
    ) {
        assert_eq!(
            status.code(),
            tonic::Code::Unavailable,
            "[{regime}] a transient catch-up snapshot read must end the stream UNAVAILABLE so \
             the Kit client re-subscribes — it treats ABORTED as terminal; got {} ({})",
            status.code(),
            status.message()
        );
        assert!(
            status.message().contains(SNAPSHOT_READ_LABEL),
            "[{regime}] the status must come from the snapshot read (`{SNAPSHOT_READ_LABEL}`), \
             not from the header read or `open_consumer`, which answer UNAVAILABLE on their own \
             and would make this test vacuous; got: {}",
            status.message()
        );
        let acked = delivered.iter().any(|msg| {
            matches!(
                &msg.response_type,
                Some(subscribe_read_layer_response::ResponseType::Header(h)) if h.total_size == 0
            )
        });
        assert!(
            !acked,
            "[{regime}] no caught-up ack may precede the failure status: the catch-up did not \
             complete"
        );
        assert_eq!(
            injector.remaining(),
            0,
            "[{regime}] the injector queue must be drained — a leftover entry means the seam \
             never fired and the status above came from somewhere else"
        );
    }

    /// Regression test: a brand-new subscriber whose catch-up snapshot read
    /// hits a transient Valkey failure. Pre-fix it ended `ABORTED`.
    async fn assert_brand_new_transient_snapshot_read_ends_unavailable(
        catchup_snapshot_enabled: bool,
    ) {
        let regime = if catchup_snapshot_enabled {
            "brand_new/snapshot"
        } else {
            "brand_new/legacy"
        };
        let tv = TestValkey::setup().await;
        let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
        svc.catchup_snapshot_enabled = catchup_snapshot_enabled;
        let (session_id, _object_id) =
            create_session_and_layer(&tv, &svc, "b27b-brand-new", "b27b-layer").await;

        let injector = arm_one_transient_snapshot_read_failure(&tv);

        let request = make_subscribe_request(&session_id, "b27b-layer", 0, 0, "joiner");
        let mut stream = handle_subscribe_read_layer(&svc, request)
            .await
            .expect("handler must return Ok — the failure is delivered on the stream")
            .into_inner();
        let (delivered, status) = drain_to_terminal_status(&mut stream).await;

        assert_ends_unavailable_from_the_snapshot_read(regime, &delivered, &status, &injector);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn subscribe_brand_new_transient_snapshot_read_ends_unavailable_snapshot_catchup() {
        assert_brand_new_transient_snapshot_read_ends_unavailable(true).await;
    }

    /// The `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false` rollback builder takes
    /// the same atomic read, so it must answer the same.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn subscribe_brand_new_transient_snapshot_read_ends_unavailable_legacy_catchup() {
        assert_brand_new_transient_snapshot_read_ends_unavailable(false).await;
    }

    /// A non-reset structural delta adding `section_id` under the root, so
    /// each commit advances the sequence and XADDs one stream entry.
    fn add_root_child(section_id: u64) -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id,
                section_name: format!("sec-{section_id}"),
                section_type: 0,
                section_order: 0,
            })],
            ..Default::default()
        }
    }

    /// An in-window reconnect under the default `catchup_snapshot_enabled`
    /// (`catch_up_in_window_snapshot`, building the coalesced diff from its
    /// atomic read) whose state read hits a transient Valkey failure.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn subscribe_in_window_reconnect_transient_snapshot_read_ends_unavailable() {
        const LAYER: &str = "b27b-in-window-layer";
        // The reconnector last saw the create (seq 1); two commits follow.
        const HAVE_VERSION: u64 = 1;

        let tv = TestValkey::setup().await;
        let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
        svc.catchup_snapshot_enabled = true;
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc, "b27b-in-window", LAYER).await;

        for section_id in [10, 11] {
            let ompf =
                crate::common::owned_delta::delta_to_ompf_bytes(&add_root_child(section_id), &[]);
            let request = make_request(streaming_update(vec![
                make_update_header(&session_id, LAYER, object_id),
                make_update_data(ompf),
            ]));
            handle_update_layer(&svc, request)
                .await
                .unwrap_or_else(|e| panic!("UpdateLayer (section {section_id}) must succeed: {e}"));
        }

        // Precondition: the entry right after `HAVE_VERSION` is retained, so
        // the handler selects the in-window regime rather than out-of-window.
        let first_after: Vec<(String, HashMap<String, bytes::Bytes>)> = tv
            .client()
            .pool
            .xrange_values(
                tv.key_schema().layer_stream(object_id),
                format!("{}-0", HAVE_VERSION + 1),
                "+",
                Some(1),
            )
            .await
            .expect("XRANGE must succeed");
        assert_eq!(
            first_after.first().map(|(id, _)| id.as_str()),
            Some("2-0"),
            "precondition: the stream must retain entry 2-0 for the in-window regime"
        );

        let injector = arm_one_transient_snapshot_read_failure(&tv);

        let request =
            make_subscribe_request(&session_id, LAYER, object_id, HAVE_VERSION, "reconnector");
        let mut stream = handle_subscribe_read_layer(&svc, request)
            .await
            .expect("handler must return Ok — the failure is delivered on the stream")
            .into_inner();
        let (delivered, status) = drain_to_terminal_status(&mut stream).await;

        assert_ends_unavailable_from_the_snapshot_read(
            "in_window/snapshot",
            &delivered,
            &status,
            &injector,
        );
    }
}
