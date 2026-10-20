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

pub(crate) mod helpers;
/// Shared JSON log capture for the completion-logging tests below.
pub(crate) mod log_capture;

/// The storage service refusing the permission probe as a malformed request:
/// `INVALID_ARGUMENT` + INFO for a request-supplied base path, `INTERNAL` +
/// ERROR for a session's stored one, and 405 still `INTERNAL`.
mod auth_invalid_request_test;
mod authz_disabled_test;
/// b.ua9: what a `SubscribeReadLayer` tailing a retired object sees through
/// the real handlers — it re-loads onto the new object after an overwrite
/// whose inline wake-up failed, and after a concurrent create replaced the
/// object it tails, and it ends `NOT_FOUND` after one orphan-sweep pass when a
/// `DeleteLayer` stopped after removing the mapping. The two seam-driven cases
/// are gated on `test-fault-injection`.
mod b_ua9_subscriber_test;
/// b.v6d: the client `operation_id` through the real `UpdateLayer` /
/// `CreateLayer` handlers — the wire answer table, header validation before
/// the body is buffered, the start-record read answering only an authorized
/// write, the `CreateLayer` re-send's obligations (the recorded object's first
/// base, the re-driven overwrite tail), and an overwrite whose first commit try
/// failed committing on its second. The seam-driven cases sit in inner
/// `#[cfg(feature = "test-fault-injection")]` modules.
mod b_v6d_operation_id_test;
/// b.yuo part D — the catch-up external-value miss protocol: a blob collected
/// after the snapshot is re-read under a per-holder `{gc}` read hold (every
/// atomic regime, and a flight's late waiter re-attaching to a successor
/// flight), a miss under a live hold is genuine and takes fetch-on-miss, a
/// lost or lapsed hold ends UNAVAILABLE, the hold and every stream
/// registration are released on each exit, and a source scan pins every build
/// to the protocol's one helper. The seam-driven cases are gated on
/// `test-fault-injection` inside the module.
mod b_yuo_catch_up_miss_test;
mod capabilities_test;
/// b.pav scope (a) — chained `UpdateLayer` writes through the real handler and
/// a real server: chain order across replicas, the chain gate's exits, its
/// deadline margin, refusals, metrics and completion line, `no_wait`,
/// `INVALID_UPDATE`, a cancel at the gate over a raw HTTP/2 client, a tonic
/// client with its own deadline timer over a delayed link, and the interim
/// commit-time bridge; under `test-fault-injection`, the counted-retry rule,
/// the superseded shape and the skipped start-record read. Load-bearing
/// because a chained write that commits out of chain order, or twice, applies
/// a client's edits in the wrong order with no error.
mod chained_writes_test;
/// SDD §17.2.1 behind a real tonic server: the auth interceptor's rejection
/// line, the metrics layer's transport backstop and drop guard, one line per
/// failure through the full stack, and silence on the health / reflection /
/// capabilities probes that Kubernetes and discovery hit continuously.
mod completion_log_server_test;
mod create_layer_test;
mod cross_replica_delete_session_test;
mod delete_layer_stream_grace_test;
mod delete_session_test;
/// b.21r — `live_subscriber_updates_total` and the SubscribeReadLayer half of
/// `live_edit_join_latency_seconds`, driven through real subscriptions and read
/// off the production exporter.
mod delivery_metrics_test;
/// b.tbw part one — the per-delta size refusals through the real handlers:
/// a delta over 99,999 entries is `INVALID_ARGUMENT` `DELTA_TOO_LARGE`, a
/// children list over 100,000 ids `CHILDREN_LIST_TOO_LONG`, each decoded as
/// the C++ client library decodes it, with nothing applied; a delta of exactly
/// the limit commits and comes back whole in its catch-up.
mod delta_size_refusal_test;
/// b.yuo — the external-value release budget's upload gate through the real
/// handlers: an upload carrying external values waits at a full budget and
/// proceeds or is refused UNAVAILABLE with nothing applied and the
/// `EXT_RELEASE_BUDGET_FULL` retry hint; inline-only
/// uploads and DeleteLayer are never gated; an unreadable total or a disabled
/// budget lets the upload through.
mod ext_release_gate_test;
/// b.682 — the CreateLayer / UpdateLayer upload content check (SDD §11.1)
/// through the real handlers and a real external-value store: a part that
/// fails its claim is refused INVALID_ARGUMENT with nothing applied, a stored
/// blob survives a later bad upload of its hash, a correct upload heals a
/// truncated blob stored before the check, and `observe` stores and counts.
mod ext_value_integrity_test;
/// SDD §5.1 — an HTTP/2 connection to or from the service survives hundreds of
/// presence-sized DATA frames left unread, on the client side (several unread
/// `JoinSession` streams on one tonic channel) and on the service side (a
/// parked join). Load-bearing because h2 0.4.16–0.4.18 closed the whole
/// connection with GOAWAY `ENHANCE_YOUR_CALM` after about 100–160 such frames,
/// killing every stream on it.
mod h2_small_frame_budget_test;
/// SDD SR-4 — the JoinSession presence lifecycle: index-first registration,
/// the lease keeper from registration to teardown, and the fenced teardown on
/// every exit. Load-bearing because a lease that lapses under a slow replay
/// announces a false `PeerLeft` to every peer (PRD AC-3).
mod join_lifecycle_test;
mod join_session_cleanup_test;
mod join_session_test;
mod jr8_chunk_count_test;
mod jr8_disconnect_test;
mod jr8_session_cache_test;
mod layer_management_test;
/// SR-1.2 — every non-empty update commit XADDs the `ompf` field. Load-bearing
/// because checkpoint tail capture and `SubscribeReadLayer` both classify an
/// ompf-less `update` entry (above sequence 1) as a §8.4 empty commit by
/// invariant (the two shapes are byte-identical on the wire), so a violation
/// restores as a no-op over a real state change and reaches every subscriber
/// as an empty delta.
mod ompf_stream_invariant_test;
mod postman_regression_test;
mod presence_broadcast_drop_test;
/// b.21r — `live_presence_events_total` and the JoinSession half of
/// `live_edit_join_latency_seconds`, driven through real JoinSession streams.
mod presence_events_metric_test;
/// PRD FR-8 — the orphan sweep's presence index backstop: a dead member left
/// un-indexed by either compound failure (a joiner dying between register and
/// re-add after a concurrent index removal; a failed index removal undoing a
/// joiner's re-add) is indexed again by one orphan-sweep pass and then swept
/// and announced once; a set emptied after the walk is re-added empty and
/// unindexed by the next sweep; a refused walk is skipped. Load-bearing because
/// the presence sweeper finds sessions only through the index, so an
/// un-indexed dead member is otherwise a phantom peer forever. Drives the
/// presence sweeper's seam, so it exists only under `test-fault-injection`.
#[cfg(feature = "test-fault-injection")]
mod presence_index_backstop_test;
/// SDD §13.2 — the JoinSession client keepalive: an opted-in join whose
/// request stream carries no bytes for three intervals is torn down (peers
/// see one `PeerLeft`) and ended `UNAVAILABLE CLIENT_KEEPALIVE_TIMEOUT`; any
/// bytes keep it, a keepalive never reaches peers, and a join that did not
/// opt in, or a service with the interval 0, is never ended for silence.
/// Load-bearing because it is the only thing that ends a client that
/// vanished behind an HTTP/2-terminating proxy. Its held-replay case drives
/// the replay seam, so that one test exists only under `test-fault-injection`.
mod presence_keepalive_test;
mod presence_recovery_test;
/// SDD SR-4.4 — a JoinSession delivers each other member exactly once across
/// its membership replay and its live events, and the set of ids that
/// suppresses live duplicates ends at the join's own event, a lag or a time
/// limit. Load-bearing because a peer registering inside the join's
/// subscribe-to-snapshot window otherwise reaches it twice. Drives the replay
/// and registration seams, so it exists only under `test-fault-injection`.
#[cfg(feature = "test-fault-injection")]
mod presence_replay_dedup_test;
/// SDD §13.2 — a JoinSession stream whose presence receiver lagged past a
/// participant's join or leave is torn down (peers see it leave) and ended
/// `UNAVAILABLE` so the client rejoins with a fresh participant list, and the
/// ending is counted. Load-bearing because a stream that went on would show
/// its client a wrong participant list with no later event to correct it.
/// Its late-first-read case seeds members, so it exists only under
/// `test-fault-injection`.
mod presence_resync_test;
/// The presence sweeper's crash-sized `left` burst reaching both pub/sub
/// relays (presence and session-deleted) of two surviving replicas, observed
/// through real JoinSession streams: every observer gets every `left` once,
/// and no relay lags. Load-bearing because a
/// lagging relay loses the skipped events for every local stream of their
/// sessions. Drives the sweeper seam, so it exists only under
/// `test-fault-injection`; two of its three crash-scale shapes, those with
/// thousands of streams, are `#[ignore]`d measurements.
#[cfg(feature = "test-fault-injection")]
mod presence_sweep_relay_burst_test;
/// SDD SR-3 — the bounded, fenced presence sweeper pass: one command in
/// flight, page and chunk bounds, leadership handoff, per-session error
/// isolation, the index race with a join, and event order against rejoining
/// keepers. Load-bearing because a sweep that removes a live connection, or
/// announces a removal twice or out of order, shows every peer a phantom leave.
/// Drives the sweeper and JoinSession seams, so it exists only under
/// `test-fault-injection`.
#[cfg(feature = "test-fault-injection")]
mod presence_sweeper_test;
/// `live_edit_pubsub_relay_lagged_total` — the pub/sub messages each
/// replica-wide relay (presence, session-deleted) skipped behind the
/// subscriber client's message channel, forced exactly with a small channel
/// and a held relay. Load-bearing because a skipped message is lost for every
/// local stream on the replica, and this counter is the only signal of it.
mod pubsub_relay_lag_test;
/// b.682 — the end-of-stream record an upload commits on, for each way an
/// HTTP/2 request stream ends, over a real tonic / hyper / h2 server.
/// Load-bearing because it pins the library facts the record rests on: a
/// hyper or h2 upgrade that read a reset as END_STREAM would let a truncated
/// upload commit.
mod request_end_wire_test;
/// b.94g — every non-OK completion of a unary layer/session RPC emits exactly
/// one structured `RPC failed` line. Load-bearing because the log line IS the
/// deliverable: before it, a client-rejected UpdateLayer existed only as a
/// Prometheus counter tick carrying a gRPC code and no request identity.
mod rpc_failure_log_test;
/// b.rgn — behind a real tonic server, trace export carries exactly one server
/// span per RPC (one for an auth-refused RPC too) and no h2 / hyper / tonic /
/// tower span. Load-bearing because exporting h2's per-frame spans overflowed
/// the export queue continuously on dev, about 800 spans per commit.
mod rpc_server_span_test;
/// SDD §7.8, §12.6, §13.2, §22.12 — a session token's cancel cause decides how
/// its streams end (a deletion `NOT_FOUND` / `OK`, a session missing from the
/// live store a retryable `UNAVAILABLE`), and `acquire_session_cancel`'s
/// recovery pass restores a flushed session but never one `DeleteSession`
/// removed. Load-bearing because neither client retries `NOT_FOUND`: a
/// restorable session ended as a deletion is lost to its clients, and a
/// deleted one restored reappears in ListSessions.
mod session_cancel_cause_test;
mod session_test;
mod session_tokens_sweep_test;
/// b.e1h, b.kgb — graceful shutdown ends live `SubscribeReadLayer` and
/// `JoinSession` streams `UNAVAILABLE` with the shutdown retry hint, idle ones
/// included, paced from just behind the server's GOAWAY so the reconnects
/// arrive as a ramp; a client that retries only that code reconnects on
/// another replica without losing an update; the listener closes with the
/// drain; and a client that stopped reading cannot hold the shutdown past its
/// bound after GOAWAY. Load-bearing because the Kit client never retries `OK`:
/// a clean shutdown ending silently ends its live feed on every rolling
/// restart.
mod shutdown_ending_test;
/// b.xje — an early live subscriber, a new joiner and an in-window reconnector
/// all end with the server's sibling order, in both catch-up modes, across
/// appends, a stale-parent move, a Kit-shaped reorder of all of a prim's child
/// prims among its interleaved attributes, and a stale partial reorder that
/// names a deleted prim and a repeat and misses a peer's newer create.
/// Load-bearing because a
/// late joiner that orders siblings differently from the clients that applied
/// every commit live is looking at a different scene.
mod sibling_order_parity_test;
/// b.8jc — a time sample at `-0.0`, in bytes that physically carry it, through
/// the real handlers: `UpdateLayer` on the fast path and both structural
/// routes and `CreateLayer` store exactly one row per section and time, at
/// `+0.0`, which the readers serve; a live subscriber still receives the
/// writer's bytes verbatim, `-0.0` included, while a late joiner gets `0.0`.
mod signed_zero_time_test;
/// SDD §17.2.1 — `SubscribeReadLayer` and `JoinSession` log every non-OK
/// completion exactly once, at setup (handler wrapper) or at teardown (the
/// spawned task's tail). Load-bearing because every transient subscribe
/// failure is `UNAVAILABLE` on the wire, so the line's `status_message` is the
/// only per-event record of which read failed.
mod stream_completion_log_test;
/// b.f4f — bounded subscription admission, the cancellation-aware outer
/// send, and the reserved terminal-trailer slot. Load-bearing because the Kit
/// client does not retry gRPC `OK`, so a server-driven tear-down that degrades
/// into a bare EOS ends the subscription for good.
mod subscribe_admission_test;
/// b.xtb stage 2 — one capacity budget bounds every capacity wait of a
/// `SubscribeReadLayer`. Every capacity refusal, and every ending of the
/// catch-up's own stream registration recorded before its first message, comes
/// before content during catch-up; a refusal comes after the caught-up ack only
/// for a steady-state overwrite re-bootstrap. Load-bearing because both demo
/// clients reset their retry counts on the first content message: an ending
/// after content at a saturated replica is a reconnect loop.
mod subscribe_capacity_budget_test;
/// b.y3m Epic 3 — catch-ups of a layer above the atomic-read limits share one
/// fenced read per replica (the layer's flight): waves, bursts, the reader cap,
/// the flight's capacity waits charged per catch-up, its sizing, and every
/// ending leaving nothing held. Load-bearing because a join wave on a huge
/// layer otherwise costs one fenced read — one writer pause — per joiner.
/// Drives test seams throughout, so it exists only under
/// `test-fault-injection`.
#[cfg(feature = "test-fault-injection")]
mod subscribe_fenced_catch_up_test;
/// b.y3m Epics 2–3 — brand-new catch-ups of large layers under concurrent
/// writers are served from one proven read — atomic at or below the
/// atomic-read limits, the layer's shared fenced read above them: structural
/// commits are applied exactly once (no double-apply), and every catch-up ends
/// verified at its proven cut. Load-bearing because the quiescence loop these
/// replace lost its proof on such layers and re-delivered what its snapshot
/// already held.
mod subscribe_proven_read_test;
mod subscribe_regime_metrics_test;
mod subscribe_regimes_test;
/// b.xtb stage 4 — a layer stream trimmed past a subscription's position
/// before its reader read it ends the subscription `stream_gap` (retryable
/// `UNAVAILABLE`, one WARN completion line) instead of skipping the trimmed
/// commits silently, unless its catch-up covered them (FR-2.6). Drives a
/// hung-read fault and a catch-up seam, so it exists only under
/// `test-fault-injection`.
#[cfg(feature = "test-fault-injection")]
mod subscribe_stream_gap_test;
mod subscribe_test;
/// b.f1v — `handle_update_layer`'s single checkpoint-recovery arm at the gRPC
/// boundary: an absent `obj:hdr` reaches it (NOT_FOUND), a partial one must not
/// (FAILED_PRECONDITION, hook never consulted).
mod update_layer_recovery_arm_test;
mod update_layer_test;
mod upload_admission_test;
/// b.xd1 — `CreateLayer` / `UpdateLayer` refused before the commit marker
/// for a dependency outage answer `UNAVAILABLE` `UPSTREAM_UNAVAILABLE` (this
/// call applied nothing) at every pre-commit site, and every `UNAVAILABLE`
/// from the commit on carries no detail. Load-bearing because the client
/// re-sends a hinted refusal as nothing-applied: a hint after the commit could
/// apply a write twice. Its seam-driven cases sit in an inner
/// `#[cfg(feature = "test-fault-injection")]` module.
mod upstream_unavailable_test;
