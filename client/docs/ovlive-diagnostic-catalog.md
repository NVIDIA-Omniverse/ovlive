<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Diagnostic event catalog

The runtime emits structured diagnostic records to registered callbacks and keeps a bounded history that [diagnostic snapshots](ovlive-diagnostic-schema.md) include. This catalog defines each event: its code and name, its default severity and the cases that override it, what its fields mean, the runtime's diagnostic defaults and the guidance for each recovery action. It is for support engineers and agents who read diagnostic output, and for engineers who forward or index it. The [machine-readable catalog](../tests/contracts/diagnostic-catalog.json) holds the same definitions with every field variant and numeric domain, and the test suite checks it against the compiled runtime.

A missing field differs from zero. Field variants describe what the runtime's emitters produce, not fields every record must carry. Preserve unknown fields and codes when forwarding a record, and never read them as success. Codes and names are never reused or redefined; additions keep catalog version 1 and diagnostic schema version 1, which are separate numbers.

## Events

Severity below is the registry default; the overrides follow the table. Code 23 is reserved: `Live.h` defines no constant for it, the runtime never emits it, and a callback filter that names it fails with INVALID_ARGUMENT.

| Code | Event | Default |
| --- | --- | --- |
| 1 | `endpoint_registered` | INFO |
| 2 | `endpoint_closed` | INFO |
| 3 | `rpc_completed` | DEBUG |
| 4 | `rpc_failed` | WARN |
| 5 | `auth_challenge` | WARN |
| 6 | `auth_replaced` | INFO |
| 7 | `auth_failed` | ERROR |
| 8 | `stream_disconnected` | WARN |
| 9 | `stream_resumed` | INFO |
| 10 | `layer_ready` | INFO |
| 11 | `object_replaced` | ERROR |
| 12 | `recovery_gate_closed` | INFO |
| 13 | `recovery_gate_opened` | INFO |
| 14 | `outcome_unknown` | ERROR |
| 15 | `backpressure_entered` | WARN |
| 16 | `backpressure_exited` | INFO |
| 17 | `save_completed` | INFO |
| 18 | `drain_completed` | INFO |
| 19 | `drain_timeout` | WARN |
| 20 | `quiescence_failed` | ERROR |
| 21 | `retry_scheduled` | INFO |
| 22 | `records_suppressed` | WARN |
| 23 | reserved | — |
| 24 | `message_accepted` | DEBUG |
| 25 | `message_write_completed` | DEBUG |
| 26 | `message_received` | DEBUG |
| 27 | `message_discarded` | WARN |
| 28 | `message_rejected` | WARN |
| 29 | `callback_failed` | ERROR |
| 30 | `flush_unsupported` | ERROR |
| 31 | `host_edits_discarded` | WARN |
| 32 | `read_paused` | WARN |
| 33 | `read_resumed` | INFO |
| 34 | `apply_lag_entered` | WARN |
| 35 | `apply_lag_exited` | INFO |

Severity overrides:

- `rpc_failed` is WARN for UNAVAILABLE, TIMEOUT, CANCELLED and UNAUTHENTICATED and ERROR for other results; a successful attempt emits `rpc_completed` at DEBUG.
- `save_completed` is DEBUG for OK and BUSY, WARN for TIMEOUT and CANCELLED and ERROR otherwise.
- `drain_completed` is INFO for OK and WARN for other results; a drain that times out emits `drain_timeout` instead.
- `quiescence_failed` reports a final or resumed runtime shutdown that timed out, at ERROR. It goes to history and metrics only, never to sink callbacks; the [shutdown metrics](ovlive-diagnostic-schema.md#shutdown) define its phase codes. BUSY refusals and nonfinal reference releases are recorded in metrics only.

"Completed" in an event name is not a success test; only the typed result establishes an operation's outcome.

### Callback failures

`callback_failed` reports an exception that escaped a native host callback. `callback_kind` is 0 request, 1 auth, 2 status or 3 session, and the record carries only numeric context: the request, operation and attempt; the auth registration, generation and challenge; the status registration, subject and delivered revision; or the session and event kind (in `reason`). Status subject and incident identify the delivered observation, not a new failure. No exception text, callback address, credentials or payload is recorded.

The failure does not change any result. A request finishes with its original result; an auth observer can receive later challenges, and the thrown challenge is not retried; a status observer is disabled while its subject stays queryable; a session disables its callbacks and fails locally with INTERNAL_ERROR. Handles still need their ordinary release. Diagnostic sink exceptions are counted separately (`sink_failures`, `sinks_disabled`) and never emit this event, and Python callbacks that the binding handles itself are outside it. The code 29 entry of `metrics.events` counts caught failures even when history is disabled or a sink is stalled.

### Unsupported flushes

`flush_unsupported` reports a host edit flush that failed its layer with UNSUPPORTED (design [section 9](ovlive-design.md#9-wire-model-section-graph-and-application-phases)). `operation` is UPDATE, `result` is UNSUPPORTED, and `reason` is 1 value type, 2 payload offset, 3 identity type change, 4 multi-update, 5 move cycle, 6 time-sample dictionary key (a dictionary-key edit of `timeSamples`) or 7 identity realization (a namespace change that identity-preserving Create, Delete and Move commands cannot realize, such as a mapper with a relative target). A flush that fails for one of these reasons emits this record; one whose compiled edit conflicts with the current scene on replay fails with the typed status only. Subject and incident identify the failed layer, whose typed status carries the same incident. Nothing was issued, and the host content keeps every edit. The record never names the path, field or value type; a USD runtime error on the host names the layer and reason, plus the path, field and value type when a single operation causes the failure.

### Discarded host edits

`host_edits_discarded` reports a live layer that closed while its host edit buffer still held records no flush had admitted, typically because the host released its last reference before the flush job ran (design [section 8](ovlive-design.md#8-authoring-and-reconciliation-model)). `count` is the number of unissued records and `result` is the layer's FAILED cause, or OK for a healthy layer. Subject identifies the closed layer and incident is its status incident, zero for a healthy layer. The records were never issued: a release discards them with the layer, and when an explicit close (a creation close, or an explicit attachment close by an owner that keeps the layer) leaves the layer alive, the host content keeps them for export but they are never sent. The layer's CLOSED status keeps the same `buffered_edit_count` with `has_unconfirmed_edits` set. The record carries no path, field or value.

### Read pacing

`read_paused` reports a layer whose host publishes slower than the service delivers: its queued read frames plus its unapplied peer coverage reached `max_requests` minus one eighth, so the read issues no further receive and HTTP/2 flow control holds the service (design [section 12](ovlive-design.md#12-stream-ordering-coverage-and-reconnect)). `read_resumed` reports the read receiving again once that depth fell to half of `max_requests`. `queue_depth` is that depth and `stream_generation` the read generation; `read_resumed` carries the current generation, which differs from the pause's after a reconnect. Subject identifies the layer, which stays `READY` once caught up. A layer that fails or closes while paused ends the interval as `FAILED` or `CLOSED` and records no `read_resumed`. The records carry no path, field or value.

### Apply lag

`apply_lag_entered` reports a layer whose host applies peer updates slower than its read delivers them: its applied sequence trails its read sequence by more than `apply_lag_warning_threshold` positions (default 256). `apply_lag_exited` reports the lag falling to half of the threshold or less, so a lag that hovers near the threshold reports one interval. Only a `READY` layer past its initial application is evaluated, and the lag counts only positions beyond the read generation's accepted catch-up target, its first complete response, which later header-only empty commits never move, so opening a long layer or catching up after a reconnect records nothing: that content arrives as one aggregate. An interval continues through `RECONNECTING`, `AUTH_WAIT` and `BLOCKED` and is evaluated again once the layer is `READY`. `read_sequence` and `applied_sequence` are the layer's sequences when the bound was crossed, the same values its status then carries, `stream_generation` is the read generation, and the exit record's `duration_ms` is the interval since the entry. Subject identifies the layer, which stays `READY`: the lag is a pace, not a fault (design [section 17](ovlive-design.md#17-operational-status)). A layer that fails or closes while lagging ends the interval as `FAILED` or `CLOSED` and records no `apply_lag_exited`. The records carry no path, field or value. The [apply-lag metrics](ovlive-diagnostic-schema.md#apply-lag) count the lagging layers and measure every peer update's receive-to-apply time.

### Service entry limits

`rpc_failed` for an UPDATE or CREATE that the live edit service refused `INVALID_ARGUMENT` over its entry limits carries the counts its `google.rpc.ErrorInfo` reports (domain `nvidia.omniverse.live.v1alpha`), so the failure names what the service counted. `reason` 1 is `DELTA_TOO_LARGE`, with the delta's `struct_commands`, `set_fields` and `time_samples` and the service's `max_entries` in `limit`; `reason` 2 is `CHILDREN_LIST_TOO_LONG`, with the list limit in `limit` and the list's length in `count`. Only a complete ErrorInfo of that domain and reason, every count a decimal number, adds the fields; any other rejection's record carries none of them. The layer or creation fails `INVALID_ARGUMENT` as for any definitive rejection; a host normally never sees this, because the runtime counts every update and seed against the service's limits and refuses one over them before any RPC (design [section 8](ovlive-design.md#8-authoring-and-reconciliation-model)).

### Write reissue

`retry_scheduled` with `operation` UPDATE or CREATE reports the backoff before the reissue of a layer write or creation that the live edit service refused `UNAVAILABLE` with its `google.rpc.ErrorInfo`, which proves nothing applied (design [section 13](ovlive-design.md#the-reissue-after-a-refusal-that-applied-nothing)). `result` is `UNAVAILABLE`, `request_id` the refused attempt, `attempt_id` the write's or creation's consecutive refusals including this one, `stream_generation` the read generation and `work_id` the write, both zero for a creation, `delay_ms` the chosen wait and `reason` its source: 1 the refusal's `google.rpc.RetryInfo` delay, which is the least wait below `reconnect_maximum_ms` and gives way to the reconnect schedule's draw when that is longer, 2 the reconnect schedule when the refusal carried none. Each refused attempt also has its `rpc_failed` record with `result` `UNAVAILABLE` and `grpc_status` 14. The refusal that reaches `reconnect_max_attempts` schedules nothing and fails the layer, or the creation, `UNAVAILABLE`; so does a creation's refusal whose backoff would end past its initial-open deadline.

`retry_scheduled` with `operation` UPDATE and `reason` 4 reports the single operation-ID re-send of an unchained layer write whose outcome is unknown, which issues at once, and the hold it starts on the layer's later writes (design [section 13](ovlive-design.md#the-single-operation-id-re-send)). `result` is `OUTCOME_UNKNOWN`, `request_id` the first attempt, `stream_generation` the read generation, `work_id` the write and `delay_ms` the remaining hold: no later write issues until the first attempt's `request_timeout_ms` deadline plus 5 s has passed, since the service may still run that attempt until its own deadline. The first attempt also has its `rpc_failed` record.

### Keepalive timeout

`stream_disconnected` for a session join (`operation` JOIN) that the live edit service ended `UNAVAILABLE` with its `google.rpc.ErrorInfo` reason `CLIENT_KEEPALIVE_TIMEOUT` (domain `nvidia.omniverse.live.v1alpha`) carries `reason` 1, so a keepalive timeout is told apart from other ends of the stream; every other session `stream_disconnected` record omits `reason`. The join rejoins with its `retry_scheduled` record as on any `UNAVAILABLE` (design [section 15](ovlive-design.md#15-sessions-presence-and-messages)).

`retry_scheduled` with `operation` UPDATE and `reason` 3 reports a resolved break of a layer's chained writes before they are re-sent (design [section 13](ovlive-design.md#chained-writes)). `count` is the writes re-sent, `target_local_ordinal` the local admission ordinal through which the chain committed, `attempt_id` the resolution rounds since the last acknowledgment or refusal that located the chain further and `delay_ms` the wait before the first re-send, zero in the first round; the first re-send carries `no_wait`. `result` is `OUTCOME_UNKNOWN` while a re-sent write's earlier attempt has an unknown outcome, otherwise `OK`. Each write's failed attempts have their own `rpc_failed` records; a chain refusal is `ABORTED`, `grpc_status` 10, with `result` `OUTCOME_UNKNOWN`. After `reconnect_max_attempts` rounds the layer fails instead of scheduling.

## Fields and interpretation

Every numeric field is a UINT64; the machine catalog's `fields`, `domains` and `variants` give each one's domain and meaning. `reason` depends on the event: the auth reason, the backpressure resolution, the save actor result, the retry cause, the write reissue delay source, the suppressed event code, the callback session event kind, the shutdown phase, the unsupported flush cause, the service's entry-limit reason or a session's keepalive timeout. Layer read retry records omit `reason` and use `result`; layer write reissue records use `result` and give the delay's source in `reason`. A finite retry's `attempt_id` identifies the next attempt; layer and session retries identify the previous one. A scheduled retry can still be prevented by cancellation, credentials or its deadline.

RPC records describe individual attempts. READ and JOIN durations are stream lifetimes; initial readiness has [its own metrics](ovlive-diagnostic-schema.md#readiness). `grpc_status` carries the raw transport code, including an explicit zero; its absence means no transport code, and the mapped public result can differ from it. Request zero with a nonzero `session_handle` is a JOIN correlation. Correlation values imply no service deduplication or idempotency.

`stream_resumed` can describe the initial connection. `layer_ready` can precede application of the first response by the host. `auth_replaced` means the runtime accepted a new token generation, not that the server authenticated it. `backpressure_exited` can resolve as failure or closure. Recovery gate events cover specific status transitions, not every write block. `message_write_completed` does not imply peer delivery, and message `queue_depth` and `pending_bytes` describe the outbound queue, even on received-message records; read-pacing `queue_depth` is the layer's incoming depth.

`canonical_resource` is the only string field: a validated canonical `live://` URI of at most 512 bytes, included only when the runtime allows resource names and the callback or snapshot requests them. In snapshots it is a top-level member of the record, outside `fields`. Records never contain tokens, certificates, signed URLs, arbitrary metadata, server text or scene and message contents. The diagnostics sample's [record encoding](ovlive-samples.md#record-encoding) and the [snapshot](ovlive-diagnostic-schema.md#document-shape) are separate representations.

Shutdown records use `reason` for the first unfinished wait; the [shutdown metrics](ovlive-diagnostic-schema.md#shutdown) define its phase codes and the shutdown duration.

## Delivery and defaults

Delivery is bounded and best effort. Severity filtering, WARN suppression, history eviction, sink overflow and shutdown can each explain a missing record. Identical repeated WARN records for one event code within 1000 ms are coalesced, and the next warning of that code that is not coalesced is preceded by a `records_suppressed` summary whose `count` and `reason` give the number and the event code; it does not summarize every dropped record. Two dedicated workers deliver records to sinks, and a sink that throws three exceptions is disabled. Metrics survive history disablement; the [schema reference](ovlive-diagnostic-schema.md#conventions) states their duration bounds and saturation rules.

| Runtime option | Default |
| --- | --- |
| `max_diagnostic_records` | 2048 |
| `max_diagnostic_bytes` | 8 MiB |
| `max_diagnostic_sink_records` | 512 queued records per sink, plus one in flight |
| `max_diagnostic_sinks` | 8 |
| `max_closed_incidents` | 128 |
| `max_status_subjects` | 4096 |
| `max_status_observers` | 64 |
| `apply_lag_warning_threshold` | 256 sequence positions |

History holds the smaller of `max_diagnostic_records` and `max_diagnostic_bytes` divided by the fixed native record size. These are logical limits, not an RSS budget. Zero capacity fields select the defaults. NULL runtime options select minimum severity INFO; a supplied zero-initialized options structure selects TRACE, because severity zero is meaningful. The disable-history flag turns history off and leaves metrics on.

## Recovery actions

Read the typed `recovery_action` together with the subject ID and revision, state, result and pending-edit evidence.

| Action | Guidance |
| --- | --- |
| 0 NONE | No recovery requested; still inspect state and result and honor host USD synchronization |
| 1 WAIT_FOR_RETRY | Let the runtime's bounded safe recovery progress; do not resend an uncertain mutation |
| 2 REFRESH_AUTH | Use the typed auth challenge and generation flow; never take credentials or decisions from logs |
| 3 RELIEVE_BACKPRESSURE | Reduce admission or let retained work progress; do not discard accepted edits |
| 4 PRESERVE_AND_REOPEN | Quiesce authoring, query the subject, export the failed layer with `SdfLayer::Export`, release all old owners, reopen authoritative state and reconcile deliberately |
| 5 RECONFIGURE | Correct the endpoint, configuration or unsupported setup; preserve accepted local work before releasing the attachment |

The exported USD file is the whole recovery artifact; diagnostic JSON cannot substitute for it. Never replay uncertain writes automatically. The [Python guide](ovlive-python.md#status-waits-and-errors) walks through the export-and-reopen procedure.
