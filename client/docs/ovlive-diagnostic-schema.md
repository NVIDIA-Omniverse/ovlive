<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Diagnostic snapshot schema and metrics reference

`omniUsdLiveCopyDiagnostics` (Python `copy_diagnostics`) returns a local JSON snapshot of the runtime's configuration, subject status, runtime-wide metrics and recent history. This reference defines every member and metric of that snapshot and explains how to validate one offline. It is for support engineers and agents who read snapshots during an investigation, and for engineers who build tools that consume them. The [event catalog](ovlive-diagnostic-catalog.md) defines the event records inside a snapshot, the [samples guide](ovlive-samples.md#diagnostics-sample) shows how to produce one, and the design's [observability principles](ovlive-design.md#18-observability-principles) explain why the runtime exposes this data.

Commands in this guide run from the repository's `client` directory, and the paths they name are relative to it; build output and evidence go to `../.cache`, the `.cache` directory at the repository root.

[`diagnostic-snapshot.schema.json`](../tests/contracts/diagnostic-snapshot.schema.json) specifies the snapshot in [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-validation). The diagnostics sample's JSON Lines callback records are a different representation that this schema does not describe.

## Taking a snapshot

A snapshot sends no RPC and writes no file. Pass an endpoint handle to limit statuses, records and closed incidents to that endpoint, or zero for all of them. The export options are:

| Option | Meaning |
| --- | --- |
| `max_records` | Most recent history records to include; zero omits records. NULL options select `max_diagnostic_records` (default 2048), as does omitting `max_records` in Python. |
| `max_output_bytes` | Output cap including the terminator; zero selects 16 MiB, and larger values clamp to 16 MiB |
| `flags` | `OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES` (1) adds `canonical_resource` when the runtime's `diagnostic_allowed_flags` also allows it; `OMNI_USD_LIVE_DIAGNOSTIC_SERVER_DETAIL` (2) is accepted and adds nothing |

Call once with a NULL buffer to learn the required size, then with a buffer of that size; BUFFER_TOO_SMALL returns the new size if history grew in between. When the document exceeds the cap, the oldest records are dropped until it fits and `history.truncated` is set.

## Document shape

| Member | Type | Meaning |
| --- | --- | --- |
| `schema_version` | uint32 | Diagnostic schema version, 1 |
| `runtime_instance_id` | uint64 | Identifies this runtime initialization; records carry the same value |
| `capture_time_ms` | uint64 | Wall-clock capture time in Unix milliseconds |
| `endpoint_filter` | uint64 | The endpoint argument; 0 means unfiltered |
| `provenance` | object | `abi_version`, `proto`, `proto_sha256`, `source_revision`, `source_sha256`, `compiler` and `artifact_scope` (`native_runtime_component`) |
| `effective_configuration` | object | Every runtime option after defaults, plus `diagnostic_min_severity`, `diagnostic_allowed_flags`, `flags` and `diagnostic_export_limit_bytes` (16 MiB) |
| `statuses` | array | One status object per attached subject |
| `metrics` | object | Runtime-wide metrics; `scope` is `"runtime"` |
| `closed_incidents` | array | Status objects of closed subjects that had an incident, at most `max_closed_incidents` |
| `history` | object | `first_sequence` and `last_sequence` of the runtime's history ring, and `truncated` |
| `recent_records` | array | Event records, oldest first |

A status object holds `endpoint`, `session`, `subject_kind`, `state`, `subject_id`, `revision`, `changed_at_ms`, `result`, `operation`, `grpc_status_present`, `grpc_status`, `recovery_action`, `progress_known`, `incident_id`, `pending_write_count`, `pending_bytes`, `has_unconfirmed_edits`, `blocked_layers`, `failed_layers`, `blocked_sessions`, `failed_sessions`, `read_sequence`, `applied_sequence` and, when permitted, `canonical_resource`. These are the typed status fields of the [reference](ovlive-reference.md), except `buffered_edit_count`, which the snapshot does not serialize. The sequences are optional for consumers, since snapshots from earlier builds lack them. `grpc_status` is present exactly when `grpc_status_present` is 1, including a raw status of zero.

A record holds `event_code`, `event_name`, `severity`, `timestamp_ms`, `elapsed_ms` (since runtime start), `runtime_instance_id`, `sequence`, `endpoint`, `subject_id`, `incident_id`, a `fields` object that maps each [catalog field](ovlive-diagnostic-catalog.md#fields-and-interpretation) name to an unsigned integer, and, when permitted, `canonical_resource`. Fields are sparse: a missing field is not zero.

`first_sequence` and `last_sequence` describe the runtime's ring, not the returned records after filtering; gaps and an empty filtered history are valid. `truncated` is true when the record count, the output cap or the closed-incident limit dropped anything. False does not mean that no older record was ever suppressed or evicted.

## Conventions

These rules apply to every metric group below.

- **Integers.** Every known numeric field is an exact unsigned integer up to its uint32 or uint64 limit; parse with an integer-capable JSON parser, because floating point loses 64-bit precision. Numeric status flags are 0 or 1; `history.truncated` and the byte-overflow indicators are JSON booleans.
- **Runtime-wide.** Metrics cover the whole runtime, even in an endpoint-filtered snapshot. They carry no endpoint, subject, request or URI labels, and they keep counting with history disabled, records filtered by severity or warnings suppressed.
- **Saturation.** Cumulative counters, sums, bucket counts and high-water values each stop at UINT64_MAX independently; current gauges stay exact. The identities stated below hold only before saturation, so never subtract saturated totals to infer a current value.
- **Duration histograms.** Every duration uses the bounds `[1, 5, 10, 50, 100, 500, 1000, 5000]` milliseconds. Bounds are inclusive upper limits and buckets are noncumulative: `counts` has nine slots, the ninth holding samples above 5000 ms, and zero is a real sample in the first. A histogram holds `bounds`, `count`, `sum_ms` and `counts`. A dimensioned histogram holds `bounds` and sparse `entries`, each with its dimension values, `count`, `sum_ms` and `counts`; a missing entry means no observation.
- **Unknown buckets.** Operation dimension 16 and result dimension 22 collect values outside the public enums; they are not public codes. Other result dimensions are the public result codes of the [reference](ovlive-reference.md).
- **Separate captures.** Statuses, metrics, each executor and history are captured separately, not atomically. Records can refer to detached subjects or older revisions, and a closed incident can overlap a CLOSED status.

## Metrics

### Record and RPC counters

| Field | Meaning |
| --- | --- |
| `emitted` | Records published, including those later suppressed or dropped |
| `suppressed` | Records below the runtime's minimum severity, plus repeated WARN records coalesced within the 1000 ms window |
| `warning_suppressed` | The coalesced WARN part of `suppressed` |
| `invalid` | Rejected records, rejected resource names and out-of-range metric dimensions |
| `history_overwritten` | Records evicted from a full history ring |
| `history_dropped` | Records not stored because history is disabled or the sequence space is exhausted |
| `sink_dropped` | Records dropped from full sink queues or discarded when a sink was disabled |
| `sink_failures`, `sinks_disabled` | Exceptions thrown by diagnostic sinks, and sinks disabled after three exceptions |
| `closed_incidents_evicted` | Closed incidents evicted from the retention ring |
| `rpc_attempts`, `rpc_completions`, `rpc_in_flight` | Transport attempts started and finished, and attempts not yet finished |
| `rpc_attempts_by_operation` | Sparse `operation`, `count` rows |
| `rpc_by_operation_result` | Sparse `operation`, `result`, `count` rows of finished attempts |
| `events` | Sparse `event_code`, `count` rows, including suppressed records |
| `messages_by_event` | `event_code`, `count`, `bytes` rows for the message events 24–28 |

### RPC, save and drain durations

| Field | Dimensions | Measures |
| --- | --- | --- |
| `rpc_duration_ms` | none; `bounds` and `counts` only | Every finished transport attempt |
| `rpc_duration_ms_by_operation_result` | `operation`, `result` | The same attempts by operation and result |
| `save_duration_ms_by_result` | `result` | Each actor save call, from the call to its return |
| `drain_duration_ms_by_result` | `result` | Each public drain wait, including TIMEOUT and other failures |

An RPC sample runs from the attempt's start to the completion-queue receipt of its final tag. READ and JOIN are streams, so their samples measure stream lifetime, not readiness, and each retry is its own sample. Save excludes first-save creation and failures before attachment. Drain does not measure shutdown or remote convergence.

### Shutdown

`shutdown_duration_ms_by_phase_result` has sparse entries keyed by `phase` and `result`. Each sample measures one whole matched-runtime `Shutdown` call, not the time spent in the named phase. NOT_INITIALIZED adds no sample; each resumed attempt after a TIMEOUT adds its own.

| Phase | Meaning |
| --- | --- |
| 0 | A nonfinal initialization reference was released; no quiescence was attempted |
| 1 | Final shutdown refused with BUSY because registrations or owners remain |
| 2 | The completion queue did not finish |
| 3 | Session cleanup did not finish |
| 4 | The authentication scheduler did not finish |
| 5 | Status observer cleanup did not finish |
| 6 | Request and operational-worker cleanup did not finish |
| 7 | Diagnostic cleanup did not finish |
| 8 | All shutdown waits completed |

Phases 2–7 name the first wait still incomplete at the shared deadline; later stages may also be blocked. Only TIMEOUT emits a `quiescence_failed` record, whose `reason` field is the phase. After a TIMEOUT the runtime keeps its final owner, so `CopyDiagnostics(0)` still works; after a successful final shutdown the runtime is gone and its last sample cannot be read.

### Authentication

`metrics.auth` describes endpoint credential waits.

| Field | Meaning |
| --- | --- |
| `blocked_endpoints`, `blocked_endpoints_high_water` | Endpoints waiting for usable credentials, and the maximum |
| `stale_replacements` | Token replacements a valid endpoint rejected as a stale generation |
| `challenges_by_reason` | Sparse `reason`, `count` rows: 1 missing, 2 expiring, 3 rejected, 0 unknown |
| `failures_by_cause` | Sparse `cause`, `count` rows: 0 provider reported failure, 1 auth deadline expired, 2 rejected-generation limit, 3 internal failure |
| `credential_wait_ms_by_resolution` | Wait durations by `resolution`: 0 credentials usable, 1 terminal auth failure, 2 endpoint closed |

A wait starts when the runtime first sees unusable credentials on an endpoint that has not failed and ends when credentials become usable locally, auth fails terminally or the endpoint is finally removed. A proactive expiring challenge while credentials are still usable starts no wait. The gauge counts endpoints, not waiting RPCs, sessions or callbacks. Cause and resolution are diagnostic values, not public results; `auth_failed` records carry the cause in `auth_failure_cause`.

### Operation authentication waits

`metrics.operation_auth_waits` has one row per `kind`, each with `current`, `high_water`, `started`, `completed` and `duration_ms_by_result`.

| Kind | Counted owner |
| --- | --- |
| 0 | An admitted finite operation, including resolver metadata calls |
| 1 | A layer actor, however many writes it has queued |
| 2 | A joined session, however many messages it has queued |

A wait starts when the operation's credential gate parks on unusable credentials. Result OK means credentials became usable locally before the request was prepared, not that an RPC was sent or accepted. A close without a failure resolves CANCELLED and an abandoned owner INTERNAL_ERROR. One blocked endpoint can hold many waiting operations, so these are not endpoint counts.

### Subjects

`metrics.subjects` has three rows keyed by `kind`: 1 endpoint, 2 session, 3 layer. Each holds `current`, `high_water`, `admitted`, `detached` and `states`, nine `state`, `current` rows for the public states 1–9; state 6 is reserved and always zero. The state counts sum to `current`, and `current = admitted - detached`.

A subject counts from its insertion into the status registry until its final removal, including while CLOSING or CLOSED. Repeated registrations of one endpoint share one subject. State counts follow the latched status, so a stale READY publication cannot hide a failed subject.

### Readiness

`metrics.readiness` has two rows: `kind` 2 session and 3 layer. Endpoints have none because their local status does not prove connectivity.

| Field | Meaning |
| --- | --- |
| `initial_waiting` | Subjects that have not reached their first READY or a final state |
| `recovering` | Subjects that left READY and have not returned or ended |
| `initial_ms_by_resolution` | Admission to the first READY, terminal failure or close, by `resolution`: 0 ready, 1 failed, 2 closed |
| `recovery_ms_by_resolution` | From leaving READY for OPENING, RECONNECTING, AUTH_WAIT or BLOCKED to READY, failure or close, by the same resolutions |

Both intervals include auth waits, retry backoff and host publication. Moving among the nonready states does not restart a recovery interval, and a direct change from READY to a terminal state has none. A subject admitted already READY, terminal or CLOSING resolves at zero duration.

### Backpressure

`metrics.backpressure` has three rows keyed by `kind` (1 endpoint, 2 session, 3 layer), each with `current` and `high_water` counts of subjects in BLOCKED and `duration_ms_by_resolution`: 0 left BLOCKED, 1 failed, 2 closed.

An interval starts when a subject enters BLOCKED, or is admitted in it, and ends once when it leaves BLOCKED, fails or closes. The `backpressure_entered` and `backpressure_exited` records share these boundaries; the exit record's `duration_ms` and `reason` carry the interval and resolution. Leaving BLOCKED for AUTH_WAIT or RECONNECTING ends the interval without implying readiness. A capacity refusal that never publishes BLOCKED has no interval, and an endpoint with blocked children is not itself BLOCKED.

### Applications

`metrics.applications` has three rows keyed by `kind`: 0 optimistic local asynchronous publication, 1 ordered asynchronous application, 2 ordered synchronous application.

| Field | Meaning |
| --- | --- |
| `current`, `high_water` | Applications admitted and not yet retired, and the maximum |
| `admitted`, `retired` | Cumulative admissions and retirements |
| `queue_ms` | Admission to the start of the host callback; asynchronous kinds only |
| `body_ms` | The host callback body |
| `post_body_retention_ms` | Callback return to retirement of the last completion owner; asynchronous kinds only |
| `lifetime_ms_by_result` | Admission to retirement, by `result` |

Admission follows the actor's ordering and jitter hold, so `queue_ms` measures host dispatch, not time since the edit arrived. Cancelled or dropped work adds lifetime samples without queue or body samples.

### Jitter

`metrics.jitter` measures waits for the next ordered peer update to become eligible.

| Field | Meaning |
| --- | --- |
| `current`, `high_water`, `started`, `completed` | Wait episodes in progress, their maximum, and cumulative starts and completions |
| `unsampled_starts` | Episodes whose decision used no measured clock estimate, such as the constant fallback or missing update timestamps |
| `frozen_generation_starts` | Episodes started from a previous read generation's eligibility; these also count as unsampled |
| `requested_wait_ms` | Remaining time to the eligibility deadline at episode start |
| `target_hold_ms` | The policy's hold target at episode start, capped by the configured maximum; sampled episodes only |
| `sample_age_ms` | Age of the clock estimate at episode start; sampled episodes only |
| `observed_hold_ms_by_resolution` | Time held up to the due time, by `resolution`: 0 eligible, 1 preempted, 2 failed, 3 closed |
| `post_due_observation_ms_by_resolution` | Time from the due time until a worker observed it, by the same resolutions |

An episode starts only when the actor sees a future due time, so immediately eligible updates create none and the counts describe waits, not updates. Preemption keeps the update, which can start another episode. Post-due time includes scheduling, decoding and other actor work.

### Apply lag

`metrics.apply_lag` measures how far live layers trail their peer stream.

| Field | Meaning |
| --- | --- |
| `current`, `high_water` | Layers inside an apply-lag warning interval, between `apply_lag_entered` and `apply_lag_exited` or their failure or close, and the maximum |
| `receive_to_apply_ms` | One sample per update from the read, including initial and catch-up content, that an ordered application installed: from the read's receipt of the update's frame to the retirement of the application that installed it |
| `receive_to_apply_max_ms` | The largest such sample |

Peer updates apply in batches of up to 64 (design [section 12](ovlive-design.md#ordered-application)), and every update of a batch measures to that batch's one retirement, so an update that waited behind a long backlog shows its whole wait. The count is the number of updates, not of applications; local acknowledgments and work applied to the journal alone after an issuance-only failure add no sample, and neither does an application that fails. The interval includes jitter holdback, waiting behind an earlier application and host publication. The status's `read_sequence` and `applied_sequence` give each layer's current lag; this group carries no per-layer label.

### Layer writes

`metrics.layer_writes` describes chained layer writes (design [section 13](ovlive-design.md#chained-writes)) and the coalescing of host turns into updates (design [section 8](ovlive-design.md#8-authoring-and-reconciliation-model)).

| Field | Meaning |
| --- | --- |
| `in_flight`, `in_flight_high_water` | Local write attempts in flight across every layer, and the maximum |
| `window_high_water` | The largest issuance window under which a write was issued; 1 on an endpoint that does not chain writes |
| `window_grown`, `window_halved` | Times a layer's window grew after a window of acknowledgments without a refusal, and halved on a refusal that applied nothing |
| `chain_resends`, `resent_writes` | Resolved chain breaks that re-sent writes, and the writes they re-sent |
| `admitted_updates`, `coalesced_turns`, `max_turns_per_update` | Admitted host updates, the USD calls and change blocks they hold, and the most one update held |

`in_flight` is a gauge; the others are cumulative counts or maxima. `coalesced_turns` divided by `admitted_updates` is the mean number of host edits per update, which exceeds one only while a layer's issuance window is full.

### Capacity

`metrics.capacity.request_slots` has `ordinary` and `timing` objects; `timing` is the single auxiliary slot beside `max_requests`. `metrics.capacity.pending_bytes.reservations` counts `PendingBytes` reservations across all endpoints. Each of these three objects holds:

| Field | Meaning |
| --- | --- |
| `current`, `high_water` | Slots or reservations held, and the maximum |
| `admitted`, `released` | Cumulative admissions and releases; `current = admitted - released` |
| `rejected` | Admissions refused by this slot or byte limit |

`pending_bytes` also holds byte volumes:

| Field | Meaning |
| --- | --- |
| `current_bytes`, `high_water_bytes` | Bytes charged, and the maximum |
| `current_bytes_overflow`, `high_water_bytes_overflow` | True when the runtime-wide sum exceeds UINT64_MAX and the value is clamped; the high-water flag stays true once set |
| `admitted_bytes`, `released_bytes`, `rejected_bytes` | Cumulative byte volumes |

A request slot is held from registration until its handle is released, even after the operation completes. Byte charges follow their owners through callback cleanup and retained receive and scene data, so completing a request does not release them. Charges are logical budget accounting, including framing and reserved capacity, not bytes sent. Per-endpoint limits govern admission; the runtime-wide sum adds no global limit. Other refusals, such as invalid arguments or oversized updates, are not counted as rejections.

### Completion-queue tags

`metrics.cq_tags` tracks gRPC completion-queue tags from submission to retirement.

| Field | Meaning |
| --- | --- |
| `submitted` | Tags handed to the transport |
| `received_ok`, `received_not_ok` | Receipts by the completion queue's `ok` flag, which is not an RPC result |
| `outstanding`, `outstanding_high_water` | Submitted tags not yet received or abandoned, and the maximum |
| `handling` | Received tags whose completion handler has not returned |
| `retained`, `retained_high_water` | Tags whose cleanup has not finished, and the maximum |
| `retired` | Tags retired after cleanup |
| `abandoned` | Submitted tags destroyed without a completion |

`submitted = outstanding + received_ok + received_not_ok + abandoned` and `submitted = retired + retained`. A unary attempt uses one tag and a stream several, so receipts are not RPC counts; an idle stream keeps a Read tag outstanding without being stalled. A held destructor can leave `outstanding` and `handling` at zero while a nonzero `retained` blocks shutdown. The counts survive diagnostic shutdown so the final drain stays visible.

### Executors

`metrics.executors.operational` describes the runtime's `callback_worker_count` callback workers and `metrics.executors.diagnostic` its two diagnostic delivery workers. Each holds:

| Field | Meaning |
| --- | --- |
| `enqueued`, `started`, `completed` | Cumulative jobs queued, dispatched and returned |
| `queued`, `running`, `queue_high_water` | Jobs waiting, jobs dispatched and not yet accounted complete, and the maximum waiting |
| `configured_workers`, `live_workers` | Configured threads, and threads that have not exited |
| `oldest_queued_age_ms`, `oldest_running_age_ms` | Age of the oldest waiting job and the longest-running dispatch; zero when there is none or under 1 ms |
| `queue_wait_ms` | Enqueue to dequeue |
| `dispatch_duration_ms` | Dequeue to the dispatch's return |

A job can deliver several callbacks or do internal work, so these count jobs, not callbacks or messages. A blocked destructor after dispatch can leave `running` at zero and `completed` incremented while `live_workers` stays nonzero and shutdown cannot finish. A stalled diagnostic sink shows in the diagnostic executor without affecting the operational one.

## Profiles and compatibility

The default schema requires the version-1 document and validates each known metric group that is present. Metric groups and configuration options are additive within `schema_version` 1, so consumers must accept their absence; a present group carries all of its known fields. `$defs/currentProducer` additionally requires every metric group, configuration option and dense kind and state row that this source emits; use it to check that a build emits everything. Unknown properties and numeric codes are opaque extensions to preserve; never read an unknown result or action as success.

## Validating a snapshot

`tools/live/diagnostic_snapshot.py` validates a snapshot offline against the checked-in schema. It needs a dedicated interpreter with this exact dependency set, which nothing installs automatically:

```sh
python3 -B -m venv ../.cache/live-schema-venv
../.cache/live-schema-venv/bin/python -B -m pip install \
  jsonschema==4.26.0 referencing==0.37.0 attrs==26.1.0 \
  jsonschema-specifications==2025.9.1 rpds-py==0.30.0 typing-extensions==4.15.0
../.cache/live-schema-venv/bin/python -B tools/live/diagnostic_snapshot.py snapshot.json
../.cache/live-schema-venv/bin/python -B tools/live/diagnostic_snapshot.py \
  --current-producer snapshot.json
```

`--current-producer` applies the current-producer profile; `--identity` prints the dependency identity instead of validating. Beyond the schema, the validator rejects duplicate keys at any level, invalid UTF-8, trailing content, nonfinite numbers, fractions in integer fields and duplicate dimension tuples within a group (the schema's `x-unique-dimensions` annotation, which generic validators ignore), and checks known event codes against their catalog names. Its input budget is 16 MiB, nesting depth 64, about one million nodes and 128 characters per number. It never fetches remote schemas, treats an input's `$schema` or `$ref` as an opaque property, and reports error classes without echoing input values.

| Exit code | Meaning |
| --- | --- |
| 0 | Valid |
| 1 | Invalid or unreadable snapshot |
| 2 | Invalid arguments |
| 125 | Dependencies missing or at the wrong versions |

The `diagnostics-schema` suite runs the same validator on real native snapshots from the runtime tests; the [testing guide](ovlive-testing.md#running-the-tests) lists its options.

## What the snapshot does and does not tell you

A snapshot reports what this runtime observed locally. It does not show that peers have converged, that a save is durable on the service or that a callback was delivered: local READY and zero pending counts are local facts, and typed status, results and callbacks remain the control interface. History and sink records are bounded and best effort, so a missing record never proves that nothing happened. Capacity, executor and completion-queue figures are logical accounting, not process RSS, memory or overhead measurements. Schema validation proves structure only, not redaction, URI correctness, provenance authenticity or consistency between captures. A snapshot never substitutes for the exported USD layer that [recovery](ovlive-diagnostic-catalog.md#recovery-actions) preserves.
