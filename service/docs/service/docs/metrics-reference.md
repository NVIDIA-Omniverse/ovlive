<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(live-edit-metrics-reference)=
# Live Edit metrics reference

This page lists the metrics an operator alerts on, builds dashboards from, or reads while diagnosing a problem.
How to collect them, and the alert rules built on them, are on {ref}`Monitoring <live-edit-monitoring>`.

(live-edit-metrics-reference-reading)=
## Reading this page

- **Names.** Metric names carry their unit: `_seconds`, `_bytes`, `_microseconds`. Counters end in `_total`. Histograms are exported as `_bucket`, `_sum` and `_count` series.
- **Types.** A gauge marked "fleet-wide" holds one value for the whole fleet: the replica doing that work reports it and the others report the same value or `0`. Aggregate it with `max()`, never `sum()`.
- **Labels.** Each row lists every label value the service emits. `service` is the pod name, or the value of `serviceLabel`. A metric with no `service` label is identified by your scrape target's `instance` or `pod` label.
- **When series appear.** Most counters report every label combination at `0` from the first scrape; a row marked "appears on first occurrence" does not. Gauges and histograms appear with their first value.
- **Conditional metrics.** Backup counters report `0` even with backups off; backup gauges and histograms appear only when `checkpoint.adapter` is not `Disabled`, except the `live_edit_checkpoint_ext_mirror_*` gauges, which read `0` with backups off. Scaler metrics exist only in `autoscaling.mode: scaler`, on the scaler pod.
- **Stability.** A release can add label values. Write queries with `sum by (label)` and `!=` exclusions rather than lists of values.

"Watch for" lines name the conditions an operator acts on; the alert rules are on {ref}`Monitoring <live-edit-monitoring-recommended-alerts>`.

(live-edit-metrics-reference-serving)=
## Is it serving?

These measure request traffic, errors and latency on each replica. Emitted always.

| Metric | Type | Unit | Labels | Description |
|---|---|---|---|---|
| `live_endpoint_calls_total` | Counter | calls | `endpoint`: `CreateSession`, `DeleteSession`, `ListSessions`, `JoinSession`, `CreateLayer`, `UpdateLayer`, `DeleteLayer`, `StatLayer`, `SubscribeReadLayer`, `Ping`, `other`; `grpc_code`: the 17 gRPC code names, `OK` to `UNAUTHENTICATED`; `service` | Completed RPCs. Every combination is present at zero from startup. |
| `live_endpoint_duration_microseconds` | Histogram | µs | `endpoint`, `grpc_code`, `service`, as above | Duration of each completed RPC; buckets from 100 µs to 30 s. For streaming RPCs this is the stream's lifetime. |
| `live_edit_inflight_rpcs` | Gauge | calls | `endpoint`, `service` | Calls in progress; for `JoinSession` and `SubscribeReadLayer`, open streams. |
| `live_edit_join_latency_seconds` | Histogram | seconds | `endpoint`: `SubscribeReadLayer`, `JoinSession`; `service` | Time from a successful join's request until the user has the full state; buckets from 5 ms to 60 s. |
| `live_active_sessions` | Gauge | sessions | `service` | Sessions with at least one `JoinSession` connection on this replica. Absent until startup completes. |
| `live_active_connections` | Gauge | connections | `service` | `JoinSession` connections open on this replica. Absent until startup completes. |
| `live_subscriber_updates_total` | Counter | deliveries | `endpoint`: `SubscribeReadLayer`; `service` | Live edits delivered to subscribers, one per recipient. Catch-up content is not counted. |
| `live_presence_events_total` | Counter | events | `event_type`: `joined`, `left`, `message`; `service` | Presence events delivered to session participants. |

`endpoint="other"` collects health checks, reflection, the capabilities service and unknown methods.

**Watch for:** server-fault codes (`INTERNAL`, `UNKNOWN`, `DATA_LOSS`) on any endpoint — read the `internal error` (`ERROR`) and `RPC failed` (`WARN`) log lines.
**Watch for:** `live_endpoint_duration_microseconds` on `JoinSession` and `SubscribeReadLayer` — it measures session length, not speed; use `live_edit_join_latency_seconds`.

Diagnostic metrics for slow joins:

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `live_subscribe_regime_total` | Counter | `regime`: `brand_new`, `in_window`, `out_of_window`, `up_to_date`; `catchup_mode`: `snapshot`, `legacy`; `service` | Which kind of catch-up subscribers needed: first connect, reconnect, or already current. Appears on first occurrence. |
| `live_subscribe_catch_up_duration_seconds` | Histogram | `regime`, `catchup_mode`, `service` | Time spent building a subscriber's catch-up. If joins are slow but this is fast, the wait was before the catch-up: admission or authorization. |
| `live_subscribe_catch_up_delivered_bytes` | Histogram | `regime`, `catchup_mode`, `service` | Size of each catch-up; it grows with the layer. |
| `live_edit_auth_call_duration_seconds` | Histogram | `backend`: `storage`, `userinfo`; `protocol`: `rest`, `grpc` | Latency of calls to the permissions and userinfo services. |
| `live_edit_presence_keepalive_timeouts_total` | Counter | `service` | `JoinSession` streams ended because the client stopped sending keepalives. |
| `live_edit_presence_rejoins_total` | Counter | `service` | Open session connections whose presence entry had lapsed and that the service re-registered itself; peers see them join again. |
| `live_edit_presence_lease_refresh_failures_total` | Counter | `service` | Failed presence lease refreshes, retried at the next refresh; rises with backing-store trouble. |
| `live_edit_presence_resyncs_total` | Counter | `service` | Session connections ended `UNAVAILABLE` because they fell too far behind on presence events; the client rejoins. |
| `live_edit_presence_broadcast_dropped_total` | Counter | `service` | Presence events a slow session connection missed. |
| `live_edit_pubsub_relay_lagged_total` | Counter | `relay`: `presence`, `session_deleted`; `service` | Notifications from other replicas that this replica fell behind on. |
| `live_edit_presence_stale_removed_total` | Counter | `service` | Participants removed because their replica stopped refreshing them, for example after a crash. |
| `live_edit_presence_stale_removal_delay_seconds` | Histogram | `service` | How long such participants stayed visible after their lease lapsed. |
| `live_edit_presence_sweep_oldest_stale_age_seconds` | Gauge | `service` | Largest removal delay in the last clean-up pass; fleet-wide, use `max()`. Users still shown after leaving show up here. |
| `live_edit_presence_sweep_duration_seconds` | Histogram | `outcome`: `completed`, `lease_lost`, `cancelled`, `error`; `service` | Duration of each presence clean-up pass. No `completed` pass for several minutes means stale participants are not being removed. |

(live-edit-metrics-reference-delivery)=
## Is live delivery keeping up?

These measure how quickly other users' edits reach subscribers, and why subscriptions end. Emitted always.

| Metric | Type | Unit | Labels | Description |
|---|---|---|---|---|
| `live_edit_subscriber_delivery_lag_seconds` | Histogram | seconds | `stage`: `enqueued`, `transport` | Largest age of each subscription's live edits per second or so, when handed to the subscriber; buckets from 1 ms to 120 s. |
| `live_edit_subscriber_stream_exits_total` | Counter | streams | `reason`: `client_disconnect`, `shutdown`, `session_cancel`, `session_absent`, `slow_reader_deadline`, `inner_watchdog`, `reader_capacity`, `catchup_capacity`, `fenced_read_wait_cap`, `stream_gap`, `error` | Subscriptions that ended, once each, by why. |

`stage="enqueued"` is the edit's age when it is queued on the subscriber's stream: the service's and the backing store's work.
`stage="transport"` is its age when the service hands it to HTTP/2; the gap between the two is time waiting for the client or proxy to read.
Lag is measured from the writing replica's clock, so clock skew between replicas shifts it.

The `reason` values on `live_edit_subscriber_stream_exits_total`:

- `client_disconnect` — the client left; most endings are this.
- `shutdown` — a replica shut down gracefully; expect about one per stream on every pod a rollout replaces. Clients reconnect.
- `session_cancel` — the session was deleted. `session_absent` — the session vanished from the backing store without being deleted; clients retry, and backups restore it when configured.
- `slow_reader_deadline`, `inner_watchdog` — the client (or the proxy in front of the service) did not keep up and the subscription was ended; the client reconnects.
- `reader_capacity` — on a Valkey Cluster, no stream reader was free for the layer. `catchup_capacity` — no catch-up slot or read permit freed in time. Both are retryable, and normally come before any content.
- `fenced_read_wait_cap` — a catch-up of a very large layer gave up waiting for its read; expected zero.
- `stream_gap` — the subscription fell behind the layer's retained updates, or the layer was restored; the client reconnects with a full catch-up.
- `error` — any other failure; the `RPC failed` line's `status_message` says which.

**Watch for:** `stage="enqueued"` p99 above a few seconds — the service or the backing store is slow; see {ref}`Edits propagate slowly <live-edit-monitoring-slow-propagation>`.
**Watch for:** a rising share of `slow_reader_deadline` or `inner_watchdog` — clients cannot keep up with the edit rate of the layers they watch.
**Watch for:** `session_absent` above zero outside a planned restore — the backing store lost data.

Diagnostic metrics for delivery:

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `live_edit_subscriber_send_stalls_total` | Counter | `phase`: `catchup`, `live` | Sends that waited on HTTP/2 flow control. Normal during `catchup`; rising `live` stalls mean the client or proxy is not reading. |
| `live_edit_subscriber_send_wait_seconds` | Histogram | `phase`: `catchup`, `live` | How long stalled sends waited. A send that waits 5 s ends the subscription as `slow_reader_deadline`. |
| `live_edit_subscriber_backlog_entries` | Histogram | none | Edits waiting in a subscription's buffer. At 4,096 sustained for 5 s the subscription ends as `inner_watchdog`. |
| `live_edit_subscriber_backlog_bytes` | Histogram | none | Bytes waiting in a subscription's buffer. At 8 MiB sustained for 5 s the subscription ends as `inner_watchdog`. |
| `live_edit_stream_mux_readers_connected` | Gauge | none | Stream-reader connections to the backing store open on this replica, at most `valkey.subscriberPoolSize`. |
| `live_edit_stream_mux_reader_connects_total` | Counter | `cause`: `bind`, `retarget`; `result`: `ok`, `failed` | Stream-reader connection attempts. `failed` means subscriptions on this replica cannot receive updates. |
| `live_edit_stream_mux_reauths_total` | Counter | `result`: `ok`, `failed` | Stream-reader re-authentication after a credential rotation. `failed` means readers will drop at the old token's expiry. |
| `live_edit_stream_mux_bind_refusals_total` | Counter | none | On a Valkey Cluster, subscriptions refused because every stream reader was busy with other layers; raise `valkey.subscriberPoolSize` or add replicas. |
| `live_edit_valkey_consumer_forced_reconnects_total` | Counter | `phase`: `xread`, `wake_ack`, `probe`, `keepalive`, `permanent` | Stream-reader connections the service rebuilt after they went silent. Occasional ticks are self-healing. |
| `live_edit_valkey_consumer_permanent_exits_total` | Counter | `phase`: `xread`, `probe` | Stream-reader failures the service could not heal; subscriptions ended and clients reconnect. |

(live-edit-metrics-reference-capacity)=
## Is it saturated?

These measure each replica's use of its subscription, catch-up, upload and memory limits. Emitted always. Limits are per replica, so read these per replica.

| Metric | Type | Unit | Labels | Description |
|---|---|---|---|---|
| `live_edit_subscriber_admitted` | Gauge | subscriptions | none | Live subscriptions on this replica; compare with `live_edit_subscriber_max_subscriptions`. |
| `live_edit_subscriber_max_subscriptions` | Gauge | subscriptions | none | This replica's subscription ceiling, set at startup from its memory limit (4096 at `5Gi`). |
| `live_edit_subscriber_admission_total` | Counter | subscribe calls | `result`: `admitted`, `refused_saturated`, `refused_waiter_cap`, `cancelled` | Admission decisions for `SubscribeReadLayer`. |
| `live_edit_subscribe_catchups_waiting` | Gauge | catch-ups | none | Catch-ups waiting for one of the replica's catch-up slots. |
| `live_edit_upload_admission_total` | Counter | uploads | `rpc`: `create_layer`, `update_layer`; `result`: `admitted`, `refused_saturated`, `refused_waiter_cap`, `refused_growth`, `cancelled` | Admission decisions for uploads against the upload memory budget. |
| `live_edit_upload_reserved_bytes` | Gauge | bytes | none | Upload memory in use within the budget. |
| `live_edit_upload_budget_bytes` | Gauge | bytes | none | The upload memory budget, set at startup from the memory limit (384 MiB at `5Gi`). |
| `live_edit_catchup_build_budget_bytes` | Gauge | bytes | none | Memory for catching up subscribers to very large layers, set at startup from the memory limit (744 MiB at `5Gi`). |
| `live_edit_structural_cache_budget_bytes` | Gauge | bytes | `service` | Most memory the structural-edit cache holds, its indexes included (about 310 MiB at `5Gi`). Unlike the two budgets above, it already includes that overhead. |
| `live_edit_retirement_budget_bytes` | Gauge | bytes | none | The clean-up memory allowance for deleted, replaced or abandoned layers, set at startup from the memory limit (10 MiB at `5Gi`). |
| `live_edit_retirement_budget_reserved_bytes` | Gauge | bytes | none | Memory in use within the clean-up allowance; one clean-up of a layer with thousands of large values can take all of it. |
| `live_edit_retirement_wait_seconds` | Histogram | seconds | `outcome`: `immediate`, `waited`, `cancelled` | How long each clean-up of a deleted, replaced or abandoned layer waited for the allowance. `cancelled`: it gave up waiting, at a deadline or shutdown. |
| `live_edit_pod_memory_limit_bytes` | Gauge | bytes | `source`: `cgroup_v2`, `cgroup_v1`, `host_memory`, `reference`; `service` | The memory limit this replica sized its budgets from; the `source` in use carries it, the others read 0. |
| `live_edit_upload_processing_wait_seconds` | Histogram | seconds | `rpc`: `create_layer`, `update_layer` | How long received uploads waited for memory before processing. Waits near the RPC deadline end in `DEADLINE_EXCEEDED`. |
| `process_resident_memory_bytes` | Gauge | bytes | none | The process's resident memory, as the kernel reports it. Present from the first scrape. |

`refused_saturated` means the limit stayed full for the whole wait; `refused_waiter_cap` means the wait queue was already full, which a sharp burst can cause before `refused_saturated` appears.
`refused_growth` is an upload whose content arrived while the budget had no room for it.
`cancelled` is a call that went away (shutdown, a deleted or missing session, a disconnected client, or an upload's deadline); it is not a capacity signal.
All refusals reach the client as a retryable `UNAVAILABLE`.
`source` is where the limit came from: `cgroup_v2` or `cgroup_v1`, the container's memory limit; `host_memory`, the node's memory, smaller than that limit; `reference`, no readable limit, so the replica sized itself as a 5 GiB pod.
The five budget gauges are set once at startup and change only when the pod restarts with another limit; compare usage with them rather than with fixed figures. Each carries the same labels as the usage gauge it is read against, so the two divide directly.

**Watch for:** `live_edit_pod_memory_limit_bytes{source="reference"}` above zero — that replica is not sized from its real memory limit; fix what its startup warning's `reason` names ({ref}`Startup and readiness <live-edit-troubleshooting-startup>`, `LiveEditPodMemoryLimitUnread` in {ref}`Additional alert rules <live-edit-monitoring-additional-alerts>`).
**Watch for:** any `refused_*` subscription or upload result on one replica for more than a few minutes — that replica is full; see {ref}`Capacity refusals <live-edit-monitoring-capacity-refusals>`.
**Watch for:** `live_edit_subscribe_catchups_waiting` above zero for 10 minutes on one replica — joins are queueing; add replicas.
**Watch for:** `live_edit_upload_processing_wait_seconds` p99 above a third of `rpc.deadlineMs` — uploads are close to failing.
**Watch for:** `live_edit_retirement_wait_seconds{outcome="waited"}` p99 of several seconds — deletes, replacements and background clean-ups queue for clean-up memory, and deletes can reach their deadline ({ref}`Limitations <live-edit-scaling-limitations>`). A larger `resources.limits.memory` raises the allowance.

(live-edit-metrics-reference-capacity-diagnostics)=
## Capacity diagnostics

Read these while investigating a capacity alert or slow uploads. Emitted always.

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `live_edit_subscriber_admission_wait_seconds` | Histogram | none | Wait for a subscription slot. A p99 rising off zero before any refusal is the early warning. |
| `live_edit_subscribe_catchups_inflight` | Gauge | none | Catch-ups running, at most `valkey.subscriberMaxConcurrentCatchups`. |
| `live_edit_subscribe_catchups_pending` | Gauge | none | New subscriptions still waiting for their first catch-up; the autoscaler adds replicas on it. |
| `live_edit_subscribe_catchup_wait_seconds` | Histogram | `result`: `acquired`, `cancelled`, `timed_out` | Wait for a catch-up slot; `timed_out` ended as `catchup_capacity`. |
| `live_edit_upload_admission_wait_seconds` | Histogram | `rpc`: `create_layer`, `update_layer` | Wait for upload admission; rises before refusals start. |
| `live_edit_upload_admission_waiters` | Gauge | none | Uploads queued for admission now. A queue that stays non-empty can delay other calls on the same connection. |
| `live_edit_uploads_in_flight` | Gauge | `phase`: `receiving`, `processing` | Admitted uploads. Many `receiving` with little throughput means slow senders hold the budget. |
| `live_edit_upload_overflow_bytes` | Gauge | none | Memory of the one upload running beyond the budget; the pod limit must cover it. |
| `live_edit_upload_overflow_waiters` | Gauge | none | Received uploads waiting for room in the budget or for their turn to run beyond it. |
| `live_edit_upload_overflow_runs_total` | Counter | `rpc`: `create_layer`, `update_layer` | Uploads too large for the budget's free room, which ran beyond it. A steady rate makes such uploads queue. |
| `live_edit_upload_object_state_bytes` | Histogram | `path`: `structural`, `structural_full`, `reset`, `watch` | Memory structural edits hold for the layer they edit; grows with the layer. |
| `live_edit_update_layer_chain_wait_seconds` | Histogram | `outcome`: `ready`, `replaced`, `refused`, `timeout`, `unavailable`, `shutdown`, `cancelled` | How long chained `UpdateLayer` calls waited for their previous write. A steady `timeout` rate means clients chain calls onto writes that never committed. |
| `live_edit_update_layer_chain_waiters` | Gauge | none | Chained `UpdateLayer` calls waiting now. |
| `live_edit_valkey_pool_waiters` | Gauge | `pool`: `txn`, `subscriber_pool` | Calls waiting for a transaction connection (`txn`) or subscribe calls waiting for a slot (`subscriber_pool`). |
| `live_edit_valkey_pool_in_use` | Gauge | `pool`: `txn`, `subscriber_pool`, `maintenance`, `snapshot` | Connections in use per pool. `maintenance` can stay above `0` for up to 5 minutes while large-value backups are checked or repaired. |
| `live_edit_valkey_pool_checkout_seconds` | Histogram | `pool`: `txn` | Wait for a transaction connection. |
| `live_edit_jemalloc_allocated_bytes` | Gauge | `service` | Memory the service holds allocated. Growing across load cycles suggests a leak; report it. |
| `live_edit_jemalloc_resident_bytes` | Gauge | `service` | Upper bound of memory the allocator holds, including freed memory not yet returned. Judge memory return by `process_resident_memory_bytes` instead. |
| `live_edit_jemalloc_dirty_bytes` | Gauge | `arenas`: `automatic`, `fanout`; `service` | Memory freed but not yet returned to the system: `fanout` for the layer-read threads, `automatic` for everything else. |
| `live_edit_retirement_budget_overcommit_bytes` | Gauge | none | Clean-up memory used beyond the reservations, when a layer held more prims or large values than its clean-up estimated; returns to zero as clean-ups finish. |
| `live_edit_structural_cache_weight_bytes` | Gauge | `service` | Structural-edit cache memory, its indexes included, up to `live_edit_structural_cache_budget_bytes`; a layer stays cached 10 minutes after its last structural edit. |
| `live_edit_update_layer_admission_entries` | Gauge | `service` | Live layers this replica has written; falls only when layers are deleted or replaced, so it stays flat after load. |

The allocator gauges are absent on builds for anything other than glibc-based Linux; the shipped image is glibc-based.

(live-edit-metrics-reference-backing-store)=
## Is the backing store healthy?

These measure the service's view of its Valkey backing store. Emitted always. Each refusal also logs a line naming the layer and, where one exists, the key.

| Metric | Type | Unit | Labels | Description |
|---|---|---|---|---|
| `live_edit_store_memory_used_ratio` | Gauge | ratio | none | How full the backing store is: 0 to 1, above 1 past its limit. Each pod refreshes its own reading about once a minute. |
| `live_edit_store_write_refused_total` | Counter | writes | `stage`: `ext_store`, `update_commit`, `create_commit`; `reason`: `out_of_memory`, `misconf`, `readonly`, `noreplicas` | Writes of uploads (and of layers restored from backup) the store refused before applying any of them. The client gets a retryable `UNAVAILABLE`. |
| `live_edit_gc_oom_collections_total` | Counter | clean-ups | `outcome`: `freed`, `freed_nothing`, `nothing_to_collect`, `failed`, `refused` | Clean-ups the service ran after the store refused a write for lack of memory, deleting released large values at once. |
| `live_edit_gc_oom_freed_bytes_total` | Counter | bytes | none | Bytes of released large values those clean-ups deleted. |
| `live_edit_snapshot_read_refused_total` | Counter | reads | `kind`: `protocol`, `parse`, `auth`, `config`, `tls`, `invalid_command`, `invalid_argument`, `routing`, `not_found`, `backend_unavailable`, `backend_out_of_memory`, `unknown` | Reads of layer state the store refused permanently. Everything that needs that layer's state fails together. |
| `live_edit_keyspace_type_error_total` | Counter | writes | `backend`: `lua`, `txn`; `site`: `update`, `create`; `mode`: `fast`, `structural` | Writes refused because a key held a value the service did not write. |
| `live_edit_snapshot_read_type_error_total` | Counter | reads | `key_family`: `header`, `sections_index`, `deleted_sections_index`, `section`, `deleted_section`, `children`, `children_index`, `fields_index`, `time_samples_index`, `field`, `time_sample`, `ext_index` | Reads of layer state refused because a key held a value the service did not write. |
| `live_edit_stream_id_collision_total` | Counter | writes | `backend`, `site`, `mode`, as above | Writes refused because a layer's update stream holds entries the service did not write. |
| `live_edit_object_id_counter_repair_total` | Counter | repairs | `outcome`: `repaired_seeded`, `repaired_noop`, `repaired_initialized`, `unhooked_initialized`, `already_repaired`, `failed`, `failed_after_repair`, `wait_timeout` | `CreateLayer` calls that found the store's layer-id counter missing and repaired it first. |
| `live_edit_valkey_errors_total` | Counter | errors | `pool`: `subscriber`, `data`, `txn`, `subscriber_pool`, `maintenance`, `snapshot`; `kind`: `command_timeout` or the client library's connection-error name | Commands that timed out, and connection errors, by connection pool. The `command_timeout` cells start at `0`; each connection-error kind appears on first occurrence. |
| `live_edit_health_probe_failures_total` | Counter | probes | `backend`: `valkey` | Readiness probes of the backing store that failed; each takes the replica out of rotation until the next success. |
| `live_edit_valkey_txn_repair_ineffective_total` | Counter | verdicts | `reason`: `leaked_multi`, `unexpected_reply` | Connection checks the service stopped acting on, because rebuilding the connection did not change the reply. |

`live_edit_store_memory_used_ratio` is absent, never `0`, while the service cannot compute it.
On Azure Managed Redis it needs `valkey.memoryBytes` set, reflects the cache as a whole, and appears about a minute after a pod starts.

The labels in operator terms:

- `reason` on store write refusals:
  - `out_of_memory` — the store is full. If released large values filled it, the service frees them within seconds; otherwise it stays full until you act.
  - `misconf` — the store cannot persist, usually because its disk is full.
  - `readonly` — the write reached a replica, as during a failover.
  - `noreplicas` — too few replicas are reachable.
- `outcome` on full-store clean-ups:
  - `freed` — memory was freed, but the store is still too small.
  - `refused` — the store did not let the service free memory while full.
  - `nothing_to_collect`, or `freed_nothing` whose `WARN` line shows `oom_skipped=0` — nothing to free. A store that keeps refusing is full of data in use.
  - `freed_nothing` whose `WARN` line shows `oom_skipped` above 0 — memory is still owed and will be freed. Not a sizing signal.
  - `failed` — a clean-up error before it freed anything.
- `kind` on read refusals: `auth`, `tls`, `config` and `routing` are deployment or store faults; `backend_unavailable` covers the conditions above that clear by themselves; `backend_out_of_memory` is a full store; the rest are service faults to report.
- `outcome` on counter repairs:
  - `repaired_seeded`, `repaired_noop`, `failed` and `failed_after_repair` — the store lost its data under running replicas.
  - `repaired_initialized` and `unhooked_initialized` — expected once on a new store.
  - `wait_timeout` — an upload stopped waiting for a slow repair.
  - `already_repaired` — bookkeeping.
- `backend`, `site` and `mode` say which write path met the bad key; the remedy does not depend on them.

**Watch for:** `max(max_over_time(live_edit_store_memory_used_ratio[5m]))` above `0.7` — plan to grow the store; above `0.85` — grow it now ({ref}`Store memory <live-edit-monitoring-store-memory>`).
**Watch for:** `reason="out_of_memory"` or `kind="backend_out_of_memory"` — page; see {ref}`Backing store full <live-edit-monitoring-store-full>`.
**Watch for:** `outcome="refused"` — the service cannot free memory on this store while it is full; add memory ({ref}`Backing store full <live-edit-monitoring-store-full>`).
**Watch for:** any increase on the type-error and collision counters — another writer is using the service's keys; see {ref}`Damaged layer keys <live-edit-monitoring-layer-keys>`.
**Watch for:** the four keyspace-loss outcomes of `live_edit_object_id_counter_repair_total` — see {ref}`Keyspace lost <live-edit-monitoring-keyspace-lost>`.
**Watch for:** `kind="command_timeout"` sustained — the store, or the network to it, is slow; check its CPU and latency.

(live-edit-metrics-reference-backing-store-diagnostics)=
## Backing-store diagnostics

Read these while investigating a backing-store alert or slow edits. Emitted always.

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `live_edit_valkey_reconnects_total` | Counter | `pool`, as on `live_edit_valkey_errors_total` | Connection re-establishments. Timeouts alone mean a slow store; with reconnects, dropped connections; on `pool="txn"` a slow store can cause both. Slot moves add a few. |
| `live_edit_valkey_unresponsive_total` | Counter | `pool`; `verdict`: `fast`, `slow` | Silent connections the service closed. `fast`: one dead connection; `slow`: the store or a node stopped answering, or a wait passed 45 s. |
| `live_edit_valkey_member_replacements_total` | Counter | `pool`: `data`, `txn`, `maintenance`, `snapshot`; `outcome`: `replaced`, `connect_failed`, `budget_exhausted` | Pool clients replaced after repeatedly going silent. Anything but an occasional `replaced` points at the network path or a store node. |
| `live_edit_valkey_router_restarts_total` | Counter | `pool`, as above | Internal connection handlers restarted after ending unexpectedly. Expected zero; report the `ERROR` line. |
| `live_edit_valkey_txn_dirty_checkouts_total` | Counter | `reason`: `leaked_multi`, `unexpected_reply`, `probe_timeout`, `probe_error`, `topology_reply` | Transaction connections found unfit when checked out. `topology_reply` lasting beyond a cluster change points at a proxy rewriting replies. |
| `live_edit_snapshot_read_duration_seconds` | Histogram | `consumer`: `catchup`, `catchup_legacy_per_commit`, `checkpoint`, `structural_update`, `ext_gc`, `structural_cache`, `structural_cache_audit`; `path`: `atomic`, `fenced`, `unfenced` | Time to read a layer's state, by reader; `ext_gc` reads only a deleted or replaced layer's large-value references. Slow reads explain slow catch-ups and backups. |
| `live_edit_update_layer_commit_duration_seconds` | Histogram | `path`: `fast`, `watch`, `structural`, `structural_full`, `reset`; `outcome`: `ok`, `error` | Duration of each `UpdateLayer` call from arrival to reply, including its wait for upload memory. |
| `live_edit_update_layer_retries_total` | Counter | `path`, as above; `outcome`: `committed`, `exhausted` | Retry attempts edits needed because of concurrent writers; `exhausted` is attempts of edits that failed `UNAVAILABLE`. |
| `live_edit_update_layer_conflicts_total` | Counter | `path`, as above; `kind`: `watch_conflict`, `watch_read_moved`, `seq_fence`, `cache_diverged`, `snapshot_fence_wait` | Commit conflicts by cause; high rates mean many writers on one layer. |
| `live_edit_update_layer_attempts_histogram` | Histogram | `path`, as above | Attempts per committed edit. |
| `live_edit_update_layer_relay_bytes` | Histogram | none | Size of each edit's update; sizes the store's update retention. |
| `live_edit_update_layer_fence_timeout_total` | Counter | none | Edits that gave up waiting behind a whole-layer read of a very large layer. |
| `live_edit_operation_id_match_total` | Counter | `rpc`: `update_layer`, `create_layer`; `outcome`: `already_committed`, `conflict` | Re-sent writes recognised by their operation id. `conflict` means a client reused an id for different content: a client bug. |
| `live_edit_gc_sweep_batch_errors_total` | Counter | `kind`: the error's kind | Failures in the background clean-up passes; rises with store errors. Appears on first occurrence. |

(live-edit-metrics-reference-external-values)=
## Are large values handled correctly?

These measure large values (external values) carried by uploads: their content check, their release budget, and their reference counts. Emitted always; the release gauges appear after the first clean-up pass, the deficit gauge and `live_edit_ext_meta_total` after the first reconciler pass.

| Metric | Type | Unit | Labels | Description |
|---|---|---|---|---|
| `live_edit_ext_value_verify_total` | Counter | values | `result`: `ok`, `size_mismatch`, `hash_mismatch` | Large values checked against their declared size and hash. |
| `live_edit_ext_release_pending_bytes` | Gauge | bytes | `service` | Bytes of released large values waiting for clean-up; fleet-wide. |
| `live_edit_ext_release_budget_bytes` | Gauge | bytes | `service`; `source`: `override`, `info`, `fallback` | The release budget: uploads carrying large values wait while pending bytes are at or above it. |
| `live_edit_ext_release_gate_total` | Counter | uploads | `outcome`: `passed_after_wait`, `refused`, `read_failed` | Uploads that met a full release budget. `refused` failed `UNAVAILABLE`. |
| `live_edit_gc_underflow_total` | Counter | events | none | Reference counts that went below zero. Expected zero. |
| `live_edit_ext_ref_deficit_total` | Gauge | references | `service` | References missing from stored values' counts, confirmed by the reconciler; fleet-wide, expected zero. A standing shortfall shows every other pass: read it with `max_over_time`. |

`ok` must climb whenever clients upload large values; three flat cells mean no such uploads arrived.
Under `layer.extValueVerify: enforce` (default) each mismatch is a refused upload, almost always a client computing the hash differently.
Under `observe` each mismatch is a value stored unchecked.
`source` on the budget says where it came from: `override` (set explicitly, or derived from `valkey.memoryBytes`), `info` (derived from the store's reported `maxmemory`) or `fallback` (neither is available).

**Watch for:** mismatches under `observe` — before switching back to `enforce`, `ok` should climb while both mismatch cells stay flat.
**Watch for:** pending bytes above 80 % of the budget — clean-up is behind and uploads may soon wait.
**Watch for:** `live_edit_gc_underflow_total` above zero, or `live_edit_ext_ref_deficit_total` above zero at any point in 6 hours — see {ref}`External-value integrity <live-edit-monitoring-ext-integrity>`.

Diagnostic metrics for large values:

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `live_edit_ext_release_oldest_age_seconds` | Gauge | `service` | Age of the oldest released value waiting; above 60 s means clean-up is falling behind. |
| `live_edit_ext_release_pending_count` | Gauge | `service` | Released values waiting for clean-up. |
| `live_edit_ext_read_holds` | Gauge | `service` | Catch-ups currently holding clean-up off; long holds explain a growing backlog. |
| `live_edit_ext_meta_total` | Gauge | `service` | Distinct large values stored; a sizing input. |
| `live_edit_catchup_ext_value_miss_total` | Counter | `outcome`: `stale_retried`, `genuine`; `read`: `atomic`, `shared_fenced` | Catch-ups that found a large value missing. `genuine` means a value was gone. |
| `live_edit_gc_decrement_total` | Counter | `outcome`: `applied`, `replayed`, `stale` | Reference releases applied. A sustained `stale` rate means a stalled store shard or a clock running ahead. |
| `live_edit_gc_older_build_total` | Counter | `event`: `lock_taken_over`, `claim_dropped`, `release_stamped`, `carrier_dropped` | Clean-up work by a replica of an older release on the same store. Expected only during upgrades and rollbacks. |
| `live_edit_ext_reconcile_total` | Counter | `outcome`: `candidate_registered`, `repaired`, `healed`, `deferred`, `skipped_recent_store`, `skipped_unreadable_stamp`, `enumeration_failed`, `deficit_observed`, `deficit_cleared`, `skipped_index_drift` | Reference-count reconciler work, by outcome; the outcomes to act on follow the table. |
| `live_edit_time_sample_signed_zero_merge_total` | Counter | `outcome`: `merged`, `rekeyed`, `changed`, `fenced`, `not_live`, `undecodable`, `failed` | Time samples at time `-0.0` from an earlier release, folded into `0.0` by the reconciler; the outcomes to act on follow the table. |
| `live_edit_ext_retire_retry_backlog` | Gauge | `service` | Releases waiting to be retried after a store failure; should drain. |
| `live_edit_ext_ref_release_enqueue_failed_total` | Counter | none | Releases that could not be recorded; the storage is reclaimed later by the reconciler. |

Repeated `enumeration_failed` on `live_edit_ext_reconcile_total` means reconciler passes never finish: from store errors, a wrong-type key its `WARN` line names, or a store too large for one pass (raise `gc.extReconcileIntervalSecs`).
Its `deficit_observed`, or a sustained `skipped_unreadable_stamp`, is worth a support case.
A burst on `live_edit_time_sample_signed_zero_merge_total` after an upgrade is expected; `merged` or `rekeyed` recurring once every replica is upgraded, or recurring `undecodable` or `failed`, is worth a support case.

(live-edit-metrics-reference-backups)=
## Are backups current?

These measure backups (checkpoints) to the backup store; large values are under {ref}`Are large values backed up? <live-edit-metrics-reference-large-value-backups>`, restores under {ref}`Restore diagnostics <live-edit-metrics-reference-restores>`. Counters report `0` even with backups off; gauges and histograms appear only when `checkpoint.adapter` is not `Disabled`.

| Metric | Type | Unit | Labels | Description |
|---|---|---|---|---|
| `live_edit_checkpoint_writes_total` | Counter | writes | `result`: `success`, `error`, `skipped`; `kind`: `layer`, `layer_tail`, `session`, `counters_sidecar` | Backup writes by outcome. Every evaluated layer counts each pass, `skipped` when already current. |
| `live_edit_checkpoint_pending_age_seconds` | Histogram | seconds | none | How long layers with unsaved edits have waited for their backup; buckets from 5 s to 2 h. |
| `live_edit_checkpoint_oldest_pending_age_seconds` | Gauge | seconds | `service` | The longest any single layer has waited; fleet-wide. |
| `live_edit_checkpoint_layers_never_written` | Gauge | layers | `service` | Layers with no backup at all; fleet-wide. |
| `live_edit_checkpoint_config_interval_seconds` | Gauge | seconds | `service` | This replica's backup interval, `checkpoint.interval`; alert thresholds use it. |
| `live_edit_checkpoint_config_stuck_threshold_seconds` | Gauge | seconds | `service` | The age past which no configured delay explains a layer's wait; alert thresholds use it. |
| `live_edit_checkpoint_adapter_redirect_total` | Counter | requests | `verb`: `get`, `put`; `outcome` (`get` only `followed`, `target_failed`, `malformed`): `followed`, `completed`, `target_failed`, `complete_failed`, `multipart_reissued`, `malformed`, `preflight_redirect`, `preflight_unavailable`, `preflight_stale`; `service` | Backup-store requests redirected to blob storage for large files. |

With healthy backups the pending-age p99 sits a little above `checkpoint.interval`.
`verb="put"` redirects are normal for large backups; `target_failed` or `complete_failed` mean the pod cannot reach the blob-storage host.
`verb="get", outcome="followed"` means restores now depend on blob-storage access; the read succeeded, but check the backup store's configuration.

**Watch for:** pending-age p99 above five intervals, or no `kind="layer"` writes at all — see {ref}`Backups behind <live-edit-monitoring-backups>`.
**Watch for:** `layers_never_written` above zero for 10 minutes — those layers would be lost with the backing store.
**Watch for:** `result="error"` by `kind` — the backup store is failing or unreachable. `kind="counters_sidecar"` errors alone can also be expected or mean a deleted record — see {ref}`Backups behind <live-edit-monitoring-backups>`.

Diagnostic metrics for backups:

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `live_edit_checkpoint_lease_total` | Counter | `outcome`: `renewed`, `lost`, `refresh_error`; `service` | Backup-loop leadership checks. `lost` cut a pass short because it outlived its lease, usually from a slow backup store. |
| `live_edit_checkpoint_adapter_ceiling_elapsed_total` | Counter | `phase`: `call`, `enumeration`; `service` | Backup-store calls that ran to their time limit; expected zero. |
| `live_edit_checkpoint_counter_seed_secs` | Histogram | `phase`: `startup`, `repair` | Time to read the backup store's layer-id record at startup or during a repair. |
| `live_edit_checkpoint_reconcile_total` | Counter | `outcome`: `orphan_deferred`, `orphan_deleted`, `temp_deleted`, `dead_object_reaped`, `dead_deferred`, `dead_spared`, `ext_tombstoned`, `ext_collected`, `ext_spared`, `live_set_failed`; `service` | Backup-store clean-up work. `live_set_failed`: large-value clean-up stopped at a store error, a malformed or vanished file, or its time limit; its `WARN` says which. |
| `live_edit_checkpoint_ext_tombstone_backlog` | Gauge | `service` | Large values in the backup store awaiting deletion; fleet-wide. |
| `live_edit_checkpoint_dead_object_backlog` | Gauge | `service` | Backups of deleted layers awaiting deletion; fleet-wide. |
| `live_edit_checkpoint_stable_cut_total` | Counter | `outcome`: `verified`, `unverified`; `service` | Backups confirmed complete. `unverified` skipped a backup after a layer moved back, as after a restore. |
| `live_edit_checkpoint_tail_total` | Counter | `outcome`: `captured`, `captured_capped`, `rebased`, `refused_trimmed`, `refused_missing_bytes`, `refused_sentinel`, `refused_seq_gap`, `refused_no_base`, `refused_chain_full`, `refused_lineage_unrootable` | Incremental backups (`checkpoint.tail.enabled`). `refused_lineage_unrootable` means the backup store lost a full backup; repair the backup store. |
| `live_edit_checkpoint_base_bootstrap_total` | Counter | `outcome`: `persisted`, `shed`, `skipped_header_absent`, `skipped_generation_present`, `skipped_untrusted_cursor`, `unstamped_header_absent`, `unstamped_generation_present`, `unstamped_foreign_restore`, `unstamped_write_error`, `failed`; `site`: `creation_base`, `restore_rebase` | Extra full backups of new and restored layers. `shed` at `restore_rebase`: a restored layer lacks a fresh backup until its next pass. |

(live-edit-metrics-reference-large-value-backups)=
## Are large values backed up?

These measure the copies of large values to the backup store, and the repair that uploads the copies that were skipped ({ref}`Backups of large values <live-edit-durability-large-values>`). Counters report `0` even with backups off; the `live_edit_checkpoint_ext_mirror_*` gauges read `0` with backups off, and `live_edit_checkpoint_ext_backup_owed` appears only with `checkpoint.enabled: true` and an adapter other than `Disabled`.

| Metric | Type | Unit | Labels | Description |
|---|---|---|---|---|
| `live_edit_checkpoint_ext_backup_owed` | Gauge | values | `service` | Large values older than a few minutes with no confirmed backup copy, as of the last complete check; fleet-wide. |

One replica updates `live_edit_checkpoint_ext_backup_owed` every 5 minutes; a check that cannot finish leaves the last value in place.

**Watch for:** `live_edit_checkpoint_ext_backup_owed` above zero for an hour — large values are going without a backup; read `live_edit_checkpoint_ext_backup_repair_total` by `outcome` below.

Diagnostic metrics for large-value backups:

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `live_edit_checkpoint_ext_mirror_dropped_total` | Counter | `reason`: `queue_full`, `budget_full`, `oversized`, `shutdown` | Large values whose backup copy was skipped, repaired minutes later except `oversized`; the reasons follow the table. Every `reason` reads `0` from startup. |
| `live_edit_checkpoint_ext_mirror_errors_total` | Counter | none | Large values the backup store failed to accept; repaired minutes later. |
| `live_edit_checkpoint_ext_mirror_bytes` | Gauge | `state`: `queued`, `in_flight` | Memory held by large values waiting to be copied to the backup store, or being copied or repaired. |
| `live_edit_checkpoint_ext_mirror_budget_bytes` | Gauge | none | The most memory those values can hold, set at startup from the memory limit (128 MiB at `5Gi`). |
| `live_edit_checkpoint_ext_backup_repair_total` | Counter | `outcome`: `already_stored`, `resent`, `resend_failed`, `probe_failed`, `value_gone`, `budget_busy`, `oversized`, `confirmed_copy_missing` | Large values the backup repair checked or uploaded again; the outcomes follow the table. Every `outcome` reads `0` from startup. |
| `live_edit_checkpoint_ext_backup_mark_failed_total` | Counter | `op`: `confirm`, `clear` | Records of a backup copy the backing store failed to write; the repair checks those values again. Expected near zero. |

`reason` on `live_edit_checkpoint_ext_mirror_dropped_total`, each a large value left without a backup until the backup repair uploads it a few minutes later:

- `budget_full`: the values waiting for the backup store filled their memory, `live_edit_checkpoint_ext_mirror_budget_bytes`. Check the backup store's latency and `live_edit_checkpoint_ext_mirror_errors_total`; if the store is healthy, raise `checkpoint.extMirrorWorkers` (at most `16`) or `resources.limits.memory`.
- `queue_full`: `checkpoint.extMirrorQueueCapacity` values were already waiting. Raise it, or `checkpoint.extMirrorWorkers`. A slow backing store can cause it too: each copy is recorded there once uploaded.
- `oversized`: a restored value larger than the copy memory, possible only for one uploaded under a higher `layer.maxSize`. Neither the copy nor the repair uploads it until `layer.maxSize` is raised to at least its size (the `bytes_len` in its `WARN`), or `resources.limits.memory` raises `live_edit_checkpoint_ext_mirror_budget_bytes` above it.
- `shutdown`: values still waiting when a replica stopped. A few during a rollout under upload load are expected.

`outcome` on `live_edit_checkpoint_ext_backup_repair_total`:

- `already_stored`: the backup store already held the copy, so nothing was uploaded. It includes the daily re-check of every confirmed copy, and the one-time check of every large value after a restore or a change of backup location.
- `resent`: a missing copy was uploaded from the backing store.
- `resend_failed`: the value stays without a backup until a later check. A failed or timed-out upload stops uploads until the next check, 5 minutes later; a failed backing-store read affects that value alone.
- `probe_failed`: the backup store did not answer whether it held a copy; checks stop until the next one. Check the backup store's health.
- `budget_busy`: the copy memory stayed full for a minute, so uploads wait for the next check; act as for `budget_full`.
- `oversized`: as `oversized` above; the value stays without a backup and is counted again at every check.
- `value_gone`: the value was deleted before it could be uploaded; nothing to do.
- `confirmed_copy_missing`: a copy recorded as stored was missing from the backup store; it is uploaded again, by a later check if uploads stopped for this one. Check for something outside the service deleting it, such as a lifecycle rule on the backup prefix, and for a rising `live_edit_checkpoint_ext_backup_mark_failed_total{op="clear"}`.

(live-edit-metrics-reference-restores)=
## Restore diagnostics

Read these while a restore from backup runs, or after one ({ref}`After a disaster-recovery restore <live-edit-monitoring-after-restore>`). All are counters, reported as `0` even with backups off.

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `live_edit_checkpoint_ext_recovery_total` | Counter | `result`: `recovered`, `adapter_miss`, `adapter_transient`, `adapter_permanent`, `adapter_other` | Missing large values fetched back from the backup store. `adapter_miss` means the backup store lacks it too. |
| `live_edit_checkpoint_recovery_attempts_total` | Counter | `result`: `restored`, `not_recoverable`, `already_present`, `error`; `scope`: `layer`, `session`; `service` | Restores from backup, triggered when a request finds a layer or session missing. |
| `live_edit_checkpoint_recovery_stream_refused_total` | Counter | `service` | Restored layers whose leftover update stream was discarded. |
| `live_edit_checkpoint_recovery_stream_unresolved_total` | Counter | `reason`: `probe_unreadable`, `unlink_unverified`, `wrongtype_key`; `service` | Restores that could not settle a layer's leftover update stream; the reasons follow the table. |
| `live_edit_checkpoint_restore_lineage_total` | Counter | `outcome`: `gathered_base`, `gathered_composed`, `stepped_down_link_broken`, `stepped_down_link_mismatch`, `stepped_down_decode`, `stepped_down_missing_meta`, `stepped_down_budget`, `stepped_down_compose`, `exhausted`, `exhausted_budget`; `service` | Which backup a restore used. `stepped_down_*`: an older backup; `exhausted`: none usable (`NOT_FOUND`); `exhausted_budget`: the backup store was too slow (`UNAVAILABLE`, retry). |

`reason` on `live_edit_checkpoint_recovery_stream_unresolved_total`: `probe_unreadable` and `unlink_unverified` refuse the restore, so check the store and retry; `wrongtype_key` restores the layer but leaves another writer's key to remove ({ref}`Damaged layer keys <live-edit-monitoring-layer-keys>`).

(live-edit-metrics-reference-credentials)=
## Are credentials and auth services healthy?

These measure the token files the service reads and its calls to the permissions and userinfo services. Token metrics are emitted only when a token file is configured.

| Metric | Type | Unit | Labels | Description |
|---|---|---|---|---|
| `live_edit_service_token_available` | Gauge | 0 or 1 | none | `1` while the replica holds a usable service-identity token. Present only when `serviceIdentity.accessTokenFile.enabled` is `true`. |
| `live_edit_service_token_expires_in_seconds` | Gauge | seconds | `role`: `valkey`, `storage-api`, `unspecified` | Time until the cached token expires; negative once it has. Absent until a token is first loaded. |
| `live_edit_service_token_refresh_headroom_seconds` | Gauge | seconds | `role`: `valkey`, `storage-api`, `unspecified` | Life the old token had left when its replacement arrived; recorded once per rotation. |
| `live_edit_auth_errors_total` | Counter | calls | `backend`: `storage`, `userinfo`; `code`: `timeout`, `unreachable`, `invalid_response` | Failed calls to the permissions (`storage`) and userinfo services. |

`role="valkey"` is the backing-store credential; `role="storage-api"` is the backup-store credential; `unspecified` does not appear on a deployed replica.
On `role="valkey"`, headroom below 180 s means some connections are dropped and re-established at each rotation.
While the auth services fail, cached authorizations keep working and other calls fail `UNAVAILABLE`.

**Watch for:** `live_edit_service_token_expires_in_seconds` below zero — see {ref}`Credential expired <live-edit-monitoring-credential>`.
**Watch for:** a sustained `live_edit_auth_errors_total` rate — the permissions or userinfo service is unreachable or slow.

Diagnostic metrics for credentials and auth:

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `live_edit_service_token_parse_errors_total` | Counter | none | Token files the service could not parse; the `WARN` line's `role` and `path` say which. |
| `live_edit_auth_cache_hits_total` | Counter | `kind`: `permissions`, `userinfo` | Authorizations served from cache. |
| `live_edit_auth_cache_misses_total` | Counter | `kind`: `permissions`, `userinfo` | Authorizations that called the auth service; the load on it. |
| `live_edit_userinfo_fallbacks_total` | Counter | none | Callers the userinfo service did not know, named from their token instead. A rate close to the userinfo call rate means the wrong directory. |

(live-edit-metrics-reference-autoscaler)=
## Is the autoscaler working?

These measure the replica autoscaler's decisions. Emitted only in `autoscaling.mode: scaler`, on the scaler pod's port `9090`.

| Metric | Type | Unit | Labels | Description |
|---|---|---|---|---|
| `live_edit_scaler_decisions_total` | Counter | polls | `direction`: `up`, `down`, `none`; `reason`: `min`, `cpu`, `catchup`, `occupancy`, `below_target`, `at_target`, `scale_down_blocked`, `no_data` | One per poll whose cluster reads succeeded, by what it decided. |
| `live_edit_scaler_target_replicas` | Gauge | replicas | none | The replica count the scaler set or kept. |

`up` carries the signal that raised the count; `none` with `no_data` means no server pod answered the scaler.

**Watch for:** no decisions for 10 minutes, or only `no_data` — the replica count is frozen; see {ref}`Scaler not deciding <live-edit-monitoring-scaler>`.
**Watch for:** `live_edit_scaler_target_replicas` at `autoscaling.maxReplicas` while replicas refuse work — the fleet is full.

Diagnostic metrics for the autoscaler:

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `live_edit_scaler_desired_replicas` | Gauge | `term`: `cpu`, `catchup`, `occupancy`, `final` | The count each signal asked for; shows which signal drives the fleet. |
| `live_edit_scaler_pods` | Gauge | `state`: `reporting`, `not_reporting`, `unready`, `terminating` | Server pods by state; anything but `reporting` holds scale-down. |
| `live_edit_scaler_catchup_rate` | Gauge | `scope`: `fleet`, `pod`; `pod` | Catch-ups per second per replica the scaler plans with. |
| `live_edit_scaler_poll_errors_total` | Counter | `stage`: `endpoints`, `scale`, `snapshot`, `deletion_cost`, `patch_scale` | Failed cluster reads and writes. Sustained `endpoints` or `scale` errors mean the scaler's RBAC is missing a rule. |
| `live_edit_scaler_scale_down_blocked_total` | Counter | `reason`: `startup`, `window`, `unsettled`, `interval`, `feasibility` | Why a wanted scale-down was held. Each is deliberate; `unsettled` lasting 15 minutes points at a pod the scaler cannot account for. |

(live-edit-metrics-reference-tracing)=
## Is trace export working?

These measure OTLP trace export. Emitted only when `observability.tracing.enabled` is `true`; their presence is the sign that export is on.

| Metric | Type | Unit | Labels | Description |
|---|---|---|---|---|
| `live_edit_trace_spans_exported_total` | Counter | spans | none | Spans delivered to the collector. |
| `live_edit_trace_spans_dropped_total` | Counter | spans | `reason`: `queue_full`, `export_failed` | Spans lost before reaching the collector. |

`queue_full`: export did not keep up and the queue of `observability.tracing.batch.maxQueueSize` spans was full. `export_failed`: the collector refused or did not answer, or the TLS handshake with an `https://` collector failed, for example on an untrusted certificate.
A `trace spans dropped` `WARN` line, at most once a minute, carries the collector's last error.

**Watch for:** sustained drops — the trace view is missing spans; raise `observability.tracing.batch.maxQueueSize` or fix the collector. Clients are not affected.

(live-edit-metrics-reference-self-checks)=
## Self-checks that should read zero

Each of these counts an internal consistency check that fired.
On a healthy deployment every one stays at zero.
The service has already protected the data; any increase means: collect the logs of every replica and open a support case.

| Metric | Labels | Cell that should stay at zero |
|---|---|---|
| `live_edit_structural_cache_divergence_total` | none | all |
| `live_edit_structural_cache_audit_total` | `outcome`: `match`, `diverged`, `skipped` | `diverged` |
| `live_edit_update_layer_structural_fallbacks_total` | `reason`: `repeat_divergence`, `over_work_limit`, `reset_over_work_limit` | `repeat_divergence` |
| `live_edit_storage_watch_unwatch_failures_total` | `site`: `explicit`, `drop` | `drop` |
| `live_edit_storage_watch_voided_total` | `path`: `update_layer`, `create_session`, `restore_residue_wipe`, `restore_write_object`; `disposition`: `aborted_pre_exec`, `committed_unwatched`, `compensated`, `witness_unavailable`, `fingerprint_unavailable` | `committed_unwatched` |
| `live_edit_stream_mux_poison_keys_total` | none | all |
| `live_edit_ext_ref_undecodable_total` | `site`: `lua_token`, `row_hash`, `ext_index_field`, `retirement_index_field` | all |
| `live_edit_ext_index_drift_total` | `site`: `reconcile`, `torn_commit`, `structural_settle`; `direction`: `under`, `over` | `under` |

The `LiveEditSelfCheckFired` alert in {ref}`Internal-check alerts <live-edit-monitoring-internal-alerts>` covers all eight.
An upgrade from, or rollback to, a release that stores time samples at `-0.0` as written can raise some of them with no support case needed: {ref}`Upgrading from such a release <live-edit-operations-upgrade-time-zero>`.

(live-edit-metrics-reference-other)=
## Metrics not listed here

The service exports further `live_edit_*` and `live_*` series for support engineers; their meaning can change between releases, so do not build alerts or dashboards on them.

(live-edit-metrics-reference-see-next)=
## See next

{ref}`Troubleshooting <live-edit-troubleshooting>` and {ref}`Troubleshooting the backing store, capacity and rollouts <live-edit-troubleshooting-capacity>` cover symptoms and their fixes; {ref}`Operations <live-edit-operations>` covers upgrades and rollbacks.
