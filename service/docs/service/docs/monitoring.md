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

(live-edit-monitoring)=
# Live Edit monitoring

This page covers how to observe a running Live Edit deployment: health checks, logs, metrics, traces, and the alerts to set up.
It assumes the service is installed as described in {ref}`Deployment <live-edit-deployment>`.
Every metric named here is described in the {ref}`Metrics reference <live-edit-metrics-reference>`.

(live-edit-monitoring-health)=
## Health checks

Each replica serves the standard gRPC health-checking protocol on its gRPC port (`service.grpc.port`, default `50001`).

| Health service name | Reports |
|---|---|
| `""` (empty) | The process is alive and answering. The chart's liveness probe uses it. |
| `nvidia.omniverse.live.v1alpha.LiveService` | The replica is ready for traffic. The chart's startup and readiness probes use it. |

A replica reports `NOT_SERVING` on the readiness name until startup finishes, and again whenever its backing store stops answering.
When a service-identity token file is configured, readiness also waits for the first token.
When backups are configured, readiness also waits for the startup check of the backup store's counters record.
Readiness does not depend on the permissions or userinfo services, so an outage there does not take every replica out of rotation.

A ready replica or a successful `Ping` does not prove that users can open their base assets or that backups are being written.
Use the alerts below for those.

(live-edit-monitoring-logs)=
## Logs

The service writes logs to standard output.
Set the level with `observability.logging.level` (default `info`).
Set `observability.logging.structured.enabled: true` for one JSON object per line, which log pipelines can parse; the chart default is human-readable text.

Each failed RPC logs exactly one line:

- `RPC failed` at `INFO` for client-caused failures, and at `WARN` for authorization denials and server-side failures: capacity refusals, backing-store failures, and `INTERNAL`, `UNKNOWN` or `DATA_LOSS` codes without a recorded cause.
- `internal error` at `ERROR` for a server-side fault with a recorded cause, in its `error` field.
- `RPC deadline exceeded` at `WARN` when the server deadline ran out, with `endpoint`, `deadline_ms` and `elapsed_ms`.
- `RPC abandoned before completion` at `INFO` when the client cancelled or its own timeout expired first, with `elapsed_ms` and `client_timeout_ms`.

The first two lines carry `endpoint`, `grpc_code`, `status_message` and `stage`, plus the `session_id`, `layer_name`, `client_id` and `base_path` the request named.
The `grpc_code` and `endpoint` values match the labels of `live_endpoint_calls_total`, so a spike on a dashboard leads straight to its log lines.
Clients receive only a generic message for a server-side fault, so these lines are the place to find its cause.
The service never logs bearer tokens or other credentials.

(live-edit-monitoring-metrics)=
## Collecting metrics

Each replica serves Prometheus metrics at `/metrics` on port `9090` (`service.metrics.port`) from the moment the process starts.

- **Prometheus Operator.** Set `serviceMonitor.enabled: true` and add the labels your Prometheus selects on to `serviceMonitor.labels`.
  The ServiceMonitor scrapes every 30 s by default (`serviceMonitor.interval`).
  Install the Prometheus Operator's CRDs first, or the install fails.
- **Replica autoscaler.** In the default `autoscaling.mode: scaler`, the scaler pod serves its own `live_edit_scaler_*` metrics on port `9090`.
  It has no Service, so the ServiceMonitor does not reach it: scrape the pod, for example with a PodMonitor or with annotations set through `autoscaling.scaler.podAnnotations`.
- **OTLP push.** To push the same metrics to an OpenTelemetry collector as well, set `observability.metrics.enabled: true` and `observability.metrics.endpoint.grpc`, an `http://` or `https://` URL ({ref}`Configuration <live-edit-configuration-observability>`).
  The `/metrics` endpoint stays on; names, labels and buckets are the same on both paths.
  After a push fails, a `WARN` line `OTLP metrics export failed` appears at most once a minute, counting the failures since the previous line, with the collector's last error in `last_export_error`.

```yaml
serviceMonitor:
  enabled: true
  labels:
    release: {PROMETHEUS_RELEASE}
observability:
  metrics:
    enabled: true
    endpoint:
      grpc: "http://{COLLECTOR_HOST}:4317"
```

To verify, port-forward one pod and read a series:

```bash
kubectl -n {NAMESPACE} port-forward pod/{POD_NAME} 9090:9090 &
curl -s localhost:9090/metrics | grep '^live_endpoint_calls_total' | head -3
```

Success is three `live_endpoint_calls_total{…}` lines with a value of `0` or more.

Most counters start at `0`, many metrics have no `service` label, and fleet-wide gauges must be aggregated with `max()`: read {ref}`Reading the reference <live-edit-metrics-reference-reading>` before building dashboards.
The examples on these pages group by your scrape target's `instance` label.

(live-edit-monitoring-traces)=
## Traces

The service can export one span per `LiveService` RPC over OTLP/gRPC, to an `http://` or `https://` collector ({ref}`Configuration <live-edit-configuration-observability>`).

```yaml
observability:
  serviceName: ""          # defaults to the chart name
  tracing:
    enabled: true
    endpoint:
      grpc: "http://{COLLECTOR_HOST}:4317"
    samplingRatio: "1.0"
    filter: "live_edit=info"
```

- A span is named after its method, for example `nvidia.omniverse.live.v1alpha.LiveService/UpdateLayer`, and carries the call's gRPC code in `rpc.status_code`.
  The service's own `INFO`-and-above log lines from the call are attached as span events.
- A request carrying a W3C `traceparent` header joins the caller's trace and follows its sampling decision; other requests are sampled at `samplingRatio`.
- A `JoinSession` or `SubscribeReadLayer` span covers the whole stream and is exported only when the stream ends.
- Keep `filter` at its default except while diagnosing: library spans add hundreds of spans per edit.
- Spans wait in a queue of `observability.tracing.batch.maxQueueSize` spans (default 2048, about 1 KB of memory each); lost spans are counted on `live_edit_trace_spans_dropped_total`.

(live-edit-monitoring-requests)=
## Request rate, errors and latency

`live_endpoint_calls_total` counts every completed RPC by `endpoint` and `grpc_code`; `live_endpoint_duration_microseconds` records each one's duration.
Both are recorded once, when the call completes.
Calls to the health service, reflection and the capabilities service are counted under `endpoint="other"`; leave them out of service-level dashboards.

**Error codes.** Read the gRPC code to tell whose problem a failure is:

| Code | Meaning for an operator |
|---|---|
| `INVALID_ARGUMENT`, `NOT_FOUND`, `ALREADY_EXISTS`, `FAILED_PRECONDITION`, `OUT_OF_RANGE`, `RESOURCE_EXHAUSTED`, `PERMISSION_DENIED`, `UNAUTHENTICATED` | The request was wrong or not allowed. A sudden rise usually means a client change or a permissions change. |
| `ABORTED` | A chained `UpdateLayer` was refused because its previous write was not the client's latest. Nothing was applied. |
| `CANCELLED` | The client cancelled, disconnected, or its own deadline expired first. |
| `DEADLINE_EXCEEDED` | The server deadline ran out: `rpc.deadlineMs` (30 s) for most calls, `rpc.sessionListTimeoutMs` (2 s) for `ListSessions`. |
| `UNAVAILABLE` | Retryable: a capacity refusal, a backing store or auth service that is unreachable or refusing, or a replica shutting down. Clients normally retry. |
| `INTERNAL`, `UNKNOWN`, `DATA_LOSS` | A server-side fault. Each logs an `internal error` line at `ERROR`, or an `RPC failed` line at `WARN`. |

```promql
# Calls per second by endpoint and code
sum by (endpoint, grpc_code) (rate(live_endpoint_calls_total{endpoint!="other"}[5m]))

# p99 latency of single-response calls, in seconds
histogram_quantile(0.99, sum by (le, endpoint) (
  rate(live_endpoint_duration_microseconds_bucket{endpoint!~"JoinSession|SubscribeReadLayer|other"}[5m])
)) / 1e6
```

The duration histogram is in **microseconds**, with buckets from 100 µs to 30 s; the top bucket matches the default deadline.
A `CreateLayer` or `UpdateLayer` duration includes receiving the upload, so large uploads over slow links read as slow calls.

**Streaming RPCs.** `JoinSession` and `SubscribeReadLayer` are recorded when the stream ends, so their "duration" is how long the user stayed connected.
They normally end `CANCELLED` when the client leaves, and `UNAVAILABLE` on every pod a rolling restart replaces.
Leave them out of error-ratio and latency objectives, and use these instead:

| Question | Metric |
|---|---|
| How long does it take to open a layer or join a session? | `live_edit_join_latency_seconds`, by `endpoint`: from the request arriving until the user has the full state. Only successful joins are recorded. |
| How late do other users' edits arrive? | `live_edit_subscriber_delivery_lag_seconds{stage="enqueued"}`: an edit's age when it is handed to a subscriber's stream. |
| How many streams are open? | `live_edit_inflight_rpcs`, by `endpoint`; `live_active_connections` for session connections. |

```promql
# p95 time to open a layer or join a session
histogram_quantile(0.95, sum by (le, endpoint) (rate(live_edit_join_latency_seconds_bucket[5m])))

# p95 live delivery lag by stage
histogram_quantile(0.95, sum by (le, stage) (rate(live_edit_subscriber_delivery_lag_seconds_bucket[5m])))
```

(live-edit-monitoring-capacity)=
## Capacity and saturation

Live Edit's limits are per replica, and established streams do not move when replicas are added.
A fleet-wide average can look healthy while one replica refuses work, so build capacity panels per replica.
How the limits are sized and how the autoscaler responds is on {ref}`Scaling <live-edit-scaling>`.

| Resource | In use | Limit | Refusal signal |
|---|---|---|---|
| Live subscriptions | `live_edit_subscriber_admitted` | `live_edit_subscriber_max_subscriptions` (4096 at the `5Gi` limit) | `live_edit_subscriber_admission_total{result=~"refused_.*"}` |
| Catch-ups of new subscribers | `live_edit_subscribe_catchups_inflight`, `live_edit_subscribe_catchups_waiting` | `valkey.subscriberMaxConcurrentCatchups` (8) | `live_edit_subscriber_stream_exits_total{reason="catchup_capacity"}` |
| Upload memory | `live_edit_upload_reserved_bytes` | `live_edit_upload_budget_bytes` (384 MiB at the `5Gi` limit) | `live_edit_upload_admission_total{result=~"refused_.*"}` |
| Released large values awaiting clean-up | `live_edit_ext_release_pending_bytes` | `live_edit_ext_release_budget_bytes` | `live_edit_ext_release_gate_total{outcome="refused"}` |
| Pod memory | `process_resident_memory_bytes` | the container memory limit, `live_edit_pod_memory_limit_bytes` | the pod is OOM-killed |
| Backing-store memory | `live_edit_store_memory_used_ratio` | `1` (the store is full) | `live_edit_store_write_refused_total{reason="out_of_memory"}` |

A replica refusing subscriptions at its ceiling shows flat CPU and memory, so do not judge subscription capacity from pod utilisation.

Pod memory should return close to its post-start level within a few minutes of load ending, apart from the structural-edit cache.
That cache keeps up to `live_edit_structural_cache_budget_bytes` (about 310 MiB at the `5Gi` limit, its indexes included) for 10 minutes after a layer's last structural edit; `live_edit_structural_cache_weight_bytes` shows its size.
If `live_edit_jemalloc_allocated_bytes` keeps growing across load cycles beyond that, open a support case.

(live-edit-monitoring-store-memory)=
### Store memory

`live_edit_store_memory_used_ratio` is how full the backing store is ({ref}`metric details <live-edit-metrics-reference-backing-store>`); read it with `max()` across pods.
Above `0.7`, plan to grow the store; above `0.85`, grow it now, sizing it as {ref}`Memory <live-edit-backing-store-memory>` describes.
An alert on it is recommended, never required.
The series can be missing for a minute, so evaluate alerts over `max_over_time(live_edit_store_memory_used_ratio[5m])`, as the rules below do.

(live-edit-monitoring-recommended-alerts)=
## Recommended alerts

Alerts give early warning; none is required for the service to work, and the service runs normally without any of them.
If you set up only a few, start with these.

| Alert | Fires when | Severity, `for` | Runbook |
|---|---|---|---|
| `LiveEditBackingStoreMemoryWarning` | The backing store is more than 70 % full | warning, 15m | {ref}`Store memory <live-edit-monitoring-store-memory>` |
| `LiveEditBackingStoreMemoryHigh` | The backing store is more than 85 % full | page, 5m | {ref}`Store full <live-edit-monitoring-store-full>` |
| `LiveEditBackingStoreFull` | The backing store refused a write because it is out of memory | page, 0m | {ref}`Store full <live-edit-monitoring-store-full>` |
| `LiveEditServerErrors` | More than 1 % of an endpoint's calls end in a server fault | page, 10m | {ref}`Server errors <live-edit-monitoring-server-errors>` |
| `LiveEditSlowRequests` | p99 of calls other than uploads and streams is above 5 s | warning, 15m | `LiveEditBackingStoreTimeouts` in {ref}`Additional alert rules <live-edit-monitoring-additional-alerts>` |
| `LiveEditReplicasNotReady` | Fewer replicas are ready than requested (needs kube-state-metrics) | warning, 15m | {ref}`Probes <live-edit-operations-probes>` |
| `LiveEditBackupsBehind` | Backups lag more than five backup intervals; only when backups are on | page, 5m | {ref}`Backups behind <live-edit-monitoring-backups>` |
| `LiveEditCredentialExpired` | A cached service or backing-store token has expired; only with a token file | page, 1m | {ref}`Credential expired <live-edit-monitoring-credential>` |

The rules are in Prometheus rule-file format; for the Prometheus Operator, put the `groups` list under `spec` of a `PrometheusRule`.
The expressions group by `instance`; use `pod` if that is your scrape label.
`{DEPLOYMENT}` is the server Deployment's name, as `kubectl -n {NAMESPACE} get deployments` lists it.
The latency threshold is a starting point: set it from your own baseline after a week of traffic.

```yaml
groups:
- name: live-edit-recommended
  rules:
  - alert: LiveEditBackingStoreMemoryWarning
    expr: max(max_over_time(live_edit_store_memory_used_ratio[5m])) > 0.7
    for: 15m
    labels: {severity: warning}
  - alert: LiveEditBackingStoreMemoryHigh
    expr: max(max_over_time(live_edit_store_memory_used_ratio[5m])) > 0.85
    for: 5m
    labels: {severity: page}
  - alert: LiveEditBackingStoreFull
    expr: |
      sum by (instance) (increase(live_edit_store_write_refused_total{reason="out_of_memory"}[5m])) > 0
      or sum by (instance) (increase(live_edit_snapshot_read_refused_total{kind="backend_out_of_memory"}[15m])) > 0
    for: 0m
    labels: {severity: page}
  - alert: LiveEditServerErrors
    expr: |
      sum by (endpoint) (rate(live_endpoint_calls_total{endpoint!~"JoinSession|SubscribeReadLayer|other", grpc_code=~"INTERNAL|UNKNOWN|DATA_LOSS"}[5m]))
        / sum by (endpoint) (rate(live_endpoint_calls_total{endpoint!~"JoinSession|SubscribeReadLayer|other"}[5m])) > 0.01
    for: 10m
    labels: {severity: page}
  - alert: LiveEditSlowRequests
    expr: |
      histogram_quantile(0.99, sum by (le, endpoint) (rate(live_endpoint_duration_microseconds_bucket{endpoint!~"CreateLayer|UpdateLayer|JoinSession|SubscribeReadLayer|other"}[5m]))) > 5e6
    for: 15m
    labels: {severity: warning}
  - alert: LiveEditReplicasNotReady
    expr: |
      kube_deployment_status_replicas_available{namespace="{NAMESPACE}", deployment="{DEPLOYMENT}"}
        < kube_deployment_spec_replicas{namespace="{NAMESPACE}", deployment="{DEPLOYMENT}"}
    for: 15m
    labels: {severity: warning}
  - alert: LiveEditBackupsBehind
    expr: |
      histogram_quantile(0.99, sum by (le) (rate(live_edit_checkpoint_pending_age_seconds_bucket[5m])))
        > 5 * max(live_edit_checkpoint_config_interval_seconds)
    for: 5m
    labels: {severity: page}
  - alert: LiveEditCredentialExpired
    expr: min by (instance, role) (live_edit_service_token_expires_in_seconds) < 0
    for: 1m
    labels: {severity: page}
```

(live-edit-monitoring-runbooks-store)=
## Runbooks: errors and backing-store refusals

(live-edit-monitoring-server-errors)=
### Server errors

1. Search the logs for `internal error` lines (`ERROR`) and `RPC failed` lines (`WARN`) with the alert's `endpoint` and a server-fault `grpc_code`; the `error` or `status_message` field names the cause.
2. If the lines name the backing store, check the store alerts on this page first.
3. Otherwise collect the lines and the replica's logs around them, and open a support case.

(live-edit-monitoring-store-full)=
### Backing store full

The store is configured never to evict, so it refuses every write that needs memory.
Uploads fail `UNAVAILABLE` with nothing applied.
On the first refused write, the service deletes the released large values the store holds, whatever their age.
If they filled the store, refusals stop within seconds; the store is still too small for its workload.
A store full of layer data keeps refusing until you add memory or, on self-hosted Valkey or Azure Managed Redis, until clients delete layers; waiting does not clear it.
While the clean-up runs, uploads carrying large values pause, usually for under a second.

1. Read `live_edit_gc_oom_collections_total` by `outcome`, and the `WARN` line clean-ups log.
   One replica runs them at a time, so search every replica's logs; it logs at most one such line every 30 s.
   The outcomes mean:
   - `freed`: the service freed memory; still grow the store.
   - `refused`: the store does not let the service free memory while it is full; only adding memory helps.
   - `nothing_to_collect`, or `freed_nothing` whose line shows `oom_skipped=0`, while writes stay refused: the store is full of data in use.
   - `freed_nothing` whose line shows `oom_skipped` above 0: memory is still owed and will be freed; not a sizing signal.
   - `failed`: the clean-up hit an error before it freed anything; its `error` field and a `GC pass failed` line say why. The next refused write starts another a second or more later; while it keeps failing, adding memory is still the fix.
2. Raise `maxmemory` (and the store's memory limit with it), or free memory.
   On Azure Managed Redis, scale the cache to a larger SKU, then set `valkey.memoryBytes` to the new total.
3. Size the store again as described in {ref}`Backing store <live-edit-backing-store>`.
4. Verify: `live_edit_store_write_refused_total{reason="out_of_memory"}` stops increasing and `live_edit_store_memory_used_ratio` falls below `0.85`.

(live-edit-monitoring-store-refusing)=
### Store refusing reads or writes

Read the `kind` or `reason` label; each points at a different owner.

| Label value | Fix |
|---|---|
| `auth` | The store rejected the credentials: check the connection credentials or the token sidecar ({ref}`Backing store <live-edit-backing-store>`). |
| `tls`, `config` | Fix the store URL, its host name or the cluster-mode setting. |
| `routing` | Check the store's shard health and slot coverage. |
| `misconf` | The store cannot persist: usually a full disk. |
| `readonly`, `noreplicas`, `backend_unavailable` | A failover that did not complete, or replicas lost. |
| `protocol`, `parse`, `invalid_command`, `invalid_argument`, `not_found`, `unknown` | Collect the log line, whose `detail` quotes the store's reply, and open a support case. |

Deleting layer keys does not help here.

(live-edit-monitoring-runbooks-data)=
## Runbooks: lost or damaged data

(live-edit-monitoring-keyspace-lost)=
### Keyspace lost

The backing store lost its data while the service kept running: a restart without persistence, a failover to a stale replica, or eviction.
Every live session and layer in the store is gone; clients are disconnected and reconnect.

With backups configured, one replica can also log this `WARN`, repeating about every minute: `checkpoint counters-sync: live le:counter:object_id is below the stored counters sidecar; not lowering the sidecar`.
It means the backing store went back to older data, even when `LiveEditKeyspaceLost` did not fire, for example after a failover to a stale replica that still holds an older copy of the service's keys.
Until a pod restarts, new layers can get ids that backups already hold, so restart the replicas: new layer ids then start above the backups' record.

1. Confirm the event in your store's own logs and metrics.
2. With backups configured, watch `live_edit_checkpoint_recovery_attempts_total{scope="layer"}` climb as layers are restored on access, then work through {ref}`After a restore <live-edit-monitoring-after-restore>`.
3. If `outcome="failed_after_repair"` ticked, the store is dropping writes it already acknowledged: check that it runs `maxmemory-policy noeviction` and is not failing over repeatedly. Nothing the service writes is durable until that is fixed.
4. If only `outcome="failed"` ticked, read the `ERROR` line for that repair: it names whether the backup store or the backing store failed.

(live-edit-monitoring-ext-integrity)=
### External-value integrity

Large values (external values) are stored once and reference-counted.

- **`live_edit_gc_underflow_total` increased:** collect the logs of every replica for the 15 minutes before and open a support case.
- **`live_edit_ext_ref_deficit_total` has been above zero in the last 6 hours:** a stored value may have been removed while a layer still uses it. With backups configured, compare `live_edit_checkpoint_ext_recovery_total` over the same window:
  - `result="recovered"`: the value was restored from the backup store; data is safe.
  - `result="adapter_miss"`: the backup store does not have it either. Re-upload the affected layer, and open a support case with the reconciler's `WARN` lines, which name the value.
  - neither: nothing has read the value yet.
- **`LiveEditReleaseBudgetNearFull`:** clean-up is not keeping up. Check `LiveEditCleanupBehind` and `live_edit_ext_read_holds`; raise `valkey.extReleaseMaxBytes` only if `live_edit_store_memory_used_ratio` shows the store has room.

(live-edit-monitoring-layer-keys)=
## Damaged layer keys

Something other than Live Edit wrote to the service's keys, which all start with `le:`.
The affected layer's writes, and usually its reads and backups, fail with `INTERNAL` until the key is fixed; other layers keep working.
A re-upload of the layer refused at the named key leaves the layer refusing edits until you remove the key.
The service then completes the re-upload by itself, with no further call, although it answered the client `INTERNAL`.
If the named key is the layer's own entry, the service does not complete the re-upload; see step 3.
If the named key is one of a deleted or replaced layer's indices, ending in `:obj:secs`, `:obj:dels`, `:obj:chs`, `:obj:flds` or `:obj:tss`, the layer's clean-up stopped partway; see step 4.
The deleted layer, or a replaced layer's old content, is gone (a `DeleteLayer` that met the key answered `INTERNAL`), but its other keys stay in the store.
`LiveEditLayerKeysDamaged` fires for such a key; in rare cases the background clean-up stops at it without counting it, and only its `WARN`, `failed to clean up orphaned object`, names the key.

1. Find the `ERROR` or `WARN` line from that time; it names the layer's object id and the exact key.
2. Find and stop the other writer: another application sharing the store, a debugging session, or a restore of foreign data.
3. If you can run commands against the store (for example `valkey-cli` on a self-hosted store), delete the named key; the service recreates it on the next write. Otherwise, for example on a private Azure Managed Redis cache, open a support case with the log line.
   One key is different: a layer's own entry, `le:{<session id>}:layer:<hex layer name>`.
   Fix it first: until it is removed, the background clean-up stops at it on every pass, for the whole deployment.
   Removing it leaves that layer with no content, so a client must create the layer again.
4. After a clean-up that stopped partway, also delete the layer's remaining keys, except `obj:extret` and `obj:extret:attempts`, which the background clean-up still needs, and `obj:stream`, which expires on its own.
   Set `{OBJECT_ID}` to the object id the line names, and `{HOST}` and `{PORT}` to your store; add `--tls` or `-a` if your store needs them.
   Use the bare number, for example `OBJECT_ID=42`; the commands add the braces.
   These keys all live on one node, and a scan on any other cluster node finds and deletes nothing.
   So on a cluster, first run these on any node (`{NODE_HOST}`, `{NODE_PORT}`): the first prints the keys' slot, and the second lists each shard's slot ranges and nodes.
   Use the primary of the shard whose range holds that slot as `{HOST}` and `{PORT}` below.

   ```bash
   OBJECT_ID={OBJECT_ID}
   valkey-cli -h {NODE_HOST} -p {NODE_PORT} CLUSTER KEYSLOT "{${OBJECT_ID}}"
   valkey-cli -h {NODE_HOST} -p {NODE_PORT} CLUSTER SHARDS
   ```

   Check first, with the scan alone, that it lists keys starting `le:{<object id>}:obj:`.
   If it lists nothing, the id or the node is wrong: stop and correct it.
   If it lists `le:{<object id>}:obj:hdr`, the layer is still live or its clean-up has not stopped yet: stop and recheck the id.

   ```bash
   OBJECT_ID={OBJECT_ID}
   valkey-cli -c -h {HOST} -p {PORT} --scan --pattern "le:{${OBJECT_ID}}:obj:*"
   ```

   Then delete the keys:

   ```bash
   OBJECT_ID={OBJECT_ID}
   valkey-cli -c -h {HOST} -p {PORT} --scan --pattern "le:{${OBJECT_ID}}:obj:*" \
     | grep -v -e ':obj:extret$' -e ':obj:extret:attempts$' -e ':obj:stream$' \
     | xargs -r valkey-cli -c -h {HOST} -p {PORT} unlink
   ```

5. Verify: the counters stop increasing and clients can edit the layer again.
   After step 4, run the same scan again:

   ```bash
   OBJECT_ID={OBJECT_ID}
   valkey-cli -c -h {HOST} -p {PORT} --scan --pattern "le:{${OBJECT_ID}}:obj:*"
   ```

   It lists nothing, or only the layer's `obj:extret` and `obj:extret:attempts` keys, which the background clean-up removes later; for 30 seconds after the failure it can also list `obj:stream`.
   If `obj:stream` is still listed after a minute, unlink it too.

(live-edit-monitoring-runbooks-service)=
## Runbooks: capacity, dependencies and delivery

(live-edit-monitoring-capacity-refusals)=
### Capacity refusals

Refusals reach clients as a retryable `UNAVAILABLE`; each one logs an `RPC failed` line at `WARN` naming the session and layer turned away.

1. Compare the replica's usage with its limit in {ref}`Capacity and saturation <live-edit-monitoring-capacity>`.
2. **Subscriptions:** add replicas so new subscriptions land elsewhere. In `scaler` mode, raise `autoscaling.maxReplicas` if `live_edit_scaler_target_replicas` already equals it. To give each replica more subscriptions, raise `resources.limits.memory`; the ceiling follows it.
3. **Catch-ups** (`reason="catchup_capacity"`): a join storm. Raise `valkey.subscriberMaxConcurrentCatchups` only together with the CPU limit; otherwise add replicas.
4. **Uploads:** if `live_edit_upload_reserved_bytes / live_edit_upload_budget_bytes` is near 1, raise `resources.limits.memory`, which the upload memory follows, or add replicas. If it is low, one session has too many uploads in flight; the refusal lines name it.
5. Verify: the refusal counters stop increasing.

(live-edit-monitoring-backups)=
### Backups behind or stopped

Layers that are not backed up are lost if the backing store loses its data.

1. Check `live_edit_checkpoint_writes_total{result="error"}` by `kind`: errors mean the backup store is failing or unreachable, except as in step 2.
2. If the errors are `kind="counters_sidecar"` only, and one replica logs a `WARN` containing `answered absent, but this process has already seen the counters sidecar` about every minute:
   - Shortly after the first layer was created on a new backup prefix, this is expected when your Storage service caches "not found" answers, and it stops within that cache time.
   - If it continues past that, the service's layer-id record, `{BACKUP_PREFIX}/.live-edit/counters.json`, was deleted. Restart the replicas with `kubectl -n {NAMESPACE} rollout restart deployment/{DEPLOYMENT}`; a replica writes it again within about two minutes.
3. If replicas log a `WARN` containing `the post-write checkpoint listing is empty`, backups are written but the service cannot list them: old backups are not deleted, and a restore finds no backup.
   Check that the pods reach the Storage service's list endpoint through any gateway or proxy in front of it, and that the list endpoint resolves `checkpoint.basePath`, in the form you set it, to the same prefix that writes use.
4. Search for the `WARN` line naming the stuck layer's `object_id`.
5. If large layers fail while small ones succeed, the pod cannot reach the backup store's blob-storage host; open the network access described in {ref}`Durability <live-edit-durability-network>`.
6. If `LiveEditBackupLoopStopped` fired with no errors, collect the logs of every replica, then restart them with `kubectl -n {NAMESPACE} rollout restart deployment/{DEPLOYMENT}`.
7. Verify: the p99 of `live_edit_checkpoint_pending_age_seconds` falls below the backup interval, and `kind="counters_sidecar"` errors and the `post-write checkpoint listing is empty` warnings stop.

(live-edit-monitoring-credential)=
### Credential expired

The `role` label names the token: `valkey` (backing-store connections) or `storage-api` (backup-store calls).

1. Check the token-provisioner sidecar's logs and that both containers mount the shared token volume at the same path.
2. A `role="valkey"` expiry makes the replica not ready; a `role="storage-api"` expiry stops backups.
3. Verify: `live_edit_service_token_expires_in_seconds` jumps back to a full token lifetime.

(live-edit-monitoring-connection-repair)=
### Connection repair disabled

The service found connections to the backing store in an unexpected state, rebuilt them, and found the same state again, so it stopped rebuilding.
This is almost always a proxy or connection broker in front of the store that rewrites replies.
Restarting replicas does not help.

1. Read the `ERROR` line's `detail` (the reply) and `node` fields.
2. Take the reply to whoever owns the proxy or the store endpoint.

(live-edit-monitoring-scaler)=
### Scaler not deciding

While this fires, the replica count is frozen.

1. Check that the scaler pod is running and scraped.
2. Check `live_edit_scaler_poll_errors_total` by `stage`: `endpoints` or `scale` means the scaler's Role or RoleBinding is missing a rule.
3. If decisions are all `reason="no_data"`, the scaler cannot reach the server pods' metrics port: check NetworkPolicies.
4. Verify: `live_edit_scaler_decisions_total` increases about once per poll interval.

(live-edit-monitoring-slow-propagation)=
### Edits propagate slowly

Work outward from the service:

1. **`stage="enqueued"` lag high, `live_edit_subscriber_send_stalls_total{phase="live"}` flat:** the service or the store is slow. Check `LiveEditBackingStoreTimeouts` and replica CPU. The `subscriber delivery is lagging` log line names the affected subscriptions.
2. **`stage="transport"` lag well above `enqueued`, or live stalls rising:** the client, or the proxy in front of the service, is not reading fast enough.
3. **Both low while users see delays:** the delay is past the service, in a proxy's buffers or the client. A proxy with a large HTTP/2 window hides a slow client from the service; check the proxy's own flow-control statistics.

(live-edit-monitoring-self-check)=
### Self-check fired

One of the service's internal consistency checks failed.
The service has already protected the data, so no immediate action is needed.
Collect the logs of every replica for the 15 minutes before the alert, note which metric increased, and open a support case.
During or after an upgrade from, or a rollback to, a release that stores time samples at `-0.0` as written, first check whether the upgrade note expects the increase: {ref}`Upgrading from such a release <live-edit-operations-upgrade-time-zero>`.

(live-edit-monitoring-after-restore)=
## After a disaster-recovery restore

Restores happen automatically, when a client opens a layer or session the backing store has lost.
For a day afterwards, watch these signals; the `WARN` lines they come with name the layers involved.

| Signal | What it means | What to do |
|---|---|---|
| `live_edit_checkpoint_recovery_stream_unresolved_total{reason=~"probe_unreadable\|unlink_unverified"}` | A layer could not be restored because the backing store did not answer reliably. | Check the store's health; the next request for the layer tries again. |
| `live_edit_checkpoint_recovery_stream_unresolved_total{reason="wrongtype_key"}` | A layer was restored but cannot be edited: another writer left a key in its place. | Follow {ref}`Damaged layer keys <live-edit-monitoring-layer-keys>`. |
| `live_edit_checkpoint_base_bootstrap_total{outcome="shed"}` | A layer, usually a restored one, has not been backed up again yet; it is backed up at the next backup pass. | Nothing, unless `LiveEditBackupsBehind` also fires. |
| `live_edit_checkpoint_stable_cut_total{outcome="unverified"}` | Some layers skipped one backup right after the restore. | Nothing; they are backed up at a later backup pass. |
| `live_edit_checkpoint_recovery_stream_refused_total` | A restored layer's leftover pending updates were discarded. | Nothing; expected during a large restore. |
| `live_edit_checkpoint_ext_recovery_total{result="adapter_miss"}` | A large value a restored layer uses has no backup copy, so requests that open that layer fail with `INTERNAL`. | Have a user upload the value again or replace the attribute value that uses it ({ref}`Backups of large values <live-edit-durability-large-values>`). |

Expect subscribers of restored layers to reconnect once, seen as `live_edit_subscriber_stream_exits_total{reason="stream_gap"}`.

(live-edit-monitoring-additional-alerts)=
## Additional alert rules

These rules cover narrower conditions. Add them as you need them; none is required.

**Capacity and dependencies:**

| Alert | Fires when | Severity, `for` | Next step |
|---|---|---|---|
| `LiveEditSubscriptionsRefused` | A replica refuses live subscriptions for lack of capacity | page, 5m | {ref}`Capacity refusals <live-edit-monitoring-capacity-refusals>` |
| `LiveEditUploadsRefused` | A replica refuses uploads for lack of upload memory | page, 5m | {ref}`Capacity refusals <live-edit-monitoring-capacity-refusals>` |
| `LiveEditUploadsWaiting` | p99 upload processing wait is above 10 s | warning, 10m | {ref}`Capacity refusals <live-edit-monitoring-capacity-refusals>` |
| `LiveEditUnavailableErrors` | More than 5 % of an endpoint's calls end `UNAVAILABLE` | warning, 15m | `RPC failed` lines at `WARN` name the cause |
| `LiveEditSlowJoins` | p95 time to open a layer or join a session is above 10 s | warning, 15m | {ref}`Capacity refusals <live-edit-monitoring-capacity-refusals>` |
| `LiveEditDeliveryLag` | p99 live delivery lag is above 5 s | warning, 15m | {ref}`Slow propagation <live-edit-monitoring-slow-propagation>` |
| `LiveEditMemoryNearLimit` | A pod's working set is above 90 % of its limit (needs cAdvisor and kube-state-metrics) | warning, 15m | {ref}`Capacity <live-edit-monitoring-capacity>` |
| `LiveEditBackingStoreRefusingWrites` | The store keeps refusing writes | warning, 15m | {ref}`Store refusing <live-edit-monitoring-store-refusing>` |
| `LiveEditStoreRefusingReads` | The store permanently refuses reads of layer state | page, 0m | {ref}`Store refusing <live-edit-monitoring-store-refusing>` |
| `LiveEditBackingStoreTimeouts` | Backing-store commands keep timing out | warning, 10m | store CPU and latency |
| `LiveEditStoreCleanupRefused` | The full store refused the service's clean-up of released large values | warning, 0m | {ref}`Store full <live-edit-monitoring-store-full>` |
| `LiveEditReleaseBudgetNearFull` | Released large values are above 80 % of their budget | warning, 10m | {ref}`External values <live-edit-monitoring-ext-integrity>` |
| `LiveEditCleanupBehind` | Released large values wait more than 60 s for clean-up | warning, 10m | {ref}`External values <live-edit-monitoring-ext-integrity>` |
| `LiveEditConnectionRepairDisabled` | The service stopped repairing backing-store connections | warning, 5m | {ref}`Connection repair <live-edit-monitoring-connection-repair>` |
| `LiveEditScalerNotDeciding` | The autoscaler is not setting the replica count (`scaler` mode only) | warning, 10m | {ref}`Scaler <live-edit-monitoring-scaler>` |

A client that sends each upload only once fails on the first refusal; on such a fleet set `LiveEditUploadsRefused` to `for: 0m`.

**Specialised:**

| Alert | Fires when | Severity, `for` |
|---|---|---|
| `LiveEditTopologyRepliesPersist` | Connection checks keep getting cluster-topology replies; check for a proxy in front of the store | warning, 45m |
| `LiveEditOlderReleaseOnStore` | A replica of an older release acts on the same store; silence while upgrading | warning, 0m |
| `LiveEditCredentialRefreshLeadShort` | The backing-store token rotated with under 180 s to spare; raise the sidecar's refresh lead | warning, 0m |
| `LiveEditReconcilerIncomplete` | The reference-count reconciler keeps abandoning its pass; its `WARN` lines say why | warning, 0m |
| `LiveEditReleasesNotRecorded` | A release of a large value could not be recorded | warning, 0m |
| `LiveEditScaleDownHeld` | Scale-down is held by a pod the scaler cannot account for | warning, 15m |
| `LiveEditPodMemoryLimitUnread` | A replica read no memory limit and sized itself as a 5 GiB pod; fix what its startup warning's `reason` names ({ref}`Startup and readiness <live-edit-troubleshooting-startup>`) | warning, 0m |
| `LiveEditTraceSpansLost` | Trace spans are dropped; raise `observability.tracing.batch.maxQueueSize` or fix the collector | ticket, 15m |

(live-edit-monitoring-backup-alerts)=
## Backup alert rules

These optional rules watch backups. Add them only with `checkpoint.adapter` set: with backups off, `LiveEditBackupLoopStopped` fires permanently.

| Alert | Fires when | Severity, `for` | Next step |
|---|---|---|---|
| `LiveEditBackupLoopStopped` | No layer evaluated for backup in 10 minutes; also fires on a deployment with no layers | warning, 10m | {ref}`Backups behind <live-edit-monitoring-backups>` |
| `LiveEditLayerBackupStuck` | One layer has gone unbacked-up past three times its expected bound | warning, 5m | {ref}`Backups behind <live-edit-monitoring-backups>` |
| `LiveEditLayerNeverBackedUp` | A layer has no backup at all | warning, 10m | {ref}`Backups behind <live-edit-monitoring-backups>` |
| `LiveEditBackupBlobEndpointUnreachable` | Large backups cannot upload to the backup store's blob storage | warning, 15m | {ref}`Durability <live-edit-durability-network>` |
| `LiveEditBackupStoreDamaged` | Incremental backups were found with no full backup under them | warning, 0m | repair the backup store |
| `LiveEditExternalValueMirrorDropping` | Copies of large values are skipped for lack of copy capacity; each is repaired minutes later | warning, 5m | the `reason` in {ref}`Are large values backed up? <live-edit-metrics-reference-large-value-backups>` |
| `LiveEditLargeValueBackupsMissing` | Large values have gone an hour without a backup copy | warning, 0m | `live_edit_checkpoint_ext_backup_repair_total` by `outcome` in {ref}`Are large values backed up? <live-edit-metrics-reference-large-value-backups>`: `probe_failed`, and `resend_failed` with a `WARN` containing `could not upload` or `timed out`, mean the backup store is failing or slow, as in {ref}`Backups behind or stopped <live-edit-monitoring-backups>`; any other `resend_failed` is a backing-store read that failed for one value; `budget_busy` means copy capacity is short: act as for `budget_full`, checking the backup store's latency first; `oversized` names its value in a `WARN`. |
| `LiveEditBackupCleanupBlocked` | Large-value clean-up stopped at a backup-store error, a malformed or vanished file, or its time limit | warning, 0m | A `WARN` containing `could not read the live ext-hash set` carries the reason. If it contains `was named by the layers listing but read as absent`, a file was listed and then not found: no action is needed, though the alert stays up for 6 hours; if `live_set_failed` keeps rising pass after pass, take the `WARN` lines to your Storage service's owner. A `permanent checkpoint-store error` ending in a line and column means one of the `.ext-refs.json` files under `{BACKUP_PREFIX}/.live-edit/layers/` is malformed; the line does not name the file. Open a support case, and do not delete the file: clean-up could then delete large values a backup still needs. Any other error means the backup store failed or refused the request: check that it is reachable and accepts the service's requests, as in {ref}`Backups behind or stopped <live-edit-monitoring-backups>`. A `WARN` containing `exceeded its collect-phase budget`, with `live_edit_checkpoint_adapter_ceiling_elapsed_total{phase="enumeration"}` rising, means the backup store answered too slowly for one pass: check its latency. |
| `LiveEditBackupStorageGrowing` | Backup-store clean-up backlogs kept growing for 6 hours | warning, 1h | `LiveEditBackupCleanupBlocked` first. Then a `WARN` containing `exceeded its enumeration share` means the backup store listed too slowly for one pass: check its latency. A `WARN` containing `could not enumerate the layers namespace` means clean-up could not list the backup store: see {ref}`Backups behind or stopped <live-edit-monitoring-backups>`. An `INFO` line containing `listed no artifacts in the layers namespace` means clean-up listed no layer backups. On a store with none yet, deleted layers wait, and the backlog drains after the next layer backup. If replicas also log `the post-write checkpoint listing is empty`, the service cannot list its backups: see {ref}`Backups behind or stopped <live-edit-monitoring-backups>`. A `WARN` containing `two checkpoint listings in one reconciliation pass disagreed about a dead object` means the backup store answered two listings differently in one pass; that deleted layer's backups are kept, and their clean-up is retried next pass. One alone needs no action, but if it recurs, take it to your Storage service's owner. |
| `LiveEditRestoresRedirected` | Restores are redirected through blob storage | warning, 0m | the backup store's inline-read setting |

`LiveEditExternalValueMirrorDropping` warns that copy capacity is short: the service repairs each skipped copy a few minutes later, except an `oversized` one, and until then the value has no backup ({ref}`Backups of large values <live-edit-durability-large-values>`).
It leaves out `reason="shutdown"`: a replica that stops with large values still waiting for the backup store counts them there, so rollouts under upload load can fire it, and no setting prevents that.
`LiveEditLargeValueBackupsMissing` reads the result of the last check that finished.
A slow backup store does not silence it: each check gives up on existence requests and uploads after 4 minutes and still counts every value without a copy.
Only a check that cannot finish counting within its 5-minute limit reports nothing, so the alert can stay silent while that repeats.
Look for repeated `WARN` lines containing `backup repair pass did not complete its walk`, and check the backing store's latency.

(live-edit-monitoring-internal-alerts)=
## Internal-check alerts

These rules watch the service's own consistency checks. If one fires, collect the logs of every replica and open a support case; the runbook says what to check first.

| Alert | Fires when | Severity, `for` | Runbook |
|---|---|---|---|
| `LiveEditKeyspaceLost` | The backing store lost its data while replicas kept running | page, 0m | {ref}`Keyspace lost <live-edit-monitoring-keyspace-lost>` |
| `LiveEditLayerKeysDamaged` | A layer's keys hold values the service did not write | page, 0m | {ref}`Damaged layer keys <live-edit-monitoring-layer-keys>` |
| `LiveEditExternalValueIntegrity` | A large value's reference count went wrong | warning, 0m | {ref}`External values <live-edit-monitoring-ext-integrity>` |
| `LiveEditCounterRepairSlow` | Uploads give up waiting for a layer-id counter repair | warning, 10m | backup-store latency |
| `LiveEditSelfCheckFired` | An internal consistency check fired | warning, 0m | {ref}`Self-check <live-edit-monitoring-self-check>` |

Silence `LiveEditOlderReleaseOnStore` and `LiveEditExternalValueIntegrity` while you upgrade or roll back.

(live-edit-monitoring-additional-rules)=
## Additional alert rule definitions

The rules for the tables above, in the same format as the recommended rules.

```yaml
groups:
- name: live-edit-additional
  rules:
  - alert: LiveEditSubscriptionsRefused
    expr: |
      sum by (instance) (increase(live_edit_subscriber_admission_total{result=~"refused_saturated|refused_waiter_cap"}[5m])) > 0
      or sum by (instance) (increase(live_edit_subscriber_stream_exits_total{reason=~"catchup_capacity|reader_capacity"}[5m])) > 0
    for: 5m
    labels: {severity: page}
  - alert: LiveEditUploadsRefused
    expr: |
      sum by (instance) (increase(live_edit_upload_admission_total{result=~"refused_saturated|refused_waiter_cap|refused_growth"}[5m])) > 0
    for: 5m
    labels: {severity: page}
  - alert: LiveEditUploadsWaiting
    expr: histogram_quantile(0.99, sum by (instance, le) (rate(live_edit_upload_processing_wait_seconds_bucket[5m]))) > 10
    for: 10m
    labels: {severity: warning}
  - alert: LiveEditUnavailableErrors
    expr: |
      sum by (endpoint) (rate(live_endpoint_calls_total{endpoint!~"JoinSession|SubscribeReadLayer|other", grpc_code="UNAVAILABLE"}[5m]))
        / sum by (endpoint) (rate(live_endpoint_calls_total{endpoint!~"JoinSession|SubscribeReadLayer|other"}[5m])) > 0.05
    for: 15m
    labels: {severity: warning}
  - alert: LiveEditSlowJoins
    expr: histogram_quantile(0.95, sum by (le, endpoint) (rate(live_edit_join_latency_seconds_bucket[5m]))) > 10
    for: 15m
    labels: {severity: warning}
  - alert: LiveEditDeliveryLag
    expr: histogram_quantile(0.99, sum by (le) (rate(live_edit_subscriber_delivery_lag_seconds_bucket{stage="enqueued"}[5m]))) > 5
    for: 15m
    labels: {severity: warning}
  - alert: LiveEditMemoryNearLimit
    expr: |
      max by (namespace, pod) (container_memory_working_set_bytes{container="live-edit-service"})
        / max by (namespace, pod) (kube_pod_container_resource_limits{resource="memory", container="live-edit-service"}) > 0.9
    for: 15m
    labels: {severity: warning}
  - alert: LiveEditBackingStoreRefusingWrites
    expr: |
      sum(rate(live_edit_store_write_refused_total{reason=~"misconf|readonly|noreplicas"}[5m])) > 0
      or sum(rate(live_edit_snapshot_read_refused_total{kind="backend_unavailable"}[5m])) > 0
    for: 15m
    labels: {severity: warning}
  - alert: LiveEditStoreRefusingReads
    expr: |
      sum by (instance, kind) (increase(live_edit_snapshot_read_refused_total{kind!~"backend_unavailable|backend_out_of_memory"}[15m])) > 0
    for: 0m
    labels: {severity: page}
  - alert: LiveEditBackingStoreTimeouts
    expr: sum by (instance) (rate(live_edit_valkey_errors_total{kind="command_timeout"}[5m])) > 0
    for: 10m
    labels: {severity: warning}
  - alert: LiveEditStoreCleanupRefused
    expr: sum(increase(live_edit_gc_oom_collections_total{outcome="refused"}[15m])) > 0
    for: 0m
    labels: {severity: warning}
  - alert: LiveEditReleaseBudgetNearFull
    expr: max(live_edit_ext_release_pending_bytes) > 0.8 * min(max without (source) (live_edit_ext_release_budget_bytes) > 0)
    for: 10m
    labels: {severity: warning}
  - alert: LiveEditCleanupBehind
    expr: max(live_edit_ext_release_oldest_age_seconds) > 60
    for: 10m
    labels: {severity: warning}
  - alert: LiveEditConnectionRepairDisabled
    expr: sum by (instance, reason) (rate(live_edit_valkey_txn_repair_ineffective_total[5m])) > 0
    for: 5m
    labels: {severity: warning}
  - alert: LiveEditScalerNotDeciding
    expr: |
      absent(live_edit_scaler_decisions_total)
      or sum(rate(live_edit_scaler_decisions_total[10m])) == 0
      or (sum(rate(live_edit_scaler_decisions_total{reason="no_data"}[10m])) > 0
          and on() sum(rate(live_edit_scaler_decisions_total{reason!="no_data"}[10m])) == 0)
    for: 10m
    labels: {severity: warning}
- name: live-edit-backups
  rules:
  - alert: LiveEditBackupLoopStopped
    expr: sum(rate(live_edit_checkpoint_writes_total{kind="layer"}[10m])) == 0
    for: 10m
    labels: {severity: warning}
  - alert: LiveEditLayerBackupStuck
    expr: max(live_edit_checkpoint_oldest_pending_age_seconds) > 3 * max(live_edit_checkpoint_config_stuck_threshold_seconds)
    for: 5m
    labels: {severity: warning}
  - alert: LiveEditLayerNeverBackedUp
    expr: max(live_edit_checkpoint_layers_never_written) > 0
    for: 10m
    labels: {severity: warning}
  - alert: LiveEditBackupBlobEndpointUnreachable
    expr: increase(live_edit_checkpoint_adapter_redirect_total{verb="put", outcome=~"target_failed|complete_failed"}[15m]) > 0
    for: 15m
    labels: {severity: warning}
  - alert: LiveEditBackupStoreDamaged
    expr: increase(live_edit_checkpoint_tail_total{outcome="refused_lineage_unrootable"}[15m]) > 0
    for: 0m
    labels: {severity: warning}
  - alert: LiveEditExternalValueMirrorDropping
    expr: sum by (instance, reason) (rate(live_edit_checkpoint_ext_mirror_dropped_total{reason!="shutdown"}[5m])) > 0
    for: 5m
    labels: {severity: warning}
  - alert: LiveEditLargeValueBackupsMissing
    expr: min_over_time((max(live_edit_checkpoint_ext_backup_owed))[1h:]) > 0
    for: 0m
    labels: {severity: warning}
  - alert: LiveEditBackupCleanupBlocked
    expr: increase(live_edit_checkpoint_reconcile_total{outcome="live_set_failed"}[6h]) > 0
    for: 0m
    labels: {severity: warning}
  - alert: LiveEditBackupStorageGrowing
    expr: |
      (min_over_time((max(live_edit_checkpoint_ext_tombstone_backlog))[6h:]) > 0
        and deriv((max(live_edit_checkpoint_ext_tombstone_backlog))[6h:]) > 0)
      or (min_over_time((max(live_edit_checkpoint_dead_object_backlog))[6h:]) > 0
        and deriv((max(live_edit_checkpoint_dead_object_backlog))[6h:]) > 0)
    for: 1h
    labels: {severity: warning}
  - alert: LiveEditRestoresRedirected
    expr: increase(live_edit_checkpoint_adapter_redirect_total{verb="get", outcome="followed"}[1h]) > 0
    for: 0m
    labels: {severity: warning}
- name: live-edit-specialised
  rules:
  - alert: LiveEditTopologyRepliesPersist
    expr: sum by (instance) (rate(live_edit_valkey_txn_dirty_checkouts_total{reason="topology_reply"}[15m])) > 0
    for: 45m
    labels: {severity: warning}
  - alert: LiveEditOlderReleaseOnStore
    expr: sum(increase(live_edit_gc_older_build_total[15m])) > 0
    for: 0m
    labels: {severity: warning}
  - alert: LiveEditCredentialRefreshLeadShort
    expr: min by (instance) (min_over_time(live_edit_service_token_refresh_headroom_seconds{role="valkey"}[1d])) < 180
    for: 0m
    labels: {severity: warning}
  - alert: LiveEditReconcilerIncomplete
    expr: sum(increase(live_edit_ext_reconcile_total{outcome="enumeration_failed"}[6h])) > 2
    for: 0m
    labels: {severity: warning}
  - alert: LiveEditReleasesNotRecorded
    expr: sum(increase(live_edit_ext_ref_release_enqueue_failed_total[1h])) > 0
    for: 0m
    labels: {severity: warning}
  - alert: LiveEditScaleDownHeld
    expr: increase(live_edit_scaler_scale_down_blocked_total{reason="unsettled"}[2m]) > 0
    for: 15m
    labels: {severity: warning}
  - alert: LiveEditPodMemoryLimitUnread
    expr: max by (instance) (live_edit_pod_memory_limit_bytes{source="reference"}) > 0
    for: 0m
    labels: {severity: warning}
  - alert: LiveEditTraceSpansLost
    expr: sum by (instance, reason) (increase(live_edit_trace_spans_dropped_total[15m])) > 0
    for: 15m
    labels: {severity: ticket}
- name: live-edit-internal-checks
  rules:
  - alert: LiveEditKeyspaceLost
    expr: |
      sum by (instance) (increase(live_edit_object_id_counter_repair_total{outcome=~"repaired_seeded|repaired_noop|failed|failed_after_repair"}[15m])) > 0
    for: 0m
    labels: {severity: page}
  - alert: LiveEditLayerKeysDamaged
    expr: |
      sum by (instance) (increase(live_edit_keyspace_type_error_total[15m])) > 0
      or sum by (instance) (increase(live_edit_snapshot_read_type_error_total[15m])) > 0
      or sum by (instance) (increase(live_edit_stream_id_collision_total[15m])) > 0
    for: 0m
    labels: {severity: page}
  - alert: LiveEditExternalValueIntegrity
    expr: |
      sum(increase(live_edit_gc_underflow_total[15m])) > 0
      or max_over_time(max(live_edit_ext_ref_deficit_total)[6h:]) > 0
    for: 0m
    labels: {severity: warning}
  - alert: LiveEditCounterRepairSlow
    expr: increase(live_edit_object_id_counter_repair_total{outcome="wait_timeout"}[30m]) > 0
    for: 10m
    labels: {severity: warning}
  - alert: LiveEditSelfCheckFired
    expr: |
      sum(increase(live_edit_structural_cache_divergence_total[15m])) > 0
      or sum(increase(live_edit_structural_cache_audit_total{outcome="diverged"}[15m])) > 0
      or sum(increase(live_edit_update_layer_structural_fallbacks_total{reason="repeat_divergence"}[15m])) > 0
      or sum(increase(live_edit_storage_watch_unwatch_failures_total{site="drop"}[15m])) > 0
      or sum(increase(live_edit_storage_watch_voided_total{disposition="committed_unwatched"}[15m])) > 0
      or sum(increase(live_edit_stream_mux_poison_keys_total[15m])) > 0
      or sum(increase(live_edit_ext_ref_undecodable_total[15m])) > 0
      or sum(increase(live_edit_ext_index_drift_total{direction="under"}[15m])) > 0
    for: 0m
    labels: {severity: warning}
```

(live-edit-monitoring-references)=
## References

- [gRPC health checking protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md) — the protocol the health services implement.
- [Prometheus Operator API](https://prometheus-operator.dev/docs/api-reference/api/) — the `ServiceMonitor`, `PodMonitor` and `PrometheusRule` resources.
- [W3C Trace Context](https://www.w3.org/TR/trace-context/) — the `traceparent` header.

(live-edit-monitoring-see-next)=
## See next

{ref}`Metrics reference <live-edit-metrics-reference>` describes every metric these alerts use.
