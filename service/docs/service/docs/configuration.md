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

(live-edit-configuration)=
# Live Edit configuration reference

This page lists the Helm values that configure Live Edit, the environment variable each one sets, and the chart default.
Read {ref}`Deployment <live-edit-deployment>` first; the topic pages linked below explain the decisions behind the settings that need one.
`helm show values` prints the full description of every value.

(live-edit-configuration-how-settings-apply)=
## How settings are applied

The chart renders the service settings into a ConfigMap that the container reads once at startup; the backing-store URL comes from a Secret or from `valkey.url`.

- **Changes roll the pods.** A `helm upgrade` that changes a server setting on this page rolls the server pods; a change to a scaler setting rolls only the scaler pod. Clients of a replaced pod reconnect; see {ref}`Operations <live-edit-operations>`.
- **Secret changes do not.** Rotating the Secret that holds the backing-store URL takes effect only when the pods restart.
- **Invalid values stop early.** The chart refuses some invalid values at render time, naming the value. On any other invalid setting the pod logs `configuration error:` with the variable's name, exits, and never becomes Ready; the previous pods keep serving.

(live-edit-configuration-without-the-chart)=
### Running the container without the chart

Set the environment variables in the second column. For the server, an unset variable takes the service's built-in default, which matches the chart default except for these:

| Environment variable | Built-in default |
|---|---|
| `LIVE_EDIT_VALKEY_URL` | `redis://localhost:6379` |
| `LIVE_EDIT_STORAGE_SERVICE_ENDPOINT` | none; required while authorization is on |
| `LIVE_EDIT_USERINFO_ENDPOINT` | none; required while the userinfo service is on |
| `LIVE_EDIT_LOG_FORMAT` | `json` |
| `LIVE_EDIT_VALKEY_POOL_SIZE`, `LIVE_EDIT_SUBSCRIBER_POOL_SIZE` | `0`: twice the CPU count (the data pool at least 8) |
| `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS` | derived from `LIVE_EDIT_RPC_DEADLINE_MS`: 3600 at the default deadline |
| `LIVE_EDIT_CHECKPOINT_ENABLED` | `false` |
| `LIVE_EDIT_SERVICE_TOKEN_FILE`, `LIVE_EDIT_VALKEY_TOKEN_FILE` | none; set the path when the mode needs it |
| `LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS` | the subscription limit; leave unset, since `0` fails startup |
| `OTEL_TRACES_SAMPLER` | set `parentbased_traceidratio`, or `OTEL_TRACES_SAMPLER_ARG` has no effect |

Give the server container a memory limit, such as `docker run --memory 5g`: the service sizes its memory budgets from it.
Without one, or when it cannot read the limit, it logs a warning at startup and sizes itself as a 5 GiB pod, whatever memory the host has.

The scaler (`live-edit scaler`) requires `LIVE_EDIT_SCALER_NAMESPACE`, `LIVE_EDIT_SCALER_DEPLOYMENT`, `LIVE_EDIT_SCALER_SERVICE`, `LIVE_EDIT_SCALER_MIN_REPLICAS` and `LIVE_EDIT_SCALER_MAX_REPLICAS`; its CPU target defaults to `0` (off), and `LIVE_EDIT_SCALER_CPU_REQUEST` is required once you set one.

(live-edit-configuration-value-syntax)=
## Value syntax

- **Durations** are whole numbers in the unit the name ends with: `Ms` (`_MS` in a variable) is milliseconds, `Secs` or `Seconds` (`_SECS`) is seconds. The `checkpoint` intervals without a unit are seconds. A unit suffix such as `15s` is refused.
- **Sizes** are whole numbers of bytes, for names ending `Bytes` (`_BYTES`) and for `layer.maxSize`. A suffix such as `128Mi` is refused, except in the Kubernetes `resources` values. In `resources.limits.memory` and `resources.requests.memory` the chart also refuses the `m`, `u` and `n` suffixes (fractions of a byte, almost always a typo for `Mi` or `M`) and a zero memory limit. Other forms Kubernetes accepts, such as `5e9`, are read as Kubernetes reads them.
- **Whole numbers** are plain digits. For most settings the chart refuses a negative or fractional number, or text, at install, naming the value. It does not check `valkey.memoryBytes`, `valkey.extReleaseMaxBytes`, `valkey.snapshotAtomicReadMax*`, `valkey.subscriberAdmissionMaxWaiters`, `layer.maxSize`, `layer.uploadAdmission*`, `shutdown.streamEnd*`, `observability.tracing.batch.*`, `checkpoint.tail.max*` or `autoscaling.*`. Write plain digits there, because anything else can be read as a different number, such as `0`, or be refused only when the pods start. For `autoscaling.*`, quote values of a million or more.
- **Booleans** are `true` or `false`.
- **Unset** values (empty or `null`) let the service choose; the Default column says what it chooses. Emptying a value that has a non-empty default can fail the render or startup, or change behaviour; leave values you don't mean to change out of your values file.
- **`0`** means off, unlimited or refused, depending on the setting.

(live-edit-configuration-service)=
## Service ports and labels

| Helm value | Environment variable | Default | Description |
|---|---|---|---|
| `service.grpc.port` | `LIVE_EDIT_PORT` | `50001` | gRPC port for clients and health probes. |
| `service.metrics.port` | `LIVE_EDIT_METRICS_PORT` | `9090` | HTTP port serving Prometheus `/metrics` and the autoscaler's load reads. |
| `serviceLabel` | `LIVE_EDIT_SERVICE_LABEL` | unset (the pod name) | Value of the `service` label on every metric. Set a stable name to keep one series across pod restarts. |

(live-edit-configuration-authorization)=
## Authorization and identity

{ref}`Authentication <live-edit-authentication>` explains how to choose these.

| Helm value | Environment variable | Default | Description |
|---|---|---|---|
| `authorization.enabled` | `LIVE_EDIT_AUTHZ_ENABLED` | `true` | Check each request's token and path permissions with the Storage Service. `false` requires `userInfo.enabled: true`. |
| `authorization.storageServiceEndpoint` | `LIVE_EDIT_STORAGE_SERVICE_ENDPOINT` | `http://storage-service:8012` | Storage Service URL for permission checks. Include `https://` or `http://`; a bare host name fails startup. |
| `authorization.protocol` | `LIVE_EDIT_AUTH_PROTOCOL` | `rest` | `rest` or `grpc`, for both permission and user-identity calls. |
| `authorization.timeoutMs` | `LIVE_EDIT_AUTH_TIMEOUT_MS` | `5000` | Timeout of one authorization call. |
| `authorization.cacheTtlSecs` | `LIVE_EDIT_AUTH_CACHE_TTL_SECS` | `600` | How long an allowed decision is cached. Revoked access can keep working this long. |
| `authorization.cacheNegativeTtlSecs` | `LIVE_EDIT_AUTH_CACHE_NEGATIVE_TTL_SECS` | `30` | How long a denied decision is cached. |
| `userInfo.enabled` | `LIVE_EDIT_USERINFO_SERVICE_ENABLED` | `false` | Resolve user identity with the userinfo service instead of the token's claims. |
| `userInfo.endpoint` | `LIVE_EDIT_USERINFO_ENDPOINT` | unset (`http://userinfo-service:8080`, or `:50051` for `grpc`) | Userinfo service URL, used when `userInfo.enabled` is `true`. Include the scheme. |
| `serviceIdentity.accessTokenFile.enabled` | — | `false` | Give the service a token file for its outbound calls (sets the next row's variable). Required by the `StorageApi` checkpoint adapter and by access-token Valkey authentication. |
| `checkpoint.credentials.accessTokenFile.path` | `LIVE_EDIT_SERVICE_TOKEN_FILE` | `/etc/secrets/access-token/access_token` | Where the service reads its outbound token. The service rereads it, so renewals need no restart. |

(live-edit-configuration-observability)=
## Logs, traces and metrics export

{ref}`Monitoring <live-edit-monitoring>` covers what to collect.

| Helm value | Environment variable | Default | Description |
|---|---|---|---|
| `observability.logging.level` | `LIVE_EDIT_LOG_LEVEL` | `info` | `trace`, `debug`, `info`, `warn` or `error`. Applies to the scaler too. |
| `observability.logging.structured.enabled` | `LIVE_EDIT_LOG_FORMAT` | `false` | `true` writes JSON logs (`json`); `false` writes human-readable logs (`pretty`). Set `true` for log pipelines. |
| `observability.tracing.enabled` | — | `false` | Export traces over OTLP/gRPC. Requires `observability.tracing.endpoint.grpc`. |
| `observability.tracing.endpoint.grpc` | `OTEL_EXPORTER_OTLP_ENDPOINT` | `""` | OTLP/gRPC collector URL for traces: `http://` or `https://`, as described below the table. |
| `observability.tracing.samplingRatio` | `OTEL_TRACES_SAMPLER_ARG` | `"1.0"` | Fraction of requests traced, `0` to `1`. A caller's own sampling decision is kept. |
| `observability.tracing.filter` | `LIVE_EDIT_TRACE_FILTER` | `live_edit=info` | Which spans are exported, independent of the log level. |
| `observability.metrics.enabled` | — | `false` | Also push metrics over OTLP/gRPC. The `/metrics` scrape stays on. |
| `observability.metrics.endpoint.grpc` | `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | `""` | OTLP/gRPC collector URL for metrics: `http://` or `https://`, as described below the table. |
| `observability.serviceName` | `OTEL_SERVICE_NAME` | unset (the chart name) | Service name on exported traces and metrics. |

Each collector URL is `http://` (plaintext) or `https://` (TLS) and names a host; any other value stops startup with `configuration error:` naming the variable.
Over `http://`, anyone who can watch the network between the pods and the collector can read the telemetry, including the request fields such as `base_path` on span events and the pod name in the `service` metric label.
Over `https://`, the service checks the collector's certificate against the CA bundle in its container image, read once at startup, and sends no client certificate; an unreadable bundle stops startup.
For a collector signed by a private CA, mount a bundle of the public roots plus your CA as {ref}`TLS <live-edit-backing-store-self-hosted-tls>` describes.
Without the chart, `SSL_CERT_FILE` or `SSL_CERT_DIR` can name that bundle instead; the `OTEL_EXPORTER_OTLP_CERTIFICATE` and client-certificate variables are not read.
Traces go only to `OTEL_EXPORTER_OTLP_ENDPOINT` and the metrics push only to `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`; `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is never read, and an empty variable turns its export off.
Exports are sent uncompressed: while trace export is on, any value of `OTEL_EXPORTER_OTLP_TRACES_COMPRESSION` or `OTEL_EXPORTER_OTLP_COMPRESSION`, `none` included, stops startup with `configuration error:` naming the variable, and likewise `OTEL_EXPORTER_OTLP_METRICS_COMPRESSION` or `OTEL_EXPORTER_OTLP_COMPRESSION` while metrics push is on; leave them unset.

(live-edit-configuration-backing-store-connection)=
## Backing-store connection

{ref}`Backing store <live-edit-backing-store>` explains the authentication modes and provider setup.

| Helm value | Environment variable | Default | Description |
|---|---|---|---|
| `envSecretName` | `LIVE_EDIT_VALKEY_URL` | `live-edit-env` | Secret whose `valkey-url` key holds the connection URL, used when `valkey.url` is empty. |
| `valkey.url.value` | `LIVE_EDIT_VALKEY_URL` | `""` | The connection URL inline, for a URL that carries no password. |
| `valkey.url.valueFrom` | `LIVE_EDIT_VALKEY_URL` | `{}` | The connection URL from any Kubernetes variable source, such as a `secretKeyRef`. |
| `valkey.credentials.accessTokenFile.enabled` | `LIVE_EDIT_VALKEY_AUTH_MODE` | `false` | `true` authenticates with a token file (`accessTokenFile`). Needs a `rediss://` URL and `serviceIdentity.accessTokenFile.enabled: true`. |
| `valkey.credentials.accessTokenFile.path` | `LIVE_EDIT_VALKEY_TOKEN_FILE` | `/etc/secrets/access-token/valkey_access_token` | Token file for that mode, in the same directory as the outbound token. |
| `valkey.credentials.accessTokenFile.username` | `LIVE_EDIT_VALKEY_TOKEN_USERNAME` | `""` | Valkey user name for that mode. Required. |
| `valkey.credentials.static.enabled` | `LIVE_EDIT_VALKEY_AUTH_MODE` | `false` | `static` mode, a password in the URL. Leaving all three modes off selects it too. |
| `valkey.credentials.awsIam.enabled` | `LIVE_EDIT_VALKEY_AUTH_MODE` | `false` | `true` authenticates with AWS IAM (`iam`). Needs a `rediss://` URL. Leave both modes off for a password in the URL (`static`). |
| `valkey.credentials.awsIam.service` | `LIVE_EDIT_VALKEY_IAM_SERVICE` | `memorydb` | `memorydb` or `elasticache`. |
| `valkey.credentials.awsIam.username` | `LIVE_EDIT_VALKEY_IAM_USERNAME` | `""` | IAM-enabled Valkey user. Required in `iam` mode. |
| `valkey.credentials.awsIam.region` | `LIVE_EDIT_VALKEY_IAM_REGION` | `""` | AWS region of the cluster. Required in `iam` mode. |
| `valkey.credentials.awsIam.clusterHost` | `LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST` | `""` | The cluster or cache **name**, not its endpoint host name. Required in `iam` mode. |

(live-edit-configuration-capacity)=
## Capacity and retention

{ref}`Scaling <live-edit-scaling>` explains how these relate to pod resources.
The service sizes its per-replica memory budgets (subscriptions, catch-ups of large layers, uploads, the structural-edit cache and, with backups on, large values waiting for the backup store) from `resources.limits.memory` at startup; no setting sets them.
`LIVE_EDIT_SUBSCRIBER_MAX_SUBSCRIPTIONS`, `LIVE_EDIT_CATCHUP_BUILD_MEMORY_BUDGET_MIB`, `LIVE_EDIT_UPLOAD_MEMORY_BUDGET_MIB` and `LIVE_EDIT_STRUCTURAL_CACHE_MAX_MIB` no longer exist: see {ref}`Removed settings <live-edit-operations-removed-settings>`.
Some settings on this page also take pod memory and raise the smallest memory limit that starts; {ref}`Starting size <live-edit-scaling-starting-size>` lists them.

| Helm value | Environment variable | Default | Description |
|---|---|---|---|
| `valkey.poolSize` | `LIVE_EDIT_VALKEY_POOL_SIZE` | `64` | Backing-store connections per replica for general commands. At least `8`. |
| `valkey.subscriberPoolSize` | `LIVE_EDIT_SUBSCRIBER_POOL_SIZE` | `256` | Stream-reader connections per replica. On a clustered store, also how many distinct layers a replica serves at once. |
| `valkey.layerStreamMaxlen` | `LIVE_EDIT_LAYER_STREAM_MAXLEN` | `4096` | Recent updates the backing store keeps per layer. `0` removes the count limit. |
| `valkey.layerStreamMaxBytes` | `LIVE_EDIT_LAYER_STREAM_MAX_BYTES` | `134217728` | Bytes of recent updates kept per layer (128 MiB); the smaller limit applies. Lower it to save backing-store memory. `0` removes the byte limit. |
| `valkey.memoryBytes` | `LIVE_EDIT_VALKEY_MEMORY_BYTES` | unset | The store's total memory in bytes, as the provider states it. Set it only where the store reports no memory limit, in practice Azure Managed Redis; a reported limit wins. Update it after a resize. Below `100000000` stops startup. |
| `valkey.extReleaseMaxBytes` | `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` | unset (derived from the store's memory) | When deleted large values awaiting clean-up exceed this many bytes, uploads carrying large values wait, then fail with a retryable `UNAVAILABLE`. `0` turns the limit off. |
| `resources.limits.memory` | — | `5Gi` | Pod memory limit; required. Every per-replica memory budget follows it, and below the size your settings need the pod stops at startup naming that size. Resizing it in place restarts the container. {ref}`Starting size <live-edit-scaling-starting-size>` |
| `resources.limits.cpu` | — | `2` | Pod CPU limit. |
| `resources.requests.memory` | — | 70% of `resources.limits.memory` (`3584Mi` at `5Gi`) | Pod memory request, used for scheduling; no memory budget depends on it. Unset, it is 70% of the limit, rounded down to whole MiB. A set request is used as written; the chart refuses one above the limit. |
| `resources.requests.cpu` | `LIVE_EDIT_SCALER_CPU_REQUEST` | `1` | Pod CPU request; the autoscaler's CPU target is a percentage of it. |

(live-edit-configuration-requests-and-uploads)=
## Requests and uploads

| Helm value | Environment variable | Default | Description |
|---|---|---|---|
| `rpc.deadlineMs` | `LIVE_EDIT_RPC_DEADLINE_MS` | `30000` | Deadline for every call except the two long-lived streams. Raise `terminationGracePeriodSeconds` with it. |
| `layer.maxSize` | `LIVE_EDIT_MAX_LAYER_SIZE` | `104857600` | Largest single upload, in bytes (100 MiB); larger ones fail `INVALID_ARGUMENT`. Raising it takes pod memory from the catch-up budget, and a memory limit too small for it stops startup. |
| `layer.uploadAdmissionWaitMs` | `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS` | unset (2000 at the default deadline) | How long an upload waits for the replica's upload memory. Must be below `rpc.deadlineMs`. `0` never waits. |
| `layer.extValueVerify` | `LIVE_EDIT_EXT_VALUE_VERIFY` | `enforce` | `enforce` refuses an upload whose large value does not match its declared size or hash; `observe` logs and stores it. Use `observe` only while rolling out clients. |

(live-edit-configuration-presence-and-shutdown)=
## Presence and shutdown

| Helm value | Environment variable | Default | Description |
|---|---|---|---|
| `presence.keepaliveIntervalMs` | `LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS` | `15000` | Keepalive interval for clients that opt in; one silent for three intervals leaves its session. `0` turns it off. `1`–`999` is refused. |
| `shutdown.timeoutSecs` | `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS` | `5` | How long a stopping pod keeps serving after leaving the load balancer. |
| `shutdown.streamEndRate` | `LIVE_EDIT_SHUTDOWN_STREAM_END_RATE` | `25` | Streams a stopping pod ends per second, so clients reconnect gradually. `0` ends all at once. |
| `shutdown.streamEndMaxWindowSecs` | `LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS` | `20` | Longest time spent ending streams. |
| `shutdown.overallBudgetSecs` | `LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS` | `30` | Cleanup time after the server stops. At least `shutdown.timeoutSecs` + 15. |
| `terminationGracePeriodSeconds` | — | `80` | Must cover the shutdown settings and `rpc.deadlineMs`; the chart refuses a value that does not and names the minimum. |

(live-edit-configuration-checkpoints)=
## Checkpoints

{ref}`Durability <live-edit-durability>` explains when to use checkpoints and which adapter to choose.
With the chart defaults the service refuses to start until you choose an adapter or turn checkpoints off.

| Helm value | Environment variable | Default | Description |
|---|---|---|---|
| `checkpoint.enabled` | `LIVE_EDIT_CHECKPOINT_ENABLED` | `true` | Write periodic backups of every layer. With `false`, a backing store reporting append-only persistence off refuses startup. |
| `checkpoint.adapter` | `LIVE_EDIT_CHECKPOINT_ADAPTER` | `Disabled` | `StorageApi`, `LocalDisk`, `InMemory` (testing only) or `Disabled`. |
| `checkpoint.storageApiEndpoint` | `LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT` | `""` | Storage API URL. Required for `StorageApi`; include the scheme. |
| `checkpoint.basePath` | `LIVE_EDIT_CHECKPOINT_BASE_PATH` | `""` | Storage path under which backups are written. Required for `StorageApi`. |
| `checkpoint.localDiskPath` | `LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH` | `""` | Directory on a persistent volume. Required for `LocalDisk`. |
| `checkpoint.interval` | `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS` | `60` | Longest gap between full backups of a changing layer. |
| `checkpoint.quiesce` | `LIVE_EDIT_CHECKPOINT_QUIESCE_SECS` | `30` | Back up a layer early once it has been idle this long. |
| `checkpoint.retainGenerations` | `LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS` | `2` | Full backups kept per layer. |
| `checkpoint.tail.enabled` | `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED` | `true` | Take incremental backups between full ones. |
| `checkpoint.counterSafetyMargin` | `LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN` | `1000000` | Headroom above the last backed-up layer id when the service recovers from a backing-store data loss. Raise it if more than a million layers could be created during a backup-store outage. Lowering it risks reused layer ids. `1` to `1000000000`. |

(live-edit-configuration-autoscaling)=
## Autoscaling

{ref}`Scaling <live-edit-scaling>` explains the replica model and how to size these.
The scaler settings apply when `autoscaling.mode` is `scaler`.

| Helm value | Environment variable | Default | Description |
|---|---|---|---|
| `autoscaling.enabled` | — | `true` | Let an autoscaler own the replica count. `false` uses `replicaCount` (default `3`). |
| `autoscaling.mode` | — | `scaler` | `scaler` (the service's own autoscaler) or `hpa` (a HorizontalPodAutoscaler on CPU). |
| `autoscaling.minReplicas` | `LIVE_EDIT_SCALER_MIN_REPLICAS` | `3` | Fewest replicas. |
| `autoscaling.maxReplicas` | `LIVE_EDIT_SCALER_MAX_REPLICAS` | `10` | Most replicas. Your backing store's connection limit must allow twice this many. |
| `autoscaling.targetCPUUtilizationPercentage` | `LIVE_EDIT_SCALER_CPU_TARGET_PERCENT` | `80` | Target CPU use, as a percentage of `resources.requests.cpu`. In `scaler` mode, `0` turns the CPU target off. |
| `autoscaling.targetMemoryUtilizationPercentage` | — | `null` | Memory target, as a percentage of `resources.requests.memory`; `hpa` mode only. |
| `autoscaling.scaler.pollIntervalSeconds` | `LIVE_EDIT_SCALER_POLL_INTERVAL_SECS` | `5` | Seconds between load reads. |
| `autoscaling.scaler.catchupTargetSeconds` | `LIVE_EDIT_SCALER_CATCHUP_TARGET_SECS` | `10` | Add replicas so waiting subscribers are served within this time. |
| `autoscaling.scaler.occupancyHeadroomPercent` | `LIVE_EDIT_SCALER_OCCUPANCY_HEADROOM_PERCENT` | `80` | Add replicas before subscriptions pass this share of each replica's limit. |
| `autoscaling.scaler.scaleDown.windowSeconds` | `LIVE_EDIT_SCALER_SCALE_DOWN_WINDOW_SECS` | `600` | How long demand must stay lower before a replica is removed. |
| `autoscaling.scaler.scaleDown.intervalSeconds` | `LIVE_EDIT_SCALER_SCALE_DOWN_INTERVAL_SECS` | `60` | Least time between removals, one replica each. |
| `autoscaling.scaler.podDeletionCost` | `LIVE_EDIT_SCALER_POD_DELETION_COST` | `false` | Remove the replica with the fewest subscriptions first. |
| `autoscaling.scaler.rbac.create` | — | `true` | Create the scaler's Role and RoleBinding. Set `false` if you grant them yourself. |

(live-edit-configuration-advanced)=
## Advanced tuning

Leave these at their defaults unless a topic page, a startup error or NVIDIA support directs you to change one. Startup errors name the environment variable; find its Helm value here.

| Helm value | Environment variable | Default |
|---|---|---|
| `authorization.cacheMaxEntries` | `LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES` | `10000` |
| `authorization.maxAttempts` | `LIVE_EDIT_AUTH_MAX_ATTEMPTS` | `3` |
| `authorization.retryInitialDelayMs` | `LIVE_EDIT_AUTH_RETRY_INITIAL_DELAY_MS` | `100` |
| `observability.tracing.batch.maxQueueSize` | `OTEL_BSP_MAX_QUEUE_SIZE` | `2048` |
| `observability.tracing.batch.maxExportBatchSize` | `OTEL_BSP_MAX_EXPORT_BATCH_SIZE` | `512` |
| `observability.tracing.batch.scheduleDelayMs` | `OTEL_BSP_SCHEDULE_DELAY` | `5000` |
| `valkey.connectTimeoutSecs` | `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` | `30` |
| `valkey.commandTimeoutMs` | `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` | `5000` |
| `valkey.reconnectMaxDelayMs` | `LIVE_EDIT_VALKEY_RECONNECT_MAX_DELAY_MS` | `30000` |
| `valkey.subscriberMaxConcurrentCatchups` | `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS` | `8` |
| `valkey.subscriberAdmissionWaitMs` | `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS` | `20000` |
| `valkey.subscriberAdmissionMaxWaiters` | `LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS` | `0` (unset: the subscription limit) |
| `valkey.maintenancePoolSize` | `LIVE_EDIT_MAINTENANCE_POOL_SIZE` | `1` |
| `valkey.catchupSnapshotReadConcurrency` | `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY` | `1` |
| `valkey.catchupFencedReadConcurrency` | `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY` | `1` |
| `valkey.snapshotPoolSize` | `LIVE_EDIT_SNAPSHOT_POOL_SIZE` | unset (`6`) |
| `valkey.snapshotAtomicReadMaxItems` | `LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS` | `37554` |
| `valkey.snapshotAtomicReadMaxBytes` | `LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES` | `604619` |
| `valkey.credentials.accessTokenFile.refreshLeadSecs` | `LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS` | `180` |
| `userInfo.negativeTtlSecs` | `LIVE_EDIT_AUTH_USERINFO_NEGATIVE_TTL_SECS` | `5` |
| `serviceIdentity.clientCredentials.enabled` | `LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED` | `false` (no effect; leave it) |
| `rpc.sessionListTimeoutMs` | `LIVE_EDIT_SESSION_LIST_TIMEOUT_MS` | `2000` |
| `layer.uploadAdmissionMaxWaiters` | `LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS` | `64` |
| `layer.updateRetryMaxAttempts` | `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` | `128` |
| `layer.updatePerObjectConcurrency` | `LIVE_EDIT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY` | `16` |
| `layer.updateLayerBackend` | `LIVE_EDIT_UPDATE_LAYER_BACKEND` | `lua` |
| `layer.catchupSnapshotEnabled` | `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED` | `true` |
| `layer.structuralCacheAuditRate` | `LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_RATE` | `0.001` |
| `layer.structuralCacheAuditAboveCeiling` | `LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_ABOVE_CEILING` | `true` |
| `session.retryMaxAttempts` | `LIVE_EDIT_SESSION_RETRY_MAX_ATTEMPTS` | `32` |
| `session.retryInitialDelayMs` | `LIVE_EDIT_SESSION_RETRY_INITIAL_DELAY_MS` | `1` |
| `presence.broadcastCapacity` | `LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY` | `4096` |
| `presence.sweeperIntervalSecs` | `LIVE_EDIT_SWEEPER_INTERVAL_SECS` | `15` |
| `gc.extReconcileIntervalSecs` | `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS` | `3600` |
| `checkpoint.tick` | `LIVE_EDIT_CHECKPOINT_TICK_SECS` | `10` |
| `checkpoint.seqSafetyMargin` | `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` | `1000` |
| `checkpoint.tail.maxChain` | `LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN` | `8` |
| `checkpoint.tail.maxEntries` | `LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES` | `4096` |
| `checkpoint.tail.maxBytes` | `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` | `67108864` |
| `checkpoint.credentials.accessTokenFile.pollIntervalSecs` | `LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS` | `15` |
| `checkpoint.connectTimeoutMs` | `LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS` | `5000` |
| `checkpoint.requestTimeoutMs` | `LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS` | `30000` |
| `checkpoint.countersSyncInterval` | `LIVE_EDIT_CHECKPOINT_COUNTERS_SYNC_SECS` | `60` |
| `checkpoint.extMirrorQueueCapacity` | `LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY` | `1000` |
| `checkpoint.extMirrorWorkers` | `LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS` | `2` (`1` to `16`) |
| `checkpoint.reconcileIntervalSecs` | `LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS` | unset (at least `3600`) |
| `allocator.mallocConf` | `_RJEM_MALLOC_CONF` | unset; diagnosis only, leave it unset |

(live-edit-configuration-values-map)=
## Map of `values.yaml`

| Section | Controls | Covered on |
|---|---|---|
| `image`, `serviceAccount`, `podAnnotations`, `podLabels`, `podSecurityContext`, `securityContext` | image and pod identity | {ref}`Deployment <live-edit-deployment>` |
| `service` | Service object and ports | this page, {ref}`Deployment <live-edit-deployment>` |
| `authorization`, `userInfo`, `serviceIdentity` | request authorization, user identity, outbound tokens | {ref}`Authentication <live-edit-authentication>` |
| `observability`, `serviceMonitor` | logs, traces, metrics export, Prometheus scraping | {ref}`Monitoring <live-edit-monitoring>` |
| `envSecretName`, `valkey` | backing-store connection, authentication and capacity | {ref}`Backing store <live-edit-backing-store>` |
| `rpc`, `layer`, `session`, `presence`, `shutdown`, `gc` | request handling, uploads, sessions, shutdown | this page |
| `checkpoint` | backups and recovery | {ref}`Durability <live-edit-durability>` |
| `allocator` | memory allocator diagnosis | leave unset |
| `resources`, `autoscaling`, `podDisruptionBudget`, `replicaCount` | pod size and replica count | {ref}`Scaling <live-edit-scaling>` |
| `startupProbe`, `livenessProbe`, `readinessProbe`, `terminationGracePeriodSeconds`, `strategy` | health checks and rollouts | {ref}`Operations <live-edit-operations>` |
| `httpProxy`, `networkPolicy` | ingress and network access | {ref}`Deployment <live-edit-deployment>` |
| `volumes`, `volumeMounts`, `nodeSelector`, `tolerations`, `affinity` | standard pod placement and storage | Kubernetes documentation |
| `tests`, `smokeTest` | smoke tests (`helm test`) | {ref}`Deployment <live-edit-deployment>` |
| `nameOverride`, `fullnameOverride`, `minReadySeconds`, `revisionHistoryLimit`, `conformance` | resource names, rollout pacing, release-validation hooks | leave at default |

(live-edit-configuration-see-next)=
## See next

{ref}`Backing store <live-edit-backing-store>`: choose and connect the Valkey backing store.
