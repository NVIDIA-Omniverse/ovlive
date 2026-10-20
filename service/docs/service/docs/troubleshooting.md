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

(live-edit-troubleshooting)=
# Troubleshooting

This page lists common startup, readiness and client-access symptoms of a Live Edit deployment, what to check and how to fix them.
It assumes a deployment installed as in {ref}`Deployment <live-edit-deployment>`; alerts and their runbooks are on {ref}`Monitoring <live-edit-monitoring>`.
Backing-store, capacity and rollout symptoms are on {ref}`Troubleshooting the backing store, capacity and rollouts <live-edit-troubleshooting-capacity>`.

(live-edit-troubleshooting-startup)=
## Startup and readiness

| Symptom | Check and fix |
|---|---|
| Pod restarts with `live-edit refuses to start with no durability configured` | Choose a durability mode ({ref}`Startup durability check <live-edit-durability-startup-check>`). |
| Pod exits with `startup refused: the backing store refused the atomic snapshot read`, or warns `atomic snapshot-read self-test could not run` | The store failed, or could not run, the startup script check. Follow {ref}`Startup check <live-edit-backing-store-startup-check>`. |
| Warning `could not verify Valkey AOF state` | Expected on managed stores with checkpoints off. Confirm persistence in the provider's console, or enable checkpoints. |
| Pod stays unready | Follow {ref}`A pod stays unready <live-edit-troubleshooting-unready>`. |
| Warning `service token file still absent or unparseable after startup fast-poll window` | The token provider has not written a readable file at the warning's `path`. Check the provider's logs and that both containers mount `/etc/secrets/access-token`. |
| Warning `cached service token expired before a refreshed one appeared` | The token provider refreshes after the token expires. With either `role` the pod turns unready, and with `role=storage-api` backups also fail; a fault shared by every sidecar takes the whole fleet out of rotation. Raise the provider's refresh lead ({ref}`Service identity <live-edit-authentication-service-identity>`). |
| Warning `replacement service token was picked up with less headroom` | Backing-store connections reconnect once per token rotation. Raise the provider's refresh lead, or raise `valkey.credentials.accessTokenFile.refreshLeadSecs` to at least the value the warning names. |
| Warning naming `LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED=true` | That mode is not supported and has no effect. Set `serviceIdentity.clientCredentials.enabled: false` and use the token file. |
| `helm upgrade` fails to render, naming a value | The chart refuses an inconsistent combination. Set the named value as the message says. |
| Pod exits with a `configuration error:` naming `this pod's memory limit` and the MiB `this configuration needs` | The memory limit is too small for the settings. Raise `resources.limits.memory` to at least the size the error names, which costs nothing else. Failing that, lower the settings the error lists; for the terms of the 20 % memory margin, lower `layer.maxSize` or, when the error names `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED`, turn off incremental backups (`checkpoint.tail.enabled: false`). Without incremental backups, a newly created or just-restored layer has no backup of its current state until the next full backup pass, and after a restore that pass can come one backup interval (`checkpoint.interval`) later ({ref}`Incremental backups and retention <live-edit-durability-incremental>`). Do not turn off `checkpoint.enabled` or set `checkpoint.adapter` to `Disabled` for this: that removes backups. |
| Pod exits with `configuration error: no pod memory limit can fit this configuration` | The settings need more memory than any limit gives. Lower the settings the error tells you to lower: `valkey.catchupFencedReadConcurrency` to at most the value it gives, the largest fixed terms' settings, or, for the 20 % memory margin's terms, `layer.maxSize`. When the error names `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED`, turning off incremental backups also helps, at the cost the row above states. Do not turn off `checkpoint.enabled` or set `checkpoint.adapter` to `Disabled` for this: that removes backups. |
| Pod exits with `configuration error:` naming `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` or a compression variable | Set the collector URL to `http://` or `https://` with a host, or unset the compression variable. An error saying `TLS to the collector could not be set up` means the CA bundle is unreadable: fix the mounted bundle ({ref}`Logs, traces and metrics export <live-edit-configuration-observability>`). |
| Pod exits with `configuration error:` naming a variable that `is no longer read` | The setting was removed. Remove it from the pod's environment ({ref}`Removed settings <live-edit-operations-removed-settings>`). |
| `helm upgrade` fails to render with `… is set to …, but this chart no longer sets it` | Remove the named value from your values file ({ref}`Removed settings <live-edit-operations-removed-settings>`). |
| `helm upgrade` fails to render with `resources.limits.memory must be set` | Set `resources.limits.memory` ({ref}`Starting size <live-edit-scaling-starting-size>`). |
| Pod exits with `invalid value for LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS: must be greater than 0` | `0` (no limit) is refused. Set `valkey.subscriberMaxConcurrentCatchups` to a count, about 4 per CPU. |
| Warning `no readable pod memory limit` | The replica read no memory limit and sized itself as a 5 GiB pod. By the warning's `reason`: `unlimited`, set `resources.limits.memory`; `not_found`, check that `/sys/fs/cgroup` is mounted in the container with its memory controller enabled; `unreadable`, fix the file and error the reason names. |

(live-edit-troubleshooting-unready)=
## A pod stays unready

1. Read the server log with `kubectl logs {POD} -n storage-apis -c live-edit-service`, and the token sidecar's with `kubectl logs {POD} -n storage-apis -c live-edit-service-token-provisioner`.
2. `Valkey connect attempt failed, retrying`: the `error` field names the cause.
   - A name-resolution error: the store's host does not resolve from the pod. On Azure Managed Redis, link the `privatelink.redis.azure.net` zone to the cluster's virtual network ({ref}`Connect the private network <live-edit-backing-store-azure-network>`).
   - A timeout or refused connection: a network rule blocks the store. On Azure Managed Redis, allow port 10000 and ports 8500–8599 outbound from the pods, including in your NetworkPolicy.
   - `WRONGPASS`: follow {ref}`Troubleshooting WRONGPASS <live-edit-backing-store-azure-wrongpass>`.
3. `service token file still absent or unparseable`: read the sidecar's log.
   - An Entra `AADSTS70021` error means no federated credential matches the pod's token: check the credential's issuer (the cluster's OIDC issuer URL), audience (`api://AzureADTokenExchange`) and subject, which must be `system:serviceaccount:{NAMESPACE}:{SERVICE_ACCOUNT}`, where `kubectl get deployment live-edit-service -n storage-apis -o jsonpath='{.spec.template.spec.serviceAccountName}'` prints the account.
   - Any other error: the token provider's configuration is invalid, or it writes to paths other than the chart's token paths ({ref}`Configure the token sidecar <live-edit-authentication-configure-the-token-sidecar>`).
4. `counter-seed counters-sidecar fetch failed`: with checkpoints on, the pod cannot read the backup store. Check the service token and that the pod reaches `checkpoint.storageApiEndpoint`.
5. None of these: check the startup refusals in {ref}`Startup and readiness <live-edit-troubleshooting-startup>` and what readiness waits for in {ref}`Health checks <live-edit-monitoring-health>`.

(live-edit-troubleshooting-clients)=
## Clients and access

| Symptom | Check and fix |
|---|---|
| Uploads fail `INVALID_ARGUMENT` naming `valueOrExtHash` or `extValueSize` | A client computes a large value's hash or size wrongly. Until it is fixed, `layer.extValueVerify: observe` accepts such values; then return to `enforce`. |
| Uploads fail `INVALID_ARGUMENT` with a message starting `delta too large` or naming `ReorderChildren` | A client sends a delta over the per-delta limits (99,999 entries, or 100,000 children in one reorder). No setting raises them: the client must split its content into smaller deltas ({ref}`Deltas that are too large <live-edit-api-validation-too-large>`). |
| A user whose access was revoked can still edit | Access decisions are cached ({ref}`User-token authorization <live-edit-authentication-user-token-authorization>`), and streams are checked only when they open, so changes apply on reconnect. |
| A participant who left stays shown | Only clients that send `JoinSession` keepalives are removed when they vanish. Check that `presence.keepaliveIntervalMs` is not `0` and that the client sends keepalives. |
| `live_edit_presence_keepalive_timeouts_total` rises in bursts with no network trouble | Live clients may be removed falsely. Set `presence.keepaliveIntervalMs: 0` while the client is fixed. |
| `ListSessions` is unexpectedly empty for a path, or a session seems missing | The client may spell the path differently from the one the session was created under. The INFO line `CreateSession canonicalized base_path` and the DEBUG line `ListSessions canonicalized base_path` show `received` and `canonical`; set `observability.logging.level: debug` to see the second. |

(live-edit-troubleshooting-inspect)=
## Inspect a deployment

These commands read the deployment without changing it:

```bash
kubectl logs deployment/live-edit-service -n storage-apis -c live-edit-service --tail=100
kubectl describe pods -n storage-apis -l app.kubernetes.io/instance=live-edit-service
```

(live-edit-troubleshooting-see-next)=
## See next

{ref}`Troubleshooting the backing store, capacity and rollouts <live-edit-troubleshooting-capacity>`: backing-store, capacity and rollout symptoms.
