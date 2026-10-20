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

(live-edit-backing-store)=
# Live Edit backing store

Live Edit stores all session and layer state in Valkey.
The store is the source of truth, not a cache: every replica reads and writes the same store, and losing it loses live edits that are not in a checkpoint.
This page lists what the store must provide, how Live Edit connects and authenticates to it, and how to size it.
Read it before {ref}`Deployment <live-edit-deployment>`, then follow the page for your platform:

| You run the store… | Follow |
|---|---|
| yourself, in your Kubernetes cluster | {ref}`Self-hosted Valkey <live-edit-backing-store-self-hosted>` |
| on Azure | {ref}`Azure Managed Redis <live-edit-backing-store-azure>` |
| on AWS | {ref}`Amazon MemoryDB or ElastiCache <live-edit-backing-store-aws>` |

(live-edit-backing-store-requirements)=
## Requirements

Every store must meet all of these:

- **Engine:** Valkey 7.2 or newer, or Redis OSS 7.x on a compatible managed service.
- **One deployment per keyspace:** give each Live Edit deployment its own store or keyspace; all of its replicas use the same one.
- **No eviction:** set `maxmemory-policy noeviction`.
  A full store then refuses uploads with the retryable `UNAVAILABLE` and applies nothing, instead of silently deleting committed layer data.
  On managed platforms this is a create-time choice.
- **A memory limit:** on a store you run yourself, set `maxmemory` below the container or host memory limit.
- **Connection limit:** the store must accept connections from twice your maximum replica count (see {ref}`Connections <live-edit-backing-store-connections>`).
- **Cluster mode:** a clustered store must keep each multi-key operation's keys in one hash slot.
  Live Edit's keys carry hash tags that guarantee this, so no key or application change is needed; connect with the cluster URL scheme.
- **Script reads:** a Lua script must be allowed to read keys it did not declare, in its declared key's hash slot.
  Standard Valkey and Redis, Azure Managed Redis and Amazon MemoryDB allow it.
  Each replica checks this at startup (see {ref}`Startup check <live-edit-backing-store-startup-check>`).
- **Pub/sub:** allow `PUBLISH` inside Lua scripts, and on a cluster deliver every published message to subscribers on every node, in publish order.
  Session presence depends on it.
- **Clocks:** keep cluster nodes' clocks synchronized (NTP or the provider's equivalent), within 15 seconds of each other.
- **TLS:** use TLS for any non-loopback connection.
  The server certificate must chain to a public root, or you must mount a CA bundle holding your private CA and the public roots into the Live Edit container (see {ref}`TLS <live-edit-backing-store-self-hosted-tls>`).
- **Network:** let service pods reach every node and port a cluster advertises, not only its bootstrap endpoint.
- **Access control:** grant the service's user `+@all` commands on the `le:` keyspace and channels, for example the ACL `on ~le:* &le:* +@all`.
  Restrict keys and channels, never commands: removing commands breaks cluster discovery and rarely used paths.
- **Durability:** configure store persistence, checkpoints, or both, as {ref}`Durability <live-edit-durability>` describes.
  Live Edit refuses to start when it can confirm that both are off.

(live-edit-backing-store-startup-check)=
### Startup check

Before it reports ready, each replica runs one script that reads an undeclared key in its declared key's slot.

- If the store refuses that read, the replica logs an error naming this requirement and exits.
  Use a store that allows it; there is no workaround.
- If the store refuses the script itself, the replica exits with an error naming the script and the store's error.
  Check the ACL.
- If the store cannot answer (loading, busy, cluster down, or no reply within `valkey.commandTimeoutMs`), the replica logs a warning and starts anyway.
  The check is not repeated, so if the warning appears, confirm the store meets the requirement.

(live-edit-backing-store-connection-url)=
## Connection URL

Choose the scheme for the topology the endpoint presents:

| Endpoint | URL form |
|---|---|
| Single primary, TLS | `rediss://{USER}:{PASSWORD}@{HOST}:{PORT}` |
| Cluster, TLS (Azure Managed Redis, MemoryDB, ElastiCache, self-hosted cluster) | `rediss-cluster://{USER}:{PASSWORD}@{HOST}:{PORT}` |

A single-primary URL against a multi-shard cluster fails on cross-shard keys.
Use `redis://` or `redis-cluster://` only for a local store without TLS.
URL-encode reserved characters in credentials.

A URL that carries a credential goes in the `live-edit-env` Secret under key `valkey-url`, as {ref}`Deployment <live-edit-deployment>` shows.
A URL without one (the passwordless modes below) can be set inline:

```yaml
valkey:
  url:
    value: "rediss-cluster://{HOST}:{PORT}"
```

`valkey.url.valueFrom` takes any Kubernetes environment source, such as a ConfigMap key, instead.

(live-edit-backing-store-auth)=
## Authentication

Enable at most one of `valkey.credentials.static`, `accessTokenFile` or `awsIam`; the chart refuses to render with more than one.
With none enabled, Live Edit uses the credentials in the URL.
Prefer a passwordless mode in production: it removes the long-lived secret.

| Mode | Use with | Credential |
|---|---|---|
| `static` (default) | self-hosted Valkey, development, access keys | user and password in the URL |
| `accessTokenFile` | Azure Managed Redis with Microsoft Entra ID, or any store that accepts a renewable bearer token | token written to a file by your token provisioner |
| `awsIam` | Amazon MemoryDB, ElastiCache for Valkey | AWS IAM token generated by the service from the pod's role |

(live-edit-backing-store-auth-static)=
### Password or ACL credentials

Put the full URL, with user and password, in the `live-edit-env` Secret.
Changing the Secret does not restart pods: roll the deployment after rotating the credential.

(live-edit-backing-store-auth-token-file)=
### Access-token file

```yaml
valkey:
  url:
    value: "rediss-cluster://{HOST}:{PORT}"
  credentials:
    accessTokenFile:
      enabled: true
      username: "{PRINCIPAL_ID}"
      path: /etc/secrets/access-token/valkey_access_token
```

Your token provisioner writes the token to `path`, with the audience the store accepts, in the {ref}`token file format <live-edit-authentication-token-file-format>`.
This mode requires `serviceIdentity.accessTokenFile.enabled: true`; the chart refuses to render without it.
On Azure Managed Redis, replace the token as {ref}`Token refresh <live-edit-backing-store-azure-token-refresh>` says.

(live-edit-backing-store-auth-aws-iam)=
### AWS IAM

```yaml
valkey:
  url:
    value: "rediss-cluster://{CLUSTER_ENDPOINT}:6379"
  credentials:
    awsIam:
      enabled: true
      service: memorydb          # or elasticache
      username: "{ACL_USER}"
      region: "{AWS_REGION}"
      clusterHost: "{CLUSTER_NAME}"
```

`clusterHost` is the cluster or cache name, not its endpoint hostname; with the hostname, authentication fails.
The pod gets its AWS credentials from its ServiceAccount's IAM role (`serviceAccount.annotations."eks.amazonaws.com/role-arn"`).
{ref}`The AWS page <live-edit-backing-store-aws>` creates the user and role.

(live-edit-backing-store-sizing)=
## Sizing

Size memory first, then per-shard CPU, then check connections.
The platform pages give starting sizes.

(live-edit-backing-store-memory)=
### Memory

Memory grows with the number of layers being edited at once, not with sessions or users.
For each such layer, plan for:

- **Layer state:** about 1.2 MB for a layer of 10,000 fields; about 12 MB for 100,000 fields.
- **Recent updates:** the layer keeps its last 4,096 updates, capped at 128 MiB (`valkey.layerStreamMaxBytes`), plus about 20 % overhead.
  That is a few MiB for small edits and about 154 MiB for layers of large updates.
  Read your mean update size from `live_edit_update_layer_relay_bytes`.
- **Large values** (meshes, textures and other external values): stored once per distinct content, plus replaced values and the values of deleted layers awaiting clean-up, usually for seconds; a burst of very large deletions holds them longer.

To save memory, lower `valkey.layerStreamMaxBytes` before buying more.
Keep a busy layer's update bytes per checkpoint interval well inside it, or incremental backups lose coverage ({ref}`Durability <live-edit-durability>`).

**Headroom:** give the store about twice your expected peak memory.
With `noeviction`, a full store refuses writes.
On the first refused write, Live Edit deletes the released large values the store holds.
A store they filled recovers within seconds; a store full of layer data stays full until you add memory or, on self-hosted Valkey or Azure Managed Redis, until clients delete layers.
While it is full, Live Edit's own clean-up writes a little past `maxmemory`: about 1.5 MB at typical release rates, more during a large backlog, for about two minutes.
If `maxmemory` sits right at the container or host memory limit, that overshoot can get the server killed for running out of memory.
Keep `maxmemory` a few MB or more below that limit to absorb it.

**Memory settings:** self-hosted Valkey, MemoryDB and ElastiCache report their memory limit to Live Edit, so leave `valkey.memoryBytes` unset on them.
Azure Managed Redis does not: set `valkey.memoryBytes` to the cache's total memory, as {ref}`its size section <live-edit-backing-store-azure-size>` shows.

**Is it filling up?** Watch `live_edit_store_memory_used_ratio`; {ref}`Store memory <live-edit-monitoring-store-memory>` gives the levels at which to grow the store and how to alert on it.

(live-edit-backing-store-cpu)=
### CPU

If a shard's CPU stays near one full core, add shards; a larger node does not help.
One very busy layer always stays on one shard.

(live-edit-backing-store-connections)=
### Connections

A rollout briefly doubles the replica count, so the store must accept connections from twice `autoscaling.maxReplicas` replicas.
At the chart defaults:

| Store | What to do |
|---|---|
| Self-hosted Valkey | set `maxclients 20000`, as {ref}`the self-hosted page <live-edit-backing-store-self-hosted-config>` does |
| Azure Managed Redis | keep `autoscaling.maxReplicas` within {ref}`the size table <live-edit-backing-store-azure-size>` |
| MemoryDB, ElastiCache for Valkey | nothing: the limit is 65,000 per node |

If the store runs out of connections, new pods cannot connect, the rollout stalls, and some subscriptions fail until it ends.

(live-edit-backing-store-moving-data)=
## When the store moves data between shards

A clustered store moves data between shards when you add or remove shards, and a managed store does so when it scales or rebalances.
How Live Edit is affected depends on how the store moves it:

- **Key-by-key moves** (the classic reshard) refuse writes to the layers and sessions in each moving slot, with the retryable `UNAVAILABLE`, until that slot's move ends, so a slow move means long refusals. Catch-ups and backups of those layers fail and are retried, so expect backups to lag during the move.
- **Atomic slot migration** (Valkey 9.0 or later) keeps serving the moving slots and pauses writes only briefly at each cutover.

Clients retry the refusals; other layers and sessions are unaffected.
Live subscriptions to a moving layer pause until its move ends.
If a session's move lasts longer than about a minute, other participants can see a member leave and rejoin.

During a move expect `live_endpoint_calls_total{grpc_code="UNAVAILABLE"}` to rise for the affected calls, a few `live_edit_valkey_reconnects_total`, and, in long moves, `live_edit_presence_lease_refresh_failures_total` and `live_edit_presence_rejoins_total`.
They stop when the move ends.
On a self-hosted cluster, move slots as {ref}`Adding shards <live-edit-backing-store-self-hosted-add-shards>` describes.

(live-edit-backing-store-watch)=
## What to watch

| Signal | Act when |
|---|---|
| `live_edit_store_memory_used_ratio` | as {ref}`Store memory <live-edit-monitoring-store-memory>` says |
| `live_edit_store_write_refused_total{reason="out_of_memory"}` | any increase: the store is full; add memory, even if the refusals stop by themselves |
| `live_edit_gc_oom_collections_total{outcome="refused"}` | any increase: the service cannot free memory on this store while it is full; add memory |
| `live_edit_ext_release_pending_bytes` against `live_edit_ext_release_budget_bytes` | pending nears the budget: clean-up is falling behind, and uploads carrying large values start to wait |
| `live_edit_valkey_errors_total` | a sustained rate: connection, authentication or timeout faults |
| `live_edit_valkey_consumer_permanent_exits_total` | above zero: a credential or endpoint problem |

Alert rules are on {ref}`Monitoring <live-edit-monitoring>`; metric details are on {ref}`Metrics reference <live-edit-metrics-reference>`.

(live-edit-backing-store-see-next)=
## See next

Set up the store on your platform: {ref}`self-hosted <live-edit-backing-store-self-hosted>`, {ref}`Azure <live-edit-backing-store-azure>` or {ref}`AWS <live-edit-backing-store-aws>`.
