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

(live-edit-scaling)=
# Scaling and sizing

This page covers how many replicas to run, how to size their pods, how autoscaling behaves and the per-replica limits clients can reach.
It assumes a deployment installed as in {ref}`Deployment <live-edit-deployment>`; size the backing store separately ({ref}`Backing store <live-edit-backing-store>`).

(live-edit-scaling-replica-model)=
## Replica model

Every replica serves every request, so clients need no session affinity.
Long-lived streams (`JoinSession` and `SubscribeReadLayer`) stay on the replica that accepted them.
A new replica serves only the joins that arrive after it is ready; it takes no streams or queued joins from busy replicas.
When a replica shuts down, its clients reconnect to the others, spread over up to 20 seconds.

The unit of load is a **subscription**: one client with one layer open.
A user with ten layers open holds ten.
When a client joins a layer, the replica first sends it the layer's current state (a catch-up), which is where most of a join's CPU and memory goes.

(live-edit-scaling-starting-size)=
## Starting size

Start from the chart defaults and adjust from what you measure under representative load.

| Setting | Chart default | Adjust when |
|---|---|---|
| Replicas | 3 to 10 (`autoscaling.minReplicas`, `autoscaling.maxReplicas`) | The defaults serve up to about 33,000 subscriptions; see the 10-replica rule below |
| CPU | request `1`, limit `2` (`resources`) | Replicas run at the CPU limit during join bursts at `maxReplicas` |
| Memory | request `3584Mi`, limit `5Gi` (`resources`) | See below |
| Subscriptions per replica | 4,096 at the `5Gi` limit | Follows the memory limit; see below |
| Catch-ups at once per replica | 8 (`valkey.subscriberMaxConcurrentCatchups`) | You change the CPU limit: keep about 4 per CPU |

As a guide, a default replica catches up on the order of 30 to 50 joins per second on layers of up to about 7,000 prims.

The memory request defaults to 70% of the limit, and follows it.
The memory limit is required, and it sizes the replica: at startup the service derives every per-replica memory budget from it, so you change these amounts only by changing `resources.limits.memory`:

| `resources.limits.memory` | Subscriptions per replica | Memory for catching up large layers | Upload memory | Structural-edit cache |
|---|---|---|---|---|
| `4Gi` | 3,276 | 433 MiB | 307 MiB | 204 MiB |
| `5Gi` (default) | 4,096 | 744 MiB | 384 MiB | 256 MiB |
| `8Gi` | 6,553 | 1,607 MiB | 614 MiB | 409 MiB |

A larger limit never gives less of any of these amounts, so you can size the limit to any value at or above the minimum below.
The startup line `pod memory budgets derived from the pod memory limit` and the gauges in {ref}`Is it saturated? <live-edit-metrics-reference-capacity>` show the limit read and the amounts in force.

At the chart's other defaults the smallest limit that starts is `2709Mi` on any CPU count without a backup store, and `2809Mi` on any CPU count with one ({ref}`Enable checkpoints <live-edit-durability-enable-checkpoints>`).
With backups on, each replica also sets aside memory, sized from the limit, for large values waiting to be copied to the backup store: 102 MiB at `4Gi`, 128 MiB at `5Gi` and 204 MiB at `8Gi`.
Raising a connection pool (`valkey.poolSize`, `valkey.subscriberPoolSize`, `valkey.maintenancePoolSize` or `valkey.snapshotPoolSize`), `valkey.subscriberMaxConcurrentCatchups`, `valkey.catchupFencedReadConcurrency`, `layer.uploadAdmissionMaxWaiters`, `layer.maxSize` or `valkey.snapshotAtomicReadMaxBytes` raises that minimum, and takes memory from catching up large layers.
A pod below the minimum for its settings exits at startup with `configuration error:` naming the smallest limit it needs, and during an upgrade the old pods keep serving.
A `helm upgrade` that changes the limit rolls the pods; resizing a running pod's memory in place, on Kubernetes 1.27 and later, restarts its container.

(live-edit-scaling-autoscaling)=
## Autoscaling

`autoscaling.mode` chooses what sets the replica count:

- **`scaler`** (default): a small scaler pod adds replicas for CPU at 80% of the CPU request, for joins waiting more than about 10 seconds to start their catch-up, and for subscriptions above 80% of the ceiling.
- **`hpa`**: a HorizontalPodAutoscaler on CPU utilization, optionally memory (`autoscaling.targetMemoryUtilizationPercentage`).
  It cannot see waiting joins or subscriptions, so a join burst can wait at the minimum replica count.

Choose `scaler` unless the identity that installs the chart cannot create Roles and RoleBindings in the namespace.
In that case, set `autoscaling.scaler.rbac.create: false` and grant the scaler's ServiceAccount the same rules yourself, or use `hpa`.
In `hpa` mode every container in the server pod needs a CPU request (and a memory request if you enable the memory target), injected sidecars included; the chart refuses to render if the token sidecar's requests are removed or zero.

What to expect in both modes:

- Scale-up is immediate.
- Scale-down waits for 10 minutes of lower demand, then removes one pod per minute; each removal reconnects that pod's clients.
- In `scaler` mode, scale-down also waits while a rollout runs or any pod is unready.
- A fresh install starts at one replica until the autoscaler raises it to `minReplicas`.
- If the scaler stops, the replica count stays where it is; alert on it ({ref}`Monitoring <live-edit-monitoring>`).

The scaler pod uses the server's `nodeSelector` and `tolerations` but not its `affinity`; set `autoscaling.scaler.affinity` if it needs one.
If you write your own network policies, let the scaler pod reach port 9090 on the server pods.
In `hpa` mode, set `autoscaling.behavior: null` to restore the Kubernetes scale-down defaults; `{}` keeps the chart's damping.

For a fixed replica count, set `autoscaling.enabled: false` and `replicaCount`, and keep `podDisruptionBudget.minAvailable` below that count.

(live-edit-scaling-read-concurrency)=
## Keep fleet-wide layer reads at 10 or fewer

Keep `autoscaling.maxReplicas` × `valkey.catchupSnapshotReadConcurrency` at or below 10, and the same for `valkey.catchupFencedReadConcurrency`.
Both settings default to `1`, so the defaults support up to 10 replicas.
Above 10, a join wave across the whole fleet can slow edits for every user on the store.
So one deployment serves up to about 33,000 subscriptions; for more, run another deployment with its own backing store.
The scaler logs a warning when `autoscaling.maxReplicas` × `valkey.catchupSnapshotReadConcurrency` exceeds 10.

(live-edit-scaling-subscriber-capacity)=
## Subscriber capacity

Each replica has two limits clients can reach:

- **Subscription ceiling**, which follows `resources.limits.memory` (4,096 at the default `5Gi`; {ref}`Starting size <live-edit-scaling-starting-size>`).
  A subscribe past it waits up to `valkey.subscriberAdmissionWaitMs` (default 20 seconds), then fails with a retryable `UNAVAILABLE`.
- **Catch-ups at once** (`valkey.subscriberMaxConcurrentCatchups`, default 8).
  Later joins wait their turn with the stream open and nothing delivered yet.

On a clustered backing store, `valkey.subscriberPoolSize` (default 256) also caps how many distinct layers one replica serves at once ({ref}`Backing store connections <live-edit-backing-store-connections>`).

Watch `live_edit_subscriber_admitted` per replica against `live_edit_subscriber_max_subscriptions`, sustained `live_edit_subscribe_catchups_waiting`, and the refusal counters in {ref}`Capacity and saturation <live-edit-monitoring-capacity>`.
When replicas stay near either limit, raise `autoscaling.minReplicas`, or `autoscaling.maxReplicas` within the rule above.
New replicas relieve only later joins, so set `autoscaling.minReplicas` for the join bursts you expect at once.

(live-edit-scaling-rollout-headroom)=
## Rollout headroom

A rollout runs up to a third more pods ({ref}`What to expect during a rollout <live-edit-operations-rollouts>`); leave room for them in node capacity and ResourceQuota.
Until old pods finish shutting down, up to twice the replica count holds backing-store connections, so the store must accept connections for twice `autoscaling.maxReplicas` replicas ({ref}`Backing store connections <live-edit-backing-store-connections>`).

(live-edit-scaling-limitations)=
## Limitations

- **Structural edits to very large layers.** Creating, deleting, moving or reordering prims in a layer of more than roughly 35,000 placed assets can fail with `DEADLINE_EXCEEDED`, with nothing applied, while others edit the same layer.
  A placed asset here is a prim with a reference and four attributes.
  With no concurrent edits, the failures start at roughly 69,000 placed assets.
  Stores with higher network latency, such as managed services, and prims with more attributes reach it at smaller sizes.
  Split very large scenes across several layers.
- **Deleting and replacing many layers at once.** Each replica cleans up deleted and replaced layers a few at a time, within memory that grows with `resources.limits.memory`.
  Clean-ups of all but very small layers (a handful of prims and large values) run at most four at a time per replica at `5Gi`, and two at the smallest limit.
  Large values lower that: layers of about 1,000 large values each are cleaned up two at a time at `5Gi`, and one at a time at the smallest limit.
  Once its layer holds more than about 5,100 large values at `5Gi`, or any large value at the smallest limit, such a clean-up runs beside very small layers' clean-ups only.
  Above about 7,600 large values at `5Gi`, or 2,500 at the smallest limit, it runs alone.
  Clean-ups start in arrival order, so a delete of any layer can wait behind earlier deletes, replacements and background clean-ups, for seconds.
  A `DeleteLayer` still waiting at its deadline fails `DEADLINE_EXCEEDED` with nothing changed, and a `DeleteSession` can stop partway the same way; clients retry them as {ref}`Status codes <live-edit-api-errors-status-codes>` describes.
  A replacement's clean-up waits after `CreateLayer` has answered, so the replaced layer's storage is freed later.
  Watch `live_edit_retirement_wait_seconds`; a larger memory limit gives clean-ups more room.
- **Replicas per deployment.** See {ref}`the 10-replica rule <live-edit-scaling-read-concurrency>`.

All scaling and capacity settings: {ref}`Autoscaling settings <live-edit-configuration-autoscaling>` and {ref}`Capacity settings <live-edit-configuration-capacity>`.

(live-edit-scaling-see-next)=
## See next

{ref}`Monitoring <live-edit-monitoring>`: health, metrics and alerts.
