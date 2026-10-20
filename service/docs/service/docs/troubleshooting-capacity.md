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

(live-edit-troubleshooting-capacity)=
# Troubleshooting the backing store, capacity and rollouts

This page lists symptoms of backing-store trouble, capacity limits and rollouts, what to check and how to fix them.
It assumes a deployment installed as in {ref}`Deployment <live-edit-deployment>`.
Startup, readiness and client-access symptoms are on {ref}`Troubleshooting <live-edit-troubleshooting>`; alerts and their runbooks are on {ref}`Monitoring <live-edit-monitoring>`.

(live-edit-troubleshooting-capacity-backing-store)=
## Backing store

| Symptom | Check and fix |
|---|---|
| Backing-store authentication fails with `WRONGPASS` on Azure Managed Redis | Follow {ref}`Troubleshooting WRONGPASS <live-edit-backing-store-azure-wrongpass>`. |
| Writes fail `UNAVAILABLE` with `the backing store refused a write`; `live_edit_store_write_refused_total{reason="out_of_memory"}` rises | The store is full. The service frees released large values by itself, but a store full of layer data does not recover until you add memory; follow {ref}`Backing store full <live-edit-monitoring-store-full>`. |
| Writes to some layers or sessions fail `UNAVAILABLE` for minutes while others succeed | The store is moving their data between shards; the refusals stop when the move ends ({ref}`When the store moves data between shards <live-edit-backing-store-moving-data>`). On a self-hosted cluster, `valkey-cli --cluster check` lists any slot still moving. |
| Uploads of large values wait or fail naming `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` | Released large values await clean-up faster than it runs. On Azure Managed Redis, set `valkey.memoryBytes` to the cache's total memory (its SKU size, {ref}`Choose a size <live-edit-backing-store-azure-size>`); other stores report their own size when their `maxmemory` is set. |
| ERROR `GC loop ended before shutdown`, or WARN `the GC loop has made no progress for too long` with no later INFO `the GC loop is making progress again` | That replica has stopped cleaning up released large values; another replica takes the work over, but this one does none until it restarts. Restart the pod. |
| Backups fail or stay pending | Check token renewal, the identity's permissions on the prefix and {ref}`network access <live-edit-durability-network>`; then follow {ref}`Backups behind or stopped <live-edit-monitoring-backups>`. |

(live-edit-troubleshooting-capacity-rollouts)=
## Capacity and rollouts

| Symptom | Check and fix |
|---|---|
| New pods stay `Pending` during a rollout | The namespace lacks room for the extra pods. Add capacity, or set `strategy.rollingUpdate.maxSurge: 1`. |
| A rollout stalls with pods unable to connect to the backing store | The store's connection limit is reached. Raise it to fit twice `autoscaling.maxReplicas` replicas ({ref}`Backing store connections <live-edit-backing-store-connections>`). |
| Subscribes wait or fail `UNAVAILABLE` | Replicas are at their subscriber limits ({ref}`Subscriber capacity <live-edit-scaling-subscriber-capacity>`). |
| Replica count does not change under load | Check the scaler pod's logs, its Role and RoleBinding, and that network policies let it reach port 9090 on the server pods ({ref}`Scaler not deciding <live-edit-monitoring-scaler>`). |
| Uploads fail `UNAVAILABLE` under load, or large uploads time out | The replica's upload memory is busy. Add replicas, or raise `resources.limits.memory`, which the upload memory follows. |
| Structural edits to a very large layer fail `DEADLINE_EXCEEDED` | The layer is past the {ref}`large-layer limit <live-edit-scaling-limitations>`. Split the scene across several layers. |
| Pod memory stays high after load ends | Compare `process_resident_memory_bytes` with `live_edit_jemalloc_allocated_bytes`. If allocated memory stays high, a replica that served structural edits keeps each edited layer cached for 10 minutes after its last structural edit ({ref}`Capacity and saturation <live-edit-monitoring-capacity>`). If allocated memory is low but resident memory stays high, read the startup `global allocator options` warnings: remove any `allocator.mallocConf` override, and if the transparent huge page state could not be read, make `/sys/kernel/mm/transparent_hugepage/enabled` visible to the container or set the node's setting to `madvise` or `never`. |

(live-edit-troubleshooting-capacity-see-next)=
## See next

{ref}`Configuration <live-edit-configuration>`: every Helm value, its default and when it takes effect.
