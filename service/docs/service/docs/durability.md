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

(live-edit-durability)=
# Durability, backups and recovery

This page covers how Live Edit protects session and layer state, how to turn on backups, and what to do when the backing store is lost.
It assumes a backing store set up as in {ref}`Backing store <live-edit-backing-store>` and the service identity from {ref}`Service identity <live-edit-authentication-service-identity>`.

(live-edit-durability-protection-layers)=
## Two protection layers

- **Store persistence** (Valkey AOF; the transaction log on Amazon MemoryDB) protects live state across store restarts and failovers.
- **Checkpoints** back up each edited layer and session to a separate backup store, so the service can rebuild them after the store's whole dataset is lost.

Run both in production. Together they give four modes:

| Mode | Loss on a store restart | Loss when the store's data is gone | At startup |
|---|---|---|---|
| Persistence on, checkpoints on (recommended) | About 1 s with AOF; none on MemoryDB | Edits since the newest backup | Allowed |
| Persistence on, checkpoints off | About 1 s with AOF; none on MemoryDB | All sessions and layers | Allowed |
| Persistence off, checkpoints on | Edits since the newest backup | Edits since the newest backup | Allowed |
| Persistence off, checkpoints off | All sessions and layers | All sessions and layers | Refused (warning only on managed stores) |

Each edited layer is backed up within `checkpoint.interval` seconds (default `60`) of its previous backup, or sooner once idle for `checkpoint.quiesce` seconds (default `30`).
At the defaults, expect to lose up to about 70 seconds of edits; failed or delayed backups lose more, so monitor them ({ref}`Monitoring <live-edit-monitoring>`).

Choose checkpoints without persistence only when your store cannot persist, such as Azure Managed Redis with active geo-replication.

(live-edit-durability-startup-check)=
## The startup durability check

A replica refuses to start, and its pod restarts in `CrashLoopBackOff`, when checkpoints are enabled with the `Disabled` adapter or are off while the store reports persistence off.
The chart defaults to the first case (`checkpoint.enabled: true`, `checkpoint.adapter: Disabled`), so a fresh install starts only once you choose a mode.
The log shows:

```text
live-edit refuses to start with no durability configured: enable Valkey AOF, enable LIVE_EDIT_CHECKPOINT_ENABLED with a configured adapter, or both
```

```{warning}
Managed stores (Azure Managed Redis, Amazon MemoryDB, Amazon ElastiCache) do not let the service read their persistence setting.
With checkpoints off, the service there logs a warning containing `could not verify Valkey AOF state` and starts, even with no protection.
Confirm persistence in the provider's console, or run checkpoints.
```

(live-edit-durability-enable-checkpoints)=
## Enable checkpoints

Choose the adapter:

- **`StorageApi`**, for production: backups go to a Storage service prefix that every replica shares.
- **`LocalDisk`**, only for a single replica with a persistent volume at `checkpoint.localDiskPath`.
  Per-pod or container disks are not a shared backup store.
- **`InMemory`** is lost on restart; never use it for recovery.

The `checkpoint` values for each adapter:

```yaml
# StorageApi
checkpoint:
  enabled: true
  adapter: StorageApi
  storageApiEndpoint: "{STORAGE_ENDPOINT}"   # https://storage.example.com
  basePath: "{BACKUP_PREFIX}"                # https://assets.example.com/live-edit-backups/
---
# LocalDisk
checkpoint:
  enabled: true
  adapter: LocalDisk
  localDiskPath: "{VOLUME_MOUNT_PATH}"
```

{ref}`Create the values file <live-edit-deployment-create-the-values-file>` shows them in a complete values file.
Give each deployment its own backup prefix.
`StorageApi` writes with the service's own identity, which needs the token provider from {ref}`Configure the token sidecar <live-edit-authentication-configure-the-token-sidecar>`; the chart refuses to render `StorageApi` without it.
Grant that identity read, list, write and delete under the prefix.

To rely on store persistence alone, set `checkpoint.enabled: false` instead.
Pacing and retention settings: {ref}`Checkpoint settings <live-edit-configuration-checkpoints>`.

(live-edit-durability-network)=
### Network access to the backup store

Restores and small backups use `checkpoint.storageApiEndpoint` alone.
Backups above the Storage service's upload size limit (2 MiB by default) go to a short-lived blob-storage URL it returns, usually on another host.
Before enabling checkpoints, confirm that the Live Edit pods can resolve that host, trust its TLS certificate, and reach it through your network policies and egress rules; allowing the Storage service does not allow its blob storage.

A missing route shows as small backups succeeding while large ones never land, with `live_edit_checkpoint_adapter_redirect_total{verb="put",outcome="target_failed"}` rising steadily.

(live-edit-durability-incremental)=
### Incremental backups and retention

Between full backups the service can write smaller incremental backups, which recovery uses automatically.
They are on by default; `checkpoint.tail.enabled: false` keeps full backups only.

`checkpoint.retainGenerations` (default `2`) sets how many recovery points each layer keeps.
A full backup and the incremental backups on top of it count as one.
A restored layer takes a fresh full backup that uses one slot, so to keep N recovery points through a restore, set N + 1.

The service deletes its own old backups and cleans up partial ones.
An empty listing looks the same as a backup store the service cannot list or one at the wrong path, so the service deletes large values only after a listing names at least one backup.
As a result, once every layer has been deleted, the large values their backups held stay in the backup store until a layer is backed up again.
Layers created and deleted while the backup store holds no layer backups wait the same way: `live_edit_checkpoint_dead_object_backlog` can keep growing, and the backlog drains after the next layer backup.
Do not delete backup objects by hand or apply bucket lifecycle rules to the prefix: either can remove a backup that a recovery point still needs.

(live-edit-durability-large-values)=
### Backups of large values

Each replica copies every uploaded large value to the backup store separately, shortly after the upload, holding the values waiting to be copied in memory sized from `resources.limits.memory` ({ref}`Starting size <live-edit-scaling-starting-size>`).
A copy is skipped when large uploads arrive faster than the backup store accepts them, when the backup store refuses it, or when a replica stops or crashes before sending it.
Editing is unaffected, and the value stays in the backing store.
Every 5 minutes, one replica checks the backup store for each large value with no confirmed copy, and uploads any the backup store lacks from the backing store.
With `StorageApi` at its default timeouts a skipped copy is normally repaired within about 3 to 8 minutes, and with `LocalDisk` within about 1 to 6 minutes.
If the backing store's data is lost before then, requests that open a restored layer using that value fail with `INTERNAL` until a user uploads the value again or replaces the attribute value that uses it.
Repairs upload one value at a time within the copy memory, so pod memory and its minimum are unchanged.
A value too large for that memory, possible only for one uploaded under a higher `layer.maxSize`, is never copied until you restore that `layer.maxSize` or raise `resources.limits.memory` so that the copy memory, `live_edit_checkpoint_ext_mirror_budget_bytes`, exceeds its size ({ref}`Are large values backed up? <live-edit-metrics-reference-large-value-backups>`).
With `LocalDisk`, a replica killed or crashing while writing a copy can leave a partial `.tmp` file in `.live-edit/ext/` under `checkpoint.localDiskPath`; the service never reads it, and you can delete it. Deleting one a replica is still writing only fails that copy, which the repair uploads again later.

The service also re-checks each confirmed copy about once a day.
After the service restores from backups, or after you change `checkpoint.storageApiEndpoint`, `checkpoint.basePath` or `checkpoint.localDiskPath`, it checks every large value once and uploads the ones the backup store lacks, which after a change of location is usually all of them.
Watch `live_edit_checkpoint_ext_backup_owed` and add the `LiveEditLargeValueBackupsMissing` rule ({ref}`Backup alert rules <live-edit-monitoring-backup-alerts>`); `live_edit_checkpoint_ext_mirror_dropped_total` by `reason` shows why copies are skipped ({ref}`Are large values backed up? <live-edit-metrics-reference-large-value-backups>`).

(live-edit-durability-validate)=
## Validate backups

Before relying on recovery:

1. Edit a small disposable layer and a large one (several MiB), and wait a minute.
   Confirm that `live_edit_checkpoint_writes_total{result="success"}` increases, `result="error"` does not, and objects appear under the backup prefix.
   One exception is expected on a new backup prefix: if your Storage service caches "not found" answers, `kind="counters_sidecar"` errors can rise for up to that cache time after the first layer is created, then stop by themselves.
   A `WARN` containing `the post-write checkpoint listing is empty` means the service writes backups but cannot list them; see {ref}`Backups behind or stopped <live-edit-monitoring-backups>`.
2. Note the IDs of a few sessions and their layers' content, then copy the backup prefix to a new prefix with your Storage tools.
3. Install a second release in its own namespace, with an empty backing store and `checkpoint.basePath` set to the copy.
   Grant the test release's service identity read, list, write and delete on the copied prefix.
   Never point it at the production prefix: its own backup clean-up would delete production backups.
4. Through the test release, open each noted session by its ID and its layers with a fresh client.
   Recovery works when `live_edit_checkpoint_recovery_attempts_total{result="restored"}` rises on the test release and the content matches what you noted.
5. Compare the recovery time and the recovered content with your requirements, then uninstall the test release.

(live-edit-durability-store-loss)=
## When the backing store is lost

With checkpoints on, recovery is automatic; there is no restore command.
Open streams end with `UNAVAILABLE` and clients reconnect.
The first request for a missing session or layer rebuilds it from its newest backup; the layer keeps its identity, but its content goes back to that backup.

```{warning}
A layer whose content is over the 99,999-entry limit of one upload ({ref}`Deltas that are too large <live-edit-api-validation-too-large>`) is not rebuilt from its backup.
Requests for it fail with `INTERNAL`, and a replica usually logs a `WARN` containing `a backup over the per-delta entry limit cannot be restored on access`.
For an older backup, it may instead log an `ERROR` `internal error` line whose `error` field starts `checkpoint recovery failed for object {OBJECT_ID}:` (`layer-state recovery failed for object {OBJECT_ID}:` for a subscription).
That line ends `one delta may carry at most 99999 entries in total`.
The service does not fall back to an older backup of the layer.
To bring the layer back, a user who still has its content creates the layer again, split across several uploads; otherwise the service cannot recover it.
A layer created again this way is still over the limit, so its backups still cannot rebuild it; to have it protected, split its content into layers of fewer than 99,999 entries each.
Creating it again replaces the layer, and the service then deletes the old layer's backups; copy them out of the backup prefix first if you want to keep them.
The layer's own backups are the files under `{BACKUP_PREFIX}/.live-edit/layers/` whose names start with `{OBJECT_ID}.`, the object ID from either log line.
Its large values are the `{BACKUP_PREFIX}/.live-edit/ext/{HASH}.blob` files named in the `ext_hashes` list of the layer's `.ext-refs.json` files; the service deletes those too once no remaining backup names them.
```

1. Bring the backing store back, empty, with the same requirements and connection settings ({ref}`Backing store <live-edit-backing-store>`).
2. Wait for the pods to report ready: readiness drops while the store is unreachable and returns when it answers.
3. Open a known session by its ID and confirm its layers' content.
   `live_edit_checkpoint_recovery_attempts_total{result="restored"}` counts rebuilt sessions and layers; `result="error"` should stay flat, except for layers over the entry limit (warning above).
   For the next day, watch the signals in {ref}`After a disaster-recovery restore <live-edit-monitoring-after-restore>`.
4. Ask users to reopen their sessions by ID and reload the recovered layers, as below, before they resume editing.

```{important}
Only a request that names a session's ID restores it.
Until each session has been opened by its ID, session lists are empty and the session looks lost.
Do not create a replacement session with the same name: that makes a new, separate session, and the original stays recoverable by its ID.
```

The service has no command that restores a session; a client opening it by ID does.
To find the IDs, list `{BACKUP_PREFIX}/.live-edit/sessions/` with your Storage tools: each `{SESSION_ID}.json` file there is one backed-up session and holds its name and base path.
Give users the IDs of their sessions, or have them reopen sessions from links or history their client keeps.

```{warning}
After a recovery, a client that reconnects can keep content newer than the backup, including prims created after it.
Reload each recovered layer from empty client state rather than resuming the client's cached state.
Keep `layer.catchupSnapshotEnabled: true`; turning it off can also leave reconnecting clients with content from before the recovery, and, after a reorder, with a parent's children in a different order from other users.
Rehearse this reload with your clients during recovery testing.
```

With checkpoints off, the sessions and layers are gone and clients must create them again.

(live-edit-durability-see-next)=
## See next

{ref}`Scaling <live-edit-scaling>`: replica sizing, autoscaling and limits.
