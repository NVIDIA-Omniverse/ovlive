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

This page covers how Live Edit protects session and layer state, how to turn on backups, and what to do when the backing store is lost or when the service holds itself for disaster recovery.
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
With `StorageApi`, also set up {ref}`Access to the backup store <live-edit-durability-backup-store-access>`.

To rely on store persistence alone, set `checkpoint.enabled: false` instead.
Pacing and retention settings: {ref}`Checkpoint settings <live-edit-configuration-checkpoints>`.

(live-edit-durability-incremental)=
### Incremental backups and retention

Between full backups the service can write smaller incremental backups, which recovery uses automatically.
They are on by default; `checkpoint.tail.enabled: false` keeps full backups only.

`checkpoint.retainGenerations` (default `2`) sets how many recovery points each layer keeps.
A full backup and the incremental backups on top of it count as one.
A restored layer takes a fresh full backup that uses one slot, so to keep N recovery points through a restore, set N + 1.

The service deletes its own old backups and cleans up partial ones.
The service deletes large values from the backup store only while it holds at least one layer backup, so once every layer has been deleted, the large values their backups held stay there until a layer is backed up again.
Layers created and deleted while the backup store holds no layer backups wait the same way: `live_edit_checkpoint_dead_object_backlog` can keep growing, and the backlog drains after the next layer backup.
Do not delete backup objects by hand or apply bucket lifecycle rules to the prefix: either can remove a backup that a recovery point still needs.

(live-edit-durability-large-values)=
### Backups of large values

Each replica copies every uploaded large value to the backup store separately, shortly after the upload, holding the values waiting to be copied in memory sized from `resources.limits.memory` ({ref}`Starting size <live-edit-scaling-starting-size>`).
A copy is skipped when large uploads arrive faster than the backup store accepts them, when the backup store refuses it, or when a replica stops or crashes before sending it.
Editing is unaffected, and the value stays in the backing store.
The service repairs a skipped copy from the backing store: with `StorageApi` at its default timeouts normally within about 3 to 8 minutes, and with `LocalDisk` within about 1 to 6 minutes.
If the backing store's data is lost before then, requests that open a restored layer using that value fail with `INTERNAL` until a user uploads the value again or replaces the attribute value that uses it.
After you lower `layer.maxSize`, a large value uploaded under the higher limit may never be copied; restore `layer.maxSize`, or raise `resources.limits.memory` until `live_edit_checkpoint_ext_mirror_budget_bytes` exceeds the value's size ({ref}`Are large values backed up? <live-edit-metrics-reference-large-value-backups>`).
With `LocalDisk`, a replica killed or crashing while writing a copy can leave a partial `.tmp` file in `.live-edit/ext/` under `checkpoint.localDiskPath`; the service never reads it, and you can delete it. Deleting one a replica is still writing only fails that copy, which the repair uploads again later.

After the service restores from backups, or after you change `checkpoint.storageApiEndpoint`, `checkpoint.basePath` or `checkpoint.localDiskPath`, it checks every large value once and uploads the ones the backup store lacks, which after a change of location is usually all of them.
Watch `live_edit_checkpoint_ext_backup_owed` and add the `LiveEditLargeValueBackupsMissing` rule ({ref}`Backup alert rules <live-edit-monitoring-backup-alerts>`); `live_edit_checkpoint_ext_mirror_dropped_total` by `reason` shows why copies are skipped ({ref}`Are large values backed up? <live-edit-metrics-reference-large-value-backups>`).

(live-edit-durability-backup-store-access)=
## Access to the backup store

`StorageApi` writes with the service's own identity, which needs the token provider from {ref}`Configure the token sidecar <live-edit-authentication-configure-the-token-sidecar>`; the chart refuses to render `StorageApi` without it.
Grant that identity read, list, write and delete under the backup prefix.
Those rights must include listing `{BACKUP_PREFIX}/.live-edit/sessions/` and checking whether a stored file exists, which the disaster-recovery Jobs rely on.
The Jobs also write, list and delete a small test file under `{BACKUP_PREFIX}/.live-edit/dr-canary/`; one left by a stopped Job is harmless, and you can delete it.
If your Storage service stores its files in an S3-compatible service other than Amazon S3, that service must list objects with strong consistency: a listing started after an upload succeeded includes that upload.

(live-edit-durability-network)=
### Network access to the backup store

Restores and small backups use `checkpoint.storageApiEndpoint` alone.
Backups above the Storage service's upload size limit (2 MiB by default) go to a short-lived blob-storage URL it returns, usually on another host.
Before enabling checkpoints, confirm that the Live Edit pods can resolve that host, trust its TLS certificate, and reach it through your network policies and egress rules; allowing the Storage service does not allow its blob storage.

A missing route shows as small backups succeeding while large ones never land, with `live_edit_checkpoint_adapter_redirect_total{verb="put",outcome="target_failed"}` rising steadily.

(live-edit-durability-validate)=
## Validate backups

Before relying on recovery:

1. Edit a small disposable layer and a large one (several MiB), and wait a minute.
   Confirm that `live_edit_checkpoint_writes_total{result="success"}` increases, `result="error"` does not, and objects appear under the backup prefix.
   One exception is expected on a new backup prefix: `kind="counters_sidecar"` errors can rise after the first layer is created for up to about 15 minutes, or as long as your Storage service caches "not found" answers, then stop by themselves.
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

With checkpoints on, the service restores lost sessions and layers from their backups as clients request them.
When the service holds itself for disaster recovery instead, follow {ref}`When the service is held <live-edit-durability-held>`.
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

To find the IDs, list `{BACKUP_PREFIX}/.live-edit/sessions/` with your Storage tools: each `{SESSION_ID}.json` file there is one backed-up session and holds its name and base path.
Give users the IDs of their sessions, or have them reopen sessions from links or history their client keeps.

```{warning}
After a recovery, a client that reconnects can keep content newer than the backup, including prims created after it.
Reload each recovered layer from empty client state rather than resuming the client's cached state.
Rehearse this reload with your clients during recovery testing.
```

With checkpoints off, the sessions and layers are gone and clients must create them again.

(live-edit-durability-held)=
## When the service is held

When the backing store loses data, the service can hold itself for disaster recovery: every replica refuses calls until the data is restored or its loss accepted ({ref}`Held for disaster recovery <live-edit-monitoring-held>`).
Only a disaster-recovery Job ends the hold, and you start one with a change of `dr` values.

1. Find the incident: each replica logs an `ERROR`, `the fleet is held for disaster recovery`, whose `incident` field holds the incident ID, such as `20261013t1708-a3f9`.
2. Choose the action:
   - With a backup store, restore first ({ref}`Run the restore Job <live-edit-durability-run-restore>`); a restore discards nothing.
   - Accept the loss when a restore stays held on something no backup can supply, and always when no backup store is configured ({ref}`Accept a loss <live-edit-durability-accept-loss>`).
3. Check what a Job needs:
   - Kubernetes 1.29 or later: with an action set, the chart refuses to render for an older cluster.
   - Room in the namespace for one more pod of the Job's resources, `dr.resources`, which default to the service's `resources` ({ref}`Resources for the restore Job <live-edit-durability-restore-job>`).
   - The service's network access to the backing store and the backup store.
     A Job's pod carries `app.kubernetes.io/component: dr-job`, not `server`, so a network policy that selects server pods does not apply to it.
     If your namespace denies egress by default, give pods with that label and `app.kubernetes.io/instance: {RELEASE}` the service's egress ({ref}`Configure ingress <live-edit-deployment-configure-ingress>`).
   - If a service mesh in the namespace injects its proxy as a regular container, exclude pods labelled `app.kubernetes.io/component: dr-job` from injection, or use the mesh's native-sidecar mode.
     Otherwise the proxy keeps the Job's pod running after the Job's work ends, so the Job neither completes nor fails.

For the values change that starts or stops a Job:

- Change only `dr` values in it.
  Setting or clearing them changes neither the Deployment nor its settings, so nothing restarts.
  Never combine an action with a chart, image or settings upgrade, and never apply one during a rollout.
- Set one action at a time: setting both fails the render.
- The chart's smoke tests do not run while an action is set.
  With `tests.enabled: true`, pass `--no-hooks` to an upgrade that clears every action while the service is still held, such as one that stops a Job.
  Otherwise the smoke tests run against the held service and fail the upgrade, and with `--atomic` the rollback creates the Job again.
- Never pass `--wait-for-jobs`.
  Helm would wait for the Job to finish, which can take longer than the upgrade's timeout, and would wait out the timeout on a Job that failed at once; with `--atomic`, the rollback then deletes the Job and its log.
- With `--atomic`, first delete any server pod that is not ready: if the upgrade's wait times out on it, the rollback deletes the Job.

(live-edit-durability-run-restore)=
## Run the restore Job

1. Merge into your values file, with the incident from the `ERROR` line:

   ```yaml
   dr:
     restore:
       incident: "{INCIDENT}"
   ```

2. Apply it:

   ```bash
   helm upgrade live-edit-service ./live-edit-service \
     --namespace storage-apis -f live-edit-values.yaml
   ```

3. Find the Job by its labels, then follow its log:

   ```bash
   kubectl get jobs -n storage-apis \
     -l app.kubernetes.io/component=dr-job,app.kubernetes.io/instance=live-edit-service
   kubectl logs -f job/{JOB_NAME} -n storage-apis -c live-edit-service
   ```

   For a release named `live-edit-service` the Job is `live-edit-service-dr-restore-{INCIDENT}`, with `-{ATTEMPT}` appended when `dr.restore.attempt` is set.

4. Verify:
   - the Job shows `1/1` completions;
   - its log's last line starts `the restore Job cleared the hold for incident`;
   - `live_edit_dr_held` reads `0` on every replica, and each server pod has restarted once.

   Then follow {ref}`After the hold clears <live-edit-durability-after-hold>`.
   If the Job fails, see {ref}`Read a Job's log <live-edit-durability-job-log>`.

What a restore costs:

- The Job writes nothing until a quiet period after the hold began has passed: about 3.5 minutes with `StorageApi` at the default timeouts, and about 2 minutes otherwise.
- With `StorageApi`, before it trusts that a backup or large value is missing, it waits until `checkpoint.notFoundCacheSecs` (default `900` seconds) after the quiet period.
  While it waits it logs `INFO` lines starting `waiting` that give the seconds left and what it waits for; a wait to plan again follows an `INFO` starting `the restore plan saw`.
- The restore takes longer the more layers and data the backups hold.
  With 4 CPUs and the backing store on the same network, one layer of about 194,000 specs, about 50 MB of backups, takes on the order of 45 seconds to restore.
- When the hold clears, every replica exits within seconds and restarts once, and clients reconnect.

(live-edit-durability-restore-job)=
### Resources for the restore Job

The restore and accept-loss Jobs need a memory limit of at least 3,282 MiB, or 2,687 MiB with incremental backups off or no backup store.
`dr.resources` sets them: each key it sets replaces the service's `resources` value, so `limits: {memory: 8Gi}` keeps the service's CPU, and an unset memory request is 70% of the limit.
Below the minimum, a Job exits with code 2 before connecting to anything, logging a `configuration error:` line that names the smallest limit and `dr.resources.limits.memory`.
Any other setting refused at startup also exits 2; Kubernetes fails the Job at once rather than retrying it.
Each Job restores up to `checkpoint.restoreParallelism` layers at once (default `8`).
If it cannot read its memory limit, it restores one layer at a time and logs a `WARN` starting `the {MODE} Job could not read a memory limit` (`{MODE}`: `restore` or `accept-loss`).
If a Job runs out of memory, raise `dr.resources.limits.memory` and set a new `attempt`: a Job keeps the resources it was created with.
Lowering `checkpoint.restoreParallelism` instead changes the service's settings, so its replicas restart, held.
More CPU in `dr.resources` shortens the restore of a large store, since a Job writes several layers at once.

(live-edit-durability-job-log)=
## Read a Job's log

A Job's log is the record of its run: its metrics are readable only on its pod while it runs.
When a Job fails at once, `kubectl describe job {JOB_NAME} -n storage-apis` shows the exit code, such as `failed with exit code 3`.

| Exit code | Meaning | Kubernetes | Do |
|---|---|---|---|
| `0` | The hold cleared, was already cleared, or the service was not held | Completes the Job | {ref}`After the hold clears <live-edit-durability-after-hold>` |
| `3` | The service stays held; the last `ERROR` line says why | Fails the Job at once | See below |
| `4` | The hold names a different incident; the Job cleared nothing | Fails the Job at once | Set the action's `incident` to the one the replicas' `ERROR` line names |
| `2` | A refused configuration, before the Job connected: a memory limit below the minimum, or a setting the service refuses at startup | Fails the Job at once | Fix the setting the `configuration error:` line names, then set a new `attempt` |
| `1` | A failure a retry may fix: another restore or accept-loss Job is running, a backing-store or backup-store fault, the backing store failed over, restarted or moved data between shards while the Job read it, or the Job was stopped | Retries it, up to `dr.backoffLimit` (default `6`) times | If the retries run out, fix what the last `ERROR` line names, then set a new `attempt`. A line containing `changed while the Job read`, `slot map change`, `migrating or importing`, `out of date` or `is no longer a primary` means the store was changing: once the store change has finished, set a new `attempt` |

Any other code, such as `137` for a pod killed for running out of memory, is retried like `1`.
A pod lost to an eviction or a node drain is retried without counting against `dr.backoffLimit`.
If the store moves data between shards, scales, fails over or restarts while a restore or accept-loss Job reads it, the Job usually detects it, stops, and Kubernetes re-runs it, up to `dr.backoffLimit` times.
It cannot detect every such change, so let store changes finish before starting a restore or accept-loss Job, and make no store change while it runs.

A restore that stays held ends with an `ERROR` containing `exits 3`:

- `the restore Job restored everything it could and recounted the references of the large values, but` counts what remains, and the `WARN` lines before it name each item.
  For items the backup store `could not confirm`, it answered with an error rather than "not found": fix the service identity's access to it ({ref}`Access to the backup store <live-edit-durability-backup-store-access>`), then run the restore again with a new `attempt`.
  For layers, sessions or large values that no backup can supply, {ref}`accept the loss <live-edit-durability-accept-loss>`.
- A line containing `cannot be shown to` means the backing store's `CLUSTER NODES` answer did not show which node serves the data the Job reads, so the Job trusts none of its reads; open a support case with the line.
- Any other exit-3 line names its cause; grant a backup-store permission it names, otherwise open a support case with the line.

The restore's summary, logged whether or not the service stays held:

- per restored layer, an `INFO` starting `layer {LAYER} of session {SESSION_ID} (object {OBJECT_ID}): restored from backup generation`, with where the layer resumes and how many of its large values are in the backing store or fetched from the backup store on their first read;
- per layer that was due for deletion but still had a copy, an `INFO` containing `kept its surviving copy and cleared its retirement marks`, with where it resumes; the layer and its large values are kept;
- a `WARN` containing `is restored from an older backup` for a layer whose newest backup could not be used, so it lost more edits;
- a `WARN` containing `no session backup; it is not restored` for a session nothing can restore;
- a totals line starting `the restore Job's summary:`, which counts the large values only the backup store holds: each is fetched the first time a layer that uses it is opened after the restart, which makes that open slower ({ref}`what to expect <live-edit-monitoring-held>`).

(live-edit-durability-accept-loss)=
## Accept a loss

Accept-loss restores everything a backup can supply, as a restore does, then permanently discards what none can, and ends the hold.
It discards:

- layers with no usable backup, removed from their sessions;
- sessions with no record in the backing store and no usable session backup, with every layer they name, complete ones included, and their session backups;
- the attribute values and time samples whose large value is in no store.

It keeps backups no layer references.
A layer whose only backups are full backups over the per-delta entry limit, written by an earlier release, is removed from its session, but its backups are kept; recovering it needs a support case.

```{warning}
Accept-loss cannot be undone, and its log is the only record of what it discarded.
Save the log before the Job is deleted ({ref}`After the hold clears <live-edit-durability-after-hold>`).
```

1. Save the restore Job's log, then clear `dr.restore` and merge into your values file:

   ```yaml
   dr:
     acceptLoss:
       incident: "{INCIDENT}"
       confirm: "discard-lost-data"
   ```

   The render fails without `confirm`, with any other value, and with a confirmation left set while `incident` is not.
2. Apply it, find the Job and follow its log as in {ref}`Run the restore Job <live-edit-durability-run-restore>`; it is named `live-edit-service-dr-accept-loss-{INCIDENT}`.
   With `StorageApi`, it reads nothing until `checkpoint.notFoundCacheSecs` (default `900` seconds) after the quiet period.
   It can therefore take up to 15 minutes longer than a restore; while it waits it logs `INFO` lines starting `waiting` that give the seconds left.
   Removing the values that are in no store takes longer on a large layer, since the Job reads all of that layer's values.
3. Verify as for a restore; the log's last lines start `the accept-loss Job's summary: discarded` and `the accept-loss Job cleared the hold for incident`.
4. Have users reload each affected layer from empty client state; discarded layers and sessions no longer exist.

**Items it cannot confirm lost.**
When the backup store answers a read with an error rather than "not found", the Job cannot confirm that item lost, so it discards nothing at all.
It still restores and recounts, then logs an `ERROR` starting `the accept-loss Job restored everything it could and recounted the references of the large values, but discarded nothing` and exits 3.
A layer or value of a session it discards goes with that session whatever the backup store answered.
Then, in order:

1. Fix the backup store's access (the service identity, its permissions, the address) and run the Job again with a new `dr.acceptLoss.attempt`.
2. As a last resort, also set `dr.acceptLoss.confirmUnconfirmed: "discard-unconfirmed-data"`.
   The Job then discards those items too, although a backup may still hold them, and logs a `WARN` per item saying what it did.
   It keeps a layer's surviving copy, and leaves the backups of an object it cannot trace to a layer.
   For such an object, grant the identity read on its backups, or remove them by hand once the loss is accepted.

(live-edit-durability-accept-loss-log)=
## Read an accept-loss log

An accept-loss log holds the restore's lines, each naming `the accept-loss Job`, and these:

| Level | Line starts | Meaning | Do |
|---|---|---|---|
| `INFO` | `the accept-loss Job restored everything it could;` | The restore is done; the discards start | Nothing |
| `INFO` | `the accept-loss Job discards` | What it is about to discard | Nothing |
| `WARN` | `the accept-loss Job discarded layer` | A layer with no usable backup is removed | Nothing |
| `WARN` | `the accept-loss Job discarded session` | A session with no record and no usable backup is removed, with its layers | Nothing |
| `WARN` | `the accept-loss Job removed {N} large values that are in no store` | Values removed from one layer, with up to 100 prims and attributes that used them | Have users upload again what they still have |
| `WARN` | `the allocator purge did not run` | Logged once; the Job may run slower | Nothing: the Job continues |
| `WARN` | `the accept-loss Job left layer` | A layer of that name was created or replaced after the Job planned, so the Job left it | Nothing |
| `WARN` | `the accept-loss Job removed the mapping of layer` | A layer whose backups are over the per-delta entry limit is removed from its session; its backups are kept | Open a support case with the line to recover it |
| `WARN` | `the accept-loss Job cannot read backup generation` | It skipped a full backup over the per-delta entry limit for an older one | Nothing |
| `WARN` | `the accept-loss Job could not confirm` | Only with `confirmUnconfirmed`: an item the backup store answered with an error; the line says what the Job did with it | Nothing |
| `INFO` | `the accept-loss Job's summary: discarded` | What it discarded, before it ends the hold | Save the log |
| `ERROR` | `the accept-loss Job restored everything it could and recounted` | Exit 3: items it could not confirm lost; nothing discarded | See {ref}`Accept a loss <live-edit-durability-accept-loss>` |
| `ERROR` | `the accept-loss Job found object` | Exit 1: a layer it was about to discard is complete, so another writer is active | Find and stop the other writer, let the Job run again, and open a support case |
| `WARN` | `the accept-loss Job removed objects`, `the accept-loss Job removed nothing from the dead-object set` | After that `ERROR`: what the Job undid so its retry keeps a complete layer | Nothing |
| `ERROR` | `the accept-loss Job's discards are inconsistent with its plan` | Exit 1: a defect in the Job; it discarded nothing | Open a support case with the line |
| `ERROR` | `the accept-loss Job could not remove`, `the accept-loss Job could not rewrite`, `the accept-loss Job could not read which` | Exit 1: a backing-store or backup-store step failed | If the retries run out, fix that store, then set a new `attempt` |

(live-edit-durability-after-hold)=
## After the hold clears

1. Save each Job's log before anything deletes it, from every pod it ran, so a retried Job keeps its earlier pods' lines:

   ```bash
   kubectl logs -n storage-apis -l batch.kubernetes.io/job-name={JOB_NAME} \
     -c live-edit-service --prefix --tail=-1 > {JOB_NAME}.log
   ```

2. Clear every `dr` value you set, and apply the change with `helm upgrade`.
   Helm deletes the Job and its pod, and their log with them.
   A Job left in place makes a later upgrade to a new chart version or image, or one that changes a value the Job's pod uses, such as `resources` or `podLabels`, fail with `field is immutable`, because a Job's pod cannot change.
3. Verify: the `kubectl get jobs` command in {ref}`Run the restore Job <live-edit-durability-run-restore>` lists no Job.

To run a Job again, after it failed or with different `dr.resources`, set a new `dr.restore.attempt` or `dr.acceptLoss.attempt`, such as `2`.
The Job gets a new name, and Helm replaces the old Job with it.
To stop a running Job, clear its values, with `--no-hooks` if the smoke tests are on ({ref}`When the service is held <live-edit-durability-held>`): Helm deletes it, the service stays held, and a later run continues from what it wrote.

Kubernetes deletes a finished Job, its pod and its log `dr.ttlSecondsAfterFinished` seconds after it finishes (default `604800`, 7 days); `0` or null keeps it until you clear its values.
If the values are still set at a later `helm upgrade`, Helm creates the Job again; it acts only on its own incident's hold, so once that hold has cleared it exits 0 and changes nothing.

(live-edit-durability-see-next)=
## See next

{ref}`Scaling <live-edit-scaling>`: replica sizing, autoscaling and limits.
