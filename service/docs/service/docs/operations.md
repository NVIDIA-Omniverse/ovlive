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

(live-edit-operations)=
# Upgrades and rollback

This page covers changing a running deployment, upgrading and rolling back, and rotating credentials; diagnosing problems is on {ref}`Troubleshooting <live-edit-troubleshooting>`, and rollout symptoms on {ref}`Troubleshooting the backing store, capacity and rollouts <live-edit-troubleshooting-capacity>`.
It assumes a deployment installed as in {ref}`Deployment <live-edit-deployment>`, with its values file kept outside the extracted chart directory so a new chart does not overwrite it.

(live-edit-operations-configuration-changes)=
## Change configuration

Apply every change with `helm upgrade` and your complete values files, in their original order; do not use `--reuse-values`, which can miss settings a newer chart requires.
A change to a server setting rolls the server pods, and their clients reconnect to other replicas.

```bash
helm lint ./live-edit-service -f live-edit-values.yaml
helm upgrade live-edit-service ./live-edit-service \
  --namespace storage-apis -f live-edit-values.yaml --wait
kubectl rollout status deployment/live-edit-service -n storage-apis
```

The rollout succeeds when every pod reports ready.
Then repeat the {ref}`smoke tests <live-edit-deployment-smoke-tests>`.

(live-edit-operations-upgrade)=
## Upgrade

1. Fetch and extract the new chart into a new directory:

   ```bash
   export LIVE_EDIT_VERSION={NEW_VERSION}
   helm fetch "https://helm.ngc.nvidia.com/nvidia/omniverse/charts/live-edit-service-${LIVE_EDIT_VERSION}.tgz" \
     --username='$oauthtoken' --password="${NGC_API_KEY}"
   mkdir "chart-${LIVE_EDIT_VERSION}" && tar -xf "live-edit-service-${LIVE_EDIT_VERSION}.tgz" -C "chart-${LIVE_EDIT_VERSION}"
   ```

2. If your values file sets `image.tag`, update it to the new version or remove it (the chart defaults to its own version).
   Compare the old and new charts' `values.yaml` for changed defaults.
   Remove any value listed under {ref}`Removed settings <live-edit-operations-removed-settings>`, rename every other chart setting whose name holds an underscore to its camelCase name ({ref}`Renamed settings <live-edit-operations-renamed-settings>`), and check that `resources.limits.memory` is set and at least the minimum for your settings ({ref}`Starting size <live-edit-scaling-starting-size>`).
   If you set `valkey.commandTimeoutMs: 0`, set a timeout in milliseconds instead, such as the default `5000`: the chart refuses to render `0`.
   If `observability.tracing.endpoint.grpc` or `observability.metrics.endpoint.grpc` is set, check that it starts `http://` or `https://`; change `grpc://` to `http://`, or the new pods stop at startup.
   Keep `checkpoint.tick` at 19 or less and below `checkpoint.interval`; at 20 or more, `helm upgrade` fails before any pod changes, naming the setting.
   Write every whole-number value as plain digits ({ref}`Value syntax <live-edit-configuration-value-syntax>`); a negative or fractional number or text, including one carried over by `--reuse-values`, makes `helm upgrade` fail before any pod changes, naming the value.
3. Confirm the namespace has {ref}`rollout headroom <live-edit-scaling-rollout-headroom>`, and that the identity applying the chart can create Roles and RoleBindings there in the default `scaler` autoscaling mode (otherwise see {ref}`Autoscaling <live-edit-scaling-autoscaling>`).
   Check that no `dr` value is set: a disaster-recovery Job left in place fails the upgrade with `field is immutable`, and an action must never be set in the same upgrade ({ref}`After the hold clears <live-edit-durability-after-hold>`).
4. Upgrade and verify:

   ```bash
   helm upgrade live-edit-service "./chart-${LIVE_EDIT_VERSION}/live-edit-service" \
     --namespace storage-apis -f live-edit-values.yaml --wait --atomic
   kubectl rollout status deployment/live-edit-service -n storage-apis
   ```

   The upgrade succeeds when Helm reports the new revision deployed and every pod is ready.
   Then repeat the smoke tests and check that backups still succeed ({ref}`Durability <live-edit-durability>`).

(live-edit-operations-renamed-settings)=
### Renamed settings

Every chart value name is now camelCase, one capital letter per former underscore: `valkey.memory_bytes` is now `valkey.memoryBytes`.
Environment variables did not change; {ref}`Configuration <live-edit-configuration>` lists every new name.
Rename each old name in your values files and `--set` flags, then check that `helm template` with them renders without error.
While your values use an old name, the chart refuses to render, so `helm upgrade` stops before any pod changes; the error lists each old name with its new one.

- **`helm upgrade --reuse-values` fails on this upgrade**: it carries the old names. Upgrade with your renamed values files; `--reset-then-reuse-values` (Helm 3.14 or later) works only if the release's own earlier values used no old name.
- **`helm rollback` works**, because it restores the earlier chart with that revision's own values. Installing an earlier chart with your renamed values does not fail, but that chart ignores the renamed values and uses its own defaults for them; give an earlier chart your values files from before the rename.

(live-edit-operations-removed-settings)=
### Removed settings

These settings no longer exist.
Unless its row says otherwise, a pod whose environment still sets one stops at startup with a `configuration error:` naming it, and the chart refuses to render one set to anything but its old default.

| Environment variable | Former Helm value | What to do |
|---|---|---|
| `LIVE_EDIT_SUBSCRIBER_MAX_SUBSCRIPTIONS` | `valkey.subscriber_max_subscriptions` | Remove it. To change subscriptions per replica, change `resources.limits.memory`. |
| `LIVE_EDIT_CATCHUP_BUILD_MEMORY_BUDGET_MIB` | `valkey.catchup_build_memory_budget_mib` | Remove it. To change the memory for catching up large layers, change `resources.limits.memory`. |
| `LIVE_EDIT_UPLOAD_MEMORY_BUDGET_MIB` | `layer.upload_memory_budget_mib` | Remove it. To change upload memory, change `resources.limits.memory`. |
| `LIVE_EDIT_STRUCTURAL_CACHE_MAX_MIB` | `layer.structural_cache_max_mib` | Remove it. To change the structural-edit cache, change `resources.limits.memory`; to stop using the cache, set `layer.updateLayerBackend: lua-watch-structural`. |
| `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED` | `layer.catchupSnapshotEnabled` (`layer.catchup_snapshot_enabled` in 0.1.x) | Remove it. No setting replaces it; to get the behaviour `false` selected, roll back to an earlier release. |
| `LIVE_EDIT_GC_POLL_INTERVAL_SECS` | `gc.poll_interval_secs` | Remove it. It is ignored rather than refused, and nothing replaces it. |

(live-edit-operations-upgrade-from-0-2-0)=
## Upgrading from 0.2.0 or earlier

0.2.0 and earlier releases handle a spec that is created, renamed or moved onto a name a sibling already uses in different ways from this release.
While pods of both versions run side by side, during an upgrade or a rollback, a rare edit can be lost even though the service accepted it.

To avoid that, upgrade without the two versions overlapping, at the cost of a short outage: clients cannot connect while no pod runs, and reconnect once the new pods are ready.

1. Scale the service to zero with the chart you installed your current release from:

   ```bash
   helm upgrade live-edit-service {CURRENT_CHART_DIR}/live-edit-service \
     --namespace storage-apis -f live-edit-values.yaml \
     --set autoscaling.enabled=false --set replicaCount=0 --wait
   kubectl get pods -n storage-apis -l app.kubernetes.io/instance=live-edit-service,app.kubernetes.io/component=server
   ```

   Repeat the second command until it reports `No resources found`: a stopping pod can take up to `terminationGracePeriodSeconds` to exit.
2. Upgrade to the new chart with your usual values files, as in {ref}`Upgrade <live-edit-operations-upgrade>`.
   Run the upgrade with your values files, not `--reuse-values`, or the service stays at zero pods with autoscaling off.

A rolling rollback to 0.2.0 or earlier carries the same risk.
To roll back without it, first run step 1 with this release's chart, then roll back with `helm rollback` to the revision that ran your earlier release before the upgrade, not one that scaled it to zero ({ref}`Roll back <live-edit-operations-rollback>`).

After a rolling rollback, restart the pods once more when the rollback completes:

```bash
kubectl rollout restart deployment/live-edit-service -n storage-apis
kubectl rollout status deployment/live-edit-service -n storage-apis
```

If the two versions overlap, these self-checks can increase and fire the `LiveEditSelfCheckFired` warning while both versions run, and afterwards at most once more per layer on each pod:

- `live_edit_structural_cache_divergence_total`;
- `live_edit_structural_cache_audit_total{outcome="diverged"}`.

These increases need no support case.

Not expected, so follow {ref}`Self-check fired <live-edit-monitoring-self-check>`:

- an increase of any other metric in the {ref}`self-check table <live-edit-metrics-reference-self-checks>`;
- once the versions no longer overlap, more than one increase for the same layer on the same pod.

(live-edit-operations-upgrade-time-zero)=
## Upgrading from a release that stores time samples at `-0.0` as written

Every earlier release (0.1.0 and 0.1.1) and every earlier 0.2.0 pre-release stores time samples at `-0.0` as written, so it can keep a sample at `-0.0` beside one at `0.0`.
This release keeps one sample, at `0.0`, and needs no action from you.
The service merges such samples by itself and counts them on `live_edit_time_sample_signed_zero_merge_total`.
Samples that old pods write during a rollout are merged within about one `gc.extReconcileIntervalSecs` (an hour by default) after the last old pod stops.
With `gc.extReconcileIntervalSecs: 0` nothing is merged, but reads stay correct.
Rolling back needs no step: merged samples stay merged, and earlier releases read them correctly.

During such an upgrade or a rollback to such a release, these self-checks can increase and fire the `LiveEditSelfCheckFired` warning:

- `live_edit_structural_cache_divergence_total`;
- `live_edit_structural_cache_audit_total{outcome="diverged"}`;
- rarely, `live_edit_update_layer_structural_fallbacks_total{reason="repeat_divergence"}`.

On an upgrade, expect any number of them until the `merged` and `rekeyed` outcomes of `live_edit_time_sample_signed_zero_merge_total` stop rising after the rollout.
After that, each affected layer can add at most one more increase on each replica.
On a rollback, expect them while old and new pods run side by side.
After the rollback completes, each affected layer can still add at most one increase on each replica.
These increases need no support case; the `WARN` line with each identifies the layer by its `object_id`.

Not expected, so follow {ref}`Self-check fired <live-edit-monitoring-self-check>`:

- an increase of any other metric in the {ref}`self-check table <live-edit-metrics-reference-self-checks>`, `live_edit_ext_index_drift_total{direction="under"}` included;
- after a rollback completes, more than one increase for the same layer on the same replica.

After an upgrade, also not expected:

- any increase more than about one `gc.extReconcileIntervalSecs` after the last old pod stops, if `merged` and `rekeyed` have not risen since the rollout began;
- once `merged` and `rekeyed` stop rising, more than one further increase for the same layer on the same replica.

(live-edit-operations-rollouts)=
## What to expect during a rollout

Each rollout step starts new pods before it drains old ones, so serving capacity never drops.
Each step adds about a third of the replica count (`strategy.rollingUpdate.maxSurge: "33%"`):

| Replicas | 1–3 | 4–6 | 7–9 | 10–12 | 13–15 | 16–18 | 19–21 |
|---|---|---|---|---|---|---|---|
| Extra pods during a rollout | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

A rollout takes at most 3 steps (up to 99 replicas), about 2.5 minutes with pods that start in 20 seconds, which fits Helm's default 5-minute `--wait`.
Each old pod ends its clients' streams gradually over up to 20 seconds, and those clients reconnect to other replicas.
Room the rollout needs: {ref}`Rollout headroom <live-edit-scaling-rollout-headroom>`.

- **No room for the extra pods:** set `strategy.rollingUpdate.maxSurge: 1`.
  Pods then move one at a time, about a minute each; set `--timeout` to about one minute per replica (`--timeout 10m` at 10 replicas).
- **A slow startup:** a pod that needs most of its {ref}`startup allowance <live-edit-operations-probes>` does not fit a 5-minute `--wait`.
  That happens only when the backing store or token provider is degraded; `--atomic` rolls the upgrade back, and you retry once it recovers.
- **Longer shutdown settings:** when you raise `shutdown.*` values or `rpc.deadlineMs`, raise `terminationGracePeriodSeconds` too.
  The chart refuses to render a grace period that is too short and names the values involved.
- **A smaller memory limit:** new pods exit at startup if `resources.limits.memory` is below what their settings need, naming the size; `--atomic` rolls the upgrade back, and you retry with a larger limit.

(live-edit-operations-probes)=
## Health probes and startup

The chart configures three gRPC health probes on port 50001; keep them.

| Probe | Health service | Acts after |
|---|---|---|
| `startupProbe` | `nvidia.omniverse.live.v1alpha.LiveService` | 62 failures 5 seconds apart (310 seconds): restarts the pod |
| `livenessProbe` | `""` (the process) | 3 failures 10 seconds apart: restarts the pod |
| `readinessProbe` | `nvidia.omniverse.live.v1alpha.LiveService` | 3 failures 5 seconds apart: removes the pod from the Service |

The gRPC port and `/metrics` answer within moments of start, and a normal start takes seconds.
A pod stopped while it is still starting, by a rollback or a scale-down, logs `received SIGTERM during startup; stopping before serving` and exits within about 11 seconds, without serving.
What readiness checks: {ref}`Health checks <live-edit-monitoring-health>`.

While the service is held for disaster recovery, readiness and liveness stay `SERVING`, so the probes neither restart held pods nor remove them from the Service ({ref}`Held for disaster recovery <live-edit-monitoring-held>`).
A pod that starts while the service is held, such as one Kubernetes reschedules, starts held and becomes ready.
It refuses every call `UNAVAILABLE` naming the incident, and still answers health checks, the capabilities service, `/metrics` and the autoscaler.
When the hold clears, each held pod exits within seconds and Kubernetes restarts it into normal service, so expect one restart on every pod.

If a pod, held or not, cannot read the backing store for 30 seconds, it reports not ready, so the readiness probe removes it from the Service.
A pod that is not held also refuses calls and ends its clients' streams, which reconnect to other pods.
Such a pod does not exit while the store stays unreachable: it keeps trying, and the liveness probe does not restart it.
When a read succeeds again, a pod that is not held exits within seconds and Kubernetes restarts it, so expect one restart per affected pod.
So a backing-store outage longer than 30 seconds takes every pod out of the Service until the store returns.
Pods that were running serve again about a minute at most after it returns (longer for a pod that already restarted in the last 10 minutes).
If one pod stays not ready while the others serve, a fault inside that pod keeps it from reading the store.
Delete that pod, and Kubernetes replaces it.
A starting pod reads once whether the service is held before it serves; if that read fails, the pod exits with an error and Kubernetes restarts it, under its usual backoff.

The 310-second allowance covers a degraded start: slow backing-store connections, a slow backup store, or a token provider that has not yet written its first token.
Do not lower `startupProbe.failureThreshold`, or pods that are still starting are restarted.
If you raise the backing-store timeouts, raise it to match:

| `valkey.connectTimeoutSecs` | `valkey.commandTimeoutMs` | `startupProbe.failureThreshold` |
|---|---|---|
| `30` (default) | `5000` (default) | `62` (default) |
| `45` | `5000` | `77` |
| `60` | `5000` | `92` |
| `60` | `10000` | `96` |

(live-edit-operations-rollback)=
## Roll back

Roll back the whole release:

```bash
helm history live-edit-service -n storage-apis
helm rollback live-edit-service {REVISION} -n storage-apis --wait
kubectl rollout status deployment/live-edit-service -n storage-apis
```

The rollback succeeds when every pod is ready; then repeat the smoke tests.
Do not roll back with `kubectl rollout undo`: it restores the pod template but not the configuration.
Do not roll back while a disaster-recovery Job runs: a rollback to a revision before its `dr` values deletes the Job; the service stays held, and a later run continues from what it wrote.

(live-edit-operations-credentials)=
## Rotate credentials

- **Backing-store URL or password** (Secret `live-edit-env`, key `valkey-url`): pods read it only at start.
  Update the Secret, then run `kubectl rollout restart deployment/live-edit-service -n storage-apis`.
  If the store accepts two passwords at once, add the new one before the restart and remove the old one after it.
- **Token files** (service identity, backing-store access tokens): the token provider refreshes them and the service rereads them, with no restart.
- **Image pull Secret** (`ngcpull-secret`): update it before the next rollout; running pods are unaffected.

Provider-specific rotation is on {ref}`Azure Managed Redis <live-edit-backing-store-azure>` and {ref}`Amazon MemoryDB <live-edit-backing-store-aws>`.

(live-edit-operations-references)=
## References

- [Helm upgrade](https://helm.sh/docs/helm/helm_upgrade/): `helm upgrade` flags, including `--wait`, `--atomic` and `--timeout`.
- [Helm rollback](https://helm.sh/docs/helm/helm_rollback/): rolling a release back to an earlier revision.
- [Kubernetes rolling updates](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#rolling-update-deployment): how `maxSurge` and `maxUnavailable` work.

(live-edit-operations-see-next)=
## See next

{ref}`Troubleshooting <live-edit-troubleshooting>`: symptoms, checks and fixes.
