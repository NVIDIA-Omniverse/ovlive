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

(live-edit-backing-store-self-hosted)=
# Self-hosted Valkey

This page sets up Valkey in your own Kubernetes cluster with the Bitnami Valkey Helm charts, for Live Edit's backing store.
Read {ref}`Backing store <live-edit-backing-store>` first for the requirements and sizing model.
Self-hosting gives you full control of version, network and persistence, and makes you responsible for memory sizing, eviction policy, durability, failover and monitoring.
The charts' defaults do not meet Live Edit's requirements, so set every value on this page.

(live-edit-backing-store-self-hosted-topology)=
## Choose a topology

| Chart and mode | Failover | Use for |
|---|---|---|
| `bitnami/valkey`, `architecture: standalone` | none | development and evaluation |
| `bitnami/valkey`, `architecture: replication` | manual only | a data copy without automatic failover |
| `bitnami/valkey-cluster` | automatic | production |

Choose `standalone` when losing the store for a while is acceptable: losing its volume or node means downtime and the loss of writes since the last checkpoint.
Choose `valkey-cluster` for production: it promotes a replica when a primary fails and lets you add shards for memory and write capacity.
Read replicas in `replication` mode do not offload Live Edit; all its traffic goes to the primary.

```{warning}
Sentinel mode (`sentinel.enabled: true`) is not supported.
It removes the `<release>-primary` Service Live Edit connects to, and the Service it creates instead can route writes to a read-only replica.
```

(live-edit-backing-store-self-hosted-images)=
## Mirror the images

Bitnami's free catalog no longer publishes versioned images, so the chart's default image may not pull.
Copy the chart and the image you validate into a registry you control, pin both, and point the chart at your copy:

```yaml
global:
  security:
    allowInsecureImages: true   # required when the image is not from Bitnami's registry
image:
  registry: registry.example.com
  repository: mirror/valkey
  tag: "{PINNED_TAG}"
  pullSecrets:
    - name: registry-pull-secret
```

(live-edit-backing-store-self-hosted-config)=
## Required configuration

The `bitnami/valkey` chart sets neither a memory limit nor an eviction policy.
Set these in `commonConfiguration`:

```yaml
commonConfiguration: |-
  appendonly yes
  appendfsync everysec
  save ""
  maxmemory 3072mb
  maxmemory-policy noeviction
  maxclients 20000
```

- `maxmemory-policy noeviction` is required.
- `maxmemory` must sit below the pod's memory limit, leaving about 20 % for persistence and replication buffers.
  Without it, the kernel's out-of-memory killer, not Valkey, decides what happens when memory runs out.
- `maxclients 20000` covers a rollout at the chart's default `autoscaling.maxReplicas` of 10; Valkey's default of 10,000 does not.
  Raise it if you raise `autoscaling.maxReplicas` ({ref}`Connections <live-edit-backing-store-connections>`).
- `appendonly yes` with `appendfsync everysec` limits crash loss to about one second; `save ""` turns off RDB snapshots, which AOF replaces.

(live-edit-backing-store-self-hosted-auth)=
## Authentication

Deliver the password from its own Secret, never inline in values, and create it before installing Valkey:

```bash
kubectl create secret generic live-edit-valkey \
  --from-literal=valkey-password="${VALKEY_PASSWORD}" --namespace storage-apis
```

Keep it separate from Live Edit's `live-edit-env` Secret, which the Live Edit uninstall steps delete.
When you rotate the password, update both Secrets.

```yaml
auth:
  enabled: true
  existingSecret: live-edit-valkey
  existingSecretPasswordKey: valkey-password
```

(live-edit-backing-store-self-hosted-tls)=
## TLS

```yaml
tls:
  enabled: true
  authClients: false          # Live Edit presents no client certificate
  existingSecret: live-edit-valkey-tls
  certFilename: tls.crt
  certKeyFilename: tls.key
  certCAFilename: ca.crt
```

`live-edit-valkey-tls` holds the server certificate, key and CA, usually issued by cert-manager.
Set `authClients: false`: with the chart default, `true`, Valkey demands a client certificate and rejects every Live Edit connection.

Live Edit verifies the server certificate against the public root certificates in its container image, not against the chart's `ca.crt`.
If a private CA (including a cert-manager issuer) signs the certificate, startup fails with an unknown-issuer error until you add that CA.
Build one PEM bundle of the public roots plus your CA, store it in a ConfigMap, and mount it over the image's bundle in your Live Edit values:

```yaml
volumes:
  - name: ca-bundle
    configMap:
      name: live-edit-ca-bundle    # key ca-certificates.crt: public roots + your CA
volumeMounts:
  - name: ca-bundle
    mountPath: /etc/ssl/certs/ca-certificates.crt
    subPath: ca-certificates.crt
    readOnly: true
```

Keep the public roots in the bundle: it replaces the image's bundle, and Live Edit also needs them for its other TLS connections, such as authentication services and telemetry collectors.

(live-edit-backing-store-self-hosted-persistence)=
## Persistence

Size the volume at least twice `maxmemory`, since AOF rewrites briefly need double space, and use a StorageClass with `allowVolumeExpansion: true`:

```yaml
primary:
  persistence:
    enabled: true
    size: 8Gi
    storageClass: {EXPANDABLE_STORAGE_CLASS}
```

(live-edit-backing-store-self-hosted-starting-points)=
## Starting values

**Development** (`bitnami/valkey`, standalone):

```yaml
architecture: standalone
auth:
  enabled: true
  existingSecret: live-edit-valkey
  existingSecretPasswordKey: valkey-password
commonConfiguration: |-
  appendonly yes
  appendfsync everysec
  save ""
  maxmemory 1500mb
  maxmemory-policy noeviction
  maxclients 20000
tls:
  enabled: true
  authClients: false
  existingSecret: live-edit-valkey-tls
  certFilename: tls.crt
  certKeyFilename: tls.key
  certCAFilename: ca.crt
primary:
  resources:                 # replace the nano preset: scripts need a full core
    requests: { cpu: "1",  memory: 2Gi }
    limits:   { cpu: "1",  memory: 2Gi }
  persistence:
    enabled: true
    size: 8Gi
```

**Production** (`bitnami/valkey-cluster`, 3 primaries with one replica each):

```yaml
cluster:
  nodes: 6
  replicas: 1
existingSecret: live-edit-valkey
existingSecretPasswordKey: valkey-password
tls:
  enabled: true
  authClients: false
  existingSecret: live-edit-valkey-tls
  certFilename: tls.crt
  certKeyFilename: tls.key
  certCAFilename: ca.crt
valkey:
  useAOFPersistence: "yes"
  configmap: |-
    maxmemory 3072mb
    maxmemory-policy noeviction
    maxclients 20000
  resources:
    requests: { cpu: "2", memory: 4Gi }
    limits:   { cpu: "2", memory: 6Gi }
persistence:
  size: 8Gi
  storageClass: {EXPANDABLE_STORAGE_CLASS}
metrics:
  enabled: true              # exporter for the memory and CPU signals
```

`maxmemory` is per node.
Give each primary at least one dedicated core, two for production; never use the `nano` preset.
Add the image settings above, then install the chart that matches the file.
Name a cluster release with `valkey-cluster` in it so its Service keeps that name:

```bash
# development
helm install live-edit-valkey oci://registry-1.docker.io/bitnamicharts/valkey \
  --namespace storage-apis -f valkey-values.yaml
# production
helm install live-edit-valkey-cluster oci://registry-1.docker.io/bitnamicharts/valkey-cluster \
  --namespace storage-apis -f valkey-values.yaml
```

(live-edit-backing-store-self-hosted-connect)=
## Connect Live Edit

Put the URL, with the URL-encoded password, in the `live-edit-env` Secret's `valkey-url` key, as {ref}`Deployment <live-edit-deployment>` shows:

| Topology | URL |
|---|---|
| `standalone` or `replication` | `rediss://:{PASSWORD}@{RELEASE}-primary:6379/0` |
| `valkey-cluster` | `rediss-cluster://:{PASSWORD}@{RELEASE}:6379` |

Leave `valkey.credentials` at its defaults (password in the URL).

(live-edit-backing-store-self-hosted-verify)=
## Verify

From `valkey-cli` connected to each primary, run:

```bash
CONFIG GET maxmemory-policy   # noeviction
CONFIG GET maxclients         # 20000; Valkey lowers it if the open-file limit is too low
INFO persistence              # aof_enabled:1
```

After installing Live Edit, check that its pods are ready and that `live_edit_store_memory_used_ratio` reports a value within a minute.
Leave `valkey.memoryBytes` unset: Valkey reports its `maxmemory` to Live Edit.

(live-edit-backing-store-self-hosted-durability)=
## Durability

Live Edit checks AOF at startup on self-hosted Valkey: with AOF and checkpoints both off, it refuses to start.
A failover loses writes the replica had not yet received, so pair AOF with {ref}`checkpoints <live-edit-durability>` in production.

(live-edit-backing-store-self-hosted-scale)=
## When to grow

| Signal | Threshold | Action |
|---|---|---|
| `live_edit_store_memory_used_ratio` | as {ref}`Store memory <live-edit-monitoring-store-memory>` says | raise `maxmemory` and the memory limit together; past one node's practical memory, move to a cluster |
| Primary CPU (optional, if you run the Valkey exporter) | above 0.8 core sustained | raise CPU to 2–4 cores; past one saturated core, add shards |
| Persistence volume used ÷ capacity (optional, if you run the Valkey exporter) | above 0.75 | grow the volume |
| `connected_clients` ÷ `maxclients` (optional, if you run the Valkey exporter) | above 0.70 outside rollouts | raise `maxclients` or add shards |
| `live_edit_store_write_refused_total{reason="out_of_memory"}` | any increase | the store is full: raise `maxmemory` and the memory limit now, even if the refusals stop by themselves |

Raising memory or CPU restarts the Valkey pods: a short failover on a cluster, a write outage on a single primary.

(live-edit-backing-store-self-hosted-cluster)=
## Moving to a cluster

Moving from `bitnami/valkey` to `bitnami/valkey-cluster` is a migration to a new chart with a data move, not an in-place upgrade; plan a maintenance window.
Live Edit needs no change other than the new `rediss-cluster://` URL.
Live Edit verifies every node's certificate against the hostname in the URL.
If your nodes advertise IP addresses and each has its own certificate, include that hostname in every certificate's subject alternative names.
The startup log line `Valkey transport configured` shows `tls_host_mapping=default-host` when this hostname check is in effect.

(live-edit-backing-store-self-hosted-add-shards)=
## Adding shards

Adding or removing shards moves hash slots between primaries.
A key-by-key move refuses writes to the layers and sessions in each moving slot until that slot's move ends ({ref}`When the store moves data between shards <live-edit-backing-store-moving-data>`).
This chart's own way of adding nodes, `cluster.update.addNodes: true` on `helm upgrade`, rebalances with a key-by-key move, so run it off-peak.

To keep writes flowing, move slots with atomic slot migration, which needs Valkey 9.0 or later on every node:

- With `valkey-cli` 9.1 or later, reshard with `valkey-cli --cluster reshard {NODE}:6379 --cluster-use-atomic-slot-migration`, adding the TLS and password options you use with `valkey-cli`.
- With an older `valkey-cli`, send the move to the slots' current owner, naming the target by its ID from `CLUSTER NODES`, then poll until it completes:

  ```bash
  valkey-cli -h {SOURCE_NODE} -p 6379 CLUSTER MIGRATESLOTS SLOTSRANGE {FIRST_SLOT} {LAST_SLOT} NODE {TARGET_NODE_ID}
  valkey-cli -h {SOURCE_NODE} -p 6379 CLUSTER GETSLOTMIGRATIONS
  ```

On Valkey before 9.0, reshard off-peak, a few slots at a time.

```{warning}
Let every slot's move finish.
Never assign a slot to another node with `CLUSTER SETSLOT … NODE` while keys remain on its old owner: the store can delete those keys.
```

Verify: `valkey-cli --cluster check {NODE}:6379`, with your TLS and password options, reports `All 16384 slots covered` and no open slots.

(live-edit-backing-store-self-hosted-references)=
## References

- [Bitnami Valkey chart](https://github.com/bitnami/charts/tree/main/bitnami/valkey): standalone and replication modes.
- [Bitnami Valkey Cluster chart](https://github.com/bitnami/charts/tree/main/bitnami/valkey-cluster): the clustered chart.
- [Bitnami catalog changes](https://github.com/bitnami/charts/issues/35164): why to mirror images.
- [`valkey.conf`](https://github.com/valkey-io/valkey/blob/unstable/valkey.conf): `maxclients` and other server settings.
- [Atomic slot migration](https://valkey.io/topics/atomic-slot-migration/): moving slots with `CLUSTER MIGRATESLOTS` and checking progress.

(live-edit-backing-store-self-hosted-see-next)=
## See next

{ref}`Durability <live-edit-durability>`: choose AOF, checkpoints or both.
