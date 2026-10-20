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

(live-edit-backing-store-azure)=
# Azure Managed Redis

This page provisions an Azure Managed Redis cache for Live Edit's backing store, with Microsoft Entra ID authentication and private networking, and connects Live Edit to it.
Read {ref}`Backing store <live-edit-backing-store>` first for the requirements and sizing model.
Use Azure Managed Redis, not Azure Cache for Redis or its Enterprise tiers.

(live-edit-backing-store-azure-before)=
## Before you start

- Install the `az` CLI and the `redisenterprise` extension 2.75.0 or newer: `az extension add -n redisenterprise`.
- Select the subscription that will hold the cache: `az account set --subscription {SUBSCRIPTION}`, then confirm with `az account show`.
- Enable the OIDC issuer and workload identity on your AKS cluster.
- Confirm Azure Managed Redis is offered in your AKS cluster's region, and create the cache in that region.

Two settings are fixed when the cache is created and need a new cache to change:

- **Clustering policy `OSSCluster`.** Live Edit connects to the shards directly; `EnterpriseCluster` funnels traffic through one proxy and `NoCluster` cannot scale out.
- **Eviction policy `NoEviction`.**

(live-edit-backing-store-azure-size)=
## Choose a size

Use the Balanced tier:

| SKU | Memory | Use for | Max `autoscaling.maxReplicas` |
|---|---|---|---|
| `Balanced_B0` | 0.5 GB | evaluation | 7 |
| `Balanced_B1` | 1 GB | development | 7 |
| `Balanced_B3` | 3 GB | small workloads | 7 |
| `Balanced_B5` | 6 GB | small workloads | 7 |
| `Balanced_B10` | 12 GB | production starting point | 10 |
| `Balanced_B20` | 24 GB | larger production | 10 |

1. Pick the smallest size whose memory is at least twice your expected peak.
2. Set `autoscaling.maxReplicas` to 7 on `Balanced_B0` to `Balanced_B5`; larger sizes take the chart default of 10.
3. Set `valkey.memoryBytes` to your SKU's memory in bytes, for example `12000000000` for `Balanced_B10`.
   The cache does not report its memory to Live Edit; the service works out each shard's share itself.
   After resizing the cache, update `valkey.memoryBytes` to the new total.

The SKU suffix is not the size in GB.

(live-edit-backing-store-azure-create)=
## 1. Create the cache

Set your parameters once:

```bash
RG=live-edit-rg
LOC={REGION}
CACHE=live-edit-amr
SKU=Balanced_B10              # development: Balanced_B1
IDENTITY=live-edit-amr-id
AKS={AKS_CLUSTER}
AKS_RG={AKS_RESOURCE_GROUP}
VNET={VNET}
SUBNET={SUBNET}
NAMESPACE=storage-apis
SA=live-edit-service            # the chart's ServiceAccount, or your serviceAccount.name
```

```bash
az group create -g "$RG" -l "$LOC"
az redisenterprise create \
  --cluster-name "$CACHE" --resource-group "$RG" --location "$LOC" \
  --sku "$SKU" \
  --clustering-policy OSSCluster \
  --eviction-policy NoEviction \
  --client-protocol Encrypted \
  --minimum-tls-version 1.2 \
  --high-availability Enabled \
  --persistence aof-enabled=true aof-frequency="1s" \
  --access-keys-auth Disabled \
  --public-network-access Disabled \
  --port 10000
az redisenterprise wait --cluster-name "$CACHE" -g "$RG" --created
```

- Do not pass `--zones` or `--capacity`: Balanced sizes reject them, and zone redundancy is automatic where the region supports it.
- For development, use `--sku Balanced_B1 --high-availability Disabled` and `--public-network-access Enabled` with an IP allowlist.
  Without high availability the cache has no persistence.
- A create that is accepted and then fails with a bare `OperationFailed` is a subscription or region capacity or policy restriction; resolve it with your Azure administrator.

(live-edit-backing-store-azure-identity)=
## 2. Create the managed identity

```bash
az identity create -g "$RG" -n "$IDENTITY"
OBJECT_ID=$(az identity show -g "$RG" -n "$IDENTITY" --query principalId -o tsv)   # Object ID, not client ID
CLIENT_ID=$(az identity show -g "$RG" -n "$IDENTITY" --query clientId -o tsv)
OIDC=$(az aks show -g "$AKS_RG" -n "$AKS" --query oidcIssuerProfile.issuerUrl -o tsv)
az identity federated-credential create \
  --name live-edit --identity-name "$IDENTITY" -g "$RG" \
  --issuer "$OIDC" \
  --subject "system:serviceaccount:${NAMESPACE}:${SA}" \
  --audience api://AzureADTokenExchange
```

(live-edit-backing-store-azure-access-policy)=
## 3. Grant data access

The identity needs a data-access policy on the cache, not an Azure role.
Assign it, then limit it to Live Edit's keys and channels:

```bash
az redisenterprise database access-policy-assignment create \
  --cluster-name "$CACHE" --database-name default --resource-group "$RG" \
  --access-policy-assignment-name liveedit --access-policy-name default \
  --object-id "$OBJECT_ID"
SUB=$(az account show --query id -o tsv)
POLICY="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Cache/redisEnterprise/$CACHE/databases/default/accessPolicyAssignments/liveedit"
az resource update --latest-include-preview --ids "$POLICY" \
  --set 'properties.accessString=+@all ~le:* &le:*'
```

Apply the access string once and wait for it before any other change: overlapping updates can leave the assignment `Failed`, which blocks authentication.
Verify:

```bash
az resource show --latest-include-preview --ids "$POLICY" \
  --query properties.provisioningState -o tsv   # expect: Succeeded
```

(live-edit-backing-store-azure-network)=
## 4. Connect the private network

```bash
CACHE_ID=$(az redisenterprise show --cluster-name "$CACHE" -g "$RG" --query id -o tsv)
az network private-endpoint create -g "$RG" -n "${CACHE}-pe" \
  --vnet-name "$VNET" --subnet "$SUBNET" \
  --private-connection-resource-id "$CACHE_ID" \
  --group-id redisEnterprise --connection-name "${CACHE}-conn"
az network private-dns zone create -g "$RG" -n privatelink.redis.azure.net
az network private-dns link vnet create -g "$RG" --zone-name privatelink.redis.azure.net \
  --name "${VNET}-link" --virtual-network "$VNET" --registration-enabled false
az network private-endpoint dns-zone-group create -g "$RG" --endpoint-name "${CACHE}-pe" \
  --name default --private-dns-zone privatelink.redis.azure.net --zone-name redis
```

Allow port 10000 and the shard port range 8500–8599 in network security rules, outbound from the pods too.
Leave the cache's idle-client timeout at its default (off).

(live-edit-backing-store-azure-connect)=
## 5. Connect Live Edit

Verify the cache first:

```bash
az redisenterprise test-connection --cluster-name "$CACHE" -g "$RG" --auth entra
```

Start from the {ref}`complete values example <live-edit-deployment-create-the-values-file>`, which already covers this cache, and set these values for yours:

```yaml
valkey:
  url:
    value: "rediss-cluster://{CACHE}.{REGION}.redis.azure.net:10000"
  memoryBytes: 12000000000   # your SKU's memory in bytes (Balanced_B10)
  credentials:
    accessTokenFile:
      enabled: true
      username: "{OBJECT_ID}"
autoscaling:
  maxReplicas: 10             # 7 on Balanced_B0 to Balanced_B5
```

Set `serviceAccount.annotations."azure.workload.identity/client-id"` to `$CLIENT_ID`.
The URL carries no credential, so it need not be a Secret.

After installing, check that the pods are ready and that `live_edit_store_memory_used_ratio` reports a value within two minutes; on this cache it first appears about a minute after a pod starts.

(live-edit-backing-store-azure-token-refresh)=
## Token refresh

Configure your token provisioner to request the scope `https://redis.azure.com/.default` and to replace the Redis token at least 7 minutes (420 s) before it expires, in the {ref}`token file format <live-edit-authentication-token-file-format>`.

If the lead is too short, each token rotation (typically daily) logs a warning naming the refresh headroom, and live updates pause briefly while connections reconnect.
A replica with no valid token reports `NOT_SERVING` until the provisioner writes one.

(live-edit-backing-store-azure-durability)=
## Durability

AOF persistence, enabled at create above, requires high availability; a development cache without it keeps nothing across a restart, and Live Edit cannot detect that at startup.
Azure's own backups run at most hourly, so restoring one can lose up to an hour of edits.
AOF cannot be combined with active geo-replication.
A failover can lose the last writes the replica had not yet received.
For recovery from loss of the whole cache, enable {ref}`checkpoints <live-edit-durability>`.

(live-edit-backing-store-azure-monitoring)=
## Monitoring and maintenance

To tell when the cache is filling up, see {ref}`Store memory <live-edit-monitoring-store-memory>`.

During Azure maintenance and token rotation the cache can drop or silently stall connections; Live Edit detects and rebuilds them.
Expect occasional increases of `live_edit_valkey_consumer_forced_reconnects_total` around those events, and at most a brief pause in live updates.
These should stay at zero: `live_edit_valkey_consumer_permanent_exits_total` and `live_edit_stream_mux_reauths_total{result="failed"}`.

(live-edit-backing-store-azure-wrongpass)=
## Troubleshooting `WRONGPASS`

The cache answers every rejected Entra login with `WRONGPASS invalid username-password pair`.
Live Edit logs a checklist with an `identity_verdict` field naming the causes it ruled out: one warning at startup (`phase=connect`), and an error at most every five minutes if a running replica is refused (`phase=steady-state`).
Work through the causes in order:

1. **Client ID instead of Object ID.** `valkey.credentials.accessTokenFile.username` must be the identity's Object ID (`principalId`). The log prints the Object ID to use.
2. **Wrong token audience.** The token provisioner must request `https://redis.azure.com/.default`. The log prints the token's `aud` claim; compare it with that resource.
3. **Wrong identity.** The token's `oid` does not match the username: the ServiceAccount's `azure.workload.identity/client-id` annotation is missing or names another identity, or the identity was re-created with a new Object ID. The log prints both IDs.
4. **No access policy, or not `Succeeded`.** Check the assignment:

   ```bash
   az resource show --latest-include-preview --ids "$POLICY" \
     --query "{state:properties.provisioningState, objectId:properties.objectId}" -o json
   ```

   Wait while it is `Updating`.
   `Failed` blocks authentication; re-apply a valid access string and wait for `Succeeded`.

(live-edit-backing-store-azure-limitations)=
## Limitations

While Azure moves data between shards during a scale operation, writes to the layers and sessions being moved can be refused, and their backups can lag, until their move ends, as {ref}`When the store moves data between shards <live-edit-backing-store-moving-data>` describes.

(live-edit-backing-store-azure-access-keys)=
## Access-key fallback

For a bootstrap without Entra, create the cache with `--access-keys-auth Enabled`, read a key, and use static authentication with the URL in the `live-edit-env` Secret:

```bash
az redisenterprise database list-keys --cluster-name "$CACHE" -g "$RG" --database-name default
# rediss-cluster://default:{ACCESS_KEY}@{CACHE}.{REGION}.redis.azure.net:10000
```

Prefer Entra in production.

(live-edit-backing-store-azure-references)=
## References

- [What is Azure Managed Redis?](https://learn.microsoft.com/en-us/azure/redis/overview): tiers and connection limits per size.
- [Azure Managed Redis architecture](https://learn.microsoft.com/en-us/azure/redis/architecture): clustering policies, shards, reserved memory.
- [Azure Managed Redis pricing](https://azure.microsoft.com/en-us/pricing/details/managed-redis/): every SKU and its size.
- [`az redisenterprise`](https://learn.microsoft.com/en-us/cli/azure/redisenterprise): CLI reference.
- [Persistence](https://learn.microsoft.com/en-us/azure/redis/how-to-persistence): AOF and RDB options.
- [Entra authentication](https://learn.microsoft.com/en-us/azure/redis/entra-for-authentication): data-access policies.
- [AKS workload identity](https://learn.microsoft.com/en-us/azure/aks/workload-identity-overview): the ServiceAccount annotation and pod label.

(live-edit-backing-store-azure-see-next)=
## See next

{ref}`Durability <live-edit-durability>`: choose checkpoints for recovery from loss of the cache.
