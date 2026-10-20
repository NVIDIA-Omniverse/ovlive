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

(live-edit-backing-store-aws)=
# Amazon MemoryDB and ElastiCache

This page provisions an Amazon MemoryDB for Valkey cluster for Live Edit's backing store, with IAM authentication, and connects Live Edit to it.
Read {ref}`Backing store <live-edit-backing-store>` first for the requirements and sizing model.
MemoryDB writes every change to a Multi-AZ transaction log before acknowledging it, so a node or zone failure loses no committed edit.
ElastiCache for Valkey (node-based) works with the differences listed {ref}`below <live-edit-backing-store-aws-elasticache>`.

You need a VPC with private subnets in at least two availability zones, a security group for the cluster, and an EKS cluster with an OIDC provider.

(live-edit-backing-store-aws-parameters)=
## Parameters

```bash
REGION={AWS_REGION}
ACCOUNT={AWS_ACCOUNT_ID}
EKS={EKS_CLUSTER}
CLUSTER=live-edit-mdb
NODE_TYPE=db.r7g.large                 # development: db.t4g.small
SUBNET_GROUP=live-edit-subnets
SECURITY_GROUP={SECURITY_GROUP_ID}
PARAM_GROUP=default.memorydb-valkey7   # already noeviction
ACL_USER=live-edit
ACL=live-edit-acl
IAM_ROLE=live-edit-memorydb
NAMESPACE=storage-apis
SA=live-edit-service            # the chart's ServiceAccount, or your serviceAccount.name
```

(live-edit-backing-store-aws-param-group)=
## 1. Check the eviction policy

The default parameter group already sets `noeviction`.
If you use a custom group, set `maxmemory-policy` to `noeviction` in it.
Verify:

```bash
aws memorydb describe-parameters --parameter-group-name "$PARAM_GROUP" --region "$REGION" \
  --query "Parameters[?Name=='maxmemory-policy'].Value" --output text   # noeviction
```

(live-edit-backing-store-aws-subnets)=
## 2. Create the subnet group

```bash
aws memorydb create-subnet-group --subnet-group-name "$SUBNET_GROUP" \
  --subnet-ids {SUBNET_A} {SUBNET_B} --region "$REGION"
```

(live-edit-backing-store-aws-acl)=
## 3. Create the IAM-enabled user

```bash
aws memorydb create-user --user-name "$ACL_USER" \
  --authentication-mode Type=iam \
  --access-string "on ~le:* &le:* +@all" --region "$REGION"
aws memorydb create-acl --acl-name "$ACL" --user-names "$ACL_USER" --region "$REGION"
```

For a development cluster without IAM, use `--authentication-mode Type=password,Passwords={PASSWORD}` and static authentication with `rediss-cluster://{ACL_USER}:{PASSWORD}@{ENDPOINT}:6379` in the `live-edit-env` Secret.

(live-edit-backing-store-aws-create)=
## 4. Create the cluster

```bash
aws memorydb create-cluster --cluster-name "$CLUSTER" \
  --node-type "$NODE_TYPE" --engine valkey \
  --num-shards 2 --num-replicas-per-shard 1 \
  --acl-name "$ACL" --parameter-group-name "$PARAM_GROUP" \
  --subnet-group-name "$SUBNET_GROUP" --security-group-ids "$SECURITY_GROUP" \
  --tls-enabled --region "$REGION"
aws ec2 authorize-security-group-ingress --group-id "$SECURITY_GROUP" \
  --protocol tcp --port 6379 \
  --source-group {EKS_NODE_OR_CLUSTER_SECURITY_GROUP} --region "$REGION"
```

The ingress rule admits the pods; without it every connection times out.
For development, use `--node-type db.t4g.small --num-shards 1 --num-replicas-per-shard 0`.
Do not use `t4g` nodes in production: sustained script load exhausts their CPU credits.

(live-edit-backing-store-aws-iam)=
## 5. Create the IAM role

```bash
OIDC=$(aws eks describe-cluster --name "$EKS" --region "$REGION" \
  --query 'cluster.identity.oidc.issuer' --output text | sed 's~https://~~')
cat > trust-policy.json <<EOF
{ "Version": "2012-10-17", "Statement": [{ "Effect": "Allow",
  "Principal": { "Federated": "arn:aws:iam::${ACCOUNT}:oidc-provider/${OIDC}" },
  "Action": "sts:AssumeRoleWithWebIdentity",
  "Condition": { "StringEquals": {
    "${OIDC}:sub": "system:serviceaccount:${NAMESPACE}:${SA}",
    "${OIDC}:aud": "sts.amazonaws.com" } } }] }
EOF
cat > connect-policy.json <<EOF
{ "Version": "2012-10-17", "Statement": [{ "Effect": "Allow",
  "Action": "memorydb:connect",
  "Resource": [ "arn:aws:memorydb:${REGION}:${ACCOUNT}:cluster/${CLUSTER}",
                "arn:aws:memorydb:${REGION}:${ACCOUNT}:user/${ACL_USER}" ] }] }
EOF
aws iam create-role --role-name "$IAM_ROLE" --assume-role-policy-document file://trust-policy.json
aws iam put-role-policy --role-name "$IAM_ROLE" --policy-name memorydb-connect \
  --policy-document file://connect-policy.json
```

(live-edit-backing-store-aws-connect)=
## 6. Connect Live Edit

Read the endpoint:

```bash
aws memorydb describe-clusters --cluster-name "$CLUSTER" --region "$REGION" \
  --query 'Clusters[0].ClusterEndpoint.Address' --output text
```

Start from the {ref}`complete values example <live-edit-deployment-create-the-values-file>`.
Replace its `valkey` block and its `serviceAccount` annotation with these.
The IAM role authenticates Live Edit to MemoryDB only; the backing store needs no token file:

```yaml
serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: "arn:aws:iam::{AWS_ACCOUNT_ID}:role/live-edit-memorydb"
valkey:
  url:
    value: "rediss-cluster://{CLUSTER_ENDPOINT}:6379"
  credentials:
    awsIam:
      enabled: true
      service: memorydb
      username: "live-edit"
      region: "{AWS_REGION}"
      clusterHost: "live-edit-mdb"   # the cluster name, not the endpoint
```

After installing, check that the pods are ready and that `live_edit_store_memory_used_ratio` reports a value within a minute.
Leave `valkey.memoryBytes` unset: MemoryDB and ElastiCache report their memory limit.
With checkpoints disabled, a startup warning `could not verify Valkey AOF state` is expected: MemoryDB blocks `CONFIG`, and its transaction log replaces AOF.

(live-edit-backing-store-aws-storage-token)=
### Checkpoints to the Storage API on AWS

Live Edit needs a Storage service token only when checkpoints go to the Storage API (`checkpoint.adapter: StorageApi`).
The IAM role cannot produce that token.
Your token provisioner must obtain it from the identity provider your Storage API trusts, for example through a client-credentials flow with the client secret in a Kubernetes Secret, and write it in the {ref}`token file format <live-edit-authentication-token-file-format>`.
Without such a provisioner, drop the complete example's `serviceIdentity` block and its `StorageApi` checkpoint settings, and choose another option from {ref}`Durability <live-edit-durability>`; MemoryDB's transaction log and snapshots already protect against node and zone failure.

(live-edit-backing-store-aws-elasticache)=
## ElastiCache for Valkey

Apply the same eviction, TLS, cluster-mode and access-string settings, and change:

| Setting | MemoryDB | ElastiCache for Valkey |
|---|---|---|
| `valkey.credentials.awsIam.service` | `memorydb` | `elasticache` |
| IAM action to grant on the cache and user | `memorydb:connect` | `elasticache:Connect` |
| `clusterHost` | cluster name | replication group or cache name |

Grant `elasticache:Connect` on `arn:aws:elasticache:{AWS_REGION}:{AWS_ACCOUNT_ID}:replicationgroup:{NAME}` and `arn:aws:elasticache:{AWS_REGION}:{AWS_ACCOUNT_ID}:user:{USER}`.
Create the ElastiCache user with identical user name and user ID.

ElastiCache reserves 25 % of node memory by default (`reserved-memory-percent`), so choose a node whose remaining memory is about twice your expected peak.
ElastiCache replication is asynchronous: configure {ref}`durability <live-edit-durability>` as for any store without a transaction log.

(live-edit-backing-store-aws-sizing)=
## Sizing

- **Memory:** choose a node type whose published `maxmemory` is about twice your expected peak per shard (`db.r7g.large`: about 13 GiB).
  To tell when it is filling up, see {ref}`Store memory <live-edit-monitoring-store-memory>`.
- **CPU:** prefer `r7g` nodes, and add shards for write headroom.
- **Connections:** the limit is 65,000 per node, about four times the rollout peak at the chart defaults.

(live-edit-backing-store-aws-durability)=
## Durability

MemoryDB needs no AOF, and its snapshots add point-in-time restore.
Checkpoints are optional; add them if you need recovery from deletion or loss of the whole cluster ({ref}`Durability <live-edit-durability>`).

(live-edit-backing-store-aws-limitations)=
## Limitations

- A test confirms MemoryDB allows the script reads Live Edit requires; Live Edit itself has not been tested on MemoryDB.
- While MemoryDB or ElastiCache moves data between shards, writes to the layers and sessions being moved can be refused, and their backups can lag, until their move ends, as {ref}`When the store moves data between shards <live-edit-backing-store-moving-data>` describes.
- ElastiCache for Valkey has not been tested.

(live-edit-backing-store-aws-references)=
## References

- [MemoryDB IAM authentication](https://docs.aws.amazon.com/MemoryDB/latest/devguide/auth-iam.html): users and `memorydb:connect`.
- [MemoryDB parameters](https://docs.aws.amazon.com/memorydb/latest/devguide/parametergroups.redis.html): `maxmemory` per node type and `maxclients`.
- [ElastiCache reserved memory](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/redis-memory-management.html): `reserved-memory-percent`.
- [ElastiCache IAM authentication](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/auth-iam.html): users, ARNs and `elasticache:Connect`.
- [IAM roles for service accounts](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html): IRSA setup.

(live-edit-backing-store-aws-see-next)=
## See next

{ref}`Durability <live-edit-durability>`: decide whether to add checkpoints.
