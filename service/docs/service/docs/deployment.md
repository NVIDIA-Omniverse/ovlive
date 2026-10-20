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

(live-edit-deployment)=
# Deploy the Live Edit Service

**Goal**: Install Live Edit in Kubernetes from the NGC Helm chart, connected to a durable backing store and your Storage Service.
This page takes you from the prerequisites to a verified installation, then covers Discovery registration and uninstalling.
Backing-store setup, durability and authentication have their own pages; each step links to the page it needs.

```{important}
The chart does not start with its default values.
By default checkpoints are enabled (`checkpoint.enabled: true`) with no checkpoint storage (`checkpoint.adapter: Disabled`), and every replica exits with `live-edit refuses to start with no durability configured`.
Choose a durability mode in {ref}`Configure checkpoints <live-edit-deployment-checkpoints>` before you install.
```

(live-edit-deployment-ports)=
## Service ports

| Port | Protocol | Purpose |
|---|---|---|
| 50001 | gRPC over HTTP/2 | Live Edit API, gRPC health checks and Discovery capabilities |
| 9090 | HTTP | Prometheus `/metrics` |

The service accepts plaintext gRPC inside the cluster.
Terminate client TLS at an HTTP/2-capable ingress, and restrict direct access to both ports.

(live-edit-deployment-prerequisites)=
## Prerequisites

Have these ready before you install:

- A Kubernetes cluster that supports gRPC probes (1.24 or later), Helm 3, `kubectl`, and an NGC API key that can pull the Live Edit chart and images.
- A Valkey-compatible backing store that meets the {ref}`backing-store requirements <live-edit-backing-store>`, set up for your provider: {ref}`self-hosted <live-edit-backing-store-self-hosted>`, {ref}`Azure Managed Redis <live-edit-backing-store-azure>` or {ref}`Amazon MemoryDB <live-edit-backing-store-aws>`.
  Give each Live Edit deployment its own store.
- An authenticated Storage Service that holds the base assets your clients collaborate on and validates callers' bearer tokens.
  {ref}`Authentication <live-edit-authentication>` lists what Live Edit needs from it.
- A durability decision ({ref}`Durability <live-edit-durability>`).
  Storage-backed checkpoints need a dedicated backup location in your Storage Service and a service identity with read, list, write and delete access to it.
- For clients outside the cluster: Contour (or another HTTP/2-capable ingress), a DNS name and a TLS certificate.
- Network egress from the Live Edit pods to DNS, every backing-store node and port, the Storage Service, the User Info service if you use it, your identity provider, any storage endpoints that checkpoint uploads are redirected to, and your telemetry collector.

The Storage endpoint, a base asset address and the checkpoint location are three different settings.
For example, the endpoint can be `https://storage.example.com`, a base asset `https://assets.example.com/project/scene.usd`, and the checkpoint location `https://assets.example.com/live-edit-backups/`.

(live-edit-deployment-pull-the-helm-chart)=
## Pull the Helm chart

Replace `{VERSION}` with the release you are installing; the chart deploys the container image of the same version by default.

```bash
export NGC_API_KEY={NGC_API_KEY}

# pull the chart from NGC and unpack it
helm fetch "https://helm.ngc.nvidia.com/nvidia/omniverse/charts/live-edit-service-{VERSION}.tgz" \
  --username='$oauthtoken' --password="${NGC_API_KEY}"
tar -xvf "live-edit-service-{VERSION}.tgz"
```

Run the remaining commands from the directory that holds the unpacked `live-edit-service/` chart.
Keep your values file in this directory, not inside the chart, so that unpacking a newer chart for an upgrade leaves it in place.

(live-edit-deployment-create-secrets)=
## Create secrets

The examples use the namespace `storage-apis`; substitute yours throughout.

```bash
# create the namespace if it does not exist
kubectl create namespace storage-apis

# image-pull secret for nvcr.io
kubectl create secret docker-registry ngcpull-secret \
  --docker-server=nvcr.io --docker-username='$oauthtoken' \
  --docker-password="${NGC_API_KEY}" --namespace storage-apis
```

If your backing-store URL carries a password (password or ACL authentication), store the whole URL in a Secret named `live-edit-env` under the key `valkey-url`:

```bash
kubectl create secret generic live-edit-env \
  --from-literal=valkey-url="{VALKEY_URL}" \
  --namespace storage-apis
```

URL-encode reserved characters in the password, and never put a credential-bearing URL in Helm values.
For a URL without a credential (token-based or IAM authentication), skip this Secret and set the URL as shown in {ref}`Connection URL <live-edit-backing-store-connection-url>`.

Verify (leave out `live-edit-env` if you skipped it):

```bash
kubectl get secret ngcpull-secret live-edit-env -n storage-apis
```

(live-edit-deployment-create-the-values-file)=
## Create the values file

Create `live-edit-values.yaml`.
This is the complete file for the main production path: Azure Managed Redis with Microsoft Entra ID, Storage-backed checkpoints and the token sidecar.
The other pages show only what changes from it.

```yaml
image:
  pullSecrets:
    - name: ngcpull-secret

# workload identity for the token sidecar
serviceAccount:
  annotations:
    azure.workload.identity/client-id: "{MANAGED_IDENTITY_CLIENT_ID}"

# access checks against your Storage Service
authorization:
  storageServiceEndpoint: "https://storage.example.com"

# backing store: Azure Managed Redis with Entra ID
valkey:
  url:
    value: "rediss-cluster://{CACHE_NAME}.{REGION}.redis.azure.net:10000"
  memoryBytes: 12000000000   # the cache's total memory (Balanced_B10); see the Azure size table
  credentials:
    accessTokenFile:
      enabled: true
      username: "{MANAGED_IDENTITY_OBJECT_ID}"

# token provisioner: writes the Storage and backing-store token files
serviceIdentity:
  accessTokenFile:
    enabled: true
    provisionerSidecar:
      container:
        image: "{TOKEN_PROVISIONER_IMAGE}"
        env: []   # your provisioner's settings: identity, token scopes, file paths

# Storage-backed checkpoints
checkpoint:
  enabled: true
  adapter: StorageApi
  storageApiEndpoint: "https://storage.example.com"
  basePath: "https://assets.example.com/live-edit-backups/"

autoscaling:
  maxReplicas: 10
```

Where each value comes from:

- `serviceAccount` and `serviceIdentity`: {ref}`Configure the token sidecar <live-edit-authentication-configure-the-token-sidecar>`.
  The chart creates the ServiceAccount `live-edit-service` for the release name used on this page; federate the managed identity to it.
- `authorization`: your Storage Service REST endpoint, with its `https://` scheme ({ref}`Request authorization <live-edit-authentication-user-token-authorization>`).
- `valkey`: {ref}`Connect Live Edit <live-edit-backing-store-azure-connect>`, including `memoryBytes`, the cache's total memory for your size from {ref}`Choose a size <live-edit-backing-store-azure-size>`.
- `checkpoint`: {ref}`Configure checkpoints <live-edit-deployment-checkpoints>`.
- `autoscaling.maxReplicas`: keep it at 10 or fewer, and at 7 or fewer on Azure Managed Redis `Balanced_B0` to `Balanced_B5` ({ref}`Choose a size <live-edit-backing-store-azure-size>`).
  The chart's defaults also run an autoscaler and a disruption budget ({ref}`Scaling <live-edit-scaling>`).

For another backing store, replace the `valkey` block and the provisioner's backing-store token:

- Amazon MemoryDB: {ref}`Connect Live Edit <live-edit-backing-store-aws-connect>`.
- Self-hosted Valkey, or any store with a password in its URL: delete the `valkey` block and keep the URL in the `live-edit-env` Secret from {ref}`Create secrets <live-edit-deployment-create-secrets>`.

Clients that send opaque (non-JWT) tokens also need `userInfo` ({ref}`Caller identity <live-edit-authentication-user-identity>`).
Every other setting is in {ref}`Configuration <live-edit-configuration>`.
When you add a snippet from another page, merge it into the matching block; a top-level key may appear only once.

(live-edit-deployment-checkpoints)=
## Configure checkpoints

This step is required: with the chart defaults, the service refuses to start ({ref}`The startup durability check <live-edit-durability-startup-check>`).
{ref}`Durability <live-edit-durability>` explains what each mode protects against.
Choose one:

- **Storage-backed checkpoints**, recommended for production: choose this when you must recover live sessions after losing the whole backing store.
  Keep the `checkpoint` block of the example, with a `basePath` dedicated to this deployment, and the token sidecar.
- **Backing-store persistence only**: choose this when your store persists its data (the append-only file on self-hosted Valkey, or the managed provider's persistence) and losing the whole store is an acceptable risk.
  Replace the `checkpoint` block with:

  ```yaml
  checkpoint:
    enabled: false
  ```

  Keep the provisioner's Storage token unless you also remove the sidecar: while `serviceIdentity.accessTokenFile.enabled` is `true`, replicas need it to become ready.

```{warning}
Without checkpoints, losing the whole backing store loses every live session and layer.
```

Verify the choice renders before you install:

```bash
helm template live-edit-service ./live-edit-service -f live-edit-values.yaml \
  | grep -E 'LIVE_EDIT_CHECKPOINT_(ENABLED|ADAPTER)'
```

The output shows `LIVE_EDIT_CHECKPOINT_ENABLED` as `"false"`, or `"true"` with `LIVE_EDIT_CHECKPOINT_ADAPTER` as `"StorageApi"`.

(live-edit-deployment-configure-ingress)=
## Configure ingress

To reach the service from outside the cluster, the chart can create a Contour HTTPProxy.
Merge into `live-edit-values.yaml`, with your domain and an existing TLS Secret:

```yaml
httpProxy:
  enabled: true
  fqdn:
    host: grpc.live-edit
    domain: example.com
  tls:
    enabled: true
    secretName: live-edit-tls
```

To have cert-manager issue the certificate instead, set `httpProxy.tls.certManager.enabled: true` and its `issuerRef`.
The ingress must route HTTP/2 to port 50001.
Any ingress or proxy in front of the service must balance per request, not per connection: HTTP/2 carries all of a client's calls on one connection, so connection-level balancing pins each client to one replica.
Contour, Envoy and Istio balance per request; no session affinity is needed.

What to expect from the ingress:

- `JoinSession` and `SubscribeReadLayer` routes have 240-minute timeouts: the ingress ends a subscription at most 240 minutes after it opens, and a join after 240 minutes without traffic.
  Clients reconnect when a stream ends.
- Every other route, including layer uploads, uses `httpProxy.timeoutPolicy` (30 seconds by default).
  Raise it if large uploads over slow links time out at the ingress.
- A client that disappears without closing its connection, such as a laptop that sleeps, leaves its session's participant list about 45 seconds after its last keepalive.
  A client that sends no keepalives stays listed until the ingress ends its stream; if your ingress supports a downstream HTTP/2 keepalive, enable it.
  On an Envoy-based ingress that is `http2_protocol_options.connection_keepalive`; Contour 1.33.1 has no setting for it.

The chart's optional `networkPolicy` (off by default) allows egress to DNS, to pods labelled `app.kubernetes.io/name: valkey` in the release namespace on port 6379, and to ports 80, 443, 8011 and 8080 anywhere.
It does not reach a managed backing store, or Storage REST on its usual port 8012.
Network policies add up, so with it enabled, add a policy for the missing egress; for Azure Managed Redis (port 10000 and shard ports 8500–8599) and Storage REST:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: live-edit-service-extra-egress
  namespace: storage-apis
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/instance: live-edit-service
      app.kubernetes.io/component: server
  policyTypes:
    - Egress
  egress:
    - to:
        - ipBlock:
            cidr: "{CACHE_SUBNET_CIDR}"
      ports:
        - protocol: TCP
          port: 10000
        - protocol: TCP
          port: 8500
          endPort: 8599
    - ports:
        - protocol: TCP
          port: 8012
```

For Amazon MemoryDB, allow port 6379 to the cluster's subnets instead of the Azure ports.
Verify with `kubectl get networkpolicy -n storage-apis`, then confirm the pods become ready.

(live-edit-deployment-install)=
## Install

```bash
# validate the chart and your values
helm lint ./live-edit-service -f live-edit-values.yaml
helm template live-edit-service ./live-edit-service -n storage-apis -f live-edit-values.yaml > /dev/null

# install
helm upgrade --install live-edit-service ./live-edit-service \
  --namespace storage-apis -f live-edit-values.yaml

# wait for every replica to become ready
kubectl rollout status deployment/live-edit-service -n storage-apis --timeout=10m
```

`helm template` fails with a message naming the value when a required setting is missing or two settings conflict.
A replica becomes ready once it can reach the backing store, the backup store when checkpoints are on, and an unexpired service token when the token sidecar is on ({ref}`Health checks <live-edit-monitoring-health>`).
The startup probe allows it up to 285 seconds.

(live-edit-deployment-verify)=
## Verify

Check that the pods are running and ready, and read the startup log:

```bash
kubectl get pods -n storage-apis -l app.kubernetes.io/instance=live-edit-service,app.kubernetes.io/component=server
kubectl logs deployment/live-edit-service -n storage-apis -c live-edit-service --tail=100
```

Every server pod shows all of its containers ready (for example `2/2` with the token sidecar) and no restarts.
Query the health service through a port-forward with [grpc-health-probe](https://github.com/grpc-ecosystem/grpc-health-probe):

```bash
kubectl port-forward -n storage-apis svc/live-edit-service 50001:50001 &
grpc-health-probe -addr=localhost:50001 -service=nvidia.omniverse.live.v1alpha.LiveService
```

Success is `status: SERVING`.
{ref}`Probes <live-edit-operations-probes>` explains the probes and what a replica checks before it reports ready.
If a pod restarts with `live-edit refuses to start with no durability configured`, revisit {ref}`Configure checkpoints <live-edit-deployment-checkpoints>`; for other failures, see {ref}`Troubleshooting <live-edit-troubleshooting>`.

Health checks do not exercise authorization or checkpoints, so also run the smoke tests and a two-client test.

(live-edit-deployment-smoke-tests)=
### Run the smoke tests

The chart's smoke tests create and delete a disposable file and a live session under a test location, exercising sessions, joins and subscriptions; they do not edit layer content.
They call Storage over REST at `authorization.storageServiceEndpoint`, as a client principal that needs write access to the test location.
Create a Secret with that principal's OAuth client credentials:

```bash
kubectl create secret generic live-edit-smoke-auth \
  --from-literal=credentials="{CLIENT_ID}:{CLIENT_SECRET}" \
  --namespace storage-apis
```

Merge into `live-edit-values.yaml`:

```yaml
smokeTest:
  image:
    pullSecrets:
      - name: ngcpull-secret

tests:
  enabled: true
  storageApiResourceBase: "https://assets.example.com/live-edit-tests/"
  auth:
    microsoftUrl:
      value: "https://login.microsoftonline.com/{TENANT_ID}/v2.0/.well-known/openid-configuration"
    credentials:
      valueFrom:
        secretKeyRef:
          name: live-edit-smoke-auth
          key: credentials
    scope:
      value: "{STORAGE_API_APP_ID}/.default"
```

`microsoftUrl` is your identity provider's OpenID discovery document, not its token endpoint.
`{STORAGE_API_APP_ID}` is the application ID of your Storage Service's Microsoft Entra app registration.
With the ingress enabled, the tests connect through its host name, which needs in-cluster DNS and a valid certificate; until then, add `tests.grpcEndpoint: "http://live-edit-service.storage-apis.svc.cluster.local:50001"`.

With `tests.enabled: true` the tests also run after every install and upgrade, and a failing test fails that `helm upgrade` and marks the release failed.
Run them now:

```bash
helm upgrade live-edit-service ./live-edit-service -n storage-apis -f live-edit-values.yaml
helm test live-edit-service -n storage-apis --logs
```

Success is every test reported as passed and `Phase: Succeeded`.

(live-edit-deployment-two-client-test)=
### Test with two clients

Before production use, connect two Live Edit clients to a disposable base asset, join the same session, publish a layer edit from one, and confirm the other receives it.
With Storage-backed checkpoints, also validate a backup and restore as {ref}`Durability <live-edit-durability>` describes.

(live-edit-deployment-register-with-discovery)=
## Register with Discovery

When the Discovery operator runs in your cluster, opt the Live Edit Service into registration.
Merge into `live-edit-values.yaml` and upgrade:

```yaml
service:
  annotations:
    discovery.omniverse.nvidia.com/enabled: "true"
```

The service advertises itself as the `live-service` type with its gRPC API.
Verify what it advertises, through the port-forward from {ref}`Verify <live-edit-deployment-verify>` (this call needs no token):

```bash
grpcurl -plaintext localhost:50001 \
  nvidia.omniverse.discovery.capabilities.v2alpha.CapabilitiesService/ListServices
```

The response names `live-service` and `nvidia.omniverse.live.v1alpha.LiveService`.
With a static Discovery configuration instead, register the externally reachable gRPC endpoint (the ingress host) under the `live-service` type.

(live-edit-deployment-uninstall)=
## Uninstall

```bash
# remove the release
helm uninstall live-edit-service -n storage-apis

# remove smoke-test resources that Helm leaves behind
kubectl delete job,serviceaccount -n storage-apis \
  -l app.kubernetes.io/instance=live-edit-service --ignore-not-found

# remove the secrets you created
kubectl delete secret live-edit-env live-edit-smoke-auth ngcpull-secret \
  -n storage-apis --ignore-not-found

# verify nothing is left
kubectl get all,serviceaccount,secret -n storage-apis -l app.kubernetes.io/instance=live-edit-service
```

If cert-manager issued the ingress certificate, also delete its Secret (`live-edit-tls` in the example).
Delete the namespace only if nothing else runs in it.

Uninstalling removes no data.
Live sessions stay in the backing store and checkpoints stay under `checkpoint.basePath`, and a new installation that uses the same store picks them up again.
To remove the data, delete the backing store you created for this deployment and the objects under `checkpoint.basePath`.

```{warning}
Deleting the backing store and the checkpoints permanently removes every live session and any edits not yet saved to the base assets.
```

(live-edit-deployment-references)=
## References

- [Helm documentation](https://helm.sh/docs/): the `helm` commands used on this page.
- [Contour](https://projectcontour.io/docs/): the ingress controller whose HTTPProxy resource the chart creates.
- [cert-manager](https://cert-manager.io/docs/): automatic TLS certificates for the ingress.
- [grpc-health-probe](https://github.com/grpc-ecosystem/grpc-health-probe): checks the gRPC health service.
- [grpcurl](https://github.com/fullstorydev/grpcurl): a command-line gRPC client for the Discovery check.
- [Kubernetes gRPC probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#define-a-grpc-liveness-probe): the probe type the chart uses.

(live-edit-deployment-see-next)=
## See next

{ref}`Authentication <live-edit-authentication>`: how client requests are authorized, and the service identity that Storage-backed checkpoints need.
