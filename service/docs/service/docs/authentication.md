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

(live-edit-authentication)=
# Authentication and service identity

This page covers how Live Edit authorizes and identifies callers, and the token sidecar it uses for its own outbound calls.
Read it with {ref}`Deployment <live-edit-deployment>`; backing-store authentication is on {ref}`Backing store <live-edit-backing-store>`.

(live-edit-authentication-user-token-authorization)=
## Request authorization

Every call needs a bearer token; authentication cannot be turned off.
With `authorization.enabled: true` (the default), the service forwards the caller's token to your Storage Service to check access to the session's base asset: over REST (the default) as `HEAD {authorization.storageServiceEndpoint}/v1alpha/fileobject/by-address/{address}`, over gRPC (`authorization.protocol: grpc`) as `FileObjectService.Stat`.
The Storage Service validates the token and applies your access policies.
Give `authorization.storageServiceEndpoint` its `https://` scheme; the service refuses to start with a bare host name.
Over REST, Storage must answer `HEAD` on that path ({ref}`Verify <live-edit-authentication-verify>`).

A changed permission takes effect within the cache times: 600 seconds for an allow (`authorization.cacheTtlSecs`, never past the token's expiry) and 30 seconds for a denial (`authorization.cacheNegativeTtlSecs`).

```{important}
Read and modifying operations use the same Storage read check.
Any caller who can read the base asset can also modify its live sessions and layers, so Storage read-versus-write permissions cannot give a participant read-only access to Live Edit.
```

```{warning}
Setting `authorization.enabled: false` lets every authenticated caller reach every session, and it requires `userInfo.enabled: true`.
The User Info service resolves identities but does not validate tokens, so in that mode no one checks a token's signature or expiry unless a validating proxy fronts User Info.
Turning off both authorization and User Info is refused at startup.
```

(live-edit-authentication-user-identity)=
## Caller identity

The service shows each caller to other participants by a name that grants no access.
With `userInfo.enabled: false` (the default), it reads the name from the token's JWT claims: `name` or `preferred_username`; an app-only token is shown by its app's name with ` (app)` appended; otherwise `sub` or `oid`.
A token with none of these claims fails the call.

Opaque (non-JWT) tokens need the User Info service.
Without it, such calls fail with `INTERNAL`, and the service logs `opaque bearer token requires the userinfo service; deploy the userinfo service or use JWT-format bearer tokens`.
To use User Info, merge into your values file:

```yaml
userInfo:
  enabled: true
  endpoint: "https://userinfo.example.com"
```

When `endpoint` is empty, the chart uses the in-cluster `http://userinfo-service:8080` (REST) or `http://userinfo-service:50051` (gRPC).
`authorization.protocol` selects REST or gRPC for both Storage and User Info.
With both authorization and User Info on, a caller that User Info does not know is shown by its token's claims instead of failing.

(live-edit-authentication-service-identity)=
## Service identity for outbound calls

The service authenticates its own outbound calls with tokens it reads from files in the pod.
A token-provisioning container that you supply writes those files from a sidecar: any provisioner you already use, such as a workload-identity helper, a Vault agent or a script.
You need it for:

- Storage-backed checkpoints (`checkpoint.adapter: StorageApi`), for the service's calls to the Storage Service.
  The chart refuses to render this adapter until `serviceIdentity.accessTokenFile.enabled` is `true`.
- Token-file authentication to the backing store, such as Azure Managed Redis with Microsoft Entra ID ({ref}`Access-token file <live-edit-backing-store-auth-token-file>`).

(live-edit-authentication-token-file-format)=
### Token file format

The provisioner writes one file per token:

| Default path | Path setting | Token for | Needed |
|---|---|---|---|
| `/etc/secrets/access-token/access_token` | `checkpoint.credentials.accessTokenFile.path` | Your Storage Service; on Microsoft Entra ID, the scope `{STORAGE_API_APP_ID}/.default` | Always, when `serviceIdentity.accessTokenFile.enabled` is `true` |
| `/etc/secrets/access-token/valkey_access_token` | `valkey.credentials.accessTokenFile.path` | The backing store; on Azure Managed Redis, the scope `https://redis.azure.com/.default` | With backing-store token authentication |

`{STORAGE_API_APP_ID}` is the application ID of your Storage Service's Microsoft Entra app registration; your Storage Service administrator provides it.
Each file holds JSON, with `expires_on` the token's real expiry in RFC 3339:

```json
{"access_token": "{TOKEN}", "expires_on": "2030-01-01T00:00:00Z"}
```

The provisioner must:

- write each file atomically (write a temporary file, then rename it);
- keep every file in `/etc/secrets/access-token`, with a distinct name;
- replace each token at least 7 minutes (420 seconds) before it expires.

Always provide the Storage token: while `serviceIdentity.accessTokenFile.enabled` is `true`, a replica stays `NOT_SERVING` until that file holds an unexpired token, even when only the backing store uses tokens.

(live-edit-authentication-configure-the-token-sidecar)=
### Configure the token sidecar

1. Give the provisioner an identity that can obtain both tokens.
   With Azure workload identity, create a managed identity and federate it to the Live Edit ServiceAccount, as in {ref}`Create the managed identity <live-edit-backing-store-azure-identity>`.
   The chart creates the ServiceAccount `live-edit-service` for the release name `live-edit-service`; a release name that does not contain `live-edit-service` gets `{RELEASE_NAME}-live-edit-service`.
   In your Storage Service, grant the identity read, list, write and delete access to the checkpoint location; for the backing store, grant access as {ref}`Azure Managed Redis <live-edit-backing-store-azure-access-policy>` describes.
2. Set the `serviceIdentity` block (and, with Azure workload identity, the `serviceAccount` annotation) as in the {ref}`complete values example <live-edit-deployment-create-the-values-file>`, with your provisioner's `image`, `env` and `args` under `serviceIdentity.accessTokenFile.provisionerSidecar.container`.
   - The chart names the container `live-edit-service-token-provisioner`, replacing any `name` you set, and mounts a shared in-memory volume at `/etc/secrets/access-token` that the provisioner can write.
   - The chart adds the `azure.workload.identity/use: "true"` pod label when the `azure.workload.identity/client-id` annotation is present.
     With a ServiceAccount you created yourself (`serviceAccount.name`), annotate that ServiceAccount and still set the annotation here, so the pod gets the label.
3. Check the render:

   ```bash
   helm template live-edit-service ./live-edit-service -f live-edit-values.yaml \
     | grep -E 'token-provisioner|token-exchange|LIVE_EDIT_SERVICE_TOKEN_FILE|workload.identity/use'
   ```

   The output shows the sidecar container, the `token-exchange` volume, `LIVE_EDIT_SERVICE_TOKEN_FILE` and, with workload identity, the pod label.

The sidecar requests 20m CPU and 32Mi memory by default; override them under `serviceIdentity.accessTokenFile.provisionerSidecar.container.resources`.
With `autoscaling.mode: hpa`, keep the CPU request positive, and the memory request too if you enable the memory target; the chart refuses to render otherwise.
Without a sidecar image, the chart adds no sidecar; mount the token files at the same paths through `volumes` and `volumeMounts`.
Leave `serviceIdentity.clientCredentials.enabled` at `false`: the service does not implement that mode, and turning it on only logs a warning.

(live-edit-authentication-verify)=
## Verify

1. Every replica reports ready; see {ref}`Verify <live-edit-deployment-verify>`.
2. With the sidecar, each replica's `live_edit_service_token_available` metric on port 9090 is `1`:

   ```bash
   kubectl port-forward -n storage-apis deployment/live-edit-service 9090:9090 &
   curl -s localhost:9090/metrics | grep '^live_edit_service_token_available'
   ```

3. An authorized client can create and join a session, and the {ref}`smoke tests <live-edit-deployment-smoke-tests>` pass.

| Symptom | Check | Fix |
|---|---|---|
| Replica not ready; log warns `service token file still absent or unparseable` with `role=storage-api` | The provisioner's log: `kubectl logs {POD} -n storage-apis -c live-edit-service-token-provisioner` | Fix the provisioner's identity or scope, or make it write to `checkpoint.credentials.accessTokenFile.path` |
| Same warning with `role=valkey`, and the pod restarts | The same sidecar log | Fix the backing-store token ({ref}`Access-token file <live-edit-backing-store-auth-token-file>`) |
| Every call returns `INTERNAL`; the log shows `auth backend returned HTTP 405` | Whether Storage serves `HEAD` on `/v1alpha/fileobject/by-address/` | Upgrade or configure Storage to answer `HEAD`, or set `authorization.protocol: grpc` |
| Calls return `UNAUTHENTICATED` at once; the service logs the call with a `rejection` field | `missing_header`: no `authorization` header; `not_bearer`: not a `Bearer` token; `empty_token`: nothing after `Bearer `; `invalid_header_value`: characters other than visible ASCII | Fix the client or a proxy that strips or rewrites the `authorization` header |
| Calls return `PERMISSION_DENIED` | The caller's read access to the base asset in Storage | Grant read access; allow 30 seconds for the cached denial to expire |

(live-edit-authentication-references)=
## References

- [Microsoft Entra Workload ID for AKS](https://learn.microsoft.com/azure/aks/workload-identity-overview): the workload identity the sidecar uses.
- [RFC 3339](https://datatracker.ietf.org/doc/html/rfc3339): the timestamp format of `expires_on`.
- [JSON Web Token (RFC 7519)](https://datatracker.ietf.org/doc/html/rfc7519): the token format the service reads caller names from.

(live-edit-authentication-see-next)=
## See next

{ref}`Configuration <live-edit-configuration>`: every chart setting, its default and when to change it.
