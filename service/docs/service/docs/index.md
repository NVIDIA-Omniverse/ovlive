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

(live-edit-service-overview)=
# Live Edit Service

Live Edit lets several users edit the same Omniverse scene layers together in a shared live session.
Clients publish layer edits, subscribe to other participants' edits, and exchange presence and peer messages.
Each session belongs to a base asset served by your Storage Service.

The service exposes the `nvidia.omniverse.live.v1alpha.LiveService` gRPC API; it has no REST editing API.
Clients need Live Edit protocol support: installing the service does not make an arbitrary USD application collaborative.

(live-edit-service-overview-architecture)=
## Architecture

Live Edit runs as a set of identical replicas behind an HTTP/2 ingress.
Any replica serves any request, so clients need no session affinity.
The replicas keep all live state in a Valkey-compatible backing store, which is the authoritative copy of every live session, not an expendable cache.
Optional checkpoints, written to separate storage, let the service rebuild live state if the backing store is lost.
Checkpoints protect live session data only; they do not replace your application's workflow for saving edits into the base asset.

The service listens for gRPC on port 50001 and serves Prometheus metrics on port 9090 ({ref}`Service ports <live-edit-deployment-ports>`).
It depends on these systems:

| System | Required | Used for |
|---|---|---|
| Valkey-compatible backing store | Yes | Live sessions, layers and presence |
| Storage Service | Yes, with authorization on (the default) | Checking each caller's access to the session's base asset; holding checkpoints when you choose Storage-backed checkpoints |
| User Info service | Only for opaque (non-JWT) bearer tokens | Resolving the caller's display name |
| Token-provisioner sidecar | With Storage-backed checkpoints or token-based backing-store authentication | The service's own identity for outbound calls |
| HTTP/2 ingress with TLS (Contour) | For clients outside the cluster | Terminating client TLS |
| Discovery operator | Optional | Advertising the service to clients |
| Prometheus and an OpenTelemetry collector | Optional | Metrics and traces |

(live-edit-service-overview-documentation)=
## Documentation

- {ref}`Deployment <live-edit-deployment>`: prerequisites, chart, secrets, values, install, verification, Discovery registration and uninstall.
- {ref}`Authentication <live-edit-authentication>`: how client requests are authorized and identified, and the service's own identity for outbound calls.
- {ref}`Configuration <live-edit-configuration>`: every chart setting, its default and when to change it.
- {ref}`Backing store <live-edit-backing-store>`: Valkey requirements, choosing a provider, connection authentication and sizing.
  Provider setup: {ref}`self-hosted Valkey <live-edit-backing-store-self-hosted>`, {ref}`Azure Managed Redis <live-edit-backing-store-azure>` and {ref}`Amazon MemoryDB <live-edit-backing-store-aws>`.
- {ref}`Durability <live-edit-durability>`: durability modes, checkpoints and recovery from backing-store loss.
- {ref}`Scaling <live-edit-scaling>`: replicas, sizing, autoscaling and limits.
- {ref}`Monitoring <live-edit-monitoring>`: health, logs, traces, dashboards and alerts.
- {ref}`Metrics reference <live-edit-metrics-reference>`: the metrics to alert on, dashboard and diagnose with.
- {ref}`Operations <live-edit-operations>`: configuration changes, upgrade and rollback.
- {ref}`Troubleshooting <live-edit-troubleshooting>`: startup, readiness, clients and access.
- {ref}`Troubleshooting the backing store, capacity and rollouts <live-edit-troubleshooting-capacity>`: backing-store faults, backups, capacity and rollouts.
- {ref}`Client API <live-edit-api-overview>`: for developers writing a Live Edit client — the RPCs, errors, stream endings and input limits a client relies on.

Use the Helm chart and container image of the same release; these pages describe that release.

```{toctree}
:maxdepth: 1
:caption: Contents:
deployment
authentication
configuration
backing-store
backing-store-self-hosted
backing-store-azure
backing-store-aws
durability
scaling
monitoring
metrics-reference
operations
troubleshooting
troubleshooting-capacity
```

(live-edit-service-overview-see-next)=
## See next

{ref}`Deployment <live-edit-deployment>`: install Live Edit from the Helm chart.
