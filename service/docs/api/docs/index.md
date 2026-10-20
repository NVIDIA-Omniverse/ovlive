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

(live-edit-api-overview)=
# Live Edit client API

This guide is for developers who write a client for the Live Edit service.
It states what a client can rely on: the RPCs, the session and layer lifecycle, safe re-sends, stream endings, errors and input limits.
The {ref}`protocol files <live-edit-api-overview-protocol-files>` define every message and field.

```{toctree}
:maxdepth: 1
:caption: Contents:
writing-layers
streams
errors
validation
```

(live-edit-api-overview-services)=
## API services

The service serves two gRPC services, gRPC health checking and server reflection on one HTTP/2 port.
`nvidia.omniverse.live.v1alpha.LiveService` carries all editing, presence and subscription traffic:

| RPC | Type | Purpose | Permission checked |
|---|---|---|---|
| `CreateSession` | Unary | Create a session for a base layer path and a name; returns its `session_id` | Write on the base path |
| `ListSessions` | Unary | List the sessions of a base layer path | Read on the base path |
| `DeleteSession` | Unary | Delete a session and all its layers | Write on the session |
| `JoinSession` | Bidirectional stream | Presence events and peer messages | Read on the session |
| `CreateLayer` | Client stream | Create or replace a layer's content; returns a new `object_id` | Write on the session |
| `UpdateLayer` | Client stream | Apply one delta to a layer; returns its `sequence_num` | Write on the session |
| `DeleteLayer` | Unary | Delete a layer | Write on the session |
| `StatLayer` | Unary | The layer's creator, creation time and last-modified time | Read on the session |
| `SubscribeReadLayer` | Server stream | The layer's content, then every later update | Read on the session |
| `Ping` | Unary | Server time for round-trip and jitter estimation, and the answering replica's features | None |

`nvidia.omniverse.discovery.capabilities.v2alpha.CapabilitiesService` has one RPC, `ListServices`.
It advertises `LiveService` under the discovery type `live-service` and needs no credentials.

(live-edit-api-overview-authentication)=
## Authentication

Send `authorization: Bearer <token>` metadata on every `LiveService` call, `Ping` included.
A missing, non-ASCII, non-`Bearer` or empty header is refused `UNAUTHENTICATED` before the call runs.
The service then checks the token's access to the session's base path, as the table above lists, and answers `UNAUTHENTICATED` or `PERMISSION_DENIED` when access is refused.
Both refusals carry an empty message.

(live-edit-api-overview-lifecycle)=
## Session and layer lifecycle

1. `CreateSession` with a base layer path and a session name returns a `session_id`.
   Names are unique per base path, and `ListSessions` lets other users find the session.
2. `JoinSession` shows the user to the session's other members and relays peer messages.
3. `CreateLayer` uploads a layer's content and returns its `object_id`, the layer's object generation.
4. `UpdateLayer` applies one delta to that object and returns a `sequence_num`, which increases with every committed update.
5. `SubscribeReadLayer` delivers the layer's content, then every later update as it commits.
6. `CreateLayer` on an existing layer name replaces the layer with a new object; object ids are never reused.
   Writes naming the old `object_id` fail `FAILED_PRECONDITION`, and subscribers receive the new object's content.
7. `DeleteLayer` removes a layer, and `DeleteSession` the session with all its layers; their subscriptions end `NOT_FOUND`.

(live-edit-api-overview-finding-layers)=
## Finding layers

The API does not list a session's layers.
Clients agree on layer names themselves: by a naming convention in your application, or by announcing names in {ref}`peer messages <live-edit-api-streams-joining>`.
Peer messages are not stored, so a member that joins later learns names only from the convention or from a member who announces them again.
`StatLayer` answers `NOT_FOUND` for a name that does not exist, and a subscription to it ends `NOT_FOUND`.

(live-edit-api-overview-common-flows)=
## Common flows

- {ref}`Create a session and layer, then update it <live-edit-api-writing-layers-example>`.
- {ref}`Join a session <live-edit-api-streams-example-join>`.
- {ref}`Subscribe and resume after a dropped stream <live-edit-api-streams-example-subscribe>`.

(live-edit-api-overview-deadlines)=
## Deadlines

Every call except `JoinSession` and `SubscribeReadLayer` has a server-side deadline, 30 seconds by default and 2 seconds for `ListSessions`; a deployment can change both.
A shorter `grpc-timeout` on your call applies instead.
A call past its deadline ends `DEADLINE_EXCEEDED`.
`CreateLayer` and `UpdateLayer` are not exempt: the whole upload, including any wait for capacity, must finish in time.
The two streams run until you close them or the service {ref}`ends them <live-edit-api-streams-endings>`.

(live-edit-api-overview-client-requirements)=
## Client requirements

- **Message size.** Each gRPC message must be at most 4 MB.
  A larger request fails `OUT_OF_RANGE`, or `CANCELLED` when it is an upload chunk; send upload content in chunks.
- **Rust clients on `h2`.** A client built on the `h2` crate, through tonic or hyper, needs h2 0.4.19 or later.
  Versions 0.4.16 to 0.4.18 close their own connection after about 100 unread small messages, which ordinary presence traffic can reach.
  Every stream on that connection then fails `RESOURCE_EXHAUSTED`.
- **Read streams promptly.** Even on h2 0.4.19, a client that stops reading under heavy presence fan-out can have its connection closed.
  Read on a thread application work cannot block, or raise the receive window (`initial_connection_window_size` on a tonic `Endpoint`).
- **Rich error details.** Decode `google.rpc.Status` from the `grpc-status-details-bin` trailer to use the {ref}`retry hints <live-edit-api-errors-retry-hints>`; the status code alone stays correct.
- **Feature checks.** `Ping` reports whether the replica that answered supports {ref}`operation ids <live-edit-api-writing-layers-operation-id>` (`operation_ids`) and {ref}`chained writes <live-edit-api-writing-layers-chained>` (`chained_writes`).
  During a rolling upgrade the next call can reach another version, so handle a replica that lacks a feature on any call.

(live-edit-api-overview-protocol-files)=
## Protocol files

| File in the Live Edit source | Defines |
|---|---|
| `proto/nvidia/omniverse/live/v1alpha/live_service.proto` | `LiveService`, its messages and the field-level contract |
| `oodf-ffi/schema/OmniObjectDelta.fbs` | The OODF delta carried in layer content |

Layer content and deltas, in upload chunks and subscription chunks alike, are OMPF containers, all integers little-endian:

| Bytes | Field |
|---|---|
| 4 | Magic `OMPF`, the `u32` `0x46504d4f` |
| 4 | `numParts`, a `u32` |
| 8 × `numParts` | One `u64` offset per part, from the start of the container; no end entry, so the last part runs to the end |
| rest | The parts: part 0 is the OODF delta, and parts 1 to N are the external values, in `extValueIndex` order |

The retry hints use the standard `google.rpc` error messages ({ref}`References <live-edit-api-errors-references>`).

(live-edit-api-overview-see-next)=
## See next

{ref}`Writing layers <live-edit-api-writing-layers>`: uploads, safe re-sends and chained writes.
