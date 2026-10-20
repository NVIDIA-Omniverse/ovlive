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

(live-edit-api-errors)=
# Errors and retries

This page lists the status codes the service returns, the retry hints some refusals carry, when an upload's outcome is unknown, and what calls answer while the service's storage is rebalanced.
Stream endings are in {ref}`How streams end <live-edit-api-streams-endings>`, and re-send rules in {ref}`Writing layers <live-edit-api-writing-layers>`.

(live-edit-api-errors-status-codes)=
## Status codes

| Status | Meaning | Applied? | What to do |
|---|---|---|---|
| `INVALID_ARGUMENT` | The request is malformed or over a limit ({ref}`Input validation <live-edit-api-validation>`) | No | Do not retry it unchanged; `DELTA_TOO_LARGE` or `CHILDREN_LIST_TOO_LONG` means the delta is too large ({ref}`Deltas that are too large <live-edit-api-validation-too-large>`) |
| `UNAUTHENTICATED` | The bearer token is missing, malformed or rejected | No | Obtain a valid token |
| `PERMISSION_DENIED` | The caller lacks the permission the RPC needs | No | Do not retry |
| `NOT_FOUND` | The session or layer does not exist, was deleted, or could not be recovered from backup | No | Do not retry; list or resync |
| `ALREADY_EXISTS` | The session name is taken on that base path, or an `operation_id` was already committed with a different request | No, except for one id sent for two layers at once, which can apply later unless the layer is deleted, or replaced by a `CreateLayer` that does not carry that id, first (rule 1 of {ref}`Safe re-sends <live-edit-api-writing-layers-operation-id>`) | Choose another name, or mint a new id for a new intent; for a replacing `CreateLayer`, read the layer before assuming nothing applied |
| `FAILED_PRECONDITION` | The `object_id` is not the layer's current object: the layer was replaced | No | Resync to the current object |
| `ABORTED` | Only a chained `UpdateLayer` whose predecessor has not committed ({ref}`Chained writes <live-edit-api-writing-layers-chained>`) | No | Recover the chain |
| `CANCELLED` | An upload's stream ended before you finished sending, or one of its chunks was over 4 MB or undecodable | No; {ref}`unknown <live-edit-api-errors-unknown-outcome>` if you had finished sending | Re-send, re-chunking first if a chunk was over 4 MB, since the same chunks fail again |
| `DEADLINE_EXCEEDED` | The call ran past its deadline | Unknown for an upload; no if it was still sending | Re-send an upload with the same `operation_id`. A large upload or a structural edit to a very large layer fails again: change the request ({ref}`Sending an upload <live-edit-api-writing-layers-uploads>`) |
| `UNAVAILABLE` | A transient condition: shutdown, capacity, or an unavailable dependency | No with a {ref}`retry hint <live-edit-api-errors-retry-hints>`; otherwise possibly | Retry with your own backoff, waiting at least any `retry_delay`; for uploads, see {ref}`unknown outcomes <live-edit-api-errors-unknown-outcome>` |
| `RESOURCE_EXHAUSTED` | Never returned by the service itself: an HTTP/2 layer closed the connection | Unknown for uploads in flight | See {ref}`Transport errors <live-edit-api-errors-transport>` |
| `OUT_OF_RANGE` | A single gRPC message exceeded 4 MB | No | Send content in smaller chunks |
| `INTERNAL` | A server fault; the service's own `INTERNAL` carries only the message `internal error` | Not guaranteed | Do not retry the request unchanged; report it to the operator. A `CreateLayer` that replaces a layer may still take effect later: read the layer before acting on the assumption that it did not. One case has its own action: every update to one layer failing while other layers accept updates means that layer is used up: replace it ({ref}`Delta content <live-edit-api-validation-delta>`) |

Some answers need care:

- **`CreateSession` after `UNAVAILABLE`.** A retry can answer `ALREADY_EXISTS` because the failed attempt still holds the name.
  Call `ListSessions` for the base path: it returns the session if the attempt completed, and otherwise frees the name for the next create.
- **`DeleteLayer` after `UNAVAILABLE` or `DEADLINE_EXCEEDED`.** The failed attempt may already have deleted the layer, so a retry answering `NOT_FOUND` means it is gone.
  The layer's subscriptions then end `NOT_FOUND` once the service finishes the delete, usually within minutes.
  A delete of any layer can wait behind other deletes and replacements on a busy server and reach the deadline having changed nothing; retry it.
- **`DeleteSession` after `UNAVAILABLE` or `DEADLINE_EXCEEDED`.** The failed attempt may already have deleted some of the session's layers.
  Re-send it: the retry resumes the deletion where it stopped, and `NOT_FOUND` on the re-send means the session is gone.
- **`FAILED_PRECONDITION` while the layer still has its old object.** A `CreateLayer` that replaces a layer and answers a plain `UNAVAILABLE`, `DEADLINE_EXCEEDED` or `CANCELLED` can leave the layer refusing edits, with no `ErrorInfo`, until its replacement settles.
  It settles with no further call, normally within a minute and otherwise within minutes.
  Re-send that `CreateLayer` with the same `operation_id` to learn the outcome, then edit the object it returns ({ref}`Creating and replacing layers <live-edit-api-writing-layers-create-outcome>`).
  A replacing `CreateLayer` answered `INTERNAL` can leave the layer refusing edits the same way, and it may settle only once the service's operator fixes the fault: report the error to the operator rather than resyncing and retrying in a loop.
- **gRPC retry policies** re-issue `UNAVAILABLE` automatically; if you enable one for uploads, set `operation_id` so its re-sends cannot apply twice.

(live-edit-api-errors-retry-hints)=
## Retry hints

When the service refuses a call or ends a stream for a reason worth retrying, and the call provably applied nothing, its `UNAVAILABLE` carries a hint.
The hint is a `google.rpc.Status` in the `grpc-status-details-bin` trailer.
It holds a `google.rpc.ErrorInfo`, with domain `nvidia.omniverse.live.v1alpha`, a `reason` and `metadata`, and a `google.rpc.RetryInfo` with `retry_delay`.

Every row is an `UNAVAILABLE`, and "retry" means after at least `retry_delay`, keeping your own backoff.

| Reason and metadata | When | Applied? | What to do |
|---|---|---|---|
| `SERVER_SHUTTING_DOWN` | The replica is shutting down | No | Reconnect or re-send; another replica answers |
| `SUBSCRIPTION_CAPACITY`, metadata `cause` | A subscribe found the replica at its subscription limit | No | Retry |
| `READER_CAPACITY` | A subscribe could not start its live feed in time on a busy replica | No | Retry |
| `CATCHUP_CAPACITY`, metadata `wait` (sometimes absent) | A subscribe could not start its catch-up in time on a busy replica | No | Retry |
| `UPLOAD_CAPACITY`, metadata `cause` | The replica's upload capacity is full | No | Retry |
| `STORE_WRITE_REFUSED`, metadata `condition` | The service could not store data the call needed, for example because its storage is full | No | Retry; report it to the operator if it persists, since full storage may not clear by itself |
| `SNAPSHOT_FENCE_TIMEOUT` | An `UpdateLayer` waited its limit while the layer was briefly closed to writes | No | Retry |
| `EXT_RELEASE_BUDGET_FULL` | An upload with external values waited its deadline out because the service had no room for more external values yet | No | Retry |
| `CLIENT_KEEPALIVE_TIMEOUT` | A `JoinSession` sent no data for three keepalive intervals | No | Rejoin, and send keepalives from the network thread ({ref}`Keepalives <live-edit-api-streams-keepalives>`) |
| `CHAIN_CHECK_UNAVAILABLE` | A chained `UpdateLayer` could not read its chain's state | No | Retry |
| `UPSTREAM_UNAVAILABLE` | A service an upload depends on was unavailable before its commit | No | Retry |

The other reasons carry no metadata.
Metadata values name the exact limit or condition that was hit; they are diagnostic only: log them, and never branch on them.

Use a hint as follows:

- **`retry_delay` is a minimum wait.** Keep your own backoff and wait the larger of the two, adding no jitter: the service already randomizes `retry_delay`.
  Read it from each response; shutdown and keepalive endings carry under 100 ms, so the first reconnect stays prompt.
- **An unknown `reason` on `UNAVAILABLE` means "retry after that wait".** New reasons may be added.
- **On `CreateLayer` and `UpdateLayer`, a hint means this call applied nothing.**
  A re-send is safe unless an earlier attempt of the same write is still in doubt; then re-send with that attempt's `operation_id`.
  On other calls it means only "retry after that wait".
- **Recognise a hint by its domain and `reason`, never by the message text.**
  The only messages a client matches are the hint-less refusal in rule 5 of {ref}`Safe re-sends <live-edit-api-writing-layers-operation-id>` and the `session is missing` stream ending ({ref}`How streams end <live-edit-api-streams-endings>`).
- **These rules cover `UNAVAILABLE` only.** The two chained-write refusals and the two {ref}`size refusals <live-edit-api-validation-too-large>` carry an `ErrorInfo` on other codes, with no `RetryInfo`, and mean what their own rows state.

(live-edit-api-errors-unknown-outcome)=
## When an upload's outcome is unknown

Every `UNAVAILABLE` the service gives a `CreateLayer` or `UpdateLayer` before its commit starts carries a hint, so a plain one may have applied.
Treat these answers as outcome unknown, and re-send with the same `operation_id`:

- a plain `UNAVAILABLE`, including `UpdateLayer outcome unknown: …` after the service's own retries;
- `DEADLINE_EXCEEDED`, which does not say how far the call got;
- `CANCELLED` or a lost connection after you finished sending;
- an answer from an earlier-version replica during a rolling upgrade, or from a proxy or your own transport, for example an `UNAVAILABLE` or `INTERNAL`.

The exception is the `UNAVAILABLE` that rule 5 of {ref}`Safe re-sends <live-edit-api-writing-layers-operation-id>` describes, which a re-send can apply twice.
A `CreateLayer` answered one of these can still take effect after the answer ({ref}`Creating and replacing layers <live-edit-api-writing-layers-create-outcome>`).
So can a replacing `CreateLayer` answered `INTERNAL`, or `ALREADY_EXISTS` when its `operation_id` was also sent for another layer at once (rule 1 of {ref}`Safe re-sends <live-edit-api-writing-layers-operation-id>`).

(live-edit-api-errors-unsafe-resend)=
## When a re-send is not safe

The service cannot tell you whether an upload with an unknown outcome landed when:

- it carried no `operation_id`, or its replica does not support them (`Ping` reports `operation_ids` false, or an `OK` carries `OPERATION_OUTCOME_UNSPECIFIED`);
- it was refused with the rule-5 `UNAVAILABLE`.

Then:

1. Read the layer's current content: resubscribe from your {ref}`position <live-edit-api-streams-position>`.
   The catch-up is not filtered by `client_id`, so it includes your write if it landed; peers' later edits can overwrite it.
2. Decide in your application whether the edit is there.
3. If it is missing, send it again as a new write, with a new `operation_id`.
   If applying it twice is harmless in your application, you can re-send it without checking.

A `CreateLayer` re-sent without an id replaces the layer again with a new `object_id`, discarding any update made since the first one.

(live-edit-api-errors-storage-moves)=
## While the service's storage is rebalanced

A deployment's operator can rebalance the service's storage, for example when adding storage capacity.
While a layer's or session's data moves, calls that touch it can be refused for the whole move; other layers and sessions carry on.
Hinted refusals follow {ref}`Retry hints <live-edit-api-errors-retry-hints>`, and `CreateSession`, `DeleteSession` and `DeleteLayer` follow {ref}`Status codes <live-edit-api-errors-status-codes>`.
The other refusals, each retried with your own backoff:

| Status | Reason and metadata | Applied? | What to do |
|---|---|---|---|
| `UNAVAILABLE` | none, on `UpdateLayer` or `CreateLayer` | Unknown | Re-send with the same `operation_id`; the layer or update exists once |
| `UNAVAILABLE` | none, on `JoinSession` | No | Join again |

A live subscription to a moving layer pauses until the move ends; a catch-up of it can end `UNAVAILABLE`: resubscribe from your {ref}`position <live-edit-api-streams-position>`.
Presence during a long move is in {ref}`Joining a session <live-edit-api-streams-joining>`.

(live-edit-api-errors-transport)=
## Transport errors

The service's handlers never return `RESOURCE_EXHAUSTED`.
When every stream on a connection fails with it at once, an HTTP/2 layer closed the connection with GOAWAY `ENHANCE_YOUR_CALM`.
On h2 0.4.16 to 0.4.18 that is usually your own library ({ref}`Client requirements <live-edit-api-overview-client-requirements>`).
On a direct connection, the service does the same when about 2,000 small messages you sent sit unread, or after more than 100 empty, non-final DATA frames.
In a Rust client, the `h2::Error` in the status's source chain says which side: `is_library()` for your own library, `is_remote()` for the service or a proxy.

(live-edit-api-errors-references)=
## References

- [gRPC richer error model](https://grpc.io/docs/guides/error/#richer-error-model) — how `grpc-status-details-bin` carries `google.rpc.Status`.
- [`google/rpc/error_details.proto`](https://github.com/googleapis/googleapis/blob/master/google/rpc/error_details.proto) — the `ErrorInfo` and `RetryInfo` messages.

(live-edit-api-errors-see-next)=
## See next

{ref}`Input validation <live-edit-api-validation>`: the field rules and limits behind `INVALID_ARGUMENT`.
