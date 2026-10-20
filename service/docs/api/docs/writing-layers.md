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

(live-edit-api-writing-layers)=
# Writing layers

This page covers `CreateLayer` and `UpdateLayer`: how to send an upload, when it applies, and how to re-send one safely.
Read the {ref}`overview <live-edit-api-overview>` first.

(live-edit-api-writing-layers-uploads)=
## Sending an upload

Both RPCs are client streams.
Send exactly one header message, then zero or more chunk messages carrying the content bytes, then close your side of the stream.

- **Size.** One upload may total 100 MiB (104,857,600 bytes) of content by default; a deployment can change the limit.
  A larger upload fails `INVALID_ARGUMENT` and applies nothing, as does one with more than one chunk per KiB of the limit.
  Chunks of tens of KB or more never reach the chunk cap.
  The limit is per call, not per layer: to load more, create the layer with part of the content and add the rest in `UpdateLayer` calls under the limit, or split the scene across layers.
- **Entries.** One delta may carry at most 99,999 structural commands, `SetField`s and time samples in total; a larger one is refused `DELTA_TOO_LARGE` whatever its byte size.
  Split it the same way ({ref}`Deltas that are too large <live-edit-api-validation-too-large>`).
- **Time.** The whole upload must reach the service within the call's {ref}`deadline <live-edit-api-overview-deadlines>`, 30 seconds by default.
  Sending 100 MiB in that time takes about 3.5 MB/s sustained.
  An upload still sending when its deadline passes fails `DEADLINE_EXCEEDED` and applies nothing; on a slower link, send smaller uploads the same way.
- **Header first.** The service authorizes the call and checks the header before it reads any content.
  `UpdateLayer` also checks that the layer exists and `object_id` is current, so those failures arrive before the content is read.
- **Waits.** Content under the size limit is never refused for its byte size: a large upload, or one reaching a busy replica, may wait within its deadline.
  When upload capacity stays full, the call fails `UNAVAILABLE` with the `UPLOAD_CAPACITY` {ref}`retry hint <live-edit-api-errors-retry-hints>`, having applied nothing.
  A structural edit (creating, deleting, moving or renaming prims, or a reset) to a very large layer can run past the deadline and fail `DEADLINE_EXCEEDED` with nothing applied.
  This starts from roughly 35,000 placed assets (a prim with a reference and four attributes) when other edits land meanwhile, roughly 70,000 otherwise, and fewer on a slower deployment.
  Retrying the same edit fails the same way: split very large scenes across several layers.

An upload applies only if its stream ends normally, with you closing your side after the last chunk.
If you cancel it, or its stream is reset or its connection lost before you finish, nothing applies and the service answers `CANCELLED`.
A header followed by a normal end is a valid empty upload.
A cancel or connection loss after you finished sending may already have applied: re-send with the same `operation_id`.

(live-edit-api-writing-layers-objects)=
## Objects and client ids

Every `CreateLayer` that commits allocates a new `object_id`, even when it replaces an existing layer, and the new object starts at sequence number 1.
`UpdateLayer` names the object it applies to, and fails `FAILED_PRECONDITION` once the layer is replaced or `NOT_FOUND` once it is deleted.
Take the current `object_id` from `CreateLayer` or `SubscribeReadLayer`, and rebuild your copy of the layer from that object.

`UpdateLayer`'s header timestamp is your estimate of the server's receive time; compare it with the response's server timestamp to smooth clock jitter.

`client_id` identifies a writer:

- A subscription with the same non-empty `client_id` does not receive that writer's updates live.
  Use a different `client_id` on the write and the subscription, or leave one empty, to receive every update.
- Use one `client_id` per writer per object, and never share one, the empty one included, between writers to the same object.
  The service cannot detect sharing, and writers that share one can have an edit applied twice.
- Keep at most one unchained `UpdateLayer` in flight per layer and `client_id`; to pipeline, use {ref}`chained writes <live-edit-api-writing-layers-chained>`.

(live-edit-api-writing-layers-operation-id)=
## Safe re-sends with `operation_id`

A call whose reply was lost, to `DEADLINE_EXCEEDED`, `UNAVAILABLE` or a dropped connection, may have committed.
Without an `operation_id`, re-sending it applies it again: an update lands twice, and a create makes a new object.
A re-sent delta that creates a section is refused `INVALID_ARGUMENT` instead, so you report a failure for an edit that landed.
Without ids, follow {ref}`When a re-send is not safe <live-edit-api-errors-unsafe-resend>`.

Set `operation_id` on the header to make re-sends exactly-once; a random UUIDv4 is recommended.
On `UpdateLayer` it needs a non-empty `client_id`.
The response's `operation_outcome` says what happened:

| `operation_outcome` | Meaning |
|---|---|
| `OPERATION_OUTCOME_APPLIED` | This call committed. |
| `OPERATION_OUTCOME_ALREADY_COMMITTED` | An earlier call with this id committed. This one applied nothing and returns that commit's `sequence_num` and timestamp, or its `object_id`. Treat it as success. |
| `OPERATION_OUTCOME_UNSPECIFIED` | You sent no id, or the answering replica does not support ids, so an `OK` to a re-send may be a second application. |

The same id with different content, or for `CreateLayer` a different `layer_name`, fails `ALREADY_EXISTS` and applies nothing, except in the case rule 1 below describes.
A `CreateLayer` id is remembered in its session for 24 hours from when its content took effect, even if the layer is replaced or deleted meanwhile.

To rely on it:

1. Mint one id per write intent, and never reuse it for another.
   An id identifies one request: never send it for a different layer or different content.
   Sent for two layers at once, a `CreateLayer` that replaces a layer can fail `ALREADY_EXISTS` and still take effect: that layer refuses edits (`FAILED_PRECONDITION`) for up to a day, then takes the refused call's content.
   Deleting that layer, or replacing it with a `CreateLayer` that does not carry that id, first ends the refusals instead.
2. Re-send with the same id, byte-identical chunks, and the same `session_id`, `layer_name`, `object_id` and `client_id`; the timestamp may change.
3. On `UpdateLayer`, re-send before that `client_id`'s next `UpdateLayer` to the same object, because only the latest operation per object and `client_id` is remembered.
4. Accept an `ALREADY_COMMITTED` sequence number even if it is at or below ones you have seen, and do not use its timestamp as a jitter sample.
5. Do not re-send an `UNAVAILABLE` whose message begins `this attempt applied nothing, but an earlier attempt of this call may have committed`.
   It carries no retry hint or reason, so match that message prefix, which does not change.
   Another commit under your `client_id`, usually your own earlier attempt of this call landing late, landed while this call was in flight, or the object was restored from backup.
   The outcome is unknown, and a re-send may apply twice: follow {ref}`When a re-send is not safe <live-edit-api-errors-unsafe-resend>`.

No part of an `UpdateLayer` carrying an `operation_id` applies after the same `client_id`'s next write to that object commits, if that write commits after the service has received the whole call.
So a late retry of an abandoned call cannot overwrite a newer write.
Rules 3 and 5 and this promise do not apply to chained calls.

The records do not survive the loss of the service's backing store followed by a restore from backup: a re-send after one may apply again.

(live-edit-api-writing-layers-create-outcome)=
## Creating and replacing layers

A `CreateLayer` for a `layer_name` that exists replaces the layer with a new object.
Once the service has seen a call's content take effect as the layer, it answers `OK`.
Any error this version of the service answers, other than a plain `UNAVAILABLE` (one without a {ref}`retry hint <live-edit-api-errors-retry-hints>`), `DEADLINE_EXCEEDED` or `CANCELLED`, means the content had not taken effect when the call was answered, except as described below.
An answer from a replica of an earlier version during a rolling upgrade, from a proxy or your own transport, or a connection lost after you finished sending leaves the outcome unknown too ({ref}`When an upload's outcome is unknown <live-edit-api-errors-unknown-outcome>`).

After a plain `UNAVAILABLE`, `DEADLINE_EXCEEDED` or `CANCELLED`, the outcome is unknown:

- The layer may be unchanged, may already hold this call's content, or may take it after the answer, with no further call from you.
- While a replacement is unsettled, `UpdateLayer` calls naming the layer's previous `object_id` can fail `FAILED_PRECONDITION`.

Re-send the same request with the same `operation_id`.
If the earlier call's content took effect less than 24 hours before, the re-send changes nothing and answers that call's `object_id` with `OPERATION_OUTCOME_ALREADY_COMMITTED`.
If it did not take effect, the re-send creates or replaces the layer itself.
Then write to the `object_id` the re-send returns.

A replacing `CreateLayer` that fails `INTERNAL`, or `ALREADY_EXISTS` when its `operation_id` was also sent for another layer at once (rule 1 of {ref}`Safe re-sends <live-edit-api-writing-layers-operation-id>`), may also still take effect later.
Read the layer with `SubscribeReadLayer` before acting on the assumption that it did not.
Until it settles, `UpdateLayer` calls naming the layer's previous `object_id` can fail `FAILED_PRECONDITION`.
After an `INTERNAL`, it may settle only once the service's operator fixes the fault: report the error to the operator rather than resyncing and retrying in a loop.

Always send an `operation_id` on `CreateLayer`.
A replacing `CreateLayer` re-sent without one may replace the layer again, discarding edits made since the first one took effect.

(live-edit-api-writing-layers-example)=
## Example: create a session and layer, then update it

Each step names the RPC, the fields that matter and what to check; the {ref}`protocol file <live-edit-api-overview-protocol-files>` has every field.

1. `CreateSession` with `path` and `name`; keep the returned `session_id`.
   On `ALREADY_EXISTS`, call `ListSessions` for the path and use that session or choose another name.
2. `CreateLayer`: send a header with `session_id`, `layer_name` and a new `operation_id`, then the OMPF content in chunks under 4 MB, then close your side.
   Keep the response's `object_id`.
3. `UpdateLayer`: send a header with `session_id`, `layer_name`, that `object_id`, your writer's `client_id`, a new `operation_id` and `timestamp_microseconds_since_epoch`, then the delta's chunks, then close.
4. Check the response: `OPERATION_OUTCOME_APPLIED` or `OPERATION_OUTCOME_ALREADY_COMMITTED` means the update is in, at `sequence_num`.
5. On an `UNAVAILABLE` with a {ref}`retry hint <live-edit-api-errors-retry-hints>`, nothing was applied: re-send after at least `retry_delay`, keeping your own backoff.
   On `DEADLINE_EXCEEDED` or an `UNAVAILABLE` without a hint, re-send the same header and chunks with the same `operation_id` ({ref}`Safe re-sends <live-edit-api-writing-layers-operation-id>`).
   The exception is an `UNAVAILABLE` whose message begins `this attempt applied nothing, but an earlier attempt of this call may have committed`: a re-send can apply the update twice.
   Do not re-send it; check whether the update landed, as {ref}`When a re-send is not safe <live-edit-api-errors-unsafe-resend>` describes.
   On `FAILED_PRECONDITION` the layer was replaced: take the new `object_id` from a subscription before writing again.

(live-edit-api-writing-layers-chained)=
## Chained writes

To keep several `UpdateLayer` calls to one layer in flight and in order, send each with `chained_header` instead of `header`.
It adds `previous_operation_id`: the `operation_id` of the same `client_id`'s previous write to that object, empty for the first.
A chained call commits only directly after its predecessor, at most once, whichever replica serves it.

A call whose predecessor has not committed waits for it.
It is never refused because its predecessor's call failed, so when a write does not commit, cancel the calls waiting behind it and re-send them.
A call still waiting near its deadline is refused at least half a margin before that deadline; the margin is a tenth of the deadline, at least 50 ms.
Set deadlines long against your round trip, or your own deadline can fire first.
The exact timing floors are in the proto's `ChainedUpdateLayerHeader` comment.

Each of these refusals means this call applied nothing:

| Status | Reason and metadata | Applied? | What to do |
|---|---|---|---|
| `ABORTED` | `PREDECESSOR_NOT_COMMITTED`; metadata `last_record` and `last_operation_id` | No | Recover from the point `last_record` gives (below). |
| `UNAVAILABLE` | `CHAIN_CHECK_UNAVAILABLE`, with `RetryInfo` | No | The service could not read your chain's state; retry after at least `retry_delay`, keeping your own backoff. |
| `INVALID_ARGUMENT` | `INVALID_UPDATE` | No | Fix the request: it is malformed. |
| `INVALID_ARGUMENT` | `DELTA_TOO_LARGE` or `CHILDREN_LIST_TOO_LONG`, in place of `INVALID_UPDATE` | No | Split the delta ({ref}`Deltas that are too large <live-edit-api-validation-too-large>`). |
| `INVALID_ARGUMENT` | None | No | The replica does not support chained writes; re-send until one that does answers. |
| `FAILED_PRECONDITION`, `NOT_FOUND` | None | No | The object was replaced or deleted; resync the layer. |

`last_record` is one of:

- `operation`: every write of your chain through `last_operation_id` committed, and none after it.
- `absent`: if you had committed writes on the object, it was restored from backup; resync the layer.
- `no_operation_id`: your latest commit there carried no `operation_id`.
  Chain from an unchained write that carries one, or use a new `client_id`.

**Recovering.** Re-send the writes after that point, in order and unchanged.
Set `no_wait` on the first re-send, and on any whose predecessor may already have been passed.
A `no_wait` call never waits: it commits, answers `ALREADY_COMMITTED`, or is refused at once naming your chain's latest write.
Once a later write is acknowledged, every earlier one committed, so cancel any re-send of them still waiting.

**Settled calls.** A chained call is settled by `OK`, `INVALID_ARGUMENT`, `FAILED_PRECONDITION`, `NOT_FOUND`, or a refusal carrying an `ErrorInfo` of domain `nvidia.omniverse.live.v1alpha`.
After any other ending it is unsettled and may still commit later, directly after its predecessor.
Re-sending it unchanged, with the same predecessor, is always safe.

**Switching to unchained writes.** An unchained call commits wherever it lands, so send one only when every earlier chained call is settled, or a write can apply twice.
Settle them by re-sending the unsettled writes chained, in order, the first with `no_wait`.
During a mixed-version rollout, re-send until a replica with chained writes answers.

(live-edit-api-writing-layers-see-next)=
## See next

{ref}`Subscriptions and presence <live-edit-api-streams>`: reading layers, joining sessions, and how streams end.
