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

(live-edit-api-streams)=
# Subscriptions and presence

This page covers the two long-lived streams, `SubscribeReadLayer` and `JoinSession`, what each stream ending means, and what a client sees after a restore from backup.
Read the {ref}`overview <live-edit-api-overview>` first.

(live-edit-api-streams-subscribing)=
## Subscribing to a layer

Send `SubscribeReadLayer` with the `object_id` and `sequence_num` you already hold, or `0` and `0` for full content, and your `client_id`.
Each response is a header or a chunk.
A header starts one content update, and the chunks after it carry its OMPF bytes until `total_size` bytes have arrived.

The stream first catches you up:

- With `0`/`0`, or an `object_id` that is no longer current, you receive the current object's full content.
- Behind on the current object, you receive one diff covering every update you missed.
- Already current, you receive no content.

A catch-up's `sectionOrder` is the child's position, from 0, among all of its parent's children, and a diff restates every child of a parent with an added, renamed, moved or reordered child.
A deployment can turn this catch-up behaviour off; its catch-ups then carry only the changed children, each with an order value that is not its position.

Every catch-up then ends with one caught-up header: `total_size` `0`, both timestamps `0`, and no chunks.
Its `object_id` and `sequence_num` are the position you are now caught up to.
After it, you receive one update per commit as it happens.
No update has a `total_size` of `0`: an empty upload is delivered as a small OMPF container holding an empty delta, with its server timestamp set.
So a header with `total_size` `0` is always the caught-up header.
The service sends chunks of at most 1 MiB, so the default 4 MB receive limit is enough.

A busy replica can make the first content, or a capacity refusal, wait up to the subscription's capacity budget, 20 seconds by default, and again after each layer replacement.
Set any first-open timeout above that.
While the service is briefly unable to read updates, a live subscription pauses rather than ending, for up to a few minutes.

When the layer is replaced by `CreateLayer`, the stream sends the new object's full content, under its new `object_id`, followed by another caught-up header.
Discard your copy of the old object.
When the layer or its session is deleted, the stream ends `NOT_FOUND`.

A subscription to a layer that does not exist yet ends `NOT_FOUND` before any content; the service does not wait for the layer to be created.
Subscribe once you know the layer exists ({ref}`Finding layers <live-edit-api-overview-finding-layers>`).

(live-edit-api-streams-position)=
## Tracking your position

Your position on a layer is the `object_id` and `sequence_num` of the last header you received on the stream whose content you received in full, the caught-up header included.
An update cut off mid-chunks does not count.
Read both from every header; never count messages, because sequence numbers can skip.

Live updates written under your subscription's non-empty `client_id` are not sent to you, so the stream skips the numbers of your own writes.
Never advance your position from an `UpdateLayer` response: its reply can arrive before a peer's earlier update reaches you on the stream, and resubscribing from the reply's number would skip that update for good.

To resubscribe, send your position from the stream.
Catch-up content is not filtered by `client_id`, so it also carries your own writes after that position, which your copy already holds, including any write whose reply you lost.

(live-edit-api-streams-example-subscribe)=
## Example: subscribe and resume

Each step names the RPC, the fields that matter and what to check; the {ref}`protocol file <live-edit-api-overview-protocol-files>` has every field.

1. `SubscribeReadLayer` with `session_id`, `layer_name`, `object_id` `0`, `sequence_num` `0` and your `client_id`.
2. Read a header, then chunks until `total_size` bytes have arrived: the layer's full content.
   Build your copy from it.
3. Read the caught-up header, `total_size` `0`.
   Save its `object_id` and `sequence_num` as your position.
4. Each later header and its chunks is one update: apply it, then save the header's position.
   A header with a new `object_id` carries the replacement's full content: rebuild your copy from it.
5. When the stream ends `UNAVAILABLE`, wait at least `retry_delay`, keeping your own backoff, and subscribe again with your saved position.
   An update whose chunks were cut off does not count; the catch-up delivers it again.
   You receive one diff covering what you missed, then a caught-up header, and step 4 continues.
   On `NOT_FOUND`, stop ({ref}`How streams end <live-edit-api-streams-endings>`).

(live-edit-api-streams-joining)=
## Joining a session

Open `JoinSession` with a `JoinSessionHeader` carrying the `session_id`, your `user_application`, and `client_keepalive` (below).
The first response is `SessionConnected`, carrying your `connection_id`.
One `PeerJoined` follows for each member already in the session, and then live `PeerJoined`, `PeerLeft` and `PeerMessage` events.
The service does not send you events about your own connection.

Send a `SessionMessageRequest` to deliver its `data` to the session's other members.
Peer messages are best effort and are not stored:

- A member that joins later does not receive earlier messages.
- A message can be lost; the sender is not told.
- A member can receive the same `PeerMessage` twice: the service can resend one after a backing-store stall, and you may re-send after rejoining.
  Tolerate repeats.

A rejoin gets a new `connection_id`, so the other members see the user leave and join again.

While the deployment {ref}`rebalances its storage <live-edit-api-errors-storage-moves>`, a member already joined keeps its stream.
If the session's data takes longer than about a minute to move, the other members can see a `PeerLeft` and then a `PeerJoined` for that member.
Treat the pair as the same member leaving and returning.

(live-edit-api-streams-keepalives)=
## Keepalives

Behind a proxy, a client that vanishes without closing its stream, such as a sleeping laptop, can stay listed in the session for hours.
Send keepalives so the service removes a vanished client about three intervals after its last data:

1. Set `client_keepalive` in the `JoinSessionHeader`.
2. Read `keepalive_interval_ms` from that join's `SessionConnected`.
   Send keepalives only when it is non-zero, and decide again on every join: an older replica can end the stream `INVALID_ARGUMENT` on a keepalive.
3. Send a `JoinSessionKeepalive` whenever nothing has been sent on the stream for `keepalive_interval_ms` (15 seconds by default).
   Any message you send counts as well.
   Keepalives are never delivered to other members.
4. Send keepalives from the transport or network thread, never queued behind application work, or a busy application can end a live join.

A join that sent nothing for three intervals is removed: the other members receive one `PeerLeft`, and the stream ends `UNAVAILABLE` with the `CLIENT_KEEPALIVE_TIMEOUT` hint.
A join without keepalives is never ended for silence.
`SubscribeReadLayer` carries nothing from the client and has no keepalive.

(live-edit-api-streams-example-join)=
## Example: join a session

1. Open `JoinSession` and send a `header` with `session_id`, `user_application` and `client_keepalive` set.
2. Read `SessionConnected`: keep `connection_id`, and if `keepalive_interval_ms` is non-zero, send a `keepalive` whenever you have sent nothing for that long ({ref}`Keepalives <live-edit-api-streams-keepalives>`).
3. Build the member list from the `PeerJoined` events that follow, one per member already present, then apply live `PeerJoined` and `PeerLeft`.
4. Send a `message` to reach the other members; they receive it as `PeerMessage`.
5. When the stream ends `UNAVAILABLE`, rejoin after at least `retry_delay`, keeping your own backoff, and rebuild the member list from scratch.
   When it ends `OK`, the session was deleted: do not rejoin.

(live-edit-api-streams-endings)=
## How streams end

A stream ending queues behind the messages you have not read yet, so you see it only after reading them.
A client that has disconnected receives no status.
Every other ending the service makes carries a status, except a `JoinSession` whose session was deleted, which ends normally with `OK`.

| Ending | Stream | Meaning | Do |
|---|---|---|---|
| `UNAVAILABLE` `server shutting down`, hint `SERVER_SHUTTING_DOWN` | Both | The replica is shutting down, as in a rolling restart | Reconnect after at least `retry_delay`, a minimum wait; keep your own backoff |
| `UNAVAILABLE`, hint `SUBSCRIPTION_CAPACITY`, `READER_CAPACITY` or `CATCHUP_CAPACITY` | Subscribe | The replica had no capacity; arrives before any content, except after a layer replacement | Retry after at least `retry_delay`; keep your own backoff |
| `UNAVAILABLE` `session is missing from the live store; retry so it can be recovered` | Both | The service lost the session's state | Retry, then follow {ref}`After a restore from backup <live-edit-api-streams-restore>`; without a backup the retry answers `NOT_FOUND` |
| Any other `UNAVAILABLE` | Subscribe | You did not read for 5 seconds while a message waited, your position aged out of the layer's history, or a transient failure | Resubscribe; read promptly |
| Any other `UNAVAILABLE` | Join | You missed presence events, sent no data for three {ref}`keepalive <live-edit-api-streams-keepalives>` intervals, or a transient failure | Rejoin, which brings a fresh member list |
| `NOT_FOUND` | Both | The session or layer was deleted, does not exist yet, or was lost and could not be recovered | Do not retry; subscribe again only once the layer exists |
| `OK` | Join | The session was deleted | Do not rejoin |
| `INTERNAL` | Subscribe | A server fault reading the layer | Do not retry; report it to the operator |

Resubscribe with your {ref}`position from the stream <live-edit-api-streams-position>`: the catch-up delivers everything committed while you were away.
No stream ends `ABORTED`.

Streams are authorized once, when they open.
A token that expires later does not end an open stream, but every new call, a reconnect included, needs a valid token.
Refresh tokens before they expire, so a reconnect is not refused `UNAUTHENTICATED`.

(live-edit-api-streams-restore)=
## After a restore from backup

When a deployment keeps backups, a session the service lost is restored from its latest backup by the first request that names the session's ID.
Your client sees:

- Its open streams end with the `session is missing` `UNAVAILABLE` above, and the retry restores the session.
  The message text is the only sign of this ending.
- `ListSessions` does not return a lost session until a request has named its ID.
  Keep the IDs of the sessions your users work in and reopen them by ID; a new session with the same name is a separate session.
- Each layer keeps its `object_id`, and its sequence numbers continue well above any number issued before the loss.
- Edits made after the backup are lost, including ones you received or had acknowledged.
  A catch-up from your position does not remove content that only those lost edits added, so resubscribe each layer with `0` and `0` and replace your copy.
- Records of committed `operation_id`s can be lost too, so a re-send can apply again, and a chained write can report `last_record` `absent`.

(live-edit-api-streams-see-next)=
## See next

{ref}`Errors and retries <live-edit-api-errors>`: status codes, retry hints and unknown outcomes.
