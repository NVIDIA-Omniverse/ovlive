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

# Upstream issues and pull requests for the fred patch

Ready-to-file text for <https://github.com/aembke/fred.rs>, describing the
changes in `vendor/fred-10.1.0.patch` (see `vendor/fred-10.1.0.SOURCE.md`).
It is written for fred's maintainers and names nothing internal to this
repository. Issues 1 and 7 have a pull request ready below; issues 2 to 6,
8 and 9 describe the defects and the fix the patch makes, for a pull request
each once the maintainers agree on the approach. Once filed, record the links in
the SOURCE.md's Upstream section.

File-and-line references are to the published 10.1.0 crate.

---

## Issue 1

**Title:** router task exits when an ASK redirection cannot be followed

**Body:**

### Summary

In cluster mode, when a command receives `-ASK` and has no write attempts
left, the client's router task exits. The client keeps reporting `Connected`
and its command channel stays open, so every later command on that client
hangs until its timeout fires, and `force_reconnection()` never returns. The
client does not recover until it is rebuilt.

### Versions

fred 10.1.0 (also current `main`), tokio runtime, `redis-cluster://` config.

### Cause

`process_ask` (`src/router/commands.rs:251-285`) calls
`send_asking_with_policy` with the command's `attempts_remaining`. When that is
0, `send_asking_with_policy` returns `Routing` / "Max attempts reached."
(`src/router/utils.rs:468-469`). `process_ask` then answers the caller with the
error and also returns it:

```rust
if let Err(e) = asking_result {
  command.respond_to_caller(Err(e.clone()));
  return Err(e);
}
```

`process_command` returns that error to `read_or_write`, whose `?`
(`:470-471`) ends the loop in `start()` (`:556-566`). The router clears its
retry buffer, disconnects, cancels queued commands and the task exits. The
client state is not changed, and no `on_error` / `on_reconnect` event fires,
so nothing visible to the application marks the client as dead.

`process_moved` handles the same situation (redirect budget spent) by
answering the caller and returning `Ok(())` (`:304-307`), so a spent `MOVED`
does not end the router.

The same exit happens when every `ASKING` write attempt fails on a broken
connection, since `send_asking_with_policy` then also returns an error that
`process_ask` propagates.

### Reproduction

A mock cluster node is enough; no real cluster is needed.

1. Serve one fake node that answers fred's handshake (`PING`, `CLIENT ID`,
   `INFO`, `CLUSTER INFO`, and `CLUSTER SLOTS` mapping every slot to itself),
   and replies `-ASK 1234 <its own address>` to any command whose key contains
   `asktrap`.
2. Build a client with `redis-cluster://<mock>`, a 5 s default command timeout
   and an exponential reconnect policy; connect it and wait for the connection.
3. Send one command at one attempt, for example
   `client.with_options(&Options { max_attempts: Some(1), ..Default::default() })`
   with a `GET asktrap`.
4. The caller gets `Err(Routing: Max attempts reached.)` (correct), but the
   `ConnectHandle` returned by `init()` / `connect()` completes with the same
   error immediately afterwards.
5. `client.state()` still returns `Connected` and `is_connected()` is `true`.
   A following `client.get("other")` fails with `Timeout` after 5 s.
   `force_reconnection()` never returns.

With the default three attempts the router survives one `ASK` (it sends
`ASKING` and retries), but a third consecutive `ASK` on the same command ends
it the same way. In a pool, the pool keeps routing a share of its commands to
the dead client, and a command in flight on it when it dies fails with
`Unknown: channel closed`.

### Expected

An `ASK` the command cannot follow fails that command only. The router keeps
running and serves the next command.

---

## Pull request for issue 1

**Title:** fix(router): fail the command, not the router, when ASKING cannot be sent

**Body:**

When `send_asking_with_policy` fails in `process_ask` (most often because the
command has no write attempts left, "Max attempts reached."), `process_ask`
answered the caller and then also returned the error. `read_or_write`
propagates it with `?`, which ends the router loop in `start()`: the router
task exits while the client still reports `Connected`, and every later command
on that client times out.

This makes `process_ask` behave like `process_moved` does when its redirect
budget is spent: answer the caller with the error and return `Ok(())`. The
caller still receives the same error, so error handling in applications is
unchanged; only the router survives.

Fixes #<issue number>.

```diff
--- a/src/router/commands.rs
+++ b/src/router/commands.rs
@@ -272,8 +272,8 @@ async fn process_ask(
   ))
   .await;
   if let Err(e) = asking_result {
-    command.respond_to_caller(Err(e.clone()));
-    return Err(e);
+    command.respond_to_caller(Err(e));
+    return Ok(());
   }
 
   if let Err(error) = write_command_t!(inner, router, command) {
```

Testing: with the mock node from the issue, after the patch the redirected
command still fails with `Routing: Max attempts reached.`, the next command on
the same client answers in under a millisecond, a command in flight on the
client at the time is answered rather than failed with `channel closed`, and
`force_reconnection()` completes. A test for the repository's cluster suite
could send a one-attempt command to a key in a slot that is mid-migration and
assert that a following `PING` on the same client succeeds.

---

## Issue 2

**Title:** unresponsive check never fires on a connection that keeps receiving commands

**Body:**

### Summary

`UnresponsiveConfig::max_timeout` is meant to close a connection that has
stopped answering. It measures the time since the connection's **newest**
unanswered write, so a silent connection that keeps being handed commands is
never closed, however long its oldest command has waited. In a `Pool`, the
round-robin keeps routing a share of every caller's commands to it, and each
one fails at its command timeout until traffic happens to pause for
`max_timeout`.

### Versions

fred 10.1.0 (also current `main`), any server type.

### Cause

`Connection::write` stamps `last_write` on every non-pubsub write
(`src/protocol/connection.rs:802-803`, from `src/router/utils.rs:98,106`).
`poll_connection` clears it whenever the read loop takes a frame
(`src/router/types.rs:45,49`) and declares the connection
`"Unresponsive connection."` when `now - last_write > max_timeout`
(`:53-62`). Each new write moves the stamp forward, so under steady traffic
the condition is never true.

Two related effects:

- A reply read outside `poll_connection` never clears the stamp: the drain
  before a cluster-wide command (`src/router/mod.rs:401-436`, which the
  cluster credential refresh's `_AuthAllCluster` runs) reads every in-flight
  reply with `Connection::drain`. A quiet node whose last reply was read
  there keeps a stale stamp and is declared unresponsive `max_timeout` later,
  although it is healthy.
- Nothing calls `broadcast_unresponsive` (`src/modules/inner.rs:168`), so
  `on_unresponsive` never fires; the verdict reaches the application only as
  a connection error.

### Reproduction

1. Put a TCP proxy between a client and the server that, on a chosen marker
   key, stops forwarding the server's replies on that connection (keep the
   socket open, send no FIN or RST).
2. Build a client with `max_timeout: Some(5 s)` and a 2 s command timeout,
   send a `GET` of the marker key, then keep sending one `GET` per 500 ms.
3. By the code above, every command fails with `Timeout` after 2 s and the
   connection is never closed while the loop runs; stop the loop and it is
   closed 5 s later. We saw the same with a real silent connection under
   load: each episode of timeouts lasted until the first gap in writes longer
   than `max_timeout`, up to about 15 minutes.

### Expected

A connection whose oldest outstanding command has had no reply for the
configured time is closed, whatever is written to it meanwhile.

### Proposed fix

Judge the stall from the oldest outstanding command instead: the time since
the later of when its frame was written to the socket and the connection's
last read progress (a frame decoded, or the read buffer grew, so a large reply
still arriving is not silence). Clear it on any frame, wherever it is read.
A single fixed threshold then has to choose between slow detection and
closing every connection to a node that is busy with a long script, so the
fix we run adds an opt-in faster rule: close a connection that has stalled for
a short bound only when another connection that shares an opt-in registry got
a reply from the same cluster slot range after the stall began, which proves
the node is answering others. The longer `max_timeout` stays as the rule for a
node that is silent to everyone. Give the faster rule its own error text, so
applications can tell one dead socket from a silent node. We can send this as
a pull request.

---

## Issue 3

**Title:** router waits on one connection's I/O have no deadline

**Body:**

### Summary

Several waits inside the router task have no deadline and do not run the
unresponsive check, so one silent connection stalls every command on the
client for as long as it stays silent.

### Versions

fred 10.1.0 (also current `main`).

### Cause

These waits on one connection's I/O run inside the router task, outside the
read loop's `select!` (`src/router/commands.rs:454-494`) where the
unresponsive scan lives:

- `drain_all` before a cluster-wide command (`src/router/mod.rs:401-436`),
  which the periodic cluster credential refresh runs;
- the cluster-wide command's reply reads (`src/router/clustered.rs:61-77`);
- the `ASKING` reply read;
- a `Transaction`'s `EXEC` and skip reads;
- every socket write and flush;
- the close of a failed connection (`Connection::close`,
  `src/protocol/connection.rs:904-912`): a flush of fred's write buffer and,
  under TLS, a `close_notify` before the shutdown, under the 5 s
  `CONNECTION_CLOSE_TIMEOUT_MS`. After a write or flush that could not
  complete, the bytes still buffered are ones the kernel will not take, so
  that close always waits the full 5 s.

### Expected

Each such wait is bounded. When the bound expires, that connection fails
(and is rebuilt by the reconnect policy) and the router carries on.

### Proposed fix

Bound `Connection`'s write, flush and frame-read wrappers by
`UnresponsiveConfig::max_timeout` when it is set, fail the connection with
the unresponsive error into the caller's existing error path, and mark it so
the read loop closes it too where the caller ignores the error. Once a
connection is marked, every later write, feed, flush or read on it fails at
once with the same error, before any I/O, until the router closes it.
Otherwise the cleanup that callers run on the connection they just failed on
waits again while holding the router: a `Transaction`'s `DISCARD` after its
`EXEC` read expired (bounded only by `internal_command_timeout`), a re-flush
of frames left fed, a drain, an `ASKING` retry. A connection judged dead
(a marked one, or one the read loop's unresponsive check is closing) is
closed without a flush or a transport shutdown: its in-flight commands are
returned as before and dropping it closes the socket. The peer is not
reading or not acknowledging, so a graceful close could only wait, and were
it only slow, the flush would hand it frames that would run after fred had
already re-sent them or reported them failed.
Every other close keeps the stock 5 s bound. With these, one silent
connection holds the router for at most one bound per wait chain, its close
included, or per attempt of a `Transaction`. With `max_timeout: None`
nothing changes.

---

## Issue 4

**Title:** a written command can stay unflushed while the router waits idle, reconnects or waits on another connection

**Body:**

### Summary

The router writes a command without flushing it when the command-queue
counter says another command follows, and relies on that later write to
flush. Some paths leave a fed frame unflushed with no later write coming, so
the command is never sent until another command arrives on the client, and
its caller times out. In cluster mode a fed frame can also wait through a
whole reconnect or cluster sync to another node, and is then sent to its
healthy node long after its caller gave up. And it waits behind the router's
writes and flushes on other connections, one after another, so with several
connections stalled it reaches its healthy node one stall after another
later.

### Versions

fred 10.1.0 (also current `main`).

### Cause

The flush decision reads `ClientInner::counters.cmd_buffer_len`
(`src/router/utils.rs:56`, `src/protocol/connection.rs:242-245`,
`src/router/commands.rs:164`).

- A timed-out command skipped at write (`src/router/utils.rs:71-78`) leads to
  a flush of that command's own connection only, and the router-wide flush
  runs only after a flushed write; a skipped `Reconnect` returns without
  flushing (`src/router/commands.rs:357-364`). Either can leave another
  connection's fed frame behind with the counter correct.
- The counter itself can drift from the channel. Its decrement,
  `fetch_sub(1).saturating_sub(1)` (`src/utils.rs:182-184`), saturates only
  the returned value: the stored counter wraps on a decrement at zero. A
  counter left above the channel's real length holds every write back.
  Commands can also leave the channel without being counted down:
  `drain_command_rx` (`src/router/commands.rs:497-510`) cancels the commands
  still queued when a router ends, and the same channel goes to the next
  router.
- In cluster mode, the router command that leads to a reconnect or a cluster
  sync is itself queued and counted: a `Reconnect` deferred after a failed
  read or write (`src/router/utils.rs:118-149`), a `SyncCluster`, or a
  `-MOVED` (`src/router/commands.rs:288-315`). So the command routed just
  ahead of it is fed, not flushed. `reconnect_with_policy` and
  `sync_cluster_with_policy` (`src/router/utils.rs:383-409`, `:557-586`) then
  run `clustered::sync` on the router task: it keeps every connection without
  reader errors, unflushed, and connects the nodes it adds, after a
  `CLUSTER SLOTS` whose fallback tries the known nodes one after another
  (`src/router/clustered.rs:359-397`, `:518-592`). Nothing polls the fed
  frame's connection meanwhile. The frame is sent only by the retry buffer's
  flush after a successful attempt (`src/router/mod.rs:396`): with an
  unreachable node, after one or more full connection timeouts, to a healthy
  node that runs it, whether or not its caller has given up.
- In cluster mode, while a frame sits fed, the router task also waits on
  other connections' I/O: a command's write to another node and the flush
  after it (`src/router/commands.rs:158`, `:164-166`), and the `ASKING` write.
  The fed frame leaves only at the next flush point, after those waits. With
  each socket wait bounded (Issue 3), k connections that stopped reading with
  their kernel send buffers full hold the frame for up to k bounds; without a
  bound, for as long as the first of them stays stuck. Between router
  commands, the unresponsive check can judge the healthy connection instead,
  since its oldest command has had no reply since it was fed, and close it,
  failing a command its server never received.
- The router-wide flush after a send (`Connections::flush`,
  `src/router/connections.rs:351-356`) runs every connection's flush under
  `try_join_all`, which stops at the first error and drops the flushes it
  has not finished. A connection that fails fast and comes earlier in the
  map's order leaves another connection's fed frame for the next flush
  point. It also allocates on every flushed write.

### Expected

The router never waits for new work, connects a new connection on the router
task, or waits on one connection's I/O while another connection holds frames
it fed but did not flush, except inside a flush that runs every such
connection's flush at once. A fed frame then reaches its socket within one
bounded wait, however many of the client's connections stall.

### Proposed fix

At the top of `read_or_write`, when the command channel is empty, flush every
connection whose feed count or write buffer is not empty: fred's own rule,
"flush when no queued command follows", checked against the channel rather
than the counter. On a clustered client, also flush every such connection,
whatever the channel holds, at the top of every pass of
`reconnect_with_policy` and `sync_cluster_with_policy`, before the pass's
backoff sleep (so the sleep's reads can take the frame's reply) and its
attempt. A centralized or sentinel client needs no such flush: its attempt
closes its one connection first, and the close flushes it.

Also in cluster mode, before a write that can reach a socket, when a
connection other than the write's destination holds fed frames, flush every
connection holding them, the destination included. A write can reach the
socket when fred's own flush rule would send it (`Counters::should_send`, or
`CommandKind::should_flush`), or when it is a feed onto a write buffer at
tokio-util's 8 KiB backpressure boundary, which flushes that buffer first.
A feed below the boundary only encodes into the buffer and cannot block, so
it triggers nothing and batching across connections is kept; when only the
destination holds fed frames, its own write batches them as today. When the
command-queue counter is zero or the command kind always flushes, the write
stays a send, and the flush moves only what the router-wide flush right after
it would have sent. When the destination's feed count is past
`max_feed_count`, the flush also sends the destination's fed frames and
`Connection::flush` resets its feed count, so every fed frame leaves as the
send and router-wide flush would have sent it, but the write itself is then
only fed and leaves at the next flush point. For a feed at the boundary,
which today is followed by no router-wide flush, the flush sends other connections' fed frames earlier than the
next flush point (the next send, or the idle flush) would. The cost
per command is one more `Router::route` and a few atomic loads, plus one
walk of the connections when the write can reach a socket.

Run every flush of several connections (the idle flush, the reconnect
flush, the flush before a write and the router-wide flush) over only the
connections holding fed frames, all at once and each to its own end
(`join_all`, returning the first error). It then holds the router for at
most one bound however many connections stall, a fast error never drops
another's flush, and it allocates only when two or more connections need a
flush. Still waiting on one connection: the flush before a `CLIENT UNBLOCK`
under `Blocking::Interrupt`, and the stock 5 s close after a read error or
in `disconnect_all`, which can wait only on a peer that stopped reading with
its socket still open: up to 5 s once for the close after a read error or
the end of the stream, and up to 5 s per such peer in `disconnect_all`, which
closes them one after another.

At each receive,
set the counter to the channel's length
plus one (the command just received) by compare-and-set, and log the
difference it corrects: a difference of one either way, which a send in
flight as the counter is read produces on its own, at DEBUG; a larger one as
a WARN, rate-limited per client. Make the decrement saturate the stored
value.

---

## Issue 5

**Title:** RESP2 data replies shaped like pubsub messages are dropped on connections that never subscribed

**Body:**

### Summary

On every connection, fred classifies a reply that is shaped like a pubsub
message as pubsub. A data reply of that shape, for example `HMGET` returning
`["message", "ch", "x"]`, `LRANGE` or `SMEMBERS` returning a three-element
array that starts `message`, or a Lua script returning
`{"subscribe", "x", 1}`, is routed to the pubsub path instead of its command.
The command's caller then receives the next frame, and every later caller on
the connection receives the previous caller's reply.

### Versions

fred 10.1.0 (also current `main`), RESP2.

### Cause

`check_pubsub_message` (`src/router/responses.rs:136-211`) and the RESP2 to
RESP3 conversion, which turns such an array into a `Push` frame at any
depth, apply the pubsub shape test to every connection, whether or not it has
sent a subscription command.

### Expected

A connection that has never sent `SUBSCRIBE`, `PSUBSCRIBE`, `SSUBSCRIBE` or
an unsubscribe form matches every reply to its command.

### Proposed fix

Record on each connection whether it has sent a subscription command, and
classify non-`Push` frames as pubsub only on such a connection; convert RESP2
frames on every other connection without the pubsub rewrite. Real `Push`
frames keep the current handling.

---

## Issue 6

**Title:** feature request: replace one member of a `Pool` with a fresh client

**Body:**

### Summary

A `Pool` holds a fixed `Vec<Client>` for its life. An application that finds
one member misbehaving (it keeps failing after its connections are rebuilt)
cannot swap it for a freshly built client: the pool hands out borrowed
`&Client`s (`next`, `last`, and `ClientLike::inner`, which every command goes
through), so a member cannot be replaced safely while a borrow may be live.

### Versions

fred 10.1.0 (also current `main`).

### Proposal

Map each logical member to a cell: the pool's own clients fill the first
cells, and a fixed number of spare cells follow, each filled at most once.
`replace_member(i, fresh)` stores an already-connected client in the next
empty spare cell, points member `i` at it and returns the client it replaced,
for the caller to quit. Every pool operation resolves the member's current
cell. A replaced client stays in its cell, so a `&Client` handed out earlier
stays valid without `unsafe`; the spare-cell count bounds that retention, and
`replace_member` fails cleanly once the spares are used. A `retire` helper
closes a replaced client's command channel once its router has ended, so a
later send through an old borrow fails at once instead of waiting forever.
`Pool::clients()` would return an owned snapshot of the current members, a
signature change.

---

## Issue 7

**Title:** a connection keeps its largest pipeline, write and read buffers for its whole lifetime

**Body:**

### Summary

Every pipelined connection keeps the largest load it has ever carried until
the connection is replaced. One large pipeline, command or reply leaves its
memory allocated on that connection while it sits idle, and on a pool every
member can end up holding its own high-water mark. A process's retained memory
then depends on which connections happened to carry large traffic, not on
configuration, and no setting bounds it.

### Versions

fred 10.1.0 (also current `main`), tokio runtime, centralized, clustered and
sentinel configs.

### Cause

Four holders grow and never shrink:

- `Connection::buffer`, the in-flight `VecDeque<Command>` (64 slots to start,
  `src/protocol/connection.rs:707`). The router writes a whole pipeline before
  it reads any reply (`process_pipeline`, `src/router/commands.rs:230-248`), so
  the queue grows to the pipeline's length; a slot is `size_of::<Command>()`,
  376 bytes in a 64-bit build. A 65,536-command pipeline leaves 24.6 MB of
  queue capacity behind.
- The `Framed` write buffer (8 KiB to start). A frame is encoded whole, so the
  buffer grows to the largest frame written. Flushing advances the `BytesMut`,
  which lowers `capacity()`, but the next `reserve` reclaims the whole
  allocation, so it lives as long as the connection.
- The `Framed` read buffer (8 KiB to start). The codec decodes only complete
  frames, so the buffer grows to the largest frame read; once the decoded frame
  is dropped, the next `reserve` reclaims that allocation too.
- `Router::retry_buffer` (`src/router/mod.rs:61`), which takes a dropped
  connection's in-flight commands on reconnect and keeps its capacity after
  replaying them.

`max_command_buffer_len` bounds the router's command channel and
`max_feed_count` the number of frames between flushes; neither bounds these.

A clustered client holds one such connection per primary, plus a backchannel
whose buffers are never cut back either, so what one client can keep grows
with the cluster's primary count. And the cluster routing table keeps each
`CLUSTER SLOTS` reply alive: the reply is parsed without copying, so every
slot range's node id, host and TLS server name is a view into the read buffer
the reply was read into. The routing state, the connection-map keys added at
that sync, the next backchannel's server, and any `Server` an application
clones from the routing each keep that whole allocation alive after the sync,
however the backchannel's buffer is later cut, and the pins accumulate with
the primary count and the sync history. A sentinel client does the same once
per connection: the primary's host from `SENTINEL GET-MASTER-ADDR-BY-NAME` is
converted to a `Str` without copying, and the primary connection, its codec,
the client's recorded primary and the backchannel's connection-id key keep the
sentinel connection's whole read buffer alive for as long as that connection
lives.

### Reproduction

1. Connect a centralized client with a pool of one.
2. Send one pipeline of 50,000 `SET` commands, or one `SET` with a 16 MiB
   value, or a `GET` of a 16 MiB value, and wait for the replies.
3. Drop every reply and measure the process's live heap (for example with
   jemalloc's `stats.allocated`): it stays above its value before step 2 by
   about the pipeline's queue size or the value's size, for as long as the
   connection lives. Repeating step 2 on other members of a pool adds the same
   amount per member.

### Expected

An option to bound what a connection keeps once its traffic drains, so that
large calls cost memory while they run and not for the connection's lifetime.

---

## Pull request for issue 7

**Title:** feat(connection): opt-in limits on what a connection keeps once drained

**Body:**

Adds an optional per-client setting with three limits: in-flight commands,
bytes written and bytes per frame read. The default, `None`, keeps today's
behaviour. With limits set, each holder is cut back once its traffic drains,
inside the router task that owns the connection, so nothing is locked:

- the in-flight queue, after a reply pops a command: when its capacity is above
  the command limit and at most half that many commands are queued, it shrinks
  to the limit (the half keeps a queue near the limit from reallocating on
  every reply);
- the router's retry buffer, once a replay empties it: it is replaced with an
  unallocated `VecDeque::new()`, its starting state, since it is used only
  during a replay;
- the write buffer, after a flush leaves it empty: when an encode since it was
  last replaced left more than the write limit in it, it is replaced with a
  fresh 8 KiB buffer;
- the read buffer, after a frame is taken and the buffer is empty: when a frame
  decoded since it was last replaced was larger than the read limit, it is
  replaced with a fresh 8 KiB buffer. The check runs wherever a frame is taken
  off a pipelined connection, before the frame's caller is answered: in the
  read loop, after a reply pops its command, and in `Connection::read`, which
  covers the frames read outside the read loop — notably a transaction's EXEC
  reply, which carries every queued command's reply in one frame.

Per-connection limits alone let a clustered client keep them once per primary,
since it holds a connection to each and its traffic reaches all of them. So
each client keeps at most one warm connection at the limits once drained, and
every other drained connection is cut to an idle shape: no queue or write
buffer allocation and a fresh 8 KiB read buffer.

- At the top of each pass of the router loop, the router elects the warm
  connection by newest read progress (a one-connection client's is warm from
  the first pass after it connects; warmth moves only when the warm connection
  is drained and another has
  strictly newer progress), applies the limits to it once drained and cuts
  every other drained connection that carried traffic since its last cut. A
  reply that leaves a non-warm connection drained also cuts it at once, before
  the reply's caller is answered.
- That pass runs about once per command, so its own work allocates nothing:
  it walks the connections twice through an iterator over the router's own
  connection collections, folding the election over their state in the first
  walk (keeping only the current warm connection and the newest read progress)
  and applying each connection's decision in the second. Its only allocations
  are the buffers its cuts and limits install, so a pass on an idle client
  allocates nothing.
- In the same pass, the router takes the backchannel's lock with `try_write`,
  so it never waits on a caller using the backchannel, and when the
  backchannel holds nothing and carried anything since its last cut, replaces
  both its buffers with fresh 8 KiB ones. Cutting after any use, rather than
  above a size, keeps it exact without reasoning about `BytesMut` growth, and
  the backchannel is used rarely.
- The cluster routing table owns its strings: at the end of
  `ClusterRouting::rebuild`, every slot range's node id, host and TLS server
  name (and its replicas') is replaced with an owned copy, so the
  `CLUSTER SLOTS` reply's read-buffer allocation is freed when the sync ends.
- A sentinel-named primary owns its strings: in `discover_primary_node`, the
  primary's host and TLS server name are replaced with owned copies before the
  client connects to it, so the sentinel's reply is freed with its frame.

The election, cut and backchannel decisions and the two string copies are pure
functions, so they are tested without connections; the election is one fold
that the router pass and the slice form the tests call share. Centralized and sentinel
clients have one connection, which is warm from the first router pass after it
connects, so at rest the warm rule leaves them at the limits. A reply popped in
the same router pass as a reconnect, before a pass has marked the new
connection warm (a transaction retried after a reconnect drains on it), can
cut it once, which changes allocations only.

The byte buffers are judged by what they held, which the codec records on
every encode and decode, not by `BytesMut::capacity()`: after `advance` or
`split_to`, `capacity()` reads near zero for a buffer whose allocation is many
megabytes, so a check on it never fires. A decoded frame keeps its own
reference to its bytes, so replacing an empty read buffer never affects a reply
in flight. Command order, attempts, flush points and what is written are
unchanged; only capacity changes. The cost is one reallocation per drain after
traffic above a limit and, on a client with several connections, up to three
small allocations when a cut connection is used again.

Fixes #<issue number>.

Testing: with limits set, the issue's reproduction should leave the live heap
within the limits after each step, and repeating step 2 should not grow it. A
pipeline longer than the command limit on a connection killed mid-flight
(`CLIENT KILL ID`) exercises the retry buffer. A test of the write buffer must
use a value larger than the write limit sent as one command, which a check on
`capacity()` misses. A cluster test of the warm rule reads each connection's
warm and idle flags, read progress, in-flight command count and buffer
capacities through a report
compiled only under a test feature, which can also switch the rule off or mark
every connection warm, so the test shows it tells a broken rule from a working
one. The same feature counts each client's router passes and their time, and
marks the thread while a pass runs, so a counting allocator can show an idle
pass allocates nothing and a cluster test can report the pass's cost per
command.

---

## Issue 8

**Title:** an ASK redirection is never followed, and the ASKING reply is read out of order

**Body:**

### Summary

In cluster mode, a command that gets `-ASK <slot> <target>` is sent back to
the slot's owner instead of to `<target>`, so it receives `-ASK` again until
its write attempts run out and it fails `Routing` / "Max attempts reached.".
Every command on a key that has already moved during a slot migration
(resharding, scale-out) fails this way, however many attempts it has. And the
`ASKING` that precedes it is read synchronously from the target's connection,
which takes the reply of another command in flight there and leaves every
later reply on that connection paired with the wrong command.

### Versions

fred 10.1.0 (also current `main`), tokio runtime, `redis-cluster://` config.

### Cause

`process_ask` (`src/router/commands.rs:251-285`) calls
`send_asking_with_policy` (`src/router/utils.rs:441-493`), which builds an
`ASKING` command with `cluster_node = <target>`, writes it, flushes, and reads
the next frame from that connection with `read_skip_pubsub`
(`:477-488`). Then `process_ask` re-writes the command with only
`hasher = ClusterHash::Custom(slot)` set and `cluster_node` left unset, so
`Router::route` (`src/router/mod.rs:176-207`) sends it to
`cache.get_server(slot)`: the slot's owner in the routing table, which during
a migration is still the source node. The source answers `-ASK` again. The
server clears the `ASKING` flag after the next command on the target's
connection, so the redirect is never completed; each round costs a write
attempt, and the command ends with "Max attempts reached.".

The synchronous read is a second defect. With pipelined commands already in
flight on the target's connection, the frame `send_asking_with_policy` reads
is the reply to the oldest of them, not `+OK`. That reply is discarded, the
`+OK` is later matched to that command, and every subsequent reply on the
connection is delivered to the command before it. Transactions avoid this
only because `transactions::send` drains the connection first
(`src/router/transactions.rs:142`).

A one-node mock cluster that answers `-ASK` with its own address cannot show
either defect: the target is the slot's owner, so the re-sent command reaches
the right node, and the only connection is idle when `ASKING` is read. Two
nodes are needed.

### Reproduction

1. Serve two fake nodes, A and B. Both answer fred's handshake; `CLUSTER
   SLOTS` maps every slot to A. A answers `-ASK <slot> <B's address>` to any
   command whose key contains `asktrap`. B answers `ASKING` with `+OK` and
   serves keys normally, recording the frames it receives in order.
2. Connect a client with `redis-cluster://<A>` and the default three
   attempts. Send `GET asktrap`.
3. The command fails `Routing: Max attempts reached.`. A received the `GET`
   three times; B received `ASKING` twice and never the `GET`.
4. To see the misdelivery, have B hold back its reply to an earlier command
   on a key B owns (map one slot to B), send that command, then send
   `GET asktrap`: the earlier command receives `+OK` and the reply meant for
   it is lost.

### Expected

The command is written to `<target>` immediately after `ASKING`, on the same
connection, and every reply is matched to its own command.

### Proposed fix

In `process_ask`, after the redirection budget check, and only when the
command has a write attempt left (a command with none keeps today's
"Max attempts reached." with nothing written):

- take the target's connection, opening one when the client has none (the
  target may own no slot yet, as in a scale-out; the next cluster sync drops it
  again when the node is not a primary). Before opening it, flush every
  connection holding fed but unflushed frames: the connect runs on the router
  task and polls no connection, and the queued `-ASK` counted as a command to
  follow, so the command written just ahead of it may be only fed (Issue 4);
- write `ASKING` there as an ordinary pipelined command with
  `ResponseKind::Respond(None)` and no write attempt of its own, and the
  command immediately after it, in the same router step, through
  `utils::write_command`; read nothing. The in-order reply matching consumes
  the `+OK`. A replay of the connection's in-flight commands after a
  disconnect then discards the `ASKING` rather than sending it to the slot's
  owner;
- pin the command to the target for this one write only, without setting
  `cluster_node`, so a replay after a dropped connection or a later `MOVED`
  routes it by slot again;
- handle a write failure as `write_command` does: drop the connection, retry
  or fail its in-flight commands, and defer a reconnection.

A client configured with `max_redirections = 0` is never redirected, as
today. Testing: the two-node mock above, asserting that the `GET` is answered
through B, that A receives it once, that B receives `ASKING` immediately
followed by the `GET`, that a one-attempt command still fails "Max attempts
reached." with nothing sent to B, that a command in flight on B receives its
own reply, and that an `ASK` to a third node that owns no slot is followed
through a new connection.

---

## Issue 9

**Title:** pipelines, transactions and dropped calls are written after their caller gave up

**Body:**

### Summary

A command whose caller has stopped waiting can still be written to the
server, at any later time. fred refuses to write a single command whose
client-side timeout fired, but nothing stops a pipeline or a transaction whose
timeout fired, or any call whose future was dropped (a `tokio::time::timeout`
around it, an aborted task, a losing `select!` branch). While the router
reconnects under a reconnect policy with no attempt limit, such work stays
queued and is written once the server is reachable again, possibly long after
the caller returned an error. For a `MULTI … EXEC` that is a transaction the
caller believes failed being committed later; a caller that `WATCH`ed first
has already lost its watch with the old connection, so the late `EXEC` runs
unconditionally.

### Versions

fred 10.1.0 (also current `main`), tokio runtime, any server config.

### Cause

Each command carries a `timed_out` flag, and the router skips a command whose
flag is set: when writing it (`src/router/utils.rs:71-78`), when replaying the
retry buffer after a reconnect (`src/router/mod.rs:374`), and therefore after
a `MOVED`. But only `basic_request_response` (`src/utils.rs:402-421`) ever sets
it, and only when the call returns an error:

- a call whose future is dropped never sets it;
- `Pipeline::all`, `try_all` and `last` (`src/clients/pipeline.rs:263-320`)
  and `Transaction::exec` (`src/clients/transaction.rs:278-327`) never set it,
  even when their own timeout fires, and `Command::duplicate`
  (`src/protocol/command.rs:1853-1879`) gives every queued command a fresh flag
  that nothing else holds;
- the transaction router (`src/router/transactions.rs:100-261`) writes its
  frames with `Connection::write` and never reads the flag, so a transaction is
  written whatever its flags say, including fred's own whole-transaction replay
  after a reconnect (`retry!`, `:122-133`).

A pipeline can also be cut partway: if its frames stop after `MULTI` was
written but before `EXEC`, the `MULTI` stays open on that connection and the
next command sent there is queued into it instead of running.

### Reproduction

1. Put a TCP proxy between the client and the server that can drop every
   connection and refuse new ones. Use a client with a 1.5 s default command
   timeout, a reconnect policy with no attempt limit, and one connection.
2. Drop the connections and refuse reconnects. Send a pipeline (or a
   `Transaction`) of `MULTI`, `SET k v`, `EXEC`. It fails `Timeout` after
   1.5 s.
3. Wait longer than the timeout, then accept reconnects again and send `PING`
   on the same client. Once the `PING` is answered, `GET k` returns `v`: the
   pipeline the caller saw fail was written after the reconnect.
4. The same happens to a single `SET` whose future is dropped by a 100 ms
   `tokio::time::timeout` while the reconnect is refused.

### Expected

Once a call's caller has given up (its timeout fired, the client answered it
with an error, or its future was dropped), no frame of that call that has not been written
yet is written. A frame already handed to the socket cannot be recalled, so
this is "never written after", not "never executed".

### Proposed fix

- Arm a guard on one flag per call before handing the work to the router: the
  command's own flag in `basic_request_response` (and its tracing twin); for a
  pipeline or a transaction, one fresh flag assigned to every queued command's
  `timed_out` in place of the ones `duplicate` created. The guard wraps the
  reply receiver: an `Ok` reply disarms it; any other outcome, and dropping it
  before an `Ok`, sets the flag. Disarming on `Ok` keeps `Pipeline::last`
  unchanged for a pipeline spread over several nodes. The router's existing
  checks then cover every queued, replayed or redirected frame of the call.
- In `process_pipeline`, record the last `MULTI` handed to the write with its
  flag still clear, and the connection it routes to. When the router step ends
  with the flag set, write one `DISCARD` there as an ordinary tracked command
  with no write attempt left, its reply matched in order, with no synchronous
  read. If the `EXEC` had in fact been written, the `DISCARD`
  answers `-ERR DISCARD without MULTI` and changes nothing. A pipeline replayed
  from the retry buffer after a lost connection is written one command at a
  time outside this step, so a cut during that replay is not covered.
- In `transactions::send`, check the shared flag at the top of each attempt,
  after the drain, and before every frame (`ASKING`, each command, `EXEC`).
  When it is set and nothing of the attempt has been written, answer the
  caller's sender with a `Canceled` error and write nothing. When something has
  been written, close that connection without flushing or shutting it down (the
  server discards an unexecuted `MULTI` with its connection) and defer a
  reconnection, as a failed write does. A `DISCARD` would need a synchronous
  read of every pending `+QUEUED` reply first, because transaction frames are
  not in the in-flight queue; the connection was drained before the attempt,
  so closing it fails no other caller's command.

Testing: the proxy above, with a `PING` on the same client as the barrier
after reconnects are accepted again, so nothing depends on sleeps. Cover a
pipeline and a `Transaction` queued behind a refused reconnect, a single
command and a pipeline whose futures are dropped, a pipeline replayed from the
retry buffer, a `Transaction` cut after its `MULTI` was written (no `EXEC`
reaches the server and the connection is replaced), and a pipeline cut after
its `MULTI` (a `DISCARD` follows and the next command on the connection runs).
