// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Calls their callers gave up on: a command, pipeline or transaction handed to fred is never written after its
//! caller gave up on it.
//!
//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
//!
//! # Stock fred
//!
//! Every command carries a `timed_out` flag (`Command::timed_out`, `src/protocol/command.rs:1556`), and the router
//! refuses to write a command whose flag is set: at the write (`src/router/utils.rs:71-78`, which every path that
//! writes a command or a pipeline's command goes through — the command channel, the retry buffer's replay after a
//! reconnect or a cluster sync, a `-MOVED` retry), at the replay (`src/router/mod.rs:374`) and before following an
//! `-ASK` ([`super::ask::follow`]). But stock fred sets the flag in one place only, the request-response path
//! (`src/utils.rs:402-421`), and only when the call ends with an error. So:
//!
//! - a call whose future is dropped — an abort, a deadline, a cancelled client, a losing `select!` arm — never sets
//!   it, and its command is written whenever the router next reaches it, for example once a reconnect that runs
//!   inline under an unlimited reconnect policy finally succeeds;
//! - a pipeline (`src/clients/pipeline.rs:263-320`) and a transaction (`src/clients/transaction.rs:278-327`) never set
//!   it at all, even when their own timeout fires: `Command::duplicate` gives each queued command a fresh flag
//!   (`src/protocol/command.rs:1855`) that nothing ever sets;
//! - the transaction router (`src/router/transactions.rs`) writes its frames with `Connection::write` directly and
//!   never reads the flag, so a transaction is written whatever its flags say, and fred's own whole-transaction
//!   replay after a reconnect (`retry!`, `:122-133`) writes it again.
//!
//! # The rule
//!
//! Once a call's caller has given up — its timeout fired, fred answered it with an error, or its future was dropped — no
//! frame of that call that fred has not yet written is written. The rule is *never written after*, not *never
//! executed*: a frame already handed on before the caller gave up is not recalled (see *The residual*).
//!
//! # One flag per call
//!
//! Each fred entry point arms a [`Guard`] on one flag before it hands the work to the router: a single command's own
//! flag (`basic_request_response`, `src/utils.rs:402`, and its tracing twin, `:459`, which this build does not compile
//! but which is hooked the same way so the behaviour does not depend on features), or, for a batch, one fresh flag
//! that [`Guard::share`] assigns to every command of the pipeline (`try_send_all`, `send_all` and `send_last`,
//! `src/clients/pipeline.rs:268`, `:295`, `:313`) or of the transaction (`exec`, `src/clients/transaction.rs:321`),
//! in place of the flags `Command::duplicate` gave them. [`Guard::watch`] wraps the call's reply receiver:
//!
//! - a reply that resolves `Ok` disarms the guard, so a completed call never touches the flag. That keeps
//!   `Pipeline::last` as it was for a pipeline spread over several nodes, whose earlier commands may still be in
//!   flight when the last one answers (the service never calls `last`). A server's error reply to a command reaches
//!   the receiver as an `Ok` frame too (`src/protocol/responders.rs:442`) and disarms the guard, which is harmless:
//!   the call is complete;
//! - any other outcome — fred's own timeout, an `Err` fred itself sends on the receiver (a failed write, a lost
//!   connection, a pipeline answered early with its first error, a transaction's error), an errored receiver — sets
//!   the flag, as does dropping the guard at any point before an `Ok`, which covers a dropped future, an abort and an
//!   unwinding panic.
//!
//! Because a batch shares one flag, setting it covers every frame of the batch at once, and the router's existing
//! checks need no change: a pipeline's command still in fred's channel, behind a reconnect, in the retry buffer or
//! redirected is skipped where a timed-out single command already was.
//!
//! # Pipelines
//!
//! A pipeline is written in one router step (`process_pipeline`, `src/router/commands.rs:230-248`), one command at a
//! time, and the flag can be set partway through. The commands after that point are skipped at the write, which can
//! leave a `MULTI` written and its `EXEC` not: an open transaction on that connection, which would turn the next
//! commands sent there — the service's checkout hygiene probe on a `txn` connection, a `WatchGuard`'s `UNWATCH` —
//! into `QUEUED` replies. So [`PipelineStep`] records the last `MULTI` the step hands to the write with its flag still
//! clear — by kind, `CommandKind::Multi` or a custom command whose first word is `MULTI` (as the service's
//! `WatchedTxn` sends it) — and the connection it routes to, and [`after_pipeline`], at the end of the step, writes
//! one `DISCARD` there when the flag is then set. Nothing needs to know whether the `EXEC` was written: when it was,
//! the `DISCARD` answers `-ERR DISCARD without MULTI` and changes nothing.
//!
//! The `DISCARD` is an ordinary tracked command: written through the router's own write (`src/router/utils.rs:65`)
//! on the `MULTI`'s connection, in the same router step, so no other caller's frame comes between, its reply matched
//! in order and dropped (`ResponseKind::Respond(None)`), and nothing is read synchronously: a synchronous read on a
//! connection with commands in flight takes another caller's reply ([`super::ask`]'s module docs). It is written with
//! no write attempt left, as [`super::ask`] writes `ASKING`, so a replay of the
//! connection's in-flight commands after it drops discards it instead of sending it to a fresh connection, where it
//! would have no `MULTI` to discard. The step opens no connection, so the connection found under the `MULTI`'s server
//! is the one the `MULTI` was written to; when a write error dropped it meanwhile there is none, nothing is written,
//! and the store discarded the `MULTI` with its connection. A `MULTI` routed to a replica (`use_replica`, with the
//! `replicas` feature, which the service does not build) is not recorded, because routing it twice would advance the
//! replica round-robin.
//!
//! The `DISCARD` covers a cut during the router step that writes the pipeline, and only that. A pipeline whose commands
//! have a write attempt left and whose connection drops is replayed from the retry buffer one command at a time
//! (`Router::retry_buffer`, `src/router/mod.rs:366-397`), with no [`PipelineStep`]: a flag set partway through that
//! replay skips the rest but leaves a replayed `MULTI` open. The service sends every pipeline holding a `MULTI` on its
//! `txn` pool, at one write attempt, so it never replays one.
//!
//! # fred `Transaction`s
//!
//! The transaction router checks the shared flag at the top of each attempt, again after the drain that starts it
//! (which can wait up to the unresponsive bound), before the `ASKING` it sends after an `-ASK`, before each command
//! and before `EXEC` (`src/router/transactions.rs:121`, `:145`, `:174`, `:189`, `:196`). Once the flag is set:
//!
//! - when no frame of the attempt has been written, the caller's sender is answered with a `Canceled` error
//!   ([`TRANSACTION_ABANDONED`]) and nothing is written ([`refuse_transaction`]). That covers a transaction queued
//!   behind a reconnect and fred's own whole-transaction replay after one, which starts a new attempt;
//! - otherwise — an `ASKING` or a `MULTI` was written — the connection is closed without a flush or a shutdown and a
//!   reconnection is deferred, as a failed write does ([`close_transaction`]). The store discards an unexecuted `MULTI`
//!   when its connection closes. The close is marked on the connection ([`ConnectionState`]), and `Connection::close`'s
//!   hook ([`super::close`], change 3's) returns at once on a marked connection.
//!
//! Closing, rather than writing `DISCARD`, is deliberate here: transaction frames are written outside the in-flight
//! queue, so a `DISCARD` would need a synchronous read of every pending `+QUEUED` reply first. The attempt drained
//! every connection before it began, so the closed connection holds no other caller's command. The close reaches the
//! client's error events once, as a `Canceled` error ([`CONNECTION_CLOSED`]), and costs one reconnect. That reconnect
//! is deferred to the back of the router's command channel, so a command already queued there is routed before it.
//! On a centralized client, whose one connection was closed, that command finds no connection and fails "Failed to
//! route command." (or is kept for a retry when it has a write attempt left); a cluster client routes it as after
//! any failed write. A routing miss means the router holds no connection at all, so no transaction state can be left
//! for such a command to find; the service reads that miss on its `txn` pool (a checkout probe, an `UNWATCH`) as a
//! connection already gone.
//!
//! # What happens to each frame
//!
//! | When the caller gives up, the frame is | Pipeline (and the service's `WatchedTxn`) | fred `Transaction` |
//! |---|---|---|
//! | still in fred's command channel (e.g. behind a reconnect) | skipped at the write | refused at the attempt's check; nothing written |
//! | not yet written, the router partway through the batch | skipped at the write; `DISCARD` when a `MULTI` was written | not written; the connection closed |
//! | written, awaiting a reply, when the connection drops | not replayed | the next attempt refused at its check |
//! | redirected (`-ASK` / `-MOVED`) | not followed | — |
//! | an `EXEC` already written, only its reply late | **runs: this cannot be stopped** | **runs: this cannot be stopped** |
//!
//! # The residual
//!
//! A frame handed to fred's write buffer or to the socket before the flag was set is not recalled. It leaves at the
//! router's next flush, and each wait on its own connection's I/O there is change 3's bounded write or flush wait
//! (`UnresponsiveConfig::max_timeout`, 45 s in the service): it reaches the socket within one such wait, or its
//! connection is failed. On a clustered client that holds however many of the client's connections stall, inline
//! reconnects, cluster syncs and `-ASK` connects included: before the router waits, connects inline, or makes a write
//! that can reach a socket while another connection holds fed frames, it flushes every connection holding them, all
//! at once, and fred's router-wide flush runs its connections' flushes at once too (change 5, [`super::liveness`]'s
//! *Never waiting on unsent frames*). Its exceptions: the flush before a `CLIENT UNBLOCK`, which only the service's
//! stream readers reach, and they write nothing to the keyspace; fred's stock close, up to 5 s on a peer that stopped
//! reading with its socket still open: once for `drop_connection`'s close after a read error or the end of the stream,
//! and per such peer at a router's end, where `disconnect_all` closes them one after another (a failed cluster-wide
//! command runs `drain_all` first); a `Moved`'s `cluster_cache_update_delay`, which the service leaves at 0; and the
//! `replicas` feature, which it does not build. On a centralized or sentinel client a frame fed just before a reconnect
//! can also wait out that pass's backoff sleep (up to about 200 ms: the service's 100 ms first reconnect delay plus
//! fred's up-to-100 ms jitter) before the pass's close flushes it. Once in the kernel's send queue, its delivery and
//! execution are TCP's and the store's, for a single command as for an `EXEC` (SOURCE.md change 9, *The residual*).
//!
//! # Not guarded
//!
//! Senders that hand the router a command without `basic_request_response`, none of them a keyspace data write the
//! service sends: `QUIT` and `SHUTDOWN` (`src/commands/impls/server.rs:38`, `:69`), `HELLO` / `AUTH` (`:189`, `:212`),
//! the cluster-wide commands — script and function administration (`src/commands/impls/lua.rs:77`, `:94`, `:121`,
//! `:295`, `:330`, `:397`, `:449`), a cluster-wide `FLUSHALL` (`server.rs:133`) and client tracking
//! (`src/commands/impls/tracking.rs:88`, `:113`), which go to every node through a path that never reads the flag —
//! and the `SCAN` family's pages (`src/commands/impls/scan.rs`, `src/types/scan.rs`). The service reaches one of them:
//! a `NOSCRIPT` reload's `SCRIPT LOAD` goes through `Script::load` to `script_load_cluster`
//! (`src/types/scripts.rs:200`), which on a cluster client sends it to every node directly (`lua.rs:77`; on any other
//! client it is an ordinary, guarded `SCRIPT LOAD`). It writes the script cache, not the keyspace, so it stays
//! unguarded.

use crate::{
  error::{Error, ErrorKind},
  modules::inner::ClientInner,
  protocol::{
    command::{Command, CommandKind, ResponseSender, RouterCommand},
    connection::Connection,
    types::Server,
  },
  router::{utils, Router},
  runtime::{AtomicBool, RefCount},
  utils as client_utils,
};
use std::future::Future;

/// The error a fred `Transaction` the router refused is answered with: its caller gave up before any frame of the
/// current attempt was written. The caller has already returned, so no caller sees it.
pub const TRANSACTION_ABANDONED: &str = "Transaction abandoned by its caller: this attempt was not written.";

/// The error a connection closed under an abandoned fred `Transaction` reports, on the client's error events and to
/// the transaction's sender.
pub const CONNECTION_CLOSED: &str = "Connection closed: its transaction was abandoned partway through its write.";

/// Per-connection state, on `ConnectionState::abandon`.
#[derive(Debug, Default)]
pub struct ConnectionState {
  /// Set by [`close_transaction`] just before it closes the connection: `Connection::close` then neither flushes nor
  /// shuts the transport down ([`super::close`]). Never cleared; it is dropped with the connection.
  close_at_once: bool,
}

/// Whether `conn` is being closed under an abandoned transaction, so its close must not flush or shut it down.
pub(crate) fn closes_at_once(conn: &Connection) -> bool {
  conn.live_edit.abandon.close_at_once
}

/// The give-up guard of one call (module docs, *One flag per call*): armed on the flag every command of the call
/// shares, it sets that flag when it is dropped still armed. Only an `Ok` reply disarms it ([`Guard::watch`]).
pub(crate) struct Guard {
  flag: Option<RefCount<AtomicBool>>,
}

impl Guard {
  /// Arm on a single command's own flag.
  pub(crate) fn arm(flag: &RefCount<AtomicBool>) -> Guard {
    Guard {
      flag: Some(flag.clone()),
    }
  }

  /// Give every command of the batch `command` (a pipeline or a transaction) one fresh flag, replacing the flags
  /// `Command::duplicate` gave them, and arm on it. Any other router command has no commands to share it with.
  pub(crate) fn share(command: &mut RouterCommand) -> Guard {
    let flag = RefCount::new(AtomicBool::new(false));
    let commands: &mut [Command] = match command {
      RouterCommand::Pipeline { commands } => commands,
      #[cfg(feature = "transactions")]
      RouterCommand::Transaction { commands, .. } => commands,
      _ => &mut [],
    };
    for command in commands.iter_mut() {
      command.timed_out = flag.clone();
    }
    Guard { flag: Some(flag) }
  }

  /// Await the call's reply receiver `reply`, disarming the guard when it resolves `Ok` with an `Ok` reply — a
  /// server's error reply to a command included, which arrives as an `Ok` frame: the call is complete. Any other
  /// outcome (an `Err` fred itself sent, or an errored receiver), and dropping the returned future before it resolves
  /// (fred's own timeout drops it, as does a caller that gives up), sets the flag.
  pub(crate) async fn watch<F, T, E>(mut self, reply: F) -> Result<Result<T, Error>, E>
  where
    F: Future<Output = Result<Result<T, Error>, E>>,
  {
    let result = reply.await;
    if matches!(result, Ok(Ok(_))) {
      self.flag = None;
    }
    result
  }
}

impl Drop for Guard {
  fn drop(&mut self) {
    if let Some(flag) = self.flag.take() {
      client_utils::set_bool_atomic(&flag, true);
    }
  }
}

/// Whether the caller of `command` has given up on it.
pub(crate) fn command_given_up(command: &Command) -> bool {
  client_utils::read_bool_atomic(&command.timed_out)
}

/// Whether the caller of the batch `commands` has given up on it. Its commands share one flag; any of them is read.
pub(crate) fn given_up(commands: &[Command]) -> bool {
  commands.iter().any(command_given_up)
}

/// Whether `kind` opens a transaction: `MULTI` by kind, or a custom command whose name — its first word, any case —
/// is `MULTI`.
fn is_multi(kind: &CommandKind) -> bool {
  match kind {
    CommandKind::_Custom(custom) => custom
      .cmd
      .split_whitespace()
      .next()
      .is_some_and(|name| name.eq_ignore_ascii_case("MULTI")),
    other => other.is_multi(),
  }
}

/// The `MULTI` a pipeline's router step handed to the write with its caller still waiting.
struct OpenMulti {
  /// The server the `MULTI` was routed to.
  server: Server,
  /// The flag the `MULTI` shares with the rest of its batch.
  flag:   RefCount<AtomicBool>,
}

/// What one pipeline's router step wrote that a cut can leave open (module docs, *Pipelines*).
#[derive(Default)]
pub(crate) struct PipelineStep {
  open: Option<OpenMulti>,
}

impl PipelineStep {
  /// Called just before `command` is handed to the router's write: record it when it is a `MULTI` whose caller is
  /// still waiting, with the server the write will route it to.
  pub(crate) fn before_write(&mut self, router: &mut Router, command: &Command) {
    if !is_multi(&command.kind) || command_given_up(command) {
      return;
    }
    // With the `replicas` feature, routing a replica command advances the replica round-robin, so a second route
    // could name another replica than the write's (module docs).
    #[cfg(feature = "replicas")]
    if command.use_replica {
      return;
    }
    if let Some(conn) = router.route(command) {
      self.open = Some(OpenMulti {
        server: conn.server.clone(),
        flag:   command.timed_out.clone(),
      });
    }
  }
}

/// The end of a pipeline's router step: when the step wrote a `MULTI` and its caller has since given up, write one
/// `DISCARD` on that `MULTI`'s connection (module docs, *Pipelines*). An error ends the router task, as the router's
/// own write does (`src/router/commands.rs:196-204`).
pub(crate) async fn after_pipeline(
  inner: &RefCount<ClientInner>,
  router: &mut Router,
  step: PipelineStep,
) -> Result<(), Error> {
  let open = match step.open {
    Some(open) if client_utils::read_bool_atomic(&open.flag) => open,
    _ => return Ok(()),
  };
  // The `DISCARD` is sent at once and can wait on the `MULTI`'s socket for up to a bound: first send what any other
  // connection holds fed but unflushed, all connections at once, so no frame fed elsewhere waits through it
  // (`super::before_write_to`; the liveness module's *Never waiting on unsent frames*).
  super::before_write_to(inner, router, &open.server).await;
  let conn = match router.get_connection_mut(&open.server) {
    Some(conn) => conn,
    None => {
      _debug!(
        inner,
        "No connection to {} left to discard an abandoned pipeline's MULTI on.",
        open.server
      );
      return Ok(());
    },
  };

  _debug!(inner, "Discarding an abandoned pipeline's MULTI on {}", open.server);
  let mut discard = Command::new(CommandKind::Discard, Vec::new());
  // No write attempt left: a replay of this connection's in-flight commands discards it.
  let _ = discard.decr_check_attempted();
  match utils::write_command(inner, conn, discard, true).await {
    Ok(_) => Ok(()),
    Err((err, _)) => {
      _debug!(inner, "Error sending DISCARD to {}: {:?}", open.server, err);
      utils::drop_connection(inner, router, &open.server, &err).await;
      utils::defer_reconnection(inner, router, None, err, false)
    },
  }
}

/// A fred `Transaction` whose caller gave up before any frame of the current attempt was written: answer its sender
/// and write nothing (module docs, *fred `Transaction`s*).
pub(crate) fn refuse_transaction(inner: &RefCount<ClientInner>, id: u64, tx: ResponseSender) -> Result<(), Error> {
  _debug!(inner, "Ignore writing abandoned transaction {}", id);
  let _ = tx.send(Err(Error::new(ErrorKind::Canceled, TRANSACTION_ABANDONED)));
  Ok(())
}

/// A fred `Transaction` whose caller gave up after a frame of the current attempt was written on the connection to
/// `server`: close that connection without a flush or a shutdown, answer the sender, and defer a reconnection, as a
/// failed write does (module docs, *fred `Transaction`s*). An error ends the router task, as the router's own write
/// does.
pub(crate) async fn close_transaction(
  inner: &RefCount<ClientInner>,
  router: &mut Router,
  server: Server,
  id: u64,
  tx: ResponseSender,
) -> Result<(), Error> {
  _debug!(
    inner,
    "Closing the connection to {} under abandoned transaction {}",
    server,
    id
  );
  if let Some(conn) = router.get_connection_mut(&server) {
    conn.live_edit.abandon.close_at_once = true;
  }
  let error = Error::new(ErrorKind::Canceled, CONNECTION_CLOSED);
  let _ = tx.send(Err(error.clone()));
  utils::drop_connection(inner, router, &server, &error).await;
  utils::defer_reconnection(inner, router, None, error, false)
}
