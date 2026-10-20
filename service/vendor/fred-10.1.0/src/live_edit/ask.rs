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

//! Following a cluster `-ASK` redirect to its target node.
//!
//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
//!
//! A node answers `-ASK <slot> <target>` while `<slot>` is migrating away from it and the command's keys are no
//! longer there. The client must send `ASKING` and then the command to `<target>` on one connection: the target serves
//! an importing slot only to a connection that sent `ASKING`, and clears that flag after the next command (Valkey
//! 9.0.3 `networking.c:3260-3262`). The slot's owner in the client's routing table stays the source until the
//! migration ends.
//!
//! Stock `process_ask` (`src/router/commands.rs:251-285`) gets both halves wrong:
//!
//! - it sends `ASKING` to the target (`send_asking_with_policy`, `src/router/utils.rs:441-493`) and then re-writes the
//!   command with only its hash slot set, so `Router::route` (`src/router/mod.rs:176-207`) sends it to the slot's
//!   owner, the source, which answers `-ASK` again, until the command's write attempts run out and it fails `Routing`
//!   "Max attempts reached.". No `-ASK` is ever followed.
//! - it reads the `ASKING` reply synchronously off the target's connection (`src/router/utils.rs:477-488`). With
//!   commands already in flight there, that read takes another caller's reply, and every later reply on the
//!   connection is paired with the wrong command.
//!
//! [`follow`] instead writes `ASKING` and the command back to back on the target's connection, in one router step,
//! and reads nothing: `ASKING` is an ordinary in-flight command whose `+OK` the reply matching consumes in order
//! (`src/router/clustered.rs:291-330`, `ResponseKind::Respond(None)`). The pin to the target lasts for that one write:
//! no field of the command is changed to name the target, so a replay after a dropped connection, or a later `-MOVED`,
//! routes it by slot again. `ASKING` is written with no write attempt left, so a replay of a dropped connection's
//! in-flight commands discards it (`Command::should_finish_with_error`, `src/protocol/command.rs:1771-1773`) rather
//! than sending it to the slot's owner. When the client has no connection to the target (a node that owns no slot yet,
//! as in a scale-out), one is opened for it (`Connections::add_connection`, `src/router/connections.rs:375`), bounded
//! by [`ASK_TARGET_CONNECT_CAP`] because it runs on the router task, and announced on the client's reconnect events
//! as a cluster sync announces a connection it adds; the next cluster sync that does not list the node as a primary
//! closes it (`src/router/clustered.rs:205-218`, `:551-561`), and the commands still in flight on it are retried,
//! routed by slot. Before it connects ([`super::before_inline_connect`]), and before it writes when another connection
//! holds fed frames ([`super::before_write_to`]), every connection holding fed but unflushed frames is flushed, all at
//! once: the `Ask` was queued behind the command ahead of it, which may be only fed, and neither step polls its socket.
//!
//! **A script never follows an `-ASK`** ([`is_script_command`]). The source routes a script by its declared keys only
//! (Valkey 9.0.3 `cluster.c:1263-1273`), so a script that also reads undeclared keys of the same slot is sent to the
//! target as soon as its declared keys have moved. There the script's own client inherits the `ASKING` flag
//! (`script.c:431-433`), and in an importing slot a single-key command is served locally even when its key has not
//! arrived yet (`cluster.c:1281-1287`): an undeclared row still on the source reads as absent, with no error, so the
//! script returns a partial answer or acts on one. A script answered `-ASK` therefore fails as it did before this
//! change — `Routing` "Max attempts reached.", which the caller's classifier already treats as transient — with
//! nothing written and nothing sent to the target. Plain commands, which name every key they touch, still follow.
//!
//! The caller, `process_ask`, keeps everything before the follow: the redirection budget is checked first, so a client
//! configured with no redirections is never redirected, and a command with no write attempt left still fails `Routing`
//! "Max attempts reached." with nothing written (`send_asking_with_policy`'s check, `src/router/utils.rs:468-469`,
//! which returns before any I/O).

use crate::{
  error::{Error, ErrorKind},
  modules::inner::ClientInner,
  protocol::{
    command::{Command, CommandKind},
    types::Server,
  },
  router::{connections::Connections, utils, Router},
  runtime::RefCount,
  types::{config::Blocking, ClientUnblockFlag},
  utils as client_utils,
};
use std::time::Duration;

/// The most an on-demand connection to an `-ASK` target may take, connect, TLS handshake and setup exchange together.
///
/// The connection is opened inline on the client's single router task, so every other command on the client waits
/// while it runs; at the client's own connection timeout (the service's default is 30 s) an unreachable target would
/// stall them all for that long, during exactly the window a node may be failing. 2 s leaves margin for a TLS
/// handshake on a healthy managed node while bounding that stall. A client whose own connection timeout is shorter
/// keeps it ([`ask_target_connect_timeout`]).
pub const ASK_TARGET_CONNECT_CAP: Duration = Duration::from_secs(2);

/// The bound on an on-demand `-ASK` target connection for a client whose connection timeout is `client`: the shorter
/// of the two, and the cap when the client's is zero (fred's "no timeout", `src/utils.rs:291-298`).
pub fn ask_target_connect_timeout(client: Duration) -> Duration {
  if client.is_zero() {
    ASK_TARGET_CONNECT_CAP
  } else {
    client.min(ASK_TARGET_CONNECT_CAP)
  }
}

/// The script commands a client can send, by name: every one runs a script that may read keys it did not declare.
/// fred has kinds for four of them; the `_RO` forms of `EVAL` / `EVALSHA` exist only as custom commands.
const SCRIPT_COMMANDS: [&str; 6] = ["EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO", "FCALL", "FCALL_RO"];

/// Whether `kind` runs a script, so that an `-ASK` to it must not be followed (module docs). Matched by command kind,
/// never by caller: fred's `EVAL`, `EVALSHA`, `FCALL` and `FCALL_RO` kinds (`CommandKind::is_eval`,
/// `src/protocol/command.rs:1520-1525`), and a custom command (`CommandKind::_Custom`, as `CustomCommand::new_static(
/// "EVALSHA", …)` builds) whose name — its first word, any case — is one of [`SCRIPT_COMMANDS`].
pub(crate) fn is_script_command(kind: &CommandKind) -> bool {
  match kind {
    CommandKind::_Custom(custom) => custom
      .cmd
      .split_whitespace()
      .next()
      .is_some_and(|name| SCRIPT_COMMANDS.iter().any(|script| name.eq_ignore_ascii_case(script))),
    other => other.is_eval(),
  }
}

/// The error a failed or cut connect to an `-ASK` target answers its command with; the text names the target either
/// way.
///
/// An allowlist: only a configuration or credential fault keeps its own kind, so the caller's classifier answers it as
/// the fault it is (not transient: retrying cannot fix it). In this build (rustls) those are:
///
/// - `Tls`: an invalid TLS server name (`src/protocol/connection.rs:390`), or a TLS connector that does not match the
///   backend (`:373`, unreachable with one TLS backend compiled in). Not a bad certificate: a certificate or handshake
///   failure reaches fred as an `std::io::Error` (tokio-rustls 0.26.4 `src/common/mod.rs:115`, through the handshake's
///   `?` at `connection.rs:393`), which fred's `From<io::Error>` makes `IO` (`src/error.rs:179-182`), so it is
///   retried here, as fred's own reconnect path retries it;
/// - `Auth` (a `WRONGPASS` or `NOAUTH`), `Config`, `Url`: a credential, configuration or address fault.
///
/// Every other failure answers `IO`, the transient shape a lost connection already has, because retrying is what fixes
/// it. Besides a connect failure, the cap's `Timeout` and a `Canceled`, that includes two setup outcomes a
/// kind-preserving rule would wrongly leave permanent:
///
/// - a node that is joining or rejoining the cluster reports `cluster_state:fail` for a while (Valkey 9.0.3
///   `cluster_legacy.c:6311-6313`, `:6358-6362`, `:6373-6377`), which fred's connection setup
///   (`src/protocol/connection.rs:620-648`, its cluster-state check) answers `Protocol` "Invalid or missing cluster
///   state." (`:615`);
/// - a server error during setup (`-LOADING` while a node replays its data, `-BUSY` behind a long script) arrives as
///   `Unknown`, and once this text wraps it, it no longer starts with the reply token the caller's not-serving check
///   matches.
///
/// Whatever the answer, the failure is broadcast with its own kind (in [`follow`]), so the error metric counts it by
/// what actually happened.
fn ask_connect_error(target: &Server, err: &Error) -> Error {
  // `ErrorKind::Tls` exists only in a build with a TLS feature.
  #[cfg(any(
    feature = "enable-native-tls",
    feature = "enable-rustls",
    feature = "enable-rustls-ring"
  ))]
  let tls = matches!(err.kind(), ErrorKind::Tls);
  #[cfg(not(any(
    feature = "enable-native-tls",
    feature = "enable-rustls",
    feature = "enable-rustls-ring"
  )))]
  let tls = false;
  let kind = if tls || matches!(err.kind(), ErrorKind::Auth | ErrorKind::Config | ErrorKind::Url) {
    err.kind().clone()
  } else {
    ErrorKind::IO
  };
  Error::new(kind, format!("Failed connecting to ASK target {target}: {err}"))
}

/// Send `ASKING` and then `command` to the `-ASK` target `server` for `slot`, on one connection, reading nothing.
///
/// The caller has checked the command's redirection budget and that it has a write attempt left. An error ends the
/// router task, as stock `write_command`'s does (`src/router/commands.rs:196-204`).
pub(crate) async fn follow(
  inner: &RefCount<ClientInner>,
  router: &mut Router,
  server: Server,
  slot: u16,
  mut command: Command,
) -> Result<(), Error> {
  // A script is never followed (module docs): it fails as before this change, with nothing written or sent. Deleting
  // this `if` block is the whole revert of that rule.
  if is_script_command(&command.kind) {
    command.respond_to_caller(Err(Error::new(ErrorKind::Routing, "Max attempts reached.")));
    return Ok(());
  }

  // A command its caller has given up on writes nothing, not even its `ASKING`.
  if client_utils::read_bool_atomic(&command.timed_out) {
    _debug!(
      inner,
      "Ignore following ASK for timed out command: {}",
      command.kind.to_str_debug()
    );
    return Ok(());
  }

  if router.connections.get_connection_mut(&server).is_none() {
    let target = target_server(inner, router, server.clone(), slot);
    _debug!(inner, "Connecting to ASK target {} for slot {}", target, slot);
    // Send what any connection holds fed but unflushed first (through the bounded flush, all connections at once, so
    // at most one bound however many hold fed frames). The connect holds the router task and polls no connection, and
    // while this `Ask` was queued it counted as a command to follow, so the command written just ahead of it may be
    // only fed; that frame would otherwise wait out the connect. The same entry point as before every reconnect and
    // cluster-sync attempt (the liveness module's *Never waiting on unsent frames*); its clustered-client gate never
    // skips it here, because `follow` runs only on a clustered client.
    super::before_inline_connect(inner, router).await;
    // The whole step — connect, TLS handshake and the connection's setup exchange — runs under one bound
    // ([`ASK_TARGET_CONNECT_CAP`]). `add_connection` inserts the connection into the map only after both succeed
    // (`src/router/connections.rs:381-383`), so a step cut short by the bound leaves nothing in the map; dropping the
    // future drops the half-made socket.
    let connected = client_utils::timeout(
      router.connections.add_connection(inner, &target),
      ask_target_connect_timeout(inner.connection_timeout()),
    )
    .await;
    if let Err(err) = connected {
      _debug!(inner, "Failed connecting to ASK target {}: {:?}", target, err);
      command.respond_to_caller(Err(ask_connect_error(&target, &err)));
      // On the client's error events, as the router reports a connection it lost (`src/router/responses.rs:311-319`),
      // with the failure's own kind.
      inner.notifications.broadcast_error(err, Some(target));
      return Ok(());
    }
    // As a cluster sync does for a connection it adds (`src/router/clustered.rs:574`, `src/router/mod.rs:322`).
    inner.notifications.broadcast_reconnect(target);
    inner.backchannel.update_connection_ids(&router.connections);
  }
  // Send what any other connection holds fed but unflushed before this step's writes, all connections at once: the
  // `ASKING` and the command can each wait on the target's socket for up to a bound, and a frame fed elsewhere would
  // otherwise wait through both (the liveness module's *Never waiting on unsent frames*, before a write).
  super::before_write_to(inner, router, &server).await;
  let conn = match router.connections.get_connection_mut(&server) {
    Some(conn) => conn,
    None => {
      command.respond_to_caller(Err(Error::new(
        ErrorKind::IO,
        format!("No connection to ASK target {server}."),
      )));
      return Ok(());
    },
  };

  // `ASKING` takes no write attempt of the command's, and keeps none of its own: a replay of this connection's
  // in-flight commands after it drops discards it instead of sending it to the slot's owner.
  let mut asking = Command::new_asking(slot);
  let _ = asking.decr_check_attempted();
  let asking_failed = utils::write_command(inner, conn, asking, false).await.err();
  if let Some((err, _)) = asking_failed {
    _debug!(inner, "Error sending ASKING to {}: {:?}", server, err);
    // The command was not written: it is retried or failed like any command whose write did not happen.
    if command.attempts_remaining == 0 {
      command.respond_to_caller(Err(err.clone()));
    } else {
      router.retry_command(command);
    }
    utils::drop_connection(inner, router, &server, &err).await;
    return utils::defer_reconnection(inner, router, None, err, false);
  }

  // The command's own write attempt, as `write_command` takes it (`src/router/commands.rs:111`); the caller checked
  // that one is left.
  if let Err(err) = command.decr_check_attempted() {
    command.respond_to_caller(Err(err));
    return Ok(());
  }
  let is_blocking = command.blocks_connection();
  // `write_command` skips a command whose timeout landed since the check above, leaving the `ASKING` for the next
  // command on this connection (the record's change 8 lists what that next command can be).
  match utils::write_command(inner, conn, command, false).await {
    Ok(flushed) => {
      _trace!(inner, "Sent command to ASK target {}. Flushed: {}", conn.server, flushed);
      if is_blocking {
        inner.backchannel.set_blocked(&conn.server);
      }
      if !flushed && inner.counters.read_cmd_buffer_len() == 0 {
        let _ = conn.flush().await;
      }
      let should_interrupt =
        is_blocking && inner.counters.read_cmd_buffer_len() > 0 && inner.config.blocking == Blocking::Interrupt;
      if should_interrupt {
        let _ = conn.flush().await;
        _debug!(inner, "Interrupt after write.");
        if let Err(e) = client_utils::interrupt_blocked_connection(inner, ClientUnblockFlag::Error).await {
          _warn!(inner, "Failed to unblock connection: {:?}", e);
        }
      }
      Ok(())
    },
    Err((err, _)) => {
      // The command is in the connection's in-flight queue, behind its `ASKING`: dropping the connection retries or
      // fails it there, as for any lost connection.
      _debug!(inner, "Error sending command to ASK target {}: {:?}", server, err);
      utils::drop_connection(inner, router, &server, &err).await;
      utils::defer_reconnection(inner, router, None, err, false)
    },
  }
}

/// The `Server` to open a connection to for an `-ASK` target the client has no connection to.
///
/// A primary in the routing table is taken from there, which carries the TLS server name the cluster sync gave it. A
/// node the table does not list (it owns no slot yet) gets the client's TLS host mapping applied as a sync would apply
/// it, with the TLS server name of `slot`'s current owner as the default host; without one, the parsed `server` is used
/// as it is.
fn target_server(inner: &RefCount<ClientInner>, router: &Router, server: Server, slot: u16) -> Server {
  let cache = match router.connections {
    Connections::Clustered { ref cache, .. } => cache,
    _ => return server,
  };
  if let Some(known) = cache.slots().iter().map(|range| &range.primary).find(|primary| **primary == server) {
    return known.clone();
  }

  #[cfg(any(
    feature = "enable-rustls",
    feature = "enable-native-tls",
    feature = "enable-rustls-ring"
  ))]
  {
    let mut server = server;
    if let (Some(tls), Some(owner_name)) = (
      inner.config.tls.as_ref(),
      cache.get_server(slot).and_then(|owner| owner.tls_server_name.as_ref()),
    ) {
      server.set_tls_server_name(&tls.hostnames, owner_name);
    }
    server
  }
  #[cfg(not(any(
    feature = "enable-rustls",
    feature = "enable-native-tls",
    feature = "enable-rustls-ring"
  )))]
  {
    let _ = (inner, slot);
    server
  }
}
