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

//! A `Pool`'s logical members, each re-pointable to a fresh, already-connected client.
//!
//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
//!
//! Stock fred's `Pool` holds a fixed `Vec<Client>`, and its member is the client at index `i` for the pool's life.
//! A client that keeps failing — whatever per-client state does it — can then only be reconnected, never swapped for
//! a new one, because the pool hands out borrowed `&Client`s (`next`, `last`, and `ClientLike::inner`'s
//! `&RefCount<ClientInner>`, which every command interface goes through) with no lifetime the pool could wait for.
//!
//! Here each logical member `i` names a **cell**: cells `0 .. size` are the pool's original clients, unchanged, and
//! `max(size, 4)` **spare cells** follow, each filled at most once. [`replace_member`] stores a fresh client in the
//! next empty spare cell and points member `i` at it; every pool operation (`next`, `next_connected`, `last`,
//! `ClientLike::inner`, `clients`, `member`, `connect`, `connect_pool`, `init`, `wait_for_connect`,
//! `force_reconnection`, `quit`, `update_perf_config`, `active_connections`, `set_resolver`, the heartbeat) resolves
//! the member's current cell.
//!
//! **Why retired clients stay in their cells.** A `&Client` handed out before a replacement must stay valid after it,
//! and cells are never emptied or overwritten, so it does: the retired client remains as an inert husk (its caller ends
//! its router with a `QUIT`, aborts it only as a last resort, then releases what fred would leave behind with
//! [`retire`]) until the last clone of the `Pool` drops. No `unsafe` is needed. The cost is bounded by the spare cells:
//! once they are used up, [`replace_member`] fails with [`BudgetExhausted`] and the member keeps its client; a process
//! restart resets the budget.
//!
//! Clones of a `Pool` share its cells, so every holder sees a replacement at once.

use crate::{
  clients::{Client, Pool},
  modules::inner::ClientInner,
  runtime::RefCount,
};
use std::{
  fmt,
  sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
  },
  time::{Duration, Instant},
};
use tokio::task::AbortHandle;

/// The fewest spare cells a pool gets, whatever its size.
pub const MIN_SPARE_CELLS: usize = 4;

/// A pool's member-to-cell map and its spare cells, on `PoolInner::live_edit`.
pub struct PoolState {
  /// The spare cells, cell `size + j` being `spares[j]`; each is set at most once and never cleared.
  spares:     Box<[OnceLock<Client>]>,
  /// Per logical member, the cell it currently names.
  current:    Box<[AtomicUsize]>,
  /// The next spare cell [`replace_member`] fills.
  next_spare: AtomicUsize,
}

impl PoolState {
  /// The state of a pool of `size` members: member `i` names cell `i`, and `max(size, 4)` spare cells are empty.
  pub(crate) fn new(size: usize) -> Self {
    PoolState {
      spares:     (0 .. size.max(MIN_SPARE_CELLS)).map(|_| OnceLock::new()).collect(),
      current:    (0 .. size).map(AtomicUsize::new).collect(),
      next_spare: AtomicUsize::new(0),
    }
  }
}

impl fmt::Debug for PoolState {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("PoolState")
      .field("spares", &self.spares.len())
      .field("spares_used", &self.next_spare.load(Ordering::Acquire).min(self.spares.len()))
      .finish()
  }
}

/// [`replace_member`] found no empty spare cell: the member keeps its client.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetExhausted;

impl fmt::Display for BudgetExhausted {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("every spare cell of the pool is used")
  }
}

impl std::error::Error for BudgetExhausted {}

/// The client in cell `cell` of a pool whose original clients are `clients`.
fn cell<'a>(clients: &'a [Client], state: &'a PoolState, cell: usize) -> Option<&'a Client> {
  match cell.checked_sub(clients.len()) {
    None => clients.get(cell),
    Some(spare) => state.spares.get(spare).and_then(OnceLock::get),
  }
}

/// Member `i`'s current client (`i` modulo the pool size). A cell is published only after it is filled, so the
/// fallback to the original client is unreachable.
pub(crate) fn member<'a>(clients: &'a [Client], state: &'a PoolState, i: usize) -> &'a Client {
  let i = i % clients.len();
  let named = state.current[i].load(Ordering::Acquire);
  cell(clients, state, named).unwrap_or(&clients[i])
}

/// Every member's current client, in member order.
pub(crate) fn members<'a>(clients: &'a [Client], state: &'a PoolState) -> impl Iterator<Item = &'a Client> + 'a {
  (0 .. clients.len()).map(move |i| member(clients, state, i))
}

/// Point member `i` of `pool` at `fresh`, an already-connected client, and return the client it named before (to be
/// retired by the caller: a `QUIT`, an abort of its router only as a last resort, then [`retire`]). New commands go to
/// `fresh` from the moment this returns; a `&Client` taken before stays valid (see the module docs). Nothing in flight
/// on the retired client moves.
///
/// Fails with [`BudgetExhausted`], changing nothing, when every spare cell is used. Panics if `i` is not a member.
pub fn replace_member(pool: &Pool, i: usize, fresh: Client) -> Result<Client, BudgetExhausted> {
  let (clients, state) = (&pool.inner.clients, &pool.inner.live_edit);
  assert!(i < clients.len(), "member {i} of a pool of {}", clients.len());
  let spare = state
    .next_spare
    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |next| {
      (next < state.spares.len()).then_some(next + 1)
    })
    .map_err(|_| BudgetExhausted)?;
  // The spare cell is this call's alone (the counter handed it out once), so the set cannot find it filled.
  let _ = state.spares[spare].set(fresh);
  let previous = state.current[i].swap(clients.len() + spare, Ordering::AcqRel);
  Ok(cell(clients, state, previous).unwrap_or(&clients[i]).clone())
}

/// How many more replacements `pool` can make before [`replace_member`] fails with [`BudgetExhausted`].
pub fn spare_cells_left(pool: &Pool) -> usize {
  let state = &pool.inner.live_edit;
  state
    .spares
    .len()
    .saturating_sub(state.next_spare.load(Ordering::Acquire))
}

/// How long [`retire`] waits for the router it follows to finish before it releases the client anyway.
pub const RETIRE_ROUTER_WAIT: Duration = Duration::from_secs(5);

/// How long [`retire`] gives the backchannel's close (`Backchannel::clear_router_state`) before it drops it.
///
/// The close is one socket's: take the backchannel's async write lock, take the connection out, flush whatever is
/// buffered on it, then shut it down — and fred puts no timeout on any of it (`ExclusiveConnection::disconnect`). On a
/// healthy socket it takes well under a millisecond. It waits in two places:
///
/// - **for the lock**, while another holder has it: the client's router (its cluster-state sync in
///   `router/clustered.rs`, and the unblock of a blocked connection), or a backchannel command sent on the client
///   (`CLIENT UNBLOCK`, `FUNCTION KILL`, and the like), which first `PING`s an existing backchannel connection with no
///   timeout (`check_and_create_transport`).
///   The live-edit service sends no backchannel command on a pool member's client, so in a retirement this wait is
///   reachable only when the old router is still running as the release starts: [`retire`] got no `router`, or the
///   router did not finish within [`RETIRE_ROUTER_WAIT`];
/// - **in the flush**, when the peer stopped reading.
///
/// A longer wait helps neither: a holder or a peer that is stuck stays stuck. So 1 s bounds a single socket close with
/// ample margin, and the release task as a whole ends within [`RETIRE_ROUTER_WAIT`] plus this. What a close cut short
/// leaves behind depends on where it was cut; see [`retire`].
pub const RETIRE_BACKCHANNEL_CLOSE_TIMEOUT: Duration = Duration::from_secs(1);

/// Release what a retired or discarded client still holds once its caller has ended its router (a `QUIT`, and an
/// abort as the last resort), which fred releases only on paths a retirement may not take:
///
/// - its **command channel**. A router that ends through fred's normal `QUIT` exit puts the command receiver back on
///   the client, so the channel stays open with nothing reading it: a later send — a `&Client` borrowed before the
///   replacement, such as a background walk requesting its next `SCAN` page — would be accepted and wait forever
///   unless it carries its own timeout, which a scan stream never does. Here the receiver is taken, closed, and every
///   command still queued in it is cancelled (`RouterCommand::cancel`: the caller gets `Canceled`, a scan stream an
///   `Err`). A send after that fails at once through fred's closed-channel path, with an `Unknown` error ("Failed to
///   send command to router." or "Client is not initialized.", by call path) — which the service classifies as
///   permanent; the only callers that hold a member's `&Client` that
///   long are background walks, which retry on their next tick;
/// - its **credential-refresh task**, which fred stops when the router loop exits normally (a `QUIT` that completed)
///   or when the client's `ClientInner` drops — and a retired client's never drops, because its spare cell keeps it
///   (and the task holds a clone of it). Left running it would fetch credentials, and log failed refreshes, for the
///   life of the process;
/// - its **backchannel connection**, which fred closes after a completed `QUIT` (`check_and_disconnect`) or when the
///   next router starts (`clear_router_state`) — neither happens to a client whose `QUIT` did not complete and whose
///   router was aborted, so the husk would keep a socket to a node, with its buffers, until the process exits.
///
/// `router` is the abort handle of the router the caller ended or aborted, when it has one. An abort takes effect
/// only at the router's next await, and a router that has just connected re-arms the credential task with no await
/// in between, so the release waits (up to [`RETIRE_ROUTER_WAIT`]) until that router has finished; only then is
/// nothing left that could re-arm the task or put the receiver back.
///
/// Does not block: the release runs on a task spawned on the current Tokio runtime, so this can run from a drop
/// guard. Outside a runtime the parts that need no await run at once. Harmless when any of them is already gone.
///
/// The spawned task is detached and bounded: it waits at most [`RETIRE_ROUTER_WAIT`] for the router, and gives the
/// backchannel's close at most [`RETIRE_BACKCHANNEL_CLOSE_TIMEOUT`]. Dropping that close part-way never corrupts the
/// backchannel, but what it leaves depends on where it was cut, and the expiry logs one WARN either way, naming the
/// client:
///
/// - **cut short after it took the backchannel's lock**: the connection is already out of the backchannel, so the
///   drop releases the lock with the backchannel holding no connection and drops the taken `ExclusiveConnection`,
///   whose socket the OS then closes; only the unflushed bytes of a socket nobody reads again are lost;
/// - **cut short while waiting for the lock** (see [`RETIRE_BACKCHANNEL_CLOSE_TIMEOUT`] for the holders and when that
///   can happen in a retirement): the drop just leaves the lock's queue and nothing is taken, so the backchannel
///   connection stays on the client until the client is dropped. A client kept in a spare cell (a pool member's
///   retired client) is never dropped before its `Pool`, so that one socket to a node, with its buffers, stays open
///   until the pool's last clone drops — the leak this release exists to close, left as a known residual for the rare
///   retirement whose old router outlived the wait. A client kept in no cell (a discarded fresh client, a `txn` slot's
///   swapped-out client) keeps the socket only until its last clone drops, normally soon after its router ends.
pub fn retire(client: &Client, router: Option<AbortHandle>) {
  let inner = client.inner.clone();
  match tokio::runtime::Handle::try_current() {
    Ok(runtime) => {
      runtime.spawn(async move {
        if let Some(router) = router {
          let deadline = Instant::now() + RETIRE_ROUTER_WAIT;
          while !router.is_finished() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
          }
        }
        release(&inner);
        if tokio::time::timeout(
          RETIRE_BACKCHANNEL_CLOSE_TIMEOUT,
          inner.backchannel.clear_router_state(&inner),
        )
        .await
        .is_err()
        {
          _warn!(
            inner,
            "Retired client's backchannel close timed out after {:?} (RETIRE_BACKCHANNEL_CLOSE_TIMEOUT); its \
             backchannel connection may stay open until the client is dropped (for a pool member, with its pool).",
            RETIRE_BACKCHANNEL_CLOSE_TIMEOUT
          );
        }
      });
    },
    Err(_) => release(&inner),
  }
}

/// The parts of [`retire`] that need no await: stop the credential-refresh task, and take, close and drain the command
/// receiver.
fn release(inner: &RefCount<ClientInner>) {
  #[cfg(feature = "credential-provider")]
  inner.abort_credential_refresh_task();
  if let Some(mut rx) = inner.take_command_rx() {
    rx.close();
    while let Ok(command) = rx.try_recv() {
      command.cancel();
    }
  }
}

/// How many spare cells `pool` has in all: `max(size, 4)`.
pub fn spare_cells(pool: &Pool) -> usize {
  pool.inner.live_edit.spares.len()
}
