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

# fred 10.1.0: vendored with one patch

`vendor/fred-10.1.0/` is the published `fred` 10.1.0 crate from crates.io with
the local changes below, all carried by one patch file. The root `Cargo.toml` points fred at it through
`[patch.crates-io]`, so the service binary, the tests and the examples all
build this copy instead of the registry one.

## Upstream source

- **Crate:** `fred` 10.1.0 (MIT), <https://crates.io/crates/fred/10.1.0>
- **Repository:** <https://github.com/aembke/fred.rs>. The 10.1.0 release
  commit on `main` is `f222ad7bfba844dbdc57e93da61b0a5483858df9`
  (2025-02-27); the crate's `.cargo_vcs_info.json` records the commit it was
  packaged from, `29d4790e8522a3b0c67531081fc0488585dbc665`.
- **Archive:** <https://static.crates.io/crates/fred/fred-10.1.0.crate>
- **Crate SHA256:** `3a7b2fd0f08b23315c13b6156f971aeedb6f75fb16a29ac1872d2eabccc1490e`
  (the crates.io index checksum, which `Cargo.lock` carried before the patch)
- **Vendor date:** 2026-10-02

The tree is the `.crate` archive extracted as is, not a copy of cargo's
`registry/src` directory, which adds a `.cargo-ok` marker. Every file the patch
does not name is byte-identical to the archive. The copy in the GitHub source
release lacks two of them, `Cargo.lock` and `.vscode/settings.json`, which git
ignore rules drop when the release is committed; neither is read by any build,
and `scripts/check-vendored-fred.sh` accepts their absence there, never a
change. In this repository both must be present.

`vendor/fred-10.1.0.SHA256SUMS` records every file of that archive as
published, before the patch: 171 lines in GNU `sha256sum` format (the digest,
two spaces, the path), with paths relative to the crate root and no `./`
prefix (for example `src/lib.rs`), sorted byte-wise (`LC_ALL=C`). It was
generated from inside the extracted, checksum-verified archive, before the
patch was applied, with:

```sh
find . -type f -printf '%P\0' | LC_ALL=C sort -z | xargs -0 sha256sum > ../fred-10.1.0.SHA256SUMS
```

So it tells, without the network, which files of the tree are upstream's as
published: every listed file the patch does not edit has its listed digest in
the tree, the 23 files the patch edits do not, and the six files the patch
adds under `src/live_edit/` are not listed. `Cargo.lock` and
`.vscode/settings.json` stay listed for the GitHub source release, where they
are absent. `tools/tests/test_vendored_fred_scripts.py` holds the manifest to
these facts against the tree and the patch.

## License

- **Upstream files** are fred's, under the MIT license its `Cargo.toml`
  declares (`license = "MIT"`): `LICENSE-MIT`, Copyright 2024 Alec Embke. The
  crate also ships a `LICENSE-APACHE` under the same copyright line; it is kept
  as published, and this repository takes the crate under the declared MIT
  license. Every upstream notice stays as published, and fred's own
  `Cargo.toml` `license` field is upstream's declaration and is not edited.
  `vendor/fred-10.1.0.SHA256SUMS` records the upstream files' published bytes.
- **NVIDIA's modifications to upstream files.** The patch edits 23 of fred's
  files, and every hunk that edits them carries a `live-edit patch` comment
  on at least one of its added lines. Those
  modifications are Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES and are
  licensed under the MIT license, so each modified file stays under MIT alone.
- **NVIDIA's new files.** The six files the patch adds under `src/live_edit/`
  (`abandon.rs`, `ask.rs`, `liveness.rs`, `mod.rs`, `pool.rs` and
  `retention.rs`) are
  Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES and licensed under the
  Apache License 2.0. Each carries NVIDIA's SPDX Apache-2.0 header, and the
  repository's `LICENSE.txt` holds the license text.

The vendored crate as a whole is therefore `MIT AND Apache-2.0`.

## The patch

`vendor/fred-10.1.0.patch`, applied with `patch -p1` inside the tree:

```diff
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -213,7 +213,7 @@
     "enable-native-tls",
     "native-tls/vendored",
 ]
-
+live-edit-retention-report = [] # live-edit patch: see vendor/fred-10.1.0.SOURCE.md (test-only; only live-edit's test-support enables it)
 [lib]
 name = "fred"
 path = "src/lib.rs"
--- a/src/clients/pipeline.rs
+++ b/src/clients/pipeline.rs
@@ -265,14 +265,14 @@
     return Vec::new();
   }
 
-  let (mut command, rx) = prepare_all_commands(commands, false);
+  let (mut command, rx) = prepare_all_commands(commands, false); let abandon = crate::live_edit::abandon::Guard::share(&mut command); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   command.inherit_options(inner);
   let timeout_dur = command.timeout_dur().unwrap_or_else(|| inner.default_command_timeout());
 
   if let Err(e) = interfaces::send_to_router(inner, command) {
     return vec![Err(e)];
   };
-  let frame = match utils::timeout(rx, timeout_dur).await {
+  let frame = match utils::timeout(abandon.watch(rx), timeout_dur).await { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     Ok(result) => match result {
       Ok(f) => f,
       Err(e) => return vec![Err(e)],
@@ -292,12 +292,12 @@
     return Ok(Value::Array(Vec::new()));
   }
 
-  let (mut command, rx) = prepare_all_commands(commands, true);
+  let (mut command, rx) = prepare_all_commands(commands, true); let abandon = crate::live_edit::abandon::Guard::share(&mut command); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   command.inherit_options(inner);
   let timeout_dur = command.timeout_dur().unwrap_or_else(|| inner.default_command_timeout());
 
   interfaces::send_to_router(inner, command)?;
-  let frame = utils::timeout(rx, timeout_dur).await??;
+  let frame = utils::timeout(abandon.watch(rx), timeout_dur).await??; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   protocol_utils::frame_to_results(frame)
 }
 
@@ -310,11 +310,11 @@
   let (tx, rx) = oneshot_channel();
   let mut commands: Vec<Command> = commands.into_iter().collect();
   commands[len - 1].response = ResponseKind::Respond(Some(tx));
-  let mut command = RouterCommand::Pipeline { commands };
+  let mut command = RouterCommand::Pipeline { commands }; let abandon = crate::live_edit::abandon::Guard::share(&mut command); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   command.inherit_options(inner);
   let timeout_dur = command.timeout_dur().unwrap_or_else(|| inner.default_command_timeout());
 
   interfaces::send_to_router(inner, command)?;
-  let frame = utils::timeout(rx, timeout_dur).await??;
+  let frame = utils::timeout(abandon.watch(rx), timeout_dur).await??; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   protocol_utils::frame_to_results(frame)
 }
--- a/src/clients/pool.rs
+++ b/src/clients/pool.rs
@@ -21,10 +21,10 @@
 #[cfg(not(feature = "glommio"))]
 pub use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};
 
-struct PoolInner {
-  clients:          Vec<Client>,
+pub(crate) struct PoolInner { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
+  pub(crate) clients: Vec<Client>, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   counter:          AtomicUsize,
-  prefer_connected: AtomicBool,
+  prefer_connected: AtomicBool, pub(crate) live_edit: crate::live_edit::pool::PoolState, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 }
 
 /// A cheaply cloneable round-robin client pool.
@@ -44,7 +44,7 @@
 /// [clients](Self::clients), [next](Self::next), or [last](Self::last) to operate on individual clients if needed.
 #[derive(Clone)]
 pub struct Pool {
-  inner: RefCount<PoolInner>,
+  pub(crate) inner: RefCount<PoolInner>, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 }
 
 impl fmt::Debug for Pool {
@@ -67,7 +67,7 @@
     } else {
       Ok(Pool {
         inner: RefCount::new(PoolInner {
-          clients,
+          live_edit: crate::live_edit::pool::PoolState::new(clients.len()), clients, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
           counter: AtomicUsize::new(0),
           prefer_connected: AtomicBool::new(true),
         }),
@@ -100,7 +100,7 @@
 
       Ok(Pool {
         inner: RefCount::new(PoolInner {
-          clients,
+          live_edit: crate::live_edit::pool::PoolState::new(clients.len()), clients, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
           counter: AtomicUsize::new(0),
           prefer_connected: AtomicBool::new(true),
         }),
@@ -114,16 +114,16 @@
     utils::set_bool_atomic(&self.inner.prefer_connected, val)
   }
 
-  /// Read the individual clients in the pool.
-  pub fn clients(&self) -> &[Client] {
-    &self.inner.clients
-  }
+  /// Read the individual clients in the pool: each member's current client, in member order, as an owned snapshot (live-edit patch: see vendor/fred-10.1.0.SOURCE.md).
+  pub fn clients(&self) -> Vec<Client> { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
+    crate::live_edit::pool::members(&self.inner.clients, &self.inner.live_edit).cloned().collect() // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
+  } #[doc = "Member `i`'s current client (`i` modulo the pool size): the pool's original client, or the fresh one `live_edit::pool::replace_member` pointed it at."] pub fn member(&self, i: usize) -> &Client { crate::live_edit::pool::member(&self.inner.clients, &self.inner.live_edit, i) } // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 
   /// Connect each client to the server, returning the task driving each connection.
   ///
   /// Use the base [connect](Self::connect) function to return one handle that drives all connections via [join](https://docs.rs/futures/latest/futures/macro.join.html).
   pub fn connect_pool(&self) -> Vec<ConnectHandle> {
-    self.inner.clients.iter().map(|c| c.connect()).collect()
+    crate::live_edit::pool::members(&self.inner.clients, &self.inner.live_edit).map(|c| c.connect()).collect() // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   }
 
   /// Read the size of the pool.
@@ -136,24 +136,24 @@
     let mut idx = utils::incr_atomic(&self.inner.counter) % self.inner.clients.len();
 
     for _ in 0 .. self.inner.clients.len() {
-      let client = &self.inner.clients[idx];
+      let client = self.member(idx); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       if client.is_connected() {
         return client;
       }
       idx = (idx + 1) % self.inner.clients.len();
     }
 
-    &self.inner.clients[idx]
+    self.member(idx) // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   }
 
   /// Read the client that should run the next command.
   pub fn next(&self) -> &Client {
-    &self.inner.clients[utils::incr_atomic(&self.inner.counter) % self.inner.clients.len()]
+    self.member(utils::incr_atomic(&self.inner.counter) % self.inner.clients.len()) // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   }
 
   /// Read the client that ran the last command.
   pub fn last(&self) -> &Client {
-    &self.inner.clients[utils::read_atomic(&self.inner.counter) % self.inner.clients.len()]
+    self.member(utils::read_atomic(&self.inner.counter) % self.inner.clients.len()) // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   }
 
   /// Create a client that interacts with the replica nodes associated with the [next](Self::next) client.
@@ -178,7 +178,7 @@
   /// Update the internal [PerformanceConfig](crate::types::config::PerformanceConfig) on each client in place with
   /// new values.
   fn update_perf_config(&self, config: PerformanceConfig) {
-    for client in self.inner.clients.iter() {
+    for client in crate::live_edit::pool::members(&self.inner.clients, &self.inner.live_edit) { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       client.update_perf_config(config.clone());
     }
   }
@@ -187,7 +187,7 @@
   ///
   /// This may contain duplicates when separate clients are connected to the same server.
   fn active_connections(&self) -> Vec<Server> {
-    self.inner.clients.iter().flat_map(|c| c.active_connections()).collect()
+    crate::live_edit::pool::members(&self.inner.clients, &self.inner.live_edit).flat_map(|c| c.active_connections()).collect() // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   }
 
   /// Override the DNS resolution logic for all clients in the pool.
@@ -196,7 +196,7 @@
   #[allow(refining_impl_trait)]
   fn set_resolver(&self, resolver: RefCount<dyn Resolve>) -> impl Future + Send {
     async move {
-      for client in self.inner.clients.iter() {
+      for client in crate::live_edit::pool::members(&self.inner.clients, &self.inner.live_edit) { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
         client.set_resolver(resolver.clone()).await;
       }
     }
@@ -211,7 +211,7 @@
   ///
   /// See [init](Self::init) for an alternative shorthand.
   fn connect(&self) -> ConnectHandle {
-    let clients = self.inner.clients.clone();
+    let clients = self.clients(); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     spawn(async move {
       let tasks: Vec<_> = clients.iter().map(|c| c.connect()).collect();
       for result in join_all(tasks).await.into_iter() {
@@ -227,7 +227,7 @@
   /// When running against a cluster this function will also refresh the cached cluster routing table.
   fn force_reconnection(&self) -> impl Future<Output = FredResult<()>> + Send {
     async move {
-      try_join_all(self.inner.clients.iter().map(|c| c.force_reconnection())).await?;
+      try_join_all(crate::live_edit::pool::members(&self.inner.clients, &self.inner.live_edit).map(|c| c.force_reconnection())).await?; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       Ok(())
     }
   }
@@ -235,7 +235,7 @@
   /// Wait for all the clients to connect to the server.
   fn wait_for_connect(&self) -> impl Future<Output = FredResult<()>> + Send {
     async move {
-      try_join_all(self.inner.clients.iter().map(|c| c.wait_for_connect())).await?;
+      try_join_all(crate::live_edit::pool::members(&self.inner.clients, &self.inner.live_edit).map(|c| c.wait_for_connect())).await?; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       Ok(())
     }
   }
@@ -266,9 +266,9 @@
     #[allow(unused_mut)]
     async move {
       let mut rxs: Vec<_> = self
-        .inner
-        .clients
-        .iter()
+        .clients() // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
+        .iter() // live-edit patch: each member's current client
+        // live-edit patch
         .map(|c| c.inner().notifications.connect.load().subscribe())
         .collect();
 
@@ -283,7 +283,7 @@
         });
 
       if let Some(err) = init_err {
-        for client in self.inner.clients.iter() {
+        for client in crate::live_edit::pool::members(&self.inner.clients, &self.inner.live_edit) { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
           utils::reset_router_task(client.inner());
         }
 
@@ -303,7 +303,7 @@
   /// pool.
   fn quit(&self) -> impl Future<Output = FredResult<()>> + Send {
     async move {
-      join_all(self.inner.clients.iter().map(|c| c.quit())).await;
+      join_all(crate::live_edit::pool::members(&self.inner.clients, &self.inner.live_edit).map(|c| c.quit())).await; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 
       Ok(())
     }
@@ -321,7 +321,7 @@
       loop {
         sleep(interval).await;
 
-        if let Err(error) = try_join_all(self.inner.clients.iter().map(|c| c.ping::<()>(None))).await {
+        if let Err(error) = try_join_all(crate::live_edit::pool::members(&self.inner.clients, &self.inner.live_edit).map(|c| c.ping::<()>(None))).await { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
           if break_on_error {
             return Err(error);
           }
--- a/src/clients/transaction.rs
+++ b/src/clients/transaction.rs
@@ -313,15 +313,15 @@
     id,
     commands.len(),
   );
-  let command = RouterCommand::Transaction {
+  let mut command = RouterCommand::Transaction { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     id,
     tx,
     commands,
     abort_on_error,
-  };
+  }; let abandon = crate::live_edit::abandon::Guard::share(&mut command); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   let timeout_dur = trx_options.timeout.unwrap_or_else(|| inner.default_command_timeout());
 
   interfaces::send_to_router(inner, command)?;
-  let frame = utils::timeout(rx, timeout_dur).await??;
+  let frame = utils::timeout(abandon.watch(rx), timeout_dur).await??; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   protocol_utils::frame_to_results(frame)
 }
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -55,7 +55,7 @@
 mod protocol;
 mod router;
 mod trace;
-mod utils;
+mod utils; pub mod live_edit; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 
 /// Redis client implementations.
 pub mod clients;
--- a/src/live_edit/abandon.rs
+++ b/src/live_edit/abandon.rs
@@ -0,0 +1,389 @@
+// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+// SPDX-License-Identifier: Apache-2.0
+//
+// Licensed under the Apache License, Version 2.0 (the "License");
+// you may not use this file except in compliance with the License.
+// You may obtain a copy of the License at
+//
+//     http://www.apache.org/licenses/LICENSE-2.0
+//
+// Unless required by applicable law or agreed to in writing, software
+// distributed under the License is distributed on an "AS IS" BASIS,
+// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+// See the License for the specific language governing permissions and
+// limitations under the License.
+
+//! Calls their callers gave up on: a command, pipeline or transaction handed to fred is never written after its
+//! caller gave up on it.
+//!
+//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
+//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
+//!
+//! # Stock fred
+//!
+//! Every command carries a `timed_out` flag (`Command::timed_out`, `src/protocol/command.rs:1556`), and the router
+//! refuses to write a command whose flag is set: at the write (`src/router/utils.rs:71-78`, which every path that
+//! writes a command or a pipeline's command goes through — the command channel, the retry buffer's replay after a
+//! reconnect or a cluster sync, a `-MOVED` retry), at the replay (`src/router/mod.rs:374`) and before following an
+//! `-ASK` ([`super::ask::follow`]). But stock fred sets the flag in one place only, the request-response path
+//! (`src/utils.rs:402-421`), and only when the call ends with an error. So:
+//!
+//! - a call whose future is dropped — an abort, a deadline, a cancelled client, a losing `select!` arm — never sets
+//!   it, and its command is written whenever the router next reaches it, for example once a reconnect that runs
+//!   inline under an unlimited reconnect policy finally succeeds;
+//! - a pipeline (`src/clients/pipeline.rs:263-320`) and a transaction (`src/clients/transaction.rs:278-327`) never set
+//!   it at all, even when their own timeout fires: `Command::duplicate` gives each queued command a fresh flag
+//!   (`src/protocol/command.rs:1855`) that nothing ever sets;
+//! - the transaction router (`src/router/transactions.rs`) writes its frames with `Connection::write` directly and
+//!   never reads the flag, so a transaction is written whatever its flags say, and fred's own whole-transaction
+//!   replay after a reconnect (`retry!`, `:122-133`) writes it again.
+//!
+//! # The rule
+//!
+//! Once a call's caller has given up — its timeout fired, fred answered it with an error, or its future was dropped — no
+//! frame of that call that fred has not yet written is written. The rule is *never written after*, not *never
+//! executed*: a frame already handed on before the caller gave up is not recalled (see *The residual*).
+//!
+//! # One flag per call
+//!
+//! Each fred entry point arms a [`Guard`] on one flag before it hands the work to the router: a single command's own
+//! flag (`basic_request_response`, `src/utils.rs:402`, and its tracing twin, `:459`, which this build does not compile
+//! but which is hooked the same way so the behaviour does not depend on features), or, for a batch, one fresh flag
+//! that [`Guard::share`] assigns to every command of the pipeline (`try_send_all`, `send_all` and `send_last`,
+//! `src/clients/pipeline.rs:268`, `:295`, `:313`) or of the transaction (`exec`, `src/clients/transaction.rs:321`),
+//! in place of the flags `Command::duplicate` gave them. [`Guard::watch`] wraps the call's reply receiver:
+//!
+//! - a reply that resolves `Ok` disarms the guard, so a completed call never touches the flag. That keeps
+//!   `Pipeline::last` as it was for a pipeline spread over several nodes, whose earlier commands may still be in
+//!   flight when the last one answers (the service never calls `last`). A server's error reply to a command reaches
+//!   the receiver as an `Ok` frame too (`src/protocol/responders.rs:442`) and disarms the guard, which is harmless:
+//!   the call is complete;
+//! - any other outcome — fred's own timeout, an `Err` fred itself sends on the receiver (a failed write, a lost
+//!   connection, a pipeline answered early with its first error, a transaction's error), an errored receiver — sets
+//!   the flag, as does dropping the guard at any point before an `Ok`, which covers a dropped future, an abort and an
+//!   unwinding panic.
+//!
+//! Because a batch shares one flag, setting it covers every frame of the batch at once, and the router's existing
+//! checks need no change: a pipeline's command still in fred's channel, behind a reconnect, in the retry buffer or
+//! redirected is skipped where a timed-out single command already was.
+//!
+//! # Pipelines
+//!
+//! A pipeline is written in one router step (`process_pipeline`, `src/router/commands.rs:230-248`), one command at a
+//! time, and the flag can be set partway through. The commands after that point are skipped at the write, which can
+//! leave a `MULTI` written and its `EXEC` not: an open transaction on that connection, which would turn the next
+//! commands sent there — the service's checkout hygiene probe on a `txn` connection, a `WatchGuard`'s `UNWATCH` —
+//! into `QUEUED` replies. So [`PipelineStep`] records the last `MULTI` the step hands to the write with its flag still
+//! clear — by kind, `CommandKind::Multi` or a custom command whose first word is `MULTI` (as the service's
+//! `WatchedTxn` sends it) — and the connection it routes to, and [`after_pipeline`], at the end of the step, writes
+//! one `DISCARD` there when the flag is then set. Nothing needs to know whether the `EXEC` was written: when it was,
+//! the `DISCARD` answers `-ERR DISCARD without MULTI` and changes nothing.
+//!
+//! The `DISCARD` is an ordinary tracked command: written through the router's own write (`src/router/utils.rs:65`)
+//! on the `MULTI`'s connection, in the same router step, so no other caller's frame comes between, its reply matched
+//! in order and dropped (`ResponseKind::Respond(None)`), and nothing is read synchronously: a synchronous read on a
+//! connection with commands in flight takes another caller's reply ([`super::ask`]'s module docs). It is written with
+//! no write attempt left, as [`super::ask`] writes `ASKING`, so a replay of the
+//! connection's in-flight commands after it drops discards it instead of sending it to a fresh connection, where it
+//! would have no `MULTI` to discard. The step opens no connection, so the connection found under the `MULTI`'s server
+//! is the one the `MULTI` was written to; when a write error dropped it meanwhile there is none, nothing is written,
+//! and the store discarded the `MULTI` with its connection. A `MULTI` routed to a replica (`use_replica`, with the
+//! `replicas` feature, which the service does not build) is not recorded, because routing it twice would advance the
+//! replica round-robin.
+//!
+//! The `DISCARD` covers a cut during the router step that writes the pipeline, and only that. A pipeline whose commands
+//! have a write attempt left and whose connection drops is replayed from the retry buffer one command at a time
+//! (`Router::retry_buffer`, `src/router/mod.rs:366-397`), with no [`PipelineStep`]: a flag set partway through that
+//! replay skips the rest but leaves a replayed `MULTI` open. The service sends every pipeline holding a `MULTI` on its
+//! `txn` pool, at one write attempt, so it never replays one.
+//!
+//! # fred `Transaction`s
+//!
+//! The transaction router checks the shared flag at the top of each attempt, again after the drain that starts it
+//! (which can wait up to the unresponsive bound), before the `ASKING` it sends after an `-ASK`, before each command
+//! and before `EXEC` (`src/router/transactions.rs:121`, `:145`, `:174`, `:189`, `:196`). Once the flag is set:
+//!
+//! - when no frame of the attempt has been written, the caller's sender is answered with a `Canceled` error
+//!   ([`TRANSACTION_ABANDONED`]) and nothing is written ([`refuse_transaction`]). That covers a transaction queued
+//!   behind a reconnect and fred's own whole-transaction replay after one, which starts a new attempt;
+//! - otherwise — an `ASKING` or a `MULTI` was written — the connection is closed without a flush or a shutdown and a
+//!   reconnection is deferred, as a failed write does ([`close_transaction`]). The store discards an unexecuted `MULTI`
+//!   when its connection closes. The close is marked on the connection ([`ConnectionState`]), and `Connection::close`'s
+//!   hook ([`super::close`], change 3's) returns at once on a marked connection.
+//!
+//! Closing, rather than writing `DISCARD`, is deliberate here: transaction frames are written outside the in-flight
+//! queue, so a `DISCARD` would need a synchronous read of every pending `+QUEUED` reply first. The attempt drained
+//! every connection before it began, so the closed connection holds no other caller's command. The close reaches the
+//! client's error events once, as a `Canceled` error ([`CONNECTION_CLOSED`]), and costs one reconnect. That reconnect
+//! is deferred to the back of the router's command channel, so a command already queued there is routed before it.
+//! On a centralized client, whose one connection was closed, that command finds no connection and fails "Failed to
+//! route command." (or is kept for a retry when it has a write attempt left); a cluster client routes it as after
+//! any failed write. A routing miss means the router holds no connection at all, so no transaction state can be left
+//! for such a command to find; the service reads that miss on its `txn` pool (a checkout probe, an `UNWATCH`) as a
+//! connection already gone.
+//!
+//! # What happens to each frame
+//!
+//! | When the caller gives up, the frame is | Pipeline (and the service's `WatchedTxn`) | fred `Transaction` |
+//! |---|---|---|
+//! | still in fred's command channel (e.g. behind a reconnect) | skipped at the write | refused at the attempt's check; nothing written |
+//! | not yet written, the router partway through the batch | skipped at the write; `DISCARD` when a `MULTI` was written | not written; the connection closed |
+//! | written, awaiting a reply, when the connection drops | not replayed | the next attempt refused at its check |
+//! | redirected (`-ASK` / `-MOVED`) | not followed | — |
+//! | an `EXEC` already written, only its reply late | **runs: this cannot be stopped** | **runs: this cannot be stopped** |
+//!
+//! # The residual
+//!
+//! A frame handed to fred's write buffer or to the socket before the flag was set is not recalled. It leaves at the
+//! router's next flush, and each wait on its own connection's I/O there is change 3's bounded write or flush wait
+//! (`UnresponsiveConfig::max_timeout`, 45 s in the service): it reaches the socket within one such wait, or its
+//! connection is failed. On a clustered client that holds however many of the client's connections stall, inline
+//! reconnects, cluster syncs and `-ASK` connects included: before the router waits, connects inline, or makes a write
+//! that can reach a socket while another connection holds fed frames, it flushes every connection holding them, all
+//! at once, and fred's router-wide flush runs its connections' flushes at once too (change 5, [`super::liveness`]'s
+//! *Never waiting on unsent frames*). Its exceptions: the flush before a `CLIENT UNBLOCK`, which only the service's
+//! stream readers reach, and they write nothing to the keyspace; fred's stock close, up to 5 s on a peer that stopped
+//! reading with its socket still open: once for `drop_connection`'s close after a read error or the end of the stream,
+//! and per such peer at a router's end, where `disconnect_all` closes them one after another (a failed cluster-wide
+//! command runs `drain_all` first); a `Moved`'s `cluster_cache_update_delay`, which the service leaves at 0; and the
+//! `replicas` feature, which it does not build. On a centralized or sentinel client a frame fed just before a reconnect
+//! can also wait out that pass's backoff sleep (up to about 200 ms: the service's 100 ms first reconnect delay plus
+//! fred's up-to-100 ms jitter) before the pass's close flushes it. Once in the kernel's send queue, its delivery and
+//! execution are TCP's and the store's, for a single command as for an `EXEC` (SOURCE.md change 9, *The residual*).
+//!
+//! # Not guarded
+//!
+//! Senders that hand the router a command without `basic_request_response`, none of them a keyspace data write the
+//! service sends: `QUIT` and `SHUTDOWN` (`src/commands/impls/server.rs:38`, `:69`), `HELLO` / `AUTH` (`:189`, `:212`),
+//! the cluster-wide commands — script and function administration (`src/commands/impls/lua.rs:77`, `:94`, `:121`,
+//! `:295`, `:330`, `:397`, `:449`), a cluster-wide `FLUSHALL` (`server.rs:133`) and client tracking
+//! (`src/commands/impls/tracking.rs:88`, `:113`), which go to every node through a path that never reads the flag —
+//! and the `SCAN` family's pages (`src/commands/impls/scan.rs`, `src/types/scan.rs`). The service reaches one of them:
+//! a `NOSCRIPT` reload's `SCRIPT LOAD` goes through `Script::load` to `script_load_cluster`
+//! (`src/types/scripts.rs:200`), which on a cluster client sends it to every node directly (`lua.rs:77`; on any other
+//! client it is an ordinary, guarded `SCRIPT LOAD`). It writes the script cache, not the keyspace, so it stays
+//! unguarded.
+
+use crate::{
+  error::{Error, ErrorKind},
+  modules::inner::ClientInner,
+  protocol::{
+    command::{Command, CommandKind, ResponseSender, RouterCommand},
+    connection::Connection,
+    types::Server,
+  },
+  router::{utils, Router},
+  runtime::{AtomicBool, RefCount},
+  utils as client_utils,
+};
+use std::future::Future;
+
+/// The error a fred `Transaction` the router refused is answered with: its caller gave up before any frame of the
+/// current attempt was written. The caller has already returned, so no caller sees it.
+pub const TRANSACTION_ABANDONED: &str = "Transaction abandoned by its caller: this attempt was not written.";
+
+/// The error a connection closed under an abandoned fred `Transaction` reports, on the client's error events and to
+/// the transaction's sender.
+pub const CONNECTION_CLOSED: &str = "Connection closed: its transaction was abandoned partway through its write.";
+
+/// Per-connection state, on `ConnectionState::abandon`.
+#[derive(Debug, Default)]
+pub struct ConnectionState {
+  /// Set by [`close_transaction`] just before it closes the connection: `Connection::close` then neither flushes nor
+  /// shuts the transport down ([`super::close`]). Never cleared; it is dropped with the connection.
+  close_at_once: bool,
+}
+
+/// Whether `conn` is being closed under an abandoned transaction, so its close must not flush or shut it down.
+pub(crate) fn closes_at_once(conn: &Connection) -> bool {
+  conn.live_edit.abandon.close_at_once
+}
+
+/// The give-up guard of one call (module docs, *One flag per call*): armed on the flag every command of the call
+/// shares, it sets that flag when it is dropped still armed. Only an `Ok` reply disarms it ([`Guard::watch`]).
+pub(crate) struct Guard {
+  flag: Option<RefCount<AtomicBool>>,
+}
+
+impl Guard {
+  /// Arm on a single command's own flag.
+  pub(crate) fn arm(flag: &RefCount<AtomicBool>) -> Guard {
+    Guard {
+      flag: Some(flag.clone()),
+    }
+  }
+
+  /// Give every command of the batch `command` (a pipeline or a transaction) one fresh flag, replacing the flags
+  /// `Command::duplicate` gave them, and arm on it. Any other router command has no commands to share it with.
+  pub(crate) fn share(command: &mut RouterCommand) -> Guard {
+    let flag = RefCount::new(AtomicBool::new(false));
+    let commands: &mut [Command] = match command {
+      RouterCommand::Pipeline { commands } => commands,
+      #[cfg(feature = "transactions")]
+      RouterCommand::Transaction { commands, .. } => commands,
+      _ => &mut [],
+    };
+    for command in commands.iter_mut() {
+      command.timed_out = flag.clone();
+    }
+    Guard { flag: Some(flag) }
+  }
+
+  /// Await the call's reply receiver `reply`, disarming the guard when it resolves `Ok` with an `Ok` reply — a
+  /// server's error reply to a command included, which arrives as an `Ok` frame: the call is complete. Any other
+  /// outcome (an `Err` fred itself sent, or an errored receiver), and dropping the returned future before it resolves
+  /// (fred's own timeout drops it, as does a caller that gives up), sets the flag.
+  pub(crate) async fn watch<F, T, E>(mut self, reply: F) -> Result<Result<T, Error>, E>
+  where
+    F: Future<Output = Result<Result<T, Error>, E>>,
+  {
+    let result = reply.await;
+    if matches!(result, Ok(Ok(_))) {
+      self.flag = None;
+    }
+    result
+  }
+}
+
+impl Drop for Guard {
+  fn drop(&mut self) {
+    if let Some(flag) = self.flag.take() {
+      client_utils::set_bool_atomic(&flag, true);
+    }
+  }
+}
+
+/// Whether the caller of `command` has given up on it.
+pub(crate) fn command_given_up(command: &Command) -> bool {
+  client_utils::read_bool_atomic(&command.timed_out)
+}
+
+/// Whether the caller of the batch `commands` has given up on it. Its commands share one flag; any of them is read.
+pub(crate) fn given_up(commands: &[Command]) -> bool {
+  commands.iter().any(command_given_up)
+}
+
+/// Whether `kind` opens a transaction: `MULTI` by kind, or a custom command whose name — its first word, any case —
+/// is `MULTI`.
+fn is_multi(kind: &CommandKind) -> bool {
+  match kind {
+    CommandKind::_Custom(custom) => custom
+      .cmd
+      .split_whitespace()
+      .next()
+      .is_some_and(|name| name.eq_ignore_ascii_case("MULTI")),
+    other => other.is_multi(),
+  }
+}
+
+/// The `MULTI` a pipeline's router step handed to the write with its caller still waiting.
+struct OpenMulti {
+  /// The server the `MULTI` was routed to.
+  server: Server,
+  /// The flag the `MULTI` shares with the rest of its batch.
+  flag:   RefCount<AtomicBool>,
+}
+
+/// What one pipeline's router step wrote that a cut can leave open (module docs, *Pipelines*).
+#[derive(Default)]
+pub(crate) struct PipelineStep {
+  open: Option<OpenMulti>,
+}
+
+impl PipelineStep {
+  /// Called just before `command` is handed to the router's write: record it when it is a `MULTI` whose caller is
+  /// still waiting, with the server the write will route it to.
+  pub(crate) fn before_write(&mut self, router: &mut Router, command: &Command) {
+    if !is_multi(&command.kind) || command_given_up(command) {
+      return;
+    }
+    // With the `replicas` feature, routing a replica command advances the replica round-robin, so a second route
+    // could name another replica than the write's (module docs).
+    #[cfg(feature = "replicas")]
+    if command.use_replica {
+      return;
+    }
+    if let Some(conn) = router.route(command) {
+      self.open = Some(OpenMulti {
+        server: conn.server.clone(),
+        flag:   command.timed_out.clone(),
+      });
+    }
+  }
+}
+
+/// The end of a pipeline's router step: when the step wrote a `MULTI` and its caller has since given up, write one
+/// `DISCARD` on that `MULTI`'s connection (module docs, *Pipelines*). An error ends the router task, as the router's
+/// own write does (`src/router/commands.rs:196-204`).
+pub(crate) async fn after_pipeline(
+  inner: &RefCount<ClientInner>,
+  router: &mut Router,
+  step: PipelineStep,
+) -> Result<(), Error> {
+  let open = match step.open {
+    Some(open) if client_utils::read_bool_atomic(&open.flag) => open,
+    _ => return Ok(()),
+  };
+  // The `DISCARD` is sent at once and can wait on the `MULTI`'s socket for up to a bound: first send what any other
+  // connection holds fed but unflushed, all connections at once, so no frame fed elsewhere waits through it
+  // (`super::before_write_to`; the liveness module's *Never waiting on unsent frames*).
+  super::before_write_to(inner, router, &open.server).await;
+  let conn = match router.get_connection_mut(&open.server) {
+    Some(conn) => conn,
+    None => {
+      _debug!(
+        inner,
+        "No connection to {} left to discard an abandoned pipeline's MULTI on.",
+        open.server
+      );
+      return Ok(());
+    },
+  };
+
+  _debug!(inner, "Discarding an abandoned pipeline's MULTI on {}", open.server);
+  let mut discard = Command::new(CommandKind::Discard, Vec::new());
+  // No write attempt left: a replay of this connection's in-flight commands discards it.
+  let _ = discard.decr_check_attempted();
+  match utils::write_command(inner, conn, discard, true).await {
+    Ok(_) => Ok(()),
+    Err((err, _)) => {
+      _debug!(inner, "Error sending DISCARD to {}: {:?}", open.server, err);
+      utils::drop_connection(inner, router, &open.server, &err).await;
+      utils::defer_reconnection(inner, router, None, err, false)
+    },
+  }
+}
+
+/// A fred `Transaction` whose caller gave up before any frame of the current attempt was written: answer its sender
+/// and write nothing (module docs, *fred `Transaction`s*).
+pub(crate) fn refuse_transaction(inner: &RefCount<ClientInner>, id: u64, tx: ResponseSender) -> Result<(), Error> {
+  _debug!(inner, "Ignore writing abandoned transaction {}", id);
+  let _ = tx.send(Err(Error::new(ErrorKind::Canceled, TRANSACTION_ABANDONED)));
+  Ok(())
+}
+
+/// A fred `Transaction` whose caller gave up after a frame of the current attempt was written on the connection to
+/// `server`: close that connection without a flush or a shutdown, answer the sender, and defer a reconnection, as a
+/// failed write does (module docs, *fred `Transaction`s*). An error ends the router task, as the router's own write
+/// does.
+pub(crate) async fn close_transaction(
+  inner: &RefCount<ClientInner>,
+  router: &mut Router,
+  server: Server,
+  id: u64,
+  tx: ResponseSender,
+) -> Result<(), Error> {
+  _debug!(
+    inner,
+    "Closing the connection to {} under abandoned transaction {}",
+    server,
+    id
+  );
+  if let Some(conn) = router.get_connection_mut(&server) {
+    conn.live_edit.abandon.close_at_once = true;
+  }
+  let error = Error::new(ErrorKind::Canceled, CONNECTION_CLOSED);
+  let _ = tx.send(Err(error.clone()));
+  utils::drop_connection(inner, router, &server, &error).await;
+  utils::defer_reconnection(inner, router, None, error, false)
+}
--- a/src/live_edit/ask.rs
+++ b/src/live_edit/ask.rs
@@ -0,0 +1,336 @@
+// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+// SPDX-License-Identifier: Apache-2.0
+//
+// Licensed under the Apache License, Version 2.0 (the "License");
+// you may not use this file except in compliance with the License.
+// You may obtain a copy of the License at
+//
+//     http://www.apache.org/licenses/LICENSE-2.0
+//
+// Unless required by applicable law or agreed to in writing, software
+// distributed under the License is distributed on an "AS IS" BASIS,
+// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+// See the License for the specific language governing permissions and
+// limitations under the License.
+
+//! Following a cluster `-ASK` redirect to its target node.
+//!
+//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
+//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
+//!
+//! A node answers `-ASK <slot> <target>` while `<slot>` is migrating away from it and the command's keys are no
+//! longer there. The client must send `ASKING` and then the command to `<target>` on one connection: the target serves
+//! an importing slot only to a connection that sent `ASKING`, and clears that flag after the next command (Valkey
+//! 9.0.3 `networking.c:3260-3262`). The slot's owner in the client's routing table stays the source until the
+//! migration ends.
+//!
+//! Stock `process_ask` (`src/router/commands.rs:251-285`) gets both halves wrong:
+//!
+//! - it sends `ASKING` to the target (`send_asking_with_policy`, `src/router/utils.rs:441-493`) and then re-writes the
+//!   command with only its hash slot set, so `Router::route` (`src/router/mod.rs:176-207`) sends it to the slot's
+//!   owner, the source, which answers `-ASK` again, until the command's write attempts run out and it fails `Routing`
+//!   "Max attempts reached.". No `-ASK` is ever followed.
+//! - it reads the `ASKING` reply synchronously off the target's connection (`src/router/utils.rs:477-488`). With
+//!   commands already in flight there, that read takes another caller's reply, and every later reply on the
+//!   connection is paired with the wrong command.
+//!
+//! [`follow`] instead writes `ASKING` and the command back to back on the target's connection, in one router step,
+//! and reads nothing: `ASKING` is an ordinary in-flight command whose `+OK` the reply matching consumes in order
+//! (`src/router/clustered.rs:291-330`, `ResponseKind::Respond(None)`). The pin to the target lasts for that one write:
+//! no field of the command is changed to name the target, so a replay after a dropped connection, or a later `-MOVED`,
+//! routes it by slot again. `ASKING` is written with no write attempt left, so a replay of a dropped connection's
+//! in-flight commands discards it (`Command::should_finish_with_error`, `src/protocol/command.rs:1771-1773`) rather
+//! than sending it to the slot's owner. When the client has no connection to the target (a node that owns no slot yet,
+//! as in a scale-out), one is opened for it (`Connections::add_connection`, `src/router/connections.rs:375`), bounded
+//! by [`ASK_TARGET_CONNECT_CAP`] because it runs on the router task, and announced on the client's reconnect events
+//! as a cluster sync announces a connection it adds; the next cluster sync that does not list the node as a primary
+//! closes it (`src/router/clustered.rs:205-218`, `:551-561`), and the commands still in flight on it are retried,
+//! routed by slot. Before it connects ([`super::before_inline_connect`]), and before it writes when another connection
+//! holds fed frames ([`super::before_write_to`]), every connection holding fed but unflushed frames is flushed, all at
+//! once: the `Ask` was queued behind the command ahead of it, which may be only fed, and neither step polls its socket.
+//!
+//! **A script never follows an `-ASK`** ([`is_script_command`]). The source routes a script by its declared keys only
+//! (Valkey 9.0.3 `cluster.c:1263-1273`), so a script that also reads undeclared keys of the same slot is sent to the
+//! target as soon as its declared keys have moved. There the script's own client inherits the `ASKING` flag
+//! (`script.c:431-433`), and in an importing slot a single-key command is served locally even when its key has not
+//! arrived yet (`cluster.c:1281-1287`): an undeclared row still on the source reads as absent, with no error, so the
+//! script returns a partial answer or acts on one. A script answered `-ASK` therefore fails as it did before this
+//! change — `Routing` "Max attempts reached.", which the caller's classifier already treats as transient — with
+//! nothing written and nothing sent to the target. Plain commands, which name every key they touch, still follow.
+//!
+//! The caller, `process_ask`, keeps everything before the follow: the redirection budget is checked first, so a client
+//! configured with no redirections is never redirected, and a command with no write attempt left still fails `Routing`
+//! "Max attempts reached." with nothing written (`send_asking_with_policy`'s check, `src/router/utils.rs:468-469`,
+//! which returns before any I/O).
+
+use crate::{
+  error::{Error, ErrorKind},
+  modules::inner::ClientInner,
+  protocol::{
+    command::{Command, CommandKind},
+    types::Server,
+  },
+  router::{connections::Connections, utils, Router},
+  runtime::RefCount,
+  types::{config::Blocking, ClientUnblockFlag},
+  utils as client_utils,
+};
+use std::time::Duration;
+
+/// The most an on-demand connection to an `-ASK` target may take, connect, TLS handshake and setup exchange together.
+///
+/// The connection is opened inline on the client's single router task, so every other command on the client waits
+/// while it runs; at the client's own connection timeout (the service's default is 30 s) an unreachable target would
+/// stall them all for that long, during exactly the window a node may be failing. 2 s leaves margin for a TLS
+/// handshake on a healthy managed node while bounding that stall. A client whose own connection timeout is shorter
+/// keeps it ([`ask_target_connect_timeout`]).
+pub const ASK_TARGET_CONNECT_CAP: Duration = Duration::from_secs(2);
+
+/// The bound on an on-demand `-ASK` target connection for a client whose connection timeout is `client`: the shorter
+/// of the two, and the cap when the client's is zero (fred's "no timeout", `src/utils.rs:291-298`).
+pub fn ask_target_connect_timeout(client: Duration) -> Duration {
+  if client.is_zero() {
+    ASK_TARGET_CONNECT_CAP
+  } else {
+    client.min(ASK_TARGET_CONNECT_CAP)
+  }
+}
+
+/// The script commands a client can send, by name: every one runs a script that may read keys it did not declare.
+/// fred has kinds for four of them; the `_RO` forms of `EVAL` / `EVALSHA` exist only as custom commands.
+const SCRIPT_COMMANDS: [&str; 6] = ["EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO", "FCALL", "FCALL_RO"];
+
+/// Whether `kind` runs a script, so that an `-ASK` to it must not be followed (module docs). Matched by command kind,
+/// never by caller: fred's `EVAL`, `EVALSHA`, `FCALL` and `FCALL_RO` kinds (`CommandKind::is_eval`,
+/// `src/protocol/command.rs:1520-1525`), and a custom command (`CommandKind::_Custom`, as `CustomCommand::new_static(
+/// "EVALSHA", …)` builds) whose name — its first word, any case — is one of [`SCRIPT_COMMANDS`].
+pub(crate) fn is_script_command(kind: &CommandKind) -> bool {
+  match kind {
+    CommandKind::_Custom(custom) => custom
+      .cmd
+      .split_whitespace()
+      .next()
+      .is_some_and(|name| SCRIPT_COMMANDS.iter().any(|script| name.eq_ignore_ascii_case(script))),
+    other => other.is_eval(),
+  }
+}
+
+/// The error a failed or cut connect to an `-ASK` target answers its command with; the text names the target either
+/// way.
+///
+/// An allowlist: only a configuration or credential fault keeps its own kind, so the caller's classifier answers it as
+/// the fault it is (not transient: retrying cannot fix it). In this build (rustls) those are:
+///
+/// - `Tls`: an invalid TLS server name (`src/protocol/connection.rs:390`), or a TLS connector that does not match the
+///   backend (`:373`, unreachable with one TLS backend compiled in). Not a bad certificate: a certificate or handshake
+///   failure reaches fred as an `std::io::Error` (tokio-rustls 0.26.4 `src/common/mod.rs:115`, through the handshake's
+///   `?` at `connection.rs:393`), which fred's `From<io::Error>` makes `IO` (`src/error.rs:179-182`), so it is
+///   retried here, as fred's own reconnect path retries it;
+/// - `Auth` (a `WRONGPASS` or `NOAUTH`), `Config`, `Url`: a credential, configuration or address fault.
+///
+/// Every other failure answers `IO`, the transient shape a lost connection already has, because retrying is what fixes
+/// it. Besides a connect failure, the cap's `Timeout` and a `Canceled`, that includes two setup outcomes a
+/// kind-preserving rule would wrongly leave permanent:
+///
+/// - a node that is joining or rejoining the cluster reports `cluster_state:fail` for a while (Valkey 9.0.3
+///   `cluster_legacy.c:6311-6313`, `:6358-6362`, `:6373-6377`), which fred's connection setup
+///   (`src/protocol/connection.rs:620-648`, its cluster-state check) answers `Protocol` "Invalid or missing cluster
+///   state." (`:615`);
+/// - a server error during setup (`-LOADING` while a node replays its data, `-BUSY` behind a long script) arrives as
+///   `Unknown`, and once this text wraps it, it no longer starts with the reply token the caller's not-serving check
+///   matches.
+///
+/// Whatever the answer, the failure is broadcast with its own kind (in [`follow`]), so the error metric counts it by
+/// what actually happened.
+fn ask_connect_error(target: &Server, err: &Error) -> Error {
+  // `ErrorKind::Tls` exists only in a build with a TLS feature.
+  #[cfg(any(
+    feature = "enable-native-tls",
+    feature = "enable-rustls",
+    feature = "enable-rustls-ring"
+  ))]
+  let tls = matches!(err.kind(), ErrorKind::Tls);
+  #[cfg(not(any(
+    feature = "enable-native-tls",
+    feature = "enable-rustls",
+    feature = "enable-rustls-ring"
+  )))]
+  let tls = false;
+  let kind = if tls || matches!(err.kind(), ErrorKind::Auth | ErrorKind::Config | ErrorKind::Url) {
+    err.kind().clone()
+  } else {
+    ErrorKind::IO
+  };
+  Error::new(kind, format!("Failed connecting to ASK target {target}: {err}"))
+}
+
+/// Send `ASKING` and then `command` to the `-ASK` target `server` for `slot`, on one connection, reading nothing.
+///
+/// The caller has checked the command's redirection budget and that it has a write attempt left. An error ends the
+/// router task, as stock `write_command`'s does (`src/router/commands.rs:196-204`).
+pub(crate) async fn follow(
+  inner: &RefCount<ClientInner>,
+  router: &mut Router,
+  server: Server,
+  slot: u16,
+  mut command: Command,
+) -> Result<(), Error> {
+  // A script is never followed (module docs): it fails as before this change, with nothing written or sent. Deleting
+  // this `if` block is the whole revert of that rule.
+  if is_script_command(&command.kind) {
+    command.respond_to_caller(Err(Error::new(ErrorKind::Routing, "Max attempts reached.")));
+    return Ok(());
+  }
+
+  // A command its caller has given up on writes nothing, not even its `ASKING`.
+  if client_utils::read_bool_atomic(&command.timed_out) {
+    _debug!(
+      inner,
+      "Ignore following ASK for timed out command: {}",
+      command.kind.to_str_debug()
+    );
+    return Ok(());
+  }
+
+  if router.connections.get_connection_mut(&server).is_none() {
+    let target = target_server(inner, router, server.clone(), slot);
+    _debug!(inner, "Connecting to ASK target {} for slot {}", target, slot);
+    // Send what any connection holds fed but unflushed first (through the bounded flush, all connections at once, so
+    // at most one bound however many hold fed frames). The connect holds the router task and polls no connection, and
+    // while this `Ask` was queued it counted as a command to follow, so the command written just ahead of it may be
+    // only fed; that frame would otherwise wait out the connect. The same entry point as before every reconnect and
+    // cluster-sync attempt (the liveness module's *Never waiting on unsent frames*); its clustered-client gate never
+    // skips it here, because `follow` runs only on a clustered client.
+    super::before_inline_connect(inner, router).await;
+    // The whole step — connect, TLS handshake and the connection's setup exchange — runs under one bound
+    // ([`ASK_TARGET_CONNECT_CAP`]). `add_connection` inserts the connection into the map only after both succeed
+    // (`src/router/connections.rs:381-383`), so a step cut short by the bound leaves nothing in the map; dropping the
+    // future drops the half-made socket.
+    let connected = client_utils::timeout(
+      router.connections.add_connection(inner, &target),
+      ask_target_connect_timeout(inner.connection_timeout()),
+    )
+    .await;
+    if let Err(err) = connected {
+      _debug!(inner, "Failed connecting to ASK target {}: {:?}", target, err);
+      command.respond_to_caller(Err(ask_connect_error(&target, &err)));
+      // On the client's error events, as the router reports a connection it lost (`src/router/responses.rs:311-319`),
+      // with the failure's own kind.
+      inner.notifications.broadcast_error(err, Some(target));
+      return Ok(());
+    }
+    // As a cluster sync does for a connection it adds (`src/router/clustered.rs:574`, `src/router/mod.rs:322`).
+    inner.notifications.broadcast_reconnect(target);
+    inner.backchannel.update_connection_ids(&router.connections);
+  }
+  // Send what any other connection holds fed but unflushed before this step's writes, all connections at once: the
+  // `ASKING` and the command can each wait on the target's socket for up to a bound, and a frame fed elsewhere would
+  // otherwise wait through both (the liveness module's *Never waiting on unsent frames*, before a write).
+  super::before_write_to(inner, router, &server).await;
+  let conn = match router.connections.get_connection_mut(&server) {
+    Some(conn) => conn,
+    None => {
+      command.respond_to_caller(Err(Error::new(
+        ErrorKind::IO,
+        format!("No connection to ASK target {server}."),
+      )));
+      return Ok(());
+    },
+  };
+
+  // `ASKING` takes no write attempt of the command's, and keeps none of its own: a replay of this connection's
+  // in-flight commands after it drops discards it instead of sending it to the slot's owner.
+  let mut asking = Command::new_asking(slot);
+  let _ = asking.decr_check_attempted();
+  let asking_failed = utils::write_command(inner, conn, asking, false).await.err();
+  if let Some((err, _)) = asking_failed {
+    _debug!(inner, "Error sending ASKING to {}: {:?}", server, err);
+    // The command was not written: it is retried or failed like any command whose write did not happen.
+    if command.attempts_remaining == 0 {
+      command.respond_to_caller(Err(err.clone()));
+    } else {
+      router.retry_command(command);
+    }
+    utils::drop_connection(inner, router, &server, &err).await;
+    return utils::defer_reconnection(inner, router, None, err, false);
+  }
+
+  // The command's own write attempt, as `write_command` takes it (`src/router/commands.rs:111`); the caller checked
+  // that one is left.
+  if let Err(err) = command.decr_check_attempted() {
+    command.respond_to_caller(Err(err));
+    return Ok(());
+  }
+  let is_blocking = command.blocks_connection();
+  // `write_command` skips a command whose timeout landed since the check above, leaving the `ASKING` for the next
+  // command on this connection (the record's change 8 lists what that next command can be).
+  match utils::write_command(inner, conn, command, false).await {
+    Ok(flushed) => {
+      _trace!(inner, "Sent command to ASK target {}. Flushed: {}", conn.server, flushed);
+      if is_blocking {
+        inner.backchannel.set_blocked(&conn.server);
+      }
+      if !flushed && inner.counters.read_cmd_buffer_len() == 0 {
+        let _ = conn.flush().await;
+      }
+      let should_interrupt =
+        is_blocking && inner.counters.read_cmd_buffer_len() > 0 && inner.config.blocking == Blocking::Interrupt;
+      if should_interrupt {
+        let _ = conn.flush().await;
+        _debug!(inner, "Interrupt after write.");
+        if let Err(e) = client_utils::interrupt_blocked_connection(inner, ClientUnblockFlag::Error).await {
+          _warn!(inner, "Failed to unblock connection: {:?}", e);
+        }
+      }
+      Ok(())
+    },
+    Err((err, _)) => {
+      // The command is in the connection's in-flight queue, behind its `ASKING`: dropping the connection retries or
+      // fails it there, as for any lost connection.
+      _debug!(inner, "Error sending command to ASK target {}: {:?}", server, err);
+      utils::drop_connection(inner, router, &server, &err).await;
+      utils::defer_reconnection(inner, router, None, err, false)
+    },
+  }
+}
+
+/// The `Server` to open a connection to for an `-ASK` target the client has no connection to.
+///
+/// A primary in the routing table is taken from there, which carries the TLS server name the cluster sync gave it. A
+/// node the table does not list (it owns no slot yet) gets the client's TLS host mapping applied as a sync would apply
+/// it, with the TLS server name of `slot`'s current owner as the default host; without one, the parsed `server` is used
+/// as it is.
+fn target_server(inner: &RefCount<ClientInner>, router: &Router, server: Server, slot: u16) -> Server {
+  let cache = match router.connections {
+    Connections::Clustered { ref cache, .. } => cache,
+    _ => return server,
+  };
+  if let Some(known) = cache.slots().iter().map(|range| &range.primary).find(|primary| **primary == server) {
+    return known.clone();
+  }
+
+  #[cfg(any(
+    feature = "enable-rustls",
+    feature = "enable-native-tls",
+    feature = "enable-rustls-ring"
+  ))]
+  {
+    let mut server = server;
+    if let (Some(tls), Some(owner_name)) = (
+      inner.config.tls.as_ref(),
+      cache.get_server(slot).and_then(|owner| owner.tls_server_name.as_ref()),
+    ) {
+      server.set_tls_server_name(&tls.hostnames, owner_name);
+    }
+    server
+  }
+  #[cfg(not(any(
+    feature = "enable-rustls",
+    feature = "enable-native-tls",
+    feature = "enable-rustls-ring"
+  )))]
+  {
+    let _ = (inner, slot);
+    server
+  }
+}
--- a/src/live_edit/liveness.rs
+++ b/src/live_edit/liveness.rs
@@ -0,0 +1,1741 @@
+// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+// SPDX-License-Identifier: Apache-2.0
+//
+// Licensed under the Apache License, Version 2.0 (the "License");
+// you may not use this file except in compliance with the License.
+// You may obtain a copy of the License at
+//
+//     http://www.apache.org/licenses/LICENSE-2.0
+//
+// Unless required by applicable law or agreed to in writing, software
+// distributed under the License is distributed on an "AS IS" BASIS,
+// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+// See the License for the specific language governing permissions and
+// limitations under the License.
+
+//! Connection liveness: how fred decides a pooled connection has gone silent, how long any router wait on one
+//! connection's I/O may last (closing a connection judged dead waits on nothing), that the router never waits idle,
+//! connects inline or waits on one connection's write while frames it has not sent sit on another, which replies count
+//! as pubsub frames, and what fred records when it closes a connection as unresponsive.
+//!
+//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
+//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
+//!
+//! # The unresponsive check
+//!
+//! Stock fred stamps `last_write` on every non-pubsub write, clears it whenever the read loop takes a frame, and
+//! closes a connection whose stamp is older than `UnresponsiveConfig::max_timeout`. That measures the *newest*
+//! write: a connection that keeps receiving writes never trips it, however long its oldest command has waited, and
+//! a reply read outside the read loop (a drain) leaves a stale stamp that can trip it on a healthy connection.
+//!
+//! This module replaces the rule. For each connection with at least one outstanding non-pubsub command that is not
+//! `blocked`, the *stall* is the time since the later of:
+//!
+//! - when the oldest outstanding non-pubsub command was written: handed to the socket by a send or flush that
+//!   completed (`Command::network_start`, re-stamped here when its frame is flushed, so a large frame's own upload
+//!   time does not count as silence) — or, while it is still only fed, the later of its prepare time and the
+//!   connection's last completed write, feed or flush, so a large frame being uploaded ahead of it is not silence
+//!   either; and
+//! - when the connection last made read progress: a frame decoded, or its read buffer grew.
+//!
+//! Two verdicts close the connection through fred's existing path; in-flight commands with attempts left are re-sent
+//! on the replacement connection, and a command with none fails with the verdict's error:
+//!
+//! - **fast**: the stall exceeds [`Settings::fast_timeout`] *and* some other connection in the process received a
+//!   reply from the same cluster slot range at least [`Settings::peer_evidence_lag`] after this connection's stall
+//!   began ([`PeerReplies`]). The shard answers others, so this socket is the dead one. A command is judged by its
+//!   slot's range only when the client's cluster routing gives that slot to the connection's node, whether the slot
+//!   comes from a key, a `ClusterHash::Custom` pin or a hashed argument: that slot's shard runs it. Any other — a
+//!   command with nothing to hash, one whose hashed slot another node owns (sent to a named node), or any sent
+//!   before the client has cluster routing — is judged at node level: any reply from the connection's node counts. A
+//!   centralized client's node owns every slot. Its error text is [`UNRESPONSIVE_FAST`].
+//! - **slow**: the stall exceeds `UnresponsiveConfig::max_timeout`, whatever peers are doing. This covers a shard
+//!   that is silent for everyone. Its error text is stock fred's, [`UNRESPONSIVE`].
+//!
+//! [`verdict_of`] tells the two apart from an error, so a caller never compares the texts itself.
+//!
+//! The check runs where stock fred's ran, in the read loop's `poll_connection`, which `read_or_write` re-polls at
+//! least every `UnresponsiveConfig::interval`. The peer evidence (a read of the slot range's reply times) is
+//! evaluated at most once per interval per connection, and only once a stall has passed the evidence lag.
+//!
+//! # Bounded waits
+//!
+//! Every wait on one connection's I/O outside the read loop (a write, a flush, a frame read by a drain, a broadcast
+//! reply read, an `ASKING` read, a transaction's `EXEC` and skip reads) is bounded by `max_timeout`, the slow bound.
+//! When the bound expires that connection is treated as failed: the wait returns an IO error with the [`UNRESPONSIVE`]
+//! text into the caller's existing error path, and the connection is marked so the read loop also reports it, which
+//! covers the callers that ignore a flush error. With `max_timeout: None` (fred's default) no wait is bounded.
+//!
+//! From then on every write, feed, flush or read on the marked connection fails at once with the same error, before
+//! any I/O and without a second log line, until fred closes it. The expiry already decided the connection is dead and
+//! nothing recovers it, so a second wait could only delay what each caller then does anyway, while holding the router
+//! and every command queued behind it: callers that clean up on the connection they just failed on (a transaction's
+//! `DISCARD`, a re-flush of frames left fed, a drain, an `ASKING` retry) would otherwise each wait up to another bound
+//! (the `DISCARD` up to the lesser of that and fred's `internal_command_timeout`). The marker is never cleared; it is
+//! dropped with the connection.
+//!
+//! The close is the last wait on a connection, on every path that closes one. Stock `Connection::close` first flushes
+//! fred's write buffer and shuts the transport down (under TLS, sending `close_notify` first), under fred's 5 s
+//! `CONNECTION_CLOSE_TIMEOUT_MS`. After an expired write or flush the bytes still buffered are, by construction, bytes
+//! the kernel would not take, so that close always waits out the 5 s; after a read-loop verdict it waits whenever the
+//! socket's send queue refuses what fred still holds, which on a fast verdict can push the re-send past the command
+//! timeout. So a connection judged dead — an expired wait, or a verdict the read loop reached (`closed_by`) — is
+//! closed without a flush or a shutdown ([`close`]): its in-flight commands are returned as before, and dropping the
+//! connection then closes the socket without blocking. So one silent connection holds the router for at most one bound
+//! per wait chain, or per attempt of a transaction, its close included. Every other close keeps fred's stock close.
+//! A frame fed on one connection does not wait through other connections' waits: before any of them, every fed frame
+//! is flushed, all connections at once (*Never waiting on unsent frames*, which lists the exceptions), so the frame
+//! reaches its socket within one bound, or its connection is failed, however many of the client's connections stall.
+//!
+//! # Never waiting on unsent frames
+//!
+//! fred writes a command without flushing it when its command-queue counter says another command follows, and
+//! relies on a later write to flush it. Two paths leave such a frame fed but never flushed even with a balanced
+//! counter (a timed-out command skipped at write flushes only its own connection; a skipped `Reconnect` returns
+//! without flushing), and a counter that drifted above the channel's length would hold every write back. And while a
+//! frame sits fed, the router can wait on other connections' bounded I/O, each wait up to the slow bound; waited one
+//! after another, k silent connections would hold the frame for k bounds.
+//!
+//! The rule, on a clustered client: the router never waits on one connection's I/O while another connection holds fed
+//! but unflushed frames, except inside a flush that runs every such connection's flush at once, each to its own end
+//! ([`flush_fed_concurrently`]). That flush holds the router for at most one bound however many connections stall in
+//! it, a fast error on one never cuts another's short, and it flushes only connections holding fed frames: with none
+//! it is one walk of the connections and with one a plain flush, so neither adds an allocation to what the flush
+//! itself does; two or more are joined (`join_all`), which allocates the join. A centralized or sentinel client has
+//! one connection, so no frame of it can wait behind another connection's I/O. So:
+//!
+//! - **before the router waits** ([`before_wait`], at the top of `read_or_write`): when its command channel is empty,
+//!   every connection holding fed but unflushed frames is flushed, all at once. The channel itself is checked, not the
+//!   counter: this is fred's own rule, "flush when no queued command follows", applied by construction;
+//! - **before every inline connect** of a clustered client (`live_edit::before_inline_connect`, called at the top of
+//!   every pass of `reconnect_with_policy` and `sync_cluster_with_policy`, before its backoff sleep and its attempt,
+//!   and by `live_edit::ask::follow`, before it opens a connection to an `-ASK` target): every connection holding
+//!   fed but unflushed frames is flushed whatever the channel holds, all at once. The router command that leads to
+//!   the connect (a `Reconnect`, `SyncCluster`, `Moved` or `Ask`) is itself queued and counted, so the command routed
+//!   just ahead of it is fed, not flushed, and the queued command keeps the channel non-empty, so [`before_wait`]
+//!   skips that frame. The connect then runs inline, polling no other connection, so neither a write nor a read-loop
+//!   verdict reaches a frame left fed on a healthy node's connection until the connect ends and a later flush sends
+//!   it (after a reconnect or sync, the retry buffer's). Flushing at the top of each pass, before the backoff sleep and
+//!   not only before the attempt, also lets the sleep's reads take that connection's reply, so no verdict closes a
+//!   healthy connection over a frame that was only unsent. The cost is at most one bound per pass or `-ASK` connect,
+//!   however many connections hold fed frames: the same wait fred's own router-wide flush after a successful attempt
+//!   would have spent, only earlier. A centralized or sentinel client's attempt closes its one connection first, and
+//!   that close flushes it (or, on one judged dead, drops it), so this flush is not run there;
+//! - **before a write that can reach a socket** ([`before_socket_write`], through `live_edit::before_socket_write` at
+//!   the top of the single-connection branch of the router's `write_command`, which every single command, every
+//!   pipeline command, the retry buffer's replay and a `-MOVED` retry's write go through): the command is routed to its
+//!   destination, and when the write can reach that connection's socket and another connection holds fed frames,
+//!   every connection holding them, the destination included, is flushed all at once before the write. The write can
+//!   reach the socket ([`write_reaches_socket`]) when fred's own flush rule would send it — the command-queue counter
+//!   at zero, the destination's feed count past `max_feed_count`, or a kind fred always sends (`should_flush`, with
+//!   custom commands and scripts among them) — or when it is a feed onto a write buffer at tokio-util's
+//!   8 KiB backpressure boundary, which flushes that buffer first. A feed below the boundary only encodes into the
+//!   buffer and cannot block, so it triggers nothing, which keeps fred's batching across connections below the
+//!   boundary; and when only the destination holds fed frames nothing is flushed first, so its own write batches them
+//!   as stock does. This patch's rare writes outside `write_command` (`ask::follow`'s, `abandon::after_pipeline`'s)
+//!   flush whenever another connection holds fed frames, without the predicate ([`before_write_to`]). Per command the
+//!   check costs one more `Router::route` and a few atomic loads, plus a walk of the connections when the write can
+//!   reach a socket. It flushes only connections holding fed frames, allocating only when two or more do. For the
+//!   counter and command-kind terms the write stays a send, and the flush sends only what fred's router-wide flush
+//!   right after it would have sent; that flush then finds nothing left and allocates nothing, where stock's allocated
+//!   on every flushed write. For the feed-count term the flush also sends the destination's fed frames and resets its
+//!   feed count, so every fed frame leaves as stock's send and router-wide flush would have sent it, but the write
+//!   itself is then only fed, and leaves at the next flush point. For a feed at the 8 KiB boundary, which stock follows
+//!   with no router-wide flush, it sends other connections' fed frames earlier than stock's next flush point (the next
+//!   send, or `before_wait`) would. The router also waits earlier: a silent connection holding fed frames is waited
+//!   on, up to one bound, before a write elsewhere, where that write could once fail first and a verdict then drop it
+//!   unwaited, so a multi-node outage can recover up to one bound later; the hold stays at most one bound per silent
+//!   connection, as an expiry fails later I/O on it at once. A healthy connection whose one outstanding command is fed
+//!   now sends it first, so is not judged silent;
+//! - **in fred's router-wide flush** (`Connections::flush`, through `live_edit::flush_router_wide`: after a successful
+//!   send, after the retry buffer's replay, and at the start of `drain_all`): only connections holding fed frames are
+//!   flushed, all at once and each to its own end, and the first error is returned, which the callers log or return
+//!   as before. Stock's `try_join_all` stopped at the first error, dropping the other flushes; so now a `drain_all`
+//!   whose flush meets a fast error waits for the others, up to one bound, before it returns that error;
+//! - **at each receive** ([`on_receive`]): the counter is set to the channel's length plus one (the command just
+//!   received, which `process_command` then subtracts), so a drift lasts at most until the next command. Every
+//!   sender increments the counter before it sends (`send_to_router`), and the router reads the channel's length
+//!   before the counter, so a send in flight at that moment shows as a counter one above the channel; the reset then
+//!   drops that sender's increment, so once its command lands the counter reads one below, which the next receive
+//!   resets, whichever command it takes. Commands that leave the channel without being counted down show the same
+//!   way: fred's `drain_command_rx` cancels the commands still queued when a router ends without decrementing, and
+//!   the same channel goes back to the next router, whose first receive finds one above per cancelled command. A
+//!   difference of one either way is usually that race, or a single command cancelled when a router ended, and logs
+//!   at DEBUG; a larger one (several sends in flight at once, or several commands cancelled) logs one WARN per client
+//!   per [`DRIFT_LOG_INTERVAL`] with its size. Either way the reset has already corrected it.
+//! - the counter's decrement saturates at zero ([`saturating_decr`]) instead of wrapping.
+//!
+//! Not covered, each a wait on one connection that can still run while another holds fed frames:
+//!
+//! - the flush before a `CLIENT UNBLOCK` under `Blocking::Interrupt` (`write_command`, and its copy in
+//!   `live_edit::ask::follow`), after a blocking command is written. Only the service's stream readers set
+//!   `Blocking::Interrupt`, and they write nothing to the keyspace;
+//! - closes that keep fred's stock close, each bounded by its 5 s close timeout. Its flush can wait only on a peer that
+//!   half-closed and stopped reading; an errored or closed socket does not hold it. `drop_connection`'s close, as after
+//!   a read error or the end of the stream, is one such wait. `disconnect_all` closes a client's connections one after
+//!   another, so a frame fed on one late in that walk can wait up to 5 s per such peer; only a router's end reaches it
+//!   with fed frames, since a failed cluster-wide command runs `drain_all` first;
+//! - a `Moved`'s wait of `cluster_cache_update_delay` (`delay_cluster_sync`; fred's default 0, unset by the service);
+//! - the `replicas` feature (the replica loops and routing), which the service does not build: [`before_socket_write`]
+//!   skips a command routed to a replica, because routing it twice would advance the replica round-robin.
+//!
+//! # Pubsub classification
+//!
+//! Stock fred classifies any reply shaped like a pubsub frame (a three-element array starting `"message"`, a three-
+//! or four-element array starting `"subscribe"`, ...) as pubsub on every connection and drops it, so a data reply of
+//! that shape is never matched to its command and every later caller on the connection gets the previous caller's
+//! reply. Here a connection classifies non-`Push` frames as pubsub only once it has sent a subscription command
+//! (`subscribe`, `unsubscribe` and their pattern and shard forms); every other connection matches them to their
+//! commands, and counts them. `Push` frames, which are pushes by construction, keep the stock handling.
+//!
+//! # What a verdict records
+//!
+//! Each fast or slow verdict, and each expired bounded wait, logs one WARN (target `fred::live_edit::liveness`)
+//! starting `Unresponsive connection to <node>`, with these fields:
+//!
+//! - **the event**: the client id (the line's prefix), the node, the verdict (`fast`, `slow`, or `expired <write|
+//!   flush|read>`), the stall, and the age of the newest reply from the same slot range on another connection (from
+//!   the connection's node, for a command judged at node level);
+//! - **this connection**: the in-flight count (replies owed); the oldest outstanding command's kind, whether its
+//!   frame reached the socket (`sent`, with the age since it did) or is still only in fred's write buffer (`fed, not
+//!   yet sent`, with the age since fred prepared it); the commands fed since the last completed send or flush
+//!   (`unsent`), the write buffer's bytes and fred's feed count; the read buffer's bytes and the time since the last
+//!   read progress; whether the connection has sent a subscription command, and how many pubsub-shaped frames it has
+//!   received in its life (on one that never subscribed, frames stock fred would have dropped); the longest single
+//!   write, flush or read wait on it since the stall began; and the longest time a command sent since the stall began
+//!   sat fed before it was sent (the send re-stamps `network_start`, so "sent X ago" alone would hide a frame that
+//!   sat unsent);
+//! - **this client**: the command-queue counter, the command channel's length when the router last waited, the time
+//!   since the router last took a command, and one summary per other connection of the client (node, in flight,
+//!   oldest outstanding age, unsent, write-buffer bytes, longest wait). An expired wait runs outside the read loop,
+//!   where the other connections are not reachable: its line says so;
+//! - **the kernel's view of the socket**, from the `/proc/self/net/tcp` (or `tcp6`) row of the connection's local and
+//!   remote addresses: `tx_queue` (bytes the kernel holds unsent or unacknowledged), `rx_queue` (bytes received, not
+//!   yet read), the active timer (`retransmit`, `zero-window probe`, `keepalive`) and `retrnsmt` (consecutive
+//!   unrecovered retransmit timeouts). Read only at a verdict, with std file I/O; `unavailable` when there is no such
+//!   file or no matching row (not Linux, a unix socket).
+//!
+//! Where the stuck bytes sit names the cause: in fred's write buffer (`unsent` or write-buffer bytes above zero), a
+//! client flush problem; a counter that differs from the channel length, a counter leak; a kernel `tx_queue` above
+//! zero with a retransmit timer or `retrnsmt` above zero and fred's buffers empty, the network path; both kernel
+//! queues empty and no reply, the server or a proxy; an `rx_queue` above zero, the client's read side; a long wait on
+//! another connection, the client's shared router; a pubsub-shaped count above zero, the RESP2 shape (handled).
+
+use crate::{
+  error::{Error, ErrorKind},
+  modules::inner::{ClientInner, CommandReceiver},
+  protocol::{
+    command::Command,
+    connection::{Connection, ConnectionKind},
+    types::{ClusterRouting, ProtocolFrame, Server},
+  },
+  router::{connections::Connections, Router},
+  runtime::RefCount,
+  utils as client_utils,
+};
+use futures::{SinkExt, StreamExt};
+use redis_protocol::{
+  resp2::types::BytesFrame as Resp2Frame,
+  resp3::types::{BytesFrame as Resp3Frame, Resp3Frame as _Resp3Frame},
+};
+use std::{
+  collections::HashMap,
+  fmt,
+  fmt::Write as _,
+  future::Future,
+  net::SocketAddr,
+  sync::{
+    atomic::{AtomicU64, AtomicUsize, Ordering},
+    Arc,
+    Mutex,
+  },
+  time::{Duration, Instant},
+};
+
+/// The error text of a slow verdict and of every expired bounded wait: stock fred's text, unchanged, so callers that
+/// match on it keep working.
+pub const UNRESPONSIVE: &str = "Unresponsive connection.";
+
+/// The error text of a fast verdict: the connection was silent while its shard answered other connections.
+pub const UNRESPONSIVE_FAST: &str = "Unresponsive connection (fast verdict).";
+
+/// Which unresponsive verdict closed a connection ([`verdict_of`]).
+#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
+pub enum Verdict {
+  /// The fast verdict ([`UNRESPONSIVE_FAST`]): this connection was silent while the same slot range answered others.
+  Fast,
+  /// The slow verdict or an expired bounded wait ([`UNRESPONSIVE`]): no progress for the slow bound, whatever peers
+  /// were doing.
+  Slow,
+}
+
+impl Verdict {
+  /// `fast` or `slow`.
+  pub fn as_str(self) -> &'static str {
+    match self {
+      Verdict::Fast => "fast",
+      Verdict::Slow => "slow",
+    }
+  }
+}
+
+/// Which unresponsive verdict `error` reports, if any: an IO error with the [`UNRESPONSIVE_FAST`] text is a fast
+/// verdict, one with the [`UNRESPONSIVE`] text a slow verdict or an expired bounded wait (the two share stock fred's
+/// text). Every other error is `None`. Classify through this rather than by comparing texts.
+pub fn verdict_of(error: &Error) -> Option<Verdict> {
+  if *error.kind() != ErrorKind::IO {
+    return None;
+  }
+  match error.details() {
+    UNRESPONSIVE_FAST => Some(Verdict::Fast),
+    UNRESPONSIVE => Some(Verdict::Slow),
+    _ => None,
+  }
+}
+
+fn verdict_error(verdict: Verdict) -> Error {
+  Error::new(ErrorKind::IO, match verdict {
+    Verdict::Fast => UNRESPONSIVE_FAST,
+    Verdict::Slow => UNRESPONSIVE,
+  })
+}
+
+/// The number of cluster hash slots.
+const SLOTS: usize = 16_384;
+
+/// The minimum spacing of near-miss INFO lines per [`PeerReplies`], unless a near miss is longer than every one
+/// logged in the current window (so the longest is never suppressed).
+const NEAR_MISS_LOG_INTERVAL: Duration = Duration::from_secs(10);
+
+/// The minimum spacing of command-queue counter drift WARN lines per client ([`on_receive`]).
+pub const DRIFT_LOG_INTERVAL: Duration = Duration::from_secs(60);
+
+/// Liveness settings, on `ConnectionConfig::live_edit.liveness`. The default is stock fred's rule set apart from
+/// the fixes this module always applies: no fast verdict and no peer registry.
+#[derive(Clone, Debug, PartialEq, Eq)]
+pub struct Settings {
+  /// The fast verdict's stall bound. `None` disables the fast verdict: only the slow bound
+  /// (`UnresponsiveConfig::max_timeout`) applies.
+  ///
+  /// Default: `None`
+  pub fast_timeout:      Option<Duration>,
+  /// How long after a connection's stall began a peer's reply from the same slot range must arrive to count as
+  /// evidence that the shard is answering.
+  ///
+  /// Default: 1 sec
+  pub peer_evidence_lag: Duration,
+  /// The registry of reply times shared by every client of one service. Every reply a connection receives stamps
+  /// it; the fast verdict reads it. `None` stamps nothing and disables the fast verdict.
+  ///
+  /// Default: `None`
+  pub peer_replies:      Option<PeerReplies>,
+}
+
+impl Default for Settings {
+  fn default() -> Self {
+    Settings {
+      fast_timeout:      None,
+      peer_evidence_lag: Duration::from_secs(1),
+      peer_replies:      None,
+    }
+  }
+}
+
+/// When each cluster hash slot, and each node for commands judged at node level, last answered a command on any
+/// connection that shares this registry. Cloning shares the registry; it is dropped with its last clone.
+///
+/// Stamping a reply is a read of the client's cluster routing (to place the slot on the connection's node) and one
+/// relaxed atomic store; a stalled connection reads its slot range only when it judges a stall.
+#[derive(Clone)]
+pub struct PeerReplies {
+  shared: Arc<Registry>,
+}
+
+struct Registry {
+  epoch:       Instant,
+  /// Per slot: nanoseconds since `epoch`, plus one, of the newest reply; `0` = none yet.
+  slots:       Box<[AtomicU64]>,
+  /// Per node, for replies to commands judged at node level, encoded as `slots`.
+  nodes:       Mutex<HashMap<Server, Arc<AtomicU64>>>,
+  near_misses: Mutex<NearMisses>,
+}
+
+#[derive(Default)]
+struct NearMisses {
+  window_start: Option<Instant>,
+  longest:      Duration,
+  suppressed:   u64,
+}
+
+impl PeerReplies {
+  /// A new, empty registry.
+  pub fn new() -> Self {
+    PeerReplies {
+      shared: Arc::new(Registry {
+        epoch:       Instant::now(),
+        slots:       (0 .. SLOTS).map(|_| AtomicU64::new(0)).collect(),
+        nodes:       Mutex::new(HashMap::new()),
+        near_misses: Mutex::new(NearMisses::default()),
+      }),
+    }
+  }
+
+  /// Whether `self` and `other` are the same registry.
+  pub fn same_as(&self, other: &PeerReplies) -> bool {
+    Arc::ptr_eq(&self.shared, &other.shared)
+  }
+
+  fn encode(&self, at: Instant) -> u64 {
+    (at.saturating_duration_since(self.shared.epoch).as_nanos() as u64).saturating_add(1)
+  }
+
+  fn decode(&self, value: u64) -> Option<Instant> {
+    value
+      .checked_sub(1)
+      .map(|nanos| self.shared.epoch + Duration::from_nanos(nanos))
+  }
+
+  fn node(&self, server: &Server) -> Arc<AtomicU64> {
+    let mut nodes = self.shared.nodes.lock().unwrap_or_else(|e| e.into_inner());
+    nodes.entry(server.clone()).or_default().clone()
+  }
+
+  fn stamp(&self, cell: &AtomicU64, at: Instant) {
+    cell.store(self.encode(at), Ordering::Relaxed);
+  }
+
+  /// The newest reply time over the inclusive slot `ranges` and, when given, `node`.
+  fn newest(&self, ranges: &[(u16, u16)], node: Option<&AtomicU64>) -> Option<Instant> {
+    let mut newest = node.map(|n| n.load(Ordering::Relaxed)).unwrap_or(0);
+    for &(start, end) in ranges {
+      let (start, end) = (start as usize, (end as usize).min(SLOTS - 1));
+      if start > end {
+        continue;
+      }
+      for cell in self.shared.slots[start ..= end].iter() {
+        newest = newest.max(cell.load(Ordering::Relaxed));
+      }
+    }
+    self.decode(newest)
+  }
+}
+
+impl Default for PeerReplies {
+  fn default() -> Self {
+    PeerReplies::new()
+  }
+}
+
+impl fmt::Debug for PeerReplies {
+  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
+    f.debug_struct("PeerReplies").finish_non_exhaustive()
+  }
+}
+
+impl PartialEq for PeerReplies {
+  fn eq(&self, other: &Self) -> bool {
+    self.same_as(other)
+  }
+}
+
+impl Eq for PeerReplies {}
+
+/// Per-client liveness state, on `ClientInner::live_edit`, shared with every connection of the client. Written only
+/// by the client's router task; read when a verdict is logged.
+#[derive(Clone, Debug)]
+pub struct ClientState {
+  shared: Arc<ClientShared>,
+}
+
+#[derive(Debug)]
+struct ClientShared {
+  epoch:        Instant,
+  /// When the router last took a command off its channel: nanoseconds since `epoch`, plus one; `0` = never.
+  last_command: AtomicU64,
+  /// The command channel's length when the router last waited ([`before_wait`]).
+  queued:       AtomicUsize,
+  drift:        Mutex<DriftLog>,
+}
+
+#[derive(Debug, Default)]
+struct DriftLog {
+  logged:     Option<Instant>,
+  suppressed: u64,
+}
+
+impl Default for ClientState {
+  fn default() -> Self {
+    ClientState {
+      shared: Arc::new(ClientShared {
+        epoch:        Instant::now(),
+        last_command: AtomicU64::new(0),
+        queued:       AtomicUsize::new(0),
+        drift:        Mutex::new(DriftLog::default()),
+      }),
+    }
+  }
+}
+
+impl ClientState {
+  fn mark_command(&self, now: Instant) {
+    let nanos = now.saturating_duration_since(self.shared.epoch).as_nanos() as u64;
+    self.shared.last_command.store(nanos.saturating_add(1), Ordering::Relaxed);
+  }
+
+  fn last_command(&self) -> Option<Instant> {
+    self
+      .shared
+      .last_command
+      .load(Ordering::Relaxed)
+      .checked_sub(1)
+      .map(|nanos| self.shared.epoch + Duration::from_nanos(nanos))
+  }
+}
+
+/// Per-connection liveness state. Created with the connection (`crate::live_edit::pipelined`), dropped with it; a
+/// reconnect starts fresh.
+#[derive(Debug, Default)]
+pub struct ConnectionState {
+  /// The bound on every wait on this connection's I/O outside the read loop: the client's
+  /// `UnresponsiveConfig::max_timeout` when the connection was created.
+  wait_bound:       Option<Duration>,
+  /// Whether this connection has sent a subscription command, so pubsub-shaped frames on it are pubsub.
+  pub(crate) subscribed: bool,
+  /// How many pubsub-shaped frames this connection has received in its life (on one that never subscribed: frames it
+  /// matched to their commands, which stock fred would have dropped). Shared with the frame classification, which
+  /// runs where only the router is at hand.
+  pubsub_shaped:    Arc<AtomicU64>,
+  /// The last read progress: a frame decoded or the read buffer grew.
+  last_progress:    Option<Instant>,
+  /// The read buffer's length at the last observation.
+  read_len:         usize,
+  /// Tracked commands written since the last completed send or flush.
+  unsent:           usize,
+  /// Which bounded wait expired, once one has: the connection is failed, every later write, feed, flush or read on it
+  /// fails at once, and its close does not flush ([`close`]). Never cleared.
+  expired:          Option<&'static str>,
+  /// The read-loop verdict that is closing this connection: set where the read loop reaches a fast or slow verdict
+  /// ([`verdict`]), so the close that follows does not flush ([`close`]). Never cleared; dropped with the connection.
+  /// Unlike `expired` it fails no later I/O: the router closes the connection in the step that reached the verdict
+  /// ([`defer`]).
+  closed_by:        Option<Verdict>,
+  /// The current stall, while the connection has an outstanding command.
+  episode:          Option<Episode>,
+  /// When the peer evidence for the current stall was last read.
+  evidence_checked: Option<Instant>,
+  /// This connection's node cell in the peer registry, for replies to commands judged at node level.
+  node_stamp:       Option<Arc<AtomicU64>>,
+  /// The longest single write, flush or read wait on this connection that ended since its current stall began.
+  longest_wait:     Option<Wait>,
+  /// The longest time a tracked command sat fed (prepared, in fred's write buffer) before a send or flush put it on
+  /// the socket, among the commands sent since the current stall began. [`on_sent`] re-stamps a command's
+  /// `network_start` when it is sent, so this is the only record of that delay.
+  longest_fed:      Option<Wait>,
+  /// When the connection's last write, feed or flush completed: a fed command's frame cannot have waited on the
+  /// socket for longer than since then ([`stall_start`]).
+  last_write:       Option<Instant>,
+  /// A verdict reached in the read loop, logged once every connection of the client has been polled.
+  pending:          Option<Pending>,
+  /// The owning client, set when the connection is created, for a log line written where only the connection is at
+  /// hand (an expired wait). The router task owns both, so this makes no cycle.
+  inner:            Option<ClientHandle>,
+}
+
+/// The owning client of a connection ([`ConnectionState`]); `Debug` prints its id.
+struct ClientHandle(RefCount<ClientInner>);
+
+impl fmt::Debug for ClientHandle {
+  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
+    f.write_str(&self.0.id)
+  }
+}
+
+#[derive(Debug, Clone, Copy)]
+struct Episode {
+  /// When the stall began.
+  since:    Instant,
+  /// Whether a peer answered the same slot range at least the evidence lag after `since`.
+  evidence: bool,
+}
+
+/// One completed wait on a connection's I/O.
+#[derive(Debug, Clone, Copy)]
+struct Wait {
+  what:  &'static str,
+  took:  Duration,
+  ended: Instant,
+}
+
+#[derive(Debug, Clone, Copy)]
+struct Pending {
+  verdict: Verdict,
+  stall:   Duration,
+  since:   Instant,
+}
+
+/// How a frame read from one connection is classified: its `subscribed` flag, and the connection's pubsub-shaped
+/// frame count (absent when the connection is gone).
+#[derive(Debug, Clone)]
+pub(crate) struct Classify {
+  subscribed: bool,
+  shaped:     Option<Arc<AtomicU64>>,
+}
+
+/// Initialize a new pipelined connection's state from its client's settings.
+pub(crate) fn attach(inner: &RefCount<ClientInner>, conn: &mut Connection) {
+  conn.live_edit.liveness.wait_bound = inner.connection.unresponsive.max_timeout;
+  conn.live_edit.liveness.inner = Some(ClientHandle(inner.clone()));
+}
+
+macro_rules! per_transport {
+  ($transport:expr, $framed:ident => $body:expr) => {
+    match $transport {
+      ConnectionKind::Tcp(ref t) => {
+        let $framed = t.get_ref();
+        $body
+      },
+      #[cfg(feature = "unix-sockets")]
+      ConnectionKind::Unix(ref t) => {
+        let $framed = t.get_ref();
+        $body
+      },
+      #[cfg(any(feature = "enable-rustls", feature = "enable-rustls-ring"))]
+      ConnectionKind::Rustls(ref t) => {
+        let $framed = t.get_ref();
+        $body
+      },
+      #[cfg(feature = "enable-native-tls")]
+      ConnectionKind::NativeTls(ref t) => {
+        let $framed = t.get_ref();
+        $body
+      },
+    }
+  };
+}
+
+fn read_buffer_len(transport: &ConnectionKind) -> usize {
+  per_transport!(transport, framed => framed.read_buffer().len())
+}
+
+fn write_buffer_len(transport: &ConnectionKind) -> usize {
+  per_transport!(transport, framed => framed.write_buffer().len())
+}
+
+/// The owning client's id (`fred-...`), which the codec carries.
+fn client_name(transport: &ConnectionKind) -> &str {
+  per_transport!(transport, framed => &*framed.codec().name)
+}
+
+/// The TCP socket's local and remote addresses, when the transport is TCP (plain or rustls).
+fn socket_addrs(transport: &ConnectionKind) -> Option<(SocketAddr, SocketAddr)> {
+  let tcp = match transport {
+    ConnectionKind::Tcp(ref t) => t.get_ref().get_ref(),
+    #[cfg(any(feature = "enable-rustls", feature = "enable-rustls-ring"))]
+    ConnectionKind::Rustls(ref t) => t.get_ref().get_ref().get_ref().0,
+    #[allow(unreachable_patterns)]
+    _ => return None,
+  };
+  Some((tcp.local_addr().ok()?, tcp.peer_addr().ok()?))
+}
+
+fn on_progress(conn: &mut Connection, now: Instant) {
+  let read_len = read_buffer_len(&conn.transport);
+  let state = &mut conn.live_edit.liveness;
+  state.last_progress = Some(now);
+  state.read_len = read_len;
+}
+
+/// When `conn` last made read progress (a frame taken off it, or its read buffer grew), if it has.
+pub(crate) fn last_progress(conn: &Connection) -> Option<Instant> {
+  conn.live_edit.liveness.last_progress
+}
+
+/// The read loop took a frame (or the end of the stream) off the connection.
+pub(crate) fn on_frame(conn: &mut Connection, now: Instant) {
+  on_progress(conn, now);
+}
+
+/// Count a tracked write (`write`'s `check_unresponsive`); it is re-stamped when its frame is flushed.
+pub(crate) fn count_unsent(conn: &mut Connection) {
+  conn.live_edit.liveness.unsent += 1;
+}
+
+/// A send or flush completed: every tracked command written since the previous one is now on the socket, so its
+/// wait starts now.
+///
+/// Before re-stamping, each command's fed-to-sent delay (from its prepare-time `network_start`) is kept when it is the
+/// longest since the current stall began ([`ConnectionState`]'s `longest_fed`).
+fn on_sent(conn: &mut Connection, now: Instant) {
+  let mut left = std::mem::take(&mut conn.live_edit.liveness.unsent);
+  let mut longest = Duration::ZERO;
+  for command in conn.buffer.iter_mut().rev() {
+    if left == 0 {
+      break;
+    }
+    if command.kind.is_pubsub() {
+      continue;
+    }
+    if let Some(prepared) = command.network_start {
+      longest = longest.max(now.saturating_duration_since(prepared));
+    }
+    command.network_start = Some(now);
+    left -= 1;
+  }
+  let state = &mut conn.live_edit.liveness;
+  let since = state.episode.map(|episode| episode.since);
+  let fed = Wait {
+    what:  "fed",
+    took:  longest,
+    ended: now,
+  };
+  state.longest_fed = match state.longest_fed {
+    Some(kept) if since.map_or(true, |since| kept.ended >= since) && kept.took >= fed.took => Some(kept),
+    _ => Some(fed),
+  };
+}
+
+/// Record one completed wait on the connection's I/O, keeping the longest that ended since the current stall began.
+fn record_wait(conn: &mut Connection, what: &'static str, started: Instant) {
+  let ended = Instant::now();
+  let wait = Wait {
+    what,
+    took: ended.saturating_duration_since(started),
+    ended,
+  };
+  let state = &mut conn.live_edit.liveness;
+  if what != "read" {
+    state.last_write = Some(ended);
+  }
+  let since = state.episode.map(|episode| episode.since);
+  state.longest_wait = match state.longest_wait {
+    Some(kept) if since.map_or(true, |since| kept.ended >= since) && kept.took >= wait.took => Some(kept),
+    _ => Some(wait),
+  };
+}
+
+/// Mark the connection failed after a bounded wait expired, log it, and return the error for the caller's path.
+fn expire(conn: &mut Connection, what: &'static str, started: Instant) -> Error {
+  record_wait(conn, what, started);
+  conn.live_edit.liveness.expired = Some(what);
+  if log::log_enabled!(log::Level::Warn) {
+    let now = Instant::now();
+    let bound = conn.live_edit.liveness.wait_bound.unwrap_or_default();
+    let newest_peer = match conn.live_edit.liveness.inner.as_ref() {
+      Some(ClientHandle(inner)) => describe_newest_peer(inner, conn, now),
+      None => "unknown".to_owned(),
+    };
+    let mut line = format!(
+      "{}: Unresponsive connection to {} (expired {}): a {} on it made no progress for {:?}; failing the \
+       connection. Newest reply from the same slot range on another connection: {}. ",
+      client_name(&conn.transport),
+      conn.server,
+      what,
+      what,
+      bound,
+      newest_peer,
+    );
+    describe_connection(&mut line, conn, now, Some(bound));
+    describe_client(&mut line, conn, now);
+    line.push_str("Other connections: not captured (the wait ran outside the read loop). ");
+    describe_kernel(&mut line, conn);
+    log::warn!("{}", line);
+  }
+  verdict_error(Verdict::Slow)
+}
+
+async fn within<T, F>(bound: Option<Duration>, ft: F) -> Option<T>
+where
+  F: Future<Output = T>,
+{
+  match bound {
+    Some(bound) => tokio::time::timeout(bound, ft).await.ok(),
+    None => Some(ft.await),
+  }
+}
+
+/// `SinkExt::send` (write and flush), bounded. Fails at once, without writing, on a connection whose bounded wait
+/// already expired (see the module's *Bounded waits*).
+pub(crate) async fn send(conn: &mut Connection, frame: ProtocolFrame) -> Result<(), Error> {
+  if conn.live_edit.liveness.expired.is_some() {
+    return Err(verdict_error(Verdict::Slow));
+  }
+  let bound = conn.live_edit.liveness.wait_bound;
+  let started = Instant::now();
+  match within(bound, conn.transport.send(frame)).await {
+    Some(Ok(())) => {
+      record_wait(conn, "write", started);
+      on_sent(conn, Instant::now());
+      Ok(())
+    },
+    Some(Err(e)) => {
+      record_wait(conn, "write", started);
+      Err(e)
+    },
+    None => Err(expire(conn, "write", started)),
+  }
+}
+
+/// `SinkExt::feed` (buffer, flushing only past the sink's backpressure boundary), bounded. Fails at once, without
+/// buffering, on a connection whose bounded wait already expired.
+pub(crate) async fn feed(conn: &mut Connection, frame: ProtocolFrame) -> Result<(), Error> {
+  if conn.live_edit.liveness.expired.is_some() {
+    return Err(verdict_error(Verdict::Slow));
+  }
+  let bound = conn.live_edit.liveness.wait_bound;
+  let started = Instant::now();
+  match within(bound, conn.transport.feed(frame)).await {
+    Some(result) => {
+      record_wait(conn, "write", started);
+      result
+    },
+    None => Err(expire(conn, "write", started)),
+  }
+}
+
+/// `SinkExt::flush`, bounded. Fails at once, without flushing, on a connection whose bounded wait already expired,
+/// even when it has nothing to flush.
+pub(crate) async fn flush(conn: &mut Connection) -> Result<(), Error> {
+  if conn.live_edit.liveness.expired.is_some() {
+    return Err(verdict_error(Verdict::Slow));
+  }
+  let bound = conn.live_edit.liveness.wait_bound;
+  let started = Instant::now();
+  match within(bound, conn.transport.flush()).await {
+    Some(Ok(())) => {
+      record_wait(conn, "flush", started);
+      on_sent(conn, Instant::now());
+      Ok(())
+    },
+    Some(Err(e)) => {
+      record_wait(conn, "flush", started);
+      Err(e)
+    },
+    None => Err(expire(conn, "flush", started)),
+  }
+}
+
+/// `StreamExt::next` (one frame), bounded; the outer error is an expired bound, this one's or, failing at once
+/// without reading, an earlier one on the same connection.
+pub(crate) async fn next(conn: &mut Connection) -> Result<Option<Result<ProtocolFrame, Error>>, Error> {
+  if conn.live_edit.liveness.expired.is_some() {
+    return Err(verdict_error(Verdict::Slow));
+  }
+  let bound = conn.live_edit.liveness.wait_bound;
+  let started = Instant::now();
+  match within(bound, conn.transport.next()).await {
+    Some(item) => {
+      record_wait(conn, "read", started);
+      on_progress(conn, Instant::now());
+      Ok(item)
+    },
+    None => Err(expire(conn, "read", started)),
+  }
+}
+
+/// `Connection::close`'s transport close (`SinkExt::close`: flush fred's write buffer, then shut the transport down,
+/// under TLS after queueing `close_notify`), which `Connection::close` bounds by `CONNECTION_CLOSE_TIMEOUT_MS`.
+///
+/// On a connection judged dead — a bounded wait expired (`expired`), or the read loop reached a verdict on it
+/// (`closed_by`) — returns at once instead: no flush, no `close_notify`, no shutdown, no log and no metric (see the
+/// module's *Bounded waits*). The peer is not reading or not acknowledging, so the close could only wait; and were it
+/// only slow, flushing would hand it frames that fred re-sends or fails anyway. The caller still drains and returns
+/// the in-flight commands, and dropping the connection afterwards closes the socket without blocking.
+pub(crate) async fn close(conn: &mut Connection) -> Result<(), Error> {
+  let state = &conn.live_edit.liveness;
+  if state.expired.is_some() || state.closed_by.is_some() {
+    return Ok(());
+  }
+  conn.transport.close().await
+}
+
+/// Whether `conn` holds frames fred fed but has not flushed.
+fn has_unflushed(conn: &Connection) -> bool {
+  client_utils::read_atomic(&conn.counters.feed_count) > 0 || write_buffer_len(&conn.transport) > 0
+}
+
+/// Every connection of the router, primaries and (with the `replicas` feature) replicas: the primaries' (a
+/// centralized or sentinel client's one connection, or the cluster map's), then the replicas'. An iterator over the
+/// router's own collections, so walking it allocates nothing; the connection buffer sweep walks it twice on every
+/// router pass. Two walks with no connection added or removed between them visit the connections in the same order.
+pub(crate) fn connections_mut(router: &mut Router) -> impl Iterator<Item = &mut Connection> {
+  let primaries = match router.connections {
+    Connections::Centralized { ref mut connection } | Connections::Sentinel { ref mut connection } => {
+      Primaries::One(connection.iter_mut())
+    },
+    Connections::Clustered {
+      ref mut connections, ..
+    } => Primaries::Map(connections.values_mut()),
+  };
+  #[cfg(feature = "replicas")]
+  let primaries = primaries.chain(router.replicas.connections.values_mut());
+  primaries
+}
+
+/// The primaries' connections, for [`connections_mut`]: the one connection of a centralized or sentinel client, or a
+/// cluster client's connection map.
+enum Primaries<'a> {
+  One(std::option::IterMut<'a, Connection>),
+  Map(std::collections::hash_map::ValuesMut<'a, Server, Connection>),
+}
+
+impl<'a> Iterator for Primaries<'a> {
+  type Item = &'a mut Connection;
+
+  fn next(&mut self) -> Option<Self::Item> {
+    match self {
+      Primaries::One(connection) => connection.next(),
+      Primaries::Map(connections) => connections.next(),
+    }
+  }
+
+  fn size_hint(&self) -> (usize, Option<usize>) {
+    match self {
+      Primaries::One(connection) => connection.size_hint(),
+      Primaries::Map(connections) => connections.size_hint(),
+    }
+  }
+}
+
+/// The top of the router's `read_or_write`: record the command channel's length, and when it is empty — no queued
+/// command will flush what an earlier write fed — flush every connection holding fed but unflushed frames, all of
+/// them at once ([`flush_fed_concurrently`]). The channel is checked, not the counter: fred's own rule, "flush when no
+/// queued command follows", applied by construction.
+pub(crate) async fn before_wait(inner: &RefCount<ClientInner>, router: &mut Router, rx: &CommandReceiver) {
+  let queued = rx.len();
+  inner.live_edit.liveness.shared.queued.store(queued, Ordering::Relaxed);
+  if queued > 0 {
+    return;
+  }
+  let _ = flush_fed_concurrently(connections_mut(router), FlushSite::Wait(inner)).await;
+}
+
+/// Before the router writes `command` to the one connection it routes to, on a clustered client
+/// (`live_edit::before_socket_write`, the top of the single-connection branch of the router's `write_command`): when
+/// that write can reach the connection's socket ([`write_reaches_socket`]) and another connection holds fed but
+/// unflushed frames, flush every connection holding them, the destination included, all at once
+/// ([`flush_fed_concurrently`]). The write can then wait on its own connection's socket for up to a bound, and a frame
+/// fed elsewhere would otherwise wait through that, and through each later such write, before any flush sent it. When
+/// only the destination holds fed frames nothing is flushed here, so its own write batches them as stock does.
+///
+/// The check is one more `Router::route` (a slot lookup and a map probe, as the write's own), a few atomic loads and,
+/// when the write can reach the socket, one walk of the client's connections ([`has_unflushed`] on each), and allocates
+/// nothing; its flush adds an allocation only when two or more connections need one. With the `replicas` feature (not
+/// built by the service) a command routed to a replica is not checked: routing it twice would advance the replica
+/// round-robin, so the check could name another replica than the write's.
+pub(crate) async fn before_socket_write(inner: &RefCount<ClientInner>, router: &mut Router, command: &Command) {
+  // Test-only: count this check and time it (`retention::test_support::write_check_stats`); compiled out otherwise.
+  #[cfg(feature = "live-edit-retention-report")]
+  let mut check = super::retention::test_support::WriteCheck::begin(inner);
+  #[cfg(feature = "replicas")]
+  if command.use_replica {
+    return;
+  }
+  let target = match router.route(command) {
+    Some(conn) if write_reaches_socket(inner, conn, command) => address(conn),
+    _ => return,
+  };
+  if !others_fed(router, target) {
+    return;
+  }
+  #[cfg(feature = "live-edit-retention-report")]
+  check.flushing();
+  flush_before_write(inner, router, target).await;
+}
+
+/// Before one of the live-edit patch's own writes to `server` on a clustered client (`live_edit::before_write_to`:
+/// [`super::ask::follow`]'s `ASKING` and command, [`super::abandon::after_pipeline`]'s `DISCARD`): when another
+/// connection holds fed but unflushed frames, flush every connection holding them, the destination included, all at
+/// once, as [`before_socket_write`] does. These paths are rare, so the write predicate is not applied: any write there
+/// is treated as one that can reach the socket. Nothing is done when the client has no connection to `server`.
+pub(crate) async fn before_write_to(inner: &RefCount<ClientInner>, router: &mut Router, server: &Server) {
+  // Test-only: count this check and time it, as in `before_socket_write`.
+  #[cfg(feature = "live-edit-retention-report")]
+  let mut check = super::retention::test_support::WriteCheck::begin(inner);
+  let target = match router.get_connection_mut(server) {
+    Some(conn) => address(conn),
+    None => return,
+  };
+  if !others_fed(router, target) {
+    return;
+  }
+  #[cfg(feature = "live-edit-retention-report")]
+  check.flushing();
+  flush_before_write(inner, router, target).await;
+}
+
+/// Whether writing `command` to `conn` can reach `conn`'s socket: fred's own flush rule for the write
+/// (`prepare_command`, `src/router/utils.rs:56`), or a feed that must first flush what `conn` holds because its write
+/// buffer is at the backpressure boundary. A feed below the boundary only encodes the frame into the write buffer and
+/// cannot block, so it is not such a write: below the boundary fred's batching across connections is kept. Every term
+/// reads state only the router changes, apart from the command-queue counter, which senders can only raise while the
+/// router runs (a sender lowers it again only when its send finds the channel closed): so whenever fred's write, or
+/// the flush after it, touches the socket, this was true. Each term is its own function, so each can be judged, and
+/// reverted, alone.
+fn write_reaches_socket(inner: &RefCount<ClientInner>, conn: &Connection, command: &Command) -> bool {
+  queue_empty(conn) || feed_count_full(inner, conn) || flushes_by_kind(command) || at_backpressure(conn)
+}
+
+/// The counter-at-zero term (`Counters::should_send`'s "no queued command follows"): the write is a send. It also
+/// covers `write_command`'s flush of the destination after a write that did not flush
+/// (`src/router/commands.rs:164-166`), which runs only when the counter is zero: after a command the router skipped
+/// because its caller gave up, or could not encode.
+fn queue_empty(conn: &Connection) -> bool {
+  client_utils::read_atomic(&conn.counters.cmd_buffer_len) == 0
+}
+
+/// The feed-count term (`Counters::should_send`'s other half): the destination has fed more than `max_feed_count`
+/// frames (fred's default 200, unset by the service): a send, unless the flush before it resets the count first.
+fn feed_count_full(inner: &RefCount<ClientInner>, conn: &Connection) -> bool {
+  client_utils::read_atomic(&conn.counters.feed_count) as u64 > inner.max_feed_count()
+}
+
+/// The command-kind term (`CommandKind::should_flush`): the kinds fred always sends, every custom command and script
+/// among them. fred's third term, a command to every cluster node, never reaches a single-connection write.
+fn flushes_by_kind(command: &Command) -> bool {
+  command.kind.should_flush()
+}
+
+/// The backpressure term: the destination's write buffer is at tokio-util's backpressure boundary, so a feed first
+/// flushes what it holds (`FramedImpl::poll_ready`, tokio-util 0.7.19 `codec/framed_impl.rs:262-268`).
+fn at_backpressure(conn: &Connection) -> bool {
+  write_buffer_len(&conn.transport) >= super::retention::BACKPRESSURE_BYTES
+}
+
+/// A connection's address, to tell the write's destination apart from the other connections of one walk. A plain
+/// integer, unlike a pointer, so it can be held across an `await`; it is only compared, never turned back into a
+/// reference.
+fn address(conn: &Connection) -> usize {
+  conn as *const Connection as usize
+}
+
+/// Whether a connection other than the one at `target` ([`address`]) holds fed but unflushed frames. Walking the
+/// connections allocates nothing.
+fn others_fed(router: &mut Router, target: usize) -> bool {
+  connections_mut(router).any(|conn| address(conn) != target && has_unflushed(conn))
+}
+
+/// The flush of [`before_socket_write`] and [`before_write_to`]: every connection holding fed but unflushed frames,
+/// the destination at `target` included, all at once. The destination's name is taken first, for the DEBUG line a
+/// failed flush logs.
+async fn flush_before_write(inner: &RefCount<ClientInner>, router: &mut Router, target: usize) {
+  let to = match connections_mut(router).find(|conn| address(conn) == target) {
+    Some(conn) => conn.server.clone(),
+    None => return,
+  };
+  let _ = flush_fed_concurrently(connections_mut(router), FlushSite::Write { inner, to: &to }).await;
+}
+
+/// Where [`flush_fed_concurrently`] runs: it names the DEBUG line a failed flush logs.
+#[derive(Clone, Copy)]
+pub(crate) enum FlushSite<'a> {
+  /// [`before_wait`]: `Failed to flush fed frames to <node> before waiting`.
+  Wait(&'a RefCount<ClientInner>),
+  /// `live_edit::before_inline_connect`: `Failed to flush fed frames to <node> before an inline connect`.
+  InlineConnect(&'a RefCount<ClientInner>),
+  /// [`before_socket_write`] and [`before_write_to`], before a write to `to`: `Failed to flush fed frames to <node>
+  /// before a write to <to>`.
+  Write {
+    inner: &'a RefCount<ClientInner>,
+    to:    &'a Server,
+  },
+  /// fred's router-wide flush (`Connections::flush`, through `live_edit::flush_router_wide`): no line, as in stock;
+  /// its callers log or return the error.
+  RouterWide,
+}
+
+/// Flush every connection of `connections` holding fed but unflushed frames, all of them at once, each through the
+/// bounded flush and to its own end, and return the first error in the walk's order, if any. Every flush of a
+/// clustered client's connections taken together runs here: [`before_wait`]'s, the inline-connect flush
+/// (`live_edit::before_inline_connect`), the flush before a write ([`before_socket_write`], [`before_write_to`]) and
+/// fred's router-wide flush (`Connections::flush`, after a send, after the retry buffer's replay and in `drain_all`).
+///
+/// Run concurrently, a flush holds the router for at most one bound however many connections stall in it; one after
+/// another, k silent connections would hold it for k bounds. Each connection's flush runs to its own end, so a fast
+/// error on one never cuts another's short, as stock's `try_join_all` would. Only connections holding fed frames are
+/// flushed: with none the walk is all, and with one its flush runs inline, so neither adds an allocation to what the
+/// flush itself does. Two or more are joined (`join_all`), which allocates the join. A failed flush logs at DEBUG, as
+/// `site` names, and is left to the connection's read side, or returned to the router-wide flush's caller; an expired
+/// flush has marked the connection failed, and a connection already marked fails at once, with no I/O.
+pub(crate) async fn flush_fed_concurrently<'c, I>(connections: I, site: FlushSite<'_>) -> Result<(), Error>
+where
+  I: Iterator<Item = &'c mut Connection>,
+{
+  let mut fed = connections.filter(|conn| has_unflushed(conn));
+  let first = match fed.next() {
+    Some(conn) => conn,
+    None => return Ok(()),
+  };
+  let second = match fed.next() {
+    Some(conn) => conn,
+    None => return flush_one(first, site).await,
+  };
+  let flushes = [first, second].into_iter().chain(fed).map(|conn| flush_one(conn, site));
+  let results = futures::future::join_all(flushes).await;
+  results.into_iter().find(Result::is_err).unwrap_or(Ok(()))
+}
+
+/// One connection's flush for [`flush_fed_concurrently`], logging a failure as `site` names.
+async fn flush_one(conn: &mut Connection, site: FlushSite<'_>) -> Result<(), Error> {
+  let result = conn.flush().await;
+  if let Err(ref e) = result {
+    match site {
+      FlushSite::Wait(inner) => {
+        _debug!(inner, "Failed to flush fed frames to {} before waiting: {:?}", conn.server, e);
+      },
+      FlushSite::InlineConnect(inner) => {
+        _debug!(inner, "Failed to flush fed frames to {} before an inline connect: {:?}", conn.server, e);
+      },
+      FlushSite::Write { inner, to } => {
+        _debug!(inner, "Failed to flush fed frames to {} before a write to {}: {:?}", conn.server, to, e);
+      },
+      FlushSite::RouterWide => {},
+    }
+  }
+  result
+}
+
+/// The router took a command off its channel: set the command-queue counter to the channel's length plus one (the
+/// command just taken, which `process_command` subtracts next), and log a drift found doing so. The reset is a
+/// compare-and-set against the value read, so a sender that increments between that read and the reset is never
+/// overwritten; one that incremented before the read and has not sent yet has its increment dropped, and once its
+/// command lands the next receive, whichever command it takes, finds the counter one below and resets it again.
+///
+/// A difference of one either way is usually that race, or a single command cancelled when a router ended (see the
+/// module's *Never waiting on unsent frames*), and logs at DEBUG. A larger one logs a WARN, rate-limited per client to
+/// one per [`DRIFT_LOG_INTERVAL`].
+pub(crate) fn on_receive(inner: &RefCount<ClientInner>, rx: &CommandReceiver) {
+  inner.live_edit.liveness.mark_command(Instant::now());
+  let counter = &inner.counters.cmd_buffer_len;
+  let expected = rx.len().saturating_add(1);
+  let observed = client_utils::read_atomic(counter);
+  if observed == expected
+    || counter
+      .compare_exchange(observed, expected, Ordering::AcqRel, Ordering::Acquire)
+      .is_err()
+  {
+    return;
+  }
+  let drift = observed as i128 - expected as i128;
+  if drift.abs() == 1 {
+    _debug!(
+      inner,
+      "Command-queue counter off by one: the counter read {} where the command channel holds {} commands plus the \
+       one just taken ({:+}), usually a send in flight as it was read or that send's command landing after a reset, \
+       or a single command cancelled when a router ended; reset it to {}.",
+      observed,
+      expected - 1,
+      drift,
+      expected
+    );
+    return;
+  }
+  let now = Instant::now();
+  let mut log = inner.live_edit.liveness.shared.drift.lock().unwrap_or_else(|e| e.into_inner());
+  let due = log
+    .logged
+    .map_or(true, |logged| now.saturating_duration_since(logged) >= DRIFT_LOG_INTERVAL);
+  if due {
+    _warn!(
+      inner,
+      "Command-queue counter drift: the counter read {} where the command channel holds {} commands plus the one \
+       just taken ({:+}); reset it to {}. More than one send in flight as it was read, or commands that left the \
+       channel without being counted down, such as those cancelled when a router ended ({} further drifts larger \
+       than one not logged since the previous line).",
+      observed,
+      expected - 1,
+      drift,
+      expected,
+      log.suppressed
+    );
+    log.logged = Some(now);
+    log.suppressed = 0;
+  } else {
+    log.suppressed += 1;
+  }
+}
+
+/// Decrement the command-queue counter, saturating at zero instead of wrapping as stock fred's `decr_atomic` does.
+/// Returns the new value.
+pub(crate) fn saturating_decr(counter: &AtomicUsize) -> usize {
+  match counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| Some(value.saturating_sub(1))) {
+    Ok(previous) | Err(previous) => previous.saturating_sub(1),
+  }
+}
+
+/// Record a written command: a subscription command marks the connection a pubsub connection. Returns
+/// `cmd.has_no_responses()`, the condition stock `push_command` tests.
+pub(crate) fn on_push(conn: &mut Connection, cmd: &Command) -> bool {
+  if cmd.kind.is_pubsub() {
+    conn.live_edit.liveness.subscribed = true;
+  }
+  cmd.has_no_responses()
+}
+
+/// How frames read from `conn` are classified.
+pub(crate) fn classify(conn: &Connection) -> Classify {
+  Classify {
+    subscribed: conn.live_edit.liveness.subscribed,
+    shaped:     Some(Arc::clone(&conn.live_edit.liveness.pubsub_shaped)),
+  }
+}
+
+/// How frames read from the connection to `server` are classified. A missing connection keeps the stock
+/// classification.
+pub(crate) fn subscribed(router: &mut Router, server: &Server) -> Classify {
+  router.get_connection_mut(server).map(|conn| classify(conn)).unwrap_or(Classify {
+    subscribed: true,
+    shaped:     None,
+  })
+}
+
+/// Whether `frame` may be classified as a pubsub frame under `classify`.
+pub(crate) fn pubsub_classified(classify: &Classify, frame: &Resp3Frame) -> bool {
+  classify.subscribed || matches!(frame, Resp3Frame::Push { .. })
+}
+
+/// [`pubsub_classified`], called once per frame (`subscription_response` is whether the frame is shaped like a
+/// subscription response): a pubsub-shaped frame it does not classify as pubsub is counted on the connection.
+pub(crate) fn classify_frame(classify: &Classify, frame: &Resp3Frame, subscription_response: bool) -> bool {
+  let classified = pubsub_classified(classify, frame);
+  if !classified
+    && (subscription_response
+      || frame.is_normal_pubsub_message()
+      || frame.is_pattern_pubsub_message()
+      || frame.is_shard_pubsub_message())
+  {
+    if let Some(shaped) = classify.shaped.as_ref() {
+      shaped.fetch_add(1, Ordering::Relaxed);
+    }
+  }
+  classified
+}
+
+/// Convert a decoded frame to RESP3. On a connection that never subscribed, a RESP2 array stays an array at every
+/// depth: stock conversion turns any pubsub-shaped RESP2 array, nested ones included, into a `Push` frame.
+pub(crate) fn into_resp3(conn: &Connection, frame: ProtocolFrame) -> Resp3Frame {
+  match frame {
+    ProtocolFrame::Resp2(frame) if !conn.live_edit.liveness.subscribed => resp2_as_data(frame),
+    frame => frame.into_resp3(),
+  }
+}
+
+fn resp2_as_data(frame: Resp2Frame) -> Resp3Frame {
+  match frame {
+    Resp2Frame::SimpleString(data) => Resp3Frame::SimpleString { data, attributes: None },
+    Resp2Frame::Error(data) => Resp3Frame::SimpleError { data, attributes: None },
+    Resp2Frame::Integer(data) => Resp3Frame::Number { data, attributes: None },
+    Resp2Frame::BulkString(data) => Resp3Frame::BlobString { data, attributes: None },
+    Resp2Frame::Array(frames) => Resp3Frame::Array {
+      data:       frames.into_iter().map(resp2_as_data).collect(),
+      attributes: None,
+    },
+    Resp2Frame::Null => Resp3Frame::Null,
+  }
+}
+
+/// The slot a command on the connection to `server` is stamped and judged by: its hash slot when the cluster routing
+/// gives that slot to `server`, `None` when the command is judged at node level.
+///
+/// fred hashes a command's first string argument whatever it names (`ClusterHash::FirstKey`), and a hash slot
+/// says where a command was sent only when the router sent it by that slot. A command with nothing to hash (`PING`,
+/// `TIME`) has no slot. A command sent to a named node (`Options::cluster_node`: `with_cluster_node`, a `SCAN` page
+/// continuing on its node) can hash to a slot another node owns, as can one the router sent to an arbitrary
+/// connection because it holds none to the slot's owner. Judging either by that slot's range would count another
+/// node's replies as evidence for this one, so both are judged by their connection's node.
+fn judged_slot(state: &ClusterRouting, server: &Server, command: &Command) -> Option<u16> {
+  command
+    .cluster_hash()
+    .filter(|slot| state.get_server(*slot) == Some(server))
+}
+
+/// A reply was matched to `command` (popped from `conn`'s in-flight queue): stamp the peer registry with its slot,
+/// or with the connection's node for a command judged at node level ([`judged_slot`]). Returns `command`.
+pub(crate) fn on_reply(inner: &RefCount<ClientInner>, conn: &mut Connection, command: Command) -> Command {
+  if let Some(peers) = inner.connection.live_edit.liveness.peer_replies.as_ref() {
+    let now = Instant::now();
+    let slot = if inner.config.server.is_clustered() {
+      inner
+        .with_cluster_state(|state| Ok(judged_slot(state, &conn.server, &command)))
+        .unwrap_or(None)
+    } else {
+      command.cluster_hash()
+    };
+    match slot {
+      Some(slot) => peers.stamp(&peers.shared.slots[slot as usize % SLOTS], now),
+      None => {
+        let server = &conn.server;
+        let node = conn
+          .live_edit
+          .liveness
+          .node_stamp
+          .get_or_insert_with(|| peers.node(server));
+        peers.stamp(node, now);
+      },
+    }
+  }
+  command
+}
+
+fn oldest_outstanding(conn: &Connection) -> Option<&Command> {
+  conn.buffer.iter().find(|command| !command.kind.is_pubsub())
+}
+
+/// Whether the oldest outstanding command's frame is still only fed (one of the last `unsent` tracked commands).
+fn oldest_is_unsent(conn: &Connection) -> bool {
+  let tracked = conn.buffer.iter().filter(|command| !command.kind.is_pubsub()).count();
+  tracked > 0 && conn.live_edit.liveness.unsent >= tracked
+}
+
+/// The newest reply on any connection sharing `peers` from the slot range of `conn`'s oldest outstanding command,
+/// or from `conn`'s node for a command judged at node level ([`judged_slot`]) or a centralized client. Without the
+/// client's cluster routing no slot can be placed on the node, so only the node's own cell counts.
+fn newest_peer_reply(inner: &RefCount<ClientInner>, conn: &Connection, peers: &PeerReplies) -> Option<Instant> {
+  if !inner.config.server.is_clustered() {
+    return peers.newest(&[(0, (SLOTS - 1) as u16)], Some(&peers.node(&conn.server)));
+  }
+  let oldest = oldest_outstanding(conn);
+  let (ranges, by_node) = inner
+    .with_cluster_state(|state| {
+      let slot = oldest.and_then(|command| judged_slot(state, &conn.server, command));
+      let ranges = state
+        .slots()
+        .iter()
+        .filter(|range| match slot {
+          Some(slot) => range.start <= slot && slot <= range.end,
+          None => range.primary == conn.server,
+        })
+        .map(|range| (range.start, range.end))
+        .collect::<Vec<_>>();
+      Ok((ranges, slot.is_none()))
+    })
+    .unwrap_or_else(|_| (Vec::new(), true));
+  if by_node {
+    peers.newest(&ranges, Some(&peers.node(&conn.server)))
+  } else {
+    peers.newest(&ranges, None)
+  }
+}
+
+/// Update the connection's progress and return when its current stall began, or `None` when it has nothing to
+/// judge (no outstanding non-pubsub command, or `blocked`). A connection whose bounded wait expired is always
+/// judged.
+///
+/// Ends the previous stall when it changed: one that had peer evidence and recovered by itself is a near miss.
+pub(crate) fn stall_start(inner: &RefCount<ClientInner>, conn: &mut Connection, now: Instant) -> Option<Instant> {
+  if conn.live_edit.liveness.expired.is_some() {
+    return Some(now);
+  }
+  let read_len = read_buffer_len(&conn.transport);
+  let written = if conn.blocked {
+    None
+  } else {
+    // A command still only fed is measured from the later of its prepare time and the connection's last completed
+    // write, feed or flush: a large frame being uploaded ahead of it is progress, not silence.
+    let unsent = oldest_is_unsent(conn);
+    let last_write = conn.live_edit.liveness.last_write;
+    oldest_outstanding(conn).map(|command| {
+      let written = command.network_start.unwrap_or(now);
+      match last_write {
+        Some(last_write) if unsent && last_write > written => last_write,
+        _ => written,
+      }
+    })
+  };
+  let state = &mut conn.live_edit.liveness;
+  if read_len > state.read_len {
+    state.last_progress = Some(now);
+  }
+  state.read_len = read_len;
+  let since = written.map(|written| match state.last_progress {
+    Some(progress) if progress > written => progress,
+    _ => written,
+  });
+
+  if let Some(ended) = state.episode.filter(|episode| Some(episode.since) != since) {
+    state.episode = None;
+    state.evidence_checked = None;
+    if ended.evidence {
+      let end = state
+        .last_progress
+        .filter(|progress| *progress > ended.since)
+        .unwrap_or(now);
+      near_miss(inner, &conn.server, end.saturating_duration_since(ended.since));
+    }
+  }
+  if let Some(since) = since {
+    if state.episode.is_none() {
+      state.episode = Some(Episode { since, evidence: false });
+      if state.longest_wait.is_some_and(|wait| wait.ended < since) {
+        state.longest_wait = None;
+      }
+      if state.longest_fed.is_some_and(|fed| fed.ended < since) {
+        state.longest_fed = None;
+      }
+    }
+  }
+  since
+}
+
+/// Whether the stall that began at `since` is a verdict (fast or slow, see the module docs), and if so the error
+/// that closes the connection. `slow` is the client's `UnresponsiveConfig::max_timeout`. A verdict is logged by
+/// [`log_verdicts`] once every connection of the client has been polled; an expired wait was logged when it expired.
+pub(crate) fn verdict(
+  inner: &RefCount<ClientInner>,
+  conn: &mut Connection,
+  now: Instant,
+  since: Instant,
+  slow: Duration,
+) -> Option<Error> {
+  if conn.live_edit.liveness.expired.is_some() {
+    return Some(verdict_error(Verdict::Slow));
+  }
+  let settings = &inner.connection.live_edit.liveness;
+  let stall = now.saturating_duration_since(since);
+  if stall > slow {
+    return Some(defer(conn, Verdict::Slow, stall, since));
+  }
+
+  let (fast, peers) = match (settings.fast_timeout, settings.peer_replies.as_ref()) {
+    (Some(fast), Some(peers)) => (fast, peers),
+    _ => return None,
+  };
+  if stall <= settings.peer_evidence_lag {
+    return None;
+  }
+  let has_evidence = conn.live_edit.liveness.episode.map(|e| e.evidence).unwrap_or(false);
+  if !has_evidence {
+    let due = conn
+      .live_edit
+      .liveness
+      .evidence_checked
+      .map(|checked| now.saturating_duration_since(checked) >= inner.connection.unresponsive.interval)
+      .unwrap_or(true);
+    if !due {
+      return None;
+    }
+    conn.live_edit.liveness.evidence_checked = Some(now);
+    let answered = newest_peer_reply(inner, conn, peers)
+      .map(|newest| newest >= since + settings.peer_evidence_lag)
+      .unwrap_or(false);
+    if !answered {
+      return None;
+    }
+    if let Some(episode) = conn.live_edit.liveness.episode.as_mut() {
+      episode.evidence = true;
+    }
+  }
+
+  if stall > fast {
+    Some(defer(conn, Verdict::Fast, stall, since))
+  } else {
+    None
+  }
+}
+
+/// Record a fresh fast or slow verdict on `conn` — the one place the read loop reaches one — and return its error.
+/// The verdict is logged by [`log_verdicts`] (`pending`) and marks the connection judged dead (`closed_by`), so its
+/// close does not flush ([`close`]).
+///
+/// The read loop returns that error as the connection's poll result, and the router closes the connection in the same
+/// step: `process_response` drops a connection whose result is an error (`router/utils.rs` `disconnect!`). Neither
+/// `select!` that polls the read loop drops a ready result: `read_or_write` is biased toward it, and in
+/// `read_and_sleep` a verdict exists only once the read branch is polled and ready. An error from an earlier result of
+/// the same batch can leave it unprocessed: in `read_or_write` that error ends the router, whose `disconnect_all`
+/// closes the connection next; `read_and_sleep` returns early only when fred will not reconnect (a client that is
+/// quitting, or whose reconnect policy is spent), and the connection is then closed at the router's end or judged
+/// again at its next poll.
+fn defer(conn: &mut Connection, verdict: Verdict, stall: Duration, since: Instant) -> Error {
+  conn.live_edit.liveness.pending = Some(Pending { verdict, stall, since });
+  conn.live_edit.liveness.closed_by = Some(verdict);
+  verdict_error(verdict)
+}
+
+/// Whether the read loop reached a verdict on `conn` in this poll that is not logged yet.
+pub(crate) fn has_pending(conn: &Connection) -> bool {
+  conn.live_edit.liveness.pending.is_some()
+}
+
+/// Log every verdict the read loop reached on `connections` in this poll, each with a summary of the client's other
+/// connections. Called after every connection has been polled (`ReadAllFuture`, `ReadFuture`), once
+/// [`has_pending`] found one.
+pub(crate) fn log_verdicts<'a, I>(inner: &RefCount<ClientInner>, connections: I)
+where
+  I: IntoIterator<Item = &'a mut Connection>,
+{
+  let mut all: Vec<&mut Connection> = connections.into_iter().collect();
+  let pending: Vec<(usize, Pending)> = all
+    .iter_mut()
+    .enumerate()
+    .filter_map(|(i, conn)| conn.live_edit.liveness.pending.take().map(|pending| (i, pending)))
+    .collect();
+  if !log::log_enabled!(log::Level::Warn) {
+    return;
+  }
+  let now = Instant::now();
+  for (i, pending) in pending {
+    let conn = &*all[i];
+    let newest_peer = describe_newest_peer(inner, conn, now);
+    let mut line = format!(
+      "Unresponsive connection to {} ({} verdict): no reply progress for {:?}; newest reply from the same slot \
+       range on another connection: {}. ",
+      conn.server,
+      pending.verdict.as_str(),
+      pending.stall,
+      newest_peer
+    );
+    let since = Some(now.saturating_duration_since(pending.since));
+    describe_connection(&mut line, conn, now, since);
+    describe_client(&mut line, conn, now);
+    line.push_str("Other connections: ");
+    let mut others = 0;
+    for (j, other) in all.iter().enumerate() {
+      if j == i {
+        continue;
+      }
+      if others > 0 {
+        line.push_str("; ");
+      }
+      others += 1;
+      describe_other(&mut line, other, now);
+    }
+    if others == 0 {
+      line.push_str("none");
+    }
+    line.push_str(". ");
+    describe_kernel(&mut line, conn);
+    line.push_str(" Closing it and reconnecting.");
+    _warn!(inner, "{}", line);
+  }
+}
+
+/// The age of a command's `network_start`: since fred prepared its frame (`write_command`) while it is only fed,
+/// since the send or flush that put it on the socket once sent ([`on_sent`] re-stamps it). Without fred's `metrics`
+/// feature a command carries no other timestamp.
+fn command_age(command: &Command, now: Instant) -> String {
+  ago(now, command.network_start)
+}
+
+/// The age of the newest reply on another connection from the slot range of `conn`'s oldest outstanding command, or
+/// from `conn`'s node ([`newest_peer_reply`]).
+fn describe_newest_peer(inner: &RefCount<ClientInner>, conn: &Connection, now: Instant) -> String {
+  inner
+    .connection
+    .live_edit
+    .liveness
+    .peer_replies
+    .as_ref()
+    .and_then(|peers| newest_peer_reply(inner, conn, peers))
+    .map(|newest| format!("{:?} ago", now.saturating_duration_since(newest)))
+    .unwrap_or_else(|| "none recorded".to_owned())
+}
+
+fn ago(now: Instant, at: Option<Instant>) -> String {
+  match at {
+    Some(at) => format!("{:?} ago", now.saturating_duration_since(at)),
+    None => "never".to_owned(),
+  }
+}
+
+fn in_stall(wait: Option<Wait>, stall: Option<Duration>, now: Instant) -> Option<Wait> {
+  wait.filter(|wait| stall.map_or(true, |stall| now.saturating_duration_since(wait.ended) <= stall))
+}
+
+fn describe_wait(wait: Option<Wait>, stall: Option<Duration>, now: Instant) -> String {
+  match in_stall(wait, stall, now) {
+    Some(wait) => format!("{} {:?}", wait.what, wait.took),
+    None => "none".to_owned(),
+  }
+}
+
+/// The longest fed-to-sent delay of a command sent since the stall began: how long a frame sat in fred's write buffer
+/// before it reached the socket, which `network_start`'s re-stamp at the send hides.
+fn describe_fed(fed: Option<Wait>, stall: Option<Duration>, now: Instant) -> String {
+  match in_stall(fed, stall, now) {
+    Some(fed) => format!("{:?}", fed.took),
+    None => "none".to_owned(),
+  }
+}
+
+/// This connection's fields of a verdict line. `stall` limits the longest wait to the waits that ended within it.
+fn describe_connection(line: &mut String, conn: &Connection, now: Instant, stall: Option<Duration>) {
+  let state = &conn.live_edit.liveness;
+  let oldest = match oldest_outstanding(conn) {
+    Some(command) => format!(
+      "{}, {} {}",
+      command.kind.to_str_debug(),
+      if oldest_is_unsent(conn) {
+        "fed, not yet sent; prepared"
+      } else {
+        "sent"
+      },
+      command_age(command, now)
+    ),
+    None => "none".to_owned(),
+  };
+  let _ = write!(
+    line,
+    "This connection: {} in flight; oldest outstanding command {}; {} commands fed since the last send or flush, \
+     {} bytes in the write buffer, feed count {}; {} bytes in the read buffer, last read progress {}; subscribed {}, \
+     {} pubsub-shaped frames received; longest wait since the stall began: {}; longest time a command sent since \
+     the stall began sat fed before it was sent: {}. ",
+    conn.buffer.len(),
+    oldest,
+    state.unsent,
+    write_buffer_len(&conn.transport),
+    client_utils::read_atomic(&conn.counters.feed_count),
+    read_buffer_len(&conn.transport),
+    ago(now, state.last_progress),
+    state.subscribed,
+    state.pubsub_shaped.load(Ordering::Relaxed),
+    describe_wait(state.longest_wait, stall, now),
+    describe_fed(state.longest_fed, stall, now),
+  );
+}
+
+/// The client's fields of a verdict line, read through `conn`.
+fn describe_client(line: &mut String, conn: &Connection, now: Instant) {
+  let (queued, last_command) = match conn.live_edit.liveness.inner.as_ref() {
+    Some(ClientHandle(inner)) => (
+      inner.live_edit.liveness.shared.queued.load(Ordering::Relaxed).to_string(),
+      ago(now, inner.live_edit.liveness.last_command()),
+    ),
+    None => ("unknown".to_owned(), "unknown".to_owned()),
+  };
+  let _ = write!(
+    line,
+    "This client: command-queue counter {}, command channel length {} when the router last waited, last command \
+     taken by the router {}. ",
+    client_utils::read_atomic(&conn.counters.cmd_buffer_len),
+    queued,
+    last_command,
+  );
+}
+
+/// One other connection's summary in a verdict line.
+fn describe_other(line: &mut String, conn: &Connection, now: Instant) {
+  let state = &conn.live_edit.liveness;
+  let stall = state.episode.map(|episode| now.saturating_duration_since(episode.since));
+  let _ = write!(
+    line,
+    "{}: {} in flight, oldest {}, {} unsent, {} bytes in the write buffer, longest wait {}, longest fed before \
+     sent {}",
+    conn.server,
+    conn.buffer.len(),
+    oldest_outstanding(conn)
+      .map(|command| command_age(command, now))
+      .unwrap_or_else(|| "none".to_owned()),
+    state.unsent,
+    write_buffer_len(&conn.transport),
+    describe_wait(state.longest_wait, stall, now),
+    describe_fed(state.longest_fed, stall, now),
+  );
+}
+
+/// The kernel's view of the connection's socket, from `/proc/self/net/tcp` or `tcp6`; `unavailable` with no such file
+/// or no matching row.
+fn describe_kernel(line: &mut String, conn: &Connection) {
+  line.push_str("Kernel socket: ");
+  match socket_addrs(&conn.transport).and_then(|(local, remote)| kernel_row(local, remote)) {
+    Some(row) => line.push_str(&row),
+    None => line.push_str("unavailable"),
+  }
+  line.push('.');
+}
+
+/// A socket address as `/proc/net/tcp{,6}` prints it: the address as the native-endian 32-bit words of its network
+/// byte order bytes, then the port, all in upper-case hex.
+fn proc_address(addr: SocketAddr) -> String {
+  let mut out = String::new();
+  match addr {
+    SocketAddr::V4(v4) => {
+      let _ = write!(out, "{:08X}", u32::from_ne_bytes(v4.ip().octets()));
+    },
+    SocketAddr::V6(v6) => {
+      let octets = v6.ip().octets();
+      for word in 0 .. 4 {
+        let bytes = [
+          octets[word * 4],
+          octets[word * 4 + 1],
+          octets[word * 4 + 2],
+          octets[word * 4 + 3],
+        ];
+        let _ = write!(out, "{:08X}", u32::from_ne_bytes(bytes));
+      }
+    },
+  }
+  let _ = write!(out, ":{:04X}", addr.port());
+  out
+}
+
+fn timer_name(code: u64) -> String {
+  match code {
+    0 => "none".to_owned(),
+    1 => "retransmit".to_owned(),
+    2 => "keepalive".to_owned(),
+    3 => "time-wait".to_owned(),
+    4 => "zero-window probe".to_owned(),
+    other => format!("code {other}"),
+  }
+}
+
+/// The `tx_queue`, `rx_queue`, timer and `retrnsmt` fields of the `/proc/self/net/tcp{,6}` row whose local and remote
+/// addresses are `local` and `remote`.
+fn kernel_row(local: SocketAddr, remote: SocketAddr) -> Option<String> {
+  let path = match (local, remote) {
+    (SocketAddr::V4(_), SocketAddr::V4(_)) => "/proc/self/net/tcp",
+    (SocketAddr::V6(_), SocketAddr::V6(_)) => "/proc/self/net/tcp6",
+    _ => return None,
+  };
+  let (local, remote) = (proc_address(local), proc_address(remote));
+  let table = std::fs::read_to_string(path).ok()?;
+  let hex = |field: &str| u64::from_str_radix(field, 16).ok();
+  for row in table.lines().skip(1) {
+    let fields: Vec<&str> = row.split_whitespace().collect();
+    if fields.len() < 7 || !fields[1].eq_ignore_ascii_case(&local) || !fields[2].eq_ignore_ascii_case(&remote) {
+      continue;
+    }
+    let (tx, rx) = fields[4].split_once(':')?;
+    let (timer, _) = fields[5].split_once(':')?;
+    return Some(format!(
+      "tx_queue {} bytes, rx_queue {} bytes, timer {}, retrnsmt {}",
+      hex(tx)?,
+      hex(rx)?,
+      timer_name(hex(timer)?),
+      hex(fields[6])?
+    ));
+  }
+  None
+}
+
+/// Log a near miss: a stall that had peer evidence and recovered by itself. Rate-limited per registry to one line
+/// per [`NEAR_MISS_LOG_INTERVAL`], except that a near miss longer than every one logged in the current window is
+/// always logged.
+fn near_miss(inner: &RefCount<ClientInner>, server: &Server, stall: Duration) {
+  let peers = match inner.connection.live_edit.liveness.peer_replies.as_ref() {
+    Some(peers) => peers,
+    None => return,
+  };
+  let now = Instant::now();
+  let mut log = peers.shared.near_misses.lock().unwrap_or_else(|e| e.into_inner());
+  let window_over = log
+    .window_start
+    .map(|start| now.saturating_duration_since(start) >= NEAR_MISS_LOG_INTERVAL)
+    .unwrap_or(true);
+  if window_over || stall > log.longest {
+    _info!(
+      inner,
+      "Near miss on the connection to {}: no reply progress for {:?} while the same slot range answered other \
+       connections, then it recovered by itself ({} shorter near misses not logged since the previous line).",
+      server,
+      stall,
+      log.suppressed
+    );
+    if window_over {
+      log.window_start = Some(now);
+    }
+    log.longest = stall;
+    log.suppressed = 0;
+  } else {
+    log.suppressed += 1;
+  }
+}
--- a/src/live_edit/mod.rs
+++ b/src/live_edit/mod.rs
@@ -0,0 +1,321 @@
+// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+// SPDX-License-Identifier: Apache-2.0
+//
+// Licensed under the Apache License, Version 2.0 (the "License");
+// you may not use this file except in compliance with the License.
+// You may obtain a copy of the License at
+//
+//     http://www.apache.org/licenses/LICENSE-2.0
+//
+// Unless required by applicable law or agreed to in writing, software
+// distributed under the License is distributed on an "AS IS" BASIS,
+// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+// See the License for the specific language governing permissions and
+// limitations under the License.
+
+//! The live-edit service's changes to fred 10.1.0.
+//!
+//! fred is vendored by the live-edit repository as the published 10.1.0 crate plus one patch file. The patch edits
+//! existing lines only, so every file keeps its line count; new code lives in this module, and each hook in an
+//! existing file carries a `live-edit patch` comment. The record of every change, with the reason for each, is
+//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
+//!
+//! Shared state is one settings field on fred's config (`ConnectionConfig::live_edit`, a [`Settings`]), one
+//! per-client field (`ClientInner::live_edit`, a [`ClientState`]), one per-connection field
+//! (`Connection::live_edit`, a [`ConnectionState`]), one per-pool field (`PoolInner::live_edit`, a
+//! [`pool::PoolState`]) and one field on each connection's codec (`Codec::live_edit`, a [`CodecState`]). Hooks in
+//! existing files call the functions here, which dispatch to the submodules:
+//!
+//! - [`abandon`]: a command, pipeline or transaction is never written after its caller gave up on it (its timeout
+//!   fired, fred answered it with an error, or its future was dropped); a pipeline cut during the router step that
+//!   writes it, and a fred `Transaction` cut partway, never leave a `MULTI` open. A pipeline replayed from the retry
+//!   buffer is not covered; the service sends every `MULTI` pipeline at one write attempt, so it never replays one
+//!   (the entry points, `process_pipeline` and the transaction router call it directly).
+//! - [`ask`]: following a cluster `-ASK` redirect to its target node, with a pipelined `ASKING` and no synchronous
+//!   read (`process_ask` calls [`ask::follow`] directly).
+//! - [`liveness`]: the unresponsive check, bounded router waits, closing a dead connection without waiting on it,
+//!   flushing fed frames before the router waits, connects inline or writes where the write can reach a socket, all
+//!   connections at once, the command-queue counter's resync, pubsub classification and the state a verdict records.
+//! - [`pool`]: a `Pool`'s logical members, each re-pointable to a fresh client.
+//! - [`retention`]: what a connection keeps once its traffic drains (connection buffer limits).
+
+pub mod abandon;
+pub mod ask;
+pub mod liveness;
+pub mod pool;
+pub mod retention;
+
+pub use retention::CodecState;
+
+use crate::{
+  error::Error,
+  modules::inner::{ClientInner, CommandReceiver},
+  protocol::{
+    codec::Codec,
+    command::Command,
+    connection::{Connection, ExclusiveConnection},
+    types::{ProtocolFrame, Server},
+  },
+  router::Router,
+  runtime::RefCount,
+};
+use redis_protocol::resp3::types::BytesFrame as Resp3Frame;
+use std::{
+  collections::{HashMap, VecDeque},
+  sync::atomic::AtomicUsize,
+  time::{Duration, Instant},
+};
+
+pub(crate) use liveness::Classify;
+
+/// Settings for the live-edit additions, on `ConnectionConfig::live_edit`. The default keeps fred's stock
+/// behaviour apart from the fixes that always apply (see each submodule).
+#[derive(Clone, Debug, Default, PartialEq, Eq)]
+pub struct Settings {
+  /// Connection liveness settings.
+  pub liveness:  liveness::Settings,
+  /// Per-connection buffer limits ([`retention`]). `None`, the default, keeps stock fred's buffers.
+  pub retention: Option<retention::RetentionLimits>,
+}
+
+/// Per-client state for the live-edit additions, on `ClientInner::live_edit`.
+#[derive(Clone, Debug, Default)]
+pub struct ClientState {
+  pub(crate) liveness:  liveness::ClientState,
+  /// The test-only retention report, sweep mode, and sweep and write-check counters ([`retention::test_support`]),
+  /// compiled only with the `live-edit-retention-report` feature.
+  #[cfg(feature = "live-edit-retention-report")]
+  pub(crate) retention: std::sync::Arc<retention::test_support::ReportState>,
+}
+
+/// Per-connection state for the live-edit additions, on `Connection::live_edit`.
+#[derive(Debug, Default)]
+pub struct ConnectionState {
+  pub(crate) liveness: liveness::ConnectionState,
+  pub(crate) abandon:  abandon::ConnectionState,
+}
+
+/// Make a pipelined connection from an initialized one (stock `into_pipelined`) and set up its state.
+pub(crate) fn pipelined(inner: &RefCount<ClientInner>, transport: ExclusiveConnection, replica: bool) -> Connection {
+  let mut conn = transport.into_pipelined(replica);
+  liveness::attach(inner, &mut conn);
+  conn
+}
+
+/// `Connection::push_command`: record the command; returns whether it has no responses.
+pub(crate) fn on_push(conn: &mut Connection, cmd: &Command) -> bool {
+  liveness::on_push(conn, cmd)
+}
+
+/// `Connection::write`: count a tracked write.
+pub(crate) fn on_write(conn: &mut Connection) {
+  liveness::count_unsent(conn);
+}
+
+/// A reply was matched to `command`, popped from `conn`'s in-flight queue. Returns `command`.
+pub(crate) fn on_reply(inner: &RefCount<ClientInner>, conn: &mut Connection, command: Command) -> Command {
+  let command = liveness::on_reply(inner, conn, command);
+  retention::on_reply(conn);
+  command
+}
+
+/// The read loop took a frame, or the end of the stream, off `conn`.
+pub(crate) fn on_frame(conn: &mut Connection, now: Instant) {
+  liveness::on_frame(conn, now);
+  retention::on_read(conn);
+}
+
+/// Write and flush a frame.
+pub(crate) async fn send(conn: &mut Connection, frame: ProtocolFrame) -> Result<(), Error> {
+  liveness::send(conn, frame).await?;
+  retention::on_flushed(conn);
+  Ok(())
+}
+
+/// Buffer a frame.
+pub(crate) async fn feed(conn: &mut Connection, frame: ProtocolFrame) -> Result<(), Error> {
+  liveness::feed(conn, frame).await
+}
+
+/// Flush the connection.
+pub(crate) async fn flush(conn: &mut Connection) -> Result<(), Error> {
+  liveness::flush(conn).await?;
+  retention::on_flushed(conn);
+  Ok(())
+}
+
+/// Read the next frame.
+pub(crate) async fn next(conn: &mut Connection) -> Result<Option<Result<ProtocolFrame, Error>>, Error> {
+  let item = liveness::next(conn).await?;
+  // Read rule here: a fred `Transaction`'s `exec` (CreateLayer's `client.multi()`) reads its EXEC reply via `next`.
+  retention::on_read(conn);
+  Ok(item)
+}
+
+/// `Connection::close`: close the transport, unless the connection was judged dead or is closed under an abandoned
+/// transaction ([`abandon::close_transaction`]); either returns at once, with no flush and no shutdown.
+pub(crate) async fn close(conn: &mut Connection) -> Result<(), Error> {
+  if abandon::closes_at_once(conn) {
+    return Ok(());
+  }
+  liveness::close(conn).await
+}
+
+/// Convert a frame read from `conn` to RESP3.
+pub(crate) fn into_resp3(conn: &Connection, frame: ProtocolFrame) -> Resp3Frame {
+  liveness::into_resp3(conn, frame)
+}
+
+/// How frames read from `conn` are classified (pubsub or data).
+pub(crate) fn classify(conn: &Connection) -> Classify {
+  liveness::classify(conn)
+}
+
+/// How frames read from the connection to `server` are classified (pubsub or data).
+pub(crate) fn subscribed(router: &mut Router, server: &Server) -> Classify {
+  liveness::subscribed(router, server)
+}
+
+/// Whether `frame` may be classified as a pubsub frame under `classify`.
+pub(crate) fn pubsub_classified(classify: &Classify, frame: &Resp3Frame) -> bool {
+  liveness::pubsub_classified(classify, frame)
+}
+
+/// [`pubsub_classified`], once per frame, counting a pubsub-shaped frame it does not classify as pubsub.
+pub(crate) fn classify_frame(classify: &Classify, frame: &Resp3Frame, subscription_response: bool) -> bool {
+  liveness::classify_frame(classify, frame, subscription_response)
+}
+
+/// The read loop found `conn` pending: when its current stall began, if it has one to judge.
+pub(crate) fn stall_start(inner: &RefCount<ClientInner>, conn: &mut Connection, now: Instant) -> Option<Instant> {
+  liveness::stall_start(inner, conn, now)
+}
+
+/// The error that closes `conn` as unresponsive, when the stall that began at `since` is a verdict.
+pub(crate) fn verdict(
+  inner: &RefCount<ClientInner>,
+  conn: &mut Connection,
+  now: Instant,
+  since: Instant,
+  slow: Duration,
+) -> Option<Error> {
+  liveness::verdict(inner, conn, now, since, slow)
+}
+
+/// The read loop has polled every connection of a clustered (or replica) client: log the verdicts it reached.
+pub(crate) fn log_verdicts_all(inner: &RefCount<ClientInner>, connections: &mut HashMap<Server, Connection>) {
+  if connections.values().any(liveness::has_pending) {
+    liveness::log_verdicts(inner, connections.values_mut());
+  }
+}
+
+/// The read loop has polled a centralized client's connection: log the verdict it reached, if any.
+pub(crate) fn log_verdicts_one(inner: &RefCount<ClientInner>, connection: &mut Connection) {
+  if liveness::has_pending(connection) {
+    liveness::log_verdicts(inner, std::iter::once(connection));
+  }
+}
+
+/// The top of the router's `read_or_write`: the connection-buffer sweep, then flush fed frames when no queued command
+/// follows.
+pub(crate) async fn before_wait(inner: &RefCount<ClientInner>, router: &mut Router, rx: &CommandReceiver) {
+  // Test-only: the retention report request pending before this pass's sweep begins, if any. Read first so that the
+  // sweep below runs under every sweep mode a test set before that request.
+  #[cfg(feature = "live-edit-retention-report")]
+  let report_request = retention::test_support::pending_report(inner);
+  retention::sweep(inner, router);
+  // Test-only: answer that request with what this pass's sweep left. Deliberately outside `sweep`, so a build with
+  // the sweep call removed still answers reports, and that perturbation fails on what the report shows rather than
+  // on a report that never comes.
+  #[cfg(feature = "live-edit-retention-report")]
+  if let Some(request) = report_request {
+    retention::test_support::answer_report(inner, router, request);
+  }
+  liveness::before_wait(inner, router, rx).await;
+}
+
+/// Before every inline connect, the one entry point of all three sites: the top of every pass of
+/// `reconnect_with_policy` and `sync_cluster_with_policy`, before the pass's backoff sleep and its attempt, and
+/// [`ask::follow`], before it opens a connection to an `-ASK` target. On a clustered client, flush every connection
+/// holding fed but unflushed frames, all at once and whatever the command channel holds
+/// ([`liveness::flush_fed_concurrently`]; [`liveness`]'s *Never waiting on unsent frames*), so the connect waits at
+/// most one bound for it however many connections hold fed frames. The connect runs inline, polling no connection,
+/// and keeps the connections it does not replace, so a frame left fed on one would otherwise wait through it.
+///
+/// Any other client does nothing here: its reconnect attempt closes its one connection before it connects, and that
+/// close flushes it (or, when the connection was judged dead, returns at once without flushing). Flushing first would
+/// only replace that close's flush, bounded by fred's 5 s close timeout, with one bounded by the slow liveness bound
+/// (`UnresponsiveConfig::max_timeout`). `ask::follow` runs only on a clustered client, so the gate never skips it.
+pub(crate) async fn before_inline_connect(inner: &RefCount<ClientInner>, router: &mut Router) {
+  if inner.config.server.is_clustered() {
+    let site = liveness::FlushSite::InlineConnect(inner);
+    let _ = liveness::flush_fed_concurrently(liveness::connections_mut(router), site).await;
+  }
+}
+
+/// Before the router writes `command` to the one connection it routes to: the top of the single-connection branch of
+/// `write_command` (`src/router/commands.rs`), which every single command, every pipeline command, the retry buffer's
+/// replay after a reconnect or cluster sync, and a `-MOVED` retry's write go through. On a clustered client, when the
+/// write can reach a socket and another connection holds fed but unflushed frames, flush every connection holding
+/// them, the destination included, all at once ([`liveness::before_socket_write`]; [`liveness`]'s *Never waiting on
+/// unsent frames*), so no frame fed elsewhere waits behind this write's own socket wait.
+///
+/// Any other client has one connection, so no other connection's frames can wait behind the write: it returns at
+/// once.
+pub(crate) async fn before_socket_write(inner: &RefCount<ClientInner>, router: &mut Router, command: &Command) {
+  if inner.config.server.is_clustered() {
+    liveness::before_socket_write(inner, router, command).await;
+  }
+}
+
+/// Before one of this patch's own writes to `server` outside `write_command`: [`ask::follow`]'s `ASKING` and command
+/// to an `-ASK` target, and [`abandon::after_pipeline`]'s `DISCARD`. On a clustered client, when another connection
+/// holds fed but unflushed frames, flush every connection holding them, all at once, as [`before_socket_write`] does,
+/// without its write predicate: these paths are rare ([`liveness::before_write_to`]). Any other client returns at
+/// once, as there.
+pub(crate) async fn before_write_to(inner: &RefCount<ClientInner>, router: &mut Router, server: &Server) {
+  if inner.config.server.is_clustered() {
+    liveness::before_write_to(inner, router, server).await;
+  }
+}
+
+/// fred's router-wide flush of a clustered client's connections (`Connections::flush`, which `Router::flush` runs
+/// after a successful send, after the retry buffer's replay and at the start of `drain_all`): flush every connection
+/// holding fed but unflushed frames, all at once and each to its own end, and return the first error, which the
+/// callers log or return as they did stock's ([`liveness::flush_fed_concurrently`]). Stock flushed every connection
+/// through `try_join_all`, which allocated on every call and dropped the other flushes at the first error; this
+/// flushes only connections holding fed frames, allocates only when two or more do, and lets a fast error on one
+/// never cut another's flush short. A centralized or sentinel client keeps stock's flush of its one connection.
+pub(crate) async fn flush_router_wide<'a>(connections: impl Iterator<Item = &'a mut Connection>) -> Result<(), Error> {
+  liveness::flush_fed_concurrently(connections, liveness::FlushSite::RouterWide).await
+}
+
+/// The router took a command off its channel: resync the command-queue counter.
+pub(crate) fn on_receive(inner: &RefCount<ClientInner>, rx: &CommandReceiver) {
+  liveness::on_receive(inner, rx);
+}
+
+/// Decrement the command-queue counter, saturating at zero.
+pub(crate) fn saturating_decr(counter: &AtomicUsize) -> usize {
+  liveness::saturating_decr(counter)
+}
+
+/// The retention state for a new codec of the client `inner`.
+pub(crate) fn codec_state(inner: &RefCount<ClientInner>) -> CodecState {
+  retention::codec_state(inner)
+}
+
+/// The codec encoded a frame, leaving `buffered` bytes in the write buffer.
+pub(crate) fn on_encode(codec: &Codec, buffered: usize) {
+  retention::on_encode(codec, buffered);
+}
+
+/// The codec decoded a frame of `size` bytes.
+pub(crate) fn on_decode(codec: &Codec, size: usize) {
+  retention::on_decode(codec, size);
+}
+
+/// The router replayed its retry buffer after a reconnect.
+pub(crate) fn on_retry_replayed(inner: &RefCount<ClientInner>, retry_buffer: &mut VecDeque<Command>) {
+  retention::on_retry_replayed(inner, retry_buffer);
+}
--- a/src/live_edit/pool.rs
+++ b/src/live_edit/pool.rs
@@ -0,0 +1,267 @@
+// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+// SPDX-License-Identifier: Apache-2.0
+//
+// Licensed under the Apache License, Version 2.0 (the "License");
+// you may not use this file except in compliance with the License.
+// You may obtain a copy of the License at
+//
+//     http://www.apache.org/licenses/LICENSE-2.0
+//
+// Unless required by applicable law or agreed to in writing, software
+// distributed under the License is distributed on an "AS IS" BASIS,
+// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+// See the License for the specific language governing permissions and
+// limitations under the License.
+
+//! A `Pool`'s logical members, each re-pointable to a fresh, already-connected client.
+//!
+//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
+//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
+//!
+//! Stock fred's `Pool` holds a fixed `Vec<Client>`, and its member is the client at index `i` for the pool's life.
+//! A client that keeps failing — whatever per-client state does it — can then only be reconnected, never swapped for
+//! a new one, because the pool hands out borrowed `&Client`s (`next`, `last`, and `ClientLike::inner`'s
+//! `&RefCount<ClientInner>`, which every command interface goes through) with no lifetime the pool could wait for.
+//!
+//! Here each logical member `i` names a **cell**: cells `0 .. size` are the pool's original clients, unchanged, and
+//! `max(size, 4)` **spare cells** follow, each filled at most once. [`replace_member`] stores a fresh client in the
+//! next empty spare cell and points member `i` at it; every pool operation (`next`, `next_connected`, `last`,
+//! `ClientLike::inner`, `clients`, `member`, `connect`, `connect_pool`, `init`, `wait_for_connect`,
+//! `force_reconnection`, `quit`, `update_perf_config`, `active_connections`, `set_resolver`, the heartbeat) resolves
+//! the member's current cell.
+//!
+//! **Why retired clients stay in their cells.** A `&Client` handed out before a replacement must stay valid after it,
+//! and cells are never emptied or overwritten, so it does: the retired client remains as an inert husk (its caller ends
+//! its router with a `QUIT`, aborts it only as a last resort, then releases what fred would leave behind with
+//! [`retire`]) until the last clone of the `Pool` drops. No `unsafe` is needed. The cost is bounded by the spare cells:
+//! once they are used up, [`replace_member`] fails with [`BudgetExhausted`] and the member keeps its client; a process
+//! restart resets the budget.
+//!
+//! Clones of a `Pool` share its cells, so every holder sees a replacement at once.
+
+use crate::{
+  clients::{Client, Pool},
+  modules::inner::ClientInner,
+  runtime::RefCount,
+};
+use std::{
+  fmt,
+  sync::{
+    atomic::{AtomicUsize, Ordering},
+    OnceLock,
+  },
+  time::{Duration, Instant},
+};
+use tokio::task::AbortHandle;
+
+/// The fewest spare cells a pool gets, whatever its size.
+pub const MIN_SPARE_CELLS: usize = 4;
+
+/// A pool's member-to-cell map and its spare cells, on `PoolInner::live_edit`.
+pub struct PoolState {
+  /// The spare cells, cell `size + j` being `spares[j]`; each is set at most once and never cleared.
+  spares:     Box<[OnceLock<Client>]>,
+  /// Per logical member, the cell it currently names.
+  current:    Box<[AtomicUsize]>,
+  /// The next spare cell [`replace_member`] fills.
+  next_spare: AtomicUsize,
+}
+
+impl PoolState {
+  /// The state of a pool of `size` members: member `i` names cell `i`, and `max(size, 4)` spare cells are empty.
+  pub(crate) fn new(size: usize) -> Self {
+    PoolState {
+      spares:     (0 .. size.max(MIN_SPARE_CELLS)).map(|_| OnceLock::new()).collect(),
+      current:    (0 .. size).map(AtomicUsize::new).collect(),
+      next_spare: AtomicUsize::new(0),
+    }
+  }
+}
+
+impl fmt::Debug for PoolState {
+  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
+    f.debug_struct("PoolState")
+      .field("spares", &self.spares.len())
+      .field("spares_used", &self.next_spare.load(Ordering::Acquire).min(self.spares.len()))
+      .finish()
+  }
+}
+
+/// [`replace_member`] found no empty spare cell: the member keeps its client.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub struct BudgetExhausted;
+
+impl fmt::Display for BudgetExhausted {
+  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
+    f.write_str("every spare cell of the pool is used")
+  }
+}
+
+impl std::error::Error for BudgetExhausted {}
+
+/// The client in cell `cell` of a pool whose original clients are `clients`.
+fn cell<'a>(clients: &'a [Client], state: &'a PoolState, cell: usize) -> Option<&'a Client> {
+  match cell.checked_sub(clients.len()) {
+    None => clients.get(cell),
+    Some(spare) => state.spares.get(spare).and_then(OnceLock::get),
+  }
+}
+
+/// Member `i`'s current client (`i` modulo the pool size). A cell is published only after it is filled, so the
+/// fallback to the original client is unreachable.
+pub(crate) fn member<'a>(clients: &'a [Client], state: &'a PoolState, i: usize) -> &'a Client {
+  let i = i % clients.len();
+  let named = state.current[i].load(Ordering::Acquire);
+  cell(clients, state, named).unwrap_or(&clients[i])
+}
+
+/// Every member's current client, in member order.
+pub(crate) fn members<'a>(clients: &'a [Client], state: &'a PoolState) -> impl Iterator<Item = &'a Client> + 'a {
+  (0 .. clients.len()).map(move |i| member(clients, state, i))
+}
+
+/// Point member `i` of `pool` at `fresh`, an already-connected client, and return the client it named before (to be
+/// retired by the caller: a `QUIT`, an abort of its router only as a last resort, then [`retire`]). New commands go to
+/// `fresh` from the moment this returns; a `&Client` taken before stays valid (see the module docs). Nothing in flight
+/// on the retired client moves.
+///
+/// Fails with [`BudgetExhausted`], changing nothing, when every spare cell is used. Panics if `i` is not a member.
+pub fn replace_member(pool: &Pool, i: usize, fresh: Client) -> Result<Client, BudgetExhausted> {
+  let (clients, state) = (&pool.inner.clients, &pool.inner.live_edit);
+  assert!(i < clients.len(), "member {i} of a pool of {}", clients.len());
+  let spare = state
+    .next_spare
+    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |next| {
+      (next < state.spares.len()).then_some(next + 1)
+    })
+    .map_err(|_| BudgetExhausted)?;
+  // The spare cell is this call's alone (the counter handed it out once), so the set cannot find it filled.
+  let _ = state.spares[spare].set(fresh);
+  let previous = state.current[i].swap(clients.len() + spare, Ordering::AcqRel);
+  Ok(cell(clients, state, previous).unwrap_or(&clients[i]).clone())
+}
+
+/// How many more replacements `pool` can make before [`replace_member`] fails with [`BudgetExhausted`].
+pub fn spare_cells_left(pool: &Pool) -> usize {
+  let state = &pool.inner.live_edit;
+  state
+    .spares
+    .len()
+    .saturating_sub(state.next_spare.load(Ordering::Acquire))
+}
+
+/// How long [`retire`] waits for the router it follows to finish before it releases the client anyway.
+pub const RETIRE_ROUTER_WAIT: Duration = Duration::from_secs(5);
+
+/// How long [`retire`] gives the backchannel's close (`Backchannel::clear_router_state`) before it drops it.
+///
+/// The close is one socket's: take the backchannel's async write lock, take the connection out, flush whatever is
+/// buffered on it, then shut it down — and fred puts no timeout on any of it (`ExclusiveConnection::disconnect`). On a
+/// healthy socket it takes well under a millisecond. It waits in two places:
+///
+/// - **for the lock**, while another holder has it: the client's router (its cluster-state sync in
+///   `router/clustered.rs`, and the unblock of a blocked connection), or a backchannel command sent on the client
+///   (`CLIENT UNBLOCK`, `FUNCTION KILL`, and the like), which first `PING`s an existing backchannel connection with no
+///   timeout (`check_and_create_transport`).
+///   The live-edit service sends no backchannel command on a pool member's client, so in a retirement this wait is
+///   reachable only when the old router is still running as the release starts: [`retire`] got no `router`, or the
+///   router did not finish within [`RETIRE_ROUTER_WAIT`];
+/// - **in the flush**, when the peer stopped reading.
+///
+/// A longer wait helps neither: a holder or a peer that is stuck stays stuck. So 1 s bounds a single socket close with
+/// ample margin, and the release task as a whole ends within [`RETIRE_ROUTER_WAIT`] plus this. What a close cut short
+/// leaves behind depends on where it was cut; see [`retire`].
+pub const RETIRE_BACKCHANNEL_CLOSE_TIMEOUT: Duration = Duration::from_secs(1);
+
+/// Release what a retired or discarded client still holds once its caller has ended its router (a `QUIT`, and an
+/// abort as the last resort), which fred releases only on paths a retirement may not take:
+///
+/// - its **command channel**. A router that ends through fred's normal `QUIT` exit puts the command receiver back on
+///   the client, so the channel stays open with nothing reading it: a later send — a `&Client` borrowed before the
+///   replacement, such as a background walk requesting its next `SCAN` page — would be accepted and wait forever
+///   unless it carries its own timeout, which a scan stream never does. Here the receiver is taken, closed, and every
+///   command still queued in it is cancelled (`RouterCommand::cancel`: the caller gets `Canceled`, a scan stream an
+///   `Err`). A send after that fails at once through fred's closed-channel path, with an `Unknown` error ("Failed to
+///   send command to router." or "Client is not initialized.", by call path) — which the service classifies as
+///   permanent; the only callers that hold a member's `&Client` that
+///   long are background walks, which retry on their next tick;
+/// - its **credential-refresh task**, which fred stops when the router loop exits normally (a `QUIT` that completed)
+///   or when the client's `ClientInner` drops — and a retired client's never drops, because its spare cell keeps it
+///   (and the task holds a clone of it). Left running it would fetch credentials, and log failed refreshes, for the
+///   life of the process;
+/// - its **backchannel connection**, which fred closes after a completed `QUIT` (`check_and_disconnect`) or when the
+///   next router starts (`clear_router_state`) — neither happens to a client whose `QUIT` did not complete and whose
+///   router was aborted, so the husk would keep a socket to a node, with its buffers, until the process exits.
+///
+/// `router` is the abort handle of the router the caller ended or aborted, when it has one. An abort takes effect
+/// only at the router's next await, and a router that has just connected re-arms the credential task with no await
+/// in between, so the release waits (up to [`RETIRE_ROUTER_WAIT`]) until that router has finished; only then is
+/// nothing left that could re-arm the task or put the receiver back.
+///
+/// Does not block: the release runs on a task spawned on the current Tokio runtime, so this can run from a drop
+/// guard. Outside a runtime the parts that need no await run at once. Harmless when any of them is already gone.
+///
+/// The spawned task is detached and bounded: it waits at most [`RETIRE_ROUTER_WAIT`] for the router, and gives the
+/// backchannel's close at most [`RETIRE_BACKCHANNEL_CLOSE_TIMEOUT`]. Dropping that close part-way never corrupts the
+/// backchannel, but what it leaves depends on where it was cut, and the expiry logs one WARN either way, naming the
+/// client:
+///
+/// - **cut short after it took the backchannel's lock**: the connection is already out of the backchannel, so the
+///   drop releases the lock with the backchannel holding no connection and drops the taken `ExclusiveConnection`,
+///   whose socket the OS then closes; only the unflushed bytes of a socket nobody reads again are lost;
+/// - **cut short while waiting for the lock** (see [`RETIRE_BACKCHANNEL_CLOSE_TIMEOUT`] for the holders and when that
+///   can happen in a retirement): the drop just leaves the lock's queue and nothing is taken, so the backchannel
+///   connection stays on the client until the client is dropped. A client kept in a spare cell (a pool member's
+///   retired client) is never dropped before its `Pool`, so that one socket to a node, with its buffers, stays open
+///   until the pool's last clone drops — the leak this release exists to close, left as a known residual for the rare
+///   retirement whose old router outlived the wait. A client kept in no cell (a discarded fresh client, a `txn` slot's
+///   swapped-out client) keeps the socket only until its last clone drops, normally soon after its router ends.
+pub fn retire(client: &Client, router: Option<AbortHandle>) {
+  let inner = client.inner.clone();
+  match tokio::runtime::Handle::try_current() {
+    Ok(runtime) => {
+      runtime.spawn(async move {
+        if let Some(router) = router {
+          let deadline = Instant::now() + RETIRE_ROUTER_WAIT;
+          while !router.is_finished() && Instant::now() < deadline {
+            tokio::time::sleep(Duration::from_millis(10)).await;
+          }
+        }
+        release(&inner);
+        if tokio::time::timeout(
+          RETIRE_BACKCHANNEL_CLOSE_TIMEOUT,
+          inner.backchannel.clear_router_state(&inner),
+        )
+        .await
+        .is_err()
+        {
+          _warn!(
+            inner,
+            "Retired client's backchannel close timed out after {:?} (RETIRE_BACKCHANNEL_CLOSE_TIMEOUT); its \
+             backchannel connection may stay open until the client is dropped (for a pool member, with its pool).",
+            RETIRE_BACKCHANNEL_CLOSE_TIMEOUT
+          );
+        }
+      });
+    },
+    Err(_) => release(&inner),
+  }
+}
+
+/// The parts of [`retire`] that need no await: stop the credential-refresh task, and take, close and drain the command
+/// receiver.
+fn release(inner: &RefCount<ClientInner>) {
+  #[cfg(feature = "credential-provider")]
+  inner.abort_credential_refresh_task();
+  if let Some(mut rx) = inner.take_command_rx() {
+    rx.close();
+    while let Ok(command) = rx.try_recv() {
+      command.cancel();
+    }
+  }
+}
+
+/// How many spare cells `pool` has in all: `max(size, 4)`.
+pub fn spare_cells(pool: &Pool) -> usize {
+  pool.inner.live_edit.spares.len()
+}
--- a/src/live_edit/retention.rs
+++ b/src/live_edit/retention.rs
@@ -0,0 +1,1090 @@
+// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+// SPDX-License-Identifier: Apache-2.0
+//
+// Licensed under the Apache License, Version 2.0 (the "License");
+// you may not use this file except in compliance with the License.
+// You may obtain a copy of the License at
+//
+//     http://www.apache.org/licenses/LICENSE-2.0
+//
+// Unless required by applicable law or agreed to in writing, software
+// distributed under the License is distributed on an "AS IS" BASIS,
+// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+// See the License for the specific language governing permissions and
+// limitations under the License.
+
+//! Connection buffer retention: a cap on what a connection keeps once its traffic drains.
+//!
+//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
+//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
+//!
+//! # Why
+//!
+//! A stock connection has four holders whose allocations only ever grow, so what it keeps is the largest load it
+//! has ever carried, freed only when the connection itself is replaced:
+//!
+//! - **the in-flight queue** (`Connection::buffer`, 64 slots to start): the router writes a whole pipeline before it
+//!   reads a reply, so the queue grows to the longest pipeline in flight. A slot is [`IN_FLIGHT_SLOT_BYTES`].
+//! - **the write buffer** (tokio-util's `Framed`, 8 KiB to start): a frame is encoded whole, so it grows to the
+//!   largest frame plus what was buffered ahead of it.
+//! - **the read buffer** (8 KiB to start): fred decodes only complete frames, so it grows to hold the largest frame
+//!   read.
+//! - **the router's retry buffer** (`Router::retry_buffer`, empty and unallocated to start): it takes a lost
+//!   connection's in-flight commands for replay.
+//!
+//! Draining a byte buffer does not free its allocation: a flush advances the buffer and a decode splits the frame
+//! off, and the next `reserve` reclaims the whole allocation for reuse.
+//!
+//! A client also holds a backchannel connection (`Backchannel::transport`), opened lazily beside its pipelined
+//! connections: for `CLIENT UNBLOCK` under `Blocking::Interrupt` (from a reader's first interrupt), and for
+//! `CLUSTER SLOTS` at a cluster client's topology sync (re-created at every sync of a healthy client). Its buffers
+//! grow like any other connection's, and nothing in stock fred cuts them back.
+//!
+//! # The rule
+//!
+//! With `Settings::retention` (on `ConnectionConfig::live_edit`) set to a [`RetentionLimits`], each holder is cut
+//! back once its traffic drains. Every check runs in the router task that owns the holder.
+//!
+//! A connection is **drained** when its in-flight queue is empty and its write and read buffers hold nothing (no
+//! unflushed frame, no partial frame). Each client has at most one **warm** connection, the only one allowed to keep
+//! the class limits once drained; every other drained connection is cut to the **idle shape**: an in-flight queue
+//! with no allocation, an unallocated write buffer and a fresh 8 KiB read buffer ([`cut`]).
+//!
+//! The class rules, which every connection gets as its traffic passes and the warm one keeps once drained:
+//!
+//! - **in-flight queue**, after a reply is matched to the command at its head ([`on_reply`]): when its capacity is
+//!   above `in_flight_commands` and at most half that many commands are still queued, it shrinks to
+//!   `in_flight_commands`. The half keeps a queue that hovers near the limit from reallocating on every reply.
+//! - **write buffer**, after a write-and-flush or a flush leaves it empty ([`on_flushed`]): when an encode since it
+//!   was last replaced left more than `write_buffer_bytes` in it, it is replaced by a fresh 8 KiB buffer.
+//! - **read buffer**, when a frame is taken off the connection and the buffer is left empty ([`on_read`]): when a
+//!   frame decoded since it was last replaced was larger than `read_buffer_bytes`, it is replaced by a fresh 8 KiB
+//!   buffer. Every way a frame is taken off a pipelined connection runs the check before the frame's caller is
+//!   answered: the read loop (`live_edit::on_frame`), the reply pop ([`on_reply`], which also covers replies a drain
+//!   reads), and `Connection::read` through its `live_edit::next` hook, which covers the frames fred reads outside
+//!   the read loop, such as a `Transaction`'s EXEC reply.
+//!
+//! The client-wide rules:
+//!
+//! - **the sweep** ([`sweep`]), the first thing the router does on each pass of its loop (`live_edit::before_wait`,
+//!   at the top of `read_or_write`): it elects the warm connection ([`elect`]), applies the class rules to the warm
+//!   connection once drained (a backstop: every read and flush is also checked when it happens), and cuts every other
+//!   drained connection that carried traffic since its last cut ([`action`]). The router passes once per command,
+//!   pipeline or batch of reads and, with the unresponsive check configured, at least once per its interval while it
+//!   waits. The sweep's own work allocates nothing (*The sweep allocates nothing*, below).
+//! - **the cut at the reply pop** ([`on_reply`]): a reply that leaves a non-warm connection drained cuts it there,
+//!   before the reply's caller is answered.
+//! - **retry buffer**, after the router replays it ([`on_retry_replayed`]): once empty, it is replaced by an
+//!   unallocated `VecDeque::new()`, stock fred's starting state. It is used only during a replay, so it keeps
+//!   nothing between replays.
+//! - **backchannel**, in the sweep ([`cut_backchannel`]): once it holds nothing (both buffers empty) and it carried
+//!   anything since its last cut (its `idle` flag is clear), both its buffers are replaced by fresh 8 KiB ones and
+//!   `idle` is set.
+//! - **cluster routing table**, at each rebuild from a `CLUSTER SLOTS` reply ([`own_slot_ranges`]): every string the
+//!   table keeps is copied out of the reply, so the reply's read-buffer allocation is freed when the sync ends.
+//! - **sentinel primary**, when a sentinel names the primary to connect to ([`own_server`]): its host is copied out
+//!   of the sentinel's reply before fred connects, so the sentinel connection's read buffer is not kept alive.
+//!
+//! Only capacity changes. Command order, write attempts, flush points, what is written and when are untouched, and a
+//! byte buffer is replaced only while it holds nothing. A decoded frame holds its own reference to its bytes, so
+//! replacing the read buffer never touches a reply on its way to a caller: the old allocation is freed when the
+//! last frame cut from it is dropped, which a frame kept past its call would prevent. fred keeps two such values, the
+//! cluster routing table and the primary `Server` a sentinel names, and [`own_slot_ranges`] and [`own_server`] make
+//! both own their strings, so on every topology no value fred keeps past a call holds a view into a read buffer.
+//! The cost is one reallocation per drain after traffic above a limit, and on a client with several connections, up
+//! to three small allocations when a cut connection is used again.
+//!
+//! # Why one warm connection, and how it is chosen
+//!
+//! A cluster client holds a connection to every primary, and its traffic reaches all of them over time, so a limit
+//! per connection alone lets a client keep its class limits once per primary. One warm connection per client makes
+//! what a client keeps independent of the primary count: one connection at the class limits, the rest at the idle
+//! shape.
+//!
+//! - **Read progress chooses it.** The warm connection should be the one the client's traffic is using now, so that
+//!   the connection kept ready is the one the next call most likely lands on. The newest read progress (a frame taken
+//!   off the connection, or its read buffer grew) marks it, and it is already recorded per connection by the liveness
+//!   check (`liveness::last_progress`) on every read, so choosing by it adds nothing to the command path. Write
+//!   recency would need a hook in fred's routing path or a clock read per write.
+//! - **Warmth moves only when the warm connection is drained**, and only to a connection with strictly newer read
+//!   progress. A connection is never demoted while a call it carries is in flight, so that call's buffers are left
+//!   alone until it ends, and two busy connections cannot trade warmth back and forth on every frame. A tie keeps it.
+//! - **A client with one connection is always warm**, from the first sweep after the connection is made. That
+//!   connection carries all of the client's traffic, and one connection at the class limits is already what the
+//!   bound allows per client: cutting it would only make every call regrow. So, at rest, the rule changes nothing
+//!   for a standalone or sentinel store, for every single-node client, and for a one-primary cluster. The one
+//!   exception is a reply popped in the same router pass as the reconnect, before any sweep: a fred `Transaction`
+//!   retried after a reconnect (`retry!` → `reconnect_with_policy` → `drain_all`,
+//!   `router/transactions.rs:122-142`) drains on a connection not yet marked warm, and can cut it once. That
+//!   changes allocations only; the next sweep marks it warm.
+//! - **The idle read buffer stays at 8 KiB**, tokio-util's own starting size. tokio-util reserves one byte before
+//!   each read, so an unallocated read buffer would grow by doubling from a few bytes and take several reads for
+//!   even a 1 KiB reply. At 8 KiB, a connection used again reads a reply of up to 8 KiB in one read.
+//! - **The backchannel is taken with `try_write`.** Its lock is a tokio `RwLock` that a caller task holds while it
+//!   uses the backchannel (to interrupt a blocked command, or for a topology sync). The router task must never wait
+//!   on a caller, so the sweep takes the lock only when it is free; a busy lock skips the backchannel for that pass,
+//!   and a later pass tries again.
+//! - **The backchannel is cut after any traffic**, not after traffic above a size. That makes "8 KiB read + 8 KiB
+//!   write" exact by construction: a size threshold would have to prove, from bytes' growth rules, that no frame
+//!   under it can grow the buffer (one from 4 to 8 KiB can double it to 16 KiB behind a setup reply's prefix), and
+//!   that a credential in its HELLO never grows the write buffer. The cost is two 8 KiB allocations per use, and use
+//!   is rare: one per cluster sync, which already opens a new connection, and one per reader interrupt.
+//!
+//! # Why the cluster routing table and the sentinel primary own their strings
+//!
+//! fred parses `CLUSTER SLOTS` without copying: every string in the reply frame, and so each `SlotRange`'s `id`,
+//! primary `host` and TLS server name, is a view into the read-buffer allocation the reply was read into. Those
+//! strings outlive the sync that read them. They are kept by the routing state (in `ClientInner` and in the
+//! router's cache), by every connection-map key and `Connection::server` added at that sync, by the next
+//! backchannel's `server` (`connect_any` takes it from the old routing), and by each `Server` a caller clones from
+//! the routing, such as the live-edit service's stream readers, whose client config keeps their owner's `Server`
+//! for life. Each one would pin a whole read-buffer allocation, however the backchannel's buffer is cut, and the
+//! pins would grow with the primary count and the sync history. [`own_slot_ranges`], called at the end of
+//! `ClusterRouting::rebuild`, copies them out, so the reply's allocation is freed when the sync ends and every
+//! `Server` cloned later shares the small owned strings.
+//!
+//! A sentinel store has the same problem once per connection. fred converts the `SENTINEL GET-MASTER-ADDR-BY-NAME`
+//! reply to the primary's host `Str` without copying (`FromValue for Str`), so the `Server` it builds views the
+//! sentinel connection's read buffer, and it lives on as the primary connection's `server` and `default_host`, its
+//! codec's server, the client's recorded sentinel primary and the backchannel's connection-id key. Every client of
+//! a sentinel store, the live-edit service's stream readers included, discovers its own primary this way.
+//! [`own_server`], called on that `Server` before fred connects to it (`router/sentinel.rs`,
+//! `discover_primary_node`), copies its strings out. The sentinel node list fred refreshes from `SENTINEL sentinels`
+//! is converted through owned `String`s and needs no copy. These two, the routing table and the sentinel primary,
+//! are the only reply strings fred keeps past the call that read them.
+//!
+//! The election, the per-connection decision, the backchannel decision and the two string copies are pure
+//! functions ([`elect`], [`elect_streaming`], [`action`], [`plan`], [`cut_backchannel`], [`own_slot_ranges`],
+//! [`own_server`]), so they can be tested without connections.
+//!
+//! # The sweep allocates nothing
+//!
+//! The sweep runs on every router pass, about once per command on a busy client, so its own work allocates nothing:
+//! no list of the connections, their views or their actions. It walks the router's connections twice, through an
+//! iterator over the router's own collections (`liveness::connections_mut`). The first walk folds the election over
+//! the connections' views ([`elect_streaming`]), keeping in locals only the current warm connection's index and view
+//! and the newest read progress's index and time; the second sets each connection's warm flag from the result and
+//! applies its [`action`], reading its view again. Nothing adds or removes a connection between the walks, so both
+//! visit the connections in the same order, and what the second walk does to one connection changes no other
+//! connection's view. The decisions are [`plan`]'s: [`elect`] is the same fold over a slice, and [`plan`] then
+//! applies [`action`] to each connection with its elected flag, as the second walk does.
+//!
+//! The only allocations in a sweep are those its cuts and class rules make: a cut's fresh read buffer, a class rule's
+//! shrunk in-flight queue or replacement byte buffer, and a backchannel cut's two fresh buffers. A pass that changes
+//! no buffer, which is every pass of a client at rest once its connections are cut back, allocates nothing.
+//!
+//! # Byte buffers are judged by what they held, never by `capacity()`
+//!
+//! `BytesMut::capacity()` does not show how large a drained buffer's allocation is: `advance` (a flush) and
+//! `split_to` (a decode) lower it by the bytes consumed while the allocation stays whole, for the next `reserve` to
+//! reclaim. A drained write buffer whose allocation is 16 MiB reports a capacity near zero, so a check keyed on
+//! `capacity()` would never fire. The codec instead records, per connection, the write buffer's length after each
+//! encode and the size of each decoded frame ([`on_encode`], [`on_decode`]): what the allocation grew to hold.
+//!
+//! # What a drained connection keeps
+//!
+//! A limit caps what a buffer may have *held* before it is replaced, not its allocation, which `BytesMut` grows by
+//! at least doubling. [`RetentionLimits::kept_queue_bytes`], [`RetentionLimits::kept_write_bytes`] and
+//! [`RetentionLimits::kept_read_bytes`] bound what the warm connection keeps once drained; every other connection
+//! keeps the idle shape, [`FRESH_BUFFER_BYTES`] of read buffer; a replayed retry buffer keeps nothing; a backchannel
+//! keeps two [`FRESH_BUFFER_BYTES`] buffers once the sweep has seen it hold nothing. The bounds are derived from fred
+//! 10.1.0, tokio-util 0.7.19
+//! (`FramedImpl`) and bytes 1.12.1 (`BytesMut::split_to` and `BytesMut::reserve`), and must be re-derived when any
+//! of them changes.
+//!
+//! # Test support
+//!
+//! With fred's `live-edit-retention-report` feature, which only the live-edit service's `test-support` feature
+//! enables, `test_support` lets a test see and break the warm rule on a running client:
+//!
+//! - **a report** per client: `request_report` makes a request, and the first router pass whose sweep began after
+//!   it answers right after that sweep (`live_edit::before_wait`, which reads the pending request before the sweep)
+//!   with one `ConnectionReport` per connection (its flags, read progress, in-flight command count and buffer
+//!   capacities); `take_report` collects it. A report shows exactly what that pass's sweep left, and a sweep mode
+//!   set before the request is the mode that sweep ran. Its byte-buffer capacities are `BytesMut::capacity()`, so
+//!   they are exact only for a buffer nothing has been consumed from, which is what a cut leaves; the rule itself
+//!   still never reads `capacity()` (above).
+//! - **sweep modes** per client (`set_sweep_for_test`): `Skipped` models a build without the sweep (no warm
+//!   connection, every drained connection cut at its reply pop), `EveryConnectionWarm` the per-connection rule alone
+//!   (every connection keeps the class limits), and `AsBuilt`, the default, the sweep as built.
+//! - **sweep counters** per client (`sweep_stats`): how many sweeps ran and the wall time they took, in nanoseconds,
+//!   counted from the client's creation in every sweep mode, for a test that reports what the sweep costs per pass.
+//!   A client without limits runs no sweep and counts none.
+//! - **write-check counters** per client (`write_check_stats`): how many checks before a write
+//!   (`liveness::before_socket_write`, `liveness::before_write_to`, which run only on a clustered client) ran, the
+//!   wall time of their decisions in nanoseconds, and how many went on to flush, with those flushes' time, counted
+//!   from the client's creation, for a test that reports what the check costs per write. Unlike the sweep counters
+//!   these count whether or not the client has limits.
+//! - **an in-sweep flag** per thread (`in_sweep`): true only while a sweep runs on the calling thread. The sweep is
+//!   synchronous, so the flag is set and cleared on the router's thread within one pass; a test's counting allocator
+//!   reads it to count the allocations made inside the sweep. It is a const-initialised thread-local `Cell<bool>`,
+//!   so reading or setting it never allocates.
+//!
+//! None of it exists in a build without the feature, the release build included: no state on the client, no check,
+//! clock read or thread-local in the router's pass or in its writes.
+
+use crate::{
+  live_edit::liveness,
+  modules::inner::ClientInner,
+  protocol::{
+    codec::Codec,
+    command::Command,
+    connection::{Connection, ConnectionKind},
+    types::{Server, SlotRange},
+  },
+  router::Router,
+  runtime::RefCount,
+};
+use bytes::BytesMut;
+use bytes_utils::Str;
+use std::{
+  collections::VecDeque,
+  sync::atomic::{AtomicBool, AtomicUsize, Ordering},
+  time::Instant,
+};
+
+/// The bytes one in-flight queue or retry buffer slot takes: one `Command`.
+pub const IN_FLIGHT_SLOT_BYTES: usize = std::mem::size_of::<Command>();
+
+/// The size of a replacement read buffer, and of the backchannel's buffers: tokio-util's `Framed` starts both buffers
+/// at 8 KiB (`INITIAL_CAPACITY`, `codec/framed_impl.rs:25`), so a replaced buffer is the same as a new connection's.
+pub const FRESH_BUFFER_BYTES: usize = 8 * 1024;
+
+/// tokio-util's write backpressure boundary (`FramedImpl::poll_ready`, `codec/framed_impl.rs:262-268`): a write
+/// starts only while less than this is buffered; at or past it, a feed first flushes the buffer (the liveness
+/// module's flush before a write that can reach a socket reads it too).
+pub(crate) const BACKPRESSURE_BYTES: usize = 8 * 1024;
+
+/// The largest capacity bytes 1.12.1 carries over when a shared buffer reallocates (`original_capacity_from_repr`,
+/// `MAX_ORIGINAL_CAPACITY_WIDTH = 17`).
+const MAX_CARRIED_CAPACITY: usize = 64 * 1024;
+
+/// What one connection may keep once its traffic drains, on `ConnectionConfig::live_edit.retention`. Read once per
+/// connection, when its codec is created; a reconnect reads it again.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub struct RetentionLimits {
+  /// The in-flight queue's capacity, in commands, once at most half that many are queued.
+  pub in_flight_commands: usize,
+  /// The most the write buffer may have held since it was last replaced without being replaced once it drains.
+  pub write_buffer_bytes: usize,
+  /// The largest frame the read buffer may have held since it was last replaced without being replaced once it
+  /// drains.
+  pub read_buffer_bytes:  usize,
+}
+
+impl RetentionLimits {
+  /// The most a drained in-flight queue keeps: `in_flight_commands` slots. Once the queue is empty its next check
+  /// shrinks any larger capacity to the limit.
+  pub const fn kept_queue_bytes(&self) -> usize {
+    self.in_flight_commands * IN_FLIGHT_SLOT_BYTES
+  }
+
+  /// An upper bound on the allocation a drained write buffer keeps when nothing it held exceeded
+  /// `write_buffer_bytes`: `2 × (write_buffer_bytes + 8 KiB)`. An encode reserves the frame's length once. A
+  /// reserve that does not fit grows the allocation to the larger of twice itself and what it needs: the contents
+  /// plus a consumed prefix that is either reclaimed first or shorter than what is already buffered, which is under
+  /// 8 KiB when a write starts (tokio-util flushes once 8 KiB is buffered).
+  pub const fn kept_write_bytes(&self) -> usize {
+    2 * (self.write_buffer_bytes + BACKPRESSURE_BYTES)
+  }
+
+  /// An upper bound on the allocation a drained read buffer keeps when no frame it held exceeded
+  /// `read_buffer_bytes`: just under `4 × max(read_buffer_bytes, 64 KiB)`.
+  ///
+  /// The decoder takes each frame off the buffer with `split_to`, which makes the allocation shared and leaves the
+  /// consumed frame behind as an offset into it, and tokio-util reserves one byte before each read. A `reserve` on
+  /// the buffer then takes one of two branches:
+  ///
+  /// - **unique** (every frame cut from the allocation was dropped): it reuses the space after the contents, or
+  ///   copies the partial frame to the front when the consumed prefix is at least as long as it; it allocates only
+  ///   when the partial frame is longer than the prefix, and then at twice the old allocation, which the prefix
+  ///   and the partial frame filled, so at under four times the partial frame;
+  /// - **shared** (a frame cut from it is still held): it allocates the larger of the partial frame plus one byte
+  ///   and the buffer's original capacity, which bytes carries over only up to 64 KiB, and grows from there by the
+  ///   unique rule.
+  ///
+  /// A partial frame is shorter than its frame, so either branch leaves the allocation under four times the larger
+  /// of the largest frame and 64 KiB, however the frames before it were sized or held.
+  pub const fn kept_read_bytes(&self) -> usize {
+    let held = if self.read_buffer_bytes > MAX_CARRIED_CAPACITY {
+      self.read_buffer_bytes
+    } else {
+      MAX_CARRIED_CAPACITY
+    };
+    4 * held
+  }
+}
+
+/// Per-connection retention state, on the connection's codec (`Codec::live_edit`): the limits, the largest
+/// write-buffer length and decoded frame since the matching buffer was last replaced, and the warm and idle flags.
+/// Created with the codec, dropped with it; a reconnect starts fresh (not warm, not idle).
+///
+/// It lives on the codec because only the codec sees each encode's and decode's size, and the connection reaches it
+/// through its transport. The fields are atomics only because a codec must be `Sync`; they are touched by one task
+/// at a time: the router task that owns the connection or, for a backchannel, whichever task holds its lock.
+#[derive(Debug, Default)]
+pub struct CodecState {
+  limits:        Option<RetentionLimits>,
+  largest_write: AtomicUsize,
+  largest_frame: AtomicUsize,
+  /// Whether this is its client's warm connection, the one that keeps the class limits once drained. Set by the
+  /// sweep's election.
+  warm:          AtomicBool,
+  /// Whether the connection is in the idle shape (for a backchannel: fresh 8 KiB buffers) and has carried nothing
+  /// since: set by a cut, cleared by every encode and decode, so a connection is cut only if it carried something
+  /// since its last cut.
+  idle:          AtomicBool,
+}
+
+/// The retention state for a new connection of the client `inner`.
+pub(crate) fn codec_state(inner: &RefCount<ClientInner>) -> CodecState {
+  CodecState {
+    limits: inner.connection.live_edit.retention,
+    ..CodecState::default()
+  }
+}
+
+fn raise(mark: &AtomicUsize, value: usize) {
+  if value > mark.load(Ordering::Relaxed) {
+    mark.store(value, Ordering::Relaxed);
+  }
+}
+
+/// The codec encoded a frame, leaving `buffered` bytes in the write buffer.
+pub(crate) fn on_encode(codec: &Codec, buffered: usize) {
+  if codec.live_edit.limits.is_some() {
+    raise(&codec.live_edit.largest_write, buffered);
+    codec.live_edit.idle.store(false, Ordering::Relaxed);
+  }
+}
+
+/// The codec decoded a frame of `size` bytes.
+pub(crate) fn on_decode(codec: &Codec, size: usize) {
+  if codec.live_edit.limits.is_some() {
+    raise(&codec.live_edit.largest_frame, size);
+    codec.live_edit.idle.store(false, Ordering::Relaxed);
+  }
+}
+
+/// `$body` with `$framed` bound to the transport's `Framed`, whose type differs per transport kind.
+macro_rules! with_framed {
+  ($transport:expr, $framed:ident => $body:expr) => {
+    match $transport {
+      ConnectionKind::Tcp(ref mut t) => {
+        let $framed = t.get_mut();
+        $body
+      },
+      #[cfg(feature = "unix-sockets")]
+      ConnectionKind::Unix(ref mut t) => {
+        let $framed = t.get_mut();
+        $body
+      },
+      #[cfg(any(feature = "enable-rustls", feature = "enable-rustls-ring"))]
+      ConnectionKind::Rustls(ref mut t) => {
+        let $framed = t.get_mut();
+        $body
+      },
+      #[cfg(feature = "enable-native-tls")]
+      ConnectionKind::NativeTls(ref mut t) => {
+        let $framed = t.get_mut();
+        $body
+      },
+    }
+  };
+}
+
+/// What the warm election and the cut decision read about one connection.
+#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
+pub struct ConnectionView {
+  /// Whether the connection is its client's warm connection now.
+  pub warm:           bool,
+  /// Whether the connection is in the idle shape and has carried nothing since its last cut.
+  pub idle:           bool,
+  /// Commands in its in-flight queue.
+  pub queued:         usize,
+  /// Bytes in its write buffer (fed but unflushed frames).
+  pub write_buffered: usize,
+  /// Bytes in its read buffer (a partial frame).
+  pub read_buffered:  usize,
+  /// Its newest read progress, if any.
+  pub last_progress:  Option<Instant>,
+}
+
+impl ConnectionView {
+  /// Whether the connection is drained: nothing queued, nothing in either buffer.
+  pub fn drained(&self) -> bool {
+    self.queued == 0 && self.write_buffered == 0 && self.read_buffered == 0
+  }
+}
+
+/// What the sweep does to one connection.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub enum Action {
+  /// Nothing: the connection is not drained, or is already in the idle shape and carried nothing since.
+  Keep,
+  /// Apply the class rules: the drained warm connection.
+  ClassRules,
+  /// Cut it to the idle shape: a drained connection that is not warm and carried something since its last cut.
+  Cut,
+}
+
+/// The index of the warm connection among a client's `connections`, given each one's current state: the one
+/// connection of a one-connection client; else the current warm connection `W`, unless `W` is drained and another
+/// connection has strictly newer read progress, which then becomes warm; else, with no `W`, the connection with the
+/// newest read progress (the first of a tie). `None` when there is no connection, or no `W` and no connection has
+/// made read progress yet.
+///
+/// [`elect_streaming`] over the slice: the fold the sweep runs over the router's connections.
+pub fn elect(connections: &[ConnectionView]) -> Option<usize> {
+  elect_streaming(connections.iter().copied())
+}
+
+/// [`elect`] as a fold over a client's connection views in order, keeping only the count, the first connection
+/// marked warm (`W`: its index and view) and the newest read progress (its index and time; the first of a tie). The
+/// sweep folds it over the views of the router's connections as it walks them, so the election allocates nothing.
+/// The index it returns counts `views` from 0, in the order given.
+pub fn elect_streaming(views: impl IntoIterator<Item = ConnectionView>) -> Option<usize> {
+  let mut count = 0;
+  let mut current: Option<(usize, ConnectionView)> = None;
+  let mut newest: Option<(usize, Instant)> = None;
+  for (i, view) in views.into_iter().enumerate() {
+    count = i + 1;
+    if current.is_none() && view.warm {
+      current = Some((i, view));
+    }
+    if let Some(progress) = view.last_progress {
+      if newest.map_or(true, |(_, newest)| progress > newest) {
+        newest = Some((i, progress));
+      }
+    }
+  }
+  if count <= 1 {
+    return if count == 0 { None } else { Some(0) };
+  }
+  match current {
+    None => newest.map(|(n, _)| n),
+    Some((w, current)) => match newest {
+      Some((n, progress)) if n != w && current.drained() && Some(progress) > current.last_progress => Some(n),
+      _ => Some(w),
+    },
+  }
+}
+
+/// What the sweep does to one connection whose `warm` flag reflects the election.
+pub fn action(view: &ConnectionView) -> Action {
+  if !view.drained() {
+    Action::Keep
+  } else if view.warm {
+    Action::ClassRules
+  } else if view.idle {
+    Action::Keep
+  } else {
+    Action::Cut
+  }
+}
+
+/// The sweep's decisions for one client's `connections`: the warm connection ([`elect`]) and, with each connection's
+/// `warm` flag set from it, what is done to each ([`action`]). The sweep makes the same two passes over the router's
+/// connections without collecting either result: the election's fold, then each connection's action as it is
+/// reached.
+pub fn plan(connections: &[ConnectionView]) -> (Option<usize>, Vec<Action>) {
+  let warm = elect(connections);
+  let actions = connections
+    .iter()
+    .enumerate()
+    .map(|(i, view)| elected_action(view, warm == Some(i)))
+    .collect();
+  (warm, actions)
+}
+
+/// [`action`] for a connection with its `warm` flag set from the election: `warm` is whether it was elected.
+fn elected_action(view: &ConnectionView, warm: bool) -> Action {
+  action(&ConnectionView { warm, ..*view })
+}
+
+/// Whether the sweep replaces a backchannel's read and write buffers with fresh [`FRESH_BUFFER_BYTES`] ones: when it
+/// holds nothing (`read_buffered` and `write_buffered` both 0) and it carried anything since its last cut (`idle`
+/// clear: a cut sets it, every encode and decode clears it, and a new connection starts with it clear).
+pub fn cut_backchannel(read_buffered: usize, write_buffered: usize, idle: bool) -> bool {
+  read_buffered == 0 && write_buffered == 0 && !idle
+}
+
+/// Make every string `ranges` keeps an owned copy: each range's `id`, its primary's `host` and TLS server name, and,
+/// with the `replicas` feature, the same strings of each replica.
+///
+/// fred parses `CLUSTER SLOTS` without copying, so each of these strings is a view into the read-buffer allocation
+/// the reply was read into, and anything that keeps one (the routing state, connection keys, a backchannel's
+/// `server`, a `Server` a caller cloned from the routing) keeps that whole allocation alive. Called at the end of
+/// `ClusterRouting::rebuild`, the only place a running client builds its routing table, so the reply's allocation
+/// is freed when its sync ends. The contents are unchanged, so every comparison, hash and lookup is unchanged.
+pub fn own_slot_ranges(ranges: &mut [SlotRange]) {
+  for range in ranges.iter_mut() {
+    range.id = owned(&range.id);
+    own_server(&mut range.primary);
+    #[cfg(feature = "replicas")]
+    for replica in range.replicas.iter_mut() {
+      own_server(replica);
+    }
+  }
+}
+
+/// An owned copy of `value`, in an allocation of its own.
+fn owned(value: &Str) -> Str {
+  Str::from(String::from(&**value))
+}
+
+/// Make the strings `server` keeps owned copies: its `host` and, with TLS, its TLS server name. The contents are
+/// unchanged, so every comparison, hash and lookup is unchanged.
+///
+/// Used for each `Server` a cluster routing table keeps ([`own_slot_ranges`]), and called on the primary a sentinel
+/// names (`router/sentinel.rs`, `discover_primary_node`) before fred connects to it. fred converts the
+/// `SENTINEL GET-MASTER-ADDR-BY-NAME` reply to a `Str` without copying, so that host is a view into the sentinel
+/// connection's read buffer; the `Server` built from it lives on as the primary connection's `server` and
+/// `default_host`, its codec's server, the client's recorded sentinel primary and the backchannel's connection-id
+/// key, each of which would keep that whole allocation alive for the connection's life.
+pub fn own_server(server: &mut Server) {
+  server.host = owned(&server.host);
+  #[cfg(any(
+    feature = "enable-rustls",
+    feature = "enable-native-tls",
+    feature = "enable-rustls-ring"
+  ))]
+  if let Some(ref name) = server.tls_server_name {
+    server.tls_server_name = Some(owned(name));
+  }
+}
+
+/// The view of `conn` the decisions read.
+fn view(conn: &mut Connection) -> ConnectionView {
+  let queued = conn.buffer.len();
+  let last_progress = liveness::last_progress(conn);
+  with_framed!(conn.transport, framed => {
+    let state = &framed.codec().live_edit;
+    ConnectionView {
+      warm: state.warm.load(Ordering::Relaxed),
+      idle: state.idle.load(Ordering::Relaxed),
+      queued,
+      write_buffered: framed.write_buffer().len(),
+      read_buffered: framed.read_buffer().len(),
+      last_progress,
+    }
+  })
+}
+
+/// The limits of `conn`'s client, when it has any.
+fn limits_of(conn: &mut Connection) -> Option<RetentionLimits> {
+  with_framed!(conn.transport, framed => framed.codec().live_edit.limits)
+}
+
+fn set_warm(conn: &mut Connection, warm: bool) {
+  with_framed!(conn.transport, framed => framed.codec().live_edit.warm.store(warm, Ordering::Relaxed))
+}
+
+/// Shrink `queue` to `limit` slots when its capacity is above the limit and at most half the limit is queued.
+fn shrink_queue(queue: &mut VecDeque<Command>, limit: usize) {
+  if queue.capacity() > limit && queue.len() <= limit / 2 {
+    queue.shrink_to(limit);
+  }
+}
+
+/// Apply the class rules to `conn`: the queue rule, the read-buffer rule and the write-buffer rule.
+fn class_rules(conn: &mut Connection, limits: RetentionLimits) {
+  shrink_queue(&mut conn.buffer, limits.in_flight_commands);
+  on_read(conn);
+  on_flushed(conn);
+}
+
+/// Cut a drained `conn` to the idle shape: its in-flight queue's allocation freed, an unallocated write buffer and a
+/// fresh 8 KiB read buffer. The caller has checked that it is drained.
+fn cut(conn: &mut Connection) {
+  conn.buffer.shrink_to(0);
+  with_framed!(conn.transport, framed => {
+    *framed.write_buffer_mut() = BytesMut::new();
+    *framed.read_buffer_mut() = BytesMut::with_capacity(FRESH_BUFFER_BYTES);
+    let state = &framed.codec().live_edit;
+    state.largest_write.store(0, Ordering::Relaxed);
+    state.largest_frame.store(0, Ordering::Relaxed);
+    state.idle.store(true, Ordering::Relaxed);
+  })
+}
+
+/// The first statement of each router pass (`live_edit::before_wait`): elect the client's warm connection, apply
+/// the class rules to it once drained, cut every other drained connection that carried something since its last
+/// cut, and cut the backchannel's buffers when it qualifies.
+///
+/// Two walks over the router's connections (`liveness::connections_mut`), with no collection of them: the first
+/// folds the election over their views ([`elect_streaming`]), the second sets each one's warm flag and applies its
+/// [`action`]. Nothing adds or removes a connection between the walks, so both visit them in the same order and the
+/// election's index names the same connection in each; what the second walk does to one connection changes no other
+/// connection's view. Apart from the buffers a cut or a class rule installs, the sweep allocates nothing (the module
+/// docs' *The sweep allocates nothing*).
+pub(crate) fn sweep(inner: &RefCount<ClientInner>, router: &mut Router) {
+  let limits = match inner.connection.live_edit.retention {
+    Some(limits) => limits,
+    None => return,
+  };
+  // Test-only: mark this thread as inside the sweep and time the pass, until the guard drops as `sweep` returns.
+  #[cfg(feature = "live-edit-retention-report")]
+  let _pass = test_support::SweepPass::begin(inner);
+  #[cfg(feature = "live-edit-retention-report")]
+  match test_support::sweep_mode(inner) {
+    test_support::SweepForTest::AsBuilt => {},
+    test_support::SweepForTest::Skipped => {
+      for conn in liveness::connections_mut(router) {
+        set_warm(conn, false);
+      }
+      return;
+    },
+    test_support::SweepForTest::EveryConnectionWarm => {
+      for conn in liveness::connections_mut(router) {
+        set_warm(conn, true);
+        if action(&ConnectionView {
+          warm: true,
+          ..view(conn)
+        }) == Action::ClassRules
+        {
+          class_rules(conn, limits);
+        }
+      }
+      sweep_backchannel(inner);
+      return;
+    },
+  }
+  let warm = elect_streaming(liveness::connections_mut(router).map(view));
+  for (i, conn) in liveness::connections_mut(router).enumerate() {
+    let elected = warm == Some(i);
+    let seen = view(conn);
+    set_warm(conn, elected);
+    match elected_action(&seen, elected) {
+      Action::Keep => {},
+      Action::ClassRules => class_rules(conn, limits),
+      Action::Cut => cut(conn),
+    }
+  }
+  sweep_backchannel(inner);
+}
+
+/// Cut the backchannel's buffers to fresh 8 KiB ones when [`cut_backchannel`] says so, and mark it idle; a held lock
+/// skips it for this pass.
+fn sweep_backchannel(inner: &RefCount<ClientInner>) {
+  let mut guard = match inner.backchannel.transport.try_write() {
+    Ok(guard) => guard,
+    Err(_) => return,
+  };
+  if let Some(ref mut transport) = *guard {
+    with_framed!(transport.transport, framed => {
+      let idle = framed.codec().live_edit.idle.load(Ordering::Relaxed);
+      if cut_backchannel(framed.read_buffer().len(), framed.write_buffer().len(), idle) {
+        *framed.read_buffer_mut() = BytesMut::with_capacity(FRESH_BUFFER_BYTES);
+        *framed.write_buffer_mut() = BytesMut::with_capacity(FRESH_BUFFER_BYTES);
+        let state = &framed.codec().live_edit;
+        state.largest_write.store(0, Ordering::Relaxed);
+        state.largest_frame.store(0, Ordering::Relaxed);
+        state.idle.store(true, Ordering::Relaxed);
+      }
+    })
+  }
+}
+
+/// A reply was matched to the command popped from the head of `conn`'s in-flight queue: apply the queue rule and the
+/// read-buffer rule (this also covers replies read outside the read loop, by a drain), then, when the pop left a
+/// connection that is not warm drained, cut it before the reply's caller is answered.
+pub(crate) fn on_reply(conn: &mut Connection) {
+  if let Some(limits) = limits_of(conn) {
+    shrink_queue(&mut conn.buffer, limits.in_flight_commands);
+    on_read(conn);
+    if action(&view(conn)) == Action::Cut {
+      cut(conn);
+    }
+  }
+}
+
+/// A frame was taken off `conn`: replace its read buffer when it is empty and held a frame above the limit.
+pub(crate) fn on_read(conn: &mut Connection) {
+  with_framed!(conn.transport, framed => {
+    let state = &framed.codec().live_edit;
+    let over = match state.limits {
+      Some(limits) => state.largest_frame.load(Ordering::Relaxed) > limits.read_buffer_bytes,
+      None => false,
+    };
+    if over && framed.read_buffer().is_empty() {
+      framed.codec().live_edit.largest_frame.store(0, Ordering::Relaxed);
+      *framed.read_buffer_mut() = BytesMut::with_capacity(FRESH_BUFFER_BYTES);
+    }
+  })
+}
+
+/// A write-and-flush or a flush completed on `conn`: replace its write buffer when it is empty and held more than
+/// the limit.
+pub(crate) fn on_flushed(conn: &mut Connection) {
+  with_framed!(conn.transport, framed => {
+    let state = &framed.codec().live_edit;
+    let over = match state.limits {
+      Some(limits) => state.largest_write.load(Ordering::Relaxed) > limits.write_buffer_bytes,
+      None => false,
+    };
+    if over && framed.write_buffer().is_empty() {
+      framed.codec().live_edit.largest_write.store(0, Ordering::Relaxed);
+      *framed.write_buffer_mut() = BytesMut::with_capacity(FRESH_BUFFER_BYTES);
+    }
+  })
+}
+
+/// The router replayed its retry buffer after a reconnect: once it is empty, return it to stock's unallocated
+/// `VecDeque::new()`. A replay that stopped on a write error leaves its remaining commands in it, and then it is
+/// left as is.
+pub(crate) fn on_retry_replayed(inner: &RefCount<ClientInner>, retry_buffer: &mut VecDeque<Command>) {
+  if inner.connection.live_edit.retention.is_some() && retry_buffer.is_empty() {
+    *retry_buffer = VecDeque::new();
+  }
+}
+
+/// Test support: a per-client report of each connection's retention state, test sweep modes, per-client sweep and
+/// write-check counters and a per-thread in-sweep flag. Compiled only with the `live-edit-retention-report` feature
+/// (see the module docs' *Test support*).
+#[cfg(feature = "live-edit-retention-report")]
+pub mod test_support {
+  use super::{limits_of, liveness, view, Connection, ConnectionKind, RetentionLimits};
+  use crate::{interfaces::ClientLike, modules::inner::ClientInner, router::Router, runtime::RefCount};
+  use std::{
+    cell::Cell,
+    sync::{
+      atomic::{AtomicU64, AtomicU8, Ordering},
+      Mutex,
+      PoisonError,
+      TryLockError,
+    },
+    time::Instant,
+  };
+
+  /// How a client's sweep runs, set per client with [`set_sweep_for_test`]. Each mode other than `AsBuilt` models a
+  /// broken warm rule, so a test can show its check tells the two apart.
+  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
+  pub enum SweepForTest {
+    /// The sweep as built.
+    #[default]
+    AsBuilt,
+    /// The sweep clears every connection's warm flag and does nothing else: no class rules, no cuts, no backchannel.
+    /// The state a build without the sweep reaches: no warm connection, and every drained connection cut at its
+    /// next reply pop.
+    Skipped,
+    /// The sweep marks every connection warm, applies the class rules to each drained one and sweeps the
+    /// backchannel: the per-connection rule alone, with no single warm connection per client.
+    EveryConnectionWarm,
+  }
+
+  impl SweepForTest {
+    fn from_u8(value: u8) -> Self {
+      match value {
+        1 => SweepForTest::Skipped,
+        2 => SweepForTest::EveryConnectionWarm,
+        _ => SweepForTest::AsBuilt,
+      }
+    }
+
+    fn as_u8(self) -> u8 {
+      match self {
+        SweepForTest::AsBuilt => 0,
+        SweepForTest::Skipped => 1,
+        SweepForTest::EveryConnectionWarm => 2,
+      }
+    }
+  }
+
+  /// One connection of a client, as the router saw it right after a sweep.
+  #[derive(Clone, Debug, PartialEq, Eq)]
+  pub struct ConnectionReport {
+    /// The connection's server, `host:port`.
+    pub server:         String,
+    /// The limits on the connection's codec: its client's class limits, or `None` without limits.
+    pub limits:         Option<RetentionLimits>,
+    /// Whether it is its client's warm connection.
+    pub warm:           bool,
+    /// Whether it is in the idle shape and has carried nothing since its last cut.
+    pub idle:           bool,
+    /// Whether it is drained: nothing queued, nothing in either buffer.
+    pub drained:        bool,
+    /// Whether it has made read progress (a frame taken off it, or its read buffer grew) since it connected.
+    pub read_progress:  bool,
+    /// The commands in its in-flight queue (`Connection::buffer`, the commands written and not yet answered) when
+    /// the report was taken: exact (`VecDeque::len`).
+    pub queued:         usize,
+    /// The in-flight queue's capacity, in commands (`VecDeque::capacity`): exact.
+    pub queue_capacity: usize,
+    /// The write buffer's `BytesMut::capacity()`. Exact only for a buffer nothing has been consumed from, such as
+    /// the idle shape's unallocated `BytesMut::new()` (0); otherwise a lower bound on the allocation, since a flush
+    /// lowers `capacity()` by the bytes it consumed while the allocation stays whole (the module docs' *Byte
+    /// buffers are judged by what they held*).
+    pub write_capacity: usize,
+    /// The read buffer's `BytesMut::capacity()`. Exact only for a buffer nothing has been consumed from, such as
+    /// the idle shape's fresh `with_capacity(FRESH_BUFFER_BYTES)` (8,192); otherwise a lower bound on the
+    /// allocation, since a decode lowers `capacity()` by the frame it split off.
+    pub read_capacity:  usize,
+  }
+
+  /// The per-client report state, on `live_edit::ClientState`: the request count, the request last answered, the
+  /// report slot, the sweep mode, and the sweep and write-check counters. Created with the client's `ClientInner` and
+  /// dropped with it.
+  ///
+  /// Requests are numbered rather than flagged so that a request made while a pass is between its pending check and
+  /// its answer is not answered by that pass, whose sweep began before it.
+  #[derive(Debug, Default)]
+  pub struct ReportState {
+    /// How many reports have been requested: [`request_report`] adds one, under the slot's lock, with `Release`.
+    requests:     AtomicU64,
+    /// The request the router last answered. Written only by the client's router.
+    answered:     AtomicU64,
+    report:       Mutex<Option<Vec<ConnectionReport>>>,
+    mode:         AtomicU8,
+    /// How many sweeps the client's routers ran. Written only by the router, as each sweep ends ([`SweepPass`]).
+    sweep_passes: AtomicU64,
+    /// The wall time those sweeps took, in nanoseconds. Written with `sweep_passes`.
+    sweep_nanos:  AtomicU64,
+    /// How many write checks the client's routers ran ([`WriteCheck`]). Written only by the router, as each check
+    /// ends.
+    write_checks: AtomicU64,
+    /// The wall time of those checks' decisions, in nanoseconds. Written with `write_checks`.
+    check_nanos:  AtomicU64,
+    /// How many of those checks went on to flush. Written with `write_checks`.
+    flushes:      AtomicU64,
+    /// The wall time of those flushes, in nanoseconds. Written with `write_checks`.
+    flush_nanos:  AtomicU64,
+  }
+
+  /// A client's sweep counters ([`sweep_stats`]): how many sweeps its routers ran since the client was created, and
+  /// the wall time they took. A test takes two readings and divides the differences for the time per pass.
+  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
+  pub struct SweepStats {
+    /// Sweeps run, in every sweep mode.
+    pub passes: u64,
+    /// Their total wall time, in nanoseconds: from just after the sweep finds the client's limits to its return,
+    /// the backchannel's cut included.
+    pub nanos:  u64,
+  }
+
+  /// A client's write-check counters ([`write_check_stats`]): how many checks before a write
+  /// (`liveness::before_socket_write`, `liveness::before_write_to`) its routers ran on a clustered client since the
+  /// client was created, the wall time of their decisions, and how many went on to flush, with those flushes' time. A
+  /// test takes two readings and divides the differences for the time per check.
+  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
+  pub struct WriteCheckStats {
+    /// Checks run: every call on a clustered client, the early returns included (no route, a write that cannot
+    /// reach the socket, no other connection holding fed frames).
+    pub checks:      u64,
+    /// Their decisions' total wall time, in nanoseconds: from the check's first line to its return, or, for a check
+    /// that flushes, to the start of that flush. Routing, the write predicate and the walk of the other connections
+    /// are all inside it; the flush is not.
+    pub nanos:       u64,
+    /// Checks that went on to flush every connection holding fed frames.
+    pub flushes:     u64,
+    /// Those flushes' total wall time, in nanoseconds: finding the destination's name, then the concurrent flush
+    /// with its socket waits.
+    pub flush_nanos: u64,
+  }
+
+  /// One write check in progress, held by `liveness::before_socket_write` and `liveness::before_write_to` from their
+  /// first line (each runs only on a clustered client): reads the clock when made. [`WriteCheck::flushing`] ends the
+  /// decision's time where the check goes on to flush; when dropped, as the check returns or its future is dropped,
+  /// it counts the check, its decision's time and, when it flushed, the flush and its time ([`write_check_stats`]).
+  /// Neither end allocates.
+  pub(crate) struct WriteCheck<'a> {
+    state:         &'a ReportState,
+    started:       Instant,
+    flush_started: Option<Instant>,
+  }
+
+  impl<'a> WriteCheck<'a> {
+    /// Begin a write check on the client `inner`.
+    pub(crate) fn begin(inner: &'a RefCount<ClientInner>) -> Self {
+      WriteCheck {
+        state:         &inner.live_edit.retention,
+        started:       Instant::now(),
+        flush_started: None,
+      }
+    }
+
+    /// The check decided to flush: its decision ends here, and the flush's time starts.
+    pub(crate) fn flushing(&mut self) {
+      let now = Instant::now();
+      add_nanos(&self.state.check_nanos, now.saturating_duration_since(self.started));
+      self.flush_started = Some(now);
+    }
+  }
+
+  impl Drop for WriteCheck<'_> {
+    fn drop(&mut self) {
+      match self.flush_started {
+        Some(flush_started) => {
+          add_nanos(&self.state.flush_nanos, flush_started.elapsed());
+          self.state.flushes.fetch_add(1, Ordering::Relaxed);
+        },
+        None => add_nanos(&self.state.check_nanos, self.started.elapsed()),
+      }
+      self.state.write_checks.fetch_add(1, Ordering::Relaxed);
+    }
+  }
+
+  fn add_nanos(counter: &AtomicU64, took: std::time::Duration) {
+    counter.fetch_add(u64::try_from(took.as_nanos()).unwrap_or(u64::MAX), Ordering::Relaxed);
+  }
+
+  thread_local! {
+    /// Whether a sweep is running on this thread ([`in_sweep`]). Const-initialised and without a destructor, so
+    /// reading or setting it never allocates or registers anything, even from inside an allocator.
+    static IN_SWEEP: Cell<bool> = const { Cell::new(false) };
+  }
+
+  /// Whether a sweep is running on the calling thread: true from just after the sweep finds the client's limits to
+  /// its return, on the router's thread, and false everywhere else. The sweep is synchronous, so no other task runs
+  /// on that thread in between. For a test's counting global allocator, which reads it on every allocation to count
+  /// those the sweep makes; it never allocates or panics.
+  pub fn in_sweep() -> bool {
+    IN_SWEEP.try_with(Cell::get).unwrap_or(false)
+  }
+
+  /// One sweep in progress, held by `sweep` from just after it finds the client's limits: sets the in-sweep flag
+  /// ([`in_sweep`]) and reads the clock when made; when dropped, as `sweep` returns or unwinds, counts the pass and
+  /// its time on the client ([`sweep_stats`]) and clears the flag. Neither end allocates.
+  pub(crate) struct SweepPass<'a> {
+    state:   &'a ReportState,
+    started: Instant,
+  }
+
+  impl<'a> SweepPass<'a> {
+    /// Begin a sweep of the client `inner`.
+    pub(crate) fn begin(inner: &'a RefCount<ClientInner>) -> Self {
+      IN_SWEEP.with(|flag| flag.set(true));
+      SweepPass {
+        state:   &inner.live_edit.retention,
+        started: Instant::now(),
+      }
+    }
+  }
+
+  impl Drop for SweepPass<'_> {
+    fn drop(&mut self) {
+      let nanos = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
+      self.state.sweep_nanos.fetch_add(nanos, Ordering::Relaxed);
+      self.state.sweep_passes.fetch_add(1, Ordering::Relaxed);
+      IN_SWEEP.with(|flag| flag.set(false));
+    }
+  }
+
+  fn state<C: ClientLike>(client: &C) -> &ReportState {
+    &client.inner().live_edit.retention
+  }
+
+  /// Ask `client`'s router for a report: clears any report not yet taken and makes a new request, which supersedes
+  /// any earlier one still pending. Take the answer with [`take_report`].
+  ///
+  /// The answer comes from the first router pass whose sweep began after this call, and shows what that sweep left.
+  /// So a mode set with [`set_sweep_for_test`] before this call is the mode that sweep ran: the router reads the
+  /// request (`Acquire`) before its sweep reads the mode, and this call's `Release` store comes after the mode's. A
+  /// pass that finds the report slot held by a test thread leaves the request for the next pass, whose sweep also
+  /// began after it, so the router never waits on a test thread.
+  ///
+  /// The router passes at least once per unresponsive-check interval while it waits, so a request is answered within
+  /// about one interval when the check is configured, with or without traffic. A router that has ended never answers.
+  ///
+  /// Pass a client (a `Client`, a pool member, a `SubscriberClient`), never a `Pool`: a `Pool`'s `ClientLike::inner`
+  /// picks a member round-robin, so the request and the take could reach different clients.
+  pub fn request_report<C: ClientLike>(client: &C) {
+    let state = state(client);
+    let mut slot = state.report.lock().unwrap_or_else(PoisonError::into_inner);
+    *slot = None;
+    state.requests.fetch_add(1, Ordering::Release);
+  }
+
+  /// The report `client`'s router stored for the last [`request_report`], if it has answered; taking it empties the
+  /// slot. Pass a client, never a `Pool` (see [`request_report`]).
+  pub fn take_report<C: ClientLike>(client: &C) -> Option<Vec<ConnectionReport>> {
+    state(client).report.lock().unwrap_or_else(PoisonError::into_inner).take()
+  }
+
+  /// `client`'s sweep counters ([`SweepStats`]), read now. They count from the client's creation, across
+  /// reconnects and in every sweep mode, so a test reads them before and after the work it measures. Pass a client,
+  /// never a `Pool` (see [`request_report`]).
+  pub fn sweep_stats<C: ClientLike>(client: &C) -> SweepStats {
+    let state = state(client);
+    SweepStats {
+      passes: state.sweep_passes.load(Ordering::Relaxed),
+      nanos:  state.sweep_nanos.load(Ordering::Relaxed),
+    }
+  }
+
+  /// `client`'s write-check counters ([`WriteCheckStats`]), read now. They count from the client's creation, across
+  /// reconnects, so a test reads them before and after the work it measures. Pass a client, never a `Pool` (see
+  /// [`request_report`]).
+  pub fn write_check_stats<C: ClientLike>(client: &C) -> WriteCheckStats {
+    let state = state(client);
+    WriteCheckStats {
+      checks:      state.write_checks.load(Ordering::Relaxed),
+      nanos:       state.check_nanos.load(Ordering::Relaxed),
+      flushes:     state.flushes.load(Ordering::Relaxed),
+      flush_nanos: state.flush_nanos.load(Ordering::Relaxed),
+    }
+  }
+
+  /// Set how `client`'s sweep runs ([`SweepForTest`]); it stays until set again, across reconnects, for the life of
+  /// the client. Every sweep that begins after a later [`request_report`] call runs under it, so a report requested
+  /// after this call shows a sweep in this mode; a pass already running may still sweep under the previous one. Pass a
+  /// client, never a `Pool` (see [`request_report`]).
+  pub fn set_sweep_for_test<C: ClientLike>(client: &C, mode: SweepForTest) {
+    state(client).mode.store(mode.as_u8(), Ordering::Release);
+  }
+
+  /// The sweep mode of the client `inner`, read at the top of `sweep`.
+  pub(crate) fn sweep_mode(inner: &RefCount<ClientInner>) -> SweepForTest {
+    SweepForTest::from_u8(inner.live_edit.retention.mode.load(Ordering::Acquire))
+  }
+
+  /// The request pending at the top of a router pass (`live_edit::before_wait`, before the sweep), if any. Its
+  /// `Acquire` load makes every mode set before that request visible to the sweep that follows. Two atomic loads and
+  /// no allocation when none is pending.
+  pub(crate) fn pending_report(inner: &RefCount<ClientInner>) -> Option<u64> {
+    let state = &inner.live_edit.retention;
+    let request = state.requests.load(Ordering::Acquire);
+    (request != state.answered.load(Ordering::Relaxed)).then_some(request)
+  }
+
+  /// Answer `request`, which [`pending_report`] found before this pass's sweep (`live_edit::before_wait`, right after
+  /// the sweep): one [`ConnectionReport`] per connection of the router. Left pending for the next pass when a test
+  /// thread holds the report slot, or when a newer request was made since the pass began (its sweep began before that
+  /// request).
+  pub(crate) fn answer_report(inner: &RefCount<ClientInner>, router: &mut Router, request: u64) {
+    let state = &inner.live_edit.retention;
+    let mut slot = match state.report.try_lock() {
+      Ok(slot) => slot,
+      Err(TryLockError::WouldBlock) => return,
+      Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
+    };
+    // Requests are counted under this lock, so this read sees the latest one.
+    if state.requests.load(Ordering::Relaxed) != request {
+      return;
+    }
+    let report = liveness::connections_mut(router).map(report_of).collect();
+    *slot = Some(report);
+    state.answered.store(request, Ordering::Relaxed);
+  }
+
+  fn report_of(conn: &mut Connection) -> ConnectionReport {
+    let seen = view(conn);
+    let limits = limits_of(conn);
+    let server = conn.server.to_string();
+    let queue_capacity = conn.buffer.capacity();
+    let (write_capacity, read_capacity) =
+      with_framed!(conn.transport, framed => (framed.write_buffer().capacity(), framed.read_buffer().capacity()));
+    ConnectionReport {
+      server,
+      limits,
+      warm: seen.warm,
+      idle: seen.idle,
+      drained: seen.drained(),
+      read_progress: seen.last_progress.is_some(),
+      queued: seen.queued,
+      queue_capacity,
+      write_capacity,
+      read_capacity,
+    }
+  }
+}
--- a/src/modules/inner.rs
+++ b/src/modules/inner.rs
@@ -453,7 +453,7 @@
   /// A backchannel that can be used to control the router connections even while the connections are blocked.
   pub backchannel:   RefCount<Backchannel>,
   /// Server state cache for various deployment types.
-  pub server_state:  RwLock<ServerState>,
+  pub server_state:  RwLock<ServerState>, #[doc = "The live-edit additions' per-client state."] pub live_edit: crate::live_edit::ClientState, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 
   /// An mpsc sender for commands to the router.
   pub command_tx: RefSwap<RefCount<CommandSender>>,
@@ -530,7 +530,7 @@
 
       backchannel,
       command_rx,
-      server_state,
+      server_state, live_edit: Default::default(), // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       command_tx,
       state,
       counters,
--- a/src/protocol/codec.rs
+++ b/src/protocol/codec.rs
@@ -61,7 +61,7 @@
     res
   );
   log_resp2_frame(&codec.name, &item, true);
-  sample_stats(codec, false, len as i64);
+  sample_stats(codec, false, len as i64); crate::live_edit::on_encode(codec, dst.len()); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 
   Ok(())
 }
@@ -80,7 +80,7 @@
   if let Some((frame, amt, _)) = resp2_decode(src)? {
     trace!("{}: Parsed {} bytes from {}", codec.name, amt, codec.server);
     log_resp2_frame(&codec.name, &frame, false);
-    sample_stats(codec, true, amt as i64);
+    sample_stats(codec, true, amt as i64); crate::live_edit::on_decode(codec, amt); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 
     Ok(Some(protocol_utils::check_resp2_auth_error(codec, frame)))
   } else {
@@ -101,7 +101,7 @@
     res
   );
   log_resp3_frame(&codec.name, &item, true);
-  sample_stats(codec, false, len as i64);
+  sample_stats(codec, false, len as i64); crate::live_edit::on_encode(codec, dst.len()); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 
   Ok(())
 }
@@ -118,7 +118,7 @@
   }
 
   if let Some((frame, amt, _)) = resp3_decode(src)? {
-    sample_stats(codec, true, amt as i64);
+    sample_stats(codec, true, amt as i64); crate::live_edit::on_decode(codec, amt); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 
     if codec.streaming_state.is_some() && frame.is_streaming() {
       return Err(Error::new(
@@ -186,7 +186,7 @@
   pub name:            Str,
   pub server:          Server,
   pub resp3:           RefCount<AtomicBool>,
-  pub streaming_state: Option<StreamedFrame<Resp3Frame>>,
+  pub streaming_state: Option<StreamedFrame<Resp3Frame>>, #[doc = "The live-edit additions' per-connection codec state."] pub live_edit: crate::live_edit::CodecState, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   #[cfg(feature = "metrics")]
   pub req_size_stats:  RefCount<RwLock<MovingStats>>,
   #[cfg(feature = "metrics")]
@@ -199,7 +199,7 @@
       server:                                     server.clone(),
       name:                                       inner.id.clone(),
       resp3:                                      inner.shared_resp3(),
-      streaming_state:                            None,
+      streaming_state:                            None, live_edit: crate::live_edit::codec_state(inner), // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       #[cfg(feature = "metrics")]
       req_size_stats:                             inner.req_size_stats.clone(),
       #[cfg(feature = "metrics")]
--- a/src/protocol/connection.rs
+++ b/src/protocol/connection.rs
@@ -29,7 +29,7 @@
   pin::Pin,
   str,
   task::{Context, Poll},
-  time::{Duration, Instant},
+  time::Duration, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 };
 use tokio_util::codec::Framed;
 
@@ -716,7 +716,7 @@
       version,
       counters,
       id,
-      last_write: None,
+      live_edit: Default::default(), // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       transport: self.transport,
       blocked: false,
       #[cfg(feature = "replicas")]
@@ -739,7 +739,7 @@
   pub version:      Option<Version>,
   pub id:           Option<i64>,
   pub counters:     Counters,
-  pub last_write:   Option<Instant>,
+  pub live_edit:    crate::live_edit::ConnectionState, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   pub blocked:      bool,
   #[cfg(feature = "replicas")]
   pub replica:      bool,
@@ -800,21 +800,21 @@
     check_unresponsive: bool,
   ) -> Result<(), Error> {
     if check_unresponsive {
-      self.last_write = Some(Instant::now());
+      crate::live_edit::on_write(self); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     }
 
     if flush {
       self.counters.reset_feed_count();
-      self.transport.send(frame.into()).await
+      crate::live_edit::send(self, frame.into()).await // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     } else {
       self.counters.incr_feed_count();
-      self.transport.feed(frame.into()).await
+      crate::live_edit::feed(self, frame.into()).await // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     }
   }
 
   /// Put a command at the back of the in-flight command buffer.
   pub fn push_command(&mut self, mut cmd: Command) {
-    if cmd.has_no_responses() {
+    if crate::live_edit::on_push(self, &cmd) { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       cmd.respond_to_caller(Ok(Resp3Frame::Null));
     } else {
       if cmd.blocks_connection() {
@@ -829,8 +829,8 @@
   /// This function is not cancel-safe.
   #[inline(always)]
   pub async fn read(&mut self) -> Result<Option<Resp3Frame>, Error> {
-    match self.transport.next().await {
-      Some(f) => f.map(|f| Some(f.into_resp3())),
+    match crate::live_edit::next(self).await? { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
+      Some(f) => f.map(|f| Some(crate::live_edit::into_resp3(&*self, f))), // live-edit patch
       None => Ok(None),
     }
   }
@@ -846,7 +846,7 @@
 
       if let Some(err) = responses::check_fatal_errors(inner, &self.server, &frame) {
         return Err(err);
-      } else if let Some(frame) = responses::check_pubsub_message(inner, &self.server, frame) {
+      } else if let Some(frame) = responses::check_pubsub_message(inner, &self.server, frame, crate::live_edit::classify(self)) { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
         return Ok(Some(frame));
       } else {
         continue;
@@ -865,7 +865,7 @@
 
       if let Some(err) = responses::check_fatal_errors(inner, &self.server, &frame) {
         return Err(err);
-      } else if let Some(frame) = responses::check_pubsub_message(inner, &self.server, frame) {
+      } else if let Some(frame) = responses::check_pubsub_message(inner, &self.server, frame, crate::live_edit::classify(self)) { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
         if is_clustered {
           clustered::process_response_frame(inner, self, frame)?;
         } else {
@@ -893,7 +893,7 @@
   /// Flush the sink and reset the feed counter.
   pub async fn flush(&mut self) -> Result<(), Error> {
     trace!("Flushing socket to {}", self.server);
-    self.transport.flush().await?;
+    crate::live_edit::flush(self).await?; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     self.counters.reset_feed_count();
     Ok(())
   }
@@ -903,7 +903,7 @@
   /// Returns the in-flight commands that had not received a response.
   pub async fn close(&mut self) -> VecDeque<Command> {
     let _ = utils::timeout(
-      self.transport.close(),
+      crate::live_edit::close(self), // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       Duration::from_millis(CONNECTION_CLOSE_TIMEOUT_MS),
     )
     .await;
--- a/src/protocol/types.rs
+++ b/src/protocol/types.rs
@@ -556,7 +556,7 @@
     self.data = cluster::parse_cluster_slots(cluster_slots, default_host)?;
     self.data.sort_by(|a, b| a.start.cmp(&b.start));
 
-    cluster::modify_cluster_slot_hostnames(inner, &mut self.data, default_host);
+    cluster::modify_cluster_slot_hostnames(inner, &mut self.data, default_host); crate::live_edit::retention::own_slot_ranges(&mut self.data); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     Ok(())
   }
 
--- a/src/router/centralized.rs
+++ b/src/router/centralized.rs
@@ -26,7 +26,7 @@
 ) -> Result<(), Error> {
   _trace!(inner, "Parsing response frame from {}", conn.server);
   let mut command = match conn.buffer.pop_front() {
-    Some(command) => command,
+    Some(command) => crate::live_edit::on_reply(inner, conn, command), // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     None => {
       _debug!(
         inner,
@@ -101,7 +101,7 @@
       };
       let mut transport = connection::create(inner, &server, None).await?;
       transport.setup(inner, None).await?;
-      let connection = transport.into_pipelined(false);
+      let connection = crate::live_edit::pipelined(inner, transport, false); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       inner.notifications.broadcast_reconnect(server);
 
       *writer = Some(connection);
--- a/src/router/clustered.rs
+++ b/src/router/clustered.rs
@@ -295,7 +295,7 @@
 ) -> Result<(), Error> {
   _trace!(inner, "Parsing response frame from {}", conn.server);
   let mut command = match conn.buffer.pop_front() {
-    Some(command) => command,
+    Some(command) => crate::live_edit::on_reply(inner, conn, command), // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     None => {
       _debug!(
         inner,
@@ -570,7 +570,7 @@
       _debug!(inner, "Connecting to cluster node {}", server);
       let mut transport = connection::create(&_inner, &server, None).await?;
       transport.setup(&_inner, None).await?;
-      let connection = transport.into_pipelined(false);
+      let connection = crate::live_edit::pipelined(&_inner, transport, false); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       inner.notifications.broadcast_reconnect(server.clone());
       _new_writers.lock().insert(server, connection);
       Ok::<_, Error>(())
--- a/src/router/commands.rs
+++ b/src/router/commands.rs
@@ -138,7 +138,7 @@
 
         (false, None)
       }
-    } else {
+    } else { crate::live_edit::before_socket_write(inner, router, &command).await; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       let conn = match router.route(&command) {
         Some(conn) => conn,
         None => {
@@ -232,7 +232,7 @@
   router: &mut Router,
   commands: Vec<Command>,
 ) -> Result<(), Error> {
-  _debug!(inner, "Writing pipeline with {} commands", commands.len());
+  _debug!(inner, "Writing pipeline with {} commands", commands.len()); let mut abandon = crate::live_edit::abandon::PipelineStep::default(); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 
   for mut command in commands.into_iter() {
     // trying to pipeline `SSUBSCRIBE` is problematic since successful responses arrive out-of-order via pubsub push
@@ -241,10 +241,10 @@
     // never pipeline `ssubscribe`, even if the caller asks.
     command.can_pipeline = command.kind != CommandKind::Ssubscribe;
 
-    write_command_t!(inner, router, command)?;
+    abandon.before_write(router, &command); write_command_t!(inner, router, command)?; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   }
 
-  Ok(())
+  crate::live_edit::abandon::after_pipeline(inner, router, abandon).await // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 }
 
 /// Send `ASKING` to the provided server, then retry the provided command.
@@ -262,7 +262,7 @@
     command.respond_to_caller(Err(e));
     return Ok(());
   }
-  let attempts_remaining = command.attempts_remaining;
+  let attempts_remaining = command.attempts_remaining; if attempts_remaining > 0 { return Box::pin(crate::live_edit::ask::follow(inner, router, server, slot, command)).await; } // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   let asking_result = Box::pin(utils::send_asking_with_policy(
     inner,
     router,
@@ -272,8 +272,8 @@
   ))
   .await;
   if let Err(e) = asking_result {
-    command.respond_to_caller(Err(e.clone()));
-    return Err(e);
+    command.respond_to_caller(Err(e)); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
+    return Ok(());
   }
 
   if let Err(error) = write_command_t!(inner, router, command) {
@@ -456,7 +456,7 @@
   router: &mut Router,
   rx: &mut CommandReceiver,
 ) -> Result<(), Error> {
-  if inner.connection.unresponsive.max_timeout.is_some() {
+  crate::live_edit::before_wait(inner, router, rx).await; if inner.connection.unresponsive.max_timeout.is_some() { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     let sleep_ft = sleep(inner.connection.unresponsive.interval);
     pin!(sleep_ft);
 
@@ -468,7 +468,7 @@
         }
       },
       Some(command) = rx.recv() => {
-        process_command(inner, router, command).await?;
+        crate::live_edit::on_receive(inner, rx); process_command(inner, router, command).await?; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       },
       _ = sleep_ft => {
         // break out and return early, starting another call to poll_next on all the sockets,
@@ -485,7 +485,7 @@
         }
       },
       Some(command) = rx.recv() => {
-        process_command(inner, router, command).await?;
+        crate::live_edit::on_receive(inner, rx); process_command(inner, router, command).await?; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       },
     };
   }
--- a/src/router/connections.rs
+++ b/src/router/connections.rs
@@ -11,7 +11,7 @@
   runtime::RefCount,
   types::config::Server,
 };
-use futures::future::try_join_all;
+// live-edit patch: see vendor/fred-10.1.0.SOURCE.md (stock imported `futures::future::try_join_all` here, for `flush`)
 use semver::Version;
 use std::collections::{HashMap, VecDeque};
 
@@ -351,7 +351,7 @@
       Connections::Clustered {
         connections: ref mut writers,
         ..
-      } => try_join_all(writers.values_mut().map(|writer| writer.flush()))
+      } => crate::live_edit::flush_router_wide(writers.values_mut()) // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
         .await
         .map(|_| ()),
     }
@@ -380,7 +380,7 @@
     {
       let mut transport = connection::create(inner, server, None).await?;
       transport.setup(inner, None).await?;
-      writers.insert(server.clone(), transport.into_pipelined(false));
+      writers.insert(server.clone(), crate::live_edit::pipelined(inner, transport, false)); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       Ok(())
     } else {
       Err(Error::new(ErrorKind::Config, "Expected clustered configuration."))
--- a/src/router/mod.rs
+++ b/src/router/mod.rs
@@ -393,7 +393,7 @@
       }
     }
 
-    let _ = self.flush().await;
+    let _ = self.flush().await; crate::live_edit::on_retry_replayed(inner, &mut self.retry_buffer); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     Ok(())
   }
 
--- a/src/router/replicas.rs
+++ b/src/router/replicas.rs
@@ -284,7 +284,7 @@
           .lock()
           .insert(transport.server.clone(), id);
       }
-      self.connections.insert(replica.clone(), transport.into_pipelined(true));
+      self.connections.insert(replica.clone(), crate::live_edit::pipelined(inner, transport, true)); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     }
 
     self.routing.add(primary, replica);
--- a/src/router/responses.rs
+++ b/src/router/responses.rs
@@ -163,8 +163,8 @@
 /// Check if the frame is part of a pubsub message, and if so route it to any listeners.
 ///
 /// If not then return it to the caller for further processing.
-pub fn check_pubsub_message(inner: &RefCount<ClientInner>, server: &Server, frame: Resp3Frame) -> Option<Resp3Frame> {
-  if is_subscription_response(&frame) {
+pub(crate) fn check_pubsub_message(inner: &RefCount<ClientInner>, server: &Server, frame: Resp3Frame, subscribed: crate::live_edit::Classify) -> Option<Resp3Frame> { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
+  if crate::live_edit::classify_frame(&subscribed, &frame, is_subscription_response(&frame)) && is_subscription_response(&frame) { // live-edit patch
     _debug!(inner, "Dropping unused subscription response.");
     return None;
   }
@@ -173,8 +173,8 @@
     return None;
   }
 
-  let is_pubsub =
-    frame.is_normal_pubsub_message() || frame.is_pattern_pubsub_message() || frame.is_shard_pubsub_message();
+  let is_pubsub = crate::live_edit::pubsub_classified(&subscribed, &frame) // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
+    && (frame.is_normal_pubsub_message() || frame.is_pattern_pubsub_message() || frame.is_shard_pubsub_message());
   if !is_pubsub {
     return Some(frame);
   }
@@ -295,15 +295,15 @@
 /// Check for special errors, pubsub messages, or other special response frames.
 ///
 /// The frame is returned to the caller for further processing if necessary.
-pub fn preprocess_frame(
+pub(crate) fn preprocess_frame( // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   inner: &RefCount<ClientInner>,
   server: &Server,
-  frame: Resp3Frame,
+  frame: Resp3Frame, subscribed: crate::live_edit::Classify, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 ) -> Result<Option<Resp3Frame>, Error> {
   if let Some(error) = check_fatal_errors(inner, server, &frame) {
     Err(error)
   } else {
-    Ok(check_pubsub_message(inner, server, frame))
+    Ok(check_pubsub_message(inner, server, frame, subscribed)) // live-edit patch
   }
 }
 
--- a/src/router/sentinel.rs
+++ b/src/router/sentinel.rs
@@ -225,7 +225,7 @@
       ))]
       tls_server_name: None,
     }
-  };
+  }; let mut server = server; crate::live_edit::retention::own_server(&mut server); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 
   let mut transport = stry!(connection::create(inner, &server, None).await);
   stry!(transport.setup(inner, None).await);
@@ -307,7 +307,7 @@
     .update_sentinel_nodes(&transport.server, sentinels);
   let _ = update_sentinel_backchannel(inner, &transport).await;
 
-  *writer = Some(transport.into_pipelined(false));
+  *writer = Some(crate::live_edit::pipelined(inner, transport, false)); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   Ok(())
 }
 
--- a/src/router/transactions.rs
+++ b/src/router/transactions.rs
@@ -118,7 +118,7 @@
   let mut redirections = 0;
   let mut asking: Option<(Server, u16)> = None;
 
-  'outer: loop {
+  'outer: loop { if crate::live_edit::abandon::given_up(&commands) { return crate::live_edit::abandon::refuse_transaction(inner, id, tx); } // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     macro_rules! retry {
       ($err:expr) => {{
         attempted += 1;
@@ -142,7 +142,7 @@
     if let Err(err) = router.drain_all(inner).await {
       _debug!(inner, "Error draining router before transaction: {:?}", err);
       retry!(None);
-    }
+    } if crate::live_edit::abandon::given_up(&commands) { return crate::live_edit::abandon::refuse_transaction(inner, id, tx); } // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     // find the server that should receive the transaction
     let conn = match asking.as_ref() {
       Some((server, _)) => match router.get_connection_mut(server) {
@@ -171,14 +171,14 @@
         },
       };
 
-      if let Err(err) = conn.write(frame, true, false).await {
+      if crate::live_edit::abandon::given_up(&commands) { return crate::live_edit::abandon::refuse_transaction(inner, id, tx); } if let Err(err) = conn.write(frame, true, false).await { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
         _debug!(inner, "Error sending trx command: {:?}", err);
         retry!(Some(err));
       }
     }
 
     // write all the commands before EXEC
-    for command in commands.iter_mut() {
+    for (abandon_index, command) in commands.iter_mut().enumerate() { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       let (frame, _) = match utils::prepare_command(inner, &conn.counters, command) {
         Ok(frame) => frame,
         Err(err) => {
@@ -186,14 +186,14 @@
           return Ok(());
         },
       };
-      if let Err(err) = conn.write(frame, true, false).await {
+      if crate::live_edit::abandon::command_given_up(command) { if asking.is_none() && abandon_index == 0 { return crate::live_edit::abandon::refuse_transaction(inner, id, tx); } let server = conn.server.clone(); return crate::live_edit::abandon::close_transaction(inner, router, server, id, tx).await; } if let Err(err) = conn.write(frame, true, false).await { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
         _debug!(inner, "Error sending trx command: {:?}", err);
         discard_retry!(conn, Some(err));
       }
     }
 
     // send EXEC and process all the responses
-    match exec(inner, conn, expected).await {
+    if crate::live_edit::abandon::given_up(&commands) { let server = conn.server.clone(); return crate::live_edit::abandon::close_transaction(inner, router, server, id, tx).await; } match exec(inner, conn, expected).await { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       Ok(responses) => match process_responses(responses, abort_on_error) {
         Ok(result) => {
           let _ = tx.send(Ok(result));
--- a/src/router/types.rs
+++ b/src/router/types.rs
@@ -1,5 +1,5 @@
 use crate::{
-  error::{Error, ErrorKind},
+  error::Error, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   modules::inner::ClientInner,
   protocol::{connection::Connection, types::Server},
   runtime::RefCount,
@@ -42,20 +42,20 @@
 ) {
   match Pin::new(&mut conn.transport).poll_next(cx) {
     Poll::Ready(Some(frame)) => {
-      conn.last_write = None;
-      buf.push((conn.server.clone(), Some(frame.map(|f| f.into_resp3()))));
+      crate::live_edit::on_frame(conn, *now); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
+      buf.push((conn.server.clone(), Some(frame.map(|f| crate::live_edit::into_resp3(conn, f))))); // live-edit patch
     },
     Poll::Ready(None) => {
-      conn.last_write = None;
+      crate::live_edit::on_frame(conn, *now); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       buf.push((conn.server.clone(), None));
     },
     Poll::Pending => {
       if let Some(duration) = inner.connection.unresponsive.max_timeout {
-        if let Some(last_write) = conn.last_write {
-          if now.saturating_duration_since(last_write) > duration && !conn.blocked {
+        if let Some(last_write) = crate::live_edit::stall_start(inner, conn, *now) { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
+          if let Some(error) = crate::live_edit::verdict(inner, conn, *now, last_write, duration) { // live-edit patch
             buf.push((
               conn.server.clone(),
-              Some(Err(Error::new(ErrorKind::IO, "Unresponsive connection."))),
+              Some(Err(error)), // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
             ));
           }
         }
@@ -117,7 +117,7 @@
       poll_connection(_self.inner, conn, cx, &mut out, &now);
     }
 
-    if out.is_empty() {
+    crate::live_edit::log_verdicts_all(_self.inner, _self.connections); if out.is_empty() { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       Poll::Pending
     } else {
       Poll::Ready(out)
@@ -167,7 +167,7 @@
       poll_connection(_self.inner, conn, cx, &mut out, &now);
     }
 
-    if out.is_empty() {
+    crate::live_edit::log_verdicts_one(_self.inner, _self.connection); if out.is_empty() { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       Poll::Pending
     } else {
       Poll::Ready(out)
--- a/src/router/utils.rs
+++ b/src/router/utils.rs
@@ -256,7 +256,7 @@
 
   match result {
     Some(Ok(frame)) => {
-      let frame = match responses::preprocess_frame(inner, server, frame) {
+      let frame = match responses::preprocess_frame(inner, server, frame, crate::live_edit::subscribed(router, server)) { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
         Ok(frame) => frame,
         Err(err) => {
           _debug!(inner, "Error reading frame from server {}: {:?}", server, err);
@@ -383,7 +383,7 @@
 pub async fn reconnect_with_policy(inner: &RefCount<ClientInner>, router: &mut Router) -> Result<(), Error> {
   let mut delay = next_reconnection_delay(inner)?;
 
-  loop {
+  loop { crate::live_edit::before_inline_connect(inner, router).await; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     if !delay.is_zero() {
       _debug!(inner, "Sleeping for {} ms.", delay.as_millis());
       read_and_sleep(inner, router, delay).await?;
@@ -557,7 +557,7 @@
 pub async fn sync_cluster_with_policy(inner: &RefCount<ClientInner>, router: &mut Router) -> Result<(), Error> {
   let mut delay = Duration::from_millis(0);
 
-  loop {
+  loop { crate::live_edit::before_inline_connect(inner, router).await; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
     if !delay.is_zero() {
       _debug!(inner, "Sleeping for {} ms.", delay.as_millis());
       read_and_sleep(inner, router, delay).await?;
--- a/src/runtime/_tokio.rs
+++ b/src/runtime/_tokio.rs
@@ -101,7 +101,7 @@
   rx: ReceiverKind<T>,
 }
 
-impl<T: Send + 'static> Receiver<T> {
+impl<T: Send + 'static> Receiver<T> { #[doc = "The number of values queued in the channel."] pub fn len(&self) -> usize { match self.rx { ReceiverKind::Bounded(ref rx) => rx.len(), ReceiverKind::Unbounded(ref rx) => rx.len() } } #[doc = "Refuse further sends, keeping the values already queued."] pub fn close(&mut self) { match self.rx { ReceiverKind::Bounded(ref mut rx) => rx.close(), ReceiverKind::Unbounded(ref mut rx) => rx.close() } } // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   pub async fn recv(&mut self) -> Option<T> {
     match self.rx {
       ReceiverKind::Bounded(ref mut tx) => tx.recv().await,
--- a/src/types/config.rs
+++ b/src/types/config.rs
@@ -419,7 +419,7 @@
   /// Default: `5`
   pub max_redirections:             u32,
   /// Unresponsive connection configuration options.
-  pub unresponsive:                 UnresponsiveConfig,
+  pub unresponsive:                 UnresponsiveConfig, #[doc = "The live-edit additions' settings. Default: stock behaviour."] pub live_edit: crate::live_edit::Settings, // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   /// An unexpected `NOAUTH` error is treated the same as a general connection failure, causing the client to
   /// reconnect based on the [ReconnectPolicy](crate::types::config::ReconnectPolicy). This is [recommended](https://github.com/StackExchange/StackExchange.Redis/issues/1273#issuecomment-651823824) if callers are using ElastiCache.
   ///
@@ -485,7 +485,7 @@
       reconnect_on_auth_error: false,
       disable_cluster_health_check: false,
       tcp: TcpConfig::default(),
-      unresponsive: UnresponsiveConfig::default(),
+      unresponsive: UnresponsiveConfig::default(), live_edit: Default::default(), // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       #[cfg(feature = "replicas")]
       replica: ReplicaConfig::default(),
       #[cfg(feature = "custom-reconnect-errors")]
--- a/src/utils.rs
+++ b/src/utils.rs
@@ -180,7 +180,7 @@
 }
 
 pub fn decr_atomic(size: &AtomicUsize) -> usize {
-  size.fetch_sub(1, Ordering::AcqRel).saturating_sub(1)
+  crate::live_edit::saturating_decr(size) // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
 }
 
 pub fn incr_atomic(size: &AtomicUsize) -> usize {
@@ -399,19 +399,19 @@
   let (tx, rx) = oneshot_channel();
   command.response = ResponseKind::Respond(Some(tx));
 
-  let timed_out = command.timed_out.clone();
+  let timed_out = command.timed_out.clone(); let abandon = crate::live_edit::abandon::Guard::arm(&timed_out); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   let timeout_dur = prepare_command(client, &mut command);
   check_blocking_policy(inner, &command).await?;
   client.send_command(command)?;
 
   if timeout_dur.is_zero() {
-    rx.map_err(move |error| {
+    abandon.watch(rx).map_err(move |error| { // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       set_bool_atomic(&timed_out, true);
       Error::from(error)
     })
     .await?
   } else {
-    timeout(rx, timeout_dur)
+    timeout(abandon.watch(rx), timeout_dur) // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
       .and_then(|r| async { r })
       .map_err(move |error| {
         set_bool_atomic(&timed_out, true);
@@ -456,7 +456,7 @@
   cmd_span.record("cmd.req", &req_size);
 
   let queued_span = trace::create_queued_span(cmd_span.id(), inner);
-  let timed_out = command.timed_out.clone();
+  let timed_out = command.timed_out.clone(); let abandon = crate::live_edit::abandon::Guard::arm(&timed_out); // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   _trace!(
     inner,
     "Setting command trace ID: {:?} for {} ({})",
@@ -471,7 +471,7 @@
   check_blocking_policy(inner, &command).await?;
   client.send_command(command)?;
 
-  let ft = async { rx.instrument(cmd_span).await.map_err(|e| e.into()).and_then(|r| r) };
+  let ft = async { abandon.watch(rx).instrument(cmd_span).await.map_err(|e| e.into()).and_then(|r| r) }; // live-edit patch: see vendor/fred-10.1.0.SOURCE.md
   let result = if timeout_dur.is_zero() {
     ft.await
   } else {
```

The patch edits existing lines only, so every existing file keeps its line
count and every `fred-10.1.0/...:NNN` citation in this repository's source,
tests and docs still points at the same code. New code lives in one new module,
`src/live_edit/` (declared on `src/lib.rs`'s `mod utils;` line), whose module
docs describe it; every hunk that edits an existing file carries a
`live-edit patch` comment on at least one of its added lines. The
module's shared state is one settings field on fred's config
(`ConnectionConfig::live_edit`, default: stock behaviour), one per-client
field (`ClientInner::live_edit`, which also holds change 7's test-only report
state, compiled only with the `live-edit-retention-report` feature), one
per-connection field (`Connection::live_edit`, which replaces stock fred's
`last_write`), one per-pool field (`PoolInner::live_edit`, change 6) and one
field on each connection's codec (`Codec::live_edit`, change 7). The one
edited line outside `src/` is that feature's declaration in fred's
`Cargo.toml` (change 7), which replaces a blank line.

The patch carries these changes:

1. `process_ask` no longer ends the router task (`src/router/commands.rs`).
2. The unresponsive check judges a connection's oldest outstanding command,
   with read progress and peer evidence, instead of its newest write; a fast
   verdict has its own error text, and every verdict records where the stuck
   bytes sit.
3. Every wait in the router on one connection's I/O is bounded, and a
   connection judged dead is closed without waiting on it.
4. RESP2 pubsub-shaped data replies reach their commands.
5. The router never waits idle, nor on a clustered client connects inline or
   makes a write that can reach a socket, while a connection holds fed but
   unflushed frames; every flush of a clustered client's connections taken
   together runs them all at once, each to its own end; and the command-queue
   counter is resynced to the channel at every receive.
6. A `Pool`'s members can be re-pointed to a fresh client.
7. Opt-in limits on what a connection keeps once its traffic drains, with one
   warm connection per client at the limits and every other drained
   connection, and the backchannel, cut to a small idle shape; and the cluster
   routing table and a sentinel-named primary own their strings
   (`src/live_edit/retention.rs`).
8. `process_ask` follows an `-ASK` to its target: a pipelined `ASKING` and
   the command on the target's connection, with nothing read synchronously;
   a script command is never followed (`src/live_edit/ask.rs`).
9. A command, pipeline or transaction is never written after its caller gave
   up on it (its timeout fired, fred answered it with an error, or its future
   was dropped), and a pipeline cut during the router step that writes it, or a
   fred `Transaction` cut partway, never leaves a `MULTI` open
   (`src/live_edit/abandon.rs`).

### 1. `process_ask` no longer ends the router task

In cluster mode, a node answers `-ASK <slot> <node>` while the slot is
migrating and the keys a command names are no longer on the source node, which
includes a key that never existed. fred's router handles the redirect in
`process_ask` (`src/router/commands.rs:251-285`): stock, it sends `ASKING` to
the target node with the command's remaining write attempts, then re-sends the
command (change 8 replaces that half whenever the command has a write attempt
left). When the command has no write attempt left, `send_asking_with_policy`
fails with `Routing` "Max attempts reached." (`src/router/utils.rs:468-469`).
Stock `process_ask` then answers the caller with that error **and also returns
it**. The `?` in `read_or_write` (`:470-471`) propagates it out of the router
loop in `start()` (`:556-566`), and the router task ends: the retry buffer is
cleared, the connections are closed, queued commands are cancelled and the task
exits.

Nothing marks the end. The client's state stays `Connected`, its command
channel stays open, and no event fires that a caller can observe. Every later
command on that client waits for its timeout and fails, until the process
restarts. `force_reconnection()` never answers, because no router is left to
serve it. In a pool, fred keeps routing its share of the traffic to the dead
client, and any command in flight on it when it died fails with a
non-transient `channel closed`.

`process_moved` already handles the same situation correctly: when the
redirect budget is spent it answers the caller and returns `Ok(())`. The patch
does the same in `process_ask`. A failed `ASKING` now fails its own command and
leaves the router running. It also covered an `ASKING` write that failed on a
broken connection once the command's attempts ran out, which ended the router
the same way; since change 8, `send_asking_with_policy` is called only for a
command with no attempt left, which fails before any write.

The service is exposed because several of its commands run with one write
attempt: the `UpdateLayer` commit `EVALSHA`, the operation-id start-record
`HGET` and every stream-reader `XREAD`. A slot migration (a cluster reshard or
a managed-cluster scale-out) can then end a data-pool or reader-pool client for
the life of the pod. Commands at fred's default three attempts were exposed
only when the `ASK` arrived on the third write, which stock fred made certain
on a moved key: it re-sent the command to the source, which answered `-ASK`
until the attempts ran out (change 8).

Evidence, from a mock cluster node that answers `-ASK` for chosen keys, with
clients built the way `ValkeyClient::connect` builds them:

| | Stock 10.1.0 | Patched |
|---|---|---|
| One-attempt `EVALSHA`, `HGET` or `XREAD` gets `-ASK` | the caller gets `Routing: Max attempts reached.`; the router task ends; the client's next command times out after 5 s; `state()` still reads `Connected` | the caller gets the same error; the next command answers in under 1 ms |
| A command in flight on the same client | fails `channel closed` | answered |
| `force_reconnection()` afterwards | never answers | completes |
| `-ASK` with redirections disabled (`max_redirections = 0`) | the command fails `Too many redirections.`; the router survives | unchanged |
| `-MOVED` to a one-attempt command (control) | the command fails `Too many failed write attempts.`; the router survives | unchanged |

That mock is one node whose `-ASK` names itself, so the redirect's target was
also the slot's owner and its only connection was idle: it could not show that
the stock follow re-sends the command to the source, nor that its synchronous
`ASKING` read takes another command's reply. Change 8 records both.

The error the caller sees is unchanged, so the service's error classification
is unaffected. The service also supervises every pooled client's router task
(`src/storage/router_supervisor.rs`) and restarts one that ends any other way;
the patch removes the trigger, the supervisor covers the other paths.

### 2. The unresponsive check

Stock fred stamps `Connection::last_write` on every non-pubsub write
(`src/protocol/connection.rs:802-803`, from `src/router/utils.rs:98,106`),
clears it whenever the read loop takes a frame (`src/router/types.rs:45,49`),
and declares a connection `"Unresponsive connection."` when
`now - last_write > max_timeout` (`:53-62`). That measures the *newest* write:
a connection that keeps receiving writes never trips it, however long its
oldest command has waited, so a socket that went silent under load stays in
rotation. And a reply read outside the read loop (the drain a cluster
credential refresh runs, `src/router/mod.rs:401-436`) never clears the stamp,
so a quiet healthy connection can trip it later.

The patch (`src/live_edit/liveness.rs`, hooked from `src/router/types.rs`)
judges the *stall*: the time since the later of when the oldest outstanding
non-pubsub command reached the socket (while it is still only fed, the later of
its prepare time and the connection's last completed write, feed or flush, so a
large frame uploading ahead of it is not silence) and the connection's last
read progress (a frame decoded or its read buffer grew). `blocked` connections
are not judged. Two verdicts close the connection through fred's existing path
(`src/router/types.rs:55-58`):

- **fast** (opt-in, `ConnectionConfig::live_edit.liveness.fast_timeout`): the
  stall exceeds the fast bound and another connection sharing the same
  `PeerReplies` registry received a reply from the same cluster slot range at
  least `peer_evidence_lag` after the stall began. Every reply stamps its
  command's slot in the registry when it is matched to its command
  (`src/router/centralized.rs:28-29`, `src/router/clustered.rs:297-298`) —
  or its connection's node, when the client's cluster routing does not give
  that slot to the node: a command with nothing to hash, or one sent to a
  named node (`with_cluster_node`, a `SCAN` page) whose first argument hashes
  to another node's slot. Such a command is judged by its node's replies. Its
  error text is `"Unresponsive connection (fast verdict)."`.
- **slow**: the stall exceeds `UnresponsiveConfig::max_timeout`. Its error
  text is stock fred's `"Unresponsive connection."`, which an expired bounded
  wait (change 3) shares.

`live_edit::liveness::verdict_of` classifies an error as a fast or slow
verdict, so a caller never compares the texts. In-flight commands with
attempts left are re-sent on the replacement connection, as on any lost
connection; a command with none fails with the verdict's error.

Each verdict, and each expired bounded wait, logs one WARN (target
`fred::live_edit::liveness`, starting `Unresponsive connection to <node>`)
recording where the stuck bytes sit, so the next occurrence names its cause:
the verdict, the stall and the newest peer reply's age; for the connection,
the in-flight count, the oldest outstanding command (kind, and whether its
frame was sent or is still only fed, with its age), the commands fed since the
last send or flush, the write- and read-buffer bytes, fred's feed count, the
time since the last read progress, whether it subscribed, its lifetime count
of pubsub-shaped frames (counted at `src/router/responses.rs:167`) and the
longest single write, flush or read wait since the stall began (timed in the
bounded-wait wrappers) and the longest time a command sent since the stall
began sat fed before it was sent (the send re-stamps the command's
`network_start`, so this is recorded just before); for the client, the command-queue counter, the command
channel's length when the router last waited and the time since the router
last took a command (`ClientInner::live_edit`, `src/modules/inner.rs:456,533`),
and a summary of each other connection, logged once every connection has been
polled (`src/router/types.rs:120,170`); and the kernel's view of the socket
from `/proc/self/net/tcp` or `tcp6` (`tx_queue`, `rx_queue`, the active timer,
`retrnsmt`), read only at a verdict and `unavailable` without a matching row.
A stall that had peer evidence and recovered by itself logs a rate-limited
INFO "near miss".

### 3. Bounded router waits

Stock fred waits without a deadline on one connection's I/O in several places
inside the router task: the drain before a cluster-wide command, the
broadcast's reply read, the `ASKING` read, a transaction's `EXEC` and skip
reads, and every write and flush. A silent connection there stalls the whole
client. The patch bounds each such wait (`Connection::write`, `flush` and
`read`) by `UnresponsiveConfig::max_timeout`; when the bound expires that
connection fails with an IO error carrying the unresponsive text, into the
caller's existing error path, and the connection is marked so the read loop
closes it too where the caller ignores the error. From then on every write,
feed, flush or read on the marked connection fails at once with the same
error, before any I/O and without a second WARN, until fred closes it: the
cleanup callers run on the connection they just failed on (a transaction's
`DISCARD` through `request_response`, `src/router/transactions.rs:17-27,134-140`;
a re-flush of frames left fed; a drain; an `ASKING` retry) would otherwise wait
again, holding the router meanwhile. The marker is never cleared; it is
dropped with the connection.

The close is the last wait on a connection. Stock `Connection::close`
(`src/protocol/connection.rs:904-912`) runs the transport's close (fred's write
buffer flushed, then the transport shut down, under TLS after a
`close_notify`) under `CONNECTION_CLOSE_TIMEOUT_MS`, 5 s. After an expired
write or flush the bytes still buffered are ones the kernel would not take, so
that close always waits out the 5 s; after a read-loop verdict it waits
whenever the socket's send queue refuses what fred still holds, which on a
fast verdict can push the re-send past the command timeout. The patch routes
that close through a hook (`src/protocol/connection.rs:906`): a connection
judged dead — an expired wait, or a fast or slow verdict the read loop reached,
which marks the connection (`closed_by`, set where the verdict is reached) —
is closed with no flush, no `close_notify` and no shutdown, and every other
connection keeps the stock close. The timeout and the drain around it are
unchanged, so the in-flight commands are returned, re-sent or failed exactly
as before; dropping the connection then closes the socket without blocking.
So one silent connection holds the router for at most one bound per wait
chain, its close included, or per attempt of a transaction. With
`max_timeout: None`, fred's default, nothing is bounded, nothing expires,
nothing is judged dead and every close is stock.

### 4. RESP2 pubsub classification

In RESP2 stock fred treats any reply shaped like a pubsub frame as pubsub on
every connection (`src/router/responses.rs:136-211`, and the RESP2-to-RESP3
conversion, which turns such an array into a `Push` frame at any depth): a
three-element array starting `"message"`, a three- or four-element array
starting `"subscribe"` or `"unsubscribe"`, and so on. A data reply of that
shape (an `SMEMBERS`, `LRANGE` or `HMGET` returning `["message", x, y]`) is
dropped instead of being matched to its command, and every later caller on
the connection then gets the previous caller's reply. The patch classifies
non-`Push` frames as pubsub only on a connection that has sent a subscription
command (`src/protocol/connection.rs:817`), and converts RESP2 frames on every
other connection without the pubsub rewrite.

### 5. Fed frames are flushed before the router waits, connects inline or writes

fred writes a command without flushing it when its command-queue counter
(`ClientInner::counters.cmd_buffer_len`) says another queued command follows,
and relies on that later write to flush (`src/router/utils.rs:56`,
`src/protocol/connection.rs:242-245`, `src/router/commands.rs:164`). Two paths
leave a fed frame unflushed even with a correct counter: a timed-out command
skipped at write flushes only its own connection
(`src/router/utils.rs:71-78`), and the router-wide flush runs only after a
flushed write; a skipped `Reconnect` returns without flushing
(`src/router/commands.rs:357-364`). And a counter that stays above the
channel's real length would hold every write back. Either way the router then
waits idle with the command fed and never sent, until another command arrives.

On a clustered client a frame also waits while the router connects. The router
command that leads to an inline connect is itself queued and counted: a
`Reconnect` deferred after a failed read or write
(`src/router/utils.rs:118-149`), a `SyncCluster`, a `-MOVED`
(`src/router/commands.rs:288-315`), or, with change 8, an `-ASK` to a target
the client has no connection to. So the command routed just ahead of it is
fed, not flushed, and the queued command keeps the channel non-empty, so the
first flush below skips it. The connect then runs on the router task. A
reconnect or sync attempt (`clustered::sync`) keeps every connection without
reader errors, unflushed, and connects the nodes it adds, after a
`CLUSTER SLOTS` whose fallback tries the known nodes one after another
(`src/router/clustered.rs:359-397`, `:518-592`). It polls no connection, so
neither a write nor a read-loop verdict reaches the fed frame, which leaves
only with a flush after the connect: after a successful reconnect or sync,
the retry buffer's (`src/router/mod.rs:396`). With a blackholed node that is
tens of seconds after the frame's caller may have given up, and the frame goes
to a healthy node, which runs it.

And while a frame sits fed, the router can wait on other connections' I/O,
each wait bounded by change 3 at up to 45 s in the service: a command's write
to another connection and the flush after it (`src/router/commands.rs:158`,
`:164-166`), change 8's `ASKING` and command, change 9's `DISCARD`, and the
flush before the router waits, which walked the connections one after another.
With k connections that stopped reading with their kernel send buffers full, a
frame fed on a healthy connection could reach its socket about k × 45 s after
it was fed. Stock's router-wide flush (`Connections::flush`,
`src/router/connections.rs:351-356`) runs its connections together, but under
`try_join_all`, which stops at the first error and drops the flushes it has not
finished, so a connection that failed fast left another's fed frames for the
next flush point. Between router commands the read loop often judged the
healthy connection instead: its fed frame's stall counts from when it was fed,
so after a full bound elsewhere it got a verdict and was closed without a
flush, losing the frame and a healthy connection.

The patch:

- at the top of `read_or_write` (`src/router/commands.rs:459`), when the
  command channel is empty (`Receiver::len`, `src/runtime/_tokio.rs:104`),
  flushes every connection whose feed count or write buffer is not empty,
  through the bounded flush (change 3), all of them at once (the shared flush,
  below). The channel is checked, not the counter: fred's own rule, flush when
  no queued command follows, by construction. Run concurrently, frames left
  fed on several connections (by a counter that drifted high, say, which
  holds back the flush before a write below) leave together, within one
  bound. **Reverting its concurrency**:
  in `live_edit::liveness::before_wait`, replace the `flush_fed_concurrently`
  call with a walk of `connections_mut(router)` that flushes each
  `has_unflushed` connection in turn;
- before every inline connect of a clustered client, flushes every connection
  holding fed but unflushed frames through the same bounded flush, all of them
  at once, whatever the channel holds. All three sites call one entry point,
  `live_edit::before_inline_connect`, which calls
  `live_edit::liveness::flush_fed_concurrently`: the top of every pass of
  `reconnect_with_policy` (`src/router/utils.rs:386`) and
  `sync_cluster_with_policy` (`:560`), before the pass's backoff sleep and its
  attempt, and `live_edit::ask::follow`, before it opens a connection to an
  `-ASK` target (`follow` runs only on a clustered client, so the entry
  point's cluster gate never skips it). Flushing before the sleep,
  not only before the attempt, also lets the sleep's reads take the frame's
  reply, so no verdict closes a healthy connection over a frame that was only
  unsent. Only a clustered client runs it: a centralized or sentinel client's
  attempt closes its one connection before it connects, and that close
  flushes it (or, on a connection judged dead, drops it, change 3), so a flush
  first would only replace the close's 5 s bound with the slow liveness bound.
  Cost: at most one bound per pass or `-ASK` connect, however many
  connections hold fed frames: the same wait fred's router-wide flush after
  a successful attempt (`Router::flush`, `src/router/connections.rs:351-356`)
  would have spent, only earlier. Not covered: the read-polling sleep `delay_cluster_sync`
  makes before a `-MOVED` sync
  (`src/router/utils.rs:546-552`), which the flush does not precede; it
  sleeps for `cluster_cache_update_delay`, whose fred default 0 the service
  keeps, so it never sleeps there (two guard tests in the service's
  `src/storage/client.rs`, run by `scripts/guard-tests.sh`, fail if the pool
  builder or any code under `src/` sets it). Nor are the replica loops
  (`add_replica_with_policy`, `sync_replicas_with_policy`), compiled only with
  the `replicas` feature, which `scripts/check-vendored-fred.sh` keeps out of
  every workspace build (*How the copy is wired in*). **Reverting it**:
  on `src/router/utils.rs:386` and `:560` delete the
  `crate::live_edit::before_inline_connect(inner, router).await;` statement
  and its comment, restoring `  loop {`, and in `live_edit::ask::follow`
  delete the `before_inline_connect` call (and its comment); then
  `before_inline_connect` and `FlushSite::InlineConnect` go;
- before a write that can reach a socket, on a clustered client: when the
  write can reach its connection's socket and another connection holds fed
  but unflushed frames, flushes every connection holding them, the
  destination included, all at once, before the write. The hook is
  `live_edit::before_socket_write` at the top of the single-connection branch
  of `write_command` (`src/router/commands.rs:141`), which every single
  command, every pipeline command, the retry buffer's replay and a `-MOVED`
  retry's write go through; it routes the command (`Router::route`, as the
  write's own) and calls `live_edit::liveness::before_socket_write`. A
  centralized or sentinel client has one connection, so the hook returns at
  once there. The write can reach the socket when fred's own flush rule
  (`src/router/utils.rs:56`) would send it — the command-queue counter at zero
  (`Counters::should_send`, `src/protocol/connection.rs:242-245`, which also
  covers `write_command`'s flush of the destination after a skipped command,
  `src/router/commands.rs:164-166`), the destination's feed count past
  `max_feed_count` (fred's default 200, which the service leaves unset), or a
  command kind fred always sends (`CommandKind::should_flush`,
  `src/protocol/command.rs:1465-1481`: every custom command and script among
  them) — or when it is a feed onto a write buffer at tokio-util's 8 KiB
  backpressure boundary (`FramedImpl::poll_ready`, tokio-util 0.7.19
  `codec/framed_impl.rs:262-268`, change 7's `BACKPRESSURE_BYTES`), which
  flushes that buffer first. Each term is its own function (`queue_empty`,
  `feed_count_full`, `flushes_by_kind`, `at_backpressure`, joined by
  `write_reaches_socket`), so each can be judged and reverted alone. A feed
  below the boundary only encodes into the buffer and cannot block, so it
  triggers nothing, which keeps fred's batching across connections below the
  boundary; and when only the destination holds fed frames nothing is flushed
  first, so its own write batches them as stock does. Every term reads state only the router
  changes, apart from the command-queue counter, which senders only raise
  while the router runs, so a check that finds the write cannot reach the
  socket is never followed by a write that does. The patch's own writes outside
  `write_command` — `live_edit::ask::follow`'s `ASKING` and command, and
  `live_edit::abandon::after_pipeline`'s `DISCARD` — call
  `live_edit::before_write_to` first, which flushes whenever another
  connection holds fed frames, without the predicate, because those paths are
  rare. Cost: per command on a clustered client, one more `Router::route` (a
  slot lookup and a map probe) and a few atomic loads; per write that can
  reach a socket, one walk of the connections (one atomic load and one length
  each), which allocates nothing. The decision's cost per write check — the
  route, the predicate and the walk, averaged over every call on a clustered
  client, early returns included, and not the flush — measured 182 ns over
  all shapes (179 ns pipelined, 267 ns single, 164 ns in bursts) against a
  1,000 ns criterion per shape, as change 7's sweep-cost criterion is: a
  sandbox measurement on 2026-10-16, release build, the service's CI feature
  set, a local three-primary Valkey 9.0.3 cluster on aarch64; CI's x86_64
  runner prints its own figure. The service's
  `write_check_cost_on_a_cluster_is_within_its_budget_bbo2`
  (`tests/integration/storage/write_check_cost_test.rs`, reading change 7's
  test-only counters) asserts the criterion in a release build only, and in
  every build the expected number of pre-flushes on each shape (none for
  pipelines or single commands, one per burst), so unchanged batching is pass
  or fail too for the shapes it measures, all small commands below the
  boundary; it prints the flushes' own I/O (about 33 µs per pre-flush in
  the burst shape, in that run) without asserting it. It runs in the
  `test-cluster-disruptive` CI job's sweep-cost release step, beside change
  7's cost test, on every merge request that changes `src/live_edit/`,
  `src/router/commands.rs` (the hook's call site) or the test. Under the
  counter-at-zero and command-kind terms the write stays a send, and the
  flush sends only what fred's router-wide flush right after it would have
  sent; that flush then finds nothing left. Under the feed-count term the
  flush also sends the destination's fed frames and resets its feed count
  (`Connection::flush`, `src/protocol/connection.rs:894-898`), so every fed
  frame leaves as stock's send and router-wide flush would have sent it, but
  the write itself is then only fed, and leaves at the next flush point. For
  a feed at the 8 KiB boundary, which
  stock follows with no router-wide flush (`write_command` runs it only after
  a write that flushed, `src/router/commands.rs:191-195`), it sends other
  connections' fed frames earlier than stock's next flush point (the next
  send, or the flush before the router waits) would, and large pipelined
  uploads are the writes that reach it. The router waits earlier than stock did: a silent connection
  holding fed frames is waited on, up to one bound, before a write elsewhere,
  where that write could once fail first and a read-loop verdict then drop the
  silent connection with no wait. The router's hold stays at most one bound
  per silent connection in all, because an expiry marks the connection and
  later I/O on it fails at once, but in a multi-node outage the client's
  recovery can take up to one bound longer. In exchange a healthy connection
  whose only outstanding command is a fed frame is no longer judged silent
  after a long wait elsewhere: its frame is sent first, and its reply counts
  as progress. **Reverting it**: on `src/router/commands.rs:141` restore
  `    } else {`, and in `live_edit::ask::follow` and
  `live_edit::abandon::after_pipeline` delete the `before_write_to` call and
  its comment; then `live_edit::before_socket_write` and `before_write_to`,
  their `liveness` counterparts with the write predicate's functions,
  `FlushSite::Write` and the `WriteCheck` counters go, and
  `retention::BACKPRESSURE_BYTES` can be private again. One term reverts
  alone by deleting it from `write_reaches_socket`;
- in fred's router-wide flush (`Connections::flush`, which `Router::flush`
  runs after a successful send, after the retry buffer's replay and at the
  start of `drain_all`), on a clustered client: flushes only the connections
  holding fed frames, all at once and each to its own end, through
  `live_edit::flush_router_wide` (`src/router/connections.rs:354`; the
  stock `try_join_all` import on `:14` becomes a comment, keeping the line
  count), and returns the first error, which the callers log or return as
  before. So a fast error on one connection never drops another's flush; a
  `drain_all` whose flush meets a fast error now waits for the other flushes,
  up to one bound, before it returns that error. Stock allocated on every
  call; this allocates only when two or more connections hold fed frames. Two
  edges: a connection whose bounded wait expired and that has nothing fed is
  skipped, where stock's flush of it failed at once (a `drain_all` still fails
  at once on it when it has commands in flight, and otherwise the read loop
  judges it at the next pass); and an expired connection still holding
  unflushed bytes counts as fed, so each write elsewhere that can reach a
  socket tries its flush again, which fails at once with a DEBUG line, until
  the read loop closes it. Neither adds a wait. A centralized or sentinel
  client keeps stock's flush of its one connection. **Reverting it**: on
  `src/router/connections.rs:14` restore `use futures::future::try_join_all;`
  and on `:354` restore
  `      } => try_join_all(writers.values_mut().map(|writer| writer.flush()))`;
  then `flush_router_wide` and `FlushSite::RouterWide` go;
- runs every flush of a clustered client's connections taken together — the
  flush before the router waits, before an inline connect, before a write and
  fred's router-wide flush — through one function,
  `live_edit::liveness::flush_fed_concurrently`: one bounded flush per
  connection holding fed frames, all at once (`join_all`, not
  `try_join_all`), each to its own end, returning the first error in the
  walk's order. Run concurrently, a flush holds the router for at most one
  bound however many connections stall in it; one after another, k silent
  connections would hold it for k bounds. With nothing fed it is one walk of
  the connections, and with one fed connection a plain flush of it, so neither
  adds an allocation to what the flush itself does; two or more are joined,
  which allocates the join. The flush itself allocates nothing in the steady
  state (tokio 1.53.1's `Timeout` polls the inner future first, and
  tokio-util's `poll_flush` writes in place); the exceptions are change 7's
  class rule installing a fresh buffer after the flush
  (`retention::on_flushed`) and the formatting of a DEBUG or WARN line on an
  error or an expiry. A failed flush logs at DEBUG: `Failed to flush fed
  frames to <node> before waiting`, `… before an inline connect` or `… before
  a write to <node>`; the router-wide flush logs nothing itself, as stock, and
  returns the error to its callers. An expired flush marks the connection
  failed and logs change 2's `(expired flush)` WARN, as any expired wait does;
- at each receive (`src/router/commands.rs:471,488`), sets the counter to the
  channel's length plus one (the command just taken, which `process_command`
  then subtracts), by compare-and-set so a sender's increment between the read
  and the reset is never overwritten, and logs a drift it finds there. Every
  sender increments the counter before it sends (`src/interfaces.rs:49-50`)
  and the router reads the channel's length before the counter, so a send in
  flight then reads one above the channel; the reset drops that increment, so
  once its command lands the next receive, whichever command it takes, reads
  one below. Commands that leave the channel without being counted down show
  the same way: stock `drain_command_rx` (`src/router/commands.rs:505-510`)
  cancels the commands queued when a router ends without decrementing, and the
  same channel goes back to the next router (`src/router/commands.rs:575`),
  whose first receive reads one above per cancelled command. A difference of
  one either way is usually that race, or a single command cancelled when a
  router ended, and logs at DEBUG; a larger one (several sends in flight at
  once, or several commands cancelled) logs one WARN per client per minute,
  with its size. Either way the reset has corrected it;
- makes the counter's decrement saturate at zero (`src/utils.rs:183`) instead
  of wrapping.

Not covered, besides the `-MOVED` sleep and the replica loops above, each a
wait on one connection that can still run while another holds fed frames:
the flush before a `CLIENT UNBLOCK` under `Blocking::Interrupt`
(`src/router/commands.rs:169-177`, and its copy in `live_edit::ask::follow`),
which only the service's stream readers set, and they are single-node clients
that write nothing to the keyspace; and fred's stock close, each bounded by
its 5 s close timeout, in `drop_connection` after a read error or the end of
the stream, and in `disconnect_all` after a failed cluster-wide command and
when a router ends. That close's flush can wait only on a peer that stopped
reading with its socket still open (one that half-closed): a socket that
errored, or that the peer closed, fails it at once. `disconnect_all` closes a
client's connections one after another (`src/router/connections.rs:276-280`),
so a frame fed on one late in that walk can wait up to 5 s per such peer,
closed one after another. Only a router's end reaches that walk with fed
frames, because a failed cluster-wide command runs `drain_all` first; so the
per-peer sum does not touch the service's keyspace writes within a write
bound built on change 9's residual. `drop_connection`'s close is one such
wait, up to 5 s, on one such peer. No flush is added before the close, because no test can
deterministically put the end of the stream inside one router step.

The service's `tests/integration/storage/fed_frame_flush_test.rs` pins the
hooks on a four-node cluster facade, each test failing with only its own
piece reverted in place: the inline-connect sites ((R), (S), (K)); the flush
before a write through each predicate term alone (T1 the counter at zero, on
a send; T2 the command kind; T3 the backpressure boundary; T6 the feed
count); the concurrent flush before the router waits (T4); and the flush in
`live_edit::ask::follow` before its writes (T5). F3, T1, T3 and T4 also fail
with `flush_fed_concurrently` made sequential. Its module docs list what it
does not cover: each flush running every connection to its own end (no
public-client construction puts a fast-failing flush ahead of a pending one),
the flush before `after_pipeline`'s `DISCARD`, the exceptions above, the
concurrency of `ask::follow`'s inline-connect flush, the counter-at-zero
term's share of the flush after a skipped command (every buffer that could
wait there is past the boundary, so that term fires too), and the
cluster-only gates.

### 6. `Pool` members can be re-pointed to a fresh client

A stock `Pool` holds a fixed `Vec<Client>`; member `i` is the same client for
the pool's life, and the pool hands out borrowed `&Client`s (`next`, `last`,
and `ClientLike::inner`, which every command goes through) that nothing could
wait for before swapping one. The patch (`src/live_edit/pool.rs`, hooked into
`src/clients/pool.rs` on existing lines) maps each logical member to a
**cell**: cells `0 .. size` are the pool's own clients, and `max(size, 4)`
spare cells follow, each filled at most once.
`live_edit::pool::replace_member(&pool, i, fresh)` stores an already-connected
client in the next empty spare cell, points member `i` at it and returns the
client it named before, for the caller to quit; every pool operation resolves
the member's current cell. `Pool::clients()` returns an owned snapshot of the
current members (a signature change from `&[Client]`), and `Pool::member(i)`
is new.

Retired clients are never removed from their cells, so a `&Client` handed out
before a replacement stays valid after it, without `unsafe`; they are dropped
with the last clone of the `Pool`. The spare cells bound that retention: when
they are used up `replace_member` fails with `BudgetExhausted` and changes
nothing. Clones of a `Pool` share its cells. Because a retired client's
`ClientInner` is never dropped, and its router may end through fred's normal
`QUIT` exit (which puts the command receiver back on the client) or be
aborted, `live_edit::pool::retire(&client, router)` releases what fred would
otherwise leave behind, once that router has finished: it takes, closes
(`Receiver::close`, `src/runtime/_tokio.rs:104`) and drains the command
receiver, cancelling every queued command, so a later send through a borrowed
`&Client` fails at once instead of waiting forever; it stops the
credential-refresh task; and it closes the backchannel connection. It does
so on a detached task spawned on the current runtime, bounded: it waits at
most `RETIRE_ROUTER_WAIT` (5 s) for the router, and gives the backchannel's
close, one socket's, at most `RETIRE_BACKCHANNEL_CLOSE_TIMEOUT` (1 s), since
fred puts no timeout on that close, and logs one WARN, naming the client, when
that expires. What a close cut short leaves depends on where it was cut. Cut
after it took the backchannel's async lock, it has already taken the
connection out of the backchannel, and dropping it closes the socket. Cut
while still waiting for that lock, it has taken nothing: the connection stays
on the client until the client is dropped. For a client kept in a spare cell
(a pool member's retired client), which is never dropped before its `Pool`,
that one socket stays open until the pool's last clone drops; that is a known
residual. A client kept in no cell (a discarded fresh client, a `txn` slot's
swapped-out client) keeps it only until its last clone drops, normally soon
after its router ends. The WARN says "until the client is dropped (for a pool
member, with its pool)". The lock's other holders are the client's router
(its cluster-state sync, the unblock of a blocked connection) and backchannel
commands sent on the client, which `PING` first with no timeout; the service
sends none on a pool member, so the wait is reachable only when the old
router is still running as the release starts (no router handle was given,
or the router did not finish within `RETIRE_ROUTER_WAIT`).

### 7. Connection buffer limits

A stock connection has four holders whose allocations only ever grow, so it
keeps the largest load it has ever carried until the connection itself is
replaced:

- the in-flight queue, `Connection::buffer`, a `VecDeque<Command>` of 64 slots
  to start (`src/protocol/connection.rs:707`): one slot per written command,
  popped per reply (`src/router/centralized.rs:28`,
  `src/router/clustered.rs:297`). The router writes a whole pipeline before it
  reads a reply (`src/router/commands.rs:230-248`), so the queue grows to the
  longest pipeline in flight;
- the write buffer, tokio-util's `Framed` `BytesMut`, 8 KiB to start: a frame
  is encoded whole, so it grows to the largest frame plus what was buffered
  ahead of it;
- the read buffer, 8 KiB to start: fred decodes only complete frames
  (`src/protocol/codec.rs:69-166`), so it grows to hold the largest frame read;
- the router's retry buffer, `Router::retry_buffer` (`src/router/mod.rs:61`),
  empty and unallocated to start (`:81`): it takes a lost connection's
  in-flight commands for replay and keeps its capacity afterwards.

Draining a byte buffer does not free its allocation: a flush advances it and a
decode splits the frame off, and the next `reserve` reclaims the whole
allocation. No fred setting bounds any of the four (`max_command_buffer_len`
bounds the router's command channel, `max_feed_count` the number of frames
between flushes, not their size). What a replica's connections keep therefore
depends on their history, not on configuration. Measured on stock 10.1.0 by
the service's upload memory envelope test (`tests/upload_memory_envelope/`),
memory still live after a call returned: 75.3, 124.75 and 223.6 MB after a
20,000-field WATCH commit at txn pool sizes 2, 4 and 8; 33.4 MB after ten
16 MiB blob writes; a 4 MiB read buffer after a cold-start structural read.

The patch adds `ConnectionConfig::live_edit.retention`, an
`Option<RetentionLimits>` with three limits: in-flight commands, bytes written
and bytes per frame read. `None`, the default, is stock behaviour. The limits
are copied onto each connection's codec when the connection is created and
never change at runtime. Every check runs in the router task that owns the
holder, so nothing is locked:

- **in-flight queue**, after a reply is matched to the command at its head
  (`src/router/centralized.rs:29`, `src/router/clustered.rs:298`): when its
  capacity is above the command limit and at most half that many commands are
  queued, it shrinks to the limit.
- **retry buffer**, after the router replays it (`src/router/mod.rs:396`):
  once the replay empties it, it is replaced by an unallocated
  `VecDeque::new()`, stock's starting state. It is used only during a replay,
  so it keeps nothing between replays.
- **write buffer**, after a write-and-flush or a flush leaves it empty
  (`src/protocol/connection.rs:808`, `:896`): when an encode since it was last
  replaced left more than the write limit in it, it is replaced by a fresh
  8 KiB buffer.
- **read buffer**, when a frame is taken off the connection and the buffer is
  left empty: when a frame decoded since it was last replaced was larger than
  the read limit, it is replaced by a fresh 8 KiB buffer. Every way a frame is
  taken off a pipelined connection runs the check before the frame's caller is
  answered: the read loop (`src/router/types.rs:45`); a reply matched to its
  command (the reply pop above, which also covers replies read by a drain);
  and `Connection::read` (`src/protocol/connection.rs:832`, through the
  module's `next` hook), which covers the frames fred reads outside the read
  loop. Of those, the large one is a `Transaction`'s EXEC reply
  (`src/router/transactions.rs:42`), the sum of every queued command's reply;
  the check runs before `exec` answers its caller. A connection's setup
  exchange and a client's backchannel read through `ExclusiveConnection`
  instead; the backchannel is covered by the sweep below.

Those per-connection limits alone let a client keep its class limits once per
connection, and a cluster client holds a connection to every primary
(`src/router/mod.rs:196-203` routes by slot), with traffic reaching all of them
over time. So each client also keeps at most one **warm** connection, the only
one allowed to keep the class limits once drained (no queued command, nothing
in either buffer); every other drained connection is cut to the **idle shape**:
its queue's allocation freed (`shrink_to(0)`), an unallocated write buffer and
a fresh 8 KiB read buffer.

- **The sweep** runs as the first statement of `live_edit::before_wait`
  (change 5's hook at the top of `read_or_write`,
  `src/router/commands.rs:459`), ahead of its `queued > 0` early return, so it
  runs on every pass of the router loop: once per command, pipeline or batch of
  reads and, with the unresponsive check configured (as the service does on
  every builder), at least once per its interval while the router waits. It
  elects the warm connection by the newest read progress, which the
  unresponsive check already records per connection on every frame taken off
  it (change 2): a one-connection client's connection is warm from the first
  sweep after it connects; the
  current warm connection keeps warmth unless it is drained and another
  connection has strictly newer progress; with none, the connection with the
  newest progress is elected. It then applies the class rules to the drained
  warm connection and cuts every other drained connection that carried
  something since its last cut (an `idle` flag on the codec state, set by a
  cut and cleared by every encode and decode). The sweep's own work allocates
  nothing, since it runs about once per command: it walks the client's
  connections twice through an iterator over the router's own collections,
  folding the election over their views in the first walk
  (`retention::elect_streaming`, which keeps only the current warm
  connection's index and view and the newest read progress's index and time
  in locals) and setting each one's warm flag and applying its action in the
  second. No connection is added or removed between the walks, so both visit
  the connections in the same order, and acting on one connection changes no
  other's view. Its only allocations are the buffers its cuts and class rules
  install, so a pass of a client at rest allocates nothing.
- **At the reply pop**, a reply that leaves a non-warm connection drained cuts
  it there, before the reply's caller is answered.
- **The backchannel**, inside the same sweep: the sweep takes the backchannel's
  lock with `try_write` (a tokio `RwLock`, `src/runtime/_tokio.rs:45`, which a
  caller task holds while it uses the backchannel), so a busy lock skips one
  pass and the router never waits on a caller. When the backchannel holds
  nothing (both buffers empty) and carried anything since its last cut (the
  same `idle` flag, which a new connection starts with clear), both its
  buffers are replaced by fresh 8 KiB ones and `idle` is set. Cutting after
  any traffic, rather than after a frame above some size, keeps "8 KiB read +
  8 KiB write" exact without a growth derivation: a 4–8 KiB reply behind a
  setup reply's consumed prefix can double an 8 KiB buffer to 16 KiB, and a
  large credential in HELLO would grow the write buffer. The cost is two 8 KiB
  allocations per use, and use is rare: one per cluster sync of a healthy
  client, which fred re-opens the backchannel for (`src/router/clustered.rs:410-411`,
  `:470-478`), and one per reader interrupt.
- **The cluster routing table owns its strings**
  (`src/protocol/types.rs:559`, the end of `ClusterRouting::rebuild`, the only
  place a running client builds it; `from_cluster_slots` is unchanged). fred
  parses `CLUSTER SLOTS` without copying (`src/protocol/utils.rs:295-299`,
  `src/protocol/cluster.rs:66-112`), so each `SlotRange`'s `id`, primary
  `host` and TLS server name is a view into the read-buffer allocation the
  reply was read into, kept past the sync by the routing state, every
  connection-map key and `Connection::server` added at that sync, the next
  backchannel's `server` (`connect_any`), and any `Server` a caller clones
  from the routing (the service's stream readers keep their owner's `Server`
  in their client config for life). Each would pin a whole allocation however
  the backchannel's buffer is cut. `retention::own_slot_ranges` replaces each
  of those strings, and the replicas' under the `replicas` feature, with an
  owned copy, so the reply's allocation is freed when the sync ends.
- **A sentinel-named primary owns its strings**
  (`src/router/sentinel.rs:228`, in `discover_primary_node`, after the
  primary's `Server` is built and before fred connects to it). fred converts
  the `SENTINEL GET-MASTER-ADDR-BY-NAME` reply to the host `Str` without
  copying (`FromValue for Str`, `src/modules/response.rs:253-261`), so that
  host is a view into the sentinel connection's read buffer. The `Server`
  built from it lives on as the primary connection's `server` and
  `default_host`, its codec's server, the client's recorded sentinel primary
  (`ServerState::update_sentinel_nodes`) and the backchannel's connection-id
  key (`update_sentinel_backchannel`), each of which would keep that whole
  allocation alive for the connection's life, on every client of a sentinel
  store. `retention::own_server` replaces the host and TLS server name with
  owned copies. The hook rebinds the immutable `server` as mutable on the same
  line, keeping the line count. The sentinel node list from `SENTINEL
  sentinels` is converted through owned `String`s and needs no copy. A search
  of fred's router, connection, client-state and backchannel code for
  zero-copy conversions (`into_bytes_str`, `as_bytes_str`, `(Str, …)`) finds
  only these two sites.

The election, the cut decision, the backchannel decision and the two string
copies are public pure functions
(`fred::live_edit::retention::{elect, elect_streaming, action, plan, cut_backchannel, own_slot_ranges, own_server}`),
so they can be tested without connections; `elect` and `plan` run the same
fold as the sweep, `elect_streaming`, over a slice. The sweep reaches the
client's connections through change 2's `liveness::connections_mut`, made
`pub(crate)` for it and returning an iterator over the router's own
collections (a two-variant enum over the centralized or sentinel connection
and the cluster map, chained with the replica map under the `replicas`
feature) instead of a collected `Vec`, so walking the connections allocates
nothing, in the sweep and in change 5's walks alike (the check before a write,
and the shared flush, which allocates only to join two or more connections'
flushes); it reads read progress through `liveness::last_progress`, a `pub(crate)`
accessor added for it. Those are this change's edits in
`src/live_edit/liveness.rs`. A one-connection client, which includes every
client of a standalone or sentinel store, is warm from the first sweep after
it connects, so at rest the warm rule leaves it at its class limits. The one
exception is a reply popped in the same router pass as a reconnect, before any
sweep: a `Transaction` retried after a reconnect (`retry!` →
`reconnect_with_policy` → `drain_all`, `src/router/transactions.rs:122-142`)
drains on a connection not yet marked warm and can cut it once, which changes
allocations only.

The byte buffers are judged by what they held, which the codec records
(`src/protocol/codec.rs:64`, `:83`, `:104`, `:121`), never by
`BytesMut::capacity()`: `advance` and `split_to` lower `capacity()` by the
bytes consumed while the allocation stays whole, so a drained buffer whose
allocation is 16 MiB reports a capacity near zero and a check keyed on it would
never fire.

Only capacity changes: command order, write attempts, flush points and what is
written are untouched, so is the unresponsive check's state (change 2), and a
byte buffer is replaced only while it holds nothing, and only drained
connections are cut, so no partial or unflushed frame is lost and a blocked
command, which is queued, is never touched. A decoded frame holds its own
reference to its bytes, so a reply on its way to a caller is unaffected; the
old allocation is freed when the last frame cut from it is dropped, and no
value fred keeps past a call holds such a view, on any topology, once the
routing table and the sentinel-named primary own their strings. The cost is
one reallocation per drain after traffic above a limit and, on a client with
several connections, up to three small allocations when a cut connection is
used again. The kept bytes per holder, and their derivation from the growth
policies of tokio-util 0.7.19 and bytes 1.12.1, are on `RetentionLimits`
(`src/live_edit/retention.rs`), whose module docs give the reasons for the
warm rule's choices; the service's limits per connection class and the memory
bound they give are in `src/storage/connection_buffers.rs`.

**A test-only report, test sweep modes and sweep and write-check instruments.** No byte count can tell one warm
connection per client from several, or from none, at every primary count and
limit, so the service's tests read the warm rule's state directly. With the
fred feature `live-edit-retention-report`, declared on fred's `Cargo.toml`
line 216 (a blank line before the patch, so the line count holds),
`retention::test_support` adds:

- **a per-client report.** `request_report(&client)` clears any report not
  yet taken and makes a new request; requests are numbered, not flagged.
  `live_edit::before_wait` reads the pending request number before the sweep
  and, right after the sweep, stores one `ConnectionReport` per connection
  (server, limits, the warm and idle flags, whether it is drained, whether it
  has made read progress, the commands in its in-flight queue, and the queue,
  write and read capacities);
  `take_report(&client)` collects it. Because the number is read before the
  sweep and checked again under the report slot's lock, only a pass whose
  sweep began after the request answers it, so a report shows exactly what
  that pass's sweep left, and a sweep mode set before the request is the mode
  that sweep ran. A pass that finds the slot held by a test thread
  (`try_lock`) leaves the request for the next pass, so the router never
  waits on a test thread. The router passes at least once per unresponsive
  check interval, so a request is answered within about one interval, with
  or without traffic. The answer is a separate call after `sweep`, not part
  of it, so a build with the sweep call removed still answers and fails on
  what the report shows rather than on a timeout. The queue capacity is exact
  (`VecDeque::capacity`); the byte-buffer capacities are
  `BytesMut::capacity()`, exact only for a buffer nothing has been consumed
  from, which is what a cut leaves, and the rule itself still never reads
  them.
- **sweep modes per client** (`set_sweep_for_test`): `Skipped` models a build
  without the sweep (no warm connection, every drained connection cut at its
  reply pop), `EveryConnectionWarm` the per-connection rule alone (every
  connection keeps the class limits), and `AsBuilt`, the default, the sweep
  as built. They let a test show its check tells a broken rule from a working
  one on the cluster it runs against.
- **sweep counters per client** (`sweep_stats(&client)`, a `SweepStats` of
  `passes` and `nanos`): how many sweeps ran and their total wall time,
  counted from the client's creation in every sweep mode by a guard
  (`SweepPass`) that `sweep` takes just after it finds the client's limits
  and that counts the pass as it drops. A test reads them before and after
  its workload and divides the differences for the time per pass.
- **write-check counters per client** (`write_check_stats(&client)`, a
  `WriteCheckStats` of `checks`, `nanos`, `flushes` and `flush_nanos`): how
  many of change 5's checks before a write (`liveness::before_socket_write`,
  `liveness::before_write_to`, which run only on a clustered client) ran, the
  wall time of their decisions up to the start of any flush, and how many went
  on to flush, with those flushes' time. A guard (`WriteCheck`) taken on each
  check's first line counts it as it drops, so every call is counted, early
  returns included, whether or not the client has limits. A test divides the
  differences of two readings for the decision's time per check.
- **an in-sweep flag per thread** (`in_sweep()`): true only while a sweep
  runs on the calling thread, set and cleared by the same guard. The sweep is
  synchronous, so the flag never spans an await. It is a const-initialised
  thread-local `Cell<bool>` with no destructor, so reading or setting it
  never allocates, and a test's counting global allocator reads it to count
  the allocations made inside the sweep.

The per-client state is one field on `live_edit::ClientState`, created with
the client's `ClientInner` and dropped with it; a client is passed, never a
`Pool`, whose `ClientLike::inner` round-robins across members. None of it
exists without the feature: no field, no thread-local, no check or clock read
in the router's pass or in its writes. Only the service's
`test-support` feature enables it, so the release build never contains it
(*How the copy is wired in*).

### 8. `process_ask` follows an `-ASK` to its target

A node answers `-ASK <slot> <target>` while `<slot>` is migrating away from it
and the command's keys are no longer there. The client must then send
`ASKING` and the command to `<target>` on one connection: the target serves an
importing slot only to a connection that sent `ASKING`, and clears that flag
after the next command (Valkey 9.0.3 `src/networking.c:3260-3262`). The
routing table keeps the source as the slot's owner until the migration ends.

Stock `process_ask` (`src/router/commands.rs:251-285`) never completes the
redirect. `send_asking_with_policy` (`src/router/utils.rs:441-493`) sends
`ASKING` to the target with `cluster_node` set to it; `process_ask` then
re-writes the command with only `hasher = ClusterHash::Custom(slot)`, leaving
`cluster_node` unset, so `Router::route` (`src/router/mod.rs:176-207`) sends it
to `cache.get_server(slot)`, the source, which answers `-ASK` again. Each round
costs a write attempt, and the command ends `Routing` "Max attempts reached."
whatever its budget: every command on a key that has already moved fails for
as long as the slot migrates. And `send_asking_with_policy` reads the `ASKING`
reply synchronously, with `read_skip_pubsub` on the target's connection
(`:477-488`, `src/protocol/connection.rs:840-855`). With commands already in
flight there, that read takes the oldest one's reply instead of `+OK`; the
`+OK` is then matched to that command, and every later reply on the connection
to the command before its own. Transactions avoid it only because
`transactions::send` drains the connection first
(`src/router/transactions.rs:142`).

The service met the first defect on a real cluster: during a staged slot
migration, a structural read's plain `SCARD` of a moved key failed
`transient I/O: SCARD deleted sections index failed (Routing Error): Routing
Error: Max attempts reached.` at the default three attempts (CI pipeline
73100357, job 481449796). Change 1's one-node mock could not show it.

The patch (`src/live_edit/ask.rs`, called from `process_ask` on one existing
line, `src/router/commands.rs:265`) keeps `process_ask`'s redirection-budget
check first, so a client with no redirections (the service's `txn` pool) is
never redirected, and keeps stock behaviour for a command with no write
attempt left: `send_asking_with_policy` fails it `Routing` "Max attempts
reached." before writing anything (`src/router/utils.rs:468-469`), so a
one-attempt command (the `UpdateLayer` commit `EVALSHA`, the operation-id
start-record `HGET`) is still written at most once and fails as before. For a
command with a write attempt left, `live_edit::ask::follow`:

- takes the target's connection, or opens one when the client has none — a
  node that owns no slot yet, as in a scale-out (`Connections::add_connection`,
  `src/router/connections.rs:375`), announced on the client's reconnect events
  as a cluster sync announces a connection it adds (`src/router/clustered.rs:574`).
  A target the routing table lists as a primary gets that entry's TLS server
  name; one it does not gets the client's TLS host mapping applied with the
  slot owner's TLS server name as the default host. The connect runs inline on
  the client's single router task, so every other command on the client waits
  while it runs: the whole step — TCP connect, TLS handshake and the setup
  exchange `add_connection` runs (`HELLO` / `AUTH`, `SELECT`, `CLIENT ID`, the
  version and cluster-state checks, `src/protocol/connection.rs:620-648`) — is
  bounded by `live_edit::ask::ASK_TARGET_CONNECT_CAP`, 2 s, or the client's own
  connection timeout when that is shorter (`ask_target_connect_timeout`; the
  cap also applies when the client's is zero, fred's "no timeout"). Without it
  the step would inherit the client's connection timeout (the service's
  default is 30 s), and an unreachable target would stall the whole client for
  that long; 2 s leaves margin for a TLS handshake on a healthy managed node.
  Before it connects, every connection holding fed but unflushed frames is
  flushed (change 5), so no fed frame waits out the connect; and before it
  writes the `ASKING` and the command, they are flushed again whenever another
  connection holds fed frames, so none waits through those writes either.
  The bound wraps the `add_connection` call (`utils::timeout`), which inserts
  the connection only after connect and setup both succeed
  (`src/router/connections.rs:381-383`), so a cut step leaves nothing in the
  connection map. A failed or cut connect fails only that command, with an
  error naming the target, and nothing is written. Its kind follows an
  allowlist (`live_edit::ask::ask_connect_error`): a configuration or
  credential fault — `Tls`, `Auth` (a `WRONGPASS` or `NOAUTH`), `Config`,
  `Url` — keeps its own kind, so the caller's classifier answers it as the
  permanent fault it is. Under rustls, the TLS backend this build uses,
  `Tls` covers only an invalid TLS server name
  (`src/protocol/connection.rs:390`) and a TLS connector that does not match
  the backend (`:373`, unreachable with one TLS backend compiled in). A bad
  certificate or a failed handshake is not `Tls`: it reaches fred as an
  `std::io::Error` (tokio-rustls 0.26.4 `src/common/mod.rs:115`, through the
  handshake's `?` at `connection.rs:393`), which fred's `From<io::Error>`
  makes `IO` (`src/error.rs:179-182`), so it is retried here, as fred's own
  reconnect path retries it. Every other failure is `IO`, the transient
  shape a lost connection already has, because retrying fixes it. That
  covers a connect failure, the cap's `Timeout` and a `Canceled`, and two
  setup outcomes a kind-preserving rule would leave permanent: a node still
  joining or rejoining the cluster reports `cluster_state:fail` for a while
  (Valkey 9.0.3 `src/cluster_legacy.c:6311-6313`, `:6358-6362`, `:6373-6377`),
  which the setup's cluster-state check answers `Protocol` "Invalid or missing
  cluster state." (`src/protocol/connection.rs:615`); and a server error
  during setup (`-LOADING`, `-BUSY`) arrives as `Unknown`, and once wrapped in
  this error's text no longer starts with the reply token the service's
  not-serving check matches. The failure is also
  broadcast on the client's error events with its own kind and the target,
  as the router reports a connection it lost (`src/router/responses.rs:311-319`).
  The caller's retry, or the next command redirected there, may try again
  under the same bound. A non-blocking connect, for these and for fred's
  cluster-sync connects alike, is a separate follow-up; fred's own
  cluster-sync connects are unchanged. The next cluster sync that does not
  list the node as a primary closes the connection
  (`src/router/clustered.rs:551-561`) and retries the commands still in
  flight on it, routed by slot;
- writes `ASKING` there as an ordinary pipelined command
  (`Command::new_asking`, `ResponseKind::Respond(None)`, through
  `utils::write_command`) with no write attempt of its own, then the command,
  in the same router step, taking the command's attempt as `write_command`
  does (`src/router/commands.rs:111`). Nothing is read: the ordered reply
  matching consumes the `+OK` (`src/router/clustered.rs:291-330`). A replay of
  the connection's in-flight commands after it drops discards the `ASKING`
  (`Command::should_finish_with_error`, `src/protocol/command.rs:1771-1773`)
  rather than sending it to the slot's owner;
- pins the command to the target for that one write only: no field of the
  command names the target, so a replay after a dropped connection, or a later
  `-MOVED`, routes it by slot again;
- handles a write failure as `write_command` does
  (`src/router/commands.rs:186-204`): the connection is dropped, its in-flight
  commands are retried or failed, and a reconnection is deferred. When the
  `ASKING` write fails, the command was not written and is retried or failed
  with it.

A command the caller has already timed out writes nothing, not even its
`ASKING`. A command whose timeout lands between the two writes (the second
write skips it, `src/router/utils.rs:71-78`) leaves an `ASKING` that the next
command on that connection consumes. That command reached the target by slot,
for a slot the target owns, where `ASKING` changes nothing; by another `-ASK`,
which sends its own `ASKING` first; or pinned to the node by its caller
(`with_cluster_node`), the one case where it could be served for a slot the
node is importing instead of being redirected.

**A script never follows an `-ASK`.** Valkey routes a script by its declared
keys only (Valkey 9.0.3 `src/cluster.c:1263-1273`): the source answers `-ASK`
once the declared keys have left, whatever undeclared keys of the same slot
the script also reads. On the target, the script's own client inherits the
caller's `ASKING` flag (`src/script.c:431-433`), and in an importing slot a
single-key command under `ASKING` is served locally even when its key has not
arrived yet (`src/cluster.c:1281-1287`). So an undeclared key still on the
source reads as absent, with no error: a following script returns a partial
answer, or acts on one (the service's snapshot and window reads, its session
checkpoint read, and its external-value index recount, which writes a
repair). `follow` therefore fails every script command answered `-ASK` before
it touches the target — `Routing` "Max attempts reached.", the shape it had
before this change, which the service's classifier treats as transient —
with nothing written and nothing sent. It matches by command kind, never by
caller (`live_edit::ask::is_script_command`): fred's `EVAL`, `EVALSHA`,
`FCALL` and `FCALL_RO` kinds (`CommandKind::is_eval`,
`src/protocol/command.rs:1520-1525`), and a custom command whose name, its
first word in any case, is `EVAL`, `EVALSHA`, `EVAL_RO`, `EVALSHA_RO`,
`FCALL` or `FCALL_RO` (fred has no kind for the two `_RO` eval forms; the
service sends `EVALSHA` as a custom command for its rebuilding and one-shot
script calls). Plain commands, which name every key they touch, still
follow. This does not touch a script that runs on the source and meets a
moved undeclared key there: that is the script's own in-script refusal
(`src/script.c:446-452` or `:456-459`), which no `-ASK` carries. Reverting
this rule alone is deleting its one `if` block at the top of `follow`.

The service's integration suite pins the behaviour with a two-node mock
cluster, which a one-node mock cannot do: a default-budget command on a moved
key is answered through the target, the source receives it once and the
target receives `ASKING` immediately followed by it; a one-attempt command
still fails "Max attempts reached." with nothing sent to the target; a command
in flight on the target receives its own reply; an `-ASK` to a node that owns
no slot is followed over a new connection; and no router task ends.

`send_asking_with_policy` keeps one caller, `process_ask`'s zero-attempt path,
which returns before routing or writing anything, so its synchronous read and
its "Unable to route command or reconnect." fallback are no longer reached.
It stays because the patch edits existing lines only. **Reverting the change**
is one edit: delete the `if attempts_remaining > 0 { … }` statement on
`src/router/commands.rs:265`, leaving `let attempts_remaining =
command.attempts_remaining;`, and every `-ASK` takes the stock follow again
(with change 1).

### 9. A call its caller gave up on is never written

Every command carries a `timed_out` flag (`Command::timed_out`,
`src/protocol/command.rs:1556`), and the router refuses to write a command
whose flag is set: at the write (`src/router/utils.rs:71-78`, which every path
that writes a command or a pipeline's command goes through: the command
channel, the retry buffer's replay after a reconnect or a cluster sync,
`process_moved`), at the replay (`src/router/mod.rs:374`) and, since change 8,
before following an `-ASK` (`live_edit::ask::follow`). Stock fred sets the flag
in one place, `basic_request_response` (`src/utils.rs:402-421`), and only when
the call ends with an error. So:

- a call whose future is dropped never sets it: an abort, a deadline, a
  cancelled client, the losing branch of a `select!`;
- a pipeline (`src/clients/pipeline.rs:263-320`: `try_all`, `all`, `last`) or
  a transaction (`Transaction::exec`, `src/clients/transaction.rs:278-327`)
  never sets it, even when its own timeout fires: `Command::duplicate`
  (`src/protocol/command.rs:1855`) gives each queued command a fresh flag that
  nothing else holds;
- the transaction router (`src/router/transactions.rs:100-261`) writes its
  frames with `Connection::write` directly and never reads the flag, so a fred
  `Transaction` is written whatever its flags say, and fred's own
  whole-transaction replay after a reconnect (`retry!`, `:122-133`, `continue
  'outer` at `:130`) writes it again.

The router reconnects inline, and the service runs it under a reconnect policy
with no attempt limit, so the pools heal after any outage. A pipeline or
transaction queued behind such a reconnect is therefore written once the store
is reachable again, possibly long after its caller's timeout fired and its
caller answered the error, and a `MULTI … EXEC` written then runs unwatched:
the `WATCH` went with the old connection. Every `EXEC`-class write the service
sends is exposed: its `WatchedTxn` pipelines (the WATCH-route `UpdateLayer`
commit, `CreateSession`, the DR restore's residue wipe and materialisation),
its fred `Transaction`s (`CreateLayer`'s object write, the session delete, the
session restore's two steps), and any write whose task is aborted. A bound on
when the service's last writes can land — the DR hold's, which the restore's
quiet period rests on — then holds for those writes only while their
connection holds.

The patch (`src/live_edit/abandon.rs`, hooked on existing lines) makes the
rule: once a call's caller has given up — its timeout fired, fred answered it
with an error, or its future was dropped — no frame of that call fred has not
yet written is written.

- **One flag per call.** Each entry point arms a guard
  (`live_edit::abandon::Guard`) before it hands the work to the router: on the
  command's own flag in `basic_request_response` (`src/utils.rs:402`) and its
  tracing twin (`:459`, not compiled in this build, hooked so the behaviour
  does not depend on features); for a batch, on one fresh flag
  `Guard::share` assigns to every command of the pipeline
  (`src/clients/pipeline.rs:268`, `:295`, `:313`) or the transaction
  (`src/clients/transaction.rs:321`), in place of `duplicate`'s. The guard
  wraps the call's reply receiver (`Guard::watch`, at `src/utils.rs:408`,
  `:414`, `:474`, `src/clients/pipeline.rs:275`, `:300`, `:318`,
  `src/clients/transaction.rs:325`): a reply that resolves `Ok` disarms it —
  a server's error reply to a command included, which reaches the receiver as
  an `Ok` frame (`src/protocol/responders.rs:442`), harmlessly, since the call
  is complete; any other outcome — fred's own timeout, an `Err` fred itself
  sends on the receiver (a failed write, a lost connection, a pipeline answered
  early with its first error, a transaction's error), an errored receiver —
  and dropping it before an `Ok` set the flag. Disarming on `Ok` keeps
  `Pipeline::last` as it was for a pipeline spread over several nodes, whose
  earlier commands may still be in flight when the last one answers (the
  service never calls `last`). A batch's commands share the flag, so setting
  it covers every frame of the batch, and the existing checks apply: a
  pipeline's command still in the channel, behind a reconnect, in the retry
  buffer or redirected is skipped where a timed-out single command already
  was.
- **A pipeline cut after its `MULTI`.** A pipeline is written one command at a
  time in one router step (`process_pipeline`, `src/router/commands.rs:230-248`),
  and the flag can be set partway, which can leave a `MULTI` written and its
  `EXEC` not: an open transaction on that connection, which turns the next
  commands there — the `txn` pool's checkout probe, a `WatchGuard`'s
  `UNWATCH` — into `QUEUED` replies, a false `leaked_multi` verdict and a
  spurious `WatchGuard` ERROR. `process_pipeline` records the last `MULTI` it
  hands to the write with its flag still clear (`:235`, `:244`) — by kind,
  `CommandKind::Multi` or a custom command whose first word is `MULTI`, as
  `WatchedTxn` sends it — and the server it routes to, and at the end of the
  step (`:247`, `live_edit::abandon::after_pipeline`) writes one `DISCARD`
  there when the flag is then set. The `DISCARD` is an ordinary tracked
  command written through `utils::write_command` in the same router step, so
  no other caller's frame comes between; its reply is matched in order and
  dropped (`ResponseKind::Respond(None)`), and nothing is read synchronously.
  It has no write attempt left, as change 8's `ASKING`, so a replay of the
  connection's in-flight commands discards it. Nothing needs to know whether
  the `EXEC` was written: if it was, the `DISCARD` answers `-ERR DISCARD
  without MULTI` and changes nothing. The step opens no connection, so the
  connection under the `MULTI`'s server is the one it was written to, or there
  is none (a write error dropped it, and the store discarded the `MULTI` with
  it) and nothing is written. A `MULTI` routed to a replica (`use_replica`,
  `replicas` feature, which the service does not build) is not recorded,
  because routing it a second time would advance the replica round-robin.
  The `DISCARD` covers a cut during the router step that writes the pipeline,
  and only that: a pipeline whose commands have a write attempt left and whose
  connection drops is replayed from the retry buffer one command at a time
  (`Router::retry_buffer`, `src/router/mod.rs:366-397`), with no
  `PipelineStep`, so a flag set partway through that replay skips the rest but
  leaves a replayed `MULTI` open. The service sends every pipeline holding a
  `MULTI` on its `txn` pool, at one write attempt, so it never replays one.
- **A fred `Transaction`.** `transactions::send` checks the shared flag at the
  top of each attempt (`src/router/transactions.rs:121`), after the drain
  that starts it (`:145`, which can wait up to the slow bound), before the
  `ASKING` it sends after an `-ASK` (`:174`), before each command (`:181`
  numbers them, `:189`) and before `EXEC` (`:196`). Once it is set: when no
  frame of the attempt has been written, the caller's sender is answered with
  a `Canceled` error and nothing is written
  (`live_edit::abandon::refuse_transaction`); otherwise — an `ASKING` or the
  `MULTI` was written — the connection is closed without a flush or a
  shutdown, and a reconnection is deferred, as a failed write does
  (`close_transaction`). The store discards an unexecuted `MULTI` when its
  connection closes. The close is marked on the connection
  (`ConnectionState::abandon`), and change 3's close hook
  (`live_edit::close`, from `src/protocol/connection.rs:906`) returns at once
  on a marked connection. Closing rather than writing `DISCARD` is
  deliberate: transaction frames are written outside the in-flight queue, so a
  `DISCARD` would need a synchronous read of every pending `+QUEUED` reply.
  The attempt drained every connection before it began, so the closed
  connection holds no other caller's command. The close reaches the client's
  error events once, as `Canceled` (the service counts it under
  `live_edit_valkey_errors_total{pool="txn",kind="Canceled"}`), and costs one
  reconnect (`live_edit_valkey_reconnects_total{pool="txn"}`); there is no new
  metric. That reconnect is deferred to the back of the router's command
  channel, so a command already queued there is routed before it: on a
  centralized client, whose one connection was closed, it finds no connection
  and fails "Failed to route command." (or is kept for a retry when it has a
  write attempt left); a cluster client routes it as after any failed write. A
  routing miss means the router holds no connection at all, so no transaction
  state can be left for such a command to find, and the service reads it on
  its `txn` pool — a new borrower's checkout probe, a `WatchGuard`'s `UNWATCH`
  — as a connection already gone (`storage::txn_checkout`,
  `storage::watch_guard`), not as a dirty slot. The skips log at DEBUG. These checks cover a transaction queued
  behind a reconnect and fred's own whole-transaction replay alike: the replay
  starts a new attempt.

What happens to each frame once the flag is set:

| The frame, when the caller gives up | Pipeline (and the service's `WatchedTxn`) | fred `Transaction` |
|---|---|---|
| still in fred's command channel (behind a reconnect) | skipped at the write | refused at the attempt's check; nothing written |
| not yet written, the router partway through the batch | skipped at the write; a `DISCARD` when a `MULTI` was written | not written; the connection closed |
| written, awaiting a reply, when the connection drops | not replayed (the `txn` pool already sends one attempt) | the next attempt refused at its check |
| redirected (`-ASK` / `-MOVED`) | not followed | — |
| an `EXEC` already written, only its reply late | **runs: this cannot be stopped** | **runs: this cannot be stopped** |

**The residual.** A frame handed to fred's write buffer or the socket before
the flag was set is not recalled. It leaves at the router's next flush, and
each wait on its own connection's I/O there is change 3's bounded write or
flush wait (`UnresponsiveConfig::max_timeout`, 45 s in the service): it
reaches the socket within one such wait, or its connection is failed, however
many of the client's connections stall. On a clustered client that holds
across inline reconnects, cluster syncs, `-ASK` connects and the router's
waits on other connections, because change 5 flushes every fed frame, all
connections at once, before the router waits, before it connects inline, and
before a write that can reach a socket while another connection holds fed
frames, and runs fred's router-wide flush on its connections at once, each to
its own end. Change 5 names the exceptions: the flush before a
`CLIENT UNBLOCK`, which only the service's stream readers reach, and they
write nothing to the keyspace; and fred's stock close, up to 5 s on a peer
that stopped reading with its socket still open: once for
`drop_connection`'s close after a read error or the end of the stream, and
per such peer at a router's end, where `disconnect_all` closes them one after
another (a failed cluster-wide command runs `drain_all` first). On a
centralized or sentinel client, which has one connection, a frame fed just
before a reconnect can also wait out that pass's backoff sleep (up to about
200 ms: the service's 100 ms first reconnect delay plus fred's up-to-100 ms
jitter) before the pass's close flushes it. Once it is in the kernel's send
queue, its delivery and execution are TCP's and the store's. This applies to a
single command exactly as to an `EXEC`. So every `EXEC`-class write reaches the
socket, if at all, within its own client timeout (the command timeout, or the
upload commits' size-scaled transaction timeout) plus at most that write
bound, like a single command, outside change 5's exceptions and the
centralized or sentinel backoff sleep.

**Not guarded:** senders that hand the router a command without
`basic_request_response`, none of them a keyspace data write the service
sends — `QUIT` and `SHUTDOWN` (`src/commands/impls/server.rs:38`, `:69`),
`HELLO` / `AUTH` (`:189`, `:212`), the cluster-wide commands (script and
function administration, `src/commands/impls/lua.rs:77`, `:94`, `:121`,
`:295`, `:330`, `:397`, `:449`; a cluster-wide `FLUSHALL`, `server.rs:133`;
client tracking, `src/commands/impls/tracking.rs:88`, `:113`), which go to
every node through a path that never reads the flag, and the `SCAN` family's
pages (`src/commands/impls/scan.rs`, `src/types/scan.rs`). The router's
control messages (`Reconnect`, `SyncCluster`, `SyncReplicas`) carry no
command. The service reaches one of the senders: a `NOSCRIPT` reload's
`SCRIPT LOAD` goes through `Script::load` to `script_load_cluster`
(`src/types/scripts.rs:200`), which on a cluster client sends it to every node
directly (`src/commands/impls/lua.rs:77`; on any other client it is an
ordinary, guarded `SCRIPT LOAD`); it writes the script cache, not the
keyspace.

The rule is "never written after the caller gave up", not "never executed".
It also covers background writes: a write cut by a task abort or a shutdown is
not delivered later. The service's `WatchGuard` `UNWATCH` funnel, which gives
up on its `UNWATCH` after its deadline, relies on the `txn` pool's checkout
probe for an `UNWATCH` fred had not yet written.

**Reverting the change**, hook by hook (each is one existing line):

- single commands: on `src/utils.rs:402` and `:459` delete the `let abandon =
  …;` statement, and on `:408`, `:414` and `:474` replace `abandon.watch(rx)`
  with `rx` (fred's own timeout and error still set the flag, as stock);
- pipelines: on `src/clients/pipeline.rs:268`, `:295` and `:313` delete the
  `let abandon = …;` statement, and on `:275`, `:300` and `:318` replace
  `abandon.watch(rx)` with `rx`;
- `Transaction::exec`: on `src/clients/transaction.rs:316` restore `let
  command`, on `:321` delete the `let abandon = …;` statement, and on `:325`
  replace `abandon.watch(rx)` with `rx`;
- the pipeline `DISCARD`: on `src/router/commands.rs:235` delete the `let mut
  abandon = …;` statement, on `:244` the `abandon.before_write(…);` call, and
  restore `:247` to `Ok(())`;
- the transaction router: on `src/router/transactions.rs:121`, `:145`, `:174`,
  `:189` and `:196` delete the leading `if … given_up … { … }` statement (each
  check reverts alone), and restore `:181` to `for command in
  commands.iter_mut() {`;
- the close marker: in `src/live_edit/mod.rs`'s `close`, delete the
  `closes_at_once` check.

Then `src/live_edit/abandon.rs` and its `pub mod abandon;` line, with
`ConnectionState::abandon`, go.

## How the copy is wired in

- `Cargo.toml`: `[patch.crates-io] fred = { path = "vendor/fred-10.1.0" }`,
  which covers both the dependency and the dev-dependency, and the tree listed
  in `[workspace] exclude` so it is not a workspace member. `Cargo.lock`
  records fred with no `source` or `checksum`.
- `scripts/check-vendored-fred.sh`, run by the `fred:vendor-drift-check` CI
  job, rebuilds the tree from the crates.io archive and this patch and fails on
  any difference (in the GitHub source release only, `Cargo.lock` and
  `.vscode/settings.json` may be absent; see the top of this file), on a quoted
  diff above that differs from the patch file, or
  on a `Cargo.toml` that no longer patches fred to this tree under
  `[patch.crates-io]` (the same line under another table does not count).
- The root `Cargo.toml`'s `test-support` feature,
  `["dep:libc", "fred/live-edit-retention-report"]`, is the only edge that
  enables change 7's test-only report (`fred::live_edit::retention::test_support`);
  the release image is built with `cargo build --locked --release --workspace`
  and no features (`Dockerfile`), so it never contains it. The same script
  fences that: on the comment-stripped root `Cargo.toml` the feature must
  appear exactly once, on the `test-support = [...]` line; `cargo tree --locked
  --workspace --target all -e features -i fred` with no features selected must
  not list it on fred, which catches an edge the manifest check cannot see (a
  workspace member depending on live-edit with `test-support`, a `default`
  entry) and counts dev-dependency edges too, deliberately stricter than the
  image; and a control run with `--features live-edit/test-support` must
  list it, so a change in `cargo tree`'s output format fails the check rather
  than passing it. Each cause (an enabling edge, `cargo tree` unable to run,
  a failed control) has its own failure text.
- The same script keeps fred's `replicas` feature out of every build, because
  change 5's inline-connect flush does not cover the replica connect loops it
  compiles in, and its check before a write skips a command routed to a
  replica: `cargo tree --locked --workspace --target all -e features -i
  fred --all-features` must not list it on fred. `--all-features` selects
  every feature of every workspace member at once, and features only add, so
  a feature off there is off in every build the workspace can make; the
  retention fence's control run already shows `cargo tree` prints a resolved
  fred feature in the form this check matches. An enabling edge and a
  `cargo tree` that cannot run each have their own failure text.
- `scripts/audit-vendored-fred.sh`, run by the same job after the drift check,
  is fred's only RustSec check: with no `source` in `Cargo.lock`, fred is
  skipped by both `supply-chain:audit` (cargo-audit) and `supply-chain:deny`
  (cargo-deny). It audits the crates.io release this tree was vendored from,
  through a one-package lockfile carrying the crates.io source and the checksum
  above, with `cargo audit --color never --deny warnings`. It fails on any
  advisory, informational ones (unmaintained, unsound) included, and on a yank
  when the crates.io index is reachable (otherwise cargo-audit skips the yank
  check); an audit that cannot run fails as a separately reported case.
  cargo-audit is pinned by `CARGO_AUDIT_VERSION` in `.gitlab-ci.yml`, shared
  with `supply-chain:audit`. The job gates release.
- `tools/package_oss.py` ships the tree, the patch, this file and the
  upstream-file manifest in the OSS source archive under
  `sources/patched-crates/`, because `cargo vendor` skips path packages.
- `tools/generate_attributions.py` (`VENDORED_CRATES`) attributes fred's
  license from the tree, because `dd-rust-license-tool` lists only packages
  with a source. It fails on a source-less package that is neither a workspace
  member nor a `VENDORED_CRATES` entry at its locked version, so a re-vendor
  that misses the entry fails rather than dropping fred from `ATTRIBUTIONS.md`.
- `tools/license_headers.py` checks the tree file by file against
  `vendor/fred-10.1.0.SHA256SUMS` and the patch (`VENDORED_TREES`): a file the
  manifest lists keeps upstream's notice and must not carry NVIDIA's header,
  and if its bytes differ from the listed digest the patch must edit it; any
  other file is an NVIDIA addition, which the patch must add and which carries
  NVIDIA's Apache-2.0 header.
- The crate's own `.gitignore` lists `Cargo.lock`, so
  `vendor/fred-10.1.0/Cargo.lock` must be added with `git add -f` when the tree
  is first committed or re-vendored. The drift job fails if it is missing.

## Re-vendoring

To move to another fred release that still needs the patch:

1. Download `https://static.crates.io/crates/fred/fred-<version>.crate` and
   check its SHA-256 against the crates.io index (or against a `Cargo.lock`
   that resolved it from the registry).
2. Extract it into `vendor/`: `tar -xzf fred-<version>.crate -C vendor`.
3. Before applying anything, regenerate the upstream-file manifest from inside
   `vendor/fred-<version>` with the `find … | xargs -0 sha256sum` command
   under *Upstream source*, writing `../fred-<version>.SHA256SUMS`, and
   update the file count stated there.
4. Re-apply the change. `patch -p1 --fuzz=0 -d vendor/fred-<version> -i
   vendor/fred-10.1.0.patch`; if it does not apply, make the edit by hand,
   keeping the line count, and regenerate the patch with
   `diff -u --label a/<path> --label b/<path> <extracted original> <patched copy>`
   for each changed file, and `diff -u --label a/<path> --label b/<path>
   /dev/null <new file>` for each file under `src/live_edit/`, concatenated in
   byte-wise path order (`LC_ALL=C`).
5. Rename this file and the patch to the new version, update the checksum,
   date and quoted diff above, the counts of edited files under *License* and
   in `EDITED_COUNT` (`tools/tests/test_vendored_fred_scripts.py`) if the
   patch changed, and update every place that names the
   version: the `Cargo.toml` `[patch]` path and `exclude` entry,
   `VERSION` in `scripts/check-vendored-fred.sh`, `VENDORED_CRATES` in
   `tools/generate_attributions.py`, `VENDORED_TREES` in
   `tools/license_headers.py`, the fred entries in
   `tools/license-classification.json`, the fred paragraph of `NOTICE`,
   README § License, `oss/SOURCE-INVENTORY.md` §§ 2, 4 and 6, the fred row
   and the *License* column note of `oss/RUST-DEPENDENCY-LICENSES.md`, the
   vendored-trees license-header rule in the contributor guide,
   the `live-edit patch` comments in the patched files, and the
   `fred-10.1.0/...:NNN` citations (a new release moves line numbers).
6. Regenerate `Cargo.lock` and `ATTRIBUTIONS.md`, run
   `sh scripts/check-vendored-fred.sh` and `sh scripts/audit-vendored-fred.sh`,
   and commit the old tree's removal with the new tree (`git add -f` its
   `Cargo.lock`). When bumping `CARGO_AUDIT_VERSION`, re-check the cargo-audit
   summary wording the audit script matches (`error: <n> vulnerabilities
   found!` / `error: <n> denied warnings found!`, from the new release's
   `presenter.rs`).

## Dropping the patch

When an upstream release carries every change above, or when the service moves
off fred (a replacement client must provide the same behaviours: an `-ASK`
with a write attempt left is followed to its target with `ASKING` and the
command on one connection and no reply read out of order, any connection it
opens to that target bounded so an unreachable node cannot stall the client
for its full connection timeout, a script answered `-ASK` never followed, a
redirect
that cannot be followed fails only its command, a silent connection is detected
by its oldest outstanding command with peer evidence and its verdict says which
rule fired, no wait on one connection's I/O is unbounded, a connection judged
dead is closed without waiting on it, a data reply is never
taken for a pubsub frame, a fed command is never left unsent while the client
waits, connects inline or writes to another connection, a pool member can be swapped for a fresh client, a command, pipeline or
transaction is never written after its caller gave up on it (its timeout
fired, the client answered it with an error, or its call was dropped), a
pipeline cut during the step that writes it and a transaction cut partway never
leave a `MULTI` open (a pipeline replayed after a lost connection is not
covered; the service sends every `MULTI` pipeline at one write attempt, so it
never replays one), and what a client
keeps once its traffic drains is bounded with one warm connection per client
at its class limits and every other connection, and the backchannel, at a
small idle shape, so the service's memory bound for its connections
(`src/storage/connection_buffers.rs`) still holds and does not grow with the
store's primary count, or is re-derived for the new client):

1. Remove the `[patch.crates-io]` entry and the `exclude` entry from
   `Cargo.toml`, and the `fred/live-edit-retention-report` entry from its
   `test-support` feature, together with the tests' calls into
   `fred::live_edit::retention::test_support` (the connection buffer tests'
   reports, sweep modes, sweep counters and in-sweep flag, and the
   write-check measurement's counters); then regenerate
   `Cargo.lock`; fred regains its
   registry `source` and `checksum`.
2. Delete `vendor/fred-10.1.0/`, this file, the patch, the upstream-file
   manifest `vendor/fred-10.1.0.SHA256SUMS` (with `VendoredTreeRecordTests` in
   `tools/tests/test_vendored_fred_scripts.py`, which reads it) and
   `vendor/fred-10.1.0.UPSTREAM.md`, the `fred:vendor-drift-check` job and its
   entries in the release-gate `needs` lists, `scripts/check-vendored-fred.sh`,
   `scripts/audit-vendored-fred.sh` (the job's audit step goes with the job;
   `CARGO_AUDIT_VERSION` stays for `supply-chain:audit`), the
   `VENDORED_CRATES` entry in `tools/generate_attributions.py`, the
   `VENDORED_TREES` entry in `tools/license_headers.py` and the fred
   entries in `tools/license-classification.json`.
3. Rewrite what states that the copy exists: the fred paragraph of
   `NOTICE`, README § License, `oss/SOURCE-INVENTORY.md` §§ 1, 2, 4 and 6,
   the fred row and the *License* column note of
   `oss/RUST-DEPENDENCY-LICENSES.md` (fred becomes a `crates.io` MIT row),
   and the vendored-trees license-header rule in the contributor guide.
4. Regenerate `ATTRIBUTIONS.md`; fred is then attributed from its registry
   archive again.

The router supervisor stays as long as the service uses fred: it covers the
router-ending paths the patch does not touch.

## Upstream

Upstream `main` still has the stock `process_ask` (both its router exit and
its `-ASK` follow), unresponsive check, router
waits, pubsub classification, flush gate, command-queue counter, `Pool`,
connection buffers and give-up handling of pipelines, transactions and dropped
calls (identical to 10.1.0 as of 2026-10-15: `main` is still the 10.1.0
release commit `f222ad7b`). The project is dormant: no commit since the 10.1.0 release on
2025-02-27, issue #377 ("is this repo dead?", August 2026) is unanswered, and
the open pull requests from September 2026 (#380, #381) have no review. No
upstream issue reports these bugs. The issue and pull-request text, ready to
file, is in `vendor/fred-10.1.0.UPSTREAM.md`; record the links here once they
are filed.
