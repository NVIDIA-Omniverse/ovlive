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

//! Shared harness for the b.zt3 suites (`b_zt3_*`): a `ValkeyClient` behind a
//! [`ScriptReplyProxy`], a rule that silences the connection carrying a marker
//! `GET`, a member that keeps reading or echoing to one cluster node (the peer
//! replies a fast verdict may take as evidence), bounded polling, the metric
//! names, and the bridge that lets a test capture fred's own log lines.
//!
//! fred logs through the `log` crate, not `tracing`. [`bridge_fred_log`]
//! installs `tracing_log`'s `LogTracer` (through `tracing_subscriber`'s global
//! `try_init`, with a registry that records nothing), so a fred line becomes a
//! `tracing` event the thread-local `ThreadCapture` sees. It is process-wide
//! and first-come: under nextest every test is its own process, so it always
//! lands; under a plain `cargo test` of the whole binary another test may have
//! set the global subscriber first, and the fred-line assertions then see
//! nothing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::clients::Client;
use fred::interfaces::{ClientInterface, KeysInterface};
use fred::types::config::Server;
use tokio::time::Instant;

use live_edit::storage::client::{
    with_liveness_overrides_for_test, LivenessOverrides, ValkeyClient,
};
use live_edit::storage::test_support::connect_test_client;

use super::snapshot_read_support::{ProxyAction, ProxyRule, ScriptReplyProxy};

/// Bound on connecting a client through a proxy.
pub(crate) const CONNECT_BUDGET: Duration = Duration::from_secs(30);

/// How often a [`PeerReads`] member reads: the shard is seen answering
/// another connection well after any stall began (the fast verdict wants a
/// peer reply at least the 1 s evidence lag after it).
pub(crate) const PEER_READ_INTERVAL: Duration = Duration::from_millis(200);

/// `live_edit_valkey_router_restarts_total{pool}`.
pub(crate) const ROUTER_RESTARTS: &str = "live_edit_valkey_router_restarts_total";

/// `live_edit_valkey_unresponsive_total{pool, verdict}`.
pub(crate) const UNRESPONSIVE: &str = "live_edit_valkey_unresponsive_total";

/// `live_edit_valkey_member_replacements_total{pool, outcome}`.
pub(crate) const MEMBER_REPLACEMENTS: &str = "live_edit_valkey_member_replacements_total";

/// A proxy rule stalling the connection that carries a `GET` of one of
/// `markers`, the first time each is sent: the `GET` re-sent on a rebuilt
/// connection is forwarded.
pub(crate) fn stall_marker_gets(markers: &[String]) -> ProxyRule {
    let left: Mutex<HashMap<Vec<u8>, usize>> = Mutex::new(
        markers
            .iter()
            .map(|marker| (marker.as_bytes().to_vec(), 1))
            .collect(),
    );
    Arc::new(move |args: &[Vec<u8>]| {
        if !args.first()?.eq_ignore_ascii_case(b"GET") {
            return None;
        }
        let mut left = left.lock().unwrap();
        let stalls = left.get_mut(args.get(1)?.as_slice())?;
        (*stalls > 0).then(|| {
            *stalls -= 1;
            ProxyAction::Stall
        })
    })
}

/// A `ValkeyClient` with a `data_members`-member data pool connected through
/// `proxy`, its unresponsive bounds replaced by `overrides` where they name
/// one ([`LivenessOverrides`]).
pub(crate) async fn proxied_client(
    proxy: &ScriptReplyProxy,
    data_members: u32,
    overrides: LivenessOverrides,
) -> ValkeyClient {
    proxied_client_with(proxy, data_members, overrides, |_| {}).await
}

/// [`proxied_client`], with `configure` applied to the service config last
/// (for example a longer command timeout).
pub(crate) async fn proxied_client_with(
    proxy: &ScriptReplyProxy,
    data_members: u32,
    overrides: LivenessOverrides,
    configure: impl FnOnce(&mut live_edit::config::Config),
) -> ValkeyClient {
    let mut config = crate::common::test_config();
    config.valkey_url = proxy.url();
    config.valkey_pool_size = data_members;
    configure(&mut config);
    tokio::time::timeout(
        CONNECT_BUDGET,
        with_liveness_overrides_for_test(overrides, connect_test_client(&config)),
    )
    .await
    .expect("the client must connect through the proxy within the budget")
    .expect("the client must connect through the proxy")
}

/// Write `value` at each of `keys`, through `client`.
pub(crate) async fn write_keys(client: &ValkeyClient, keys: &[String], value: &str) {
    for key in keys {
        let _: () = client
            .pool
            .set(key, value, None, None, false)
            .await
            .expect("the key must be written");
    }
}

/// A client reading one key ([`PeerReads::start`]), or echoing to one cluster
/// node ([`PeerReads::echoes_on`]), every [`PEER_READ_INTERVAL`] until dropped.
pub(crate) struct PeerReads(tokio::task::JoinHandle<()>);

impl PeerReads {
    pub(crate) fn start(client: Client, key: String) -> Self {
        Self(tokio::spawn(async move {
            loop {
                // Only the reply's arrival matters (it stamps the shared
                // peer-reply registry); a failed read is no evidence and
                // the next one tries again.
                let _: Result<Option<String>, _> = client.get(&key).await;
                tokio::time::sleep(PEER_READ_INTERVAL).await;
            }
        }))
    }

    /// A client sending `ECHO message` to the cluster node `node` every
    /// [`PEER_READ_INTERVAL`] until dropped: fred sends it to `node` whichever
    /// node owns the slot `message` hashes to.
    pub(crate) fn echoes_on(client: Client, node: Server, message: String) -> Self {
        Self(tokio::spawn(async move {
            let on_node = client.with_cluster_node(node);
            loop {
                // As in `start`: only the reply's arrival matters.
                let _: Result<String, _> = on_node.echo(message.as_str()).await;
                tokio::time::sleep(PEER_READ_INTERVAL).await;
            }
        }))
    }
}

impl Drop for PeerReads {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Polls until `cond` holds, by `deadline`.
pub(crate) async fn await_until(what: &str, deadline: Instant, cond: impl Fn() -> bool) {
    while !cond() {
        assert!(Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Route fred's `log` records into `tracing` for this process (see the module
/// docs). Idempotent.
pub(crate) fn bridge_fred_log() {
    use tracing_subscriber::util::SubscriberInitExt;
    static BRIDGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    BRIDGED.get_or_init(|| {
        // Fails only when a global subscriber (and with it the bridge) is
        // already installed.
        let _ = tracing_subscriber::registry().try_init();
    });
}

/// The messages of every captured event whose message contains `needle`.
pub(crate) fn messages_containing(
    captured: &crate::grpc::log_capture::Captured,
    needle: &str,
) -> Vec<String> {
    captured
        .events
        .iter()
        .filter_map(|e| e["fields"]["message"].as_str())
        .filter(|m| m.contains(needle))
        .map(str::to_owned)
        .collect()
}
