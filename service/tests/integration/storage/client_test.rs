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

// Integration tests for ValkeyClient: connection, data commands, pub/sub isolation,
// failure handling, and pool-size configuration.
//
// Requires a running Valkey instance. Address is read from TEST_VALKEY_URL,
// falling back to LIVE_EDIT_VALKEY_ADDR, then defaulting to redis://127.0.0.1:6379.
// Tests skip gracefully if Valkey is unreachable.

use std::time::Duration;

use fred::{
    interfaces::{ClientLike, EventInterface, KeysInterface, PubsubInterface},
    types::Message,
};
use tokio::sync::oneshot;
use uuid::Uuid;

use live_edit::config::Config;
use live_edit::storage::client::{ConnectError, ValkeyClient};
use live_edit::storage::test_support::TestValkey;

/// Build a minimal Config pointing at the local test Valkey instance.
///
/// b.ohv item 8: delegate to `common::test_config()` and override only the
/// pool size. The pre-migration version of this helper duplicated every
/// `Config` field; new upstream fields had to be added to every site.
fn test_config(pool_size: u32) -> Config {
    let mut cfg = crate::common::test_config();
    cfg.valkey_pool_size = pool_size;
    cfg
}

/// Return a unique per-test key prefix, e.g. `le:test:<uuid>:`.
fn test_prefix() -> String {
    format!("le:test:{}:", Uuid::new_v4())
}

/// Connect to Valkey; return None (and skip) if the instance is not reachable.
async fn maybe_connect(pool_size: u32) -> Option<ValkeyClient> {
    let cfg = test_config(pool_size);
    match tokio::time::timeout(
        Duration::from_secs(5),
        live_edit::storage::test_support::connect_test_client(&cfg),
    )
    .await
    {
        Ok(Ok(client)) => Some(client),
        _ => None,
    }
}

// ── t3.1it.ra.in.2c ──────────────────────────────────────────────────────────

#[tokio::test]
async fn valkey_client_ping_data_pool_succeeds() {
    let Some(client) = maybe_connect(1).await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    client
        .pool
        .ping::<()>(None)
        .await
        .expect("PING on data pool must succeed");
}

#[tokio::test]
async fn valkey_client_set_and_get_round_trip_returns_stored_value() {
    let Some(client) = maybe_connect(1).await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let prefix = test_prefix();
    let key = format!("{prefix}roundtrip");
    let value = "hello-live-edit";

    client
        .pool
        .set::<(), _, _>(&key, value, None, None, false)
        .await
        .expect("SET must succeed");

    let retrieved: String = client.pool.get(&key).await.expect("GET must succeed");

    assert_eq!(
        retrieved, value,
        "retrieved value must match what was stored"
    );

    // Cleanup — best effort.
    let _ = client.pool.del::<(), _>(&key).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn valkey_client_pubsub_does_not_block_concurrent_data_commands() {
    let Some(client) = maybe_connect(2).await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let prefix = test_prefix();
    let channel = format!("{prefix}chan");
    let data_key = format!("{prefix}data");

    // Subscribe to the channel.
    client
        .subscriber
        .subscribe(&channel)
        .await
        .expect("SUBSCRIBE must succeed");

    // Set up a receiver so we can wait for the published message without sleeping.
    let mut msg_rx = client.subscriber.message_rx();
    let channel_clone = channel.clone();

    let (msg_tx, msg_rx_oneshot) = oneshot::channel::<Message>();
    let msg_tx = std::sync::Arc::new(std::sync::Mutex::new(Some(msg_tx)));

    tokio::spawn(async move {
        while let Ok(msg) = msg_rx.recv().await {
            if msg.channel.as_bytes() == channel_clone.as_bytes() {
                let mut guard = msg_tx.lock().unwrap();
                if let Some(tx) = guard.take() {
                    let _ = tx.send(msg);
                }
                break;
            }
        }
    });

    // Issue a data SET on the pool concurrently while subscribed.
    client
        .pool
        .set::<(), _, _>(&data_key, "concurrent-value", None, None, false)
        .await
        .expect("SET on pool while subscriber is active must not block");

    // Publish a message via an individual pool client (Pool does not implement
    // PubsubInterface; use pool.next() to get a single Client).
    client
        .pool
        .next()
        .publish::<(), _, _>(&channel, "test-message")
        .await
        .expect("PUBLISH must succeed");

    let received = tokio::time::timeout(Duration::from_secs(5), msg_rx_oneshot)
        .await
        .expect("message must arrive within 5s")
        .expect("message channel must not be dropped");

    let got = received.value.as_str().unwrap_or_default();
    assert_eq!(
        got.as_ref(),
        "test-message",
        "received pub/sub message must match published value"
    );

    // Cleanup — best effort.
    let _ = client.pool.del::<(), _>(&data_key).await;
    let _ = client.subscriber.unsubscribe(&channel).await;
}

#[tokio::test]
async fn valkey_client_unreachable_address_returns_err_not_panic() {
    // Port 1 is reserved/unreachable; the connect attempt must return Err, not panic.
    //
    // We set a short 2 s connect timeout so the retry budget exhausts quickly.
    // The 5 s outer timeout ensures the test fails definitively (rather than
    // hanging forever) if the implementation never returns.
    let mut cfg = test_config(1);
    cfg.valkey_url = "redis://127.0.0.1:1".to_string();
    cfg.valkey_connect_timeout_secs = 2;

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        ValkeyClient::connect(&cfg, None, None),
    )
    .await;

    match result {
        Ok(Ok(_)) => panic!("connecting to port 1 should not succeed"),
        Ok(Err(_)) => { /* correct: retry budget exhausted, returned Err */ }
        Err(_) => panic!("connect to unreachable address never returned within 5 s"),
    }
}

#[tokio::test]
async fn valkey_client_pool_size_configuration_respected() {
    const POOL_SIZE: u32 = 3;

    let Some(client) = maybe_connect(POOL_SIZE).await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    assert_eq!(
        client.pool.size(),
        POOL_SIZE as usize,
        "pool.size() must match the configured valkey_pool_size"
    );
}

/// Server-side `CLIENT ID` of every connection in `pool`.
async fn connection_ids(pool: &fred::clients::Pool) -> Vec<i64> {
    let mut ids = Vec::new();
    for conn in pool.clients() {
        ids.push(
            conn.custom::<i64, String>(
                fred::types::CustomCommand::new_static("CLIENT ID", None::<u16>, false),
                vec![],
            )
            .await
            .expect("CLIENT ID must succeed on a connected pool"),
        );
    }
    ids
}

/// The snapshot lane (Spec SDD SR-10.1) is built at connect with
/// `LIVE_EDIT_SNAPSHOT_POOL_SIZE` connections of its own: every one answers a
/// command once `connect` returns, and none of them is a data-pool connection —
/// a lane that shared the data pool's sockets would leave a multi-MiB snapshot
/// reply queued in front of a writer's command, which is what it exists to
/// prevent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_builds_the_snapshot_lane_with_its_own_configured_connections() {
    const SNAPSHOT_POOL_SIZE: u32 = 3;

    let mut cfg = test_config(2);
    cfg.snapshot_pool_size = SNAPSHOT_POOL_SIZE;
    let Ok(Ok(client)) = tokio::time::timeout(
        Duration::from_secs(5),
        live_edit::storage::test_support::connect_test_client(&cfg),
    )
    .await
    else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    assert_eq!(
        client.snapshot_pool.size(),
        SNAPSHOT_POOL_SIZE as usize,
        "the lane must have exactly LIVE_EDIT_SNAPSHOT_POOL_SIZE connections"
    );
    let lane_ids = connection_ids(&client.snapshot_pool).await;
    let data_ids = connection_ids(&client.pool).await;
    let distinct: std::collections::HashSet<i64> = lane_ids.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        lane_ids.len(),
        "every lane connection must be its own socket; got CLIENT IDs {lane_ids:?}"
    );
    assert!(
        lane_ids.iter().all(|id| !data_ids.contains(id)),
        "no lane connection may be a data-pool connection; lane {lane_ids:?}, data {data_ids:?}"
    );

    client.quit().await;
}

/// `ValkeyClient::quit` closes the snapshot lane with the other pools: once it
/// returns, the server no longer lists any of the lane's connections.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quit_closes_every_snapshot_lane_connection() {
    use fred::interfaces::ClientInterface;

    let mut cfg = test_config(1);
    cfg.snapshot_pool_size = 2;
    let Ok(Ok(client)) = tokio::time::timeout(
        Duration::from_secs(5),
        live_edit::storage::test_support::connect_test_client(&cfg),
    )
    .await
    else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    let Some(observer_client) = maybe_connect(1).await else {
        eprintln!("SKIP: second Valkey connection not reachable");
        return;
    };
    let observer = observer_client.pool.next().clone();

    let lane_ids: Vec<String> = connection_ids(&client.snapshot_pool)
        .await
        .into_iter()
        .map(|id| id.to_string())
        .collect();
    let listed = || async {
        observer
            .client_list::<String, String>(None, Some(lane_ids.clone()))
            .await
            .expect("CLIENT LIST ID must succeed on the observer connection")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count()
    };
    assert_eq!(
        listed().await,
        lane_ids.len(),
        "precondition: the server lists every lane connection before quit"
    );

    client.quit().await;

    // The server drops a connection when it reads QUIT or sees the close; poll
    // briefly rather than assume it has happened by the time quit returns.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let remaining = listed().await;
        if remaining == 0 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{remaining} snapshot-lane connection(s) still open after quit — the lane is missing \
             from ValkeyClient::quit"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    observer_client.quit().await;
}

/// `Blocking::Interrupt` is enabled on the stream readers' clients and on
/// NOTHING else. (A reader's client exists once the reader has first served a
/// subscription: readers connect lazily.)
///
/// Why the readers need it: each is parked in `XREAD BLOCK`. Since b.19i a
/// reader sends its other commands, its credential re-AUTH included, only
/// between reads, and fred runs no credential refresh on it, so what is left is
/// the residual case: the reader's next command after a client-side `XREAD`
/// timeout that did not force a reconnect would queue behind up to 30 s of
/// that still-blocked read. `Interrupt` makes fred clear the block with
/// `CLIENT UNBLOCK` first; the `-UNBLOCKED` goes to the timed-out read's
/// already-dropped caller, so nothing is re-planned or counted.
///
/// Why it must not spread: the data / txn / maintenance / snapshot pools and the
/// pub/sub connection are MULTIPLEXED. A command queued behind another there is
/// ordinary pipelining, so `Interrupt` would have fred fire `CLIENT UNBLOCK`
/// against healthy traffic.
///
/// Read back through fred's `ClientLike::client_config()`, i.e. off the live
/// clients each pool actually built. The source-level single-mutation-site
/// guard lives in `src/storage/client.rs`
/// (`connect_sets_blocking_interrupt_exactly_once_on_the_subscriber_pool_config`);
/// neither subsumes the other.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_enables_blocking_interrupt_on_the_subscriber_pool_only() {
    use fred::types::config::Blocking;

    let Some(client) = maybe_connect(2).await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    // Readers connect lazily, the first time they serve a subscription, each
    // from the readers' one config template: connect reader 0 (standalone
    // routing sends object 0 to it) and read the policy off its client.
    super::mux_test_helpers::connect_reader_zero(&client).await;

    let readers = client.stream_mux.reader_clients_for_test();
    assert!(!readers.is_empty());
    for reader in readers {
        assert_eq!(
            reader.client_config().blocking,
            Blocking::Interrupt,
            "every stream reader must carry Blocking::Interrupt so a command queued behind \
             its XREAD BLOCK clears it with CLIENT UNBLOCK instead of waiting out the block"
        );
    }

    let txn_slot = client.acquire_txn().await;
    let non_blocking: [(&str, Blocking); 5] = [
        ("data pool", client.pool.next().client_config().blocking),
        ("txn pool", txn_slot.client_config().blocking),
        (
            "maintenance pool",
            client.maintenance_pool.next().client_config().blocking,
        ),
        (
            "snapshot pool",
            client.snapshot_pool.next().client_config().blocking,
        ),
        (
            "pub/sub subscriber",
            client.subscriber.client_config().blocking,
        ),
    ];
    for (label, blocking) in non_blocking {
        assert_eq!(
            blocking,
            Blocking::Block,
            "the {label} must keep fred's default Blocking::Block: it is multiplexed and \
             issues no blocking commands, so a command queued behind another is normal \
             pipelining — enabling Interrupt there would CLIENT UNBLOCK live traffic"
        );
    }
    drop(txn_slot);

    // Guard the premise: every pool above is built from the same
    // `build_fred_config` output, so `Blocking::Block` on five of them only
    // proves scoping while it is still fred's default. If a fred bump flipped
    // the default, the five assertions above would be asserting the mutation
    // rather than its absence.
    assert_eq!(
        Blocking::default(),
        Blocking::Block,
        "precondition: fred's Blocking default must be Block — if it ever changes, the \
         five non-subscriber assertions above stop meaning \"undiverged\""
    );

    client.quit().await;
}

/// Connection buffer limits (`storage::connection_buffers`) reach every
/// client `connect` builds: each member of the data, txn, maintenance and
/// snapshot pools and the pub/sub client carries `GENERAL_LIMITS`, and every
/// stream reader carries `READER_LIMITS`. Read back through fred's
/// `ClientLike::connection_config()` off the live clients, so a builder that
/// misses `apply_limits` — or sets it after the clone or build that the
/// pool is made from — fails here. It also holds `PoolSizes::from_config`,
/// which `at_rest_bound_bytes` counts clients from, to the general clients
/// and stream readers `connect` built from the same config. A reader's client
/// exists once the reader has first served a subscription (readers connect
/// lazily, each from the `reader_builder` template).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_sets_the_connection_buffer_limits_of_its_class_on_every_client() {
    use live_edit::storage::connection_buffers::{GENERAL_LIMITS, READER_LIMITS};

    assert_ne!(
        GENERAL_LIMITS, READER_LIMITS,
        "precondition: the two classes differ, so a client given the other class's limits fails"
    );
    let Some(client) = maybe_connect(2).await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let mut general: Vec<(String, Option<fred::live_edit::retention::RetentionLimits>)> =
        Vec::new();
    for (label, pool) in [
        ("data pool", &client.pool),
        ("maintenance pool", &client.maintenance_pool),
        ("snapshot pool", &client.snapshot_pool),
    ] {
        for (i, member) in pool.clients().iter().enumerate() {
            general.push((
                format!("{label} member {i}"),
                member.connection_config().live_edit.retention,
            ));
        }
    }
    // Every txn slot, each held at once so each checkout is a different slot.
    let mut txn_slots = Vec::new();
    for _ in 0..client.txn_pool.slots() {
        txn_slots.push(client.acquire_txn().await);
    }
    for (i, slot) in txn_slots.iter().enumerate() {
        general.push((
            format!("txn pool checkout {i}"),
            slot.connection_config().live_edit.retention,
        ));
    }
    drop(txn_slots);
    general.push((
        "pub/sub subscriber".to_owned(),
        client.subscriber.connection_config().live_edit.retention,
    ));
    assert_eq!(
        general.len(),
        2 * 2 + 1 + 1 + 1,
        "fixture self-check: every member of the two pools of 2, the maintenance and snapshot \
         pools of 1, and the pub/sub client"
    );
    for (label, retention) in &general {
        assert_eq!(
            *retention,
            Some(GENERAL_LIMITS),
            "the {label} must carry the general connection buffer limits"
        );
    }
    // `PoolSizes::from_config` is what `at_rest_bound_bytes` counts clients
    // from (the upload memory envelope's test 11 bound): it must count the
    // general clients and the readers `connect` built from the same config, so
    // an over-count cannot loosen that bound unseen.
    let pools = live_edit::storage::connection_buffers::PoolSizes::from_config(&test_config(2));
    assert_eq!(
        pools.general_clients(),
        general.len(),
        "PoolSizes::from_config counts the general clients connect built"
    );
    assert_eq!(
        pools.readers,
        client.stream_mux.snapshot().readers.len(),
        "PoolSizes::from_config counts the stream readers connect built"
    );

    super::mux_test_helpers::connect_reader_zero(&client).await;
    let readers = client.stream_mux.reader_clients_for_test();
    assert!(
        !readers.is_empty(),
        "fixture self-check: a reader connected to serve the subscription"
    );
    for reader in readers {
        assert_eq!(
            reader.connection_config().live_edit.retention,
            Some(READER_LIMITS),
            "every stream reader built from the reader template must carry the reader limits"
        );
    }

    client.quit().await;
}

/// The connection buffer limits survive a member replacement
/// (`storage::member_replacement`): the fresh client swapped in for a data,
/// maintenance or snapshot pool member, or for a `txn` slot, carries
/// `GENERAL_LIMITS`, as `at_rest_bound_bytes` counts it (in K while it
/// connects, as a member once it takes over). That rests on the replacement
/// building its fresh client with `Client::clone_new`, which copies the old
/// client's connection config; a fresh client built without the old one's
/// connection config, or from a copy that dropped `live_edit.retention`,
/// fails here. Each fresh client is checked twice: its connection config, and
/// its retention report ([`retention_report`]), whose every connection's
/// codec must carry the limits, which also shows the fresh client answers
/// from report state of its own. Each replacement is driven by
/// `ValkeyClient::fast_verdict_for_test`, counted against the member's
/// current client as its error hook counts a real verdict, with the
/// production trigger and a zero retirement grace, so each old client's
/// retirement ends at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_member_replacements_fresh_client_carries_the_general_connection_buffer_limits() {
    use live_edit::storage::client::{
        POOL_LABEL_DATA, POOL_LABEL_MAINTENANCE, POOL_LABEL_SNAPSHOT, POOL_LABEL_TXN,
    };
    use live_edit::storage::connection_buffers::GENERAL_LIMITS;
    use live_edit::storage::member_replacement::{
        ReplaceKnobs, MEMBER_REPLACE_AFTER_FAST_VERDICTS, MEMBER_REPLACE_COOLDOWN,
        MEMBER_REPLACE_WINDOW,
    };
    use tokio::time::Instant;

    /// Bound on each replacement's swap and on the replacements' end.
    const SETTLE: Duration = Duration::from_secs(10);

    let Some(client) = maybe_connect(2).await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    client.set_member_replacement_knobs_for_test(ReplaceKnobs {
        after_fast_verdicts: MEMBER_REPLACE_AFTER_FAST_VERDICTS,
        window: MEMBER_REPLACE_WINDOW,
        cooldown: MEMBER_REPLACE_COOLDOWN,
        grace: Duration::ZERO,
    });
    let trigger = |pool: &str, member: usize| {
        for _ in 0..MEMBER_REPLACE_AFTER_FAST_VERDICTS {
            client.fast_verdict_for_test(pool, member);
        }
    };

    let mut fresh: Vec<(String, fred::clients::Client)> = Vec::new();
    for (pool_label, pool) in [
        (POOL_LABEL_DATA, &client.pool),
        (POOL_LABEL_MAINTENANCE, &client.maintenance_pool),
        (POOL_LABEL_SNAPSHOT, &client.snapshot_pool),
    ] {
        let old = ClientLike::id(pool.member(0)).to_owned();
        trigger(pool_label, 0);
        super::b_zt3_support::await_until(
            &format!("the {pool_label} pool's member 0 must be swapped for a fresh client"),
            Instant::now() + SETTLE,
            || ClientLike::id(pool.member(0)) != old,
        )
        .await;
        fresh.push((
            format!("{pool_label} pool member 0's fresh client"),
            pool.member(0).clone(),
        ));
    }

    let old_slot = client.txn_slot_client_ids_for_test()[0].clone();
    trigger(POOL_LABEL_TXN, 0);
    super::b_zt3_support::await_until(
        "txn slot 0 must be swapped for a fresh client",
        Instant::now() + SETTLE,
        || client.txn_slot_client_ids_for_test()[0] != old_slot,
    )
    .await;
    let fresh_slot = client.txn_slot_client_ids_for_test()[0].clone();
    // Every slot held at once, so one of the checkouts is slot 0's.
    let mut held = Vec::new();
    for _ in 0..client.txn_pool.slots() {
        held.push(client.acquire_txn().await);
    }
    let slot = held
        .iter()
        .find(|checkout| ClientLike::id(&***checkout) == fresh_slot)
        .expect("fixture self-check: one checkout holds slot 0's fresh client");
    fresh.push(("txn slot 0's fresh client".to_owned(), (**slot).clone()));
    drop(held);

    super::b_zt3_support::await_until(
        "every replacement and its retirement must finish",
        Instant::now() + SETTLE,
        || client.member_replacement_tasks_for_test() == 0,
    )
    .await;
    assert_eq!(
        fresh.len(),
        4,
        "fixture self-check: one per replaceable pool"
    );
    let mut wrong = Vec::new();
    for (label, fresh_client) in &fresh {
        let configured = fresh_client.connection_config().live_edit.retention;
        if configured != Some(GENERAL_LIMITS) {
            wrong.push(format!(
                "{label}: its connection config carries {configured:?}"
            ));
        }
        let report = retention_report(label, fresh_client).await;
        if report.is_empty() || report.iter().any(|c| c.limits != Some(GENERAL_LIMITS)) {
            wrong.push(format!(
                "{label}: its connections' codecs carry {:?}",
                report.iter().map(|c| c.limits).collect::<Vec<_>>()
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "every fresh client a member replacement swaps in must carry the general connection \
         buffer limits ({GENERAL_LIMITS:?}), in its config and on each connection: {wrong:#?}"
    );

    client.quit().await;
}

/// `client`'s retention report (`fred::live_edit::retention::test_support`,
/// compiled in by `test-support`): one entry per connection, answered by the
/// first router pass whose sweep began after the request. Polled every 10 ms
/// for up to 10 s; panics naming `label` and the client when no answer comes
/// (a router ended by `QUIT` or an abort never answers). A fresh client's
/// report comes from its own report state, which `replace_member`'s fresh
/// `ClientInner` creates.
async fn retention_report(
    label: &str,
    client: &fred::clients::Client,
) -> Vec<fred::live_edit::retention::test_support::ConnectionReport> {
    use fred::live_edit::retention::test_support::{request_report, take_report};
    request_report(client);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(report) = take_report(client) {
            return report;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{label} (fred client {}): its router answered no retention report within 10 s",
            client.id()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// SR-1.1 regression (t3.tr5.t4.oq): a `rediss://` URL must attempt a TLS
/// handshake, not silently downgrade to plaintext. Pre-feature (no
/// `enable-rustls-ring`), fred's `.tls` config was always `None`, so a
/// `rediss://` URL parsed fine but connected in plaintext and this connect
/// SUCCEEDED. With the defaulted `valkey-tls` feature routing fred's rustls
/// connector, pointing a `rediss://` URL at the ordinary *plaintext* test
/// Valkey now fails the handshake: rustls sends a ClientHello, Valkey's RESP
/// parser rejects it (first byte back `-`/0x2d is not a valid TLS ContentType),
/// rustls errors, and the retry budget exhausts into `ConnectError::Timeout`.
/// That success→failure asymmetry is the fails-without-the-fix property.
#[tokio::test]
async fn valkey_client_rediss_url_against_plaintext_server_fails_to_connect() {
    // Plaintext control first: a `rediss://` failure is meaningless if Valkey
    // is simply down (the test would pass vacuously). Prove the plaintext path
    // connects, then drop the client immediately.
    let Some(control) = maybe_connect(1).await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    drop(control);

    // Scheme guard: only meaningful when TEST_VALKEY_URL is a plaintext
    // `redis://` endpoint. If an operator points it at a cluster or an
    // already-TLS endpoint, skip rather than produce a spurious result.
    let mut cfg = test_config(1);
    if !cfg.valkey_url.starts_with("redis://") {
        eprintln!("SKIP: TEST_VALKEY_URL is not a plaintext redis:// URL");
        return;
    }

    // Rewrite only the scheme prefix. A bare `str::replace` would corrupt a URL
    // whose password happens to contain the `redis://` substring.
    cfg.valkey_url = format!("rediss://{}", &cfg.valkey_url["redis://".len()..]);
    // Exhaust the retry budget fast.
    cfg.valkey_connect_timeout_secs = 2;

    let result = tokio::time::timeout(
        Duration::from_secs(15),
        ValkeyClient::connect(&cfg, None, None),
    )
    .await;

    let inner = match result {
        Ok(inner) => inner,
        Err(_) => panic!(
            "connect to rediss:// plaintext server never returned within 15 s \
             (the retry budget should have exhausted into ConnectError::Timeout)"
        ),
    };

    match inner {
        Ok(_) => panic!(
            "connecting a rediss:// URL to a plaintext Valkey SUCCEEDED — this is \
             the pre-fix silent-plaintext-downgrade regression: the TLS scheme was \
             honoured for parsing but ignored for transport (fred `.tls` was None)"
        ),
        Err(ConnectError::Timeout { .. }) => { /* correct: TLS handshake failed, retry budget exhausted */
        }
        Err(ConnectError::BadUrl(_)) | Err(ConnectError::Build(_)) => panic!(
            "rediss:// connect failed before reaching the transport (BadUrl/Build) — \
             the URL never got far enough to prove TLS was actually attempted"
        ),
        Err(other) => panic!(
            "expected ConnectError::Timeout from an exhausted TLS-handshake retry \
             budget, got: {other:?}"
        ),
    }
}

// ── Cluster-gated TLS checks (t3.tr5.t4.wz.64) ───────────────────────────────
//
// CEILING — READ BEFORE EXTENDING. The cluster these tests run against
// (`LIVE_EDIT_TEST_CLUSTER_URL`, set by CI's `test-cluster:` job and by
// `scripts/dev-valkey.sh cluster-up`) is PLAINTEXT. That is a hard limit on
// what they can prove. On a plaintext config fred's `modify_cluster_slot_hostnames`
// (`fred-10.1.0/src/protocol/cluster.rs:226`) early-returns because
// `inner.config.tls` is `None`, so the TLS host-mapping policy
// (`TlsHostMapping::DefaultHost`) is NEVER consulted on this path. Therefore
// NEITHER test below validates SR-1.5's actual host-mapping claim.
//
// What they DO prove, and only this:
//   1. Clustered TLS is genuinely attempted, not silently downgraded — a
//      `rediss-cluster://` URL pointed at the plaintext cluster must fail the
//      handshake (test 1). `from_url_clustered` is a separate initializer from
//      the standalone `from_url_centralized` that `t3.tr5.t4.oq.rb` covers, and
//      drives a different router path, so this is not redundant with that test.
//   2. Plaintext `redis-cluster://` still works end to end (test 2) — the Task's
//      explicit non-regression criterion, as named local evidence.
//
// ROLLOUT-ONLY, OWNED BY TASK 4 (`t2.tr5.t4.nq`): whether
// `TlsHostMapping::DefaultHost` actually makes cert verification succeed against
// a real managed clustered endpoint (Azure Managed Redis / AWS MemoryDB). Task 4
// owns the durable procedure: connect a `rediss-cluster://` `LIVE_EDIT_VALKEY_URL`
// at the real endpoint and confirm PING/health passes on ALL shards, not just the
// seed. Known failure signature to watch for: the SEED connection succeeds and
// only the post-topology-discovery NODE connections fail, with the failing host
// differing from the host in the URL. No plaintext cluster can reproduce that.

/// Test 1 — clustered TLS is attempted, not silently downgraded.
///
/// Pointing a `rediss-cluster://` URL at the plaintext gated cluster must fail
/// the handshake. Pre-Task-1 (`.tls` always `None`, so fred's dispatch at
/// `fred-10.1.0/src/protocol/connection.rs:983-991` never selected `new_rustls`)
/// the identical call connected successfully in plaintext — that success→failure
/// asymmetry is the fails-without-the-fix property. It also guards Task 3's own
/// change: an implementation that replaces `.tls` wholesale with a broken or
/// `None` connector turns this back into a successful plaintext connect.
///
/// Determinism mirrors the standalone `rediss://` test above: rustls sends a
/// ClientHello, Valkey's RESP parser rejects it (first byte back `-`/0x2d is not
/// a valid TLS ContentType), rustls errors, and the retry budget exhausts into
/// `ConnectError::Timeout`.
#[tokio::test]
async fn valkey_client_rediss_cluster_url_against_plaintext_cluster_fails_to_connect() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    if !cluster_url.starts_with("redis-cluster://") {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL is not a plaintext redis-cluster:// URL");
        return;
    }

    // Non-vacuity control: prove the plaintext cluster is up and genuinely
    // clustered before asserting anything about the TLS attempt.
    // `setup_with_url` panics if the cluster is unreachable, so a down cluster
    // fails loudly rather than letting the TLS assertion pass vacuously.
    let tv = TestValkey::setup_with_url(cluster_url.clone(), 1).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment \
         (use `redis-cluster://...`); the clustered-TLS attempt is only \
         meaningful when `is_clustered()` is true"
    );
    drop(tv);

    // Rewrite only the scheme prefix. A bare `str::replace` would corrupt a URL
    // whose credentials happen to contain the `redis-cluster://` substring.
    let rewritten = format!(
        "rediss-cluster://{}",
        &cluster_url["redis-cluster://".len()..]
    );

    let mut cfg = test_config(1);
    cfg.valkey_url = rewritten;
    // Exhaust the retry budget fast.
    cfg.valkey_connect_timeout_secs = 2;

    let result = tokio::time::timeout(
        Duration::from_secs(15),
        ValkeyClient::connect(&cfg, None, None),
    )
    .await;

    let inner = match result {
        Ok(inner) => inner,
        Err(_) => panic!(
            "connect to rediss-cluster:// plaintext cluster never returned within 15 s \
             (the retry budget should have exhausted into ConnectError::Timeout)"
        ),
    };

    match inner {
        Ok(_) => panic!(
            "connecting a rediss-cluster:// URL to a plaintext cluster SUCCEEDED — this is \
             the clustered silent-plaintext-downgrade regression: the TLS scheme was \
             honoured for parsing but ignored for transport (fred `.tls` was None on the \
             clustered path)"
        ),
        Err(ConnectError::Timeout { .. }) => { /* correct: TLS handshake failed, retry budget exhausted */
        }
        Err(ConnectError::BadUrl(_)) | Err(ConnectError::Build(_)) => panic!(
            "rediss-cluster:// connect failed before reaching the transport (BadUrl/Build) — \
             the URL never got far enough to prove clustered TLS was actually attempted"
        ),
        Err(other) => panic!(
            "expected ConnectError::Timeout from an exhausted TLS-handshake retry \
             budget, got: {other:?}"
        ),
    }
}

/// Test 2 — plaintext `redis-cluster://` still round-trips after the change.
///
/// This overlaps existing cluster-gated coverage CI already runs
/// (`src/storage/counters.rs`, `tests/integration/storage/orphan_sweep_test.rs`,
/// `tests/integration/checkpoint/e2e_cluster_recovery_test.rs`), all of which
/// reach `ValkeyClient::connect` through `TestValkey::setup_with_url`. Its value
/// here is being *named, local* evidence for the Task's plaintext-cluster
/// non-regression criterion, rather than a claim resting on unrelated tests.
#[tokio::test]
async fn valkey_client_plaintext_cluster_url_round_trips_after_tls_host_mapping_change() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    if !cluster_url.starts_with("redis-cluster://") {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL is not a plaintext redis-cluster:// URL");
        return;
    }

    let tv = TestValkey::setup_with_url(cluster_url, 2).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment \
         (use `redis-cluster://...`); the plaintext-cluster round-trip is only \
         meaningful when `is_clustered()` is true"
    );

    let key = format!("{}roundtrip", tv.prefix());
    let value = "hello-plaintext-cluster";

    tv.client()
        .pool
        .set::<(), _, _>(&key, value, None, None, false)
        .await
        .expect("SET on plaintext cluster must succeed");

    let retrieved: String = tv
        .client()
        .pool
        .get(&key)
        .await
        .expect("GET on plaintext cluster must succeed");

    assert_eq!(
        retrieved, value,
        "round-tripped value must match what was written to the plaintext cluster"
    );

    // TestValkey's Drop does the cluster-safe SCAN+UNLINK cleanup.
}
