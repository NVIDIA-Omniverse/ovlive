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

//! b.7td — `txn_pool` checkout hygiene: detect a slot carrying leaked
//! transaction state, recycle it, and keep the recycled slot instrumented.
//!
//! # The defect
//!
//! The field report was an `AUTH` rejected by the server with "AUTH … not
//! allowed in transactions", followed by credential-refresh timeouts,
//! "Unresponsive connection" events and a `DEADLINE_EXCEEDED` cascade on a
//! cluster-mode managed backing store. The `AUTH` is not the fault — it is the
//! first thing to *touch* a pooled connection that was already carrying an open
//! `MULTI` left behind by an earlier borrower, and it is the most likely thing
//! to touch it because the credential-refresh path fans out to every primary.
//!
//! fred can neither detect nor repair that residue (`skip_results()` is a no-op
//! on the transaction write path, `discard()` reads exactly one frame, and
//! `drop_broken_connections` / `force_reconnection()` only act on connections
//! that raised a *transport* error — a dirty-but-healthy socket is invisible to
//! them). So both halves are ours: `storage::txn_checkout::probe_txn_slot`
//! detects, and `ValkeyClient::acquire_txn*` repairs by recycling the slot.
//!
//! # What lives here
//!
//! | Test | What it pins |
//! |---|---|
//! | [`clean_txn_checkout_probes_the_slot_and_reports_no_dirty_verdict`] | the probe runs on **every** checkout (positively observed, server-side) and a healthy slot yields the Clean verdict — no counter cell moves |
//! | [`pinned_txn_checkout_detects_a_leaked_multi_recycles_and_lets_the_borrower_commit`] | the whole repair on `acquire_txn_for`: leaked `MULTI` detected, `reason="leaked_multi"` ticked exactly once, connection rebuilt, and the borrower's own `WATCH`/`MULTI`/`EXEC` commits on it |
//! | [`shim_txn_checkout_detects_a_leaked_multi_recycles_and_lets_the_borrower_commit`] | the same over the unpinned [`ValkeyClient::acquire_txn`] shim, which no production call site uses any more but several tests here and in sibling files still take a raw slot through |
//! | [`recycled_txn_slot_keeps_reporting_reconnects_and_errors`] | the blind-hooks regression: `quit()` swaps fred's broadcast senders, so a recycle that did not re-arm the metric hooks would leave that slot's `valkey_reconnects_total{pool="txn"}` / `valkey_errors_total{pool="txn"}` permanently silent |
//! | [`no_unpinned_acquire_txn_call_site_survives_under_src`] | the source-shape fence, now absolute over `src/`: a production call site on the unpinned shim probes an ARBITRARY cluster node, so a dirty slot passes its checkout and fails some later request |
//! | [`seam_driven`] (module) | every verdict a healthy server cannot produce, the unknown-state recycle gate, the state-proof repair-ineffective breaker, and the cluster-topology classifier arm (b.289 sub-task 3) — **requires `--features test-fault-injection`** |
//!
//! The five `reason` cells' pre-registration at zero on a fresh registry — and
//! the two cells of the breaker's own `live_edit_valkey_txn_repair_ineffective_total`
//! — are pinned next to their peers in
//! `tests/integration/observability_metrics_test.rs`. The per-node scoping of
//! the state-proof streak and breaker (b.289 sub-task 4) needs ≥2 primaries and
//! lives in `b_289_tests.rs`, behind the `LIVE_EDIT_TEST_CLUSTER_URL` gate;
//! everything here runs against one endpoint, where every state-proof verdict
//! lands in the single shared fallback bucket.
//! The rotation-gated re-`AUTH` half of b.7td (Change C) is unit-tested in
//! `src/storage/token_file_credentials.rs`, and the per-connection hook-handle
//! registry the recycle re-arms is pinned in-crate in `src/storage/client.rs`
//! (`mod tests::txn_slot_hook_registry`) — that map is private and has no
//! accessor, deliberately.
//!
//! # What none of these tests prove
//!
//! **They do not prove the field failure is gone.** The observed incident needs
//! a cluster-mode managed endpoint, a real token rotation and the proxy in
//! front of it; the `AUTH`-rejected-inside-`MULTI` shape cannot be produced
//! against a single-node Valkey at all. What is proven here is the mechanism
//! the fix rests on: that a leaked `MULTI` is detected on the next checkout
//! rather than inherited, that the slot is repaired before the borrower uses
//! it, and that the repair does not cost the pod its connection-health signal.
//!
//! # Why single-node, and no cluster-gated additions
//!
//! Every assertion here is about state carried on **one** connection, which a
//! single-node fixture reproduces exactly — unlike b.ji8's routing defect,
//! which was invisible without multiple primaries. A `LIVE_EDIT_TEST_CLUSTER_URL`
//! variant would add one thing in principle (that a pinned probe reaches the
//! borrower's own primary) and nothing in practice: the pin is
//! `ClusterHash::Custom(slot)` on the probe command, already pinned as a pure
//! property by `txn_checkout::tests::probe_commands_pin_the_callers_slot` and
//! exercised end-to-end by b.ji8's cluster tests over the identical
//! `ClusterHash::from(&str)` conversion. The routing hazard that a cluster
//! *would* catch — a probe landing on the wrong primary — is a property of the
//! proxy in front of a managed endpoint (an OSS cluster's own router is exactly
//! what the pin steers), and CI has no such proxy. So a cluster variant would
//! spend a shard fixture to re-assert a `ClusterHash` value two other tests
//! already own. It is deliberately not added.
//!
//! # Wait budgets vs. nextest's hang-killer
//!
//! `.config/nextest.toml` marks a test slow at 40 s and hard-terminates it at
//! 120 s, and a SIGKILLed test prints nothing. Only one test here blocks on a
//! background condition, and it holds ONE deadline over it:
//! [`recycled_txn_slot_keeps_reporting_reconnects_and_errors`] waits at most
//! [`RECONNECT_OBSERVATION_BUDGET`] (15 s). Every other body is a bounded
//! sequence of round trips against a local Valkey, plus the repair's own
//! `TXN_RECYCLE_*` budgets, which the production `const` assertion in
//! `storage::client` already bounds at 10 s.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::interfaces::{ClientLike, KeysInterface, TransactionInterface};
use fred::types::{CustomCommand, Resp3Frame, Value as FredValue};
use tracing_subscriber::fmt::MakeWriter;

use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::test_support::TestValkey;

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and vocabulary
// ═════════════════════════════════════════════════════════════════════════════

/// One deadline covering the "fred noticed the killed connection and rebuilt
/// it" wait in [`recycled_txn_slot_keeps_reporting_reconnects_and_errors`].
///
/// fred's reconnect backoff starts at 100 ms (`RECONNECT_MIN_DELAY_MS` in
/// `storage::client`), so the healthy path observes both counters inside a few
/// hundred milliseconds and pays none of this. 15 s is two orders of magnitude
/// above that — anything reaching it is a client that stopped reconnecting, not
/// a loaded runner — and stays well clear of the 120 s hang-killer so the
/// diagnostic panic below actually prints.
const RECONNECT_OBSERVATION_BUDGET: Duration = Duration::from_secs(15);

/// Probe granularity for the bounded counter wait. Small relative to fred's
/// 100 ms first reconnect delay so the observation is not a quantisation
/// artefact.
const COUNTER_PROBE_INTERVAL: Duration = Duration::from_millis(25);

/// Every `reason` cell of `live_edit_valkey_txn_dirty_checkouts_total`.
///
/// Restated here rather than imported: `TXN_DIRTY_REASONS` is `pub(crate)` in
/// `observability::metrics` and an integration test is a separate crate. That
/// is the same standing the sibling label-set assertions in
/// `observability_metrics_test.rs` have, and the duplication is the point — a
/// silent rename of a `reason` must fail a test rather than quietly move a
/// dashboard's series.
const DIRTY_REASONS: [&str; 5] = [
    "leaked_multi",
    "unexpected_reply",
    "probe_timeout",
    "probe_error",
    "topology_reply",
];

/// The `reason` a leaked `MULTI` must be reported under: `DISCARD` answered
/// `+OK`, which is only reachable when a transaction was already open.
const REASON_LEAKED_MULTI: &str = "leaked_multi";

/// The dirty-checkout event, verbatim message from
/// `ValkeyClient::acquire_txn_inner`.
///
/// Asserted on despite the test-writing guide's "don't assert exact log
/// messages" default, for the same reason `b_ji8_tests.rs` asserts on the
/// UNWATCH funnel's retry line: the `pinned` field on this event is the **only**
/// observable that distinguishes `acquire_txn_for` (slot-pinned probe) from the
/// `acquire_txn` shim (fred's default routing). Both tick the same counter cell
/// and both return the same guard, so without this field the two scenarios
/// below would be indistinguishable and one of them would be dead weight.
///
/// **One message, two levels** (b.7td round 3). An admitted recycle logs it at
/// `warn!`; a verdict whose repair the streak gate suppressed logs the identical
/// field set at `debug!`, because the regime that gate exists for turns every
/// checkout in the pool dirty and a per-checkout WARN there floods the log
/// pipeline. Suppression stays observable — the counter ticks either way and the
/// `debug!` still carries `suppressed=true` — which is why
/// [`seam_driven::checkout_capturing_logs`] captures at DEBUG.
const DIRTY_CHECKOUT_EVENT: &str = "txn pool checkout found the slot unfit for a transaction";

/// The recycle's success `info!`, verbatim prefix from
/// `ValkeyClient::repair_held_txn_slot`. Distinguishes "the slot was rebuilt" from
/// the two failure arms, which log at `error!` and hand the borrower an
/// unrepaired connection.
const RECYCLE_SUCCESS_INFO: &str = "txn slot recycled";

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures + helpers
// ═════════════════════════════════════════════════════════════════════════════

/// `MakeWriter` that captures every byte written to it into a shared buffer.
///
/// Mirrors the per-module helper in `b_ji8_tests.rs` (itself mirroring
/// `src/grpc/auth_interceptor.rs::tests::CapturingWriter`) — the established
/// project pattern is to duplicate these few lines rather than hoist them into
/// `tests/common`.
#[derive(Clone)]
struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

impl CapturingWriter {
    fn new() -> (Self, Arc<Mutex<Vec<u8>>>) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        (Self(buf.clone()), buf)
    }
}

impl<'a> MakeWriter<'a> for CapturingWriter {
    type Writer = CapturingWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl std::io::Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A `txn_pool` of exactly one connection, with the observability hooks
/// installed.
///
/// `pool_size = 1` is load-bearing in every test here: it is what guarantees
/// that the slot released after the `MULTI` injection is the same fred `Client`
/// the next `acquire_txn*` hands back, so the probe under test is provably
/// looking at the connection the test dirtied.
///
/// The hooks matter for a second reason beyond the counters:
/// `acquire_txn_inner` ticks the dirty counter through `metrics_snapshot()`,
/// which is `None` until they are installed — a test that skipped this call
/// would exercise the probe and observe nothing.
async fn fixture() -> (TestValkey, Arc<Metrics>) {
    let tv = TestValkey::setup_with_pool_size(1).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    (tv, metrics)
}

/// Read `live_edit_valkey_txn_dirty_checkouts_total{reason}`. Pre-registered in
/// `Metrics::new`, so present at zero from boot rather than absent.
fn read_dirty(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_txn_dirty_checkouts_total",
        &[("reason", reason)],
    )
    .unwrap_or(0.0) as u64
}

/// Every `reason` cell at once, so a test can assert on the whole counter
/// rather than the one cell it expects — a verdict misfiling itself under a
/// sibling reason is exactly the regression a single-cell assertion misses.
fn read_all_dirty(metrics: &Metrics) -> Vec<(&'static str, u64)> {
    DIRTY_REASONS
        .iter()
        .map(|reason| (*reason, read_dirty(metrics, reason)))
        .collect()
}

/// Read `live_edit_valkey_reconnects_total{pool="txn"}`.
fn read_txn_reconnects(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_reconnects_total",
        &[("pool", "txn")],
    )
    .unwrap_or(0.0) as u64
}

/// Sum every `live_edit_valkey_errors_total{pool="txn", kind=…}` series.
///
/// Summed rather than read per-cell because the hook labels each event with
/// fred's own `ErrorKind`, and which kind a killed connection surfaces as is
/// fred's business, not this test's contract. The contract is that the hook is
/// still listening at all.
fn read_txn_errors(metrics: &Metrics) -> u64 {
    render(metrics)
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter(|line| line.starts_with("live_edit_valkey_errors_total{"))
        .filter(|line| line.contains(r#"pool="txn""#))
        .filter_map(|line| line.rsplit_once(char::is_whitespace))
        .filter_map(|(_, value)| value.trim().parse::<f64>().ok())
        .map(|value| value as u64)
        .sum()
}

/// The server's `CLIENT LIST` entry for exactly one connection, read from a
/// **second** client so the observation never runs on the connection it
/// describes.
///
/// This is the **positive** observable that the checkout probe actually ran. A
/// clean verdict is otherwise invisible by construction — it logs nothing and
/// ticks nothing — so without a server-side witness the happy-path test would
/// be asserting only that four counters stayed at zero, which a probe that was
/// never issued satisfies just as well.
///
/// Per-connection rather than the server-global `INFO commandstats`: the suite
/// shares one Valkey, so a global counter can only say "*a* probe ran
/// somewhere during this window" and a sibling test's checkout satisfies it.
/// `CLIENT LIST ID <id>` is scoped to the one txn slot this test dirtied, so
/// its `cmd` and `tot-cmds` fields are attributable to this test's own
/// checkout and nothing else.
async fn client_list_entry(observer: &ValkeyClient, connection_id: i64) -> String {
    observer
        .pool
        .custom::<String, String>(
            CustomCommand::new_static("CLIENT LIST", None::<u16>, false),
            vec!["ID".to_string(), connection_id.to_string()],
        )
        .await
        .expect("CLIENT LIST ID must succeed")
}

/// One space-separated `name=value` field of a [`client_list_entry`].
///
/// Returns `None` when the field is absent, which a caller must treat as a
/// fixture problem rather than a b.7td regression: `cmd` and `tot-cmds` are
/// both reported by every server this suite runs against (Valkey 7.2+).
fn client_list_field<'a>(entry: &'a str, name: &str) -> Option<&'a str> {
    let prefix = format!("{name}=");
    entry
        .split_whitespace()
        .find_map(|field| field.strip_prefix(&prefix))
}

/// The `tot-cmds` counter of a [`client_list_entry`] — commands this ONE
/// connection has executed since it was opened.
fn total_commands(entry: &str) -> u64 {
    client_list_field(entry, "tot-cmds")
        .unwrap_or_else(|| {
            panic!(
                "this server's CLIENT LIST reports no `tot-cmds` field, so the positive half of \
                 the clean-checkout test cannot be observed. That is a fixture problem, not a \
                 b.7td regression. Entry: {entry:?}"
            )
        })
        .parse()
        .expect("tot-cmds is an integer")
}

/// Open a `MULTI` on `client` and leave it open — the residue a borrower that
/// dies mid-transaction leaves behind on a pooled connection.
///
/// Issued as a raw custom command rather than through fred's
/// `TransactionInterface`, deliberately: fred ships `MULTI`…`EXEC` as ONE
/// router command, so its transaction API cannot leave a `MULTI` dangling. That
/// is also why this is the honest reproduction — the leak in the field did not
/// come from fred's own transaction machinery either, but from a connection
/// whose command and reply streams had drifted apart.
async fn inject_open_multi(client: &fred::clients::Client) {
    let frame = client
        .custom_raw(
            CustomCommand::new_static("MULTI", None::<u16>, false),
            Vec::<FredValue>::new(),
        )
        .await
        .expect("raw MULTI must reach the server");
    assert!(
        matches!(&frame, Resp3Frame::SimpleString { data, .. } if data.as_ref().eq_ignore_ascii_case(b"OK")),
        "the injected MULTI must be acknowledged, or the slot this test releases is not \
         actually dirty and every assertion downstream is vacuous; got {frame:?}"
    );
}

/// The Valkey-assigned id of the connection `client` is talking over.
///
/// Changes on every reconnect, which makes it the behavioural witness that a
/// dirty slot was genuinely rebuilt rather than cleaned in place.
async fn client_id(client: &fred::clients::Client) -> i64 {
    client
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT ID", None::<u16>, false),
            vec![],
        )
        .await
        .expect("CLIENT ID must succeed on a txn-pool connection")
}

/// Poll `predicate` until it holds or `budget` elapses.
async fn poll_until(budget: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let started = std::time::Instant::now();
    loop {
        if predicate() {
            return true;
        }
        if started.elapsed() >= budget {
            return false;
        }
        tokio::time::sleep(COUNTER_PROBE_INTERVAL).await;
    }
}

/// Lines carrying `message`, from a captured `fmt`-layer buffer.
fn lines_with<'a>(logged: &'a str, message: &str) -> Vec<&'a str> {
    logged
        .lines()
        .filter(|line| line.contains(message))
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 — the probe runs on every checkout, and a healthy slot is Clean
// ═════════════════════════════════════════════════════════════════════════════

/// An ordinary checkout must issue both probe commands and conclude Clean.
///
/// Two halves, and both are needed. The per-connection `CLIENT LIST` witness is
/// the only *positive* evidence the probe happened at all — the Clean verdict is
/// silent by design — and the counter half is what says the probe did not cry
/// wolf on a perfectly good connection. A false-positive here is not cosmetic:
/// it would recycle a healthy connection on every single `txn_pool`
/// acquisition, turning the hot `UpdateLayer` path into a reconnect per retry
/// attempt.
///
/// **Attribution is per-connection, and has to be.** The suite shares one
/// Valkey, so a server-global `INFO commandstats` delta can only say "*a* probe
/// ran somewhere in this window" — a sibling test's checkout satisfies it just
/// as well, and the assertion then holds even if this client never probed. The
/// witness here is `CLIENT LIST ID <slot>` read from a **second** client:
/// `tot-cmds` and `cmd` belong to the one txn connection this fixture owns
/// (`pool_size = 1`), so nothing another test does can move them.
///
/// The `ECHO` before the measured checkout is what makes `cmd=unwatch`
/// non-vacuous: the *setup* checkout's own probe also ends on an `UNWATCH`, so
/// without displacing it the final assertion would be satisfied by a checkout
/// that probed nothing.
#[tokio::test]
async fn clean_txn_checkout_probes_the_slot_and_reports_no_dirty_verdict() {
    let (tv, metrics) = fixture().await;
    let observer = TestValkey::setup_with_pool_size(1).await;
    let key = tv.key_schema().obj_header("7td-clean");

    // Learn the single txn slot's server-side connection id, then leave a
    // command on it that is NOT the probe's.
    let slot_connection_id = {
        let guard = tv.client().acquire_txn_for(&key).await;
        let id = client_id(&guard).await;
        let echoed: String = guard
            .custom(
                CustomCommand::new_static("ECHO", None::<u16>, false),
                vec!["7td-witness".to_string()],
            )
            .await
            .expect("ECHO must succeed on the txn-pool connection");
        assert_eq!(echoed, "7td-witness");
        id
    };

    let before = client_list_entry(observer.client(), slot_connection_id).await;
    assert_eq!(
        client_list_field(&before, "cmd"),
        Some("echo"),
        "precondition: the witness ECHO must be the last command on the slot, so a `cmd=unwatch` \
         below is produced by the checkout under test rather than left over from the setup \
         checkout's own probe. Entry: {before:?}"
    );
    let commands_before = total_commands(&before);

    {
        let _guard = tv.client().acquire_txn_for(&key).await;
    }

    let after = client_list_entry(observer.client(), slot_connection_id).await;
    assert_eq!(
        client_list_field(&after, "cmd"),
        Some("unwatch"),
        "a txn checkout must end on the UNWATCH probe: a leaked WATCH with no MULTI behind it \
         produces no distinguishable reply, so it cannot be detected — only cleared, and cleared \
         AFTER the DISCARD has proven no MULTI is open (run the other way round, UNWATCH is \
         answered `+QUEUED` and clears nothing). Entry: {after:?}"
    );
    assert!(
        total_commands(&after) >= commands_before + 2,
        "a txn checkout must issue BOTH probe commands on this connection — DISCARD, whose error \
         reply (`-ERR … without MULTI`) IS the clean verdict, then UNWATCH. A checkout that \
         issues neither has no verdict at all and inherits whatever the previous borrower left \
         behind. This connection's own tot-cmds moved {commands_before} → {commands_after}, \
         i.e. by {delta} rather than by at least 2",
        commands_after = total_commands(&after),
        delta = total_commands(&after).saturating_sub(commands_before)
    );

    assert_eq!(
        read_all_dirty(&metrics),
        DIRTY_REASONS.map(|reason| (reason, 0)).to_vec(),
        "a healthy slot must produce the Clean verdict on every cell. A tick here means the \
         probe cried wolf, which recycles a good connection on EVERY txn acquisition — a \
         reconnect per UpdateLayer retry attempt on the hot path"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T2 — a leaked MULTI is detected, counted, repaired, and the borrower commits
// ═════════════════════════════════════════════════════════════════════════════

/// The full repair over [`ValkeyClient::acquire_txn_for`] — the pinned form
/// every migrated call site now uses.
#[tokio::test]
async fn pinned_txn_checkout_detects_a_leaked_multi_recycles_and_lets_the_borrower_commit() {
    leaked_multi_scenario(Pinning::Pinned).await;
}

/// The same repair over the unpinned [`ValkeyClient::acquire_txn`] shim.
///
/// The shim is not dead code even though b.4t3 left it with **no production
/// call site**: several tests — [`Pinning::Shim`] below, `watch_leak_test.rs`,
/// `b_ji8_tests.rs`, `b_k1r_tests.rs`, `observability_metrics_test.rs` — take a
/// raw `txn_pool` slot with no transaction of their own and therefore no key to
/// name, and a future caller that genuinely cannot know a key before checkout
/// would land here too. So the probe has to work on this form as well. It is a
/// separate test rather than an extra assertion because the two forms take
/// different code paths through `probe_command` (`ClusterHash::Custom(slot)` vs
/// `ClusterHash::FirstKey`), and a regression that made the pinned form
/// mandatory would leave every shim checkout unprobed while the pinned test
/// stayed green.
#[tokio::test]
async fn shim_txn_checkout_detects_a_leaked_multi_recycles_and_lets_the_borrower_commit() {
    leaked_multi_scenario(Pinning::Shim).await;
}

/// Which `acquire_txn*` form [`leaked_multi_scenario`] drives.
#[derive(Clone, Copy)]
enum Pinning {
    /// `acquire_txn_for(hash_key)` — probe pinned to the borrower's slot.
    Pinned,
    /// `acquire_txn()` — fred's default routing for the probe commands.
    Shim,
}

impl Pinning {
    /// The value the WARN's `pinned` field must carry, as the `fmt` layer
    /// renders a `bool` (unquoted).
    fn expected_field(self) -> &'static str {
        match self {
            Pinning::Pinned => "pinned=true",
            Pinning::Shim => "pinned=false",
        }
    }
}

/// Manufacture a dirty slot, check it out, and assert the whole repair.
///
/// Arrange: open a `MULTI` on the single pooled connection and release it —
/// exactly the state a borrower that dies mid-transaction leaves behind, and a
/// state fred can neither see nor clean (`drop_broken_connections` and
/// `force_reconnection()` act only on transport errors; this socket is
/// perfectly healthy).
///
/// Act: check the slot out again.
///
/// Assert, in the order the repair happens:
///
/// 1. the verdict is filed under `reason="leaked_multi"` exactly once, and no
///    sibling cell moves;
/// 2. the WARN names the right probe form (`pinned=…`), which is the only
///    observable separating the two `acquire_txn*` entry points;
/// 3. the recycle reports success rather than one of its two `error!` arms;
/// 4. — the one that matters to a caller — the borrower's own
///    `WATCH`/`MULTI`/`EXEC` **commits** on the returned guard. Without the
///    repair it cannot: every command on a connection with an open `MULTI` is
///    answered `+QUEUED`, so the borrower's `EXEC` collects the debris of both
///    transactions.
///
/// The whole body runs under one thread-local capturing subscriber on the
/// default current-thread runtime, so the events emitted inline by
/// `acquire_txn_inner` land in one buffer. Capture stops before the assertions
/// so a panic's own output is not swallowed.
async fn leaked_multi_scenario(pinning: Pinning) {
    let (tv, metrics) = fixture().await;
    let key = tv.key_schema().obj_header("7td-dirty");

    // Arrange — dirty the one pooled connection and hand it back, recording
    // the server-side identity of the socket that gets dirtied.
    let dirtied_connection_id = {
        let guard = tv.client().acquire_txn_for(&key).await;
        let id = client_id(&guard).await;
        inject_open_multi(&guard).await;
        id
    };
    assert_eq!(
        read_dirty(&metrics, REASON_LEAKED_MULTI),
        0,
        "precondition: the checkout that INJECTED the MULTI probed a clean slot, so nothing \
         may have been counted yet — otherwise the assertion below cannot attribute its tick \
         to the leak"
    );

    let (writer, captured) = CapturingWriter::new();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        // The dirty verdict is `warn!` and the recycle success is `info!`.
        .with_max_level(tracing::Level::INFO)
        // ANSI escapes would otherwise make the field matches depend on
        // whether colour was auto-enabled.
        .with_ansi(false)
        .finish();
    let capture = tracing::subscriber::set_default(subscriber);

    // Act — the next borrower checks the slot out.
    let guard = match pinning {
        Pinning::Pinned => tv.client().acquire_txn_for(&key).await,
        Pinning::Shim => tv.client().acquire_txn().await,
    };

    let logged = {
        let buffer = captured.lock().expect("capture mutex is never poisoned");
        String::from_utf8_lossy(&buffer).into_owned()
    };
    drop(capture);

    // 1 — the verdict, and only that verdict.
    assert_eq!(
        read_dirty(&metrics, REASON_LEAKED_MULTI),
        1,
        "an acknowledged DISCARD means a MULTI was already open on this pooled connection, and \
         must tick live_edit_valkey_txn_dirty_checkouts_total{{reason=\"leaked_multi\"}} exactly \
         once — this counter is the only first-order evidence of the leak an operator gets. \
         Pre-b.7td the sole symptom was fred's `Failed to refresh credentials` warning, which \
         named the credential refresh rather than the leak and only appeared on the passwordless \
         auth modes. Captured output: {logged:?}"
    );
    for (reason, count) in read_all_dirty(&metrics) {
        if reason == REASON_LEAKED_MULTI {
            continue;
        }
        assert_eq!(
            count, 0,
            "reason=\"{reason}\" must stay at zero: a leaked MULTI is a distinct fault from a \
             wedged or frame-offset connection, and misfiling it would send an operator after \
             the wrong failure"
        );
    }

    // 2 — which probe form ran.
    let warns = lines_with(&logged, DIRTY_CHECKOUT_EVENT);
    assert_eq!(
        warns.len(),
        1,
        "exactly one dirty-checkout WARN must be emitted for one dirty checkout. Captured \
         output: {logged:?}"
    );
    assert!(
        warns[0].contains(pinning.expected_field()),
        "the WARN must carry `{field}`: it is the only observable that separates the \
         slot-pinned `acquire_txn_for` probe from the `acquire_txn` shim, and on a cluster \
         that difference decides whether the verdict describes the borrower's own primary or \
         an arbitrary one. Event: {event:?}",
        field = pinning.expected_field(),
        event = warns[0]
    );

    // 3 — the socket was genuinely rebuilt, not merely cleaned in place.
    //
    // The server-side connection id is the behavioural witness, and it is the
    // load-bearing one: the probe's own `DISCARD` already closed the leaked
    // `MULTI` (that reply IS the verdict), so a build that skipped the recycle
    // entirely would still hand back a connection the borrower can transact on.
    // What it would NOT do is resynchronize a reply stream that had drifted out
    // of step with its command stream — the state a `+OK` from `DISCARD` says
    // this connection was in, and the one fred provably cannot repair. Only a
    // fresh socket does that, so "the id changed" is the assertion that
    // separates the fix from a probe that merely reports.
    let rebuilt_connection_id = client_id(&guard).await;
    assert_ne!(
        rebuilt_connection_id, dirtied_connection_id,
        "the dirty slot must be rebuilt (quit + reconnect), not just DISCARDed: the borrower is \
         about to run a transaction on it, and a connection that was found carrying an abandoned \
         MULTI may also have replies queued that nobody read. fred cannot resynchronize that — \
         `skip_results()` is a no-op on the transaction write path, `discard()` reads exactly one \
         frame, and `force_reconnection()` only acts on connections that raised a transport \
         error. Server-side connection id was still {dirtied_connection_id}"
    );

    // 3b — the repair ran to completion rather than taking an `error!` arm.
    assert_eq!(
        lines_with(&logged, RECYCLE_SUCCESS_INFO).len(),
        1,
        "the dirty slot must be rebuilt and its metric hooks re-armed. The two failure arms log \
         at `error!` with a `consequence` field and hand the borrower an unrepaired connection — \
         deliberate (a failed repair must not become a failed acquisition), but not what a \
         healthy local Valkey should ever produce. Captured output: {logged:?}"
    );

    // 4 — the borrower can actually use the slot.
    assert_borrower_can_commit(&guard, &key).await;
}

/// The assertion that matters to a caller: the borrower's own
/// `WATCH`/`MULTI`/`EXEC` **commits** on the guard the checkout handed back.
///
/// Not sensitive to the recycle on its own — the probe's `DISCARD` already
/// closed any leaked `MULTI`, so a build that only reported would also get here.
/// What it pins is that the repair left the connection *usable*: a `quit()`
/// whose reconnect failed, or a probe that returned mid-frame, fails here.
///
/// Without the repair a borrower cannot commit at all: every command on a
/// connection with an open `MULTI` is answered `+QUEUED`, so the borrower's
/// `EXEC` collects the debris of both transactions.
async fn assert_borrower_can_commit(guard: &fred::clients::Client, key: &str) {
    guard
        .watch::<&str>(key)
        .await
        .expect("WATCH must succeed on the recycled connection");
    let trx = guard.multi();
    let _: () = trx
        .set(key, "committed-after-recycle", None, None, false)
        .await
        .expect("MULTI queue of SET must succeed on the recycled connection");
    let reply: FredValue = trx
        .exec(true)
        .await
        .expect("EXEC must not error at the fred layer after a recycle");
    assert!(
        !reply.is_null(),
        "the borrower's own transaction must COMMIT on the repaired slot. A null EXEC means the \
         connection was handed over still carrying the previous borrower's MULTI: every command \
         the borrower sends is answered `+QUEUED`, their EXEC collects both transactions' debris, \
         and the failure surfaces on a request that did nothing wrong. Got: {reply:?}"
    );

    let committed: Option<String> = guard.get(key).await.expect("GET must succeed");
    assert_eq!(
        committed.as_deref(),
        Some("committed-after-recycle"),
        "the committed value must be readable — a non-null EXEC reply alone would also be \
         satisfied by a transaction that committed the wrong queue"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T3 — a recycled slot keeps its metric hooks
// ═════════════════════════════════════════════════════════════════════════════

/// After a recycle, the slot's reconnect / error hooks must still be listening.
///
/// fred's `quit()` calls `Notifications::close_public_receivers`, which **swaps
/// in fresh broadcast senders**; every existing `error_rx()` / `reconnect_rx()`
/// receiver sees `RecvError::Closed`, and both hook tasks in
/// `spawn_error_metric` / `spawn_reconnect_metric` `break` on `Closed`. So a
/// recycle that did not re-arm them would leave that slot permanently blind:
/// `valkey_reconnects_total{pool="txn"}` and `valkey_errors_total{pool="txn"}`
/// would never move again on that connection, and the b.cbe steady-state
/// credential-rejection hint would never fire — silently, on the exact
/// connection the service just told itself was suspect.
///
/// The seam is `CLIENT KILL ID` from a second client, scoped to the recycled
/// connection's own id so no sibling test's pool is disturbed. It is applied
/// *after* the recycle, which is what makes the observation attributable: the
/// recycle's own `init()` broadcasts a reconnect before the hooks are re-armed,
/// so it cannot be the tick this test sees.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recycled_txn_slot_keeps_reporting_reconnects_and_errors() {
    let (tv, metrics) = fixture().await;
    let killer = TestValkey::setup_with_pool_size(1).await;
    let key = tv.key_schema().obj_header("7td-rearm");

    // Arrange — force one recycle by dirtying the single pooled slot.
    {
        let guard = tv.client().acquire_txn_for(&key).await;
        inject_open_multi(&guard).await;
    }
    let victim_id: i64 = {
        let guard = tv.client().acquire_txn_for(&key).await;
        guard
            .custom::<i64, String>(
                CustomCommand::new_static("CLIENT ID", None::<u16>, false),
                vec![],
            )
            .await
            .expect("CLIENT ID must succeed on the recycled txn connection")
    };
    assert_eq!(
        read_dirty(&metrics, REASON_LEAKED_MULTI),
        1,
        "precondition: the checkout above must have found the leaked MULTI and recycled the \
         slot, or there is no re-armed hook to test"
    );

    // Baseline AFTER the recycle: the recycle's own `init()` broadcasts a
    // reconnect while no receiver is attached, so it can neither be counted nor
    // credited to the re-arm.
    let reconnects_before = read_txn_reconnects(&metrics);
    let errors_before = read_txn_errors(&metrics);

    // Act — kill exactly the recycled connection from a second client. The
    // guard is released first so fred's reconnect path has the slot to itself.
    let killed = killer
        .client()
        .pool
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT KILL", None::<u16>, false),
            vec!["ID".to_string(), victim_id.to_string()],
        )
        .await
        .expect("CLIENT KILL ID must succeed");
    assert_eq!(
        killed, 1,
        "CLIENT KILL ID must close exactly the recycled txn-pool victim connection"
    );

    // Assert — ONE deadline over both counters (see the module header's budget
    // note). fred raises the reader error and the reconnect from the same
    // disconnect, so waiting on the pair costs no more than waiting on one.
    let observed = poll_until(RECONNECT_OBSERVATION_BUDGET, || {
        read_txn_reconnects(&metrics) > reconnects_before
            && read_txn_errors(&metrics) > errors_before
    })
    .await;
    assert!(
        observed,
        "a killed connection on a RECYCLED txn slot must still move both \
         valkey_reconnects_total{{pool=\"txn\"}} ({reconnects_before} → {reconnects_now}) and \
         valkey_errors_total{{pool=\"txn\"}} ({errors_before} → {errors_now}). fred's `quit()` \
         swaps its broadcast senders, so a recycle that does not re-spawn the hooks blinds this \
         slot forever — the pod keeps serving on a connection whose health signal is gone, and \
         every `rate(valkey_reconnects_total{{pool=\"txn\"}}[5m])` alert reads a flat line",
        reconnects_now = read_txn_reconnects(&metrics),
        errors_now = read_txn_errors(&metrics),
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T4 — the source-shape fence: no new unpinned checkout
// ═════════════════════════════════════════════════════════════════════════════

/// Call-site form of the unpinned shim. Matches `.acquire_txn()` and nothing
/// else: the definition ([`ACQUIRE_TXN_DEFINITION_NEEDLE`]) has no leading dot,
/// and `acquire_txn_for(…)` / `acquire_txn_inner(…)` do not close their parens
/// in this position.
const BARE_ACQUIRE_TXN_NEEDLE: &str = ".acquire_txn()";

/// Definition site of the shim, in `src/storage/client.rs`.
///
/// Half of this fence's non-vacuity proof. The fence is now absolute — zero
/// permitted call sites under `src/` — so "found nothing" is the *passing*
/// state, and a rename of the shim would produce exactly that same nothing
/// while quietly retiring the guard. Pinning the definition string means a
/// rename fails here with a message telling the next reader to update the
/// needles rather than delete the fence.
const ACQUIRE_TXN_DEFINITION_NEEDLE: &str = "pub async fn acquire_txn(&self)";

/// Where [`ACQUIRE_TXN_DEFINITION_NEEDLE`] lives, relative to the crate root.
const ACQUIRE_TXN_DEFINITION_FILE: &str = "src/storage/client.rs";

/// Every `.rs` file under `dir`, as paths relative to `root`.
///
/// A directory walk rather than an enumerated list, for the same reason
/// `b_ji8_tests.rs::no_bare_unwatch_call_survives_outside_watch_guard` walks: a
/// NEW module that reaches for the unpinned shim is exactly the regression worth
/// catching, and a hand-maintained table would have to be remembered to see it.
///
/// `pub(super)`, with [`needle_hits`], [`is_needle_declaration`] and
/// [`code_lines_outside_strings`], because `stream_consumer_test.rs`'s
/// blocking-bridge fence (b.tmx) and `refusal_and_shutdown_guards_test.rs`'s
/// two b.kgb guards scan `src/` the same way; one scanner, tested once below,
/// rather than several that can drift.
pub(super) fn rust_sources_under(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {} must succeed: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry must be readable").path();
        if path.is_dir() {
            rust_sources_under(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(
                path.strip_prefix(root)
                    .expect("scanned paths are under the manifest dir")
                    .to_path_buf(),
            );
        }
    }
}

/// Strip Rust comments from one line, carrying `/* … */` state across lines.
///
/// A `trimmed.starts_with("//")` filter — what this fence used before b.4t3
/// round 2 — sees only *whole-line* line comments. Both other comment shapes
/// read as code to it: a trailing `let x = 1; // takes .acquire_txn() today`
/// and any needle inside a `/* … */` span. Both are false positives, and a
/// false positive on an absolute fence is a red pipeline nobody can fix except
/// by rewording a comment.
///
/// Deliberate limits, all in the conservative (fails-loud) direction:
///
/// - **string literals are kept**, so a needle inside a `"…"` still counts as a
///   hit. That is wrong in the strictest sense — a string is not a call site —
///   but the failure mode is a visible red test naming the line, whereas
///   stripping strings risks a mis-tracked quote silently swallowing real code.
///   The one string shape that actually occurs (the needle *constants* of the
///   two fences) is excluded by [`is_needle_declaration`] instead.
/// - **nested** `/* /* … */ */` is not modelled (Rust allows it, this returns to
///   code at the first `*/`) and `r#"…"#` raw strings are tracked only as far as
///   their first inner `"`. Neither shape occurs in the scanned trees, and both
///   err toward *more* text surviving into the match.
fn strip_comments(line: &str, in_block: &mut bool) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut code = String::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if *in_block {
            if c == '*' && chars.get(i + 1) == Some(&'/') {
                *in_block = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if in_string {
            code.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            code.push(c);
            i += 1;
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            // Line comment (`//`, `///`, `//!`): the rest of the line is prose.
            break;
        }
        if c == '/' && chars.get(i + 1) == Some(&'*') {
            *in_block = true;
            i += 2;
            continue;
        }
        code.push(c);
        i += 1;
    }
    code
}

/// Every line of `source` whose **code** (comments removed by
/// [`strip_comments`]) contains `needle`, as `(1-based line number, trimmed
/// code)`.
pub(super) fn needle_hits(source: &str, needle: &str) -> Vec<(usize, String)> {
    let mut in_block = false;
    let mut hits = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        let code = strip_comments(line, &mut in_block);
        let trimmed = code.trim();
        if trimmed.contains(needle) {
            hits.push((idx + 1, trimmed.to_string()));
        }
    }
    hits
}

/// Is this line the *declaration* of a needle constant rather than a call site?
///
/// `const BARE_ACQUIRE_TXN_NEEDLE: &str = ".acquire_txn()";` is code, not a
/// comment, so [`needle_hits`] matches it — and it exists in **both** fences
/// (here and `tests/integration/checkpoint/acquire_txn_pinning_fence_test.rs`).
/// Counting those two lines as "the needle still selects real call sites" is
/// circular: the anchor would hold at two even with every genuine shim caller
/// deleted, which is precisely the state it exists to detect.
pub(super) fn is_needle_declaration(trimmed: &str) -> bool {
    let decl = trimmed
        .strip_prefix("pub(crate) ")
        .or_else(|| trimmed.strip_prefix("pub "))
        .unwrap_or(trimmed);
    (decl.starts_with("const ") || decl.starts_with("static ")) && decl.contains(": &str")
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// **No** production module may take a `txn_pool` slot through the **unpinned**
/// shim. Absolute over `src/`, with no allow-list.
///
/// A shape guard in the idiom of `b_ji8_tests.rs`'s UNWATCH fence, and it exists
/// for the same reason that one does: the damage is deferred and misattributed.
/// `acquire_txn()` still probes, so a new call site added tomorrow looks
/// perfectly healthy on a single-node fixture and in every test in this suite.
/// `DISCARD` and `UNWATCH` carry no key, though, so fred's `ClusterHash::FirstKey`
/// policy finds nothing to hash and the router falls through to an ARBITRARY
/// node. On a cluster that means the verdict describes a connection the
/// borrower's transaction will never run on: a genuinely dirty slot passes its
/// checkout, and the failure surfaces later, on some other request, as the
/// `CROSSSLOT` / `DEADLINE_EXCEEDED` cascade b.7td was filed for. Nothing in a
/// behavioural test suite points at the line responsible.
///
/// The fix at any such site is one word — `acquire_txn_for(&key)` with any key
/// the transaction touches — which is why this is worth policing at the source
/// level rather than documenting.
///
/// # Why there is no allow-list any more
///
/// b.7td left one entry, `src/checkpoint/session.rs`, on lane-ownership grounds
/// rather than because its slot was unknowable — and it was knowable: the
/// DR-restore applier's Step A runs entirely on `{hex(base_path)}` and Step B
/// entirely on `{session_id}`. b.4t3 migrated both to `acquire_txn_for`, so the
/// last production caller is gone and the honest shape of this fence is
/// absolute. The checkpoint lane keeps its own mirror of this guard —
/// `tests/integration/checkpoint/acquire_txn_pinning_fence_test.rs` — which
/// additionally pins that the two pins are *still there* rather than merely
/// that the shim is not.
#[test]
fn no_unpinned_acquire_txn_call_site_survives_under_src() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources_under(&root, &root.join("src"), &mut files);
    files.sort();

    // Non-vacuity, part 1: the walk must actually reach the files that matter —
    // the last lane to be migrated (b.4t3), the module that defines the shim,
    // and the hottest migrated call site. A `src/` layout change that silently
    // narrowed the walk would otherwise leave this guard green and blind.
    for expected in [
        "src/checkpoint/session.rs",
        ACQUIRE_TXN_DEFINITION_FILE,
        "src/storage/external_values.rs",
    ] {
        assert!(
            files.iter().any(|f| f == Path::new(expected)),
            "the source scan did not visit `{expected}`; it walked {} files. Either the file \
             moved (update this list) or the walk is not reaching where it should",
            files.len()
        );
    }

    let mut violations: Vec<String> = Vec::new();

    for file in &files {
        let source = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
        // Comments are part of the file and legitimately show the shim: the
        // `WatchGuard` module docs spell the borrow out as an example, and
        // `config.rs` names it in prose. `needle_hits` removes all three
        // comment shapes (whole-line, trailing, and `/* … */` spans).
        for (line_no, code) in needle_hits(&source, BARE_ACQUIRE_TXN_NEEDLE) {
            violations.push(format!("  {}:{line_no}\n    {code}", file.display()));
        }
    }

    // Non-vacuity: with no allow-list, "found nothing" IS the passing state, so
    // a rename of the shim would retire this guard silently. Two independent
    // anchors close that hole — the definition still spelled the way the needle
    // expects, and the needle still matching real call sites somewhere in the
    // test tree (the shim's remaining honest users).
    let definition_source = std::fs::read_to_string(root.join(ACQUIRE_TXN_DEFINITION_FILE))
        .unwrap_or_else(|e| panic!("reading {ACQUIRE_TXN_DEFINITION_FILE} must succeed: {e}"));
    assert!(
        definition_source.contains(ACQUIRE_TXN_DEFINITION_NEEDLE),
        "`{ACQUIRE_TXN_DEFINITION_NEEDLE}` is no longer in {ACQUIRE_TXN_DEFINITION_FILE}, so \
         `{BARE_ACQUIRE_TXN_NEEDLE}` can no longer match anything and this fence passes \
         vacuously. If the shim was renamed, update BARE_ACQUIRE_TXN_NEEDLE and \
         ACQUIRE_TXN_DEFINITION_NEEDLE rather than deleting this guard. If it was DELETED \
         outright, the fence is genuinely obsolete and should be removed in the same commit — \
         but say so deliberately, do not let it rot green."
    );

    // Anchor 2: the needle still selects REAL call sites in the test tree — the
    // shim's remaining honest users (`b_ji8_tests.rs`, `watch_leak_test.rs`,
    // `b_k1r_tests.rs`, `observability_metrics_test.rs`, and this file's own
    // `Pinning::Shim` arm). Lines that merely DECLARE the needle are excluded
    // via `is_needle_declaration`: `const BARE_ACQUIRE_TXN_NEEDLE: &str =
    // ".acquire_txn()";` is spelled in this file and in the checkpoint lane's
    // mirror fence, so counting those would hold the anchor at two however many
    // genuine callers were deleted — self-satisfying, and blind to exactly the
    // "the needle stopped matching anything real" state it is here to detect.
    let mut test_sites = Vec::new();
    rust_sources_under(&root, &root.join("tests"), &mut test_sites);
    test_sites.sort();
    let shim_users: Vec<String> = test_sites
        .iter()
        .filter(|file| {
            let source = std::fs::read_to_string(root.join(file))
                .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
            needle_hits(&source, BARE_ACQUIRE_TXN_NEEDLE)
                .iter()
                .any(|(_, code)| !is_needle_declaration(code))
        })
        .map(|file| file.display().to_string())
        .collect();
    assert!(
        !shim_users.is_empty(),
        "`{BARE_ACQUIRE_TXN_NEEDLE}` matches no CALL SITE anywhere under tests/ either (the \
         needle-constant declarations in this file and in the checkpoint lane's fence do not \
         count), which means the needle itself has stopped selecting anything and the `src/` \
         scan above proves nothing. Same remedy as the definition anchor: fix the needle, or \
         retire this fence deliberately."
    );

    // A direct pin on the exclusion, using the one file in the tree that holds
    // the needle ONLY as a declaration. The mirror fence must be walked (or the
    // check is vacuous) and must not be counted as a shim user (or the anchor
    // is self-satisfying again).
    const MIRROR_FENCE: &str = "tests/integration/checkpoint/acquire_txn_pinning_fence_test.rs";
    assert!(
        test_sites.iter().any(|f| f == Path::new(MIRROR_FENCE)),
        "the tests/ walk did not visit `{MIRROR_FENCE}`, so the declaration-exclusion check \
         below proves nothing. Either the checkpoint lane's mirror fence moved (update this \
         constant) or the walk is not reaching where it should."
    );
    assert!(
        !shim_users.iter().any(|f| f == MIRROR_FENCE),
        "`{MIRROR_FENCE}` spells `{BARE_ACQUIRE_TXN_NEEDLE}` only as a `const … : &str = …` \
         declaration, never as a call site, so it must not be counted among the shim's users. \
         Its appearance here means `is_needle_declaration` has stopped excluding declarations — \
         and with this file's own declaration counted too, the anchor above would hold at two \
         however many genuine callers were deleted. Shim users found: {shim_users:?}"
    );

    assert!(
        violations.is_empty(),
        "unpinned `{BARE_ACQUIRE_TXN_NEEDLE}` call site(s) under src/:\n\n{}\n\nUse \
         `acquire_txn_for(&key)` with any key the transaction touches. The shim still probes, \
         but `DISCARD` / `UNWATCH` carry no key, so on a cluster fred routes them to an \
         arbitrary primary and the verdict describes a connection the borrower will never use — \
         a dirty slot passes its checkout and fails some later, unrelated request (b.7td; the \
         same routing defect b.ji8 fixed for teardown). There is no allow-list: b.4t3 migrated \
         the last production caller (`src/checkpoint/session.rs`), and every slot a `txn_pool` \
         transaction runs on has been knowable at every call site so far.",
        violations.join("\n")
    );
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// The fence's own scanner, on a fixture with every comment shape in it.
///
/// The fence above is a *filter* over the tree, and a filter that quietly
/// stopped filtering — or started over-filtering — changes what the fence means
/// without changing whether it is green. Both directions are covered here on a
/// fixture rather than on the tree, because the tree contains only some of the
/// shapes at any one time.
#[test]
fn needle_scanner_ignores_every_comment_shape_and_needle_declarations() {
    // Line numbers are 1-based and the fixture starts on the line AFTER the
    // opening delimiter, so line 1 is `fn a() {`.
    let fixture = r#"fn a() {
    let (g, t) = client.acquire_txn().await;
    let (g, t) = client.acquire_txn().await; // trailing prose about acquire_txn()
    // a whole-line comment mentioning .acquire_txn()
    /// a doc comment mentioning .acquire_txn()
    //! an inner doc comment mentioning .acquire_txn()
    /* a one-line block comment mentioning .acquire_txn() */
    let x = 1; /* trailing block comment mentioning .acquire_txn() */
    /* a block comment span
       whose second line mentions .acquire_txn()
       and which closes here */
    let y = 2;
}
const BARE_ACQUIRE_TXN_NEEDLE: &str = ".acquire_txn()";
"#;

    let hits = needle_hits(fixture, BARE_ACQUIRE_TXN_NEEDLE);
    let lines: Vec<usize> = hits.iter().map(|(line, _)| *line).collect();
    assert_eq!(
        lines,
        vec![2, 3, 14],
        "the scanner must see exactly the two real call sites (lines 2 and 3, the second of \
         which carries a trailing comment) plus the needle DECLARATION on line 14 — and nothing \
         from lines 4-11, which are line comments, doc comments, a one-line block comment, a \
         trailing block comment and a multi-line block-comment span. Got {hits:?}"
    );

    // Line 3's trailing `//` prose is stripped, so the recorded text is the code
    // only — that text is what a failure message shows the next reader.
    assert!(
        !hits[1].1.contains("trailing prose"),
        "the recorded call-site text must be code with its trailing comment removed; got {:?}",
        hits[1].1
    );

    // And the declaration filter separates line 14 from the two real sites, which
    // is what keeps the tests/-tree non-vacuity anchor from proving itself.
    assert!(
        is_needle_declaration(&hits[2].1),
        "`{}` must be recognised as a needle declaration",
        hits[2].1
    );
    for (line, code) in &hits[..2] {
        assert!(
            !is_needle_declaration(code),
            "line {line} (`{code}`) is a real call site, not a declaration"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// T4b — the second source-shape fence: fred's own pool stays inside the checkout
// ═════════════════════════════════════════════════════════════════════════════

/// The fred type whose `acquire()` is the b.f4f defect. Permitted in **code**
/// only inside [`CHECKOUT_MODULE`]; everywhere else it may appear in prose.
const EXCLUSIVE_POOL_TYPE_NEEDLE: &str = "ExclusivePool";

/// The non-blocking slot lock the checkout's scan is built on.
const SLOT_TRY_LOCK_NEEDLE: &str = "try_lock_owned";

/// fred's *blocking* slot lock — the `lock_owned().await` fallback
/// `ExclusivePool::acquire` parks on. `src/storage/checkout.rs`'s module docs
/// claim it is "absent from the call path"; this needle is what makes that a
/// checked claim rather than a comment. Matched standalone so
/// [`SLOT_TRY_LOCK_NEEDLE`] does not satisfy it.
///
/// **Deliberately un-narrowed, and therefore broader than its name.** It matches
/// `lock_owned` on ANY `tokio::sync::Mutex` anywhere under `src/` or `tests/`,
/// not only on a pool slot — the receiver is a local whose type no text scan can
/// see, so narrowing it would mean guessing at identifiers and would let the one
/// call that matters through under a different binding name. The trade is
/// deliberate: the only such call today is the targeted checkout's permit-first
/// slot lock, allow-listed by name ([`TARGETED_CHECKOUT_SIGNATURE`]), so
/// the fence costs nothing, and a
/// future unrelated `lock_owned` gets a red test rather than a silent pass. The
/// failure message says so, so whoever hits it can tell "the b.f4f defect is
/// back" from "this fence is now too broad for the tree" and resolve it
/// deliberately instead of deleting a guard that looked wrong.
const SLOT_BLOCKING_LOCK_NEEDLE: &str = "lock_owned";

/// The one module allowed to name fred's `ExclusivePool`.
const CHECKOUT_MODULE: &str = "src/storage/checkout.rs";

/// The signature of the one method allowed to await a slot's mutex:
/// `ExclusiveCheckout::checkout_slot`, the targeted checkout that takes one
/// SPECIFIC slot (by fred client id) — for `checkout_slot_for_router_restart`,
/// to restart its ended router (`storage::router_supervisor`), and for
/// `checkout_slot_for_replacement`, to swap a fresh client into it
/// (`storage::member_replacement`). It is allow-listed by name, and only for
/// one `lock_owned` call taken after its permit: a caller that holds a permit
/// and no slot while it waits counts against the scanners' free slots exactly
/// as one more holder would, which is the checkout module's soundness argument
/// for it.
const TARGETED_CHECKOUT_SIGNATURE: &str = "async fn checkout_slot(";

/// Exactly the `self.pool.<method>` calls [`CHECKOUT_MODULE`] is allowed to
/// make, as a sorted set of method names.
///
/// `size` is metadata (the permit count). `clients` is the mutex slice, and its
/// four uses are the four operations the module's soundness argument allows —
/// `ExclusiveCheckout::new`'s non-blocking `try_lock` of every slot to read its
/// client id (while it wraps a pool nothing else can reach yet),
/// `clients_for_startup_hooks` (startup only, hands back `Client` clones), the
/// `try_lock_owned` scan in `try_take_slot`, and the one slot
/// [`TARGETED_CHECKOUT_SIGNATURE`] awaits under a permit. `new` calls
/// `pool.clients()` on its argument before storing it, not `self.pool.clients()`,
/// so `pool_operations` does not see that use; the other three are
/// `self.pool.clients()` calls. `quit` is shutdown, and runs after the permit
/// semaphore is closed. Anything else is a new route into fred's pool and needs
/// its own soundness argument, which is what a failure here is asking for.
const ALLOWED_POOL_OPERATIONS: [&str; 3] = ["clients", "quit", "size"];

/// The 1-based, inclusive line span of the method whose signature line starts
/// with `signature` in `source`: from that line to the next line that is
/// exactly `    }` (the closing brace of an `impl` method).
fn method_line_span(source: &str, signature: &str) -> (usize, usize) {
    let lines: Vec<&str> = source.lines().collect();
    let start = lines
        .iter()
        .position(|line| line.trim_start().starts_with(signature))
        .unwrap_or_else(|| panic!("`{signature}` not found in {CHECKOUT_MODULE}"));
    let end = lines[start..]
        .iter()
        .position(|line| *line == "    }")
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("no closing `    }}` after `{signature}` in {CHECKOUT_MODULE}"));
    (start + 1, end + 1)
}

/// One line's code with comments **and string-literal contents** removed, with
/// every state carried across line boundaries.
enum ScanState {
    Code,
    Block,
    Str { escaped: bool },
    Raw { hashes: usize },
}

/// Strip comments *and* string-literal contents, returning `(1-based line
/// number, code)` per line.
///
/// [`strip_comments`] deliberately keeps string literals: its fence has exactly
/// one string-shaped false positive and excludes it by name
/// ([`is_needle_declaration`]). This fence's needle is a **type name**, and type
/// names turn up in prose inside multi-line `assert!` messages all over the tree
/// — four unrelated files today. Keeping strings would make the fence a
/// permanent red on comment wording, which is the failure mode
/// [`strip_comments`]' own doc warns about from the other direction.
///
/// Raw strings (`r"…"`, `r#"…"#`) are modelled, because this file's own scanner
/// fixture is one and a mis-tracked raw string would swallow real code. So are
/// **char literals** (`'"'`, `b'"'`, `'\''`) and the lifetimes they are easy to
/// confuse with — see the `ScanState::Code` branch below for why an unmodelled
/// `'"'` silently inverts the scanner's verdict for the rest of the file.
///
/// `pub(super)` for `refusal_and_shutdown_guards_test.rs`, whose guards build
/// their production-code view (`#[cfg(test)]` items removed) on top of it.
pub(super) fn code_lines_outside_strings(source: &str) -> Vec<(usize, String)> {
    let mut state = ScanState::Code;
    let mut out = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let mut code = String::new();
        let mut i = 0;
        while i < chars.len() {
            match state {
                ScanState::Block => {
                    if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                        state = ScanState::Code;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                ScanState::Str { escaped } => {
                    if escaped {
                        state = ScanState::Str { escaped: false };
                    } else if chars[i] == '\\' {
                        state = ScanState::Str { escaped: true };
                    } else if chars[i] == '"' {
                        state = ScanState::Code;
                        code.push('"');
                    }
                    i += 1;
                }
                ScanState::Raw { hashes } => {
                    let closes =
                        chars[i] == '"' && (0..hashes).all(|k| chars.get(i + 1 + k) == Some(&'#'));
                    if closes {
                        state = ScanState::Code;
                        code.push('"');
                        i += 1 + hashes;
                    } else {
                        i += 1;
                    }
                }
                ScanState::Code => {
                    let c = chars[i];
                    if c == '/' && chars.get(i + 1) == Some(&'/') {
                        break;
                    }
                    if c == '/' && chars.get(i + 1) == Some(&'*') {
                        state = ScanState::Block;
                        i += 2;
                        continue;
                    }
                    let prev_is_ident =
                        i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_');
                    if c == 'r' && !prev_is_ident {
                        let mut hashes = 0;
                        while chars.get(i + 1 + hashes) == Some(&'#') {
                            hashes += 1;
                        }
                        if chars.get(i + 1 + hashes) == Some(&'"') {
                            code.push('"');
                            state = ScanState::Raw { hashes };
                            i += 2 + hashes;
                            continue;
                        }
                    }
                    // Char literals, BEFORE the `"` branch below. `'"'` is one
                    // character of code, but without this the inner `"` opens a
                    // `ScanState::Str` that never closes on its line — and the
                    // state carries across lines, so from there every string is
                    // read as code and every stretch of code as a string, for the
                    // rest of the file. Char literals are scattered across both
                    // trees — `src/grpc/metrics_layer.rs` and
                    // `src/observability/metrics.rs` are two that carry them, and
                    // an exact roster is not worth keeping here because it goes
                    // stale on any commit that adds one. What matters is that the
                    // shape is common enough to hit, and that when the fence hit
                    // it the false negatives and false positives cancelled out —
                    // the quote parity happened to restore — so the fence passed
                    // while being wrong in both directions.
                    //
                    // A `'` is only a literal when a closing `'` follows the one
                    // character (or escape) after it; otherwise it opens a
                    // LIFETIME (`&'a str`, `'static`), which is ordinary code and
                    // must not swallow anything. Getting that backwards is the
                    // same class of bug in the other direction.
                    if c == '\'' {
                        if chars.get(i + 1) == Some(&'\\') {
                            // Escaped: '\n' '\'' '\\' '\0' '\x41' '\u{1F600}'.
                            //
                            // Scanned to the closing `'` rather than by escape
                            // shape. The per-form arithmetic this replaces got
                            // `'\xNN'` wrong — it skipped one character past the
                            // backslash, landing on the `4` of `\x41` instead of
                            // the closing quote — and fell through to the
                            // lifetime arm. Harmless for `\x` (two hex digits
                            // cannot be a `"`), but the same off-by-one on any
                            // future escape form would not be, and enumerating
                            // forms means re-deriving Rust's escape grammar here.
                            // The `\\` at `i + 1` is consumed with its escapee
                            // before the search starts, which is what stops
                            // `'\''` terminating on its own escaped quote.
                            //
                            // Bounded: a char literal is short, and the bound is
                            // what keeps a malformed one (a stray `'` that is
                            // neither literal nor lifetime) from consuming the
                            // rest of the line.
                            const MAX_ESCAPE_LEN: usize = 12; // '\u{10FFFF}'
                            let mut j = i + 3;
                            let limit = (i + 2 + MAX_ESCAPE_LEN).min(chars.len());
                            while j < limit && chars[j] != '\'' {
                                j += 1;
                            }
                            if chars.get(j) == Some(&'\'') {
                                // Dropped rather than pushed: no needle can occur
                                // inside a one-character literal, and dropping
                                // keeps the classification unambiguous.
                                i = j + 1;
                                continue;
                            }
                        } else if chars.get(i + 2) == Some(&'\'') {
                            i += 3;
                            continue;
                        }
                        // A lifetime. Ordinary code.
                        code.push(c);
                        i += 1;
                        continue;
                    }
                    if c == '"' {
                        code.push('"');
                        state = ScanState::Str { escaped: false };
                        i += 1;
                        continue;
                    }
                    code.push(c);
                    i += 1;
                }
            }
        }
        out.push((idx + 1, code.trim().to_string()));
    }
    out
}

/// Hits of `needle` in code outside comments and string literals.
fn code_needle_hits(source: &str, needle: &str) -> Vec<(usize, String)> {
    code_lines_outside_strings(source)
        .into_iter()
        .filter(|(_, code)| code.contains(needle))
        .collect()
}

/// [`code_needle_hits`], minus hits where the needle is part of a longer
/// identifier — which is what tells `lock_owned` from `try_lock_owned`.
fn standalone_needle_hits(source: &str, needle: &str) -> Vec<(usize, String)> {
    code_needle_hits(source, needle)
        .into_iter()
        .filter(|(_, code)| {
            code.match_indices(needle).any(|(at, _)| {
                at == 0
                    || !code[..at]
                        .chars()
                        .next_back()
                        .is_some_and(|c| c.is_alphanumeric() || c == '_')
            })
        })
        .collect()
}

/// `source`'s code outside comments and string literals, as one line with
/// every run of whitespace collapsed to a single space — for scans whose
/// needle can span lines.
fn flat_code(source: &str) -> String {
    code_lines_outside_strings(source)
        .iter()
        .flat_map(|(_, code)| code.split_whitespace())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The code of the function whose signature starts with `signature` in
/// `flat` (a [`flat_code`] string): from the signature up to the next ` fn `.
/// Enough for a body with no nested `fn` item, which is all it is used on.
fn flat_function_body<'a>(flat: &'a str, signature: &str) -> &'a str {
    let start = flat
        .find(signature)
        .unwrap_or_else(|| panic!("`{signature}` not found in {CHECKOUT_MODULE}'s code"));
    let body = &flat[start + signature.len()..];
    let end = body.find(" fn ").unwrap_or(body.len());
    &body[..end]
}

/// Every `self.pool.<method>` name called in `source`, in source order.
fn pool_operations(source: &str) -> Vec<String> {
    // `self.pool` and its method are frequently on different lines
    // (`try_take_slot`'s builder chain), so the scan runs over the whole file
    // with whitespace collapsed rather than line by line.
    let flat = flat_code(source);
    let mut ops = Vec::new();
    let mut rest = flat.as_str();
    while let Some(at) = rest.find("self.pool") {
        let after = rest[at + "self.pool".len()..].trim_start();
        if let Some(after_dot) = after.strip_prefix('.') {
            let name: String = after_dot
                .trim_start()
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                ops.push(name);
            }
        }
        rest = &rest[at + "self.pool".len()..];
    }
    ops
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// **No module but `storage::checkout` may name fred's `ExclusivePool`**, and
/// inside it the pool is reached by exactly three operations (b.f4f).
///
/// The sibling fence above polices *which* txn checkout a call site takes. This
/// one polices something stronger and newer: that the service's own checkout is
/// the ONLY way any slot mutex is ever locked.
///
/// That property is what the whole `ExclusiveCheckout` rests on. Its soundness
/// argument is an arithmetic one — permits == slots, every holder holds at most
/// one slot, therefore a non-blocking scan by a permit holder always finds a
/// free slot. Both halves of the premise fail the moment some other module locks
/// a slot without taking a permit: a permit holder then scans an apparently full
/// pool, burns the whole `CHECKOUT_RESCAN_DEADLINE`, and trips
/// `report_escaped_slot` — a `debug_assert!` in test builds, and in production a
/// caller re-parked on the permit queue for no reason anything explains.
///
/// Encapsulation is the real guarantee — `ExclusiveCheckout::pool` is a private
/// field and nothing hands out a `&[Arc<Mutex<Client>>]`, so today the fence
/// cannot fire. It is a regression guard on an edit that *re-opens* that
/// encapsulation (a `pub fn pool(&self)`, a second `clients()`-style accessor, a
/// test reaching for the raw pool to saturate it — the shape eight tests used
/// before b.f4f migrated them), which is cheap to write and easy to do without
/// noticing what it costs.
///
/// Scope includes `tests/` deliberately: a test that holds a raw pool slot
/// alongside permit-based checkouts is exactly the over-admission this forbids,
/// and it would show up as a mysteriously flaky sibling test rather than as a
/// failure at the line responsible.
///
/// Inside the module, the pool's slots are locked without a permit in exactly
/// one place: fred's own `ExclusivePool::quit`, which locks every slot to send
/// its `QUIT`. `ExclusiveCheckout::quit` closes the permit semaphore first, so
/// no new permit is granted while the `QUIT` holds the slots, and a scanner
/// already holding one reads `is_quitting()` and keeps waiting instead of
/// escalating to `report_escaped_slot`. One method awaits a slot's mutex, and
/// only under a permit: `checkout_slot`, the targeted checkout that takes one
/// specific slot so the router supervisor can restart its ended router or the
/// member replacer can swap its client; it is allow-listed by name, and the
/// check below pins that its permit comes first.
/// The text check below covers only the
/// ordering (`quit` closes the semaphore before it quits fred's pool); the
/// behaviour is the checkout module's own tests'
/// (`quit_closes_the_permit_semaphore_before_it_takes_the_slots` and
/// `a_quitting_pool_is_never_accused_of_an_escaped_slot_under_the_seam`).
#[test]
fn no_exclusive_pool_mutex_is_locked_outside_the_checkout_module() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources_under(&root, &root.join("src"), &mut files);
    rust_sources_under(&root, &root.join("tests"), &mut files);
    files.sort();

    // Non-vacuity, part 1: the walk reaches the module under discussion and at
    // least one migrated test file.
    for expected in [
        CHECKOUT_MODULE,
        "src/storage/client.rs",
        "tests/integration/storage/stream_consumer_test.rs",
    ] {
        assert!(
            files.iter().any(|f| f == Path::new(expected)),
            "the source scan did not visit `{expected}`; it walked {} files. Either the file \
             moved (update this list) or the walk is not reaching where it should",
            files.len()
        );
    }

    let checkout_source = std::fs::read_to_string(root.join(CHECKOUT_MODULE))
        .unwrap_or_else(|e| panic!("reading {CHECKOUT_MODULE} must succeed: {e}"));

    // Non-vacuity, part 2: the needles still select real code inside the one
    // module that is allowed to use them. A rename of fred's type, or a rewrite
    // of the scan, would otherwise leave every assertion below trivially
    // satisfied.
    assert!(
        code_needle_hits(&checkout_source, EXCLUSIVE_POOL_TYPE_NEEDLE)
            .iter()
            .any(|(_, code)| !is_needle_declaration(code)),
        "`{EXCLUSIVE_POOL_TYPE_NEEDLE}` no longer appears in CODE in `{CHECKOUT_MODULE}`, so the \
         tree-wide ban below matches nothing and passes vacuously. If fred's type was renamed, \
         update EXCLUSIVE_POOL_TYPE_NEEDLE rather than deleting this guard."
    );

    let mut violations: Vec<String> = Vec::new();
    for file in &files {
        if file == Path::new(CHECKOUT_MODULE) {
            continue;
        }
        let source = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
        for needle in [EXCLUSIVE_POOL_TYPE_NEEDLE, SLOT_TRY_LOCK_NEEDLE] {
            for (line_no, code) in code_needle_hits(&source, needle) {
                if is_needle_declaration(&code) {
                    continue;
                }
                violations.push(format!("  {}:{line_no}\n    {code}", file.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "fred's `{EXCLUSIVE_POOL_TYPE_NEEDLE}` (or a raw `{SLOT_TRY_LOCK_NEEDLE}` on one of its \
         slots) is named in code outside `{CHECKOUT_MODULE}`:\n\n{}\n\nEvery exclusive-slot \
         checkout must go through `ExclusiveCheckout`: `acquire_txn_for(&key)` / `acquire_txn()`. \
         A raw hold breaks the permits == slots arithmetic the checkout's \
         scan depends on — the holder occupies a slot no permit accounts for, so a legitimately \
         admitted caller scans a full pool and trips `report_escaped_slot` (b.f4f).",
        violations.join("\n")
    );

    // The blocking fallback must be absent from the whole tree, checkout module
    // included: `ExclusiveCheckout` scans with `try_lock_owned` and parks on the
    // semaphore instead, which is the entire fix. A `lock_owned().await` on a
    // slot would reintroduce fred's "await ONE selected mutex" behaviour inside
    // our own code. The one exception is allow-listed by name: the targeted
    // checkout of one specific slot (`checkout_slot`, shared by the router
    // restart and a slot's client replacement), which awaits that slot's mutex
    // only AFTER taking a permit (checked below).
    let (targeted_first, targeted_last) =
        method_line_span(&checkout_source, TARGETED_CHECKOUT_SIGNATURE);
    let mut allowed_targeted_locks = 0;
    let mut blocking: Vec<String> = Vec::new();
    for file in &files {
        let source = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
        for (line_no, code) in standalone_needle_hits(&source, SLOT_BLOCKING_LOCK_NEEDLE) {
            if is_needle_declaration(&code) {
                continue;
            }
            if file == Path::new(CHECKOUT_MODULE)
                && (targeted_first..=targeted_last).contains(&line_no)
            {
                allowed_targeted_locks += 1;
                continue;
            }
            blocking.push(format!("  {}:{line_no}\n    {code}", file.display()));
        }
    }
    assert_eq!(
        allowed_targeted_locks, 1,
        "`{TARGETED_CHECKOUT_SIGNATURE}..)` in `{CHECKOUT_MODULE}` (lines \
         {targeted_first}-{targeted_last}) must lock its slot with exactly one \
         `{SLOT_BLOCKING_LOCK_NEEDLE}` call. Zero means the allow-list entry is stale (drop it); \
         more means a second blocking lock rides on the allow-list"
    );
    let checkout_flat = flat_code(&checkout_source);
    let targeted_body = flat_function_body(&checkout_flat, TARGETED_CHECKOUT_SIGNATURE);
    let permit_at = targeted_body.find("self.acquire_permit().await");
    let lock_at = targeted_body.find(".lock_owned().await");
    assert!(
        matches!((permit_at, lock_at), (Some(permit), Some(lock)) if permit < lock),
        "`checkout_slot` must take its permit (`self.acquire_permit().await`) \
         BEFORE it awaits the slot's mutex: a waiter holding a slot-less permit is what keeps the \
         scanners' arithmetic sound, while a slot locked without a permit is the escaped slot \
         `report_escaped_slot` exists for. Positions found: permit {permit_at:?}, lock \
         {lock_at:?} in:\n{targeted_body}"
    );
    assert!(
        blocking.is_empty(),
        "a blocking `{SLOT_BLOCKING_LOCK_NEEDLE}` call appeared:\n\n{}\n\nTwo ways to be \
         here, and they need different answers.\n\n(1) It is on an EXCLUSIVE-POOL SLOT — the \
         regression this fence is for. `ExclusiveCheckout` takes a FIFO semaphore permit and \
         then scans every slot with `{SLOT_TRY_LOCK_NEEDLE}`, which never blocks; awaiting one \
         selected mutex instead is precisely the fred 10.1.0 behaviour b.f4f exists to replace, \
         where a parked waiter stays parked while every other slot goes idle. Fix the call \
         site.\n\n(2) It is on some OTHER `tokio::sync::Mutex` and has nothing to do with the \
         pool. This needle cannot tell the two apart — the receiver's type is invisible to a \
         text scan — and it is deliberately broad because zero such calls existed when it was \
         written. That is no longer true, so decide: narrow the needle, or allow-list this \
         site by name. Do not simply delete the fence.",
        blocking.join("\n")
    );

    // …and inside the checkout module, the pool is reached by exactly the three
    // operations the soundness argument accounts for.
    let mut operations = pool_operations(&checkout_source);
    operations.sort();
    operations.dedup();
    assert_eq!(
        operations, ALLOWED_POOL_OPERATIONS,
        "`{CHECKOUT_MODULE}` now calls a different set of methods on fred's pool. The allowed \
         set is `size` (the permit count), `clients` (whose only four uses are \
         `ExclusiveCheckout::new`'s non-blocking `try_lock` of every slot to read its client \
         id, on the pool argument before it is stored and so not counted here, \
         `clients_for_startup_hooks`, which hands back `Client` clones at startup, the \
         `{SLOT_TRY_LOCK_NEEDLE}` scan in `try_take_slot`, and the one slot \
         `checkout_slot` awaits under a permit) and `quit`. \
         Everything else fred \
         offers on an `{EXCLUSIVE_POOL_TYPE_NEEDLE}` locks slots without taking a permit \
         (`init`, `connect_pool`, `wait_for_connect`, `force_reconnection`, \
         `update_perf_config`), so adding one needs its own answer to \"what does a concurrent \
         scanner see while this runs?\" — `quit`'s answer is the semaphore it closes first."
    );

    // Non-vacuity, part 3: the scan above actually found the operations rather
    // than an empty list (a `self.pool` rename would otherwise pass as "no
    // disallowed operations").
    assert_eq!(
        pool_operations(&checkout_source).len(),
        6,
        "expected exactly six `self.pool.<method>` calls in `{CHECKOUT_MODULE}` (two `size`, \
         three `clients` — `clients_for_startup_hooks`, `try_take_slot`'s scan and \
         `checkout_slot`'s slot — and one `quit`); the scanner found {:?}. \
         If the field was renamed, update `pool_operations`; \
         if a call was added or removed, update this count deliberately.",
        pool_operations(&checkout_source)
    );

    // `quit` must close the semaphore before fred's pool QUIT locks every
    // slot, so no new permit is granted while the QUIT holds slots.
    let flat = flat_code(&checkout_source);
    let quit_body = flat_function_body(&flat, "pub(crate) async fn quit(&self)");
    let order: Vec<Option<usize>> = ["self.permits.close()", "self.pool.quit()"]
        .iter()
        .map(|step| quit_body.find(step))
        .collect();
    assert!(
        order.iter().all(Option::is_some) && order.windows(2).all(|w| w[0] < w[1]),
        "`ExclusiveCheckout::quit` in `{CHECKOUT_MODULE}` must close the permit semaphore \
         before it quits fred's pool. Positions found: {order:?} in:\n{quit_body}"
    );

    // Non-vacuity, part 4: the `try_lock_owned` scan is still there. Without it
    // the tree-wide ban above is satisfied by a checkout that does not scan at
    // all.
    assert_eq!(
        standalone_needle_hits(&checkout_source, SLOT_TRY_LOCK_NEEDLE).len(),
        1,
        "`{CHECKOUT_MODULE}` must contain exactly one `{SLOT_TRY_LOCK_NEEDLE}` call — the \
         non-blocking scan in `try_take_slot`. Zero means the ban asserted above is vacuous."
    );
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// The second fence's scanner, on a fixture carrying every shape it must see
/// through — and every shape it must not trip on.
///
/// [`code_lines_outside_strings`] is what makes the `ExclusivePool` fence usable
/// at all: the needle is a TYPE NAME, and type names appear in prose inside
/// multi-line `assert!` messages in four unrelated files today. A scanner that
/// stopped stripping strings would turn the fence into a permanent red on
/// comment wording; one that started stripping too much would make it silently
/// blind. Both directions are covered here, on a fixture rather than on the tree,
/// because the tree contains only some of the shapes at any one time.
#[test]
fn code_scanner_strips_comments_and_string_literals_but_keeps_code() {
    let fixture = concat!(
        "fn a() {\n",
        "    let p: ExclusivePool = make();\n",
        "    // a whole-line comment mentioning ExclusivePool\n",
        "    /// a doc comment mentioning ExclusivePool\n",
        "    /* a one-line block comment mentioning ExclusivePool */\n",
        "    let x = 1; /* trailing block comment mentioning ExclusivePool */\n",
        "    /* a block-comment span\n",
        "       whose second line mentions ExclusivePool\n",
        "       and which closes here */\n",
        "    assert!(ok, \"a message mentioning ExclusivePool \\\n",
        "         and continuing onto a second line that also says ExclusivePool\");\n",
        "    let q = r#\"a raw string mentioning ExclusivePool\"#;\n",
        "    let quote = '\"';\n",
        "    let esc = '\\'';\n",
        "    let hexq = '\\x22';\n",
        "    let byteq = b'\"';\n",
        "    let uni = '\\u{1F600}';\n",
        "    let r: &'static str = \"a string mentioning ExclusivePool\";\n",
        "    let guard = slot.try_lock_owned();\n",
        "    let other = mutex.lock_owned().await;\n",
        "}\n",
    );

    let type_hits: Vec<usize> = code_needle_hits(fixture, EXCLUSIVE_POOL_TYPE_NEEDLE)
        .into_iter()
        .map(|(line, _)| line)
        .collect();
    assert_eq!(
        type_hits,
        vec![2],
        "the scanner must see exactly the one real code occurrence (line 2) — not the line \
         comment, the doc comment, either block-comment shape, the multi-line assert message, \
         the raw string, or the ordinary string on line 18. Got lines {type_hits:?}"
    );

    // The standalone filter separates the blocking lock from the non-blocking
    // one; a plain substring match cannot, and the fence's tree-wide ban on
    // `lock_owned` would then fire on every `try_lock_owned` in the checkout.
    let blocking: Vec<usize> = standalone_needle_hits(fixture, SLOT_BLOCKING_LOCK_NEEDLE)
        .into_iter()
        .map(|(line, _)| line)
        .collect();
    assert_eq!(
        blocking,
        vec![20],
        "`{SLOT_BLOCKING_LOCK_NEEDLE}` must match only the bare call on line 20, never the \
         `{SLOT_TRY_LOCK_NEEDLE}` on line 19. Got lines {blocking:?}"
    );
    let try_lock: Vec<usize> = standalone_needle_hits(fixture, SLOT_TRY_LOCK_NEEDLE)
        .into_iter()
        .map(|(line, _)| line)
        .collect();
    assert_eq!(
        try_lock,
        vec![19],
        "`{SLOT_TRY_LOCK_NEEDLE}` must match line 19 and nothing else; got {try_lock:?}"
    );

    // ── The char-literal trap, asserted DIRECTLY ────────────────────────────
    //
    // The three needle assertions above cannot see a char-literal bug at all,
    // and an earlier version of this test wrongly claimed they could. The reason
    // is parity: the trap region holds an EVEN number of raw `"` (line 13's
    // `'"'` and line 16's `b'"'`), so a scanner that mishandles them opens string
    // state on 13 and closes it again on 16 — restoring `Code` before line 18.
    // Lines 14, 15 and 17 hold no raw `"` and cannot flip the state whatever is
    // done to them. Every needle assertion above reads lines 18-20, which are
    // therefore reached in `Code` state either way, and all three keep passing
    // with the entire `if c == '\''` branch deleted.
    //
    // So the trap has to be read where it happens. These are the exact code
    // strings lines 13-18 must reduce to: each char literal wholly consumed (the
    // scanner drops their contents, so only the surrounding code survives), the
    // `b` prefix of the byte literal retained because it IS code, and line 18's
    // string emptied to its delimiters while its `&'static` LIFETIME is kept —
    // the one `'` on these six lines that must NOT start a literal.
    let trap: Vec<String> = code_lines_outside_strings(fixture)
        .into_iter()
        .filter(|(line, _)| (13..=18).contains(line))
        .map(|(_, code)| code)
        .collect();
    assert_eq!(
        trap,
        vec![
            "let quote = ;",
            "let esc = ;",
            "let hexq = ;",
            "let byteq = b;",
            "let uni = ;",
            "let r: &'static str = \"\";",
        ],
        "the char-literal trap region did not reduce to its code. Each line isolates one form: \
         13 the plain `'\"'`, 14 the escaped `'\\''`, 15 the hex `'\\x22'` (a quote by another \
         spelling), 16 the BYTE literal `b'\"'`, 17 the long `'\\u{{1F600}}'` escape, 18 a \
         `&'static` lifetime that must survive as code next to a string that must not. A \
         mishandled literal shows up here as its `'` or its contents leaking into the code, or \
         as a neighbouring line emptied because string state was left open across it. It does \
         NOT show up in the needle assertions above — the quote parity across 13 and 16 \
         restores `Code` before they look. Got {trap:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T5 / T6 — every verdict the probe can reach, and the recycle gate
// ═════════════════════════════════════════════════════════════════════════════

/// Seam-driven coverage of the verdicts a healthy Valkey cannot produce
/// (b.7td round 2).
///
/// **Compiled only under `--features test-fault-injection`** (together with
/// `test-support`, which every integration binary needs). The
/// `ProbeFaultInjector` seam and `ValkeyClient::install_probe_fault_injector_for_test`
/// do not exist without it, so this module is `#[cfg]`-ed out rather than
/// failing to build — which means a missing feature makes these tests silently
/// *absent*, not red. Run them with:
///
/// ```text
/// cargo nextest run -p live-edit --test integration \
///     --features test-support,test-fault-injection storage::txn_checkout_test
/// ```
///
/// # Why a seam at all
///
/// Four of the five outcomes `probe_txn_slot` can reach are unreachable against
/// a real server: there is no way to make Valkey answer `DISCARD` with an
/// integer, no way to make an argument-free command hang, and no way to ask it
/// for a `Canceled`. Before the seam those arms were covered only by
/// `txn_checkout`'s in-crate unit tests, which call `classify()` directly — so
/// nothing pinned what the *checkout* does with a verdict once it has one.
///
/// # What these tests assert, and what they deliberately do not
///
/// Only downstream consequences: which metric cell moves, what the WARN says,
/// whether the connection was rebuilt, and whether the borrower can transact on
/// what it was handed. The error-kind → `reason` mapping itself is owned by
/// `txn_checkout::tests::probe_failures_are_sorted_by_what_they_say_about_the_connection`
/// and is not re-asserted here; duplicating it would make the same decision
/// fail in two places and drift in one.
#[cfg(feature = "test-fault-injection")]
mod seam_driven {
    use fred::error::ErrorKind;
    use live_edit::storage::txn_checkout::{ProbeFault, ProbeFaultInjector};

    use super::*;

    /// The canonical `-MOVED` payload the topology cases inject.
    ///
    /// Three space-separated tokens and canonically cased — deliberately the
    /// rendering fred's `is_redirection` *would* intercept on a clustered
    /// client — because the seam answers the probe directly and never goes near
    /// the router, so what this drives is the classifier arm in isolation.
    /// Which renderings actually escape fred's interception in production (a
    /// non-clustered client, a fourth token, a different casing) is a fred
    /// property, pinned as a prefix table on
    /// `storage::txn_checkout::tests::topology_prefixes_match_the_leading_token_only`
    /// and reasoned out with file:line provenance on `is_state_proof_reason`'s
    /// rustdoc — it is not something an integration test against a single-node
    /// Valkey could observe.
    const MOVED_PAYLOAD: &str = "MOVED 1234 10.0.0.1:6379";

    /// The `reason` a cluster-topology / node-state transition error frame in a
    /// probe slot must be reported under (b.289 sub-task 3) — an unknown-state
    /// verdict, so it is streak-gated and can never reach the repair-ineffective
    /// breaker.
    ///
    /// Lives inside this module rather than beside [`REASON_LEAKED_MULTI`]
    /// because only the seam can produce the verdict: a healthy Valkey never
    /// answers a `DISCARD` with a topology error, so no unfeatured test has any
    /// use for the name.
    const REASON_TOPOLOGY_REPLY: &str = "topology_reply";

    /// The re-arm interval the recycle gate uses, restated here.
    ///
    /// `TXN_RECYCLE_REARM_EVERY` is `pub(crate)` and an integration test is a
    /// separate crate, so this is a copy — held honest by
    /// [`assert_dirty_event`], which checks it against the `recycle_rearm_every`
    /// field the production WARN carries on every dirty checkout.
    const RECYCLE_REARM_EVERY: u32 = 8;

    /// Every `reason` cell, with `expected` in the named one and zero
    /// everywhere else — the shape [`read_all_dirty`] returns.
    ///
    /// Asserting on the whole counter rather than one cell is the point: a
    /// verdict that misfiles itself under a sibling `reason` sends an operator
    /// after the wrong failure, and a single-cell assertion cannot see it.
    fn only_cell(reason: &str, expected: u64) -> Vec<(&'static str, u64)> {
        DIRTY_REASONS
            .iter()
            .map(|cell| (*cell, if *cell == reason { expected } else { 0 }))
            .collect()
    }

    /// How many consecutive state-proof verdicts are repaired before the
    /// breaker withholds the repair, restated here.
    ///
    /// `TXN_STATE_PROOF_REPAIR_ATTEMPTS` is `pub(crate)` and an integration test
    /// is a separate crate, so this is a copy — held honest by
    /// [`assert_ineffective_escalation`], which checks it against the
    /// `repair_attempts` field the production ERROR carries.
    const STATE_PROOF_REPAIR_ATTEMPTS: u32 = 3;

    /// Every `reason` cell of `live_edit_valkey_txn_repair_ineffective_total`.
    ///
    /// A strict subset of [`DIRTY_REASONS`], and the subsetting is the contract:
    /// only a verdict the probe *proved* from a reply it actually received can
    /// reach this breaker. The unknown-state pair has a differently-shaped rate
    /// limit — a re-arming streak gate that never stops repairing — so a
    /// `probe_timeout` cell appearing on this counter would mean the two
    /// mechanisms had been crossed.
    const STATE_PROOF_REASONS: [&str; 2] = ["leaked_multi", "unexpected_reply"];

    /// The breaker's escalation event, verbatim prefix from
    /// `ValkeyClient::acquire_txn_inner`.
    ///
    /// Distinct from [`DIRTY_CHECKOUT_EVENT`] on purpose, and asserted on for
    /// the same reason that one is: this is the *only* log an operator gets
    /// telling them the service has stopped trying to repair a slot and is
    /// handing it over unrepaired. It has to be greppable and it has to be
    /// distinguishable from the ordinary dirty verdict, which by then is firing
    /// on every checkout in the pool.
    const REPAIR_INEFFECTIVE_ERROR: &str = "txn slot recycling is not clearing the dirty verdict";

    /// Read `live_edit_valkey_txn_repair_ineffective_total{reason}`.
    /// Pre-registered in `Metrics::new`, so present at zero from boot.
    fn read_ineffective(metrics: &Metrics, reason: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_valkey_txn_repair_ineffective_total",
            &[("reason", reason)],
        )
        .unwrap_or(0.0) as u64
    }

    /// Both `reason` cells of the breaker counter at once — same whole-counter
    /// discipline [`read_all_dirty`] applies to its sibling.
    fn read_all_ineffective(metrics: &Metrics) -> Vec<(&'static str, u64)> {
        STATE_PROOF_REASONS
            .iter()
            .map(|reason| (*reason, read_ineffective(metrics, reason)))
            .collect()
    }

    /// Both breaker cells, with `expected` in the named one and zero in the
    /// other — the shape [`read_all_ineffective`] returns.
    fn only_ineffective_cell(reason: &str, expected: u64) -> Vec<(&'static str, u64)> {
        STATE_PROOF_REASONS
            .iter()
            .map(|cell| (*cell, if *cell == reason { expected } else { 0 }))
            .collect()
    }

    /// Check the one breaker-escalation ERROR in `logged`.
    ///
    /// Also asserts the ordinary dirty-checkout event is **absent**: the
    /// escalation replaces it rather than accompanying it, so one verdict
    /// produces one log line. Two lines describing the same verdict at two
    /// levels would double every entry in the regime this breaker exists for —
    /// one where every checkout in the pool produces the verdict and
    /// `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (128) re-enters the checkout
    /// per retry attempt.
    fn assert_ineffective_escalation(logged: &str, reason: &str, streak: u32) {
        let errors = lines_with(logged, REPAIR_INEFFECTIVE_ERROR);
        assert_eq!(
            errors.len(),
            1,
            "streak {streak} must escalate exactly once: it is the verdict that trips the breaker \
             (or a re-arm of it), and this ERROR is the only place an operator learns the service \
             has stopped repairing the slot and is handing it out unrepaired. Captured output: \
             {logged:?}"
        );
        let error = errors[0];
        assert!(
            error.contains(&format!("reason=\"{reason}\"")),
            "the escalation must name the verdict it gave up on — it is what points an \
             investigation at the classifier arm responsible. Event: {error:?}"
        );
        assert_event_field(
            error,
            "state_proof_streak",
            streak,
            "it is the value the breaker decided on, and the only thing distinguishing the \
             verdict that TRIPPED it from a periodic re-arm of an incident already hours old",
        );
        assert_event_field(
            error,
            "repair_attempts",
            STATE_PROOF_REPAIR_ATTEMPTS,
            "a reader must be able to size the evidence behind the decision without the source — \
             and this is what keeps the module's local copy of the constant from drifting from \
             production's",
        );
        assert_event_field(
            error,
            "escalation_rearm_every",
            RECYCLE_REARM_EVERY,
            "an operator reading one ERROR every eighth checkout needs to know the silence \
             between them is deliberate rather than the incident ending",
        );
        // The escalation is now the SCOPE statement too (b.289 sub-task 4):
        // hygiene is off for this node, not for the pool.
        assert_unnameable_node(error, "repair-ineffective escalation");
        assert!(
            lines_with(logged, DIRTY_CHECKOUT_EVENT).is_empty(),
            "the escalation must REPLACE the ordinary dirty-checkout event, not accompany it: one \
             verdict, one log line. Captured output: {logged:?}"
        );
    }

    /// One dirty verdict, named by the cell it must land in and the fault that
    /// produces it.
    #[derive(Clone, Copy, Debug)]
    enum DirtyCase {
        /// `DISCARD` acknowledged: a `MULTI` was already open.
        LeakedMulti,
        /// `DISCARD` answered with a frame no verdict allows — the canonical
        /// "the reply stream is offset against the command stream" shape.
        UnexpectedDiscardFrame,
        /// `DISCARD` answered with a `without MULTI` error naming a DIFFERENT
        /// command. The b.7td round-2 classifier defect: on a bare-tail match
        /// this read as the clean verdict, which is precisely backwards — a
        /// reply to somebody else's `EXEC` arriving in the `DISCARD` slot IS
        /// the offset stream.
        UnexpectedExecWithoutMulti,
        /// The `UNWATCH` probe — reached only when `DISCARD` answered
        /// cleanly — answered `+QUEUED`, i.e. it was swallowed by a
        /// transaction that step 1 said was not there.
        UnexpectedUnwatchReply,
        /// A probe that never answered inside its deadline.
        ProbeTimeout,
        /// A probe that failed with an error kind that does not say the
        /// connection is gone.
        ProbeError,
        /// `DISCARD` answered with a `-MOVED` that fred's router did not
        /// intercept (b.289 sub-task 3).
        ///
        /// Reachable in production because fred's `is_redirection` demands a
        /// case-sensitive prefix AND exactly three space-separated tokens, and
        /// because the txn pool runs `max_redirections = 0`. Pre-b.289 this
        /// read as `unexpected_reply` — state-proof — so three of them during
        /// one resharding latched a breaker whose escalation ERROR accuses the
        /// classifier.
        TopologyMovedOnDiscard,
        /// The same verdict reached through the **UNWATCH** slot, which requires
        /// the `DISCARD` probe to take the real round trip and come back clean
        /// first. That is what proves the second classifier arm was wired too —
        /// a fix applied to only one of them is invisible from the DISCARD side.
        TopologyMovedOnUnwatch,
        /// A non-`MOVED` prefix, end to end. `LOADING` is never intercepted by
        /// fred on this build at all (the `custom-reconnect-errors` feature is
        /// off), so it is the arm with no upstream interception to hide behind.
        TopologyLoadingOnDiscard,
        /// A lower-cased rendering. Every one of fred's own interceptions is a
        /// case-**sensitive** `starts_with`, so this is precisely the shape a
        /// proxy can hand through, and it must still be recognised.
        TopologyLowerCaseClusterdownOnDiscard,
        /// The same payload in RESP3's **other** error framing. A `$`-framed
        /// `-MOVED` is the same reshard as a `-`-framed one, and classifying it
        /// as `unexpected_reply` would file a routine topology change as
        /// state-proof purely because of the encoding.
        TopologyBlobErrorOnDiscard,
    }

    impl DirtyCase {
        /// Queue the fault that drives this case onto the right probe command.
        fn arm(self, injector: &ProbeFaultInjector) {
            match self {
                DirtyCase::LeakedMulti => injector.push_discard(ProbeFault::ok()),
                DirtyCase::UnexpectedDiscardFrame => injector.push_discard(ProbeFault::number(7)),
                DirtyCase::UnexpectedExecWithoutMulti => {
                    injector.push_discard(ProbeFault::simple_error("ERR EXEC without MULTI"))
                }
                // Deliberately on the UNWATCH queue: reaching this arm at all
                // requires the DISCARD probe to have taken the REAL round-trip
                // and come back clean, which is the only way to prove the two
                // `unexpected_reply` arms are independently reachable.
                DirtyCase::UnexpectedUnwatchReply => {
                    injector.push_unwatch(ProbeFault::simple_string("QUEUED"))
                }
                DirtyCase::ProbeTimeout => injector.push_discard(ProbeFault::hang()),
                DirtyCase::ProbeError => injector.push_discard(ProbeFault::error(
                    ErrorKind::Auth,
                    "injected probe rejection",
                )),
                DirtyCase::TopologyMovedOnDiscard => {
                    injector.push_discard(ProbeFault::simple_error(MOVED_PAYLOAD))
                }
                // On the UNWATCH queue for the same reason
                // `UnexpectedUnwatchReply` is: reaching the second classifier
                // arm at all requires the DISCARD probe to have taken the REAL
                // round trip and come back clean.
                DirtyCase::TopologyMovedOnUnwatch => {
                    injector.push_unwatch(ProbeFault::simple_error(MOVED_PAYLOAD))
                }
                DirtyCase::TopologyLoadingOnDiscard => injector.push_discard(
                    ProbeFault::simple_error("LOADING Redis is loading the dataset in memory"),
                ),
                DirtyCase::TopologyLowerCaseClusterdownOnDiscard => injector
                    .push_discard(ProbeFault::simple_error("clusterdown The cluster is down")),
                // Built as a raw `ProbeFault::Frame` because the seam has no
                // `blob_error` constructor — deliberately not added, since the
                // enum is public and the frame is two lines here. Same payload
                // as `TopologyMovedOnDiscard`, so the pair is a differential on
                // the ENCODING and nothing else.
                DirtyCase::TopologyBlobErrorOnDiscard => {
                    injector.push_discard(ProbeFault::Frame(Resp3Frame::BlobError {
                        data: bytes::Bytes::from_static(MOVED_PAYLOAD.as_bytes()),
                        attributes: None,
                    }))
                }
            };
        }

        /// The `reason` cell this case must land in.
        fn reason(self) -> &'static str {
            match self {
                DirtyCase::LeakedMulti => "leaked_multi",
                DirtyCase::UnexpectedDiscardFrame
                | DirtyCase::UnexpectedExecWithoutMulti
                | DirtyCase::UnexpectedUnwatchReply => "unexpected_reply",
                DirtyCase::ProbeTimeout => "probe_timeout",
                DirtyCase::ProbeError => "probe_error",
                DirtyCase::TopologyMovedOnDiscard
                | DirtyCase::TopologyMovedOnUnwatch
                | DirtyCase::TopologyLoadingOnDiscard
                | DirtyCase::TopologyLowerCaseClusterdownOnDiscard
                | DirtyCase::TopologyBlobErrorOnDiscard => REASON_TOPOLOGY_REPLY,
            }
        }

        /// The streak pair the event must carry.
        ///
        /// Each verdict class advances exactly one of the two streaks and logs
        /// `0` for the other, so on a fresh client every case here is the first
        /// of *its* streak and reads `1` in its own field and `0` in the
        /// sibling. See [`Streaks`].
        fn expected_streaks(self) -> Streaks {
            match self {
                DirtyCase::ProbeTimeout
                | DirtyCase::ProbeError
                | DirtyCase::TopologyMovedOnDiscard
                | DirtyCase::TopologyMovedOnUnwatch
                | DirtyCase::TopologyLoadingOnDiscard
                | DirtyCase::TopologyLowerCaseClusterdownOnDiscard
                | DirtyCase::TopologyBlobErrorOnDiscard => Streaks::unknown_state(1),
                DirtyCase::LeakedMulti
                | DirtyCase::UnexpectedDiscardFrame
                | DirtyCase::UnexpectedExecWithoutMulti
                | DirtyCase::UnexpectedUnwatchReply => Streaks::state_proof(1),
            }
        }
    }

    /// The two streak fields every dirty-checkout event carries, as one value.
    ///
    /// Passed as a pair rather than as the one number the decision was made on,
    /// because the *other* field reading zero is half the contract (b.7td
    /// round 3). `acquire_txn_inner` keeps two streaks and consults exactly one
    /// of them per verdict; it logs both, so the event always shows the number
    /// the gate that ran actually decided on **next to** a zero standing for
    /// "the other gate had no say here". Were the fields to bleed — a
    /// state-proof verdict rendering the unknown-state streak's stored value,
    /// say — the event would read as though a suppression window it is not
    /// subject to had been consulted, and an operator sizing the incident from
    /// the log would size the wrong one.
    ///
    /// **The two streaks have different scopes** (b.289 sub-task 4). The
    /// unknown-state streak — `probe_timeout` / `probe_error` /
    /// `topology_reply` — is pool-wide, because the storm it bounds is
    /// pool-wide. The state-proof streak is keyed by cluster node, with one
    /// shared fallback bucket for verdicts whose node cannot be named. Every
    /// test in this file runs against a centralized endpoint, so every verdict
    /// lands in that shared bucket and `txn_state_proof_streak_for_test()` —
    /// which reports the MAX across buckets — reads exactly the single number it
    /// reported before b.289. The per-node behaviour that a single endpoint
    /// cannot express (node A latching while node B's `Clean`s do not re-arm it)
    /// lives in `b_289_tests.rs`, behind the cluster gate.
    ///
    /// Note the deliberate asymmetry with the *stored* streaks: only a `Clean`
    /// checkout resets either, so a state-proof verdict logging
    /// `unknown_state_streak=0` while `txn_unknown_state_streak_for_test()`
    /// still reads 2 is correct, not a bug. That distinction is what
    /// [`the_two_streak_gates_advance_and_log_independently`] pins.
    #[derive(Clone, Copy, Debug)]
    struct Streaks {
        unknown_state: u32,
        state_proof: u32,
    }

    impl Streaks {
        /// An unknown-state verdict (`probe_timeout` / `probe_error`) at this
        /// streak length; the state-proof gate had no say.
        fn unknown_state(streak: u32) -> Self {
            Self {
                unknown_state: streak,
                state_proof: 0,
            }
        }

        /// A state-proof verdict (`leaked_multi` / `unexpected_reply`) at this
        /// streak length; the unknown-state gate had no say.
        fn state_proof(streak: u32) -> Self {
            Self {
                unknown_state: 0,
                state_proof: streak,
            }
        }
    }

    /// Check the one dirty-checkout event in `logged` against the case.
    ///
    /// Asserted on despite the test-writing guide's "don't assert exact log
    /// messages" default, for the same reason the `pinned` field is above: the
    /// `reason` / `suppressed` / `unknown_state_streak` triple is the only
    /// place an operator can see *which* verdict fired and whether the repair
    /// was admitted, and two of the four cells are otherwise indistinguishable
    /// (both `unexpected_reply`).
    ///
    /// Level-agnostic on purpose (b.7td round 3): the event is `warn!` when the
    /// repair was admitted and `debug!` when the gate suppressed it, but the
    /// message and the field set are identical, and what this helper pins is
    /// that suppression stays *observable* either way. See
    /// [`DIRTY_CHECKOUT_EVENT`].
    fn assert_dirty_event(logged: &str, reason: &str, streaks: Streaks, suppressed: bool) {
        let warns = lines_with(logged, DIRTY_CHECKOUT_EVENT);
        assert_eq!(
            warns.len(),
            1,
            "exactly one dirty-checkout event must be emitted for one dirty checkout. Captured \
             output: {logged:?}"
        );
        let warn = warns[0];
        assert!(
            warn.contains(&format!("reason=\"{reason}\"")),
            "the event must file this verdict under reason=\"{reason}\" — the label is the series \
             an operator's dashboard and alert are built on, and it is what tells a leaked MULTI \
             apart from a wedged connection. Event: {warn:?}"
        );
        assert!(
            warn.contains(&format!("suppressed={suppressed}")),
            "the event must carry suppressed={suppressed}: suppression is never silent — the \
             counter ticks either way, so `suppressed` is the ONLY signal distinguishing \
             'the slot was repaired' from 'the repair was rate-limited'. Event: {warn:?}"
        );
        assert_event_field(
            warn,
            "unknown_state_streak",
            streaks.unknown_state,
            "it is the value the unknown-state recycle gate actually decided on — and `0` on a \
             verdict that gate never saw, even when a stale unknown-state streak is still stored",
        );
        assert_event_field(
            warn,
            "state_proof_streak",
            streaks.state_proof,
            "it is the value the repair-ineffective breaker actually decided on — and `0` on a \
             verdict that breaker never saw. Without this field a withheld repair is \
             indistinguishable from an admitted one that happened to change nothing",
        );
        assert_event_field(
            warn,
            "recycle_rearm_every",
            RECYCLE_REARM_EVERY,
            "a reader must be able to size the suppression window without the source — and this \
             is what keeps the module's local copy of the interval from drifting from \
             production's",
        );
        assert_unnameable_node(warn, "dirty-checkout event");
    }

    /// Every verdict event must carry the `node` field, and on this fixture it
    /// must be empty (b.289 sub-task 4).
    ///
    /// Two claims in one, and both are worth the exactness. The field's
    /// *presence* is the only place an operator learns which cluster node a
    /// withheld repair applies to — the breaker is per node now, but node
    /// identity is deliberately NOT a metric label (unbounded cardinality over a
    /// deployment's lifetime), so the log is the whole scoping story.
    ///
    /// Its *emptiness here* is the contract for a centralized endpoint, which is
    /// what the sandbox and CI's single-node `test:` job provide:
    /// `probe_node_bucket` finds no cached cluster state, returns the shared
    /// fallback bucket, and the deployment reads exactly as the pre-b.289
    /// pool-wide one did. A non-empty value on a single-node fixture would mean
    /// the resolver had invented an identity, and the shared bucket every
    /// `Clean` collects would stop being the one these verdicts land in. The
    /// populated case needs ≥2 primaries and is pinned in
    /// `b_289_tests.rs`'s cluster-gated test.
    fn assert_unnameable_node(event: &str, what: &str) {
        assert_eq!(
            event_field(event, "node"),
            Some("\"\""),
            "the {what} must carry an EMPTY `node` field on a centralized endpoint. The field \
             itself is required — the state-proof breaker is per node and its scope appears \
             nowhere else, since node identity is a log field and deliberately not a metric \
             label — and empty is the right value here: with no cached cluster state the verdict \
             belongs to the shared fallback bucket that ANY Clean re-arms, which is what makes a \
             single-endpoint deployment behave exactly as the pre-b.289 pool-wide contract did. \
             Event: {event:?}"
        );
    }

    /// One space-separated `name=value` field of a captured `fmt`-layer event.
    ///
    /// `detail=…` carries spaces and therefore splits into several tokens, which
    /// is harmless: none of them can look like `<name>=` for any field asserted
    /// on here.
    fn event_field<'a>(event: &'a str, name: &str) -> Option<&'a str> {
        let prefix = format!("{name}=");
        event
            .split_whitespace()
            .find_map(|field| field.strip_prefix(&prefix))
    }

    /// Assert one numeric field of an event by **exact token**, not substring.
    ///
    /// `contains("state_proof_streak=1")` is satisfied by an event carrying
    /// `state_proof_streak=13`, which is not hypothetical here: the breaker test
    /// walks one streak from 1 to 13 and every assertion in it is about *which*
    /// number the gate decided on, so a prefix match would let the two ends of
    /// that walk pass for each other.
    fn assert_event_field(event: &str, name: &str, expected: impl std::fmt::Display, why: &str) {
        let expected = expected.to_string();
        assert_eq!(
            event_field(event, name),
            Some(expected.as_str()),
            "the event must carry {name}={expected}: {why}. Event: {event:?}"
        );
    }

    /// Install a fresh injector on the fixture's client and hand both back.
    async fn seam_fixture() -> (TestValkey, Arc<Metrics>, Arc<ProbeFaultInjector>) {
        let (tv, metrics) = fixture().await;
        let injector = Arc::new(ProbeFaultInjector::new());
        tv.client()
            .install_probe_fault_injector_for_test(Arc::clone(&injector));
        (tv, metrics, injector)
    }

    /// One checkout, with everything it logs captured.
    ///
    /// The slot is released before returning, so the caller can check out again
    /// — `pool_size = 1` makes a second acquisition while the first guard is
    /// alive a deadlock, not a queue.
    ///
    /// **Captured at DEBUG, filtered to `live_edit`** (b.7td round 3). A dirty
    /// verdict whose repair the streak gate suppressed logs at `debug!` rather
    /// than `warn!` — the suppression regime is a store-wide latency event,
    /// where a per-checkout WARN is a log flood describing a decision not to act
    /// — so an INFO capture would see the admitted recycles and nothing else,
    /// and the gate tests below would be unable to tell "suppressed" from "never
    /// happened". The `live_edit` target filter keeps fred's own debug chatter
    /// out of the buffer that panic messages print.
    async fn checkout_capturing_logs(tv: &TestValkey, key: &str) -> (i64, String) {
        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_env_filter("live_edit=debug")
            .with_ansi(false)
            .finish();
        let capture = tracing::subscriber::set_default(subscriber);

        let guard = tv.client().acquire_txn_for(key).await;
        let connection_id = client_id(&guard).await;
        drop(guard);

        let logged = {
            let buffer = captured.lock().expect("capture mutex is never poisoned");
            String::from_utf8_lossy(&buffer).into_owned()
        };
        drop(capture);
        (connection_id, logged)
    }

    /// Drive one dirty cell end to end and assert only its downstream
    /// consequences.
    ///
    /// Arrange: a clean checkout, which both establishes the slot's current
    /// server-side connection id and pins the injector-absent path — the seam
    /// is inert with an empty queue, so this checkout takes the real
    /// round-trip.
    ///
    /// Act: queue one fault and check the slot out again.
    ///
    /// Assert, in the order the repair happens: the cell ticks exactly once and
    /// no sibling moves; the WARN files it under the right `reason` with the
    /// gate's own decision attached; the socket was rebuilt; the borrower can
    /// transact on what it was handed. Then one more checkout, with the queue
    /// drained, to pin that a consumed fault does not linger.
    ///
    /// The connection-id change is asserted for every case here, including the
    /// two unknown-state ones: those are the FIRST of their streak on a fresh
    /// client, and the first of a streak always repairs. The gate's suppressing
    /// behaviour is covered separately, below.
    async fn dirty_cell_scenario(case: DirtyCase) {
        let (tv, metrics, injector) = seam_fixture().await;
        let key = tv.key_schema().obj_header("7td-seam");

        let connection_before = {
            let guard = tv.client().acquire_txn_for(&key).await;
            client_id(&guard).await
        };
        assert_eq!(
            read_all_dirty(&metrics),
            only_cell("none", 0),
            "precondition ({case:?}): a checkout with an EMPTY fault queue must take the real \
             probe and find the slot clean. A tick here means the seam is not inert when unused, \
             which would make every assertion below unattributable"
        );

        case.arm(&injector);
        assert_eq!(
            injector.remaining(),
            1,
            "precondition ({case:?}): exactly one fault must be queued"
        );

        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .finish();
        let capture = tracing::subscriber::set_default(subscriber);

        let guard = tv.client().acquire_txn_for(&key).await;
        let connection_after = client_id(&guard).await;

        let logged = {
            let buffer = captured.lock().expect("capture mutex is never poisoned");
            String::from_utf8_lossy(&buffer).into_owned()
        };
        drop(capture);

        assert_eq!(
            injector.remaining(),
            0,
            "({case:?}) the queued fault must be consumed by the checkout. A fault still sitting \
             in the queue means the probe never ran the command it was queued for, and the \
             verdict below came from somewhere else"
        );

        // 1 — exactly one cell moved.
        assert_eq!(
            read_all_dirty(&metrics),
            only_cell(case.reason(), 1),
            "({case:?}) live_edit_valkey_txn_dirty_checkouts_total must tick reason=\"{reason}\" \
             exactly once and leave every sibling cell at zero. This counter is the only \
             first-order evidence of the fault an operator gets, and a verdict filed under the \
             wrong cell sends them after a different failure entirely. Captured output: {logged:?}",
            reason = case.reason()
        );

        // 1b — the FIRST verdict of any class must be repaired, so the
        // repair-ineffective breaker must not have fired. A breaker that
        // tripped here would withhold the repair on the one verdict that is
        // always real residue.
        assert_eq!(
            read_all_ineffective(&metrics),
            only_ineffective_cell("none", 0),
            "({case:?}) live_edit_valkey_txn_repair_ineffective_total must stay at zero on a \
             single verdict: the breaker's whole premise is that one rebuild clears real residue, \
             so it may only fire after several rebuilds in a row have demonstrably not. Captured \
             output: {logged:?}"
        );

        // 2 — the WARN, including the gate's own decision.
        assert_dirty_event(&logged, case.reason(), case.expected_streaks(), false);

        // 3 — the socket was genuinely rebuilt, not merely reported on.
        assert_eq!(
            lines_with(&logged, RECYCLE_SUCCESS_INFO).len(),
            1,
            "({case:?}) the dirty slot must be rebuilt and its metric hooks re-armed. The two \
             failure arms log at `error!` and hand the borrower an unrepaired connection — \
             deliberate, but not what a healthy local Valkey should ever produce. Captured \
             output: {logged:?}"
        );
        assert_ne!(
            connection_after, connection_before,
            "({case:?}) the slot must be rebuilt (quit + reconnect). Every verdict here says the \
             connection's state is either provably corrupt or unknown, and fred cannot \
             resynchronize a reply stream that has drifted out of step with its command stream: \
             `skip_results()` is a no-op on the transaction write path, `discard()` reads exactly \
             one frame, and `force_reconnection()` only acts on connections that raised a \
             TRANSPORT error. Only a fresh socket clears it. Server-side connection id was still \
             {connection_before}"
        );

        // 4 — the borrower can actually use what it was handed.
        assert_borrower_can_commit(&guard, &key).await;
        drop(guard);

        // 5 — a consumed fault does not linger: the next checkout probes for
        // real, finds the slot clean, and leaves it alone.
        let (connection_final, quiet) = checkout_capturing_logs(&tv, &key).await;
        assert_eq!(
            connection_final, connection_after,
            "({case:?}) with the queue drained the probe must fall through to the real \
             round-trip and find the repaired slot clean. A second rebuild here means the seam \
             leaked its fault into a later checkout, and every test that installs an injector \
             would be scripting the checkouts after the one it meant to"
        );
        assert_eq!(
            read_all_dirty(&metrics),
            only_cell(case.reason(), 1),
            "({case:?}) the clean follow-up checkout must tick nothing. Captured output: {quiet:?}"
        );
        assert_eq!(
            tv.client().txn_unknown_state_streak_for_test(),
            0,
            "({case:?}) a Clean verdict must reset the unknown-state streak, so the next \
             unknown-state verdict is the first of a FRESH streak and repairs immediately rather \
             than inheriting a suppression window from an incident that is over"
        );
        assert_eq!(
            tv.client().txn_state_proof_streak_for_test(),
            0,
            "({case:?}) a Clean verdict must reset the state-proof streak too: the slot has just \
             demonstrated that the classifier and this endpoint DO agree about a reply, which is \
             exactly the evidence the repair-ineffective breaker latches on the absence of. \
             Leaving it set would carry a stale breaker into the next incident and withhold the \
             very first repair of it. The accessor reports the MAX across every node bucket \
             (b.289 sub-task 4), so zero here means no bucket survived the Clean — which on a \
             centralized endpoint is the shared fallback bucket every Clean collects"
        );
        assert_eq!(
            tv.client().txn_state_proof_streak_nodes_for_test(),
            Vec::<String>::new(),
            "({case:?}) a Clean verdict must COLLECT the bucket, not merely zero it. The map \
             returning to empty is what keeps the routing-table clone off the hot Clean path in \
             steady state: `acquire_txn_inner` resolves the node on a Clean only while some \
             bucket is uncollected, so a bucket left behind at zero would make the highest \
             frequency acquirer in the process pay for a `cached_cluster_state()` clone forever"
        );
    }

    /// `DISCARD` acknowledged — a `MULTI` was open on the pooled connection.
    ///
    /// The same cell `pinned_txn_checkout_detects_a_leaked_multi_recycles_and_lets_the_borrower_commit`
    /// reaches with a REAL injected `MULTI`; that test remains the honest
    /// reproduction. This one adds the uniform table treatment (whole-counter
    /// assertion, the full WARN field set, the drained-queue fall-through) that
    /// the other five cases get, so a change to the dirty-verdict path is
    /// checked identically across all four cells.
    #[tokio::test]
    async fn a_canned_ok_from_discard_files_the_leaked_multi_cell() {
        dirty_cell_scenario(DirtyCase::LeakedMulti).await;
    }

    /// An integer where a status or an error belongs: the reply stream is
    /// offset against the command stream. Unreachable against a real server,
    /// which is exactly why it needs the seam — this is the shape the probe
    /// exists to catch and the one nothing else in the suite can produce.
    #[tokio::test]
    async fn a_non_error_frame_from_discard_files_the_unexpected_reply_cell() {
        dirty_cell_scenario(DirtyCase::UnexpectedDiscardFrame).await;
    }

    /// The classifier defect, end to end (b.7td round 2).
    ///
    /// `EXEC` and `UNWATCH` produce the same `without MULTI` tail `DISCARD`
    /// does, so a classifier matching the bare tail read `-ERR EXEC without
    /// MULTI` — a reply to somebody ELSE's command arriving in the `DISCARD`
    /// slot, i.e. the definition of an offset stream — as the CLEAN verdict and
    /// shipped the slot unrepaired. `txn_checkout`'s unit test pins the
    /// predicate; this pins that the checkout built on it actually recycles.
    #[tokio::test]
    async fn an_exec_without_multi_error_files_the_unexpected_reply_cell() {
        dirty_cell_scenario(DirtyCase::UnexpectedExecWithoutMulti).await;
    }

    /// The second `unexpected_reply` arm, which only exists because the probe
    /// has two commands.
    ///
    /// Reaching it requires the `DISCARD` probe to take the real round-trip and
    /// come back clean, so this is also the test that proves the seam is keyed
    /// per command rather than per checkout — a shared queue could not drive
    /// this arm without first satisfying `DISCARD`. `+QUEUED` is the shape a
    /// real server produces when an `UNWATCH` is swallowed by a transaction
    /// step 1 just said was not there.
    #[tokio::test]
    async fn a_queued_reply_to_the_unwatch_probe_files_the_unexpected_reply_cell() {
        dirty_cell_scenario(DirtyCase::UnexpectedUnwatchReply).await;
    }

    /// A probe that never answers. The delay runs through the production
    /// `tokio::time::timeout`, so the `Timeout` this classifies is produced by
    /// the real mechanism rather than asserted into existence.
    #[tokio::test]
    async fn a_hung_probe_files_the_probe_timeout_cell() {
        dirty_cell_scenario(DirtyCase::ProbeTimeout).await;
    }

    /// A probe rejected with an error kind that does NOT say the connection is
    /// gone: the slot's state is unknown, which is dirty, and filed under its
    /// own cell so an operator can tell a rejection from a wedge.
    #[tokio::test]
    async fn a_rejected_probe_files_the_probe_error_cell() {
        dirty_cell_scenario(DirtyCase::ProbeError).await;
    }

    /// b.21r, txn-probe funnel: a probe `DISCARD` whose real per-command
    /// deadline expires counts exactly one
    /// `valkey_errors_total{pool="txn", kind="command_timeout"}`. A probe the
    /// server REJECTS (an `Auth` error here) is a dirty verdict too, but not a
    /// timeout, and counts nothing — the control that shows the count keys on
    /// the error kind rather than on the dirty verdict.
    ///
    /// The hang runs through the production `tokio::time::timeout`, so the
    /// `Timeout` counted is produced by the real mechanism.
    #[tokio::test]
    async fn a_hung_probe_counts_one_txn_command_timeout_and_a_rejected_one_counts_none() {
        let (tv, metrics, injector) = seam_fixture().await;
        let key = tv.key_schema().obj_header("b21r-probe-timeout");

        injector.push_discard(ProbeFault::error(
            ErrorKind::Auth,
            "injected probe rejection",
        ));
        drop(tv.client().acquire_txn_for(&key).await);
        assert_eq!(
            injector.remaining(),
            0,
            "precondition: the rejection was consumed"
        );
        assert_eq!(
            live_edit::storage::test_support::command_timeout_row(&metrics),
            live_edit::storage::test_support::command_timeouts_only_on("txn", 0),
            "a rejected probe is not a command timeout"
        );

        injector.push_discard(ProbeFault::hang());
        drop(tv.client().acquire_txn_for(&key).await);
        assert_eq!(
            injector.remaining(),
            0,
            "precondition: the hang was consumed"
        );
        assert_eq!(
            live_edit::storage::test_support::command_timeout_row(&metrics),
            live_edit::storage::test_support::command_timeouts_only_on("txn", 1),
            "one probe whose deadline expired must count exactly once, on the txn pool"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // T6b — cluster-topology replies in a probe slot (b.289 sub-task 3)
    // ═════════════════════════════════════════════════════════════════════════

    /// A `-MOVED` in the `DISCARD` slot is a node-state transition, not an
    /// offset reply stream.
    ///
    /// The verdict pre-b.289 was `unexpected_reply`, which is **state-proof**:
    /// three of them and the repair-ineffective breaker latches, withholding
    /// checkout hygiene and logging an ERROR that accuses the classifier — all
    /// during a resharding an operator started on purpose. The frame did arrive,
    /// but what it proves is that the *node* declined the command, not that this
    /// connection's reply stream has drifted, so it belongs on the streak-gated
    /// unknown-state path with `probe_timeout` / `probe_error`.
    ///
    /// [`dirty_cell_scenario`] carries the whole-counter assertion, the WARN
    /// field set (including `state_proof_streak=0` and `unknown_state_streak=1`,
    /// which is the split under test), the rebuild witness and the
    /// drained-queue fall-through. The "never reaches the breaker" half needs a
    /// sustained run and is
    /// [`sustained_topology_replies_never_reach_the_repair_ineffective_breaker`].
    #[tokio::test]
    async fn a_moved_reply_to_the_discard_probe_files_the_topology_reply_cell() {
        dirty_cell_scenario(DirtyCase::TopologyMovedOnDiscard).await;
    }

    /// The second classifier arm, reached only through the `UNWATCH` probe.
    ///
    /// `probe_txn_slot` classifies two replies and b.289 wired the topology arm
    /// into both. Reaching this one requires the `DISCARD` probe to take the
    /// real round trip and come back clean, so a fix applied to the `DISCARD`
    /// classifier alone is green on every `DISCARD`-driven case above and red
    /// only here.
    #[tokio::test]
    async fn a_moved_reply_to_the_unwatch_probe_files_the_topology_reply_cell() {
        dirty_cell_scenario(DirtyCase::TopologyMovedOnUnwatch).await;
    }

    /// A non-`MOVED` prefix, end to end.
    ///
    /// `LOADING` is the arm with no upstream interception to hide behind: fred's
    /// `check_global_reconnect_errors` list — the only place `LOADING` is
    /// recognised — sits behind the `custom-reconnect-errors` cargo feature,
    /// which this build does not enable, so a `-LOADING` from a replica coming
    /// back up reaches the probe classifier unconditionally. Covering more than
    /// `MOVED` is what proves the classifier reads a *table* rather than one
    /// hard-coded prefix.
    #[tokio::test]
    async fn a_loading_reply_files_the_topology_reply_cell() {
        dirty_cell_scenario(DirtyCase::TopologyLoadingOnDiscard).await;
    }

    /// A lower-cased rendering must classify identically.
    ///
    /// This is not decoration: every interception fred performs is a
    /// case-**sensitive** `starts_with`, so a non-canonically-cased
    /// `-clusterdown …` from a managed proxy is exactly the frame that reaches
    /// this classifier instead of being handled upstream. For `CLUSTERDOWN` —
    /// which fred otherwise tests on every read frame, with no token-count
    /// escape — casing is the *only* way it gets here at all, so a
    /// case-sensitive classifier would leave the one reachable `CLUSTERDOWN`
    /// path filed as state-proof.
    #[tokio::test]
    async fn a_lower_cased_clusterdown_reply_files_the_topology_reply_cell() {
        dirty_cell_scenario(DirtyCase::TopologyLowerCaseClusterdownOnDiscard).await;
    }

    /// The same payload in RESP3's other error framing must reach the same
    /// verdict.
    ///
    /// RESP3 lets a server answer with `$`-framed `BlobError` where it might
    /// have sent `-`-framed `SimpleError`, and the two are one condition. A
    /// classifier that read only `SimpleError` would file a routine reshard as
    /// `unexpected_reply` — state-proof, breaker-eligible — purely because of
    /// the encoding. Paired with
    /// [`a_moved_reply_to_the_discard_probe_files_the_topology_reply_cell`],
    /// which injects the identical payload as a `SimpleError`, so the two
    /// together are a differential on the framing and nothing else.
    #[tokio::test]
    async fn a_blob_framed_moved_reply_files_the_topology_reply_cell() {
        dirty_cell_scenario(DirtyCase::TopologyBlobErrorOnDiscard).await;
    }

    /// How many consecutive topology replies the sustained run drives.
    ///
    /// Six: two past [`STATE_PROOF_REPAIR_ATTEMPTS`] + 1 = 4, the streak at
    /// which the breaker would trip if these verdicts were state-proof. Stopping
    /// at 4 would show only the tripping verdict; going two further shows the
    /// breaker staying silent through the checkouts that would have been its
    /// *withheld* ones, which is the state an operator would actually be looking
    /// at mid-reshard.
    const TOPOLOGY_STREAK_CHECKOUTS: u32 = 6;

    /// A resharding must not be able to latch the repair-ineffective breaker.
    ///
    /// # The failure this bounds
    ///
    /// The breaker's ERROR says "rebuilding this slot is not clearing the dirty
    /// verdict", and its investigation is "the classifier and the endpoint
    /// disagree about a reply shape". Both are false statements about a
    /// resharding: the verdict is correct, the endpoint is behaving, and the
    /// event was operator-initiated. Pre-b.289 a topology reply was
    /// `unexpected_reply` — state-proof — so the *fourth* one in a row latched
    /// the breaker, and every checkout after it was handed over unrepaired with
    /// hygiene switched off until some checkout probed clean again. On a cluster
    /// mid-reshard, where the node often cannot be named at all, that latch
    /// lands on the shared fallback bucket, i.e. at its widest scope.
    ///
    /// # What is asserted
    ///
    /// 1. The `topology_reply` cell ticks on every one of the six verdicts and
    ///    no sibling cell moves — the counter is the operator's evidence and it
    ///    must not go quiet just because the repair is gated.
    /// 2. `live_edit_valkey_txn_repair_ineffective_total` stays at zero on
    ///    **both** its cells, and the escalation ERROR is never emitted. This is
    ///    the assertion the whole test exists for.
    /// 3. The state-proof streak stays 0 throughout while the pool-wide
    ///    unknown-state streak advances 1..=6 — the two are different budgets,
    ///    and a verdict advancing the wrong one is how a topology reply would
    ///    reach the breaker in the first place.
    /// 4. The repair is streak-gated exactly like its unknown-state siblings:
    ///    the first of the streak rebuilds and the rest are suppressed inside
    ///    the re-arm window, so a reshard touching every slot in the pool is not
    ///    answered with a pool-wide reconnect storm on top of it.
    ///
    /// # Runtime
    ///
    /// A canned error frame short-circuits `probe_txn_slot` with no wall-clock
    /// deadline, so the six checkouts pay only the one real rebuild — unlike the
    /// `probe_timeout` gate test, which must hang the probe for 650 ms per
    /// checkout to reach its verdict at all.
    #[tokio::test]
    async fn sustained_topology_replies_never_reach_the_repair_ineffective_breaker() {
        let (tv, metrics, injector) = seam_fixture().await;
        let key = tv.key_schema().obj_header("289-topology-streak");

        // Baseline BEFORE the queue is armed: it must take the real probe and
        // come back Clean, or it would consume the first fault and shift every
        // streak below by one.
        let mut connection_id = {
            let guard = tv.client().acquire_txn_for(&key).await;
            client_id(&guard).await
        };
        injector.push_discard_repeated(TOPOLOGY_STREAK_CHECKOUTS as usize, || {
            ProbeFault::simple_error(MOVED_PAYLOAD)
        });

        let mut recycled_at: Vec<u32> = Vec::new();

        for streak in 1..=TOPOLOGY_STREAK_CHECKOUTS {
            let (observed_id, logged) = checkout_capturing_logs(&tv, &key).await;

            assert_dirty_event(
                &logged,
                REASON_TOPOLOGY_REPLY,
                Streaks::unknown_state(streak),
                streak != 1,
            );
            assert!(
                lines_with(&logged, REPAIR_INEFFECTIVE_ERROR).is_empty(),
                "checkout {streak}: the breaker's escalation ERROR must never be emitted for a \
                 topology reply. It reads 'rebuilding is not clearing the dirty verdict' and \
                 points the investigation at the classifier — both false statements about a \
                 resharding, which is an event the operator started on purpose. Captured output: \
                 {logged:?}"
            );
            assert_eq!(
                tv.client().txn_unknown_state_streak_for_test(),
                streak,
                "checkout {streak}: a topology reply must advance the POOL-WIDE unknown-state \
                 streak, which is what schedules the next re-arm of its repair"
            );
            assert_eq!(
                tv.client().txn_state_proof_streak_for_test(),
                0,
                "checkout {streak}: a topology reply must leave the state-proof streak at zero. \
                 That streak is the breaker's whole input, so advancing it is exactly how a \
                 routine reshard would latch a breaker it must never be able to reach — and this \
                 accessor reads the MAX across every node bucket, so a verdict that leaked into \
                 any bucket, named or shared, fails here"
            );
            assert_eq!(
                tv.client().txn_state_proof_streak_nodes_for_test(),
                Vec::<String>::new(),
                "checkout {streak}: a topology reply must create no streak bucket at all — not \
                 even the shared fallback one. A bucket materialising here would mean the verdict \
                 took the state-proof path and merely happened not to have reached the threshold \
                 yet"
            );

            if observed_id != connection_id {
                recycled_at.push(streak);
                connection_id = observed_id;
            }
        }

        assert_eq!(
            read_all_dirty(&metrics),
            only_cell(REASON_TOPOLOGY_REPLY, TOPOLOGY_STREAK_CHECKOUTS as u64),
            "every verdict must be counted under `topology_reply` and no sibling cell may move. \
             This cell is the one an operator is EXPECTED to see non-zero during a topology \
             change, so misfiling a reshard under `unexpected_reply` would page them on an event \
             they scheduled"
        );
        assert_eq!(
            read_all_ineffective(&metrics),
            only_ineffective_cell("none", 0),
            "{TOPOLOGY_STREAK_CHECKOUTS} consecutive topology replies — two past the streak at \
             which a state-proof verdict trips the breaker — must leave \
             live_edit_valkey_txn_repair_ineffective_total flat on BOTH cells. The counter is the \
             continuous record that hygiene has been switched off for a node; a reshard must not \
             be able to write to it"
        );
        assert_eq!(
            recycled_at,
            vec![1],
            "the repair must be streak-gated exactly like its unknown-state siblings: the first \
             of the streak rebuilds — a slot that really is wedged still heals immediately — and \
             the rest are suppressed inside the re-arm window. A reshard turns every slot in the \
             pool dirty at once, so an ungated repair would answer it with a pool-wide quit + \
             reconnect storm at 128 retry attempts per request. Rebuilds were observed at \
             {recycled_at:?}"
        );

        // The queue is drained, so this checkout takes the real probe, finds the
        // slot clean, and ends the streak — the reshard is over.
        let (_, quiet) = checkout_capturing_logs(&tv, &key).await;
        assert_eq!(
            tv.client().txn_unknown_state_streak_for_test(),
            0,
            "a Clean checkout must end the streak, so the first verdict after the topology change \
             settles repairs immediately rather than serving out a stale suppression window. \
             Captured output: {quiet:?}"
        );
    }

    /// A probe failure whose kind says the connection is already gone must move
    /// **nothing**.
    ///
    /// A `MULTI` and a `WATCH` both die with their connection, so there is no
    /// residue for the next borrower to inherit and fred's own reconnect
    /// machinery owns the recovery. This is the common failure on a pool that
    /// sees failover, managed-proxy recycles and ordinary reconnects — counting
    /// it would fire the dirty-checkout alert on routine churn, and recycling on
    /// it would answer a reconnect with another reconnect.
    #[tokio::test]
    async fn a_moot_probe_failure_moves_no_cell_and_leaves_the_streak_alone() {
        moot_probe_failure_scenario(ErrorKind::Canceled, "injected connection teardown").await;
    }

    /// fred's routing miss is moot too, end to end through the checkout: the
    /// router holds no connection at all, so no `MULTI` or `WATCH` can be left.
    /// The vendored fred's change 9 makes it reachable on this pool — a fred
    /// `Transaction` cut partway closes its connection and defers the
    /// reconnect, and a borrower's probe can be routed ahead of that
    /// reconnect. Counting it as a `probe_error` would feed the slot's recycle
    /// gate a reconnect it is already running.
    #[tokio::test]
    async fn a_routing_miss_probe_failure_is_moot_and_moves_no_cell() {
        moot_probe_failure_scenario(ErrorKind::Unknown, "Failed to route command.").await;
    }

    /// The moot scenario, its probe failing with `kind` / `details`.
    ///
    /// The streak is advanced to 1 first, deliberately: asserting it is
    /// *untouched* requires a non-zero value to be untouched. Against a zero
    /// streak this test could not tell "moot leaves the streak alone" from
    /// "moot resets it", and a moot verdict that reset the streak would re-arm
    /// the recycle gate on every proxy blip — turning the storm bound into a
    /// suggestion in exactly the regime it is there for.
    async fn moot_probe_failure_scenario(kind: ErrorKind, details: &'static str) {
        let (tv, metrics, injector) = seam_fixture().await;
        let key = tv.key_schema().obj_header("7td-moot");

        injector.push_discard(ProbeFault::hang());
        let (connection_before, _) = checkout_capturing_logs(&tv, &key).await;
        assert_eq!(
            tv.client().txn_unknown_state_streak_for_test(),
            1,
            "precondition: the seeded probe_timeout must leave the streak at 1"
        );
        let cells_before = read_all_dirty(&metrics);

        injector.push_discard(ProbeFault::error(kind, details));
        let (connection_after, logged) = checkout_capturing_logs(&tv, &key).await;

        assert_eq!(
            read_all_dirty(&metrics),
            cells_before,
            "a probe failure that says the connection is already gone must tick NO cell: there is \
             no leaked state to inherit from a torn-down connection, and counting it would page \
             an operator on routine failover and managed-proxy recycles. Captured output: \
             {logged:?}"
        );
        assert!(
            lines_with(&logged, DIRTY_CHECKOUT_EVENT).is_empty(),
            "a moot verdict must not emit the dirty-checkout WARN — it is not a dirty slot. \
             Captured output: {logged:?}"
        );
        assert!(
            lines_with(&logged, RECYCLE_SUCCESS_INFO).is_empty(),
            "a moot verdict must not recycle: fred is already rebuilding the connection, and a \
             quit + reconnect on top of that is load added to a transition. Captured output: \
             {logged:?}"
        );
        assert_eq!(
            connection_after, connection_before,
            "the slot must be handed over untouched after a moot verdict"
        );
        assert_eq!(
            tv.client().txn_unknown_state_streak_for_test(),
            1,
            "a moot verdict must leave the unknown-state streak exactly as it found it — neither \
             advanced (it is not evidence of a wedge) nor reset (it is not evidence the incident \
             is over). Resetting here would re-arm the recycle gate on every reconnect, which is \
             the storm the gate bounds"
        );
    }

    /// How many consecutive unknown-state checkouts the gate test drives.
    ///
    /// Ten covers the first re-arm boundary with a checkout on either side of
    /// it: recycles at streak 1 and 9, suppression at 2–8 and at 10.
    const STREAK_CHECKOUTS: u32 = 10;

    /// The recycle is rate-limited for unknown-state verdicts; the detection
    /// never is.
    ///
    /// A `probe_timeout` says only "no answer arrived in 650 ms", which is also
    /// what a store-wide latency event says — and under one of those every
    /// checkout across the pool times out at once. An ungated repair would
    /// answer a slow backing store with a pool-wide `QUIT` + reconnect storm,
    /// compounding because `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (128)
    /// re-enters the checkout per retry attempt. So the repair fires on the
    /// first of a streak and every [`RECYCLE_REARM_EVERY`]th thereafter, while
    /// the counter ticks on all ten.
    ///
    /// The server-side connection id is the witness for the repair half: it
    /// changes on a rebuild and on nothing else, so "recycled at streaks 1 and
    /// 9, and only there" is observed rather than inferred from a log line.
    ///
    /// Cost: ten checkouts × the real 650 ms probe deadline ≈ 7 s, spent in the
    /// production `tokio::time::timeout` rather than simulated. Well inside
    /// nextest's 40 s slow mark.
    #[tokio::test]
    async fn the_recycle_gate_repairs_on_the_first_of_a_streak_then_every_eighth() {
        let (tv, metrics, injector) = seam_fixture().await;
        let key = tv.key_schema().obj_header("7td-streak");

        // The baseline checkout runs BEFORE the queue is armed: it must take the
        // real probe and come back Clean, or it would consume the first fault
        // and shift every streak in the loop by one.
        let mut connection_id = {
            let guard = tv.client().acquire_txn_for(&key).await;
            client_id(&guard).await
        };
        injector.push_discard_repeated(STREAK_CHECKOUTS as usize, ProbeFault::hang);

        let mut recycled_at: Vec<u32> = Vec::new();

        for streak in 1..=STREAK_CHECKOUTS {
            let expected_recycle = streak == 1 || streak == 1 + RECYCLE_REARM_EVERY;
            let (observed_id, logged) = checkout_capturing_logs(&tv, &key).await;

            assert_dirty_event(
                &logged,
                "probe_timeout",
                Streaks::unknown_state(streak),
                !expected_recycle,
            );
            assert_eq!(
                tv.client().txn_unknown_state_streak_for_test(),
                streak,
                "checkout {streak}: the streak must advance on EVERY unknown-state verdict, \
                 suppressed or not — it is what schedules the next re-arm, so a streak that only \
                 counted the repairs would never reach one"
            );

            if observed_id != connection_id {
                recycled_at.push(streak);
                connection_id = observed_id;
            }
        }

        assert_eq!(
            recycled_at,
            vec![1, 1 + RECYCLE_REARM_EVERY],
            "the slot must be rebuilt on the FIRST unknown-state verdict — delaying it would \
             reintroduce the wedged-slot window the probe was written for — and then not again \
             until the re-arm at streak {rearm}, so a genuinely wedged slot still heals while a \
             slow backing store is not answered with a reconnect per checkout per retry attempt. \
             Rebuilds were observed at {recycled_at:?}",
            rearm = 1 + RECYCLE_REARM_EVERY
        );
        assert_eq!(
            read_all_dirty(&metrics),
            only_cell("probe_timeout", STREAK_CHECKOUTS as u64),
            "the counter must tick on every dirty verdict, gated or not: it means 'the probe \
             found this slot unfit', which is true whether or not the repair was admitted. A \
             counter that only tracked the repairs would make a suppression window look like a \
             healthy pool"
        );
        assert_eq!(
            read_all_ineffective(&metrics),
            only_ineffective_cell("none", 0),
            "ten consecutive unknown-state verdicts must never reach the repair-ineffective \
             breaker. The two rate limits are deliberately different mechanisms — this gate \
             re-arms on a cadence because its input (a slow store) is expected to pass, while the \
             breaker latches until a Clean proves the endpoint can answer correctly — and crossing \
             them would put a store-wide latency event on a counter whose meaning is 'rebuilding \
             the connection changed nothing'"
        );

        // The queue is drained, so this checkout takes the real probe and comes
        // back Clean — which ends the streak.
        let (_, quiet) = checkout_capturing_logs(&tv, &key).await;
        assert_eq!(
            tv.client().txn_unknown_state_streak_for_test(),
            0,
            "a Clean checkout must reset the streak: whatever made the last ten time out is over, \
             so the next verdict that does not is the first of a fresh streak and must repair \
             immediately rather than serve out the remainder of a stale suppression window. \
             Captured output: {quiet:?}"
        );
    }

    /// A state-proof verdict recycles even inside a suppression window.
    ///
    /// The gate applies to `probe_timeout` / `probe_error` and to nothing else.
    /// Suppressing a `leaked_multi` would hand a *provably* poisoned connection
    /// to a borrower — the exact outcome this module exists to prevent — and it
    /// would do so precisely when the pool is already unhealthy, which is when
    /// a leaked `MULTI` is most likely to be there in the first place.
    #[tokio::test]
    async fn a_state_proof_verdict_recycles_inside_the_suppression_window() {
        let (tv, metrics, injector) = seam_fixture().await;
        let key = tv.key_schema().obj_header("7td-mixed");

        // Streak 1 repairs; streak 2 is inside the window and must not.
        injector.push_discard_repeated(2, ProbeFault::hang);
        let (after_first, _) = checkout_capturing_logs(&tv, &key).await;
        let (after_second, suppressed_log) = checkout_capturing_logs(&tv, &key).await;
        assert_dirty_event(
            &suppressed_log,
            "probe_timeout",
            Streaks::unknown_state(2),
            true,
        );
        assert_eq!(
            after_second, after_first,
            "precondition: the second unknown-state verdict must be SUPPRESSED, or the assertion \
             below cannot show that a state-proof verdict escapes a window it is not in"
        );

        injector.push_discard(ProbeFault::ok());
        let (after_leak, logged) = checkout_capturing_logs(&tv, &key).await;

        assert_dirty_event(&logged, "leaked_multi", Streaks::state_proof(1), false);
        assert_ne!(
            after_leak, after_second,
            "a leaked MULTI must recycle the slot IMMEDIATELY, whatever the unknown-state streak \
             is doing. The gate exists to keep a slow backing store from being answered with a \
             reconnect storm; a `DISCARD` that was acknowledged is not a slow store, it is direct \
             evidence that this connection is carrying another borrower's transaction, and \
             handing it over is the failure b.7td was filed for"
        );
        assert_eq!(
            lines_with(&logged, RECYCLE_SUCCESS_INFO).len(),
            1,
            "the state-proof repair must run to completion. Captured output: {logged:?}"
        );
        assert_eq!(
            read_all_dirty(&metrics)
                .into_iter()
                .find(|(reason, _)| *reason == "leaked_multi"),
            Some(("leaked_multi", 1)),
            "the state-proof verdict must be counted under its own cell"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // T7 — the repair-ineffective breaker (b.7td round 3)
    // ═════════════════════════════════════════════════════════════════════════

    /// How many consecutive state-proof verdicts the breaker test drives.
    ///
    /// Thirteen, and each boundary is load-bearing. The breaker trips at
    /// [`STATE_PROOF_REPAIR_ATTEMPTS`] + 1 = 4 and re-escalates every
    /// [`RECYCLE_REARM_EVERY`]th after that, so 12 is the second escalation —
    /// reaching it is what proves the cadence is periodic rather than a
    /// one-shot ERROR that happens to fire once. Going one past it to 13 proves
    /// the ERROR then goes quiet again while the repair stays withheld and the
    /// counter keeps ticking, which is precisely the state that would otherwise
    /// be indistinguishable from "the breaker released".
    const BREAKER_CHECKOUTS: u32 = 13;

    /// Rebuilding the slot stops once rebuilding has been shown not to work —
    /// and the fact that it stopped is neither silent nor a log flood.
    ///
    /// # The failure this bounds
    ///
    /// Recycling on every state-proof verdict is right for real residue: one
    /// rebuild clears an abandoned `MULTI`, the next checkout reads `Clean`, and
    /// the streak never gets past 1. It is *self-amplifying* when the verdict
    /// comes from the classifier and the server disagreeing instead — a managed
    /// proxy that renders the `DISCARD` error without naming `DISCARD` reads as
    /// `unexpected_reply` forever, and a freshly built connection answers
    /// identically. Every checkout in the pool then quits and reconnects, for as
    /// long as that deployment renders that reply that way, with
    /// `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (128) re-entering the
    /// checkout per retry attempt. That is a pod that has replaced serving
    /// traffic with reconnecting, over a verdict a rebuild cannot fix.
    ///
    /// # What is asserted, in the order the breaker moves through it
    ///
    /// 1. **Streaks 1–3 repair.** The one-shot path is untouched: the
    ///    server-side connection id changes on each, which is the only witness
    ///    that separates a rebuild from a probe that merely reported.
    /// 2. **Streak 4 withholds and escalates.** The id stops changing — the
    ///    borrower is handed the slot unrepaired — and the distinct ERROR fires.
    /// 3. **Streaks 5–11 and 13 withhold quietly.** The id stays put, the
    ///    ordinary event still carries `suppressed=true` and an advancing
    ///    `state_proof_streak` at `debug!`, and no ERROR is emitted. This is the
    ///    anti-flood half, and it is what makes the DEBUG capture in
    ///    [`checkout_capturing_logs`] necessary rather than incidental.
    /// 4. **Streak 12 re-escalates.** Every eighth after the trip, so a
    ///    long-running incident keeps saying so.
    /// 5. **The counter ticks on every withheld verdict** — ten of them — while
    ///    the dirty counter ticks on all thirteen. The two together are what
    ///    keep a withheld repair from being invisible: the breaker's rate reads
    ///    "how often are we handing out a slot we cannot fix", which is a
    ///    different investigation from its sibling's "the probe found slots
    ///    unfit".
    /// 6. **A `Clean` checkout re-arms it.** The next canned verdict recycles
    ///    again. Without this the breaker would be a one-way latch and a
    ///    transient burst — a borrower that really did die mid-transaction four
    ///    times in a row — would disable the repair for the life of the process.
    ///
    /// # Runtime
    ///
    /// A canned `+OK` on `DISCARD` short-circuits `probe_txn_slot` before the
    /// `UNWATCH` round-trip and costs no wall-clock deadline, so the thirteen
    /// checkouts pay only three real rebuilds (streaks 1–3) plus one at the
    /// re-arm. Nothing here waits on the 650 ms probe deadline, unlike the
    /// unknown-state gate test above, which must hang the probe to reach its
    /// verdict at all.
    #[tokio::test]
    async fn the_breaker_withholds_the_repair_once_rebuilding_stops_clearing_the_verdict() {
        let (tv, metrics, injector) = seam_fixture().await;
        let key = tv.key_schema().obj_header("7td-breaker");

        // The baseline checkout runs BEFORE the queue is armed: it must take the
        // real probe and come back Clean, or it would consume the first fault
        // and shift every streak in the loop by one.
        let mut connection_id = {
            let guard = tv.client().acquire_txn_for(&key).await;
            client_id(&guard).await
        };
        injector.push_discard_repeated(BREAKER_CHECKOUTS as usize, ProbeFault::ok);

        let mut recycled_at: Vec<u32> = Vec::new();
        let mut escalated_at: Vec<u32> = Vec::new();

        for streak in 1..=BREAKER_CHECKOUTS {
            let withheld = streak > STATE_PROOF_REPAIR_ATTEMPTS;
            let (observed_id, logged) = checkout_capturing_logs(&tv, &key).await;

            if lines_with(&logged, REPAIR_INEFFECTIVE_ERROR).is_empty() {
                assert_dirty_event(
                    &logged,
                    "leaked_multi",
                    Streaks::state_proof(streak),
                    withheld,
                );
            } else {
                escalated_at.push(streak);
                assert_ineffective_escalation(&logged, "leaked_multi", streak);
            }

            assert_eq!(
                tv.client().txn_state_proof_streak_for_test(),
                streak,
                "checkout {streak}: the state-proof streak must advance on EVERY verdict, \
                 withheld or not — it is what schedules the next escalation, so a streak that \
                 only counted the repairs would stall at 3 and the breaker would never re-arm \
                 its ERROR"
            );
            assert_eq!(
                tv.client().txn_unknown_state_streak_for_test(),
                0,
                "checkout {streak}: a state-proof verdict must leave the unknown-state streak \
                 alone. The two gates are independent budgets over independent failure classes, \
                 and a leaked MULTI is no evidence at all about whether the backing store is slow"
            );
            assert_eq!(
                read_dirty(&metrics, REASON_LEAKED_MULTI),
                streak as u64,
                "checkout {streak}: the dirty counter must keep ticking after the breaker trips. \
                 It records the VERDICT, which is still true; only the repair is withheld. A \
                 counter that went quiet with the repair would make the worst regime — every \
                 checkout unfit, none of them fixable — look like a pool that had healed"
            );
            assert_eq!(
                read_ineffective(&metrics, REASON_LEAKED_MULTI),
                streak.saturating_sub(STATE_PROOF_REPAIR_ATTEMPTS) as u64,
                "checkout {streak}: live_edit_valkey_txn_repair_ineffective_total must tick on \
                 EVERY withheld verdict and on no repaired one. The ERROR is rate-limited to one \
                 in eight, so this counter is the only continuous record that the service is \
                 handing out slots it has given up on — an operator's alert has nothing else to \
                 fire on between escalations"
            );

            if observed_id != connection_id {
                recycled_at.push(streak);
                connection_id = observed_id;
            }
        }

        assert_eq!(
            recycled_at,
            (1..=STATE_PROOF_REPAIR_ATTEMPTS).collect::<Vec<_>>(),
            "the slot must be rebuilt on each of the first {STATE_PROOF_REPAIR_ATTEMPTS} \
             state-proof verdicts and then NEVER again while the streak runs. The first three are \
             the evidence: verdicts 2, 3 and 4 each land on the socket the previous verdict's \
             recycle built, so a 4th identical answer is the third post-rebuild re-observation and \
             proves the rebuild is not what is wrong. Continuing past it is a pool-wide quit + \
             reconnect loop that cannot converge, at 128 retry attempts per request. Rebuilds were \
             observed at {recycled_at:?}"
        );
        assert_eq!(
            escalated_at,
            vec![
                STATE_PROOF_REPAIR_ATTEMPTS + 1,
                STATE_PROOF_REPAIR_ATTEMPTS + 1 + RECYCLE_REARM_EVERY
            ],
            "the ERROR must fire on the verdict that trips the breaker and then only every \
             {RECYCLE_REARM_EVERY}th. Loud on every withheld checkout is a log flood describing \
             one unchanging fact at exactly the moment the pipeline can least absorb one; loud \
             only once would let a multi-hour incident scroll out of a retention window and read \
             as resolved. Escalations were observed at {escalated_at:?}"
        );
        assert_eq!(
            read_all_dirty(&metrics),
            only_cell(REASON_LEAKED_MULTI, BREAKER_CHECKOUTS as u64),
            "every verdict must be counted under its own cell, and no sibling may move"
        );
        assert_eq!(
            read_all_ineffective(&metrics),
            only_ineffective_cell(
                REASON_LEAKED_MULTI,
                (BREAKER_CHECKOUTS - STATE_PROOF_REPAIR_ATTEMPTS) as u64
            ),
            "the breaker counter must carry exactly the withheld verdicts, filed under the reason \
             that was withheld — that label is what points the investigation at the classifier arm \
             responsible, and it is the only thing separating 'the proxy re-renders our DISCARD \
             error' from 'borrowers keep dying inside MULTI'"
        );

        // Re-arm: the queue is drained, so this checkout takes the real probe,
        // finds the slot clean, and ends the streak.
        let (clean_id, quiet) = checkout_capturing_logs(&tv, &key).await;
        assert_eq!(
            tv.client().txn_state_proof_streak_for_test(),
            0,
            "a Clean checkout must reset the state-proof streak. The slot has just demonstrated \
             that the classifier and this endpoint DO agree about a reply, which is the one piece \
             of evidence that can retire the breaker's conclusion. Captured output: {quiet:?}"
        );
        assert!(
            lines_with(&quiet, DIRTY_CHECKOUT_EVENT).is_empty()
                && lines_with(&quiet, REPAIR_INEFFECTIVE_ERROR).is_empty(),
            "the Clean checkout must be silent. Captured output: {quiet:?}"
        );

        injector.push_discard(ProbeFault::ok());
        let (rearmed_id, rearmed_log) = checkout_capturing_logs(&tv, &key).await;
        assert_dirty_event(
            &rearmed_log,
            REASON_LEAKED_MULTI,
            Streaks::state_proof(1),
            false,
        );
        assert_ne!(
            rearmed_id, clean_id,
            "after a Clean checkout the very next state-proof verdict must be repaired again. The \
             breaker is a conclusion about evidence, not a latch: a transient burst — a borrower \
             that really did die mid-transaction several times running — must not disable the \
             repair for the life of the process, or a pod that saw one bad minute serves poisoned \
             slots until it is restarted"
        );
        assert_eq!(
            read_all_ineffective(&metrics),
            only_ineffective_cell(
                REASON_LEAKED_MULTI,
                (BREAKER_CHECKOUTS - STATE_PROOF_REPAIR_ATTEMPTS) as u64
            ),
            "a repaired verdict must not tick the breaker counter. Captured output: {rearmed_log:?}"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // T8 — the two streaks are independent budgets
    // ═════════════════════════════════════════════════════════════════════════

    /// Each gate advances and reports its own streak, and neither disturbs nor
    /// borrows the other's.
    ///
    /// `acquire_txn_inner` keeps two counters — one pool-wide, one per node
    /// (b.289 sub-task 4; on this centralized fixture the per-node one has a
    /// single shared bucket, so both read as one number) — and consults exactly
    /// one per verdict, logging both. Three things could go wrong and only a
    /// test that seeds *both* can see any of them:
    ///
    /// - **A verdict advancing the wrong counter.** A `leaked_multi` that
    ///   incremented the unknown-state streak would walk the pool into a
    ///   suppression window it never earned, and the next genuine
    ///   `probe_timeout` — a real wedge — would go unrepaired.
    /// - **A verdict resetting the other counter.** Only a `Clean` checkout is
    ///   evidence that a condition is over. A state-proof verdict clearing the
    ///   unknown-state streak would re-arm that gate on every checkout during a
    ///   store-wide latency event, which is exactly the reconnect storm it
    ///   bounds.
    /// - **The event rendering the other counter's stored value.** The log line
    ///   is where an operator sizes the incident. `unknown_state_streak` on a
    ///   state-proof verdict must read `0` — "this gate had no say" — and not
    ///   the 2 still sitting in the atomic from an unrelated earlier verdict.
    ///
    /// That last distinction is the subtle one and the reason the unknown-state
    /// streak is seeded first: the assertions below hold the *stored* value at 2
    /// while requiring the *logged* value to be 0, so a naive implementation
    /// that printed the atomic would fail on a value that is otherwise correct.
    #[tokio::test]
    async fn the_two_streak_gates_advance_and_log_independently() {
        let (tv, metrics, injector) = seam_fixture().await;
        let key = tv.key_schema().obj_header("7td-independent");

        // Seed the unknown-state streak. One hung probe: the first of its
        // streak, so it repairs, and it leaves the atomic at 1.
        injector.push_discard(ProbeFault::hang());
        let (_, seeded) = checkout_capturing_logs(&tv, &key).await;
        assert_dirty_event(&seeded, "probe_timeout", Streaks::unknown_state(1), false);
        assert_eq!(
            tv.client().txn_state_proof_streak_for_test(),
            0,
            "an unknown-state verdict must not advance the state-proof streak: a probe that never \
             answered is no evidence at all about whether a rebuilt connection would answer \
             correctly, which is the only thing the breaker measures"
        );

        // A state-proof verdict on top of it: its own streak starts at 1, and
        // the unknown-state streak it never consulted is neither advanced nor
        // reset — while the event reports 0 for it regardless.
        injector.push_discard(ProbeFault::ok());
        let (_, state_proof_log) = checkout_capturing_logs(&tv, &key).await;
        assert_dirty_event(
            &state_proof_log,
            REASON_LEAKED_MULTI,
            Streaks::state_proof(1),
            false,
        );
        assert_eq!(
            tv.client().txn_unknown_state_streak_for_test(),
            1,
            "a state-proof verdict must leave the STORED unknown-state streak exactly as it found \
             it. Resetting it would re-arm the unknown-state gate on any interleaved leaked MULTI, \
             re-opening the pool-wide reconnect storm that gate exists to bound; advancing it \
             would open a suppression window the pool never earned and leave the next genuine \
             wedge unrepaired. Note this is a DIFFERENT number from the 0 the event above logged, \
             and both are correct: the log reports which gate decided, the atomic remembers what \
             each gate has seen"
        );

        // And back the other way: the next unknown-state verdict continues its
        // own streak from 2 — proving the state-proof verdict did not reset it —
        // while reporting 0 for the state-proof streak still sitting at 1.
        injector.push_discard(ProbeFault::hang());
        let (_, second_unknown) = checkout_capturing_logs(&tv, &key).await;
        assert_dirty_event(
            &second_unknown,
            "probe_timeout",
            Streaks::unknown_state(2),
            true,
        );
        assert_eq!(
            tv.client().txn_state_proof_streak_for_test(),
            1,
            "an unknown-state verdict must leave the STORED state-proof streak alone too. \
             Resetting it would let an interleaved slow-store blip erase the breaker's evidence \
             indefinitely — on a pool where every checkout is producing both classes, the breaker \
             would never reach its threshold and the unbounded recycle loop it exists to stop \
             would run forever"
        );
        assert_eq!(
            read_all_ineffective(&metrics),
            only_ineffective_cell("none", 0),
            "neither gate may reach the breaker at these streak lengths"
        );

        // A Clean checkout is the one verdict that clears BOTH, and it has to
        // clear both: it is positive evidence that the connection answers
        // correctly AND promptly.
        let (_, quiet) = checkout_capturing_logs(&tv, &key).await;
        assert_eq!(
            (
                tv.client().txn_unknown_state_streak_for_test(),
                tv.client().txn_state_proof_streak_for_test()
            ),
            (0, 0),
            "a Clean checkout must reset BOTH streaks — it is simultaneously evidence that the \
             store is answering promptly again and that the classifier and this endpoint agree \
             about a reply. Captured output: {quiet:?}"
        );
    }
}
