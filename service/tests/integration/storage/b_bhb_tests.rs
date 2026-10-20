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

//! b.bhb — a watched transaction must commit only on the connection that
//! holds its `WATCH`, and the service must be able to prove it did.
//!
//! # The defect
//!
//! Valkey scopes a `WATCH` to the *connection* that issued it and clears the
//! watch set the moment that connection goes away. Both WATCHed commit paths —
//! `LayerStore::update_layer_once_watch` (the structural slow path) and
//! `SessionStore::create_once` — sent their commit through fred's `Transaction`
//! (`client.multi()` … `exec()`), a separate router command from the `WATCH`.
//! fred re-issues a failed command on a fresh connection
//! (`ConnectionConfig::max_command_attempts`, default 3) and follows a `MOVED`
//! to a fresh node (`max_redirections`, default 5) — for a `Transaction` it
//! replays the whole `MULTI … EXEC` that way. A replayed `EXEC` runs
//! **unwatched**: Valkey commits it unconditionally, and the optimistic-locking
//! contract both call sites are built on is void with no error anywhere.
//!
//! The damage differs by site and neither is benign:
//!
//! * `create_once` loses its duplicate-name check — two sessions, one name, at
//!   one base path, which is the uniqueness guarantee `create` exists to give.
//! * `update_layer_once_watch` produces a TORN commit: the `HSET`s land while the
//!   `XADD {seq}-0` is refused by the winner's entry, so materialized state and
//!   the live relay stream disagree about what the layer contains.
//!
//! **No pre-`EXEC` check in our own code can see any of this**, because the
//! rewrite happens after the commands are handed to fred's router. That is why
//! every test of the three defence layers below stages a REAL connection loss —
//! `CLIENT KILL ID` against the exact socket carrying the `WATCH` — rather than
//! injecting a synthetic error. A simulated `Err` would exercise the
//! classification arms and prove nothing about the mechanism. (The in-doubt
//! reconciliation block is the one deliberate exception; see *Two staging
//! mechanisms* below for why it has to be, and what it still drives for real.)
//!
//! # What lives here
//!
//! | Test | Layer of the fix | What it pins |
//! |---|---|---|
//! | [`create_session_commit_dispatched_after_a_reconnect_keeps_the_name_unique`] | L3 witness + compensation | **the regression.** Pre-fix the `EXEC` lands unwatched and two members carry one name; post-fix the in-transaction `CLIENT ID` names the wrong connection, the commit is undone, and the retry re-decides |
//! | [`update_layer_torn_commit_after_a_void_watch_is_detected_and_reported_non_retriable`] | L3, the documented residual | a commit that ran unwatched and tore (`HSET`s landed, `XADD` collided) is *counted* on the alertable cell and reported non-retriable. Pre-fix the identical tear was a bare `INTERNAL` with nothing saying a concurrency check had been skipped |
//! | [`create_session_racing_create_inside_a_dead_watch_window_keeps_the_name_unique`] | L1 cost | the non-regression the attempt cap could have broken: a read taken under a dead `WATCH` must still resolve to the duplicate verdict, not a hard error |
//! | [`update_layer_racing_writer_inside_a_dead_watch_window_commits_at_the_next_sequence`] | L1 + L2 | the same, on the layer path: both writers' commits land at consecutive sequences with no torn state |
//! | [`update_layer_connection_replaced_before_dispatch_aborts_and_retries_cleanly`] | L2 pre-dispatch check | the cheap layer, in isolation: a replaced connection is caught before anything is written, and the retry commits normally |
//! | [`create_session_connection_replaced_before_dispatch_aborts_and_retries_cleanly`] | L2, session wiring | the same on the session path, which reports through a different metrics handle — and the availability half: a routine reconnect must not fail the caller's create |
//! | [`update_layer_witness_mismatch_on_a_live_object_accepts_the_commit`] | L3 witness, layer disposition | a proven-unwatched commit whose `XADD` was accepted is *reported as committed* — never as conditional, never retried (a retry would only rediscover it through the b.75i commit token, minus the external-value releases this attempt holds) — and ticks the alertable cell |
//! | [`update_layer_witness_mismatch_on_a_retired_object_surfaces_failed_precondition`] | L3 witness, retirement arm | the same proof, on an object retired underneath the commit, keeps the ordinary `FailedPrecondition` shape |
//! | [`txn_pool_pins_one_attempt_and_no_redirection_without_disturbing_the_data_pool`] | L1 pool config | the config fence, read back from a live pool: the divergence exists, is txn-pool-only, and everything else still matches the shared builder |
//! | [`create_session_reads_under_a_killed_watch_connection_stay_retriable`] | L1 cost, classification | the other half of the L1 bill: the `WATCH` and the duplicate scan lost fred's transparent retry, so a reconnect landing on either must classify as retriable. Pre-round-2 both wrapped `Internal` and a routine reconnect failed the caller's `CreateSession` outright |
//! | [`update_layer_exec_error_that_is_not_a_stream_id_collision_is_not_counted_as_a_void_watch`] | L3 counting discipline | the alertable cell's false-positive guard: a `WRONGTYPE` frame at `EXEC` tears the commit exactly as a stream-ID collision does, but is reachable with the `WATCH` fully intact, so it must NOT tick. Post-b.459 the only way there is the probe→`EXEC` window, which the test now stages through the pre-dispatch hook |
//! | [`layer_in_doubt_after_a_landed_commit_reports_the_commit`] | in-doubt reconciliation | a lost `EXEC` reply over a commit that DID land must be reported as committed by the reconcile itself, at the sequence and server timestamp the stream actually carries — a blind `Aborted` leaves it to the retry's b.75i commit-token check, which reports it without the external-value releases the reconcile enqueues |
//! | [`layer_in_doubt_with_nothing_dispatched_retries_and_commits_exactly_once`] | in-doubt reconciliation | the opposite keyspace answer: nothing landed, so the retriable verdict is the right one and the retry commits once |
//! | [`layer_in_doubt_without_a_dispatch_leaves_no_open_multi_on_the_txn_slot`] | in-doubt, slot hygiene | the arm must hand its txn slot back free of transaction residue: the next borrower's b.7td checkout probe must not tick `valkey_txn_dirty_checkouts_total` — see the test's own note on which residue that can and cannot see |
//! | [`layer_in_doubt_over_a_foreign_stream_entry_retries_at_the_next_sequence`] | in-doubt reconciliation | the discrimination that makes the arm safe: an entry at `{seq}-0` written by SOMEONE ELSE is not evidence our commit landed |
//! | [`create_in_doubt_after_a_landed_commit_completes_the_create`] | in-doubt reconciliation | the session twin: the member landed and is still unique, so the create must finish (metadata and all) rather than retry into an `AlreadyExists` for the session it just created |
//! | [`create_in_doubt_with_nothing_dispatched_retries_and_creates_once`] | in-doubt reconciliation | nothing landed → retriable → exactly one member |
//! | [`create_in_doubt_when_a_competitor_won_the_name_first_reports_already_exists`] | in-doubt reconciliation | our member did NOT land and the name is taken: `AlreadyExists`, nothing to compensate |
//! | [`create_in_doubt_over_an_unwatched_commit_compensates_and_reports_already_exists`] | in-doubt + compensation | the uniqueness-critical branch: our member landed *unwatched* alongside a competitor's, so reconciliation must re-scan, undo our own commit, and hand the caller the loss |
//! | [`layer_in_doubt_recognises_its_own_commit_when_the_client_timestamp_exceeds_i64_max`] | in-doubt reconciliation | a client-controlled timestamp above `i64::MAX` gets through the in-doubt reconcile with exactly one application of the delta, recognised by the reconcile itself (no `commit_already_landed` tick). `LayerNotification::parse` reads `client_ts` as `u64`, so this does not tell the payload-bytes comparison apart from a parse-and-compare form |
//! | [`layer_in_doubt_recognises_its_own_commit_when_the_client_id_breaks_the_notification_parse`] | in-doubt reconciliation | **the ownership-by-bytes guard (SDD §7.5.2).** A `client_id` containing `\|` — which `LayerStore::update_layer` writes verbatim and `LayerNotification::parse` rejects — is still recognised as ours by the reconcile itself: one application, no `commit_already_landed` tick. A reconcile that re-parsed the entry and compared fields would disown it |
//! | [`a_store_that_refuses_client_id_disarms_both_layers_and_still_commits`] | L2 + L3 degradation | the 100%-outage shape, reproduced for real against a Valkey ACL user denied the `client` command family: `CreateSession` and a structural `UpdateLayer` must still SUCCEED, with `witness_unavailable` + `fingerprint_unavailable` ticking and `aborted_pre_exec` staying at zero |
//! | [`layer_in_doubt_exec_timeout_counts_one_txn_command_timeout`] / [`create_in_doubt_exec_timeout_counts_one_txn_command_timeout`] | in-doubt, timeout count | an in-doubt `EXEC` carrying fred's `Timeout` ticks `valkey_errors_total{pool="txn", kind="command_timeout"}` exactly once — counted where every `InDoubt` is built, never again by the call site — and the retry still commits |
//! | [`layer_in_doubt_exec_timeout_with_a_timed_out_reconcile_read_ticks_txn_and_data`] | in-doubt, timeout count | one stalled `UpdateLayer` ticks twice: the lost `EXEC` reply on txn, and the reconciling `XRANGE` (really stalled behind a server-side `BLPOP`) on data |
//! | `cluster_*` (3) | all three | the session regression, the torn-commit detection, and the `CLIENT ID`-refused degrade again on a real multi-shard cluster, where L2/L3 resolve the watched slot's owner through fred's cached routing view instead of trivially naming the single connection |
//!
//! The metric's pre-initialisation (all ten `{path, disposition}` cells present
//! at zero on a fresh registry) is pinned next to its peers in
//! `tests/integration/observability_metrics_test.rs`. The reply-shape decision
//! table (`Conflict` / `Committed` / `ExecError` / `InDoubt`), the baseline
//! probe's own table (`Trusted` vs. each `Unavailable` reason), and the
//! comparability predicates the pre-dispatch gate composes are pinned as pure
//! unit tests in `src/storage/watched_txn.rs`; this file drives one branch of
//! each per run, which is exactly why the tables need their own coverage.
//!
//! # Three staging mechanisms, and why each is needed
//!
//! The tests above the in-doubt block stage a **real** connection loss —
//! `CLIENT KILL ID` against the exact socket carrying the `WATCH`. That is the
//! only honest way to drive L1/L2/L3, because the whole defect lives in what
//! fred's router does with a command after our code has handed it over.
//!
//! The `InDoubt` arm is not reached that way. A real `InDoubt` comes from a
//! truncated reply set — the socket dying with the pipeline in flight, which
//! the `CLIENT KILL` tests demonstrably do not land in (they reach `Committed`
//! or `ExecError` instead), or the `EXEC` reply outliving the command timeout.
//! A connection stall (`test_support::ConnStall`) can stage the timeout, but
//! only as "the pipeline reached the server and will run once released"; it
//! cannot stage "no reply at all" with the keyspace chosen. The two arms
//! (`reconcile_in_doubt_commit`, `reconcile_in_doubt_create`) are the code that
//! decides whether a commit whose answer was lost actually happened, so leaving
//! them uncovered would leave the exactly-once contract untested. They are
//! therefore driven, deterministically, through
//! `with_watched_exec_override_for_test`, which substitutes the verdict at the
//! one point where everything downstream — the `mark_unwatched` bookkeeping,
//! the guard's armed state, the reconciliation itself — still runs exactly as
//! in production. Two of its variants differ in whether the keyspace was
//! mutated (the question the arms exist to answer); the third,
//! `SkipExecReturnTimedOut`, differs from `SkipExecReturnInDoubt` only in the
//! error's kind, for the timeout-count tests. What the override does NOT
//! simulate is the transport failure; that is the other half of the file's
//! job.
//!
//! The third mechanism is a **Valkey ACL user denied the `client` command
//! family** ([`acl_client_denied_fixture`]), which is what T11 needs and neither
//! of the other two can express. Both identity layers read `CLIENT ID`, and the
//! question there is not "what happens when a connection is lost" but "what
//! happens when the store simply will not answer that command, on every
//! connection, forever" — the proxied-managed-store case (Azure Managed Redis,
//! Redis Enterprise) where a fail-closed reading of either layer is a total
//! outage. `-client` reproduces both refusals at once and for real: the
//! handshake `CLIENT ID` fred swallows into "no id", and the in-`MULTI` one the
//! server rejects at queue time so `EXEC` answers `EXECABORT`. The fixture
//! self-gates and prints a `SKIP:` line on a store without ACL support, and
//! always deletes the user again — including on a panic, via a `Drop` guard.
//!
//! # Backend
//!
//! Every disturbed `UpdateLayer` store runs `lua-watch-structural`
//! ([`WATCH_PATH_BACKEND`]): the scenarios commit structural deltas, which take
//! the WATCH path there, while the default `lua` backend commits them through
//! the structural cache's script, with no WATCH to lose.
//!
//! # Feature gate
//!
//! Every test here uses `with_watch_window_hook_for_test` /
//! `with_pre_dispatch_hook_for_test`, which exist only under
//! `test-fault-injection` — so this whole module is `#[cfg]`'d out without it
//! and the run reports green having covered nothing. That is the silent
//! activation condition (a missing env var prints a `SKIP:` line; a missing
//! feature prints nothing), so any hand-rolled local invocation must pass
//! `--features test-support,test-fault-injection`.
//!
//! # Gating of the cluster trio
//!
//! The three `cluster_` tests self-gate on `LIVE_EDIT_TEST_CLUSTER_URL` and
//! print a `SKIP:` line when it is unset, following the inline-gate idiom in
//! `b_ji8_tests.rs` and `cluster_subscriber_test.rs`. CI's `test-cluster:` job
//! runs them. Neither carries `#[ignore]`: `CLIENT KILL ID` here targets a
//! connection this test's own client owns, so unlike `CLIENT PAUSE` or a slot
//! migration it disturbs no sibling test on the node — the same reasoning under
//! which `cluster_subscriber_test`'s own `kill_reserved_connection` scenarios
//! run in the default job.
//!
//! # Wait budgets vs. nextest's hang-killer
//!
//! `.config/nextest.toml` marks a test slow at 40 s and hard-terminates it at
//! 120 s, and a SIGKILLed test prints nothing — so every diagnostic panic below
//! would be lost exactly when it is needed. Each test holds ONE deadline over
//! its whole body rather than a per-step budget that multiplies:
//!
//! | Test | Waits | Worst case |
//! |---|---|---|
//! | every scenario in this file | 1 × [`SCENARIO_BUDGET`] | **45 s** |
//!
//! [`RECONNECT_BUDGET`] bounds the reconnect poll *inside* a hook and is
//! therefore already covered by the enclosing scenario deadline; it exists so a
//! reconnect that never happens fails with a diagnosis rather than by eating the
//! outer budget silently. The healthy path returns in tens of milliseconds and
//! pays none of it.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use fred::clients::Client;
use fred::interfaces::{
    ClientInterface, ClientLike, ClusterInterface, HashesInterface, KeysInterface, SetsInterface,
    StreamsInterface,
};
use fred::types::client::ClientKillFilter;
use fred::types::config::{Options, Server};
use fred::types::{ClusterHash, CustomCommand, Value};
use futures::FutureExt;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::commit_chain::update_stream_payload;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::notifications::LayerNotification;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::watched_txn::{
    WatchWindowHook, WatchedExecOverride, WatchedExecOverrideQueue,
};
use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets — see the module-level table for the arithmetic
// ═════════════════════════════════════════════════════════════════════════════

/// The backend of every store whose `UpdateLayer` commit a scenario disturbs:
/// `lua-watch-structural`, under which the scenarios' structural deltas take
/// the WATCH/MULTI/EXEC path whose connection identity this file is about.
/// The default `lua` backend commits them through the structural cache's
/// script instead, which holds no WATCH and so never reaches these seams.
const WATCH_PATH_BACKEND: UpdateLayerBackend = UpdateLayerBackend::LuaWatchStructural;

/// One deadline covering a whole scenario body: fixture setup, the staged
/// connection loss, the competing writer, the retry loop that follows, and
/// every assertion read. The healthy path is a few dozen round trips plus one
/// fred reconnect (100 ms exponential floor) and finishes in well under a
/// second; 45 s leaves roughly two orders of magnitude of headroom for CI
/// runner load and keeps the 120 s hang-killer as a backstop rather than the
/// primary failure surface.
pub(crate) const SCENARIO_BUDGET: Duration = Duration::from_secs(45);

/// How long a hook may wait for fred to finish re-establishing the connection
/// it just had killed.
///
/// fred's reconnect policy here is exponential from a 100 ms floor
/// (`ValkeyClient::connect`), so a healthy reconnect completes in a few hundred
/// milliseconds. 10 s is two orders of magnitude above that; reaching it means
/// the client is not reconnecting at all, which is a different failure from the
/// one the enclosing scenario is about and deserves to say so.
const RECONNECT_BUDGET: Duration = Duration::from_secs(10);

/// Poll interval for [`await_reconnect`]. Small enough that the wait costs
/// nothing on the healthy path, large enough not to spin a CI core.
const RECONNECT_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Pool size for every scenario. Two exclusive `txn` slots are the structural
/// minimum — the disturbed writer holds one for the whole hook while the
/// competing writer takes another from inside it — and 4 leaves room for the
/// retry to pick a slot other than the one whose connection was just killed.
pub(crate) const POOL_SIZE: u32 = 4;

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures + helpers
// ═════════════════════════════════════════════════════════════════════════════

/// The cluster gate. `None` means "not configured — the caller must return",
/// and a `SKIP:` line has already been printed saying why.
///
/// `TestValkey::setup_with_url` panics if the cluster is unreachable, so a
/// configured-but-down cluster fails loudly rather than skipping.
async fn cluster_fixture(valkey_pool_size: u32) -> Option<TestValkey> {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); \
             run `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster:` job run \
             this. On a single connection the L2/L3 identity checks name the only connection \
             there is, so the cluster-specific half — resolving the watched slot's owner \
             through fred's cached routing view — is only exercised here"
        );
        return None;
    };
    if !cluster_url.starts_with("redis-cluster://") {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL is not a plaintext redis-cluster:// URL \
             (got a URL with a different scheme)"
        );
        return None;
    }

    let tv = TestValkey::setup_with_url(cluster_url, valkey_pool_size).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (use \
         `redis-cluster://...`). The cluster arm of `WatchFingerprint::take` only runs on a \
         multi-connection client, so a single-node fixture smuggled in through this gate \
         would pass vacuously and report cluster coverage the suite does not have"
    );
    Some(tv)
}

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

// ─────────────────────────────────────────────────────────────────────────────
// The `CLIENT ID`-denied fixture (T11)
// ─────────────────────────────────────────────────────────────────────────────

/// Password for the ACL-restricted fixture. Fixed rather than generated: the
/// user is unique per run and is deleted before the test returns, so the secret
/// carries no meaning — the *username* is what must not collide.
const ACL_FIXTURE_PASSWORD: &str = "bhb-acl-fixture-pw";

/// `ACL SETUSER` arguments for a user that can do everything **except** run the
/// `client` command family.
///
/// The single subtraction is the whole design of the fixture: `~*` `&*`
/// `+@all` leave the user indistinguishable from `default` on every other axis,
/// so a failure in the scenario cannot be blamed on an over-tight ACL. `-client`
/// reproduces, for real, what a proxied managed store does when it refuses to
/// answer `CLIENT ID` — including the two distinct refusals that matter here:
/// the handshake `CLIENT ID` fred issues (swallowed into "no id", per
/// `Connection::cache_connection_id`) and the one queued inside a `MULTI`
/// (rejected at queue time, so `EXEC` answers `EXECABORT`).
fn acl_setuser_args(user: &str) -> Vec<String> {
    vec![
        "SETUSER".to_owned(),
        user.to_owned(),
        "on".to_owned(),
        format!(">{ACL_FIXTURE_PASSWORD}"),
        "~*".to_owned(),
        "&*".to_owned(),
        "+@all".to_owned(),
        "-client".to_owned(),
    ]
}

/// Send one `ACL` subcommand, optionally pinned to a single cluster node.
///
/// Pinning is mandatory on a cluster: ACL users are **per-node** state, not
/// replicated by the cluster bus, so a `SETUSER` that lands on an arbitrary
/// primary leaves the others rejecting the credential and the fixture's client
/// fails to connect for a reason that has nothing to do with the mechanism
/// under test.
async fn acl_command(
    client: &Client,
    node: Option<&Server>,
    args: Vec<String>,
) -> Result<Value, fred::error::Error> {
    let command = CustomCommand::new_static("ACL", None::<u16>, false);
    match node {
        Some(node) => {
            client
                .with_options(&pinned_to(node))
                .custom::<Value, String>(command, args)
                .await
        }
        None => client.custom::<Value, String>(command, args).await,
    }
}

/// Every node the fixture must provision the ACL user on: all primaries of a
/// clustered client, or `[None]` (unpinned) for a centralized one.
fn acl_target_nodes(client: &Client) -> Vec<Option<Server>> {
    if !client.is_clustered() {
        return vec![None];
    }
    client
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .unique_primary_nodes()
        .into_iter()
        .map(Some)
        .collect()
}

/// Deletes the fixture's ACL user on the way out — **including on a panic**,
/// which is the whole reason this is a `Drop` impl rather than a call at the
/// end of the test body. A leaked user would survive on the shared sandbox
/// Valkey and, since it carries a password, would quietly change what a later
/// run's `ACL SETUSER` is doing.
///
/// Holds the *control* fixture (the default-user connection that created the
/// user) so the delete has a connection to run on regardless of what happened
/// to the restricted one.
struct AclUserGuard {
    control: TestValkey,
    nodes: Vec<Option<Server>>,
    user: String,
}

impl Drop for AclUserGuard {
    fn drop(&mut self) {
        let client = self.control.client().pool.next().clone();
        let nodes = self.nodes.clone();
        let user = self.user.clone();
        // Same two guards `TestValkey::drop` takes, for the same reasons: no
        // ambient runtime, or a current-thread one, makes `block_in_place`
        // unusable. Leaking the user is the lesser evil to panicking in a drop.
        let Ok(rt_handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        if rt_handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::MultiThread {
            return;
        }
        tokio::task::block_in_place(|| {
            rt_handle.block_on(async move {
                for node in &nodes {
                    let _ = acl_command(
                        &client,
                        node.as_ref(),
                        vec!["DELUSER".to_owned(), user.clone()],
                    )
                    .await;
                }
            });
        });
    }
}

/// Rewrite `raw` so it authenticates as `user`.
///
/// `None` means the ambient URL cannot carry an injected credential and the
/// caller must SKIP: either the scheme is not the one expected, or the URL
/// already carries credentials — rewriting those would silently change which
/// identity the control half of the fixture runs as, and the fixture's whole
/// claim is that the two halves differ in exactly one respect.
fn credentialed_url(raw: &str, scheme: &str, user: &str) -> Option<String> {
    let rest = raw.strip_prefix(scheme)?;
    if rest.contains('@') {
        return None;
    }
    Some(format!("{scheme}{user}:{ACL_FIXTURE_PASSWORD}@{rest}"))
}

/// A `TestValkey` whose every fred client authenticates as a user the store
/// **denies the `client` command family to**, paired with the guard that
/// deletes that user again.
///
/// Bind the two in this order — `let Some((_guard, tv)) = …` — so `tv` (and its
/// key cleanup) drops FIRST: `ACL DELUSER` disconnects every client
/// authenticated as the user, so deleting it while the fixture is still cleaning
/// up would strand its keys.
///
/// `None` means the environment cannot host the fixture and a `SKIP:` line
/// explaining which part has already been printed. A store that *accepts*
/// `CLIENT ID` from the restricted user is one of those cases and not a
/// failure: it means the ACL was not enforced the way this fixture assumes, so
/// the scenario would pass vacuously.
async fn acl_client_denied_fixture(
    raw_url: String,
    scheme: &str,
    pool_size: u32,
) -> Option<(AclUserGuard, TestValkey)> {
    let user = format!("bhb_noclient_{}", uuid::Uuid::new_v4().simple());
    let Some(url) = credentialed_url(&raw_url, scheme, &user) else {
        eprintln!(
            "SKIP: the configured Valkey URL cannot carry an injected credential (expected a \
             bare `{scheme}host:port` with no `user:pass@`), so this fixture cannot connect as \
             an ACL-restricted user"
        );
        return None;
    };

    let control = TestValkey::setup_with_url(raw_url, 2).await;
    let nodes = acl_target_nodes(control.client().pool.next());
    let mut provision_error = None;
    for node in &nodes {
        if let Err(e) = acl_command(
            control.client().pool.next(),
            node.as_ref(),
            acl_setuser_args(&user),
        )
        .await
        {
            provision_error = Some(e);
            break;
        }
    }

    // Built before the failure check, not after: a `SETUSER` that landed on
    // node 1 and failed on node 2 has still created the user, and the guard is
    // the only thing that removes it.
    let guard = AclUserGuard {
        control,
        nodes,
        user,
    };
    if let Some(e) = provision_error {
        eprintln!(
            "SKIP: `ACL SETUSER` was rejected by this Valkey ({e}) — the store does not support \
             ACL users, or the test credential is not permitted to create one. The unit-level \
             proof of the same decision table lives in `src/storage/watched_txn.rs`'s \
             `baseline_from_probe_exec_decides_every_reply_shape`"
        );
        drop(guard);
        return None;
    }
    // Built AFTER the guard on purpose: if fred cannot complete a handshake
    // against a store that denies `CLIENT ID`, that is a real finding and this
    // panics — but it must not also leak the ACL user while doing so.
    let restricted = TestValkey::setup_with_url(url, pool_size).await;

    // The fixture only means anything if the denial is real. `CLIENT ID` is the
    // exact command both identity layers depend on, so ask for it directly
    // rather than inferring the ACL took effect.
    match restricted
        .client()
        .pool
        .next()
        .custom::<Value, String>(
            CustomCommand::new_static("CLIENT ID", None::<u16>, false),
            vec![],
        )
        .await
    {
        Err(e) => {
            let text = format!("{e}");
            assert!(
                text.contains("NOPERM") || text.contains("no permissions"),
                "`CLIENT ID` must be refused for an ACL reason, not some other failure — a \
                 connection error here would make every assertion below meaningless; got {text}"
            );
        }
        Ok(id) => {
            eprintln!(
                "SKIP: this store answered `CLIENT ID` ({id:?}) for a user denied the `client` \
                 command family, so the ACL is not enforced the way this fixture requires and \
                 the scenario would pass vacuously"
            );
            // Order matters — see this function's doc comment: `ACL DELUSER`
            // disconnects everything authenticated as the user.
            drop(restricted);
            drop(guard);
            return None;
        }
    }

    Some((guard, restricted))
}

/// The smallest object `create_layer` accepts. Contents are irrelevant here —
/// every assertion is about which connection a commit ran on.
fn minimal_object_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 1,
        ..Default::default()
    }
}

/// A delta carrying one `CreateSection`.
///
/// Structural commands are what send `update_layer` down the WATCH/MULTI/EXEC
/// slow path under [`WATCH_PATH_BACKEND`], the rollback for the default `lua`
/// backend's structural route — the dispatch a structural write takes on a
/// replica running that rollback.
fn section_delta(section_id: u64, name: &str) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id,
            section_name: name.to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    }
}

/// Read `live_edit_storage_watch_voided_total{path, disposition}`.
fn read_watch_voided(metrics: &Metrics, path: &str, disposition: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_storage_watch_voided_total",
        &[("path", path), ("disposition", disposition)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_storage_watch_voided_total{{path=\"{path}\", \
             disposition=\"{disposition}\"}} is absent from the exposition. Every cell is \
             pre-registered at zero in `Metrics::new`, so absence means the store never \
             received this metrics handle and every count read from it is vacuous"
        )
    }) as u64
}

/// Read `live_edit_keyspace_type_error_total{backend, site}` (b.459).
///
/// Used here only to prove a **zero**: the pre-`MULTI` type probe passed, so the
/// corruption the `ExecError` arm sees arrived after it. The counter's positive
/// coverage lives in `b_459_tests.rs`.
fn read_keyspace_type_error(metrics: &Metrics, backend: &str, site: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_keyspace_type_error_total",
        &[("backend", backend), ("site", site)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_keyspace_type_error_total{{backend=\"{backend}\", site=\"{site}\"}} is \
             absent from the exposition. Both cells are pre-registered at zero in \
             `Metrics::new`, so absence means the store never received this metrics handle \
             and every count read from it is vacuous"
        )
    }) as u64
}

/// Read `live_edit_update_layer_commit_already_landed_total{backend}` (b.75i).
///
/// Used here only to prove a **zero**: the arm under test reported the landed
/// commit itself, rather than answering `Aborted` and leaving it to the next
/// attempt's commit-token check — which would report the same sequence and so
/// make the stream and header assertions pass either way. The counter's
/// positive coverage lives in `b_75i_tests.rs`.
fn read_commit_already_landed(metrics: &Metrics, backend: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_commit_already_landed_total",
        &[("backend", backend)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_update_layer_commit_already_landed_total{{backend=\"{backend}\"}} is \
             absent from the exposition. Both cells are pre-registered at zero in \
             `Metrics::new`, so absence means the store never received this metrics handle \
             and the zero this test asserts is vacuous"
        )
    }) as u64
}

/// Every `reason` cell of `live_edit_valkey_txn_dirty_checkouts_total` (b.7td;
/// `topology_reply` added by b.289 sub-task 3).
///
/// Restated rather than imported — `TXN_DIRTY_REASONS` is `pub(crate)` in
/// `observability::metrics` and an integration test is a separate crate — which
/// is the same standing (and the same deliberate duplication) as
/// `txn_checkout_test.rs`'s own copy.
const TXN_DIRTY_REASONS: [&str; 5] = [
    "leaked_multi",
    "unexpected_reply",
    "probe_timeout",
    "probe_error",
    "topology_reply",
];

/// `live_edit_valkey_txn_dirty_checkouts_total`, every `reason` cell at once.
///
/// Read as a whole rather than one cell: the interesting assertion is "the
/// checkout probe found nothing wrong", and a verdict filing itself under a
/// sibling reason is exactly what a single-cell read would miss.
fn read_txn_dirty_checkouts(metrics: &Metrics) -> Vec<(&'static str, u64)> {
    let exposition = render(metrics);
    TXN_DIRTY_REASONS
        .iter()
        .map(|reason| {
            let value = sample_value(
                &exposition,
                "live_edit_valkey_txn_dirty_checkouts_total",
                &[("reason", *reason)],
            )
            .unwrap_or_else(|| {
                panic!(
                    "live_edit_valkey_txn_dirty_checkouts_total{{reason=\"{reason}\"}} is absent \
                     from the exposition. Every cell is pre-registered at zero in `Metrics::new`, \
                     so absence means this is not the registry the client reports into"
                )
            });
            (*reason, value as u64)
        })
        .collect()
}

/// Read `live_edit_valkey_pool_in_use{pool}`.
fn read_pool_in_use(metrics: &Metrics, pool: &str) -> i64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_pool_in_use",
        &[("pool", pool)],
    )
    .unwrap_or(0.0) as i64
}

/// `Options` pinning a command to one cluster node.
fn pinned_to(node: &Server) -> Options {
    Options {
        cluster_node: Some(node.clone()),
        max_attempts: Some(1),
        ..Default::default()
    }
}

/// Name the connection the `WATCH` on `watched_key` is riding: `(server, id)`.
///
/// Deliberately derived the same way `WatchFingerprint::take` derives it —
/// fred's own connection map, and on a cluster fred's own cached routing view —
/// so the socket this file kills is by construction the socket production
/// believes it is watching on. Re-deriving slot ownership independently would
/// let the two drift and still agree.
pub(crate) fn guarded_connection(client: &Client, watched_key: &str) -> (Server, i64) {
    let ids = client.connection_ids();
    assert!(
        !ids.is_empty(),
        "the guarded client must have at least one live connection — a WATCH cannot outlive \
         its connection, so an empty map means the disturbance below would land nowhere"
    );

    if !client.is_clustered() {
        assert_eq!(
            ids.len(),
            1,
            "a centralized client has exactly one connection; {ids:?} means the fixture is \
             not the one this helper assumes"
        );
        let (server, id) = ids.into_iter().next().expect("checked non-empty");
        return (server, id);
    }

    let ClusterHash::Custom(slot) = ClusterHash::from(watched_key) else {
        panic!("`ClusterHash::from(&str)` must yield `Custom(slot)`; the WATCH sites rely on it")
    };
    let server = client
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .get_server(slot)
        .unwrap_or_else(|| panic!("no primary owns slot {slot} in the cached routing table"))
        .clone();
    let id = *ids.get(&server).unwrap_or_else(|| {
        panic!("fred has no connection id recorded for {server}, the owner of slot {slot}")
    });
    (server, id)
}

/// `CLIENT KILL ID <id>` on `node`, issued from the *data* pool — a real,
/// server-side close of exactly the socket carrying the `WATCH`, from a
/// connection that is not the one being killed.
pub(crate) async fn kill_connection(killer: &Client, node: &Server, conn_id: i64) {
    let filter = vec![ClientKillFilter::ID(conn_id.to_string())];
    // `cluster_node` is meaningless on a centralized client; branching keeps
    // the same helper honest on both fixtures.
    let killed: Result<u64, _> = if killer.is_clustered() {
        killer
            .with_options(&pinned_to(node))
            .client_kill(filter)
            .await
    } else {
        killer.client_kill(filter).await
    };
    let killed =
        killed.unwrap_or_else(|e| panic!("CLIENT KILL ID {conn_id} on {node} must succeed: {e}"));
    assert_eq!(
        killed, 1,
        "CLIENT KILL must have closed exactly the one connection under test (id={conn_id} on \
         {node}); a 0 means the WATCH's connection had already gone away and this scenario \
         never staged the disturbance it asserts about"
    );
}

/// Block until fred has finished re-establishing `node`'s connection.
///
/// Turns "the connection was killed" into "the connection has been REPLACED",
/// which is what puts a *completed* reconnect inside the residual window the
/// pre-dispatch check cannot cover — the only way to reach the L3 witness
/// deterministically instead of by racing a real failover.
pub(crate) async fn await_reconnect(client: &Client, node: &Server, old_id: i64) {
    let deadline = Instant::now() + RECONNECT_BUDGET;
    loop {
        if client
            .connection_ids()
            .get(node)
            .is_some_and(|id| *id != old_id)
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "fred did not re-establish its connection to {node} within {RECONNECT_BUDGET:?} \
             after CLIENT KILL ID {old_id}. Without a completed reconnect the commit below \
             would fail to write rather than landing unwatched, so the scenario would assert \
             a different mechanism than the one it names"
        );
        tokio::time::sleep(RECONNECT_POLL_INTERVAL).await;
    }
}

/// Wrap a hook so its body runs on the FIRST invocation only.
///
/// Load-bearing, not a convenience: every scenario here ends inside the call
/// site's own `Aborted` retry loop, and a hook that re-staged its disturbance
/// on the retry would kill the fresh connection too — the attempt could never
/// converge and the test would spin to its deadline instead of asserting an
/// outcome.
pub(crate) fn once_only(body: WatchWindowHook) -> WatchWindowHook {
    let fired = Arc::new(AtomicBool::new(false));
    Arc::new(move |client| {
        let fired = fired.clone();
        let body = body.clone();
        async move {
            if fired.swap(true, Ordering::SeqCst) {
                return;
            }
            body(client).await;
        }
        .boxed()
    })
}

/// Every member of `le:path:{hex(base_path)}:names` whose name half is `name`.
async fn name_members(tv: &TestValkey, base_path: &str, name: &str) -> Vec<String> {
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().path_names_index(base_path))
        .await
        .expect("SMEMBERS on the names index must succeed");
    members
        .into_iter()
        .filter(|m| m.split('\u{0}').next() == Some(name))
        .collect()
}

/// The per-object stream as `(id, payload)`, in order. Fields are read as
/// bytes: an entry's relay `ompf` field is binary.
async fn stream_payloads(tv: &TestValkey, object_id: u64) -> Vec<(String, String)> {
    let entries: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(tv.key_schema().layer_stream(object_id), "-", "+", None)
        .await
        .expect("XRANGE over the layer stream must succeed");
    entries
        .into_iter()
        .map(|(id, fields)| {
            let payload = fields
                .get("payload")
                .map(|p| String::from_utf8(p.to_vec()).expect("a stream payload is UTF-8"))
                .unwrap_or_default();
            (id, payload)
        })
        .collect()
}

/// Assert the stream carries exactly one entry at `{seq}-0` and that its
/// payload names `client_id` — i.e. that commit, and no other, took that
/// sequence.
fn assert_stream_entry(entries: &[(String, String)], seq: u64, client_id: &str) {
    let want = format!("{seq}-0");
    let matches: Vec<&(String, String)> = entries.iter().filter(|(id, _)| *id == want).collect();
    assert_eq!(
        matches.len(),
        1,
        "the stream must carry exactly one entry at {want}; got {entries:?}"
    );
    // Payload format: update|{oid}|{seq}|{client_id}|{client_ts}|{server_ts}
    let parts: Vec<&str> = matches[0].1.split('|').collect();
    assert_eq!(
        parts.first(),
        Some(&"update"),
        "entry {want} must be an update notification; got {:?}",
        matches[0].1
    );
    assert_eq!(
        parts.get(3),
        Some(&client_id),
        "entry {want} must name the writer that committed it — a different client id here \
         means this test attributed the commit to the wrong writer and every conclusion it \
         draws about the race is unsound; got {:?}",
        matches[0].1
    );
}

/// Wrap a hook so its body runs on the FIRST invocation only, while `counter`
/// records EVERY invocation.
///
/// One `create_once` / `update_layer_once_watch` attempt calls its hook exactly once,
/// so the counter is an attempt counter — which is how a test tells "the
/// disturbance was absorbed and the call retried" apart from "the call sailed
/// through untouched and this test proved nothing".
fn counting_once(counter: Arc<AtomicUsize>, body: WatchWindowHook) -> WatchWindowHook {
    let fired = Arc::new(AtomicBool::new(false));
    Arc::new(move |client| {
        let fired = fired.clone();
        let counter = counter.clone();
        let body = body.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            if fired.swap(true, Ordering::SeqCst) {
                return;
            }
            body(client).await;
        }
        .boxed()
    })
}

/// A FIFO override queue for `with_watched_exec_override_for_test`. One entry is
/// consumed per watched commit attempt that reaches dispatch; once it drains,
/// later attempts run the real pipeline and decode the real verdict — which is
/// what lets a single queued entry disturb attempt 1 and leave the retry clean.
fn override_queue(items: &[WatchedExecOverride]) -> WatchedExecOverrideQueue {
    Arc::new(Mutex::new(items.iter().copied().collect::<VecDeque<_>>()))
}

/// Asserts the queue handed to `with_watched_exec_override_for_test` actually
/// drained — i.e. the disturbance these tests are built around was really
/// injected.
///
/// This matters because several of these tests assert exactly the outcome an
/// *undisturbed* commit produces (the whole point: reconciliation must make the
/// disturbed path indistinguishable from the clean one). That makes them
/// vacuous if the seam silently stops firing — a refactor that moves the commit
/// off the watched-dispatch path, or renames the seam into a no-op, would leave
/// every assertion green while testing nothing. The production seam pops one
/// entry per dispatch (`q.lock().unwrap().pop_front()` in `LayerStore` /
/// `SessionStore`), so "queue is empty afterwards" is an exact statement that
/// every queued override was consumed, not an approximation.
fn assert_override_consumed(queue: &WatchedExecOverrideQueue) {
    let remaining = queue.lock().unwrap();
    assert!(
        remaining.is_empty(),
        "every queued watched-exec override must have been consumed by a commit attempt; \
         {} entr{} left unpopped ({remaining:?}). An undrained queue means the seam never \
         fired, so the disturbance under test never happened and the assertions above passed \
         against an ordinary, uninterrupted commit",
        remaining.len(),
        if remaining.len() == 1 { "y" } else { "ies" },
    );
}

/// The `server_ts` field of an `update|{oid}|{seq}|{client}|{client_ts}|{server_ts}`
/// stream payload.
fn payload_server_ts(payload: &str) -> u64 {
    payload
        .split('|')
        .nth(5)
        .unwrap_or_else(|| panic!("update payload must carry six fields; got {payload:?}"))
        .parse()
        .unwrap_or_else(|e| panic!("server_ts field of {payload:?} must parse as u64: {e}"))
}

/// The `seq` field of an object header.
async fn header_seq(tv: &TestValkey, object_id: u64) -> u64 {
    let seq: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().obj_header(&object_id.to_string()), "seq")
        .await
        .expect("HGET obj header seq must succeed");
    seq.unwrap_or_else(|| panic!("object {object_id} has no header seq"))
        .parse()
        .expect("header seq must be numeric")
}

/// `le:{session_id}:sess` as a map.
async fn session_meta(tv: &TestValkey, session_id: &str) -> HashMap<String, String> {
    tv.client()
        .pool
        .hgetall(tv.key_schema().session_meta(session_id))
        .await
        .expect("HGETALL session meta must succeed")
}

// ═════════════════════════════════════════════════════════════════════════════
// Scenario bodies — shared by the single-node tests and their cluster twins
// ═════════════════════════════════════════════════════════════════════════════

/// Race a second `create` into the window between the duplicate scan and the
/// `EXEC`, with the scanning connection already replaced.
///
/// This is the session half of the b.bhb regression. The `WATCH` is server-side
/// gone (its connection was killed and fred has finished replacing it), so the
/// `EXEC` commits unconditionally on a socket that never saw the `WATCH` and the
/// duplicate-name scan that ran a moment earlier fenced nothing.
///
/// **Pre-fix this leaves two members carrying one name** and returns `Ok` to
/// both callers — the silent uniqueness loss the ticket was filed on. There is
/// no error anywhere to observe it by, which is why the invariant (member
/// count), not the error shape, is what this asserts.
async fn scenario_create_session_witness(tv: &TestValkey) {
    const NAME: &str = "bhb-session";
    const BASE_PATH: &str = "/bhb-stage";

    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let competitor = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let killer = tv.client().pool.next().clone();
    let names_key = tv.key_schema().path_names_index(BASE_PATH);

    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let competitor = competitor.clone();
        let killer = killer.clone();
        let names_key = names_key.clone();
        async move {
            let (node, conn_id) = guarded_connection(client, &names_key);
            kill_connection(&killer, &node, conn_id).await;
            // A *completed* reconnect is what makes this the L3 case rather
            // than the in-flight-death case: the commit below is written
            // successfully, onto a socket that never held the WATCH.
            await_reconnect(client, &node, conn_id).await;
            competitor
                .create(NAME, BASE_PATH, "bhb-competitor")
                .await
                .expect("the competing create must succeed — it is uncontended");
        }
        .boxed()
    }));

    let disturbed = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_pre_dispatch_hook_for_test(hook);

    let result = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.create(NAME, BASE_PATH, "bhb-disturbed"),
    )
    .await
    .expect("the disturbed create did not settle within the scenario budget");

    // ── The invariant. ──
    let holders = name_members(tv, BASE_PATH, NAME).await;
    assert_eq!(
        holders.len(),
        1,
        "exactly one session may hold the name {NAME:?} at {BASE_PATH:?}. Two members is the \
         b.bhb defect itself: the disturbed create's EXEC ran on a connection that never held \
         the WATCH, so the duplicate scan it had just completed fenced nothing. Got: {holders:?}"
    );
    match result {
        Err(ServiceError::AlreadyExists(_)) => {}
        other => panic!(
            "the disturbed create raced a competitor for the same name and must lose: its \
             commit was undone and the retry re-ran the duplicate scan under a live WATCH. \
             `Ok` here means it kept a session whose uniqueness was never checked. Got: {other:?}"
        ),
    }

    // ── The proof, as the operator sees it. ──
    assert_eq!(
        read_watch_voided(&metrics, "create_session", "compensated"),
        1,
        "the in-transaction CLIENT ID witness must have caught the unwatched commit exactly \
         once and compensated it. 0 means the witness never fired — the mechanism under test \
         did not run and the single surviving member above is luck"
    );
    assert_eq!(
        read_watch_voided(&metrics, "create_session", "committed_unwatched"),
        0,
        "compensation succeeded, so the alertable residual cell must stay clean — it means \
         `the duplicate check did not run and could not be undone`, which is a different \
         (and worse) outcome than the one this scenario stages"
    );
}

/// Kill the `WATCH`'s connection with the commit already queued and a competing
/// writer holding the sequence the disturbed writer computed.
///
/// This is the layer path's **torn commit**, and the scenario is the residual
/// the fix documents rather than removes: the commit is queued, the `WATCH`'s
/// connection goes away, and the pipeline is written onto the replacement — so
/// the `EXEC` runs with no watch set, the `HSET`s land, and the `XADD {seq}-0`
/// is refused by the competitor's entry. Materialized state and the live relay
/// stream now disagree about what the layer contains. Only the Lua structural
/// commit (b.289) can remove that; b.bhb's obligation is that it can never
/// happen *silently*.
///
/// So what this pins is detection, and it is a real regression guard: pre-fix
/// the identical tear surfaced as a bare `INTERNAL` with nothing anywhere
/// saying an optimistic-concurrency check had been skipped — no metric, no
/// `error!`, and therefore no way for an operator to tell this apart from a
/// keyspace anomaly. Post-fix it ticks the alertable `committed_unwatched`
/// cell, and the caller still gets a non-retriable error rather than an ack for
/// a commit whose stream entry does not exist.
///
/// Reaching an error frame at `EXEC` is itself the evidence: this path watches
/// the stream key, so under a LIVE watch the competitor's entry would have
/// cancelled the transaction instead. An `Ok` here would mean the tear was
/// acked — directly, or by a retry whose b.75i commit-token check found the
/// record the torn `MULTI` wrote alongside its other `HSET`s.
///
/// **Why the hook waits for the reconnect.** The kill alone does not decide
/// which of two mechanisms the commit meets: whether the pipeline is written to
/// the replacement socket (an unwatched `EXEC` → `ExecError` → the torn commit
/// this scenario is named for) or fails to route at all (`InDoubt` →
/// reconciliation reads the competitor's foreign `2-0`, answers `Aborted`, and
/// the retry commits cleanly at 3) depends on whether fred's router has
/// dequeued its deferred `Reconnect` before our dispatch enqueues — an ordering
/// this test does not control. Awaiting the replacement forces the first, and
/// changes nothing else: L2's pre-dispatch check has already passed by the time
/// the hook runs, so the commit still goes out unwatched and still collides
/// with the competitor at `{2}-0`.
async fn scenario_update_layer_torn_commit_detected(tv: &TestValkey) {
    const LAYER: &str = "bhb-layer";
    const COMPETITOR_CLIENT: &str = "bhb-competitor";
    const DISTURBED_CLIENT: &str = "bhb-disturbed";

    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(metrics.clone());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bhb-session", "/bhb-stage", "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = plain
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let competitor = Arc::new(plain.clone());
    let killer = tv.client().pool.next().clone();
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let competitor_session = session_id.clone();

    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let competitor = competitor.clone();
        let killer = killer.clone();
        let header_key = header_key.clone();
        let competitor_session = competitor_session.clone();
        async move {
            // Order is the whole scenario: the competitor takes sequence 2
            // FIRST — so the disturbed writer's already-queued `XADD 2-0` is
            // now a collision — and only then is the WATCH's connection
            // killed, leaving the queued commit to be written with no watch
            // set behind it.
            let (seq, _) = competitor
                .update_layer_relayed(
                    &competitor_session,
                    LAYER,
                    object_id,
                    Some(section_delta(1, "CompetitorSection")),
                    COMPETITOR_CLIENT,
                    1_000,
                )
                .await
                .expect("the competing structural update must commit — it is uncontended");
            assert_eq!(
                seq, 2,
                "the competitor must take the sequence the disturbed writer already computed \
                 for itself; without that collision the unwatched EXEC below would simply \
                 succeed and this scenario would be asserting the wrong disposition"
            );
            let (node, conn_id) = guarded_connection(client, &header_key);
            kill_connection(&killer, &node, conn_id).await;
            // Wait for the REPLACEMENT socket before returning, for the same
            // reason the session scenario does — see the note above the
            // function on why the outcome is otherwise a router race.
            await_reconnect(client, &node, conn_id).await;
        }
        .boxed()
    }));

    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone())
        .with_pre_dispatch_hook_for_test(hook);

    let result = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(2, "DisturbedSection")),
            DISTURBED_CLIENT,
            2_000,
        ),
    )
    .await
    .expect("the disturbed update did not settle within the scenario budget");

    // ── The caller's verdict. ──
    match result {
        Err(ServiceError::Internal(msg)) => assert!(
            msg.contains("EXEC"),
            "the tear must be reported as an EXEC failure so the operator sees where it \
             happened; got {msg:?}"
        ),
        Ok((seq, _)) => panic!(
            "the disturbed commit must NOT be acked: its XADD was refused, so there is no \
             stream entry at sequence {seq} for a subscriber to relay, and reporting success \
             tells the client a write landed that the live feed will never carry"
        ),
        Err(ServiceError::Aborted(msg)) => panic!(
            "the tear must be non-retriable: the HSETs already landed — the b.75i commit-token \
             record among them — so a retry would find that record and ack this torn commit, \
             a sequence with no stream entry. Got: {msg:?}"
        ),
        Err(other) => panic!("unexpected classification for a torn commit: {other:?}"),
    }

    // ── The detection, which is the whole deliverable for this residual. ──
    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "committed_unwatched"),
        1,
        "an EXEC that ran and returned an error frame is only reachable with a void WATCH — \
         under a live one the competitor's stream entry would have cancelled it — so it must \
         tick the alertable cell exactly once. 0 is the pre-b.bhb state: the same tear, \
         reported as a bare INTERNAL, indistinguishable from a keyspace anomaly"
    );

    // ── The live feed is not corrupted by the tear. ──
    let entries = stream_payloads(tv, object_id).await;
    assert_stream_entry(&entries, 2, COMPETITOR_CLIENT);
    assert!(
        entries.iter().all(|(id, _)| id != "3-0"),
        "the disturbed writer's XADD was refused, so nothing may sit at 3-0; got {entries:?}"
    );
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    assert!(
        snapshot.sections.contains_key(&1),
        "the competitor's committed section must survive the disturbed writer's failure — \
         its own commit is unaffected by another writer's void WATCH; got {:?}",
        snapshot.sections.keys().collect::<HashSet<_>>()
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 — the two regressions (single node)
// ═════════════════════════════════════════════════════════════════════════════

/// b.bhb: a `CreateSession` commit dispatched after its `WATCH`'s connection was
/// replaced must not keep a name it never fenced.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_commit_dispatched_after_a_reconnect_keeps_the_name_unique() {
    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    scenario_create_session_witness(&tv).await;
}

/// b.bhb: an `UpdateLayer` commit that ran without its `WATCH` and tore must be
/// counted, not just failed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_torn_commit_after_a_void_watch_is_detected_and_reported_non_retriable() {
    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    scenario_update_layer_torn_commit_detected(&tv).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T2 — the cost of L1: a connection lost mid-window must still resolve
// ═════════════════════════════════════════════════════════════════════════════

/// b.bhb non-regression: the reads taken *under* a `WATCH` lost fred's
/// transparent retry along with the commit, and must still reach a verdict.
///
/// The connection carrying the `WATCH` dies before the duplicate-name scan has
/// run, and a competitor claims the name while it is down. The scan then runs
/// on the replacement connection with the `WATCH` already void — which is
/// harmless *here* precisely because the scan is a read: it sees the
/// competitor's member and the caller loses the name honestly.
///
/// The invariant is the same one the regression above pins from the other side
/// (one name, one holder); what this adds is that capping the txn pool at one
/// attempt did not turn a routine reconnect on this path into a hard failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_racing_create_inside_a_dead_watch_window_keeps_the_name_unique() {
    const NAME: &str = "bhb-window-session";
    const BASE_PATH: &str = "/bhb-window-stage";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let competitor = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let killer = tv.client().pool.next().clone();
    let names_key = tv.key_schema().path_names_index(BASE_PATH);

    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let competitor = competitor.clone();
        let killer = killer.clone();
        let names_key = names_key.clone();
        async move {
            let (node, conn_id) = guarded_connection(client, &names_key);
            kill_connection(&killer, &node, conn_id).await;
            // Waiting for the replacement is what makes the scan below run at
            // all: without it the scan is issued into a dead socket and this
            // test would be measuring the error path instead of the read.
            await_reconnect(client, &node, conn_id).await;
            competitor
                .create(NAME, BASE_PATH, "bhb-competitor")
                .await
                .expect("the competing create must succeed — it is uncontended");
        }
        .boxed()
    }));

    let disturbed = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_watch_window_hook_for_test(hook);

    let result = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.create(NAME, BASE_PATH, "bhb-disturbed"),
    )
    .await
    .expect("the disturbed create did not settle within the scenario budget");

    assert!(
        matches!(result, Err(ServiceError::AlreadyExists(_))),
        "the scan ran after the competitor committed, so the name is taken and the caller \
         must be told so. `Ok` would mean a second holder of the name was written; a \
         transport error would mean the attempt cap turned a routine reconnect into a \
         failure the caller cannot act on. Got: {result:?}"
    );
    let holders = name_members(&tv, BASE_PATH, NAME).await;
    assert_eq!(
        holders.len(),
        1,
        "exactly one session may hold {NAME:?} at {BASE_PATH:?}; got {holders:?}"
    );
}

/// b.bhb non-regression, layer path: a connection lost inside the `WATCH`
/// window, with a competing structural writer landing while it is down, must
/// still converge on two commits at consecutive sequences.
///
/// The disturbed writer has not yet read the header when its connection dies,
/// so whichever way the race falls — the read fails and re-enters the retry
/// loop, or it succeeds on the replacement connection and the pre-dispatch
/// identity check aborts it there — the attempt re-runs against fresh state.
/// Both routes must end in a clean commit at the next sequence, with the
/// competitor's write intact.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_racing_writer_inside_a_dead_watch_window_commits_at_the_next_sequence() {
    const LAYER: &str = "bhb-window-layer";
    const COMPETITOR_CLIENT: &str = "bhb-window-competitor";
    const DISTURBED_CLIENT: &str = "bhb-window-disturbed";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bhb-window-session", "/bhb-window-stage2", "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = plain
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let competitor = Arc::new(plain.clone());
    let killer = tv.client().pool.next().clone();
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let competitor_session = session_id.clone();

    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let competitor = competitor.clone();
        let killer = killer.clone();
        let header_key = header_key.clone();
        let competitor_session = competitor_session.clone();
        async move {
            let (node, conn_id) = guarded_connection(client, &header_key);
            kill_connection(&killer, &node, conn_id).await;
            competitor
                .update_layer_relayed(
                    &competitor_session,
                    LAYER,
                    object_id,
                    Some(section_delta(1, "CompetitorSection")),
                    COMPETITOR_CLIENT,
                    1_000,
                )
                .await
                .expect("the competing structural update must commit — it is uncontended");
        }
        .boxed()
    }));

    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_watch_window_hook_for_test(hook);

    let (seq, _) = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(2, "DisturbedSection")),
            DISTURBED_CLIENT,
            2_000,
        ),
    )
    .await
    .expect("the disturbed update did not settle within the scenario budget")
    .expect(
        "a connection lost inside the WATCH window is retriable: the attempt must re-run \
         against fresh state and commit. A terminal error here means the loss escaped the \
         Aborted classification the retry loop is driven by",
    );

    assert_eq!(
        seq, 3,
        "the competitor took sequence 2 while the connection was down, so the disturbed \
         writer must commit at 3 — re-applying at 2 would be the lost update"
    );
    let entries = stream_payloads(&tv, object_id).await;
    assert_stream_entry(&entries, 2, COMPETITOR_CLIENT);
    assert_stream_entry(&entries, 3, DISTURBED_CLIENT);

    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let sections: HashSet<u64> = snapshot.sections.keys().copied().collect();
    assert_eq!(
        sections,
        HashSet::from([1u64, 2u64]),
        "both writers' sections must be materialized; got {sections:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T3 — L2: the pre-dispatch connection-identity check
// ═════════════════════════════════════════════════════════════════════════════

/// b.bhb: `UpdateLayer` must notice, before it dispatches, that the connection
/// carrying its `WATCH` has been replaced — and abort into its own retry rather
/// than send a commit that would run unconditionally.
///
/// This is the cheap layer, and the one that covers the common case: the whole
/// check is a local map read (no round trip), and the window it closes is
/// everything from the `WATCH` up to the moment the pipeline is handed to the
/// router. Nothing is written, so the retry re-reads the header under a fresh
/// `WATCH` and commits normally — the caller never sees the disturbance at all.
///
/// The `aborted_pre_exec` assertion is what keeps this test from being a
/// tautology: an ordinary uncontended commit also returns `Ok` at sequence 2,
/// so without the counter this would pass identically with the check deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_connection_replaced_before_dispatch_aborts_and_retries_cleanly() {
    const LAYER: &str = "bhb-l2-layer";
    const CLIENT: &str = "bhb-l2-writer";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(metrics.clone());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bhb-l2-session", "/bhb-l2-stage", "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = plain
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let killer = tv.client().pool.next().clone();
    let header_key = tv.key_schema().obj_header(&object_id.to_string());

    // The fingerprint is taken immediately after `mark_watched`, i.e. BEFORE
    // this hook runs — so a completed reconnect inside the hook is exactly the
    // condition the pre-dispatch comparison exists to catch.
    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let killer = killer.clone();
        let header_key = header_key.clone();
        async move {
            let (node, conn_id) = guarded_connection(client, &header_key);
            kill_connection(&killer, &node, conn_id).await;
            await_reconnect(client, &node, conn_id).await;
        }
        .boxed()
    }));

    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone())
        .with_watch_window_hook_for_test(hook);

    let (seq, _) = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(1, "L2Section")),
            CLIENT,
            1_000,
        ),
    )
    .await
    .expect("the disturbed update did not settle within the scenario budget")
    .expect(
        "an aborted-before-dispatch attempt is retriable and uncontended, so the retry must \
         commit. An error escaping here means the abort was classified as terminal",
    );

    assert_eq!(seq, 2, "the retry commits the delta exactly once");
    assert_stream_entry(&stream_payloads(&tv, object_id).await, 2, CLIENT);

    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "aborted_pre_exec"),
        1,
        "the pre-dispatch check must have caught the replaced connection exactly once. 0 \
         means the commit was dispatched under a WATCH that no longer existed and only \
         happened to be uncontended"
    );
    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "committed_unwatched"),
        0,
        "nothing was dispatched, so nothing may be reported as having committed unwatched"
    );
}

/// b.bhb: the same pre-dispatch check on `CreateSession`, which reports through
/// a different metrics handle (`ValkeyClient::metrics_snapshot`) than the layer
/// path (`LayerStore::with_metrics`) and therefore cannot be assumed wired by
/// its sibling above.
///
/// Doubles as the availability half of the L1 cost: a `create` whose connection
/// is replaced mid-window must still succeed, not surface a transport error to
/// a caller who did nothing wrong.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_connection_replaced_before_dispatch_aborts_and_retries_cleanly() {
    const NAME: &str = "bhb-l2-session";
    const BASE_PATH: &str = "/bhb-l2-session-stage";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let killer = tv.client().pool.next().clone();
    let names_key = tv.key_schema().path_names_index(BASE_PATH);

    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let killer = killer.clone();
        let names_key = names_key.clone();
        async move {
            let (node, conn_id) = guarded_connection(client, &names_key);
            kill_connection(&killer, &node, conn_id).await;
            await_reconnect(client, &node, conn_id).await;
        }
        .boxed()
    }));

    let disturbed = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_watch_window_hook_for_test(hook);

    let session_id = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.create(NAME, BASE_PATH, "bhb-user"),
    )
    .await
    .expect("the disturbed create did not settle within the scenario budget")
    .expect(
        "a connection replaced inside the WATCH window is retriable and this create is \
         uncontended, so it must succeed. An error here would mean a routine reconnect on \
         the txn pool now fails a CreateSession the caller can do nothing about",
    );
    assert!(!session_id.is_empty(), "a created session must have an id");

    let holders = name_members(&tv, BASE_PATH, NAME).await;
    assert_eq!(
        holders.len(),
        1,
        "the retry must not leave a second member behind from the aborted attempt — nothing \
         was dispatched, so there is nothing to clean up; got {holders:?}"
    );
    assert_eq!(
        read_watch_voided(&metrics, "create_session", "aborted_pre_exec"),
        1,
        "the pre-dispatch check must have caught the replaced connection exactly once. 0 \
         means an uncontended create simply succeeded and this test proved nothing"
    );
    assert_eq!(
        read_watch_voided(&metrics, "create_session", "compensated"),
        0,
        "nothing was dispatched, so there is nothing to compensate"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T4 — L3: the in-transaction witness and its two layer dispositions
// ═════════════════════════════════════════════════════════════════════════════

/// b.bhb: a proven-unwatched `UpdateLayer` commit on a still-live object is
/// reported as committed — and counted.
///
/// The `EXEC` array came back all-OK, which is itself the evidence that matters:
/// the `XADD {seq}-0` was accepted, so no other writer took that sequence and
/// the lost update the void `WATCH` licensed did not happen. The commit is real
/// and the caller must be told so here; returning `Aborted` would send it back
/// through the retry loop, where the next attempt's b.75i commit-token check
/// would rediscover it — but without the external-value releases only this
/// attempt holds. `commit_already_landed_total` staying at zero is what proves
/// this arm, and not that backstop, reported the commit.
///
/// What it must NOT do is stay silent. `committed_unwatched` is the alertable
/// cell, and this is the shape that ticks it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_witness_mismatch_on_a_live_object_accepts_the_commit() {
    const LAYER: &str = "bhb-witness-layer";
    const CLIENT: &str = "bhb-witness-writer";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(metrics.clone());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bhb-witness-session", "/bhb-witness-stage", "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = plain
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let killer = tv.client().pool.next().clone();
    let header_key = tv.key_schema().obj_header(&object_id.to_string());

    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let killer = killer.clone();
        let header_key = header_key.clone();
        async move {
            let (node, conn_id) = guarded_connection(client, &header_key);
            kill_connection(&killer, &node, conn_id).await;
            await_reconnect(client, &node, conn_id).await;
        }
        .boxed()
    }));

    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone())
        .with_pre_dispatch_hook_for_test(hook);

    let (seq, _) = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(1, "WitnessSection")),
            CLIENT,
            1_000,
        ),
    )
    .await
    .expect("the disturbed update did not settle within the scenario budget")
    .expect(
        "the commit landed on the replacement connection and nothing contended for its \
         sequence, so it must be reported as committed — an error would tell the client its \
         write was lost when the stream says otherwise",
    );

    assert_eq!(seq, 2, "the commit took the sequence it computed");
    assert_stream_entry(&stream_payloads(&tv, object_id).await, 2, CLIENT);

    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "committed_unwatched"),
        1,
        "a commit the server confirms ran on a connection that never held its WATCH must be \
         visible exactly once on the alertable cell. 0 means the witness never fired and this \
         test proved only that an ordinary commit works"
    );
    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "aborted_pre_exec"),
        0,
        "the pre-dispatch check passed by construction (the hook runs after it), so the \
         caught-window cell must stay clean — a tick would mean the disturbance landed \
         somewhere other than the residual window this test names"
    );
    assert_eq!(
        read_commit_already_landed(&metrics, "txn"),
        0,
        "the witness arm itself must report the commit. A tick means it answered `Aborted` and \
         the retry's commit-token check reported the commit instead — same sequence, but the \
         external-value releases this attempt owed were dropped"
    );
}

/// b.bhb: the same proof on an object retired underneath the commit keeps the
/// ordinary retirement classification.
///
/// The witness says the transaction ran without its `WATCH`, so the object's
/// liveness was never actually fenced. Re-running the retirement read set is
/// what turns that into an answer: the header now carries `superseded=1`, so the
/// caller gets the same `FailedPrecondition` any ordinary superseded write
/// would, rather than an `Ok` for a commit whose keys are orphans on a doomed
/// object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_witness_mismatch_on_a_retired_object_surfaces_failed_precondition() {
    const LAYER: &str = "bhb-retired-layer";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(metrics.clone());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bhb-retired-session", "/bhb-retired-stage", "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = plain
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let killer = tv.client().pool.next().clone();
    let data_pool = tv.client().pool.next().clone();
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let header_key_for_hook = header_key.clone();

    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let killer = killer.clone();
        let data_pool = data_pool.clone();
        let header_key = header_key_for_hook.clone();
        async move {
            let (node, conn_id) = guarded_connection(client, &header_key);
            kill_connection(&killer, &node, conn_id).await;
            await_reconnect(client, &node, conn_id).await;
            // Retire the object from a different connection, exactly as a
            // concurrent CreateLayer overwrite would. Under a live WATCH this
            // would cancel the EXEC; the point of the scenario is that the
            // WATCH is gone, so it does not.
            let _: () = data_pool
                .hset(&header_key, [("superseded", "1")])
                .await
                .expect("marking the header superseded must succeed");
        }
        .boxed()
    }));

    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone())
        .with_pre_dispatch_hook_for_test(hook);

    let result = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(1, "RetiredSection")),
            "bhb-retired-writer",
            1_000,
        ),
    )
    .await
    .expect("the disturbed update did not settle within the scenario budget");

    match result {
        Err(ServiceError::FailedPrecondition(msg)) => assert!(
            msg.contains("superseded"),
            "the retirement re-check must surface the same superseded sentinel an ordinary \
             overwrite race produces, so the caller's handling is identical whether or not \
             the WATCH held; got {msg:?}"
        ),
        other => panic!(
            "a proven-unwatched commit onto a retired object must surface the retirement \
             shape. `Ok` would report success for keys that are orphans on a doomed object; \
             `Aborted` would spin the retry loop against a header that will never come back. \
             Got: {other:?}"
        ),
    }
    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "committed_unwatched"),
        1,
        "the unwatched commit is counted whichever way its disposition falls — the operator \
         signal is `a commit ran on the wrong connection`, not `and it was also rejected`"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T5 — the pool-config fence
// ═════════════════════════════════════════════════════════════════════════════

/// b.bhb: the `txn` pool must be built at `max_command_attempts = 1` /
/// `max_redirections = 0`, and nothing else may be.
///
/// Read back from a **live** pool rather than from a re-derived builder: a
/// builder-level assertion would have to re-apply the same mutation it is
/// checking and could only ever agree with itself. `ClientLike::connection_config`
/// reports the config the connections were actually built with, so a refactor
/// that drops the `with_connection_config` closure, or reverts the pool to the
/// shared builder, fails here.
///
/// Three claims, and the second two are as load-bearing as the first:
///
/// * **1, not 0.** `write_command` calls `Command::decr_check_attempted()`
///   before writing ANY regular command and errors at 0, so a pool at 0 would
///   break the `WATCH` itself, every read under it, and the checkout hygiene
///   probe. 1 is the smallest value that writes each command exactly once.
///   (`Options::max_attempts: Some(0)` is not an escape either —
///   `inherit_options` rewrites a per-command 0 back to the pool default, which
///   is why the ceiling has to live on the pool config.)
/// * **The divergence is txn-pool-only.** The data pool's transparent retry is
///   load-bearing for the Lua fast path and every read; applying this ceiling
///   there would trade a rare correctness bug for a common availability one.
/// * **Everything else still matches.** The txn builder is a `clone()` of the
///   shared one, so credentials, TLS, timeouts and the unresponsive check
///   cannot drift from the data pool's by construction. A future
///   `Builder::from_config` rebuild would silently drop them all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_pool_pins_one_attempt_and_no_redirection_without_disturbing_the_data_pool() {
    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;

    let data = tv.client().pool.next().connection_config().clone();
    let txn_client = tv
        .client()
        .acquire_txn_for(&tv.key_schema().obj_header("bhb-fence"))
        .await;
    let txn = txn_client.connection_config().clone();

    assert_eq!(
        txn.max_command_attempts, 1,
        "the txn pool must write each command exactly once: anything higher lets fred \
         re-issue a command taken under a WATCH onto a connection that never held it, which \
         is the whole defect. Zero is not the answer either — it fails every command on the \
         pool before it is written"
    );
    assert_eq!(
        txn.max_redirections, 0,
        "a MOVED on one member of a pipelined MULTI would otherwise be re-sent ALONE to the \
         new node as an unqueued, immediately-executed write"
    );

    assert!(
        data.max_command_attempts > 1,
        "the DATA pool must keep fred's transparent retry — it is load-bearing for the Lua \
         fast path and every read. Finding the txn ceiling here means the mutation was \
         applied to the shared builder and every pool inherited it; got {}",
        data.max_command_attempts
    );
    assert!(
        data.max_redirections > 0,
        "the data pool must still follow MOVED redirections on a cluster; got {}",
        data.max_redirections
    );

    assert_eq!(
        (
            txn.connection_timeout,
            txn.reconnect_on_auth_error,
            txn.unresponsive.max_timeout,
            txn.unresponsive.interval,
        ),
        (
            data.connection_timeout,
            data.reconnect_on_auth_error,
            data.unresponsive.max_timeout,
            data.unresponsive.interval,
        ),
        "the txn pool is a clone of the shared builder with two fields overridden. Any other \
         difference means it was rebuilt from scratch, which silently drops the connect \
         timeout, the credential-refresh re-AUTH, and the unresponsive-connection check that \
         every other pool gets"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T6 — the OTHER half of the L1 bill: reads taken under the WATCH
// ═════════════════════════════════════════════════════════════════════════════

/// b.bhb: a reconnect landing on `create_once`'s `WATCH` or its duplicate-name
/// `SMEMBERS` must be **retriable**, never a terminal `Internal`.
///
/// # Why this exists
///
/// Capping the txn pool at `max_command_attempts = 1` was not free. Every
/// command on that pool — including the two the create path issues *under* its
/// `WATCH` — lost fred's transparent retry, so a routine reconnect that fred
/// used to absorb by re-issuing the command now surfaces to our code. Both
/// sites used to wrap it as `ServiceError::Internal`, which after the cap turns
/// an ordinary blip into a failed `CreateSession` the caller can do nothing
/// about. They now route through `map_transient_fred_err`, so the transient
/// kinds become `Aborted` and `retry_aborted_with_jitter` takes a fresh
/// checkout and a fresh `WATCH`.
///
/// Retrying is also the only *correct* answer for the `SMEMBERS`: it is the
/// fence the commit relies on, so a read that never completed must not be
/// treated as "no duplicate found".
///
/// # Why N iterations and not one
///
/// The disturbance is a real `CLIENT KILL ID` with **no** wait for the
/// replacement, and which command eats it is a genuine race inside fred's
/// router (module docs, `src/storage/watched_txn.rs`): if the router processes
/// the queued reconnect first, the `SMEMBERS` succeeds on the new connection and
/// the pre-dispatch identity check catches the swap instead. All three landing
/// spots — transient read, pre-dispatch check, in-transaction witness — are
/// correct outcomes and all three are absorbed by the same retry loop, so the
/// assertions below are the ones that hold across every one of them:
///
/// * the create **succeeds**, and
/// * exactly one attempt-visible retry happened (so the disturbance was really
///   staged — without this the test would pass identically against a no-op
///   hook), and
/// * exactly one member holds the name, and
/// * the alertable `committed_unwatched` cell stays clean.
///
/// Which route each iteration took is *reported*, not asserted: pinning a
/// distribution would be pinning fred's scheduler. On a loopback Valkey the
/// silent-`Aborted` route — the one that contains the reclassified `WATCH` /
/// `SMEMBERS` failures — takes roughly a third to a half of the iterations, so
/// pre-round-2 (both sites wrapping `Internal`) this test fails within the first
/// few. Twenty iterations is chosen to make that certain, not to sample a
/// distribution.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_reads_under_a_killed_watch_connection_stay_retriable() {
    /// Enough repetitions that the router race is sampled from both sides on a
    /// normal run, while the whole loop still fits inside one scenario budget
    /// (each iteration is a handful of round trips plus one reconnect).
    const ITERATIONS: usize = 20;

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    // Route tally, reported at the end. Index 0 is "no void-watch cell moved",
    // which is where the classification under test lands — but it is not
    // exclusively that: a pipeline whose write failed *after* the pre-dispatch
    // check produces `InDoubt`, whose reconciliation finds nothing and also
    // answers `Aborted` silently. Both are correct absorptions of the same
    // disturbance and neither is distinguishable from the outside, so the
    // label below says so rather than claiming more than the tally knows.
    let mut routes = [0usize; 3];

    tokio::time::timeout(SCENARIO_BUDGET, async {
        for i in 0..ITERATIONS {
            let name = format!("bhb-transient-{i}");
            let base_path = format!("/bhb-transient-{i}");
            let names_key = tv.key_schema().path_names_index(&base_path);
            let killer = tv.client().pool.next().clone();

            let before_pre_exec = read_watch_voided(&metrics, "create_session", "aborted_pre_exec");
            let before_compensated = read_watch_voided(&metrics, "create_session", "compensated");

            let attempts = Arc::new(AtomicUsize::new(0));
            let hook = counting_once(
                attempts.clone(),
                Arc::new(move |client: &Client| {
                    let killer = killer.clone();
                    let names_key = names_key.clone();
                    async move {
                        let (node, conn_id) = guarded_connection(client, &names_key);
                        // Deliberately NO `await_reconnect`: leaving fred
                        // mid-recovery is what puts the write failure on the
                        // `SMEMBERS` below rather than on the commit.
                        kill_connection(&killer, &node, conn_id).await;
                    }
                    .boxed()
                }),
            );

            let disturbed = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
                .with_watch_window_hook_for_test(hook);

            let session_id = disturbed
                .create(&name, &base_path, "bhb-user")
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "iteration {i}: a connection killed inside the WATCH window must be \
                         absorbed as a retriable failure and the create must still succeed. \
                         `Internal` here is the pre-round-2 behaviour: the WATCH / SMEMBERS \
                         wrap classified a routine reconnect as terminal, so capping the txn \
                         pool at one write attempt turned every reconnect on this path into a \
                         failed CreateSession. Got: {e:?}"
                    )
                });
            assert!(!session_id.is_empty(), "a created session must have an id");

            assert!(
                attempts.load(Ordering::SeqCst) >= 2,
                "iteration {i}: `create_once` ran only {} time(s), so nothing was retried — \
                 the connection kill did not disturb this attempt at all and every assertion \
                 here is vacuous. Whichever of the three routes the race takes, each one ends \
                 in `Aborted` and a second attempt",
                attempts.load(Ordering::SeqCst)
            );

            let holders = name_members(&tv, &base_path, &name).await;
            assert_eq!(
                holders.len(),
                1,
                "iteration {i}: exactly one session may hold {name:?} at {base_path:?} — a \
                 second member means an attempt that was retried also left its own commit \
                 behind; got {holders:?}"
            );

            let pre_exec =
                read_watch_voided(&metrics, "create_session", "aborted_pre_exec") - before_pre_exec;
            let compensated =
                read_watch_voided(&metrics, "create_session", "compensated") - before_compensated;
            match (pre_exec, compensated) {
                (0, 0) => routes[0] += 1,
                (_, 0) => routes[1] += 1,
                _ => routes[2] += 1,
            }
        }
    })
    .await
    .expect("the disturbed create loop did not settle within the scenario budget");

    eprintln!(
        "b.bhb create-path disturbance routes over {ITERATIONS} iterations: \
         silent Aborted (transient WATCH/SMEMBERS classification, or an in-doubt reconcile \
         that found nothing) = {}, pre-dispatch check = {}, witness compensation = {}",
        routes[0], routes[1], routes[2]
    );

    assert_eq!(
        read_watch_voided(&metrics, "create_session", "committed_unwatched"),
        0,
        "`committed_unwatched` on the create path means `the duplicate check did not run AND \
         its commit could not be undone` — a name that may now be held twice. No iteration \
         above stages that, so any tick is a real compensation failure, not a route"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T7 — counting discipline on the alertable cell
// ═════════════════════════════════════════════════════════════════════════════

/// b.bhb: an `EXEC` error frame that is NOT the stream-ID collision must not be
/// counted as a void `WATCH`.
///
/// # The distinction being pinned
///
/// `update_layer_once_watch`'s `ExecError` arm sees a torn commit: the frames that ran
/// landed, the failing one did not. Exactly one error shape there is *evidence*
/// the watch set was gone — `XADD {new_seq}-0` refused for an equal-or-smaller
/// id. This path watches the stream key, so under a live `WATCH` a competing
/// entry at that id would have cancelled the `EXEC` instead, and an entry that
/// predated the `WATCH` is caught by the pre-`MULTI` collision probe. Neither
/// survives a live watch.
///
/// A `WRONGTYPE` frame proves nothing of the sort. Out-of-band type corruption
/// on a field / section / index key is reachable with the watch fully intact —
/// nothing about watching the header, the stream and the snapshot fence
/// prevents it — so counting it
/// would put a false positive on the one cell operators page on. That is the
/// whole value of `committed_unwatched`: its baseline is zero.
///
/// # Staging — the probe→`EXEC` residual window (b.459)
///
/// This test used to plant its string *before* the call, which was the whole
/// staging: the section key is not in the sections index, so nothing before the
/// `MULTI` read it and Valkey type-checked at execution time. b.459 added a
/// pre-`MULTI` keyspace type probe over every key the commit will write, so that
/// staging no longer reaches this arm at all — the commit is now refused before
/// `MULTI` with a different message and no stream entry.
///
/// The arm is still reachable, and this test now drives *exactly* what makes it
/// so. Only the header, stream and snapshot-fence keys are WATCHed, so a writer that retypes a
/// mutated key **after** the probe and **before** the `EXEC` is invisible to both
/// guards. `with_pre_dispatch_hook_for_test` fires at precisely that point — the
/// probes have run and passed, the pipeline has not been sent — and the string is
/// planted there, from a second connection. That is the residual window b.459
/// documents rather than closes, and this test is what keeps it honest: if a
/// later change widened the `WATCH` to cover mutated keys, the `EXEC` would
/// cancel and this test would fail on the `Aborted` rather than pass silently.
///
/// The metric assertions split the two guards apart. `keyspace_type_error_total`
/// must read **zero**: the probe ran and found the keyspace clean, which is the
/// positive evidence that the corruption arrived inside the window rather than
/// before it. `committed_unwatched` must read zero for the original reason —
/// a `WRONGTYPE` frame is reachable with the watch fully intact.
///
/// The stream assertion is not decoration: it is the positive evidence that the
/// `EXEC` actually **ran** (its `XADD` sibling committed). Without it a queue-time
/// or pre-flight rejection would satisfy the error assertion and the arm under
/// test would never have been entered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_exec_error_that_is_not_a_stream_id_collision_is_not_counted_as_a_void_watch()
{
    const LAYER: &str = "bhb-wrongtype-layer";
    const CLIENT: &str = "bhb-wrongtype-writer";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(metrics.clone());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bhb-wrongtype-session", "/bhb-wrongtype-stage", "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = plain
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // Plant a string where the commit will HSET a hash — from a SECOND
    // connection, inside the probe→`EXEC` window. The b.459 probe has already
    // run and passed at this point, and the section key is not among the two
    // keys under `WATCH`, so neither guard can see this land.
    let section_key = tv.key_schema().obj_section(&object_id.to_string(), "1");
    let planter = tv.client().clone();
    let planted_key = section_key.clone();
    let hook: WatchWindowHook = once_only(Arc::new(move |_guarded: &Client| {
        let planter = planter.clone();
        let planted_key = planted_key.clone();
        async move {
            let _: () = planter
                .pool
                .set(&planted_key, "not-a-hash", None, None, false)
                .await
                .expect("planting the wrong-typed section key must succeed");
        }
        .boxed()
    }));

    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone())
        .with_pre_dispatch_hook_for_test(hook);

    let result = tokio::time::timeout(
        SCENARIO_BUDGET,
        store.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(1, "WrongTypeSection")),
            CLIENT,
            1_000,
        ),
    )
    .await
    .expect("the wrong-typed update did not settle within the scenario budget");

    match result {
        Err(ServiceError::Internal(msg)) => {
            assert!(
                msg.contains("EXEC"),
                "the tear must be reported as an EXEC failure so the operator sees where it \
                 happened; got {msg:?}"
            );
            assert!(
                msg.contains("WRONGTYPE"),
                "the server's own error must survive the wrap — it is the only thing that \
                 tells an operator the failure was keyspace type corruption rather than a \
                 concurrency outcome; got {msg:?}"
            );
        }
        Ok((seq, _)) => panic!(
            "a commit whose section HSET was refused must NOT be acked at sequence {seq}: the \
             materialized state is missing the section the client was told had landed"
        ),
        Err(ServiceError::Aborted(msg)) => panic!(
            "type corruption is not retriable — the rest of the MULTI landed, the b.75i \
             commit-token record with it, so a retry would find that record and ack this \
             half-applied commit. Got: {msg:?}"
        ),
        Err(other) => panic!("unexpected classification for a WRONGTYPE tear: {other:?}"),
    }

    // Positive evidence the EXEC ran: its XADD sibling committed. Valkey does
    // not roll a transaction back on an execution-time error, so the entry is
    // there even though the HSET beside it failed — this IS the tear.
    assert_stream_entry(&stream_payloads(&tv, object_id).await, 2, CLIENT);

    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "committed_unwatched"),
        0,
        "a WRONGTYPE frame is reachable with the WATCH fully intact, so it is not evidence of \
         anything about the watch set. Ticking here puts a false positive on the alertable \
         cell whose entire value is that its baseline is zero — an operator paged by it would \
         go looking for a connection-identity failure that never happened"
    );
    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "aborted_pre_exec"),
        0,
        "no connection was disturbed, so the pre-dispatch check must not have fired either"
    );
    assert_eq!(
        read_keyspace_type_error(&metrics, "txn", "update"),
        0,
        "the b.459 pre-`MULTI` type probe ran against a CLEAN keyspace — the string arrives \
         after it — so it must not have ticked. A tick here means the planting moved back \
         outside the probe→EXEC window and this test is no longer driving the residual \
         window at all: it would be re-testing the probe, and the `ExecError` arm above \
         would be unreachable"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T8 — the in-doubt reconciliation arms (override seam)
// ═════════════════════════════════════════════════════════════════════════════

/// What [`update_landed_in_doubt`] observed.
struct LandedInDoubt {
    /// The `(seq, server_ts)` the in-doubt `update_layer` reported.
    seq: u64,
    server_ts: u64,
    object_id: u64,
    /// The object's stream as `{id → payload}`, read after the update.
    entries: Vec<(String, String)>,
    /// The object header's `seq`, read after the update.
    header_seq: u64,
    metrics: Arc<Metrics>,
}

/// One structural `update_layer` as `client_id` at `client_ts` on a fresh
/// layer whose `EXEC` verdict is discarded after the real pipeline committed
/// (`ReplaceOutcomeAfterExec`): the in-doubt arm over a commit that landed.
/// Asserts the override was consumed, and panics with `on_error` if the update
/// fails.
async fn update_landed_in_doubt(client_id: &str, client_ts: u64, on_error: &str) -> LandedInDoubt {
    const LAYER: &str = "bhb-indoubt-layer";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(metrics.clone());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bhb-indoubt", "/bhb-indoubt", "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = plain
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let overrides = override_queue(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone())
        .with_watched_exec_override_for_test(overrides.clone());

    let (seq, server_ts) = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(1, "InDoubtSection")),
            client_id,
            client_ts,
        ),
    )
    .await
    .expect("the in-doubt update did not settle within the scenario budget")
    .unwrap_or_else(|e| panic!("{on_error}: {e:?}"));
    assert_override_consumed(&overrides);

    LandedInDoubt {
        seq,
        server_ts,
        object_id,
        entries: stream_payloads(&tv, object_id).await,
        header_seq: header_seq(&tv, object_id).await,
        metrics,
    }
}

/// b.bhb: an `EXEC` reply lost over a commit that **did** land must be reported
/// as the commit it was, at the sequence and server timestamp the stream carries.
///
/// This is the arm the ticket added, and the failure it removed was
/// exactly-once: pre-b.bhb an in-doubt `EXEC` was an unconditional `Aborted`,
/// so a commit that had already landed went back through the retry loop and
/// applied the same delta a second time at `new_seq + 1`. The `XRANGE` over the
/// deterministic `{new_seq}-0` id, matched against this writer's own
/// `client_id` / `client_timestamp`, is what turns "unknown" into an answer.
///
/// Since b.75i a blind `Aborted` would no longer double-apply — the retry's
/// commit-token check would report the commit — but it would drop the
/// external-value releases only this attempt holds. So the reconcile must
/// still answer here, and `commit_already_landed_total` staying at zero is
/// what tells it apart from the backstop.
///
/// The `server_ts` equality is not a detail: reconciliation reports the value it
/// read back from the entry rather than the one it computed before dispatch, so
/// what the caller is told is sourced from the commit that actually exists.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_in_doubt_after_a_landed_commit_reports_the_commit() {
    const CLIENT: &str = "bhb-indoubt-landed-writer";

    let LandedInDoubt {
        seq,
        server_ts,
        entries,
        header_seq,
        metrics,
        ..
    } = update_landed_in_doubt(
        CLIENT,
        1_000,
        "the commit landed, so reconciliation must find it and report success. An error here \
         tells the client its write was lost while the stream and the materialized state both \
         say otherwise",
    )
    .await;

    assert_eq!(seq, 2, "the reported sequence is the one that committed");

    assert_stream_entry(&entries, 2, CLIENT);
    assert_eq!(
        entries.len(),
        2,
        "the stream must carry exactly CreateLayer's 1-0 and this commit's 2-0. A third entry \
         is the delta applied twice at 3-0 — a landed commit reported as retriable AND not \
         caught by the retry's commit-token check. Got {entries:?}"
    );
    assert_eq!(header_seq, 2, "the header must have advanced exactly once");
    assert_eq!(
        read_commit_already_landed(&metrics, "txn"),
        0,
        "the reconcile itself must report the commit. A tick means it answered `Aborted` and \
         the retry's commit-token check reported the commit instead, without the \
         external-value releases the reconcile would have enqueued"
    );

    let landed_payload = entries
        .iter()
        .find(|(id, _)| id == "2-0")
        .map(|(_, payload)| payload.clone())
        .expect("entry 2-0 was asserted present above");
    assert_eq!(
        server_ts,
        payload_server_ts(&landed_payload),
        "the reported server timestamp must be the one the committed entry carries — \
         reconciliation reads it back rather than re-reporting the value computed before \
         dispatch, so that what the caller is told is sourced from the commit that exists"
    );

    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "committed_unwatched"),
        0,
        "a lost reply says nothing about which connection ran the transaction, so the \
         void-watch cell must stay clean"
    );
}

/// b.bhb: an `EXEC` reply lost over a transaction that never reached the wire
/// must be retriable, and the retry must commit exactly once.
///
/// The mirror image of the test above, and the reason the seam has two variants:
/// both hand the call site the identical `InDoubt`, and the *only* thing that
/// distinguishes them is what the keyspace holds. Reconciliation finds nothing at
/// `{new_seq}-0`, so the honest answer is the retriable one — and the retry, now
/// running without an override, commits at the sequence it originally computed
/// because nothing else moved the header.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_in_doubt_with_nothing_dispatched_retries_and_commits_exactly_once() {
    const LAYER: &str = "bhb-indoubt-nothing-layer";
    const CLIENT: &str = "bhb-indoubt-nothing-writer";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(metrics.clone());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bhb-indoubt-nothing", "/bhb-indoubt-nothing", "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = plain
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let overrides = override_queue(&[WatchedExecOverride::SkipExecReturnInDoubt]);
    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone())
        .with_watched_exec_override_for_test(overrides.clone());

    let (seq, _) = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(1, "InDoubtSection")),
            CLIENT,
            1_000,
        ),
    )
    .await
    .expect("the in-doubt update did not settle within the scenario budget")
    .expect(
        "nothing was written, so the in-doubt verdict must be retriable and the uncontended \
         retry must commit. A terminal error means the arm reported a failure for a write \
         that had simply not happened yet",
    );
    assert_override_consumed(&overrides);

    assert_eq!(
        seq, 2,
        "nothing else moved the header, so the retry commits at the sequence the first \
         attempt computed"
    );
    let entries = stream_payloads(&tv, object_id).await;
    assert_stream_entry(&entries, 2, CLIENT);
    assert_eq!(
        entries.len(),
        2,
        "CreateLayer's 1-0 plus one commit — the abandoned attempt must have left nothing \
         behind; got {entries:?}"
    );
    assert_eq!(header_seq(&tv, object_id).await, 2);

    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "committed_unwatched"),
        0,
        "nothing was dispatched, so nothing may be reported as having committed unwatched"
    );
}

/// b.bhb: an in-doubt verdict over a pipeline that never reached the wire must
/// leave **no transaction residue** on the txn slot it borrowed.
///
/// Under `SkipExecReturnInDoubt` the pipeline is dropped before it is written, so
/// the site reaches its `InDoubt` arm having sent nothing: the keyspace is
/// untouched and the `WATCH` is still live on that connection, which is why —
/// unlike every other arm — this one must NOT call `mark_unwatched()`. The retry
/// budget is pinned to 1 so the call ends on that arm rather than re-`WATCH`ing
/// and `EXEC`ing on the same slot, which would clear whatever it left behind and
/// mask the question entirely.
///
/// **What the follow-up checkout observes.** The `txn_pool` hygiene probe (b.7td,
/// `storage::txn_checkout::probe_txn_slot`) runs `DISCARD` then `UNWATCH` on the
/// slot and classifies the replies: a `+OK` to `DISCARD` means a `MULTI` was open
/// (`reason="leaked_multi"`), and a reply matching neither expectation means the
/// connection's reply stream is offset (`reason="unexpected_reply"`) — which is
/// the shape an abandoned in-flight pipeline leaves. Both tick
/// `live_edit_valkey_txn_dirty_checkouts_total`, so **that counter staying flat
/// across the follow-up `acquire_txn_for` is what this test asserts**, with
/// `valkey_pool_in_use{pool="txn"}` read while the slot is held to prove the
/// counter is live rather than reading a registry nobody reports into.
///
/// A leaked *`WATCH`* specifically is NOT observable and this test does not claim
/// it: the probe's `UNWATCH` answers `+OK` whether or not a watch set was live and
/// nothing counts it, so that residue would be repaired unseen. Closing that needs
/// a counter on the probe's repair, not another test here — the same gap
/// `watch_leak_test.rs`'s b.jov assertion has.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_in_doubt_without_a_dispatch_leaves_no_open_multi_on_the_txn_slot() {
    const LAYER: &str = "bhb-indoubt-slot-layer";

    // pool_size = 1: the `ExclusivePool` has one slot, so the follow-up checkout
    // below provably probes the same connection the abandoned attempt borrowed.
    // Anything larger and it could pick a different slot and pass vacuously.
    let tv = TestValkey::setup_with_pool_size(1).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bhb-indoubt-slot", "/bhb-indoubt-slot", "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = plain
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let overrides = override_queue(&[WatchedExecOverride::SkipExecReturnInDoubt]);
    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_update_layer_retry_max_attempts(1)
        .with_watched_exec_override_for_test(overrides.clone());

    let result = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(1, "InDoubtSection")),
            "bhb-indoubt-slot-writer",
            1_000,
        ),
    )
    .await
    .expect("the in-doubt update did not settle within the scenario budget");
    assert_override_consumed(&overrides);
    // ── Secondary: the verdict itself, and that nothing was written. ──
    assert!(
        matches!(result, Err(ServiceError::Aborted(_))),
        "with a one-attempt budget the retriable verdict must surface as exhausted `Aborted`; \
         anything else means the arm classified a never-dispatched transaction as terminal. \
         Got: {result:?}"
    );
    assert_eq!(
        stream_payloads(&tv, object_id).await.len(),
        1,
        "only CreateLayer's 1-0 may exist — the abandoned attempt wrote nothing"
    );

    // ── The assertion: the next borrower's probe finds the slot clean. ──
    let before = read_txn_dirty_checkouts(&metrics);
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let slot = tv.client().acquire_txn_for(&header_key).await;
    let after = read_txn_dirty_checkouts(&metrics);

    // Vacuity guard: the gauge is incremented by the same `metrics_snapshot()`
    // the dirty counter is ticked through, so a nonzero reading here proves the
    // flat counter above means "the probe found nothing", not "this client
    // reports into no registry at all".
    assert_eq!(
        read_pool_in_use(&metrics, "txn"),
        1,
        "the txn slot is held right now, so valkey_pool_in_use{{pool=\"txn\"}} must read 1. A 0 \
         means the observability hooks are not wired to this client and the dirty-checkout \
         assertion below cannot fail for any reason"
    );
    assert_eq!(
        after, before,
        "the checkout after the in-doubt arm found the slot DIRTY. On a one-slot pool this is \
         the same connection the abandoned attempt borrowed, so the probe is reporting residue \
         that arm left: reason=\"leaked_multi\" means it went back to the pool with a MULTI \
         open, reason=\"unexpected_reply\" means its reply stream is offset by a command whose \
         answer nobody read. Either way the next borrower's transaction inherits it"
    );

    drop(slot);
}

/// b.bhb: an entry at `{new_seq}-0` written by **someone else** is not evidence
/// that our commit landed.
///
/// This is the discrimination that makes the whole in-doubt arm safe. The probe
/// is an `XRANGE` over one deterministic id, and the naive reading of a hit is
/// "the transaction ran" — which here would ack a sequence that carries a
/// different writer's delta and drop ours on the floor. Matching the entry's
/// `client_id` and `client_timestamp` against this writer's own is what makes the
/// hit mean "*mine*".
///
/// Staging note: the competitor commits inside the pre-dispatch hook, so the real
/// `EXEC` verdict would have been a WATCH `Conflict` (it modified both watched
/// keys). `ReplaceOutcomeAfterExec` discards that verdict and hands the site
/// `InDoubt` instead — which is the point: the arm must reach the right answer
/// from the keyspace alone, having been told nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_in_doubt_over_a_foreign_stream_entry_retries_at_the_next_sequence() {
    const LAYER: &str = "bhb-indoubt-foreign-layer";
    const COMPETITOR_CLIENT: &str = "bhb-indoubt-foreign-competitor";
    const DISTURBED_CLIENT: &str = "bhb-indoubt-foreign-writer";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(metrics.clone());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bhb-indoubt-foreign", "/bhb-indoubt-foreign", "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = plain
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let competitor = Arc::new(plain.clone());
    let competitor_session = session_id.clone();
    let hook: WatchWindowHook = once_only(Arc::new(move |_client: &Client| {
        let competitor = competitor.clone();
        let competitor_session = competitor_session.clone();
        async move {
            let (seq, _) = competitor
                .update_layer_relayed(
                    &competitor_session,
                    LAYER,
                    object_id,
                    Some(section_delta(1, "CompetitorSection")),
                    COMPETITOR_CLIENT,
                    1_000,
                )
                .await
                .expect("the competing structural update must commit — it is uncontended");
            assert_eq!(
                seq, 2,
                "the competitor must take the sequence the disturbed writer already computed \
                 for itself; without that the reconciling XRANGE would find nothing and this \
                 test would be exercising the empty-probe branch instead"
            );
        }
        .boxed()
    }));

    let overrides = override_queue(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone())
        .with_pre_dispatch_hook_for_test(hook)
        .with_watched_exec_override_for_test(overrides.clone());

    let (seq, _) = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(2, "DisturbedSection")),
            DISTURBED_CLIENT,
            2_000,
        ),
    )
    .await
    .expect("the in-doubt update did not settle within the scenario budget")
    .expect(
        "our commit did not land, so the verdict must be retriable and the retry must commit \
         against the advanced header",
    );
    assert_override_consumed(&overrides);

    assert_eq!(
        seq, 3,
        "reading the competitor's entry as our own would have acked sequence 2 — a sequence \
         carrying a DIFFERENT writer's delta, with ours silently dropped"
    );
    let entries = stream_payloads(&tv, object_id).await;
    assert_stream_entry(&entries, 2, COMPETITOR_CLIENT);
    assert_stream_entry(&entries, 3, DISTURBED_CLIENT);
    assert_eq!(entries.len(), 3, "1-0, the competitor's 2-0, and our 3-0");

    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let sections: HashSet<u64> = snapshot.sections.keys().copied().collect();
    assert_eq!(
        sections,
        HashSet::from([1u64, 2u64]),
        "both writers' sections must be materialized; got {sections:?}"
    );

    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "committed_unwatched"),
        0,
        "the WATCH held throughout — the competitor's write is what would have cancelled the \
         EXEC — so nothing here is evidence of a void watch"
    );
}

/// b.bhb: a `CreateSession` whose `EXEC` reply was lost over a commit that landed
/// must be **completed**, not retried.
///
/// The retry is not neutral on this path: every attempt runs under a fresh
/// `session_id`, so a retry would commit a second id and leave the landed one
/// behind as a metadata-less names member under the same name — a ghost this
/// call's scans skip, but which every other caller sees until a ListSessions
/// removes it. (Before each attempt had its own id, the retry found the landed
/// member and handed the caller `AlreadyExists` for its own session.) One
/// `SISMEMBER` on the composite `name\0{uuid}` member answers "did my transaction
/// run?" without guessing, because that member is unique to this attempt.
///
/// The metadata assertion is what pins the *completion*: uniqueness committing is
/// not the whole of `create`, and an arm that returned `Ok` without running
/// `finish_create` would leave a member pointing at an empty `session_meta` —
/// invisible to `list`, and `NotFound` to a subsequent `DeleteSession`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_in_doubt_after_a_landed_commit_completes_the_create() {
    const NAME: &str = "bhb-indoubt-create-landed";
    const BASE_PATH: &str = "/bhb-indoubt-create-landed";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let overrides = override_queue(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
    let disturbed = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_watched_exec_override_for_test(overrides.clone());

    let session_id = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.create(NAME, BASE_PATH, "bhb-user"),
    )
    .await
    .expect("the in-doubt create did not settle within the scenario budget")
    .expect(
        "the commit landed and the name is still unique, so the create must be completed. \
         `AlreadyExists` here is the pre-b.bhb behaviour in its sharpest form: the caller is \
         told the name is taken by the very session they just created",
    );
    assert_override_consumed(&overrides);

    let holders = name_members(&tv, BASE_PATH, NAME).await;
    assert_eq!(
        holders.len(),
        1,
        "exactly one member may hold {NAME:?}; got {holders:?}"
    );
    assert_eq!(
        holders[0],
        format!("{NAME}\u{0}{session_id}"),
        "the surviving member must be the one this call committed"
    );

    let in_path_index: bool = tv
        .client()
        .pool
        .sismember(tv.key_schema().path_index(BASE_PATH), &session_id)
        .await
        .expect("SISMEMBER on the path index must succeed");
    assert!(
        in_path_index,
        "both SADDs ran in one transaction, so the session-id index must carry it too"
    );

    let meta = session_meta(&tv, &session_id).await;
    assert_eq!(
        meta.get("name").map(String::as_str),
        Some(NAME),
        "the in-doubt-but-landed arm must run the post-EXEC metadata write, not just return \
         `Ok`: a member whose session_meta is empty is invisible to `list` and `NotFound` to \
         `DeleteSession`. Got: {meta:?}"
    );
    assert_eq!(meta.get("base_path").map(String::as_str), Some(BASE_PATH));
    assert_eq!(meta.get("created_by").map(String::as_str), Some("bhb-user"));

    assert_eq!(
        read_watch_voided(&metrics, "create_session", "compensated"),
        0,
        "the name was unique, so there was nothing to undo — compensating here would delete a \
         perfectly good session's uniqueness claim"
    );
    assert_eq!(
        read_watch_voided(&metrics, "create_session", "committed_unwatched"),
        0
    );
}

/// b.bhb: a `CreateSession` whose transaction never reached the wire must be
/// retriable, and the retry must create exactly one session.
///
/// The other keyspace answer to the same `InDoubt`: the `SISMEMBER` finds
/// nothing, so the attempt provably did not run and `Aborted` is correct. The
/// retry runs under a fresh `session_id`, so had the abandoned attempt's commit
/// run after all, it would show up as a second member under a different id —
/// which is why "exactly one member, carrying the returned id" is a real
/// assertion here rather than an arithmetic accident.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_in_doubt_with_nothing_dispatched_retries_and_creates_once() {
    const NAME: &str = "bhb-indoubt-create-nothing";
    const BASE_PATH: &str = "/bhb-indoubt-create-nothing";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let overrides = override_queue(&[WatchedExecOverride::SkipExecReturnInDoubt]);
    let disturbed = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_watched_exec_override_for_test(overrides.clone());

    let session_id = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.create(NAME, BASE_PATH, "bhb-user"),
    )
    .await
    .expect("the in-doubt create did not settle within the scenario budget")
    .expect(
        "nothing was written, so the verdict must be retriable and the uncontended retry must \
         create the session",
    );
    assert_override_consumed(&overrides);

    let holders = name_members(&tv, BASE_PATH, NAME).await;
    assert_eq!(
        holders.len(),
        1,
        "exactly one member may hold {NAME:?} — the abandoned attempt wrote nothing and the \
         retry wrote once; got {holders:?}"
    );
    assert_eq!(holders[0], format!("{NAME}\u{0}{session_id}"));
    assert_eq!(
        session_meta(&tv, &session_id).await.get("name").cloned(),
        Some(NAME.to_owned()),
        "the retry must complete the create, metadata included"
    );

    assert_eq!(
        read_watch_voided(&metrics, "create_session", "compensated"),
        0,
        "nothing was dispatched, so there is nothing to compensate"
    );
}

/// b.bhb: when the lost `EXEC` did not run and the name was taken while it was in
/// doubt, the caller loses the race honestly.
///
/// The competitor commits inside the pre-dispatch hook, so the real verdict would
/// have been a WATCH `Conflict`; the override replaces it with `InDoubt`, and the
/// arm has to work out from the keyspace that its own member is absent. `Aborted`
/// sends it back through the retry loop, whose duplicate scan — now running under
/// a live `WATCH` against a names set that holds the competitor — answers
/// `AlreadyExists`.
///
/// The negative metric assertions are the point of separating this from its
/// sibling below: nothing of ours landed, so there is nothing to compensate, and
/// a compensation tick here would mean the arm `SREM`ed members it had never
/// written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_in_doubt_when_a_competitor_won_the_name_first_reports_already_exists() {
    const NAME: &str = "bhb-indoubt-create-lost";
    const BASE_PATH: &str = "/bhb-indoubt-create-lost";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let competitor = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let winner_id: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let winner_slot = winner_id.clone();

    let hook: WatchWindowHook = once_only(Arc::new(move |_client: &Client| {
        let competitor = competitor.clone();
        let winner_slot = winner_slot.clone();
        async move {
            let id = competitor
                .create(NAME, BASE_PATH, "bhb-competitor")
                .await
                .expect("the competing create must succeed — it is uncontended");
            *winner_slot.lock().unwrap() = Some(id);
        }
        .boxed()
    }));

    let overrides = override_queue(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
    let disturbed = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_pre_dispatch_hook_for_test(hook)
        .with_watched_exec_override_for_test(overrides.clone());

    let result = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.create(NAME, BASE_PATH, "bhb-disturbed"),
    )
    .await
    .expect("the in-doubt create did not settle within the scenario budget");
    assert_override_consumed(&overrides);

    assert!(
        matches!(result, Err(ServiceError::AlreadyExists(_))),
        "our transaction was cancelled by the competitor's write, so the retry's duplicate \
         scan must find the taken name and say so. `Ok` would mean a second holder of the \
         name was written. Got: {result:?}"
    );

    let winner = winner_id.lock().unwrap().clone().expect("the hook ran");
    let holders = name_members(&tv, BASE_PATH, NAME).await;
    assert_eq!(
        holders,
        vec![format!("{NAME}\u{0}{winner}")],
        "the competitor must be the sole holder of the name"
    );

    assert_eq!(
        read_watch_voided(&metrics, "create_session", "compensated"),
        0,
        "our commit never landed, so compensating would SREM members we never wrote — and on \
         a composite member keyed by our own uuid, that is a no-op that hides a real bug"
    );
    assert_eq!(
        read_watch_voided(&metrics, "create_session", "committed_unwatched"),
        0
    );
}

/// b.bhb: when the lost `EXEC` **did** run — unwatched, alongside a competitor's
/// commit — reconciliation must re-scan, undo its own write, and hand the caller
/// the loss.
///
/// This is the uniqueness-critical branch, and the one place two answers are both
/// true at once: our member landed (so `SISMEMBER` says yes) *and* the
/// duplicate-name check that ran before it fenced nothing (because the `WATCH`
/// was void by the time the `EXEC` was written). Reporting the `SISMEMBER` hit as
/// success is the b.bhb defect in its final form: two sessions, one name, one
/// base path, with no error anywhere.
///
/// Staging is the real mechanism, not a simulation: the `WATCH`'s connection is
/// killed and fred is allowed to finish replacing it, so the commit genuinely runs
/// on a socket that never held the watch set. Only the *verdict* is substituted,
/// which is what routes it into the reconciliation arm instead of the
/// witness-mismatch arm — the same conclusion has to be reached twice, by two
/// different routes, and this pins the second one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_in_doubt_over_an_unwatched_commit_compensates_and_reports_already_exists() {
    const NAME: &str = "bhb-indoubt-create-unwatched";
    const BASE_PATH: &str = "/bhb-indoubt-create-unwatched";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let competitor = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let killer = tv.client().pool.next().clone();
    let names_key = tv.key_schema().path_names_index(BASE_PATH);
    let winner_id: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let winner_slot = winner_id.clone();

    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let competitor = competitor.clone();
        let killer = killer.clone();
        let names_key = names_key.clone();
        let winner_slot = winner_slot.clone();
        async move {
            let (node, conn_id) = guarded_connection(client, &names_key);
            kill_connection(&killer, &node, conn_id).await;
            // A COMPLETED reconnect: the commit below is written successfully,
            // onto a socket that never held the WATCH. Without this the write
            // would fail instead of landing, and the arm under test would see
            // an absent member rather than its own unwatched commit.
            await_reconnect(client, &node, conn_id).await;
            let id = competitor
                .create(NAME, BASE_PATH, "bhb-competitor")
                .await
                .expect("the competing create must succeed — it is uncontended");
            *winner_slot.lock().unwrap() = Some(id);
        }
        .boxed()
    }));

    let overrides = override_queue(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
    let disturbed = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_pre_dispatch_hook_for_test(hook)
        .with_watched_exec_override_for_test(overrides.clone());

    let result = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.create(NAME, BASE_PATH, "bhb-disturbed"),
    )
    .await
    .expect("the in-doubt create did not settle within the scenario budget");
    assert_override_consumed(&overrides);

    assert!(
        matches!(result, Err(ServiceError::AlreadyExists(_))),
        "the commit landed without its WATCH and the name was already taken, so this attempt \
         must lose the race. `Ok` is the uniqueness loss the whole ticket is about: the \
         SISMEMBER hit was believed without re-checking what else claims the name. Got: \
         {result:?}"
    );

    let winner = winner_id.lock().unwrap().clone().expect("the hook ran");
    let holders = name_members(&tv, BASE_PATH, NAME).await;
    assert_eq!(
        holders,
        vec![format!("{NAME}\u{0}{winner}")],
        "compensation must have removed OUR member and left the competitor's — two members \
         here is the defect itself; zero would mean it removed the wrong one"
    );

    assert_eq!(
        read_watch_voided(&metrics, "create_session", "compensated"),
        1,
        "reconciliation must have undone its own unwatched commit exactly once. 0 means the \
         duplicate re-scan never ran and the single surviving member above is luck"
    );
    assert_eq!(
        read_watch_voided(&metrics, "create_session", "committed_unwatched"),
        0,
        "the compensating SREMs succeeded, so the residual cell — `the duplicate check did \
         not run and could not be undone` — must stay clean"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T9 — cluster-gated twins of the two regressions
// ═════════════════════════════════════════════════════════════════════════════

/// b.bhb on a real cluster: the session regression, with the watched slot's
/// owner resolved through fred's cached routing view.
///
/// Single-node coverage cannot exercise that resolution — `WatchFingerprint`'s
/// one-connection arm names the only connection there is, and every route is
/// the right route. Here the connection identity must be looked up by slot, and
/// a wrong lookup would compare the WATCH's node against some other node's
/// connection id and fabricate (or miss) a void-watch verdict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_create_session_commit_dispatched_after_a_reconnect_keeps_the_name_unique() {
    let Some(tv) = cluster_fixture(POOL_SIZE).await else {
        return;
    };
    scenario_create_session_witness(&tv).await;
}

/// b.bhb on a real cluster: the layer torn-commit detection. Same reasoning as
/// its session twin above, plus the `max_redirections = 0` half of the pool
/// config — on a cluster a `MOVED` is reachable, and a redirected member of a
/// pipelined `MULTI` would be re-sent alone to the new node as an unqueued,
/// immediately-executed write.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_update_layer_torn_commit_after_a_void_watch_is_detected() {
    let Some(tv) = cluster_fixture(POOL_SIZE).await else {
        return;
    };
    scenario_update_layer_torn_commit_detected(&tv).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T10 — in-doubt reconciliation must recognise its own entry when the client
//       timestamp exceeds i64::MAX
// ═════════════════════════════════════════════════════════════════════════════

/// A `client_timestamp` above `i64::MAX`.
///
/// `UpdateLayer`'s proto types this field as an unvalidated `uint64` and
/// `LayerStore::update_layer` takes it as a `u64` and formats it straight into
/// the stream payload — nothing between the wire and the `XADD` narrows or
/// rejects it, so this is a value a real client can send today. `u64::MAX - 1`
/// rather than `u64::MAX` only so an off-by-one saturation somewhere could not
/// be mistaken for a correct round trip.
const CLIENT_TS_ABOVE_I64_MAX: u64 = u64::MAX - 1;

/// b.bhb (CR-3): an in-doubt `EXEC` over a commit that landed with a client
/// timestamp above `i64::MAX` must still be recognised as **ours**.
///
/// `reconcile_in_doubt_commit` decides whether the entry sitting at
/// `{new_seq}-0` is this attempt's by comparing the read-back `payload` field
/// against the payload this attempt built. `LayerNotification::parse` reads
/// `client_ts` as `u64`, so a timestamp above `i64::MAX` parses either way:
/// this test does NOT tell the byte comparison apart from a
/// parse-and-compare form. The sibling
/// `layer_in_doubt_recognises_its_own_commit_when_the_client_id_breaks_the_notification_parse`
/// is the guard against a return to one.
///
/// What it pins is the end-to-end outcome for a client-controlled timestamp
/// above `i64::MAX`: the in-doubt reconcile recognises the landed entry as
/// ours, the delta is applied exactly once (one stream entry at `2-0`, header
/// at 2), and `commit_already_landed_total` stays at zero — the reconcile
/// recognises the entry itself rather than leaving it to the retry's b.75i
/// commit-token check.
///
/// The staging is the same `ReplaceOutcomeAfterExec` the sibling
/// `layer_in_doubt_after_a_landed_commit_reports_the_commit` uses — the real
/// pipeline runs and commits, only the verdict is discarded — so the ONLY
/// difference between the two tests is the timestamp. That is deliberate: it
/// makes this a single-variable regression guard rather than a second scenario.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_in_doubt_recognises_its_own_commit_when_the_client_timestamp_exceeds_i64_max() {
    const CLIENT: &str = "bhb-indoubt-hugets-writer";

    let LandedInDoubt {
        seq,
        server_ts,
        entries,
        header_seq,
        metrics,
        ..
    } = update_landed_in_doubt(
        CLIENT,
        CLIENT_TS_ABOVE_I64_MAX,
        "the commit landed, so reconciliation must find it and report success. An error here \
         means the in-doubt reconcile failed on an entry carrying a client timestamp above \
         i64::MAX",
    )
    .await;

    assert_eq!(
        seq, 2,
        "the reported sequence is the one that committed, not the one a re-application would \
         have taken"
    );

    // ── The regression itself: exactly one application of the delta. ──
    assert_stream_entry(&entries, 2, CLIENT);
    assert_eq!(
        entries.len(),
        2,
        "the stream must carry exactly CreateLayer's 1-0 and this commit's 2-0. A third entry \
         at 3-0 means the landed commit was not recognised as ours and the retry loop applied \
         the same delta again. Got {entries:?}"
    );
    assert_eq!(
        header_seq, 2,
        "the header must have advanced exactly once — a 3 means the delta was double-applied \
         even if the stream were later trimmed"
    );
    assert_eq!(
        read_commit_already_landed(&metrics, "txn"),
        0,
        "the reconcile must recognise its own entry. A tick is the CR-3 defect as it shows \
         today: the entry was disowned, the attempt answered `Aborted`, and only the retry's \
         commit-token check kept the delta from applying twice"
    );

    // ── The timestamp really did survive the round trip un-narrowed. ──
    let landed_payload = entries
        .iter()
        .find(|(id, _)| id == "2-0")
        .map(|(_, payload)| payload.clone())
        .expect("entry 2-0 was asserted present above");
    let payload_client_ts = landed_payload
        .split('|')
        .nth(4)
        .unwrap_or_else(|| panic!("update payload must carry six fields; got {landed_payload:?}"));
    assert_eq!(
        payload_client_ts,
        CLIENT_TS_ABOVE_I64_MAX.to_string(),
        "the entry must carry the writer's timestamp verbatim — if the write path itself \
         narrowed it to i64 the byte comparison would match for the wrong reason and this test \
         would stop guarding anything"
    );
    assert_eq!(
        server_ts,
        payload_server_ts(&landed_payload),
        "the reported server timestamp must still be the one the committed entry carries"
    );

    assert_eq!(
        read_watch_voided(&metrics, "update_layer", "committed_unwatched"),
        0,
        "a lost reply says nothing about which connection ran the transaction"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// b.izg — in-doubt reconciliation decides ownership on the payload bytes, so
//         it recognises its own entry even when the parse rejects it
// ═════════════════════════════════════════════════════════════════════════════

/// A `client_id` that `LayerStore::update_layer` writes into the stream payload
/// verbatim but `LayerNotification::parse` rejects: its `|` splits the
/// `client_id` slot in two, so the `client_ts` slot reads `"writer"`.
///
/// No other writer-emittable value is rejected by the parse today: `object_id`
/// and `seq` are server-assigned, `client_ts` parses as `u64` (the proto's
/// type), `server_ts` is the server clock in microseconds, and any `|`-free
/// `client_id` lands whole in its slot. gRPC `UpdateLayer` refuses a `|` at
/// ingress (`validate_client_id`), so the storage API — the layer the reconcile
/// lives in, which validates no `client_id` — is the only writer of this value.
const CLIENT_ID_THE_PARSE_REJECTS: &str = "bhb-indoubt-pipe|writer";

/// b.izg: an in-doubt `EXEC` over a commit that landed must be recognised as
/// **ours** even when the landed payload is one `LayerNotification::parse`
/// rejects — the guard that ownership is decided on the stored payload bytes
/// (SDD §7.5.2).
///
/// A reconcile that re-parsed the entry and compared fields would fail to parse
/// this attempt's own entry, disown it, and answer `Aborted`. The retry's b.75i
/// commit-token check would then report the commit — so the stream and header
/// still show one application — but it ticks `commit_already_landed_total` and
/// drops the external-value releases the reconcile enqueues. That tick is what
/// this test discriminates on.
///
/// Staged with `ReplaceOutcomeAfterExec`, like
/// `layer_in_doubt_after_a_landed_commit_reports_the_commit`: the real pipeline
/// commits and only the verdict is discarded. The only difference between the
/// two tests is the `client_id`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_in_doubt_recognises_its_own_commit_when_the_client_id_breaks_the_notification_parse()
{
    const CLIENT_TS: u64 = 1_000;

    let LandedInDoubt {
        seq,
        server_ts,
        object_id,
        entries,
        header_seq,
        metrics,
    } = update_landed_in_doubt(
        CLIENT_ID_THE_PARSE_REJECTS,
        CLIENT_TS,
        "the commit landed, so the update must report success whether the reconcile or the \
         retry's commit-token check finds it",
    )
    .await;

    // ── The input really is one the parse rejects, written verbatim. ──
    let landed_payload = entries
        .iter()
        .find(|(id, _)| id == "2-0")
        .map(|(_, payload)| payload.clone())
        .unwrap_or_else(|| panic!("the commit must sit at 2-0; got {entries:?}"));
    assert_eq!(
        landed_payload,
        update_stream_payload(
            object_id,
            2,
            CLIENT_ID_THE_PARSE_REJECTS,
            CLIENT_TS,
            server_ts
        ),
        "the entry must carry the writer's client_id verbatim — an escaped or rejected `|` \
         would make this test exercise a payload the parse accepts"
    );
    assert!(
        LayerNotification::parse(&landed_payload).is_err(),
        "the precondition of this guard: the parse must reject the landed payload. If it now \
         accepts it, this test no longer tells a byte comparison from a field comparison — find \
         another writer-emittable value the parse rejects. Payload: {landed_payload:?}"
    );

    // ── Exactly one application. ──
    assert_eq!(seq, 2, "the reported sequence is the one that committed");
    assert_eq!(
        entries.len(),
        2,
        "the stream must carry exactly CreateLayer's 1-0 and this commit's 2-0. Got {entries:?}"
    );
    assert_eq!(header_seq, 2, "the header must have advanced exactly once");

    // ── The regression guard: the reconcile itself recognised the entry. ──
    assert_eq!(
        read_commit_already_landed(&metrics, "txn"),
        0,
        "the in-doubt reconcile must recognise its own entry by its bytes. A tick means it \
         disowned an entry the parse rejects — a field-wise comparison — answered `Aborted`, and \
         only the retry's commit-token check kept the delta from applying twice"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T11 — a store that REFUSES `CLIENT ID`: both identity layers disarm, and
//        every watched commit still lands
// ═════════════════════════════════════════════════════════════════════════════

/// The 100%-outage shape, reproduced against a real store rather than argued
/// about.
///
/// A proxied managed store (Azure Managed Redis, Redis Enterprise) may refuse
/// `CLIENT ID` outright. Both of b.bhb's identity layers are built on that
/// command, and both had to be written to degrade rather than fail closed —
/// because on such a store the failure is not intermittent, it is **every
/// connection, forever**:
///
/// - **L2** reads fred's backchannel connection-id map, which fred fills from a
///   handshake `CLIENT ID` whose failure `Connection::cache_connection_id`
///   swallows into `None`. The map is therefore permanently empty on a
///   perfectly healthy connection → `WatchFingerprint::Unnamed`. Read as "the
///   connection is gone" it would refuse **every** dispatch and take
///   `CreateSession` and every structural `UpdateLayer` to retry exhaustion.
///   The correct verdict is `DispatchUncomparable`: skip L2, dispatch, and tick
///   `fingerprint_unavailable`.
/// - **L3** queues `CLIENT ID` inside the `MULTI`. Queued blind on such a store,
///   the server rejects it at queue time and answers the `EXEC` with
///   `EXECABORT` — turning every watched commit into a terminal `INTERNAL`.
///   That is why the baseline probe exists: it discovers the refusal on a
///   throwaway transaction (`WitnessBaseline::Unavailable("exec_error")`), the
///   commit pipeline then queues no witness at all, and `witness_unavailable`
///   is ticked.
///
/// The fixture is a Valkey ACL user denied the whole `client` command family
/// (`-client`), which reproduces both refusals at once and for real — not a
/// fault injector, not a mocked reply. What it asserts is therefore the thing
/// an operator cares about: **on such a store the service still works**, and
/// says so in two metric cells rather than silently.
///
/// It is worth being explicit about what this scenario does NOT prove: with L3
/// disarmed a commit that ran on a connection which never held its `WATCH`
/// would go undetected here. That is the documented cost, and
/// `witness_unavailable` is the signal that the deployment is paying it.
///
/// The structural edit is made twice: on the WATCH path
/// ([`WATCH_PATH_BACKEND`]), which the `watch_voided` cells are about, and
/// then on the default `lua` backend's structural route, which holds no WATCH
/// and must commit on such a store all the same.
async fn scenario_client_id_denied_store_degrades_without_refusing(tv: &TestValkey) {
    const NAME: &str = "bhb-noclient-session";
    const BASE_PATH: &str = "/bhb-noclient";
    const LAYER: &str = "bhb-noclient-layer";
    const CLIENT: &str = "bhb-noclient-writer";

    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    // ── The mechanism, read directly: fred learned no connection id. ──
    // This is `WatchFingerprint::take`'s entire input, so an empty map here is
    // what makes `Unnamed` — and therefore `DispatchUncomparable` — the verdict
    // the commits below run under. Without it the scenario would be an
    // ordinary happy-path test.
    assert!(
        tv.client().pool.next().connection_ids().is_empty(),
        "fred must hold no CLIENT ID for a connection whose handshake `CLIENT ID` was refused \
         — that empty map is the `WatchFingerprint::Unnamed` state this scenario exists to \
         drive. A populated map means the ACL did not take effect on the pool's connections"
    );

    // ── CreateSession, the uniqueness-critical WATCHed path. ──
    let session_id = tokio::time::timeout(
        SCENARIO_BUDGET,
        SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(NAME, BASE_PATH, "bhb-user"),
    )
    .await
    .expect("CreateSession did not settle within the scenario budget")
    .expect(
        "CreateSession must SUCCEED on a store that refuses CLIENT ID. A failure here is the \
         100% outage the two disarm paths exist to prevent: fail-closed L2 refuses every \
         dispatch to retry exhaustion, and a blindly-queued L3 witness makes the server answer \
         EXECABORT — a terminal INTERNAL on every call, on every replica",
    );
    let holders = name_members(tv, BASE_PATH, NAME).await;
    assert_eq!(
        holders,
        vec![format!("{NAME}\u{0}{session_id}")],
        "exactly one member, and it must be the session this call created — the disarm must \
         cost detection, never correctness on an uncontended write"
    );

    // ── The structural UpdateLayer slow path, same question. ──
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone());
    let object_id = store
        .create_layer(
            &session_id,
            LAYER,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (seq, _) = tokio::time::timeout(
        SCENARIO_BUDGET,
        store.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(1, "NoClientSection")),
            CLIENT,
            1_000,
        ),
    )
    .await
    .expect("the structural update did not settle within the scenario budget")
    .expect(
        "a structural UpdateLayer on the WATCH path must SUCCEED on a store that refuses \
         CLIENT ID — same reasoning as CreateSession above; this is the path a structural \
         edit takes on a replica running the lua-watch-structural rollback, and on the \
         default backend for an uncacheable object",
    );
    assert_eq!(
        seq, 2,
        "the commit takes the sequence after CreateLayer's 1"
    );

    let entries = stream_payloads(tv, object_id).await;
    assert_stream_entry(&entries, 2, CLIENT);
    assert_eq!(
        entries.len(),
        2,
        "CreateLayer's 1-0 plus this commit's 2-0 — a disarmed witness must not make the write \
         path retry or double-apply; got {entries:?}"
    );

    // ── What the operator is told. ──
    for path in ["create_session", "update_layer"] {
        assert!(
            read_watch_voided(&metrics, path, "witness_unavailable") >= 1,
            "path={path:?} must report that the in-transaction witness (L3) could not be \
             armed. 0 means the baseline probe read this store as healthy and the commit \
             pipeline queued a CLIENT ID the server rejects — the EXECABORT outage — or that \
             it disarmed silently, which leaves an operator reading a flat \
             `committed_unwatched` with no way to tell `no incidents` from `the detector was \
             never switched on`"
        );
        assert!(
            read_watch_voided(&metrics, path, "fingerprint_unavailable") >= 1,
            "path={path:?} must report that the pre-dispatch identity check (L2) could not be \
             evaluated — same reasoning as its sibling above"
        );
        assert_eq!(
            read_watch_voided(&metrics, path, "aborted_pre_exec"),
            0,
            "path={path:?}: nothing disturbed a connection, so L2 must not have REFUSED a \
             single dispatch. A nonzero count here is the fail-closed reading of `Unnamed`, \
             and on this store it would fire on every attempt of every call, forever — the \
             outage this whole scenario exists to rule out"
        );
        assert_eq!(
            read_watch_voided(&metrics, path, "committed_unwatched"),
            0,
            "path={path:?}: nothing ran unwatched, and a disarmed witness must never accuse. \
             A tick here means a mutation's reply — an HSET's integer, say — was read as a \
             connection id and compared against a baseline that does not exist"
        );
    }
    assert_eq!(
        read_watch_voided(&metrics, "create_session", "compensated"),
        0,
        "nothing was accused, so nothing may have been undone — a compensation here would have \
         SREM'd the uniqueness claim of the perfectly good session asserted above"
    );

    // ── The same client-visible question on the default `lua` backend. ──
    // Its structural route plans on the structural cache and commits through
    // the update script, with no WATCH and no witness, so it ticks no
    // `watch_voided` cell; what it must do on this store is commit.
    let default_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(metrics.clone());
    let (seq, _) = tokio::time::timeout(
        SCENARIO_BUDGET,
        default_store.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(2, "NoClientSectionDefault")),
            CLIENT,
            1_001,
        ),
    )
    .await
    .expect("the default-backend structural update did not settle within the scenario budget")
    .expect(
        "a structural UpdateLayer on the default lua backend must SUCCEED on a store that \
         refuses CLIENT ID — this is the path a Kit client hits on every prim creation",
    );
    assert_eq!(
        seq, 3,
        "the commit takes the sequence after the WATCH leg's 2"
    );
    assert_eq!(
        sample_value(
            &render(&metrics),
            "live_edit_update_layer_attempts_histogram_count",
            &[("path", "structural_full")],
        ),
        Some(1.0),
        "fixture: the default-backend commit took the structural route (a cold start), not \
         the WATCH path"
    );
    let entries = stream_payloads(tv, object_id).await;
    assert_stream_entry(&entries, 3, CLIENT);
    assert_eq!(
        entries.len(),
        3,
        "CreateLayer's 1-0, the WATCH leg's 2-0 and this commit's 3-0; got {entries:?}"
    );
}

/// b.bhb: the service must keep working on a store that denies `CLIENT ID`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_that_refuses_client_id_disarms_both_layers_and_still_commits() {
    let raw_url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned());
    // Binding order is load-bearing: `tv` drops before `_guard`, so the
    // fixture's key cleanup runs before `ACL DELUSER` disconnects it.
    let Some((_guard, tv)) = acl_client_denied_fixture(raw_url, "redis://", POOL_SIZE).await else {
        return;
    };
    scenario_client_id_denied_store_degrades_without_refusing(&tv).await;
}

/// b.bhb on a real cluster: the same degrade, with the ACL user provisioned on
/// every primary.
///
/// Worth its own run rather than being assumed from the single-node twin: on a
/// cluster `WatchFingerprint::take` reaches its multi-connection arm, where
/// "the slot's owner has no id" (`Some(None)` → `Unnamed`) and "the routing
/// view could not be read" (`None` → the `AnyConnection` digest) are separate
/// branches. Only the first is correct on a `CLIENT ID`-denying store — the
/// digest would be computed over an empty map on every node and compare equal
/// to itself, silently re-arming an L2 check that is reading nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_a_store_that_refuses_client_id_disarms_both_layers_and_still_commits() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); run \
             `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster:` job run this"
        );
        return;
    };
    let Some((_guard, tv)) =
        acl_client_denied_fixture(cluster_url, "redis-cluster://", POOL_SIZE).await
    else {
        return;
    };
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment — a single-node \
         fixture smuggled in through this gate would exercise the centralized arm of \
         `WatchFingerprint::take` and report cluster coverage the suite does not have"
    );
    scenario_client_id_denied_store_degrades_without_refusing(&tv).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// In-doubt EXEC timeouts on live_edit_valkey_errors_total{kind="command_timeout"}
// ═════════════════════════════════════════════════════════════════════════════
//
// `WatchedTxn::exec` counts an in-doubt verdict carrying fred's `Timeout` once,
// on the pool its caller names, and no call site counts it again. These drive
// that count through both commit sites with the `SkipExecReturnTimedOut`
// override, and compare the WHOLE command-timeout row: a tick on the wrong pool,
// a missing tick, or a second tick from a call site's own arm all fail the same
// assertion. Each also pins the client-visible result as unchanged — counting
// steers nothing.

/// An `UpdateLayer` whose `EXEC` reply timed out counts exactly one
/// `command_timeout` on the txn pool, and the retry still commits once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_in_doubt_exec_timeout_counts_one_txn_command_timeout() {
    const LAYER: &str = "indoubt-timeout-layer";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let (session_id, object_id) = layer_fixture(&tv, "indoubt-timeout", LAYER).await;

    let overrides = override_queue(&[WatchedExecOverride::SkipExecReturnTimedOut]);
    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone())
        .with_watched_exec_override_for_test(overrides.clone());

    let (seq, _) = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(section_delta(1, "InDoubtSection")),
            "indoubt-timeout-writer",
            1_000,
        ),
    )
    .await
    .expect("the in-doubt update did not settle within the scenario budget")
    .expect("nothing was written, so the retry must commit — counting the timeout steers nothing");
    assert_override_consumed(&overrides);
    assert_eq!(
        seq, 2,
        "the retry commits at the sequence the first attempt computed"
    );

    assert_eq!(
        live_edit::storage::test_support::command_timeout_row(&metrics),
        live_edit::storage::test_support::command_timeouts_only_on("txn", 1),
        "one in-doubt EXEC timeout must count exactly once, on the txn pool"
    );
}

/// A `CreateSession` whose `EXEC` reply timed out counts exactly one
/// `command_timeout` on the txn pool, and the retry still creates the session
/// once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_in_doubt_exec_timeout_counts_one_txn_command_timeout() {
    const NAME: &str = "indoubt-timeout-create";
    const BASE_PATH: &str = "/indoubt-timeout-create";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let overrides = override_queue(&[WatchedExecOverride::SkipExecReturnTimedOut]);
    let disturbed = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_watched_exec_override_for_test(overrides.clone());

    let session_id = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.create(NAME, BASE_PATH, "bhb-user"),
    )
    .await
    .expect("the in-doubt create did not settle within the scenario budget")
    .expect("nothing was written, so the retry must create the session");
    assert_override_consumed(&overrides);
    assert_eq!(
        name_members(&tv, BASE_PATH, NAME).await,
        vec![format!("{NAME}\u{0}{session_id}")],
        "the retry creates exactly one session"
    );

    assert_eq!(
        live_edit::storage::test_support::command_timeout_row(&metrics),
        live_edit::storage::test_support::command_timeouts_only_on("txn", 1),
        "one in-doubt EXEC timeout must count exactly once, on the txn pool"
    );
}

/// One stalled `UpdateLayer` ticks twice, on two pools: once for the `EXEC`
/// reply that timed out (txn), and once for the reconciling `XRANGE` — the read
/// that asks whether the commit landed — timing out too (data). The retry then
/// commits once.
///
/// The `XRANGE` times out for real. The pre-dispatch hook parks a server-side
/// `BLPOP` on every data-pool connection (`ConnStall`), so the reconciling read
/// queues behind one and outlives the fixture's 5 s command timeout. The stalls
/// are released the moment the retry's first data-pool read reaches the server
/// — detected from the connections' byte counts, not from the metric — so that
/// read is answered well inside its deadline and adds nothing. The data tick
/// can therefore only be the `XRANGE`'s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn layer_in_doubt_exec_timeout_with_a_timed_out_reconcile_read_ticks_txn_and_data() {
    use live_edit::storage::test_support::{stall_pool, ConnStall};

    const LAYER: &str = "indoubt-double-tick-layer";

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let (session_id, object_id) = layer_fixture(&tv, "indoubt-double-tick", LAYER).await;

    let stalls: Arc<tokio::sync::Mutex<Vec<ConnStall>>> = Arc::default();
    let parked = Arc::new(AtomicBool::new(false));
    let hook: WatchWindowHook = {
        let (stalls, parked) = (stalls.clone(), parked.clone());
        let data = tv.client().pool.clone();
        let observer = tv.client().maintenance_pool.next().clone();
        let key_prefix = format!("{}:data-stall:", tv.key_schema().obj_header("stall"));
        Arc::new(move |_guarded: &Client| {
            let (stalls, parked) = (stalls.clone(), parked.clone());
            let (data, observer, key_prefix) = (data.clone(), observer.clone(), key_prefix.clone());
            async move {
                if parked.load(Ordering::SeqCst) {
                    return;
                }
                let mut held = stalls.lock().await;
                held.extend(stall_pool(&data, &observer, &key_prefix).await);
                parked.store(true, Ordering::SeqCst);
            }
            .boxed()
        })
    };
    let overrides = override_queue(&[WatchedExecOverride::SkipExecReturnTimedOut]);
    let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(WATCH_PATH_BACKEND)
        .with_metrics(metrics.clone())
        .with_pre_dispatch_hook_for_test(hook)
        .with_watched_exec_override_for_test(overrides.clone());

    // Bytes the server has read across every stalled data-pool connection.
    async fn data_bytes_in(stalls: &[ConnStall]) -> u64 {
        let mut total = 0;
        for stall in stalls {
            total += stall
                .net_in()
                .await
                .expect("a stalled data-pool connection must stay open");
        }
        total
    }
    // Keyed to the traffic, not to the metric under test: release once the
    // XRANGE has arrived AND a later data-pool command has too. Nothing else
    // uses the data pool in between, so that later command is the retry's —
    // sent only after the XRANGE timed out.
    let release_once_the_retry_arrives = async {
        let deadline = Instant::now() + SCENARIO_BUDGET;
        let wait_until = |what: &'static str| {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {what} to reach a stalled data-pool connection"
            );
        };
        while !parked.load(Ordering::SeqCst) {
            wait_until("the pre-dispatch hook to park the data pool");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let held = stalls.lock().await;
        let parked_bytes = data_bytes_in(&held).await;
        while data_bytes_in(&held).await == parked_bytes {
            wait_until("the reconciling XRANGE");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        // The XRANGE is one small frame; the next data-pool bytes cannot come
        // before its 5 s timeout, so this settle is slack, not a race.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let after_xrange = data_bytes_in(&held).await;
        while data_bytes_in(&held).await == after_xrange {
            wait_until("the retry's first data-pool read");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        drop(held);
        let mut released = Vec::new();
        for stall in stalls.lock().await.drain(..) {
            released.push(stall.release().await);
        }
        released
    };
    let (outcome, released) = tokio::join!(
        tokio::time::timeout(
            SCENARIO_BUDGET,
            disturbed.update_layer_relayed(
                &session_id,
                LAYER,
                object_id,
                Some(section_delta(1, "InDoubtSection")),
                "indoubt-double-tick-writer",
                1_000,
            ),
        ),
        release_once_the_retry_arrives,
    );
    let (seq, _) = outcome
        .expect("the in-doubt update did not settle within the scenario budget")
        .expect("the retry after the unanswered reconcile must commit");
    assert_override_consumed(&overrides);
    assert_eq!(
        seq, 2,
        "nothing landed on the first attempt, so the retry commits at 2"
    );
    assert_eq!(
        released.len(),
        POOL_SIZE as usize,
        "every data-pool connection was stalled"
    );
    for result in released {
        assert!(
            !matches!(result, Ok(Value::Null)) && result.is_ok(),
            "each stall must have held until its release; got {result:?}"
        );
    }

    let mut expected = live_edit::storage::test_support::command_timeouts_only_on("txn", 1);
    for (pool, n) in &mut expected {
        if *pool == "data" {
            *n = 1;
        }
    }
    assert_eq!(
        live_edit::storage::test_support::command_timeout_row(&metrics),
        expected,
        "one stalled UpdateLayer ticks exactly twice: the lost EXEC reply on txn and the \
         reconciling XRANGE on data"
    );
}

/// A session and one created layer, for the in-doubt timeout tests.
async fn layer_fixture(tv: &TestValkey, name: &str, layer: &str) -> (String, u64) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), "bhb-user")
        .await
        .expect("session creation must succeed");
    let object_id = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(
            &session_id,
            layer,
            "bhb-user",
            &minimal_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    (session_id, object_id)
}
