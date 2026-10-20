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

//! b.289 — cluster-mode follow-ups from b.ji8: the per-node state-proof streak
//! (sub-task 4) and the keyless-command audit's source-level fences
//! (sub-task 1).
//!
//! # What lives here
//!
//! | Test | What it pins | Where it runs |
//! |---|---|---|
//! | `seam_driven::streak_buckets::a_state_proof_verdict_through_the_unpinned_shim_lands_in_the_shared_bucket` | the shim has no key to hash, so its verdict has no node identity and must land in the shared fallback bucket | needs `test-fault-injection` |
//! | `seam_driven::streak_buckets::a_clean_checkout_collects_the_bucket_rather_than_zeroing_it` | a `Clean` **removes** the entry, so the map returns to empty and the `Clean` path stops paying for a routing-table clone | needs `test-fault-injection` |
//! | `seam_driven::streak_buckets::a_clean_between_verdicts_restarts_the_streak_and_prevents_the_latch` | the breaker counts *consecutive* verdicts: four state-proof verdicts with a `Clean` in the middle must not latch it | needs `test-fault-injection` |
//! | `seam_driven::cluster_a_clean_on_one_primary_does_not_rearm_the_breaker_on_another` | **the sub-task 4 defect itself**: node A latches while node B's `Clean`s do not re-arm it | + `LIVE_EDIT_TEST_CLUSTER_URL` |
//! | `seam_driven::cluster_a_clean_on_a_named_primary_collects_the_shared_fallback_bucket` | **the b.289 round-1 defect**: a `Clean` naming a real primary must still collect the shared bucket, which nothing else ever names | + `LIVE_EDIT_TEST_CLUSTER_URL` |
//! | [`no_zero_key_evalsha_call_site_survives_under_src_storage_or_checkpoint`] | a keyless `EVALSHA` is `ClusterHash::Random`, i.e. an arbitrary node | anywhere |
//! | [`no_bare_scan_call_site_survives_outside_an_is_clustered_dispatch`] | a bare `SCAN` family call walks one shard's slot range and silently misses the rest | anywhere |
//! | [`the_multi_call_site_inventory_matches_the_watch_guard_header`] | the `.multi()` inventory prose in `watch_guard.rs` / SDD §15.3 is countable, so it cannot rot | anywhere |
//! | [`the_cluster_gate_is_armed_when_ci_runs_the_test_cluster_job`] | a cluster-gated test that SKIPS inside the job that exists to run it is a silent hole | anywhere |
//! | [`the_test_cluster_ci_job_still_arms_both_gates`] | the `test-cluster:` job's two activation conditions (cluster URL + `test-fault-injection`) cannot be dropped silently | anywhere |
//! | [`the_ci_job_wiring_parser_reads_a_job_block`] | that fence is a reader over a config file; this is the reader's own self-test | anywhere |
//!
//! Everything that drives a verdict lives under `seam_driven`, compiled only
//! with `--features test-fault-injection` — the canned probe replies come from
//! `ProbeFaultInjector`, which does not exist without it. The three source-shape
//! fences and the two CI-wiring guards need neither the feature nor a Valkey and
//! run in every shape.
//!
//! The two source-shape fences that walk the tree (`EVALSHA` keys, bare `SCAN`)
//! walk **`src/storage/` and `src/checkpoint/` both**. The routing hazard they
//! police is fred's, not one module's: the checkpoint lane issues four
//! `EVALSHA`s of exactly the shape under audit, and its keyspace walk goes
//! through `orphan_sweep::scan_keyspace`, so a bare `SCAN` added there would be
//! as invisible to a storage-only walk as one added here. Neither fence edits
//! `src/checkpoint/`; they read it.
//!
//! # The two defects
//!
//! **Sub-task 4 — the breaker could not latch on a single misbehaving node.**
//! Before b.289 both recycle streaks were pool-wide and *any* `Clean` checkout
//! reset both. On a cluster where one primary disagrees with the probe
//! classifier — a managed proxy in front of one shard rendering the `DISCARD`
//! error differently, say — every checkout pinned to that primary produces a
//! state-proof verdict and recycles, while checkouts pinned to the other
//! primaries read `Clean` and reset the streak. The repair-ineffective breaker,
//! whose whole job is to bound that recycle loop, could therefore never reach
//! its threshold: the loop it exists to stop ran unbounded in exactly the
//! partial-node case, at `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (128)
//! re-entries per request. The fix keys the state-proof streak by cluster node
//! (`host:port`), with one shared `""` bucket for verdicts whose node cannot be
//! named; the unknown-state streak stays pool-wide, because the storm *it*
//! bounds is pool-wide.
//!
//! **Sub-task 1 — keyless commands route to an arbitrary node.** fred's
//! `ClusterHash::FirstKey` finds nothing to hash on a command with no key, and
//! `Router::route` falls through to `writers.values_mut().next()`. b.ji8 fixed
//! `UNWATCH`; the audit enumerated the rest. Two families came out of it needing
//! a *shape* fence rather than a fix, because both are correct today and both
//! would be silently wrong tomorrow — see each fence's own doc comment.
//!
//! # Why the interesting half is cluster-gated
//!
//! Every assertion about *which* bucket a verdict lands in is vacuous on one
//! endpoint: `probe_node_bucket` has no cached cluster state to consult, so it
//! returns the shared `""` bucket for every checkout, pinned or not, and the
//! whole contract degenerates to the pre-b.289 pool-wide one (which is the
//! documented and deliberate behaviour for a centralized or proxied
//! deployment). The single-endpoint tests here therefore pin the parts that
//! *are* observable — where an unnameable verdict goes, and that a `Clean`
//! collects rather than zeroes — and the asymmetry that motivated the change
//! lives in the one cluster-gated test, which skips locally and runs in CI's
//! `test-cluster:` job.
//!
//! **The cluster tests carry a second, silent activation condition**: they also
//! sit behind `--features test-fault-injection`, because the state-proof
//! verdicts are injected through the `ProbeFaultInjector` seam. A missing env
//! var prints a `SKIP:` line; a missing *feature* compiles the test out
//! entirely and the run reports green having covered nothing. `test-cluster:`
//! passes the feature, so CI is fine; a hand-rolled local invocation must pass
//! it explicitly.
//!
//! **Both conditions are themselves fenced**, because "the only job that runs
//! these tests quietly stopped running them" is otherwise indistinguishable from
//! a green pipeline. [`the_test_cluster_ci_job_still_arms_both_gates`] reads
//! `.gitlab-ci.yml` and asserts the job still sets `LIVE_EDIT_TEST_CLUSTER_URL`
//! and still passes `--features test-fault-injection`;
//! [`the_cluster_gate_is_armed_when_ci_runs_the_test_cluster_job`] runs *inside*
//! that job and turns a skip there into a hard failure. The first catches the
//! config edit, the second catches everything else that could leave the gate
//! shut at run time (a cluster that never came up, an env var lost between
//! shells).
//!
//! It carries no `#[ignore]`, deliberately. Its primitives are key-scoped —
//! `DISCARD` / `UNWATCH` probes pinned to two slots, plus canned frames answered
//! inside this process's own client — so nothing it does disturbs the ~600
//! sibling tests sharing the cluster, and the default self-gating shape (per
//! `docs-internal/test-writing-guide.md`) lets the unfiltered `test-cluster:`
//! run pick it up with no CI-side filterset to keep in sync. The `#[ignore]` +
//! `test-cluster-disruptive:` shape is for tests whose primitive disturbs the
//! *node* (`CLIENT PAUSE`, `CLUSTER SETSLOT`); this is not one.
//!
//! # Wait budgets vs. nextest's hang-killer
//!
//! No test here blocks on a background condition. Every body is a bounded
//! sequence of checkouts against a local Valkey; each dirty verdict pays the
//! production `TXN_RECYCLE_*` budgets, which `storage::client`'s own `const`
//! assertion bounds at 10 s, and the injected faults are canned frames that
//! short-circuit `probe_txn_slot` before either 650 ms probe deadline. Worst
//! case is the cluster test's nine checkouts — four dirty verdicts, three of
//! which recycle, because the fourth is the one that trips the breaker and has
//! its repair WITHHELD (which is exactly what the test's
//! `read_ineffective_total == 1` assertion pins): well inside the 40 s slow
//! mark, let alone the 120 s hard kill.

use std::path::{Path, PathBuf};

#[cfg(feature = "test-fault-injection")]
mod seam_driven {
    use std::sync::{Arc, Mutex};

    use tracing_subscriber::fmt::MakeWriter;

    use live_edit::observability::metrics::test_util::{render, sample_value};
    use live_edit::observability::metrics::Metrics;
    use live_edit::storage::test_support::TestValkey;
    use live_edit::storage::txn_checkout::{ProbeFault, ProbeFaultInjector};

    // ═════════════════════════════════════════════════════════════════════════════
    // Vocabulary
    // ═════════════════════════════════════════════════════════════════════════════

    /// How many consecutive state-proof verdicts **against one node** are repaired
    /// before the breaker withholds the repair, restated here.
    ///
    /// `TXN_STATE_PROOF_REPAIR_ATTEMPTS` is `pub(crate)` and an integration test is
    /// a separate crate, so this is a copy — held honest by the `repair_attempts`
    /// field the production ERROR carries, asserted in the cluster test below.
    const STATE_PROOF_REPAIR_ATTEMPTS: u32 = 3;

    /// The breaker's escalation ERROR, verbatim prefix from
    /// `ValkeyClient::acquire_txn_inner`.
    const REPAIR_INEFFECTIVE_ERROR: &str = "txn slot recycling is not clearing the dirty verdict";

    /// The shared fallback bucket's key: verdicts whose cluster node could not be
    /// named. Spelled as a named constant because `""` in an assertion reads like
    /// an oversight rather than the deliberate sentinel it is.
    const SHARED_BUCKET: &str = "";

    // ═════════════════════════════════════════════════════════════════════════════
    // Fixtures + helpers
    // ═════════════════════════════════════════════════════════════════════════════

    /// `MakeWriter` that captures every byte written to it into a shared buffer.
    ///
    /// Mirrors the per-module helpers in `txn_checkout_test.rs` and
    /// `b_ji8_tests.rs` — the established project pattern is to duplicate these few
    /// lines rather than hoist them into `tests/common`.
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

    /// Sum of every `live_edit_valkey_txn_repair_ineffective_total` cell.
    ///
    /// Summed rather than read per-cell because every test here asserts the counter
    /// is **flat**, and "no cell moved" is the claim — reading one cell would let a
    /// verdict misfiled under the sibling reason pass unnoticed.
    fn read_ineffective_total(metrics: &Metrics) -> u64 {
        let exposition = render(metrics);
        ["leaked_multi", "unexpected_reply"]
            .iter()
            .map(|reason| {
                sample_value(
                    &exposition,
                    "live_edit_valkey_txn_repair_ineffective_total",
                    &[("reason", *reason)],
                )
                .unwrap_or_else(|| {
                    panic!(
                        "live_edit_valkey_txn_repair_ineffective_total{{reason=\"{reason}\"}} is \
                         absent from the exposition. Both cells are pre-registered at zero in \
                         `Metrics::new`, so absence means this store never received the metrics \
                         handle and every count read from it is vacuous"
                    )
                }) as u64
            })
            .sum()
    }

    /// Every line of `logged` containing `message`.
    fn lines_with<'a>(logged: &'a str, message: &str) -> Vec<&'a str> {
        logged.lines().filter(|l| l.contains(message)).collect()
    }

    // ═════════════════════════════════════════════════════════════════════════════
    // A — per-node bucket semantics on one endpoint (b.289 sub-task 4)
    // ═════════════════════════════════════════════════════════════════════════════

    /// Seam-driven coverage of the streak map's observable contract.
    ///
    /// **Compiled only under `--features test-fault-injection`** (together with
    /// `test-support`, which every integration binary needs): the state-proof
    /// verdicts here are canned `DISCARD` replies pushed through
    /// `ProbeFaultInjector`, and neither that type nor
    /// `install_probe_fault_injector_for_test` exists without the feature, so this
    /// module is `#[cfg]`-ed out rather than failing to build. A missing feature
    /// makes these tests silently *absent*, not red.
    mod streak_buckets {
        use super::*;

        /// A `txn_pool` of exactly one connection, with the observability hooks and
        /// a fresh probe-fault injector installed.
        ///
        /// `pool_size = 1` is load-bearing: it guarantees the slot a checkout takes
        /// is the same fred `Client` every time, so a streak observed across
        /// checkouts is provably one slot's. The hooks matter for a second reason
        /// beyond the counters — `acquire_txn_inner` ticks through
        /// `metrics_snapshot()`, which is `None` until they are installed, so a
        /// fixture that skipped them would exercise the path and observe nothing.
        async fn fixture() -> (TestValkey, Arc<Metrics>, Arc<ProbeFaultInjector>) {
            let tv = TestValkey::setup_with_pool_size(1).await;
            let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
            tv.client()
                .install_observability_hooks(Arc::clone(&metrics))
                .await;
            let injector = Arc::new(ProbeFaultInjector::new());
            tv.client()
                .install_probe_fault_injector_for_test(Arc::clone(&injector));
            (tv, metrics, injector)
        }

        /// One pinned checkout, with everything it logs captured at DEBUG.
        ///
        /// DEBUG rather than INFO because a verdict whose repair the gate suppressed
        /// logs at `debug!` — an INFO capture could not tell "suppressed" from
        /// "never happened". The slot is released before returning, so the caller
        /// can check out again: at `pool_size = 1` a second acquisition while the
        /// first guard is alive is a deadlock, not a queue.
        async fn checkout_for(tv: &TestValkey, key: &str) -> String {
            let (writer, captured) = CapturingWriter::new();
            let subscriber = tracing_subscriber::fmt()
                .with_writer(writer)
                .with_env_filter("live_edit=debug")
                .with_ansi(false)
                .finish();
            let capture = tracing::subscriber::set_default(subscriber);

            let guard = tv.client().acquire_txn_for(key).await;
            drop(guard);

            let logged = {
                let buffer = captured.lock().expect("capture mutex is never poisoned");
                String::from_utf8_lossy(&buffer).into_owned()
            };
            drop(capture);
            logged
        }

        /// A verdict taken through the **unpinned** `acquire_txn()` shim has no node
        /// identity, so it must land in the shared fallback bucket.
        ///
        /// The shim passes `hash_key = None`, so there is nothing to derive a slot
        /// from and `probe_node_bucket` returns the shared bucket by its first arm —
        /// the same arm a centralized or proxied endpoint takes permanently. This is
        /// the arm that has to be *total*: a `None` that produced no bucket at all
        /// would drop the verdict and make the breaker unreachable for every
        /// deployment without a cluster, and a `None` that invented an identity
        /// would create a bucket no `Clean` could name and therefore never collect.
        ///
        /// `for_node_for_test` on a node that was never seen must read `0` rather
        /// than aliasing onto whatever bucket does exist — otherwise the cluster
        /// test's "node B's streak is zero" assertion could not distinguish
        /// "B is clean" from "B is not tracked".
        #[tokio::test]
        async fn a_state_proof_verdict_through_the_unpinned_shim_lands_in_the_shared_bucket() {
            let (tv, metrics, injector) = fixture().await;

            // A canned `+OK` from DISCARD is the `leaked_multi` verdict: state-proof,
            // so it is the class that feeds the per-node streak at all.
            injector.push_discard(ProbeFault::ok());
            let guard = tv.client().acquire_txn().await;
            drop(guard);
            assert_eq!(
                injector.remaining(),
                0,
                "precondition: the queued fault must have been consumed by the checkout, or the \
                 verdict asserted on below came from somewhere else"
            );

            assert_eq!(
                tv.client().txn_state_proof_streak_nodes_for_test(),
                vec![SHARED_BUCKET.to_string()],
                "a verdict with no key to hash must occupy the shared fallback bucket and no other. \
                 An empty map would mean the verdict was dropped — the breaker would then be \
                 unreachable through the shim and on every centralized deployment — and a named \
                 bucket would mean an identity was invented for a connection whose node is genuinely \
                 unknown, leaving an entry no later `Clean` can name"
            );
            assert_eq!(
                tv.client()
                    .txn_state_proof_streak_for_node_for_test(SHARED_BUCKET),
                1,
                "the shared bucket must hold this verdict's streak"
            );
            assert_eq!(
                tv.client()
                    .txn_state_proof_streak_for_node_for_test("10.255.255.255:6379"),
                0,
                "a node that has never produced a verdict must read zero rather than aliasing onto \
                 whatever bucket happens to exist. Without that, the cluster test's `node B is at \
                 zero` assertion could not tell `B is clean` from `B is not tracked at all`"
            );
            assert_eq!(
                tv.client().txn_state_proof_streak_for_test(),
                1,
                "the max-across-buckets accessor must agree with the single bucket that exists — it \
                 is what every pre-b.289 assertion in `txn_checkout_test.rs` still reads"
            );
            assert_eq!(
                read_ineffective_total(&metrics),
                0,
                "one verdict must not reach the breaker: its premise is that a single rebuild clears \
                 real residue"
            );
        }

        /// A `Clean` checkout must **remove** the bucket, not zero it.
        ///
        /// The distinction is invisible through `for_node_for_test` (a removed entry
        /// and a zeroed one both read `0`) and it is the whole reason
        /// `txn_state_proof_streak_nodes_for_test` exists. `acquire_txn_inner` skips
        /// naming the node on the `Clean` path while the map is empty, and naming it
        /// costs a clone of fred's routing table on a clustered client — on the path
        /// `ValkeyExternalValueStore::store_values` drives hardest in the whole
        /// process. A bucket left behind at zero would keep that guard open forever
        /// and put the clone on every clean checkout for the life of the pod.
        ///
        /// # What this does *not* prove
        ///
        /// On a single endpoint the bump and the `Clean` both resolve to the shared
        /// bucket, so this test cannot discriminate the b.289 round-1 defect
        /// (`clear` removing only its exact argument, leaving the shared bucket
        /// uncollectable when a *named* node's `Clean` follows an unnameable
        /// verdict). That asymmetry needs two primaries and is covered by
        /// [`super::cluster_a_clean_on_a_named_primary_collects_the_shared_fallback_bucket`],
        /// with a source-level fence on the symmetry itself in
        /// `src/storage/client.rs`
        /// (`tests::state_proof_streak_buckets::bump_and_clear_both_resolve_through_the_shared_bucket_key_helper`)
        /// — that one is the guard that runs in every shape.
        #[tokio::test]
        async fn a_clean_checkout_collects_the_bucket_rather_than_zeroing_it() {
            let (tv, _metrics, injector) = fixture().await;
            let key = tv.key_schema().obj_header("289-collect");

            injector.push_discard(ProbeFault::ok());
            let guard = tv.client().acquire_txn().await;
            drop(guard);
            assert_eq!(
                tv.client().txn_state_proof_streak_nodes_for_test(),
                vec![SHARED_BUCKET.to_string()],
                "precondition: the seeded verdict must have created a bucket, or the removal below \
                 is asserting on an absence that was always there"
            );

            // Queue drained, so this checkout takes the real probe and reads Clean.
            let quiet = checkout_for(&tv, &key).await;

            assert_eq!(
                tv.client().txn_state_proof_streak_nodes_for_test(),
                Vec::<String>::new(),
                "a Clean checkout must COLLECT the bucket. Zeroing it instead reads identically \
                 through `for_node_for_test`, but leaves `acquire_txn_inner`'s non-empty-map guard \
                 open forever — so every Clean checkout in the process, on the hottest path there is, \
                 pays for a `cached_cluster_state()` routing-table clone it has nothing to do with. \
                 Captured output: {quiet:?}"
            );
        }

        /// The breaker counts **consecutive** verdicts, and a `Clean` in the middle
        /// restarts the count.
        ///
        /// Four state-proof verdicts in a row latch the breaker: the repair is
        /// admitted for streaks 1..=[`STATE_PROOF_REPAIR_ATTEMPTS`] and withheld
        /// from the next. That latch is already pinned by
        /// `txn_checkout_test.rs::seam_driven::the_breaker_withholds_the_repair_once_rebuilding_stops_clearing_the_verdict`,
        /// which is deliberately not duplicated here. What this adds is the other
        /// half of the same contract: the *same four verdicts* with one `Clean`
        /// between the second and the third must not latch anything, because the
        /// evidence the breaker acts on is "several rebuilds in a row failed to
        /// change the verdict" and a `Clean` is direct proof that a rebuild did.
        ///
        /// Without it the breaker would be a latch on a *cumulative* count, and a
        /// pod that saw four unrelated borrowers die mid-transaction over an hour —
        /// each one real residue, each one genuinely repaired — would stop repairing
        /// for the rest of its life.
        #[tokio::test]
        async fn a_clean_between_verdicts_restarts_the_streak_and_prevents_the_latch() {
            let (tv, metrics, injector) = fixture().await;
            let key = tv.key_schema().obj_header("289-restart");

            // The baseline runs BEFORE the queue is armed: it must take the real
            // probe and come back Clean, or it would consume the first fault and
            // shift every streak below by one.
            let baseline = checkout_for(&tv, &key).await;
            assert!(
                lines_with(&baseline, REPAIR_INEFFECTIVE_ERROR).is_empty(),
                "precondition: the baseline checkout must be quiet. Captured output: {baseline:?}"
            );

            // Verdicts 1 and 2 — inside the repaired window either way.
            for expected_streak in 1..=2u32 {
                injector.push_discard(ProbeFault::ok());
                checkout_for(&tv, &key).await;
                assert_eq!(
                    tv.client().txn_state_proof_streak_for_test(),
                    expected_streak,
                    "verdict {expected_streak} must advance the streak"
                );
            }

            // The Clean. This is the whole experiment.
            let quiet = checkout_for(&tv, &key).await;
            assert_eq!(
                tv.client().txn_state_proof_streak_for_test(),
                0,
                "the interleaved Clean must reset the streak — it is positive evidence that a rebuilt \
                 connection DOES answer this endpoint's replies the way the classifier expects, which \
                 is the one thing that retires the breaker's conclusion. Captured output: {quiet:?}"
            );

            // Verdicts 3 and 4 overall — but 1 and 2 of a fresh streak. On a
            // cumulative count the fourth would trip the breaker.
            for expected_streak in 1..=2u32 {
                injector.push_discard(ProbeFault::ok());
                let logged = checkout_for(&tv, &key).await;
                assert_eq!(
                    tv.client().txn_state_proof_streak_for_test(),
                    expected_streak,
                    "the post-Clean verdicts must count from 1 again, not continue from 2"
                );
                assert!(
                    lines_with(&logged, REPAIR_INEFFECTIVE_ERROR).is_empty(),
                    "no escalation may be emitted at streak {expected_streak}. Captured output: \
                     {logged:?}"
                );
            }

            assert_eq!(
                read_ineffective_total(&metrics),
                0,
                "four state-proof verdicts with a Clean between the second and the third must leave \
                 live_edit_valkey_txn_repair_ineffective_total flat, even though four CONSECUTIVE \
                 verdicts would latch it (pinned in txn_checkout_test.rs). The breaker's evidence is \
                 `several rebuilds in a row did not change the verdict`; a Clean is direct proof that \
                 one did, so counting cumulatively would let a pod that saw four unrelated \
                 mid-transaction deaths over an hour stop repairing for the rest of its life"
            );
        }
    }

    // ═════════════════════════════════════════════════════════════════════════════
    // B — the sub-task 4 defect itself: per-node latching (cluster-gated)
    // ═════════════════════════════════════════════════════════════════════════════

    /// The cluster gate. `None` means "not configured — the caller must return",
    /// and a `SKIP:` line has already been printed saying why.
    ///
    /// Same shape as `b_ji8_tests.rs::cluster_fixture` and
    /// `cluster_subscriber_test.rs`'s: `TestValkey::setup_with_url` panics if the
    /// cluster is unreachable, so a configured-but-down cluster fails loudly rather
    /// than skipping.
    async fn cluster_fixture(valkey_pool_size: u32) -> Option<TestValkey> {
        // The var name and the scheme come from the top-level constants the CI
        // wiring guards in section D read, so the gate and the guard that checks
        // the gate is armed cannot drift apart.
        let var = super::CLUSTER_URL_VAR;
        let scheme = super::CLUSTER_URL_SCHEME;
        let Ok(cluster_url) = std::env::var(var) else {
            eprintln!(
                "SKIP: {var} not set (no multi-shard cluster configured); run \
                 `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster:` job run this. \
                 b.289 sub-task 4's defect is a streak reset by a Clean on a DIFFERENT primary, which \
                 a single endpoint cannot express at all"
            );
            return None;
        };
        if !cluster_url.starts_with(scheme) {
            eprintln!(
                "SKIP: {var} is not a plaintext {scheme} URL (got a URL with a different scheme)"
            );
            return None;
        }

        let tv = TestValkey::setup_with_url(cluster_url, valkey_pool_size).await;
        assert!(
            fred::interfaces::ClientLike::is_clustered(tv.client().pool.next()),
            "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (use \
             `redis-cluster://...`). The assertion below is about a per-node streak map that only \
             has more than one bucket in cluster mode, so a single-node fixture smuggled in through \
             this gate would pass vacuously and report cluster coverage the suite does not have"
        );
        Some(tv)
    }

    /// Node A's state-proof streak must survive node B's `Clean` checkouts, latch
    /// the breaker, and then be re-armed by a `Clean` against A itself.
    ///
    /// # The defect, exactly
    ///
    /// Pre-b.289 both streaks were pool-wide and **any** `Clean` reset them. Drive
    /// state-proof verdicts through checkouts pinned to primary A while ordinary
    /// traffic checks out slots on primary B, and A's streak is reset by every one
    /// of B's clean checkouts. It never reaches
    /// [`STATE_PROOF_REPAIR_ATTEMPTS`] + 1, the repair-ineffective breaker never
    /// latches, and every checkout pinned to A quits and reconnects forever — the
    /// unbounded recycle loop the breaker exists to stop, running unbounded in
    /// precisely the partial-node case, at `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS`
    /// (128) re-entries per request.
    ///
    /// # Why this cannot be a single-node test
    ///
    /// `probe_node_bucket` derives its bucket from the client's *cached cluster
    /// state*, which a centralized client does not have: every verdict on one
    /// endpoint lands in the shared fallback bucket, which any `Clean` re-arms —
    /// deliberately, since a single endpoint has exactly one node to be wrong about.
    /// So the interleaving that reproduces the defect is inexpressible without two
    /// primaries, and a single-node run of this scenario passes on the buggy code.
    ///
    /// # How the two keys are chosen
    ///
    /// By searching hash tags until two land on different primaries, and asserting
    /// the search succeeded. Hardcoding two tags would bind the test to the slot
    /// layout of `scripts/dev-valkey.sh cluster-up`'s three-shard topology; the
    /// search works on any cluster with ≥2 primaries, including a managed one.
    ///
    /// # Runtime
    ///
    /// Nine checkouts, carrying four dirty verdicts of which THREE recycle: the
    /// fourth verdict is the one that trips the breaker, and a tripped breaker
    /// withholds the repair (that is what the `read_ineffective_total == 1`
    /// assertion below records), so it pays no rebuild. Each of the three that
    /// do recycle pays one real `quit()` + `init()` against a local cluster; the
    /// canned `+OK` short-circuits `probe_txn_slot` before either 650 ms probe
    /// deadline, so nothing here waits on a timeout.
    #[tokio::test]
    async fn cluster_a_clean_on_one_primary_does_not_rearm_the_breaker_on_another() {
        use fred::types::ClusterHash;

        let Some(tv) = cluster_fixture(1).await else {
            return;
        };
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let injector = Arc::new(ProbeFaultInjector::new());
        tv.client()
            .install_probe_fault_injector_for_test(Arc::clone(&injector));

        // The primary that owns `key`'s slot, named exactly the way
        // `probe_node_bucket` names it: `ClusterHash::from(&str)` →
        // `Custom(redis_keyslot(key))` → `get_server(slot).to_string()`. Re-deriving
        // it any other way would let the test and the production bucket key drift
        // and still agree.
        let owner_of = |key: &str| -> String {
            let ClusterHash::Custom(slot) = ClusterHash::from(key) else {
                panic!(
                    "ClusterHash::from(&str) must yield Custom(slot); `probe_node_bucket` relies on it"
                )
            };
            fred::interfaces::ClusterInterface::cached_cluster_state(tv.client().pool.next())
                .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
                .get_server(slot)
                .unwrap_or_else(|| {
                    panic!("no primary owns slot {slot} in the cached routing table")
                })
                .to_string()
        };

        // Two keys on two different primaries.
        let key_a = tv.key_schema().obj_header("289a");
        let node_a = owner_of(&key_a);
        let (key_b, node_b) = (0..64)
            .map(|i| tv.key_schema().obj_header(&format!("289b{i}")))
            .find_map(|key| {
                let owner = owner_of(&key);
                (owner != node_a).then_some((key, owner))
            })
            .expect(
                "this cluster has ≥2 primaries but 64 candidate hash tags all landed on the same one, \
                 which is statistically impossible for CRC16 over a balanced slot map — the routing \
                 view or the key derivation is wrong",
            );
        assert_ne!(
            node_a, node_b,
            "the two keys must be owned by DIFFERENT primaries or there is no cross-node interleaving \
             to test and this whole test degenerates to the single-node case"
        );

        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_env_filter("live_edit=debug")
            .with_ansi(false)
            .finish();
        let capture = tracing::subscriber::set_default(subscriber);

        // Four state-proof verdicts on A, each followed by a Clean on B. On the
        // pre-b.289 pool-wide streak, B's Clean resets A's count every time and the
        // breaker never latches.
        for verdict in 1..=STATE_PROOF_REPAIR_ATTEMPTS + 1 {
            injector.push_discard(ProbeFault::ok());
            let guard = tv.client().acquire_txn_for(&key_a).await;
            drop(guard);
            assert_eq!(
                injector.remaining(),
                0,
                "verdict {verdict}: the queued fault must be consumed by the checkout pinned to A"
            );
            assert_eq!(
                tv.client()
                    .txn_state_proof_streak_for_node_for_test(&node_a),
                verdict,
                "verdict {verdict}: node A's own bucket must advance. This is the count the breaker \
                 reads, and pre-b.289 it was a single pool-wide number that B's Clean below reset on \
                 every iteration"
            );

            // Ordinary healthy traffic on the other primary. The queue is empty, so
            // this takes the real probe and reads Clean.
            let guard = tv.client().acquire_txn_for(&key_b).await;
            drop(guard);
            assert_eq!(
                tv.client()
                    .txn_state_proof_streak_for_node_for_test(&node_a),
                verdict,
                "verdict {verdict}: a Clean checkout against primary B must NOT re-arm primary A. \
                 THIS is the b.289 sub-task 4 defect: with a pool-wide streak, one healthy primary's \
                 ordinary traffic held the count at zero forever, so the breaker could never latch on \
                 a single misbehaving primary and every checkout pinned to it recycled without bound"
            );
        }

        let logged = {
            let buffer = captured.lock().expect("capture mutex is never poisoned");
            String::from_utf8_lossy(&buffer).into_owned()
        };
        drop(capture);

        assert_eq!(
            tv.client()
                .txn_state_proof_streak_for_node_for_test(&node_b),
            0,
            "node B produced only Clean verdicts, so it must carry no streak of its own. A non-zero \
             count here would mean the bucket key is not the node the probe was pinned to"
        );
        let mut nodes = tv.client().txn_state_proof_streak_nodes_for_test();
        nodes.sort();
        assert_eq!(
            nodes,
            vec![node_a.clone()],
            "exactly one bucket must exist, named for primary A. The shared fallback bucket must NOT \
             be among them: every checkout here was pinned to a key whose slot the cached routing view \
             covers, so every verdict had a nameable node — a `\"\"` entry would mean the resolver \
             silently gave up and folded a nameable verdict into the bucket that any Clean anywhere \
             re-arms, which is the pool-wide behaviour by another route"
        );

        assert_eq!(
            read_ineffective_total(&metrics),
            1,
            "the breaker must latch on the verdict after {STATE_PROOF_REPAIR_ATTEMPTS} repaired ones \
             — that verdict is the third re-observation from a freshly rebuilt connection, which is \
             the evidence that rebuilding is not what is wrong. Pre-b.289 this counter stayed at zero \
             forever in this scenario. Captured output: {logged:?}"
        );
        let errors = lines_with(&logged, REPAIR_INEFFECTIVE_ERROR);
        assert_eq!(
            errors.len(),
            1,
            "the escalation ERROR must be emitted exactly once, on the verdict that trips the breaker. \
             Captured output: {logged:?}"
        );
        assert!(
            errors[0].contains(&format!("node=\"{node_a}\"")),
            "the escalation must NAME primary A. The breaker's scope is now per node and node \
             identity is deliberately not a metric label (unbounded cardinality over a deployment's \
             lifetime), so this log field is the only place an operator learns WHICH primary has had \
             checkout hygiene switched off. Event: {:?}",
            errors[0]
        );
        assert!(
            errors[0].contains(&format!("repair_attempts={STATE_PROOF_REPAIR_ATTEMPTS}")),
            "the escalation must carry the attempt budget it decided on — this is what keeps this \
             module's local copy of the constant from drifting from production's. Event: {:?}",
            errors[0]
        );

        // Re-arm: a Clean against A itself — the one thing that retires the
        // conclusion — must collect A's bucket.
        let guard = tv.client().acquire_txn_for(&key_a).await;
        drop(guard);
        assert_eq!(
            tv.client().txn_state_proof_streak_nodes_for_test(),
            Vec::<String>::new(),
            "a Clean checkout against primary A must collect A's bucket, so the very next state-proof \
             verdict against A is repaired again. The breaker is a conclusion about evidence, not a \
             one-way latch: a transient burst on one primary must not disable that primary's checkout \
             hygiene for the life of the process"
        );
    }

    /// A `Clean` checkout against a **named** primary must still collect the
    /// **shared** bucket.
    ///
    /// # The defect this is the behavioural test for
    ///
    /// b.289's round-1 implementation had `clear_state_proof_streak` remove only the
    /// bucket its argument resolved to. That is correct for a named node and wrong
    /// for the shared `""` one, which is where every verdict whose node could not be
    /// named lands — and there is no such thing as a `Clean` *against* `""` to pair
    /// with it, because the whole point of the fallback is that those verdicts have
    /// no node identity. So three unnameable verdicts latched the breaker on a
    /// bucket that every subsequent `Clean` walked straight past, permanently, for
    /// the life of the process.
    ///
    /// That is not a corner case on a real cluster: `probe_node_bucket` returns the
    /// shared bucket whenever fred's cached routing view is momentarily absent or
    /// does not cover the slot — i.e. during exactly the reshard and reconnect
    /// windows that produce probe-slot anomalies in the first place.
    ///
    /// # Why it is staged through the unpinned shim
    ///
    /// The shim passes `hash_key = None`, which is the one way to produce an
    /// unnameable verdict **deterministically** on a healthy cluster: waiting for
    /// fred's routing view to go absent is not stageable. The verdict therefore
    /// lands in `""` while the `Clean` that follows is pinned to a real key and
    /// names a real primary — the exact asymmetry the round-1 code got wrong.
    ///
    /// # Why this cannot be a single-node test
    ///
    /// On one endpoint the `Clean`'s `probe_node_bucket` also returns `""`, so the
    /// round-1 code removes the right entry by coincidence and the whole scenario
    /// passes on the buggy implementation. Two primaries are what make the `Clean`'s
    /// node nameable and the bump's node not. The source-level symmetry fence in
    /// `src/storage/client.rs`
    /// (`tests::state_proof_streak_buckets::bump_and_clear_both_resolve_through_the_shared_bucket_key_helper`)
    /// is the guard that runs everywhere; this is the behavioural proof, and it runs
    /// only in CI's `test-cluster:` job.
    ///
    /// # Phase 1 exists to close the last way this could pass on the buggy code
    ///
    /// The discriminating power of the experiment rests entirely on the final
    /// `Clean` resolving to a **named** primary. If it instead resolved to `""` —
    /// which `probe_node_bucket` returns whenever fred's cached routing view is
    /// momentarily absent, the very condition this test is *about* — then the
    /// round-1 `remove(exact_key_only)` would remove `""` and the map would come
    /// back empty on the buggy implementation too. A green run would then mean
    /// "the routing view blinked", not "the fix works", and nothing in the test
    /// could tell the two apart.
    ///
    /// Phase 1 makes that condition an assertion instead of an assumption: one
    /// state-proof verdict pinned to the same key must land in a bucket named
    /// `owner_of(key)` — a real `host:port`, which is unequal to `""` by
    /// construction — and a pinned `Clean` must then collect it. That is a
    /// scenario the round-1 code handles correctly, so phase 1 is not the
    /// discriminator; it is the *precondition* for phase 2 being one. If the
    /// routing view is absent, phase 1 fails loudly at the bucket-name assertion
    /// rather than letting phase 2 pass vacuously.
    #[tokio::test]
    async fn cluster_a_clean_on_a_named_primary_collects_the_shared_fallback_bucket() {
        use fred::types::ClusterHash;

        let Some(tv) = cluster_fixture(1).await else {
            return;
        };
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let injector = Arc::new(ProbeFaultInjector::new());
        tv.client()
            .install_probe_fault_injector_for_test(Arc::clone(&injector));

        let key = tv.key_schema().obj_header("289-shared");

        // The primary that owns `key`'s slot, named exactly the way
        // `probe_node_bucket` names it: `ClusterHash::from(&str)` →
        // `Custom(redis_keyslot(key))` → `get_server(slot).to_string()`. Token
        // for token the same derivation the sibling cluster test uses — deriving
        // it any other way would let the test and the production bucket key
        // drift and still agree.
        let owner_of = |key: &str| -> String {
            let ClusterHash::Custom(slot) = ClusterHash::from(key) else {
                panic!(
                    "ClusterHash::from(&str) must yield Custom(slot); `probe_node_bucket` relies on it"
                )
            };
            fred::interfaces::ClusterInterface::cached_cluster_state(tv.client().pool.next())
                .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
                .get_server(slot)
                .unwrap_or_else(|| {
                    panic!("no primary owns slot {slot} in the cached routing table")
                })
                .to_string()
        };
        let owner = owner_of(&key);
        assert_ne!(
            owner, SHARED_BUCKET,
            "precondition: the key this test pins its Clean to must resolve to a REAL primary. An \
             empty owner would mean the cached routing view does not cover its slot, and every \
             assertion below would be about the shared bucket talking to itself"
        );

        // ── Phase 1: the named-node round trip, which the round-1 code got right.
        //
        // Its job is to prove that a checkout pinned to `key` names `owner` in
        // BOTH directions — bump and Clean — on this cluster, right now. That is
        // what makes phase 2's `Clean` a Clean against a named node rather than
        // an accidental second Clean against `""`.
        injector.push_discard(ProbeFault::ok());
        let guard = tv.client().acquire_txn_for(&key).await;
        drop(guard);
        assert_eq!(
            injector.remaining(),
            0,
            "precondition: the queued fault must have been consumed by the pinned checkout"
        );
        assert_eq!(
            tv.client().txn_state_proof_streak_nodes_for_test(),
            vec![owner.clone()],
            "phase 1: a state-proof verdict on a checkout pinned to `{key}` must land in a bucket \
             named for the primary that owns its slot ({owner}), not in the shared `\"\"` bucket. A \
             `\"\"` here means fred's cached routing view is absent or does not cover the slot — in \
             which case the Clean in phase 2 would ALSO resolve to `\"\"`, the round-1 \
             `remove(exact-key-only)` would remove it by coincidence, and phase 2 would pass on the \
             buggy implementation. This assertion is what stops that from being a silent green"
        );

        let guard = tv.client().acquire_txn_for(&key).await;
        drop(guard);
        assert_eq!(
            tv.client().txn_state_proof_streak_nodes_for_test(),
            Vec::<String>::new(),
            "phase 1: a Clean pinned to the same key must collect the bucket it just created. This \
             is the case the b.289 round-1 code handled CORRECTLY (the Clean's node and the bump's \
             node are the same), so a failure here is not the round-1 defect — it is the pinned \
             Clean failing to name `{owner}` at all, which would leave phase 2 unable to \
             discriminate anything"
        );

        // ── Phase 2: the asymmetry itself.
        //
        // An unnameable state-proof verdict: the shim has no key, so the probe
        // cannot be attributed to a primary.
        injector.push_discard(ProbeFault::ok());
        let guard = tv.client().acquire_txn().await;
        drop(guard);
        assert_eq!(
            tv.client().txn_state_proof_streak_nodes_for_test(),
            vec![SHARED_BUCKET.to_string()],
            "phase 2 precondition: a verdict taken through the unpinned shim has no node to name \
             even on a cluster, so it must land in the shared fallback bucket. If it landed \
             anywhere else this test is not staging the asymmetry it exists for"
        );

        // A Clean checkout pinned to the same real key — which phase 1 just
        // proved names `{owner}` in both directions on this cluster.
        let guard = tv.client().acquire_txn_for(&key).await;
        drop(guard);

        assert_eq!(
            tv.client().txn_state_proof_streak_nodes_for_test(),
            Vec::<String>::new(),
            "a Clean checkout must collect the shared fallback bucket even though it named a \
             DIFFERENT (real) primary ({owner}). Removing only the bucket the Clean's own node \
             resolves to — the b.289 round-1 defect — leaves the shared bucket uncollectable: \
             nothing ever names it, so three unnameable verdicts during a routing-view rebuild \
             latch the repair-ineffective breaker for the life of the process, and every later \
             Clean walks past it. Phase 1 above already established that a checkout pinned to \
             this key resolves to {owner} and not to the shared bucket, so this failure cannot be \
             explained by an absent routing view"
        );
        assert_eq!(
            read_ineffective_total(&metrics),
            0,
            "neither verdict may reach the breaker: this test stages two state-proof verdicts, \
             each of which is the FIRST in its own bucket ({owner} in phase 1, the shared bucket \
             in phase 2) and each of which is then collected by a Clean. The breaker withholds \
             the repair only after {STATE_PROOF_REPAIR_ATTEMPTS} CONSECUTIVE verdicts against one \
             bucket, so a tick here would mean the two buckets are being counted as one"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// C — source-shape fences from the keyless-command audit (b.289 sub-task 1)
// ═════════════════════════════════════════════════════════════════════════════
//
// Shape guards in the idiom of `b_ji8_tests.rs::no_bare_unwatch_call_survives_outside_watch_guard`
// and `txn_checkout_test.rs::no_unpinned_acquire_txn_call_site_survives_under_src`,
// and they exist for the same reason those do: the damage is deferred and
// misattributed. Every command policed here works perfectly on the single-node
// fixture the whole suite runs against, and on a cluster it silently talks to
// the wrong node — surfacing later, on some unrelated request, as missing keys
// or a CROSSSLOT cascade with nothing pointing at the line responsible.

/// Directory-relative paths of every `.rs` file under `dir`.
///
/// A walk rather than an enumerated list, for the same reason the sibling fences
/// walk: a NEW module reaching for one of these shapes is exactly the regression
/// worth catching, and a hand-maintained table would have to be remembered.
pub(crate) fn rust_sources_under(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
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
/// A copy of `txn_checkout_test.rs::strip_comments` (private to that module;
/// the project pattern is to duplicate these few lines rather than hoist them),
/// with the same deliberate limits, all in the conservative direction:
///
/// - **string literals are kept**, so a needle inside a `"…"` still counts as a
///   hit. Wrong in the strictest sense, but the failure mode is a visible red
///   test naming the line, whereas stripping strings risks a mis-tracked quote
///   silently swallowing real code. No scanned tree currently spells any of
///   these needles inside a string.
/// - **nested** `/* /* … */ */` is not modelled and `r#"…"#` raw strings are
///   tracked only as far as their first inner `"`. Both err toward *more* text
///   surviving into the match.
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

/// `source` with every comment replaced by nothing, line structure preserved.
///
/// Line structure is preserved (rather than the whole file being collapsed) so
/// a violation can be reported with a usable line number and so the lookback
/// window in the SCAN fence means something.
pub(crate) fn code_lines(source: &str) -> Vec<String> {
    let mut in_block = false;
    source
        .lines()
        .map(|line| strip_comments(line, &mut in_block))
        .collect()
}

/// The argument list of the call whose name ends at `after_name` in `code`, as
/// its top-level comma-separated argument expressions.
///
/// `None` when the text after the name is not a call at all. Handles a
/// turbofish (`::<Vec<String>, _, _>`) by balancing angle brackets before
/// looking for the opening paren, and balances `()`, `[]` and `{}` — plus
/// string literals — while splitting, so a multi-line call with a `vec![…]`
/// argument comes back as one argument rather than several.
///
/// Angle brackets are balanced only in the turbofish, never inside the argument
/// list: a comparison operator there would be indistinguishable from a generic,
/// and none of the scanned call sites passes a type argument containing a comma.
/// The self-test below pins the shapes that do occur.
///
/// `after_name` is a **byte** offset — the kind `str::find` returns — and the
/// scan runs over bytes rather than `char`s for exactly that reason. Every
/// delimiter this parser cares about is ASCII, and no ASCII byte can appear
/// inside a multi-byte UTF-8 sequence, so byte-wise scanning is safe and the
/// two index spaces cannot desync. (Collecting `code.chars()` and then indexing
/// with a byte offset is the trap: the scanned source keeps its string
/// literals, and this repo's error messages are full of `—` and `…`, so the two
/// offsets diverge well before the first call site.)
pub(crate) fn call_arguments(code: &str, after_name: usize) -> Option<Vec<String>> {
    let bytes = code.as_bytes();
    let mut i = after_name;

    let skip_ws = |i: &mut usize| {
        while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
            *i += 1;
        }
    };
    skip_ws(&mut i);

    // Optional turbofish.
    if i + 2 < bytes.len() && bytes[i] == b':' && bytes[i + 1] == b':' && bytes[i + 2] == b'<' {
        i += 2;
        let mut depth = 0usize;
        while i < bytes.len() {
            match bytes[i] {
                b'<' => depth += 1,
                b'>' => {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        skip_ws(&mut i);
    }

    if i >= bytes.len() || bytes[i] != b'(' {
        return None;
    }
    i += 1;

    /// Collapse an argument's whitespace (it may span several lines) and drop it
    /// if it is only a trailing comma's empty tail.
    fn push_arg(args: &mut Vec<String>, current: &mut Vec<u8>) {
        let text = String::from_utf8_lossy(current);
        let trimmed = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if !trimmed.is_empty() {
            args.push(trimmed);
        }
        current.clear();
    }

    let mut args: Vec<String> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            current.push(c);
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => {
                in_string = true;
                current.push(c);
            }
            b'(' | b'[' | b'{' => {
                depth += 1;
                current.push(c);
            }
            b')' if depth == 0 => {
                push_arg(&mut args, &mut current);
                return Some(args);
            }
            b')' | b']' | b'}' => {
                depth -= 1;
                current.push(c);
            }
            b',' if depth == 0 => push_arg(&mut args, &mut current),
            _ => current.push(c),
        }
        i += 1;
    }
    None
}

/// The call-site form of the shared EVALSHA helper.
const EVALSHA_NEEDLE: &str = "evalsha_with_reload";

/// A file the walk must reach, used as a non-vacuity anchor.
///
/// **Not** where [`EVALSHA_NEEDLE`] is defined: `evalsha_with_reload` is a
/// method on fred's `Script` type, so its definition is in the dependency, not
/// in this tree at all. `src/storage/lua.rs` is where this repo's `Script`
/// values are declared and the first call site the audit found, which makes it
/// a stable landmark for "the walk reached the storage lane" — nothing more. If
/// it disappears the walk itself is what has gone wrong; a rename of fred's
/// method is caught instead by the site-count floor below.
const EVALSHA_WALK_ANCHOR_FILE: &str = "src/storage/lua.rs";

/// The same kind of landmark for the checkpoint lane (see
/// [`EVALSHA_WALK_ANCHOR_FILE`]) — `src/checkpoint/session.rs` holds two of the
/// four checkpoint-lane call sites.
const EVALSHA_CHECKPOINT_ANCHOR_FILE: &str = "src/checkpoint/session.rs";

/// `evalsha_with_reload` call sites under `src/storage/` when this fence last
/// counted them.
///
/// Raised 18 → 19 by b.rej, which moved `store_values`'s per-hash blob write off
/// the txn pool: its `MULTI { SET data; HINCRBY meta ref_count 1 } EXEC` became
/// one `EVALSHA` of `scripts/ext_store_value.lua` in
/// `storage::external_values`. The new site passes both `{hash_hex}`-tagged keys
/// (meta then data), which is the property the fence below actually proves —
/// a blob store that routed by `ClusterHash::Random` would write the data key
/// on whichever primary fred happened to pick.
///
/// Raised 19 → 23 by b.v9c, which added `storage::presence`: the register,
/// refresh and unregister scripts (four `{sid}`-tagged keys each) and the
/// test-only eviction script (two), the last compiled only under
/// `cfg(any(test, feature = "test-support"))` but still walked here.
///
/// Raised 23 → 30 on a recount of the tree that merged b.v9c with b.y3m,
/// b.v6d and b.nvf, each of which had raised or moved the count on its own
/// branch (main alone had counted 23 → 25 → 26 for the b.xtb, snapshot-read
/// and b.nvf sites below):
///
/// - b.xtb's `scripts/stream_mux_wake.lua` in `storage::stream_mux::reader`
///   (its reader's wake key), which had landed without raising it;
/// - b.v9c's `scripts/presence_sweep_batch.lua` in `storage::presence` (the
///   member set plus each chunk cid's lease and info hash, all
///   `{sid}`-tagged);
/// - b.y3m's atomic snapshot read (`storage::diff_read`, one `EVALSHA` of
///   `object_snapshot_read.lua` passing the object header), whose one call
///   site serves both the read and the startup self-test;
/// - b.v6d's `scripts/create_layer_commit.lua` in `storage::layer` (two sites,
///   the second compiled only under fault injection but still walked here),
///   `scripts/commit_record_retract.lua` in `storage::layer`, and
///   `stamp_obj_header_cursor_below` in `storage::lua`, less the
///   `update_layer_fast.lua` site in `storage::layer`, whose `EVALSHA` no
///   longer goes through this helper;
/// - b.nvf's per-object reference-index recount (`storage::ext_index_audit`,
///   one `EVALSHA` of `ext_index_recount.lua` passing the object's
///   `{object_id}`-tagged keys, header first — four when written, five since
///   b.cyu added the object's migration sentinel), which serves both the
///   reconciler's audit and the txn path's torn-commit arm.
///
/// Raised 30 → 36 by `storage::snapshot_fence`'s six fence and intent
/// scripts (acquire, renew, release, probe, intent register, intent renew; one
/// call site each, every one passing its `{oid}`-tagged keys).
///
/// Raised 36 → 38 by b.ua9: the orphan sweep's veto (`create_layer_veto.lua`
/// in `storage::orphan_sweep`, passing the `{session_id}`-tagged mapping,
/// deletion marker and commit claim) and the one retirement wake-up every
/// route shares (`retired_object_wake.lua` in `layer::wake_retired_object_stream`,
/// passing the object's `{oid}`-tagged stream).
const EVALSHA_STORAGE_SITES_WHEN_WRITTEN: usize = 38;

/// The same, under `src/checkpoint/`: `session_writer.rs`, `recovery.rs`, and
/// `session.rs` ×2 (one production, one in its own `#[cfg(test)]` module).
const EVALSHA_CHECKPOINT_SITES_WHEN_WRITTEN: usize = 4;

/// Both lanes together.
const EVALSHA_SITES_WHEN_WRITTEN: usize =
    EVALSHA_STORAGE_SITES_WHEN_WRITTEN + EVALSHA_CHECKPOINT_SITES_WHEN_WRITTEN;

/// Non-vacuity floor for the site count.
///
/// Deliberately below [`EVALSHA_SITES_WHEN_WRITTEN`] rather than equal to it:
/// the claim this guards is "the needle still selects real call sites", not "the
/// number is exactly N", and an equality would redden on every ordinary
/// refactor that consolidates two scripts. Set at the count that would remain if
/// an entire lane's worth of the checkpoint sites plus a couple of storage ones
/// were consolidated away — anything lower means a lane fell out of the walk or
/// the helper was renamed, which is the regression worth a red test.
///
/// Raised with the count when b.yda added two storage-lane sites (a deferral
/// script in `storage::layer` and `ext_retire_retry_claim.lua` in
/// `storage::orphan_sweep`), keeping the same six-site headroom below the
/// total. Both scripts passed keys, all `{object_id}`-tagged — which is what
/// the fence above proves. b.yuo Part R replaced the deferral script with the
/// retirement intent's scripts (`ext_retire_begin.lua`,
/// `ext_retire_settle.lua`, `ext_retire_clear.lua`, the re-token, the `{gc}`
/// resolve and the GC loop's attempt audit), all called through
/// `storage::external_values` with their keys; the floor was not moved.
///
/// Raised 16 → 21 when b.v9c's four `storage::presence` sites brought the total
/// to 27, restoring the six-site headroom (b.rej's site had raised the total
/// without moving the floor).
///
/// Raised 21 → 28 when the recount above brought the total to 34, restoring
/// the six-site headroom; raised 28 → 34 with `storage::snapshot_fence`'s six
/// sites, restoring it below the total of 40; raised 34 → 36 with b.ua9's two
/// sites, restoring it below the total of 42.
const EVALSHA_SITE_FLOOR: usize = 36;

/// Argument expressions that are an empty key list on their face.
///
/// Only *syntactically* empty forms — see the fence's doc comment for what that
/// deliberately does and does not reach.
fn is_empty_keys_expression(expr: &str) -> bool {
    let normalised = expr.trim_start_matches('&').trim();
    let stripped = normalised
        .strip_suffix(".to_vec()")
        .unwrap_or(normalised)
        .trim();
    matches!(stripped, "vec![]" | "[]" | "Vec::new()")
        || (stripped.starts_with("Vec::<") && stripped.ends_with(">::new()"))
        || (stripped.starts_with("vec![") && stripped[5..].trim_end_matches(']').trim().is_empty())
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// **No** `EVALSHA` issued under `src/storage/` **or** `src/checkpoint/` may be
/// sent with an empty key list.
///
/// # Why both lanes
///
/// The routing defect is fred's, not any one module's: it applies to every
/// `EVALSHA` this process sends, and `src/checkpoint/` sends four of them
/// (`session_writer.rs`, `recovery.rs`, `session.rs` ×2) through the same
/// `evalsha_with_reload` helper. A fence that stopped at `src/storage/` would
/// have policed the lane that happened to be under audit rather than the lane
/// the hazard lives in, and a new zero-key script added on the checkpoint side
/// would have been exactly as invisible as one added here. `src/checkpoint/` is
/// read-only to this test — it is walked, never edited.
///
/// # Why this is a source fence and not a behavioural test
///
/// fred derives an `EVALSHA`'s route from its *keys*: `ClusterHash::Custom` over
/// the first key when there is one, and `ClusterHash::Random` when there is not
/// — and `Random`'s `hash()` returns `None`, so `Router::route` falls through to
/// `writers.values_mut().next()` and the script runs on an arbitrary primary.
/// Every script in this repo reads or writes the keys it was handed, so a
/// zero-key invocation is either a no-op on the wrong node or a mutation of a
/// key range that node does not own. On the single-node fixture the whole suite
/// runs against there is exactly one node, so such a call passes every
/// behavioural test in the tree; it fails only in a cluster deployment, later,
/// as data that quietly did not change. This is also what backs the keyless
/// cluster-mode promise `docs/service/docs/backing-store.md` § Requirements
/// makes to operators.
///
/// # What it inspects, and what it deliberately cannot
///
/// The `keys` **argument expression** at each call site, not merely the number
/// of call sites: the arguments are parsed out of the call (turbofish and
/// multi-line forms included) and the second one is checked against the empty
/// literal forms. What it cannot see is a *variable* that happens to be empty at
/// run time — `script_keys` built by a chunker, say. That case is real (the
/// chunk loop in `src/storage/layer.rs` around the `UNLINK_CHUNK_SIZE` batching)
/// and is handled at the source by an explicit `chunk_len == 0 → break`, which
/// a shape fence cannot verify. What this stops is the far likelier regression:
/// a new script that genuinely takes no keys, written as `vec![]` at the call
/// site because on the developer's single-node Valkey it works.
///
/// # The helper it does not scan
///
/// `storage::lua::evalsha_reload_rebuilding` (the structural WATCH read's
/// window scripts) also sends an `EVALSHA`, but it is outside this scan: its
/// signature takes exactly one key (`key: &str`, sent with `numkeys` 1), so a
/// zero-key call through it does not compile. The type is its fence.
#[test]
fn no_zero_key_evalsha_call_site_survives_under_src_storage_or_checkpoint() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources_under(&root, &root.join("src").join("storage"), &mut files);
    rust_sources_under(&root, &root.join("src").join("checkpoint"), &mut files);
    files.sort();

    // Non-vacuity, part 1: the walk must reach the files that matter — a
    // landmark in each lane plus the two heaviest call-site modules.
    for expected in [
        EVALSHA_WALK_ANCHOR_FILE,
        EVALSHA_CHECKPOINT_ANCHOR_FILE,
        "src/storage/external_values.rs",
        "src/storage/layer.rs",
    ] {
        assert!(
            files.iter().any(|f| f == Path::new(expected)),
            "the source scan did not visit `{expected}`; it walked {} files. Either the file \
             moved (update this list) or the walk is not reaching where it should",
            files.len()
        );
    }

    let mut sites = 0usize;
    let mut violations: Vec<String> = Vec::new();
    let mut unparsed: Vec<String> = Vec::new();

    for file in &files {
        let source = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
        // Joined rather than scanned line by line: a call's argument list
        // routinely spans several lines, and the keys expression is often not
        // on the line the needle is.
        let code = code_lines(&source).join("\n");
        let mut from = 0usize;
        while let Some(offset) = code[from..].find(EVALSHA_NEEDLE) {
            let at = from + offset;
            let after = at + EVALSHA_NEEDLE.len();
            from = after;

            // The `fn evalsha_with_reload(` definition itself is not a call
            // site. It lives in the helper's own module and its first parameter
            // is a client, not a key list.
            let line_start = code[..at].rfind('\n').map(|n| n + 1).unwrap_or(0);
            if code[line_start..at].contains("fn ") {
                continue;
            }

            sites += 1;
            let line_no = code[..at].matches('\n').count() + 1;
            match call_arguments(&code, after) {
                Some(args) if args.len() >= 2 => {
                    if is_empty_keys_expression(&args[1]) {
                        violations.push(format!(
                            "  {}:{line_no}\n    keys argument: {}",
                            file.display(),
                            args[1]
                        ));
                    }
                }
                other => unparsed.push(format!(
                    "  {}:{line_no} (parsed as {other:?})",
                    file.display()
                )),
            }
        }
    }

    // Non-vacuity, part 2: the needle must still select real call sites AND the
    // parser must have understood every one of them. An `evalsha_with_reload`
    // whose arguments could not be parsed is invisible to the check above, which
    // would be a silently narrowed fence rather than a passing one.
    assert!(
        unparsed.is_empty(),
        "the argument parser could not read the call at:\n\n{}\n\nThese call sites are NOT \
         covered by the check below, so the fence is silently narrower than it claims. Either \
         the call was written in a shape the parser does not model (see `call_arguments`) or the \
         helper's signature changed — fix the parser, do not let the site go unchecked.",
        unparsed.join("\n")
    );
    assert!(
        sites >= EVALSHA_SITE_FLOOR,
        "found only {sites} `{EVALSHA_NEEDLE}` call sites under src/storage/ + src/checkpoint/. \
         There were {EVALSHA_SITES_WHEN_WRITTEN} when this fence was written ({EVALSHA_STORAGE_SITES_WHEN_WRITTEN} \
         in the storage lane, {EVALSHA_CHECKPOINT_SITES_WHEN_WRITTEN} in the checkpoint lane), so a number below \
         {EVALSHA_SITE_FLOOR} means the needle has stopped matching (a rename?) or one lane \
         dropped out of the walk, and the scan proves nothing. Update EVALSHA_NEEDLE / the walk \
         rather than deleting this guard — or, if the helper is genuinely gone, retire the fence \
         deliberately in the same commit."
    );

    assert!(
        violations.is_empty(),
        "zero-key `{EVALSHA_NEEDLE}` call site(s) under src/storage/ or src/checkpoint/:\n\n{}\n\n\
         Pass at least one key the script touches. fred routes an EVALSHA by its FIRST key \
         (`ClusterHash::Custom(slot)`); with no keys the policy is `ClusterHash::Random`, whose \
         `hash()` is `None`, so `Router::route` falls through to `writers.values_mut().next()` \
         and the script runs on an ARBITRARY primary — reading or writing a slot range that node \
         does not own. On a single-node fixture that is invisible, which is why this is policed \
         at the source (b.289 sub-task 1; same routing defect class as b.ji8's UNWATCH). This \
         fence lives in `tests/integration/storage/b_289_tests.rs` and the prose it backs is in \
         `src/storage/watch_guard.rs`'s header + SDD §15.3 — a CHECKPOINT-lane call site that \
         trips it is not misfiled: the inventory is one sentence covering both lanes.",
        violations.join("\n")
    );
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// The argument parser, on a fixture carrying every shape the tree actually
/// contains plus the ones the fence must catch.
///
/// The fence above is a *parser* over the tree, and a parser that quietly
/// stopped parsing changes what the fence means without changing whether it is
/// green — the `unparsed` assertion catches an outright failure, but not a
/// mis-split that yields the wrong argument. Both directions are covered here on
/// a fixture rather than on the tree, because the tree contains only some of the
/// shapes at any one time.
#[test]
fn the_evalsha_argument_parser_reads_the_keys_expression() {
    let fixture = concat!(
        "let a = client.evalsha_with_reload(pool.next(), script_keys, argv).await;\n",
        "let b = script\n",
        "    .evalsha_with_reload::<(String, i64), _, _>(\n",
        "        lua_client,\n",
        "        vec![header_key.to_owned(), stream_key],\n",
        "        lua_argv,\n",
        "    )\n",
        "    .await;\n",
        "let c = script.evalsha_with_reload::<i64, _, _>(client, vec![], argv).await;\n",
        "let d = script.evalsha_with_reload(client, Vec::new(), argv).await;\n",
        // Nested generics inside the turbofish. Both shapes are in the tree
        // today — `counters.rs`'s `::<Option<i64>, _, _>` and
        // `external_values.rs`'s `::<Vec<String>, _, _>` — and both are the case
        // the angle-bracket balancer exists for: a depth-counting skip that
        // stopped at the FIRST `>` would resume mid-turbofish and read
        // `, _, _>(client, …)` as the argument list, putting `_` where the keys
        // expression should be. `_` is not an empty-key literal, so the fence
        // would go quiet on those sites rather than red — a silent narrowing.
        "let e = script.evalsha_with_reload::<Option<i64>, _, _>(client, vec![counter_key], argv).await;\n",
        "let f = script.evalsha_with_reload::<Vec<String>, _, _>(client, vec![], argv).await;\n",
    );

    let mut found: Vec<String> = Vec::new();
    let mut from = 0usize;
    while let Some(offset) = fixture[from..].find(EVALSHA_NEEDLE) {
        let at = from + offset;
        let after = at + EVALSHA_NEEDLE.len();
        from = after;
        let args = call_arguments(fixture, after).expect("every fixture call must parse");
        assert_eq!(
            args.len(),
            3,
            "every fixture call takes exactly (client, keys, argv); got {args:?}"
        );
        found.push(args[1].clone());
    }

    assert_eq!(
        found,
        vec![
            "script_keys".to_string(),
            // The multi-line call's keys argument must come back as ONE
            // argument: the comma inside `vec![…]` is not a top-level separator,
            // and a parser that split on it would read `stream_key]` as the argv
            // and miss the keys entirely.
            "vec![header_key.to_owned(), stream_key]".to_string(),
            // The turbofish must be skipped, not parsed as the argument list —
            // note `::<(String, i64), _, _>` above contains both a paren and a
            // comma.
            "vec![]".to_string(),
            "Vec::new()".to_string(),
            // Nested generic in the turbofish: the skip must balance angle
            // brackets rather than stop at the first `>`.
            "vec![counter_key]".to_string(),
            "vec![]".to_string(),
        ],
        "the parser must extract the SECOND top-level argument of each call, skipping any \
         turbofish and treating a bracketed list as one argument"
    );

    assert!(
        !is_empty_keys_expression("script_keys"),
        "a variable is not a syntactically empty key list — see the fence's doc comment for why \
         that limit is deliberate"
    );
    assert!(!is_empty_keys_expression(
        "vec![header_key.to_owned(), stream_key]"
    ));
    for empty in ["vec![]", "vec![ ]", "Vec::new()", "&[]", "[]"] {
        assert!(
            is_empty_keys_expression(empty),
            "{empty:?} must be recognised as an empty key list, or a zero-key EVALSHA written \
             that way sails past the fence"
        );
    }
}

/// The `SCAN`-family needles this fence polices — the **keyless** cursors, the
/// only ones whose route is not derived from a key.
///
/// `.scan(` is fred's *unbuffered* `Client::scan` — the same one-node walk as
/// `.scan_buffered(`, differing only in back-pressure, and just as wrong on a
/// cluster. Its one hit in either walked tree today is the non-cluster arm of
/// `orphan_sweep::scan_keyspace`'s dispatch, which uses the page form so it can
/// tell a complete walk from one cut short (and there is no `Iterator::scan` to
/// false-positive on), so it costs nothing to carry and catches the one
/// spelling the audit's needle list missed.
///
/// # The keyed-cursor exemption (b.yda), and why the needles are dotted
///
/// `SSCAN` / `HSCAN` / `ZSCAN` were on this list and are deliberately NOT any
/// more. They are **keyed** commands: each cursors a single named key, fred
/// routes them by that key exactly as it routes an `HGETALL`, and there is no
/// fan-out variant because there is nothing to fan out to. The rationale this
/// fence rests on — "a bare `SCAN` walks ONE node's slot range and reports a
/// fraction of the keyspace as the whole of it" — does not apply to them at all:
/// a keyed cursor walks one key's members, on the node that owns that key, which
/// is the complete answer by construction. Demanding an `is_clustered` dispatch
/// around one would be demanding a fan-out over shards that hold none of the
/// data. `storage::ext_reconcile`'s `SSCAN le:ext:{gc}:pending` and
/// `ZSCAN le:ext:{gc}:inflight` are the call sites that made the distinction
/// concrete — both single `{gc}`-slot keys.
///
/// Every needle is therefore written with a leading **separator**, and that
/// separator is what implements the exemption rather than merely decorating it:
/// `.sscan(`, `.hscan(` and `.zscan(` all contain `scan(` as a substring, so a
/// bare `scan(` general form would re-catch every keyed cursor the moment the
/// three specific needles were removed — the exemption would be exactly as absent
/// as before, with a misleading needle name in the report. `.sscan(` does not
/// contain `.scan(`; `.sscan_buffered(` does not contain `.scan_buffered(`; and
/// neither `::sscan(` nor `::sscan_buffered(` contains its `::` counterpart. The
/// fan-out spellings still cannot self-match either (`scan_cluster_buffered(`
/// and `scan_cluster(` both put `_cluster` where a needle wants `_buffered(` or
/// `(`).
///
/// # Why both a dotted and a `::` form (b.yda)
///
/// Every keyless call site in both walked trees is a method call today, so the
/// dotted pair alone would police all of them — but that is a property of the
/// current source, not of the hazard, and a fence that rests on it is a fence
/// that a single UFCS rewrite (`Scanner::scan(client, …)`, which is what a
/// rustfmt-driven disambiguation of a trait-method collision produces) retires
/// silently while reading green. The `::` forms cost nothing to carry: they have
/// zero hits in either tree today, they cannot match a module path (`::scan(`
/// demands the `(` immediately after `scan`, so `orphan_sweep::scan_keyspace(`
/// is not a hit), and they preserve the keyed-cursor exemption for exactly the
/// same reason the dotted forms do.
/// [`the_bare_scan_needles_exempt_keyed_cursors`] pins every direction.
const BARE_SCAN_NEEDLES: [&str; 4] = [".scan_buffered(", ".scan(", "::scan_buffered(", "::scan("];

/// The keyed-cursor spellings [`BARE_SCAN_NEEDLES`] deliberately does NOT
/// police, kept as a named list so the exemption is inspectable rather than an
/// absence.
///
/// Used only by [`the_bare_scan_needles_exempt_keyed_cursors`], which asserts
/// that none of them matches a needle — the property the dotted general form
/// exists to preserve.
/// The `::` spellings are listed for the same reason [`BARE_SCAN_NEEDLES`]
/// carries `::` needles: the exemption has to survive the UFCS form of a keyed
/// cursor as well as the method form, and "no needle matches it" is a different
/// claim for each.
const KEYED_CURSOR_FORMS: [&str; 12] = [
    ".sscan(",
    ".hscan(",
    ".zscan(",
    ".sscan_buffered(",
    ".hscan_buffered(",
    ".zscan_buffered(",
    "::sscan(",
    "::hscan(",
    "::zscan(",
    "::sscan_buffered(",
    "::hscan_buffered(",
    "::zscan_buffered(",
];

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// The needle set must match the keyless cursors and miss the keyed ones.
///
/// A fixture rather than a walk of the tree, for the reason the argument-parser
/// test above gives: the tree contains only some of these shapes at any one
/// time, and the property being pinned is about the needles themselves.
///
/// Both halves are load-bearing and neither implies the other. If the general
/// needle lost its dot, every keyed cursor would silently become a violation
/// again and the exemption would be undone by a whitespace-level edit. If a
/// needle stopped matching its own spelling, the fence would go green having
/// policed nothing — which the dispatch-site count guards for the two known
/// sites but not for a form neither of them uses.
#[test]
fn the_bare_scan_needles_exempt_keyed_cursors() {
    for keyed in KEYED_CURSOR_FORMS {
        // A `::` form is a UFCS call, so it hangs off a trait name rather than a
        // receiver binding; spelling it that way keeps the fixture a line that
        // could really be written.
        let caller = if keyed.starts_with("::") {
            "SetsInterface"
        } else {
            "conn"
        };
        let line = format!("let s = Box::pin({caller}{keyed}key, \"*\", Some(100)));");
        assert!(
            BARE_SCAN_NEEDLES.iter().all(|n| !line.contains(n)),
            "{keyed} is a KEYED cursor — fred routes it by the key it names, so it cannot walk a \
             partial keyspace and needs no `is_clustered` fan-out. It must not match any needle; \
             it does in {line:?}. The usual cause is a general needle losing its leading separator, \
             which makes `scan(` a substring of every one of these."
        );
    }

    for keyless in [
        "let s = Box::pin(client.scan_buffered(pattern, Some(100), None::<ScanType>));",
        "let s = client.scan(pattern, Some(100), None::<ScanType>);",
        // The UFCS spellings (b.yda). Zero hits in either walked tree today,
        // which is precisely why they are pinned here: "no call site is written
        // this way" is a fact about today's source, and a fence that silently
        // depends on it stops policing the moment one is.
        "let s = Box::pin(Scanner::scan_buffered(client, pattern, Some(100), None::<ScanType>));",
        "let s = Scanner::scan(client, pattern, Some(100), None::<ScanType>);",
    ] {
        assert!(
            BARE_SCAN_NEEDLES.iter().any(|n| keyless.contains(n)),
            "the keyless cursor in {keyless:?} must still be caught — it is the one-node walk \
             this whole fence exists for"
        );
    }

    for fanout in [
        "let s = Box::pin(client.scan_cluster_buffered(pattern, Some(100), None::<ScanType>));",
        "let s = client.scan_cluster(pattern, Some(100), None::<ScanType>);",
        "let s = Box::pin(Scanner::scan_cluster_buffered(client, pattern, Some(100)));",
        "let s = Scanner::scan_cluster(client, pattern, Some(100));",
    ] {
        assert!(
            BARE_SCAN_NEEDLES.iter().all(|n| !fanout.contains(n)),
            "the CLUSTERED arm of a dispatch must never be its own hit, or every correctly \
             dispatched site would report a violation against itself: {fanout:?}"
        );
    }

    // The false-positive direction the `::` needles opened, and the reason they
    // demand the `(` immediately after `scan`: a module path whose next segment
    // merely STARTS with `scan` is not a cursor call, and both walked trees are
    // full of these.
    for path_call in [
        "use crate::storage::orphan_sweep::scan_keyspace;",
        "let stream = orphan_sweep::scan_keyspace(client, &pattern);",
        "crate::storage::orphan_sweep::scan_cluster_pages(client).await?;",
    ] {
        assert!(
            BARE_SCAN_NEEDLES.iter().all(|n| !path_call.contains(n)),
            "a path segment that only begins with `scan` is not a keyless cursor and must not be \
             reported as one — every needle demands the call's own opening paren right after the \
             method name: {path_call:?}"
        );
    }
}

/// The cluster-wide fan-out forms. A dispatch is only credited when one of these
/// appears in the lookback window alongside `is_clustered`.
///
/// Two spellings because fred offers both a buffered stream (`scan_cluster_buffered`)
/// and an unbuffered one (`scan_cluster`), and which one a call site wants is a
/// back-pressure decision, not a routing one. Accepting only the buffered
/// spelling would redden a correctly-dispatched site for picking the other arm's
/// twin — a false positive on the exact code the fence is asking for.
const CLUSTER_SCAN_FANOUT_FORMS: [&str; 2] = ["scan_cluster_buffered(", "scan_cluster("];

/// How many legitimate `is_clustered` SCAN dispatches the two walked lanes hold:
/// `storage::orphan_sweep::scan_keyspace` (the service's one keyspace walk,
/// shared by the orphan sweep, the reference reconciler and the checkpoint
/// tick's pass 1) and `storage::test_support::scan_matching` (shared by the
/// fixture's prefix cleanup and the object-keyspace dump). The checkpoint lane
/// holds none of its own: it walks through `scan_keyspace`.
const SCAN_DISPATCH_SITES: usize = 2;

/// How many lines of *code* above a bare-scan call site the `is_clustered`
/// dispatch may sit.
///
/// Ten. The two legitimate sites today put the guard three lines above the
/// call (`orphan_sweep::scan_keyspace`'s and `test_support::scan_matching`'s
/// `if client.is_clustered()`); ten leaves headroom for rustfmt without letting
/// an unrelated earlier statement satisfy the match. Measured over comment-stripped lines, so a
/// paragraph of prose between the guard and the call cannot push the guard out
/// of the window.
const SCAN_DISPATCH_LOOKBACK_LINES: usize = 10;

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// A bare **keyless** `SCAN` under `src/storage/` **or** `src/checkpoint/` must
/// sit inside an `is_clustered()` dispatch that also offers the cluster-wide
/// variant.
///
/// The keyed cursors (`SSCAN` / `HSCAN` / `ZSCAN`) are deliberately exempt —
/// fred routes them by the key they name, so they cannot return a partial
/// keyspace and there is no fan-out to dispatch to. See
/// [`BARE_SCAN_NEEDLES`] for the full argument and
/// [`the_bare_scan_needles_exempt_keyed_cursors`] for the pin.
///
/// # The failure this bounds
///
/// `scan_buffered` issues its `SCAN` on **one** connection, so on a cluster it
/// walks one primary's slot range and silently reports the keyspace as
/// containing only what that shard holds. Every consumer in either walked lane
/// is an enumeration — the orphan sweeper reaping dead objects, the test
/// fixture's prefix cleanup, the checkpoint tick enumerating session state — so
/// a missed shard is not an error, it is a *quietly incomplete answer*:
/// orphaned objects that are never reaped, test keys that leak into the next
/// run, sessions that are never checkpointed. There is no exception to report
/// and nothing to classify; the sweep returns `Ok` having done a fraction of its
/// job. fred's `scan_cluster` and `scan_cluster_buffered` are the fan-out (each
/// pins `hash_slot: Some(slot)` per shard and re-resolves the slot on every
/// continuation), and picking the right one is a per-call-site decision that
/// only a cluster can falsify — which no test in the default suite runs against.
///
/// # Why both lanes
///
/// Same reason as the EVALSHA fence above: the hazard is fred's routing, not one
/// module's. `src/checkpoint/checkpointer.rs` walks the keyspace too — today
/// through `orphan_sweep::scan_keyspace`, so it holds no dispatch site of its
/// own — and policing only the audited lane would leave a new bare `SCAN` on the
/// checkpoint side as invisible as one added here.
/// `src/checkpoint/` is read-only to this test — it is walked, never edited.
///
/// # What "outside a dispatch" means here, textually
///
/// A hit is a violation unless the [`SCAN_DISPATCH_LOOKBACK_LINES`] lines of
/// code above it (comments stripped) contain **both** `is_clustered` and one of
/// [`CLUSTER_SCAN_FANOUT_FORMS`]. Requiring both is what keeps the guard from
/// being satisfied by an `is_clustered()` consulted for some unrelated purpose:
/// the claim is not "this code knows whether it is clustered" but "this call is the
/// non-cluster arm of a two-arm dispatch, and the other arm exists".
#[test]
fn no_bare_scan_call_site_survives_outside_an_is_clustered_dispatch() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources_under(&root, &root.join("src").join("storage"), &mut files);
    rust_sources_under(&root, &root.join("src").join("checkpoint"), &mut files);
    files.sort();

    // Non-vacuity, part 1: the walk must reach the two known dispatch sites and
    // the checkpoint lane's keyspace walk.
    for expected in [
        "src/storage/orphan_sweep.rs",
        "src/storage/test_support.rs",
        "src/checkpoint/checkpointer.rs",
    ] {
        assert!(
            files.iter().any(|f| f == Path::new(expected)),
            "the source scan did not visit `{expected}`; it walked {} files. Either the file \
             moved (update this list) or the walk is not reaching where it should",
            files.len()
        );
    }

    let mut dispatched: Vec<String> = Vec::new();
    let mut violations: Vec<String> = Vec::new();

    for file in &files {
        let source = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
        let lines = code_lines(&source);
        for (idx, line) in lines.iter().enumerate() {
            // Neither fan-out spelling contains a needle: `scan_cluster_buffered(`
            // and `scan_cluster(` both put `_cluster` immediately after `scan`,
            // where every needle wants either `_buffered(` or `(`. So the
            // clustered arm of a dispatch is never its own hit.
            let Some(needle) = BARE_SCAN_NEEDLES
                .iter()
                .find(|needle| line.contains(**needle))
            else {
                continue;
            };
            let start = idx.saturating_sub(SCAN_DISPATCH_LOOKBACK_LINES);
            let window = lines[start..=idx].join("\n");
            let site = format!("  {}:{}\n    {}", file.display(), idx + 1, line.trim());
            if window.contains("is_clustered")
                && CLUSTER_SCAN_FANOUT_FORMS
                    .iter()
                    .any(|form| window.contains(form))
            {
                dispatched.push(site);
            } else {
                violations.push(format!("{site}\n    (needle: {needle})"));
            }
        }
    }

    // The violations assertion runs FIRST, deliberately. Deleting a dispatch
    // arm both creates a violation and drops the dispatch count, and of the two
    // messages only this one names the offending line — checking non-vacuity
    // first would answer a real regression with "a new dispatch site is fine,
    // update this count", which is advice for a different situation entirely.
    assert!(
        violations.is_empty(),
        "bare SCAN-family call site(s) under src/storage/ or src/checkpoint/ with no \
         `is_clustered` dispatch above them:\n\n{}\n\nDispatch on `client.is_clustered()` and use \
         `scan_cluster(...)` (or `scan_cluster_buffered(...)`) on the clustered arm — or, for a \
         keyspace walk, call `orphan_sweep::scan_keyspace`, which also ends a walk cut short \
         with an error. A bare `scan` / `scan_buffered` walks ONE node's slot range, so on a cluster it reports a keyspace that \
         silently omits every sibling shard — and every consumer in these two lanes is an \
         enumeration (orphan reaping, prefix cleanup, the checkpoint tick's session sweep), so \
         the result is not an error but a quietly incomplete answer that returns `Ok`. Invisible \
         on the single-node fixture this suite runs against (b.289 sub-task 1). This fence lives \
         in `tests/integration/storage/b_289_tests.rs` and the keyless/one-node routing inventory \
         it belongs to is in `src/storage/watch_guard.rs`'s header + SDD §15.3 — a CHECKPOINT-lane \
         call site tripping a fence filed under `storage` is expected: the inventory covers both \
         lanes as one sentence.",
        violations.join("\n")
    );

    // Non-vacuity: the needles must still match the two dispatch sites. With
    // no allow-list, "found nothing" is the passing state, so a rename of fred's
    // method would retire this guard silently.
    assert_eq!(
        dispatched.len(),
        SCAN_DISPATCH_SITES,
        "expected exactly the {SCAN_DISPATCH_SITES} legitimate dispatch sites \
         (`orphan_sweep::scan_keyspace` and `test_support::scan_matching`) but found {}:\n\n{}\n\nA new dispatch site is \
         fine — update SCAN_DISPATCH_SITES. A count of 0 means the needles have stopped matching \
         and this fence is asserting nothing; update BARE_SCAN_NEEDLES / CLUSTER_SCAN_FANOUT_FORMS \
         rather than deleting it. A new site in `src/checkpoint/` is caught because that lane is \
         walked from here deliberately — see this test's doc comment — but a keyspace walk there \
         should go through `orphan_sweep::scan_keyspace` rather than add a dispatch.",
        dispatched.len(),
        dispatched.join("\n")
    );
}

/// The call-site form of fred's `Transaction` constructor.
const MULTI_NEEDLE: &str = ".multi()";

/// How many `.multi()` call sites `src/storage/` holds, per the inventory in
/// `src/storage/watch_guard.rs`'s header and SDD §15.3.
///
/// Lowered 4 → 3 by b.rej: `external_values` was the fourth, and its per-hash
/// `store_values` transaction is now an `EVALSHA` (Lua is atomic under Valkey's
/// single server thread, so the exclusive session a `MULTI` needed buys nothing
/// there). Lowered 3 → 2 by b.v6d: `create_layer`'s mapping `MULTI` is now the
/// one-`EVALSHA` commit point `scripts/create_layer_commit.lua`, for the same
/// reason. One `layer` site and `session` remain.
const STORAGE_MULTI_SITES: usize = 2;

/// How many `.multi()` call sites `src/checkpoint/` holds, per the same
/// inventory.
const CHECKPOINT_MULTI_SITES: usize = 2;

/// Where the prose inventory this fence backs is written.
const MULTI_INVENTORY_FILE: &str = "src/storage/watch_guard.rs";

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// The `.multi()` inventory in `watch_guard.rs`'s header must stay countable.
///
/// # Why a counting fence is the right shape here
///
/// b.289 sub-task 1 replaced a one-line claim in that header ("`Transaction::exec`
/// already pins correctly via its own hash slot") with a two-shape inventory,
/// because the claim had gone half-stale: since b.bhb the two **WATCHed** commit
/// sites do not use fred's `Transaction` at all — they send `MULTI` / `CLIENT ID`
/// / `EXEC` as slot-pinned `CustomCommand`s inside a `Pipeline`, since a
/// `Transaction` replays the whole `MULTI … EXEC` onto a connection that never
/// held the `WATCH`. The remaining `Transaction` users are correct precisely
/// *because* `Transaction::update_hash_slot` stamps `ClusterHash::Custom(slot)`
/// onto every queued command including the keyless `MULTI`.
///
/// That distinction is now load-bearing prose in two places (the header here and
/// SDD §15.3's keyless-command inventory), and prose enumerating call sites rots
/// the moment somebody adds one. Nothing about a *new* `client.multi()` is wrong
/// — it pins correctly by construction — so no behavioural test can notice it,
/// and the inventory silently becomes a lie that the next reader trusts.
/// Counting is exactly the assertion the prose makes, so the prose fails when it
/// stops being true.
///
/// # Why the checkpoint lane is counted from here
///
/// `src/checkpoint/` is read-only to this lane, and the inventory it appears in
/// is `src/storage/watch_guard.rs`'s — the two counts are one sentence, and
/// splitting the fence across lanes would let half of it drift.
#[test]
fn the_multi_call_site_inventory_matches_the_watch_guard_header() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let count_in = |dir: &str| -> Vec<String> {
        let mut files = Vec::new();
        rust_sources_under(&root, &root.join(dir), &mut files);
        files.sort();
        let mut hits = Vec::new();
        for file in &files {
            let source = std::fs::read_to_string(root.join(file))
                .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
            for (idx, line) in code_lines(&source).iter().enumerate() {
                if line.contains(MULTI_NEEDLE) {
                    hits.push(format!(
                        "  {}:{}\n    {}",
                        file.display(),
                        idx + 1,
                        line.trim()
                    ));
                }
            }
        }
        hits
    };

    let storage = count_in("src/storage");
    let checkpoint = count_in("src/checkpoint");

    // The header this fence backs must still say what it says. A rewrite that
    // dropped the inventory would leave the counts below policing nothing.
    let header = std::fs::read_to_string(root.join(MULTI_INVENTORY_FILE))
        .unwrap_or_else(|e| panic!("reading {MULTI_INVENTORY_FILE} must succeed: {e}"));
    assert!(
        header.contains("two in `src/storage`") && header.contains("two in the checkpoint lane"),
        "`{MULTI_INVENTORY_FILE}`'s header no longer states the two-in-storage / \
         two-in-checkpoint `.multi()` inventory this fence backs, so the counts below are \
         policing a claim nobody makes any more. If the inventory changed, update BOTH the \
         header and the counts here (and SDD §15.3) together — that pairing is the whole point \
         of this test."
    );

    assert_eq!(
        storage.len(),
        STORAGE_MULTI_SITES,
        "expected {STORAGE_MULTI_SITES} `{MULTI_NEEDLE}` call sites under src/storage/ \
         (`layer` and `session`; `external_values` was a fourth until b.rej moved its per-hash \
         blob store to a data-pool EVALSHA, and `create_layer`'s mapping MULTI a third until \
         b.v6d made it the `create_layer_commit.lua` EVALSHA) but found {}:\n\n{}\n\nA new fred \
         `Transaction` site is not a bug — `Transaction::update_hash_slot` stamps \
         `ClusterHash::Custom(slot)` onto every queued command including the keyless MULTI, so it \
         pins correctly by construction. What IS a bug is leaving the inventory in \
         `{MULTI_INVENTORY_FILE}`'s header and SDD §15.3 saying a stale number. Update all three \
         together.",
        storage.len(),
        storage.join("\n")
    );
    assert_eq!(
        checkpoint.len(),
        CHECKPOINT_MULTI_SITES,
        "expected {CHECKPOINT_MULTI_SITES} `{MULTI_NEEDLE}` call sites under src/checkpoint/ \
         (both in `checkpoint::session`) but found {}:\n\n{}\n\nSame remedy as the storage count \
         above: the number lives in three places and they are one sentence.",
        checkpoint.len(),
        checkpoint.join("\n")
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// D — the cluster gate itself: is the only job that runs these tests still
//     wired to run them? (b.289 round-2 test review, items 5a/5b)
// ═════════════════════════════════════════════════════════════════════════════
//
// The two cluster-gated tests above are self-gating: no env var, no cluster, a
// `SKIP:` line and a green result. That shape is the project default (see
// `docs-internal/test-writing-guide.md`) and it is right for a developer laptop,
// where there is genuinely no cluster to run against. It is exactly wrong in
// CI's `test-cluster:` job, which exists *for* these tests: a skip there is a
// silent hole, and it looks identical to a pass in every report a human reads —
// nextest's `profile.ci` prints nothing for a passing test, so even the `SKIP:`
// line is swallowed.
//
// Two guards, because the gate can come unarmed in two unrelated ways:
//
//   * somebody edits `.gitlab-ci.yml` — drops the `LIVE_EDIT_TEST_CLUSTER_URL`
//     variable, or the `--features test-fault-injection` flag (which does not
//     skip the tests, it compiles them out, which is worse). Caught by
//     `the_test_cluster_ci_job_still_arms_both_gates`, which runs everywhere,
//     including on the MR that makes the edit.
//   * the config is right but the gate is shut at run time anyway — the cluster
//     never came up, `before_script` failed soft, the variable was lost between
//     shells. Caught by `the_cluster_gate_is_armed_when_ci_runs_the_test_cluster_job`,
//     which is a no-op everywhere except inside the job.
//
// Neither guard needs a Valkey, a cluster, or a feature flag.

/// The env var that names the multi-shard cluster.
///
/// Deliberately shared with `seam_driven::cluster_fixture` rather than spelled
/// twice: the fixture's gate and the guard that checks the gate is armed have to
/// be looking at the same variable, and two string literals is exactly how that
/// stops being true.
const CLUSTER_URL_VAR: &str = "LIVE_EDIT_TEST_CLUSTER_URL";

/// The URL scheme that makes fred do `CLUSTER SLOTS` discovery and route by
/// slot. A `redis://` URL against a cluster node is not a cluster client — it
/// gets `MOVED` for every key the seed shard does not own — so the scheme is
/// part of the gate, not cosmetic.
const CLUSTER_URL_SCHEME: &str = "redis-cluster://";

/// Prefix of every CI job name that brings up a cluster and expects the
/// cluster-gated tests to run: `test-cluster` and `test-cluster-disruptive`.
///
/// A prefix rather than an exact match so the disruptive job — which sets the
/// same variable and runs its own cluster — is covered too, and so a future
/// `test-cluster-<something>` sibling is covered by construction rather than by
/// somebody remembering this list.
const CLUSTER_JOB_NAME_PREFIX: &str = "test-cluster";

/// GitLab's per-job name variable. Set in every job, absent everywhere else,
/// which is what makes the guard below a no-op on a laptop.
const CI_JOB_NAME_VAR: &str = "CI_JOB_NAME";

/// The CI config the wiring fence reads, relative to the crate manifest dir.
const CI_CONFIG_FILE: &str = ".gitlab-ci.yml";

/// The job whose wiring is fenced. The disruptive sibling runs a filtered
/// selection that deliberately excludes these tests, so it is not the one that
/// must stay wired.
const CI_CLUSTER_JOB: &str = "test-cluster";

/// The feature flag whose absence *compiles the cluster tests out* rather than
/// skipping them.
const FAULT_INJECTION_FEATURE: &str = "test-fault-injection";

/// The lines of `ci` belonging to the top-level job `job`, comments stripped.
///
/// Located textually — from the `job:` key to the next top-level key — rather
/// than by parsing YAML, for the same reason the sibling fences here scan text:
/// this crate has no YAML dependency and adding one to assert two substrings
/// would be a poor trade. A top-level key is a line that starts at column zero,
/// is not a comment, and carries a `:`; every line inside a job block is
/// indented, so the boundary is unambiguous in this file.
///
/// Whole-line comments are dropped because `.gitlab-ci.yml`'s prose *mentions*
/// both needles (there is a comment explaining what `LIVE_EDIT_TEST_CLUSTER_URL`
/// is for), and a fence satisfied by a comment about the wiring rather than the
/// wiring is not a fence.
fn ci_job_block<'a>(ci: &'a str, job: &str) -> Vec<&'a str> {
    let is_top_level_key = |line: &str| {
        !line.is_empty()
            && !line.starts_with(char::is_whitespace)
            && !line.starts_with('#')
            && line.contains(':')
    };

    let mut lines = ci.lines();
    let header = format!("{job}:");
    lines
        .by_ref()
        .find(|line| line.trim_end() == header)
        .unwrap_or_else(|| {
            panic!(
                "`{CI_CONFIG_FILE}` no longer defines a top-level `{job}:` job. That job is the \
                 ONLY place the cluster-gated tests in this file execute — if it was renamed, \
                 update CI_CLUSTER_JOB and CLUSTER_JOB_NAME_PREFIX here; if it was deleted, the \
                 two `cluster_*` tests above now run nowhere at all and should be retired \
                 deliberately in the same commit rather than left green and vacuous."
            )
        });

    lines
        .take_while(|line| !is_top_level_key(line))
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect()
}

/// The value a job block assigns to [`CLUSTER_URL_VAR`], quotes stripped.
///
/// A separate pure function from the fence that uses it so
/// [`the_ci_job_wiring_parser_reads_a_job_block`] can exercise it against
/// fixtures. `.gitlab-ci.yml` is out of this lane — it must not be perturbed to
/// prove the fence discriminates — so the discrimination is proved on fixtures
/// instead, which is the stronger form anyway: it stays proved.
fn cluster_url_in(block: &[&str]) -> Option<String> {
    block
        .iter()
        .map(|line| line.trim())
        .find(|line| line.starts_with(CLUSTER_URL_VAR))
        .and_then(|line| line.split_once(':'))
        .map(|(_, value)| {
            value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string()
        })
}

/// The `cargo nextest run` command line a job block runs, `- ` list marker
/// stripped. Companion of [`cluster_url_in`]; same reason for being pure.
fn nextest_invocation_in(block: &[&str]) -> Option<String> {
    block
        .iter()
        .map(|line| line.trim().trim_start_matches("- ").trim())
        .find(|line| line.starts_with("cargo nextest run"))
        .map(str::to_string)
}

/// Whether a `cargo nextest run` command line arms `feature`.
///
/// Matches the bare feature name anywhere on the line rather than the literal
/// `--features <name>`, because cargo accepts several equally ordinary
/// spellings that all arm it: `--features a,b` (comma-joined), repeated
/// `--features` flags, and `--features=a`. Anchoring on `--features <name>`
/// would redden this fence on the comma-joined form with a message asserting
/// the feature had been *dropped* — a false red carrying a wrong diagnosis,
/// which costs more than the fence is worth.
///
/// A bare substring scan is safe for the one name that matters here: no other
/// feature in this workspace contains `test-fault-injection` as a substring.
/// The near miss is `test-checkpoint-fault-injection`, which does not —
/// [`the_ci_job_wiring_parser_reads_a_job_block`] pins that, since the real
/// `test-cluster:` job passes both and a collision would make this check
/// unfalsifiable.
fn arms_feature(nextest: &str, feature: &str) -> bool {
    nextest.contains(feature)
}

/// Whether a `cargo nextest run` command line narrows its selection with a
/// filterset expression.
///
/// Detects `-E` as a whole token — `-E expr`, `-E'expr'`, `-E"expr"`, `-E=expr`
/// — plus the `--filterset` long form in both its spaced and `=` spellings. A
/// substring scan for `" -E "` (what this used to do) sees only the spaced
/// form and misses every quoted-adjacent one, while a bare `contains("-E")`
/// would fire on any argument that merely contains those two characters.
fn has_filterset(nextest: &str) -> bool {
    nextest.split_whitespace().any(|token| {
        token == "-E"
            || token.starts_with("-E'")
            || token.starts_with("-E\"")
            || token.starts_with("-E=")
            || token.starts_with("--filterset")
    })
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// CI's `test-cluster:` job must still arm **both** activation conditions the
/// cluster-gated tests in this file depend on.
///
/// # Why a config fence and not a comment
///
/// The two conditions fail in opposite, equally quiet ways. Drop
/// `LIVE_EDIT_TEST_CLUSTER_URL` and the tests skip — `cluster_fixture` prints a
/// `SKIP:` line and returns `None`, and under nextest's `profile.ci`, which
/// prints nothing for a passing test, that line is never seen. Drop
/// `--features test-fault-injection` and they are not compiled at all: no test,
/// no skip, no line, and the suite's total count drops by two in a job that runs
/// ~2700. Either way the pipeline is green and the per-node breaker — whose
/// behaviour is *only* observable on a multi-primary cluster — has no
/// behavioural coverage anywhere in the repo.
///
/// So the wiring is load-bearing test infrastructure that lives outside any test
/// file, which is precisely the case `docs-internal/test-writing-guide.md`
/// § "Tests anchored to generated artifacts" answers with a wiring test: assert
/// the CI job still does the thing, from a test that runs everywhere.
///
/// # What it does not claim
///
/// That the cluster came *up*. A `before_script` that fails soft leaves the
/// variable set and the cluster absent, and this fence reads a file, not a
/// cluster. [`the_cluster_gate_is_armed_when_ci_runs_the_test_cluster_job`] is
/// the other half — it runs inside the job and checks the gate at run time.
///
/// # In the published source tree
///
/// `.publishignore` leaves `.gitlab-ci.yml` out of the GitHub source release,
/// so there this test skips with a line on stderr rather than panicking. Only
/// a missing file skips it; in this repository an unreadable one fails.
#[test]
fn the_test_cluster_ci_job_still_arms_both_gates() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !root.join(CI_CONFIG_FILE).exists() {
        eprintln!("skipped: no {CI_CONFIG_FILE} (published source tree), so no CI wiring to fence");
        return;
    }
    let ci = std::fs::read_to_string(root.join(CI_CONFIG_FILE))
        .unwrap_or_else(|e| panic!("reading {CI_CONFIG_FILE} must succeed: {e}"));
    let block = ci_job_block(&ci, CI_CLUSTER_JOB);

    // Non-vacuity: an empty or near-empty block means the boundary detection
    // above mis-parsed and every assertion below is asserting on nothing.
    assert!(
        block.len() > 10,
        "the `{CI_CLUSTER_JOB}:` block in {CI_CONFIG_FILE} parsed as only {} line(s), which cannot \
         be right for a job that declares services, variables, a before_script and a script. The \
         top-level-key boundary detection in `ci_job_block` has gone wrong — fix it rather than \
         letting the assertions below pass on an empty block:\n{:?}",
        block.len(),
        block
    );

    // Gate 1 — the cluster URL, with the scheme that actually makes fred a
    // cluster client.
    let url_value = cluster_url_in(&block).unwrap_or_else(|| {
        panic!(
            "the `{CI_CLUSTER_JOB}:` job in {CI_CONFIG_FILE} no longer sets \
                 `{CLUSTER_URL_VAR}`. Without it `cluster_fixture` in this file returns `None` \
                 and BOTH cluster-gated tests print a `SKIP:` line and pass — and nextest's \
                 `profile.ci` prints nothing for a passing test, so the skip is invisible in the \
                 job log. Those two tests are the only behavioural coverage anywhere in the repo \
                 of the per-node state-proof breaker (b.289 sub-task 4) and of a `Clean` \
                 collecting the shared fallback bucket (the b.289 round-1 defect); both are \
                 inexpressible on one endpoint. Restore the variable, or retire the tests \
                 deliberately."
        )
    });
    assert!(
        url_value.starts_with(CLUSTER_URL_SCHEME),
        "the `{CI_CLUSTER_JOB}:` job sets `{CLUSTER_URL_VAR}` to {url_value:?}, which does not use \
         the `{CLUSTER_URL_SCHEME}` scheme. `cluster_fixture` checks the scheme and SKIPS on any \
         other one — deliberately, because a `redis://` URL pointed at a cluster node is not a \
         cluster client: fred does no `CLUSTER SLOTS` discovery, does not follow `MOVED`, and the \
         per-node bucket map the tests assert on has exactly one bucket. So the wrong scheme here \
         is not a degraded run, it is a silent skip."
    );

    // Gate 2 — the feature, without which the tests do not exist.
    let nextest = nextest_invocation_in(&block).unwrap_or_else(|| {
        panic!(
            "the `{CI_CLUSTER_JOB}:` job in {CI_CONFIG_FILE} no longer invokes \
                 `cargo nextest run`. This fence reads that invocation to confirm the \
                 `{FAULT_INJECTION_FEATURE}` feature is still passed; if the job switched \
                 runners, re-anchor the fence on the new command rather than deleting it."
        )
    });
    assert!(
        arms_feature(&nextest, FAULT_INJECTION_FEATURE),
        "the `{CI_CLUSTER_JOB}:` job's `cargo nextest run` no longer names \
         `{FAULT_INJECTION_FEATURE}` anywhere on its command line, in any `--features` spelling \
         (`--features {FAULT_INJECTION_FEATURE}`, comma-joined `--features a,{FAULT_INJECTION_FEATURE}`, \
         or `--features={FAULT_INJECTION_FEATURE}` all satisfy this). This one does not skip the \
         cluster-gated tests, \
         it COMPILES THEM OUT: `seam_driven` is `#[cfg(feature = \"{FAULT_INJECTION_FEATURE}\")]` \
         because the state-proof verdicts are canned frames pushed through `ProbeFaultInjector`, \
         which does not exist without the feature. The job then reports green having run two \
         fewer tests out of ~2700, with no skip line and nothing to notice. Command read: \
         {nextest:?}"
    );

    // Gate 2b — nothing may narrow the selection past these tests. The job runs
    // the suite unfiltered today; an `-E` filterset added later would deselect
    // them exactly as silently as dropping the feature.
    assert!(
        !has_filterset(&nextest),
        "the `{CI_CLUSTER_JOB}:` job's `cargo nextest run` has grown a filterset — a `-E` token in \
         any spelling (`-E expr`, `-E'expr'`, `-E\"expr\"`, `-E=expr`) or the `--filterset` long \
         form. The job runs the \
         suite unfiltered today, which is how the self-gating cluster tests in this file get \
         picked up with no CI-side list to keep in sync. A filterset is not wrong in itself — but \
         it must be shown to still select `cluster_a_clean_on_one_primary_does_not_rearm_the_breaker_on_another` \
         and `cluster_a_clean_on_a_named_primary_collects_the_shared_fallback_bucket`, and this \
         fence updated to assert whatever the new selection guarantees. Command read: {nextest:?}"
    );
}

/// Inside CI's cluster job, a *skipped* cluster-gated test is a failure.
///
/// # What this converts
///
/// A silent hole into a red test. Everywhere else — a laptop, the single-node
/// `test:` job — `cluster_fixture` skipping is correct and this test is a no-op
/// pass. Inside a job whose entire reason to exist is running those tests
/// against a real 3-shard cluster, a skip means the job did nothing it was
/// created for, and today that outcome is indistinguishable from success: the
/// `SKIP:` line goes to stderr on a passing test, and nextest's `profile.ci`
/// prints nothing for passing tests.
///
/// # Why it is separate from the config fence
///
/// [`the_test_cluster_ci_job_still_arms_both_gates`] reads `.gitlab-ci.yml` and
/// so catches only what the config says. This one reads the *environment the
/// tests actually see*, and catches the cases the config cannot speak to: a
/// `dev-valkey.sh cluster-up` that failed soft in `before_script`, a variable
/// lost between shells (GitLab runs `after_script` in a fresh one), a runner
/// override. Config-says-yes and run-time-says-no is precisely the combination
/// that produces a green job with no coverage.
///
/// # Why it does not itself need a cluster
///
/// It asserts on env vars only — no connection, no fixture. That keeps it
/// ungated: it must run in the shape that has no `test-fault-injection` feature
/// too, because "the feature was dropped" is one of the things that leaves the
/// job covering nothing.
#[test]
fn the_cluster_gate_is_armed_when_ci_runs_the_test_cluster_job() {
    let Ok(job) = std::env::var(CI_JOB_NAME_VAR) else {
        // Not in GitLab CI at all: a laptop, a container, the sandbox. The
        // cluster gate is *supposed* to be shut here.
        return;
    };
    if !job.starts_with(CLUSTER_JOB_NAME_PREFIX) {
        // A CI job, but not one that brings up a cluster — the single-node
        // `test:` job, a lint job, a publish job. Same story.
        return;
    }

    let url = std::env::var(CLUSTER_URL_VAR).unwrap_or_else(|_| {
        panic!(
            "{CI_JOB_NAME_VAR}={job:?} but {CLUSTER_URL_VAR} is unset. This job exists to run the \
             cluster-gated tests against a real multi-shard cluster; with the variable unset every \
             one of them takes the `SKIP:` branch and PASSES, so the job goes green having \
             verified nothing — and under nextest's `profile.ci`, which suppresses output for \
             passing tests, the skip lines never reach the log. Set {CLUSTER_URL_VAR} in the \
             job's `variables:` block (see {CI_CONFIG_FILE}), or, if this job is no longer meant \
             to run them, rename it so it does not start with {CLUSTER_JOB_NAME_PREFIX:?}."
        )
    });
    assert!(
        url.starts_with(CLUSTER_URL_SCHEME),
        "{CI_JOB_NAME_VAR}={job:?} and {CLUSTER_URL_VAR}={url:?}, which is not a \
         `{CLUSTER_URL_SCHEME}` URL. Every cluster gate in the suite checks the scheme and skips \
         on anything else, because a `redis://` URL aimed at a cluster node gives fred a \
         NON-cluster client: no `CLUSTER SLOTS` discovery, no `MOVED` following, and a per-node \
         bucket map with exactly one bucket in it — which is the single-endpoint case these tests \
         exist to escape. The result is a silent full skip, not a degraded run."
    );
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// The CI-config parser, on a fixture carrying the shapes `.gitlab-ci.yml`
/// actually contains plus the ones the fence must catch.
///
/// # Why a fixture and not a perturbation
///
/// `.gitlab-ci.yml` is outside this test lane, so the usual discrimination
/// experiment — delete the variable, watch the fence redden, put it back — is
/// not available. A fixture is the better instrument regardless: the
/// perturbation proves the fence discriminated *once*, on the day somebody ran
/// it, while this proves it on every run, including after a future edit to the
/// parser. [`the_test_cluster_ci_job_still_arms_both_gates`] is a *reader* over
/// a config file, and a reader that quietly stopped reading changes what the
/// fence means without changing whether it is green.
///
/// The failure modes covered are the ways this fence could go vacuous or go
/// wrong: bleeding into the next job (so a *sibling* job's wiring satisfies
/// it), being satisfied by a comment that merely mentions the variable, failing
/// to notice an absent needle, and — the two the literal matchers used to get
/// backwards — reading a comma-joined `--features a,b` as a dropped feature and
/// a space-less `-E'expr'` as no filterset. Those last two are worse than a
/// vacuous fence: they fail, and the message names a cause that is not the
/// cause.
#[test]
fn the_ci_job_wiring_parser_reads_a_job_block() {
    // Shaped exactly like the real file: two adjacent cluster jobs, a comment
    // block between them that names the variable in prose, an indented
    // `variables:` map, and a `- cargo nextest run …` script line.
    let fixture = concat!(
        "test:\n",
        "  script:\n",
        "    - cargo nextest run --profile ci --workspace --features test-support\n",
        "\n",
        "test-cluster:\n",
        "  variables:\n",
        "    LIVE_EDIT_TEST_CLUSTER_URL: \"redis-cluster://127.0.0.1:7000\"\n",
        "    TEST_VALKEY_URL: \"redis://valkey:6379\"\n",
        "  script:\n",
        "    - cargo build -p oodf-ffi --lib --features internal-bridge\n",
        "    - cargo nextest run --profile ci --workspace --features test-fault-injection\n",
        "\n",
        "# Prose about the next job that mentions LIVE_EDIT_TEST_CLUSTER_URL and\n",
        "# --features test-fault-injection, neither of which is wiring.\n",
        "test-cluster-disruptive:\n",
        "  variables:\n",
        "    LIVE_EDIT_TEST_CLUSTER_URL: \"redis-cluster://127.0.0.1:7000\"\n",
        "  script:\n",
        "    - cargo nextest run --profile ci -E 'test(/disruptive/)' --features test-fault-injection\n",
        "\n",
        // A job spelling the same wiring the other ordinary way: features
        // comma-joined onto one flag, and a filterset written adjacent to its
        // `-E` with no space. Both are what the real file could become on any
        // routine edit, and both used to be read wrong — the first as "feature
        // dropped", the second as "no filterset".
        "test-cluster-tight:\n",
        "  script:\n",
        "    - cargo nextest run --profile ci -E'test(/tight/)' --features test-support,test-fault-injection\n",
        "\n",
        // And a job that arms only the *other* fault-injection feature. Its
        // name contains `fault-injection` but not `test-fault-injection`, which
        // is what makes bare-substring matching safe here.
        "test-cluster-checkpoint-only:\n",
        "  script:\n",
        "    - cargo nextest run --profile ci --filterset 'test(/ckpt/)' --features test-checkpoint-fault-injection\n",
    );

    let cluster = ci_job_block(fixture, "test-cluster");
    assert!(
        cluster
            .iter()
            .any(|l| l.contains("LIVE_EDIT_TEST_CLUSTER_URL")),
        "the `test-cluster` block must contain its own variables: {cluster:?}"
    );
    assert!(
        !cluster.iter().any(|l| l.contains("disruptive")),
        "the block must STOP at the next top-level key. Bleeding into \
         `test-cluster-disruptive:` would let that job's variables satisfy a fence about \
         `test-cluster:` — the two are wired separately and either can be edited alone. Read: \
         {cluster:?}"
    );
    assert!(
        !cluster.iter().any(|l| l.trim_start().starts_with('#')),
        "comment lines must be dropped. `.gitlab-ci.yml` explains in prose what \
         `LIVE_EDIT_TEST_CLUSTER_URL` is for, and a fence satisfied by the explanation rather than \
         the assignment is not a fence. Read: {cluster:?}"
    );

    assert_eq!(
        cluster_url_in(&cluster).as_deref(),
        Some("redis-cluster://127.0.0.1:7000"),
        "the URL value must come back unquoted, so the scheme check is comparing against the URL \
         and not against a leading double-quote"
    );
    let nextest = nextest_invocation_in(&cluster).expect("the fixture job runs cargo nextest");
    assert!(
        nextest.starts_with("cargo nextest run"),
        "the `- ` YAML list marker must be stripped: {nextest:?}"
    );
    assert!(
        arms_feature(&nextest, FAULT_INJECTION_FEATURE),
        "the nextest line selected must be the one that carries the features, not the \
         `cargo build` line above it: {nextest:?}"
    );
    assert!(
        !has_filterset(&nextest),
        "the unfiltered invocation must be the one read for `test-cluster:` — picking up a \
         sibling job's filtered command would make the filterset assertion in the fence \
         meaningless: {nextest:?}"
    );

    // The sibling job, read on its own terms, DOES carry a filterset. This is
    // what proves the filterset assertion in the fence can actually fire — it is
    // not an assertion nothing in this repo could ever violate.
    let disruptive = ci_job_block(fixture, "test-cluster-disruptive");
    let disruptive_nextest =
        nextest_invocation_in(&disruptive).expect("the sibling job runs cargo nextest too");
    assert!(
        has_filterset(&disruptive_nextest),
        "sanity: the fixture's sibling job is filtered, which is what makes the fence's \
         `no filterset` assertion a real discriminator rather than a tautology: \
         {disruptive_nextest:?}"
    );

    // The two spellings the previous literal matchers got wrong, on one line:
    // comma-joined features must read as ARMED (not as a dropped feature), and a
    // space-less `-E'…'` must read as FILTERED (not as unfiltered). Getting
    // either backwards is worse than having no fence — the failure message would
    // name a cause that is not the cause.
    let tight = ci_job_block(fixture, "test-cluster-tight");
    let tight_nextest =
        nextest_invocation_in(&tight).expect("the comma-joined job runs cargo nextest too");
    assert!(
        arms_feature(&tight_nextest, FAULT_INJECTION_FEATURE),
        "`--features test-support,{FAULT_INJECTION_FEATURE}` arms the feature exactly as well as a \
         dedicated flag does; reading it as absent would redden the fence with a message asserting \
         the feature was DROPPED, which is a false red carrying a wrong diagnosis: {tight_nextest:?}"
    );
    assert!(
        has_filterset(&tight_nextest),
        "`-E'expr'` written without a space is still a filterset. Missing it would let a narrowed \
         selection past the fence, which is the exact silent deselection the fence exists to \
         catch: {tight_nextest:?}"
    );

    // The `--filterset` long form, and the substring-safety claim that licenses
    // matching the bare feature name: `test-checkpoint-fault-injection` does not
    // contain `test-fault-injection`, so a job arming only the checkpoint
    // feature must NOT read as arming this one. The real `test-cluster:` job
    // passes both, so a collision here would make the feature check
    // unfalsifiable rather than merely loose.
    assert!(
        !"test-checkpoint-fault-injection".contains(FAULT_INJECTION_FEATURE),
        "the bare-substring match in `arms_feature` is only safe while no other workspace feature \
         contains `{FAULT_INJECTION_FEATURE}`; `test-checkpoint-fault-injection` now does, so the \
         fence would pass on a job that dropped the feature it checks. Switch `arms_feature` to \
         token-aware matching."
    );
    let ckpt_only = ci_job_block(fixture, "test-cluster-checkpoint-only");
    let ckpt_nextest =
        nextest_invocation_in(&ckpt_only).expect("the checkpoint-only job runs cargo nextest too");
    assert!(
        !arms_feature(&ckpt_nextest, FAULT_INJECTION_FEATURE),
        "a job arming only `test-checkpoint-fault-injection` must not read as arming \
         `{FAULT_INJECTION_FEATURE}`: {ckpt_nextest:?}"
    );
    assert!(
        has_filterset(&ckpt_nextest),
        "the `--filterset` long form must be detected as well as `-E`: {ckpt_nextest:?}"
    );

    // And the negative cases: a job with neither needle must read as absent, not
    // as empty-string-that-happens-to-pass.
    let bare = ci_job_block(fixture, "test");
    assert_eq!(
        cluster_url_in(&bare),
        None,
        "a job that does not set the variable must read as `None`, which is what makes the \
         fence's missing-variable panic reachable"
    );
    let bare_nextest = nextest_invocation_in(&bare).expect("the bare job runs cargo nextest");
    assert!(
        !arms_feature(&bare_nextest, FAULT_INJECTION_FEATURE),
        "a job that does not pass the feature must not appear to: {bare_nextest:?}"
    );
}
