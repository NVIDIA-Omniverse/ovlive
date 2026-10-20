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

//! b.v6d — the cursor stamp's optional regression floor
//! (`scripts/stamp_cursor_if_live.lua`, decider amendment 2).
//!
//! A `CreateLayer` re-send answered `ALREADY_COMMITTED` re-spawns the recorded
//! object's creation-time first base, so for the first time two
//! `CreationBase` runs can target ONE object. The bootstrap's guard (b) was a
//! read followed by the stamp, so two overlapping runs could both pass the
//! read and both stamp, both counting a persisted generation. The stamp now
//! carries the guard's seq half atomically: an ODD-length `ARGV` leads with a
//! floor, and the script refuses (`2`) when `last_checkpoint_seq` is already at
//! or above it. The checkpoint loop's three stamp sites keep sending the plain
//! EVEN-length `ARGV`, whose behaviour must not change.
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | [`stamp_cursor_argv_spells_the_plain_and_floored_shapes`] | T21 | the two `ARGV` shapes, byte for byte |
//! | [`plain_stamp_rewrites_an_equal_or_higher_cursor_as_before`] | T21 | the even shape still re-stamps an EQUAL cursor (the chain-collapse re-base does exactly that) and has no floor at all |
//! | [`floored_stamp_refuses_at_or_above_the_floor_and_writes_below_it`] | amendment 2 | the odd shape's verdicts: absent / below → stamped; equal / above / unreadable → refused with nothing written; a gone header → absent, no phantom |
//! | [`only_the_bootstrap_sends_the_floored_shape`] | T21 | the checkpoint loop's three sites call the plain stamp and the bootstrap the floored one |
//! | [`overlapping_runs::two_overlapping_creation_base_runs_persist_and_count_once`] | T22 | two `CreationBase` runs for one object, both past the pre-stamp read: one `persisted`, one `unstamped_generation_present`, one `writes_total{success,layer}` |
//! | [`overlapping_runs::restore_rebase_refuses_a_generation_stamped_after_its_pre_stamp_read`] | T23 | the re-base floor at `seeded_seq`: a newer generation stamped in the read→stamp window is neither regressed nor has its bump cleared |
//!
//! # Feature gate
//!
//! `overlapping_runs` drives `BaseBootstrap::with_pre_stamp_hook_for_test`,
//! gated on `test-checkpoint-fault-injection`, and is compiled out without it.

use std::collections::HashMap;
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface};

use live_edit::domain::types::ObjectState;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::{
    stamp_cursor_argv, stamp_obj_header_cursor, stamp_obj_header_cursor_below, CursorStamp,
};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

/// A live layer's header key, created through `create_layer` so it carries
/// every column a real header does (`seq` above all).
async fn live_header(tv: &TestValkey, name: &str) -> (u64, String) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), "bv6d-user")
        .await
        .expect("session creation must succeed");
    let object_id = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(
            &session_id,
            name,
            "bv6d-user",
            &creation_state(),
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");
    (
        object_id,
        tv.key_schema().obj_header(&object_id.to_string()),
    )
}

fn creation_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..Default::default()
    }
}

/// The two cursor pairs the creation-site stamp writes.
fn pairs(seq: u64) -> Vec<(&'static str, String)> {
    vec![
        ("last_checkpoint_seq", seq.to_string()),
        ("last_checkpoint_ms", "1700000000000".to_owned()),
    ]
}

async fn header_field(tv: &TestValkey, key: &str, field: &str) -> Option<String> {
    tv.client()
        .pool
        .hget(key, field)
        .await
        .expect("HGET obj:hdr must succeed")
}

async fn set_header_field(tv: &TestValkey, key: &str, field: &str, value: &str) {
    let _: () = tv
        .client()
        .pool
        .hset(key, [(field, value)])
        .await
        .expect("HSET obj:hdr must succeed");
}

// ═════════════════════════════════════════════════════════════════════════════
// T21 — the two ARGV shapes
// ═════════════════════════════════════════════════════════════════════════════

/// Without a floor the `ARGV` is the flat pair list — even, and exactly what
/// every stamp site sent before the floor existed. With one, the floor leads
/// as a decimal and the length turns odd, which is how the script tells the
/// shapes apart.
#[test]
fn stamp_cursor_argv_spells_the_plain_and_floored_shapes() {
    let pairs = [
        ("last_checkpoint_seq", "7".to_owned()),
        ("last_checkpoint_ms", "123".to_owned()),
        ("restore_seq_bump", "0".to_owned()),
    ];
    let plain = stamp_cursor_argv(&pairs, None);
    assert_eq!(
        plain,
        [
            "last_checkpoint_seq",
            "7",
            "last_checkpoint_ms",
            "123",
            "restore_seq_bump",
            "0"
        ],
        "the plain shape is byte-identical to the pre-floor ARGV"
    );
    assert_eq!(plain.len() % 2, 0, "the plain shape is even");

    let floored = stamp_cursor_argv(&pairs, Some(7));
    assert_eq!(floored[0], "7", "the floor leads as a decimal");
    assert_eq!(&floored[1..], plain.as_slice(), "then the same pairs");
    assert_eq!(floored.len() % 2, 1, "the floored shape is odd");
}

/// The even shape keeps its exact pre-floor behaviour: it stamps a live header
/// whatever its `last_checkpoint_seq` already holds — including the SAME value,
/// which the chain-collapse re-base writes over its own capture's stamp, and a
/// higher one (the loop's arms carry their own equality semantics and must not
/// inherit the bootstrap's floor). A gone header is still `false` with no
/// phantom.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plain_stamp_rewrites_an_equal_or_higher_cursor_as_before() {
    let tv = TestValkey::setup().await;
    let (_, key) = live_header(&tv, "bv6d-plain-stamp").await;

    for existing in ["5", "9"] {
        set_header_field(&tv, &key, "last_checkpoint_seq", existing).await;
        set_header_field(&tv, &key, "last_checkpoint_ms", "1").await;
        let stamped = stamp_obj_header_cursor(tv.client().pool.next(), &key, &pairs(5))
            .await
            .expect("the plain stamp must succeed");
        assert!(
            stamped,
            "the plain stamp over an existing cursor of {existing} must write: it has no floor"
        );
        assert_eq!(
            header_field(&tv, &key, "last_checkpoint_seq")
                .await
                .as_deref(),
            Some("5")
        );
        assert_eq!(
            header_field(&tv, &key, "last_checkpoint_ms")
                .await
                .as_deref(),
            Some("1700000000000"),
            "the re-stamp of an equal value still writes every pair"
        );
    }

    let gone = tv.key_schema().obj_header("999999999");
    let stamped = stamp_obj_header_cursor(tv.client().pool.next(), &gone, &pairs(5))
        .await
        .expect("the plain stamp must succeed on an absent header");
    assert!(!stamped, "an absent header is not stamped");
    let exists: u64 = tv
        .client()
        .pool
        .exists(&gone)
        .await
        .expect("EXISTS must succeed");
    assert_eq!(exists, 0, "and no phantom header is created");
}

/// The odd shape's verdicts, one live header per case so each starts clean:
///
/// - no `last_checkpoint_seq`, or one below the floor → `Stamped`, every pair
///   written;
/// - one equal to or above the floor, or one that is not a plain decimal →
///   `CursorAtOrAbove`, NOTHING written (the ms column keeps its old value);
/// - a header with no `seq` → `HeaderAbsent`, no phantom created.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn floored_stamp_refuses_at_or_above_the_floor_and_writes_below_it() {
    const FLOOR: u64 = 5;
    let tv = TestValkey::setup().await;

    let cases: [(Option<&str>, CursorStamp); 6] = [
        (None, CursorStamp::Stamped),
        (Some("4"), CursorStamp::Stamped),
        (Some("5"), CursorStamp::CursorAtOrAbove),
        (Some("6"), CursorStamp::CursorAtOrAbove),
        (Some("5.0"), CursorStamp::CursorAtOrAbove),
        (Some("-1"), CursorStamp::CursorAtOrAbove),
    ];
    for (i, (existing, expected)) in cases.into_iter().enumerate() {
        let (_, key) = live_header(&tv, &format!("bv6d-floored-{i}")).await;
        if let Some(existing) = existing {
            set_header_field(&tv, &key, "last_checkpoint_seq", existing).await;
        }
        set_header_field(&tv, &key, "last_checkpoint_ms", "1").await;

        let stamp =
            stamp_obj_header_cursor_below(tv.client().pool.next(), &key, &pairs(FLOOR), FLOOR)
                .await
                .expect("the floored stamp must succeed");
        assert_eq!(stamp, expected, "existing cursor {existing:?}");

        let (seq, ms) = (
            header_field(&tv, &key, "last_checkpoint_seq").await,
            header_field(&tv, &key, "last_checkpoint_ms").await,
        );
        match expected {
            CursorStamp::Stamped => assert_eq!(
                (seq.as_deref(), ms.as_deref()),
                (Some("5"), Some("1700000000000")),
                "existing cursor {existing:?}: a stamp writes every pair"
            ),
            _ => assert_eq!(
                (seq.as_deref(), ms.as_deref()),
                (existing, Some("1")),
                "existing cursor {existing:?}: a refused stamp writes nothing"
            ),
        }
    }

    // A header without `seq` (deleted, or a phantom) is checked first.
    let phantom = tv.key_schema().obj_header("888888888");
    set_header_field(&tv, &phantom, "last_checkpoint_seq", "1").await;
    let stamp =
        stamp_obj_header_cursor_below(tv.client().pool.next(), &phantom, &pairs(FLOOR), FLOOR)
            .await
            .expect("the floored stamp must succeed on a seq-less header");
    assert_eq!(stamp, CursorStamp::HeaderAbsent);
    let fields: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&phantom)
        .await
        .expect("HGETALL must succeed");
    assert_eq!(
        fields,
        HashMap::from([("last_checkpoint_seq".to_owned(), "1".to_owned())]),
        "a seq-less header is left exactly as found"
    );
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// Source fence for the T21 split: the checkpoint loop's three stamp sites go
/// through the plain `stamp_obj_header_cursor` (whose `ARGV` the tests above pin
/// as the pre-floor shape), and the base bootstrap — the only caller with a
/// floor to send — through `stamp_obj_header_cursor_below`. A loop site moved
/// onto the floored call would refuse the chain-collapse re-base's equal
/// re-stamp; a bootstrap moved back would reopen the double-persist window.
#[test]
fn only_the_bootstrap_sends_the_floored_shape() {
    const CHECKPOINTER: &str = include_str!("../../../src/checkpoint/checkpointer.rs");
    const BOOTSTRAP: &str = include_str!("../../../src/checkpoint/base_bootstrap.rs");
    const LUA: &str = include_str!("../../../src/storage/lua.rs");

    /// Occurrences of `needle` on lines that are not `//` comments, outside
    /// the file's `#[cfg(test)] mod tests` (whose cases call the primitives
    /// directly).
    fn code_calls(source: &str, needle: &str) -> usize {
        let production = source
            .split("\n#[cfg(test)]\nmod tests {")
            .next()
            .expect("split always yields a first part");
        production
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .map(|line| line.matches(needle).count())
            .sum()
    }

    assert_eq!(
        (
            code_calls(CHECKPOINTER, "stamp_obj_header_cursor("),
            code_calls(CHECKPOINTER, "stamp_obj_header_cursor_below(")
        ),
        (3, 0),
        "the checkpoint loop's three stamp arms (periodic snapshot, tail capture, chain-collapse \
         re-base) must each call the plain, floor-less stamp"
    );
    assert_eq!(
        (
            code_calls(BOOTSTRAP, "stamp_obj_header_cursor("),
            code_calls(BOOTSTRAP, "stamp_obj_header_cursor_below(")
        ),
        (0, 1),
        "the base bootstrap stamps through the floored call, once"
    );
    let plain_body = LUA
        .split("pub async fn stamp_obj_header_cursor(")
        .nth(1)
        .and_then(|rest| rest.split("\n}\n").next())
        .expect("stamp_obj_header_cursor must be defined in src/storage/lua.rs");
    assert!(
        plain_body.contains("stamp_cursor_argv(pairs, None)"),
        "the plain stamp must send the floor-less ARGV"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T22 / T23 — overlapping runs, driven through the pre-stamp seam
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-checkpoint-fault-injection")]
mod overlapping_runs {
    use std::time::Duration;

    use futures::FutureExt;
    use tokio_util::task::TaskTracker;

    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::base_bootstrap::PreStampHook;
    use live_edit::checkpoint::{
        BaseBootstrap, BootstrapOutcome, BootstrapTarget, CheckpointStore,
    };
    use live_edit::observability::metrics::Metrics;

    use super::*;
    use crate::checkpoint::helpers::{gather_base_bootstrap_cells, read_layer_writes_counter};

    /// Bound on the two overlapping runs together. Healthy, they finish in
    /// milliseconds; a run that never reaches the barrier (a guard that
    /// declined before the stamp) would otherwise hang the other forever.
    const RUN_BUDGET: Duration = Duration::from_secs(20);

    struct Bootstrap {
        unit: BaseBootstrap,
        store: Arc<dyn CheckpointStore>,
        metrics: Arc<Metrics>,
    }

    fn bootstrap(tv: &TestValkey, hook: PreStampHook) -> Bootstrap {
        let metrics = Arc::new(crate::common::metrics_scrape_only("test-v6d"));
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let unit = BaseBootstrap::new(
            Arc::clone(&store),
            tv.client().clone(),
            tv.key_schema().clone(),
            TaskTracker::new(),
            Arc::clone(&metrics),
            "test-v6d".to_owned(),
            true,
        )
        .with_pre_stamp_hook_for_test(hook);
        Bootstrap {
            unit,
            store,
            metrics,
        }
    }

    fn cell(metrics: &Metrics, outcome: &str, site: &str) -> u64 {
        gather_base_bootstrap_cells(metrics)
            .get(&(outcome.to_owned(), site.to_owned()))
            .copied()
            .unwrap_or_else(|| panic!("base_bootstrap_total{{{outcome},{site}}} is seeded"))
            as u64
    }

    /// T22: the two runs a `CreateLayer` re-send can produce for one object —
    /// the creating call's and the re-send's re-spawn — are held at the
    /// pre-stamp seam until BOTH have passed guard (a), the PUT and guard (b)'s
    /// pre-stamp read. Then both stamp. Exactly one may: the other's floored
    /// stamp finds the first's `last_checkpoint_seq = 1` and is refused. So
    /// `persisted` and `checkpoint_writes_total{success,layer}` each tick ONCE,
    /// and the refused run lands in `unstamped_generation_present`.
    ///
    /// Before the floor both stamps wrote, and both runs counted a persisted
    /// generation for one blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_overlapping_creation_base_runs_persist_and_count_once() {
        let tv = TestValkey::setup().await;
        let (object_id, key) = live_header(&tv, "bv6d-t22").await;
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let hook: PreStampHook = {
            let barrier = barrier.clone();
            Arc::new(move || {
                let barrier = barrier.clone();
                async move {
                    barrier.wait().await;
                }
                .boxed()
            })
        };
        let bs = bootstrap(&tv, hook);

        let (first, second) = tokio::time::timeout(RUN_BUDGET, async {
            tokio::join!(
                bs.unit
                    .run_once(object_id, creation_state(), BootstrapTarget::CreationBase),
                bs.unit
                    .run_once(object_id, creation_state(), BootstrapTarget::CreationBase),
            )
        })
        .await
        .expect(
            "both runs must reach the pre-stamp barrier and finish; a hang means one declined \
             before the stamp and the interleaving under test never happened",
        );

        let mut outcomes = [first, second];
        outcomes.sort_by_key(|o| !matches!(o, BootstrapOutcome::Persisted { .. }));
        assert_eq!(
            outcomes,
            [
                BootstrapOutcome::Persisted { snap_seq: 1 },
                BootstrapOutcome::UnstampedGenerationPresent
            ],
            "exactly one of two overlapping runs may stamp and count a persisted generation"
        );
        assert_eq!(cell(&bs.metrics, "persisted", "creation_base"), 1);
        assert_eq!(
            cell(&bs.metrics, "unstamped_generation_present", "creation_base"),
            1
        );
        assert_eq!(
            read_layer_writes_counter(&bs.metrics, "success"),
            1,
            "one generation completed, so checkpoint_writes_total{{success,layer}} ticks once"
        );
        assert_eq!(
            bs.store
                .list_layer_checkpoints(object_id)
                .await
                .expect("list_layer_checkpoints must succeed")
                .len(),
            1,
            "the two PUTs targeted one key and left one generation"
        );
        assert_eq!(
            header_field(&tv, &key, "last_checkpoint_seq")
                .await
                .as_deref(),
            Some("1")
        );
    }

    /// T23: the re-base's floor is its target `seeded_seq`. A newer generation
    /// — here a second restore, stamping `last_checkpoint_seq = 12` and its own
    /// `restore_seq_bump = 5` — lands between the re-base's pre-stamp read and
    /// its stamp. The re-base must report `unstamped_generation_present` and
    /// leave both columns as the newer generation left them: stamping would
    /// push the cursor back to 10 AND clear a bump that still guards the newer
    /// restore's fossils.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_rebase_refuses_a_generation_stamped_after_its_pre_stamp_read() {
        const SEEDED_SEQ: u64 = 10;
        const ABSORBED_BAND: u64 = 3;
        let tv = TestValkey::setup().await;
        let (object_id, key) = live_header(&tv, "bv6d-t23").await;
        // The header a restore leaves: seeded writer cursor, its own stamp
        // below it, and the band it added.
        set_header_field(&tv, &key, "seq", &SEEDED_SEQ.to_string()).await;
        set_header_field(&tv, &key, "last_checkpoint_seq", "4").await;
        set_header_field(&tv, &key, "restore_seq_bump", &ABSORBED_BAND.to_string()).await;

        let newer = tv.client().pool.next().clone();
        let hook_key = key.clone();
        let hook: PreStampHook = Arc::new(move || {
            let (newer, key) = (newer.clone(), hook_key.clone());
            async move {
                let _: () = newer
                    .hset(
                        &key,
                        [("last_checkpoint_seq", "12"), ("restore_seq_bump", "5")],
                    )
                    .await
                    .expect("the newer generation's stamp must land");
            }
            .boxed()
        });
        let bs = bootstrap(&tv, hook);

        let outcome = tokio::time::timeout(
            RUN_BUDGET,
            bs.unit.run_once(
                object_id,
                ObjectState {
                    sequence_num: SEEDED_SEQ,
                    ..creation_state()
                },
                BootstrapTarget::RestoreRebase {
                    seeded_seq: SEEDED_SEQ,
                    observed_commit_span: 2,
                    absorbed_band: ABSORBED_BAND,
                },
            ),
        )
        .await
        .expect("the re-base must finish");

        assert_eq!(outcome, BootstrapOutcome::UnstampedGenerationPresent);
        assert_eq!(
            (
                header_field(&tv, &key, "last_checkpoint_seq")
                    .await
                    .as_deref(),
                header_field(&tv, &key, "restore_seq_bump").await.as_deref()
            ),
            (Some("12"), Some("5")),
            "the newer generation's cursor must not be regressed, nor its bump cleared"
        );
        assert_eq!(
            cell(
                &bs.metrics,
                "unstamped_generation_present",
                "restore_rebase"
            ),
            1
        );
        assert_eq!(read_layer_writes_counter(&bs.metrics, "success"), 0);
    }
}
