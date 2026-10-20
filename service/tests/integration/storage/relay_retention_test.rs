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

//! b.yvh — the per-object relay byte budget (`LIVE_EDIT_LAYER_STREAM_MAX_BYTES`,
//! `storage::relay_retention`) on every commit route, against Valkey.
//!
//! Every test runs on each of the three routes an `UpdateLayer` commit takes:
//! the fast-path script (a field delta on the `lua` backend), the
//! WATCH/MULTI/EXEC path (a field delta on `txn`), and the script's structural
//! mode (a `CreateSection` on `lua`). Each fixture checks, from the attempts
//! histogram, that its commits really took the route.
//!
//! After every commit the fixture checks two things against a model: the
//! header's `relay_gens` equals `RelayRetention::fold` folded over every
//! commit's relay length in Rust, byte for byte (so the script's fold and the
//! WATCH path's are the same function on every input these tests reach); and
//! the stream's last entry is `{header seq}-0`, so the byte trim never reaches
//! the commit it ran beside.
//!
//! - **The byte bound** (fails without the byte trim): twelve 20 KiB commits
//!   at a 64 KiB budget retain at most B + 2 × the largest entry, the entries
//!   the model's exact trim keeps and no others, where count-only retention
//!   keeps all thirteen. `MONITOR` shows one `XTRIM <stream> MINID ~ {cut}-0`
//!   per commit whose fold dropped any generation (one fold can drop several
//!   under a single cut), with that fold's cut, and none otherwise.
//! - **Small history survives a large commit**: forty 1000-byte commits and
//!   one 16 KiB commit fit the budget, so nothing is trimmed — the per-entry
//!   formula `clamp(⌊B/s⌋, MIN, MAXLEN)` would have cut to four entries.
//!   Commits past the budget then trim no more than the model.
//! - **The budget is per object**: on one store, a large-commit layer trims
//!   to the budget while its small-commit sibling keeps its whole window and
//!   a `relay_gens` folded from its own commits only.
//! - **A single entry above the budget** keeps the newest two entries.
//! - **The entry cap** (`MAXLEN ~`) still bounds the stream with the budget
//!   on; **B = 0** writes no column and issues no `XTRIM`.
//! - **A malformed, stale or foreign column** — every shape the strict
//!   grammar reads as fresh, leading zeros, a merge past K + 1 pairs, a drop —
//!   folds to the Rust fold's value on every route; and over one shared
//!   commit sequence the three routes' columns are byte-identical, each
//!   stream keeping every entry the fold keeps.
//! - **A lost reply charges once** (`test-fault-injection`): the fast and
//!   structural post-`EVALSHA` injector, and the WATCH path's lost `EXEC` —
//!   reconciled, or answered by the retry's commit-token check — each leave
//!   the column one fold of the commit, not two.
//! - **The column is not state**: not a structural header field, and both
//!   snapshot readers return the same state with it as without it.
//! - **Delete and recreate, and an overwrite**, start a fresh column on the
//!   new object. A restore's is in `checkpoint_apply_test.rs`.
//!
//! # Stream nodes and exactness
//!
//! `XTRIM MINID ~` and `MAXLEN ~` remove whole stream nodes only. Valkey
//! starts a new node when an entry would take the current one past
//! `stream-node-max-bytes` (4096 by default), so an entry of 20 KiB or more is
//! a node of its own and its trims are exact: those tests compare the stream
//! with the model's exact trim for equality, after checking the setting
//! (`CONFIG GET`, a read). Tests over small entries compare it as a superset
//! no more than one node's bytes larger.
//!
//! # Wait budget
//!
//! The longest test makes 61 commits and one `MONITOR` drain; each commit is a
//! few round trips and the drain is bounded by `MONITOR_DRAIN_BUDGET` (20 s),
//! far inside nextest's 120 s hang-killer.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::{ConfigInterface, HashesInterface, KeysInterface, StreamsInterface};

use live_edit::config::{UpdateLayerBackend, LAYER_STREAM_MAX_BYTES_DEFAULT};
use live_edit::domain::types::ObjectState;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::notifications::RELAY_OMPF_FIELD;
use live_edit::storage::relay_retention::{
    RelayRetention, HEADER_FIELD_RELAY_GENS, RELAY_MIN_ENTRIES,
};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::storage::snapshot_read_support::{
    monitor_fence, test_valkey_addr, MonitorTap, MonitoredCommand,
};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

const KIB: usize = 1024;

/// The test-scaled byte budget, B: 64 KiB, so T = 8 KiB.
const BUDGET: u64 = 64 * 1024;

/// A large relay: a third of [`BUDGET`], and far above the default
/// `stream-node-max-bytes`, so every such entry is a stream node of its own.
const LARGE: usize = 20 * KIB;

/// A small relay: far below T, so many share a generation.
const SMALL: usize = 1000;

/// The production entry cap.
const MAXLEN: u64 = 4096;

/// The sequence every fixture layer is created at; `CreateLayer`'s stream
/// entry is `1-0`, with no relay bytes.
const CREATE_SEQ: u64 = 1;

const WRITER: &str = "relay-retention-writer";
const CREATED_BY: &str = "relay-retention-user";

// ═════════════════════════════════════════════════════════════════════════════
// Routes
// ═════════════════════════════════════════════════════════════════════════════

/// The route an `UpdateLayer` commit takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// The fast-path script: a field delta on the `lua` backend.
    Fast,
    /// The WATCH/MULTI/EXEC path: a field delta on the `txn` backend.
    Watch,
    /// The script's structural mode: a `CreateSection` on the `lua` backend.
    Structural,
}

impl Route {
    const ALL: [Route; 3] = [Route::Fast, Route::Watch, Route::Structural];

    fn backend(self) -> UpdateLayerBackend {
        match self {
            Route::Watch => UpdateLayerBackend::Txn,
            Route::Fast | Route::Structural => UpdateLayerBackend::Lua,
        }
    }

    /// The attempts-histogram `path` labels a commit on this route lands on.
    fn paths(self) -> &'static [&'static str] {
        match self {
            Route::Fast => &["fast"],
            Route::Watch => &["watch"],
            Route::Structural => &["structural", "structural_full"],
        }
    }

    /// The `i`-th commit's delta: a fresh field, or a fresh section under the
    /// root, so every commit changes state on its route.
    fn delta(self, i: u64) -> OwnedDelta {
        match self {
            Route::Fast | Route::Watch => OwnedDelta {
                set_fields: vec![SetField {
                    section_id: 1,
                    key_name: format!("relay-retention-{i}"),
                    value_or_ext_hash: vec![(i % 251) as u8],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                }],
                ..Default::default()
            },
            Route::Structural => OwnedDelta {
                struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: 100 + i,
                    section_name: format!("RelayRetention{i}"),
                    section_type: 1,
                })],
                ..Default::default()
            },
        }
    }
}

/// `delta`'s OMPF padded with one external-value part to exactly `len` bytes.
/// The part is referenced by nothing: the store relays the container
/// verbatim and none of these tests reads it back, so only its length, which
/// is what the fold accounts, matters.
fn relay_of_len(delta: &OwnedDelta, len: usize) -> Bytes {
    let overhead = delta_to_ompf_bytes(delta, &[&[]]).len();
    assert!(
        len >= overhead,
        "fixture: a {len}-byte relay cannot hold the delta's {overhead}-byte container"
    );
    let pad = vec![0x5A_u8; len - overhead];
    let relay = delta_to_ompf_bytes(delta, &[&pad]);
    assert_eq!(relay.len(), len, "fixture: the padded relay's length");
    Bytes::from(relay)
}

// ═════════════════════════════════════════════════════════════════════════════
// The model
// ═════════════════════════════════════════════════════════════════════════════

/// The Rust fold over every commit a fixture made, and the stream an exact
/// trim would leave.
struct Model {
    retention: RelayRetention,
    /// The column the fold wrote last (`None`: never written).
    column: Option<String>,
    /// Sequence → relay bytes of every entry an exact trim keeps, the
    /// unaccounted `CreateLayer` entry included.
    retained: BTreeMap<u64, u64>,
    /// Every cut the fold named, in commit order.
    cuts: Vec<u64>,
}

impl Model {
    fn new(budget: u64) -> Self {
        Self {
            retention: RelayRetention::from_budget(budget),
            column: None,
            retained: BTreeMap::from([(CREATE_SEQ, 0)]),
            cuts: Vec::new(),
        }
    }

    fn commit(&mut self, seq: u64, relay_len: u64) {
        self.retained.insert(seq, relay_len);
        if let Some(folded) = self.retention.fold(self.column.as_deref(), seq, relay_len) {
            if let Some(cut) = folded.trim_below_seq {
                self.retained.retain(|&kept, _| kept >= cut);
                self.cuts.push(cut);
            }
            self.column = Some(folded.value);
        }
    }

    fn retained_seqs(&self) -> Vec<u64> {
        self.retained.keys().copied().collect()
    }

    fn retained_bytes(&self) -> u64 {
        self.retained.values().sum()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

struct Fx {
    tv: TestValkey,
    metrics: Arc<Metrics>,
    route: Route,
    session_id: String,
    layer: String,
    oid: u64,
    store: LayerStore,
    model: Model,
    commits: u64,
}

impl Fx {
    /// A fresh layer at [`CREATE_SEQ`] on a store with entry cap `maxlen` and
    /// byte budget `budget`, committing on `route`.
    async fn new(route: Route, layer: &str, maxlen: u64, budget: u64) -> Self {
        Self::new_with(route, layer, maxlen, budget, |store| store).await
    }

    /// [`Self::new`], with `customize` applied to the store (a test seam).
    async fn new_with(
        route: Route,
        layer: &str,
        maxlen: u64,
        budget: u64,
        customize: impl FnOnce(LayerStore) -> LayerStore,
    ) -> Self {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(layer, &format!("/{layer}"), CREATED_BY)
            .await
            .expect("session creation must succeed");
        let store = customize(
            LayerStore::new(tv.client().clone(), tv.key_schema().clone(), maxlen)
                .with_layer_stream_max_bytes(budget)
                .with_update_layer_backend(route.backend())
                .with_metrics(Arc::clone(&metrics)),
        );
        let oid = create(&store, &session_id, layer).await;
        Self {
            tv,
            metrics,
            route,
            session_id,
            layer: layer.to_owned(),
            oid,
            store,
            model: Model::new(budget),
            commits: 0,
        }
    }

    /// One commit of a `relay_len`-byte relay on the fixture's route, checked
    /// against the model: the header column is the Rust fold's, and the
    /// stream's last entry is the commit's own. Returns the commit's `seq`.
    async fn commit(&mut self, relay_len: usize) -> u64 {
        let delta = self.route.delta(self.commits);
        self.commits += 1;
        let relay = relay_of_len(&delta, relay_len);
        let (seq, _) = self
            .store
            .update_layer(
                &self.session_id,
                &self.layer,
                self.oid,
                Some(delta),
                Some(relay),
                WRITER,
                self.commits,
            )
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "{:?}: commit {} must succeed: {e:?}",
                    self.route, self.commits
                )
            });
        self.model.commit(seq, relay_len as u64);
        self.assert_matches_model(seq).await;
        seq
    }

    /// The header column equals the model's, and the stream's last entry is
    /// `{seq}-0`.
    async fn assert_matches_model(&self, seq: u64) {
        assert_eq!(
            self.relay_gens().await,
            self.model.column,
            "{:?}: after the commit at {seq} the header's relay_gens must be the Rust fold of \
             every commit's relay length, byte for byte",
            self.route
        );
        assert_eq!(
            self.top_id().await,
            Some(format!("{seq}-0")),
            "{:?}: the stream's last entry must stay the commit's own: the byte trim is \
             head-only and its cut never reaches the newest entry",
            self.route
        );
    }

    async fn relay_gens(&self) -> Option<String> {
        self.tv
            .client()
            .pool
            .hget(
                self.tv.key_schema().obj_header(&self.oid.to_string()),
                HEADER_FIELD_RELAY_GENS,
            )
            .await
            .expect("HGET relay_gens must succeed")
    }

    async fn header_seq(&self) -> u64 {
        let seq: String = self
            .tv
            .client()
            .pool
            .hget(
                self.tv.key_schema().obj_header(&self.oid.to_string()),
                "seq",
            )
            .await
            .expect("HGET seq must succeed");
        seq.parse().expect("header seq must be numeric")
    }

    fn stream_key(&self) -> String {
        self.tv.key_schema().layer_stream(self.oid)
    }

    async fn top_id(&self) -> Option<String> {
        let top: Vec<(String, HashMap<String, Bytes>)> = self
            .tv
            .client()
            .pool
            .xrevrange_values(self.stream_key(), "+", "-", Some(1))
            .await
            .expect("XREVRANGE must succeed");
        top.into_iter().next().map(|(id, _)| id)
    }

    /// The stream as `(seq, relay bytes)` per entry, in order. Every entry
    /// here is a commit at `{seq}-0`.
    async fn stream(&self) -> Vec<(u64, u64)> {
        // Values read as bytes: a commit's relay `ompf` field is binary.
        let entries: Vec<(String, HashMap<String, Bytes>)> = self
            .tv
            .client()
            .pool
            .xrange_values(self.stream_key(), "-", "+", None)
            .await
            .expect("XRANGE must succeed");
        entries
            .into_iter()
            .map(|(id, fields)| {
                let seq = id
                    .strip_suffix("-0")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        panic!("every entry here is a `{{seq}}-0` commit; got {id}")
                    });
                let relay = fields.get(RELAY_OMPF_FIELD).map_or(0, |b| b.len() as u64);
                (seq, relay)
            })
            .collect()
    }

    async fn stream_seqs(&self) -> Vec<u64> {
        self.stream()
            .await
            .into_iter()
            .map(|(seq, _)| seq)
            .collect()
    }

    async fn stream_relay_bytes(&self) -> u64 {
        self.stream()
            .await
            .into_iter()
            .map(|(_, bytes)| bytes)
            .sum()
    }

    /// Plant `value` as the header column, as a foreign writer (an older
    /// replica, a restore of another generation, a hand edit) would leave it,
    /// and make the model fold from it.
    async fn plant_column(&mut self, value: &str) {
        let _: () = self
            .tv
            .client()
            .pool
            .hset(
                self.tv.key_schema().obj_header(&self.oid.to_string()),
                [(HEADER_FIELD_RELAY_GENS, value)],
            )
            .await
            .expect("HSET relay_gens must succeed");
        self.model.column = Some(value.to_owned());
    }

    /// Every commit landed on this route's path(s), exactly once.
    fn assert_route_taken(&self) {
        let text = render(&self.metrics);
        let landed: f64 = self
            .route
            .paths()
            .iter()
            .filter_map(|path| {
                sample_value(
                    &text,
                    "live_edit_update_layer_attempts_histogram_count",
                    &[("path", path)],
                )
            })
            .sum();
        assert_eq!(
            landed,
            self.commits as f64,
            "{:?}: every commit must land on {:?}; the test did not exercise its route",
            self.route,
            self.route.paths()
        );
    }
}

async fn create(store: &LayerStore, session_id: &str, layer: &str) -> u64 {
    let state = ObjectState {
        sequence_num: CREATE_SEQ,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..Default::default()
    };
    store
        .create_layer(
            session_id,
            layer,
            CREATED_BY,
            &state,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed")
}

/// Check that a [`LARGE`] entry is a stream node of its own on this server,
/// so `~` trims of such entries are exact (see the module header). A read;
/// nothing is changed.
async fn assert_large_entries_are_nodes_of_their_own(tv: &TestValkey) -> u64 {
    let setting: HashMap<String, String> = tv
        .client()
        .pool
        .next()
        .config_get("stream-node-max-bytes")
        .await
        .expect("CONFIG GET stream-node-max-bytes must succeed");
    let max_bytes: u64 = setting
        .get("stream-node-max-bytes")
        .and_then(|v| v.parse().ok())
        .expect("CONFIG GET names stream-node-max-bytes with a decimal value");
    assert!(
        max_bytes > 0 && max_bytes < LARGE as u64,
        "fixture precondition: these tests rely on a {LARGE}-byte entry being a stream node of \
         its own (the default stream-node-max-bytes is 4096); this server has {max_bytes}"
    );
    max_bytes
}

/// Every `XTRIM` the server ran on `stream_key`, as its arguments after the
/// key.
fn xtrims(feed: &[MonitoredCommand], stream_key: &str) -> Vec<Vec<String>> {
    feed.iter()
        .filter(|c| c.name() == "XTRIM" && c.args.get(1).map(String::as_str) == Some(stream_key))
        .map(|c| c.args[2..].to_vec())
        .collect()
}

async fn drain(tv: &TestValkey, tap: &mut MonitorTap) -> Vec<MonitoredCommand> {
    let fence = monitor_fence(tv.prefix());
    let _: Option<String> = tv
        .client()
        .pool
        .get(&fence)
        .await
        .expect("the fence GET must succeed");
    tap.drain_until(&fence).await
}

// ═════════════════════════════════════════════════════════════════════════════
// The byte bound
// ═════════════════════════════════════════════════════════════════════════════

/// Twelve [`LARGE`] commits at [`BUDGET`]: the stream keeps exactly the
/// entries the model's exact trim keeps, whose relay bytes stay at most
/// B + 2 × the largest entry; count-only retention keeps all thirteen. Each
/// commit whose fold dropped any generation issues one `XTRIM … MINID ~
/// {cut}-0` with that fold's cut; no other commit issues one.
async fn large_commits_retain_at_most_the_budget_plus_two_entries(route: Route) {
    const COMMITS: usize = 12;
    let mut fx = Fx::new(route, "relay-retention-bound", MAXLEN, BUDGET).await;
    assert_large_entries_are_nodes_of_their_own(&fx.tv).await;
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    for _ in 0..COMMITS {
        fx.commit(LARGE).await;
        assert_eq!(
            fx.stream_seqs().await,
            fx.model.retained_seqs(),
            "{route:?}: a stream of {LARGE}-byte entries (one node each) must keep exactly the \
             entries the fold's cuts keep"
        );
        let retained = fx.stream_relay_bytes().await;
        assert!(
            retained <= BUDGET + RELAY_MIN_ENTRIES * LARGE as u64,
            "{route:?}: the stream retains {retained} relay bytes, above B + 2 × the largest \
             entry"
        );
    }

    assert!(
        fx.stream_seqs().await.len() < COMMITS + 1,
        "{route:?}: count-only retention would keep all {} entries; the byte budget must trim",
        COMMITS + 1
    );
    let feed = drain(&fx.tv, &mut tap).await;
    let want: Vec<Vec<String>> = fx
        .model
        .cuts
        .iter()
        .map(|cut| vec!["MINID".to_owned(), "~".to_owned(), format!("{cut}-0")])
        .collect();
    assert!(
        !want.is_empty(),
        "fixture: the model must have dropped a generation"
    );
    assert_eq!(
        xtrims(&feed, &fx.stream_key()),
        want,
        "{route:?}: exactly one XTRIM MINID ~ per commit whose fold dropped a generation, at \
         that fold's cut"
    );
    fx.assert_route_taken();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_path_large_commits_retain_at_most_the_budget_plus_two_entries() {
    large_commits_retain_at_most_the_budget_plus_two_entries(Route::Fast).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_path_large_commits_retain_at_most_the_budget_plus_two_entries() {
    large_commits_retain_at_most_the_budget_plus_two_entries(Route::Watch).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_mode_large_commits_retain_at_most_the_budget_plus_two_entries() {
    large_commits_retain_at_most_the_budget_plus_two_entries(Route::Structural).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Small history
// ═════════════════════════════════════════════════════════════════════════════

/// Forty [`SMALL`] commits and one 16 KiB commit fit [`BUDGET`]: nothing is
/// trimmed, where the per-entry formula would cut to ⌊64/16⌋ = 4 entries.
/// Twenty more small commits pass the budget; the stream then keeps every
/// entry the model keeps, and at most one stream node's bytes more.
async fn small_history_survives_one_large_commit(route: Route) {
    const SMALL_COMMITS: usize = 40;
    const LARGE_AFTER_SMALL: usize = 16 * KIB;
    const MORE_SMALL_COMMITS: usize = 20;
    let mut fx = Fx::new(route, "relay-retention-history", MAXLEN, BUDGET).await;
    let node_max_bytes = assert_large_entries_are_nodes_of_their_own(&fx.tv).await;

    for _ in 0..SMALL_COMMITS {
        fx.commit(SMALL).await;
    }
    fx.commit(LARGE_AFTER_SMALL).await;
    assert!(
        fx.model.cuts.is_empty(),
        "fixture: {SMALL_COMMITS} × {SMALL} + {LARGE_AFTER_SMALL} bytes must fit the budget"
    );
    let all: Vec<u64> = (CREATE_SEQ..=CREATE_SEQ + SMALL_COMMITS as u64 + 1).collect();
    assert_eq!(
        fx.stream_seqs().await,
        all,
        "{route:?}: a large commit must not trim cheap history that fits the budget"
    );

    for _ in 0..MORE_SMALL_COMMITS {
        fx.commit(SMALL).await;
    }
    assert!(
        !fx.model.cuts.is_empty(),
        "fixture: the further small commits must pass the budget"
    );
    let stream = fx.stream_seqs().await;
    let missing: Vec<u64> = fx
        .model
        .retained_seqs()
        .into_iter()
        .filter(|seq| !stream.contains(seq))
        .collect();
    assert!(
        missing.is_empty(),
        "{route:?}: the stream must keep every entry the fold keeps; trimmed {missing:?}"
    );
    let extra = fx.stream_relay_bytes().await - fx.model.retained_bytes();
    assert!(
        extra <= node_max_bytes,
        "{route:?}: `~` may keep the rest of the node the cut falls in ({node_max_bytes} bytes \
         at most), not {extra} bytes more than the fold keeps"
    );
    fx.assert_route_taken();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_path_small_history_survives_one_large_commit() {
    small_history_survives_one_large_commit(Route::Fast).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_path_small_history_survives_one_large_commit() {
    small_history_survives_one_large_commit(Route::Watch).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_mode_small_history_survives_one_large_commit() {
    small_history_survives_one_large_commit(Route::Structural).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Two layers on one store
// ═════════════════════════════════════════════════════════════════════════════

/// The budget is per object: one store, two layers of one session, their
/// commits interleaved. Twelve [`LARGE`] commits trim the large layer to the
/// model's exact cut, while the sibling's thirty-six [`SMALL`] commits —
/// 36000 bytes, inside [`BUDGET`] — keep its whole window, and its
/// `relay_gens` is the fold of its own commits only, untouched by the large
/// layer's (a budget charged across layers, or a column written to the wrong
/// header, fails here).
async fn sibling_layers_fold_and_trim_independently(route: Route) {
    const LARGE_COMMITS: usize = 12;
    const SMALL_PER_LARGE: usize = 3;
    const SIBLING: &str = "relay-retention-sibling-small";
    let mut fx = Fx::new(route, "relay-retention-sibling-large", MAXLEN, BUDGET).await;
    assert_large_entries_are_nodes_of_their_own(&fx.tv).await;
    let sibling_oid = create(&fx.store, &fx.session_id, SIBLING).await;
    assert_ne!(sibling_oid, fx.oid, "fixture: two distinct objects");
    let sibling_header = fx.tv.key_schema().obj_header(&sibling_oid.to_string());
    let sibling_stream = fx.tv.key_schema().layer_stream(sibling_oid);
    let mut sibling_model = Model::new(BUDGET);
    let mut sibling_commits = 0u64;

    for _ in 0..LARGE_COMMITS {
        fx.commit(LARGE).await;
        for _ in 0..SMALL_PER_LARGE {
            let delta = route.delta(1000 + sibling_commits);
            sibling_commits += 1;
            let relay = relay_of_len(&delta, SMALL);
            let (seq, _) = fx
                .store
                .update_layer(
                    &fx.session_id,
                    SIBLING,
                    sibling_oid,
                    Some(delta),
                    Some(relay),
                    WRITER,
                    1000 + sibling_commits,
                )
                .await
                .unwrap_or_else(|e| {
                    panic!("{route:?}: sibling commit {sibling_commits} must succeed: {e:?}")
                });
            sibling_model.commit(seq, SMALL as u64);
        }
        assert_eq!(
            fx.stream_seqs().await,
            fx.model.retained_seqs(),
            "{route:?}: the large layer keeps exactly the entries its own fold keeps"
        );
        let sibling_column: Option<String> = fx
            .tv
            .client()
            .pool
            .hget(&sibling_header, HEADER_FIELD_RELAY_GENS)
            .await
            .expect("HGET sibling relay_gens must succeed");
        assert_eq!(
            sibling_column, sibling_model.column,
            "{route:?}: the sibling's relay_gens is the fold of its own commits only"
        );
    }

    assert!(
        !fx.model.cuts.is_empty(),
        "fixture: the large layer's commits must pass the budget"
    );
    assert!(
        sibling_model.cuts.is_empty(),
        "fixture: the sibling's {} bytes must fit the budget",
        sibling_commits * SMALL as u64
    );
    let sibling_entries: Vec<(String, HashMap<String, Bytes>)> = fx
        .tv
        .client()
        .pool
        .xrange_values(&sibling_stream, "-", "+", None)
        .await
        .expect("XRANGE sibling stream must succeed");
    let sibling_ids: Vec<String> = sibling_entries.into_iter().map(|(id, _)| id).collect();
    let all: Vec<String> = (CREATE_SEQ..=CREATE_SEQ + sibling_commits)
        .map(|seq| format!("{seq}-0"))
        .collect();
    assert_eq!(
        sibling_ids, all,
        "{route:?}: the sibling keeps its full window while the large layer trims"
    );
    assert_ne!(
        fx.relay_gens().await,
        sibling_model.column,
        "fixture: the two layers' columns differ"
    );
    let text = render(&fx.metrics);
    let landed: f64 = route
        .paths()
        .iter()
        .filter_map(|path| {
            sample_value(
                &text,
                "live_edit_update_layer_attempts_histogram_count",
                &[("path", path)],
            )
        })
        .sum();
    assert_eq!(
        landed,
        (fx.commits + sibling_commits) as f64,
        "{route:?}: every commit of both layers must land on {:?}",
        route.paths()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_path_sibling_layers_fold_and_trim_independently() {
    sibling_layers_fold_and_trim_independently(Route::Fast).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_path_sibling_layers_fold_and_trim_independently() {
    sibling_layers_fold_and_trim_independently(Route::Watch).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_mode_sibling_layers_fold_and_trim_independently() {
    sibling_layers_fold_and_trim_independently(Route::Structural).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A single entry above the budget
// ═════════════════════════════════════════════════════════════════════════════

/// Commits each larger than [`BUDGET`]: from the second on, the stream holds
/// exactly the newest two entries (MIN = 2).
async fn single_entry_above_the_budget_keeps_the_two_newest(route: Route) {
    const ABOVE_BUDGET: usize = 80 * KIB;
    let mut fx = Fx::new(route, "relay-retention-above", MAXLEN, BUDGET).await;
    assert_large_entries_are_nodes_of_their_own(&fx.tv).await;

    fx.commit(ABOVE_BUDGET).await;
    for _ in 0..3 {
        let seq = fx.commit(ABOVE_BUDGET).await;
        assert_eq!(
            fx.stream_seqs().await,
            vec![seq - 1, seq],
            "{route:?}: an entry above the budget leaves the newest {RELAY_MIN_ENTRIES} entries"
        );
    }
    fx.assert_route_taken();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_path_single_entry_above_the_budget_keeps_the_two_newest() {
    single_entry_above_the_budget_keeps_the_two_newest(Route::Fast).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_path_single_entry_above_the_budget_keeps_the_two_newest() {
    single_entry_above_the_budget_keeps_the_two_newest(Route::Watch).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_mode_single_entry_above_the_budget_keeps_the_two_newest() {
    single_entry_above_the_budget_keeps_the_two_newest(Route::Structural).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The entry cap, and the budget off
// ═════════════════════════════════════════════════════════════════════════════

/// With the default budget on, an entry cap of 3 still bounds a stream of
/// [`LARGE`] entries (one node each) at 3 entries; the budget writes its
/// column but never trims, since 8 × 20 KiB is far inside 128 MiB.
async fn entry_cap_still_bounds_the_stream(route: Route) {
    const CAP: u64 = 3;
    let mut fx = Fx::new(
        route,
        "relay-retention-cap",
        CAP,
        LAYER_STREAM_MAX_BYTES_DEFAULT,
    )
    .await;
    assert_large_entries_are_nodes_of_their_own(&fx.tv).await;
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let mut seq = 0;
    for _ in 0..8 {
        seq = fx.commit(LARGE).await;
    }
    assert_eq!(
        fx.stream_seqs().await,
        vec![seq - 2, seq - 1, seq],
        "{route:?}: MAXLEN ~ {CAP} must still bound the stream with the byte budget on"
    );
    assert!(
        fx.model.cuts.is_empty() && fx.model.column.is_some(),
        "fixture: the default budget folds every commit and drops nothing here"
    );
    let feed = drain(&fx.tv, &mut tap).await;
    assert_eq!(
        xtrims(&feed, &fx.stream_key()),
        Vec::<Vec<String>>::new(),
        "{route:?}: no generation dropped, so no byte trim"
    );
    fx.assert_route_taken();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_path_entry_cap_still_bounds_the_stream() {
    entry_cap_still_bounds_the_stream(Route::Fast).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_path_entry_cap_still_bounds_the_stream() {
    entry_cap_still_bounds_the_stream(Route::Watch).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_mode_entry_cap_still_bounds_the_stream() {
    entry_cap_still_bounds_the_stream(Route::Structural).await;
}

/// `LIVE_EDIT_LAYER_STREAM_MAX_BYTES=0`: no commit writes `relay_gens` and
/// none issues an `XTRIM`, however far past 64 KiB the relay bytes go — the
/// count-only retention of the no-redeploy rollback.
async fn budget_off_writes_no_column_and_no_trim(route: Route) {
    const COMMITS: usize = 6;
    let mut fx = Fx::new(route, "relay-retention-off", MAXLEN, 0).await;
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    for _ in 0..COMMITS {
        fx.commit(LARGE).await;
    }
    assert_eq!(
        fx.relay_gens().await,
        None,
        "{route:?}: with the budget off no commit writes relay_gens"
    );
    assert_eq!(
        fx.stream_seqs().await.len(),
        COMMITS + 1,
        "{route:?}: with the budget off nothing is trimmed below the entry cap"
    );
    let feed = drain(&fx.tv, &mut tap).await;
    assert_eq!(
        xtrims(&feed, &fx.stream_key()),
        Vec::<Vec<String>>::new(),
        "{route:?}: with the budget off no commit issues an XTRIM"
    );
    fx.assert_route_taken();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_path_budget_off_writes_no_column_and_no_trim() {
    budget_off_writes_no_column_and_no_trim(Route::Fast).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_path_budget_off_writes_no_column_and_no_trim() {
    budget_off_writes_no_column_and_no_trim(Route::Watch).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_mode_budget_off_writes_no_column_and_no_trim() {
    budget_off_writes_no_column_and_no_trim(Route::Structural).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Foreign columns, and route parity
// ═════════════════════════════════════════════════════════════════════════════

/// Every column a commit can find that this route did not write: each shape
/// the strict grammar reads as fresh, a start above the next commit (a
/// restored or re-materialized object behind a stale column), leading zeros,
/// nine generations left by a smaller budget (the fold merges past K + 1),
/// and a valid column the commit drops a generation of. Each folds to the
/// Rust fold's value on every route, which pins the script's parser and its
/// merge and drop steps to the Rust twin's.
async fn foreign_columns_fold_as_the_rust_fold_does(route: Route) {
    /// B = 800, T = 100: the planted byte counts below are hand-sized to it.
    const TINY_BUDGET: u64 = 800;
    const RELAY: usize = 512;
    const MERGE: &str = "nine generations from a smaller budget";
    const DROP: &str = "a drop";
    let mut fx = Fx::new(route, "relay-retention-foreign", MAXLEN, TINY_BUDGET).await;
    // Ten commits first, so a planted start as low as `new_seq - 9` is valid.
    for _ in 0..10 {
        fx.commit(RELAY).await;
    }

    // Each shape, built from the sequence the commit that reads it lands at.
    type Shape = (&'static str, fn(u64) -> String);
    let shapes: [Shape; 11] = [
        ("not a pair", |_| "garbage".to_owned()),
        ("empty", |_| String::new()),
        ("a trailing separator", |_| "1:2,".to_owned()),
        ("a sign", |_| "+1:2".to_owned()),
        ("three numbers", |_| "1:2:3".to_owned()),
        ("a repeated start", |_| "2:5,2:6".to_owned()),
        ("bytes above 2^53 - 1", |_| "1:9007199254740992".to_owned()),
        ("a start above the commit", |new_seq| {
            format!("{}:5", new_seq + 1)
        }),
        ("leading zeros", |_| "007:003".to_owned()),
        // T is 100 here; these ten-byte generations were closed under a
        // budget eighty times smaller.
        (MERGE, |new_seq| {
            (new_seq - 9..new_seq)
                .map(|start| format!("{start}:10"))
                .collect::<Vec<_>>()
                .join(",")
        }),
        // 700 + the commit's 512 is above 800: the older one drops.
        (DROP, |new_seq| format!("{}:700,{new_seq}:0", new_seq - 1)),
    ];
    for (why, shape) in shapes {
        let new_seq = fx.header_seq().await + 1;
        let value = shape(new_seq);
        let expected = fx
            .model
            .retention
            .fold(Some(&value), new_seq, RELAY as u64)
            .expect("the budget is on");
        fx.plant_column(&value).await;
        fx.commit(RELAY).await;
        assert_eq!(
            fx.relay_gens().await,
            Some(expected.value.clone()),
            "{route:?}: {why} ({value:?}) must fold as the Rust fold does"
        );
        match why {
            MERGE => assert_eq!(
                expected.value.split(',').count(),
                9,
                "fixture: this column must exercise the merge past K + 1 pairs"
            ),
            DROP => assert_eq!(
                expected.trim_below_seq,
                Some(new_seq - 1),
                "fixture: this column must exercise a drop"
            ),
            _ => {}
        }
    }
    fx.assert_route_taken();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_path_foreign_columns_fold_as_the_rust_fold_does() {
    foreign_columns_fold_as_the_rust_fold_does(Route::Fast).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_path_foreign_columns_fold_as_the_rust_fold_does() {
    foreign_columns_fold_as_the_rust_fold_does(Route::Watch).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_mode_foreign_columns_fold_as_the_rust_fold_does() {
    foreign_columns_fold_as_the_rust_fold_does(Route::Structural).await;
}

/// One commit sequence — small and large relays, trims included — on the
/// three routes side by side: after every commit the three columns are
/// byte-identical, and every route's stream keeps every entry the fold keeps.
/// (Which extra entries below a cut `~` keeps depends on how small entries
/// share stream nodes, so the streams are compared through the fold.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn relay_gens_and_the_retained_stream_are_identical_on_every_route() {
    const LENGTHS: [usize; 12] = [
        SMALL,
        LARGE,
        SMALL,
        SMALL,
        LARGE,
        LARGE,
        SMALL,
        LARGE,
        40 * KIB,
        SMALL,
        LARGE,
        SMALL,
    ];
    let mut fxs = Vec::new();
    for route in Route::ALL {
        fxs.push(Fx::new(route, "relay-retention-parity", MAXLEN, BUDGET).await);
    }
    for (i, len) in LENGTHS.into_iter().enumerate() {
        let mut columns = Vec::new();
        for fx in &mut fxs {
            fx.commit(len).await;
            columns.push((fx.route, fx.relay_gens().await));
            let stream = fx.stream_seqs().await;
            let trimmed: Vec<u64> = fx
                .model
                .retained_seqs()
                .into_iter()
                .filter(|seq| !stream.contains(seq))
                .collect();
            assert!(
                trimmed.is_empty(),
                "commit {i}: {:?} trimmed {trimmed:?}, which the fold keeps",
                fx.route
            );
        }
        assert!(
            columns.windows(2).all(|w| w[0].1 == w[1].1),
            "commit {i}: relay_gens must be byte-identical on every route; got {columns:?}"
        );
    }
    assert!(
        !fxs[0].model.cuts.is_empty(),
        "fixture: the sequence must trim"
    );
    for fx in &fxs {
        fx.assert_route_taken();
    }
}

/// The column is bookkeeping, not state: it is not one of the structural
/// header fields, a header carrying it is structurally complete and served by
/// `read_object_header`, and the atomic `object_snapshot_read.lua` returns the
/// same state with it as without it, matching the reference reader both times
/// (so catch-up and checkpoint payloads never carry it). The fenced read is
/// held to the atomic read's state after commits that write the column by
/// `snapshot_read_fenced_test`'s equality cases.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn relay_gens_is_not_structural_and_every_snapshot_reader_ignores_it() {
    use crate::storage::snapshot_read_support::{assert_atomic_equals_reference, view};
    use live_edit::storage::layer::{
        header_fields_are_structurally_complete, HEADER_STRUCTURAL_FIELDS,
    };

    assert!(
        !HEADER_STRUCTURAL_FIELDS.contains(&HEADER_FIELD_RELAY_GENS),
        "`{HEADER_FIELD_RELAY_GENS}` must not be structural, or every object without it (made \
         before the byte budget, or with it off) reads as structurally incomplete"
    );
    let mut fx = Fx::new(Route::Fast, "relay-retention-readers", MAXLEN, BUDGET).await;
    for _ in 0..3 {
        fx.commit(SMALL).await;
    }
    let header_key = fx.tv.key_schema().obj_header(&fx.oid.to_string());
    let header: HashMap<String, String> = fx
        .tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed");
    assert!(
        header.contains_key(HEADER_FIELD_RELAY_GENS),
        "fixture: the commits wrote the column"
    );
    assert!(
        header_fields_are_structurally_complete(&header),
        "a header carrying the column is structurally complete"
    );
    fx.store
        .read_object_header(fx.oid)
        .await
        .expect("a header carrying the column is served by read_object_header");

    let with = assert_atomic_equals_reference(
        fx.tv.client(),
        fx.tv.key_schema(),
        fx.oid,
        "with relay_gens",
    )
    .await;
    let removed: u64 = fx
        .tv
        .client()
        .pool
        .hdel(&header_key, HEADER_FIELD_RELAY_GENS)
        .await
        .expect("HDEL relay_gens must succeed");
    assert_eq!(removed, 1);
    let without = assert_atomic_equals_reference(
        fx.tv.client(),
        fx.tv.key_schema(),
        fx.oid,
        "without relay_gens",
    )
    .await;
    assert_eq!(
        view(&with.state),
        view(&without.state),
        "the snapshot readers must return the same state with the column as without it"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// A lost reply charges once
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod lost_reply {
    //! A commit whose reply is lost is folded once: its retry finds the commit
    //! landed (the script's `ALREADY_LANDED`, the WATCH path's reconcile or
    //! its commit-token check) and writes nothing, so the column is one fold
    //! of the commit. A second fold of a 20 KiB relay would show as 20 KiB
    //! more bytes in the newest generation, and here a different drop.

    use std::collections::VecDeque;
    use std::sync::Mutex;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use live_edit::storage::watched_txn::{WatchedExecOverride, WatchedExecOverrideQueue};

    use super::*;

    type FredQueue = Arc<Mutex<VecDeque<FredError>>>;

    fn empty_fred_queue() -> FredQueue {
        Arc::new(Mutex::new(VecDeque::new()))
    }

    fn lost_reply() -> FredError {
        FredError::new(FredErrorKind::Timeout, "relay-retention: reply lost")
    }

    /// What stages the lost reply on each route.
    enum Staging {
        /// The post-`EVALSHA` injector: the script commits, its reply is lost.
        PostEval(FredQueue),
        /// The WATCH path's `EXEC` commits and its verdict is replaced by
        /// in-doubt; with `reconcile` the reconcile read fails too, so the
        /// retry's commit-token check answers.
        Exec {
            overrides: WatchedExecOverrideQueue,
            reconcile: FredQueue,
            fail_reconcile: bool,
        },
    }

    async fn lost_reply_charges_once(route: Route, fail_reconcile: bool) {
        let staging = match route {
            Route::Fast | Route::Structural => Staging::PostEval(empty_fred_queue()),
            Route::Watch => Staging::Exec {
                overrides: Arc::new(Mutex::new(VecDeque::new())),
                reconcile: empty_fred_queue(),
                fail_reconcile,
            },
        };
        let mut fx = Fx::new_with(
            route,
            "relay-retention-lost-reply",
            MAXLEN,
            BUDGET,
            |store| match &staging {
                Staging::PostEval(queue) => {
                    store.with_lua_post_eval_fred_error_injector_for_test(Arc::clone(queue))
                }
                Staging::Exec {
                    overrides,
                    reconcile,
                    ..
                } => store
                    .with_watched_exec_override_for_test(Arc::clone(overrides))
                    .with_reconcile_xrange_fred_error_injector_for_test(Arc::clone(reconcile)),
            },
        )
        .await;
        // History first, so the lost commit's fold drops a generation.
        for _ in 0..3 {
            fx.commit(LARGE).await;
        }
        assert!(
            fx.model.cuts.is_empty(),
            "fixture: three 20 KiB commits fit 64 KiB"
        );

        match &staging {
            Staging::PostEval(queue) => queue.lock().unwrap().push_back(lost_reply()),
            Staging::Exec {
                overrides,
                reconcile,
                fail_reconcile,
            } => {
                overrides
                    .lock()
                    .unwrap()
                    .push_back(WatchedExecOverride::ReplaceOutcomeAfterExec);
                if *fail_reconcile {
                    reconcile.lock().unwrap().push_back(lost_reply());
                }
            }
        }
        // `Fx::commit` checks the column against ONE fold of this commit.
        let seq = fx.commit(LARGE).await;
        assert_eq!(
            fx.model.cuts.len(),
            1,
            "fixture: the lost commit's single fold drops a generation"
        );

        let drained = match &staging {
            Staging::PostEval(queue) => queue.lock().unwrap().is_empty(),
            Staging::Exec {
                overrides,
                reconcile,
                ..
            } => overrides.lock().unwrap().is_empty() && reconcile.lock().unwrap().is_empty(),
        };
        assert!(
            drained,
            "{route:?}: the staged lost reply must have been consumed"
        );
        let stream = fx.stream_seqs().await;
        assert_eq!(
            stream.iter().filter(|s| **s == seq).count(),
            1,
            "{route:?}: the lost commit landed once"
        );
        assert_eq!(fx.header_seq().await, seq, "{route:?}: one commit landed");
        let already_landed = |backend: &str| {
            sample_value(
                &render(&fx.metrics),
                "live_edit_update_layer_commit_already_landed_total",
                &[("backend", backend)],
            )
            .expect("commit_already_landed_total cells are seeded at zero")
        };
        assert_eq!(
            (already_landed("lua"), already_landed("txn")),
            match (route, fail_reconcile) {
                (Route::Watch, true) => (0.0, 1.0),
                (Route::Watch, false) => (0.0, 0.0),
                _ => (1.0, 0.0),
            },
            "{route:?}: the retry found the commit by the route's own lost-reply answer"
        );
        fx.assert_route_taken();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fast_path_lost_reply_is_charged_once() {
        lost_reply_charges_once(Route::Fast, false).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structural_mode_lost_reply_is_charged_once() {
        lost_reply_charges_once(Route::Structural, false).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn watch_path_lost_exec_reconciled_is_charged_once() {
        lost_reply_charges_once(Route::Watch, false).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn watch_path_lost_exec_answered_by_the_commit_token_is_charged_once() {
        lost_reply_charges_once(Route::Watch, true).await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// A new object starts a fresh column
// ═════════════════════════════════════════════════════════════════════════════

/// A column belongs to its object: after a delete and a re-create, and after a
/// `CreateLayer` overwrite, the new object's header carries no `relay_gens`,
/// and its first commit folds a fresh one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_recreate_and_overwrite_start_a_fresh_column() {
    let mut fx = Fx::new(Route::Fast, "relay-retention-fresh", MAXLEN, BUDGET).await;
    for _ in 0..5 {
        fx.commit(LARGE).await;
    }
    assert!(
        !fx.model.cuts.is_empty(),
        "fixture: the old object's column has dropped a generation"
    );

    fx.store
        .delete_layer(
            &fx.session_id,
            &fx.layer,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("delete_layer must succeed");
    let recreated = create(&fx.store, &fx.session_id, &fx.layer).await;
    assert_ne!(recreated, fx.oid, "a re-create is a new object");
    fx.oid = recreated;
    fx.model = Model::new(BUDGET);
    assert_eq!(
        fx.relay_gens().await,
        None,
        "a re-created object's header carries no relay_gens"
    );
    fx.commit(LARGE).await;

    let overwritten = create(&fx.store, &fx.session_id, &fx.layer).await;
    assert_ne!(overwritten, fx.oid, "an overwrite is a new object");
    fx.oid = overwritten;
    fx.model = Model::new(BUDGET);
    assert_eq!(
        fx.relay_gens().await,
        None,
        "an overwriting object's header carries no relay_gens"
    );
    fx.commit(LARGE).await;
    assert_eq!(
        fx.relay_gens().await,
        Some(format!("{}:{LARGE},{}:0", CREATE_SEQ + 1, CREATE_SEQ + 2)),
        "the first commit on the new object folds a fresh column at its own sequence"
    );
}
