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

//! b.yvh F5 — a backing store that refuses an upload's write before applying
//! anything (`-OOM`, `-MISCONF`, `-READONLY`, `-NOREPLICAS`) is answered
//! `UNAVAILABLE` "the backing store refused a write this request needed
//! because <condition>; nothing was applied, retry", once, and counted on
//! `live_edit_store_write_refused_total{stage, reason}` — instead of the
//! `INTERNAL "internal error"` it used to reach the client as.
//!
//! Every refusal is staged with [`ScriptReplyProxy`]: the store under test
//! connects through the proxy, and the proxy answers the one command (or one
//! transaction) the rule names with the refusal, as the server's own reply,
//! so fred's reply parsing and the service's classification run as they do in
//! production. Nothing changes the shared server's configuration; the two
//! tests that drive a real `-OOM` run only against their own Valkey (the
//! disruptive tests at the bottom: the upload write sites, and b.ua9's
//! retirement of a replaced and a deleted object on a full store).
//!
//! Each site asserts: the answer and its message (which never starts with the
//! overtaken prefix); on the upload stages (`ext_store`, `update_commit`) its
//! retry hint — `STORE_WRITE_REFUSED` with metadata `condition` = the
//! refusal's `reason` label and a `retry_delay` in `[2 s, 6 s)`, the client's
//! "this call applied nothing" signal — and at `create_commit` its absence:
//! every `CreateLayer` refusal answers plain `UNAVAILABLE` "… because
//! <condition>; retry", with no nothing-applied claim (SDD §18.1); nothing
//! applied (the keys the write would have changed are unchanged); one attempt
//! (the rule's hits — the update retry loop retries only `Aborted`, and a full
//! store does not clear within its milliseconds); and the one metric cell. The
//! sites:
//!
//! | stage | site |
//! |---|---|
//! | `update_commit` | the commit script, fast and structural mode (all four refusals on the fast path) |
//! | `update_commit` | the WATCH path's discarded transaction: refused at `EXEC` time, and at queue time |
//! | `ext_store` | the external-value pre-store |
//! | `create_commit` | the object-id allocation, the materialization `EXEC` (both shapes), the overwrite's `superseded` mark, a new layer's commit script |
//! | `create_commit` | the absent counter's repair writes: the unhooked `SET NX`, and through the checkpoint repair hook the `SET NX` (no counters sidecar; two per attempt) and the seed `EVAL` |
//!
//! Also: a commit script refused on attempt 2, after attempt 1's `EVALSHA`
//! timed out, answers plain `UNAVAILABLE`, outcome unknown, without the hint;
//! a discarded transaction whose cause is neither a store refusal nor the
//! store not serving keeps its non-retriable `INTERNAL`; an overwrite's commit
//! refused after the old object was marked superseded is not "nothing applied":
//! it answers the mapper's plain `UNAVAILABLE` and is not counted; all twelve
//! cells are seeded at zero; and through the `UpdateLayer` handler both the
//! commit and the pre-store refusal reach the client as `UNAVAILABLE` with the
//! message and the retry hint, decoded from the status's details.
//!
//! b.xd1 adds two sections. A discarded transaction is classified from its
//! cause: `CreateSession`'s commit discarded for a queued `-OOM` answers plain
//! `UNAVAILABLE` once, and for a queued `-LOADING` is retried; the WATCH path's
//! commit discarded for `-BUSY`, at `EXEC` time or at queue time, is retried
//! and commits once. And a store write refusal at a mapper site — outside the
//! counted upload sites — answers plain `UNAVAILABLE` naming the command and
//! the condition, once and uncounted: `DeleteLayer`'s mapping `DEL` refused
//! `-MISCONF` (the mapping intact) and `JoinSession`'s presence-index `SADD`
//! refused `-NOREPLICAS`.
//!
//! b.pa1 adds that an `-OOM`, at an upload site (the pre-store, and again once
//! the request interval has passed) or at a mapper site (`CreateSession`'s
//! discarded commit), requests the GC's out-of-memory collection: the request
//! key holds the refusing replica's label. So does an overwrite's commit run
//! refused `-OOM` whose next run lands, which no mapper ever sees: the commit
//! dispatch notes every failed run. (b.ua9's own `-OOM` sites — the overwrite
//! tail's wake-up, the orphan sweep's veto, wake-up and completion — are
//! pinned in `b_ua9_tests.rs`.) The real-`-OOM` test's store and
//! `maxmemory` guard are shared with `oom_collection_test.rs`
//! (`super::disruptive_valkey`).
//!
//! b.mhq adds that the external-value pre-store refused with an ext-mirror
//! sender wired takes no charge from the mirror's byte budget and sends
//! nothing: the mirror is offered a value only after its store commits.
//!
//! A WATCH-path transaction is staged as a whole, since a reply inside `MULTI`
//! is only ever `+QUEUED`: the rule answers the transaction's `MULTI` with
//! `+OK`, each queued command with `+QUEUED` (or, for the queue-time shape,
//! the first with the refusal), and its `EXEC` with the `EXECABORT` the server
//! sends — so no command of it reaches the store at all, which is exactly what
//! a discarded transaction is. The WATCH path's witness-baseline probe
//! (`MULTI CLIENT ID CLIENT ID EXEC`, sent with the `WATCH` before every
//! commit's transaction) is let through to the store.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, StreamsInterface};

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::counters::CountersSidecar;
use live_edit::checkpoint::{ext_mirror_channel, CheckpointObjectIdCounterRepair, CheckpointStore};
use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::{
    refusal_metadata, refusal_reason, ServiceError, STORE_WRITE_REFUSED_RETRY_DELAY,
};
use live_edit::observability::metrics::test_util::{render, sample_sum, sample_value};
use live_edit::observability::metrics::{
    Metrics, STORE_WRITE_REFUSED_REASONS, STORE_WRITE_REFUSED_STAGES,
};
use live_edit::storage::client::STORE_NOT_SERVING_SHAPES;
use live_edit::storage::counters::{CounterStore, ObjectIdCounterRepair};
use live_edit::storage::external_values::{
    ext_content_hash, ExternalValueStore, MockExternalValueStore, ValkeyExternalValueStore,
    EXT_STORE_VALUE_SCRIPT,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::{LayerStore, OPERATION_OVERTAKEN_MESSAGE};
use live_edit::storage::lua::{
    UpdateScriptMode, CREATE_LAYER_COMMIT_SCRIPT, UPDATE_LAYER_FAST_SCRIPT,
};
use live_edit::storage::session::SessionStore;
use live_edit::storage::store_oom::{OOM_COLLECTION_REQUEST_TTL, STORE_OOM_REQUEST_MIN_INTERVAL};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use crate::common::refusal_detail::assert_refusal_error;
use crate::storage::disruptive_valkey::{
    assert_noeviction, disruptive_valkey_url, RestoreMaxmemory,
};
use crate::storage::oom_request_support::wait_for_oom_request;
use crate::storage::snapshot_read_support::{
    error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// The refusals
// ═════════════════════════════════════════════════════════════════════════════

/// One store write refusal: the reply the server sends, the `reason` label it
/// is counted under, and the condition the client message names.
struct Refusal {
    reply: &'static str,
    reason: &'static str,
    condition: &'static str,
}

const OOM: Refusal = Refusal {
    reply: "OOM command not allowed when used memory > 'maxmemory'.",
    reason: "out_of_memory",
    condition: "it is out of memory",
};

const REFUSALS: [Refusal; 4] = [
    OOM,
    Refusal {
        reply: "MISCONF Valkey is configured to save RDB snapshots, but it's currently unable to \
                persist to disk.",
        reason: "misconf",
        condition: "it cannot persist data to disk",
    },
    Refusal {
        reply: "READONLY You can't write against a read only replica.",
        reason: "readonly",
        condition: "the node it reached is a read-only replica",
    },
    Refusal {
        reply: "NOREPLICAS Not enough good replicas to write.",
        reason: "noreplicas",
        condition: "too few of its replicas are reachable to accept writes",
    },
];

/// The client-facing message for `refusal`, word for word.
fn refusal_message(refusal: &Refusal) -> String {
    format!(
        "the backing store refused a write this request needed because {}; nothing was applied, \
         retry",
        refusal.condition
    )
}

/// Assert `err` carries the store refusal's retry hint —
/// `STORE_WRITE_REFUSED`, metadata `condition` = `refusal`'s `reason` label,
/// a `retry_delay` in `[2 s, 6 s)` — on the `UNAVAILABLE` a client receives,
/// and return its message.
#[track_caller]
fn assert_store_refusal_detail<'a>(err: &'a ServiceError, refusal: &Refusal) -> &'a str {
    assert_refusal_error(
        err,
        refusal_reason::STORE_WRITE_REFUSED,
        &[(refusal_metadata::CONDITION, refusal.reason)],
        STORE_WRITE_REFUSED_RETRY_DELAY,
    )
}

/// Assert `result` is the store refusal's `UNAVAILABLE`, with its message and
/// its retry hint.
#[track_caller]
fn assert_refused<T: std::fmt::Debug>(result: &Result<T, ServiceError>, refusal: &Refusal) {
    let err = match result {
        Err(err) => err,
        Ok(other) => panic!(
            "a store write refusal ({}) must answer UNAVAILABLE with the nothing-applied \
             message; got Ok({other:?})",
            refusal.reply
        ),
    };
    let message = assert_store_refusal_detail(err, refusal);
    assert_eq!(message, refusal_message(refusal));
    assert!(
        !message.starts_with(OPERATION_OVERTAKEN_MESSAGE),
        "a store refusal must never read as the overtaken answer, which clients match"
    );
}

/// The `create_commit` stage's message for `refusal`, word for word: no
/// nothing-applied claim.
fn create_refusal_message(refusal: &Refusal) -> String {
    format!(
        "the backing store refused a write this request needed because {}; retry",
        refusal.condition
    )
}

/// Assert `result` is a `CreateLayer` store refusal's answer: plain
/// `UNAVAILABLE` with [`create_refusal_message`] and no retry hint — no
/// `CreateLayer` commit-stage refusal carries one (SDD §18.1).
#[track_caller]
fn assert_create_refused<T: std::fmt::Debug>(result: &Result<T, ServiceError>, refusal: &Refusal) {
    match result {
        Err(ServiceError::Unavailable(message)) => {
            assert_eq!(message, &create_refusal_message(refusal));
        }
        other => panic!(
            "a CreateLayer store write refusal ({}) must answer plain UNAVAILABLE, without a \
             retry hint; got {other:?}",
            refusal.reply
        ),
    }
}

/// Every `live_edit_store_write_refused_total` cell, `(stage, reason) → value`.
/// Panics on an absent cell: all twelve are seeded at zero.
fn refused_cells(metrics: &Metrics) -> BTreeMap<(String, String), f64> {
    let text = render(metrics);
    let mut cells = BTreeMap::new();
    for stage in STORE_WRITE_REFUSED_STAGES {
        for reason in STORE_WRITE_REFUSED_REASONS {
            let value = sample_value(
                &text,
                "live_edit_store_write_refused_total",
                &[("stage", stage), ("reason", reason)],
            )
            .unwrap_or_else(|| {
                panic!("store_write_refused_total{{stage={stage},reason={reason}}} is absent")
            });
            cells.insert(((*stage).to_owned(), (*reason).to_owned()), value);
        }
    }
    cells
}

/// Assert exactly one refusal was counted, on `(stage, reason)`.
#[track_caller]
fn assert_counted_once(metrics: &Metrics, stage: &str, reason: &str) {
    let ticked: Vec<((String, String), f64)> = refused_cells(metrics)
        .into_iter()
        .filter(|(_, v)| *v != 0.0)
        .collect();
    assert_eq!(
        ticked,
        vec![((stage.to_owned(), reason.to_owned()), 1.0)],
        "exactly one refusal, on stage={stage} reason={reason}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Proxy rules
// ═════════════════════════════════════════════════════════════════════════════

fn command_name(args: &[Vec<u8>]) -> String {
    args.first()
        .map(|a| String::from_utf8_lossy(a).to_ascii_uppercase())
        .unwrap_or_default()
}

fn has_arg(args: &[Vec<u8>], wanted: &str) -> bool {
    args.iter().any(|a| a.as_slice() == wanted.as_bytes())
}

/// A rule that answers every `EVALSHA` / `EVAL` of the script with SHA `sha`
/// that also carries `with_arg` (when given) with `reply`, while `armed`.
fn script_refusal(
    armed: Arc<AtomicBool>,
    sha: String,
    with_arg: Option<&'static str>,
    reply: &'static str,
) -> ProxyRule {
    Arc::new(move |args: &[Vec<u8>]| {
        if !armed.load(Ordering::SeqCst) {
            return None;
        }
        let name = command_name(args);
        let is_script = (name == "EVALSHA" || name == "EVAL")
            && args
                .get(1)
                .is_some_and(|a| String::from_utf8_lossy(a).eq_ignore_ascii_case(&sha));
        (is_script && with_arg.is_none_or(|arg| has_arg(args, arg)))
            .then(|| ProxyAction::Eval(error_reply_script(reply)))
    })
}

/// A Lua body answering a status reply `+<status>`.
fn status_reply_script(status: &str) -> String {
    format!("return redis.status_reply('{status}')")
}

/// A rule that stages one discarded transaction, while `armed`: of the
/// `MULTI`s sent after arming, the first `skip` pass through untouched (the
/// WATCH path's witness-baseline probe, `MULTI CLIENT ID CLIENT ID EXEC`,
/// precedes each commit's transaction), and the next one is answered `+OK`,
/// each command queued after it `+QUEUED` — or, with `queue_time_refusal`, the
/// first one that refusal — and its `EXEC` `exec_reply`. None of that one
/// transaction reaches the store; every later `MULTI` passes through again.
/// `multis` counts every `MULTI` sent while armed, so a test can tell one
/// attempt (`skip + 1`) from a retry. Commands on other connections while the
/// transaction is staged would be caught too, so each test arms it only
/// around one call with nothing else in flight.
struct DiscardedTxn {
    armed: Arc<AtomicBool>,
    multis: Arc<AtomicUsize>,
}

impl DiscardedTxn {
    fn rule(
        skip: usize,
        queue_time_refusal: Option<&'static str>,
        exec_reply: &'static str,
    ) -> (Self, ProxyRule) {
        let armed = Arc::new(AtomicBool::new(false));
        let multis = Arc::new(AtomicUsize::new(0));
        let in_multi = Arc::new(AtomicBool::new(false));
        let queued = Arc::new(AtomicUsize::new(0));
        let rule: ProxyRule = {
            let (armed, multis) = (Arc::clone(&armed), Arc::clone(&multis));
            Arc::new(move |args: &[Vec<u8>]| {
                if !armed.load(Ordering::SeqCst) {
                    return None;
                }
                let name = command_name(args);
                if !in_multi.load(Ordering::SeqCst) {
                    if name == "MULTI" && multis.fetch_add(1, Ordering::SeqCst) == skip {
                        in_multi.store(true, Ordering::SeqCst);
                        queued.store(0, Ordering::SeqCst);
                        return Some(ProxyAction::Eval(status_reply_script("OK")));
                    }
                    return None;
                }
                if name == "EXEC" {
                    in_multi.store(false, Ordering::SeqCst);
                    return Some(ProxyAction::Eval(error_reply_script(exec_reply)));
                }
                let first = queued.fetch_add(1, Ordering::SeqCst) == 0;
                match queue_time_refusal {
                    Some(refusal) if first => Some(ProxyAction::Eval(error_reply_script(refusal))),
                    _ => Some(ProxyAction::Eval(status_reply_script("QUEUED"))),
                }
            })
        };
        (Self { armed, multis }, rule)
    }

    fn arm(&self) {
        self.armed.store(true, Ordering::SeqCst);
    }

    fn disarm(&self) {
        self.armed.store(false, Ordering::SeqCst);
    }

    fn multis(&self) -> usize {
        self.multis.load(Ordering::SeqCst)
    }
}

/// The transactions a WATCH-path attempt sends before its commit's: the
/// witness-baseline probe.
const WATCH_PROBE_TRANSACTIONS: usize = 1;

/// `EXEC`'s reply after a queued command was refused.
const EXECABORT_PREVIOUS_ERRORS: &str =
    "EXECABORT Transaction discarded because of previous errors.";

/// `EXEC`'s reply when the `EXEC` itself is refused, out of memory.
const EXECABORT_OOM: &str =
    "EXECABORT Transaction discarded because of: OOM command not allowed when used memory > \
     'maxmemory'.";

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// A `TestValkey` whose every connection goes through a [`ScriptReplyProxy`]
/// running `rule`, plus a metrics registry and a session.
struct Fx {
    // Declared before `proxy` so it drops (and cleans up through it) first.
    tv: TestValkey,
    proxy: ScriptReplyProxy,
    metrics: Arc<Metrics>,
    session_id: String,
}

const POOL_SIZE: u32 = 4;
const WRITER: &str = "store-refusal-writer";
const CREATED_BY: &str = "store-refusal-user";

impl Fx {
    async fn new(name: &str, rule: ProxyRule) -> Self {
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
        let tv = TestValkey::setup_with_url(proxy.url(), POOL_SIZE).await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(name));
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(name, &format!("/{name}"), CREATED_BY)
            .await
            .expect("session creation must succeed");
        Self {
            tv,
            proxy,
            metrics,
            session_id,
        }
    }

    fn store(&self, backend: UpdateLayerBackend) -> LayerStore {
        LayerStore::new(self.tv.client().clone(), self.tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend)
            .with_metrics(Arc::clone(&self.metrics))
    }

    async fn create(&self, store: &LayerStore, layer: &str) -> Result<u64, ServiceError> {
        let state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..Default::default()
        };
        store
            .create_layer(
                &self.session_id,
                layer,
                CREATED_BY,
                &state,
                Arc::new(MockExternalValueStore::new()),
            )
            .await
    }

    async fn mapping(&self, layer: &str) -> Option<String> {
        self.tv
            .client()
            .pool
            .get(self.tv.key_schema().layer_mapping(&self.session_id, layer))
            .await
            .expect("GET layer mapping must succeed")
    }

    /// What an update commit writes: the header, the stream's length and the
    /// commit-token record.
    async fn commit_surface(
        &self,
        oid: u64,
    ) -> (HashMap<String, String>, u64, HashMap<String, String>) {
        let oid_str = oid.to_string();
        let pool = &self.tv.client().pool;
        let header: HashMap<String, String> = pool
            .hgetall(self.tv.key_schema().obj_header(&oid_str))
            .await
            .expect("HGETALL obj:hdr must succeed");
        let xlen: u64 = pool
            .xlen(self.tv.key_schema().layer_stream(oid))
            .await
            .expect("XLEN must succeed");
        let records: HashMap<String, String> = pool
            .hgetall(self.tv.key_schema().obj_write_token(&oid_str))
            .await
            .expect("HGETALL obj:wtok must succeed");
        (header, xlen, records)
    }
}

fn field_delta() -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "store-refusal-field".to_owned(),
            value_or_ext_hash: vec![0x5E],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    }
}

fn section_delta() -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 1,
            section_name: "StoreRefusalSection".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The counter's vocabulary
// ═════════════════════════════════════════════════════════════════════════════

/// All twelve `{stage, reason}` cells exist at zero before any refusal, so
/// the `increase(...{reason="out_of_memory"}[5m]) > 0` page has a series from
/// the start.
#[tokio::test]
async fn store_write_refused_total_seeds_all_twelve_cells_at_zero() {
    let metrics = crate::common::metrics_scrape_only("store-refusal-seed");
    let cells = refused_cells(&metrics);
    assert_eq!(cells.len(), 12, "three stages × four reasons");
    assert!(
        cells.values().all(|v| *v == 0.0),
        "every cell starts at zero; got {cells:?}"
    );
    let stages: Vec<&str> = STORE_WRITE_REFUSED_STAGES.to_vec();
    assert_eq!(stages, ["ext_store", "update_commit", "create_commit"]);
    let reasons: Vec<&str> = STORE_WRITE_REFUSED_REASONS.to_vec();
    assert_eq!(
        reasons,
        ["out_of_memory", "misconf", "readonly", "noreplicas"]
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// UpdateLayer — the commit script
// ═════════════════════════════════════════════════════════════════════════════

/// The commit script in `mode`, refused with `refusal`: `UNAVAILABLE`, nothing
/// applied, one attempt, one `update_commit` cell.
async fn commit_script_refusal_is_answered_unavailable(
    mode: UpdateScriptMode,
    delta: OwnedDelta,
    refusal: &Refusal,
) {
    let armed = Arc::new(AtomicBool::new(false));
    let fx = Fx::new(
        "store-refusal-script",
        script_refusal(
            Arc::clone(&armed),
            UPDATE_LAYER_FAST_SCRIPT.sha1().to_string(),
            Some(mode.as_str()),
            refusal.reply,
        ),
    )
    .await;
    let store = fx.store(UpdateLayerBackend::Lua);
    let oid = fx
        .create(&store, "store-refusal-layer")
        .await
        .expect("create_layer must succeed");
    let before = fx.commit_surface(oid).await;

    armed.store(true, Ordering::SeqCst);
    let result = store
        .update_layer_relayed(
            &fx.session_id,
            "store-refusal-layer",
            oid,
            Some(delta),
            WRITER,
            1,
        )
        .await;
    armed.store(false, Ordering::SeqCst);

    assert_refused(&result, refusal);
    assert_eq!(
        fx.proxy.hits(),
        1,
        "{mode:?}: one attempt — a store refusal is not retried in-process"
    );
    assert_eq!(
        fx.commit_surface(oid).await,
        before,
        "{mode:?}: nothing applied — header, stream and commit-token record unchanged"
    );
    assert_counted_once(&fx.metrics, "update_commit", refusal.reason);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_path_commit_refused_by_each_store_refusal_is_answered_unavailable() {
    for refusal in &REFUSALS {
        commit_script_refusal_is_answered_unavailable(
            UpdateScriptMode::Fast,
            field_delta(),
            refusal,
        )
        .await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_mode_commit_refused_out_of_memory_is_answered_unavailable() {
    commit_script_refusal_is_answered_unavailable(
        UpdateScriptMode::Structural,
        section_delta(),
        &OOM,
    )
    .await;
}

/// The commit script refused out of memory on attempt 2, after attempt 1's
/// `EVALSHA` timed out — a lost reply, whose commit may still land — answers
/// plain `UNAVAILABLE`, outcome unknown: no retry hint, no nothing-applied
/// claim, not the overtaken prefix. The refusal is still counted once. (The
/// same refusal on attempt 1 keeps its hint:
/// [`fast_path_commit_refused_by_each_store_refusal_is_answered_unavailable`].)
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_script_refused_out_of_memory_after_a_timed_out_attempt_is_plain_unavailable() {
    let armed = Arc::new(AtomicBool::new(false));
    let fx = Fx::new(
        "store-refusal-after-retry",
        script_refusal(
            Arc::clone(&armed),
            UPDATE_LAYER_FAST_SCRIPT.sha1().to_string(),
            Some(UpdateScriptMode::Fast.as_str()),
            OOM.reply,
        ),
    )
    .await;
    let timed_out = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from([
        fred::error::Error::new(fred::error::ErrorKind::Timeout, "synthetic lost reply"),
    ])));
    let store = fx
        .store(UpdateLayerBackend::Lua)
        .with_update_layer_retry_max_attempts(2)
        .with_lua_fred_error_injector_for_test(timed_out);
    let oid = fx
        .create(&store, "store-refusal-layer")
        .await
        .expect("create_layer must succeed");
    let before = fx.commit_surface(oid).await;

    armed.store(true, Ordering::SeqCst);
    let result = store
        .update_layer_relayed(
            &fx.session_id,
            "store-refusal-layer",
            oid,
            Some(field_delta()),
            WRITER,
            1,
        )
        .await;
    armed.store(false, Ordering::SeqCst);

    let message = match &result {
        Err(ServiceError::Unavailable(message)) => message,
        other => panic!("a refusal after a counted retry must be plain UNAVAILABLE; got {other:?}"),
    };
    assert_eq!(
        message,
        &format!(
            "UpdateLayer outcome unknown: an earlier attempt of this call may have committed \
             before attempt 2 was refused (the backing store refused a write this request \
             needed because {})",
            OOM.condition
        )
    );
    assert_eq!(
        fx.proxy.hits(),
        1,
        "attempt 1's EVALSHA was never sent; attempt 2's was refused"
    );
    assert_eq!(fx.commit_surface(oid).await, before, "nothing applied");
    assert_counted_once(&fx.metrics, "update_commit", OOM.reason);
}

// ═════════════════════════════════════════════════════════════════════════════
// UpdateLayer — the WATCH path's discarded transaction
// ═════════════════════════════════════════════════════════════════════════════

/// A WATCH-path commit whose transaction the store discarded: one attempt,
/// nothing applied, no torn-commit handling (no void-WATCH tick, no `obj:ext`
/// recount), and the answer `expected` says — the store refusal's
/// `UNAVAILABLE` (counted), or for a cause that is neither a store refusal
/// nor transient the non-retriable `INTERNAL` (uncounted). A transient cause
/// is retried instead: [`watch_commit_discarded_then_retried`].
async fn watch_commit_discarded(
    queue_time_refusal: Option<&'static str>,
    exec_reply: &'static str,
    expected: Option<&Refusal>,
) {
    let (txn, rule) = DiscardedTxn::rule(WATCH_PROBE_TRANSACTIONS, queue_time_refusal, exec_reply);
    let fx = Fx::new("store-refusal-watch", rule).await;
    let store = fx.store(UpdateLayerBackend::Txn);
    let oid = fx
        .create(&store, "store-refusal-layer")
        .await
        .expect("create_layer must succeed");
    let before = fx.commit_surface(oid).await;

    txn.arm();
    let result = store
        .update_layer_relayed(
            &fx.session_id,
            "store-refusal-layer",
            oid,
            Some(field_delta()),
            WRITER,
            1,
        )
        .await;
    txn.disarm();

    assert_eq!(
        txn.multis(),
        WATCH_PROBE_TRANSACTIONS + 1,
        "one attempt — its baseline probe and its commit transaction: the discarded \
         transaction is not retried in-process"
    );
    assert_eq!(
        fx.commit_surface(oid).await,
        before,
        "nothing applied: a discarded transaction runs no queued command"
    );
    match expected {
        Some(refusal) => {
            assert_refused(&result, refusal);
            assert_counted_once(&fx.metrics, "update_commit", refusal.reason);
        }
        None => {
            assert!(
                matches!(result, Err(ServiceError::Internal(_))),
                "a transaction discarded for any other cause stays INTERNAL; got {result:?}"
            );
            assert!(
                refused_cells(&fx.metrics).values().all(|v| *v == 0.0),
                "and is not counted as a store refusal"
            );
        }
    }
    assert_eq!(
        sample_value(
            &render(&fx.metrics),
            "live_edit_storage_watch_voided_total",
            &[
                ("path", "update_layer"),
                ("disposition", "committed_unwatched")
            ],
        ),
        Some(0.0),
        "a discarded transaction is no evidence of a void WATCH"
    );
    assert_eq!(
        sample_sum(
            &render(&fx.metrics),
            "live_edit_ext_index_recount_total",
            &[("site", "torn_commit")],
        ),
        Some(0.0),
        "a discarded transaction ran nothing, so it skips the torn-commit handling: no \
         `obj:ext` recount (which the torn `ExecError` arm runs on every error shape)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_commit_whose_exec_is_refused_out_of_memory_is_answered_unavailable() {
    watch_commit_discarded(None, EXECABORT_OOM, Some(&OOM)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_commit_whose_queued_write_is_refused_out_of_memory_is_answered_unavailable() {
    watch_commit_discarded(Some(OOM.reply), EXECABORT_PREVIOUS_ERRORS, Some(&OOM)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_commit_whose_queued_write_is_refused_read_only_is_answered_unavailable() {
    watch_commit_discarded(
        Some(REFUSALS[2].reply),
        EXECABORT_PREVIOUS_ERRORS,
        Some(&REFUSALS[2]),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_commit_discarded_for_any_other_cause_stays_internal() {
    watch_commit_discarded(
        Some("ERR unknown command 'NOSUCH', with args beginning with: "),
        EXECABORT_PREVIOUS_ERRORS,
        None,
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The external-value pre-store
// ═════════════════════════════════════════════════════════════════════════════

/// The pre-store's script refused: `UNAVAILABLE`, the value not stored, one
/// attempt, one `ext_store` cell.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_value_pre_store_refused_out_of_memory_is_answered_unavailable() {
    let armed = Arc::new(AtomicBool::new(false));
    let fx = Fx::new(
        "store-refusal-ext",
        script_refusal(
            Arc::clone(&armed),
            EXT_STORE_VALUE_SCRIPT.sha1().to_string(),
            None,
            OOM.reply,
        ),
    )
    .await;
    let ext = ValkeyExternalValueStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone())
        .with_metrics(Arc::clone(&fx.metrics));
    let blob: &[u8] = b"store-refusal external value";
    let hash = ext_content_hash(blob);

    armed.store(true, Ordering::SeqCst);
    let result = ext.store_values(&[(&hash, blob)]).await;
    armed.store(false, Ordering::SeqCst);

    assert_refused(&result, &OOM);
    assert_eq!(fx.proxy.hits(), 1, "one attempt");
    let hex = sha256_to_hex(&hash);
    let pool = &fx.tv.client().pool;
    let stored: i64 = pool
        .exists(vec![
            fx.tv.key_schema().ext_meta(&hex),
            fx.tv.key_schema().ext_data(&hex),
        ])
        .await
        .expect("EXISTS must succeed");
    assert_eq!(stored, 0, "nothing stored: no data, no reference count");
    assert_counted_once(&fx.metrics, "ext_store", "out_of_memory");
}

/// T9 (b.mhq): the pre-store refused with an ext-mirror sender wired takes no
/// charge from the mirror's byte budget and sends nothing: the value's copy
/// is made before the script, but charged and sent only after the script
/// commits. The whole budget is available after the refusal, nothing is
/// queued, no byte is counted held and no drop is counted for any reason.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_value_pre_store_refused_takes_no_mirror_charge_and_sends_nothing() {
    let armed = Arc::new(AtomicBool::new(false));
    let fx = Fx::new(
        "store-refusal-ext-mirror",
        script_refusal(
            Arc::clone(&armed),
            EXT_STORE_VALUE_SCRIPT.sha1().to_string(),
            None,
            OOM.reply,
        ),
    )
    .await;
    let (sender, mut rx) = ext_mirror_channel(
        16,
        std::num::NonZeroU64::new(1 << 20).expect("non-zero"),
        Arc::clone(&fx.metrics),
    );
    let ext = ValkeyExternalValueStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone())
        .with_metrics(Arc::clone(&fx.metrics))
        .with_ext_mirror_sender(sender.clone());
    let blob: &[u8] = b"store-refusal mirrored external value";
    let hash = ext_content_hash(blob);

    armed.store(true, Ordering::SeqCst);
    let result = ext.store_values(&[(&hash, blob)]).await;
    armed.store(false, Ordering::SeqCst);

    assert_refused(&result, &OOM);
    assert_eq!(fx.proxy.hits(), 1, "one attempt");
    assert_eq!(
        sender.available_budget_bytes(),
        sender.budget_bytes(),
        "a refused store takes no charge"
    );
    assert!(rx.try_recv().is_err(), "a refused store sends nothing");
    let text = render(&fx.metrics);
    assert_eq!(
        sample_sum(&text, "live_edit_checkpoint_ext_mirror_bytes", &[]),
        Some(0.0),
        "no byte counted held"
    );
    assert_eq!(
        sample_sum(&text, "live_edit_checkpoint_ext_mirror_dropped_total", &[]),
        Some(0.0),
        "no drop counted: nothing was offered to the mirror"
    );

    // Fixture self-check: unarmed, the same store commits and sends, charged.
    ext.store_values(&[(&hash, blob)])
        .await
        .expect("an unrefused store commits");
    let item = rx.try_recv().expect("a committed store sends its value");
    assert_eq!(item.bytes().as_ref(), blob);
    assert!(
        sender.available_budget_bytes() < sender.budget_bytes(),
        "a committed store's value is charged"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// CreateLayer
// ═════════════════════════════════════════════════════════════════════════════

/// The object-id allocation refused: `UNAVAILABLE`, no id allocated, no
/// mapping, one `create_commit` cell.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_allocation_refused_out_of_memory_is_answered_unavailable() {
    let armed = Arc::new(AtomicBool::new(false));
    // The allocation is the only script that names the object-id counter key.
    let counter_key: Arc<std::sync::Mutex<String>> = Arc::new(std::sync::Mutex::new(String::new()));
    let rule: ProxyRule = {
        let (armed, counter_key) = (Arc::clone(&armed), Arc::clone(&counter_key));
        Arc::new(move |args: &[Vec<u8>]| {
            if !armed.load(Ordering::SeqCst) {
                return None;
            }
            let name = command_name(args);
            let key = counter_key.lock().unwrap().clone();
            ((name == "EVALSHA" || name == "EVAL") && has_arg(args, &key))
                .then(|| ProxyAction::Eval(error_reply_script(OOM.reply)))
        })
    };
    let fx = Fx::new("store-refusal-alloc", rule).await;
    *counter_key.lock().unwrap() = fx.tv.key_schema().counter_object_id();
    let store = fx.store(UpdateLayerBackend::Lua);
    let counter_before: Option<String> = fx
        .tv
        .client()
        .pool
        .get(fx.tv.key_schema().counter_object_id())
        .await
        .expect("GET counter must succeed");

    armed.store(true, Ordering::SeqCst);
    let result = fx.create(&store, "store-refusal-new").await;
    armed.store(false, Ordering::SeqCst);

    assert_create_refused(&result, &OOM);
    assert_eq!(fx.proxy.hits(), 1, "one attempt");
    let counter_after: Option<String> = fx
        .tv
        .client()
        .pool
        .get(fx.tv.key_schema().counter_object_id())
        .await
        .expect("GET counter must succeed");
    assert_eq!(counter_after, counter_before, "no id allocated");
    assert_eq!(fx.mapping("store-refusal-new").await, None, "no mapping");
    assert_counted_once(&fx.metrics, "create_commit", "out_of_memory");
}

// ── The object-id counter's repair writes ──────────────────────────────────
//
// A `CreateLayer` that finds the counter absent repairs it before allocating:
// with no checkpoint adapter, `SET NX counter 0`; through the checkpoint repair
// hook, `SET NX` when the adapter holds no counters sidecar, else the seed
// script (`EVAL`, a conditional `SET` to the sidecar's id + the safety
// margin). Each is one command whose only write Valkey refuses before it
// runs, so a refusal leaves the counter absent: `UNAVAILABLE` with the
// nothing-applied sentence, counted at `create_commit`, no in-process retry.
// A fresh test prefix holds no counter, so every test here starts absent.

/// The sidecar's `max_object_id_observed` for the seed case.
const SIDECAR_MAX_OBJECT_ID: u64 = 41;
/// The repair's safety margin: a test value below the shipped default
/// (`CHECKPOINT_COUNTER_SAFETY_MARGIN_DEFAULT`, 1,000,000), kept small so
/// [`SEED_ARG`] stays readable.
const REPAIR_SAFETY_MARGIN: u64 = 1000;
/// The seed script's one argument: [`SIDECAR_MAX_OBJECT_ID`] +
/// [`REPAIR_SAFETY_MARGIN`]. The allocation script takes no argument, so this
/// is what tells the seed `EVAL` from it at the proxy.
const SEED_ARG: &str = "1041";

/// A rule refusing, while `armed`, every command named `names` that names the
/// object-id counter key (set in `counter_key` once the fixture knows its
/// prefix) and carries the argument `with_arg`.
fn counter_write_refusal(
    armed: Arc<AtomicBool>,
    counter_key: Arc<std::sync::Mutex<String>>,
    names: &'static [&'static str],
    with_arg: &'static str,
) -> ProxyRule {
    Arc::new(move |args: &[Vec<u8>]| {
        if !armed.load(Ordering::SeqCst) {
            return None;
        }
        let key = counter_key.lock().unwrap().clone();
        (names.contains(&command_name(args).as_str())
            && has_arg(args, &key)
            && has_arg(args, with_arg))
        .then(|| ProxyAction::Eval(error_reply_script(OOM.reply)))
    })
}

/// The raw `le:counter:object_id` value, `None` when absent.
async fn raw_counter(fx: &Fx) -> Option<String> {
    fx.tv
        .client()
        .pool
        .get(fx.tv.key_schema().counter_object_id())
        .await
        .expect("GET counter must succeed")
}

/// One `CreateLayer` against an absent counter whose repair write the rule
/// refuses, on a store with the checkpoint repair hook when `hook` names the
/// adapter's counters sidecar (`Some(None)`: an adapter holding none), and the
/// unhooked `SET NX` repair when `hook` is `None`. Returns the answer, with
/// the counter checked still absent and no mapping written.
async fn create_against_a_refused_counter_repair(
    names: &'static [&'static str],
    with_arg: &'static str,
    hook: Option<Option<u64>>,
) -> (Fx, Result<u64, ServiceError>) {
    let armed = Arc::new(AtomicBool::new(false));
    let counter_key = Arc::new(std::sync::Mutex::new(String::new()));
    let fx = Fx::new(
        "store-refusal-counter-repair",
        counter_write_refusal(
            Arc::clone(&armed),
            Arc::clone(&counter_key),
            names,
            with_arg,
        ),
    )
    .await;
    *counter_key.lock().unwrap() = fx.tv.key_schema().counter_object_id();
    assert_eq!(
        raw_counter(&fx).await,
        None,
        "fixture: a fresh prefix holds no counter"
    );
    let mut store = fx.store(UpdateLayerBackend::Lua);
    if let Some(sidecar) = hook {
        let adapter: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        if let Some(max_object_id_observed) = sidecar {
            let body = serde_json::to_vec(&CountersSidecar {
                max_object_id_observed,
                updated_at: "2026-10-05T00:00:00Z".to_owned(),
            })
            .expect("CountersSidecar serialization must succeed");
            adapter
                .put_counters_sidecar(bytes::Bytes::from(body))
                .await
                .expect("put_counters_sidecar must succeed on the in-memory adapter");
        }
        // Wired as `main.rs` wires the hook's counter store: with the
        // metrics handle, so its refusals are counted.
        let counters = Arc::new(
            CounterStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone())
                .with_metrics(Arc::clone(&fx.metrics)),
        );
        let repair: Arc<dyn ObjectIdCounterRepair> =
            Arc::new(CheckpointObjectIdCounterRepair::new(
                adapter,
                counters,
                REPAIR_SAFETY_MARGIN,
                Arc::clone(&fx.metrics),
                std::time::Duration::from_secs(10),
                std::time::Duration::from_secs(10),
            ));
        store = store.with_object_id_counter_repair(repair);
    }

    armed.store(true, Ordering::SeqCst);
    let result = fx.create(&store, "store-refusal-new").await;
    armed.store(false, Ordering::SeqCst);

    assert_eq!(
        raw_counter(&fx).await,
        None,
        "the counter is unchanged: a refused repair write established nothing"
    );
    assert_eq!(fx.mapping("store-refusal-new").await, None, "no mapping");
    (fx, result)
}

/// The hooked repair's answer: plain `UNAVAILABLE` whose message names the
/// failed repair and ends with the refused write's sentence, with no retry
/// hint — no `CreateLayer` commit-stage refusal carries one (SDD §18.1).
#[track_caller]
fn assert_repair_refused(result: &Result<u64, ServiceError>) {
    let message = match result {
        Err(ServiceError::Unavailable(message)) => message,
        other => panic!(
            "a refused counter repair write must answer plain UNAVAILABLE, without a retry \
             hint; got {other:?}"
        ),
    };
    assert!(
        message.starts_with("object-id counter repair ")
            && message.ends_with(&create_refusal_message(&OOM)),
        "the hooked repair answers UNAVAILABLE naming the repair, with the refused write's \
         sentence; got {message:?}"
    );
}

/// No adapter: the unhooked `SET NX` refused. The F5 answer word for word,
/// one command, one `create_commit` cell, and no `unhooked_initialized` tick
/// (nothing was initialized).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_unhooked_counter_repair_refused_out_of_memory_is_answered_unavailable() {
    let (fx, result) = create_against_a_refused_counter_repair(&["SET"], "NX", None).await;
    assert_create_refused(&result, &OOM);
    assert_eq!(
        fx.proxy.hits(),
        1,
        "one SET NX: a refused repair write is not retried in-process"
    );
    assert_counted_once(&fx.metrics, "create_commit", "out_of_memory");
    assert_eq!(
        sample_value(
            &render(&fx.metrics),
            "live_edit_object_id_counter_repair_total",
            &[("outcome", "unhooked_initialized")],
        )
        .unwrap_or(0.0),
        0.0,
        "the counter was not established, so the unhooked repair is not counted as done"
    );
}

/// Through the checkpoint repair hook, an adapter holding no counters
/// sidecar: both `SET NX` writes refused — the seed routine's own (logged and
/// dropped) and the attempt's postcondition write — each one command, so two
/// commands and two `create_commit` ticks for one repair attempt, and the
/// attempt books `failed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_hooked_counter_repair_without_a_sidecar_refused_is_answered_unavailable() {
    let (fx, result) = create_against_a_refused_counter_repair(&["SET"], "NX", Some(None)).await;
    assert_repair_refused(&result);
    assert_eq!(
        fx.proxy.hits(),
        2,
        "two SET NX calls of one command each — the seed routine's and the attempt's own — and \
         neither retried in-process"
    );
    assert_eq!(
        refused_cells(&fx.metrics)
            .into_iter()
            .filter(|(_, v)| *v != 0.0)
            .collect::<Vec<_>>(),
        vec![(
            ("create_commit".to_owned(), "out_of_memory".to_owned()),
            2.0
        )],
        "each refused SET NX counts once, at create_commit"
    );
    assert_eq!(
        sample_value(
            &render(&fx.metrics),
            "live_edit_object_id_counter_repair_total",
            &[("outcome", "failed")],
        ),
        Some(1.0),
        "the repair attempt that could not establish the counter books failed"
    );
}

/// Through the checkpoint repair hook, an adapter holding a counters
/// sidecar: the seed script refused. One `EVAL`, one `create_commit` tick, and
/// the attempt books `failed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_hooked_counter_repair_seed_refused_out_of_memory_is_answered_unavailable() {
    assert_eq!(
        SEED_ARG,
        (SIDECAR_MAX_OBJECT_ID + REPAIR_SAFETY_MARGIN).to_string(),
        "fixture: the rule matches the seed the repair derives from the sidecar"
    );
    let (fx, result) = create_against_a_refused_counter_repair(
        &["EVALSHA", "EVAL"],
        SEED_ARG,
        Some(Some(SIDECAR_MAX_OBJECT_ID)),
    )
    .await;
    assert_repair_refused(&result);
    assert_eq!(
        fx.proxy.hits(),
        1,
        "one seed EVAL: a refused repair write is not retried in-process"
    );
    assert_counted_once(&fx.metrics, "create_commit", "out_of_memory");
    assert_eq!(
        sample_value(
            &render(&fx.metrics),
            "live_edit_object_id_counter_repair_total",
            &[("outcome", "failed")],
        ),
        Some(1.0),
        "the repair attempt whose seed was refused books failed"
    );
}

/// The materialization's transaction discarded, refused at `EXEC` time or at
/// queue time: `UNAVAILABLE`, no header, no mapping, one `create_commit` cell.
async fn create_layer_materialization_discarded(
    queue_time_refusal: Option<&'static str>,
    exec_reply: &'static str,
) {
    // The materialization (a fred `Transaction`, no witness probe) is the
    // first transaction a new layer's CreateLayer sends.
    let (txn, rule) = DiscardedTxn::rule(0, queue_time_refusal, exec_reply);
    let fx = Fx::new("store-refusal-materialize", rule).await;
    let store = fx.store(UpdateLayerBackend::Lua);

    txn.arm();
    let result = fx.create(&store, "store-refusal-new").await;
    txn.disarm();

    assert_create_refused(&result, &OOM);
    assert_eq!(
        txn.multis(),
        1,
        "one attempt: nothing after the refused materialization opens a transaction"
    );
    assert_eq!(fx.mapping("store-refusal-new").await, None, "no mapping");
    let counter: Option<String> = fx
        .tv
        .client()
        .pool
        .get(fx.tv.key_schema().counter_object_id())
        .await
        .expect("GET counter must succeed");
    let allocated = counter.expect("the allocation ran before the materialization");
    let header_exists: bool = fx
        .tv
        .client()
        .pool
        .exists(fx.tv.key_schema().obj_header(&allocated))
        .await
        .expect("EXISTS must succeed");
    assert!(!header_exists, "nothing materialized for the allocated id");
    assert_counted_once(&fx.metrics, "create_commit", "out_of_memory");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_materialization_refused_out_of_memory_at_exec_is_answered_unavailable() {
    create_layer_materialization_discarded(None, EXECABORT_OOM).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_materialization_refused_out_of_memory_at_queue_time_is_answered_unavailable()
{
    create_layer_materialization_discarded(Some(OOM.reply), EXECABORT_PREVIOUS_ERRORS).await;
}

/// A new layer's commit script refused: `UNAVAILABLE`, no mapping, one
/// `create_commit` cell.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_new_layer_commit_refused_out_of_memory_is_answered_unavailable() {
    let armed = Arc::new(AtomicBool::new(false));
    let fx = Fx::new(
        "store-refusal-create-commit",
        script_refusal(
            Arc::clone(&armed),
            CREATE_LAYER_COMMIT_SCRIPT.sha1().to_string(),
            None,
            OOM.reply,
        ),
    )
    .await;
    let store = fx.store(UpdateLayerBackend::Lua);

    armed.store(true, Ordering::SeqCst);
    let result = fx.create(&store, "store-refusal-new").await;
    armed.store(false, Ordering::SeqCst);

    assert_create_refused(&result, &OOM);
    assert_eq!(
        fx.proxy.hits(),
        1,
        "one attempt: a new layer is not tried again"
    );
    assert_eq!(fx.mapping("store-refusal-new").await, None, "no mapping");
    assert_counted_once(&fx.metrics, "create_commit", "out_of_memory");
}

/// An overwrite's `superseded` mark refused: `UNAVAILABLE`, the old object
/// neither superseded nor unmapped, one `create_commit` cell.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_mark_refused_out_of_memory_is_answered_unavailable() {
    let armed = Arc::new(AtomicBool::new(false));
    let rule: ProxyRule = {
        let armed = Arc::clone(&armed);
        Arc::new(move |args: &[Vec<u8>]| {
            (armed.load(Ordering::SeqCst)
                && command_name(args) == "HSET"
                && has_arg(args, "superseded"))
            .then(|| ProxyAction::Eval(error_reply_script(OOM.reply)))
        })
    };
    let fx = Fx::new("store-refusal-mark", rule).await;
    let store = fx.store(UpdateLayerBackend::Lua);
    let old = fx
        .create(&store, "store-refusal-layer")
        .await
        .expect("the first create must succeed");

    armed.store(true, Ordering::SeqCst);
    let result = fx.create(&store, "store-refusal-layer").await;
    armed.store(false, Ordering::SeqCst);

    assert_create_refused(&result, &OOM);
    assert_eq!(fx.proxy.hits(), 1, "one attempt");
    assert_eq!(
        fx.mapping("store-refusal-layer").await,
        Some(old.to_string()),
        "the mapping still names the old object"
    );
    let superseded: Option<String> = fx
        .tv
        .client()
        .pool
        .hget(
            fx.tv.key_schema().obj_header(&old.to_string()),
            "superseded",
        )
        .await
        .expect("HGET superseded must succeed");
    assert_eq!(superseded, None, "the old object is not marked superseded");
    assert_counted_once(&fx.metrics, "create_commit", "out_of_memory");
}

/// The scope boundary: an overwrite's commit script refused AFTER the old
/// object was marked superseded is not "nothing applied" — the mark landed —
/// so it keeps the overwrite arm's path: its one further try after the
/// mapping check, then the mapper's answer to the commit's own refusal — a
/// plain `UNAVAILABLE` naming the script and the condition (b.xd1), with no
/// `STORE_WRITE_REFUSED` hint and no nothing-applied claim — and not counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_commit_refused_after_the_mark_is_not_a_nothing_applied_refusal() {
    let armed = Arc::new(AtomicBool::new(false));
    let fx = Fx::new(
        "store-refusal-overwrite-commit",
        script_refusal(
            Arc::clone(&armed),
            CREATE_LAYER_COMMIT_SCRIPT.sha1().to_string(),
            None,
            OOM.reply,
        ),
    )
    .await;
    let store = fx.store(UpdateLayerBackend::Lua);
    fx.create(&store, "store-refusal-layer")
        .await
        .expect("the first create must succeed");

    armed.store(true, Ordering::SeqCst);
    let result = fx.create(&store, "store-refusal-layer").await;
    armed.store(false, Ordering::SeqCst);

    assert_eq!(
        fx.proxy.hits(),
        2,
        "the overwrite arm's own retry, unchanged by b.yvh: the refused first try, the mapping \
         check (still the old, superseded object), exactly one more try — and no third"
    );
    match &result {
        Err(ServiceError::Unavailable(message)) => assert_eq!(
            message,
            &format!(
                "EVALSHA create_layer_commit: the backing store refused a write this request \
                 needed because {}; retry",
                OOM.condition
            ),
            "the overwrite arm hands the commit's own refusal to the mapper, which names the \
             script and the condition and claims nothing about what was applied"
        ),
        other => panic!(
            "an overwrite refused after the old object was marked superseded answers the \
             mapper's plain UNAVAILABLE — never the hinted nothing-applied answer (the mark \
             landed), and since b.xd1 no longer INTERNAL; got {other:?}"
        ),
    }
    assert!(
        refused_cells(&fx.metrics).values().all(|v| *v == 0.0),
        "the overwrite-after-mark refusal is not counted as a nothing-applied store refusal"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// b.xd1 — a discarded transaction is classified from its cause
// ═════════════════════════════════════════════════════════════════════════════
//
// A discarded transaction ran nothing, so its answer is the mapper's verdict on
// its cause — the queued command's own refusal, or the `EXEC`'s `EXECABORT …
// because of: <refusal>` — never on the bare "previous errors" `EXEC` reply,
// which names no cause. A store write refusal there is a plain `UNAVAILABLE`
// (no in-process retry, not counted: the mapper sites are not audited for
// nothing-applied); the store's not-serving `-LOADING` / `-BUSY` is transient,
// so the retry loop runs the call again on a fresh `WATCH`.

/// `EXEC`'s reply when the `EXEC` itself is refused behind a long script:
/// `EXEC` does not carry `allow-busy`.
const EXECABORT_BUSY: &str = "EXECABORT Transaction discarded because of: BUSY Valkey is busy \
                              running a script. You can only call SCRIPT KILL or SHUTDOWN NOSAVE.";

/// A base path no fixture session uses, so its names index holds only what the
/// `CreateSession` under test wrote.
const DISCARD_BASE_PATH: &str = "/store-refusal-discarded-create";

/// A `CreateSession` whose commit transaction is discarded once, with
/// `queued` refusing its first queued command: the result, the `MULTI`s sent,
/// and the names the base path's names index holds afterwards.
async fn create_session_discarded_once(
    name: &str,
    queued: &'static str,
) -> (Result<String, ServiceError>, usize, Vec<String>, Fx) {
    // CreateSession sends the witness-baseline probe with its `WATCH`, as the
    // WATCH path does, so its commit is the second `MULTI` of an attempt.
    let (txn, rule) = DiscardedTxn::rule(
        WATCH_PROBE_TRANSACTIONS,
        Some(queued),
        EXECABORT_PREVIOUS_ERRORS,
    );
    let fx = Fx::new(name, rule).await;
    let sessions = SessionStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone());

    txn.arm();
    let result = sessions.create(name, DISCARD_BASE_PATH, CREATED_BY).await;
    txn.disarm();

    let names: Vec<String> = fx
        .tv
        .client()
        .pool
        .smembers(fx.tv.key_schema().path_names_index(DISCARD_BASE_PATH))
        .await
        .expect("SMEMBERS path names index must succeed");
    (result, txn.multis(), names, fx)
}

/// A queued `-OOM` cause: plain `UNAVAILABLE` naming the commit and the
/// condition, after exactly one attempt, nothing created. (Nothing to count:
/// `SessionStore` has no store-refusal counter — it never calls
/// `refuse_store_write` — so no counter is asserted here.) Pre-b.xd1 the arm
/// classified the bare `EXEC` reply: `INTERNAL`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_discarded_for_a_queued_out_of_memory_answers_plain_unavailable_once() {
    let (result, multis, names, _fx) =
        create_session_discarded_once("store-refusal-create-session-oom", OOM.reply).await;

    match &result {
        Err(ServiceError::Unavailable(message)) => assert_eq!(
            message,
            &format!(
                "EXEC create_session: the backing store refused a write this request needed \
                 because {}; retry",
                OOM.condition
            )
        ),
        other => panic!(
            "a CreateSession whose commit a full store discarded must answer plain UNAVAILABLE, \
             classified from the queued refusal; got {other:?}"
        ),
    }
    assert_eq!(
        multis,
        WATCH_PROBE_TRANSACTIONS + 1,
        "one attempt — its baseline probe and its discarded commit: a store write refusal is \
         not retried in-process"
    );
    assert!(names.is_empty(), "nothing created; got {names:?}");
}

/// A queued `-LOADING` cause (the restart reply): transient, so the call runs
/// again under a fresh session id and the session is created once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_discarded_for_a_queued_loading_is_retried_and_created_once() {
    let (result, multis, names, _fx) = create_session_discarded_once(
        "store-refusal-create-session-loading",
        STORE_NOT_SERVING_SHAPES[0],
    )
    .await;

    let session_id = result.unwrap_or_else(|e| {
        panic!(
            "a commit discarded because the store was loading ran nothing and must be retried; \
             pre-b.xd1 it answered INTERNAL; got {e:?}"
        )
    });
    assert_eq!(
        multis,
        2 * (WATCH_PROBE_TRANSACTIONS + 1),
        "two attempts: the discarded one and the one that committed"
    );
    assert_eq!(
        names.len(),
        1,
        "exactly one session holds the name — the discarded attempt left nothing; got {names:?}"
    );
    assert!(
        names[0].ends_with(&session_id),
        "the names index names the session the call returned; got {names:?}"
    );
}

/// A WATCH-path commit discarded once for a not-serving cause: retried on a
/// fresh `WATCH`, committed once, one retry counted, no store refusal counted
/// and no torn-commit handling.
async fn watch_commit_discarded_then_retried(
    queued: Option<&'static str>,
    exec_reply: &'static str,
) {
    let (txn, rule) = DiscardedTxn::rule(WATCH_PROBE_TRANSACTIONS, queued, exec_reply);
    let fx = Fx::new("store-refusal-watch-busy", rule).await;
    let store = fx.store(UpdateLayerBackend::Txn);
    let oid = fx
        .create(&store, "store-refusal-layer")
        .await
        .expect("create_layer must succeed");
    let (_, xlen_before, _) = fx.commit_surface(oid).await;

    txn.arm();
    let result = store
        .update_layer_relayed(
            &fx.session_id,
            "store-refusal-layer",
            oid,
            Some(field_delta()),
            WRITER,
            1,
        )
        .await;
    txn.disarm();

    result.unwrap_or_else(|e| {
        panic!(
            "a commit discarded because the store was busy ({queued:?} / {exec_reply:?}) ran \
             nothing and must be retried; pre-b.xd1 it answered INTERNAL; got {e:?}"
        )
    });
    assert_eq!(
        txn.multis(),
        2 * (WATCH_PROBE_TRANSACTIONS + 1),
        "two attempts: the discarded one and the one that committed"
    );
    let (_, xlen_after, _) = fx.commit_surface(oid).await;
    assert_eq!(
        xlen_after,
        xlen_before + 1,
        "the update committed exactly once"
    );
    let text = render(&fx.metrics);
    assert_eq!(
        sample_value(
            &text,
            "live_edit_update_layer_retries_total",
            &[("path", "watch"), ("outcome", "committed")],
        ),
        Some(1.0),
        "one retry, counted as committed"
    );
    assert!(
        refused_cells(&fx.metrics).values().all(|v| *v == 0.0),
        "a not-serving store is not a store write refusal"
    );
    assert_eq!(
        sample_sum(
            &text,
            "live_edit_ext_index_recount_total",
            &[("site", "torn_commit")],
        ),
        Some(0.0),
        "a discarded transaction ran nothing: no torn-commit handling"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_commit_whose_exec_is_refused_busy_is_retried_and_commits_once() {
    watch_commit_discarded_then_retried(None, EXECABORT_BUSY).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_commit_whose_queued_write_is_refused_busy_is_retried_and_commits_once() {
    watch_commit_discarded_then_retried(
        Some(STORE_NOT_SERVING_SHAPES[2]),
        EXECABORT_PREVIOUS_ERRORS,
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// b.xd1 — a store write refusal at a mapper site, one per RPC family
// ═════════════════════════════════════════════════════════════════════════════
//
// Outside the counted upload sites, a write the store refuses reaches the
// client through the mappers: plain `UNAVAILABLE` naming the command and the
// condition, never retried in-process (the condition can persist), not
// counted. Pre-b.xd1 every one of these answered `INTERNAL`.

const MISCONF: &Refusal = &REFUSALS[1];
const NOREPLICAS: &Refusal = &REFUSALS[3];

/// The plain answer the mappers give a refused `op`.
fn mapped_refusal_message(op: &str, refusal: &Refusal) -> String {
    format!(
        "{op}: the backing store refused a write this request needed because {}; retry",
        refusal.condition
    )
}

/// DeleteLayer's first write, the mapping `DEL`, refused `-MISCONF`:
/// `UNAVAILABLE`, the mapping intact (the `DEL` applied nothing and the
/// cascade stopped there), one `DEL` sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_whose_mapping_del_is_refused_misconf_answers_unavailable_once() {
    let armed = Arc::new(AtomicBool::new(false));
    let mapping_key: Arc<std::sync::Mutex<String>> = Arc::new(std::sync::Mutex::new(String::new()));
    let rule: ProxyRule = {
        let (armed, mapping_key) = (Arc::clone(&armed), Arc::clone(&mapping_key));
        Arc::new(move |args: &[Vec<u8>]| {
            let key = mapping_key.lock().unwrap().clone();
            (armed.load(Ordering::SeqCst) && command_name(args) == "DEL" && has_arg(args, &key))
                .then(|| ProxyAction::Eval(error_reply_script(MISCONF.reply)))
        })
    };
    let fx = Fx::new("store-refusal-delete-layer", rule).await;
    *mapping_key.lock().unwrap() = fx
        .tv
        .key_schema()
        .layer_mapping(&fx.session_id, "store-refusal-layer");
    let store = fx.store(UpdateLayerBackend::Lua);
    let oid = fx
        .create(&store, "store-refusal-layer")
        .await
        .expect("create_layer must succeed");

    armed.store(true, Ordering::SeqCst);
    let result = store
        .delete_layer(
            &fx.session_id,
            "store-refusal-layer",
            Arc::new(MockExternalValueStore::new()),
        )
        .await;
    armed.store(false, Ordering::SeqCst);

    match &result {
        Err(ServiceError::Unavailable(message)) => {
            assert_eq!(
                message,
                &mapped_refusal_message("DEL layer mapping", MISCONF)
            );
        }
        other => panic!(
            "a DeleteLayer whose mapping DEL the store refused must answer plain UNAVAILABLE; \
             pre-b.xd1 it answered INTERNAL; got {other:?}"
        ),
    }
    assert_eq!(
        tonic::Status::from(result.unwrap_err()).code(),
        tonic::Code::Unavailable
    );
    assert_eq!(
        fx.proxy.hits(),
        1,
        "one DEL: a store write refusal is not retried by the cascade's retry loop"
    );
    assert_eq!(
        fx.mapping("store-refusal-layer").await,
        Some(oid.to_string()),
        "the mapping still names the layer's object"
    );
    assert!(
        refused_cells(&fx.metrics).values().all(|v| *v == 0.0),
        "a mapper site's refusal is not counted"
    );
}

/// JoinSession's first presence write, the `SADD` of the session into the
/// fleet-wide index, refused `-NOREPLICAS`: the join ends `UNAVAILABLE`
/// naming the write and the condition, after one `SADD`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_whose_presence_index_write_is_refused_noreplicas_ends_unavailable() {
    use crate::grpc::helpers::{
        build_service, drain_tracker, encode_join_frame, join_header, make_join_request,
        open_join_streaming, read_to_end, AllowAllPermissions,
    };

    let armed = Arc::new(AtomicBool::new(false));
    let index_key: Arc<std::sync::Mutex<String>> = Arc::new(std::sync::Mutex::new(String::new()));
    let rule: ProxyRule = {
        let (armed, index_key) = (Arc::clone(&armed), Arc::clone(&index_key));
        Arc::new(move |args: &[Vec<u8>]| {
            let key = index_key.lock().unwrap().clone();
            (armed.load(Ordering::SeqCst) && command_name(args) == "SADD" && has_arg(args, &key))
                .then(|| ProxyAction::Eval(error_reply_script(NOREPLICAS.reply)))
        })
    };
    let fx = Fx::new("store-refusal-join", rule).await;
    *index_key.lock().unwrap() = fx.tv.key_schema().presence_active_sessions();
    let svc = build_service(&fx.tv, Arc::new(AllowAllPermissions));

    let (stream, tx) = open_join_streaming();
    tx.send(encode_join_frame(&join_header(
        &fx.session_id,
        "store-refusal-app",
    )))
    .expect("the client channel must accept the header frame");
    armed.store(true, Ordering::SeqCst);
    let status =
        match live_edit::grpc::presence::handle_join_session(&svc, make_join_request(stream)).await
        {
            Err(status) => status,
            Ok(response) => {
                let mut responses = response.into_inner();
                let (delivered, status) = read_to_end(&mut responses, "a refused join").await;
                status.unwrap_or_else(|| {
                    panic!(
                    "a join whose presence write was refused must end in error; got {delivered:?}"
                )
                })
            }
        };
    armed.store(false, Ordering::SeqCst);
    drop(tx);
    drain_tracker(&svc).await;

    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "a refused presence write is retriable for the client; pre-b.xd1 it ended INTERNAL: \
         {status:?}"
    );
    assert_eq!(
        status.message(),
        mapped_refusal_message("SADD presence active_sessions", NOREPLICAS)
    );
    assert_eq!(fx.proxy.hits(), 1, "one SADD: the join does not retry it");
}

// ═════════════════════════════════════════════════════════════════════════════
// b.pa1 — an `-OOM` requests the GC's out-of-memory collection
// ═════════════════════════════════════════════════════════════════════════════
//
// A store write refused `-OOM` is noted on the client's store out-of-memory
// signal, which — once startup armed it, at most once a second — writes the
// collection request `le:ext:{gc}:oom:request` with this replica's label, from
// a spawned task, through `gc_oom_signal.lua`. Here the fixture's client is
// armed by hand, the refusal is staged as above, and the request is read off
// the store (`oom_request_support::wait_for_oom_request`). The leader's side
// is in `oom_collection_test.rs`; the signal's turn and its wake-up in
// `src/storage/store_oom.rs`.

/// ST2: an upload's external-value pre-store refused `-OOM` — answered and
/// counted as before — requests a collection: the request key holds the
/// armed label, with at most the request's 10 s lifetime. A refusal after
/// `STORE_OOM_REQUEST_MIN_INTERVAL` has passed requests again. That a second
/// note inside the interval sends no request is pinned in
/// `src/storage/store_oom.rs`
/// (`an_armed_note_sends_its_request_and_wakes_the_gc_loop_only_after_the_send`:
/// two back-to-back armed notes, one `EVALSHA` at a stand-in store), and that
/// it wakes nothing under paused time there too; not here, where a loaded
/// runner could stretch two refusals' round trips across the interval.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_upload_out_of_memory_refusal_requests_an_oom_collection_and_again_after_the_interval() {
    let armed = Arc::new(AtomicBool::new(false));
    let fx = Fx::new(
        "store-refusal-oom-request",
        script_refusal(
            Arc::clone(&armed),
            EXT_STORE_VALUE_SCRIPT.sha1().to_string(),
            None,
            OOM.reply,
        ),
    )
    .await;
    fx.tv
        .client()
        .store_oom_signal()
        .arm(fx.tv.key_schema(), "replica-st2");
    let ext = ValkeyExternalValueStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone())
        .with_metrics(Arc::clone(&fx.metrics));
    let blob: &[u8] = b"store-refusal oom request";
    let hash = ext_content_hash(blob);
    let request_key = fx.tv.key_schema().ext_oom_request();

    armed.store(true, Ordering::SeqCst);
    let first = ext.store_values(&[(&hash, blob)]).await;
    assert_refused(&first, &OOM);
    let (label, pttl) = wait_for_oom_request(&fx.tv).await;
    assert_eq!(
        label, "replica-st2",
        "the request names the refusing replica"
    );
    let lifetime = i64::try_from(OOM_COLLECTION_REQUEST_TTL.as_millis()).unwrap();
    assert!(
        pttl > 0 && pttl <= lifetime,
        "the request lives at most {lifetime} ms; PTTL {pttl}"
    );

    // Waited out from after the first request landed, so the next refusal is
    // past the interval whatever the runner's load.
    tokio::time::sleep(STORE_OOM_REQUEST_MIN_INTERVAL + std::time::Duration::from_millis(100))
        .await;
    let _: i64 = fx
        .tv
        .client()
        .pool
        .del(&request_key)
        .await
        .expect("DEL oom:request");
    let second = ext.store_values(&[(&hash, blob)]).await;
    armed.store(false, Ordering::SeqCst);
    assert_refused(&second, &OOM);
    assert_eq!(
        wait_for_oom_request(&fx.tv).await.0,
        "replica-st2",
        "a refusal after the interval requests the next collection"
    );
    assert_eq!(
        refused_cells(&fx.metrics)[&("ext_store".to_owned(), "out_of_memory".to_owned())],
        2.0,
        "every refusal is still counted"
    );
}

/// ST3: a store write refused `-OOM` at a mapper site — `CreateSession`'s
/// commit discarded for a queued `-OOM`, answered plain `UNAVAILABLE` and
/// counted nowhere — requests a collection too: every site that receives a
/// fred error notes it on the client's signal (`FredErrOrigin`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_mapper_site_out_of_memory_requests_an_oom_collection() {
    let (txn, rule) = DiscardedTxn::rule(
        WATCH_PROBE_TRANSACTIONS,
        Some(OOM.reply),
        EXECABORT_PREVIOUS_ERRORS,
    );
    let fx = Fx::new("store-refusal-mapper-oom-request", rule).await;
    fx.tv
        .client()
        .store_oom_signal()
        .arm(fx.tv.key_schema(), "replica-st3");
    let sessions = SessionStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone());

    txn.arm();
    let result = sessions
        .create(
            "store-refusal-mapper-oom-request",
            DISCARD_BASE_PATH,
            CREATED_BY,
        )
        .await;
    txn.disarm();

    assert!(
        matches!(result, Err(ServiceError::Unavailable(_))),
        "precondition: the mapper answers the refusal plain UNAVAILABLE; got {result:?}"
    );
    assert_eq!(wait_for_oom_request(&fx.tv).await.0, "replica-st3");
    assert!(
        refused_cells(&fx.metrics).values().all(|v| *v == 0.0),
        "a mapper site's refusal is still not counted"
    );
}

/// An overwrite's first commit run refused `-OOM` by the store, its second
/// run landing: the call answers OK, so no mapper ever sees the refusal, and
/// the refusal still requests a collection. Every commit run goes through
/// one dispatch, which notes a failed run where it receives it
/// (`LayerStore::dispatch_create_layer_commit`). The rule refuses only the
/// first commit `EVALSHA`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_commit_run_refused_out_of_memory_then_landed_requests_an_oom_collection(
) {
    let refuse_next = Arc::new(AtomicBool::new(false));
    let rule: ProxyRule = {
        let refuse_next = Arc::clone(&refuse_next);
        let sha = CREATE_LAYER_COMMIT_SCRIPT.sha1().to_string();
        Arc::new(move |args: &[Vec<u8>]| {
            let name = command_name(args);
            let is_commit = (name == "EVALSHA" || name == "EVAL")
                && args
                    .get(1)
                    .is_some_and(|a| String::from_utf8_lossy(a).eq_ignore_ascii_case(&sha));
            (is_commit && refuse_next.swap(false, Ordering::SeqCst))
                .then(|| ProxyAction::Eval(error_reply_script(OOM.reply)))
        })
    };
    let fx = Fx::new("store-refusal-overwrite-commit-oom", rule).await;
    fx.tv
        .client()
        .store_oom_signal()
        .arm(fx.tv.key_schema(), "replica-overwrite-commit");
    let store = fx.store(UpdateLayerBackend::Lua);
    fx.create(&store, "store-refusal-layer")
        .await
        .expect("the first create must succeed");

    refuse_next.store(true, Ordering::SeqCst);
    let replaced = fx.create(&store, "store-refusal-layer").await;

    assert_eq!(
        fx.proxy.hits(),
        1,
        "precondition: the first run was refused"
    );
    let new_oid = replaced.expect("the overwrite's second run lands: the call answers OK");
    assert_eq!(
        fx.mapping("store-refusal-layer").await,
        Some(new_oid.to_string()),
        "precondition: the layer names the new object"
    );
    assert_eq!(
        wait_for_oom_request(&fx.tv).await.0,
        "replica-overwrite-commit",
        "the refused run requests a collection although the call answered OK"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Through the UpdateLayer handler
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod through_the_handler {
    //! The client's view: the `UpdateLayer` RPC answers a store refusal at the
    //! commit, and at the external-value pre-store before it, with gRPC
    //! `UNAVAILABLE` carrying the nothing-applied message and the
    //! `STORE_WRITE_REFUSED` retry hint — not the redacted `INTERNAL "internal
    //! error"`. (`build_service_with_layer_store` needs
    //! `test-fault-injection`.)

    use super::*;
    use crate::grpc::helpers::{
        build_service_with_layer_store, make_request, make_update_data, make_update_header,
        streaming_update,
    };
    use live_edit::grpc::layer::handle_update_layer;
    use live_edit::grpc::service::LiveEditService;

    use crate::common::refusal_detail::assert_refusal_detail;

    async fn update_status(
        fx: &Fx,
        svc: &LiveEditService,
        oid: u64,
        ompf: Vec<u8>,
    ) -> tonic::Status {
        let stream = streaming_update(vec![
            make_update_header(&fx.session_id, "store-refusal-layer", oid),
            make_update_data(ompf),
        ]);
        handle_update_layer(svc, make_request(stream))
            .await
            .expect_err("the refused UpdateLayer must fail")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_rpc_refused_at_its_commit_answers_unavailable_with_the_message() {
        let armed = Arc::new(AtomicBool::new(false));
        let fx = Fx::new(
            "store-refusal-rpc-commit",
            script_refusal(
                Arc::clone(&armed),
                UPDATE_LAYER_FAST_SCRIPT.sha1().to_string(),
                Some(UpdateScriptMode::Fast.as_str()),
                OOM.reply,
            ),
        )
        .await;
        let store = Arc::new(fx.store(UpdateLayerBackend::Lua));
        let oid = fx
            .create(&store, "store-refusal-layer")
            .await
            .expect("create_layer must succeed");
        let svc = build_service_with_layer_store(&fx.tv, Arc::clone(&store));

        armed.store(true, Ordering::SeqCst);
        let status = update_status(
            &fx,
            &svc,
            oid,
            crate::common::owned_delta::delta_to_ompf_bytes(&field_delta(), &[]),
        )
        .await;
        armed.store(false, Ordering::SeqCst);

        assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
        assert_eq!(status.message(), refusal_message(&OOM));
        assert_refusal_detail(
            &status,
            refusal_reason::STORE_WRITE_REFUSED,
            &[(refusal_metadata::CONDITION, OOM.reason)],
            STORE_WRITE_REFUSED_RETRY_DELAY,
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_rpc_refused_at_its_pre_store_answers_unavailable_with_the_message() {
        let armed = Arc::new(AtomicBool::new(false));
        let fx = Fx::new(
            "store-refusal-rpc-ext",
            script_refusal(
                Arc::clone(&armed),
                EXT_STORE_VALUE_SCRIPT.sha1().to_string(),
                None,
                OOM.reply,
            ),
        )
        .await;
        let store = Arc::new(fx.store(UpdateLayerBackend::Lua));
        let oid = fx
            .create(&store, "store-refusal-layer")
            .await
            .expect("create_layer must succeed");
        let svc = LiveEditService {
            ext_value_store: Arc::new(
                ValkeyExternalValueStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone())
                    .with_metrics(Arc::clone(&fx.metrics)),
            ),
            ..build_service_with_layer_store(&fx.tv, Arc::clone(&store))
        };
        let blob: &[u8] = b"store-refusal rpc external value";
        let delta = OwnedDelta {
            set_fields: vec![SetField {
                section_id: 1,
                key_name: "store-refusal-external".to_owned(),
                value_or_ext_hash: ext_content_hash(blob).to_vec(),
                ext_value_size: blob.len() as u64,
                ext_value_index: 1,
                set_order: 0,
            }],
            ..Default::default()
        };
        let before = fx.commit_surface(oid).await;

        armed.store(true, Ordering::SeqCst);
        let status = update_status(
            &fx,
            &svc,
            oid,
            crate::common::owned_delta::delta_to_ompf_bytes(&delta, &[blob]),
        )
        .await;
        armed.store(false, Ordering::SeqCst);

        assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
        assert_eq!(status.message(), refusal_message(&OOM));
        assert_refusal_detail(
            &status,
            refusal_reason::STORE_WRITE_REFUSED,
            &[(refusal_metadata::CONDITION, OOM.reason)],
            STORE_WRITE_REFUSED_RETRY_DELAY,
        );
        assert_eq!(
            fx.commit_surface(oid).await,
            before,
            "a pre-store refusal comes before the commit: nothing applied"
        );
        assert_counted_once(&fx.metrics, "ext_store", "out_of_memory");
    }

    /// An `UpdateLayer` whose object was wiped: its first `update_layer`
    /// call finds it gone, the checkpoint recovery restores it, and the
    /// re-issued call's commit is refused out of memory on its first attempt.
    /// When the first call took two attempts — attempt 1's `EVALSHA` timed out,
    /// a lost reply whose commit may still land — the RPC answers plain
    /// `UNAVAILABLE`, outcome unknown, with no retry hint (SDD §18.1); when it
    /// took one, the refusal keeps its `STORE_WRITE_REFUSED` hint.
    async fn reissue_refused(first_call_timed_out: bool) {
        use live_edit::checkpoint::RecoveryHook;

        // The update script's second EVALSHA — the re-issue's first — is
        // refused; the first, the first call's last attempt, reaches the store
        // and finds the object gone.
        let sent = Arc::new(AtomicUsize::new(0));
        let rule: ProxyRule = {
            let sent = Arc::clone(&sent);
            let sha = UPDATE_LAYER_FAST_SCRIPT.sha1().to_string();
            Arc::new(move |args: &[Vec<u8>]| {
                let is_update = command_name(args) == "EVALSHA"
                    && args
                        .get(1)
                        .is_some_and(|a| String::from_utf8_lossy(a).eq_ignore_ascii_case(&sha));
                (is_update && sent.fetch_add(1, Ordering::SeqCst) == 1)
                    .then(|| ProxyAction::Eval(error_reply_script(OOM.reply)))
            })
        };
        let fx = Fx::new("store-refusal-rpc-reissue", rule).await;
        let timed_out: Vec<fred::error::Error> = if first_call_timed_out {
            vec![fred::error::Error::new(
                fred::error::ErrorKind::Timeout,
                "synthetic lost reply",
            )]
        } else {
            Vec::new()
        };
        let store = Arc::new(
            fx.store(UpdateLayerBackend::Lua)
                .with_update_layer_retry_max_attempts(2)
                .with_lua_fred_error_injector_for_test(Arc::new(std::sync::Mutex::new(
                    timed_out.into_iter().collect(),
                ))),
        );
        let oid = fx
            .create(&store, "store-refusal-layer")
            .await
            .expect("create_layer must succeed");
        let mut svc = build_service_with_layer_store(&fx.tv, Arc::clone(&store));
        let checkpoints = InMemoryCheckpointStore::new();
        crate::common::seed_layer_checkpoint(
            &checkpoints,
            oid,
            &fx.session_id,
            "store-refusal-layer",
            3,
        )
        .await;
        let mut cfg = crate::common::test_config();
        cfg.checkpoint_enabled = true;
        svc.recovery = Arc::new(RecoveryHook::new(
            Arc::new(checkpoints) as Arc<dyn CheckpointStore>,
            Arc::clone(&svc.layer_store),
            Arc::new(cfg),
            Arc::clone(&svc.metrics),
        ));
        crate::common::wipe_object_keys(&fx.tv, oid).await;
        let sent_before = sent.load(Ordering::SeqCst);

        let status = update_status(
            &fx,
            &svc,
            oid,
            crate::common::owned_delta::delta_to_ompf_bytes(&field_delta(), &[]),
        )
        .await;

        assert_eq!(
            sent.load(Ordering::SeqCst) - sent_before,
            2,
            "the first call's last attempt and the re-issue's refused one"
        );
        assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
        if first_call_timed_out {
            assert_eq!(
                status.message(),
                format!(
                    "UpdateLayer outcome unknown: an earlier attempt of this call may have \
                     committed before attempt 3 was refused (the backing store refused a write \
                     this request needed because {})",
                    OOM.condition
                )
            );
            crate::common::refusal_detail::assert_no_refusal_detail(&status);
        } else {
            assert_eq!(status.message(), refusal_message(&OOM));
            assert_refusal_detail(
                &status,
                refusal_reason::STORE_WRITE_REFUSED,
                &[(refusal_metadata::CONDITION, OOM.reason)],
                STORE_WRITE_REFUSED_RETRY_DELAY,
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_reissue_refused_after_a_one_attempt_first_call_keeps_the_hint() {
        reissue_refused(false).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_reissue_refused_after_a_timed_out_first_call_attempt_is_plain() {
        reissue_refused(true).await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// A real -OOM (disruptive)
// ═════════════════════════════════════════════════════════════════════════════

/// A real `-OOM`, from a store whose `maxmemory` is below what it holds, on
/// the three upload write sites the refusal most often meets: the fast-path
/// commit script (its first write, the `XADD`), the WATCH path's transaction
/// (its queued writes refused at queue time, the `EXEC` answered
/// `EXECABORT`), and the external-value pre-store. Each answers `UNAVAILABLE`
/// with the out-of-memory message, applies nothing and ticks its cell. This is
/// the check that the staged replies above match what Valkey really sends.
///
/// Disruptive — it changes the server's `maxmemory` — so `#[ignore]`d, and it
/// runs only against its own Valkey, named by
/// `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL` (CI's `test-cluster-disruptive:` job
/// sets it to that job's `valkey-disruptive` service), never the shared one.
/// It skips when that variable is unset, except in CI (`GITLAB_CI` set), where
/// an unset variable is broken job wiring and fails it. It restores
/// `maxmemory` when it ends, on a failure too. It needs the server's
/// `maxmemory-policy` to be the required `noeviction` (the image default),
/// and checks it. The store and the `maxmemory` guard are
/// `super::disruptive_valkey`'s, shared with `oom_collection_test.rs`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_real_out_of_memory_is_answered_unavailable_at_each_site() {
    let Some(url) = disruptive_valkey_url() else {
        return;
    };
    let tv = TestValkey::setup_with_url(url, POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("store-refusal-real-oom"));
    let admin = tv.client().pool.next().clone();
    assert_noeviction(&admin).await;
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(
            "store-refusal-real-oom",
            "/store-refusal-real-oom",
            CREATED_BY,
        )
        .await
        .expect("session creation must succeed");
    let store = |backend| {
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend)
            .with_metrics(Arc::clone(&metrics))
    };
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..Default::default()
    };
    let mut oids = Vec::new();
    for (backend, layer) in [
        (UpdateLayerBackend::Lua, "real-oom-fast"),
        (UpdateLayerBackend::Txn, "real-oom-watch"),
    ] {
        let oid = store(backend)
            .create_layer(
                &session_id,
                layer,
                CREATED_BY,
                &state,
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("create_layer must succeed before the store is full");
        oids.push((backend, layer, oid));
    }
    let ext = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_metrics(Arc::clone(&metrics));
    let header_seq = |oid: u64| {
        let pool = tv.client().pool.clone();
        let key = tv.key_schema().obj_header(&oid.to_string());
        async move {
            let seq: String = pool.hget(key, "seq").await.expect("HGET seq");
            seq
        }
    };

    let full = RestoreMaxmemory::set_full(&admin).await;
    let mut answers = Vec::new();
    for (backend, layer, oid) in &oids {
        let seq_before = header_seq(*oid).await;
        let result = store(*backend)
            .update_layer_relayed(&session_id, layer, *oid, Some(field_delta()), WRITER, 1)
            .await;
        answers.push((format!("{backend:?} commit"), result.map(|_| ()).err()));
        assert_eq!(
            header_seq(*oid).await,
            seq_before,
            "{backend:?}: nothing applied"
        );
    }
    let blob: &[u8] = b"store-refusal real oom external value";
    let hash = ext_content_hash(blob);
    answers.push((
        "external-value pre-store".to_owned(),
        ext.store_values(&[(&hash, blob)]).await.err(),
    ));
    full.restore().await;

    for (site, answer) in &answers {
        let err = answer
            .as_ref()
            .unwrap_or_else(|| panic!("{site}: a real -OOM must answer UNAVAILABLE; got Ok"));
        let message = assert_store_refusal_detail(err, &OOM);
        assert_eq!(message, refusal_message(&OOM), "{site}");
    }
    let cells = refused_cells(&metrics);
    assert_eq!(
        (
            cells[&("update_commit".to_owned(), "out_of_memory".to_owned())],
            cells[&("ext_store".to_owned(), "out_of_memory".to_owned())],
        ),
        (2.0, 1.0),
        "the two commits and the pre-store each count once"
    );
}

/// b.ua9 test 24: on a store whose `maxmemory` is below what it holds, every
/// retirement route still wakes the retired object's stream and deletes it —
/// deleting keys is what frees a full store, and the shared wake script runs
/// there because its first command, a `PEXPIRE`, is a write the store accepts
/// above `maxmemory`, after which the store no longer refuses the script's
/// `XADD`:
///
/// - (a) an overwrite whose every wake-up attempt was injected to fail kept
///   its old object (unmapped, fenced); one orphan-sweep pass at grace 0 on
///   the full store re-runs the overwrite tail: `deleted` on the old stream,
///   the old header gone;
/// - (b) a `DeleteLayer` on the full store completes: mapping and header
///   gone, `deleted` on the stream;
/// - (c) a never-committed create (a new layer whose two commit runs were
///   injected to fail before their script ran) left an unmapped, unclaimed
///   object; the same pass vetoes it — the veto script's no-op `PEXPIRE`
///   opener lets its `SET` of the `vetoed` claim run above `maxmemory` — and
///   retires it: claim `vetoed`, header gone.
///
/// The sweep's canary is planted, and its leader lock taken by one pass at
/// the default grace (which leaves the replaced object to its young
/// successor, and the never-committed object to its grace), while the store
/// still takes writes: a full store refuses the canary's `SET` and the lock's
/// first `SET`, but not the lock's refresh.
///
/// Before b.ua9 (a) kept the old header — the sweep's retirement appended no
/// `deleted` and the overwrite tail's `XADD` was refused out of memory — and
/// (b) answered `UNAVAILABLE` after its mapping `DEL`, its `XADD` refused.
/// Before the veto's opener (A10), (c) kept the header: the veto's `SET`, its
/// first write, was refused out of memory and the sweep skipped the object.
///
/// Disruptive, like the test above: `#[ignore]`d, run only against
/// `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL`, skipping when that is unset outside
/// CI, restoring `maxmemory` when it ends. Selected by the
/// `test-cluster-disruptive:` job's `store_write_refusal_disruptive` filter,
/// and serialized with the other configuration-changing tests by the
/// `disruptive-valkey-config` group in `.config/nextest.toml`. Needs
/// `test-fault-injection` (the wake-up injector), which that job enables.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_full_store_still_wakes_and_retires_replaced_and_deleted_objects(
) {
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::time::Duration;

    use fred::interfaces::ConfigInterface;
    use live_edit::storage::layer::CreateCommitFault;
    use live_edit::storage::notifications::{DELETED_STREAM_PAYLOAD, NOTIFICATION_PAYLOAD_FIELD};

    use crate::storage::b_ua9_tests::{plant_sweep_canary, run_planted_sweep_pass};

    /// The overwrite cleanup's wake-up attempts: the inline one and three
    /// retries.
    const WAKE_ATTEMPTS: usize = 4;
    const REPLACED_LAYER: &str = "full-store-replaced";
    const DELETED_LAYER: &str = "full-store-deleted";
    const NEVER_COMMITTED_LAYER: &str = "full-store-never-committed";

    let Some(url) = disruptive_valkey_url() else {
        return;
    };
    let tv = TestValkey::setup_with_url(url, POOL_SIZE).await;
    let admin = tv.client().pool.next().clone();
    let policy: HashMap<String, String> = admin
        .config_get("maxmemory-policy")
        .await
        .expect("CONFIG GET maxmemory-policy must succeed");
    assert_eq!(
        policy.get("maxmemory-policy").map(String::as_str),
        Some("noeviction"),
        "fixture precondition: the disruptive store must run the required noeviction policy"
    );
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("full-store-retire", "/full-store-retire", CREATED_BY)
        .await
        .expect("session creation must succeed");
    let ext: Arc<dyn ExternalValueStore> = Arc::new(MockExternalValueStore::new());
    let plain = || LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..Default::default()
    };
    let replaced = plain()
        .create_layer(&session_id, REPLACED_LAYER, CREATED_BY, &state, ext.clone())
        .await
        .expect("create_layer must succeed before the store is full");
    let deleted = plain()
        .create_layer(&session_id, DELETED_LAYER, CREATED_BY, &state, ext.clone())
        .await
        .expect("create_layer must succeed before the store is full");
    let wake: Arc<Mutex<VecDeque<fred::error::Error>>> = Arc::new(Mutex::new(
        (0..WAKE_ATTEMPTS)
            .map(|_| fred::error::Error::new(fred::error::ErrorKind::IO, "full-store: wake lost"))
            .collect(),
    ));
    let overwriter =
        plain().with_create_layer_overwrite_xadd_fault_injector_for_test(Arc::clone(&wake));
    overwriter
        .create_layer(&session_id, REPLACED_LAYER, CREATED_BY, &state, ext.clone())
        .await
        .expect("the overwrite answers OK whatever its wake-up did");
    overwriter.wait_for_cleanup(Duration::from_secs(15)).await;
    assert!(
        wake.lock().unwrap().is_empty(),
        "precondition: every wake-up attempt failed"
    );
    // (c)'s orphan: a new layer whose two commit runs both fail before their
    // script runs. Nothing else allocates an object id meanwhile, so the
    // counter names the orphan.
    let commit_faults: Arc<Mutex<VecDeque<CreateCommitFault>>> = Arc::new(Mutex::new(
        (0..2)
            .map(|_| CreateCommitFault {
                error: fred::error::Error::new(
                    fred::error::ErrorKind::IO,
                    "full-store: commit lost",
                ),
                script_ran: false,
            })
            .collect(),
    ));
    let never_committed_answer = plain()
        .with_create_layer_commit_fault_injector_for_test(Arc::clone(&commit_faults))
        .create_layer(
            &session_id,
            NEVER_COMMITTED_LAYER,
            CREATED_BY,
            &state,
            ext.clone(),
        )
        .await;
    assert!(
        matches!(never_committed_answer, Err(ServiceError::Unavailable(_)))
            && commit_faults.lock().unwrap().is_empty(),
        "precondition: both commit runs of the new layer failed; got {never_committed_answer:?}"
    );
    let never_committed: u64 = tv
        .client()
        .pool
        .get::<String, _>(tv.key_schema().counter_object_id())
        .await
        .expect("GET of the object-id counter")
        .parse()
        .expect("the counter is a decimal object id");
    let never_committed_claim_key = tv
        .key_schema()
        .create_layer_claim(&session_id, never_committed);
    // One pass at the default grace takes the sweep's leader lock and leaves
    // the replaced object (its successor is younger than the grace) and the
    // never-committed object (younger than the grace itself).
    plant_sweep_canary(tv.client(), tv.key_schema()).await;
    run_planted_sweep_pass(tv.client(), tv.key_schema(), ext.clone(), Arc::new(plain())).await;
    plant_sweep_canary(tv.client(), tv.key_schema()).await;
    let header_exists = |oid: u64| {
        let pool = tv.client().pool.clone();
        let key = tv.key_schema().obj_header(&oid.to_string());
        async move {
            let exists: bool = pool.exists(key).await.expect("EXISTS obj:hdr");
            exists
        }
    };
    let deleted_entries = |oid: u64| {
        let pool = tv.client().pool.clone();
        let key = tv.key_schema().layer_stream(oid);
        async move {
            // Read as bytes: a commit entry's relay `ompf` field is binary.
            let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = pool
                .xrange_values(key, "-", "+", None)
                .await
                .expect("XRANGE of the object stream");
            entries
                .iter()
                .filter(|(_, fields)| {
                    fields.get(NOTIFICATION_PAYLOAD_FIELD).map(|p| &p[..])
                        == Some(DELETED_STREAM_PAYLOAD.as_bytes())
                })
                .count()
        }
    };
    assert!(
        header_exists(replaced).await,
        "precondition: the replaced object survived its failed wake-ups and the graced pass"
    );
    let never_committed_claim_before: Option<String> = tv
        .client()
        .pool
        .get(&never_committed_claim_key)
        .await
        .expect("GET of the commit claim");
    assert!(
        header_exists(never_committed).await && never_committed_claim_before.is_none(),
        "precondition: the never-committed object is materialized, unclaimed, and survived the \
         graced pass; claim {never_committed_claim_before:?}"
    );

    let full = RestoreMaxmemory::set_full(&admin).await;
    tv.client().set_orphan_sweep_grace_for_test(Duration::ZERO);
    run_planted_sweep_pass(tv.client(), tv.key_schema(), ext.clone(), Arc::new(plain())).await;
    let replaced_after = (
        deleted_entries(replaced).await,
        header_exists(replaced).await,
    );
    let never_committed_claim: Option<String> = tv
        .client()
        .pool
        .get(&never_committed_claim_key)
        .await
        .expect("GET of the commit claim");
    let never_committed_after = (
        never_committed_claim.as_deref(),
        header_exists(never_committed).await,
    );
    let delete_answer = plain()
        .delete_layer(&session_id, DELETED_LAYER, ext.clone())
        .await;
    let mapping_after: Option<String> = tv
        .client()
        .pool
        .get(tv.key_schema().layer_mapping(&session_id, DELETED_LAYER))
        .await
        .expect("GET of the layer mapping");
    let deleted_after = (deleted_entries(deleted).await, header_exists(deleted).await);
    full.restore().await;

    assert_eq!(
        replaced_after,
        (1, false),
        "(a): one sweep pass on the full store wakes the replaced object's stream and deletes \
         its header"
    );
    assert_eq!(
        never_committed_after,
        (Some("vetoed"), false),
        "(c): the same pass vetoes the never-committed object on the full store and retires it"
    );
    assert!(
        delete_answer.is_ok(),
        "(b): DeleteLayer on a full store completes; got {delete_answer:?}"
    );
    assert_eq!(mapping_after, None, "(b): the mapping is gone");
    assert_eq!(
        deleted_after,
        (1, false),
        "(b): DeleteLayer wakes the stream and deletes the header on a full store"
    );
}
