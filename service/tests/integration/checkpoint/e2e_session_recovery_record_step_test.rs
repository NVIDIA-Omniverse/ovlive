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

//! On-access session recovery's record step (step 7 of
//! `SessionRecoveryHook::try_recover_session`), as the hook meets a
//! `DeleteSession` that lands after its last marker read.
//!
//! The hook reads the session's deletion marker before its checkpoint fetch
//! (step 4a) and after it (step 6a). A `DeleteSession` on another replica can
//! still mark the session after step 6a, while Step A's `EXEC` is in flight.
//! The record step (`scripts/session_record_restore.lua`) reads the marker
//! again, atomically with its writes, so the hook answers `NoCheckpoint`,
//! counts it `not_recoverable`, and writes no record, no layer mapping and no
//! layers-index member.
//!
//! The marker is staged through a [`ScriptReplyProxy`] that holds the replies
//! on the connection of the hook's first `EXEC` (Step A's): once a reply is
//! held, the hook has passed step 6a and sent Step A, so the test writes the
//! marker, on a client that bypasses the proxy, and then releases the
//! replies. The proxy also counts every `EXEC` the hook sends: the record
//! step is one script, not a fred `Transaction`, whose replay after a lost
//! reply would re-apply its writes.
//!
//! The record step's other answers — present, and a run cut part-way — are
//! the apply-level inline tests in `src/checkpoint/session.rs`; the
//! two-replica race through the hook is the inline
//! `concurrent_recoveries_on_two_replicas_keep_a_layer_deleted_between_their_applies`
//! in `src/checkpoint/session_recovery.rs`.
//!
//! Waits: one [`SETTLE`] (20 s) for the recovery and the proxy's own budget
//! for the held reply, inside nextest's 120 s hang-killer.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::session_payload::SESSION_CHECKPOINT_SCHEMA_VERSION;
use live_edit::checkpoint::{
    CheckpointMeta, CheckpointStore, SessionCheckpoint, SessionRecoveryHook, SessionRecoveryOutcome,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::test_support::TestValkey;

use crate::common::RECOVERY_SERVICE_LABEL;
use crate::storage::snapshot_read_support::{
    client_through, test_valkey_addr, ProxyAction, ProxyRule, ReplyGate, ScriptReplyProxy,
};

/// Upper bound on the recovery under test. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(20);

/// The one layer the backup maps.
const LAYER: &str = "record-step.usda";
const OBJECT_ID: u64 = 4242;

fn recovery_attempts(metrics: &SharedMetrics, result: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", "session"),
            ("service", RECOVERY_SERVICE_LABEL),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// A backup of `session_id` mapping [`LAYER`] to [`OBJECT_ID`].
fn backup_of(session_id: &str) -> SessionCheckpoint {
    let mut backup = SessionCheckpoint {
        schema_version: SESSION_CHECKPOINT_SCHEMA_VERSION,
        session_id: session_id.to_owned(),
        name: "record-step".to_owned(),
        base_path: format!("/record-step/{session_id}"),
        created_by: "record-step-user".to_owned(),
        created_at: 1_700_000_000_000_000,
        layer_mappings: BTreeMap::from([(LAYER.to_owned(), OBJECT_ID)]),
        layers_index: BTreeSet::from([LAYER.to_owned()]),
        path_name_members: BTreeSet::new(),
        path_index_session_id_member: session_id.to_owned(),
    };
    backup.add_path_name_member("record-step", session_id);
    backup
}

/// A proxy rule that holds every reply after the first `EXEC` behind `gate`
/// and counts every `EXEC` in `execs`.
fn hold_the_first_exec(gate: &ReplyGate, execs: &Arc<AtomicUsize>) -> ProxyRule {
    let (gate, execs) = (gate.clone(), Arc::clone(execs));
    Arc::new(move |args: &[Vec<u8>]| {
        let is_exec = args
            .first()
            .is_some_and(|name| name.eq_ignore_ascii_case(b"EXEC"));
        (is_exec && execs.fetch_add(1, Ordering::SeqCst) == 0)
            .then(|| ProxyAction::HoldReplies(gate.clone()))
    })
}

/// A `DeleteSession` marks the session while Step A's `EXEC` is in flight,
/// after the hook's last marker read (fail-on-old). The record step reads the
/// marker and writes nothing: `NoCheckpoint`, counted `not_recoverable`, no
/// record, no mapping, no layers-index member, and one `EXEC` in all (Step
/// A's). Step A's path-index member is there: the hook got past step 6a and
/// applied. With the marker check dropped from the record step's script, the
/// hook restored the deleted session (`Recovered`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_recovery_marked_during_step_a_answers_no_checkpoint_and_writes_no_record() {
    let tv = TestValkey::setup().await;
    let session_id = format!("record-step-marked-{}", uuid::Uuid::new_v4());
    let backup = backup_of(&session_id);
    let store = Arc::new(InMemoryCheckpointStore::new());
    store
        .put_session_checkpoint(
            &session_id,
            backup.to_bytes().expect("the backup encodes"),
            CheckpointMeta {
                session_id: session_id.clone(),
                ..CheckpointMeta::default()
            },
        )
        .await
        .expect("the in-memory store takes the backup");
    let gate = ReplyGate::new();
    let execs = Arc::new(AtomicUsize::new(0));
    let proxy =
        ScriptReplyProxy::start(test_valkey_addr(), hold_the_first_exec(&gate, &execs)).await;
    let mut config = crate::common::test_config();
    config.checkpoint_enabled = true;
    config.service_label = RECOVERY_SERVICE_LABEL.to_owned();
    let metrics: SharedMetrics =
        Arc::new(crate::common::metrics_scrape_only(RECOVERY_SERVICE_LABEL));
    let hook = SessionRecoveryHook::new(
        store,
        client_through(&proxy).await,
        tv.key_schema().clone(),
        Arc::new(config),
        Arc::clone(&metrics),
    );
    let keys = tv.key_schema();

    let recovery = {
        let session_id = session_id.clone();
        tokio::spawn(async move { hook.try_recover_session(&session_id).await })
    };
    gate.reply_held().await;
    let _: () = tv
        .client()
        .pool
        .set(
            keys.session_deleted_marker(&session_id),
            "1",
            None,
            None,
            false,
        )
        .await
        .expect("SET the deletion marker, as DeleteSession does");
    gate.open();
    let outcome = tokio::time::timeout(SETTLE, recovery)
        .await
        .expect("the recovery must finish")
        .expect("join");

    let record: std::collections::HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(keys.session_meta(&session_id))
        .await
        .expect("HGETALL");
    assert!(
        record.is_empty(),
        "a session marked deleted must not get its record back; got {record:?}"
    );
    let mapping: Option<String> = tv
        .client()
        .pool
        .get(keys.layer_mapping(&session_id, LAYER))
        .await
        .expect("GET");
    assert_eq!(mapping, None, "nor its layer mapping");
    let layers: Vec<String> = tv
        .client()
        .pool
        .smembers(keys.layers_index(&session_id))
        .await
        .expect("SMEMBERS");
    assert!(layers.is_empty(), "nor its layers index; got {layers:?}");
    assert!(
        matches!(outcome, SessionRecoveryOutcome::NoCheckpoint),
        "a session deleted during the apply is refused as steps 4a and 6a refuse it; got \
         {outcome:?}"
    );
    assert_eq!(recovery_attempts(&metrics, "not_recoverable"), 1);
    assert_eq!(recovery_attempts(&metrics, "restored"), 0);
    assert_eq!(recovery_attempts(&metrics, "error"), 0);
    assert_eq!(
        execs.load(Ordering::SeqCst),
        1,
        "one EXEC, Step A's: the record step is a script, not a transaction"
    );
    let path_index: Vec<String> = tv
        .client()
        .pool
        .smembers(keys.path_index(&backup.base_path))
        .await
        .expect("SMEMBERS");
    assert!(
        path_index.contains(&session_id),
        "fixture self-check: the hook got past step 6a and ran Step A; got {path_index:?}"
    );
}
