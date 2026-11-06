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

//! b.2tu — the four session scripts behind ListSessions' ghost walk and
//! CreateSession's metadata write, and the walk's reclaim of stale
//! `path_index` members.
//!
//! The race tests — a walk against a `CreateSession` parked between its names
//! commit and its metadata write, against a recovery landing between the
//! walk's read and its veto, and a walker command stalled until the create
//! completed (I1, I2, I3, I5) — are the inline tests of `src/grpc/session.rs`,
//! which can call the walk's private `reconcile_ghost_members` half by half.
//! CreateSession's settle of a failed metadata write (I7, I8, the failed
//! settle, a vetoed attempt whose release fails) and the in-doubt releases
//! (I9) are the `undo` module of `b_62m_tests.rs`. The recovery lifecycle
//! after a veto is `checkpoint::e2e_session_recovery_path_index_atomicity_test`.
//! This file holds:
//!
//! - **Script contracts**, each script run directly through its
//!   `storage::lua` handle:
//!   - `session_meta_veto.lua`: `PRESENT` writes nothing; `DELETED` (the
//!     metadata absent, the session deletion marker present) writes nothing;
//!     `VETOED` writes `1` with a TTL in (0, 24 h], and a repeat refreshes it;
//!     a wrong-typed key is refused `WRONGTYPE_KEY` before the write.
//!   - `session_meta_write.lua`: refused (`DELETED`, nothing written) under
//!     the session deletion marker, and (`VETOED`, nothing written) under a
//!     veto; otherwise exactly the given fields.
//!   - `session_index_release.lua`: removes both members, idempotently; never
//!     sends an `SREM` to a set its `TYPE` probe found absent (SDD §18.1, A1),
//!     shown on the store's `MONITOR` feed; a wrong-typed key is refused
//!     before any write.
//!   - `session_path_index_reconcile.lua`: removes exactly the `path_index`
//!     members no names member carries, splitting a names member on its first
//!     NUL as `KeySchema::split_path_name_member` does (a member with no NUL
//!     carries no id); removes more than one 500-member chunk; refuses a
//!     wrong-typed key before any write.
//! - **A re-sent metadata write after a delete**: a `CreateSession` whose
//!   landed metadata write is sent again after another call deleted the
//!   session answers `OK` with its id, and the session stays deleted, with
//!   no checkpoint write enqueued (its stamp key absent) and no backup
//!   written for it.
//! - **The reclaim** (I10): `SessionStore::list` reports `path_index_excess`
//!   when `path_index` holds an id the names index does not list; the
//!   ListSessions handler then removes the stale `path_index` members,
//!   leaving live sessions and a names-only session untouched. The check is
//!   by membership, not by count, so a names-only member at the same base
//!   path does not hide a `path_index`-only one. Plus the reclaim at a base
//!   path whose names set is absent (Valkey deletes a set with its last
//!   member, so an empty names set and an absent one are the same state).
//! - **The walk's retirement of a deleted id**: a ghost whose id carries the
//!   deletion marker is retired, both members released, with no veto
//!   written. (A `CreateSession` settle that meets the marker is in the
//!   `undo` module of `b_62m_tests.rs`.)
//! - **The walk's failure paths**, each staged with a wrong-typed key the
//!   script refuses: a refused veto keeps the ghost's members and sends no
//!   release; a refused release after the veto keeps the names member and a
//!   repeated walk re-vetoes idempotently; an unreadable `path_index` reads
//!   as no excess and fails nothing. ListSessions answers OK throughout.
//! - **A full store** (disruptive, a real `-OOM`): the veto, and a repeat,
//!   lands on a full store (its no-op `PEXPIRE` opener); the metadata write,
//!   with no opener, is refused and writes nothing; a `CreateSession` whose
//!   metadata write the full store refused answers `UNAVAILABLE` once, with
//!   its id vetoed and both index members released, so the client's retry
//!   answers `UNAVAILABLE` while memory is full — never `ALREADY_EXISTS` —
//!   and succeeds once it is back; and a ListSessions walk retires a ghost
//!   there. Each is `#[ignore]`d, runs only against
//!   `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL` (CI's `test-cluster-disruptive:`
//!   job), and prints a `SKIP:` line and passes when that is unset outside
//!   CI. Removing the veto's opener turned the veto, create and walk cases
//!   red (the create case on its retry answering `ALREADY_EXISTS`); the
//!   metadata-write case does not depend on it.
//! - **Cluster** (item 11): all four scripts, driven through `SessionStore`,
//!   run on a multi-shard cluster without `CROSSSLOT`. Gated on
//!   `LIVE_EDIT_TEST_CLUSTER_URL` (CI's `test-cluster:` job); it prints a
//!   `SKIP:` line and passes elsewhere.
//!
//! Needs `test-support` (and Valkey). The full-store `CreateSession` case
//! and the re-sent metadata write's case also need `test-fault-injection`
//! (the post-commit hook that fills the store, or deletes the session);
//! without it those two tests are compiled out.

use std::collections::{BTreeSet, HashMap};

use fred::interfaces::{ClientLike, HashesInterface, KeysInterface, SetsInterface};

use live_edit::storage::keys::{KeySchema, SESSION_META_VETO_TTL_MS};
use live_edit::storage::lua::{
    SESSION_INDEX_RELEASE_SCRIPT, SESSION_META_VETO_SCRIPT, SESSION_META_WRITE_SCRIPT,
    SESSION_PATH_INDEX_RECONCILE_SCRIPT,
};
use live_edit::storage::session::{GhostMember, GhostRetirement, SessionStore};
use live_edit::storage::test_support::TestValkey;

use super::disruptive_valkey::{assert_noeviction, disruptive_valkey_url, RestoreMaxmemory};
use super::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap, MonitoredCommand};
use crate::grpc::helpers::list_sessions_through_the_handler;

const CREATED_BY: &str = "b2tu-user";

/// The fixed literal every session script's error reply begins with.
const WRONGTYPE_KEY: &str = "WRONGTYPE_KEY: ";

fn session_store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

// ═════════════════════════════════════════════════════════════════════════════
// Script runners
// ═════════════════════════════════════════════════════════════════════════════

async fn run_veto(tv: &TestValkey, sid: &str) -> Result<Vec<String>, fred::error::Error> {
    let keys = tv.key_schema();
    SESSION_META_VETO_SCRIPT
        .evalsha_with_reload(
            tv.client().pool.next(),
            vec![
                keys.session_meta(sid),
                keys.session_meta_veto(sid),
                keys.session_deleted_marker(sid),
            ],
            vec![SESSION_META_VETO_TTL_MS.to_string()],
        )
        .await
}

async fn run_meta_write(
    tv: &TestValkey,
    sid: &str,
    fields: &[(&str, &str)],
) -> Result<Vec<String>, fred::error::Error> {
    let keys = tv.key_schema();
    let argv: Vec<String> = fields
        .iter()
        .flat_map(|(f, v)| [(*f).to_owned(), (*v).to_owned()])
        .collect();
    SESSION_META_WRITE_SCRIPT
        .evalsha_with_reload(
            tv.client().pool.next(),
            vec![
                keys.session_meta(sid),
                keys.session_meta_veto(sid),
                keys.session_deleted_marker(sid),
            ],
            argv,
        )
        .await
}

async fn run_release(
    tv: &TestValkey,
    base_path: &str,
    sid: &str,
    member: &str,
) -> Result<Vec<i64>, fred::error::Error> {
    let keys = tv.key_schema();
    SESSION_INDEX_RELEASE_SCRIPT
        .evalsha_with_reload(
            tv.client().pool.next(),
            vec![keys.path_index(base_path), keys.path_names_index(base_path)],
            vec![sid.to_owned(), member.to_owned()],
        )
        .await
}

async fn run_reconcile(tv: &TestValkey, base_path: &str) -> Result<i64, fred::error::Error> {
    let keys = tv.key_schema();
    SESSION_PATH_INDEX_RECONCILE_SCRIPT
        .evalsha_with_reload(
            tv.client().pool.next(),
            vec![keys.path_index(base_path), keys.path_names_index(base_path)],
            Vec::<String>::new(),
        )
        .await
}

/// Assert `result` is a script error reply leading with the fixed
/// `WRONGTYPE_KEY: ` literal and naming `key`.
fn assert_wrongtype<T: std::fmt::Debug>(result: Result<T, fred::error::Error>, key: &str) {
    match result {
        Ok(reply) => panic!("a wrong-typed {key} must be refused; got {reply:?}"),
        Err(e) => {
            let details = e.details();
            assert!(
                details.starts_with(WRONGTYPE_KEY) && details.contains(key),
                "the refusal must lead with {WRONGTYPE_KEY:?} and name {key}; got {details:?}"
            );
        }
    }
}

async fn members(tv: &TestValkey, key: String) -> BTreeSet<String> {
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(key)
        .await
        .expect("SMEMBERS must succeed");
    members.into_iter().collect()
}

async fn exists(tv: &TestValkey, key: String) -> bool {
    let found: i64 = tv
        .client()
        .pool
        .exists(key)
        .await
        .expect("EXISTS must succeed");
    found == 1
}

async fn sadd(tv: &TestValkey, key: String, members: Vec<String>) {
    let _: i64 = tv
        .client()
        .pool
        .sadd(key, members)
        .await
        .expect("SADD must succeed");
}

/// Write `sid`'s session deletion marker, as `DeleteSession` does.
async fn mark_deleted(tv: &TestValkey, sid: &str) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().session_deleted_marker(sid),
            "1",
            None,
            None,
            false,
        )
        .await
        .expect("SET the deletion marker");
}

fn set_of(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// The session ids one `ListSessions` of `base_path` lists, sorted, through
/// [`list_sessions_through_the_handler`] (recovery disabled, so every ghost's
/// probe answers `NoCheckpoint`).
async fn listed_ids(tv: &TestValkey, base_path: &str) -> Vec<String> {
    let mut ids: Vec<String> = list_sessions_through_the_handler(tv, base_path)
        .await
        .sessions
        .into_iter()
        .map(|s| s.session_id)
        .collect();
    ids.sort();
    ids
}

// ═════════════════════════════════════════════════════════════════════════════
// session_meta_veto.lua
// ═════════════════════════════════════════════════════════════════════════════

/// An id whose metadata exists is a session: the veto answers `PRESENT` and
/// writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_meta_veto_on_an_id_with_metadata_answers_present_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let sid = "b2tu-veto-present";
    let _: i64 = tv
        .client()
        .pool
        .hset(tv.key_schema().session_meta(sid), [("name", "n")])
        .await
        .expect("HSET");

    let reply = run_veto(&tv, sid).await.expect("the veto must run");

    assert_eq!(reply, vec!["PRESENT".to_owned()]);
    assert!(
        !exists(&tv, tv.key_schema().session_meta_veto(sid)).await,
        "PRESENT must write no veto"
    );
}

/// An id with no metadata is vetoed: `1`, with a TTL in (0, 24 h]; a repeat
/// answers `VETOED` again and refreshes the TTL.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_meta_veto_on_an_id_without_metadata_writes_the_veto_with_its_ttl() {
    let tv = TestValkey::setup().await;
    let sid = "b2tu-veto-absent";
    let veto_key = tv.key_schema().session_meta_veto(sid);

    let reply = run_veto(&tv, sid).await.expect("the veto must run");

    assert_eq!(reply, vec!["VETOED".to_owned()]);
    let value: Option<String> = tv.client().pool.get(&veto_key).await.expect("GET");
    assert_eq!(value.as_deref(), Some("1"));
    let pttl: i64 = tv.client().pool.pttl(&veto_key).await.expect("PTTL");
    assert!(
        pttl > 0 && pttl <= SESSION_META_VETO_TTL_MS as i64,
        "the veto's TTL must be in (0, {SESSION_META_VETO_TTL_MS}] ms; got {pttl}"
    );
    assert!(
        !exists(&tv, tv.key_schema().session_meta(sid)).await,
        "the veto writes no metadata"
    );

    // A repeat — the next walk retiring the same ghost — refreshes the TTL.
    let _: bool = tv
        .client()
        .pool
        .pexpire(&veto_key, 1_000, None)
        .await
        .expect("PEXPIRE");
    let again = run_veto(&tv, sid)
        .await
        .expect("the repeated veto must run");
    assert_eq!(again, vec!["VETOED".to_owned()]);
    let refreshed: i64 = tv.client().pool.pttl(&veto_key).await.expect("PTTL");
    assert!(
        refreshed > 1_000,
        "a repeated veto must refresh the TTL; got {refreshed} ms"
    );
}

/// An id with no metadata that carries the session deletion marker was a
/// session, since deleted (fail-on-old): the veto answers `DELETED` and writes
/// no veto, and no metadata. Without the script's marker check, it vetoed the
/// id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_meta_veto_on_a_deleted_id_answers_deleted_and_writes_no_veto() {
    let tv = TestValkey::setup().await;
    let sid = "b2tu-veto-deleted";
    mark_deleted(&tv, sid).await;

    let reply = run_veto(&tv, sid).await.expect("the veto must run");

    assert!(
        !exists(&tv, tv.key_schema().session_meta_veto(sid)).await,
        "a deleted id must get no veto"
    );
    assert!(
        !exists(&tv, tv.key_schema().session_meta(sid)).await,
        "the veto writes no metadata"
    );
    assert_eq!(reply, vec!["DELETED".to_owned()]);
}

/// A wrong-typed metadata or veto key is refused `WRONGTYPE_KEY` before the
/// veto's one write.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_meta_veto_with_a_wrong_typed_key_is_refused_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();

    let sid = "b2tu-veto-meta-wrongtype";
    let _: () = tv
        .client()
        .pool
        .set(keys.session_meta(sid), "not-a-hash", None, None, false)
        .await
        .expect("SET");
    assert_wrongtype(run_veto(&tv, sid).await, &keys.session_meta(sid));
    assert!(!exists(&tv, keys.session_meta_veto(sid)).await);

    let sid = "b2tu-veto-veto-wrongtype";
    let _: i64 = tv
        .client()
        .pool
        .hset(keys.session_meta_veto(sid), [("f", "v")])
        .await
        .expect("HSET");
    assert_wrongtype(run_veto(&tv, sid).await, &keys.session_meta_veto(sid));
    let still_hash: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(keys.session_meta_veto(sid))
        .await
        .expect("HGETALL");
    assert_eq!(
        still_hash.len(),
        1,
        "the refused run must leave the key as it was"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// session_meta_write.lua
// ═════════════════════════════════════════════════════════════════════════════

/// A vetoed id never gets metadata: the write answers `VETOED` and writes
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_meta_write_on_a_vetoed_id_answers_vetoed_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let sid = "b2tu-write-vetoed";
    assert_eq!(
        run_veto(&tv, sid).await.expect("veto"),
        vec!["VETOED".to_owned()]
    );

    let reply = run_meta_write(&tv, sid, &[("name", "n"), ("base_path", "/p")])
        .await
        .expect("the write must run");

    assert_eq!(reply, vec!["VETOED".to_owned()]);
    assert!(
        !exists(&tv, tv.key_schema().session_meta(sid)).await,
        "a vetoed id's metadata write must write nothing"
    );
}

/// A deleted session never gets its metadata back (fail-on-old): with the
/// deletion marker present the write answers `DELETED` and writes nothing.
/// Without the script's marker check, it wrote the metadata.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_meta_write_on_a_deleted_id_answers_deleted_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let sid = "b2tu-write-deleted";
    mark_deleted(&tv, sid).await;

    let reply = run_meta_write(&tv, sid, &[("name", "n"), ("base_path", "/p")])
        .await
        .expect("the write must run");

    assert!(
        !exists(&tv, tv.key_schema().session_meta(sid)).await,
        "a deleted session's metadata write must write nothing"
    );
    assert_eq!(reply, vec!["DELETED".to_owned()]);
}

/// A `CreateSession` whose metadata write landed, and whose session another
/// call's `DeleteSession` deleted before the write's reply came back, answers
/// `OK` with its id when the write is sent again (fail-on-old): the re-send
/// meets the deletion marker and writes nothing, so the session stays
/// deleted, and the call enqueues no checkpoint write for it. Without the
/// script's marker check, the re-send brought the session back.
///
/// The enqueue is read from the session's checkpoint stamp key
/// (`KeySchema::session_checkpoint_state`): a write's build mints it before
/// it finds the session gone and writes no backup, and nothing else in this
/// fixture writes it. With the `Deleted` arm enqueuing a write, the key was
/// there. No backup is written either.
///
/// Staged from the post-commit hook, which runs after the names commit and
/// before the call's metadata write: it writes the metadata, as the landed
/// first send did, then deletes the session through `SessionStore::delete`,
/// which marks it. The call's own metadata write then plays the data pool's
/// re-send.
///
/// Needs `test-fault-injection` (the post-commit hook): without the feature
/// it is compiled out.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_whose_landed_metadata_write_is_resent_after_a_delete_answers_ok_and_stays_deleted(
) {
    use std::sync::{Arc, Mutex};

    use futures::FutureExt;
    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::{CheckpointStore, SessionCheckpointWriter};
    use live_edit::storage::session::PostCommitHook;
    use tokio_util::task::TaskTracker;

    const NAME: &str = "b2tu-deleted-create";
    const BASE: &str = "/b2tu-deleted-create";
    /// One deadline over the create, and one over the backup writer's drain.
    const BUDGET: std::time::Duration = std::time::Duration::from_secs(20);

    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let attempts: Arc<Mutex<Vec<String>>> = Arc::default();
    let hook: PostCommitHook = {
        let attempts = attempts.clone();
        let (client, keys) = (tv.client().clone(), keys.clone());
        Arc::new(move |session_id: String| {
            attempts.lock().unwrap().push(session_id.clone());
            let (client, keys) = (client.clone(), keys.clone());
            async move {
                let _: () = client
                    .pool
                    .hset(
                        keys.session_meta(&session_id),
                        [
                            ("name", NAME),
                            ("base_path", BASE),
                            ("created_by", CREATED_BY),
                            ("created_at", "1"),
                        ],
                    )
                    .await
                    .expect("HSET the metadata, as the landed first send did");
                SessionStore::new(client, keys)
                    .delete(&session_id)
                    .await
                    .expect("another call's DeleteSession");
            }
            .boxed()
        })
    };
    let backups = Arc::new(InMemoryCheckpointStore::new());
    let tracker = TaskTracker::new();
    let writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&backups) as Arc<dyn CheckpointStore>,
        tv.client().clone(),
        keys.clone(),
        tracker.clone(),
        Arc::new(crate::common::metrics_scrape_only("b2tu-deleted-create")),
        "b2tu-deleted-create".to_owned(),
        true,
    ));
    let store = session_store(&tv)
        .with_checkpoint_writer(writer)
        .with_post_commit_hook_for_test(hook);

    let created = tokio::time::timeout(BUDGET, store.create(NAME, BASE, CREATED_BY))
        .await
        .expect("the create did not settle within its budget");
    tracker.close();
    tokio::time::timeout(BUDGET, tracker.wait())
        .await
        .expect("the backup writer did not drain within its budget");

    let attempts = attempts.lock().unwrap().clone();
    assert_eq!(attempts.len(), 1, "one attempt; got {attempts:?}");
    let sid = &attempts[0];
    assert!(
        !exists(&tv, keys.session_meta(sid)).await,
        "the re-sent metadata write must not bring the deleted session back"
    );
    assert!(
        exists(&tv, keys.session_deleted_marker(sid)).await,
        "the deletion marker stays"
    );
    assert_eq!(
        created.expect("the create happened, so it answers OK"),
        *sid,
        "with the attempt's id"
    );
    assert!(
        backups
            .get_session_checkpoint(sid)
            .await
            .expect("the in-memory store answers")
            .is_none(),
        "no backup is written for the deleted session"
    );
    assert!(
        !exists(&tv, keys.session_checkpoint_state(sid)).await,
        "no checkpoint write is enqueued for the deleted session: a write's build mints the \
         stamp key before it finds the session gone, so a key here means one was enqueued"
    );
}

/// Without a veto the write answers `OK` and writes exactly the fields it was
/// given.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_meta_write_without_a_veto_writes_exactly_the_given_fields() {
    let tv = TestValkey::setup().await;
    let sid = "b2tu-write-ok";
    let fields = [
        ("name", "n"),
        ("base_path", "/p"),
        ("created_by", "u"),
        ("created_at", "42"),
    ];

    let reply = run_meta_write(&tv, sid, &fields)
        .await
        .expect("the write must run");

    assert_eq!(reply, vec!["OK".to_owned()]);
    let written: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().session_meta(sid))
        .await
        .expect("HGETALL");
    let want: HashMap<String, String> = fields
        .iter()
        .map(|(f, v)| ((*f).to_owned(), (*v).to_owned()))
        .collect();
    assert_eq!(written, want);
}

// ═════════════════════════════════════════════════════════════════════════════
// session_index_release.lua
// ═════════════════════════════════════════════════════════════════════════════

/// The release removes the id's two members, and only those; a repeat is a
/// no-op.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_index_release_removes_both_members_and_is_idempotent() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let base = "/b2tu-release";
    let member = KeySchema::path_name_member("n", "sid-1");
    let other = KeySchema::path_name_member("m", "sid-2");
    sadd(
        &tv,
        keys.path_index(base),
        vec!["sid-1".into(), "sid-2".into()],
    )
    .await;
    sadd(
        &tv,
        keys.path_names_index(base),
        vec![member.clone(), other.clone()],
    )
    .await;

    assert_eq!(
        run_release(&tv, base, "sid-1", &member)
            .await
            .expect("release"),
        vec![1, 1]
    );
    assert_eq!(
        members(&tv, keys.path_index(base)).await,
        set_of(&["sid-2"])
    );
    assert_eq!(
        members(&tv, keys.path_names_index(base)).await,
        set_of(&[other.as_str()])
    );

    assert_eq!(
        run_release(&tv, base, "sid-1", &member)
            .await
            .expect("repeat release"),
        vec![0, 0],
        "releasing members already gone is a no-op"
    );
}

/// The release never names a set its `TYPE` probe found absent (SDD §18.1, A1:
/// every write after the first names a key present then). Seen on the store's
/// `MONITOR` feed: with one of the two sets absent, the script sends exactly
/// one `SREM`, to the set that exists.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_index_release_sends_no_srem_to_an_absent_set() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    // Only `path_index` exists.
    let base = "/b2tu-release-names-absent";
    let member = KeySchema::path_name_member("n", "sid-a");
    sadd(&tv, keys.path_index(base), vec!["sid-a".into()]).await;
    assert_eq!(
        run_release(&tv, base, "sid-a", &member)
            .await
            .expect("release"),
        vec![1, 0]
    );
    let fence = monitor_fence(tv.prefix());
    assert!(!exists(&tv, fence.clone()).await);
    let srems: Vec<Vec<String>> = tap
        .drain_until(&fence)
        .await
        .into_iter()
        .filter(|c| is_this_tests_script_srem(c, tv.prefix()))
        .map(|c| c.args)
        .collect();
    assert_eq!(
        srems,
        vec![vec![
            "SREM".to_owned(),
            keys.path_index(base),
            "sid-a".to_owned()
        ]],
        "the script must send one SREM, to path_index, and none to the absent names set"
    );

    // Only the names set exists.
    let base = "/b2tu-release-index-absent";
    let member = KeySchema::path_name_member("n", "sid-b");
    sadd(&tv, keys.path_names_index(base), vec![member.clone()]).await;
    assert_eq!(
        run_release(&tv, base, "sid-b", &member)
            .await
            .expect("release"),
        vec![0, 1]
    );
    let fence = monitor_fence(tv.prefix());
    assert!(!exists(&tv, fence.clone()).await);
    let srems: Vec<Vec<String>> = tap
        .drain_until(&fence)
        .await
        .into_iter()
        .filter(|c| is_this_tests_script_srem(c, tv.prefix()))
        .map(|c| c.args)
        .collect();
    assert_eq!(
        srems,
        vec![vec!["SREM".to_owned(), keys.path_names_index(base), member]],
        "the script must send one SREM, to the names set, and none to the absent path_index"
    );
}

/// A script-issued `SREM` on one of this test's keys. The feed carries every
/// client's commands on a shared store, other tests' scripts included, so the
/// key prefix is what makes a count here this test's.
fn is_this_tests_script_srem(cmd: &MonitoredCommand, prefix: &str) -> bool {
    cmd.issued_by_script()
        && cmd.name() == "SREM"
        && cmd.args.get(1).is_some_and(|key| key.starts_with(prefix))
}

/// A wrong-typed key is refused before any write: the other set keeps its
/// member.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_index_release_with_a_wrong_typed_key_is_refused_before_any_write() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let base = "/b2tu-release-wrongtype";
    let member = KeySchema::path_name_member("n", "sid-w");
    sadd(&tv, keys.path_index(base), vec!["sid-w".into()]).await;
    let _: () = tv
        .client()
        .pool
        .set(keys.path_names_index(base), "not-a-set", None, None, false)
        .await
        .expect("SET");

    assert_wrongtype(
        run_release(&tv, base, "sid-w", &member).await,
        &keys.path_names_index(base),
    );
    assert_eq!(
        members(&tv, keys.path_index(base)).await,
        set_of(&["sid-w"]),
        "the refused release must not have removed the path_index member"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// session_path_index_reconcile.lua
// ═════════════════════════════════════════════════════════════════════════════

/// The reconcile removes exactly the `path_index` members no names member
/// carries. A names member's id is what follows its FIRST NUL — the rule
/// `KeySchema::split_path_name_member` follows — and a member with no NUL
/// carries no id. The expected survivors are computed from that function, so
/// the script and the Rust parser cannot drift.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_path_index_reconcile_removes_exactly_the_members_no_names_member_carries() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let base = "/b2tu-reconcile";
    let names = vec![
        "a\0sid-1".to_owned(),
        "no-nul".to_owned(),
        "x\0y\0sid-3".to_owned(),
        "c\0sid-4".to_owned(),
    ];
    let index = vec![
        "sid-1".to_owned(),
        "no-nul".to_owned(),
        "y\0sid-3".to_owned(),
        "sid-3".to_owned(),
        "stale".to_owned(),
        "sid-4".to_owned(),
    ];
    sadd(&tv, keys.path_names_index(base), names.clone()).await;
    sadd(&tv, keys.path_index(base), index.clone()).await;
    let carried: BTreeSet<String> = names
        .iter()
        .filter_map(|m| KeySchema::split_path_name_member(m))
        .map(|(_, sid)| sid.to_owned())
        .collect();
    let want: BTreeSet<String> = index
        .iter()
        .filter(|sid| carried.contains(*sid))
        .cloned()
        .collect();
    assert_eq!(
        want,
        set_of(&["sid-1", "y\0sid-3", "sid-4"]),
        "fixture self-check: split_path_name_member splits on the first NUL and skips a \
         member with none"
    );

    let removed = run_reconcile(&tv, base).await.expect("reconcile");

    assert_eq!(removed, (index.len() - want.len()) as i64);
    assert_eq!(members(&tv, keys.path_index(base)).await, want);
    assert_eq!(
        members(&tv, keys.path_names_index(base)).await,
        names.into_iter().collect::<BTreeSet<_>>(),
        "the reconcile never touches the names index"
    );
    assert_eq!(
        run_reconcile(&tv, base).await.expect("repeat reconcile"),
        0,
        "nothing stale is left"
    );
}

/// More stale members than one `SREM` chunk (500) are all removed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_path_index_reconcile_removes_more_than_one_chunk() {
    const STALE: usize = 1_201;
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let base = "/b2tu-reconcile-chunks";
    sadd(
        &tv,
        keys.path_names_index(base),
        vec!["live\0sid-live".into()],
    )
    .await;
    let mut index: Vec<String> = (0..STALE).map(|i| format!("stale-{i}")).collect();
    index.push("sid-live".into());
    sadd(&tv, keys.path_index(base), index).await;

    let removed = run_reconcile(&tv, base).await.expect("reconcile");

    assert_eq!(removed, STALE as i64);
    assert_eq!(
        members(&tv, keys.path_index(base)).await,
        set_of(&["sid-live"])
    );
}

/// An absent `path_index` is a no-op (`0`); a wrong-typed names key is refused
/// before any write.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_path_index_reconcile_absent_index_is_a_no_op_and_a_wrong_type_is_refused() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();

    assert_eq!(
        run_reconcile(&tv, "/b2tu-reconcile-absent")
            .await
            .expect("reconcile"),
        0
    );

    let base = "/b2tu-reconcile-wrongtype";
    sadd(&tv, keys.path_index(base), vec!["sid-x".into()]).await;
    let _: () = tv
        .client()
        .pool
        .set(keys.path_names_index(base), "not-a-set", None, None, false)
        .await
        .expect("SET");
    assert_wrongtype(run_reconcile(&tv, base).await, &keys.path_names_index(base));
    assert_eq!(
        members(&tv, keys.path_index(base)).await,
        set_of(&["sid-x"]),
        "the refused reconcile must not have removed anything"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The reclaim of stale `path_index` members (I10)
// ═════════════════════════════════════════════════════════════════════════════

/// `list` reports `path_index_excess` exactly when `path_index` holds a
/// session id the names index does not list.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_list_reports_path_index_excess_only_when_path_index_holds_an_unlisted_id() {
    let tv = TestValkey::setup().await;
    let store = session_store(&tv);
    let base = "/b2tu-excess";
    store.create("one", base, CREATED_BY).await.expect("create");
    store.create("two", base, CREATED_BY).await.expect("create");

    let listed = store.list(base).await.expect("list");
    assert!(!listed.path_index_excess, "two sessions, two members each");

    sadd(&tv, tv.key_schema().path_index(base), vec!["stale".into()]).await;
    let listed = store.list(base).await.expect("list");
    assert!(
        listed.path_index_excess,
        "a path_index member no names member carries is an excess"
    );
    assert_eq!(listed.present.len(), 2);
    assert!(listed.ghosts.is_empty());
}

/// A names-only member does not hide a `path_index`-only member at the same
/// base path: the stale check is by membership, not by count.
///
/// The names-only member is a session whose `path_index` member is gone but
/// whose names member and metadata remain — the state a `DeleteSession`
/// stopped after its first `SREM` leaves (its retry finishes it). With one of
/// each beside a live session, `path_index` and the names index hold the same
/// number of ids, so a count comparison reads "no excess". The ListSessions
/// walk must still reclaim the `path_index`-only member, while the names-only
/// session — it has metadata, so it is listed, never a ghost — and the live
/// session keep every member they had.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_reclaims_a_path_index_only_member_beside_a_names_only_member() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let store = session_store(&tv);
    let base = "/b2tu-reclaim-masked";
    let live = store
        .create("live", base, CREATED_BY)
        .await
        .expect("create");
    let names_only = store
        .create("names-only", base, CREATED_BY)
        .await
        .expect("create");
    let _: i64 = tv
        .client()
        .pool
        .srem(keys.path_index(base), &names_only)
        .await
        .expect("SREM the names-only session's path_index member");
    sadd(&tv, keys.path_index(base), vec!["stale-sid".into()]).await;
    assert_eq!(
        members(&tv, keys.path_index(base)).await.len(),
        members(&tv, keys.path_names_index(base)).await.len(),
        "fixture self-check: equal counts, so only a membership test sees the stale member"
    );

    assert!(
        store.list(base).await.expect("list").path_index_excess,
        "an id in path_index that the names index does not list is an excess, whatever the \
         counts"
    );
    let listed = listed_ids(&tv, base).await;

    let mut want = vec![live.clone(), names_only.clone()];
    want.sort();
    assert_eq!(
        listed, want,
        "the live session and the names-only session (it has metadata) are listed"
    );
    assert_eq!(
        members(&tv, keys.path_index(base)).await,
        set_of(&[live.as_str()]),
        "the path_index-only member is reclaimed; the live session keeps its member, and the \
         names-only session gains none"
    );
    assert_eq!(
        members(&tv, keys.path_names_index(base)).await,
        [
            KeySchema::path_name_member("live", &live),
            KeySchema::path_name_member("names-only", &names_only),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        "the names index is untouched"
    );
    for sid in [&live, &names_only] {
        assert!(
            exists(&tv, keys.session_meta(sid)).await,
            "{sid} keeps its metadata"
        );
    }
}

/// A `path_index` member at a base path whose names set is absent — every
/// session there gone, or none ever listed — is reclaimed by the next
/// ListSessions of that base path, and a live session at another base path is
/// untouched. (A names set emptied of its last member no longer exists, so
/// "empty" and "absent" are the same state.)
///
/// The regression this pins: `SessionStore::list` returning on an empty names
/// set before it reads `path_index`, so such a member is never reclaimed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_reclaims_a_path_index_member_at_a_base_path_with_no_names() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let store = session_store(&tv);
    let base = "/b2tu-reclaim-no-names";
    let other_base = "/b2tu-reclaim-other";
    let live = store
        .create("live", other_base, CREATED_BY)
        .await
        .expect("create");
    sadd(&tv, keys.path_index(base), vec!["stale-sid".into()]).await;
    assert!(!exists(&tv, keys.path_names_index(base)).await);

    let listed = listed_ids(&tv, base).await;

    assert!(listed.is_empty(), "nothing is listed at {base}: {listed:?}");
    assert!(
        members(&tv, keys.path_index(base)).await.is_empty(),
        "the stale path_index member at a base path with no names must be reclaimed"
    );
    assert_eq!(
        members(&tv, keys.path_index(other_base)).await,
        set_of(&[live.as_str()]),
        "the live session at another base path keeps its path_index member"
    );
    assert_eq!(
        members(&tv, keys.path_names_index(other_base)).await,
        set_of(&[KeySchema::path_name_member("live", &live).as_str()]),
        "the live session at another base path keeps its names member"
    );
    assert_eq!(
        listed_ids(&tv, other_base).await,
        vec![live],
        "the live session at another base path is still listed"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The walk's retirement of a deleted id
// ═════════════════════════════════════════════════════════════════════════════

/// A ghost whose id carries the session deletion marker, its metadata absent,
/// is retired without a veto (fail-on-old): the veto answers `DELETED`, both
/// index members are released, and no veto key is written. The metadata
/// write refuses a marked id, so no create can complete it. Without the veto
/// script's marker check, the walk vetoed the id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retire_ghost_member_of_a_deleted_id_releases_both_members_and_writes_no_veto() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let base = "/b2tu-retire-deleted";
    let sid = format!("deleted-ghost-{}", uuid::Uuid::new_v4());
    let member = KeySchema::path_name_member("deleted-ghost", &sid);
    sadd(&tv, keys.path_index(base), vec![sid.clone()]).await;
    sadd(&tv, keys.path_names_index(base), vec![member.clone()]).await;
    mark_deleted(&tv, &sid).await;
    let ghost = GhostMember {
        name: "deleted-ghost".into(),
        sid: sid.clone(),
        composite_member: member,
    };

    let retired = session_store(&tv)
        .retire_ghost_member(base, &ghost)
        .await
        .expect("the retirement must run");

    assert!(
        !exists(&tv, keys.session_meta_veto(&sid)).await,
        "a deleted id must get no veto"
    );
    assert!(
        members(&tv, keys.path_index(base)).await.is_empty()
            && members(&tv, keys.path_names_index(base)).await.is_empty(),
        "both index members are released"
    );
    assert!(!exists(&tv, keys.session_meta(&sid)).await);
    assert_eq!(retired, GhostRetirement::Retired);
}

// ═════════════════════════════════════════════════════════════════════════════
// The walk's failure paths
// ═════════════════════════════════════════════════════════════════════════════
//
// Each failure is staged with a wrong-typed key the script refuses before any
// write, so the failure is the store's own refusal, not a seam. Not covered
// here, and named in the inline tests of `src/grpc/session.rs`: the walk's
// `PRESENT` → re-read → `Ok(None)` arm (a DeleteSession between the veto and
// the re-read) and the handler's arm for a failed `reconcile_path_index` (a
// wrong-typed names key fails `list` first, and a wrong-typed `path_index`
// reads as no excess, so the reconcile is never reached).

/// Seed a ghost at `base`: a names member for `sid` under `name`, and no
/// metadata. The `path_index` member is the caller's.
async fn seed_names_member(tv: &TestValkey, base: &str, name: &str, sid: &str) {
    sadd(
        tv,
        tv.key_schema().path_names_index(base),
        vec![KeySchema::path_name_member(name, sid)],
    )
    .await;
}

/// A ghost whose veto the store refuses (its veto key holds a hash) is kept:
/// the walk sends no release, both its members stay, and ListSessions still
/// answers OK, without the ghost. The refused veto wrote nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_keeps_a_ghost_whose_veto_is_refused() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let base = "/b2tu-veto-refused";
    let sid = "b2tu-veto-refused-sid";
    seed_names_member(&tv, base, "ghost", sid).await;
    sadd(&tv, keys.path_index(base), vec![sid.into()]).await;
    let _: i64 = tv
        .client()
        .pool
        .hset(keys.session_meta_veto(sid), [("f", "v")])
        .await
        .expect("HSET");

    let listed = listed_ids(&tv, base).await;

    assert!(listed.is_empty(), "the ghost is not listed: {listed:?}");
    assert_eq!(
        members(&tv, keys.path_index(base)).await,
        set_of(&[sid]),
        "a walk whose veto failed must not release the path_index member"
    );
    assert_eq!(
        members(&tv, keys.path_names_index(base)).await,
        set_of(&[KeySchema::path_name_member("ghost", sid).as_str()]),
        "a walk whose veto failed must not release the names member"
    );
    let veto: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(keys.session_meta_veto(sid))
        .await
        .expect("HGETALL");
    assert_eq!(veto.len(), 1, "the refused veto wrote nothing");
}

/// A ghost whose veto lands but whose release the store refuses (`path_index`
/// holds a string) keeps its names member and its veto; ListSessions still
/// answers OK without it, and the next walk's repeated veto answers `VETOED`
/// again and refreshes the TTL — idempotent. The store's `MONITOR` feed shows
/// that each walk did send the release (its script's `TYPE` probe of
/// `path_index`), so the kept member is the store's refusal, not a walk that
/// stopped at the veto. The same wrong-typed `path_index` makes `list`'s
/// stale-member read fail, which reads as no excess and fails nothing: a live
/// session at the base path is still listed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_keeps_a_vetoed_ghost_whose_release_is_refused() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let base = "/b2tu-release-refused";
    let sid = "b2tu-release-refused-sid";
    let live = session_store(&tv)
        .create("live", base, CREATED_BY)
        .await
        .expect("create");
    let _: i64 = tv
        .client()
        .pool
        .del(keys.path_index(base))
        .await
        .expect("DEL");
    let _: () = tv
        .client()
        .pool
        .set(keys.path_index(base), "not-a-set", None, None, false)
        .await
        .expect("SET");
    seed_names_member(&tv, base, "ghost", sid).await;
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let index_key = keys.path_index(base);

    for walk in 1..=2 {
        let listed = listed_ids(&tv, base).await;
        assert_eq!(
            listed,
            vec![live.clone()],
            "walk {walk}: ListSessions answers OK with the live session, without the ghost"
        );
        // The release reached the store: its script's `TYPE` probe of
        // `path_index`, the refusal's first step, is on the feed. No other
        // script probes this key here (the reconcile runs only on an excess,
        // and an unreadable `path_index` reads as none), and `list`'s own read
        // of it is a client command, not a script's.
        let fence = monitor_fence(tv.prefix());
        assert!(!exists(&tv, fence.clone()).await);
        let release_probes = tap
            .drain_until(&fence)
            .await
            .into_iter()
            .filter(|c| {
                c.issued_by_script()
                    && c.name() == "TYPE"
                    && c.args.get(1).is_some_and(|key| *key == index_key)
            })
            .count();
        assert_eq!(
            release_probes, 1,
            "walk {walk}: the walk must send the release after its veto (one script `TYPE` \
             probe of path_index), not stop at the veto"
        );
        let value: Option<String> = tv
            .client()
            .pool
            .get(keys.session_meta_veto(sid))
            .await
            .expect("GET veto");
        assert_eq!(
            value.as_deref(),
            Some("1"),
            "walk {walk}: the veto landed before the refused release"
        );
        if walk == 1 {
            // Age the veto, so walk 2's refresh is visible.
            let _: bool = tv
                .client()
                .pool
                .pexpire(keys.session_meta_veto(sid), 1_000, None)
                .await
                .expect("PEXPIRE");
        } else {
            let pttl: i64 = tv
                .client()
                .pool
                .pttl(keys.session_meta_veto(sid))
                .await
                .expect("PTTL");
            assert!(
                pttl > 1_000,
                "walk 2's repeated veto refreshes the TTL; got {pttl} ms"
            );
        }
        assert_eq!(
            members(&tv, keys.path_names_index(base)).await,
            [
                KeySchema::path_name_member("live", &live),
                KeySchema::path_name_member("ghost", sid),
            ]
            .into_iter()
            .collect::<BTreeSet<_>>(),
            "walk {walk}: the refused release keeps the ghost's names member"
        );
        let index: Option<String> = tv
            .client()
            .pool
            .get(keys.path_index(base))
            .await
            .expect("GET path_index");
        assert_eq!(
            index.as_deref(),
            Some("not-a-set"),
            "walk {walk}: the refused release wrote nothing"
        );
    }
}

/// `list`'s read of `path_index` failing (the key holds a string) reads as no
/// excess and never fails the list: the sessions and the ghost are still
/// returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_list_with_an_unreadable_path_index_reports_no_excess_and_succeeds() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let store = session_store(&tv);
    let base = "/b2tu-unreadable-index";
    let live = store
        .create("live", base, CREATED_BY)
        .await
        .expect("create");
    seed_names_member(&tv, base, "ghost", "b2tu-unreadable-ghost").await;
    let _: i64 = tv
        .client()
        .pool
        .del(keys.path_index(base))
        .await
        .expect("DEL");
    let _: () = tv
        .client()
        .pool
        .set(keys.path_index(base), "not-a-set", None, None, false)
        .await
        .expect("SET");

    let listed = store
        .list(base)
        .await
        .expect("a failed path_index read must not fail the list");

    assert!(
        !listed.path_index_excess,
        "an unreadable path_index reads as no excess"
    );
    assert_eq!(
        listed
            .present
            .iter()
            .map(|m| m.session_id.as_str())
            .collect::<Vec<_>>(),
        vec![live.as_str()]
    );
    assert_eq!(
        listed
            .ghosts
            .iter()
            .map(|g| g.sid.as_str())
            .collect::<Vec<_>>(),
        vec!["b2tu-unreadable-ghost"]
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// A full store (disruptive)
// ═════════════════════════════════════════════════════════════════════════════
//
// On a store at `maxmemory` under `noeviction` the veto lands — its `SET`
// follows a no-op `PEXPIRE` opener (SDD §7.5.1) — and the release's `SREM`s
// never grow memory, so CreateSession's settle and the ListSessions walk both
// free a name there; the metadata write has no opener, so a full store
// refuses it and never gains a session. Each test sets the server's
// `maxmemory`, a server-wide setting, so it runs only against
// `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL` (`super::disruptive_valkey`),
// skipping with a `SKIP:` line when that is unset outside CI; it is
// `#[ignore]`d, and its `store_write_refusal_disruptive_` name is what CI's
// `test-cluster-disruptive:` filter and the one-at-a-time
// `disruptive-valkey-config` nextest group match. Each reads what it asserts
// while the store is full, then restores `maxmemory` before asserting (the
// guard's `Drop` restores it on a panic too).

/// Connections of a disruptive test's fixture.
const DISRUPTIVE_POOL_SIZE: u32 = 4;

/// The disruptive store's fixture and an admin client on it, checked to run
/// `noeviction`; `None` when the store is not configured.
async fn disruptive_fixture() -> Option<(TestValkey, fred::clients::Client)> {
    let url = disruptive_valkey_url()?;
    let tv = TestValkey::setup_with_url(url, DISRUPTIVE_POOL_SIZE).await;
    let admin = tv.client().pool.next().clone();
    assert_noeviction(&admin).await;
    Some((tv, admin))
}

/// Assert the store refuses a plain `SET` out of memory: it is full, so what
/// the test shows ran on a full store.
async fn assert_refuses_writes(tv: &TestValkey) {
    let refused: Result<(), fred::error::Error> = tv
        .client()
        .pool
        .set(format!("{}full-probe", tv.prefix()), "x", None, None, false)
        .await;
    let error = refused.expect_err("precondition: the full store refuses a plain SET");
    assert!(
        error.details().starts_with("OOM "),
        "precondition: refused out of memory; got {error:?}"
    );
}

/// The veto lands on a full store, and so does a repeat (the walk's next
/// pass over a ghost whose release failed): the no-op `PEXPIRE` opener lets
/// the deny-oom `SET` after it run.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_session_meta_veto_lands_on_a_full_store() {
    let Some((tv, admin)) = disruptive_fixture().await else {
        return;
    };
    let sid = "b2tu-full-veto";
    let veto_key = tv.key_schema().session_meta_veto(sid);

    let full = RestoreMaxmemory::set_full(&admin).await;
    assert_refuses_writes(&tv).await;
    let first = run_veto(&tv, sid).await;
    let repeat = run_veto(&tv, sid).await;
    let value: Option<String> = tv.client().pool.get(&veto_key).await.expect("GET veto");
    full.restore().await;

    assert_eq!(
        first.expect("the veto must run on a full store"),
        vec!["VETOED".to_owned()]
    );
    assert_eq!(
        repeat.expect("a repeated veto must run on a full store"),
        vec!["VETOED".to_owned()]
    );
    assert_eq!(value.as_deref(), Some("1"), "the veto is written");
}

/// The metadata write has no opener, by decision: a full store refuses it out
/// of memory and it writes nothing, so no session is created past
/// `maxmemory`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_session_meta_write_is_refused_on_a_full_store() {
    let Some((tv, admin)) = disruptive_fixture().await else {
        return;
    };
    let sid = "b2tu-full-write";

    let full = RestoreMaxmemory::set_full(&admin).await;
    assert_refuses_writes(&tv).await;
    let result = run_meta_write(&tv, sid, &[("name", "n"), ("base_path", "/p")]).await;
    let written = exists(&tv, tv.key_schema().session_meta(sid)).await;
    full.restore().await;

    let error = result.expect_err(
        "a full store must refuse the metadata write: it has no opener, so no session is created \
         past maxmemory",
    );
    assert!(
        error.details().starts_with("OOM "),
        "refused out of memory; got {error:?}"
    );
    assert!(!written, "the refused metadata write wrote nothing");
}

/// A `CreateSession` whose names commit lands and whose metadata write a full
/// store then refuses: the settle's veto lands, the release frees both index
/// members, and the call answers a plain `UNAVAILABLE` after one attempt (a
/// store write refusal is not retried in-process). The client's retry, with
/// memory still full, answers `UNAVAILABLE` (its own commit refused) — never
/// `ALREADY_EXISTS` for a session that does not exist — and once memory is
/// back the same name is created.
///
/// The store is filled from the post-commit hook, between the commit and the
/// metadata write.
///
/// Needs `test-fault-injection` (the post-commit hook) as well as the
/// disruptive store: without the feature it is compiled out.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_create_session_whose_metadata_write_is_refused_frees_the_name(
) {
    use std::sync::{Arc, Mutex};

    use futures::FutureExt;
    use live_edit::error::ServiceError;
    use live_edit::storage::session::PostCommitHook;

    const NAME: &str = "b2tu-full-create";
    const BASE: &str = "/b2tu-full-create";
    /// One deadline over each `CreateSession` on the full store. A refused
    /// write answers at once; this only keeps a hang inside nextest's 120 s
    /// hang-killer.
    const FULL_STORE_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);
    const OUT_OF_MEMORY: &str =
        "the backing store refused a write this request needed because it is out of memory; retry";

    let Some((tv, admin)) = disruptive_fixture().await else {
        return;
    };
    let keys = tv.key_schema();

    // Every attempt's id, and the guard of the `maxmemory` the first attempt's
    // hook set.
    let attempts: Arc<Mutex<Vec<String>>> = Arc::default();
    let filled: Arc<tokio::sync::Mutex<Option<RestoreMaxmemory>>> = Arc::default();
    let hook: PostCommitHook = {
        let (attempts, filled, admin) = (attempts.clone(), filled.clone(), admin.clone());
        Arc::new(move |session_id: String| {
            attempts.lock().unwrap().push(session_id);
            let (filled, admin) = (filled.clone(), admin.clone());
            async move {
                let mut filled = filled.lock().await;
                if filled.is_none() {
                    *filled = Some(RestoreMaxmemory::set_full(&admin).await);
                }
            }
            .boxed()
        })
    };
    let store = session_store(&tv).with_post_commit_hook_for_test(hook);

    let first = tokio::time::timeout(FULL_STORE_BUDGET, store.create(NAME, BASE, CREATED_BY))
        .await
        .expect("the create did not settle within its budget");
    let full = filled
        .lock()
        .await
        .take()
        .expect("the post-commit hook must have filled the store");
    assert_refuses_writes(&tv).await;
    let attempts = attempts.lock().unwrap().clone();
    let names_left = members(&tv, keys.path_names_index(BASE)).await;
    let index_left = members(&tv, keys.path_index(BASE)).await;
    let mut vetoed = Vec::new();
    let mut with_meta = Vec::new();
    for sid in &attempts {
        vetoed.push(exists(&tv, keys.session_meta_veto(sid)).await);
        with_meta.push(exists(&tv, keys.session_meta(sid)).await);
    }
    let retry_while_full = tokio::time::timeout(
        FULL_STORE_BUDGET,
        session_store(&tv).create(NAME, BASE, CREATED_BY),
    )
    .await
    .expect("the retry did not settle within its budget");
    full.restore().await;
    let retry_restored = session_store(&tv).create(NAME, BASE, CREATED_BY).await;

    match &first {
        Err(ServiceError::Unavailable(message)) => assert_eq!(
            message,
            &format!("EVALSHA session_meta_write (create): {OUT_OF_MEMORY}"),
            "the refused metadata write answers the mapper's plain UNAVAILABLE"
        ),
        other => panic!(
            "a CreateSession whose metadata write a full store refused must answer plain \
             UNAVAILABLE; got {other:?}"
        ),
    }
    assert_eq!(
        attempts.len(),
        1,
        "one attempt: a store write refusal is not retried in-process; got {attempts:?}"
    );
    match &retry_while_full {
        Err(ServiceError::Unavailable(message)) => assert_eq!(
            message,
            &format!("EXEC create_session: {OUT_OF_MEMORY}"),
            "the retry's own commit is refused out of memory"
        ),
        Err(ServiceError::AlreadyExists(message)) => panic!(
            "the retry on a full store answered ALREADY_EXISTS ({message}) for a session that \
             does not exist: the failed attempt still holds the name"
        ),
        other => panic!(
            "the retry on a full store must answer plain UNAVAILABLE (its commit refused); got \
             {other:?}"
        ),
    }
    assert_eq!(
        (vetoed, with_meta),
        (vec![true], vec![false]),
        "the settle vetoed the attempt's id, which has no metadata"
    );
    assert!(
        names_left.is_empty() && index_left.is_empty(),
        "the release must free the attempt's names and path_index members on a full store; \
         left names {names_left:?}, path_index {index_left:?}"
    );
    let created = retry_restored.expect("once memory is back the name must be free to create");
    assert_eq!(
        listed_ids(&tv, BASE).await,
        vec![created],
        "the session created once memory is back is the one session listed"
    );
}

/// A ListSessions walk retires a ghost on a full store: the veto lands, the
/// release frees both its members, and ListSessions answers OK with the live
/// session at the base path, which keeps its members.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_list_sessions_retires_a_ghost_on_a_full_store() {
    let Some((tv, admin)) = disruptive_fixture().await else {
        return;
    };
    let keys = tv.key_schema();
    let base = "/b2tu-full-walk";
    let sid = "b2tu-full-walk-ghost";
    let live = session_store(&tv)
        .create("live", base, CREATED_BY)
        .await
        .expect("create");
    seed_names_member(&tv, base, "ghost", sid).await;
    sadd(&tv, keys.path_index(base), vec![sid.into()]).await;

    let full = RestoreMaxmemory::set_full(&admin).await;
    assert_refuses_writes(&tv).await;
    let listed = listed_ids(&tv, base).await;
    let names_left = members(&tv, keys.path_names_index(base)).await;
    let index_left = members(&tv, keys.path_index(base)).await;
    let veto: Option<String> = tv
        .client()
        .pool
        .get(keys.session_meta_veto(sid))
        .await
        .expect("GET veto");
    full.restore().await;

    assert_eq!(
        listed,
        vec![live.clone()],
        "ListSessions answers OK with the live session, without the ghost"
    );
    assert_eq!(
        names_left,
        set_of(&[KeySchema::path_name_member("live", &live).as_str()]),
        "the walk must release the ghost's names member on a full store"
    );
    assert_eq!(
        index_left,
        set_of(&[live.as_str()]),
        "the walk must release the ghost's path_index member on a full store"
    );
    assert_eq!(veto.as_deref(), Some("1"), "the walk's veto landed");
}

// ═════════════════════════════════════════════════════════════════════════════
// Cluster (item 11)
// ═════════════════════════════════════════════════════════════════════════════

/// All four session scripts run on a multi-shard cluster without `CROSSSLOT`:
/// the metadata write and the veto on the `{session_id}` slot, the release and
/// the reconcile on the `{hex(base_path)}` slot, each driven through the
/// `SessionStore` call that sends it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn b_2tu_session_scripts_run_on_a_cluster() {
    let Some(url) = live_edit::storage::test_support::test_cluster_url() else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); run \
             `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster:` job run this"
        );
        return;
    };
    let tv = TestValkey::setup_with_url(url, 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment"
    );
    let keys = tv.key_schema();
    let store = session_store(&tv);
    let base = "/b2tu-cluster";

    // session_meta_write.lua, through CreateSession.
    let live = store
        .create("live", base, CREATED_BY)
        .await
        .expect("CreateSession's metadata write must run on a cluster");

    // session_meta_veto.lua answering PRESENT, through the walk's retirement.
    let live_ghost = GhostMember {
        name: "live".into(),
        sid: live.clone(),
        composite_member: KeySchema::path_name_member("live", &live),
    };
    assert_eq!(
        store
            .retire_ghost_member(base, &live_ghost)
            .await
            .expect("the veto must run on a cluster"),
        GhostRetirement::Present
    );

    // session_meta_veto.lua answering VETOED, then session_index_release.lua.
    let ghost_sid = format!("ghost-{}", uuid::Uuid::new_v4());
    let ghost_member = KeySchema::path_name_member("ghost", &ghost_sid);
    sadd(&tv, keys.path_index(base), vec![ghost_sid.clone()]).await;
    sadd(&tv, keys.path_names_index(base), vec![ghost_member.clone()]).await;
    let ghost = GhostMember {
        name: "ghost".into(),
        sid: ghost_sid.clone(),
        composite_member: ghost_member,
    };
    assert_eq!(
        store
            .retire_ghost_member(base, &ghost)
            .await
            .expect("the veto and the release must run on a cluster"),
        GhostRetirement::Retired
    );

    // session_path_index_reconcile.lua.
    sadd(&tv, keys.path_index(base), vec!["stale-sid".into()]).await;
    assert_eq!(
        store
            .reconcile_path_index(base)
            .await
            .expect("the reconcile must run on a cluster"),
        1
    );

    assert_eq!(
        members(&tv, keys.path_index(base)).await,
        set_of(&[live.as_str()])
    );
    let listed = store.list(base).await.expect("list");
    assert_eq!(listed.present.len(), 1);
    assert!(listed.ghosts.is_empty());
}
