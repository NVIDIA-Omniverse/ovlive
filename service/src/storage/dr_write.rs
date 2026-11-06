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

//! The disaster-recovery Jobs' session-slot writes (SDD §22.22, *The restore
//! writer*): the layer mappings, the layers index and the path index.
//! `storage::dr_job` stays read-only; the Jobs' object writes are in
//! `storage::layer::dr_restore`.
//!
//! # The mapping compare-and-sets
//!
//! [`unmap_layer_if`] and [`remap_layer_if`] each run on the `{session_id}`
//! slot: `acquire_txn_for(layer mapping)`, a `WATCH` of the mapping, a `GET`
//! on the watched connection, then — only when the mapping names what the
//! caller expects — one `WatchedTxn`. So a mapping is changed only from the
//! value the caller's plan saw, never from one written after it.
//!
//! - [`unmap_layer_if`] is **the DR Jobs' single unmapping function** (the
//!   service's `DeleteLayer` unmaps through its own script,
//!   `layer_unmap.lua`, with the same three outcomes): the restore
//!   Job's completion of an abandoned delete removes a session's mappings and
//!   layers-index members only through it, and accept-loss removes a lost
//!   layer's mapping, and a lost session's mappings and layers-index members,
//!   only through it, expecting the object its plan saw. A mapping that
//!   names another object ([`UnmapOutcome::Other`]) is left alone: a session
//!   backup's newer, live mapping written over a deleted one (defer-51) can
//!   never be removed by a stale plan item. Epic 3 replaces its body with the
//!   tombstoning unmap and changes nothing else.
//! - [`remap_layer_if`] moves a layer from a fenced object to its complete
//!   successor; the `from` object is left for the orphan sweep after the
//!   un-hold.
//!
//! Each call is ONE attempt: a conflict, a torn or discarded `EXEC`, or an
//! in-doubt one whose re-read cannot settle it, answers `Aborted`, and the
//! Job retries the step. A void `WATCH` ticks
//! `storage_watch_voided_total{path="dr_job"}`.
//!
//! [`remove_path_members`] removes an abandoned delete's path-index members:
//! two `SREM`s on the `{hex(base_path)}` slot, no transaction (each is
//! idempotent and alone).

use fred::interfaces::{KeysInterface, SetsInterface};
use fred::types::ClusterHash;

use crate::error::ServiceError;
use crate::observability::metrics::{
    WATCH_VOIDED_ABORTED_PRE_EXEC, WATCH_VOIDED_COMMITTED_UNWATCHED,
    WATCH_VOIDED_FINGERPRINT_UNAVAILABLE, WATCH_VOIDED_PATH_DR_JOB,
    WATCH_VOIDED_WITNESS_UNAVAILABLE,
};
use crate::storage::client::{
    map_snapshot_load_err, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA, POOL_LABEL_TXN,
};
use crate::storage::keys::{hex_encode, KeySchema};
use crate::storage::watch_guard::WatchGuard;
use crate::storage::watched_txn::{
    dispatch_verdict, warn_fingerprint_unavailable, warn_witness_unavailable, watch_with_baseline,
    DispatchVerdict, WatchFingerprint, WatchedExec, WatchedTxn,
};

/// What [`unmap_layer_if`] did — and what `DeleteLayer`'s conditional unmap
/// (`layer_unmap.lua`) did: `LayerStore::send_layer_unmap` answers its last
/// send's decoded reply, after one send (`send_layer_unmap_once`) or, for a
/// mapping spelling the object non-canonically, two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnmapOutcome {
    /// The mapping named the expected object: it and the layers-index member
    /// were removed.
    Unmapped,
    /// The mapping was absent: the layers-index member was removed.
    Absent,
    /// The mapping names this other object: nothing was written.
    Other(u64),
}

/// What [`remap_layer_if`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemapOutcome {
    /// The mapping named `from`: it now names `to`.
    Remapped,
    /// The mapping already names `to` (a re-run): nothing was written.
    AlreadyTo,
    /// The mapping names neither (`None`: it is absent): nothing was
    /// written.
    Other(Option<u64>),
}

/// The mapping's change, inside its transaction.
enum MappingChange<'a> {
    /// `DEL` the mapping and `SREM` the layers-index member.
    Unmap { layer_name: &'a str },
    /// `SREM` the layers-index member alone.
    UnindexAbsent { layer_name: &'a str },
    /// `SET` the mapping to `to`.
    Remap { to: u64 },
}

/// A mapping's value: `None` absent, `Some(id)` an object id.
type MappingValue = Option<u64>;

fn voided(client: &ValkeyClient, disposition: &'static str) {
    if let Some(metrics) = client.metrics_snapshot() {
        metrics.inc_storage_watch_voided(WATCH_VOIDED_PATH_DR_JOB, disposition);
    }
}

/// Parse a mapping's raw value: absent, or an object id. A value that is not
/// a decimal object id is a corrupt mapping, `FailedPrecondition`: not
/// retried, and not a Valkey error.
fn parse_mapping(key: &str, raw: Option<String>) -> Result<MappingValue, ServiceError> {
    match raw {
        None => Ok(None),
        Some(raw) => raw.trim().parse::<u64>().map(Some).map_err(|_| {
            ServiceError::FailedPrecondition(format!(
                "the layer mapping {key} holds {raw:?}, which is not an object id"
            ))
        }),
    }
}

/// Read a mapping on the data pool (an in-doubt `EXEC`'s re-read).
async fn read_mapping(client: &ValkeyClient, key: &str) -> Result<MappingValue, ServiceError> {
    let raw: Option<String> = client.pool.next().get(key).await.map_err(|e| {
        map_snapshot_load_err(
            FredErrOrigin::client(client, POOL_LABEL_DATA),
            "GET layer mapping (DR Job re-read)",
            e,
        )
    })?;
    parse_mapping(key, raw)
}

/// One attempt of one mapping compare-and-set: `WATCH` the mapping, read it,
/// and when `decide` returns a change, apply it in a `WatchedTxn`. Returns
/// the value read and whether the change committed; a conflict, a torn or
/// discarded `EXEC` and an in-doubt one are `Aborted` (the caller re-reads
/// an in-doubt one).
async fn mapping_cas<'c>(
    client: &ValkeyClient,
    keys: &KeySchema,
    session_id: &str,
    mapping_key: &str,
    decide: impl FnOnce(MappingValue) -> Option<MappingChange<'c>>,
) -> Result<(MappingValue, CasResult), ServiceError> {
    let txn_origin = || FredErrOrigin::client(client, POOL_LABEL_TXN);
    let checkout = client.acquire_txn_for(mapping_key).await;
    let mut conn = WatchGuard::new(checkout, client.watch_unwatch_tracker())
        .with_metrics(client.metrics_snapshot());
    let watch_hash = ClusterHash::from(mapping_key);
    let baseline = watch_with_baseline(&conn, &watch_hash, &[mapping_key])
        .await
        .map_err(|e| map_snapshot_load_err(txn_origin(), "WATCH layer mapping (DR Job)", e))?;
    conn.mark_watched(mapping_key, &[]);
    if let Some(reason) = baseline.unavailable_reason() {
        voided(client, WATCH_VOIDED_WITNESS_UNAVAILABLE);
        warn_witness_unavailable(WATCH_VOIDED_PATH_DR_JOB, reason);
    }
    let fingerprint = WatchFingerprint::take(&conn, &watch_hash);
    let raw: Option<String> = match conn.get(mapping_key).await {
        Ok(raw) => raw,
        Err(e) => {
            let _ = conn.unwatch_pinned().await;
            return Err(map_snapshot_load_err(
                txn_origin(),
                "GET layer mapping (DR Job)",
                e,
            ));
        }
    };
    let current = match parse_mapping(mapping_key, raw) {
        Ok(current) => current,
        Err(e) => {
            let _ = conn.unwatch_pinned().await;
            return Err(e);
        }
    };
    let Some(change) = decide(current) else {
        let _ = conn.unwatch_pinned().await;
        return Ok((current, CasResult::NothingToDo));
    };

    let txn = match WatchedTxn::open(&conn, &watch_hash, &baseline).await {
        Ok(txn) => txn,
        Err(e) => {
            let _ = conn.unwatch_pinned().await;
            return Err(map_snapshot_load_err(
                txn_origin(),
                "MULTI queue (DR Job mapping)",
                e,
            ));
        }
    };
    let queued: Result<(), fred::error::Error> = async {
        match change {
            MappingChange::Unmap { layer_name } => {
                let _: () = txn.commands().del(mapping_key).await?;
                let _: () = txn
                    .commands()
                    .srem(keys.layers_index(session_id), hex_encode(layer_name))
                    .await?;
            }
            MappingChange::UnindexAbsent { layer_name } => {
                let _: () = txn
                    .commands()
                    .srem(keys.layers_index(session_id), hex_encode(layer_name))
                    .await?;
            }
            MappingChange::Remap { to } => {
                let _: () = txn
                    .commands()
                    .set(mapping_key, to.to_string(), None, None, false)
                    .await?;
            }
        }
        Ok(())
    }
    .await;
    if let Err(e) = queued {
        let _ = conn.unwatch_pinned().await;
        return Err(map_snapshot_load_err(
            txn_origin(),
            "mapping write queue (DR Job)",
            e,
        ));
    }

    match dispatch_verdict(&conn, &fingerprint, &watch_hash) {
        DispatchVerdict::Dispatch => {}
        DispatchVerdict::DispatchUncomparable => {
            voided(client, WATCH_VOIDED_FINGERPRINT_UNAVAILABLE);
            warn_fingerprint_unavailable(WATCH_VOIDED_PATH_DR_JOB);
        }
        DispatchVerdict::Refuse => {
            let _ = conn.unwatch_pinned().await;
            voided(client, WATCH_VOIDED_ABORTED_PRE_EXEC);
            return Err(ServiceError::Aborted(format!(
                "the DR Job's write of {mapping_key} was aborted before EXEC: the WATCH \
                 connection was replaced; nothing was written"
            )));
        }
    }

    match txn.exec(txn_origin()).await {
        WatchedExec::Committed { witness } => {
            conn.mark_unwatched();
            if baseline.contradicted_by(witness) {
                voided(client, WATCH_VOIDED_COMMITTED_UNWATCHED);
                tracing::warn!(
                    key = %mapping_key,
                    "the DR Job's write of {mapping_key} ran on a connection that never held its \
                     WATCH on the mapping"
                );
            }
            Ok((current, CasResult::Committed))
        }
        WatchedExec::Conflict => {
            conn.mark_unwatched();
            Err(ServiceError::Aborted(format!(
                "the DR Job's write of {mapping_key} lost its WATCH; nothing was written"
            )))
        }
        WatchedExec::ExecError(e) | WatchedExec::Discarded { exec: e, .. } => {
            Err(ServiceError::Aborted(format!(
                "the DR Job's write of {mapping_key} returned an error frame: {e}"
            )))
        }
        WatchedExec::InDoubt(e) => Ok((current, CasResult::InDoubt(e.to_string()))),
    }
}

/// How a [`mapping_cas`] attempt ended.
enum CasResult {
    /// `decide` asked for no change.
    NothingToDo,
    /// The change committed.
    Committed,
    /// The `EXEC` reply was lost.
    InDoubt(String),
}

/// Remove session `session_id`'s mapping of `layer_name` when it names
/// `expected` (module docs): with `Some(id)`, a mapping naming `id` is
/// deleted and its layers-index member removed ([`UnmapOutcome::Unmapped`]);
/// an absent mapping has its member removed ([`UnmapOutcome::Absent`]); a
/// mapping naming another object is left alone ([`UnmapOutcome::Other`]).
/// With `None`, only an absent mapping's member is removed. An in-doubt
/// `EXEC` is re-read: an absent mapping then counts as done. One attempt;
/// `Aborted` asks the caller to retry.
pub async fn unmap_layer_if(
    client: &ValkeyClient,
    keys: &KeySchema,
    session_id: &str,
    layer_name: &str,
    expected: Option<u64>,
) -> Result<UnmapOutcome, ServiceError> {
    let mapping_key = keys.layer_mapping(session_id, layer_name);
    let (current, result) =
        mapping_cas(
            client,
            keys,
            session_id,
            &mapping_key,
            |current| match current {
                Some(id) if Some(id) == expected => Some(MappingChange::Unmap { layer_name }),
                None => Some(MappingChange::UnindexAbsent { layer_name }),
                Some(_) => None,
            },
        )
        .await?;
    let done = match current {
        Some(id) if Some(id) == expected => UnmapOutcome::Unmapped,
        None => UnmapOutcome::Absent,
        Some(other) => return Ok(UnmapOutcome::Other(other)),
    };
    match result {
        CasResult::NothingToDo | CasResult::Committed => Ok(done),
        CasResult::InDoubt(e) => match read_mapping(client, &mapping_key).await? {
            None => Ok(done),
            Some(_) => Err(ServiceError::Aborted(format!(
                "the DR Job's unmapping of {mapping_key} was left in doubt and the mapping is \
                 still there: {e}"
            ))),
        },
    }
}

/// Map session `session_id`'s `layer_name` to `to` when it names `from`
/// (module docs). One attempt; an in-doubt `EXEC` is re-read; `Aborted`
/// asks the caller to retry.
pub async fn remap_layer_if(
    client: &ValkeyClient,
    keys: &KeySchema,
    session_id: &str,
    layer_name: &str,
    from: u64,
    to: u64,
) -> Result<RemapOutcome, ServiceError> {
    let mapping_key = keys.layer_mapping(session_id, layer_name);
    let (current, result) = mapping_cas(client, keys, session_id, &mapping_key, |current| {
        (current == Some(from)).then_some(MappingChange::Remap { to })
    })
    .await?;
    if current == Some(to) {
        return Ok(RemapOutcome::AlreadyTo);
    }
    if current != Some(from) {
        return Ok(RemapOutcome::Other(current));
    }
    match result {
        CasResult::NothingToDo | CasResult::Committed => Ok(RemapOutcome::Remapped),
        CasResult::InDoubt(e) => match read_mapping(client, &mapping_key).await? {
            Some(id) if id == to => Ok(RemapOutcome::Remapped),
            _ => Err(ServiceError::Aborted(format!(
                "the DR Job's remapping of {mapping_key} was left in doubt and the mapping does \
                 not name {to}: {e}"
            ))),
        },
    }
}

/// Remove an abandoned delete's path-index members under `base_path`:
/// `session_member` from `le:path:{hex(bp)}` and `name_members` from its
/// `:names` set. Two `SREM`s on the data pool, each idempotent.
pub async fn remove_path_members(
    client: &ValkeyClient,
    keys: &KeySchema,
    base_path: &str,
    session_member: &str,
    name_members: &[String],
) -> Result<(), ServiceError> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_DATA);
    let _: u64 = client
        .pool
        .next()
        .srem(keys.path_index(base_path), session_member)
        .await
        .map_err(|e| map_snapshot_load_err(origin, "SREM path index (DR Job)", e))?;
    if !name_members.is_empty() {
        let _: u64 = client
            .pool
            .next()
            .srem(keys.path_names_index(base_path), name_members.to_vec())
            .await
            .map_err(|e| map_snapshot_load_err(origin, "SREM path names index (DR Job)", e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Every answer of the mapping compare-and-sets against a real store,
    //! with what each leaves of the mapping and its layers-index member: the
    //! contract accept-loss relies on and Epic 3's body replacement must keep.
    //! The in-doubt re-read needs a lost `EXEC` reply, which no store here
    //! gives; it is not covered.

    use std::collections::BTreeSet;

    use super::*;
    use crate::storage::test_support::TestValkey;

    const SESSION: &str = "s-dr-write";
    const LAYER: &str = "l.usda";

    async fn set_mapping(tv: &TestValkey, value: &str) {
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().layer_mapping(SESSION, LAYER),
                value,
                None,
                None,
                false,
            )
            .await
            .expect("SET mapping");
    }

    async fn index_member(tv: &TestValkey) {
        let _: i64 = tv
            .client()
            .pool
            .sadd(tv.key_schema().layers_index(SESSION), hex_encode(LAYER))
            .await
            .expect("SADD layers index");
    }

    /// `(the mapping's value, whether the layer's layers-index member is
    /// present)`.
    async fn mapping_and_member(tv: &TestValkey) -> (Option<String>, bool) {
        let mapping: Option<String> = tv
            .client()
            .pool
            .get(tv.key_schema().layer_mapping(SESSION, LAYER))
            .await
            .expect("GET mapping");
        let member: bool = tv
            .client()
            .pool
            .sismember(tv.key_schema().layers_index(SESSION), hex_encode(LAYER))
            .await
            .expect("SISMEMBER");
        (mapping, member)
    }

    async fn unmap(tv: &TestValkey, expected: Option<u64>) -> Result<UnmapOutcome, ServiceError> {
        unmap_layer_if(tv.client(), tv.key_schema(), SESSION, LAYER, expected).await
    }

    async fn remap(tv: &TestValkey, from: u64, to: u64) -> Result<RemapOutcome, ServiceError> {
        remap_layer_if(tv.client(), tv.key_schema(), SESSION, LAYER, from, to).await
    }

    /// `unmap_layer_if`: a mapping naming the expected object is deleted with
    /// its member (`Unmapped`); an absent mapping has its member removed
    /// (`Absent`), whatever was expected; a mapping naming another object is
    /// left with its member (`Other`), whether another object or none was
    /// expected.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unmap_layer_if_removes_only_a_mapping_naming_the_expected_object() {
        let tv = TestValkey::setup().await;

        set_mapping(&tv, "7").await;
        index_member(&tv).await;
        assert_eq!(unmap(&tv, Some(7)).await.unwrap(), UnmapOutcome::Unmapped);
        assert_eq!(mapping_and_member(&tv).await, (None, false));

        for expected in [Some(7), None] {
            index_member(&tv).await;
            assert_eq!(
                unmap(&tv, expected).await.unwrap(),
                UnmapOutcome::Absent,
                "expected {expected:?}"
            );
            assert_eq!(
                mapping_and_member(&tv).await,
                (None, false),
                "expected {expected:?}: the member of an absent mapping is removed"
            );
        }

        set_mapping(&tv, "8").await;
        index_member(&tv).await;
        for expected in [Some(7), None] {
            assert_eq!(
                unmap(&tv, expected).await.unwrap(),
                UnmapOutcome::Other(8),
                "expected {expected:?}"
            );
            assert_eq!(
                mapping_and_member(&tv).await,
                (Some("8".to_owned()), true),
                "expected {expected:?}: a mapping to another object stays, with its member"
            );
        }
    }

    /// `remap_layer_if`: a mapping naming `from` moves to `to` (`Remapped`),
    /// and asked again answers `AlreadyTo`; an absent mapping and one naming
    /// a third object answer `Other` and are left as they are. The
    /// layers-index member is never touched.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remap_layer_if_moves_only_a_mapping_naming_from() {
        let tv = TestValkey::setup().await;
        index_member(&tv).await;

        assert_eq!(remap(&tv, 7, 9).await.unwrap(), RemapOutcome::Other(None));
        assert_eq!(mapping_and_member(&tv).await, (None, true));

        set_mapping(&tv, "7").await;
        assert_eq!(remap(&tv, 7, 9).await.unwrap(), RemapOutcome::Remapped);
        assert_eq!(mapping_and_member(&tv).await, (Some("9".to_owned()), true));
        assert_eq!(remap(&tv, 7, 9).await.unwrap(), RemapOutcome::AlreadyTo);
        assert_eq!(mapping_and_member(&tv).await, (Some("9".to_owned()), true));

        set_mapping(&tv, "8").await;
        assert_eq!(
            remap(&tv, 7, 9).await.unwrap(),
            RemapOutcome::Other(Some(8))
        );
        assert_eq!(mapping_and_member(&tv).await, (Some("8".to_owned()), true));
    }

    /// A mapping that holds no object id is a corrupt mapping: both
    /// functions answer `FailedPrecondition` (not retried) and write nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_corrupt_mapping_is_a_failed_precondition_and_nothing_is_written() {
        let tv = TestValkey::setup().await;
        set_mapping(&tv, "not-an-id").await;
        index_member(&tv).await;

        for (call, result) in [
            ("unmap Some", unmap(&tv, Some(7)).await.map(|_| ())),
            ("unmap None", unmap(&tv, None).await.map(|_| ())),
            ("remap", remap(&tv, 7, 9).await.map(|_| ())),
        ] {
            assert!(
                matches!(result, Err(ServiceError::FailedPrecondition(_))),
                "{call}: {result:?}"
            );
            assert_eq!(
                mapping_and_member(&tv).await,
                (Some("not-an-id".to_owned()), true),
                "{call}: nothing written"
            );
        }
    }

    /// `remove_path_members` removes the session's member of the path index
    /// and the names it is given from the names index, and nothing else.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remove_path_members_removes_only_the_named_members() {
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let base_path = "omniverse://example.com/p.usd";
        let pool = &tv.client().pool;
        let _: i64 = pool
            .sadd(keys.path_index(base_path), vec![SESSION, "s-other"])
            .await
            .expect("SADD");
        let _: i64 = pool
            .sadd(
                keys.path_names_index(base_path),
                vec!["n\0s-dr-write", "n\0s-other"],
            )
            .await
            .expect("SADD");

        remove_path_members(
            tv.client(),
            keys,
            base_path,
            SESSION,
            &["n\0s-dr-write".to_owned()],
        )
        .await
        .expect("the SREMs answer");

        let path: BTreeSet<String> = pool
            .smembers(keys.path_index(base_path))
            .await
            .expect("SMEMBERS");
        let names: BTreeSet<String> = pool
            .smembers(keys.path_names_index(base_path))
            .await
            .expect("SMEMBERS");
        assert_eq!(path, BTreeSet::from(["s-other".to_owned()]));
        assert_eq!(names, BTreeSet::from(["n\0s-other".to_owned()]));
    }
}
