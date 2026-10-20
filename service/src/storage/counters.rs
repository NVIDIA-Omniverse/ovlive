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

// Atomic counter operations: object ID (le:counter:object_id).
//
// External values were previously allocated a `value_id` from
// `le:counter:ext_value_id`; they are now content-addressed (data key is
// derived from the value's content hash) and that counter is no longer used
// (SDD §11.1).

use std::sync::LazyLock;

use async_trait::async_trait;
use fred::interfaces::KeysInterface;
use fred::types::scripts::Script;
use fred::types::SetOptions;

use crate::error::ServiceError;
use crate::observability::metrics::{SharedMetrics, STORE_WRITE_REFUSED_STAGE_CREATE_COMMIT};
use crate::storage::client::{
    map_fred_err_for_grpc, refuse_store_write, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA,
    POOL_LABEL_TXN,
};
use crate::storage::keys::KeySchema;

/// Compiled Lua script for the conditional object-id counter seed used at
/// startup by the AOF-off counters-sidecar recovery path (SDD SR-5.3,
/// Epic `t1.fgj.mh` / Task `t2.fgj.mh.4u`). Atomically performs
/// `if current < seed { SET seed }` on `le:counter:object_id` and returns
/// the post-script value as a string.
///
/// Kept inline (Rust string literal, not `include_str!` of a sibling
/// `.lua`) per Epic `t1.fgj.mh` notes — this is a one-off startup primitive
/// and the project convention for sidecar `.lua` files is reserved for
/// hot-path scripts. `Script::from_lua` is infallible (just hashes the
/// source); the first `evalsha_with_reload` call sends `SCRIPT LOAD` to
/// the server on `NOSCRIPT` before retrying.
///
/// The script body's conditional-set fragment is byte-for-byte from
/// SR-5.3; the trailing `return redis.call('GET', KEYS[1])` is the
/// load-bearing extension that gives Task 4 the post-script value to
/// log as `counter seeded to {value}`.
static SEED_OBJECT_ID_IF_BELOW_SCRIPT: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(
        "if tonumber(redis.call('GET', KEYS[1]) or '0') < tonumber(ARGV[1]) then\n  \
           redis.call('SET', KEYS[1], ARGV[1])\n\
         end\n\
         return redis.call('GET', KEYS[1])",
    )
});

/// Compiled Lua script for fail-closed object-id allocation (b.b2j).
///
/// `EXISTS` + `INCR` in one atomic block: returns a **nil reply** when
/// `le:counter:object_id` is absent, otherwise the newly allocated id. See
/// `scripts/alloc_object_id.lua` for why allocation must refuse rather than
/// INCR-from-absent, and why the absent verdict is nil rather than an in-band
/// numeric sentinel.
///
/// Unlike the sibling [`SEED_OBJECT_ID_IF_BELOW_SCRIPT`] this one lives in a
/// `scripts/*.lua` sidecar: it is on the `CreateLayer` request path, which is
/// the project convention's criterion for a sidecar file rather than an inline
/// literal.
static ALLOC_OBJECT_ID_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/alloc_object_id.lua")));

/// Repair hook for a lost `le:counter:object_id`.
///
/// Declared here — in `storage`, next to the allocation it un-blocks — so
/// [`crate::storage::layer::LayerStore`]'s allocation path names an abstract
/// seam rather than a concrete checkpoint type: the store stays constructible
/// (and the repair branch drivable) from a test that stands up no checkpoint
/// adapter at all, and `main.rs` chooses the implementation at wiring time.
/// This is dependency inversion for testability, not a module-layering
/// constraint — `storage` already depends on `checkpoint` elsewhere
/// (`layer.rs` uses [`crate::checkpoint::SessionCheckpointWriter`], and
/// `session.rs` / `external_values.rs` do the same). New cross-module edges
/// should nonetheless prefer this inversion over a fresh concrete import.
/// Mirrors how [`crate::storage::external_values::ExternalValueStore`] is
/// declared in `storage` and consumed elsewhere.
///
/// The production implementation
/// (`crate::checkpoint::counter_seed::CheckpointObjectIdCounterRepair`)
/// re-seeds the counter from the checkpoint-store counters sidecar plus
/// `LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN`, which is the only source that
/// can prove a floor above every object id a surviving checkpoint blob claims.
///
/// # Errors
///
/// Implementations return [`ServiceError::Unavailable`] when the repair could
/// not be completed — the counter is a repairable dependency state, so the
/// caller's `CreateLayer` should be retriable rather than fatal.
#[async_trait]
pub trait ObjectIdCounterRepair: Send + Sync {
    /// Re-establish `le:counter:object_id` at a value above every object id
    /// any durable checkpoint could still claim. Idempotent and safe to call
    /// concurrently; implementations are expected to singleflight internally.
    async fn repair_object_id_counter(&self) -> Result<(), ServiceError>;
}

/// Total attempts for the three `txn_pool` counter primitives (b.bhb).
///
/// The `txn_pool` runs at `max_command_attempts = 1` so nothing issued under a
/// `WATCH` can be replayed onto a connection that never held it (see
/// `ValkeyClient::connect` and `storage::watched_txn`). These three primitives
/// hold no `WATCH` — they ride that pool only because they are startup /
/// maintenance-scope writes that must not queue behind the data plane — but
/// they lose fred's transparent retry along with everything else on it, and
/// their callers cannot absorb that: the counter-seed routine gates
/// `counter_seed_ready` (and therefore readiness) on a single call, and
/// `create_layer`'s re-seed path retries the *allocation* exactly once.
/// A reconnect that used to be invisible would become a failed boot.
///
/// Three attempts with a short fixed delay is calibrated to the failure it
/// covers — one reconnect landing on the connection mid-command, which fred
/// re-establishes in `RECONNECT_MIN_DELAY_MS` (100 ms) — not to a store outage,
/// which the callers' own loops and the next maintenance tick already own.
const TXN_COUNTER_MAX_ATTEMPTS: u32 = 3;

/// Delay between [`TXN_COUNTER_MAX_ATTEMPTS`] attempts. Fixed rather than
/// jittered: these calls are single-flight startup / maintenance work with no
/// herd to disperse.
const TXN_COUNTER_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(50);

/// Run a `txn_pool` counter primitive, retrying its transient failures.
///
/// Only [`ServiceError::Unavailable`] is retried — that is exactly what
/// [`map_fred_err_for_grpc`] produces for `is_transient_fred_error` kinds
/// (IO / Timeout / Canceled / Cluster / Backpressure / router misses). A
/// corrupt-counter `Internal` is not retriable and must surface immediately.
/// A store write refusal also answers a plain [`ServiceError::Unavailable`],
/// but must not be retried in-process (`storage::client::store_write_refusal`),
/// so the two writing primitives return it as a settled `Ok(Err(_))` that this
/// loop passes straight through.
async fn retry_transient<F, Fut, T>(op: &'static str, mut attempt: F) -> Result<T, ServiceError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, ServiceError>>,
{
    for n in 1..TXN_COUNTER_MAX_ATTEMPTS {
        match attempt().await {
            Err(ServiceError::Unavailable(msg)) => {
                tracing::debug!(
                    op,
                    attempt = n,
                    max_attempts = TXN_COUNTER_MAX_ATTEMPTS,
                    error = %msg,
                    "transient failure on a txn-pool counter primitive, retrying"
                );
                tokio::time::sleep(TXN_COUNTER_RETRY_DELAY).await;
            }
            settled => return settled,
        }
    }
    // Last attempt: its result is the answer whatever it is, so there is no
    // "budget exhausted" branch to invent an error for.
    attempt().await
}

/// Atomic counter operations against Valkey.
#[derive(Clone)]
pub struct CounterStore {
    client: ValkeyClient,
    keys: KeySchema,
    /// Counts a store write refusal of the object-id allocation, or of either
    /// counter repair write ([`CounterStore::initialize_object_id_counter_if_absent`],
    /// [`CounterStore::seed_object_id_if_below`]), on
    /// `live_edit_store_write_refused_total{stage="create_commit"}` when
    /// wired; `None` counts nothing and changes no answer.
    metrics: Option<SharedMetrics>,
}

impl CounterStore {
    pub fn new(client: ValkeyClient, keys: KeySchema) -> Self {
        Self {
            client,
            keys,
            metrics: None,
        }
    }

    /// Attach the metrics handle a store write refusal is counted on, at
    /// `stage="create_commit"`: by [`Self::next_object_id`] and by the two
    /// counter repair writes, which run only for a `CreateLayer` that found
    /// the counter absent or at startup. `LayerStore::with_metrics` wires it
    /// for the `CreateLayer` allocation and its unhooked repair, and `main.rs`
    /// for the checkpoint repair hook's store; the startup seed's stores are
    /// left unwired, so a refusal there answers the same error but counts
    /// nothing.
    pub fn with_metrics(mut self, metrics: SharedMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// The answer to a store write refusal (`-OOM`, `-MISCONF`, `-READONLY`,
    /// `-NOREPLICAS`) of one of this store's object-id counter writes, counted
    /// at `stage="create_commit"` when [`Self::with_metrics`] wired a handle;
    /// `None` for any other error, which the caller maps as before. Each of
    /// those writes is one command or one script whose only write is its
    /// first, and Valkey refuses before that write runs, so a refused write
    /// applied nothing; the answer is still the stage's plain `UNAVAILABLE`,
    /// with no retry hint (`storage::client::store_write_refusal`). An `-OOM`
    /// is noted on the client's store out-of-memory signal whether or not a
    /// metrics handle is wired (`storage::store_oom`).
    fn refused_write(&self, e: &fred::error::Error) -> Option<ServiceError> {
        refuse_store_write(
            FredErrOrigin::metrics(
                self.metrics.as_ref(),
                POOL_LABEL_DATA,
                self.client.store_oom_signal(),
            ),
            STORE_WRITE_REFUSED_STAGE_CREATE_COMMIT,
            e,
        )
    }

    /// Allocate a new object ID, **failing closed** when the counter key is
    /// absent (b.b2j).
    ///
    /// Returns `Ok(None)` when the script answered with a nil reply, which it
    /// does exactly when `le:counter:object_id` does not exist. That is
    /// not a normal state: [`Self::initialize_object_id_counter_if_absent`]
    /// establishes the key at startup, and nothing in the service ever deletes
    /// it — so absence means the Valkey keyspace was lost underneath a
    /// surviving process. A bare `INCR` would silently return `1` there and
    /// re-mint object ids that pre-loss checkpoint blobs still claim, which
    /// clobbers live state (`write_object` HSETs the header unconditionally),
    /// inverts retention pruning, and can materialize one layer's content
    /// under another layer's identity on the DR path. The caller
    /// (`LayerStore::create_layer`) repairs the counter via
    /// [`ObjectIdCounterRepair`] and retries.
    ///
    /// Atomic server-side via a single-key Lua EVAL — `EXISTS` and `INCR`
    /// split across two round-trips would let a concurrent allocator observe
    /// the key, lose the keyspace, and INCR-from-absent anyway. The key
    /// carries no hash tag, so this is single-slot on Valkey Cluster (the same
    /// routing argument [`Self::seed_object_id_if_below`] makes).
    ///
    /// Called single-shot from `LayerStore::create_layer`, which surfaces
    /// directly to the CreateLayer gRPC handler — so fred errors route
    /// through [`map_fred_err_for_grpc`] (transient → `Unavailable`,
    /// permanent → `Internal`) to match the workspace-wide mapper-routing
    /// contract (b.u96), after one carve-out: a store write refusal (`-OOM`,
    /// `-MISCONF`, `-READONLY`, `-NOREPLICAS`) refused the script's `INCR`,
    /// so no id was allocated — plain `Unavailable` with no retry hint (no
    /// `CreateLayer` commit-stage refusal carries one), counted at
    /// `stage="create_commit"`
    /// (`storage::client::store_write_refusal`).
    /// Dispatches on the data pool (`pool.next()`), matching
    /// the pre-b.b2j `INCR` and the `update_layer` fast path — this is a
    /// request-path allocation, not a maintenance write.
    /// The reply is decoded as `Option<i64>` — not `i64` — because the absent
    /// verdict travels as a nil reply. The three-way match below is the whole
    /// contract: nil is the verdict, a positive integer is an allocation, and a
    /// non-positive integer is a corrupt counter (`INCR` ran against a value
    /// this service never writes) and must NOT be laundered into either of the
    /// other two readings — `Internal` says "the data is wrong", which is a
    /// different operator response from `Unavailable`'s "retry me".
    pub async fn next_object_id(&self) -> Result<Option<u64>, ServiceError> {
        let keys = vec![self.keys.counter_object_id()];
        let argv: Vec<String> = Vec::new();

        // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
        // connection can run this twice: the `INCR` skips an object id, which is harmless.
        let raw: Option<i64> = ALLOC_OBJECT_ID_SCRIPT
            .evalsha_with_reload::<Option<i64>, _, _>(self.client.pool.next(), keys, argv)
            .await
            .map_err(|e| {
                self.refused_write(&e).unwrap_or_else(|| {
                    map_fred_err_for_grpc(
                        FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                        "EVAL alloc_object_id",
                        e,
                    )
                })
            })?;

        match raw {
            None => Ok(None),
            Some(n) => match u64::try_from(n) {
                Ok(id) if id > 0 => Ok(Some(id)),
                _ => Err(ServiceError::Internal(format!(
                    "alloc_object_id: INCR returned a non-positive id (corrupt counter): {n}"
                ))),
            },
        }
    }

    /// Establish `le:counter:object_id` at `0` iff it does not already exist
    /// (`SET <key> 0 NX`). Returns `true` when this call created the key.
    ///
    /// Complements [`Self::next_object_id`]'s fail-closed contract: allocation
    /// can only treat absence as "keyspace lost" if the key is otherwise
    /// guaranteed to exist, which this call establishes on first boot (from
    /// the startup counter-seed routine's `Absent` arm) and re-establishes
    /// after a repair.
    ///
    /// `NX` makes it safe to race: a peer replica that seeded a real value
    /// between our decision and this write keeps its value, never getting
    /// stomped back to `0`.
    ///
    /// Dispatches through `txn_pool` — startup / repair scope, not the hot
    /// path, matching [`Self::seed_object_id_if_below`].
    ///
    /// A store write refusal of the `SET NX` answers the plain
    /// `create_commit` `Unavailable` (`Self::refused_write`), is
    /// counted at `stage="create_commit"` when wired, and is not retried: a
    /// refused `SET` ran no part of itself.
    pub async fn initialize_object_id_counter_if_absent(&self) -> Result<bool, ServiceError> {
        // b.bhb: bounded transient retry — the `txn_pool` no longer re-issues a
        // command fred could not write. See `retry_transient`. A store write
        // refusal settles the loop as `Ok(Err(_))`, so it is not retried.
        retry_transient("SET NX object_id counter", || async {
            // b.7td: pinned to the counter key's own slot so the checkout
            // hygiene probe reaches the node this write runs on. Re-acquired
            // per attempt so a retry gets a freshly-probed slot rather than the
            // one that just failed.
            let txn_client = self
                .client
                .acquire_txn_for(&self.keys.counter_object_id())
                .await;
            // `Option<String>`, not `bool`: Valkey answers `SET ... NX` with
            // `+OK` on write and a null reply when the key already existed, so
            // presence of a reply IS the verdict. Routing a null through a bool
            // conversion would depend on client-side truthiness semantics rather
            // than the protocol.
            let reply: Option<String> = match txn_client
                .set(
                    self.keys.counter_object_id(),
                    "0",
                    None,
                    Some(SetOptions::NX),
                    false,
                )
                .await
            {
                Ok(reply) => reply,
                Err(e) => {
                    if let Some(refused) = self.refused_write(&e) {
                        return Ok(Err(refused));
                    }
                    return Err(map_fred_err_for_grpc(
                        FredErrOrigin::client(&self.client, POOL_LABEL_TXN),
                        "SET NX object_id counter",
                        e,
                    ));
                }
            };
            Ok(Ok(reply.is_some()))
        })
        .await?
    }

    /// Read the current value of `le:counter:object_id` without mutating it.
    /// Returns `0` when the key is absent (pre-first-CreateLayer state),
    /// matching the SDD SR-5.1 contract for the periodic counters-sync tick.
    ///
    /// Dispatches through the `txn_pool` — this is a maintenance / startup-
    /// scope read, not a data-plane hot path, and matches the pool choice
    /// of the sibling [`Self::seed_object_id_if_below`] startup write.
    pub async fn current_object_id(&self) -> Result<u64, ServiceError> {
        // b.bhb: bounded transient retry — see `retry_transient`.
        retry_transient("GET object_id counter", || async {
            // b.7td: pinned to the counter key's own slot — see the sibling
            // above. Re-acquired per attempt so a retry gets a freshly-probed
            // slot rather than the one that just failed.
            let txn_client = self
                .client
                .acquire_txn_for(&self.keys.counter_object_id())
                .await;
            // Single-shot read from the counters-sync periodic tick; fred
            // errors route through `map_fred_err_for_grpc` (transient →
            // `Unavailable`, permanent → `Internal`) per the workspace-wide
            // mapper-routing contract (b.u96). The non-numeric parse arm
            // below is a legitimate corrupt-data `Internal` — the fred call
            // succeeded but the reply bytes failed validation.
            let raw: Option<String> = txn_client
                .get(self.keys.counter_object_id())
                .await
                .map_err(|e| {
                    map_fred_err_for_grpc(
                        FredErrOrigin::client(&self.client, POOL_LABEL_TXN),
                        "GET object_id counter",
                        e,
                    )
                })?;
            match raw {
                None => Ok(0),
                Some(s) => s.parse::<u64>().map_err(|_| {
                    ServiceError::Internal(format!(
                        "GET object_id counter: non-numeric reply (corrupt counter): {s}"
                    ))
                }),
            }
        })
        .await
    }

    /// Conditionally advance `le:counter:object_id` to `seed` iff its current
    /// value is below `seed`. Returns the post-script counter value (the new
    /// seed if it advanced; the existing value otherwise). Atomic server-side
    /// via a single Lua EVAL — safe against concurrent `next_object_id()` from
    /// other replicas. SDD SR-5.3.
    ///
    /// Dispatches through `txn_pool` because this is a startup-time mutation —
    /// the data-plane `pool` is reserved for the hot path, and the bare
    /// `le:counter:object_id` key carries no hash tag so single-slot placement
    /// on Valkey Cluster is guaranteed (no cross-slot concerns either way).
    ///
    /// A store write refusal answers the plain `create_commit` `Unavailable`
    /// (`Self::refused_write`), is counted at `stage="create_commit"` when
    /// wired, and is not retried. The script's only write is its one `SET`,
    /// and Valkey refuses a script's write before it runs, so a refused run
    /// changed nothing.
    pub async fn seed_object_id_if_below(&self, seed: u64) -> Result<u64, ServiceError> {
        // b.bhb: bounded transient retry — see `retry_transient`. Safe to
        // repeat: the script is a conditional `SET` (`if current < seed`), so a
        // second run against an already-seeded counter is a no-op. A store
        // write refusal settles the loop as `Ok(Err(_))`, so it is not retried.
        retry_transient("EVAL seed_object_id_if_below", || async {
            // Acquire an exclusive txn-pool slot. The returned checkout holds
            // the `valkey_pool_in_use{pool="txn"}` gauge increment and the
            // checkout permit for the duration of the EVAL, releasing them in
            // that order behind the mutex when it drops.
            // b.7td: pinned to the counter key's own slot — the EVAL below runs
            // on exactly that slot. Re-acquired per attempt so a retry gets a
            // freshly-probed slot rather than the one that just failed.
            let txn_client = self
                .client
                .acquire_txn_for(&self.keys.counter_object_id())
                .await;
            let keys = vec![self.keys.counter_object_id()];
            let argv = vec![seed.to_string()];

            // Single-shot EVAL from the counter-seed routine (startup or
            // repair); after the store write refusal carve-out, fred errors
            // route through `map_fred_err_for_grpc` per the workspace-wide
            // mapper-routing contract (b.u96). The non-numeric
            // parse arm below is a legitimate corrupt-data `Internal` — the
            // EVAL succeeded but the post-script GET reply failed validation.
            let raw: String = match SEED_OBJECT_ID_IF_BELOW_SCRIPT
                .evalsha_with_reload::<String, _, _>(&txn_client, keys, argv)
                .await
            {
                Ok(raw) => raw,
                Err(e) => {
                    if let Some(refused) = self.refused_write(&e) {
                        return Ok(Err(refused));
                    }
                    return Err(map_fred_err_for_grpc(
                        FredErrOrigin::client(&self.client, POOL_LABEL_TXN),
                        "EVAL seed_object_id_if_below",
                        e,
                    ));
                }
            };

            Ok(raw.parse::<u64>().map_err(|_| {
                ServiceError::Internal(format!(
                    "seed_object_id_if_below: non-numeric reply (corrupt counter): {raw}"
                ))
            }))
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    //! In-`src` tests for `CounterStore`'s three counter primitives —
    //! `seed_object_id_if_below`, `next_object_id`, and
    //! `initialize_object_id_counter_if_absent`. Use `TestValkey::setup()`
    //! for namespace-isolated single-node fixtures (mirrors the
    //! `#[cfg(test)]` blocks in `src/checkpoint/apply.rs` and
    //! `src/checkpoint/recovery.rs`). The cluster variants gate on
    //! `LIVE_EDIT_TEST_CLUSTER_URL` and skip cleanly when unset, matching
    //! the project's `tests/integration/storage/*` skip pattern.
    //!
    //! The `seed_object_id_if_below` cases pin the SR-5.3 contract that Task
    //! 4 (startup seed routine) relies on for safety: the script's `<`
    //! boundary condition must be a strict less-than so re-running the
    //! routine against the same checkpoint is a no-op rather than a counter
    //! rewrite.
    //!
    //! The `next_object_id` / `initialize_object_id_counter_if_absent` cases
    //! pin the b.b2j fail-closed allocation contract. Their load-bearing
    //! claim is the pair: allocation MUST refuse (`Ok(None)`) against an
    //! absent counter, and the key MUST otherwise be guaranteed to exist —
    //! which is what makes "absent" mean "the Valkey keyspace was lost
    //! underneath this process" rather than "nobody has created a layer
    //! yet". A plain `INCR` returns `1` in both situations, which is the
    //! object-id reuse this ticket closes.
    //!
    //! **Namespace discipline (shared Valkey).** Every case reads and writes
    //! only `tv.key_schema().counter_object_id()`, which carries the
    //! fixture's per-test prefix. Nothing here issues `FLUSHALL` / `FLUSHDB`
    //! — the backing instance is shared with concurrently-running agents and
    //! CI jobs, and `TestValkey`'s Drop already SCAN+UNLINKs the prefix.
    use super::*;
    use crate::storage::test_support::TestValkey;
    use fred::interfaces::{ClientLike, KeysInterface};

    fn store(tv: &TestValkey) -> CounterStore {
        CounterStore::new(tv.client().clone(), tv.key_schema().clone())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seed_object_id_if_below_absent_key_initializes_counter() {
        let tv = TestValkey::setup().await;
        let store = store(&tv);
        let keys = tv.key_schema();

        let returned = store
            .seed_object_id_if_below(42)
            .await
            .expect("seed against absent key must succeed");

        assert_eq!(
            returned, 42,
            "absent key: script must initialize the counter to the seed and return it",
        );

        let stored: Option<String> = tv
            .client()
            .pool
            .get(keys.counter_object_id())
            .await
            .expect("GET counter key must succeed");
        assert_eq!(
            stored.as_deref(),
            Some("42"),
            "absent key: script must write-through the seed value to the underlying key",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seed_object_id_if_below_below_current_is_noop() {
        let tv = TestValkey::setup().await;
        let store = store(&tv);
        let keys = tv.key_schema();

        tv.client()
            .pool
            .set::<(), _, _>(keys.counter_object_id(), "100", None, None, false)
            .await
            .expect("pre-seed SET must succeed");

        let returned = store
            .seed_object_id_if_below(50)
            .await
            .expect("seed below current must succeed");

        assert_eq!(
            returned, 100,
            "seed below current: script must return the existing value, not the seed",
        );

        let stored: Option<String> = tv
            .client()
            .pool
            .get(keys.counter_object_id())
            .await
            .expect("GET counter key must succeed");
        assert_eq!(
            stored.as_deref(),
            Some("100"),
            "seed below current: underlying counter must be unchanged (no-op contract)",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seed_object_id_if_below_above_current_advances_counter() {
        let tv = TestValkey::setup().await;
        let store = store(&tv);
        let keys = tv.key_schema();

        tv.client()
            .pool
            .set::<(), _, _>(keys.counter_object_id(), "100", None, None, false)
            .await
            .expect("pre-seed SET must succeed");

        let returned = store
            .seed_object_id_if_below(500)
            .await
            .expect("seed above current must succeed");

        assert_eq!(
            returned, 500,
            "seed above current: script must advance the counter to the seed and return it",
        );

        let stored: Option<String> = tv
            .client()
            .pool
            .get(keys.counter_object_id())
            .await
            .expect("GET counter key must succeed");
        assert_eq!(
            stored.as_deref(),
            Some("500"),
            "seed above current: underlying counter must be rewritten to the seed value",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seed_object_id_if_below_equal_to_current_is_noop() {
        // Pins the `<` (strict less-than) boundary: equal seed must be a
        // no-op. A buggy `<=` would re-write the counter on every Task 4
        // startup against an unchanged checkpoint, defeating the purpose
        // of the conditional seed.
        let tv = TestValkey::setup().await;
        let store = store(&tv);
        let keys = tv.key_schema();

        tv.client()
            .pool
            .set::<(), _, _>(keys.counter_object_id(), "100", None, None, false)
            .await
            .expect("pre-seed SET must succeed");

        let returned = store
            .seed_object_id_if_below(100)
            .await
            .expect("seed equal to current must succeed");

        assert_eq!(
            returned, 100,
            "seed equal to current: script must return the existing value (strict `<` boundary)",
        );

        let stored: Option<String> = tv
            .client()
            .pool
            .get(keys.counter_object_id())
            .await
            .expect("GET counter key must succeed");
        assert_eq!(
            stored.as_deref(),
            Some("100"),
            "seed equal to current: underlying counter must be unchanged (no-op on equal)",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_seed_object_id_if_below_routes_to_single_slot() {
        // Pins the routing contract from the Task body: the bare
        // `le:counter:object_id` key has no `{...}` hash tag, so it routes
        // to a single Cluster slot. A regression that accidentally added
        // a hash tag would surface as a CROSSSLOT failure here (or, worse,
        // a silent mis-route between EVAL writes and `pool.get` reads).
        let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
            eprintln!(
                "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)"
            );
            return;
        };

        let tv = TestValkey::setup_with_url(cluster_url, 4).await;
        assert!(
            tv.client().pool.next().is_clustered(),
            "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment \
             (use `redis-cluster://...`); the single-slot routing contract \
             is only exercised when `is_clustered()` is true"
        );

        let store = store(&tv);
        let keys = tv.key_schema();

        // Scenario 1: absent key initializes to seed.
        let returned = store
            .seed_object_id_if_below(42)
            .await
            .expect("cluster: seed against absent key must succeed without CROSSSLOT");
        assert_eq!(returned, 42, "cluster: absent-key initialize");
        let stored: Option<String> = tv
            .client()
            .pool
            .get(keys.counter_object_id())
            .await
            .expect("cluster: GET counter key must succeed");
        assert_eq!(
            stored.as_deref(),
            Some("42"),
            "cluster: write must round-trip through the same slot the EVAL targeted"
        );

        // Scenario 2: seed above current advances the counter.
        let returned = store
            .seed_object_id_if_below(500)
            .await
            .expect("cluster: seed above current must succeed");
        assert_eq!(returned, 500, "cluster: advance on seed > current");
        let stored: Option<String> = tv
            .client()
            .pool
            .get(keys.counter_object_id())
            .await
            .expect("cluster: GET counter key must succeed");
        assert_eq!(
            stored.as_deref(),
            Some("500"),
            "cluster: advance write-through"
        );

        // Scenario 3: seed below current is a no-op.
        let returned = store
            .seed_object_id_if_below(100)
            .await
            .expect("cluster: seed below current must succeed");
        assert_eq!(returned, 500, "cluster: no-op on seed < current");
        let stored: Option<String> = tv
            .client()
            .pool
            .get(keys.counter_object_id())
            .await
            .expect("cluster: GET counter key must succeed");
        assert_eq!(
            stored.as_deref(),
            Some("500"),
            "cluster: no-op below leaves counter"
        );

        // Scenario 4: seed equal to current is a no-op (strict `<` boundary).
        let returned = store
            .seed_object_id_if_below(500)
            .await
            .expect("cluster: seed equal to current must succeed");
        assert_eq!(returned, 500, "cluster: no-op on seed == current");
        let stored: Option<String> = tv
            .client()
            .pool
            .get(keys.counter_object_id())
            .await
            .expect("cluster: GET counter key must succeed");
        assert_eq!(
            stored.as_deref(),
            Some("500"),
            "cluster: no-op on equal leaves counter"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // b.b2j — fail-closed allocation + counter establishment.
    // ─────────────────────────────────────────────────────────────────────

    /// Read the raw counter key, bypassing `current_object_id`'s
    /// absent-collapses-to-`0` contract. The b.b2j cases need to tell
    /// "absent" apart from "present and zero" — `current_object_id` reports
    /// `0` for both, and that conflation is precisely what the `SET NX 0`
    /// establishment step exists to remove.
    async fn raw_counter(tv: &TestValkey) -> Option<String> {
        tv.client()
            .pool
            .get(tv.key_schema().counter_object_id())
            .await
            .expect("GET counter key must succeed")
    }

    /// Absent counter → `Ok(None)`, and the refused allocation leaves the
    /// key absent.
    ///
    /// Both halves matter. The `None` is the fail-closed verdict itself. The
    /// "key still absent" half pins that the refusal is a pure read: were the
    /// script to establish the key as a side effect of refusing, the caller's
    /// post-repair retry would find a counter at `0` that no checkpoint floor
    /// had ever been proven against, and allocate `1` anyway — the reuse this
    /// ticket closes, reintroduced one layer down.
    ///
    /// Pre-fix regression mode: `next_object_id` was a bare `INCR`, which
    /// returns `1` on an absent key AND creates it. Both assertions fail.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn next_object_id_absent_counter_returns_none_and_creates_no_key() {
        let tv = TestValkey::setup().await;
        let store = store(&tv);

        // `TestValkey` mints a fresh per-test prefix, so the counter is
        // absent by construction — the same shape a keyspace loss leaves
        // behind under a surviving process.
        assert_eq!(
            raw_counter(&tv).await,
            None,
            "fixture precondition: a fresh test prefix must have no counter key"
        );

        let allocated = store
            .next_object_id()
            .await
            .expect("the EVAL itself must succeed — an absent counter is a verdict, not an error");

        assert_eq!(
            allocated, None,
            "allocation against an absent le:counter:object_id must fail closed with \
             Ok(None); Some(id) here means a bare INCR minted an id from an empty \
             counter, which re-mints object ids that surviving checkpoint blobs claim"
        );
        assert_eq!(
            raw_counter(&tv).await,
            None,
            "the refused allocation must not create the counter key as a side effect — \
             a counter established at 0 without a proven checkpoint floor would let the \
             caller's retry allocate 1 and collide anyway"
        );
    }

    /// Present counter → `Ok(Some(n))`, monotonically incrementing from the
    /// stored value. Pins that fail-closed did not cost the normal path its
    /// `INCR` semantics.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn next_object_id_present_counter_increments_from_stored_value() {
        let tv = TestValkey::setup().await;
        let store = store(&tv);
        let keys = tv.key_schema();

        tv.client()
            .pool
            .set::<(), _, _>(keys.counter_object_id(), "41", None, None, false)
            .await
            .expect("pre-seed SET must succeed");

        assert_eq!(
            store.next_object_id().await.expect("EVAL must succeed"),
            Some(42),
            "allocation against a counter at 41 must return 42"
        );
        assert_eq!(
            store.next_object_id().await.expect("EVAL must succeed"),
            Some(43),
            "consecutive allocations must be strictly increasing and gap-free"
        );
        assert_eq!(
            raw_counter(&tv).await.as_deref(),
            Some("43"),
            "the counter key must hold the last allocated id"
        );
    }

    /// `SET NX 0` establishment: `true` on the call that creates the key,
    /// `false` on every call after, with the value untouched in between.
    ///
    /// The `false` return is not cosmetic — `main.rs`'s Step 3c logs on it,
    /// and more importantly it proves the write is `NX`-guarded rather than
    /// an unconditional `SET`. An unconditional `SET 0` on a second call
    /// would rewind a seeded counter to zero, which is a strictly worse
    /// version of the bug this ticket closes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn initialize_object_id_counter_if_absent_creates_once_then_reports_false() {
        let tv = TestValkey::setup().await;
        let store = store(&tv);

        let created = store
            .initialize_object_id_counter_if_absent()
            .await
            .expect("establishing an absent counter must succeed");
        assert!(
            created,
            "the first call against an absent key must report that it created the counter"
        );
        assert_eq!(
            raw_counter(&tv).await.as_deref(),
            Some("0"),
            "the established counter must hold 0 so the first allocation mints 1"
        );

        let created_again = store
            .initialize_object_id_counter_if_absent()
            .await
            .expect("a repeat establishment call must succeed");
        assert!(
            !created_again,
            "the second call must report false — the key already existed, so SET NX \
             wrote nothing"
        );
        assert_eq!(
            raw_counter(&tv).await.as_deref(),
            Some("0"),
            "the repeat call must leave the counter value untouched"
        );

        // Establishment is what un-blocks allocation: the pairing is the
        // whole contract, so assert it here rather than trusting the two
        // halves in isolation.
        assert_eq!(
            store.next_object_id().await.expect("EVAL must succeed"),
            Some(1),
            "an established-at-0 counter must allocate 1 on the first CreateLayer"
        );
    }

    /// `SET NX` must never stomp a counter a peer replica already seeded
    /// from the checkpoint-store floor.
    ///
    /// This is the race `main.rs`'s Step 3c and the repair path's `Absent`
    /// arm both run into: two replicas conclude "no sidecar" at the same
    /// instant while a third has just landed a real seed. `NX` is what makes
    /// the losing writes harmless.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn initialize_object_id_counter_if_absent_does_not_stomp_a_seeded_counter() {
        let tv = TestValkey::setup().await;
        let store = store(&tv);
        let keys = tv.key_schema();

        tv.client()
            .pool
            .set::<(), _, _>(keys.counter_object_id(), "6000", None, None, false)
            .await
            .expect("pre-seed SET must succeed");

        let created = store
            .initialize_object_id_counter_if_absent()
            .await
            .expect("establishment against an existing counter must succeed");
        assert!(
            !created,
            "establishment must report false against an already-seeded counter"
        );
        assert_eq!(
            raw_counter(&tv).await.as_deref(),
            Some("6000"),
            "a peer's checkpoint-derived seed must survive verbatim — rewinding it to 0 \
             would hand the next allocation an id every surviving checkpoint blob claims"
        );
        assert_eq!(
            store.next_object_id().await.expect("EVAL must succeed"),
            Some(6001),
            "allocation must continue above the peer's seed"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // b.b2j — the corrupt-counter arm of `next_object_id`.
    //
    // Three-way contract, and the third way is the one these cases pin: a nil
    // reply is the absent verdict, a POSITIVE integer is an allocation, and any
    // other integer means `INCR` ran against a value this service never writes.
    // The distinction is operationally load-bearing, not pedantic — `Internal`
    // says "the data is wrong, a human must look", where `Unavailable` (the
    // absent verdict's eventual shape) says "retry me". A corrupt counter
    // laundered into the absent verdict would trigger a repair that re-seeds a
    // counter which is not missing at all.
    // ─────────────────────────────────────────────────────────────────────

    /// A negative counter → `Internal`, not an allocation and not the absent
    /// verdict.
    ///
    /// `-5` INCRs to `-4`, which `u64::try_from` rejects. Nothing in this
    /// service ever writes a negative value, so reaching here means something
    /// external wrote the key — the case for a loud `Internal` rather than a
    /// retriable status.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn next_object_id_negative_counter_is_internal_not_an_allocation() {
        let tv = TestValkey::setup().await;
        let store = store(&tv);
        let keys = tv.key_schema();

        tv.client()
            .pool
            .set::<(), _, _>(keys.counter_object_id(), "-5", None, None, false)
            .await
            .expect("pre-seed SET must succeed");

        let err = store
            .next_object_id()
            .await
            .expect_err("a counter this service never writes must not produce an allocation");

        let ServiceError::Internal(msg) = &err else {
            panic!(
                "a corrupt counter is bad DATA, not an unavailable dependency: `Internal` \
                 tells an operator to investigate, `Unavailable` tells the client to retry \
                 into the same corruption forever; got {err:?}"
            );
        };
        assert!(
            msg.contains("non-positive id (corrupt counter)"),
            "the message must name the condition so the operator does not have to infer it \
             from a bare status code; got {msg:?}"
        );
        assert!(
            msg.contains("-4"),
            "and it must carry the OBSERVED post-INCR value (-4 from a stored -5), which is \
             the only clue to what wrote the key; got {msg:?}"
        );
    }

    /// The former `-1`-sentinel aliasing case, now unambiguous.
    ///
    /// An earlier revision of `alloc_object_id.lua` returned `-1` to mean "the
    /// counter is absent". A counter sitting at `-2` INCRs to exactly `-1` and
    /// so reproduced that sentinel byte-for-byte: a live (if corrupt) counter
    /// read as "the Valkey keyspace was lost", triggering a spurious repair that
    /// would re-seed a counter which was never missing. The absent verdict is a
    /// nil reply now — outside `INCR`'s range entirely — so the two cases are
    /// structurally distinguishable, and this test is what keeps them that way.
    ///
    /// Pre-fix regression mode: `Ok(None)`, the absent verdict, from a counter
    /// that exists.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn next_object_id_counter_at_minus_two_is_internal_not_the_absent_verdict() {
        let tv = TestValkey::setup().await;
        let store = store(&tv);
        let keys = tv.key_schema();

        tv.client()
            .pool
            .set::<(), _, _>(keys.counter_object_id(), "-2", None, None, false)
            .await
            .expect("pre-seed SET must succeed");

        let err = store.next_object_id().await.expect_err(
            "a counter INCRing to -1 must NOT be readable as the absent verdict — that \
             aliasing is the reason the absent verdict is a nil reply rather than an in-band \
             numeric sentinel",
        );

        let ServiceError::Internal(msg) = &err else {
            panic!("a corrupt counter must surface as Internal; got {err:?}");
        };
        assert!(
            msg.contains("-1"),
            "the observed value must be reported as -1 — the exact value the deleted sentinel \
             used, which is what makes this case worth its own test; got {msg:?}"
        );

        // The key still exists: the EVAL did INCR it. Asserting this rules out
        // the other half of the old aliasing — a caller concluding "absent" and
        // a repair then establishing a counter over live (if corrupt) data.
        assert_eq!(
            raw_counter(&tv).await.as_deref(),
            Some("-1"),
            "and the counter must still be there, holding its INCRed value. The absent \
             verdict's whole meaning is `this key does not exist`; a corrupt counter is a \
             different situation with a different fix"
        );
    }

    /// A non-integer counter → the error the fred/Lua path actually produces,
    /// which is `Internal` via the shared mapper.
    ///
    /// Distinct from the two cases above: those get a well-formed integer reply
    /// that the Rust-side `u64::try_from` rejects, so the diagnosis is made
    /// here. This one never gets a reply at all — `INCR` itself raises inside
    /// the script — so the error travels as a fred error and routes through
    /// `map_fred_err_for_grpc`. Both must land on `Internal`, and this test is
    /// what pins that the two very different mechanisms agree; a server-side
    /// script error classified as transient would have the client retrying into
    /// permanently malformed data.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn next_object_id_non_integer_counter_is_internal() {
        let tv = TestValkey::setup().await;
        let store = store(&tv);
        let keys = tv.key_schema();

        tv.client()
            .pool
            .set::<(), _, _>(keys.counter_object_id(), "not-a-number", None, None, false)
            .await
            .expect("pre-seed SET must succeed");

        let err = store
            .next_object_id()
            .await
            .expect_err("INCR against a non-integer value must not produce an allocation");

        let ServiceError::Internal(msg) = &err else {
            panic!(
                "a counter holding non-numeric bytes is permanently malformed data, so the \
                 error must be Internal — classifying it transient would have the client \
                 retrying forever against a value that will never parse; got {err:?}"
            );
        };
        assert!(
            msg.contains("EVAL alloc_object_id"),
            "the message must name the operation, which is what the shared \
             `map_fred_err_for_grpc` contract exists to guarantee; got {msg:?}"
        );

        assert_eq!(
            raw_counter(&tv).await.as_deref(),
            Some("not-a-number"),
            "and the refused allocation must leave the corrupt value in place for the \
             operator to inspect — silently overwriting it would destroy the evidence of \
             whatever wrote it"
        );
    }

    /// Cluster sibling of the three cases above: the `alloc_object_id` Lua
    /// touches one un-hash-tagged key, so it must route to a single slot on
    /// Valkey Cluster exactly as `seed_object_id_if_below` does.
    ///
    /// A regression that added a hash tag to `le:counter:object_id`, or that
    /// split the atomic `EXISTS` + `INCR` into two round-trips landing on
    /// different connections, would surface here as a CROSSSLOT error or a
    /// mis-routed read-back.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_alloc_object_id_routes_to_single_slot() {
        let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
            eprintln!(
                "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)"
            );
            return;
        };

        let tv = TestValkey::setup_with_url(cluster_url, 4).await;
        assert!(
            tv.client().pool.next().is_clustered(),
            "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment \
             (use `redis-cluster://...`); the single-slot routing contract \
             is only exercised when `is_clustered()` is true"
        );

        let store = store(&tv);

        // Scenario 1: absent counter fails closed, without CROSSSLOT.
        assert_eq!(
            store
                .next_object_id()
                .await
                .expect("cluster: alloc against an absent counter must succeed as a verdict"),
            None,
            "cluster: absent counter must fail closed exactly as on single-node"
        );

        // Scenario 2: establishment routes to the same slot the EVAL reads.
        assert!(
            store
                .initialize_object_id_counter_if_absent()
                .await
                .expect("cluster: establishment must succeed"),
            "cluster: first establishment must create the counter"
        );

        // Scenario 3: allocation now round-trips through that same slot.
        assert_eq!(
            store
                .next_object_id()
                .await
                .expect("cluster: alloc must succeed once the counter exists"),
            Some(1),
            "cluster: allocation must mint 1 from the established counter"
        );
        assert_eq!(
            store
                .next_object_id()
                .await
                .expect("cluster: second alloc must succeed"),
            Some(2),
            "cluster: consecutive allocations must stay gap-free across the pool's \
             round-robin connections"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // b.bhb — `retry_transient`, the bounded retry these three primitives
    // gained when the `txn_pool` stopped re-issuing commands
    // ═════════════════════════════════════════════════════════════════════
    //
    // The pool now runs at `max_command_attempts = 1` so nothing issued under
    // a `WATCH` can be replayed onto a connection that never held it. These
    // primitives hold no `WATCH`, but they lost fred's transparent retry along
    // with everything else on that pool — and their callers cannot absorb it:
    // the startup counter-seed gates `counter_seed_ready` (and therefore
    // readiness) on a single call. A reconnect that used to be invisible would
    // become a failed boot.
    //
    // Driven through an injected closure rather than a real Valkey: the
    // contract is entirely about *which* `ServiceError` variant re-arms the
    // loop and how many times, and a real transient error cannot be produced
    // on demand at an exact attempt index. `start_paused` makes the fixed
    // 50 ms delay auto-advance, so the retry budget costs no wall clock.

    /// Count invocations so each case can assert the exact attempt budget it
    /// consumed — "it eventually returned Ok" would pass with an unbounded
    /// loop, which is the failure mode a retry helper must not have.
    fn counting<T>(
        results: Vec<Result<T, ServiceError>>,
    ) -> (
        impl FnMut() -> std::future::Ready<Result<T, ServiceError>>,
        std::rc::Rc<std::cell::Cell<usize>>,
    ) {
        // A queue rather than an indexed clone: `ServiceError` is deliberately
        // not `Clone`, and popping also makes an over-budget attempt fail
        // loudly on an empty queue rather than silently replaying the last
        // scripted outcome forever.
        let mut scripted: std::collections::VecDeque<Result<T, ServiceError>> =
            results.into_iter().collect();
        let calls = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let seen = calls.clone();
        let attempt = move || {
            seen.set(seen.get() + 1);
            let outcome = scripted.pop_front().unwrap_or_else(|| {
                panic!(
                    "retry_transient made attempt #{} — more than the \
                     {TXN_COUNTER_MAX_ATTEMPTS} the helper is allowed",
                    seen.get()
                )
            });
            std::future::ready(outcome)
        };
        (attempt, calls)
    }

    #[tokio::test(start_paused = true)]
    async fn retry_transient_returns_the_first_success_without_retrying() {
        let (attempt, calls) = counting(vec![Ok(7u64)]);
        let out = retry_transient("test op", attempt).await;

        assert_eq!(
            out.expect("a first-attempt success must pass straight through"),
            7
        );
        assert_eq!(
            calls.get(),
            1,
            "the healthy path must cost exactly one round trip — a retry helper that \
             re-runs a successful call would double every startup write"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn retry_transient_retries_an_unavailable_and_returns_the_later_success() {
        // The failure this exists to cover: one reconnect landing on the
        // connection mid-command, which fred re-establishes in ~100 ms.
        let (attempt, calls) = counting(vec![
            Err(ServiceError::Unavailable("transient".to_owned())),
            Ok(11u64),
        ]);
        let out = retry_transient("test op", attempt).await;

        assert_eq!(
            out.expect(
                "a transient failure followed by a success must return the success — this is \
                 the whole point of the helper: pre-b.bhb fred absorbed this invisibly, and \
                 the counter-seed caller gates readiness on the result"
            ),
            11
        );
        assert_eq!(calls.get(), 2, "it must stop as soon as an attempt settles");
    }

    #[tokio::test(start_paused = true)]
    async fn retry_transient_gives_up_after_the_bounded_attempt_budget() {
        let (attempt, calls) = counting::<u64>(vec![
            Err(ServiceError::Unavailable("1".to_owned())),
            Err(ServiceError::Unavailable("2".to_owned())),
            Err(ServiceError::Unavailable("3".to_owned())),
        ]);
        let out = retry_transient("test op", attempt).await;

        match out {
            Err(ServiceError::Unavailable(msg)) => assert_eq!(
                msg, "3",
                "the LAST attempt's error must surface, not a synthesised \
                 budget-exhausted message — the caller and its logs need the real cause"
            ),
            other => {
                panic!("a store that never recovers must still surface Unavailable; got {other:?}")
            }
        }
        assert_eq!(
            calls.get(),
            TXN_COUNTER_MAX_ATTEMPTS as usize,
            "the budget is a bound, not a suggestion: an unbounded loop here would hang \
             startup behind a store outage instead of failing the readiness gate"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn retry_transient_does_not_retry_a_non_transient_error() {
        // `Internal` from these primitives means a corrupt (non-numeric)
        // counter value. Retrying re-reads the same bytes and re-fails; worse,
        // it delays a startup verdict that must surface immediately.
        let (attempt, calls) = counting::<u64>(vec![Err(ServiceError::Internal(
            "non-numeric reply (corrupt counter)".to_owned(),
        ))]);
        let out = retry_transient("test op", attempt).await;

        assert!(
            matches!(out, Err(ServiceError::Internal(_))),
            "a corrupt-counter Internal must surface unchanged"
        );
        assert_eq!(
            calls.get(),
            1,
            "only Unavailable re-arms the loop — that is exactly what \
             `map_fred_err_for_grpc` produces for the transient fred kinds, and nothing else \
             is safe to repeat"
        );
    }
}
