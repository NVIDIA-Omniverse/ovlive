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

//! Factory that turns the operator-parsed
//! [`CheckpointAdapter`](super::CheckpointAdapter) enum value (resolved by
//! `Config::from_env` from `LIVE_EDIT_CHECKPOINT_ADAPTER`) into a concrete
//! `Arc<dyn CheckpointStore>` for `main.rs` to thread into the rest of the
//! service.
//!
//! This is the single dispatch point — `main.rs` does not match on the
//! adapter enum directly, so adding a new adapter is one match arm here plus
//! the constructor in `adapters/<name>.rs`.
//!
//! ## Why a private NoopCheckpointStore for `Disabled`?
//!
//! `main.rs` constructs the `Arc<dyn CheckpointStore>` unconditionally — the
//! periodic checkpoint loop and recovery hook in later Epic-1 Tasks both take
//! the trait object by argument, and threading an `Option<Arc<...>>` through
//! every downstream call would make the type-level "checkpointing is off"
//! discipline fall on every caller. Instead, when the operator selects
//! `Disabled`, we hand out a private no-op adapter whose every method returns
//! [`CheckpointError::NotImplemented`]. A caller that accidentally drives the
//! loop or recovery hook with `LIVE_EDIT_CHECKPOINT_ENABLED=false` then surfaces
//! a clean classified error instead of silently writing or reading state — the
//! checkpoint-loop spawn site in `t2.fgj.ya.9j` is what actually gates on
//! `checkpoint_enabled`, so off-mode is a single-source-of-truth decision.
//!
//! The `NoopCheckpointStore` is intentionally NOT a public adapter under
//! `adapters/`: it is never written for in tests (we use `InMemoryCheckpointStore`
//! for that) and exposing it would invite production callers to construct it
//! directly, bypassing the env-var gate.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;

use crate::auth::service_token::ServiceTokenProvider;
use crate::config::Config;
use crate::observability::metrics::SharedMetrics;

use super::adapters::{
    InMemoryCheckpointStore, LocalDiskCheckpointStore, StorageApiCheckpointStore,
};
use super::store::{
    CheckpointAdapter, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta,
    CheckpointStore,
};

/// Build the concrete `Arc<dyn CheckpointStore>` selected by
/// `config.checkpoint_adapter`.
///
/// Maps the operator-parsed [`CheckpointAdapter`] enum to a constructed
/// adapter:
///
/// - [`CheckpointAdapter::Disabled`] → a private no-op adapter whose every
///   trait method returns [`CheckpointError::NotImplemented`]. Off-mode
///   callers cannot accidentally write or read state through this binding;
///   the periodic checkpoint loop's spawn site is what gates on
///   `LIVE_EDIT_CHECKPOINT_ENABLED`, but this layer is defense in depth.
/// - [`CheckpointAdapter::InMemory`] → fresh [`InMemoryCheckpointStore`].
/// - [`CheckpointAdapter::LocalDisk`] → [`LocalDiskCheckpointStore`] rooted
///   at `config.checkpoint_local_disk_path`. `Config::from_env` already
///   rejects an empty / missing path on the `LocalDisk` branch; this branch
///   re-checks the `Option` and returns an explicit error if the primary
///   gate was somehow bypassed, rather than silently falling back to the
///   current working directory.
/// - [`CheckpointAdapter::StorageApi`] → constructs a
///   [`StorageApiCheckpointStore`] from `config.checkpoint_storage_api_endpoint`,
///   `config.checkpoint_base_path`, `config.checkpoint_connect_timeout_ms`,
///   `config.checkpoint_request_timeout_ms`, and the
///   `service_token_provider` argument. The returned `Arc<dyn
///   CheckpointStore>` issues outbound REST calls against the storage-api
///   `by-address` surface (SDD §SR-4.1, §SR-4.6 — URL shape +
///   path-rendering contracts) and authenticates every request with the
///   sidecar-provisioned service-identity Bearer token resolved through
///   `ServiceTokenProvider` (SDD §14.6). The provider argument is
///   required. `Config::from_env` is the primary gate: it refuses
///   `StorageApi` without `LIVE_EDIT_SERVICE_TOKEN_FILE`, so no operator
///   setting reaches this factory without a provider. As defense in depth,
///   a caller that reaches it without one anyway gets an explicit `String`
///   error naming the env var rather than an adapter that would 401 on
///   every call.
///
/// # Errors
///
/// Returns a `String` error, the same type as `Config::from_env`'s. `main.rs`
/// handles the two differently: a `Config::from_env` error is printed to
/// stderr as `configuration error: …`, while this factory's error is logged
/// at ERROR (`failed to construct checkpoint store`); both exit 1. The error
/// message names the adapter that failed and includes the underlying
/// [`CheckpointError`] `Display` string.
///
/// `metrics` is threaded only to the `StorageApi` arm, and only to count b.gg7
/// ceiling elapses (`live_edit_checkpoint_adapter_ceiling_elapsed_total`). It is
/// `Option` because this factory is exercised from a dozen in-crate and
/// integration tests that have no meter and no interest in one; passing `None`
/// changes no behaviour — the ceilings still apply and still WARN, they are
/// simply not counted. Production wires `Some(metrics)` from `main.rs`.
///
/// # Note on the `async` signature
///
/// Neither current adapter construction is actually async — InMemory is a
/// `HashMap` allocation, LocalDisk is synchronous `create_dir_all`, and the
/// StorageApi constructor builds a `reqwest::Client` synchronously. The
/// future Epic-5 trait impl may do a real network handshake at
/// construction time (auth-token fetch, capability probe), so the factory
/// is `async` from day one to avoid forcing a
/// `tokio::task::spawn_blocking` later.
pub async fn build_checkpoint_store(
    config: &Config,
    service_token_provider: Option<Arc<ServiceTokenProvider>>,
    metrics: Option<SharedMetrics>,
) -> Result<Arc<dyn CheckpointStore>, String> {
    match config.checkpoint_adapter {
        CheckpointAdapter::Disabled => {
            Ok(Arc::new(NoopCheckpointStore) as Arc<dyn CheckpointStore>)
        }
        CheckpointAdapter::InMemory => {
            Ok(Arc::new(InMemoryCheckpointStore::new()) as Arc<dyn CheckpointStore>)
        }
        CheckpointAdapter::LocalDisk => {
            // Defense in depth: `Config::from_env` already rejects a missing
            // / empty `LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH` on the
            // `LocalDisk` branch (the primary gate). If a future code path
            // somehow constructs a `Config` directly and skips that check,
            // fail loudly here rather than silently creating
            // `.live-edit/{layers,sessions,ext}/` under the service's CWD.
            let root = config.checkpoint_local_disk_path.clone().ok_or_else(|| {
                "checkpoint_local_disk_path missing — Config::from_env primary gate \
                 (LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH required when \
                 LIVE_EDIT_CHECKPOINT_ADAPTER=LocalDisk) was bypassed"
                    .to_string()
            })?;
            let store = LocalDiskCheckpointStore::new(std::path::PathBuf::from(root))
                .map_err(|e| format!("failed to construct LocalDisk checkpoint store: {e}"))?;
            Ok(Arc::new(store) as Arc<dyn CheckpointStore>)
        }
        CheckpointAdapter::StorageApi => {
            // Defense in depth on each Config option: `Config::from_env`
            // already rejects a missing / empty
            // `LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT` and
            // `LIVE_EDIT_CHECKPOINT_BASE_PATH` on the `StorageApi` branch
            // (the primary gate, parallel to the LocalDisk arm above). If a
            // future code path constructs a `Config` directly and skips
            // those checks, fail loudly here.
            let endpoint = config
                .checkpoint_storage_api_endpoint
                .clone()
                .ok_or_else(|| {
                    "checkpoint_storage_api_endpoint missing — Config::from_env primary gate \
                     (LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT required when \
                     LIVE_EDIT_CHECKPOINT_ADAPTER=StorageApi) was bypassed"
                        .to_string()
                })?;
            let base_path = config.checkpoint_base_path.clone().ok_or_else(|| {
                "checkpoint_base_path missing — Config::from_env primary gate \
                 (LIVE_EDIT_CHECKPOINT_BASE_PATH required when \
                 LIVE_EDIT_CHECKPOINT_ADAPTER=StorageApi) was bypassed"
                    .to_string()
            })?;
            // The outbound `StorageApi` adapter authenticates every request
            // with the sidecar-provisioned service-identity Bearer token
            // (SDD §14.6). The provider is only constructed in `main.rs`
            // when `LIVE_EDIT_SERVICE_TOKEN_FILE` is set. Defense in depth,
            // like the two checks above: `Config::from_env` already refuses
            // `StorageApi` without the token file (the primary gate). If a
            // caller reaches here without a provider anyway, fail closed
            // with an explicit message rather than wiring an adapter that
            // would 401 on every call.
            let provider = service_token_provider.ok_or_else(|| {
                "service_token_provider missing — LIVE_EDIT_SERVICE_TOKEN_FILE is required \
                 when LIVE_EDIT_CHECKPOINT_ADAPTER=StorageApi (the outbound adapter \
                 authenticates every request with the sidecar-provisioned service-identity \
                 Bearer token; provision the file with a token-provisioner sidecar (the \
                 chart's serviceIdentity.accessTokenFile settings) before turning on the \
                 StorageApi adapter)"
                    .to_string()
            })?;
            let store = StorageApiCheckpointStore::new(
                endpoint,
                base_path,
                provider,
                Duration::from_millis(config.checkpoint_connect_timeout_ms),
                Duration::from_millis(config.checkpoint_request_timeout_ms),
            )
            .map_err(|e| format!("failed to construct StorageApi checkpoint store: {e}"))?;
            let store = match metrics {
                Some(m) => store.with_metrics(m, config.service_label.clone()),
                None => store,
            };

            Ok(Arc::new(store) as Arc<dyn CheckpointStore>)
        }
    }
}

/// Private no-op [`CheckpointStore`] handed out when
/// `LIVE_EDIT_CHECKPOINT_ADAPTER=disabled`.
///
/// Every method returns [`CheckpointError::NotImplemented`] so an off-mode
/// caller that somehow drives a write or read path surfaces a classified
/// error instead of corrupting state. The periodic checkpoint loop's spawn
/// site (Epic 1 `t2.fgj.ya.9j`) is the primary gate; this is defense in
/// depth.
struct NoopCheckpointStore;

#[async_trait]
impl CheckpointStore for NoopCheckpointStore {
    async fn put_layer_checkpoint(
        &self,
        _object_id: u64,
        _snap_seq: u64,
        _ompf: Bytes,
        _meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "put_layer_checkpoint",
        })
    }

    async fn get_latest_layer_checkpoint(
        &self,
        _object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "get_latest_layer_checkpoint",
        })
    }

    // `get_layer_checkpoint` deliberately inherits the trait default, which
    // already returns exactly the `NotImplemented { method }` this type hand-
    // writes for most other methods (see the note at the end of this impl).
    // There is no inner store to forward to.

    async fn list_layer_checkpoints(
        &self,
        _object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "list_layer_checkpoints",
        })
    }

    async fn delete_layer_checkpoint(
        &self,
        _handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "delete_layer_checkpoint",
        })
    }

    async fn put_session_checkpoint(
        &self,
        _session_id: &str,
        _body: Bytes,
        _meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "put_session_checkpoint",
        })
    }

    async fn get_session_checkpoint(
        &self,
        _session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "get_session_checkpoint",
        })
    }

    async fn delete_session_checkpoint(&self, _session_id: &str) -> Result<(), CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "delete_session_checkpoint",
        })
    }

    async fn put_ext_blob(&self, _hash_hex: &str, _bytes: Bytes) -> Result<(), CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "put_ext_blob",
        })
    }

    async fn get_ext_blob(&self, _hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "get_ext_blob",
        })
    }

    async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "delete_ext_blob",
        })
    }

    async fn put_counters_sidecar(&self, _body: Bytes) -> Result<(), CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "put_counters_sidecar",
        })
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "get_counters_sidecar",
        })
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "list_all_object_ids",
        })
    }

    // `list_referenced_ext_hashes` is intentionally NOT overridden — the
    // trait's default impl calls `list_all_object_ids`, which already
    // returns `NotImplemented` for this adapter, so the error propagates
    // through with the same classification. Overriding here would
    // duplicate the surface without changing observable behaviour.
    //
    // `get_layer_checkpoint`, `list_session_checkpoint_ids` and
    // `ext_blob_exists` are likewise intentionally NOT overridden: each trait
    // default already returns `NotImplemented { method }` naming that method,
    // which IS the off-mode contract every other method here spells out by
    // hand. Inheriting is a bug in a real adapter, but for the off-mode shim
    // it is the correct answer, so the omission is deliberate rather than an
    // oversight.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::CheckpointStore;
    use crate::config::PodMemorySource;

    fn test_meta() -> CheckpointMeta {
        CheckpointMeta {
            creator_replica_id: "test-replica".to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(1),
            // Identity-carrying even though `NoopCheckpointStore` refuses
            // every method before it could look: these tests are about the
            // `NotImplemented` classification, and a fixture that would be
            // rejected by `validate_generation_identity` on any real adapter
            // would make "the Noop store refused it" ambiguous.
            ..crate::checkpoint::adapters::identity_default()
        }
    }

    /// Every `CheckpointStore` method on the private `NoopCheckpointStore`
    /// MUST surface a classified `NotImplemented` carrying the method's name.
    ///
    /// The `Disabled` adapter is handed out by `build_checkpoint_store` when
    /// the operator sets `LIVE_EDIT_CHECKPOINT_ADAPTER=Disabled` (or accepts
    /// the default). The checkpoint-loop spawn site gates on
    /// `LIVE_EDIT_CHECKPOINT_ENABLED` and never invokes these methods in
    /// production, but defense in depth means a caller that bypasses the gate
    /// must still get a clean classified error — not a panic, not a silent
    /// success that would corrupt off-mode state.
    #[tokio::test]
    async fn noop_checkpoint_store_every_method_returns_not_implemented() {
        let store = NoopCheckpointStore;

        // Layer-state methods (Epic 1 surface — `NoopCheckpointStore` opts
        // out of even the layer methods because the operator chose off-mode).
        for (label, err) in [
            (
                "put_layer_checkpoint",
                store
                    .put_layer_checkpoint(1, 1, Bytes::from_static(b"x"), test_meta())
                    .await
                    .err(),
            ),
            (
                "get_latest_layer_checkpoint",
                store.get_latest_layer_checkpoint(1).await.err(),
            ),
            (
                // Inherited trait default — `NoopCheckpointStore` deliberately
                // does not override it, so this pins that the default is
                // `NotImplemented` and not an `Ok(None)` an off-mode caller
                // would read as "no such generation".
                "get_layer_checkpoint",
                store
                    .get_layer_checkpoint(&CheckpointHandle::base(1, 1, "noop".into()))
                    .await
                    .err(),
            ),
            (
                "list_layer_checkpoints",
                store.list_layer_checkpoints(1).await.err(),
            ),
            (
                "delete_layer_checkpoint",
                store
                    .delete_layer_checkpoint(CheckpointHandle::base(1, 1, "noop".into()))
                    .await
                    .err(),
            ),
            (
                "put_session_checkpoint",
                store
                    .put_session_checkpoint("s", Bytes::from_static(b"x"), test_meta())
                    .await
                    .err(),
            ),
            (
                "get_session_checkpoint",
                store.get_session_checkpoint("s").await.err(),
            ),
            (
                "delete_session_checkpoint",
                store.delete_session_checkpoint("s").await.err(),
            ),
            (
                // Inherited trait default, like `get_layer_checkpoint`: pins
                // that an off-mode store answers NotImplemented, never the
                // `Ok(vec![])` the DR restore Job would read as "no session
                // backups".
                "list_session_checkpoint_ids",
                store.list_session_checkpoint_ids().await.err(),
            ),
            (
                "put_ext_blob",
                store
                    .put_ext_blob("hash", Bytes::from_static(b"x"))
                    .await
                    .err(),
            ),
            ("get_ext_blob", store.get_ext_blob("hash").await.err()),
            (
                // Inherited trait default: NotImplemented, never the
                // `Ok(false)` the DR restore Job would read as "bytes absent".
                "ext_blob_exists",
                store.ext_blob_exists("hash").await.err(),
            ),
            ("delete_ext_blob", store.delete_ext_blob("hash").await.err()),
            (
                "put_counters_sidecar",
                store
                    .put_counters_sidecar(Bytes::from_static(b"x"))
                    .await
                    .err(),
            ),
            (
                "get_counters_sidecar",
                store.get_counters_sidecar().await.err(),
            ),
            (
                "list_all_object_ids",
                store.list_all_object_ids().await.err(),
            ),
            (
                // Default-impl propagation: the trait's
                // `list_referenced_ext_hashes` calls `list_all_object_ids`,
                // which returns `NotImplemented` for this adapter, so the
                // error must propagate through with the same classification.
                "list_referenced_ext_hashes",
                store.list_referenced_ext_hashes().await.err(),
            ),
            (
                // Default-impl propagation, one level further: the trait's
                // `list_live_ext_set` wraps `list_referenced_ext_hashes`, so
                // the same `NotImplemented` must reach the reconciliation
                // collect rather than an empty live set.
                "list_live_ext_set",
                store.list_live_ext_set().await.err(),
            ),
        ] {
            let err = err.unwrap_or_else(|| {
                panic!("{label} on NoopCheckpointStore must Err with NotImplemented")
            });
            assert!(
                matches!(err, CheckpointError::NotImplemented { .. }),
                "{label} must classify as NotImplemented, got {err:?}"
            );
        }
    }

    /// Each `NotImplemented` carries the method name so operator logs name
    /// the call that surfaced the off-mode error. Verify the label-to-method
    /// mapping is wired correctly for the layer-state surface (the methods
    /// the Epic-1 checkpoint loop and recovery hook would invoke) and for the
    /// DR restore Job's two reads (`list_session_checkpoint_ids`,
    /// `ext_blob_exists`), whose names come from the trait defaults.
    #[tokio::test]
    async fn noop_checkpoint_store_not_implemented_carries_method_name() {
        let store = NoopCheckpointStore;

        let err = store
            .put_layer_checkpoint(0, 0, Bytes::from_static(b""), test_meta())
            .await
            .expect_err("put_layer_checkpoint must Err");
        match err {
            CheckpointError::NotImplemented { method } => {
                assert_eq!(method, "put_layer_checkpoint");
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }

        let err = store
            .get_latest_layer_checkpoint(0)
            .await
            .expect_err("get_latest_layer_checkpoint must Err");
        match err {
            CheckpointError::NotImplemented { method } => {
                assert_eq!(method, "get_latest_layer_checkpoint");
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }

        // `get_layer_checkpoint` carries its name from the *trait default*,
        // not from a hand-written arm here — the label has to survive the
        // inherited path too, or an operator log naming the off-mode call is
        // silently wrong for the one method the shim does not override.
        let err = store
            .get_layer_checkpoint(&CheckpointHandle::base(0, 0, String::new()))
            .await
            .expect_err("get_layer_checkpoint must Err");
        match err {
            CheckpointError::NotImplemented { method } => {
                assert_eq!(method, "get_layer_checkpoint");
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }

        // The two DR restore Job methods also inherit their trait defaults,
        // so each name, too, comes from the default rather than a hand-written
        // arm.
        let err = store
            .list_session_checkpoint_ids()
            .await
            .expect_err("list_session_checkpoint_ids must Err");
        match err {
            CheckpointError::NotImplemented { method } => {
                assert_eq!(method, "list_session_checkpoint_ids");
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }

        let err = store
            .ext_blob_exists("hash")
            .await
            .expect_err("ext_blob_exists must Err");
        match err {
            CheckpointError::NotImplemented { method } => {
                assert_eq!(method, "ext_blob_exists");
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    /// A decorator that forwards only the required `CheckpointStore` methods
    /// over a working store, and so inherits the trait defaults of the two
    /// DR restore Job methods.
    ///
    /// The shape the trait rustdoc warns about: a wrapper written before
    /// `list_session_checkpoint_ids` / `ext_blob_exists` existed, or one that
    /// forgot them.
    struct RequiredOnlyDecorator {
        inner: InMemoryCheckpointStore,
    }

    #[async_trait]
    impl CheckpointStore for RequiredOnlyDecorator {
        async fn put_layer_checkpoint(
            &self,
            object_id: u64,
            snap_seq: u64,
            ompf: Bytes,
            meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            self.inner
                .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
                .await
        }

        async fn get_latest_layer_checkpoint(
            &self,
            object_id: u64,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            self.inner.get_latest_layer_checkpoint(object_id).await
        }

        async fn list_layer_checkpoints(
            &self,
            object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            self.inner.list_layer_checkpoints(object_id).await
        }

        async fn delete_layer_checkpoint(
            &self,
            handle: CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            self.inner.delete_layer_checkpoint(handle).await
        }

        async fn put_session_checkpoint(
            &self,
            session_id: &str,
            body: Bytes,
            meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            self.inner
                .put_session_checkpoint(session_id, body, meta)
                .await
        }

        async fn get_session_checkpoint(
            &self,
            session_id: &str,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            self.inner.get_session_checkpoint(session_id).await
        }

        async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
            self.inner.delete_session_checkpoint(session_id).await
        }

        async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
            self.inner.put_ext_blob(hash_hex, bytes).await
        }

        async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
            self.inner.get_ext_blob(hash_hex).await
        }

        async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
            self.inner.delete_ext_blob(hash_hex).await
        }

        async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
            self.inner.put_counters_sidecar(body).await
        }

        async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
            self.inner.get_counters_sidecar().await
        }

        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            self.inner.list_all_object_ids().await
        }
    }

    /// A decorator that does not forward `list_session_checkpoint_ids` and
    /// `ext_blob_exists` answers `NotImplemented` naming each method, even
    /// though the store behind it holds a session backup and the blob's bytes.
    ///
    /// The inner store answers both methods for real (checked first), so the
    /// `NotImplemented` can only come from the decorator's missing arm, not
    /// from an empty inner store. `Ok(vec![])` or `Ok(false)` here would be
    /// the silent failure the defaults exist to prevent: the DR restore Job
    /// would see "no session backups" and "bytes absent" for a store that has
    /// both.
    #[tokio::test]
    async fn a_decorator_that_does_not_forward_the_dr_restore_methods_answers_not_implemented_for_both(
    ) {
        const SESSION_ID: &str = "sess-held-by-the-inner-store";
        const HASH_HEX: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

        let decorator = RequiredOnlyDecorator {
            inner: InMemoryCheckpointStore::new(),
        };
        decorator
            .put_session_checkpoint(SESSION_ID, Bytes::from_static(b"session"), test_meta())
            .await
            .expect("the forwarded session put must land in the inner store");
        decorator
            .put_ext_blob(HASH_HEX, Bytes::from_static(b"ext-bytes"))
            .await
            .expect("the forwarded ext put must land in the inner store");

        // Precondition: the inner store answers both for real.
        assert_eq!(
            decorator
                .inner
                .list_session_checkpoint_ids()
                .await
                .expect("the inner store implements the session listing"),
            vec![SESSION_ID.to_owned()],
        );
        assert!(
            decorator
                .inner
                .ext_blob_exists(HASH_HEX)
                .await
                .expect("the inner store implements the existence probe"),
            "the inner store holds the blob's bytes"
        );

        match decorator.list_session_checkpoint_ids().await {
            Err(CheckpointError::NotImplemented { method }) => {
                assert_eq!(method, "list_session_checkpoint_ids");
            }
            other => panic!(
                "a decorator that does not forward list_session_checkpoint_ids must answer \
                 NotImplemented, never a list it did not read from the inner store; got \
                 {other:?}"
            ),
        }
        match decorator.ext_blob_exists(HASH_HEX).await {
            Err(CheckpointError::NotImplemented { method }) => {
                assert_eq!(method, "ext_blob_exists");
            }
            other => panic!(
                "a decorator that does not forward ext_blob_exists must answer NotImplemented, \
                 never a bool it did not read from the inner store; got {other:?}"
            ),
        }
    }

    /// `build_checkpoint_store(InMemory)` must return a `CheckpointStore`
    /// whose read paths work — not a `NoopCheckpointStore`. This pins the
    /// factory's match-arm wiring so a future refactor can't silently swap
    /// the InMemory arm to a no-op without breaking this test.
    #[tokio::test]
    async fn build_checkpoint_store_in_memory_returns_working_adapter() {
        // Use temp-env so the test does not leak global env-var mutations.
        let cfg = temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("InMemory")),
            ],
            || Config::from_env_with(PodMemorySource::Reference).expect("Config::from_env"),
        );

        let store = build_checkpoint_store(&cfg, None, None)
            .await
            .expect("InMemory factory must succeed");

        // A read on a fresh InMemory adapter is `Ok(None)`, NOT
        // `NotImplemented` — proves the factory wired the real adapter
        // rather than the no-op shim.
        let result = store
            .get_latest_layer_checkpoint(999)
            .await
            .expect("InMemory get_latest_layer_checkpoint must be Ok on empty adapter");
        assert!(
            result.is_none(),
            "fresh InMemory adapter must report no checkpoint, not NotImplemented"
        );
    }

    /// `build_checkpoint_store(Disabled)` must hand back the no-op shim —
    /// any read or write surfaces `NotImplemented`. This is the off-mode
    /// defense-in-depth contract documented on the factory and on
    /// `NoopCheckpointStore`.
    #[tokio::test]
    async fn build_checkpoint_store_disabled_returns_noop_shim() {
        let cfg = temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("Disabled")),
            ],
            || Config::from_env_with(PodMemorySource::Reference).expect("Config::from_env"),
        );

        let store = build_checkpoint_store(&cfg, None, None)
            .await
            .expect("Disabled factory must succeed (it's just an Arc allocation)");

        // Hitting any method MUST surface `NotImplemented`, not a silent
        // success.
        let err = store
            .get_latest_layer_checkpoint(1)
            .await
            .expect_err("Disabled adapter must Err on get_latest_layer_checkpoint");
        assert!(
            matches!(err, CheckpointError::NotImplemented { .. }),
            "Disabled adapter must classify as NotImplemented, got {err:?}"
        );
    }

    /// `build_checkpoint_store(LocalDisk)` must construct a working
    /// LocalDisk adapter rooted at the configured path. Validates that the
    /// factory threads `checkpoint_local_disk_path` through to the adapter
    /// constructor (the source-of-truth path mapping documented on the
    /// match arm).
    #[tokio::test]
    async fn build_checkpoint_store_local_disk_threads_root_through() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let root_str = root.to_string_lossy().into_owned();

        let cfg = temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("LocalDisk")),
                (
                    "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH",
                    Some(root_str.as_str()),
                ),
            ],
            || Config::from_env_with(PodMemorySource::Reference).expect("Config::from_env"),
        );

        let store = build_checkpoint_store(&cfg, None, None)
            .await
            .expect("LocalDisk factory must succeed on a writable tempdir");

        // Indirect proof the constructor used `root`: the namespace
        // subdirectory tree must have landed under `<root>/.live-edit/`.
        // (Direct introspection would need a downcast we don't expose; the
        // filesystem side-effect is the observable contract.)
        let ns = root.join(".live-edit");
        assert!(
            ns.join("layers").is_dir(),
            "LocalDisk constructor must have created <root>/.live-edit/layers/ at {}",
            ns.display()
        );

        // And the adapter must dispatch through the trait — not the no-op
        // shim. Fresh LocalDisk has no checkpoints, so the read is `Ok(None)`.
        let result = store
            .get_latest_layer_checkpoint(1)
            .await
            .expect("LocalDisk get on fresh tree must be Ok, not Err");
        assert!(result.is_none());
    }

    /// `build_checkpoint_store(LocalDisk)` with `checkpoint_local_disk_path =
    /// None` must surface an explicit error naming the missing field. The
    /// primary gate lives in `Config::from_env`; this test pins the
    /// defense-in-depth fallback so a future caller that constructs a
    /// `Config` directly (bypassing env parsing) does not silently fall back
    /// to creating `.live-edit/{layers,sessions,ext}/` under the service's
    /// current working directory.
    #[tokio::test]
    async fn build_checkpoint_store_local_disk_missing_path_returns_error() {
        // Construct a `Config` directly to bypass the env-var gate — the
        // whole point of this test is to exercise the in-factory check.
        let mut cfg = temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("InMemory")),
            ],
            || Config::from_env_with(PodMemorySource::Reference).expect("Config::from_env"),
        );
        // Mutate after parse: simulate a caller that swaps the adapter to
        // `LocalDisk` without populating the path, which is exactly the
        // bypass scenario the in-factory check guards against.
        cfg.checkpoint_adapter = CheckpointAdapter::LocalDisk;
        cfg.checkpoint_local_disk_path = None;

        // `Arc<dyn CheckpointStore>` does not implement `Debug`, so
        // `expect_err` will not compile here — match the `Result` manually.
        let err = match build_checkpoint_store(&cfg, None, None).await {
            Ok(_) => panic!("missing checkpoint_local_disk_path must Err"),
            Err(e) => e,
        };
        assert!(
            err.contains("checkpoint_local_disk_path"),
            "error message must name the missing field; got {err:?}"
        );
    }

    /// `build_checkpoint_store(StorageApi)` with a fully-populated `Config` +
    /// a real `Arc<ServiceTokenProvider>` must return `Ok(Arc<dyn
    /// CheckpointStore>)` backed by `StorageApiCheckpointStore`, AND the
    /// returned trait object must dispatch through to the real adapter
    /// rather than a no-op shim. This pins the dispatcher's StorageApi arm
    /// to its real-adapter contract — the struct constructs (proving
    /// Task-1 Config wiring + the constructor in `t3.fgj.jd.8e.o5` are
    /// correctly threaded), the `impl CheckpointStore for
    /// StorageApiCheckpointStore` block (Task 3 of Epic `t1.fgj.jd`)
    /// satisfies the trait object so `Arc::new(store) as Arc<dyn
    /// CheckpointStore>` succeeds, AND a method call on the trait object
    /// reaches the StorageApi impl (verified by an outbound LIST against a
    /// `wiremock::MockServer` answering the 404 storage-api returns for an
    /// empty prefix, which `get_latest_layer_checkpoint` must surface as
    /// `Ok(None)`).
    ///
    /// A no-op shim would return `Err(NotImplemented)` here; the empty-LIST
    /// `Ok(None)` is the positive signal that the StorageApi adapter's
    /// `list_layer_generations_via_enumerate` (which yields the object's
    /// `(sequence, kind)` generations, newest-first under the shared
    /// `generation_sort_key`) + `get_latest_layer_checkpoint` actually ran.
    #[tokio::test]
    async fn build_checkpoint_store_storage_api_constructs_working_adapter() {
        use crate::auth::service_token::ServiceTokenProvider;
        use crate::observability::metrics::metrics_scrape_only;
        use tokio_util::sync::CancellationToken;
        use wiremock::matchers::{method, path_regex};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Stand up a fresh mock storage-api endpoint. The dispatcher's
        // StorageApi arm will point the real adapter at this URI; the LIST
        // mock below proves the trait dispatches through the real adapter
        // (not the no-op shim).
        let mock_server = MockServer::start().await;

        // LIST mock — a 404 with no body, storage-api's answer for an
        // Enumerate of a prefix that holds no objects, so
        // `get_latest_layer_checkpoint(999)` traverses the no-checkpoints
        // branch in `StorageApiCheckpointStore` and returns `Ok(None)`.
        // The path-regex matches the Enumerate URL shape rendered by
        // `data_objects_list_url` (`/v1alpha/fileobject/data-objects/...`);
        // we intentionally don't pin the percent-encoded prefix because
        // this test pins dispatch, not URL shape (sibling adapter-level
        // tests pin URL shape).
        Mock::given(method("GET"))
            .and(path_regex(r"/v1alpha/fileobject/data-objects/"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&mock_server)
            .await;

        // Write a syntactically-valid token file so the provider constructs.
        // `ServiceTokenProvider::new` does not block on the first read —
        // construction succeeds even if the file does not exist (the polling
        // task tolerates `NotFound`). We still write a valid stub so future
        // refactors that promote construction-time validation don't silently
        // skip this test path.
        let token_dir = tempfile::tempdir().expect("token tempdir");
        let token_path = token_dir.path().join("service-token.json");
        let expires_on = chrono::Utc::now() + chrono::Duration::seconds(3600);
        let stub = format!(
            r#"{{"access_token":"stub-token","expires_on":"{}"}}"#,
            expires_on.to_rfc3339()
        );
        std::fs::write(&token_path, stub).expect("write stub token file");

        let mock_uri = mock_server.uri();
        let cfg = temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("StorageApi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some(mock_uri.as_str()),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("/projects/team-a")),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some(token_path.to_str().expect("utf-8 token path")),
                ),
            ],
            || Config::from_env_with(PodMemorySource::Reference).expect("Config::from_env"),
        );

        // Build a real `Arc<ServiceTokenProvider>` — the production path
        // (`main.rs`) constructs one against the operator-configured file
        // when `LIVE_EDIT_SERVICE_TOKEN_FILE` is set, so the factory test
        // exercises the same Arc-of-provider surface rather than mocking it.
        let metrics = Arc::new(metrics_scrape_only("builder-test"));
        let cancel = CancellationToken::new();
        let provider = Arc::new(ServiceTokenProvider::new(
            token_path.clone(),
            Duration::from_secs(60),
            metrics,
            cancel.clone(),
        ));

        // The factory must succeed and hand back an `Arc<dyn
        // CheckpointStore>`. We don't downcast to the concrete adapter
        // (the trait-object boundary is the contract under test) —
        // success of `build_checkpoint_store` is the construction signal.
        let store = build_checkpoint_store(&cfg, Some(provider), None)
            .await
            .expect("StorageApi factory must succeed with all required Config + provider");

        // Trait-dispatch round-trip: calling `get_latest_layer_checkpoint`
        // on the trait object must reach the StorageApi impl, which issues
        // an outbound LIST against the mock server. The empty-prefix 404
        // surfaces as `Ok(None)`. A no-op shim would instead return
        // `Err(NotImplemented)`, so `Ok(None)` is the positive proof that
        // the trait dispatches through the real adapter.
        let result = store
            .get_latest_layer_checkpoint(999)
            .await
            .expect("StorageApi trait dispatch must yield Ok on empty LIST");
        assert!(
            result.is_none(),
            "an empty-prefix LIST 404 must surface as Ok(None) via the real adapter, \
             not Err(NotImplemented) from a no-op shim"
        );

        // Shut the polling task down cleanly so it does not outlive the
        // test (the tempdir would otherwise be removed while the polling
        // task is still alive — harmless on Unix but noisy on Windows).
        cancel.cancel();
    }

    /// `build_checkpoint_store(StorageApi)` with the env vars set but the
    /// `Arc<ServiceTokenProvider>` arg missing must surface an explicit error
    /// naming `LIVE_EDIT_SERVICE_TOKEN_FILE`. The primary gate is
    /// `Config::from_env`, which refuses `StorageApi` without the file env
    /// var; `main.rs` constructs the provider only when it is set, and the
    /// dispatcher's defense-in-depth check refuses to wire the StorageApi
    /// adapter without one. Pin the error message so a caller that bypasses
    /// the primary gate still gets the actionable env-var name rather than a
    /// generic "service_token_provider missing" string that would force the
    /// reader to grep the source tree.
    #[tokio::test]
    async fn build_checkpoint_store_storage_api_missing_service_token_file_returns_error() {
        let cfg = temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("StorageApi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("/projects/team-a")),
                // Set only so `Config::from_env`'s primary gate lets the
                // StorageApi branch parse; no provider is built from it.
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
            ],
            || Config::from_env_with(PodMemorySource::Reference).expect("Config::from_env"),
        );

        // Pass `None` for the provider — this is exactly the bypass
        // scenario the in-factory check guards against (a caller reached
        // the factory without the provider `main.rs` builds from the token
        // file).
        let err = match build_checkpoint_store(&cfg, None, None).await {
            Ok(_) => panic!("missing service_token_provider on the StorageApi arm must Err"),
            Err(e) => e,
        };
        assert!(
            err.contains("LIVE_EDIT_SERVICE_TOKEN_FILE"),
            "error message must name the missing env var so an operator \
             gets an actionable startup message; got {err:?}"
        );
    }

    /// b.gg7 — the two faces of the adapter call ceiling cannot disagree.
    ///
    /// `Config::checkpoint_adapter_call_ceiling()` is the face reachable from
    /// a `Config` alone (startup validation, the config-echo gauges, the
    /// startup INFO line); `CheckpointStore::publish_worst_case()` is the face
    /// the constructed adapter reports and — on StorageApi — the number it
    /// actually installs as a `tokio::time::timeout`. Both route through the
    /// same expression today. Nothing in the *type system* keeps them there:
    /// they live in different modules, have different signatures, and a future
    /// change to either is a one-line edit that compiles.
    ///
    /// So the assertion is deliberately made against a **constructed adapter**
    /// rather than by re-evaluating the arithmetic — comparing an expression
    /// to itself would pass no matter how far the two drifted. And at a
    /// **non-default** request timeout, because the default is the one value
    /// where a frozen constant and live arithmetic agree by accident.
    ///
    /// Both arms of the derivation are covered: StorageApi (retry product) and
    /// a local adapter (`LOCAL_PUBLISH_WORST_CASE`, which is what the seam must
    /// report for an adapter that installs no timeout of its own).
    #[tokio::test]
    async fn build_checkpoint_store_publish_worst_case_agrees_with_the_config_call_ceiling() {
        use crate::auth::service_token::ServiceTokenProvider;
        use crate::observability::metrics::metrics_scrape_only;
        use tokio_util::sync::CancellationToken;

        // ── StorageApi, at half the default request timeout ──────────────
        //
        // 15 s × 3 transient retries × 2 auth attempts = 90 s, not the 180 s
        // the defaults produce. The reconcile interval has to clear the
        // derived grace (3 × 90 = 270 s) or `from_env` rejects the config,
        // which is itself the third consumer of this same expression.
        let token_dir = tempfile::tempdir().expect("token tempdir");
        let token_path = token_dir.path().join("service-token.json");
        let expires_on = chrono::Utc::now() + chrono::Duration::seconds(3600);
        std::fs::write(
            &token_path,
            format!(
                r#"{{"access_token":"stub-token","expires_on":"{}"}}"#,
                expires_on.to_rfc3339()
            ),
        )
        .expect("write stub token file");

        let cfg = temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("StorageApi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("/projects/team-a")),
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", Some("15000")),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some(token_path.to_str().expect("utf-8 token path")),
                ),
            ],
            || Config::from_env_with(PodMemorySource::Reference).expect("Config::from_env"),
        );
        assert_eq!(
            cfg.checkpoint_adapter_call_ceiling(),
            Duration::from_secs(90),
            "sanity: the Config face must track the configured request timeout, not the default"
        );

        let cancel = CancellationToken::new();
        let provider = Arc::new(ServiceTokenProvider::new(
            token_path.clone(),
            Duration::from_secs(60),
            Arc::new(metrics_scrape_only("builder-test")),
            cancel.clone(),
        ));
        // `Some(metrics)` exercises the production wiring of the b.gg7
        // ceiling-counter handle through the factory's StorageApi arm.
        let store = build_checkpoint_store(
            &cfg,
            Some(provider),
            Some(Arc::new(metrics_scrape_only("builder-test"))),
        )
        .await
        .expect("StorageApi factory must succeed");

        assert_eq!(
            store.publish_worst_case(),
            cfg.checkpoint_adapter_call_ceiling(),
            "the adapter the factory built must report the very ceiling the Config validated \
             the reconcile interval against and the config-echo gauge advertises — a drift here \
             means the alert threshold an operator wrote is not the bound the code enforces"
        );
        cancel.cancel();

        // ── The local arm, where the ceiling is a floor rather than a product.
        let local_cfg = temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("InMemory")),
                // Deliberately set, and deliberately irrelevant: the local
                // arms must NOT pick the StorageApi product up.
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", Some("15000")),
            ],
            || Config::from_env_with(PodMemorySource::Reference).expect("Config::from_env"),
        );
        let local_store = build_checkpoint_store(&local_cfg, None, None)
            .await
            .expect("InMemory factory must succeed");
        assert_eq!(
            local_store.publish_worst_case(),
            local_cfg.checkpoint_adapter_call_ceiling(),
            "the same agreement must hold on an adapter that installs no timeout — the Config \
             face has to report what the adapter reports, not a StorageApi product it never runs"
        );
    }
}
