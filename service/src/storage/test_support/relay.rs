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

//! `UpdateLayer` commits that relay the delta's own OMPF, for tests.
//!
//! Every delta-carrying `LayerStore::update_layer*` call must pass the
//! writer's OMPF as non-empty relay bytes (the relay precondition of the
//! `ompf`-presence invariant, SDD §12.2, documented on
//! `LayerStore::update_layer_with_operation`); a call without them is refused
//! `Internal` before any store I/O. The `UpdateLayer` handler derives both
//! values from the one request body. A test that builds an [`OwnedDelta`]
//! directly has no body, so [`RelayedUpdates`] encodes one: the delta built
//! with `oodf_ffi::build_delta` inside an OMPF container with no
//! external-value parts ([`relay_ompf`]) — the container a client sends,
//! minus the external blobs. A `None` delta (an empty commit) relays nothing.
//!
//! Leaving the external parts out is faithful only for a reader that ignores
//! them: the structural cache's replay (it leaves `ext_value_index`
//! unresolved), the `commit_chain` fold (it reads the entry's `payload`) and
//! keyspace dumps. Two stream readers do resolve them. Checkpoint tail
//! compose (`decode_tail_record_ompf`, step 4) resolves each external
//! reference against the record's own OMPF parts (`collect_external_values`)
//! and refuses a missing one as `ComposeError::ExtValueIndex`.
//! `SubscribeReadLayer` relays the container verbatim to clients, who
//! resolve it themselves. So a delta that references an external value, on
//! an entry that tail capture captures or a subscriber receives, must have
//! its parts passed directly, not through [`relay_ompf`].
//!
//! A test whose assertion depends on the relay bytes themselves — byte
//! parity with a client's container, its external-value parts — builds them
//! and passes them to `update_layer*` directly.

use std::future::Future;

use bytes::Bytes;
use oodf_ffi::OwnedDelta;

use crate::domain::ompf::OmpfContainer;
use crate::error::ServiceError;
use crate::storage::layer::{LayerStore, Operation, UpdateCommit, UpdateLayerCallSummary};

/// `delta` encoded as the OMPF container a client sends for it, with no
/// external-value parts.
pub fn relay_ompf(delta: &OwnedDelta) -> Bytes {
    let delta_bytes =
        oodf_ffi::build_delta(delta).expect("a test's idiomatic OwnedDelta always builds");
    Bytes::from(
        OmpfContainer::serialize(&delta_bytes, &[])
            .expect("a container with no external parts always serializes"),
    )
}

/// The relay bytes a call with `delta` passes: [`relay_ompf`] of the delta,
/// none for an empty commit.
pub fn relay_for(delta: &Option<OwnedDelta>) -> Option<Bytes> {
    delta.as_ref().map(relay_ompf)
}

/// `LayerStore::update_layer*`, each relaying [`relay_for`] the delta in place
/// of a relay argument.
pub trait RelayedUpdates {
    /// `LayerStore::update_layer` with the delta's relay bytes.
    fn update_layer_relayed<'a>(
        &'a self,
        session_id: &'a str,
        layer_name: &'a str,
        object_id: u64,
        delta: Option<OwnedDelta>,
        client_id: &'a str,
        client_timestamp: u64,
    ) -> impl Future<Output = Result<(u64, u64), ServiceError>> + Send + 'a;

    /// `LayerStore::update_layer_with_operation` with the delta's relay
    /// bytes.
    #[allow(clippy::too_many_arguments)]
    fn update_layer_with_operation_relayed<'a>(
        &'a self,
        session_id: &'a str,
        layer_name: &'a str,
        object_id: u64,
        delta: Option<OwnedDelta>,
        client_id: &'a str,
        client_timestamp: u64,
        operation: Option<Operation>,
    ) -> impl Future<Output = Result<UpdateCommit, ServiceError>> + Send + 'a;

    /// `LayerStore::update_layer_with_summary` with the delta's relay bytes.
    #[allow(clippy::too_many_arguments)]
    fn update_layer_with_summary_relayed<'a>(
        &'a self,
        session_id: &'a str,
        layer_name: &'a str,
        object_id: u64,
        delta: Option<OwnedDelta>,
        client_id: &'a str,
        client_timestamp: u64,
        operation: Option<Operation>,
    ) -> impl Future<
        Output = (
            Result<UpdateCommit, ServiceError>,
            Option<UpdateLayerCallSummary>,
        ),
    > + Send
           + 'a;
}

impl RelayedUpdates for LayerStore {
    fn update_layer_relayed<'a>(
        &'a self,
        session_id: &'a str,
        layer_name: &'a str,
        object_id: u64,
        delta: Option<OwnedDelta>,
        client_id: &'a str,
        client_timestamp: u64,
    ) -> impl Future<Output = Result<(u64, u64), ServiceError>> + Send + 'a {
        let relay = relay_for(&delta);
        self.update_layer(
            session_id,
            layer_name,
            object_id,
            delta,
            relay,
            client_id,
            client_timestamp,
        )
    }

    fn update_layer_with_operation_relayed<'a>(
        &'a self,
        session_id: &'a str,
        layer_name: &'a str,
        object_id: u64,
        delta: Option<OwnedDelta>,
        client_id: &'a str,
        client_timestamp: u64,
        operation: Option<Operation>,
    ) -> impl Future<Output = Result<UpdateCommit, ServiceError>> + Send + 'a {
        let relay = relay_for(&delta);
        self.update_layer_with_operation(
            session_id,
            layer_name,
            object_id,
            std::sync::Arc::new(delta),
            relay,
            client_id,
            client_timestamp,
            operation,
        )
    }

    fn update_layer_with_summary_relayed<'a>(
        &'a self,
        session_id: &'a str,
        layer_name: &'a str,
        object_id: u64,
        delta: Option<OwnedDelta>,
        client_id: &'a str,
        client_timestamp: u64,
        operation: Option<Operation>,
    ) -> impl Future<
        Output = (
            Result<UpdateCommit, ServiceError>,
            Option<UpdateLayerCallSummary>,
        ),
    > + Send
           + 'a {
        let relay = relay_for(&delta);
        async move {
            self.update_layer_with_summary(
                session_id,
                layer_name,
                object_id,
                std::sync::Arc::new(delta),
                relay,
                client_id,
                client_timestamp,
                operation,
                &mut crate::storage::commit_charge::NoCommitCharge,
            )
            .await
        }
    }
}
