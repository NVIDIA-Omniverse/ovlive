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

pub mod auth_helpers;
pub mod auth_interceptor;
pub mod capabilities;
pub mod deadline;
pub mod graceful_shutdown;
pub mod layer;
pub mod metrics_layer;
pub mod ping;
pub mod presence;
pub mod request_end;
pub mod rpc_log;
pub mod service;
pub mod session;
pub mod session_cache;
pub mod streaming;
pub mod subscribe;
pub mod subscriber_delivery;
pub mod transport;
pub mod upload_admission;

/// Generated protobuf types and tonic service traits/servers.
pub mod proto {
    tonic::include_proto!("nvidia.omniverse.live.v1alpha");

    /// The binary file descriptor set compiled by `tonic-build` in build.rs.
    /// Used to enable gRPC server reflection.
    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("live_service_descriptor");
}

/// A bearer token extracted from the `authorization` request metadata header.
///
/// Inserted into request extensions by `auth_interceptor`. RPC handlers read
/// this to forward the token to the permissions and userinfo services.
///
/// Stored as [`Arc<str>`] so cloning per-RPC (which happens once at
/// `extract_token` time, plus zero-or-more times when the token is forwarded
/// to permissions / userinfo helpers) is a bumped refcount instead of a
/// fresh allocation + memcpy of the token bytes (b.ohw low).
///
/// The `Debug` impl intentionally redacts the token value — tokens must never
/// appear in logs or traces (SDD §23.3).
///
/// b.j5t item 7: the inner [`Arc<str>`] is **private** so callers cannot
/// `bearer.0` past the redacted `Debug`. Direct field access via the
/// pre-fix `pub Arc<str>` would let `format!("{}")` (`Display`) or
/// `bearer.0.to_string()` smuggle the token bytes into a log line; making
/// the field private channels all access through [`Self::as_str`] /
/// [`Self::arc`] / [`Self::new`], which keep the redaction contract intact.
#[derive(Clone)]
pub struct BearerToken(std::sync::Arc<str>);

impl BearerToken {
    /// Construct a `BearerToken` from a raw token string. Cheap — the
    /// `Arc<str>` allocation happens once at interceptor time.
    pub fn new(token: impl Into<std::sync::Arc<str>>) -> Self {
        Self(token.into())
    }

    /// Borrow the underlying token bytes. Use this when you need to pass
    /// the token to a backend client trait method that takes `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Clone the underlying `Arc<str>` to forward the token across an
    /// auth-cache key construction or a coalesced backend call. Bumps a
    /// refcount instead of allocating + copying the bytes.
    pub fn arc(&self) -> std::sync::Arc<str> {
        std::sync::Arc::clone(&self.0)
    }
}

impl std::fmt::Debug for BearerToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BearerToken").field(&"***").finish()
    }
}
