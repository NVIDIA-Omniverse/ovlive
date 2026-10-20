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

//! Construction helpers for the external auth clients (permissions + userinfo).
//!
//! These helpers dispatch on [`AuthProtocol`] so `main.rs` and integration
//! tests can share a single entry point for building the runtime client
//! behind each `dyn` trait object. Error handling (logging + process exit)
//! is left to the caller — the helpers return `Result<_, AuthError>` and
//! let `main.rs::run` decide whether to fall over or surface the error.

use std::sync::Arc;

use crate::auth::permissions::PermissionsClient;
use crate::auth::userinfo::UserinfoClient;
use crate::auth::{
    AuthError, GrpcStorageServiceClient, GrpcUserinfoClient, JwtUserinfoClient,
    NoOpPermissionsClient, RestStorageServiceClient, RestUserinfoClient, RetryConfig,
    UserinfoFallbackClient,
};
use crate::config::{AuthProtocol, Config};
use crate::observability::metrics::SharedMetrics;

/// Build the permissions client used by the auth cache.
///
/// When `!config.authz_enabled`, returns [`NoOpPermissionsClient`] regardless
/// of the selected protocol (SDD §14.5). Otherwise the protocol field picks
/// between the REST and gRPC implementations; both receive the storage
/// service endpoint, per-attempt timeout, and retry configuration.
pub async fn build_permissions_client(
    config: &Config,
    retry: &RetryConfig,
) -> Result<Arc<dyn PermissionsClient>, AuthError> {
    if !config.authz_enabled {
        tracing::info!("authorization disabled — using no-op permissions client");
        return Ok(Arc::new(NoOpPermissionsClient));
    }

    let endpoint = config
        .storage_service_endpoint
        .as_deref()
        .expect("storage_service_endpoint validated by Config::from_env");

    match config.auth_protocol {
        AuthProtocol::Rest => {
            tracing::info!(protocol = "rest", endpoint, "connecting to storage service");
            Ok(Arc::new(RestStorageServiceClient::new(
                endpoint,
                config.auth_timeout_ms,
                retry.clone(),
            )?))
        }
        AuthProtocol::Grpc => {
            tracing::info!(protocol = "grpc", endpoint, "connecting to storage service");
            Ok(Arc::new(
                GrpcStorageServiceClient::connect(endpoint, config.auth_timeout_ms, retry.clone())
                    .await?,
            ))
        }
    }
}

/// Build the userinfo client used by the auth cache.
///
/// When `!config.userinfo_service_enabled`, returns [`JwtUserinfoClient`]
/// which extracts caller identity directly from the bearer JWT payload
/// instead of consulting an external service (SR-3.1). Otherwise the
/// protocol field picks between the REST and gRPC userinfo backends; both
/// receive the userinfo endpoint, per-attempt timeout, and retry config.
///
/// With authorization also on, the backend is wrapped in
/// [`UserinfoFallbackClient`], which answers the userinfo service's "no such
/// user" outcome from the bearer's JWT claims and counts it on `metrics`. The
/// gate is authorization because the claim decode trusts the bearer only
/// once the permissions service has validated it on the same RPC (SDD
/// §14.3); with authorization off nothing has, so the bare backend is
/// returned and that outcome keeps failing the RPC. The test-only
/// unvalidated-tokens hatch needs the userinfo service off, so it never
/// reaches this branch.
pub async fn build_userinfo_client(
    config: &Config,
    retry: &RetryConfig,
    metrics: SharedMetrics,
) -> Result<Arc<dyn UserinfoClient>, AuthError> {
    if !config.userinfo_service_enabled {
        tracing::info!("userinfo service disabled — extracting identity from JWT payload");
        return Ok(Arc::new(JwtUserinfoClient::new()));
    }

    // `userinfo_endpoint` is `Some` whenever `userinfo_service_enabled = true`
    // because `Config::from_env` enforces that invariant (SR-1.3) — the
    // early-return above guards the `None` case.
    let endpoint = config
        .userinfo_endpoint
        .as_deref()
        .expect("userinfo_endpoint validated by Config::from_env when userinfo_service_enabled");

    let backend: Arc<dyn UserinfoClient> = match config.auth_protocol {
        AuthProtocol::Rest => {
            tracing::info!(
                protocol = "rest",
                endpoint,
                "connecting to userinfo service"
            );
            Arc::new(RestUserinfoClient::new(
                endpoint,
                config.auth_timeout_ms,
                retry.clone(),
            )?)
        }
        AuthProtocol::Grpc => {
            tracing::info!(
                protocol = "grpc",
                endpoint,
                "connecting to userinfo service"
            );
            Arc::new(
                GrpcUserinfoClient::connect(endpoint, config.auth_timeout_ms, retry.clone())
                    .await?,
            )
        }
    };

    if config.authz_enabled {
        Ok(Arc::new(UserinfoFallbackClient::new(backend, metrics)))
    } else {
        Ok(backend)
    }
}
