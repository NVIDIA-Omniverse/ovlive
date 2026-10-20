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

use async_trait::async_trait;

use crate::auth::{AuthError, AuthResult, Permission};

/// Async trait for path-level read/write authorization checks.
///
/// Both methods are object-safe via `async_trait`. Implementations are
/// expected to be wrapped in `Arc<dyn PermissionsClient>`.
#[async_trait]
pub trait PermissionsClient: Send + Sync {
    async fn check_permission(
        &self,
        token: &str,
        path: &str,
        permission: Permission,
    ) -> Result<AuthResult, AuthError>;
}

/// No-op permissions client that always allows access.
///
/// Used in production when authorization is disabled (`authz_enabled = false`).
/// Distinct from [`StubPermissionsClient`](crate::auth::StubPermissionsClient),
/// which always errors and is intended for tests.
pub struct NoOpPermissionsClient;

#[async_trait]
impl PermissionsClient for NoOpPermissionsClient {
    async fn check_permission(
        &self,
        _token: &str,
        path: &str,
        permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        tracing::debug!(path, ?permission, "authz disabled — allowing request");
        Ok(AuthResult::Allowed)
    }
}

#[cfg(test)]
pub mod mock {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;

    use crate::auth::{AuthError, AuthResult, Permission};

    use super::PermissionsClient;

    /// A recorded call to [`MockPermissionsClient::check_permission`].
    #[derive(Debug, Clone)]
    pub struct PermissionsCall {
        pub token: String,
        pub path: String,
        pub permission: Permission,
    }

    /// Configurable mock for [`PermissionsClient`].
    ///
    /// Returns a pre-configured `Result` for every call and records all
    /// call arguments for later assertion.
    pub struct MockPermissionsClient {
        result: Result<AuthResult, AuthError>,
        pub calls: Arc<Mutex<Vec<PermissionsCall>>>,
    }

    impl MockPermissionsClient {
        pub fn new(result: Result<AuthResult, AuthError>) -> Self {
            Self {
                result,
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl PermissionsClient for MockPermissionsClient {
        async fn check_permission(
            &self,
            token: &str,
            path: &str,
            permission: Permission,
        ) -> Result<AuthResult, AuthError> {
            self.calls.lock().unwrap().push(PermissionsCall {
                token: token.to_string(),
                path: path.to_string(),
                permission,
            });
            match &self.result {
                Ok(r) => Ok(r.clone()),
                Err(e) => Err(e.clone()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::auth::permissions::PermissionsClient;
    use crate::auth::{AuthError, AuthResult, Permission};

    use super::mock::MockPermissionsClient;

    #[tokio::test]
    async fn mock_permissions_allowed_returns_allowed() {
        let client = MockPermissionsClient::new(Ok(AuthResult::Allowed));

        let result = client
            .check_permission("token", "/scene/a", Permission::Read)
            .await;

        assert_eq!(result.unwrap(), AuthResult::Allowed);
    }

    #[tokio::test]
    async fn mock_permissions_denied_returns_denied() {
        let client = MockPermissionsClient::new(Ok(AuthResult::Denied));

        let result = client
            .check_permission("token", "/scene/a", Permission::Write)
            .await;

        assert_eq!(result.unwrap(), AuthResult::Denied);
    }

    #[tokio::test]
    async fn mock_permissions_path_not_found_returns_path_not_found() {
        let client = MockPermissionsClient::new(Ok(AuthResult::PathNotFound));

        let result = client
            .check_permission("token", "/missing", Permission::Read)
            .await;

        assert_eq!(result.unwrap(), AuthResult::PathNotFound);
    }

    #[tokio::test]
    async fn mock_permissions_timeout_error_propagates() {
        let client = MockPermissionsClient::new(Err(AuthError::Timeout));

        let err = client
            .check_permission("token", "/scene/a", Permission::Read)
            .await
            .unwrap_err();

        assert!(matches!(err, AuthError::Timeout));
    }

    #[tokio::test]
    async fn mock_permissions_unreachable_error_propagates() {
        let client = MockPermissionsClient::new(Err(AuthError::Unreachable));

        let err = client
            .check_permission("token", "/scene/a", Permission::Read)
            .await
            .unwrap_err();

        assert!(matches!(err, AuthError::Unreachable));
    }

    #[tokio::test]
    async fn mock_permissions_invalid_response_error_propagates() {
        let client = MockPermissionsClient::new(Err(AuthError::InvalidResponse("bad json".into())));

        let err = client
            .check_permission("token", "/scene/a", Permission::Read)
            .await
            .unwrap_err();

        assert!(matches!(err, AuthError::InvalidResponse(msg) if msg == "bad json"));
    }

    #[tokio::test]
    async fn mock_permissions_records_call_arguments() {
        let client = MockPermissionsClient::new(Ok(AuthResult::Allowed));

        client
            .check_permission("bearer-abc", "/scene/layer1", Permission::Write)
            .await
            .unwrap();

        let calls = client.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].token, "bearer-abc");
        assert_eq!(calls[0].path, "/scene/layer1");
        assert_eq!(calls[0].permission, Permission::Write);
    }

    #[tokio::test]
    async fn mock_permissions_records_multiple_calls_in_order() {
        let client = MockPermissionsClient::new(Ok(AuthResult::Allowed));

        client
            .check_permission("token1", "/path/a", Permission::Read)
            .await
            .unwrap();
        client
            .check_permission("token2", "/path/b", Permission::Write)
            .await
            .unwrap();

        let calls = client.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].token, "token1");
        assert_eq!(calls[1].token, "token2");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mock_permissions_arc_dyn_sends_across_threads() {
        let client: Arc<dyn PermissionsClient> =
            Arc::new(MockPermissionsClient::new(Ok(AuthResult::Allowed)));

        let client_clone = Arc::clone(&client);
        let handle = tokio::spawn(async move {
            client_clone
                .check_permission("token", "/scene/a", Permission::Read)
                .await
        });

        let result = handle.await.unwrap();
        assert_eq!(result.unwrap(), AuthResult::Allowed);
    }

    // ── NoOpPermissionsClient tests (t3.5ci.ny.bg.8z) ─────────────────────

    use super::NoOpPermissionsClient;

    #[tokio::test]
    async fn noop_permissions_read_returns_allowed() {
        let client = NoOpPermissionsClient;

        let result = client
            .check_permission("token", "/scene/a", Permission::Read)
            .await;

        assert_eq!(result.unwrap(), AuthResult::Allowed);
    }

    #[tokio::test]
    async fn noop_permissions_write_returns_allowed() {
        let client = NoOpPermissionsClient;

        let result = client
            .check_permission("token", "/scene/a", Permission::Write)
            .await;

        assert_eq!(result.unwrap(), AuthResult::Allowed);
    }

    #[tokio::test]
    async fn noop_permissions_empty_token_returns_allowed() {
        let client = NoOpPermissionsClient;

        let result = client
            .check_permission("", "/scene/a", Permission::Read)
            .await;

        assert_eq!(result.unwrap(), AuthResult::Allowed);
    }

    #[tokio::test]
    async fn noop_permissions_any_path_returns_allowed() {
        let client = NoOpPermissionsClient;

        let result = client
            .check_permission("token", "/any/arbitrary/path", Permission::Write)
            .await;

        assert_eq!(result.unwrap(), AuthResult::Allowed);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn noop_permissions_arc_dyn_sends_across_threads() {
        let client: Arc<dyn PermissionsClient> = Arc::new(NoOpPermissionsClient);

        let client_clone = Arc::clone(&client);
        let handle = tokio::spawn(async move {
            client_clone
                .check_permission("token", "/scene/a", Permission::Read)
                .await
        });

        let result = handle.await.unwrap();
        assert_eq!(result.unwrap(), AuthResult::Allowed);
    }
}
