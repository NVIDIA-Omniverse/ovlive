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

use crate::auth::{AuthError, UserInfo};

/// Async trait for resolving a bearer token to user identity.
///
/// Implementations are expected to be wrapped in `Arc<dyn UserinfoClient>`.
#[async_trait]
pub trait UserinfoClient: Send + Sync {
    async fn get_current_user(&self, token: &str) -> Result<UserInfo, AuthError>;
}

#[cfg(test)]
pub mod mock {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;

    use crate::auth::{AuthError, UserInfo};

    use super::UserinfoClient;

    /// A recorded call to [`MockUserinfoClient::get_current_user`].
    #[derive(Debug, Clone)]
    pub struct UserinfoCall {
        pub token: String,
    }

    /// Configurable mock for [`UserinfoClient`].
    ///
    /// Returns a pre-configured `Result` for every call and records all
    /// call arguments for later assertion.
    pub struct MockUserinfoClient {
        result: Result<UserInfo, AuthError>,
        pub calls: Arc<Mutex<Vec<UserinfoCall>>>,
    }

    impl MockUserinfoClient {
        pub fn new(result: Result<UserInfo, AuthError>) -> Self {
            Self {
                result,
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl UserinfoClient for MockUserinfoClient {
        async fn get_current_user(&self, token: &str) -> Result<UserInfo, AuthError> {
            self.calls.lock().unwrap().push(UserinfoCall {
                token: token.to_string(),
            });
            match &self.result {
                Ok(info) => Ok(info.clone()),
                Err(e) => Err(e.clone()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::auth::userinfo::UserinfoClient;
    use crate::auth::{AuthError, UserInfo};

    use super::mock::MockUserinfoClient;

    fn make_user_info() -> UserInfo {
        UserInfo {
            id: "user-123".into(),
            display_name: "Alice Smith".into(),
            email: Some("alice@example.com".into()),
            username: "asmith".into(),
        }
    }

    #[tokio::test]
    async fn mock_userinfo_returns_all_fields_intact() {
        let expected = make_user_info();
        let client = MockUserinfoClient::new(Ok(expected.clone()));

        let result = client.get_current_user("bearer-token").await.unwrap();

        assert_eq!(result.id, expected.id);
        assert_eq!(result.display_name, expected.display_name);
        assert_eq!(result.email, expected.email);
        assert_eq!(result.username, expected.username);
    }

    #[tokio::test]
    async fn mock_userinfo_email_none_returns_none() {
        let user = UserInfo {
            id: "user-456".into(),
            display_name: "Bob Jones".into(),
            email: None,
            username: "bjones".into(),
        };
        let client = MockUserinfoClient::new(Ok(user));

        let result = client.get_current_user("bearer-token").await.unwrap();

        assert_eq!(result.email, None);
    }

    #[tokio::test]
    async fn mock_userinfo_timeout_error_propagates() {
        let client = MockUserinfoClient::new(Err(AuthError::Timeout));

        let err = client.get_current_user("bearer-token").await.unwrap_err();

        assert!(matches!(err, AuthError::Timeout));
    }

    #[tokio::test]
    async fn mock_userinfo_unreachable_error_propagates() {
        let client = MockUserinfoClient::new(Err(AuthError::Unreachable));

        let err = client.get_current_user("bearer-token").await.unwrap_err();

        assert!(matches!(err, AuthError::Unreachable));
    }

    #[tokio::test]
    async fn mock_userinfo_invalid_response_error_propagates() {
        let client =
            MockUserinfoClient::new(Err(AuthError::InvalidResponse("unexpected field".into())));

        let err = client.get_current_user("bearer-token").await.unwrap_err();

        assert!(matches!(err, AuthError::InvalidResponse(msg) if msg == "unexpected field"));
    }

    #[tokio::test]
    async fn mock_userinfo_records_token() {
        let client = MockUserinfoClient::new(Ok(make_user_info()));

        client.get_current_user("my-bearer-token").await.unwrap();

        let calls = client.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].token, "my-bearer-token");
    }

    #[tokio::test]
    async fn mock_userinfo_records_multiple_tokens_in_order() {
        let client = MockUserinfoClient::new(Ok(make_user_info()));

        client.get_current_user("token-first").await.unwrap();
        client.get_current_user("token-second").await.unwrap();

        let calls = client.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].token, "token-first");
        assert_eq!(calls[1].token, "token-second");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mock_userinfo_arc_dyn_sends_across_threads() {
        let client: Arc<dyn UserinfoClient> =
            Arc::new(MockUserinfoClient::new(Ok(make_user_info())));

        let client_clone = Arc::clone(&client);
        let handle =
            tokio::spawn(async move { client_clone.get_current_user("bearer-token").await });

        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}
