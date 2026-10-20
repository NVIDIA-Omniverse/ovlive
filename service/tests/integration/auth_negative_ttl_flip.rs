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

// Integration test for b.nrt item 1 — negative-TTL flip semantics.
//
// Ticket-specified scenario (not covered by the inline unit tests, which
// only assert TTL-bound timing on a single outcome): a backend that first
// denies and then allows must become visible as `Allowed` within the
// negative TTL window — and the same backend state under the much longer
// positive TTL would still have `Denied` cached. This captures the
// user-observable revocation-lag property the split-TTL design is meant to
// bound.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use live_edit::auth::cache::AuthCache;
use live_edit::auth::permissions::PermissionsClient;
use live_edit::auth::userinfo::UserinfoClient;
use live_edit::auth::{AuthError, AuthResult, Permission, UserInfo};

/// Permissions client that starts returning `Denied` and flips to `Allowed`
/// on command. Records each call for assertion on backend-hit count.
struct FlippablePermissions {
    flipped: Arc<std::sync::atomic::AtomicBool>,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl PermissionsClient for FlippablePermissions {
    async fn check_permission(
        &self,
        _token: &str,
        _path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.flipped.load(Ordering::SeqCst) {
            Ok(AuthResult::Allowed)
        } else {
            Ok(AuthResult::Denied)
        }
    }
}

struct StubUserinfo;

#[async_trait]
impl UserinfoClient for StubUserinfo {
    async fn get_current_user(&self, _token: &str) -> Result<UserInfo, AuthError> {
        Ok(UserInfo {
            id: "u1".into(),
            display_name: "U".into(),
            email: None,
            username: "u".into(),
        })
    }
}

/// Scenario: deny → flip backend to allow → within the negative TTL the
/// cached `Denied` must expire and the next lookup must see `Allowed`.
/// Positive TTL is set long enough (300 s) that if `Denied` were wrongly
/// placed in the positive cache, this test would still see `Denied` after
/// the sleep.
///
/// Uses `new_with_negative_ttl_durations` (Duration-granular) so the
/// negative TTL is sub-second and the test settles in ~200 ms of wall
/// clock instead of ~1.2 s. Production behavior is unchanged — the
/// env-driven `new_with_negative_ttl` is a thin delegate.
#[tokio::test]
async fn deny_then_allow_flip_visible_within_negative_ttl_not_positive_ttl() {
    let flipped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let perm = Arc::new(FlippablePermissions {
        flipped: Arc::clone(&flipped),
        calls: Arc::clone(&calls),
    });

    // positive_ttl = 300 s, negative_ttl = 100 ms (sub-second granularity).
    let cache = AuthCache::new_with_negative_ttl_durations(
        perm,
        Arc::new(StubUserinfo),
        Duration::from_secs(300),
        Duration::from_millis(100),
        Duration::from_millis(100),
        1_000,
        None,
    );

    // First call: backend returns Denied; cached for up to 1 s.
    let r1 = cache
        .check_permission("tok", "/p", Permission::Read)
        .await
        .unwrap();
    assert_eq!(r1, AuthResult::Denied);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Flip the backend to Allowed while the Denied entry is still cached.
    flipped.store(true, Ordering::SeqCst);

    // Inside the negative TTL window the cached Denied must still be
    // observed — i.e. the cache really is holding it and not unconditionally
    // delegating.
    let r2 = cache
        .check_permission("tok", "/p", Permission::Read)
        .await
        .unwrap();
    assert_eq!(
        r2,
        AuthResult::Denied,
        "Denied must still be cached during the negative TTL window"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "backend must not be called again inside the negative TTL"
    );

    // Wait past the 100 ms negative TTL but well short of the 300 s positive TTL.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // The flip must now be visible: Denied expired, backend called again,
    // now returns Allowed.
    let r3 = cache
        .check_permission("tok", "/p", Permission::Read)
        .await
        .unwrap();
    assert_eq!(
        r3,
        AuthResult::Allowed,
        "after negative TTL elapses, the flipped backend result must be visible"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "exactly one additional backend call after the negative TTL expiry"
    );

    // Immediately re-query: Allowed should now be in the positive cache
    // (300 s) so no further backend call.
    let r4 = cache
        .check_permission("tok", "/p", Permission::Read)
        .await
        .unwrap();
    assert_eq!(r4, AuthResult::Allowed);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "positive result must be served from the positive cache without new backend call"
    );
}
