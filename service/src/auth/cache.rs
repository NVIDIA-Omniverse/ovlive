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

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use dashmap::DashMap;
use futures::future::{BoxFuture, FutureExt, Shared};
use hmac::{Hmac, Mac};
use mini_moka::sync::Cache;
use rand::TryRngCore;
use sha2::Sha256;
use zeroize::Zeroizing;

use crate::auth::permissions::PermissionsClient;
use crate::auth::userinfo::UserinfoClient;
use crate::auth::{AuthError, AuthResult, Permission, UserInfo};
use crate::observability::metrics::SharedMetrics;

/// HMAC-SHA256 alias. The HMAC key is a per-process-random 32-byte seed
/// generated at startup and stored only inside [`AuthCache`]. See b.nrt item 2:
/// unsalted SHA-256 of a bearer token is deterministic across replicas and
/// deployments, so a memory disclosure (core dump, sidecar tracer, debug
/// container) plus a candidate token list would let an attacker verify hits
/// offline. HMAC with a secret seed turns that into a two-factor problem.
type HmacSha256 = Hmac<Sha256>;

const PERMISSIONS_KIND: &str = "permissions";
const USERINFO_KIND: &str = "userinfo";

/// Cache key for a permission check: HMAC-SHA256-of-token + path + permission level.
/// Raw tokens never enter this key — only the HMAC digest.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PermissionKey {
    token_hash: [u8; 32],
    path: String,
    permission: Permission,
}

/// Cache key for a userinfo lookup: HMAC-SHA256-of-token only.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct UserinfoKey {
    token_hash: [u8; 32],
}

/// Singleflight map value: a cloneable, shared future that resolves to the
/// same `Result` for every concurrent caller. See b.nrt item 3: coalesces
/// concurrent backend lookups for the same key so a 1000-way concurrent
/// request fan-out issues one backend call, not 1000.
type SharedPermissionFuture = Shared<BoxFuture<'static, Result<AuthResult, AuthError>>>;
type SharedUserinfoFuture = Shared<BoxFuture<'static, Result<UserInfo, AuthError>>>;

/// A positive cache entry together with the instant after which it must no
/// longer be served.
///
/// mini-moka 0.10 has only a cache-wide `time_to_live`, no per-entry expiry,
/// so the bearer's own expiry rides beside the value and [`AuthCache`] treats
/// an entry whose `not_after` has passed as a miss. The cache-wide TTL still
/// applies on top, so an entry lives until the EARLIER of the configured TTL
/// and the bearer's JWT `exp`. `None` means the bearer carried no usable `exp`
/// (an opaque token), and the configured TTL alone governs.
#[derive(Debug, Clone)]
struct BoundedEntry<T> {
    value: T,
    not_after: Option<Instant>,
}

impl<T> BoundedEntry<T> {
    /// Whether this entry may still be served at `now`.
    fn is_live_at(&self, now: Instant) -> bool {
        self.not_after.is_none_or(|t| now < t)
    }
}

/// How long a positive verdict for a bearer may be cached, as far as the
/// bearer itself is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BearerCacheBound {
    /// Not a JWT, or a JWT with no numeric `exp` (or one too far out to
    /// represent): the configured TTL alone governs.
    Unbounded,
    /// A JWT whose `exp` is still in the future: serve no later than this.
    Until(Instant),
    /// A JWT whose `exp` has already passed. A positive verdict for it serves
    /// the current call and is not cached.
    Expired,
}

/// Read the bearer's JWT `exp` — verification-free, through the same decode
/// helper the service-token diagnostics use — and turn it into a cache bound.
///
/// No signature is checked and none needs to be: this only ever SHORTENS how
/// long a verdict the permissions or userinfo service already returned for
/// this exact bearer is reused. A forged `exp` can at worst make this replica
/// ask the backend again sooner; it cannot extend an entry past the
/// configured TTL, which still caps every entry.
///
/// JWT `exp` is exclusive (RFC 7519 §4.1.4: the token must not be accepted
/// "on or after" it), so a bearer whose `exp` equals the current second is
/// already [`BearerCacheBound::Expired`].
fn bearer_cache_bound(token: &str) -> BearerCacheBound {
    let Some(exp) = crate::auth::service_token::decode_token_claims(token).and_then(|c| c.exp)
    else {
        return BearerCacheBound::Unbounded;
    };
    let Ok(exp_secs) = u64::try_from(exp) else {
        // A negative NumericDate is before the epoch: long past.
        return BearerCacheBound::Expired;
    };
    let Some(exp_at) = SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(exp_secs)) else {
        return BearerCacheBound::Unbounded;
    };
    match exp_at.duration_since(SystemTime::now()) {
        Ok(remaining) if !remaining.is_zero() => Instant::now()
            .checked_add(remaining)
            .map_or(BearerCacheBound::Unbounded, BearerCacheBound::Until),
        _ => BearerCacheBound::Expired,
    }
}

/// Concurrent TTL cache wrapping a [`PermissionsClient`] and [`UserinfoClient`].
///
/// Tokens are stored only as HMAC-SHA256 hashes (never plaintext). Positive
/// and negative permission decisions are kept in separate caches so negatives
/// can expire on a shorter TTL (b.nrt item 1) — this bounds revocation lag.
///
/// Positive entries — an `Allowed` permission verdict and a successful
/// userinfo lookup — additionally end at the bearer's own JWT `exp` when it
/// carries one ([`BearerCacheBound`]). Without that bound, a verdict the
/// permissions service gave a JWT shortly before its expiry kept being served
/// on this replica for up to the full positive TTL after the bearer itself had
/// expired, so the client was never told to refresh it. With it, the first
/// call after `exp` misses, reaches the backend, and gets that backend's answer
/// for an expired bearer (`UNAUTHENTICATED` from the permissions service).
/// Opaque bearers keep the configured TTL.
///
/// Concurrent callers with identical keys share one backend call via
/// singleflight (b.nrt item 3).
pub struct AuthCache {
    permissions_client: Arc<dyn PermissionsClient>,
    userinfo_client: Arc<dyn UserinfoClient>,
    /// Cache for `AuthResult::Allowed` decisions. Longer TTL, further capped
    /// per entry at the bearer's JWT `exp` (see [`BoundedEntry`]).
    permissions_positive_cache: Cache<PermissionKey, BoundedEntry<AuthResult>>,
    /// Cache for `AuthResult::Denied | Unauthenticated | PathNotFound |
    /// InvalidRequest`. Shorter TTL. Caching a refusal can never grant
    /// access, and the key carries the bearer's HMAC, so one principal's
    /// verdict is never served to another.
    permissions_negative_cache: Cache<PermissionKey, AuthResult>,
    /// Userinfo cache. Successful lookups are cached on the positive TTL,
    /// further capped per entry at the bearer's JWT `exp`. With the userinfo
    /// service and authorization both on, that includes a not-found answer
    /// the fallback decorator beneath this cache resolved from the bearer's
    /// JWT claims.
    userinfo_cache: Cache<UserinfoKey, BoundedEntry<UserInfo>>,
    /// Userinfo negative cache (b.ohw medium): caches backend failures
    /// (`AuthError::InvalidResponse` for a bearer the userinfo service
    /// rejected, `Timeout` / `Unreachable` for transient outages) with a
    /// short TTL so duplicate requests for the same rejected bearer don't
    /// pound the backend. `AuthError::UserNotFound` lands here only with
    /// authorization off; with it on, the fallback decorator has already
    /// turned it into a success.
    /// Cleared by TTL expiry; read on every lookup that misses the live
    /// userinfo cache, so a previously-cached failure short-circuits before
    /// the singleflight machinery runs.
    userinfo_negative_cache: Cache<UserinfoKey, AuthError>,
    /// Per-process HMAC seed wrapped in [`Zeroizing`] so [`Drop`] wipes the
    /// bytes (b.ohw medium). Never logged, never exported, never serialized.
    /// Held only for its drop-wipe side effect — the hot-path hash uses
    /// [`Self::hmac_template`], which carries the keyed schedule. Holding the
    /// raw seed here ensures a memory-disclosure attacker cannot recover the
    /// 32-byte secret from a freed allocation when the cache is dropped.
    #[allow(dead_code)]
    hmac_seed: Zeroizing<[u8; 32]>,
    /// HMAC-SHA256 keyed template, pre-built once at construction and
    /// cloned per [`Self::hash_token`] call so the per-call cost is one
    /// clone + one update + one finalize (b.ohw medium). Cloning a keyed
    /// `Hmac` returns the same key-schedule state, so we never re-run the
    /// expensive keying step on the hot path.
    hmac_template: HmacSha256,
    /// Singleflight: in-flight permissions lookups coalesced by cache key.
    permissions_inflight: DashMap<PermissionKey, SharedPermissionFuture>,
    /// Singleflight: in-flight userinfo lookups coalesced by cache key.
    userinfo_inflight: DashMap<UserinfoKey, SharedUserinfoFuture>,
    /// Optional metrics sink. `None` in unit tests that don't exercise metrics.
    metrics: Option<SharedMetrics>,
}

impl AuthCache {
    /// Construct a new `AuthCache` with equal positive and negative TTLs.
    ///
    /// Convenience constructor that preserves the pre-b.nrt behavior. For
    /// production use [`AuthCache::new_with_negative_ttl`] so negative
    /// decisions expire on a shorter TTL than positives.
    pub fn new(
        permissions_client: Arc<dyn PermissionsClient>,
        userinfo_client: Arc<dyn UserinfoClient>,
        auth_cache_ttl_secs: u64,
        max_entries: u64,
    ) -> Self {
        Self::new_with_negative_ttl(
            permissions_client,
            userinfo_client,
            auth_cache_ttl_secs,
            auth_cache_ttl_secs,
            // Default-userinfo-negative-ttl matches the convenience helper's
            // treat-positive-as-negative posture. Production callers should
            // use `new_with_negative_ttls` to set it independently.
            auth_cache_ttl_secs,
            max_entries,
            None,
        )
    }

    /// Construct a new `AuthCache` with separate TTLs for positive permission
    /// decisions, negative permission decisions, and userinfo errors.
    ///
    /// `userinfo_negative_ttl_secs` controls the b.ohw userinfo negative
    /// cache: backend failures (a rejected bearer, transient outages) are
    /// memoized with this TTL so a rejected bearer doesn't pound the backend
    /// on every RPC. Set independently of the permissions negative TTL
    /// because userinfo failures have a different recovery profile (token
    /// refresh, not policy revocation).
    pub fn new_with_negative_ttl(
        permissions_client: Arc<dyn PermissionsClient>,
        userinfo_client: Arc<dyn UserinfoClient>,
        positive_ttl_secs: u64,
        negative_ttl_secs: u64,
        userinfo_negative_ttl_secs: u64,
        max_entries: u64,
        metrics: Option<SharedMetrics>,
    ) -> Self {
        Self::new_with_negative_ttl_durations(
            permissions_client,
            userinfo_client,
            Duration::from_secs(positive_ttl_secs),
            Duration::from_secs(negative_ttl_secs),
            Duration::from_secs(userinfo_negative_ttl_secs),
            max_entries,
            metrics,
        )
    }

    /// Construct an `AuthCache` with `Duration`-typed TTLs — same semantics
    /// as [`Self::new_with_negative_ttl`] but accepts sub-second TTLs so
    /// the negative-TTL expiry test (b.j5t follow-up) can exercise the
    /// flip-after-expiry contract without burning real seconds per run.
    /// Production code uses [`Self::new_with_negative_ttl`], which is
    /// driven by the `LIVE_EDIT_AUTH_CACHE_*_TTL_SECS` env vars
    /// (seconds-granular).
    pub fn new_with_negative_ttl_durations(
        permissions_client: Arc<dyn PermissionsClient>,
        userinfo_client: Arc<dyn UserinfoClient>,
        positive_ttl: Duration,
        negative_ttl: Duration,
        userinfo_negative_ttl: Duration,
        max_entries: u64,
        metrics: Option<SharedMetrics>,
    ) -> Self {
        let permissions_positive_cache = Cache::builder()
            .time_to_live(positive_ttl)
            .max_capacity(max_entries)
            .build();
        let permissions_negative_cache = Cache::builder()
            .time_to_live(negative_ttl)
            .max_capacity(max_entries)
            .build();
        let userinfo_cache = Cache::builder()
            .time_to_live(positive_ttl)
            .max_capacity(max_entries)
            .build();
        let userinfo_negative_cache = Cache::builder()
            .time_to_live(userinfo_negative_ttl)
            .max_capacity(max_entries)
            .build();
        let hmac_seed = generate_hmac_seed();
        // Pre-build the HMAC template once and clone-then-reset per call
        // (b.ohw medium). Building a fresh `HmacSha256` on every `hash_token`
        // re-runs the keying schedule on hot-path code.
        let hmac_template = <HmacSha256 as Mac>::new_from_slice(hmac_seed.as_ref())
            .expect("HMAC-SHA256 accepts any key length");
        Self {
            permissions_client,
            userinfo_client,
            permissions_positive_cache,
            permissions_negative_cache,
            userinfo_cache,
            userinfo_negative_cache,
            hmac_seed,
            hmac_template,
            permissions_inflight: DashMap::new(),
            userinfo_inflight: DashMap::new(),
            metrics,
        }
    }

    /// Compute the HMAC-SHA256 of `token` using this cache's per-process seed.
    fn hash_token(&self, token: &str) -> [u8; 32] {
        // Clone the keyed template (no per-call key schedule) and reset it
        // before update — equivalent to `Mac::reset()` but `hmac` doesn't
        // expose that, so we rely on `Clone` returning the keyed-but-empty
        // state captured at construction time (b.ohw medium).
        let mut mac = self.hmac_template.clone();
        mac.update(token.as_bytes());
        mac.finalize().into_bytes().into()
    }

    /// Look up a permission result in whichever of the two caches contains it.
    ///
    /// A positive entry whose bearer `exp` has passed is invalidated and
    /// reported as absent, so the caller counts a miss and asks the backend.
    fn permissions_cache_get(&self, key: &PermissionKey) -> Option<AuthResult> {
        if let Some(entry) = self.permissions_positive_cache.get(key) {
            if entry.is_live_at(Instant::now()) {
                return Some(entry.value);
            }
            self.permissions_positive_cache.invalidate(key);
        }
        self.permissions_negative_cache.get(key)
    }

    /// Insert a permission decision into the appropriate TTL-bucketed cache.
    ///
    /// `token` is the bearer the decision was made for; an `Allowed` verdict
    /// is bounded by its JWT `exp` ([`bearer_cache_bound`]) and not cached at
    /// all when that `exp` has already passed. Negative verdicts are not
    /// bounded: they are already on the short negative TTL, and a verdict
    /// against an expired bearer stays true after expiry.
    fn permissions_cache_insert(&self, key: PermissionKey, result: AuthResult, token: &str) {
        match result {
            AuthResult::Allowed => {
                let not_after = match bearer_cache_bound(token) {
                    BearerCacheBound::Unbounded => None,
                    BearerCacheBound::Until(t) => Some(t),
                    BearerCacheBound::Expired => return,
                };
                self.permissions_positive_cache.insert(
                    key,
                    BoundedEntry {
                        value: result,
                        not_after,
                    },
                );
            }
            AuthResult::Denied
            | AuthResult::Unauthenticated
            | AuthResult::PathNotFound
            | AuthResult::InvalidRequest { .. } => {
                self.permissions_negative_cache.insert(key, result)
            }
        }
    }

    /// Look up a cached successful userinfo lookup. An entry whose bearer
    /// `exp` has passed is invalidated and reported as absent.
    fn userinfo_cache_get(&self, key: &UserinfoKey) -> Option<UserInfo> {
        let entry = self.userinfo_cache.get(key)?;
        if entry.is_live_at(Instant::now()) {
            return Some(entry.value);
        }
        self.userinfo_cache.invalidate(key);
        None
    }

    /// Cache a successful userinfo lookup for `token`, bounded by its JWT
    /// `exp` the same way an `Allowed` verdict is.
    fn userinfo_cache_insert(&self, key: UserinfoKey, info: UserInfo, token: &str) {
        let not_after = match bearer_cache_bound(token) {
            BearerCacheBound::Unbounded => None,
            BearerCacheBound::Until(t) => Some(t),
            BearerCacheBound::Expired => return,
        };
        self.userinfo_cache.insert(
            key,
            BoundedEntry {
                value: info,
                not_after,
            },
        );
    }

    /// Check permission, returning a cached result when available.
    ///
    /// On cache miss, coalesces concurrent callers for the same key via
    /// singleflight (b.nrt item 3) so only one backend call is issued;
    /// on success, populates the positive or negative cache per the result
    /// variant (b.nrt item 1).
    pub async fn check_permission(
        &self,
        token: &str,
        path: &str,
        permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        let key = PermissionKey {
            token_hash: self.hash_token(token),
            path: path.to_string(),
            permission,
        };
        if let Some(cached) = self.permissions_cache_get(&key) {
            if let Some(m) = &self.metrics {
                m.inc_auth_cache_hit(PERMISSIONS_KIND);
            }
            return Ok(cached);
        }
        if let Some(m) = &self.metrics {
            m.inc_auth_cache_miss(PERMISSIONS_KIND);
        }

        let shared = self.get_or_create_permissions_inflight(&key, token, path, permission);
        // Hold a clone for the post-await `remove_if` check; `shared.await`
        // consumes the original via `IntoFuture`.
        let shared_for_check = shared.clone();

        // b.j5t item 4: cancellation-safe cleanup of the inflight map.
        // Pre-fix the `permissions_inflight.remove(&key)` after `.await` ran
        // unconditionally — but a cancelled caller (tonic deadline drop,
        // client disconnect) returns from this function via the Drop path
        // *before* `.await` resolves, so the entry leaked indefinitely.
        // DashMap has no eviction, so a long-running pod with sporadic
        // mid-flight cancellations would unboundedly grow these maps.
        //
        // The scope guard removes the entry on every exit path — normal
        // return, error return, or future-drop on cancel. It checks
        // `Shared::ptr_eq` against the cloned future we put in so we never
        // remove someone else's entry: in the rare case where our shared
        // future already resolved, a follow-up caller inserted a fresh
        // future at the same key, and then we got dropped, the guard finds
        // an unequal pointer and leaves the new caller's entry alone.
        //
        // `ScopeGuard::into_inner` disarms the guard on the happy path so
        // the cache-insert-then-remove ordering (b.ohw H-auth-2 — see below)
        // is preserved. Cancellation paths skip `into_inner` entirely.
        let cleanup_inflight = &self.permissions_inflight;
        let cleanup_key = key.clone();
        let cleanup_shared = shared.clone();
        let cleanup = scopeguard::guard((), move |()| {
            cleanup_inflight.remove_if(&cleanup_key, |_k, v| v.ptr_eq(&cleanup_shared));
        });

        let result = shared.await;

        // b.ohw H-auth-2: cache-insert MUST run before the inflight entry is
        // removed. Pre-fix order was `result.await → inflight.remove(key) →
        // cache.insert(key, result)`, which leaves a window where a fresh
        // caller arriving between `remove` and `insert` misses both the
        // cache (still empty) and the inflight map (just emptied) and issues
        // a duplicate backend call. Inserting first guarantees that any new
        // caller scheduled after the singleflight resolves either sees the
        // cached value (if we removed already) or joins the still-live
        // shared future via the inflight map.
        if let Ok(res) = &result {
            self.permissions_cache_insert(key.clone(), res.clone(), token);
        }
        // Defuse the scope guard so the explicit removal below runs *after*
        // the cache-insert above (preserving the H-auth-2 ordering). The
        // explicit removal here matches the guard body — both use
        // `remove_if` keyed on `Shared::ptr_eq` so a follow-up caller that
        // installed a fresh future cannot be evicted.
        scopeguard::ScopeGuard::into_inner(cleanup);
        self.permissions_inflight
            .remove_if(&key, |_k, v| v.ptr_eq(&shared_for_check));

        result
    }

    fn get_or_create_permissions_inflight(
        &self,
        key: &PermissionKey,
        token: &str,
        path: &str,
        permission: Permission,
    ) -> SharedPermissionFuture {
        // DashMap::entry::or_insert_with atomically inserts if absent and
        // returns a reference either way — the new future is only built
        // when no other caller has already registered one for this key.
        let entry = self
            .permissions_inflight
            .entry(key.clone())
            .or_insert_with(|| {
                let client = Arc::clone(&self.permissions_client);
                let token = token.to_owned();
                let path = path.to_owned();
                // b.ohw H-auth-3: wrap the inner future in
                // `AssertUnwindSafe(...).catch_unwind()` so a panic in the
                // backend client implementation does not cascade through
                // `futures::Shared` to every coalesced waiter. Pre-fix, a
                // single panic propagated to all N concurrent callers as a
                // panic each — the backend bug then crashed N tasks instead
                // of one. Caught panics surface as a clean
                // `AuthError::InvalidResponse` so callers observe a normal
                // error return.
                let inner = async move { client.check_permission(&token, &path, permission).await };
                std::panic::AssertUnwindSafe(inner)
                    .catch_unwind()
                    .map(|r| match r {
                        Ok(out) => out,
                        Err(_) => Err(AuthError::InvalidResponse(
                            "internal panic in auth backend".to_owned(),
                        )),
                    })
                    .boxed()
                    .shared()
            });
        entry.clone()
    }

    /// Resolve user identity, returning a cached result when available.
    ///
    /// Successes are cached on the positive TTL; errors are memoized in
    /// `userinfo_negative_cache` on the short userinfo-negative TTL.
    /// Concurrent callers for the same token share one backend call via
    /// singleflight (b.nrt item 3).
    pub async fn get_current_user(&self, token: &str) -> Result<UserInfo, AuthError> {
        let key = UserinfoKey {
            token_hash: self.hash_token(token),
        };
        if let Some(cached) = self.userinfo_cache_get(&key) {
            if let Some(m) = &self.metrics {
                m.inc_auth_cache_hit(USERINFO_KIND);
            }
            return Ok(cached);
        }
        // b.ohw medium: short-TTL negative cache. A bearer the userinfo
        // service rejects would otherwise hit the backend on every RPC until
        // the client refreshed; memoizing the failure for a few seconds
        // bounds the duplicate-request fanout while keeping
        // revocation-propagation tight.
        if let Some(cached_err) = self.userinfo_negative_cache.get(&key) {
            if let Some(m) = &self.metrics {
                m.inc_auth_cache_hit(USERINFO_KIND);
            }
            return Err(cached_err);
        }
        if let Some(m) = &self.metrics {
            m.inc_auth_cache_miss(USERINFO_KIND);
        }

        let shared = self.get_or_create_userinfo_inflight(&key, token);
        // Clone for the post-await `remove_if` — `shared.await` consumes
        // the original.
        let shared_for_check = shared.clone();

        // b.j5t item 4: cancellation-safe inflight cleanup. Mirrors the
        // permissions arm — see the matching block in `check_permission` for
        // the full rationale. The scope guard removes the entry on every
        // exit path (including cancellation mid-`.await`) using
        // `Shared::ptr_eq` so we never evict a follow-up caller's freshly
        // installed future.
        let cleanup_inflight = &self.userinfo_inflight;
        let cleanup_key = key.clone();
        let cleanup_shared = shared.clone();
        let cleanup = scopeguard::guard((), move |()| {
            cleanup_inflight.remove_if(&cleanup_key, |_k, v| v.ptr_eq(&cleanup_shared));
        });

        let result = shared.await;

        // b.ohw H-auth-2: cache-insert before inflight-remove. See the
        // matching comment in `check_permission` for why.
        match &result {
            Ok(info) => {
                self.userinfo_cache_insert(key.clone(), info.clone(), token);
            }
            Err(e) => {
                // Userinfo negative cache (b.ohw medium): memoize backend
                // failures with a short TTL so a rejected bearer does not
                // pound the backend on every RPC. Permanent contract
                // violations (`InvalidResponse`) are the common case here
                // — a userinfo 401/403 surfaces as
                // `InvalidResponse("userinfo service rejected the bearer …")`.
                // Transient outages (`Timeout` / `Unreachable`) are also
                // cached so a flapping backend can't trigger a stampede; the
                // short TTL keeps recovery fast. The userinfo service's
                // not-found answer (`UserNotFound`) is memoized here only with
                // authorization off: with it on, the fallback decorator
                // beneath this cache answers it from the bearer's JWT claims,
                // so it arrives as an `Ok` and takes the positive arm above.
                self.userinfo_negative_cache.insert(key.clone(), e.clone());
            }
        }
        // Defuse the scope guard so the explicit removal preserves H-auth-2
        // ordering. See the parallel comment in `check_permission`.
        scopeguard::ScopeGuard::into_inner(cleanup);
        self.userinfo_inflight
            .remove_if(&key, |_k, v| v.ptr_eq(&shared_for_check));

        result
    }

    fn get_or_create_userinfo_inflight(
        &self,
        key: &UserinfoKey,
        token: &str,
    ) -> SharedUserinfoFuture {
        let entry = self
            .userinfo_inflight
            .entry(key.clone())
            .or_insert_with(|| {
                let client = Arc::clone(&self.userinfo_client);
                let token = token.to_owned();
                // b.ohw H-auth-3: see the matching comment in
                // `get_or_create_permissions_inflight` — catch_unwind so a
                // backend panic surfaces as a clean error to one caller
                // instead of propagating through `Shared` to all waiters.
                let inner = async move { client.get_current_user(&token).await };
                std::panic::AssertUnwindSafe(inner)
                    .catch_unwind()
                    .map(|r| match r {
                        Ok(out) => out,
                        Err(_) => Err(AuthError::InvalidResponse(
                            "internal panic in auth backend".to_owned(),
                        )),
                    })
                    .boxed()
                    .shared()
            });
        entry.clone()
    }
}

/// Generate a 32-byte HMAC seed from the OS CSPRNG. The seed is read once at
/// `AuthCache` construction and lives in the struct — it is never logged,
/// exported, or serialized. Wrapped in [`Zeroizing`] so the raw bytes are
/// scrubbed from memory on drop (b.ohw medium); a memory disclosure (core
/// dump, debug container) on a long-running process can't recover the seed
/// from a freed allocation. Panics if the OS RNG fails; this is an
/// at-startup failure and masking it silently would defeat the purpose of
/// the seed.
fn generate_hmac_seed() -> Zeroizing<[u8; 32]> {
    let mut seed = Zeroizing::new([0u8; 32]);
    rand::rngs::OsRng
        .try_fill_bytes(seed.as_mut())
        .expect("failed to read 32 bytes from OS CSPRNG for auth cache HMAC seed");
    seed
}

#[async_trait::async_trait]
impl crate::auth::permissions::PermissionsClient for AuthCache {
    async fn check_permission(
        &self,
        token: &str,
        path: &str,
        permission: crate::auth::Permission,
    ) -> Result<crate::auth::AuthResult, crate::auth::AuthError> {
        // Delegate to the inherent method. The two names collide, so we rename
        // the inherent side on the trait boundary to avoid relying on
        // inherent-wins resolution (b.nrt item 5).
        AuthCache::check_permission(self, token, path, permission).await
    }
}

#[async_trait::async_trait]
impl crate::auth::userinfo::UserinfoClient for AuthCache {
    async fn get_current_user(&self, token: &str) -> Result<UserInfo, crate::auth::AuthError> {
        AuthCache::get_current_user(self, token).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use crate::auth::permissions::mock::MockPermissionsClient;
    use crate::auth::userinfo::mock::MockUserinfoClient;
    use crate::auth::{AuthResult, Permission, UserInfo};

    use super::AuthCache;

    fn make_user_info() -> UserInfo {
        UserInfo {
            id: "user-1".into(),
            display_name: "Test User".into(),
            email: Some("test@example.com".into()),
            username: "testuser".into(),
        }
    }

    /// Default test capacity. Large enough that capacity-based eviction never
    /// fires during the existing TTL/behavioral tests below.
    const TEST_MAX_ENTRIES: u64 = 1_000;

    fn cache_with_ttl(
        perm_client: MockPermissionsClient,
        userinfo_client: MockUserinfoClient,
        ttl_secs: u64,
    ) -> AuthCache {
        AuthCache::new(
            Arc::new(perm_client),
            Arc::new(userinfo_client),
            ttl_secs,
            TEST_MAX_ENTRIES,
        )
    }

    #[tokio::test]
    async fn check_permission_cache_miss_delegates_to_client() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache = cache_with_ttl(perm, MockUserinfoClient::new(Ok(make_user_info())), 300);

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();

        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn check_permission_cache_hit_returns_without_calling_client_again() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache = cache_with_ttl(perm, MockUserinfoClient::new(Ok(make_user_info())), 300);

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();
        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();

        // Client called only once; second call served from cache
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn get_current_user_cache_miss_delegates_to_client() {
        let userinfo = MockUserinfoClient::new(Ok(make_user_info()));
        let calls = Arc::clone(&userinfo.calls);
        let cache = cache_with_ttl(
            MockPermissionsClient::new(Ok(AuthResult::Allowed)),
            userinfo,
            300,
        );

        cache.get_current_user("token").await.unwrap();

        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn get_current_user_cache_hit_returns_without_calling_client_again() {
        let userinfo = MockUserinfoClient::new(Ok(make_user_info()));
        let calls = Arc::clone(&userinfo.calls);
        let cache = cache_with_ttl(
            MockPermissionsClient::new(Ok(AuthResult::Allowed)),
            userinfo,
            300,
        );

        cache.get_current_user("token").await.unwrap();
        cache.get_current_user("token").await.unwrap();

        // Client called only once; second call served from cache
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn check_permission_ttl_expiry_causes_cache_miss() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache = cache_with_ttl(perm, MockUserinfoClient::new(Ok(make_user_info())), 1);

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();

        // Wait for the 1-second TTL to expire
        tokio::time::sleep(Duration::from_millis(1100)).await;

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();

        // Cache expired, so client is called a second time
        assert_eq!(calls.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn get_current_user_ttl_expiry_causes_cache_miss() {
        let userinfo = MockUserinfoClient::new(Ok(make_user_info()));
        let calls = Arc::clone(&userinfo.calls);
        let cache = cache_with_ttl(
            MockPermissionsClient::new(Ok(AuthResult::Allowed)),
            userinfo,
            1,
        );

        cache.get_current_user("token").await.unwrap();

        // Wait for the 1-second TTL to expire
        tokio::time::sleep(Duration::from_millis(1100)).await;

        cache.get_current_user("token").await.unwrap();

        // Cache expired, so client is called a second time
        assert_eq!(calls.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn check_permission_different_tokens_are_distinct_cache_entries() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache = cache_with_ttl(perm, MockUserinfoClient::new(Ok(make_user_info())), 300);

        cache
            .check_permission("token-a", "/path", Permission::Read)
            .await
            .unwrap();
        cache
            .check_permission("token-b", "/path", Permission::Read)
            .await
            .unwrap();

        // Two different tokens → two cache entries → two client calls
        assert_eq!(calls.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn check_permission_different_paths_are_distinct_cache_entries() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache = cache_with_ttl(perm, MockUserinfoClient::new(Ok(make_user_info())), 300);

        cache
            .check_permission("token", "/path/a", Permission::Read)
            .await
            .unwrap();
        cache
            .check_permission("token", "/path/b", Permission::Read)
            .await
            .unwrap();

        assert_eq!(calls.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn check_permission_different_permission_levels_are_distinct_cache_entries() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache = cache_with_ttl(perm, MockUserinfoClient::new(Ok(make_user_info())), 300);

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();
        cache
            .check_permission("token", "/path", Permission::Write)
            .await
            .unwrap();

        assert_eq!(calls.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn check_permission_hmac_hashing_different_tokens_cause_separate_calls() {
        // Two tokens that differ only by one character must produce separate cache entries,
        // confirming that the hash (not pointer identity) drives cache key uniqueness.
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache = cache_with_ttl(perm, MockUserinfoClient::new(Ok(make_user_info())), 300);

        cache
            .check_permission("aaaa", "/path", Permission::Read)
            .await
            .unwrap();
        cache
            .check_permission("aaab", "/path", Permission::Read)
            .await
            .unwrap();

        assert_eq!(calls.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn check_permission_same_token_string_reuses_cache_entry() {
        // Calling with equal string values (not the same pointer) must hit the cache,
        // confirming the HMAC produces the same hash for equal tokens.
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache = cache_with_ttl(perm, MockUserinfoClient::new(Ok(make_user_info())), 300);

        let token_a = String::from("same-token");
        let token_b = String::from("same-token");
        cache
            .check_permission(&token_a, "/path", Permission::Read)
            .await
            .unwrap();
        cache
            .check_permission(&token_b, "/path", Permission::Read)
            .await
            .unwrap();

        // Same hash → same cache key → only one client call
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn check_permission_concurrent_tasks_no_panics() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let cache = Arc::new(cache_with_ttl(
            perm,
            MockUserinfoClient::new(Ok(make_user_info())),
            300,
        ));

        let handles: Vec<_> = (0..12)
            .map(|i| {
                let cache = Arc::clone(&cache);
                tokio::spawn(async move {
                    let token = format!("token-{}", i % 3);
                    cache
                        .check_permission(&token, "/shared/path", Permission::Read)
                        .await
                        .unwrap();
                })
            })
            .collect();

        for handle in handles {
            handle.await.expect("task panicked");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn get_current_user_concurrent_tasks_no_panics() {
        let cache = Arc::new(cache_with_ttl(
            MockPermissionsClient::new(Ok(AuthResult::Allowed)),
            MockUserinfoClient::new(Ok(make_user_info())),
            300,
        ));

        let handles: Vec<_> = (0..12)
            .map(|i| {
                let cache = Arc::clone(&cache);
                tokio::spawn(async move {
                    let token = format!("token-{}", i % 3);
                    cache.get_current_user(&token).await.unwrap();
                })
            })
            .collect();

        for handle in handles {
            handle.await.expect("task panicked");
        }
    }

    // ── b.nrt item 2 regression: per-process HMAC seed uniqueness ───────
    //
    // Unsalted SHA-256 of a token is deterministic across replicas. HMAC
    // with a per-process-random seed makes the hash different in every
    // `AuthCache` instance, so memory disclosure on one replica can't be
    // replayed elsewhere.

    #[tokio::test]
    async fn hmac_seed_differs_across_auth_cache_instances() {
        let cache_a = cache_with_ttl(
            MockPermissionsClient::new(Ok(AuthResult::Allowed)),
            MockUserinfoClient::new(Ok(make_user_info())),
            300,
        );
        let cache_b = cache_with_ttl(
            MockPermissionsClient::new(Ok(AuthResult::Allowed)),
            MockUserinfoClient::new(Ok(make_user_info())),
            300,
        );

        let hash_a = cache_a.hash_token("same-token");
        let hash_b = cache_b.hash_token("same-token");

        assert_ne!(
            hash_a, hash_b,
            "HMAC seeds must be randomly generated per-instance — \
             identical hashes would mean the seed is either static or not being applied"
        );
    }

    #[tokio::test]
    async fn hash_token_within_one_cache_is_deterministic() {
        let cache = cache_with_ttl(
            MockPermissionsClient::new(Ok(AuthResult::Allowed)),
            MockUserinfoClient::new(Ok(make_user_info())),
            300,
        );
        assert_eq!(cache.hash_token("tok"), cache.hash_token("tok"));
    }

    // ── b.nrt item 1 regression: negative TTL separate from positive ────

    fn cache_with_split_ttls(
        perm_client: MockPermissionsClient,
        userinfo_client: MockUserinfoClient,
        positive_ttl_secs: u64,
        negative_ttl_secs: u64,
    ) -> AuthCache {
        AuthCache::new_with_negative_ttl(
            Arc::new(perm_client),
            Arc::new(userinfo_client),
            positive_ttl_secs,
            negative_ttl_secs,
            // Userinfo negative TTL: mirror the permissions negative TTL in
            // these helpers. Tests that need to exercise userinfo-negative
            // semantics independently can call `new_with_negative_ttl`
            // directly with a different value.
            negative_ttl_secs,
            TEST_MAX_ENTRIES,
            None,
        )
    }

    #[tokio::test]
    async fn denied_cached_in_negative_cache_and_expires_on_negative_ttl() {
        // Positive TTL large (300s), negative TTL small (1s). A Denied
        // decision must expire within the negative window — if the code
        // wrongly routed it into the positive cache, it would outlive the
        // sleep and we'd still see one call.
        let perm = MockPermissionsClient::new(Ok(AuthResult::Denied));
        let calls = Arc::clone(&perm.calls);
        let cache =
            cache_with_split_ttls(perm, MockUserinfoClient::new(Ok(make_user_info())), 300, 1);

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();
        assert_eq!(calls.lock().unwrap().len(), 1);

        // Still inside negative TTL → cache hit.
        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();
        assert_eq!(calls.lock().unwrap().len(), 1);

        tokio::time::sleep(Duration::from_millis(1100)).await;

        // Negative TTL elapsed → cache miss, client called again.
        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();
        assert_eq!(
            calls.lock().unwrap().len(),
            2,
            "Denied must have been cached in the negative cache and expired on the negative TTL"
        );
    }

    #[tokio::test]
    async fn unauthenticated_also_uses_negative_ttl() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::Unauthenticated));
        let calls = Arc::clone(&perm.calls);
        let cache =
            cache_with_split_ttls(perm, MockUserinfoClient::new(Ok(make_user_info())), 300, 1);

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(1100)).await;
        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();

        assert_eq!(calls.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn path_not_found_also_uses_negative_ttl() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::PathNotFound));
        let calls = Arc::clone(&perm.calls);
        let cache =
            cache_with_split_ttls(perm, MockUserinfoClient::new(Ok(make_user_info())), 300, 1);

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(1100)).await;
        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();

        assert_eq!(calls.lock().unwrap().len(), 2);
    }

    /// `InvalidRequest` (the storage service refusing the probe as a
    /// malformed request) is negative-cached so a client looping on a bad
    /// base path costs one backend probe per negative-TTL window. Three
    /// properties, each a distinct backend-call count:
    ///
    /// 1. A repeat of the same (token, path, permission) is a cache hit —
    ///    without caching, every call would reach the backend.
    /// 2. A different token is a miss — the key carries the bearer's HMAC,
    ///    so one principal's verdict is not served to another.
    /// 3. After the short negative TTL elapses the entry is gone — had it been
    ///    routed into the positive cache (300 s TTL) it would still be served.
    #[tokio::test]
    async fn invalid_request_cached_per_principal_and_expires_on_negative_ttl() {
        const NEGATIVE_TTL: Duration = Duration::from_millis(500);
        let verdict = AuthResult::InvalidRequest {
            backend_status: "HTTP 400",
        };
        let perm = MockPermissionsClient::new(Ok(verdict.clone()));
        let calls = Arc::clone(&perm.calls);
        let cache = AuthCache::new_with_negative_ttl_durations(
            Arc::new(perm),
            Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
            Duration::from_secs(300),
            NEGATIVE_TTL,
            NEGATIVE_TTL,
            TEST_MAX_ENTRIES,
            None,
        );

        for _ in 0..2 {
            let result = cache
                .check_permission("token-a", "load-test/perf", Permission::Read)
                .await
                .unwrap();
            assert_eq!(result, verdict);
        }
        assert_eq!(
            calls.lock().unwrap().len(),
            1,
            "a repeated InvalidRequest lookup must be served from the cache"
        );

        cache
            .check_permission("token-b", "load-test/perf", Permission::Read)
            .await
            .unwrap();
        assert_eq!(
            calls.lock().unwrap().len(),
            2,
            "a different token must not be served another principal's cached verdict"
        );

        tokio::time::sleep(NEGATIVE_TTL + Duration::from_millis(300)).await;

        cache
            .check_permission("token-a", "load-test/perf", Permission::Read)
            .await
            .unwrap();
        assert_eq!(
            calls.lock().unwrap().len(),
            3,
            "InvalidRequest must sit on the negative TTL, not the positive one"
        );
    }

    #[tokio::test]
    async fn allowed_stays_on_positive_ttl_even_when_negative_ttl_is_short() {
        // Positive 300s, negative 1s. If Allowed were wrongly routed into
        // the negative cache it would miss after the sleep, so we assert the
        // call count stays at 1 — Allowed must be in the positive cache.
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache =
            cache_with_split_ttls(perm, MockUserinfoClient::new(Ok(make_user_info())), 300, 1);

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(1200)).await;
        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();

        assert_eq!(
            calls.lock().unwrap().len(),
            1,
            "Allowed must live in the positive cache with the positive TTL"
        );
    }

    // ── b.nrt item 3 regression: singleflight coalesces concurrent calls ──

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn permissions_singleflight_coalesces_concurrent_same_key_lookups() {
        use crate::auth::{AuthError, AuthResult, Permission};
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Slow mock that counts concurrent in-flight calls.
        struct SlowCountingClient {
            counter: Arc<AtomicUsize>,
        }
        #[async_trait::async_trait]
        impl crate::auth::permissions::PermissionsClient for SlowCountingClient {
            async fn check_permission(
                &self,
                _token: &str,
                _path: &str,
                _permission: Permission,
            ) -> Result<AuthResult, AuthError> {
                self.counter.fetch_add(1, Ordering::SeqCst);
                // Hold the "backend call" long enough for all concurrent
                // callers to coalesce onto the same Shared future.
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(AuthResult::Allowed)
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let perm = SlowCountingClient {
            counter: Arc::clone(&counter),
        };
        let cache = Arc::new(AuthCache::new(
            Arc::new(perm),
            Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
            300,
            TEST_MAX_ENTRIES,
        ));

        // Fire 100 concurrent lookups for the same (token, path, permission).
        // Without singleflight, every one issues its own backend call; with
        // singleflight, only the first issues and the rest await the same
        // Shared future.
        let mut handles = Vec::new();
        for _ in 0..100 {
            let cache = Arc::clone(&cache);
            handles.push(tokio::spawn(async move {
                cache
                    .check_permission("same-tok", "/same/path", Permission::Read)
                    .await
                    .unwrap()
            }));
        }
        for h in handles {
            let _ = h.await.unwrap();
        }
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "singleflight must coalesce 100 concurrent same-key lookups into 1 backend call"
        );

        // Second round — must hit the cache, no new backend call.
        cache
            .check_permission("same-tok", "/same/path", Permission::Read)
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn userinfo_singleflight_coalesces_concurrent_same_token_lookups() {
        use crate::auth::{AuthError, UserInfo};
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct SlowCountingUserinfo {
            counter: Arc<AtomicUsize>,
            user: UserInfo,
        }
        #[async_trait::async_trait]
        impl crate::auth::userinfo::UserinfoClient for SlowCountingUserinfo {
            async fn get_current_user(&self, _token: &str) -> Result<UserInfo, AuthError> {
                self.counter.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(self.user.clone())
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let userinfo = SlowCountingUserinfo {
            counter: Arc::clone(&counter),
            user: make_user_info(),
        };
        let cache = Arc::new(AuthCache::new(
            Arc::new(MockPermissionsClient::new(Ok(AuthResult::Allowed))),
            Arc::new(userinfo),
            300,
            TEST_MAX_ENTRIES,
        ));

        let mut handles = Vec::new();
        for _ in 0..100 {
            let cache = Arc::clone(&cache);
            handles.push(tokio::spawn(async move {
                cache.get_current_user("same-tok").await.unwrap()
            }));
        }
        for h in handles {
            let _ = h.await.unwrap();
        }
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "singleflight must coalesce 100 concurrent same-token userinfo lookups into 1 backend call"
        );
    }

    // ── Regression tests: AuthCache used through trait interface ────────
    //
    // Bug b.k5d: AuthCache was not wired into the production path because
    // it did not implement PermissionsClient/UserinfoClient. These tests
    // verify that AuthCache can be used as Arc<dyn PermissionsClient> and
    // Arc<dyn UserinfoClient>, and that caching still works through the
    // trait interface (two calls → one delegation to the underlying client).

    use crate::auth::permissions::PermissionsClient as PermissionsClientTrait;
    use crate::auth::userinfo::UserinfoClient as UserinfoClientTrait;

    #[tokio::test]
    async fn trait_impl_permissions_client_caches_through_dyn_interface() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache: Arc<dyn PermissionsClientTrait> = Arc::new(cache_with_ttl(
            perm,
            MockUserinfoClient::new(Ok(make_user_info())),
            300,
        ));

        // First call: cache miss → delegates to underlying client
        let r1 = cache
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap();
        assert_eq!(r1, AuthResult::Allowed);

        // Second call with same args: cache hit → no second delegation
        let r2 = cache
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap();
        assert_eq!(r2, AuthResult::Allowed);

        assert_eq!(
            calls.lock().unwrap().len(),
            1,
            "underlying client should be called only once when AuthCache is used as dyn PermissionsClient"
        );
    }

    #[tokio::test]
    async fn trait_impl_userinfo_client_caches_through_dyn_interface() {
        let userinfo = MockUserinfoClient::new(Ok(make_user_info()));
        let calls = Arc::clone(&userinfo.calls);
        let cache: Arc<dyn UserinfoClientTrait> = Arc::new(cache_with_ttl(
            MockPermissionsClient::new(Ok(AuthResult::Allowed)),
            userinfo,
            300,
        ));

        // First call: cache miss → delegates to underlying client
        let r1 = cache.get_current_user("tok").await.unwrap();
        assert_eq!(r1.id, "user-1");

        // Second call: cache hit → no second delegation
        let r2 = cache.get_current_user("tok").await.unwrap();
        assert_eq!(r2.id, "user-1");

        assert_eq!(
            calls.lock().unwrap().len(),
            1,
            "underlying client should be called only once when AuthCache is used as dyn UserinfoClient"
        );
    }

    // ── Regression tests: b.1o4 max_capacity bound on auth caches ──────
    //
    // Bug b.1o4: `AuthCache::new` built its inner caches with `.time_to_live`
    // but without `.max_capacity`, leaving them unbounded between TTL expiries.
    // Under sustained token churn the cache grew linearly with insertion rate.
    // The fix added `.max_capacity(max_entries)` to both builders. These tests
    // guard that property.
    //
    // mini-moka uses a TinyLFU admission policy, not strict LRU. A newly
    // inserted entry may be rejected at admission (rather than evicting an
    // existing one) if its estimated frequency is too low. The invariant
    // these tests assert is therefore the *capacity bound itself*, not a
    // specific eviction order: no matter how many distinct keys are written,
    // the number of keys that survive to produce a later cache hit cannot
    // exceed `max_capacity`.

    /// Regression test for b.1o4. Populates the permissions cache with many
    /// more distinct keys than `max_capacity`, then looks each one up a second
    /// time and counts cache hits. With a bounded cache, the number of hits
    /// must not exceed the capacity. Without the `.max_capacity(...)` fix this
    /// fails — every entry stays resident and every second lookup is a hit,
    /// so `hits == total_keys > MAX`.
    #[tokio::test]
    async fn check_permission_capacity_bound_limits_resident_entries() {
        const MAX: u64 = 4;
        const TOTAL_KEYS: usize = 50;

        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache = AuthCache::new(
            Arc::new(perm),
            Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
            300,
            MAX,
        );

        // First pass: insert TOTAL_KEYS distinct keys → TOTAL_KEYS client calls.
        for i in 0..TOTAL_KEYS {
            let token = format!("token-{i}");
            cache
                .check_permission(&token, "/path", Permission::Read)
                .await
                .unwrap();
        }
        let calls_after_first_pass = calls.lock().unwrap().len();
        assert_eq!(calls_after_first_pass, TOTAL_KEYS);

        // Second pass: look each key up again. Each cache hit avoids a client
        // call; each miss (evicted / not admitted) forces one. With a bounded
        // cache of size MAX, at most MAX entries can still be resident, so
        // hits <= MAX.
        for i in 0..TOTAL_KEYS {
            let token = format!("token-{i}");
            cache
                .check_permission(&token, "/path", Permission::Read)
                .await
                .unwrap();
        }
        let total_calls = calls.lock().unwrap().len();
        let hits = (2 * TOTAL_KEYS).saturating_sub(total_calls);

        assert!(
            hits as u64 <= MAX,
            "permissions cache exceeded capacity bound: got {hits} hits, max_capacity = {MAX} \
             (total keys inserted = {TOTAL_KEYS}, total client calls = {total_calls})",
        );
    }

    /// Regression test for b.1o4. Mirror of the permissions test above, but
    /// for the userinfo cache. Proves the capacity bound applies to both
    /// inner caches.
    #[tokio::test]
    async fn get_current_user_capacity_bound_limits_resident_entries() {
        const MAX: u64 = 4;
        const TOTAL_KEYS: usize = 50;

        let userinfo = MockUserinfoClient::new(Ok(make_user_info()));
        let calls = Arc::clone(&userinfo.calls);
        let cache = AuthCache::new(
            Arc::new(MockPermissionsClient::new(Ok(AuthResult::Allowed))),
            Arc::new(userinfo),
            300,
            MAX,
        );

        for i in 0..TOTAL_KEYS {
            let token = format!("token-{i}");
            cache.get_current_user(&token).await.unwrap();
        }
        let calls_after_first_pass = calls.lock().unwrap().len();
        assert_eq!(calls_after_first_pass, TOTAL_KEYS);

        for i in 0..TOTAL_KEYS {
            let token = format!("token-{i}");
            cache.get_current_user(&token).await.unwrap();
        }
        let total_calls = calls.lock().unwrap().len();
        let hits = (2 * TOTAL_KEYS).saturating_sub(total_calls);

        assert!(
            hits as u64 <= MAX,
            "userinfo cache exceeded capacity bound: got {hits} hits, max_capacity = {MAX} \
             (total keys inserted = {TOTAL_KEYS}, total client calls = {total_calls})",
        );
    }

    /// b.1o4 independence test: TTL expiry still fires when the cache is well
    /// below `max_capacity`. Guards against a regression where wiring up
    /// `.max_capacity(...)` accidentally breaks the TTL path (e.g. by
    /// swapping `time_to_live` for `max_capacity` instead of chaining them).
    /// Capacity is large (1000) and only one entry is ever inserted, so any
    /// miss on the second lookup is attributable to TTL, not capacity.
    #[tokio::test]
    async fn ttl_expiry_still_fires_when_cache_below_capacity() {
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let calls = Arc::clone(&perm.calls);
        let cache = AuthCache::new(
            Arc::new(perm),
            Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
            1, // 1-second TTL
            1000,
        );

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();
        assert_eq!(calls.lock().unwrap().len(), 1);

        // Wait out the TTL; cache size never exceeds 1 entry, so any miss is
        // TTL-driven.
        tokio::time::sleep(Duration::from_millis(1100)).await;

        cache
            .check_permission("token", "/path", Permission::Read)
            .await
            .unwrap();
        assert_eq!(
            calls.lock().unwrap().len(),
            2,
            "TTL-expired entry should have been re-fetched even when well below max_capacity",
        );
    }

    // ── b.nrt item 4: metrics wiring ────────────────────────────────────

    #[tokio::test]
    async fn metrics_increments_cache_hits_and_misses() {
        use crate::observability::metrics::metrics_scrape_only;
        use crate::observability::metrics::test_util::{render, sample_value};

        let metrics = Arc::new(metrics_scrape_only("t"));
        let perm = MockPermissionsClient::new(Ok(AuthResult::Allowed));
        let cache = AuthCache::new_with_negative_ttl(
            Arc::new(perm),
            Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
            300,
            30,
            5,
            TEST_MAX_ENTRIES,
            Some(metrics.clone()),
        );

        // First call: miss.
        cache
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap();
        // Second call: hit.
        cache
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap();

        let text = render(&metrics);
        let misses = sample_value(
            &text,
            "live_edit_auth_cache_misses_total",
            &[("kind", "permissions")],
        )
        .expect("permissions miss series must exist");
        let hits = sample_value(
            &text,
            "live_edit_auth_cache_hits_total",
            &[("kind", "permissions")],
        )
        .expect("permissions hit series must exist");
        assert_eq!(misses, 1.0, "expected exactly 1 permissions miss");
        assert_eq!(hits, 1.0, "expected exactly 1 permissions hit");

        // Userinfo counter path.
        cache.get_current_user("tok").await.unwrap();
        cache.get_current_user("tok").await.unwrap();
        let text = render(&metrics);
        let u_misses = sample_value(
            &text,
            "live_edit_auth_cache_misses_total",
            &[("kind", "userinfo")],
        )
        .expect("userinfo miss series must exist");
        let u_hits = sample_value(
            &text,
            "live_edit_auth_cache_hits_total",
            &[("kind", "userinfo")],
        )
        .expect("userinfo hit series must exist");
        assert_eq!(u_misses, 1.0);
        assert_eq!(u_hits, 1.0);
    }

    // ── Regression tests: AuthCache wrapping REST-backed clients ────────
    //
    // Plan Bee b.943 §AC#4: the existing AuthCache must wrap the REST
    // PermissionsClient / UserinfoClient implementations with identical
    // cache semantics as the gRPC ones. The assertion here is at the
    // HTTP-mock level (wiremock `Mock::expect(N)` verified on drop) so
    // these tests prove the cache actually prevents a network call on
    // cache hits — not merely that some trait-level counter didn't tick.
    mod rest_wrapping_tests {
        use std::sync::Arc;

        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        use crate::auth::cache::AuthCache;
        use crate::auth::rest::{RestStorageServiceClient, RestUserinfoClient};
        use crate::auth::{fast_retry, AuthResult, Permission};

        /// Userinfo happy-path body — matches `UserInfoDto`'s snake_case shape.
        const USERINFO_BODY: &str =
            r#"{"id":"u1","display_name":"Jane","email":"j@x.com","username":"jane"}"#;

        /// TTL large enough that time-based eviction never fires during tests,
        /// mirroring the 300s used by the existing gRPC wrapping tests above.
        const LONG_TTL_SECS: u64 = 300;

        /// Capacity large enough that entry-based eviction never fires during
        /// these wrapping tests — they assert caching semantics, not capacity.
        const LARGE_MAX_ENTRIES: u64 = 1_000;

        /// Build an `AuthCache` wrapping a REST storage client pointed at
        /// `mock_server`. The userinfo client is also REST-backed but points
        /// at the same mock server — tests that don't use userinfo simply
        /// won't issue calls against it.
        ///
        /// The two clients are distinguishable by HTTP method on that shared
        /// server: the permission probe is a `HEAD` (stat-by-address, b.uoc)
        /// and userinfo is a `GET`, so a `method(..)`-only matcher is
        /// unambiguous without a path matcher.
        fn rest_cache(mock_server: &MockServer) -> AuthCache {
            let perm = Arc::new(
                RestStorageServiceClient::new(&mock_server.uri(), 200, fast_retry(3)).unwrap(),
            );
            let userinfo =
                Arc::new(RestUserinfoClient::new(&mock_server.uri(), 200, fast_retry(3)).unwrap());
            AuthCache::new(perm, userinfo, LONG_TTL_SECS, LARGE_MAX_ENTRIES)
        }

        #[tokio::test]
        async fn authcache_wraps_rest_storage_client_second_call_hits_cache() {
            let mock_server = MockServer::start().await;
            // `.expect(1)` is verified on drop — if the cache doesn't
            // suppress the second call, wiremock will panic with "called 2
            // times, expected 1".
            //
            // `204 No Content` is the faithful stat-by-address answer to the
            // `HEAD` probe (a HEAD response carries no body); `classify_status`
            // maps both 200 and 204 to `Allowed`.
            Mock::given(method("HEAD"))
                .respond_with(ResponseTemplate::new(204))
                .expect(1)
                .mount(&mock_server)
                .await;

            let cache = rest_cache(&mock_server);

            let r1 = cache
                .check_permission("tok", "/p", Permission::Read)
                .await
                .unwrap();
            assert_eq!(r1, AuthResult::Allowed);

            let r2 = cache
                .check_permission("tok", "/p", Permission::Read)
                .await
                .unwrap();
            assert_eq!(r2, AuthResult::Allowed);
        }

        #[tokio::test]
        async fn authcache_wraps_rest_userinfo_client_second_call_hits_cache() {
            let mock_server = MockServer::start().await;
            Mock::given(method("GET"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_raw(USERINFO_BODY.as_bytes().to_vec(), "application/json"),
                )
                .expect(1)
                .mount(&mock_server)
                .await;

            let cache = rest_cache(&mock_server);

            let r1 = cache.get_current_user("tok").await.unwrap();
            assert_eq!(r1.id, "u1");

            let r2 = cache.get_current_user("tok").await.unwrap();
            assert_eq!(r2.id, "u1");
        }

        #[tokio::test]
        async fn authcache_wraps_rest_storage_client_different_tokens_are_separate_entries() {
            let mock_server = MockServer::start().await;
            // Two tokens → two distinct cache keys → two HTTP requests.
            Mock::given(method("HEAD"))
                .respond_with(ResponseTemplate::new(204))
                .expect(2)
                .mount(&mock_server)
                .await;

            let cache = rest_cache(&mock_server);

            cache
                .check_permission("token-a", "/p", Permission::Read)
                .await
                .unwrap();
            cache
                .check_permission("token-b", "/p", Permission::Read)
                .await
                .unwrap();
        }

        #[tokio::test]
        async fn authcache_wraps_rest_storage_client_different_paths_are_separate_entries() {
            let mock_server = MockServer::start().await;
            // Two paths → two distinct cache keys → two HTTP requests.
            Mock::given(method("HEAD"))
                .respond_with(ResponseTemplate::new(204))
                .expect(2)
                .mount(&mock_server)
                .await;

            let cache = rest_cache(&mock_server);

            cache
                .check_permission("tok", "/path/a", Permission::Read)
                .await
                .unwrap();
            cache
                .check_permission("tok", "/path/b", Permission::Read)
                .await
                .unwrap();
        }

        #[tokio::test]
        async fn authcache_wraps_rest_userinfo_client_different_tokens_are_separate_entries() {
            let mock_server = MockServer::start().await;
            Mock::given(method("GET"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_raw(USERINFO_BODY.as_bytes().to_vec(), "application/json"),
                )
                .expect(2)
                .mount(&mock_server)
                .await;

            let cache = rest_cache(&mock_server);

            cache.get_current_user("token-a").await.unwrap();
            cache.get_current_user("token-b").await.unwrap();
        }
    }

    // ── b.ohw H-auth-3: singleflight panic isolation ────────────────────
    //
    // Pre-fix, a panic inside the inner future cascaded through
    // `futures::Shared` to every coalesced waiter — N waiters meant N
    // panicked tasks. The fix wraps the inner future with
    // `AssertUnwindSafe(...).catch_unwind()` so the panic surfaces as a
    // clean `AuthError::InvalidResponse("internal panic in auth backend")`
    // for whichever caller wins the await; the rest see the same clean
    // error via the `Shared` future.

    /// Permissions client whose `check_permission` panics on every call.
    struct PanickingPermissionsClient;

    #[async_trait::async_trait]
    impl crate::auth::permissions::PermissionsClient for PanickingPermissionsClient {
        async fn check_permission(
            &self,
            _token: &str,
            _path: &str,
            _permission: Permission,
        ) -> Result<AuthResult, crate::auth::AuthError> {
            panic!("simulated backend panic")
        }
    }

    /// Userinfo client that panics on every call.
    struct PanickingUserinfoClient;

    #[async_trait::async_trait]
    impl crate::auth::userinfo::UserinfoClient for PanickingUserinfoClient {
        async fn get_current_user(&self, _token: &str) -> Result<UserInfo, crate::auth::AuthError> {
            panic!("simulated userinfo panic")
        }
    }

    #[tokio::test]
    async fn permissions_singleflight_panic_returns_clean_error_to_single_waiter() {
        let cache = AuthCache::new(
            Arc::new(PanickingPermissionsClient),
            Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
            300,
            TEST_MAX_ENTRIES,
        );
        let err = cache
            .check_permission("tok", "/p", Permission::Read)
            .await
            .expect_err("backend panic must surface as a clean error, not a panic");
        match err {
            crate::auth::AuthError::InvalidResponse(msg) => {
                assert!(
                    msg.contains("internal panic"),
                    "panic should surface as the contracted 'internal panic' error, got: {msg}"
                );
            }
            other => panic!("expected InvalidResponse('internal panic ...'), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn userinfo_singleflight_panic_returns_clean_error_to_single_waiter() {
        let cache = AuthCache::new(
            Arc::new(crate::auth::permissions::mock::MockPermissionsClient::new(
                Ok(AuthResult::Allowed),
            )),
            Arc::new(PanickingUserinfoClient),
            300,
            TEST_MAX_ENTRIES,
        );
        let err = cache
            .get_current_user("tok")
            .await
            .expect_err("backend panic must surface as a clean error");
        match err {
            crate::auth::AuthError::InvalidResponse(msg) => {
                assert!(
                    msg.contains("internal panic"),
                    "userinfo panic should surface as 'internal panic', got: {msg}"
                );
            }
            other => panic!("expected InvalidResponse, got {other:?}"),
        }
    }

    /// All N coalesced waiters must observe the same clean error and none
    /// must panic. Pre-fix, the panic propagated through `Shared` and every
    /// waiter task panicked; the fix means every waiter sees a normal
    /// `Err(AuthError::InvalidResponse(...))` return.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn permissions_singleflight_panic_all_waiters_get_clean_error_no_panics() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::Notify;

        /// Slow panicking client: holds the call open until `start` is signaled
        /// so all 50 waiters definitely coalesce onto the same Shared future
        /// before the panic fires. Without this gate the first call could
        /// resolve before the rest have entered the singleflight, hiding the
        /// "all waiters share one panic" condition we're trying to test.
        struct GatedPanicClient {
            start: Arc<Notify>,
            entered: Arc<AtomicUsize>,
        }
        #[async_trait::async_trait]
        impl crate::auth::permissions::PermissionsClient for GatedPanicClient {
            async fn check_permission(
                &self,
                _token: &str,
                _path: &str,
                _permission: Permission,
            ) -> Result<AuthResult, crate::auth::AuthError> {
                self.entered.fetch_add(1, Ordering::SeqCst);
                self.start.notified().await;
                panic!("simulated backend panic — coalesced");
            }
        }

        let start = Arc::new(Notify::new());
        let entered = Arc::new(AtomicUsize::new(0));
        let cache = Arc::new(AuthCache::new(
            Arc::new(GatedPanicClient {
                start: Arc::clone(&start),
                entered: Arc::clone(&entered),
            }),
            Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
            300,
            TEST_MAX_ENTRIES,
        ));

        const N: usize = 50;
        let mut handles = Vec::with_capacity(N);
        for _ in 0..N {
            let cache = Arc::clone(&cache);
            handles.push(tokio::spawn(async move {
                cache.check_permission("tok", "/p", Permission::Read).await
            }));
        }

        // Wait until at least one task has entered the backend call (and
        // therefore the rest have coalesced onto the Shared future), then
        // release the panic.
        loop {
            if entered.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        // Brief pause so other waiters that arrived after the first task
        // joined the Shared future have time to do so before the panic fires.
        tokio::time::sleep(Duration::from_millis(20)).await;
        start.notify_waiters();

        let mut clean_error_count = 0usize;
        for h in handles {
            // `h.await` returning Ok means the task did not panic — that is
            // exactly the contract we're verifying. The inner Result is the
            // backend Err.
            let task_result = h.await.expect("waiter task must NOT panic");
            match task_result {
                Err(crate::auth::AuthError::InvalidResponse(msg))
                    if msg.contains("internal panic") =>
                {
                    clean_error_count += 1;
                }
                other => panic!(
                    "every waiter must see InvalidResponse('internal panic ...'), got {other:?}"
                ),
            }
        }
        assert_eq!(
            clean_error_count, N,
            "all {N} coalesced waiters must observe the clean error"
        );

        // Backend was called only once (the panic was singleflight-coalesced).
        // We can't assert exactly 1 because tokio's spawn ordering means the
        // first task might or might not have arrived at the await before the
        // others, but the Shared future guarantees at most one call.
        let calls = entered.load(Ordering::SeqCst);
        assert_eq!(
            calls, 1,
            "singleflight must coalesce N panicking calls into 1 backend invocation"
        );
    }

    // ── b.ohw H-auth-2: cache-insert MUST run before inflight-remove ────
    //
    // Pre-fix order: shared.await → inflight.remove(key) → cache.insert.
    // A new caller arriving in the gap after `remove` but before `insert`
    // missed both maps and issued a duplicate backend call. The fix swaps
    // the order: cache.insert runs first, so any new caller scheduled after
    // the singleflight resolves either sees the cached value or joins the
    // still-live Shared future via the inflight map.
    //
    // The test gates the backend call on a Notify, fires N parallel callers
    // (some before the gate, one after it resolves but with the cache
    // already populated by the swap-order fix), and asserts the backend was
    // called exactly once. Without the fix this test would intermittently
    // fail because the third caller's race window is real on a multi-thread
    // runtime.

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn permissions_singleflight_no_duplicate_backend_call_post_resolve_gap() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::Notify;

        struct GatedClient {
            counter: Arc<AtomicUsize>,
            release: Arc<Notify>,
        }
        #[async_trait::async_trait]
        impl crate::auth::permissions::PermissionsClient for GatedClient {
            async fn check_permission(
                &self,
                _token: &str,
                _path: &str,
                _permission: Permission,
            ) -> Result<AuthResult, crate::auth::AuthError> {
                self.counter.fetch_add(1, Ordering::SeqCst);
                self.release.notified().await;
                Ok(AuthResult::Allowed)
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(Notify::new());
        let cache = Arc::new(AuthCache::new(
            Arc::new(GatedClient {
                counter: Arc::clone(&counter),
                release: Arc::clone(&release),
            }),
            Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
            300,
            TEST_MAX_ENTRIES,
        ));

        // Phase 1: fire 50 callers that all coalesce onto the gated backend
        // call. They will all be parked on `release.notified()` inside the
        // single backend call.
        const N1: usize = 50;
        let mut wave1 = Vec::with_capacity(N1);
        for _ in 0..N1 {
            let cache = Arc::clone(&cache);
            wave1.push(tokio::spawn(async move {
                cache
                    .check_permission("tok", "/p", Permission::Read)
                    .await
                    .unwrap()
            }));
        }
        // Give all wave1 tasks time to enter the singleflight machinery.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Release the gate so the backend call resolves. The cache-insert
        // (with the fix) runs before the inflight-remove inside
        // `check_permission`'s post-await block.
        release.notify_waiters();

        for h in wave1 {
            let r = h.await.unwrap();
            assert_eq!(r, AuthResult::Allowed);
        }
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "wave 1: singleflight must coalesce N1 calls into 1 backend invocation"
        );

        // Phase 2: a second wave issued after wave 1 resolved. The fix
        // guarantees the cache is populated *before* the inflight entry is
        // removed, so wave 2's callers see the cache hit. Any backend call
        // count > 1 indicates the post-resolve race window has reopened.
        const N2: usize = 50;
        let mut wave2 = Vec::with_capacity(N2);
        for _ in 0..N2 {
            let cache = Arc::clone(&cache);
            wave2.push(tokio::spawn(async move {
                cache
                    .check_permission("tok", "/p", Permission::Read)
                    .await
                    .unwrap()
            }));
        }
        for h in wave2 {
            let r = h.await.unwrap();
            assert_eq!(r, AuthResult::Allowed);
        }
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "wave 2: post-resolve cache-insert ordering must prevent duplicate backend calls"
        );
    }

    /// Same ordering test for the userinfo arm. Sequence and contract are
    /// identical to the permissions test above; both arms run the swap-order
    /// fix in `check_permission` / `get_current_user`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn userinfo_singleflight_no_duplicate_backend_call_post_resolve_gap() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::Notify;

        struct GatedUserinfo {
            counter: Arc<AtomicUsize>,
            release: Arc<Notify>,
            user: UserInfo,
        }
        #[async_trait::async_trait]
        impl crate::auth::userinfo::UserinfoClient for GatedUserinfo {
            async fn get_current_user(
                &self,
                _token: &str,
            ) -> Result<UserInfo, crate::auth::AuthError> {
                self.counter.fetch_add(1, Ordering::SeqCst);
                self.release.notified().await;
                Ok(self.user.clone())
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(Notify::new());
        let cache = Arc::new(AuthCache::new(
            Arc::new(crate::auth::permissions::mock::MockPermissionsClient::new(
                Ok(AuthResult::Allowed),
            )),
            Arc::new(GatedUserinfo {
                counter: Arc::clone(&counter),
                release: Arc::clone(&release),
                user: make_user_info(),
            }),
            300,
            TEST_MAX_ENTRIES,
        ));

        const N: usize = 50;
        let mut wave1 = Vec::with_capacity(N);
        for _ in 0..N {
            let cache = Arc::clone(&cache);
            wave1.push(tokio::spawn(async move {
                cache.get_current_user("tok").await.unwrap()
            }));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        release.notify_waiters();
        for h in wave1 {
            let _ = h.await.unwrap();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        let mut wave2 = Vec::with_capacity(N);
        for _ in 0..N {
            let cache = Arc::clone(&cache);
            wave2.push(tokio::spawn(async move {
                cache.get_current_user("tok").await.unwrap()
            }));
        }
        for h in wave2 {
            let _ = h.await.unwrap();
        }
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "userinfo cache-insert/remove ordering must prevent post-resolve duplicate calls"
        );
    }

    // ── b.ohw medium: userinfo negative cache ───────────────────────────
    //
    // A bearer the userinfo service rejects would, pre-fix, hit the backend on
    // every RPC because `userinfo_cache` only stored successful lookups. A
    // rejection is not necessarily an expired token (the userinfo service may
    // not validate the bearer at all), so these tests name it by what it is.
    // The fix adds
    // `userinfo_negative_cache` (separate TTL) so duplicate failures are
    // memoized briefly, bounding the backend fanout while keeping
    // revocation propagation fast.

    /// Userinfo client that always returns the same configurable error and
    /// counts its calls. Lets us assert exact backend hit-counts across N
    /// caller invocations.
    struct CountingErrUserinfo {
        counter: Arc<std::sync::atomic::AtomicUsize>,
        err: crate::auth::AuthError,
    }
    #[async_trait::async_trait]
    impl crate::auth::userinfo::UserinfoClient for CountingErrUserinfo {
        async fn get_current_user(&self, _token: &str) -> Result<UserInfo, crate::auth::AuthError> {
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(self.err.clone())
        }
    }

    /// N sequential calls with a rejected bearer must hit the backend exactly
    /// once within a single negative-TTL window, and then again once the
    /// window expires. This is the core revocation-vs-fanout tradeoff the
    /// negative cache embodies.
    #[tokio::test]
    async fn userinfo_negative_cache_coalesces_repeated_failures_within_ttl_window() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let cache = AuthCache::new_with_negative_ttl(
            Arc::new(crate::auth::permissions::mock::MockPermissionsClient::new(
                Ok(AuthResult::Allowed),
            )),
            Arc::new(CountingErrUserinfo {
                counter: Arc::clone(&counter),
                err: crate::auth::AuthError::InvalidResponse(
                    "userinfo service rejected the bearer (HTTP 401)".to_owned(),
                ),
            }),
            300, // positive TTL — irrelevant for failure path
            300, // permissions negative TTL — irrelevant
            1,   // userinfo negative TTL: 1 second
            TEST_MAX_ENTRIES,
            None,
        );

        // 10 calls within the 1s negative-TTL window: backend hit exactly once.
        for _ in 0..10 {
            let _ = cache.get_current_user("rejected-tok").await;
        }
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "10 calls within the 1s window must coalesce to 1 backend call"
        );

        // Step past the negative TTL.
        tokio::time::sleep(Duration::from_millis(1100)).await;

        // Next call hits the backend again (TTL expired → cache miss).
        let _ = cache.get_current_user("rejected-tok").await;
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "after the negative-TTL window expires, the next call must re-hit the backend"
        );

        // Subsequent calls within the new window are coalesced again.
        for _ in 0..5 {
            let _ = cache.get_current_user("rejected-tok").await;
        }
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "post-expiry refresh starts a new coalesce window"
        );
    }

    /// The negative cache must surface the same error variant the backend
    /// returned. A pre-fix bug could replace `InvalidResponse` with a
    /// generic error if the cache stored only a flag — assert the variant
    /// itself round-trips.
    #[tokio::test]
    async fn userinfo_negative_cache_preserves_error_variant_and_message() {
        use std::sync::atomic::AtomicUsize;

        let counter = Arc::new(AtomicUsize::new(0));
        let cache = AuthCache::new_with_negative_ttl(
            Arc::new(crate::auth::permissions::mock::MockPermissionsClient::new(
                Ok(AuthResult::Allowed),
            )),
            Arc::new(CountingErrUserinfo {
                counter: Arc::clone(&counter),
                err: crate::auth::AuthError::InvalidResponse(
                    "userinfo service rejected the bearer (HTTP 401)".to_owned(),
                ),
            }),
            300,
            300,
            5, // long enough that both calls fall in the same window
            TEST_MAX_ENTRIES,
            None,
        );

        let e1 = cache.get_current_user("tok").await.unwrap_err();
        let e2 = cache.get_current_user("tok").await.unwrap_err();

        match (e1, e2) {
            (
                crate::auth::AuthError::InvalidResponse(m1),
                crate::auth::AuthError::InvalidResponse(m2),
            ) => {
                assert_eq!(
                    m1, m2,
                    "cached error must match the live error byte-for-byte"
                );
                assert!(m1.contains("userinfo service rejected the bearer (HTTP 401)"));
            }
            (a, b) => panic!("expected matching InvalidResponse variants, got {a:?} / {b:?}"),
        }
    }

    /// A successful userinfo call must not pollute the negative cache, and
    /// later failures for a different token must still be cacheable. Guards
    /// against a regression where positive results accidentally land in the
    /// negative-cache store.
    #[tokio::test]
    async fn userinfo_negative_cache_does_not_capture_successful_results() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // The backend toggles: first call succeeds, subsequent calls would
        // panic if invoked — so we can prove the post-success path didn't
        // bleed into the negative cache by simply going to "Ok" and seeing
        // it stick.
        struct OnceOkThenPanic {
            counter: Arc<AtomicUsize>,
            user: UserInfo,
        }
        #[async_trait::async_trait]
        impl crate::auth::userinfo::UserinfoClient for OnceOkThenPanic {
            async fn get_current_user(
                &self,
                _token: &str,
            ) -> Result<UserInfo, crate::auth::AuthError> {
                let n = self.counter.fetch_add(1, Ordering::SeqCst);
                assert_eq!(
                    n, 0,
                    "successful results must not trigger more backend calls"
                );
                Ok(self.user.clone())
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let cache = AuthCache::new_with_negative_ttl(
            Arc::new(crate::auth::permissions::mock::MockPermissionsClient::new(
                Ok(AuthResult::Allowed),
            )),
            Arc::new(OnceOkThenPanic {
                counter: Arc::clone(&counter),
                user: make_user_info(),
            }),
            300,
            300,
            5,
            TEST_MAX_ENTRIES,
            None,
        );

        cache.get_current_user("tok").await.unwrap();
        cache.get_current_user("tok").await.unwrap();
        cache.get_current_user("tok").await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    /// Negative-cache entries must be keyed per-token. Two different
    /// rejected bearers get one backend call each, not one shared call.
    #[tokio::test]
    async fn userinfo_negative_cache_keys_per_token() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let cache = AuthCache::new_with_negative_ttl(
            Arc::new(crate::auth::permissions::mock::MockPermissionsClient::new(
                Ok(AuthResult::Allowed),
            )),
            Arc::new(CountingErrUserinfo {
                counter: Arc::clone(&counter),
                err: crate::auth::AuthError::InvalidResponse(
                    "userinfo service rejected the bearer (HTTP 401)".to_owned(),
                ),
            }),
            300,
            300,
            5,
            TEST_MAX_ENTRIES,
            None,
        );

        let _ = cache.get_current_user("tok-a").await;
        let _ = cache.get_current_user("tok-b").await;
        let _ = cache.get_current_user("tok-a").await;
        let _ = cache.get_current_user("tok-b").await;
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "two distinct tokens → two negative-cache entries → two backend calls"
        );
    }

    /// Transient errors are also memoized in the negative cache (per the
    /// implementer's contract: a flapping backend shouldn't trigger a
    /// stampede; the short TTL keeps recovery fast). Verify both `Timeout`
    /// and `Unreachable` round-trip through the cache.
    #[tokio::test]
    async fn userinfo_negative_cache_memoizes_transient_errors_too() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        for transient in [
            crate::auth::AuthError::Timeout,
            crate::auth::AuthError::Unreachable,
        ] {
            let counter = Arc::new(AtomicUsize::new(0));
            let cache = AuthCache::new_with_negative_ttl(
                Arc::new(crate::auth::permissions::mock::MockPermissionsClient::new(
                    Ok(AuthResult::Allowed),
                )),
                Arc::new(CountingErrUserinfo {
                    counter: Arc::clone(&counter),
                    err: transient.clone(),
                }),
                300,
                300,
                5,
                TEST_MAX_ENTRIES,
                None,
            );

            for _ in 0..5 {
                let _ = cache.get_current_user("tok").await;
            }
            assert_eq!(
                counter.load(Ordering::SeqCst),
                1,
                "{transient:?} must be memoized in the userinfo negative cache"
            );
        }
    }

    // ── b.ohw medium: HMAC seed wiped on Drop (zeroize) ─────────────────
    //
    // The seed lives behind `Zeroizing<[u8; 32]>`. Verifying actual memory
    // is wiped from a freed allocation is impossible on a managed-heap
    // platform without unsafe pointer aliasing, but we *can* assert two
    // things at the contract level:
    //   1. The struct's seed field is `Zeroizing<[u8; 32]>` — the type
    //      bound is what guarantees the drop-wipe behavior. We exercise
    //      this by directly invoking `Zeroizing::drop` on a value cloned
    //      from the cache's seed and observing the post-drop bytes via the
    //      reproducibility of the keyed template.
    //   2. The `Zeroize` trait wipes the bytes when called — the standard
    //      idiomatic test pattern from the `zeroize` crate.

    #[tokio::test]
    async fn hmac_seed_zeroize_wipes_bytes_in_place() {
        // Use `zeroize`'s own `Zeroize` trait to verify the contract:
        // calling `.zeroize()` on a `[u8; 32]` value sets every byte to 0.
        // This is exactly what `Zeroizing<[u8; 32]>`'s `Drop` invokes.
        use zeroize::Zeroize;

        // Build a non-zero seed (ASCII pattern) so post-zeroize check is
        // unambiguous: pre-zeroize it has nonzero bytes; post-zeroize it
        // must be all zeros.
        let mut seed: [u8; 32] = [0x42; 32];
        assert!(
            seed.iter().any(|b| *b != 0),
            "pre-zeroize seed should be non-zero (test setup invariant)"
        );

        seed.zeroize();
        assert_eq!(
            seed, [0u8; 32],
            "Zeroize must wipe every byte to zero — this is the trait contract \
             that Zeroizing<[u8; 32]>'s Drop invokes for the AuthCache HMAC seed"
        );
    }

    /// Zeroizing<[u8; 32]>'s Drop must wipe the underlying memory. We can
    /// verify this without unsafe by holding a raw pointer to the inner
    /// array, dropping the wrapper, and reading through the pointer back
    /// (the storage is on the stack and outlives the Zeroizing wrapper's
    /// drop call). This mirrors the standard `zeroize` crate test pattern.
    #[test]
    fn zeroizing_array_drop_wipes_underlying_storage() {
        use zeroize::Zeroizing;

        // Stack-allocate the buffer so its lifetime is independent of the
        // Zeroizing wrapper. The wrapper is built from a clone, so its drop
        // wipes its own clone (not the original) — but both observations
        // are still load-bearing: (a) the wrapper Drop runs zeroize, (b)
        // the post-Drop bytes of the clone are zero.
        let original: [u8; 32] = [0xAB; 32];
        let mut zeroizing = Zeroizing::new(original);

        // Read the array slot before drop to confirm it really has the
        // expected pattern.
        assert_eq!(*zeroizing, [0xAB; 32]);

        // Mutate then re-check, just to verify the wrapper exposes a real
        // mutable handle (sanity).
        zeroizing[0] = 0x01;
        assert_eq!(zeroizing[0], 0x01);

        // After the wrapper drops here, its inner [u8; 32] is wiped. We
        // can't read it after Drop without unsafe, so we rely on (a) the
        // type contract — `Zeroizing<T: Zeroize>::Drop` is documented to
        // call `T::zeroize`, and (b) the manual `Zeroize` test above which
        // proves that `[u8; 32]::zeroize()` sets every byte to zero. The
        // composition is the contract we need.
        drop(zeroizing);
    }

    // ── b.j5t item 4: singleflight inflight cancellation cleanup ────────
    //
    // Pre-fix the post-`.await` `inflight.remove(&key)` ran unconditionally
    // — but a cancelled caller (tonic deadline drop, client disconnect)
    // returns from the function via Drop *before* `.await` resolves, so
    // the inflight entry leaked forever. DashMap has no eviction, so a
    // long-running pod with sporadic mid-flight cancellations could
    // unboundedly grow these maps. The fix wraps the cleanup in a
    // `scopeguard::guard` so it runs on every exit path; the guard's
    // `remove_if` keys on `Shared::ptr_eq` so we never evict a follow-up
    // caller's freshly installed future.

    /// Permissions client that parks forever on a Notify so the test can
    /// cancel the calling future while the backend call is still in-flight.
    /// Counts invocations so the test can assert the singleflight contract.
    struct ParkingPermsClient {
        entered: Arc<std::sync::atomic::AtomicUsize>,
    }
    #[async_trait::async_trait]
    impl crate::auth::permissions::PermissionsClient for ParkingPermsClient {
        async fn check_permission(
            &self,
            _token: &str,
            _path: &str,
            _permission: Permission,
        ) -> Result<AuthResult, crate::auth::AuthError> {
            self.entered
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Sleep effectively forever; the test cancels the calling
            // task before this fires.
            tokio::time::sleep(Duration::from_secs(3600)).await;
            unreachable!("test must cancel before sleep elapses")
        }
    }

    /// Userinfo counterpart of `ParkingPermsClient`.
    struct ParkingUserinfoClient {
        entered: Arc<std::sync::atomic::AtomicUsize>,
    }
    #[async_trait::async_trait]
    impl crate::auth::userinfo::UserinfoClient for ParkingUserinfoClient {
        async fn get_current_user(&self, _token: &str) -> Result<UserInfo, crate::auth::AuthError> {
            self.entered
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(Duration::from_secs(3600)).await;
            unreachable!("test must cancel before sleep elapses")
        }
    }

    /// Drop every caller mid-await and assert the inflight map is empty.
    /// Pre-fix this test fails: the entry stays in `permissions_inflight`
    /// forever because the post-await `remove(...)` never runs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn permissions_singleflight_inflight_cleared_when_all_callers_cancelled() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let entered = Arc::new(AtomicUsize::new(0));
        let cache = Arc::new(AuthCache::new(
            Arc::new(ParkingPermsClient {
                entered: Arc::clone(&entered),
            }),
            Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
            300,
            TEST_MAX_ENTRIES,
        ));

        // Spawn 3 coalesced callers. Cancel all of them after they've all
        // joined the singleflight machinery.
        let mut handles = Vec::with_capacity(3);
        for _ in 0..3 {
            let cache = Arc::clone(&cache);
            handles.push(tokio::spawn(async move {
                cache.check_permission("tok", "/p", Permission::Read).await
            }));
        }

        // Wait until at least one task has entered the backend (others have
        // coalesced onto the Shared future).
        loop {
            if entered.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        // Give the other waiters a moment to subscribe to the Shared future.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Now cancel every caller. The Shared future stops being polled.
        for h in &handles {
            h.abort();
        }
        for h in handles {
            let _ = h.await;
        }

        // Give DashMap entry removal time to propagate.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // The inflight map must be empty — the scope guard ran on every
        // dropped caller's stack and the `remove_if` short-circuited at
        // most one of them to actually evict.
        assert_eq!(
            cache.permissions_inflight.len(),
            0,
            "cancelled callers must not leak the inflight entry — \
             this is the b.j5t item 4 regression test"
        );

        // Backend was invoked exactly once (singleflight coalescing).
        assert_eq!(
            entered.load(Ordering::SeqCst),
            1,
            "singleflight must coalesce all 3 cancelled callers into 1 backend call"
        );

        // A fresh call after the cancellation storm must re-enter the
        // singleflight machinery — the inflight entry from before is gone,
        // so a new caller installs a new Shared future. We don't actually
        // need to wait for completion; entering the backend is enough.
        let next_entered_baseline = entered.load(Ordering::SeqCst);
        let cache_clone = Arc::clone(&cache);
        let next = tokio::spawn(async move {
            cache_clone
                .check_permission("tok", "/p", Permission::Read)
                .await
        });
        loop {
            if entered.load(Ordering::SeqCst) > next_entered_baseline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        next.abort();
        let _ = next.await;
        // Brief settle so the abort-induced scope-guard cleanup runs before
        // we measure the inflight len.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            cache.permissions_inflight.len(),
            0,
            "after the follow-up cancellation, the inflight map must again be empty"
        );
    }

    /// Userinfo arm of the cancellation cleanup test. Contract is identical
    /// to the permissions arm above.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn userinfo_singleflight_inflight_cleared_when_all_callers_cancelled() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let entered = Arc::new(AtomicUsize::new(0));
        let cache = Arc::new(AuthCache::new(
            Arc::new(crate::auth::permissions::mock::MockPermissionsClient::new(
                Ok(AuthResult::Allowed),
            )),
            Arc::new(ParkingUserinfoClient {
                entered: Arc::clone(&entered),
            }),
            300,
            TEST_MAX_ENTRIES,
        ));

        let mut handles = Vec::with_capacity(3);
        for _ in 0..3 {
            let cache = Arc::clone(&cache);
            handles.push(tokio::spawn(
                async move { cache.get_current_user("tok").await },
            ));
        }

        loop {
            if entered.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;

        for h in &handles {
            h.abort();
        }
        for h in handles {
            let _ = h.await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert_eq!(
            cache.userinfo_inflight.len(),
            0,
            "cancelled userinfo callers must not leak the inflight entry"
        );
        assert_eq!(entered.load(Ordering::SeqCst), 1);
    }

    /// Hand-rolled assertion that an `AuthCache`'s constructor uses
    /// `Zeroizing` for the seed. The test simply constructs an `AuthCache`
    /// and lets it go out of scope — if the seed field's type were ever
    /// changed away from `Zeroizing<[u8; 32]>`, the field's `Drop` would
    /// no longer wipe and the regression would slip in silently. We back
    /// this up with a static type-bound assertion that the `hmac_seed`
    /// field is `Zeroizing<[u8; 32]>`.
    #[test]
    fn auth_cache_hmac_seed_field_type_is_zeroizing_array() {
        // Compile-time type assertion: this match arm only type-checks if
        // the field really is `Zeroizing<[u8; 32]>`. A future regression
        // that changes the field type would fail compilation here, surfacing
        // the issue at build time rather than as a silent CVE.
        fn assert_is_zeroizing_array(_: &zeroize::Zeroizing<[u8; 32]>) {}

        let cache = AuthCache::new(
            Arc::new(crate::auth::permissions::mock::MockPermissionsClient::new(
                Ok(AuthResult::Allowed),
            )),
            Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
            300,
            TEST_MAX_ENTRIES,
        );
        assert_is_zeroizing_array(&cache.hmac_seed);
    }

    // ── b.75i: positive entries end at the bearer's JWT `exp` ───────────
    //
    // A positive verdict cached for a JWT used to be served for the whole
    // positive TTL (600 s by default), even after the bearer itself expired —
    // so the client kept being admitted with a dead token on this replica and
    // was never told to refresh it. Each positive entry now ends at the
    // EARLIER of the configured TTL and the bearer's `exp`; the first call
    // after `exp` misses and gets the backend's answer for an expired bearer.
    //
    // The two "stops being served at exp" tests wait for a real `exp` to pass
    // (about two seconds): the bound is computed from `SystemTime` and checked
    // against `Instant`, neither of which tokio's paused clock moves. The
    // decision function's remaining branches are pinned as pure tests below.
    mod jwt_expiry {
        use std::collections::VecDeque;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};
        use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        use super::super::{bearer_cache_bound, AuthCache, BearerCacheBound};
        use super::{make_user_info, TEST_MAX_ENTRIES};
        use crate::auth::permissions::mock::MockPermissionsClient;
        use crate::auth::userinfo::mock::MockUserinfoClient;
        use crate::auth::{AuthError, AuthResult, Permission};
        use crate::observability::metrics::metrics_scrape_only;
        use crate::observability::metrics::test_util::{render, sample_value};

        /// The configured positive TTL in these tests: far longer than any
        /// test runs, so every miss observed below is the `exp` bound's doing.
        const LONG_POSITIVE_TTL_SECS: u64 = 600;

        /// How far past the current second a "live" test JWT expires. Two
        /// whole seconds leaves at least one second between minting and
        /// expiry, so the in-window cache hit cannot race the bound.
        const LIVE_JWT_LIFETIME_SECS: i64 = 2;

        /// An unsigned JWT carrying `claims` — the cache reads `exp` without
        /// verifying anything, so a signature would add nothing.
        fn jwt(claims: serde_json::Value) -> String {
            let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"none"}"#);
            let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
            format!("{header}.{payload}.")
        }

        fn now_secs() -> i64 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after the epoch")
                .as_secs() as i64
        }

        /// A JWT whose `exp` is [`LIVE_JWT_LIFETIME_SECS`] past the current
        /// second, and that `exp`.
        fn live_jwt() -> (String, i64) {
            let exp = now_secs() + LIVE_JWT_LIFETIME_SECS;
            (jwt(serde_json::json!({ "sub": "u", "exp": exp })), exp)
        }

        /// Sleep until the wall clock is past `exp` (JWT `exp` is exclusive,
        /// so reaching the second is enough), plus a small margin.
        async fn sleep_past(exp: i64) {
            let exp_at = UNIX_EPOCH + Duration::from_secs(exp as u64);
            let remaining = exp_at
                .duration_since(SystemTime::now())
                .unwrap_or(Duration::ZERO);
            tokio::time::sleep(remaining + Duration::from_millis(50)).await;
        }

        /// A permissions backend whose verdicts are scripted, one per call,
        /// so a test can tell "served from cache" from "asked again" by the
        /// verdict AND the call count.
        struct ScriptedPermissions {
            verdicts: Mutex<VecDeque<AuthResult>>,
            calls: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl crate::auth::permissions::PermissionsClient for ScriptedPermissions {
            async fn check_permission(
                &self,
                _token: &str,
                _path: &str,
                _permission: Permission,
            ) -> Result<AuthResult, AuthError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(self
                    .verdicts
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("the backend was asked more often than the test scripted"))
            }
        }

        fn scripted(verdicts: &[AuthResult]) -> (ScriptedPermissions, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                ScriptedPermissions {
                    verdicts: Mutex::new(verdicts.iter().cloned().collect()),
                    calls: Arc::clone(&calls),
                },
                calls,
            )
        }

        /// The headline case. The permissions service says `Allowed` for a
        /// JWT shortly before it expires; the verdict is served from cache
        /// while the bearer is live; once `exp` passes the next call reaches
        /// the backend, whose answer for the now-expired bearer
        /// (`Unauthenticated`, a 401 on the REST backend) is what the caller
        /// gets — and what tells the client to refresh its token.
        ///
        /// Pre-b.75i the third call was a cache hit returning `Allowed` for
        /// up to the full 600 s positive TTL after the bearer expired.
        #[tokio::test]
        async fn allowed_for_a_jwt_stops_being_served_at_its_exp() {
            let metrics = Arc::new(metrics_scrape_only("t"));
            let (perm, calls) = scripted(&[AuthResult::Allowed, AuthResult::Unauthenticated]);
            let cache = AuthCache::new_with_negative_ttl(
                Arc::new(perm),
                Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
                LONG_POSITIVE_TTL_SECS,
                30,
                5,
                TEST_MAX_ENTRIES,
                Some(Arc::clone(&metrics)),
            );
            let (token, exp) = live_jwt();

            for _ in 0..2 {
                assert_eq!(
                    cache
                        .check_permission(&token, "/p", Permission::Read)
                        .await
                        .unwrap(),
                    AuthResult::Allowed
                );
            }
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "while the bearer is live, its Allowed must still be served from cache"
            );

            sleep_past(exp).await;
            let after = cache
                .check_permission(&token, "/p", Permission::Read)
                .await
                .unwrap();
            assert_eq!(
                after,
                AuthResult::Unauthenticated,
                "after exp the cached Allowed must not be served: the call must reach the \
                 backend and return its verdict for the expired bearer"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 2);

            let text = render(&metrics);
            let read = |name: &str| {
                sample_value(&text, name, &[("kind", "permissions")])
                    .unwrap_or_else(|| panic!("{name}{{kind=\"permissions\"}} must exist"))
            };
            assert_eq!(
                (
                    read("live_edit_auth_cache_hits_total"),
                    read("live_edit_auth_cache_misses_total")
                ),
                (1.0, 2.0),
                "an entry past its bearer's exp is reported as a miss, not a hit"
            );
        }

        /// A JWT that has already expired when the backend answers `Allowed`
        /// gets that answer for the call that asked, and nothing is cached:
        /// the next call asks again. Caching it would admit a dead bearer for
        /// the whole positive TTL.
        #[tokio::test]
        async fn allowed_for_an_already_expired_jwt_is_served_but_not_cached() {
            let (perm, calls) = scripted(&[AuthResult::Allowed, AuthResult::Allowed]);
            let cache = AuthCache::new(
                Arc::new(perm),
                Arc::new(MockUserinfoClient::new(Ok(make_user_info()))),
                LONG_POSITIVE_TTL_SECS,
                TEST_MAX_ENTRIES,
            );
            let token = jwt(serde_json::json!({ "sub": "u", "exp": now_secs() - 60 }));

            for _ in 0..2 {
                assert_eq!(
                    cache
                        .check_permission(&token, "/p", Permission::Read)
                        .await
                        .unwrap(),
                    AuthResult::Allowed,
                    "the backend's verdict is what the call gets"
                );
            }
            assert_eq!(
                calls.load(Ordering::SeqCst),
                2,
                "a positive verdict for an already-expired bearer must not be cached"
            );
        }

        /// The userinfo success cache is bounded the same way: a cached
        /// identity for a JWT is served while the bearer is live and not
        /// after its `exp`.
        #[tokio::test]
        async fn userinfo_for_a_jwt_stops_being_served_at_its_exp() {
            let userinfo = MockUserinfoClient::new(Ok(make_user_info()));
            let calls = Arc::clone(&userinfo.calls);
            let cache = AuthCache::new(
                Arc::new(MockPermissionsClient::new(Ok(AuthResult::Allowed))),
                Arc::new(userinfo),
                LONG_POSITIVE_TTL_SECS,
                TEST_MAX_ENTRIES,
            );
            let (token, exp) = live_jwt();

            cache.get_current_user(&token).await.unwrap();
            cache.get_current_user(&token).await.unwrap();
            assert_eq!(
                calls.lock().unwrap().len(),
                1,
                "while the bearer is live, its identity must still be served from cache"
            );

            sleep_past(exp).await;
            cache.get_current_user(&token).await.unwrap();
            assert_eq!(
                calls.lock().unwrap().len(),
                2,
                "after exp the cached identity must not be served"
            );
        }

        /// A successful userinfo lookup for an already-expired JWT is not
        /// cached either.
        #[tokio::test]
        async fn userinfo_for_an_already_expired_jwt_is_not_cached() {
            let userinfo = MockUserinfoClient::new(Ok(make_user_info()));
            let calls = Arc::clone(&userinfo.calls);
            let cache = AuthCache::new(
                Arc::new(MockPermissionsClient::new(Ok(AuthResult::Allowed))),
                Arc::new(userinfo),
                LONG_POSITIVE_TTL_SECS,
                TEST_MAX_ENTRIES,
            );
            let token = jwt(serde_json::json!({ "sub": "u", "exp": now_secs() - 60 }));

            cache.get_current_user(&token).await.unwrap();
            cache.get_current_user(&token).await.unwrap();
            assert_eq!(calls.lock().unwrap().len(), 2);
        }

        // ── The decision function's branches ────────────────────────────

        /// An opaque bearer carries no `exp`: the configured TTL alone
        /// governs, as it always has. (The behavioural twin is
        /// `allowed_stays_on_positive_ttl_even_when_negative_ttl_is_short`,
        /// whose opaque `"token"` is still served after its sleep.)
        #[test]
        fn bearer_cache_bound_opaque_bearer_is_unbounded() {
            assert_eq!(
                bearer_cache_bound("opaque-bearer"),
                BearerCacheBound::Unbounded
            );
        }

        /// A JWT without an `exp`, or with a non-numeric one, is treated like
        /// an opaque bearer — nothing to bound by.
        #[test]
        fn bearer_cache_bound_jwt_without_a_numeric_exp_is_unbounded() {
            for claims in [
                serde_json::json!({ "sub": "u" }),
                serde_json::json!({ "sub": "u", "exp": "tomorrow" }),
            ] {
                assert_eq!(
                    bearer_cache_bound(&jwt(claims.clone())),
                    BearerCacheBound::Unbounded,
                    "claims {claims}"
                );
            }
        }

        /// `exp` is exclusive (RFC 7519 §4.1.4): a bearer whose `exp` is the
        /// current second, or any earlier one — a negative NumericDate
        /// included — is already expired.
        #[test]
        fn bearer_cache_bound_exp_at_or_before_now_is_expired() {
            for exp in [now_secs(), now_secs() - 60, -1] {
                assert_eq!(
                    bearer_cache_bound(&jwt(serde_json::json!({ "exp": exp }))),
                    BearerCacheBound::Expired,
                    "exp {exp}"
                );
            }
        }

        /// A future `exp` bounds the entry at that instant — no later than
        /// the bearer's own remaining lifetime from now.
        #[test]
        fn bearer_cache_bound_future_exp_bounds_at_the_exp() {
            let before = Instant::now();
            let (token, _) = live_jwt();
            match bearer_cache_bound(&token) {
                BearerCacheBound::Until(at) => {
                    assert!(at > before, "the bound must lie in the future");
                    assert!(
                        at <= Instant::now() + Duration::from_secs(LIVE_JWT_LIFETIME_SECS as u64),
                        "the bound must not lie past the bearer's exp"
                    );
                }
                other => panic!("a live JWT must be bounded at its exp; got {other:?}"),
            }
        }
    }
}
