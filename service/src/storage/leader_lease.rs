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

//! Per-process leader identity and the shared fail-closed leader lease.
//!
//! The checkpoint loop (`le:checkpoint:leader`) and the presence sweeper
//! (`le:presence:sweeper:leader`) write a [`LeaderIdentity::lock_value`] into
//! their lock through `external_values::try_acquire_leader_lock` and compare
//! it in `refresh_leader_lock_if_held`. The GC sweep, the orphan sweep, the
//! external-value reconciler and the external-value backup repair share
//! `le:gc:leader`, which carries the
//! protocol-marked [`LeaderIdentity::gc_lock_value`] instead and is taken only
//! through `external_values::try_acquire_gc_leader` (its own script, which
//! takes the lock over from an older build; SDD §11.3).
//!
//! [`LeaderLease`] is the fail-closed renewal logic for a task that must
//! re-prove leadership while it walks: it holds no metrics and emits no logs,
//! so each caller maps a [`LeaseCheck`] onto its own counter and log wording.
//! `checkpoint::checkpointer`'s `CheckpointLease` is a thin wrapper over it.

use std::fmt;
use std::time::{Duration, Instant};

use crate::storage::client::ValkeyClient;
use crate::storage::external_values::{refresh_leader_lock_if_held, try_acquire_leader_lock};

/// The identity a process writes into, and compares against, every leader
/// lock it takes.
///
/// `service_label` is the operator-chosen `LIVE_EDIT_SERVICE_LABEL` (falling
/// back to `HOSTNAME`). It is NOT unique: the Helm chart invites operators to
/// pin it to a stable deployment-wide value, so two replicas can share it.
/// `instance` is generated once per process ([`Self::for_process`]), and the
/// lock value is the pair, so two processes never mistake each other's lock
/// for their own — the acquire script's "already ours, extend it" branch
/// would otherwise let every same-labelled replica believe it leads.
///
/// **One identity per process, shared by every leader-gated task.** The GC
/// sweep and its lease refresh ticker, the orphan sweep, the reconciler and the
/// external-value backup repair all
/// take `le:gc:leader` through acquire-or-refresh on their own cadences; with
/// one instance each they would lock each other out instead of riding one
/// leadership.
///
/// Only the lock VALUE is this identity. Metric `service` labels and every
/// other replica-identifying field (e.g. a checkpoint's `creator_replica_id`,
/// or the `replica_id` log field) keep using `service_label` alone, so the
/// per-process UUID never becomes a label value. Every leader-lock log line
/// adds a separate `lock_value` field carrying the exact string this process
/// writes to that lock — [`Self::lock_value`] for the checkpoint and presence
/// locks, [`Self::gc_lock_value`] for `le:gc:leader` — so two same-labelled
/// replicas can be told apart. The field names the logging process, not the
/// lock's holder: a "skipped: another replica holds the lock" line carries the
/// skipper's value.
///
/// A restart gets a new `instance`: a restarted process never inherits its
/// predecessor's lock, and waits at most one lock TTL for it to lapse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaderIdentity {
    /// The operator-facing replica label; also the metric `service` label.
    pub service_label: String,
    /// Per-process discriminator (a UUIDv4 in production).
    pub instance: String,
}

impl LeaderIdentity {
    /// An identity with an explicit `instance`. Production uses
    /// [`Self::for_process`]; tests build identities explicitly so a lock
    /// value is predictable.
    pub fn new(service_label: impl Into<String>, instance: impl Into<String>) -> Self {
        Self {
            service_label: service_label.into(),
            instance: instance.into(),
        }
    }

    /// This process's identity: `service_label` plus a freshly generated
    /// UUIDv4 `instance`. Call once at startup and clone it into every
    /// leader-gated task.
    pub fn for_process(service_label: impl Into<String>) -> Self {
        Self::new(service_label, uuid::Uuid::new_v4().to_string())
    }

    /// The value written into the checkpoint and presence leader locks:
    /// `"{service_label}:{instance}"`.
    pub fn lock_value(&self) -> String {
        self.to_string()
    }

    /// The value written into `le:gc:leader`:
    /// `"le-gc/{GC_LEADER_PROTOCOL_VERSION}|{service_label}:{instance}"`.
    ///
    /// The `le-gc/<v>|` prefix is the GC protocol marker. Builds before it
    /// (every release up to and including 0.2.0-rc.48) write the bare
    /// [`Self::lock_value`] (0.1.0 the bare label), so a value without the
    /// prefix is version 1 by absence; `external_values::try_acquire_gc_leader`
    /// takes such a lock over, and an older build's equality CAS never matches
    /// a marked value, so it leads again only if the key lapses (see
    /// `scripts/gc_leader_acquire.lua` for when it can). A `|`
    /// cannot occur in a `HOSTNAME`-derived label, and the prefix is matched
    /// from the start of the value only.
    pub fn gc_lock_value(&self) -> String {
        format!("{GC_LEADER_MARKER_PREFIX}{GC_LEADER_PROTOCOL_VERSION}|{self}")
    }
}

/// The GC leader-lock protocol version this build writes into `le:gc:leader`
/// ([`LeaderIdentity::gc_lock_value`]). Version 2 is the first marked one:
/// the build whose GC claims, release enqueues and retirement carriers an
/// older build cannot act on (SDD §11.3). `gc_leader_acquire.lua` takes over a
/// lock with no marker or a lower version at once, and a same-version lock
/// only once its TTL falls to `external_values::GC_LEADER_TAKEOVER_BELOW_MS`;
/// it leaves a higher version alone whatever its TTL, taking it over only when
/// it has no TTL at all (a lock that would never lapse). A higher-version
/// holder that stops refreshing lapses within the 30 s TTL, so nothing
/// deadlocks.
///
/// Bump it for a GC protocol change an older marked build should not lead
/// across, but the lock alone does not keep that build out: lower-version
/// peers answer `held` to the newer build rather than keeping the key alive,
/// so when every process of the newer build loses the store for the TTL (or
/// stops), the key lapses and the older marked build can acquire it; and an
/// older pass already running at a takeover runs on. A bump therefore needs
/// the same store-state safety analysis this version has against unmarked
/// builds (SDD §11.3; `storage::external_values`, "Older builds sharing the
/// store").
pub const GC_LEADER_PROTOCOL_VERSION: u32 = 2;

/// The literal that opens a marked `le:gc:leader` value, before the version
/// digits and the `|` separator.
pub const GC_LEADER_MARKER_PREFIX: &str = "le-gc/";

/// Prints [`LeaderIdentity::lock_value`].
impl fmt::Display for LeaderIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.service_label, self.instance)
    }
}

/// Verdict of a [`LeaderLease`] check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseState {
    /// The lease is provably still held by this process: either the local
    /// deadline is comfortably in the future and no refresh was due, or a
    /// refresh (or re-acquire) just landed.
    Held,
    /// The lease could NOT be proven held — it lapsed locally, the store
    /// reported the key gone or owned by another process, or the round-trip
    /// errored. The caller must stop mutating shared durable state.
    Lost,
}

/// What a [`LeaderLease`] check decided, for the caller's metric. `None` on
/// [`LeaseCheck::outcome`] means no decision was needed (held, refresh not
/// yet due, no round-trip), so a counter fed from this measures refresh
/// decisions rather than checks.
///
/// [`Self::RefreshError`] is reported **instead of** [`Self::Lost`], not in
/// addition to it. A caller whose counter treats `refresh_error` as additive
/// to `lost` (so the abort causes sum to the number of aborted walks) counts
/// both cells on `RefreshError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseOutcome {
    /// A due refresh (or a re-acquire) found the lock ours and extended it.
    Renewed,
    /// The lease is lost: the local deadline passed, or the store reported
    /// the key lapsed or held by another process.
    Lost,
    /// The round-trip failed. Unknown is not held, so the state is `Lost`.
    RefreshError,
}

/// Why a check reported [`LeaseState::Lost`] — for the caller's log line,
/// which needs more than the metric outcome: whether a round-trip happened,
/// and the store error when one failed.
#[derive(Debug)]
pub enum LostCause {
    /// The local deadline had already passed. No round-trip was made: an
    /// overshot lease cannot be re-proven, only re-acquired.
    DeadlineOverrun,
    /// The store answered that the key lapsed or another process holds it.
    NotHeld,
    /// The round-trip failed, so whether the lock is held is unknown.
    StoreError(fred::error::Error),
}

/// Result of [`LeaderLease::ensure_held`] / [`LeaderLease::reacquire`].
#[derive(Debug)]
pub struct LeaseCheck {
    /// Whether the caller may continue mutating.
    pub state: LeaseState,
    /// The decision to count, if one was made.
    pub outcome: Option<LeaseOutcome>,
    /// `Some` exactly when `state` is [`LeaseState::Lost`].
    pub cause: Option<LostCause>,
}

impl LeaseCheck {
    fn held(outcome: Option<LeaseOutcome>) -> Self {
        Self {
            state: LeaseState::Held,
            outcome,
            cause: None,
        }
    }

    fn lost(outcome: LeaseOutcome, cause: LostCause) -> Self {
        Self {
            state: LeaseState::Lost,
            outcome: Some(outcome),
            cause: Some(cause),
        }
    }
}

/// A leader lease held across one long walk, renewed as the walk proceeds.
///
/// Built after a successful `try_acquire_leader_lock`. The caller consults
/// [`Self::ensure_held`] before every step that mutates shared durable state;
/// a due check refreshes the lock, at most once per [`Self::renew_interval`].
///
/// # Renewal interval
///
/// `ttl / 3`, the classic lease-renewal ratio: it gives the refresh two full
/// chances to land before the lease it is extending would expire, so a single
/// dropped packet or one slow maintenance-pool round-trip does not cost the
/// walk its lease. Tightening it toward the TTL trades away that headroom;
/// loosening it toward zero spends round-trips on steps that took
/// milliseconds.
///
/// # Fail-closed rules
///
/// Every ambiguity resolves to [`LeaseState::Lost`]:
///
/// - An expired local deadline returns `Lost` **without a round-trip**: a
///   process that has already overshot its lease must not be allowed to
///   mutate on the strength of a refresh that might be answering for a lock
///   another process has since taken and we have since re-taken.
/// - The refresh is `refresh_leader_lock_if_held`, never the acquire variant,
///   so a lapsed key can never be silently re-claimed and reported as
///   continuity, and a foreign key is never touched.
/// - A refresh that errors returns `Lost` — an unknown lease is not a held
///   lease.
/// - The local deadline is always a **lower bound** on the true expiry: it is
///   derived from an `Instant` sampled *before* the acquire/refresh
///   round-trip, so the server-side `PEXPIRE` necessarily landed at or after
///   it.
///
/// The type holds no metrics and logs nothing; see the module doc.
#[derive(Debug)]
pub struct LeaderLease {
    /// Fully-qualified lock key (a `KeySchema::*_leader_lock` accessor
    /// output).
    lock_key: String,
    /// This process's identity — its lock value must still be the lock's
    /// value for a refresh to succeed.
    identity: LeaderIdentity,
    /// The lock TTL, applied by every refresh and re-acquire.
    ttl: Duration,
    /// Local deadline past which this process can no longer claim the lease.
    /// Always a lower bound on the server-side expiry (see the type doc).
    held_until: Instant,
    /// When the lease was last acquired or refreshed — the base for
    /// [`Self::renew_interval`].
    last_refresh: Instant,
}

impl LeaderLease {
    /// Build the lease after `try_acquire_leader_lock(.., identity, ttl)`
    /// returned `Ok(true)`.
    ///
    /// `acquired_at` must be an `Instant` sampled **before** the acquire
    /// round-trip was issued, so the local deadline under-estimates rather
    /// than over-estimates the server-side expiry. `ttl` must be the TTL the
    /// acquire used; every refresh re-applies it.
    pub fn new(
        lock_key: String,
        identity: LeaderIdentity,
        acquired_at: Instant,
        ttl: Duration,
    ) -> Self {
        Self {
            lock_key,
            identity,
            ttl,
            held_until: acquired_at + ttl,
            last_refresh: acquired_at,
        }
    }

    /// A lease with explicit deadlines, so a test can place
    /// [`Self::ensure_held`] on any arm (not due, overshot, refresh due)
    /// without waiting. `ttl` stays the lease's TTL — it sets the renewal
    /// interval and the refresh `PEXPIRE` — independently of `held_until`.
    #[cfg(test)]
    pub(crate) fn with_deadlines(
        lock_key: String,
        identity: LeaderIdentity,
        ttl: Duration,
        held_until: Instant,
        last_refresh: Instant,
    ) -> Self {
        Self {
            lock_key,
            identity,
            ttl,
            held_until,
            last_refresh,
        }
    }

    /// The lock key this lease guards.
    pub fn lock_key(&self) -> &str {
        &self.lock_key
    }

    /// The identity this lease holds the lock as.
    pub fn identity(&self) -> &LeaderIdentity {
        &self.identity
    }

    /// The renewal interval: one third of the TTL (see the type doc).
    pub fn renew_interval(&self) -> Duration {
        self.ttl / 3
    }

    /// When the next refresh falls due: [`Self::renew_interval`] after the
    /// last acquire or refresh. [`Self::ensure_held`] called at or after it
    /// issues the refresh; called before it, it answers from the local
    /// deadline alone.
    pub(crate) fn next_refresh_due(&self) -> Instant {
        self.last_refresh + self.renew_interval()
    }

    /// The local deadline (a lower bound on the server-side expiry).
    #[cfg(test)]
    pub(crate) fn held_until(&self) -> Instant {
        self.held_until
    }

    /// When the lease was last acquired or refreshed.
    #[cfg(test)]
    pub(crate) fn last_refresh(&self) -> Instant {
        self.last_refresh
    }

    /// The TTL in milliseconds, as the lock scripts take it.
    fn ttl_ms(&self) -> u64 {
        u64::try_from(self.ttl.as_millis()).unwrap_or(u64::MAX)
    }

    /// Prove the lease is still held, refreshing it if it is time to.
    ///
    /// In evaluation order:
    ///
    /// 1. **Local deadline passed** → `Lost` / [`LeaseOutcome::Lost`] /
    ///    [`LostCause::DeadlineOverrun`], no round-trip. An overshot process
    ///    must not mutate on an unprovable lease, and a refresh here could only
    ///    tell us about the lock's state *now*, not about the gap.
    /// 2. **Refresh due** (`>= renew_interval` since the last one) → issue
    ///    `refresh_leader_lock_if_held`. `Ok(true)` extends the local deadline
    ///    and returns `Held` / [`LeaseOutcome::Renewed`]; `Ok(false)` (lapsed
    ///    or held by another process) returns `Lost` / [`LeaseOutcome::Lost`];
    ///    `Err(_)` returns `Lost` / [`LeaseOutcome::RefreshError`].
    /// 3. **Otherwise** → `Held` with no outcome and no round-trip.
    pub async fn ensure_held(&mut self, client: &ValkeyClient) -> LeaseCheck {
        // Sampled once and reused for both the deadline test and the
        // post-refresh bookkeeping, so the deadline this call installs can
        // never be later than the instant the refresh was actually issued.
        let now = Instant::now();

        if now >= self.held_until {
            return LeaseCheck::lost(LeaseOutcome::Lost, LostCause::DeadlineOverrun);
        }

        if now < self.next_refresh_due() {
            return LeaseCheck::held(None);
        }

        let ttl_ms = self.ttl_ms();
        match refresh_leader_lock_if_held(client, &self.lock_key, &self.identity, ttl_ms).await {
            Ok(true) => {
                self.held_until = now + self.ttl;
                self.last_refresh = now;
                LeaseCheck::held(Some(LeaseOutcome::Renewed))
            }
            Ok(false) => LeaseCheck::lost(LeaseOutcome::Lost, LostCause::NotHeld),
            Err(e) => LeaseCheck::lost(LeaseOutcome::RefreshError, LostCause::StoreError(e)),
        }
    }

    /// Re-prove the lease from scratch with the acquire-or-refresh script,
    /// replacing the deadlines with fresh ones measured from just before the
    /// round-trip.
    ///
    /// Unlike [`Self::ensure_held`] this CAN claim a lapsed (unset) key, so it
    /// is only sound after a step that mutated nothing: it does not claim
    /// continuity across the gap, it starts a new lease, and everything after
    /// it is gated by that new lease. It still never claims a key another
    /// process holds.
    ///
    /// `Ok(true)` → `Held` / [`LeaseOutcome::Renewed`]; `Ok(false)` → `Lost` /
    /// [`LeaseOutcome::Lost`] / [`LostCause::NotHeld`]; `Err(_)` → `Lost` /
    /// [`LeaseOutcome::RefreshError`]. On `Lost` the old deadlines are left
    /// as they were.
    pub async fn reacquire(&mut self, client: &ValkeyClient) -> LeaseCheck {
        // Sampled BEFORE the round-trip, so the new deadline under-estimates
        // the server-side expiry — the same discipline as `new`'s caller
        // contract.
        let acquire_started = Instant::now();
        let ttl_ms = self.ttl_ms();
        match try_acquire_leader_lock(client, &self.lock_key, &self.identity, ttl_ms).await {
            Ok(true) => {
                self.held_until = acquire_started + self.ttl;
                self.last_refresh = acquire_started;
                LeaseCheck::held(Some(LeaseOutcome::Renewed))
            }
            Ok(false) => LeaseCheck::lost(LeaseOutcome::Lost, LostCause::NotHeld),
            Err(e) => LeaseCheck::lost(LeaseOutcome::RefreshError, LostCause::StoreError(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    //! [`LeaderIdentity`]'s lock-value shapes, and [`LeaderLease`]'s arms against
    //! a live `TestValkey`.
    //!
    //! The lease arms run on `le:presence:sweeper:leader`, the lock
    //! [`LeaderLease`] renews in production. `le:gc:leader` is no longer taken
    //! through the plain CAS the lease drives: it carries the protocol-marked
    //! [`LeaderIdentity::gc_lock_value`] and its own takeover script, whose
    //! verdicts are pinned in `tests/integration/storage/b_1jt_rollout_tests.rs`.
    //!
    //! The lease tests place a check on a chosen arm with
    //! [`LeaderLease::with_deadlines`] rather than by waiting:
    //! `tokio::time::pause` cannot move the `std::time::Instant` the lease
    //! reads, and the Valkey-backed tests run multi-thread.
    //!
    //! **"No round-trip" is proved with a tripwire, not a TTL reading.** The
    //! arms that must not reach the store run against a lock key holding a
    //! LIST. Either lock script's `GET` on it raises `WRONGTYPE`, so any
    //! round-trip would surface as [`LeaseOutcome::RefreshError`] /
    //! [`LostCause::StoreError`]; an answer of anything else proves the check
    //! returned before touching the store.

    use fred::interfaces::{KeysInterface, ListInterface};

    use super::*;
    use crate::storage::test_support::TestValkey;

    /// The lease TTL under test. Deliberately NOT `LEADER_LOCK_TTL_MS`
    /// (30 s), so a refreshed `PTTL` at or below it proves the refresh applied
    /// the lease's own TTL rather than the production constant.
    const LEASE_TTL: Duration = Duration::from_secs(20);

    /// The TTL a planted lock starts with — far below [`LEASE_TTL`], so "was
    /// the key `PEXPIRE`d?" is a yes/no reading of its `PTTL`, and long
    /// enough that it cannot lapse between planting and the reads that follow.
    const PLANTED_TTL_MS: i64 = 8_000;

    fn ours() -> LeaderIdentity {
        LeaderIdentity::new("lease-test-replica", "instance-ours")
    }

    /// An `Instant` `d` in the past; `checked_sub` so a too-young monotonic
    /// clock is a named failure, not an arithmetic panic.
    fn instant_ago(d: Duration) -> Instant {
        Instant::now()
            .checked_sub(d)
            .expect("the monotonic clock must be older than the test's look-back")
    }

    /// A lease as [`ours`] with its local deadline `held_for` in the future
    /// and its last refresh `refreshed_ago` in the past.
    fn lease(lock_key: &str, held_for: Duration, refreshed_ago: Duration) -> LeaderLease {
        LeaderLease::with_deadlines(
            lock_key.to_owned(),
            ours(),
            LEASE_TTL,
            Instant::now() + held_for,
            instant_ago(refreshed_ago),
        )
    }

    /// A lease whose local deadline is comfortably ahead but whose refresh is
    /// due, so the next `ensure_held` round-trips.
    fn refresh_due_lease(lock_key: &str) -> LeaderLease {
        lease(lock_key, LEASE_TTL, LEASE_TTL / 3)
    }

    async fn plant_lock(tv: &TestValkey, lock_key: &str, value: &str) {
        let _: () = tv
            .client()
            .pool
            .set(
                lock_key,
                value,
                Some(fred::types::Expiration::PX(PLANTED_TTL_MS)),
                None,
                false,
            )
            .await
            .expect("SET of the planted lock must succeed");
    }

    /// Make `lock_key` a LIST: any lock-script round-trip against it errors
    /// (see the module doc).
    async fn plant_tripwire(tv: &TestValkey, lock_key: &str) {
        let _: i64 = tv
            .client()
            .pool
            .lpush(lock_key, "not-a-string-lock")
            .await
            .expect("LPUSH of the tripwire must succeed");
    }

    async fn get_lock(tv: &TestValkey, lock_key: &str) -> Option<String> {
        tv.client()
            .pool
            .get(lock_key)
            .await
            .expect("GET must succeed")
    }

    async fn pttl(tv: &TestValkey, lock_key: &str) -> i64 {
        tv.client()
            .pool
            .pttl(lock_key)
            .await
            .expect("PTTL must succeed")
    }

    // ── LeaderIdentity ───────────────────────────────────────────────────

    #[test]
    fn lock_value_is_label_colon_instance_and_display_prints_it() {
        let identity = LeaderIdentity::new("replica-a", "1234");
        assert_eq!(identity.lock_value(), "replica-a:1234");
        assert_eq!(
            identity.to_string(),
            identity.lock_value(),
            "log lines print the identity through Display; it must be the lock value"
        );
    }

    /// The GC lock value is the plain lock value behind the protocol marker,
    /// `le-gc/<v>|`, with this build's version 2 — the first marked one. A
    /// value without the prefix is what every older build writes, and is what
    /// `gc_leader_acquire.lua` takes over, so a changed prefix or separator
    /// would make this build read its own peers as older builds.
    #[test]
    fn gc_lock_value_is_the_lock_value_behind_the_protocol_marker() {
        let identity = LeaderIdentity::new("replica-a", "1234");
        assert_eq!(GC_LEADER_MARKER_PREFIX, "le-gc/");
        assert_eq!(GC_LEADER_PROTOCOL_VERSION, 2);
        assert_eq!(identity.gc_lock_value(), "le-gc/2|replica-a:1234");
        assert!(
            !identity.lock_value().starts_with(GC_LEADER_MARKER_PREFIX),
            "the plain lock value (the checkpoint and presence locks, and every older build's \
             GC lock) must not carry the marker"
        );
    }

    #[test]
    fn for_process_keeps_the_label_and_mints_a_distinct_uuid_v4_instance() {
        let a = LeaderIdentity::for_process("shared-label");
        let b = LeaderIdentity::for_process("shared-label");

        assert_eq!(a.service_label, "shared-label");
        assert_eq!(b.service_label, "shared-label");
        for identity in [&a, &b] {
            let instance = uuid::Uuid::parse_str(&identity.instance)
                .unwrap_or_else(|e| panic!("instance {:?} must be a UUID: {e}", identity.instance));
            assert_eq!(instance.get_version_num(), 4, "instance must be a UUIDv4");
        }
        assert_ne!(
            a.lock_value(),
            b.lock_value(),
            "two processes sharing a service_label must write different lock values"
        );
    }

    // ── LeaderLease construction ─────────────────────────────────────────

    #[test]
    fn new_takes_both_deadlines_from_acquired_at_and_renews_every_third_of_the_ttl() {
        let acquired_at = instant_ago(Duration::from_secs(1));
        let lease = LeaderLease::new("lock".to_owned(), ours(), acquired_at, LEASE_TTL);

        assert_eq!(lease.held_until(), acquired_at + LEASE_TTL);
        assert_eq!(lease.last_refresh(), acquired_at);
        assert_eq!(lease.renew_interval(), LEASE_TTL / 3);
        assert_eq!(lease.identity(), &ours());
        assert_eq!(lease.lock_key(), "lock");
    }

    // ── ensure_held ──────────────────────────────────────────────────────

    /// Not due: `Held` with no outcome and no round-trip.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ensure_held_not_yet_due_is_held_without_a_round_trip() {
        let tv = TestValkey::setup().await;
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();
        plant_tripwire(&tv, &lock_key).await;

        let mut lease = lease(&lock_key, LEASE_TTL, Duration::ZERO);
        let check = lease.ensure_held(tv.client()).await;

        assert_eq!(check.state, LeaseState::Held, "{check:?}");
        assert_eq!(
            check.outcome, None,
            "a not-yet-due check makes no decision; a RefreshError here means it round-tripped"
        );
        assert!(check.cause.is_none(), "{check:?}");
    }

    /// Overshoot: `Lost` / `Lost` / `DeadlineOverrun`, with no round-trip
    /// even though a refresh is long overdue.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ensure_held_overshoot_is_lost_without_a_round_trip() {
        let tv = TestValkey::setup().await;
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();
        plant_tripwire(&tv, &lock_key).await;

        // `held_until = now` has passed by the time `ensure_held` samples its
        // own `now`; the last refresh is a whole TTL ago, so the refresh arm
        // would be due if the deadline check did not run first.
        let mut lease = lease(&lock_key, Duration::ZERO, LEASE_TTL);
        let check = lease.ensure_held(tv.client()).await;

        assert_eq!(check.state, LeaseState::Lost, "{check:?}");
        assert_eq!(
            check.outcome,
            Some(LeaseOutcome::Lost),
            "an overshot lease is Lost, not RefreshError — RefreshError would mean a round-trip"
        );
        assert!(
            matches!(check.cause, Some(LostCause::DeadlineOverrun)),
            "{check:?}"
        );
    }

    /// Refresh on our own lock: `Held` / `Renewed`, the key's TTL becomes
    /// the lease's TTL, its value is untouched, and the local deadlines move
    /// to an instant sampled before the round-trip.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ensure_held_refresh_on_our_lock_renews_with_the_lease_ttl() {
        let tv = TestValkey::setup().await;
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();
        plant_lock(&tv, &lock_key, &ours().lock_value()).await;

        let mut lease = refresh_due_lease(&lock_key);
        let before = Instant::now();
        let check = lease.ensure_held(tv.client()).await;
        let after = Instant::now();

        assert_eq!(check.state, LeaseState::Held, "{check:?}");
        assert_eq!(check.outcome, Some(LeaseOutcome::Renewed));
        assert!(check.cause.is_none(), "{check:?}");

        let ttl = pttl(&tv, &lock_key).await;
        assert!(
            ttl > PLANTED_TTL_MS && ttl <= LEASE_TTL.as_millis() as i64,
            "the refresh must PEXPIRE the lock to the lease's {LEASE_TTL:?}; got {ttl} ms"
        );
        assert_eq!(
            get_lock(&tv, &lock_key).await,
            Some(ours().lock_value()),
            "a refresh extends the lock; it never rewrites its value"
        );
        assert!(
            lease.last_refresh() >= before && lease.last_refresh() <= after,
            "the refresh instant must be this call's"
        );
        assert_eq!(
            lease.held_until(),
            lease.last_refresh() + LEASE_TTL,
            "the new deadline is measured from the instant sampled before the round-trip"
        );
    }

    /// Refresh on a lock another process holds: `Lost` / `Lost` /
    /// `NotHeld`, the holder's lock untouched and our deadlines unchanged.
    /// The same-label peer is SR-2.5's case: sharing `service_label` must not
    /// make a lock ours.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ensure_held_refresh_on_another_process_lock_is_lost_and_leaves_it_alone() {
        let tv = TestValkey::setup().await;
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();
        let peers = [
            (
                "a differently-labelled peer",
                LeaderIdentity::new("peer-replica", "peer"),
            ),
            (
                "a same-label peer",
                LeaderIdentity::new(ours().service_label, "instance-peer"),
            ),
        ];

        for (case, peer) in peers {
            plant_lock(&tv, &lock_key, &peer.lock_value()).await;
            let mut lease = refresh_due_lease(&lock_key);
            let (held_until, last_refresh) = (lease.held_until(), lease.last_refresh());

            let check = lease.ensure_held(tv.client()).await;

            assert_eq!(check.state, LeaseState::Lost, "{case}: {check:?}");
            assert_eq!(check.outcome, Some(LeaseOutcome::Lost), "{case}");
            assert!(
                matches!(check.cause, Some(LostCause::NotHeld)),
                "{case}: {check:?}"
            );
            assert_eq!(
                get_lock(&tv, &lock_key).await,
                Some(peer.lock_value()),
                "{case}: the holder's lock value must be untouched"
            );
            assert!(
                pttl(&tv, &lock_key).await <= PLANTED_TTL_MS,
                "{case}: the holder's lock TTL must not be extended"
            );
            assert_eq!(lease.held_until(), held_until, "{case}");
            assert_eq!(lease.last_refresh(), last_refresh, "{case}");
        }
    }

    /// Refresh on a lapsed (absent) lock: `Lost` / `Lost` / `NotHeld`, and
    /// the key is NOT re-created — the refresh never claims.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ensure_held_refresh_on_a_lapsed_lock_is_lost_and_does_not_reclaim() {
        let tv = TestValkey::setup().await;
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();

        let mut lease = refresh_due_lease(&lock_key);
        let check = lease.ensure_held(tv.client()).await;

        assert_eq!(check.state, LeaseState::Lost, "{check:?}");
        assert_eq!(check.outcome, Some(LeaseOutcome::Lost));
        assert!(matches!(check.cause, Some(LostCause::NotHeld)), "{check:?}");
        assert_eq!(
            get_lock(&tv, &lock_key).await,
            None,
            "a refresh against an absent key must not create it"
        );
    }

    /// Refresh that errors: `Lost` / `RefreshError` / `StoreError`. The
    /// helper reports `RefreshError` INSTEAD of `Lost`; counting it as
    /// additive to `lost` is the caller's job.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ensure_held_refresh_error_is_lost_with_the_store_error() {
        let tv = TestValkey::setup().await;
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();
        plant_tripwire(&tv, &lock_key).await;

        let mut lease = refresh_due_lease(&lock_key);
        let check = lease.ensure_held(tv.client()).await;

        assert_eq!(
            check.state,
            LeaseState::Lost,
            "an unknown lease is not a held lease: {check:?}"
        );
        assert_eq!(check.outcome, Some(LeaseOutcome::RefreshError));
        assert!(
            matches!(check.cause, Some(LostCause::StoreError(_))),
            "{check:?}"
        );
    }

    // ── reacquire ────────────────────────────────────────────────────────

    /// Re-acquire on a lapsed key claims it, even from a lease whose local
    /// deadline has passed, and restarts both deadlines from an instant
    /// sampled before the round-trip.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reacquire_on_a_lapsed_lock_claims_it_and_restarts_the_deadlines() {
        let tv = TestValkey::setup().await;
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();

        let mut lease = lease(&lock_key, Duration::ZERO, LEASE_TTL);
        let before = Instant::now();
        let check = lease.reacquire(tv.client()).await;
        let after = Instant::now();

        assert_eq!(check.state, LeaseState::Held, "{check:?}");
        assert_eq!(check.outcome, Some(LeaseOutcome::Renewed));
        assert!(check.cause.is_none(), "{check:?}");
        assert_eq!(
            get_lock(&tv, &lock_key).await,
            Some(ours().lock_value()),
            "the claim writes this process's lock value"
        );
        let ttl = pttl(&tv, &lock_key).await;
        assert!(
            ttl > 0 && ttl <= LEASE_TTL.as_millis() as i64,
            "the claim's TTL is the lease's {LEASE_TTL:?}; got {ttl} ms"
        );
        assert!(lease.last_refresh() >= before && lease.last_refresh() <= after);
        assert_eq!(lease.held_until(), lease.last_refresh() + LEASE_TTL);
    }

    /// Re-acquire on our own lock extends it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reacquire_on_our_lock_renews_it() {
        let tv = TestValkey::setup().await;
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();
        plant_lock(&tv, &lock_key, &ours().lock_value()).await;

        let mut lease = lease(&lock_key, LEASE_TTL, Duration::ZERO);
        let check = lease.reacquire(tv.client()).await;

        assert_eq!(check.state, LeaseState::Held, "{check:?}");
        assert_eq!(check.outcome, Some(LeaseOutcome::Renewed));
        let ttl = pttl(&tv, &lock_key).await;
        assert!(
            ttl > PLANTED_TTL_MS && ttl <= LEASE_TTL.as_millis() as i64,
            "re-acquiring our own lock must extend it to the lease's TTL; got {ttl} ms"
        );
    }

    /// Re-acquire never claims a lock a same-label peer holds: `Lost` /
    /// `Lost` / `NotHeld`, the peer untouched, our deadlines unchanged.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reacquire_on_a_same_label_peer_lock_is_lost_and_leaves_it_alone() {
        let tv = TestValkey::setup().await;
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();
        let peer = LeaderIdentity::new(ours().service_label, "instance-peer");
        plant_lock(&tv, &lock_key, &peer.lock_value()).await;

        let mut lease = lease(&lock_key, LEASE_TTL, Duration::ZERO);
        let (held_until, last_refresh) = (lease.held_until(), lease.last_refresh());
        let check = lease.reacquire(tv.client()).await;

        assert_eq!(check.state, LeaseState::Lost, "{check:?}");
        assert_eq!(check.outcome, Some(LeaseOutcome::Lost));
        assert!(matches!(check.cause, Some(LostCause::NotHeld)), "{check:?}");
        assert_eq!(get_lock(&tv, &lock_key).await, Some(peer.lock_value()));
        assert!(pttl(&tv, &lock_key).await <= PLANTED_TTL_MS);
        assert_eq!(
            lease.held_until(),
            held_until,
            "Lost leaves the old deadlines"
        );
        assert_eq!(lease.last_refresh(), last_refresh);
    }

    /// Re-acquire that errors: `Lost` / `RefreshError` / `StoreError`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reacquire_error_is_lost_with_the_store_error() {
        let tv = TestValkey::setup().await;
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();
        plant_tripwire(&tv, &lock_key).await;

        let mut lease = lease(&lock_key, LEASE_TTL, Duration::ZERO);
        let check = lease.reacquire(tv.client()).await;

        assert_eq!(check.state, LeaseState::Lost, "{check:?}");
        assert_eq!(check.outcome, Some(LeaseOutcome::RefreshError));
        assert!(
            matches!(check.cause, Some(LostCause::StoreError(_))),
            "{check:?}"
        );
    }
}
