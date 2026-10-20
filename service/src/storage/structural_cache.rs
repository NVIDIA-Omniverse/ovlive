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

//! The per-replica decoded-state cache that structural `UpdateLayer` planning
//! runs on — contract C3 of the structural-edits plan.
//!
//! Section references: `SR-n.m` is the structural-edits spec SDD (Spec Bee
//! `b.xes`); `§n.m` is `docs-internal/sdd.md`.
//!
//! # Wiring
//!
//! `LayerStore` holds one [`StructuralCache`] per replica, sized by the
//! budget derived from the pod memory limit (`config::memory_plan`), and
//! object retirement (its
//! `SemaphoreMapGuard` at `DeleteLayer`, at a `CreateLayer` overwrite and at
//! a re-sent overwrite's re-drive)
//! invalidates the retired object here. The cold start
//! (`storage::structural_cache_cold_start`, SR-4.3) fills it through
//! [`StructuralCache::admit`] and records its uncacheable verdicts, each of
//! its whole-layer reads under this cache's per-replica cold-start permits
//! (["Cold-start permits"](#cold-start-permits) below); the stream repair
//! (`storage::structural_cache_repair`, SR-4.1) advances checked-out entries;
//! and the read side (`storage::structural_cache_read`) is the one entry point
//! that drives lookup, repair and cold start for a structural call. The
//! sampled audit (`storage::structural_cache_audit`, SR-3.6) compares a copy
//! of an entry with the store and invalidates a diverged one. Under the `lua`
//! backend every structural and reset `UpdateLayer` attempt goes through the
//! read side, plans on the entry it answers and settles the guard by the
//! commit's outcome (`LayerStore::update_layer_once_structural`), which also
//! triggers the audit after a committed call. An attempt whose plan is over
//! the size gate (`storage::structural_size_gate`) commits nothing on the
//! structural route: it puts the entry back held, unchanged, and takes the
//! WATCH path. The cross-track
//! contract below — the entry and lookup types, `lookup` / `checkout` /
//! `invalidate` and the [`CheckedOut`] guard — changes only through the plan's
//! decider; the other items here (admission, negative entries, the weight
//! estimate, the gauges' sources, the cold-start permits) are the cache
//! track's own.
//!
//! # Cold-start permits
//!
//! Every whole-layer cold-start read on a replica — a cold start's or a
//! reload's, plain or fenced, and the sampled audit's fenced read — holds one
//! of the cache's [`STRUCTURAL_COLD_START_READ_PERMITS`] **plain permits**
//! across that one read, so a restart followed by first structural edits on
//! many layers queues their reads instead of running them all against the
//! shards at once. A fenced cold start and the audit's fenced read take the
//! cache's one-at-a-time **cold-start permit** first, then a plain permit; a
//! plain read takes a plain permit only. So at most two cold-start reads run
//! on a replica at once and at most one of them is fenced or raised, and a
//! plain read waits behind at most the one read holding the cold-start
//! permit, never behind the reads queued for it. The audit's plain read takes
//! no permit (one audit runs per replica at a time), and neither does the read
//! side's probe. [`StructuralCache::acquire_plain_cold_start_read_permit`],
//! [`StructuralCache::acquire_fenced_cold_start_read_permits`] and
//! [`StructuralCache::acquire_raised_cold_start_read_permits`] take them in
//! that order, released by RAII on every exit.
//!
//! A fenced cold start takes two more bounds after the pair, both shared with
//! the catch-up flights' fenced reads: a fenced-class `CatchUpReadGate`
//! permit (`P_f`) and a `MemoryBudget` grant inside `T`
//! (`storage::structural_cache_fenced`). It gives all of them back during
//! every `RETRY_AFTER` sleep. A catch-up never waits on this cache's permits,
//! so the shared bounds form no wait cycle, and since the one-at-a-time permit
//! is held before `P_f` is asked for, at most one cold start per replica ever
//! waits on `P_f`.
//!
//! # What the cache is (SR-3.1)
//!
//! Each [`CacheEntry`] holds one object's complete, normalized decoded state
//! (SR-3.2) at a `(seq, commit_chain)` cut (SR-4.5), in the compact form
//! ([`CompactState`]): per value row only its key and its external hash, the
//! field names interned once per entry, each time sample keyed by its stored
//! bits. That is everything structural planning and the structural commit's
//! pre-images read of a row, so a plan on the compact entry emits the same
//! mutations a plan on the full state would, up to the order within one
//! drain or purge run (`domain::delta::apply_delta`'s docs), which is not
//! fixed across states on either form (a clone emits the same run in the
//! same order, so the shadow check compares a plan exactly). That includes a
//! section's time 0, which a compact entry can hold at `0.0`, at `-0.0` or at
//! both: a reset or cascade deletes it at both stored spellings on either
//! form (rule E). A compact cascade purge's field half probes the purged
//! sections' rows by interned name, O(min(D × N, F)) for `D` purged
//! sections and `N` interned-name id slots (live and free), rather than a
//! scan of every field row (`domain::types::PURGE_PROBE_RATIO`); its
//! time-sample half still scans every time-sample row the object holds. The
//! cache holds normalized states only: whoever builds an entry — a
//! cold-start read, a repair, a commit swap — normalizes it first; the cache
//! stores what it is handed, its hash tables compacted ([`compact_state`])
//! and its weight recomputed, its content unchanged.
//!
//! **Where an entry's time 0 can differ from the store's at the same cut.**
//! The store spells a time-sample row by its time's bits, and since this
//! build canonicalizes every time at `oodf_ffi::parse_delta` the spelling a
//! row is stored under depends on the writing build, while the spelling a
//! replay files it under depends on the replaying build. So, for a section
//! S:
//!
//! - **This build's entry replays an older replica's `-0.0` commit** on S,
//!   stored beside a `+0.0` row or alone: the entry holds S's time 0 at
//!   `+0.0` only. Caught at S's next reset or cascade: rule E names both
//!   spellings, the one the entry lacks with an "absent" digest
//!   (`storage::structural::PreImageDigests`), so the commit answers
//!   `CACHE_DIVERGED`, reloads the entry from the store and deletes both
//!   rows — also after a this-build write at `+0.0` made the entry's `+0.0`
//!   real.
//! - **This build's entry after the reconciler's signed-zero merge**
//!   (`storage::time_sample_merge`), which rewrites rows without moving the
//!   cut: the merge only removes a `-0.0` key or moves it onto `+0.0`, so an
//!   entry still holding the old `-0.0` fails that row's existence digest at
//!   the next delete, and an entry holding only `+0.0` matches the store.
//! - **An older replica's entry replays this build's `+0.0` write**, which
//!   the verbatim relay carries as the client's `-0.0`: with no `-0.0` row
//!   in the store, the older entry's phantom `-0.0` fails its existence
//!   digest and reloads.
//! - **An older replica's entry when that `-0.0` row really exists**: the
//!   older build checks only `-0.0`, commits, and leaves this build's `+0.0`
//!   row behind. This is not caught; it is an accepted residual of a rolling
//!   upgrade from, or a rollback to, a build older than this one, closed
//!   only when no such replica runs (SDD §11.3).
//!
//! Between deletes, a write at `+0.0` takes its pre-image from the store, an
//! untouched row's count can only raise `obj:ext`, and the sampled audit may
//! drop a diverged entry first.
//!
//! **Planning only, never read by readers.** The cache serves structural write
//! planning and nothing else: no reader — `SubscribeReadLayer` and its
//! catch-up, checkpointing, `StatLayer`, external-value GC — ever reads it.
//! The compact form makes that a type rule: every reader takes the full
//! `ObjectState`, which a [`CompactState`] is not.
//! It is the one, deliberate amendment of §21.2's rule that layer content is
//! never cached in memory. A reader that consulted it would serve state that
//! is fenced only for the writer that holds it.
//!
//! # Budget, weight and eviction (SR-3.3)
//!
//! An entry weighs its state ([`estimate_state_bytes`], the
//! [`CacheEntry::state_weight_bytes`]) plus its persistent plan indexes
//! ([`CacheEntry::index_weight_bytes`]). The budget (derived from the pod
//! memory limit) bounds the states: an entry whose
//! state is heavier than the per-entry cap — a quarter of the budget, but
//! never above `u32::MAX` bytes — is never stored, and its object gets an
//! [`UncacheableReason::OverWeight`] negative entry carrying the state's
//! weight. The indexes count against neither the cap nor the cold start's
//! lower bound ([`state_weight_lower_bound`]), so which layers are cacheable
//! does not depend on them. The cache's **resident budget** is the budget ×
//! (1 + φ) ([`resident_budget_bytes`], φ = [`PLAN_INDEX_BUDGET_CENTI`] / 100),
//! which bounds the entries' whole weight, state and indexes, and is split
//! in two. Entries live in a `mini_moka::sync::Cache` keyed by `object_id`,
//! with `max_capacity` = the resident budget less its quarter, each entry
//! charged its whole weight in 4-byte units ([`slot_weight`]), and an idle
//! TTL ([`STRUCTURAL_CACHE_IDLE_TTL`] in production); its own eviction is
//! weighted W-TinyLFU, not strict LRU, while the resident bound's step
//! (below) evicts by the entries' last use. The remaining quarter bounds the
//! side store (below). The `live_edit_structural_cache_weight_bytes` gauge
//! ([`StructuralCache::weight_bytes`]) reports the whole weight, state and
//! indexes, of both.
//!
//! **The resident bound.** After every store, mini-moka's applied weight
//! plus the side store's is within the resident budget, by construction,
//! whatever an entry's indexes weigh. φ is sized to cover the indexes of
//! every weighed layer shape but tiny ones ([`PLAN_INDEX_BUDGET_CENTI`]: by
//! this weigher a four-section rig sits at 0.24, every other shape at most
//! 0.13), so such a layer keeps the state capacity the budget gives it. An
//! entry whose indexes weigh more is still cached, never refused for it
//! (one exception, below): one whose admission mini-moka refused and whose
//! whole weight is over the side store's quarter is kept there alone
//! (`Control::keep_in_side_store`), and the main cache makes room for its
//! excess. Every store — a put-back, the guard's `Drop`, a refused
//! cold-start charge's put-back — ends by applying mini-moka's pending
//! writes and, while the two weights together are over the resident budget,
//! evicting main-cache entries least recently used first, then keys still
//! checked out (`Control::hold_resident_bound`).
//! Every mini-moka call runs under the cache's one lock and only a store
//! raises either weight, so the bound holds between any two cache
//! operations, not just eventually. A heavier-indexed layer, or the
//! layers it displaces, is only evicted earlier, never made uncacheable: an
//! index above φ costs capacity, never memory.
//!
//! One entry alone heavier than the whole resident budget — or than the
//! most mini-moka's `u32` weigher charges in full, when that is lower — is
//! the one exception: the store evicts it at once, before placing anything
//! (`Control::store`, `held_entry_bound`), and its next lookup cold-starts.
//! Only its indexes can take it there, its state being within the cap: a
//! fresh layout's indexes weigh at most 1.75 × its state by the weigher's
//! per-row arithmetic, plus a fixed overhead (`size_of::<PlanIndexes>()`
//! and the small-table remainder), but a name-index vector keeps the
//! largest capacity it ever had, through rollbacks and committed deletes
//! alike, so a long-cached entry's indexes have no bound relative to its
//! state. The rebuilt entry, its indexes fresh, is within the bound at
//! every budget the configuration accepts (1 MiB up; below about a
//! kilobyte, reachable only by building the cache directly, that fixed
//! overhead alone exceeds the resident budget), so this rule never evicts
//! a rebuild as built there: one cold start per eviction, capacity, never
//! memory or cacheability. A plan whose own name-vector growth takes the
//! entry over the bound (possible at small budgets) has it evicted again at
//! its put-back, each retry of that delta cold-starting once more. So every
//! entry the cache holds weighs at most the resident budget, which is what
//! lets emptying the main cache always bring the total within it.
//!
//! An entry whose state is exactly at the cap fits both halves with indexes
//! up to φ of it, because three quarters of the resident budget is never
//! below the cap × (1 + φ), nor a quarter of it below the cap × (1 + φ). The
//! 4-byte unit lets mini-moka's `u32` weight charge every entry the cache
//! holds in full at every budget, an entry at the cap's `u32::MAX` ceiling
//! (the cap from a 16 GiB budget up) included. A budget of 0 disables the
//! cache: every lookup answers [`UncacheableReason::Disabled`]; a budget
//! below 4 bytes has a cap of 0, so every entry is over it.
//!
//! Every stored entry is compacted first ([`compact_state`], and the plan
//! indexes' own maps): a table thinned by deletes keeps its oversized bucket
//! array, which `capacity()` understates, so without compaction a
//! delete-heavy object would be charged about half its real memory for as
//! long as it stayed cached. It is then weighed in O(1)
//! ([`CacheEntry::weights_now`]): the state's weight from its tables'
//! capacities and the plan indexes' running sums ([`indexed_state_bytes`],
//! equal to [`estimate_state_bytes`] once compacted), the indexes' from their
//! own capacities. No table is reserved ahead of a plan's inserts; a table
//! grows only when an insert crosses its load limit.
//!
//! Between two compactions — while a call holds the entry checked out, its
//! repair and plan in place thinning it — a table's `capacity()` can hide
//! buckets behind tombstones, all of them once tombstones fill every usable
//! slot, so every in-call weigh (CR1,
//! CR2, the repair's RP0–RP2) reads each hash table at its **held capacity**
//! instead: the largest `capacity()` it reported just before a removal since
//! the last compaction, or its `capacity()` when larger
//! (`PlanIndexes::held_capacity`). So an entry is never weighed below the
//! bytes its tables hold. The high-waters reset at compaction, after it and
//! before the weigh ([`CacheEntry::compact_and_weigh`]), so stored weights
//! are the compacted tables' exact ones. The sampled audit's copy, which
//! holds no indexes, carries a bucket bound of its own instead
//! ([`CopyBucketBound`]).
//!
//! **The weight is a step function of the layer's size, not a leak.** Each
//! hash table holds a power-of-two bucket count, so an entry's weight jumps
//! when one of its tables doubles: on the development layer shape, the
//! fields table doubles at about 61k sections, and the entry goes from about
//! 20.0 MB to 27.4 MB with one more section. The gauge shows the same step.
//!
//! Each object's mini-moka value is a shared **slot** holding the entry. A
//! checkout takes the entry out of its slot and leaves the key resident; a
//! put-back refills the slot by re-inserting the same key, which mini-moka
//! treats as an update of a resident key (its weight refreshed), not a new
//! admission. mini-moka can silently refuse a *new* key near capacity
//! (W-TinyLFU admission). A put-back whose key is not resident — a cold-started
//! or reloaded entry, or one evicted while checked out — is therefore checked
//! after mini-moka has applied it (`sync`, then presence), and a refused entry
//! goes to a small, bounded **side store** that is served as a normal
//! [`CacheLookup::Hit`]. A refused admission never surfaces as an uncacheable
//! verdict, so it never costs the object another cold-start read.
//!
//! # Negative entries (SR-3.4)
//!
//! An object found uncacheable — over the weight cap, without
//! `chs_complete`, or too large for its fenced cold-start read to fit the
//! configured RPC deadline — gets a **negative entry**
//! carrying its reason, its estimated weight when one is known (reported in
//! the reason, never charged against the budget) and its creation instant.
//! While it lives, [`StructuralCache::lookup`] answers
//! [`CacheLookup::Uncacheable`] with no further read. Negative entries live
//! outside mini-moka (a new key there could be refused, and charging an
//! estimated weight would evict real entries) and expire at an absolute time,
//! their creation plus the negative TTL ([`STRUCTURAL_CACHE_IDLE_TTL`] in
//! production): a lookup does not extend it. Retirement drops them.
//!
//! **Each verdict is reported exactly once.** A negative entry is recorded
//! *unreported*, whichever path records it — a cold start, the probe, an
//! explicit put-back the per-entry cap refused, or the implicit put-back of a
//! dropped guard. Its report (`live_edit_structural_cache_uncacheable_total`
//! and its INFO line, `structural_cache_cold_start::report_uncacheable`) is
//! made by whoever first takes it with [`StructuralCache::take_unreported`],
//! which flips the flag under the cache's lock, so two takers never both get
//! it: the path that reached the verdict, at once, when it can report (it
//! holds the bounds the verdict compared); otherwise the first structural
//! lookup it routes (`structural_cache_read::read_for_structural_edit`). So a
//! verdict no caller saw — a dropped call's over-cap entry — is still reported
//! once, and a verdict already reported is never reported again by the
//! lookups it routes. A re-recorded negative entry is a new verdict, reported
//! again. Readers never take it: only the structural write path does.
//!
//! # Checkout and the epoch check (SR-3.4)
//!
//! A structural or reset call holds the object's per-replica `structural`
//! permit (SR-7.2), so at most one call on a replica uses an object's entry at
//! a time, and that call **takes the entry out** of the cache for its
//! duration: [`StructuralCache::lookup`] or [`StructuralCache::checkout`]
//! returns a [`CheckedOut`] guard that owns the original entry, and
//! [`StructuralCache::admit`] hands a cold-started or reloaded entry back the
//! same way. The cache does not itself serialize calls on one object: should
//! two guards for one object overlap, both put-backs pass the epoch check and
//! the later one wins. Each is a valid cut, so the next repair brings either
//! current.
//!
//! The cache keeps one cache-wide, monotonically increasing `u64` **epoch**,
//! never reset. [`StructuralCache::invalidate`] advances it and records, per
//! object, the epoch at which it happened (an audit's `diverged`, object
//! retirement, a [`CheckedOut::discard`]); every checkout records the epoch it
//! was taken at. A put-back — explicit through [`CheckedOut::put_back`] or
//! implicit through the guard's `Drop` — is refused when the object was
//! invalidated after the checkout's epoch, so a straggler that checked out
//! before a `DeleteLayer`, an overwrite or an audit drop cannot restore a
//! stale or retired entry, even if a fresh entry for the object appeared
//! meanwhile.
//!
//! A cold start's epoch is taken **before its read**: it begins with
//! [`StructuralCache::begin_cold_start`], whose [`ColdStartTicket`] registers
//! a checkout at the current epoch, and [`StructuralCache::admit`] hands the
//! read's entry back under the ticket's epoch. So an invalidation that lands
//! while the cold start is reading — a `DeleteLayer` or overwrite retiring the
//! object, an audit drop — refuses the admission (or, after it, the
//! put-back): a read taken before a retirement never restores the retired
//! object's entry. An invalidation that preceded the ticket — a reload's own
//! `discard`, an earlier retirement — never refuses it.
//!
//! An invalidation record is kept for as long as some checkout or ticket
//! taken before it is still outstanding — every such guard settles by its
//! call's RPC deadline at the latest — and is pruned once none is.
//!
//! The guard settles the entry exactly once, on whichever exit the call takes
//! (the Mechanism lifecycle of the spec SDD):
//!
//! - [`CheckedOut::put_back_held`] with the held entry: after a `Committed`
//!   commit, as its plan left it in place and
//!   [`CheckedOut::advance_committed`] advanced it; after every other
//!   outcome that keeps the object cached — `STALE`, `FENCED`, a delta
//!   `Err`, a bounded plan's overflow, a refused charge (`NeedsCharge`), the
//!   size gate, a transport error, an in-doubt reply, a terminal store-side
//!   refusal that changed nothing, or a commit token found already landed —
//!   with its pending plan rolled back first, so the original (repaired, if
//!   a repair ran) goes back;
//! - [`CheckedOut::discard`] after `CACHE_DIVERGED`, a repair anomaly, a
//!   header showing the object gone or replaced (`NotFound`, `SUPERSEDED`,
//!   `PARTIAL_HEADER`), or a `Committed` own creation that adopted a dangling
//!   parent: nothing goes back and the object is invalidated;
//! - the guard's `Drop`, when the call is dropped on its deadline or on
//!   cancellation, or unwinds from a panic: the held entry goes back by
//!   RAII, subject to the same epoch check — re-weighed against the
//!   per-entry cap first if a repair advanced it, and with a pending plan
//!   rolled back first (a call dropped mid-commit) — except that it is
//!   discarded instead when a panic unwound out of a repair step, a plan in
//!   place or a rollback that was mutating it, when a pending plan's
//!   rollback fails or panics, or when a panic unwinds through the guard
//!   while a plan is pending (a rollback then could panic again and abort
//!   the process).
//!
//! An explicit put-back says how it settled ([`PutBack`]): stored, refused by
//! the epoch check (`Invalidated`), or refused by the per-entry cap
//! (`OverCap`), which records the object's `OverWeight` negative entry. The
//! caller counts an `OverCap` refusal — the object grew past the cap while
//! cached — on `live_edit_structural_cache_reloads_total{reason="uncacheable"}`
//! and reports it as an `over_weight` verdict
//! (`live_edit_structural_cache_uncacheable_total` and its INFO line); the
//! cache itself ticks no metric and logs nothing. The implicit put-back of
//! `Drop` reports to no one, and ticks no reload; an `OverWeight` verdict it
//! records is reported by the first structural lookup it routes ("Negative
//! entries" above).
//!
//! Planning runs **in place** on the held entry ([`CheckedOut::plan_in_place`],
//! SR-3.5), with no copy of its state. `apply_delta` can leave partial state
//! on `Err` (§9.1), so the plan records each key it touches on its first
//! touch in a pre-image journal (`domain::plan_journal`), and every exit but
//! the commit rolls it back before the entry goes back
//! ([`CheckedOut::put_back_held`], or the guard's `Drop`): the entry that goes
//! back is always a valid cut. The entry's persistent plan indexes
//! ([`CacheEntry::plan`]) answer the plan's lookups and depth check. The
//! stream repair also advances the held entry in place, one replayed record
//! at a time (`CheckedOut::mutate`), and it discards the guard after any
//! replay error; between its reads the entry is always a valid cut.
//!
//! # Diagnosing a slow structural edit
//!
//! Under `lua` a structural or reset edit spends its time in up to five
//! places: the layer's per-replica structural slot, getting its cached entry
//! (hit, stream repair or whole-layer read), planning, the one commit call,
//! and retries. Work through them in this order on the replica that served the
//! slow edits (localise it by scrape-target labels: the cache families carry
//! no `service` label). Each metric's reading is in its rustdoc in
//! `observability::metrics`; healthy figures to compare against are SDD
//! §21.1, §21.2 and §21.5.
//!
//! 1. **Which route?** p99 of `live_edit_update_layer_commit_duration_seconds`
//!    (`outcome="ok"`) by `path`: `structural` planned on a cached entry,
//!    `structural_full` read the whole layer first, `reset` a reset. The commit
//!    call's own share is `live_edit_update_layer_structural_commit_seconds`.
//!    Slow `watch` commits under `lua` are fallbacks: go to step 5.
//! 2. **Retrying, or queueing?** `live_edit_update_layer_attempts_histogram`
//!    and `live_edit_update_layer_conflicts_total` by `kind` on the structural
//!    paths. Attempts near 1 with a long commit time is queueing for the
//!    layer's slot (structural edits to one layer run one at a time per
//!    replica). Rising `seq_fence` is other commits landing between plan and
//!    commit (an attempt, a repair and a re-plan each, no whole-layer read);
//!    `cache_diverged` is the divergence alert; `snapshot_fence_wait` is a wait
//!    behind a fenced read, spending no retry.
//! 3. **How often is the whole layer read?**
//!    `live_edit_structural_cache_lookups_total` by `outcome` (`cold` and
//!    `reload` read it; `hit` and `repaired` do not) and
//!    `live_edit_structural_cache_reloads_total` by `reason`.
//! 4. **What does a whole-layer read cost?**
//!    `live_edit_structural_cold_start_read_duration_seconds` and
//!    `live_edit_structural_cold_start_reads_total` by `route`. A long p99
//!    beside a short `live_edit_snapshot_read_duration_seconds`
//!    (`consumer="structural_cache"`) is loads queueing for the replica's two
//!    whole-layer read turns. A `fenced` load also waits on
//!    `live_edit_catchup_snapshot_read_wait_seconds` and
//!    `live_edit_snapshot_fence_reader_wait_seconds`, and pauses the layer's
//!    other edits for its `live_edit_snapshot_fence_hold_seconds`.
//! 5. **On the slower path?** `live_edit_update_layer_attempts_histogram_count`
//!    (`path="watch"`), `live_edit_structural_cache_uncacheable_total` and
//!    `live_edit_update_layer_structural_fallbacks_total` by `reason`. Each
//!    WATCH attempt reads the whole layer and is voided by a commit landing
//!    during that read (`watch_read_moved`), so on a large busy layer these are
//!    the slow edits.
//! 6. **Is the shard busy?** p99 of `live_edit_snapshot_read_duration_seconds`
//!    (`path="atomic"`) by `consumer`: atomic reads of any consumer hold the
//!    shard, so if fast-path commit time rose at the same moment the delay is
//!    the shard's.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use mini_moka::sync::{Cache as MokaCache, ConcurrentCacheExt};
use ordered_float::OrderedFloat;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::domain::alloc_model::{
    allocation_bytes, bucket_capacity, table_buckets, table_bytes, table_growth_bounded,
    table_growth_bytes, table_growth_bytes_held, vec_growth_bytes,
};
use crate::domain::delta::{
    apply_delta_bounded_indexed, cascade_scratch_bytes, list_delete_sink_bytes,
    reset_scratch_bytes, subtree_pushes, ApplyDeltaStats, BoundedPlan, DeltaError,
    MutationAllowance, RewriteBudget, RewriteCount, StorageMutation, StructuralInserts,
};
use crate::domain::delta_write::RebuildTable;
use crate::domain::limits::MAX_SECTION_TREE_DEPTH;
use crate::domain::pass_timing::{self, Pass};
use crate::domain::plan_index::{
    list_bytes, state_table_shape, table_slot_bytes, PlanIndexes, PlanTouches,
};
use crate::domain::plan_journal::{PlanJournal, PreImageView, TornJournal};
#[cfg(any(test, feature = "test-support"))]
use crate::domain::types::clone_with_room;
use crate::domain::types::{
    CompactFields, CompactRows, CompactSamples, CompactState, Field, FieldRows, FieldValue,
    FullRows, NameInterner, ObjectStateOf, SampleRows, Section, TimeSample, ValueRows,
};
use crate::observability::metrics::{
    STRUCTURAL_CACHE_UNCACHEABLE_FENCED_READ_FAILED, STRUCTURAL_CACHE_UNCACHEABLE_NO_CHS_COMPLETE,
    STRUCTURAL_CACHE_UNCACHEABLE_OVER_DEADLINE, STRUCTURAL_CACHE_UNCACHEABLE_OVER_WEIGHT,
    STRUCTURAL_CACHE_UNCACHEABLE_RELOAD_LIMIT,
};
use crate::storage::diff_read::SnapshotCardinalities;
use crate::storage::structural_cache_repair::{
    advance_to_successor, plan_adopts_dangling_parent, ReloadReason,
};

/// How long an entry may sit unused before it is evicted, and how long a
/// negative entry lives from its creation (SR-3.3, SR-3.4): 600 s.
pub const STRUCTURAL_CACHE_IDLE_TTL: Duration = Duration::from_secs(600);

/// The side store's entry bound. It holds only entries mini-moka's admission
/// refused, and its total weight is bounded by a quarter of the resident
/// budget ([`resident_budget_bytes`]), the quarter mini-moka's capacity
/// leaves out — except one entry alone over that quarter, which is kept
/// rather than refused, the main cache evicting to make room for its excess,
/// so the total stays within the resident budget (module docs, "The resident
/// bound"). That entry is itself within the resident budget: a store evicts
/// a heavier one at once instead of keeping it ([`held_entry_bound`]).
const SIDE_STORE_MAX_ENTRIES: usize = 64;

/// φ, in hundredths: 21, so φ = 0.21. The budget (derived from the pod
/// memory limit) sets the per-entry cap on state
/// weight; the cache bounds its entries' whole weight — state plus plan
/// indexes — by that budget × (1 + φ) ([`resident_budget_bytes`]) whatever
/// an entry's indexes weigh (module docs, "The resident bound"), and a layer
/// whose indexes weigh at most φ of its state keeps the state capacity the
/// budget gives it. Set at ⌈1.25 × the largest index-to-state ratio
/// measured (jemalloc's allocated bytes) on the weighed layer shapes⌉, in
/// hundredths: 1.25 × 0.165 → 0.21. The largest is the smallest layer, a
/// four-section rig whose index is nearly all fixed overhead; a dev-shape
/// layer's is 0.09–0.10. By this cache's own weigher
/// ([`CacheEntry::index_weight_bytes`]) the other shapes weigh 0.06–0.13,
/// and that rig 0.24: tiny layers sit above φ, their indexes mostly fixed
/// overhead. That is safe — the weigher charges every entry its actual
/// state and indexes, and the resident memory stays within the budget ×
/// (1 + φ) whatever they weigh, so such a layer, or the layers it
/// displaces, is only evicted earlier, never refused (module docs, "The
/// resident bound"): an index above φ costs capacity, never memory — so φ
/// is not raised for them: 0.31 would
/// lower the catch-up memory budget's default on every replica for the
/// sake of a four-section layer.
/// The derivation, its figures and its recalibration trigger:
/// `docs-internal/validation/structural-edit-inplace-b-ug2-20261010/report.md`
/// §2; SDD §21.2.
pub const PLAN_INDEX_BUDGET_CENTI: u64 = 21;

// The `live_edit_structural_cache_weight_bytes` gauge's description
// (`observability::metrics`) states the raised bound as "1.21 times" the
// budget: change it with φ.
const _: () = assert!(PLAN_INDEX_BUDGET_CENTI == 21);

/// The cache's resident budget for a configured budget of `budget_bytes`:
/// `budget_bytes` × (1 + φ) ([`PLAN_INDEX_BUDGET_CENTI`]), rounded up — what
/// mini-moka's capacity and the side store's bound split, and the most the
/// `live_edit_structural_cache_weight_bytes` gauge reads, whatever an
/// entry's indexes weigh (module docs, "The resident bound"). `None` when it
/// does not fit in a `u64`, which no budget derived from a pod memory limit
/// (5 % of it, `config::memory_plan`) comes near.
pub const fn resident_budget_bytes(budget_bytes: u64) -> Option<u64> {
    let raised = (budget_bytes as u128 * (100 + PLAN_INDEX_BUDGET_CENTI as u128)).div_ceil(100);
    if raised > u64::MAX as u128 {
        None
    } else {
        Some(raised as u64)
    }
}

/// The bytes one unit of mini-moka's `u32` weigher stands for: an entry is
/// charged its weight in 4-byte units, rounded up ([`slot_weight`]), so any
/// entry the cache holds — at most [`held_entry_bound`], its state within the
/// per-entry cap (at most `u32::MAX` bytes) — is charged in full at every
/// budget.
const SLOT_UNIT_BYTES: u64 = 4;

/// The number of whole-layer cold-start reads, plain or fenced, one replica
/// runs at once: 2, the cache's plain permits (module docs, "Cold-start
/// permits"). A fixed constant by design, not a knob.
///
/// A plain read's reply is bounded by the shared byte ceiling, 604,619 B at
/// the default. A fenced read has no ceiling; its memory is a
/// `MemoryBudget` grant inside the catch-up build budget `T`
/// (`storage::structural_cache_fenced`), so it is not a separate term of the
/// pod's memory budget (project SDD §21.2); nor is the sampled audit's
/// fenced read, which draws on the same budget.
pub const STRUCTURAL_COLD_START_READ_PERMITS: usize = 2;

/// One object's cached, decoded state at one `(seq, commit_chain)` cut
/// (SR-3.1), in the compact form ([`CompactState`]: per value row, its key
/// and its external hash only).
///
/// A normalized entry equals the compact projection of the stored object's
/// rows at that cut — its header scalars, live sections and children lists
/// under live parents as `snapshot_to_object_state` decodes them, and per
/// value row its key and validated external hash, each time-sample row keyed
/// by its stored time bits: that equality is the cache contract (SR-3.2).
///
/// Beside the state it holds the persistent plan indexes ([`Self::plan`]),
/// which equality ignores: two entries are equal when their cut and state
/// are, and an entry's indexes are a function of its state.
#[derive(Debug)]
pub struct CacheEntry {
    /// The object this entry caches.
    pub object_id: u64,
    /// The header's `commit_chain` at [`Self::seq`]: the object's identity and
    /// its history in one value (SR-4.5). A repair folds it over every
    /// replayed stream entry's payload and must reach the header's value.
    pub commit_chain: String,
    /// The header `seq` the entry's state is at.
    pub seq: u64,
    /// Parents whose stored children list the entry deliberately does not
    /// hold, because the parent is neither the root nor a live section — the
    /// snapshot readers' "lists under dangling parents stay excluded" filter
    /// (SR-3.2).
    pub dangling_parents: HashSet<u64>,
    /// The object's complete-mode state in the compact form, normalized
    /// (SR-3.2): no empty children list, no list under a dangling parent.
    /// Planning only: no reader takes a [`CompactState`].
    pub state: CompactState,
    /// The estimated in-memory weight of [`Self::state`],
    /// [`estimate_state_bytes`] of it: what the per-entry cap is checked
    /// against and what an [`UncacheableReason::OverWeight`] verdict reports
    /// (SR-3.3). The cache recomputes it whenever an entry is admitted or put
    /// back — in O(1) from the plan indexes' running sums when the entry
    /// holds them, equal to the estimate there, since every store compacts
    /// first — so a caller never has to keep it current. A
    /// [`successor_entry`](crate::storage::structural_cache_repair::successor_entry)
    /// result carries its predecessor's weight, a placeholder, until its
    /// put-back sets the weight of its own state.
    pub state_weight_bytes: u64,
    /// The in-memory weight of [`Self::plan`] (`PlanIndexes::weight_bytes`),
    /// 0 for an entry without them; recomputed with
    /// [`Self::state_weight_bytes`]. Not part of the cap check: it is charged
    /// against the cache's resident budget beside the state
    /// ([`Self::weight_bytes`]).
    pub index_weight_bytes: u64,
    /// The entry's last use: set when it is admitted, checked out or put
    /// back explicitly; the guard's `Drop` put-back keeps the checkout's
    /// instant unless a repair advanced the entry, which it re-weighs and
    /// stamps as an explicit put-back does (an unadvanced original keeps it,
    /// a rolled-back plan's included). The side store evicts and expires by
    /// it, and the resident
    /// bound (`Control::hold_resident_bound`) orders its victims by it;
    /// mini-moka keeps its own access times for the idle TTL.
    pub last_used: Instant,
    /// The persistent plan indexes of [`Self::state`]
    /// (`domain::plan_index`), which the stream repair keeps in step as it
    /// replays foreign commits: its `(parent_id, name)` lookups, adoptions
    /// and depth checks read them instead of rebuilding per-delta indexes
    /// from the whole state, and the plan made in place on the entry
    /// ([`CheckedOut::plan_in_place`]) keeps them in step too. Built by
    /// [`Self::new`] — at a cold start or reload, after the read's fence is
    /// released and the state normalized. `None` on a state-only copy
    /// ([`Self::copy_without_plan`], the sampled audit's), whose repair takes
    /// the unindexed path, and on a copy-based successor
    /// ([`successor_entry`](crate::storage::structural_cache_repair::successor_entry));
    /// a plan in place builds them for an entry that has none.
    pub plan: Option<PlanIndexes>,
}

impl Clone for CacheEntry {
    /// A field-wise clone, except that the plan indexes, when the entry
    /// holds them, are rebuilt from the cloned state
    /// ([`PlanIndexes::build`]): a cloned string or list is allocated at its
    /// length, so the original's running allocation sums would not describe
    /// the clone. The rebuilt indexes take the state tables' capacity
    /// high-waters from the original's, since a cloned map keeps its buckets
    /// and its tombstones (hashbrown's `RawTable::clone`), so the clone is
    /// never weighed below the buckets its tables hold; their own maps are
    /// built fresh, so theirs start at 0. The weights are copied as they are,
    /// until a settle re-weighs the clone.
    fn clone(&self) -> Self {
        let state = self.state.clone();
        let plan = self.plan.as_ref().map(|original| {
            let mut rebuilt = PlanIndexes::build(&state);
            for table in RebuildTable::ALL {
                if table.in_state() {
                    rebuilt.note_capacity(table, original.capacity_high_water(table));
                }
            }
            rebuilt
        });
        Self {
            object_id: self.object_id,
            commit_chain: self.commit_chain.clone(),
            seq: self.seq,
            dangling_parents: self.dangling_parents.clone(),
            state,
            state_weight_bytes: self.state_weight_bytes,
            index_weight_bytes: self.index_weight_bytes,
            last_used: self.last_used,
            plan,
        }
    }
}

impl PartialEq for CacheEntry {
    /// The cut, the state and the bookkeeping fields; not [`CacheEntry::plan`],
    /// which is a function of the state, nor its weight.
    fn eq(&self, other: &Self) -> bool {
        self.object_id == other.object_id
            && self.commit_chain == other.commit_chain
            && self.seq == other.seq
            && self.dangling_parents == other.dangling_parents
            && self.state == other.state
            && self.state_weight_bytes == other.state_weight_bytes
            && self.last_used == other.last_used
    }
}

impl CacheEntry {
    /// An entry for `state` at the `(seq, commit_chain)` cut, last used now,
    /// with its persistent plan indexes built from `state`
    /// ([`PlanIndexes::build`], one pass) and weighed from them
    /// ([`Self::weights_now`]). `state` must already be normalized, with its
    /// dangling parents in `dangling_parents` (SR-3.2). A commit's successor
    /// is built with
    /// [`successor_entry`](crate::storage::structural_cache_repair::successor_entry)
    /// instead, which keeps the predecessor's dangling parents as a repair
    /// would.
    pub fn new(
        object_id: u64,
        commit_chain: String,
        seq: u64,
        dangling_parents: HashSet<u64>,
        state: CompactState,
    ) -> Self {
        let plan = Some(PlanIndexes::build(&state));
        let mut entry = Self {
            object_id,
            commit_chain,
            seq,
            dangling_parents,
            state,
            state_weight_bytes: 0,
            index_weight_bytes: 0,
            last_used: Instant::now(),
            plan,
        };
        (entry.state_weight_bytes, entry.index_weight_bytes) = entry.weights_now();
        entry
    }

    /// A copy of the entry without its plan indexes: the cut and the state
    /// only. The sampled audit repairs such a copy, unindexed, so it holds
    /// one copy of the state and none of the indexes.
    pub fn copy_without_plan(&self) -> Self {
        Self {
            object_id: self.object_id,
            commit_chain: self.commit_chain.clone(),
            seq: self.seq,
            dangling_parents: self.dangling_parents.clone(),
            state: self.state.clone(),
            state_weight_bytes: self.state_weight_bytes,
            index_weight_bytes: 0,
            last_used: self.last_used,
            plan: None,
        }
    }

    /// The weight the cache charges the entry against its budget: its
    /// state's and its plan indexes' (SR-3.3). The per-entry cap reads the
    /// state's alone ([`Self::state_weight_bytes`]).
    pub fn weight_bytes(&self) -> u64 {
        self.state_weight_bytes
            .saturating_add(self.index_weight_bytes)
    }

    /// The state's and the plan indexes' weights as the entry stands now,
    /// whatever a repair or plan changed since it was last weighed: in O(1)
    /// when it holds its plan indexes ([`indexed_state_bytes`] and
    /// `PlanIndexes::weight_bytes`, every hash table read at its held
    /// capacity — equal to [`estimate_state_bytes`] at every compaction, the
    /// settle check, and above it while a table thinned since then hides
    /// buckets from the estimate's `capacity()` reading, so never below the
    /// bytes the tables hold), otherwise the state's by the whole-state
    /// estimate and no index weight.
    pub fn weights_now(&self) -> (u64, u64) {
        match self.plan.as_ref() {
            Some(indexes) => (
                indexed_state_bytes(&self.state, indexes),
                indexes.weight_bytes(),
            ),
            None => (estimate_state_bytes(&self.state), 0),
        }
    }

    /// Compact the state's tables ([`compact_state`]) and the plan indexes'
    /// maps (`PlanIndexes::compact`), reset the indexes' capacity high-waters
    /// (`PlanIndexes::reset_capacity_high_water`), then set both weights
    /// ([`Self::weights_now`]): what the cache does to every entry it admits
    /// or puts back. O(1) for tables already minimal for their length, as
    /// a table grown only by inserts is, and for an entry holding its plan
    /// indexes; no reserve is made ahead of a later plan's inserts.
    ///
    /// The reset is sound because the compaction leaves every registered
    /// table minimal for its length, its `capacity()` reading its exact
    /// bucket count (the test-support settle check, `assert_weighed`, checks
    /// each table against its true buckets). Resetting before or after the
    /// compaction gives the same result, since nothing between them reads
    /// the high-waters. What is load-bearing is that the reset comes before
    /// the weigh: weighed first, the entry would be stored at the uncompacted
    /// tables' bucket counts (`assert_weighed` catches that too).
    fn compact_and_weigh(&mut self) {
        compact_state(&mut self.state);
        if let Some(indexes) = self.plan.as_mut() {
            indexes.compact();
            indexes.reset_capacity_high_water();
        }
        (self.state_weight_bytes, self.index_weight_bytes) = self.weights_now();
    }

    /// The capacity each of the state's hash tables is weighed at: with
    /// plan indexes, its held capacity (`PlanIndexes::held_capacity`), which
    /// its tombstones cannot hide; without, its `capacity()`, which they can,
    /// so that fallback is **not** a bound on the buckets a table holds. Read
    /// by the sampled audit before it copies the entry without its indexes,
    /// to start the copy's bucket bound ([`CopyBucketBound`]); there the
    /// entry always holds its indexes (the audit trigger asserts it,
    /// `storage::structural_cache_audit`). O(1).
    pub fn held_state_capacities(&self) -> HeldCapacities {
        let mut held = HeldCapacities::live(&self.state);
        if let Some(indexes) = self.plan.as_ref() {
            for table in RebuildTable::ALL {
                if table.in_state() {
                    held.set(table, indexes.held_capacity(table, held.get(table)));
                }
            }
        }
        held
    }

    /// The test-support settle check, after [`Self::compact_and_weigh`]:
    /// every hash table of the table registry reads its true bucket count
    /// from its `capacity()` (`domain::types::true_capacity_for_test`) — the
    /// compaction shrank each, which is what makes the capacity high-water's
    /// reset sound — the O(1) state weight equals the whole-state estimate,
    /// and the plan indexes equal a rebuild of the state with their running
    /// weight undrifted.
    #[cfg(any(test, feature = "test-support"))]
    fn assert_weighed(&self) {
        let tables = match self.plan.as_ref() {
            Some(indexes) => crate::domain::plan_index::hash_tables_for_test(&self.state, indexes),
            None => crate::domain::plan_index::state_hash_tables_for_test(&self.state),
        };
        for (table, capacities) in tables {
            assert_eq!(
                table_buckets(capacities.capacity as u64),
                table_buckets(capacities.true_capacity as u64),
                "settle check: after compaction the {table:?} table's capacity() {} reads fewer \
                 buckets than the {} usable slots it holds",
                capacities.capacity,
                capacities.true_capacity,
            );
        }
        assert_eq!(
            self.state_weight_bytes,
            estimate_state_bytes(&self.state),
            "settle check: the O(1) state weight differs from estimate_state_bytes"
        );
        assert_indexes_rebuilt(self);
    }
}

/// Why an object's structural edits bypass the cache (SR-3.3, SR-3.4,
/// SR-7.4). An uncacheable object's structural and reset calls take the
/// uncacheable path; the verdict is never a client-visible error (SR-7.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UncacheableReason {
    /// The object's estimated state weight, in bytes — its plan indexes not
    /// counted — exceeds the per-entry cap: a quarter of the configured
    /// memory budget, at most `u32::MAX` bytes (SR-3.3). The value is the
    /// state's estimate, or — when the cold start's weight check refused
    /// the object before any read ([`state_weight_lower_bound`]) — the lower
    /// bound that already exceeded the cap.
    OverWeight(u64),
    /// The object's header lacks `chs_complete`, so its dangling children
    /// lists cannot be enumerated (SR-3.2).
    NoChsComplete,
    /// The cache's memory budget is 0: every structural edit is uncacheable
    /// (SR-3.3). Answered by every lookup on a disabled cache and never
    /// recorded as a negative entry.
    Disabled,
    /// This one call reloaded the object as many times as a call may
    /// (`storage::structural_cache_read`'s
    /// `STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START`), and each cold start's
    /// entry still failed its repair: the call takes the uncacheable path
    /// rather than read the object again. A verdict about the call, not the
    /// object, so — like [`Self::Disabled`] — never recorded as a negative
    /// entry: the next call on the object looks it up afresh.
    ReloadLimit,
    /// This one call's fenced cold-start read failed: its wait for capacity
    /// or for the snapshot fence reached the call's waiting cap, a capacity
    /// wait timed out, or [`FENCE_READ_ATTEMPTS`] attempts in a row were broken
    /// (`storage::structural_cache_fenced`). The call takes the uncacheable
    /// path rather than wait on. A verdict about the call, not the object, so
    /// — like [`Self::ReloadLimit`] — never recorded as a negative entry: the
    /// next call on the object cold-starts it afresh.
    ///
    /// [`FENCE_READ_ATTEMPTS`]: crate::storage::snapshot_fence::FENCE_READ_ATTEMPTS
    FencedReadFailed,
    /// The object is too large for its fenced cold-start read to fit the
    /// replica's configured RPC deadline (`LIVE_EDIT_RPC_DEADLINE_MS`): at
    /// the cold start's size estimate, the fenced read's waiting cap with the
    /// whole deadline left is zero, since the deadline does not cover the
    /// read's expected time plus one WATCH attempt's
    /// (`storage::structural_cache_fenced`, "The waiting cap"). Judged on the
    /// object's size and the configuration alone, before any fenced read, so
    /// no call on the replica could load the object: unlike
    /// [`Self::FencedReadFailed`], a verdict about the object, recorded as a
    /// negative entry (`storage::structural_cache_cold_start`, "The route").
    /// A call whose own deadline left is short never reaches it. It clears
    /// when the negative entry expires, once the layer has shrunk or the
    /// deadline has been raised.
    OverDeadline,
}

impl UncacheableReason {
    /// The `reason` label of `live_edit_structural_cache_uncacheable_total`
    /// (a [`STRUCTURAL_CACHE_UNCACHEABLE_REASONS`] value), or `None` for
    /// [`Self::Disabled`]: a disabled cache is a configuration, answered on
    /// every lookup, never a verdict, so it has no cell.
    ///
    /// [`STRUCTURAL_CACHE_UNCACHEABLE_REASONS`]: crate::observability::metrics::STRUCTURAL_CACHE_UNCACHEABLE_REASONS
    pub fn label(self) -> Option<&'static str> {
        match self {
            Self::OverWeight(_) => Some(STRUCTURAL_CACHE_UNCACHEABLE_OVER_WEIGHT),
            Self::NoChsComplete => Some(STRUCTURAL_CACHE_UNCACHEABLE_NO_CHS_COMPLETE),
            Self::ReloadLimit => Some(STRUCTURAL_CACHE_UNCACHEABLE_RELOAD_LIMIT),
            Self::FencedReadFailed => Some(STRUCTURAL_CACHE_UNCACHEABLE_FENCED_READ_FAILED),
            Self::OverDeadline => Some(STRUCTURAL_CACHE_UNCACHEABLE_OVER_DEADLINE),
            Self::Disabled => None,
        }
    }
}

/// The answer [`StructuralCache::lookup`] gives for one object.
//
// `clippy::large_enum_variant` allowed: `Hit` carries the guard by value
// because the cross-track contract fixes the variant as `Hit(CheckedOut)`.
// The value is a transient return, built and matched once per structural
// call and never stored in bulk, so its size costs one move per call.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum CacheLookup {
    /// The object had an entry; it is now out of the cache, owned by the
    /// guard (SR-3.4).
    Hit(CheckedOut),
    /// No entry and no uncacheable verdict: the call cold-starts the object
    /// (SR-4.3).
    Miss,
    /// A live uncacheable verdict for the object: the call takes the
    /// uncacheable path without another full read (SR-3.4, SR-7.4).
    Uncacheable {
        /// Why the object is uncacheable.
        reason: UncacheableReason,
    },
}

/// The per-replica structural-planning cache (SR-3). **Planning only, never
/// read by readers** — see the module documentation.
///
/// Cheap to [`Clone`]: every clone shares one inner state, so the store and
/// every checkout guard see one cache. Every method takes `&self`.
#[derive(Clone)]
pub struct StructuralCache {
    inner: Arc<CacheInner>,
}

/// State shared by every clone of a [`StructuralCache`].
struct CacheInner {
    /// The per-replica cold-start permit (one permit), which a fenced cold
    /// start and the audit's fenced read take before their plain permit:
    /// see
    /// [`StructuralCache::acquire_cold_start_permit`]. Never closed.
    cold_start_permit: Arc<Semaphore>,
    /// The per-replica plain permits ([`STRUCTURAL_COLD_START_READ_PERMITS`]),
    /// one held across every whole-layer cold-start read: see
    /// [`StructuralCache::acquire_plain_cold_start_read_permit`]. Never
    /// closed.
    plain_read_permits: Arc<Semaphore>,
    /// The cold-start reads holding their permits now, by kind: the test
    /// seam's source ([`ColdStartReadPermit`]).
    reads_in_flight: Arc<ReadsInFlight>,
    /// The test hook a cold start calls before `admit`
    /// ([`StructuralCache::set_before_admit_hook_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    before_admit_hook: Mutex<Option<BeforeAdmitHook>>,
    /// The test-only copies made from this cache's checked-out entries
    /// ([`CheckedOut::plan_state`]): the test seam's source
    /// ([`StructuralCache::plan_state_clones_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    plan_state_clones: std::sync::atomic::AtomicU64,
    /// Plans made in place on this cache's checked-out entries
    /// ([`CheckedOut::plan_in_place`];
    /// [`StructuralCache::in_place_plans_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    in_place_plans: std::sync::atomic::AtomicU64,
    /// Plans rolled back ([`StructuralCache::rollbacks_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    rollbacks: std::sync::atomic::AtomicU64,
    /// Rollbacks that rebuilt at least one table
    /// ([`StructuralCache::rollback_rebuilds_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    rollback_rebuilds: std::sync::atomic::AtomicU64,
    /// Whether each plan made in place is shadowed by a plan on a copy and
    /// checked against it, and whether the sampled audit checks its copy's
    /// bucket bound after every record
    /// ([`StructuralCache::set_shadow_check_for_test`]); on by default.
    #[cfg(any(test, feature = "test-support"))]
    shadow_check: std::sync::atomic::AtomicBool,
    /// The configured memory budget in bytes; 0 disables the cache.
    budget_bytes: u64,
    /// The per-entry cap on an entry's state weight: [`max_entry_bytes_for`]
    /// the configured budget (SR-3.3).
    max_entry_bytes: u64,
    /// The resident budget ([`resident_budget_bytes`] of the configured
    /// budget): what mini-moka's applied weight plus the side store's is
    /// brought within after every store (`Control::hold_resident_bound`).
    resident_bytes: u64,
    /// The side store's weight bound: a quarter of the resident budget
    /// ([`resident_budget_bytes`]).
    side_weight_bound: u64,
    /// How long a side-store entry may sit unused before it expires — the
    /// same idle TTL mini-moka applies to resident entries.
    idle_ttl: Duration,
    /// How long a negative entry lives from its creation.
    negative_ttl: Duration,
    /// Everything that changes. One lock for the whole cache, so a checkout's
    /// epoch, a put-back's epoch check and an invalidation are totally
    /// ordered. It is held only across in-memory map work and never across an
    /// `.await`.
    ///
    /// **Entries are freed outside it.** An entry's state can be tens of MiB
    /// (up to the per-entry cap), and freeing it is a walk of every table and
    /// allocation it holds, which would stall every other call on the replica
    /// that touches the cache. So each critical section that drops an entry —
    /// an invalidation, a negative entry, a side-store expiry or eviction, a
    /// refused put-back, a side-store copy a store replaces, an entry a
    /// store evicts at once for its weight with its object's copies
    /// (`Control::store`), a main-cache entry the resident bound evicts
    /// (`Control::hold_resident_bound`) — moves it into a [`Freed`] list
    /// instead, and the list is dropped after the lock is released. A
    /// resident entry is taken out of its mini-moka slot before its key is
    /// invalidated, so the slot mini-moka later drops is empty.
    /// The exceptions are inside mini-moka, which drops them under
    /// this lock with no hook to take them out first: the entries it evicts
    /// for capacity or idleness while applying its pending writes (`sync()`,
    /// and the housekeeping a write runs); an idle-expired entry, which a read
    /// no longer returns, so its slot cannot be emptied before its key is
    /// invalidated; and a slot a put-back replaces while it still holds an
    /// entry, which only overlapping guards for one object leave
    /// (`Control::store`).
    control: Mutex<Control>,
}

/// Entries a critical section removed, dropped by the caller once the
/// control lock is released ([`CacheInner::control`]). Empty — and so free —
/// on most calls: a `Vec` allocates only on its first push.
type Freed = Vec<CacheEntry>;

/// **Test seam.** The resident weight's two halves, read without a `sync()`
/// ([`StructuralCache::resident_bound_view_for_test`]).
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentBoundView {
    /// mini-moka's visible keys' slot weights, in bytes (Σ `slot.weight` × 4).
    pub main_bytes: u64,
    /// The heaviest of those slots, in bytes; 0 when there is none.
    pub largest_main_slot_bytes: u64,
    /// mini-moka's capacity, in bytes (its `max_capacity` × 4); 0 when disabled.
    pub main_capacity_bytes: u64,
    /// The side store's total weight, in bytes.
    pub side_bytes: u64,
    /// The side store's quarter of the resident budget, in bytes.
    pub side_quarter_bytes: u64,
    /// The asked object's side-store entry's weight, in bytes; `None` when it
    /// is not in the side store.
    pub side_entry_bytes: Option<u64>,
}

/// The mutable state behind [`CacheInner::control`].
struct Control {
    /// The cache-wide epoch: monotonically increasing, never reset (SR-3.4).
    /// Advanced by every invalidation.
    epoch: u64,
    /// The resident entries. `None` when the cache is disabled.
    entries: Option<MokaCache<u64, Slot>>,
    /// Entries mini-moka's admission refused, served as hits (module docs).
    side: HashMap<u64, CacheEntry>,
    /// Live and not-yet-pruned negative entries.
    negatives: HashMap<u64, NegativeEntry>,
    /// Each object's last invalidation epoch, kept while some checkout taken
    /// before it is outstanding.
    invalidations: HashMap<u64, u64>,
    /// Outstanding checkouts, counted by the epoch each was taken at. Its
    /// smallest key bounds which invalidation records still matter.
    outstanding: BTreeMap<u64, usize>,
    /// Main-cache keys [`Self::hold_resident_bound`] evicted:
    /// `[filled slots, checked-out keys]`
    /// ([`StructuralCache::resident_bound_evictions_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    resident_bound_evictions: [u64; 2],
    /// Entries [`Self::store`] evicted at their own store for weighing more
    /// than any entry the cache holds ([`held_entry_bound`])
    /// ([`StructuralCache::oversized_evictions_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    oversized_evictions: u64,
}

/// One object's mini-moka value: a shared slot for the entry, and the weight
/// the slot was inserted with (mini-moka reads it once, on insert).
#[derive(Clone)]
struct Slot {
    entry: Arc<Mutex<Option<CacheEntry>>>,
    weight: u32,
}

/// An object's uncacheable verdict (SR-3.4).
struct NegativeEntry {
    reason: UncacheableReason,
    created_at: Instant,
    /// Whether the verdict has been reported: `false` when recorded, set,
    /// once, by [`StructuralCache::take_unreported`] (module docs, "Negative
    /// entries").
    reported: bool,
}

impl StructuralCache {
    /// A production cache for a configured budget of `budget_bytes`: each
    /// entry's state weight is capped at a quarter of it, and the entries'
    /// resident weight, state and plan indexes, is bounded by it × (1 + φ)
    /// ([`resident_budget_bytes`]). With the [`STRUCTURAL_CACHE_IDLE_TTL`]
    /// idle and negative-entry TTLs; 0 disables it, making every object
    /// [`UncacheableReason::Disabled`] (SR-3.3).
    pub fn new(budget_bytes: u64) -> Self {
        Self::with_ttls(
            budget_bytes,
            STRUCTURAL_CACHE_IDLE_TTL,
            STRUCTURAL_CACHE_IDLE_TTL,
        )
    }

    /// A cache bounded by `budget_bytes`, with an explicit `idle_ttl` for
    /// entries and `negative_ttl` for negative entries. Production uses
    /// [`Self::new`]; tests pass short TTLs, because mini-moka only reads the
    /// real clock.
    ///
    /// # Panics
    ///
    /// If `idle_ttl` is longer than 1,000 years (mini-moka's own limit).
    pub fn with_ttls(budget_bytes: u64, idle_ttl: Duration, negative_ttl: Duration) -> Self {
        // Saturating: a budget derived from the pod memory limit is far below
        // one whose resident budget overflows, so only a test's direct
        // construction can reach it.
        let resident = resident_budget_bytes(budget_bytes).unwrap_or(u64::MAX);
        let max_entry_bytes = max_entry_bytes_for(budget_bytes);
        let side_weight_bound = resident / 4;
        let moka_bytes = resident - side_weight_bound;
        // An entry whose state is at the cap and whose indexes weigh at most
        // φ of it fits both halves (module docs).
        debug_assert!(
            u128::from(max_entry_bytes) * u128::from(100 + PLAN_INDEX_BUDGET_CENTI)
                <= u128::from(moka_bytes) * 100,
            "the per-entry cap × (1 + φ) fits mini-moka's three quarters"
        );
        let entries = (budget_bytes > 0).then(|| {
            MokaCache::builder()
                // The resident budget less the side store's quarter (module
                // docs), in the weigher's 4-byte units.
                .max_capacity(moka_bytes / SLOT_UNIT_BYTES)
                .weigher(|_object_id: &u64, slot: &Slot| slot.weight)
                .time_to_idle(idle_ttl)
                .build()
        });
        Self {
            inner: Arc::new(CacheInner {
                cold_start_permit: Arc::new(Semaphore::new(1)),
                plain_read_permits: Arc::new(Semaphore::new(STRUCTURAL_COLD_START_READ_PERMITS)),
                reads_in_flight: Arc::new(ReadsInFlight::default()),
                #[cfg(any(test, feature = "test-support"))]
                before_admit_hook: Mutex::new(None),
                #[cfg(any(test, feature = "test-support"))]
                plan_state_clones: std::sync::atomic::AtomicU64::new(0),
                #[cfg(any(test, feature = "test-support"))]
                in_place_plans: std::sync::atomic::AtomicU64::new(0),
                #[cfg(any(test, feature = "test-support"))]
                rollbacks: std::sync::atomic::AtomicU64::new(0),
                #[cfg(any(test, feature = "test-support"))]
                rollback_rebuilds: std::sync::atomic::AtomicU64::new(0),
                #[cfg(any(test, feature = "test-support"))]
                shadow_check: std::sync::atomic::AtomicBool::new(true),
                budget_bytes,
                max_entry_bytes,
                resident_bytes: resident,
                side_weight_bound,
                idle_ttl,
                negative_ttl,
                control: Mutex::new(Control {
                    epoch: 0,
                    entries,
                    side: HashMap::new(),
                    negatives: HashMap::new(),
                    invalidations: HashMap::new(),
                    outstanding: BTreeMap::new(),
                    #[cfg(any(test, feature = "test-support"))]
                    resident_bound_evictions: [0; 2],
                    #[cfg(any(test, feature = "test-support"))]
                    oversized_evictions: 0,
                }),
            }),
        }
    }

    /// The cache's memory budget in bytes (derived from the pod memory limit
    /// in production); 0 means the cache is disabled.
    /// It sets the per-entry cap on state weight; the entries' resident
    /// weight, plan indexes included, is bounded by
    /// [`resident_budget_bytes`] of it, (1 + φ) times it, whatever their
    /// indexes weigh (module docs, "The resident bound").
    pub fn budget_bytes(&self) -> u64 {
        self.inner.budget_bytes
    }

    /// The per-entry cap — a quarter of the configured budget, at most
    /// `u32::MAX` bytes (module docs): an entry whose **state** weighs more
    /// than this ([`CacheEntry::state_weight_bytes`]) is never stored
    /// (SR-3.3). Its plan indexes are not counted against it.
    pub fn max_entry_bytes(&self) -> u64 {
        self.inner.max_entry_bytes
    }

    /// Look `object_id` up for a structural or reset call.
    ///
    /// A [`CacheLookup::Hit`] **takes the entry out** of the cache: until the
    /// returned guard settles, no other lookup sees it (SR-3.4). An object
    /// with no entry answers [`CacheLookup::Miss`], unless a live negative
    /// entry or the disabled budget makes it [`CacheLookup::Uncacheable`]. A
    /// lookup never extends a negative entry's life.
    pub fn lookup(&self, object_id: u64) -> CacheLookup {
        let mut freed = Freed::new();
        let taken = {
            let mut control = self.inner.lock();
            if control.entries.is_none() {
                return CacheLookup::Uncacheable {
                    reason: UncacheableReason::Disabled,
                };
            }
            if let Some(reason) = control.live_negative(object_id, self.inner.negative_ttl) {
                return CacheLookup::Uncacheable { reason };
            }
            control.take(object_id, self.inner.idle_ttl, &mut freed)
        };
        drop(freed);
        match taken {
            Some((entry, epoch)) => CacheLookup::Hit(self.guard(entry, epoch)),
            None => CacheLookup::Miss,
        }
    }

    /// The estimated weight of `object_id`'s entry, state and plan indexes,
    /// for a structural `UpdateLayer`'s upload charge hint
    /// (`LayerStore::object_state_hint`, SDD §5.4), without taking it: the
    /// resident slot's entry's [`CacheEntry::weight_bytes`] (its slot weight,
    /// to the 4-byte unit, while the entry is checked out), read
    /// through one mini-moka access — which the call's checkout normally
    /// follows, so it costs the key no extra frequency — else the side
    /// store's entry's, read without an access. `None` for an object with no
    /// entry, and on a disabled cache.
    ///
    /// The peek feeds the write path only: no reader consults the cache
    /// (SDD §21.2).
    pub fn weight_hint(&self, object_id: u64) -> Option<u64> {
        let control = self.inner.lock();
        let entries = control.entries.as_ref()?;
        if let Some(slot) = entries.get(&object_id) {
            let held = slot
                .entry
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .as_ref()
                .map(CacheEntry::weight_bytes);
            return Some(held.unwrap_or(u64::from(slot.weight).saturating_mul(SLOT_UNIT_BYTES)));
        }
        control.side.get(&object_id).map(CacheEntry::weight_bytes)
    }

    /// The raw take: check `object_id`'s entry out, with no uncacheable
    /// verdict. `None` when the object has no entry, including on a disabled
    /// cache (SR-3.4).
    pub fn checkout(&self, object_id: u64) -> Option<CheckedOut> {
        let mut freed = Freed::new();
        let taken = self
            .inner
            .lock()
            .take(object_id, self.inner.idle_ttl, &mut freed);
        drop(freed);
        taken.map(|(entry, epoch)| self.guard(entry, epoch))
    }

    /// Begin a cold start or reload (SR-4.3): register a checkout at the
    /// current epoch, before the cold start's read, and return it as a
    /// [`ColdStartTicket`] for [`Self::admit`] to take over.
    ///
    /// Take it before the first read and hold it until the read's entry is
    /// admitted; dropping it unused (a verdict, an error, a cancelled call)
    /// releases the registration. While it is outstanding it keeps alive the
    /// record of any invalidation that lands after it, so `admit` sees that
    /// invalidation (module docs, "Checkout and the epoch check").
    pub fn begin_cold_start(&self) -> ColdStartTicket {
        let epoch = self.inner.lock().register_checkout();
        ColdStartTicket {
            epoch,
            cache: Some(self.clone()),
        }
    }

    /// Hand the cache a cold-started or reloaded entry (SR-4.3), read under
    /// `ticket`, and get it back checked out, for the call that read it to
    /// plan on and settle.
    ///
    /// The entry's state and plan indexes are compacted ([`compact_state`])
    /// and both weighed ([`CacheEntry::weights_now`]). It is not stored until the
    /// guard settles: [`CheckedOut::put_back`] (or the
    /// guard's `Drop`) then stores it, in mini-moka or — if mini-moka's
    /// admission refuses the new key — in the side store, served as a hit
    /// either way.
    ///
    /// The guard's epoch is the ticket's, taken before the read: an
    /// invalidation of the object after the ticket refuses this admission
    /// ([`AdmitRefused::Invalidated`]) or, once admitted, the guard's
    /// put-back; one that preceded the ticket (the `discard` before a reload,
    /// an earlier retirement) never does. Admission clears the object's
    /// negative entry, if one remains.
    ///
    /// # Errors
    ///
    /// Each consumes the ticket and stores nothing.
    ///
    /// - [`AdmitRefused::Uncacheable`] with [`UncacheableReason::Disabled`]
    ///   when the budget is 0; nothing is recorded.
    /// - [`AdmitRefused::Invalidated`] when the object was invalidated after
    ///   the ticket was taken: retired, or dropped by an audit, while the cold
    ///   start read. The read's cut may predate that, so it is discarded and
    ///   nothing is recorded: no negative entry, whatever the entry weighs.
    ///   Checked before the weight.
    /// - [`AdmitRefused::Uncacheable`] with [`UncacheableReason::OverWeight`]
    ///   and the state's estimated weight when the entry's state is heavier
    ///   than [`Self::max_entry_bytes`]; the object gets a negative entry.
    ///
    /// # Panics
    ///
    /// In debug builds, if `ticket` came from another cache.
    pub fn admit(
        &self,
        mut ticket: ColdStartTicket,
        mut entry: CacheEntry,
    ) -> Result<CheckedOut, AdmitRefused> {
        // The guard built below, or the release on refusal, takes over the
        // ticket's registration; disarm its `Drop`.
        let ticket_cache = ticket.cache.take();
        debug_assert!(
            ticket_cache
                .as_ref()
                .is_some_and(|cache| Arc::ptr_eq(&cache.inner, &self.inner)),
            "admit must be given a ticket from this cache"
        );
        let epoch = ticket.epoch;
        if self.inner.budget_bytes == 0 {
            self.inner.release_ticket(epoch);
            return Err(AdmitRefused::Uncacheable(UncacheableReason::Disabled));
        }
        entry.compact_and_weigh();
        #[cfg(any(test, feature = "test-support"))]
        if self.shadow_check_for_test() {
            entry.assert_weighed();
        }
        entry.last_used = Instant::now();
        let object_id = entry.object_id;
        let mut freed = Freed::new();
        let refused = {
            let mut control = self.inner.lock();
            let invalidated = control
                .invalidations
                .get(&object_id)
                .is_some_and(|&invalidated_at| invalidated_at > epoch);
            let refused = if invalidated {
                Some(AdmitRefused::Invalidated)
            } else if entry.state_weight_bytes > self.inner.max_entry_bytes {
                let reason = UncacheableReason::OverWeight(entry.state_weight_bytes);
                control.record_negative(object_id, reason, self.inner.negative_ttl, &mut freed);
                Some(AdmitRefused::Uncacheable(reason))
            } else {
                control.negatives.remove(&object_id);
                None
            };
            if refused.is_some() {
                control.release_checkout(epoch);
                control.prune_invalidations();
            }
            refused
        };
        drop(freed);
        match refused {
            None => Ok(self.guard(entry, epoch)),
            // `entry` drops here, outside the lock.
            Some(refusal) => Err(refusal),
        }
    }

    /// Record `object_id` as uncacheable for `reason` (SR-3.4): until the
    /// negative entry expires — its creation plus the negative TTL — every
    /// [`Self::lookup`] answers [`CacheLookup::Uncacheable`]. Any entry the
    /// object has is dropped. A no-op on a disabled cache, and for
    /// [`UncacheableReason::Disabled`], [`UncacheableReason::ReloadLimit`] and
    /// [`UncacheableReason::FencedReadFailed`], which are never recorded. The verdict is recorded unreported: the
    /// caller reports it by taking it with [`Self::take_unreported`], or the
    /// first structural lookup it routes does.
    pub fn record_uncacheable(&self, object_id: u64, reason: UncacheableReason) {
        if self.inner.budget_bytes == 0
            || matches!(
                reason,
                UncacheableReason::Disabled
                    | UncacheableReason::ReloadLimit
                    | UncacheableReason::FencedReadFailed
            )
        {
            return;
        }
        let mut freed = Freed::new();
        self.inner
            .lock()
            .record_negative(object_id, reason, self.inner.negative_ttl, &mut freed);
        drop(freed);
    }

    /// Take `object_id`'s live negative entry **for reporting**: its reason
    /// — with the estimated weight an `OverWeight` verdict carries — if the
    /// verdict has not been reported yet, marking it reported in the same
    /// critical section, so of any number of callers exactly one gets it per
    /// negative-entry lifetime (module docs, "Negative entries"). `None` when
    /// the object has no live negative entry, when its verdict was already
    /// taken, and on a disabled cache. Checks nothing out and never extends
    /// the entry's life.
    ///
    /// The structural write path's only: the sites that reach a verdict take
    /// it to report it at once, and a structural lookup answered
    /// [`CacheLookup::Uncacheable`] takes a verdict no one reported. No reader
    /// calls it.
    pub fn take_unreported(&self, object_id: u64) -> Option<UncacheableReason> {
        let mut control = self.inner.lock();
        control.live_negative(object_id, self.inner.negative_ttl)?;
        let negative = control.negatives.get_mut(&object_id)?;
        if negative.reported {
            return None;
        }
        negative.reported = true;
        Some(negative.reason)
    }

    /// Invalidate `object_id`: drop any entry and any negative entry it has,
    /// and record the advanced epoch as its last invalidation, so every
    /// put-back by a checkout taken at an earlier epoch is refused (SR-3.4).
    /// Called on an audit's `diverged` (SR-3.6), on object retirement
    /// (`LayerStore`'s `SemaphoreMapGuard`, SR-7.2) and by
    /// [`CheckedOut::discard`]. A no-op on a disabled cache, which hands out
    /// no checkouts.
    pub fn invalidate(&self, object_id: u64) {
        if self.inner.budget_bytes == 0 {
            return;
        }
        let mut freed = Freed::new();
        {
            let mut control = self.inner.lock();
            control.invalidate(object_id, &mut freed);
            control.prune_invalidations();
        }
        drop(freed);
    }

    /// The number of entries held: mini-moka's resident keys (a checked-out
    /// entry's key stays resident) plus the side store. Source of the
    /// `live_edit_structural_cache_entries` gauge.
    pub fn entry_count(&self) -> u64 {
        let mut freed = Freed::new();
        let count = {
            let mut control = self.inner.lock();
            control.expire_side(self.inner.idle_ttl, &mut freed);
            let resident = control.entries.as_ref().map_or(0, |entries| {
                entries.sync();
                entries.entry_count()
            });
            resident.saturating_add(control.side.len() as u64)
        };
        drop(freed);
        count
    }

    /// The estimated weight held, in bytes, state and plan indexes
    /// ([`CacheEntry::weight_bytes`]): mini-moka's weighted size, in bytes
    /// (each entry rounded up to the weigher's 4-byte unit; a checked-out key
    /// stays charged at its last weight), plus the side store's. Within the
    /// resident budget ([`resident_budget_bytes`]) whatever an entry's
    /// indexes weigh: every store brings the cache within it (module docs,
    /// "The resident bound"). Source of the
    /// `live_edit_structural_cache_weight_bytes` gauge.
    pub fn weight_bytes(&self) -> u64 {
        let mut freed = Freed::new();
        let weight = {
            let mut control = self.inner.lock();
            control.expire_side(self.inner.idle_ttl, &mut freed);
            let resident = control.entries.as_ref().map_or(0, |entries| {
                entries.sync();
                entries.weighted_size().saturating_mul(SLOT_UNIT_BYTES)
            });
            control.side.values().fold(resident, |sum, entry| {
                sum.saturating_add(entry.weight_bytes())
            })
        };
        drop(freed);
        weight
    }

    /// **Test-only.** The weight held, in bytes, read **without** applying
    /// mini-moka's pending writes: each visible resident key's slot weight
    /// (its entry's whole weight, to the 4-byte unit, or its last weight
    /// while checked out) plus the side store's. It sums the real resident
    /// weight, not mini-moka's counter, which only a `sync` updates, so a
    /// test can check that a store left nothing pending over the resident
    /// budget (`Control::hold_resident_bound`). Compiled under
    /// `test-support` too, for `tests/proptests.rs`'s resident-bound
    /// property.
    #[cfg(any(test, feature = "test-support"))]
    pub fn unsynced_weight_bytes_for_test(&self) -> u64 {
        let control = self.inner.lock();
        let resident = control.entries.as_ref().map_or(0, |entries| {
            entries.iter().fold(0u64, |sum, resident| {
                sum.saturating_add(
                    u64::from(resident.value().weight).saturating_mul(SLOT_UNIT_BYTES),
                )
            })
        });
        control.side.values().fold(resident, |sum, entry| {
            sum.saturating_add(entry.weight_bytes())
        })
    }

    /// **Test seam.** The main-cache keys the resident bound
    /// (`Control::hold_resident_bound`) has evicted since the cache was
    /// built: `(filled slots, checked-out keys)`. Monotonic.
    #[cfg(any(test, feature = "test-support"))]
    pub fn resident_bound_evictions_for_test(&self) -> (u64, u64) {
        let [filled, checked_out] = self.inner.lock().resident_bound_evictions;
        (filled, checked_out)
    }

    /// **Test seam.** The entries a store has evicted at once, since the
    /// cache was built, for weighing more than any entry the cache holds
    /// (`held_entry_bound`, `Control::store`): each was dropped instead of
    /// placed. Separate from [`Self::resident_bound_evictions_for_test`],
    /// which counts only the main-cache keys the resident bound's step
    /// evicted. Monotonic.
    #[cfg(any(test, feature = "test-support"))]
    pub fn oversized_evictions_for_test(&self) -> u64 {
        self.inner.lock().oversized_evictions
    }

    /// **Test seam.** The two halves of the resident weight as they stand,
    /// read **without** a `sync()` ([`ResidentBoundView`]), with `object_id`'s
    /// side-store entry, if any.
    #[cfg(any(test, feature = "test-support"))]
    pub fn resident_bound_view_for_test(&self, object_id: u64) -> ResidentBoundView {
        let control = self.inner.lock();
        let (main_bytes, largest_main_slot_bytes) =
            control.entries.as_ref().map_or((0, 0), |entries| {
                entries
                    .iter()
                    .fold((0u64, 0u64), |(sum, largest), resident| {
                        let bytes =
                            u64::from(resident.value().weight).saturating_mul(SLOT_UNIT_BYTES);
                        (sum.saturating_add(bytes), largest.max(bytes))
                    })
            });
        let main_capacity_bytes = if control.entries.is_some() {
            (self.inner.resident_bytes - self.inner.side_weight_bound) / SLOT_UNIT_BYTES
                * SLOT_UNIT_BYTES
        } else {
            0
        };
        ResidentBoundView {
            main_bytes,
            largest_main_slot_bytes,
            main_capacity_bytes,
            side_bytes: control
                .side
                .values()
                .fold(0, |sum, entry| sum.saturating_add(entry.weight_bytes())),
            side_quarter_bytes: self.inner.side_weight_bound,
            side_entry_bytes: control.side.get(&object_id).map(CacheEntry::weight_bytes),
        }
    }

    /// Wait for and take the per-replica one-at-a-time **cold-start permit**
    /// (the spec SDD's Mechanism lifecycle, "Cold-start permit"; SR-4.3 step
    /// 3) — that permit alone, without a plain permit.
    ///
    /// One permit per cache, and so per replica (`LayerStore` holds one cache;
    /// every clone shares it): at most one fenced cold start or audit fenced
    /// read runs on a replica at a time, so a replica joining with many large
    /// active layers queues their fenced reads instead of running them
    /// together, and at most one of them waits on the catch-up bounds the
    /// fenced read also takes. Only those reads take it — a fenced cold
    /// start's, through [`Self::acquire_fenced_cold_start_read_permits`]; the
    /// sampled audit's above the plain ceilings (`storage::structural_cache_audit`,
    /// SR-3.6), through [`Self::try_acquire_fenced_cold_start_read_permits`],
    /// only when free; and a test's raised read, through
    /// [`Self::acquire_raised_cold_start_read_permits`] — each before a plain
    /// permit (module docs, "Cold-start permits"); a plain read never takes
    /// it, and so never queues behind the reads waiting for it. Holding it
    /// alone is for a test that keeps fenced reads waiting.
    ///
    /// Released by RAII when the guard drops, including on a cancelled wait.
    /// Waits in FIFO order. A caller's deadline bounds the wait by dropping
    /// the future.
    pub async fn acquire_cold_start_permit(&self) -> ColdStartPermit {
        let permit = Arc::clone(&self.inner.cold_start_permit)
            .acquire_owned()
            .await
            .expect("the cold-start semaphore is never closed");
        ColdStartPermit { _permit: permit }
    }

    /// Wait for and take one of the per-replica **plain permits** for a plain
    /// whole-layer cold-start read: a cold start's or a reload's read at the
    /// shared atomic-read ceilings (module docs, "Cold-start permits").
    ///
    /// [`STRUCTURAL_COLD_START_READ_PERMITS`] permits per cache, and so per
    /// replica, shared with the fenced and raised reads, each of which holds
    /// one too. Hold the returned guard across exactly one read: a cold start
    /// hands it with the read's snapshot to its `load`, which drops it once the
    /// header checks pass and a structural `UpdateLayer`'s commit charge has
    /// taken the decoded entry (seam A) — or, when that charge refuses the
    /// entry (CR0), only after the admitted entry is put back, unless the
    /// read's decoded state and index build are over
    /// `plain_cold_start_hold_allowance_bytes`, when it drops at seam A and
    /// nothing is stored (`storage::structural_cache_cold_start`) — and gives it
    /// back at once on a `TOO_LARGE` answer, before the fenced read's
    /// permits are taken, so a cold start never holds one permit while it
    /// waits for another. It is released by
    /// RAII on every exit, including a cancelled wait or read and a panic.
    /// The sampled audit's plain read and the read side's probe do not take
    /// it.
    ///
    /// Waits in FIFO order. A caller's deadline bounds the wait by dropping
    /// the future.
    pub async fn acquire_plain_cold_start_read_permit(&self) -> ColdStartReadPermit {
        let plain = self.acquire_plain_permit().await;
        ColdStartReadPermit::new(&self.inner.reads_in_flight, ReadKind::Plain, None, plain)
    }

    /// Wait for and take the cache's permits of one fenced cold-start read —
    /// a cold start's or a reload's above the plain ceilings, or predicted
    /// over the plain byte ceiling: the one-at-a-time cold-start permit
    /// ([`Self::acquire_cold_start_permit`]) first, then a plain permit
    /// (module docs, "Cold-start permits"), counted as a `fenced` read in
    /// flight. The read then takes its catch-up bounds after both
    /// (`storage::structural_cache_fenced`).
    ///
    /// The returned guard is held across the fenced read, given back during
    /// each `RETRY_AFTER` sleep, and released by RAII on every exit,
    /// including a wait cancelled between the two permits and a panic.
    ///
    /// Each wait is FIFO. A caller's deadline bounds both by dropping the
    /// future.
    pub async fn acquire_fenced_cold_start_read_permits(&self) -> ColdStartReadPermit {
        let one_at_a_time = self.acquire_cold_start_permit().await;
        let plain = self.acquire_plain_permit().await;
        ColdStartReadPermit::new(
            &self.inner.reads_in_flight,
            ReadKind::Fenced,
            Some(one_at_a_time),
            plain,
        )
    }

    /// Take the permits of one fenced read **without waiting**: the
    /// one-at-a-time cold-start permit and a plain permit, both only if free
    /// now, else `None` with nothing held (a one-at-a-time permit already
    /// taken is given back). Never joins either FIFO queue, so a caller that
    /// must not delay others — the sampled audit — never gets ahead of a
    /// waiting cold start. Counted as a `fenced` read in flight, as
    /// [`Self::acquire_fenced_cold_start_read_permits`].
    pub fn try_acquire_fenced_cold_start_read_permits(&self) -> Option<ColdStartReadPermit> {
        let one_at_a_time = Arc::clone(&self.inner.cold_start_permit)
            .try_acquire_owned()
            .ok()
            .map(|permit| ColdStartPermit { _permit: permit })?;
        let plain = Arc::clone(&self.inner.plain_read_permits)
            .try_acquire_owned()
            .ok()?;
        Some(ColdStartReadPermit::new(
            &self.inner.reads_in_flight,
            ReadKind::Fenced,
            Some(one_at_a_time),
            plain,
        ))
    }

    /// Wait for and take the permits of one raised-ceiling whole-layer read.
    /// **No production caller:** cold starts and the sampled audit take the
    /// fenced read above the plain ceilings; tests call it to hold both
    /// permits. The one-at-a-time cold-start permit
    /// ([`Self::acquire_cold_start_permit`]) first, then a plain permit
    /// (module docs, "Cold-start permits"). The order is the same as a fenced
    /// cold start's, so at most one of the two kinds ever holds a plain
    /// permit, and a plain read waits behind at most that one.
    ///
    /// Take it only after the read's item early-out has passed, and hold the
    /// returned guard across exactly that one read; it releases both permits
    /// by RAII on every exit, including a wait cancelled between the two (the
    /// cold-start permit already taken is released then) and a panic.
    ///
    /// Each wait is FIFO. A caller's deadline bounds both by dropping the
    /// future.
    pub async fn acquire_raised_cold_start_read_permits(&self) -> ColdStartReadPermit {
        let raised = self.acquire_cold_start_permit().await;
        let plain = self.acquire_plain_permit().await;
        ColdStartReadPermit::new(
            &self.inner.reads_in_flight,
            ReadKind::Raised,
            Some(raised),
            plain,
        )
    }

    /// **Test seam.** The cold-start reads holding their permits now, plain,
    /// fenced and raised (module docs, "Cold-start permits"): each
    /// [`ColdStartReadPermit`] outstanding, the sampled audit's fenced read
    /// included (counted as fenced). A fenced or raised read still waiting for its plain permit,
    /// and a test holding [`Self::acquire_cold_start_permit`] alone, count in
    /// none. A fenced read counts while it holds its pair, so not during a
    /// `RETRY_AFTER` sleep, which gives the pair back.
    #[cfg(any(test, feature = "test-support"))]
    pub fn cold_start_reads_in_flight_for_test(&self) -> ColdStartReadsInFlight {
        let in_flight = &self.inner.reads_in_flight;
        ColdStartReadsInFlight {
            plain: in_flight.plain.load(Ordering::Acquire),
            fenced: in_flight.fenced.load(Ordering::Acquire),
            raised: in_flight.raised.load(Ordering::Acquire),
        }
    }

    /// **Test seam.** How many copies of a checked-out entry's state
    /// [`CheckedOut::plan_state`] has made since the cache was built. The
    /// cached structural route plans in place and makes none, so a call on
    /// it leaves this count unchanged. Every clone of the cache shares the
    /// count.
    #[cfg(any(test, feature = "test-support"))]
    pub fn plan_state_clones_for_test(&self) -> u64 {
        self.inner.plan_state_clones.load(Ordering::Acquire)
    }

    /// **Test seam.** How many plans this cache's checked-out entries have
    /// made in place ([`CheckedOut::plan_in_place`]) since the cache was
    /// built: one per plan the cached structural route runs.
    #[cfg(any(test, feature = "test-support"))]
    pub fn in_place_plans_for_test(&self) -> u64 {
        self.inner.in_place_plans.load(Ordering::Acquire)
    }

    /// **Test seam.** How many of those plans were rolled back: every exit of
    /// an attempt other than its commit.
    #[cfg(any(test, feature = "test-support"))]
    pub fn rollbacks_for_test(&self) -> u64 {
        self.inner.rollbacks.load(Ordering::Acquire)
    }

    /// **Test seam.** How many of those rollbacks rebuilt at least one table
    /// at its pre-plan length because it lacked the room to re-create the
    /// keys its plan removed (`domain::plan_journal`). Which tables each
    /// rebuilt is checked, by the shadow check, against the rebuilds its
    /// plan projected.
    #[cfg(any(test, feature = "test-support"))]
    pub fn rollback_rebuilds_for_test(&self) -> u64 {
        self.inner.rollback_rebuilds.load(Ordering::Acquire)
    }

    /// **Test seam.** Turn the shadow check off (or back on) for every plan
    /// this cache's entries make in place. On by default in every
    /// test-support build: each plan in place is shadowed by a plan of the
    /// same delta on a copy of the entry, and its mutations, its rollback
    /// and its committed successor must equal the copy's, with its indexes
    /// equal to a rebuild ([`CheckedOut::plan_in_place`]). It also gates the
    /// sampled audit's check of its copy's bucket bound against the true
    /// bucket counts, on the starting bound and after every charged record
    /// ([`CopyBucketBound::assert_holds_for_test`]). A measurement harness
    /// turns it off, since the copy is what the route removed and the
    /// bound check clones every table.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_shadow_check_for_test(&self, on: bool) {
        self.inner.shadow_check.store(on, Ordering::Release);
    }

    /// Whether the shadow check is on ([`Self::set_shadow_check_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    pub fn shadow_check_for_test(&self) -> bool {
        self.inner.shadow_check.load(Ordering::Acquire)
    }

    /// **Test seam.** Install `hook`, called by every cold start that loads
    /// an entry with the object id, after the snapshot is decoded (the cold
    /// start's seam B) — and its memory given back, unless the cold start's
    /// commit charge was refused at seam A, when the memory is kept until the
    /// entry is put back — and before [`Self::admit`] (seam C), so a test can
    /// observe what the cold start still holds there. `None` removes it.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_before_admit_hook_for_test(&self, hook: Option<BeforeAdmitHook>) {
        *self
            .inner
            .before_admit_hook
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = hook;
    }

    /// Run the before-admit test hook, if one is installed
    /// ([`Self::set_before_admit_hook_for_test`]); nothing in production.
    pub(crate) fn before_admit(&self, _object_id: u64) {
        #[cfg(any(test, feature = "test-support"))]
        {
            let hook = self
                .inner
                .before_admit_hook
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            if let Some(hook) = hook {
                hook(_object_id);
            }
        }
    }

    /// One plain permit, owned.
    async fn acquire_plain_permit(&self) -> OwnedSemaphorePermit {
        Arc::clone(&self.inner.plain_read_permits)
            .acquire_owned()
            .await
            .expect("the cold-start plain-permit semaphore is never closed")
    }

    /// A guard over `entry`, whose checkout was registered at `epoch`. Built
    /// outside the lock: a guard dropped while the lock is held would
    /// deadlock in its own `Drop`.
    fn guard(&self, entry: CacheEntry, epoch: u64) -> CheckedOut {
        CheckedOut {
            original: Some(entry),
            checkout_epoch: epoch,
            cache: self.clone(),
            mutating: false,
            advanced: false,
            pending: None,
        }
    }
}

/// Why [`StructuralCache::admit`] stored nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmitRefused {
    /// The object is uncacheable: [`UncacheableReason::OverWeight`] (recorded
    /// as a negative entry) or [`UncacheableReason::Disabled`] (recorded
    /// nowhere).
    Uncacheable(UncacheableReason),
    /// The object was invalidated after the cold start's ticket was taken —
    /// retired by a `DeleteLayer` or an overwrite, or dropped by an audit —
    /// so the read may predate it. Nothing was stored or recorded; the caller
    /// re-probes, which answers a retired object as gone or replaced.
    Invalidated,
}

/// A cold start's registration at the epoch it began
/// ([`StructuralCache::begin_cold_start`]), taken before its read and handed
/// to [`StructuralCache::admit`]. Dropping it unadmitted releases the
/// registration.
#[must_use = "a cold start's ticket is handed to `admit`; dropping it abandons the cold start"]
#[derive(Debug)]
pub struct ColdStartTicket {
    /// The epoch the ticket was registered at.
    epoch: u64,
    /// The cache it is registered with; `None` once `admit` took it over.
    cache: Option<StructuralCache>,
}

impl Drop for ColdStartTicket {
    /// Release the registration of a ticket `admit` did not take over.
    fn drop(&mut self) {
        if let Some(cache) = self.cache.take() {
            cache.inner.release_ticket(self.epoch);
        }
    }
}

/// The per-replica one-at-a-time cold-start permit
/// ([`StructuralCache::acquire_cold_start_permit`]). A fenced cold start and
/// the sampled audit's fenced read hold it inside their
/// [`ColdStartReadPermit`], beside a plain permit. Dropping it releases the
/// permit.
#[derive(Debug)]
pub struct ColdStartPermit {
    _permit: OwnedSemaphorePermit,
}

/// The permits one whole-layer cold-start read holds across that read: a
/// plain permit, and for a fenced or raised read the
/// one-at-a-time cold-start permit taken before it
/// ([`StructuralCache::acquire_plain_cold_start_read_permit`],
/// [`StructuralCache::acquire_fenced_cold_start_read_permits`],
/// [`StructuralCache::acquire_raised_cold_start_read_permits`]). Dropping it
/// releases them.
#[must_use = "a cold-start read permit is held across the read; dropping it at once releases it"]
#[derive(Debug)]
pub struct ColdStartReadPermit {
    /// The read's in-flight count, decremented in `Drop` — before the
    /// permits below release, so the count never exceeds what they allow.
    in_flight: Arc<ReadsInFlight>,
    /// Which count it is in.
    kind: ReadKind,
    /// The one-at-a-time cold-start permit, for a fenced or raised read;
    /// `None` for a plain one.
    _one_at_a_time: Option<ColdStartPermit>,
    /// The plain permit.
    _plain: OwnedSemaphorePermit,
}

impl ColdStartReadPermit {
    /// The guard over permits already taken, counted in flight from now.
    fn new(
        in_flight: &Arc<ReadsInFlight>,
        kind: ReadKind,
        one_at_a_time: Option<ColdStartPermit>,
        plain: OwnedSemaphorePermit,
    ) -> Self {
        in_flight.counter(kind).fetch_add(1, Ordering::AcqRel);
        Self {
            in_flight: Arc::clone(in_flight),
            kind,
            _one_at_a_time: one_at_a_time,
            _plain: plain,
        }
    }
}

impl Drop for ColdStartReadPermit {
    fn drop(&mut self) {
        self.in_flight
            .counter(self.kind)
            .fetch_sub(1, Ordering::AcqRel);
    }
}

/// Which kind of cold-start read a [`ColdStartReadPermit`] is counted as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadKind {
    /// A plain read: a plain permit only.
    Plain,
    /// A fenced cold start: the one-at-a-time permit and a plain permit.
    Fenced,
    /// A raised-ceiling read (no production caller; tests only): the
    /// one-at-a-time permit and a plain permit.
    Raised,
}

/// The cold-start reads holding their permits, by kind
/// ([`CacheInner::reads_in_flight`]).
#[derive(Debug, Default)]
struct ReadsInFlight {
    plain: AtomicUsize,
    fenced: AtomicUsize,
    raised: AtomicUsize,
}

impl ReadsInFlight {
    /// The counter of `kind`'s reads.
    fn counter(&self, kind: ReadKind) -> &AtomicUsize {
        match kind {
            ReadKind::Plain => &self.plain,
            ReadKind::Fenced => &self.fenced,
            ReadKind::Raised => &self.raised,
        }
    }
}

/// **Test seam.** The cold-start reads holding their permits at one instant
/// ([`StructuralCache::cold_start_reads_in_flight_for_test`]).
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColdStartReadsInFlight {
    /// Plain reads: a plain permit only.
    pub plain: usize,
    /// Fenced cold starts holding their pair: the one-at-a-time cold-start
    /// permit and a plain permit.
    pub fenced: usize,
    /// Raised-ceiling reads (no production caller; tests only):
    /// the one-at-a-time cold-start permit and a plain permit.
    pub raised: usize,
}

/// **Test seam.** A hook a cold start calls before [`StructuralCache::admit`]
/// ([`StructuralCache::set_before_admit_hook_for_test`]), with the object id.
#[cfg(any(test, feature = "test-support"))]
pub type BeforeAdmitHook = Arc<dyn Fn(u64) + Send + Sync>;

impl fmt::Debug for StructuralCache {
    /// The configuration only: printing entries would take the cache's lock
    /// and walk every state.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StructuralCache")
            .field("budget_bytes", &self.inner.budget_bytes)
            .field("idle_ttl", &self.inner.idle_ttl)
            .field("negative_ttl", &self.inner.negative_ttl)
            .finish_non_exhaustive()
    }
}

impl CacheInner {
    /// The control lock. A poisoned lock is taken anyway: every critical
    /// section leaves the maps consistent between its own steps, and a cache
    /// that refused all further use after one panic would cost a cold start
    /// per call for the life of the process.
    fn lock(&self) -> MutexGuard<'_, Control> {
        self.control.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Settle a checkout taken at `checkout_epoch`, putting `entry` back
    /// unless the epoch check or the per-entry cap refuses it. Returns how it
    /// settled ([`PutBack`]), and the entries the settlement removed —
    /// `entry` itself when it was refused — for the caller to drop outside
    /// the lock. An over-cap refusal records the object's `OverWeight`
    /// negative entry.
    fn settle_put_back(&self, checkout_epoch: u64, entry: CacheEntry) -> (PutBack, Freed) {
        let mut freed = Freed::new();
        let mut control = self.lock();
        let object_id = entry.object_id;
        let refused = control
            .invalidations
            .get(&object_id)
            .is_some_and(|&invalidated_at| invalidated_at > checkout_epoch);
        control.release_checkout(checkout_epoch);
        let settled = if refused {
            freed.push(entry);
            PutBack::Invalidated
        } else if entry.state_weight_bytes > self.max_entry_bytes {
            let reason = UncacheableReason::OverWeight(entry.state_weight_bytes);
            control.record_negative(object_id, reason, self.negative_ttl, &mut freed);
            freed.push(entry);
            PutBack::OverCap
        } else {
            control.store(
                entry,
                self.idle_ttl,
                self.side_weight_bound,
                self.resident_bytes,
                &mut freed,
            );
            PutBack::Stored
        };
        control.prune_invalidations();
        (settled, freed)
    }

    /// Release a cold-start ticket's registration at `epoch` that no guard
    /// took over.
    fn release_ticket(&self, epoch: u64) {
        let mut control = self.lock();
        control.release_checkout(epoch);
        control.prune_invalidations();
    }

    /// Settle a checkout taken at `checkout_epoch` by putting nothing back,
    /// and invalidate `object_id`. Returns the entries the invalidation
    /// removed, for the caller to drop outside the lock.
    fn settle_discard(&self, checkout_epoch: u64, object_id: u64) -> Freed {
        let mut freed = Freed::new();
        let mut control = self.lock();
        control.release_checkout(checkout_epoch);
        control.invalidate(object_id, &mut freed);
        control.prune_invalidations();
        freed
    }
}

impl Control {
    /// Register a checkout at the current epoch and return that epoch.
    fn register_checkout(&mut self) -> u64 {
        *self.outstanding.entry(self.epoch).or_insert(0) += 1;
        self.epoch
    }

    /// Release a checkout registered at `epoch`. Tolerates a guard that was
    /// never registered.
    fn release_checkout(&mut self, epoch: u64) {
        if let Some(count) = self.outstanding.get_mut(&epoch) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.outstanding.remove(&epoch);
            }
        }
    }

    /// Drop the invalidation records no outstanding checkout can be refused
    /// by: a record at epoch `e` refuses only a checkout taken before `e`,
    /// and every future checkout is taken at or after the current epoch.
    fn prune_invalidations(&mut self) {
        match self.outstanding.keys().next().copied() {
            None => self.invalidations.clear(),
            Some(oldest) => self.invalidations.retain(|_, &mut at| at > oldest),
        }
    }

    /// Take `object_id`'s entry out — from the side store, else from its
    /// resident mini-moka slot — and register the checkout. A resident key
    /// whose slot is empty (its entry is checked out) has no entry to take.
    /// An expired side-store entry goes to `freed`.
    fn take(
        &mut self,
        object_id: u64,
        idle_ttl: Duration,
        freed: &mut Freed,
    ) -> Option<(CacheEntry, u64)> {
        let entries = self.entries.as_ref()?;
        let now = Instant::now();
        let taken = match self.side.remove(&object_id) {
            Some(entry) if now.saturating_duration_since(entry.last_used) < idle_ttl => {
                // A read of the absent key raises its frequency in mini-moka's
                // sketch, so a later put-back has a better chance of
                // admission into the main cache.
                let _ = entries.get(&object_id);
                Some(entry)
            }
            // Expired in the side store: dropped, like an idle mini-moka
            // entry.
            Some(expired) => {
                freed.push(expired);
                None
            }
            None => take_resident(entries, object_id),
        };
        let mut entry = taken?;
        entry.last_used = now;
        let epoch = self.register_checkout();
        Some((entry, epoch))
    }

    /// Store `entry`: refill its resident slot (an update of the key), or
    /// admit it as a new key and, if mini-moka refuses it, keep it in the
    /// side store — unless it is too heavy to hold (below). Then, on every
    /// path, bring the cache within the resident budget `resident_bytes`
    /// ([`Self::hold_resident_bound`]). A side-store copy it replaces,
    /// side-store entries it evicts, an entry too heavy to hold and the
    /// main-cache entries the bound evicts go to `freed`.
    ///
    /// **An entry too heavy to hold is evicted at once.** First, before
    /// anything is placed: an entry whose whole weight, state and plan
    /// indexes, is over [`held_entry_bound`] — the resident budget, or the
    /// most mini-moka's `u32` weigher charges in full when that is lower —
    /// is not placed. Its object's side-store copy and resident key
    /// go too (the resident slot emptied into `freed` before the key is
    /// invalidated), and the entry itself goes to `freed`. Its state is
    /// within the per-entry cap, so only an index outgrown since its build
    /// takes it there: a name-index vector keeps the largest capacity it ever
    /// had, through rollbacks and committed deletes alike
    /// (`PlanIndexes::name_vector_room`). No epoch advances and no negative
    /// entry or verdict is recorded, so the put-back still answers
    /// [`PutBack::Stored`] (evicted in the same critical section); the next
    /// lookup misses and cold-starts, its indexes built afresh, within the
    /// bound at every budget the configuration accepts, 1 MiB up (a fresh
    /// entry's indexes weigh at most 1.75 × its state by the weigher's
    /// per-row arithmetic, plus a fixed overhead, `size_of::<PlanIndexes>()`
    /// and the small-table remainder, which alone exceeds the resident
    /// budget only below about a kilobyte, a budget reachable only by
    /// building the cache directly), so this rule never evicts that rebuild
    /// as built there: one cold start per eviction, capacity, never
    /// cacheability. A plan whose own growth takes the entry over the bound
    /// has it evicted again at its put-back.
    /// Because only a store raises any weight and this rule runs on every
    /// store, every entry the cache holds weighs at most the bound, which is
    /// what makes [`Self::hold_resident_bound`] finish within the resident
    /// budget. Placing the check first means neither the main cache nor the
    /// side store's other entries are evicted on such an entry's account.
    /// Not counted on a metric and not logged, like the resident bound's own
    /// evictions.
    fn store(
        &mut self,
        entry: CacheEntry,
        idle_ttl: Duration,
        side_weight_bound: u64,
        resident_bytes: u64,
        freed: &mut Freed,
    ) {
        if self.entries.is_some() && entry.weight_bytes() > held_entry_bound(resident_bytes) {
            // `remove_entry` reads the key (`take_resident`), and mini-moka
            // counts a read in its frequency sketch, a miss included.
            // `place` avoids such reads because they would count on every
            // put-back. This path is rare and the read adds one count, which
            // only makes the evicted key's re-admission marginally easier,
            // so it is not worth a guard.
            self.remove_entry(entry.object_id, freed);
            freed.push(entry);
            #[cfg(any(test, feature = "test-support"))]
            {
                self.oversized_evictions += 1;
            }
        } else {
            self.place(entry, idle_ttl, side_weight_bound, freed);
        }
        self.hold_resident_bound(resident_bytes, freed);
    }

    /// [`Self::store`]'s placement, before the resident bound: refill
    /// `entry`'s resident slot, or admit it as a new key, or keep it in the
    /// side store when mini-moka refuses it.
    fn place(
        &mut self,
        entry: CacheEntry,
        idle_ttl: Duration,
        side_weight_bound: u64,
        freed: &mut Freed,
    ) {
        let Some(entries) = self.entries.as_ref() else {
            freed.push(entry);
            return;
        };
        let object_id = entry.object_id;
        let slot = Slot {
            entry: Arc::new(Mutex::new(None)),
            weight: slot_weight(entry.weight_bytes()),
        };
        // A resident key's old slot is replaced whole, not read first: a
        // read would count towards the key's frequency in mini-moka's sketch
        // on every put-back. It is empty — the put-back follows a checkout
        // that emptied it — unless two guards for the object overlapped and
        // the other settled first (the later put-back wins, module docs),
        // which the per-object `structural` permit rules out in production;
        // only then does mini-moka drop an entry with it, under this lock.
        let resident = entries.contains_key(&object_id);
        *slot.entry.lock().unwrap_or_else(PoisonError::into_inner) = Some(entry);
        entries.insert(object_id, slot.clone());
        freed.extend(self.side.remove(&object_id));
        if resident {
            return;
        }
        // A new key: mini-moka decides admission when it applies the write.
        entries.sync();
        if entries.contains_key(&object_id) {
            return;
        }
        let refused = slot
            .entry
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(entry) = refused {
            self.keep_in_side_store(entry, idle_ttl, side_weight_bound, freed);
        }
    }

    /// Keep an entry mini-moka refused, evicting the side store's least
    /// recently used entries into `freed` until it fits both bounds. An
    /// entry alone over the weight bound is still kept, alone: its state is
    /// within the per-entry cap, so only an index heavier than φ of its state
    /// takes it past the bound, and [`Self::store`] has already evicted any
    /// entry over the resident budget ([`held_entry_bound`]), so it is
    /// within that. The side store is then over its quarter, and the main
    /// cache makes room for the excess ([`Self::hold_resident_bound`], which
    /// [`Self::store`] runs next), so the total stays within the resident
    /// budget.
    fn keep_in_side_store(
        &mut self,
        entry: CacheEntry,
        idle_ttl: Duration,
        weight_bound: u64,
        freed: &mut Freed,
    ) {
        self.expire_side(idle_ttl, freed);
        let mut total: u64 = self
            .side
            .values()
            .fold(0, |sum, held| sum.saturating_add(held.weight_bytes()));
        while !self.side.is_empty()
            && (self.side.len() >= SIDE_STORE_MAX_ENTRIES
                || total.saturating_add(entry.weight_bytes()) > weight_bound)
        {
            let Some(oldest) = self
                .side
                .iter()
                .min_by_key(|(_, held)| held.last_used)
                .map(|(&id, _)| id)
            else {
                break;
            };
            if let Some(evicted) = self.side.remove(&oldest) {
                total = total.saturating_sub(evicted.weight_bytes());
                freed.push(evicted);
            }
        }
        // `place` removed any copy of this object first, so this replaces
        // nothing; a replaced entry would still be freed outside the lock.
        freed.extend(self.side.insert(entry.object_id, entry));
    }

    /// **The resident bound** (module docs): bring mini-moka's applied
    /// weight plus the side store's within `resident_bytes`, the resident
    /// budget, whatever an entry's indexes weigh. Run at the end of every
    /// [`Self::store`], inside its critical section; the entries of its
    /// victims go to `freed`.
    ///
    /// Each round applies mini-moka's pending writes (`sync`), reads its
    /// weighted size and adds the side store's. Within the budget, it
    /// returns. Otherwise it collects mini-moka's visible keys — `iter`
    /// touches neither the popularity sketch nor the idle timers, and holds
    /// read locks, so the keys are collected before any is evicted — orders
    /// them, filled slots by their entries' last use, least recent first,
    /// then keys still checked out (their slots are empty), and evicts in
    /// that order until its own count, less each victim's slot weight, is
    /// within the budget: the slot's entry into `freed`, then the key from
    /// mini-moka. The next round's `sync` applies those removals and checks
    /// again.
    ///
    /// **Why it is a bound, not an eventual one.** Every mini-moka call goes
    /// through [`Control::entries`], behind the cache's one lock, and
    /// mini-moka 0.10 has no background thread: its housekeeping, its own
    /// evictions included, runs inline in the caller's insert, read or
    /// invalidation. So nothing changes mini-moka's weight outside a critical
    /// section, `sync` here drains its pending writes completely, and the
    /// weight read after it is exact. Only a store raises either weight —
    /// a resident key's update, which mini-moka never refuses, a new key's
    /// admission, or the side store's keep — and every other operation only
    /// lowers them. So the bound holds between any two cache operations, and
    /// a store ends with no pending write, a grown resident entry's included.
    ///
    /// **Checked-out keys come last.** Their memory is held whole in their
    /// calls' upload charges as well (SDD §5.4), so the weight counts them
    /// twice; evicting one frees nothing now but moves its accounting wholly
    /// to that charge, and its put-back is then a new admission, which runs
    /// this step again. The just-stored entry is itself a victim only when,
    /// after every filled slot less recently used than it has gone, it, any
    /// slot used since, the checked-out keys' last weights and the side
    /// store are still over the resident budget. Its last use is the
    /// put-back's instant on an explicit put-back; the guard's `Drop` keeps
    /// the checkout's instant unless a repair advanced the entry, which it
    /// re-weighs and stamps as an explicit put-back does
    /// ([`CacheEntry::last_used`]).
    ///
    /// **It always finishes.** Every entry the cache holds weighs at most the
    /// resident budget: [`Self::store`] evicts a heavier one at its own store,
    /// before this step runs ([`held_entry_bound`]), and only a store raises
    /// any weight. A side store over its quarter holds exactly one entry
    /// (`keep_in_side_store` keeps an over-quarter entry alone), so it weighs
    /// at most the resident budget, and emptying the main cache always
    /// suffices. A per-row bound on the indexes would not do: a fresh
    /// layout's indexes weigh at most 1.75 × its state by the weigher's
    /// per-row arithmetic, but a name-index vector keeps the largest capacity
    /// it ever had, through rollbacks and committed deletes, so an entry's
    /// indexes can outgrow its state without limit. Should a round find no
    /// visible key and its next `sync` lower nothing, it stops; a debug
    /// assertion checks the bound then holds, which only a defect in the
    /// store's rule could break.
    ///
    /// Evicting changes no cap, no verdict and no epoch, and records no
    /// negative entry: a victim's next lookup misses and cold-starts, which
    /// costs capacity, never cacheability. Not counted on a metric, as
    /// mini-moka's own capacity evictions are not: it shows on the weight
    /// and entry gauges and on the cold lookups.
    fn hold_resident_bound(&mut self, resident_bytes: u64, freed: &mut Freed) {
        let Some(entries) = self.entries.as_ref() else {
            return;
        };
        // Constant across the step: it evicts from the main cache only.
        let side = self
            .side
            .values()
            .fold(0u64, |sum, held| sum.saturating_add(held.weight_bytes()));
        // The applied weight at a round that found no visible key.
        let mut stalled_at: Option<u64> = None;
        loop {
            entries.sync();
            let applied = entries.weighted_size().saturating_mul(SLOT_UNIT_BYTES);
            let mut total = applied.saturating_add(side);
            if total <= resident_bytes {
                return;
            }
            if stalled_at.is_some_and(|stalled| applied >= stalled) {
                // A defect tripwire, never a case to handle: the store's rule
                // keeps every held entry within the resident budget
                // (`Self::store`, `held_entry_bound`), so with no main-cache
                // key left the side store's lone over-quarter entry fits.
                debug_assert!(
                    total <= resident_bytes,
                    "the structural cache could not bring its weight ({total} B; side store \
                     {side} B) within its resident budget ({resident_bytes} B) with no main-cache \
                     key left to evict: an entry over the store's held-entry bound was placed"
                );
                return;
            }
            let mut victims: Vec<(Option<Instant>, u64, Slot)> = entries
                .iter()
                .map(|resident| {
                    let slot = resident.value().clone();
                    let last_used = slot
                        .entry
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .as_ref()
                        .map(|entry| entry.last_used);
                    (last_used, *resident.key(), slot)
                })
                .collect();
            if victims.is_empty() {
                stalled_at = Some(applied);
                continue;
            }
            // Filled slots first, least recently used first; then the keys
            // still checked out.
            victims.sort_by_key(|(last_used, ..)| (last_used.is_none(), *last_used));
            for (_, object_id, slot) in victims {
                if total <= resident_bytes {
                    break;
                }
                let taken = slot
                    .entry
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take();
                #[cfg(any(test, feature = "test-support"))]
                {
                    self.resident_bound_evictions[usize::from(taken.is_none())] += 1;
                }
                freed.extend(taken);
                entries.invalidate(&object_id);
                total =
                    total.saturating_sub(u64::from(slot.weight).saturating_mul(SLOT_UNIT_BYTES));
            }
        }
    }

    /// Move side-store entries idle for `idle_ttl` or longer into `freed`.
    fn expire_side(&mut self, idle_ttl: Duration, freed: &mut Freed) {
        let now = Instant::now();
        freed.extend(
            self.side
                .extract_if(|_, entry| now.saturating_duration_since(entry.last_used) >= idle_ttl)
                .map(|(_, entry)| entry),
        );
    }

    /// `object_id`'s negative-entry reason if one is live; an expired one is
    /// removed. Never extends the entry's life.
    fn live_negative(&mut self, object_id: u64, ttl: Duration) -> Option<UncacheableReason> {
        let negative = self.negatives.get(&object_id)?;
        if negative.created_at.elapsed() < ttl {
            return Some(negative.reason);
        }
        self.negatives.remove(&object_id);
        None
    }

    /// Record a negative entry for `object_id`, created now and unreported
    /// (module docs, "Negative entries"), replacing any earlier one, and
    /// remove any entry the object holds into `freed`. Expired negative
    /// entries are
    /// pruned on the way, so the map holds only the objects found uncacheable
    /// within one TTL.
    fn record_negative(
        &mut self,
        object_id: u64,
        reason: UncacheableReason,
        ttl: Duration,
        freed: &mut Freed,
    ) {
        self.negatives
            .retain(|_, negative| negative.created_at.elapsed() < ttl);
        self.negatives.insert(
            object_id,
            NegativeEntry {
                reason,
                created_at: Instant::now(),
                reported: false,
            },
        );
        self.remove_entry(object_id, freed);
    }

    /// Advance the epoch, record it as `object_id`'s last invalidation, and
    /// drop the object's negative entry and remove its entry into `freed`.
    fn invalidate(&mut self, object_id: u64, freed: &mut Freed) {
        self.epoch += 1;
        self.invalidations.insert(object_id, self.epoch);
        self.negatives.remove(&object_id);
        self.remove_entry(object_id, freed);
    }

    /// Remove `object_id`'s entry from the side store and from mini-moka into
    /// `freed`. The resident slot is emptied before its key is invalidated,
    /// so the value mini-moka drops holds no state (see
    /// [`CacheInner::control`]).
    fn remove_entry(&mut self, object_id: u64, freed: &mut Freed) {
        freed.extend(self.side.remove(&object_id));
        if let Some(entries) = self.entries.as_ref() {
            freed.extend(take_resident(entries, object_id));
            entries.invalidate(&object_id);
        }
    }
}

/// Take the entry out of `object_id`'s resident mini-moka slot, leaving the
/// slot empty; `None` when the key is not resident or its entry is checked
/// out. A read of the key, so it counts towards the key's frequency in
/// mini-moka's sketch like any other lookup.
fn take_resident(entries: &MokaCache<u64, Slot>, object_id: u64) -> Option<CacheEntry> {
    entries.get(&object_id).and_then(|slot| {
        slot.entry
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    })
}

/// How an explicit put-back ([`CheckedOut::put_back`],
/// [`CheckedOut::put_back_held`]) settled (SR-3.3, SR-3.4; C3 amendment,
/// 2026-10-04, decider-approved). A refusal is not an error: the entry is
/// dropped, and only [`Self::OverCap`] wants counting by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PutBack {
    /// The put-back was accepted and the entry placed: in mini-moka, or in
    /// the side store — or, when it alone weighs more than any held entry
    /// may, evicted at once (below). A side-store placement stays: the
    /// resident bound (`Control::hold_resident_bound`) evicts main-cache keys
    /// only. An entry placed in mini-moka is not promised to stay resident:
    /// the resident bound, in the same critical section, evicts it too when,
    /// after every filled slot less recently used than it has gone, its
    /// weight, any slot used since, the checked-out keys' last weights and
    /// the side store are still over the resident budget. Its last use is
    /// the put-back's instant on an explicit put-back; the guard's `Drop`
    /// keeps the checkout's instant unless a repair advanced the entry, which
    /// it re-weighs and stamps as an explicit put-back does
    /// ([`CacheEntry::last_used`]), so a slot checked out and put back after
    /// an unadvanced entry's checkout ranks after it. An entry that alone
    /// weighs more than any held entry may (`held_entry_bound`: the resident
    /// budget at most) is not placed at all: the store evicts it, and its
    /// object's resident key, at once (`Control::store`). Either way its
    /// next lookup then misses — capacity, never cacheability.
    Stored,
    /// Refused by the epoch check: the object was invalidated after the
    /// checkout (a retirement, an audit's `diverged`, another guard's
    /// `discard`).
    Invalidated,
    /// Refused by the per-entry cap: the entry outgrew it while checked out
    /// (a commit's successor, or a repaired held entry), and the object now
    /// has an [`UncacheableReason::OverWeight`] negative entry. The caller
    /// counts it on `live_edit_structural_cache_reloads_total{reason="uncacheable"}`
    /// and reports the verdict on `live_edit_structural_cache_uncacheable_total`,
    /// taking it with [`StructuralCache::take_unreported`].
    OverCap,
}

impl PutBack {
    /// Whether the put-back was accepted and the entry placed
    /// ([`Self::Stored`]); the store can still have evicted it in the same
    /// critical section, for its weight or by the resident bound.
    pub fn is_stored(self) -> bool {
        self == Self::Stored
    }
}

/// A checked-out cache entry: the drop guard a structural call holds while
/// the entry is out of the cache (SR-3.4).
///
/// It owns the original [`CacheEntry`], the epoch the checkout was taken at,
/// the pending plan's journal while a plan in place is unsettled, and a
/// handle to the cache. It settles exactly once:
/// [`put_back_held`](Self::put_back_held) or [`discard`](Self::discard)
/// consume it (or, in test and test-support builds, `put_back` with a given
/// entry), and if none runs — the call was dropped on its deadline or on
/// cancellation, or a panic unwound through it — its `Drop` puts the held
/// entry back, subject to the epoch check, after rolling a pending plan
/// back. `Drop` discards instead when a panic unwound out of a repair step, a
/// plan in place or a rollback, when the pending plan's rollback fails or
/// panics, or when a panic unwinds through the guard with a plan pending.
#[derive(Debug)]
pub struct CheckedOut {
    /// The taken original entry. `Some` from checkout until a consuming
    /// method (`put_back_held`, `discard`, or the test-support `put_back`)
    /// takes it, so [`Self::entry`] always sees it; only
    /// `Drop` can observe `None`, after a consuming method took it.
    original: Option<CacheEntry>,
    /// The cache epoch at checkout, compared at put-back (SR-3.4).
    checkout_epoch: u64,
    /// The cache the entry returns to.
    cache: StructuralCache,
    /// Set while [`Self::mutate`] runs its closure, and while a plan in
    /// place, its rollback or its commit step runs. Still set at `Drop` only
    /// when one of those unwound, leaving the entry possibly half-replayed
    /// or half-planned: `Drop` then discards instead of putting back.
    mutating: bool,
    /// Set, and never cleared, once [`Self::mutate`] or
    /// [`Self::plan_in_place`] has run: the held entry may have grown or
    /// shrunk since it was weighed, so `Drop` compacts and re-weighs it
    /// before putting it back.
    advanced: bool,
    /// The plan made in place on the held entry, from [`Self::plan_in_place`]
    /// until it is committed ([`Self::advance_committed`]) or rolled back —
    /// by [`Self::put_back_held`], or by `Drop` on any other exit.
    pending: Option<PendingPlan>,
}

/// A plan made in place on a checked-out entry and not yet settled: its
/// pre-image journal and what it touched.
#[derive(Debug)]
struct PendingPlan {
    journal: PlanJournal,
    touches: PlanTouches,
    /// The same plan on a copy, to check this one against
    /// ([`StructuralCache::set_shadow_check_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    shadow: Option<Box<ShadowPlan>>,
    /// Whether the plan overflowed its budget: its rollback may then rebuild
    /// a table it never projected (the shadow check's exemption).
    #[cfg(any(test, feature = "test-support"))]
    overflowed: bool,
}

/// The test-support shadow of a plan made in place: the entry before it (a
/// state-only copy), and the same delta's plan on that copy once it ran.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug)]
struct ShadowPlan {
    pre: CacheEntry,
    planned: Option<CompactState>,
}

impl CheckedOut {
    /// The held entry: the original as it was checked out, as the stream
    /// repair (SR-4.1) has advanced it in place, and — while a plan is
    /// pending ([`Self::plan_in_place`]) — as that plan left it.
    pub fn entry(&self) -> &CacheEntry {
        self.original
            .as_ref()
            .expect("a CheckedOut holds its entry until put_back or discard consumes it")
    }

    /// **Test support.** A clone of the entry's state, sized for the delta's
    /// own `inserts` so no table of the copy grows when they are planned on
    /// it: each table whose length plus its inserts would pass its capacity
    /// is cloned at that capacity (`domain::types::clone_with_room`), every
    /// other one is a plain clone. A test's oracle plans on it; the cached
    /// route plans in place instead ([`Self::plan_in_place`]). Counted
    /// ([`StructuralCache::plan_state_clones_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    pub fn plan_state(&self, inserts: PlannedInserts) -> CompactState {
        self.cache
            .inner
            .plan_state_clones
            .fetch_add(1, Ordering::AcqRel);
        let state = &self.entry().state;
        let room = |n: u64| usize::try_from(n).unwrap_or(usize::MAX);
        pass_timing::time(Pass::PlanClone, || CompactState {
            object_id: state.object_id,
            sequence_num: state.sequence_num,
            first_version: state.first_version,
            source_format: state.source_format.clone(),
            next_set_order: state.next_set_order,
            sections: clone_with_room(&state.sections, room(inserts.sections)),
            children: clone_with_room(&state.children, room(inserts.children_list_keys)),
            fields: state
                .fields
                .clone_with_room(room(inserts.set_fields), room(inserts.set_fields)),
            time_samples: state
                .time_samples
                .clone_with_room(room(inserts.time_samples)),
        })
    }

    /// Plan `delta` on the held entry itself, in place (SR-3.5): the
    /// structural route's one plan per attempt, with no copy of the state.
    /// [`domain::delta::apply_delta_bounded_indexed`] runs on the entry's
    /// state with its plan indexes ([`CacheEntry::plan`], built here first
    /// if the entry has none) and a fresh pre-image journal, which records
    /// every key the plan touches on its first touch.
    ///
    /// The plan is then **pending**, and the entry is the planned state
    /// until the attempt settles:
    ///
    /// - committed: [`Self::advance_committed`] keeps the planned state as
    ///   the entry at the commit's cut, then [`Self::put_back_held`] stores
    ///   it;
    /// - every other outcome — a delta `Err`, an overflow or refused charge
    ///   (`NeedsCharge`), the size gate, a ceiling, `STALE`, `FENCED`, a
    ///   token already landed or recorded, overtaken, a transport or in-doubt
    ///   error: [`Self::put_back_held`] rolls the plan back first, so the
    ///   entry that goes back is the pre-plan cut exactly, never a
    ///   half-planned one;
    /// - `CACHE_DIVERGED` and the gone-object refusals: [`Self::discard`];
    /// - a dropped call: `Drop` rolls back, then puts back; a panic inside
    ///   the plan or the rollback, or one unwinding through the guard while a
    ///   plan is pending, discards instead.
    ///
    /// One plan per checkout.
    ///
    /// [`domain::delta::apply_delta_bounded_indexed`]: crate::domain::delta::apply_delta_bounded_indexed
    pub fn plan_in_place(
        &mut self,
        delta: Option<oodf_ffi::OwnedDelta>,
        allowance: MutationAllowance,
        budget: &mut dyn RewriteBudget,
        stats: &mut ApplyDeltaStats,
    ) -> Result<BoundedPlan, DeltaError> {
        assert!(
            self.pending.is_none(),
            "one plan per checkout: the pending plan must settle first"
        );
        #[cfg(any(test, feature = "test-support"))]
        let shadow = self.cache.shadow_check_for_test().then(|| {
            Box::new(ShadowPlan {
                pre: self.entry().copy_without_plan(),
                planned: None,
            })
        });
        #[cfg(any(test, feature = "test-support"))]
        let oracle_delta = shadow.as_ref().map(|_| delta.clone());
        #[cfg(any(test, feature = "test-support"))]
        self.cache
            .inner
            .in_place_plans
            .fetch_add(1, Ordering::AcqRel);

        let entry = self
            .original
            .as_mut()
            .expect("a CheckedOut holds its entry until put_back or discard consumes it");
        let CacheEntry { state, plan, .. } = entry;
        let indexes = plan.get_or_insert_with(|| PlanIndexes::build(state));
        let mut journal = PlanJournal::new(state, indexes);
        let mut touches = PlanTouches::default();
        self.mutating = true;
        self.advanced = true;
        let result = apply_delta_bounded_indexed(
            state,
            delta,
            allowance,
            budget,
            stats,
            indexes,
            &mut touches,
            Some(&mut journal),
        );
        self.mutating = false;
        self.pending = Some(PendingPlan {
            journal,
            touches,
            #[cfg(any(test, feature = "test-support"))]
            shadow,
            #[cfg(any(test, feature = "test-support"))]
            overflowed: matches!(result, Ok(BoundedPlan::Overflow { .. })),
        });
        #[cfg(any(test, feature = "test-support"))]
        if let Some(delta) = oracle_delta {
            self.shadow_check_plan(delta, &result);
            assert_weighed_at_or_above_its_buckets(self.entry(), "a plan in place");
        }
        result
    }

    /// The test-support shadow check of a plan just made in place: the same
    /// delta planned on the copy taken before it, unbounded, must emit the
    /// same mutations and leave the same state, when the plan in place ran
    /// within its allowance; and must fail with the same error when the plan
    /// in place failed, unless the failure was the failpoint's injected one
    /// (`domain::delta::failpoint`). An overflow is not compared: the copy's
    /// plan is unbounded. Whatever the plan's outcome, the planned entry is
    /// then checked to weigh at or above the bytes its tables' buckets hold
    /// (`assert_weighed_at_or_above_its_buckets`), where CR2 reads it.
    #[cfg(any(test, feature = "test-support"))]
    fn shadow_check_plan(
        &mut self,
        delta: Option<oodf_ffi::OwnedDelta>,
        result: &Result<BoundedPlan, DeltaError>,
    ) {
        let Some(shadow) = self.pending.as_mut().and_then(|p| p.shadow.as_mut()) else {
            return;
        };
        let mutations = match result {
            Ok(BoundedPlan::Within { mutations, .. }) => mutations,
            Ok(BoundedPlan::Overflow { .. }) => return,
            Err(DeltaError::InconsistentState { operation, .. })
                if *operation == crate::domain::delta::failpoint::INJECTED =>
            {
                return;
            }
            Err(ours) => {
                let mut planned = shadow.pre.state.clone();
                let oracle = crate::domain::delta::failpoint::paused(|| {
                    crate::domain::delta::apply_delta_bounded(
                        &mut planned,
                        delta,
                        MutationAllowance::UNBOUNDED,
                        &mut |_: RewriteCount| None::<u64>,
                        &mut ApplyDeltaStats::default(),
                    )
                });
                match oracle {
                    Err(theirs) => assert_eq!(
                        ours.to_string(),
                        theirs.to_string(),
                        "shadow check: the plan in place failed otherwise than the same plan on a copy"
                    ),
                    Ok(theirs) => panic!(
                        "shadow check: the plan in place failed ({ours}) where the same plan on a copy ended {theirs:?}"
                    ),
                }
                return;
            }
        };
        let mut planned = shadow.pre.state.clone();
        let unbounded = MutationAllowance::UNBOUNDED;
        let oracle = crate::domain::delta::failpoint::paused(|| {
            crate::domain::delta::apply_delta_bounded(
                &mut planned,
                delta,
                unbounded,
                &mut |_: RewriteCount| None::<u64>,
                &mut ApplyDeltaStats::default(),
            )
        });
        match oracle {
            Ok(BoundedPlan::Within {
                mutations: expected,
                ..
            }) => assert_eq!(
                mutations, &expected,
                "shadow check: the plan in place emitted other mutations than the same plan on a copy"
            ),
            other => panic!(
                "shadow check: the plan on a copy ended {other:?} where the plan in place ran within"
            ),
        }
        let held = &self.original.as_ref().expect("held").state;
        assert_eq!(
            held, &planned,
            "shadow check: the plan in place left another state than the same plan on a copy"
        );
        shadow.planned = Some(planned);
    }

    /// The entry's state as it stood before the pending plan, read through
    /// the plan's journal: what the commit's pre-image digests, its size gate
    /// and its untouched-row counts read (`storage::structural::PreState`).
    ///
    /// # Panics
    ///
    /// When no plan is pending.
    pub fn pre_state(&self) -> PreImageView<'_> {
        let pending = self
            .pending
            .as_ref()
            .expect("pre_state reads the pending plan's journal");
        let entry = self.entry();
        let indexes = entry
            .plan
            .as_ref()
            .expect("an entry planned in place holds its plan indexes");
        pending.journal.view(&entry.state, indexes)
    }

    /// Settle a pending plan whose commit landed at `seq` with
    /// `commit_chain`: keep the planned state as the entry at that cut. The
    /// step a stream repair applies to a replayed commit
    /// (`structural_cache_repair::advance_to_successor`): refuse an adopted
    /// dangling parent, keep the dangling-parent set from the lists the
    /// commit's `mutations` delete (`StorageMutation::deleted_list`),
    /// normalize the lists the plan touched, advance the cut.
    /// The journal is dropped; no state is copied or dropped. Put the entry
    /// back with [`Self::put_back_held`] after.
    ///
    /// # Errors
    ///
    /// [`ReloadReason::DanglingAdopted`] when the plan created one of the
    /// entry's dangling parents: the caller discards the guard. The refusal
    /// comes before anything is settled, so the plan stays pending and any
    /// other settle of the guard rolls it back.
    ///
    /// # Panics
    ///
    /// When no plan is pending.
    pub fn advance_committed(
        &mut self,
        mutations: &[StorageMutation],
        seq: u64,
        commit_chain: String,
    ) -> Result<(), ReloadReason> {
        assert!(
            self.pending.is_some(),
            "advance_committed settles the pending plan"
        );
        if plan_adopts_dangling_parent(self.entry(), &self.entry().state) {
            #[cfg(any(test, feature = "test-support"))]
            if let Some(ShadowPlan {
                pre,
                planned: Some(planned),
            }) = self.pending.as_ref().and_then(|p| p.shadow.as_deref())
            {
                assert!(
                    plan_adopts_dangling_parent(pre, planned),
                    "shadow check: the plan in place adopted a dangling parent the plan on a copy did not"
                );
            }
            return Err(ReloadReason::DanglingAdopted);
        }
        let pending = self
            .pending
            .take()
            .expect("advance_committed settles the pending plan");
        let entry = self
            .original
            .as_mut()
            .expect("a CheckedOut holds its entry until put_back or discard consumes it");
        #[cfg(any(test, feature = "test-support"))]
        let oracle = pending.shadow.and_then(|shadow| {
            let ShadowPlan { pre, planned } = *shadow;
            planned.map(|planned| {
                crate::storage::structural_cache_repair::successor_entry(
                    &pre,
                    planned,
                    mutations,
                    seq,
                    commit_chain.clone(),
                )
            })
        });
        self.mutating = true;
        let result = pass_timing::time(Pass::SuccessorNormalize, || {
            advance_to_successor(
                entry,
                mutations.iter().filter_map(StorageMutation::deleted_list),
                seq,
                commit_chain,
                Some(&pending.touches.lists),
            )
        });
        self.mutating = false;
        #[cfg(any(test, feature = "test-support"))]
        if let Some(oracle) = oracle {
            match (&result, oracle) {
                (Ok(()), Ok(successor)) => {
                    let entry = self.entry();
                    assert_eq!(
                        (
                            &entry.state,
                            &entry.dangling_parents,
                            entry.seq,
                            &entry.commit_chain
                        ),
                        (
                            &successor.state,
                            &successor.dangling_parents,
                            successor.seq,
                            &successor.commit_chain
                        ),
                        "shadow check: the committed entry differs from the copy's successor"
                    );
                    assert_indexes_rebuilt(entry);
                }
                (Err(ours), Err(theirs)) => assert_eq!(*ours, theirs, "shadow check"),
                (ours, theirs) => panic!(
                    "shadow check: the commit settled {ours:?} in place but {:?} on a copy",
                    theirs.map(|_| ())
                ),
            }
        }
        result
    }

    /// Roll the pending plan back, if one is pending: the held entry returns
    /// to the pre-plan cut exactly, its contents if not its layout
    /// (`PlanJournal::rollback`). `Ok` with nothing pending.
    fn rollback_pending(&mut self) -> Result<(), TornJournal> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };
        let entry = self
            .original
            .as_mut()
            .expect("a CheckedOut holds its entry until put_back or discard consumes it");
        let CacheEntry { state, plan, .. } = entry;
        let Some(indexes) = plan.as_mut() else {
            return Err(TornJournal);
        };
        #[cfg(any(test, feature = "test-support"))]
        let projected = pending.journal.projected_rebuilds();
        self.mutating = true;
        let result = pending.journal.rollback(state, indexes);
        self.mutating = false;
        #[cfg(any(test, feature = "test-support"))]
        {
            self.cache.inner.rollbacks.fetch_add(1, Ordering::AcqRel);
            if let Ok(rebuilt) = &result {
                if !rebuilt.is_empty() {
                    self.cache
                        .inner
                        .rollback_rebuilds
                        .fetch_add(1, Ordering::AcqRel);
                }
                // Every rebuild was charged before the write that made it
                // necessary, unless the plan overflowed (a refused plan's
                // scratch runs above the charge until it returns, SDD §5.4).
                assert!(
                    pending.overflowed || rebuilt.is_subset(projected),
                    "shadow check: the rollback rebuilt {rebuilt:?}, but the plan projected \
                     only {projected:?}"
                );
            }
            if let (Ok(_), Some(shadow)) = (&result, pending.shadow) {
                let entry = self.entry();
                assert_eq!(
                    (
                        &entry.state,
                        &entry.dangling_parents,
                        entry.seq,
                        &entry.commit_chain
                    ),
                    (
                        &shadow.pre.state,
                        &shadow.pre.dangling_parents,
                        shadow.pre.seq,
                        &shadow.pre.commit_chain
                    ),
                    "shadow check: the rollback did not restore the pre-plan entry"
                );
                assert_indexes_rebuilt(entry);
            }
        }
        result.map(|_| ())
    }

    /// What a rollback of the pending plan as it stands could rebuild, in
    /// bytes — an upper bound, each rebuild priced exactly
    /// (`PlanJournal::rollback_rebuild_bytes`); 0 with no plan pending. CR2
    /// holds it beside the planned entry, so a rollback after CR2 runs
    /// inside the call's charge.
    pub fn rollback_rebuild_bytes(&self) -> u64 {
        let Some(pending) = self.pending.as_ref() else {
            return 0;
        };
        let entry = self.entry();
        entry.plan.as_ref().map_or(0, |indexes| {
            pending
                .journal
                .rollback_rebuild_bytes(&entry.state, indexes)
        })
    }

    /// Settle by dropping the held entry and invalidating the object, from
    /// a guard whose entry cannot go back.
    fn discard_held(&mut self) {
        self.pending = None;
        if let Some(original) = self.original.take() {
            let dropped = self
                .cache
                .inner
                .settle_discard(self.checkout_epoch, original.object_id);
            drop(dropped);
            drop(original);
        }
    }

    /// Run `f` on the held entry in place: the stream repair
    /// (`storage::structural_cache_repair`, SR-4.1) advances the entry this
    /// way, record by record, without cloning its state. Planning uses
    /// [`Self::plan_in_place`] instead.
    ///
    /// `f` must leave the entry a valid cut — its state replayed exactly to
    /// its `seq` and `commit_chain` — or the caller must settle the guard with
    /// [`Self::discard`] (the repair does, after any replay error). Should `f`
    /// panic, the guard's `Drop` discards the entry rather than putting back a
    /// half-replayed state. A test-support build with the shadow check on
    /// checks after `f`, as after a plan in place, that the entry is weighed
    /// at or above the bytes its tables' buckets hold.
    pub(crate) fn mutate<R>(&mut self, f: impl FnOnce(&mut CacheEntry) -> R) -> R {
        debug_assert!(
            self.pending.is_none(),
            "mutate runs before a plan in place, never while one is pending"
        );
        let entry = self
            .original
            .as_mut()
            .expect("a CheckedOut holds its entry until put_back or discard consumes it");
        self.mutating = true;
        self.advanced = true;
        let out = f(entry);
        self.mutating = false;
        #[cfg(any(test, feature = "test-support"))]
        if self.cache.shadow_check_for_test() {
            assert_weighed_at_or_above_its_buckets(self.entry(), "a repair's replayed round");
        }
        out
    }

    /// Settle the checkout by putting a given `entry` back in place of the
    /// held one, a pending plan's included: the tests' put-back of an entry
    /// they built — a successor from `successor_entry` on a planned copy, an
    /// entry grown past the cap. Compiled into test and test-support builds
    /// only: the cached route never replaces the held entry, but settles a
    /// commit in place ([`Self::advance_committed`]) and every exit through
    /// [`Self::put_back_held`] (or [`Self::discard`]). `entry` must be for
    /// the checked-out object and normalized, as a successor from
    /// `successor_entry` is; its state and plan indexes are compacted
    /// ([`compact_state`]) and its weights recomputed here.
    ///
    /// Returns how the put-back settled ([`PutBack`]). [`PutBack::Stored`]:
    /// the put-back was accepted — though the store can evict the entry in
    /// the same critical section, for its own weight, over the most a held
    /// entry may weigh (`Control::store`, `held_entry_bound`: then it is
    /// never placed), or by the resident bound
    /// (`Control::hold_resident_bound`) after placing it (see
    /// [`PutBack::Stored`]). Otherwise the put-back is refused and `entry` is
    /// dropped:
    ///
    /// - [`PutBack::Invalidated`]: the object was invalidated after this
    ///   checkout's epoch (a retirement, an audit's `diverged`, another
    ///   guard's `discard`); the object stays as the invalidation left it;
    /// - [`PutBack::OverCap`]: `entry`'s state is heavier than the per-entry cap,
    ///   [`StructuralCache::max_entry_bytes`] (SR-3.3 admission): the object
    ///   gets an
    ///   [`UncacheableReason::OverWeight`] negative entry, so its structural
    ///   edits take the uncacheable path until that expires. The caller
    ///   counts it (the module docs, "Checkout and the epoch check").
    ///
    /// An accepted entry is stored even if mini-moka's admission refuses its
    /// key: it is kept in the side store and still served as a hit.
    #[cfg(any(test, feature = "test-support"))]
    pub fn put_back(mut self, entry: CacheEntry) -> PutBack {
        // `entry` replaces the held entry whole, a pending plan's included.
        self.pending = None;
        let original = self
            .original
            .take()
            .expect("a CheckedOut holds its entry until put_back or discard consumes it");
        debug_assert_eq!(
            entry.object_id, original.object_id,
            "put_back must return the checked-out object's entry"
        );
        pass_timing::time(Pass::SettleDrop, || drop(original));
        self.settle_with(entry)
    }

    /// Settle the checkout by putting the held entry back — the original, the
    /// original as a repair advanced it (`CheckedOut::mutate`), or the entry
    /// a `Committed` plan advanced in place ([`Self::advance_committed`]) —
    /// with its state and plan indexes compacted and its weights recomputed
    /// ([`CacheEntry::compact_and_weigh`]). The put-back for every exit that
    /// keeps the object cached (SR-3.4): a `Committed` commit, after
    /// `advance_committed`; and, rolling back the plan, `STALE`, `FENCED`, a
    /// delta `Err`, a bounded plan's overflow, a refused charge, the size
    /// gate, a transport error, an in-doubt reply, a terminal store-side
    /// refusal that changed nothing, or a commit token found already landed.
    /// Returns how it settled ([`PutBack`]). [`PutBack::Stored`]: the
    /// put-back was accepted — though the store can evict the entry in the
    /// same critical section, for its own weight, over the most a held entry
    /// may weigh (`Control::store`, `held_entry_bound`: then it is never
    /// placed), or by the resident bound (`Control::hold_resident_bound`)
    /// after placing it (see [`PutBack::Stored`]).
    /// Otherwise it is refused and dropped:
    /// [`PutBack::Invalidated`] when the object was invalidated after this
    /// checkout's epoch, [`PutBack::OverCap`] when the entry's state is
    /// heavier than the per-entry cap ([`StructuralCache::max_entry_bytes`]),
    /// which gives the object an [`UncacheableReason::OverWeight`] negative
    /// entry (a held entry a repair or a commit grew past the cap). An
    /// accepted entry is stored even if mini-moka's admission refuses its
    /// key: it is kept in the side store and still served as a hit.
    ///
    /// A pending plan ([`Self::plan_in_place`]) is rolled back first, so the
    /// entry that goes back is the pre-plan cut — including on a mid-call
    /// `NeedsCharge`, whose put-back the call's wait follows. A committed
    /// plan was settled by [`Self::advance_committed`] and goes back as
    /// planned. Should the rollback fail (a defect: a torn journal), the
    /// object is invalidated and nothing goes back
    /// ([`PutBack::Invalidated`]).
    pub fn put_back_held(self) -> PutBack {
        self.put_back_held_for_resume(|_| ()).0
    }

    /// [`Self::put_back_held`], also answering `resume` read on the entry as
    /// it settled — rolled back, compacted and weighed, its stored weights
    /// current — when the put-back was accepted ([`PutBack::Stored`]); `None`
    /// otherwise, when nothing of it is cached for a resumed attempt to plan
    /// on. An accepted entry the store evicted in the same critical section
    /// — for its own weight, over the most a held entry may weigh
    /// (`Control::store`), or by the resident bound
    /// (`Control::hold_resident_bound`) — still answers `Some`, though
    /// nothing of it stays cached: the resumed attempt then cold-starts (see
    /// [`PutBack::Stored`]). O(1)
    /// beyond the put-back itself and `resume`.
    ///
    /// For a call that suspends after a plan in place: the rollback restores
    /// the state's contents and, once compacted here, its state weight byte
    /// for byte — the field-name interner's too, every name back at its
    /// pre-plan id (`domain::plan_journal`), so no rolled-back attempt, and
    /// no run of them, leaves the entry heavier against the per-entry cap —
    /// but not its tables' layout — a table still minimal for its length keeps the
    /// tombstones of the keys the rollback removed (compaction reallocates
    /// only an oversized table, [`compact_state`]), which leaves it less
    /// room, while a table the plan grew is shrunk back without them; and a
    /// name-index vector the plan grew keeps its grown capacity, while one
    /// rebuilt or re-created sits at its restored length
    /// (`PlanIndexes::restore_name_index`). So the resumed attempt's CR1 can
    /// need more or less than the pre-plan entry's did, and a refused plan
    /// growth reads that need here, on the entry it will resume on
    /// (`storage::layer`, the `Overflow` arm). The resumed plan's own
    /// rollback-rebuild projections follow the room and capacities it meets
    /// and are not predicted here: it can ask once more.
    pub fn put_back_held_for_resume<R>(
        mut self,
        resume: impl FnOnce(&CacheEntry) -> R,
    ) -> (PutBack, Option<R>) {
        if self.rollback_pending().is_err() {
            self.discard_held();
            return (PutBack::Invalidated, None);
        }
        let held = self
            .original
            .take()
            .expect("a CheckedOut holds its entry until put_back or discard consumes it");
        self.settle_reading(held, resume)
    }

    /// Compact and weigh `entry` ([`CacheEntry::compact_and_weigh`]: O(1) for
    /// an entry holding its plan indexes whose tables are minimal), then
    /// settle the checkout by storing it, subject to the epoch check and the
    /// per-entry cap on its state weight. The guard's own entry has already
    /// been taken out, so its `Drop` does nothing more. A test-support build
    /// with the shadow check on also checks the O(1) weight against the
    /// whole-state estimate and the indexes against a rebuild.
    fn settle_with(&self, entry: CacheEntry) -> PutBack {
        self.settle_reading(entry, |_| ()).0
    }

    /// [`Self::settle_with`], also answering `read` of the compacted and
    /// weighed entry when it was stored.
    fn settle_reading<R>(
        &self,
        mut entry: CacheEntry,
        read: impl FnOnce(&CacheEntry) -> R,
    ) -> (PutBack, Option<R>) {
        pass_timing::time(Pass::SettleWeigh, || entry.compact_and_weigh());
        #[cfg(any(test, feature = "test-support"))]
        if self.cache.shadow_check_for_test() {
            entry.assert_weighed();
        }
        entry.last_used = Instant::now();
        let read = read(&entry);
        let (settled, freed) = self.cache.inner.settle_put_back(self.checkout_epoch, entry);
        drop(freed);
        let read = matches!(settled, PutBack::Stored).then_some(read);
        (settled, read)
    }

    /// Settle the checkout by putting nothing back, and invalidate the object
    /// (SR-3.4): the next structural call on it cold-starts it. Every other
    /// outstanding checkout of the object taken before now has its put-back
    /// refused. Used after:
    ///
    /// - `CACHE_DIVERGED`;
    /// - a repair anomaly (the reload replaces the entry);
    /// - a header showing the object gone or replaced — `NotFound`,
    ///   `SUPERSEDED` or `PARTIAL_HEADER` — whether the repair saw it or the
    ///   commit did;
    /// - a `Committed` commit whose plan created a section whose id is in the
    ///   entry's `dangling_parents` (SR-3.2 adoption): the creation makes a
    ///   stored list visible that the planned successor does not hold, so it
    ///   is dropped instead of swapped in, and the next call reloads
    ///   (`dangling_adopted`).
    pub fn discard(mut self) {
        assert!(
            self.original.is_some(),
            "a CheckedOut holds its entry until put_back or discard consumes it"
        );
        self.discard_held();
    }
}

impl Drop for CheckedOut {
    /// The RAII exit: when no consuming method settled the guard, put the held
    /// entry back, subject to the epoch check (SR-3.4).
    ///
    /// - An entry `CheckedOut::mutate` never touched is the original, put
    ///   back unchanged at the weight it was checked out with.
    /// - An entry a repair advanced goes back the way
    ///   [`CheckedOut::put_back_held`] puts it: compacted and re-weighed
    ///   before the epoch and per-entry-cap checks, so an entry the repair
    ///   grew past the cap becomes an [`UncacheableReason::OverWeight`]
    ///   negative entry instead of slipping in under its old weight. That
    ///   covers every implicit exit after a repair: a store error or a `?`
    ///   mid-repair, a deadline or cancellation, a caller dropping a ready
    ///   entry. No one hears this put-back, so the verdict stays unreported
    ///   until the first structural lookup it routes reports it
    ///   ([`StructuralCache::take_unreported`]).
    /// - A guard dropped while `mutate`'s closure, a plan in place or its
    ///   rollback unwound discards instead (see `CheckedOut::mutate`).
    /// - A guard dropped with a plan pending — the call dropped mid-commit —
    ///   rolls the plan back first (inside `catch_unwind`), then puts back as
    ///   above; should the rollback fail or panic, it discards. A commit that
    ///   did land leaves the entry one cut behind, which the next call's
    ///   repair advances. While a panic unwinds through the guard with a
    ///   plan pending it discards without a rollback: a second panic there
    ///   would abort the process.
    fn drop(&mut self) {
        if self.mutating {
            self.discard_held();
            return;
        }
        if self.pending.is_some() {
            if std::thread::panicking() {
                self.discard_held();
                return;
            }
            let rolled_back =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.rollback_pending()));
            match rolled_back {
                Ok(Ok(())) => {}
                Ok(Err(_)) => {
                    self.discard_held();
                    return;
                }
                Err(_panic) => {
                    self.discard_held();
                    // A test-support build's failed shadow check must fail
                    // its test, not vanish into this discard.
                    #[cfg(any(test, feature = "test-support"))]
                    std::panic::resume_unwind(_panic);
                    #[cfg(not(any(test, feature = "test-support")))]
                    return;
                }
            }
        }
        if self.advanced {
            if let Some(held) = self.original.take() {
                self.settle_with(held);
            }
            return;
        }
        if let Some(original) = self.original.take() {
            let (_, freed) = self
                .cache
                .inner
                .settle_put_back(self.checkout_epoch, original);
            drop(freed);
        }
    }
}

/// The test-support shadow check's index check: an entry's plan indexes
/// describe exactly what a fresh build of its state would, and their O(1)
/// weight equals their recomputed one.
#[cfg(any(test, feature = "test-support"))]
fn assert_indexes_rebuilt(entry: &CacheEntry) {
    if let Some(indexes) = entry.plan.as_ref() {
        assert!(
            indexes.same_content_as(&PlanIndexes::build(&entry.state)),
            "shadow check: the plan indexes differ from a rebuild of the entry's state"
        );
        assert_eq!(
            indexes.weight_bytes(),
            indexes.weight_bytes_recomputed(),
            "shadow check: the plan indexes' running weight drifted"
        );
    }
}

// ── The weight estimate (SR-3.3) ────────────────────────────────────────────

/// One hash-table slot of a state's `sections`.
const SECTION_SLOT_BYTES: u64 = size_of::<(u64, Section)>() as u64;
/// One hash-table slot of the full form's `fields`.
const FIELD_SLOT_BYTES: u64 = size_of::<((u64, String), Field)>() as u64;
/// One hash-table slot of the full form's `time_samples`.
const TIME_SAMPLE_SLOT_BYTES: u64 = size_of::<((u64, OrderedFloat<f64>), TimeSample)>() as u64;
/// One hash-table slot of the compact form's `fields`.
const COMPACT_FIELD_SLOT_BYTES: u64 = CompactFields::ROW_SLOT_BYTES as u64;
/// One hash-table slot of the compact form's `time_samples`.
const COMPACT_TIME_SAMPLE_SLOT_BYTES: u64 = CompactSamples::ROW_SLOT_BYTES as u64;
/// One hash-table slot of a state's `children`.
const CHILDREN_SLOT_BYTES: u64 = size_of::<(u64, Vec<u64>)>() as u64;
/// One child id in a children list.
const CHILD_ID_BYTES: u64 = size_of::<u64>() as u64;

/// A row form's share of [`estimate_state_bytes`]: the bytes its value rows
/// occupy. Implemented for both forms: the cache charges the compact one
/// ([`CompactRows`]); the full one ([`FullRows`]) is the rule the cache
/// charged before its entries were compact, kept for the calibration test and
/// the measurement tools, which compare the two.
pub trait WeighedRows: ValueRows {
    /// The bytes of `fields` and `time_samples`: their tables, keys and
    /// values.
    fn value_rows_bytes(fields: &Self::Fields, time_samples: &Self::Samples) -> u64;
}

impl WeighedRows for FullRows {
    fn value_rows_bytes(
        fields: &HashMap<(u64, String), Field>,
        time_samples: &HashMap<(u64, OrderedFloat<f64>), TimeSample>,
    ) -> u64 {
        let mut bytes = table_bytes(fields.capacity(), FIELD_SLOT_BYTES);
        for ((_, key), field) in fields {
            // The key's name and the field's own `key_name` are two allocations.
            bytes = bytes
                .saturating_add(allocation_bytes(key.capacity() as u64))
                .saturating_add(allocation_bytes(field.key_name.capacity() as u64))
                .saturating_add(value_heap_bytes(&field.value));
        }
        bytes = bytes.saturating_add(table_bytes(time_samples.capacity(), TIME_SAMPLE_SLOT_BYTES));
        for sample in time_samples.values() {
            bytes = bytes.saturating_add(value_heap_bytes(&sample.value));
        }
        bytes
    }
}

impl WeighedRows for CompactRows {
    fn value_rows_bytes(fields: &CompactFields, time_samples: &CompactSamples) -> u64 {
        let hash_bytes = allocation_bytes(CompactFields::HASH_BYTES as u64);
        let names = fields.names();
        let mut bytes = table_bytes(fields.capacity(), COMPACT_FIELD_SLOT_BYTES)
            .saturating_add((fields.external_rows() as u64).saturating_mul(hash_bytes))
            .saturating_add(table_bytes(
                names.ids_capacity(),
                NameInterner::ID_ENTRY_BYTES as u64,
            ))
            .saturating_add(allocation_bytes(
                (names.slots_capacity() as u64).saturating_mul(NameInterner::SLOT_BYTES as u64),
            ))
            .saturating_add(allocation_bytes(
                (names.free_capacity() as u64).saturating_mul(size_of::<u32>() as u64),
            ));
        for name in names.names() {
            bytes = bytes.saturating_add(allocation_bytes(NameInterner::name_allocation_bytes(
                name.len(),
            ) as u64));
        }
        bytes
            .saturating_add(table_bytes(
                time_samples.capacity(),
                COMPACT_TIME_SAMPLE_SLOT_BYTES,
            ))
            .saturating_add((time_samples.external_rows() as u64).saturating_mul(hash_bytes))
    }
}

/// The estimated heap and inline bytes `state` occupies: for a
/// [`CompactState`], the weight the cache charges an entry against its
/// budget (SR-3.3).
///
/// Per-entry constants for sections, value rows and children lists — each
/// map's slot size from the types themselves, times the bucket count the
/// map's capacity implies, plus one control byte per bucket — plus every
/// heap allocation, rounded up to jemalloc's size classes: section names and
/// children lists, and the value rows' own ([`WeighedRows`]). A compact row
/// holds no value: an external row charges its boxed 32-byte hash, and the
/// field names are charged once per state, in its name interner. On the full
/// form, every key, string and inline value is charged and an external
/// value only its inline hash (its bytes live in the store). The estimate's
/// error against jemalloc's measured allocation is recorded by its
/// calibration test.
pub fn estimate_state_bytes<R: WeighedRows>(state: &ObjectStateOf<R>) -> u64 {
    let mut bytes = (size_of::<ObjectStateOf<R>>() as u64)
        .saturating_add(allocation_bytes(state.source_format.capacity() as u64));

    bytes = bytes.saturating_add(table_bytes(state.sections.capacity(), SECTION_SLOT_BYTES));
    for section in state.sections.values() {
        bytes = bytes.saturating_add(allocation_bytes(section.name.capacity() as u64));
    }

    bytes = bytes.saturating_add(table_bytes(state.children.capacity(), CHILDREN_SLOT_BYTES));
    for kids in state.children.values() {
        bytes = bytes.saturating_add(allocation_bytes(
            (kids.capacity() as u64).saturating_mul(CHILD_ID_BYTES),
        ));
    }

    bytes.saturating_add(R::value_rows_bytes(&state.fields, &state.time_samples))
}

/// [`estimate_state_bytes`] of a compact `state` in O(1), from `indexes`,
/// the plan indexes kept in step with it (`domain::plan_index`): the same
/// terms, each of the state's hash tables in the table registry
/// (`domain::delta_write::RebuildTable`) read at its held capacity
/// (`PlanIndexes::held_capacity`), and the parts the estimate walks the
/// state for — section names, children lists, interned field names and the
/// external rows' boxed hashes — from the indexes' running sums and
/// counters. Equal to the estimate byte for byte at every compaction, where
/// each held capacity is the table's `capacity()` (the settle check asserts
/// it in test-support builds); between compactions, at or above it, a table
/// thinned since then read at the buckets its tombstones hide from
/// `capacity()`, never below the bytes it holds.
pub fn indexed_state_bytes(state: &CompactState, indexes: &PlanIndexes) -> u64 {
    indexed_state_bytes_with(state, indexes, |table, capacity| {
        indexes.held_capacity(table, capacity)
    })
}

/// [`indexed_state_bytes`] with each of the state's hash tables read at
/// `read(table, capacity())`.
fn indexed_state_bytes_with(
    state: &CompactState,
    indexes: &PlanIndexes,
    read: impl Fn(RebuildTable, usize) -> usize,
) -> u64 {
    let hash_bytes = allocation_bytes(CompactFields::HASH_BYTES as u64);
    let names = state.fields.names();
    let tables = RebuildTable::ALL
        .iter()
        .filter_map(|&table| {
            let (_, capacity) = state_table_shape(table, state)?;
            Some(table_bytes(read(table, capacity), table_slot_bytes(table)))
        })
        .fold(0, u64::saturating_add);
    (size_of::<CompactState>() as u64)
        .saturating_add(allocation_bytes(state.source_format.capacity() as u64))
        .saturating_add(tables)
        .saturating_add(indexes.section_names_bytes())
        .saturating_add(indexes.lists_bytes())
        .saturating_add(indexes.external_field_rows().saturating_mul(hash_bytes))
        .saturating_add(allocation_bytes(
            (names.slots_capacity() as u64).saturating_mul(NameInterner::SLOT_BYTES as u64),
        ))
        .saturating_add(allocation_bytes(
            (names.free_capacity() as u64).saturating_mul(size_of::<u32>() as u64),
        ))
        .saturating_add(indexes.interned_names_bytes())
        .saturating_add(indexes.external_sample_rows().saturating_mul(hash_bytes))
}

/// **Test support.** [`indexed_state_bytes`] with each of the state's hash
/// tables at its true capacity (`domain::types::true_capacity_for_test`):
/// the same running sums, the tables at the buckets they really hold. The
/// shadow check's weight floor ([`CheckedOut::plan_in_place`]). O(the
/// tables).
#[cfg(any(test, feature = "test-support"))]
pub fn indexed_state_bytes_at_true_capacity_for_test(
    state: &CompactState,
    indexes: &PlanIndexes,
) -> u64 {
    let tables = crate::domain::plan_index::state_hash_tables_for_test(state);
    indexed_state_bytes_with(state, indexes, |table, capacity| {
        tables
            .iter()
            .find(|&&(listed, _)| listed == table)
            .map_or(capacity, |(_, capacities)| capacities.true_capacity)
    })
}

/// The test-support shadow check's weight floor: `entry`, after `after` (a
/// plan in place, a repair's replayed round), is weighed at or above the
/// bytes its hash tables hold — each table's held capacity
/// (`PlanIndexes::held_capacity`) reads at least the bucket count its true
/// capacity does (`domain::types::true_capacity_for_test`), and so
/// [`CacheEntry::weights_now`] is at least the same weights with every table
/// at its true capacity. Nothing to check on an entry without plan indexes,
/// which is weighed by the whole-state estimate and never in a call's
/// charge. O(the tables): one clone and drain of each.
#[cfg(any(test, feature = "test-support"))]
fn assert_weighed_at_or_above_its_buckets(entry: &CacheEntry, after: &str) {
    let Some(indexes) = entry.plan.as_ref() else {
        return;
    };
    let tables = crate::domain::plan_index::hash_tables_for_test(&entry.state, indexes);
    for &(table, capacities) in &tables {
        let held = indexes.held_capacity(table, capacities.capacity);
        let slot = table_slot_bytes(table);
        assert!(
            table_bytes(held, slot) >= table_bytes(capacities.true_capacity, slot),
            "shadow check: after {after}, the {table:?} table is weighed at a held capacity of \
             {held} slots, fewer buckets than the {} usable slots it holds (capacity() reads {})",
            capacities.true_capacity,
            capacities.capacity,
        );
    }
    let true_capacity = |table: RebuildTable, capacity: usize| {
        tables
            .iter()
            .find(|&&(listed, _)| listed == table)
            .map_or(capacity, |(_, capacities)| capacities.true_capacity)
    };
    let (state_weight, index_weight) = entry.weights_now();
    let state_floor = indexed_state_bytes_with(&entry.state, indexes, true_capacity);
    assert!(
        state_weight >= state_floor,
        "shadow check: after {after}, the entry's state weighs {state_weight} B, below the \
         {state_floor} B its tables' buckets hold"
    );
    let index_floor = indexes.weight_bytes_at_true_capacity_for_test();
    assert!(
        index_weight >= index_floor,
        "shadow check: after {after}, the entry's plan indexes weigh {index_weight} B, below the \
         {index_floor} B their maps' buckets hold"
    );
}

/// **Test support.** [`estimate_state_bytes`] of a compact `state` with each
/// of its hash tables at its true capacity
/// (`domain::types::true_capacity_for_test`) instead of its `capacity()`:
/// the bytes the state really holds, by the estimate's rule, whatever its
/// tombstones hide. O(the state).
#[cfg(any(test, feature = "test-support"))]
pub fn estimate_state_bytes_at_true_capacity_for_test(state: &CompactState) -> u64 {
    crate::domain::plan_index::state_hash_tables_for_test(state)
        .into_iter()
        .fold(estimate_state_bytes(state), |bytes, (table, capacities)| {
            let slot = table_slot_bytes(table);
            bytes
                .saturating_sub(table_bytes(capacities.capacity, slot))
                .saturating_add(table_bytes(capacities.true_capacity, slot))
        })
}

/// The `insert` calls a structural delta can make on each table of the state
/// it plans on, and whether it is a reset: what
/// [`estimate_in_place_growth_bytes`] prices the plan's in-place growth
/// from (CR1), and what the test-support copy [`CheckedOut::plan_state`]
/// clones room for. Counted from the delta alone, each an upper bound on the
/// insert calls that table can take inside the plan — new keys and
/// overwrites alike, since an overwrite on a full table grows it too
/// ([`table_growth_bytes`]). The structural bounds are the engine's own
/// ([`StructuralInserts`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlannedInserts {
    /// Section rows: one per `CreateSection` and `DiffChangeSection`
    /// ([`StructuralInserts::sections`]).
    pub sections: u64,
    /// Children-list keys: two per `CreateSection` and `DiffChangeSection`
    /// (its parent's list, when the parent has none yet, and the newborn's
    /// own list, which its adopted orphans join), one per `MoveSection` (its
    /// destination's) and one per `ReorderChildren` (its parent's list,
    /// installed over the existing key)
    /// ([`StructuralInserts::children_list_keys`]).
    pub children_list_keys: u64,
    /// Field rows and interned field names: the delta's set fields.
    pub set_fields: u64,
    /// Time-sample rows: the delta's time samples.
    pub time_samples: u64,
    /// Insert calls on the plan indexes' external-hash counts: the delta's
    /// set fields and time samples that carry an external value
    /// (`ext_value_size > 0`). Only those rows' writes
    /// (`domain::delta::apply_set_field`, `apply_time_sample`) can make a
    /// row hold an external hash, and the counts take an insert call only for
    /// a row that holds one (`PlanIndexes::count_hash`), at most once per
    /// such write. A rollback's re-creations are not counted here: it
    /// rebuilds a map without the room at its pre-plan length, charged on
    /// its own (`RewriteCount::rollback_rebuild_bytes`).
    pub external_rows: u64,
    /// Insert calls on the plan indexes' parent-keyed name map: two per
    /// birth, one per move and one per reorder
    /// ([`StructuralInserts::name_key_inserts`]).
    pub name_key_inserts: u64,
    /// Whether the delta is a reset (`is_diff` with `base_version` 0): a
    /// hooked plan then moves every table out into its journal and plans
    /// into fresh ones allocated at the old capacities.
    pub reset: bool,
}

impl PlannedInserts {
    /// The inserts `delta` can make; none for an empty commit. O(elements),
    /// allocating nothing.
    pub fn of(delta: Option<&oodf_ffi::OwnedDelta>) -> Self {
        let count = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
        delta.map_or_else(Self::default, |delta| {
            let structural = StructuralInserts::of(delta);
            let external_fields = delta
                .set_fields
                .iter()
                .filter(|field| field.ext_value_size > 0)
                .count();
            let external_samples = delta
                .time_samples
                .iter()
                .filter(|sample| sample.ext_value_size > 0)
                .count();
            Self {
                sections: structural.sections(),
                children_list_keys: structural.children_list_keys(),
                set_fields: count(delta.set_fields.len()),
                time_samples: count(delta.time_samples.len()),
                external_rows: count(external_fields).saturating_add(count(external_samples)),
                name_key_inserts: structural.name_key_inserts(),
                reset: delta.is_diff && delta.base_version == 0,
            }
        })
    }

    /// The insert calls the delta can make on the hash table `table` of the
    /// table registry: the state's sections, children-list keys, field rows,
    /// time-sample rows and interned names (one per set field), and the plan
    /// indexes' parent keys and external-hash counts.
    pub fn calls(&self, table: RebuildTable) -> u64 {
        match table {
            RebuildTable::Sections => self.sections,
            RebuildTable::Children => self.children_list_keys,
            RebuildTable::FieldRows | RebuildTable::NameIds => self.set_fields,
            RebuildTable::SampleRows => self.time_samples,
            RebuildTable::NameIndex => self.name_key_inserts,
            RebuildTable::ExtCounts => self.external_rows,
        }
    }
}

/// What a plan made in place on `state` (with its plan indexes `indexes`)
/// for a delta of `inserts` can allocate beside the held entry, outside its
/// rewrite: CR1's growth term, priced 1:1. O(1).
///
/// - **A table that grows.** Each hash table of the table registry
///   (`domain::delta_write::RebuildTable`) whose length plus its insert calls
///   ([`PlannedInserts::calls`]) passes its capacity is charged what its
///   growth can hold at once ([`table_growth_bytes_held`]): at least the
///   doubled table hashbrown grows to, `table_bytes(max(len + inserts, full +
///   1))` with `full` the usable slots of its current buckets — even when
///   tombstones its removals left put its length far below `full` — and,
///   when the inserts outgrow one doubling, the table before the last as
///   well. With `indexes`, the current buckets are read from the table's
///   held capacity (`PlanIndexes::held_capacity`), which those tombstones
///   cannot hide; without, from its `capacity()`. The old table is still
///   held in the entry's weight, read the same way. The name interner's id →
///   name vector is charged its
///   growth by `Vec`'s doubling rule ([`vec_growth_bytes`]) when its live
///   names plus the new ones pass its capacity (new names fill freed slots
///   first). A table with room costs nothing.
/// - **A reset.** A hooked reset moves every table out into the plan's
///   journal and plans into fresh ones allocated at the old capacities
///   (`domain::delta_write::take_for_reset`), so each of those is charged
///   whole — sections, children, field rows, the interner's id table, its
///   two vectors, time samples — and grown past that only if the delta's
///   inserts outnumber the old capacity. Those fresh tables hold no
///   tombstones, so their growth is read from their `capacity()`. The plan
///   indexes restart empty and are priced from zero.
/// - **The plan indexes' maps** ([`PlanIndexes::growth_bytes`]): the name
///   index's parent keys and the external-hash counts, by the same rule (at
///   their held capacities), the counts' insert calls bounded by the delta's
///   external rows ([`PlannedInserts::external_rows`]).
///
/// Not here:
/// - the children lists and name-index vectors a plan grows, each with a
///   parent's listing that the plan projects at its whole length before it
///   builds it (the rewrite's list entries);
/// - a rollback's re-creations: it rebuilds a table without the room at its
///   pre-plan length rather than growing it, and the plan projects that
///   rebuild before the write that can make it necessary
///   (`RewriteCount::rollback_rebuild_bytes`), which CR2 keeps
///   (`CheckedOut::rollback_rebuild_bytes`);
/// - the plan's journal. Its records of the rows and lists the plan
///   rewrites — rows a cascade, a reset or an adoption writes, lists it
///   rewrites — follow the rewrite's row and list terms. Its records of the
///   delta's own elements (each set field, each time sample, and each row a
///   create, a diff-change or a move writes) are outside the rewrite
///   (`RewriteCount::rows` excludes own elements) and are covered by the
///   upload's per-element charges
///   (`grpc::upload_admission::ELEMENT_CHARGE_*`).
pub fn estimate_in_place_growth_bytes(
    state: &CompactState,
    indexes: Option<&PlanIndexes>,
    inserts: PlannedInserts,
) -> u64 {
    in_place_growth(state, indexes, None, inserts).0
}

/// [`estimate_in_place_growth_bytes`], each of the state's hash tables
/// priced from: with `bound` (the sampled audit's copy, which holds no
/// indexes), the bucket count the bound reads ([`table_growth_bounded`]),
/// also answering, per table, what the bound's raise reads
/// ([`PricedBuckets`]); otherwise, outside a reset, with `indexes` from the
/// table's held capacity ([`table_growth_bytes_held`]), without them from
/// its `capacity()`. A hooked reset plans into fresh tables at the live
/// capacities (`domain::delta_write::take_for_reset`), exact, so its fresh
/// tables are priced from `capacity()`. The copy's index-less reset
/// allocates no fresh table: it drains its sections and value rows in place
/// and `clear()`s its children (`domain::delta_write::ResetTables`), each
/// keeping its buckets — and `clear()` on a map with no items left skips
/// the reset of its control bytes, so it keeps that map's tombstones. Its
/// `fresh` term is therefore a pure over-price, and its growth is priced
/// from the bound even then.
fn in_place_growth(
    state: &CompactState,
    indexes: Option<&PlanIndexes>,
    bound: Option<&HeldCapacities>,
    inserts: PlannedInserts,
) -> (u64, PricedBuckets) {
    let reset = inserts.reset;
    let mut priced = PricedBuckets::default();
    // Each hash table of the state, from the table registry: a reset's fresh
    // copy at the old capacity, plus what its growth holds at once when the
    // insert calls pass its room.
    let tables = RebuildTable::ALL
        .iter()
        .filter_map(|&table| {
            let (len, capacity) = state_table_shape(table, state)?;
            let len_before = len;
            let slot = table_slot_bytes(table);
            let calls = inserts.calls(table);
            let (len, fresh) = if reset {
                (0, table_bytes(capacity, slot))
            } else {
                (len, 0)
            };
            let growth = match (bound, indexes) {
                (Some(bound), _) => {
                    let grown = table_growth_bounded(len, capacity, bound.get(table), calls, slot);
                    priced.price(table, len_before, len, capacity, calls, grown.grown_buckets);
                    grown.bytes
                }
                (None, Some(indexes)) if !reset => table_growth_bytes_held(
                    len,
                    capacity,
                    indexes.held_capacity(table, capacity),
                    calls,
                    slot,
                ),
                (None, _) => table_growth_bytes(len, capacity, calls, slot),
            };
            Some(fresh.saturating_add(growth))
        })
        .fold(0, u64::saturating_add);
    // One vector of `slot_bytes` elements: a reset's fresh copy at the old
    // capacity, plus its growth by `Vec`'s doubling rule.
    let vector = |len: usize, capacity: usize, pushes: u64, slot_bytes: u64| {
        let (len, fresh) = if reset {
            (
                0,
                allocation_bytes((capacity as u64).saturating_mul(slot_bytes)),
            )
        } else {
            (len, 0)
        };
        fresh.saturating_add(vec_growth_bytes(len, capacity, pushes, slot_bytes))
    };
    let names = state.fields.names();
    let free = if reset {
        allocation_bytes((names.free_capacity() as u64).saturating_mul(size_of::<u32>() as u64))
    } else {
        0
    };
    let index_growth = indexes.map_or(0, |indexes| {
        indexes.growth_bytes(|table| inserts.calls(table), reset)
    });
    let bytes = tables
        .saturating_add(vector(
            names.len(),
            names.slots_capacity(),
            inserts.set_fields,
            NameInterner::SLOT_BYTES as u64,
        ))
        .saturating_add(free)
        .saturating_add(index_growth);
    (bytes, priced)
}

/// One capacity per hash table of the state — the table registry's
/// [`RebuildTable::in_state`] tables, by [`RebuildTable::index`]: a stack
/// value, carried by the sampled audit from the held entry to its copy's
/// bucket bound ([`CopyBucketBound`]). The plan indexes' own maps have no
/// slot: an index-less copy holds none.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeldCapacities([usize; RebuildTable::STATE_COUNT]);

impl HeldCapacities {
    /// Each of `state`'s hash tables at its `capacity()`.
    pub fn live(state: &CompactState) -> Self {
        let mut live = Self::default();
        for table in RebuildTable::ALL {
            if let Some((_, capacity)) = state_table_shape(table, state) {
                live.set(table, capacity);
            }
        }
        live
    }

    /// `table`'s capacity; 0 for one of the plan indexes' own maps.
    pub fn get(&self, table: RebuildTable) -> usize {
        self.0.get(table.index()).copied().unwrap_or(0)
    }

    /// Set `table`'s capacity; nothing for one of the plan indexes' own
    /// maps.
    pub fn set(&mut self, table: RebuildTable, capacity: usize) {
        if let Some(slot) = self.0.get_mut(table.index()) {
            *slot = capacity;
        }
    }
}

/// What one record's growth estimate on an index-less copy priced each of
/// the state's hash tables at, and what the record can remove from it: the
/// inputs of the copy's bucket bound's raise ([`CopyBucketBound::applied`]),
/// answered by [`estimate_replay_record_growth_bounded`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PricedBuckets {
    /// Per state table: its length before the record plus the record's
    /// insert calls on it (on a reset, the calls alone: the table is emptied
    /// first).
    wanted: [u64; RebuildTable::STATE_COUNT],
    /// Per state table: its true length before the record, a reset's
    /// included (unlike [`Self::wanted`]'s): with `calls` and the length
    /// after, what bounds the keys the record removed
    /// ([`Self::removals_after`]).
    len_before: [u64; RebuildTable::STATE_COUNT],
    /// Per state table: the record's insert calls on it, an upper bound on
    /// the keys the record adds to it (the premise its growth is priced on);
    /// read by [`Self::removals_after`].
    calls: [u64; RebuildTable::STATE_COUNT],
    /// Per state table: the bucket count the record's priced growth reaches,
    /// doubled from the bound; `None` when the calls fit the room
    /// (`capacity() − len`), so the table cannot grow.
    grown: [Option<u64>; RebuildTable::STATE_COUNT],
    /// Per state table: an upper bound on the keys the record removes from
    /// it — its own deletes, its cascades and purges, and the children lists
    /// the per-record normalization drops (`domain::replay::normalize_children`).
    removals: [u64; RebuildTable::STATE_COUNT],
}

impl PricedBuckets {
    /// `table`, holding `len_before` keys at `capacity` and priced as holding
    /// `len` (0 on a reset), takes `calls` insert calls, its growth priced to
    /// `grown_buckets` (its bound's own when the calls fit).
    fn price(
        &mut self,
        table: RebuildTable,
        len_before: usize,
        len: usize,
        capacity: usize,
        calls: u64,
        grown_buckets: u64,
    ) {
        let index = table.index();
        let wanted = (len as u64).saturating_add(calls);
        if index < RebuildTable::STATE_COUNT {
            self.wanted[index] = wanted;
            self.grown[index] = (wanted > capacity as u64).then_some(grown_buckets);
            self.len_before[index] = len_before as u64;
            self.calls[index] = calls;
        }
    }

    /// The keys the record removed from `table`, bounded once it applied and
    /// the table holds `len_after`: at most [`Self::removals`], and at most
    /// `len_before + calls − len_after` — the removals are exactly the length
    /// before plus the keys the record added less the length after (a key
    /// added and removed within the record counts on both sides), and it
    /// adds no more keys than it makes insert calls, the premise its growth
    /// is priced on.
    pub fn removals_after(&self, table: RebuildTable, len_after: usize) -> u64 {
        let index = table.index();
        let (Some(&len_before), Some(&calls)) = (self.len_before.get(index), self.calls.get(index))
        else {
            return 0;
        };
        self.removals(table).min(
            len_before
                .saturating_add(calls)
                .saturating_sub(len_after as u64),
        )
    }

    /// `table` can lose up to `removals` keys to the record.
    fn may_remove(&mut self, table: RebuildTable, removals: u64) {
        if let Some(slot) = self.removals.get_mut(table.index()) {
            *slot = removals;
        }
    }

    /// The keys the record can remove from `table` (an upper bound).
    pub fn removals(&self, table: RebuildTable) -> u64 {
        self.removals.get(table.index()).copied().unwrap_or(0)
    }

    /// The bucket count the record's growth of `table` was priced to, or
    /// `None` when its calls fit its room.
    pub fn grown_buckets(&self, table: RebuildTable) -> Option<u64> {
        self.grown.get(table.index()).copied().flatten()
    }
}

/// The sampled audit's index-less copy's **bucket bound**: per hash table of
/// the state, a bucket count `b` the table is known not to exceed, and `t`, a
/// bound on the tombstones it holds if it holds exactly `b` buckets. The
/// copy holds no plan indexes, so no capacity high-water: its tables'
/// `capacity()` can hide buckets behind the tombstones the clone kept and
/// the records its repair replays leave. Each record's growth is priced from
/// `b` ([`estimate_replay_record_growth_bounded`]), never from `capacity()`.
/// A stack value on the copy's `CopyWeight` (`storage::structural_cache_repair`),
/// dropped with the audit's repair; it weighs nothing.
///
/// The invariant, per table, with `B` its true bucket count and `T` its true
/// tombstones: `B ≤ b`, and `B = b` implies `T ≤ t`. It starts at the held
/// entry's held capacities ([`Self::new`], [`CacheEntry::held_state_capacities`]),
/// with `t = full(b) − capacity()` (`full` a bucket count's usable slots).
/// After each record [`Self::applied`] moves it, from the record's removals
/// `R` ([`PricedBuckets::removals_after`]):
///
/// - `R = 0`: `b` rises only to what `capacity()` then shows, exact after a
///   growth; `t` is kept, or becomes 0 when the table grew;
/// - `R > 0`: `b` rises to the bucket count the record's growth was priced
///   to only when a growth past `b` was possible (`len + calls + t ≥
///   full(b)`) and the capacity the record left is one a table grown past
///   `b` can read (`capacity() ≥ full(b⁺) − R`, `b⁺` the next bucket count
///   up: `2b`, or 4 from none), `t` then becoming `R`; otherwise `b` stays
///   and `t` grows by `R`;
/// - either way `t` is at most `full(b) − capacity()`.
///
/// The full argument — why each step keeps the invariant, how often a growth
/// that never happens can raise `b` (a cost in coverage, never memory), and
/// why the charge covers the copy — is SDD §21.2 *Sampled audit* and
/// [`estimate_replay_record_growth_bounded`]'s doc. The removals rest on a
/// hand-kept list of the engine's removal paths: a new engine removal path
/// must be added to `may_remove`'s list in the record's estimate; the
/// test-support shadow check asserts the bound after every copy record
/// (`CopyBucketBound::assert_holds_for_test`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyBucketBound {
    /// Per state table: `full(b)`, the usable slots of the bounded bucket
    /// count (a capacity that reads exactly `b`).
    bound: HeldCapacities,
    /// Per state table: `t`.
    tombstones: HeldCapacities,
}

impl CopyBucketBound {
    /// The bound of a copy whose state is `state`, cloned from an entry whose
    /// tables were weighed at `start` ([`CacheEntry::held_state_capacities`]):
    /// `b` from `start` (never below `state`'s own reading), `t = full(b) −
    /// capacity()`.
    pub fn new(state: &CompactState, start: HeldCapacities) -> Self {
        let live = HeldCapacities::live(state);
        let mut bound = HeldCapacities::default();
        let mut tombstones = HeldCapacities::default();
        for table in RebuildTable::ALL {
            if !table.in_state() {
                continue;
            }
            let capacity = live.get(table);
            let full = full_of(table_buckets(start.get(table).max(capacity) as u64));
            bound.set(table, full);
            tombstones.set(table, full.saturating_sub(capacity));
        }
        Self { bound, tombstones }
    }

    /// The bound, per state table, as a capacity that reads exactly its
    /// bucket count `b` (`alloc_model::table_bytes` of it is the table at
    /// `b` buckets).
    pub fn capacities(&self) -> HeldCapacities {
        self.bound
    }

    /// The tombstone bound `t`, per state table (type docs).
    pub fn tombstones(&self) -> HeldCapacities {
        self.tombstones
    }

    /// **Test support.** Assert the bound's invariant (type docs) on
    /// `state`, each state table read by the true-bucket oracle
    /// (`domain::types::true_capacity_for_test`): its true bucket count `B`
    /// is at most `b`, and at `B = b` its true tombstones `T` — its true
    /// capacity less its `capacity()` — are at most `t`. The copy's twin of
    /// the shadow check's weight floor on an indexed entry; the repair runs
    /// it after every record of the sampled audit's copy while the check is
    /// on (`storage::structural_cache_repair::RepairHooks`), so a removal path
    /// the record's estimate does not list fails here. `context` names the
    /// point checked. O(the tables): one clone and drain of each.
    #[cfg(any(test, feature = "test-support"))]
    pub fn assert_holds_for_test(&self, state: &CompactState, context: &str) {
        for (table, capacities) in crate::domain::plan_index::state_hash_tables_for_test(state) {
            let true_buckets = table_buckets(capacities.true_capacity as u64);
            let b = table_buckets(self.bound.get(table) as u64);
            let t = self.tombstones.get(table);
            let true_tombstones = capacities.true_capacity.saturating_sub(capacities.capacity);
            assert!(
                true_buckets <= b,
                "copy bound check, {context}: the {table:?} table holds B = {true_buckets} \
                 buckets, above its bound b = {b} (t = {t}, T = {true_tombstones})"
            );
            if true_buckets == b {
                assert!(
                    true_tombstones <= t,
                    "copy bound check, {context}: the {table:?} table, at its bound b = {b} \
                     buckets (B = {true_buckets}), holds T = {true_tombstones} tombstones, above \
                     its tombstone bound t = {t}"
                );
            }
        }
    }

    /// A record priced at `priced` ([`estimate_replay_record_growth_bounded`],
    /// from this bound) was applied, and normalized, leaving `state`: move
    /// each table's `b` and `t` by the type docs' rule. O(1).
    pub fn applied(&mut self, priced: &PricedBuckets, state: &CompactState) {
        for table in RebuildTable::ALL {
            let Some((len_after, after)) = state_table_shape(table, state) else {
                continue;
            };
            let full = self.bound.get(table);
            let buckets = table_buckets(full as u64);
            let shown = table_buckets(after as u64);
            let tombstones = self.tombstones.get(table);
            // The record's removals, bounded by the table's length change.
            let removals =
                usize::try_from(priced.removals_after(table, len_after)).unwrap_or(usize::MAX);
            let wanted = priced.wanted.get(table.index()).copied().unwrap_or(0);
            // The fewest usable slots a table grown past `b` reads after the
            // record: its last resize, to at least the next bucket count up,
            // came inside the record and left no tombstone, so at most the
            // record's removals since.
            let past_b = full_of(table_buckets(bucket_capacity(buckets).saturating_add(1)).max(4))
                .saturating_sub(removals);
            let (grown, held_tombstones) = match priced.grown_buckets(table) {
                // Removes nothing: `capacity()` shows any growth exactly.
                _ if removals == 0 => {
                    let grown = buckets.max(shown);
                    (grown, if grown == buckets { tombstones } else { 0 })
                }
                // Can remove, a growth past `b` was possible, and the
                // capacity the record left is one a table grown past `b`
                // can read: raised to the bucket count the growth was priced
                // to from `b`.
                Some(priced_to)
                    if wanted.saturating_add(tombstones as u64) >= full as u64
                        && after >= past_b =>
                {
                    (priced_to.max(shown), removals)
                }
                // Can remove, and the table did not grow past `b`.
                _ => (buckets.max(shown), tombstones.saturating_add(removals)),
            };
            let full = full_of(grown);
            self.bound.set(table, full);
            self.tombstones
                .set(table, held_tombstones.min(full.saturating_sub(after)));
        }
    }
}

/// The usable slots of `buckets` buckets, as a `usize` capacity.
fn full_of(buckets: u64) -> usize {
    usize::try_from(bucket_capacity(buckets)).unwrap_or(usize::MAX)
}

/// What one replayed stream record can allocate beside the cache entry it
/// replays onto, before it is applied ([`estimate_replay_record_growth`]):
/// the stream repair's per-record charge point, RP2
/// (`storage::structural_cache_repair`), holds it with the entry's weight.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayRecordGrowth {
    /// Bytes, priced 1:1, that the entry can keep once the record is applied:
    /// the tables, children lists and name-index vectors the record grows,
    /// and the lists a `ReorderChildren` installs. The sampled audit's copy
    /// adds it to the growth it holds for the records replayed so far
    /// (`storage::structural_cache_repair`, `CopyWeight`).
    pub growth: u64,
    /// Bytes, priced 1:1, that the record's apply allocates and frees before
    /// it returns: its cascades' walks, sets and purge
    /// ([`cascade_scratch_bytes`]), a reset's registrations
    /// ([`reset_scratch_bytes`]) and, through the list-delete sink, the
    /// sink's ids and its one mutation in flight
    /// ([`list_delete_sink_bytes`]) and the `SetChildren` copies it builds.
    /// RP2 holds it with `growth` while the record applies; it never enters
    /// the audit copy's held growth.
    pub scratch: u64,
    /// The engine's per-record scratch, in counts, priced at the cached
    /// route's rewrite prices: the listings of the parents it touches
    /// (`list_entries`), its depth check's walk (`walked_sections`) and, on an
    /// entry without plan indexes, the per-delta name and adoption indexes
    /// (`indexed_sections`, `adoption_entries`). Rows and rollback rebuilds
    /// stay 0: a replay emits rows only into a sink it drops, and has no
    /// journal.
    pub rewrite: RewriteCount,
}

/// What replaying one decoded record, `delta` (`None`: an empty commit),
/// onto `state` — a normalized cache entry's state, with its plan indexes
/// `indexes` when it holds them — can allocate beside the entry before the
/// record's own children-list normalization runs: RP2's growth and rewrite
/// terms (`storage::structural_cache_repair`, "The charge points"). Pure
/// arithmetic over the state's lengths and capacities and the record's
/// commands, O(commands + the depth walk's hops up) plus the subtree walks
/// of a record that can cascade (below). It allocates tallies that grow
/// with the record's commands, not with the state: `parent_after`, `born`,
/// `IntroducedNames::names`, the `ReplayTally` maps and `listed`, and
/// `incremental_walk_bound`'s `seen` set — roughly 150 to 250 B per
/// `CreateSection` [model]. They are allocated before RP2 holds this
/// estimate, so RP2 cannot cover them; RP1's hold for the record's chunk
/// does, since it carries the largest record's parse at `k_parse`
/// (`grpc::upload_admission::PARSE_EXPANSION_CENTI`, 15.88 per byte of the
/// delta part: about 635 to 1,270 B per `CreateSection` at 40 to 80 B of
/// OODF each), and the charge only grows until the plan's peak. T7's
/// `fsz-repair-1mib-sections-record` row (one record of about 1 MiB of new
/// sections) pins it: every phase of its repair, this estimate's included,
/// stays within the charge in force. `list_deletes` is whether the record
/// replays through the list-delete sink (`MutationSink::ListDeletes`: the
/// entry has dangling parents).
///
/// Every stored entry is compacted (`compact_and_weigh`) and every list a
/// replay touches is trimmed to its length after each record
/// (`domain::plan_index::normalize_touched`), so a record meets tables and
/// lists with no spare room: the first insert into a table at its load limit
/// doubles it, and the first child added under a parent of `L` children
/// reallocates its list to `2L`, old and new alive at once. None of that
/// scales with the record's delta bytes, so the repair's per-byte growth
/// term cannot cover it; this prices it per record.
///
/// `growth`, in bytes, 1:1:
///
/// - **Tables**: [`estimate_in_place_growth_bytes`] at the record's
///   [`PlannedInserts`] — on a compacted table at its load limit, one insert
///   prices the doubled table. With plan indexes each table is priced from
///   its held capacity (`PlanIndexes::held_capacity`), so the earlier
///   records' tombstones cannot make it double from fewer buckets than it
///   holds; the old table is in the entry's weight, read at the same held
///   capacity. The sampled audit's index-less copy prices each from its
///   bucket bound instead ([`estimate_replay_record_growth_bounded`]).
/// - **Children lists**: for each parent the record lists ids under (a
///   `CreateSection`'s or a born `DiffChangeSection`'s parent, a
///   `MoveSection`'s or a reparenting `DiffChangeSection`'s destination, a
///   birth's adopted orphans), [`vec_growth_bytes`] of its list by that many
///   pushes of 8 B.
/// - **Name-index vectors** (an entry with plan indexes): for each parent the
///   record names rows under (the same parents, less the adopted orphans,
///   which already name theirs), [`vec_growth_bytes`] of its vector by that
///   many entries, from `PlanIndexes::name_vector_room` — these vectors are
///   not trimmed, so they can have room.
/// - **A `ReorderChildren`** installs a list built beside the old one: the
///   new list's bytes (`list_bytes`) at its parent's length plus its pushes.
///
/// `scratch`, in bytes, 1:1 — what the apply allocates and frees before it
/// returns, none of which the entry keeps:
///
/// - **Cascades** ([`cascade_scratch_bytes`]), when the record can cascade at
///   all: it carries a `DeleteSection` or a `MoveSection` (edges 2–4 delete
///   the moved section, edge 5 the destination's same-name sibling), a
///   pre-record row sits at a `(parent, name)` that a `CreateSection`, a
///   `DiffChangeSection` or a `MoveSection` destination introduces (the
///   same-name sibling a birth cascades), or two of those commands introduce
///   the same `(parent, name)`. Otherwise it is 0. The pushes `P` are the
///   births plus [`subtree_pushes`] summed over the roots the record can
///   delete, on the state before it: each `DeleteSection` target with a row,
///   each `MoveSection` id, each `DiffChangeSection` id with a row, each
///   pre-record row at an introduced `(parent, name)` other than the
///   introducing command's own id (with plan indexes found by
///   `PlanIndexes::rows_named`, without them by the one pass over the
///   sections that also counts the orphans), and each orphan a birth adopts.
///   That is every subtree a cascade can reach: a list gains entries only
///   through `insert_child_into_parent`, which serves births, moves,
///   reparents and adoptions, and each of those is a root or a birth here.
///   The walks stop once their pushes beyond their roots pass the state's
///   sections (a well-formed tree lists each section once, so walks of
///   disjoint subtrees stay within that), or at a walk deeper than
///   `SUBTREE_WALK_FRAMES` levels; `P` is then the whole state's: the
///   roots, every list entry the state holds (the plan indexes' list bytes
///   over 8 B with them, the lists' lengths summed without), every entry the
///   record pushes, and the births.
/// - **A reset** ([`reset_scratch_bytes`]) at the state's sections and
///   lists. A reset that can also cascade fills the same plan sets, so each
///   set is priced at its whole insert count: the reset's sections plus `P`,
///   its lists plus the record's touched parents.
/// - **The list-delete sink** ([`list_delete_sink_bytes`]), with
///   `list_deletes`: its ids — one per list a cascade deletes (`P`), per
///   section and list a reset deletes, and per touched parent — and its one
///   mutation in flight; and each touched parent's `SetChildren`, built with
///   a copy of its final list (delta.rs:2951) and dropped, at its length plus
///   its pushes.
///
/// `rewrite`, in counts (zero for a record that cannot change the section
/// tree: an empty commit, or one with no reset and no structural command
/// but `DiffDeleteSection`, which also skips the depth walk):
///
/// - **`list_entries`**: each touched parent's listing — the engine's
///   per-parent membership set, order vector and child registrations — at its
///   length plus its pushes, as a plan's listing is projected.
/// - **`walked_sections`**, the depth check: with plan indexes that passed
///   the full walk (`depth_ok`) and no reset, the incremental walk's hops up
///   from each listed child to the root, each section counted once as the
///   walk's memo counts it (a path past [`MAX_SECTION_TREE_DEPTH`] + 1 hops
///   falls back to the full walk, priced as below; each adopted orphan is
///   one more hop), plus its walk down: one per listed child (adopted
///   orphans included) and the record's births, and every section once more
///   when a command can carry a subtree into the walk (a move, a reparent, a
///   birth adopting orphans). Otherwise — no indexes, indexes that did not
///   pass, or a reset — the full walk: every section plus the births.
/// - **Without plan indexes** (the sampled audit's copy; on the charged route
///   only a test's), the engine's per-delta indexes as well:
///   `indexed_sections`, the sections plus the record's births and moves,
///   when it probes a name; `adoption_entries`, every section, when it
///   births.
///
/// Not here: the per-element memory of the record's own rows — what it
/// writes from its set fields, time samples and section commands, in the
/// entry and in a mutation the sink builds — which the repair's per-byte
/// growth term carries.
pub fn estimate_replay_record_growth(
    state: &CompactState,
    indexes: Option<&PlanIndexes>,
    delta: Option<&oodf_ffi::OwnedDelta>,
    list_deletes: bool,
) -> ReplayRecordGrowth {
    replay_record_growth(state, indexes, None, delta, list_deletes).0
}

/// [`estimate_replay_record_growth`] on the sampled audit's index-less copy,
/// each of its state's hash tables priced from the copy's bucket bound
/// `bound` ([`CopyBucketBound`]) rather than from its `capacity()`, which the
/// copy's tombstones can hide buckets from: RP2's estimate on that copy
/// (`storage::structural_cache_repair`, `unindexed_record_growth`). Also
/// answers what the bound's raise reads once the record is applied
/// ([`PricedBuckets`]: each table's length plus calls, the bucket count its
/// growth was priced to, and the keys the record can remove from it,
/// [`CopyBucketBound::applied`]).
///
/// Per table, `growth` prices the doublings from the bound's bucket count
/// `b` ([`table_growth_bounded`]): one doubling, the table grown to; several,
/// the last two less the current table at `capacity()`'s reading. The
/// copy's held term is its stored weight plus the growth priced so far
/// (`CopyWeight`'s `W0 + G`).
///
/// **Why `G` covers the copy (soundness).** Two invariants hold per table
/// after every record: `b` is at or above the table's true bucket count `B`
/// ([`CopyBucketBound`]'s raise rule keeps it), and the charge holds the
/// table at `b` — the copy, which the audit's line term funds
/// (`storage::structural_cache_audit`, "Memory"), plus `G`. The second holds
/// for two reasons:
///
/// - **At the trigger, `b` is the true bucket count.** `b` starts at the
///   committed entry's held capacity, read from its plan indexes' capacity
///   high-water ([`CacheEntry::held_state_capacities`]). The entry always
///   holds its indexes there: the audit is spawned only after
///   `CheckedOut::advance_committed`, which settles a plan made in place,
///   and a plan in place builds the indexes of an entry that has none
///   (`CheckedOut::plan_in_place`). A held capacity reads above the truth
///   only after a rollback rebuilt a table smaller, and every rollback is
///   followed by a compacting put-back before the entry can commit again.
/// - **`b` passes `B` only by a raise to a priced growth**, whose bytes, at
///   least the raised table's, `G` already holds.
///
/// **One doubling is priced `bytes(2b)`, the grown table alone, and the
/// second invariant is what makes that sound.** From a true `B < b` (so `2B ≤
/// b`, both powers of two) a single doubling reaches at most `2B ≤ b`; at `B
/// = b` it reaches `2b`. Either way the table it grows from, at most `b`, is
/// one the charge already holds, and the table it grows to is at most `2b`.
///
/// **Several doublings in one record** are priced as the last two tables —
/// grown from `b`, to `max(len + calls, full(b) + 1)` slots, at least the
/// `max(items + 1, full(B) + 1)` a growth from `B ≤ b` reaches, the items
/// never above `len + calls` — less the current table read at `capacity()`,
/// which is never above `B`, and the charge holds every table the copy has
/// at its true size at least (the copy's funding, plus `G` for any table
/// grown since). This branch needs only that, not that the charge holds the
/// table at `b`: subtracting at the `capacity()` reading rather than at `b`
/// is the conservative choice.
///
/// What is monotone in the starting bucket count is the held table plus its
/// growth — `bytes(b) + bytes(2b)` for one doubling, `bytes(L) + bytes(L/2)`
/// for several, `L` the last table grown to — **not the growth term alone**,
/// which on the several-doublings branch, `bytes(L) + bytes(L/2) −
/// bytes(current)`, falls as the current table grows.
pub fn estimate_replay_record_growth_bounded(
    state: &CompactState,
    bound: &CopyBucketBound,
    delta: Option<&oodf_ffi::OwnedDelta>,
    list_deletes: bool,
) -> (ReplayRecordGrowth, PricedBuckets) {
    replay_record_growth(state, None, Some(&bound.capacities()), delta, list_deletes)
}

/// [`estimate_replay_record_growth`], each state table's growth priced from
/// `bound` when given ([`estimate_replay_record_growth_bounded`]), and what
/// the bound's raise reads.
fn replay_record_growth(
    state: &CompactState,
    indexes: Option<&PlanIndexes>,
    bound: Option<&HeldCapacities>,
    delta: Option<&oodf_ffi::OwnedDelta>,
    list_deletes: bool,
) -> (ReplayRecordGrowth, PricedBuckets) {
    use oodf_ffi::StructCommand;

    let count = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
    let inserts = PlannedInserts::of(delta);
    let (tables, mut priced) = in_place_growth(state, indexes, bound, inserts);
    // Any record through the list-delete sink builds its mutations one at a
    // time, whatever it carries.
    let sink_alone = if list_deletes {
        list_delete_sink_bytes(0, 0)
    } else {
        0
    };
    // An empty commit, and a record with no reset and no structural command
    // but `DiffDeleteSection` (which the engine ignores), remove nothing:
    // only a cascade, a reset and the children normalization after a
    // structural command do, and a normalized state's lists stay normalized
    // when no list is written. So `priced` keeps no removals.
    let Some(delta) = delta else {
        return (
            ReplayRecordGrowth {
                growth: tables,
                scratch: sink_alone,
                rewrite: RewriteCount::default(),
            },
            priced,
        );
    };
    let reset = delta.is_diff && delta.base_version == 0;
    let changes_tree = reset
        || delta
            .struct_commands
            .iter()
            .any(|command| !matches!(command, StructCommand::DiffDeleteSection(_)));
    if !changes_tree {
        return (
            ReplayRecordGrowth {
                growth: tables,
                scratch: sink_alone,
                rewrite: RewriteCount::default(),
            },
            priced,
        );
    }

    let sections = count(state.sections.len());
    let mut tally = ReplayTally::default();
    let mut reordered: Vec<u64> = Vec::new();
    // Each moved or born id's parent after the record, in command order.
    let mut parent_after: HashMap<u64, u64> = HashMap::new();
    let mut born: Vec<u64> = Vec::new();
    let mut probes = false;
    let mut moves: u64 = 0;
    // A command that can carry an existing subtree into the walk down.
    let mut carries_subtree = false;
    // The cascades' roots the commands name (each `DeleteSection` target
    // with a row, each `MoveSection` id, each `DiffChangeSection` id with a
    // row), the `(parent, name)`s they introduce, and whether the record
    // can cascade whatever the state holds at those names.
    let mut commanded: HashSet<u64> = HashSet::new();
    let mut introduced = IntroducedNames::default();
    let mut deletes_or_moves = false;

    let row_parent = |id: u64| state.sections.get(&id).map(|row| row.parent_id);
    for command in &delta.struct_commands {
        match command {
            StructCommand::CreateSection(c) => {
                probes = true;
                tally.list_under(c.parent_id, c.section_id);
                parent_after.insert(c.section_id, c.parent_id);
                born.push(c.section_id);
                introduced.introduce(c.parent_id, &c.section_name, c.section_id);
            }
            StructCommand::DiffChangeSection(c) => {
                probes = true;
                introduced.introduce(c.parent_id, &c.section_name, c.section_id);
                match row_parent(c.section_id) {
                    // A rename in place re-files its entry in the same
                    // vector, counted as one more entry; a reparent lists it
                    // under its destination, carrying its subtree.
                    Some(old_parent) => {
                        commanded.insert(c.section_id);
                        tally.touched.insert(old_parent);
                        if old_parent == c.parent_id {
                            tally.name_under(c.parent_id);
                        } else {
                            tally.list_under(c.parent_id, c.section_id);
                            carries_subtree = true;
                        }
                    }
                    None => {
                        tally.list_under(c.parent_id, c.section_id);
                        born.push(c.section_id);
                    }
                }
                parent_after.insert(c.section_id, c.parent_id);
            }
            StructCommand::MoveSection(c) => {
                probes = true;
                moves = moves.saturating_add(1);
                carries_subtree = true;
                deletes_or_moves = true;
                commanded.insert(c.section_id);
                introduced.introduce(c.new_parent_id, &c.new_name, c.section_id);
                tally.touched.insert(c.old_parent_id);
                if let Some(old_parent) = row_parent(c.section_id) {
                    tally.touched.insert(old_parent);
                }
                tally.list_under(c.new_parent_id, c.section_id);
                parent_after.insert(c.section_id, c.new_parent_id);
            }
            StructCommand::ReorderChildren(c) => {
                tally.touched.insert(c.section_id);
                reordered.push(c.section_id);
            }
            StructCommand::DeleteSection(c) => {
                deletes_or_moves = true;
                tally.touched.insert(c.parent_id);
                if let Some(old_parent) = row_parent(c.section_id) {
                    commanded.insert(c.section_id);
                    tally.touched.insert(old_parent);
                }
            }
            StructCommand::DiffDeleteSection(_) => {}
        }
    }

    // A birth adopts the rows already naming its id: each is listed under
    // the newborn (they name it already, so no name-index entry moves), and
    // its subtree joins the walk. With plan indexes their count is the
    // newborn's name-index vector; without, one pass over the sections
    // counts them for every birth at once.
    //
    // The same lookups find the cascades' other roots, each walked as it is
    // found ([`SubtreeWalks`]), the commanded ones excepted (walked once,
    // below): the rows at an introduced `(parent, name)` other than its
    // introducer — with plan indexes through `rows_named`, without them in
    // the same pass that counts the orphans — and the orphans themselves,
    // which a cascade of their newborn takes along. An orphan's walk is
    // spent for nothing when the record turns out unable to cascade; the
    // root `0` adopts nothing (`adopt_orphans`), so its rows are no
    // orphans' roots.
    let mut walks = SubtreeWalks::new(state);
    let mut conflicts = false;
    let orphans_of: HashMap<u64, u64> = if let Some(indexes) = indexes {
        for (&(parent, name), &own) in &introduced.names {
            for id in indexes.rows_named(&state.sections, parent, name) {
                if own != Some(id) {
                    conflicts = true;
                    walks.root_unless(&commanded, id);
                }
            }
        }
        born.iter()
            .map(|&id| {
                if id != 0 {
                    for orphan in indexes.rows_naming(id) {
                        walks.root_unless(&commanded, orphan);
                    }
                }
                (id, count(indexes.name_vector_room(id).0))
            })
            .collect()
    } else {
        let mut orphans: HashMap<u64, u64> = born.iter().map(|&id| (id, 0)).collect();
        if !orphans.is_empty() || !introduced.names.is_empty() {
            for row in state.sections.values() {
                let mut root = false;
                if let Some(n) = orphans.get_mut(&row.parent_id) {
                    *n = n.saturating_add(1);
                    root = row.parent_id != 0;
                }
                if introduced.conflicts_with(row) {
                    conflicts = true;
                    root = true;
                }
                if root {
                    walks.root_unless(&commanded, row.section_id);
                }
            }
        }
        orphans
    };
    let can_cascade = deletes_or_moves || introduced.shared || conflicts;
    // What the record can remove from each state table, for the copy's
    // bucket bound (`CopyBucketBound::applied`). A reset or a cascade can
    // remove any key a table holds or the record inserts. Otherwise only the
    // children normalization after the record drops a list: the old
    // parent's list a `DiffChangeSection` reparent empties, or a list a
    // `ReorderChildren` re-derives empty — at most one per such command. A
    // list is created only under a live parent or the root, and a birth's
    // own list only for the orphans it adopts, so no other list ends empty
    // or dangling. A new engine removal path must be added here
    // (`PricedBuckets::may_remove`); the test-support shadow check asserts
    // the bound after every copy record.
    if bound.is_some() {
        let normalized_lists = count(
            delta
                .struct_commands
                .iter()
                .filter(|command| {
                    matches!(
                        command,
                        StructCommand::DiffChangeSection(_) | StructCommand::ReorderChildren(_)
                    )
                })
                .count(),
        );
        for table in RebuildTable::ALL {
            let Some((len, _)) = state_table_shape(table, state) else {
                continue;
            };
            let removals = if reset || can_cascade {
                count(len).saturating_add(inserts.calls(table))
            } else if table == RebuildTable::Children {
                normalized_lists
            } else {
                0
            };
            priced.may_remove(table, removals);
        }
    }
    // Each adopted orphan is listed too: one more hop up (to its newborn,
    // which its own listing walks) and one more root down.
    let mut adopted: u64 = 0;
    for (&newborn, &orphans) in &orphans_of {
        if orphans > 0 {
            *tally.pushes.entry(newborn).or_insert(0) += orphans;
            tally.touched.insert(newborn);
            adopted = adopted.saturating_add(orphans);
            carries_subtree = true;
        }
    }
    let ReplayTally {
        pushes,
        name_pushes,
        touched,
        listed,
    } = tally;

    // A parent's list after the record, at most: its length plus its pushes.
    let list_len = |parent: u64| {
        let len = state
            .children
            .get(&parent)
            .map_or(0, |list| count(list.len()));
        len.saturating_add(pushes.get(&parent).copied().unwrap_or(0))
    };
    let usize_of = |n: u64| usize::try_from(n).unwrap_or(usize::MAX);
    let mut growth = tables;
    for (&parent, &n) in &pushes {
        let (len, capacity) = state
            .children
            .get(&parent)
            .map_or((0, 0), |list| (list.len(), list.capacity()));
        growth = growth.saturating_add(vec_growth_bytes(len, capacity, n, size_of::<u64>() as u64));
    }
    if let Some(indexes) = indexes {
        for (&parent, &n) in &name_pushes {
            let (len, capacity) = indexes.name_vector_room(parent);
            growth = growth.saturating_add(vec_growth_bytes(
                len,
                capacity,
                n,
                PlanIndexes::NAME_ENTRY_BYTES as u64,
            ));
        }
    }
    for &parent in &reordered {
        growth = growth.saturating_add(list_bytes(usize_of(list_len(parent))));
    }
    let mut list_entries: u64 = 0;
    // Through the list-delete sink, each touched parent's `SetChildren` is
    // built with a copy of its final list and dropped.
    let mut scratch: u64 = 0;
    for &parent in &touched {
        let len = list_len(parent);
        list_entries = list_entries.saturating_add(len);
        if list_deletes {
            scratch = scratch.saturating_add(list_bytes(usize_of(len)));
        }
    }

    let births = count(born.len());

    // The cascades' pushes: none for a record that cannot cascade; else the
    // commanded roots' walks join the others, and with the births they
    // bound every push (the doc's root enumeration). Past the walks' room,
    // the whole state's list entries instead: O(1) from the plan indexes'
    // list bytes (each list's allocation holds at least 8 B per entry), one
    // pass over the lists without them.
    let record_pushes = pushes.values().fold(0u64, |sum, &n| sum.saturating_add(n));
    let cascade_pushes = if can_cascade {
        for &root in &commanded {
            walks.root(root);
        }
        let whole_state_entries = || match indexes {
            Some(indexes) => indexes.lists_bytes() / size_of::<u64>() as u64,
            None => state
                .children
                .values()
                .fold(0u64, |sum, list| sum.saturating_add(count(list.len()))),
        };
        walks
            .pushes(whole_state_entries, record_pushes)
            .saturating_add(births)
    } else {
        0
    };
    let hooked = indexes.is_some();
    let touched_lists = count(touched.len());
    let names = state.fields.names();
    let name_room = (names.len(), names.free_len(), names.free_capacity());
    let lists = count(state.children.len());
    if reset {
        // One plan's sets take the reset's and the cascades' inserts alike,
        // so each is priced at their sum (`reset_scratch_bytes`).
        scratch = scratch
            .saturating_add(reset_scratch_bytes(
                sections.saturating_add(cascade_pushes),
                lists.saturating_add(touched_lists),
                hooked,
            ))
            .saturating_add(cascade_scratch_bytes(
                cascade_pushes,
                touched_lists.saturating_add(lists),
                hooked,
                name_room,
            ));
    } else {
        scratch = scratch.saturating_add(cascade_scratch_bytes(
            cascade_pushes,
            touched_lists,
            hooked,
            name_room,
        ));
    }
    if list_deletes {
        let reset_deletes = if reset {
            sections.saturating_add(lists)
        } else {
            0
        };
        scratch = scratch.saturating_add(list_delete_sink_bytes(
            cascade_pushes.saturating_add(reset_deletes),
            touched_lists,
        ));
    }
    let full_walk = sections.saturating_add(births);
    let walked_sections = match indexes {
        Some(indexes) if indexes.depth_ok() && !reset => {
            incremental_walk_bound(state, &listed, &parent_after).map_or(full_walk, |up| {
                let down = count(listed.len())
                    .saturating_add(adopted)
                    .saturating_add(births)
                    .saturating_add(if carries_subtree { full_walk } else { 0 });
                up.saturating_add(adopted).saturating_add(down)
            })
        }
        _ => full_walk,
    };
    let (indexed_sections, adoption_entries) = if indexes.is_none() {
        (
            if probes {
                sections.saturating_add(births).saturating_add(moves)
            } else {
                0
            },
            if births > 0 { sections } else { 0 },
        )
    } else {
        (0, 0)
    };
    (
        ReplayRecordGrowth {
            growth,
            scratch,
            rewrite: RewriteCount {
                list_entries,
                indexed_sections,
                adoption_entries,
                walked_sections,
                ..RewriteCount::default()
            },
        },
        priced,
    )
}

/// The `(parent, name)`s a record's `CreateSection`s, `DiffChangeSection`s
/// and `MoveSection` destinations introduce ([`estimate_replay_record_growth`]):
/// a row already there, other than the introducing command's own, is a
/// same-name sibling the command can cascade-delete.
#[derive(Default)]
struct IntroducedNames<'d> {
    /// Each introduced `(parent, name)` → the id of the one command's section
    /// that introduces it, or `None` once commands for two different ids do.
    names: HashMap<(u64, &'d str), Option<u64>>,
    /// Two commands for different ids introduce one `(parent, name)`: the
    /// later can cascade the earlier's section, whatever the state held.
    shared: bool,
}

impl<'d> IntroducedNames<'d> {
    /// The command for `own` introduces `(parent, name)`.
    fn introduce(&mut self, parent: u64, name: &'d str, own: u64) {
        match self.names.entry((parent, name)) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(Some(own));
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                if *slot.get() != Some(own) {
                    *slot.get_mut() = None;
                    self.shared = true;
                }
            }
        }
    }

    /// Whether `row` sits at an introduced `(parent, name)` and is not its
    /// one introducer's own section.
    fn conflicts_with(&self, row: &'d Section) -> bool {
        self.names
            .get(&(row.parent_id, row.name.as_str()))
            .is_some_and(|&own| own != Some(row.section_id))
    }
}

/// The subtree walks over the roots a record can cascade-delete
/// ([`estimate_replay_record_growth`]): every root counted, and the pushes
/// beyond them ([`subtree_pushes`], each walk counting its root as one push)
/// summed while they fit `room`, the state's sections; past it, or past a
/// walk's depth limit, the walks stop and the whole state's pushes are
/// priced instead (`whole_state`).
struct SubtreeWalks<'s> {
    state: &'s CompactState,
    roots: u64,
    beyond_roots: u64,
    room: u64,
    whole_state: bool,
}

impl<'s> SubtreeWalks<'s> {
    fn new(state: &'s CompactState) -> Self {
        Self {
            state,
            roots: 0,
            beyond_roots: 0,
            room: u64::try_from(state.sections.len()).unwrap_or(u64::MAX),
            whole_state: false,
        }
    }

    /// Count `root` and walk its subtree, unless the walks already stopped.
    fn root(&mut self, root: u64) {
        self.roots = self.roots.saturating_add(1);
        if self.whole_state {
            return;
        }
        match subtree_pushes(self.state, root, self.room.saturating_add(1)) {
            Some(pushes) => {
                let beyond = pushes.saturating_sub(1);
                self.beyond_roots = self.beyond_roots.saturating_add(beyond);
                self.room = self.room.saturating_sub(beyond);
            }
            None => self.whole_state = true,
        }
    }

    /// [`Self::root`], unless `root` is one of `commanded`, which are walked
    /// once on their own.
    fn root_unless(&mut self, commanded: &HashSet<u64>, root: u64) {
        if !commanded.contains(&root) {
            self.root(root);
        }
    }

    /// The pushes of the walked roots: each root and the pushes beyond it;
    /// once the walks stopped, each root, every list entry of the state
    /// (`whole_state_entries`) and every entry the record pushes onto a list
    /// (`record_pushes`) — each list's entries are pushed at most once over
    /// the record, since a cascade removes the list of every id it visits.
    fn pushes(&self, whole_state_entries: impl FnOnce() -> u64, record_pushes: u64) -> u64 {
        if self.whole_state {
            self.roots
                .saturating_add(whole_state_entries())
                .saturating_add(record_pushes)
        } else {
            self.roots.saturating_add(self.beyond_roots)
        }
    }
}

/// One record's children-list and name-index tallies, per parent
/// ([`estimate_replay_record_growth`]).
#[derive(Default)]
struct ReplayTally {
    /// Ids pushed onto each parent's children list.
    pushes: HashMap<u64, u64>,
    /// Entries put into each parent's name-index vector.
    name_pushes: HashMap<u64, u64>,
    /// Every parent whose list the engine lists (its per-parent listing) or
    /// dirties.
    touched: HashSet<u64>,
    /// `(parent, child)` per id listed, in command order.
    listed: Vec<(u64, u64)>,
}

impl ReplayTally {
    /// `child` is listed under `parent`, a row naming `parent` entering its
    /// name-index vector with it.
    fn list_under(&mut self, parent: u64, child: u64) {
        *self.pushes.entry(parent).or_insert(0) += 1;
        self.name_under(parent);
        self.listed.push((parent, child));
    }

    /// A row naming `parent` is filed into its name-index vector, and its
    /// list is touched.
    fn name_under(&mut self, parent: u64) {
        *self.name_pushes.entry(parent).or_insert(0) += 1;
        self.touched.insert(parent);
    }
}

/// The hops up the incremental depth check
/// (`domain::plan_index::incremental_depth_verdict`) takes for `listed`, the
/// `(parent, child)` listings of one record, after it applies: each listed
/// child and each ancestor up to the root, counted once, as the walk's memo
/// counts them. Parents come from `parent_after` (each moved or born id's
/// parent after the record) and otherwise from `state`'s rows. `None` when a
/// path runs past [`MAX_SECTION_TREE_DEPTH`] + 1 hops: the check then falls
/// back to the full walk. A missing row or a cycle ends a path, as it ends
/// the check's.
fn incremental_walk_bound(
    state: &CompactState,
    listed: &[(u64, u64)],
    parent_after: &HashMap<u64, u64>,
) -> Option<u64> {
    let cap = u64::try_from(MAX_SECTION_TREE_DEPTH)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let parent_of = |id: u64| {
        parent_after
            .get(&id)
            .copied()
            .or_else(|| state.sections.get(&id).map(|row| row.parent_id))
    };
    let mut seen: HashSet<u64> = HashSet::new();
    let mut hops: u64 = 0;
    for &(_, child) in listed {
        let mut cursor = child;
        let mut path: u64 = 0;
        while cursor != 0 && seen.insert(cursor) {
            hops = hops.saturating_add(1);
            path = path.saturating_add(1);
            if path > cap {
                return None;
            }
            match parent_of(cursor) {
                Some(parent) => cursor = parent,
                None => break,
            }
        }
    }
    Some(hops)
}

/// A lower bound on the weight [`estimate_state_bytes`] would charge the
/// compact entry of an object whose indexes hold `cards`, known before any
/// full read (SR-3.3; the cold start's weight check,
/// `storage::structural_cache_cold_start`).
///
/// [`estimate_state_bytes`]'s own per-entry constants for the compact form
/// times the cardinalities, every string, name and hash excluded: the
/// [`CompactState`] itself, and the sections, fields and time-samples tables
/// at the bucket count their cardinality implies. The children table is left
/// out (lists under dangling parents are not loaded, so `obj:chs` over-counts
/// it), as are tombstones, which the state does not hold. Each table the
/// entry holds has at least as many slots as its index has members, and
/// [`table_bytes`] never falls as the slot count grows, so on a well-formed
/// object the bound never exceeds the estimate. On a corrupt index — a
/// member whose hash is missing, which the decoded state leaves out — it
/// can: such an object may be judged over the cap early, which costs it the
/// uncacheable path, never a wrong entry.
pub fn state_weight_lower_bound(cards: &SnapshotCardinalities) -> u64 {
    let slots = |n: u64| usize::try_from(n).unwrap_or(usize::MAX);
    (size_of::<CompactState>() as u64)
        .saturating_add(table_bytes(slots(cards.sections), SECTION_SLOT_BYTES))
        .saturating_add(table_bytes(slots(cards.fields), COMPACT_FIELD_SLOT_BYTES))
        .saturating_add(table_bytes(
            slots(cards.time_samples),
            COMPACT_TIME_SAMPLE_SLOT_BYTES,
        ))
}

/// Shrink each of `state`'s hash tables — `sections`, `children`, `fields`,
/// `time_samples`, its only maps; none is nested — to the smallest bucket
/// count its length needs, and, on the compact form, drop the name interner's
/// trailing free ids and spare capacity (`FieldRows::shrink_to_fit`). The
/// cache calls this on every state it stores
/// before weighing it — in [`StructuralCache::admit`], and in the guard's
/// one settling step (`CheckedOut::settle_with`) behind
/// [`CheckedOut::put_back`], [`CheckedOut::put_back_held`] and the `Drop` of
/// a guard whose entry a repair or plan advanced (`CacheEntry::compact_and_weigh`)
/// — so the weight ([`estimate_state_bytes`], or its O(1) equal
/// [`indexed_state_bytes`]) sees a table with no oversized bucket array.
/// Only a guard dropped with its entry untouched puts it back as it was,
/// already compacted at its admission.
///
/// Removals leave tombstones that a `HashMap` reports as spent capacity, and
/// a table never shrinks on its own; `clone()` keeps the bucket layout. So a
/// state thinned by deletes would otherwise hold — and keep holding, across
/// every commit — about twice the memory its `capacity()` implies.
/// Compacting reallocates only a table whose bucket array is larger than its
/// length needs, which also clears its tombstones and gives the memory back.
///
/// Cost: allocation-neutral and O(1) per table that is already minimal for
/// its length, as a table grown only by inserts is; one O(n) rehash of a
/// table thinned (or preallocated) below what its buckets hold. On the
/// compact form, add one pass over the name interner's free ids
/// (`NameInterner::shrink_to_fit` prunes its free list on every call, an
/// already-minimal entry included): O(free ids), the interner's unused name
/// slots, not the layer's rows. Children
/// lists and strings are not touched: the estimate reads their own
/// capacities, which are exact.
pub fn compact_state<R: ValueRows>(state: &mut ObjectStateOf<R>) {
    state.sections.shrink_to_fit();
    state.children.shrink_to_fit();
    state.fields.shrink_to_fit();
    state.time_samples.shrink_to_fit();
}

/// The per-entry cap on state weight for a configured budget of
/// `budget_bytes` (SR-3.3): a quarter of the budget, but never above
/// `u32::MAX`. A quarter first reaches `u32::MAX` just below a 16 GiB budget
/// (`budget_bytes / 4 == u32::MAX` at 16 GiB − 4 bytes). The cap and the
/// cold start's lower bound on state weight (`state_weight_lower_bound`)
/// decide which layers are cacheable; the plan indexes count against neither.
/// The pod memory plan's audit term counts two entries at it
/// (`config::memory_plan`).
pub fn max_entry_bytes_for(budget_bytes: u64) -> u64 {
    (budget_bytes / 4).min(u64::from(u32::MAX))
}

/// The weight mini-moka charges for an entry of `weight_bytes`, state and
/// plan indexes: its weigher takes a `u32`, so in [`SLOT_UNIT_BYTES`]-byte
/// units, rounded up. A stored entry weighs at most [`held_entry_bound`],
/// never above `u32::MAX` units, so every stored entry is charged in full;
/// saturating at `u32::MAX` rather than wrapping is only a backstop no
/// stored entry reaches.
fn slot_weight(weight_bytes: u64) -> u32 {
    u32::try_from(weight_bytes.div_ceil(SLOT_UNIT_BYTES)).unwrap_or(u32::MAX)
}

/// The most one entry the cache holds may weigh, state and plan indexes, for
/// a resident budget of `resident_bytes`: that budget, or `u32::MAX`
/// [`SLOT_UNIT_BYTES`]-byte units — the most [`slot_weight`] charges in full
/// — when that is lower (from a budget of about 13.2 GiB up). A store evicts
/// a heavier entry at once instead of placing it (`Control::store`), so the
/// resident bound always finishes within its budget and mini-moka never
/// under-charges an entry it holds.
const fn held_entry_bound(resident_bytes: u64) -> u64 {
    let most_charged = u32::MAX as u64 * SLOT_UNIT_BYTES;
    if resident_bytes < most_charged {
        resident_bytes
    } else {
        most_charged
    }
}

/// The heap bytes a field value owns beyond its slot.
fn value_heap_bytes(value: &FieldValue) -> u64 {
    match value {
        FieldValue::Inline(bytes) => allocation_bytes(bytes.capacity() as u64),
        FieldValue::External { .. } => 0,
    }
}

/// The C3 API tests that survive from the stub (plan `b.dkt`, Epic 2) — the
/// variants construct as the contract states, clones share one cache, and an
/// empty cache answers `Miss` / `None` — and the cache's behavioural suite:
/// the epoch check and the pruning of its bookkeeping, put-back on every
/// outcome class, checkout, eviction by budget and idle TTL (never strict LRU
/// order: mini-moka's policy is W-TinyLFU), the per-entry cap and the 32-bit
/// slot weight, compaction before weighing, negative entries, the side store
/// for refused admissions and its entry bound, and clone-before-plan. The
/// estimate's calibration against jemalloc is in the binary's tests
/// (`src/allocator/structural_cache_calibration.rs`).
///
/// Expiry tests use short TTLs passed to [`StructuralCache::with_ttls`] and
/// real sleeps, because mini-moka reads only the real clock; they assert only
/// "gone after the TTL plus slack" and "still there while provably inside the
/// TTL", never a duration.
///
/// The tests that need a [`CheckedOut`] without a lookup build it from its
/// private fields, which this module can see. A few assertions read private
/// state where it is the property under test: where an entry is stored
/// (mini-moka or the side store), and the epoch bookkeeping that pruning
/// empties.
#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};

    use super::*;
    use crate::domain::delta::apply_delta;
    use crate::domain::types::{compact, true_capacity_for_test, ObjectState};

    /// A nonzero budget: a zero budget is the contract's `Disabled` cache, a
    /// different case from "nothing cached yet".
    const BUDGET_BYTES: u64 = 64 * 1024 * 1024;
    /// The object every single-object test uses.
    const OBJECT_ID: u64 = 42;

    fn entry(object_id: u64, seq: u64) -> CacheEntry {
        CacheEntry {
            object_id,
            commit_chain: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            seq,
            dangling_parents: HashSet::from([77]),
            state: CompactState::default(),
            state_weight_bytes: 1_024,
            index_weight_bytes: 0,
            last_used: Instant::now(),
            plan: None,
        }
    }

    /// A guard over `entry`, built from its fields.
    fn guard(cache: &StructuralCache, entry: CacheEntry) -> CheckedOut {
        CheckedOut {
            original: Some(entry),
            checkout_epoch: 0,
            cache: cache.clone(),
            mutating: false,
            advanced: false,
            pending: None,
        }
    }

    #[test]
    fn lookup_empty_cache_any_object_returns_miss() {
        let cache = StructuralCache::new(BUDGET_BYTES);
        for object_id in [0, OBJECT_ID, u64::MAX] {
            let answer = cache.lookup(object_id);
            assert!(
                matches!(answer, CacheLookup::Miss),
                "lookup({object_id}) must answer Miss; got {answer:?}"
            );
        }
    }

    #[test]
    fn checkout_empty_cache_any_object_returns_none() {
        let cache = StructuralCache::new(BUDGET_BYTES);
        for object_id in [0, OBJECT_ID, u64::MAX] {
            let answer = cache.checkout(object_id);
            assert!(
                answer.is_none(),
                "checkout({object_id}) must answer None; got {answer:?}"
            );
        }
    }

    #[test]
    fn structural_cache_clone_shares_inner_state() {
        let cache = StructuralCache::new(BUDGET_BYTES);
        let clone = cache.clone();
        assert!(
            Arc::ptr_eq(&cache.inner, &clone.inner),
            "every clone must share one inner state"
        );
    }

    #[test]
    fn checked_out_entry_returns_the_held_entry() {
        let cache = StructuralCache::new(BUDGET_BYTES);
        let held = entry(OBJECT_ID, 5);
        let checked_out = guard(&cache, held.clone());
        assert_eq!(checked_out.entry(), &held);
    }

    #[test]
    fn uncacheable_reason_variants_construct_as_specified() {
        let reasons = [
            UncacheableReason::OverWeight(17 * 1024 * 1024),
            UncacheableReason::NoChsComplete,
            UncacheableReason::Disabled,
            UncacheableReason::ReloadLimit,
            UncacheableReason::FencedReadFailed,
        ];
        for (i, a) in reasons.iter().enumerate() {
            for (j, b) in reasons.iter().enumerate() {
                assert_eq!(i == j, a == b, "{a:?} vs {b:?}");
            }
        }
        let UncacheableReason::OverWeight(weight) = reasons[0] else {
            unreachable!("reasons[0] is OverWeight");
        };
        assert_eq!(
            weight,
            17 * 1024 * 1024,
            "OverWeight carries the estimated weight"
        );
    }

    /// Every verdict reason maps to its own
    /// `live_edit_structural_cache_uncacheable_total` cell, the mapping covers
    /// the whole seeded vocabulary in its order, and `Disabled` — a
    /// configuration, never a verdict — has no cell.
    #[test]
    fn uncacheable_reason_labels_are_the_seeded_vocabulary_and_disabled_has_none() {
        use crate::observability::metrics::STRUCTURAL_CACHE_UNCACHEABLE_REASONS;

        let labels: Vec<&str> = [
            UncacheableReason::OverWeight(1),
            UncacheableReason::NoChsComplete,
            UncacheableReason::ReloadLimit,
            UncacheableReason::FencedReadFailed,
            UncacheableReason::OverDeadline,
        ]
        .into_iter()
        .map(|reason| {
            reason
                .label()
                .unwrap_or_else(|| panic!("{reason:?} is a verdict and must have a label"))
        })
        .collect();
        assert_eq!(labels, STRUCTURAL_CACHE_UNCACHEABLE_REASONS);
        assert_eq!(
            UncacheableReason::OverWeight(17 * 1024 * 1024).label(),
            UncacheableReason::OverWeight(1).label(),
            "the weight is on the log line, never in the label"
        );
        assert_eq!(UncacheableReason::Disabled.label(), None);
    }

    /// `take_unreported` hands a recorded verdict out exactly once: the
    /// first take gets it, with its weight; every later take — the verdict now
    /// reported — gets nothing, while lookups still route the object by it. A
    /// re-recorded verdict is a new one, taken once more. No verdict, an
    /// expired one and a disabled cache give nothing.
    #[test]
    fn take_unreported_hands_each_verdict_out_once() {
        const NEGATIVE_TTL: Duration = Duration::from_millis(200);
        const SLACK: Duration = Duration::from_millis(100);
        let cache = StructuralCache::with_ttls(BUDGET_BYTES, LONG_TTL, LONG_TTL);
        cache.record_uncacheable(OBJECT_ID, UncacheableReason::OverWeight(99));

        assert_eq!(
            cache.take_unreported(OBJECT_ID),
            Some(UncacheableReason::OverWeight(99)),
            "the first take reports the verdict, with its weight"
        );
        assert_eq!(
            cache.take_unreported(OBJECT_ID),
            None,
            "a verdict already taken (reported) is never handed out again"
        );
        assert!(
            matches!(
                cache.lookup(OBJECT_ID),
                CacheLookup::Uncacheable {
                    reason: UncacheableReason::OverWeight(99)
                }
            ),
            "taking it does not consume the negative entry: lookups still route by it"
        );

        cache.record_uncacheable(OBJECT_ID, UncacheableReason::NoChsComplete);
        assert_eq!(
            cache.take_unreported(OBJECT_ID),
            Some(UncacheableReason::NoChsComplete),
            "a re-recorded verdict is a new one, reported once more"
        );

        assert_eq!(cache.take_unreported(OBJECT_ID + 1), None, "no verdict");
        let short = StructuralCache::with_ttls(BUDGET_BYTES, LONG_TTL, NEGATIVE_TTL);
        short.record_uncacheable(OBJECT_ID, UncacheableReason::NoChsComplete);
        std::thread::sleep(NEGATIVE_TTL + SLACK);
        assert_eq!(
            short.take_unreported(OBJECT_ID),
            None,
            "an expired verdict is not reported"
        );
        let disabled = StructuralCache::new(0);
        disabled.record_uncacheable(OBJECT_ID, UncacheableReason::NoChsComplete);
        assert_eq!(disabled.take_unreported(OBJECT_ID), None, "disabled cache");
    }

    /// Two threads racing to take one unreported verdict: exactly one gets it
    /// (the flag flips under the cache's lock).
    #[test]
    fn take_unreported_racing_takers_get_the_verdict_exactly_once() {
        for _ in 0..200 {
            let cache = StructuralCache::new(BUDGET_BYTES);
            cache.record_uncacheable(OBJECT_ID, UncacheableReason::NoChsComplete);
            let barrier = Arc::new(std::sync::Barrier::new(4));
            let takers: Vec<_> = (0..4)
                .map(|_| {
                    let (cache, barrier) = (cache.clone(), Arc::clone(&barrier));
                    std::thread::spawn(move || {
                        barrier.wait();
                        cache.take_unreported(OBJECT_ID).is_some()
                    })
                })
                .collect();
            let got = takers
                .into_iter()
                .map(|t| t.join().expect("taker"))
                .filter(|&got| got)
                .count();
            assert_eq!(got, 1, "exactly one taker reports the verdict");
        }
    }

    /// The `Drop` of a guard whose repair grew its entry past the cap records
    /// an `OverWeight` verdict nobody heard: it stays unreported, so the next
    /// take (the first structural lookup it routes) reports it, with the
    /// grown weight — and only once.
    #[test]
    fn take_unreported_hands_out_the_verdict_a_dropped_grown_guard_recorded() {
        let cache = cache_with(4 * shape_weight());
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 1, state_with_sections(SHAPE_SECTIONS))
        ));
        let grown = cut(OBJECT_ID, 2, state_with_sections(SHAPE_SECTIONS * 4));
        let grown_weight = charged_weight(SHAPE_SECTIONS * 4);
        assert!(
            grown_weight > cache.max_entry_bytes(),
            "fixture: over the cap"
        );
        let mut guard = hit(&cache, OBJECT_ID);

        advance(&mut guard, &grown);
        drop(guard);

        assert_eq!(
            cache.take_unreported(OBJECT_ID),
            Some(UncacheableReason::OverWeight(grown_weight))
        );
        assert_eq!(cache.take_unreported(OBJECT_ID), None);
    }

    #[test]
    fn cache_lookup_variants_construct_as_specified() {
        let cache = StructuralCache::new(BUDGET_BYTES);
        let held = entry(OBJECT_ID, 5);

        match CacheLookup::Hit(guard(&cache, held.clone())) {
            CacheLookup::Hit(checked_out) => assert_eq!(
                checked_out.entry(),
                &held,
                "Hit carries the guard over the taken entry"
            ),
            other => panic!("expected Hit; got {other:?}"),
        }
        let uncacheable = CacheLookup::Uncacheable {
            reason: UncacheableReason::NoChsComplete,
        };
        match uncacheable {
            CacheLookup::Uncacheable { reason } => {
                assert_eq!(reason, UncacheableReason::NoChsComplete)
            }
            other => panic!("expected Uncacheable; got {other:?}"),
        }
    }

    // ── Fixtures for the behavioural suite ──────────────────────────────────

    /// A TTL no test comes near, for the expiry a test is not exercising.
    const LONG_TTL: Duration = Duration::from_secs(3_600);
    /// A second object, distinct from [`OBJECT_ID`].
    const OTHER_OBJECT_ID: u64 = 43;
    /// Sections in the equal-weight states the capacity fixtures fill the
    /// cache with: any small count works, as long as every such entry weighs
    /// the same.
    const SHAPE_SECTIONS: u64 = 8;
    /// The objects [`full_cache`] fills mini-moka's three quarters with.
    const RESIDENTS: [u64; 3] = [101, 102, 103];
    /// A new object cold-started into a [`full_cache`].
    const CANDIDATE: u64 = 200;
    /// How often [`full_cache`] reads each resident, so its frequency in
    /// mini-moka's sketch beats a new key's (one `Miss` read) by a wide
    /// margin: W-TinyLFU admits a new key over a victim only if the key is
    /// read strictly more often.
    const RESIDENT_READS: usize = 8;

    /// A normalized state of `sections` sections under the root, each with one
    /// inline field. Two states with the same count weigh the same, whatever
    /// object they are cached for.
    fn state_with_sections(sections: u64) -> ObjectState {
        let mut state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usd".to_owned(),
            ..ObjectState::default()
        };
        let mut root_children = Vec::new();
        for id in 1..=sections {
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: 0,
                    name: format!("s{id}"),
                    section_type: 6,
                    section_order: id as u32,
                    first_version: 1,
                    last_version: 1,
                },
            );
            state.fields.insert(
                (id, "default".to_owned()),
                Field {
                    section_id: id,
                    key_name: "default".to_owned(),
                    value: FieldValue::Inline(vec![7; 24]),
                    set_order: id,
                    authoring_set_order: id,
                    version: 1,
                },
            );
            root_children.push(id);
        }
        if !root_children.is_empty() {
            state.children.insert(0, root_children);
        }
        state
    }

    /// An entry for `object_id` at the cut `seq`, with a `commit_chain`
    /// distinct per `seq`.
    /// The entry holds `state`'s compact projection, as every cache entry is
    /// compact.
    fn cut(object_id: u64, seq: u64, state: ObjectState) -> CacheEntry {
        compact_cut(object_id, seq, compact(&state))
    }

    /// [`cut`] for a state already in the compact form.
    fn compact_cut(object_id: u64, seq: u64, state: CompactState) -> CacheEntry {
        CacheEntry::new(object_id, format!("{seq:040x}"), seq, HashSet::new(), state)
    }

    /// A cache with no expiry in reach of a test.
    fn cache_with(budget_bytes: u64) -> StructuralCache {
        StructuralCache::with_ttls(budget_bytes, LONG_TTL, LONG_TTL)
    }

    /// Cold-start `entry` as a structural call does: admit it, then settle by
    /// putting it back. Returns whether the put-back was accepted
    /// ([`PutBack::Stored`]), which a store can still have evicted in the
    /// same critical section, for its weight or by the resident bound.
    fn cold_start(cache: &StructuralCache, entry: CacheEntry) -> bool {
        let object_id = entry.object_id;
        let guard = cache
            .admit(cache.begin_cold_start(), entry.clone())
            .unwrap_or_else(|reason| panic!("object {object_id} must be admissible: {reason:?}"));
        guard.put_back(entry).is_stored()
    }

    /// The guard of a lookup that must hit.
    fn hit(cache: &StructuralCache, object_id: u64) -> CheckedOut {
        match cache.lookup(object_id) {
            CacheLookup::Hit(guard) => guard,
            other => panic!("lookup({object_id}) must hit; got {other:?}"),
        }
    }

    /// Assert `got` is the cut `want`: same object, `(seq, commit_chain)`,
    /// dangling parents and state. Weight and last use are the cache's to
    /// refresh.
    fn assert_same_cut(got: &CacheEntry, want: &CacheEntry, context: &str) {
        assert_eq!(got.object_id, want.object_id, "{context}: object_id");
        assert_eq!(
            (got.seq, &got.commit_chain),
            (want.seq, &want.commit_chain),
            "{context}: the (seq, commit_chain) cut"
        );
        assert_eq!(
            got.dangling_parents, want.dangling_parents,
            "{context}: dangling parents"
        );
        assert_eq!(got.state, want.state, "{context}: state");
    }

    /// Whether mini-moka holds `object_id`'s key.
    fn resident_in_main(cache: &StructuralCache, object_id: u64) -> bool {
        let control = cache.inner.lock();
        control
            .entries
            .as_ref()
            .is_some_and(|entries| entries.contains_key(&object_id))
    }

    /// Whether the side store holds `object_id`'s entry.
    fn in_side_store(cache: &StructuralCache, object_id: u64) -> bool {
        cache.inner.lock().side.contains_key(&object_id)
    }

    /// The weight the cache charges an entry for
    /// `state_with_sections(sections)`: the estimate of its compact projection
    /// with its tables compacted, which is what `admit` and `put_back` store.
    /// Every fixture stores such a projection ([`cut`]), whose strings and
    /// lists have no spare capacity.
    fn charged_weight(sections: u64) -> u64 {
        let mut state = compact(&state_with_sections(sections));
        compact_state(&mut state);
        estimate_state_bytes(&state)
    }

    /// The state weight of one [`SHAPE_SECTIONS`]-section entry, which the
    /// per-entry cap reads.
    fn shape_weight() -> u64 {
        charged_weight(SHAPE_SECTIONS)
    }

    /// The whole weight, state and plan indexes, the cache charges a stored
    /// entry for `state_with_sections(sections)`, to the weigher's 4-byte
    /// unit: what mini-moka, the side store and the weight gauge count.
    fn whole_weight(sections: u64) -> u64 {
        let mut entry = cut(0, 1, state_with_sections(sections));
        entry.compact_and_weigh();
        entry.weight_bytes().div_ceil(SLOT_UNIT_BYTES) * SLOT_UNIT_BYTES
    }

    /// The cache's resident budget: its configured budget × (1 + φ).
    fn resident_budget(cache: &StructuralCache) -> u64 {
        resident_budget_bytes(cache.budget_bytes()).expect("a test budget fits")
    }

    /// A cache whose budget is four shape weights — so the per-entry cap is
    /// exactly one shape, mini-moka's capacity (the budget less its quarter)
    /// is three, and the side store holds one shape's weight — with the three
    /// [`RESIDENTS`] filling mini-moka, each read [`RESIDENT_READS`] times.
    fn full_cache(idle_ttl: Duration, negative_ttl: Duration) -> StructuralCache {
        full_cache_of(SHAPE_SECTIONS, idle_ttl, negative_ttl)
    }

    /// [`full_cache`] with residents of `sections` sections each.
    fn full_cache_of(sections: u64, idle_ttl: Duration, negative_ttl: Duration) -> StructuralCache {
        let weight = charged_weight(sections);
        let cache = StructuralCache::with_ttls(4 * weight, idle_ttl, negative_ttl);
        assert_eq!(
            cache.max_entry_bytes(),
            weight,
            "fixture: the cap is one resident"
        );
        for object_id in RESIDENTS {
            assert!(cold_start(
                &cache,
                cut(object_id, 1, state_with_sections(sections).clone())
            ));
        }
        for _ in 0..RESIDENT_READS {
            for object_id in RESIDENTS {
                drop(hit(&cache, object_id));
            }
        }
        // Applies the pending reads to mini-moka's frequency sketch.
        assert_eq!(cache.entry_count(), RESIDENTS.len() as u64);
        for object_id in RESIDENTS {
            assert!(
                resident_in_main(&cache, object_id),
                "fixture: resident {object_id} must be in mini-moka, which has room for three"
            );
        }
        cache
    }

    // ── The epoch check (SR-3.4) ────────────────────────────────────────────

    #[test]
    fn put_back_after_invalidate_during_checkout_is_refused() {
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 5, state_with_sections(3))
        ));
        let guard = hit(&cache, OBJECT_ID);

        // An audit's `diverged` while the call holds the entry.
        cache.invalidate(OBJECT_ID);

        assert!(
            guard.put_back(cut(OBJECT_ID, 6, state_with_sections(4))) == PutBack::Invalidated,
            "a put-back by a checkout taken before an invalidation must be refused"
        );
        assert!(
            matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss),
            "the refused successor must not be stored"
        );
    }

    #[test]
    fn checked_out_drop_after_invalidate_during_checkout_does_not_restore_the_original() {
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 5, state_with_sections(3))
        ));
        let guard = hit(&cache, OBJECT_ID);

        cache.invalidate(OBJECT_ID);
        drop(guard);

        assert!(
            matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss),
            "the RAII put-back is epoch-checked like an explicit one"
        );
    }

    #[test]
    fn put_back_after_invalidation_is_refused_even_when_a_fresh_entry_appeared_meanwhile() {
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 5, state_with_sections(3))
        ));
        let straggler = hit(&cache, OBJECT_ID);

        cache.invalidate(OBJECT_ID);
        let fresh = cut(OBJECT_ID, 9, state_with_sections(5));
        assert!(
            cold_start(&cache, fresh.clone()),
            "a cold start after the invalidation is stored"
        );

        assert!(
            straggler.put_back(cut(OBJECT_ID, 6, state_with_sections(4))) == PutBack::Invalidated,
            "the straggler's checkout predates the invalidation: refused even though the \
             object has an entry again"
        );
        assert_same_cut(
            hit(&cache, OBJECT_ID).entry(),
            &fresh,
            "the fresh entry must survive the straggler's refused put-back",
        );
    }

    #[test]
    fn put_back_after_another_guards_discard_is_refused() {
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 5, state_with_sections(3))
        ));
        let earlier = hit(&cache, OBJECT_ID);

        // A second guard for the object (a reload) discards: it invalidates.
        let reload = cache
            .admit(
                cache.begin_cold_start(),
                cut(OBJECT_ID, 7, state_with_sections(4)),
            )
            .expect("admissible");
        reload.discard();

        assert!(
            earlier.put_back(cut(OBJECT_ID, 6, state_with_sections(4))) == PutBack::Invalidated,
            "a discard invalidates every checkout of the object taken before it"
        );
        assert!(matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss));
    }

    #[test]
    fn put_back_with_no_invalidation_of_its_object_is_accepted() {
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 5, state_with_sections(3))
        ));
        assert!(cold_start(
            &cache,
            cut(OTHER_OBJECT_ID, 5, state_with_sections(3))
        ));
        let guard = hit(&cache, OBJECT_ID);

        // The epoch advances, but for another object.
        cache.invalidate(OTHER_OBJECT_ID);

        let successor = cut(OBJECT_ID, 6, state_with_sections(4));
        assert!(
            guard.put_back(successor.clone()).is_stored(),
            "an invalidation of another object must not refuse this put-back"
        );
        assert_same_cut(hit(&cache, OBJECT_ID).entry(), &successor, "successor");
    }

    /// PM F5: a reload's ticket, taken by `begin_cold_start` after the
    /// invalidation that preceded it — a reload's own `discard`, or a
    /// retirement — carries an epoch that invalidation never refuses, so the
    /// reload is admitted and its put-back accepted. (`admit`'s guard takes
    /// the ticket's epoch.)
    #[test]
    fn admit_after_an_invalidation_is_not_refused_by_it() {
        for preceding in ["discard", "invalidate"] {
            let cache = cache_with(BUDGET_BYTES);
            // A checkout of another object taken before the invalidation keeps
            // the invalidation's record alive, so this cannot pass on the
            // record having been pruned.
            assert!(cold_start(
                &cache,
                cut(OTHER_OBJECT_ID, 1, state_with_sections(1))
            ));
            let older = hit(&cache, OTHER_OBJECT_ID);
            assert!(cold_start(
                &cache,
                cut(OBJECT_ID, 5, state_with_sections(3))
            ));
            let stale = hit(&cache, OBJECT_ID);
            match preceding {
                "discard" => stale.discard(),
                _ => {
                    drop(stale);
                    cache.invalidate(OBJECT_ID);
                }
            }

            let reloaded = cut(OBJECT_ID, 8, state_with_sections(4));
            assert!(
                cold_start(&cache, reloaded.clone()),
                "after a {preceding}: a reload admitted after the invalidation must be stored"
            );
            assert_same_cut(
                hit(&cache, OBJECT_ID).entry(),
                &reloaded,
                &format!("reload after a {preceding}"),
            );
            drop(older);
        }
    }

    // ── The cold start's ticket ─────────────────────────────────────────────

    /// An invalidation of the object between `begin_cold_start` and `admit` —
    /// a retirement or an audit drop while the cold start reads — refuses the
    /// admission: the read may predate it. Nothing is stored and nothing is
    /// recorded, not even for an entry over the weight cap (the invalidation
    /// is checked before the weight), and the ticket's registration is
    /// released.
    #[test]
    fn admit_after_an_invalidation_since_its_ticket_is_refused_and_records_nothing() {
        for (what, state) in [
            ("an admissible entry", state_with_sections(3)),
            ("an entry over the cap", state_with_sections(64)),
        ] {
            let cache = cache_with(charged_weight(64) * 4 - 4);
            assert!(
                charged_weight(3) <= cache.max_entry_bytes()
                    && charged_weight(64) > cache.max_entry_bytes(),
                "fixture: only the heavy entry is over the cap"
            );
            let ticket = cache.begin_cold_start();

            cache.invalidate(OBJECT_ID);
            let refused = cache.admit(ticket, cut(OBJECT_ID, 5, state)).map(|_| ());

            assert_eq!(refused, Err(AdmitRefused::Invalidated), "{what}");
            assert!(
                matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss),
                "{what}: nothing stored and no negative entry recorded"
            );
            assert_eq!(
                (cache.entry_count(), bookkeeping(&cache, OBJECT_ID)),
                (0, (false, 0)),
                "{what}: the refused ticket's registration is released and the record pruned"
            );
        }
    }

    /// An invalidation of another object since the ticket refuses nothing.
    #[test]
    fn admit_after_an_invalidation_of_another_object_since_its_ticket_is_accepted() {
        let cache = cache_with(BUDGET_BYTES);
        let ticket = cache.begin_cold_start();

        cache.invalidate(OTHER_OBJECT_ID);
        let entry = cut(OBJECT_ID, 5, state_with_sections(3));
        let guard = cache
            .admit(ticket, entry.clone())
            .expect("another object's invalidation must not refuse this admission");

        assert!(guard.put_back(entry.clone()).is_stored());
        assert_same_cut(hit(&cache, OBJECT_ID).entry(), &entry, "admitted");
    }

    /// While a ticket is outstanding, the record of an invalidation after it
    /// is kept — so its `admit` can still see it; dropping the ticket unused
    /// (a verdict, an error, a cancelled cold start) releases its registration
    /// and the record is pruned.
    #[test]
    fn invalidation_record_kept_while_an_older_ticket_is_outstanding_then_pruned_on_its_drop() {
        let cache = cache_with(BUDGET_BYTES);
        let ticket = cache.begin_cold_start();
        assert_eq!(
            bookkeeping(&cache, OBJECT_ID),
            (false, 1),
            "the ticket registers a checkout"
        );

        cache.invalidate(OBJECT_ID);
        assert_eq!(
            bookkeeping(&cache, OBJECT_ID),
            (true, 1),
            "the record must be kept while a ticket taken before it is outstanding"
        );

        drop(ticket);
        assert_eq!(
            bookkeeping(&cache, OBJECT_ID),
            (false, 0),
            "an unused ticket's drop releases its registration, and the record is pruned"
        );
    }

    /// Every refusal consumes the ticket's registration: a disabled cache, an
    /// over-weight entry and an invalidated object all leave nothing
    /// outstanding.
    #[test]
    fn admit_refusals_release_the_tickets_registration() {
        let disabled = StructuralCache::with_ttls(0, LONG_TTL, LONG_TTL);
        let small = cache_with(charged_weight(3) * 4);
        let invalidated = cache_with(BUDGET_BYTES);
        let cases: [(&str, &StructuralCache, ObjectState, bool); 3] = [
            ("disabled", &disabled, state_with_sections(1), false),
            ("over weight", &small, state_with_sections(64), false),
            ("invalidated", &invalidated, state_with_sections(1), true),
        ];
        for (what, cache, state, invalidate) in cases {
            let ticket = cache.begin_cold_start();
            if invalidate {
                cache.invalidate(OBJECT_ID);
            }

            assert!(
                cache.admit(ticket, cut(OBJECT_ID, 1, state)).is_err(),
                "fixture: {what} must refuse"
            );

            let control = cache.inner.lock();
            assert!(
                control.outstanding.is_empty() && control.invalidations.is_empty(),
                "{what}: nothing outstanding after the refusal: {:?} / {:?}",
                control.outstanding,
                control.invalidations
            );
        }
    }

    #[test]
    fn invalidate_drops_a_resident_entry_and_a_negative_entry() {
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 5, state_with_sections(3))
        ));
        cache.record_uncacheable(OTHER_OBJECT_ID, UncacheableReason::NoChsComplete);

        cache.invalidate(OBJECT_ID);
        cache.invalidate(OTHER_OBJECT_ID);

        assert!(matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss));
        assert!(
            matches!(cache.lookup(OTHER_OBJECT_ID), CacheLookup::Miss),
            "retirement drops the negative entry too"
        );
    }

    // ── Put-back by outcome (SR-3.4) ────────────────────────────────────────

    /// How a structural call settles its checkout (the module docs' list).
    #[derive(Debug, Clone, Copy)]
    enum Settle {
        /// `put_back` with the successor entry.
        Successor,
        /// `put_back` with the original entry.
        Original,
        /// `discard`.
        Discard,
        /// Neither: the call is dropped and the guard's `Drop` runs.
        Abandon,
    }

    /// Every SR-3.4 outcome class and how it settles the checkout.
    const OUTCOMES: [(&str, Settle); 14] = [
        ("Committed", Settle::Successor),
        ("STALE", Settle::Original),
        ("FENCED", Settle::Original),
        ("delta Err", Settle::Original),
        ("transport error", Settle::Original),
        ("in-doubt reply", Settle::Original),
        (
            "terminal store-side refusal that changed nothing",
            Settle::Original,
        ),
        ("CACHE_DIVERGED", Settle::Discard),
        ("repair anomaly", Settle::Discard),
        ("NotFound", Settle::Discard),
        ("SUPERSEDED", Settle::Discard),
        ("PARTIAL_HEADER", Settle::Discard),
        (
            "Committed own creation adopting a dangling parent",
            Settle::Discard,
        ),
        ("deadline or cancellation drops the call", Settle::Abandon),
    ];

    #[test]
    fn checked_out_settled_for_each_outcome_class_leaves_the_documented_entry() {
        for (outcome, settle) in OUTCOMES {
            let cache = cache_with(BUDGET_BYTES);
            let original = cut(OBJECT_ID, 5, state_with_sections(3));
            let successor = cut(OBJECT_ID, 6, state_with_sections(4));
            assert!(cold_start(&cache, original.clone()));
            let guard = hit(&cache, OBJECT_ID);

            let expected = match settle {
                Settle::Successor => {
                    assert!(
                        guard.put_back(successor.clone()).is_stored(),
                        "{outcome}: stored"
                    );
                    Some(&successor)
                }
                Settle::Original => {
                    let held = guard.entry().clone();
                    assert!(guard.put_back(held).is_stored(), "{outcome}: stored");
                    Some(&original)
                }
                Settle::Discard => {
                    guard.discard();
                    None
                }
                Settle::Abandon => {
                    drop(guard);
                    Some(&original)
                }
            };

            match (cache.lookup(OBJECT_ID), expected) {
                (CacheLookup::Hit(found), Some(want)) => {
                    assert_same_cut(found.entry(), want, outcome)
                }
                (CacheLookup::Miss, None) => {}
                (other, want) => panic!(
                    "{outcome} ({settle:?}) must leave {}; got {other:?}",
                    want.map_or("no entry".to_owned(), |e| format!(
                        "the entry at seq {}",
                        e.seq
                    ))
                ),
            }
        }
    }

    #[test]
    fn checked_out_unwinding_from_a_panic_puts_the_original_back() {
        let cache = cache_with(BUDGET_BYTES);
        let original = cut(OBJECT_ID, 5, state_with_sections(3));
        assert!(cold_start(&cache, original.clone()));
        let guard = hit(&cache, OBJECT_ID);

        let unwound = catch_unwind(AssertUnwindSafe(move || {
            let _held = guard;
            panic!("planning panicked while holding the entry");
        }));

        assert!(unwound.is_err(), "the closure panics");
        assert_same_cut(hit(&cache, OBJECT_ID).entry(), &original, "after unwind");
    }

    // ── A guard the stream repair advanced (`CheckedOut::mutate`) ───────────

    /// Advance `guard`'s held entry in place to `to`'s cut, the way the stream
    /// repair does: `seq`, `commit_chain` and state, with the plan indexes
    /// kept in step with the state (here by a rebuild), leaving the stored
    /// weights at the checked-out entry's.
    fn advance(guard: &mut CheckedOut, to: &CacheEntry) {
        guard.mutate(|held| {
            held.seq = to.seq;
            held.commit_chain = to.commit_chain.clone();
            held.state = to.state.clone();
            held.plan = Some(PlanIndexes::build(&held.state));
        });
    }

    /// A repair that grew the held entry past the per-entry cap must not slip
    /// it back in under the weight it was checked out with: the guard's
    /// `Drop` re-weighs it, and the cap turns it into an `OverWeight`
    /// negative entry.
    #[test]
    fn checked_out_drop_after_mutate_grew_the_entry_past_the_cap_records_overweight() {
        let cache = cache_with(4 * shape_weight());
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 1, state_with_sections(SHAPE_SECTIONS))
        ));
        let grown = cut(OBJECT_ID, 2, state_with_sections(SHAPE_SECTIONS * 4));
        let grown_weight = charged_weight(SHAPE_SECTIONS * 4);
        assert!(
            grown_weight > cache.max_entry_bytes(),
            "fixture: the grown state is over the cap"
        );
        let mut guard = hit(&cache, OBJECT_ID);

        advance(&mut guard, &grown);
        drop(guard);

        match cache.lookup(OBJECT_ID) {
            CacheLookup::Uncacheable {
                reason: UncacheableReason::OverWeight(weight),
            } => assert_eq!(
                weight, grown_weight,
                "the verdict carries the advanced entry's re-weighed weight"
            ),
            other => panic!(
                "an advanced entry grown past the cap must leave an OverWeight negative entry; \
                 got {other:?}"
            ),
        }
    }

    /// The `Drop` of an advanced guard goes through the same epoch check as
    /// an explicit put-back: an invalidation after the checkout refuses it.
    #[test]
    fn checked_out_drop_after_mutate_is_refused_by_a_later_invalidation() {
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 5, state_with_sections(3))
        ));
        let mut guard = hit(&cache, OBJECT_ID);
        advance(&mut guard, &cut(OBJECT_ID, 6, state_with_sections(4)));

        cache.invalidate(OBJECT_ID);
        drop(guard);

        assert!(
            matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss),
            "an advanced guard's RAII put-back after an invalidation must be refused"
        );
    }

    /// A panic inside `mutate` leaves the held entry possibly half-replayed:
    /// the unwinding guard puts nothing back and invalidates the object, so a
    /// cold start that began before the panic is refused too.
    #[test]
    fn checked_out_panic_inside_mutate_discards_and_invalidates() {
        let cache = cache_with(BUDGET_BYTES);
        let original = cut(OBJECT_ID, 5, state_with_sections(3));
        assert!(cold_start(&cache, original.clone()));
        let earlier_cold_start = cache.begin_cold_start();
        let guard = hit(&cache, OBJECT_ID);

        let unwound = catch_unwind(AssertUnwindSafe(move || {
            let mut guard = guard;
            guard.mutate(|held| {
                held.seq += 1;
                panic!("a replay step panicked mid-record");
            });
        }));

        assert!(unwound.is_err(), "the mutation panics");
        assert!(
            matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss),
            "the half-mutated entry must not be put back, nor the original restored"
        );
        assert_eq!(
            cache.admit(earlier_cold_start, original).map(|_| ()),
            Err(AdmitRefused::Invalidated),
            "the discard invalidates the object: a cold start begun before it is refused"
        );
    }

    /// `put_back_held` stores the entry as the repair left it — its cut and
    /// state — weighed afresh, not at the weight it was checked out with.
    #[test]
    fn put_back_held_stores_the_advanced_entry_re_weighed() {
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 5, state_with_sections(3))
        ));
        let advanced = cut(OBJECT_ID, 6, state_with_sections(5));
        assert_ne!(
            charged_weight(5),
            charged_weight(3),
            "fixture: the advanced state weighs differently"
        );
        let mut guard = hit(&cache, OBJECT_ID);
        advance(&mut guard, &advanced);

        assert!(
            guard.put_back_held().is_stored(),
            "the advanced entry is stored"
        );

        let found = hit(&cache, OBJECT_ID);
        assert_same_cut(found.entry(), &advanced, "put_back_held");
        assert_eq!(
            found.entry().state_weight_bytes,
            charged_weight(5),
            "the stored entry carries the advanced state's weight"
        );
    }

    // ── The per-entry cap and the disabled cache (SR-3.3) ───────────────────

    #[test]
    fn admit_entry_exactly_at_the_cap_is_stored() {
        let weight = shape_weight();
        let cache = cache_with(4 * weight);
        assert_eq!(cache.max_entry_bytes(), weight);

        let entry = cut(OBJECT_ID, 1, state_with_sections(SHAPE_SECTIONS).clone());
        assert!(
            cold_start(&cache, entry.clone()),
            "an entry at the cap fits"
        );
        assert_same_cut(hit(&cache, OBJECT_ID).entry(), &entry, "at the cap");
    }

    #[test]
    fn admit_entry_over_the_cap_is_uncacheable_overweight() {
        let cache = cache_with(4 * shape_weight());
        let heavy = state_with_sections(SHAPE_SECTIONS * 4);
        let heavy_weight = charged_weight(SHAPE_SECTIONS * 4);
        assert!(
            heavy_weight > cache.max_entry_bytes(),
            "fixture: over the cap"
        );

        let refused = cache
            .admit(cache.begin_cold_start(), cut(OBJECT_ID, 1, heavy))
            .map(|_| ());

        assert_eq!(
            refused,
            Err(AdmitRefused::Uncacheable(UncacheableReason::OverWeight(
                heavy_weight
            ))),
            "admission names the estimated weight"
        );
        match cache.lookup(OBJECT_ID) {
            CacheLookup::Uncacheable {
                reason: UncacheableReason::OverWeight(weight),
            } => assert_eq!(weight, heavy_weight),
            other => panic!("the refusal must leave an OverWeight negative entry; got {other:?}"),
        }
    }

    /// The cap is checked against the compact weight: an object whose inline
    /// values alone would put its full form far over the cap is admitted,
    /// weighed without them (b.mr6).
    #[test]
    fn admit_entry_heavy_only_in_inline_values_is_weighed_without_them() {
        let cache = cache_with(4 * shape_weight());
        let mut heavy_values = state_with_sections(SHAPE_SECTIONS);
        for field in heavy_values.fields.values_mut() {
            field.value = FieldValue::Inline(vec![7; 64 * 1024]);
        }
        let mut full = heavy_values.clone();
        compact_state(&mut full);
        assert!(
            estimate_state_bytes(&full) > 4 * cache.max_entry_bytes(),
            "fixture: the full form must be far over the cap"
        );

        let admitted = cache
            .admit(cache.begin_cold_start(), cut(OBJECT_ID, 1, heavy_values))
            .expect("the compact entry is within the cap");

        assert_eq!(
            admitted.entry().state_weight_bytes,
            shape_weight(),
            "the entry weighs what the same rows with small values weigh"
        );
    }

    /// Interner GC (M2): the names of rows a plan removed are freed, and the
    /// put-back's compaction trims the interner, so the stored entry's value
    /// rows — its row table and interner — weigh exactly what a fresh entry
    /// of the remaining rows weighs, and the entry weighs less than before
    /// the removal. (Only the value rows are compared: the plan's root
    /// children list keeps the capacity it had, which the estimate charges.)
    #[test]
    fn put_back_after_removing_rows_frees_their_names_and_the_weight_falls() {
        const KEPT: u64 = 10;
        const GROWN: u64 = 20;
        /// A distinct field name per section, long enough to weigh.
        fn name(section_id: u64) -> String {
            format!("primvars:customAttribute{section_id:04}")
        }
        /// `state_with_sections(sections)`'s compact projection with one field
        /// of its own name per section, interned in section order.
        fn named(sections: u64) -> CompactState {
            let mut state = compact(&state_with_sections(sections));
            state.fields = CompactFields::default();
            for id in 1..=sections {
                state.fields.insert(id, &name(id), None);
            }
            state
        }
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(&cache, compact_cut(OBJECT_ID, 1, named(GROWN))));
        let before = hit(&cache, OBJECT_ID);
        let weight_before = before.entry().state_weight_bytes;

        // The plan deletes the sections inserted last; their names held the
        // highest ids.
        let mut plan = before.plan_state(PlannedInserts::default());
        let deletes = (KEPT + 1..=GROWN)
            .map(|section_id| {
                StructCommand::DeleteSection(oodf_ffi::CommandDeleteSection {
                    parent_id: 0,
                    section_id,
                })
            })
            .collect();
        apply_delta(
            &mut plan,
            Some(OwnedDelta {
                struct_commands: deletes,
                ..OwnedDelta::default()
            }),
        )
        .expect("the deletes apply");
        assert!(
            plan.fields.names().id_of(&name(GROWN)).is_none(),
            "a removed row's name is released by the purge itself"
        );
        assert!(before.put_back(compact_cut(OBJECT_ID, 2, plan)).is_stored());

        let stored = hit(&cache, OBJECT_ID);
        let mut fresh = named(KEPT);
        compact_state(&mut fresh);
        let rows_weight = |state: &CompactState| {
            <CompactRows as WeighedRows>::value_rows_bytes(&state.fields, &state.time_samples)
        };
        assert_eq!(
            stored.entry().state.fields.names().len() as u64,
            KEPT,
            "only the kept rows' names are interned"
        );
        assert_eq!(
            rows_weight(&stored.entry().state),
            rows_weight(&fresh),
            "the stored value rows weigh what a fresh entry's rows weigh"
        );
        assert!(
            stored.entry().state_weight_bytes < weight_before,
            "the weight falls ({} vs {weight_before})",
            stored.entry().state_weight_bytes
        );
    }

    /// The childless live parent the copy-headroom fixtures insert under.
    const CHILDLESS: u64 = 100;

    /// A state with `lists` children-list keys: `lists − 1` parents under
    /// the root each listing one child (ids `1,000 + i`), plus the root's list,
    /// and [`CHILDLESS`] under the root with no list; plus `orphans` rows
    /// naming id 500, which has no row (they are listed nowhere). Compacted,
    /// its children table has the smallest bucket count `lists` fits in: 7
    /// keys fill 8 buckets (capacity 7), 13 keys leave one free of 16's 14.
    fn children_table_of(lists: u64, orphans: u64) -> ObjectState {
        let mut state = state_with_sections(0);
        let section = |id: u64, parent_id: u64, order: u32| Section {
            section_id: id,
            parent_id,
            name: format!("s{id}"),
            section_type: 6,
            section_order: order,
            first_version: 1,
            last_version: 1,
        };
        let mut root = Vec::new();
        for i in 1..lists {
            state.sections.insert(i, section(i, 0, i as u32));
            state.sections.insert(1_000 + i, section(1_000 + i, i, 0));
            state.children.insert(i, vec![1_000 + i]);
            root.push(i);
        }
        state
            .sections
            .insert(CHILDLESS, section(CHILDLESS, 0, lists as u32));
        root.push(CHILDLESS);
        state.children.insert(0, root);
        for o in 0..orphans {
            state
                .sections
                .insert(600 + o, section(600 + o, 500, o as u32));
        }
        state
    }

    /// CR1's growth term ([`estimate_in_place_growth_bytes`]) covers each
    /// table the delta's inserts grow in place. At a threshold — where the
    /// state's table rehashes into twice its buckets inside the plan (each
    /// case self-checks that) — the term is at least what the grown state
    /// weighs more: sections at 7,168 = 7/8 × 8,192 buckets plus one
    /// `CreateSection`; a full children table (7 of 7) plus a `MoveSection`
    /// into a childless parent (one new list key); a children table one key
    /// short of full (13 of 14) plus a birth under a childless parent that
    /// adopts two orphans (two new keys: the parent's list and the newborn's,
    /// so room for one is not enough). Below the threshold (7,000 sections
    /// plus one `CreateSection`) no table grows and the term is 0.
    #[test]
    fn in_place_growth_charges_each_table_the_deltas_inserts_grow() {
        let create = |parent_id: u64, section_id: u64| OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id,
                section_id,
                section_name: "new".to_owned(),
                section_type: 6,
            })],
            ..OwnedDelta::default()
        };
        let move_to_childless = OwnedDelta {
            struct_commands: vec![StructCommand::MoveSection(oodf_ffi::CommandMoveSection {
                old_parent_id: 1,
                new_parent_id: CHILDLESS,
                section_id: 1_001,
                new_name: "moved".to_owned(),
            })],
            ..OwnedDelta::default()
        };
        // (label, state, delta, at a threshold)
        let cases = [
            (
                "7,168 sections plus one new section",
                state_with_sections(7_168),
                create(0, 9_000_000),
                true,
            ),
            (
                "a full children table plus a move into a childless parent",
                children_table_of(7, 0),
                move_to_childless,
                true,
            ),
            (
                "a children table one key short of full plus a birth under a childless \
                 parent adopting two orphans",
                children_table_of(13, 2),
                create(CHILDLESS, 500),
                true,
            ),
            (
                "7,000 sections plus one new section (no threshold)",
                state_with_sections(7_000),
                create(0, 9_000_000),
                false,
            ),
        ];
        let capacities = |state: &CompactState| {
            (
                state.sections.capacity(),
                state.children.capacity(),
                state.fields.capacity(),
                state.time_samples.capacity(),
            )
        };
        for (label, state, delta, threshold) in cases {
            let cache = cache_with(BUDGET_BYTES);
            assert!(cold_start(&cache, cut(OBJECT_ID, 1, state)), "{label}");
            let guard = hit(&cache, OBJECT_ID);
            let entry = guard.entry();
            let mut plain = entry.state.clone();
            apply_delta(&mut plain, Some(delta.clone())).expect("the delta applies");
            assert_eq!(
                capacities(&plain) != capacities(&entry.state),
                threshold,
                "{label}: fixture self-check: whether a plain clone rehashes inside the plan \
                 ({:?} → {:?})",
                capacities(&entry.state),
                capacities(&plain)
            );
            let inserts = PlannedInserts::of(Some(&delta));
            let growth = estimate_in_place_growth_bytes(&entry.state, entry.plan.as_ref(), inserts);
            if threshold {
                // The grown tables, whole, cover what the state weighs more
                // once they have grown (the old ones are in the held weight).
                let grown_by = estimate_state_bytes(&plain) - estimate_state_bytes(&entry.state);
                assert!(
                    growth >= grown_by && growth > 0,
                    "{label}: the growth term ({growth}) covers the grown tables ({grown_by})"
                );
            } else {
                assert_eq!(
                    growth, 0,
                    "{label}: no table grows, so the growth term is 0"
                );
            }
            drop(guard);
        }
    }

    #[test]
    fn put_back_successor_over_the_cap_is_refused_and_records_overweight() {
        let cache = cache_with(4 * shape_weight());
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 1, state_with_sections(SHAPE_SECTIONS).clone())
        ));
        let guard = hit(&cache, OBJECT_ID);
        let grown = state_with_sections(SHAPE_SECTIONS * 4);
        let grown_weight = charged_weight(SHAPE_SECTIONS * 4);
        assert!(
            grown_weight > cache.max_entry_bytes(),
            "fixture: over the cap"
        );

        assert!(
            guard.put_back(cut(OBJECT_ID, 2, grown)) == PutBack::OverCap,
            "a successor over the cap is not stored"
        );
        match cache.lookup(OBJECT_ID) {
            CacheLookup::Uncacheable {
                reason: UncacheableReason::OverWeight(weight),
            } => assert_eq!(
                weight, grown_weight,
                "the reason carries the successor's weight"
            ),
            other => panic!(
                "an over-cap successor must leave an OverWeight negative entry; got {other:?}"
            ),
        }
    }

    #[test]
    fn disabled_cache_answers_disabled_and_stores_nothing() {
        let cache = cache_with(0);

        assert_eq!(
            cache
                .admit(
                    cache.begin_cold_start(),
                    cut(OBJECT_ID, 1, state_with_sections(1))
                )
                .map(|_| ()),
            Err(AdmitRefused::Uncacheable(UncacheableReason::Disabled))
        );
        cache.record_uncacheable(OBJECT_ID, UncacheableReason::NoChsComplete);
        assert!(matches!(
            cache.lookup(OBJECT_ID),
            CacheLookup::Uncacheable {
                reason: UncacheableReason::Disabled
            }
        ));
        assert!(cache.checkout(OBJECT_ID).is_none());
        assert_eq!((cache.entry_count(), cache.weight_bytes()), (0, 0));
    }

    #[test]
    fn record_uncacheable_disabled_reason_is_never_recorded() {
        let cache = cache_with(BUDGET_BYTES);

        cache.record_uncacheable(OBJECT_ID, UncacheableReason::Disabled);

        assert!(
            matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss),
            "Disabled is the budget's verdict, never a negative entry"
        );
    }

    #[test]
    fn record_uncacheable_reload_limit_reason_is_never_recorded() {
        let cache = cache_with(BUDGET_BYTES);
        let entry = cut(OBJECT_ID, 5, state_with_sections(2));
        assert!(cold_start(&cache, entry.clone()));

        cache.record_uncacheable(OBJECT_ID, UncacheableReason::ReloadLimit);

        let guard = hit(&cache, OBJECT_ID);
        assert_same_cut(
            guard.entry(),
            &entry,
            "ReloadLimit is one call's verdict: no negative entry, the entry kept",
        );
    }

    #[test]
    fn record_uncacheable_fenced_read_failed_reason_is_never_recorded() {
        let cache = cache_with(BUDGET_BYTES);
        let entry = cut(OBJECT_ID, 5, state_with_sections(2));
        assert!(cold_start(&cache, entry.clone()));

        cache.record_uncacheable(OBJECT_ID, UncacheableReason::FencedReadFailed);

        assert_eq!(
            cache.take_unreported(OBJECT_ID),
            None,
            "FencedReadFailed is one call's verdict: nothing to report from the cache"
        );
        let guard = hit(&cache, OBJECT_ID);
        assert_same_cut(
            guard.entry(),
            &entry,
            "FencedReadFailed is one call's verdict: no negative entry, the entry kept",
        );
    }

    // ── Eviction by budget and idle TTL (SR-3.3; E1: not LRU order) ─────────

    #[test]
    fn inserting_past_the_budget_evicts_and_keeps_the_held_weight_within_it() {
        const INSERTED: u64 = 12;
        let weight = shape_weight();
        let cache = cache_with(4 * weight);

        for object_id in 1..=INSERTED {
            assert!(
                cold_start(
                    &cache,
                    cut(object_id, 1, state_with_sections(SHAPE_SECTIONS).clone())
                ),
                "an admissible cold start is always stored (main cache or side store)"
            );
        }

        assert!(
            cache.weight_bytes() <= resident_budget(&cache),
            "held weight {} must stay within the {}-byte resident budget",
            cache.weight_bytes(),
            resident_budget(&cache)
        );
        // Every entry out at once, so no put-back reshuffles the count.
        let held: Vec<CheckedOut> = (1..=INSERTED)
            .filter_map(|object_id| match cache.lookup(object_id) {
                CacheLookup::Hit(guard) => Some(guard),
                CacheLookup::Miss => None,
                other => panic!("object {object_id}: a full cache never answers {other:?}"),
            })
            .collect();
        assert!(
            !held.is_empty() && held.len() <= 4,
            "a budget of four entries' weight holds at most four (three in mini-moka, one \
             in the side store); {} of {INSERTED} hit",
            held.len()
        );
    }

    #[test]
    fn idle_entry_is_gone_after_the_idle_ttl_plus_slack() {
        const IDLE_TTL: Duration = Duration::from_millis(300);
        const SLACK: Duration = Duration::from_millis(300);
        let cache = StructuralCache::with_ttls(BUDGET_BYTES, IDLE_TTL, LONG_TTL);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 5, state_with_sections(3))
        ));

        std::thread::sleep(IDLE_TTL + SLACK);

        assert!(
            matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss),
            "an entry idle for longer than the idle TTL is evicted"
        );
    }

    #[test]
    fn side_store_entry_is_gone_after_the_idle_ttl_plus_slack() {
        // Longer than the other expiry tests: the fixture's few dozen
        // in-memory operations must finish well inside the TTL, or the
        // residents expire and the candidate is admitted instead of refused.
        const IDLE_TTL: Duration = Duration::from_millis(600);
        const SLACK: Duration = Duration::from_millis(400);
        let cache = full_cache(IDLE_TTL, LONG_TTL);
        assert!(matches!(cache.lookup(CANDIDATE), CacheLookup::Miss));
        assert!(cold_start(
            &cache,
            cut(CANDIDATE, 1, state_with_sections(SHAPE_SECTIONS).clone())
        ));
        assert!(
            in_side_store(&cache, CANDIDATE),
            "fixture: admission must refuse the candidate into the side store"
        );

        std::thread::sleep(IDLE_TTL + SLACK);

        assert!(
            matches!(cache.lookup(CANDIDATE), CacheLookup::Miss),
            "a side-store entry expires by the same idle TTL"
        );
    }

    /// E1's slot design: a checkout takes the entry out of its slot but the key
    /// stays resident and charged, so its put-back is an update of that key,
    /// never a new admission W-TinyLFU could refuse.
    ///
    /// Discriminating shape: the checked-out object is read once, the other
    /// two residents often. Were its key dropped at checkout, the candidate
    /// cold-started meanwhile would take the freed room, and the put-back — a
    /// new key, rarely read, into a full cache — would be refused into the
    /// side store.
    #[test]
    fn put_back_of_a_checked_out_entry_keeps_its_key_in_the_main_cache() {
        let weight = shape_weight();
        let cache = cache_with(4 * weight);
        let [hot_a, hot_b, checked_out] = RESIDENTS;
        for object_id in RESIDENTS {
            assert!(cold_start(
                &cache,
                cut(object_id, 1, state_with_sections(SHAPE_SECTIONS).clone())
            ));
        }
        for _ in 0..RESIDENT_READS {
            drop(hit(&cache, hot_a));
            drop(hit(&cache, hot_b));
        }
        let guard = hit(&cache, checked_out);
        assert_eq!(
            (cache.entry_count(), cache.weight_bytes()),
            (3, 3 * whole_weight(SHAPE_SECTIONS)),
            "a checked-out entry's key stays resident and charged"
        );

        assert!(matches!(cache.lookup(CANDIDATE), CacheLookup::Miss));
        assert!(cold_start(
            &cache,
            cut(CANDIDATE, 1, state_with_sections(SHAPE_SECTIONS).clone())
        ));
        assert!(
            in_side_store(&cache, CANDIDATE),
            "the checked-out key still fills its share of mini-moka, so the candidate is refused"
        );

        let successor = cut(checked_out, 2, state_with_sections(1).clone());
        let successor_weight = whole_weight(1);
        let candidate_weight = cache.inner.lock().side[&CANDIDATE].weight_bytes();
        assert!(guard.put_back(successor.clone()).is_stored());

        assert!(
            resident_in_main(&cache, checked_out) && !in_side_store(&cache, checked_out),
            "the put-back must refill the resident key, not re-admit it"
        );
        assert_eq!(
            cache.weight_bytes(),
            2 * whole_weight(SHAPE_SECTIONS) + successor_weight + candidate_weight,
            "the put-back refreshes the key's weight (two residents, the lighter successor, \
             and the side-store candidate)"
        );
        assert_same_cut(hit(&cache, checked_out).entry(), &successor, "successor");
    }

    // ── Negative entries (SR-3.4) ───────────────────────────────────────────

    /// The negative entry expires at an absolute time — its creation plus the
    /// negative TTL — however often it is looked up and however busy the
    /// cache is (W-TinyLFU churn cannot evict it: it lives outside
    /// mini-moka).
    ///
    /// Timing-safe both ways: a lookup is required to answer `Uncacheable`
    /// only when it returned less than the TTL after an instant taken before
    /// the entry was recorded, and to miss only once the TTL plus slack has
    /// passed since an instant taken after. A lookup that extended the life
    /// would keep the entry past the final check, because lookups run until
    /// three quarters of the TTL.
    #[test]
    fn negative_entry_routes_uncacheable_until_its_ttl_under_a_busy_full_cache_then_misses() {
        const NEGATIVE_TTL: Duration = Duration::from_millis(400);
        const SLACK: Duration = Duration::from_millis(200);
        const UNMARKED_OBJECT: u64 = 900;
        let cache = full_cache(LONG_TTL, NEGATIVE_TTL);

        let before = Instant::now();
        cache.record_uncacheable(UNMARKED_OBJECT, UncacheableReason::NoChsComplete);
        let recorded = Instant::now();

        let mut lookups = 0;
        let mut churn = 1_000;
        while before.elapsed() < NEGATIVE_TTL * 3 / 4 {
            let answer = cache.lookup(UNMARKED_OBJECT);
            let within = before.elapsed() < NEGATIVE_TTL;
            match answer {
                CacheLookup::Uncacheable {
                    reason: UncacheableReason::NoChsComplete,
                } => {}
                other if within => panic!(
                    "lookup {lookups}, inside the negative TTL: must answer \
                     Uncacheable(NoChsComplete); got {other:?}"
                ),
                _ => {}
            }
            lookups += 1;
            // Keep the cache full and busy: a new key cold-started into it,
            // and a resident read.
            assert!(cold_start(
                &cache,
                cut(churn, 1, state_with_sections(SHAPE_SECTIONS).clone())
            ));
            churn += 1;
            drop(cache.lookup(RESIDENTS[lookups % RESIDENTS.len()]));
        }
        assert!(
            lookups >= 2,
            "the entry must be looked up repeatedly; got {lookups}"
        );

        if let Some(remaining) = (NEGATIVE_TTL + SLACK).checked_sub(recorded.elapsed()) {
            std::thread::sleep(remaining);
        }

        assert!(
            matches!(cache.lookup(UNMARKED_OBJECT), CacheLookup::Miss),
            "after its TTL the negative entry is gone and the object is cold-started again"
        );
    }

    #[test]
    fn record_uncacheable_drops_the_objects_entry() {
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 5, state_with_sections(3))
        ));

        cache.record_uncacheable(OBJECT_ID, UncacheableReason::NoChsComplete);

        assert!(matches!(
            cache.lookup(OBJECT_ID),
            CacheLookup::Uncacheable {
                reason: UncacheableReason::NoChsComplete
            }
        ));
        assert_eq!(
            cache.entry_count(),
            0,
            "the entry is dropped with the verdict"
        );
    }

    #[test]
    fn admit_clears_a_negative_entry() {
        let cache = cache_with(BUDGET_BYTES);
        cache.record_uncacheable(OBJECT_ID, UncacheableReason::NoChsComplete);

        let entry = cut(OBJECT_ID, 5, state_with_sections(3));
        assert!(cold_start(&cache, entry.clone()));

        assert_same_cut(hit(&cache, OBJECT_ID).entry(), &entry, "after admission");
    }

    // ── Refused admission: the side store ───────────────────────────────────

    /// A cold start near capacity that W-TinyLFU refuses is still a cached
    /// entry: served as a `Hit`, never `Uncacheable`, so the object costs no
    /// second cold-start read.
    #[test]
    fn admission_refused_near_capacity_is_served_as_a_hit_from_the_side_store() {
        let cache = full_cache(LONG_TTL, LONG_TTL);
        let entry = cut(CANDIDATE, 1, state_with_sections(SHAPE_SECTIONS).clone());
        assert!(matches!(cache.lookup(CANDIDATE), CacheLookup::Miss));

        assert!(
            cold_start(&cache, entry.clone()),
            "a refused admission still stores the entry"
        );
        assert!(
            in_side_store(&cache, CANDIDATE) && !resident_in_main(&cache, CANDIDATE),
            "fixture: mini-moka must have refused the candidate"
        );
        let side_weight = cache.inner.lock().side[&CANDIDATE].weight_bytes();
        assert_eq!(
            (cache.entry_count(), cache.weight_bytes()),
            (4, 3 * whole_weight(SHAPE_SECTIONS) + side_weight),
            "the gauges count the side store"
        );
        assert!(
            cache.weight_bytes() <= resident_budget(&cache),
            "the total stays within the resident budget"
        );

        for round in 1..=3 {
            match cache.lookup(CANDIDATE) {
                CacheLookup::Hit(guard) => {
                    assert_same_cut(guard.entry(), &entry, &format!("lookup {round}"))
                }
                other => panic!(
                    "lookup {round} after a refused admission must hit (no second cold start, \
                     never an uncacheable route); got {other:?}"
                ),
            }
        }
    }

    // ── The resident bound after every store (`Control::hold_resident_bound`) ─

    /// [`RESIDENTS`] in the order the resident-bound fixtures last use them,
    /// least recent first: neither their ids' order nor the order
    /// [`full_cache`] reads them in, so an eviction by id or by mini-moka's
    /// own queue order is told apart from one by last use.
    const LRU_ORDER: [u64; 3] = [RESIDENTS[1], RESIDENTS[0], RESIDENTS[2]];

    /// New objects stored after the side store's heavy entry.
    const FILLER_OBJECT_ID: u64 = 300;
    const NEW_OBJECT_ID: u64 = 301;

    /// A normalized external-heavy state: `sections` sections under the
    /// root, each holding `rows` fields of the same names (`x0`, `x1`, …,
    /// interned once) whose external hashes no other row holds — a shape
    /// whose plan indexes weigh well over φ of its state: each external
    /// row's bucket in the external-hash count table against its value
    /// bucket and boxed hash.
    fn external_heavy_state(sections: u64, rows: u64) -> ObjectState {
        let mut state = state_with_sections(0);
        for id in 1..=sections {
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: 0,
                    name: format!("s{id}"),
                    section_type: 6,
                    section_order: id as u32,
                    first_version: 1,
                    last_version: 1,
                },
            );
            for row in 0..rows {
                let mut hash = [0u8; 32];
                hash[..8].copy_from_slice(&id.to_le_bytes());
                hash[8..16].copy_from_slice(&row.to_le_bytes());
                state.fields.insert(
                    (id, format!("x{row}")),
                    Field {
                        section_id: id,
                        key_name: format!("x{row}"),
                        value: FieldValue::External {
                            hash,
                            size: 64,
                            ext_value_index: 0,
                        },
                        set_order: row + 1,
                        authoring_set_order: row + 1,
                        version: 1,
                    },
                );
            }
        }
        if sections > 0 {
            state.children.insert(0, (1..=sections).collect());
        }
        state
    }

    /// [`cut`], compacted and weighed as `admit` and every put-back store it.
    fn weighed(object_id: u64, seq: u64, state: ObjectState) -> CacheEntry {
        let mut entry = cut(object_id, seq, state);
        entry.compact_and_weigh();
        entry
    }

    /// The weight mini-moka charges `entry`'s slot, in bytes: its whole
    /// weight rounded up to the weigher's 4-byte unit. (The side store
    /// counts an entry's weight unrounded.)
    fn slot_bytes(entry: &CacheEntry) -> u64 {
        u64::from(slot_weight(entry.weight_bytes())) * SLOT_UNIT_BYTES
    }

    /// `(sections, rows)` of every [`external_heavy_state`] of 4, 8, 16 or
    /// 32 rows per section whose state weighs at most `cap`, with its slot
    /// weight ([`slot_bytes`]), lightest first. (A state's weight grows with
    /// its sections, so each row count's search stops at the first over the
    /// cap.)
    fn external_heavy_within(cap: u64) -> Vec<((u64, u64), u64)> {
        let mut shapes = Vec::new();
        for rows in [4, 8, 16, 32] {
            for sections in 1.. {
                let entry = weighed(0, 1, external_heavy_state(sections, rows));
                if entry.state_weight_bytes > cap {
                    break;
                }
                shapes.push(((sections, rows), slot_bytes(&entry)));
            }
        }
        shapes.sort_by_key(|&(_, slot)| slot);
        shapes
    }

    /// The sections of each resident in the resident-bound fixture
    /// ([`full_cache_of`]): enough that the per-entry cap, one resident's
    /// state, holds an external-heavy state whose rows, not its fixed
    /// overhead, set its index-to-state ratio.
    const BOUND_RESIDENT_SECTIONS: u64 = 64;

    /// The side store's total weight.
    fn side_bytes(cache: &StructuralCache) -> u64 {
        cache
            .inner
            .lock()
            .side
            .values()
            .map(CacheEntry::weight_bytes)
            .sum()
    }

    /// The resident-bound fixture: a [`full_cache_of`] residents of
    /// [`BOUND_RESIDENT_SECTIONS`] sections, last used in [`LRU_ORDER`], a
    /// couple of milliseconds apart, into which the heaviest external-heavy
    /// candidate whose state is within the per-entry cap is cold-started. Its
    /// plan indexes weigh at least half its state (φ is 0.21), so its whole
    /// weight is over the side store's quarter of the resident budget, though
    /// within the budget itself, as every freshly built entry within the cap
    /// is (T10(b)), so the store does not evict it at once; mini-moka refuses
    /// it, and the side store keeps it alone. Returns the cache and the
    /// candidate, weighed as stored.
    fn heavy_entry_kept_alone_beside_a_full_main_cache() -> (StructuralCache, CacheEntry) {
        let cache = full_cache_of(BOUND_RESIDENT_SECTIONS, LONG_TTL, LONG_TTL);
        for object_id in LRU_ORDER {
            std::thread::sleep(Duration::from_millis(2));
            drop(hit(&cache, object_id));
        }
        let &((sections, rows), _) = external_heavy_within(cache.max_entry_bytes())
            .last()
            .expect("fixture: an external-heavy state within the cap");
        let candidate = weighed(CANDIDATE, 1, external_heavy_state(sections, rows));
        assert!(
            candidate.state_weight_bytes <= cache.max_entry_bytes()
                && candidate.index_weight_bytes * 2 >= candidate.state_weight_bytes,
            "fixture: a state within the {} B cap whose indexes weigh at least half of it; \
             state {} B, indexes {} B",
            cache.max_entry_bytes(),
            candidate.state_weight_bytes,
            candidate.index_weight_bytes
        );
        assert!(
            candidate.weight_bytes() > cache.inner.side_weight_bound,
            "fixture: the candidate ({} B) is alone over the side store's {} B quarter",
            candidate.weight_bytes(),
            cache.inner.side_weight_bound
        );
        assert!(matches!(cache.lookup(CANDIDATE), CacheLookup::Miss));
        assert!(
            cold_start(&cache, candidate.clone()),
            "a refused admission still stores the entry"
        );
        assert!(
            in_side_store(&cache, CANDIDATE) && !resident_in_main(&cache, CANDIDATE),
            "fixture: mini-moka refused the candidate and the side store kept it"
        );
        (cache, candidate)
    }

    /// W4-T1 (the decider's test): an entry whose state is within the cap
    /// but whose plan indexes weigh at least half of it is kept alone in the
    /// side store over its quarter, and the store brings the cache back
    /// within the resident budget by evicting the main cache's least recently
    /// used residents — and only as many as it must: the most recently used
    /// stay. The heavy entry is still served as a hit, and an evicted
    /// resident's next lookup is a miss, never an uncacheable verdict.
    ///
    /// Without the resident bound the three residents stay beside the heavy
    /// entry, and the total — the first precondition below — is past the
    /// resident budget.
    #[test]
    fn heavy_indexed_side_store_entry_evicts_the_least_recently_used_residents_to_fit() {
        let (cache, candidate) = heavy_entry_kept_alone_beside_a_full_main_cache();
        let budget = resident_budget(&cache);
        let resident = whole_weight(BOUND_RESIDENT_SECTIONS);
        let heavy = candidate.weight_bytes();
        let residents = RESIDENTS.len();
        assert!(
            residents as u64 * resident + heavy > budget,
            "fixture: the residents ({resident} B each) and the heavy entry ({heavy} B) are past \
             the {budget} B resident budget"
        );
        // Least recently used first, until within the budget.
        let evicted = (1..=residents)
            .find(|&k| (residents - k) as u64 * resident + heavy <= budget)
            .expect("emptying the main cache makes room");
        assert!(
            evicted < residents,
            "fixture: the heavy entry fits beside a resident, so the most recently used one stays"
        );

        assert!(
            cache.weight_bytes() <= budget,
            "the cache holds {} B, past its {budget} B resident budget",
            cache.weight_bytes()
        );
        for (rank, &object_id) in LRU_ORDER.iter().enumerate() {
            assert_eq!(
                resident_in_main(&cache, object_id),
                rank >= evicted,
                "resident {object_id}, number {} of three by last use, least recent first: the \
                 {evicted} least recently used are evicted, the others stay",
                rank + 1
            );
        }
        assert_eq!(cache.entry_count(), (1 + residents - evicted) as u64);
        for round in 1..=3 {
            match cache.lookup(CANDIDATE) {
                CacheLookup::Hit(guard) => {
                    assert_same_cut(guard.entry(), &candidate, &format!("lookup {round}"))
                }
                other => panic!("lookup {round} of the heavy entry must hit; got {other:?}"),
            }
            assert!(
                cache.weight_bytes() <= budget,
                "after lookup {round}'s put-back, within the resident budget"
            );
        }
        for &object_id in &LRU_ORDER[..evicted] {
            assert!(
                matches!(cache.lookup(object_id), CacheLookup::Miss),
                "evicted resident {object_id} misses: capacity, never an uncacheable verdict"
            );
            assert_eq!(
                cache.take_unreported(object_id),
                None,
                "no verdict recorded"
            );
        }
        for &object_id in &LRU_ORDER[evicted..] {
            drop(hit(&cache, object_id));
        }
    }

    /// W4-T2: with W4-T1's heavy entry in the side store, every store leaves
    /// the cache within the resident budget at once — summed over each
    /// resident key's slot weight, not mini-moka's counter (which only a
    /// `sync` updates), and read without a `sync`
    /// ([`StructuralCache::unsynced_weight_bytes_for_test`]):
    ///
    /// 1. a new key that mini-moka admits into the room it has, chosen to
    ///    fit within the budget and leave less slack than step 2 grows by;
    /// 2. the put-back of a resident that grew (its state within the cap,
    ///    its indexes now heavy): an update of its key, which mini-moka never
    ///    refuses and counts only at its next `sync`;
    /// 3. another new key that mini-moka admits.
    ///
    /// Steps 2 and 3 each alone would take the total past the budget (the
    /// preconditions), while the total before each was within: so a bound
    /// enforced only when the side store keeps an entry misses both, and one
    /// that reads mini-moka's counter before applying the store's pending
    /// write misses step 2.
    #[test]
    fn every_store_beside_a_heavy_side_store_entry_holds_the_resident_budget_unsynced() {
        let (cache, candidate) = heavy_entry_kept_alone_beside_a_full_main_cache();
        let budget = resident_budget(&cache);
        let cap = cache.max_entry_bytes();
        let resident = whole_weight(BOUND_RESIDENT_SECTIONS);
        let unsynced = || cache.unsynced_weight_bytes_for_test();
        assert!(
            unsynced() <= budget,
            "after the heavy entry's store: {} B held, past the {budget} B resident budget",
            unsynced()
        );
        let grown_id = *LRU_ORDER
            .iter()
            .rfind(|&&object_id| resident_in_main(&cache, object_id))
            .expect("fixture: a resident stays beside the heavy entry");
        let heavy_shapes = external_heavy_within(cap);
        let heaviest = heavy_shapes.last().expect("fixture: a heavy shape").1;
        // Every shape within the cap, lightest first: resident-like ones of
        // up to the residents' sections, and the external-heavy ones.
        let mut shapes: Vec<(ObjectState, u64)> = (0..=BOUND_RESIDENT_SECTIONS)
            .map(|sections| (state_with_sections(sections), whole_weight(sections)))
            .chain(
                heavy_shapes
                    .iter()
                    .map(|&((sections, rows), slot)| (external_heavy_state(sections, rows), slot)),
            )
            .collect();
        shapes.sort_by_key(|(_, slot)| *slot);
        // Mini-moka's room now: its capacity, in whole weigher units, less
        // what it holds besides the side store's one entry.
        let room = || {
            let side = side_bytes(&cache);
            assert_eq!(
                side,
                candidate.weight_bytes(),
                "fixture: the heavy entry is still the side store's one entry"
            );
            ((cache.inner.resident_bytes - cache.inner.side_weight_bound) / SLOT_UNIT_BYTES
                * SLOT_UNIT_BYTES)
                .checked_sub(unsynced() - side)
                .expect("mini-moka holds no more than its capacity once its writes are applied")
        };
        // Admit `object_id` of the lightest shape whose slot lies in
        // `(above, at_most]`, within mini-moka's room; returns the slot.
        let admit = |object_id: u64, above: u64, at_most: u64| {
            let room = room();
            let (state, slot) = shapes
                .iter()
                .find(|&&(_, slot)| slot > above && slot <= at_most.min(room))
                .cloned()
                .unwrap_or_else(|| {
                    panic!("fixture: a shape of slot in ({above}, {at_most}] within {room} B")
                });
            assert!(matches!(cache.lookup(object_id), CacheLookup::Miss));
            assert!(cold_start(&cache, cut(object_id, 1, state)));
            assert!(
                resident_in_main(&cache, object_id),
                "fixture: mini-moka admitted {object_id} ({slot} B into {room} B of room)"
            );
            slot
        };

        // 1. A new key that fits, leaving less slack than the growth below.
        let slack = budget - unsynced();
        let filler = admit(
            FILLER_OBJECT_ID,
            slack.saturating_sub(heaviest - resident),
            slack,
        );
        assert!(
            unsynced() <= budget,
            "after the filler's admission ({filler} B): {} B held, past the {budget} B \
             resident budget",
            unsynced()
        );

        // 2. A resident that grew.
        let before = unsynced();
        let &((sections, rows), grown_slot) = heavy_shapes
            .iter()
            .find(|&&(_, slot)| before - resident + slot > budget)
            .unwrap_or_else(|| {
                panic!(
                    "fixture: a shape within the cap whose put-back passes the budget: {before} \
                     B held of {budget} B, the resident {resident} B, the heaviest shape \
                     {heaviest} B"
                )
            });
        let grown = cut(grown_id, 2, external_heavy_state(sections, rows));
        let guard = hit(&cache, grown_id);
        assert_eq!(guard.put_back(grown), PutBack::Stored);
        assert!(
            unsynced() <= budget,
            "after the grown resident's put-back ({resident} B to {grown_slot} B): {} B held, \
             past the {budget} B resident budget",
            unsynced()
        );

        // 3. A new key that does not fit beside what is held.
        let slack = budget - unsynced();
        let slot = admit(NEW_OBJECT_ID, slack, u64::MAX);
        assert!(
            unsynced() <= budget,
            "after the new entry's admission ({slot} B over {slack} B of slack): {} B held, past \
             the {budget} B resident budget",
            unsynced()
        );
        assert!(cache.weight_bytes() <= budget);
    }

    /// An entry for `object_id` whose state and plan indexes weigh
    /// `state_bytes` and `index_bytes`, set by hand: stored straight through
    /// `Control::store` ([`store_as_weighed`]), and through a checkout's
    /// `Drop`, nothing re-weighs it. Its state is empty, so a put-back that
    /// re-weighs it stores it light.
    fn hand_weighed(object_id: u64, state_bytes: u64, index_bytes: u64) -> CacheEntry {
        CacheEntry {
            state_weight_bytes: state_bytes,
            index_weight_bytes: index_bytes,
            ..entry(object_id, 1)
        }
    }

    /// Store `entry` as a put-back's settle does — `Control::store`, the
    /// resident bound included — at the weights it carries, with no epoch or
    /// cap check. Returns the entries the store removed, to be freed outside
    /// the lock.
    fn store_as_weighed(cache: &StructuralCache, entry: CacheEntry) -> Freed {
        let inner = &cache.inner;
        let mut freed = Freed::new();
        inner.lock().store(
            entry,
            inner.idle_ttl,
            inner.side_weight_bound,
            inner.resident_bytes,
            &mut freed,
        );
        freed
    }

    /// The resident-bound policy fixtures' per-entry cap, so the configured
    /// budget is four times it, the resident budget 484,000 B, mini-moka's
    /// share 363,000 B and the side store's quarter 121,000 B.
    const POLICY_CAP: u64 = 100_000;

    /// A cache of [`POLICY_CAP`] holding `residents`, each stored at its
    /// hand-set weights and then read [`RESIDENT_READS`] times, so a new
    /// key's admission over them is refused.
    fn policy_cache(residents: &[CacheEntry]) -> StructuralCache {
        let cache = cache_with(4 * POLICY_CAP);
        assert_eq!(cache.max_entry_bytes(), POLICY_CAP);
        for resident in residents {
            assert!(store_as_weighed(&cache, resident.clone()).is_empty());
        }
        for _ in 0..RESIDENT_READS {
            for resident in residents {
                drop(hit(&cache, resident.object_id));
            }
        }
        for resident in residents {
            assert!(
                resident_in_main(&cache, resident.object_id),
                "fixture: resident {} in mini-moka",
                resident.object_id
            );
        }
        cache
    }

    /// The object ids of `freed`'s entries, sorted.
    fn freed_ids(freed: &Freed) -> Vec<u64> {
        let mut ids: Vec<u64> = freed.iter().map(|entry| entry.object_id).collect();
        ids.sort_unstable();
        ids
    }

    /// W4-T3 (a): the resident bound evicts filled slots before keys still
    /// checked out, whatever their order of use. A key was checked out
    /// before either filled resident was last used — so by time alone it is
    /// the least recently used — and a heavy entry refused into the side
    /// store takes the total past the resident budget by less than one
    /// resident: the bound evicts the least recently used filled resident
    /// alone. The checked-out key stays resident, its put-back refills it,
    /// and the eviction advances no epoch and records no verdict. Weights
    /// are hand-set ([`hand_weighed`]).
    #[test]
    fn resident_bound_evicts_a_filled_slot_before_a_key_checked_out_earlier() {
        const OLDER: u64 = 401;
        const NEWER: u64 = 402;
        const CHECKED_OUT: u64 = 403;
        let cache = policy_cache(&[
            hand_weighed(OLDER, 90_000, 10_000),
            hand_weighed(NEWER, 90_000, 10_000),
            hand_weighed(CHECKED_OUT, 90_000, 10_000),
        ]);
        let budget = resident_budget(&cache);
        let checked_out = hit(&cache, CHECKED_OUT);
        for object_id in [OLDER, NEWER] {
            std::thread::sleep(Duration::from_millis(2));
            drop(hit(&cache, object_id));
        }
        let epoch = cache.inner.lock().epoch;
        // f = 1.4: 240,000 B, alone over the 121,000 B quarter, and 540,000 B
        // in all, past the 484,000 B budget by less than one resident.
        let freed = store_as_weighed(&cache, hand_weighed(CANDIDATE, POLICY_CAP, 140_000));

        assert!(
            in_side_store(&cache, CANDIDATE),
            "fixture: mini-moka refused the heavy entry and the side store kept it"
        );
        assert!(
            cache.unsynced_weight_bytes_for_test() <= budget,
            "within the {budget} B resident budget"
        );
        assert!(
            !resident_in_main(&cache, OLDER) && resident_in_main(&cache, NEWER),
            "the least recently used filled resident is evicted, the other stays"
        );
        assert!(
            resident_in_main(&cache, CHECKED_OUT),
            "the key checked out stays resident: filled slots go first"
        );
        assert_eq!(
            freed_ids(&freed),
            vec![OLDER],
            "the evicted entry is handed out to be freed outside the lock"
        );
        assert_eq!(cache.inner.lock().epoch, epoch, "no epoch advance");
        assert_eq!(cache.take_unreported(OLDER), None, "no verdict recorded");
        assert!(matches!(cache.lookup(OLDER), CacheLookup::Miss));

        assert_eq!(checked_out.put_back_held(), PutBack::Stored);
        assert!(
            resident_in_main(&cache, CHECKED_OUT) && !in_side_store(&cache, CHECKED_OUT),
            "the checked-out key's put-back refills its resident slot"
        );
        assert!(cache.unsynced_weight_bytes_for_test() <= budget);
    }

    /// W4-T3 (b): a key still checked out is evicted only while the total is
    /// still past the resident budget once every filled slot has gone, and
    /// its later put-back is admitted again. The checked-out key and the
    /// heavy entry refused into the side store each weigh 250,000 B (f =
    /// 1.5): together past the 484,000 B budget. One checked-out key alone
    /// beside the side store's entry needs f above about 1.42 on both at the
    /// cap: past every layer shape the resident-bound proptest weighs (about
    /// 0.9 at most), and close to the per-row ceiling a freshly built layout
    /// can reach (1.745, T10(b)), so the weights are hand-set
    /// ([`hand_weighed`]); several checked-out keys, or name-index capacity a
    /// long-cached entry kept, can reach the same path with real weights. The bound
    /// evicts the filled resident, then the checked-out key, freeing only
    /// the filled one's entry (the checked-out slot is empty); it advances
    /// no epoch, so the put-back is not refused as invalidated, and, the key
    /// gone, it is a new admission into mini-moka.
    #[test]
    fn resident_bound_evicts_a_checked_out_key_only_while_still_over_and_its_put_back_is_admitted()
    {
        const FILLED: u64 = 411;
        const CHECKED_OUT: u64 = 412;
        let cache = policy_cache(&[
            hand_weighed(FILLED, 90_000, 10_000),
            hand_weighed(CHECKED_OUT, POLICY_CAP, 150_000),
        ]);
        let budget = resident_budget(&cache);
        let checked_out = hit(&cache, CHECKED_OUT);
        std::thread::sleep(Duration::from_millis(2));
        drop(hit(&cache, FILLED));
        let epoch = cache.inner.lock().epoch;
        let freed = store_as_weighed(&cache, hand_weighed(CANDIDATE, POLICY_CAP, 150_000));

        assert!(
            in_side_store(&cache, CANDIDATE),
            "fixture: mini-moka refused the heavy entry and the side store kept it"
        );
        assert!(
            !resident_in_main(&cache, FILLED) && !resident_in_main(&cache, CHECKED_OUT),
            "the filled resident first, then — still past the budget — the checked-out key"
        );
        assert_eq!(
            cache.unsynced_weight_bytes_for_test(),
            250_000,
            "the heavy entry alone remains"
        );
        assert_eq!(
            freed_ids(&freed),
            vec![FILLED],
            "only the filled slot held an entry to free"
        );
        assert_eq!(cache.inner.lock().epoch, epoch, "no epoch advance");
        for object_id in [FILLED, CHECKED_OUT] {
            assert_eq!(
                cache.take_unreported(object_id),
                None,
                "no verdict recorded"
            );
        }

        assert_eq!(
            checked_out.put_back_held(),
            PutBack::Stored,
            "the evicted checked-out key's put-back is accepted, not refused as invalidated"
        );
        assert!(
            resident_in_main(&cache, CHECKED_OUT) && !in_side_store(&cache, CHECKED_OUT),
            "and admitted into mini-moka again, re-weighed light"
        );
        assert!(cache.unsynced_weight_bytes_for_test() <= budget);
        drop(hit(&cache, CHECKED_OUT));
    }

    // ── An entry too heavy to hold (`Control::store`, `held_entry_bound`) ───

    /// T14(a): a new key whose whole weight is over the resident budget — a
    /// state at the cap and a hand-set index four times it, 500,000 B
    /// against the 484,000 B resident budget (and over mini-moka's whole
    /// 363,000 B) — is evicted at its own store, before anything is placed:
    /// it is in neither store, it is the only entry freed (no main-cache
    /// resident is evicted on its account, nor the light entry mini-moka
    /// refused into the side store beside it, which a placement through
    /// `Control::keep_in_side_store` would flush to make room for it), the
    /// cache is within its resident budget, no
    /// epoch advances, no verdict is recorded, and its next lookup misses.
    /// The cold start that follows, at build capacities, is then stored and
    /// served, and not evicted again: the heaviest external-heavy layout
    /// within the cap, whose fresh indexes weigh the most of any
    /// [`external_heavy_within`] shape.
    ///
    /// Without the rule, mini-moka refuses the entry, the side store keeps it
    /// alone, and the resident bound evicts every resident and then stops
    /// above its budget, which its debug assertion reports.
    #[test]
    fn store_of_a_new_key_over_the_resident_budget_evicts_it_at_once_and_its_cold_start_stays() {
        const RESIDENT_IDS: [u64; 3] = [421, 422, 423];
        const SIDE_NEIGHBOUR: u64 = 424;
        let cache = policy_cache(&RESIDENT_IDS.map(|id| hand_weighed(id, 90_000, 10_000)));
        let budget = resident_budget(&cache);
        // A side-store neighbour: over mini-moka's room beside the residents,
        // never read, so refused, and within the side store's quarter.
        let main_bytes = (cache.inner.resident_bytes - cache.inner.side_weight_bound)
            / SLOT_UNIT_BYTES
            * SLOT_UNIT_BYTES;
        let room = main_bytes - RESIDENT_IDS.len() as u64 * 100_000;
        let neighbour = hand_weighed(SIDE_NEIGHBOUR, 50_000, 20_000);
        assert!(
            neighbour.weight_bytes() > room
                && neighbour.weight_bytes() <= cache.inner.side_weight_bound,
            "fixture: the neighbour ({} B) is over mini-moka's {room} B of room and within the \
             side store's {} B quarter",
            neighbour.weight_bytes(),
            cache.inner.side_weight_bound
        );
        assert!(store_as_weighed(&cache, neighbour).is_empty());
        assert!(
            in_side_store(&cache, SIDE_NEIGHBOUR) && !resident_in_main(&cache, SIDE_NEIGHBOUR),
            "fixture: mini-moka refused the neighbour and the side store kept it"
        );
        let heavy = hand_weighed(CANDIDATE, POLICY_CAP, 4 * POLICY_CAP);
        assert!(
            heavy.weight_bytes() > held_entry_bound(budget),
            "fixture: {} B is over the {budget} B resident budget",
            heavy.weight_bytes()
        );
        let epoch = cache.inner.lock().epoch;
        let bound_evictions = cache.resident_bound_evictions_for_test();

        let freed = store_as_weighed(&cache, heavy);

        assert!(
            !in_side_store(&cache, CANDIDATE) && !resident_in_main(&cache, CANDIDATE),
            "the heavy entry is held in neither store"
        );
        assert!(
            in_side_store(&cache, SIDE_NEIGHBOUR),
            "the side store's other entry stays: none is evicted on the heavy entry's account"
        );
        assert_eq!(
            freed_ids(&freed),
            vec![CANDIDATE],
            "the heavy entry alone is handed out to be freed"
        );
        for object_id in RESIDENT_IDS {
            assert!(
                resident_in_main(&cache, object_id),
                "resident {object_id} stays: none is evicted on the heavy entry's account"
            );
        }
        assert_eq!(
            cache.resident_bound_evictions_for_test(),
            bound_evictions,
            "the resident bound evicted nothing"
        );
        assert!(
            cache.unsynced_weight_bytes_for_test() <= budget,
            "within the {budget} B resident budget"
        );
        assert_eq!(cache.oversized_evictions_for_test(), 1);
        assert_eq!(cache.inner.lock().epoch, epoch, "no epoch advance");
        assert_eq!(
            cache.take_unreported(CANDIDATE),
            None,
            "no verdict recorded"
        );
        assert!(matches!(cache.lookup(CANDIDATE), CacheLookup::Miss));

        // The cold start that follows, at build capacities.
        let &((sections, rows), _) = external_heavy_within(POLICY_CAP)
            .last()
            .expect("fixture: an external-heavy state within the cap");
        let rebuilt = weighed(CANDIDATE, 2, external_heavy_state(sections, rows));
        assert!(cold_start(&cache, rebuilt.clone()));
        match cache.lookup(CANDIDATE) {
            CacheLookup::Hit(guard) => assert_same_cut(guard.entry(), &rebuilt, "the cold start"),
            other => panic!("the cold-started entry must hit; got {other:?}"),
        }
        assert!(cache.unsynced_weight_bytes_for_test() <= budget);
        assert_eq!(
            cache.oversized_evictions_for_test(),
            1,
            "the rebuilt entry ({} B) is within the held-entry bound and is not evicted again",
            rebuilt.weight_bytes()
        );
    }

    /// T14(a)'s boundary: an entry weighing exactly `held_entry_bound` — the
    /// whole 484,000 B resident budget, a state at the cap and a hand-set
    /// index of the rest — is held, not evicted: the rule evicts only an
    /// entry heavier than the bound. Over mini-moka's whole capacity, it is
    /// refused into the side store and kept there alone, and the resident
    /// bound evicts the main cache's resident to make room for it, which
    /// leaves the cache exactly at its resident budget.
    #[test]
    fn store_of_an_entry_at_exactly_the_held_entry_bound_keeps_it() {
        const RESIDENT: u64 = 441;
        let cache = policy_cache(&[hand_weighed(RESIDENT, 90_000, 10_000)]);
        let budget = resident_budget(&cache);
        let bound = held_entry_bound(budget);
        assert_eq!(bound, budget, "fixture: the bound is the resident budget");
        let at_bound = hand_weighed(CANDIDATE, POLICY_CAP, bound - POLICY_CAP);
        assert_eq!(
            at_bound.weight_bytes(),
            bound,
            "fixture: exactly at the bound"
        );

        let freed = store_as_weighed(&cache, at_bound);

        assert!(
            in_side_store(&cache, CANDIDATE) && !resident_in_main(&cache, CANDIDATE),
            "an entry at the bound is kept, in the side store"
        );
        assert_eq!(
            cache.oversized_evictions_for_test(),
            0,
            "not evicted for its weight"
        );
        assert!(
            !resident_in_main(&cache, RESIDENT),
            "the resident bound evicted the resident to make room"
        );
        assert_eq!(freed_ids(&freed), vec![RESIDENT]);
        assert_eq!(
            cache.unsynced_weight_bytes_for_test(),
            budget,
            "exactly at the budget"
        );
    }

    /// T14(b): the put-back of a resident key — an update, which mini-moka
    /// never refuses — whose whole weight is over the resident budget is
    /// evicted at once, with its key, and every other resident stays. The
    /// key is checked out when its heavy entry is stored, as at a put-back
    /// (weights hand-set, [`store_as_weighed`]); no epoch advances, so the
    /// checkout's own later settle is still accepted.
    ///
    /// Without the rule the update is applied, which takes mini-moka past its
    /// capacity: the other residents are evicted for it before the heavy
    /// entry goes.
    #[test]
    fn store_of_a_resident_key_over_the_resident_budget_evicts_its_key_and_keeps_the_others() {
        const OLDER: u64 = 431;
        const NEWER: u64 = 432;
        const GROWN: u64 = 433;
        let cache = policy_cache(&[
            hand_weighed(OLDER, 90_000, 10_000),
            hand_weighed(NEWER, 90_000, 10_000),
            hand_weighed(GROWN, 90_000, 10_000),
        ]);
        let budget = resident_budget(&cache);
        let checked_out = hit(&cache, GROWN);
        assert!(
            resident_in_main(&cache, GROWN),
            "fixture: the checked-out key stays resident, its slot empty"
        );
        let epoch = cache.inner.lock().epoch;
        let bound_evictions = cache.resident_bound_evictions_for_test();

        let freed = store_as_weighed(&cache, hand_weighed(GROWN, POLICY_CAP, 4 * POLICY_CAP));

        assert!(
            !resident_in_main(&cache, GROWN) && !in_side_store(&cache, GROWN),
            "the heavy put-back's key is gone and nothing of it is held"
        );
        assert!(
            resident_in_main(&cache, OLDER) && resident_in_main(&cache, NEWER),
            "the other residents stay"
        );
        assert_eq!(
            freed_ids(&freed),
            vec![GROWN],
            "the heavy entry alone is handed out to be freed (the checked-out slot was empty)"
        );
        assert_eq!(cache.resident_bound_evictions_for_test(), bound_evictions);
        assert_eq!(cache.oversized_evictions_for_test(), 1);
        assert!(cache.unsynced_weight_bytes_for_test() <= budget);
        assert_eq!(cache.inner.lock().epoch, epoch, "no epoch advance");
        assert_eq!(cache.take_unreported(GROWN), None, "no verdict recorded");

        assert_eq!(
            checked_out.put_back_held(),
            PutBack::Stored,
            "the checkout's own settle is not refused as invalidated"
        );
        assert!(cache.unsynced_weight_bytes_for_test() <= budget);
    }

    // ── T14n: name-index capacity a long-cached entry keeps ─────────────────
    //
    // A name-index vector keeps the largest capacity it ever had, through a
    // rollback (`PlanIndexes::restore_name_index`) and a committed delete
    // (`name_remove`) alike, so plans that grow different parents' vectors
    // add up on one cached entry with no bound relative to its state. These
    // tests drive real deltas — name vectors only, no time samples — until
    // the entry's put-back weighs more than the resident budget, and check
    // the store evicts it there.
    //
    // At hashbrown group widths 8 (aarch64) and 16 (x86_64) alike: every
    // weight read here is the weigher's arithmetic over tables a put-back
    // has compacted to their length, whose bucket count hashbrown sets from
    // the length alone (`domain::alloc_model::table_bytes`), and over `Vec`
    // capacities; the cycles' growth is a name vector's, and the crossing's
    // is measured on a scratch copy of the same entry rather than predicted.

    /// The T14n caches' budget: 1 MiB, the smallest non-zero budget in whole
    /// MiB, so the fewest planned sections take an entry past its resident
    /// budget.
    const RETENTION_BUDGET_BYTES: u64 = 1024 * 1024;

    /// The residents filling the T14n cache's main cache: many and light, so
    /// the resident bound, evicting as the entry grows, leaves some beside it
    /// at the crossing.
    const RETENTION_RESIDENTS: u64 = 32;

    /// How often each T14n resident is read: past the frequency sketch's
    /// ceiling of 15, so every resident counts 15 and no new key, however
    /// often the cycles look it up, is ever admitted over one (mini-moka
    /// admits a new key only on a frequency strictly above its victims').
    const SATURATING_READS: usize = 16;

    /// How a T14n cycle leaves name-index capacity behind.
    #[derive(Debug, Clone, Copy)]
    enum Retention {
        /// Plan the creation of sections under one parent, ending in a delta
        /// `Err` (a duplicate id), and put the entry back rolled back.
        RolledBack,
        /// Commit the creation of sections under one parent, then commit
        /// their deletion.
        Committed,
    }

    impl Retention {
        /// `(parents, sections a build-up cycle creates, sections the
        /// crossing cycle creates)`. Each count is one less than (rolled
        /// back) or exactly (committed) a power of two, so the parent's
        /// vector, one entry long before it, grows to a power of two. The
        /// crossing creates more than a build-up cycle, so its growth is
        /// over the weight the build-up leaves the residents by more than
        /// one resident: some stay beside the entry until it crosses. A
        /// committed cycle's creations are bounded by the per-entry cap,
        /// since the state holds them until their deletion commits.
        fn sizes(self) -> (u64, u64, u64) {
            match self {
                Self::RolledBack => (48, 2_047, 8_191),
                Self::Committed => (96, 512, 1_024),
            }
        }
    }

    /// A normalized state of `parents` sections under the root, each with one
    /// child of its own. Every parent keeps its name-index key while a cycle
    /// creates and removes others under it (a rollback drops only keys its
    /// plan created), so its vector keeps whatever capacity the cycle grew.
    fn parents_with_one_child(parents: u64) -> ObjectState {
        let section = |section_id: u64, parent_id: u64, name: String, order: u64| Section {
            section_id,
            parent_id,
            name,
            section_type: 6,
            section_order: order as u32,
            first_version: 1,
            last_version: 1,
        };
        let mut state = state_with_sections(0);
        for parent in 1..=parents {
            let child = parents + parent;
            state
                .sections
                .insert(parent, section(parent, 0, format!("p{parent}"), parent));
            state
                .sections
                .insert(child, section(child, parent, "c".to_owned(), 1));
            state.children.insert(parent, vec![child]);
        }
        state.children.insert(0, (1..=parents).collect());
        state
    }

    /// The first id of the sections a T14n cycle creates under `parent`: a
    /// range of its own, clear of the fixture's ids and every other parent's.
    fn created_ids(parent: u64) -> u64 {
        1_000_000 + parent * 100_000
    }

    /// The creation of `count` sections under `parent`.
    fn creations(parent: u64, count: u64) -> OwnedDelta {
        OwnedDelta {
            struct_commands: (0..count)
                .map(|n| {
                    StructCommand::CreateSection(CommandCreateSection {
                        parent_id: parent,
                        section_id: created_ids(parent) + n,
                        section_name: format!("n{n}"),
                        section_type: 6,
                    })
                })
                .collect(),
            ..OwnedDelta::default()
        }
    }

    /// [`creations`], then one more creation of the first id: refused
    /// `DuplicateSection` once all the others have applied.
    fn failing_creations(parent: u64, count: u64) -> OwnedDelta {
        let mut delta = creations(parent, count);
        delta
            .struct_commands
            .extend(creations(parent, 1).struct_commands);
        delta
    }

    /// The deletion of the `count` sections [`creations`] made under `parent`.
    fn deletions(parent: u64, count: u64) -> OwnedDelta {
        OwnedDelta {
            struct_commands: (0..count)
                .map(|n| {
                    StructCommand::DeleteSection(oodf_ffi::CommandDeleteSection {
                        parent_id: parent,
                        section_id: created_ids(parent) + n,
                    })
                })
                .collect(),
            ..OwnedDelta::default()
        }
    }

    /// Check `object_id` out, plan `delta` in place — refused by its last
    /// command — and put the entry back, rolled back. Returns how it settled,
    /// and the state before the plan, which the rollback restores.
    fn rolled_back_plan(
        cache: &StructuralCache,
        object_id: u64,
        delta: OwnedDelta,
    ) -> (PutBack, CompactState) {
        let mut guard = hit(cache, object_id);
        let state = guard.entry().state.clone();
        let plan = plan_unbounded(&mut guard, delta);
        assert!(
            matches!(plan, Err(DeltaError::DuplicateSection { .. })),
            "fixture: the delta is refused at its duplicate creation; got {plan:?}"
        );
        (guard.put_back_held(), state)
    }

    /// Check `object_id` out, plan `delta` in place, commit it at `seq` and
    /// put the entry back. Returns how it settled, and the committed state.
    fn committed_plan(
        cache: &StructuralCache,
        object_id: u64,
        delta: OwnedDelta,
        seq: u64,
    ) -> (PutBack, CompactState) {
        let mut guard = hit(cache, object_id);
        let mutations = match plan_unbounded(&mut guard, delta) {
            Ok(BoundedPlan::Within { mutations, .. }) => mutations,
            other => panic!("fixture: the delta applies; got {other:?}"),
        };
        guard
            .advance_committed(&mutations, seq, format!("{seq:040x}"))
            .expect("no dangling parent is adopted");
        let state = guard.entry().state.clone();
        (guard.put_back_held(), state)
    }

    /// One T14n cycle's crossing step under `parent`, creating `count`
    /// sections: the rolled-back plan, or the committed creation (its
    /// deletion is not the crossing's: the creation alone takes the entry
    /// over). Returns how it settled, the seq the entry is at, and its state.
    fn retention_crossing_step(
        route: Retention,
        cache: &StructuralCache,
        object_id: u64,
        parent: u64,
        count: u64,
        seq: u64,
    ) -> (PutBack, u64, CompactState) {
        match route {
            Retention::RolledBack => {
                let (settled, state) =
                    rolled_back_plan(cache, object_id, failing_creations(parent, count));
                (settled, seq, state)
            }
            Retention::Committed => {
                let (settled, state) =
                    committed_plan(cache, object_id, creations(parent, count), seq + 1);
                (settled, seq + 1, state)
            }
        }
    }

    /// The whole weight, state and plan indexes, of `object_id`'s
    /// side-store entry, split `(state, indexes)`; `None` when the side
    /// store does not hold it.
    fn side_weights(cache: &StructuralCache, object_id: u64) -> Option<(u64, u64)> {
        cache
            .inner
            .lock()
            .side
            .get(&object_id)
            .map(|entry| (entry.state_weight_bytes, entry.index_weight_bytes))
    }

    /// How much the crossing step — under a parent no cycle has grown —
    /// raises the whole weight of a fresh entry for `base`, measured on a
    /// scratch cache with room for it: the same growth the T14n entry's own
    /// crossing step makes, its state back at `base` and that parent's
    /// vector as fresh.
    fn retention_crossing_growth(route: Retention, base: &ObjectState) -> u64 {
        let (parents, _, crossing) = route.sizes();
        let scratch = cache_with(BUDGET_BYTES);
        assert!(cold_start(&scratch, cut(OTHER_OBJECT_ID, 1, base.clone())));
        let before = hit(&scratch, OTHER_OBJECT_ID).entry().weight_bytes();
        let (settled, ..) =
            retention_crossing_step(route, &scratch, OTHER_OBJECT_ID, parents, crossing, 1);
        assert_eq!(settled, PutBack::Stored, "fixture: the scratch entry stays");
        let after = hit(&scratch, OTHER_OBJECT_ID).entry().weight_bytes();
        after
            .checked_sub(before)
            .expect("fixture: the crossing step grows the entry")
    }

    /// Fill `cache`'s main cache with [`RETENTION_RESIDENTS`] equal
    /// hand-weighed residents, leaving it less room than any real entry
    /// weighs, and read each [`SATURATING_READS`] times. Returns their ids.
    fn fill_main_with_saturated_residents(cache: &StructuralCache) -> Vec<u64> {
        let inner = &cache.inner;
        let main_bytes =
            (inner.resident_bytes - inner.side_weight_bound) / SLOT_UNIT_BYTES * SLOT_UNIT_BYTES;
        let each = main_bytes / RETENTION_RESIDENTS / SLOT_UNIT_BYTES * SLOT_UNIT_BYTES;
        let residents: Vec<u64> = (0..RETENTION_RESIDENTS).map(|n| 1_000 + n).collect();
        for &object_id in &residents {
            assert!(store_as_weighed(cache, hand_weighed(object_id, each, 0)).is_empty());
        }
        for _ in 0..SATURATING_READS {
            for &object_id in &residents {
                drop(hit(cache, object_id));
            }
        }
        for &object_id in &residents {
            assert!(
                resident_in_main(cache, object_id),
                "fixture: resident {object_id} in mini-moka"
            );
        }
        residents
    }

    /// The T14n scenario for `route` (the two tests below). A 1 MiB cache
    /// whose main cache is full of saturated residents keeps the entry — a
    /// state of parents with one child each — in its side store: refused by
    /// mini-moka on every put-back, its frequency never above a resident's.
    /// Build-up cycles, each on a parent of its own, leave its index heavier
    /// while its state weighs what it did (hard preconditions, checked after
    /// every cycle), and the resident bound evicts residents as it grows,
    /// as it does for any heavy side-store entry; no cycle alone takes a
    /// fresh entry over the resident budget. Once the crossing step's growth
    /// would take the entry over, that step's put-back is evicted at once:
    /// counted, held in neither store, the cache within its resident budget,
    /// every resident still beside it staying, no epoch advance, no verdict,
    /// and a lookup miss. A cold start of the same state then stores it with
    /// its indexes fresh, within T10(b)'s fresh-index bound, and it is not
    /// evicted again.
    ///
    /// Without the rule, mini-moka refuses the crossing put-back, the side
    /// store keeps it alone, and the resident bound evicts every resident and
    /// then stops above its budget, which its debug assertion reports.
    ///
    /// About 2.5 s per arm in a debug build (measured on aarch64): crossing
    /// a 1 MiB budget's resident budget through name-vector capacity alone
    /// takes some tens of thousands of planned sections, each plan also
    /// re-planned on a copy by the test-support shadow check. 1 MiB is the
    /// least budget the configuration accepts; a smaller one, reachable only
    /// by building the cache directly, would shrink the work but test a
    /// budget no deployment runs.
    fn retained_name_capacity_over_the_resident_budget_is_evicted_at_its_store(route: Retention) {
        let (parents, build_up, crossing) = route.sizes();
        let base = parents_with_one_child(parents);
        let crossing_growth = retention_crossing_growth(route, &base);
        let cache = cache_with(RETENTION_BUDGET_BYTES);
        let budget = resident_budget(&cache);
        let residents = fill_main_with_saturated_residents(&cache);
        assert!(matches!(cache.lookup(CANDIDATE), CacheLookup::Miss));
        assert!(cold_start(&cache, cut(CANDIDATE, 1, base.clone())));
        let (base_state, fresh_index) = side_weights(&cache, CANDIDATE)
            .expect("fixture: mini-moka refused the entry and the side store kept it");
        assert!(
            base_state + fresh_index + crossing_growth <= budget,
            "fixture: no cycle alone takes the fresh entry ({} B) over the {budget} B resident \
             budget; the crossing step grows it by {crossing_growth} B",
            base_state + fresh_index
        );

        // The build-up.
        let mut seq = 1;
        let mut parent = 1;
        let mut index = fresh_index;
        while base_state + index + crossing_growth <= budget {
            assert!(parent < parents, "fixture: a parent left for the build-up");
            match route {
                Retention::RolledBack => assert_eq!(
                    rolled_back_plan(&cache, CANDIDATE, failing_creations(parent, build_up)).0,
                    PutBack::Stored
                ),
                Retention::Committed => {
                    let (settled, _) =
                        committed_plan(&cache, CANDIDATE, creations(parent, build_up), seq + 1);
                    assert_eq!(settled, PutBack::Stored);
                    assert!(
                        side_weights(&cache, CANDIDATE).is_some()
                            && cache.unsynced_weight_bytes_for_test() <= budget,
                        "parent {parent}: the committed creations are kept in the side store, \
                         within the resident budget"
                    );
                    let (settled, _) =
                        committed_plan(&cache, CANDIDATE, deletions(parent, build_up), seq + 2);
                    assert_eq!(settled, PutBack::Stored);
                    seq += 2;
                }
            }
            let (state, grown) = side_weights(&cache, CANDIDATE).unwrap_or_else(|| {
                panic!("parent {parent}: the entry must still be in the side store")
            });
            assert!(!resident_in_main(&cache, CANDIDATE));
            assert_eq!(
                state, base_state,
                "parent {parent}: the state weighs what it did before the cycle"
            );
            assert!(
                grown > index,
                "parent {parent}: the cycle must leave the index heavier ({index} B to {grown} B)"
            );
            assert!(cache.unsynced_weight_bytes_for_test() <= budget);
            assert_eq!(cache.oversized_evictions_for_test(), 0, "parent {parent}");
            index = grown;
            parent += 1;
        }

        // The crossing.
        let beside: Vec<u64> = residents
            .iter()
            .copied()
            .filter(|&object_id| resident_in_main(&cache, object_id))
            .collect();
        assert!(
            !beside.is_empty(),
            "fixture: residents are beside the entry at the crossing"
        );
        let bound_evictions = cache.resident_bound_evictions_for_test();
        let epoch = cache.inner.lock().epoch;

        let (settled, seq, crossed) =
            retention_crossing_step(route, &cache, CANDIDATE, parent, crossing, seq);

        assert_eq!(
            settled,
            PutBack::Stored,
            "accepted, and evicted in the same critical section"
        );
        assert_eq!(cache.oversized_evictions_for_test(), 1);
        assert!(
            !in_side_store(&cache, CANDIDATE) && !resident_in_main(&cache, CANDIDATE),
            "the entry is held in neither store"
        );
        assert!(
            cache.unsynced_weight_bytes_for_test() <= budget,
            "within the {budget} B resident budget"
        );
        for &object_id in &beside {
            assert!(
                resident_in_main(&cache, object_id),
                "resident {object_id} stays: none is evicted on the entry's account"
            );
        }
        assert_eq!(cache.resident_bound_evictions_for_test(), bound_evictions);
        assert_eq!(cache.inner.lock().epoch, epoch, "no epoch advance");
        assert_eq!(
            cache.take_unreported(CANDIDATE),
            None,
            "no verdict recorded"
        );
        assert!(matches!(cache.lookup(CANDIDATE), CacheLookup::Miss));

        // The cold start that follows, its indexes built afresh.
        assert!(cold_start(&cache, compact_cut(CANDIDATE, seq, crossed)));
        let rebuilt = hit(&cache, CANDIDATE);
        let (state, index) = (
            rebuilt.entry().state_weight_bytes,
            rebuilt.entry().index_weight_bytes,
        );
        assert!(
            within_fresh_index_bound(state, index),
            "the rebuilt entry's indexes ({index} B) are within the fresh-index bound of its \
             {state} B state"
        );
        drop(rebuilt);
        assert_eq!(
            cache.oversized_evictions_for_test(),
            1,
            "the rebuilt entry is not evicted again"
        );
        assert!(cache.unsynced_weight_bytes_for_test() <= budget);
    }

    /// T14n, rolled-back arm: different deltas, each growing one parent's
    /// name vector and refused, put back rolled back
    /// ([`retained_name_capacity_over_the_resident_budget_is_evicted_at_its_store`]).
    #[test]
    fn rolled_back_plans_growing_name_vectors_past_the_resident_budget_evict_the_entry_at_its_store(
    ) {
        retained_name_capacity_over_the_resident_budget_is_evicted_at_its_store(
            Retention::RolledBack,
        );
    }

    /// T14n, committed arm: committed creations under one parent, then their
    /// committed deletion, a parent per cycle
    /// ([`retained_name_capacity_over_the_resident_budget_is_evicted_at_its_store`]).
    #[test]
    fn committed_grow_then_delete_cycles_past_the_resident_budget_evict_the_entry_at_its_store() {
        retained_name_capacity_over_the_resident_budget_is_evicted_at_its_store(
            Retention::Committed,
        );
    }

    // ── Clone-before-plan (SR-3.5, SR-10.4) ─────────────────────────────────

    /// Field names [`failing_delta`] sets before it fails, none of which the
    /// fixture entry holds: each is interned in the plan clone's own name
    /// interner.
    const NEW_NAMES: [&str; 2] = ["fresh-inline", "fresh-external"];

    /// A delta that creates section 300, sets two fields of names no row of
    /// the fixture holds ([`NEW_NAMES`], one inline and one external), and
    /// then fails `SetOrderOverflow` on a third field: its `set_order` passes
    /// validation, but added to the `next_set_order` the first two fields
    /// advanced it reaches the ceiling. `apply_delta` leaves the section and
    /// the two new rows — and their interned names — in place, so the failure
    /// is necessarily mid-delta.
    fn failing_delta() -> OwnedDelta {
        let set_field =
            |section_id: u64, key_name: &str, value: Vec<u8>, set_order: u64| oodf_ffi::SetField {
                section_id,
                key_name: key_name.to_owned(),
                ext_value_size: if value.len() == 32 { 64 } else { 0 },
                value_or_ext_hash: value,
                ext_value_index: 0,
                set_order,
            };
        OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: 300,
                section_name: "Doomed".to_owned(),
                section_type: 6,
            })],
            set_fields: vec![
                set_field(300, NEW_NAMES[0], vec![1, 2, 3], 0),
                set_field(1, NEW_NAMES[1], vec![0xEE; 32], 0),
                set_field(
                    2,
                    "never-set",
                    vec![4],
                    crate::domain::limits::MAX_DELTA_SET_ORDER - 1,
                ),
            ],
            ..OwnedDelta::default()
        }
    }

    /// A delta `Err` on the plan clone leaves the put-back entry as it was,
    /// its name interner included: the clone's interner is its own (T10), so
    /// the names the failed plan interned die with the clone and never reach
    /// the stored entry.
    #[test]
    fn planning_a_failing_delta_on_the_clone_leaves_the_put_back_entry_unchanged() {
        let cache = cache_with(BUDGET_BYTES);
        let original = cut(OBJECT_ID, 5, state_with_sections(3));
        assert!(cold_start(&cache, original.clone()));
        let guard = hit(&cache, OBJECT_ID);

        let mut plan = guard.plan_state(PlannedInserts::default());
        let err = apply_delta(&mut plan, Some(failing_delta())).expect_err("the delta fails");
        assert!(
            matches!(
                err,
                crate::domain::delta::DeltaError::SetOrderOverflow { .. }
            ),
            "fixture: the third field must be the one that fails; got {err:?}"
        );
        assert_ne!(
            plan, original.state,
            "fixture: the failing delta must have mutated the clone before failing"
        );
        for name in NEW_NAMES {
            assert!(
                plan.fields.names().id_of(name).is_some(),
                "fixture: the failed plan interned {name:?} in the clone"
            );
            assert!(
                guard.entry().state.fields.names().id_of(name).is_none(),
                "the held entry's interner must not see the clone's new name {name:?}"
            );
        }
        drop(plan);
        // A delta `Err` puts the original back (SR-3.4).
        let held = guard.entry().clone();
        assert!(guard.put_back(held).is_stored());

        let stored = hit(&cache, OBJECT_ID);
        assert_same_cut(
            stored.entry(),
            &original,
            "the entry after a failed plan on its clone",
        );
        let mut names: Vec<&str> = stored.entry().state.fields.names().names().collect();
        names.sort_unstable();
        assert_eq!(
            names,
            ["default"],
            "the stored entry's interner holds only its own rows' names"
        );
    }

    // ── Compaction before weighing (SR-3.3) ─────────────────────────────────

    /// Time samples per section in [`thinned_state`].
    const SAMPLES_PER_SECTION: u64 = 4;

    /// A compact state grown to `GROWN` sections — each with its own children
    /// list and [`SAMPLES_PER_SECTION`] time samples besides its field — and
    /// thinned to `KEPT` through the row stores' own removal
    /// (`purge_sections`, as a cascade removes rows): all four of its tables
    /// keep the bucket arrays the growth gave them. Still normalized: every
    /// remaining children list's parent is a live section or the root.
    fn thinned_state() -> CompactState {
        const GROWN: u64 = 200;
        const KEPT: u64 = 20;
        let mut state = compact(&with_time_samples(
            state_with_sections(GROWN),
            SAMPLES_PER_SECTION,
        ));
        for id in 1..=GROWN {
            state.children.insert(id, vec![GROWN + id]);
        }
        let removed: HashSet<u64> = (KEPT + 1..=GROWN).collect();
        for id in &removed {
            state.sections.remove(id);
            state.children.remove(id);
        }
        state.fields.purge_sections(&removed, |_, _| {});
        state.time_samples.purge_sections(&removed, |_, _| {});
        state
            .children
            .get_mut(&0)
            .expect("the root's children list")
            .retain(|&id| id <= KEPT);
        state
    }

    /// Each of a compact `state`'s four tables (a cache entry's): its name,
    /// its `capacity()`, and the capacity of a table built for its length.
    fn compact_table_capacities(state: &CompactState) -> [(&'static str, usize, usize); 4] {
        [
            (
                "sections",
                state.sections.capacity(),
                HashMap::<u64, Section>::with_capacity(state.sections.len()).capacity(),
            ),
            (
                "children",
                state.children.capacity(),
                HashMap::<u64, Vec<u64>>::with_capacity(state.children.len()).capacity(),
            ),
            (
                "fields",
                state.fields.capacity(),
                CompactFields::with_capacity(FieldRows::len(&state.fields)).capacity(),
            ),
            (
                "time_samples",
                state.time_samples.capacity(),
                CompactSamples::with_capacity(SampleRows::len(&state.time_samples)).capacity(),
            ),
        ]
    }

    /// Assert each of a compact `state`'s four tables (a cache entry's) has
    /// the capacity a table built for its length has: no larger bucket array
    /// than its length needs.
    fn assert_compact_tables_minimal(state: &CompactState, context: &str) {
        for (table, capacity, built_for_len) in compact_table_capacities(state) {
            assert_eq!(
                capacity, built_for_len,
                "{context}: the {table} table must be compacted to its length"
            );
        }
    }

    #[test]
    fn admit_and_put_back_store_a_thinned_state_compacted_and_weighed_compacted() {
        let thinned = thinned_state();
        for (table, capacity, built_for_len) in compact_table_capacities(&thinned) {
            assert!(
                capacity > built_for_len,
                "fixture: thinning must leave the {table} table oversized ({capacity} vs \
                 {built_for_len})"
            );
        }
        let uncompacted_weight = estimate_state_bytes(&thinned);
        let mut compacted = thinned.clone();
        compact_state(&mut compacted);
        let compacted_weight = estimate_state_bytes(&compacted);
        assert!(
            compacted_weight < uncompacted_weight,
            "fixture: thinning must leave the tables oversized ({compacted_weight} vs \
             {uncompacted_weight})"
        );

        // `admit` compacts the cold-started state before weighing it.
        let cache = cache_with(BUDGET_BYTES);
        let admitted = cache
            .admit(
                cache.begin_cold_start(),
                compact_cut(OBJECT_ID, 1, thinned.clone()),
            )
            .expect("admissible");
        assert_eq!(
            admitted.entry().state_weight_bytes,
            compacted_weight,
            "admit"
        );
        assert_compact_tables_minimal(&admitted.entry().state, "admit");
        assert_eq!(
            admitted.entry().state,
            thinned,
            "compaction changes no content"
        );
        admitted.discard();

        // `put_back` compacts the successor before weighing it.
        assert!(cold_start(
            &cache,
            cut(OTHER_OBJECT_ID, 1, state_with_sections(1))
        ));
        let guard = hit(&cache, OTHER_OBJECT_ID);
        assert!(guard
            .put_back(compact_cut(OTHER_OBJECT_ID, 2, thinned.clone()))
            .is_stored());
        let stored = hit(&cache, OTHER_OBJECT_ID);
        assert_eq!(
            stored.entry().state_weight_bytes,
            compacted_weight,
            "put_back"
        );
        assert_compact_tables_minimal(&stored.entry().state, "put_back");
        assert_eq!(
            stored.entry().state,
            thinned,
            "compaction changes no content"
        );
    }

    // ── The 32-bit slot weight and the per-entry cap ───────────────────────

    #[test]
    fn slot_weight_charges_4_byte_units_rounded_up_and_saturates_past_u32_max_units() {
        let most = u64::from(u32::MAX) * SLOT_UNIT_BYTES;
        assert_eq!(
            slot_weight(most + 1),
            u32::MAX,
            "one unit past u32::MAX saturates"
        );
        assert_eq!(slot_weight(u64::MAX), u32::MAX, "never wraps");
        assert_eq!(slot_weight(most), u32::MAX);
        assert_eq!(slot_weight(4_096), 1_024);
        assert_eq!(slot_weight(4_097), 1_025, "a part unit is charged whole");
        assert_eq!(slot_weight(1), 1);
        assert_eq!(slot_weight(0), 0);
    }

    /// The per-entry cap is a quarter of the budget up to `u32::MAX`, and
    /// `u32::MAX` from there on, so an entry whose state is at the cap and
    /// whose plan indexes weigh φ of it is charged its whole weight: never
    /// saturated, at the boundary or above it. 16 GiB − 4 bytes is the first
    /// budget whose quarter is `u32::MAX`; at exactly 16 GiB the quarter,
    /// 2^32, is one past it.
    #[test]
    fn max_entry_bytes_is_a_quarter_of_the_budget_capped_at_u32_max() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let ceiling = u64::from(u32::MAX);
        for (budget, want) in [
            (0, 0),
            (3, 0),
            (256 * 1024 * 1024, 64 * 1024 * 1024),
            (16 * GIB - 8, ceiling - 1),
            (16 * GIB - 4, ceiling),
            (16 * GIB, ceiling),
            (16 * GIB + 4, ceiling),
            (64 * GIB, ceiling),
            (u64::MAX, ceiling),
        ] {
            assert_eq!(max_entry_bytes_for(budget), want, "budget {budget}");
            let cache = StructuralCache::new(budget);
            assert_eq!(cache.max_entry_bytes(), want, "cache of budget {budget}");
            let whole = want + want * PLAN_INDEX_BUDGET_CENTI / 100;
            assert!(
                slot_weight(whole) < u32::MAX
                    && u64::from(slot_weight(whole)) * SLOT_UNIT_BYTES >= whole,
                "budget {budget}: an entry at the cap with its indexes is charged its whole weight"
            );
        }
    }

    /// The most one held entry may weigh is the resident budget, or the
    /// `u32::MAX` 4-byte units mini-moka's weigher charges in full when that
    /// is lower, on both sides of the crossover; an entry at that bound is
    /// charged in full, never saturated.
    #[test]
    fn held_entry_bound_is_the_resident_budget_capped_at_u32_max_units() {
        let most_charged = u64::from(u32::MAX) * SLOT_UNIT_BYTES;
        for (resident, want) in [
            (0, 0),
            (484_000, 484_000),
            (most_charged - 1, most_charged - 1),
            (most_charged, most_charged),
            (most_charged + 1, most_charged),
            (u64::MAX, most_charged),
        ] {
            let bound = held_entry_bound(resident);
            assert_eq!(bound, want, "resident budget {resident}");
            assert_eq!(
                u64::from(slot_weight(bound)),
                bound.div_ceil(SLOT_UNIT_BYTES),
                "resident budget {resident}: an entry at the bound is charged in full"
            );
        }
    }

    // ── T10(b): a fresh entry within the cap is within the held-entry bound ─
    //
    // Every weight below is the weigher's arithmetic (`domain::alloc_model`):
    // bucket counts from `table_buckets` / `bucket_capacity` and size classes
    // from `allocation_bytes`, the same at hashbrown group widths 8 and 16.

    /// The size classes' rounding ceiling above 128 B, `(numerator,
    /// denominator)`: jemalloc's four classes per doubling round a request up
    /// by less than a quarter ([`allocation_bytes`]), checked by
    /// [`fresh_index_ratio_ceiling_is_derived_from_the_weighers_slot_sizes`].
    const ROUNDING: (u128, u128) = (5, 4);

    /// The hash tables' densest load, `(slots, buckets)`: 14 usable slots in
    /// 16 buckets, 7/8 ([`bucket_capacity`]).
    fn densest_load() -> (u128, u128) {
        (u128::from(bucket_capacity(16)), 16)
    }

    /// The bytes one bucket of `table` costs: its slot and its control byte.
    fn bucket_bytes(table: RebuildTable) -> u128 {
        u128::from(table_slot_bytes(table)) + 1
    }

    /// f_max: the most a freshly built entry's plan indexes weigh per byte
    /// of its state, beyond [`fresh_index_fixed_bytes`], by the weigher's
    /// per-row arithmetic — an integer rational `(numerator, denominator)`,
    /// 185/106 ≈ 1.745 with today's slot sizes. Fresh means as
    /// `PlanIndexes::build` lays them out (a cold start); a name vector that
    /// has since kept a larger capacity escapes it.
    ///
    /// The larger of two rows' ratios, each index cost against the state the
    /// same row holds at least:
    ///
    /// - **an external value row** (a field or a time sample): at worst one
    ///   bucket of the external-hash counts at their sparsest, a table of `b`
    ///   buckets holding more than half of `b`'s 7/8 load, so under 16/7
    ///   buckets per hash, rounded up by a quarter; against its value table's
    ///   bucket at its densest, 8/7 per row, of the lighter of the two
    ///   tables' slots, and its boxed hash. Unrounded, 74/53 ≈ 1.40: the
    ///   resident-bound proptest's "about 1.4 per external row";
    /// - **a section row**: its parent's key in the name index at its
    ///   sparsest and rounded the same, and its vector entry at twice the
    ///   entry (a vector built by pushes from one entry holds at most twice
    ///   its length); against its section-table bucket at its densest.
    fn fresh_index_ratio_ceiling() -> (u128, u128) {
        let (external, section) = per_row_index_ratios(ROUNDING);
        if external.0 * section.1 >= section.0 * external.1 {
            external
        } else {
            section
        }
    }

    /// [`fresh_index_ratio_ceiling`]'s two rows' ratios, `(external value
    /// row, section row)`, each `(numerator, denominator)`, with the index
    /// tables' allocations rounded up by `rounding` (`(1, 1)`: unrounded).
    fn per_row_index_ratios(rounding: (u128, u128)) -> ((u128, u128), (u128, u128)) {
        let (slots, buckets) = densest_load();
        let (up, down) = rounding;
        let hash = u128::from(allocation_bytes(CompactFields::HASH_BYTES as u64));
        let value_row =
            (CompactFields::ROW_SLOT_BYTES.min(CompactSamples::ROW_SLOT_BYTES) as u128) + 1;
        let external = (
            up * 2 * buckets * (PlanIndexes::EXT_SLOT_BYTES as u128 + 1),
            down * (buckets * value_row + hash * slots),
        );
        let entry = PlanIndexes::NAME_ENTRY_BYTES as u128;
        let section = (
            up * 2 * buckets * (PlanIndexes::NAME_SLOT_BYTES as u128 + 1)
                + 2 * entry * down * slots,
            down * buckets * bucket_bytes(RebuildTable::Sections),
        );
        (external, section)
    }

    /// `(numerator, denominator)` in lowest terms.
    fn reduced((numerator, denominator): (u128, u128)) -> (u128, u128) {
        let (mut a, mut b) = (numerator, denominator);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        (numerator / a, denominator / a)
    }

    /// C_fixed: the bytes of a freshly built entry's plan indexes the per-row
    /// ratio does not cover — the `PlanIndexes` struct, and for each of its
    /// two maps the most a table of 8 buckets or fewer allocates (whose load
    /// can be sparser than the ratio's) plus the rounding of a larger
    /// table's trailing control bytes. About a kilobyte.
    fn fresh_index_fixed_bytes() -> u64 {
        let (up, down) = ROUNDING;
        let trailing = u64::try_from(
            u128::from(crate::domain::alloc_model::TABLE_TRAILING_CONTROL_BYTES) * up / down,
        )
        .expect("a few bytes");
        let small = bucket_capacity(8) as usize;
        [RebuildTable::NameIndex, RebuildTable::ExtCounts]
            .into_iter()
            .map(|table| table_bytes(small, table_slot_bytes(table)) + trailing)
            .fold(size_of::<PlanIndexes>() as u64, u64::saturating_add)
    }

    /// Whether indexes of `index_bytes` beside a state of `state_bytes` are
    /// within the fresh-index bound: f_max × the state + C_fixed.
    fn within_fresh_index_bound(state_bytes: u64, index_bytes: u64) -> bool {
        let (numerator, denominator) = fresh_index_ratio_ceiling();
        u128::from(index_bytes) * denominator
            <= numerator * u128::from(state_bytes)
                + u128::from(fresh_index_fixed_bytes()) * denominator
    }

    /// T10(b), the derivation: the ratio's premises hold of the weigher —
    /// its named slot sizes are the ones it weighs tables by; a table is
    /// never denser than 7/8 and a minimal one of 16 buckets or more never
    /// sparser than 16/7 buckets per row; a request above 128 B is rounded
    /// up by less than a quarter, and every index table's smallest
    /// allocation is above 128 B, while a name vector, at a power-of-two
    /// capacity of at most twice its length as a build lays it out, is
    /// allocated exactly — and the ratio it gives is at most 1.75, the
    /// figure the module docs state. Both figures the docs quote are pinned:
    /// f_max is exactly 185/106 ≈ 1.745, and the same row's ratio before
    /// rounding exactly 74/53 ≈ 1.40, so a slot-size change that moves
    /// either fails here rather than leaving those docs stale.
    #[test]
    fn fresh_index_ratio_ceiling_is_derived_from_the_weighers_slot_sizes() {
        for (table, named) in [
            (RebuildTable::FieldRows, CompactFields::ROW_SLOT_BYTES),
            (RebuildTable::SampleRows, CompactSamples::ROW_SLOT_BYTES),
            (RebuildTable::NameIndex, PlanIndexes::NAME_SLOT_BYTES),
            (RebuildTable::ExtCounts, PlanIndexes::EXT_SLOT_BYTES),
        ] {
            assert_eq!(table_slot_bytes(table), named as u64, "{table:?}");
        }

        // The load: never denser than 7/8, and a minimal table of `b ≥ 16`
        // buckets holds more than `bucket_capacity(b / 2)` rows, so at most
        // 16/7 buckets per row. Small tables are at least 8/7 too.
        let (slots, buckets) = densest_load();
        assert_eq!((slots, buckets), (14, 16), "7/8");
        for rows in 1..=7u64 {
            assert!(u128::from(table_buckets(rows)) * slots >= buckets * u128::from(rows));
        }
        for shift in 2..=40 {
            let b = 1u64 << shift;
            assert!(u128::from(bucket_capacity(b)) * buckets <= slots * u128::from(b));
            if b >= 16 {
                let fewest = bucket_capacity(b / 2) + 1;
                assert_eq!(table_buckets(fewest), b, "{fewest} rows take {b} buckets");
                assert!(u128::from(b) * slots <= 2 * buckets * u128::from(fewest));
            }
        }

        // The rounding: by less than a quarter above 128 B, nearly a quarter
        // just past a power of two.
        let (up, down) = ROUNDING;
        let requests = (129..=1u64 << 16)
            .chain((16..=40).flat_map(|shift| [(1u64 << shift) + 1, (1u64 << shift) + 7]));
        for n in requests {
            assert!(
                u128::from(allocation_bytes(n)) * down <= up * u128::from(n),
                "{n} B rounds to {} B",
                allocation_bytes(n)
            );
        }
        assert!(allocation_bytes((1 << 20) + 1) * 100 > ((1 << 20) + 1) * 124);
        for table in [RebuildTable::NameIndex, RebuildTable::ExtCounts] {
            assert!(
                table_bytes(1, table_slot_bytes(table)) > 128,
                "{table:?}'s smallest table is rounded by at most a quarter"
            );
        }
        let entry = PlanIndexes::NAME_ENTRY_BYTES as u64;
        for shift in 0..=40 {
            assert_eq!(allocation_bytes(entry << shift), entry << shift);
        }
        for children in [1u64, 2, 3, 4, 5, 8, 9, 100, 1_000, 5_000] {
            let mut state = parents_with_one_child(1);
            for n in 1..children {
                let id = created_ids(1) + n;
                state.sections.insert(
                    id,
                    Section {
                        section_id: id,
                        parent_id: 1,
                        name: format!("n{n}"),
                        section_type: 6,
                        section_order: (n + 1) as u32,
                        first_version: 1,
                        last_version: 1,
                    },
                );
                state
                    .children
                    .get_mut(&1)
                    .expect("the parent's list")
                    .push(id);
            }
            let indexes = PlanIndexes::build(&compact(&state));
            let (len, capacity) = indexes.name_vector_room(1);
            assert_eq!(len as u64, children);
            assert!(
                capacity.is_power_of_two() && capacity <= 2 * len,
                "{children} children: a built vector of capacity {capacity}"
            );
        }

        let (numerator, denominator) = fresh_index_ratio_ceiling();
        assert!(
            numerator * 4 <= 7 * denominator,
            "f_max = {numerator}/{denominator} must be at most 1.75"
        );
        // The two figures the docs quote: 1.745 sound, 1.40 unrounded.
        assert_eq!(
            reduced((numerator, denominator)),
            (185, 106),
            "f_max is 185/106 ≈ 1.745, the external value row's"
        );
        let (unrounded, _) = per_row_index_ratios((1, 1));
        assert_eq!(
            reduced(unrounded),
            (74, 53),
            "the external value row's ratio before rounding is 74/53 ≈ 1.40"
        );
    }

    /// T10(b): an entry at the per-entry cap whose plan indexes are fresh —
    /// f_max × its state + C_fixed at most — weighs less than
    /// `held_entry_bound`, at every budget whose resident budget fits a
    /// `u64`: so the store never evicts the cold start of an entry it evicted
    /// for its weight, and no layer loops between eviction and reload. The
    /// budgets: 1 MiB, the least; the 256 MiB of the reference pod; either
    /// side of 16 GiB, where the cap reaches `u32::MAX`; 64 GiB, where the
    /// bound is mini-moka's `u32::MAX` units; and the largest that fits.
    #[test]
    fn fresh_entry_at_the_cap_is_within_the_held_entry_bound_at_every_accepted_budget() {
        const MIB: u64 = 1024 * 1024;
        const GIB: u64 = 1024 * MIB;
        // Just above u64::MAX / (1 + φ) in MiB, then down to the first whose
        // resident budget fits.
        let mut largest = u64::try_from(
            u128::from(u64::MAX) * 100
                / u128::from(100 + PLAN_INDEX_BUDGET_CENTI)
                / u128::from(MIB),
        )
        .expect("fits")
            + 2;
        while resident_budget_bytes(largest * MIB).is_none() {
            largest -= 1;
        }
        let (numerator, denominator) = fresh_index_ratio_ceiling();
        let fixed = fresh_index_fixed_bytes();
        for budget in [
            MIB,
            256 * MIB,
            16 * GIB - MIB,
            16 * GIB,
            64 * GIB,
            largest * MIB,
        ] {
            let resident = resident_budget_bytes(budget).expect("an accepted budget");
            let bound = held_entry_bound(resident);
            assert_eq!(
                bound,
                resident.min(u64::from(u32::MAX) * SLOT_UNIT_BYTES),
                "budget {budget}"
            );
            let cap = u128::from(max_entry_bytes_for(budget));
            let heaviest = cap * (denominator + numerator) + u128::from(fixed) * denominator;
            assert!(
                heaviest < u128::from(bound) * denominator,
                "budget {budget}: a fresh entry at the {cap} B cap weighs up to {} B, not below \
                 the {bound} B held-entry bound",
                heaviest / denominator
            );
        }
    }

    /// T10(b)'s anchor: a real entry at the per-row extreme — 896 external
    /// fields, `bucket_capacity(1024)`, fill their table's 1,024 buckets to
    /// its densest, and with one external time sample their 897 distinct
    /// hashes take the external-hash counts to 2,048 buckets, their sparsest
    /// — is within the fresh-index bound as built. 128 sections of 7 fields
    /// each, so the field names are interned once.
    #[test]
    fn fresh_entry_at_the_per_row_extreme_is_within_the_fresh_index_bound() {
        let rows = bucket_capacity(1024);
        let mut state = external_heavy_state(rows / 7, 7);
        let time = OrderedFloat(0.0);
        state.time_samples.insert(
            (1, time),
            TimeSample {
                section_id: 1,
                time,
                value: FieldValue::External {
                    hash: [0xAB; 32],
                    size: 64,
                    ext_value_index: 0,
                },
                version: 1,
            },
        );
        let entry = weighed(OBJECT_ID, 1, state);
        let indexes = entry
            .plan
            .as_ref()
            .expect("an entry holds its plan indexes");
        assert_eq!(FieldRows::len(&entry.state.fields) as u64, rows);
        assert_eq!(
            table_buckets(entry.state.fields.capacity() as u64),
            1024,
            "fixture: the field table at its densest"
        );
        let (hashes, capacity) = indexes
            .own_table_shape(RebuildTable::ExtCounts)
            .expect("the external-hash counts");
        assert_eq!(hashes as u64, rows + 1);
        assert_eq!(
            table_buckets(capacity as u64),
            2048,
            "fixture: the external-hash counts at their sparsest"
        );
        assert!(
            within_fresh_index_bound(entry.state_weight_bytes, entry.index_weight_bytes),
            "indexes of {} B beside a {} B state",
            entry.index_weight_bytes,
            entry.state_weight_bytes
        );
    }

    // ── Checkout ────────────────────────────────────────────────────────────

    #[test]
    fn checkout_takes_the_entry_out_until_it_settles() {
        let cache = cache_with(BUDGET_BYTES);
        let entry = cut(OBJECT_ID, 5, state_with_sections(3));
        assert!(cold_start(&cache, entry.clone()));

        let guard = cache
            .checkout(OBJECT_ID)
            .expect("a cached object checks out");
        assert_same_cut(guard.entry(), &entry, "checkout");

        assert!(
            cache.checkout(OBJECT_ID).is_none(),
            "a second checkout must find the entry gone"
        );
        assert!(
            matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss),
            "a lookup must find the entry gone"
        );
        drop(guard);
        assert_same_cut(hit(&cache, OBJECT_ID).entry(), &entry, "after it settles");
    }

    /// `checkout` is the raw take: it gives no uncacheable verdict. An object
    /// can hold an entry and a live negative entry at once — a guard taken
    /// before `record_uncacheable` puts back unrefused, because recording a
    /// verdict is not an invalidation.
    #[test]
    fn checkout_ignores_a_live_negative_entry_that_lookup_answers() {
        let cache = cache_with(BUDGET_BYTES);
        let entry = cut(OBJECT_ID, 5, state_with_sections(3));
        assert!(cold_start(&cache, entry.clone()));
        let guard = hit(&cache, OBJECT_ID);
        cache.record_uncacheable(OBJECT_ID, UncacheableReason::NoChsComplete);
        assert!(
            guard.put_back(entry.clone()).is_stored(),
            "fixture: stored beside the verdict"
        );

        assert!(matches!(
            cache.lookup(OBJECT_ID),
            CacheLookup::Uncacheable {
                reason: UncacheableReason::NoChsComplete
            }
        ));
        let taken = cache
            .checkout(OBJECT_ID)
            .expect("checkout must take the entry despite the negative entry");
        assert_same_cut(taken.entry(), &entry, "checkout beside a verdict");
    }

    // ── Pruning the epoch bookkeeping ───────────────────────────────────────

    /// Whether the cache holds an invalidation record for `object_id`, and how
    /// many checkouts it counts as outstanding.
    fn bookkeeping(cache: &StructuralCache, object_id: u64) -> (bool, usize) {
        let control = cache.inner.lock();
        (
            control.invalidations.contains_key(&object_id),
            control.outstanding.values().sum(),
        )
    }

    #[test]
    fn invalidation_record_kept_while_an_older_checkout_is_outstanding_then_pruned() {
        let cache = cache_with(BUDGET_BYTES);
        assert!(cold_start(
            &cache,
            cut(OBJECT_ID, 1, state_with_sections(1))
        ));
        assert!(cold_start(
            &cache,
            cut(OTHER_OBJECT_ID, 1, state_with_sections(1))
        ));
        let older = hit(&cache, OBJECT_ID);

        cache.invalidate(OTHER_OBJECT_ID);
        assert_eq!(
            bookkeeping(&cache, OTHER_OBJECT_ID),
            (true, 1),
            "the record must be kept while a checkout taken before it is outstanding"
        );

        drop(older);
        assert_eq!(
            bookkeeping(&cache, OTHER_OBJECT_ID),
            (false, 0),
            "once that checkout settles, no checkout can be refused by the record"
        );
    }

    #[test]
    fn epoch_bookkeeping_is_empty_once_every_checkout_and_invalidation_settles() {
        let cache = cache_with(BUDGET_BYTES);
        for object_id in [OBJECT_ID, OTHER_OBJECT_ID, 44, 45] {
            assert!(cold_start(
                &cache,
                cut(object_id, 1, state_with_sections(1))
            ));
        }
        let put_back = hit(&cache, OBJECT_ID);
        let discarded = hit(&cache, OTHER_OBJECT_ID);
        let dropped = cache.checkout(44).expect("cached");
        let refused = hit(&cache, 45);
        let reload = cache
            .admit(
                cache.begin_cold_start(),
                cut(OBJECT_ID, 9, state_with_sections(2)),
            )
            .expect("admissible");
        cache.invalidate(45);
        cache.invalidate(46);

        assert!(put_back
            .put_back(cut(OBJECT_ID, 2, state_with_sections(1)))
            .is_stored());
        discarded.discard();
        drop(dropped);
        assert_eq!(
            refused.put_back(cut(45, 2, state_with_sections(1))),
            PutBack::Invalidated
        );
        drop(reload);

        let control = cache.inner.lock();
        assert!(
            control.outstanding.is_empty(),
            "every checkout settled: {:?}",
            control.outstanding
        );
        assert!(
            control.invalidations.is_empty(),
            "no checkout is outstanding, so no record can matter: {:?}",
            control.invalidations
        );
    }

    // ── The side store's entry bound ────────────────────────────────────────

    /// With entries far lighter than the side store's weight bound, its entry
    /// count binds: past [`SIDE_STORE_MAX_ENTRIES`] refused admissions, each
    /// new one evicts the least recently used.
    ///
    /// The oldest entry is used (a hit, then settled) before the overflow, so
    /// recency differs from id order: it must survive while the next-oldest
    /// go. Eviction by smallest id would evict it. (Using a side-store entry
    /// takes it out and puts it back, so its recency and its re-insertion
    /// order coincide by construction.)
    #[test]
    fn side_store_past_its_entry_bound_evicts_the_least_recently_used() {
        const OVERFLOW: u64 = 6;
        const FIRST: u64 = 1_000;
        const FULL: u64 = SIDE_STORE_MAX_ENTRIES as u64;
        // Residents of 1,000 sections make the side store's weight bound
        // (about one resident) hundreds of tiny entries wide.
        let cache = full_cache_of(1_000, LONG_TTL, LONG_TTL);
        let tiny_weight = whole_weight(0);
        assert!(
            tiny_weight * (FULL + OVERFLOW) < cache.inner.side_weight_bound,
            "fixture: the weight bound must not bind"
        );
        // The residents' plan indexes weigh less than φ of their state, so
        // mini-moka keeps room beside them: fill it with tiny ballast entries
        // until it refuses one, and drop that one from the side store.
        let full = (100_000..110_000).any(|ballast| {
            assert!(cold_start(&cache, cut(ballast, 1, state_with_sections(0))));
            let refused = !resident_in_main(&cache, ballast);
            if refused {
                cache.invalidate(ballast);
            }
            refused
        });
        assert!(
            full && cache.inner.lock().side.is_empty(),
            "fixture: mini-moka is full and the side store empty"
        );
        let refuse = |object_id: u64| {
            assert!(cold_start(
                &cache,
                cut(object_id, 1, state_with_sections(0))
            ));
            assert!(
                !resident_in_main(&cache, object_id),
                "fixture: mini-moka must refuse tiny entry {object_id}"
            );
        };

        for object_id in FIRST..FIRST + FULL {
            refuse(object_id);
        }
        // Use the oldest: it becomes the most recently used.
        drop(hit(&cache, FIRST));
        assert!(
            in_side_store(&cache, FIRST) && !resident_in_main(&cache, FIRST),
            "fixture: the used entry must settle back into the side store"
        );
        for object_id in FIRST + FULL..FIRST + FULL + OVERFLOW {
            refuse(object_id);
        }

        assert_eq!(cache.inner.lock().side.len(), SIDE_STORE_MAX_ENTRIES);
        assert!(
            in_side_store(&cache, FIRST),
            "the oldest-inserted entry was used most recently of the early ones: it must survive"
        );
        for object_id in FIRST + 1..=FIRST + OVERFLOW {
            assert!(
                !in_side_store(&cache, object_id),
                "the {OVERFLOW} least recently used must have been evicted; {object_id} remains"
            );
        }
        for object_id in FIRST + OVERFLOW + 1..FIRST + FULL + OVERFLOW {
            assert!(
                in_side_store(&cache, object_id),
                "the more recently used must remain; {object_id} is gone"
            );
        }
    }

    // ── Entries freed outside the control lock ──────────────────────────────

    /// The critical sections that remove an entry hand it out in `freed`
    /// instead of dropping it under the control lock: an invalidation and a
    /// negative entry (a resident entry is taken out of its slot before the
    /// key goes, so mini-moka is left an empty slot), and a side-store
    /// expiry, both the sweep and a lookup that finds its entry expired.
    #[test]
    fn removed_entries_are_handed_out_to_be_freed_outside_the_lock() {
        let cache = cache_with(BUDGET_BYTES);
        let resident = cut(OBJECT_ID, 5, state_with_sections(3));

        // Invalidation of a resident entry.
        assert!(cold_start(&cache, resident.clone()));
        assert!(resident_in_main(&cache, OBJECT_ID), "fixture: resident");
        let slot = {
            let control = cache.inner.lock();
            control
                .entries
                .as_ref()
                .and_then(|entries| entries.get(&OBJECT_ID))
                .expect("fixture: resident slot")
        };
        let mut freed = Freed::new();
        cache.inner.lock().invalidate(OBJECT_ID, &mut freed);
        assert_eq!(freed.len(), 1, "the invalidated entry is handed out");
        assert_same_cut(&freed[0], &resident, "invalidate");
        assert!(
            slot.entry
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_none(),
            "the slot mini-moka drops is empty"
        );

        // A negative entry over a resident entry.
        let other = cut(OTHER_OBJECT_ID, 2, state_with_sections(2));
        assert!(cold_start(&cache, other.clone()));
        let mut freed = Freed::new();
        cache.inner.lock().record_negative(
            OTHER_OBJECT_ID,
            UncacheableReason::NoChsComplete,
            LONG_TTL,
            &mut freed,
        );
        assert_eq!(
            freed.len(),
            1,
            "the negative entry's object entry is handed out"
        );
        assert_same_cut(&freed[0], &other, "record_negative");

        // Side-store expiry: the sweep, and a lookup of an expired entry.
        let idle_ttl = Duration::from_millis(20);
        let cache = StructuralCache::with_ttls(BUDGET_BYTES, idle_ttl, LONG_TTL);
        {
            let mut control = cache.inner.lock();
            control.side.insert(OBJECT_ID, resident.clone());
            control.side.insert(OTHER_OBJECT_ID, other.clone());
        }
        std::thread::sleep(idle_ttl * 3);
        let mut freed = Freed::new();
        assert!(
            cache
                .inner
                .lock()
                .take(OBJECT_ID, idle_ttl, &mut freed)
                .is_none(),
            "an expired side-store entry is not taken"
        );
        assert_eq!(
            freed.len(),
            1,
            "the expired entry a lookup found is handed out"
        );
        assert_same_cut(&freed[0], &resident, "take of an expired entry");
        let mut freed = Freed::new();
        cache.inner.lock().expire_side(idle_ttl, &mut freed);
        assert_eq!(
            freed.len(),
            1,
            "the expired entry the sweep found is handed out"
        );
        assert_same_cut(&freed[0], &other, "expire_side");
        assert!(cache.inner.lock().side.is_empty());
    }

    /// The cold start's pre-read weight check (SR-3.3, P7) never refuses a
    /// cacheable object on a well-formed object whose indexes do not change
    /// between the probe and the read: on states of every table size — below
    /// the 8-slot
    /// small-table rounding, at it, and at thousands of sections — the lower
    /// bound from the index cardinalities is at most the weight the cache
    /// charges the compacted compact entry, and it grows with the
    /// cardinalities. The fixture's rows share one field name and hold no
    /// external value, so the bound's exclusions (names, hashes) are at their
    /// smallest and the estimate is at its closest to it.
    #[test]
    fn state_weight_lower_bound_never_exceeds_the_charged_weight() {
        let mut previous = 0;
        for sections in [0u64, 1, 3, 7, 8, 9, 100, 1_000, 5_000] {
            let mut state = compact(&state_with_sections(sections));
            compact_state(&mut state);
            let charged = estimate_state_bytes(&state);
            let cards = SnapshotCardinalities {
                sections,
                tombstones: 4,
                fields: sections,
                time_samples: 0,
                children_index: Some(u64::from(sections > 0)),
            };
            let lower = state_weight_lower_bound(&cards);
            assert!(
                lower <= charged,
                "{sections} sections: lower bound {lower} above the charged weight {charged}"
            );
            assert!(
                lower >= previous,
                "{sections} sections: the bound fell from {previous} to {lower}"
            );
            previous = lower;
        }
        assert!(
            previous > (size_of::<CompactState>() as u64),
            "at 5,000 sections the bound counts the tables, not only the state itself"
        );
    }

    /// `state` with `per_section` time samples on each of its sections, each
    /// a small inline value.
    fn with_time_samples(mut state: ObjectState, per_section: u64) -> ObjectState {
        let ids: Vec<u64> = state.sections.keys().copied().collect();
        for id in ids {
            for i in 0..per_section {
                let time = OrderedFloat(i as f64);
                state.time_samples.insert(
                    (id, time),
                    TimeSample {
                        section_id: id,
                        time,
                        value: FieldValue::Inline(vec![3; 8]),
                        version: 1,
                    },
                );
            }
        }
        state
    }

    /// The time-samples term of the weight check: on states that hold time
    /// samples — few enough for the small-table rounding, and thousands —
    /// the bound counting them is at most the charged weight, and it counts
    /// them: it is larger than the same state's bound without them by at least
    /// the samples' table, so a bound that dropped the term would fall short
    /// of the charged weight's growth.
    #[test]
    fn state_weight_lower_bound_counts_time_samples_and_never_exceeds_the_charged_weight() {
        for (sections, per_section) in [(1u64, 1u64), (2, 3), (4, 2), (100, 4), (1_000, 5)] {
            let mut state = compact(&with_time_samples(
                state_with_sections(sections),
                per_section,
            ));
            compact_state(&mut state);
            let samples = sections * per_section;
            assert_eq!(
                SampleRows::len(&state.time_samples) as u64,
                samples,
                "fixture: the samples"
            );
            let charged = estimate_state_bytes(&state);
            let cards = |time_samples| SnapshotCardinalities {
                sections,
                tombstones: 0,
                fields: sections,
                time_samples,
                children_index: Some(1),
            };
            let lower = state_weight_lower_bound(&cards(samples));
            let without = state_weight_lower_bound(&cards(0));
            assert!(
                lower <= charged,
                "{sections} sections × {per_section} samples: lower bound {lower} above the \
                 charged weight {charged}"
            );
            assert_eq!(
                lower - without,
                table_bytes(samples as usize, COMPACT_TIME_SAMPLE_SLOT_BYTES),
                "{sections} sections × {per_section} samples: the bound adds exactly the \
                 samples' table"
            );
            assert!(lower > without, "the samples are counted");
        }
    }

    // ── The plan in place (b.ug2) ────────────────────────────────────────────

    /// One `CreateSection` of `section_id` under `parent_id`.
    fn create_delta(parent_id: u64, section_id: u64) -> OwnedDelta {
        OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id,
                section_id,
                section_name: format!("new{section_id}"),
                section_type: 6,
            })],
            ..OwnedDelta::default()
        }
    }

    /// Plan `delta` in place on `guard`'s held entry, unbounded.
    fn plan_unbounded(
        guard: &mut CheckedOut,
        delta: OwnedDelta,
    ) -> Result<BoundedPlan, DeltaError> {
        guard.plan_in_place(
            Some(delta),
            MutationAllowance::UNBOUNDED,
            &mut |_: RewriteCount| None::<u64>,
            &mut ApplyDeltaStats::default(),
        )
    }

    /// How an attempt that planned in place ends, and the entry its settle
    /// must leave cached.
    #[derive(Debug, Clone, Copy)]
    enum PlanExit {
        /// `advance_committed`, then `put_back_held`: the planned successor.
        Committed,
        /// `put_back_held` with the plan pending: the plan is rolled back and
        /// the pre-plan cut goes back.
        RolledBack,
        /// `discard`: nothing cached.
        Discarded,
        /// The guard dropped with the plan pending (a deadline or a
        /// cancellation mid-commit): rolled back, then put back.
        Dropped,
    }

    /// T8: the four settle primitives a plan made in place within its
    /// allowance can end in — the exit table of `CheckedOut::plan_in_place`
    /// maps every attempt outcome onto one of them. Which outcome reaches
    /// which primitive is the dispatch's, and is driven through it in
    /// `tests/integration/storage/structural_dispatch_test.rs`: the commit by
    /// every committing test; the put-back's rollback by
    /// `structural_stale_on_a_cold_started_attempt_ticks_on_structural_full_and_the_retry_repairs_one_entry`
    /// (`STALE`), `structural_fenced_reply_waits_uncounted_puts_the_entry_back_and_commits_when_it_ends`
    /// (`FENCED`), the `structural_put_back_*` tests (a delta `Err`, a
    /// transport error, an in-doubt reply, a stream-ID collision) and the
    /// `structural_commit_*_puts_the_entry_back` tests (`ALREADY_LANDED`,
    /// `Recorded`, `OVERTAKEN`) — and outside that file, the size gate's by
    /// `structural_size_gate_test.rs`'s
    /// `structural_edit_planned_one_unit_over_the_work_limit_commits_through_watch`
    /// and a refused charge's by `commit_charge_test.rs`'s T5a–T5h; the
    /// discard by
    /// `structural_cache_diverged_writes_nothing_drops_the_entry_and_the_retry_reloads_and_commits`
    /// and `structural_commit_refused_not_found_or_superseded_drops_the_entry`;
    /// the drop by
    /// `structural_call_dropped_after_its_commit_landed_rolls_back_and_the_next_call_repairs_ug2`.
    const PLAN_EXITS: [(&str, PlanExit); 4] = [
        ("commit", PlanExit::Committed),
        ("put-back rollback", PlanExit::RolledBack),
        ("discard", PlanExit::Discarded),
        ("drop", PlanExit::Dropped),
    ];

    #[test]
    fn plan_in_place_settles_each_exit_to_its_documented_entry() {
        const PLANNED: u64 = 100;
        for (outcome, exit) in PLAN_EXITS {
            let cache = cache_with(BUDGET_BYTES);
            let original = cut(OBJECT_ID, 5, state_with_sections(3));
            assert!(cold_start(&cache, original.clone()));
            let (plans, rollbacks) = (cache.in_place_plans_for_test(), cache.rollbacks_for_test());
            let mut guard = hit(&cache, OBJECT_ID);
            let mutations = match plan_unbounded(&mut guard, create_delta(0, PLANNED)) {
                Ok(BoundedPlan::Within { mutations, .. }) => mutations,
                other => panic!("{outcome}: the plan applies: {other:?}"),
            };
            assert!(
                guard.entry().state.sections.contains_key(&PLANNED),
                "{outcome}: the plan is in place on the held entry"
            );
            let expected = match exit {
                PlanExit::Committed => {
                    let mut planned = original.state.clone();
                    apply_delta(&mut planned, Some(create_delta(0, PLANNED)))
                        .expect("the copy's plan applies");
                    let successor = crate::storage::structural_cache_repair::successor_entry(
                        &original,
                        planned,
                        &mutations,
                        6,
                        format!("{:040x}", 6),
                    )
                    .expect("no dangling parent is adopted");
                    guard
                        .advance_committed(&mutations, 6, format!("{:040x}", 6))
                        .expect("the commit settles");
                    assert!(guard.put_back_held().is_stored(), "{outcome}: stored");
                    Some(successor)
                }
                PlanExit::RolledBack => {
                    assert!(guard.put_back_held().is_stored(), "{outcome}: stored");
                    Some(original.clone())
                }
                PlanExit::Discarded => {
                    guard.discard();
                    None
                }
                PlanExit::Dropped => {
                    drop(guard);
                    Some(original.clone())
                }
            };
            assert_eq!(
                cache.in_place_plans_for_test() - plans,
                1,
                "{outcome}: one plan in place"
            );
            let rolled_back = matches!(exit, PlanExit::RolledBack | PlanExit::Dropped);
            assert_eq!(
                cache.rollbacks_for_test() - rollbacks,
                u64::from(rolled_back),
                "{outcome}: rolled back iff the exit keeps the object cached without its commit"
            );
            match (cache.lookup(OBJECT_ID), &expected) {
                (CacheLookup::Hit(found), Some(want)) => {
                    assert_same_cut(found.entry(), want, outcome);
                    assert_eq!(
                        found.entry().state_weight_bytes,
                        estimate_state_bytes(&found.entry().state),
                        "{outcome}: weighed exactly"
                    );
                    assert_indexes_rebuilt(found.entry());
                }
                (CacheLookup::Miss, None) => {}
                (other, want) => panic!(
                    "{outcome} ({exit:?}) must leave {}; got {other:?}",
                    want.as_ref().map_or("no entry".to_owned(), |e| format!(
                        "the entry at seq {}",
                        e.seq
                    ))
                ),
            }
        }
    }

    /// T8: a plan that fails — a delta `Err`, or a bounded plan whose budget
    /// refuses (`Overflow`, the call's `NeedsCharge`) — leaves a
    /// half-planned entry that `put_back_held` rolls back before it goes
    /// back: the pre-plan cut is cached.
    #[test]
    fn failed_plan_in_place_is_rolled_back_before_its_put_back() {
        let cache = cache_with(BUDGET_BYTES);
        let original = cut(OBJECT_ID, 5, state_with_sections(3));
        assert!(cold_start(&cache, original.clone()));

        // A delta `Err` after a first command applied: the second creates
        // an id the first just created.
        let mut guard = hit(&cache, OBJECT_ID);
        let mut failing = create_delta(0, 100);
        failing
            .struct_commands
            .extend(create_delta(0, 100).struct_commands);
        assert!(
            plan_unbounded(&mut guard, failing).is_err(),
            "fixture: the delta is refused part way"
        );
        assert!(guard.put_back_held().is_stored());
        assert_same_cut(
            hit(&cache, OBJECT_ID).entry(),
            &original,
            "after a delta Err",
        );

        // `Overflow`: every count priced from an allowance of nothing, and a
        // budget that refuses.
        let mut guard = hit(&cache, OBJECT_ID);
        let priced = MutationAllowance {
            bytes: 0,
            row_bytes: 1,
            list_entry_bytes: 1,
            index_entry_bytes: 1,
            adoption_entry_bytes: 1,
            walk_entry_bytes: 1,
        };
        let plan = guard.plan_in_place(
            Some(create_delta(0, 100)),
            priced,
            &mut |_: RewriteCount| None::<u64>,
            &mut ApplyDeltaStats::default(),
        );
        assert!(
            matches!(plan, Ok(BoundedPlan::Overflow { .. })),
            "fixture: the plan overflows: {plan:?}"
        );
        assert!(guard.put_back_held().is_stored());
        assert_same_cut(
            hit(&cache, OBJECT_ID).entry(),
            &original,
            "after an overflow",
        );
    }

    /// T8: a commit whose plan created one of the entry's dangling parents is
    /// refused by `advance_committed` (`DanglingAdopted`) before anything
    /// settles; the caller discards, and nothing is cached.
    #[test]
    fn committed_plan_adopting_a_dangling_parent_is_refused_and_discarded() {
        const DANGLING: u64 = 77;
        let cache = cache_with(BUDGET_BYTES);
        let mut state = compact(&state_with_sections(3));
        compact_state(&mut state);
        let original = CacheEntry::new(
            OBJECT_ID,
            format!("{:040x}", 5),
            5,
            HashSet::from([DANGLING]),
            state,
        );
        assert!(cold_start(&cache, original));
        let mut guard = hit(&cache, OBJECT_ID);
        let mutations = match plan_unbounded(&mut guard, create_delta(0, DANGLING)) {
            Ok(BoundedPlan::Within { mutations, .. }) => mutations,
            other => panic!("the plan applies: {other:?}"),
        };
        assert_eq!(
            guard.advance_committed(&mutations, 6, format!("{:040x}", 6)),
            Err(ReloadReason::DanglingAdopted)
        );
        guard.discard();
        assert!(matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss));
    }

    /// T8: the epoch check still refuses the put-back of a plan made in place
    /// when the object was invalidated meanwhile, explicit or by `Drop`; and a
    /// panic unwinding through a guard with a plan pending discards the entry
    /// and invalidates the object, never putting back a half-planned entry.
    #[test]
    fn plan_in_place_put_back_is_refused_after_an_invalidation_and_discarded_on_a_panic() {
        for explicit in [true, false] {
            let cache = cache_with(BUDGET_BYTES);
            assert!(cold_start(
                &cache,
                cut(OBJECT_ID, 5, state_with_sections(3))
            ));
            let mut guard = hit(&cache, OBJECT_ID);
            assert!(plan_unbounded(&mut guard, create_delta(0, 100)).is_ok());
            cache.invalidate(OBJECT_ID);
            if explicit {
                assert_eq!(guard.put_back_held(), PutBack::Invalidated);
            } else {
                drop(guard);
            }
            assert!(
                matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss),
                "explicit {explicit}: refused by the epoch check"
            );
        }

        let cache = cache_with(BUDGET_BYTES);
        let original = cut(OBJECT_ID, 5, state_with_sections(3));
        assert!(cold_start(&cache, original.clone()));
        let earlier_cold_start = cache.begin_cold_start();
        let guard = hit(&cache, OBJECT_ID);
        let unwound = catch_unwind(AssertUnwindSafe(move || {
            let mut guard = guard;
            assert!(plan_unbounded(&mut guard, create_delta(0, 100)).is_ok());
            panic!("the attempt panicked with its plan pending");
        }));
        assert!(unwound.is_err(), "the closure panics");
        assert!(
            matches!(cache.lookup(OBJECT_ID), CacheLookup::Miss),
            "a guard unwinding with a plan pending puts nothing back"
        );
        assert_eq!(
            cache.admit(earlier_cold_start, original).map(|_| ()),
            Err(AdmitRefused::Invalidated),
            "the discard invalidates the object"
        );
    }

    /// T8 ▲, the hard rule's direct regression: the per-entry cap reads the
    /// state's weight alone. An entry whose state weighs exactly the cap is
    /// stored at admission, and again at its put-back after a plan in place
    /// rolled back, though its state and plan indexes together weigh more
    /// than the cap: the indexes never make a cacheable layer uncacheable.
    #[test]
    fn entry_with_its_state_at_the_cap_is_stored_though_its_indexes_take_it_past() {
        let cap = shape_weight();
        let cache = cache_with(4 * cap);
        assert_eq!(
            cache.max_entry_bytes(),
            cap,
            "fixture: the cap is one shape"
        );
        assert!(
            cold_start(
                &cache,
                cut(OBJECT_ID, 1, state_with_sections(SHAPE_SECTIONS))
            ),
            "admitted and stored"
        );
        let mut guard = hit(&cache, OBJECT_ID);
        let entry = guard.entry();
        assert_eq!(
            entry.state_weight_bytes, cap,
            "fixture: the state is at the cap"
        );
        assert!(
            entry.index_weight_bytes > 0 && entry.weight_bytes() > cap,
            "fixture: with its plan indexes the entry weighs {} B, past the {cap} B cap",
            entry.weight_bytes()
        );
        assert!(plan_unbounded(&mut guard, create_delta(0, 9_000)).is_ok());
        assert_eq!(
            guard.put_back_held(),
            PutBack::Stored,
            "rolled back to a state at the cap, it is stored"
        );
        let found = hit(&cache, OBJECT_ID);
        assert_eq!(found.entry().state_weight_bytes, cap);
        assert!(found.entry().weight_bytes() > cap);
    }

    /// The no-accumulation fixture's sections, 1 to 6 under the root.
    const INTERNER_SECTIONS: u64 = 6;

    /// An inline set field of `key_name` on `section_id`.
    fn set_inline(section_id: u64, key_name: &str) -> oodf_ffi::SetField {
        oodf_ffi::SetField {
            section_id,
            key_name: key_name.to_owned(),
            value_or_ext_hash: vec![9],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }
    }

    /// The no-accumulation fixture, compacted: sections 1 to
    /// [`INTERNER_SECTIONS`] under the root, each holding a field name of its
    /// own (`u{id}`) and the shared `shared`, sections 5 and 6 also `g{id}`.
    /// The names are interned in a set order — `shared` (id 0), `g4` to `g6`
    /// (ids 1 to 3), `u1` to `u6` (ids 4 to 9) — and `g4`'s row is then
    /// removed, so the interner holds a free id (1) below live ones (a
    /// non-empty free list), and `u6` holds the highest id: a rollback that
    /// left `u6`'s pre-plan id free would leave it trailing, where
    /// compaction trims it and the state weight moves. `tests/proptests.rs`'s
    /// `interner_fixture` is the same state, for the pure half of the test.
    fn interner_fixture_state() -> CompactState {
        let mut state = ObjectState {
            object_id: OBJECT_ID,
            sequence_num: 5,
            first_version: 1,
            source_format: "usd".to_owned(),
            next_set_order: 10,
            ..ObjectState::default()
        };
        for id in 1..=INTERNER_SECTIONS {
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: 0,
                    name: format!("s{id}"),
                    section_type: 6,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
        }
        state.children.insert(0, (1..=INTERNER_SECTIONS).collect());
        let mut compacted = compact(&state);
        let rows = (1..=INTERNER_SECTIONS)
            .map(|id| (id, "shared".to_owned()))
            .chain((4..=INTERNER_SECTIONS).map(|id| (id, format!("g{id}"))))
            .chain((1..=INTERNER_SECTIONS).map(|id| (id, format!("u{id}"))));
        for (id, key_name) in rows {
            compacted.fields.insert(id, &key_name, None);
        }
        compacted.fields.remove(4, "g4").expect("fixture: g4's row");
        compact_state(&mut compacted);
        let names = compacted.fields.names();
        assert!(
            names.free_capacity() > 0 && names.id_of("u6") == Some(9),
            "fixture: a free id below live ones, and u6 at the highest id"
        );
        compacted
    }

    /// Cycle `i`'s delta on the no-accumulation fixture `pre`, and whether
    /// its plan runs on a budget that refuses part way. Every third cycle is
    /// each of: delete a section and write the names it released on another
    /// section beside a new name; delete a section, create it again with
    /// the same id and write its old names after a new name on another
    /// section (the changed-id path); one of those two on a budget that
    /// refuses at one of its first asks.
    fn accumulation_delta(pre: &CompactState, i: u64) -> (OwnedDelta, bool) {
        // Each of the three kinds meets every section in turn, `u6` (the
        // highest id) included.
        let k = 1 + (i / 3) % INTERNER_SECTIONS;
        let other = 1 + (i / 3 + 1) % INTERNER_SECTIONS;
        let old_names: Vec<String> = [format!("u{k}"), "shared".to_owned(), format!("g{k}")]
            .into_iter()
            .filter(|name| pre.fields.contains(k, name))
            .collect();
        let delete = StructCommand::DeleteSection(oodf_ffi::CommandDeleteSection {
            parent_id: 0,
            section_id: k,
        });
        let release = || OwnedDelta {
            struct_commands: vec![delete.clone()],
            set_fields: old_names
                .iter()
                .map(|name| set_inline(other, name))
                .chain(std::iter::once(set_inline(other, &format!("new{i}"))))
                .collect(),
            ..OwnedDelta::default()
        };
        let recreate = || OwnedDelta {
            struct_commands: vec![
                delete.clone(),
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: k,
                    section_name: format!("s{k}"),
                    section_type: 6,
                }),
            ],
            set_fields: std::iter::once(set_inline(other, &format!("fresh{i}")))
                .chain(old_names.iter().map(|name| set_inline(k, name)))
                .collect(),
            ..OwnedDelta::default()
        };
        match i % 3 {
            0 => (release(), false),
            1 => (recreate(), false),
            _ if (i / 3).is_multiple_of(2) => (release(), true),
            _ => (recreate(), true),
        }
    }

    /// The rolled-back attempts of the no-accumulation test.
    const ACCUMULATION_CYCLES: u64 = 64;

    /// No accumulation through the cache, at the per-entry cap (decider; the
    /// hard rule): an entry whose state weighs exactly the per-entry cap,
    /// with a free field-name id below live ones, stays
    /// [`PutBack::Stored`] — never [`PutBack::OverCap`] — through
    /// [`ACCUMULATION_CYCLES`] attempts planned in place on it and rolled
    /// back at their put-back, each a different delta
    /// ([`accumulation_delta`]); after every put-back its stored
    /// `state_weight_bytes` is the first, and it holds the pre-plan
    /// contents. The shadow check (on by default) also holds every rollback
    /// against the same plan on a copy. Hard preconditions over the run:
    /// some plan's budget refused it part way. Discriminated by disabling
    /// the rollback's re-key of a row whose name moved
    /// (`domain::plan_journal`): the changed-id cycle on section 6 leaves
    /// `u6`'s pre-plan id free at the top, compaction trims it, and the
    /// stored weight moves. The pure half, with the interner compared id by
    /// id, is `tests/proptests.rs`'s
    /// `rolled_back_attempts_never_accumulate_weight_on_one_entry`.
    ///
    /// C2 (b.k39): run with each arm of the compact field purge forced on
    /// this thread, where the cache plans in place and its shadow check
    /// plans on a copy (the fixture's one-section purges take the scan arm
    /// by content). On each, after every put-back the stored entry's
    /// interner, as compacted, also matches the pre-plan one (id-vector and
    /// free-list capacities, each name's id), and both arms end with the
    /// same stored entry. Hard precondition: the forced arm's purges
    /// released a name, and no purge ran on the other arm.
    #[test]
    fn rolled_back_attempts_at_the_cap_keep_the_entry_stored_at_its_first_weight() {
        use crate::domain::types::purge_arm::{self, PurgeArm};

        let pre = interner_fixture_state();
        let mut names: Vec<String> = pre.fields.names().names().map(str::to_owned).collect();
        names.sort_unstable();
        let layout = |state: &CompactState| {
            let interner = state.fields.names();
            (
                interner.slots_capacity(),
                interner.free_capacity(),
                names
                    .iter()
                    .map(|name| interner.id_of(name))
                    .collect::<Vec<_>>(),
            )
        };
        let pre_layout = layout(&pre);
        let mut settled_by_arm = Vec::new();
        for arm in [PurgeArm::Probe, PurgeArm::Scan] {
            let _forced = purge_arm::force(Some(arm));
            let (weight, state) = rolled_back_at_the_cap(&pre, arm, |state| {
                assert_eq!(
                    layout(state),
                    pre_layout,
                    "{arm:?}: the stored interner (id-vector capacity, free-list capacity, each \
                     name's id)"
                );
            });
            settled_by_arm.push((layout(&state), weight, state));
        }
        assert_eq!(
            settled_by_arm[0], settled_by_arm[1],
            "the probe and the scan arm end with the same stored entry"
        );
    }

    /// [`rolled_back_attempts_at_the_cap_keep_the_entry_stored_at_its_first_weight`]'s
    /// run with `arm` forced on this thread, `check` called on the stored
    /// state after every put-back: the stored entry's state weight and
    /// state at the end.
    fn rolled_back_at_the_cap(
        pre: &CompactState,
        arm: crate::domain::types::purge_arm::PurgeArm,
        check: impl Fn(&CompactState),
    ) -> (u64, CompactState) {
        use crate::domain::types::purge_arm::{self, PurgeArm};

        let cap = estimate_state_bytes(pre);
        let cache = cache_with(4 * cap);
        assert_eq!(
            cache.max_entry_bytes(),
            cap,
            "fixture: the state weighs the per-entry cap exactly"
        );
        assert!(cold_start(&cache, compact_cut(OBJECT_ID, 5, pre.clone())));
        let first = hit(&cache, OBJECT_ID).entry().state_weight_bytes;
        assert_eq!(first, cap, "fixture: stored at the cap");
        let mut overflows = 0u32;
        let mut released = 0u64;
        for i in 0..ACCUMULATION_CYCLES {
            let (delta, refusing) = accumulation_delta(pre, i);
            let asks = std::cell::Cell::new(0u64);
            let refuse_at = (i / 6) % 3;
            let mut budget = |needed: RewriteCount| {
                let ask = asks.get();
                asks.set(ask + 1);
                (!refusing || ask != refuse_at).then(|| needed.bytes(1, 1, 1, 1, 1))
            };
            let allowance = if refusing {
                MutationAllowance {
                    bytes: 0,
                    row_bytes: 1,
                    list_entry_bytes: 1,
                    index_entry_bytes: 1,
                    adoption_entry_bytes: 1,
                    walk_entry_bytes: 1,
                }
            } else {
                MutationAllowance::UNBOUNDED
            };
            let mut guard = hit(&cache, OBJECT_ID);
            purge_arm::reset();
            let plan = guard.plan_in_place(
                Some(delta),
                allowance,
                &mut budget,
                &mut ApplyDeltaStats::default(),
            );
            match plan {
                Ok(BoundedPlan::Within { .. }) => {}
                Ok(BoundedPlan::Overflow { .. }) => overflows += 1,
                Err(e) => panic!("{arm:?}, cycle {i}: fixture: the delta applies: {e}"),
            }
            let counts = purge_arm::counts();
            let off_arm = match arm {
                PurgeArm::Probe => counts.scan_purges,
                PurgeArm::Scan => counts.probe_purges,
            };
            assert_eq!(
                off_arm, 0,
                "{arm:?}, cycle {i}: a purge ran on the other arm: {counts:?}"
            );
            released += counts.names_released;
            assert_eq!(
                guard.put_back_held(),
                PutBack::Stored,
                "{arm:?}, cycle {i}: the rolled-back entry is stored, never over the cap"
            );
            let held = hit(&cache, OBJECT_ID);
            assert_eq!(
                held.entry().state_weight_bytes,
                first,
                "{arm:?}, cycle {i}: the state weight after the put-back"
            );
            assert_eq!(
                &held.entry().state,
                pre,
                "{arm:?}, cycle {i}: the pre-plan contents"
            );
            check(&held.entry().state);
        }
        assert!(
            overflows > 0,
            "{arm:?}: fixture: some plan's budget refused it part way"
        );
        assert!(
            released > 0,
            "{arm:?}: fixture: the forced arm's purges released a name"
        );
        let held = hit(&cache, OBJECT_ID);
        let entry = held.entry();
        (entry.state_weight_bytes, entry.state.clone())
    }

    /// T7: the shadow check is on by default in every test and test-support
    /// build, so every plan in place the suites make is checked against the
    /// same plan on a copy; a measurement harness turns it off explicitly.
    #[test]
    fn shadow_check_is_on_by_default() {
        let cache = StructuralCache::new(BUDGET_BYTES);
        assert!(cache.shadow_check_for_test(), "on by default");
        cache.set_shadow_check_for_test(false);
        assert!(!cache.shadow_check_for_test(), "turned off explicitly");
        assert!(
            !cache.clone().shadow_check_for_test(),
            "every clone of the cache shares the switch"
        );
    }

    /// W2-T4: CR1's growth term prices the external-hash counts for the
    /// delta's external rows alone — its set fields and time samples that
    /// carry an external value — not for its inline ones: on counts with no
    /// room, `N` inline set fields price no growth of them, and `k` external
    /// rows price the counts' growth through `k` insert calls, whatever `N`.
    #[test]
    fn in_place_growth_prices_external_hash_growth_for_the_external_rows_only() {
        let mut full = state_with_sections(3);
        for (id, field) in full.fields.values_mut().enumerate() {
            field.value = FieldValue::External {
                hash: [id as u8 + 1; 32],
                size: 64,
                ext_value_index: 0,
            };
        }
        let mut state = compact(&full);
        compact_state(&mut state);
        let mut indexes = PlanIndexes::build(&state);
        indexes.compact();
        assert_eq!(
            indexes.hash_key_room(),
            0,
            "fixture: the external-hash counts have no room ({} hashes)",
            indexes.hash_keys()
        );
        let field = |section_id: u64, external: bool| oodf_ffi::SetField {
            section_id,
            key_name: format!("k{section_id}"),
            value_or_ext_hash: if external { vec![0xee; 32] } else { vec![1] },
            ext_value_size: if external { 64 } else { 0 },
            ext_value_index: 0,
            set_order: 1,
        };
        let sample = |external: bool| oodf_ffi::TimeSample {
            section_id: 1,
            time: 1.0,
            value_or_ext_hash: if external { vec![0xdd; 32] } else { vec![1] },
            ext_value_size: if external { 64 } else { 0 },
            ext_value_index: 0,
        };
        let delta = |inline: u64, external_fields: u64, external_samples: bool| OwnedDelta {
            set_fields: (1..=inline)
                .map(|id| field(id, false))
                .chain((1..=external_fields).map(|id| field(id, true)))
                .collect(),
            time_samples: vec![sample(external_samples)],
            ..OwnedDelta::default()
        };
        // The index maps' growth alone: the estimate with the indexes less
        // the estimate without them.
        let index_growth = |delta: &OwnedDelta| {
            let inserts = PlannedInserts::of(Some(delta));
            estimate_in_place_growth_bytes(&state, Some(&indexes), inserts)
                - estimate_in_place_growth_bytes(&state, None, inserts)
        };
        let ext_slot = PlanIndexes::EXT_SLOT_BYTES as u64;
        let hashes = indexes.hash_keys();
        let grown_by = |calls: u64| table_growth_bytes(hashes, hashes, calls, ext_slot);
        for (inline, external_fields, external_samples) in [
            (5, 0, false),
            (0, 1, false),
            (5, 1, false),
            (5, 0, true),
            (5, 2, true),
        ] {
            let delta = delta(inline, external_fields, external_samples);
            let external = external_fields + u64::from(external_samples);
            assert_eq!(
                PlannedInserts::of(Some(&delta)).external_rows,
                external,
                "{inline} inline, {external} external: the external rows"
            );
            assert_eq!(
                index_growth(&delta),
                grown_by(external),
                "{inline} inline and {external} external rows price the counts' growth for \
                 {external} insert calls"
            );
        }
        assert!(
            grown_by(1) > 0,
            "fixture: one insert call grows the full counts"
        );
    }

    // ── estimate_replay_record_growth (the stream repair's RP2) ──────────────
    //
    // Checked against what std actually allocates when the record replays:
    // each test applies the record to a clone of the state through the
    // repair's own decode and apply steps and reads back every table's and
    // list's capacity. Every case is a single growth of one table or one
    // list, so std's allocation is the new table or buffer alone (the old one
    // is the entry's); every other table is given room, since the estimate
    // prices a creation's insert calls on the children-list table (its
    // parent's key and its own) whether or not the key is new, where std
    // allocates nothing for an existing key. The hash tables are at
    // capacities of 15 or more, where
    // hashbrown's bucket count does not depend on its group width (8 on
    // aarch64, 16 on x86_64), and only insert, so no tombstone decides a
    // growth: the cases hold at either width.

    /// The parent the growth cases list their children under.
    const GROWTH_PARENT: u64 = 1;
    /// The section the growth cases' records create.
    const GROWTH_NEW: u64 = 900;

    /// A compact state: [`GROWTH_PARENT`] under the root, listing `children`
    /// sections (ids from 100), every list and table at exact capacity, as a
    /// stored entry is (`compact_and_weigh`, `normalize_touched`).
    fn growth_base(children: u64) -> CompactState {
        let section = |id: u64, parent: u64| Section {
            section_id: id,
            parent_id: parent,
            name: format!("s{id}"),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        };
        let mut state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..ObjectState::default()
        };
        state
            .sections
            .insert(GROWTH_PARENT, section(GROWTH_PARENT, 0));
        state.children.insert(0, vec![GROWTH_PARENT]);
        let kids: Vec<u64> = (100..100 + children).collect();
        for &id in &kids {
            state.sections.insert(id, section(id, GROWTH_PARENT));
        }
        state.children.insert(GROWTH_PARENT, kids);
        let mut compacted = compact(&state);
        compact_state(&mut compacted);
        for list in compacted.children.values_mut() {
            list.shrink_to_fit();
        }
        compacted
    }

    /// One `CreateSection` of [`GROWTH_NEW`] under `parent`.
    fn create_under(parent: u64) -> OwnedDelta {
        OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: parent,
                section_id: GROWTH_NEW,
                section_name: "new".to_owned(),
                section_type: 1,
            })],
            ..Default::default()
        }
    }

    /// `delta` replayed onto a clone of `state` — decoded from its OMPF and
    /// applied as the repair applies it, through `sink` — and the state
    /// after.
    fn replayed(
        state: &CompactState,
        delta: &OwnedDelta,
        sink: &mut crate::domain::delta::MutationSink,
    ) -> CompactState {
        let ompf = crate::domain::ompf::OmpfContainer::serialize(
            &oodf_ffi::build_delta(delta).expect("build_delta"),
            &[],
        )
        .expect("serialize OMPF");
        let mut after = state.clone();
        let record = crate::domain::replay::decode_stream_record(
            after.sequence_num,
            Some(&ompf),
            after.sequence_num + 1,
        )
        .expect("the record decodes");
        crate::domain::replay::apply_decoded_record_into(&mut after, record, sink)
            .expect("the record applies");
        after
    }

    /// What std allocated growing `before` into `after`, one growth each: the
    /// table each hash table that changed capacity grew to, and the buffer
    /// each children list that changed capacity grew to (a list new in
    /// `after` included).
    fn std_growth(before: &CompactState, after: &CompactState) -> u64 {
        let table = |before: usize, after: usize, slot: u64| {
            if before == after {
                0
            } else {
                table_bytes(after, slot)
            }
        };
        let lists: u64 = after
            .children
            .iter()
            .filter(|(parent, list)| {
                before
                    .children
                    .get(parent)
                    .is_none_or(|old| old.capacity() != list.capacity())
            })
            .map(|(_, list)| list_bytes(list.capacity()))
            .sum();
        table(
            before.sections.capacity(),
            after.sections.capacity(),
            SECTION_SLOT_BYTES,
        ) + table(
            before.children.capacity(),
            after.children.capacity(),
            CHILDREN_SLOT_BYTES,
        ) + lists
    }

    /// A creation into a sections table at its load limit (28 entries at
    /// capacity 28, 32 buckets) under a parent whose list has room: the
    /// estimate's growth is exactly the table std grows to, and nothing else
    /// grows.
    #[test]
    fn estimate_replay_record_growth_prices_a_full_tables_doubling_at_the_table_std_grows_to() {
        let mut state = growth_base(27);
        state
            .children
            .get_mut(&GROWTH_PARENT)
            .expect("the parent's list")
            .reserve(4);
        state.children.reserve(8);
        assert_eq!(
            (state.sections.len(), state.sections.capacity()),
            (28, 28),
            "fixture: a full sections table"
        );
        let delta = create_under(GROWTH_PARENT);

        let estimate = estimate_replay_record_growth(&state, None, Some(&delta), false);
        let after = replayed(
            &state,
            &delta,
            &mut crate::domain::delta::MutationSink::Discard,
        );

        assert!(
            after.sections.capacity() > state.sections.capacity(),
            "fixture: std grew the sections table"
        );
        assert_eq!(
            estimate.growth,
            std_growth(&state, &after),
            "the doubled sections table, as std allocated it"
        );
        assert_eq!(
            estimate.growth,
            table_bytes(after.sections.capacity(), SECTION_SLOT_BYTES)
        );
    }

    /// A creation under a parent of 100 children whose list is at its
    /// length (trimmed, as every touched list is after each replayed
    /// record), on a sections table with room: the estimate's growth is
    /// exactly the buffer std grows the list to (twice it), with the
    /// discarding sink and the list-delete sink alike — the copy below is
    /// not growth. Through the list-delete sink (an entry with dangling
    /// parents), the dirty parent's `SetChildren` is built with a copy of
    /// its final list and dropped: priced in `scratch`, at the list a
    /// collecting sink would hold, beside the sink's own ids and its one
    /// mutation in flight (`list_delete_sink_bytes`, at no list a cascade
    /// deletes and the one touched parent). The creation cannot cascade
    /// (no row is named `new` under the parent), so nothing else is
    /// scratch, and with the discarding sink there is none.
    ///
    /// Pins RP2's list term: zeroed, the growth is 0 against std's real
    /// doubling. Holds at group width 8 and 16: the list is a `Vec`, which
    /// no group width touches, and the tables have room at either width.
    #[test]
    fn estimate_replay_record_growth_prices_a_trimmed_lists_doubling_and_the_collected_copy() {
        let mut state = growth_base(100);
        state.sections.reserve(8);
        state.children.reserve(8);
        let list = &state.children[&GROWTH_PARENT];
        assert_eq!(
            (list.len(), list.capacity()),
            (100, 100),
            "fixture: a list at its length"
        );
        assert!(
            state.sections.capacity() > state.sections.len(),
            "fixture: the sections table has room"
        );
        let delta = create_under(GROWTH_PARENT);

        let discarding = estimate_replay_record_growth(&state, None, Some(&delta), false);
        let list_deleting = estimate_replay_record_growth(&state, None, Some(&delta), true);
        let after = replayed(
            &state,
            &delta,
            &mut crate::domain::delta::MutationSink::Discard,
        );
        let mut sink = crate::domain::delta::MutationSink::collecting();
        replayed(&state, &delta, &mut sink);

        assert_eq!(
            discarding.growth,
            std_growth(&state, &after),
            "the parent's list at the buffer std grew it to"
        );
        assert_eq!(
            discarding.growth,
            list_bytes(after.children[&GROWTH_PARENT].capacity())
        );
        let crate::domain::delta::MutationSink::Collect(mutations) = sink else {
            panic!("a collecting sink");
        };
        let copied: u64 = mutations
            .iter()
            .filter_map(|mutation| match mutation {
                StorageMutation::SetChildren {
                    section_id,
                    children,
                } if *section_id == GROWTH_PARENT => Some(list_bytes(children.len())),
                _ => None,
            })
            .sum();
        assert!(copied > 0, "fixture: the sink copied the parent's list");
        assert_eq!(
            (discarding.scratch, list_deleting.growth),
            (0, discarding.growth),
            "the discarding sink prices no scratch, and the sink changes no growth"
        );
        assert_eq!(
            list_deleting.scratch,
            copied + crate::domain::delta::list_delete_sink_bytes(0, 1),
            "through the list-delete sink, the parent's SetChildren copy is scratch, at the list a \
             collecting sink holds, beside the sink's ids and its one mutation in flight"
        );
    }

    /// With plan indexes, a creation pushes one entry onto its parent's
    /// name-index vector, which is not trimmed: the estimate prices exactly
    /// what that vector grows by as the indexed replay pushes it, from its
    /// own length and capacity (`PlanIndexes::name_vector_room`), on top of
    /// the list's growth and the plan indexes' own tables
    /// (`PlanIndexes::growth_bytes`, CR1's rule); nothing when it has room.
    /// Two parents: one whose vector is full when built, one with room.
    #[test]
    fn estimate_replay_record_growth_prices_the_parents_name_vector_as_the_indexed_replay_grows_it()
    {
        for children in [100u64, 128] {
            let mut state = growth_base(children);
            state.sections.reserve(8);
            state.children.reserve(8);
            let mut indexes = PlanIndexes::build(&state);
            let (len, capacity) = indexes.name_vector_room(GROWTH_PARENT);
            assert_eq!(len, children as usize, "fixture: one entry per child");
            let delta = create_under(GROWTH_PARENT);
            let without = estimate_replay_record_growth(&state, None, Some(&delta), false).growth;
            // The plan indexes' own tables (the name index's parent keys),
            // priced by their own rule for the record's insert calls.
            let inserts = PlannedInserts::of(Some(&delta));
            let index_tables = indexes.growth_bytes(|table| inserts.calls(table), false);

            let estimate =
                estimate_replay_record_growth(&state, Some(&indexes), Some(&delta), false);
            let ompf = crate::domain::ompf::OmpfContainer::serialize(
                &oodf_ffi::build_delta(&delta).expect("build_delta"),
                &[],
            )
            .expect("serialize OMPF");
            let mut after = state.clone();
            let mut touches = PlanTouches::default();
            let seq = after.sequence_num + 1;
            crate::domain::replay::replay_stream_record_indexed(
                &mut after,
                Some(&ompf),
                seq,
                &mut crate::domain::delta::MutationSink::Discard,
                &mut indexes,
                &mut touches,
            )
            .expect("the indexed replay applies");
            let (_, grown_to) = indexes.name_vector_room(GROWTH_PARENT);
            let vector = if grown_to == capacity {
                0
            } else {
                allocation_bytes(grown_to as u64 * PlanIndexes::NAME_ENTRY_BYTES as u64)
            };

            assert_eq!(
                estimate.growth,
                without + index_tables + vector,
                "the name vector ({len} of {capacity}) as the indexed replay grew it (to \
                 {grown_to})"
            );
        }
    }

    /// The rewrite counts: a record that cannot change the tree (a field
    /// write, an empty commit) prices none; a creation on an entry without
    /// plan indexes prices the per-delta indexes the engine builds per
    /// record — every section and the birth in its name index, every
    /// section in its adoption index — and the full depth walk, every
    /// section and the birth; with plan indexes that passed the full walk,
    /// none of the per-delta indexes and only the incremental walk, far
    /// below the full one.
    #[test]
    fn estimate_replay_record_growth_counts_the_index_less_scratch_and_the_walk() {
        let state = growth_base(100);
        let sections = state.sections.len() as u64;
        let field_write = OwnedDelta {
            set_fields: vec![oodf_ffi::SetField {
                section_id: GROWTH_PARENT,
                key_name: "k".to_owned(),
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            }],
            ..Default::default()
        };
        for (what, delta) in [
            ("a field write", Some(&field_write)),
            ("an empty commit", None),
        ] {
            assert_eq!(
                estimate_replay_record_growth(&state, None, delta, false).rewrite,
                RewriteCount::default(),
                "{what}: no rewrite"
            );
        }

        let delta = create_under(GROWTH_PARENT);
        let index_less = estimate_replay_record_growth(&state, None, Some(&delta), false).rewrite;
        assert_eq!(
            (
                index_less.indexed_sections,
                index_less.adoption_entries,
                index_less.walked_sections
            ),
            (sections + 1, sections, sections + 1),
            "without plan indexes: the per-delta name and adoption indexes and the full walk"
        );
        assert_eq!(
            index_less.list_entries,
            state.children[&GROWTH_PARENT].len() as u64 + 1,
            "the parent's listing at its length after the record"
        );

        let indexes = PlanIndexes::build(&state);
        assert!(indexes.depth_ok(), "fixture: the tree passed the full walk");
        let indexed =
            estimate_replay_record_growth(&state, Some(&indexes), Some(&delta), false).rewrite;
        assert_eq!(
            (indexed.indexed_sections, indexed.adoption_entries),
            (0, 0),
            "with plan indexes: no per-delta index"
        );
        assert!(
            indexed.walked_sections > 0 && indexed.walked_sections < sections,
            "the incremental walk ({}) is below the full walk ({sections})",
            indexed.walked_sections
        );
    }

    // ── RP2's scratch: the subtrees a record can cascade (b.fsz dq-1) ────────
    //
    // `estimate_replay_record_growth`'s `scratch` prices every cascade a
    // record can make at the pushes of the subtrees it can delete
    // (`domain::delta::cascade_scratch_bytes` over `subtree_pushes`). On a
    // tree, the engine's cascade walk pushes exactly the root and every list
    // below it, so these tests check the estimate against the model at the
    // engine's own pushes for each kind of root — a `DeleteSection` target, a
    // `MoveSection` id, a re-parented `DiffChangeSection` id, a same-name
    // sibling of a creation, and a birth with the orphans it adopts — on an
    // entry with plan indexes and on its copy without them, which must find
    // the same roots. Each case's pushes and touched lists are counted by hand
    // from the fixture below. The model's byte terms are checked against std's
    // collections in `domain::delta`'s tests; here only the pushes matter, and
    // they do not depend on hashbrown's group width (8 on aarch64, 16 on
    // x86_64): the walk counts list entries, never table slots.

    /// The world, under the root.
    const CASCADE_WORLD: u64 = 1;
    /// A group of [`CASCADE_GROUP_CHILDREN`] leaves (ids from 100) under the
    /// world.
    const CASCADE_GROUP: u64 = 2;
    const CASCADE_GROUP_CHILDREN: u64 = 50;
    /// A second group, of [`CASCADE_SECOND_CHILDREN`] leaves (ids from 200).
    const CASCADE_SECOND: u64 = 3;
    const CASCADE_SECOND_CHILDREN: u64 = 30;
    /// A leaf under the world.
    const CASCADE_LEAF: u64 = 4;
    /// An orphan: a section naming [`CASCADE_MISSING`], which has no row,
    /// with [`CASCADE_ORPHAN_CHILDREN`] leaves of its own (ids from 300).
    const CASCADE_ORPHAN: u64 = 5;
    const CASCADE_ORPHAN_CHILDREN: u64 = 20;
    const CASCADE_MISSING: u64 = 900;

    /// The cascade fixture, compacted as a stored entry is: the world with
    /// the two groups and the leaf, and the orphan's subtree under the
    /// missing parent (no list under it, as a normalized entry holds none).
    /// Every section is named `s{id}`.
    fn cascade_base() -> CompactState {
        let mut state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..ObjectState::default()
        };
        let add = |state: &mut ObjectState, id: u64, parent: u64| {
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: parent,
                    name: format!("s{id}"),
                    section_type: 1,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
            if state.sections.contains_key(&parent) || parent == 0 {
                state.children.entry(parent).or_default().push(id);
            }
        };
        add(&mut state, CASCADE_WORLD, 0);
        for id in [CASCADE_GROUP, CASCADE_SECOND, CASCADE_LEAF] {
            add(&mut state, id, CASCADE_WORLD);
        }
        for id in 100..100 + CASCADE_GROUP_CHILDREN {
            add(&mut state, id, CASCADE_GROUP);
        }
        for id in 200..200 + CASCADE_SECOND_CHILDREN {
            add(&mut state, id, CASCADE_SECOND);
        }
        add(&mut state, CASCADE_ORPHAN, CASCADE_MISSING);
        for id in 300..300 + CASCADE_ORPHAN_CHILDREN {
            add(&mut state, id, CASCADE_ORPHAN);
        }
        let mut compacted = compact(&state);
        compact_state(&mut compacted);
        for list in compacted.children.values_mut() {
            list.shrink_to_fit();
        }
        compacted
    }

    /// One record of `commands`.
    fn record_of(commands: Vec<StructCommand>) -> OwnedDelta {
        OwnedDelta {
            struct_commands: commands,
            ..Default::default()
        }
    }

    fn delete_cmd(parent_id: u64, section_id: u64) -> StructCommand {
        StructCommand::DeleteSection(oodf_ffi::CommandDeleteSection {
            parent_id,
            section_id,
        })
    }

    fn create_cmd(parent_id: u64, section_id: u64, name: &str) -> StructCommand {
        StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: name.to_owned(),
            section_type: 1,
        })
    }

    /// The name interner's `(live, free length, free capacity)`, as the
    /// estimate reads it.
    fn name_room(state: &CompactState) -> (usize, usize, usize) {
        let names = state.fields.names();
        (names.len(), names.free_len(), names.free_capacity())
    }

    /// The scratch the estimate prices for `delta` on `state`, through the
    /// discarding sink, on the entry with plan indexes and on its copy
    /// without: `(indexed, copy)`.
    fn scratch_on_both(state: &CompactState, delta: &OwnedDelta) -> (u64, u64) {
        let indexes = PlanIndexes::build(state);
        (
            estimate_replay_record_growth(state, Some(&indexes), Some(delta), false).scratch,
            estimate_replay_record_growth(state, None, Some(delta), false).scratch,
        )
    }

    /// Each root kind's cascade is priced at the engine's own pushes over
    /// the subtree it deletes, with plan indexes (their touched-list set
    /// included) and without, which find the same roots: a `DeleteSection`
    /// of a group (1 + 50 pushes); a `MoveSection` of it to a missing parent
    /// (its edge that deletes the moved section); a `DiffChangeSection`
    /// re-parenting the second group (1 + 30) under the leaf, then a
    /// `DeleteSection` of the leaf, whose cascade takes the group along (1 +
    /// 31); a `CreateSection` named as the group under the world, its
    /// same-name sibling, whose root no command names (1 birth + 51); and a
    /// birth of the missing parent, adopting the orphan, deleted again (1
    /// birth + the orphan's 1 + 20). The touched lists, by hand: the
    /// world's (every case), the move's missing destination, the leaf's (the
    /// re-parent's destination) and the newborn's (its adoption).
    #[test]
    fn estimate_replay_record_growth_prices_each_cascade_roots_subtree_at_the_engines_pushes() {
        let state = cascade_base();
        let names = name_room(&state);
        let group = 1 + CASCADE_GROUP_CHILDREN;
        let second = 1 + CASCADE_SECOND_CHILDREN;
        let orphan = 1 + CASCADE_ORPHAN_CHILDREN;
        let cases: [(&str, OwnedDelta, u64, u64); 5] = [
            (
                "a DeleteSection",
                record_of(vec![delete_cmd(CASCADE_WORLD, CASCADE_GROUP)]),
                group,
                1,
            ),
            (
                "a MoveSection to a missing parent",
                record_of(vec![StructCommand::MoveSection(
                    oodf_ffi::CommandMoveSection {
                        old_parent_id: CASCADE_WORLD,
                        new_parent_id: 999,
                        section_id: CASCADE_GROUP,
                        new_name: "moved".to_owned(),
                    },
                )]),
                group,
                2,
            ),
            (
                "a DiffChangeSection re-parent, then a DeleteSection of its new parent",
                record_of(vec![
                    StructCommand::DiffChangeSection(oodf_ffi::CommandDiffChangeSection {
                        parent_id: CASCADE_LEAF,
                        section_id: CASCADE_SECOND,
                        section_name: format!("s{CASCADE_SECOND}"),
                        section_type: 1,
                        section_order: 0,
                    }),
                    delete_cmd(CASCADE_WORLD, CASCADE_LEAF),
                ]),
                1 + second,
                2,
            ),
            (
                "a CreateSection over its same-name sibling",
                record_of(vec![create_cmd(
                    CASCADE_WORLD,
                    777,
                    &format!("s{CASCADE_GROUP}"),
                )]),
                1 + group,
                1,
            ),
            (
                "a birth adopting an orphan, deleted again",
                record_of(vec![
                    create_cmd(CASCADE_WORLD, CASCADE_MISSING, "adopter"),
                    delete_cmd(CASCADE_WORLD, CASCADE_MISSING),
                ]),
                1 + orphan,
                2,
            ),
        ];
        for (what, delta, pushes, touched) in cases {
            assert_eq!(
                scratch_on_both(&state, &delta),
                (
                    cascade_scratch_bytes(pushes, touched, true, names),
                    cascade_scratch_bytes(pushes, touched, false, names),
                ),
                "{what}: the cascade model at the engine's {pushes} pushes and {touched} touched \
                 lists, with plan indexes and without"
            );
        }
    }

    /// A record that cannot cascade — a creation with no same-name sibling,
    /// a rename in place with none, a reorder, a field write, a
    /// `DiffDeleteSection` (ignored on ingest) and an empty commit — prices
    /// no scratch through the discarding sink, with plan indexes or without.
    #[test]
    fn estimate_replay_record_growth_prices_no_scratch_for_a_record_that_cannot_cascade() {
        let state = cascade_base();
        let field_write = OwnedDelta {
            set_fields: vec![oodf_ffi::SetField {
                section_id: CASCADE_LEAF,
                key_name: "k".to_owned(),
                value_or_ext_hash: vec![1],
                ..Default::default()
            }],
            ..Default::default()
        };
        let cases = [
            (
                "a creation with no same-name sibling",
                record_of(vec![create_cmd(CASCADE_WORLD, 777, "fresh")]),
            ),
            (
                "a rename in place with no same-name sibling",
                record_of(vec![StructCommand::DiffChangeSection(
                    oodf_ffi::CommandDiffChangeSection {
                        parent_id: CASCADE_WORLD,
                        section_id: CASCADE_LEAF,
                        section_name: "renamed".to_owned(),
                        section_type: 1,
                        section_order: 0,
                    },
                )]),
            ),
            (
                "a reorder",
                record_of(vec![StructCommand::ReorderChildren(
                    oodf_ffi::CommandReorderChildren {
                        section_id: CASCADE_WORLD,
                        children_list_id: 0,
                        children_list: vec![CASCADE_LEAF, CASCADE_SECOND, CASCADE_GROUP],
                    },
                )]),
            ),
            ("a field write", field_write),
            (
                "a DiffDeleteSection",
                record_of(vec![StructCommand::DiffDeleteSection(
                    oodf_ffi::CommandDiffDeleteSection {
                        section_id: CASCADE_GROUP,
                    },
                )]),
            ),
        ];
        for (what, delta) in cases {
            assert_eq!(scratch_on_both(&state, &delta), (0, 0), "{what}");
        }
        let indexes = PlanIndexes::build(&state);
        assert_eq!(
            (
                estimate_replay_record_growth(&state, Some(&indexes), None, false).scratch,
                estimate_replay_record_growth(&state, None, None, false).scratch,
            ),
            (0, 0),
            "an empty commit"
        );
    }

    /// When a cascade's walk cannot finish within the state's sections — a
    /// cycle in the stored lists — or runs past `SUBTREE_WALK_FRAMES` levels
    /// below its root, the estimate prices the whole state's pushes: the
    /// root and every list entry the state holds (with plan indexes, their
    /// list bytes over 8 B, at least the entries; without, the lists'
    /// lengths summed).
    #[test]
    fn estimate_replay_record_growth_prices_the_whole_state_past_the_walks_room_or_depth() {
        let mut cyclic = cascade_base();
        // The second group's first leaf lists the group: a cycle.
        cyclic.children.insert(200, vec![CASCADE_SECOND]);
        let mut deep = cascade_base();
        let mut parent = CASCADE_LEAF;
        for id in 1_000..1_000 + crate::domain::delta::SUBTREE_WALK_FRAMES as u64 + 1 {
            deep.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: parent,
                    name: format!("s{id}"),
                    section_type: 1,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
            deep.children.insert(parent, vec![id]);
            parent = id;
        }
        for (what, state, root) in [
            ("a cycle", &cyclic, CASCADE_SECOND),
            ("a chain past the walk's frames", &deep, CASCADE_LEAF),
        ] {
            let delta = record_of(vec![delete_cmd(CASCADE_WORLD, root)]);
            let names = name_room(state);
            let entries: u64 = state.children.values().map(|list| list.len() as u64).sum();
            let indexes = PlanIndexes::build(state);
            let indexed_entries = indexes.lists_bytes() / 8;
            assert!(
                indexed_entries >= entries,
                "fixture self-check: {what}: the indexes' list bytes cover every entry"
            );
            assert_eq!(
                (
                    estimate_replay_record_growth(state, Some(&indexes), Some(&delta), false)
                        .scratch,
                    estimate_replay_record_growth(state, None, Some(&delta), false).scratch,
                ),
                (
                    cascade_scratch_bytes(1 + indexed_entries, 1, true, names),
                    cascade_scratch_bytes(1 + entries, 1, false, names),
                ),
                "{what}: the root and the whole state's list entries"
            );
        }
    }

    /// The table registry's guard. Every hash table a cache entry owns — the
    /// state's, its field store's and name interner's, and the plan indexes'
    /// own — is listed by the destructuring enumeration
    /// (`domain::plan_index::hash_tables_for_test`, which fails to build
    /// while a field added to any of those types is unclassified) under
    /// exactly one [`RebuildTable`]; every registered table is listed; the
    /// registry's lookup reads the listed map; and every O(1) weigher reads
    /// each registered table at its held capacity: raising one table's
    /// capacity high-water raises [`CacheEntry::weights_now`] by exactly
    /// that table's bytes at the raised capacity less its bytes at
    /// `capacity()`, keeps the indexes' running weight equal to its
    /// recomputation, and raises the in-place growth term of inserts that
    /// pass the table's room. So a new map is weighed only through a
    /// registry entry, and a registered one by every weigher. (The per-table
    /// insert calls below are an exhaustive match, so a new variant fails to
    /// build here too until it says which calls land on it.)
    #[test]
    fn every_hash_table_of_an_entry_is_registered_and_weighed_at_its_held_capacity() {
        use crate::domain::plan_index::{hash_tables_for_test, table_shape};

        let mut state = CompactState {
            sequence_num: 1,
            first_version: 1,
            ..CompactState::default()
        };
        for id in 1..=3u64 {
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: 0,
                    name: format!("s{id}"),
                    section_type: 1,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
        }
        state.children.insert(0, vec![1, 2, 3]);
        state.fields.insert(1, "external", Some([7; 32]));
        state.fields.insert(2, "inline", None);
        state
            .time_samples
            .insert(1, 1.0f64.to_bits(), Some([8; 32]));
        compact_state(&mut state);
        let mut entry = CacheEntry::new(1, "chain".to_owned(), 1, HashSet::new(), state);

        let indexes = entry
            .plan
            .as_ref()
            .expect("CacheEntry::new builds the indexes");
        let tables = hash_tables_for_test(&entry.state, indexes);
        assert_eq!(tables.len(), RebuildTable::COUNT, "{tables:?}");
        for table in RebuildTable::ALL {
            let listed: Vec<_> = tables.iter().filter(|(t, _)| *t == table).collect();
            assert_eq!(listed.len(), 1, "{table:?} is listed once: {tables:?}");
            let (len, capacity) = table_shape(table, &entry.state, indexes);
            assert_eq!(
                capacity, listed[0].1.capacity,
                "the registry reads {table:?}'s own map"
            );
            assert!(len > 0, "fixture: {table:?} holds a key");
        }

        for table in RebuildTable::ALL {
            let (len, capacity) =
                table_shape(table, &entry.state, entry.plan.as_ref().expect("indexes"));
            let raised = usize::try_from(bucket_capacity(
                table_buckets(capacity as u64).saturating_mul(4),
            ))
            .expect("fits");
            // Insert calls just past the table's room, on it alone among
            // the tables its calls field prices.
            let calls = (capacity - len + 1) as u64;
            let inserts = match table {
                RebuildTable::Sections => PlannedInserts {
                    sections: calls,
                    ..PlannedInserts::default()
                },
                RebuildTable::Children => PlannedInserts {
                    children_list_keys: calls,
                    ..PlannedInserts::default()
                },
                RebuildTable::FieldRows | RebuildTable::NameIds => PlannedInserts {
                    set_fields: calls,
                    ..PlannedInserts::default()
                },
                RebuildTable::SampleRows => PlannedInserts {
                    time_samples: calls,
                    ..PlannedInserts::default()
                },
                RebuildTable::NameIndex => PlannedInserts {
                    name_key_inserts: calls,
                    ..PlannedInserts::default()
                },
                RebuildTable::ExtCounts => PlannedInserts {
                    external_rows: calls,
                    ..PlannedInserts::default()
                },
            };
            let (state_before, index_before) = entry.weights_now();
            let growth_before =
                estimate_in_place_growth_bytes(&entry.state, entry.plan.as_ref(), inserts);

            entry
                .plan
                .as_mut()
                .expect("indexes")
                .note_capacity(table, raised);
            let (state_after, index_after) = entry.weights_now();
            let growth_after =
                estimate_in_place_growth_bytes(&entry.state, entry.plan.as_ref(), inserts);
            let slot = table_slot_bytes(table);
            let delta = table_bytes(raised, slot) - table_bytes(capacity, slot);
            assert!(delta > 0, "fixture: {table:?} raised to more buckets");
            let expected = if table.in_state() {
                (state_before + delta, index_before)
            } else {
                (state_before, index_before + delta)
            };
            assert_eq!(
                (state_after, index_after),
                expected,
                "{table:?} is weighed at its held capacity, and by one weigher only"
            );
            let indexes = entry.plan.as_ref().expect("indexes");
            assert_eq!(indexes.weight_bytes(), indexes.weight_bytes_recomputed());
            assert!(
                growth_after > growth_before,
                "{table:?}'s growth doubles from its held capacity: {growth_after} > \
                 {growth_before}"
            );
            entry
                .plan
                .as_mut()
                .expect("indexes")
                .reset_capacity_high_water();
        }
    }

    // ── Tables thinned by tombstones (b.5h3) ─────────────────────────────────
    //
    // hashbrown erases a key to a tombstone (DELETED) or to EMPTY by its
    // neighbours: DELETED exactly when the full or deleted control bytes
    // around its slot — up to a group's width each side — number at least a
    // group's width (`RawTableInner::erase`). The group is 8 bytes wide on
    // aarch64 (where these tests are developed) and 16 on x86_64 (CI and
    // production), so a fixture of random keys can thin into tombstones on
    // one and not on the other. These fixtures place every key by hand
    // instead: each key is picked, under its map's own hasher, to have a
    // given home slot (the low bits of its hash), and keys are inserted in
    // ascending home-slot order, so each lands on its home slot (an insert
    // takes the first EMPTY or DELETED byte of the group at its home, its
    // home itself while it is free). The keys then form one contiguous run
    // of full buckets from slot 0. A run at least a group long makes every
    // erase inside it a tombstone, in any order — a tombstone keeps the run's
    // length, and EMPTY arises only inside a run shorter than a group — and
    // every run here is longer than 16, the widest group: so the same
    // erases leave the same tombstones at width 8 and 16, by construction.
    // Each test self-checks that premise on the platform it runs on: an
    // erase that left EMPTY would give the slot back to `capacity()`, so
    // `capacity()` falling by exactly the number of erases proves every
    // erase left a tombstone.

    /// Sections the home-slot fixture holds: a table reserved for 56 has 64
    /// buckets (`capacity_to_buckets`, `alloc_model::table_buckets`), so
    /// filling it leaves no room.
    const HOME_SLOT_SECTIONS: usize = 56;
    /// The mask the fixture's ids are picked under: their home slots in a
    /// 128-bucket table, 0 to 55, so in the 64-bucket table (whose mask is
    /// the low 6 of these bits) they are 0 to 55 as well, and a grown
    /// 128-bucket table places them on the same slots (the audit copy's
    /// growth, [`audit_copy_hold_covers_a_growth_priced_from_the_raised_bound`]).
    const HOME_SLOT_MASK: u64 = 127;
    /// The deletes that thin the fixture past its regime: with 29 of its 56
    /// keys erased to tombstones, `capacity()` reads 27 slots, which
    /// `table_bytes` rounds to 32 buckets, half the 64 it holds.
    const THINNING_DELETES: usize = 29;

    /// For each home slot in `slots` under `hasher` and `mask`, the first id
    /// from `first` up whose hash lands on it; in slot order.
    fn ids_by_home_slot(
        hasher: &std::collections::hash_map::RandomState,
        mask: u64,
        slots: std::ops::Range<usize>,
        first: u64,
    ) -> Vec<u64> {
        use std::hash::BuildHasher;
        let mut by_slot: Vec<Option<u64>> = vec![None; slots.len()];
        let mut id = first;
        while by_slot.iter().any(Option::is_none) {
            let home = (hasher.hash_one(id) & mask) as usize;
            if slots.contains(&home) && by_slot[home - slots.start].is_none() {
                by_slot[home - slots.start] = Some(id);
            }
            id += 1;
        }
        by_slot.into_iter().flatten().collect()
    }

    /// A section under the root, named for its id.
    fn root_section(section_id: u64) -> Section {
        Section {
            section_id,
            parent_id: 0,
            name: format!("h{section_id}"),
            section_type: 6,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        }
    }

    /// The home-slot fixture: [`HOME_SLOT_SECTIONS`] sections under the
    /// root, in a sections table reserved for them (64 buckets), each on its
    /// home slot 0 to 55 (module comment above): one contiguous run of 56
    /// full buckets, at no room. Returns the state and the section ids in
    /// home-slot order.
    fn home_slot_state() -> (CompactState, Vec<u64>) {
        let mut sections: HashMap<u64, Section> = HashMap::with_capacity(HOME_SLOT_SECTIONS);
        let ids = ids_by_home_slot(sections.hasher(), HOME_SLOT_MASK, 0..HOME_SLOT_SECTIONS, 1);
        for &id in &ids {
            sections.insert(id, root_section(id));
        }
        assert_eq!(
            (
                sections.len(),
                sections.capacity(),
                true_capacity_for_test(&sections)
            ),
            (HOME_SLOT_SECTIONS, HOME_SLOT_SECTIONS, HOME_SLOT_SECTIONS),
            "fixture: 56 sections fill the 64-bucket table, no room"
        );
        let mut state = CompactState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..CompactState::default()
        };
        state.sections = sections;
        state.children.insert(0, ids.clone());
        (state, ids)
    }

    /// One `DeleteSection` per id of `ids`, each under the root.
    fn deletes_of(ids: &[u64]) -> OwnedDelta {
        OwnedDelta {
            struct_commands: ids
                .iter()
                .map(|&section_id| {
                    StructCommand::DeleteSection(oodf_ffi::CommandDeleteSection {
                        parent_id: 0,
                        section_id,
                    })
                })
                .collect(),
            ..OwnedDelta::default()
        }
    }

    /// One `CreateSection` per id of `ids`, each under the root and named
    /// for its id.
    fn creates_of(ids: &[u64]) -> Vec<StructCommand> {
        ids.iter()
            .map(|&section_id| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id,
                    section_name: format!("h{section_id}"),
                    section_type: 6,
                })
            })
            .collect()
    }

    /// The oracle's true capacity of `state`'s `table`: the usable slots of
    /// the buckets it holds.
    fn true_capacity_of(state: &CompactState, table: RebuildTable) -> usize {
        crate::domain::plan_index::true_state_table_capacity_for_test(table, state)
            .expect("a state table")
    }

    /// The home-slot fixture cached as a cold start leaves it (its put-back's
    /// compaction keeps the 64 buckets: 56 sections need them all), with the
    /// shadow check `shadow`, checked out, and its last `deletes` sections by
    /// home slot deleted by a plan in place. Returns the cache, the guard
    /// holding the planned entry, the plan's mutations and the fixture's ids
    /// in home-slot order. Self-checks that every delete left a tombstone.
    fn thinned_by_a_plan(
        deletes: usize,
        shadow: bool,
    ) -> (StructuralCache, CheckedOut, Vec<StorageMutation>, Vec<u64>) {
        let cache = cache_with(BUDGET_BYTES);
        cache.set_shadow_check_for_test(shadow);
        let (state, ids) = home_slot_state();
        assert!(cold_start(&cache, compact_cut(OBJECT_ID, 5, state)));
        let mut guard = hit(&cache, OBJECT_ID);
        assert_eq!(
            (
                guard.entry().state.sections.capacity(),
                true_capacity_of(&guard.entry().state, RebuildTable::Sections)
            ),
            (HOME_SLOT_SECTIONS, HOME_SLOT_SECTIONS),
            "fixture: the cached entry keeps its 64-bucket table full"
        );
        let doomed = &ids[HOME_SLOT_SECTIONS - deletes..];
        let mutations = match plan_unbounded(&mut guard, deletes_of(doomed)) {
            Ok(BoundedPlan::Within { mutations, .. }) => mutations,
            other => panic!("fixture: the deletes plan: {other:?}"),
        };
        let state = &guard.entry().state;
        assert_eq!(
            (
                state.sections.len(),
                state.sections.capacity(),
                true_capacity_of(state, RebuildTable::Sections)
            ),
            (
                HOME_SLOT_SECTIONS - deletes,
                HOME_SLOT_SECTIONS - deletes,
                HOME_SLOT_SECTIONS
            ),
            "fixture premise on this platform: every delete left a tombstone (capacity() fell by \
             each), and the table keeps its 64 buckets"
        );
        (cache, guard, mutations, ids)
    }

    /// T5: the true-capacity oracle (`domain::types::true_capacity_for_test`)
    /// reads `capacity()` on a map with no tombstones — fresh, reserved,
    /// filled, compacted — and the whole load of the buckets it holds, 56 of
    /// 64, on the thinned fixture, whose `capacity()` reads 27; and reading it
    /// leaves the map as it was.
    #[test]
    fn true_capacity_oracle_reads_the_buckets_held_whatever_the_tombstones() {
        let empty: HashMap<u64, u64> = HashMap::new();
        assert_eq!(true_capacity_for_test(&empty), 0, "no table");
        let reserved: HashMap<u64, u64> = HashMap::with_capacity(HOME_SLOT_SECTIONS);
        assert_eq!(
            true_capacity_for_test(&reserved),
            reserved.capacity(),
            "a fresh reserved table"
        );
        let (state, ids) = home_slot_state();
        let mut sections = state.sections;
        assert_eq!(
            true_capacity_for_test(&sections),
            sections.capacity(),
            "a filled table with no tombstones"
        );
        for id in &ids[HOME_SLOT_SECTIONS - THINNING_DELETES..] {
            sections.remove(id);
        }
        assert_eq!(
            sections.capacity(),
            HOME_SLOT_SECTIONS - THINNING_DELETES,
            "fixture premise on this platform: every erase left a tombstone"
        );
        assert_eq!(
            true_capacity_for_test(&sections) as u64,
            bucket_capacity(64),
            "the thinned table: the whole load of its 64 buckets"
        );
        assert_eq!(
            table_buckets(sections.capacity() as u64),
            32,
            "which capacity() reads as 32"
        );
        assert_eq!(
            sections.capacity(),
            HOME_SLOT_SECTIONS - THINNING_DELETES,
            "the oracle reads a clone: the map keeps its tombstones"
        );
        sections.shrink_to_fit();
        assert_eq!(
            true_capacity_for_test(&sections),
            sections.capacity(),
            "compacted: no tombstones left"
        );
    }

    /// T1, the regression: a plan in place deleting 29 of the fixture's 56
    /// sections leaves its 64-bucket table with 29 tombstones, so its
    /// `capacity()` reads 27 slots, 32 buckets. The entry's O(1) weight — what
    /// CR2 and the plan's allowance read — is still at or above the bytes its
    /// tables' buckets hold, the whole-state estimate at their true
    /// capacities (here equal: the high-water read the 56 slots before the
    /// first delete). Weighed from `capacity()`, the sections table would
    /// count 32 of its 64 buckets. The shadow check is off, so this
    /// assertion is the one that fails without the high-water (the shadow
    /// check's own floor is [`shadow_check_flags_an_entry_weighed_below_its_tables_buckets`]).
    #[test]
    fn plan_in_place_thinning_a_table_into_tombstones_weighs_it_at_its_buckets() {
        let (_cache, guard, _, _) = thinned_by_a_plan(THINNING_DELETES, false);
        let entry = guard.entry();
        assert_eq!(
            table_buckets(entry.state.sections.capacity() as u64),
            32,
            "fixture: capacity() reads half the 64 buckets"
        );
        let (state_weight, index_weight) = entry.weights_now();
        let state_floor = estimate_state_bytes_at_true_capacity_for_test(&entry.state);
        assert!(
            state_weight >= state_floor,
            "the planned entry's state weighs {state_weight} B, below the {state_floor} B its \
             tables' buckets hold"
        );
        let indexes = entry.plan.as_ref().expect("the entry holds its indexes");
        let index_floor = indexes.weight_bytes_at_true_capacity_for_test();
        assert!(
            index_weight >= index_floor,
            "the planned entry's indexes weigh {index_weight} B, below the {index_floor} B their \
             maps' buckets hold"
        );
    }

    /// T8: a clone of T1's planned entry (`CacheEntry::clone`, which rebuilds
    /// the indexes from the cloned state) keeps the sections table's 64
    /// buckets and its tombstones (a cloned map copies the bucket array), and
    /// is weighed at or above them: the clone carries the original's
    /// high-waters for the state's tables.
    #[test]
    fn clone_of_a_thinned_entry_weighs_its_tables_at_their_buckets() {
        let (_cache, guard, _, _) = thinned_by_a_plan(THINNING_DELETES, false);
        let cloned = guard.entry().clone();
        assert_eq!(
            (
                cloned.state.sections.capacity(),
                true_capacity_of(&cloned.state, RebuildTable::Sections)
            ),
            (HOME_SLOT_SECTIONS - THINNING_DELETES, HOME_SLOT_SECTIONS),
            "fixture: the clone keeps the 64 buckets and the tombstones"
        );
        let floor = estimate_state_bytes_at_true_capacity_for_test(&cloned.state);
        let (state_weight, _) = cloned.weights_now();
        assert!(
            state_weight >= floor,
            "the clone's state weighs {state_weight} B, below the {floor} B its tables' buckets \
             hold"
        );
    }

    /// T3, the reset's order: T1's plan committed (`advance_committed`) and
    /// put back (`put_back_held`) is compacted — 27 sections in 32 buckets —
    /// and stored at exactly the compacted state's weight, the whole-state
    /// estimate and the oracle's alike, its indexes' high-waters zeroed: the
    /// high-waters reset after the compaction and before the weigh. Reset
    /// after the weigh, the stored weight would read the sections table at
    /// the 64 buckets its uncompacted predecessor held. The shadow check is
    /// off, so the settle check (`assert_weighed`) does not pre-empt this
    /// assertion.
    #[test]
    fn put_back_of_a_thinned_plan_stores_the_compacted_weight_with_the_high_waters_reset() {
        let (cache, mut guard, mutations, _) = thinned_by_a_plan(THINNING_DELETES, false);
        guard
            .advance_committed(&mutations, 6, format!("{:040x}", 6))
            .expect("the commit settles");
        assert!(guard.put_back_held().is_stored(), "stored");
        let found = hit(&cache, OBJECT_ID);
        let stored = found.entry();
        let compacted = HOME_SLOT_SECTIONS - THINNING_DELETES;
        assert_eq!(
            (
                stored.state.sections.len(),
                table_buckets(true_capacity_of(&stored.state, RebuildTable::Sections) as u64)
            ),
            (compacted, 32),
            "fixture: the put-back compacted 27 sections into 32 buckets"
        );
        assert_eq!(
            stored.state_weight_bytes,
            estimate_state_bytes(&stored.state),
            "stored at the compacted state's estimate"
        );
        assert_eq!(
            stored.state_weight_bytes,
            estimate_state_bytes_at_true_capacity_for_test(&stored.state),
            "stored at the bytes its compacted tables hold"
        );
        let indexes = stored.plan.as_ref().expect("the entry holds its indexes");
        assert_eq!(stored.index_weight_bytes, indexes.weight_bytes());
        for table in RebuildTable::ALL {
            assert_eq!(
                indexes.capacity_high_water(table),
                0,
                "{table:?}'s high-water reset at the put-back's compaction"
            );
        }
    }

    /// T2, the growth term: with 28 of the fixture's sections deleted in
    /// place the table shows 28 slots at 28 sections (no room) while it
    /// holds 64 buckets with 28 tombstones. One more section makes hashbrown
    /// grow it to `max(29, 56 + 1)` slots, 128 buckets (measured on a clone
    /// by the oracle); CR1's growth term for that one insert
    /// (`estimate_in_place_growth_bytes`) prices at least that table.
    /// Doubled from `capacity()`'s 32 buckets it would price 64, the table
    /// the entry already holds.
    #[test]
    fn in_place_growth_of_a_thinned_table_prices_the_table_std_grows_to() {
        const DELETES: usize = 28;
        let (_cache, guard, _, ids) = thinned_by_a_plan(DELETES, false);
        let entry = guard.entry();
        let sections = &entry.state.sections;
        assert_eq!(
            (sections.len(), sections.capacity()),
            (HOME_SLOT_SECTIONS - DELETES, HOME_SLOT_SECTIONS - DELETES),
            "fixture: no room"
        );
        let mut grown = sections.clone();
        let fresh = ids.iter().max().expect("ids") + 1;
        grown.insert(fresh, root_section(fresh));
        let grown_to = true_capacity_for_test(&grown);
        assert_eq!(
            table_buckets(grown_to as u64),
            128,
            "fixture premise on this platform: one insert grows the thinned table to 128 buckets"
        );
        let priced = estimate_in_place_growth_bytes(
            &entry.state,
            entry.plan.as_ref(),
            PlannedInserts {
                sections: 1,
                ..PlannedInserts::default()
            },
        );
        let real = table_bytes(grown_to, SECTION_SLOT_BYTES);
        assert!(
            priced >= real,
            "one section insert is priced {priced} B, below the {real} B table std grows to"
        );
    }

    /// T4, the repair's arm: the same 29 deletes replayed through the indexed
    /// replay (`replay_stream_record_indexed`, what a repair round runs on a
    /// cached entry) leave the same tombstones, and `indexed_state_bytes` —
    /// RP2's and CR1's held term — stays at or above the bytes the tables'
    /// buckets hold.
    #[test]
    fn indexed_replay_thinning_a_table_into_tombstones_weighs_it_at_its_buckets() {
        let (mut state, ids) = home_slot_state();
        let mut indexes = PlanIndexes::build(&state);
        let delta = deletes_of(&ids[HOME_SLOT_SECTIONS - THINNING_DELETES..]);
        let ompf = crate::domain::ompf::OmpfContainer::serialize(
            &oodf_ffi::build_delta(&delta).expect("build_delta"),
            &[],
        )
        .expect("serialize OMPF");
        let seq = state.sequence_num + 1;
        let mut touches = PlanTouches::default();
        crate::domain::replay::replay_stream_record_indexed(
            &mut state,
            Some(&ompf),
            seq,
            &mut crate::domain::delta::MutationSink::Discard,
            &mut indexes,
            &mut touches,
        )
        .expect("the indexed replay applies");
        crate::domain::plan_index::normalize_touched(&mut state, &mut indexes, &touches.lists);
        assert_eq!(
            (
                state.sections.capacity(),
                true_capacity_of(&state, RebuildTable::Sections)
            ),
            (HOME_SLOT_SECTIONS - THINNING_DELETES, HOME_SLOT_SECTIONS),
            "fixture premise on this platform: every replayed delete left a tombstone"
        );
        let weight = indexed_state_bytes(&state, &indexes);
        let floor = estimate_state_bytes_at_true_capacity_for_test(&state);
        assert!(
            weight >= floor,
            "the replayed state weighs {weight} B, below the {floor} B its tables' buckets hold"
        );
    }

    /// T9: the shadow check's weight floor (`assert_weighed_at_or_above_its_buckets`,
    /// run after every plan in place and repair round with the shadow check
    /// on) passes on T1's thinned plan, and fires on an entry weighed below
    /// its buckets — here the same entry with its high-waters zeroed, as a
    /// removal that skipped its sample would leave it — naming the table.
    /// Its two wiring points, after `CheckedOut::plan_in_place` and after
    /// `CheckedOut::mutate`, are not pinned: with the weighers correct no
    /// entry reaches either point weighed below its buckets, so removing a
    /// call would fail nothing without a fault seam that corrupts the weight
    /// between the mutation and the check.
    #[test]
    fn shadow_check_flags_an_entry_weighed_below_its_tables_buckets() {
        let (_cache, guard, _, _) = thinned_by_a_plan(THINNING_DELETES, true);
        let mut unsampled = guard.entry().clone();
        unsampled
            .plan
            .as_mut()
            .expect("the entry holds its indexes")
            .reset_capacity_high_water();
        let flagged = catch_unwind(AssertUnwindSafe(|| {
            assert_weighed_at_or_above_its_buckets(&unsampled, "a lost sample");
        }))
        .expect_err("the floor fires on an entry weighed below its buckets");
        let message = flagged
            .downcast_ref::<String>()
            .expect("a formatted panic message");
        assert!(
            message.contains("shadow check: after a lost sample, the Sections table"),
            "{message}"
        );
    }

    // ── The sampled audit copy's bucket bound (b.5h3) ───────────────────────

    /// The sampled audit's copy of `entry`, taken as the audit takes it — the
    /// capacities the entry weighs its tables at read first
    /// (`CacheEntry::held_state_capacities`), then the copy without its plan
    /// indexes (`CacheEntry::copy_without_plan`) — and its bucket bound
    /// started there (`CopyBucketBound::new`).
    fn audit_copy(entry: &CacheEntry) -> (CacheEntry, CopyBucketBound) {
        let start = entry.held_state_capacities();
        let copy = entry.copy_without_plan();
        let bound = CopyBucketBound::new(&copy.state, start);
        (copy, bound)
    }

    /// One record of `delta` on the audit copy's `state`, as the repair
    /// replays a copy's record: priced from `bound`
    /// (`estimate_replay_record_growth_bounded`), decoded and applied with no
    /// hooks, its children lists normalized, and the bound raised from what
    /// it priced (`CopyBucketBound::applied`). Answers the record's estimate
    /// and what it priced.
    fn replay_on_copy(
        state: &mut CompactState,
        bound: &mut CopyBucketBound,
        delta: &OwnedDelta,
    ) -> (ReplayRecordGrowth, PricedBuckets) {
        let (estimate, priced) =
            estimate_replay_record_growth_bounded(state, bound, Some(delta), false);
        let ompf = crate::domain::ompf::OmpfContainer::serialize(
            &oodf_ffi::build_delta(delta).expect("build_delta"),
            &[],
        )
        .expect("serialize OMPF");
        let record = crate::domain::replay::decode_stream_record(
            state.sequence_num,
            Some(&ompf),
            state.sequence_num + 1,
        )
        .expect("the record decodes");
        crate::domain::replay::apply_decoded_record_into(
            state,
            record,
            &mut crate::domain::delta::MutationSink::Discard,
        )
        .expect("the record applies");
        crate::domain::replay::normalize_children(state);
        bound.applied(&priced, state);
        (estimate, priced)
    }

    /// The copy's bucket-bound invariant (`CopyBucketBound`), per state
    /// table, against the oracle: the bound `b` is at least the buckets `B`
    /// the table holds, and at `B = b` the tombstone bound `t` is at least
    /// the tombstones it holds (`full(B) − capacity()`).
    fn assert_copy_bound(state: &CompactState, bound: &CopyBucketBound, context: &str) {
        for (table, capacities) in crate::domain::plan_index::state_hash_tables_for_test(state) {
            let held = table_buckets(capacities.true_capacity as u64);
            let b = table_buckets(bound.capacities().get(table) as u64);
            assert!(
                held <= b,
                "{context}: the {table:?} table holds {held} buckets, above its bound {b}"
            );
            if held == b {
                let tombstones = capacities.true_capacity - capacities.capacity;
                let t = bound.tombstones().get(table);
                assert!(
                    tombstones <= t,
                    "{context}: the {table:?} table, at its bound of {b} buckets, holds \
                     {tombstones} tombstones, above its tombstone bound {t}"
                );
            }
        }
    }

    /// The bytes the state's hash tables grew by from `before` to `after`,
    /// by the oracle: each table whose true bucket count rose, at its new
    /// table (one doubling per table here; its old table is in the holder's
    /// weight).
    fn tables_grown(before: &CompactState, after: &CompactState) -> u64 {
        crate::domain::plan_index::state_hash_tables_for_test(after)
            .into_iter()
            .filter(|&(table, capacities)| {
                table_buckets(capacities.true_capacity as u64)
                    > table_buckets(true_capacity_of(before, table) as u64)
            })
            .map(|(table, capacities)| {
                table_bytes(capacities.true_capacity, table_slot_bytes(table))
            })
            .sum()
    }

    /// T11, the bound's raise: the audit's copy of T1's committed successor
    /// (its sections table 64 buckets, 29 tombstones, `capacity()` 27) starts
    /// its bound at the held 64 buckets. Record 1 creates 30 sections — the
    /// first insert rehashes the table in place, the 30th grows it to 128
    /// buckets — then deletes 56 of the 57, leaving 56 tombstones that its
    /// `capacity()` (56) reads as 64 buckets; the bound is raised to the 128
    /// the record's growth was priced to. Record 2 creates 56 sections into
    /// empty slots, the 56th growing the table to 256 buckets; priced from
    /// the raised bound, the record's growth covers that table. A bound never
    /// raised past its start would price record 2 from 64 buckets, a growth
    /// to 128, below the 256-bucket table std allocates.
    ///
    /// Record 1 sits exactly on the raise's capacity test: its removals, tight
    /// (`PricedBuckets::removals_after`), are 27 + 30 − 1 = 56, and the
    /// `capacity()` it leaves, 56, is exactly `full(128) − 56` = 112 − 56, at
    /// width 16 as at 8: the 57 rows the resize placed sit in one run from
    /// slot 0, longer than 16, so each of the 56 deletes leaves a tombstone,
    /// and the 55 slots of growth the resize left (112 − 57) do not depend on
    /// the width; `capacity()` is the 1 row left plus those 55. So
    /// the test's `≥` is pinned here too: a strict `>` keeps the bound at 64
    /// buckets below the 128 the table holds after record 1 (the bound check
    /// after record 1 fails), and record 2 would be priced to 128 buckets
    /// instead of 256. Every id is placed
    /// by home slot under the copy's hasher in the 128-bucket table
    /// ([`HOME_SLOT_MASK`]), so the tombstones are the same at width 8 and 16;
    /// the copy's children table is reserved past the records' insert calls,
    /// so the records price only the sections table's growth.
    #[test]
    fn audit_copy_hold_covers_a_growth_priced_from_the_raised_bound() {
        let (_cache, mut guard, mutations, ids) = thinned_by_a_plan(THINNING_DELETES, false);
        guard
            .advance_committed(&mutations, 6, format!("{:040x}", 6))
            .expect("the commit settles");
        let start = guard.entry().held_state_capacities();
        let mut copy = guard.entry().copy_without_plan();
        drop(guard);
        copy.state.children.reserve(256);
        let mut bound = CopyBucketBound::new(&copy.state, start);
        let state = &mut copy.state;
        assert_eq!(
            bound.capacities().get(RebuildTable::Sections),
            HOME_SLOT_SECTIONS,
            "fixture: the bound starts at the held 64 buckets"
        );
        assert_copy_bound(state, &bound, "the copy, before any record");

        // Record 1: 30 creates on home slots 27 to 56 (the survivors hold 0
        // to 26), then 56 deletes, all but the section on slot 0.
        let hasher = state.sections.hasher().clone();
        let created = ids_by_home_slot(&hasher, HOME_SLOT_MASK, 27..57, 1_000_000);
        let survivors = &ids[..HOME_SLOT_SECTIONS - THINNING_DELETES];
        let doomed: Vec<u64> = survivors[1..].iter().chain(&created).copied().collect();
        let first = OwnedDelta {
            struct_commands: creates_of(&created)
                .into_iter()
                .chain(deletes_of(&doomed).struct_commands)
                .collect(),
            ..OwnedDelta::default()
        };
        let before = state.clone();
        let (estimate, _) = replay_on_copy(state, &mut bound, &first);
        assert_eq!(
            (
                state.sections.len(),
                state.sections.capacity(),
                true_capacity_of(state, RebuildTable::Sections)
            ),
            (1, 56, 112),
            "fixture premise on this platform: grown to 128 buckets, then 56 tombstones"
        );
        let grown = tables_grown(&before, state);
        assert!(
            estimate.growth >= grown,
            "record 1: priced {} B, below the {grown} B its tables grew by",
            estimate.growth
        );
        assert_copy_bound(state, &bound, "after record 1");

        // Record 2: 55 creates into the empty slots 57 to 111, then one more,
        // which grows the table.
        let mut created = ids_by_home_slot(&hasher, HOME_SLOT_MASK, 57..112, 2_000_000);
        created.push(3_000_000);
        let second = OwnedDelta {
            struct_commands: creates_of(&created),
            ..OwnedDelta::default()
        };
        let before = state.clone();
        let (estimate, priced) = replay_on_copy(state, &mut bound, &second);
        assert_eq!(
            table_buckets(true_capacity_of(state, RebuildTable::Sections) as u64),
            256,
            "fixture premise on this platform: record 2 grew the table to 256 buckets"
        );
        assert_eq!(
            priced.grown_buckets(RebuildTable::Sections),
            Some(256),
            "record 2's growth priced from the raised bound"
        );
        let grown = tables_grown(&before, state);
        assert!(
            estimate.growth >= grown,
            "record 2: priced {} B, below the {grown} B its tables grew by",
            estimate.growth
        );
        assert_copy_bound(state, &bound, "after record 2");
    }

    /// Field names the no-compounding fixture interns: 7, so the interner's
    /// name → id table is full at 8 buckets.
    const INTERNED_NAMES: usize = 7;

    /// T10, no compounding: on the audit's copy of an entry whose interner
    /// table is full (7 names at 8 buckets), 256 records each overwrite 8
    /// existing fields under names already interned. Each record prices a
    /// growth of the interner table (8 insert calls on no room) that never
    /// happens — the names are found, never inserted — and removes nothing.
    /// The bound moves at most once, after the first record (the field-row
    /// table, full too, does grow on its first overwrite: `insert` reserves
    /// before its lookup), and then stays flat for the other 255. Raised to
    /// each record's priced growth instead, the interner's bound would
    /// double on every record.
    #[test]
    fn audit_copy_bound_stays_flat_over_records_pricing_growth_that_never_happens() {
        const RECORDS: usize = 256;
        const FIELDS_PER_RECORD: usize = 8;
        let mut state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..ObjectState::default()
        };
        let sections: Vec<u64> = (1..=8).collect();
        for &id in &sections {
            state.sections.insert(id, root_section(id));
            for n in 0..INTERNED_NAMES {
                let key_name = format!("n{n}");
                state.fields.insert(
                    (id, key_name.clone()),
                    Field {
                        section_id: id,
                        key_name,
                        value: FieldValue::Inline(vec![1]),
                        set_order: 1,
                        authoring_set_order: 1,
                        version: 1,
                    },
                );
            }
        }
        state.children.insert(0, sections.clone());
        let mut compacted = compact(&state);
        compact_state(&mut compacted);
        let names = compacted.fields.names();
        assert_eq!(
            (names.len(), names.ids_capacity()),
            (INTERNED_NAMES, INTERNED_NAMES),
            "fixture: the interner table full"
        );
        let entry = compact_cut(OBJECT_ID, 1, compacted);
        let (mut copy, mut bound) = audit_copy(&entry);
        let state = &mut copy.state;
        let interner_buckets = true_capacity_of(state, RebuildTable::NameIds);

        let mut after_first = None;
        for k in 0..RECORDS {
            let delta = OwnedDelta {
                set_fields: (0..FIELDS_PER_RECORD)
                    .map(|i| oodf_ffi::SetField {
                        section_id: sections[(k + i) % sections.len()],
                        key_name: format!("n{}", (k + i) % INTERNED_NAMES),
                        value_or_ext_hash: vec![(k % 251) as u8],
                        ext_value_size: 0,
                        ext_value_index: 0,
                        set_order: 1,
                    })
                    .collect(),
                ..OwnedDelta::default()
            };
            let (_, priced) = replay_on_copy(state, &mut bound, &delta);
            assert!(
                priced.grown_buckets(RebuildTable::NameIds).is_some(),
                "fixture: record {k} prices an interner growth"
            );
            assert_eq!(
                true_capacity_of(state, RebuildTable::NameIds),
                interner_buckets,
                "fixture: the interner table never grows (record {k})"
            );
            assert_copy_bound(state, &bound, &format!("after record {k}"));
            match after_first {
                None => after_first = Some(bound.capacities()),
                Some(flat) => assert_eq!(
                    bound.capacities(),
                    flat,
                    "record {k}: the bound moved after the first record"
                ),
            }
        }
    }

    /// The bound's tombstone count for a record that removes from a table
    /// only through the per-record children normalization: on the audit's
    /// copy, a `DiffChangeSection` re-parents the only child of one of 49
    /// parents to the root, so the normalization drops that parent's emptied
    /// list. The children table — the root's list and the 49 parents', 50
    /// keys on home slots 0 to 49 of 64 buckets, 6 slots of room (module
    /// comment above, the run long enough at either width) — keeps its
    /// buckets and gains a tombstone. The record cannot cascade and inserts
    /// within the room, so the bound keeps its 64 buckets, and its tombstone
    /// bound must count the normalization's removal: at `B = b`, `t ≥ T`.
    /// Counting no removal for the normalization leaves `t` at 0.
    #[test]
    fn audit_copy_bound_counts_the_tombstone_a_normalized_list_leaves() {
        let (state, parents) = children_run_state();
        let entry = compact_cut(OBJECT_ID, 1, state);
        let (mut copy, mut bound) = audit_copy(&entry);
        let (_, priced) = replay_on_copy(&mut copy.state, &mut bound, &reparent_to_root(0));
        let children = &copy.state.children;
        assert!(
            !children.contains_key(&parents[0]),
            "fixture: the normalization dropped the emptied list"
        );
        assert_eq!(
            (
                children.len(),
                true_capacity_of(&copy.state, RebuildTable::Children) - children.capacity()
            ),
            (RUN_PARENTS, 1),
            "fixture premise on this platform: the dropped list left a tombstone"
        );
        assert_eq!(
            priced.grown_buckets(RebuildTable::Children),
            None,
            "fixture: the record's insert calls fit the room"
        );
        assert_copy_bound(&copy.state, &bound, "after the re-parent");
    }

    /// The children-run fixture's parents under the root, each listing one
    /// child.
    const RUN_PARENTS: usize = 49;
    /// The first id of the children-run fixture's children: parent `k`'s
    /// (in home-slot order) is `RUN_CHILD_FIRST + k`.
    const RUN_CHILD_FIRST: u64 = 1_000_000;

    /// One `DiffChangeSection` re-parenting the children-run fixture's child
    /// `k` to the root, renamed `moved`: its old parent's list empties.
    fn reparent_to_root(k: u64) -> OwnedDelta {
        OwnedDelta {
            struct_commands: vec![StructCommand::DiffChangeSection(
                oodf_ffi::CommandDiffChangeSection {
                    parent_id: 0,
                    section_id: RUN_CHILD_FIRST + k,
                    section_name: "moved".to_owned(),
                    section_type: 6,
                    section_order: 0,
                },
            )],
            ..OwnedDelta::default()
        }
    }

    /// The children-run fixture: [`RUN_PARENTS`] parents under the root, each
    /// listing one child (`x`), and a children table reserved for 56 (64
    /// buckets) holding the root's list and the parents' — 50 keys on home
    /// slots 0 to 49 (module comment above): a run long enough that every
    /// erase in it leaves a tombstone at either width, with 6 slots of room.
    /// Returns the state and the parents in home-slot order.
    fn children_run_state() -> (CompactState, Vec<u64>) {
        use std::hash::BuildHasher;
        const PARENTS: usize = RUN_PARENTS;
        const CHILD_FIRST: u64 = RUN_CHILD_FIRST;
        // A children table reserved for 56 (64 buckets) whose hasher puts
        // the root's key on a home slot below 50, so the 50 keys can fill
        // home slots 0 to 49.
        let (mut children, root_home) = loop {
            let map: HashMap<u64, Vec<u64>> = HashMap::with_capacity(HOME_SLOT_SECTIONS);
            let home = (map.hasher().hash_one(0u64) & 63) as usize;
            if home <= PARENTS {
                break (map, home);
            }
        };
        let hasher = children.hasher().clone();
        let mut slots = ids_by_home_slot(&hasher, 63, 0..PARENTS + 1, 1);
        slots[root_home] = 0;
        let parents: Vec<u64> = slots.iter().copied().filter(|&id| id != 0).collect();
        assert_eq!(
            parents.len(),
            PARENTS,
            "fixture: the root's slot taken from a parent"
        );
        let mut sections = HashMap::new();
        for (k, &parent) in parents.iter().enumerate() {
            let child = CHILD_FIRST + k as u64;
            sections.insert(parent, root_section(parent));
            sections.insert(
                child,
                Section {
                    section_id: child,
                    parent_id: parent,
                    name: "x".to_owned(),
                    section_type: 6,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
        }
        for &key in &slots {
            let list = if key == 0 {
                parents.clone()
            } else {
                let k = parents.iter().position(|&p| p == key).expect("a parent");
                vec![CHILD_FIRST + k as u64]
            };
            children.insert(key, list);
        }
        let mut state = CompactState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..CompactState::default()
        };
        state.sections = sections;
        state.children = children;
        assert_eq!(
            (
                state.children.len(),
                state.children.capacity(),
                true_capacity_of(&state, RebuildTable::Children)
            ),
            (PARENTS + 1, HOME_SLOT_SECTIONS, HOME_SLOT_SECTIONS),
            "fixture: 50 lists in 64 buckets"
        );
        (state, parents)
    }

    // ── A stream of cascading records on the audit copy (b.5h3) ─────────────

    /// Sections in the cascade stream's layer, all under the root.
    const STREAM_SECTIONS: u64 = 2_000;
    /// The section the stream's records create their sections under, so the
    /// root's long list only shrinks and the stream's listing grows by
    /// `Vec` doubling, its growth priced only on the records that double it.
    const STREAM_PARENT: u64 = 900_000;
    /// Field names the stream's layer interns: 224, the load limit of a
    /// 256-bucket table, so the interner's name → id table has no room.
    const STREAM_NAMES: u64 = 224;
    /// The most records the stream replays before it must have reached the
    /// cap.
    const STREAM_RECORD_LIMIT: u64 = 2_000;
    /// The fewest records the stream must replay before the audit's charge
    /// passes its cap. With the bound frozen, the charge grows per record by
    /// what that record prices — the interner's phantom doubling, its id →
    /// name vector's, the children table's, and the new sections' listing as
    /// it doubles — about 27 KB, against about 7.2 MB of headroom past the
    /// per-record walk and indexes, so the cap is reached after 266 records
    /// (measured on aarch64; the count does not depend on the group width,
    /// see the test's docs). Measured on the same stream with one part of
    /// the raise rule taken out, the interner's bound rises again and again
    /// and the cap is reached sooner: after 173 records with the removals
    /// left loose, 139 with no capacity test.
    /// 200 sits above every count without the fix and below the count with
    /// it: a return of the bound's compounding fails here, while a modest
    /// recalibration of the per-record terms does not.
    const STREAM_RECORD_FLOOR: u64 = 200;

    /// A stream of cascading records on the audit's copy, and the named pin
    /// of both parts of the bound's raise rule (its tight removals and its
    /// capacity test): on the audit's copy
    /// of a 2,000-section layer whose interner table is at its load limit
    /// (224 names in 256 buckets), each record deletes one section and
    /// creates one, under [`STREAM_PARENT`], with one field on a name
    /// already interned. Every record
    /// can cascade (it deletes), so its estimate's removals are every key of
    /// every table, and it prices a doubling of the interner table (one insert
    /// call on no room) that never happens: the name is found, not inserted.
    /// After every record the interner's bound has not moved — its removals,
    /// tight (`PricedBuckets::removals_after`), are 1, and a table grown past
    /// 256 buckets would read at least `full(512) − 1` = 447 slots, not the 224
    /// it reads — and every table's bound holds its invariant against the
    /// oracle. Loose removals (224 + 1) or no capacity test would raise the
    /// interner's bound at the first record.
    ///
    /// The charge the audit's copy holds at each record's RP2 is priced as
    /// the repair prices it (`ObjectStateNeed::cached_repair_record`, less
    /// the copy's own weight `W0`, as `UploadBudgetAuditCharge::target`), its
    /// reply terms one entry of the record's own OMPF. The stream counts the
    /// records until that passes `SAMPLED_AUDIT_CHARGE_CAP_BYTES`, prints the
    /// count and asserts it is at least [`STREAM_RECORD_FLOOR`]. The count
    /// does not depend on the hash group's width: the tables priced a growth
    /// — the interner, which loses no name, and the children table, which
    /// loses no list — hold no tombstone, and the sections and field-row
    /// tables, the only ones a record erases from, keep room for every
    /// record's insert whichever of their erases leave tombstones
    /// (self-checked per record), so no width-dependent reading enters the
    /// price.
    #[test]
    fn audit_copy_bound_takes_no_phantom_raise_on_a_cascade_stream_and_counts_records_to_the_cap() {
        use crate::grpc::upload_admission::{
            object_state_charge_for_test, SAMPLED_AUDIT_CHARGE_CAP_BYTES,
        };
        use crate::storage::commit_charge::{ObjectStateNeed, RepairTransient};

        let mut layer = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..ObjectState::default()
        };
        layer
            .sections
            .insert(STREAM_PARENT, root_section(STREAM_PARENT));
        for id in 1..=STREAM_SECTIONS {
            layer.sections.insert(id, root_section(id));
            let key_name = format!("n{}", id % STREAM_NAMES);
            layer.fields.insert(
                (id, key_name.clone()),
                Field {
                    section_id: id,
                    key_name,
                    value: FieldValue::Inline(vec![1]),
                    set_order: 1,
                    authoring_set_order: 1,
                    version: 1,
                },
            );
        }
        layer.children.insert(
            0,
            (1..=STREAM_SECTIONS)
                .chain([STREAM_PARENT])
                .collect::<Vec<_>>(),
        );
        let mut compacted = compact(&layer);
        compact_state(&mut compacted);
        let names = compacted.fields.names();
        assert_eq!(
            (names.len(), names.ids_capacity()),
            (STREAM_NAMES as usize, STREAM_NAMES as usize),
            "fixture: the interner table at its load limit"
        );
        let entry = compact_cut(OBJECT_ID, 1, compacted);
        let w0 = entry.state_weight_bytes;
        let (mut copy, mut bound) = audit_copy(&entry);
        let state = &mut copy.state;
        let interner_bound = bound.capacities().get(RebuildTable::NameIds);
        let interner_capacity = true_capacity_of(state, RebuildTable::NameIds);

        let (mut grown, mut replayed) = (0u64, 0u64);
        let mut reached = None;
        for k in 0..STREAM_RECORD_LIMIT {
            let born = 1_000_000 + k;
            let delta = OwnedDelta {
                struct_commands: deletes_of(&[1 + k])
                    .struct_commands
                    .into_iter()
                    .chain([StructCommand::CreateSection(CommandCreateSection {
                        parent_id: STREAM_PARENT,
                        section_id: born,
                        section_name: format!("h{born}"),
                        section_type: 6,
                    })])
                    .collect(),
                set_fields: vec![oodf_ffi::SetField {
                    section_id: born,
                    key_name: format!("n{}", k % STREAM_NAMES),
                    value_or_ext_hash: vec![2],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 1,
                }],
                ..OwnedDelta::default()
            };
            let delta_bytes = oodf_ffi::build_delta(&delta).expect("build_delta");
            let ompf_bytes =
                crate::domain::ompf::OmpfContainer::serialize(&delta_bytes, &[]).expect("OMPF");
            let (estimate, priced) = replay_on_copy(state, &mut bound, &delta);
            let delta_bytes = delta_bytes.len() as u64;
            let need = ObjectStateNeed::cached_repair_record(
                w0 + grown,
                estimate.growth + estimate.scratch,
                estimate.rewrite,
                RepairTransient {
                    reply_entry_count: 1,
                    reply_bytes: ompf_bytes.len() as u64,
                    parse_delta_bytes: delta_bytes,
                    replay_delta_bytes: replayed + delta_bytes,
                },
            );
            let charge = object_state_charge_for_test(&need).saturating_sub(w0);
            if charge > SAMPLED_AUDIT_CHARGE_CAP_BYTES {
                reached = Some((k, charge));
                break;
            }
            assert!(
                priced.grown_buckets(RebuildTable::NameIds).is_some(),
                "fixture: record {k} prices an interner growth"
            );
            for table in [RebuildTable::Sections, RebuildTable::FieldRows] {
                assert_eq!(
                    priced.grown_buckets(table),
                    None,
                    "fixture: record {k}'s insert fits the {table:?} table's room"
                );
            }
            assert_eq!(
                true_capacity_of(state, RebuildTable::NameIds),
                interner_capacity,
                "fixture: the interner table never grows (record {k})"
            );
            assert_eq!(
                bound.capacities().get(RebuildTable::NameIds),
                interner_bound,
                "record {k}: the interner's bound moved on a growth that never happened"
            );
            assert_copy_bound(state, &bound, &format!("after record {k}"));
            grown += estimate.growth;
            replayed += delta_bytes;
        }
        let width = if cfg!(target_arch = "x86_64") {
            "16 (x86_64)"
        } else if cfg!(target_arch = "aarch64") {
            "8 (aarch64)"
        } else {
            "unknown"
        };
        let (records, charge) = reached.unwrap_or_else(|| {
            panic!("the charge stays under the cap for {STREAM_RECORD_LIMIT} records")
        });
        eprintln!(
            "CASCADE-STREAM width={width} records_before_the_cap={records} charge_at_refusal={charge} \
             cap={SAMPLED_AUDIT_CHARGE_CAP_BYTES} growth_held={grown}"
        );
        assert!(
            records >= STREAM_RECORD_FLOOR,
            "the copy's charge passed the cap after {records} records, before the floor of \
             {STREAM_RECORD_FLOOR}"
        );
    }

    /// Sections in the delete-and-rename fixture: the load limit of 128
    /// buckets.
    const SPARSE_RUN_SECTIONS: usize = 112;

    /// The bound's raise in the safe direction: a record that deletes most of
    /// a full table and renames rows it keeps raises the table's bound past
    /// the buckets the table keeps. The audit's copy holds 112 sections at
    /// the load limit of 128 buckets, each on its home slot, in 16 runs of 7
    /// full buckets each followed by one empty bucket (ids picked by home
    /// slot under the copy's hasher, every slot `s` with `s % 8 != 7`,
    /// inserted in slot order). One record deletes 100 sections, then renames
    /// 10 of the 12 it keeps twice each (`DiffChangeSection`): 20 insert
    /// calls on the sections table that add no key.
    ///
    /// Every erase leaves an EMPTY bucket, not a tombstone: a run shorter
    /// than a hash group (7 < 8, the narrowest) leaves EMPTY wherever it is
    /// erased, and an erase only shortens a run — so the same at width 8 and
    /// 16, by construction. An EMPTY erase gives its slot back to
    /// `capacity()`, which stays at 112 while the table keeps 12 rows, and the
    /// renames grow nothing. But the record's removals, tight, are 112 + 20 −
    /// 12 = 120, so a table grown past 128 buckets could read as few as
    /// `full(256) − 120` = 104 slots, and 112 passes that test: the record
    /// priced a growth (132 wanted over 112 slots), so the bound rises to 256
    /// buckets though the table stays at 128. The over-read is the safe
    /// direction — the bound stays at or above the buckets the table holds —
    /// and costs the audit only a larger price for later records. It happens
    /// once: a second record of the same kind, priced a growth past the
    /// raised bound, fails the capacity test against it and leaves the bound
    /// at 256.
    #[test]
    fn audit_copy_bound_rises_on_a_delete_and_rename_record_whose_erases_leave_no_tombstone() {
        let mut sections: HashMap<u64, Section> = HashMap::with_capacity(SPARSE_RUN_SECTIONS);
        let ids: Vec<u64> = ids_by_home_slot(sections.hasher(), HOME_SLOT_MASK, 0..128, 1)
            .into_iter()
            .enumerate()
            .filter(|&(slot, _)| slot % 8 != 7)
            .map(|(_, id)| id)
            .collect();
        for &id in &ids {
            sections.insert(id, root_section(id));
        }
        let mut state = CompactState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..CompactState::default()
        };
        state.sections = sections;
        state.children.insert(0, ids.clone());
        assert_eq!(
            (
                state.sections.len(),
                state.sections.capacity(),
                true_capacity_of(&state, RebuildTable::Sections)
            ),
            (
                SPARSE_RUN_SECTIONS,
                SPARSE_RUN_SECTIONS,
                SPARSE_RUN_SECTIONS
            ),
            "fixture: 112 sections fill the 128-bucket table"
        );
        let entry = compact_cut(OBJECT_ID, 1, state);
        let (mut copy, mut bound) = audit_copy(&entry);
        assert_eq!(
            table_buckets(bound.capacities().get(RebuildTable::Sections) as u64),
            128,
            "fixture: the bound starts at the table's 128 buckets"
        );
        let (doomed, kept) = ids.split_at(100);
        let rename = |section_id: u64, round: u32| {
            StructCommand::DiffChangeSection(oodf_ffi::CommandDiffChangeSection {
                parent_id: 0,
                section_id,
                section_name: format!("renamed{section_id}-{round}"),
                section_type: 6,
                section_order: 0,
            })
        };
        let record = OwnedDelta {
            struct_commands: deletes_of(doomed)
                .struct_commands
                .into_iter()
                .chain(
                    (1..=2).flat_map(|round| kept[..10].iter().map(move |&id| rename(id, round))),
                )
                .collect(),
            ..OwnedDelta::default()
        };
        let (_, priced) = replay_on_copy(&mut copy.state, &mut bound, &record);
        let sections = &copy.state.sections;
        assert_eq!(
            (
                sections.len(),
                sections.capacity(),
                true_capacity_of(&copy.state, RebuildTable::Sections)
            ),
            (12, SPARSE_RUN_SECTIONS, SPARSE_RUN_SECTIONS),
            "fixture premise on this platform: every erase left EMPTY, and the table kept its \
             128 buckets"
        );
        assert_eq!(
            priced.grown_buckets(RebuildTable::Sections),
            Some(256),
            "fixture: the record priced a growth of the sections table"
        );
        assert_eq!(
            table_buckets(bound.capacities().get(RebuildTable::Sections) as u64),
            256,
            "the bound rose to 256 buckets, above the 128 the table keeps"
        );
        assert_copy_bound(&copy.state, &bound, "after the record");

        // A second record of the same kind: deletes the 2 rows not renamed,
        // then renames the other 10, 101 times in all: enough calls to price a growth past the
        // raised bound (12 + 101 = 113 wanted over 112 slots, priced to 512
        // buckets). Its removals, tight, are 2 + 101 = 103, so a table grown
        // past 256 buckets would read at least `full(512) − 103` = 345 slots;
        // it reads 112, so the bound stays.
        let (renamed, spare) = kept.split_at(10);
        let second = OwnedDelta {
            struct_commands: deletes_of(spare)
                .struct_commands
                .into_iter()
                .chain((3..=103).map(|round| rename(renamed[round as usize % 10], round)))
                .collect(),
            ..OwnedDelta::default()
        };
        let (_, priced) = replay_on_copy(&mut copy.state, &mut bound, &second);
        let sections = &copy.state.sections;
        assert_eq!(
            (
                sections.len(),
                sections.capacity(),
                true_capacity_of(&copy.state, RebuildTable::Sections)
            ),
            (10, SPARSE_RUN_SECTIONS, SPARSE_RUN_SECTIONS),
            "fixture premise on this platform: the second record's erases left EMPTY too, and \
             the table kept its 128 buckets"
        );
        assert_eq!(
            priced.grown_buckets(RebuildTable::Sections),
            Some(512),
            "fixture: the second record priced a growth past the raised bound"
        );
        assert_eq!(
            table_buckets(bound.capacities().get(RebuildTable::Sections) as u64),
            256,
            "the second record did not raise the bound again"
        );
        assert_copy_bound(&copy.state, &bound, "after the second record");
    }

    // ── Where the capacity high-water is sampled (b.5h3) ────────────────────
    //
    // One test per sampling site: on a fresh entry (its tables compacted,
    // its high-waters 0), one removal goes through the site, and the
    // high-water of the table it removed from equals the `capacity()` that
    // table read just before. Each removal leaves a tombstone, so that
    // reading is above the one after: a sample taken after the removal, or
    // none, fails. Where the fixture cannot place the removed key by home
    // slot (a map private to its store or to the plan indexes), the key is
    // chosen by running the same removal on a clone first — a clone keeps the
    // map's layout — and keeping one whose erase left a tombstone there.

    /// `table`'s `capacity()` and true capacity in `state` and `indexes`.
    fn shape_of(
        state: &CompactState,
        indexes: &PlanIndexes,
        table: RebuildTable,
    ) -> crate::domain::types::MapCapacities {
        crate::domain::plan_index::hash_tables_for_test(state, indexes)
            .into_iter()
            .find(|&(listed, _)| listed == table)
            .map(|(_, capacities)| capacities)
            .expect("a registered table")
    }

    /// A journalled plan in place of `delta` on `state` and its `indexes`, as
    /// `CheckedOut::plan_in_place` makes it, unbounded: its journal and
    /// touches, and whether it applied.
    fn plan_with_indexes(
        state: &mut CompactState,
        indexes: &mut PlanIndexes,
        delta: &OwnedDelta,
    ) -> (PlanJournal, PlanTouches, bool) {
        let mut journal = PlanJournal::new(state, indexes);
        let mut touches = PlanTouches::default();
        let result = apply_delta_bounded_indexed(
            state,
            Some(delta.clone()),
            MutationAllowance::UNBOUNDED,
            &mut |_: RewriteCount| None::<u64>,
            &mut ApplyDeltaStats::default(),
            indexes,
            &mut touches,
            Some(&mut journal),
        );
        let applied = matches!(result, Ok(BoundedPlan::Within { .. }));
        (journal, touches, applied)
    }

    /// The plan indexes a fresh cache entry holds for `state`: built and
    /// compacted, every high-water 0.
    fn fresh_indexes(state: &CompactState) -> PlanIndexes {
        let mut indexes = PlanIndexes::build(state);
        indexes.compact();
        indexes
    }

    /// `table`'s high-water after a removal through a site, against the
    /// `capacity()` it read just before (`before`), with the premise that
    /// the removal left a tombstone (`after` below `before`).
    fn assert_sampled_before(
        indexes: &PlanIndexes,
        table: RebuildTable,
        before: usize,
        after: usize,
        site: &str,
    ) {
        assert!(
            after < before,
            "fixture premise on this platform: {site}'s removal left a tombstone in {table:?} \
             (capacity() {before} before, {after} after)"
        );
        assert_eq!(
            indexes.capacity_high_water(table),
            before,
            "{site}: {table:?}'s high-water is its capacity() just before the removal"
        );
    }

    /// The writer's removal hook from `remove_list`: deleting a parent
    /// of the children-run fixture erases its list from the children table,
    /// a run on home slots 0 to 49, so a tombstone at either width.
    #[test]
    fn remove_list_samples_the_children_table_before_erasing_a_list() {
        let (mut state, parents) = children_run_state();
        let mut indexes = fresh_indexes(&state);
        let before = shape_of(&state, &indexes, RebuildTable::Children).capacity;
        let (_, _, applied) =
            plan_with_indexes(&mut state, &mut indexes, &deletes_of(&parents[..1]));
        assert!(applied, "fixture: the delete applies");
        assert!(
            !state.children.contains_key(&parents[0]),
            "fixture: the plan itself erased the deleted parent's list"
        );
        let after = shape_of(&state, &indexes, RebuildTable::Children).capacity;
        assert_sampled_before(
            &indexes,
            RebuildTable::Children,
            before,
            after,
            "remove_list",
        );
    }

    /// 56 sections under the root, each holding one inline field under a
    /// name of its own and one time sample, compacted: the field rows, the
    /// interner's name → id table and the samples each fill 64 buckets to
    /// their load limit.
    fn value_rows_state() -> CompactState {
        let mut layer = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..ObjectState::default()
        };
        let ids: Vec<u64> = (1..=HOME_SLOT_SECTIONS as u64).collect();
        for &id in &ids {
            layer.sections.insert(id, root_section(id));
            let key_name = format!("v{id}");
            layer.fields.insert(
                (id, key_name.clone()),
                Field {
                    section_id: id,
                    key_name,
                    value: FieldValue::Inline(vec![1]),
                    set_order: 1,
                    authoring_set_order: 1,
                    version: 1,
                },
            );
        }
        layer.children.insert(0, ids.clone());
        let mut state = compact(&layer);
        for &id in &ids {
            state.time_samples.insert(id, 1.0f64.to_bits(), None);
        }
        compact_state(&mut state);
        state
    }

    /// The purge's removal hooks: deleting a section purges its field
    /// row, releases its name and purges its sample, each table noted at the
    /// capacity it read as the purge began. The section is one whose three
    /// erases left tombstones on a clone (the tables' layouts are their
    /// stores'), a fresh layout drawn until one does.
    #[test]
    fn purge_samples_the_value_tables_as_it_begins() {
        const TABLES: [RebuildTable; 3] = [
            RebuildTable::FieldRows,
            RebuildTable::NameIds,
            RebuildTable::SampleRows,
        ];
        let thinned_all = |state: &CompactState, victim: u64| {
            let mut clone = state.clone();
            let mut indexes = fresh_indexes(&clone);
            let before = TABLES.map(|t| shape_of(&clone, &indexes, t).capacity);
            let (_, _, applied) =
                plan_with_indexes(&mut clone, &mut indexes, &deletes_of(&[victim]));
            applied
                && TABLES
                    .iter()
                    .zip(before)
                    .all(|(&t, before)| shape_of(&clone, &indexes, t).capacity < before)
        };
        let (mut state, victim) = (0..64)
            .find_map(|_| {
                let state = value_rows_state();
                let victim = (1..=HOME_SLOT_SECTIONS as u64).find(|&id| thinned_all(&state, id))?;
                Some((state, victim))
            })
            .expect("fixture: a section whose purge leaves three tombstones");
        let mut indexes = fresh_indexes(&state);
        for table in TABLES {
            assert_eq!(
                shape_of(&state, &indexes, table).capacity,
                HOME_SLOT_SECTIONS,
                "fixture: {table:?} at its load limit"
            );
        }
        let before = TABLES.map(|t| shape_of(&state, &indexes, t).capacity);
        let (_, _, applied) = plan_with_indexes(&mut state, &mut indexes, &deletes_of(&[victim]));
        assert!(applied, "fixture: the delete applies");
        let unsampled: Vec<_> = TABLES
            .iter()
            .zip(before)
            .filter(|&(&table, before)| {
                let after = shape_of(&state, &indexes, table).capacity;
                after >= before || indexes.capacity_high_water(table) != before
            })
            .map(|(&table, before)| (table, before, indexes.capacity_high_water(table)))
            .collect();
        assert!(
            unsampled.is_empty(),
            "the purge did not note these tables at their capacity() as it began (table, \
             capacity, high-water): {unsampled:?}"
        );
    }

    /// `name_remove`: deleting the only child of one of the
    /// children-run fixture's parents drops that parent's key from the name
    /// index. The parent is one whose key's erase left a tombstone on a
    /// clone of the indexes, fresh indexes drawn until one does.
    #[test]
    fn name_remove_samples_the_name_index_before_dropping_a_parent_key() {
        let (state, parents) = children_run_state();
        let delete_child = |k: usize| OwnedDelta {
            struct_commands: vec![StructCommand::DeleteSection(
                oodf_ffi::CommandDeleteSection {
                    parent_id: parents[k],
                    section_id: RUN_CHILD_FIRST + k as u64,
                },
            )],
            ..OwnedDelta::default()
        };
        let (indexes, k) = (0..64)
            .find_map(|_| {
                let indexes = fresh_indexes(&state);
                let k = (0..RUN_PARENTS).find(|&k| {
                    let (mut s, mut i) = (state.clone(), indexes.clone());
                    let before = shape_of(&s, &i, RebuildTable::NameIndex).capacity;
                    plan_with_indexes(&mut s, &mut i, &delete_child(k)).2
                        && shape_of(&s, &i, RebuildTable::NameIndex).capacity < before
                })?;
                Some((indexes, k))
            })
            .expect("fixture: a parent key whose erase leaves a tombstone");
        let (mut state, mut indexes) = (state.clone(), indexes);
        let before = shape_of(&state, &indexes, RebuildTable::NameIndex).capacity;
        let (_, _, applied) = plan_with_indexes(&mut state, &mut indexes, &delete_child(k));
        assert!(applied, "fixture: the delete applies");
        let after = shape_of(&state, &indexes, RebuildTable::NameIndex).capacity;
        assert_sampled_before(
            &indexes,
            RebuildTable::NameIndex,
            before,
            after,
            "name_remove",
        );
    }

    /// 50 sections under the root, each holding one external field `e`
    /// under a hash of its own, compacted: the plan indexes' external-hash
    /// counts hold 50 hashes in 64 buckets, 6 slots of room.
    fn external_rows_state() -> CompactState {
        let mut layer = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..ObjectState::default()
        };
        let ids: Vec<u64> = (1..=50).collect();
        for &id in &ids {
            layer.sections.insert(id, root_section(id));
            layer.fields.insert(
                (id, "e".to_owned()),
                Field {
                    section_id: id,
                    key_name: "e".to_owned(),
                    value: FieldValue::External {
                        hash: [id as u8; 32],
                        size: 64,
                        ext_value_index: 0,
                    },
                    set_order: 1,
                    authoring_set_order: 1,
                    version: 1,
                },
            );
        }
        layer.children.insert(0, ids);
        let mut state = compact(&layer);
        compact_state(&mut state);
        state
    }

    /// `count_hash`'s removal: deleting a section whose external hash
    /// no other row holds drops the hash from the counts. The section is one
    /// whose hash's erase left a tombstone on a clone of the indexes.
    #[test]
    fn count_hash_samples_the_external_counts_before_dropping_a_hash() {
        let state = external_rows_state();
        let (indexes, victim) = (0..64)
            .find_map(|_| {
                let indexes = fresh_indexes(&state);
                let victim = (1..=50u64).find(|&id| {
                    let (mut s, mut i) = (state.clone(), indexes.clone());
                    let before = shape_of(&s, &i, RebuildTable::ExtCounts).capacity;
                    plan_with_indexes(&mut s, &mut i, &deletes_of(&[id])).2
                        && shape_of(&s, &i, RebuildTable::ExtCounts).capacity < before
                })?;
                Some((indexes, victim))
            })
            .expect("fixture: a hash whose erase leaves a tombstone");
        let (mut state, mut indexes) = (state.clone(), indexes);
        let before = shape_of(&state, &indexes, RebuildTable::ExtCounts).capacity;
        let (_, _, applied) = plan_with_indexes(&mut state, &mut indexes, &deletes_of(&[victim]));
        assert!(applied, "fixture: the delete applies");
        let after = shape_of(&state, &indexes, RebuildTable::ExtCounts).capacity;
        assert_sampled_before(
            &indexes,
            RebuildTable::ExtCounts,
            before,
            after,
            "count_hash",
        );
    }

    /// `restore_ext_counts`, through `count_hash`: a plan writes a
    /// field `f` under a new external hash, adding it to the counts, and its
    /// rollback drops it again. The hash is one whose erase left a tombstone
    /// on a clone. The plan removes nothing from the counts, so their
    /// high-water is 0 until the rollback's removal.
    #[test]
    fn restore_ext_counts_samples_the_external_counts_before_the_rollback_drops_a_hash() {
        let state = external_rows_state();
        let write = |tag: u8| OwnedDelta {
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "f".to_owned(),
                value_or_ext_hash: vec![tag; 32],
                ext_value_size: 64,
                ext_value_index: 0,
                set_order: 1,
            }],
            ..OwnedDelta::default()
        };
        let rolled_back = |state: &CompactState, indexes: &PlanIndexes, tag: u8| {
            let (mut s, mut i) = (state.clone(), indexes.clone());
            let (journal, _, applied) = plan_with_indexes(&mut s, &mut i, &write(tag));
            let before = shape_of(&s, &i, RebuildTable::ExtCounts).capacity;
            let high_water = i.capacity_high_water(RebuildTable::ExtCounts);
            journal.rollback(&mut s, &mut i).expect("the rollback");
            (applied, high_water, before, s, i)
        };
        let (indexes, tag) = (0..64)
            .find_map(|_| {
                let indexes = fresh_indexes(&state);
                let tag = (200..=255u8).find(|&tag| {
                    let (applied, _, before, s, i) = rolled_back(&state, &indexes, tag);
                    applied && shape_of(&s, &i, RebuildTable::ExtCounts).capacity < before
                })?;
                Some((indexes, tag))
            })
            .expect("fixture: a new hash whose erase leaves a tombstone");
        let (applied, planned_high_water, before, state, indexes) =
            rolled_back(&state, &indexes, tag);
        assert!(applied, "fixture: the write applies");
        assert_eq!(
            planned_high_water, 0,
            "fixture: the plan removed nothing from the counts"
        );
        let after = shape_of(&state, &indexes, RebuildTable::ExtCounts).capacity;
        assert_sampled_before(
            &indexes,
            RebuildTable::ExtCounts,
            before,
            after,
            "restore_ext_counts",
        );
    }

    /// `restore_name_index` step 3: a plan creates a section under one
    /// of the children-run fixture's children, adding that child's key to
    /// the name index, and its rollback drops the key again. The child is
    /// one whose key's erase left a tombstone on a clone. The plan removes
    /// nothing from the name index, so its high-water is 0 until the
    /// rollback's removal.
    #[test]
    fn restore_name_index_samples_the_name_index_before_the_rollback_drops_a_parent_key() {
        let (state, _) = children_run_state();
        let create_under = |k: usize| OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: RUN_CHILD_FIRST + k as u64,
                section_id: 2_000_000,
                section_name: "grandchild".to_owned(),
                section_type: 6,
            })],
            ..OwnedDelta::default()
        };
        let rolled_back = |state: &CompactState, indexes: &PlanIndexes, k: usize| {
            let (mut s, mut i) = (state.clone(), indexes.clone());
            let (journal, _, applied) = plan_with_indexes(&mut s, &mut i, &create_under(k));
            let before = shape_of(&s, &i, RebuildTable::NameIndex).capacity;
            let high_water = i.capacity_high_water(RebuildTable::NameIndex);
            journal.rollback(&mut s, &mut i).expect("the rollback");
            (applied, high_water, before, s, i)
        };
        let (indexes, k) = (0..64)
            .find_map(|_| {
                let indexes = fresh_indexes(&state);
                let k = (0..RUN_PARENTS).find(|&k| {
                    let (applied, _, before, s, i) = rolled_back(&state, &indexes, k);
                    applied && shape_of(&s, &i, RebuildTable::NameIndex).capacity < before
                })?;
                Some((indexes, k))
            })
            .expect("fixture: a new parent key whose erase leaves a tombstone");
        let (applied, planned_high_water, before, state, indexes) =
            rolled_back(&state, &indexes, k);
        assert!(applied, "fixture: the create applies");
        assert_eq!(
            planned_high_water, 0,
            "fixture: the plan removed nothing from the name index"
        );
        let after = shape_of(&state, &indexes, RebuildTable::NameIndex).capacity;
        assert_sampled_before(
            &indexes,
            RebuildTable::NameIndex,
            before,
            after,
            "restore_name_index",
        );
    }

    /// `normalize_touched`: a plan re-parents the only child of one of
    /// the children-run fixture's parents to the root, leaving that parent's
    /// list empty, and the touched normalization drops it from the children
    /// table (a run on home slots 0 to 49: a tombstone at either width).
    #[test]
    fn normalize_touched_samples_the_children_table_before_dropping_an_emptied_list() {
        let (mut state, parents) = children_run_state();
        let mut indexes = fresh_indexes(&state);
        let (_, touches, applied) =
            plan_with_indexes(&mut state, &mut indexes, &reparent_to_root(0));
        assert!(applied, "fixture: the re-parent applies");
        assert!(
            state.children.get(&parents[0]).is_some_and(Vec::is_empty),
            "fixture: the plan left the emptied list to the normalization"
        );
        assert_eq!(
            indexes.capacity_high_water(RebuildTable::Children),
            0,
            "fixture: the plan removed nothing from the children table"
        );
        let before = shape_of(&state, &indexes, RebuildTable::Children).capacity;
        crate::domain::plan_index::normalize_touched(&mut state, &mut indexes, &touches.lists);
        assert!(
            !state.children.contains_key(&parents[0]),
            "fixture: the normalization dropped the list"
        );
        let after = shape_of(&state, &indexes, RebuildTable::Children).capacity;
        assert_sampled_before(
            &indexes,
            RebuildTable::Children,
            before,
            after,
            "normalize_touched",
        );
    }

    /// `PlanJournal::rollback`'s note at its start: a plan creates one
    /// section with a field under a new name and a time sample — inserts
    /// alone, so no table is noted while it plans; the rollback then notes
    /// each of the state's five tables at the capacity it reads, before it
    /// removes the section, its row, its name and its sample. Every kept
    /// section holds a field under a name of its own and a time sample, so
    /// each of the five tables reads a capacity above 0 and the note of each
    /// is checked against a value it could not reach by default. The
    /// sections table is a run on home slots 0 to 49 of 64 buckets, and the
    /// created section's home slot is 50, so it extends the run and its erase
    /// leaves a tombstone at either width.
    #[test]
    fn rollback_samples_each_state_table_as_it_begins() {
        let mut sections: HashMap<u64, Section> = HashMap::with_capacity(HOME_SLOT_SECTIONS);
        let ids = ids_by_home_slot(sections.hasher(), 63, 0..51, 1);
        let (kept, created) = (&ids[..50], ids[50]);
        for &id in kept {
            sections.insert(id, root_section(id));
        }
        let mut state = CompactState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..CompactState::default()
        };
        state.sections = sections;
        state.children.insert(0, kept.to_vec());
        for &id in kept {
            state.fields.insert(id, &format!("k{id}"), None);
            state.time_samples.insert(id, 1.0f64.to_bits(), None);
        }
        let mut indexes = fresh_indexes(&state);
        let (journal, _, applied) = plan_with_indexes(
            &mut state,
            &mut indexes,
            &OwnedDelta {
                struct_commands: creates_of(&[created]),
                set_fields: vec![oodf_ffi::SetField {
                    section_id: created,
                    key_name: "created".to_owned(),
                    value_or_ext_hash: vec![1],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 1,
                }],
                time_samples: vec![oodf_ffi::TimeSample {
                    section_id: created,
                    time: 1.0,
                    value_or_ext_hash: vec![1],
                    ext_value_size: 0,
                    ext_value_index: 0,
                }],
                ..OwnedDelta::default()
            },
        );
        assert!(applied, "fixture: the create applies");
        let state_tables: Vec<RebuildTable> = RebuildTable::ALL
            .into_iter()
            .filter(|table| table.in_state())
            .collect();
        for &table in &state_tables {
            assert_eq!(
                indexes.capacity_high_water(table),
                0,
                "fixture: the plan removed nothing from {table:?}"
            );
        }
        let before: Vec<usize> = state_tables
            .iter()
            .map(|&table| shape_of(&state, &indexes, table).capacity)
            .collect();
        for (&table, &before) in state_tables.iter().zip(&before) {
            assert!(before > 0, "fixture: {table:?} reads a capacity above 0");
        }
        journal
            .rollback(&mut state, &mut indexes)
            .expect("the rollback");
        for (&table, &before) in state_tables.iter().zip(&before) {
            assert_eq!(
                indexes.capacity_high_water(table),
                before,
                "the rollback noted {table:?} at its capacity() as it began"
            );
        }
        let after = shape_of(&state, &indexes, RebuildTable::Sections).capacity;
        assert_sampled_before(
            &indexes,
            RebuildTable::Sections,
            before[RebuildTable::Sections.index()],
            after,
            "the rollback",
        );
    }
}
