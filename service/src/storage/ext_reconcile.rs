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

// External-value `ref_count` reconciler (b.yda).
//
// A leader-gated periodic pass that compares every `le:ext:meta:{hash_hex}`'s
// `ref_count` against the number of LIVE OBJECTS that actually reference the
// hash, and hands back the difference through the ordinary GC release channel.
// The same pass audits and repairs each live object's per-object reference
// index — see "The per-object index audit, and the drain gate" below — and
// merges its legacy `-0.0` time-sample rows — "The signed-zero merge".
//
// ── What it exists to drain ───────────────────────────────────────────────
//
// `ref_count` counts live objects (b.qpf). Every transition of an object's
// reference set emits exactly one release event and every release event
// decrements exactly once — but a handful of documented shapes leave an
// increment standing that no event will ever hand back:
//
// * a process that dies between a commit's `EXEC` and its release enqueue;
// * references leaked before b.qpf landed, which nothing migrated;
// * a DR restore's suppressed surplus (`SurplusPolicy::Suppress`) and a
//   composed lineage's pre-stored-but-superseded blobs;
// * an `ExecError`-torn commit;
// * the proven-unwatched "retired underneath" commit verdict, which
//   deliberately does not enqueue (leak, not loss — see `storage::layer`);
// * a corrupt-keyspace drop site where the hash could not be decoded at all,
//   so nothing could name the blob to release it.
//
// Every one of them leaks in the same direction: `ref_count` too high. The GC
// sweep cannot see any of them, because it only performs decrements someone
// owed it. This pass is the only thing in the service that compares the count
// against reality.
//
// ── Why it is safe to act on a SCAN-derived answer ────────────────────────
//
// `SCAN` has a weak guarantee: a key added or removed during a walk may or may
// not appear. Acting on a single walk's arithmetic would therefore risk
// decrementing a live reference — data loss, since the GC deletes at zero. The
// pass never does. Three guards stack:
//
// 1. **Two observations a grace apart, on the surplus AMOUNT.** A candidate is
//    `{hash_hex}:{surplus}`, registered with `ZADD NX` so its score is the FIRST
//    time that exact amount was seen, and it becomes claimable only a full grace
//    later. Anything transient — a pre-store awaiting its commit, a commit
//    awaiting its enqueue, an enqueue awaiting the GC, an object the walk
//    happened to miss — changes the amount, produces a different member, and
//    restarts the clock. What survives is a surplus that is genuinely standing
//    still, and the repair is exactly the amount the member encodes.
// 2. **The `owed` subtraction.** A release already sitting on
//    `le:ext:{gc}:pending` or `le:ext:{gc}:inflight` is an obligation the GC
//    still owes; counting it as surplus would decrement twice. This is also what
//    makes the pass idempotent against ITSELF: between emitting a repair and the
//    GC draining it, the repair is in `owed`, so the surplus reads zero and the
//    candidate retires as healed.
// 3. **The `last_store_ms` pre-flight stamp** (written by
//    `scripts/ext_store_value.lua` on every pre-flight store, read here).
//    This is NOT a secondary guard. Guards 1 and 2 do not cover the pre-store
//    window: `store_values` increments `ref_count` BEFORE the commit that
//    materialises the row, so an upload in flight across the pass reads as pure
//    surplus (the row is not there for the object walk to see, and no release
//    is owed yet for the owed subtraction to find). Two independent uploads a
//    grace apart both spell `{hash_hex}:1`, and `ZADD NX` pins the first — so
//    guard 1 confirms the amount, not the cause, and repairing it would hand
//    back an increment a live commit is about to rely on. Over-release, from a
//    pass whose whole purpose is to refuse it.
//
//    Only the stamp distinguishes those, and the gate it drives is THREE-way,
//    not two:
//
//    * **Present and older than the cutoff ⇒ eligible.** Nothing has pre-stored
//      this hash for a full grace.
//    * **Present and newer than the cutoff ⇒ ineligible**, `skipped_recent_store`.
//      An increment may still be awaiting its commit. This is the healthy,
//      busy-store arm.
//    * **Present but UNPARSEABLE ⇒ ineligible**, `skipped_unreadable_stamp`.
//      Nothing in this service writes a stamp that cannot be read
//      (`string.format("%d", ms)`), so a field the pass cannot parse is evidence
//      of corrupt or foreign keyspace. Fail closed: acting on the strength of a
//      value the pass failed to understand is exactly the shape "fail closed,
//      absolutely" below refuses. Folding it into the absent case would read
//      corruption as "nothing ever stored this" and license a repair.
//    * **ABSENT ⇒ eligible.** Every pre-flight store stamps, so an absent field
//      is positive evidence that no pre-store has touched this hash since the
//      writer deployed — the hash's `ref_count` was raised by a build that no
//      longer runs, or by a path that has since been fixed, and its surplus is
//      exactly the standing leak this pass exists to drain.
//
//    **What the Valkey-side clock buys, and what it does not.** Both sides of
//    that comparison are sampled Valkey-side — the stamp by
//    `ext_store_value.lua`, the cutoff by phase 1's keyless `TIME` — which takes
//    the CLIENT clock out of it: no service replica's own clock can move the
//    cutoff. It does NOT reduce the comparison to one clock under Cluster. The
//    keyless `TIME` lands on an arbitrary primary while the stamp was written
//    by the node owning `{hash_hex}`, so an inter-node residual remains, and it
//    strips this guard only if the sampled node's clock runs AHEAD of the
//    stamping node by more than the grace (≥ 10 min) — enough to make a
//    just-stored stamp look older than the cutoff and re-open the pre-store
//    window this guard closes. Cluster nodes must be time-synchronised well
//    inside that bound; `docs/service/docs/backing-store.md` § Requirements
//    states the requirement (within 15 s), and any NTP-synchronised fleet
//    clears it by orders of magnitude.
//
//    **The absent-is-eligible rule, and the rollout bound on it.** No released
//    build can produce the mixed fleet (b.yda and b.rej are both unreleased), so
//    the two-step rollout the b.yda hand-off offered as an option is not needed
//    and is deliberately not taken: there is no config flag and no staged
//    release, the writer and this flip ship in one commit. Even inside a dev/rc
//    rollout the exposure is bounded to one full reconcile interval: candidates
//    register `ZADD NX` at pass time and are claimable only a full grace later,
//    passes run every `interval`, and `Config::validate_ext_reconcile_interval`
//    rejects `interval < grace`, so the earliest repair of an unstamped meta is
//    the second pass after a new replica first takes `le:gc:leader` (≥ 600 s,
//    default 3600 s). What remains true: a deployment upgrading from a build
//    older than this one can, ONLY if pre-writer replicas are still accepting
//    uploads a full reconcile interval after the first new replica takes
//    leadership, repair one reference against a peer's in-flight pre-store.
//    `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0` (existing, WARN-logged) is the
//    off switch for such a window.
//
// ── Phase order is load-bearing ───────────────────────────────────────────
//
// The phases run objects → retry carriers → pending → inflight → metas, and
// that order is a correctness argument, not a convenience.
//
// An obligation flows through the pass's inputs in exactly one direction:
// an object's reference becomes, at retirement, either a member on a retry
// carrier or a member on `pending`; a carrier's member becomes a `pending`
// member when the sweep drains it; a `pending` member becomes an `inflight`
// member when the GC claims it; an `inflight` member becomes a decrement. Each
// stage is walked BEFORE the stage it flows into. A transition that happens
// mid-pass therefore lands in the destination stage, which this pass has not
// walked yet, after the pass already counted it in the source stage — so the
// obligation is counted TWICE. Double-counting inflates `referenced + owed`,
// which under-states the surplus: the leak direction, and self-correcting on
// the next pass.
//
// The double-count is deliberate, routine, and the reason the DEFICIT arm needs
// its own confirmation: an under-stated surplus that crosses zero is a negative
// surplus this pass invented. See "Deficit is confirmed, then observed" below.
//
// Walking in the opposite order inverts that. A mid-pass transition would land
// in a stage already walked and leave a stage not yet walked, so NEITHER walk
// sees it — the obligation vanishes from the arithmetic, the surplus is
// over-stated, and the pass repairs a reference someone still owes a decrement
// for. Over-release.
//
// **The carrier → `pending` transition is not atomic, and it OVERLAPS rather
// than gaps (b.yuo Part R).** The argument above assumes each transition is
// atomic, so the obligation is in exactly one of the two stages at every
// instant; that holds for `pending` → `inflight` (one Lua claim moves the
// member). A retirement intent's carrier (`le:{oid}:obj:extret`) is not moved in
// one step: its members are enqueued onto `le:ext:{gc}:pending` first and the
// carrier is deleted only afterwards, once the attempt is known to have landed
// (`scripts/ext_retire_clear.lua`) — nothing deletes a carrier before its
// members reach `pending`, except a cancel, after which nothing is owed, and
// the drop of an older build's SET carrier (b.1jt), after which nothing is
// owed unless that build never paid it — a leak this pass then sees as
// surplus and repairs, the direction a drop is meant to fail in. So in
// between, the obligation is in BOTH stages, and a pass that walks the carrier
// and then `pending` counts it twice: the leak direction, transient, like every
// other double count here. There is no window in which neither walk sees it.
//
// The same overlap holds for the whole of a retirement in flight: its begin
// writes the carrier while the object is still live, so between the begin and
// the teardown the object walk counts the reference in `referenced` and the
// carrier walk counts it in `owed`. Leak direction again, bounded by the
// retirement's own duration.
//
// ── Fail closed, absolutely ───────────────────────────────────────────────
//
// Any error in the object walk, the carrier walk, the owed walk or the meta
// walk — a per-object read, a SCAN-family step error, a walk that crossed a
// change of serving process, an undecodable hash the walk cannot attribute, a
// budget elapse, a CANCELLATION — aborts the whole pass with
// NO repair at all and NO gauge publish. The orphan sweep's
// `continue`-on-SCAN-error posture is deliberately NOT copied: a partial object
// walk UNDER-counts `referenced`, and an under-counted `referenced` reads as
// surplus, which is precisely the over-release this design exists to refuse.
// Deferring every repair for one tick costs nothing; the ZSET is durable and the
// surplus is not going anywhere.
//
// Cancellation is in that list for the same reason and with no exception. A
// walk cut short at shutdown produces exactly the truncated map an errored walk
// does, so it forfeits the pass rather than returning a partial answer: no
// gauges are published from it, and no candidate is pinned — a first
// observation pinned from a truncated walk would sit on the ZSET at an amount
// that was never real, waiting for a future pass to confirm it.
//
// A walk of this pass is complete only on its terminal step, and only when
// one server process served every step of it. The object, carrier and owed
// walks are guarded (`storage::scan_guard`): explicit-cursor steps that end
// only on cursor `0` or with an error (a step lost in flight fails visibly, where
// fred's SCAN-family streams end with no error), each sent with one attempt
// and retried in place only after a transient error and while the serving
// process's `run_id` reads unchanged (a deterministic error ends the walk on
// its first attempt), and each step's keys or members used only once the
// witness sent after it in its pipeline matched the walk's start. A keyspace
// walk on a cluster also sends each primary's commands to one named node,
// compares its slot map and every primary's witness at its start and end, and
// refuses an open slot or a node whose own myself line does not list exactly
// its primary's slots (the client's routing table was out of date: the walk
// re-syncs it, then fails).
// A walk that crossed a change of serving process or slot map, whose node
// pin missed its primary's slots, or whose `INFO server` reports no usable
// run_id (or `CLUSTER NODES` no usable myself line), ends with an error, and
// that forfeits the pass like any other. A stale table would make the same
// skip on every pass, which guard 1 cannot catch. On Azure Managed Redis the
// run_id is a node endpoint's, and a primary shard that moves behind an
// unchanged endpoint and slot map keeps it [tested, SDD §11.3.2]: guard 1's
// two observations absorb one such missed walk, unless an earlier, unrelated
// observation of the same hash happened to record exactly the inflated
// amount (`act_on_candidate` recomputes from the current pass's walks). The
// restore Job's recount reads no witness ("The restore Job's authoritative
// recount", below). The meta walk goes through
// `orphan_sweep::scan_keyspace`, a coverage walk: every primary of a fresh
// slot map on a node pin, its owner checked at the walk's start and end, but
// no serving-process witness. Any error it ends with — a store fault, a
// primary left unwalked or an unproven pin's routing table found out of date
// (after either it re-syncs the walking pool member's routing table, so a
// later pass on that member is complete once its re-sync succeeds; a failed
// or timed-out sync leaves the miss in place, and each maintenance-pool
// member keeps its own table), a changed map, an ownership it could not
// prove — forfeits the pass like any other. A change of serving
// process inside one walk can still skip metas undetected; that is safe by
// direction: a meta it misses is never evaluated, so it is never registered
// or repaired, and the miss can only leave a surplus standing (a leak) for
// that pass.
//
// ── Deficit is confirmed, then observed, and never repaired ───────────────
//
// When `referenced + owed` exceeds `ref_count`, the arithmetic says an
// over-release happened. It is NEVER repaired: incrementing would resurrect a
// meta whose data the GC may already have deleted, producing a `ref_count` that
// claims a blob which is gone. The pass reports, and does nothing else.
//
// **But a single pass's negative surplus is not evidence of anything.** The
// phase order above buys its safety by DOUBLE-COUNTING mid-pass transitions, and
// a double-count is exactly a spurious deficit: a hash whose last reference is
// dropped by a commit or a retirement between the object walk and the pending
// walk is counted once as a live reference and once as an owed release, and
// reads as `surplus = −1` on a perfectly healthy store. That is routine, not
// rare. The stale-carrier-over-a-live-object case in `walk_retire_carriers` is a
// second source of the same shape.
//
// So a deficit goes through the SAME two-observation protocol a surplus does,
// and for the same reason — the amount has to hold still across a grace before
// it means anything. A negative surplus registers `{hash_hex}:-{deficit}` on the
// same `le:ext:{gc}:recon` ZSET; a grace later the claim re-reads `ref_count`
// fresh and recomputes. Same amount ⇒ **confirmed**: tick `deficit_observed` and
// publish the magnitude on `live_edit_ext_ref_deficit_total`. Anything else ⇒ it
// was one of the transients: tick `deficit_cleared` and retire the member
// silently.
//
// **Confirmation buys an amount that held still, not a persistent deficit, and
// the alert contract is written to match: alert on a deficit that RECURS over
// several reconciler intervals, not on one report.** On a hash whose references
// churn — dropped and re-acquired at a rate comparable to a pass — the same
// mid-pass double-count can recur at the same amount on the confirming pass, and
// confirm a transient. What would settle it is attributing each obligation to
// the object that owes it, so a re-observed deficit could be told from a fresh
// one; the pass tallies per hash only, and b.1sv tracks the per-object tally.
// Recurrence is the available proxy in the meantime: a churn coincidence does
// not repeat window after window, and a real over-release never heals.
//
// The deficit arm shares the ZSET and the grace with the surplus arm but NOT the
// `last_store_ms` eligibility rule. That rule exists because a pre-flight
// increment awaiting its commit looks like surplus; nothing about an absent,
// fresh or unreadable stamp can make a DEFICIT spurious, so a meta in any of
// those states is registered, confirmed and reported exactly like a
// comfortably-stamped one.
//
// The confirm arm is confirm-only. It never emits a release member, never
// increments, and cannot repair in either direction.
//
// ── The per-object index audit, and the drain gate ────────────────────────
//
// The pass also audits every live object's per-object reference index,
// `le:{oid}:obj:ext` (`KeySchema::obj_ext_index`), and that is a different
// count from the one above: per hash, how many of the object's value rows hold
// it. The commit paths trust it — a rewrite whose stored count reaches 0
// releases the object's reference — so an index that under-counts the rows can
// release a reference a live row still holds, and the GC then deletes a blob in
// use. An index that over-counts only leaks until the object retires.
//
// The object walk already reads every row's raw `hash` column and the whole
// index, so it compares them per object at no extra read. On a disagreement it
// runs the atomic recount (`storage::ext_index_audit`,
// `scripts/ext_index_recount.lua`), which recounts the rows by the decoder's
// rule and rewrites the differing fields in one atomic unit. The recount
// releases nothing: an over-count it removes leaves a `ref_count` surplus for
// a later pass of this reconciler to drain through the ordinary
// two-observation protocol (subject to every guard that protocol applies), and
// an under-count it raises changes no global count.
//
// **The drain gate.** An index under-counted by an unindexed (pre-index)
// object's SEEDING commit becomes a loss only if this pass drains the surplus
// that keeps the blob alive: the seeding commit paid a pre-flight `+1` that
// reads as surplus here, and draining it leaves the next rewrite's release to
// delete the blob under the row the index missed. Drift from other producers
// has no such surplus behind it: a torn commit's under-count (or, written by
// an older build whose commit paths collapsed a `-0.0` / `+0.0` row pair, the
// signed-zero aliasing's) can over-release on the next rewrite, before any
// pass runs, and the audit only shortens that exposure. So a hash whose object index the
// recount could NOT repair — the object is above the recount's row ceiling,
// was not live when it ran, or carries a `hash` column the recount reports as
// malformed (whatever it answered: such a row is uppercase or mixed-case hex
// that the decoder counts and the recount skips, so the recount's count is
// not authoritative; it applies raises only and withholds every lowering) —
// goes into a per-pass `unrepaired_drift` set, and phases 4 and 5 neither
// register nor repair a surplus for it (`skipped_index_drift`). Deficits are
// unaffected.
//
// The gate is sufficient, for that seeding producer, by ORDERING, not timing
// — for drains this build performs: the `last_store_ms` guard keeps the
// seeding commit's surplus ineligible for at least a grace after the seed,
// every drain happens in phase 5, and the object walk that audits and gates
// runs in phase 2 of the same pass. By the time any drain of this build can
// happen, an audit has either repaired the object or gated the hash.
//
// What the audit guarantees, **from the first audit pass on this build**, is
// **no loss through a drain**, not a repair and not no loss in general:
//
// * Drains outside this build are not covered. A seed surplus that a pre-fix
//   reconciler (no audit, no gate) already drained, or that an old-build
//   replica holding `le:gc:leader` drains during a rolling upgrade, leaves an
//   under-count with no surplus behind it, exposed to the commit paths until
//   the first audit pass on this build repairs it.
// * On the txn backend, a commit whose `WATCH` was void is not cancelled by a
//   repair's header stamp: it can overwrite a repair this pass made in phase
//   2, restoring the under-count, and phase 5 can then drain the surplus
//   behind it. That combination can still lose through a drain.
// * A gated hash leaks until its object is repaired, shrinks below the ceiling
//   or retires.
// * A recount that fails forfeits the pass, like every other walk error. A key
//   that keeps failing it (a wrong-typed header, index, row index or value
//   row) forfeits every pass until an operator clears it: nothing is repaired
//   and nothing is drained, so no drain can complete a loss, but an
//   under-count already standing stays exposed to the commit paths.
// * `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0` disables the audit with the rest
//   of the pass. A pre-index object's seeding under-count whose surplus is
//   still standing then cannot become a loss (nothing drains it), but drift
//   from any other producer is never repaired by a pass and stays exposed to
//   the commit paths.
//
// ── The signed-zero merge ─────────────────────────────────────────────────
//
// The object walk also folds every legacy `-0.0` time-sample row into the
// `+0.0` key (`storage::time_sample_merge`), so the store converges to one
// row per section and USD time. It finds them for free — the walk lists each
// object's `obj:tss`, and a `-0.0` member ends `:8000000000000000` — and runs
// before the object's references are read, then lists the object again, so
// the same pass's index audit recounts against the merged rows. The merge
// writes no `obj:ext` field and releases nothing: a row it drops can only
// leave the index over-counted, the leak direction this audit then repairs.
// Unlike every other step of the walk it never forfeits the pass: it is not
// needed for correctness (readers already resolve a pair by the same rule),
// so a failure is counted, logged, and retried on the next pass.
//
// ── The restore Job's authoritative recount (SDD SR-9.8) ──────────────────
//
// `recount_authoritative` is the second mode of this module, used only by the
// DR restore Job (`dr::job::finish`), after its last write and while the
// fleet is held. It walks the same three families — objects, retirement
// carriers, owed releases — each by the DR Jobs' completeness proofs rather
// than the serving-process guard (below), and then SETS each named hash's
// `ref_count` to exactly `refs + owed` (`scripts/ext_ref_count_set.lua`, one
// atomic call per hash on its meta key alone), where:
//
// * `refs(h)` is the number of object headers whose rows ∪ `obj:ext` name h,
//   once per object, over EVERY `{*}:obj:hdr` — live, unmapped and superseded
//   objects alike, since each releases its references once when it retires;
// * `owed(h)` is its retirement-carrier members plus its `{gc}:pending` and
//   `{gc}:inflight` members.
//
// Every hash in referenced ∪ owed gets a meta: a value whose bytes only the
// backup store holds gets a meta with no data, and a hash only an owed
// release names gets one at `owed`, which the GC then drains to 0. The count
// is exact — lowered as well as raised — and so is never below what names the
// hash and never 0 while anything names it. The recount deletes no meta, and
// leaves a meta that nothing names as it is (a bounded leak the periodic pass
// above drains). It writes nothing else: the object walk runs in its
// `Authoritative` mode, which skips the signed-zero merge and the index audit
// (both writes), and the meta walk, the two observations, the stamp gate, the
// repair cap and the drain gate are not run at all. The periodic pass is
// unchanged.
//
// **Why none of those guards is needed.** The guards above exist for writers
// running concurrently with a walk. The recount runs while the fleet is held,
// after the Job's quiet period has bounded every writer the hold aborted and
// after the Job's own last write, so nothing writes the keyspace the walks
// read: a complete walk is an exact snapshot. Each walk is shown complete on
// the held store or fails (`storage::scan_guard`, *The DR Jobs' walks*),
// whichever server process answered each step, and no `INFO server` is read:
//
// * the object and carrier walks are agreeing keyspace walks: each pattern
//   is walked twice, the second walk after the first ends, and its result
//   used only when both returned the same set (the object walk counts each
//   object once as its first walk meets it, then a headers-only second walk
//   must return exactly the objects counted);
// * each object's `obj:flds` and `obj:tss`, and `inflight`, are read by
//   ascending rank pages with `ZCARD` in each page's pipeline, complete only
//   when the pages returned the start `ZCARD` and the end `ZCARD` equals it;
// * each object's `obj:ext` and `pending` are read by counted cursor walks,
//   `HLEN` / `SCARD` in each step's pipeline, their members de-duplicated
//   exactly, complete only when the distinct count equals the cardinality
//   read with the first and the last step.
//
// A proof that does not hold fails the recount, exit 1 (the store changed
// while the Job read it), as does a slot map that changed or a node pin
// that missed its primary's slots; an unusable myself line fails it, exit 3.
// Every decode fails closed, and a double count (a retiring object counted
// as a reference and through its carrier or release) only errs high, a leak.
// Nothing reads a count — GC, collect, backup repair, this pass — until the
// un-hold, which takes the recount's receipt and so runs only after every
// assignment succeeded.
//
// **Bounded per object.** The recount reads each object's `obj:flds` and
// `obj:tss` by `ZRANGE` pages of `ROW_HASH_CHUNK_ROWS`, each page's rows'
// hashes read in one pipeline and folded into the object's hash set before
// the next page, then its `obj:ext` by counted `HSCAN` steps: one page, one
// chunk and one `HSCAN` step at a time, whatever the object's size
// (`object_referenced_hashes_paged`). What grows is one entry per distinct
// hash: the object's set and the pass's `referenced` and `owed` maps (and
// one entry per distinct `obj:ext` field, by its text, in that index's
// de-duplication set, and per distinct member in `pending`'s). Rank pages
// need no de-duplication, so the
// row indices hold nothing per row. The periodic pass keeps its whole
// listings, which its signed-zero merge needs.
//
// **No silent omission.** A recount that lowers anything may never come from
// a walk that left an object out. Every arm by which a periodic walk skips,
// forfeits or holds an object, and what the authoritative mode does instead
// (exit 1: the Job's store fault or a walk not shown complete, retried;
// exit 3: a corrupt record or a missing myself line, the service stays
// held):
//
// | Walk | Periodic arm | Authoritative |
// |---|---|---|
// | objects, carriers, owed | cancellation forfeits the pass | never fires (a token nothing cancels); the Job stops by dropping the recount, with no receipt and so no un-hold |
// | objects, carriers, owed | the pass budget elapses: forfeit | no budget |
// | objects, carriers, owed | a SCAN / SSCAN / ZSCAN step that failed transiently on every attempt or with a deterministic error on one, an `INFO server`, `CLUSTER SLOTS` or `CLUSTER NODES` read of the walk's guard that errs, or, for a keyspace walk on a cluster, a node pin the client's routing table no longer lists as a primary (refused before a step, a witness or a `CLUSTER NODES` read is sent): forfeit | a SCAN step that failed the same way, a `ZRANGE` page, `HSCAN` / `SSCAN` step or cardinality read that errs, a `CLUSTER SLOTS` or `CLUSTER NODES` read that errs, or a refused node pin: fails the recount, exit 1 |
// | objects, carriers, owed | a walk that crossed a change of serving process (a step's `run_id` differs from the walk's first), or, for a keyspace walk on a cluster, whose slot map changed, which met an open slot, or whose pinned node's myself line does not list exactly its primary's slots, or for which the client's table names no node (the routing table is re-synced first): forfeit | a keyspace walk whose slot map changed, which met an open slot, or whose node pin missed its primary's slots; two keyspace walks that returned different keys; a rank or counted walk whose count check did not hold: fails the recount, exit 1, before any count is set |
// | objects, carriers, owed | an `INFO server` reporting no usable run_id (not 40 hex characters, or all zeros), or, for a keyspace walk on a cluster, a `CLUSTER NODES` with no myself line, a myself line short of its 8 fixed fields, or a slot field outside the `N` / `A-B` grammar: forfeit | no `INFO server` is read; an unusable myself line fails, exit 3 |
// | objects | an index read (`ZRANGE` / `HGETALL` in the periodic pass; a `ZRANGE` page or `HSCAN` step in the recount) that errs: forfeit | fails, exit 1; a `WRONGTYPE` reply fails, exit 3 |
// | objects | — | an `HSCAN` step answering an odd number of elements (a protocol error): fails, exit 1 |
// | objects, carriers, owed | any other store error (a read, a pipeline, a reply decode): forfeit | fails, exit 1; a `WRONGTYPE` reply (a key of the wrong type) fails, exit 3 |
// | objects | an `obj:hdr` key that is not text, or names no object id: forfeit | fails, exit 3 |
// | objects | a row-hash pipeline answering the wrong number of replies: forfeit | fails, exit 3 |
// | objects | a row `hash` column `hash_from_hex` rejects: forfeit | fails, exit 3 |
// | objects | an `obj:ext` field that is not a hash: forfeit | fails, exit 3 |
// | objects | index drift (`obj:ext` disagrees with the rows): the index recount (a write), and when it cannot repair, the hash is held from the surplus arms for the pass | no recount and no hold: the object's references are counted as rows ∪ `obj:ext`, so a hash either side names is counted |
// | objects | an `obj:ext` field whose count is not a positive decimal: an index mismatch, as above | the field still names its hash, which is counted |
// | objects | an index recount that fails: forfeit | not run |
// | objects | the signed-zero merge (a write that never forfeits) | not run; a `-0.0` row and its `+0.0` twin are both read, and their hashes counted once for the object |
// | objects | a row with no `hash` column: nothing to count | the same: such a row holds no external value |
// | objects | an object id that is not decimal: used only as text | counted; only the counter's surviving term passes over it (no allocated id is non-decimal) |
// | carriers | a carrier of type `none` (deleted since the SCAN saw it): nothing owed | the same; nothing deletes one while held |
// | carriers | a carrier of a type other than hash or set: forfeit | fails, exit 3 |
// | carriers, owed | a member that is not text, or whose hash does not decode: forfeit | fails, exit 3 |
// | metas (phases 4–5) | recent or unreadable stamp, index-drift hold, two observations, repair cap, deficit confirmation | not run: every named hash is set exactly |
// | the per-hash set | — | a meta of the wrong type fails, exit 3; any other error, exit 1 |
//
// Rows with no header are walked in neither mode: they never release (a
// retirement writes its carrier first, which is counted as owed).
//
// In the authoritative mode a failure reading one object's references also
// names the object, and the key when the failing read names one
// (`in_object`), so the Job's exit-3 or exit-1 line says what support must
// look at. The periodic mode's errors are unchanged.
//
// ── The meta-without-data shape, and every reader of it ───────────────────
//
// The recount creates a shape nothing else creates: `le:ext:meta:{h}` with
// `ref_count` N ≥ 1 and no stamp or backup field, and `le:ext:data:{h}`
// absent — a value whose bytes only the backup store holds. Outside the held
// Job nothing creates it: the GC deletes meta and data together, and every
// store writes both. It ends three ways: fetch-on-miss stores the bytes, a
// client upload stores them (`SET NX`), or the GC deletes both keys at 0. No
// client write can reference such a value without uploading its bytes: every
// external row's `ext_value_index` must resolve to a part of the same upload,
// or the call is refused before anything is stored
// (`external_values::collect_external_values`). Every reader of the pair:
//
// | Reader | On a meta without data | Verdict |
// |---|---|---|
// | GC decrement (`gc_decrement_and_collect.lua`, via `gc_member`) | `HINCRBY -1`; at 0 the `STRLEN` (a `pcall`) reads 0, `DEL meta data`, freed 0 | correct, unchanged |
// | GC out-of-memory collection (the same script) | frees 0 bytes for it, which is accurate | unchanged |
// | release ledger size (`read_release_blob_size`) | `STRLEN` 0 bytes | unchanged (true Valkey bytes) |
// | pre-flight store (`ext_store_value.lua`, `store_values_with`) | `SET NX` writes the bytes, `+1`, stamp: the shape becomes normal | unchanged |
// | fetch-on-miss rewrite (`store_recovered_values`, the script's `recover` mode) | stores the bytes and keeps the count: no `+1` and no stamp on a present meta | CHANGED: it paid `+1`, a permanent leak |
// | Recover (`recover_value`, the catch-up's restore, and `read_batch`, data absent and meta present; both through `recover_miss`) | DEBUG before the fetch, and the recovery counter's `meta="present"`; an ERROR only when the value could not be fetched, never on a transient `UNAVAILABLE` | CHANGED: it logged an integrity ERROR on every first read after a restore |
// | probe reads (`load_values`, `probe_values`) | the data `GET` misses: a normal answer | unchanged |
// | length probe (`probe_value_lengths`) | `STRLEN` 0, then `EXISTS` false: missing, a normal answer | unchanged |
// | ranged read (`read_value_range`) | an empty `GETRANGE`: read as missing | unchanged |
// | catch-up (`diff_read::resolve_external_values`, `storage::ompf_payload`) | length-probe miss, read hold, re-read, genuine miss, Recover through `recover_value` (fetch-on-miss) before the layout, in batches of at most 4 charged to the catch-up payload budget, each batch re-probed first; a value gone by the closing check is restored the same way, one at a time | CHANGED (D1): restored one value per call, in charged batches, not all at once uncharged |
// | this pass's meta walk and candidate re-read | reads `ref_count` and `last_store_ms` only; after an exact recount the surplus is 0, and an absent stamp is eligible | unchanged |
// | this pass's `obj:ext` audit | reads neither meta nor data | n/a |
// | checkpoint collect R4 (`recon_fence::ext_values_live`) | `EXISTS meta`: live, so the value is deferred and its tombstone re-dated | unchanged: this is what keeps the collect from deleting the backup copy |
// | collect R5 (`clear_ext_backup` and its heal) | needs a meta without data created between R4 and the delete; outside the held Job nothing creates one | unreachable; unchanged |
// | collect R6 heal-on-keep (`heal_from_valkey`) | the read finds no bytes; the copy is probed, and only a copy absent too is reported | CHANGED: it reported every such value unavailable for up to 24 h |
// | collect R3 fence (`recon_ext_commit.lua`) | reads `{recon}` only | n/a |
// | backup repair (`checkpoint::ext_backup_repair`) | owed; probed: a present copy is confirmed (`already_stored`); an absent one reads `STRLEN` 0, `value_gone`, nothing written | unchanged; one existence check per recount meta |
// | mirror confirm (`ext_backup_confirm.lua`) | runs only after a store, and no mirror item exists for a recount meta | n/a |
// | restore coverage (`storage::dr_job::ext_value_stored`) | `EXISTS data` is false, so a re-run probes the backup store | unchanged |
// | restore backfill (`dr::job::execute::object`) | `SET NX` writes the tail bytes, `+1`; the recount then makes the count exact | unchanged |
//
// SDD §11.3, §22.22.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::time::{Duration, Instant};

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface,
};
use fred::types::{ClusterHash, CustomCommand, Value};
use futures::StreamExt;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::observability::metrics::{
    Metrics, SharedMetrics, EXT_INDEX_RECOUNT_SITE_RECONCILE,
    EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED, EXT_RECONCILE_OUTCOME_DEFERRED,
    EXT_RECONCILE_OUTCOME_DEFICIT_CLEARED, EXT_RECONCILE_OUTCOME_DEFICIT_OBSERVED,
    EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED, EXT_RECONCILE_OUTCOME_HEALED,
    EXT_RECONCILE_OUTCOME_REPAIRED, EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT,
    EXT_RECONCILE_OUTCOME_SKIPPED_RECENT_STORE, EXT_RECONCILE_OUTCOME_SKIPPED_UNREADABLE_STAMP,
    EXT_REF_RELEASE_REASON_RECONCILE,
};
use crate::storage::client::{
    is_wrongtype_reply, FredErrOrigin, ValkeyClient, POOL_LABEL_MAINTENANCE,
};
use crate::storage::ext_index_audit::{recount_ext_index, RecountOutcome};
use crate::storage::external_values::{
    enqueue_ext_releases, hash_from_hex, pending_member_hash_hex, recon_claim_expired,
    recon_retire, try_acquire_gc_leader, ExtRelease,
};
use crate::storage::keys::{sha256_to_hex, KeySchema};
use crate::storage::leader_lease::LeaderIdentity;
use crate::storage::lua::EXT_REF_COUNT_SET_SCRIPT;
use crate::storage::orphan_sweep::scan_keyspace;
use crate::storage::retirement::{page_by_cursor, scan_step_firsts, CursorPageReader};
use crate::storage::scan_guard::{
    collect_keyspace_agreed, keyspace_walks_agree, page_by_rank_counted, walk_counted, CountedScan,
    CountedStepTaker, GuardedStep, KeyedWalk, KeyspaceWalk, RankPageTaker, RankWalk, ServingChange,
    WalkError, WalkIncomplete, Witness, WitnessMissing, WALK_STEP_ATTEMPTS,
};
use crate::storage::time_sample_merge::{merge_signed_zero_rows, negative_zero_sections};

/// `COUNT` hint for every cursored enumeration in the pass. Matches the orphan
/// sweep's page size — same keyspace, same reason: large enough that a big
/// keyspace does not cost a round trip per handful of keys, small enough that a
/// single page cannot monopolise the server's event loop.
const SCAN_PAGE: u32 = 100;

/// How many candidate members one pass claims from `le:ext:{gc}:recon`.
///
/// Bounds a pass's destructive batch, exactly as the checkpoint reconciliation's
/// batch does. Anything above it keeps its first-observation score and waits for
/// the next pass, so a large backlog drains over several passes rather than in
/// one burst that would put thousands of members on `pending` at once.
const RECONCILE_BATCH: usize = 256;

/// Ceiling on how many repair members one pass emits for ONE hash.
///
/// A surplus this large is already pathological, and the cap bounds the blast
/// radius of a surplus computed from a walk that was wrong in some way the three
/// guards did not catch: at worst `EXT_RECON_MAX_REPAIR_PER_HASH` decrements per
/// pass rather than an unbounded burst. It costs nothing in the healthy case
/// (surpluses are ones and twos) and a genuinely huge surplus still drains — the
/// next pass re-observes the remainder, registers it as a new amount, and
/// repairs another slice a grace later.
const EXT_RECON_MAX_REPAIR_PER_HASH: u64 = 64;

/// How many `ext:meta` keys share one pipelined `HMGET` round trip.
const META_HMGET_CHUNK: usize = 128;

/// The `ext:meta` field carrying the Valkey-side millisecond timestamp of the
/// most recent `store_values` pre-flight increment for that hash.
///
/// Written by `scripts/ext_store_value.lua` on every pre-flight store (b.rej);
/// **read-only here**. It is the only one of the three guards that can
/// distinguish a standing surplus from an upload whose pre-flight increment has
/// not reached its commit yet.
///
/// Three states, three verdicts — see the module header's guard 3 for the full
/// argument. Present and older than the cutoff is eligible; present and newer
/// is `skipped_recent_store`; present and UNPARSEABLE is
/// `skipped_unreadable_stamp` (fail closed on corrupt keyspace); ABSENT is
/// **eligible**, because every store stamps, so an absent field is positive
/// evidence that nothing has pre-stored the hash since the writer deployed.
///
/// The byte spelling is a cross-file contract: the Lua writers (the store
/// script, and `ext_backup_confirm.lua`, which compares it) name the field
/// literally and this constant is the one Rust spelling, read here and by the
/// backup repair's meta walk (`storage::ext_backup`), so they must match
/// exactly.
pub(crate) const LAST_STORE_MS_FIELD: &str = "last_store_ms";

/// The `ext:meta` field carrying the live-object reference count.
const REF_COUNT_FIELD: &str = "ref_count";

/// What one meta's [`LAST_STORE_MS_FIELD`] reads as.
///
/// Three states, because post-b.rej the eligibility gate gives ABSENT and
/// UNPARSEABLE **opposite** verdicts and a two-state `Option<u64>` cannot carry
/// that distinction. Collapsing them — which is what the pre-b.rej
/// `parse::<u64>().ok()` did, when both were ineligible and the difference was
/// immaterial — would now read a corrupt value as "nothing ever stored this
/// hash" and license a repair on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StampState {
    /// No `last_store_ms` field on the meta. **Eligible**: every pre-flight
    /// store stamps, so absence is positive evidence that none has touched this
    /// hash since the writer deployed.
    Absent,
    /// The field is present but is not a decimal-millisecond integer — or the
    /// reply carried no slot for it at all, an arity `HMGET` never produces and
    /// which is therefore corruption-shaped rather than absence.
    /// **Ineligible**, `skipped_unreadable_stamp` — fail closed.
    Unreadable,
    /// The field parsed. Eligible iff it is older than the pass's cutoff.
    At(u64),
}

/// Classify the `last_store_ms` element of an `HMGET ref_count last_store_ms`
/// reply.
///
/// `field` is the raw slot from the decoded reply. The two `Option` layers mean
/// different things and get **different** verdicts:
///
///   * the OUTER one is "the reply had no such element" — an arity `HMGET` never
///     actually produces, since it answers one element per requested field. It
///     is therefore not absence but corruption-shaped: something other than the
///     `HMGET` this pass issued produced the reply. [`StampState::Unreadable`],
///     fail closed, for the same reason an unparseable value is — reading it as
///     absence would call it "nothing ever stored this hash" and license a
///     repair on the strength of a reply the pass could not account for.
///   * the INNER one is Valkey's nil for a field the hash does not carry, which
///     is the genuine [`StampState::Absent`].
fn classify_stamp(field: Option<&Option<String>>) -> StampState {
    match field {
        None => StampState::Unreadable,
        Some(None) => StampState::Absent,
        Some(Some(raw)) => match raw.parse::<u64>() {
            Ok(ms) => StampState::At(ms),
            Err(_) => StampState::Unreadable,
        },
    }
}

/// A [`PassError::Valkey`] for a command this pass issued on the maintenance
/// pool, counting it on `valkey_errors_total{kind="command_timeout"}` when it
/// timed out.
fn valkey_err(client: &ValkeyClient, op: &'static str, source: fred::error::Error) -> PassError {
    FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE).observe_error(&source);
    PassError::Valkey { op, source }
}

/// Anything that forfeits a pass. Every variant means the same thing to the
/// caller — publish nothing, repair nothing, tick `enumeration_failed` — and the
/// distinction exists only so the log line can name what went wrong.
#[derive(Debug)]
enum PassError {
    Valkey {
        op: &'static str,
        source: fred::error::Error,
    },
    /// The pass's own wall-clock budget elapsed mid-enumeration. Not an error in
    /// the keyspace: the pass simply may not hold a lease it can no longer prove
    /// it holds, and a half-finished walk is exactly the input that must never
    /// drive a repair.
    BudgetElapsed { phase: &'static str },
    /// The pass was cancelled mid-enumeration (shutdown). Forfeits for exactly
    /// the same reason [`PassError::BudgetElapsed`] does: a walk cut short
    /// produces a truncated map, and a truncated map under-counts the referenced
    /// side, which reads as surplus. "Fail closed, absolutely" includes this.
    Cancelled { phase: &'static str },
    /// A value the walk could not attribute to a hash. Aborts for the same
    /// reason a read error does: an unattributable reference is a reference the
    /// pass would silently omit from `referenced`, which reads as surplus.
    Undecodable { what: &'static str, text: String },
    /// A repair could not be published. Raised only from the mutating phase, and
    /// handled per-candidate rather than per-pass — the member keeps its
    /// first-observation score and is re-claimed next pass.
    Repair(String),
    /// A failure reading one object's references, in the authoritative walk
    /// only ([`in_object`]): `source`, with the object it was reading and,
    /// where the failing read names one key, that key. The periodic walk
    /// returns `source` bare.
    InObject {
        object_id: String,
        key: Option<String>,
        source: Box<PassError>,
    },
    /// A guarded walk crossed a change of serving process (the periodic
    /// pass's) or of the slot map, or its node pin missed its primary's slots
    /// (`storage::scan_guard`): its result may have skipped members. Not a
    /// fred error, so never counted on `valkey_errors_total`. The periodic
    /// pass forfeits; the restore Job's recount fails, exit 1. Boxed: the
    /// `NotOwner` case carries two slot sets.
    ServingChanged(Box<ServingChange>),
    /// A guarded walk's `INFO server` reported no usable run_id (the
    /// periodic pass's walks), or a keyspace walk's `CLUSTER NODES` no
    /// usable myself line (`storage::scan_guard`): the walk cannot be
    /// checked. Not a fred error. The periodic pass forfeits; the recount
    /// fails, exit 3.
    WitnessMissing(WitnessMissing),
    /// One of the recount's walks whose completeness proof did not hold
    /// (`storage::scan_guard`, *The DR Jobs' walks*): the store changed while
    /// the Job read it. Not a fred error. Only the recount's walks make it;
    /// the recount fails, exit 1, before any count is set. Boxed, as
    /// `WalkError` boxes it.
    Incomplete(Box<WalkIncomplete>),
}

/// A guarded walk's error ([`WalkError`]) as the pass's: a step that failed
/// transiently on every attempt, or deterministically on its first, is
/// `step_op`'s store fault, a guard read's failure its own
/// command's, both counted by [`valkey_err`]; a change, a missing witness or
/// an incomplete walk keeps its own variant and is not counted.
fn walk_err(client: &ValkeyClient, step_op: &'static str, e: WalkError) -> PassError {
    match e {
        WalkError::Step { source, .. } => valkey_err(client, step_op, source),
        WalkError::Read { op, source } => valkey_err(client, op, source),
        WalkError::Changed(change) => PassError::ServingChanged(change),
        WalkError::WitnessMissing(missing) => PassError::WitnessMissing(missing),
        WalkError::Incomplete(incomplete) => PassError::Incomplete(incomplete),
    }
}

/// `object {id}`, or `{key} (object {id})` when the key is known: what a
/// [`PassError::InObject`] names.
fn object_label(object_id: &str, key: Option<&str>) -> String {
    match key {
        Some(key) => format!("{key} (object {object_id})"),
        None => format!("object {object_id}"),
    }
}

impl std::fmt::Display for PassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassError::Valkey { op, source } => write!(f, "{op} failed: {source}"),
            PassError::BudgetElapsed { phase } => {
                write!(f, "the pass budget elapsed during {phase}")
            }
            PassError::Cancelled { phase } => {
                write!(f, "the pass was cancelled during {phase}")
            }
            PassError::Undecodable { what, text } => {
                write!(f, "could not decode {what}: {text:?}")
            }
            PassError::Repair(msg) => write!(f, "could not publish a repair: {msg}"),
            PassError::InObject {
                object_id,
                key,
                source,
            } => write!(f, "{source} ({})", object_label(object_id, key.as_deref())),
            PassError::ServingChanged(change) => write!(f, "{change}"),
            PassError::WitnessMissing(missing) => write!(f, "{missing}"),
            PassError::Incomplete(incomplete) => write!(f, "{incomplete}"),
        }
    }
}

/// The reconciler's derived grace window: `max(600 s, 10 × RPC deadline)`.
///
/// **Derived, never configured.** The grace is the safety guard, not a tuning
/// knob — an operator who shortened it would be shortening the window in which a
/// legitimately-in-flight increment is protected from being read as surplus, and
/// the failure that buys is data loss, not latency. It is bounded below by ten
/// minutes because that comfortably exceeds every in-flight window in the
/// service, and it scales with `LIVE_EDIT_RPC_DEADLINE_MS` because the longest
/// such window is the gap between a pre-store and the commit that justifies it,
/// which is bounded by the RPC deadline.
pub fn ext_reconcile_grace(rpc_deadline_ms: u64) -> Duration {
    Duration::from_secs(600).max(Duration::from_millis(rpc_deadline_ms.saturating_mul(10)))
}

/// Spawn the reference-count reconciler as a periodic background task.
///
/// Runs one pass immediately on start, then every `interval`. Each tick is gated
/// on the shared `le:gc:leader` lease — the same lease the GC sweep, the
/// orphan sweep, the external-value backup repair and the checkpoint clean-up
/// take, each on its own
/// interval (SDD §22.7.2's precedent: independent cadences, one lease). The GC loop's once-a-second pass is far too
/// frequent for a keyspace walk, which is why this does not ride it as a step.
///
/// **Safe under duplicate leadership by construction**, which is what riding
/// that 30 s-TTL lock demands of any consumer: candidate members are `ZADD NX`
/// so a second leader cannot re-pin a first observation, and repair members are
/// deterministic strings so a second leader's `SADD` collapses into the first's.
///
/// Exits cleanly on `cancel`. Cancellation during an ENUMERATION forfeits the
/// whole pass — no gauges, no candidates pinned, no repair — because a
/// truncated walk under-counts the referenced side (see "Fail closed,
/// absolutely"). Cancellation during the mutating phase merely stops at the next
/// candidate boundary: each repair is independently owed, so a partially-applied
/// phase 5 is not a hazard, and leaving the remainder for the next pass is free.
///
/// `leader` is the process-wide [`LeaderIdentity`] — the same one the GC task,
/// the orphan sweep, the external-value backup repair and the checkpoint
/// clean-up hold on
/// `le:gc:leader`. Its marked GC lock value
/// ([`LeaderIdentity::gc_lock_value`]) is what the lock stores; its
/// `service_label` alone labels this task's gauges.
pub fn spawn_ext_ref_reconcile(
    client: ValkeyClient,
    keys: KeySchema,
    leader: LeaderIdentity,
    interval: Duration,
    grace: Duration,
    cancel: CancellationToken,
    metrics: Option<SharedMetrics>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        tracing::info!(
            interval_secs = interval.as_secs(),
            grace_secs = grace.as_secs(),
            replica_id = %leader.service_label,
            lock_value = %leader.gc_lock_value(),
            "external-value ref_count reconciler started"
        );
        let replica_id = leader.service_label.as_str();

        loop {
            match try_acquire_gc_leader(&client, &keys, &leader, metrics.as_ref()).await {
                Ok(true) => {
                    run_one_pass(
                        &client,
                        &keys,
                        replica_id,
                        interval,
                        grace,
                        &cancel,
                        metrics.as_ref(),
                    )
                    .await;
                }
                Ok(false) => {
                    // Zero-publish from every non-leader. An OTel `Gauge` never
                    // decays, so a replica that held the lease, saw a surplus,
                    // and then lost the lease would keep exporting that surplus
                    // forever and pin the fleet-wide `max()` at a level that no
                    // longer exists.
                    if let Some(m) = metrics.as_ref() {
                        m.record_ext_reconcile_gauges(0, 0, 0, 0, replica_id);
                    }
                    tracing::debug!(
                        replica_id = %leader.service_label,
                        lock_value = %leader.gc_lock_value(),
                        "ref_count reconcile skipped: another replica holds the leader lock"
                    );
                }
                Err(e) => {
                    // Publish NOTHING on a failed probe — unknown is not zero.
                    if let Some(m) = metrics.as_ref() {
                        m.inc_gc_sweep_batch_error("leader_lock_error");
                    }
                    tracing::warn!(
                        replica_id = %leader.service_label,
                        lock_value = %leader.gc_lock_value(),
                        error = %e,
                        "ref_count reconcile leader lock acquisition failed"
                    );
                }
            }

            tokio::select! {
                () = tokio::time::sleep(interval) => {}
                () = cancel.cancelled() => {
                    tracing::info!("external-value ref_count reconciler shutting down");
                    return;
                }
            }
        }
    })
}

/// Run one pass, log its outcome, and publish its gauges. Never propagates: the
/// next interval retries.
async fn run_one_pass(
    client: &ValkeyClient,
    keys: &KeySchema,
    replica_id: &str,
    interval: Duration,
    grace: Duration,
    cancel: &CancellationToken,
    metrics: Option<&SharedMetrics>,
) {
    // Half the interval, minted once. The pass does not renew its 30 s-TTL
    // lease itself: while this replica's GC loop runs and makes progress, the
    // GC task's once-a-second refresh ticker, independent of every pass, keeps
    // it from lapsing while this pass runs, however long (b.1jt;
    // `external_values::spawn_gc_task`; the GC loop marks progress at least
    // once a second between its passes and within a few store round trips
    // inside one, even while this pass runs). If that loop has died, or
    // stalled for `external_values::GC_PROGRESS_STALL_AFTER` (20 s), the ticker
    // has stopped renewing and a peer takes the key over about 20 s after its
    // last refresh, possibly mid-pass — the behaviour before the ticker
    // existed. That is safe, because the pass is safe under two leaders (see
    // `spawn_ext_ref_reconcile`). What the budget bounds is the pass
    // overlapping its own NEXT tick, which would stack two walks of the whole
    // keyspace on one replica.
    let budget = interval / 2;
    let started = Instant::now();

    match run_pass(client, keys, grace, budget, started, cancel, metrics).await {
        Ok(summary) => {
            if let Some(m) = metrics {
                m.record_ext_reconcile_gauges(
                    summary.meta_total,
                    summary.referenced_hashes,
                    summary.surplus_total,
                    summary.deficit_total,
                    replica_id,
                );
            }
            tracing::info!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                meta_total = summary.meta_total,
                referenced_hashes = summary.referenced_hashes,
                surplus_total = summary.surplus_total,
                confirmed_deficit_total = summary.deficit_total,
                confirmed_deficit_hashes = summary.deficit_hash_count,
                deficits_observed_unconfirmed = summary.deficits_observed_unconfirmed,
                deficits_cleared = summary.deficits_cleared,
                candidates_registered = summary.candidates_registered,
                repaired_hashes = summary.repaired_hashes,
                repaired_references = summary.repaired_references,
                healed = summary.healed,
                index_mismatched_objects = summary.index_mismatched_objects,
                index_repaired_objects = summary.index_repaired_objects,
                index_unrepaired_objects = summary.index_unrepaired_objects,
                index_drift_skips = summary.index_drift_skips,
                time_sample_rows_merged = summary.time_sample_rows_merged,
                "external-value ref_count reconcile pass complete"
            );
        }
        // Cancellation forfeits the pass exactly as an error does — no gauges,
        // no repair — but it is an expected shutdown event, so it does NOT tick
        // `enumeration_failed`. That cell is alertable, and a counter that
        // moves on every rolling restart is a counter nobody can alert on.
        Err(PassError::Cancelled { phase }) => {
            tracing::debug!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                phase,
                "external-value ref_count reconcile pass cancelled; no repair was made"
            );
        }
        Err(e) => {
            // No gauge publish at all. A forfeited pass has no honest reading —
            // and its surplus figure in particular is exactly the number that
            // must never be believed, because a truncated object walk
            // under-counts `referenced`.
            if let Some(m) = metrics {
                m.inc_ext_reconcile(EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED, 1);
            }
            tracing::warn!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                error = %e,
                "external-value ref_count reconcile pass forfeited; no repair was made"
            );
        }
    }
}

/// What one completed pass observed. Only produced on the success path — a
/// forfeited pass yields no summary at all, deliberately.
#[derive(Debug, Default)]
struct PassSummary {
    meta_total: u64,
    referenced_hashes: u64,
    surplus_total: u64,
    /// Summed magnitude of the deficits this pass CONFIRMED in phase 5 — the
    /// only figure that reaches `live_edit_ext_ref_deficit_total`, whose
    /// documented alert shape is a deficit that RECURS over several intervals.
    /// Phase 4's raw negative surpluses do not appear here; they are registered
    /// for confirmation and counted in [`Self::deficits_observed_unconfirmed`].
    deficit_total: u64,
    /// How many distinct hashes CONFIRMED a deficit — the tick count for
    /// `deficit_observed`, as distinct from `deficit_total`, which sums the
    /// magnitude.
    deficit_hash_count: u64,
    /// How many distinct hashes phase 4 saw a negative surplus on, before any
    /// confirmation. Log-line only, deliberately: a busy store produces these
    /// routinely from the pass's own mid-pass double-counting, so this figure
    /// drives no gauge and no alertable cell.
    deficits_observed_unconfirmed: u64,
    /// How many claimed deficit candidates did NOT reproduce on the fresh
    /// re-read — the `deficit_cleared` tick count.
    deficits_cleared: u64,
    /// How many candidates this pass OFFERED to `le:ext:{gc}:recon` — the size
    /// of the set handed to `recon_claim_expired`, and the `candidate_registered`
    /// tick count. Re-offers are included: the pass rebuilds the candidate set
    /// from scratch each time and the `ZADD NX` that keeps a first-observation
    /// score runs Valkey-side, so this figure is a level (metas currently
    /// anomalous), never a count of newly-observed ones.
    candidates_registered: u64,
    repaired_hashes: u64,
    repaired_references: u64,
    healed: u64,
    /// Objects whose `obj:ext` disagreed with their raw row counts in phase 2's
    /// index audit, so a recount ran.
    index_mismatched_objects: u64,
    /// Of those, how many the recount repaired.
    index_repaired_objects: u64,
    /// Of those, how many it could not repair (too large, not live, or a
    /// malformed row reported, whatever the answer), whose hashes the drain
    /// gate held for this pass.
    index_unrepaired_objects: u64,
    /// Surpluses the drain gate held — metas in phase 4 plus claimed
    /// candidates in phase 5 — the `skipped_index_drift` tick count.
    index_drift_skips: u64,
    /// Time-sample rows phase 2's signed-zero merge left as one row at
    /// `+0.0` (`storage::time_sample_merge`): merged pairs plus re-keyed
    /// lone `-0.0` rows.
    time_sample_rows_merged: u64,
}

/// The phases, in the one order that is safe. See the module header —
/// "Phase order is load-bearing" — for why this sequence and not another.
async fn run_pass(
    client: &ValkeyClient,
    keys: &KeySchema,
    grace: Duration,
    budget: Duration,
    started: Instant,
    cancel: &CancellationToken,
    metrics: Option<&SharedMetrics>,
) -> Result<PassSummary, PassError> {
    // ── Phase 1: sample the clock, Valkey-side ────────────────────────────
    //
    // Same discipline as every script in this module's neighbourhood, and it
    // buys exactly one thing: no CLIENT clock enters the comparison. A local
    // `SystemTime::now()` compared against a Valkey-written `last_store_ms`
    // would let a clock-ahead service replica judge a just-stamped meta as
    // stale — stripping guard 3 from exactly the increment the guard protects —
    // and would leave every replica disagreeing about where the cutoff sits.
    //
    // What it does NOT do is make the comparison single-clock under Cluster.
    // `valkey_time_ms` issues a keyless `TIME` that lands on an arbitrary
    // primary, while `last_store_ms` was written by the node owning
    // `{hash_hex}`, so an inter-node residual remains. That residual costs the
    // guard only if the sampled node's clock runs AHEAD of the stamping node by
    // more than the grace (`max(600 s, 10 × LIVE_EDIT_RPC_DEADLINE_MS)`, so
    // ≥ 10 min): below that, a just-stored stamp still reads fresh and the
    // pre-store window stays closed. Cluster nodes must therefore be
    // time-synchronised well inside that bound — any NTP-synchronised fleet is,
    // by orders of magnitude — and `docs/service/docs/backing-store.md`
    // § Requirements states that as a deployment requirement.
    let pass_start_ms = valkey_time_ms(client).await?;

    // Everything downstream dates against this one instant, both in phase 4 and
    // again on the fresh re-read in phase 5, so the two answers cannot disagree
    // about which window "recent" means.
    let stamp_cutoff = pass_start_ms.saturating_sub(u64::try_from(grace.as_millis()).unwrap_or(0));

    // ── Phase 2: the object walk, and the per-object index audit ──────────
    //
    // The audit repairs, or gates, every object whose `obj:ext` disagrees with
    // its rows BEFORE phases 4 and 5 can drain anything — see `walk_objects`.
    let ObjectWalk {
        referenced, audit, ..
    } = walk_objects(
        client,
        keys,
        budget,
        started,
        cancel,
        metrics.map(|m| m.as_ref()),
        WalkMode::Periodic,
    )
    .await?;

    // ── Phases 2b, 3a, 3b: what is already owed, in flow order ────────────
    //
    // Obligations flow object → carrier → pending → inflight → decrement, and
    // each stage is walked BEFORE the stage it flows into, so a mid-pass
    // transition is double-counted rather than missed entirely. See the module
    // header; the ordering here is the whole of the argument's implementation
    // and must not be rearranged for tidiness.
    let mut owed: HashMap<[u8; 32], u64> = HashMap::new();
    // 2b — retirement retry carriers.
    walk_retire_carriers(client, keys, &mut owed, budget, started, cancel).await?;
    // 3a then 3b — pending, then inflight, inside `walk_owed`.
    walk_owed(client, keys, &mut owed, budget, started, cancel).await?;

    // ── Phase 4: the meta walk ────────────────────────────────────────────
    let (mut summary, candidates, skips) = walk_metas(
        client,
        keys,
        &referenced,
        &owed,
        &audit.unrepaired_drift,
        stamp_cutoff,
        budget,
        started,
        cancel,
    )
    .await?;
    summary.referenced_hashes = referenced.len() as u64;
    summary.index_mismatched_objects = audit.mismatched_objects;
    summary.index_repaired_objects = audit.repaired_objects;
    summary.index_unrepaired_objects = audit.unrepaired_objects;
    summary.index_drift_skips = skips.index_drift;
    summary.time_sample_rows_merged = audit.time_sample_rows_merged;

    // `deficit_observed` is deliberately NOT ticked here. A phase-4 negative
    // surplus is a first observation, and first observations are not evidence —
    // the pass's own phase order manufactures them. The tick happens in phase 5,
    // on the arm that has seen the same deficit twice a grace apart.
    if let Some(m) = metrics {
        m.inc_ext_reconcile(
            EXT_RECONCILE_OUTCOME_SKIPPED_RECENT_STORE,
            skips.recent_store,
        );
        m.inc_ext_reconcile(
            EXT_RECONCILE_OUTCOME_SKIPPED_UNREADABLE_STAMP,
            skips.unreadable_stamp,
        );
        m.inc_ext_reconcile(EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT, skips.index_drift);
    }

    // ── Phase 5: register, claim, act ─────────────────────────────────────
    //
    // Past this line the pass MUTATES, so it is also past the point where a
    // budget elapse may forfeit it — a repair half-emitted is not a hazard
    // (every member is independently owed), and abandoning the claim without
    // retiring the members would leave them claimable with a stale amount.
    summary.candidates_registered = candidates.len() as u64;
    let claimed = recon_claim_expired(
        client,
        &keys.ext_recon_candidates(),
        grace,
        RECONCILE_BATCH,
        &candidates,
    )
    .await
    // `recon_claim_expired` counts its own command timeouts.
    .map_err(|e| PassError::Valkey {
        op: "claim ext recon candidates",
        source: e,
    })?;

    if let Some(m) = metrics {
        m.inc_ext_reconcile(
            EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED,
            summary.candidates_registered,
        );
        // A claim that filled the batch means more was claimable than the batch
        // allowed; the remainder keeps its first-observation score and is
        // claimed next pass.
        if claimed.len() >= RECONCILE_BATCH {
            m.inc_ext_reconcile(EXT_RECONCILE_OUTCOME_DEFERRED, 1);
        }
    }

    for member in &claimed {
        if cancel.is_cancelled() {
            break;
        }
        match act_on_candidate(
            client,
            keys,
            member,
            &referenced,
            &owed,
            &audit.unrepaired_drift,
            stamp_cutoff,
        )
        .await
        {
            Ok(CandidateOutcome::Skipped { outcome }) => {
                // The fresh stamp read disagreed with the phase-4 one — the meta
                // was (re-)stamped between the two, or its stamp became
                // unreadable — or the hash is held by this pass's index-drift
                // gate. The member keeps its first-observation score and is
                // re-checked next pass; nothing was emitted.
                if outcome == EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT {
                    summary.index_drift_skips += 1;
                }
                if let Some(m) = metrics {
                    m.inc_ext_reconcile(outcome, 1);
                }
            }
            Ok(CandidateOutcome::Repaired { references }) => {
                summary.repaired_hashes += 1;
                summary.repaired_references += references;
                if let Some(m) = metrics {
                    m.inc_ext_reconcile(EXT_RECONCILE_OUTCOME_REPAIRED, 1);
                    m.inc_ext_ref_released(EXT_REF_RELEASE_REASON_RECONCILE, references);
                }
            }
            Ok(CandidateOutcome::Healed) => {
                summary.healed += 1;
                if let Some(m) = metrics {
                    m.inc_ext_reconcile(EXT_RECONCILE_OUTCOME_HEALED, 1);
                }
            }
            // The deficit arm's two terminal outcomes. Only the first reaches
            // the alertable gauge, and only after the same amount has stood for
            // a full grace.
            Ok(CandidateOutcome::DeficitConfirmed { deficit }) => {
                summary.deficit_total += deficit;
                summary.deficit_hash_count += 1;
                if let Some(m) = metrics {
                    m.inc_ext_reconcile(EXT_RECONCILE_OUTCOME_DEFICIT_OBSERVED, 1);
                }
            }
            Ok(CandidateOutcome::DeficitCleared) => {
                summary.deficits_cleared += 1;
                if let Some(m) = metrics {
                    m.inc_ext_reconcile(EXT_RECONCILE_OUTCOME_DEFICIT_CLEARED, 1);
                }
            }
            Err(e) => {
                // Per-candidate, not per-pass. Unlike the enumeration phases,
                // one candidate failing tells us nothing about the others: the
                // member stays on the ZSET with its original score, so the next
                // pass re-claims it and re-checks it against a fresh count.
                tracing::warn!(
                    member = %member,
                    error = %e,
                    "could not act on a ref_count reconcile candidate; it stays claimable"
                );
            }
        }
    }

    Ok(summary)
}

/// Read Valkey's own clock in milliseconds.
///
/// Keyless, so in a cluster it routes to an arbitrary primary — in general NOT
/// the node that wrote the `last_store_ms` this is compared against, which is
/// the one owning that hash's `{hash_hex}` slot. So this read removes the
/// CLIENT clock from the comparison, not all skew from it: an inter-node
/// residual survives, and is covered by the deployment requirement that cluster
/// nodes be time-synchronised (`docs/service/docs/backing-store.md`
/// § Requirements) — a requirement
/// any NTP-synchronised fleet clears by orders of magnitude against a grace of
/// ten minutes or more. See the phase-1 call site for what the residual would
/// have to exceed to matter. What this must NOT be is this process's local
/// clock.
async fn valkey_time_ms(client: &ValkeyClient) -> Result<u64, PassError> {
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let reply: Vec<String> = maint_pool
        .next()
        .custom(
            CustomCommand::new_static("TIME", ClusterHash::FirstKey, false),
            Vec::<Value>::new(),
        )
        .await
        .map_err(|e| valkey_err(client, "TIME", e))?;
    let secs = reply.first().and_then(|s| s.parse::<u64>().ok());
    let usecs = reply.get(1).and_then(|s| s.parse::<u64>().ok());
    match (secs, usecs) {
        (Some(s), Some(us)) => Ok(s.saturating_mul(1000).saturating_add(us / 1000)),
        _ => Err(PassError::Undecodable {
            what: "the TIME reply",
            text: reply.join(" "),
        }),
    }
}

/// Phase 2 — count, per hash, how many LIVE OBJECTS reference it.
///
/// Per object the answer is the **union** of two sources: the hashes its value
/// rows currently hold, and the fields of its `le:{oid}:obj:ext` index. That is
/// byte-for-byte what `schedule_old_object_ext_gc` releases at retirement, so
/// this pass and retirement can never disagree about what an object owes. The
/// union is deduped per object before it contributes, because `ref_count` counts
/// OBJECTS: an object holding one hash in twenty rows still holds exactly one
/// reference.
///
/// # Why the row walk is not optional
///
/// "The index is non-empty, so trust it" is wrong, and wrong in the dangerous
/// direction. A partially-seeded `obj:ext` records membership for the hashes some
/// commit happened to touch and omits the rest — the pre-b.qpf drift documented
/// on `KeySchema::obj_ext_index` — so trusting it alone would UNDER-count
/// `referenced`, and an under-counted `referenced` reads as surplus. The rows are
/// the ground truth the index is a cache of.
///
/// The raw `hash` column is read rather than the decoded `FieldValue`, unlike
/// every per-object reader of `obj:ext`, which counts a row with a `value`
/// column as holding no hash (`KeySchema::obj_ext_index`). On a pre-b.zvd
/// shadowed row — one carrying both a `value` and a stale `hash` column — the
/// raw read over-counts `referenced`, which suppresses a repair. Leak
/// direction, deliberately chosen.
///
/// # The per-object index audit
///
/// The same read also audits each object's `obj:ext`. Per object it compares
/// the stored index against the raw per-hash row counts it already holds
/// ([`ObjectReferences::index_mismatch`]), and on any disagreement runs the
/// atomic recount (`storage::ext_index_audit`), which repairs the index by the
/// decoder's rule and releases nothing. A recount that answers `too_large` or
/// `not_live`, or that reports any malformed `hash` column whatever it
/// answered (`clean`, `repaired` or `Withheld`; it applies raises only then),
/// is not authoritative for the object — such a row passed this walk's
/// `hash_from_hex`, so the decoder counts what the recount skips — and leaves
/// drift in place, so the object's mismatched hashes go
/// into the returned `unrepaired_drift` set, and phases 4 and 5 neither
/// register nor claim a surplus for them (`skipped_index_drift`). A recount
/// that fails forfeits the pass, like every other error in this walk.
///
/// This walk runs before the meta and claim phases of the same pass, and that
/// ordering is what makes the gate sufficient against the one producer it
/// targets. An index under-counted by a pre-index object's seeding commit
/// becomes a loss only through a drain — the surplus is the very increment
/// that keeps the blob alive while a live row still holds it — and every drain
/// happens in phase 5, after this walk has either repaired the object or gated
/// the hash. The guarantee is no loss through a drain this build performs,
/// from its first audit pass: a surplus a pre-fix or old-build reconciler
/// already drained leaves an unbacked under-count exposed until this walk
/// repairs it; a void-`WATCH` txn commit can overwrite this walk's repair
/// before phase 5 drains; and drift from a torn commit, or an older build's
/// signed-zero aliasing, has no surplus behind it and can over-release on a
/// commit path before any pass runs (module header).
///
/// # The signed-zero merge
///
/// Before reading an object's references the walk merges its legacy `-0.0`
/// time-sample rows (`storage::time_sample_merge`; module header, "The
/// signed-zero merge") and lists the object again, so the reference read and
/// the audit see the merged rows. The merge's own failures are counted and
/// logged inside it and never forfeit the pass; the second listing, like
/// every read of this walk, does.
///
/// # The authoritative mode
///
/// [`WalkMode::Authoritative`] (the restore Job's recount,
/// [`recount_authoritative`]) writes nothing: it runs neither the
/// signed-zero merge nor the index audit, and so holds no hash on drift. It
/// counts each object's references exactly as the periodic mode does — rows
/// ∪ `obj:ext` — and fails on every arm the periodic mode forfeits on (module
/// header, "No silent omission"). It also records the largest decimal object
/// id it walked, for the counter floor's surviving term. Its header walk is
/// the first of an agreeing pair (`storage::scan_guard`, *The DR Jobs'
/// walks*): it counts each object once, however often the walk returns its
/// header, and records the ids it counted; once it ends, a second walk of
/// the headers, reading only keys, must return exactly those ids, or the
/// walk fails ([`PassError::Incomplete`]) before the caller sets any count.
#[allow(clippy::too_many_arguments)]
async fn walk_objects(
    client: &ValkeyClient,
    keys: &KeySchema,
    budget: Duration,
    started: Instant,
    cancel: &CancellationToken,
    metrics: Option<&Metrics>,
    mode: WalkMode,
) -> Result<ObjectWalk, PassError> {
    let prefix = keys.prefix();
    // Same population and the same key shape the orphan sweep enumerates. The
    // braces are the literal Cluster hash tag, so the glob goes inside them.
    let pattern = keys.obj_header_scan_pattern();

    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let scan_client = maint_pool.next();
    const PHASE: &str = "the object walk";
    // The periodic pass's guarded keyspace walk
    // (`storage::scan_guard::KeyspaceWalk::start`): each primary's commands
    // go to one named node, every step's keys come back only once the
    // witness sent in its pipeline matched its primary's start, and the walk
    // ends only once its end fingerprint matched its start's. The recount's
    // is the first of an agreeing pair (`KeyspaceWalk::start_job`: the pins
    // and ownership checks, no witness), its result checked against a second
    // walk below. A change of serving process or slot map, or a node pin that
    // missed its primary's slots, ends either with an error, which forfeits
    // the pass (or fails the recount) like any walk error.
    let start_walk = async {
        match mode {
            WalkMode::Periodic => KeyspaceWalk::start(scan_client, &pattern).await,
            WalkMode::Authoritative => KeyspaceWalk::start_job(scan_client, &pattern).await,
        }
    };
    let mut walk = tokio::select! {
        biased;
        () = cancel.cancelled() => return Err(PassError::Cancelled { phase: PHASE }),
        walk = start_walk => {
            walk.map_err(|e| walk_err(client, OP_SCAN_HEADERS, e))?
        }
    };
    let mut page = Vec::new().into_iter();

    let mut referenced: HashMap<[u8; 32], u64> = HashMap::new();
    let mut audit = IndexAudit::default();
    let mut objects: u64 = 0;
    let mut max_object_id: Option<u64> = None;
    // The recount's: every object id it counted, once each, for the second
    // walk to agree with.
    let mut counted: HashSet<String> = HashSet::new();

    loop {
        // Cancellation forfeits rather than truncating — a short walk omits
        // every hash the objects it never reached reference, and an omitted
        // reference reads as surplus on each of them.
        let Some(fred_key) = page.next() else {
            let next = tokio::select! {
                // `biased` so a cancel that arrives while a step is in flight
                // is taken at once, not on a coin-flip: "cancellation aborts
                // the whole pass" is a claim the module header makes without
                // qualification, and random branch selection would make it a
                // claim about the average case.
                biased;
                () = cancel.cancelled() => return Err(PassError::Cancelled { phase: PHASE }),
                next = walk.next_page() => next,
            };
            // NOT `continue` on an error — see the module header. A skipped
            // object omits every hash it references, which reads as surplus
            // on each of them.
            match next.map_err(|e| walk_err(client, OP_SCAN_HEADERS, e))? {
                Some(keys) => {
                    page = keys.into_iter();
                    continue;
                }
                None => break,
            }
        };
        // And at every key between two steps.
        if cancel.is_cancelled() {
            return Err(PassError::Cancelled { phase: PHASE });
        }
        if started.elapsed() > budget {
            return Err(PassError::BudgetElapsed { phase: PHASE });
        }
        let Some(key) = fred_key.as_str() else {
            return Err(PassError::Undecodable {
                what: "an obj:hdr key",
                text: format!("{fred_key:?}"),
            });
        };
        let Some(object_id) =
            crate::storage::orphan_sweep::extract_hash_tag_id(key, prefix, ":obj:hdr")
        else {
            return Err(PassError::Undecodable {
                what: "an object_id from an obj:hdr key",
                text: key.to_string(),
            });
        };
        // The recount counts each object once: a header the walk returns
        // again (a SCAN may repeat a key) is skipped. The periodic pass keeps
        // its double count, which only errs high (a leak).
        if mode == WalkMode::Authoritative && !counted.insert(object_id.clone()) {
            continue;
        }

        let refs = match mode {
            // The restore Job's recount: the object's indices are read by
            // rank pages and counted `HSCAN` steps, each folded before the
            // next, so what it holds per object is one page, one row chunk
            // and one `HSCAN` step, however many rows it has
            // ([`object_referenced_hashes_paged`]). No
            // signed-zero merge (a write): it reads both rows of a `-0.0` /
            // `+0.0` pair, and its hash set counts their hashes once. A
            // failure reading the references names the object, and the key
            // where the failing read names one, for the Job's ERROR.
            WalkMode::Authoritative => {
                object_referenced_hashes_paged(client, keys, &object_id, ROW_HASH_CHUNK_ROWS)
                    .await
                    .map_err(|e| in_object(keys, &object_id, e))?
            }
            // The periodic pass keeps its whole listings: the signed-zero
            // merge needs the whole time-sample list. The merge runs before
            // the object's references are read: a legacy `-0.0` time-sample
            // row is folded into the `+0.0` key
            // (`storage::time_sample_merge`), and the object is listed again,
            // so the reference read and the index audit below see the merged
            // rows. Never forfeits the pass: its failures are counted and
            // logged inside the merge, and the walk goes on with the fresh
            // listing.
            WalkMode::Periodic => {
                let mut members = list_value_members(client, keys, &object_id).await?;
                let legacy_sections = negative_zero_sections(&members.time_samples);
                if !legacy_sections.is_empty() {
                    let merge =
                        merge_signed_zero_rows(client, keys, &object_id, &legacy_sections, metrics)
                            .await;
                    audit.time_sample_rows_merged += merge.rows_merged();
                    members = list_value_members(client, keys, &object_id).await?;
                }
                object_referenced_hashes(client, keys, &object_id, &members).await?
            }
        };
        for hash in &refs.hashes {
            *referenced.entry(*hash).or_insert(0) += 1;
        }
        objects += 1;
        if let Ok(id) = object_id.parse::<u64>() {
            max_object_id = Some(max_object_id.map_or(id, |max| max.max(id)));
        }

        // The index audit is a write, and its drift gate guards a drain the
        // authoritative mode never makes: that mode counts the object's
        // references as rows ∪ `obj:ext` above and moves on.
        if mode == WalkMode::Authoritative || refs.index_mismatch.is_empty() {
            continue;
        }
        audit.mismatched_objects += 1;
        // The recount ticks its own counters and logs a repair. Its outcome
        // here only decides whether the mismatched hashes are gated.
        let (maint_pool, _maint_token) = client.acquire_maintenance();
        match recount_ext_index(
            client,
            maint_pool.next(),
            POOL_LABEL_MAINTENANCE,
            keys,
            &object_id,
            EXT_INDEX_RECOUNT_SITE_RECONCILE,
            metrics,
        )
        .await
        {
            // Any malformed row, whatever the reply: gate. At this site a row
            // the recount calls malformed cannot be garbage — this walk has
            // already forfeited the pass on any `hash` column `hash_from_hex`
            // rejects — so it is one the decoder counts (uppercase or
            // mixed-case hex) and the recount skips. The recount's count is
            // then below the decoder's for that hash, and neither a `clean`
            // nor a `repaired` answer makes the index safe to drain against.
            // `Withheld` always carries malformed rows, so it lands here too.
            Ok(RecountOutcome::Clean { malformed_rows })
            | Ok(RecountOutcome::Repaired { malformed_rows, .. })
            | Ok(RecountOutcome::Withheld { malformed_rows, .. })
                if malformed_rows > 0 =>
            {
                audit.unrepaired_objects += 1;
                audit.unrepaired_drift.extend(refs.index_mismatch);
            }
            Ok(RecountOutcome::Repaired { .. }) => audit.repaired_objects += 1,
            Ok(RecountOutcome::Clean { .. }) => {}
            Ok(
                RecountOutcome::TooLarge { .. }
                | RecountOutcome::NotLive
                | RecountOutcome::Withheld { .. },
            ) => {
                audit.unrepaired_objects += 1;
                audit.unrepaired_drift.extend(refs.index_mismatch);
            }
            // Fail closed: the drift is neither repaired nor provably absent,
            // so no surplus may be drained on its strength. The recount
            // already counted a timeout, so this does not go through
            // `valkey_err`.
            Err(source) => {
                return Err(PassError::Valkey {
                    op: "EVALSHA ext_index_recount",
                    source,
                })
            }
        }
    }

    // The recount's second walk of the headers, started only now that the
    // first has answered its end (`storage::scan_guard`, *The DR Jobs'
    // walks*): it reads only keys, and must return exactly the objects the
    // first counted before any count is set. Each page's object ids go
    // straight into `walked`, so the walk holds its id set and one page.
    if mode == WalkMode::Authoritative {
        let mut second = KeyspaceWalk::start_job(scan_client, &pattern)
            .await
            .map_err(|e| walk_err(client, OP_SCAN_HEADERS, e))?;
        let mut walked: HashSet<String> = HashSet::with_capacity(counted.len());
        // `None` only once the walk's end checks held.
        while let Some(page) = second
            .next_page()
            .await
            .map_err(|e| walk_err(client, OP_SCAN_HEADERS, e))?
        {
            for fred_key in page {
                let Some(key) = fred_key.as_str() else {
                    return Err(PassError::Undecodable {
                        what: "an obj:hdr key",
                        text: format!("{fred_key:?}"),
                    });
                };
                let Some(object_id) =
                    crate::storage::orphan_sweep::extract_hash_tag_id(key, prefix, ":obj:hdr")
                else {
                    return Err(PassError::Undecodable {
                        what: "an object_id from an obj:hdr key",
                        text: key.to_string(),
                    });
                };
                walked.insert(object_id);
            }
        }
        keyspace_walks_agree(&pattern, &counted, &walked)
            .map_err(|e| walk_err(client, OP_SCAN_HEADERS, e))?;
    }

    Ok(ObjectWalk {
        referenced,
        audit,
        objects,
        max_object_id,
    })
}

/// Which pass the object walk serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WalkMode {
    /// The periodic reconcile pass: the signed-zero merge and the index audit
    /// run, and the audit's drift gate is filled.
    Periodic,
    /// The restore Job's recount ([`recount_authoritative`]): read-only, no
    /// merge, no audit, no gate.
    Authoritative,
}

/// What the object walk read.
#[derive(Debug, Default)]
struct ObjectWalk {
    /// Per hash, how many objects reference it (rows ∪ `obj:ext`).
    referenced: HashMap<[u8; 32], u64>,
    /// The index audit's findings (periodic mode only; empty otherwise).
    audit: IndexAudit,
    /// How many object headers the walk read.
    objects: u64,
    /// The largest decimal object id among them.
    max_object_id: Option<u64>,
}

/// What phase 2's per-object index audit found, for the drain gate and the
/// pass-complete log line.
#[derive(Debug, Default)]
struct IndexAudit {
    /// Hashes some live object's index disagreed with its rows on, where the
    /// recount could not repair it (`too_large` / `not_live`, or any reply
    /// reporting a malformed row). Phases 4 and 5 never register or claim a
    /// surplus for these. Lives for one pass.
    unrepaired_drift: HashSet<[u8; 32]>,
    /// Objects whose index disagreed with the raw row counts.
    mismatched_objects: u64,
    /// Of those, how many the recount repaired (no malformed row reported).
    repaired_objects: u64,
    /// Of those, how many it could not (too large, not live, or a malformed
    /// row reported, whatever the answer).
    unrepaired_objects: u64,
    /// Not part of the index audit, carried here for the pass-complete log
    /// line: time-sample rows the walk's signed-zero merge left as one row at
    /// `+0.0` (merged pairs and re-keyed lone `-0.0` rows).
    time_sample_rows_merged: u64,
}

/// The object walk's per-object reads, by the `op` or `what` its
/// [`PassError`] carries: the periodic pass's whole listings and index read,
/// and the restore Job's recount's rank pages (`ZRANGE`, as the listings)
/// and `HSCAN` steps, whose failures [`in_object`] locates at their key.
const OP_LIST_FIELDS: &str = "ZRANGE obj:flds";
const OP_LIST_TIME_SAMPLES: &str = "ZRANGE obj:tss";
const OP_READ_EXT_INDEX: &str = "HGETALL obj:ext";
const OP_DECODE_EXT_INDEX: &str = "decode HGETALL obj:ext";
const OP_SCAN_EXT_INDEX: &str = "HSCAN obj:ext";
/// The keyspace walks' and the owed walks' steps, by the `op` their
/// [`PassError`] carries: the periodic pass's `ZSCAN` of `inflight`, the
/// recount's `ZRANGE` pages of it.
const OP_SCAN_HEADERS: &str = "SCAN obj:hdr";
const OP_SCAN_CARRIERS: &str = "SCAN obj:extret";
const OP_SCAN_PENDING: &str = "SSCAN ext_pending";
const OP_SCAN_INFLIGHT: &str = "ZSCAN ext_inflight";
const OP_PAGE_INFLIGHT: &str = "ZRANGE ext_inflight";
const WHAT_EXT_INDEX_FIELD: &str = "an obj:ext field during the reconcile object walk";

/// The authoritative walk's failure reading `object_id`'s references: the
/// error with the object and, where the failing read names one key, that key
/// ([`PassError::InObject`]): a `ZRANGE` page of `obj:flds` or `obj:tss`, an
/// `HSCAN` step of `obj:ext`, or an `obj:ext` field that is not a hash. The
/// row-hash pipelines' failures (a server error on any row, a reply that
/// does not decode, a wrong arity, a row `hash` column that is not a hash)
/// name the object alone: a pipeline answers for all its keys at once. A
/// count check that did not hold ([`PassError::Incomplete`]) names its key
/// itself.
fn in_object(keys: &KeySchema, object_id: &str, source: PassError) -> PassError {
    let key = match &source {
        PassError::Valkey { op, .. } => match *op {
            OP_LIST_FIELDS => Some(keys.obj_fields_index(object_id)),
            OP_LIST_TIME_SAMPLES => Some(keys.obj_time_samples_index(object_id)),
            OP_SCAN_EXT_INDEX => Some(keys.obj_ext_index(object_id)),
            _ => None,
        },
        PassError::Undecodable { what, .. } if *what == WHAT_EXT_INDEX_FIELD => {
            Some(keys.obj_ext_index(object_id))
        }
        _ => None,
    };
    PassError::InObject {
        object_id: object_id.to_owned(),
        key,
        source: Box::new(source),
    }
}

/// An object's value-row index members, as the periodic walk lists them.
struct ValueMembers {
    /// `obj:flds` members, `{section_id}:{hex(key_name)}`.
    fields: Vec<String>,
    /// `obj:tss` members, `{section_id}:{hex_f64(time)}`.
    time_samples: Vec<String>,
}

/// List an object's `obj:flds` and `obj:tss` members, each whole in one
/// `ZRANGE 0 -1`: the periodic pass's listing, which its signed-zero merge
/// needs whole. The restore Job's recount pages them instead
/// ([`object_referenced_hashes_paged`]).
async fn list_value_members(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
) -> Result<ValueMembers, PassError> {
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let conn = maint_pool.next();

    let fields: Vec<String> = conn
        .zrange::<Vec<String>, _, _, _>(
            keys.obj_fields_index(object_id),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .map_err(|e| valkey_err(client, OP_LIST_FIELDS, e))?;
    let time_samples: Vec<String> = conn
        .zrange::<Vec<String>, _, _, _>(
            keys.obj_time_samples_index(object_id),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .map_err(|e| valkey_err(client, OP_LIST_TIME_SAMPLES, e))?;
    Ok(ValueMembers {
        fields,
        time_samples,
    })
}

/// What one object's walk read: the hashes it references, and the hashes whose
/// per-object index field disagrees with the raw row count.
struct ObjectReferences {
    /// The distinct external hashes the object references — rows ∪ `obj:ext`.
    hashes: HashSet<[u8; 32]>,
    /// The hashes whose `obj:ext` field is not exactly the number of rows whose
    /// raw `hash` column names it: a field absent for a hash some row holds, a
    /// field present for a hash no row holds, a stored count that differs, or a
    /// stored value that is not a positive decimal count. Empty when the index
    /// agrees with the rows, and always in the restore Job's recount, which
    /// runs no index audit ([`object_referenced_hashes_paged`]).
    ///
    /// Compared on the RAW `hash` column, at no read beyond the union's own,
    /// while the index counts by the decoder's rule (a row with a `value`
    /// column holds no hash). Raw is never below the decoder's count, so an
    /// index that under-counts always disagrees with raw and is never missed;
    /// a legacy dual-shape row makes raw read high, which triggers a recount
    /// that answers `clean` (or can mask an over-count, which only leaks).
    index_mismatch: Vec<[u8; 32]>,
}

/// The most row-hash reads one pipeline of the object walk queues
/// ([`fold_row_hash_chunk`]): 1,024 rows. Every pipelined command holds
/// client memory until the pipeline's replies are read, about 2.5 KB each
/// [measured: the recount held 1,707 MiB of heap reading a 194,000-spec
/// object's 719,922 rows in one pipeline, record
/// `restore-memory-envelope-tv3-20261022`], so a chunk holds about 2.5 MB
/// whatever the object's size. The cost is one round trip per chunk: 704 for
/// that object (⌈719,922 / 1,024⌉).
///
/// In the restore Job's recount it is also the page: each `ZRANGE` page of
/// `obj:flds` and `obj:tss` (and of `inflight`) spans this many ranks, so a
/// page's rows are one chunk, and each `HSCAN` of `obj:ext` asks for this
/// many elements ([`object_referenced_hashes_paged`]). The periodic pass's
/// two listings stay whole ([`list_value_members`]): its signed-zero merge
/// needs the whole time-sample list.
pub const ROW_HASH_CHUNK_ROWS: usize = 1_024;

/// The objects whose row-hash chunks lose their last reply, by their `obj:ext`
/// key ([`arm_row_hash_reply_drop_for_test`]). Compiled out of release builds.
#[cfg(any(test, feature = "test-support"))]
static ROW_HASH_REPLY_DROP: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Test seam: every row-hash chunk the walk reads for the object whose
/// `obj:ext` key is `ext_index_key` loses its last reply, as a misaligned
/// pipeline reply would, until the returned guard drops; so a test can see
/// the walk's arity check fail closed. Keyed by the prefixed key, so tests
/// sharing a process do not reach each other's objects. Compiled out of
/// release builds.
#[cfg(any(test, feature = "test-support"))]
pub fn arm_row_hash_reply_drop_for_test(ext_index_key: &str) -> ArmedRowHashReplyDrop {
    ROW_HASH_REPLY_DROP
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(ext_index_key.to_owned());
    ArmedRowHashReplyDrop {
        ext_index_key: ext_index_key.to_owned(),
    }
}

/// The armed seam of [`arm_row_hash_reply_drop_for_test`]; disarms its
/// object when dropped. Bind it (`let _armed = …`) for as long as the walk
/// should be cut. Compiled out of release builds.
#[cfg(any(test, feature = "test-support"))]
#[must_use = "the seam is disarmed when this guard drops"]
#[derive(Debug)]
pub struct ArmedRowHashReplyDrop {
    ext_index_key: String,
}

#[cfg(any(test, feature = "test-support"))]
impl Drop for ArmedRowHashReplyDrop {
    fn drop(&mut self) {
        let mut armed = ROW_HASH_REPLY_DROP
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(at) = armed.iter().position(|key| *key == self.ext_index_key) {
            armed.swap_remove(at);
        }
    }
}

/// `replies` less its last reply when the seam is armed for the object whose
/// `obj:ext` key is `ext_index_key`; `replies` unchanged otherwise.
#[cfg(any(test, feature = "test-support"))]
fn drop_a_row_hash_reply_if_armed(ext_index_key: &str, mut replies: Vec<Value>) -> Vec<Value> {
    let armed = ROW_HASH_REPLY_DROP
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .any(|key| key == ext_index_key);
    if armed {
        replies.pop();
    }
    replies
}

/// The periodic pass's read of one object's references: the distinct
/// external hashes it references — rows ∪ `obj:ext` — and the hashes whose
/// index field disagrees with the rows, for the rows `members` lists
/// ([`list_value_members`]), the rows read [`ROW_HASH_CHUNK_ROWS`] at a time
/// ([`object_referenced_hashes_in_chunks`]).
async fn object_referenced_hashes(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    members: &ValueMembers,
) -> Result<ObjectReferences, PassError> {
    object_referenced_hashes_in_chunks(client, keys, object_id, members, ROW_HASH_CHUNK_ROWS).await
}

/// [`object_referenced_hashes`], the rows read `chunk_rows` at a time
/// ([`fold_row_hash_chunk`]), each chunk folded into the hash set and the raw
/// per-hash row counts before the next is queued, so the client holds one
/// chunk's commands and replies at once, not the object's. Then the index's
/// `HGETALL`, alone, once every row is counted: the reads are no more atomic
/// with each other than the one pipeline they replace was.
///
/// What the periodic walk still holds per object grows with it, all of it far
/// below the pipeline it replaces: the two index listings
/// ([`list_value_members`]), the index reply, one field per distinct hash the
/// index names, and the hash set and the per-hash row counts, one entry per
/// distinct hash. The restore Job's recount holds none of the per-row terms
/// ([`object_referenced_hashes_paged`]).
async fn object_referenced_hashes_in_chunks(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    members: &ValueMembers,
    chunk_rows: usize,
) -> Result<ObjectReferences, PassError> {
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let conn = maint_pool.next();

    let mut hashes: HashSet<[u8; 32]> = HashSet::new();
    // Raw per-hash row counts, for the index audit below.
    let mut row_counts: HashMap<[u8; 32], u64> = HashMap::new();

    // Each chunk's row keys, built as the chunk is queued: the field rows',
    // then the time-sample rows'.
    let mut rows = members
        .fields
        .iter()
        .map(|member| keys.obj_field_from_member(object_id, member))
        .chain(
            members
                .time_samples
                .iter()
                .map(|member| keys.obj_ts_from_member(object_id, member)),
        );
    loop {
        let chunk: Vec<String> = rows.by_ref().take(chunk_rows.max(1)).collect();
        if chunk.is_empty() {
            break;
        }
        fold_row_hash_chunk(
            client,
            conn,
            keys,
            object_id,
            chunk,
            &mut hashes,
            Some(&mut row_counts),
        )
        .await?;
    }

    // The index: one field per distinct hash the object's index names.
    let index_reply: Value = conn
        .hgetall(keys.obj_ext_index(object_id))
        .await
        .map_err(|e| valkey_err(client, OP_READ_EXT_INDEX, e))?;
    let raw_index: HashMap<String, String> = index_reply
        .convert()
        .map_err(|e| valkey_err(client, OP_DECODE_EXT_INDEX, e))?;
    let mut index_mismatch: Vec<[u8; 32]> = Vec::new();
    let mut indexed: HashSet<[u8; 32]> = HashSet::with_capacity(raw_index.len());
    for (hash_hex, stored) in &raw_index {
        match hash_from_hex(hash_hex) {
            Some(hash) => {
                hashes.insert(hash);
                indexed.insert(hash);
                // "Present ⟺ count > 0": a field agrees only when it stores
                // exactly the (non-zero) number of rows naming the hash.
                let rows = row_counts.get(&hash).copied().unwrap_or(0);
                if rows == 0 || stored.parse::<u64>().ok() != Some(rows) {
                    index_mismatch.push(hash);
                }
            }
            None => {
                return Err(PassError::Undecodable {
                    what: WHAT_EXT_INDEX_FIELD,
                    text: hash_hex.clone(),
                })
            }
        }
    }
    // A hash some row holds and the index has no field for.
    for hash in row_counts.keys() {
        if !indexed.contains(hash) {
            index_mismatch.push(*hash);
        }
    }

    Ok(ObjectReferences {
        hashes,
        index_mismatch,
    })
}

/// Fold one chunk of an object's row keys, `row_keys`, into `hashes`: one
/// pipelined `HGET <row> hash` per row on `conn`, its replies folded before
/// the caller queues the next chunk, and, with `row_counts`, each hash's raw
/// row count (the periodic pass's index audit). The one chunk fold of both
/// walk modes ([`object_referenced_hashes_in_chunks`],
/// [`object_referenced_hashes_paged`]).
///
/// The raw `hash` column decides: a row with no `hash` holds no external
/// value, and a dual-shape row's `hash` is counted (a high count only leaks).
/// The reply's arity is checked, and a column `hash_from_hex` rejects fails
/// closed.
async fn fold_row_hash_chunk(
    client: &ValkeyClient,
    conn: &fred::clients::Client,
    keys: &KeySchema,
    object_id: &str,
    row_keys: Vec<String>,
    hashes: &mut HashSet<[u8; 32]>,
    mut row_counts: Option<&mut HashMap<[u8; 32], u64>>,
) -> Result<(), PassError> {
    let rows = row_keys.len();
    let pipeline = conn.pipeline();
    for row_key in row_keys {
        let _: () = pipeline
            .hget(row_key, "hash")
            .await
            .map_err(|e| valkey_err(client, "queue HGET row hash", e))?;
    }
    let replies: Vec<Value> = pipeline
        .all()
        .await
        .map_err(|e| valkey_err(client, "HGET row hashes", e))?;
    #[cfg(any(test, feature = "test-support"))]
    let replies = drop_a_row_hash_reply_if_armed(&keys.obj_ext_index(object_id), replies);
    #[cfg(not(any(test, feature = "test-support")))]
    let _ = (keys, object_id);

    // A misaligned reply vector would attribute one row's hash to another
    // row, so each chunk's length is checked rather than assumed — same
    // posture as `read_ext_pre_image`.
    if replies.len() != rows {
        return Err(PassError::Undecodable {
            what: "a row-hash pipeline reply of the expected arity",
            text: format!("{} frames for {rows} rows", replies.len()),
        });
    }

    for reply in replies {
        let hash_hex: Option<String> = reply
            .convert()
            .map_err(|e| valkey_err(client, "decode HGET row hash", e))?;
        let Some(hash_hex) = hash_hex else { continue };
        match hash_from_hex(&hash_hex) {
            Some(hash) => {
                hashes.insert(hash);
                if let Some(counts) = row_counts.as_deref_mut() {
                    *counts.entry(hash).or_insert(0) += 1;
                }
            }
            // Fail closed. A row whose hash cannot be named is a reference
            // this walk would silently omit, and an omitted reference reads
            // as surplus on the blob it points at. Forfeiting the pass leaves
            // the surplus standing, which is the safe direction; the WARN in
            // `run_one_pass` names the failure so the corruption is visible.
            // The recount fails, exit 3.
            None => {
                return Err(PassError::Undecodable {
                    what: "a value row's hash column during the reconcile object walk",
                    text: hash_hex,
                })
            }
        }
    }
    Ok(())
}

/// The restore Job's recount's read of one object's references: the distinct
/// external hashes it references, rows ∪ `obj:ext`, with no index audit
/// (`index_mismatch` stays empty). `obj:flds` then `obj:tss` are read by
/// ascending `ZRANGE` pages of `page` ranks with `ZCARD` in each page's
/// pipeline (`storage::scan_guard::page_by_rank_counted`), each page's
/// members turned into row keys and read in chunks of at most `page` rows
/// ([`fold_row_hash_chunk`]), so one chunk a page; then `obj:ext` by counted
/// `HSCAN … COUNT page` steps with `HLEN` in each step's pipeline
/// (`storage::scan_guard::walk_counted`), each field de-duplicated by its
/// text and decoded as a hash. Every page and step is folded into the object's
/// hash set before the next is issued, on the maintenance pool: the three
/// walks on one client of it, the row chunks on any client.
///
/// Complete because the Job runs it on a held, quiet store, where rank order
/// and cardinality are the same on any process (`storage::scan_guard`, *The
/// DR Jobs' walks*): each index is complete only when its pages returned the
/// `ZCARD` read with the first page and the last page's `ZCARD` equals it,
/// and `obj:ext` only when its distinct fields equal the `HLEN` read with
/// its first and its last step; otherwise the read fails
/// ([`PassError::Incomplete`]) before the caller sets any count.
///
/// What it holds per object, whatever the object's row count: one page's
/// members, one chunk's commands and replies, and one `HSCAN` step; beside
/// them, one entry per distinct hash in the object's set and the pass's
/// maps, and one per distinct `obj:ext` field in its fields' set. A step
/// whose reply does not pair, a
/// page's or a step's error, a row hash or an `obj:ext` field that does not
/// decode fails the read closed; the caller names the object and, for a
/// page or a step, its key ([`in_object`]).
async fn object_referenced_hashes_paged(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    page: usize,
) -> Result<ObjectReferences, PassError> {
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    // The one client every walk of this object holds for its pages and
    // steps.
    let walk_client = maint_pool.next();
    let page = page.max(1);
    let mut hashes: HashSet<[u8; 32]> = HashSet::new();
    for (index_key, time_samples, op) in [
        (keys.obj_fields_index(object_id), false, OP_LIST_FIELDS),
        (
            keys.obj_time_samples_index(object_id),
            true,
            OP_LIST_TIME_SAMPLES,
        ),
    ] {
        page_by_rank_counted(
            &RankWalk::new(walk_client, &index_key, false),
            page as u64,
            &mut RecountRowPages {
                client,
                pool: maint_pool,
                keys,
                object_id,
                time_samples,
                op,
                chunk: page,
                hashes: &mut hashes,
            },
        )
        .await?;
    }
    walk_counted(
        &CountedScan::hash(walk_client, &keys.obj_ext_index(object_id)),
        page as u64,
        &mut RecountExtFields {
            client,
            fields: HashSet::new(),
            hashes: &mut hashes,
        },
    )
    .await?;
    Ok(ObjectReferences {
        hashes,
        index_mismatch: Vec::new(),
    })
}

/// The recount's pages of one value index ([`object_referenced_hashes_paged`]):
/// each page's members become row keys, read `chunk` rows to a pipeline.
struct RecountRowPages<'w> {
    client: &'w ValkeyClient,
    /// The maintenance pool, for the row chunks.
    pool: &'w fred::clients::Pool,
    keys: &'w KeySchema,
    object_id: &'w str,
    /// `obj:tss` (else `obj:flds`): how a member names its row.
    time_samples: bool,
    op: &'static str,
    chunk: usize,
    hashes: &'w mut HashSet<[u8; 32]>,
}

/// The pass's error for a guarded keyed walk whose witness changed from
/// `start` to `later` (the cursor pager's [`CursorPageReader::changed`]).
fn changed_error(walk: &KeyedWalk<'_>, start: &Witness, later: &Witness) -> PassError {
    PassError::ServingChanged(Box::new(walk.change(start, later)))
}

impl RankPageTaker for RecountRowPages<'_> {
    type Error = PassError;

    fn walk_error(&self, e: WalkError) -> PassError {
        walk_err(self.client, self.op, e)
    }

    async fn take_page(&mut self, members: Vec<Value>) -> Result<(), PassError> {
        let members: Vec<String> = Value::Array(members)
            .convert()
            .map_err(|e| valkey_err(self.client, self.op, e))?;
        let (keys, object_id) = (self.keys, self.object_id);
        let time_samples = self.time_samples;
        // The page's members become its row keys, each member dropped as its
        // key is built.
        let mut rows = members.into_iter().map(|member| {
            if time_samples {
                keys.obj_ts_from_member(object_id, &member)
            } else {
                keys.obj_field_from_member(object_id, &member)
            }
        });
        loop {
            let chunk: Vec<String> = rows.by_ref().take(self.chunk.max(1)).collect();
            if chunk.is_empty() {
                return Ok(());
            }
            fold_row_hash_chunk(
                self.client,
                self.pool.next(),
                keys,
                object_id,
                chunk,
                self.hashes,
                None,
            )
            .await?;
        }
    }
}

/// The recount's counted steps over `obj:ext`
/// ([`object_referenced_hashes_paged`]): each `HSCAN` step's fields,
/// de-duplicated by their exact text into `fields` (whose size the count
/// check compares with `HLEN`, which counts fields by their text), each
/// decoded as a hash and folded into the object's set. A hash `obj:ext`
/// holds in two letter cases (`hash_from_hex` accepts either) is two fields
/// and one hash.
struct RecountExtFields<'w> {
    client: &'w ValkeyClient,
    /// The distinct fields taken, by their text.
    fields: HashSet<String>,
    hashes: &'w mut HashSet<[u8; 32]>,
}

impl CountedStepTaker for RecountExtFields<'_> {
    type Error = PassError;

    fn walk_error(&self, e: WalkError) -> PassError {
        walk_err(self.client, OP_SCAN_EXT_INDEX, e)
    }

    async fn take_step(&mut self, members: Vec<Value>) -> Result<(), PassError> {
        let fields: Vec<String> = Value::Array(members)
            .convert()
            .map_err(|e| valkey_err(self.client, OP_SCAN_EXT_INDEX, e))?;
        for field in fields {
            match hash_from_hex(&field) {
                Some(hash) => {
                    self.hashes.insert(hash);
                    self.fields.insert(field);
                }
                // A field the recount cannot name is a reference it would
                // omit: exit 3. The field's count is not read; a field whose
                // count is `0` or not decimal still names its hash.
                None => {
                    return Err(PassError::Undecodable {
                        what: WHAT_EXT_INDEX_FIELD,
                        text: field,
                    })
                }
            }
        }
        Ok(())
    }

    fn distinct(&self) -> u64 {
        self.fields.len() as u64
    }
}

/// Phase 2b — fold the retirement intents' carriers into `owed`.
///
/// `le:{oid}:obj:extret` holds the release members a retirement owes, written
/// before its enqueue and removed only once an attempt is known to have landed
/// (b.yda; write-ahead since b.yuo Part R,
/// `storage::layer::schedule_old_object_ext_gc`). Those are obligations the GC
/// is or will be handed, so they belong in `owed` exactly as a `pending`
/// member does.
///
/// # Why omitting them is an over-release, not a rounding error
///
/// A carrier normally belongs to an object that has already been torn down (the
/// exception is below), so it contributes NOTHING to `referenced` — the object
/// walk cannot see it, there is
/// no `obj:hdr` left to scan. Without this walk its entire amount reads as pure
/// surplus on the first pass that sees it, and it is stable (the carrier does
/// not change while it waits), so it survives the two-observation guard and is
/// repaired. The GC would then decrement once for the repair and once more when
/// the sweep finally drains the carrier: the obligation paid twice, against a
/// reference given up once.
///
/// Runs BEFORE the pending and inflight walks, because a carrier's members flow
/// INTO pending. See the module header.
///
/// That ordering gives the usual "double-counted, never missed" guarantee here
/// too: a carrier is deleted only after its members are on `pending` (or after
/// a cancel, when nothing is owed, or the drop of an older build's carrier,
/// which leaks at worst), so an obligation moving between the two
/// walks is seen by both, never by neither. Module header, "Phase order is
/// load-bearing".
///
/// # Both carrier types
///
/// This build writes carriers as HASHes (member → 1); an older build's are
/// SETs, at the same key (b.1jt). Both are owed, so the walk reads each by its
/// type (`TYPE`, then `HKEYS` or `SMEMBERS`), and a carrier gone since the
/// SCAN saw it counts nothing. A key of any other type forfeits the pass: an
/// unread carrier would under-count `owed`, the over-release direction.
///
/// # A carrier whose object is live again, and why this walk still needs no
/// special case
///
/// A carrier can stand beside a live object in two ways. Every retirement
/// writes its carrier while its object is still live, and removes it once its
/// enqueue lands, normally milliseconds later. And a carrier can outlive its
/// object's death: a checkpoint restore, or a teardown that never completed, can
/// put the same `object_id` back in service while its carrier still stands. The
/// orphan sweep's settle CANCELS such a carrier rather than paying it — the
/// members are dropped, because the live incarnation inherited the standing
/// `+1` they stand for (`scripts/ext_retire_settle.lua`) — so the amount is
/// never actually owed.
///
/// Until the retirement clears its carrier or the settle cancels it, this walk
/// folds the carrier into `owed` while the live object also contributes to
/// `referenced`, and the same reference is counted on both sides. The
/// over-statement is bounded by the retirement's own duration, or by the time
/// the orphan sweep takes to settle the carrier, and self-clears with no code
/// here, because the clear or the cancel DELetes the carrier and the next pass
/// does not see it. Its effect for that one pass is a larger `owed`, which
/// shrinks `surplus`: usually that just withholds a repair (the safe
/// direction), and where it pushes `surplus` below zero it produces a deficit
/// reading that did not happen.
///
/// That last case is why the deficit arm confirms rather than reports. A
/// spurious deficit from this source is transient, so it cannot survive a full
/// grace at the same amount, and the confirm arm in [`act_on_candidate`]
/// retires it as `deficit_cleared` without ever touching the alertable gauge.
/// Filtering live ids out here is still not the remedy: it would be a second
/// liveness gate, racing the same restore from a different slot, where the
/// settle's gate is atomic with the thing it protects.
///
/// Fails closed on every error shape, like its sibling walks — and unlike the
/// orphan sweep's own `continue`-on-SCAN-error backstop over the same keys,
/// which can only defer work where this can only over-release.
async fn walk_retire_carriers(
    client: &ValkeyClient,
    keys: &KeySchema,
    owed: &mut HashMap<[u8; 32], u64>,
    budget: Duration,
    started: Instant,
    cancel: &CancellationToken,
) -> Result<(), PassError> {
    let prefix = keys.prefix();
    // The braces are the literal Cluster hash tag, so the glob goes inside them.
    let pattern = format!("{prefix}{{*}}:obj:extret");

    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let conn = maint_pool.next();
    const PHASE: &str = "the retire-carrier walk";
    // The guarded keyspace walk, as in the object walk.
    let mut walk = tokio::select! {
        biased;
        () = cancel.cancelled() => return Err(PassError::Cancelled { phase: PHASE }),
        walk = KeyspaceWalk::start(conn, &pattern) => {
            walk.map_err(|e| walk_err(client, OP_SCAN_CARRIERS, e))?
        }
    };
    let mut page = Vec::new().into_iter();

    loop {
        let Some(fred_key) = page.next() else {
            let next = tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(PassError::Cancelled { phase: PHASE }),
                next = walk.next_page() => next,
            };
            match next.map_err(|e| walk_err(client, OP_SCAN_CARRIERS, e))? {
                Some(keys) => {
                    page = keys.into_iter();
                    continue;
                }
                None => break,
            }
        };
        if cancel.is_cancelled() {
            return Err(PassError::Cancelled { phase: PHASE });
        }
        if started.elapsed() > budget {
            return Err(PassError::BudgetElapsed { phase: PHASE });
        }
        tally_carrier(client, conn, &fred_key, owed).await?;
    }

    Ok(())
}

/// The restore Job's recount's carrier walk: the carriers' keys by an
/// agreeing keyspace walk (`storage::scan_guard::collect_keyspace_agreed`:
/// two walks of the pattern, the second after the first ends, used only
/// when both returned the same set), then each carrier read whole into
/// `owed` as the periodic walk reads it ([`tally_carrier`]). Two walks that
/// disagree fail it ([`PassError::Incomplete`]) before any carrier is read.
/// No budget and no cancellation: the Job ends the recount by dropping it.
async fn walk_retire_carriers_agreed(
    client: &ValkeyClient,
    keys: &KeySchema,
    owed: &mut HashMap<[u8; 32], u64>,
) -> Result<(), PassError> {
    let pattern = format!("{}{{*}}:obj:extret", keys.prefix());
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let conn = maint_pool.next();
    let carriers = collect_keyspace_agreed(conn, &pattern)
        .await
        .map_err(|e| walk_err(client, OP_SCAN_CARRIERS, e))?;
    for carrier in &carriers {
        tally_carrier(client, conn, carrier, owed).await?;
    }
    Ok(())
}

/// Read one retirement carrier whole (`TYPE`, then `HKEYS` or `SMEMBERS` on
/// `conn`) and tally its members into `owed` ([`tally_owed_member`]).
///
/// TYPE dispatch (b.1jt): this build's carriers are HASHes (member → 1), an
/// older build's are SETs, and both are owed. A carrier deleted since the
/// SCAN saw it (cleared, cancelled or dropped) owes nothing more. Any other
/// type is not a carrier a walk can read, and an unread carrier would
/// under-count `owed`, so it fails closed, as a key that is not text does.
async fn tally_carrier(
    client: &ValkeyClient,
    conn: &fred::clients::Client,
    fred_key: &fred::types::Key,
    owed: &mut HashMap<[u8; 32], u64>,
) -> Result<(), PassError> {
    let Some(carrier_key) = fred_key.as_str() else {
        return Err(PassError::Undecodable {
            what: "an obj:extret key",
            text: format!("{fred_key:?}"),
        });
    };
    let kind: String = conn
        .r#type(carrier_key.to_string())
        .await
        .map_err(|e| valkey_err(client, "TYPE obj:extret", e))?;
    let members: Vec<Value> = match kind.as_str() {
        "hash" => conn
            .hkeys(carrier_key.to_string())
            .await
            .map_err(|e| valkey_err(client, "HKEYS obj:extret", e))?,
        "set" => conn
            .smembers(carrier_key.to_string())
            .await
            .map_err(|e| valkey_err(client, "SMEMBERS obj:extret", e))?,
        "none" => Vec::new(),
        _ => {
            return Err(PassError::Undecodable {
                what: "an obj:extret carrier of a type other than hash or set",
                text: format!("{carrier_key} is a {kind}"),
            })
        }
    };
    for value in &members {
        tally_owed_member(value, owed)?;
    }
    Ok(())
}

/// Phases 3a and 3b — how many decrements the GC already owes per hash,
/// pending first and inflight second.
///
/// `le:ext:{gc}:pending` and `le:ext:{gc}:inflight` are single keys on one slot,
/// so their cursored forms route by key and need no cluster fan-out. Members are
/// tallied through `pending_member_hash_hex`, which parses every spelling the
/// protocol has ever written — the legacy bare `{hash_hex}`, b.qpf's
/// `{hash_hex}:{object_id}:{event}`, and b.yda's own `{hash_hex}:0:rec{n}`.
///
/// **The two walks below are ordered, not merely sequenced.** A member moves
/// pending → inflight when the GC claims it, so pending must be walked first: a
/// member that moves mid-pass is then seen by both walks (counted twice, surplus
/// under-stated, leak direction), where the reverse order would have it seen by
/// neither (surplus over-stated, over-release). Same argument as phase 2b's
/// placement; see the module header.
///
/// Accumulates into the caller's `owed` rather than returning its own map, so
/// the carrier walk's tally and these two share one total. The periodic
/// pass's; the restore Job's recount reads the same two keys by its own
/// proofs ([`walk_owed_counted`]).
async fn walk_owed(
    client: &ValkeyClient,
    keys: &KeySchema,
    owed: &mut HashMap<[u8; 32], u64>,
    budget: Duration,
    started: Instant,
    cancel: &CancellationToken,
) -> Result<(), PassError> {
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let conn = maint_pool.next();

    // 3a — pending, then 3b — inflight. Each is a guarded walk through the
    // cursor pager (`storage::retirement::page_by_cursor` over a
    // `storage::scan_guard::KeyedWalk` on `conn`): explicit-cursor steps to
    // cursor `0`, so a walk ends only on its terminal step or with an error
    // (a step lost in flight fails visibly), and a step's members are tallied
    // only once its witness, sent in one pipeline with the step, matched the
    // walk's start. A keyed cursor routes by its key to one node.
    for (kind, key, command, op, phase) in [
        // FIRST, because members flow pending → inflight.
        (
            OwedKind::Pending,
            keys.ext_pending(),
            "SSCAN",
            OP_SCAN_PENDING,
            "the pending walk",
        ),
        // SECOND, and only second. The score is not read, so this build's
        // encoded claims (claim ms + `GC_CLAIM_SCORE_OFFSET_MS`) and an
        // older build's bare ones count alike (b.1jt).
        (
            OwedKind::Inflight,
            keys.ext_inflight(),
            "ZSCAN",
            OP_SCAN_INFLIGHT,
            "the inflight walk",
        ),
    ] {
        page_by_cursor(&mut OwedSteps {
            client,
            walk: KeyedWalk::new(conn, command, &key),
            kind,
            op,
            phase,
            owed: &mut *owed,
            budget,
            started,
            cancel,
        })
        .await?;
    }

    Ok(())
}

/// The restore Job's recount's owed walks (`storage::scan_guard`, *The DR
/// Jobs' walks*), pending first and inflight second as in [`walk_owed`]:
/// `pending` by a counted `SSCAN` walk, `SCARD` in each step's pipeline,
/// each distinct member tallied once (a member a step repeats is not counted
/// twice), complete only when the distinct members equal the `SCARD` read
/// with the first and the last step; then `inflight` by ascending `ZRANGE`
/// pages of [`ROW_HASH_CHUNK_ROWS`] with `ZCARD` in each page's pipeline,
/// complete only when the pages returned the first page's `ZCARD` and the
/// last page's equals it. A proof that does not hold fails the walk
/// ([`PassError::Incomplete`]) before any count is set. Holds one step or
/// page, the tally, and `pending`'s distinct members. No budget and no
/// cancellation: the Job ends the recount by dropping it.
async fn walk_owed_counted(
    client: &ValkeyClient,
    keys: &KeySchema,
    owed: &mut HashMap<[u8; 32], u64>,
) -> Result<(), PassError> {
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let conn = maint_pool.next();
    // FIRST, because members flow pending → inflight.
    walk_counted(
        &CountedScan::set(conn, &keys.ext_pending()),
        u64::from(SCAN_PAGE),
        &mut CountedPending {
            client,
            seen: HashSet::new(),
            owed: &mut *owed,
        },
    )
    .await?;
    // SECOND. The score is not read, so this build's encoded claims and an
    // older build's bare ones count alike (b.1jt).
    page_by_rank_counted(
        &RankWalk::new(conn, &keys.ext_inflight(), false),
        ROW_HASH_CHUNK_ROWS as u64,
        &mut InflightPages {
            client,
            owed: &mut *owed,
        },
    )
    .await
}

/// The recount's counted steps over `pending` ([`walk_owed_counted`]): each
/// distinct member tallied once into `owed`.
struct CountedPending<'w> {
    client: &'w ValkeyClient,
    /// The distinct members taken, by their bytes.
    seen: HashSet<String>,
    owed: &'w mut HashMap<[u8; 32], u64>,
}

impl CountedStepTaker for CountedPending<'_> {
    type Error = PassError;

    fn walk_error(&self, e: WalkError) -> PassError {
        walk_err(self.client, OP_SCAN_PENDING, e)
    }

    async fn take_step(&mut self, members: Vec<Value>) -> Result<(), PassError> {
        for value in &members {
            // A member that is not text is `tally_owed_member`'s undecodable
            // record; a text member a step repeats is tallied once.
            let repeated = value
                .as_str()
                .is_some_and(|member| !self.seen.insert(member.into_owned()));
            if !repeated {
                tally_owed_member(value, self.owed)?;
            }
        }
        Ok(())
    }

    fn distinct(&self) -> u64 {
        self.seen.len() as u64
    }
}

/// The recount's rank pages over `inflight` ([`walk_owed_counted`]): each
/// member tallied into `owed`.
struct InflightPages<'w> {
    client: &'w ValkeyClient,
    owed: &'w mut HashMap<[u8; 32], u64>,
}

impl RankPageTaker for InflightPages<'_> {
    type Error = PassError;

    fn walk_error(&self, e: WalkError) -> PassError {
        walk_err(self.client, OP_PAGE_INFLIGHT, e)
    }

    async fn take_page(&mut self, members: Vec<Value>) -> Result<(), PassError> {
        for value in &members {
            tally_owed_member(value, self.owed)?;
        }
        Ok(())
    }
}

/// Which owed walk an [`OwedSteps`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwedKind {
    /// `SSCAN le:ext:{gc}:pending`: every element is a member.
    Pending,
    /// `ZSCAN le:ext:{gc}:inflight`: members and scores interleaved.
    Inflight,
}

/// One owed walk's steps ([`walk_owed`]): each step's members tallied into
/// `owed`. Cancellation forfeits at every step, in flight or between two,
/// exactly as in the object walk, and so does the pass budget. A truncated
/// `owed` tally is worse than a truncated object walk, not better: an
/// omitted obligation OVER-states the surplus, which is the over-release
/// direction the pass exists to refuse.
struct OwedSteps<'w> {
    client: &'w ValkeyClient,
    walk: KeyedWalk<'w>,
    kind: OwedKind,
    op: &'static str,
    phase: &'static str,
    owed: &'w mut HashMap<[u8; 32], u64>,
    budget: Duration,
    started: Instant,
    cancel: &'w CancellationToken,
}

impl CursorPageReader for OwedSteps<'_> {
    type Error = PassError;
    /// The reply's own `Value`, so a member that is not text is
    /// [`tally_owed_member`]'s undecodable record, as from a carrier.
    type Member = Value;

    /// The step, its witnesses in its pipeline, raced against the pass's
    /// cancellation.
    async fn read_step(
        &mut self,
        cursor: &str,
        start: Option<&Witness>,
    ) -> Result<GuardedStep<Value>, PassError> {
        let step = tokio::select! {
            biased;
            () = self.cancel.cancelled() => {
                return Err(PassError::Cancelled { phase: self.phase })
            }
            step = self
                .walk
                .step(cursor, u64::from(SCAN_PAGE), start, WALK_STEP_ATTEMPTS) => step,
        };
        let step = step.map_err(|e| walk_err(self.client, self.op, e))?;
        match self.kind {
            OwedKind::Pending => Ok(step),
            OwedKind::Inflight => step
                .try_map_members(|elements| {
                    scan_step_firsts(self.walk.command(), self.walk.key(), elements)
                })
                .map_err(|e| valkey_err(self.client, self.op, e)),
        }
    }

    fn changed(&self, start: &Witness, later: &Witness) -> PassError {
        changed_error(&self.walk, start, later)
    }

    async fn take_page(&mut self, members: Vec<Value>) -> Result<(), PassError> {
        if self.cancel.is_cancelled() {
            return Err(PassError::Cancelled { phase: self.phase });
        }
        if self.started.elapsed() > self.budget {
            return Err(PassError::BudgetElapsed { phase: self.phase });
        }
        for value in &members {
            tally_owed_member(value, self.owed)?;
        }
        Ok(())
    }
}

/// Add one owed-side member to the tally — from a retry carrier (phase 2b),
/// from `pending` (3a) or from `inflight` (3b). All three spellings are the same
/// text, which is why one helper serves them.
///
/// Fails closed on an undecodable member for the OPPOSITE reason the object walk
/// does: an omitted `owed` entry over-states the surplus, which is the
/// over-release direction. Every walk therefore aborts rather than skips, and
/// the pass has no "mostly enumerated" state at all.
fn tally_owed_member(value: &Value, owed: &mut HashMap<[u8; 32], u64>) -> Result<(), PassError> {
    let Some(member) = value.as_str() else {
        return Err(PassError::Undecodable {
            what: "a release-channel member",
            text: format!("{value:?}"),
        });
    };
    let hash_hex = pending_member_hash_hex(member.as_ref());
    match hash_from_hex(hash_hex) {
        Some(hash) => {
            *owed.entry(hash).or_insert(0) += 1;
            Ok(())
        }
        None => Err(PassError::Undecodable {
            what: "the hash of a release-channel member",
            text: member.to_string(),
        }),
    }
}

/// The two ineligibility tallies phase 4 accumulates, both of them
/// `ext_reconcile_total{outcome}` cells.
///
/// There is deliberately no tally for an ABSENT `last_store_ms`: post-b.rej that
/// case is ELIGIBLE and becomes a candidate rather than a skip (module header,
/// guard 3).
#[derive(Debug, Default)]
struct SkipTally {
    /// A positive surplus whose `last_store_ms` is newer than the cutoff — an
    /// increment may still be awaiting its commit. The healthy, busy-store arm.
    recent_store: u64,
    /// A positive surplus on a meta whose `last_store_ms` is PRESENT but cannot
    /// be parsed. Nothing in the service writes such a value, so this is a
    /// keyspace-integrity signal rather than a load one — and it is kept
    /// strictly apart from the absent case, which the pass now treats as
    /// eligible: reading corruption as "nothing ever stored this" would license
    /// a repair on a value the pass failed to understand.
    unreadable_stamp: u64,
    /// A positive surplus on a hash in this pass's `unrepaired_drift` set —
    /// some live object's index disagreed with its rows on the hash and the
    /// recount could not repair it. Held, not registered: draining it could
    /// complete an over-release (see `walk_objects`).
    index_drift: u64,
}

/// Phase 4 — walk `le:ext:meta:*`, compute each hash's surplus, and build the
/// candidate list.
///
/// Returns `(summary-with-gauge-readings, candidates, skips)`. `candidates` are
/// `{hash_hex}:{amount}` members, in both directions:
///
/// * **positive** `{surplus}` for every meta with a positive surplus **whose
///   `last_store_ms` is absent, or present and older than `stamp_cutoff`**. A
///   meta stamped inside the grace is not a candidate, and neither is one whose
///   stamp is present but unreadable — see the module header's guard 3 for why
///   those two fail closed while the absent case does not.
/// * **negative** `-{deficit}` for every meta whose `ref_count` reads below
///   `referenced + owed`, with no stamp gate at all — a deficit is registered
///   for CONFIRMATION, never for repair, and no stamp state can make one
///   spurious.
///
/// Both share one ZSET, one grace and one claim; the direction is carried in the
/// sign of the amount and dispatched on in [`act_on_candidate`].
///
/// A positive surplus on a hash in `unrepaired_drift` (phase 2's index audit)
/// is not registered at all, whatever its stamp — `skipped_index_drift`. A
/// deficit on such a hash registers as usual.
#[allow(clippy::too_many_arguments)]
async fn walk_metas(
    client: &ValkeyClient,
    keys: &KeySchema,
    referenced: &HashMap<[u8; 32], u64>,
    owed: &HashMap<[u8; 32], u64>,
    unrepaired_drift: &HashSet<[u8; 32]>,
    stamp_cutoff: u64,
    budget: Duration,
    started: Instant,
    cancel: &CancellationToken,
) -> Result<(PassSummary, Vec<String>, SkipTally), PassError> {
    let pattern = keys.ext_meta_scan_pattern();
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let scan_client = maint_pool.next();
    let mut stream = scan_keyspace(scan_client, &pattern);

    let mut summary = PassSummary::default();
    let mut candidates: Vec<String> = Vec::new();
    let mut skips = SkipTally::default();
    let mut batch: Vec<String> = Vec::with_capacity(META_HMGET_CHUNK);

    loop {
        // Cancellation forfeits, as in every other walk. A truncated meta walk
        // is the one shape whose partial answer looks most usable — the metas it
        // did read were read correctly — and it is still not usable: this pass
        // pins first observations from it, and a candidate pinned from a
        // truncated walk waits on the ZSET at an amount that was measured
        // against a complete `referenced` map but only a partial set of metas,
        // for a future pass to confirm.
        let next = tokio::select! {
            biased;
            () = cancel.cancelled() => return Err(PassError::Cancelled {
                phase: "the meta walk",
            }),
            item = stream.next() => item,
        };
        let finished = next.is_none();
        if let Some(scan_result) = next {
            if started.elapsed() > budget {
                return Err(PassError::BudgetElapsed {
                    phase: "the meta walk",
                });
            }
            let fred_key =
                scan_result.map_err(|e| valkey_err(client, "SCAN ext:meta", e.into_fred()))?;
            let Some(key) = fred_key.as_str() else {
                return Err(PassError::Undecodable {
                    what: "an ext:meta key",
                    text: format!("{fred_key:?}"),
                });
            };
            batch.push(key.to_string());
        }

        if batch.len() >= META_HMGET_CHUNK || (finished && !batch.is_empty()) {
            read_meta_batch(
                client,
                keys,
                &batch,
                referenced,
                owed,
                unrepaired_drift,
                stamp_cutoff,
                &mut summary,
                &mut candidates,
                &mut skips,
            )
            .await?;
            batch.clear();
        }

        if finished {
            break;
        }
    }

    Ok((summary, candidates, skips))
}

/// `HMGET ref_count last_store_ms` over one chunk of meta keys, folded into the
/// running summary and candidate list.
#[allow(clippy::too_many_arguments)]
async fn read_meta_batch(
    client: &ValkeyClient,
    keys: &KeySchema,
    batch: &[String],
    referenced: &HashMap<[u8; 32], u64>,
    owed: &HashMap<[u8; 32], u64>,
    unrepaired_drift: &HashSet<[u8; 32]>,
    stamp_cutoff: u64,
    summary: &mut PassSummary,
    candidates: &mut Vec<String>,
    skips: &mut SkipTally,
) -> Result<(), PassError> {
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let conn = maint_pool.next();
    let pipeline = conn.pipeline();
    for key in batch {
        let _: () = pipeline
            .hmget(key.as_str(), vec![REF_COUNT_FIELD, LAST_STORE_MS_FIELD])
            .await
            .map_err(|e| valkey_err(client, "queue HMGET ext:meta", e))?;
    }
    let replies: Vec<Value> = pipeline
        .all()
        .await
        .map_err(|e| valkey_err(client, "HMGET ext:meta", e))?;
    if replies.len() != batch.len() {
        return Err(PassError::Undecodable {
            what: "an ext:meta HMGET pipeline reply of the expected arity",
            text: format!("{} frames for {} keys", replies.len(), batch.len()),
        });
    }

    for (key, reply) in batch.iter().zip(replies) {
        summary.meta_total += 1;

        let Some(hash_hex) = keys.ext_meta_hash_hex(key) else {
            return Err(PassError::Undecodable {
                what: "the hash_hex of an ext:meta key",
                text: key.clone(),
            });
        };
        let Some(hash) = hash_from_hex(hash_hex) else {
            return Err(PassError::Undecodable {
                what: "the hash of an ext:meta key",
                text: key.clone(),
            });
        };

        let fields: Vec<Option<String>> = reply
            .convert()
            .map_err(|e| valkey_err(client, "decode HMGET ext:meta", e))?;
        // An absent or unparseable `ref_count` is a meta this pass cannot judge.
        // Skipping it is safe in the only direction that matters — it can only
        // leave a surplus standing — so unlike the walks above this does not
        // forfeit the pass.
        let Some(ref_count) = fields
            .first()
            .and_then(|v| v.as_ref())
            .and_then(|s| s.parse::<i64>().ok())
        else {
            continue;
        };
        let last_store_ms = classify_stamp(fields.get(1));

        let live = referenced.get(&hash).copied().unwrap_or(0);
        let queued = owed.get(&hash).copied().unwrap_or(0);
        let surplus = ref_count
            - i64::try_from(live).unwrap_or(i64::MAX)
            - i64::try_from(queued).unwrap_or(i64::MAX);

        if surplus < 0 {
            // A FIRST observation, and nothing more. Registered on the same
            // `{recon}` ZSET the surplus arm uses, as `{hash_hex}:-{deficit}` —
            // negative amounts are unambiguous in the member string, and the
            // claim script is member-opaque, so the two directions share one
            // protocol and one grace. Phase 5's confirm arm decides whether this
            // is an over-release or one of the transients the pass's own phase
            // order manufactures; only that arm touches the alertable gauge.
            //
            // The `last_store_ms` eligibility rule that gates the surplus arm
            // below is deliberately NOT applied here. It exists because a
            // pre-flight increment awaiting its commit reads as surplus; nothing
            // about an absent, fresh or unreadable stamp can make a DEFICIT
            // spurious, so a meta in any of those states registers exactly like
            // a comfortably-stamped one.
            //
            // DEBUG, not WARN: on a busy store this fires routinely (see the
            // module header). The WARN that an operator should act on is emitted
            // by the confirm arm, on the second observation.
            summary.deficits_observed_unconfirmed += 1;
            tracing::debug!(
                hash = %hash_hex,
                ref_count,
                referenced = live,
                owed = queued,
                deficit = surplus.unsigned_abs(),
                "ext:meta ref_count reads BELOW the live-object reference count; registered for \
                 confirmation a grace from now, not reported — the pass's own phase order \
                 double-counts a mid-pass transition and produces exactly this reading"
            );
            candidates.push(format!("{hash_hex}:{surplus}"));
            continue;
        }
        if surplus == 0 {
            continue;
        }

        summary.surplus_total += surplus.unsigned_abs();

        // The b.rej pre-flight stamp — the ONLY guard that can tell a standing
        // surplus from an upload whose `ref_count` increment has not reached
        // its commit yet, because the pre-store window is invisible to both of
        // the others (no row for the object walk to see, no release for the
        // owed subtraction to find). Three states, and the two that are NOT a
        // readable timestamp get opposite verdicts:
        //
        //   * absent      → eligible. Every pre-flight store stamps, so an
        //                   absent field is positive evidence that none has
        //                   touched this hash since the writer deployed — the
        //                   standing leak this pass exists to drain.
        //   * unreadable  → INELIGIBLE. Nothing here writes an unparseable
        //                   stamp, so this is corrupt or foreign keyspace and
        //                   the pass fails closed rather than reading it as
        //                   absence (guard 3).
        //   * >= cutoff   → INELIGIBLE. An increment may still be in flight.
        //   * <  cutoff   → eligible. Nothing has pre-stored this hash for a
        //                   full grace.
        //
        // Ineligible metas are not registered at all — the surplus is still
        // summed into the gauge above, so the pass reports what it sees, it just
        // does not pin a first observation it may not act on.
        //
        // The index-drift gate sits beside the stamp gate and refuses for a
        // different reason: some live object's `obj:ext` disagreed with its
        // rows on this hash and the phase-2 recount could not repair it, so
        // the index may under-count a live row's reference — and when a
        // pre-index seeding commit produced that under-count, this surplus is
        // the increment that keeps the blob alive until it is repaired.
        if unrepaired_drift.contains(&hash) {
            skips.index_drift += 1;
            continue;
        }
        match last_store_ms {
            StampState::Unreadable => {
                skips.unreadable_stamp += 1;
                continue;
            }
            StampState::At(ms) if ms >= stamp_cutoff => {
                skips.recent_store += 1;
                continue;
            }
            StampState::Absent | StampState::At(_) => {}
        }

        candidates.push(format!("{hash_hex}:{surplus}"));
    }

    Ok(())
}

/// What acting on one claimed candidate did.
enum CandidateOutcome {
    /// The surplus still stood at the claimed amount, and this many repair
    /// members were emitted onto `le:ext:{gc}:pending`.
    Repaired { references: u64 },
    /// The surplus no longer matches the amount the member encodes, so the
    /// candidate healed on its own and nothing was emitted.
    Healed,
    /// The FRESH `last_store_ms` re-read says this meta may not be judged right
    /// now — it carries a stamp newer than the cutoff, or one that cannot be
    /// parsed — or the hash is in this pass's `unrepaired_drift` set (phase 2's
    /// index audit could not repair some object's index on it). Nothing was
    /// emitted and the member was NOT retired: it keeps its first-observation
    /// score and is re-checked next pass, against a fresh count and a fresh
    /// stamp. `outcome` is the `ext_reconcile_total` cell to tick.
    ///
    /// An ABSENT fresh stamp does not reach here — post-b.rej that is the
    /// eligible case.
    ///
    /// Reachable only from the surplus arm. The deficit arm has no eligibility
    /// gate to fail — see the module header.
    Skipped { outcome: &'static str },
    /// A deficit member whose deficit still stood, at the same amount, a full
    /// grace after it was first observed. This is the one reading the pass
    /// treats as evidence of an over-release: it ticks `deficit_observed` and
    /// publishes `deficit` on `live_edit_ext_ref_deficit_total`. Nothing is
    /// emitted — the deficit direction is never repaired.
    DeficitConfirmed { deficit: u64 },
    /// A deficit member whose deficit did not reproduce on the fresh re-read —
    /// a mid-pass double-count, a stale carrier over a live object, or a real
    /// deficit that changed amount. Retired without alerting; if a deficit is
    /// still there at a different amount, phase 4 registers that amount as its
    /// own first observation.
    DeficitCleared,
}

/// Re-check one claimed candidate against a FRESH read and act.
///
/// The re-read is what makes the two-observation protocol a check rather than a
/// ritual: the claim proves the amount was stable a grace ago, and this proves
/// it is still stable now, at the instant of the repair.
///
/// # Two arms, dispatched on the sign of the claimed amount
///
/// A member's amount carries its direction. **Negative** ⇒ the deficit arm,
/// which is confirm-only: it decides whether a negative surplus this pass's own
/// phase order may have manufactured is a real over-release, and reports or
/// clears it. It never emits anything. **Positive** ⇒ the surplus arm below,
/// which may emit repair members. The eligibility gate and everything documented
/// under it belongs to the surplus arm alone.
///
/// # Both fields are re-read, and both are load-bearing
///
/// `ref_count` AND `last_store_ms`, in one `HMGET`, and the eligibility test
/// runs on the FRESH stamp rather than on the one phase 4 read. Phase 4's
/// answer is up to `budget` (half the tick interval) old by the time this runs,
/// and the whole point of the stamp is to catch an upload that started recently
/// — an upload that began after phase 4 read the meta is precisely the case a
/// stale answer would miss. The pre-store increments `ref_count` BEFORE the
/// commit that materialises the row, so such an upload presents as a stable
/// surplus to every other check this pass makes.
///
/// `referenced` and `owed` do still come from this pass's own walks: re-walking
/// the keyspace per candidate would cost a full pass per hash, and both of those
/// err toward under-stating the surplus when they are stale (a newly-created
/// object or a newly-queued release only ever ADDS to them), which suppresses a
/// repair rather than inventing one. The stamp is the only input whose staleness
/// errs the other way, which is why it alone is re-read.
///
/// The fresh gate below is the same three-way test phase 4 applies, for the same
/// reasons: an absent stamp is eligible (every store stamps, so absence proves
/// none has), a stamp inside the grace is `skipped_recent_store`, and a stamp
/// that is present but unparseable is `skipped_unreadable_stamp` — corrupt
/// keyspace the pass fails closed on rather than mistaking for absence (module
/// header, guard 3).
///
/// Beside it sits the index-drift gate: a surplus on a hash in
/// `unrepaired_drift` is never repaired this pass (`skipped_index_drift`) — the
/// claim cannot be refused, since the member was registered on an earlier pass,
/// so the gate is applied here. See `walk_objects`.
#[allow(clippy::too_many_arguments)]
async fn act_on_candidate(
    client: &ValkeyClient,
    keys: &KeySchema,
    member: &str,
    referenced: &HashMap<[u8; 32], u64>,
    owed: &HashMap<[u8; 32], u64>,
    unrepaired_drift: &HashSet<[u8; 32]>,
    stamp_cutoff: u64,
) -> Result<CandidateOutcome, PassError> {
    let recon_key = keys.ext_recon_candidates();

    let Some((hash_hex, claimed_amount)) = member.rsplit_once(':') else {
        // Not a member this pass wrote. Retire it rather than re-claiming it
        // every pass forever.
        retire_member(client, &recon_key, member).await;
        return Ok(CandidateOutcome::Healed);
    };
    // `i64`, not `u64`: the amount carries the DIRECTION in its sign. A leading
    // `-` is a deficit member registered for confirmation (phase 4), which the
    // confirm-only arm below handles; anything positive is a surplus candidate.
    // `0` is written by neither arm and is retired by the zero guard below.
    let (Some(hash), Ok(claimed_amount)) = (hash_from_hex(hash_hex), claimed_amount.parse::<i64>())
    else {
        retire_member(client, &recon_key, member).await;
        return Ok(CandidateOutcome::Healed);
    };

    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let fresh_fields: Vec<Option<String>> = maint_pool
        .next()
        .hmget(
            keys.ext_meta(hash_hex),
            vec![REF_COUNT_FIELD, LAST_STORE_MS_FIELD],
        )
        .await
        .map_err(|e| {
            valkey_err(
                client,
                "HMGET ext:meta ref_count last_store_ms (recheck)",
                e,
            )
        })?;
    let fresh = fresh_fields.first().and_then(|v| v.clone());
    let fresh_stamp = classify_stamp(fresh_fields.get(1));

    let live = referenced.get(&hash).copied().unwrap_or(0);
    let queued = owed.get(&hash).copied().unwrap_or(0);
    let fresh_surplus = fresh.and_then(|s| s.parse::<i64>().ok()).map(|rc| {
        rc - i64::try_from(live).unwrap_or(i64::MAX) - i64::try_from(queued).unwrap_or(i64::MAX)
    });

    // ── The deficit arm: CONFIRM-ONLY, and it returns before anything below ──
    //
    // The second of the two observations a deficit needs. The first was phase
    // 4's, a grace ago; this one recomputes against a fresh `ref_count` and the
    // same pass-level `live` / `queued`. Same amount ⇒ it outlived a full grace
    // unchanged, which neither of the benign sources (the pass's own mid-pass
    // double-count, a retire carrier over a live object) can do, so it is
    // reported. Anything else ⇒ it was one of them.
    //
    // No eligibility gate: `last_store_ms` is read above but deliberately not
    // consulted here. It distinguishes a standing surplus from an in-flight
    // pre-store, and a pre-store cannot manufacture a deficit — it moves
    // `ref_count` UP.
    //
    // Nothing is ever emitted from this arm. It cannot repair in either
    // direction, by construction: there is no code path from here to
    // `enqueue_ext_releases` and none to an `HINCRBY`.
    //
    // Only a live meta confirms. A meta the GC has deleted reads `None`
    // above, so `fresh_surplus` is `None`, never equal to the claimed amount,
    // and the candidate is cleared. Every store writes meta and data together
    // and the GC deletes them together, so a confirmed deficit's value is
    // still stored when it is reported (SDD §11.3.2): in Valkey, or, for a
    // meta a DR restore's recount created (module header, "The
    // meta-without-data shape"), in the backup store alone until a read
    // fetches it back.
    if claimed_amount < 0 {
        if fresh_surplus != Some(claimed_amount) {
            retire_member(client, &recon_key, member).await;
            return Ok(CandidateOutcome::DeficitCleared);
        }
        let deficit = claimed_amount.unsigned_abs();
        tracing::warn!(
            hash = %hash_hex,
            deficit,
            referenced = live,
            owed = queued,
            "ext:meta ref_count stands BELOW the live-object reference count and has done for a \
             full grace at the same amount; an over-release already happened. The value is still \
             stored, in Valkey or the backup store, but it will be collected while live objects \
             reference it once the count reaches 0, and it may already have been collected once \
             and stored again. Reported only — never repaired, because incrementing could \
             resurrect a meta whose blob the GC deletes in between"
        );
        // Retired once counted, so the alertable gauge reads what THIS pass
        // confirmed. A deficit that is still there re-registers on the next
        // pass's phase 4 and confirms again a grace later.
        retire_member(client, &recon_key, member).await;
        return Ok(CandidateOutcome::DeficitConfirmed { deficit });
    }

    // ── The surplus arm ─────────────────────────────────────────────────────
    //
    // A zero amount is a member neither arm writes — a hand-edited key, or a
    // build that wrote a different spelling. Retire it rather than carrying it
    // into the repair path, where it would emit no members and still report a
    // repaired hash.
    if claimed_amount == 0 {
        retire_member(client, &recon_key, member).await;
        return Ok(CandidateOutcome::Healed);
    }
    let claimed_surplus = claimed_amount.unsigned_abs();

    // The amount check FIRST, and deliberately ahead of the eligibility gate
    // below. It is the arm that RETIRES a member, and a member whose meta is
    // gone entirely reaches it with `fresh_surplus == None`: gating on the stamp
    // first would send that member down the "no stamp" arm, which leaves it
    // registered — forever, since the meta it names will never come back to
    // change the answer. Retiring is not a repair, so nothing about the stamp
    // guard is weakened by letting the void candidates go first.
    if fresh_surplus != Some(claimed_amount) {
        // Healed: a commit landed, a release drained, or the meta is gone. The
        // amount is the whole claim, so a different amount is a different
        // candidate and this one is void.
        retire_member(client, &recon_key, member).await;
        return Ok(CandidateOutcome::Healed);
    }

    // The eligibility gate, re-run on the FRESH stamp. A meta that is freshly
    // stamped is one this pass may not repair, however stable its surplus looked
    // a grace ago — an in-flight pre-store looks exactly that stable, and its
    // increment lands on `ref_count` before the row that would have told the
    // object walk about it. A stamp that is present but unreadable is refused
    // for the other reason: the pass will not act on a value it could not
    // understand. Leave the member registered in both cases rather than retiring
    // it: the surplus may well be real, and a later pass gets to decide with a
    // stamp it can date.
    //
    // An ABSENT stamp passes. Every pre-flight store writes one, so its absence
    // is positive evidence that none has touched this hash — see guard 3.
    //
    // The index-drift gate is checked first and leaves the member registered
    // too: some live object's index may under-count this hash, and (for a
    // pre-index seeding under-count) this surplus keeps the blob alive until a
    // later pass repairs that index.
    if unrepaired_drift.contains(&hash) {
        return Ok(CandidateOutcome::Skipped {
            outcome: EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT,
        });
    }
    match fresh_stamp {
        StampState::Unreadable => {
            return Ok(CandidateOutcome::Skipped {
                outcome: EXT_RECONCILE_OUTCOME_SKIPPED_UNREADABLE_STAMP,
            })
        }
        StampState::At(ms) if ms >= stamp_cutoff => {
            return Ok(CandidateOutcome::Skipped {
                outcome: EXT_RECONCILE_OUTCOME_SKIPPED_RECENT_STORE,
            })
        }
        StampState::Absent | StampState::At(_) => {}
    }

    let to_repair = claimed_surplus.min(EXT_RECON_MAX_REPAIR_PER_HASH);
    // Repair by RELEASE EVENT, never by decrement-in-place. Riding the existing
    // pending → claim → `gc_decrement_and_collect` protocol means the repair
    // inherits its crash-safety, its underflow accounting and its
    // collect-at-zero behaviour for free; a bare `HINCRBY -n` here would have
    // none of them and would race the GC's own decrements.
    //
    // `surplus` rather than an index is encoded in each member so a replay of
    // THIS repair collapses in the Set (idempotent within the event) while a
    // genuinely different repair of the same hash, at a different amount, stays
    // distinct. Emitting `to_repair` members of one amount would collapse to
    // one, so the amount rides down: `{n}` counts down from the claimed surplus.
    let releases: Vec<ExtRelease> = (0..to_repair)
        .map(|i| {
            let amount = u32::try_from(claimed_surplus - i).unwrap_or(u32::MAX);
            ExtRelease::reconcile(hash, amount)
        })
        .collect();

    enqueue_ext_releases(client, keys, &releases)
        .await
        .map_err(|e| PassError::Repair(e.to_string()))?;

    // Retired only AFTER the repair landed. The reverse order would lose the
    // candidate on a failed enqueue; this way a failure leaves the member
    // claimable next pass, which re-checks it against a fresh count first.
    retire_member(client, &recon_key, member).await;
    Ok(CandidateOutcome::Repaired {
        references: to_repair,
    })
}

/// `ZREM` one member off the candidate ZSET, WARNing on failure.
///
/// Every retire in [`act_on_candidate`] is best-effort by design — the member is
/// an observation, not an obligation, so a failed `ZREM` costs one redundant
/// re-claim next pass and nothing else, which is why none of them propagate.
///
/// It is still not nothing, and must not be silent. A `ZREM` that keeps failing
/// leaves members accumulating on `le:ext:{gc}:recon` against a `noeviction`
/// store, and — on the repair arm — leaves a member whose repair ALREADY landed
/// claimable again, where the next pass's fresh re-read is the only thing
/// standing between it and a second repair of the same surplus. Same WARN
/// posture as `orphan_sweep::drain_retire_carrier`'s retire failures.
async fn retire_member(client: &ValkeyClient, recon_key: &str, member: &str) {
    let members = [member.to_owned()];
    if let Err(e) = recon_retire(client, recon_key, &members).await {
        tracing::warn!(
            member = %member,
            error = %e,
            "could not retire a ref_count reconcile candidate; it stays claimable and the next \
             pass re-checks it against a fresh count"
        );
    }
}

// ── The restore Job's authoritative recount ─────────────────────────────────

/// How many per-hash count sets the recount runs at once; `before_write` is
/// awaited before each group of this many.
pub(crate) const RECOUNT_WRITE_CHUNK: usize = 64;

/// What the restore Job's recount did ([`recount_authoritative`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecountReport {
    /// The distinct hashes whose count it set: every hash referenced or owed.
    pub values: u64,
    /// The object headers it walked.
    pub objects: u64,
    /// Metas it created, with no data (`created`).
    pub created: u64,
    /// Counts it raised.
    pub raised: u64,
    /// Counts it lowered.
    pub lowered: u64,
    /// Counts that already held the target.
    pub unchanged: u64,
    /// Counts that were absent or unreadable, rewritten.
    pub repaired: u64,
    /// Of `values`, the hashes only an owed release names (no object
    /// references them): metas the GC drains to 0.
    pub owed_only: u64,
    /// The largest decimal object id among the headers walked: the counter
    /// floor's surviving term.
    pub max_object_id: Option<u64>,
}

/// Proof that an authoritative recount completed without error: the only
/// way to obtain one is [`recount_authoritative`]'s `Ok`. The un-hold takes
/// it by reference (`storage::dr_hold::unhold`), so no un-hold that has not
/// recounted compiles. Not `Clone`, with no public constructor; lives for the
/// process that made it.
#[derive(Debug)]
pub(crate) struct RecountReceipt {
    report: RecountReport,
}

impl RecountReceipt {
    /// What the recount did.
    pub(crate) fn report(&self) -> &RecountReport {
        &self.report
    }
}

/// Why an authoritative recount did not complete. Any of them leaves the
/// counts partly set and the fleet held: no receipt was made.
#[derive(Debug)]
pub(crate) enum RecountError<E> {
    /// A store or transport fault, or a keyspace walk whose slot map changed
    /// or whose node pin missed its primary's slots (`storage::scan_guard`):
    /// worth a retry of the whole Job, which walks again from the start with
    /// a fresh routing table.
    Store(String),
    /// A walk whose completeness proof did not hold (`storage::scan_guard`,
    /// *The DR Jobs' walks*): the store changed while the Job read it. The
    /// text names the walk (its command and its key or pattern) and the
    /// counts it compared. Worth a retry of the whole Job, which walks the
    /// store as it then is.
    Incomplete(String),
    /// A keyspace walk's `CLUSTER NODES` gave no usable myself line
    /// (`storage::scan_guard::WitnessMissing`), so the walk cannot be shown
    /// to reach the node that serves its primary: the text names the
    /// command, the walk, the node and what was seen. A retry would read it
    /// again.
    WitnessMissing(String),
    /// A record the recount cannot read: a key, field or member that does
    /// not decode, or a key of the wrong type. A retry reads it again.
    Corrupt {
        /// What the record is.
        what: String,
        /// The record: its key, or the text that did not decode, led by the
        /// object and its key when the object walk located the failure.
        key: String,
    },
    /// `before_write` refused.
    Stopped(E),
}

impl<E> RecountError<E> {
    /// The recount's reading of a walk's failure (module header, "No silent
    /// omission"): an undecodable record or a `WRONGTYPE` reply is corrupt,
    /// a missing myself line and an incomplete walk are their own cases, and
    /// anything else — a changed slot map among them — a store fault. A
    /// failure the object walk located ([`PassError::InObject`]) leads `key`
    /// with the object, and its key when known, so the Job's line names what
    /// support must look at.
    fn from_walk(e: PassError) -> Self {
        match e {
            PassError::InObject {
                object_id,
                key,
                source,
            } => {
                let located = object_label(&object_id, key.as_deref());
                match Self::from_walk(*source) {
                    Self::Corrupt { what, key: detail } => Self::Corrupt {
                        what,
                        key: format!("{located}: {detail}"),
                    },
                    Self::Store(why) => Self::Store(format!("{why} ({located})")),
                    Self::Incomplete(why) => Self::Incomplete(format!("{why} ({located})")),
                    Self::WitnessMissing(why) => Self::WitnessMissing(format!("{why} ({located})")),
                    stopped @ Self::Stopped(_) => stopped,
                }
            }
            PassError::Incomplete(incomplete) => Self::Incomplete(incomplete.to_string()),
            PassError::WitnessMissing(missing) => Self::WitnessMissing(missing.to_string()),
            PassError::Undecodable { what, text } => Self::Corrupt {
                what: what.to_owned(),
                key: text,
            },
            PassError::Valkey { op, source } if is_wrongtype_reply(&source) => Self::Corrupt {
                what: format!("a key of the wrong type ({op})"),
                key: source.details().to_string(),
            },
            other => Self::Store(other.to_string()),
        }
    }
}

/// How one hash's count set ended.
enum CountSet {
    Created,
    Raised,
    Lowered,
    Unchanged,
    Repaired,
}

/// The restore Job's authoritative recount (SDD SR-9.8; module header, "The
/// restore Job's authoritative recount"): set every referenced or owed
/// hash's `ref_count` to exactly its object references (rows ∪ `obj:ext`,
/// over every object header) plus its owed releases (retirement carriers,
/// `{gc}:pending`, `{gc}:inflight`), creating a meta with no data where none
/// exists. Deletes no meta, and leaves one nothing names alone.
///
/// Call it only while the fleet is held and after the caller's last other
/// write: it reads no guard. The walks write nothing; the sets run
/// [`RECOUNT_WRITE_CHUNK`] at a time, each group after `before_write`
/// resolved `Ok` (the restore Job passes its write permit's `begin_unit`, so
/// storage takes no dependency on the Job).
///
/// `Ok` carries the [`RecountReceipt`] the un-hold needs. An error leaves the
/// counts it already set (each exact, so a re-run sets the same values) and
/// makes no receipt.
pub(crate) async fn recount_authoritative<E, F, Fut>(
    client: &ValkeyClient,
    keys: &KeySchema,
    mut before_write: F,
) -> Result<RecountReceipt, RecountError<E>>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    // No budget and no cancellation: the Job ends the recount by dropping
    // it. A walk forfeits only on an error, which fails the recount.
    let never = CancellationToken::new();
    let started = Instant::now();
    let budget = Duration::MAX;

    let walk = walk_objects(
        client,
        keys,
        budget,
        started,
        &never,
        None,
        WalkMode::Authoritative,
    )
    .await
    .map_err(RecountError::from_walk)?;
    let mut owed: HashMap<[u8; 32], u64> = HashMap::new();
    walk_retire_carriers_agreed(client, keys, &mut owed)
        .await
        .map_err(RecountError::from_walk)?;
    walk_owed_counted(client, keys, &mut owed)
        .await
        .map_err(RecountError::from_walk)?;

    let mut report = RecountReport {
        objects: walk.objects,
        max_object_id: walk.max_object_id,
        ..RecountReport::default()
    };
    let mut targets: Vec<([u8; 32], u64)> = walk
        .referenced
        .iter()
        .map(|(hash, refs)| {
            (
                *hash,
                refs.saturating_add(owed.get(hash).copied().unwrap_or(0)),
            )
        })
        .collect();
    for (hash, owed_count) in &owed {
        if !walk.referenced.contains_key(hash) {
            targets.push((*hash, *owed_count));
            report.owed_only += 1;
        }
    }
    targets.sort_unstable();
    report.values = targets.len() as u64;

    for chunk in targets.chunks(RECOUNT_WRITE_CHUNK) {
        before_write().await.map_err(RecountError::Stopped)?;
        let sets = chunk
            .iter()
            .map(|(hash, target)| set_ref_count(client, keys, hash, *target));
        for outcome in futures::future::join_all(sets).await {
            match outcome? {
                CountSet::Created => report.created += 1,
                CountSet::Raised => report.raised += 1,
                CountSet::Lowered => report.lowered += 1,
                CountSet::Unchanged => report.unchanged += 1,
                CountSet::Repaired => report.repaired += 1,
            }
        }
    }

    Ok(RecountReceipt { report })
}

/// One `EVALSHA` of `scripts/ext_ref_count_set.lua`: `hash`'s count set to
/// `target`, on its meta key alone, on the data pool.
async fn set_ref_count<E>(
    client: &ValkeyClient,
    keys: &KeySchema,
    hash: &[u8; 32],
    target: u64,
) -> Result<CountSet, RecountError<E>> {
    if target == 0 {
        // Unreachable: a hash is a target only when something names it.
        return Err(RecountError::Store(format!(
            "the recount computed a count of 0 for {}, which it never sets",
            sha256_to_hex(hash)
        )));
    }
    let meta_key = keys.ext_meta(&sha256_to_hex(hash));
    let reply: (String, String) = EXT_REF_COUNT_SET_SCRIPT
        .evalsha_with_reload::<(String, String), _, _>(
            client.pool.next(),
            vec![meta_key.clone()],
            vec![target.to_string()],
        )
        .await
        .map_err(|e| {
            FredErrOrigin::client(client, crate::storage::client::POOL_LABEL_DATA)
                .observe_error(&e);
            if is_wrongtype_reply(&e) {
                RecountError::Corrupt {
                    what: "an external-value meta of the wrong type".to_owned(),
                    key: meta_key.clone(),
                }
            } else {
                RecountError::Store(format!("EVALSHA ext_ref_count_set on {meta_key}: {e}"))
            }
        })?;
    match reply.0.as_str() {
        "created" => Ok(CountSet::Created),
        "raised" => Ok(CountSet::Raised),
        "lowered" => Ok(CountSet::Lowered),
        "unchanged" => Ok(CountSet::Unchanged),
        "repaired" => Ok(CountSet::Repaired),
        "wrong_type" => Err(RecountError::Corrupt {
            what: format!("an external-value meta of type {}, not a hash", reply.1),
            key: meta_key,
        }),
        other => Err(RecountError::Store(format!(
            "ext_ref_count_set answered {other:?} for {meta_key}, outside its contract"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use fred::interfaces::KeysInterface;

    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::{metrics_scrape_only, Metrics};
    use crate::storage::external_values::{
        acquire_gc_leader_lock, process_pending_gc, ExternalValueStore, ValkeyExternalValueStore,
        GC_CLAIM_SCORE_OFFSET_MS, GC_LEADER_TAKEOVER_BELOW_MS, LEADER_LOCK_TTL_MS,
    };
    use crate::storage::keys::sha256_to_hex;
    use crate::storage::test_support::TestValkey;

    // ══ b.yda W3 — the external-value `ref_count` reconciler ═════════════════
    //
    // Valkey-backed and IN-CRATE, for two reasons that both matter.
    //
    // `run_one_pass`, `run_pass` and `PassError` are private, and the pass is
    // the unit: every guard it implements — the two-observation protocol, the
    // `owed` subtraction, the `last_store_ms` eligibility gate, fail-closed
    // enumeration — is a property of one whole pass over a whole keyspace, not
    // of any function an integration test could reach. `spawn_ext_ref_reconcile`
    // is the only public door and it takes a leader lease and a timer, which
    // would make every arm below a race.
    //
    // And the keyspace IS the fixture. The pass reads five families no API
    // constructs together (each object's `obj:hdr`, value rows and `obj:ext`,
    // the retire carriers, pending, inflight, `ext:meta`), and half the arms
    // need a state no production path
    // can produce — a `ref_count` that disagrees with reality is, by definition,
    // the bug this pass exists to repair. So the metas are staged through the
    // real `store_values` where that produces the right count and by hand where
    // it cannot.
    //
    // ── The two-observation protocol, driven without waiting ────────────────
    //
    // A candidate is claimable only once its FIRST-observation score has aged
    // past the grace, and `recon_claim_expired`'s script guarantees a member
    // registered by a call cannot be returned by that same call (its score is
    // `now_ms`, strictly above any positive-grace cutoff). So a repair takes two
    // passes by construction, and these tests advance the clock by rewriting the
    // ZSET score to the epoch between them — [`age_candidates`] — rather than
    // sleeping. `ZADD NX` is what makes that sound: the second pass re-registers
    // the same member and the script declines to move the score it just aged,
    // which is the same property production depends on.
    //
    // ── Why `run_one_pass` rather than `run_pass` ───────────────────────────
    //
    // It is the production per-tick entry point, and the gauge-publish and
    // `enumeration_failed` decisions — half of what the fail-closed arms assert
    // — live in it rather than in `run_pass`. Its `budget` is `interval / 2`, so
    // every arm passes a generous [`TEST_INTERVAL`]: a budget elapse is its own
    // failure mode and is not what any of these are about.

    /// Interval handed to `run_one_pass`, whose only effect here is the pass
    /// budget (`interval / 2`). Generous on purpose — a budget elapse forfeits
    /// the pass, which would make every arm below flaky under CI load.
    const TEST_INTERVAL: Duration = Duration::from_secs(600);

    /// Grace handed to `run_one_pass`. Positive, and that is load-bearing:
    /// `recon_claim_expired`'s "a member cannot be returned by the call that
    /// registered it" guarantee holds only for a positive grace, so a zero here
    /// would collapse the two-observation protocol into one pass and every arm
    /// would silently stop testing it. The value is otherwise inert — candidates
    /// are aged by rewriting their score, never by waiting.
    const TEST_GRACE: Duration = Duration::from_secs(60);

    /// How far back a "not recently stored" stamp is dated — comfortably beyond
    /// [`TEST_GRACE`], so `stamp_cutoff` cannot be raced by a slow host.
    const OLD_STAMP_LAG_MS: u64 = 600_000;

    fn hash_of(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn hex_of(byte: u8) -> String {
        sha256_to_hex(&hash_of(byte))
    }

    fn store(tv: &TestValkey) -> ValkeyExternalValueStore {
        ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
    }

    /// Valkey's own clock, so a seeded `last_store_ms` is comparable with the
    /// `TIME` the pass samples in phase 1. A local `SystemTime::now()` would
    /// expose every stamp assertion to host/server skew — the same reason the
    /// production path reads the clock server-side.
    async fn now_ms(tv: &TestValkey) -> u64 {
        valkey_time_ms(tv.client())
            .await
            .expect("reading Valkey's clock must succeed")
    }

    /// The field-row index member [`seed_object_referencing`] uses for the
    /// row holding `hash_hex` — one row per (object, hash).
    fn seeded_row_member(hash_hex: &str) -> String {
        KeySchema::field_member("0", &format!("ref-{hash_hex}"))
    }

    /// Materialize one live object that references `hash_hex`: a header, one
    /// external-value field row holding the hash, its `obj:flds` member, and an
    /// `obj:ext` field counting that one row.
    ///
    /// `obj:hdr` is what the object walk SCANs; the row and `obj:ext` are the
    /// two sources it unions per object. The index must agree with the rows: a
    /// bare index entry with no row is an over-count, which the pass's
    /// per-object audit would repair (removing the field) and so change what
    /// the object references from the next pass on.
    async fn seed_object_referencing(tv: &TestValkey, object_id: &str, hash_hex: &str) {
        let keys = tv.key_schema();
        let _: () = tv
            .client()
            .pool
            .hset(keys.obj_header(object_id), [("seq", "1")])
            .await
            .expect("seeding obj:hdr must succeed");
        let member = seeded_row_member(hash_hex);
        let _: () = tv
            .client()
            .pool
            .hset(
                keys.obj_field_from_member(object_id, &member),
                [("hash", hash_hex), ("size", "1")],
            )
            .await
            .expect("seeding the value row must succeed");
        let _: i64 = tv
            .client()
            .pool
            .zadd(
                keys.obj_fields_index(object_id),
                None,
                None,
                false,
                false,
                (1.0, member.as_str()),
            )
            .await
            .expect("seeding obj:flds must succeed");
        let _: () = tv
            .client()
            .pool
            .hset(keys.obj_ext_index(object_id), [(hash_hex, "1")])
            .await
            .expect("seeding obj:ext must succeed");
    }

    /// Remove an object from the live population — every key the seed wrote,
    /// so the liveness picture the next walk sees is unambiguous.
    async fn unseed_object(tv: &TestValkey, object_id: &str) {
        let keys = tv.key_schema();
        let members: Vec<String> = tv
            .client()
            .pool
            .zrange(
                keys.obj_fields_index(object_id),
                0,
                -1,
                None,
                false,
                None,
                false,
            )
            .await
            .expect("reading obj:flds must succeed");
        let mut doomed = vec![
            keys.obj_header(object_id),
            keys.obj_ext_index(object_id),
            keys.obj_fields_index(object_id),
        ];
        for member in &members {
            doomed.push(keys.obj_field_from_member(object_id, member));
        }
        let _: u64 = tv
            .client()
            .pool
            .del(doomed)
            .await
            .expect("removing the object must succeed");
    }

    /// Overwrite a meta's `ref_count` outright.
    ///
    /// Staged by hand rather than through `store_values` because the counts the
    /// batch-cap arm needs (a surplus of 70 on one live object) would otherwise
    /// cost seventy round trips to produce a state that is, by definition, one no
    /// production path can reach — a `ref_count` that disagrees with the objects
    /// referencing the hash is exactly the corruption this pass exists to repair.
    /// The meta itself is still minted by the real `store_values`, so the blob,
    /// the `size` field and the data key are the production shapes.
    async fn set_ref_count(tv: &TestValkey, hash_hex: &str, count: u64) {
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().ext_meta(hash_hex),
                [(REF_COUNT_FIELD, count.to_string())],
            )
            .await
            .expect("seeding ref_count must succeed");
    }

    /// Date a meta's pre-flight stamp.
    ///
    /// A plain `HSET`, so it OVERWRITES whatever `ext_store_value.lua` wrote
    /// during the real `store_values` that minted the meta. That is what lets
    /// every arm below stage an arbitrary age on a meta the production writer
    /// has already stamped, and it is the same command the script itself uses —
    /// an `HSETNX` here would silently no-op post-b.rej and leave every "old
    /// stamp" arm running against a fresh one.
    async fn set_stamp(tv: &TestValkey, hash_hex: &str, ms: u64) {
        set_raw_stamp(tv, hash_hex, &ms.to_string()).await;
    }

    /// Write a raw `last_store_ms` value, INCLUDING one that is not a decimal
    /// integer — the corrupt-keyspace shape `classify_stamp` must fail closed on
    /// and no production writer can produce.
    async fn set_raw_stamp(tv: &TestValkey, hash_hex: &str, raw: &str) {
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().ext_meta(hash_hex),
                [(LAST_STORE_MS_FIELD, raw)],
            )
            .await
            .expect("seeding last_store_ms must succeed");
    }

    /// Strip a meta's `last_store_ms`, returning how many fields the `HDEL`
    /// actually removed.
    ///
    /// The only way to stage an UNSTAMPED meta post-b.rej: every
    /// `store_values` stamps, so an absent field can no longer be produced by
    /// simply not calling `set_stamp`. The return value is asserted by the
    /// caller — a `0` would mean the writer had not stamped and the arm was
    /// staging the wrong fixture.
    async fn clear_stamp(tv: &TestValkey, hash_hex: &str) -> u64 {
        tv.client()
            .pool
            .hdel(tv.key_schema().ext_meta(hash_hex), LAST_STORE_MS_FIELD)
            .await
            .expect("HDEL last_store_ms must succeed")
    }

    /// Rewrite every candidate's score to the epoch, so the next pass sees it as
    /// having aged a full grace.
    ///
    /// This is the whole of the time travel these tests do, and it is sound only
    /// because the registration is `ZADD NX`: the next pass re-offers the same
    /// member and the script declines to move its score.
    async fn age_candidates(tv: &TestValkey) {
        let key = tv.key_schema().ext_recon_candidates();
        for member in candidates(tv).await {
            let _: i64 = tv
                .client()
                .pool
                .zadd(&key, None, None, false, false, (0.0, member.as_str()))
                .await
                .expect("re-scoring a candidate must succeed");
        }
    }

    async fn candidates(tv: &TestValkey) -> Vec<String> {
        let mut members: Vec<String> = tv
            .client()
            .pool
            .zrange(
                tv.key_schema().ext_recon_candidates(),
                0,
                -1,
                None,
                false,
                None,
                false,
            )
            .await
            .expect("ZRANGE recon candidates must succeed");
        members.sort();
        members
    }

    async fn pending(tv: &TestValkey) -> Vec<String> {
        let mut members: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().ext_pending())
            .await
            .expect("SMEMBERS pending must succeed");
        members.sort();
        members
    }

    async fn ref_count(tv: &TestValkey, hash_hex: &str) -> Option<i64> {
        let raw: Option<String> = tv
            .client()
            .pool
            .hget(tv.key_schema().ext_meta(hash_hex), REF_COUNT_FIELD)
            .await
            .expect("HGET ref_count must succeed");
        raw.map(|v| v.parse().expect("ref_count must be an integer"))
    }

    /// `live_edit_ext_reconcile_total{outcome}`, panicking on an absent series:
    /// every cell is zero-seeded in `Metrics::new`, so absence means the pass
    /// was handed a different handle and every count read off it is vacuous.
    fn outcome(metrics: &Metrics, cell: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_ext_reconcile_total",
            &[("outcome", cell)],
        )
        .unwrap_or_else(|| {
            panic!(
                "live_edit_ext_reconcile_total{{outcome=\"{cell}\"}} is absent — it is zero-seeded \
                 at construction, so this can only be a different `Metrics`"
            )
        }) as u64
    }

    fn released_reconcile(metrics: &Metrics) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_ext_ref_released_total",
            &[("reason", "reconcile")],
        )
        .expect("live_edit_ext_ref_released_total{reason=\"reconcile\"} is zero-seeded")
            as u64
    }

    /// One reconciler GAUGE, as an `Option`.
    ///
    /// The four are published ONLY by `record_ext_reconcile_gauges`, so `None`
    /// is the observable form of "this pass published nothing at all" — which is
    /// exactly what a forfeited pass must do, and what two arms below assert.
    fn gauge(metrics: &Metrics, name: &str, replica: &str) -> Option<f64> {
        sample_value(&render(metrics), name, &[("service", replica)])
    }

    /// The four gauges as one tuple, for the arms that assert a whole reading.
    fn gauges(metrics: &Metrics, replica: &str) -> Option<(f64, f64, f64, f64)> {
        Some((
            gauge(metrics, "live_edit_ext_meta_total", replica)?,
            gauge(metrics, "live_edit_ext_referenced_hashes_total", replica)?,
            gauge(metrics, "live_edit_ext_ref_surplus_total", replica)?,
            gauge(metrics, "live_edit_ext_ref_deficit_total", replica)?,
        ))
    }

    async fn one_pass(tv: &TestValkey, metrics: &Arc<Metrics>, replica: &str) {
        run_one_pass(
            tv.client(),
            tv.key_schema(),
            replica,
            TEST_INTERVAL,
            TEST_GRACE,
            &CancellationToken::new(),
            Some(&(metrics.clone() as SharedMetrics)),
        )
        .await;
    }

    // ── The derived grace ───────────────────────────────────────────────────

    /// The grace is `max(600 s, 10 × RPC deadline)`, and it is DERIVED rather
    /// than configured.
    ///
    /// It is the guard that protects a legitimately in-flight pre-store from
    /// being read as surplus, so shortening it buys data loss rather than
    /// latency — which is why no env var reaches it. The floor and the scaling
    /// term are pinned separately because they bind on different deployments:
    /// the floor holds at the default 30 s deadline, and the scaling term is
    /// what keeps a deployment with a long deadline safe.
    #[test]
    fn the_reconcile_grace_is_the_larger_of_the_floor_and_ten_rpc_deadlines() {
        assert_eq!(
            ext_reconcile_grace(30_000),
            Duration::from_secs(600),
            "at the default 30 s RPC deadline the ten-minute FLOOR binds — 10 × 30 s is 300 s, \
             which is under it"
        );
        assert_eq!(
            ext_reconcile_grace(120_000),
            Duration::from_secs(1_200),
            "past 60 s of deadline the SCALING term binds: the longest in-flight window in the \
             service is the gap between a pre-store and the commit that justifies it, and that is \
             bounded by the RPC deadline"
        );
        assert_eq!(
            ext_reconcile_grace(0),
            Duration::from_secs(600),
            "and the floor still holds at a degenerate deadline, so no configuration can drive \
             the grace to zero"
        );
    }

    // ── 1. the surplus repair, end to end ───────────────────────────────────

    /// A surplus that holds still for a grace is registered on one pass and
    /// repaired on the next, through the ordinary GC release channel.
    ///
    /// The headline path. `ref_count` reads 2 for a hash exactly one live object
    /// references — the shape every residual on the ticket produces (a commit
    /// that died before its enqueue, a restore's suppressed surplus, a reference
    /// leaked before b.qpf) — and nothing else in the service can see it: the GC
    /// only performs decrements someone owed it, so an increment nobody owes a
    /// decrement for is invisible to it.
    ///
    /// Four claims:
    ///
    /// * **Pass 1 repairs NOTHING.** A single walk's arithmetic is not evidence:
    ///   `SCAN` may miss a key that exists throughout, and acting on one reading
    ///   would decrement a live reference. The pass registers and returns.
    /// * **Pass 2 emits the repair as a RELEASE MEMBER**, not an `HINCRBY`.
    ///   Riding the pending → claim → `gc_decrement_and_collect` protocol gives
    ///   the repair that protocol's crash-safety, underflow accounting and
    ///   collect-at-zero behaviour; a bare decrement here would have none of them
    ///   and would race the GC's own.
    /// * **The member spells `{hash}:0:rec{n}`.** The `0` is the documented
    ///   no-object sentinel (object ids are minted by `INCR` from 1, so no real
    ///   object can spell one) and the amount rides in the token, so a REPLAY of
    ///   this repair collapses in the Set while a genuinely different repair of
    ///   the same hash, at a different amount, stays distinct.
    /// * **The GC then really decrements**, to exactly the live-object count,
    ///   and does NOT collect: one object still references the blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_standing_surplus_is_registered_on_one_pass_and_repaired_on_the_next() {
        const REPLICA: &str = "yda-recon-repair";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x11);

        // Two increments, one live object: the surplus the pass exists to drain.
        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0x11), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        seed_object_referencing(&tv, "940001", &hex).await;
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
        assert_eq!(ref_count(&tv, &hex).await, Some(2));

        // ── Pass 1: observe and register, repair nothing ────────────────────
        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            candidates(&tv).await,
            vec![format!("{hex}:1")],
            "the candidate carries the AMOUNT, not just the hash. Anything transient — a \
             pre-store awaiting its commit, an enqueue awaiting the GC, an object the walk missed \
             — changes the amount, produces a DIFFERENT member and restarts the grace, so what \
             survives to be repaired is a surplus that genuinely stood still"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED),
            1
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED),
            0,
            "one observation is not evidence. `SCAN` may miss a key that exists for the whole \
             walk, so repairing on a single reading would decrement a reference that is live"
        );
        assert!(
            pending(&tv).await.is_empty(),
            "and nothing may reach the release channel from a first observation"
        );
        assert_eq!(
            gauges(&metrics, REPLICA),
            Some((1.0, 1.0, 1.0, 0.0)),
            "one meta, one referenced hash, one surplus reference, no deficit. The four publish \
             together because they are only meaningful against each other"
        );
        assert_eq!(
            ref_count(&tv, &hex).await,
            Some(2),
            "and the keyspace is untouched"
        );

        // ── Pass 2, a grace later ───────────────────────────────────────────
        age_candidates(&tv).await;
        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            pending(&tv).await,
            vec![format!("{hex}:0:rec1")],
            "the repair rides the ordinary release channel as one member per owed decrement. The \
             `0` is the no-object sentinel — this release belongs to no object, because it hands \
             back an increment whose owning object is unknown — and the `rec1` token carries the \
             amount so a replay of THIS repair collapses in the Set instead of decrementing twice"
        );
        assert_eq!(outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED), 1);
        assert_eq!(
            released_reconcile(&metrics),
            1,
            "counted under its own `reason` cell, in REFERENCES — the same unit `rewrite`, \
             `surplus_acquire` and `retirement` use, so an operator can read the four together"
        );
        assert!(
            candidates(&tv).await.is_empty(),
            "and the candidate is retired only AFTER the repair landed: the reverse order would \
             lose it on a failed enqueue"
        );

        // ── The GC turns the member into an actual decrement ────────────────
        let stats = process_pending_gc(
            &ext,
            None,
            crate::storage::external_values::GcPolicy::IMMEDIATE,
        )
        .await;
        assert_eq!(stats.processed, 1, "one owed decrement, one claim");
        assert_eq!(
            stats.deleted, 0,
            "and NO collect: one live object still references the blob. A collect here would be \
             the over-release this whole design refuses"
        );
        assert_eq!(
            ref_count(&tv, &hex).await,
            Some(1),
            "the count now matches the live-object count, which is what `ref_count` means"
        );
    }

    // ── 1b. the repair AMOUNT, above one and above the per-hash cap ─────────

    /// A surplus of `surplus` is repaired as `surplus` DISTINCT members, in
    /// batches of at most [`EXT_RECON_MAX_REPAIR_PER_HASH`], across as many
    /// passes as that takes — and ends at exactly the live-object count.
    ///
    /// The arm above proves the protocol at a surplus of ONE, where the whole
    /// member-emission loop degenerates: `(0..to_repair)` runs once, the cap
    /// cannot bind, and `claimed_surplus - i` is indistinguishable from a bare
    /// `claimed_surplus`, from a constant, or from an index. All three of those
    /// are wrong, and each is wrong in its own direction:
    ///
    /// * **One member per repair, whatever the surplus.** Hands back one
    ///   reference against a surplus of `n` and leaves `n - 1` standing forever —
    ///   the leak the pass exists to drain, drained at one per grace.
    /// * **`n` members that all spell the same thing.** The release channel is a
    ///   `SADD`, so a Set collapses them to ONE. Reads identically to the above
    ///   from the emitter's side and is the reason the amount rides in the token
    ///   at all.
    /// * **The index rather than the amount** (`rec0`, `rec1`, …). Distinct
    ///   within one repair, but a replay of the SAME repair a grace later spells
    ///   the same tokens for a DIFFERENT amount, and two genuinely different
    ///   repairs of one hash collide — so a replay decrements twice.
    ///
    /// The unit test in `external_values.rs` pins the member *spelling* against a
    /// closure that reproduces the emitter's arithmetic; this arm is the one that
    /// drives the real `act_on_candidate` and reads what actually landed on
    /// `pending`, so a divergence between the two shows up here rather than in a
    /// test agreeing with itself.
    ///
    /// # Why the cap needs its own amount
    ///
    /// `to_repair = claimed_surplus.min(EXT_RECON_MAX_REPAIR_PER_HASH)` is a
    /// blast-radius bound on a single pass, NOT a ceiling on what is repairable:
    /// the remainder must come back as a fresh candidate at the smaller amount
    /// and drain on a later pass. Two failures live here and neither is visible
    /// below the cap — a build that repaired `min(...)` and then RETIRED the
    /// hash as done would strand the remainder permanently, and one that emitted
    /// all 70 members would put the whole surplus through in one pass, which is
    /// the bound's whole purpose. The 70-reference arm is what separates
    /// "capped" from "capped and resumable".
    async fn a_standing_surplus_is_repaired_in_capped_batches(
        surplus: u64,
        hash_byte: u8,
        object_id: &str,
        replica: &str,
    ) {
        /// How many live objects reference the hash throughout. The floor every
        /// repair must stop at, and never pass.
        const LIVE: u64 = 1;

        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(replica));
        let ext = store(&tv);
        let hex = hex_of(hash_byte);

        // The meta and its blob through the real store; the corrupt count by
        // hand, because no production path produces one.
        ext.store_values(&[(&hash_of(hash_byte), b"recon-payload".as_slice())])
            .await
            .expect("the pre-flight store must succeed");
        seed_object_referencing(&tv, object_id, &hex).await;
        set_ref_count(&tv, &hex, LIVE + surplus).await;
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
        assert_eq!(
            ref_count(&tv, &hex).await,
            Some(i64::try_from(LIVE + surplus).expect("the fixture count fits an i64"))
        );

        let expected_rounds = surplus.div_ceil(EXT_RECON_MAX_REPAIR_PER_HASH);
        let mut remaining = surplus;
        let mut rounds = 0;
        let mut repaired_references = 0;

        while remaining > 0 {
            rounds += 1;
            assert!(
                rounds <= expected_rounds,
                "the repair must converge in {expected_rounds} round(s) — {remaining} reference(s) \
                 were still owed after {}. A round that repairs nothing means the remainder is not \
                 being re-observed at its new amount",
                rounds - 1
            );

            // ── Observe. The amount is the candidate's identity ─────────────
            one_pass(&tv, &metrics, replica).await;
            assert_eq!(
                candidates(&tv).await,
                vec![format!("{hex}:{remaining}")],
                "round {rounds}: the surplus still standing is {remaining}, and the candidate \
                 member must say so. After a capped repair this is the assertion that the \
                 REMAINDER came back as a candidate in its own right rather than being carried \
                 over at the old amount (which a fresh re-read would then void) or dropped"
            );

            // ── Repair, a grace later ───────────────────────────────────────
            age_candidates(&tv).await;
            one_pass(&tv, &metrics, replica).await;

            let batch = remaining.min(EXT_RECON_MAX_REPAIR_PER_HASH);
            let mut expected: Vec<String> = (0..batch)
                .map(|i| format!("{hex}:0:rec{}", remaining - i))
                .collect();
            expected.sort();
            assert_eq!(
                pending(&tv).await,
                expected,
                "round {rounds}: exactly {batch} members, spelled by counting the amount DOWN from \
                 {remaining}. Asserted as a whole set because every failure mode here is a set \
                 identity: one member (or {batch} equal ones, which the Set collapses to one) hands \
                 back one reference against {remaining}; index-spelled tokens would replay onto a \
                 different amount and decrement twice"
            );
            repaired_references += batch;
            assert_eq!(
                released_reconcile(&metrics),
                repaired_references,
                "round {rounds}: the counter is in REFERENCES, so it moves by the batch size, not \
                 by one per hash — an operator reading `reconcile` against `rewrite` and \
                 `retirement` is comparing like with like only if all four count the same unit"
            );
            assert_eq!(
                outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED),
                rounds,
                "while the OUTCOME cell counts hashes acted on: one per pass that repaired"
            );
            assert!(
                candidates(&tv).await.is_empty(),
                "round {rounds}: the candidate is retired once its repair landed — including the \
                 capped round, where the hash is NOT done. Leaving it would re-claim a member whose \
                 amount the fresh re-read now disagrees with, which voids it as `healed` and \
                 strands the remainder for a whole extra grace"
            );

            // ── The GC turns the members into actual decrements ─────────────
            let stats = process_pending_gc(
                &ext,
                None,
                crate::storage::external_values::GcPolicy::IMMEDIATE,
            )
            .await;
            assert_eq!(
                stats.processed,
                usize::try_from(batch).expect("a batch is at most the per-hash cap"),
                "round {rounds}: every member is independently owed, so the GC claims all {batch}"
            );
            assert_eq!(
                stats.deleted, 0,
                "and never collects: one live object references the blob throughout, at every \
                 intermediate count this walk passes through"
            );
            remaining -= batch;
            assert_eq!(
                ref_count(&tv, &hex).await,
                Some(i64::try_from(LIVE + remaining).expect("the fixture count fits an i64")),
                "round {rounds}: the count steps down by exactly the batch — never past the live \
                 count, which is the over-release this whole design refuses"
            );
        }

        assert_eq!(rounds, expected_rounds);
        assert_eq!(
            ref_count(&tv, &hex).await,
            Some(i64::try_from(LIVE).expect("LIVE fits an i64")),
            "and the walk ENDS at the live-object count, which is what `ref_count` means"
        );
        assert_eq!(
            released_reconcile(&metrics),
            surplus,
            "having handed back exactly the surplus it claimed — no more (over-release, i.e. data \
             loss) and no less (the leak that survives)"
        );

        // One more pass over the settled keyspace: nothing left to see.
        one_pass(&tv, &metrics, replica).await;
        assert!(
            candidates(&tv).await.is_empty(),
            "a settled hash registers no candidate at all — `ref_count` now equals `referenced`, \
             so the surplus is zero and phase 4 has nothing to offer"
        );
        assert!(pending(&tv).await.is_empty());
        assert_eq!(
            gauges(&metrics, replica),
            Some((1.0, 1.0, 0.0, 0.0)),
            "and the gauges say so: one meta, one referenced hash, ZERO surplus, no deficit"
        );
    }

    /// Three references: above the degenerate case, below the cap. One pass
    /// repairs all three.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_surplus_of_three_is_repaired_as_three_distinct_members_in_one_batch() {
        a_standing_surplus_is_repaired_in_capped_batches(3, 0x31, "940050", "yda-recon-surplus-3")
            .await;
    }

    /// Seventy references: above the cap, so the repair takes two passes and the
    /// remainder has to be re-observed at its new amount in between.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_surplus_above_the_per_hash_cap_is_repaired_across_two_passes() {
        a_standing_surplus_is_repaired_in_capped_batches(
            70,
            0x46,
            "940051",
            "yda-recon-surplus-70",
        )
        .await;
    }

    // ── 2. the `last_store_ms` eligibility gate ─────────────────────────────

    /// A meta carrying NO `last_store_ms` is **ELIGIBLE**: it registers on one
    /// pass and is repaired on the next, exactly like a comfortably-dated one.
    ///
    /// **This arm is the inversion itself, and it is the regression test for
    /// b.rej's second half.** Before b.rej the same fixture asserted the exact
    /// opposite — nothing registered, nothing repaired, the skip cell ticked —
    /// because no writer stamped and an undatable meta could equally have been a
    /// peer's in-flight pre-store. Now every pre-flight store stamps, so an
    /// absent field is positive evidence that NO store has touched this hash
    /// since the writer deployed, and its surplus is precisely the standing leak
    /// the pass exists to drain. Revert either half of the change — the writer
    /// or the `StampState::Absent` arm — and this test goes red: without the
    /// `Absent => eligible` arm nothing is ever registered here.
    ///
    /// **Why the fixture has to DELETE the stamp.** Post-b.rej an unstamped meta
    /// is no longer the default state; it cannot be staged by simply not calling
    /// `set_stamp`, because the real `store_values` below writes one. `HDEL`
    /// after the stores is the only honest model of the legacy meta this arm is
    /// about: a `ref_count` raised by a build that predates the writer, or by a
    /// path since fixed. The `HDEL` must report that it removed a field —
    /// asserted below — which is also a second, independent pin that the writer
    /// really stamps.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_legacy_meta_with_no_pre_flight_stamp_is_registered_and_repaired() {
        const REPLICA: &str = "rej-recon-no-stamp";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x22);

        // Two increments, one live object: a surplus of one.
        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0x22), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        seed_object_referencing(&tv, "940002", &hex).await;
        assert_eq!(
            clear_stamp(&tv, &hex).await,
            1,
            "the fixture models a LEGACY meta by removing the stamp the real `store_values` \
             wrote, so the `HDEL` must actually remove something. A zero here means the writer \
             never stamped — in which case this arm is staging the pre-b.rej keyspace by \
             accident rather than modelling it deliberately, and the b.rej writer is broken"
        );

        // ── Pass 1: an undatable meta is now a first observation ────────────
        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            candidates(&tv).await,
            vec![format!("{hex}:1")],
            "an ABSENT stamp is eligible post-b.rej: every pre-flight store stamps, so the field \
             missing is positive evidence that none has touched this hash. Pre-b.rej this list \
             was asserted EMPTY — that flip is what this arm exists for"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED),
            1
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_UNREADABLE_STAMP),
            0,
            "and absent must NOT be reported as unreadable. The two are opposite verdicts now, so \
             folding them back together would read a corrupt value as `nothing ever stored this` \
             — a repair licensed by a field the pass failed to understand"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_RECENT_STORE),
            0,
            "nor as a recent store: there is no stamp to be inside the grace"
        );
        assert_eq!(
            gauges(&metrics, REPLICA),
            Some((1.0, 1.0, 1.0, 0.0)),
            "one meta, one referenced hash, one surplus reference, no deficit"
        );
        assert_eq!(
            ref_count(&tv, &hex).await,
            Some(2),
            "one observation still repairs nothing — the two-observation protocol is unchanged"
        );

        // ── Pass 2, a grace later: the repair the old posture refused ───────
        age_candidates(&tv).await;
        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            pending(&tv).await,
            vec![format!("{hex}:0:rec1")],
            "the surplus is handed back on the ordinary release channel. Pre-b.rej no number of \
             passes could produce this member, which was the whole `armed and idle` posture"
        );
        assert_eq!(outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED), 1);
        assert!(candidates(&tv).await.is_empty());
    }

    /// The writer's OWN stamp — never one this test wrote — is what the pass
    /// dates a meta by.
    ///
    /// Every other arm in this section hand-seeds `last_store_ms` and so proves
    /// only that the READER honours a field; none of them would notice a writer
    /// that stamped the wrong key, the wrong field name, or a value the reader
    /// cannot parse. This arm seeds nothing: the meta is minted by two real
    /// `store_values` calls and judged on what they left behind.
    ///
    /// The three passes are the three verdicts the gate can reach, in the order
    /// a real hash reaches them:
    ///
    /// 1. **Freshly stored ⇒ `skipped_recent_store`.** The stamp the writer just
    ///    wrote is inside the grace, so the pass declines — this is the in-flight
    ///    pre-store the whole guard exists for. A writer that stamped nothing
    ///    would REGISTER here instead (absent is eligible), which is how this
    ///    step catches a missing stamp rather than a wrong one.
    /// 2. **Aged ⇒ registered.** Only the stamp moved between passes 1 and 2:
    ///    same count, same objects, same everything else. So the transition is
    ///    attributable to the field the writer wrote and nothing else.
    /// 3. **A grace later ⇒ repaired**, to exactly the live-object count.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_writers_own_stamp_gates_the_pass_from_skip_to_repair() {
        const REPLICA: &str = "rej-recon-writer-stamp";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x24);

        // Surplus 1 vs live 1, entirely through the production writer.
        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0x24), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        seed_object_referencing(&tv, "940024", &hex).await;

        // ── Pass 1: the writer's stamp is fresh ─────────────────────────────
        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_RECENT_STORE),
            1,
            "the pass must read the stamp `store_values` itself wrote and find it inside the \
             grace. This is the load-bearing step: a writer that stamped nothing, or stamped a \
             field by another name, leaves the meta ABSENT — which is now ELIGIBLE, so the pass \
             would register a candidate here instead of skipping it"
        );
        assert!(
            candidates(&tv).await.is_empty(),
            "and an increment that may still be awaiting its commit is not pinned as a first \
             observation"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_UNREADABLE_STAMP),
            0,
            "the value the writer wrote must PARSE. `string.format(\"%d\", ms)` exists precisely \
             so a large Lua double never reaches the field as `1.7654e+12`, which the reader \
             would classify as corrupt keyspace and fail closed on forever"
        );

        // ── Pass 2: age the writer's stamp; nothing else changes ────────────
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            candidates(&tv).await,
            vec![format!("{hex}:1")],
            "the ONLY thing that changed between passes 1 and 2 is the age of the stamp, so the \
             skip → register transition is attributable to that field and nothing else"
        );

        // ── Pass 3, a grace later ───────────────────────────────────────────
        age_candidates(&tv).await;
        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(pending(&tv).await, vec![format!("{hex}:0:rec1")]);
        assert_eq!(outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED), 1);

        let stats = process_pending_gc(
            &ext,
            None,
            crate::storage::external_values::GcPolicy::IMMEDIATE,
        )
        .await;
        assert_eq!(stats.processed, 1);
        assert_eq!(
            ref_count(&tv, &hex).await,
            Some(1),
            "and the count lands on the live-object count — the repair is real, not just a \
             member on a Set"
        );
    }

    /// A `last_store_ms` that is PRESENT but unparseable is ineligible, on its
    /// own cell, and stays that way pass after pass.
    ///
    /// This is the fail-closed half of the three-way gate, and it is only
    /// meaningful because the absent case became eligible in the same commit: a
    /// reader that folded the two together would read a corrupt or foreign value
    /// as "nothing ever stored this hash" and license a repair on the strength of
    /// a field it could not understand. Nothing in the service can write such a
    /// value, so reaching this state at all is evidence about the keyspace, not
    /// about load — hence a cell of its own rather than a share of
    /// `skipped_recent_store`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_meta_whose_stamp_is_unparseable_is_skipped_and_never_repaired() {
        const REPLICA: &str = "rej-recon-garbage-stamp";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x23);

        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0x23), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        seed_object_referencing(&tv, "940023", &hex).await;
        // Not a decimal-millisecond integer. The shape a hand-edited key, or a
        // Lua `tostring` on a large double, would leave behind.
        set_raw_stamp(&tv, &hex, "1.7654e+12").await;

        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_UNREADABLE_STAMP),
            1,
            "a value the pass cannot parse must be REFUSED and counted, not silently treated as \
             absence — absence is now eligible, so folding the two together would repair against \
             a corrupt field"
        );
        assert!(
            candidates(&tv).await.is_empty(),
            "and nothing may be pinned on the strength of a stamp the pass failed to read"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED),
            0
        );
        assert_eq!(
            gauges(&metrics, REPLICA),
            Some((1.0, 1.0, 1.0, 0.0)),
            "the surplus is still REPORTED: the pass declines to act, not to look"
        );

        // Ageing changes nothing — nothing was registered to age.
        age_candidates(&tv).await;
        one_pass(&tv, &metrics, REPLICA).await;
        assert!(
            pending(&tv).await.is_empty(),
            "and no number of passes turns an unreadable stamp into a repair"
        );
        assert_eq!(outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED), 0);
        assert_eq!(
            ref_count(&tv, &hex).await,
            Some(2),
            "the keyspace is untouched throughout"
        );
    }

    /// A stamp that becomes unreadable BETWEEN the two observations is refused
    /// at the fresh re-read, and the candidate stays registered.
    ///
    /// `act_on_candidate` re-reads the stamp precisely because the phase-4 one is
    /// a pass old, and this is the arm that proves the re-read runs the same
    /// three-way test rather than the old two-state one. The member keeping its
    /// first-observation score is the other half of the contract: the surplus may
    /// well be real, so the pass defers judgement to a later pass with a stamp it
    /// can date instead of retiring the candidate and losing the observation.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stamp_that_turns_unreadable_between_passes_is_skipped_at_the_fresh_re_read() {
        const REPLICA: &str = "rej-recon-stamp-rots";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x25);

        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0x25), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        seed_object_referencing(&tv, "940025", &hex).await;
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;

        // ── Pass 1: a comfortably-dated surplus registers ───────────────────
        one_pass(&tv, &metrics, REPLICA).await;
        assert_eq!(candidates(&tv).await, vec![format!("{hex}:1")]);

        // ── The stamp rots between the observations ─────────────────────────
        set_raw_stamp(&tv, &hex, "not-a-timestamp").await;
        age_candidates(&tv).await;
        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_UNREADABLE_STAMP),
            2,
            "TWO ticks on one pass, and both are correct: phase 4 re-walks the meta and refuses \
             to offer it as a candidate again, then the claim phase hands the ALREADY-registered \
             member to `act_on_candidate`, whose fresh re-read refuses it a second time. The \
             second tick is the one this arm is about — see the `repaired` assertion below for \
             what distinguishes it from phase 4 acting alone"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED),
            0,
            "and nothing is repaired on a stamp the pass could not read. This is the assertion \
             that isolates the FRESH re-read: phase 4's refusal cannot prevent a repair — the \
             member was registered a pass ago and is claimable now — so a re-read that kept the \
             old `parse().ok()` shape would see `None`, read it as the (now eligible) ABSENT case, \
             and hand back a reference on the strength of a field it could not parse"
        );
        assert!(pending(&tv).await.is_empty());
        assert_eq!(
            candidates(&tv).await,
            vec![format!("{hex}:1")],
            "the member keeps its FIRST-observation score rather than being retired: the surplus \
             may well be real, and retiring it here would throw away the observation and restart \
             the grace on the next pass"
        );
        assert_eq!(ref_count(&tv, &hex).await, Some(2));
    }

    /// `classify_stamp`'s four cases, at the boundary the two `Option` layers
    /// draw.
    ///
    /// The two layers mean different things and get OPPOSITE verdicts, which is
    /// exactly the distinction a two-state `Option<u64>` cannot carry — and the
    /// outer `None` is the round-2 fail-closed correction: an `HMGET` answers one
    /// element per requested field, so a reply with no slot at all is not absence
    /// but evidence that something other than this pass's `HMGET` produced it.
    /// Reading it as `Absent` would make a short reply eligible for repair.
    ///
    /// A unit test rather than an integration arm because the short-arity reply
    /// cannot be staged against a real Valkey at all: `HMGET` will not produce
    /// one, which is the very reason the case fails closed.
    #[test]
    fn classify_stamp_separates_absent_from_unreadable() {
        assert_eq!(
            classify_stamp(None),
            StampState::Unreadable,
            "a reply with NO slot for the field is corruption-shaped, not absence — `HMGET` \
             answers one element per requested field, so this cannot come from the pass's own \
             read. Fail closed: classifying it `Absent` would make it eligible for repair"
        );
        assert_eq!(
            classify_stamp(Some(&None)),
            StampState::Absent,
            "Valkey's nil for a field the hash does not carry is the genuine absent case, and \
             post-b.rej that is ELIGIBLE"
        );
        assert_eq!(
            classify_stamp(Some(&Some("garbage".to_string()))),
            StampState::Unreadable,
            "a present-but-unparseable value is refused rather than folded into absence"
        );
        assert_eq!(
            classify_stamp(Some(&Some("123".to_string()))),
            StampState::At(123),
            "and a decimal-millisecond integer parses to the instant it names — the shape \
             `ext_store_value.lua`'s `string.format(\"%d\", ms)` guarantees"
        );
    }

    /// A meta stamped INSIDE the grace is ineligible too, and says so on its own
    /// cell.
    ///
    /// This is the case the gate actually exists for once b.rej lands: an upload
    /// whose pre-flight `+1` has not reached its commit yet presents as a
    /// perfectly stable surplus to both of the other guards. Repairing it hands
    /// back an increment a live commit is about to rely on — an over-release
    /// from the pass whose whole purpose is to refuse one.
    ///
    /// Its own cell rather than sharing `skipped_unreadable_stamp`, because the
    /// two mean different things operationally: `skipped_recent_store` climbing
    /// is a busy store working correctly, while `skipped_unreadable_stamp`
    /// climbing is a keyspace-integrity signal — nothing in the service writes
    /// an unparseable stamp.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_meta_stamped_inside_the_grace_is_skipped_as_a_recent_store() {
        const REPLICA: &str = "yda-recon-recent";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x33);

        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0x33), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        seed_object_referencing(&tv, "940003", &hex).await;
        set_stamp(&tv, &hex, now_ms(&tv).await).await;

        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_RECENT_STORE),
            1,
            "a stamp inside the grace means an increment may still be awaiting its commit — the \
             one thing that presents as a stable surplus to every other guard this pass has"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_UNREADABLE_STAMP),
            0,
            "and it must not be reported on the OTHER skip cell: `skipped_recent_store` climbing \
             is a busy store working, `skipped_unreadable_stamp` climbing is corrupt keyspace"
        );
        assert!(candidates(&tv).await.is_empty());
        assert!(pending(&tv).await.is_empty());
    }

    // ── 3. the `owed` subtraction ───────────────────────────────────────────

    /// Which release-channel stage holds the already-owed decrement.
    #[derive(Clone, Copy, Debug)]
    enum OwedStage {
        /// `le:ext:{gc}:pending` — enqueued, not yet claimed.
        Pending,
        /// `le:ext:{gc}:inflight` — claimed by a GC worker, not yet applied;
        /// scored with a bare claim time, as an older build claims (b.1jt).
        Inflight,
        /// `le:ext:{gc}:inflight`, scored as this build claims — claim ms +
        /// `GC_CLAIM_SCORE_OFFSET_MS` (b.1jt). The walk ignores the score, so
        /// both encodings are owed alike.
        InflightEncoded,
        /// `le:{oid}:obj:extret` — this build's retirement carrier, a HASH
        /// (b.1jt), which the orphan sweep's retire-retry phase settles: it
        /// clears the carrier once an attempt is known landed, pays the
        /// members onto `pending`, or cancels them over a live id.
        RetireCarrier,
        /// `le:{oid}:obj:extret` written as an older build writes it, a SET,
        /// which the retire-retry phase clears or drops but never pays
        /// (b.1jt). Owed until then all the same.
        LegacyRetireCarrier,
    }

    impl OwedStage {
        fn label(self) -> &'static str {
            match self {
                OwedStage::Pending => "pending",
                OwedStage::Inflight => "inflight",
                OwedStage::InflightEncoded => "inflight-encoded",
                OwedStage::RetireCarrier => "carrier",
                OwedStage::LegacyRetireCarrier => "legacy-carrier",
            }
        }
    }

    /// A decrement the GC already owes is subtracted from the surplus, from
    /// whichever of the three stages currently holds it — in either of the
    /// encodings an `inflight` score and a carrier can have since b.1jt.
    ///
    /// Counting an owed release as surplus repairs it a second time, and the GC
    /// then decrements twice for one reference given up once. This is also what
    /// makes the pass idempotent against ITSELF: between emitting a repair and
    /// the GC draining it, the repair sits in `owed`, so the surplus reads zero
    /// and the candidate retires as healed rather than being repaired again.
    ///
    /// The retire CARRIER arm is the one that was missing in round 1, and it is
    /// the worst of the three: a carrier normally belongs to an object that has
    /// already been torn down, so it contributes NOTHING to `referenced` —
    /// there is no `obj:hdr` left to scan — and it does not change while it
    /// waits. Its whole amount therefore reads as a pure, perfectly stable
    /// surplus that sails through the two-observation guard and gets repaired,
    /// after which the sweep drains the carrier and the GC decrements again.
    ///
    /// The control is the same fixture with the owed member absent: the surplus
    /// is 1 and a candidate IS registered. Without it, "no candidate" is
    /// satisfied by a pass that registers nothing at all.
    async fn an_already_owed_decrement_is_not_surplus(stage: OwedStage) {
        let replica = format!("yda-recon-owed-{}", stage.label());
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(&replica));
        let ext = store(&tv);
        let hex = hex_of(0x44);
        let keys = tv.key_schema();

        // One increment, NO live object: the object retired and its release is
        // sitting somewhere in the channel.
        ext.store_values(&[(&hash_of(0x44), b"recon-payload".as_slice())])
            .await
            .expect("the pre-flight store must succeed");
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;

        // ── The control: nothing owed, so the surplus is real ───────────────
        one_pass(&tv, &metrics, &replica).await;
        assert_eq!(
            candidates(&tv).await,
            vec![format!("{hex}:1")],
            "control: with the release nowhere in the channel, the increment IS a surplus and the \
             pass pins it. Without this the assertion below is satisfied by a pass that pins \
             nothing at all"
        );
        let _: i64 = tv
            .client()
            .pool
            .del(keys.ext_recon_candidates())
            .await
            .expect("clearing the candidate ZSET must succeed");

        // ── Now stage the owed decrement ────────────────────────────────────
        let member = ExtRelease::retirement(hash_of(0x44), 940_004).member();
        match stage {
            OwedStage::Pending => {
                let _: u64 = tv
                    .client()
                    .pool
                    .sadd(keys.ext_pending(), member.as_str())
                    .await
                    .expect("seeding pending must succeed");
            }
            OwedStage::Inflight => {
                let _: i64 = tv
                    .client()
                    .pool
                    .zadd(
                        keys.ext_inflight(),
                        None,
                        None,
                        false,
                        false,
                        (1.0, member.as_str()),
                    )
                    .await
                    .expect("seeding inflight must succeed");
            }
            OwedStage::InflightEncoded => {
                let encoded = (now_ms(&tv).await + GC_CLAIM_SCORE_OFFSET_MS) as f64;
                let _: i64 = tv
                    .client()
                    .pool
                    .zadd(
                        keys.ext_inflight(),
                        None,
                        None,
                        false,
                        false,
                        (encoded, member.as_str()),
                    )
                    .await
                    .expect("seeding inflight must succeed");
            }
            OwedStage::RetireCarrier => {
                let _: () = tv
                    .client()
                    .pool
                    .hset(
                        keys.obj_ext_retire_carrier("940004"),
                        [(member.as_str(), "1")],
                    )
                    .await
                    .expect("seeding the retire carrier must succeed");
            }
            OwedStage::LegacyRetireCarrier => {
                let _: u64 = tv
                    .client()
                    .pool
                    .sadd(keys.obj_ext_retire_carrier("940004"), member.as_str())
                    .await
                    .expect("seeding the older build's retire carrier must succeed");
            }
        }

        one_pass(&tv, &metrics, &replica).await;

        assert!(
            candidates(&tv).await.is_empty(),
            "the decrement is already owed from `{}`, so the increment is NOT surplus. Repairing \
             it would emit a second release for one reference given up once, and the GC would \
             decrement twice — an over-release, with meta+data deleted under whatever else still \
             points at the blob",
            stage.label()
        );
        assert_eq!(
            gauge(
                metrics.as_ref(),
                "live_edit_ext_ref_surplus_total",
                &replica
            ),
            Some(0.0),
            "and the gauge must agree: the surplus really is zero, not merely unregistered"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_pending_release_is_not_counted_as_surplus() {
        an_already_owed_decrement_is_not_surplus(OwedStage::Pending).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_inflight_release_is_not_counted_as_surplus() {
        an_already_owed_decrement_is_not_surplus(OwedStage::Inflight).await;
    }

    /// This build's claim, scored claim ms + the offset (b.1jt), is owed like
    /// a bare-scored one: the walk reads membership, never the score.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_encoded_inflight_claim_is_not_counted_as_surplus() {
        an_already_owed_decrement_is_not_surplus(OwedStage::InflightEncoded).await;
    }

    /// This build's carrier is a HASH (b.1jt). **Fails on the build before
    /// b.1jt**: its walk read every carrier with `SMEMBERS`, which fails
    /// `WRONGTYPE` on a HASH and forfeits the pass, so the surplus gauge
    /// still reads the control pass's 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_deferred_retirement_release_on_a_carrier_is_not_counted_as_surplus() {
        an_already_owed_decrement_is_not_surplus(OwedStage::RetireCarrier).await;
    }

    /// An older build's SET carrier is owed too, until the orphan sweep clears
    /// or drops it: the walk dispatches on the carrier's type (b.1jt).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_release_on_an_older_build_set_carrier_is_not_counted_as_surplus() {
        an_already_owed_decrement_is_not_surplus(OwedStage::LegacyRetireCarrier).await;
    }

    // ── 4. the claim-time re-read ───────────────────────────────────────────

    /// A candidate whose surplus MOVED between the two observations is retired
    /// as healed, with nothing emitted.
    ///
    /// The claim proves the amount was stable a grace ago; the fresh `HMGET`
    /// proves it is still stable now, at the instant of the repair. Without the
    /// second read the pass would apply a repair sized by a reading up to half a
    /// tick interval old — and the amount IS the whole claim, so a different
    /// amount is a different candidate.
    ///
    /// The new amount is re-registered by the same pass's phase 4, starting its
    /// own grace. That is the protocol converging rather than a leak: a surplus
    /// that is genuinely 2 gets repaired as 2, a grace later.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_candidate_whose_amount_moved_heals_instead_of_repairing() {
        const REPLICA: &str = "yda-recon-healed";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x55);

        ext.store_values(&[(&hash_of(0x55), b"recon-payload".as_slice())])
            .await
            .expect("the pre-flight store must succeed");
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;

        one_pass(&tv, &metrics, REPLICA).await;
        assert_eq!(candidates(&tv).await, vec![format!("{hex}:1")]);
        age_candidates(&tv).await;

        // The surplus moves between the two observations.
        let _: i64 = tv
            .client()
            .pool
            .hincrby(tv.key_schema().ext_meta(&hex), REF_COUNT_FIELD, 1)
            .await
            .expect("moving ref_count must succeed");

        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_HEALED),
            1,
            "the amount is the whole claim, so a different amount is a different candidate and \
             this one is void"
        );
        assert_eq!(outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED), 0);
        assert!(
            pending(&tv).await.is_empty(),
            "and NOTHING may be emitted — a repair sized by the stale amount would hand back one \
             reference against a surplus of two, or two against a surplus of one"
        );
        assert_eq!(
            candidates(&tv).await,
            vec![format!("{hex}:2")],
            "the new amount is registered as its own first observation, so the protocol converges \
             on it a grace later rather than losing the surplus"
        );
    }

    /// A candidate whose META is gone is retired, not carried forever.
    ///
    /// The amount check runs deliberately AHEAD of the eligibility gate for this
    /// case: a meta that no longer exists reaches the re-read with
    /// `fresh_surplus == None`, and gating on the stamp first would send it down
    /// the "no stamp" arm — which leaves the member REGISTERED, forever, since
    /// the meta it names will never come back to change the answer. Retiring is
    /// not a repair, so letting the void candidates go first weakens nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_candidate_whose_meta_vanished_is_retired_without_repairing() {
        const REPLICA: &str = "yda-recon-vanished";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x66);

        ext.store_values(&[(&hash_of(0x66), b"recon-payload".as_slice())])
            .await
            .expect("the pre-flight store must succeed");
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;

        one_pass(&tv, &metrics, REPLICA).await;
        assert_eq!(candidates(&tv).await, vec![format!("{hex}:1")]);
        age_candidates(&tv).await;

        let _: i64 = tv
            .client()
            .pool
            .del(tv.key_schema().ext_meta(&hex))
            .await
            .expect("removing the meta must succeed");

        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED), 0);
        assert!(pending(&tv).await.is_empty());
        assert!(
            candidates(&tv).await.is_empty(),
            "the member must be RETIRED. Left registered it is re-claimed every pass forever, \
             against a meta that will never come back to change the answer — and it accumulates \
             on a `noeviction` store"
        );
    }

    /// Members neither arm of the pass writes — a zero amount and an
    /// unparseable string — are retired rather than carried or acted on.
    ///
    /// Only a hand-edited key or a build writing a different spelling produces
    /// them. The zero amount is the dangerous one: carried into the repair path
    /// it would emit no members and still report a repaired hash, so an operator
    /// would read a repair rate against a pass that repaired nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unrecognised_candidate_members_are_retired_and_repair_nothing() {
        const REPLICA: &str = "yda-recon-junk";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let hex = hex_of(0x77);
        let recon_key = tv.key_schema().ext_recon_candidates();

        for member in [format!("{hex}:0"), "not-a-member".to_owned()] {
            let _: i64 = tv
                .client()
                .pool
                .zadd(&recon_key, None, None, false, false, (0.0, member.as_str()))
                .await
                .expect("seeding a junk candidate must succeed");
        }

        one_pass(&tv, &metrics, REPLICA).await;

        assert!(
            candidates(&tv).await.is_empty(),
            "both must be retired: neither arm of the pass writes a zero amount or a member \
             without a parseable one, so re-claiming them every pass costs a round trip forever"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED),
            0,
            "and a zero amount in particular must NOT reach the repair path, where it would emit \
             no members and still report a repaired hash"
        );
        assert!(pending(&tv).await.is_empty());
    }

    // ── 5. the deficit arm ──────────────────────────────────────────────────

    /// A deficit that reproduces at the same amount a grace later is CONFIRMED
    /// and reported — and never repaired.
    ///
    /// `referenced + owed` above `ref_count` says an over-release already
    /// happened. The pass reports and does nothing else: incrementing would
    /// resurrect a meta whose data the GC may already have deleted, producing a
    /// `ref_count` that claims a blob which is gone.
    ///
    /// The confirmation is not ceremony. The pass's phase order buys its safety
    /// by DOUBLE-COUNTING a mid-pass transition — an obligation that moves from
    /// a stage already walked into one not yet walked is counted in both — and a
    /// double-count is exactly a spurious negative surplus, on a perfectly
    /// healthy store. A stale retire carrier over a live object is a second
    /// source of the same shape. So a single pass's deficit is registered, not
    /// reported, and only the second observation touches the alertable gauge.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_deficit_that_survives_a_grace_is_confirmed_and_never_repaired() {
        const REPLICA: &str = "yda-recon-deficit";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x88);

        // One increment, TWO live objects referencing the hash: `ref_count`
        // reads one below reality.
        ext.store_values(&[(&hash_of(0x88), b"recon-payload".as_slice())])
            .await
            .expect("the pre-flight store must succeed");
        seed_object_referencing(&tv, "940010", &hex).await;
        seed_object_referencing(&tv, "940011", &hex).await;

        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            candidates(&tv).await,
            vec![format!("{hex}:-1")],
            "the direction rides in the SIGN of the amount, so both arms share one ZSET, one \
             grace and one claim, and the member stays unambiguous"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_DEFICIT_OBSERVED),
            0,
            "phase 4 must NOT tick the alertable cell. A first negative surplus is manufactured \
             routinely by the pass's own phase order, so reporting one would page an operator for \
             a healthy store"
        );
        assert_eq!(
            gauge(metrics.as_ref(), "live_edit_ext_ref_deficit_total", REPLICA),
            Some(0.0),
            "and the gauge carries only what phase 5 CONFIRMED, which on this pass is nothing"
        );

        age_candidates(&tv).await;
        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_DEFICIT_OBSERVED),
            1,
            "the same amount, a full grace apart, is the one reading this pass treats as evidence \
             of an over-release: neither benign source can hold an amount still for that long"
        );
        assert_eq!(
            gauge(metrics.as_ref(), "live_edit_ext_ref_deficit_total", REPLICA),
            Some(1.0),
            "and the magnitude reaches the alertable gauge, whose documented recipe is a deficit \
             that RECURS over several intervals rather than one report"
        );
        assert!(
            pending(&tv).await.is_empty(),
            "the deficit arm is confirm-only: there is no code path from it to \
             `enqueue_ext_releases` and none to an `HINCRBY`"
        );
        assert_eq!(
            ref_count(&tv, &hex).await,
            Some(1),
            "and `ref_count` is untouched. Incrementing to 'repair' it would resurrect a meta \
             whose blob the GC may already have collected — a count claiming data that is gone"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED),
            0,
            "a deficit is never a repair, in either direction"
        );
    }

    /// A deficit that does NOT reproduce is retired silently, on its own cell.
    ///
    /// The other half of the confirm protocol, and the common case: the
    /// double-count that produced the first observation resolved, so the second
    /// reading disagrees. It must not touch the alertable gauge, and it must be
    /// distinguishable on the counter from a surplus that healed —
    /// `deficit_cleared` and `healed` are separate cells because a deficit
    /// clearing is the pass's own bookkeeping resolving, while a surplus healing
    /// means a repair was withheld.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_deficit_that_does_not_reproduce_is_cleared_without_alerting() {
        const REPLICA: &str = "yda-recon-deficit-cleared";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x99);

        ext.store_values(&[(&hash_of(0x99), b"recon-payload".as_slice())])
            .await
            .expect("the pre-flight store must succeed");
        seed_object_referencing(&tv, "940020", &hex).await;
        seed_object_referencing(&tv, "940021", &hex).await;

        one_pass(&tv, &metrics, REPLICA).await;
        assert_eq!(candidates(&tv).await, vec![format!("{hex}:-1")]);
        age_candidates(&tv).await;

        // The transient resolves: the second object is gone, and the count was
        // right all along.
        unseed_object(&tv, "940021").await;

        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_DEFICIT_CLEARED),
            1,
            "reported on its own cell rather than as `healed`: a deficit clearing is the pass's \
             own double-count resolving, while a surplus healing means a repair was withheld"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_DEFICIT_OBSERVED),
            0,
            "and the alertable cell must stay still — this is precisely the reading the confirm \
             protocol exists to suppress"
        );
        assert_eq!(
            gauge(metrics.as_ref(), "live_edit_ext_ref_deficit_total", REPLICA),
            Some(0.0)
        );
        assert!(
            candidates(&tv).await.is_empty(),
            "the member is retired, and the surplus is now zero so phase 4 registers nothing in \
             its place"
        );
    }

    // ── 6. fail closed ──────────────────────────────────────────────────────

    /// One unreadable object forfeits the WHOLE pass: no gauges, no candidates,
    /// one `enumeration_failed`.
    ///
    /// The orphan sweep's `continue`-on-error posture is deliberately NOT copied
    /// here, and the asymmetry is the point: a partial walk there defers work,
    /// while a partial walk here UNDER-counts `referenced`, and an under-counted
    /// `referenced` reads as surplus on every hash the skipped objects held.
    /// That is the over-release this pass exists to refuse. Deferring every
    /// repair for one tick costs nothing — the ZSET is durable and the surplus
    /// is not going anywhere.
    ///
    /// The `None` gauges are the sharpest half. A forfeited pass has no honest
    /// reading, and its surplus figure in particular is exactly the number that
    /// must never be believed; publishing a "partial" one would put a fabricated
    /// level on a gauge an operator reads as a fleet-wide `max()`.
    ///
    /// The wrong-typed `obj:flds` is a real `WRONGTYPE` from a real Valkey — a
    /// standing condition, not an injected one — so the classification under
    /// test is the one production runs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unreadable_object_forfeits_the_pass_with_no_gauges_and_no_candidates() {
        const REPLICA: &str = "yda-recon-fail-closed";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0xAA);

        // A perfectly repairable surplus, so the forfeit is the only reason
        // nothing happens.
        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0xAA), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;

        seed_object_referencing(&tv, "940030", &hex).await;
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().obj_fields_index("940030"),
                "planted-by-an-out-of-band-writer",
                None,
                None,
                false,
            )
            .await
            .expect("planting a string over obj:flds must succeed");

        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
            1,
            "the forfeit must be COUNTED — a pass that silently repairs nothing every hour is \
             indistinguishable from a healthy one"
        );
        assert_eq!(
            gauges(&metrics, REPLICA),
            None,
            "and NOTHING may be published. A truncated object walk under-counts `referenced`, \
             which reads as surplus, so the one number an operator must not be shown is exactly \
             the one a partial pass would compute"
        );
        assert!(
            candidates(&tv).await.is_empty(),
            "nor may a first observation be pinned from a truncated walk: it would sit on the \
             ZSET at an amount that was never real, waiting for a future pass to confirm it"
        );
        assert!(pending(&tv).await.is_empty());
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED),
            0
        );
    }

    /// The retire-carrier walk reads a carrier by its type (b.1jt): a HASH
    /// (this build) or a SET (an older build). A key at a carrier's address of
    /// any other type is one the walk cannot read, and an unread carrier
    /// under-counts `owed` — the over-release direction — so it forfeits the
    /// pass (`PassError::Undecodable`) rather than skipping the key.
    ///
    /// A surplus is registered on a clean pass and aged, so the next pass would
    /// repair it; a STRING is then planted at `le:{oid}:obj:extret`. That pass
    /// forfeits — counted — and repairs nothing: nothing reaches `pending`,
    /// `ref_count` is untouched, and the candidate stays registered. The
    /// control removes the string, and the same candidate is repaired, so the
    /// forfeit was the only reason nothing happened.
    ///
    /// The walk's other arms are pinned by
    /// [`a_deferred_retirement_release_on_a_carrier_is_not_counted_as_surplus`]
    /// (HASH) and
    /// [`a_release_on_an_older_build_set_carrier_is_not_counted_as_surplus`]
    /// (SET).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_carrier_of_another_type_forfeits_the_pass_and_repairs_nothing() {
        const REPLICA: &str = "b1jt-recon-carrier-type";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0xAB);
        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0xAB), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
        seed_object_referencing(&tv, "940031", &hex).await;

        one_pass(&tv, &metrics, REPLICA).await;
        assert_eq!(
            candidates(&tv).await,
            vec![format!("{hex}:1")],
            "precondition: the surplus is registered"
        );
        age_candidates(&tv).await;

        let carrier = tv.key_schema().obj_ext_retire_carrier("940032");
        let _: () = tv
            .client()
            .pool
            .set(
                &carrier,
                "planted-by-an-out-of-band-writer",
                None,
                None,
                false,
            )
            .await
            .expect("planting a string at a carrier's address must succeed");

        let mut owed = HashMap::new();
        let walked = walk_retire_carriers(
            tv.client(),
            tv.key_schema(),
            &mut owed,
            TEST_INTERVAL,
            Instant::now(),
            &CancellationToken::new(),
        )
        .await;
        assert!(
            matches!(&walked, Err(PassError::Undecodable { text, .. }) if text.contains("string")),
            "the walk refuses a carrier it cannot read: {:?}",
            walked.as_ref().err().map(ToString::to_string)
        );

        one_pass(&tv, &metrics, REPLICA).await;
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
            1,
            "the pass forfeits, and the forfeit is counted"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED),
            0,
            "and it repairs nothing"
        );
        assert!(pending(&tv).await.is_empty(), "nothing reached pending");
        assert_eq!(ref_count(&tv, &hex).await, Some(2), "ref_count untouched");
        assert_eq!(
            candidates(&tv).await,
            vec![format!("{hex}:1")],
            "the candidate is left for a pass that can read every carrier"
        );

        let _: i64 = tv
            .client()
            .pool
            .del(&carrier)
            .await
            .expect("removing the plant must succeed");
        one_pass(&tv, &metrics, REPLICA).await;
        assert_eq!(
            pending(&tv).await,
            vec![format!("{hex}:0:rec1")],
            "control: with the string gone, the same candidate is repaired"
        );
    }

    // ── b.gns — an object or owed walk that crosses a change of serving process ─
    //
    // The pass's object walk (`storage::scan_guard::KeyspaceWalk`) and its owed
    // walks (the `pending` SSCAN and the `inflight` ZSCAN, each a
    // `scan_guard::KeyedWalk` driven by the cursor pager) read a witness, the
    // serving process's run_id, at their start (the object walk alone, the
    // owed walks in their first step's pipeline) and with every step (in one
    // pipeline with it), and end with `PassError::ServingChanged` when it
    // changes: a
    // cursor continued on another server process can skip members. These
    // arms forge that change through the walk seam
    // (`scan_guard::arm_walk_fault_for_test`, keyed by the walk's pattern or
    // key, each carrying this test's prefix, so no other test's walk is
    // touched): the walk's first witness read reports one run_id and its
    // second another, so the walk ends after its first step. They pin the
    // forfeit on the two walks whose under-count becomes an over-release: the
    // object walk (`referenced`) and the owed walks (`owed`).
    //
    // Fail-on-old, by in-place perturbation (red, then green once restored):
    // with the guard's comparisons removed (`scan_guard::changed_between`
    // answering no change, and the cursor pager's run_id comparison never
    // true), T-W2 and both T-W3 cases go red on their first
    // `enumeration_failed` assertion (0, not 1): each pass reads the crossed
    // walk as complete.

    use crate::storage::scan_guard::{
        arm_walk_fault_for_test, forge_process_change_for_test, ArmedWalkFault, WalkFault,
        TEST_RUN_ID_A as FORGED_RUN_A, TEST_RUN_ID_B as FORGED_RUN_B,
    };

    /// Forge a change of serving process on every walk of `target` (a key,
    /// or a keyspace walk's pattern) that starts while the guard is held:
    /// its first witness read reports [`FORGED_RUN_A`] and its second
    /// [`FORGED_RUN_B`], so a standalone walk ends after its first step,
    /// whatever the server's own run_id is.
    fn forge_process_change(target: &str) -> ArmedWalkFault {
        forge_process_change_for_test(target, 1)
    }

    /// The one call these arms make to the GC, under
    /// [`GcPolicy::IMMEDIATE`](crate::storage::external_values::GcPolicy::IMMEDIATE),
    /// so whatever is owed is claimed in this pass, with no age floor.
    async fn run_gc(ext: &ValkeyExternalValueStore) -> crate::storage::external_values::GcStats {
        process_pending_gc(
            ext,
            None,
            crate::storage::external_values::GcPolicy::IMMEDIATE,
        )
        .await
    }

    /// T-W2 (rider 7, b.gns): an object walk that crosses a change of serving
    /// process, on both passes a grace apart, forfeits both — so a hash a live
    /// object references, at its true count, is never repaired and its meta
    /// and data survive the GC.
    ///
    /// The live data-loss path this closes: a walk whose cursor was continued
    /// on another process can skip objects, which under-counts `referenced`
    /// and over-states the surplus; two such passes that agree on the amount
    /// would pass the confirm, enqueue a release, and the GC would delete a
    /// meta and data the live object still references. The change is forged
    /// after the walk's first step on both passes; the control pass at the
    /// end, with the seam disarmed, sees no surplus at all.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_object_walk_that_crosses_a_process_change_forfeits_both_passes_and_the_gc_keeps_the_blob(
    ) {
        const REPLICA: &str = "gns-recon-object-walk-crossed";
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0x5A);

        // One increment, one live object: the count is exactly right.
        ext.store_values(&[(&hash_of(0x5A), b"recon-payload".as_slice())])
            .await
            .expect("the pre-flight store must succeed");
        seed_object_referencing(&tv, "940501", &hex).await;
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
        assert_eq!(ref_count(&tv, &hex).await, Some(1));

        // The guards for the object walk, held for both passes and dropped
        // before the control pass.
        let object_walk = forge_process_change(&keys.obj_header_scan_pattern());

        one_pass(&tv, &metrics, REPLICA).await;
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
            1,
            "pass 1 forfeits on the walk that crossed a change of serving process"
        );
        assert!(
            candidates(&tv).await.is_empty(),
            "no first observation from a walk whose result cannot be used"
        );
        assert_eq!(gauges(&metrics, REPLICA), None, "and nothing is published");

        age_candidates(&tv).await;
        one_pass(&tv, &metrics, REPLICA).await;
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
            2,
            "pass 2 forfeits too"
        );
        assert!(candidates(&tv).await.is_empty());
        assert!(
            pending(&tv).await.is_empty(),
            "no repair reaches the release channel"
        );
        assert_eq!(gauges(&metrics, REPLICA), None);
        assert_eq!(outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED), 0);

        let stats = run_gc(&ext).await;
        assert_eq!(stats.processed, 0, "the GC has nothing owed to it");
        assert_eq!(ref_count(&tv, &hex).await, Some(1));
        let exists = |key: String| {
            let pool = tv.client().pool.clone();
            async move {
                pool.exists::<bool, _>(key)
                    .await
                    .expect("EXISTS must succeed")
            }
        };
        assert!(
            exists(keys.ext_meta(&hex)).await && exists(keys.ext_data(&hex)).await,
            "the meta and the data a live object references both survive"
        );

        // Control: with the seam disarmed, the same keyspace has no surplus.
        drop(object_walk);
        one_pass(&tv, &metrics, REPLICA).await;
        assert_eq!(
            gauges(&metrics, REPLICA),
            Some((1.0, 1.0, 0.0, 0.0)),
            "control: one meta, one referenced hash, no surplus, no deficit"
        );
        assert!(candidates(&tv).await.is_empty());
    }

    /// T-W3 (b.gns), and PM 4 (b.f3t): an owed walk the pass cannot show is
    /// complete — the `pending` `SSCAN` or the `inflight` `ZSCAN`, whichever
    /// holds the owed release, under the walk seam's fault `arm` armed on
    /// it (`fault` names it, for the replica and the messages) — forfeits
    /// the pass rather than tally an `owed` it cannot show is complete. The
    /// fixture's true surplus is zero: `ref_count` 2, one live object, and
    /// the other reference's release already owed from `stage`. A walk whose
    /// cursor was continued on another process can skip that release, which
    /// would register a surplus of one. Every event the pass logged
    /// ([`capture_events`]), for a caller that pins its forfeit line.
    async fn an_owed_walk_under_a_fault_forfeits_the_pass_and_registers_nothing(
        stage: OwedStage,
        fault: &str,
        arm: fn(&str) -> ArmedWalkFault,
    ) -> Vec<serde_json::Value> {
        let replica = format!("gns-recon-owed-walk-{fault}-{}", stage.label());
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let metrics = Arc::new(metrics_scrape_only(&replica));
        let ext = store(&tv);
        let hex = hex_of(0x5B);

        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0x5B), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        seed_object_referencing(&tv, "940502", &hex).await;
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
        let owed = ExtRelease::retirement(hash_of(0x5B), 940_503).member();
        let walk = match stage {
            OwedStage::Pending => {
                let _: u64 = tv
                    .client()
                    .pool
                    .sadd(keys.ext_pending(), owed.as_str())
                    .await
                    .expect("seeding pending must succeed");
                keys.ext_pending()
            }
            OwedStage::Inflight => {
                let _: i64 = tv
                    .client()
                    .pool
                    .zadd(
                        keys.ext_inflight(),
                        None,
                        None,
                        false,
                        false,
                        (1.0, owed.as_str()),
                    )
                    .await
                    .expect("seeding inflight must succeed");
                keys.ext_inflight()
            }
            OwedStage::InflightEncoded
            | OwedStage::RetireCarrier
            | OwedStage::LegacyRetireCarrier => {
                unreachable!("only the pending and the bare-scored inflight arms are driven here")
            }
        };
        assert_eq!(ref_count(&tv, &hex).await, Some(2));

        // The guards for this walk, held for the whole pass.
        let armed = arm(&walk);
        let events = capture_events(one_pass(&tv, &metrics, &replica)).await;
        drop(armed);

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
            1,
            "the pass forfeits on the {} walk under {fault}",
            stage.label()
        );
        assert!(
            candidates(&tv).await.is_empty(),
            "no surplus is registered from an `owed` the {} walk cannot show is complete",
            stage.label()
        );
        assert_eq!(gauges(&metrics, &replica), None);
        events
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_pending_walk_that_crosses_a_process_change_forfeits_the_pass_and_registers_nothing()
    {
        an_owed_walk_under_a_fault_forfeits_the_pass_and_registers_nothing(
            OwedStage::Pending,
            "crossed",
            forge_process_change,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_inflight_walk_that_crosses_a_process_change_forfeits_the_pass_and_registers_nothing(
    ) {
        an_owed_walk_under_a_fault_forfeits_the_pass_and_registers_nothing(
            OwedStage::Inflight,
            "crossed",
            forge_process_change,
        )
        .await;
    }

    /// A writer appending every formatted log line to a shared buffer.
    #[derive(Clone)]
    struct LogCapture(Arc<std::sync::Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogCapture {
        type Writer = LogCapture;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Drive `future` under a capturing JSON subscriber at INFO and return
    /// every event it logged, parsed.
    ///
    /// `WithSubscriber` rather than `set_default`: the pass awaits Valkey on
    /// a multi-thread runtime, and a thread-local dispatcher would stop
    /// capturing at the first `.await` that resumes on another worker. A
    /// second, recording-nothing dispatcher stays registered for the
    /// process, so a callsite another test thread hits first is not cached
    /// as `Interest::never` while this capture is live (the same reasoning
    /// as `storage::layer`'s `capture_info_events`).
    async fn capture_events<F>(future: F) -> Vec<serde_json::Value>
    where
        F: std::future::Future<Output = ()>,
    {
        use tracing::instrument::WithSubscriber;

        static SECOND: std::sync::OnceLock<tracing::Dispatch> = std::sync::OnceLock::new();
        SECOND.get_or_init(|| tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default()));

        let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(LogCapture(Arc::clone(&buf)))
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .finish();
        future
            .with_subscriber(tracing::Dispatch::new(subscriber))
            .await;
        let text =
            String::from_utf8_lossy(&buf.lock().unwrap_or_else(|p| p.into_inner())).into_owned();
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("a captured line is JSON ({e}): {line}"))
            })
            .collect()
    }

    /// PM 4 (b.f3t; policy 5, kept): the periodic pass still reads the
    /// serving-process witness on its owed walks, and forfeits on a store
    /// whose `INFO server` reports no usable run_id. On the owed-walk
    /// fixture with the release owed in `pending`, with `pending`'s walk
    /// forged to read a placeholder run_id (`"0"`), or no `run_id` line, at
    /// its first witness read (the walk seam's `RunId`), the pass forfeits
    /// (`enumeration_failed` ticks once, no gauge is published, nothing is
    /// registered: the shared helper's assertions) and logs one WARN, the
    /// forfeit, whose error names `INFO server` and what it read.
    ///
    /// Fail-on-old (in-place perturbation, red then green once restored):
    /// with `scan_guard::parse_run_id` reading every reply as a usable
    /// run_id, the pass completes and `enumeration_failed` stays 0.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_pending_walk_whose_store_reports_no_usable_run_id_forfeits_the_pass_and_registers_nothing(
    ) {
        type Arm = fn(&str) -> ArmedWalkFault;
        let cases: [(&str, &str, Arm); 2] = [
            ("run_id \"0\"", "no-run-id-0", |walk| {
                arm_walk_fault_for_test(
                    walk,
                    WalkFault::RunId {
                        read: 0,
                        run_id: Some("0".to_owned()),
                    },
                )
            }),
            ("no run_id line", "no-run-id-line", |walk| {
                arm_walk_fault_for_test(
                    walk,
                    WalkFault::RunId {
                        read: 0,
                        run_id: None,
                    },
                )
            }),
        ];
        for (seen, fault, arm) in cases {
            let events = an_owed_walk_under_a_fault_forfeits_the_pass_and_registers_nothing(
                OwedStage::Pending,
                fault,
                arm,
            )
            .await;
            let warns: Vec<&serde_json::Value> =
                events.iter().filter(|e| e["level"] == "WARN").collect();
            assert_eq!(warns.len(), 1, "{seen}: one WARN: {warns:?}");
            assert_eq!(
                warns[0]["fields"]["message"],
                "external-value ref_count reconcile pass forfeited; no repair was made",
                "{seen}"
            );
            let error = warns[0]["fields"]["error"].as_str().unwrap_or_default();
            for part in ["`INFO server`", seen, "SSCAN walk of"] {
                assert!(
                    error.contains(part),
                    "{seen}: the WARN names {part:?}: {error}"
                );
            }
        }
    }

    /// A pass that runs out of BUDGET forfeits exactly as an unreadable object
    /// does — no gauges, no candidates, one `enumeration_failed`.
    ///
    /// The third of the three ways `run_pass` can end early, and the only one
    /// with no seam and no fixture corruption behind it: the budget is
    /// `interval / 2`, minted from the caller's own argument, so a near-zero
    /// interval makes the elapse deterministic rather than load-dependent. The
    /// one live object is what guarantees the walk reaches a budget check at all
    /// — the check sits after the `SCAN` yields a key, so an empty keyspace would
    /// complete the pass and this arm would assert the healthy path by accident.
    ///
    /// It shares `PassError`'s error arm with the Valkey and decode failures, and
    /// is deliberately NOT grouped with `Cancelled`: an elapsed budget means this
    /// replica cannot walk the keyspace inside a tick, which is an operational
    /// condition worth alerting on (the keyspace outgrew the interval, or the
    /// store is slow), while a cancellation is a rolling restart. A build that
    /// moved `BudgetElapsed` onto the quiet arm would leave a fleet whose
    /// reconciler had stopped repairing anything looking exactly like a healthy
    /// one.
    ///
    /// The truncation hazard is identical to the unreadable-object arm's: a walk
    /// cut short under-counts `referenced`, and an under-counted `referenced`
    /// reads as surplus on every hash the objects it never reached hold. Hence
    /// the same three assertions, and `None` gauges in particular.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_pass_that_exhausts_its_budget_forfeits_with_no_gauges_and_no_candidates() {
        const REPLICA: &str = "yda-recon-budget";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0xCC);

        // A perfectly repairable surplus, so the forfeit is the only reason
        // nothing happens — same fixture as the unreadable-object arm, minus the
        // corruption.
        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0xCC), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
        seed_object_referencing(&tv, "940040", &hex).await;

        // `interval / 2` is the budget, and phase 1's `TIME` round trip has
        // already outrun a nanosecond of it by the time the object walk reaches
        // its first check.
        run_one_pass(
            tv.client(),
            tv.key_schema(),
            REPLICA,
            Duration::from_nanos(1),
            TEST_GRACE,
            &CancellationToken::new(),
            Some(&(metrics.clone() as SharedMetrics)),
        )
        .await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
            1,
            "an elapsed budget is a FAILURE, not a quiet skip: it says this replica cannot walk \
             the keyspace inside one tick, so the reconciler has silently stopped repairing \
             anything. Grouping it with `Cancelled` would make that indistinguishable from health"
        );
        assert_eq!(
            gauges(&metrics, REPLICA),
            None,
            "and a truncated walk publishes NOTHING — it under-counts `referenced`, which reads as \
             surplus on every hash the objects it never reached hold, so its surplus figure is \
             precisely the number an operator must not be shown"
        );
        assert!(
            candidates(&tv).await.is_empty(),
            "nor may a first observation be pinned from it: the amount would be one that was never \
             real, waiting for a later pass to confirm it"
        );
        assert!(pending(&tv).await.is_empty());
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED),
            0
        );
        assert_eq!(
            ref_count(&tv, &hex).await,
            Some(2),
            "and the keyspace is untouched — the forfeit happens in phase 2, long before anything \
             mutates"
        );
    }

    /// Cancellation forfeits the pass exactly as an error does — and does NOT
    /// tick `enumeration_failed`.
    ///
    /// Both halves are load-bearing. A walk cut short at shutdown produces the
    /// same truncated map an errored walk does, so it must forfeit; but
    /// `enumeration_failed` is alertable, and a counter that moves on every
    /// rolling restart is a counter nobody can alert on. The two arms of
    /// `run_one_pass`'s `match` are what separate them, and this is the only
    /// test that can tell them apart.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancelled_pass_forfeits_without_ticking_the_alertable_failure_cell() {
        const REPLICA: &str = "yda-recon-cancelled";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0xBB);

        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0xBB), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;

        let cancel = CancellationToken::new();
        cancel.cancel();
        run_one_pass(
            tv.client(),
            tv.key_schema(),
            REPLICA,
            TEST_INTERVAL,
            TEST_GRACE,
            &cancel,
            Some(&(metrics.clone() as SharedMetrics)),
        )
        .await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
            0,
            "a shutdown is an expected event, and this cell is alertable. A tick per rolling \
             restart makes `increase(...) > 0` useless as an alert"
        );
        assert_eq!(
            gauges(&metrics, REPLICA),
            None,
            "but the pass still FORFEITS: a walk cut short under-counts `referenced` exactly as \
             an errored one does, and `fail closed, absolutely` has no shutdown exception"
        );
        assert!(
            candidates(&tv).await.is_empty(),
            "and no first observation is pinned from it"
        );
        assert!(pending(&tv).await.is_empty());
    }

    // ── 6b. batch pressure ──────────────────────────────────────────────────

    /// A claim that FILLS the batch reports `deferred` once, and the over-batch
    /// remainder keeps its first-observation score and drains on the next pass.
    ///
    /// `RECONCILE_BATCH` bounds how many candidates one pass acts on, and the
    /// `deferred` cell is the only signal an operator has that a pass ran to the
    /// bound — that a backlog exists, and that repairs are being spread over
    /// several intervals rather than performed. It ticks on
    /// `claimed.len() >= RECONCILE_BATCH`, an inference from a full page rather
    /// than a measured backlog, which is exactly the shape a boundary error hides
    /// in: `>` instead of `>=` never ticks at all (a claim cannot exceed its own
    /// `LIMIT`), and per-member instead of per-pass turns a level indicator into
    /// a count of candidates.
    ///
    /// # Why the members are unparseable junk
    ///
    /// The cell is about the CLAIM, not about what acting on a member does, and
    /// an unparseable member is the one candidate shape that reaches a terminal
    /// outcome with no `ext:meta` behind it — so 300 of them cost 300 `ZREM`s and
    /// nothing else, and the arm stays about batch pressure. It is also a real
    /// shape rather than a contrived one: `unrecognised_candidate_members_are_
    /// retired_and_repair_nothing` covers the same members at a size below the
    /// batch.
    ///
    /// # The remainder's SCORE is the load-bearing half
    ///
    /// A deferred candidate must come back at its ORIGINAL first-observation
    /// time, or the two-observation protocol resets every time a pass is busy:
    /// a backlog would then push every candidate's grace out indefinitely and the
    /// pass would repair nothing while reporting `deferred` forever. The score is
    /// preserved because `recon_claim_expired` registers with `ZADD NX` and
    /// removal is the caller's own `ZREM` — the members this pass never claimed
    /// are never touched at all.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_claim_that_fills_the_batch_defers_the_remainder_to_the_next_pass() {
        const REPLICA: &str = "yda-recon-deferred";
        /// Comfortably over one batch, and the excess is small enough that the
        /// second pass cannot fill a batch of its own.
        const OVER_BATCH: usize = 44;

        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let recon_key = tv.key_schema().ext_recon_candidates();

        // Zero-padded so the lexicographic order among equal scores — which is
        // what `ZRANGEBYSCORE ... LIMIT` walks — is the numeric one, making the
        // split between claimed and deferred deterministic rather than
        // whatever `%d` happens to sort like.
        let seeded: Vec<String> = (0..RECONCILE_BATCH + OVER_BATCH)
            .map(|i| format!("yda-unparseable-candidate-{i:04}"))
            .collect();
        for member in &seeded {
            // Score 0: aged past any grace, so every one of them is claimable
            // from the first pass.
            let _: i64 = tv
                .client()
                .pool
                .zadd(&recon_key, None, None, false, false, (0.0, member.as_str()))
                .await
                .expect("seeding a candidate must succeed");
        }
        assert_eq!(
            candidates(&tv).await.len(),
            RECONCILE_BATCH + OVER_BATCH,
            "fixture precondition: strictly more candidates than one batch, or this arm asserts \
             the un-deferred path"
        );

        // ── Pass 1: the claim fills the batch ───────────────────────────────
        one_pass(&tv, &metrics, REPLICA).await;

        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_DEFERRED),
            1,
            "ONE tick, for the pass — not one per candidate over the bound, and not zero. This is \
             the only place a backlog becomes visible: every other cell reports work that was \
             done, so a pass silently running to its bound every interval looks like a pass with \
             nothing to do"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_HEALED),
            RECONCILE_BATCH as u64,
            "and exactly one batch was acted on — the bound is a bound on ACTION, so a build that \
             claimed a page and then walked the whole ZSET anyway would read higher here while \
             still ticking `deferred`"
        );

        let left = candidates(&tv).await;
        assert_eq!(
            left,
            seeded[RECONCILE_BATCH..].to_vec(),
            "the untouched remainder is exactly the tail the claim's `LIMIT` did not reach, in \
             score-then-lexicographic order"
        );
        for member in &left {
            let score: Option<f64> = tv
                .client()
                .pool
                .zscore(&recon_key, member.as_str())
                .await
                .expect("ZSCORE must succeed");
            assert_eq!(
                score,
                Some(0.0),
                "and it keeps its FIRST-observation score. A pass that re-scored what it deferred \
                 would restart the grace on every busy interval, so a backlog would defer its own \
                 candidates forever and nothing would ever be repaired: {member}"
            );
        }

        // ── Pass 2: the remainder drains, and nothing is deferred ───────────
        one_pass(&tv, &metrics, REPLICA).await;

        assert!(
            candidates(&tv).await.is_empty(),
            "the deferred remainder is claimable immediately — its score never moved — so the very \
             next pass finishes it"
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_HEALED),
            (RECONCILE_BATCH + OVER_BATCH) as u64
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_DEFERRED),
            1,
            "and `deferred` does NOT tick again: this claim came back short of the bound, which is \
             what an operator reads as the backlog having cleared"
        );
        assert!(pending(&tv).await.is_empty(), "nothing was ever repairable");
    }

    // ── 7. leader gating ────────────────────────────────────────────────────

    /// A replica that does not hold `le:gc:leader` runs no pass, repairs
    /// nothing, and zero-publishes all four gauges under its OWN label.
    ///
    /// The zero-publish is the non-obvious half. An OTel `Gauge` never decays,
    /// so a replica that held the lease, saw a surplus and then lost it would go
    /// on exporting that surplus forever and pin the fleet-wide `max()` at a
    /// level that no longer exists. Publishing zeros under its own `service`
    /// label is what keeps the fleet aggregate honest across a failover.
    ///
    /// Driven through `spawn_ext_ref_reconcile` rather than `run_one_pass`,
    /// because the lease check lives in the spawned loop and is exactly what is
    /// under test. The gauges are polled in a bounded loop rather than slept
    /// for; the healthy path reaches the publish in milliseconds.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_non_leader_replica_publishes_zeros_and_repairs_nothing() {
        const REPLICA: &str = "yda-recon-follower";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(REPLICA));
        let ext = store(&tv);
        let hex = hex_of(0xCC);

        // A surplus a leader WOULD register, so "nothing happened" is a
        // consequence of the lease rather than of an empty keyspace.
        for _ in 0..2 {
            ext.store_values(&[(&hash_of(0xCC), b"recon-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }
        set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;

        // The peer takes the lock as a marked build of this protocol does
        // (b.1jt): a same-version holder with a fresh TTL is one this
        // replica's takeover script leaves alone. An unmarked value would be
        // an older build's, which the follower takes over at once.
        let held = acquire_gc_leader_lock(
            tv.client(),
            &tv.key_schema().gc_leader_lock(),
            &LeaderIdentity::new("some-other-replica", "peer-instance"),
            LEADER_LOCK_TTL_MS,
            GC_LEADER_TAKEOVER_BELOW_MS,
        )
        .await
        .expect("seizing the leader lock must succeed");
        assert!(
            held.is_leader(),
            "fixture precondition: the peer must really hold the lease: {held:?}"
        );

        let cancel = CancellationToken::new();
        let handle = spawn_ext_ref_reconcile(
            tv.client().clone(),
            tv.key_schema().clone(),
            LeaderIdentity::new(REPLICA, "follower-instance"),
            TEST_INTERVAL,
            TEST_GRACE,
            cancel.clone(),
            Some(metrics.clone() as SharedMetrics),
        );

        let mut published = None;
        for _ in 0..200 {
            published = gauges(&metrics, REPLICA);
            if published.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        cancel.cancel();
        handle.await.expect("the reconciler task must exit cleanly");

        assert_eq!(
            published,
            Some((0.0, 0.0, 0.0, 0.0)),
            "a non-leader must zero-publish all four under its own `service` label. An OTel gauge \
             never decays, so a demoted leader that simply stopped publishing would pin the \
             fleet-wide max() at a surplus that no longer exists"
        );
        assert!(
            candidates(&tv).await.is_empty(),
            "and it must not enumerate: the surplus is right there, and only the lease holder may \
             act on it"
        );
        assert!(pending(&tv).await.is_empty());
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED),
            0
        );
        assert_eq!(
            outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
            0
        );
    }

    /// A pass command that gets no reply within the command timeout reaches
    /// `valkey_err`, which counts exactly one
    /// `valkey_errors_total{pool="maintenance", kind="command_timeout"}` and
    /// forfeits with `PassError::Valkey`. The `TIME` read is held behind a
    /// server-side `BLPOP` on the maintenance connection, so the `Timeout` is
    /// fred's real one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_timed_out_pass_command_counts_one_maintenance_command_timeout() {
        use crate::storage::test_support::{
            command_timeout_row, command_timeouts_only_on, release_all,
        };
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("ext-reconcile-timeout"));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;

        let stalls = tv.stall_maintenance_pool().await;
        let read = valkey_time_ms(tv.client()).await;
        release_all(stalls).await;

        assert!(
            matches!(
                &read,
                Err(PassError::Valkey { source, .. })
                    if *source.kind() == fred::error::ErrorKind::Timeout
            ),
            "the stalled TIME must forfeit with fred's Timeout; got {read:?}"
        );
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("maintenance", 1),
            "the timed-out TIME must count once, on the maintenance pool"
        );
    }

    // ══ The per-object index audit and the drain gate ═══════════════════════
    //
    // The pass audits each live object's `obj:ext` against its rows (phase 2)
    // and holds any surplus on a hash whose object it could not repair (phases
    // 4 and 5). Two fixture styles, each where it is the honest one:
    //
    // * The drift the audit exists for is produced by the REAL commit paths:
    //   a layer committed while its index is absent (the shape a pre-index
    //   replica left behind) seeds an under-count with the next commit that
    //   changes a hash's row count. [`seed_legacy_under_count`] drives exactly
    //   that through `LayerStore`, on both commit backends, so the arms that
    //   end in a GC tick are asserting about production state.
    // * The direction and liveness arms need index shapes no single commit
    //   produces side by side (an over-count, a missing field, a dual-shape
    //   row), so they are staged by hand with [`seed_row`].
    mod index_audit {
        use super::*;

        use crate::storage::test_support::relay::RelayedUpdates;

        use crate::config::UpdateLayerBackend;
        use crate::domain::types::ObjectState;
        use crate::observability::metrics::{
            EXT_INDEX_DRIFT_DIRECTION_OVER, EXT_INDEX_DRIFT_DIRECTION_UNDER,
            EXT_INDEX_RECOUNT_OUTCOME_CLEAN, EXT_INDEX_RECOUNT_OUTCOME_FAILED,
            EXT_INDEX_RECOUNT_OUTCOME_NOT_LIVE, EXT_INDEX_RECOUNT_OUTCOME_REPAIRED,
            EXT_INDEX_RECOUNT_OUTCOME_TOO_LARGE, EXT_REF_RELEASE_REASONS,
        };
        use crate::storage::ext_index_audit::HEADER_FIELD_EXT_INDEX_REPAIRED_MS;
        use crate::storage::layer::LayerStore;
        use crate::storage::session::SessionStore;
        use oodf_ffi::{OwnedDelta, SetField};

        /// The section every committed row hangs off.
        const SECTION: u64 = 1;
        /// The two committed field rows: R1 and R2 of the scenario.
        const ROW_ONE: &str = "color";
        const ROW_TWO: &str = "roughness";
        /// Declared size of every committed external value; the rows only
        /// need to decode.
        const EXT_SIZE: u64 = 1_024;
        /// H, the hash both committed rows end up holding.
        const H: u8 = 0x51;
        /// K, the hash R2 is later rewritten to.
        const K: u8 = 0x52;

        /// A one-row delta setting field `key` to the external hash
        /// `hash_byte`. Fast-path eligible, so under `Lua` it runs
        /// `update_layer_fast.lua` and under `Txn` the WATCH path's
        /// non-snapshot branch — the two commit paths that trust `obj:ext`.
        fn external_field(key: &str, hash_byte: u8, set_order: u64) -> OwnedDelta {
            OwnedDelta {
                set_fields: vec![SetField {
                    section_id: SECTION,
                    key_name: key.to_owned(),
                    value_or_ext_hash: hash_of(hash_byte).to_vec(),
                    ext_value_size: EXT_SIZE,
                    ext_value_index: 1,
                    set_order,
                }],
                ..Default::default()
            }
        }

        /// The state `create_layer` materializes for a layer whose content
        /// arrives by `update_layer`.
        fn empty_state() -> ObjectState {
            ObjectState {
                sequence_num: 1,
                first_version: 1,
                source_format: "usda".to_owned(),
                next_set_order: 0,
                ..Default::default()
            }
        }

        /// A layer whose `obj:ext` under-counts H, produced by the real commit
        /// paths.
        struct LegacySeeded {
            ls: LayerStore,
            ext: Arc<ValkeyExternalValueStore>,
            session_id: String,
            layer: String,
            oid: u64,
        }

        /// Commit R1 := H, delete the object's `obj:ext` (the shape a replica
        /// that predates the index left behind), then commit R2 := H.
        ///
        /// The second commit seeds the index from the rows IT touched: H
        /// reads `0` before, so it writes the absolute `0 + 1`, while R1
        /// still holds H — an under-count of one. Its pre-flight `+1` is not
        /// handed back (`before` is 0 and R2 held nothing), so `ref_count`
        /// reads 2 for a hash one live object references: a surplus of one,
        /// which is the increment keeping the blob alive under R1.
        ///
        /// Each pre-flight `store_values` is staged by hand, one per commit,
        /// because the storage layer is driven directly — the gRPC layer pays
        /// it in production.
        async fn seed_legacy_under_count(
            tv: &TestValkey,
            backend: UpdateLayerBackend,
            metrics: &Arc<Metrics>,
            layer: &str,
        ) -> LegacySeeded {
            let ext = Arc::new(store(tv));
            let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
                .create(layer, &format!("/{layer}"), "nvf-user")
                .await
                .expect("session creation must succeed");
            let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                .with_update_layer_backend(backend)
                .with_metrics(metrics.clone());
            let oid = ls
                .create_layer(&session_id, layer, "nvf-user", &empty_state(), ext.clone())
                .await
                .expect("create_layer must succeed");

            pre_store(&ext, H).await;
            ls.update_layer_relayed(
                &session_id,
                layer,
                oid,
                Some(external_field(ROW_ONE, H, 0)),
                "nvf-writer",
                1,
            )
            .await
            .expect("R1 := H must commit");

            let _: u64 = tv
                .client()
                .pool
                .del(tv.key_schema().obj_ext_index(&oid.to_string()))
                .await
                .expect("deleting obj:ext must succeed");

            pre_store(&ext, H).await;
            ls.update_layer_relayed(
                &session_id,
                layer,
                oid,
                Some(external_field(ROW_TWO, H, 1)),
                "nvf-writer",
                2,
            )
            .await
            .expect("R2 := H must commit");

            assert_eq!(
                obj_ext(tv, oid).await,
                HashMap::from([(hex_of(H), "1".to_owned())]),
                "fixture precondition: the seeding commit wrote H's count from the one row it \
                 touched, while TWO rows hold H — the under-count this arm is about. A 2 here \
                 means this backend planned from the rows (a snapshot) and the arm is vacuous"
            );
            assert_eq!(
                ref_count(tv, &hex_of(H)).await,
                Some(2),
                "fixture precondition: two pre-flight increments, one live object"
            );
            LegacySeeded {
                ls,
                ext,
                session_id,
                layer: layer.to_owned(),
                oid,
            }
        }

        /// One pre-flight `store_values` `+1` on `hash_byte`.
        async fn pre_store(ext: &ValkeyExternalValueStore, hash_byte: u8) {
            ext.store_values(&[(&hash_of(hash_byte), b"nvf-payload".as_slice())])
                .await
                .expect("the pre-flight store must succeed");
        }

        /// Commit R2 := K on the seeded layer — the rewrite that, on an
        /// unrepaired under-counted index, takes H's stored count to 0 and
        /// releases H while R1 still holds it.
        async fn rewrite_row_two_to_k(seeded: &LegacySeeded) {
            pre_store(&seeded.ext, K).await;
            seeded
                .ls
                .update_layer_relayed(
                    &seeded.session_id,
                    &seeded.layer,
                    seeded.oid,
                    Some(external_field(ROW_TWO, K, 2)),
                    "nvf-writer",
                    3,
                )
                .await
                .expect("R2 := K must commit");
        }

        async fn obj_ext(tv: &TestValkey, oid: u64) -> HashMap<String, String> {
            tv.client()
                .pool
                .hgetall(tv.key_schema().obj_ext_index(&oid.to_string()))
                .await
                .expect("HGETALL obj:ext must succeed")
        }

        /// Whether the object's header carries the repair stamp.
        async fn repair_stamped(tv: &TestValkey, oid: &str) -> bool {
            tv.client()
                .pool
                .hexists(
                    tv.key_schema().obj_header(oid),
                    HEADER_FIELD_EXT_INDEX_REPAIRED_MS,
                )
                .await
                .expect("HEXISTS obj:hdr must succeed")
        }

        /// `(meta_present, data_present)` for one blob, read as a pair because
        /// the GC deletes both together.
        async fn blob_present(tv: &TestValkey, hash_byte: u8) -> (bool, bool) {
            let hex = hex_of(hash_byte);
            let meta: bool = tv
                .client()
                .pool
                .exists(tv.key_schema().ext_meta(&hex))
                .await
                .expect("EXISTS ext_meta must succeed");
            let data: bool = tv
                .client()
                .pool
                .exists(tv.key_schema().ext_data(&hex))
                .await
                .expect("EXISTS ext_data must succeed");
            (meta, data)
        }

        /// `live_edit_ext_index_recount_total{site="reconcile", outcome}`,
        /// panicking on an absent series (every cell is zero-seeded).
        fn recounts(metrics: &Metrics, outcome: &str) -> u64 {
            sample_value(
                &render(metrics),
                "live_edit_ext_index_recount_total",
                &[("site", "reconcile"), ("outcome", outcome)],
            )
            .unwrap_or_else(|| panic!("recount cell outcome={outcome} is zero-seeded"))
                as u64
        }

        /// `live_edit_ext_index_drift_total{site="reconcile", direction}`.
        fn drift(metrics: &Metrics, direction: &str) -> u64 {
            sample_value(
                &render(metrics),
                "live_edit_ext_index_drift_total",
                &[("site", "reconcile"), ("direction", direction)],
            )
            .unwrap_or_else(|| panic!("drift cell direction={direction} is zero-seeded"))
                as u64
        }

        /// `live_edit_ext_ref_released_total{reason}`.
        fn released(metrics: &Metrics, reason: &str) -> u64 {
            sample_value(
                &render(metrics),
                "live_edit_ext_ref_released_total",
                &[("reason", reason)],
            )
            .unwrap_or_else(|| panic!("released cell reason={reason} is zero-seeded"))
                as u64
        }

        /// [`one_pass`] against an arbitrary client — the gate arm hands the
        /// pass a client whose recount row ceiling is lowered.
        async fn one_pass_on(
            client: &ValkeyClient,
            tv: &TestValkey,
            metrics: &Arc<Metrics>,
            replica: &str,
        ) {
            run_one_pass(
                client,
                tv.key_schema(),
                replica,
                TEST_INTERVAL,
                TEST_GRACE,
                &CancellationToken::new(),
                Some(&(metrics.clone() as SharedMetrics)),
            )
            .await;
        }

        // ── A: the seed's surplus is not drained under an under-counted index

        /// An under-count left by a seeding commit is repaired by the first
        /// pass, so draining the seed's surplus and then rewriting one of the
        /// two rows never deletes the blob the other row still holds.
        ///
        /// Without the audit the pass sees one live object referencing H with
        /// `ref_count` 2, registers the surplus and drains it a grace later —
        /// correctly, as far as the global count goes. But the index still
        /// says one row holds H, so the R2 := K rewrite takes it to 0 and
        /// releases H's last reference while R1 holds it: the GC deletes a
        /// blob in use. With the audit, pass 1 rewrites the index to 2 first,
        /// and the rewrite leaves H at 1 with nothing released.
        async fn a_seeded_under_count_is_repaired_before_its_surplus_drains(
            backend: UpdateLayerBackend,
            replica: &str,
        ) {
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only(replica));
            let seeded = seed_legacy_under_count(&tv, backend, &metrics, replica).await;
            let hex_h = hex_of(H);
            set_stamp(&tv, &hex_h, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;

            // ── Pass 1: the audit repairs; the surplus is registered ─────────
            one_pass(&tv, &metrics, replica).await;

            assert_eq!(
                obj_ext(&tv, seeded.oid).await,
                HashMap::from([(hex_h.clone(), "2".to_owned())]),
                "the audit must rewrite H's count to the TWO rows holding it, in the same pass \
                 that first sees the surplus. Still 1 means the next rewrite of either row \
                 releases H while the other row holds it"
            );
            assert_eq!(recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_REPAIRED), 1);
            assert_eq!(
                drift(&metrics, EXT_INDEX_DRIFT_DIRECTION_UNDER),
                1,
                "one index field corrected upward — the dangerous direction"
            );
            assert!(
                repair_stamped(&tv, &seeded.oid.to_string()).await,
                "a repair stamps the header, which is what cancels an in-flight txn commit \
                 planned from the old count"
            );
            assert_eq!(
                candidates(&tv).await,
                vec![format!("{hex_h}:1")],
                "the surplus itself is genuine — one live object, two increments — and the \
                 repaired index is what makes draining it safe, so it registers as usual"
            );

            // ── Pass 2: the surplus drains ───────────────────────────────────
            age_candidates(&tv).await;
            one_pass(&tv, &metrics, replica).await;
            assert_eq!(pending(&tv).await, vec![format!("{hex_h}:0:rec1")]);

            // ── The rewrite of R2, then one GC tick ──────────────────────────
            rewrite_row_two_to_k(&seeded).await;
            assert_eq!(
                released(&metrics, "rewrite"),
                0,
                "R2 gave H up but R1 still holds it, so the repaired count goes 2 → 1 and \
                 nothing is released"
            );
            process_pending_gc(
                seeded.ext.as_ref(),
                None,
                crate::storage::external_values::GcPolicy::IMMEDIATE,
            )
            .await;

            assert_eq!(
                blob_present(&tv, H).await,
                (true, true),
                "H must SURVIVE: R1 still references it. Gone here is the data loss — the \
                 drained surplus plus a rewrite release from an index that missed R1"
            );
            assert_eq!(
                ref_count(&tv, &hex_h).await,
                Some(1),
                "one live object references H, and the count says so"
            );
            assert_eq!(
                obj_ext(&tv, seeded.oid).await,
                HashMap::from([(hex_h, "1".to_owned()), (hex_of(K), "1".to_owned())]),
                "and the index now equals the rows: R1 holds H, R2 holds K"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn lua_a_seeded_under_count_is_repaired_before_its_surplus_drains() {
            a_seeded_under_count_is_repaired_before_its_surplus_drains(
                UpdateLayerBackend::Lua,
                "nvf-audit-a-lua",
            )
            .await;
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn txn_a_seeded_under_count_is_repaired_before_its_surplus_drains() {
            a_seeded_under_count_is_repaired_before_its_surplus_drains(
                UpdateLayerBackend::Txn,
                "nvf-audit-a-txn",
            )
            .await;
        }

        // ── The drain gate ──────────────────────────────────────────────────

        /// A surplus on a hash whose object's index the recount could NOT
        /// repair is neither registered (phase 4) nor drained from an earlier
        /// registration (phase 5), so the under-counted index's later rewrite
        /// release lands on the surplus instead of the last live reference.
        ///
        /// The recount is refused by its row ceiling: the pass runs on a client
        /// whose atomic-read item limit is 2, so the recount reads at most one
        /// row and the two-row object answers `too_large`.
        ///
        /// Phase 5's gate is reached through a candidate staged by hand,
        /// standing in for one registered on an earlier pass: with phase 4's
        /// gate holding, no pass over this fixture can register one, and a
        /// claim gate only a registration could reach would otherwise go
        /// untested.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_surplus_on_an_unrepairable_index_is_held_until_the_rewrite() {
            const REPLICA: &str = "nvf-audit-gate";
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only(REPLICA));
            let seeded =
                seed_legacy_under_count(&tv, UpdateLayerBackend::Lua, &metrics, REPLICA).await;
            let hex_h = hex_of(H);
            set_stamp(&tv, &hex_h, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;

            // Row ceiling = item limit / 2 = 1, below the object's two rows.
            let mut gated = tv.client().clone();
            gated.snapshot_atomic_read_max_items = 2;

            // ── Pass 1: the recount refuses; the meta is not registered ──────
            one_pass_on(&gated, &tv, &metrics, REPLICA).await;

            assert_eq!(recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_TOO_LARGE), 1);
            assert_eq!(
                obj_ext(&tv, seeded.oid).await,
                HashMap::from([(hex_h.clone(), "1".to_owned())]),
                "a refused recount writes nothing"
            );
            assert!(
                candidates(&tv).await.is_empty(),
                "phase 4 must NOT register H's surplus: the index may under-count a live row, \
                 and this surplus is the increment keeping the blob alive meanwhile"
            );
            assert_eq!(
                outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT),
                1
            );

            // ── Pass 2, with a registration from an earlier pass ─────────────
            let member = format!("{hex_h}:1");
            let _: i64 = tv
                .client()
                .pool
                .zadd(
                    tv.key_schema().ext_recon_candidates(),
                    None,
                    None,
                    false,
                    false,
                    (0.0, member.as_str()),
                )
                .await
                .expect("staging an aged candidate must succeed");
            one_pass_on(&gated, &tv, &metrics, REPLICA).await;

            assert!(
                pending(&tv).await.is_empty(),
                "phase 5 must not drain a claimed surplus on a gated hash either"
            );
            assert_eq!(
                candidates(&tv).await,
                vec![member],
                "the gated candidate keeps its registration, to be re-judged once the index \
                 can be repaired"
            );
            assert_eq!(
                outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT),
                3,
                "pass 2 holds the surplus twice — once at the meta walk, once at the claim"
            );
            assert_eq!(outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED), 0);
            assert_eq!(ref_count(&tv, &hex_h).await, Some(2));

            // ── The rewrite releases on the short count; the held surplus pays
            rewrite_row_two_to_k(&seeded).await;
            assert_eq!(
                released(&metrics, "rewrite"),
                1,
                "the unrepaired index still reads 1 → 0 for H, so the rewrite releases it — \
                 exactly the release the held surplus has to absorb"
            );
            process_pending_gc(
                seeded.ext.as_ref(),
                None,
                crate::storage::external_values::GcPolicy::IMMEDIATE,
            )
            .await;
            assert_eq!(
                blob_present(&tv, H).await,
                (true, true),
                "H must survive the rewrite: R1 still holds it, and the held surplus is what \
                 absorbed the index's premature release"
            );
            assert_eq!(ref_count(&tv, &hex_h).await, Some(1));
        }

        /// The gate also holds when the recount answers `not_live`: an object
        /// that is retiring (`superseded`) still contributes its rows to the
        /// walk, so its unrepaired index gates the hash like an oversized one.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_surplus_on_a_not_live_object_index_is_not_registered() {
            const REPLICA: &str = "nvf-audit-gate-not-live";
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only(REPLICA));
            let seeded =
                seed_legacy_under_count(&tv, UpdateLayerBackend::Lua, &metrics, REPLICA).await;
            let hex_h = hex_of(H);
            set_stamp(&tv, &hex_h, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
            let _: () = tv
                .client()
                .pool
                .hset(
                    tv.key_schema().obj_header(&seeded.oid.to_string()),
                    [("superseded", "1")],
                )
                .await
                .expect("marking the header superseded must succeed");

            one_pass(&tv, &metrics, REPLICA).await;

            assert_eq!(recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_NOT_LIVE), 1);
            assert_eq!(
                obj_ext(&tv, seeded.oid).await,
                HashMap::from([(hex_h.clone(), "1".to_owned())])
            );
            assert!(candidates(&tv).await.is_empty());
            assert_eq!(
                outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT),
                1
            );
            assert_eq!(ref_count(&tv, &hex_h).await, Some(2));
        }

        // ── The audit in both directions ───────────────────────────────────

        /// Which index family a hand-staged row belongs to.
        #[derive(Clone, Copy)]
        enum Family {
            Field,
            TimeSample,
        }

        /// Stage one value row with `columns`, and its index member.
        async fn seed_row(
            tv: &TestValkey,
            oid: &str,
            family: Family,
            name: &str,
            columns: &[(&str, &str)],
        ) {
            let keys = tv.key_schema();
            let (index, member, row) = match family {
                Family::Field => {
                    let member = KeySchema::field_member("1", name);
                    let row = keys.obj_field_from_member(oid, &member);
                    (keys.obj_fields_index(oid), member, row)
                }
                Family::TimeSample => {
                    let time: f64 = name.parse().expect("a time-sample name is its time");
                    let member = KeySchema::time_sample_member("1", time);
                    let row = keys.obj_ts_from_member(oid, &member);
                    (keys.obj_time_samples_index(oid), member, row)
                }
            };
            let _: () = tv
                .client()
                .pool
                .hset(row, columns.to_vec())
                .await
                .expect("seeding a value row must succeed");
            let _: i64 = tv
                .client()
                .pool
                .zadd(index, None, None, false, false, (1.0, member.as_str()))
                .await
                .expect("seeding a row index member must succeed");
        }

        async fn seed_live_header(tv: &TestValkey, oid: &str) {
            let _: () = tv
                .client()
                .pool
                .hset(tv.key_schema().obj_header(oid), [("seq", "1")])
                .await
                .expect("seeding obj:hdr must succeed");
        }

        /// `HGETALL obj:ext` as sorted `(hash_hex, stored)` pairs.
        async fn index_sorted(tv: &TestValkey, oid: &str) -> Vec<(String, String)> {
            let index: HashMap<String, String> = tv
                .client()
                .pool
                .hgetall(tv.key_schema().obj_ext_index(oid))
                .await
                .expect("HGETALL obj:ext must succeed");
            let mut fields: Vec<(String, String)> = index.into_iter().collect();
            fields.sort();
            fields
        }

        async fn seed_index(tv: &TestValkey, oid: &str, fields: &[(&str, &str)]) {
            let _: () = tv
                .client()
                .pool
                .hset(tv.key_schema().obj_ext_index(oid), fields.to_vec())
                .await
                .expect("seeding obj:ext must succeed");
        }

        /// One pass repairs an over-count, an under-count and a missing field
        /// to the value-aware row counts, ticks drift by direction, stamps only
        /// the repaired objects, and releases nothing.
        ///
        /// Five objects, each a distinct case of the walk's raw comparison:
        ///
        /// | object | rows holding the hash | index | recount |
        /// |---|---|---|---|
        /// | over    | field, plus a dual-shape field | 3 | repaired to 1 (over) |
        /// | under   | field, plus a time sample      | 1 | repaired to 2 (under) |
        /// | missing | field                          | — | repaired to 1 (under) |
        /// | clean   | field                          | 1 | not recounted |
        /// | legacy  | a dual-shape field only        | — | `clean` |
        ///
        /// The dual-shape rows are what make "value-aware" observable: the
        /// walk's raw read counts them, the recount must not. The `over`
        /// object's answer is therefore 1, not the raw 2, and the `legacy`
        /// object's raw mismatch triggers a recount that finds nothing to fix.
        /// The time sample in `under` keeps the recount's second row family
        /// in the count.
        ///
        /// "Releases nothing" is asserted on both release surfaces: no
        /// pending member and no `ext_ref_released_total` tick in any reason.
        /// Lowering an over-count leaves its surplus to a later pass's ordinary
        /// two-observation drain; raising an under-count changes no global
        /// count at all.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn one_pass_repairs_index_drift_in_both_directions_and_releases_nothing() {
            const REPLICA: &str = "nvf-audit-directions";
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only(REPLICA));
            let (over, under, missing, clean, legacy) =
                ("950101", "950102", "950103", "950104", "950105");
            let [h1, h2, h3, h4, h5] = [0x61, 0x62, 0x63, 0x64, 0x65].map(hex_of);

            for oid in [over, under, missing, clean, legacy] {
                seed_live_header(&tv, oid).await;
            }
            seed_row(
                &tv,
                over,
                Family::Field,
                "a",
                &[("hash", &h1), ("size", "1")],
            )
            .await;
            seed_row(
                &tv,
                over,
                Family::Field,
                "b",
                &[("value", "inline"), ("hash", &h1), ("size", "1")],
            )
            .await;
            seed_index(&tv, over, &[(&h1, "3")]).await;

            seed_row(
                &tv,
                under,
                Family::Field,
                "a",
                &[("hash", &h2), ("size", "1")],
            )
            .await;
            seed_row(
                &tv,
                under,
                Family::TimeSample,
                "2.5",
                &[("hash", &h2), ("size", "1")],
            )
            .await;
            seed_index(&tv, under, &[(&h2, "1")]).await;

            seed_row(
                &tv,
                missing,
                Family::Field,
                "a",
                &[("hash", &h3), ("size", "1")],
            )
            .await;

            seed_row(
                &tv,
                clean,
                Family::Field,
                "a",
                &[("hash", &h4), ("size", "1")],
            )
            .await;
            seed_index(&tv, clean, &[(&h4, "1")]).await;

            seed_row(
                &tv,
                legacy,
                Family::Field,
                "a",
                &[("value", "inline"), ("hash", &h5), ("size", "1")],
            )
            .await;

            one_pass(&tv, &metrics, REPLICA).await;

            assert_eq!(
                index_sorted(&tv, over).await,
                vec![(h1.clone(), "1".to_owned())],
                "over-count lowered to the VALUE-AWARE count: the dual-shape row holds no hash, \
                 so 1 — not the raw 2"
            );
            assert_eq!(
                index_sorted(&tv, under).await,
                vec![(h2.clone(), "2".to_owned())],
                "under-count raised to both rows, the time sample included"
            );
            assert_eq!(
                index_sorted(&tv, missing).await,
                vec![(h3.clone(), "1".to_owned())],
                "a missing field is an under-count from 0"
            );
            assert_eq!(
                index_sorted(&tv, clean).await,
                vec![(h4.clone(), "1".to_owned())]
            );
            assert_eq!(
                index_sorted(&tv, legacy).await,
                Vec::<(String, String)>::new(),
                "the dual-shape row's stale hash is not a reference, so nothing is written"
            );

            assert_eq!(drift(&metrics, EXT_INDEX_DRIFT_DIRECTION_UNDER), 2);
            assert_eq!(drift(&metrics, EXT_INDEX_DRIFT_DIRECTION_OVER), 1);
            assert_eq!(recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_REPAIRED), 3);
            assert_eq!(
                recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_CLEAN),
                1,
                "the legacy object's raw mismatch is recounted and found clean; the clean \
                 object is never recounted at all"
            );

            for (oid, repaired) in [
                (over, true),
                (under, true),
                (missing, true),
                (clean, false),
                (legacy, false),
            ] {
                assert_eq!(
                    repair_stamped(&tv, oid).await,
                    repaired,
                    "object {oid}: `ext_index_repaired_ms` is written by a repair and only by \
                     a repair"
                );
            }

            assert!(
                pending(&tv).await.is_empty(),
                "the audit releases nothing — no pending member"
            );
            for reason in EXT_REF_RELEASE_REASONS {
                assert_eq!(released(&metrics, reason), 0, "reason={reason}");
            }
        }

        // ── The index's own leg, and the over-count removal ────────────────

        /// An object whose `obj:ext` alone names H — no row holds it — counts
        /// as referencing H on the pass that finds it, has the field REMOVED,
        /// and the `+1` that frees is drained on later passes as ordinary
        /// surplus.
        ///
        /// Three sites, each with nothing else pinning it:
        ///
        /// * The walk's union of rows ∪ index. On pass 1 the index is all
        ///   that names H; if the walk counted rows only, H would read as
        ///   unreferenced surplus and register on pass 1, the pass that
        ///   removed the field. The union keeps the index entry a reference
        ///   for the pass that finds it, so the freed `+1` first shows up as
        ///   surplus on pass 2.
        /// * "Present ⟺ count > 0": a field for a hash no row holds is
        ///   deleted, not lowered to 0.
        /// * The recount releases nothing: the freed reference surfaces as
        ///   surplus on pass 2 and drains through the two-observation
        ///   protocol on pass 3, under every guard that protocol applies.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn an_index_only_reference_is_removed_and_its_freed_increment_drains_later() {
            const REPLICA: &str = "nvf-audit-index-only";
            const OID: &str = "950201";
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only(REPLICA));
            let ext = store(&tv);
            let hex_h = hex_of(0x66);
            pre_store(&ext, 0x66).await;
            set_stamp(&tv, &hex_h, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
            seed_live_header(&tv, OID).await;
            seed_index(&tv, OID, &[(hex_h.as_str(), "1")]).await;

            // ── Pass 1: the index still counts; the field is removed ─────────
            one_pass(&tv, &metrics, REPLICA).await;

            assert!(
                candidates(&tv).await.is_empty(),
                "on the pass that finds it, the index entry is a reference: referenced 1 = \
                 ref_count 1, so there is no surplus to register yet"
            );
            assert_eq!(
                index_sorted(&tv, OID).await,
                Vec::<(String, String)>::new(),
                "a field for a hash no row holds is DELETED — present ⟺ count > 0 — not \
                 lowered to 0"
            );
            assert_eq!(drift(&metrics, EXT_INDEX_DRIFT_DIRECTION_OVER), 1);
            assert_eq!(drift(&metrics, EXT_INDEX_DRIFT_DIRECTION_UNDER), 0);
            assert!(
                pending(&tv).await.is_empty(),
                "the recount releases nothing"
            );
            assert_eq!(ref_count(&tv, &hex_h).await, Some(1));

            // ── Pass 2: the freed increment is ordinary surplus ──────────────
            one_pass(&tv, &metrics, REPLICA).await;
            assert_eq!(
                candidates(&tv).await,
                vec![format!("{hex_h}:1")],
                "with the field gone nothing references H, so its +1 registers as surplus"
            );

            // ── Pass 3, a grace later: it drains ─────────────────────────────
            age_candidates(&tv).await;
            one_pass(&tv, &metrics, REPLICA).await;
            assert_eq!(pending(&tv).await, vec![format!("{hex_h}:0:rec1")]);
        }

        /// A stored count of `"0"`, or one that is not a plain decimal count,
        /// is drift — in the walk's raw comparison AND in the recount.
        ///
        /// * `zero`: no row holds H, and the index stores `"0"`. Numerically
        ///   equal to the row count, so only the walk's `rows == 0` clause
        ///   and the script's `want == 0` clause see it; either missing
        ///   leaves the field standing.
        /// * `garbled`: one row holds H, and the index stores `"1.0"` —
        ///   numerically EQUAL to the row count, but not a decimal count. The
        ///   walk's `u64` parse refuses it, so it must disagree with the row
        ///   count rather than be skipped; the script's decimal-only
        ///   `parse_count` refuses it too and reads it as 0, so the repair is
        ///   an under-count corrected to 1. A looser parse on either side
        ///   (Lua's `tonumber` reads it as 1) would call it clean and leave
        ///   `"1.0"` standing. (`"+1"` would not test the walk: Rust's `u64`
        ///   parse accepts it.)
        ///
        /// The two objects are asserted together so a failure shows both.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_zero_or_non_decimal_stored_count_is_drift_in_the_walk_and_the_recount() {
            const REPLICA: &str = "nvf-audit-unparseable";
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only(REPLICA));
            let (zero, garbled) = ("950301", "950302");
            let (hz, hg) = (hex_of(0x6A), hex_of(0x6B));
            seed_live_header(&tv, zero).await;
            seed_index(&tv, zero, &[(hz.as_str(), "0")]).await;
            seed_live_header(&tv, garbled).await;
            seed_row(
                &tv,
                garbled,
                Family::Field,
                "a",
                &[("hash", hg.as_str()), ("size", "1")],
            )
            .await;
            seed_index(&tv, garbled, &[(hg.as_str(), "1.0")]).await;

            one_pass(&tv, &metrics, REPLICA).await;

            assert_eq!(
                (
                    index_sorted(&tv, zero).await,
                    index_sorted(&tv, garbled).await
                ),
                (Vec::new(), vec![(hg.clone(), "1".to_owned())]),
                "a `0` field is removed and a `1.0` one rewritten with the true count"
            );
            assert_eq!(
                (
                    drift(&metrics, EXT_INDEX_DRIFT_DIRECTION_OVER),
                    drift(&metrics, EXT_INDEX_DRIFT_DIRECTION_UNDER)
                ),
                (1, 1),
                "the `0` field is corrected downward (removed); the `1.0` one, read as no count \
                 (0), upward"
            );
            assert_eq!(recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_REPAIRED), 2);
        }

        // ── Fail-closed, and what the gate leaves alone ────────────────────

        /// A recount that FAILS forfeits the whole pass: no gauges, no
        /// candidate, no release, and nothing written.
        ///
        /// The fail-closed leg of the audit. A failed recount means the
        /// object's drift is neither repaired nor provably absent, so a
        /// surplus on its hash must not drain on the strength of this pass —
        /// and since the hash is not in the gate's set either, only the
        /// forfeit stops it. Here the surplus is fully eligible (two
        /// increments, one referencing object, an old stamp), so the forfeit
        /// is the only reason nothing registers.
        ///
        /// Staged as a standing condition, not an injected one: `obj:hdr` is
        /// planted as a STRING. The walk SCANs `obj:hdr` keys with no type
        /// filter and never reads them, so it reaches the object, sees the
        /// row's hash with no index field, and runs the recount — whose type
        /// probe refuses the header (`WRONGTYPE_KEY`) before any write.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_failed_recount_forfeits_the_pass_and_writes_nothing() {
            const REPLICA: &str = "nvf-audit-recount-failed";
            const OID: &str = "950401";
            const PLANTED: &str = "planted-by-an-out-of-band-writer";
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only(REPLICA));
            let ext = store(&tv);
            let hex = hex_of(0x67);
            pre_store(&ext, 0x67).await;
            pre_store(&ext, 0x67).await;
            set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
            seed_row(
                &tv,
                OID,
                Family::Field,
                "a",
                &[("hash", hex.as_str()), ("size", "1")],
            )
            .await;
            let _: () = tv
                .client()
                .pool
                .set(tv.key_schema().obj_header(OID), PLANTED, None, None, false)
                .await
                .expect("planting a string over obj:hdr must succeed");

            one_pass(&tv, &metrics, REPLICA).await;

            assert_eq!(
                outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
                1,
                "a failed recount forfeits the pass, counted on the alertable cell"
            );
            assert_eq!(recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_FAILED), 1);
            assert_eq!(
                gauges(&metrics, REPLICA),
                None,
                "a forfeited pass publishes nothing"
            );
            assert!(
                candidates(&tv).await.is_empty(),
                "H's surplus is eligible on every other guard, so registering it here would mean \
                 the pass carried on past an object whose drift it could neither repair nor gate"
            );
            assert!(pending(&tv).await.is_empty());
            assert_eq!(
                index_sorted(&tv, OID).await,
                Vec::<(String, String)>::new(),
                "the refused recount wrote nothing to the index"
            );
            let header: Option<String> = tv
                .client()
                .pool
                .get(tv.key_schema().obj_header(OID))
                .await
                .expect("GET obj:hdr must succeed");
            assert_eq!(
                header.as_deref(),
                Some(PLANTED),
                "nor to the header: the refusal comes before the first write"
            );
        }

        /// The drain gate holds SURPLUSES only: a deficit on a gated hash
        /// still registers for confirmation.
        ///
        /// A deficit is never repaired, so holding it would buy no safety
        /// and would hide an over-release from the alert that exists for
        /// one. The gate therefore sits after the deficit arm in the meta
        /// walk. Here H's object has an unrepairable index (two rows, a
        /// recount ceiling of one row) and H's `ref_count` reads 0 against
        /// one referencing object.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_deficit_on_a_gated_hash_still_registers() {
            const REPLICA: &str = "nvf-audit-gated-deficit";
            const OID: &str = "950501";
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only(REPLICA));
            let ext = store(&tv);
            let hex = hex_of(0x69);
            pre_store(&ext, 0x69).await;
            set_ref_count(&tv, &hex, 0).await;
            seed_live_header(&tv, OID).await;
            for name in ["a", "b"] {
                seed_row(
                    &tv,
                    OID,
                    Family::Field,
                    name,
                    &[("hash", hex.as_str()), ("size", "1")],
                )
                .await;
            }
            seed_index(&tv, OID, &[(hex.as_str(), "1")]).await;

            let mut gated = tv.client().clone();
            gated.snapshot_atomic_read_max_items = 2;
            one_pass_on(&gated, &tv, &metrics, REPLICA).await;

            assert_eq!(
                recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_TOO_LARGE),
                1,
                "precondition: the recount refused, so H is in the gate's set"
            );
            assert_eq!(
                candidates(&tv).await,
                vec![format!("{hex}:-1")],
                "the deficit registers for confirmation exactly as on an ungated hash"
            );
            assert_eq!(
                outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT),
                0,
                "and the gate counted nothing: it holds surpluses only"
            );
        }

        // ── A malformed row holds the hash, whatever the recount answers ───

        /// A layer carrying a `hash` column the recount reports as malformed
        /// holds its mismatched hashes from surplus drain on every pass,
        /// whichever of its three answers the recount gives.
        ///
        /// H is held by two lowercase rows and one UPPERCASE row. The walk's
        /// `hash_from_hex` accepts uppercase, so it counts 3; the recount
        /// counts only the lowercase rows, 2, and reports one malformed row.
        /// Its count therefore omits a row the decoder serves, and no answer
        /// makes the index safe to drain against. Three stored counts reach
        /// the three answers:
        ///
        /// * stored `"2"`: the recount agrees with itself and answers
        ///   `CLEAN { malformed: 1 }`, writing nothing;
        /// * stored `"1"`: it raises to 2 and answers
        ///   `REPAIRED { malformed: 1 }`;
        /// * stored `"4"`: it would lower to 2, withholds that, and answers
        ///   `WITHHELD { raised: 0, withheld: 1, malformed: 1 }`, leaving
        ///   `"4"`.
        ///
        /// `index_after_pass_1` is the index each answer leaves, asserted as
        /// the precondition that the intended answer was reached. Either way
        /// the walk still disagrees with the index on every later pass (3
        /// against 2, or against 4), so the hold recurs. H's surplus of 1
        /// (two increments, one referencing layer, an old stamp) is eligible
        /// on every other guard. Pass 2 adds a candidate registered by an
        /// earlier pass, so the claim gate is exercised as well as the meta
        /// gate.
        async fn a_malformed_row_holds_the_hash_whatever_the_recount_answers(
            stored: &str,
            index_after_pass_1: &str,
            replica: &str,
        ) {
            const OID: &str = "950601";
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only(replica));
            let ext = store(&tv);
            let hex = hex_of(0xE1);
            pre_store(&ext, 0xE1).await;
            pre_store(&ext, 0xE1).await;
            set_stamp(&tv, &hex, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
            seed_live_header(&tv, OID).await;
            for name in ["a", "b"] {
                seed_row(
                    &tv,
                    OID,
                    Family::Field,
                    name,
                    &[("hash", hex.as_str()), ("size", "1")],
                )
                .await;
            }
            seed_row(
                &tv,
                OID,
                Family::Field,
                "c",
                &[("hash", hex.to_uppercase().as_str()), ("size", "1")],
            )
            .await;
            seed_index(&tv, OID, &[(hex.as_str(), stored)]).await;

            // ── Pass 1 ───────────────────────────────────────────────────────
            one_pass(&tv, &metrics, replica).await;

            assert_eq!(
                index_sorted(&tv, OID).await,
                vec![(hex.clone(), index_after_pass_1.to_owned())],
                "fixture precondition: the recount ran and gave the intended answer — a `1` is \
                 raised to 2 (REPAIRED), a `2` left alone (CLEAN), a `4` left alone because its \
                 lowering is withheld (WITHHELD)"
            );
            assert!(
                candidates(&tv).await.is_empty(),
                "H's surplus must NOT register: the recount's count omits the uppercase row the \
                 decoder serves, so the layer is not repaired whatever the recount answered"
            );
            assert_eq!(
                outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT),
                1
            );

            // ── Pass 2, with a registration from an earlier pass ────────────
            let member = format!("{hex}:1");
            let _: i64 = tv
                .client()
                .pool
                .zadd(
                    tv.key_schema().ext_recon_candidates(),
                    None,
                    None,
                    false,
                    false,
                    (0.0, member.as_str()),
                )
                .await
                .expect("staging an aged candidate must succeed");
            one_pass(&tv, &metrics, replica).await;

            assert!(
                pending(&tv).await.is_empty(),
                "the hold recurs on the next pass, so a claimed surplus is not drained either"
            );
            assert_eq!(candidates(&tv).await, vec![member]);
            assert_eq!(
                outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT),
                3,
                "pass 2 holds H at the meta walk and at the claim"
            );
            assert_eq!(outcome(&metrics, EXT_RECONCILE_OUTCOME_REPAIRED), 0);
            assert_eq!(ref_count(&tv, &hex).await, Some(2));
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_malformed_row_holds_the_hash_when_the_recount_answers_clean() {
            a_malformed_row_holds_the_hash_whatever_the_recount_answers(
                "2",
                "2",
                "nvf-audit-malformed-clean",
            )
            .await;
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_malformed_row_holds_the_hash_when_the_recount_answers_repaired() {
            a_malformed_row_holds_the_hash_whatever_the_recount_answers(
                "1",
                "2",
                "nvf-audit-malformed-repaired",
            )
            .await;
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_malformed_row_holds_the_hash_when_the_recount_answers_withheld() {
            a_malformed_row_holds_the_hash_whatever_the_recount_answers(
                "4",
                "4",
                "nvf-audit-malformed-withheld",
            )
            .await;
        }

        // ══ b.8jc — the walk's signed-zero merge, inside a whole pass ═══════
        //
        // The merge's own decision table, guards and races are pinned against
        // its public API in `tests/integration/storage/time_sample_merge_test.rs`.
        // What only a pass shows is pinned here: that the walk finds and
        // merges a legacy `-0.0` row, that the merge itself releases nothing
        // while the same pass's index audit lowers the over-count the dropped
        // row leaves, that a `-0.0` row an older build writes after a pass
        // is merged by the next one, and that a failed merge is counted and
        // logged without costing the pass. Nested in `index_audit` for its
        // staging and reading helpers.
        mod signed_zero_merge {
            use super::*;

            use crate::observability::metrics::{
                TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOMES,
                TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_FAILED,
                TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_MERGED,
                TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_REKEYED,
            };
            use crate::storage::time_sample_merge::NEGATIVE_ZERO_TIME_HEX;

            /// The section every staged row is on.
            const SECTION: &str = "4";

            /// The stored key of `SECTION`'s time-sample row at `time`.
            fn row_key(tv: &TestValkey, oid: &str, time: f64) -> String {
                tv.key_schema().obj_time_sample(oid, SECTION, time)
            }

            /// Stage `SECTION`'s row at `time` with `columns`, and its `obj:tss`
            /// member at score `version`, as an older build's commit wrote it.
            async fn stage_row(
                tv: &TestValkey,
                oid: &str,
                time: f64,
                version: u64,
                columns: &[(&str, &str)],
            ) {
                let mut columns = columns.to_vec();
                let version_str = version.to_string();
                columns.push(("version", version_str.as_str()));
                let _: () = tv
                    .client()
                    .pool
                    .hset(row_key(tv, oid, time), columns)
                    .await
                    .expect("staging a time-sample row must succeed");
                let _: i64 = tv
                    .client()
                    .pool
                    .zadd(
                        tv.key_schema().obj_time_samples_index(oid),
                        None,
                        None,
                        false,
                        false,
                        (version as f64, KeySchema::time_sample_member(SECTION, time)),
                    )
                    .await
                    .expect("staging an obj:tss member must succeed");
            }

            async fn tss_members(tv: &TestValkey, oid: &str) -> Vec<String> {
                let mut members: Vec<String> = tv
                    .client()
                    .pool
                    .zrange(
                        tv.key_schema().obj_time_samples_index(oid),
                        0,
                        -1,
                        None,
                        false,
                        None,
                        false,
                    )
                    .await
                    .expect("ZRANGE obj:tss must succeed");
                members.sort();
                members
            }

            async fn row(tv: &TestValkey, oid: &str, time: f64) -> HashMap<String, String> {
                tv.client()
                    .pool
                    .hgetall(row_key(tv, oid, time))
                    .await
                    .expect("HGETALL a time-sample row must succeed")
            }

            /// `live_edit_time_sample_signed_zero_merge_total{outcome}`, which is
            /// zero-seeded.
            fn merged(metrics: &Metrics, outcome: &str) -> u64 {
                sample_value(
                    &render(metrics),
                    "live_edit_time_sample_signed_zero_merge_total",
                    &[("outcome", outcome)],
                )
                .unwrap_or_else(|| panic!("merge cell outcome={outcome} is zero-seeded"))
                    as u64
            }

            /// A pass merges a stored pair by rule B — here the newer `-0.0` row,
            /// external of `W`, over the `+0.0` row, external of `L` — into one
            /// row at `+0.0` holding `W`. The merge writes no `obj:ext` field and
            /// releases nothing: `L`'s count is left over by one, which the same
            /// pass's index audit then lowers (removes), while `W`, which the
            /// surviving row holds, keeps its count of 1 and is named by no
            /// release.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn a_pass_merges_a_pair_releases_nothing_and_its_audit_lowers_the_over_count() {
                const REPLICA: &str = "b8jc-merge-pass";
                const OID: &str = "980101";
                let tv = TestValkey::setup().await;
                let metrics = Arc::new(metrics_scrape_only(REPLICA));
                let (loser, winner) = (hex_of(0x81), hex_of(0x82));
                seed_live_header(&tv, OID).await;
                stage_row(&tv, OID, 0.0, 1, &[("hash", &loser), ("size", "8")]).await;
                stage_row(&tv, OID, -0.0, 2, &[("hash", &winner), ("size", "8")]).await;
                seed_index(&tv, OID, &[(&loser, "1"), (&winner, "1")]).await;

                one_pass(&tv, &metrics, REPLICA).await;

                assert_eq!(
                    tss_members(&tv, OID).await,
                    vec![KeySchema::time_sample_member(SECTION, 0.0)],
                    "one member, at +0.0"
                );
                assert!(row(&tv, OID, -0.0).await.is_empty(), "the -0.0 row is gone");
                assert_eq!(
                    row(&tv, OID, 0.0).await.get("hash"),
                    Some(&winner),
                    "the +0.0 row holds the newer -0.0 row's value"
                );
                assert_eq!(
                    index_sorted(&tv, OID).await,
                    vec![(winner.clone(), "1".to_owned())],
                    "the audit removed the dropped row's count; the winner's stays 1"
                );
                assert_eq!(
                    drift(&metrics, EXT_INDEX_DRIFT_DIRECTION_OVER),
                    1,
                    "the merge left an over-count, the leak direction, which the audit lowered"
                );
                assert!(
                    pending(&tv).await.is_empty(),
                    "nothing is released: neither the merge nor the audit releases"
                );
                assert_eq!(
                    (
                        merged(&metrics, TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_MERGED),
                        merged(&metrics, TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_REKEYED)
                    ),
                    (1, 0)
                );
            }

            /// A `-0.0` row an older build writes after a pass — during a
            /// rolling upgrade, beside a `+0.0` row this build wrote — is merged
            /// by the next pass, which keeps the newer row's value at `+0.0`.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn a_negative_zero_row_written_after_a_pass_is_merged_by_the_next_pass() {
                const REPLICA: &str = "b8jc-merge-next-pass";
                const OID: &str = "980201";
                let tv = TestValkey::setup().await;
                let metrics = Arc::new(metrics_scrape_only(REPLICA));
                seed_live_header(&tv, OID).await;
                stage_row(&tv, OID, 0.0, 1, &[("value", "AQ==")]).await;

                one_pass(&tv, &metrics, REPLICA).await;
                assert_eq!(
                    merged(&metrics, TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_MERGED),
                    0,
                    "fixture: nothing to merge on the first pass"
                );

                stage_row(&tv, OID, -0.0, 2, &[("value", "Ag==")]).await;
                assert!(
                    tss_members(&tv, OID)
                        .await
                        .iter()
                        .any(|m| m.ends_with(NEGATIVE_ZERO_TIME_HEX)),
                    "fixture: the older build's row is listed"
                );

                one_pass(&tv, &metrics, REPLICA).await;

                assert_eq!(
                    tss_members(&tv, OID).await,
                    vec![KeySchema::time_sample_member(SECTION, 0.0)]
                );
                assert!(row(&tv, OID, -0.0).await.is_empty());
                assert_eq!(
                    row(&tv, OID, 0.0).await.get("value").map(String::as_str),
                    Some("Ag=="),
                    "the newer -0.0 row's value survives at +0.0"
                );
                assert_eq!(
                    merged(&metrics, TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_MERGED),
                    1
                );
            }

            /// The captured events whose message contains `needle`.
            fn events_with<'a>(
                events: &'a [serde_json::Value],
                needle: &str,
            ) -> Vec<&'a serde_json::Value> {
                events
                    .iter()
                    .filter(|e| {
                        e["fields"]["message"]
                            .as_str()
                            .is_some_and(|m| m.contains(needle))
                    })
                    .collect()
            }

            /// A merge that fails is counted `failed`, logged at WARN, and does
            /// not cost the pass: the walk goes on, audits the next object and
            /// completes.
            ///
            /// The broken object's `-0.0` row is ordinary, but its section's
            /// `+0.0` key holds a string, so the merge's read (`HGETALL` of
            /// both rows) is refused `WRONGTYPE`. The `+0.0` key has no
            /// `obj:tss` member, so nothing else the walk reads names it. A
            /// second object carries an over-counted `obj:ext`, which the
            /// pass's index audit lowers.
            ///
            /// "The pass completes" is read off the two things only a
            /// completed pass produces: the reconciler gauges, published by
            /// `run_one_pass` on success alone, and the pass-complete INFO
            /// line; and `enumeration_failed`, which a forfeit ticks, stays 0.
            /// The second object's repair alone would not show it, because the
            /// walk's `SCAN` may reach that object before the broken one.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn a_failed_merge_is_counted_and_logged_and_the_pass_completes() {
                const REPLICA: &str = "b8jc-merge-failed";
                const BROKEN: &str = "980301";
                const DRIFTED: &str = "980302";
                let tv = TestValkey::setup().await;
                let metrics = Arc::new(metrics_scrape_only(REPLICA));

                seed_live_header(&tv, BROKEN).await;
                stage_row(&tv, BROKEN, -0.0, 2, &[("value", "Ag==")]).await;
                let _: () = tv
                    .client()
                    .pool
                    .set(row_key(&tv, BROKEN, 0.0), "not-a-row", None, None, false)
                    .await
                    .expect("staging a string at the +0.0 row key must succeed");

                seed_live_header(&tv, DRIFTED).await;
                let h = hex_of(0x83);
                seed_row(
                    &tv,
                    DRIFTED,
                    Family::Field,
                    "a",
                    &[("hash", &h), ("size", "1")],
                )
                .await;
                seed_index(&tv, DRIFTED, &[(&h, "2")]).await;

                let events = capture_events(one_pass(&tv, &metrics, REPLICA)).await;

                for &cell in TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOMES {
                    let expected = u64::from(cell == TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_FAILED);
                    assert_eq!(
                        merged(&metrics, cell),
                        expected,
                        "outcome={cell}: the one row is counted failed, and only that"
                    );
                }
                let warns = events_with(&events, "the signed-zero merge of a layer's");
                assert_eq!(warns.len(), 1, "one merge-failure line: {events:?}");
                let warn = warns[0];
                assert_eq!(warn["level"], "WARN");
                assert_eq!(warn["fields"]["object_id"], BROKEN);
                assert_eq!(warn["fields"]["rows"], 1);
                assert!(
                    warn["fields"]["error"]
                        .as_str()
                        .is_some_and(|e| e.contains("WRONGTYPE")),
                    "the line carries the store's refusal: {warn}"
                );

                assert_eq!(
                    tss_members(&tv, BROKEN).await,
                    vec![KeySchema::time_sample_member(SECTION, -0.0)],
                    "the failed merge wrote nothing: the -0.0 member stays"
                );
                assert_eq!(
                    row(&tv, BROKEN, -0.0)
                        .await
                        .get("value")
                        .map(String::as_str),
                    Some("Ag=="),
                    "the failed merge wrote nothing: the -0.0 row stays"
                );

                assert!(
                    gauges(&metrics, REPLICA).is_some(),
                    "the pass completed: only a completed pass publishes the gauges"
                );
                assert_eq!(
                    outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
                    0,
                    "the pass was not forfeited"
                );
                let complete =
                    events_with(&events, "external-value ref_count reconcile pass complete");
                assert_eq!(complete.len(), 1, "the pass-complete line: {events:?}");
                assert_eq!(
                    complete[0]["fields"]["time_sample_rows_merged"], 0,
                    "nothing was merged"
                );
                assert!(
                    events_with(&events, "reconcile pass forfeited").is_empty(),
                    "no forfeit line"
                );

                assert_eq!(
                    index_sorted(&tv, DRIFTED).await,
                    vec![(h.clone(), "1".to_owned())],
                    "the walk audited the other object: its over-count is lowered"
                );
                assert_eq!(drift(&metrics, EXT_INDEX_DRIFT_DIRECTION_OVER), 1);
            }
        }

        // ══ The restore Job's authoritative recount (SDD SR-9.8) ═══════════
        //
        // `recount_authoritative` against a hand-staged keyspace, with a
        // `before_write` that never refuses: the Job's permit and kill point
        // are its caller's (`dr::job::finish`), pinned end to end in
        // `tests/integration/dr/restore_finish_test.rs`. Nested in
        // `index_audit` for its row, header and index staging helpers.
        //
        // Fail-on-old, each by an in-place perturbation of the recount
        // (red, then green once restored):
        //
        // * the two PD-1 condition-2 pins: with the walk's rows no longer
        //   inserted into the union, the index-drift pin reads 1, not 2 (a
        //   live row's reference lowered away); with the authoritative walk
        //   skipping an object whose index disagrees with its rows, the
        //   skipped-object pin reads 1, not 2;
        // * "fails if the guards are left on": with the count sets replaced by
        //   one periodic pass (the candidate path), the fresh-stamped surplus
        //   stays at 5.
        mod authoritative_recount {
            use super::*;

            use crate::storage::ext_backup::{BACKED_UP_IN_FIELD, BACKUP_CLEARED_MS_FIELD};

            /// The recount with a `before_write` that never refuses.
            async fn recount(tv: &TestValkey) -> Result<RecountReceipt, RecountError<()>> {
                recount_authoritative(tv.client(), tv.key_schema(), || async { Ok::<(), ()>(()) })
                    .await
            }

            /// [`recount`], which must complete; its report.
            async fn recounted(tv: &TestValkey) -> RecountReport {
                recount(tv)
                    .await
                    .expect("the recount completes")
                    .report()
                    .clone()
            }

            /// `hash_hex`'s meta, every field.
            async fn meta(tv: &TestValkey, hash_hex: &str) -> HashMap<String, String> {
                tv.client()
                    .pool
                    .hgetall(tv.key_schema().ext_meta(hash_hex))
                    .await
                    .expect("HGETALL ext:meta must succeed")
            }

            /// Whether `hash_hex`'s bytes are in the store.
            async fn data_exists(tv: &TestValkey, hash_hex: &str) -> bool {
                tv.client()
                    .pool
                    .exists::<bool, _>(tv.key_schema().ext_data(hash_hex))
                    .await
                    .expect("EXISTS ext:data must succeed")
            }

            /// Stage one release of `hash`, owed by object `oid`, where
            /// `stage` holds it.
            async fn owe(tv: &TestValkey, stage: OwedStage, hash: [u8; 32], oid: u64) {
                let keys = tv.key_schema();
                let member = ExtRelease::retirement(hash, oid).member();
                let carrier = keys.obj_ext_retire_carrier(&oid.to_string());
                match stage {
                    OwedStage::Pending => {
                        let _: u64 = tv
                            .client()
                            .pool
                            .sadd(keys.ext_pending(), member.as_str())
                            .await
                            .expect("seeding pending must succeed");
                    }
                    OwedStage::Inflight | OwedStage::InflightEncoded => {
                        let _: i64 = tv
                            .client()
                            .pool
                            .zadd(
                                keys.ext_inflight(),
                                None,
                                None,
                                false,
                                false,
                                (1.0, member.as_str()),
                            )
                            .await
                            .expect("seeding inflight must succeed");
                    }
                    OwedStage::RetireCarrier => {
                        let _: () = tv
                            .client()
                            .pool
                            .hset(carrier, [(member.as_str(), "1")])
                            .await
                            .expect("seeding the retire carrier must succeed");
                    }
                    OwedStage::LegacyRetireCarrier => {
                        let _: u64 = tv
                            .client()
                            .pool
                            .sadd(carrier, member.as_str())
                            .await
                            .expect("seeding the older build's carrier must succeed");
                    }
                }
            }

            /// A meta as a store and a confirmed backup copy leave it: data,
            /// `ref_count` 1, a stamp, `backed_up_in` and `backup_cleared_ms`.
            async fn stored_and_confirmed(tv: &TestValkey, hash_byte: u8) {
                pre_store(&store(tv), hash_byte).await;
                let _: () = tv
                    .client()
                    .pool
                    .hset(
                        tv.key_schema().ext_meta(&hex_of(hash_byte)),
                        [
                            (BACKED_UP_IN_FIELD, "0123456789abcdef:1"),
                            (BACKUP_CLEARED_MS_FIELD, "7"),
                        ],
                    )
                    .await
                    .expect("seeding the backup fields must succeed");
            }

            /// The recount sets each hash's count to exactly its object
            /// references plus every owed release, wherever the release
            /// waits: `pending`, `inflight`, a retirement carrier (a HASH) and
            /// an older build's SET carrier. It raises an existing count,
            /// creates a missing meta with `ref_count` alone and no data, and
            /// writes no field but `ref_count`: the stamp and both backup
            /// fields stay as they were. Its report counts each outcome and
            /// the largest object id it walked.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_sets_each_count_to_its_references_plus_every_owed_release() {
                let tv = TestValkey::setup().await;
                let (h, k) = (hex_of(0xA1), hex_of(0xA2));
                seed_object_referencing(&tv, "960101", &h).await;
                seed_object_referencing(&tv, "960102", &h).await;
                seed_object_referencing(&tv, "960102", &k).await;
                let stages = [
                    OwedStage::Pending,
                    OwedStage::Inflight,
                    OwedStage::RetireCarrier,
                    OwedStage::LegacyRetireCarrier,
                ];
                for (i, stage) in stages.into_iter().enumerate() {
                    owe(&tv, stage, hash_of(0xA1), 960_201 + i as u64).await;
                }
                stored_and_confirmed(&tv, 0xA1).await;
                let before = meta(&tv, &h).await;

                let report = recounted(&tv).await;

                assert_eq!(
                    ref_count(&tv, &h).await,
                    Some(6),
                    "h: two referencing objects plus one release in each of the four owed places"
                );
                let after = meta(&tv, &h).await;
                for field in [
                    LAST_STORE_MS_FIELD,
                    BACKED_UP_IN_FIELD,
                    BACKUP_CLEARED_MS_FIELD,
                ] {
                    assert_eq!(
                        after.get(field),
                        before.get(field),
                        "h: the recount writes `ref_count` only, never `{field}`"
                    );
                }
                assert_eq!(
                    meta(&tv, &k).await,
                    HashMap::from([(REF_COUNT_FIELD.to_owned(), "1".to_owned())]),
                    "k had no meta: one is created with its count and nothing else"
                );
                assert!(
                    !data_exists(&tv, &k).await,
                    "the created meta has no data: the meta-without-data shape"
                );
                assert_eq!(
                    report,
                    RecountReport {
                        values: 2,
                        objects: 2,
                        created: 1,
                        raised: 1,
                        lowered: 0,
                        unchanged: 0,
                        repaired: 0,
                        owed_only: 0,
                        max_object_id: Some(960_102),
                    }
                );
            }

            /// Every object header counts (Q2), not only mapped ones: an
            /// object a layer mapping names, one no mapping names
            /// (`LiveUnmapped`, a delete's teardown not finished) and one an
            /// overwrite superseded each hold a reference, because each
            /// releases it once when the sweep retires it after the un-hold.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_counts_unmapped_and_superseded_objects_too() {
                let tv = TestValkey::setup().await;
                let keys = tv.key_schema();
                let h = hex_of(0xA3);
                let (mapped, unmapped, superseded) = ("960301", "960302", "960303");
                for oid in [mapped, unmapped, superseded] {
                    seed_object_referencing(&tv, oid, &h).await;
                }
                let _: () = tv
                    .client()
                    .pool
                    .hset(
                        keys.obj_header(mapped),
                        [("session_id", "s-mapped"), ("layer_name", "l.usda")],
                    )
                    .await
                    .expect("seeding the mapped header's identity must succeed");
                let _: () = tv
                    .client()
                    .pool
                    .set(
                        keys.layer_mapping("s-mapped", "l.usda"),
                        mapped,
                        None,
                        None,
                        false,
                    )
                    .await
                    .expect("seeding the mapping must succeed");
                let _: () = tv
                    .client()
                    .pool
                    .hset(keys.obj_header(superseded), [("superseded", "1")])
                    .await
                    .expect("flagging the header superseded must succeed");

                recounted(&tv).await;

                assert_eq!(
                    ref_count(&tv, &h).await,
                    Some(3),
                    "the mapped, the unmapped and the superseded object each count once"
                );
            }

            /// PD-1: an over-count is LOWERED to exactly references plus
            /// owed, never below what names the hash and never deleted.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_lowers_an_over_count_to_exactly_its_references_and_owed() {
                let tv = TestValkey::setup().await;
                let h = hex_of(0xA4);
                pre_store(&store(&tv), 0xA4).await;
                set_ref_count(&tv, &h, 9).await;
                seed_object_referencing(&tv, "960401", &h).await;
                owe(&tv, OwedStage::Pending, hash_of(0xA4), 960_402).await;

                let report = recounted(&tv).await;

                assert_eq!(
                    ref_count(&tv, &h).await,
                    Some(2),
                    "one object plus one pending release"
                );
                assert!(data_exists(&tv, &h).await, "the value itself is untouched");
                assert_eq!((report.lowered, report.raised), (1, 0));
            }

            /// ADD 10: a meta nothing names — no object, no owed release —
            /// keeps its count and its data: the recount never deletes a
            /// meta, and leaves an unnamed one to the periodic pass. A named
            /// hash beside it is set, so the recount ran.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_leaves_a_meta_nothing_names_as_it_is() {
                let tv = TestValkey::setup().await;
                let (unnamed, named) = (hex_of(0xA5), hex_of(0xA6));
                pre_store(&store(&tv), 0xA5).await;
                set_ref_count(&tv, &unnamed, 4).await;
                let before = meta(&tv, &unnamed).await;
                seed_object_referencing(&tv, "960501", &named).await;

                let report = recounted(&tv).await;

                assert_eq!(
                    meta(&tv, &unnamed).await,
                    before,
                    "the unnamed meta is not touched"
                );
                assert!(data_exists(&tv, &unnamed).await);
                assert_eq!(ref_count(&tv, &named).await, Some(1), "control");
                assert_eq!(report.values, 1, "only the named hash is a target");
            }

            /// A hash only an owed release names gets a meta at the owed
            /// count, with no data, so the GC's decrement finds a meta and
            /// drains it to 0, deleting it, rather than missing it.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_gives_a_hash_only_a_release_names_a_meta_the_gc_drains() {
                let tv = TestValkey::setup().await;
                let h = hex_of(0xA7);
                owe(&tv, OwedStage::Pending, hash_of(0xA7), 960_601).await;

                let report = recounted(&tv).await;

                assert_eq!(ref_count(&tv, &h).await, Some(1));
                assert!(!data_exists(&tv, &h).await);
                assert_eq!((report.owed_only, report.created), (1, 1));

                let stats = run_gc(&store(&tv)).await;
                assert_eq!(stats.deleted, 1, "the GC drains the release to 0");
                assert!(
                    meta(&tv, &h).await.is_empty(),
                    "and deletes the meta it reached"
                );
            }

            /// A meta whose `ref_count` is not a decimal integer, or absent,
            /// is rewritten to the target (`repaired`).
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_rewrites_an_unreadable_or_absent_count() {
                let tv = TestValkey::setup().await;
                let (garbled, absent) = (hex_of(0xA8), hex_of(0xA9));
                for (byte, hex) in [(0xA8, &garbled), (0xA9, &absent)] {
                    pre_store(&store(&tv), byte).await;
                    seed_object_referencing(&tv, &format!("9607{byte:02x}"), hex).await;
                }
                let _: () = tv
                    .client()
                    .pool
                    .hset(
                        tv.key_schema().ext_meta(&garbled),
                        [(REF_COUNT_FIELD, "abc")],
                    )
                    .await
                    .expect("garbling the count must succeed");
                let _: u64 = tv
                    .client()
                    .pool
                    .hdel(tv.key_schema().ext_meta(&absent), REF_COUNT_FIELD)
                    .await
                    .expect("removing the count must succeed");

                let report = recounted(&tv).await;

                assert_eq!(ref_count(&tv, &garbled).await, Some(1));
                assert_eq!(ref_count(&tv, &absent).await, Some(1));
                assert_eq!(report.repaired, 2);
            }

            /// A meta key of the wrong type is a corrupt record (PD-4): the
            /// recount fails naming the key, and makes no receipt.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_fails_corrupt_on_a_meta_of_the_wrong_type() {
                let tv = TestValkey::setup().await;
                let h = hex_of(0xAA);
                let meta_key = tv.key_schema().ext_meta(&h);
                let _: () = tv
                    .client()
                    .pool
                    .set(&meta_key, "not-a-hash", None, None, false)
                    .await
                    .expect("staging a string meta must succeed");
                seed_object_referencing(&tv, "960801", &h).await;

                match recount(&tv).await {
                    Err(RecountError::Corrupt { key, .. }) => assert_eq!(key, meta_key),
                    other => panic!("a wrong-type meta is a corrupt record, got {other:?}"),
                }
            }

            /// Which reference record [`the_recount_fails_corrupt_on_an_undecodable_reference_record_and_sets_nothing`]
            /// garbles.
            #[derive(Clone, Copy, Debug)]
            enum Garbled {
                /// An `obj:ext` field that is not a hash.
                IndexField,
                /// A value row's `hash` column that is not a hash.
                RowHash,
                /// A `{gc}:pending` member that does not decode.
                PendingMember,
                /// A `{gc}:inflight` member that does not decode.
                InflightMember,
                /// A member of this build's retirement carrier (a HASH) that
                /// does not decode.
                HashCarrierMember,
                /// A member of an older build's retirement carrier (a SET)
                /// that does not decode.
                SetCarrierMember,
            }

            /// A reference record the walks cannot decode — an `obj:ext`
            /// field, a row's `hash` column, an owed member wherever it waits
            /// (`pending`, `inflight`, either carrier type) — fails the
            /// recount as corrupt (PD-4, exit 3 in the Job) before any count
            /// is set: a hash another object names cleanly gets no meta.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_fails_corrupt_on_an_undecodable_reference_record_and_sets_nothing()
            {
                for garbled in [
                    Garbled::IndexField,
                    Garbled::RowHash,
                    Garbled::PendingMember,
                    Garbled::InflightMember,
                    Garbled::HashCarrierMember,
                    Garbled::SetCarrierMember,
                ] {
                    let tv = TestValkey::setup().await;
                    let clean = hex_of(0xAB);
                    seed_object_referencing(&tv, "960901", &clean).await;
                    const OID: &str = "960902";
                    seed_live_header(&tv, OID).await;
                    match garbled {
                        Garbled::IndexField => seed_index(&tv, OID, &[("not-a-hash", "1")]).await,
                        Garbled::RowHash => {
                            seed_row(
                                &tv,
                                OID,
                                Family::Field,
                                "a",
                                &[("hash", "zz"), ("size", "1")],
                            )
                            .await;
                        }
                        Garbled::PendingMember => {
                            let _: u64 = tv
                                .client()
                                .pool
                                .sadd(tv.key_schema().ext_pending(), "not-a-release")
                                .await
                                .expect("seeding a garbled pending member must succeed");
                        }
                        Garbled::InflightMember => {
                            let _: i64 = tv
                                .client()
                                .pool
                                .zadd(
                                    tv.key_schema().ext_inflight(),
                                    None,
                                    None,
                                    false,
                                    false,
                                    (1.0, "not-a-release"),
                                )
                                .await
                                .expect("seeding a garbled inflight member must succeed");
                        }
                        Garbled::HashCarrierMember => {
                            let _: () = tv
                                .client()
                                .pool
                                .hset(
                                    tv.key_schema().obj_ext_retire_carrier(OID),
                                    [("not-a-release", "1")],
                                )
                                .await
                                .expect("seeding a garbled carrier member must succeed");
                        }
                        Garbled::SetCarrierMember => {
                            let _: u64 = tv
                                .client()
                                .pool
                                .sadd(tv.key_schema().obj_ext_retire_carrier(OID), "not-a-release")
                                .await
                                .expect("seeding a garbled legacy carrier member must succeed");
                        }
                    }

                    let result = recount(&tv).await;

                    assert!(
                        matches!(result, Err(RecountError::Corrupt { .. })),
                        "{garbled:?}: an undecodable record is corrupt, got {result:?}"
                    );
                    assert!(
                        meta(&tv, &clean).await.is_empty(),
                        "{garbled:?}: no count is set from a walk that failed"
                    );
                }
            }

            /// Which reference key
            /// [`the_recount_fails_corrupt_on_a_reference_key_of_the_wrong_type_and_sets_nothing`]
            /// stages as a STRING.
            #[derive(Clone, Copy, Debug)]
            enum WrongType {
                /// A retirement carrier: neither a HASH nor a SET.
                Carrier,
                /// An object's `obj:ext` index.
                ExtIndex,
                /// An object's `obj:flds` row index.
                FieldsIndex,
            }

            /// A reference key of the wrong type — a retirement carrier that
            /// is neither a HASH nor a SET, an `obj:ext` or an `obj:flds` that
            /// is a STRING — fails the recount as corrupt (PD-4, exit 3 in the
            /// Job), never as a store fault, and before any count is set.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_fails_corrupt_on_a_reference_key_of_the_wrong_type_and_sets_nothing(
            ) {
                for wrong in [
                    WrongType::Carrier,
                    WrongType::ExtIndex,
                    WrongType::FieldsIndex,
                ] {
                    let tv = TestValkey::setup().await;
                    let keys = tv.key_schema();
                    let clean = hex_of(0xAF);
                    seed_object_referencing(&tv, "961401", &clean).await;
                    const OID: &str = "961402";
                    seed_live_header(&tv, OID).await;
                    let key = match wrong {
                        WrongType::Carrier => keys.obj_ext_retire_carrier(OID),
                        WrongType::ExtIndex => keys.obj_ext_index(OID),
                        WrongType::FieldsIndex => keys.obj_fields_index(OID),
                    };
                    let _: () = tv
                        .client()
                        .pool
                        .set(&key, "not-a-collection", None, None, false)
                        .await
                        .expect("staging a STRING must succeed");

                    let result = recount(&tv).await;

                    assert!(
                        matches!(result, Err(RecountError::Corrupt { .. })),
                        "{wrong:?}: a key of the wrong type is corrupt, got {result:?}"
                    );
                    assert!(
                        meta(&tv, &clean).await.is_empty(),
                        "{wrong:?}: no count is set from a walk that failed"
                    );
                }
            }

            /// The recount's failure reading one object's references names the
            /// object, and the key when the failing read names one, ahead of
            /// what it could not read, so the Job's exit-3 line says what
            /// support must look at: an `obj:ext` field that is not a hash
            /// names `obj:ext`; a row `hash` column that is not a hash, read
            /// in a row pipeline that answers for every key of its chunk at
            /// once, names the object alone. (A wrong-typed index's step:
            /// [`the_recount_names_each_value_index_whose_step_meets_the_wrong_type`].)
            ///
            /// Fail-on-old (in-place perturbation of the object walk, red then
            /// green once restored): with the walk's locator passing every
            /// error through bare in both modes, the key reads `not-a-hash`
            /// with no object.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_names_the_object_and_its_key_when_a_reference_read_fails() {
                const OID: &str = "961501";
                let index_key = |tv: &TestValkey| tv.key_schema().obj_ext_index(OID);

                let tv = TestValkey::setup().await;
                seed_live_header(&tv, OID).await;
                seed_index(&tv, OID, &[("not-a-hash", "1")]).await;
                match recount(&tv).await {
                    Err(RecountError::Corrupt { what, key }) => {
                        assert_eq!(what, WHAT_EXT_INDEX_FIELD);
                        assert_eq!(
                            key,
                            format!("{} (object {OID}): not-a-hash", index_key(&tv))
                        );
                    }
                    other => panic!("a garbled obj:ext field is corrupt, got {other:?}"),
                }

                let tv = TestValkey::setup().await;
                seed_live_header(&tv, OID).await;
                seed_row(
                    &tv,
                    OID,
                    Family::Field,
                    "a",
                    &[("hash", "zz"), ("size", "1")],
                )
                .await;
                match recount(&tv).await {
                    Err(RecountError::Corrupt { key, .. }) => {
                        assert_eq!(key, format!("object {OID}: zz"));
                    }
                    other => panic!("a garbled row hash is corrupt, got {other:?}"),
                }
            }

            /// Each value index the recount walks, staged as a STRING, fails
            /// the recount corrupt at its own read's label — its rank pages,
            /// `ZRANGE obj:flds` and `ZRANGE obj:tss`, and its counted
            /// `HSCAN obj:ext` — and names that key and the object, then the
            /// store's `WRONGTYPE`. Each index is its own case on its own
            /// store, and every case is checked before the test fails, so a
            /// regression names every index it reaches.
            ///
            /// Fail-on-old (in-place perturbation of `in_object`, the three
            /// labels no longer mapped to their keys; re-run for the rank
            /// pages' and the counted walk's labels): all three cases fail,
            /// each naming the object alone.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_names_each_value_index_whose_step_meets_the_wrong_type() {
                const OID: &str = "961502";
                /// An index's key in the schema.
                type KeyOf = fn(&KeySchema) -> String;
                let cases: [(&str, KeyOf); 3] = [
                    (OP_LIST_FIELDS, |keys| keys.obj_fields_index(OID)),
                    (OP_LIST_TIME_SAMPLES, |keys| {
                        keys.obj_time_samples_index(OID)
                    }),
                    (OP_SCAN_EXT_INDEX, |keys| keys.obj_ext_index(OID)),
                ];
                let mut failures: Vec<String> = Vec::new();
                for (op, key_of) in cases {
                    let tv = TestValkey::setup().await;
                    seed_live_header(&tv, OID).await;
                    let index = key_of(tv.key_schema());
                    let _: () = tv
                        .client()
                        .pool
                        .set(&index, "not-a-collection", None, None, false)
                        .await
                        .expect("staging a STRING index must succeed");
                    match recount(&tv).await {
                        Err(RecountError::Corrupt { what, key }) => {
                            let located = format!("{index} (object {OID}): ");
                            if what != format!("a key of the wrong type ({op})")
                                || !key.starts_with(&located)
                                || !key[located.len()..].starts_with("WRONGTYPE")
                            {
                                failures.push(format!(
                                    "{op}: want the step's label and `{located}WRONGTYPE…`, got \
                                     what {what:?}, key {key:?}"
                                ));
                            }
                        }
                        other => failures.push(format!("{op}: want Corrupt, got {other:?}")),
                    }
                }
                assert!(failures.is_empty(), "{}", failures.join("\n"));
            }

            /// An `HSCAN obj:ext` step answering an odd number of elements,
            /// which cannot be paired (`retirement::scan_step_firsts`), fails
            /// the recount as a store fault (exit 1 in the Job), located at
            /// the index it stepped, never as a corrupt record: the reply is
            /// the server's protocol, not stored data. Pure: the walk's error
            /// is built by hand from the pairing's own error.
            #[test]
            fn an_index_step_that_cannot_be_paired_fails_the_recount_as_a_store_fault() {
                let keys = KeySchema::new("le:");
                let index = keys.obj_ext_index("42");
                let unpaired = crate::storage::retirement::scan_step_firsts(
                    "HSCAN",
                    &index,
                    vec!["m".to_owned(), "1".to_owned(), "n".to_owned()],
                )
                .expect_err("three elements cannot be paired");
                let walk_error = PassError::Valkey {
                    op: OP_SCAN_EXT_INDEX,
                    source: unpaired,
                };
                match RecountError::<()>::from_walk(in_object(&keys, "42", walk_error)) {
                    RecountError::Store(why) => {
                        assert!(
                            why.starts_with(&format!("{OP_SCAN_EXT_INDEX} failed: ")),
                            "{why}"
                        );
                        assert!(why.contains("answered 3 elements"), "{why}");
                        assert!(why.ends_with(&format!("({index} (object 42))")), "{why}");
                    }
                    other => panic!("an unpaired step is a store fault, got {other:?}"),
                }
            }

            /// A store fault, not a corrupt record, located in an object: the
            /// recount's reading of the walk's error is `Store` (exit 1 in the
            /// Job), its text the fault's, led by nothing and ended by the
            /// object and key. A `WRONGTYPE` reply at the same read is
            /// `Corrupt`. Pure: the walk's errors are built by hand.
            #[test]
            fn a_located_store_fault_is_a_store_error_and_a_located_wrongtype_is_corrupt() {
                let keys = KeySchema::new("le:");
                let fault = PassError::Valkey {
                    op: OP_LIST_FIELDS,
                    source: fred::error::Error::new(fred::error::ErrorKind::IO, "connection reset"),
                };
                match RecountError::<()>::from_walk(in_object(&keys, "42", fault)) {
                    RecountError::Store(why) => assert_eq!(
                        why,
                        format!(
                            "{OP_LIST_FIELDS} failed: IO Error: connection reset ({} (object 42))",
                            keys.obj_fields_index("42")
                        )
                    ),
                    other => panic!("a located IO fault is a store fault, got {other:?}"),
                }

                let wrongtype = PassError::Valkey {
                    op: OP_LIST_FIELDS,
                    source: fred::error::Error::new(
                        fred::error::ErrorKind::InvalidArgument,
                        "WRONGTYPE Operation against a key holding the wrong kind of value",
                    ),
                };
                assert!(matches!(
                    RecountError::<()>::from_walk(in_object(&keys, "42", wrongtype)),
                    RecountError::Corrupt { .. }
                ));
            }

            /// An `obj:ext` field whose stored count is `0` or not a decimal,
            /// with no row naming its hash, still names the hash: the
            /// recount counts it once for its object (the periodic pass
            /// would call it drift and rewrite the index; the recount writes
            /// no index, which stays as it is).
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_counts_an_index_field_whose_count_is_zero_or_not_decimal() {
                let tv = TestValkey::setup().await;
                let (zero, garbled) = ("961601", "961602");
                let (hz, hg) = (hex_of(0xC1), hex_of(0xC2));
                seed_live_header(&tv, zero).await;
                seed_index(&tv, zero, &[(hz.as_str(), "0")]).await;
                seed_live_header(&tv, garbled).await;
                seed_index(&tv, garbled, &[(hg.as_str(), "x")]).await;

                let report = recounted(&tv).await;

                assert_eq!(
                    (ref_count(&tv, &hz).await, ref_count(&tv, &hg).await),
                    (Some(1), Some(1)),
                    "each field names its hash, whatever count it stores"
                );
                assert_eq!(report.values, 2);
                assert_eq!(
                    (
                        index_sorted(&tv, zero).await,
                        index_sorted(&tv, garbled).await
                    ),
                    (
                        vec![(hz.clone(), "0".to_owned())],
                        vec![(hg.clone(), "x".to_owned())]
                    ),
                    "the index is left as it is"
                );
            }

            /// The authoritative walk writes nothing: a legacy `-0.0`
            /// time-sample row beside its `+0.0` twin, and an `obj:ext` that
            /// disagrees with the rows, are left exactly as they are (the
            /// periodic pass's merge and audit would rewrite both, pinned by
            /// `signed_zero_merge` and the drift tests above). Each hash the
            /// object's rows ∪ index names is counted once for the object.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_leaves_a_negative_zero_row_and_a_drifted_index_as_they_are() {
                const OID: &str = "961001";
                let tv = TestValkey::setup().await;
                let keys = tv.key_schema();
                let (plus, minus, field) = (hex_of(0xAC), hex_of(0xAD), hex_of(0xAE));
                seed_live_header(&tv, OID).await;
                seed_row(
                    &tv,
                    OID,
                    Family::TimeSample,
                    "0",
                    &[("hash", plus.as_str()), ("size", "1")],
                )
                .await;
                seed_row(
                    &tv,
                    OID,
                    Family::TimeSample,
                    "-0",
                    &[("hash", minus.as_str()), ("size", "1")],
                )
                .await;
                seed_row(
                    &tv,
                    OID,
                    Family::Field,
                    "a",
                    &[("hash", field.as_str()), ("size", "1")],
                )
                .await;
                // `minus` has no field and `field` is over-counted: drift
                // both ways.
                seed_index(&tv, OID, &[(plus.as_str(), "1"), (field.as_str(), "3")]).await;
                let tss = |tv: &TestValkey| {
                    let (pool, key) = (tv.client().pool.clone(), keys.obj_time_samples_index(OID));
                    async move {
                        let mut members: Vec<String> = pool
                            .zrange(key, 0, -1, None, false, None, false)
                            .await
                            .expect("ZRANGE obj:tss must succeed");
                        members.sort();
                        members
                    }
                };
                let header = |tv: &TestValkey| {
                    let (pool, key) = (tv.client().pool.clone(), keys.obj_header(OID));
                    async move {
                        pool.hgetall::<HashMap<String, String>, _>(key)
                            .await
                            .expect("HGETALL obj:hdr must succeed")
                    }
                };
                let (tss_before, index_before, header_before) = (
                    tss(&tv).await,
                    index_sorted(&tv, OID).await,
                    header(&tv).await,
                );
                assert_eq!(tss_before.len(), 2, "fixture: the -0.0 and +0.0 rows");

                recounted(&tv).await;

                assert_eq!(tss(&tv).await, tss_before, "no merge: both rows stay");
                assert_eq!(
                    index_sorted(&tv, OID).await,
                    index_before,
                    "no audit: the drifted index stays"
                );
                assert_eq!(header(&tv).await, header_before, "no repair stamp");
                for hex in [&plus, &minus, &field] {
                    assert_eq!(
                        ref_count(&tv, hex).await,
                        Some(1),
                        "{hex}: counted once for the one object, whichever side names it"
                    );
                }
            }

            /// **PD-1 condition 2 (a), index drift.** Object A's live row
            /// names h while A's `obj:ext` omits h; object B names h in both.
            /// h's meta is over-counted at 7. The recount lowers it to
            /// exactly rows ∪ `obj:ext` — 2, one per object — never past A's
            /// row. Fail-on-old: with the walk's row hashes left out of the
            /// union, the count reads 1, lowered past A's live reference.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_counts_a_live_row_its_objects_index_omits() {
                const DRIFTED: &str = "961101";
                let tv = TestValkey::setup().await;
                let (h, other) = (hex_of(0xB1), hex_of(0xB2));
                pre_store(&store(&tv), 0xB1).await;
                set_ref_count(&tv, &h, 7).await;
                seed_live_header(&tv, DRIFTED).await;
                seed_row(
                    &tv,
                    DRIFTED,
                    Family::Field,
                    "a",
                    &[("hash", h.as_str()), ("size", "1")],
                )
                .await;
                seed_row(
                    &tv,
                    DRIFTED,
                    Family::Field,
                    "b",
                    &[("hash", other.as_str()), ("size", "1")],
                )
                .await;
                seed_index(&tv, DRIFTED, &[(other.as_str(), "1")]).await;
                seed_object_referencing(&tv, "961102", &h).await;

                recounted(&tv).await;

                assert_eq!(
                    ref_count(&tv, &h).await,
                    Some(2),
                    "lowered from 7 to rows ∪ obj:ext: A's row and B's reference"
                );
            }

            /// **PD-1 condition 2 (b), skipped object.** Object A is one the
            /// periodic pass holds: its `obj:ext` disagrees with its rows and
            /// one row's `hash` column is UPPERCASE, which the index recount
            /// reports malformed, so the periodic pass holds h from every
            /// surplus drain (`a_malformed_row_holds_the_hash_whatever_the_recount_answers`).
            /// Object B names h cleanly. h's meta is over-counted at 9 with an
            /// old stamp. The periodic pass holds h (the precondition, with
            /// the stored index `"4"` it withholds rather than rewrites); the
            /// recount then counts A, setting h to exactly 2. Fail-on-old:
            /// with the authoritative walk skipping an object whose index
            /// disagrees with its rows, h reads 1, lowered past A.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_counts_an_object_the_periodic_pass_holds_for_index_drift() {
                const REPLICA: &str = "tv3-recount-held-object";
                const HELD: &str = "961201";
                let tv = TestValkey::setup().await;
                let metrics = Arc::new(metrics_scrape_only(REPLICA));
                let h = hex_of(0xB3);
                pre_store(&store(&tv), 0xB3).await;
                set_ref_count(&tv, &h, 9).await;
                set_stamp(&tv, &h, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
                seed_live_header(&tv, HELD).await;
                seed_row(
                    &tv,
                    HELD,
                    Family::Field,
                    "a",
                    &[("hash", h.as_str()), ("size", "1")],
                )
                .await;
                seed_row(
                    &tv,
                    HELD,
                    Family::Field,
                    "b",
                    &[("hash", h.to_uppercase().as_str()), ("size", "1")],
                )
                .await;
                seed_index(&tv, HELD, &[(h.as_str(), "4")]).await;
                seed_object_referencing(&tv, "961202", &h).await;

                one_pass(&tv, &metrics, REPLICA).await;
                assert_eq!(
                    outcome(&metrics, EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT),
                    1,
                    "precondition: the periodic pass holds h for A's drift"
                );
                assert_eq!(ref_count(&tv, &h).await, Some(9), "and lowers nothing");

                recounted(&tv).await;

                assert_eq!(
                    ref_count(&tv, &h).await,
                    Some(2),
                    "the recount counts A (once) and B: never lowered past the held object"
                );
            }

            /// **"Fails if the guards are left on."** The periodic pass would
            /// leave a surplus on a meta stamped moments ago (the
            /// `last_store_ms` gate) and would need two observations a grace
            /// apart even on an old one. The recount runs neither: one call
            /// sets the freshly stamped count of 5 to exactly 1 and registers
            /// no candidate. Fail-on-old: with the count sets replaced by one
            /// periodic pass (the candidate path), the count stays 5.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_sets_a_freshly_stamped_surplus_at_once_with_no_guard() {
                let tv = TestValkey::setup().await;
                let h = hex_of(0xB4);
                pre_store(&store(&tv), 0xB4).await;
                set_ref_count(&tv, &h, 5).await;
                set_stamp(&tv, &h, now_ms(&tv).await).await;
                seed_object_referencing(&tv, "961301", &h).await;

                recounted(&tv).await;

                assert_eq!(
                    ref_count(&tv, &h).await,
                    Some(1),
                    "set exactly in one call, its fresh stamp notwithstanding"
                );
                assert!(candidates(&tv).await.is_empty(), "no candidate path");
            }

            /// **R-recon.** A periodic pass right after the recount finds
            /// nothing to do: no candidate, surplus 0 and deficit 0, for a
            /// recount-created meta without data (no stamp) and a lowered
            /// over-count alike. The control, the same keyspace before the
            /// recount, registers the over-count's surplus.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn a_periodic_pass_right_after_the_recount_registers_no_candidate() {
                const REPLICA: &str = "tv3-recount-then-pass";
                let tv = TestValkey::setup().await;
                let metrics = Arc::new(metrics_scrape_only(REPLICA));
                let (over, created) = (hex_of(0xB5), hex_of(0xB6));
                pre_store(&store(&tv), 0xB5).await;
                set_ref_count(&tv, &over, 3).await;
                set_stamp(&tv, &over, now_ms(&tv).await - OLD_STAMP_LAG_MS).await;
                seed_object_referencing(&tv, "961401", &over).await;
                seed_object_referencing(&tv, "961402", &created).await;

                one_pass(&tv, &metrics, REPLICA).await;
                assert_eq!(
                    candidates(&tv).await,
                    vec![format!("{over}:2")],
                    "control: before the recount the over-count is a surplus"
                );
                let _: i64 = tv
                    .client()
                    .pool
                    .del(tv.key_schema().ext_recon_candidates())
                    .await
                    .expect("clearing the candidates must succeed");

                recounted(&tv).await;
                one_pass(&tv, &metrics, REPLICA).await;

                assert!(candidates(&tv).await.is_empty());
                assert_eq!(
                    gauges(&metrics, REPLICA),
                    Some((2.0, 2.0, 0.0, 0.0)),
                    "two metas, two referenced hashes, no surplus, no deficit"
                );
                assert!(
                    !data_exists(&tv, &created).await,
                    "fixture: a meta without data"
                );
            }

            // ══ b.f3t — the recount's completeness proofs; b.gns — the ═══════
            // ══ periodic pass's serving-process guard ═════════════════════════
            //
            // The recount reads no serving-process witness (`storage::scan_guard`,
            // *The DR Jobs' walks*). Each of its walks is shown complete on the
            // held store by its own proof — the object and carrier walks by an
            // agreeing second walk, `obj:flds`, `obj:tss` and `inflight` by
            // `ZCARD`-checked rank pages, `obj:ext` and `pending` by
            // `HLEN` / `SCARD`-checked cursor walks — and a proof that does not
            // hold fails the recount (`RecountError::Incomplete`, exit 1 in the
            // Job) before any count is set. A skip is forged through the walk
            // seam's `DropMember` (`scan_guard::arm_walk_fault_for_test`), keyed
            // by the walk's key or pattern, which carries the test's prefix; a
            // header the object walk returns twice through its `RepeatMember`;
            // a member added under a rank walk, or repeated by a counted one,
            // by a wrapper around the recount's own page or step taker. A placeholder
            // run_id, or a change of one, no longer stops the recount.
            //
            // The periodic pass keeps the serving-process guard on its owed
            // walks: the owed readers' test below forges a change of serving
            // process through the same seam. The cluster-gated arms forge a
            // pinned node's `CLUSTER NODES` reply on a real three-shard cluster,
            // which the recount's keyspace walks still check.
            //
            // Fail-on-old, by in-place perturbation (red, then green once
            // restored): with `scan_guard::keyspace_walks_agree` answering `Ok`
            // whatever the two sets, the object and carrier drops complete and
            // set their counts; with `scan_guard::check_count` answering `Ok`
            // whatever the counts, the index, `obj:ext`, `pending` and
            // `inflight` drops and the added members complete (those two run
            // together, each case caught only by its own walk's check); with
            // `KeyspaceWalk::start_job` beginning a guarded walk again (its
            // `INFO server` witness put back), the unusable and changing run_id
            // cases fail. The repeat and letter-case pins name their own.
            mod serving_process_guard {
                use super::*;

                use fred::interfaces::ClusterInterface;

                use crate::storage::scan_guard::{RoutingRefresh, SlotSet, WalkEnd};

                /// The fixture's object, which references [`Fixture::h`].
                const OID: &str = "960901";

                /// The object whose retirement carrier [`seed`] stages.
                const CARRIER_OID: u64 = 960_902;

                /// What [`seed`] staged: `h`, referenced by [`OID`] through a
                /// field row, a time-sample row and its `obj:ext`, and
                /// over-counted at 9, and `k`, which only owed releases name
                /// (a carrier, `pending` and `inflight`), with no meta. A
                /// recount that completes lowers `h` to 1 and creates `k` at
                /// 3, so every count reading as before shows none was set.
                struct Fixture {
                    h: String,
                    k: String,
                }

                async fn seed(tv: &TestValkey) -> Fixture {
                    let h = hex_of(0xB1);
                    pre_store(&store(tv), 0xB1).await;
                    set_ref_count(tv, &h, 9).await;
                    seed_object_referencing(tv, OID, &h).await;
                    seed_row(
                        tv,
                        OID,
                        Family::TimeSample,
                        "0.5",
                        &[("hash", h.as_str()), ("size", "1")],
                    )
                    .await;
                    for (stage, oid) in [
                        (OwedStage::RetireCarrier, CARRIER_OID),
                        (OwedStage::Pending, 960_903),
                        (OwedStage::Inflight, 960_904),
                    ] {
                        owe(tv, stage, hash_of(0xB2), oid).await;
                    }
                    Fixture { h, k: hex_of(0xB2) }
                }

                /// Both metas, every field.
                async fn metas(
                    tv: &TestValkey,
                    fixture: &Fixture,
                ) -> (HashMap<String, String>, HashMap<String, String>) {
                    (meta(tv, &fixture.h).await, meta(tv, &fixture.k).await)
                }

                /// Drop `key` from walk number `walk` of `pattern`, whichever
                /// step returns it: a keyspace walk of a shared store answers
                /// the test's key at a step no test can know in advance, so the
                /// seam's `DropMember` names the key on every step of the walk
                /// (`step: None`). A drop that never fires leaves the walks
                /// agreeing, and the test that expects them to disagree fails.
                fn drop_key(pattern: &str, walk: usize, key: &str) -> ArmedWalkFault {
                    arm_walk_fault_for_test(
                        pattern,
                        WalkFault::DropMember {
                            walk,
                            step: None,
                            member: Some(key.to_owned()),
                        },
                    )
                }

                /// Drop the first member of the first reply of a walk of the
                /// one key `target`, which holds one member in the fixture.
                fn drop_first(target: &str) -> ArmedWalkFault {
                    arm_walk_fault_for_test(
                        target,
                        WalkFault::DropMember {
                            walk: 0,
                            step: Some(0),
                            member: None,
                        },
                    )
                }

                /// One skip the recount's proofs must catch: what is skipped,
                /// the walk's key or pattern, for a keyspace walk the walk's
                /// number and the key it skips (else the first member of the
                /// key's first reply, [`drop_first`]), and what the
                /// `RecountError::Incomplete` text must name.
                struct Skip {
                    what: &'static str,
                    target: String,
                    key: Option<(usize, String)>,
                    names: String,
                }

                impl Skip {
                    /// Arm the skip; it holds while the guard is held.
                    fn arm(&self) -> ArmedWalkFault {
                        match &self.key {
                            Some((walk, key)) => drop_key(&self.target, *walk, key),
                            None => drop_first(&self.target),
                        }
                    }
                }

                /// Every walk of the recount, each with one member skipped:
                /// [`OID`]'s header from the first object walk (so the object
                /// is never counted) and from the second (so the counted set
                /// has an object the second walk did not return); its
                /// `obj:flds`, `obj:tss` and `obj:ext`; [`CARRIER_OID`]'s
                /// carrier from each carrier walk; `pending` and `inflight`.
                fn skips(keys: &KeySchema) -> Vec<Skip> {
                    let headers = keys.obj_header_scan_pattern();
                    let carriers = format!("{}{{*}}:obj:extret", keys.prefix());
                    let header = keys.obj_header(OID);
                    let carrier = keys.obj_ext_retire_carrier(&CARRIER_OID.to_string());
                    let one_of = |walk: &str, key: &str, card: &str| {
                        format!(
                            "the {walk} walk of {key} returned 0 members, and {card} read 1 at \
                             its start and 1 at its end"
                        )
                    };
                    let (flds, tss, ext) = (
                        keys.obj_fields_index(OID),
                        keys.obj_time_samples_index(OID),
                        keys.obj_ext_index(OID),
                    );
                    let (pending, inflight) = (keys.ext_pending(), keys.ext_inflight());
                    let disagree = |pattern: &str| {
                        format!("two SCAN walks of {pattern} returned different keys")
                    };
                    vec![
                        Skip {
                            what: "the first object walk",
                            names: disagree(&headers),
                            target: headers.clone(),
                            key: Some((0, header.clone())),
                        },
                        Skip {
                            what: "the second object walk",
                            names: disagree(&headers),
                            target: headers,
                            key: Some((1, header)),
                        },
                        Skip {
                            what: "obj:flds",
                            names: one_of("ZRANGE", &flds, "ZCARD"),
                            target: flds,
                            key: None,
                        },
                        Skip {
                            what: "obj:tss",
                            names: one_of("ZRANGE", &tss, "ZCARD"),
                            target: tss,
                            key: None,
                        },
                        Skip {
                            what: "obj:ext",
                            names: one_of("HSCAN", &ext, "HLEN"),
                            target: ext,
                            key: None,
                        },
                        Skip {
                            what: "the first carrier walk",
                            names: disagree(&carriers),
                            target: carriers.clone(),
                            key: Some((0, carrier.clone())),
                        },
                        Skip {
                            what: "the second carrier walk",
                            names: disagree(&carriers),
                            target: carriers,
                            key: Some((1, carrier)),
                        },
                        Skip {
                            what: "pending",
                            names: one_of("SSCAN", &pending, "SCARD"),
                            target: pending,
                            key: None,
                        },
                        Skip {
                            what: "inflight",
                            names: one_of("ZRANGE", &inflight, "ZCARD"),
                            target: inflight,
                            key: None,
                        },
                    ]
                }

                /// T1–T4 for the recount (b.f3t). One member skipped from
                /// each walk of the recount in turn ([`skips`]) — a header
                /// from either object walk, a member of each value index, a
                /// carrier from either carrier walk, a `pending` and an
                /// `inflight` release — fails the recount with
                /// `RecountError::Incomplete` (exit 1 in the Job) whose text
                /// names the walk and the counts or key sets it compared, and
                /// every meta reads exactly as before: no count was set. The
                /// control, with no seam armed, sets both counts.
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn a_recount_whose_walk_skips_a_member_fails_incomplete_and_sets_no_count() {
                    let cases = skips(&KeySchema::new("count-only:")).len();
                    let mut failures: Vec<String> = Vec::new();
                    for index in 0..cases {
                        let tv = TestValkey::setup().await;
                        let fixture = seed(&tv).await;
                        let before = metas(&tv, &fixture).await;
                        let skip = skips(tv.key_schema()).swap_remove(index);
                        let (what, names) = (skip.what, skip.names.as_str());

                        let armed = skip.arm();
                        let result = recount(&tv).await;
                        drop(armed);

                        match result {
                            Err(RecountError::Incomplete(why)) if why.contains(names) => {}
                            Err(RecountError::Incomplete(why)) => {
                                failures.push(format!("{what}: the text names {names:?}: {why}"))
                            }
                            other => failures.push(format!(
                                "{what}: a skipped member fails the recount as incomplete, got \
                                 {:?}",
                                other.map(|receipt| receipt.report().clone())
                            )),
                        }
                        if metas(&tv, &fixture).await != before {
                            failures.push(format!("{what}: a count was set"));
                        }
                    }
                    assert!(failures.is_empty(), "{}", failures.join("\n"));

                    let tv = TestValkey::setup().await;
                    let fixture = seed(&tv).await;
                    recounted(&tv).await;
                    assert_eq!(
                        (
                            ref_count(&tv, &fixture.h).await,
                            ref_count(&tv, &fixture.k).await
                        ),
                        (Some(1), Some(3)),
                        "control: unarmed, the recount lowers h and creates k"
                    );
                }

                /// Every key or pattern the recount walks.
                fn recount_targets(keys: &KeySchema) -> Vec<String> {
                    vec![
                        keys.obj_header_scan_pattern(),
                        keys.obj_fields_index(OID),
                        keys.obj_time_samples_index(OID),
                        keys.obj_ext_index(OID),
                        format!("{}{{*}}:obj:extret", keys.prefix()),
                        keys.ext_pending(),
                        keys.ext_inflight(),
                    ]
                }

                /// T6 for the recount (b.f3t). The recount reads no
                /// serving-process witness, so a store whose `INFO server`
                /// reports a placeholder run_id (`"0"`) or no `run_id` line,
                /// or whose run_id changes between two reads, no longer stops
                /// it: with each forged through the walk seam on every walk of
                /// the recount ([`recount_targets`]: a `RunId` fault on the
                /// first witness read, or a change of serving process at the
                /// second), the recount completes, lowering `h` to 1 and
                /// creating `k` at 3. A missing myself line still fails it
                /// (the cluster-gated test below).
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn a_recount_runs_on_a_store_whose_run_id_is_unusable_or_changes() {
                    type Forge = fn(&str) -> ArmedWalkFault;
                    let cases: [(&str, Forge); 3] = [
                        ("run_id \"0\"", |target| {
                            arm_walk_fault_for_test(
                                target,
                                WalkFault::RunId {
                                    read: 0,
                                    run_id: Some("0".to_owned()),
                                },
                            )
                        }),
                        ("no run_id line", |target| {
                            arm_walk_fault_for_test(
                                target,
                                WalkFault::RunId {
                                    read: 0,
                                    run_id: None,
                                },
                            )
                        }),
                        ("a change of run_id", forge_process_change),
                    ];
                    let mut failures: Vec<String> = Vec::new();
                    for (case, forge) in cases {
                        let tv = TestValkey::setup().await;
                        let fixture = seed(&tv).await;
                        let armed: Vec<ArmedWalkFault> = recount_targets(tv.key_schema())
                            .iter()
                            .map(|target| forge(target.as_str()))
                            .collect();
                        let result = recount(&tv).await;
                        drop(armed);

                        if let Err(e) = result {
                            failures.push(format!("{case}: the recount completes, got {e:?}"));
                            continue;
                        }
                        let counts = (
                            ref_count(&tv, &fixture.h).await,
                            ref_count(&tv, &fixture.k).await,
                        );
                        if counts != (Some(1), Some(3)) {
                            failures.push(format!(
                                "{case}: the recount lowers h to 1 and creates k at 3, got \
                                 {counts:?}"
                            ));
                        }
                    }
                    assert!(failures.is_empty(), "{}", failures.join("\n"));
                }

                /// T4's letter case (b.f3t). `obj:ext` holding one hash under
                /// two fields, its hex in lower and in upper case (the recount
                /// decodes either): `HLEN` counts two fields, and the recount
                /// de-duplicates `obj:ext` by each field's exact text, so its
                /// count check holds and the recount completes, counting the
                /// object's one reference to the hash once (`ref_count` 1).
                ///
                /// Fail-on-old (in-place perturbation, red then green once
                /// restored): with `RecountExtFields` keeping each field
                /// lower-cased, so de-duplicated by the hash it decodes to, the
                /// walk returns one distinct field where `HLEN` read two, and
                /// the recount fails incomplete.
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn a_recount_counts_a_hash_obj_ext_holds_in_two_letter_cases_once() {
                    const CASED: &str = "960905";
                    let tv = TestValkey::setup().await;
                    let hash = hex_of(0xBC);
                    pre_store(&store(&tv), 0xBC).await;
                    set_ref_count(&tv, &hash, 7).await;
                    seed_live_header(&tv, CASED).await;
                    let upper = hash.to_ascii_uppercase();
                    assert_ne!(upper, hash, "fixture: the hex has letters to case");
                    seed_index(&tv, CASED, &[(hash.as_str(), "1"), (upper.as_str(), "1")]).await;

                    recounted(&tv).await;

                    assert_eq!(
                        ref_count(&tv, &hash).await,
                        Some(1),
                        "one object references the hash once"
                    );
                }

                /// T2's repeat (b.f3t; PM 1). The recount's first object walk
                /// returning [`OID`]'s header twice — the walk seam's
                /// `RepeatMember`, a copy of the header right after it in the
                /// step that returns it, as a SCAN may return a key again —
                /// counts the object once: the recount completes, its report
                /// counts one object, and `h`, which the object references
                /// once, is set to 1 (`k`, owed only, to 3). The second walk,
                /// unarmed, returns the header once, so the two walks'
                /// de-duplicated sets agree.
                ///
                /// Fail-on-old (in-place perturbation, red then green once
                /// restored): with `walk_objects` no longer skipping an object
                /// it already counted (its `counted.insert` check), the report
                /// counts two objects and `h` is set to 2.
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn a_recount_counts_an_object_whose_header_its_walk_returns_twice_once() {
                    let tv = TestValkey::setup().await;
                    let fixture = seed(&tv).await;
                    let keys = tv.key_schema();

                    let armed = arm_walk_fault_for_test(
                        &keys.obj_header_scan_pattern(),
                        WalkFault::RepeatMember {
                            walk: 0,
                            step: None,
                            member: Some(keys.obj_header(OID)),
                        },
                    );
                    let report = recounted(&tv).await;
                    drop(armed);

                    assert_eq!(report.objects, 1, "the object is counted once: {report:?}");
                    assert_eq!(
                        (
                            ref_count(&tv, &fixture.h).await,
                            ref_count(&tv, &fixture.k).await
                        ),
                        (Some(1), Some(3)),
                        "h counts its one reference once"
                    );
                }

                /// A page taker that hands every page to the recount's own
                /// taker it wraps and, once its first page is taken, runs
                /// `after_first` on the store: a member added between a
                /// walk's first page and its last.
                struct AddingAfterFirstPage<'t, T> {
                    inner: T,
                    tv: &'t TestValkey,
                    after_first: Option<AddMember>,
                }

                /// A member to add under a rank walk: `obj:flds`'s field row
                /// `name` naming `hash` (its row and its index member), or a
                /// release of `hash` into `inflight`.
                enum AddMember {
                    FieldRow { name: &'static str, hash: String },
                    Inflight { hash: [u8; 32], oid: u64 },
                }

                impl<T: RankPageTaker> RankPageTaker for AddingAfterFirstPage<'_, T> {
                    type Error = T::Error;

                    fn walk_error(&self, e: WalkError) -> T::Error {
                        self.inner.walk_error(e)
                    }

                    async fn take_page(&mut self, members: Vec<Value>) -> Result<(), T::Error> {
                        self.inner.take_page(members).await?;
                        match self.after_first.take() {
                            Some(AddMember::FieldRow { name, hash }) => {
                                seed_row(
                                    self.tv,
                                    OID,
                                    Family::Field,
                                    name,
                                    &[("hash", hash.as_str()), ("size", "1")],
                                )
                                .await;
                            }
                            Some(AddMember::Inflight { hash, oid }) => {
                                owe(self.tv, OwedStage::Inflight, hash, oid).await;
                            }
                            None => {}
                        }
                        Ok(())
                    }
                }

                /// T3 (b.f3t), a member added under a rank walk. The
                /// recount's rank pages of `obj:flds` (its row taker,
                /// `RecountRowPages`) and of `inflight` (`InflightPages`),
                /// each over three members by pages of 2, with a member added
                /// once the first page is taken: the walk returns four members
                /// where its first page's `ZCARD` read three, and its last
                /// page's `ZCARD` reads four, so it fails
                /// (`PassError::Incomplete`, a `CountMismatch` naming the key,
                /// `ZCARD` 3 at its start and 4 at its end) — before the
                /// recount sets any count, since every set follows every walk
                /// (`a_recount_whose_walk_skips_a_member_fails_incomplete_and_sets_no_count`).
                /// Each control, with nothing added, completes.
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn the_recounts_rank_pages_fail_when_a_member_is_added_between_its_first_and_last_page(
                ) {
                    const PAGE: u64 = 2;
                    let mut failures: Vec<String> = Vec::new();
                    for (index, add) in [
                        ("obj:flds", true),
                        ("obj:flds", false),
                        ("inflight", true),
                        ("inflight", false),
                    ] {
                        let case = format!("{index}, a member added: {add}");
                        let tv = TestValkey::setup().await;
                        let keys = tv.key_schema();
                        let h = hex_of(0xB4);
                        let (pool, _token) = tv.client().acquire_maintenance();
                        let conn = pool.next();
                        let mut hashes: HashSet<[u8; 32]> = HashSet::new();
                        let mut owed: HashMap<[u8; 32], u64> = HashMap::new();
                        let (key, result) = if index == "obj:flds" {
                            for name in ["f1", "f2", "f3"] {
                                seed_row(
                                    &tv,
                                    OID,
                                    Family::Field,
                                    name,
                                    &[("hash", h.as_str()), ("size", "1")],
                                )
                                .await;
                            }
                            let key = keys.obj_fields_index(OID);
                            let mut taker = AddingAfterFirstPage {
                                inner: RecountRowPages {
                                    client: tv.client(),
                                    pool,
                                    keys,
                                    object_id: OID,
                                    time_samples: false,
                                    op: OP_LIST_FIELDS,
                                    chunk: PAGE as usize,
                                    hashes: &mut hashes,
                                },
                                tv: &tv,
                                after_first: add.then(|| AddMember::FieldRow {
                                    name: "f9",
                                    hash: h.clone(),
                                }),
                            };
                            let result = page_by_rank_counted(
                                &RankWalk::new(conn, &key, false),
                                PAGE,
                                &mut taker,
                            )
                            .await;
                            (key, result)
                        } else {
                            for oid in [961_101, 961_102, 961_103] {
                                owe(&tv, OwedStage::Inflight, hash_of(0xB4), oid).await;
                            }
                            let key = keys.ext_inflight();
                            let mut taker = AddingAfterFirstPage {
                                inner: InflightPages {
                                    client: tv.client(),
                                    owed: &mut owed,
                                },
                                tv: &tv,
                                after_first: add.then_some(AddMember::Inflight {
                                    hash: hash_of(0xB4),
                                    oid: 961_109,
                                }),
                            };
                            let result = page_by_rank_counted(
                                &RankWalk::new(conn, &key, false),
                                PAGE,
                                &mut taker,
                            )
                            .await;
                            (key, result)
                        };
                        match (add, result) {
                            (false, Ok(())) => {}
                            (true, Err(PassError::Incomplete(incomplete))) => {
                                let want = WalkIncomplete::CountMismatch {
                                    walk: "ZRANGE",
                                    target: Arc::from(key.as_str()),
                                    card: "ZCARD",
                                    at_start: 3,
                                    returned: 4,
                                    at_end: 4,
                                    expected_end: 3,
                                };
                                if *incomplete != want {
                                    failures.push(format!("{case}: {incomplete:?}, want {want:?}"));
                                }
                            }
                            (_, other) => failures.push(format!("{case}: got {other:?}")),
                        }
                    }
                    assert!(failures.is_empty(), "{}", failures.join("\n"));
                }

                /// A step taker that hands every step to the recount's own
                /// taker it wraps, each step after the first with the first
                /// step's members again: a member a counted cursor walk
                /// returns twice, as a rehash between two steps can make it.
                struct Repeating<T> {
                    inner: T,
                    first: Option<Vec<Value>>,
                    /// The steps handed on with a repeat.
                    repeats: usize,
                }

                impl<T: CountedStepTaker> CountedStepTaker for Repeating<T> {
                    type Error = T::Error;

                    fn walk_error(&self, e: WalkError) -> T::Error {
                        self.inner.walk_error(e)
                    }

                    async fn take_step(&mut self, mut members: Vec<Value>) -> Result<(), T::Error> {
                        match &self.first {
                            None => self.first = Some(members.clone()),
                            Some(first) => {
                                members.extend(first.iter().cloned());
                                self.repeats += 1;
                            }
                        }
                        self.inner.take_step(members).await
                    }

                    fn distinct(&self) -> u64 {
                        self.inner.distinct()
                    }
                }

                /// More members than the store keeps in a compact encoding of
                /// `setting` (`*-max-listpack-entries`, read, never set), so a
                /// cursor step answers about its `COUNT` and a walk takes many
                /// steps.
                async fn past_compact_limit(tv: &TestValkey, setting: &str) -> usize {
                    use fred::interfaces::ConfigInterface;

                    let limit: HashMap<String, String> = tv
                        .client()
                        .pool
                        .config_get(setting)
                        .await
                        .unwrap_or_else(|e| panic!("CONFIG GET {setting}: {e}"));
                    limit
                        .get(setting)
                        .and_then(|n| n.parse::<usize>().ok())
                        .unwrap_or_else(|| panic!("the store reports {setting}"))
                        .max(128)
                        + 72
                }

                /// T4 (b.f3t), a repeated member. The recount's counted
                /// cursor walks of a table-encoded `obj:ext` (its field taker,
                /// `RecountExtFields`) and `pending` (`CountedPending`), by
                /// steps of 10, each step after the first handed on with the
                /// first step's members again ([`Repeating`]): each walk
                /// completes — its distinct members still equal `HLEN` /
                /// `SCARD` — reading every field's hash once, and tallying
                /// each `pending` release once (the tally equals the
                /// releases staged).
                ///
                /// Fail-on-old (in-place perturbations, red then green once
                /// restored): with `RecountExtFields::distinct` counting every
                /// field it was handed, repeats included, the `obj:ext` walk
                /// fails incomplete; with `CountedPending` tallying a repeated
                /// member again, the `pending` tally passes the releases
                /// staged.
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn the_recounts_counted_walks_count_a_member_a_step_repeats_once() {
                    const STEP: u64 = 10;
                    let tv = TestValkey::setup().await;
                    let keys = tv.key_schema();
                    let (pool, _token) = tv.client().acquire_maintenance();
                    let conn = pool.next();

                    let fields = past_compact_limit(&tv, "hash-max-listpack-entries").await;
                    let ext = keys.obj_ext_index(OID);
                    let index: Vec<(String, String)> = (0..fields)
                        .map(|i| (sha256_to_hex(&many_hash(0x63, i)), "1".to_owned()))
                        .collect();
                    let _: () = tv
                        .client()
                        .pool
                        .hset(&ext, index)
                        .await
                        .expect("seeding obj:ext must succeed");
                    let mut hashes = HashSet::new();
                    let mut taker = Repeating {
                        inner: RecountExtFields {
                            client: tv.client(),
                            fields: HashSet::new(),
                            hashes: &mut hashes,
                        },
                        first: None,
                        repeats: 0,
                    };
                    walk_counted(&CountedScan::hash(conn, &ext), STEP, &mut taker)
                        .await
                        .unwrap_or_else(|e| panic!("obj:ext: the walk completes, got {e:?}"));
                    assert!(taker.repeats > 0, "fixture: obj:ext: a step was repeated");
                    assert_eq!(hashes.len(), fields, "obj:ext: every field's hash, once");

                    let releases = past_compact_limit(&tv, "set-max-listpack-entries").await;
                    let members: Vec<String> = (0..releases)
                        .map(|i| ExtRelease::retirement(many_hash(0x64, i), 961_200).member())
                        .collect();
                    let _: u64 = tv
                        .client()
                        .pool
                        .sadd(keys.ext_pending(), members)
                        .await
                        .expect("seeding pending must succeed");
                    let mut owed: HashMap<[u8; 32], u64> = HashMap::new();
                    let mut taker = Repeating {
                        inner: CountedPending {
                            client: tv.client(),
                            seen: HashSet::new(),
                            owed: &mut owed,
                        },
                        first: None,
                        repeats: 0,
                    };
                    walk_counted(
                        &CountedScan::set(conn, &keys.ext_pending()),
                        STEP,
                        &mut taker,
                    )
                    .await
                    .unwrap_or_else(|e| panic!("pending: the walk completes, got {e:?}"));
                    assert!(taker.repeats > 0, "fixture: pending: a step was repeated");
                    assert_eq!(
                        owed.values().sum::<u64>(),
                        releases as u64,
                        "pending: each release tallied once"
                    );
                }

                /// A cursor-pager reader that counts the steps it took
                /// ([`CursorPageReader::take_page`]) and passes everything
                /// else to the reader it wraps.
                struct Counting<R> {
                    inner: R,
                    pages: usize,
                }

                impl<R: CursorPageReader> CursorPageReader for Counting<R> {
                    type Error = R::Error;
                    type Member = R::Member;

                    async fn read_step(
                        &mut self,
                        cursor: &str,
                        start: Option<&Witness>,
                    ) -> Result<GuardedStep<R::Member>, R::Error> {
                        self.inner.read_step(cursor, start).await
                    }

                    fn changed(&self, start: &Witness, later: &Witness) -> R::Error {
                        self.inner.changed(start, later)
                    }

                    async fn take_page(&mut self, members: Vec<R::Member>) -> Result<(), R::Error> {
                        self.pages += 1;
                        self.inner.take_page(members).await
                    }
                }

                /// Forge the walk of `target`'s witness reads `0..=steps` as
                /// [`FORGED_RUN_A`] and read `steps + 1` as
                /// [`FORGED_RUN_B`]: the walk's first `steps` steps match its
                /// start, and the next one's witness does not.
                fn forge_change_after(target: &str, steps: usize) -> ArmedWalkFault {
                    forge_process_change_for_test(target, steps + 1)
                }

                /// `change` is a change of serving process on the `walk` walk
                /// of `target`, from [`FORGED_RUN_A`] to [`FORGED_RUN_B`].
                fn assert_process_change(change: &ServingChange, walk: &str, target: &str) {
                    match change {
                        ServingChange::Process {
                            walk: w,
                            target: t,
                            before,
                            after,
                            ..
                        } => assert_eq!(
                            (*w, &**t, before.as_str(), after.as_str()),
                            (walk, target, FORGED_RUN_A, FORGED_RUN_B)
                        ),
                        other => panic!("a change of serving process, got {other:?}"),
                    }
                }

                /// Distinct hashes, told apart by `i`.
                fn many_hash(byte: u8, i: usize) -> [u8; 32] {
                    let mut hash = [byte; 32];
                    hash[..8].copy_from_slice(&(i as u64).to_be_bytes());
                    hash
                }

                /// Test 1 for the periodic pass's owed readers (b.gns; the
                /// recount reads the same keys by its own proofs, above):
                /// `SSCAN` of `pending` and `ZSCAN` of `inflight`, each
                /// holding 400 releases (a
                /// table, so a step of 100 answers about 100). A change
                /// forged in the witness after the second step ends each walk
                /// with the change, naming the key and both run_ids, and the
                /// reader tallied only the first step. Each control walk,
                /// unarmed, takes more than one step and tallies all 400.
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn the_owed_readers_take_only_the_steps_before_a_process_change() {
                    const RELEASES: usize = 400;
                    for (kind, command, op) in [
                        (OwedKind::Pending, "SSCAN", OP_SCAN_PENDING),
                        (OwedKind::Inflight, "ZSCAN", OP_SCAN_INFLIGHT),
                    ] {
                        let tv = TestValkey::setup().await;
                        let keys = tv.key_schema();
                        let members: Vec<String> = (0..RELEASES)
                            .map(|i| ExtRelease::retirement(many_hash(0x62, i), 961_000).member())
                            .collect();
                        let key = match kind {
                            OwedKind::Pending => {
                                let _: u64 = tv
                                    .client()
                                    .pool
                                    .sadd(keys.ext_pending(), members)
                                    .await
                                    .expect("seeding pending must succeed");
                                keys.ext_pending()
                            }
                            OwedKind::Inflight => {
                                let scored: Vec<(f64, String)> =
                                    members.into_iter().map(|m| (1.0, m)).collect();
                                let _: i64 = tv
                                    .client()
                                    .pool
                                    .zadd(keys.ext_inflight(), None, None, false, false, scored)
                                    .await
                                    .expect("seeding inflight must succeed");
                                keys.ext_inflight()
                            }
                        };
                        let (pool, _token) = tv.client().acquire_maintenance();
                        let never = CancellationToken::new();
                        let started = Instant::now();
                        let tallied = |owed: &HashMap<[u8; 32], u64>| owed.values().sum::<u64>();

                        let mut all = HashMap::new();
                        let mut control = Counting {
                            inner: OwedSteps {
                                client: tv.client(),
                                walk: KeyedWalk::new(pool.next(), command, &key),
                                kind,
                                op,
                                phase: "the owed walk",
                                owed: &mut all,
                                budget: Duration::MAX,
                                started,
                                cancel: &never,
                            },
                            pages: 0,
                        };
                        page_by_cursor(&mut control)
                            .await
                            .unwrap_or_else(|e| panic!("{command}: control completes, got {e}"));
                        let control_pages = control.pages;
                        assert!(control_pages > 1, "{command}: fixture: more than one step");
                        assert_eq!(
                            tallied(&all),
                            RELEASES as u64,
                            "{command}: control tallies all"
                        );

                        let forged = forge_change_after(&key, 1);
                        let mut taken = HashMap::new();
                        let mut crossed = Counting {
                            inner: OwedSteps {
                                client: tv.client(),
                                walk: KeyedWalk::new(pool.next(), command, &key),
                                kind,
                                op,
                                phase: "the owed walk",
                                owed: &mut taken,
                                budget: Duration::MAX,
                                started,
                                cancel: &never,
                            },
                            pages: 0,
                        };
                        let result = page_by_cursor(&mut crossed).await;
                        let pages = crossed.pages;
                        drop(forged);
                        match result {
                            Err(PassError::ServingChanged(change)) => {
                                assert_process_change(&change, command, &key);
                            }
                            other => {
                                panic!("{command}: the forged change ends the walk, got {other:?}")
                            }
                        }
                        assert_eq!(pages, 1, "{command}: only the first step is taken");
                        let tallied = tallied(&taken);
                        assert!(
                            0 < tallied && tallied < RELEASES as u64,
                            "{command}: only the first step's releases are tallied: {tallied}"
                        );
                    }
                }

                // ── On a real cluster ──────────────────────────────────────

                /// The fixture on `LIVE_EDIT_TEST_CLUSTER_URL`, or `None`
                /// after a `SKIP:` line.
                async fn cluster_fixture() -> Option<TestValkey> {
                    let Some(url) = crate::storage::test_support::test_cluster_url() else {
                        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
                        return None;
                    };
                    let tv = TestValkey::setup_with_url(url, 2).await;
                    assert!(
                        tv.client().pool.next().is_clustered(),
                        "LIVE_EDIT_TEST_CLUSTER_URL names a cluster"
                    );
                    Some(tv)
                }

                /// The cluster's primaries as `CLUSTER SLOTS` names them
                /// (`host:port`), each with its slot set, in the order a
                /// keyspace walk visits them: by its lowest slot.
                async fn map_primaries(tv: &TestValkey) -> Vec<(String, SlotSet)> {
                    let reply: Value = tv
                        .client()
                        .pool
                        .next()
                        .cluster_slots()
                        .await
                        .expect("CLUSTER SLOTS must succeed");
                    let mut ranges: Vec<(u16, u16, String)> = reply
                        .into_array()
                        .into_iter()
                        .map(|entry| {
                            let mut fields = entry.into_array().into_iter();
                            let slot = |value: Option<Value>| {
                                value
                                    .and_then(|v| v.as_i64())
                                    .and_then(|n| u16::try_from(n).ok())
                                    .expect("a slot number")
                            };
                            let (start, end) = (slot(fields.next()), slot(fields.next()));
                            let mut node =
                                fields.next().expect("a primary").into_array().into_iter();
                            let host = node.next().and_then(|h| h.as_string()).unwrap_or_default();
                            let port = node.next().and_then(|p| p.as_i64()).expect("a port");
                            (start, end, format!("{host}:{port}"))
                        })
                        .collect();
                    ranges.sort_unstable();
                    let mut primaries: Vec<(String, Vec<(u16, u16)>)> = Vec::new();
                    for (start, end, primary) in ranges {
                        match primaries.iter_mut().find(|(p, _)| *p == primary) {
                            Some((_, owned)) => owned.push((start, end)),
                            None => primaries.push((primary, vec![(start, end)])),
                        }
                    }
                    primaries
                        .into_iter()
                        .map(|(primary, owned)| (primary, SlotSet::from_ranges(owned)))
                        .collect()
                }

                /// A `CLUSTER NODES` reply whose myself line lists `owned`.
                fn myself_reply(owned: &SlotSet) -> String {
                    let slots: Vec<String> = owned
                        .ranges()
                        .iter()
                        .map(|&(start, end)| format!("{start}-{end}"))
                        .collect();
                    format!(
                        "0000000000000000000000000000000000000001 127.0.0.1:7000@17000 \
                         myself,master - 0 0 1 connected {}\n",
                        slots.join(" ")
                    )
                }

                /// `owned` less its highest slot.
                fn less_its_last_slot(owned: &SlotSet) -> SlotSet {
                    let mut ranges = owned.ranges().to_vec();
                    let (start, end) = ranges.pop().expect("a primary serves a slot");
                    if start < end {
                        ranges.push((start, end - 1));
                    }
                    SlotSet::from_ranges(ranges)
                }

                /// Test 18 (b.gns; cluster-gated, `test-cluster:`). A pinned
                /// node whose myself line lists one slot fewer than the
                /// walk's map gives its primary — forged on the object
                /// walk's first `CLUSTER NODES` read — is `NotOwner` at the
                /// walk's start: the periodic pass forfeits with it, naming
                /// the primary, the node and both slot sets, after
                /// re-syncing the client's routing table, and
                /// `enumeration_failed` ticks once. That `NotOwner` is no
                /// fred error, so never counted on `valkey_errors_total`, is
                /// `scan_guard`'s unit test
                /// `a_missing_witness_names_its_command_and_what_was_seen_and_is_not_a_fred_error`
                /// (that metric counts only fred's `Timeout`, so an assertion
                /// on it here could not fail).
                ///
                /// Fail-on-old (not run in this sandbox, which has no
                /// cluster): with the slot-set comparison in `read_primary`
                /// removed, the pass completes and `enumeration_failed`
                /// stays 0.
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn a_cluster_pass_whose_object_walk_misses_its_primarys_slots_forfeits() {
                    const REPLICA: &str = "gns-recon-cluster-not-owner";
                    let Some(tv) = cluster_fixture().await else {
                        return;
                    };
                    let metrics = Arc::new(metrics_scrape_only(REPLICA));
                    seed_object_referencing(&tv, OID, &hex_of(0xB3)).await;
                    let (primary, expected) = map_primaries(&tv)
                        .await
                        .into_iter()
                        .next()
                        .expect("a primary");
                    let served = less_its_last_slot(&expected);
                    let pattern = tv.key_schema().obj_header_scan_pattern();
                    let forged = || {
                        arm_walk_fault_for_test(
                            &pattern,
                            WalkFault::ClusterNodesReply {
                                read: 0,
                                reply: myself_reply(&served),
                            },
                        )
                    };

                    let armed = forged();
                    let pass = run_pass(
                        tv.client(),
                        tv.key_schema(),
                        TEST_GRACE,
                        TEST_INTERVAL,
                        Instant::now(),
                        &CancellationToken::new(),
                        None,
                    )
                    .await;
                    drop(armed);
                    match pass {
                        Err(PassError::ServingChanged(change)) => match *change {
                            ServingChange::NotOwner {
                                primary: ref p,
                                ref node,
                                expected: ref e,
                                served: ref s,
                                at,
                                ref refresh,
                                ..
                            } => {
                                assert_eq!(p, &primary, "the map's first primary");
                                assert!(node.is_some(), "the pinned node is named");
                                assert_eq!((e, s.as_ref()), (&expected, Some(&served)));
                                assert_eq!(at, WalkEnd::Start);
                                assert_eq!(refresh, &RoutingRefresh::Refreshed);
                                let text = change.to_string();
                                for part in
                                    [primary.clone(), expected.to_string(), served.to_string()]
                                {
                                    assert!(
                                        text.contains(&part),
                                        "the line names {part:?}: {text}"
                                    );
                                }
                            }
                            other => panic!("NotOwner, got {other:?}"),
                        },
                        other => panic!("the pass forfeits on NotOwner, got {other:?}"),
                    }

                    let armed = forged();
                    one_pass(&tv, &metrics, REPLICA).await;
                    drop(armed);
                    assert_eq!(
                        outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
                        1
                    );
                    assert_eq!(gauges(&metrics, REPLICA), None, "nothing is published");
                    println!("CLUSTER-GATED RAN: a_cluster_pass_whose_object_walk_misses_its_primarys_slots_forfeits");
                }

                /// Test 19, the recount and the periodic pass (b.gns, policy
                /// 6; cluster-gated, `test-cluster:`). A pinned node's
                /// `CLUSTER NODES` reply with no myself line — forged on the
                /// object walk's first read — is a missing witness: the
                /// recount fails with `RecountError::WitnessMissing` (exit 3
                /// in the Job) naming `CLUSTER NODES` and "no myself line",
                /// setting no count, and the periodic pass forfeits, ticking
                /// `enumeration_failed` once.
                ///
                /// Fail-on-old (not run in this sandbox): with an unusable
                /// myself line mapped to a `WalkError::Read` again, as
                /// before policy 6, the recount fails as a store fault
                /// (exit 1) instead.
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn a_cluster_walk_with_no_myself_line_fails_the_recount_as_a_missing_witness()
                {
                    const REPLICA: &str = "gns-recon-cluster-no-myself";
                    let Some(tv) = cluster_fixture().await else {
                        return;
                    };
                    let metrics = Arc::new(metrics_scrape_only(REPLICA));
                    let fixture = seed(&tv).await;
                    let before = metas(&tv, &fixture).await;
                    let pattern = tv.key_schema().obj_header_scan_pattern();
                    let forged = || {
                        arm_walk_fault_for_test(
                            &pattern,
                            WalkFault::ClusterNodesReply {
                                read: 0,
                                reply: "0000000000000000000000000000000000000001 \
                                        127.0.0.1:7000@17000 master - 0 0 1 connected 0-16383\n"
                                    .to_owned(),
                            },
                        )
                    };

                    let armed = forged();
                    let result = recount(&tv).await;
                    drop(armed);
                    match result {
                        Err(RecountError::WitnessMissing(why)) => {
                            for part in ["`CLUSTER NODES`", "no myself line", pattern.as_str()] {
                                assert!(why.contains(part), "names {part:?}: {why}");
                            }
                        }
                        other => panic!(
                            "no myself line fails the recount as a missing witness, got {:?}",
                            other.map(|receipt| receipt.report().clone())
                        ),
                    }
                    assert_eq!(metas(&tv, &fixture).await, before, "no count is set");

                    let armed = forged();
                    one_pass(&tv, &metrics, REPLICA).await;
                    drop(armed);
                    assert_eq!(
                        outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
                        1
                    );
                    println!(
                        "CLUSTER-GATED RAN: a_cluster_walk_with_no_myself_line_fails_the_recount_as_a_missing_witness"
                    );
                }
            }
        }

        // ── The row reads, chunk by chunk ───────────────────────────────────
        //
        // `object_referenced_hashes` reads an object's row hashes
        // `ROW_HASH_CHUNK_ROWS` rows to a pipeline, folding each chunk's
        // replies before it sends the next, so the client holds one chunk's
        // commands and replies however large the object is. The chunking must
        // change nothing the walk reads, and it must bound what is in flight.
        mod row_hash_chunks {
            use super::*;

            use std::collections::BTreeSet;

            use fred::interfaces::{ClientInterface, ConfigInterface};

            use crate::storage::test_support::{
                object_encoding_for_test, release_all, CONN_STALL_WAIT,
            };

            /// The object every test here reads.
            const OID: &str = "962001";

            /// How long the bounded-read tests watch the stalled connection,
            /// once the first chunk (or the recount's first rank page's
            /// pipeline) has arrived, for any more of its commands. Well inside the
            /// fixture's 5 s command timeout, which the stalled reads must
            /// not reach.
            const QUIET_WINDOW: Duration = Duration::from_millis(300);

            /// The walk's read of [`OID`]'s references, its rows read
            /// `chunk_rows` at a time: `(hashes, index_mismatch)`, both
            /// sorted.
            async fn references_in_chunks(
                tv: &TestValkey,
                chunk_rows: usize,
            ) -> (Vec<[u8; 32]>, Vec<[u8; 32]>) {
                let members = list_value_members(tv.client(), tv.key_schema(), OID)
                    .await
                    .expect("the members list");
                let refs = object_referenced_hashes_in_chunks(
                    tv.client(),
                    tv.key_schema(),
                    OID,
                    &members,
                    chunk_rows,
                )
                .await
                .unwrap_or_else(|e| panic!("chunks of {chunk_rows}: the walk reads, got {e:?}"));
                let mut hashes: Vec<[u8; 32]> = refs.hashes.into_iter().collect();
                hashes.sort_unstable();
                let mut mismatch = refs.index_mismatch;
                mismatch.sort_unstable();
                (hashes, mismatch)
            }

            /// The restore Job's recount's read of [`OID`]'s references, its
            /// row indices read by rank pages of `step` ranks and its
            /// `obj:ext` by counted `HSCAN` steps of `step` elements
            /// ([`object_referenced_hashes_paged`]): the hashes, sorted. It
            /// runs no index audit, so it reports no index mismatch.
            async fn references_paged(tv: &TestValkey, step: usize) -> Vec<[u8; 32]> {
                let refs = object_referenced_hashes_paged(tv.client(), tv.key_schema(), OID, step)
                    .await
                    .unwrap_or_else(|e| panic!("steps of {step}: the read reads, got {e:?}"));
                assert_eq!(
                    refs.index_mismatch,
                    Vec::<[u8; 32]>::new(),
                    "steps of {step}: the recount audits no index"
                );
                let mut hashes: Vec<[u8; 32]> = refs.hashes.into_iter().collect();
                hashes.sort_unstable();
                hashes
            }

            /// Seed [`OID`] with every row shape on compact indices: five
            /// field rows and three time-sample rows, A on two fields and a
            /// time sample, B and C on one field each, D on a time sample,
            /// and an inline field and time sample (no hash). Its index
            /// stores A at 3 and D at 1 (agreeing), B at 2 (an over-count),
            /// no field for C, and E, which no row holds. Returns the hashes
            /// the object references and the index mismatches, both sorted.
            async fn seed_every_row_shape(tv: &TestValkey) -> (Vec<[u8; 32]>, Vec<[u8; 32]>) {
                let (a, b, c, d, e) = (
                    hex_of(0xA1),
                    hex_of(0xB1),
                    hex_of(0xC1),
                    hex_of(0xD1),
                    hex_of(0xE1),
                );
                let external =
                    |hash: &str| vec![("hash", hash.to_owned()), ("size", "1".to_owned())];
                let inline = vec![("value", "AA==".to_owned())];
                for (name, columns) in [
                    ("f0", external(&a)),
                    ("f1", external(&a)),
                    ("f2", inline.clone()),
                    ("f3", external(&b)),
                    ("f4", external(&c)),
                ] {
                    let columns: Vec<(&str, &str)> =
                        columns.iter().map(|(k, v)| (*k, v.as_str())).collect();
                    seed_row(tv, OID, Family::Field, name, &columns).await;
                }
                for (time, columns) in [("0.5", external(&a)), ("1", external(&d)), ("2", inline)] {
                    let columns: Vec<(&str, &str)> =
                        columns.iter().map(|(k, v)| (*k, v.as_str())).collect();
                    seed_row(tv, OID, Family::TimeSample, time, &columns).await;
                }
                seed_index(
                    tv,
                    OID,
                    &[
                        (a.as_str(), "3"),
                        (b.as_str(), "2"),
                        (d.as_str(), "1"),
                        (e.as_str(), "1"),
                    ],
                )
                .await;
                let sorted = |mut hashes: Vec<[u8; 32]>| {
                    hashes.sort_unstable();
                    hashes
                };
                (
                    sorted(vec![
                        hash_of(0xA1),
                        hash_of(0xB1),
                        hash_of(0xC1),
                        hash_of(0xD1),
                        hash_of(0xE1),
                    ]),
                    sorted(vec![hash_of(0xB1), hash_of(0xC1), hash_of(0xE1)]),
                )
            }

            /// The periodic walk reads the same hashes and the same index
            /// mismatches whatever its chunk size: one row a chunk (each a
            /// pipeline of one read, an inline row's nil reply among them),
            /// chunks that end inside the field rows, one that spans the last
            /// field row and the first time-sample row, one that ends exactly
            /// at the last field row, one exactly the object's rows, and
            /// chunks larger than the object, the production size among them
            /// (the object of [`seed_every_row_shape`]).
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn object_referenced_hashes_reads_the_same_references_at_every_chunk_size() {
                let tv = TestValkey::setup().await;
                let expected = seed_every_row_shape(&tv).await;

                for chunk_rows in [1, 2, 3, 5, 8, 9, ROW_HASH_CHUNK_ROWS] {
                    assert_eq!(
                        references_in_chunks(&tv, chunk_rows).await,
                        expected,
                        "chunks of {chunk_rows} rows: the hashes and the index mismatches"
                    );
                }
            }

            /// One hash per index of `byte`, told apart by `i`: the table
            /// fixture's many distinct hashes.
            fn table_hash(byte: u8, i: usize) -> [u8; 32] {
                let mut hash = [byte; 32];
                hash[..8].copy_from_slice(&(i as u64).to_be_bytes());
                hash
            }

            /// Seed [`OID`] with indices the store keeps as tables, past the
            /// compact encodings' 128 entries, and return the hashes it
            /// references, sorted: 150 field rows (every tenth inline, every
            /// tenth after it dual-shape, a `value` beside a `hash`, whose raw
            /// column both walks count), on 16 hashes; 140 time-sample rows
            /// (every tenth inline) on 16 more, and a legacy `-0.0` / `+0.0`
            /// pair naming two hashes of their own; and an `obj:ext` of 12
            /// fields more than the store keeps compact, none of which a row
            /// names. Asserts the three encodings.
            async fn seed_table_encoded_object(tv: &TestValkey) -> Vec<[u8; 32]> {
                type Row = (String, String, String, Vec<(&'static str, String)>);
                let keys = tv.key_schema();
                let mut referenced: BTreeSet<[u8; 32]> = BTreeSet::new();
                // Each row: its index, its member, its key and its columns.
                let mut rows: Vec<Row> = Vec::new();
                let external =
                    |hash: &[u8; 32]| vec![("hash", sha256_to_hex(hash)), ("size", "1".to_owned())];
                for i in 0..150usize {
                    let member = KeySchema::field_member("1", &format!("k{i:03}"));
                    let row_key = keys.obj_field_from_member(OID, &member);
                    let hash = table_hash(0x40, i % 16);
                    let columns = match i % 10 {
                        0 => vec![("value", "AA==".to_owned())],
                        1 => {
                            referenced.insert(hash);
                            let mut columns = external(&hash);
                            columns.push(("value", "AA==".to_owned()));
                            columns
                        }
                        _ => {
                            referenced.insert(hash);
                            external(&hash)
                        }
                    };
                    rows.push((keys.obj_fields_index(OID), member, row_key, columns));
                }
                let samples = (0..140usize)
                    .map(|i| (i as f64 + 0.25, table_hash(0x50, i % 16), i % 10 == 0))
                    .chain([(-0.0, hash_of(0x5A), false), (0.0, hash_of(0x5B), false)]);
                for (time, hash, inline) in samples {
                    let member = KeySchema::time_sample_member("1", time);
                    let row_key = keys.obj_ts_from_member(OID, &member);
                    let columns = if inline {
                        vec![("value", "AA==".to_owned())]
                    } else {
                        referenced.insert(hash);
                        external(&hash)
                    };
                    rows.push((keys.obj_time_samples_index(OID), member, row_key, columns));
                }
                let seeding = tv.client().pool.next().pipeline();
                for (index, member, row_key, columns) in rows {
                    let _: () = seeding
                        .hset(row_key, columns)
                        .await
                        .expect("queueing a row");
                    let _: () = seeding
                        .zadd(index, None, None, false, false, (1.0, member.as_str()))
                        .await
                        .expect("queueing its index member");
                }
                // Past the store's own compact limit for hashes, read, never
                // set: a store may keep more than the default 128 fields
                // compact.
                let limit: HashMap<String, String> = tv
                    .client()
                    .pool
                    .config_get("hash-max-listpack-entries")
                    .await
                    .expect("CONFIG GET hash-max-listpack-entries");
                let ext_fields = limit
                    .get("hash-max-listpack-entries")
                    .and_then(|n| n.parse::<usize>().ok())
                    .expect("the store reports its hash listpack limit")
                    .max(128)
                    + 12;
                let index: Vec<(String, String)> = (0..ext_fields)
                    .map(|i| {
                        let hash = table_hash(0x60, i);
                        referenced.insert(hash);
                        (sha256_to_hex(&hash), "1".to_owned())
                    })
                    .collect();
                let _: () = seeding
                    .hset(keys.obj_ext_index(OID), index)
                    .await
                    .expect("queueing obj:ext");
                let _: Vec<Value> = seeding
                    .all()
                    .await
                    .expect("seeding the object must succeed");
                for (key, encoding) in [
                    (keys.obj_fields_index(OID), "skiplist"),
                    (keys.obj_time_samples_index(OID), "skiplist"),
                    (keys.obj_ext_index(OID), "hashtable"),
                ] {
                    assert_eq!(
                        object_encoding_for_test(tv.client(), &key).await,
                        encoding,
                        "fixture: {key} is table-encoded"
                    );
                }
                referenced.into_iter().collect()
            }

            /// Part A's equivalence: the recount's paged read names exactly
            /// the hashes the periodic walk's whole-listing read names, at
            /// pages and steps of 1, 2, 3, 5, 8 and 9 and the production
            /// page, on compact indices (the every-shape object, whose
            /// `obj:ext` step answers the whole hash) and on table-encoded
            /// ones (whose `obj:ext` steps answer about the step, so that
            /// walk takes many steps, and whose `-0.0` / `+0.0` pair the rank
            /// pages read as two rows). A rank page spans exactly its ranks
            /// on either encoding. On the table object both also equal the
            /// fixture's own list of what it references.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recounts_paged_read_names_the_periodic_reads_hashes_at_every_step() {
                let tv = TestValkey::setup().await;
                let (shapes, _) = seed_every_row_shape(&tv).await;
                let periodic = references_in_chunks(&tv, ROW_HASH_CHUNK_ROWS).await.0;
                assert_eq!(periodic, shapes, "fixture: the periodic read");
                for step in [1, 2, 3, 5, 8, 9, ROW_HASH_CHUNK_ROWS] {
                    assert_eq!(
                        references_paged(&tv, step).await,
                        periodic,
                        "compact indices, steps of {step}"
                    );
                }

                let tv = TestValkey::setup().await;
                let table = seed_table_encoded_object(&tv).await;
                let periodic = references_in_chunks(&tv, ROW_HASH_CHUNK_ROWS).await.0;
                assert_eq!(periodic, table, "fixture: the periodic read");
                for step in [1, 2, 3, 5, 8, 9, ROW_HASH_CHUNK_ROWS] {
                    assert_eq!(
                        references_paged(&tv, step).await,
                        periodic,
                        "table-encoded indices, steps of {step}"
                    );
                }
            }

            /// One command as the server reads it: a RESP array of bulk
            /// strings.
            fn frame_len(args: &[&str]) -> u64 {
                let header = format!("*{}\r\n", args.len()).len();
                let body: usize = args
                    .iter()
                    .map(|arg| format!("${}\r\n{arg}\r\n", arg.len()).len())
                    .sum();
                (header + body) as u64
            }

            /// One of the recount's rank pages of `key` as the server reads
            /// it: `ZRANGE <key> <start> <stop>` and `ZCARD <key>`, sent in
            /// one pipeline.
            fn rank_page_len(key: &str, start: u64, stop: u64) -> u64 {
                frame_len(&["ZRANGE", key, &start.to_string(), &stop.to_string()])
                    + frame_len(&["ZCARD", key])
            }

            /// `tot-net-in` of connection `conn_id`, read from another
            /// connection: every byte the server has read from it.
            async fn net_in_of(tv: &TestValkey, conn_id: &str) -> u64 {
                let listing: String = tv
                    .client()
                    .pool
                    .next()
                    .client_list::<String, String>(None, Some(vec![conn_id.to_owned()]))
                    .await
                    .expect("CLIENT LIST ID must succeed");
                listing
                    .split_whitespace()
                    .find_map(|field| field.strip_prefix("tot-net-in="))
                    .unwrap_or_else(|| panic!("connection {conn_id} is listed: {listing:?}"))
                    .parse()
                    .expect("tot-net-in is a byte count")
            }

            /// The last command connection `conn_id` sent, as `CLIENT LIST`
            /// names it.
            async fn last_command_of(tv: &TestValkey, conn_id: &str) -> String {
                let listing: String = tv
                    .client()
                    .pool
                    .next()
                    .client_list::<String, String>(None, Some(vec![conn_id.to_owned()]))
                    .await
                    .expect("CLIENT LIST ID must succeed");
                listing
                    .split_whitespace()
                    .find_map(|field| field.strip_prefix("cmd="))
                    .unwrap_or_else(|| panic!("connection {conn_id} is listed: {listing:?}"))
                    .to_owned()
            }

            /// T3's bound (b.f3t): the recount reads a row index by rank
            /// pages, one page in flight, with each page's `ZCARD` in its
            /// pipeline, and never a `ZSCAN`, a whole listing or a witness.
            /// On an object of 384 field rows (a table-encoded `obj:flds`)
            /// read by pages of 4, with the maintenance connection — the one
            /// connection the read's pages, its row reads and its `obj:ext`
            /// steps share — held in a server-side `BLPOP`, the read's first
            /// pipeline reaches the server and then nothing more does:
            /// exactly `ZRANGE obj:flds 0 3` and `ZCARD obj:flds`, byte for
            /// byte, for the whole [`QUIET_WINDOW`], so the count costs no
            /// round trip of its own and no second page goes out before the
            /// first is answered. Released, the read completes with the
            /// object's one hash, and the connection has sent, byte for byte,
            /// exactly: `obj:flds`'s 97 pages (the last, past its 384 ranks,
            /// empty), each full page followed by its four rows' `HGET`s;
            /// `obj:tss`'s one empty page; and one `HSCAN obj:ext 0 COUNT 4`
            /// with its `HLEN`, the last command. A `ZSCAN`, a whole listing
            /// or an `INFO server` anywhere would change the count.
            ///
            /// New behaviour (b.f3t), shown non-vacuous by its exact counts
            /// rather than a run against the old code, whose `ZSCAN` steps
            /// and witnesses this read no longer has.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recounts_paged_read_sends_one_rank_page_at_a_time_and_no_zscan_or_whole_listing(
            ) {
                const PAGE: usize = 4;
                let tv = TestValkey::setup().await;
                let keys = tv.key_schema().clone();
                let hash = hex_of(0xA4);
                let rows = 3 * 128;
                let members: Vec<String> = (0..rows)
                    .map(|row| KeySchema::field_member("1", &format!("k{row:05}")))
                    .collect();
                let seeding = tv.client().pool.next().pipeline();
                for member in &members {
                    let _: () = seeding
                        .hset(
                            keys.obj_field_from_member(OID, member),
                            [("hash", hash.as_str()), ("size", "1")],
                        )
                        .await
                        .expect("queueing a row");
                    let _: () = seeding
                        .zadd(
                            keys.obj_fields_index(OID),
                            None,
                            None,
                            false,
                            false,
                            (1.0, member.as_str()),
                        )
                        .await
                        .expect("queueing its index member");
                }
                let _: Vec<Value> = seeding.all().await.expect("seeding the rows must succeed");
                seed_index(&tv, OID, &[(hash.as_str(), &rows.to_string())]).await;
                let index = keys.obj_fields_index(OID);
                assert_eq!(
                    object_encoding_for_test(tv.client(), &index).await,
                    "skiplist",
                    "fixture: obj:flds is table-encoded"
                );
                let first_page = rank_page_len(&index, 0, PAGE as u64 - 1);
                // Every byte the whole read sends, in order: `obj:flds`'s
                // pages, each full one followed by its rows' reads, then the
                // empty page past its last rank; `obj:tss`'s one empty page;
                // `obj:ext`'s one counted step (one field: compact, so one
                // step answers it whole).
                let mut whole_read: u64 = 0;
                for (page, chunk) in members.chunks(PAGE).enumerate() {
                    let start = (page * PAGE) as u64;
                    whole_read += rank_page_len(&index, start, start + PAGE as u64 - 1);
                    for member in chunk {
                        whole_read +=
                            frame_len(&["HGET", &keys.obj_field_from_member(OID, member), "hash"]);
                    }
                }
                whole_read += rank_page_len(&index, rows as u64, (rows + PAGE) as u64 - 1);
                whole_read += rank_page_len(&keys.obj_time_samples_index(OID), 0, PAGE as u64 - 1);
                let ext = keys.obj_ext_index(OID);
                whole_read += frame_len(&["HSCAN", &ext, "0", "COUNT", &PAGE.to_string()])
                    + frame_len(&["HLEN", &ext]);

                let stalls = tv.stall_maintenance_pool().await;
                assert_eq!(stalls.len(), 1, "fixture: one maintenance connection");
                let conn_id = stalls[0].conn_id().to_owned();
                let baseline = stalls[0].net_in().await.expect("the stalled connection");
                let read = {
                    let client = tv.client().clone();
                    let keys = keys.clone();
                    tokio::spawn(async move {
                        object_referenced_hashes_paged(&client, &keys, OID, PAGE).await
                    })
                };
                // Wait for the whole first pipeline; released before any
                // assertion, so a failed one leaves no connection stalled.
                let deadline = tokio::time::Instant::now() + CONN_STALL_WAIT;
                let arrived = loop {
                    let sent = stalls[0].net_in().await.expect("the stalled connection") - baseline;
                    if sent >= first_page || tokio::time::Instant::now() >= deadline {
                        break sent;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                };
                if arrived >= first_page {
                    tokio::time::sleep(QUIET_WINDOW).await;
                }
                let sent = stalls[0].net_in().await.expect("the stalled connection") - baseline;
                release_all(stalls).await;

                assert!(
                    arrived >= first_page,
                    "the first page's pipeline (its ZRANGE and its ZCARD, {first_page} B) did not \
                     reach the server within {CONN_STALL_WAIT:?} ({arrived} B did)"
                );
                assert_eq!(
                    sent, first_page,
                    "while its first page waits for its replies, the read sends only that page's \
                     ZRANGE 0 3 and ZCARD: no second page, no ZSCAN, no whole listing"
                );
                let refs = tokio::time::timeout(CONN_STALL_WAIT, read)
                    .await
                    .expect("the read finishes once released")
                    .expect("the read task did not panic")
                    .expect("released, the read succeeds");
                assert_eq!(
                    refs.hashes,
                    HashSet::from([hash_of(0xA4)]),
                    "every page read: the one hash"
                );
                assert_eq!(
                    net_in_of(&tv, &conn_id).await - baseline,
                    whole_read,
                    "the connection sent the rank pages, their row reads and obj:ext's one \
                     counted step, and nothing else"
                );
                assert_eq!(
                    last_command_of(&tv, &conn_id).await,
                    "hlen",
                    "the read ends on obj:ext's count, not a whole listing"
                );
            }

            /// A rank page spans exactly its ranks whatever the index's
            /// encoding, so the recount holds at most one page of rows: on
            /// ten field rows (a `listpack` `obj:flds`, whose `ZSCAN` step
            /// would answer all ten whatever its `COUNT`) read by pages of 4,
            /// with the row-hash reply seam armed, the first chunk is the
            /// first page's four rows, so the arity check fails it "3 frames
            /// for 4 rows". Disarmed, the same read is clean.
            ///
            /// Fails with the recount's rows read in chunks of one (in-place
            /// perturbation of `RecountRowPages::take_page`): the first
            /// chunk is one row, "0 frames for 1 rows".
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn the_recount_reads_a_compact_index_one_page_of_rows_at_a_time() {
                let tv = TestValkey::setup().await;
                let keys = tv.key_schema().clone();
                let hash = hex_of(0xA5);
                for row in 0..10 {
                    seed_row(
                        &tv,
                        OID,
                        Family::Field,
                        &format!("k{row}"),
                        &[("hash", hash.as_str()), ("size", "1")],
                    )
                    .await;
                }
                seed_index(&tv, OID, &[(hash.as_str(), "10")]).await;
                assert_eq!(
                    object_encoding_for_test(tv.client(), &keys.obj_fields_index(OID)).await,
                    "listpack",
                    "fixture: obj:flds is compact"
                );

                let armed = arm_row_hash_reply_drop_for_test(&keys.obj_ext_index(OID));
                match object_referenced_hashes_paged(tv.client(), &keys, OID, 4).await {
                    Err(PassError::Undecodable { what, text }) => {
                        assert_eq!(what, "a row-hash pipeline reply of the expected arity");
                        assert_eq!(text, "3 frames for 4 rows", "the first chunk is the page");
                    }
                    other => panic!(
                        "a short chunk fails the read closed, got {:?}",
                        other.map(|refs| refs.hashes)
                    ),
                }
                drop(armed);
                let refs = object_referenced_hashes_paged(tv.client(), &keys, OID, 4)
                    .await
                    .expect("disarmed, the read is clean");
                assert_eq!(refs.hashes, HashSet::from([hash_of(0xA5)]));
            }

            /// `HGET <row> hash` as the server reads it: a RESP array of
            /// three bulk strings.
            fn hget_hash_frame_len(row_key: &str) -> u64 {
                format!(
                    "*3\r\n$4\r\nHGET\r\n${}\r\n{row_key}\r\n$4\r\nhash\r\n",
                    row_key.len()
                )
                .len() as u64
            }

            /// A large object's reference read never sends more than one
            /// chunk of row reads before that chunk is answered. Its object
            /// has two chunks and one row more of field rows; with the
            /// maintenance connection held in a server-side `BLPOP`, the
            /// read's first chunk reaches the server and then nothing more
            /// does: exactly the first [`ROW_HASH_CHUNK_ROWS`] rows' `HGET`s,
            /// byte for byte, for the whole [`QUIET_WINDOW`]. Released, the
            /// read finishes its three chunks and reads the object's one hash.
            ///
            /// The window bounds what an absence can show: a correct read
            /// sends nothing more however long it is held, so the window
            /// cannot fail it; a read that sent every chunk at once would
            /// have to take longer than the window to send the rest to pass.
            ///
            /// Fails without the chunking (in-place perturbation of
            /// `object_referenced_hashes`, passing `usize::MAX` rows a chunk):
            /// all 2,049 rows' reads arrive at once.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn a_large_objects_reference_read_sends_one_chunk_of_row_reads_at_a_time() {
                let tv = TestValkey::setup().await;
                let keys = tv.key_schema().clone();
                let hash = hex_of(0xA2);
                let rows = 2 * ROW_HASH_CHUNK_ROWS + 1;
                let seeding = tv.client().pool.next().pipeline();
                for row in 0..rows {
                    let member = KeySchema::field_member("1", &format!("k{row:05}"));
                    let _: () = seeding
                        .hset(
                            keys.obj_field_from_member(OID, &member),
                            [("hash", hash.as_str()), ("size", "1")],
                        )
                        .await
                        .expect("queueing a row");
                    let _: () = seeding
                        .zadd(
                            keys.obj_fields_index(OID),
                            None,
                            None,
                            false,
                            false,
                            (1.0, member.as_str()),
                        )
                        .await
                        .expect("queueing its index member");
                }
                let _: Vec<Value> = seeding.all().await.expect("seeding the rows must succeed");
                seed_index(&tv, OID, &[(hash.as_str(), &rows.to_string())]).await;
                let members = list_value_members(tv.client(), &keys, OID)
                    .await
                    .expect("the members list");
                assert_eq!(members.fields.len(), rows, "fixture: every row listed");
                let first_chunk: u64 = members.fields[..ROW_HASH_CHUNK_ROWS]
                    .iter()
                    .map(|member| hget_hash_frame_len(&keys.obj_field_from_member(OID, member)))
                    .sum();

                let stalls = tv.stall_maintenance_pool().await;
                assert_eq!(stalls.len(), 1, "fixture: one maintenance connection");
                let baseline = stalls[0].net_in().await.expect("the stalled connection");
                let read = {
                    let client = tv.client().clone();
                    let keys = keys.clone();
                    tokio::spawn(async move {
                        object_referenced_hashes(&client, &keys, OID, &members).await
                    })
                };
                let deadline = tokio::time::Instant::now() + CONN_STALL_WAIT;
                loop {
                    let sent = stalls[0].net_in().await.expect("the stalled connection") - baseline;
                    if sent >= first_chunk {
                        break;
                    }
                    assert!(
                        tokio::time::Instant::now() < deadline,
                        "the first chunk's {first_chunk} B did not reach the server within \
                         {CONN_STALL_WAIT:?} ({sent} B did)"
                    );
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                tokio::time::sleep(QUIET_WINDOW).await;
                let sent = stalls[0].net_in().await.expect("the stalled connection") - baseline;
                release_all(stalls).await;

                assert_eq!(
                    sent, first_chunk,
                    "while the first chunk waits for its replies, only its {ROW_HASH_CHUNK_ROWS} \
                     reads are sent (all {rows} rows' would be far more)"
                );
                let refs = tokio::time::timeout(CONN_STALL_WAIT, read)
                    .await
                    .expect("the read finishes once released")
                    .expect("the read task did not panic")
                    .expect("the read succeeds");
                assert_eq!(
                    (refs.hashes, refs.index_mismatch),
                    (HashSet::from([hash_of(0xA2)]), Vec::new()),
                    "every chunk read: the one hash, its index agreeing"
                );
            }

            /// The walk's arity check fails closed: with a chunk's pipeline
            /// answering one reply short (the test seam
            /// `arm_row_hash_reply_drop_for_test`, armed for this object's
            /// `obj:ext` key), the reference read answers
            /// `PassError::Undecodable` "a row-hash pipeline reply of the
            /// expected arity", naming both counts, rather than pairing the
            /// replies with the wrong rows; the periodic pass forfeits
            /// (counted `enumeration_failed`) and the authoritative recount,
            /// whose paged read folds its chunks through the same
            /// `fold_row_hash_chunk`, fails corrupt, naming the object,
            /// setting no count. The guard dropped, the same read is clean.
            ///
            /// Fails without the check (in-place perturbation of
            /// `fold_row_hash_chunk`, its arity condition never true): the
            /// short chunk is folded and the read succeeds.
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn a_row_hash_reply_short_of_its_chunk_fails_the_walk_closed() {
                const REPLICA: &str = "tv3-row-hash-arity";
                let tv = TestValkey::setup().await;
                let keys = tv.key_schema().clone();
                let metrics = Arc::new(metrics_scrape_only(REPLICA));
                let hash = hex_of(0xA3);
                seed_live_header(&tv, OID).await;
                for name in ["f0", "f1", "f2"] {
                    seed_row(
                        &tv,
                        OID,
                        Family::Field,
                        name,
                        &[("hash", hash.as_str()), ("size", "1")],
                    )
                    .await;
                }
                seed_index(&tv, OID, &[(hash.as_str(), "3")]).await;
                let members = list_value_members(tv.client(), &keys, OID)
                    .await
                    .expect("the members list");

                let armed = arm_row_hash_reply_drop_for_test(&keys.obj_ext_index(OID));
                let read = object_referenced_hashes(tv.client(), &keys, OID, &members).await;
                match read {
                    Err(PassError::Undecodable { what, text }) => {
                        assert_eq!(what, "a row-hash pipeline reply of the expected arity");
                        assert_eq!(text, "2 frames for 3 rows");
                    }
                    other => panic!("a short reply fails the read closed, got {:?}", other.err()),
                }
                one_pass(&tv, &metrics, REPLICA).await;
                assert_eq!(
                    outcome(&metrics, EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED),
                    1,
                    "the periodic pass forfeits"
                );
                let recount =
                    recount_authoritative(tv.client(), &keys, || async { Ok::<(), ()>(()) }).await;
                match recount {
                    Err(RecountError::Corrupt { what, key }) => {
                        assert_eq!(what, "a row-hash pipeline reply of the expected arity");
                        assert_eq!(key, format!("object {OID}: 2 frames for 3 rows"));
                    }
                    other => panic!("the recount fails corrupt, got {:?}", other.err()),
                }
                let meta: HashMap<String, String> = tv
                    .client()
                    .pool
                    .hgetall(keys.ext_meta(&hash))
                    .await
                    .expect("HGETALL ext:meta must succeed");
                assert!(meta.is_empty(), "no count is set from a walk that failed");

                drop(armed);
                let refs = object_referenced_hashes(tv.client(), &keys, OID, &members)
                    .await
                    .expect("disarmed, the read is clean");
                assert_eq!(
                    (refs.hashes, refs.index_mismatch),
                    (HashSet::from([hash_of(0xA3)]), Vec::new())
                );
            }

            /// The restore memory envelope's record (plan b.tv3), which
            /// records the recount's step and chunk and its flat peaks.
            const RESTORE_ENVELOPE_DERIVATION_JSON: &str =
                "docs-internal/validation/restore-memory-envelope-tv3-20261022/derivation.json";

            /// The record's ratio of the larger shape's peak to the
            /// smaller's that still reads flat: the plateau ceiling the
            /// record's other paged measurements use (1.15).
            const FLAT_CEILING: f64 = 1.15;

            /// [`ROW_HASH_CHUNK_ROWS`] is the paged recount's step and
            /// chunk the record measured, `recount_paged.step_rows`, and
            /// still `recount_row_hash_chunk.value`. The recount's peaks
            /// are flat across its two shapes, at least 4 × apart in rows:
            /// each recorded ratio (heap, resident) is the larger shape's
            /// peak over the smaller's, to four places, and within
            /// [`FLAT_CEILING`]; the row ratio is the two shapes' rows, to
            /// three places. No row bound is stated any more: the per-row
            /// line and bound are retired (`line_mib` and `bound_rows` gone,
            /// kept as history under `retired_row_bound`). An edit to the
            /// constant or to the record without the other fails here; the
            /// published source tree, which has no `docs-internal/`, skips.
            #[test]
            fn row_hash_chunk_is_the_recorded_paged_recount_step_and_its_peaks_are_flat() {
                let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
                if !root.join("docs-internal").is_dir() {
                    eprintln!(
                        "skipped: no docs-internal/ (published source tree), so no \
                         {RESTORE_ENVELOPE_DERIVATION_JSON}"
                    );
                    return;
                }
                let raw = std::fs::read_to_string(root.join(RESTORE_ENVELOPE_DERIVATION_JSON))
                    .unwrap_or_else(|e| {
                        panic!("the committed record {RESTORE_ENVELOPE_DERIVATION_JSON}: {e}")
                    });
                let record: serde_json::Value =
                    serde_json::from_str(&raw).expect("the record is valid JSON");
                let integer = |pointer: &str| {
                    record
                        .pointer(pointer)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or_else(|| {
                            panic!("the record carries {pointer} as a non-negative integer")
                        })
                };
                let number = |pointer: &str| {
                    record
                        .pointer(pointer)
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or_else(|| panic!("the record carries {pointer} as a number"))
                };
                let to_places = |value: f64, places: i32| {
                    let scale = 10f64.powi(places);
                    (value * scale).round() / scale
                };

                assert_eq!(
                    integer("/recount_paged/step_rows"),
                    ROW_HASH_CHUNK_ROWS as u64,
                    "the step is the record's recount_paged.step_rows"
                );
                assert_eq!(
                    integer("/recount_row_hash_chunk/value"),
                    ROW_HASH_CHUNK_ROWS as u64,
                    "the chunk is still the record's recount_row_hash_chunk.value"
                );

                let rows: Vec<u64> = record
                    .pointer("/recount_paged/value_rows")
                    .and_then(serde_json::Value::as_array)
                    .expect("the record carries the two shapes' rows")
                    .iter()
                    .map(|n| n.as_u64().expect("a row count"))
                    .collect();
                let [small, large] = rows[..] else {
                    panic!("two shapes, got {rows:?}");
                };
                let row_ratio = large as f64 / small as f64;
                assert!(
                    row_ratio >= 4.0,
                    "the shapes are at least 4 x apart in rows: {row_ratio}"
                );
                assert_eq!(
                    number("/recount_paged/value_rows_ratio"),
                    to_places(row_ratio, 3),
                    "the recorded row ratio is the shapes' rows"
                );
                for (peaks, ratio) in [
                    ("heap_peak_bytes_max", "flat_ratio_heap"),
                    ("rss_growth_bytes_max", "flat_ratio_rss"),
                ] {
                    let peak = |shape_rows: u64| {
                        integer(&format!("/recount_paged/{peaks}/{shape_rows}_rows"))
                    };
                    let computed = peak(large) as f64 / peak(small) as f64;
                    let recorded = number(&format!("/recount_paged/{ratio}"));
                    assert_eq!(
                        recorded,
                        to_places(computed, 4),
                        "{ratio} is the larger shape's {peaks} over the smaller's"
                    );
                    assert!(
                        recorded <= FLAT_CEILING,
                        "{ratio}: {recorded} reads flat, at most {FLAT_CEILING}"
                    );
                }

                for retired in ["line_mib", "bound_rows"] {
                    assert!(
                        record
                            .pointer(&format!("/recount_row_hash_chunk/{retired}"))
                            .is_none(),
                        "recount_row_hash_chunk.{retired} is retired, no longer a stated bound"
                    );
                }
                assert!(
                    record
                        .pointer("/recount_row_hash_chunk/retired_row_bound")
                        .is_some(),
                    "the retired bound is kept as history"
                );
            }
        }
    }
}
