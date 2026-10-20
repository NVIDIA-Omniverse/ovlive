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

//! Integration tests for the checkpoint subsystem.
//!
//! Each child module owns one slice of checkpoint coverage:
//!
//! - [`atomicity_local_disk_test`] — Task `t2.fgj.ya.r7` Subtask
//!   `t3.fgj.ya.r7.xm`: verifies the LocalDisk adapter's write-temp +
//!   rename atomicity invariant via both temp-file naming observation and
//!   a mid-write SIGKILL of a child-process harness.
//! - [`atomicity_session_local_disk_test`] — Task `t2.fgj.7v.cq` Subtask
//!   `t3.fgj.7v.cq.tr`: session-checkpoint sibling to the layer test
//!   above. Spawns the shared `local_disk_write_kill_harness` example in
//!   `session` mode and SIGKILLs it mid-write to prove
//!   `put_session_checkpoint` honours the same write-temp + rename
//!   atomicity discipline at the recovery-hook view of the filesystem.
//! - [`assembler_test`] — Task `t2.fgj.ya.78` Subtask `t3.fgj.ya.78.kn`:
//!   end-to-end conformance test for the layer-checkpoint assembler against
//!   a live Valkey, decoding the OMPF through the real `oodf-ffi` bridge
//!   and asserting isomorphism with the live snapshot state.
//!
//! - [`conformance`] — Task `t2.fgj.ya.74` Subtask `t3.fgj.ya.74.dy` and
//!   siblings: the SR-1.3 trait-level conformance harness. Parametrized
//!   over a setup closure shape so each adapter (InMemory, LocalDisk) can
//!   register its own factory; per-case bodies land in the per-group
//!   Subtasks (`.9d`, `.3v`, `.d3`, `.op`, `.c3`).
//!
//! - [`recovery_admission_test`] — Task `t2.fgj.ya.qm` Subtask
//!   `t3.fgj.ya.qm.h7`: per-`object_id` K=1 admission discipline (SR-2.6)
//!   exercised across concurrent restorers — same-oid race collapses to
//!   one `apply_checkpoint` pipeline, distinct-oid restorers run in
//!   parallel, and the admission DashMap drains to zero after every
//!   burst.
//! - [`recovery_hook_test`] — Task `t2.fgj.ya.qm` Subtask
//!   `t3.fgj.ya.qm.v7`: handler-level recovery-hook integration tests
//!   pinning the at-most-once retry contract (UpdateLayer / StatLayer /
//!   SubscribeReadLayer) and the `LIVE_EDIT_CHECKPOINT_ENABLED=false`
//!   disabled-mode short-circuit. Gated behind
//!   `test-checkpoint-fault-injection`.
//!
//! ## Task `t2.fgj.ya.wg` — End-to-end recovery integration tests
//!
//! Subtask `t3.fgj.ya.wg.9n` lands the [`helpers`] scaffolding (harness
//! builder, layer-state wipe helper, checkpoint-poll helper) and seven
//! `e2e_*_test` stub files that the sibling Subtasks (`.rt`, `.cm`,
//! `.nx`, `.w4`, `.4q`, `.rh`, `.r1`) replace with per-AC test bodies.
//! Each `e2e_*` file targets one of Epic AC #7–#10:
//!
//! - [`e2e_update_layer_recovery_test`] — `t3.fgj.ya.wg.rt` (AC #7).
//! - [`e2e_subscribe_read_layer_recovery_test`] — `t3.fgj.ya.wg.cm` (AC #7).
//! - [`e2e_stat_layer_recovery_test`] — `t3.fgj.ya.wg.nx` (AC #7).
//! - [`e2e_concurrent_recovery_test`] — `t3.fgj.ya.wg.w4` (AC #8).
//! - [`e2e_object_id_preservation_test`] — `t3.fgj.ya.wg.4q` (AC #9).
//! - [`e2e_live_mapping_guard_test`] — `t3.fgj.ya.wg.rh` (AC #10).
//! - [`e2e_cluster_recovery_test`] — `t3.fgj.ya.wg.r1` (AC #7 cluster
//!   gate).
//!
//! ## Task `t2.fgj.7v.o7` — End-to-end session-recovery integration tests
//!
//! Subtask `t3.fgj.7v.o7.z7` lands the [`helpers`] additions
//! (`session_recovery_metric`, `wait_for_recovery_metric`,
//! `full_valkey_wipe` — the three already-landed Task 4/6 helpers
//! `wipe_session_state`, `wait_for_session_checkpoint`, and
//! `build_session_recovery_hook_for_e2e` are reused unchanged) and eight
//! `e2e_session_*_test` stub modules that the sibling Subtasks
//! (`.hr`, `.c6`, `.6f`, `.xt`, `.c4`, `.u6`, `.wz`, `.v9`) replace with
//! per-AC test bodies. Each module targets one Epic AC:
//!
//! - [`e2e_session_checkpoint_lifecycle_test`] — `t3.fgj.7v.o7.hr` (AC #1).
//! - [`e2e_session_full_wipe_recovery_test`] — `t3.fgj.7v.o7.c6` (AC #2).
//! - [`e2e_session_recovery_atomicity_test`] — `t3.fgj.7v.o7.6f` (AC #3).
//! - [`e2e_session_recovery_path_index_atomicity_test`] —
//!   `t3.fgj.7v.o7.xt` (AC #4).
//! - [`e2e_session_recovery_orphan_sweep_tolerance_test`] —
//!   `t3.fgj.7v.o7.c4` (AC #5).
//! - [`e2e_session_recovery_admission_test`] — `t3.fgj.7v.o7.u6` (AC #6).
//! - [`e2e_session_list_sessions_recovery_test`] — `t3.fgj.7v.o7.wz`
//!   (AC #7).
//! - [`e2e_session_cluster_recovery_test`] — `t3.fgj.7v.o7.v9` (Cluster
//!   gate).
//!
//! ## Task `t2.fgj.mh.zp` — End-to-end DR restart-storm integration test
//!
//! Subtask `t3.fgj.mh.zp.2w` (this test-infrastructure scaffold) lands
//! the [`restart_storm_harness`] module that two sibling Subtasks
//! consume:
//!
//! - `t3.fgj.mh.zp.vy` (positive) — pins Epic 6 AC #4: 100 sequential
//!   restarts × ~50 `CreateLayer` calls under
//!   `safety_margin = 1000` (a test value below the 1,000,000 production
//!   default) MUST produce 5000 unique `object_id`s.
//! - `t3.fgj.mh.zp.d8` (negative control) — same loop with
//!   `safety_margin = 0` MUST surface at least one collision within
//!   10 iterations. Proves the safety margin is what's preventing
//!   collisions, not coincidence.
//!
//! Neither test body lives in the harness module itself — see the
//! per-Subtask `restart_storm_test.rs` (lands in
//! `t3.fgj.mh.zp.vy`) that the harness module is wired against.
//!
//! ## b.b2j — runtime object-id-counter repair
//!
//! - [`counter_repair_test`] — the surviving-pods keyspace-loss shape.
//!   `counter_seed_test` covers the one-shot startup seed; this module covers
//!   what happens when the keyspace is lost *after* startup, which the
//!   liveness probe does not restart the pod for. Its first test is the
//!   fails-without-the-fix regression guard (pre-b.b2j the first post-wipe
//!   `CreateLayer` minted `object_id = 1`).
//! - [`e2e_counter_repair_test`] — the same failure driven through
//!   `handle_create_layer` against the fully-wired [`helpers::CheckpointHarness`].
//!   The sibling module builds its own repair hook in every test, so it cannot
//!   catch a *wiring* regression; this one asserts the hook is installed by
//!   observing behaviour only a hooked `LayerStore` produces.
//!
//! ## b.dfh Task `t2.dfh.gy.4o` — kind-aware restore-source selection
//!
//! - [`recovery_mixed_kind_selection_test`] — Subtask `t3.dfh.gy.4o.h6`:
//!   FR-6 restore-source selection driven end-to-end through
//!   `handle_stat_layer` against a **LocalDisk** store holding real `.tail`
//!   artifacts. Covers what the hook-level unit tests (InMemory) structurally
//!   cannot: LocalDisk stem parsing, mixed-kind listing order, and meta
//!   round-trip. Two shapes — tails contiguous onto the newest base (SR-3.1:
//!   restore the composed lineage, leave the tails intact) and a tails-only
//!   store (observably identical to a never-checkpointed layer, with no
//!   `error` tick). Its first test asserted SR-2.5's superseded
//!   newest-base-wins rule until Epic `t1.dfh.yj` swapped selection; its
//!   module header records the before/after.
//!
//! ## b.dfh Task `t2.dfh.57.1u` — kind-aware wait helpers + tail fixtures
//!
//! Subtask `t3.dfh.57.1u.y6` lands the [`helpers`] surface the three sibling
//! tail suites and Epic `t1.dfh.yj` consume, plus the harness self-guards that
//! keep it honest:
//!
//! - `wait_for_checkpoint{,_at_least}` select a **base**; the newest generation
//!   of any kind is [`helpers::wait_for_any_kind_checkpoint_at_least`].
//! - The hand-put generation builders (`put_base_generation` /
//!   `put_tail_generation`, both filling their payload from
//!   `common::empty_reset_ompf`), lifted out of
//!   [`recovery_mixed_kind_selection_test`] and parameterised on
//!   `&Arc<dyn CheckpointStore>` so the storage-side `checkpointer_*` suites
//!   share them.
//! - `drive_captured_tail` — the one composable step producing a REAL captured
//!   tail through the live loop, as opposed to a hand-put tail's synthetic
//!   records.
//! - The tail metric readers (`read_tail_counter`, `read_tail_writes_counter`,
//!   `gather_tail_outcome_cells`) and their label-discipline banner. Epic 5's
//!   SR-7 adjudication added a third whole-family gatherer beside them,
//!   `helpers::gather_base_bootstrap_cells` (`live_edit_checkpoint_base_bootstrap_total`,
//!   keyed `(outcome, site)`) — same discipline, and the same reason for parsing
//!   the family WHOLE: an out-of-band label value surfaces as an EXTRA key rather
//!   than being invisible to a named-cell probe.
//! - [`helpers::CheckpointHarnessOptions`] — one harness body behind the four
//!   named constructors, carrying the SR-7.1 first-base bootstrap wired to the
//!   harness's own store under the production three-way gate.
//!
//! The Task's own test module on this side of the tree (its two siblings are
//! storage-side `checkpointer_tail_*` suites):
//!
//! - [`e2e_tail_bootstrap_test`] — Subtask `t3.dfh.57.1u.e6` (Part 1): PRD AC-5
//!   end-to-end. The SR-7.1 bootstrap driven through the real `CreateLayer`
//!   commit path and observed through the real periodic loop — success (base at
//!   `snap_seq = 1`, both cursor columns in one HSET, then a tail rooted at it),
//!   failure (`refused_no_base` with nothing stamped, then normal checkpointing
//!   once the store heals), and the flag-flip rollout caveat. Deliberately NOT
//!   feature-gated; the module header says why.
//!
//! ## b.dfh Task `t2.dfh.yj.wg` — restore-lineage selection
//!
//! - [`restore_lineage_selection_test`] — Subtask `t3.dfh.yj.wg.e2`: Epic 4's
//!   `gather_restore_lineage` driven **directly** against a
//!   `LocalDiskCheckpointStore` over a `tempfile::tempdir()` — selection only
//!   (no handler, no harness, no restore) and Valkey-free by construction.
//!   Restates the gather contract over real on-disk artifacts, where stem
//!   parsing, mixed-kind listing order, and meta JSON round-trip meet it: the
//!   head lineage in replay order over a mixed-kind listing, a **real** torn
//!   `.tail.meta.json` residual gathering off the envelope header instead,
//!   a broken link stepping down to the base with one `StepDownRecord`, and
//!   tails-only / empty both resolving to `Ok(None)` rather than an `Err`.
//!
//! ## b.dfh Task `t2.dfh.yj.b7` — composed restore through the recovery hook
//!
//! - [`e2e_composed_restore_test`] — Subtask `t3.dfh.yj.b7.x7`: the *wiring*
//!   layer between Task 2's selection tests and Task 6's proof matrix — gather
//!   → compose → ext pre-store → one `write_object`, observed through
//!   `handle_stat_layer` against a **real-commit lineage** (loop-written base,
//!   tail hand-encoded from the exact OMPF bytes real `UpdateLayer`s sent).
//!   Five tests: the composed round trip with a faithful-column oracle and the
//!   four cursor columns, the base-only regression fence, both step-down arms
//!   (to a base and to an older composed lineage), the span-source
//!   discrimination that pins the guard band's header sourcing, and the
//!   `RecoveryOutcome` vocabulary guard.
//! - [`e2e_composed_restore_ext_order_test`] — Subtask `t3.dfh.yj.b7.74`: the
//!   SR-3.4 half of the same wiring — the tail-carried external values are
//!   pre-stored **strictly before** the single `write_object`, which is what
//!   makes an interrupted restore repairable at all (`apply_checkpoint`'s
//!   EXISTS short-circuit forecloses post-write repair). Three tests: a
//!   recording decorator over the real `ValkeyExternalValueStore` proving the
//!   pre-store fired and saw no `obj:hdr`, one firing of the
//!   `RESTORE_SEAM_AFTER_EXT_PRESTORE` seam proving the durable pair (values
//!   present, object absent), and a characterization of the accepted
//!   ref-count-per-restore over-count. The AC-8 injection matrix and
//!   retry-convergence are Task 6's (`t2.dfh.yj.u3`), not this file's.
//!
//! ## b.dfh Task `t2.dfh.yj.4f` — SR-3.6 post-restore gap re-base
//!
//! - [`restore_rebase_test`] — Subtask `t3.dfh.yj.4f.gw`: every restore seeds
//!   `seq` a whole band above the generation it restored (`band >= 1`
//!   unconditionally, post-b.8y5), so SR-3.6's old conditional no-gap arm is
//!   dead and the re-base runs on base-only restores too. Nine tests: the
//!   landing shape on the base-only, surviving-stream and composed lineages
//!   (one `Base` at `seeded_seq`, all THREE cursor columns, `restore_seq_bump`
//!   RETIRED to `0`, `{success,layer}` +1, and no tail-outcome cell moved); the
//!   three restore-site cursor-guard probes (the restore's own non-zero
//!   `last_checkpoint_seq` must NOT abort — a verbatim copy of the
//!   creation-time guard would kill every re-base; a mid-flight higher cursor
//!   leaves blob-present + every column untouched; an unparseable column
//!   declines entirely); the PUT-failure arm over both classifications, where
//!   the correct claim is columns UNCHANGED (carrying the restore's own stamp),
//!   never "absent"; and the fossil-free property driven through
//!   `capture_layer_tail` directly — success rooting at `seeded_seq`, failure
//!   refusing `PostRestoreUnrebased` → `refused_seq_gap` on BOTH stream shapes.
//!   Deliberately NOT feature-gated; the module header says why.
//! - [`bootstrap_concurrency_bound_test`] — Subtask `t3.dfh.yj.4f.yd`, which
//!   also discharges Epic-3 interim hazard #4: `BaseBootstrap::run_once`'s
//!   admission semaphore (`try_acquire_owned` — acquire-or-shed, NEVER queue),
//!   read from the `BOOTSTRAP_MAX_INFLIGHT` `pub const` rather than spelled as a
//!   literal. A `DelegatingCheckpointStore` newtype carries an in-flight
//!   high-water mark and a shared release gate, so every claim rests on a
//!   structural peak counter and a total PUT count — never on elapsed time.
//!   Eight tests: the bound honoured under `bound × 4` simultaneous triggers
//!   (peak `== bound`, so the probe is provably non-vacuous); shed-vs-queue
//!   discriminated by the total PUT count read AFTER the gate opens and
//!   everything drains; no spurious shed at exactly `bound`; ONE bound spanning
//!   BOTH call sites, at a shrunk bound and again at the production bound, with
//!   one `Arc<BaseBootstrap>` `Arc::clone`d to the two holders exactly as
//!   `main.rs` wires them; the shed's terminality through the `spawn_creation_base`
//!   wrapper and the shared `TaskTracker`; the metric fence pinning the landed
//!   posture that a shed moves NO cell of `checkpoint_writes_total` and no
//!   tail-outcome cell; and the `with_max_inflight(0)` clamp the shrunk-bound
//!   cases stand on. Deliberately NOT feature-gated; the module header says why.
//!
//! ## b.dfh Task `t2.dfh.yj.u3` — Epic 4's proof burden
//!
//! - [`golden_tail_replay_test`] — Subtask `t3.dfh.yj.u3.tp` (PRD AC-6 /
//!   SR-6.2): COMMITTED persisted-tail artifacts under
//!   `tests/fixtures/tail_lineage/`, built from real captured Kit traffic
//!   (`tools/fixtures/gze_inbound_deltas.json`) by the committed generator
//!   `examples/generate_tail_lineage_goldens.rs`, replayed through
//!   `compose_from_artifacts` and asserted byte/state-stable. The only evidence
//!   class in the Task whose inputs are frozen bytes, so it is the one that
//!   survives future refactors: a change to apply/codec semantics that alters
//!   replay of a stored artifact fails CI and forces an explicit decision.
//!   Six tests — the composed-state gate on the faithful column basis, the
//!   chained two-generation identity, the envelope byte-stability re-encode,
//!   the §8.4 `EmptyDelta`-record non-vacuity guard, the SR-6.1
//!   unknown-`format_version` arm (specific variant + index attribution +
//!   `Permanent` step-down), and the orphan-fixture meta-test. Valkey-free
//!   (`compose_from_artifacts` is pure) and deliberately NOT feature-gated; the
//!   module header carries the gate contract, the regeneration command, and the
//!   recorded mutation verification.
//! - [`restore_fault_injection_matrix_test`] — Subtask `t3.dfh.yj.u3.uc`
//!   (PRD AC-8): the restore-interruption matrix. Fourteen arms — a classified
//!   mid-gather fetch error (propagates verbatim, NO lineage cell moves), a
//!   missing member (`stepped_down_link_broken`), a rejected composition
//!   (`stepped_down_compose`), the `RESTORE_SEAM_AFTER_EXT_PRESTORE` seam, a
//!   failed SR-3.6 re-base PUT, TWO degraded-adapter arms — two wedged members
//!   over a generous budget (the selection-wide `LinkFact::Wedged` cache: a
//!   wedged member is addressed at most once for the whole selection) and one
//!   wedged member at the SHIPPED `budget == 2 × reserve` ratio (b.bu8: the
//!   base's fetch is reserved out of the budget by member kind, so the per-member
//!   cost is kind-dependent rather than one reserve per member, and the floor
//!   answer survives) — the three b.4et
//!   stream-adjudication seams, and four b.f4t residue-wipe arms (one after the
//!   wipe, three INSIDE its `WATCH` … `EXEC` window via
//!   `RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`: a peer's complete header fencing
//!   the restore, a single-field write yielding `Transient`, and the seam's own
//!   refusal) — each asserting the two claims this file owns: no
//!   partially-applied object is ever observable, and the retry converges to the
//!   exact expected state once the fault clears. Composed and base-only paths
//!   both covered — plus a fifteenth arm (b.hxk) where the budget runs out with
//!   the healthy base UNREAD, which is the one budget shape that ends
//!   `exhausted_budget` → retriable `UNAVAILABLE` rather than a floor restore.
//!   Module UNGATED; the eight injector-driven arms are
//!   `#[cfg(feature = "test-checkpoint-fault-injection")]` and the module header
//!   says why that is a silent activation condition. The deadline-driven
//!   `stepped_down_budget` producer's exact ARITHMETIC stays delegated by
//!   citation to `t3.dfh.yj.wg.xn`'s unit coverage; arm 15 drives its
//!   client-visible consequence end-to-end.
//! - [`composed_restore_ext_matrix_test`] — Subtask `t3.dfh.yj.u3.ad` (PRD
//!   AC-3): the external-value presence/absence matrix on one composed restore,
//!   and the only place both ext paths run against one restored layer — the
//!   eager SR-3.4 tail-carried pre-store and the lazy, pre-existing SR-3.2
//!   base-referenced fetch-on-miss. Four tests over one lineage carrying all
//!   three classes plus a hash referenced by base-and-tail and one by two tail
//!   commits: a mirrored base reference recovering byte-identically
//!   (`recovered` +1, refill, and the proof it was NOT pre-stored), an
//!   unrecoverable one leaving the restore intact and missing only at read time
//!   (`adapter_miss` + the original `MetaNotFound`, reference and size
//!   unchanged), tail-carried values reachable with `get_ext_blob` NEVER called
//!   (the claim that separates a working pre-store from an accidental lazy
//!   recovery), and the mixed-lineage completeness partition. Fences against
//!   `e2e_composed_restore_ext_order_test`'s ordering probe and `ref_count`
//!   characterization.
//! - [`e2e_full_wipe_composed_restore_test`] — Subtask `t3.dfh.yj.u3.ue`
//!   (Epic AC): the Epic's criterion end to end. A `full_valkey_wipe` plus ONE
//!   production-handler dispatch restores a tail-lineage layer to exact state
//!   (composed-head cursor, faithful columns, replayed versions, byte-identical
//!   ext values, path index, counter re-seed); the restored never-quiescent
//!   layer's NEXT tail capture then SUCCEEDS through the real periodic loop,
//!   refuses observably as `refused_seq_gap` when the SR-3.6 re-base PUT fails,
//!   and converges once a base lands. Both stream shapes: fresh (whole-keyspace
//!   wipe) and surviving (`wipe_layer_state_preserving_stream`, fossils folded).

//!
//! ## b.dfh Task `t2.dfh.be.5s` — the fence on the shared harness's wiring
//!
//! - [`harness_parity_test`] — Subtask `t3.dfh.be.5s.gk` (Issue b.kfg): six
//!   tests asserting that the STOCK harness's recovery hook is the one
//!   `src/main.rs:731-738` ships, so the next harness edit cannot silently drop
//!   a production builder call the way b.kfg did for a whole Epic. One
//!   structural (`Arc::ptr_eq` on the single `BaseBootstrap`) and four
//!   behavioural, because `RecoveryHook` exposes no accessor for its optional
//!   collaborators: a composed restore's tail-carried ext blob materialising in
//!   Valkey fences `.with_ext_value_store(..)`, a landed re-base fences
//!   `.with_base_bootstrap(..)`, and the opt-out plus the derived three-way gate
//!   fence the two ways "no re-base" is legitimately correct; a sixth re-runs
//!   the landed-re-base assertion through `install_recovery_hook`, the second
//!   entry point into the same wiring, which the other five never observe.
//!
//! ## b.dfh Task `t2.dfh.be.84` — the chain-collapse re-base
//!
//! - [`rebase_refusal_test`] — Subtask `t3.dfh.be.84.j7` (SR-4.5): the refusal
//!   arms of `rebase_layer_lineage`, driven **directly** against a
//!   `LocalDiskCheckpointStore` over a `tempfile::tempdir()` behind a recording
//!   `DelegatingCheckpointStore` newtype that can fail one targeted
//!   `(snap_seq, kind)` member fetch — which the `CheckpointFaultInjector` seam
//!   cannot express. Six tests: a middle tail's fetch failing and the base's
//!   fetch failing (both `RebaseRefusal::Gather`, proving the arm is not
//!   tail-specific), the retry-next-tick convergence that separates "refused"
//!   from "wedged", the compose arm with its SR-4.5 suspect-lineage tripwire
//!   asserted through a `current_thread` log capture on a distinctive substring
//!   plus the offending generation's structured fields, the stepped-down-walk
//!   arm (`LineageBroken`) asserting the tripwire stays SILENT and
//!   `record_compose_failure` is never called (design decision D6), and the
//!   `Permanent` classification that forbids retrying a compose failure against
//!   the same bytes. Every refusal asserts one shared five-part intactness
//!   claim: the variant, an empty mutating-call log, an identical listing, an
//!   identical artifact enumeration with no `Temp` member, and four unmoved
//!   `obj:hdr` cursor columns. Scope fence: the ext-backfill refusal arm is
//!   `t3.dfh.be.84.p4`'s (it needs that Subtask's ext-bearing fixture), the
//!   `BasePut` arm and every metric claim are the loop-level lane's
//!   (`t3.dfh.be.84.1k`) because the unit deliberately holds no metrics handle.
//!   Deliberately NOT feature-gated; the module header says why.
//! - [`rebase_ext_backfill_test`] — Subtask `t3.dfh.be.84.p4` (SR-4.3): the
//!   ext-value backfill as a HARD precondition for the base PUT, driven directly
//!   against the same LocalDisk-behind-a-recording-newtype shape the refusal lane
//!   uses, but with the two mutating calls under test appended to ONE interleaved
//!   op log — `put_ext_blob` recorded on COMPLETION (durable acknowledgement is
//!   what SR-4.3 is about), `put_layer_checkpoint` recorded at ISSUE — so the
//!   ordering claim is the recorded-index comparison `max(PutExt) < PutLayer`
//!   rather than a bare count. The fixture lineage (`base@100` + three tails,
//!   K = 3) carries three hash classes: a cross-commit duplicate referenced by two
//!   tail commits, three singly-referenced tail-inline parts, and a **base-only**
//!   hash the base references with zero OMPF parts. Seven tests: the ordering
//!   proof, completeness against `ComposedLineage::ext_values` (with dedup and the
//!   D4 residual asserted in both directions — the base-only hash is never
//!   backfilled yet is still sidecar-listed), the two hard-precondition rows
//!   (first part fails / last part fails, each proving no base PUT anywhere, an
//!   intact lineage, and a retry that lands), idempotency by hash across a second
//!   re-base, byte-identical readback of every backfilled blob, and the elapsed
//!   per-blob-timeout row proving a SLOW adapter takes the same `ExtBackfill` arm
//!   a failing one does. Valkey-free by construction: the unit takes
//!   `&dyn CheckpointStore` and nothing else, and the cursor-immobility fence is
//!   the refusal lane's arm-independent claim. No metric assertions, for the same
//!   no-metrics-handle reason.
//!
//! ## b.dfh Task `t2.dfh.be.s8` — the SR-7 observability adjudication
//!
//! - [`bootstrap_observability_test`] — Subtask `t3.dfh.be.s8.fa`: one case per
//!   `BaseBootstrap::run_once` arm that the adjudication gave a cell on the new
//!   `live_edit_checkpoint_base_bootstrap_total{outcome,site}` family, each
//!   asserting the returned `BootstrapOutcome` variant FIRST (non-vacuity), then
//!   the adjudicated cell's exact delta, then flat-ness of every sibling cell of
//!   the 9 × 2 grid, then flat-ness of the whole `checkpoint_tail_total{outcome}`
//!   family, then the `checkpoint_writes_total{result, kind="layer"}` posture in
//!   both directions. Ten tests: the RESTORE-site shed, guard (a), the vanished
//!   header, the untrusted cursor column, guard (b)'s seq and bump halves (the
//!   two arms with no coverage above a predicate unit test before this Task), the
//!   reachable HMGET half of `unstamped_write_error`, the failure arm, the
//!   tails-only `refused_lineage_unrootable` capture verdict, and the SUM
//!   IDENTITY — one tick per `run_once` invocation, the property every ratio in
//!   the adjudication depends on. Scope fences: the creation-site
//!   `{shed, creation_base}` / `{persisted, creation_base}` deltas belong to
//!   `bootstrap_concurrency_bound_test.rs`'s case 6, and the `metrics.rs`
//!   `#[cfg(test)]` vocabulary pins to the vocabulary lane. Deliberately NOT
//!   feature-gated; the module header says why.
//!
//! ## b.dfh Task `t2.dfh.be.hm` — the long-running hot-layer lifecycle
//!
//! - [`e2e_rebase_lifecycle_test`] — Subtasks `t3.dfh.be.hm.eg` /
//!   `t3.dfh.be.hm.q2`: **eg** — exact-state fidelity across ONE loop-level
//!   SR-4.1 chain-collapse re-base. A real-commit `1 base + K tails` lineage
//!   (K = 2, every range carrying structural + inline + time-sample records and
//!   the last one an external value) with the oracle pinned at the lineage head
//!   `S`; a hot writer committing ABOVE `S` keeping the layer due, while the
//!   test-only unprovable-cut seam has every assembly's proven read judged
//!   regressed (the only way a proven read skips a layer; compiled only under
//!   `test-checkpoint-fault-injection`); the REAL periodic loop reaching the
//!   capture fallback, refusing
//!   `ChainFull` and collapsing the chain into ONE base at `S`
//!   (`tail_total{rebased}` +1, `writes_total{success, layer}` +1,
//!   `last_checkpoint_seq = S`, superseded lineage left intact); then
//!   `wipe_layer_state` + `handle_stat_layer`, which resolves `gathered_base`
//!   (chain 0 — the collapse worked, read as a whole-vocabulary before/after
//!   delta) and reproduces the oracle on the consolidated superset
//!   faithful-column basis, with the SR-4.3 ext backfill read back
//!   byte-identically out of the checkpoint store. "No generation above `S` but
//!   the re-base's own base" is STRUCTURAL, not a poll: a `RebaseFenceStore`
//!   newtype admits exactly `Base@S` while armed and its op log is asserted.
//!   Scope fences: the composition ENGINE is covered for free by
//!   `tests/fixtures/tail_lineage/` + `proptests.rs::ac4_versions_above` and is
//!   cited, not re-proven; the ext-backfill ORDERING / completeness claims are
//!   `t3.dfh.be.84.p4`'s; retention is Task 4's; the tick-by-tick multi-cycle
//!   half is in-crate (`t3.dfh.be.hm.hq`); and
//!   `e2e_full_wipe_composed_restore_test`'s four SR-3.6 arms stay untouched.
//!   **q2** — interleaving B, in the same file: a DR restore landing
//!   MID-lifecycle on a lineage at `j = 1 < K`. `wipe_layer_state` +
//!   `handle_stat_layer` under the stock production hook wiring; the SR-3.6
//!   re-base mints a lone `Base@seeded_seq`, retires `restore_seq_bump` and ticks
//!   `base_bootstrap_total{persisted, restore_rebase}` +1 — and that base then
//!   JOINS the lineage arithmetic: the loop captures K tails whose
//!   `base_snap_seq` is `seeded_seq`, an SR-4.1 chain collapse fires on THAT
//!   lineage (`rebased` +1), and at `R = 2` retention retires the pre-DR lineage
//!   WHOLE — base and tail together, no stranded tail. Three binding deferrals
//!   discharged: **cell separation** (`rebased` flat across the SR-3.6 step while
//!   `writes_total{*, layer}` moves, read as a whole-`CHECKPOINT_TAIL_OUTCOMES`
//!   before/after map ITERATED from the production const, never a hardcoded
//!   count), the **third base producer** (a restore-minted lone base is its own
//!   retention unit and displaces an older lineage), and **restore cost bounded**
//!   (one `gather_restore_lineage` taken with the store frozen at chain-K pays
//!   exactly `1 + K` `get_layer_checkpoint` calls; one taken after the lifecycle
//!   is still inside that bound). Recorded finding, since FIXED under this Task:
//!   a collapse over a chain the LOOP captured used to skip its own cursor stamp
//!   — and therefore its own retention prune — because the head tail's capture
//!   had already stamped the same sequence and the shared `>=` guard fired on
//!   equality. `collapse_cursor_blocks_stamp` (strict `>`) now lets the collapse
//!   stamp, tick `writes_total{success, layer}` and prune on its own tick; the
//!   arm's assertions pin the fixed contract rather than narrate the defect.

//! ## Issue `b.4t3` — slot-pinned `txn_pool` checkouts in the DR-restore applier
//!
//! - [`acquire_txn_pinning_fence_test`] — the lane's source-shape fence: no
//!   unpinned `acquire_txn()` anywhere under `src/checkpoint/`, and exactly the
//!   two `acquire_txn_for(` pins SR-2.7's two-step applier requires. Valkey-free
//!   and feature-free. The slot arithmetic behind those pins, and the
//!   behavioural proof that the hygiene probe runs on the applier's own
//!   checkouts, are in-crate in `src/checkpoint/session.rs`'s test module
//!   (`apply_session_checkpoint` is `pub(crate)`).
//!
//! ## Issue `b.wsx` — the external-value backup repair
//!
//! - [`ext_backup_repair_test`] — the backup confirmation on an external
//!   value's meta (`backed_up_in`) and the repair pass that re-sends the copies
//!   the ext-mirror never confirmed, against a real Valkey and a recording
//!   backup store: copies lost to a shutdown discard, a failed upload or a
//!   `budget_full` drop re-sent; the classes (backed, young, owed, audit) and
//!   another store's identity; the confirm script's contract; the shared byte
//!   budget; leadership; the audit; per-value and pass-ending failures; the
//!   concurrency bounds; the mirror worker's confirmation; and passes that end
//!   early. The collect's clear of the field is a `checkpointer` unit test.

pub mod helpers;

mod acquire_txn_pinning_fence_test;
mod adapter_in_memory_test;
mod adapter_local_disk_test;
mod adapter_storage_api_401_refresh_test;
mod adapter_storage_api_test;
mod assembler_test;
mod atomicity_local_disk_test;
mod atomicity_session_local_disk_test;
mod authoring_rank_snapshot_recovery_test;
// b.v6d: the cursor stamp's two ARGV shapes (the loop's plain stamp, the
// bootstrap's floored one) and the floor closing the read→stamp window two
// overlapping bootstrap runs for one object could both pass.
mod b_v6d_stamp_floor_test;
mod bootstrap_concurrency_bound_test;
mod bootstrap_observability_test;
mod composed_restore_ext_matrix_test;
mod conformance;
mod counter_repair_test;
mod counter_seed_test;
mod e2e_cluster_recovery_test;
mod e2e_composed_restore_ext_order_test;
mod e2e_composed_restore_test;
mod e2e_concurrent_recovery_test;
mod e2e_counter_repair_test;
mod e2e_ext_value_orphan_sweep_reconcile_test;
mod e2e_ext_value_recovery_test;
mod e2e_full_wipe_composed_restore_test;
mod e2e_live_mapping_guard_test;
mod e2e_object_id_preservation_test;
mod e2e_rebase_lifecycle_test;
mod e2e_restore_over_surviving_residue_test;
mod e2e_session_checkpoint_lifecycle_test;
mod e2e_session_cluster_recovery_test;
mod e2e_session_full_wipe_recovery_test;
mod e2e_session_list_sessions_recovery_test;
mod e2e_session_recovery_admission_test;
mod e2e_session_recovery_atomicity_test;
mod e2e_session_recovery_concurrent_handlers_test;
mod e2e_session_recovery_no_checkpoint_test;
mod e2e_session_recovery_orphan_sweep_tolerance_test;
mod e2e_session_recovery_path_index_atomicity_test;
mod e2e_session_recovery_per_handler_test;
mod e2e_stat_layer_recovery_test;
mod e2e_storage_api_recovery_test;
mod e2e_subscribe_read_layer_recovery_test;
mod e2e_tail_bootstrap_test;
mod e2e_update_layer_recovery_test;
mod ext_backup_repair_test;
mod ext_mirror_e2e_test;
mod first_base_bootstrap_test;
mod golden_tail_replay_test;
mod harness_parity_test;
mod lifecycle_delete_session_test;
mod lifecycle_disabled_adapter_test;
mod lifecycle_session_publish_watermark_test;
mod lifecycle_session_writes_error_test;
mod lifecycle_shutdown_join_test;
mod rebase_ext_backfill_test;
mod rebase_refusal_test;
mod reconcile_dead_object_test;
mod recovery_admission_test;
mod recovery_hook_test;
mod recovery_mixed_kind_selection_test;
mod restart_storm_harness;
mod restart_storm_test;
mod restore_fault_injection_matrix_test;
mod restore_lineage_selection_test;
mod restore_permanent_refusal_test;
mod restore_rebase_test;
mod restore_write_fence_test;
