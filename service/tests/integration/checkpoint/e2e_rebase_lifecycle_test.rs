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

//! The hot-layer re-base **lifecycle**, at loop level — Subtasks
//! `t3.dfh.be.hm.eg` and `t3.dfh.be.hm.q2` under Task `t2.dfh.be.hm`, Epic
//! `t1.dfh.be`.
//!
//! Two claims share this file and share its fixture builder, its structural
//! fence and its harness wiring. Tests 1–2 are **eg**'s: exact-state fidelity
//! across ONE chain-collapse re-base. Test 3 is **q2**'s: the lifecycle
//! CONTINUES across a DR restore landing in the middle of it. Each has its own
//! claim section below; where a section names no Subtask it applies to both.
//!
//! ## The claim (Tests 1–2, `t3.dfh.be.hm.eg`)
//!
//! A restore taken **after** the periodic loop's own SR-4.1 chain-collapse
//! re-base equals the **pre-re-base live state at the same sequence**, on the
//! consolidated superset faithful-column basis. The subject is the LOOP-level
//! composition — capture → re-base → retention → restore — not the composition
//! engine underneath it.
//!
//! ## What is fenced for FREE, and is therefore NOT re-proven here
//!
//! The composition ENGINE already has two independent regression fences, both
//! landed, both cheaper and sharper than anything this file could add:
//!
//! - the committed golden fixtures under `tests/fixtures/tail_lineage/`
//!   (`golden_tail_replay_test.rs`), which pin the byte-level replay of a base +
//!   tail lineage against recorded artifacts; and
//! - the AC-4 live-backend property `tests/proptests.rs::ac4_versions_above`,
//!   which pins the per-entry version algebra over generated lineages.
//!
//! This file **cites** both and asserts neither. What no landed test covers, and
//! what this file is for, is that *the loop's own re-base produces a base a
//! production restore path can restore from*, byte-for-byte on the faithful
//! columns.
//!
//! Two tests, one per side of that comparison:
//!
//! 1. [`a_loop_level_rebase_restores_to_the_pre_rebase_state_at_s`] — the
//!    headline. The real periodic loop collapses a `1 base + K tails` chain into
//!    ONE base at `S`, and a production-handler restore from it reproduces the
//!    pre-re-base oracle at `S`.
//! 2. [`the_same_lineage_before_the_rebase_restores_the_composed_content_at_s`]
//!    — the same fixture with no collapse: it resolves `gathered_composed`
//!    rather than `gathered_base`, and reproduces the SAME oracle on the SAME
//!    basis. That is what makes Test 1's `gathered_base` evidence that a chain
//!    was collapsed rather than evidence that there was never a chain.
//!
//! ## The claim (Test 3, `t3.dfh.be.hm.q2`) — interleaving B
//!
//! 3. [`the_lifecycle_continues_across_a_dr_restore_onto_the_restore_minted_base`]
//!    — a DR restore lands MID-lifecycle, on a lineage at `j = 1 < K`. SR-3.6
//!    mints a lone `Base@seeded_seq`; that base then **joins the lineage
//!    arithmetic** — the loop captures `K` tails rooted at it, an SR-4.1
//!    chain-collapse re-base fires on THAT lineage, and retention retires the
//!    pre-DR lineage whole rather than stranding its tail.
//!
//! Three things only this arm can show, each a binding Epic deferral:
//!
//! - **Cell separation.** SR-3.6's post-restore re-base is a WRITE only:
//!   `checkpoint_writes_total{*, layer}` moves and `tail_total{rebased}` does
//!   NOT, that cell being SR-4.1's alone. A lifecycle in which BOTH re-bases
//!   occur is the only place the separation is falsifiable, and it is read as a
//!   whole-vocabulary before/after map ([`assert_tail_deltas`]) rather than as a
//!   probe on one cell.
//! - **The third base producer.** SR-3.6 mints a base on every restore, and a
//!   lone base is its own retention unit, so it consumes a retain slot. Only a
//!   lifecycle drive can show the consequence: at `R = 2` the pre-DR lineage
//!   goes out of the window, and the assertion that matters is that it goes out
//!   WHOLE.
//! - **Restore cost bounded, measured at the bound.** One
//!   `gather_restore_lineage` taken with the store frozen at chain-`K` pays
//!   exactly `1 + K` `get_layer_checkpoint` calls, and one taken after the
//!   lifecycle is still inside that bound.
//!
//! Test 3 also turned up a loop-level finding — a collapse over a chain the
//! loop CAPTURED used to skip its own cursor stamp, and therefore its own write
//! cell and its own retention prune, deterministically. The production fix is
//! `checkpointer::collapse_cursor_blocks_stamp`; the equality case it turns on
//! is stated on the test and pinned by its assertions rather than left for a
//! reader to re-derive.
//!
//! ## The version column — where the replay witness lives, and where it does not
//!
//! Both tests call `assert_restored_entry_versions_flattened`, and neither reads
//! the persisted versions as a replay witness, because since b.2ti there is
//! nothing there to read. `apply_restore` re-stamps the three versions a diff
//! builder FILTERS on — `Section::last_version` / `Field::version` /
//! `TimeSample::version` — onto `seeded_seq` after composition and before
//! `write_object`: a restore is ONE write of the WHOLE object, it preserves
//! `object_id` so surviving subscribers do not re-bootstrap, and
//! `src/domain/diff.rs`'s `version > have_version` filter would otherwise
//! withhold the restored content from exactly the reconnectors the restore
//! exists to correct.
//!
//! So on those three columns the two tests are IDENTICAL — a single value — and
//! the collapse is invisible in them. What each still asserts is the flattening
//! itself plus the presence half: every entry the `(S0, S]` range wrote must
//! come back. The in-order-replay witness — that the tail records were replayed
//! in order rather than merely all applied — lives on the two lanes that read
//! composition's output BEFORE the restore persists it:
//! `src/checkpoint/compose.rs`'s in-crate `versions_above` tests and
//! `tests/proptests.rs`'s `ac4_versions_above` property, both comparing
//! in-memory composed state against live. Neither is touched by the re-stamp.
//!
//! **`Section::first_version` is the fourth column and the exception.** The
//! re-stamp deliberately spares it — `plan_faithful_coalesced`'s abandoned-prim
//! retraction arm fires on `first_version <= have_version`, so raising it would
//! strand the orphan sections a restored base carries as unretractable — so it
//! comes back at whatever produced the payload. That is the ONE version column
//! in which the two tests still differ, and it differs by decode-vs-replay
//! rather than by content: Test 1 restores a collapsed artifact, one reset OMPF,
//! so every section decodes to `1`; Test 2's composed gather replays the
//! tail-created sections at their real sequences. Each test names its own shape
//! through `RestoredFirstVersions` (`DecodedBase` / `ComposedLineage`) and the
//! helper asserts it exactly. It is not an IN-ORDER witness — it says nothing
//! about the order the records were applied in — but on Test 2 it is a second,
//! independent detector of a dropped tail: a base-only compose would bring the
//! tail-created sections back at `1` where the oracle's own sequence is
//! expected. The presence half above stays the primary arm for that.
//!
//! **History, kept because the pre-b.2ti asymmetry is what shaped this file.**
//! Before the re-stamp the persisted versions DID differ between the two tests,
//! and it was measured rather than assumed: the old
//! `assert_replayed_entry_versions` failed on Test 1 at
//! `field (100000, "tail-field-0") … left: 1, right: 4` — the collapsed base's
//! decoder flattening every entry to `version = 1` against the live commit's
//! version 4 — while passing on Test 2 over the identical fixture, oracle and
//! range, and while every other assertion in Test 1, the faithful columns
//! included, passed. That is also free one-directional mutation evidence that
//! the faithful-column comparison and the version comparison are independent, so
//! neither can be passing because the other is.
//!
//! ## Recorded mutation verification
//!
//! The headline claim is easy to assert vacuously — a comparison against an
//! oracle is green whenever both sides are wrong the same way — so it was
//! mutation-checked in both halves. Each mutation was REVERTED and none is
//! committed: every mutation was a self-contained insert delimited by an
//! explicit marker comment, the reverts restored the surrounding lines verbatim,
//! and the file was re-run green afterwards. (For M1/M2, which edited THIS file,
//! a whole-file SHA-256 before/after is NOT the evidence and is not claimed —
//! this section itself was written after the reverts, so the file legitimately
//! differs. M3 edited production code and a SHA-256 IS claimed for it.)
//!
//! - **M1 — base-only compose, against Test 1.** Deleting every generation above
//!   `S0` immediately before the wipe leaves `base@S0` as the lineage, which is
//!   content-identical to what a re-base that DROPPED THE TAILS would have PUT
//!   (both assertions under test read state and versions only, so the
//!   substitution is exact for them). [`assert_faithful_columns`] goes red on its
//!   FIRST arm — sections — naming precisely the five structural commands the
//!   tail ranges carried:
//!
//!   > `post-re-base restore at S: restored sections must equal the oracle on`
//!   > `(parent_id, name, section_type, section_order)`
//!   > … `left` carries sections `1..=120` only; `right` additionally carries
//!   > `100000: (1, "tail-section-0", 2, 0)`, `100001: (1, "tail-section-1", 2, 1)`,
//!   > `100002: (1, "tail-section-2", 2, 2)`, `100003: (1, "tail-section-3", 2, 3)`,
//!   > `100004: (1, "tail-section-4", 2, 4)`
//!
//!   Five, not six, and that is the fixture being read back correctly: the sixth
//!   tail-range commit is the ext-value one, which carries a `SetField` and no
//!   structural command. The helper's field and time-sample arms sit BEHIND the
//!   sections arm in the same function and diverge on the same missing content;
//!   the sections arm is simply the one that fires first.
//! - **M2 — the same base-only compose against Test 2**, with that test's lineage
//!   sweep and faithful-column call temporarily suppressed so the version arm is
//!   reached at all. The version helper (then
//!   `assert_replayed_entry_versions`, now
//!   [`assert_restored_entry_versions_flattened`], whose presence arm carries
//!   the same claim and the same message) goes red:
//!
//!   > `pre-re-base composed restore at S: field (100000, "tail-field-0") was`
//!   > `written above the base at version 4 and MUST exist after the restore`
//!
//! - **The version probe** described in the section above is a third, independent
//!   result in the opposite direction: on the UNMUTATED, pre-b.2ti Test 1 the
//!   faithful columns passed while the version comparison failed
//!   (`left: 1, right: 4`), so the two comparisons cannot be covering for each
//!   other.
//! - **M3 — the SR-3.6 post-restore re-base made to tick `rebased`, against
//!   Test 3** (`t3.dfh.be.hm.q2`). This is the mutation the cell-separation
//!   deferral was filed against, and before Test 3 no landed test could detect
//!   it. A five-line insert at `BaseBootstrap::run_once`
//!   (`src/checkpoint/base_bootstrap.rs`, the single tick site) ticking
//!   `inc_checkpoint_tail("rebased")` on the `RestoreRebase` × `Persisted` arm.
//!   Step 3's whole-vocabulary zero-delta goes red:
//!
//!   > `across the SR-3.6 post-restore re-base (EVERY tail outcome must be flat:`
//!   > `this step writes a base, it captures nothing and it collapses nothing):`
//!   > `` `live_edit_checkpoint_tail_total{outcome="rebased"}` MUST move exactly ``
//!   > `0 over this window, moved 1. … before {…, "rebased": 0.0, …}, after`
//!   > `{…, "rebased": 1.0}` — `left: 1.0, right: 0.0`
//!
//!   Reverted byte-identically: `sha256(src/checkpoint/base_bootstrap.rs)` reads
//!   `4b7de599…0204f278` both before the edit and after the revert, and
//!   `git diff` on that file is empty. Note the failure text carries the WHOLE
//!   ten-cell map on both sides, which is the point of reading it as a map — the
//!   same insert pointed at any other outcome would be just as red, where a probe
//!   naming `rebased` would not have looked.
//! - **M4 — the two `RestoredFirstVersions` variants SWAPPED between the two
//!   tests** (b.2ti round 3), verifying that the one version column the restore
//!   still preserves is asserted non-vacuously and that the two tests genuinely
//!   differ in it. Both go red, in opposite directions, on the tail-created
//!   sections:
//!
//!   > Test 2 under `DecodedBase`: `pre-re-base composed restore at S: section`
//!   > `100000 MUST come back with first_version PRESERVED at 1 (DecodedBase,`
//!   > `lineage root 3, oracle 4); it read 4` — `left: 4, right: 1`
//!   >
//!   > Test 1 under `ComposedLineage`: `post-re-base restore at S: section 100000`
//!   > `MUST come back with first_version PRESERVED at 4 (ComposedLineage,`
//!   > `lineage root 3, oracle 4); it read 1` — `left: 1, right: 4`
//!
//!   That is the decode-vs-replay difference measured rather than asserted: the
//!   collapsed artifact really does bring `100000` back at the decoder's `1`
//!   while the composed gather really does replay it at its live sequence `4`.
//!   Both edits were reverted; the swap touched only this file's two call
//!   arguments and `git diff` shows them at their landed variants.
//!
//! ## The load-bearing premise: the re-base PUTs at `S`, not at the live head
//!
//! The chain-collapse re-base composes the **existing lineage** and PUTs its new
//! base at `S` = the *lineage* head sequence (`rebase_layer_lineage` in
//! `src/checkpoint/rebase.rs`, called from `rebase_on_chain_full` in
//! `src/checkpoint/checkpointer.rs`). It does **not** read the live layer and it
//! does not PUT at the live head.
//!
//! Two consequences, and together they are the whole design of the fixture:
//!
//! 1. A tick whose assembly persists nothing is *required* — otherwise the tick
//!    writes an ordinary base and the capture fallback that owns the
//!    `ChainFull` → re-base arm is never reached. Since every read proves its
//!    cut (Spec SDD SR-4.1, SR-11.5) only a header-sequence regression skips an
//!    assembly, and the test-only unprovable-cut seam [`spawn_tail_loop`] arms
//!    is what stages one. A hot writer committing **above** `S` keeps the layer
//!    due on every tick (the skip-if-no-commits gate).
//! 2. That writer's commits are **not in the composed base**. So the correct
//!    comparison target is an oracle pinned at `S` *before the writer starts* —
//!    which is exactly what [`RebaseLineageFixture::head_oracle`] is.
//!
//! Get this backwards (compare against the live state at the end of the drive)
//! and the test fails for a reason the design guarantees, every time.
//!
//! ## Why the REAL periodic loop, and never `rebase_layer_lineage` directly
//!
//! Calling the unit directly is Task 3's narrow instrument
//! (`rebase_refusal_test.rs`, `rebase_ext_backfill_test.rs` — both drive
//! `rebase_layer_lineage` against a `&dyn CheckpointStore` with no Valkey at
//! all). Substituting it here would void the claim: the *loop-level* composition
//! is the assembly persisting nothing, the capture unit returning
//! `ChainFull`, `rebase_on_chain_full`'s two gates, the compose, the ext
//! backfill, the cursor stamp and the prune — none of which a direct call
//! exercises. This file spawns a real [`spawn_checkpoint_loop`] and polls the
//! loop's own verdict off the metric family.
//!
//! ## "No generation above `S` except the re-base's own base" is STRUCTURAL
//!
//! Enforced by [`RebaseFenceStore`], a [`DelegatingCheckpointStore`]-based
//! newtype that — once armed — rejects every layer PUT except the one the
//! re-base is expected to make. Not a poll-and-hope: an earlier draft of the
//! Epic-4 precedent (`e2e_full_wipe_composed_restore_test.rs`, its module
//! header) flaked at roughly one run in eight on exactly this class of race, and
//! the failure mode here is worse than a flake — a single ordinary base landing
//! above `S` resets the chain to 0 and the re-base can then never happen in that
//! run at all. See the type's own rustdoc for the two arms and the error variant.
//!
//! ## Boundary against `e2e_full_wipe_composed_restore_test.rs`
//!
//! That file's **four arms stay untouched by both Subtasks**, and the file is
//! unmodified. It owns SR-3.6's *post-restore* re-base (the fresh `Base` a DR
//! restore mints at `seeded_seq`) under a `full_valkey_wipe`, and its module
//! header states in so many words that the SR-4.1 chain-collapse re-base is
//! structurally out of reach there and that nothing in it asserts `rebased`.
//! This file is the other half of that split: it is where `rebased` is asserted,
//! over a `wipe_layer_state` partial loss and a surviving session. The two
//! re-bases are different units with different triggers; where this file says
//! "re-base" unqualified it always means SR-4.1's, and SR-3.6's is always named
//! in full.
//!
//! Test 3 EXTENDS one specific arm over there —
//! `a_restored_never_quiescent_layer_captures_its_next_tail_instead_of_refusing`
//! (its Test 2), which proves the FIRST hop of a mid-lifecycle restore (a
//! restored never-quiescent layer captures its next tail instead of losing DR
//! coverage forever) and stops there. None of that arm's assertions are re-made
//! here; Test 3 starts where it ends and carries the lineage on to `K` and
//! through a collapse. Its Test 3, the failed-re-base `refused_seq_gap` refusal
//! arm, is landed there and is deliberately not re-derived here either.
//!
//! Everything reusable is CONSUMED from Task 2's consolidated surface
//! (`helpers.rs`): the harness and its options, `install`-time ext-store wiring,
//! `put_tail_generation_from_records`, `wait_for_checkpoint_at_least`,
//! `wipe_layer_state`, `DelegatingCheckpointStore`, `gather_tail_outcome_cells`,
//! `gather_base_bootstrap_cells`, `read_restore_lineage_counter`,
//! `read_layer_writes_counter`, `read_tail_writes_counter`,
//! `assert_faithful_columns`, `versions_above`. There is deliberately **no**
//! file-local re-implementation of any of them — a fourth copy is precisely what
//! Task 2 exists to prevent. `helpers.rs` is UNMODIFIED by both Subtasks: every
//! piece either already existed on Task 2's surface or is specific to this
//! file's fixture and belongs here.
//!
//! ## The tick-by-tick half of this Task lives IN-CRATE
//!
//! Subtask `t3.dfh.be.hm.hq` drives `run_one_tick` directly in
//! `src/checkpoint/checkpointer.rs`'s test modules and owns the multi-cycle
//! capped-chain / restore-cost accounting. This lane is integration-scoped
//! because its claim needs live Valkey state (`read_object_state_for_test`), the
//! production recovery hook, a real `ValkeyExternalValueStore` and Task 2's
//! consolidated faithful-column assertions — none of which exist in-crate.
//!
//! ## Retention: a variable in Test 3 and deliberately not in Tests 1–2
//!
//! Tests 1–2 spawn their loop at [`DRIVE_RETAIN_GENERATIONS`] (8), far above any
//! unit count their fixtures reach, so the prune never leaves its early return
//! and retention is not a variable in a failure there.
//!
//! Test 3 does the opposite ON PURPOSE, at [`LIFECYCLE_RETAIN_GENERATIONS`] (2,
//! the production default): its whole third-base-producer claim is that the
//! restore-minted base consumes a retain slot and displaces an older lineage, and
//! that is unobservable at any quota the fixture cannot reach.
//!
//! What neither re-proves is Task 4's own material: the per-prune ORDERING
//! (tails newest-first, base last), the orphan bucket, and the re-base tie being
//! TWO units so that a single collapse at `retain_generations = 2` prunes NOTHING
//! are all pinned at unit and store level by `t2.dfh.be.mk`. Test 3 asserts the
//! JOINT lifecycle consequence — no stranded tail, a lineage still gathered
//! cleanly, restore cost still inside `1 + K` — and escalates rather than softens
//! if any of them fails.
//!
//! ## Synchronization + budget
//!
//! No `tokio::time::sleep` is used as synchronization anywhere: every wait is a
//! bounded poll on observable state under a NAMED module constant, and the loop
//! drain gets ONE deadline rather than one per message.
//!
//! | Wait | Bound | What a timeout means |
//! |------|-------|----------------------|
//! | `wait_for_session_checkpoint{,_quiesced}` | [`STORE_DEADLINE_SECS`] (10 s, clamped up to the helpers' 30 s load-tolerant floor) | the session-checkpoint writer never landed a blob / never drained |
//! | `wait_for_checkpoint_at_least` (the lineage-root base) | [`STORE_DEADLINE_SECS`], same clamp | the periodic loop never assembled a base at or above the fixture's committed sequence |
//! | `wait_for_writer_idle` (post-restore SR-3.6 drain) | [`STORE_DEADLINE_SECS`], same clamp | a `BaseBootstrap` task never completed, or was spawned onto a different tracker |
//! | loop drain after `cancel()` | [`LOOP_DRAIN_BUDGET`] (5 s) | a wedged loop; bounded so it fails promptly instead of hanging the binary |
//! | [`HotWriter::warm_up`] | [`WARM_UP_DEADLINE`] (20 s) | the writer never committed — the panic prints its own report |
//! | "the loop re-based" poll | [`REBASE_DEADLINE`] (25 s) | the loop never reached the `ChainFull` → re-base arm; the panic prints the listing, the fence's op log and the whole outcome-cell map |
//! | "the chain filled to K and the collapse is held" poll (Test 3) | [`CHAIN_FILL_DEADLINE`] (30 s) | the restored layer never captured K tails onto the restore-base, or never reached the `ChainFull` arm at all; the panic names which outcome cell moved instead |
//! | "retention retired the pre-DR lineage" poll (Test 3) | [`PRUNE_DEADLINE`] (15 s) | the collapse never reached `prune_after_write`, or the prune ran and did not retire the over-quota unit |
//! | "a capture landed on the collapsed base" poll (Test 3) | [`PRUNE_DEADLINE`] (15 s) | the lifecycle did not resume on the new base, leaving the restore-cost identity below vacuous |
//! | `gather_restore_lineage` per-fetch reserve (Test 3) | [`GATHER_FETCH_RESERVE`] (30 s) | an in-memory fetch outlived 30 s; a step-down here would land on `stepped_down_budget` and mask the cost claim |
//!
//! Every bound above is a FAILURE bound: on the happy path each wait returns in
//! one or two 1 s ticks. Each timing-out wait panics with its own diagnostic, so
//! the practical time-to-diagnostic is ONE bound, never their sum.
//!
//! Test 1's waits are: two session-checkpoint waits, one
//! `wait_for_checkpoint_at_least`, the fixture's loop drain, the writer warm-up,
//! the re-base poll, the drive's loop drain, and one post-restore
//! `wait_for_writer_idle`. Stated at the NOMINAL `STORE_DEADLINE_SECS` that sums
//! to ~95 s; stated at the helpers' clamped 30 s load-tolerant floor it sums to
//! ~175 s. **Test 3 is the larger one**: the same fixture waits, plus one
//! post-restore `wait_for_writer_idle`, the writer warm-up, the chain-fill poll,
//! the re-base poll, the prune and re-capture polls and THREE loop drains —
//! ~160 s nominal, ~240 s at the clamped floor. Both are ABOVE `.config/nextest.toml`'s 120 s
//! hard kill, which prints nothing at all when it fires. That is exactly why no
//! individual bound here is generous: the design intent is that at most ONE wait
//! ever times out and reports itself, and a run that managed to stack several
//! clamped store waits would be hard-killed silently instead. If a bound is ever
//! raised, raise the argument with it.
//!
//! **This is a CONSIDERED DIVERGENCE from the precedent, not a continuation of
//! it.** The landed file this one otherwise follows,
//! `e2e_full_wipe_composed_restore_test.rs`, deliberately kept its own summed
//! failure budget UNDER the kill — its header records "~65 s of failure budget
//! plus fixture setup — inside `.config/nextest.toml`'s 120 s hard kill". The
//! sum above is knowingly the other side of that line, on the
//! one-wait-ever-times-out argument just made. So a future e2e file must NOT
//! cite this header as precedent for a sum above the kill: it has to re-make
//! that argument on its own waits, or size them the way the full-wipe file did.
//!
//! Measured on the post-fix tree — the `collapse_cursor_blocks_stamp` fix and
//! the re-capture poll it added both in — over ten consecutive clean runs (two
//! batches of five; `cargo nextest run -j 4 -p live-edit --test integration
//! --features test-support checkpoint::e2e_rebase_lifecycle_test`, the three
//! tests running concurrently): Test 1 2.075–3.167 s, Test 2 1.476–2.498 s,
//! Test 3 7.381–8.460 s, whole file 7.382–8.461 s wall clock. 30/30 test
//! executions PASS, zero FLAKY (`.config/nextest.toml` sets `retries = 1`, so a
//! TRY-1-fail/TRY-2-pass would be reported FLAKY rather than passed — none
//! was). Every test sits far inside the 40 s slow mark and the 120 s hard kill.
//! Test 3 is the long one because it is the only arm that
//! waits on the LOOP several separate times — two captures, a collapse,
//! retention and one resumed capture, each at the 1 s tick cadence — and every
//! one of those is a real tick, not a poll spinning on a race.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; under the
//! project sandbox that variable is preset (CLAUDE.md).
//!
//! # The named Epic-AC evidence index (`t3.dfh.be.hm.85`)
//!
//! Additive: nothing above was reflowed. This block is IDENTICAL to the one at
//! the end of `mod rebase_lifecycle_tests`' header in
//! `src/checkpoint/checkpointer.rs`, and is repeated here so either file
//! answers the question on its own. The Epic
//! close-out sweep (`t2.dfh.be.vz`) reads it; it names functions rather than
//! describing them.
//!
//! **Scope: deliberately partial.** Epic `t1.dfh.be` has FOUR acceptance
//! clauses; Task `t2.dfh.be.hm` evidences ONE of them (clause 1, in its three
//! parts) plus the two claims the Task body adds. The closing table names the
//! owners of the other three, so a reader does not read this as a complete
//! index with three rows missing.
//!
//! Two files carry the Task's six tests. Repository-relative paths, stated once:
//!
//! - **IN-CRATE** —
//!   `src/checkpoint/checkpointer.rs`,
//!   `mod rebase_lifecycle_tests`.
//! - **E2E** —
//!   `tests/integration/checkpoint/e2e_rebase_lifecycle_test.rs`
//!   (this file).
//!
//! | Clause | Evidenced by | File |
//! |---|---|---|
//! | Epic clause 1a — *the chain capped at K* | `hot_layer_lifecycle_holds_the_chain_at_k_across_two_rebase_cycles_and_advances_coverage` (its per-cycle `assert_chain_capped` calls, listing count AND true link walk, at every rise of both cycles) AND `restore_cost_stays_within_one_base_plus_k_tails_at_every_point_of_two_rebase_cycles` (`chain_len <= K` at all seven enumerated points, plus the cap sample on EVERY tick of each collapse drive) | IN-CRATE |
//! | Epic clause 1b — *re-base generations appearing* | `hot_layer_lifecycle_holds_the_chain_at_k_across_two_rebase_cycles_and_advances_coverage`, phase B (`phase_b_collapse`): `tail_total{rebased}` +1, `writes_total{success, layer}` +1 beside `{skipped, layer}` +1, a `Base` at `S`, twice — once per cycle | IN-CRATE |
//! | Epic clause 1c — *restore cost bounded* | `restore_cost_stays_within_one_base_plus_k_tails_at_every_point_of_two_rebase_cycles` (via `sample_restore_cost`: one listing, `gets == 1 + chain_len`, no step-down) | IN-CRATE |
//! | Task-body claim — *both interleavings at loop level* | **A**, verified snapshot superseding mid-lifecycle (SR-4.4): `a_verified_snapshot_superseding_mid_chain_re_roots_the_lineage_and_the_lifecycle_reaches_a_later_rebase` — IN-CRATE. **B**, a DR restore landing mid-lifecycle (SR-3.6): [`the_lifecycle_continues_across_a_dr_restore_onto_the_restore_minted_base`] — E2E | both |
//! | Task-body claim — *exact-state fidelity across a re-base cycle* | [`a_loop_level_rebase_restores_to_the_pre_rebase_state_at_s`] and its no-collapse sibling [`the_same_lineage_before_the_rebase_restores_the_composed_content_at_s`] | E2E |
//!
//! ## The rest of the Epic AC, and who owns it
//!
//! | Epic clause | Owner |
//! |---|---|
//! | 2 — *no interleaving (incl. the injected multi-replica AC-7 races) ever deletes a base referenced by a retained tail* | Task 4, `t2.dfh.be.mk` |
//! | 3 — *the re-base-idempotency duplicate-leader test is green* | Task 3's `pa` lane, `t3.dfh.be.84.pa` |
//! | 4 — *cumulative docs updated per the Plan Bee's doc-impact list* | Task 7, `t2.dfh.be.vz` |
//!
//! ## Lever-live / no-attempt-count audit, for this file's half
//!
//! Recorded in full in the in-crate block; the half that is about THIS file:
//! [`setup_rebase_harness`] is the single harness builder, it sets
//! `opts.tail.enabled = true`, and it ASSERTS the flag reached `harness.config`
//! as a fixture precondition — so the SR-4.2 lever is live in all three tests
//! by construction. Nothing here reads `stable_cut_total` or
//! `checkpoint_lease_total`, and no assertion presumes an
//! attempt count; the `attempts` identifier in this file is
//! [`RebaseFenceStore`]'s record of PUT calls the store SAW, a structural
//! ledger, never a retry count. The only `Instant` reads are bounded-poll
//! failure deadlines, and [`poll_until`] re-evaluates its predicate once more
//! AFTER the deadline, so a slow machine loses no verdict.

// Tests 1 and 3 are compiled only under `test-checkpoint-fault-injection` (the
// unprovable-cut seam `spawn_tail_loop` arms), so without the feature the
// fixtures only they use read as dead. Test 2 runs in every shape.
#![cfg_attr(
    not(feature = "test-checkpoint-fault-injection"),
    allow(dead_code, unused_imports)
)]

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tonic::Request;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    decode_tail_envelope, gather_restore_lineage, no_deadline, spawn_checkpoint_loop,
    CheckpointAdapter, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointKind,
    CheckpointLoopConfig, CheckpointMeta, CheckpointStore, LayerArtifact, TailCaptureConfig,
};
use live_edit::domain::types::FieldValue;
use live_edit::grpc::layer::{handle_create_layer, handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::{CreateSessionRequest, StatLayerRequest};
use live_edit::grpc::session::handle_create_session;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::{
    SharedMetrics, CHECKPOINT_RESTORE_LINEAGE_OUTCOMES, CHECKPOINT_TAIL_OUTCOMES,
};
use live_edit::storage::diff_read::ObjectSnapshot;
use live_edit::storage::external_values::{ext_content_hash, ValkeyExternalValueStore};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::read_object_state_for_test;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{
    CommandDiffChangeSection, OwnedDelta, SetField, StructCommand, TimeSample as OodfTimeSample,
};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;

// ─────────────────────────────────────────────────────────────────────────────
// Named waits and fixture constants — see the module header's budget table
// ─────────────────────────────────────────────────────────────────────────────

/// Checkpoint-loop tick cadence. `1` is the value every e2e recovery file in
/// this module uses: the harness sets `interval = quiesce = 0`, so a
/// just-written layer is due on the very next tick.
const TICK_SECS: u64 = 1;

/// Deadline handed to every `helpers::wait_for_*` call. Those helpers clamp it
/// up to their own 30 s load-tolerant floor, so this is the *lower* bound on
/// how long a contended Valkey may take before the wait is called a failure.
const STORE_DEADLINE_SECS: u64 = 10;

/// Bound on the post-`cancel()` drain of a checkpoint loop's `JoinHandle`.
const LOOP_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// Ceiling on [`HotWriter::warm_up`].
const WARM_UP_DEADLINE: Duration = Duration::from_secs(20);

/// Ceiling on the "the loop collapsed the chain" poll.
const REBASE_DEADLINE: Duration = Duration::from_secs(25);

/// Poll cadence for every bounded poll in this file. Not a synchronization
/// sleep: the predicate is observable state, and the cadence only bounds how
/// hard the loop hammers Valkey while it waits.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// **K** — the tail-chain bound this fixture drives the loop to.
///
/// Two, deliberately small. It has to be a real CHAIN (a one-tail lineage would
/// make `tail_{n+1}.from_seq == tail_n.to_seq` unobservable and would collapse
/// the "composed from a chain" claim into "composed from a tail"), and every
/// extra link costs a whole tail-range of driven commits for nothing new. The
/// capture unit's bound comparison is `chain_len >= max_chain` over the count of
/// generations above the newest base, so exactly K hand-put tails puts the
/// fixture ON the bound and the very first eligible tick takes the re-base arm.
const MAX_CHAIN: usize = 2;

/// Commits the hot writer must land before the periodic loop is allowed to run.
///
/// A state precondition polled off the writer's own counter, not a timing hack:
/// a loop that meets a layer with no commit above its chain head finds it not
/// due (the skip-if-no-commits gate) and the drive waits on a gate rather than
/// on the arm under test.
const WARM_UP_COMMITS: u64 = 20;

/// Sections seeded into the fixture layer.
///
/// 120 is `helpers::drive_captured_tail`'s number. Which arm a tick takes is
/// the unprovable-cut seam's ([`spawn_tail_loop`]), not the width's; the width
/// makes the exact-state oracle non-trivial.
const WIDE_SECTIONS: u64 = 120;

/// Stream retention for the hot writer's own `LayerStore`.
///
/// Deliberately far above the harness's 4096, matching the Epic-4 precedent: the
/// writer commits continuously for as long as the drive runs, and a `MAXLEN`
/// trim reaching past a capture window's floor would show up as an unrelated
/// refusal rather than as the arm under test.
const WRITER_STREAM_MAXLEN: u64 = 1_000_000;

/// Retention quota for the loop this file spawns. Well above anything planted
/// here: the landed prune counts LINEAGE UNITS, and eight is far past any unit
/// count this fixture reaches, so the prune never leaves its early return. See
/// the module header — retention is Task 4's claim, not this file's.
const DRIVE_RETAIN_GENERATIONS: usize = 8;

/// **R** — the retention quota Test 3's loop runs at, and the OPPOSITE posture to
/// [`DRIVE_RETAIN_GENERATIONS`] on purpose.
///
/// Two, which is the production default (`LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS`)
/// and — more to the point — the value at which the Epic's **third base producer**
/// actually displaces something. Test 3's lifecycle ends with THREE lineage units
/// in the listing (the collapse's chain-0 `base@S'`, the restore-base unit
/// `base@seeded_seq + K tails`, and the pre-DR unit `base@S0 + tail@S`), so at
/// `R = 2` the post-collapse prune retires the pre-DR unit — which is precisely
/// the consequence `retention.rs`'s own header states for an operator ("a DR
/// restore immediately demotes the pre-restore lineage to the last retained
/// slot"). At [`DRIVE_RETAIN_GENERATIONS`] nothing would ever be pruned and the
/// third-base-producer assertions would all be vacuously green.
const LIFECYCLE_RETAIN_GENERATIONS: usize = 2;

/// **j** — the tail-chain length Test 3's fixture reaches BEFORE the DR event.
///
/// One: strictly under [`MAX_CHAIN`], which is the Subtask's literal requirement
/// (`j < K`). A fixture already at `K` would collapse on the first eligible tick,
/// so there would be no un-collapsed mid-lifecycle lineage for the restore to
/// land in the middle of — and the restore would then be gathering a chain-0
/// base rather than the composed lineage a mid-lifecycle DR event actually meets.
const PRE_DR_TAIL_RANGES: usize = 1;

/// Ceiling on Test 3's "the loop filled the restore-base chain to K and is being
/// held at the collapse" poll.
const CHAIN_FILL_DEADLINE: Duration = Duration::from_secs(30);

/// Ceiling on each of Test 3's two post-collapse polls: "retention retired the
/// superseded pre-DR lineage", and "the lifecycle resumed with a capture on the
/// collapsed base".
///
/// The prune runs inside the SAME tick that collapsed the chain
/// (`rebase_on_chain_full`'s tail call to `prune_after_write`), but `rebased`
/// ticks BEFORE the cursor stamp and therefore before the prune — so the first
/// wait is for the remainder of one tick, or for the next one if the drain cut
/// that tick short. The second is for one capture at the 1 s loop cadence.
const PRUNE_DEADLINE: Duration = Duration::from_secs(15);

/// Per-fetch reserve handed to Test 3's `gather_restore_lineage` calls.
///
/// Only ever a FAILURE bound here: both gathers run against an in-memory delegate
/// with no loop competing for it, so every fetch returns immediately. It exists
/// because the parameter is not optional, and it is generous because a budget
/// step-down would show up as a `stepped_down_budget` lineage outcome and mask the
/// cost claim the fetch count is making.
const GATHER_FETCH_RESERVE: Duration = Duration::from_secs(30);

/// The section every hot-writer commit and every pre-base commit writes into.
/// Section `1` exists in every layer this file creates.
const ROOT_SECTION_ID: u64 = 1;

/// First section id minted by a tail-range commit. Each such commit creates its
/// own section, so the composed replay has structural work to do and not only
/// field writes.
const TAIL_SECTION_BASE_ID: u64 = 100_000;

/// Authoring rank stamped on every `SetField` this file's fixture builds.
/// Only needs to be stable; nothing here depends on a particular value.
const AUTHORING_RANK: u64 = 0;

/// Byte length of the external value one tail range carries. Large enough to be
/// unambiguously an out-of-band OMPF part, small enough to stay free.
const EXT_BLOB_LEN: usize = 4 * 1024;

/// The one-based OMPF part index the ext-carrying `SetField` points at. Part 0
/// is the OODF delta; `delta_to_ompf_bytes(delta, &[blob])` puts `blob` at
/// Part 1.
const EXT_VALUE_INDEX: u32 = 1;

/// Commits driven inside EACH tail range.
///
/// Three, not one: every range must carry structural commands, inline fields AND
/// a time sample, and a single-record envelope makes an off-by-one in the
/// record/span relationship unobservable.
const TAIL_RANGE_COMMITS: u64 = 3;

/// The `key_name` of the field the ext-carrying commit writes. Asserted by name
/// after the restore so the ext half cannot pass vacuously on a layer that lost
/// the reference entirely.
const EXT_FIELD_KEY: &str = "tail-ext-field";

const SESSION_BASE_PATH: &str = "/stage/rebase-lifecycle";
const CLIENT_ID: &str = "rebase-lifecycle-client";
const WRITER_CLIENT_ID: &str = "rebase-lifecycle-hot-writer";

// ─────────────────────────────────────────────────────────────────────────────
// Payload builders
// ─────────────────────────────────────────────────────────────────────────────

/// CreateLayer payload: [`WIDE_SECTIONS`] sections under the root plus one
/// inline field each. See [`WIDE_SECTIONS`] for why the width is load-bearing.
fn wide_initial_delta() -> OwnedDelta {
    let mut struct_commands = Vec::with_capacity(WIDE_SECTIONS as usize);
    let mut set_fields = Vec::with_capacity(WIDE_SECTIONS as usize);
    for section_id in 1..=WIDE_SECTIONS {
        struct_commands.push(StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id,
            section_name: format!("rebase-lifecycle-section-{section_id}"),
            section_type: 1,
            section_order: (section_id - 1) as u32,
        }));
        set_fields.push(SetField {
            section_id,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAA, (section_id & 0xFF) as u8, 0xCC],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        });
    }
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands,
        set_fields,
        ..Default::default()
    }
}

/// A pre-base commit: one new inline field on the root section.
///
/// `base_version: 1` (any non-zero) marks this an INCREMENTAL diff. With
/// `base_version == 0` and `is_diff == true` the domain engine takes the reset
/// branch and wipes prior state, which would empty the fixture.
fn pre_base_commit_delta(index: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: format!("pre-base-field-{index}"),
            value_or_ext_hash: vec![0x10 + index as u8, 0x11, 0x12],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// A tail-range commit: a NEW section under the root, one inline field inside
/// it, and one time sample.
///
/// All three record kinds in one delta on purpose. A composed re-base that
/// replayed field writes but dropped structural commands (or vice versa) would
/// still satisfy a fields-only oracle; it cannot satisfy this one.
///
/// `index` is GLOBAL across the ranges (range n contributes indices
/// `n * TAIL_RANGE_COMMITS ..`), so no two ranges collide on a section id.
fn tail_commit_delta(index: u64) -> OwnedDelta {
    let section_id = TAIL_SECTION_BASE_ID + index;
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ROOT_SECTION_ID,
            section_id,
            section_name: format!("tail-section-{index}"),
            section_type: 2,
            section_order: index as u32,
        })],
        set_fields: vec![SetField {
            section_id,
            key_name: format!("tail-field-{index}"),
            value_or_ext_hash: vec![0xC0 + index as u8, 0xDE, index as u8],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        time_samples: vec![OodfTimeSample {
            section_id,
            time: index as f64,
            value_or_ext_hash: vec![0xF0 + index as u8],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    }
}

/// The tail-range commit that carries an EXTERNAL VALUE.
///
/// This is what puts the value bytes into the tail record's OMPF Parts 1..N,
/// which is the only thing SR-4.3's re-base backfill has to work with:
/// `compose_from_artifacts` collects `(hash, Bytes)` pairs off exactly these
/// parts, and `rebase_layer_lineage` `put_ext_blob`-acks that list as a HARD
/// precondition before its single base PUT.
fn tail_ext_value_delta(hash: &[u8; 32], blob_len: usize) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: EXT_FIELD_KEY.to_string(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: blob_len as u64,
            ext_value_index: EXT_VALUE_INDEX,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// Deterministic external-value payload. An RNG would make a failing run
/// unreproducible for no benefit — the bytes only have to be non-trivial and
/// stable so `ext_content_hash` of them is the fixture's own contract.
fn ext_blob() -> Vec<u8> {
    (0..EXT_BLOB_LEN).map(|i| (i % 251) as u8).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// The structural fence
// ─────────────────────────────────────────────────────────────────────────────

/// One `put_layer_checkpoint` attempt, as the wrapper saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PutAttempt {
    object_id: u64,
    snap_seq: u64,
    kind: CheckpointKind,
    /// Whether the wrapper let it through to the delegate.
    accepted: bool,
}

/// `CheckpointStore` newtype that records every layer PUT and, once **armed**,
/// admits exactly ONE of them: the chain-collapse re-base's own `Base` at the
/// lineage head sequence `S`.
///
/// # Why a store-level newtype and not the fault-injection seam
///
/// Same reason as its Epic-4 sibling `RebasePutControlStore`
/// (`e2e_full_wipe_composed_restore_test.rs`): the `CheckpointFaultInjector`
/// seam dispatches on a method-name label — `put_layer_checkpoint` — which every
/// producer here shares, so it structurally cannot target one without the
/// others. A [`DelegatingCheckpointStore`]-based newtype keyed on
/// `(kind, snap_seq, armed)` can, and costs no cargo feature, which is what keeps
/// this module ungated.
///
/// # Targeting — SHARPER than the Epic-4 sibling, and deliberately so
///
/// `RebasePutControlStore` keys on `kind == Base` alone because SR-3.6's target
/// (`seeded_seq`) is not knowable before the restore commits. Here it IS
/// knowable: the chain-collapse re-base PUTs at `S` = the *lineage* head
/// sequence, which the fixture planted itself. So this fence keys on the exact
/// address, and its two arms are:
///
/// 1. **Every `Tail` PUT is rejected.** This is the Subtask's literal
///    requirement. Once the re-base has landed `Base@S` the chain is back to
///    zero, so the very next eligible tick would CAPTURE a tail above `S` — and
///    the listing's newest handle would then be that tail rather than `Base@S`,
///    silently changing which lineage the restore under test gathers.
/// 2. **Every `Base` PUT at a sequence other than `S` is rejected.** This is the
///    arm that makes the fixture deterministic. If an assembly is ever not
///    skipped — a tick before the unprovable-cut seam is armed, or one after it
///    is disarmed — it writes an ordinary base at the LIVE head; that base
///    resets the chain to zero and the re-base can then never be reached again
///    in that run — a lost run, not a retryable flake. Rejected, nothing lands,
///    the chain stays at K, and the next tick re-attempts.
///
/// The conjunction is the claim the test then reads off the op log: while armed,
/// the ONLY generation that can land is `Base@S`.
///
/// # The second mode — `arm_capture_only`, added by `t3.dfh.be.hm.q2`
///
/// [`Self::arm_capture_only`] is the CAPTURE-phase posture the lifecycle arm
/// needs and the exact-address arm above cannot express: **admit every `Tail`
/// PUT, reject every `Base` PUT.**
///
/// It exists for the same determinism reason as arm 2 above and closes the same
/// race from the other side. The lifecycle arm's chain is built by the loop's own
/// captures rather than hand-put, so the address the eventual collapse will PUT
/// at is not knowable until those captures have landed. Between the restore and
/// that moment an assembly that is not skipped would write an ordinary
/// base at the LIVE head, which resets the chain to zero AND re-roots the next
/// capture away from the restore-base — silently falsifying the very claim
/// (`base_snap_seq == seeded_seq`) the arm exists to make. Rejecting every base
/// makes that structurally impossible; the tick is simply lost and re-attempted.
///
/// The mode holds the collapse itself as well, which is the second thing it buys:
/// with the chain parked AT `K` and no base able to land, the checkpoint store is
/// FROZEN, so the arm can take its chain-at-`K` restore-cost measurement against a
/// store that cannot move under it. The held collapse is observable — every held
/// tick ticks `refused_chain_full` (SR-4.5's `RebaseRefusal::BasePut` fold) and
/// `checkpoint_writes_total{error, layer}` — and the arm asserts that rise as
/// positive evidence the loop reached the `ChainFull` arm rather than never
/// getting there. Releasing is then one call to [`Self::arm`] with the now-known
/// address.
///
/// The two modes are mutually exclusive; each setter clears the other.
///
/// # The error variant, and why the loop tolerates it
///
/// [`CheckpointError::Transient`]. Every rejected PUT reaches the loop as an
/// `Err` from the unit that issued it, and both consumers treat that as a
/// *retry-next-tick* refusal rather than as a fatal condition:
///
/// - A rejected **tail** PUT propagates out of `capture_layer_tail` (its PUT is
///   `store.put_layer_checkpoint(..).await?`, no retry loop — "the tick is the
///   retry"). `attempt_tail_capture_fallback`'s `Err` arm then WARNs, ticks
///   `checkpoint_writes_total{error, layer_tail}`, samples the layer as pending
///   and continues to the next layer. It moves **no** `checkpoint_tail_total`
///   cell — that family has no error cell by design — which is why the
///   `captured` flat-ness assertion below is not weakened by this fence.
/// - A rejected **base** PUT from the assembler arm is the same shape one tick
///   earlier, and likewise leaves the lineage completely intact.
///
/// Nothing is persisted on either, so the fixture the next tick observes is
/// byte-identical to the one this tick observed. That is what makes "reject and
/// let the loop retry" sound here, where "fail the loop" would not be.
///
/// # Delegation
///
/// Forwards **every** trait method, including the four with non-trivial trait
/// defaults — `read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`. Omitting those would not
/// leave a call un-instrumented, it would SUBSTITUTE behaviour behind the wrapper
/// and make assertions about it vacuously green (`read_ext_refs_sidecar`'s
/// default returns an EMPTY set; `get_layer_checkpoint`'s returns
/// `NotImplemented`, which would report "no such generation" to the restore-
/// source selection whose whole job is deciding what exists).
struct RebaseFenceStore {
    delegate: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    attempts: Mutex<Vec<PutAttempt>>,
    armed: AtomicBool,
    /// `t3.dfh.be.hm.q2`'s capture-phase mode: admit `Tail`, reject `Base`.
    /// Mutually exclusive with [`Self::armed`].
    capture_only: AtomicBool,
    /// The one address the fence admits while armed: `(object_id, S)`.
    fenced_object_id: AtomicU64,
    admitted_base_seq: AtomicU64,
    /// Every `get_layer_checkpoint` the wrapper has forwarded, for the
    /// restore-cost accounting `t3.dfh.be.hm.q2` reads around ONE
    /// `gather_restore_lineage`. Counted here rather than in a second recording
    /// newtype because this store is already the one the harness installs
    /// everywhere, and a second wrapper would be a fourth copy of the delegation
    /// boilerplate below for one counter.
    layer_fetches: AtomicU64,
}

impl RebaseFenceStore {
    fn new() -> Self {
        Self {
            delegate: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            attempts: Mutex::new(Vec::new()),
            armed: AtomicBool::new(false),
            capture_only: AtomicBool::new(false),
            fenced_object_id: AtomicU64::new(0),
            admitted_base_seq: AtomicU64::new(0),
            layer_fetches: AtomicU64::new(0),
        }
    }

    /// Arm the fence for `object_id`, admitting only `Base@head_seq`.
    fn arm(&self, object_id: u64, head_seq: u64) {
        self.fenced_object_id.store(object_id, Ordering::SeqCst);
        self.admitted_base_seq.store(head_seq, Ordering::SeqCst);
        self.capture_only.store(false, Ordering::SeqCst);
        self.armed.store(true, Ordering::SeqCst);
    }

    /// Arm the CAPTURE-phase mode for `object_id`: admit every `Tail`, reject
    /// every `Base`. See the type's rustdoc for why this is the only posture that
    /// can express "the loop builds the chain itself, and nothing may re-root it".
    fn arm_capture_only(&self, object_id: u64) {
        self.fenced_object_id.store(object_id, Ordering::SeqCst);
        self.armed.store(false, Ordering::SeqCst);
        self.capture_only.store(true, Ordering::SeqCst);
    }

    fn disarm(&self) {
        self.armed.store(false, Ordering::SeqCst);
        self.capture_only.store(false, Ordering::SeqCst);
    }

    /// Total `get_layer_checkpoint` calls forwarded so far. Read as a
    /// before/after delta around a single `gather_restore_lineage`, with no loop
    /// running, so the delta addresses that gather and nothing else.
    fn layer_fetches(&self) -> u64 {
        self.layer_fetches.load(Ordering::SeqCst)
    }

    fn attempts(&self) -> Vec<PutAttempt> {
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Every attempt recorded after `from_index` (a snapshot of
    /// `attempts().len()` taken before the act step).
    fn attempts_since(&self, from_index: usize) -> Vec<PutAttempt> {
        self.attempts().into_iter().skip(from_index).collect()
    }

    /// `true` when this PUT must be refused.
    fn rejects(&self, object_id: u64, snap_seq: u64, kind: CheckpointKind) -> bool {
        let armed = self.armed.load(Ordering::SeqCst);
        let capture_only = self.capture_only.load(Ordering::SeqCst);
        if !armed && !capture_only {
            return false;
        }
        if object_id != self.fenced_object_id.load(Ordering::SeqCst) {
            return false;
        }
        if capture_only {
            // Every Tail through, every Base refused — including the collapse's
            // own, which is what parks the chain at K.
            return matches!(kind, CheckpointKind::Base);
        }
        match kind {
            CheckpointKind::Tail => true,
            CheckpointKind::Base => snap_seq != self.admitted_base_seq.load(Ordering::SeqCst),
        }
    }
}

#[async_trait]
impl CheckpointStore for RebaseFenceStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        let rejected = self.rejects(object_id, snap_seq, meta.kind);
        // Recorded BEFORE the arms diverge: the test wants to know the call was
        // ISSUED, whatever happens to it afterwards.
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(PutAttempt {
                object_id,
                snap_seq,
                kind: meta.kind,
                accepted: !rejected,
            });

        if rejected {
            return Err(CheckpointError::Transient {
                source: format!(
                    "RebaseFenceStore refused {:?}@{snap_seq} for object {object_id} (SR-4.1 \
                     chain-collapse E2E fence: only Base@{} is admitted while armed)",
                    meta.kind,
                    self.admitted_base_seq.load(Ordering::SeqCst)
                )
                .into(),
            });
        }

        self.delegate
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would report "no such generation"
    // for a generation the delegate holds, to the restore-source selection whose
    // whole job is deciding what exists.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        // Counted before the forward, so a fetch that ERRORS is still a fetch the
        // restore-cost accounting paid for.
        self.layer_fetches.fetch_add(1, Ordering::SeqCst);
        self.delegate.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.delegate.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.delegate.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.delegate
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.delegate.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.delegate.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.delegate.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.delegate.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.delegate.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.delegate.list_live_ext_set().await
    }

    // ── The four non-trivial trait defaults ──────────────────────────────
    //
    // Forwarded because omitting them SUBSTITUTES behaviour rather than merely
    // skipping instrumentation: `read_ext_refs_sidecar`'s default returns an
    // EMPTY set, which would make every sidecar-shaped assertion vacuously
    // green.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.delegate.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.delegate
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.delegate.publish_worst_case()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hot writer + bounded polling
// ─────────────────────────────────────────────────────────────────────────────

/// A writer that commits field-set deltas as fast as the backing store will take
/// them, through its OWN `LayerStore` at [`WRITER_STREAM_MAXLEN`].
///
/// Both halves of the payload shape are load-bearing: a field-only delta rides
/// `update_layer`'s Lua fast path (a structural command takes a slower path —
/// the structural route on `lua`, or the WATCH/MULTI/EXEC slow path on the
/// rollback backends, which commits roughly an order of magnitude less often —
/// and a slower writer lets the assembler win), and an `ompf`-less update entry
/// is read by the classifier as an EMPTY commit by invariant (b.pw6).
///
/// Its own `LayerStore` rather than `harness.svc.layer_store` because the
/// harness's is pinned at the production default retention of 4096; see
/// [`WRITER_STREAM_MAXLEN`]. This is `helpers::drive_captured_tail`'s shape.
///
/// **Its commits are deliberately NOT in the composed base.** They sit above the
/// lineage head `S`, which the re-base does not read — see the module header's
/// premise section. The writer exists to keep the layer due on every tick, and
/// for nothing else.
struct HotWriter {
    handle: JoinHandle<()>,
    stop: Arc<AtomicBool>,
    commits: Arc<AtomicU64>,
    error: Arc<Mutex<Option<String>>>,
}

impl HotWriter {
    fn spawn(tv: &TestValkey, session_id: &str, layer_name: &str, object_id: u64) -> Self {
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            WRITER_STREAM_MAXLEN,
        ));
        let stop = Arc::new(AtomicBool::new(false));
        let commits = Arc::new(AtomicU64::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        let handle = {
            let stop = Arc::clone(&stop);
            let commits = Arc::clone(&commits);
            let error = Arc::clone(&error);
            let session_id = session_id.to_string();
            let layer_name = layer_name.to_string();
            tokio::spawn(async move {
                let mut nonce = 500_000u64;
                while !stop.load(Ordering::Relaxed) {
                    let delta = OwnedDelta {
                        set_fields: vec![SetField {
                            section_id: ROOT_SECTION_ID,
                            key_name: "hot-field".to_string(),
                            // Distinct per commit, so a silently-dropped delta
                            // cannot be mistaken for one that was applied.
                            value_or_ext_hash: nonce.to_le_bytes().to_vec(),
                            ext_value_size: 0,
                            ext_value_index: 0,
                            set_order: nonce,
                        }],
                        ..Default::default()
                    };
                    let ompf = Bytes::from(delta_to_ompf_bytes(&delta, &[]));
                    // No pacing sleep: the writer commits as fast as the
                    // backing store will take it.
                    if let Err(e) = layer_store
                        .update_layer(
                            &session_id,
                            &layer_name,
                            object_id,
                            Some(delta),
                            Some(ompf),
                            WRITER_CLIENT_ID,
                            0,
                        )
                        .await
                    {
                        // Normal at the END of a fixture (the pool is torn down
                        // at test exit), but recorded either way: a writer that
                        // died on its second commit and one whose arm was simply
                        // not reached produce the same "arm never reached"
                        // symptom.
                        *error.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
                        return;
                    }
                    commits.fetch_add(1, Ordering::Relaxed);
                    nonce += 1;
                }
            })
        };

        Self {
            handle,
            stop,
            commits,
            error,
        }
    }

    fn report(&self) -> String {
        let n = self.commits.load(Ordering::Relaxed);
        match self.error.lock().unwrap_or_else(|p| p.into_inner()).clone() {
            Some(e) => format!("{n} writer commits, then died with: {e}"),
            None => format!("{n} writer commits, no writer error"),
        }
    }

    /// Block until the writer has landed [`WARM_UP_COMMITS`], so the loop meets
    /// an already-hot layer. A state precondition, not a timing hack.
    async fn warm_up(&self) {
        let deadline = Instant::now() + WARM_UP_DEADLINE;
        while self.commits.load(Ordering::Relaxed) < WARM_UP_COMMITS {
            assert!(
                Instant::now() < deadline,
                "hot writer never reached {WARM_UP_COMMITS} warm-up commits within \
                 {WARM_UP_DEADLINE:?} ({})",
                self.report()
            );
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    async fn stop_and_join(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.await;
    }
}

/// Bounded poll on an async predicate. Returns whether it held before the
/// deadline. No naked `sleep` is used as synchronization anywhere in this file;
/// [`POLL_INTERVAL`] is the cadence of a state poll, not a wait for a race.
async fn poll_until<F, Fut>(deadline: Duration, mut pred: F) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = Instant::now();
    while start.elapsed() < deadline {
        if pred().await {
            return true;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    pred().await
}

/// Bounded poll until one `live_edit_checkpoint_tail_total{outcome}` cell rises
/// above the value it held in `before`.
///
/// The METRIC, deliberately, and never "a Tail exists in the listing": this
/// fixture HAND-PUTS K tails as its lineage, so a listing predicate is satisfied
/// from the first poll and the drive would return instantly while asserting
/// nothing. The rising cell is the loop's own verdict for THIS drive.
async fn poll_tail_outcome_rose(
    metrics: &SharedMetrics,
    before: &HashMap<String, f64>,
    outcome: &'static str,
    deadline: Duration,
) -> bool {
    let baseline = *before.get(outcome).unwrap_or(&0.0);
    let metrics = Arc::clone(metrics);
    poll_until(deadline, || {
        let metrics = Arc::clone(&metrics);
        async move {
            gather_tail_outcome_cells(&metrics)
                .get(outcome)
                .is_some_and(|value| *value > baseline)
        }
    })
    .await
}

/// The `outcome → value` delta of `live_edit_checkpoint_tail_total` between two
/// [`gather_tail_outcome_cells`] snapshots.
///
/// The family carries `outcome` ONLY — deliberately service-free — so a matcher
/// naming a `service` label would match NOTHING and every assertion would read
/// zero. The shared readers in `helpers.rs` carry that discipline; this is only
/// their subtraction.
fn tail_delta(before: &HashMap<String, f64>, after: &HashMap<String, f64>, outcome: &str) -> f64 {
    let pre = *before.get(outcome).unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_tail_total{{outcome={outcome:?}}} was ABSENT from the BEFORE \
             snapshot. Every member of CHECKPOINT_TAIL_OUTCOMES is zero-seeded at Metrics::new, \
             so an absent cell means the vocabulary or the label set moved"
        )
    });
    let post = *after.get(outcome).unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_tail_total{{outcome={outcome:?}}} was ABSENT from the AFTER \
             snapshot but present in the BEFORE one — the family's label set changed mid-test"
        )
    });
    post - pre
}

/// Assert the rendered `checkpoint_tail_total` cell set IS the pinned
/// vocabulary, whole.
///
/// The Task-5 obligation, transferred here: `refused_lineage_unrootable` made
/// the vocabulary TEN, and a sweep that hardcoded nine would stop covering the
/// new member silently. Read against the production const — never a literal
/// list — so a member added tomorrow reaches this file automatically, and an
/// OUT-OF-BAND outcome (one the pre-init loop never seeded, materialised on its
/// first occurrence) shows up here as an extra key.
fn assert_tail_vocabulary_is_whole(cells: &HashMap<String, f64>, context: &str) {
    let rendered: HashSet<&str> = cells.keys().map(String::as_str).collect();
    let pinned: HashSet<&str> = CHECKPOINT_TAIL_OUTCOMES.iter().copied().collect();
    assert_eq!(
        rendered,
        pinned,
        "{context}: the rendered `live_edit_checkpoint_tail_total` cell set must be exactly \
         CHECKPOINT_TAIL_OUTCOMES ({} members). A MISSING key means the zero-seeding loop stopped \
         covering a member and every delta read on it would fold to a panic; an EXTRA key is an \
         out-of-band outcome materialised on first occurrence, which no per-name probe would look \
         for",
        CHECKPOINT_TAIL_OUTCOMES.len()
    );
}

/// What one `checkpoint_tail_total{outcome}` cell is allowed to do over a window.
#[derive(Debug, Clone, Copy)]
enum TailDelta {
    /// The cell moved by exactly this much.
    Exactly(f64),
    /// The cell moved by at least this much — used only where the loop's tick
    /// cadence, not the claim, decides the count.
    AtLeast(f64),
}

/// Assert the per-outcome deltas of `live_edit_checkpoint_tail_total` over one
/// window, across the **whole** [`CHECKPOINT_TAIL_OUTCOMES`] vocabulary.
///
/// This is the "before/after map" form the Subtask's cell-separation item asks
/// for, and the reason it is a map rather than a per-name probe is the failure it
/// is designed to catch: a change that made a DIFFERENT arm tick — or that minted
/// an out-of-band outcome, which materialises its series on first occurrence —
/// is invisible to a probe that names one cell, and shows up here as a
/// non-conforming key or a non-zero delta on an outcome nobody listed.
///
/// Outcomes absent from `expected` MUST be flat. The vocabulary is ITERATED from
/// the production const, never restated: Task 5 appended
/// `refused_lineage_unrootable` as the tenth member, and a hardcoded list would
/// have stopped covering precisely the newest arm.
fn assert_tail_deltas(
    before: &HashMap<String, f64>,
    after: &HashMap<String, f64>,
    expected: &[(&str, TailDelta)],
    context: &str,
) {
    assert_tail_vocabulary_is_whole(before, &format!("{context} (before snapshot)"));
    assert_tail_vocabulary_is_whole(after, &format!("{context} (after snapshot)"));
    for outcome in CHECKPOINT_TAIL_OUTCOMES {
        let got = tail_delta(before, after, outcome);
        let want = expected
            .iter()
            .find(|(name, _)| name == outcome)
            .map(|(_, delta)| *delta)
            .unwrap_or(TailDelta::Exactly(0.0));
        match want {
            TailDelta::Exactly(value) => assert_eq!(
                got,
                value,
                "{context}: `live_edit_checkpoint_tail_total{{outcome={outcome:?}}}` MUST move \
                 exactly {value} over this window, moved {got}. Read against the WHOLE \
                 CHECKPOINT_TAIL_OUTCOMES vocabulary ({} members); before {before:?}, after \
                 {after:?}",
                CHECKPOINT_TAIL_OUTCOMES.len()
            ),
            TailDelta::AtLeast(value) => assert!(
                got >= value,
                "{context}: `live_edit_checkpoint_tail_total{{outcome={outcome:?}}}` MUST move at \
                 least {value} over this window, moved {got}. before {before:?}, after {after:?}"
            ),
        }
    }
}

/// Assert the per-cell deltas of `live_edit_checkpoint_base_bootstrap_total` over
/// one window, across the whole `outcome × site` grid.
///
/// Same discipline as [`assert_tail_deltas`] and for the same reason — cells
/// absent from `expected` must be flat, and the grid is read whole so a tick that
/// landed on the WRONG site (`creation_base` where `restore_rebase` was owed, the
/// swap `BOOTSTRAP_SITE_LABEL_PINS` exists to prevent) is a failure here rather
/// than invisible.
fn assert_bootstrap_deltas(
    before: &HashMap<(String, String), f64>,
    after: &HashMap<(String, String), f64>,
    expected: &[((&str, &str), f64)],
    context: &str,
) {
    let before_keys: HashSet<&(String, String)> = before.keys().collect();
    let after_keys: HashSet<&(String, String)> = after.keys().collect();
    assert_eq!(
        before_keys, after_keys,
        "{context}: the rendered `live_edit_checkpoint_base_bootstrap_total` cell SET must not \
         change over the window. An extra key is an out-of-vocabulary outcome or site materialised \
         on first occurrence, which no per-cell probe would look for"
    );
    for (cell, pre) in before {
        let post = after
            .get(cell)
            .expect("the key sets were just asserted equal");
        let got = post - pre;
        let want = expected
            .iter()
            .find(|((outcome, site), _)| outcome == &cell.0 && site == &cell.1)
            .map(|(_, value)| *value)
            .unwrap_or(0.0);
        assert_eq!(
            got, want,
            "{context}: `live_edit_checkpoint_base_bootstrap_total{{outcome={:?}, site={:?}}}` \
             MUST move exactly {want} over this window, moved {got}. This family is \
             one-increment-per-invocation (`BaseBootstrap::run_once` is a thin wrapper around the \
             single tick site), so a delta is an invocation count. before {before:?}, after \
             {after:?}",
            cell.0, cell.1
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Harness wiring
// ─────────────────────────────────────────────────────────────────────────────

/// A harness at `tick_secs = TICK_SECS` with the tail flag **ON**, `max_chain =
/// K`, the caller's `store`, and a real Valkey-backed external-value store built
/// through Task 2's `ext_value_store_factory`.
///
/// `wire_restore_rebase` is left at its **production** default (`true`): this
/// lane's subject is SR-4.1's chain-collapse re-base, and SR-3.6's post-restore
/// re-base is simply production behaviour that must be allowed to run around it.
/// The named opt-out is Epic 4 Test 1's and is not used here.
///
/// **No `install_recovery_hook` call, deliberately.** That entry point exists to
/// chain a test-only builder onto the harness's OWN hook, and this lane needs
/// none: the stock harness already ships the production wiring — the single
/// `BaseBootstrap` and the single ext-value store the factory below built, both
/// threaded into `svc.recovery` the way `src/main.rs` does. Calling it with an
/// identity closure would rebuild the same hook from the same pieces. That the
/// stock wiring really is production's is asserted structurally in
/// `harness_parity_test.rs`; this file does not re-argue it.
async fn setup_rebase_harness(store: Arc<dyn CheckpointStore>) -> CheckpointHarness {
    let mut opts = CheckpointHarnessOptions::new(TICK_SECS);
    opts.tail.enabled = true;
    opts.tail.max_chain = MAX_CHAIN;
    opts.store = Some(store);
    // Built through the FACTORY rather than inherited from the harness default,
    // so the posture is explicit at the call site instead of implicit: a REAL
    // `ValkeyExternalValueStore` over the harness's own `TestValkey` (its key
    // prefix and its `Drop`-time cleanup), with `.with_metrics(..)` attached and
    // `.with_checkpoint_store(..)` deliberately NOT armed.
    //
    // The omission is load-bearing for the ext half below. With SR-3.2
    // fetch-on-miss armed, a value that was never written to Valkey could
    // silently self-heal out of the checkpoint adapter on the next read, and the
    // "the re-base backfilled this blob" claim would be indistinguishable from
    // "something lazily fetched it". Leaving it off keeps the two separable.
    opts.ext_value_store_factory = Some(Box::new(|ctx: ExtStoreCtx<'_>| {
        Arc::new(
            ValkeyExternalValueStore::new(ctx.tv.client().clone(), ctx.tv.key_schema().clone())
                .with_metrics(Arc::clone(ctx.metrics)),
        )
    }));

    let harness = setup_checkpoint_harness_with_options(opts).await;
    assert!(
        harness.config.checkpoint_tail_enabled,
        "fixture precondition: the harness Config must carry the tail flag ON — the SR-7.1 \
         bootstrap gate, the loop's TailCaptureConfig and the capture path all read it, and with \
         it off the ChainFull -> re-base arm is unreachable"
    );
    assert_eq!(
        harness.config.checkpoint_tail_max_chain as usize, MAX_CHAIN,
        "fixture precondition: K must have reached the harness Config — the loop re-derives its \
         own TailCaptureConfig from it via `TailCaptureConfig::from_config`, so a K that stayed at \
         the 8 default would leave this fixture's {MAX_CHAIN}-tail lineage far under the bound and \
         the re-base arm would never be reached"
    );
    // Derived, never hardcoded: a literal `true` here would keep passing if
    // production stopped gating. Same three-way conjunction `main.rs` spells at
    // its `BaseBootstrap::new` call site.
    let bootstrap_enabled = harness.config.checkpoint_enabled
        && !matches!(
            harness.config.checkpoint_adapter,
            CheckpointAdapter::Disabled
        )
        && harness.config.checkpoint_tail_enabled;
    assert!(
        bootstrap_enabled,
        "fixture precondition: the SR-7.1 three-way gate must resolve TRUE on this harness, \
         otherwise CreateLayer plants no lineage root and the hand-put tails would have no base to \
         chain onto"
    );
    harness
}

/// Spawn a periodic checkpoint loop against the harness's own wiring, with the
/// tail knobs RE-DERIVED from `harness.config` — the same single source of truth
/// `main.rs` and the harness's own loop read.
///
/// Used after the fixture has fenced the harness's loop: the leader lock is an
/// acquire-or-refresh CAS keyed on `replica_id`, so a second loop under the same
/// [`SERVICE_LABEL`] re-acquires rather than waiting out the TTL — which is also
/// what lets the lifecycle arm stop and re-spawn a loop mid-drive.
///
/// `retain_generations` is a PARAMETER rather than a module constant because the
/// two arms need opposite postures and each one's is load-bearing: Test 1 runs at
/// [`DRIVE_RETAIN_GENERATIONS`] so retention is not a variable in a failure at
/// all, and Test 3 runs at [`LIFECYCLE_RETAIN_GENERATIONS`] precisely so that it
/// IS one.
///
/// The loop runs on `unprovable_cut_client` (`helpers.rs`), armed for
/// `object_id`: every assembly of the layer pins above every cut, its proven
/// read is judged regressed, and the tick falls back to the capture — and,
/// at the bound, to the ChainFull re-base behind it. Without the seam every
/// read proves its cut however hot the layer is (Spec SDD SR-4.1, SR-11.5) and
/// every tick writes a base. The seam stays armed for the rest of the test.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn spawn_tail_loop(
    harness: &CheckpointHarness,
    object_id: u64,
    retain_generations: usize,
) -> (CancellationToken, JoinHandle<()>) {
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        unprovable_cut_client(&harness.tv, object_id),
        harness.tv.key_schema().clone(),
        Arc::clone(&harness.store),
        CheckpointLoopConfig {
            tick: Duration::from_secs(TICK_SECS),
            // Both gates open so every tick reaches the assembler; the capture
            // fallback — and behind it the ChainFull re-base arm — is downstream
            // of the assembly being SKIPPED, not of a gate.
            interval: Duration::ZERO,
            quiesce: Duration::ZERO,
            retain_generations,
            counters_sync: Duration::from_secs(3600),
            reconcile_interval: Duration::from_secs(3600),
            leader: crate::common::test_leader(SERVICE_LABEL),
            tail: TailCaptureConfig::from_config(&harness.config),
        },
        Some(Arc::clone(&harness.metrics)),
        cancel.clone(),
    );
    (cancel, handle)
}

/// Cancel a spawned loop and join it under ONE deadline.
async fn drain_loop(cancel: CancellationToken, handle: JoinHandle<()>) {
    cancel.cancel();
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, handle).await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Drive + read helpers
// ─────────────────────────────────────────────────────────────────────────────

/// A `StatLayerRequest` carrying the stub bearer token the harness's
/// `AllowAllPermissions` / `StubUserinfo` wiring expects.
fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Issue one `UpdateLayer` through the real handler and return
/// `(sequence_num, the OMPF bytes it sent)`.
///
/// Returning the sent bytes is the whole point: they are what
/// [`put_tail_generation_from_records`] encodes into the tail envelope, so the
/// artifact carries exactly what the live commit carried — including, for the
/// ext-value commit, the blob as OMPF Part 1. That is byte-for-byte what the
/// capture path persists (`le:{oid}:obj:stream`'s `ompf` field is the writer's
/// raw inbound OMPF).
async fn drive_commit(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: &OwnedDelta,
    external_values: &[&[u8]],
    client_timestamp: u64,
) -> (u64, Bytes) {
    let ompf = delta_to_ompf_bytes(delta, external_values);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            layer_name,
            object_id,
            client_timestamp,
            CLIENT_ID,
        ),
        make_update_data(ompf.clone()),
    ]);
    let sequence_num = handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("UpdateLayer must succeed while building the lineage fixture")
        .into_inner()
        .sequence_num;
    (sequence_num, Bytes::from(ompf))
}

/// `HGETALL le:{object_id}:obj:hdr`.
async fn read_obj_header(harness: &CheckpointHarness, object_id: u64) -> HashMap<String, String> {
    harness
        .tv
        .client()
        .pool
        .hgetall(&harness.tv.key_schema().obj_header(&object_id.to_string()))
        .await
        .expect("HGETALL obj:hdr must succeed")
}

/// Parse a `u64` column off an `obj:hdr` map with a diagnostic on absence.
fn header_u64(header: &HashMap<String, String>, field: &str) -> u64 {
    let raw = header.get(field).unwrap_or_else(|| {
        panic!(
            "obj:hdr MUST carry `{field}`; got fields: {:?}",
            header.keys().collect::<Vec<_>>()
        )
    });
    raw.parse::<u64>()
        .unwrap_or_else(|e| panic!("obj:hdr `{field}` MUST parse as u64; got {raw:?}: {e}"))
}

/// Like [`header_u64`] but folds an absent column to `0`. Only for the restore-
/// cursor columns, which a never-restored layer legitimately lacks.
fn header_u64_or_zero(header: &HashMap<String, String>, field: &str) -> u64 {
    header
        .get(field)
        .map(|raw| {
            raw.parse::<u64>()
                .unwrap_or_else(|e| panic!("obj:hdr `{field}` MUST parse as u64; got {raw:?}: {e}"))
        })
        .unwrap_or(0)
}

/// `EXISTS` on a fully-qualified key.
async fn key_exists(harness: &CheckpointHarness, key: &str) -> bool {
    harness
        .tv
        .client()
        .pool
        .exists(key)
        .await
        .expect("EXISTS must succeed against the live test Valkey")
}

/// The newest-first generation listing for `object_id`.
async fn listing(harness: &CheckpointHarness, object_id: u64) -> Vec<CheckpointHandle> {
    harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed")
}

/// Snapshot every cell of `live_edit_checkpoint_restore_lineage_total` for a
/// before/after delta. The whole pinned vocabulary, never one cell: a step-down
/// that ticks the WRONG reason is a different defect from one that does not tick
/// at all, and only a full sweep tells them apart.
fn lineage_cells(harness: &CheckpointHarness) -> Vec<(&'static str, u64)> {
    CHECKPOINT_RESTORE_LINEAGE_OUTCOMES
        .iter()
        .map(|outcome| {
            (
                *outcome,
                read_restore_lineage_counter(&harness.metrics, outcome),
            )
        })
        .collect()
}

/// Assert the per-outcome deltas since `before`. Outcomes absent from `expected`
/// must be flat.
fn assert_lineage_deltas(
    harness: &CheckpointHarness,
    before: &[(&'static str, u64)],
    expected: &[(&str, u64)],
    context: &str,
) {
    for (outcome, pre) in before {
        let want = expected
            .iter()
            .find(|(name, _)| name == outcome)
            .map(|(_, value)| *value)
            .unwrap_or(0);
        let got = read_restore_lineage_counter(&harness.metrics, outcome) - pre;
        assert_eq!(
            got, want,
            "{context}: lineage outcome {outcome:?} MUST tick {want} time(s), got {got}. This \
             family is NOT one-tick-per-attempt — one attempt that steps down twice before \
             resolving ticks three cells"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The fixture: a real-commit K-link lineage built through the real handlers
// ─────────────────────────────────────────────────────────────────────────────

/// The live state at one instant — the comparison target the post-re-base
/// restore must reproduce.
struct Oracle {
    snapshot: ObjectSnapshot,
    header: HashMap<String, String>,
}

/// Everything the test needs from the planted lineage.
struct RebaseLineageFixture {
    session_id: String,
    layer_name: String,
    object_id: u64,
    /// `S0` — the loop-written base's `snap_seq`, and the layer's live sequence
    /// at the instant that base was assembled. The lineage ROOT.
    s0: u64,
    /// `S` — the lineage HEAD: the last commit inside the newest hand-put tail's
    /// span, and therefore the address the chain-collapse re-base will PUT at.
    head_seq: u64,
    /// The external value one tail-range commit carried, and its hex digest.
    ext_blob: Vec<u8>,
    ext_hash_hex: String,
    /// The live state at `S`, captured BEFORE the hot writer starts: what the
    /// post-re-base restore must reproduce. See the module header's premise.
    head_oracle: Oracle,
}

async fn capture_oracle(harness: &CheckpointHarness, object_id: u64) -> Oracle {
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect("read_object_state_for_test must succeed while the live layer is intact");
    let header = read_obj_header(harness, object_id).await;
    Oracle { snapshot, header }
}

/// The oracle's own self-check, run at capture time: the decoded snapshot and
/// the raw header agree on the sequence, and the layer has never been restored.
///
/// The second half is what gives the post-restore assertions their meaning — the
/// column is proved absent BEFORE the restore, so its post-restore value is a
/// change this restore made rather than a residue that was already there. It is
/// also a precondition of the re-base itself: `rebase_on_chain_full`'s FIRST
/// gate declines a collapse outright on a non-zero `restore_seq_bump`
/// (`refused_seq_gap`), so a fixture that arrived pre-restored would never reach
/// the arm under test.
fn assert_live_oracle_is_unrestored(oracle: &Oracle) {
    assert_eq!(
        header_u64(&oracle.header, "seq"),
        oracle.snapshot.sequence_num,
        "the live obj:hdr and the decoded snapshot must agree on `seq`"
    );
    assert_eq!(
        header_u64_or_zero(&oracle.header, "restore_seq_bump"),
        0,
        "a layer that has only ever been written live MUST carry no guard band. A non-zero value \
         here means a restore already ran — and `rebase_on_chain_full`'s fossil gate would then \
         decline the collapse this test exists to observe"
    );
}

/// CreateSession → CreateLayer (wide) → two commits → wait for the loop's base
/// → **fence the loop** → drive `tail_ranges` contiguous tail RANGES of real
/// commits → hand-put that many chained tails from the recorded OMPF → capture
/// the head oracle.
///
/// `tail_ranges` is a parameter, not [`MAX_CHAIN`], because the two arms need the
/// chain in opposite states. Test 1 and Test 2 pass `MAX_CHAIN` — the lineage is
/// ON the bound, so the very first eligible tick takes the `chain_len >=
/// max_chain` branch. Test 3 passes [`PRE_DR_TAIL_RANGES`] (`j < K`), because its
/// subject is a restore landing on a lineage that has NOT reached the bound; a
/// fixture already at `K` would collapse before the DR event and there would be
/// no mid-lifecycle state to interrupt.
///
/// The fence is `cancel()` plus a bounded join on the loop's `JoinHandle`. After
/// it, no tick exists to land a generation, so nothing can move the fixture
/// underneath the assertions: not the retention prune, and not a fresh base at a
/// sequence the hand-put tails do not link onto. The test spawns a fresh loop via
/// [`spawn_tail_loop`] once it has armed the fence.
///
/// The chain is built so that `tail_{n+1}.from_seq == tail_n.to_seq` — SR-2.1's
/// contiguity — with the FIRST tail rooted at `S0`, i.e. exactly the shape the
/// capture unit's `chain_len >= max_chain` pre-filter counts and the re-base's
/// link walk verifies.
async fn setup_rebase_lineage(
    harness: &mut CheckpointHarness,
    name: &str,
    tail_ranges: usize,
) -> RebaseLineageFixture {
    assert!(
        tail_ranges >= 1,
        "the fixture must plant at least one tail: `head_seq` is the newest hand-put tail's \
         `to_seq`, and with none it would fold to the base sequence and every link assertion \
         below would read the lineage root as the lineage head"
    );

    // ── The session, through the real RPC entry point.
    let session_id = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: SESSION_BASE_PATH.to_owned(),
            name: format!("{name}-session"),
        }),
    )
    .await
    .expect("CreateSession must succeed against an AllowAllPermissions harness")
    .into_inner()
    .session_id;
    wait_for_session_checkpoint(&harness.store, &session_id, STORE_DEADLINE_SECS).await;

    // ── The layer, through the real streaming CreateLayer handler — the path
    // that actually calls `svc.base_bootstrap.spawn_creation_base(..)`.
    let layer_name = format!("{name}-layer");
    let ompf = delta_to_ompf_bytes(&wide_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, &layer_name),
        make_create_data(ompf),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed while building the lineage fixture")
        .into_inner()
        .object_id;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );
    // Drains BOTH the CreateLayer session-checkpoint write and the SR-7.1
    // creation bootstrap task (both ride the harness's shared TaskTracker).
    wait_for_session_checkpoint_quiesced(harness, &session_id, STORE_DEADLINE_SECS).await;

    // ── Two ordinary commits before the base, so the base carries state no
    // single delta produced and the restored snapshot is a genuine round trip of
    // accumulated live state.
    let mut committed_seq = 0;
    for index in 0..2 {
        let (seq, _) = drive_commit(
            harness,
            &session_id,
            &layer_name,
            object_id,
            &pre_base_commit_delta(index),
            &[],
            1_000 + index,
        )
        .await;
        committed_seq = seq;
    }

    // Gate on the committed sequence, not merely "any base": with
    // `interval = quiesce = 0` the loop can fire between CreateLayer and the
    // commits and land a generation that predates them, which would leave `S0`
    // non-deterministic and the tails' link arithmetic meaningless.
    let base_handle = wait_for_checkpoint_at_least(
        &harness.store,
        object_id,
        committed_seq,
        STORE_DEADLINE_SECS,
    )
    .await;

    // ── The fence ────────────────────────────────────────────────────────
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, loop_task).await;

    let s0 = base_handle.snap_seq;
    assert_eq!(
        s0, committed_seq,
        "fixture precondition: the loop's base MUST have been assembled at the layer's live \
         sequence ({committed_seq}). A higher value means a commit landed that this fixture does \
         not know about, and the first hand-put tail's `from_seq = S0` link would name a \
         predecessor covering commits nothing replays"
    );

    // ── K contiguous tail RANGES of real commits, hand-put as K chained tails.
    //
    // Each range carries structural + inline + time-sample records; the LAST
    // commit of the LAST range carries the EXTERNAL VALUE, so the ext reference
    // is the newest write in the collapsed span and the re-base's SR-4.3
    // backfill has something to do.
    let blob = ext_blob();
    let ext_hash = ext_content_hash(&blob);
    let ext_hash_hex = sha256_to_hex(&ext_hash);

    let mut expected = s0 + 1;
    let mut chain_from = s0;
    let mut head_seq = s0;
    for range in 0..tail_ranges as u64 {
        let is_last_range = range + 1 == tail_ranges as u64;
        let mut records: Vec<(u64, Bytes)> = Vec::new();
        for step in 0..TAIL_RANGE_COMMITS {
            let is_last_commit = step + 1 == TAIL_RANGE_COMMITS;
            let global_index = range * TAIL_RANGE_COMMITS + step;
            let (delta, externals): (OwnedDelta, Vec<&[u8]>) = if is_last_range && is_last_commit {
                (
                    tail_ext_value_delta(&ext_hash, blob.len()),
                    vec![blob.as_slice()],
                )
            } else {
                (tail_commit_delta(global_index), Vec::new())
            };
            let record = drive_commit(
                harness,
                &session_id,
                &layer_name,
                object_id,
                &delta,
                &externals,
                2_000 + global_index,
            )
            .await;
            assert_eq!(
                record.0, expected,
                "fixture precondition: commit sequences must be contiguous. Range {range} commit \
                 {step} returned {}, expected {expected}. A gap here means another writer shares \
                 this layer, and the hand-put tail would carry a span its records cannot cover",
                record.0
            );
            expected += 1;
            records.push(record);
        }

        let handle = put_tail_generation_from_records(
            &harness.store,
            object_id,
            &session_id,
            &layer_name,
            s0,
            chain_from,
            &records,
            HAND_PUT_WRITTEN_AT_UNIX_MS,
            0,
        )
        .await;
        assert_eq!(
            handle.snap_seq,
            records.last().expect("records were driven").0,
            "a tail's handle sequence IS its envelope `to_seq`"
        );
        // SR-2.1 contiguity, asserted link by link as the chain is built rather
        // than re-derived afterwards: `tail_{n+1}.from_seq == tail_n.to_seq`.
        // The re-base's link walk is what verifies this in production; a fixture
        // that got it wrong would surface as a refusal, not as a wrong answer.
        chain_from = handle.snap_seq;
        head_seq = handle.snap_seq;
    }

    // The ext-value commit went through the service's real
    // `ValkeyExternalValueStore` (the factory's), so both content-addressed keys
    // must exist. Otherwise the ext half of this test would be comparing against
    // nothing.
    for key in [
        harness.tv.key_schema().ext_meta(&ext_hash_hex),
        harness.tv.key_schema().ext_data(&ext_hash_hex),
    ] {
        assert!(
            key_exists(harness, &key).await,
            "fixture precondition: `{key}` MUST exist after the ext-carrying UpdateLayer. Its \
             absence means the service is holding an in-process store (a MockExternalValueStore, \
             say) rather than the factory's real Valkey-backed one, and the ext half of this file \
             asserts nothing"
        );
    }

    // ── The lineage shape the capture unit will read ─────────────────────
    let generations = listing(harness, object_id).await;
    let base_index = generations
        .iter()
        .position(|h| h.kind == CheckpointKind::Base)
        .unwrap_or_else(|| {
            panic!("the lineage must be rooted at a Base. Listing: {generations:?}")
        });
    assert_eq!(
        base_index, tail_ranges,
        "fixture precondition: the capture unit's chain count is the POSITION of the newest Base \
         in the newest-first listing (`lineage_shape`), and it must equal the number of ranges \
         this fixture planted ({tail_ranges}). At `tail_ranges == MAX_CHAIN` ({MAX_CHAIN}) that \
         puts the lineage ON the bound so the very first eligible tick takes the \
         `chain_len >= max_chain` branch; BELOW it (Test 3's `j < K`) it is the mid-lifecycle \
         state a DR restore then interrupts. Listing: {generations:?}"
    );
    assert_eq!(
        generations[0].kind,
        CheckpointKind::Tail,
        "and the listing head must be the newest hand-put Tail. Listing: {generations:?}"
    );
    assert_eq!(
        generations[0].snap_seq, head_seq,
        "which sits at S ({head_seq}). Listing: {generations:?}"
    );

    // ── The oracle at S — captured while the layer is still EXACTLY at S ──
    let head_oracle = capture_oracle(harness, object_id).await;
    assert_eq!(
        head_oracle.snapshot.sequence_num, head_seq,
        "fixture precondition: the oracle must be captured at the lineage head sequence S, with \
         no commit yet above it. This is the comparison target the whole lane rests on — the \
         re-base composes the EXISTING lineage and PUTs at S, so a hot writer's later commits are \
         deliberately outside what the new base carries"
    );
    assert_live_oracle_is_unrestored(&head_oracle);

    RebaseLineageFixture {
        session_id,
        layer_name,
        object_id,
        s0,
        head_seq,
        ext_blob: blob,
        ext_hash_hex,
        head_oracle,
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The claim
// ═════════════════════════════════════════════════════════════════════════════

/// Epic AC (fidelity half): a restore taken **after** the periodic loop's own
/// chain-collapse re-base equals the pre-re-base live state at `S` on the
/// consolidated superset faithful-column basis.
///
/// The composition this test runs, and that no narrower test runs end to end:
///
/// 1. a real-commit `1 base + K tails` lineage, head `S`, with the oracle pinned
///    at `S`;
/// 2. a hot writer committing ABOVE `S` — keeping the layer due, and
///    deliberately outside what the re-base composes — and the unprovable-cut
///    seam skipping every assembly;
/// 3. the REAL periodic loop reaching the capture fallback, the capture unit
///    returning `ChainFull`, and `rebase_on_chain_full` collapsing the chain into
///    ONE new base at `S` (`tail_total{rebased}` +1,
///    `writes_total{success, layer}` +1, `last_checkpoint_seq = S`);
/// 4. a `wipe_layer_state` partial loss and a production-handler restore, which
///    resolves `gathered_base` — chain 0, i.e. the collapse worked;
/// 5. the restored state compared to the oracle on the faithful columns, plus
///    the SR-4.3 ext-backfill's lifecycle consequence.
///
/// `handle_stat_layer` is the restore handler, matching the sibling arms
/// (`e2e_composed_restore_test.rs`, and Tests 2–4 of
/// `e2e_full_wipe_composed_restore_test.rs`). The reason is exactness: StatLayer
/// is READ-ONLY, so the restored state at the moment of comparison is precisely
/// what the restore materialized, with no re-issued commit to exclude from the
/// oracle by name. `handle_update_layer` would work too but would put a field on
/// the restored side that the oracle cannot have, which is a comparison this
/// lane does not need to weaken itself with.
///
/// Compiled only under `test-checkpoint-fault-injection` (the unprovable-cut
/// seam); without the feature it is compiled out, not skipped.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_loop_level_rebase_restores_to_the_pre_rebase_state_at_s() {
    let fence = Arc::new(RebaseFenceStore::new());
    let store: Arc<dyn CheckpointStore> = Arc::clone(&fence) as Arc<dyn CheckpointStore>;
    let mut harness = setup_rebase_harness(store).await;

    let fixture = setup_rebase_lineage(&mut harness, "rebase-fidelity", MAX_CHAIN).await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    // Non-vacuity, asserted on the ORACLE before anything is driven: the fixture
    // really did write inside the span the re-base has to replay. Without this
    // the faithful-column comparison could be satisfied by a base that carried
    // only the pre-base state.
    let oracle_above_base = versions_above(&fixture.head_oracle.snapshot, fixture.s0);
    assert!(
        !oracle_above_base.is_empty(),
        "fixture precondition: the oracle MUST carry entries stamped above the lineage root S0 \
         ({}) — those are the commits the tails cover and the re-base must replay. With none, the \
         whole comparison would be satisfied by a base-only lineage",
        fixture.s0
    );

    // ── Arm the structural fence, THEN start the writer, THEN re-arm the loop ─
    //
    // Ordering is load-bearing: armed before any tick exists, so the first PUT
    // the wrapper can possibly see is already fenced. See `RebaseFenceStore`.
    let attempts_before = fence.attempts().len();
    fence.arm(fixture.object_id, fixture.head_seq);

    let tail_before = gather_tail_outcome_cells(&harness.metrics);
    assert_tail_vocabulary_is_whole(&tail_before, "before the re-base drive");
    let layer_writes_success_before = read_layer_writes_counter(&harness.metrics, "success");

    let writer = HotWriter::spawn(
        &harness.tv,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.object_id,
    );
    writer.warm_up().await;
    let (cancel, loop_handle) =
        spawn_tail_loop(&harness, fixture.object_id, DRIVE_RETAIN_GENERATIONS).await;

    let rebased =
        poll_tail_outcome_rose(&harness.metrics, &tail_before, "rebased", REBASE_DEADLINE).await;

    let writer_report = writer.report();
    writer.stop_and_join().await;
    drain_loop(cancel, loop_handle).await;

    let tail_after = gather_tail_outcome_cells(&harness.metrics);
    assert_tail_vocabulary_is_whole(&tail_after, "after the re-base drive");
    let generations = listing(&harness, fixture.object_id).await;
    let put_attempts = fence.attempts_since(attempts_before);

    assert!(
        rebased,
        "the periodic loop MUST collapse the chain-at-K lineage within {REBASE_DEADLINE:?}. \
         {writer_report}. Listing: {generations:?}. Fence op log: {put_attempts:?}. Outcome cells \
         before: {tail_before:?}, after: {tail_after:?}. A rising `refused_chain_full` means the \
         re-base was REACHED and refused (SR-4.5 — read the WARN for which arm); a rising \
         `refused_seq_gap` means the fossil gate declined it; a flat family altogether means the \
         unprovable-cut seam did not fire and the capture fallback was never reached"
    );
    assert!(
        tail_delta(&tail_before, &tail_after, "rebased") >= 1.0,
        "`live_edit_checkpoint_tail_total{{outcome=\"rebased\"}}` MUST have ticked over the \
         drive; before {tail_before:?}, after {tail_after:?}"
    );
    assert_eq!(
        tail_delta(&tail_before, &tail_after, "refused_chain_full"),
        0.0,
        "`refused_chain_full` MUST be flat — it is the fold every SR-4.5 re-base REFUSAL lands \
         on, and one tick here means a collapse was attempted and did not land. before \
         {tail_before:?}, after {tail_after:?}"
    );
    assert_eq!(
        tail_delta(&tail_before, &tail_after, "captured")
            + tail_delta(&tail_before, &tail_after, "captured_capped"),
        0.0,
        "BOTH capture spellings MUST be flat: while the fence is armed no Tail PUT can land, and \
         a rejected one reaches the loop as an `Err` that moves no outcome cell at all. A tick on \
         either means a tail landed ABOVE S and the listing head is no longer the re-base's base \
         — a cap-bounded capture lands the same artifact an uncapped one would, so reading \
         `captured` alone would let a fast-runner capture through. before {tail_before:?}, after \
         {tail_after:?}"
    );
    assert_eq!(
        read_layer_writes_counter(&harness.metrics, "success") - layer_writes_success_before,
        1,
        "exactly ONE `checkpoint_writes_total{{result=\"success\", kind=\"layer\"}}` over the \
         drive — the re-base's own base PUT (`kind = \"layer\"`, not `layer_tail`: a re-base \
         produces a BASE, SR-8.2). While the fence is armed nothing else can land, so this cell \
         addresses the re-base and nothing else. Fence op log: {put_attempts:?}"
    );

    // ── The fence's own verdict: exactly one generation landed, and it is
    //    `Base@S`. This is the structural form of "no tail above S" — read off
    //    the op log, not off a poll.
    let accepted: Vec<PutAttempt> = put_attempts
        .iter()
        .copied()
        .filter(|a| a.accepted)
        .collect();
    assert_eq!(
        accepted,
        vec![PutAttempt {
            object_id: fixture.object_id,
            snap_seq: fixture.head_seq,
            kind: CheckpointKind::Base,
            accepted: true,
        }],
        "while armed, the ONLY layer PUT the fence may admit is the chain-collapse re-base's own \
         Base at the LINEAGE head S ({}) — not at the live head, which is where the hot writer's \
         commits are. Full op log: {put_attempts:?}",
        fixture.head_seq
    );
    assert!(
        !generations
            .iter()
            .any(|h| h.kind == CheckpointKind::Tail && h.snap_seq > fixture.head_seq),
        "and the listing must carry NO Tail above S ({}) — the fence's first arm, confirmed \
         against the store rather than only against the op log. Listing: {generations:?}",
        fixture.head_seq
    );
    assert_eq!(
        generations
            .first()
            .map(|h| (h.kind, h.snap_seq))
            .unwrap_or_else(|| panic!("the listing must be non-empty. Listing: {generations:?}")),
        (CheckpointKind::Base, fixture.head_seq),
        "PRECONDITION for the restore under test: the listing's newest handle must be Base@S \
         ({}). The new base TIES with the surviving Tail@S and the cross-adapter tie-break sorts \
         Base FIRST, which is what makes the restore gather chain 0. Listing: {generations:?}",
        fixture.head_seq
    );
    assert!(
        generations
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.snap_seq == fixture.s0),
        "and the SUPERSEDED lineage must still be intact: the re-base deletes NOTHING (retention \
         owns every deletion — Task 4), so base@S0 ({}) and its tails are still listed. Listing: \
         {generations:?}",
        fixture.s0
    );
    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_eq!(
        header_u64(&header, "last_checkpoint_seq"),
        fixture.head_seq,
        "the re-base stamps `last_checkpoint_seq = S` — the collapsed head IS the newest sequence \
         now covered by a durable generation"
    );

    // ── The SR-4.3 ext backfill, in the checkpoint store ──────────────────
    //
    // The re-base `put_ext_blob`-acks every tail-inline external value as a HARD
    // precondition of its base PUT. This is that write, read back. The backfill's
    // ORDERING (`max(PutExt) < PutLayer`) and its completeness against
    // `ComposedLineage::ext_values` are Task 3's `t3.dfh.be.84.p4`
    // (`rebase_ext_backfill_test.rs`) and are deliberately not restated here;
    // what this lane owns is the LIFECYCLE consequence — the blob is in the
    // store, under its hash, after a loop-level collapse.
    let backfilled = harness
        .store
        .get_ext_blob(&fixture.ext_hash_hex)
        .await
        .expect("get_ext_blob must succeed against the harness adapter")
        .unwrap_or_else(|| {
            panic!(
                "the chain-collapse re-base MUST have backfilled the tail-inline external value \
                 {} into the checkpoint store — SR-4.3 makes a durable ack of every such blob a \
                 HARD precondition of the base PUT, so a base at S with no blob behind its \
                 by-hash reference would be a lineage whose tails were the last durable copy",
                fixture.ext_hash_hex
            )
        });
    assert_eq!(
        backfilled.as_ref(),
        fixture.ext_blob.as_slice(),
        "and the backfilled blob MUST be byte-identical to what the tail-range commit sent"
    );

    // ── Disarm before the restore ────────────────────────────────────────
    //
    // The fence has done its job (its claims are asserted above) and the loop is
    // drained. Left armed it would additionally reject the SR-3.6 POST-RESTORE
    // re-base's own base PUT at `seeded_seq` — a different unit, production
    // behaviour, and not this lane's subject. Disarming keeps the restore on the
    // production posture `wire_restore_rebase = true` selects.
    fence.disarm();

    // ── The partial state loss, and the production-handler restore ────────
    wipe_layer_state(&harness.tv, fixture.object_id).await;
    assert!(
        !key_exists(
            &harness,
            &harness
                .tv
                .key_schema()
                .obj_header(&fixture.object_id.to_string())
        )
        .await,
        "wipe_layer_state must remove obj:hdr; a surviving header means StatLayer would answer \
         from live state and the recovery branch would silently not fire"
    );

    let lineage_pre = lineage_cells(&harness);

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "StatLayer after the layer-state wipe MUST restore from the re-based base and return \
             Ok, got {:?}: {}",
            status.code(),
            status.message()
        )
    });
    // Drain the fire-and-forget SR-3.6 re-base task before reading anything it
    // stamps, so the header read below is not racing it.
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    // ── The lineage-outcome sweep: `gathered_base`, chain 0 ───────────────
    //
    // A full-vocabulary before/after delta, never a single cell. `gathered_base`
    // is the assertion that the COLLAPSE WORKED: the same fixture BEFORE the
    // re-base is a `1 base + K tails` chain and would have resolved
    // `gathered_composed` (the restore would have had to fetch and replay K tail
    // artifacts). After it, the newest generation is a self-contained base and
    // the restore fetches exactly one artifact — which is the "restore cost
    // bounded" half of the Epic AC, observed here as a lineage outcome.
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("gathered_base", 1)],
        "post-re-base restore",
    );

    // ── Exact state, on the consolidated SUPERSET faithful basis ──────────
    let restored = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &fixture.object_id.to_string(),
    )
    .await
    .expect("read_object_state_for_test must succeed against the restored layer");
    assert_eq!(
        restored.object_id, fixture.object_id,
        "the restore must rebuild state under the SAME object_id rather than re-minting one"
    );
    assert_faithful_columns(
        &restored,
        &fixture.head_oracle.snapshot,
        FaithfulColumnsOptions::default(),
        "post-re-base restore at S",
    );
    // ── The version columns, stated as what they ARE ──────────────────────
    //
    // A restore — base-only here, composed in Test 2 — is ONE write of the WHOLE
    // object at its own `seeded_seq`, and since b.2ti `apply_restore` re-stamps
    // the three versions a diff builder FILTERS on (`Section::last_version`,
    // `Field::version`, `TimeSample::version`) onto it before `write_object`. So
    // those columns are a single value on BOTH tests, and the collapse is no
    // longer observable in them at all. See the module header's "the version
    // column" section for what that moved and where the in-order-replay witness
    // now lives.
    //
    // `Section::first_version` is exempt from the re-stamp, and it is the ONE
    // version column where the two tests still differ — which is why this call
    // passes `DecodedBase` and Test 2 passes `ComposedLineage`. The restore here
    // resolved `gathered_base` (asserted above): the collapsed artifact is a
    // single reset OMPF, so `decode_checkpoint_ompf` stamps EVERY section at the
    // decoder's `1`, including the five the tail ranges created above `S0`.
    // Test 2's composed gather replays those five and preserves their real
    // sequences. The difference is a decode-vs-replay property, not a content
    // one — `assert_faithful_columns` above already pinned that the CONTENT is
    // identical across the two.
    //
    // `seeded_seq` is read off the restored header rather than computed: the
    // guard band is `max(margin, the collapsed base meta's observed_commit_span)`
    // and the span is a property of the drive, not a fixture constant.
    let restored_header = read_obj_header(&harness, fixture.object_id).await;
    let seeded_seq = header_u64(&restored_header, "seq");
    assert!(
        seeded_seq > fixture.head_seq,
        "precondition for the version claim below: the restore must seed the writer cursor \
         strictly above the lineage head ({seeded_seq} vs {})",
        fixture.head_seq
    );
    // `fixture.s0` is both the presence-range floor and the lineage root here.
    // Under `DecodedBase` the root is inert (every section is predicted at the
    // decoder's stamp regardless), but it is still passed honestly rather than
    // as a placeholder — Test 2 below reads it for real.
    assert_restored_entry_versions_flattened(
        &restored,
        &fixture.head_oracle.snapshot,
        RestoredVersionRanges {
            replayed_range_floor: fixture.s0,
            lineage_root: fixture.s0,
        },
        seeded_seq,
        RestoredFirstVersions::DecodedBase,
        FaithfulColumnsOptions::default(),
        "post-re-base restore at S",
    );
    assert_eq!(
        oracle_above_base,
        versions_above(&fixture.head_oracle.snapshot, fixture.s0),
        "and the ORACLE's own map must be unchanged since the fixture's non-vacuity check — it is \
         what makes the assertion above a claim about a fixture that HAS a replayed range. \
         Without it, the presence arm would be satisfied by an oracle that had nothing above S0 \
         either, and would prove nothing"
    );

    // The ext half of the faithful basis, named rather than left implicit:
    // `assert_faithful_columns` compares external references by `(hash, size)`,
    // so this asserts the SAME thing on the one field that matters, with a
    // message that says which field and why. A restore that dropped the ext
    // reference entirely would fail the map comparison above with a diff over
    // 100+ fields; this one names it.
    let restored_ext = restored
        .fields
        .iter()
        .find(|f| f.key_name == EXT_FIELD_KEY)
        .unwrap_or_else(|| {
            panic!(
                "the restored layer MUST carry the ext-referencing field {EXT_FIELD_KEY:?} — it \
                 was written by the LAST commit of the LAST tail range, so its absence means the \
                 collapsed base lost the top of the replayed span"
            )
        });
    match &restored_ext.value {
        FieldValue::External { hash, size, .. } => {
            assert_eq!(
                sha256_to_hex(hash),
                fixture.ext_hash_hex,
                "and it must reference the fixture's own content hash"
            );
            assert_eq!(
                *size,
                fixture.ext_blob.len() as u64,
                "at the fixture's own byte length"
            );
        }
        other => panic!(
            "{EXT_FIELD_KEY:?} must come back as an EXTERNAL reference, not {other:?}. An inline \
             value here means the collapsed base folded the blob into the field, which would \
             double-store it and break the content-addressed refcount"
        ),
    }
    // `wipe_layer_state` is `le:{oid}:obj:*` ONLY — it deliberately spares the
    // content-addressed `le:ext:*` namespace (its rustdoc says so). So this is a
    // SURVIVAL claim, not a re-store claim, and it is stated as one: the value
    // the restored reference addresses is still there and still byte-identical.
    // The re-store-after-total-ext-loss claim is Epic 4's
    // (`e2e_full_wipe_composed_restore_test.rs` Test 1, under `full_valkey_wipe`)
    // and the fetch-on-miss fallback is `composed_restore_ext_matrix_test`'s;
    // neither is restated here.
    let live_blob: Bytes = harness
        .tv
        .client()
        .pool
        .get(&harness.tv.key_schema().ext_data(&fixture.ext_hash_hex))
        .await
        .expect("GET ext_data must succeed after the restore");
    assert_eq!(
        live_blob.as_ref(),
        fixture.ext_blob.as_slice(),
        "the external value the restored reference addresses MUST still be byte-identical to what \
         the tail-range commit sent"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The "before" side of the same claim
// ═════════════════════════════════════════════════════════════════════════════

/// The SAME fixture, restored **without** a re-base: it resolves
/// `gathered_composed` and reproduces the composed CONTENT at `S`.
///
/// # What this test pins, stated exactly
///
/// - the composed content at `S` is restored — the same oracle, on the same
///   superset faithful-column basis, as Test 1's collapsed-base restore;
/// - the persisted field / time-sample `version` and `Section::last_version`
///   columns are UNIFORM at the restore's own `seeded_seq`, carrying no
///   per-commit information (b.2ti);
/// - `Section::first_version` is PRESERVED from the composed state — the
///   decoder's stamp for base-era sections, the replayed sequence for
///   tail-created ones — because `apply_restore` exempts that one column from
///   the re-stamp;
/// - and every entry the `(S0, S]` range wrote is still PRESENT afterwards.
///
/// **The per-entry version ALGEBRA is not witnessed here.** The claim that the
/// tail records were replayed IN ORDER, rather than merely all applied, lives on
/// the two lanes that read composition's output before a restore persists it:
/// `src/checkpoint/compose.rs`'s in-crate `versions_above` tests and
/// `tests/proptests.rs::ac4_versions_above`. Both compare in-memory composed
/// state against live and neither is touched by the re-stamp. This test name
/// used to promise "the live versions"; it never can, post-b.2ti, and the
/// module header's version-column section records what moved.
///
/// # Why this is a test and not a sentence in the module header
///
/// Test 1's headline claim is a comparison, and a comparison needs both sides.
/// Three things only this arm can establish, and each one is load-bearing for
/// reading Test 1:
///
/// 1. **`gathered_base` is evidence.** Test 1 asserts the post-re-base restore
///    resolves `gathered_base` — chain 0. That is only meaningful if the same
///    fixture would otherwise have resolved `gathered_composed`, i.e. if the
///    chain really was there to collapse. Asserted here rather than asserted by
///    narration.
/// 2. **The faithful columns are the same on both sides.** Both arms compare
///    against an oracle captured at `S` by the identical fixture builder and on
///    the identical superset basis, so "the collapsed base restores to the same
///    state the un-collapsed chain does" is transitive across the two.
/// 3. **The `(S0, S]` range comes back through a composed gather too.** Both
///    arms run [`assert_restored_entry_versions_flattened`] over the identical
///    range, oracle and seed, so "the collapsed base carries what the
///    un-collapsed chain replays" is asserted on the version-and-presence basis
///    as well as the content one. Pre-b.2ti this arm was where the two sides
///    genuinely differed on ALL FOUR version columns (see the module header's
///    version-column section); the restore's re-stamp removed that difference on
///    three of them, and the fourth — `Section::first_version`, which the
///    re-stamp exempts — is now the only version column that still separates the
///    two arms, which is why they pass different [`RestoredFirstVersions`].
///
/// This arm does NOT re-prove the composition engine (the golden fixtures and
/// the AC-4 property own that — module header). It runs no loop, no hot writer
/// and no fence: the lineage is planted, the loop is already fenced by the
/// fixture builder, and the restore is driven straight through the handler.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_same_lineage_before_the_rebase_restores_the_composed_content_at_s() {
    // The fence store is a transparent pass-through until `arm()` is called, and
    // this arm never calls it — the same harness constructor is used so the two
    // tests cannot drift on wiring.
    let fence = Arc::new(RebaseFenceStore::new());
    let store: Arc<dyn CheckpointStore> = Arc::clone(&fence) as Arc<dyn CheckpointStore>;
    let mut harness = setup_rebase_harness(store).await;

    let fixture = setup_rebase_lineage(&mut harness, "pre-rebase-composed", MAX_CHAIN).await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    // The chain is still a chain: no collapse has run, so the listing head is
    // the newest hand-put Tail and a restore must gather and replay K artifacts.
    let generations = listing(&harness, fixture.object_id).await;
    assert_eq!(
        generations
            .first()
            .map(|h| (h.kind, h.snap_seq))
            .unwrap_or_else(|| panic!("the listing must be non-empty. Listing: {generations:?}")),
        (CheckpointKind::Tail, fixture.head_seq),
        "PRECONDITION: with no re-base run, the listing's newest handle must still be the \
         hand-put Tail@S ({}). Listing: {generations:?}",
        fixture.head_seq
    );

    wipe_layer_state(&harness.tv, fixture.object_id).await;

    let lineage_pre = lineage_cells(&harness);

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "StatLayer after the layer-state wipe MUST restore the composed lineage and return \
             Ok, got {:?}: {}",
            status.code(),
            status.message()
        )
    });
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("gathered_composed", 1)],
        "pre-re-base composed restore",
    );

    let restored = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &fixture.object_id.to_string(),
    )
    .await
    .expect("read_object_state_for_test must succeed against the restored layer");
    assert_faithful_columns(
        &restored,
        &fixture.head_oracle.snapshot,
        FaithfulColumnsOptions::default(),
        "pre-re-base composed restore at S",
    );
    // `replayed_range_floor = lineage_root = S0`, so the presence arm's range is
    // exactly `(S0, S]` — the span the K tails cover — and the helper's own two
    // non-vacuity guards fire if the fixture wrote no field, or created no
    // section, inside it. (The tail ranges below do both.) Every
    // entry of that range must come back, and (b.2ti) every one of them must
    // read the restore's own `seeded_seq` on the three diff-FILTERED version
    // columns, exactly as on Test 1.
    //
    // `RestoredFirstVersions::ComposedLineage` is where this arm parts company
    // with Test 1's `DecodedBase`, and it is the whole reason the exempt column
    // is worth passing per call site: this restore resolved `gathered_composed`
    // (asserted above), so the tail-created sections come back at the sequences
    // composition REPLAYED, while Test 1's collapsed artifact decodes them to
    // `1`. Both are correct outputs of their own restore, and both are `<
    // seeded_seq` — what the two sides of this file's comparison differ in is
    // CONTENT (identical) and the exempt column's provenance, never the filtered
    // columns.
    let restored_header = read_obj_header(&harness, fixture.object_id).await;
    let seeded_seq = header_u64(&restored_header, "seq");
    assert!(
        seeded_seq > fixture.head_seq,
        "precondition: the composed restore must seed the writer cursor strictly above the \
         lineage head ({seeded_seq} vs {})",
        fixture.head_seq
    );
    assert_restored_entry_versions_flattened(
        &restored,
        &fixture.head_oracle.snapshot,
        RestoredVersionRanges {
            replayed_range_floor: fixture.s0,
            lineage_root: fixture.s0,
        },
        seeded_seq,
        RestoredFirstVersions::ComposedLineage,
        FaithfulColumnsOptions::default(),
        "pre-re-base composed restore at S",
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Interleaving B — a DR restore lands mid-lifecycle and the lifecycle continues
// ═════════════════════════════════════════════════════════════════════════════

/// Epic AC (lifecycle half), `t3.dfh.be.hm.q2`: a DR restore landing in the
/// MIDDLE of a hot layer's tail lifecycle mints a base that **joins the lineage
/// arithmetic** — the layer keeps capturing on it, the chain reaches `K`, a
/// chain-collapse re-base fires on the restore-base lineage, and retention
/// retires the pre-DR lineage whole rather than stranding it.
///
/// The composition, in the order the test drives it:
///
/// 1. **Mid-lifecycle.** A real-commit `1 base + j tails` lineage at
///    [`PRE_DR_TAIL_RANGES`] (`j = 1 < K = 2`) on a wide layer — a lineage that
///    has NOT reached the bound, which is what makes the restore an interruption
///    rather than an epilogue.
/// 2. **The DR event.** [`wipe_layer_state`] plus a production-handler
///    `handle_stat_layer` restore under the stock production hook wiring
///    (`wire_restore_rebase` at its `true` default). The restore gathers the
///    composed pre-DR lineage; SR-3.6 then mints a lone `Base@seeded_seq`,
///    retires `restore_seq_bump`, and stamps `last_checkpoint_seq = seeded_seq`.
/// 3. **Cell separation, across that step.** SR-3.6's re-base is a WRITE:
///    `checkpoint_writes_total{success, layer}` moves by exactly one and the
///    ENTIRE `checkpoint_tail_total` vocabulary is flat — `rebased` included,
///    that cell being reserved for SR-4.1's chain-collapse re-base.
/// 4. **The lifecycle continues.** The writer stays hot, the loop captures `K`
///    tails rooted at the restore-base, and a chain-collapse re-base fires
///    THERE: `rebased` +1, and only there.
/// 5. **The third-base-producer arithmetic**, at `R = 2`
///    ([`LIFECYCLE_RETAIN_GENERATIONS`]): the restore-minted base consumed a
///    retain slot, so retention retires the pre-DR lineage — and it retires it
///    WHOLE, base and tail together, rather than stranding the tail.
///
/// # The loop-level finding this arm turned up
///
/// This arm and its in-crate sibling (`checkpointer.rs`'s
/// `rebase_lifecycle_tests`) were the first drives in the repo to collapse a
/// chain the LOOP captured itself, and they found that such a collapse
/// deterministically skipped its own cursor stamp — and therefore its own
/// `{success, layer}` tick and its own retention prune. The head tail's capture
/// already stamped `last_checkpoint_seq = to_seq`, the collapse re-bases at
/// exactly that `to_seq`, and the guard was `base_bootstrap`'s shared
/// `cursor_regression_skips` (`observed >= target`), which blocks on equality —
/// every time, not as a race.
///
/// The fix is `checkpointer::collapse_cursor_blocks_stamp`: the collapse site
/// blocks only on `observed > to_seq`, so equality stamps (idempotently on
/// `seq`, freshly on `ms`) and the arm continues into its success path. What
/// the assertions below therefore pin is the FIXED contract —
/// `checkpoint_writes_total{success, layer}` moves by exactly one for a
/// production chain collapse, beside `tail_total{rebased}`. Test 1 never meets
/// the equality case at all, because its `K` tails are hand-put and never stamp
/// the cursor; that is exactly the difference between a planted chain and a
/// lived one, and it is why this arm exists.
///
/// # Why `wipe_layer_state` and not `full_valkey_wipe`
///
/// Both are DR events and the Subtask allows either. The partial-loss shape is
/// chosen because it isolates the claim: everything this arm asserts is about the
/// LAYER's checkpoint lineage — its arithmetic, its metric cells, its retention —
/// and a whole-keyspace wipe would additionally drag in the session-recovery hop,
/// the path index and the object-id counter re-seed, every one of which is
/// `e2e_full_wipe_composed_restore_test.rs`'s owned claim and would be restated
/// noise here. It also keeps this file's DR posture identical to Test 1's, so the
/// two arms differ in the INTERLEAVING under test and in nothing else. The
/// whole-keyspace shape is not skipped, it is already covered next door.
///
/// # Why the fence, in both of its modes
///
/// Neither the capture phase nor the collapse can be left to a poll. A layer
/// with ONE assembly the seam does not skip lands an ordinary base at the LIVE
/// head, which resets the chain to zero and — worse for this arm specifically —
/// re-roots every subsequent capture away from the restore-base, silently
/// falsifying the `base_snap_seq == seeded_seq` claim rather than failing it
/// honestly. [`RebaseFenceStore::arm_capture_only`] makes that structurally
/// impossible, and because it also holds the collapse it FREEZES the store at
/// chain-`K` — which is what lets the restore-cost measurement below run against
/// a store that cannot move under it. [`RebaseFenceStore::arm`] then releases
/// exactly one address.
///
/// # What this arm does NOT restate
///
/// `e2e_full_wipe_composed_restore_test.rs` is untouched by this Subtask. This
/// arm extends its Test 2
/// (`a_restored_never_quiescent_layer_captures_its_next_tail_instead_of_refusing`),
/// which proves the FIRST hop — a restored never-quiescent layer captures its
/// next tail instead of losing DR coverage forever — and stops there; none of its
/// assertions are re-made here, and in particular Test 3's failed-re-base
/// `refused_seq_gap` refusal arm is landed there and is not re-derived. Exact
/// state fidelity across a collapse is Test 1's above. The per-prune ORDERING
/// (tails newest-first, base last) is Task 4's at unit and store level; what this
/// arm asserts is the joint lifecycle consequence.
///
/// Compiled only under `test-checkpoint-fault-injection` (the unprovable-cut
/// seam); without the feature it is compiled out, not skipped.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_lifecycle_continues_across_a_dr_restore_onto_the_restore_minted_base() {
    let fence = Arc::new(RebaseFenceStore::new());
    let store: Arc<dyn CheckpointStore> = Arc::clone(&fence) as Arc<dyn CheckpointStore>;
    let mut harness = setup_rebase_harness(store).await;

    // R, stated against the harness's own Config rather than only against the
    // loop this test spawns, so "the default retain" is a read value and not a
    // claim in a comment.
    assert_eq!(
        harness.config.checkpoint_retain_generations as usize, LIFECYCLE_RETAIN_GENERATIONS,
        "fixture precondition: this arm's third-base-producer claim is stated at R = \
         {LIFECYCLE_RETAIN_GENERATIONS} (the production default), and the loop below is spawned \
         with exactly that. A harness Config carrying a different quota would mean the two halves \
         of the fixture disagree about R"
    );

    // ── 1. The mid-lifecycle state: a lineage at j < K, loop already fenced ──
    let fixture = setup_rebase_lineage(&mut harness, "dr-mid-lifecycle", PRE_DR_TAIL_RANGES).await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    let pre_dr = listing(&harness, fixture.object_id).await;
    assert_eq!(
        pre_dr
            .first()
            .map(|h| (h.kind, h.snap_seq))
            .unwrap_or_else(|| panic!("the listing must be non-empty. Listing: {pre_dr:?}")),
        (CheckpointKind::Tail, fixture.head_seq),
        "PRECONDITION: the DR event must land on an UN-collapsed lineage whose head is a Tail — \
         that is what makes this a restore in the middle of a lifecycle rather than after one. \
         Listing: {pre_dr:?}"
    );
    assert_eq!(
        pre_dr
            .iter()
            .filter(|h| h.kind == CheckpointKind::Tail)
            .count(),
        PRE_DR_TAIL_RANGES,
        "and its chain must be at j = {PRE_DR_TAIL_RANGES}, strictly under K ({MAX_CHAIN}): a \
         lineage already ON the bound would collapse on the first eligible tick and there would \
         be no mid-lifecycle state left to interrupt. Listing: {pre_dr:?}"
    );

    // ── 2. The DR event ─────────────────────────────────────────────────────
    let tail_before_restore = gather_tail_outcome_cells(&harness.metrics);
    let bootstrap_before_restore = gather_base_bootstrap_cells(&harness.metrics);
    let layer_writes_before_restore = read_layer_writes_counter(&harness.metrics, "success");
    let tail_writes_before_restore = read_tail_writes_counter(&harness.metrics, "success");
    let lineage_before_restore = lineage_cells(&harness);

    wipe_layer_state(&harness.tv, fixture.object_id).await;
    assert!(
        !key_exists(
            &harness,
            &harness
                .tv
                .key_schema()
                .obj_header(&fixture.object_id.to_string())
        )
        .await,
        "wipe_layer_state must remove obj:hdr; a surviving header means StatLayer would answer \
         from live state and the recovery branch would silently not fire"
    );

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "StatLayer after the mid-lifecycle layer-state wipe MUST restore the composed lineage \
             and return Ok, got {:?}: {}",
            status.code(),
            status.message()
        )
    });
    // The SR-3.6 re-base is fire-and-forget on the session-checkpoint tracker;
    // every witness below reads something it stamps.
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    // The restore itself gathered the mid-lifecycle CHAIN — the other half of
    // "j < K": had the pre-DR lineage been a lone base this would read
    // `gathered_base` and the interleaving would be a different one.
    assert_lineage_deltas(
        &harness,
        &lineage_before_restore,
        &[("gathered_composed", 1)],
        "the mid-lifecycle DR restore",
    );

    // ── The SR-3.6 landed-witnesses ─────────────────────────────────────────
    let restored_header = read_obj_header(&harness, fixture.object_id).await;
    let seeded_seq = header_u64(&restored_header, "seq");
    assert!(
        seeded_seq > fixture.head_seq,
        "post-b.8y5 the guard band is unconditional, so the restore must seed the writer cursor \
         strictly above the composed head ({seeded_seq} vs {})",
        fixture.head_seq
    );
    assert_eq!(
        header_u64_or_zero(&restored_header, "restore_seq_bump"),
        0,
        "the SR-3.6 re-base must RETIRE the fossil guard's input. While this column is non-zero \
         `tail_capture` refuses `PostRestoreUnrebased` BEFORE the stream read on every tick \
         forever — and `rebase_on_chain_full`'s FIRST gate declines a collapse outright — so the \
         whole lifecycle this arm drives would be unreachable"
    );
    assert_eq!(
        header_u64(&restored_header, "last_checkpoint_seq"),
        seeded_seq,
        "and a successful re-base advances the cursor to the Base it just wrote"
    );
    let post_restore = listing(&harness, fixture.object_id).await;
    assert_eq!(
        post_restore
            .first()
            .map(|h| (h.kind, h.snap_seq))
            .unwrap_or_else(|| panic!("the listing must be non-empty. Listing: {post_restore:?}")),
        (CheckpointKind::Base, seeded_seq),
        "the SR-3.6 re-base must have landed a lone Base at the seeded cursor ({seeded_seq}), and \
         it must be the listing HEAD — `seeded_seq = restore_floor + band` with `band >= 1` \
         unconditionally, so it sorts strictly above every generation already in the store. That \
         is what makes it a lineage root the next capture can chain onto. Listing: \
         {post_restore:?}"
    );
    assert_bootstrap_deltas(
        &bootstrap_before_restore,
        &gather_base_bootstrap_cells(&harness.metrics),
        &[(("persisted", "restore_rebase"), 1.0)],
        "the SR-3.6 post-restore re-base",
    );

    // ── 3. Cell separation, ACROSS the restore step ─────────────────────────
    //
    // The binding deferral this lane discharges. SR-3.6's re-base is a WRITE and
    // nothing more: it must move `writes_total{*, layer}` and NOT the
    // `checkpoint_tail_total{rebased}` cell, which is SR-4.1's alone. Read as a
    // whole-vocabulary before/after map (see [`assert_tail_deltas`]) rather than
    // as a probe on `rebased`, so a tick that landed on some OTHER arm — or on an
    // out-of-band outcome that materialises its series on first occurrence — is a
    // failure here instead of being invisible.
    let tail_after_restore = gather_tail_outcome_cells(&harness.metrics);
    assert_tail_deltas(
        &tail_before_restore,
        &tail_after_restore,
        &[],
        "across the SR-3.6 post-restore re-base (EVERY tail outcome must be flat: this step \
         writes a base, it captures nothing and it collapses nothing)",
    );
    assert_eq!(
        read_layer_writes_counter(&harness.metrics, "success") - layer_writes_before_restore,
        1,
        "…while `checkpoint_writes_total{{result=\"success\", kind=\"layer\"}}` moved by exactly \
         ONE. Both halves are the claim: a flat tail family alone would also be satisfied by a \
         re-base that never ran at all, which is the reading this pairing rules out"
    );
    assert_eq!(
        read_tail_writes_counter(&harness.metrics, "success") - tail_writes_before_restore,
        0,
        "and `kind=\"layer_tail\"` must be flat — SR-3.6 mints a BASE (SR-8.2). A tick here would \
         mean the restore produced a tail artifact, which no arm of the bootstrap unit does"
    );

    // ── 4a. The lifecycle continues: capture K tails on the restore-base ────
    //
    // Capture-only from here, so nothing but a Tail can land while the loop
    // builds the chain. See the test's own "Why the fence" section.
    fence.arm_capture_only(fixture.object_id);
    let attempts_before_fill = fence.attempts().len();
    let tail_before_fill = gather_tail_outcome_cells(&harness.metrics);

    let writer = HotWriter::spawn(
        &harness.tv,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.object_id,
    );
    writer.warm_up().await;
    let (fill_cancel, fill_loop) =
        spawn_tail_loop(&harness, fixture.object_id, LIFECYCLE_RETAIN_GENERATIONS).await;

    let fill_baseline_refusals = *tail_before_fill.get("refused_chain_full").unwrap_or(&0.0);
    let filled = {
        let store = Arc::clone(&harness.store);
        let metrics = Arc::clone(&harness.metrics);
        let object_id = fixture.object_id;
        poll_until(CHAIN_FILL_DEADLINE, || {
            let store = Arc::clone(&store);
            let metrics = Arc::clone(&metrics);
            async move {
                // BOTH halves, in this order. The metric half is the LOOP's own
                // verdict that it reached the `ChainFull` arm and was held there,
                // which is what makes "the chain is at K" a fact about the
                // production path rather than about artifact bookkeeping; the
                // listing half is the state the measurement below reads.
                let held = gather_tail_outcome_cells(&metrics)
                    .get("refused_chain_full")
                    .is_some_and(|value| *value > fill_baseline_refusals);
                if !held {
                    return false;
                }
                store
                    .list_layer_checkpoints(object_id)
                    .await
                    .map(|handles| {
                        handles
                            .iter()
                            .filter(|h| h.kind == CheckpointKind::Tail && h.snap_seq > seeded_seq)
                            .count()
                            >= MAX_CHAIN
                    })
                    .unwrap_or(false)
            }
        })
        .await
    };

    // The loop is drained but the WRITER keeps running: the release phase below
    // needs the layer DUE on every tick, and a layer with no commit above its
    // cursor is dropped by the skip-if-no-commits gate before the `ChainFull`
    // arm can run.
    drain_loop(fill_cancel, fill_loop).await;

    let tail_after_fill = gather_tail_outcome_cells(&harness.metrics);
    let filled_listing = listing(&harness, fixture.object_id).await;
    let fill_attempts = fence.attempts_since(attempts_before_fill);
    assert!(
        filled,
        "the restored layer MUST capture {MAX_CHAIN} tails onto the SR-3.6 restore-base and reach \
         the held `ChainFull` arm within {CHAIN_FILL_DEADLINE:?}. {}. Listing: {filled_listing:?}. \
         Fence op log: {fill_attempts:?}. Outcome cells before: {tail_before_fill:?}, after: \
         {tail_after_fill:?}. A rising `refused_seq_gap` means the SR-3.6 re-base did not retire \
         `restore_seq_bump`; a rising `refused_no_base` means it landed no Base; a flat family \
         altogether means the unprovable-cut seam did not fire and the capture fallback was never \
         reached",
        writer.report()
    );

    // Which of the two capture labels fired is SR-1.4's cap decision, not this
    // arm's claim, so they are read as a SUM: both produce exactly one Tail
    // artifact and both advance the chain by one link.
    let captured_delta = tail_delta(&tail_before_fill, &tail_after_fill, "captured");
    let capped_delta = tail_delta(&tail_before_fill, &tail_after_fill, "captured_capped");
    assert_eq!(
        captured_delta + capped_delta,
        MAX_CHAIN as f64,
        "exactly K ({MAX_CHAIN}) tails must have been captured onto the restore-base — the chain \
         stops there because the capture unit's pre-filter returns `ChainFull` from then on. \
         captured {captured_delta}, captured_capped {capped_delta}; before {tail_before_fill:?}, \
         after {tail_after_fill:?}"
    );
    assert_tail_deltas(
        &tail_before_fill,
        &tail_after_fill,
        &[
            // Pinned by the SUM assertion immediately above; unconstrained here
            // so the split between them is not over-specified.
            ("captured", TailDelta::AtLeast(0.0)),
            ("captured_capped", TailDelta::AtLeast(0.0)),
            // The held collapse. Positive evidence the loop REACHED the
            // `ChainFull` → re-base arm: SR-4.5 folds the fence's refused base
            // PUT (`RebaseRefusal::BasePut`) onto this cell.
            ("refused_chain_full", TailDelta::AtLeast(1.0)),
        ],
        "the post-restore capture phase (`rebased` must be FLAT here: the chain reached K but the \
         fence held the collapse, so no re-base landed)",
    );

    // The fence's own verdict on the capture phase, off the op log.
    let fill_accepted: Vec<PutAttempt> = fill_attempts
        .iter()
        .copied()
        .filter(|attempt| attempt.accepted)
        .collect();
    assert!(
        fill_accepted.iter().all(|attempt| {
            attempt.kind == CheckpointKind::Tail
                && attempt.object_id == fixture.object_id
                && attempt.snap_seq > seeded_seq
        }),
        "while capture-only, the ONLY generations the fence may admit are Tails above the \
         restore-base ({seeded_seq}). A Base here would have reset the chain and re-rooted every \
         later capture away from the restore-base. Full op log: {fill_attempts:?}"
    );
    assert_eq!(
        fill_accepted.len(),
        MAX_CHAIN,
        "and exactly K ({MAX_CHAIN}) of them. Full op log: {fill_attempts:?}"
    );

    let mut captured_tails: Vec<&CheckpointHandle> = filled_listing
        .iter()
        .filter(|h| h.kind == CheckpointKind::Tail && h.snap_seq > seeded_seq)
        .collect();
    captured_tails.sort_by_key(|handle| handle.snap_seq);
    assert_eq!(
        captured_tails.len(),
        MAX_CHAIN,
        "the store must hold exactly K ({MAX_CHAIN}) tails above the restore-base. Listing: \
         {filled_listing:?}"
    );
    // **S'** — the address the chain-collapse re-base will PUT at: the lineage
    // head, i.e. the newest captured tail's `to_seq`. Not knowable before this
    // point, which is exactly why the capture phase needed the kind-keyed fence
    // mode rather than the address-keyed one.
    let collapse_seq = captured_tails
        .last()
        .expect("the count was just asserted")
        .snap_seq;

    // ── 4b. The restore-base joined the arithmetic — and the chain-at-K cost ─
    //
    // Read off ONE `gather_restore_lineage`, i.e. through the production restore
    // path's own link walk rather than through a re-derivation of it. The store
    // is frozen (capture-only holds every Base, and the loop is drained), so the
    // fetch delta addresses this gather and nothing else.
    let fetches_before_k_gather = fence.layer_fetches();
    let mut k_selection = gather_restore_lineage(
        harness.store.as_ref(),
        fixture.object_id,
        no_deadline(),
        GATHER_FETCH_RESERVE,
    )
    .await
    .expect("the single listing behind a gather must succeed");
    let k_lineage = k_selection
        .next_restorable()
        .await
        .expect("a healthy chain-at-K lineage is never an error")
        .expect("a restore-base plus K contiguous captured tails must be restorable");
    let k_fetches = fence.layer_fetches() - fetches_before_k_gather;

    assert_eq!(
        k_lineage.base_snap_seq(),
        seeded_seq,
        "the chain the loop built roots at the SR-3.6 RESTORE-BASE ({seeded_seq}) — the whole \
         third-base-producer claim in one number. A different root means the restore-minted base \
         was bypassed and the captures chained onto a pre-restore generation, which would replay \
         over an abandoned timeline"
    );
    assert_eq!(
        k_lineage.chain_len(),
        MAX_CHAIN,
        "and the link walk resolves K ({MAX_CHAIN}) tails onto it — the TRUE chain length SR-9.2 \
         specifies, not the listing-position approximation"
    );
    assert_eq!(
        k_lineage.head_seq, collapse_seq,
        "whose head is the newest captured tail's to_seq"
    );
    for tail in &k_lineage.tails {
        assert_eq!(
            tail.header.base_snap_seq, seeded_seq,
            "and every member's own envelope header names the restore-base ({seeded_seq}) as its \
             root — `lineage_shape` derives a new tail's `base_snap_seq` from the newest Base in \
             the listing at capture time, so this is the writer's own attribution, read back"
        );
    }
    assert!(
        k_selection.step_downs().is_empty(),
        "the gather must resolve the chain-at-K lineage with NO step-downs; got {:?}",
        k_selection.step_downs()
    );
    assert_eq!(
        k_fetches,
        1 + MAX_CHAIN as u64,
        "restore cost at the WORST point of the lifecycle: exactly 1 base + K tails ({}) \
         `get_layer_checkpoint` calls. This is the bound the Epic AC states, measured at the one \
         moment the chain is actually on it",
        1 + MAX_CHAIN
    );

    // ── 4c. Release the collapse: `rebased` +1, and only here ────────────────
    let tail_before_collapse = gather_tail_outcome_cells(&harness.metrics);
    let layer_writes_before_collapse = read_layer_writes_counter(&harness.metrics, "success");
    let attempts_before_collapse = fence.attempts().len();
    fence.arm(fixture.object_id, collapse_seq);
    let (collapse_cancel, collapse_loop) =
        spawn_tail_loop(&harness, fixture.object_id, LIFECYCLE_RETAIN_GENERATIONS).await;

    let rebased = poll_tail_outcome_rose(
        &harness.metrics,
        &tail_before_collapse,
        "rebased",
        REBASE_DEADLINE,
    )
    .await;

    let collapse_writer_report = writer.report();
    drain_loop(collapse_cancel, collapse_loop).await;

    let tail_after_collapse = gather_tail_outcome_cells(&harness.metrics);
    let collapsed_listing = listing(&harness, fixture.object_id).await;
    let collapse_attempts = fence.attempts_since(attempts_before_collapse);
    assert!(
        rebased,
        "the periodic loop MUST collapse the restore-base lineage once the fence admits \
         Base@{collapse_seq}, within {REBASE_DEADLINE:?}. {collapse_writer_report}. Listing: \
         {collapsed_listing:?}. Fence op log: {collapse_attempts:?}. Outcome cells before: \
         {tail_before_collapse:?}, after: {tail_after_collapse:?}"
    );
    assert_tail_deltas(
        &tail_before_collapse,
        &tail_after_collapse,
        &[("rebased", TailDelta::Exactly(1.0))],
        "the chain-collapse release window (`rebased` +1 HERE and nowhere else in this test; \
         `refused_chain_full` must now be flat because the fence admits the address the collapse \
         PUTs at)",
    );
    let collapse_accepted: Vec<PutAttempt> = collapse_attempts
        .iter()
        .copied()
        .filter(|attempt| attempt.accepted)
        .collect();
    assert_eq!(
        collapse_accepted,
        vec![PutAttempt {
            object_id: fixture.object_id,
            snap_seq: collapse_seq,
            kind: CheckpointKind::Base,
            accepted: true,
        }],
        "and the ONLY generation the fence admitted over the release window is the collapse's own \
         Base at the lineage head S' ({collapse_seq}). Full op log: {collapse_attempts:?}"
    );
    // ── The EQUALITY case, pinned rather than narrated ───────────────────────
    //
    // This is the shape only a LIVED chain presents, and it is the one the
    // production defect this file's header records lived in:
    //
    //   the capture that produced the head tail already stamped
    //   `last_checkpoint_seq = to_seq`, and the collapse re-bases at exactly that
    //   `to_seq`, so the pre-stamp guard sees `observed == to_seq` on every
    //   collapse — deterministically, not as a race.
    //
    // `collapse_cursor_blocks_stamp` (`observed > to_seq`) admits it, so the arm
    // stamps — idempotently on `last_checkpoint_seq`, freshly on
    // `last_checkpoint_ms` — ticks its write cell and reaches
    // `prune_after_write`. Under the pre-fix shared `>=` guard this delta read
    // ZERO, which is what made `writes_total{success, layer}` unusable as a "a
    // chain collapse landed and was recorded" signal and deferred retention to
    // the next successful write. Test 1 cannot see any of this: its `K` tails are
    // HAND-PUT and never stamp the cursor, leaving `last_checkpoint_seq` at
    // `S0 < S`, i.e. permanently on the below-target branch.
    assert_eq!(
        read_layer_writes_counter(&harness.metrics, "success") - layer_writes_before_collapse,
        1,
        "the collapse must tick EXACTLY ONE `checkpoint_writes_total{{result=\"success\", \
         kind=\"layer\"}}` cell — kind=\"layer\" because a re-base produces a BASE. A ZERO here is \
         the pre-fix defect returning: the stamp blocked at equality against the cursor the head \
         tail's own capture had already advanced to the same sequence, taking the write cell and \
         the retention prune with it. Check `collapse_cursor_blocks_stamp` rather than softening \
         this. Fence op log: {collapse_attempts:?}"
    );

    // ── 4d. Retention, and one capture onto the collapsed base ───────────────
    //
    // Retention runs from `prune_after_write`, which the collapse's own arm calls
    // once its stamp lands — so with THREE units in the listing at R =
    // `LIFECYCLE_RETAIN_GENERATIONS` the pre-DR unit is normally already retired
    // by the time the collapse tick ends. The poll is kept rather than replaced
    // by a single read because the collapse loop is CANCELLED as soon as
    // `rebased` rises (that cell ticks before the stamp), so the remainder of
    // that tick may be cut off mid-flight; whether the prune belongs to the
    // collapse tick or to the write that follows it is a claim the in-crate
    // `rebase_lifecycle_tests` phase C owns at tick granularity, and this arm
    // deliberately does not re-make it at loop granularity.
    //
    // The second observation is a REQUIREMENT of the (b)/(c) restore-cost
    // assertions at the end, not decoration: they need at least one tail rooted
    // at the collapsed base, and before the cursor fix that capture was implied
    // by the prune poll (the deferred prune could only run on a later successful
    // write, i.e. on that very capture). It no longer is, so it is waited for
    // explicitly.
    //
    // The fence goes back to capture-only — admitting tails, still refusing every
    // base. Deliberately not "disarm and let anything land": an unfenced
    // assembler base would open a fourth unit and change the arithmetic under the
    // assertions.
    fence.arm_capture_only(fixture.object_id);
    let (prune_cancel, prune_loop) =
        spawn_tail_loop(&harness, fixture.object_id, LIFECYCLE_RETAIN_GENERATIONS).await;
    let pruned = {
        let store = Arc::clone(&harness.store);
        let object_id = fixture.object_id;
        let (s0, pre_dr_head) = (fixture.s0, fixture.head_seq);
        poll_until(PRUNE_DEADLINE, || {
            let store = Arc::clone(&store);
            async move {
                store
                    .list_layer_checkpoints(object_id)
                    .await
                    .map(|handles| {
                        !handles.iter().any(|h| {
                            (h.kind == CheckpointKind::Base && h.snap_seq == s0)
                                || (h.kind == CheckpointKind::Tail && h.snap_seq == pre_dr_head)
                        })
                    })
                    .unwrap_or(false)
            }
        })
        .await
    };
    let recaptured = {
        let store = Arc::clone(&harness.store);
        let object_id = fixture.object_id;
        poll_until(PRUNE_DEADLINE, || {
            let store = Arc::clone(&store);
            async move {
                store
                    .list_layer_checkpoints(object_id)
                    .await
                    .map(|handles| {
                        handles
                            .iter()
                            .any(|h| h.kind == CheckpointKind::Tail && h.snap_seq > collapse_seq)
                    })
                    .unwrap_or(false)
            }
        })
        .await
    };

    let writer_report = writer.report();
    writer.stop_and_join().await;
    drain_loop(prune_cancel, prune_loop).await;
    let final_listing = listing(&harness, fixture.object_id).await;

    // ── 5. The third-base-producer arithmetic, at R = 2 ─────────────────────
    //
    // Three lineage units exist once the collapse has landed — the chain-0
    // `base@S'`, the restore-base unit `base@seeded_seq + K tails`, and the pre-DR
    // unit `base@S0 + tail@S`. The SR-3.6 base is the THIRD producer: it consumed
    // a retain slot, and at R = 2 that is what pushes the pre-DR lineage out of
    // the window — exactly the consequence `retention.rs`'s own header states for
    // an operator ("a DR restore immediately demotes the pre-restore lineage to
    // the last retained slot"). What the binding deferral asks is whether it goes
    // out WHOLE.
    //
    // Falsifiable, not decorative: under the pre-Epic BASE-COUNT prune the delete
    // set here would be `{base@S0}` alone and `tail@S` would survive with no base
    // to replay it onto — which is exactly what assertion (a) below catches. That
    // implementation, and the values it produces, are recorded in
    // `retention.rs`'s own mutation section.
    assert!(
        pruned,
        "retention MUST retire the superseded pre-DR lineage within {PRUNE_DEADLINE:?} of the \
         collapse — three units at R = {LIFECYCLE_RETAIN_GENERATIONS} means the oldest is over \
         quota, and `rebase_on_chain_full` ends by calling `prune_after_write`. {writer_report}. \
         Listing: {final_listing:?}"
    );
    assert!(
        recaptured,
        "and the lifecycle must RESUME on the collapsed base — at least one tail captured above S' \
         ({collapse_seq}) within {PRUNE_DEADLINE:?}, with the fence admitting tails only. This is \
         the precondition the restore-cost assertions (b)/(c) below stand on: without a tail on \
         the new base the walk they measure is a bare chain-0 base and the `1 base + chain_len \
         tails` identity is vacuous. {writer_report}. Listing: {final_listing:?}"
    );
    assert!(
        !final_listing
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.snap_seq == fixture.s0)
            && !final_listing
                .iter()
                .any(|h| h.kind == CheckpointKind::Tail && h.snap_seq == fixture.head_seq),
        "and it must go WHOLE — base@S0 ({}) AND tail@S ({}) together. Listing: {final_listing:?}",
        fixture.s0,
        fixture.head_seq
    );
    assert!(
        final_listing
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.snap_seq == seeded_seq),
        "while the restore-base ({seeded_seq}) survives: it roots the retained unit that carries \
         the K captured tails, and deleting it is the exact stranding this arm exists to rule \
         out. Listing: {final_listing:?}"
    );

    // (a) No retained tail is left without its base. Read per tail off the
    //     AUTHORITATIVE envelope header rather than off listing adjacency, so a
    //     prune that deleted a base while leaving tails rooted at it fails here
    //     rather than being papered over by the same grouping that planned the
    //     delete.
    let surviving_bases: HashSet<u64> = final_listing
        .iter()
        .filter(|h| h.kind == CheckpointKind::Base)
        .map(|h| h.snap_seq)
        .collect();
    for handle in final_listing
        .iter()
        .filter(|h| h.kind == CheckpointKind::Tail)
    {
        let blob = harness
            .store
            .get_layer_checkpoint(handle)
            .await
            .expect("get_layer_checkpoint must succeed")
            .expect("a handle from the listing must address a live artifact");
        let envelope = decode_tail_envelope(&blob.bytes)
            .expect("every retained tail artifact must be a decodable envelope");
        let root = envelope.header().base_snap_seq;
        assert!(
            surviving_bases.contains(&root),
            "STRANDED TAIL — tail@{} roots at base@{root}, which retention deleted. Surviving \
             bases: {surviving_bases:?}. Listing: {final_listing:?}. This is a defect in \
             lineage-unit retention (Task 4, `t2.dfh.be.mk`), NOT an assertion to soften: a tail \
             whose base is gone can never be replayed, and the pre-Epic base-count prune is \
             exactly the implementation that produces this shape. ESCALATE",
            handle.snap_seq
        );
    }

    // (b) + (c) The lifecycle ends with a restore the gather resolves cleanly and
    //     at a cost still inside the bound — and, the point of the collapse,
    //     rooted at the COLLAPSED base rather than at the restore-base, so the
    //     chain the walk pays for is the handful of tails captured since the
    //     collapse rather than the `K` it replaced.
    let fetches_before_final_gather = fence.layer_fetches();
    let mut final_selection = gather_restore_lineage(
        harness.store.as_ref(),
        fixture.object_id,
        no_deadline(),
        GATHER_FETCH_RESERVE,
    )
    .await
    .expect("the single listing behind a gather must succeed");
    let final_lineage = final_selection
        .next_restorable()
        .await
        .expect("a healthy post-collapse lineage is never an error")
        .expect("the collapsed base must be restorable");
    let final_fetches = fence.layer_fetches() - fetches_before_final_gather;

    assert!(
        final_selection.step_downs().is_empty(),
        "(b) after the whole lifecycle the gather must still resolve a COMPLETE lineage with no \
         step-downs — a step-down here would mean retention removed something the newest \
         restorable lineage still links to. Got {:?}. Listing: {final_listing:?}. ESCALATE to \
         Task 4 rather than relaxing this",
        final_selection.step_downs()
    );
    assert_eq!(
        final_lineage.base_snap_seq(),
        collapse_seq,
        "and it roots at the collapsed base S' ({collapse_seq})"
    );
    assert!(
        (1..=MAX_CHAIN).contains(&final_lineage.chain_len()),
        "carrying between one and K ({MAX_CHAIN}) tails: phase 4d above waited for at least one \
         capture onto the collapsed base, and the capture unit's `ChainFull` pre-filter stops it \
         at K. Got {}. Listing: {final_listing:?}",
        final_lineage.chain_len()
    );
    assert!(
        final_fetches <= 1 + MAX_CHAIN as u64,
        "(c) restore cost after the whole lifecycle must still be at most 1 base + K tails ({}); \
         paid {final_fetches}. The collapse is what keeps it there — without it the walk would \
         still be crossing the K tails the collapsed base replaced, on TOP of the ones captured \
         since. Listing: {final_listing:?}",
        1 + MAX_CHAIN
    );
    assert_eq!(
        final_fetches,
        1 + final_lineage.chain_len() as u64,
        "and the measured cost is exactly `1 base + chain_len tails` — stated as an identity \
         rather than only as an inequality, so a gather that started paying for step-downs or \
         re-fetches would fail here rather than sit quietly under the bound"
    );
}
