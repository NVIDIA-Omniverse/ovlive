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

//! Write-atomicity integration tests for `LocalDiskCheckpointStore`.
//!
//! Subtask `t3.fgj.ya.r7.xm` under Task `t2.fgj.ya.r7`, extended for tail
//! generations by Subtask `t3.dfh.gy.ib.8s`. Verifies the write-temp +
//! rename atomicity invariant on the LocalDisk adapter via five
//! complementary tests — two for the base OMPF, three for the tail pair:
//!
//! 1. [`temp_file_naming_no_simultaneous_visibility`] — lower-fidelity, no
//!    child process. Polls the layers directory during a concurrent
//!    `put_layer_checkpoint` and asserts that the final `<oid>.<snap>.ompf`
//!    name and any `<oid>.<snap>.*.tmp` companion are never observable at
//!    the same instant. POSIX `rename(2)` guarantees this — the final path
//!    only becomes visible when the temp path is unlinked atomically.
//!
//! 2. [`mid_write_sigkill_leaves_no_published_file`] — high-fidelity,
//!    spawns the [`local_disk_write_kill_harness`] example binary, waits
//!    for its `STARTED_WRITE` ready marker, then `SIGKILL`s it mid-write.
//!    SIGKILL bypasses Drop handlers and the harness's tokio runtime
//!    shutdown, so any temp file the harness was streaming is left
//!    half-written. A fresh `LocalDiskCheckpointStore` opened on the same
//!    root MUST return `Ok(None)` from `get_latest_layer_checkpoint`
//!    because the `rename(2)` step never completed.
//!
//! 3. [`tail_temp_file_naming_no_simultaneous_visibility`] — test 1's shape
//!    for a `<oid>.<to_seq>.tail` put. Adds a second, stronger assertion the
//!    base test does not carry: whenever the final `.tail` name IS observed,
//!    its size already equals the full payload. That is what distinguishes
//!    rename-published from streamed-in-place, and it is the assertion that
//!    fails if a tail write ever bypasses `atomic_write` for a direct
//!    `tokio::fs::write` to the final path — a regression an inline unit
//!    test cannot see, because a unit test only observes the post-put
//!    steady state.
//!
//! 4. [`tail_artifact_never_visible_without_its_meta`] — the publish-**order**
//!    half, which `atomic_write` alone does not give (it makes each file
//!    atomic, not the pair ordered). LocalDisk publishes a tail meta-FIRST,
//!    then the envelope by rename, precisely so that a `.tail`-stem-driven
//!    `list_layer_checkpoints` never enumerates a tail whose meta is absent.
//!    Every poll iteration that observes the final `.tail` name must
//!    therefore also observe `<oid>.<to_seq>.tail.meta.json`. The converse is
//!    NOT asserted: under meta-first the meta legitimately exists alone for
//!    the length of the publish window.
//!
//! 5. [`tail_delete_never_leaves_the_artifact_without_its_meta`] — the same
//!    implication on the way out. `delete_layer_checkpoint` unlinks the
//!    envelope before the meta, the mirror of the publish order, and the
//!    adapter states the two as one claim holding "in both directions of the
//!    artifact's lifetime". Only a mid-delete observer can see a swap of
//!    those two unlinks; by the time the call returns both files are gone.
//!
//! Bounded polling per `docs-internal/test-writing-guide.md` `## Synchronization
//! Patterns` — total budget is 5 seconds, never an unconditional
//! `tokio::time::sleep`.
//!
//! # Why there is no tail SIGKILL variant
//!
//! Deliberate scoping, not an oversight. The rename-atomicity *mechanism* is
//! the shared [`LocalDiskCheckpointStore`] `atomic_write` helper — one code
//! path used by the base OMPF, the meta companions, the ext-refs sidecar and
//! the tail envelope alike — and test 2 already pins it against SIGKILL. A
//! tail arm would require extending `examples/local_disk_write_kill_harness.rs`
//! (an engineer-lane, non-test file) to grow a tail mode, for zero new
//! mechanism coverage: it would re-prove `rename(2)`'s atomicity on a second
//! filename. What IS tail-specific — the publish order and the
//! full-size-on-visibility property — is not observable through a
//! kill-and-inspect-the-corpse test at all, and is covered by tests 3 and 4
//! above, which watch the directory *while* the put is in flight.
//!
//! [`local_disk_write_kill_harness`]: ../../../examples/local_disk_write_kill_harness.rs

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{CheckpointHandle, CheckpointKind, CheckpointMeta, CheckpointStore};
use tempfile::{tempdir, TempDir};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// 5-second cap on every polling loop in this file, per the test-writing
/// guide's `## Synchronization Patterns` rule.
const POLL_BUDGET: Duration = Duration::from_secs(5);

/// Probe cadence inside the base-OMPF polling loop. Short enough that we
/// observe the `.tmp` file before the rename completes on most machines, but
/// not a busy-loop. The tail loop deliberately paces on `yield_now` instead —
/// see [`concurrent_tail_put_probed`] for the measurement that forced it.
const POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Payload size for both atomicity tests. Large enough that
/// `tokio::fs::write` takes a non-trivial slice of wall time so the
/// observer / killer has a reasonable window in which to act, while
/// staying small enough that the test does not stress CI disk quotas.
/// 8 MiB matches the Subtask body's recommendation.
const PAYLOAD_BYTES: usize = 8 * 1024 * 1024;

/// Test 1 — temp-file naming verification (no child process).
///
/// Spawns `put_layer_checkpoint` on a background task and concurrently
/// polls the `<root>/.live-edit/layers/` directory. The assertion is that
/// we never observe both `<N>.<K>.ompf` and a `<N>.<K>.*.tmp` companion at
/// the same instant — `rename(2)` makes the final name appear atomically
/// only after the temp name is gone.
///
/// On extremely fast hosts the put may complete before any poll iteration
/// sees the temp file at all. That is still a passing outcome: it means
/// the rename serialization held. The negative assertion is what we care
/// about; observing the temp file before the rename is a bonus signal.
#[tokio::test]
async fn temp_file_naming_no_simultaneous_visibility() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let layers_dir = root.join(".live-edit").join("layers");

    let object_id: u64 = 17;
    let snap_seq: u64 = 42;
    let final_name = format!("{}.{}.ompf", object_id, snap_seq);
    // The atomicity invariant under test is for the OMPF write specifically.
    // `put_layer_checkpoint` also writes a `.meta.json` companion (and
    // optionally an `.ext-refs.json`), each through its own
    // write-temp-then-rename — each is independently atomic but lands AFTER
    // the OMPF, so the OMPF's final-name visibility and a meta.json temp
    // file are expected to coexist in time. Scope `tmp_prefix` to the OMPF
    // temp shape only so the assertion below pins exactly the OMPF
    // atomicity contract.
    let tmp_prefix = format!("{}.{}.ompf.", object_id, snap_seq);

    let store = LocalDiskCheckpointStore::new(root.clone()).expect("store::new");
    // Sanity — the constructor `create_dir_all`s the layers subdir so the
    // poll loop below never races on the parent directory's existence.
    assert!(
        layers_dir.is_dir(),
        "constructor must create the layers subdir, got missing: {}",
        layers_dir.display()
    );

    let payload = Bytes::from(vec![0xCD_u8; PAYLOAD_BYTES]);
    let meta = CheckpointMeta {
        creator_replica_id: "test".to_string(),
        written_at_unix_ms: 0,
        snap_seq: Some(snap_seq),
        // Identity carried for the same reason the tail fixture below carries
        // it: `validate_generation_identity` refuses an identity-empty put
        // before any path is derived, so a fixture without it would never reach
        // the temp-file publish this case is about.
        session_id: "sess-atomicity".to_string(),
        layer_name: "layer-atomicity".to_string(),
        ..CheckpointMeta::default()
    };

    // Spawn the put on a background task so we can poll the directory
    // while it is in flight.
    let put_task = tokio::spawn({
        let store_layers_dir = layers_dir.clone();
        async move {
            let res = store
                .put_layer_checkpoint(object_id, snap_seq, payload, meta)
                .await;
            // Final-state sanity: only the .ompf file exists, no .tmp
            // residue. Return this from the spawn so the assertions
            // happen on the test thread (panicking inside the spawn
            // would hang the test through the JoinHandle).
            (res, store_layers_dir)
        }
    });

    // Poll loop. We do NOT wait for the put to finish before starting to
    // poll — the whole point is to inspect the dir state mid-write.
    let deadline = Instant::now() + POLL_BUDGET;
    let mut saw_tmp = false;
    let mut saw_final = false;
    loop {
        let snapshot = scan_layer_filenames(&layers_dir, &final_name, &tmp_prefix).await;
        let SnapshotEntries {
            final_present,
            tmp_present,
        } = snapshot;

        // The load-bearing atomicity assertion: never both at the same
        // instant. If this fires, the adapter is not using a write-temp +
        // rename pattern and POSIX atomicity is gone.
        assert!(
            !(final_present && tmp_present),
            "observed both {} and a {}*.tmp companion simultaneously in {}",
            final_name,
            tmp_prefix,
            layers_dir.display()
        );

        if tmp_present {
            saw_tmp = true;
        }
        if final_present {
            saw_final = true;
            // Once the final file is published the put has either
            // returned or is about to; stop polling.
            break;
        }
        if Instant::now() >= deadline {
            // 5-second budget exhausted without observing the final
            // file. Drop out of the loop and let the JoinHandle assertion
            // below surface the actual put outcome.
            break;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }

    let (put_result, observed_layers_dir) = put_task.await.expect("put_task panicked");
    let handle = put_result.expect("put_layer_checkpoint must succeed");
    assert_eq!(handle.object_id, object_id);
    assert_eq!(handle.snap_seq, snap_seq);

    // Final-state assertion: exactly one entry matching the final name,
    // zero entries matching the .tmp prefix.
    let final_snapshot = scan_layer_filenames(&observed_layers_dir, &final_name, &tmp_prefix).await;
    assert!(
        final_snapshot.final_present,
        "expected {} to exist after put returned",
        final_name
    );
    assert!(
        !final_snapshot.tmp_present,
        "expected no leftover {}*.tmp residue after successful put",
        tmp_prefix
    );

    // `saw_final` is always true here because the loop only exits on
    // observing it or on timeout, and the JoinHandle assertion above
    // confirmed the put succeeded. `saw_tmp` is best-effort — on a fast
    // host the tmp file may have come and gone between probes, which is
    // still atomic. Touch the variable so clippy does not flag it as
    // dead.
    let _ = saw_tmp;
    let _ = saw_final;
}

/// Test 2 — mid-write SIGKILL of the harness child process.
///
/// Spawns the `local_disk_write_kill_harness` example, reads its stdout
/// until the `STARTED_WRITE` ready marker, then `kill()`s the child
/// (SIGKILL on Unix via `tokio::process::Child::kill().await`). After the
/// child exits we open a fresh `LocalDiskCheckpointStore` against the same
/// root and assert `get_latest_layer_checkpoint(object_id) == Ok(None)`
/// — proof that the half-written temp file (if any) never landed at the
/// final path.
#[tokio::test]
async fn mid_write_sigkill_leaves_no_published_file() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let layers_dir = root.join(".live-edit").join("layers");

    let object_id: u64 = 91;
    let snap_seq: u64 = 7;
    let final_name = format!("{}.{}.ompf", object_id, snap_seq);
    // Restrict the prefix to OMPF tmp files only. The harness exits after
    // the OMPF put; meta.json isn't even attempted. See test 1's comment
    // for the broader rationale.
    let tmp_prefix = format!("{}.{}.ompf.", object_id, snap_seq);

    let harness_path = locate_harness_binary();

    let mut child = Command::new(&harness_path)
        // argv[1] selects the harness mode (`layer` vs `session`). See the
        // harness's module doc-comment for the per-mode argv shape.
        .arg("layer")
        .arg(&root)
        .arg(object_id.to_string())
        .arg(snap_seq.to_string())
        .arg(PAYLOAD_BYTES.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap_or_else(|err| {
            panic!(
                "failed to spawn harness binary {}: {}",
                harness_path.display(),
                err
            )
        });

    // Read stdout line-by-line, with a bounded total budget, until we see
    // the `STARTED_WRITE` ready marker.
    let stdout = child.stdout.take().expect("harness stdout pipe");
    let mut reader = BufReader::new(stdout).lines();
    let mut saw_ready = false;
    let read_deadline = Instant::now() + POLL_BUDGET;
    while Instant::now() < read_deadline {
        let remaining = read_deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(remaining, reader.next_line()).await {
            Ok(Ok(Some(line))) => {
                if line == "STARTED_WRITE" {
                    saw_ready = true;
                    break;
                }
            }
            Ok(Ok(None)) => break, // stdout closed before ready marker.
            Ok(Err(_)) => break,   // IO error reading stdout.
            Err(_) => break,       // budget elapsed.
        }
    }
    assert!(
        saw_ready,
        "harness did not print STARTED_WRITE within the {:?} budget — \
         binary at {} may be missing or broken",
        POLL_BUDGET,
        harness_path.display()
    );

    // Kill the child immediately after we see the ready marker.
    // `tokio::process::Child::kill().await` sends SIGKILL on Unix —
    // bypasses Drop handlers, the tokio runtime shutdown path, and any
    // cleanup the harness's `atomic_write` would have done on a graceful
    // `?`-style error return.
    child.kill().await.expect("kill harness");
    let status = child.wait().await.expect("wait harness");
    assert!(
        !status.success(),
        "harness exited successfully despite SIGKILL — kill may have raced \
         the rename. status: {:?}",
        status
    );

    // Assertion 1: the published .ompf MUST NOT exist. A .tmp orphan MAY
    // exist (and is acceptable garbage operators can scrub manually per
    // the adapter's module-level doc-comment).
    let final_path = layers_dir.join(&final_name);
    assert!(
        !tokio::fs::try_exists(&final_path).await.unwrap_or(false),
        "atomicity violation: {} exists despite SIGKILL mid-write",
        final_path.display()
    );

    // Assertion 2: a fresh adapter opened on the same root reports no
    // checkpoint for the object. This is the recovery-hook's view of the
    // world; if it saw a partial write, recovery would deserialize
    // garbage and crash the replica.
    let recovered_store = LocalDiskCheckpointStore::new(root.clone()).expect("store::new");
    let recovered = recovered_store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest_layer_checkpoint");
    assert!(
        recovered.is_none(),
        "recovery-time read returned Some after SIGKILL mid-write \
         (atomicity violation): {:?}",
        recovered.map(|b| b.handle)
    );

    // `tmp_prefix` is only used to document the observable surface in the
    // failure messages above; touch it so clippy does not flag it as
    // unused.
    let _ = tmp_prefix;
}

/// Test 3 — tail temp-file naming, plus the full-size-on-visibility check.
///
/// Mirrors [`temp_file_naming_no_simultaneous_visibility`] for a
/// `<oid>.<to_seq>.tail` put and asserts two things on every probe:
///
/// 1. the final `.tail` name and an in-flight `<oid>.<to_seq>.tail.*.tmp`
///    companion are never observed in the same iteration — the `rename(2)`
///    serialization; and
/// 2. whenever the final `.tail` name IS observed, it already holds the whole
///    payload. This is the assertion that distinguishes *rename-published*
///    from *streamed-in-place*: a `tokio::fs::write` straight to the final
///    path would make the name visible at 0 bytes and grow it under the
///    reader, so this fires the moment a tail write bypasses `atomic_write`.
///
/// As with test 1, a put that completes before the first poll iteration is
/// still a pass — the negative assertion is what carries the contract, and
/// catching the `.tmp` companion in the act is a bonus signal, not a
/// requirement. Both assertions are also re-checked once on the settled
/// post-put state so the test is never vacuous.
#[tokio::test]
async fn tail_temp_file_naming_no_simultaneous_visibility() {
    let object_id: u64 = 23;
    let base_snap_seq: u64 = 40;
    let to_seq: u64 = 91;
    let artifact_name = tail_artifact_name(object_id, to_seq);

    let outcome = concurrent_tail_put_probed(object_id, base_snap_seq, to_seq, |snapshot| {
        assert!(
            !(snapshot.artifact_present && snapshot.artifact_tmp_present),
            "observed both {} and a {}.*.tmp companion simultaneously — the \
             tail envelope is not being published by rename(2)",
            artifact_name,
            artifact_name
        );
        if let Some(len) = snapshot.artifact_len {
            assert_eq!(
                len, PAYLOAD_BYTES as u64,
                "observed {} at {} bytes, expected the full {} — a partially \
                 sized final name means the tail bytes are being streamed to \
                 the published path instead of written to a temp and renamed",
                artifact_name, len, PAYLOAD_BYTES
            );
        }
    })
    .await;

    assert_eq!(outcome.handle.object_id, object_id);
    assert_eq!(outcome.handle.snap_seq, to_seq);
    assert_eq!(outcome.handle.kind, CheckpointKind::Tail);

    // Settled-state assertions: the envelope is present at full size and no
    // `.tmp` residue survived the put. These re-run both in-flight
    // assertions against a state the poll loop is guaranteed to have
    // reached, so neither can pass vacuously on a host where the put
    // outran every probe.
    let settled = scan_tail_filenames(&outcome.layers_dir, object_id, to_seq).await;
    assert!(
        settled.artifact_present,
        "expected {} to exist after the tail put returned",
        artifact_name
    );
    assert!(
        !settled.artifact_tmp_present,
        "expected no leftover {}.*.tmp residue after a successful tail put",
        artifact_name
    );
    assert_eq!(
        settled.artifact_len,
        Some(PAYLOAD_BYTES as u64),
        "settled {} must hold the whole payload",
        artifact_name
    );
}

/// Test 4 — tail publish **order**: a listed artifact always has its meta.
///
/// `atomic_write` makes each individual file appear atomically; it says
/// nothing about the order two files appear in. LocalDisk's tail branch
/// publishes the meta companion FIRST and the envelope second, so that
/// `list_layer_checkpoints` — which is driven off the `.tail` stem — can
/// never enumerate a tail whose meta is missing.
///
/// The observable form of that obligation: every poll iteration in which the
/// final `.tail` name is present must also see
/// `<oid>.<to_seq>.tail.meta.json`. The reverse is deliberately NOT asserted
/// — under meta-first the meta legitimately exists on its own for the whole
/// length of the envelope write, which for an 8 MiB payload is most of the
/// put.
#[tokio::test]
async fn tail_artifact_never_visible_without_its_meta() {
    let object_id: u64 = 24;
    let base_snap_seq: u64 = 12;
    let to_seq: u64 = 77;
    let artifact_name = tail_artifact_name(object_id, to_seq);
    let meta_name = tail_meta_name(object_id, to_seq);

    let mut saw_meta_alone = false;
    let outcome = concurrent_tail_put_probed(object_id, base_snap_seq, to_seq, |snapshot| {
        // The load-bearing ordering assertion: a visible envelope always has
        // its meta. Written as a guarded assert rather than one boolean so
        // the direction being pinned reads off the page — the converse
        // (meta alone) is legal and deliberately unasserted.
        if snapshot.artifact_present {
            assert!(
                snapshot.meta_present,
                "observed {} without its {} companion — the tail pair is not \
                 publishing meta-first, so a concurrent lister can enumerate \
                 a tail whose meta is absent",
                artifact_name, meta_name
            );
        } else if snapshot.meta_present {
            // The expected publish window. Recorded as a positive signal
            // that the probe cadence really is sampling mid-put; not
            // asserted, because a fast host may skip straight past it.
            saw_meta_alone = true;
        }
    })
    .await;

    assert_eq!(outcome.handle.snap_seq, to_seq);
    assert_eq!(outcome.handle.kind, CheckpointKind::Tail);

    // Settled state: both halves of the pair landed, so the ordering
    // assertion above was checked against a directory that really does
    // hold the artifact rather than against an empty one.
    let settled = scan_tail_filenames(&outcome.layers_dir, object_id, to_seq).await;
    assert!(
        settled.artifact_present && settled.meta_present,
        "expected both {} and {} after the tail put returned (artifact: {}, meta: {})",
        artifact_name,
        meta_name,
        settled.artifact_present,
        settled.meta_present
    );

    // Best-effort observation only — see the comment at the assignment.
    let _ = saw_meta_alone;
}

/// Test 5 — the same ordering invariant on the way *out*.
///
/// `put_layer_checkpoint`'s tail branch publishes meta → envelope;
/// `delete_layer_checkpoint`'s tail branch removes envelope → meta. The
/// adapter states the consequence as one claim spanning both: "envelope
/// visible ⇒ meta present holds at every instant, **in both directions of
/// the artifact's lifetime**". Test 4 pins the publish direction; without
/// this test the delete direction rests on the delete loop's array literal
/// happening to list the two paths in the right order — a swap that no
/// steady-state test can see, because both files are gone by the time the
/// call returns.
///
/// So: poll while a `delete_layer_checkpoint` is in flight and assert the
/// same implication. The legal intermediate state is meta-alone, exactly as
/// during publish; envelope-alone is the violation in both directions.
#[tokio::test]
async fn tail_delete_never_leaves_the_artifact_without_its_meta() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let layers_dir = root.join(".live-edit").join("layers");

    let object_id: u64 = 25;
    let base_snap_seq: u64 = 3;
    let to_seq: u64 = 58;
    let artifact_name = tail_artifact_name(object_id, to_seq);
    let meta_name = tail_meta_name(object_id, to_seq);

    let store = Arc::new(LocalDiskCheckpointStore::new(root).expect("store::new"));

    // Arrange: a settled tail pair on disk. Awaited to completion — the
    // publish direction is test 4's subject, not this one's.
    let payload = Bytes::from(vec![0xAB_u8; PAYLOAD_BYTES]);
    let handle = store
        .put_layer_checkpoint(
            object_id,
            to_seq,
            payload,
            tail_meta(base_snap_seq, base_snap_seq, to_seq),
        )
        .await
        .expect("tail put_layer_checkpoint must succeed");
    assert_eq!(handle.kind, CheckpointKind::Tail);
    let before = scan_tail_filenames(&layers_dir, object_id, to_seq).await;
    assert!(
        before.artifact_present && before.meta_present,
        "arrange step must leave both {} and {} on disk",
        artifact_name,
        meta_name
    );

    // Act: delete on a background task so the loop below can observe the
    // two unlinks separately.
    let delete_task = tokio::spawn({
        let store = Arc::clone(&store);
        async move { store.delete_layer_checkpoint(handle).await }
    });

    let deadline = Instant::now() + POLL_BUDGET;
    loop {
        let snapshot = scan_tail_filenames(&layers_dir, object_id, to_seq).await;

        if snapshot.artifact_present {
            assert!(
                snapshot.meta_present,
                "observed {} without its {} companion mid-delete — the tail \
                 delete is removing the meta first, so a concurrent lister \
                 can enumerate a tail whose meta is already gone",
                artifact_name, meta_name
            );
        }

        if !snapshot.artifact_present && !snapshot.meta_present {
            // Both unlinked; nothing further to observe.
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        // Same cadence rationale as the put loop — see
        // [`concurrent_tail_put_probed`].
        tokio::task::yield_now().await;
    }

    delete_task
        .await
        .expect("delete_task panicked")
        .expect("delete_layer_checkpoint must succeed");

    let settled = scan_tail_filenames(&layers_dir, object_id, to_seq).await;
    assert!(
        !settled.artifact_present && !settled.meta_present && !settled.artifact_tmp_present,
        "delete must leave nothing of the tail behind (artifact: {}, meta: {}, tmp: {})",
        settled.artifact_present,
        settled.meta_present,
        settled.artifact_tmp_present
    );
}

/// The concurrent-tail-put probe harness shared by tests 3 and 4.
///
/// Spawns `put_layer_checkpoint` for a lineage-complete tail onto a
/// background task and polls the layers directory while it is in flight,
/// handing every probe to `probe` so each test contributes only its own
/// assertion. Polling is bounded by [`POLL_BUDGET`], per the test-writing
/// guide's `## Synchronization Patterns`; the loop exits as soon as the
/// envelope is published, so the budget is a ceiling and not a cost the
/// tests normally pay.
///
/// # Why this loop paces on `yield_now`, not [`POLL_INTERVAL`]
///
/// Measured, not assumed. At the 5 ms [`POLL_INTERVAL`] the base tests use,
/// this loop sampled exactly two states on a tmpfs-backed `tempdir` —
/// everything-absent, then everything-published: the whole meta-write →
/// meta-rename → 8 MiB envelope-write → envelope-rename sequence fits inside
/// a single interval, so not one in-flight state was ever observed and both
/// tail assertions degenerated to post-hoc checks of the settled directory.
/// A rename-published envelope and one streamed in place look identical
/// once the put has returned, which is precisely the regression these tests
/// exist to catch.
///
/// Yield-paced probing (each iteration still awaits a real `read_dir`, so
/// the put's blocking-pool work is never starved) samples the full sequence
/// instead: absent → meta alone → meta + envelope `.tmp` → published
/// envelope with no `.tmp`. No new timing constant is introduced and the
/// budget bound is unchanged; only the cadence between probes differs, and
/// only for the tail loop.
///
/// Panics from `probe` propagate on the test's own task (the poll loop runs
/// inline here, not inside the spawn), so a failing assertion surfaces as a
/// normal test failure rather than a hang on the `JoinHandle`.
async fn concurrent_tail_put_probed<F>(
    object_id: u64,
    base_snap_seq: u64,
    to_seq: u64,
    mut probe: F,
) -> TailPutOutcome
where
    F: FnMut(&TailSnapshotEntries),
{
    let dir = tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let layers_dir = root.join(".live-edit").join("layers");

    let store = LocalDiskCheckpointStore::new(root).expect("store::new");
    // Sanity — the constructor `create_dir_all`s the layers subdir so the
    // poll loop below never races on the parent directory's existence.
    assert!(
        layers_dir.is_dir(),
        "constructor must create the layers subdir, got missing: {}",
        layers_dir.display()
    );

    let payload = Bytes::from(vec![0xAB_u8; PAYLOAD_BYTES]);
    let meta = tail_meta(base_snap_seq, base_snap_seq, to_seq);

    // Spawn the put so the loop below can inspect the directory mid-write.
    let put_task = tokio::spawn(async move {
        store
            .put_layer_checkpoint(object_id, to_seq, payload, meta)
            .await
    });

    let deadline = Instant::now() + POLL_BUDGET;
    loop {
        let snapshot = scan_tail_filenames(&layers_dir, object_id, to_seq).await;
        probe(&snapshot);

        if snapshot.artifact_present {
            // Published; the put has either returned or is about to.
            break;
        }
        if Instant::now() >= deadline {
            // Budget exhausted without seeing the envelope. Drop out and let
            // the `JoinHandle` expectation below surface the put's real
            // outcome, which is the more informative failure.
            break;
        }
        // Cadence rationale in this function's doc-comment — a 5 ms sleep
        // here skips the entire publish window on a fast filesystem.
        tokio::task::yield_now().await;
    }

    let handle = put_task
        .await
        .expect("put_task panicked")
        .expect("tail put_layer_checkpoint must succeed");

    TailPutOutcome {
        handle,
        layers_dir,
        _root: dir,
    }
}

/// What [`concurrent_tail_put_probed`] hands back for the settled-state half
/// of each test.
struct TailPutOutcome {
    handle: CheckpointHandle,
    layers_dir: PathBuf,
    /// Keeps the temp root alive for the caller's post-put scans — dropping
    /// the `TempDir` deletes the tree.
    _root: TempDir,
}

/// `<oid>.<to_seq>.tail` — the published tail envelope's filename.
fn tail_artifact_name(object_id: u64, to_seq: u64) -> String {
    format!("{}.{}.tail", object_id, to_seq)
}

/// `<oid>.<to_seq>.tail.meta.json` — the tail's meta companion. Kind-qualified
/// so it cannot collide with a base's `<oid>.<snap_seq>.meta.json` at the same
/// sequence, which is a live case once a re-base lands `base@S` alongside
/// `tail@to_seq == S`.
fn tail_meta_name(object_id: u64, to_seq: u64) -> String {
    format!("{}.{}.tail.meta.json", object_id, to_seq)
}

/// A lineage-complete tail `CheckpointMeta`. `base_snap_seq` and `from_seq`
/// are both mandatory on a tail put (the shared `validate_tail_put` gate
/// rejects a `None` in either as `Permanent`), and `sidecar_bytes` must stay
/// `None` — a tail carries no ext-refs sidecar.
fn tail_meta(base_snap_seq: u64, from_seq: u64, to_seq: u64) -> CheckpointMeta {
    CheckpointMeta {
        creator_replica_id: "test".to_string(),
        written_at_unix_ms: 0,
        snap_seq: Some(to_seq),
        session_id: "sess-atomicity".to_string(),
        layer_name: "layer-atomicity".to_string(),
        kind: CheckpointKind::Tail,
        base_snap_seq: Some(base_snap_seq),
        from_seq: Some(from_seq),
        ..CheckpointMeta::default()
    }
}

/// Result of one directory-scan probe of a tail generation's file family.
struct TailSnapshotEntries {
    /// `<oid>.<to_seq>.tail` — the published envelope.
    artifact_present: bool,
    /// Byte length of the published envelope, read in the same probe that
    /// observed it. `None` when the envelope was absent (or, vanishingly,
    /// when the stat failed — treated as "no size evidence this round"
    /// rather than as a violation).
    artifact_len: Option<u64>,
    /// An in-flight `<oid>.<to_seq>.tail.<uuid>.tmp` envelope write.
    artifact_tmp_present: bool,
    /// `<oid>.<to_seq>.tail.meta.json` — the published meta companion.
    meta_present: bool,
}

/// Scan the layers directory once and classify every entry belonging to the
/// tail generation `(object_id, to_seq)`.
///
/// The classification has to be exact, because the meta companion's own temp
/// file (`<oid>.<to_seq>.tail.meta.json.<uuid>.tmp`) shares the envelope temp
/// file's `<oid>.<to_seq>.tail.` prefix and its `.tmp` suffix. Counting it as
/// an envelope temp would make test 3's simultaneity assertion fire on the
/// perfectly legal state "envelope published, meta temp still being cleaned
/// up". The discriminator: the envelope's temp has exactly one dot-free
/// segment (the UUID) between the prefix and `.tmp`, whereas the meta's temp
/// carries `meta.json.<uuid>` there.
///
/// A missing directory is reported as all-absent — the constructor can race
/// the first probe in rare scheduling orders, and polling again beats failing.
async fn scan_tail_filenames(
    layers_dir: &Path,
    object_id: u64,
    to_seq: u64,
) -> TailSnapshotEntries {
    let artifact_name = tail_artifact_name(object_id, to_seq);
    let meta_name = tail_meta_name(object_id, to_seq);
    let artifact_tmp_prefix = format!("{}.", artifact_name);

    let mut snapshot = TailSnapshotEntries {
        artifact_present: false,
        artifact_len: None,
        artifact_tmp_present: false,
        meta_present: false,
    };

    let mut reader = match tokio::fs::read_dir(layers_dir).await {
        Ok(r) => r,
        Err(_) => return snapshot,
    };
    while let Ok(Some(entry)) = reader.next_entry().await {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if name_str == artifact_name {
            snapshot.artifact_present = true;
            snapshot.artifact_len = entry.metadata().await.ok().map(|m| m.len());
        } else if name_str == meta_name {
            snapshot.meta_present = true;
        } else if let Some(suffix) = name_str.strip_prefix(&artifact_tmp_prefix) {
            if let Some(middle) = suffix.strip_suffix(".tmp") {
                // Envelope temp iff the segment between the stem and `.tmp`
                // is the bare UUID; anything with a dot in it belongs to a
                // companion (today: the meta's own temp).
                if !middle.contains('.') {
                    snapshot.artifact_tmp_present = true;
                }
            }
        }
    }
    snapshot
}

/// Result of one directory-scan probe.
struct SnapshotEntries {
    final_present: bool,
    tmp_present: bool,
}

/// Scan the layers directory once and report whether the final `.ompf` and
/// any `<prefix>*.tmp` companion are present. Missing-directory is treated
/// as `(false, false)` — the constructor races with the first probe in
/// rare CI scheduling orders, and we'd rather poll again than fail.
async fn scan_layer_filenames(
    layers_dir: &Path,
    final_name: &str,
    tmp_prefix: &str,
) -> SnapshotEntries {
    let mut final_present = false;
    let mut tmp_present = false;
    let mut reader = match tokio::fs::read_dir(layers_dir).await {
        Ok(r) => r,
        Err(_) => {
            return SnapshotEntries {
                final_present,
                tmp_present,
            }
        }
    };
    while let Ok(Some(entry)) = reader.next_entry().await {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if name_str == final_name {
            final_present = true;
        } else if name_str.starts_with(tmp_prefix) && name_str.ends_with(".tmp") {
            tmp_present = true;
        }
    }
    SnapshotEntries {
        final_present,
        tmp_present,
    }
}

/// Locate the `local_disk_write_kill_harness` example binary.
///
/// `CARGO_BIN_EXE_<name>` is not set for examples (only for `[[bin]]`
/// targets), so we derive the conventional path from the test binary's
/// own location instead. Cargo lays out integration-test binaries under
/// `<target>/<profile>/deps/<test-name>-<hash>` and example binaries
/// under `<target>/<profile>/examples/<name>`. Walking two parents up
/// from `env::current_exe()` gives `<target>/<profile>/`, where the
/// `examples/` directory lives.
///
/// This avoids hardcoding `target/` (`CARGO_TARGET_DIR` may relocate it)
/// and the `debug` vs `release` profile (so `cargo test --release` works
/// too). The Subtask's acceptance criteria includes running
/// `cargo build --example local_disk_write_kill_harness` before
/// `cargo test`, which guarantees the binary exists at this path.
fn locate_harness_binary() -> PathBuf {
    let current_exe = std::env::current_exe().expect("current_exe");
    let profile_dir = current_exe
        .parent() // .../deps/
        .and_then(|p| p.parent()) // .../<profile>/
        .expect("test binary path must have a profile ancestor");
    let bin_name = if cfg!(windows) {
        "local_disk_write_kill_harness.exe"
    } else {
        "local_disk_write_kill_harness"
    };
    let path = profile_dir.join("examples").join(bin_name);
    assert!(
        path.exists(),
        "harness example binary not found at {} — \
         run `cargo build --example local_disk_write_kill_harness` first",
        path.display()
    );
    path
}
