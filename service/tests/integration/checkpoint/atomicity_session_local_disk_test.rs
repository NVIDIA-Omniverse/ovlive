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

//! Write-atomicity integration test for `LocalDiskCheckpointStore`'s
//! `put_session_checkpoint`.
//!
//! Subtask `t3.fgj.7v.cq.tr` under Task `t2.fgj.7v.cq` (Session and
//! layer-mapping DR). Sibling to
//! [`atomicity_local_disk_test`](super::atomicity_local_disk_test) — the
//! layer-checkpoint companion under Task `t2.fgj.ya.r7`.
//!
//! The atomicity invariant for `put_session_checkpoint` is the same
//! write-temp + rename discipline the layer path uses, so the proof shape
//! mirrors that test's `mid_write_sigkill_leaves_no_published_file`:
//! spawn the `local_disk_write_kill_harness` example binary in `session`
//! mode, wait for the `STARTED_WRITE` ready marker, then `SIGKILL` the
//! child mid-write. SIGKILL bypasses Drop handlers and tokio runtime
//! shutdown, so any temp file the harness was streaming is left
//! half-written. A fresh `LocalDiskCheckpointStore` opened on the same
//! root MUST return `Ok(None)` from `get_session_checkpoint` because the
//! `rename(2)` step never completed.
//!
//! Coverage of the temp-file naming pattern (lower-fidelity, no child
//! process) lives inline in `src/checkpoint/adapters/local_disk.rs`'s
//! `#[cfg(test)]` module — this integration file holds only the
//! high-fidelity SIGKILL proof, paralleling the contribution scope of
//! the layer-side atomicity Subtask `t3.fgj.ya.r7.xm`.
//!
//! Bounded polling per `docs-internal/test-writing-guide.md` `## Synchronization
//! Patterns` — total budget is 5 seconds, never an unconditional
//! `tokio::time::sleep`.
//!
//! [`local_disk_write_kill_harness`]: ../../../examples/local_disk_write_kill_harness.rs

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::CheckpointStore;
use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// 5-second cap on the stdout-read loop, per the test-writing guide's
/// `## Synchronization Patterns` rule. Matches the layer-atomicity test.
const POLL_BUDGET: Duration = Duration::from_secs(5);

/// Payload size for the SIGKILL test. Large enough that `tokio::fs::write`
/// takes a non-trivial slice of wall time so the killer has a reasonable
/// window in which to act, while staying small enough that the test does
/// not stress CI disk quotas. 8 MiB matches the established constant in
/// the layer-side atomicity test.
const PAYLOAD_BYTES: usize = 8 * 1024 * 1024;

/// Mid-write SIGKILL of the harness child process — session-checkpoint
/// variant.
///
/// Spawns the `local_disk_write_kill_harness` example in `session` mode,
/// reads its stdout until the `STARTED_WRITE` ready marker, then `kill()`s
/// the child (SIGKILL on Unix via `tokio::process::Child::kill().await`).
/// After the child exits we open a fresh `LocalDiskCheckpointStore`
/// against the same root and assert:
///
/// 1. `<root>/.live-edit/sessions/<session_id>.json` MUST NOT exist —
///    proof that the half-written temp file (if any) never landed at the
///    final path via `rename(2)`.
///
/// 2. `get_session_checkpoint(session_id)` on a fresh adapter returns
///    `Ok(None)` — the recovery-hook's view of the world must not see a
///    torn JSON file, which is the load-bearing invariant for crash
///    recovery downstream of Tasks `t2.fgj.7v.p9` (write path) and
///    `t2.fgj.7v.fp` (recovery hook).
///
/// A `<session_id>.json.*.tmp` orphan MAY exist (acceptable garbage
/// operators scrub manually per the adapter's module-level doc-comment);
/// the test does not assert on its presence either way.
#[tokio::test]
async fn mid_write_sigkill_leaves_no_published_session_file() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let sessions_dir = root.join(".live-edit").join("sessions");

    // Plain UTF-8 session id, no path separators or shell metacharacters —
    // the adapter joins this directly into the on-disk path so it must be
    // a safe filename component.
    let session_id = "sess-atomicity-1";
    let final_name = format!("{}.json", session_id);

    let harness_path = locate_harness_binary();

    let mut child = Command::new(&harness_path)
        // argv[1] selects the harness mode (`layer` vs `session`). See the
        // harness's module doc-comment for the per-mode argv shape.
        .arg("session")
        .arg(&root)
        .arg(session_id)
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

    // Read stdout line-by-line, bounded by a 5-second total budget, until
    // we see the `STARTED_WRITE` ready marker.
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

    // Kill the child immediately after the ready marker.
    // `tokio::process::Child::kill().await` sends SIGKILL on Unix —
    // bypasses Drop handlers, tokio runtime shutdown, and any cleanup the
    // adapter's `atomic_write` would have done on a graceful `?`-style
    // error return.
    child.kill().await.expect("kill harness");
    let status = child.wait().await.expect("wait harness");
    assert!(
        !status.success(),
        "harness exited successfully despite SIGKILL — kill may have raced \
         the rename. status: {:?}",
        status
    );

    // Assertion 1: the published .json MUST NOT exist. A .tmp orphan MAY
    // exist (and is acceptable garbage operators can scrub manually per
    // the adapter's module-level doc-comment).
    let final_path = sessions_dir.join(&final_name);
    assert!(
        !tokio::fs::try_exists(&final_path).await.unwrap_or(false),
        "atomicity violation: {} exists despite SIGKILL mid-write",
        final_path.display()
    );

    // Assertion 2: a fresh adapter opened on the same root reports no
    // checkpoint for the session. This is the recovery-hook's view of
    // the world; if it saw a partial write, recovery would deserialize
    // a torn JSON body and either crash the replica or, worse, restore
    // stale state.
    let recovered_store = LocalDiskCheckpointStore::new(root.clone()).expect("store::new");
    let recovered = recovered_store
        .get_session_checkpoint(session_id)
        .await
        .expect("get_session_checkpoint");
    assert!(
        recovered.is_none(),
        "recovery-time read returned Some after SIGKILL mid-write \
         (atomicity violation): handle = {:?}",
        recovered.as_ref().map(|b| &b.handle)
    );
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
/// Mirrors the layer-side `locate_harness_binary` in
/// `atomicity_local_disk_test.rs`; the two tests share the same harness
/// binary, distinguished only by argv[1].
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
