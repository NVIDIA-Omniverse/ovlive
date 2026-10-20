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

//! Test harness — child-process driver for the LocalDisk write-atomicity
//! mid-write-kill integration tests under
//! `tests/integration/checkpoint/atomicity_local_disk_test.rs` and
//! `tests/integration/checkpoint/atomicity_session_local_disk_test.rs`.
//!
//! The harness is spawned by the test, prints `STARTED_WRITE\n` and flushes
//! stdout, then begins a write call against a `LocalDiskCheckpointStore`
//! rooted at the directory passed in argv. The parent test reads stdout
//! until it sees `STARTED_WRITE`, then sends SIGKILL via
//! `tokio::process::Child::kill().await`. The atomicity contract under
//! test: after SIGKILL the final published file MUST NOT exist (only an
//! unrenamed temp file may linger).
//!
//! NOT a production example — this binary exists solely to be invoked by
//! the integration tests under
//! `CARGO_BIN_EXAMPLE_local_disk_write_kill_harness`.
//!
//! ## Modes
//!
//! The harness dispatches on argv[1] (`mode`). One mode per checkpoint
//! kind that exercises the write-temp + rename path on the LocalDisk
//! adapter:
//!
//! - `layer` — calls `put_layer_checkpoint`, writing
//!   `<root>/.live-edit/layers/<oid>.<snap>.ompf` (plus a `.meta.json`
//!   companion the SIGKILL test does not observe because the kill arrives
//!   before the OMPF write returns). Argv shape:
//!   `(layer, root, object_id, snap_seq, payload_bytes)`.
//!
//! - `session` — calls `put_session_checkpoint`, writing
//!   `<root>/.live-edit/sessions/<session_id>.json`. The body is opaque
//!   bytes per SR-1.5; the harness builds a `Vec<u8>` of the requested
//!   size filled with `0xCD`. Argv shape:
//!   `(session, root, session_id, payload_bytes)`.
//!
//! In both modes the harness allocates the payload BEFORE printing
//! `STARTED_WRITE` so the wall-clock window between the ready marker and
//! the first `tokio::fs::write` syscall is dominated by the actual write,
//! not bookkeeping.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use bytes::Bytes;
use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{CheckpointMeta, CheckpointStore};

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mode, rest) = match argv.split_first() {
        Some((m, r)) => (m.as_str(), r),
        None => {
            eprintln!(
                "usage: local_disk_write_kill_harness <mode> ...\n  \
                 layer   <root> <object_id> <snap_seq> <payload_bytes>\n  \
                 session <root> <session_id> <payload_bytes>"
            );
            return ExitCode::from(2);
        }
    };
    match mode {
        "layer" => run_layer(rest).await,
        "session" => run_session(rest).await,
        other => {
            eprintln!("unknown mode: {} (expected 'layer' or 'session')", other);
            ExitCode::from(2)
        }
    }
}

async fn run_layer(args: &[String]) -> ExitCode {
    let [root, object_id, snap_seq, payload_bytes] = match args {
        [a, b, c, d] => [a, b, c, d],
        _ => {
            eprintln!(
                "layer mode: expected <root> <object_id> <snap_seq> <payload_bytes>, got {} args",
                args.len()
            );
            return ExitCode::from(2);
        }
    };
    let root = PathBuf::from(root);
    let object_id: u64 = object_id
        .parse()
        .expect("layer mode: object_id must be a u64");
    let snap_seq: u64 = snap_seq
        .parse()
        .expect("layer mode: snap_seq must be a u64");
    let payload_bytes: usize = payload_bytes
        .parse()
        .expect("layer mode: payload_bytes must be a usize");

    let store = LocalDiskCheckpointStore::new(root).expect("LocalDiskCheckpointStore::new failed");

    // Allocate the payload before announcing readiness so the time between
    // STARTED_WRITE and the first `tokio::fs::write` syscall is dominated
    // by the actual write, not bookkeeping.
    let payload = Bytes::from(vec![0xAB_u8; payload_bytes]);

    print_started_write();

    let meta = CheckpointMeta {
        creator_replica_id: "harness".to_string(),
        written_at_unix_ms: 0,
        snap_seq: Some(snap_seq),
        ..CheckpointMeta::default()
    };

    // If we ever reach the end of this call, the parent kill arrived late
    // — the test will catch that by asserting the final file is absent.
    // We do not unwrap the result because the kill may race the await
    // point; reaching this line at all is the test's "kill arrived late"
    // signal.
    let _ = store
        .put_layer_checkpoint(object_id, snap_seq, payload, meta)
        .await;
    ExitCode::SUCCESS
}

async fn run_session(args: &[String]) -> ExitCode {
    let [root, session_id, payload_bytes] = match args {
        [a, b, c] => [a, b, c],
        _ => {
            eprintln!(
                "session mode: expected <root> <session_id> <payload_bytes>, got {} args",
                args.len()
            );
            return ExitCode::from(2);
        }
    };
    let root = PathBuf::from(root);
    let session_id = session_id.clone();
    let payload_bytes: usize = payload_bytes
        .parse()
        .expect("session mode: payload_bytes must be a usize");

    let store = LocalDiskCheckpointStore::new(root).expect("LocalDiskCheckpointStore::new failed");

    // Allocate before announcing readiness for the same reason as the
    // layer path — keep the STARTED_WRITE → first `tokio::fs::write`
    // window dominated by the syscall, not allocation.
    let payload = Bytes::from(vec![0xCD_u8; payload_bytes]);

    print_started_write();

    let meta = CheckpointMeta {
        creator_replica_id: "harness".to_string(),
        written_at_unix_ms: 0,
        ..CheckpointMeta::default()
    };

    // Same "kill arrived late" signal as the layer path: if the call
    // returns, the test's assertion that the published file is absent
    // will catch any atomicity violation. The adapter currently discards
    // `meta` for session-checkpoint puts (provenance lives inside the
    // JSON body per SR-2.7).
    let _ = store
        .put_session_checkpoint(&session_id, payload, meta)
        .await;
    ExitCode::SUCCESS
}

/// Print the ready marker and flush — the parent test reads stdout
/// line-by-line until this exact line appears, then kills us.
fn print_started_write() {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let _ = handle.write_all(b"STARTED_WRITE\n");
    let _ = handle.flush();
}
