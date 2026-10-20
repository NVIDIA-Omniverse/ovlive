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

//! Task 3.4.D — compile-time and runtime thread-safety guarantees for the
//! public `oodf-ffi` surface.
//!
//! Compile-time: pins `Send + Sync` on the two public types Epic 4 will move
//! across async task boundaries. If either regresses, this file will fail to
//! compile (not just fail at runtime).
//!
//! Runtime: N threads independently build + parse their own deltas in a tight
//! loop, with no shared mutable state. Asserts the cxx bridge and the C++
//! `parse_delta` / `build_delta` free functions can be invoked concurrently
//! without deadlock, data race, or panic. Uses `std::thread` only — no
//! tokio, no sleeps.

use oodf_ffi::{
    build_delta, parse_delta, CommandCreateSection, OodfError, OwnedDelta, SetField, StructCommand,
};
use std::sync::{Arc, Barrier};
use std::thread;

// Compile-time Send + Sync assertions. Evaluated at compile time — no runtime
// cost. If either type loses Send or Sync, cargo build fails here.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OwnedDelta>();
    assert_send_sync::<OodfError>();
};

#[test]
fn parallel_round_trip_across_threads() {
    const THREAD_COUNT: u64 = 8;
    const ITERATIONS_PER_THREAD: usize = 50;

    // Barrier so every worker thread is woken at the same instant — without
    // it, `thread::spawn` ramp-up time on cold runs (~200 µs/thread on a
    // loaded macOS CI box) can let the first thread finish its entire
    // 50-iteration loop before the last thread is scheduled, defeating the
    // "concurrent invocation" intent of this test (b.ohv item 7). After the
    // barrier all threads enter their per-thread loops within a few
    // microseconds of each other, maximising the overlap the C++ FFI must
    // tolerate.
    let barrier = Arc::new(Barrier::new(THREAD_COUNT as usize));

    let handles: Vec<_> = (0..THREAD_COUNT)
        .map(|tid| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                // Per-thread input — no sharing. Each thread has a distinct
                // section_id so failures are localizable via the error messages.
                let delta = OwnedDelta {
                    is_diff: tid % 2 == 0,
                    base_version: tid,
                    struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                        parent_id: tid,
                        section_id: 1000 + tid,
                        section_name: format!("thread-{tid}"),
                        section_type: tid as u8,
                    })],
                    set_fields: vec![],
                    time_samples: vec![],
                    source_format: format!("fmt-{tid}"),
                };

                // Lockstep entry — every thread blocks here until all N
                // are at the barrier, then they all proceed simultaneously.
                barrier.wait();

                for _ in 0..ITERATIONS_PER_THREAD {
                    let bytes = build_delta(&delta).expect("build_delta cannot fail");
                    let parsed = parse_delta(&bytes)
                        .expect("concurrent round-trip must succeed for every iteration");
                    assert_eq!(
                        parsed, delta,
                        "thread {tid}: parsed delta must equal original"
                    );
                }
            })
        })
        .collect();

    for (tid, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .unwrap_or_else(|e| panic!("thread {tid} panicked: {e:?}"));
    }
}

// ── b.8uk: hardening against hidden global C++ state ─────────────────────
//
// The tests above give each thread distinct inputs. If the C++ parse/build
// implementation kept any global or static mutable state (e.g. a shared
// FlatBufferBuilder, a process-wide error buffer), that state would only
// surface under *shared-input* concurrency. The tests below supply the
// same byte buffer to N parsers and the same `Arc<OwnedDelta>` to N builders,
// so any hidden shared-state data race would produce either (a) a parse
// mismatch, (b) a panic, or (c) under tsan, a reported race.

#[test]
fn shared_buffer_parsed_in_parallel_produces_identical_deltas() {
    const THREAD_COUNT: usize = 8;
    const ITERATIONS_PER_THREAD: usize = 32;

    // Build once on the main thread, then share `Arc<Vec<u8>>` across threads.
    let source = OwnedDelta {
        is_diff: true,
        base_version: 123,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id: 2,
            section_name: "shared-buffer-test".to_string(),
            section_type: 3,
        })],
        set_fields: vec![SetField {
            section_id: 2,
            key_name: "shared-key".to_string(),
            value_or_ext_hash: (0..64u8).collect(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        time_samples: vec![],
        source_format: "shared-fmt".to_string(),
    };
    let bytes = Arc::new(build_delta(&source).expect("build_delta cannot fail"));
    let expected = Arc::new(source);

    // Barrier-driven lockstep entry (b.ohv item 7). Without it, the first
    // few threads can complete their entire 32-iteration loop before later
    // threads start — defeating the shared-input contention coverage.
    let barrier = Arc::new(Barrier::new(THREAD_COUNT));

    let handles: Vec<_> = (0..THREAD_COUNT)
        .map(|tid| {
            let bytes = Arc::clone(&bytes);
            let expected = Arc::clone(&expected);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..ITERATIONS_PER_THREAD {
                    let parsed = parse_delta(bytes.as_slice())
                        .expect("shared-buffer parse must succeed for every iteration");
                    assert_eq!(
                        parsed, *expected,
                        "thread {tid}: shared-buffer parse must equal expected"
                    );
                }
            })
        })
        .collect();

    for (tid, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .unwrap_or_else(|e| panic!("shared-buffer thread {tid} panicked: {e:?}"));
    }
}

#[test]
fn shared_owned_delta_built_in_parallel_produces_identical_bytes() {
    // A hidden shared FlatBufferBuilder or global scratch pad on the C++
    // side would surface as non-byte-identical outputs across threads when
    // the input is pinned. Build the same delta N times concurrently and
    // assert every thread's output equals the single-threaded reference.
    const THREAD_COUNT: usize = 8;
    const ITERATIONS_PER_THREAD: usize = 32;

    let source = Arc::new(OwnedDelta {
        is_diff: false,
        base_version: 7,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 10,
            section_id: 20,
            section_name: "shared-build".to_string(),
            section_type: 1,
        })],
        set_fields: vec![],
        time_samples: vec![],
        source_format: "usdc".to_string(),
    });
    let reference = Arc::new(build_delta(&source).expect("build_delta cannot fail"));

    // Lockstep entry barrier (b.ohv item 7) — without it, the test's spawn
    // ramp-up can let early threads complete before later threads ever
    // reach `build_delta`, defeating the shared-input contention coverage.
    let barrier = Arc::new(Barrier::new(THREAD_COUNT));

    let handles: Vec<_> = (0..THREAD_COUNT)
        .map(|tid| {
            let source = Arc::clone(&source);
            let reference = Arc::clone(&reference);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..ITERATIONS_PER_THREAD {
                    let built = build_delta(&source).expect("build_delta cannot fail");
                    assert_eq!(
                        built, *reference,
                        "thread {tid}: per-iteration build must equal the single-threaded reference"
                    );
                }
            })
        })
        .collect();

    for (tid, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .unwrap_or_else(|e| panic!("shared-build thread {tid} panicked: {e:?}"));
    }
}
