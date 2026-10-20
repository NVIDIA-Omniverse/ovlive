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

//! The stack-fill probe: a thread's stack bounds, painting the unused stack
//! below a frame with a pattern, and finding the deepest word that no longer
//! holds it.
//!
//! The measured thread paints from its entry frame, so every frame anything
//! below that entry frame uses afterwards lies in the painted range. A frame
//! that was allocated but whose deepest bytes were never written is not
//! seen: the probe under-reports by that tail. A stack probe touching each
//! page of a large frame writes, if at all, inside the frame being
//! allocated, so it makes the probe report no memory the frame did not take.

use std::hint::black_box;
use std::ptr;

pub const KIB: usize = 1024;
pub const MIB: usize = 1024 * KIB;

/// The stack every tokio worker of the service gets: Rust's default thread
/// stack (std's `DEFAULT_MIN_STACK_SIZE`, 2 MiB). `main.rs`'s
/// `#[tokio::main]` sets no `thread_stack_size`, so tokio passes no size and
/// std falls back to `RUST_MIN_STACK` or to this; neither the chart nor the
/// image sets `RUST_MIN_STACK`. Changes only with SDD §15.1.
pub const PRODUCT_WORKER_STACK: usize = 2 * MIB;

/// The most stack one case may use: a quarter of [`PRODUCT_WORKER_STACK`].
/// The other three quarters are the margin for frames that nest inside an
/// RPC on the worker but that the guard does not count: the real auth
/// clients' (the cases use mocks), codegen differences between this binary
/// and the service binary and between architectures, the tail of the
/// deepest frame the probe under-reports, and the difference between the
/// guard's current-thread scheduler loop and production's multi-thread
/// workers'. Work that runs independently of an RPC — background tasks,
/// fred's router tasks, closures on the blocking pool — adds nothing to an
/// RPC's depth and is not covered by this budget at all (Issue b.w7c).
/// Changes only alongside the worker stack size, never to make a breach
/// pass (SDD §15.1).
pub const STACK_BUDGET: usize = 512 * KIB;

/// How far the reported stack size of a measured thread may sit below
/// [`PRODUCT_WORKER_STACK`]: a C library that carves its guard page (up to a
/// 64 KiB page) out of the requested size reports that much less.
const WORKER_STACK_TOLERANCE_BELOW: usize = 64 * KIB;

/// How far it may sit above: page rounding of the request, and std raising
/// a request below the platform minimum. Far below the 8 MiB `RUST_MIN_STACK`
/// the cargo-run tests get (`.cargo/config.toml` `[env]`), so an inherited
/// size fails the check.
const WORKER_STACK_TOLERANCE_ABOVE: usize = 256 * KIB;

/// The paint: an odd, distinctive word no zero-fill or small integer
/// matches.
const PATTERN: u64 = 0x57AC_C0DE_57AC_C0DE;

/// How far below the entry frame's anchor the paint stops. Room for the
/// part of the entry frame below its anchor and for the paint and scan
/// calls' own frames, which must not paint over themselves.
const ENTRY_RESERVE: usize = 16 * KIB;

/// One thread's stack, as the C library reports it.
#[derive(Clone, Copy, Debug)]
pub struct StackBounds {
    /// The lowest usable address. For a thread `pthread_create` made, glibc
    /// places the guard at the low end of the stack mapping and reports the
    /// stack above it: `pthread_getattr_np` sets the size to the block less
    /// the guard, and `pthread_attr_getstack` returns the top less that size
    /// (glibc `nptl/pthread_getattr_np.c`, `nptl/pthread_attr_getstack.c`,
    /// `guard_position` in `nptl/allocatestack.c`).
    pub lo: usize,
    /// The usable size, the guard excluded.
    pub size: usize,
}

impl StackBounds {
    /// The calling thread's stack.
    pub fn of_current_thread() -> Self {
        // SAFETY: `attr` is initialised by `pthread_getattr_np` before any
        // read and destroyed once; the out-pointers are valid locals.
        unsafe {
            let mut attr: libc::pthread_attr_t = std::mem::zeroed();
            assert_eq!(
                libc::pthread_getattr_np(libc::pthread_self(), &mut attr),
                0,
                "pthread_getattr_np failed"
            );
            let mut addr: *mut libc::c_void = ptr::null_mut();
            let mut size: libc::size_t = 0;
            let stack = libc::pthread_attr_getstack(&attr, &mut addr, &mut size);
            libc::pthread_attr_destroy(&mut attr);
            assert_eq!(stack, 0, "pthread_attr_getstack failed");
            StackBounds {
                lo: addr as usize,
                size,
            }
        }
    }
}

/// Whether `bounds` is a stack the size of a product worker's: within a few
/// pages of [`PRODUCT_WORKER_STACK`]. A thread whose explicit size did not
/// take — one sized by an inherited `RUST_MIN_STACK` of 8 MiB, or handed a
/// larger cached stack by glibc (see `harness`) — fails it.
pub fn check_worker_stack(what: &str, bounds: &StackBounds) -> Result<(), String> {
    let lowest = PRODUCT_WORKER_STACK - WORKER_STACK_TOLERANCE_BELOW;
    let highest = PRODUCT_WORKER_STACK + WORKER_STACK_TOLERANCE_ABOVE;
    if (lowest..=highest).contains(&bounds.size) {
        Ok(())
    } else {
        Err(format!(
            "{what} has a {} KiB stack, not the product worker's {} KiB (accepted: {}..={} \
             KiB). Its explicit stack size did not take — an inherited `RUST_MIN_STACK` sizes \
             a thread that sets none — so it does not stand for the product's 2 MiB workers",
            bounds.size / KIB,
            PRODUCT_WORKER_STACK / KIB,
            lowest / KIB,
            highest / KIB,
        ))
    }
}

/// The painted range of one measurement, `[lo, hi)`.
#[derive(Clone, Copy, Debug)]
pub struct Painted {
    pub lo: usize,
    pub hi: usize,
}

fn page_size() -> usize {
    // SAFETY: `sysconf` only reads a system constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(page).expect("sysconf(_SC_PAGESIZE) failed")
}

/// Paint the calling thread's unused stack below `entry` (the address of a
/// local in the measured thread's entry frame), from one page above the
/// lowest usable address (already above the guard; the page is a margin)
/// up to [`ENTRY_RESERVE`] below `entry`.
///
/// Must be called from the entry frame, with nothing running on the thread:
/// everything in the painted range is unused at this point.
#[inline(never)]
pub fn paint_below(entry: usize, bounds: &StackBounds) -> Painted {
    let page = page_size();
    let lo = (bounds.lo + page + 7) & !7;
    let hi = (entry - ENTRY_RESERVE) & !7;
    let here = black_box(&page) as *const usize as usize;
    assert!(
        lo < hi && hi < here,
        "probe self-check: the paint range [{lo:#x}, {hi:#x}) must lie below this frame \
         ({here:#x}) and above the guard"
    );
    let mut word = lo;
    while word < hi {
        // SAFETY: `[lo, hi)` is this thread's own stack, between the guard
        // and the current frame, and nothing is using it (see above).
        unsafe { ptr::write_volatile(word as *mut u64, PATTERN) };
        word += 8;
    }
    painted_fence();
    Painted { lo, hi }
}

/// The lowest word in `painted` that no longer holds the pattern: the
/// deepest write since the paint. `None` when the whole range is intact.
#[inline(never)]
pub fn deepest_write(painted: &Painted) -> Option<usize> {
    painted_fence();
    let mut word = painted.lo;
    while word < painted.hi {
        // SAFETY: as in `paint_below`: this thread's own unused stack.
        if unsafe { ptr::read_volatile(word as *const u64) } != PATTERN {
            return Some(word);
        }
        word += 8;
    }
    None
}

/// Keep the compiler from moving the volatile accesses across the work
/// between paint and scan.
fn painted_fence() {
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// The calibration frame's size: large against the runtime's own frames,
/// small against the budget.
pub const CALIBRATION_FRAME: usize = 256 * KIB;

/// How far above [`CALIBRATION_FRAME`] the calibration may measure: the
/// runtime's frames between the entry frame and the spawned task that runs
/// the frame, under 1 KiB measured on aarch64, with room for architecture
/// and toolchain differences.
pub const CALIBRATION_SLACK: usize = 32 * KIB;

/// A sync frame of [`CALIBRATION_FRAME`] bytes, all written: what the probe
/// must measure.
#[inline(never)]
pub fn calibration_frame() -> u8 {
    let mut frame = [0u8; CALIBRATION_FRAME];
    black_box(&mut frame);
    frame[black_box(CALIBRATION_FRAME - 1)]
}
