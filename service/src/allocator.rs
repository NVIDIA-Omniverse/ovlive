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

//! The service binary's global allocator: jemalloc, with the options that make
//! it return freed memory to the OS compiled into the binary (b.u5k).
//!
//! Under stock glibc malloc the process's resident set ratchets up to each load
//! peak and stays there: glibc returns memory only from the top of each arena
//! heap, so a burst leaves fragmented free space that only `malloc_trim`
//! releases. jemalloc purges freed pages on a decay schedule, and background
//! threads keep purging while the process is idle. SDD §21.2 carries the
//! measurements.
//!
//! Declared only from `main.rs`, never from `lib.rs`: a global allocator in the
//! library would be installed into every integration-test binary, and into any
//! standalone binary linking the library with its own `#[global_allocator]`,
//! which would then stop compiling.
//!
//! Only the Rust heap moves to jemalloc. tikv-jemalloc-sys builds jemalloc
//! with the `_rjem_` symbol prefix, so C and C++ code (the OODF library,
//! libstdc++) keep calling glibc's `malloc`.
//!
//! Compiled only on linux-gnu (see the `Cargo.toml` target table). On macOS
//! jemalloc has no background-thread support, so the compiled-in
//! `background_thread:true` would make jemalloc's initialisation fail and every
//! allocation abort the process; musl has no background-thread support either.
//! Other targets keep the system allocator.
//!
//! # The fan-out arena
//!
//! The snapshot fan-out's threads (`storage::fanout_thread`: `fanout-read`,
//! the `watch-read` threads, `snapshot-io`) all allocate from one arena this
//! module creates at startup ([`install_fanout_arena`]), with a short
//! compiled-in dirty decay ([`FANOUT_ARENA_DIRTY_DECAY_MS`]) in place of
//! jemalloc's default 10 s. Each thread in an arena of its own left its
//! freed pages resident in every one of those arenas until its decay ran,
//! which under concurrent WATCH writers exceeded the window term the
//! catch-up memory budget funds; in one arena purged promptly the slack is
//! one arena's whatever the number of fan-out threads (the fan-out module
//! docs, *One arena*, and the record
//! `docs-internal/validation/fanout-executor-b26e-s2-20261009/`).
//!
//! The library cannot call jemalloc (it is this binary's allocator alone),
//! so the binding is a hook: [`install_fanout_arena`] registers
//! [`bind_thread_to_fanout_arena`] with the library, and each fan-out thread
//! runs it as it starts. `main` installs the arena before the Valkey client
//! connects, so every fan-out thread binds; the INFO line of
//! [`log_effective_options`] reports the arena's index and decay times. If
//! the arena cannot be set up, no hook is registered, the threads keep
//! jemalloc's default arenas, and one WARN says so: memory placement fails
//! open, never the process.
//!
//! For measurement only, a build with the test-only `test-support` cargo
//! feature lets an environment variable replace the decay at startup
//! ([`fanout_arena_dirty_decay_ms`]); the feature is in no default or
//! release build (`Cargo.toml`, `scripts/check-measurement-features.sh`, and
//! the Dockerfile's measurement-feature gate).

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::mem::{size_of, MaybeUninit};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// The jemalloc options compiled into the binary. This is the single source of
/// truth for them; [`EXPECTED_THP`] and [`EXPECTED_BACKGROUND_THREAD`] state the
/// same two values in typed form and must move with it.
///
/// - `background_thread:true` — background threads purge freed pages on the
///   decay schedule even while no allocation activity drives purging, which is
///   what returns memory after load ends.
/// - `thp:never` — jemalloc marks its mappings `MADV_NOHUGEPAGE`. On a node
///   whose transparent huge pages are `always`, khugepaged otherwise collapses
///   purged ranges back into resident huge pages, and resident memory creeps up
///   while idle. We cannot assume the node setting (dev nodes are `always`), so
///   the binary pins it.
const COMPILED_MALLOC_CONF: &CStr = c"background_thread:true,thp:never";

/// The `thp` value [`COMPILED_MALLOC_CONF`] sets.
const EXPECTED_THP: &str = "never";

/// The `opt.thp` value jemalloc reports, whatever was configured, when it
/// cannot open, read or recognise `/sys/kernel/mm/transparent_hugepage/enabled`
/// (or was built without THP support): it then disables its THP handling, so
/// `thp:never` is not applied.
const THP_NOT_SUPPORTED: &str = "not supported";

/// The `background_thread` value [`COMPILED_MALLOC_CONF`] sets.
const EXPECTED_BACKGROUND_THREAD: bool = true;

/// jemalloc's compiled-in configuration symbol, defined by this binary.
///
/// **Why a symbol and not a build or deploy environment variable.** jemalloc
/// declares `malloc_conf` weak and reads it as its second configuration
/// source, so this strong definition wins at link time and every build of the
/// binary carries the options: the Docker image, a local `cargo build
/// --release`, the debug binary the conformance runner starts, `cargo run`.
/// `JEMALLOC_SYS_WITH_MALLOC_CONF` would only guard builds that go through
/// whichever environment sets it; a build script cannot set a dependency's
/// build-script environment at all.
///
/// **Still overridable per key.** jemalloc applies its sources in order — the
/// `--with-malloc-conf` string, this symbol, the `/etc/_rjem_malloc.conf`
/// symlink name (absent in the distroless image), then the `_RJEM_MALLOC_CONF`
/// environment variable — and a later source overrides an earlier one key by
/// key. `_RJEM_MALLOC_CONF` (the chart's `allocator.mallocConf`) therefore
/// overrides these options; plain `MALLOC_CONF` is ignored by this prefixed
/// build.
///
/// **Coupled to the prefixed symbol name.** `_rjem_malloc_conf` is the name
/// jemalloc reads because tikv-jemalloc-sys builds it with the `_rjem_` prefix.
/// Enabling tikv-jemalloc-sys's `unprefixed_malloc_on_supported_platforms`
/// feature renames the symbol jemalloc reads to `malloc_conf`, and this export
/// would then be silently ignored. The test module pins that jemalloc reads
/// this definition.
#[export_name = "_rjem_malloc_conf"]
static MALLOC_CONF: Option<&'static c_char> =
    // SAFETY: `COMPILED_MALLOC_CONF` is a `'static` C string literal, so the
    // pointer to its first byte is valid for `'static` and the reference never
    // dangles. jemalloc only reads through it, up to the terminating NUL.
    Some(unsafe { &*COMPILED_MALLOC_CONF.as_ptr() });

/// Value types `mallctl` reads and writes for the names used in this module.
/// Every bit pattern jemalloc writes for these names is a valid value of the
/// Rust type: C `bool` (0 or 1), `unsigned`, `uint64_t`, `size_t`, `ssize_t`
/// (the decay times), `const char *`, and the `uint64_t *` that
/// `thread.allocatedp` returns (a pointer to the calling thread's own
/// allocation counter, available whether or not `stats` is compiled in).
trait MallctlValue: Copy {}
impl MallctlValue for bool {}
impl MallctlValue for u32 {}
impl MallctlValue for u64 {}
impl MallctlValue for usize {}
impl MallctlValue for isize {}
impl MallctlValue for *const c_char {}
impl MallctlValue for *mut u64 {}

/// Read one fixed-size `mallctl` value, or `None` when jemalloc rejects the
/// name or reports a size other than `T`'s.
fn read<T: MallctlValue>(name: &CStr) -> Option<T> {
    let mut value = MaybeUninit::<T>::uninit();
    let mut len = size_of::<T>();
    // SAFETY: `name` is NUL-terminated; `value` has room for `len` bytes and
    // `len` is a valid out-parameter. No new value is written (null, 0).
    let rc = unsafe {
        tikv_jemalloc_sys::mallctl(
            name.as_ptr(),
            value.as_mut_ptr().cast::<c_void>(),
            &mut len,
            ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len != size_of::<T>() {
        return None;
    }
    // SAFETY: jemalloc returned success and wrote exactly `size_of::<T>()`
    // bytes of a value of the type `MallctlValue` vouches for.
    Some(unsafe { value.assume_init() })
}

/// Read a `const char *` option as an owned string.
fn read_str(name: &CStr) -> Option<String> {
    let value: *const c_char = read(name)?;
    if value.is_null() {
        return None;
    }
    // SAFETY: jemalloc's string options point at NUL-terminated static
    // storage that lives for the whole process.
    Some(
        unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// Write one fixed-size `mallctl` value; jemalloc's error code (an `errno`
/// value) when it rejects the name or the value.
fn write<T: MallctlValue>(name: &CStr, mut value: T) -> Result<(), c_int> {
    // SAFETY: `name` is NUL-terminated; `value` is a valid `T` of the size
    // passed, which jemalloc only reads. No old value is read (null, null).
    let rc = unsafe {
        tikv_jemalloc_sys::mallctl(
            name.as_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            (&mut value as *mut T).cast::<c_void>(),
            size_of::<T>(),
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(rc)
    }
}

/// Advance jemalloc's statistics epoch so the next `stats.*` read reflects
/// current state rather than the last refresh.
fn advance_epoch() -> Option<()> {
    let mut epoch: u64 = 1;
    // SAFETY: `epoch` is a valid `uint64_t` of the size passed; no old value is
    // read (null, null).
    let rc = unsafe {
        tikv_jemalloc_sys::mallctl(
            c"epoch".as_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            (&mut epoch as *mut u64).cast::<c_void>(),
            size_of::<u64>(),
        )
    };
    (rc == 0).then_some(())
}

/// The effective `opt.thp` option: `default`, `always` or `never`, or
/// [`THP_NOT_SUPPORTED`] (`not supported`) when jemalloc could not detect the
/// host's transparent-huge-page state.
fn effective_thp() -> Option<String> {
    read_str(c"opt.thp")
}

/// Whether background threads are enabled right now (the runtime
/// `background_thread` control, not only the option it started from).
fn background_thread_enabled() -> Option<bool> {
    read(c"background_thread")
}

/// The effective `opt.narenas` option: the arena limit jemalloc runs with.
fn effective_narenas() -> Option<u32> {
    read(c"opt.narenas")
}

/// Whether jemalloc's heap profiler is compiled in (`config.prof`). True only
/// in a build with the `jemalloc-profiling` cargo feature.
fn profiling_compiled_in() -> Option<bool> {
    read(c"config.prof")
}

/// Bytes currently allocated by the application through jemalloc
/// (`stats.allocated`), after advancing the statistics epoch. Source for the
/// `live_edit_jemalloc_allocated_bytes` gauge; `None` on a `mallctl` error.
pub(crate) fn allocated_bytes() -> Option<u64> {
    advance_epoch()?;
    read::<usize>(c"stats.allocated").map(|bytes| bytes as u64)
}

/// jemalloc's upper bound on bytes in physically resident data pages it has
/// mapped (`stats.resident`), after advancing the statistics epoch. A maximum,
/// not a precise count: demand-zeroed pages never touched are included, so it
/// can exceed process RSS. Source for the `live_edit_jemalloc_resident_bytes`
/// gauge; `None` on a `mallctl` error.
pub(crate) fn resident_bytes() -> Option<u64> {
    advance_epoch()?;
    read::<usize>(c"stats.resident").map(|bytes| bytes as u64)
}

/// jemalloc's `MALLCTL_ARENAS_ALL`: the arena index whose `stats.arenas.<i>.*`
/// statistics sum every arena (`include/jemalloc/jemalloc_macros.h.in:47` in
/// the vendored jemalloc 5.3.0; `src/ctl.c` merges each initialised arena,
/// manual ones included, into it).
const MALLCTL_ARENAS_ALL: u32 = 4096;

/// Bytes in dirty pages — freed by the service, not yet purged back to the OS
/// — after advancing the statistics epoch, split between the fan-out arena
/// and every other arena: `(automatic, fanout)`, each `stats.arenas.<i>.pdirty`
/// × `arenas.page`. `automatic` is every arena but the fan-out arena
/// (jemalloc's automatic arenas, its huge arena included: the sum over all
/// arenas less the fan-out arena's); `fanout` is `None` when the fan-out arena
/// was not set up ([`install_fanout_arena`]), whose threads then allocate in
/// the automatic arenas. Source for the `live_edit_jemalloc_dirty_bytes{arenas}`
/// gauge, read at each collection like the other allocator gauges; `None` on
/// a `mallctl` error.
pub(crate) fn dirty_bytes() -> Option<(u64, Option<u64>)> {
    advance_epoch()?;
    let page = read::<usize>(c"arenas.page")? as u64;
    let pdirty = |index: u32| -> Option<u64> {
        let name = CString::new(format!("stats.arenas.{index}.pdirty")).ok()?;
        read::<usize>(&name).map(|pages| pages as u64)
    };
    let all = pdirty(MALLCTL_ARENAS_ALL)?;
    let fanout = match FANOUT_ARENA_INDEX.get() {
        Some(&index) => Some(pdirty(index)?),
        None => None,
    };
    Some((
        all.saturating_sub(fanout.unwrap_or(0)).saturating_mul(page),
        fanout.map(|pages| pages.saturating_mul(page)),
    ))
}

// ── The fan-out arena ───────────────────────────────────────────────────

/// The fan-out arena's dirty-page decay, in milliseconds: how long a page the
/// fan-out threads freed stays resident before the arena's background thread
/// purges it. 1,000 ms.
///
/// Short, because the fan-out's resident memory beyond its live heap is what
/// this arena keeps unpurged, and the window term of the catch-up memory
/// budget's derivation (`storage::memory_budget`) funds only one read
/// stage's transient: at jemalloc's default 10 s, four concurrent WATCH
/// writers' reads left more than the term in every measured round (the
/// fan-out module docs, *One arena*).
///
/// 1,000 ms is the longest of the three decays measured, 0, 100 and
/// 1,000 ms, that kept every round of the concurrent arm (four structural
/// WATCH writers and one fenced 10x catch-up on a release build) under the
/// 143.0 MiB window term with the latency targets met. Its peak resident
/// growth beyond the memory the budgets held at that peak was at most
/// 67.5 MiB at 2 CPUs in the sweep, and 24.4 and 23.4 MiB at 2 and 6 CPUs
/// in the acceptance runs. Its 4-writer p50 and catch-up grant hold at
/// 2 CPUs, against the build before the fan-out threads, were within the
/// +25 % target: sweep p50 +2.4 % and hold −2.8 %, acceptance −0.0 % and
/// −3.9 % (medians). 0 ms, which purges on every free with a page-return
/// system call on the freeing thread, cost +17.6 % on the writers' p50
/// against the build before the fan-out threads (+14.8 % against 1,000 ms),
/// and 0.34 s more `fanout-read` CPU per round than 1,000 ms. A lone
/// catch-up's or cold start's grant hold is the same at 100 and 1,000 ms.
/// The record:
/// `docs-internal/validation/fanout-executor-b26e-s2-20261009/`.
///
/// The decay also covers the fan-out's largest buffers. A thread on one of
/// jemalloc's automatic arenas takes every allocation of `oversize_threshold`
/// (8 MiB) or more from jemalloc's huge arena, whose decay is 0 (it purges
/// eagerly); a thread bound to a manual arena, as every fan-out thread is,
/// takes them from that arena (the vendored jemalloc 5.3.0 in
/// tikv-jemalloc-sys: `arena_choose_maybe_huge`,
/// `include/jemalloc/internal/arena_inlines_b.h:21-39`, and the huge arena's
/// eager purge, `src/arena.c:1725-1738`). So those buffers' freed pages wait
/// out this decay as well, and the 1,000 ms measurement includes them.
///
/// Compiled in, like [`COMPILED_MALLOC_CONF`], never a setting: set on the
/// arena itself after it is created, so `_RJEM_MALLOC_CONF`'s
/// `dirty_decay_ms`, which sets the default arenas' decay, does not reach it.
/// Only a build with the test-only `test-support` feature can replace it at
/// startup, for measurement ([`fanout_arena_dirty_decay_ms`]).
const FANOUT_ARENA_DIRTY_DECAY_MS: isize = 1_000;

/// The fan-out arena's muzzy-page decay, in milliseconds: 0, jemalloc's own
/// default, set explicitly so no override of the default arenas' options
/// reaches this arena. With it at 0, a purge of a dirty page returns it to
/// the OS outright (`MADV_DONTNEED`), and resident memory falls when the
/// decay runs. Above 0, a purge would only mark the page reclaimable
/// (`MADV_FREE`), which the kernel takes back only under memory pressure, so
/// the page would stay in the process's resident set and the short dirty
/// decay would not lower it.
const FANOUT_ARENA_MUZZY_DECAY_MS: isize = 0;

/// The environment variable a `test-support` build reads at startup in place
/// of [`FANOUT_ARENA_DIRTY_DECAY_MS`]: whole milliseconds, 0 or more. For
/// measurement only: a measurement binary is a release build with
/// `--features test-support`, which the feature-free release image never is.
/// Such a binary is not the shipped one beyond this override: the feature
/// also turns on the vendored fred's retention-report counters and compiles
/// the crate's `cfg(any(test, feature = "test-support"))` test seams, some on
/// measured paths (the caveat `.gitlab-ci.yml` records at its
/// `test-release-stack-guard` job), so a re-measurement runs with those
/// seams in. Its name is in no build without the feature, which the Dockerfile's
/// measurement-feature gate checks on the release binary (locally:
/// `scripts/check-measurement-features.sh --binary`, and `--control-binary`
/// on a `test-support` build, which must carry it). Both search for this
/// literal, and require it declared exactly once in this file: rename it
/// there too.
#[cfg(feature = "test-support")]
const FANOUT_ARENA_DECAY_OVERRIDE_ENV: &str = "MEASUREMENT_FANOUT_ARENA_DIRTY_DECAY_MS";

/// The fan-out arena as created and configured ([`install_fanout_arena`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FanoutArena {
    /// The arena's index, which each fan-out thread writes to `thread.arena`.
    pub(crate) index: u32,
    /// Its dirty-page decay, read back from the arena.
    pub(crate) dirty_decay_ms: isize,
    /// Its muzzy-page decay, read back from the arena.
    pub(crate) muzzy_decay_ms: isize,
    /// Whether the dirty decay came from the measurement-only override
    /// rather than [`FANOUT_ARENA_DIRTY_DECAY_MS`]. Always false without the
    /// `test-support` feature, and with it while the variable is unset.
    pub(crate) decay_overridden: bool,
}

/// Why the fan-out arena was not set up: the fan-out threads then allocate
/// from jemalloc's default arenas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FanoutArenaError {
    /// `arenas.create` failed, with jemalloc's error code.
    Create { rc: c_int },
    /// `arenas.create` replied with an index of `len` bytes, not an
    /// `unsigned`.
    CreateReplySize { len: usize },
    /// Setting one of the arena's decay times failed.
    SetDecay { option: String, rc: c_int },
    /// A decay time read back from the arena was unreadable or not the value
    /// set.
    DecayReadBack {
        option: String,
        set: isize,
        read: Option<isize>,
    },
    /// Another fan-out thread start hook was already registered, so the
    /// threads would not run this binary's.
    HookAlreadyRegistered,
}

impl fmt::Display for FanoutArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Create { rc } => write!(f, "`arenas.create` failed with error {rc}"),
            Self::CreateReplySize { len } => {
                write!(f, "`arenas.create` replied with {len} bytes, not an index")
            }
            Self::SetDecay { option, rc } => {
                write!(f, "setting `{option}` failed with error {rc}")
            }
            Self::DecayReadBack { option, set, read } => match read {
                Some(read) => write!(f, "`{option}` reads back {read} ms after {set} ms was set"),
                None => write!(f, "`{option}` could not be read back after it was set"),
            },
            Self::HookAlreadyRegistered => {
                f.write_str("a fan-out thread start hook was already registered")
            }
        }
    }
}

/// The fan-out arena's setup, done once per process ([`install_fanout_arena`]).
static FANOUT_ARENA: OnceLock<Result<FanoutArena, FanoutArenaError>> = OnceLock::new();

/// The fan-out arena's index, for the start hook: set only once the arena is
/// configured, and before the hook is registered.
static FANOUT_ARENA_INDEX: OnceLock<u32> = OnceLock::new();

/// Whether a fan-out thread has already logged that it could not bind to the
/// fan-out arena: the WARN is logged once per process.
static FANOUT_BIND_FAILURE_LOGGED: AtomicBool = AtomicBool::new(false);

/// The dirty decay to configure: [`FANOUT_ARENA_DIRTY_DECAY_MS`], or, in a
/// `test-support` build only, the value of
/// [`FANOUT_ARENA_DECAY_OVERRIDE_ENV`] when it is set to whole milliseconds,
/// 0 or more (an unparsable value is WARNed and the constant used). The
/// second element is whether the override applied.
///
/// Gated on `test-support`, the crate's existing test-only feature, rather
/// than a feature of its own: `scripts/check-vendored-fred.sh` and
/// `scripts/check-measurement-features.sh` already keep `test-support` out
/// of every build that does not ask for it on the command line, and every
/// CI lint and test shape already compiles it.
fn fanout_arena_dirty_decay_ms() -> (isize, bool) {
    #[cfg(feature = "test-support")]
    if let Some(raw) = std::env::var_os(FANOUT_ARENA_DECAY_OVERRIDE_ENV) {
        match raw
            .to_str()
            .and_then(|text| text.parse::<isize>().ok())
            .filter(|ms| *ms >= 0)
        {
            Some(ms) => return (ms, true),
            None => tracing::warn!(
                value = ?raw,
                "{FANOUT_ARENA_DECAY_OVERRIDE_ENV} is not whole milliseconds, 0 or more; the \
                 fan-out arena keeps its compiled-in dirty decay"
            ),
        }
    }
    (FANOUT_ARENA_DIRTY_DECAY_MS, false)
}

/// Create a jemalloc arena with the default extent hooks; its index.
fn create_arena() -> Result<u32, FanoutArenaError> {
    let mut index = MaybeUninit::<u32>::uninit();
    let mut len = size_of::<u32>();
    // SAFETY: `arenas.create` writes the new arena's `unsigned` index into
    // `index`, which has room for `len` bytes; no new value is passed (null,
    // 0), so the arena gets jemalloc's default extent hooks.
    let rc = unsafe {
        tikv_jemalloc_sys::mallctl(
            c"arenas.create".as_ptr(),
            index.as_mut_ptr().cast::<c_void>(),
            &mut len,
            ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(FanoutArenaError::Create { rc });
    }
    if len != size_of::<u32>() {
        return Err(FanoutArenaError::CreateReplySize { len });
    }
    // SAFETY: jemalloc returned success and wrote an `unsigned` of the size
    // checked above.
    Ok(unsafe { index.assume_init() })
}

/// The `arena.<index>.<option>` name of one arena's option.
fn arena_option(index: u32, option: &str) -> CString {
    CString::new(format!("arena.{index}.{option}")).expect("an arena option name has no NUL")
}

/// Set one decay time of arena `index` and read it back.
fn set_arena_decay(index: u32, option: &str, ms: isize) -> Result<isize, FanoutArenaError> {
    let name = arena_option(index, option);
    let display = name.to_string_lossy().into_owned();
    write::<isize>(&name, ms).map_err(|rc| FanoutArenaError::SetDecay {
        option: display.clone(),
        rc,
    })?;
    match read::<isize>(&name) {
        Some(read) if read == ms => Ok(read),
        read => Err(FanoutArenaError::DecayReadBack {
            option: display,
            set: ms,
            read,
        }),
    }
}

/// Create the fan-out arena, configure its decay, and register the hook that
/// binds each fan-out thread to it (`storage::fanout_thread`, *One arena*).
/// Once per process; later calls return the first call's outcome.
///
/// Run by `main` at startup, before the Valkey client connects and so before
/// any fan-out thread starts. Fails open: on any failure no hook is
/// registered, the fan-out threads allocate from jemalloc's default arenas,
/// and [`log_effective_options`] WARNs once. An arena created before a later
/// step failed stays unused.
pub(crate) fn install_fanout_arena() -> &'static Result<FanoutArena, FanoutArenaError> {
    FANOUT_ARENA.get_or_init(|| {
        let (dirty_ms, decay_overridden) = fanout_arena_dirty_decay_ms();
        let index = create_arena()?;
        let dirty_decay_ms = set_arena_decay(index, "dirty_decay_ms", dirty_ms)?;
        let muzzy_decay_ms = set_arena_decay(index, "muzzy_decay_ms", FANOUT_ARENA_MUZZY_DECAY_MS)?;
        // Set before the hook can run: the hook reads it.
        let _ = FANOUT_ARENA_INDEX.set(index);
        live_edit::storage::fanout_thread::register_fanout_thread_start_hook(
            bind_thread_to_fanout_arena,
        )
        .map_err(|_| FanoutArenaError::HookAlreadyRegistered)?;
        Ok(FanoutArena {
            index,
            dirty_decay_ms,
            muzzy_decay_ms,
            decay_overridden,
        })
    })
}

/// The fan-out thread start hook: bind the calling thread to the fan-out
/// arena (`thread.arena`), so it and its thread cache allocate from it. Runs
/// on each fan-out thread as it starts. A failed bind leaves the thread on
/// its default arena and WARNs, once per process; it never panics.
fn bind_thread_to_fanout_arena() {
    let Some(&index) = FANOUT_ARENA_INDEX.get() else {
        return;
    };
    if let Err(rc) = write::<u32>(c"thread.arena", index) {
        if !FANOUT_BIND_FAILURE_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                thread = std::thread::current().name(),
                fanout_arena = index,
                rc,
                "a snapshot fan-out thread could not be bound to the fan-out jemalloc arena; it \
                 allocates from its default arena, whose freed pages stay resident for jemalloc's \
                 default decay (logged once)"
            );
        }
    }
}

/// The arena the calling thread allocates from (`thread.arena`): what a test
/// reads on each fan-out thread to pin its binding to [`FanoutArena::index`].
#[cfg(test)]
fn current_thread_arena() -> Option<u32> {
    read(c"thread.arena")
}

/// An effective option that will keep freed memory resident, or one the
/// binary cannot read and so cannot confirm.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RetentionRisk {
    /// `thp` is not `never`, so on a node with transparent huge pages set to
    /// `always` khugepaged re-fills purged ranges.
    Thp { effective: String },
    /// jemalloc reports `thp` as [`THP_NOT_SUPPORTED`]: it could not read or
    /// recognise the host's transparent-huge-page state, so it applies no THP
    /// handling at all and `thp:never` is not in effect. Distinct from
    /// [`Self::Thp`] because it happens with the compiled-in options untouched
    /// — it is a host condition, not an override.
    ThpHostStateUnknown,
    /// Background threads are off, so nothing purges while the process idles.
    BackgroundThreadOff,
    /// The option could not be read, so the binary cannot confirm it.
    Unreadable { option: &'static str },
}

impl RetentionRisk {
    fn message(&self) -> String {
        match self {
            Self::Thp { effective } => format!(
                "jemalloc thp is `{effective}`, not `{EXPECTED_THP}`: on a node whose \
                 transparent huge pages are `always`, khugepaged re-fills purged memory and it \
                 stays resident"
            ),
            Self::ThpHostStateUnknown => format!(
                "jemalloc could not read the host's transparent huge page state \
                 (/sys/kernel/mm/transparent_hugepage/enabled), so `thp:{EXPECTED_THP}` is not \
                 being applied: if the host's transparent huge pages are `always`, khugepaged can \
                 still re-fill purged memory and keep it resident"
            ),
            Self::BackgroundThreadOff => "jemalloc background threads are off: freed memory is \
                                          not purged while the process is idle"
                .to_string(),
            Self::Unreadable { option } => {
                format!(
                    "jemalloc option `{option}` could not be read; memory return is unconfirmed"
                )
            }
        }
    }
}

/// The retention risks in the effective options. Empty for the compiled-in
/// options on a host whose transparent-huge-page state jemalloc can read; a
/// risk means an `_RJEM_MALLOC_CONF` override, a host THP state jemalloc
/// could not detect, or an option jemalloc would not report (`None`, a failed
/// `mallctl` read). Pure so the warn/info decision is testable
/// without a process whose options or host differ.
fn retention_risks(thp: Option<&str>, background_thread: Option<bool>) -> Vec<RetentionRisk> {
    let mut risks = Vec::new();
    match thp {
        Some(effective) if effective == EXPECTED_THP => {}
        Some(THP_NOT_SUPPORTED) => risks.push(RetentionRisk::ThpHostStateUnknown),
        Some(effective) => risks.push(RetentionRisk::Thp {
            effective: effective.to_string(),
        }),
        None => risks.push(RetentionRisk::Unreadable { option: "opt.thp" }),
    }
    match background_thread {
        Some(enabled) if enabled == EXPECTED_BACKGROUND_THREAD => {}
        Some(_) => risks.push(RetentionRisk::BackgroundThreadOff),
        None => risks.push(RetentionRisk::Unreadable {
            option: "background_thread",
        }),
    }
    risks
}

/// Log the allocator options the process is actually running with, once at
/// startup. INFO always; WARN per option that will keep freed memory resident
/// or cannot be confirmed. With the compiled-in options there is none on a
/// host whose transparent-huge-page state jemalloc can read, so a WARN means
/// that `_RJEM_MALLOC_CONF` overrode them (`malloc_conf_env_set` is then
/// true), that jemalloc could not detect the host's THP state and so is not
/// applying `thp:never`, or that jemalloc would not report an option (a failed
/// `mallctl` read). A warning, not a refusal to start: the override
/// exists for diagnosis and tuning, and the host state is outside the
/// service's control. A field whose `mallctl` read failed is omitted from the
/// INFO line.
///
/// The same INFO line carries the fan-out arena ([`install_fanout_arena`]'s
/// outcome, `fanout_arena`): its index and its two decay times as read back
/// from it, and whether the measurement-only override set the dirty decay.
/// When the arena could not be set up those fields are omitted and one WARN
/// says why.
pub(crate) fn log_effective_options(fanout_arena: &Result<FanoutArena, FanoutArenaError>) {
    let thp = effective_thp();
    let background_thread = background_thread_enabled();
    let malloc_conf_env_set = std::env::var_os("_RJEM_MALLOC_CONF").is_some();
    let arena = fanout_arena.as_ref().ok();
    tracing::info!(
        allocator = "jemalloc",
        thp = thp.as_deref(),
        background_thread,
        narenas = effective_narenas(),
        profiling_compiled_in = profiling_compiled_in(),
        malloc_conf_env_set,
        fanout_arena = arena.map(|a| a.index),
        fanout_arena_dirty_decay_ms = arena.map(|a| a.dirty_decay_ms),
        fanout_arena_muzzy_decay_ms = arena.map(|a| a.muzzy_decay_ms),
        fanout_arena_decay_overridden = arena.map(|a| a.decay_overridden),
        "global allocator options"
    );
    for risk in retention_risks(thp.as_deref(), background_thread) {
        tracing::warn!(malloc_conf_env_set, "{}", risk.message());
    }
    if let Err(error) = fanout_arena {
        tracing::warn!(
            %error,
            "the snapshot fan-out threads' dedicated jemalloc arena could not be set up: they \
             allocate from jemalloc's default arenas, whose freed pages stay resident for \
             jemalloc's default decay"
        );
    }
}

/// The structural-planning cache's weight estimate measured against this
/// allocator: it can only be calibrated where jemalloc is the global
/// allocator, which is this binary.
#[cfg(test)]
mod structural_cache_calibration;

/// These tests run in the `live-edit` bin test binary, which links this module
/// and so allocates through the `#[global_allocator]` above: what they read is
/// the allocator and options the service binary runs with. They assume no
/// `_RJEM_MALLOC_CONF` in the test environment; an override present there is
/// exactly what the effective-option tests exist to catch.
#[cfg(test)]
mod tests {
    use std::hint::black_box;

    use super::*;

    /// jemalloc's kernel-THP probe (`init_thp_state` in `src/pages.c`) reads
    /// this file. When it cannot open it, cannot read it, or does not
    /// recognise its content, jemalloc overwrites `opt.thp` with
    /// [`THP_NOT_SUPPORTED`] whatever the configuration asked for.
    const THP_SYSFS: &str = "/sys/kernel/mm/transparent_hugepage/enabled";

    /// The three states `init_thp_state` recognises, byte for byte. Any other
    /// content is treated like a missing file.
    const THP_SYSFS_RECOGNISED: [&str; 3] = [
        "always [madvise] never\n",
        "[always] madvise never\n",
        "always madvise [never]\n",
    ];

    /// Size of the probe allocation in the global-allocator test. Large
    /// enough that no incidental allocation on the test thread can account
    /// for the delta on its own.
    const PROBE_BYTES: usize = 8 << 20;

    // ── (a) jemalloc reads this binary's option string ──────────────────

    #[test]
    fn malloc_conf_symbol_jemalloc_reads_is_this_binarys_export() {
        // SAFETY: a plain read of an immutable `'static` the linker resolved;
        // jemalloc never writes it.
        let seen = unsafe { tikv_jemalloc_sys::malloc_conf };
        let seen = seen.map(|c| c as *const c_char);
        assert_eq!(
            seen,
            Some(COMPILED_MALLOC_CONF.as_ptr()),
            "the `malloc_conf` symbol jemalloc reads must resolve to this module's \
             `_rjem_malloc_conf` export. A different pointer (or None, jemalloc's own weak \
             default) means the export name no longer matches the symbol this build of \
             jemalloc reads, and the compiled-in options are silently ignored"
        );
        // SAFETY: the pointer was just proven to be `COMPILED_MALLOC_CONF`'s,
        // a NUL-terminated `'static` string.
        let text = unsafe { CStr::from_ptr(seen.expect("checked above")) };
        assert_eq!(text, c"background_thread:true,thp:never");
    }

    // ── (b) the effective options are the retention options ─────────────

    #[test]
    fn effective_opt_thp_is_never() {
        let effective = effective_thp();
        match std::fs::read_to_string(THP_SYSFS) {
            Ok(state) if THP_SYSFS_RECOGNISED.contains(&state.as_str()) => {
                assert_eq!(
                    effective.as_deref(),
                    Some(EXPECTED_THP),
                    "this host exposes {THP_SYSFS} as {state:?}, so jemalloc honours `opt.thp`: \
                     it must be `never` (compiled in by `COMPILED_MALLOC_CONF`). Anything else \
                     means the compiled-in string changed or `_RJEM_MALLOC_CONF` overrode it"
                );
            }
            other => {
                // jemalloc could not read or recognise the host's THP state, so
                // it applies no THP handling and reports `not supported`
                // whatever was compiled in; the host may still have THP
                // enabled (the `ThpHostStateUnknown` WARN case). Say so loudly
                // rather than pass as if the option had been checked.
                eprintln!(
                    "NOTE: `opt.thp` could NOT be checked on this host. {THP_SYSFS} is \
                     {other:?}, which jemalloc does not recognise, so it reports `opt.thp` as \
                     `not supported` regardless of the compiled-in `thp:never`."
                );
                assert_eq!(
                    effective.as_deref(),
                    Some(THP_NOT_SUPPORTED),
                    "with {THP_SYSFS} unreadable or unrecognised, jemalloc must report `not \
                     supported`; any other value means this test's model of `init_thp_state` \
                     is wrong and the strict branch above may be mis-selected"
                );
            }
        }
    }

    #[test]
    fn effective_opt_background_thread_is_true() {
        assert_eq!(
            read::<bool>(c"opt.background_thread"),
            Some(EXPECTED_BACKGROUND_THREAD),
            "`opt.background_thread` must be true (compiled in by `COMPILED_MALLOC_CONF`); \
             false means the compiled-in string changed or `_RJEM_MALLOC_CONF` overrode it"
        );
    }

    #[test]
    fn runtime_background_threads_are_enabled() {
        // The option only requests background threads; the runtime control
        // is whether jemalloc actually started them. They are what purges
        // freed pages while the process is idle.
        assert_eq!(background_thread_enabled(), Some(true));
    }

    // ── (c) Rust allocations go through jemalloc ────────────────────────

    #[test]
    fn global_allocator_routes_rust_heap_allocations_through_jemalloc() {
        let counter: *mut u64 = read(c"thread.allocatedp")
            .expect("`thread.allocatedp` must be readable from the running jemalloc");
        // SAFETY: `thread.allocatedp` returns a pointer to the calling
        // thread's own allocation counter, valid for the thread's lifetime;
        // it is only read here, on the thread that owns it.
        let before = unsafe { counter.read_volatile() };
        let probe = black_box(vec![1u8; PROBE_BYTES]);
        // SAFETY: as above.
        let after = unsafe { counter.read_volatile() };
        drop(probe);

        let delta = after.wrapping_sub(before);
        assert!(
            delta >= PROBE_BYTES as u64,
            "an {PROBE_BYTES}-byte `Vec` must be counted by jemalloc's per-thread allocation \
             counter, but it grew by only {delta} bytes: Rust heap allocations are not going \
             through jemalloc (is the `#[global_allocator]` missing or cfg'd off?)"
        );
    }

    // ── the gauge sources read live jemalloc statistics ─────────────────

    /// Held across the `stats.allocated` readings below. Far larger than any
    /// other allocation in this test binary, so a sibling test freeing its
    /// [`PROBE_BYTES`] between the two readings (plain `cargo test` runs this
    /// module's tests on concurrent threads; `stats.*` are process-wide) cannot
    /// hide the rise.
    const STATS_PROBE_BYTES: usize = 64 << 20;

    #[test]
    fn gauge_sources_read_process_wide_stats_that_track_allocation() {
        let before = allocated_bytes().expect(
            "`stats.allocated` must be readable: without it (the `stats` feature off, or a \
             misspelled mallctl name) the `live_edit_jemalloc_allocated_bytes` gauge vanishes",
        );
        let probe = black_box(vec![1u8; STATS_PROBE_BYTES]);
        let allocated = allocated_bytes().expect("`stats.allocated` readable");
        let resident = resident_bytes().expect(
            "`stats.resident` must be readable: without it the \
             `live_edit_jemalloc_resident_bytes` gauge vanishes",
        );
        drop(probe);

        let rise = allocated.saturating_sub(before);
        let floor = (STATS_PROBE_BYTES - PROBE_BYTES) as u64;
        assert!(
            rise >= floor,
            "holding {STATS_PROBE_BYTES} bytes must raise `stats.allocated` by at least {floor} \
             (the probe less one concurrent sibling free), but it rose by {rise}: the reading \
             is stale (is the statistics epoch still advanced before each read?)"
        );
        assert!(
            resident >= allocated,
            "`stats.resident` ({resident}) is an upper bound on jemalloc's resident data pages, \
             never below its active pages, which hold every allocation; it cannot be below \
             `stats.allocated` ({allocated}), so a smaller value means the two sources are \
             swapped or reading the wrong statistic"
        );
    }

    /// b.fsz MR-D4 (D4-T10): the `live_edit_jemalloc_dirty_bytes` source reads
    /// live statistics. Its two figures are whole pages, the `fanout` figure
    /// present exactly when the fan-out arena is set up, and the `automatic`
    /// figure rises when the service frees memory that jemalloc keeps dirty:
    /// 64 MiB in 1 MiB blocks (below the huge threshold, so they go back to an
    /// automatic arena, and above the thread cache's sizes) freed at once are
    /// purged by the arena's decay over seconds, not before the next reading.
    #[test]
    fn dirty_bytes_reads_whole_pages_and_rises_when_freed_pages_wait_for_purging() {
        const BLOCK: usize = 1 << 20;
        const BLOCKS: usize = 64;
        let page = read::<usize>(c"arenas.page").expect("`arenas.page` must be readable") as u64;
        let (before, fanout) = dirty_bytes().expect(
            "`stats.arenas.<i>.pdirty` must be readable: without it the \
             `live_edit_jemalloc_dirty_bytes` gauge vanishes",
        );
        assert_eq!(before % page, 0, "the automatic figure is whole pages");
        assert_eq!(
            fanout.is_some(),
            FANOUT_ARENA_INDEX.get().is_some(),
            "a `fanout` figure exactly when the fan-out arena was set up"
        );
        if let Some(fanout) = fanout {
            assert_eq!(fanout % page, 0, "the fan-out figure is whole pages");
        }

        let blocks: Vec<Vec<u8>> = (0..BLOCKS).map(|_| black_box(vec![1u8; BLOCK])).collect();
        drop(black_box(blocks));
        let (after, _) = dirty_bytes().expect("readable");

        let rise = after.saturating_sub(before);
        let floor = (BLOCKS * BLOCK / 4) as u64;
        assert!(
            rise >= floor,
            "freeing {} bytes in {BLOCK}-byte blocks must leave at least {floor} of them dirty \
             in the automatic arenas until decay purges them, but the figure rose by {rise} \
             (from {before} to {after}): the reading is stale or of the wrong arena",
            BLOCKS * BLOCK
        );
    }

    // ── (d) the heap profiler is compiled in only for the diagnostic build

    #[test]
    fn profiler_is_compiled_in_exactly_when_the_diagnostic_feature_is_on() {
        assert_eq!(
            profiling_compiled_in(),
            Some(cfg!(feature = "jemalloc-profiling")),
            "`config.prof` must follow the `jemalloc-profiling` cargo feature; a profiler in a \
             build without the feature means something else enabled \
             `tikv-jemalloc-sys/profiling`, which the release build must never carry"
        );
    }

    // ── (e) retention_risks: which effective options warn ───────────────

    #[test]
    fn retention_risks_compiled_in_options_report_none() {
        assert_eq!(
            retention_risks(Some(EXPECTED_THP), Some(EXPECTED_BACKGROUND_THREAD)),
            Vec::<RetentionRisk>::new()
        );
    }

    #[test]
    fn retention_risks_thp_other_than_never_reports_the_effective_value() {
        for effective in ["always", "default"] {
            assert_eq!(
                retention_risks(Some(effective), Some(true)),
                vec![RetentionRisk::Thp {
                    effective: effective.to_string()
                }],
                "thp:{effective} lets khugepaged re-fill purged ranges on an `always` node"
            );
        }
    }

    #[test]
    fn retention_risks_thp_not_supported_reports_unknown_host_state_not_an_override() {
        // jemalloc writes `not supported` over the compiled-in `thp:never`
        // when it cannot read the host's THP state; no override is involved,
        // so it must not be reported as one.
        assert_eq!(
            retention_risks(Some(THP_NOT_SUPPORTED), Some(true)),
            vec![RetentionRisk::ThpHostStateUnknown]
        );
    }

    #[test]
    fn retention_risks_background_thread_off_reports_it() {
        assert_eq!(
            retention_risks(Some(EXPECTED_THP), Some(false)),
            vec![RetentionRisk::BackgroundThreadOff]
        );
    }

    #[test]
    fn retention_risks_both_overridden_reports_both() {
        assert_eq!(
            retention_risks(Some("always"), Some(false)),
            vec![
                RetentionRisk::Thp {
                    effective: "always".to_string()
                },
                RetentionRisk::BackgroundThreadOff,
            ]
        );
    }

    #[test]
    fn retention_risks_unreadable_option_is_reported_not_assumed_safe() {
        assert_eq!(
            retention_risks(None, Some(true)),
            vec![RetentionRisk::Unreadable { option: "opt.thp" }]
        );
        assert_eq!(
            retention_risks(Some(EXPECTED_THP), None),
            vec![RetentionRisk::Unreadable {
                option: "background_thread"
            }]
        );
    }

    // ── (f) the fan-out arena ───────────────────────────────────────────
    //
    // These pins call `install_fanout_arena`, which registers the library's
    // process-wide fan-out thread start hook: every fan-out thread this test
    // process starts afterwards binds to the arena. They assume
    // `MEASUREMENT_FANOUT_ARENA_DIRTY_DECAY_MS` is unset (the default): a
    // `test-support` build, which every CI shape of these tests is, compiles
    // in the measurement-only override that reads it. The override's own
    // pins set the variable only in child processes of their own
    // (`install_fanout_arena_takes_the_decay_override_only_as_whole_ms_0_or_more_b26e`).

    /// Bound on one task's report from a fan-out thread.
    #[cfg(feature = "test-support")]
    const THREAD_REPORT_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

    /// The arena `install_fanout_arena` set up; fails the test if it could
    /// not set one up.
    fn installed_fanout_arena() -> FanoutArena {
        *install_fanout_arena()
            .as_ref()
            .expect("the fan-out arena must be set up in the service binary's own allocator")
    }

    /// The arena's decay times, read back from the arena itself, are the
    /// compiled-in constants, and `install_fanout_arena` reports the same
    /// values (the fields of the startup INFO line), not overridden. A
    /// decay set to any other value, or a report of the requested value
    /// rather than the arena's, fails here.
    #[test]
    fn install_fanout_arena_sets_the_arenas_compiled_in_decay_times_b26e() {
        let arena = installed_fanout_arena();
        assert_eq!(
            read::<isize>(&arena_option(arena.index, "dirty_decay_ms")),
            Some(FANOUT_ARENA_DIRTY_DECAY_MS),
            "arena {}'s dirty decay must be the compiled-in {FANOUT_ARENA_DIRTY_DECAY_MS} ms \
             (in a `test-support` build, with \
             MEASUREMENT_FANOUT_ARENA_DIRTY_DECAY_MS unset)",
            arena.index
        );
        assert_eq!(
            read::<isize>(&arena_option(arena.index, "muzzy_decay_ms")),
            Some(FANOUT_ARENA_MUZZY_DECAY_MS),
            "arena {}'s muzzy decay must be the compiled-in {FANOUT_ARENA_MUZZY_DECAY_MS} ms",
            arena.index
        );
        assert_eq!(
            (
                arena.dirty_decay_ms,
                arena.muzzy_decay_ms,
                arena.decay_overridden
            ),
            (
                FANOUT_ARENA_DIRTY_DECAY_MS,
                FANOUT_ARENA_MUZZY_DECAY_MS,
                false
            ),
            "`install_fanout_arena` reports the decay times the arena holds, not overridden"
        );
    }

    /// The arena binds only the fan-out threads: after `install_fanout_arena`
    /// neither the thread that called it nor a worker of an ordinary tokio
    /// runtime (the service's own) allocates from it. jemalloc assigns a new
    /// thread one of its automatic arenas, never one `arenas.create` made,
    /// so a thread on the fan-out arena got there by a bind.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn install_fanout_arena_leaves_the_calling_thread_and_service_workers_off_the_arena_b26e()
    {
        let arena = installed_fanout_arena();
        assert_ne!(
            current_thread_arena(),
            Some(arena.index),
            "the thread that installed the fan-out arena must not be bound to it"
        );
        let (worker, worker_arena) = tokio::spawn(async {
            (
                std::thread::current().name().map(str::to_owned),
                current_thread_arena(),
            )
        })
        .await
        .expect("the service-worker task completes");
        assert_eq!(
            worker.as_deref(),
            Some("tokio-rt-worker"),
            "fixture: the task ran on a worker of the test's own runtime"
        );
        assert!(
            worker_arena.is_some(),
            "fixture: `thread.arena` is readable on a service worker"
        );
        assert_ne!(
            worker_arena,
            Some(arena.index),
            "a service worker must allocate from one of jemalloc's own arenas, not the fan-out \
             arena"
        );
    }

    /// Every thread of every fan-out runtime — `fanout-read`, `snapshot-io`
    /// and each `watch-read` thread, its worker and a blocking-pool thread
    /// it starts — allocates from the fan-out arena once
    /// `install_fanout_arena` ran, as each binds in its start hook.
    ///
    /// Compiled only with `test-support` (`FanoutThreads::new_for_test`),
    /// which every CI shape of this binary's tests passes: without it the
    /// test is absent, not passing.
    #[cfg(feature = "test-support")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn install_fanout_arena_binds_every_thread_of_every_fanout_runtime_to_the_arena_b26e() {
        use live_edit::storage::fanout_thread::{
            watch_read_thread_name, FanoutThreads, FANOUT_READ_THREAD_NAME, FANOUT_RUNTIMES,
            SNAPSHOT_IO_THREAD_NAME,
        };

        let arena = installed_fanout_arena();
        let threads = FanoutThreads::new_for_test().expect("the fan-out threads start");
        let mut runtimes = vec![
            (
                FANOUT_READ_THREAD_NAME.to_owned(),
                threads.fanout_read_handle(),
            ),
            (
                SNAPSHOT_IO_THREAD_NAME.to_owned(),
                threads.snapshot_io_handle(),
            ),
        ];
        runtimes.extend(
            threads
                .watch_read_handles()
                .iter()
                .enumerate()
                .map(|(i, handle)| (watch_read_thread_name(i), handle)),
        );
        assert_eq!(
            runtimes.len(),
            FANOUT_RUNTIMES,
            "fixture: one handle per runtime"
        );

        let this_thread = || {
            (
                std::thread::current().name().map(str::to_owned),
                current_thread_arena(),
            )
        };
        for (runtime, handle) in runtimes {
            let worker = tokio::time::timeout(
                THREAD_REPORT_BUDGET,
                handle.spawn(async move { this_thread() }),
            )
            .await
            .unwrap_or_else(|_| panic!("{runtime}: its worker answers in time"))
            .expect("the worker task completes");
            let blocking =
                tokio::time::timeout(THREAD_REPORT_BUDGET, handle.spawn_blocking(this_thread))
                    .await
                    .unwrap_or_else(|_| panic!("{runtime}: its blocking pool answers in time"))
                    .expect("the blocking task completes");
            for (kind, (ran_on, thread_arena)) in [("worker", worker), ("blocking", blocking)] {
                assert_eq!(
                    ran_on.as_deref(),
                    Some(runtime.as_str()),
                    "fixture: the {kind} task ran on a thread of `{runtime}`"
                );
                assert_eq!(
                    thread_arena,
                    Some(arena.index),
                    "`{runtime}`'s {kind} thread must allocate from the fan-out arena {}",
                    arena.index
                );
            }
        }
    }

    /// Set in a child process of
    /// [`install_fanout_arena_takes_the_decay_override_only_as_whole_ms_0_or_more_b26e`]
    /// to the case it runs: that test, re-run in the child, then reports the
    /// child's fan-out arena instead of spawning cases.
    #[cfg(feature = "test-support")]
    const DECAY_OVERRIDE_CASE_ENV: &str = "LIVE_EDIT_TEST_FANOUT_ARENA_DECAY_CASE";

    /// The marker before a case's child process's one report.
    #[cfg(feature = "test-support")]
    const DECAY_OVERRIDE_REPORT: &str = "fanout-arena-decay-case:";

    /// Bound on one case's child process, which starts this test binary and
    /// runs one test.
    #[cfg(feature = "test-support")]
    const DECAY_OVERRIDE_CASE_BUDGET: std::time::Duration = std::time::Duration::from_secs(60);

    /// Counts the WARN events that name the decay override's variable.
    #[cfg(feature = "test-support")]
    struct OverrideWarns(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    /// Whether any field of an event mentions the decay override's variable.
    #[cfg(feature = "test-support")]
    struct MentionsOverride(bool);

    #[cfg(feature = "test-support")]
    impl tracing::field::Visit for MentionsOverride {
        fn record_debug(&mut self, _: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0 |= format!("{value:?}").contains(FANOUT_ARENA_DECAY_OVERRIDE_ENV);
        }
    }

    #[cfg(feature = "test-support")]
    impl tracing::Subscriber for OverrideWarns {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            if *event.metadata().level() == tracing::Level::WARN {
                let mut mentions = MentionsOverride(false);
                event.record(&mut mentions);
                if mentions.0 {
                    self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }

        fn enter(&self, _: &tracing::span::Id) {}

        fn exit(&self, _: &tracing::span::Id) {}
    }

    /// In a case's child process: install the fan-out arena, once, under a
    /// subscriber counting the override's WARNs, and describe what it set:
    /// the decay `install_fanout_arena` reports, the decay read back from the
    /// arena itself, whether the override applied, and the WARNs.
    #[cfg(feature = "test-support")]
    fn decay_override_case_report() -> String {
        let warns = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let arena = tracing::subscriber::with_default(
            OverrideWarns(std::sync::Arc::clone(&warns)),
            installed_fanout_arena,
        );
        format!(
            "dirty_decay_ms={} read_back={:?} overridden={} override_warns={}",
            arena.dirty_decay_ms,
            read::<isize>(&arena_option(arena.index, "dirty_decay_ms")),
            arena.decay_overridden,
            warns.load(std::sync::atomic::Ordering::SeqCst)
        )
    }

    /// Run one case's child process to its end and return its exit status,
    /// stdout and stderr. Both pipes are drained on threads of their own
    /// while the child runs, so a child that writes more than a pipe buffer
    /// holds never blocks on a full pipe; a child still running after
    /// [`DECAY_OVERRIDE_CASE_BUDGET`] is killed and fails the case with
    /// whatever it wrote.
    #[cfg(feature = "test-support")]
    fn run_decay_override_case(
        case: &str,
        command: &mut std::process::Command,
    ) -> (std::process::ExitStatus, String, String) {
        fn drain(mut pipe: impl std::io::Read + Send + 'static) -> std::thread::JoinHandle<String> {
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                // A read error ends the drain; the bytes read so far are
                // still the child's output.
                pipe.read_to_end(&mut bytes).ok();
                String::from_utf8_lossy(&bytes).into_owned()
            })
        }

        let mut child = command.spawn().expect("the case's child process starts");
        let stdout = drain(child.stdout.take().expect("the child's stdout is piped"));
        let stderr = drain(child.stderr.take().expect("the child's stderr is piped"));
        let deadline = std::time::Instant::now() + DECAY_OVERRIDE_CASE_BUDGET;
        let status = loop {
            if let Some(status) = child
                .try_wait()
                .expect("the case's child process can be waited on")
            {
                break Some(status);
            }
            if std::time::Instant::now() >= deadline {
                child.kill().ok();
                child.wait().ok();
                break None;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        // The child has ended, so both pipes reach end of file.
        let stdout = stdout.join().expect("the stdout drain ends");
        let stderr = stderr.join().expect("the stderr drain ends");
        match status {
            Some(status) => (status, stdout, stderr),
            None => panic!(
                "{case}: the child process did not end within \
                 {DECAY_OVERRIDE_CASE_BUDGET:?}:\n{stdout}\n{stderr}"
            ),
        }
    }

    /// The measurement-only decay override (`test-support` builds): the
    /// variable set to whole milliseconds, 0 or more, sets the arena's dirty
    /// decay, read back from the arena, and reports it overridden; a value
    /// that is not (a negative one, which jemalloc itself would accept as
    /// "never purge", or one with a unit) WARNs once, naming the variable,
    /// and keeps the compiled-in decay, not overridden; unset keeps it with
    /// no WARN.
    ///
    /// `install_fanout_arena` runs once per process and reads the variable
    /// inside that one run, so each case runs in a child process of its own:
    /// this test binary re-run on this one test, with the variable set for
    /// the child alone. Re-running the test binary, rather than starting the
    /// service binary, needs no Valkey or service configuration and reads
    /// the arena directly instead of scraping a startup log line. A child
    /// that ran no test prints no report line and fails the case.
    #[cfg(feature = "test-support")]
    #[test]
    fn install_fanout_arena_takes_the_decay_override_only_as_whole_ms_0_or_more_b26e() {
        if std::env::var_os(DECAY_OVERRIDE_CASE_ENV).is_some() {
            // A case's child process.
            println!("{DECAY_OVERRIDE_REPORT} {}", decay_override_case_report());
            return;
        }

        let compiled = FANOUT_ARENA_DIRTY_DECAY_MS;
        let kept = |warns: usize| {
            format!(
                "dirty_decay_ms={compiled} read_back=Some({compiled}) overridden=false \
                 override_warns={warns}"
            )
        };
        let set = |ms: isize| {
            format!("dirty_decay_ms={ms} read_back=Some({ms}) overridden=true override_warns=0")
        };
        let cases: [(&str, Option<&str>, String); 5] = [
            ("unset", None, kept(0)),
            ("whole ms", Some("250"), set(250)),
            ("zero", Some("0"), set(0)),
            ("negative", Some("-1"), kept(1)),
            ("unparsable", Some("250ms"), kept(1)),
        ];
        // libtest names a test by its path below the crate.
        let module = module_path!()
            .split_once("::")
            .map_or(module_path!(), |(_, below_crate)| below_crate);
        let test =
            format!("{module}::install_fanout_arena_takes_the_decay_override_only_as_whole_ms_0_or_more_b26e");
        let binary = std::env::current_exe().expect("the test binary's path");

        for (case, value, expected) in cases {
            let mut command = std::process::Command::new(&binary);
            command
                .args(["--exact", test.as_str(), "--nocapture", "--test-threads=1"])
                .env(DECAY_OVERRIDE_CASE_ENV, case)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            match value {
                Some(value) => command.env(FANOUT_ARENA_DECAY_OVERRIDE_ENV, value),
                None => command.env_remove(FANOUT_ARENA_DECAY_OVERRIDE_ENV),
            };
            let (status, stdout, stderr) = run_decay_override_case(case, &mut command);
            assert!(
                status.success(),
                "{case}: the child process failed ({status}):\n{stdout}\n{stderr}"
            );
            // libtest prints its `test <name> ...` progress on the line the
            // report starts on.
            let reports: Vec<&str> = stdout
                .lines()
                .filter_map(|line| line.split_once(DECAY_OVERRIDE_REPORT))
                .map(|(_, report)| report.trim())
                .collect();
            assert_eq!(
                reports,
                [expected.as_str()],
                "{case} ({FANOUT_ARENA_DECAY_OVERRIDE_ENV}={value:?}): the fan-out arena's dirty \
                 decay, as reported and as read back, whether it was overridden, and the \
                 override's WARNs; the child's output:\n{stdout}\n{stderr}"
            );
        }
    }
}
