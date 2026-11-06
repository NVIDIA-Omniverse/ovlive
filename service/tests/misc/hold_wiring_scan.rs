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

//! Source scans of the disaster-recovery hold's wiring (SDD SR-5, SR-6) — the
//! parts no runtime test reaches.
//!
//! `src/main.rs` runs only as the production binary, so the hold tests
//! (`tests/integration/dr/`) compose its pieces in process; these scans pin
//! that `main.rs` wires them the way those tests assume. The scans read
//! code lines only — a `//` comment naming a call cannot satisfy them — and
//! anchor on the call, not its arguments' formatting.
//!
//! - every inventory writer (`dr::hold::HOLD_WRITERS`) is registered on the
//!   abort set, and before the serving point;
//! - every task `main.rs` spawns is classified: a writer whose inventory
//!   entry has a task or token registered (not only a not-spawned record), or
//!   a listed non-writer with the reason, so a new spawn in `main.rs` fails
//!   here until it is one or the other. This covers spawns written in
//!   `main.rs` only, so it is no crate-wide guarantee. A task a library
//!   function or another task spawns is not classified by this scan; the
//!   audit in `src/dr/hold/writers.rs` is the record for every such spawn
//!   (the layer store's tracked cleanups are also checked below, by
//!   `layer_store_spawns_every_tracked_cleanup_through_spawn_cleanup`);
//! - a writer whose token leaves a pass in flight running — the GC loop, the
//!   checkpoint clean-up — has its task's handle registered, not only its
//!   token;
//! - the on-access repairs' hold write fence — the token registered under
//!   `HoldWriter::OnAccessRepairs` — is the one the external-value store's
//!   fetch-on-miss rewrite is built with (`with_recovery_hold_fence`);
//! - the health supervisor's `verdict_stale` input and the scaling source's
//!   `held` are receivers of the settings `spawn_hold_reactor` is handed, so
//!   readiness and the autoscaler freeze follow the reactor's signals;
//! - a failed startup hold read, one single read, exits non-zero at once,
//!   with no retry or fallback verdict;
//! - a replica held at startup runs held mode and returns before it spawns
//!   the normal-mode hold poll, with held mode's claim of the startup parts
//!   and the shutdown request made before the socket handoff, and a failed
//!   held mode ending the process non-zero;
//! - the process's one SIGTERM and SIGINT listener registers both handlers
//!   when it is installed rather than when it is first polled, and `main`
//!   installs it before anything that can wait;
//! - the hold poll is the first background task `run` spawns but for the
//!   bootstrap and metrics servers and the startup signal watcher, which
//!   come before the held-mode decision;
//! - held mode (`src/grpc/hold/held_mode.rs`) spawns only its listed tasks
//!   and names no backup store or writer;
//! - the hold poll's token is a child of the process shutdown request, and
//!   the serving path's shutdown signal cancels it first;
//! - the serving path claims the startup parts before the bootstrap
//!   handoff, the serving point and readiness, and skips the serving point,
//!   readiness and the serving line, but not the handoff, when a stop is
//!   requested by the end of the handoff, logging that it only drains;
//! - each tracker's abort follows its shutdown drain;
//! - the serving point is reached before readiness opens;
//! - a task the hold or a stale verdict aborted is reported as aborted, not
//!   as a failure, and the line names both causes;
//! - the layer store spawns every tracked cleanup through `spawn_cleanup`,
//!   the one place that races it against the abort token.

use std::collections::BTreeMap;

use live_edit::dr::hold::{HoldWriter, HOLD_WRITERS};

const MAIN: &str = include_str!("../../src/main.rs");
const TRANSPORT: &str = include_str!("../../src/grpc/transport.rs");
const STARTUP: &str = include_str!("../../src/observability/startup.rs");
const LAYER_STORE: &str = include_str!("../../src/storage/layer.rs");
const HELD_MODE: &str = include_str!("../../src/grpc/hold/held_mode.rs");

/// `src`'s lines with their indices, `//` comment lines left out.
fn code_lines(src: &str) -> impl Iterator<Item = (usize, &str)> {
    src.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
}

/// The first code line of `src` containing `needle`.
fn first(src: &str, needle: &str) -> usize {
    code_lines(src)
        .find(|(_, line)| line.contains(needle))
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| panic!("no code line contains `{needle}`"))
}

/// Every code line of `src` containing `needle`.
fn all(src: &str, needle: &str) -> Vec<usize> {
    code_lines(src)
        .filter(|(_, line)| line.contains(needle))
        .map(|(idx, _)| idx)
        .collect()
}

/// The writer each `hold_writers.register_*` call in `main.rs` names, with
/// the call's line: the first `HoldWriter::<Name>` from the call on.
fn registrations() -> Vec<(String, usize)> {
    registrations_of("hold_writers.register_")
}

/// [`registrations`] of the calls whose text starts with `call`.
fn registrations_of(call: &str) -> Vec<(String, usize)> {
    let lines: Vec<&str> = MAIN.lines().collect();
    all(MAIN, call)
        .into_iter()
        .map(|start| {
            let call: String = lines[start..lines.len().min(start + 6)].join(" ");
            let after = call
                .split("HoldWriter::")
                .nth(1)
                .unwrap_or_else(|| panic!("the call at line {} names no HoldWriter", start + 1));
            let name: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            (name, start)
        })
        .collect()
}

/// What a task `main.rs` spawns is, for the hold.
#[derive(Clone, Copy, Debug)]
enum Spawned {
    /// A store- or backup-writing task (or one aborted with them on a shared
    /// token): the hold aborts it through this inventory entry.
    Writer(HoldWriter),
    /// Writes neither store, with why (the audit in
    /// `src/dr/hold/writers.rs` has the full list).
    NotAWriter(&'static str),
}

/// Every task-spawning function `main.rs` calls, how many call sites it has
/// there, and what the task is for the hold. The scan below fails on any
/// spawn in `main.rs` missing from this list, or with another count, so a new
/// background task cannot land without being classified here, and a writer
/// cannot be classified without its registration.
const MAIN_SPAWNS: &[(&str, usize, Spawned)] = &[
    (
        "spawn",
        3,
        Spawned::NotAWriter(
            "`tokio::spawn` of the bootstrap gRPC server and of the metrics server, which serve, \
             and of the startup signal watcher, which writes only the readiness status and stops \
             those two servers",
        ),
    ),
    (
        "spawn_hold_poll",
        1,
        Spawned::NotAWriter("one read-only EVALSHA of the hold marker"),
    ),
    (
        "spawn_hold_reactor",
        1,
        Spawned::NotAWriter("acts on the gate, the streams and the abort set, and logs"),
    ),
    (
        "spawn_store_memory_refresh",
        1,
        Spawned::NotAWriter("INFO and CLUSTER SLOTS reads"),
    ),
    ("spawn_gc_task", 1, Spawned::Writer(HoldWriter::ExtGc)),
    (
        "spawn_orphan_sweep",
        1,
        Spawned::Writer(HoldWriter::OrphanSweep),
    ),
    (
        "spawn_ext_ref_reconcile",
        1,
        Spawned::Writer(HoldWriter::ExtRefReconcile),
    ),
    (
        "spawn_checkpoint_loop",
        1,
        Spawned::Writer(HoldWriter::CheckpointLoop),
    ),
    (
        "spawn_ext_mirror_workers",
        1,
        Spawned::Writer(HoldWriter::ExtMirrorWorkers),
    ),
    (
        "spawn_ext_backup_repair",
        1,
        Spawned::Writer(HoldWriter::ExtBackupRepair),
    ),
    (
        "spawn_checkpoint_cleanup",
        1,
        Spawned::Writer(HoldWriter::CheckpointCleanup),
    ),
    (
        "spawn_health_supervisor",
        1,
        Spawned::NotAWriter("PINGs; writes only the readiness status"),
    ),
    (
        "spawn_presence_subscriber",
        1,
        Spawned::Writer(HoldWriter::PresenceAndSessionTasks),
    ),
    (
        "spawn_presence_sweeper",
        1,
        Spawned::Writer(HoldWriter::PresenceAndSessionTasks),
    ),
    (
        "spawn_cache_sweeper",
        1,
        Spawned::Writer(HoldWriter::PresenceAndSessionTasks),
    ),
    (
        "spawn_session_delete_subscriber",
        1,
        Spawned::Writer(HoldWriter::PresenceAndSessionTasks),
    ),
    (
        "spawn_session_tokens_sweeper",
        1,
        Spawned::Writer(HoldWriter::PresenceAndSessionTasks),
    ),
    (
        "spawn_blocking",
        1,
        Spawned::NotAWriter("the metrics provider's shutdown, at process exit"),
    ),
];

/// Every task-spawning call on a code line of `src`: the called function's
/// name (`spawn` or `spawn_<name>`, whatever its path) and the line. A name
/// that is part of a longer identifier (`register_not_spawned`) or not
/// followed by `(` (an import) is not a call.
fn spawn_calls(src: &str) -> Vec<(String, usize)> {
    let mut calls = Vec::new();
    for (idx, line) in code_lines(src) {
        for (at, _) in line.match_indices("spawn") {
            let before = line[..at].chars().next_back();
            if before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
                continue;
            }
            let name: String = line[at..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if line[at + name.len()..].starts_with('(') {
                calls.push((name, idx));
            }
        }
    }
    calls
}

/// Every entry of the inventory has a `register_task`, `register_token` or
/// `register_not_spawned` call in `main.rs`, and every such call comes before
/// the serving point, so the entry routine finds them all registered.
#[test]
fn main_registers_every_hold_writer_before_the_serving_point() {
    let registered = registrations();
    let serving_point = first(MAIN, "hold_serving_point.reach(");
    for writer in HOLD_WRITERS {
        let name = format!("{writer:?}");
        assert!(
            registered.iter().any(|(n, _)| *n == name),
            "main.rs registers nothing for HoldWriter::{name}: a hold would leave it running"
        );
    }
    for (name, line) in &registered {
        assert!(
            *line < serving_point,
            "HoldWriter::{name} is registered at line {} after the serving point (line {})",
            line + 1,
            serving_point + 1
        );
    }
}

/// Every task `main.rs` spawns is classified in [`MAIN_SPAWNS`], with its
/// count of call sites, and every writer among them has a task or a token
/// registered on the abort set for its inventory entry, before the serving
/// point. A `register_not_spawned` record alone does not count: it is the
/// arm of a replica that did not spawn the task, so a writer whose spawned
/// arm registers nothing — the external-value backup repair before its
/// registration landed — would pass the inventory check above and fail here.
///
/// **Scope: spawns written in `main.rs` only.** This is not a crate-wide
/// guarantee. A task that a library function or another task spawns on its
/// own is not classified by this scan; the audit in `src/dr/hold/writers.rs`
/// is the record for every such spawn. One family of them is checked
/// elsewhere in this file: the layer store's tracked cleanups
/// (`layer_store_spawns_every_tracked_cleanup_through_spawn_cleanup`).
// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
#[test]
fn main_classifies_every_spawned_task_and_registers_every_writer() {
    let mut found: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (name, line) in spawn_calls(MAIN) {
        found.entry(name).or_default().push(line);
    }
    for (name, lines) in &found {
        let lines_1: Vec<usize> = lines.iter().map(|l| l + 1).collect();
        let Some((_, count, _)) = MAIN_SPAWNS.iter().find(|entry| entry.0 == name.as_str()) else {
            panic!(
                "main.rs spawns a task through `{name}` (lines {lines_1:?}), which this scan does \
                 not classify. If it writes the store or the backup store, add a HoldWriter \
                 variant for it, register its handle and token on hold_writers before the \
                 serving point, and list it here as a writer; if it writes neither, list it \
                 here as not a writer, with why, and in the audit of src/dr/hold/writers.rs"
            );
        };
        assert_eq!(
            lines.len(),
            *count,
            "main.rs calls `{name}` at lines {lines_1:?}, not the {count} sites this scan \
             classifies: classify the new one"
        );
    }
    for (name, _, _) in MAIN_SPAWNS {
        assert!(
            found.contains_key(*name),
            "MAIN_SPAWNS lists `{name}`, which main.rs no longer calls: remove it"
        );
    }

    let serving_point = first(MAIN, "hold_serving_point.reach(");
    let mut aborts = registrations_of("hold_writers.register_task(");
    aborts.extend(registrations_of("hold_writers.register_token("));
    for (name, _, spawned) in MAIN_SPAWNS {
        let writer = match spawned {
            Spawned::NotAWriter(why) => {
                assert!(
                    !why.is_empty(),
                    "`{name}` is listed as not a writer with no reason"
                );
                continue;
            }
            Spawned::Writer(writer) => writer,
        };
        let wanted = format!("{writer:?}");
        assert!(
            HOLD_WRITERS.contains(writer),
            "`{name}` spawns HoldWriter::{wanted}, which is missing from HOLD_WRITERS"
        );
        assert!(
            aborts
                .iter()
                .any(|(n, line)| *n == wanted && *line < serving_point),
            "main.rs spawns `{name}` but registers no task or token for HoldWriter::{wanted} \
             before the serving point: a hold or a stale verdict would leave it running"
        );
    }
}

/// The inventory writers whose token does not stop a pass already running at
/// hold entry, with why: only the abort of the task's handle does. The scan
/// above accepts a task or a token, so it cannot tell a writer registered by
/// its token alone from one registered by its handle.
const ONLY_THE_ABORT_STOPS_A_PASS: &[(HoldWriter, &str)] = &[
    (
        HoldWriter::ExtGc,
        "the GC loop reads its token only between passes; hold_writer_abort_test.rs",
    ),
    (
        HoldWriter::CheckpointCleanup,
        "the clean-up's pass never reads its token between a delete and its post-delete step, \
         so a delete in flight runs its post-delete step and its settle; \
         hold_quiescence_test.rs, \
         a_cleanup_pass_in_flight_at_entry_starts_nothing_after_it_and_ends_aborted",
    ),
];

/// Every writer in [`ONLY_THE_ABORT_STOPS_A_PASS`] has its task's handle
/// registered on the abort set before the serving point, not only its token.
// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
#[test]
fn main_registers_the_task_of_every_writer_whose_token_leaves_a_pass_running() {
    let serving_point = first(MAIN, "hold_serving_point.reach(");
    let tasks = registrations_of("hold_writers.register_task(");
    for (writer, why) in ONLY_THE_ABORT_STOPS_A_PASS {
        let name = format!("{writer:?}");
        assert!(
            tasks
                .iter()
                .any(|(n, line)| *n == name && *line < serving_point),
            "main.rs registers no task handle for HoldWriter::{name} before the serving point: \
             its token alone leaves a pass in flight at hold entry running ({why})"
        );
    }
}

/// The one code line of `src` containing `needle`, asserting there is exactly
/// one, so a duplicate wiring cannot pass on the first match.
fn only(src: &str, needle: &str) -> (usize, String) {
    let hits: Vec<(usize, &str)> = code_lines(src)
        .filter(|(_, line)| line.contains(needle))
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one code line containing `{needle}`, found {:?}",
        hits.iter().map(|(i, _)| i + 1).collect::<Vec<_>>()
    );
    (hits[0].0, hits[0].1.to_owned())
}

/// The text inside the outermost parentheses that follow `call` on `line`
/// (`call` included in the match, e.g. `.with_recovery_hold_fence(`), trimmed.
fn call_argument(line: &str, call: &str) -> String {
    let start = line.find(call).expect("the call is on the line") + call.len();
    let mut depth = 1usize;
    for (i, c) in line[start..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return line[start..start + i].trim().to_owned();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced parentheses after `{call}` on: {line}")
}

/// The value of `field` in the struct literal `main.rs` opens on the one code
/// line containing `opener`: the text after `field:` on the literal's line
/// that starts with it, its trailing comma dropped. The literal ends at the
/// first closing line indented no deeper than the opener, so a nested block
/// (a closure, an inner literal) does not end it.
fn literal_field(opener: &str, field: &str) -> (usize, String) {
    let (open, open_line) = only(MAIN, opener);
    let indent = open_line.len() - open_line.trim_start().len();
    let lines: Vec<&str> = MAIN.lines().collect();
    for (idx, line) in lines.iter().enumerate().skip(open + 1) {
        let trimmed = line.trim_start();
        if trimmed.starts_with('}') && line.len() - trimmed.len() <= indent {
            break;
        }
        if let Some(value) = trimmed.strip_prefix(&format!("{field}:")) {
            return (idx, value.trim().trim_end_matches(',').trim().to_owned());
        }
    }
    panic!(
        "the literal opened by `{opener}` (line {}) sets no `{field}`",
        open + 1
    )
}

/// The settings `main.rs` hands `spawn_hold_reactor`: the call's last
/// argument, a plain binding.
fn reactor_settings_binding() -> String {
    let (start, _) = only(MAIN, "spawn_hold_reactor(");
    let lines: Vec<&str> = MAIN.lines().collect();
    let call: String = lines[start..]
        .iter()
        .take_while(|l| !l.trim_start().starts_with(");"))
        .copied()
        .collect::<Vec<_>>()
        .join(" ");
    let args = call_argument(&format!("{call})"), "spawn_hold_reactor(");
    let last = args
        .split(',')
        .map(str::trim)
        .rfind(|a| !a.is_empty())
        .expect("spawn_hold_reactor takes arguments")
        .to_owned();
    assert!(
        last.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
        "spawn_hold_reactor's settings must be a plain binding the signals are subscribed \
         from, got `{last}`"
    );
    only(
        MAIN,
        &format!("let {last} = HoldReactorSettings::production()"),
    );
    last
}

/// `field` (a struct-literal field set to `wrap` around a binding, e.g.
/// `Some(rx)`, or to the binding itself for an empty `wrap`) is a receiver
/// subscribed from the reactor settings' `signal` sender: its binding is
/// made, exactly once, by `let rx = <settings>.<signal>.subscribe();`.
fn assert_receiver_of_reactor_signal(opener: &str, field: &str, wrap: &str, signal: &str) {
    let settings = reactor_settings_binding();
    let (line, value) = literal_field(opener, field);
    let binding = if wrap.is_empty() {
        value.clone()
    } else {
        value
            .strip_prefix(&format!("{wrap}("))
            .and_then(|v| v.strip_suffix(')'))
            .unwrap_or_else(|| {
                panic!(
                    "`{field}` (line {}) must be `{wrap}(<receiver>)`, got `{value}`",
                    line + 1
                )
            })
            .to_owned()
    };
    let subscribed = format!("let {binding} = {settings}.{signal}.subscribe();");
    let (sub_line, _) = only(MAIN, &subscribed);
    assert!(
        code_lines(MAIN)
            .filter(|(_, l)| l.contains(&format!("let {binding} ")))
            .count()
            == 1,
        "`{binding}` must be bound once, by `{subscribed}`"
    );
    let (spawn, _) = only(MAIN, "spawn_hold_reactor(");
    assert!(
        sub_line < spawn,
        "`{binding}` must be subscribed before the settings are moved into the reactor"
    );
}

/// The external-value store's fetch-on-miss rewrite (SDD SR-6.2) is fenced by
/// the same token the abort set registers under `HoldWriter::OnAccessRepairs`,
/// so the hold and stale-verdict entries close it. Fails if the
/// `with_recovery_hold_fence` line is removed, duplicated, or handed any
/// other token.
#[test]
fn main_fences_the_fetch_on_miss_rewrite_with_the_on_access_repairs_token() {
    let (_, registration) = only(MAIN, "register_token(HoldWriter::OnAccessRepairs,");
    let registered = call_argument(&registration, "register_token(")
        .trim_start_matches("HoldWriter::OnAccessRepairs,")
        .trim()
        .trim_end_matches(".clone()")
        .to_owned();
    only(
        MAIN,
        &format!("let {registered} = CancellationToken::new();"),
    );
    let (fence_line, fence) = only(MAIN, ".with_recovery_hold_fence(");
    let handed = call_argument(&fence, ".with_recovery_hold_fence(");
    assert_eq!(
        handed.trim_end_matches(".clone()"),
        registered,
        "the fetch-on-miss rewrite must be fenced by the token registered under \
         HoldWriter::OnAccessRepairs (line {})",
        fence_line + 1
    );
    let serving_point = first(MAIN, "hold_serving_point.reach(");
    assert!(
        fence_line < serving_point,
        "the store is built with its fence before the serving point"
    );
}

/// The store out-of-memory signal (`storage::store_oom`) is put behind the
/// same token the abort set registers under `HoldWriter::OnAccessRepairs`, on
/// the replica's own client, so after hold or stale-verdict entry no
/// out-of-memory collection request is sent (kl-retire F3; SDD §22.21). The
/// call is one statement across lines (`valkey` / `.store_oom_signal()` /
/// `.set_hold_fence(…)`), compared with its whitespace removed. Fails if the
/// statement is removed, duplicated, made on another value, or handed any
/// other token, or comes after the serving point.
#[test]
fn main_fences_the_store_oom_signal_with_the_on_access_repairs_token() {
    let (_, registration) = only(MAIN, "register_token(HoldWriter::OnAccessRepairs,");
    let registered = call_argument(&registration, "register_token(")
        .trim_start_matches("HoldWriter::OnAccessRepairs,")
        .trim()
        .trim_end_matches(".clone()")
        .to_owned();
    let (fence_line, _) = only(MAIN, ".set_hold_fence(");
    let lines: Vec<&str> = MAIN.lines().collect();
    // The statement: back from the call to the line after the previous one
    // that ends a statement or opens a block.
    let start = (0..fence_line)
        .rev()
        .find(|&i| {
            let line = lines[i].trim_end();
            line.ends_with(';') || line.ends_with('{') || line.ends_with('}')
        })
        .map_or(0, |i| i + 1);
    let statement: String = lines[start..=fence_line]
        .iter()
        .filter(|line| !line.trim_start().starts_with("//"))
        .flat_map(|line| line.chars())
        .filter(|c| !c.is_whitespace())
        .collect();
    assert_eq!(
        statement,
        format!("valkey.store_oom_signal().set_hold_fence({registered}.clone());"),
        "the replica's out-of-memory signal must be fenced by the token registered under \
         HoldWriter::OnAccessRepairs (line {})",
        fence_line + 1
    );
    let token_created = first(
        MAIN,
        &format!("let {registered} = CancellationToken::new();"),
    );
    let serving_point = first(MAIN, "hold_serving_point.reach(");
    assert!(
        token_created < fence_line && fence_line < serving_point,
        "the signal is fenced after its token is created and before the serving point"
    );
}

/// The health supervisor's `verdict_stale` input is a receiver of the
/// `verdict_stale` sender in the settings `spawn_hold_reactor` is handed, so
/// readiness goes NOT_SERVING when the reactor reads the verdict as stale.
/// Fails if the field is removed, set to `None`, or fed another receiver.
#[test]
fn main_feeds_readiness_the_reactors_verdict_stale_signal() {
    assert_receiver_of_reactor_signal(
        "HealthSupervisorConfig {",
        "verdict_stale",
        "Some",
        "verdict_stale",
    );
}

/// The scaling source's `held` is a receiver of the `held` sender in the
/// settings `spawn_hold_reactor` is handed, so the `/scaling/v1` snapshot,
/// and with it the autoscaler freeze, follows the reactor's hold entry.
/// Fails if the field is fed any other receiver.
#[test]
fn main_feeds_the_scaling_snapshot_the_reactors_held_signal() {
    assert_receiver_of_reactor_signal("ServiceScalingSource {", "held", "", "held");
}

/// A replica already held at startup runs held mode inside the `is_held()`
/// branch and returns, before it spawns the normal-mode hold poll: held mode
/// is the branch's first statement, and `return;` its last, so nothing of
/// the normal startup runs after held mode ends. Fails if the branch goes
/// back to exiting at once (the restart loop held mode replaced), or falls
/// through into the normal startup.
#[test]
fn main_runs_held_mode_at_startup_before_spawning_the_hold_poll() {
    let lines: Vec<&str> = MAIN.lines().collect();
    let held = first(MAIN, "hold_at_startup.verdict.is_held()");
    let run = first(MAIN, "run_held_mode_then_exit(");
    let poll = first(MAIN, "spawn_hold_poll(");
    assert_eq!(run, held + 1, "held mode must be the held branch's body");
    let ret = (run..poll)
        .find(|&i| lines[i].trim() == "return;")
        .expect("the held branch returns before the normal-mode hold poll");
    assert_eq!(
        lines[ret + 1].trim(),
        "}",
        "`return;` must end the held branch (line {})",
        ret + 1
    );
}

/// A failed startup hold read exits non-zero at once (SDD SR-5.5): `run`
/// makes one read, `evaluate_hold_at_startup` bound straight to
/// `hold_at_startup`, and its error arm's last statement is
/// `std::process::exit(NO_VERDICT_EXIT_CODE)`, a non-zero code. Between the
/// poll settings and the held-mode decision there is no loop, retry or wait,
/// and no other exit, so a replica that cannot read the marker never serves
/// on a fallback verdict and never retries in place; the kubelet restarts it.
/// The read's own bound and its error classes are `hold_poll_test.rs`'s
/// (`*_startup_fails`).
#[test]
fn main_exits_non_zero_at_once_when_the_startup_hold_read_fails() {
    let lines: Vec<&str> = MAIN.lines().collect();
    let settings = first(
        MAIN,
        "let hold_poll_settings = HoldPollSettings::from_config(",
    );
    let decision = first(MAIN, "hold_at_startup.verdict.is_held()");
    assert!(
        all(MAIN, "evaluate_hold_at_startup_").is_empty(),
        "main.rs calls only the single-read `evaluate_hold_at_startup`, no variant of it"
    );
    let (call, call_line) = only(MAIN, "evaluate_hold_at_startup(");
    assert!(
        call_line
            .trim_start()
            .starts_with("let hold_at_startup = evaluate_hold_at_startup("),
        "the startup hold verdict is one single-read call bound directly (line {}): {call_line}",
        call + 1
    );
    assert!(settings < call && call < decision);

    let region: Vec<(usize, &str)> = (settings..decision)
        .map(|i| (i, lines[i]))
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
        .collect();
    for (idx, line) in &region {
        for word in line.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            assert!(
                !["loop", "while", "for", "continue", "sleep", "retry", "interval"].contains(&word),
                "line {} between the startup read and the held-mode decision retries or waits \
                 (`{word}`): a failed read must exit at once: {line}",
                idx + 1
            );
        }
    }
    let exits: Vec<&(usize, &str)> = region
        .iter()
        .filter(|(_, line)| line.contains("process::exit("))
        .collect();
    assert_eq!(
        exits.len(),
        1,
        "one exit between the startup read and the decision: {exits:?}"
    );
    let (exit, exit_line) = *exits[0];
    assert_eq!(
        exit_line.trim(),
        "std::process::exit(NO_VERDICT_EXIT_CODE);",
        "a failed startup hold read exits NO_VERDICT_EXIT_CODE (line {})",
        exit + 1
    );
    assert_ne!(
        live_edit::grpc::hold::NO_VERDICT_EXIT_CODE,
        0,
        "the no-verdict exit is non-zero, so the restart reads as a failure"
    );
    let arm = (call..exit)
        .find(|&i| lines[i].contains(".unwrap_or_else(|e| {"))
        .expect("the read's error arm follows the call");
    assert!(
        lines[exit + 1].trim() == "});" && exit + 2 == decision,
        "the exit ends the error arm (opened at line {}), and the held-mode decision follows it",
        arm + 1
    );
}

/// The end of `fn_line`'s body in `src`: the first line after it that is a
/// closing brace in column 0 (rustfmt's top-level `fn` close).
fn fn_end(src: &str, fn_line: usize) -> usize {
    src.lines()
        .enumerate()
        .skip(fn_line + 1)
        .find(|(_, line)| *line == "}")
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| panic!("the fn at line {} closes", fn_line + 1))
}

/// In normal mode the hold poll is the first background task `run` spawns
/// (SDD SR-5.1a; PM r2 N1), the two servers that answer before it and the
/// startup signal watcher aside: the only spawn calls in `run` before
/// `spawn_hold_poll(` are the `tokio::spawn` of the bootstrap gRPC server, of
/// the metrics server and of the watcher (`StartupClaim::watch`), in that
/// order, all before the held-mode decision. So the poll runs before any
/// store- or backup-writing task exists, a hold that starts during the rest
/// of startup reaches the replica, and no such task runs in held mode, which
/// returns at the decision. Fails if a spawn is added, or moved, anywhere
/// above the poll — before the decision as well as between it and the poll.
#[test]
fn main_spawns_the_hold_poll_before_every_background_task_but_the_two_servers_and_the_watcher() {
    let run = first(MAIN, "async fn run(");
    let run_end = fn_end(MAIN, run);
    let decision = first(MAIN, "hold_at_startup.verdict.is_held()");
    let poll = first(MAIN, "spawn_hold_poll(");
    assert!(
        run < decision && decision < poll && poll < run_end,
        "the decision and the poll are in run(), in that order"
    );
    let in_run: Vec<(String, usize)> = spawn_calls(MAIN)
        .into_iter()
        .filter(|(name, line)| *line > run && *line < run_end && name != "spawn_hold_poll")
        .collect();
    assert!(
        in_run
            .iter()
            .any(|(name, line)| name == "spawn_store_memory_refresh" && *line > poll),
        "the scan sees run()'s background spawns after the poll: {in_run:?}"
    );
    let before_poll: Vec<&(String, usize)> =
        in_run.iter().filter(|(_, line)| *line < poll).collect();
    // The bootstrap server is the one spawn inside the block bound to
    // `bootstrap_server`, from its opener to the first later line that closes
    // it at the opener's indentation (`};`).
    let lines: Vec<&str> = MAIN.lines().collect();
    let (bootstrap, opener) = only(MAIN, "let bootstrap_server = {");
    let close = format!("{}}};", &opener[..opener.len() - opener.trim_start().len()]);
    let bootstrap_end = (bootstrap + 1..lines.len())
        .find(|&i| lines[i] == close)
        .expect("the bootstrap_server block closes");
    let in_block: Vec<usize> = before_poll
        .iter()
        .filter(|(_, line)| (bootstrap..bootstrap_end).contains(line))
        .map(|(_, line)| *line)
        .collect();
    assert_eq!(
        in_block.len(),
        1,
        "one spawn call in the bootstrap_server block (lines {}–{}): {before_poll:?}",
        bootstrap + 1,
        bootstrap_end + 1
    );
    let metrics = first(MAIN, "let metrics_task = tokio::spawn(");
    let (watcher, _) = only(MAIN, "tokio::spawn(startup_watcher)");
    assert_eq!(
        before_poll
            .iter()
            .map(|(name, line)| (name.as_str(), *line))
            .collect::<Vec<_>>(),
        [
            ("spawn", in_block[0]),
            ("spawn", metrics),
            ("spawn", watcher)
        ],
        "only the bootstrap server's, the metrics server's and the startup signal watcher's \
         `tokio::spawn` come before the hold poll (line {}); anything else spawned there would \
         run in held mode too, or before the poll could stop it",
        poll + 1
    );
    assert!(
        watcher < decision,
        "both servers and the watcher are spawned before the held-mode decision"
    );
}

/// Held mode failing — its server failed, or its hold reactor panicked or was
/// cancelled, so nothing acts on the verdict any more — ends the process
/// non-zero (SDD SR-5.6): in `run_held_mode_then_exit`, after
/// `run_held_mode(`, the `served.is_err()` arm calls `std::process::exit`
/// with a non-zero code, and that is the function's only exit. Fails if the
/// arm is removed or exits 0, so a failed held mode would read as a clean
/// stop and the pod would not be restarted.
#[test]
fn main_exits_non_zero_when_held_mode_fails() {
    let lines: Vec<&str> = MAIN.lines().collect();
    let start = first(MAIN, "async fn run_held_mode_then_exit(");
    let end = fn_end(MAIN, start);
    let body: Vec<(usize, &str)> = (start + 1..end)
        .map(|i| (i, lines[i]))
        .filter(|(_, l)| !l.trim_start().starts_with("//"))
        .collect();
    let held = body
        .iter()
        .find(|(_, l)| l.contains("run_held_mode("))
        .map(|(i, l)| {
            assert!(
                l.trim_start().starts_with("let served = run_held_mode("),
                "held mode's result is bound to `served` (line {}): {l}",
                i + 1
            );
            *i
        })
        .expect("run_held_mode_then_exit runs held mode");
    let arm = body
        .iter()
        .find(|(i, l)| *i > held && l.trim() == "if served.is_err() {")
        .map(|(i, _)| *i)
        .expect("after held mode, an `if served.is_err()` arm");
    let indent = lines[arm].len() - lines[arm].trim_start().len();
    let arm_end = (arm + 1..end)
        .find(|&i| lines[i].trim() == "}" && lines[i].len() - lines[i].trim_start().len() == indent)
        .expect("the arm closes");
    let exits: Vec<&(usize, &str)> = body
        .iter()
        .filter(|(_, l)| l.contains("process::exit("))
        .collect();
    assert_eq!(
        exits.len(),
        1,
        "run_held_mode_then_exit has one process exit: {exits:?}"
    );
    let (exit, exit_line) = *exits[0];
    assert!(
        arm < exit && exit < arm_end,
        "the exit (line {}) is inside the `served.is_err()` arm (lines {}–{})",
        exit + 1,
        arm + 1,
        arm_end + 1
    );
    let code: i32 = call_argument(exit_line, "process::exit(")
        .parse()
        .unwrap_or_else(|e| panic!("the exit code is a literal ({e}): {exit_line}"));
    assert_ne!(code, 0, "a failed held mode exits non-zero: {exit_line}");
}

/// The process's one SIGTERM and SIGINT listener registers both handlers when
/// it is installed — before the socket handoff of every mode, held mode's
/// included — not when it is first polled (SDD §5.5, SR-5.6): in
/// `observability::startup`, `ShutdownSignal::install` is a plain `fn`, not
/// an `async fn`, whose body registers `signal(SignalKind::terminate())` and
/// `signal(SignalKind::interrupt())` and stores both, and `recv` only awaits
/// the stored listeners. And `main` installs it as its first step, before
/// the logging settings load and before its first `.await`, and registers no
/// other handler anywhere. As an `async fn`, with a registration moved into
/// `recv`, or with a lazy `ctrl_c()`, a signal to PID 1 before the first
/// poll would be dropped and the pod would wait for SIGKILL.
#[test]
fn the_shutdown_signal_registers_both_handlers_when_installed_not_when_polled() {
    let lines: Vec<&str> = STARTUP.lines().collect();
    let (def, def_line) = only(STARTUP, "fn install(");
    assert!(
        def_line.trim_start().starts_with("pub fn install()")
            && def_line.contains("-> io::Result<Self>"),
        "a plain fn, not an `async fn` (line {}): {def_line}",
        def + 1
    );
    let body_end = (def + 1..lines.len())
        .find(|&i| lines[i] == "    }")
        .expect("install's body closes");
    for kind in ["SignalKind::terminate()", "SignalKind::interrupt()"] {
        let at: Vec<usize> = all(STARTUP, kind);
        assert_eq!(
            at.len(),
            1,
            "`{kind}` is registered exactly once in observability::startup: {at:?}"
        );
        assert!(
            (def..body_end).contains(&at[0]) && lines[at[0]].contains("signal("),
            "`signal({kind})` is registered inside `install` (lines {}–{}), not on first poll",
            def + 1,
            body_end + 1
        );
    }
    let (recv, _) = only(STARTUP, "pub async fn recv(&mut self)");
    let recv_end = (recv + 1..lines.len())
        .find(|&i| lines[i] == "    }")
        .expect("recv's body closes");
    for listener in ["self.terminate.recv()", "self.interrupt.recv()"] {
        assert!(
            (recv..recv_end).any(|i| lines[i].contains(listener)),
            "`recv` awaits the listener `install` stored: `{listener}`"
        );
    }
    assert!(
        all(STARTUP, "ctrl_c").is_empty(),
        "no lazily registered `ctrl_c()` listener"
    );

    let main = first(MAIN, "async fn main()");
    let install = first(MAIN, "ShutdownSignal::install()");
    let settings = first(MAIN, "TracingSettings::from_env()");
    let first_await = code_lines(MAIN)
        .find(|(idx, line)| *idx > main && line.contains(".await"))
        .map(|(idx, _)| idx)
        .expect("main awaits");
    assert!(
        main < install && install < settings && install < first_await,
        "main installs the listener (line {}) before the logging settings load (line {}) and \
         before its first await (line {})",
        install + 1,
        settings + 1,
        first_await + 1
    );
    for other in ["SignalKind", "ctrl_c", "signal::unix"] {
        assert!(
            all(MAIN, other).is_empty(),
            "main.rs registers no signal handler of its own (`{other}`): the one listener is \
             `ShutdownSignal`"
        );
    }
}

/// The serving paths claim the startup parts — the bootstrap server and the
/// `/metrics` server — and the shutdown request before they take the socket
/// over (SDD §5.5, SR-5.6), so a signal either stopped startup first, in
/// which case they never serve, or reaches them through the request: in
/// `run_held_mode_then_exit`, `claim_serving()`, then the handoff, then held
/// mode; in `run`, `claim_serving()`, then the handoff, then the hold's
/// serving point, then readiness.
#[test]
fn main_claims_the_startup_parts_before_the_handoff() {
    let lines: Vec<&str> = MAIN.lines().collect();
    let start = first(MAIN, "async fn run_held_mode_then_exit(");
    let end = fn_end(MAIN, start);
    let in_body = |needle: &str| {
        (start..end)
            .find(|&i| {
                let line = lines[i];
                !line.trim_start().starts_with("//") && line.contains(needle)
            })
            .unwrap_or_else(|| panic!("run_held_mode_then_exit calls no `{needle}`"))
    };
    let claim = in_body("claim_serving()");
    let handoff = in_body("hand_over_from_bootstrap_server(");
    let held = in_body("run_held_mode(");
    assert!(
        claim < handoff && handoff < held,
        "held mode: the claim (line {}), then the handoff (line {}), then held mode (line {})",
        claim + 1,
        handoff + 1,
        held + 1
    );

    let run = first(MAIN, "async fn run(");
    let run_end = fn_end(MAIN, run);
    let in_run = |needle: &str| -> Vec<usize> {
        all(MAIN, needle)
            .into_iter()
            .filter(|&i| run < i && i < run_end)
            .collect()
    };
    let claims = in_run("claim_serving()");
    assert_eq!(claims.len(), 1, "run claims once: {claims:?}");
    // Two handoffs in run: the startup stop's, inside the closure handed to
    // the watcher, which runs only when a signal claimed startup first, and
    // the serving path's, after its claim.
    let handoffs = in_run("hand_over_from_bootstrap_server(");
    assert_eq!(
        handoffs.len(),
        2,
        "run hands the socket over twice: {handoffs:?}"
    );
    let (stop, _) = only(MAIN, "let startup_stop = {");
    let (watcher, _) = only(MAIN, "tokio::spawn(startup_watcher)");
    assert!(
        stop < handoffs[0] && handoffs[0] < watcher,
        "the first handoff (line {}) is the startup stop's (lines {}–{})",
        handoffs[0] + 1,
        stop + 1,
        watcher + 1
    );
    let serving_handoff = handoffs[1];
    let serving_point = first(MAIN, "hold_serving_point.reach(");
    let readiness = first(MAIN, "api_serving_tx.send(true)");
    assert!(
        claims[0] < serving_handoff && serving_handoff < serving_point && serving_point < readiness,
        "run: the claim (line {}), then the handoff (line {}), then the serving point (line {}), \
         then readiness (line {})",
        claims[0] + 1,
        serving_handoff + 1,
        serving_point + 1,
        readiness + 1
    );
}

/// A stop requested by the time the serving path's bootstrap handoff ends
/// skips the hold's serving point and readiness (SDD §5.5): in `run`, after
/// `claim_serving()`, `stop_requested` is bound to a clone of the claimed
/// shutdown request; the serving point (`hold_serving_point.reach(`), the
/// readiness flip (`api_serving_tx.send(true)`) and the INFO `gRPC server
/// serving full API` sit inside one `if !stop_requested.is_cancelled()`
/// branch that opens after the handoff, and its `else` logs the drain-only
/// INFO (`shutdown requested before serving; …`) instead. The handoff stays
/// outside both, so a stopping replica still hands the socket over and
/// drains. Fails if the guard or its `else` is dropped, if any of those
/// lines leaves its branch or moves to the other, or if the handoff moves
/// into the guard.
#[test]
fn main_skips_the_serving_point_and_readiness_when_a_stop_is_requested_by_the_handoff() {
    let lines: Vec<&str> = MAIN.lines().collect();
    let run = first(MAIN, "async fn run(");
    let run_end = fn_end(MAIN, run);
    let in_run = |needle: &str| -> Vec<usize> {
        all(MAIN, needle)
            .into_iter()
            .filter(|&i| run < i && i < run_end)
            .collect()
    };
    let claims = in_run("claim_serving()");
    assert_eq!(claims.len(), 1, "run claims once: {claims:?}");
    let claim = claims[0];
    let (binding, _) = only(MAIN, "let stop_requested = shutdown_request.clone();");
    let signals = in_run("let shutdown_signal = async move");
    assert_eq!(
        signals.len(),
        1,
        "run builds one shutdown signal: {signals:?}"
    );
    let moved = signals[0];
    assert!(
        claim < binding && binding < moved,
        "run binds `stop_requested` (line {}) after its claim (line {}) and before the shutdown \
         signal takes the request (line {})",
        binding + 1,
        claim + 1,
        moved + 1
    );

    // The guard's two branches, bounded by the lines that close each at the
    // guard's indentation: the if-branch ends at `} else {`, the else-branch
    // at the `}` after it.
    let (guard, guard_line) = only(MAIN, "if !stop_requested.is_cancelled() {");
    let indent = guard_line.len() - guard_line.trim_start().len();
    let closing_after = |from: usize| {
        (from + 1..run_end)
            .find(|&i| {
                let line = lines[i];
                line.trim_start().starts_with('}') && line.len() - line.trim_start().len() == indent
            })
            .unwrap_or_else(|| panic!("the branch opened at line {} closes in run", from + 1))
    };
    let else_line = closing_after(guard);
    assert_eq!(
        lines[else_line].trim(),
        "} else {",
        "the guard's if-branch (line {}) is followed by an else-branch (line {})",
        guard + 1,
        else_line + 1
    );
    let guard_end = closing_after(else_line);
    assert_eq!(
        lines[guard_end].trim(),
        "}",
        "the else-branch closes the guard (line {})",
        guard_end + 1
    );
    let handoffs = in_run("hand_over_from_bootstrap_server(");
    let serving_handoff = *handoffs
        .last()
        .expect("run hands the socket over on its serving path");
    assert!(
        claim < serving_handoff && serving_handoff < guard,
        "the serving path's handoff (line {}) comes after the claim (line {}) and before the \
         branch opens (line {})",
        serving_handoff + 1,
        claim + 1,
        guard + 1
    );
    for handoff in &handoffs {
        assert!(
            !(guard..guard_end).contains(handoff),
            "the handoff at line {} is outside the branch (lines {}–{}): a stopping replica \
             still hands the socket over",
            handoff + 1,
            guard + 1,
            guard_end + 1
        );
    }
    for call in [
        "hold_serving_point.reach(",
        "api_serving_tx.send(true)",
        "\"gRPC server serving full API\"",
    ] {
        let (at, _) = only(MAIN, call);
        assert!(
            guard < at && at < else_line,
            "`{call}` (line {}) sits inside the `if !stop_requested.is_cancelled()` branch (lines \
             {}–{})",
            at + 1,
            guard + 1,
            else_line + 1
        );
    }
    let (draining, _) = only(MAIN, "\"shutdown requested before serving;");
    assert!(
        else_line < draining && draining < guard_end,
        "the drain-only INFO (line {}) sits in the guard's else-branch (lines {}–{})",
        draining + 1,
        else_line + 1,
        guard_end + 1
    );
}

/// Held mode (`grpc::hold::run_held_mode`) spawns exactly the tasks its
/// contract lists — the hold poll, its own reactor and its own health
/// supervisor — and its source names no backup (checkpoint) store, no
/// external-value mirror or repair, and none of the inventory writers, so it
/// can write nothing to the backup store (SDD SR-5.1, SR-5.3). A spawn added
/// there, or a backup-store handle threaded in, fails here.
#[test]
fn held_mode_spawns_only_its_listed_tasks_and_holds_no_backup_store() {
    let mut found: BTreeMap<String, usize> = BTreeMap::new();
    for (name, _) in spawn_calls(HELD_MODE) {
        *found.entry(name).or_default() += 1;
    }
    let expected: BTreeMap<String, usize> = [
        "spawn_hold_poll",
        "spawn_held_mode_reactor",
        "spawn_health_supervisor",
    ]
    .into_iter()
    .map(|name| (name.to_owned(), 1))
    .collect();
    assert_eq!(
        found, expected,
        "held mode spawns its hold poll, its reactor and its health supervisor, once each, and \
         nothing else"
    );
    for (idx, line) in code_lines(HELD_MODE) {
        let lower = line.to_ascii_lowercase();
        for forbidden in ["checkpoint", "ext_mirror", "ext_backup", "hold_writer"] {
            assert!(
                !lower.contains(forbidden),
                "held_mode.rs line {} names `{forbidden}`: held mode is handed no backup store \
                 and runs no writer: {line}",
                idx + 1
            );
        }
    }
}

/// The hold poll and the hold reactor stop first on every shutdown, in every
/// phase: their token is a child of the process shutdown request
/// (`let hold_poll_shutdown = startup_claim.request();`), so the request
/// cancels it during startup as after the serving point, and the first
/// statement after the serving path's shutdown signal awaits the request
/// cancels it again, ahead of everything else the shutdown does: from there
/// the shutdown runs alone, held or not.
#[test]
fn main_stops_the_hold_poll_first_on_the_shutdown_request() {
    only(MAIN, "let hold_poll_shutdown = startup_claim.request();");
    let lines: Vec<&str> = MAIN.lines().collect();
    let signal = first(MAIN, "let shutdown_signal = async move");
    let statements: Vec<&str> = (signal + 1..lines.len())
        .map(|i| lines[i].trim())
        .filter(|t| !t.is_empty() && !t.starts_with("//"))
        .take(2)
        .collect();
    assert_eq!(
        statements,
        [
            "shutdown_request.cancelled().await;",
            "hold_poll_shutdown.cancel();"
        ],
        "the shutdown signal awaits the process shutdown request, then stops the hold poll and \
         reactor before anything else"
    );
}

/// Shutdown keeps its close-and-wait drains: each tracker's abort comes after
/// its drain, so tracked work finishes first and only what outlives the drain
/// is aborted. The layer store's drain and abort are the closure `main.rs`
/// hands the post-server sequence; the session-checkpoint tracker's are in the
/// sequence itself (`grpc::transport::run_post_server_sequence`).
#[test]
fn the_shutdown_aborts_each_tracker_only_after_its_drain() {
    let cleanup_drain = first(MAIN, ".wait_for_cleanup(");
    let cleanup_abort = first(MAIN, ".abort_cleanup_tasks()");
    assert!(
        cleanup_drain < cleanup_abort,
        "the layer store's cleanups must be aborted after their drain"
    );
    let session_drain = first(TRANSPORT, "session_checkpoints.wait()");
    let session_abort = first(TRANSPORT, "session_checkpoint_abort.cancel()");
    assert!(
        session_drain < session_abort,
        "the session-checkpoint tracker must be aborted after its drain"
    );
}

/// The serving point is reached before readiness opens, so the replica never
/// reports itself serving while a hold would still make it exit instead of
/// entering the hold.
#[test]
fn main_reaches_the_hold_serving_point_before_readiness_opens() {
    let serving_point = first(MAIN, "hold_serving_point.reach(");
    let readiness = first(MAIN, "api_serving_tx.send(true)");
    assert!(serving_point < readiness);
}

/// The shutdown join (`grpc::transport::run_post_server_sequence`) reports a
/// task the hold or a stale verdict aborted as aborted, at INFO, before the
/// arm that reports a failure. Both entries abort the same set, so the line
/// names both causes rather than only the hold's.
#[test]
fn the_shutdown_join_reports_a_task_the_hold_aborted_as_aborted_not_failed() {
    let aborted_arm = first(
        TRANSPORT,
        "join_err.is_cancelled() && report.hold_writers.is_aborted()",
    );
    let line = first(
        TRANSPORT,
        "\"background task was aborted earlier, at hold or stale-verdict entry\"",
    );
    let failure_arm = first(TRANSPORT, "\"background task failed during shutdown\"");
    assert!(aborted_arm < line && line < failure_arm);
    assert!(
        TRANSPORT.lines().collect::<Vec<_>>()[aborted_arm + 1..line]
            .iter()
            .any(|l| l.contains("tracing::info!")),
        "the hold-aborted arm logs at INFO"
    );
}

/// Every tracked cleanup of the layer store goes through `spawn_cleanup`,
/// which races it against the abort token: `cleanup_tracker.spawn(` appears
/// only inside `spawn_cleanup`, and the three tracked cleanups — the
/// `CreateLayer` commit finisher, the overwrite cleanup and the torn-commit
/// retract — call it.
#[test]
fn layer_store_spawns_every_tracked_cleanup_through_spawn_cleanup() {
    let lines: Vec<&str> = LAYER_STORE.lines().collect();
    let spawn_cleanup = first(LAYER_STORE, "fn spawn_cleanup<");
    let body_end = (spawn_cleanup + 1..lines.len())
        .find(|&i| lines[i].starts_with("    }"))
        .expect("spawn_cleanup's body closes");
    let raw_spawns = all(LAYER_STORE, "cleanup_tracker.spawn(");
    assert!(
        !raw_spawns.is_empty(),
        "spawn_cleanup spawns on the tracker"
    );
    for line in &raw_spawns {
        assert!(
            (spawn_cleanup..body_end).contains(line),
            "line {} spawns on the cleanup tracker outside spawn_cleanup, so the abort token \
             cannot end it",
            line + 1
        );
    }
    assert_eq!(
        all(LAYER_STORE, "self.spawn_cleanup(").len(),
        3,
        "the finisher, the overwrite cleanup and the torn-commit retract"
    );
}
