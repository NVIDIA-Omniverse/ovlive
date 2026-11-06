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

//! b.ntm (T13a) — a source guard over `SubscribeReadLayer`'s waits: every
//! `select!` that races the client's disconnect (`tx.closed()`) on the
//! subscription's response queue also races the subscription's live-delivery
//! hold ending (`hold.ended()`, or `hold_ended_or_never` where the hold is
//! optional).
//!
//! The hold's ending is how the live-delivery budget ends the subscription
//! holding the most undelivered updates (`delivery_budget`), and how a
//! catch-up waiting for catch-up payload memory ends a subscription that
//! stopped reading its catch-up pages (`slow_reader_deadline`; SDD §12.3,
//! §12.6). Both endings fire the hold's token and nothing else: a wait that
//! races only the disconnect keeps waiting after it, holding the memory the
//! ending was meant to release, until whatever it waits for comes. A new wait
//! written without the arm passes every behavioural test that does not end a
//! subscription during that wait.
//!
//! **What counts.** The production code of every file under `src/` that uses
//! the response queue's sender (`ResponseSender`), read through
//! `refusal_and_shutdown_guards_test::production_code` (comments, string
//! literal contents and test-only items removed). Each `select!` invocation is
//! taken whole, by brace matching from its `{`; one whose body has a
//! `.closed()` arm must have a hold arm too. Every `.closed()` in those files
//! must sit inside a `select!`, so a race written with another combinator
//! fails here rather than going unscanned.
//!
//! **What it cannot see:** that the hold arm maps to the hold's own ending
//! (`end_on_hold`), or that a wait which does not race the disconnect at all
//! needs the arm; those are the behavioural tests'
//! (`grpc::subscribe::tests::send_racing_endings_*`,
//! `grpc::delivery_budget_test`).
//!
//! Static: reads source files, needs no Valkey and no feature.

// Guard tests: scripts/guard-tests.sh runs this whole module by name; rename it there too.

use std::path::{Path, PathBuf};

use super::refusal_and_shutdown_guards_test::production_code;
use super::txn_checkout_test::rust_sources_under;

/// The response queue's sender: a file that names it can race its `closed()`.
const SENDER_TYPE: &str = "ResponseSender";

/// The disconnect arm's call.
const CLOSED_CALL: &str = ".closed()";

/// The hold-ending arms, either of which satisfies the rule.
const HOLD_ARMS: [&str; 2] = ["hold.ended()", "hold_ended_or_never("];

/// The `select!` invocations racing the disconnect today, at least: the idle
/// wait in steady state, the catch-up slot and read-permit waits, the
/// subscription's ending race, the stream-reader reservation, and the send
/// race. Non-vacuity: a scan that found fewer is reading the wrong code.
const AT_LEAST_RACES: usize = 6;

/// The file whose waits the rule exists for.
const SUBSCRIBE_FILE: &str = "src/grpc/subscribe.rs";

/// One `select!` in `code` (flattened production code): its byte range, from
/// the macro's name to its closing brace.
fn select_bodies(code: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(found) = code[from..].find("select!") {
        let start = from + found;
        let Some(open) = code[start..].find(['{', '(']).map(|o| start + o) else {
            break;
        };
        let (opener, closer) = if code.as_bytes()[open] == b'{' {
            ('{', '}')
        } else {
            ('(', ')')
        };
        let mut depth = 0i32;
        let mut end = code.len();
        for (i, c) in code[open..].char_indices() {
            if c == opener {
                depth += 1;
            } else if c == closer {
                depth -= 1;
                if depth == 0 {
                    end = open + i + 1;
                    break;
                }
            }
        }
        out.push((start, end));
        from = end.max(start + 1);
    }
    out
}

/// What the scan of one file's production code found: the `select!`s racing
/// the disconnect, those of them with no hold arm, and the `.closed()` calls
/// outside any `select!`, each by its first line of code.
#[derive(Debug, Default, PartialEq, Eq)]
struct HoldScan {
    races: usize,
    missing_hold: Vec<String>,
    outside_select: Vec<String>,
}

fn scan(source: &str) -> HoldScan {
    let code: String = production_code(source)
        .lines
        .iter()
        .map(|(_, line)| line.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let bodies = select_bodies(&code);
    let mut out = HoldScan::default();
    let first_line = |at: usize| code[at..].lines().next().unwrap_or("").to_owned();
    for &(start, end) in &bodies {
        let body = &code[start..end];
        if body.contains(CLOSED_CALL) {
            out.races += 1;
            if !HOLD_ARMS.iter().any(|arm| body.contains(arm)) {
                out.missing_hold
                    .push(first_line(code[..start].rfind('\n').map_or(0, |n| n + 1)));
            }
        }
    }
    let mut from = 0;
    while let Some(found) = code[from..].find(CLOSED_CALL) {
        let at = from + found;
        if !bodies.iter().any(|&(start, end)| start <= at && at < end) {
            out.outside_select
                .push(first_line(code[..at].rfind('\n').map_or(0, |n| n + 1)));
        }
        from = at + CLOSED_CALL.len();
    }
    out
}

/// Every production file under `src/` that names the response queue's sender,
/// with its source.
fn files_using_the_response_queue() -> Vec<(PathBuf, String)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut walked = Vec::new();
    rust_sources_under(&root, &root.join("src"), &mut walked);
    walked.sort();
    walked
        .into_iter()
        .filter_map(|path| {
            let source = std::fs::read_to_string(root.join(&path))
                .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", path.display()));
            let uses = production_code(&source)
                .lines
                .iter()
                .any(|(_, line)| line.contains(SENDER_TYPE));
            uses.then_some((path, source))
        })
        .collect()
}

/// T13a: in every production file that uses the response queue, every
/// `select!` that races the client's disconnect also races the
/// subscription's hold ending, and no `.closed()` races outside a `select!`.
#[test]
fn every_select_racing_the_disconnect_also_races_the_subscription_holds_ending() {
    let files = files_using_the_response_queue();
    assert!(
        files
            .iter()
            .any(|(path, _)| path == Path::new(SUBSCRIBE_FILE)),
        "the scan did not reach {SUBSCRIBE_FILE}: it found {:?}",
        files.iter().map(|(p, _)| p).collect::<Vec<_>>()
    );
    let mut races = 0;
    let mut failures = Vec::new();
    for (path, source) in &files {
        let found = scan(source);
        races += found.races;
        for site in found.missing_hold {
            failures.push(format!(
                "{}: a select! racing `tx.closed()` has no hold-ending arm (one of {HOLD_ARMS:?}): \
                 `{site}`",
                path.display()
            ));
        }
        for site in found.outside_select {
            failures.push(format!(
                "{}: `.closed()` raced outside a select!, which this guard cannot check: `{site}`",
                path.display()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "SubscribeReadLayer's waits (b.ntm, SDD §12.6): every wait that races the client's \
         disconnect must race the subscription's live-delivery hold ending too, or the \
         live-delivery budget's and the stalled-catch-up ending cannot end it there:\n{}",
        failures.join("\n")
    );
    assert!(
        races >= AT_LEAST_RACES,
        "only {races} select!s racing the disconnect were found, fewer than the \
         {AT_LEAST_RACES} the subscription has: the scan is not reading its waits"
    );
}

/// The scanner on fixtures: a `select!` racing the disconnect with the hold
/// arm passes, in either spelling; one without it is reported; a disconnect
/// raced outside a `select!` is reported; a `.closed()` in a comment, a
/// string or a test module is not code and is not reported.
#[test]
fn the_hold_scanner_reports_a_missing_hold_arm_and_a_race_outside_a_select() {
    let with_hold = r#"
async fn ok(ctx: &Ctx) {
    tokio::select! {
        biased;
        () = ctx.tx.closed() => {}
        () = ctx.hold.ended() => {}
    }
    tokio::select! {
        () = tx.closed() => {}
        () = hold_ended_or_never(hold) => {}
    }
}
"#;
    assert_eq!(
        scan(with_hold),
        HoldScan {
            races: 2,
            ..HoldScan::default()
        }
    );

    let without = r#"
async fn missing(ctx: &Ctx) {
    tokio::select! {
        () = ctx.shutdown.cancelled() => {}
        () = ctx.tx.closed() => {}
    }
}
"#;
    let found = scan(without);
    assert_eq!((found.races, found.missing_hold.len()), (1, 1), "{found:?}");

    let outside = r#"
async fn outside(ctx: &Ctx) {
    futures::future::select(ctx.tx.closed(), other).await;
}
"#;
    assert_eq!(scan(outside).outside_select.len(), 1);

    let not_code = r#"
// tokio::select! { () = tx.closed() => {} }
fn f() { let _ = "tx.closed()"; }
#[cfg(test)]
mod tests {
    async fn t() { tokio::select! { () = tx.closed() => {} } }
}
"#;
    assert_eq!(scan(not_code), HoldScan::default());
}
