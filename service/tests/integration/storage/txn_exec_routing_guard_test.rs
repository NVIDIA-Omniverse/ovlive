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

//! b.cyu — the guard behind SDD §15.3's rule: **a fred `Transaction`'s `EXEC`
//! is sent only through `ValkeyClient::exec_txn`.**
//!
//! The `txn_pool` allows a fred `Transaction` no redirect, and fred's
//! transaction router fails a `MOVED` / `ASK` past that zero budget without
//! re-syncing the member's routing. Each pool member is its own client with
//! its own routing table, so a member that met a redirect would keep sending
//! the moved slot's transactions to the old owner. `exec_txn` runs the
//! transaction's `EXEC` and, on an error, re-syncs the routing of the checkout
//! it was given before returning the error. Routing every `EXEC` through it
//! puts the re-sync on the error path by construction; this guard pins that
//! no `EXEC` goes around it. That the re-sync runs on the transaction's *own*
//! checkout rests on each caller passing the checkout it called `.multi()`
//! on — `exec_txn`'s documented precondition, which the helper cannot check
//! (a fred `Transaction` exposes no handle on its client). This guard checks
//! it statically instead (the last bullet below).
//!
//! - **No raw `EXEC`.** No production code under `src/` (comments stripped,
//!   `#[cfg(test)] mod tests` cut) calls a fred `Transaction`'s `exec`
//!   outside `ValkeyClient::exec_txn` in `src/storage/client.rs`. Every form
//!   is caught: `.exec(`, the turbofish `.exec::<`, a chained `.exec` on its
//!   own line, and the path form `Transaction::exec(`.
//! - **Telling a `WatchedTxn` apart.** `WatchedTxn::exec` (the `WATCH`ed
//!   path, which routes its `EXEC` as a pinned keyless command, SDD §7.5) is
//!   not a fred `Transaction`, and is not flagged. The scan treats an `exec`
//!   call as a `WatchedTxn`'s only when it can see the type: the path
//!   `WatchedTxn::exec(`, or a plain local receiver whose nearest `let` in the
//!   same `fn` binds it from one of `WatchedTxn`'s constructors,
//!   `WatchedTxn::open(` or `WatchedTxn::open_with_timeout(` (directly or
//!   after `match `), or annotates it `: WatchedTxn`; or, only when
//!   the `fn` has no `let` of that name, a parameter declared exactly
//!   `<name>: WatchedTxn` (the name preceded by `(` or `,`, the type followed
//!   by no identifier character). Every other receiver — a fred `Transaction`
//!   from `.multi()` (including one shadowing a `WatchedTxn` parameter), a
//!   chained expression, a field, a local the scan cannot type — counts as a
//!   raw `EXEC`, so the guard fails closed. String literals are kept in the
//!   scan, so a `.exec(` spelled inside one fails too; reword the string.
//!   The constructors are matched by exact name, not as an `open` prefix, so
//!   a local bound from any other `WatchedTxn::` function counts as raw; the
//!   guard also pins that each listed constructor is defined in
//!   `impl WatchedTxn` in `src/storage/watched_txn.rs` and returns
//!   `Result<Self, _>`. The `exec` of a local bound from a new constructor is
//!   flagged until the constructor is added to `WATCHED_TXN_CONSTRUCTORS`.
//! - **The exemption is real and re-syncs.** Exactly one raw `EXEC` is
//!   exempt, in `src/storage/client.rs::exec_txn`, and that `fn` calls
//!   `resync_txn_routing_after_redirect(`.
//! - **Landmarks.** The scan sees the four transaction sites as of b.cyu —
//!   `SessionStore::delete`, `LayerStore::write_object` and checkpoint
//!   apply's Step A and Step B — each `fn` sending its transactions through
//!   `exec_txn` at least once per `.multi()`, and it sees and exempts the
//!   `WatchedTxn::exec` sites.
//! - **Each `exec_txn` gets the transaction's own checkout.** For every
//!   `exec_txn` call under `src/` (the helper's definition aside), its
//!   transaction argument is a local whose nearest `let` in the same `fn` is
//!   `let <trx> = <checkout>.multi();`, and its checkout argument, `&` and
//!   whitespace aside, is that same `<checkout>` text, with no plain
//!   `let <checkout> =` / `let <checkout>:` (or `let mut`) between the
//!   transaction's `let` and the call. A call whose transaction the scan
//!   cannot trace to a `.multi()` that way fails too. The check compares
//!   names as text: a checkout name re-bound any other way is not detected —
//!   a pattern (`let (client, _) =`, `let Some(client) = … else`, `if let`, a
//!   `match` arm, `for`), a closure parameter, or an assignment to a `mut`
//!   binding.
//!
//! What it does not do: run anything. On a standalone store the re-sync is a
//! no-op, so its run-time effect is pinned only on a cluster, and only for the
//! sites the cluster-gated `b_cyu_slot_migration_test` reaches —
//! `DeleteSession` (T-C1, T-C2) and the `CreateLayer` materialization (T-C3).
//! For checkpoint apply's Step A and Step B, the re-sync is on their error
//! path by construction (their `EXEC` goes through `exec_txn`), and on their
//! own checkout because this guard checks the precondition statically;
//! neither is pinned by a behavioural test.
//!
//! Static: reads source files, needs no Valkey and no feature.

// Guard tests: this module is pinned in scripts/guard-tests.sh `MODULES`; a
// test added here is a guard, and renaming or removing the module means
// editing that list too.

use std::path::Path;

use super::b_289_tests::call_arguments;
use super::script_migration_guard_test::{enclosing_fn_at, production_sources};

/// Where the one exempt raw `EXEC` lives.
const HELPER_FILE: &str = "src/storage/client.rs";
/// The helper every fred `Transaction`'s `EXEC` goes through.
const HELPER_FN: &str = "exec_txn";
/// What the helper must call on an error.
const RESYNC: &str = "resync_txn_routing_after_redirect(";
/// Where `WatchedTxn` is defined.
const WATCHED_TXN_FILE: &str = "src/storage/watched_txn.rs";
/// The `WatchedTxn` constructors an unannotated `let` may bind an `exec`
/// receiver from, matched by exact name: a new constructor joins the list
/// only once it is shown to return a `WatchedTxn`, which
/// [`watched_txn_constructors_return_a_watched_txn`] pins for each listed one.
const WATCHED_TXN_CONSTRUCTORS: [&str; 2] = ["open", "open_with_timeout"];

/// What an `exec` call is made on.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Receiver {
    /// A plain local or parameter: `txn.exec(…)`.
    Ident(String),
    /// The path form: `Type::exec(…)`, naming `Type`.
    Path(String),
    /// Anything else — a chained call, a field, an index: the scan cannot
    /// type it.
    Other,
}

/// One `exec` call found in production code.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ExecCall {
    /// Byte offset of `exec` in the scanned code.
    at: usize,
    /// The enclosing `fn`'s byte offset and name.
    fn_start: usize,
    fn_name: String,
    receiver: Receiver,
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The identifier ending just before byte `end` of `code` (after skipping
/// whitespace back), with its start; `None` when the byte there is not an
/// identifier character.
fn ident_before(code: &str, end: usize) -> Option<(usize, String)> {
    let bytes = code.as_bytes();
    let mut i = end;
    while i > 0 && bytes[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    let stop = i;
    while i > 0 && is_ident(bytes[i - 1]) {
        i -= 1;
    }
    (i < stop).then(|| (i, code[i..stop].to_owned()))
}

/// Every call of a method or associated function named exactly `exec` in
/// `code`: `.exec(`, `.exec::<…>(`, either with the `.exec` on its own line,
/// and `Path::exec(`. A definition (`fn exec(`) and a longer name
/// (`exec_txn`, `exec_overridden`) are not calls of `exec`.
fn exec_calls(code: &str) -> Vec<ExecCall> {
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    for (at, _) in code.match_indices("exec") {
        let end = at + "exec".len();
        if at > 0 && is_ident(bytes[at - 1]) || bytes.get(end).is_some_and(|b| is_ident(*b)) {
            continue;
        }
        // A call: `(`, or a turbofish `::<`, after optional whitespace.
        let mut i = end;
        while bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
            i += 1;
        }
        if !(bytes.get(i) == Some(&b'(') || code[i..].starts_with("::<")) {
            continue;
        }
        // What precedes it: `.` (a method call) or `::` (the path form).
        let mut j = at;
        while j > 0 && bytes[j - 1].is_ascii_whitespace() {
            j -= 1;
        }
        let receiver = if j > 0 && bytes[j - 1] == b'.' {
            let dot = j - 1;
            match ident_before(code, dot) {
                Some((start, name)) => {
                    let mut k = start;
                    while k > 0 && bytes[k - 1].is_ascii_whitespace() {
                        k -= 1;
                    }
                    // `a.b.exec` / `m::b.exec`: a field or a path, not a
                    // plain local.
                    if k > 0 && matches!(bytes[k - 1], b'.' | b':') {
                        Receiver::Other
                    } else {
                        Receiver::Ident(name)
                    }
                }
                None => Receiver::Other,
            }
        } else if j >= 2 && bytes[j - 2] == b':' && bytes[j - 1] == b':' {
            match ident_before(code, j - 2) {
                Some((_, name)) => Receiver::Path(name),
                None => Receiver::Other,
            }
        } else {
            continue;
        };
        let (fn_start, fn_name) = enclosing_fn_at(code, at);
        out.push(ExecCall {
            at,
            fn_start,
            fn_name,
            receiver,
        });
    }
    out
}

/// Whether `call` is visibly a `WatchedTxn`'s `exec` (see the module doc).
fn is_watched_txn(code: &str, call: &ExecCall) -> bool {
    match &call.receiver {
        Receiver::Path(name) => name == "WatchedTxn",
        Receiver::Other => false,
        Receiver::Ident(name) => {
            let body = &code[call.fn_start..call.at];
            // The nearest `let` decides; a `let` shadows any parameter.
            if let Some((after, annotated)) = nearest_let(body, name) {
                let rest = body[after..].trim_start();
                return if annotated {
                    starts_with_type(rest, "WatchedTxn")
                } else {
                    let rest = rest.strip_prefix("match ").unwrap_or(rest);
                    WATCHED_TXN_CONSTRUCTORS
                        .iter()
                        .any(|ctor| rest.starts_with(&format!("WatchedTxn::{ctor}(")))
                };
            }
            // No `let`: a parameter declared exactly `<name>: WatchedTxn`.
            let signature = &body[..body.find('{').unwrap_or(body.len())];
            let compact: String = signature.chars().filter(|c| !c.is_whitespace()).collect();
            let needle = format!("{name}:");
            compact.match_indices(&needle).any(|(at, _)| {
                at > 0
                    && matches!(compact.as_bytes()[at - 1], b'(' | b',')
                    && starts_with_type(&compact[at + needle.len()..], "WatchedTxn")
            })
        }
    }
}

/// Whether `text` starts with the type name `ty` and no longer identifier.
fn starts_with_type(text: &str, ty: &str) -> bool {
    text.strip_prefix(ty)
        .is_some_and(|tail| !tail.bytes().next().is_some_and(is_ident))
}

/// The nearest `let` of `name` in `body` — `let name =`, `let mut name =`,
/// `let name:` or `let mut name:` — as the byte offset just past it and
/// whether it is annotated (`let name: T`) rather than initialised.
fn nearest_let(body: &str, name: &str) -> Option<(usize, bool)> {
    [
        (format!("let {name} ="), false),
        (format!("let mut {name} ="), false),
        (format!("let {name}:"), true),
        (format!("let mut {name}:"), true),
    ]
    .into_iter()
    .filter_map(|(needle, annotated)| {
        body.rfind(needle.as_str())
            .map(|at| (at + needle.len(), annotated))
    })
    .max_by_key(|(after, _)| *after)
}

/// The raw fred `Transaction` `EXEC`s in `code` (the source of `file`), as
/// `file::fn` labels, split into `(violations, exempt)`: the exempt one is
/// the helper's own, in [`HELPER_FILE`]'s [`HELPER_FN`].
fn raw_execs(file: &Path, code: &str) -> (Vec<String>, Vec<String>) {
    let mut violations = Vec::new();
    let mut exempt = Vec::new();
    for call in exec_calls(code) {
        if is_watched_txn(code, &call) {
            continue;
        }
        let label = format!("{}::{}", file.display(), call.fn_name);
        if file == Path::new(HELPER_FILE) && call.fn_name == HELPER_FN {
            exempt.push(label);
        } else {
            let line = code[..call.at].matches('\n').count() + 1;
            violations.push(format!(
                "{label} (code line {line}, receiver {:?})",
                call.receiver
            ));
        }
    }
    (violations, exempt)
}

/// The `fn`s of `code` that open a fred `Transaction`, as
/// `(name, .multi() count, exec_txn call count)`, by position.
fn transactions_and_helper_calls(code: &str) -> Vec<(String, usize, usize)> {
    let mut per_fn: std::collections::BTreeMap<usize, (String, usize, usize)> =
        std::collections::BTreeMap::new();
    for (at, _) in code.match_indices(".multi()") {
        let (start, name) = enclosing_fn_at(code, at);
        per_fn.entry(start).or_insert((name, 0, 0)).1 += 1;
    }
    for (at, _) in code.match_indices(&format!(".{HELPER_FN}")) {
        let after = &code[at + 1 + HELPER_FN.len()..];
        if !(after.starts_with('(') || after.starts_with("::<")) {
            continue;
        }
        let (start, _) = enclosing_fn_at(code, at);
        if let Some(entry) = per_fn.get_mut(&start) {
            entry.2 += 1;
        }
    }
    per_fn.into_values().collect()
}

/// Every `exec_txn` call in `code` (`.exec_txn(` / `.exec_txn::<`), as its
/// enclosing `fn`'s name and, for that call, `Ok(checkout)` when its checkout
/// argument names the receiver of the `.multi()` its transaction argument
/// was bound from, or `Err(why)` when it does not or the scan cannot trace
/// it (see the module doc).
fn exec_txn_checkouts(code: &str) -> Vec<(String, Result<String, String>)> {
    let mut out = Vec::new();
    for (at, _) in code.match_indices(&format!(".{HELPER_FN}")) {
        let after_name = at + 1 + HELPER_FN.len();
        let after = &code[after_name..];
        if !(after.starts_with('(') || after.starts_with("::<")) {
            continue;
        }
        let (fn_start, fn_name) = enclosing_fn_at(code, at);
        out.push((fn_name, checkout_verdict(code, fn_start, at, after_name)));
    }
    out
}

/// `text` without whitespace.
fn squash(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

/// The verdict on one `exec_txn` call (see [`exec_txn_checkouts`]): the call
/// is at `at` of `code`, in the `fn` starting at `fn_start`, and its name
/// ends at `after_name`.
fn checkout_verdict(
    code: &str,
    fn_start: usize,
    at: usize,
    after_name: usize,
) -> Result<String, String> {
    let args = call_arguments(code, after_name).ok_or("cannot parse the call")?;
    let [checkout, trx, ..] = args.as_slice() else {
        return Err(format!("fewer than two arguments: {args:?}"));
    };
    let checkout = squash(checkout.trim_start_matches('&'));
    let trx = trx.trim();
    if trx.is_empty() || !trx.bytes().all(is_ident) {
        return Err(format!(
            "the transaction argument `{trx}` is not a local bound from `.multi()`"
        ));
    }
    let body = &code[fn_start..at];
    let Some((after_let, false)) = nearest_let(body, trx) else {
        return Err(format!("no `let {trx} = <checkout>.multi();` in the fn"));
    };
    let init = &body[after_let..];
    let init = squash(&init[..init.find(';').unwrap_or(init.len())]);
    let Some(opened_on) = init.strip_suffix(".multi()") else {
        return Err(format!(
            "`{trx}` is bound from `{init}`, not from `<checkout>.multi()`"
        ));
    };
    // The names match as text; a `let` of the checkout's name between the
    // transaction's `let` and the call means the text now names another
    // client.
    if opened_on == checkout {
        if let Some((rebound, _)) = nearest_let(body, &checkout) {
            if rebound > after_let {
                return Err(format!(
                    "`{checkout}` is re-bound with `let` after `{trx}` was opened on it, so the \
                     `{checkout}` given to `{HELPER_FN}` is not the client `{trx}` was opened on"
                ));
            }
        }
        Ok(checkout)
    } else {
        Err(format!(
            "`{trx}` was opened on `{opened_on}`, but `{HELPER_FN}` is given `{checkout}` as its \
             checkout"
        ))
    }
}

/// For each of [`WATCHED_TXN_CONSTRUCTORS`], `Ok(())` when `code` (the source
/// of [`WATCHED_TXN_FILE`]) defines it as `fn <name>(` inside its
/// `impl WatchedTxn {` block with the return type `Result<Self, …>`, and
/// `Err(why)` otherwise.
fn constructor_verdicts(code: &str) -> Vec<(&'static str, Result<(), String>)> {
    // The `impl WatchedTxn {` at the start of a line, up to its closing `}`
    // (the first one at the start of a line).
    let block = code
        .match_indices("impl WatchedTxn {")
        .find(|(at, _)| *at == 0 || code.as_bytes()[at - 1] == b'\n')
        .map(|(start, _)| {
            let block = &code[start..];
            &block[..block.find("\n}").unwrap_or(block.len())]
        });
    WATCHED_TXN_CONSTRUCTORS
        .iter()
        .map(|&ctor| {
            let verdict = (|| {
                let block = block.ok_or("no `impl WatchedTxn {` block")?;
                let at = block
                    .find(&format!(" fn {ctor}("))
                    .ok_or_else(|| format!("no `fn {ctor}(` in `impl WatchedTxn`"))?;
                let signature = &block[at..];
                let signature = squash(&signature[..signature.find('{').unwrap_or(0)]);
                if signature.contains(")->Result<Self,") {
                    Ok(())
                } else {
                    Err(format!(
                        "`{ctor}` does not return `Result<Self, _>`: `{signature}`"
                    ))
                }
            })();
            (ctor, verdict)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// The guards
// ─────────────────────────────────────────────────────────────────────────────

/// Each constructor the scan accepts as binding a `WatchedTxn` really returns
/// one, so the exemption cannot cover another type's `exec`.
#[test]
fn watched_txn_constructors_return_a_watched_txn() {
    let (_, code) = production_sources()
        .into_iter()
        .find(|(file, _)| file == Path::new(WATCHED_TXN_FILE))
        .unwrap_or_else(|| panic!("{WATCHED_TXN_FILE} must exist"));
    let wrong: Vec<_> = constructor_verdicts(&code)
        .into_iter()
        .filter_map(|(ctor, verdict)| verdict.err().map(|why| format!("{ctor}: {why}")))
        .collect();
    assert!(
        wrong.is_empty(),
        "these entries of `WATCHED_TXN_CONSTRUCTORS` are not `WatchedTxn` constructors in \
         {WATCHED_TXN_FILE}: {wrong:?}. The guard exempts the `exec` of a local bound from them, \
         so each must return `Result<Self, _>`; drop or rename the entry"
    );
}

/// No production code calls a fred `Transaction`'s `exec` outside
/// `ValkeyClient::exec_txn`; that helper's one raw `EXEC` is the only
/// exemption, and the helper re-syncs on an error. The `WatchedTxn::exec`
/// sites are seen and told apart.
#[test]
fn no_fred_transaction_exec_outside_exec_txn() {
    let mut violations = Vec::new();
    let mut exempt = Vec::new();
    let mut watched = Vec::new();
    let mut helper_body = None;
    for (file, code) in production_sources() {
        let (v, e) = raw_execs(&file, &code);
        violations.extend(v);
        exempt.extend(e);
        for call in exec_calls(&code) {
            if is_watched_txn(&code, &call) {
                watched.push(format!("{}::{}", file.display(), call.fn_name));
            }
        }
        if file == Path::new(HELPER_FILE) {
            let at = code
                .find(&format!("fn {HELPER_FN}"))
                .unwrap_or_else(|| panic!("{HELPER_FILE} must define `{HELPER_FN}`"));
            let body = &code[at..];
            let end = body.find("\n    }\n").map_or(body.len(), |e| e + 6);
            helper_body = Some(body[..end].to_owned());
        }
    }
    assert!(
        violations.is_empty(),
        "a fred `Transaction`'s EXEC is sent outside `ValkeyClient::{HELPER_FN}`: \
         {violations:?}. fred re-syncs no routing when a transaction is redirected, so send it \
         as `self.client.{HELPER_FN}::<R>(&checkout, trx, key, op, abort_on_error)`, which \
         re-syncs the checkout's routing on an error (SDD §15.3). A `WatchedTxn`'s `exec` is \
         recognised only when its type is visible (see this module's doc). If the flagged call \
         is neither a fred `Transaction`'s nor a `WatchedTxn`'s `exec` (for example \
         `std::os::unix::process::CommandExt::exec`), exempt it in this guard \
         (`txn_exec_routing_guard_test::is_watched_txn` / `raw_execs`) with a note on what it is"
    );
    assert_eq!(
        exempt,
        vec![format!("{HELPER_FILE}::{HELPER_FN}")],
        "exactly one raw EXEC is exempt: the helper's own"
    );
    let helper_body = helper_body.expect("the helper's body");
    assert!(
        helper_body.contains(RESYNC),
        "`{HELPER_FN}` must call `{RESYNC}` on an error; its body:\n{helper_body}"
    );
    for landmark in [
        "src/checkpoint/residue.rs::wipe_restore_residue",
        "src/storage/layer.rs::update_layer_once_watch_attempt",
        "src/storage/layer.rs::write_restored_object",
        "src/storage/session.rs::create_once",
    ] {
        assert!(
            watched.iter().any(|w| w == landmark),
            "the scan must see and exempt the WatchedTxn::exec in {landmark}; it saw {watched:?}"
        );
    }
}

/// Each `fn` that opens a fred `Transaction` (`.multi()`) sends it through
/// `exec_txn` at least once per `.multi()`; the four transaction sites as of
/// b.cyu are among them. A new site that does passes without being listed.
#[test]
fn every_fred_transaction_site_sends_its_exec_through_exec_txn() {
    let mut sites = Vec::new();
    let mut short = Vec::new();
    for (file, code) in production_sources() {
        for (name, multis, helper_calls) in transactions_and_helper_calls(&code) {
            let site = format!("{}::{name} ×{multis}", file.display());
            if helper_calls < multis {
                short.push(format!("{site} (exec_txn calls: {helper_calls})"));
            }
            sites.push(site);
        }
    }
    assert!(
        short.is_empty(),
        "these fns open a fred Transaction (`.multi()`) more often than they call \
         `.{HELPER_FN}`: {short:?}"
    );
    for landmark in [
        "src/checkpoint/session.rs::apply_session_checkpoint ×2",
        "src/storage/layer.rs::write_object ×1",
        "src/storage/session.rs::delete ×1",
    ] {
        assert!(
            sites.iter().any(|s| s == landmark),
            "the scan must see the transaction site {landmark}; it saw {sites:?}"
        );
    }
}

/// Every `exec_txn` call passes, as its checkout, the client its transaction
/// was opened on (`let trx = <checkout>.multi();`): `exec_txn`'s documented
/// precondition, which the helper itself cannot check, and what makes its
/// re-sync land on the transaction's own `txn_pool` member.
#[test]
fn every_exec_txn_call_gets_the_checkout_its_transaction_was_opened_on() {
    let mut checked = Vec::new();
    let mut wrong = Vec::new();
    for (file, code) in production_sources() {
        for (name, verdict) in exec_txn_checkouts(&code) {
            let site = format!("{}::{name}", file.display());
            match verdict {
                Ok(checkout) => checked.push(format!("{site} ({checkout})")),
                Err(why) => wrong.push(format!("{site}: {why}")),
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "these `{HELPER_FN}` calls do not pass the checkout their transaction was opened on, or \
         the scan cannot trace it: {wrong:?}. `{HELPER_FN}` re-syncs the routing of the checkout \
         it is given, so pass the client `.multi()` was called on, with the transaction bound \
         as `let trx = <checkout>.multi();` in the same fn"
    );
    for (landmark, calls) in [
        (
            "src/checkpoint/session.rs::apply_session_checkpoint (exclusive_client)",
            2,
        ),
        ("src/storage/layer.rs::write_object (client)", 1),
        ("src/storage/session.rs::delete (client)", 2),
    ] {
        assert_eq!(
            checked.iter().filter(|c| *c == landmark).count(),
            calls,
            "the scan must check {calls} `{HELPER_FN}` call(s) in {landmark}; it checked \
             {checked:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scanner self-tests
// ─────────────────────────────────────────────────────────────────────────────

/// A synthetic source: every raw form, every way a `WatchedTxn` is
/// recognised, longer names and a definition that are not calls, and a copy
/// of the helper.
const SYNTHETIC: &str = "\
fn raw_plain(c: &Client) {
    let trx = c.multi();
    trx.exec(true).await;
}
fn raw_turbofish(c: &Client) {
    let trx = c.multi();
    let v = trx.exec::<Value>(true).await;
}
fn raw_chained(c: &Client) {
    c.multi()
        .exec::<()>(true)
        .await;
}
fn raw_own_line(c: &Client) {
    let trx = c.multi();
    let r = trx
        .exec(true)
        .await;
}
fn raw_path(trx: Transaction) {
    Transaction::exec(trx, true).await;
}
fn raw_field(&self) {
    self.trx.exec(true).await;
}
fn raw_shadowed(c: &Client) {
    let txn = WatchedTxn::open(&c, &h, &b).await?;
    let txn = c.multi();
    txn.exec(true).await;
}
fn raw_shadowing_a_watched_param(txn: WatchedTxn, c: &Client) {
    let txn = c.multi();
    txn.exec(true).await;
}
fn raw_beside_a_suffix_param(old_txn: WatchedTxn, txn: Transaction) {
    txn.exec(true).await;
}
fn raw_with_a_longer_type(txn: WatchedTxnGuard) {
    txn.exec(true).await;
}
fn raw_from_another_watched_txn_fn(c: &Client) {
    let txn = WatchedTxn::open_pipeline(&c, &h).await?;
    txn.exec(true).await;
}
fn raw_from_a_longer_constructor_name(c: &Client) {
    let txn = match WatchedTxn::open_with_timeout_raw(&c, &h, &b, t).await { Ok(t) => t };
    txn.exec(true).await;
}
fn watched_bound(&self) {
    let txn = match WatchedTxn::open(&c, &h, &b).await { Ok(t) => t, Err(e) => return e };
    let out = txn.exec(origin).await;
}
fn watched_bound_with_timeout(&self) {
    let txn = match WatchedTxn::open_with_timeout(
        &c,
        &h,
        &b,
        t,
    )
    .await
    {
        Ok(txn) => txn,
        Err(e) => return e,
    };
    let out = txn.exec(origin).await;
}
fn watched_bound_with_timeout_directly(&self) {
    let txn = WatchedTxn::open_with_timeout(&c, &h, &b, t).await?;
    txn.exec(origin).await;
}
fn watched_own_line(&self) {
    let txn = WatchedTxn::open(&c, &h, &b).await?;
    match txn
        .exec(FredErrOrigin::client(c, P))
        .await
    {}
}
fn watched_annotated(&self) {
    let txn: WatchedTxn = make();
    txn.exec(origin).await;
}
fn watched_param(origin: O, txn: WatchedTxn) {
    txn.exec(origin).await;
}
fn watched_path(txn: WatchedTxn) {
    WatchedTxn::exec(txn, origin).await;
}
fn not_calls(&self) {
    txn.exec_overridden(o, over).await;
    self.client.exec_txn::<()>(&c, trx, k, \"op\", true).await;
    let execute = 1;
}
pub(crate) async fn exec(self, origin: O) {}
pub(crate) async fn exec_txn<R>(&self, trx: Transaction) {
    let result = trx.exec::<R>(abort_on_error).await;
}
";

/// The scan finds every raw form — `.exec(`, `.exec::<`, a chained call, a
/// `.exec` on its own line, the `Transaction::exec(` path form, a field
/// receiver, a `WatchedTxn` local shadowed by a `Transaction`, a
/// `WatchedTxn` parameter shadowed by a `Transaction` local, a receiver
/// beside a parameter whose name only ends in it (`old_txn: WatchedTxn`), a
/// parameter of a longer type (`WatchedTxnGuard`), and a local bound from a
/// `WatchedTxn::` function that is not a listed constructor (another name,
/// or one that only starts with `open_with_timeout`) — and passes over every
/// visible `WatchedTxn` (bound from `WatchedTxn::open` or
/// `WatchedTxn::open_with_timeout`, directly or through `match`, with `.exec`
/// on its own line, annotated, a parameter, the path form).
/// `exec_overridden`, `exec_txn`, `execute` and `fn exec(` are not calls of
/// `exec`.
#[test]
fn the_exec_scan_sees_every_raw_form_and_tells_a_watched_txn_apart() {
    let calls = exec_calls(SYNTHETIC);
    let raw: Vec<&str> = calls
        .iter()
        .filter(|c| !is_watched_txn(SYNTHETIC, c))
        .map(|c| c.fn_name.as_str())
        .collect();
    let watched: Vec<&str> = calls
        .iter()
        .filter(|c| is_watched_txn(SYNTHETIC, c))
        .map(|c| c.fn_name.as_str())
        .collect();
    assert_eq!(
        raw,
        vec![
            "raw_plain",
            "raw_turbofish",
            "raw_chained",
            "raw_own_line",
            "raw_path",
            "raw_field",
            "raw_shadowed",
            "raw_shadowing_a_watched_param",
            "raw_beside_a_suffix_param",
            "raw_with_a_longer_type",
            "raw_from_another_watched_txn_fn",
            "raw_from_a_longer_constructor_name",
            "exec_txn",
        ]
    );
    assert_eq!(
        watched,
        vec![
            "watched_bound",
            "watched_bound_with_timeout",
            "watched_bound_with_timeout_directly",
            "watched_own_line",
            "watched_annotated",
            "watched_param",
            "watched_path",
        ]
    );
}

/// The helper's own raw `EXEC` is exempt only in [`HELPER_FILE`]: the same
/// source scanned as another file reports it too.
#[test]
fn only_the_helpers_own_exec_is_exempt_and_only_in_its_file() {
    let (violations, exempt) = raw_execs(Path::new(HELPER_FILE), SYNTHETIC);
    assert_eq!(exempt, vec![format!("{HELPER_FILE}::{HELPER_FN}")]);
    assert_eq!(violations.len(), 12, "{violations:?}");
    let (violations, exempt) = raw_execs(Path::new("src/storage/other.rs"), SYNTHETIC);
    assert!(exempt.is_empty());
    assert_eq!(violations.len(), 13, "{violations:?}");
    assert!(violations
        .iter()
        .any(|v| v.starts_with("src/storage/other.rs::exec_txn ")));
}

/// The constructor check accepts a listed name defined in `impl WatchedTxn`
/// returning `Result<Self, …>` (its signature across lines), and reports one
/// returning another type, one defined only outside `impl WatchedTxn`, and a
/// file with no `impl WatchedTxn` at all.
#[test]
fn the_constructor_check_wants_each_listed_name_to_return_self() {
    let good = "\
pub(crate) struct WatchedTxn {}
impl WatchedTxn {
    pub(crate) async fn open(
        client: &Client,
    ) -> Result<Self, fred::error::Error> {
        todo!()
    }
    pub(crate) async fn open_with_timeout(c: &Client, t: Duration) -> Result<Self, Error> {
        todo!()
    }
}
";
    assert_eq!(
        constructor_verdicts(good),
        vec![("open", Ok(())), ("open_with_timeout", Ok(()))]
    );

    let bad = "\
impl WatchedTxn {
    pub(crate) async fn open(client: &Client) -> Result<Self, Error> {
        todo!()
    }
    pub(crate) async fn open_with_timeout(c: &Client) -> Result<Transaction, Error> {
        todo!()
    }
}
";
    let verdicts = constructor_verdicts(bad);
    assert_eq!(verdicts[0], ("open", Ok(())));
    let why = verdicts[1].1.as_ref().unwrap_err();
    assert!(
        why.contains("does not return `Result<Self, _>`"),
        "another return type is reported; got {why}"
    );

    let outside = "\
impl WatchedTxn {
    pub(crate) async fn open(client: &Client) -> Result<Self, Error> {
        todo!()
    }
}
impl Other {
    pub(crate) async fn open_with_timeout(c: &Client) -> Result<Self, Error> {
        todo!()
    }
}
";
    let verdicts = constructor_verdicts(outside);
    assert_eq!(verdicts[0], ("open", Ok(())));
    assert!(
        verdicts[1].1.is_err(),
        "a `Self` of another impl is not a WatchedTxn; got {:?}",
        verdicts[1]
    );

    for (ctor, verdict) in constructor_verdicts("fn open() -> Result<Self, Error> {}\n") {
        assert!(verdict.is_err(), "{ctor}: no `impl WatchedTxn` must fail");
    }
}

/// The checkout scan accepts an `exec_txn` given the client its transaction
/// was opened on (plain or turbofish, `&` and line breaks aside, the
/// nearest `let` of the transaction deciding, the checkout bound before it in
/// an outer block), and reports one whose checkout name is re-bound with
/// `let` after the transaction's `let`, one given another client, one whose
/// transaction is not a local, one whose transaction is bound from something
/// other than `.multi()`, and one with no `let` of it.
#[test]
fn the_checkout_scan_matches_each_exec_txn_to_its_multi() {
    let code = "\
fn same(&self) {
    let trx = exclusive_client.multi();
    self.client
        .exec_txn::<Value>(
            &exclusive_client,
            trx,
            &key,
            \"op\",
            true,
        )
        .await;
}
fn shadowed_then_same(&self) {
    let trx = other.multi();
    let trx = client.multi();
    self.client.exec_txn(&client, trx, &key, \"op\", true).await;
}
fn checkout_bound_before(&self) {
    let client = first_checkout;
    let client = acquire().await;
    {
        let trx = client.multi();
        self.client.exec_txn(&client, trx, &key, \"op\", true).await;
    }
}
fn checkout_rebound_after(&self) {
    let trx = client.multi();
    let client = other_checkout;
    self.client.exec_txn(&client, trx, &key, \"op\", true).await;
}
fn mismatched(&self) {
    let trx = exclusive_client.multi();
    self.client.exec_txn::<Value>(&client, trx, &key, \"op\", true).await;
}
fn not_a_local(&self) {
    self.client.exec_txn::<()>(&client, client.multi(), &key, \"op\", true).await;
}
fn not_from_multi(&self) {
    let trx = make_transaction(&client);
    self.client.exec_txn::<()>(&client, trx, &key, \"op\", true).await;
}
fn untraced(trx: Transaction) {
    self.client.exec_txn::<()>(&client, trx, &key, \"op\", true).await;
}
";
    let found = exec_txn_checkouts(code);
    let names: Vec<&str> = found.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "same",
            "shadowed_then_same",
            "checkout_bound_before",
            "checkout_rebound_after",
            "mismatched",
            "not_a_local",
            "not_from_multi",
            "untraced",
        ]
    );
    assert_eq!(found[0].1, Ok("exclusive_client".to_owned()));
    assert_eq!(found[1].1, Ok("client".to_owned()));
    assert_eq!(
        found[2].1,
        Ok("client".to_owned()),
        "a checkout bound (even re-bound) before the transaction's `let` is the one it was \
         opened on"
    );
    for (name, verdict) in &found[3..] {
        assert!(verdict.is_err(), "{name} must be reported; got {verdict:?}");
    }
    let rebound = found[3].1.as_ref().unwrap_err();
    assert!(
        rebound.contains("`client` is re-bound with `let` after `trx` was opened on it"),
        "the re-bind names the checkout and the transaction; got {rebound}"
    );
    let mismatch = found[4].1.as_ref().unwrap_err();
    assert!(
        mismatch
            .contains("`trx` was opened on `exclusive_client`, but `exec_txn` is given `client`"),
        "the mismatch names both clients; got {mismatch}"
    );
}

/// The site scan counts each `fn`'s `.multi()` calls and its `exec_txn`
/// calls (plain and turbofish), keeping two `fn`s of one name apart.
#[test]
fn the_site_scan_counts_each_fns_transactions_and_helper_calls() {
    let code = "\
fn ok(&self) {
    let trx = checkout.multi();
    self.client
        .exec_txn::<()>(&checkout, trx, k, \"op\", true)
        .await;
}
fn two_but_one(&self) {
    let a = checkout.multi();
    let b = checkout
        .multi();
    self.client.exec_txn(&checkout, a, k, \"op\", true).await;
}
fn raw(&self) {
    let t = checkout.multi();
    t.exec(true).await;
}
fn ok(&self) {
    let again = checkout.multi();
    self.client.exec_txn::<Value>(&checkout, again, k, \"op\", true).await;
}
";
    assert_eq!(
        transactions_and_helper_calls(code),
        vec![
            ("ok".to_owned(), 1, 1),
            ("two_but_one".to_owned(), 2, 1),
            ("raw".to_owned(), 1, 0),
            ("ok".to_owned(), 1, 1),
        ]
    );
}
