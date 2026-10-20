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

//! The fence route guard: every `update_layer` commit route honours the
//! snapshot fence (b.y3m SR-5.4), enforced mechanically rather than by
//! convention.
//!
//! The behavioural tests (`tests/integration/storage/snapshot_fence_writer_test.rs`)
//! prove each route waits on a live fence. They cannot prove that a route
//! added later has a fence check at all: a new route arrives with no fence
//! test, and nothing fails. This module closes that gap at two levels.
//!
//! - **The dispatch, at compile time.** [`proof`] matches
//!   [`UpdateLayerRoute`] with no catch-all arm and maps each route to the
//!   function the dispatch calls for it, the fence check that route runs
//!   ([`FenceCheck`]), and the behavioural tests that prove it. With a new
//!   variant the crate's test build (`cargo test`, `clippy --all-targets`,
//!   which CI runs) does not compile until it is mapped here, and
//!   [`every_route_variant_is_in_routes`] fails if it is mapped but left out of
//!   [`ROUTES`].
//! - **The source, at test time.** The rest read the source, with comments,
//!   string literals and `#[cfg(test)]` items blanked
//!   ([`production_code`]): the dispatch in `update_layer_once` matches
//!   `route` with no catch-all arm and calls exactly the mapped functions; each
//!   route's function runs its fence check; every function in `src/` that
//!   invokes the update script decodes the script's `FENCED` reply and is a
//!   mapped script route; every function that queues a WATCH-path commit (the
//!   commit-token record every update commit writes) is a mapped WATCH attempt
//!   with the fence key in its `WATCH` set, the fence probe and the `EXEC`
//!   re-probe; and the script's step 1d precedes the mode parse and every
//!   write, so both modes run it. A function that holds either anchor
//!   (`UPDATE_LAYER_FAST_SCRIPT`, `queue_commit_token_record`) but commits
//!   nothing is excused only by a row of [`NON_COMMIT_SITES`].
//! - **The anchors cannot be side-stepped.** Each anchor is reachable only
//!   under its own name: no `as` rename and no item outside a function
//!   holding it (a `static`/`const` alias) other than its own definition. For
//!   the update script, also no second `Script` built from its source and no
//!   second `include_str!` of the file.
//!
//! A source scan pins the shape of the code, not its behaviour; the
//! behavioural tests [`proof`] names pin the behaviour. The scan's anchors are
//! named where they are used, and each was perturbed in place to confirm it
//! fails (the W1b commit messages list the perturbations).

use std::path::{Path, PathBuf};

use super::UpdateLayerRoute;

const LAYER_RS: &str = "src/storage/layer.rs";
const STRUCTURAL_RS: &str = "src/storage/structural.rs";
const UPDATE_SCRIPT_LUA: &str = "src/storage/scripts/update_layer_fast.lua";
const WRITER_TESTS: &str = "tests/integration/storage/snapshot_fence_writer_test.rs";

/// Every [`UpdateLayerRoute`] variant. [`every_route_variant_is_in_routes`]
/// holds this to the enum's declaration, and [`proof`]'s exhaustive match to
/// the variants, so between them a new route is mapped and checked.
const ROUTES: [UpdateLayerRoute; 3] = [
    UpdateLayerRoute::Fast,
    UpdateLayerRoute::Structural,
    UpdateLayerRoute::Watch,
];

/// How a route honours the snapshot fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FenceCheck {
    /// The route commits through the update script, whose step 1d refuses a
    /// commit under a live fence in both modes. `script_fn`, in `file`, issues
    /// the `EVALSHA` and decodes the `FENCED` reply.
    Script {
        file: &'static str,
        script_fn: &'static str,
    },
    /// The route commits through `WATCH`/`MULTI`/`EXEC`. `attempt_fn`, in
    /// `layer.rs`, `WATCH`es the fence key, runs the fence probe beside its
    /// header read and re-probes a `WATCH`-cancelled `EXEC`.
    Watch { attempt_fn: &'static str },
}

/// What one route must show to honour the fence.
#[derive(Debug)]
struct RouteProof {
    /// The `LayerStore` method the dispatch calls for the route.
    entry: &'static str,
    /// The fence check the route runs.
    fence: FenceCheck,
    /// The route another route falls back to inside its entry, which must
    /// honour the fence by its own proof.
    fallback: Option<UpdateLayerRoute>,
    /// Behavioural tests in [`WRITER_TESTS`] that commit on the route under a
    /// live fence.
    tests: &'static [&'static str],
}

/// The fence proof of `route`. No catch-all arm, deliberately: adding an
/// [`UpdateLayerRoute`] variant fails to compile here until the new route is
/// given an entry, a fence check and the tests that prove it.
fn proof(route: UpdateLayerRoute) -> RouteProof {
    match route {
        UpdateLayerRoute::Fast => RouteProof {
            entry: "update_layer_once_lua",
            fence: FenceCheck::Script {
                file: LAYER_RS,
                script_fn: "update_layer_once_lua",
            },
            fallback: None,
            tests: &[
                "fast_path_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release",
                "a_permanently_renewed_fence_fails_a_fast_path_commit_unavailable_at_the_cap",
                "the_step_1b_guards_answer_ahead_of_a_live_fence_on_every_path",
            ],
        },
        UpdateLayerRoute::Structural => RouteProof {
            entry: "update_layer_once_structural",
            fence: FenceCheck::Script {
                file: STRUCTURAL_RS,
                script_fn: "commit_structural",
            },
            fallback: Some(UpdateLayerRoute::Watch),
            tests: &[
                "cached_structural_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release",
                "cold_start_structural_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release",
                "reset_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release",
                "uncacheable_fallback_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release",
                "a_permanently_renewed_fence_fails_a_cached_structural_commit_unavailable_at_the_cap",
            ],
        },
        UpdateLayerRoute::Watch => RouteProof {
            entry: "update_layer_once_watch",
            fence: FenceCheck::Watch {
                attempt_fn: "update_layer_once_watch_attempt",
            },
            fallback: None,
            tests: &[
                "watch_knob_structural_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release",
                "txn_field_only_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release",
                "txn_structural_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release",
                "a_fence_granted_between_watch_and_exec_is_an_uncounted_wait_for_a_txn_field_commit",
                "a_permanently_renewed_fence_fails_a_txn_structural_commit_unavailable_at_the_cap",
            ],
        },
    }
}

// ── Source access ────────────────────────────────────────────────────────

fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(repo_path(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// `src` with every comment and every string / char literal's contents
/// replaced by spaces (newlines kept), so byte offsets and line numbers still
/// match the file and a brace or a name inside a literal or a comment is
/// invisible to the scans. Lifetimes (`'a`) are left as code.
fn blank_comments_and_literals(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for c in &mut out[from..to] {
            if *c != b'\n' {
                *c = b' ';
            }
        }
    };
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'/' if b.get(i + 1) == Some(&b'/') => {
                let end = b[i..]
                    .iter()
                    .position(|&c| c == b'\n')
                    .map_or(b.len(), |p| i + p);
                blank(&mut out, i, end);
                i = end;
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let (mut depth, mut j) = (1, i + 2);
                while j < b.len() && depth > 0 {
                    if b[j] == b'/' && b.get(j + 1) == Some(&b'*') {
                        depth += 1;
                        j += 2;
                    } else if b[j] == b'*' && b.get(j + 1) == Some(&b'/') {
                        depth -= 1;
                        j += 2;
                    } else {
                        j += 1;
                    }
                }
                blank(&mut out, i, j);
                i = j;
            }
            b'r' if !is_ident_byte(b, i.wrapping_sub(1))
                && matches!(b.get(i + 1), Some(b'"') | Some(b'#')) =>
            {
                // A raw string `r"…"` / `r#"…"#` (also the tail of `br…`).
                let mut j = i + 1;
                let mut hashes = 0;
                while b.get(j) == Some(&b'#') {
                    hashes += 1;
                    j += 1;
                }
                if b.get(j) != Some(&b'"') {
                    i += 1;
                    continue;
                }
                let body = j + 1;
                let mut k = body;
                loop {
                    if k >= b.len() {
                        break;
                    }
                    if b[k] == b'"' && (0..hashes).all(|h| b.get(k + 1 + h) == Some(&b'#')) {
                        break;
                    }
                    k += 1;
                }
                blank(&mut out, body, k.min(b.len()));
                i = (k + 1 + hashes).min(b.len());
            }
            b'"' => {
                let mut j = i + 1;
                while j < b.len() && b[j] != b'"' {
                    j += if b[j] == b'\\' { 2 } else { 1 };
                }
                blank(&mut out, i + 1, j.min(b.len()));
                i = j + 1;
            }
            b'\'' => {
                // A char literal (`'x'`, `'\n'`, `'\''`, `'\u{1F600}'`, `'{'`)
                // or a lifetime / label (`'a`), told apart by where the next
                // `'` sits. An escape's closing `'` is searched for past the
                // escaped character, so `'\''` closes at its third quote.
                let close = if b.get(i + 1) == Some(&b'\\') {
                    b.get(i + 3..)
                        .and_then(|rest| rest.iter().position(|&c| c == b'\''))
                        .map(|p| i + 3 + p)
                } else {
                    let len = std::str::from_utf8(&b[i + 1..])
                        .ok()
                        .and_then(|s| s.chars().next())
                        .map_or(1, char::len_utf8);
                    (b.get(i + 1 + len) == Some(&b'\'')).then_some(i + 1 + len)
                };
                match close {
                    Some(end) => {
                        blank(&mut out, i + 1, end);
                        i = end + 1;
                    }
                    None => i += 1,
                }
            }
            _ => i += 1,
        }
    }
    String::from_utf8(out).expect("blanking keeps UTF-8: only ASCII bytes are written")
}

fn is_ident_byte(b: &[u8], i: usize) -> bool {
    b.get(i)
        .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

/// The offset one past the `}` matching the `{` at `open`.
fn matching_brace(code: &str, open: usize) -> Option<usize> {
    let b = code.as_bytes();
    debug_assert_eq!(b[open], b'{');
    let mut depth = 0usize;
    for (k, &c) in b.iter().enumerate().skip(open) {
        match c {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(k + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// `src` as production code: [`blank_comments_and_literals`], then every item
/// under `#[cfg(test)]` blanked — a `mod tests { … }`, a test-only `fn`, a
/// `#[cfg(test)] mod x;` — so test code that invokes the script or builds a
/// transaction directly is not read as a commit route. `cfg(any(test, …))`
/// seams inside production functions stay: they are compiled into the
/// test-fault-injection build.
fn production_code(src: &str) -> String {
    let mut code = blank_comments_and_literals(src);
    const ATTR: &str = "#[cfg(test)]";
    let mut from = 0;
    while let Some(p) = code[from..].find(ATTR).map(|p| from + p) {
        let after = p + ATTR.len();
        let next_brace = code[after..].find('{').map(|q| after + q);
        let next_semi = code[after..].find(';').map(|q| after + q);
        let end = match (next_brace, next_semi) {
            (Some(open), Some(semi)) if semi < open => semi + 1,
            (Some(open), _) => matching_brace(&code, open).unwrap_or(code.len()),
            (None, Some(semi)) => semi + 1,
            (None, None) => code.len(),
        };
        let blanked: String = code[p..end]
            .chars()
            .map(|c| if c == '\n' { '\n' } else { ' ' })
            .collect();
        code.replace_range(p..end, &blanked);
        from = end;
    }
    code
}

/// One `fn` item with a body: its name and the byte span of the body,
/// braces included.
#[derive(Debug, Clone)]
struct FnSpan {
    name: String,
    body: std::ops::Range<usize>,
}

/// Every `fn` with a body in `code` (nested ones included), in source order.
fn fn_spans(code: &str) -> Vec<FnSpan> {
    let b = code.as_bytes();
    let mut spans = Vec::new();
    let mut from = 0;
    while let Some(p) = code[from..].find("fn ").map(|p| from + p) {
        from = p + 3;
        if is_ident_byte(b, p.wrapping_sub(1)) {
            continue;
        }
        let name: String = code[p + 3..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        // The body opens at the first `{` after the signature, unless a `;`
        // outside its parentheses and brackets (`[u8; 4]` is not one) ends a
        // body-less declaration first.
        let mut depth = 0i32;
        let mut open = None;
        for (k, &c) in b.iter().enumerate().skip(from) {
            match c {
                b'(' | b'[' => depth += 1,
                b')' | b']' => depth -= 1,
                b';' if depth == 0 => break,
                b'{' if depth == 0 => {
                    open = Some(k);
                    break;
                }
                _ => {}
            }
        }
        let Some(open) = open else { continue };
        if let Some(end) = matching_brace(code, open) {
            spans.push(FnSpan {
                name,
                body: open..end,
            });
        }
    }
    spans
}

/// The innermost `fn` whose body contains `at`.
fn enclosing_fn(spans: &[FnSpan], at: usize) -> Option<&FnSpan> {
    spans
        .iter()
        .filter(|s| s.body.contains(&at))
        .min_by_key(|s| s.body.len())
}

/// The body of the one `fn` named `name` in `code`.
fn fn_body<'a>(code: &'a str, name: &str) -> &'a str {
    let spans: Vec<FnSpan> = fn_spans(code)
        .into_iter()
        .filter(|s| s.name == name)
        .collect();
    assert_eq!(
        spans.len(),
        1,
        "expected exactly one `fn {name}` with a body, found {}",
        spans.len()
    );
    &code[spans[0].body.clone()]
}

/// Every offset where `word` occurs as a whole identifier path segment.
fn word_offsets(code: &str, word: &str) -> Vec<usize> {
    let b = code.as_bytes();
    code.match_indices(word)
        .map(|(p, _)| p)
        .filter(|&p| !is_ident_byte(b, p.wrapping_sub(1)) && !is_ident_byte(b, p + word.len()))
        .collect()
}

/// `code` with all whitespace removed, so a check reads the same whatever
/// rustfmt does to line breaks.
fn squash(code: &str) -> String {
    code.chars().filter(|c| !c.is_whitespace()).collect()
}

/// How many times `code` calls `self.<method>(`, however rustfmt breaks the
/// chain (`self\n    .method(`).
fn method_calls(code: &str, method: &str) -> usize {
    squash(code).matches(&format!("self.{method}(")).count()
}

/// The argument list of the first call `callee(` in `code`, parentheses
/// excluded.
fn call_args<'a>(code: &'a str, callee: &str) -> Option<&'a str> {
    call_args_range(code, callee).map(|r| &code[r])
}

/// The byte range of [`call_args`]'s answer in `code`, so the same range can
/// be read from the unblanked source.
fn call_args_range(code: &str, callee: &str) -> Option<std::ops::Range<usize>> {
    let open = word_offsets(code, callee)
        .into_iter()
        .find(|&p| code[p + callee.len()..].trim_start().starts_with('('))?;
    let open = open + callee.len() + code[open + callee.len()..].find('(')?;
    let mut depth = 0usize;
    for (k, c) in code[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + 1..open + k);
                }
            }
            _ => {}
        }
    }
    None
}

/// The `.rs` files under `src/`, as repository-relative paths.
fn src_rs_files() -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"));
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    walk(&root.join("src"), &mut files);
    let mut rel: Vec<String> = files
        .iter()
        .map(|p| {
            p.strip_prefix(root)
                .expect("under the manifest dir")
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    rel.sort();
    rel
}

/// Where in a file an anchor occurrence sits.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Context {
    /// Inside the body of the `fn` of this name (innermost).
    Fn(String),
    /// In a `use` declaration outside any function.
    Use,
    /// In another item outside any function: `static NAME`, `const NAME`,
    /// `const fn NAME`, `async fn NAME` or `fn NAME` (a function's own
    /// signature, not its body), or `item at line N` when the item's header is
    /// not one of those.
    Item(String),
}

/// One occurrence of an anchor in production code.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AnchorSite {
    file: String,
    context: Context,
    /// The occurrence is followed by `as <name>`: the anchor is renamed, and a
    /// scan by its name would miss every use of the new name.
    renamed: bool,
}

/// The context of the occurrence at `at` in `code`.
fn context_of(code: &str, spans: &[FnSpan], at: usize) -> Context {
    if let Some(f) = enclosing_fn(spans, at) {
        return Context::Fn(f.name.clone());
    }
    // The item outside any function is the one whose header is the nearest
    // line at column 0, at or above the occurrence's own line: a multi-line
    // `use` path's continuation lines are indented and its closing `};` is
    // passed over, and blanked comment lines start with a space. An item
    // nested in an `impl` or `mod` reads as that block's header, so it fails
    // the callers' known-item checks rather than passing them.
    let line_no = code[..at].matches('\n').count() + 1;
    let lines: Vec<&str> = code.lines().collect();
    let header = lines[..line_no.min(lines.len())]
        .iter()
        .rev()
        .find(|l| {
            !l.is_empty()
                && !l.starts_with(char::is_whitespace)
                && !l.starts_with("#[")
                && !l.starts_with('}')
        })
        .map_or("", |l| strip_visibility(l));
    if header.starts_with("use ") {
        return Context::Use;
    }
    for keyword in ["static ", "const fn ", "const ", "async fn ", "fn "] {
        if let Some(rest) = header.strip_prefix(keyword) {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            return Context::Item(format!("{keyword}{name}"));
        }
    }
    Context::Item(format!("item at line {line_no}"))
}

fn strip_visibility(s: &str) -> &str {
    for vis in ["pub(crate) ", "pub(super) ", "pub "] {
        if let Some(rest) = s.strip_prefix(vis) {
            return rest.trim_start();
        }
    }
    s
}

/// Every occurrence of `anchor` (a whole identifier) in production code
/// across `src/`, one row per distinct `(file, context, renamed)`.
fn anchor_sites(anchor: &str) -> Vec<AnchorSite> {
    let files = src_rs_files();
    assert!(
        files.len() > 50,
        "the walk found only {} `.rs` files under src/; a walk that sees nothing proves nothing",
        files.len()
    );
    let mut sites = Vec::new();
    for file in files {
        let code = production_code(&read(&file));
        let spans = fn_spans(&code);
        for at in word_offsets(&code, anchor) {
            let after = code[at + anchor.len()..].trim_start();
            let renamed =
                after.starts_with("as") && after[2..].starts_with(|c: char| c.is_whitespace());
            let site = AnchorSite {
                file: file.clone(),
                context: context_of(&code, &spans, at),
                renamed,
            };
            if !sites.contains(&site) {
                sites.push(site);
            }
        }
    }
    sites.sort();
    sites
}

/// Every production function, across `src/`, that holds an occurrence of
/// `anchor` in its body, as `(file, fn)`. Occurrences outside any function
/// are left to the callers' [`anchor_sites`] checks.
fn functions_holding(anchor: &str) -> Vec<(String, String)> {
    anchor_sites(anchor)
        .into_iter()
        .filter_map(|s| match s.context {
            Context::Fn(name) => Some((s.file, name)),
            Context::Use | Context::Item(_) => None,
        })
        .fold(Vec::new(), |mut out, site| {
            if !out.contains(&site) {
                out.push(site);
            }
            out
        })
}

/// Functions that hold a commit anchor — they mention
/// `UPDATE_LAYER_FAST_SCRIPT`, or call `queue_commit_token_record` — yet
/// commit nothing, so they are not commit routes: `(file, function, reason)`,
/// the reason one line. Both "every site is a mapped route" scans subtract
/// these rows, and [`every_non_commit_site_row_matches_a_site`] fails on a row
/// that no longer matches a site. Empty today: every function holding an
/// anchor is a mapped route.
const NON_COMMIT_SITES: &[(&str, &str, &str)] = &[];

fn allowlisted(file: &str, function: &str) -> bool {
    NON_COMMIT_SITES
        .iter()
        .any(|(f, n, _)| *f == file && *n == function)
}

/// The remedy every "is a mapped route" failure names.
const MAP_OR_ALLOWLIST: &str = "map the function as a route in `proof` (an `UpdateLayerRoute` \
     the dispatch calls, with its fence check), or, if it does not commit, list it in \
     `NON_COMMIT_SITES` with a one-line reason";

/// The causes and remedy every "held by another item" failure names.
const ITEM_ALIAS_CAUSES: &str = "either an alias the scans cannot follow — use the anchor by its \
     own name — or a `use` inside an inline `mod`, which the scanner reads as that block — \
     move the `use` to the file's top level";

// ── The route enumeration ─────────────────────────────────────────────────

/// [`ROUTES`] lists every variant [`UpdateLayerRoute`] declares, each once.
/// [`proof`]'s exhaustive match makes a new variant map before it compiles;
/// this makes the tests below, which walk [`ROUTES`], check it.
#[test]
fn every_route_variant_is_in_routes() {
    let code = blank_comments_and_literals(&read(LAYER_RS));
    let decl = code
        .find("enum UpdateLayerRoute")
        .expect("`enum UpdateLayerRoute` is declared in layer.rs");
    let open = decl + code[decl..].find('{').expect("the enum has a body");
    let close = matching_brace(&code, open).expect("the enum body closes");
    let body = &code[open + 1..close - 1];
    // Attributes on variants (`#[…]`) are stripped; every remaining
    // comma-separated item names one variant.
    let declared: Vec<String> = body
        .split(',')
        .map(|item| {
            item.lines()
                .filter(|l| !l.trim_start().starts_with("#["))
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_owned()
        })
        .filter(|item| !item.is_empty())
        .collect();
    let listed: Vec<String> = ROUTES.iter().map(|r| format!("{r:?}")).collect();
    assert_eq!(
        declared, listed,
        "ROUTES must list every `UpdateLayerRoute` variant in declaration order; a route missing \
         from ROUTES is mapped by `proof` but never checked"
    );
}

/// The dispatch in `update_layer_once` takes its route from
/// `update_layer_route`, matches it with no catch-all arm, and calls, for each
/// route, exactly the entry [`proof`] maps it to — and no other route entry.
/// A fourth route therefore cannot be dispatched without a variant, which
/// [`proof`] must map before the crate's tests compile.
#[test]
fn the_dispatch_matches_every_route_and_calls_its_mapped_entry() {
    let code = production_code(&read(LAYER_RS));
    let body = fn_body(&code, "update_layer_once");
    assert!(
        squash(body).contains("letroute=self.update_layer_route(class);"),
        "`update_layer_once` must take its route from `self.update_layer_route(class)`"
    );
    let at = body
        .find("match route {")
        .expect("`update_layer_once` dispatches with `match route {`");
    let open = at + "match route ".len();
    let close = matching_brace(body, open).expect("the dispatch match closes");
    let arms = &body[open + 1..close - 1];

    // Top-level arms only: split at depth 0 on `=>`.
    let mut heads = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    let bytes = arms.as_bytes();
    let mut k = 0;
    while k < bytes.len() {
        match bytes[k] {
            b'{' | b'(' | b'[' => depth += 1,
            b'}' | b')' | b']' => {
                depth -= 1;
                if depth == 0 && bytes[k] == b'}' {
                    start = k + 1;
                }
            }
            b',' if depth == 0 => start = k + 1,
            b'=' if depth == 0 && bytes.get(k + 1) == Some(&b'>') => {
                heads.push((arms[start..k].trim().to_owned(), k + 2));
            }
            _ => {}
        }
        k += 1;
    }
    for (head, _) in &heads {
        assert!(
            head.split('|')
                .all(|p| p.trim().starts_with("UpdateLayerRoute::")),
            "the dispatch arm `{head} =>` is not a named `UpdateLayerRoute` variant; a catch-all \
             arm would send a new route to an existing route's code unchecked"
        );
    }

    let entries: Vec<&str> = ROUTES.iter().map(|&r| proof(r).entry).collect();
    for (i, (head, arm_start)) in heads.iter().enumerate() {
        let arm_end = heads.get(i + 1).map_or(arms.len(), |(_, s)| *s);
        let arm = &arms[*arm_start..arm_end];
        let called: Vec<&str> = entries
            .iter()
            .copied()
            .filter(|e| method_calls(arm, e) > 0)
            .collect();
        let routes: Vec<UpdateLayerRoute> = ROUTES
            .iter()
            .copied()
            .filter(|r| head.contains(&format!("UpdateLayerRoute::{r:?}")))
            .collect();
        assert_eq!(routes.len(), 1, "the arm `{head}` names exactly one route");
        assert_eq!(
            called,
            vec![proof(routes[0]).entry],
            "the arm `{head}` must call exactly `self.{}(…)`, the entry `proof` maps the route to",
            proof(routes[0]).entry
        );
    }
    let dispatched: Vec<String> = heads.iter().map(|(h, _)| h.clone()).collect();
    let expected: Vec<String> = ROUTES
        .iter()
        .map(|r| format!("UpdateLayerRoute::{r:?}"))
        .collect();
    assert_eq!(
        dispatched, expected,
        "the dispatch has one arm per route, in ROUTES order"
    );

    // No route entry is called from `update_layer_once` outside the match, so
    // nothing can branch around the dispatch.
    for e in entries {
        let calls = method_calls(body, e);
        let in_match = method_calls(arms, e);
        assert_eq!(
            calls, in_match,
            "`update_layer_once` calls `{e}` outside the dispatch match"
        );
    }
}

/// Each route runs the fence check [`proof`] maps it to, through its entry.
#[test]
fn every_route_runs_its_fence_check() {
    let layer = production_code(&read(LAYER_RS));
    let structural = production_code(&read(STRUCTURAL_RS));
    let code_of = |file: &str| -> &str {
        match file {
            LAYER_RS => &layer,
            STRUCTURAL_RS => &structural,
            other => panic!("no source loaded for {other}"),
        }
    };
    for route in ROUTES {
        let p = proof(route);
        let entry = fn_body(&layer, p.entry);
        match p.fence {
            FenceCheck::Script { file, script_fn } => {
                assert!(
                    p.entry == script_fn || call_args(entry, script_fn).is_some(),
                    "{route:?}: `{}` must call `{script_fn}`, the function that runs the update \
                     script for it",
                    p.entry
                );
                assert_script_site_decodes_fenced(code_of(file), file, script_fn);
                if p.entry != script_fn {
                    // The script route's entry turns the decoded refusal into
                    // the call's uncounted wait.
                    assert!(
                        squash(entry).contains("CommitOutcome::Fenced{")
                            && entry.contains("WaitUncounted"),
                        "{route:?}: `{}` must turn `CommitOutcome::Fenced` into \
                         `AttemptOutcome::WaitUncounted`",
                        p.entry
                    );
                }
            }
            FenceCheck::Watch { attempt_fn } => {
                assert!(
                    method_calls(entry, attempt_fn) > 0,
                    "{route:?}: `{}` must run its attempt through `{attempt_fn}`",
                    p.entry
                );
                assert_watch_attempt_honours_fence(&layer, attempt_fn);
            }
        }
        if let Some(fallback) = p.fallback {
            let target = proof(fallback).entry;
            assert!(
                method_calls(entry, target) > 0,
                "{route:?}: declares a fallback to {fallback:?} but `{}` never calls `{target}`",
                p.entry
            );
        }
        // A route entry calls no other route's entry unless it is the
        // declared fallback — a fallback is a route, and must be proved.
        for other in ROUTES {
            let target = proof(other).entry;
            if other == route || Some(other) == p.fallback {
                continue;
            }
            assert!(
                method_calls(entry, target) == 0,
                "{route:?}: `{}` calls `{target}` ({other:?}) without declaring it as its fallback",
                p.entry
            );
        }
    }
}

/// `script_fn`'s body invokes the update script and decodes its `FENCED`
/// reply (`decode_fenced_reply`).
fn assert_script_site_decodes_fenced(code: &str, file: &str, script_fn: &str) {
    let body = fn_body(code, script_fn);
    assert!(
        !word_offsets(body, "UPDATE_LAYER_FAST_SCRIPT").is_empty(),
        "`{script_fn}` ({file}) is mapped as a script route but does not invoke \
         UPDATE_LAYER_FAST_SCRIPT"
    );
    assert!(
        body.contains("decode_fenced_reply("),
        "`{script_fn}` ({file}) invokes the update script but never decodes its `FENCED` reply \
         with `decode_fenced_reply`: a commit refused by step 1d would read as a failure, not a \
         fence wait"
    );
}

/// `attempt_fn`'s body `WATCH`es the fence key with the header (in the one
/// `watch_with_baseline` call), runs the fence probe beside its header read
/// (`read_header_commit_token_and_fence`, which runs `probe_fence_on`), and
/// re-probes a `WATCH`-cancelled `EXEC` (`classify_exec_conflict`, which runs
/// `probe_fence`).
fn assert_watch_attempt_honours_fence(layer: &str, attempt_fn: &str) {
    let body = fn_body(layer, attempt_fn);
    assert!(
        squash(body).contains("letfence_key=self.keys.obj_snapshot_fence("),
        "`{attempt_fn}` must bind `fence_key` to the object's snapshot-fence key"
    );
    let watched = call_args(body, "watch_with_baseline")
        .unwrap_or_else(|| panic!("`{attempt_fn}` must WATCH through `watch_with_baseline`"));
    assert!(
        !word_offsets(watched, "fence_key").is_empty()
            && !word_offsets(watched, "header_key").is_empty(),
        "`{attempt_fn}`'s WATCH set must hold the snapshot-fence key with the header: a fence \
         granted after the WATCH must fail the EXEC. Watched: {}",
        squash(watched)
    );
    assert!(
        body.contains("read_header_commit_token_and_fence("),
        "`{attempt_fn}` must probe the fence beside its header read \
         (`read_header_commit_token_and_fence`)"
    );
    assert!(
        fn_body(layer, "read_header_commit_token_and_fence").contains("probe_fence_on("),
        "`read_header_commit_token_and_fence` must run the fence probe (`probe_fence_on`)"
    );
    assert!(
        method_calls(body, "classify_exec_conflict") > 0,
        "`{attempt_fn}` must re-probe a WATCH-cancelled EXEC (`classify_exec_conflict`)"
    );
    assert!(
        fn_body(layer, "classify_exec_conflict").contains("probe_fence("),
        "`classify_exec_conflict` must re-probe the fence (`probe_fence`)"
    );
}

// ── Sites outside the dispatch ───────────────────────────────────────────

/// Every production function in `src/` that invokes the update script is a
/// script route [`proof`] maps, and decodes the script's `FENCED` reply. A new
/// `EVALSHA` site — a route added beside the dispatch rather than in it —
/// fails here until it is a mapped route with a fence check.
#[test]
fn every_update_script_invocation_is_a_mapped_route_that_decodes_fenced() {
    let mut mapped: Vec<(String, String)> = ROUTES
        .iter()
        .filter_map(|&r| match proof(r).fence {
            FenceCheck::Script { file, script_fn } => Some((file.to_owned(), script_fn.to_owned())),
            FenceCheck::Watch { .. } => None,
        })
        .collect();
    mapped.sort();
    let mut sites: Vec<(String, String)> = functions_holding("UPDATE_LAYER_FAST_SCRIPT")
        .into_iter()
        .filter(|(file, f)| !allowlisted(file, f))
        .collect();
    sites.sort();
    for (file, f) in &sites {
        let code = production_code(&read(file));
        let body = fn_body(&code, f);
        assert!(
            body.contains("decode_fenced_reply("),
            "`{f}` ({file}) uses UPDATE_LAYER_FAST_SCRIPT but never decodes its `FENCED` reply \
             with `decode_fenced_reply`, so a commit refused by step 1d would read as a failure, \
             not a fence wait. If it commits, decode `FENCED` and {MAP_OR_ALLOWLIST}"
        );
    }
    let unmapped: Vec<&(String, String)> = sites.iter().filter(|s| !mapped.contains(s)).collect();
    assert!(
        unmapped.is_empty(),
        "these functions use UPDATE_LAYER_FAST_SCRIPT but are not the script route of any \
         `UpdateLayerRoute`: {unmapped:?}. For each, {MAP_OR_ALLOWLIST}"
    );
    assert_eq!(
        sites, mapped,
        "every script route `proof` maps must use UPDATE_LAYER_FAST_SCRIPT itself"
    );
}

/// The update script is reachable only under its own name, from the one
/// `static` that compiles it from the one embedded copy of its source, so
/// [`every_update_script_invocation_is_a_mapped_route_that_decodes_fenced`]
/// sees every invocation. Fails on an `as` rename of `UPDATE_LAYER_FAST_SCRIPT`
/// or `UPDATE_LAYER_FAST_SOURCE`, on either name held by an item other than the
/// known ones in `lua.rs` (a second `static`/`const` alias outside any
/// function), on a second `Script` built from the source, and on a second
/// `include_str!`/`include_bytes!` of `update_layer_fast.lua`.
#[test]
fn the_update_script_is_reachable_only_under_its_own_name() {
    const LUA_RS: &str = "src/storage/lua.rs";
    let item = |label: &str| AnchorSite {
        file: LUA_RS.to_owned(),
        context: Context::Item(label.to_owned()),
        renamed: false,
    };
    let in_fn = |name: &str| AnchorSite {
        file: LUA_RS.to_owned(),
        context: Context::Fn(name.to_owned()),
        renamed: false,
    };

    // `UPDATE_LAYER_FAST_SCRIPT`: functions are the previous test's business;
    // outside them, only un-renamed `use`s and the `static` itself.
    for site in anchor_sites("UPDATE_LAYER_FAST_SCRIPT") {
        assert!(
            !site.renamed,
            "{site:?}: UPDATE_LAYER_FAST_SCRIPT is renamed with `as`; the scans find the update \
             script by its name, so use it unrenamed"
        );
        match &site.context {
            Context::Fn(_) | Context::Use => {}
            Context::Item(_) => assert_eq!(
                site,
                item("static UPDATE_LAYER_FAST_SCRIPT"),
                "UPDATE_LAYER_FAST_SCRIPT is held by an item other than its own `static`: \
                 {ITEM_ALIAS_CAUSES}"
            ),
        }
    }

    // `UPDATE_LAYER_FAST_SOURCE`: exactly the static's initializer, the
    // const's own definition and the compile-time checks of the script's
    // shared prefix — no second `Script::from_lua` anywhere.
    let mut expected = vec![
        item("static UPDATE_LAYER_FAST_SCRIPT"),
        item("const UPDATE_LAYER_FAST_SOURCE"),
        item("const UPDATE_SCRIPT_LOCALS"),
        in_fn("lua_local"),
        in_fn("lua_locals_named"),
    ];
    expected.sort();
    assert_eq!(
        anchor_sites("UPDATE_LAYER_FAST_SOURCE"),
        expected,
        "UPDATE_LAYER_FAST_SOURCE is used outside the known set in lua.rs: a second `Script` \
         compiled from the update script's source would be invisible to the scans. Use \
         UPDATE_LAYER_FAST_SCRIPT, or, for a new compile-time check of the source, add it here"
    );

    // The file is embedded once.
    let mut includes = Vec::new();
    for file in src_rs_files() {
        let src = read(&file);
        let code = production_code(&src);
        let spans = fn_spans(&code);
        for mac in ["include_str", "include_bytes"] {
            for at in word_offsets(&code, mac) {
                let Some(args) = call_args_range(&code[at..], &format!("{mac}!")) else {
                    continue;
                };
                // Offsets match between the blanked code and the source, so
                // the blanked literal is read back from the source.
                if src[at + args.start..at + args.end].contains("update_layer_fast.lua") {
                    includes.push(AnchorSite {
                        file: file.clone(),
                        context: context_of(&code, &spans, at),
                        renamed: false,
                    });
                }
            }
        }
    }
    assert_eq!(
        includes,
        vec![item("const UPDATE_LAYER_FAST_SOURCE")],
        "update_layer_fast.lua must be embedded once, as UPDATE_LAYER_FAST_SOURCE"
    );
}

/// The WATCH-path anchor, `queue_commit_token_record`, is reachable only under
/// its own name, so
/// [`every_watch_path_commit_is_a_mapped_attempt_that_honours_the_fence`] sees
/// every WATCH-path commit. Fails on an `as` rename and on the name held by an
/// item outside a function other than its own definition in `layer.rs` (a
/// `static`/`const` function-pointer alias).
#[test]
fn the_commit_token_helper_is_reachable_only_under_its_own_name() {
    let definition = AnchorSite {
        file: LAYER_RS.to_owned(),
        context: Context::Item("fn queue_commit_token_record".to_owned()),
        renamed: false,
    };
    let sites = anchor_sites("queue_commit_token_record");
    assert!(
        sites.contains(&definition),
        "`fn queue_commit_token_record` is no longer defined at the top level of layer.rs; \
         re-point the WATCH-path scans at the helper that now queues the commit-token record. \
         Found: {sites:?}"
    );
    for site in sites {
        assert!(
            !site.renamed,
            "{site:?}: queue_commit_token_record is renamed with `as`; the WATCH-commit scan finds \
             a WATCH-path commit by this name, so use it unrenamed"
        );
        match &site.context {
            Context::Fn(_) | Context::Use => {}
            Context::Item(_) => assert_eq!(
                site, definition,
                "queue_commit_token_record is held by an item other than its own definition: \
                 {ITEM_ALIAS_CAUSES}"
            ),
        }
    }
}

/// Every [`NON_COMMIT_SITES`] row names a function that still holds a commit
/// anchor and is the only function of that name in its file, and gives a
/// reason, so the allowlist cannot keep excusing a function that was renamed,
/// moved or deleted, nor silently excuse a second same-named function that
/// commits.
#[test]
fn every_non_commit_site_row_matches_a_site() {
    let mut sites = functions_holding("UPDATE_LAYER_FAST_SCRIPT");
    sites.extend(functions_holding("queue_commit_token_record"));
    for (file, function, reason) in NON_COMMIT_SITES {
        assert!(
            sites.contains(&(file.to_string(), function.to_string())),
            "NON_COMMIT_SITES row `{file}` `{function}` matches no function holding a commit \
             anchor; delete the stale row"
        );
        // A row is keyed by name, so it must name exactly one function in its
        // file: `fn_body` fails on a second `fn` of the same name.
        let code = production_code(&read(file));
        fn_body(&code, function);
        assert!(
            !reason.trim().is_empty(),
            "NON_COMMIT_SITES row `{file}` `{function}` gives no reason"
        );
    }
}

/// Every production function in `src/` that queues a WATCH-path update commit
/// is a WATCH attempt [`proof`] maps, and honours the fence. The anchor is
/// `queue_commit_token_record`: every update commit records the call's
/// commit-token record (SR-4.2a), and on the WATCH path this is the helper
/// that queues it into the `MULTI`; a new WATCH…EXEC commit site fails here
/// until it is a mapped route with the fence in its WATCH set, the probe and
/// the re-probe.
#[test]
fn every_watch_path_commit_is_a_mapped_attempt_that_honours_the_fence() {
    let mut mapped: Vec<(String, String)> = ROUTES
        .iter()
        .filter_map(|&r| match proof(r).fence {
            FenceCheck::Watch { attempt_fn } => Some((LAYER_RS.to_owned(), attempt_fn.to_owned())),
            FenceCheck::Script { .. } => None,
        })
        .collect();
    mapped.sort();
    let mut sites: Vec<(String, String)> = functions_holding("queue_commit_token_record")
        .into_iter()
        .filter(|(file, f)| f != "queue_commit_token_record" && !allowlisted(file, f))
        .collect();
    sites.sort();
    let unmapped: Vec<&(String, String)> = sites.iter().filter(|s| !mapped.contains(s)).collect();
    assert!(
        unmapped.is_empty(),
        "these functions queue a WATCH-path update commit but are not the WATCH attempt of any \
         `UpdateLayerRoute`: {unmapped:?}. For each, {MAP_OR_ALLOWLIST}"
    );
    let layer = production_code(&read(LAYER_RS));
    for (_, f) in &sites {
        assert_watch_attempt_honours_fence(&layer, f);
    }
    assert_eq!(
        sites, mapped,
        "every WATCH attempt `proof` maps must queue its commit through \
         `queue_commit_token_record`"
    );
}

// ── The script ────────────────────────────────────────────────────────────

/// The update script's step 1d — the `FENCED` refusal — runs before the mode
/// slot is read and before any write, so it is the same guard in both modes
/// and a fenced call writes nothing. Before it, the script reads the fence key
/// and issues only read commands.
#[test]
fn the_update_scripts_fence_refusal_precedes_the_mode_parse_and_every_write() {
    let raw = read(UPDATE_SCRIPT_LUA);
    // The comment stripper below knows only `--` line comments and quoted
    // strings. A Lua long bracket — a `--[[ … ]]` / `--[=[ … ]=]` comment or a
    // `[[ … ]]` / `[=[ … ]=]` string — can span lines and hide or fake a
    // command, so the script must hold none, and one added later fails here
    // rather than being misread.
    let b = raw.as_bytes();
    for (k, _) in raw.match_indices('[') {
        let mut j = k + 1;
        while b.get(j) == Some(&b'=') {
            j += 1;
        }
        assert!(
            b.get(j) != Some(&b'['),
            "update_layer_fast.lua opens a Lua long bracket at line {}; this guard's comment \
             stripper does not parse long comments or long strings — use `--` comments and \
             quoted strings, or teach the stripper",
            raw[..k].matches('\n').count() + 1
        );
    }
    // Lua comments (`--` to end of line, outside a string) blanked.
    let script: String = raw
        .lines()
        .map(|line| {
            let mut quote: Option<char> = None;
            let mut cut = line.len();
            let mut prev = '\0';
            for (k, c) in line.char_indices() {
                match quote {
                    Some(q) if c == q && prev != '\\' => quote = None,
                    Some(_) => {}
                    None if c == '"' || c == '\'' => quote = Some(c),
                    None if c == '-' && prev == '-' => {
                        cut = k - 1;
                        break;
                    }
                    None => {}
                }
                prev = c;
            }
            &line[..cut]
        })
        .collect::<Vec<_>>()
        .join("\n");
    let squashed = squash(&script);
    let fenced = squashed
        .find("return{\"FENCED\",")
        .expect("the script returns `{\"FENCED\", …}` (step 1d)");
    let mode = squashed
        .find("ARGV[SHARED_ARGV+1]")
        .expect("the script reads its mode slot `ARGV[SHARED_ARGV + 1]`");
    assert!(
        fenced < mode,
        "step 1d's `FENCED` refusal must come before the mode slot is read, so both modes run it"
    );
    let before = &squashed[..fenced];
    assert!(
        before.contains("KEYS[KEY_FENCE]"),
        "the script must read the snapshot-fence key (`KEYS[KEY_FENCE]`) before refusing `FENCED`"
    );
    const READS: &[&str] = &["HGETALL", "HGET", "HMGET", "TYPE", "TIME", "EXISTS"];
    for call in [
        "redis.call(\"",
        "redis.call('",
        "redis.pcall(\"",
        "redis.pcall('",
    ] {
        for (p, _) in before.match_indices(call) {
            let command: String = before[p + call.len()..]
                .chars()
                .take_while(|c| c.is_ascii_alphabetic())
                .collect();
            assert!(
                READS.contains(&command.as_str()),
                "the script issues `{command}` before step 1d's `FENCED` refusal; only reads \
                 ({READS:?}) may precede it, or a fenced call would write"
            );
        }
    }
}

// ── The proving tests exist ───────────────────────────────────────────────

/// Every behavioural test [`proof`] names exists in [`WRITER_TESTS`] as a test
/// function, so a route's mapping cannot cite a test that was renamed or
/// deleted.
#[test]
fn every_route_names_existing_fence_tests() {
    let src = read(WRITER_TESTS);
    let lines: Vec<&str> = src.lines().collect();
    for route in ROUTES {
        let p = proof(route);
        assert!(!p.tests.is_empty(), "{route:?} names no fence test");
        for name in p.tests {
            let at = lines
                .iter()
                .position(|l| {
                    let t = l.trim_start();
                    t.starts_with(&format!("async fn {name}("))
                        || t.starts_with(&format!("fn {name}("))
                })
                .unwrap_or_else(|| panic!("{route:?}: no `fn {name}` in {WRITER_TESTS}"));
            let attrs = &lines[at.saturating_sub(3)..at];
            assert!(
                attrs.iter().any(|l| l.contains("test")),
                "{route:?}: `{name}` in {WRITER_TESTS} is not annotated as a test"
            );
        }
    }
}

// ── The scanner itself ────────────────────────────────────────────────────

#[test]
fn blanking_hides_comments_and_literals_and_keeps_offsets() {
    let src = "fn a<'x>(s: &'x str) -> char { let _ = \"}{ fn b() {\"; // }\n '{' }\n\
               /* fn c() { */ fn d() { r#\"}\"#; }";
    let code = blank_comments_and_literals(src);
    assert_eq!(code.len(), src.len());
    assert_eq!(code.lines().count(), src.lines().count());
    let names: Vec<String> = fn_spans(&code).into_iter().map(|s| s.name).collect();
    assert_eq!(names, ["a", "d"], "{code}");
    assert!(code.contains("'x"), "a lifetime stays code: {code}");

    // An escaped quote closes at its third `'`: read as closing at the
    // escaped one, `'\''` would swallow the next literal's opening quote and
    // expose its `{`, and `a` would never close.
    for src in [
        r"fn a() { let _ = ('\'','{'); } fn d() {}",
        r"fn a() { let _ = '\''; let _ = '}'; } fn d() {}",
    ] {
        let code = blank_comments_and_literals(src);
        let names: Vec<String> = fn_spans(&code).into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["a", "d"], "{code}");
    }
}

#[test]
fn context_of_names_the_enclosing_fn_use_or_item() {
    // An item nested in an `impl` or an inline `mod` reads as that block,
    // `item at line N`, which no caller's known-item set holds: fail closed.
    let src = "use a::{\n    b::{C, D},\n    SCRIPT,\n};\n\
               use a::{B, SCRIPT as S};\npub static SCRIPT: X =\n    X::new(SCRIPT);\n\
               const SRC: &str = SCRIPT;\nfn f() { SCRIPT; }\n\
               impl X {\n    const C: Y = SCRIPT;\n}\n\
               mod m {\n    use a::SCRIPT;\n}\n\
               pub(crate) fn g(_: SCRIPT) {}\n";
    let code = production_code(src);
    let spans = fn_spans(&code);
    let contexts: Vec<Context> = word_offsets(&code, "SCRIPT")
        .into_iter()
        .map(|at| context_of(&code, &spans, at))
        .collect();
    assert_eq!(
        contexts,
        [
            Context::Use,
            Context::Use,
            Context::Item("static SCRIPT".to_owned()),
            Context::Item("static SCRIPT".to_owned()),
            Context::Item("const SRC".to_owned()),
            Context::Fn("f".to_owned()),
            Context::Item("item at line 11".to_owned()),
            Context::Item("item at line 14".to_owned()),
            Context::Item("fn g".to_owned()),
        ]
    );
}

#[test]
fn production_code_drops_cfg_test_items_only() {
    let src = "fn live() { X; }\n#[cfg(test)]\nmod tests { fn t() { X; } }\n\
               #[cfg(any(test, feature = \"f\"))]\nfn seam() { X; }\n#[cfg(test)]\nmod more;\n";
    let code = production_code(src);
    let names: Vec<String> = fn_spans(&code).into_iter().map(|s| s.name).collect();
    assert_eq!(names, ["live", "seam"]);
}

#[test]
fn call_args_reads_the_balanced_argument_list() {
    let code = "let w = watch_with_baseline(&c, &h, &[a(1), b.as_str()]).await;";
    assert_eq!(
        call_args(code, "watch_with_baseline"),
        Some("&c, &h, &[a(1), b.as_str()]")
    );
}
