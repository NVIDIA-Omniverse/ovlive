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

// Guard tests: scripts/guard-tests.sh runs this whole module by name; rename it there too.

//! Source fence for the rule: **a test that needs Valkey connects its bare
//! client through `live_edit::storage::test_support::connect_test_client_or_skip`,
//! or fails the test on a failed connect in the connect's own statement; it
//! never turns an unreachable store into a skip of its own.**
//!
//! CI always sets `TEST_VALKEY_URL`, and nextest prints nothing for a passing
//! test, so a test that returns early when its connect fails or is slow passes
//! with none of its assertions run, and nobody sees it. The helper fails the
//! test when the variable is set (or under `GITLAB_CI`), waits out a slow
//! store under its hang guard, and skips only a local run with the variable
//! unset.
//!
//! The fence walks every `.rs` file under the crate's `src/`, `tests/`,
//! `examples/` and `benches/`, except `src/storage/test_support.rs` (the
//! helper's home, whose own connects turn a failure into its skip or its
//! panic) and this file (whose synthetic inputs spell the shapes it flags).
//! Comments are stripped first, by a one-pass lexer ([`code_text`]) that also
//! knows strings, so a `//` in a URL is not a comment. Two rules:
//!
//! - **R1, an unwrapped connect** (under `src/` and `tests/`). The calls
//!   checked are `connect_test_client(` and
//!   `connect_test_client_with_credential_provider(`, matched as whole
//!   identifiers (so `connect_test_client_or_skip(` is not one), and, under
//!   `tests/` only, fred's own connects `.init()` and `.wait_for_connect()`
//!   when the next token is `.await` or `)` (so a synchronous
//!   `tracing_subscriber::fmt().init();` is not one). A call's statement runs
//!   from the call to the first `;` at or below the call's brace depth, or to
//!   a `}` that closes the block holding the call and is not followed by `.`
//!   or `?`: `match provider { … => connect_test_client(…).await, … }.expect(…)`
//!   is one statement. Braces are counted with string and char literal
//!   contents blanked, since messages carry `{e}`. The call is a violation
//!   when its statement holds none of `.expect(`, `.expect_err(`,
//!   `.unwrap()`, `.unwrap_err()` or `panic!(`: a `match` that maps a failure
//!   to `None`, a `let Ok(..) = … else { return; }`, a `.ok()` / `.ok()?`.
//!   A call in the condition or scrutinee of an `if`, `while` (`let` or not)
//!   or `match` is held to a narrower span: the keyword is found by looking
//!   back to the statement's start (the nearest `;`, `{` or `}`), and the
//!   call is a violation unless an unwrap sits between it and the block's
//!   opening `{`, so an `.expect(` in the block's body, or after it, does not
//!   count (`if let Ok(c) = connect_test_client(…).await { …expect… }`).
//! - **R2, a reachability skip line** (all four trees). A code line holding
//!   `SKIP` (that case only) and `reachab` (any case), the text of the skip
//!   lines the old helpers printed (`SKIP: Valkey not reachable`).
//!
//! Positive controls: the walk finds more than 100 files, R1 finds more than
//! [`MIN_CHECKED_CONNECTS`] calls to check, and `test_support.rs` still defines
//! `pub async fn connect_test_client_or_skip`, so the fence fails rather than
//! passes when the tree, its needles or the helper move.
//!
//! Out of scope: `ValkeyClient::connect(` calls, which
//! `scripts/check-valkey-connect-allowlist.sh` confines to a few files (the
//! direct calls in `client_test.rs` are failure-path tests whose `Err` is the
//! subject); the examples' and benches' connects (R2 only reaches there); skip
//! lines for other reasons (a URL's shape, ACL support, an unset cluster
//! variable), none of which say "reachab".
//!
//! What the fence cannot see, or reads wrongly (each left unhandled to keep
//! the scanner simple):
//!
//! - a statement, other than a condition or scrutinee, that panics on one
//!   failure path and returns on another (the statement holds a `panic!(`, so
//!   it passes);
//! - more generally, a call held to its whole statement passes when anything
//!   in that statement unwraps, even something other than the connect: a
//!   let-else whose `else` block unwraps another value before returning
//!   (`let Ok(c) = connect_test_client(&cfg).await else { other().await.expect("x"); return; };`),
//!   or a `for` loop over the connect
//!   (`for c in connect_test_client(&cfg).await { … }`: the loop's `}` does
//!   not end the statement, so an unwrap in its body or in the statement
//!   after it passes the call);
//! - likewise a call in a `match` guard
//!   (`Some(_) if connect_test_client(…).await.is_ok() => …`): its "block's
//!   `{`" is whatever `{` comes next, so its span can run into later arms, and
//!   an unwrap there passes the call;
//! - a condition or scrutinee whose arms, body or `else` do fail the test
//!   (`match connect_test_client(…).await { Ok(c) => c, Err(e) => panic!(…) }`,
//!   `if let Ok(c) = connect_test_client(…).await { … } else { panic!(…) }`)
//!   is flagged: unwrap the connect before the block instead;
//! - the look-back for `if` / `while` / `match` stops at the nearest `;`,
//!   `{` or `}`, so a keyword before a closure body or a block inside the
//!   condition (`if let Ok(c) = f(|| { … }, connect_test_client(…))`) is not
//!   seen, and that call is held to its whole statement;
//! - a raw fred client connected without `.init()` or `.wait_for_connect()`,
//!   or a `.init()` / `.wait_for_connect()` whose next token is neither
//!   `.await` nor `)`;
//! - anything inside `src/storage/test_support.rs`;
//! - a skip message split across lines, or one worded without `SKIP` and
//!   "reachab";
//! - a reachability check done some other way, such as a raw TCP probe;
//! - a connect wrapped in an expression whose closing `}` is followed by `)`
//!   before the unwrap (`f(match … { … }).expect(…)`): the statement ends at
//!   that `}`, so such a call is flagged although it fails the test;
//! - a `;` inside the call's own parentheses with no brace around it (an array
//!   repeat `[x; n]`), which ends the statement early (outside a condition or
//!   scrutinee, where [`condition_block_open`] skips it);
//! - a char literal is recognised only as one character or a short escape
//!   (up to `'\u{10ffff}'`); a `'` in any other shape is read as a lifetime,
//!   so what follows it is read as code.

use std::path::{Path, PathBuf};

use super::b_289_tests::rust_sources_under;

/// Directories under the crate root R2 walks.
const SKIP_LINE_ROOTS: &[&str] = &["src", "tests", "examples", "benches"];

/// Directories under the crate root R1 walks.
const CONNECT_ROOTS: &[&str] = &["src", "tests"];

/// The directory under which R1 also checks fred's own connects.
const FRED_CONNECT_ROOT: &str = "tests";

/// The helper's home, excluded from both rules.
const HELPER_FILE: &str = "src/storage/test_support.rs";

/// This file, excluded from both rules: its synthetic inputs spell the shapes
/// it flags.
const THIS_FILE: &str = "tests/integration/storage/test_valkey_connect_skip_guard_test.rs";

/// The helper the rule points at, as `test_support.rs` must still define it.
const HELPER_DEFINITION: &str = "pub async fn connect_test_client_or_skip(";

/// The connect helpers whose result R1 holds to an unwrap, as whole
/// identifiers followed by their `(`.
const CONNECT_HELPERS: &[&str] = &[
    "connect_test_client(",
    "connect_test_client_with_credential_provider(",
];

/// fred's own connects, checked under [`FRED_CONNECT_ROOT`] when the next
/// token is `.await` or `)`.
const FRED_CONNECTS: &[&str] = &[".init()", ".wait_for_connect()"];

/// What makes a connect's statement fail the test on a failed connect.
const FAILS_THE_TEST: &[&str] = &[
    ".expect(",
    ".expect_err(",
    ".unwrap()",
    ".unwrap_err()",
    "panic!(",
];

/// Fewer calls than this for R1 to check means its needles no longer match
/// the tree (it checked 94 when it was written).
const MIN_CHECKED_CONNECTS: usize = 50;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `/`-separated form of a crate-relative path.
fn slash_path(path: &Path) -> String {
    path.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// `code` from `from` on, past any leading ASCII whitespace.
fn after_whitespace(code: &str, from: usize) -> &str {
    code.get(from..).unwrap_or_default().trim_start()
}

/// What [`code_text`] does with the contents of string and char literals.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Literals {
    /// Keep them: R2's skip lines are string literals.
    Keep,
    /// Replace them with spaces: R1 counts braces and looks for needles in
    /// code, and messages carry `{e}` and name the calls.
    Blank,
}

/// `source` with every comment replaced by spaces and, with
/// [`Literals::Blank`], the contents of every string and char literal too.
/// Each replaced byte becomes one space and newlines are kept, so byte
/// offsets and line numbers survive.
///
/// One pass over the whole file, so a `//` inside a string (a URL) is not a
/// comment and a string spanning lines stays one string. Handles `//` and
/// nested `/* */` comments, strings and byte strings with their escapes, raw
/// strings (`r"…"`, `br#"…"#`) to their own terminator, and char literals of
/// one character or a short escape (`'"'`, `'{'`, `'\''`, `'\u{7f}'`); any
/// other `'` is a lifetime or a label.
///
/// A copy, rewritten in places, of `blank_comments_and_literals` in
/// `src/storage/layer/fence_route_guard.rs` (private to that module; the
/// project pattern is to duplicate these few lines rather than hoist them),
/// with a mode that keeps literals added for R2.
fn code_text(source: &str, literals: Literals) -> String {
    let src = source.as_bytes();
    let len = src.len();
    let mut out = src.to_vec();
    let blank = |out: &mut [u8], from: usize, to: usize| {
        for byte in &mut out[from.min(len)..to.min(len)] {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    };
    let blank_literal = |out: &mut [u8], from: usize, to: usize| {
        if literals == Literals::Blank {
            blank(out, from, to);
        }
    };
    let mut i = 0;
    while i < len {
        let next = src.get(i + 1).copied();
        match src[i] {
            b'/' if next == Some(b'/') => {
                let end = src[i..]
                    .iter()
                    .position(|&b| b == b'\n')
                    .map_or(len, |p| i + p);
                blank(&mut out, i, end);
                i = end;
            }
            b'/' if next == Some(b'*') => {
                let mut depth = 0usize;
                let mut end = i;
                while end < len {
                    match (src[end], src.get(end + 1)) {
                        (b'/', Some(b'*')) => {
                            depth += 1;
                            end += 2;
                        }
                        (b'*', Some(b'/')) => {
                            depth -= 1;
                            end += 2;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => end += 1,
                    }
                }
                blank(&mut out, i, end);
                i = end;
            }
            b'"' => {
                let mut end = i + 1;
                while end < len && src[end] != b'"' {
                    end += if src[end] == b'\\' { 2 } else { 1 };
                }
                blank_literal(&mut out, i + 1, end);
                i = end + 1;
            }
            b'r' if starts_a_token(src, i) => {
                let mut quote = i + 1;
                while src.get(quote) == Some(&b'#') {
                    quote += 1;
                }
                if src.get(quote) != Some(&b'"') {
                    i += 1;
                    continue;
                }
                let hashes = &src[i + 1..quote];
                let mut end = quote + 1;
                while end < len && !(src[end] == b'"' && src[end + 1..].starts_with(hashes)) {
                    end += 1;
                }
                blank_literal(&mut out, quote + 1, end);
                i = end + 1 + hashes.len();
            }
            b'\'' => {
                let close = if next == Some(b'\\') {
                    src[(i + 3).min(len)..(i + 12).min(len)]
                        .iter()
                        .position(|&b| b == b'\'')
                        .map(|p| i + 3 + p)
                } else {
                    let width = source[i + 1..].chars().next().map_or(0, char::len_utf8);
                    (width > 0 && src.get(i + 1 + width) == Some(&b'\'')).then_some(i + 1 + width)
                };
                match close {
                    Some(close) => {
                        blank_literal(&mut out, i + 1, close);
                        i = close + 1;
                    }
                    None => i += 1,
                }
            }
            _ => i += 1,
        }
    }
    // Every replaced run is a whole comment or a whole literal body, each of
    // which holds its multi-byte characters whole, so the text stays UTF-8.
    String::from_utf8(out).expect("blanking whole comments and literals keeps the text UTF-8")
}

/// Whether the `r` at `at` can open a raw string: not inside an identifier,
/// except as `br`.
fn starts_a_token(src: &[u8], at: usize) -> bool {
    let before = |n: usize| at.checked_sub(n).map(|i| src[i]);
    match before(1) {
        None => true,
        Some(b'b') => !before(2).is_some_and(is_identifier_byte),
        Some(byte) => !is_identifier_byte(byte),
    }
}

/// Byte offsets in `code` (literals blanked) of every call R1 checks.
fn connect_calls(code: &str, fred_connects: bool) -> Vec<usize> {
    let mut calls = Vec::new();
    for helper in CONNECT_HELPERS {
        calls.extend(
            code.match_indices(helper)
                .map(|(at, _)| at)
                .filter(|&at| at == 0 || !is_identifier_byte(code.as_bytes()[at - 1])),
        );
    }
    if fred_connects {
        for connect in FRED_CONNECTS {
            calls.extend(code.match_indices(connect).map(|(at, _)| at).filter(|&at| {
                let next = after_whitespace(code, at + connect.len());
                next.starts_with(".await") || next.starts_with(')')
            }));
        }
    }
    calls.sort_unstable();
    calls
}

/// The statement of the call at `at`: to the first `;` at or below the call's
/// brace depth, or to a `}` closing the block that holds the call when the
/// next token is not `.` or `?` (a block value chained onward stays in the
/// statement, and the block it closes into becomes the call's).
fn statement(code: &str, at: usize) -> &str {
    let mut depth = 0i32;
    let mut floor = 0i32;
    for (offset, byte) in code.as_bytes()[at..].iter().enumerate() {
        let i = at + offset;
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth < floor {
                    if after_whitespace(code, i + 1).starts_with(['.', '?']) {
                        floor = depth;
                    } else {
                        return &code[at..=i];
                    }
                }
            }
            b';' if depth <= floor => return &code[at..=i],
            _ => {}
        }
    }
    &code[at..]
}

/// When the call at `at` is in the condition or scrutinee of an `if`, a
/// `while` (`let` or not) or a `match`, the offset of the block's opening
/// `{`; otherwise `None`. The keyword is looked for as a whole word between
/// the statement's start (just past the nearest `;`, `{` or `}` before the
/// call) and the call; the block's `{` is the first one after the call
/// outside its parentheses. A struct literal cannot stand unparenthesised in
/// a condition or scrutinee, so that `{` is the block's. A `;` outside the
/// parentheses before any such `{` means the call was not in a condition
/// after all, and gives `None`; a `;` inside them (a closure body in the
/// condition, `map_err(|e| { log(e); e })`) does not.
fn condition_block_open(code: &str, at: usize) -> Option<usize> {
    let start = code[..at].rfind([';', '{', '}']).map_or(0, |i| i + 1);
    let in_condition = code[start..at]
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|word| matches!(word, "if" | "while" | "match"));
    if !in_condition {
        return None;
    }
    let mut parens = 0i32;
    for (offset, byte) in code.as_bytes()[at..].iter().enumerate() {
        match byte {
            b'(' => parens += 1,
            b')' => parens -= 1,
            b'{' if parens <= 0 => return Some(at + offset),
            b';' if parens <= 0 => return None,
            _ => {}
        }
    }
    None
}

/// R1 over one file: every checked call whose statement does not fail the
/// test, as `line: text`, and how many calls were checked. A call in a
/// condition or scrutinee ([`condition_block_open`]) is held to the span from
/// the call to its block's `{`, so an unwrap inside the block does not count.
fn unwrapped_connects(source: &str, fred_connects: bool) -> (Vec<String>, usize) {
    let code = code_text(source, Literals::Blank);
    let lines: Vec<&str> = source.lines().collect();
    let calls = connect_calls(&code, fred_connects);
    let unwrapped = calls
        .iter()
        .filter(|&&at| {
            let span = match condition_block_open(&code, at) {
                Some(open) => &code[at..open],
                None => statement(&code, at),
            };
            !FAILS_THE_TEST.iter().any(|fails| span.contains(fails))
        })
        .map(|&at| {
            let index = code[..at].matches('\n').count();
            format!("{}: {}", index + 1, lines[index].trim())
        })
        .collect();
    (unwrapped, calls.len())
}

/// The byte offset of the start of every line of `text` holding `SKIP` and
/// `reachab` (any case), found from the `SKIP`s rather than line by line,
/// since the scan reads every file and most hold none.
fn skip_line_starts(text: &str) -> Vec<usize> {
    let mut starts: Vec<usize> = text
        .match_indices("SKIP")
        .map(|(at, _)| text[..at].rfind('\n').map_or(0, |newline| newline + 1))
        .collect();
    starts.dedup();
    starts.retain(|&start| {
        let line = text[start..].lines().next().unwrap_or_default();
        line.to_lowercase().contains("reachab")
    });
    starts
}

/// R2 over one file: every code line holding `SKIP` and `reachab` (any
/// case), as `line: text`.
fn reachability_skip_lines(source: &str) -> Vec<String> {
    let code = code_text(source, Literals::Keep);
    skip_line_starts(&code)
        .into_iter()
        .map(|start| {
            let number = code[..start].matches('\n').count() + 1;
            let line = code[start..].lines().next().unwrap_or_default();
            format!("{number}: {}", line.trim())
        })
        .collect()
}

/// No file outside the helper's home connects to Valkey without failing the
/// test on a failed connect, and none prints a skip for an unreachable store.
#[test]
fn no_test_turns_an_unreachable_valkey_into_a_skip_of_its_own() {
    let root = crate_root();
    let mut files = Vec::new();
    for dir in SKIP_LINE_ROOTS {
        let dir = root.join(dir);
        if dir.is_dir() {
            rust_sources_under(&root, &dir, &mut files);
        }
    }
    assert!(
        files.len() > 100,
        "the walk found only {} file(s); it is not reading the tree",
        files.len()
    );

    let mut unwrapped = Vec::new();
    let mut skip_lines = Vec::new();
    let mut checked = 0;
    let mut helper_defined = false;
    for file in &files {
        let name = slash_path(file);
        let source = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("the fence must be able to read {name}: {e}"));
        if name == HELPER_FILE {
            helper_defined = source.contains(HELPER_DEFINITION);
            continue;
        }
        if name == THIS_FILE {
            continue;
        }
        let tree = name.split('/').next().unwrap_or_default();
        let fred_connects = tree == FRED_CONNECT_ROOT;
        // Most files hold no needle at all, and these raw-text checks keep
        // the lexer off them: a file none of them is found in has no site.
        // `find`, not `contains`: in the unoptimised test build
        // `str::contains` searched this tree several times slower.
        let fred_needles: &[&str] = if fred_connects { FRED_CONNECTS } else { &[] };
        let may_connect = CONNECT_ROOTS.contains(&tree)
            && CONNECT_HELPERS
                .iter()
                .chain(fred_needles)
                .any(|needle| source.find(needle).is_some());
        let may_skip = !skip_line_starts(&source).is_empty();
        if may_connect {
            let (found, calls) = unwrapped_connects(&source, fred_connects);
            checked += calls;
            unwrapped.extend(found.into_iter().map(|site| format!("{name}:{site}")));
        }
        if may_skip {
            skip_lines.extend(
                reachability_skip_lines(&source)
                    .into_iter()
                    .map(|line| format!("{name}:{line}")),
            );
        }
    }

    assert!(
        helper_defined,
        "{HELPER_FILE} no longer defines `{HELPER_DEFINITION}`; the helper moved or was \
         renamed, and this fence must follow it"
    );
    assert!(
        checked > MIN_CHECKED_CONNECTS,
        "R1 checked only {checked} connect call(s); its needles no longer match the tree"
    );
    // One report for both rules, so a site that breaks both is named by both.
    let mut report = Vec::new();
    if !unwrapped.is_empty() {
        report.push(format!(
            "R1: these Valkey connects do not fail the test when the connect fails, so an \
             unreachable or slow store becomes a silent pass. Connect through \
             `test_support::connect_test_client_or_skip` (a two-client test joins two calls), \
             or `.expect` the connect in its own statement:\n{}",
            unwrapped.join("\n")
        ));
    }
    if !skip_lines.is_empty() {
        report.push(format!(
            "R2: these lines print a skip for an unreachable Valkey. With TEST_VALKEY_URL set \
             an unreachable store must fail the test; \
             `test_support::connect_test_client_or_skip` prints the only skip line, for a \
             local run with the variable unset:\n{}",
            skip_lines.join("\n")
        ));
    }
    assert!(report.is_empty(), "{}", report.join("\n\n"));
}

/// R1 on synthetic sources: each shape the old helpers used is flagged, and
/// each shape that fails the test, or is not a store connect, is not.
#[test]
fn the_unwrapped_connect_scanner_flags_each_old_shape_and_passes_each_unwrapping_one() {
    let flagged = [
        // The per-file helper: a failure or the outer timeout maps to `None`.
        "async fn maybe_connect() -> Option<ValkeyClient> {\n    match tokio::time::timeout(\n        Duration::from_secs(5),\n        connect_test_client(&cfg),\n    )\n    .await\n    {\n        Ok(Ok(c)) => Some(c),\n        _ => None,\n    }\n}",
        // `let Ok(Ok(..)) else` with a skip line.
        "let Ok(Ok(client)) = tokio::time::timeout(Duration::from_secs(5), connect_test_client(&cfg)).await else {\n    eprintln!(\"SKIP: {e}\");\n    return;\n};",
        // `let Ok(..) else` on the connect itself.
        "let Ok(client) = connect_test_client(&cfg).await else {\n    eprintln!(\"SKIP\");\n    return;\n};",
        // The silent let-else: no line printed at all.
        "let Ok(client) = connect_test_client_with_credential_provider(&cfg, provider).await else { return; };",
        // A raw fred pool's `.init()` under a timeout, mapped to `None`.
        "let connected = tokio::time::timeout(Duration::from_secs(5), pool.init()).await;\nmatch connected {\n    Ok(Ok(_)) => Some(pool),\n    _ => None,\n}",
        // `.ok()` and `.ok()?`.
        "let client = connect_test_client(&cfg).await.ok();",
        "let client = connect_test_client(&cfg).await.ok()?;",
        // fred's `.wait_for_connect()`, discarded.
        "let _ = client.wait_for_connect().await;",
        // A statement ends at its terminator: an unwrap in the code after it
        // (past the `;`, or past the `}` closing the block the call is in)
        // does not count.
        "let client = connect_test_client(&cfg).await.ok();\nlet other = other().await.expect(\"x\");",
        "async fn connect() -> Result<ValkeyClient, ConnectError> {\n    connect_test_client(&cfg).await\n}\n\nasync fn t() { let c = connect().await.expect(\"store\"); }",
        // A connect in a condition or scrutinee: an unwrap in the block's
        // body, or after it, does not unwrap the connect.
        "if let Ok(c) = connect_test_client(&cfg).await {\n    run(c).await.expect(\"x\");\n}",
        "match connect_test_client(&cfg).await {\n    Ok(c) => run(c).await.expect(\"x\"),\n    Err(_) => return,\n}",
        "while let Err(e) = pool.init().await {\n    retry(e).await.expect(\"x\");\n}",
        "if connect_test_client(&cfg).await.is_err() {\n    return;\n}\nrun().await.expect(\"x\");",
        // The call nested in the condition's parentheses: the block's `{`
        // comes after the `)` closing `timeout(…)`, below the call's own
        // paren depth.
        "if let Ok(Ok(c)) = tokio::time::timeout(B, connect_test_client(&cfg)).await {\n    run(c).await.expect(\"x\");\n}",
        // A `;` inside a closure in the condition is not the statement's end.
        "if let Ok(c) = connect_test_client(&cfg).await.map_err(|e| { log(e); e }) {\n    run(c).await.expect(\"x\");\n}",
    ];
    // Every case is checked before failing, so a red run names each missed one.
    let missed: Vec<&str> = flagged
        .into_iter()
        .filter(|source| {
            let (unwrapped, calls) = unwrapped_connects(source, true);
            unwrapped.len() != 1 || calls != 1
        })
        .collect();
    assert!(
        missed.is_empty(),
        "each of these has one checked call, which must be flagged:\n{}",
        missed.join("\n---\n")
    );

    let passed = [
        "let client = connect_test_client(&cfg).await.expect(\"connect\");",
        // A timeout around the connect, with an `expect` for each layer.
        "let client = tokio::time::timeout(B, connect_test_client(&cfg))\n    .await\n    .expect(\"connect within {B:?}\")\n    .expect(\"connect\");",
        // One statement across a block chained onward: both calls are in it.
        "let client = match provider {\n    Some(p) => connect_test_client_with_credential_provider(&config, p).await,\n    None => connect_test_client(&config).await,\n}\n.expect(\"the client must connect\");",
        "let client = connect_test_client(&cfg)\n    .await\n    .unwrap_or_else(|e| panic!(\"connect {}: {e}\", cfg.valkey_url));",
        "let err = connect_test_client(&cfg).await.expect_err(\"the connect must fail\");",
        // A tail expression: the fn's closing `}` ends the statement.
        "async fn connect() -> ValkeyClient {\n    connect_test_client(&config)\n        .await\n        .expect(\"connect\")\n}",
        "pool.init().await.expect(\"the pool connects\");",
        "tokio::time::timeout(BOUND, pool.wait_for_connect())\n    .await\n    .expect(\"within the bound\")\n    .expect(\"connects\");",
        // A brace inside a message does not end the statement early.
        "let client = connect_test_client(&cfg).await.unwrap_or_else(|e| {\n    panic!(\"connect to {} failed: {e}\", \"}\")\n});",
        // A condition on a value the connect's unwrap already produced, and a
        // connect inside an `if` block rather than in its condition.
        "if let Some(x) = connect_test_client(&cfg).await.expect(\"store\").thing() {\n    use_it(x);\n}",
        "if ready {\n    let c = connect_test_client(&cfg).await.expect(\"store\");\n}",
        // A closure's `{` inside the condition's parentheses is not the
        // block's: the unwrap inside it still sits before the block.
        "if let Some(x) = connect_test_client(&cfg).await.unwrap_or_else(|e| { panic!(\"{e}\") }).thing() {\n    use_it(x);\n}",
        // Nor does a `;` in a raw string or a char literal.
        "let client = connect_test_client(&named(r#\"a\"; b\"#)).await.expect(\"connect\");",
        "let client = connect_test_client(&split(';', '{')).await.expect(\"connect\");",
        // A `//` inside a string spanning lines is not a comment.
        "let client = connect_test_client(&cfg)\n    .await\n    .unwrap_or_else(|e| panic!(\"connect to redis://valkey:6379\n        failed: {e}\"));",
    ];
    for source in passed {
        let (unwrapped, calls) = unwrapped_connects(source, true);
        assert!(calls >= 1, "a call is checked in {source:?}");
        assert!(
            unwrapped.is_empty(),
            "nothing is flagged in {source:?}: {unwrapped:?}"
        );
    }

    let not_checked = [
        // The helper itself, by its whole identifier.
        "let Some(client) = connect_test_client_or_skip(&cfg).await else {\n    return;\n};",
        // Comments are stripped.
        "// let client = connect_test_client(&cfg).await.ok();\n/* pool.init().await.ok(); */",
        // A message naming the call is not a call.
        "panic!(\"connect_test_client(&cfg) returned {e}\");",
        // A synchronous `init()`: the next token is `;`.
        "tracing_subscriber::fmt().init();",
        // A longer identifier ending in the helper's name.
        "my_connect_test_client(&cfg).await.ok();",
    ];
    for source in not_checked {
        assert_eq!(
            unwrapped_connects(source, true),
            (Vec::new(), 0),
            "no call is checked in {source:?}"
        );
    }

    // fred's own connects are checked under `tests/` only.
    assert_eq!(
        unwrapped_connects("let _ = pool.init().await;", false),
        (Vec::new(), 0),
        "outside tests/, `.init()` is not a checked call"
    );

    assert_eq!(
        unwrapped_connects(
            "fn f() {}\nlet client = connect_test_client(&cfg)\n    .await\n    .ok();",
            true
        )
        .0,
        vec!["2: let client = connect_test_client(&cfg)".to_string()],
        "a flagged call is reported on the line it starts on"
    );
}

/// R2 on synthetic sources: the old skip lines are flagged; other skip
/// reasons, other uses of the words and comments are not.
#[test]
fn the_reachability_skip_scanner_flags_the_old_skip_lines_only() {
    for source in [
        "eprintln!(\"SKIP: Valkey not reachable\");",
        "eprintln!(\"SKIP: second Valkey connection not reachable\");",
        "eprintln!(\"SKIP: Valkey unreachable at {url}\");",
        // A `//` in the message's URL is not a comment.
        "eprintln!(\"SKIP: redis://valkey:6379 is not reachable\");",
    ] {
        assert_eq!(
            reachability_skip_lines(source).len(),
            1,
            "flagged: {source:?}"
        );
    }
    for source in [
        // Lower-case `skip` is a word in ordinary code and messages.
        "AuditSkip::Unreachable => {}",
        "unreachable!(\"data messages are skipped above\")",
        // Other skip reasons.
        "eprintln!(\"SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set\");",
        "eprintln!(\"SKIP: TEST_VALKEY_URL is not a bare redis://host:port\");",
        // A comment.
        "// eprintln!(\"SKIP: Valkey not reachable\");",
    ] {
        assert!(
            reachability_skip_lines(source).is_empty(),
            "not flagged: {source:?}"
        );
    }
    assert_eq!(
        reachability_skip_lines("fn f() {}\n\n    eprintln!(\"SKIP: Valkey not reachable\");"),
        vec!["3: eprintln!(\"SKIP: Valkey not reachable\");".to_string()],
        "a flagged line is reported by its number"
    );
}
