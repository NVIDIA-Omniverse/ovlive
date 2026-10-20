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

//! Source fence for the rule: **no test running against the shared
//! `TEST_VALKEY_URL` server changes server-wide Valkey configuration.** A
//! configuration change runs only against a test's own server, named by
//! `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL`
//! (`docs-internal/test-writing-guide.md` § Valkey Integration Tests).
//!
//! `CONFIG SET` is server-global: every database and every client on the
//! server sees it, so a test flipping a parameter on the shared server changes
//! it under every other test process and every other suite using that server
//! for as long as the change stands. No in-process lock can serialize that,
//! since nextest runs each test in its own process.
//!
//! The fence walks every `.rs` file under the crate's `src/`, `tests/`,
//! `benches/` and `examples/` and finds each configuration-writing site in
//! code, with comments stripped:
//!
//! - a call of fred's `config_set`, `config_rewrite` or `config_resetstat`
//!   (an identifier match, so `config_get` and a longer identifier such as
//!   `..._config_sets_...` do not count);
//! - a string literal naming the whole command, `"CONFIG SET"`,
//!   `"CONFIG REWRITE"` or `"CONFIG RESETSTAT"` (any case), as a
//!   `CustomCommand` name;
//! - a `"CONFIG"` literal whose subcommand is `"SET"`, `"REWRITE"` or
//!   `"RESETSTAT"` (any case, string or byte string): a `CustomCommand` named
//!   `CONFIG` with the subcommand as its first argument, or the same frame
//!   encoded by hand. The subcommand is the first literal naming a `CONFIG`
//!   subcommand (`GET` included) among the next
//!   [`SPLIT_SUBCOMMAND_TOKEN_WINDOW`] string literals, so a key-`SET` frame
//!   that follows a `CONFIG GET` does not count.
//!
//! A string that merely mentions the command, such as an assertion message
//! (`"CONFIG SET timeout must succeed"`), is not a site. `CONFIG GET` reads
//! and is allowed. `SCRIPT FLUSH` is outside the rule and not flagged: every
//! script call site in service code reloads the script on `NOSCRIPT` and
//! retries (fred's `evalsha_with_reload`,
//! `storage::lua::evalsha_once_with_reload` on the `update_layer` fast path,
//! or `storage::lua::evalsha_reload_rebuilding` on the structural WATCH
//! read's script windows), so a flush costs a concurrent test one reload.
//!
//! Every site must be in a file listed in [`ALLOWED_SITES`], and each listed
//! file must hold exactly the listed number of sites: a fourth site in a file
//! allowed three fails, and so does a removed one, so the count is tightened
//! rather than left as slack for a later addition. A listed file is one whose
//! configuration-changing tests run only against their own server.
//!
//! The scan excludes this file, whose matcher tables spell the literals it
//! looks for. It does not scan `tools/` (the failover harness runs against its
//! own `HARNESS_URL`) or non-Rust files. The discrimination tests at the
//! bottom feed the scanner synthetic sources, and the comment-only real files
//! that name the command are checked to produce no site.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Directories under the crate root the fence walks.
const SCANNED_ROOTS: &[&str] = &["src", "tests", "benches", "examples"];

/// This file, relative to the crate root. Excluded from the walk: its matcher
/// tables and synthetic inputs spell the very literals it looks for.
const THIS_FILE: &str = "tests/integration/storage/shared_valkey_config_fence_test.rs";

/// Files whose configuration-changing tests run only against their own server
/// (`LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL`), with the exact number of sites
/// each holds.
///
/// - `stream_mux_health_test.rs`: the disruptive idle-keepalive test sets the
///   server's `timeout` and restores it (`RestoreIdleTimeout::set`, `restore`
///   and its `Drop`). It is `#[ignore]`d and returns before connecting when
///   `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL` is unset.
/// - `disruptive_valkey.rs`: the `maxmemory` guard of the disruptive
///   real-`-OOM` tests (`store_write_refusal_test.rs`,
///   `oom_collection_test.rs`) sets the server's `maxmemory` and restores it
///   (`RestoreMaxmemory::set`, `restore` and its `Drop`). Every one of those
///   tests is `#[ignore]`d and returns before connecting when
///   `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL` is unset.
const ALLOWED_SITES: &[(&str, usize)] = &[
    ("tests/integration/storage/stream_mux_health_test.rs", 3),
    ("tests/integration/storage/disruptive_valkey.rs", 3),
];

/// fred's configuration-writing methods (`ConfigInterface`).
const CONFIG_WRITE_METHODS: &[&str] = &["config_set", "config_rewrite", "config_resetstat"];

/// `CONFIG` subcommands that change server state.
const CONFIG_WRITE_SUBCOMMANDS: &[&str] = &["SET", "REWRITE", "RESETSTAT"];

/// Every `CONFIG` subcommand name the split rule recognises, reads included:
/// the first of these after a `"CONFIG"` literal is its subcommand.
const CONFIG_SUBCOMMANDS: &[&str] = &["GET", "SET", "REWRITE", "RESETSTAT", "HELP"];

/// How many string literals after a `"CONFIG"` literal the split rule reads
/// looking for its subcommand. The subcommand is the command's first argument,
/// so it is the next literal unless one such as a `CustomCommand` cluster-hash
/// key sits in between, which this window covers. A literal `"GET"` ahead of a
/// key-`SET` frame is excluded by the first-subcommand rule, not by this
/// window. The window matters when the `CONFIG` command has no literal
/// subcommand nearby (it is a variable, say): it keeps an unrelated `"SET"`
/// more than that many literals on from counting as that command's subcommand.
const SPLIT_SUBCOMMAND_TOKEN_WINDOW: usize = 3;

/// Fewer `.rs` files than this means the walk lost its roots (a moved crate
/// root, a wrong `CARGO_MANIFEST_DIR`) and found nothing to check.
const MIN_SCANNED_FILES: usize = 200;

// ═════════════════════════════════════════════════════════════════════════════
// Lexing
// ═════════════════════════════════════════════════════════════════════════════

/// The tokens the scanner cares about. Everything else — punctuation, numbers,
/// lifetimes, char literals, comments — is dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String),
    /// A string or byte-string literal's source text between its quotes,
    /// escapes undecoded.
    Str(String),
}

/// `source`'s identifiers and string literals with their 1-based line numbers,
/// comments (line, block, nested block, doc) stripped.
fn lex(source: &str) -> Vec<(usize, Token)> {
    let chars: Vec<char> = source.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let mut line = 1;
    let at = |k: usize| chars.get(k).copied();

    while i < chars.len() {
        let c = chars[i];
        match c {
            '\n' => {
                line += 1;
                i += 1;
            }
            '/' if at(i + 1) == Some('/') => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '/' if at(i + 1) == Some('*') => {
                let mut depth = 0usize;
                while i < chars.len() {
                    if chars[i] == '/' && at(i + 1) == Some('*') {
                        depth += 1;
                        i += 2;
                    } else if chars[i] == '*' && at(i + 1) == Some('/') {
                        depth -= 1;
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        if chars[i] == '\n' {
                            line += 1;
                        }
                        i += 1;
                    }
                }
            }
            '"' => {
                let start_line = line;
                let (text, next, newlines) = read_quoted(&chars, i + 1);
                out.push((start_line, Token::Str(text)));
                line += newlines;
                i = next;
            }
            '\'' => {
                // A char literal (`'x'`, `'\n'`, `'\''`) or a lifetime (`'a`).
                if at(i + 1) == Some('\\') {
                    // Past the quote, the backslash and the escaped char (which
                    // may itself be a quote), then on to the closing quote.
                    i += 3;
                    while i < chars.len() && chars[i] != '\'' {
                        i += 1;
                    }
                    i += 1;
                } else if at(i + 2) == Some('\'') {
                    i += 3;
                } else {
                    i += 1;
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                // String-literal prefixes: `b"…"`, `c"…"`, raw `r#"…"#` / `br"…"`.
                let raw = matches!(ident.as_str(), "r" | "br" | "cr");
                if raw && matches!(at(i), Some('"') | Some('#')) {
                    let mut hashes = 0;
                    while at(i + hashes) == Some('#') {
                        hashes += 1;
                    }
                    if at(i + hashes) == Some('"') {
                        let start_line = line;
                        let body_start = i + hashes + 1;
                        let mut k = body_start;
                        loop {
                            if k >= chars.len() {
                                break;
                            }
                            if chars[k] == '"' && (1..=hashes).all(|h| at(k + h) == Some('#')) {
                                break;
                            }
                            if chars[k] == '\n' {
                                line += 1;
                            }
                            k += 1;
                        }
                        let text: String = chars[body_start..k.min(chars.len())].iter().collect();
                        out.push((start_line, Token::Str(text)));
                        i = k + 1 + hashes;
                        continue;
                    }
                    // `r#ident`: a raw identifier; the `#` is dropped next turn.
                    continue;
                }
                if matches!(ident.as_str(), "b" | "c") && at(i) == Some('"') {
                    let start_line = line;
                    let (text, next, newlines) = read_quoted(&chars, i + 1);
                    out.push((start_line, Token::Str(text)));
                    line += newlines;
                    i = next;
                    continue;
                }
                out.push((line, Token::Ident(ident)));
            }
            _ => i += 1,
        }
    }
    out
}

/// Read a `"`-delimited literal's body starting at `from` (just past the
/// opening quote). Returns the body, the index past the closing quote and the
/// number of newlines the body spans.
fn read_quoted(chars: &[char], from: usize) -> (String, usize, usize) {
    let mut k = from;
    let mut newlines = 0;
    while k < chars.len() && chars[k] != '"' {
        if chars[k] == '\\' {
            if chars.get(k + 1) == Some(&'\n') {
                newlines += 1;
            }
            k += 2;
            continue;
        }
        if chars[k] == '\n' {
            newlines += 1;
        }
        k += 1;
    }
    let end = k.min(chars.len());
    (chars[from..end].iter().collect(), end + 1, newlines)
}

// ═════════════════════════════════════════════════════════════════════════════
// Sites
// ═════════════════════════════════════════════════════════════════════════════

/// A literal's text with surrounding whitespace trimmed, inner runs collapsed
/// to one space, upper-cased: the form command names compare in.
fn command_words(literal: &str) -> String {
    literal
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_uppercase()
}

/// Whether the subcommand of a `"CONFIG"` literal followed by `rest` is a
/// write: among the next [`SPLIT_SUBCOMMAND_TOKEN_WINDOW`] string literals, the
/// first naming a `CONFIG` subcommand decides. None in reach is not a site.
fn split_subcommand_is_write(rest: &[(usize, Token)]) -> bool {
    rest.iter()
        .filter_map(|(_, token)| match token {
            Token::Str(text) => Some(command_words(text)),
            Token::Ident(_) => None,
        })
        .take(SPLIT_SUBCOMMAND_TOKEN_WINDOW)
        .find(|words| CONFIG_SUBCOMMANDS.contains(&words.as_str()))
        .is_some_and(|sub| CONFIG_WRITE_SUBCOMMANDS.contains(&sub.as_str()))
}

/// The 1-based lines of `source` holding a configuration-writing site (see
/// the module header for the three shapes), sorted, one entry per site.
fn config_write_sites(source: &str) -> Vec<usize> {
    let tokens = lex(source);
    let mut sites = Vec::new();
    for (idx, (line, token)) in tokens.iter().enumerate() {
        match token {
            Token::Ident(ident) if CONFIG_WRITE_METHODS.contains(&ident.as_str()) => {
                sites.push(*line);
            }
            Token::Str(text) => {
                let words = command_words(text);
                let whole_command = CONFIG_WRITE_SUBCOMMANDS
                    .iter()
                    .any(|sub| words == format!("CONFIG {sub}"));
                let split_command =
                    words == "CONFIG" && split_subcommand_is_write(&tokens[idx + 1..]);
                if whole_command || split_command {
                    sites.push(*line);
                }
            }
            Token::Ident(_) => {}
        }
    }
    sites
}

/// The rule's verdict on `sites_by_file` (crate-relative path → site lines):
/// one line per violation, empty when the tree complies.
fn violations(sites_by_file: &BTreeMap<String, Vec<usize>>) -> Vec<String> {
    let mut out = Vec::new();
    for (file, lines) in sites_by_file {
        if lines.is_empty() {
            continue;
        }
        match ALLOWED_SITES.iter().find(|(f, _)| f == file) {
            None => out.extend(lines.iter().map(|l| format!("{file}:{l}"))),
            Some((_, allowed)) if lines.len() != *allowed => out.extend(lines.iter().map(|l| {
                format!(
                    "{file}:{l} (this file is allowed exactly {allowed} site(s) and holds {})",
                    lines.len()
                )
            })),
            Some(_) => {}
        }
    }
    for (file, allowed) in ALLOWED_SITES {
        if sites_by_file.get(*file).is_none_or(Vec::is_empty) {
            out.push(format!(
                "{file}: listed in ALLOWED_SITES with {allowed} site(s) but holds none"
            ));
        }
    }
    out
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.rs` file under `dir`, recursively.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("the fence must be able to list {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("a directory entry must be readable").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Crate-relative `/`-separated path of `path`.
fn relative(path: &Path) -> String {
    path.strip_prefix(crate_root())
        .expect("every scanned file is under the crate root")
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn read_crate_file(relative_path: &str) -> String {
    let path = crate_root().join(relative_path);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("the fence must be able to read {}: {e}", path.display()))
}

// ═════════════════════════════════════════════════════════════════════════════
// The fence
// ═════════════════════════════════════════════════════════════════════════════

/// The tree holds no configuration-writing site outside [`ALLOWED_SITES`], and
/// exactly the listed number inside it.
#[test]
fn shared_valkey_config_fence_tree_changes_config_only_at_allowed_private_server_sites() {
    let root = crate_root();
    let mut files = Vec::new();
    for dir in SCANNED_ROOTS {
        rust_files(&root.join(dir), &mut files);
    }
    assert!(
        files.len() >= MIN_SCANNED_FILES,
        "the fence walked only {} .rs files under {SCANNED_ROOTS:?} of {}; it must see the whole \
         crate to check anything",
        files.len(),
        root.display()
    );

    let mut sites_by_file = BTreeMap::new();
    for path in &files {
        let rel = relative(path);
        if rel == THIS_FILE {
            continue;
        }
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("the fence must be able to read {rel}: {e}"));
        let sites = config_write_sites(&source);
        if !sites.is_empty() {
            sites_by_file.insert(rel, sites);
        }
    }

    let found = violations(&sites_by_file);
    assert!(
        found.is_empty(),
        "server-wide Valkey configuration change outside the private-server allowlist:\n  {}\n\
         `CONFIG SET` / `REWRITE` / `RESETSTAT` changes the whole server, under every other test \
         process and suite sharing it. Only a test running against its own server, named by \
         LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL, may change server configuration, never one on the \
         shared TEST_VALKEY_URL server (docs-internal/test-writing-guide.md § Valkey Integration \
         Tests). A test that reads a parameter uses `config_get`; a test that needs a different \
         server answer can stage it through `ScriptReplyProxy` (as durability_guard_test.rs \
         does). If the site does run only against its own server, list its file in \
         ALLOWED_SITES in {THIS_FILE} with the file's exact site count.",
        found.join("\n  ")
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Discrimination
// ═════════════════════════════════════════════════════════════════════════════

/// One file's sites, as the fence would record them.
fn one_file(file: &str, source: &str) -> BTreeMap<String, Vec<usize>> {
    let mut sites = BTreeMap::new();
    sites.insert(file.to_string(), config_write_sites(source));
    for (allowed_file, allowed) in ALLOWED_SITES {
        if *allowed_file != file {
            sites.insert(allowed_file.to_string(), vec![1; *allowed]);
        }
    }
    sites
}

/// Each configuration-writing shape, in a file not on the allowlist, is a
/// violation naming its `file:line`.
#[test]
fn shared_valkey_config_fence_write_in_unlisted_file_is_a_violation_naming_its_line() {
    const FILE: &str = "tests/integration/storage/some_new_test.rs";
    let shapes = [
        (
            "fred config_set",
            "fn t() {\n    pool.config_set(\"appendonly\", \"no\").await;\n}\n",
        ),
        (
            "turbofish config_set",
            "fn t() {\n    client.config_set::<_, _>(\"timeout\", 0).await;\n}\n",
        ),
        (
            "fred config_rewrite",
            "fn t() {\n    pool.config_rewrite().await;\n}\n",
        ),
        (
            "fred config_resetstat",
            "fn t() {\n    pool.config_resetstat().await;\n}\n",
        ),
        (
            "whole-command CustomCommand",
            concat!(
                "fn t() {\n",
                "    let c = CustomCommand::new_static(\"CONFIG SET\", None::<u16>, false);\n",
                "}\n",
            ),
        ),
        (
            "CustomCommand CONFIG with a SET argument",
            concat!(
                "fn t() {\n",
                "    conn.custom::<(), String>(CustomCommand::new_static(\"CONFIG\", None::<u16>, ",
                "false), vec![\"set\".into(), \"appendonly\".into()]);\n",
                "}\n",
            ),
        ),
        (
            "CustomCommand CONFIG built one statement before its REWRITE argument",
            concat!(
                "fn t() {\n",
                "    let c = CustomCommand::new_static(\"config\", None::<u16>, false);\n",
                "    conn.custom::<(), String>(c, vec![\"REWRITE\".into()]);\n",
                "}\n",
            ),
        ),
        (
            "hand-encoded byte-string frame",
            "fn t() {\n    let frame = encode(&[b\"CONFIG\", b\"RESETSTAT\"]);\n}\n",
        ),
        (
            "CustomCommand CONFIG with a literal cluster-hash key before its SET argument",
            concat!(
                "fn t() {\n",
                "    let c = CustomCommand::new_static(\"CONFIG\", \"{slot}\", false);\n",
                "    conn.custom::<(), String>(c, vec![\"SET\".into(), \"timeout\".into()]);\n",
                "}\n",
            ),
        ),
    ];
    for (shape, source) in shapes {
        let found = violations(&one_file(FILE, source));
        assert_eq!(
            found,
            vec![format!("{FILE}:2")],
            "{shape}: a server configuration change in a file off the allowlist must be one \
             violation naming its file and line"
        );
    }
}

/// One site more than an allowlisted file's count fails, naming every site in
/// the file; so does one fewer, and so does none at all.
#[test]
fn shared_valkey_config_fence_allowed_file_with_a_changed_site_count_is_a_violation() {
    let (file, allowed) = ALLOWED_SITES[0];
    let site = "    client.config_set(\"timeout\", 2).await;\n";
    let source_with = |n: usize| format!("fn t() {{\n{}}}\n", site.repeat(n));

    assert_eq!(
        violations(&one_file(file, &source_with(allowed))),
        Vec::<String>::new(),
        "self-check: the allowed count itself must pass"
    );

    let over = violations(&one_file(file, &source_with(allowed + 1)));
    assert_eq!(
        over.len(),
        allowed + 1,
        "one site more than the allowed {allowed} in {file} must fail, naming each site; got \
         {over:?}"
    );
    assert!(
        over[allowed].starts_with(&format!("{file}:{} ", allowed + 2)),
        "the violation must name the added site's line; got {over:?}"
    );

    let under = violations(&one_file(file, &source_with(allowed - 1)));
    assert_eq!(
        under.len(),
        allowed - 1,
        "one site fewer than the allowed {allowed} in {file} must fail too, so the count is \
         tightened rather than left as slack; got {under:?}"
    );

    // A listed file that holds no site at all (its sites removed, or the file
    // renamed) must fail too: a stale entry would otherwise let `allowed` new
    // sites into that file unseen.
    assert_eq!(
        violations(&one_file(file, "fn t() {}\n")),
        vec![format!(
            "{file}: listed in ALLOWED_SITES with {allowed} site(s) but holds none"
        )],
        "an allowlisted file with no site must be one violation naming the stale entry"
    );
}

/// Mentions of the command in comments and doc comments, assertion messages
/// that merely name it, reads (`config_get`, `CONFIG GET`), longer identifiers
/// containing `config_set`, and `SCRIPT FLUSH` are not sites.
#[test]
fn shared_valkey_config_fence_comments_messages_and_reads_are_not_sites() {
    let source = r##"
//! Module doc: `CONFIG SET appendonly no` would affect every tenant.
/// Doc comment: `pool.config_set("appendonly", "no")`.
// Line comment: "CONFIG" then "SET".
/* Block /* nested */ comment: config_set("x", "y") CONFIG REWRITE */
fn t() {
    let map: HashMap<String, String> = pool.config_get("appendonly").await.expect("CONFIG GET");
    let c = CustomCommand::new_static("CONFIG", None::<u16>, false);
    let _ = conn.custom::<(), String>(c, vec!["GET".into(), "timeout".into()]);
    pool.config_set_like_helper();
    fn apply_tls_host_mapping_tls_scheme_config_sets_explicit_policy() {}
    client.expect("CONFIG SET timeout must succeed");
    let s = r#"pool.config_set("appendonly", "no")"#;
    let ch = '"';
    let q = "a \" quote and CONFIG SET";
    let _: () = pool.script_flush(false).await.expect("SCRIPT FLUSH");
}
"##;
    assert_eq!(
        config_write_sites(source),
        Vec::<usize>::new(),
        "none of these changes server configuration"
    );
}

/// A `"SET"` that belongs to another frame, not to the `CONFIG` command, is
/// not a site. The first two shapes are proxy rules matching `CONFIG GET` and
/// a separate key-`SET` frame, the `"SET"` within reach of the
/// [`SPLIT_SUBCOMMAND_TOKEN_WINDOW`] (and, in the first, 6 lines on, inside
/// the 10-line window an earlier rule used): the literal `"GET"` decides
/// first. In the third the `CONFIG` command's subcommand is a variable, and
/// the key `"SET"` comes 4 literals later, beyond the window.
#[test]
fn shared_valkey_config_fence_config_get_rule_beside_a_key_set_frame_is_not_a_site() {
    let shapes = [
        (
            "CONFIG GET appendonly, then a SET key frame",
            concat!(
                "fn rule(args: &[Vec<u8>]) -> Option<ProxyAction> {\n",
                "    if args[0].eq_ignore_ascii_case(b\"CONFIG\")\n",
                "        && args[1].eq_ignore_ascii_case(b\"GET\")\n",
                "        && args[2].eq_ignore_ascii_case(b\"appendonly\")\n",
                "    {\n",
                "        return Some(stage_appendonly());\n",
                "    }\n",
                "    args[0].eq_ignore_ascii_case(b\"SET\").then(stage_key_write)\n",
                "}\n",
            ),
        ),
        (
            "CONFIG GET with no parameter literal, then a SET key frame",
            concat!(
                "fn rule(args: &[Vec<u8>]) -> Option<ProxyAction> {\n",
                "    let config_get = args[0] == b\"CONFIG\" && args[1] == b\"GET\";\n",
                "    let key_set = args[0] == b\"SET\";\n",
                "    (config_get || key_set).then(stage)\n",
                "}\n",
            ),
        ),
        (
            "CONFIG with a variable subcommand, then a SET key frame 4 literals on",
            concat!(
                "fn t(sub: &str) {\n",
                "    let c = CustomCommand::new_static(\"CONFIG\", None::<u16>, false);\n",
                "    let v: String = conn\n",
                "        .custom(c, vec![sub.into(), \"maxmemory\".into()])\n",
                "        .await\n",
                "        .expect(\"the CONFIG read must answer\");\n",
                "    let key = \"some-key\";\n",
                "    let _: () = conn.custom(set, vec![\"SET\".into(), key.into()]).await;\n",
                "}\n",
            ),
        ),
    ];
    for (shape, source) in shapes {
        assert_eq!(
            config_write_sites(source),
            Vec::<usize>::new(),
            "{shape}: the `\"SET\"` is a separate key frame's command, not the `CONFIG` \
             command's subcommand"
        );
    }
}

/// The real files whose comments discuss `CONFIG SET` on the shared server
/// (they explain why they do not use it) produce no site.
#[test]
fn shared_valkey_config_fence_real_comment_only_mentions_are_not_sites() {
    for file in [
        "tests/integration/checkpoint/restart_storm_harness.rs",
        "tests/integration/storage/update_layer_lua_test.rs",
    ] {
        let source = read_crate_file(file);
        assert!(
            source.contains("CONFIG SET"),
            "self-check: {file} must still mention `CONFIG SET` for this test to mean anything"
        );
        assert_eq!(
            config_write_sites(&source),
            Vec::<usize>::new(),
            "{file} only mentions `CONFIG SET` in comments; it must produce no site"
        );
    }
}
