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

//! The binary's key schema and its test-only prefix override,
//! `LIVE_EDIT_TEST_ONLY_KEY_PREFIX` (`live_edit::config::key_prefix`).
//!
//! Source scans, each a guard pinned by name in `scripts/guard-tests.sh`:
//!
//! - **A release build cannot read the override.** In
//!   `src/config/key_prefix.rs` the variable's name appears as a quoted
//!   literal once, in the `test-support`-gated const; every code line naming
//!   it is inside an item gated on `test-support`; the build without the
//!   feature has exactly one `binary_key_schema_from_env`, whose whole body is
//!   `Ok(KeySchema::default())`; and no other source file names the variable
//!   outside a comment. The measurement-feature fence
//!   (`scripts/check-measurement-features.sh`, and the Dockerfile's gate on
//!   the release binary) checks the built binaries do not carry the name.
//! - **Every key the binary builds is under the one schema `main` hands
//!   `run`.** `src/main.rs` builds no `KeySchema` itself: it calls
//!   `binary_key_schema_from_env()` once, before it calls `run`, which takes
//!   the schema as a parameter. Outside test code, no file but
//!   `src/config/key_prefix.rs` calls `KeySchema::default`, and none but it
//!   and the `test-support` fixtures (`src/storage/test_support.rs`) calls
//!   `KeySchema::new`; so a held binary cannot build the unprefixed
//!   fleet-wide marker key under a test prefix.
//!
//! The scans read code lines only (a `//` comment cannot satisfy or fail
//! them) and leave out items gated on `test` (`#[cfg(test)]`,
//! `#[cfg(all(test, …))]`), and the files of test modules declared out of
//! line. They do not see a schema built through `Default::default()` on a
//! field whose type is inferred.
//!
//! `test_only_key_schema`'s refusals are pinned at the end, in a
//! `test-support` build.

use std::path::{Path, PathBuf};

const KEY_PREFIX: &str = include_str!("../../src/config/key_prefix.rs");
const MAIN: &str = include_str!("../../src/main.rs");

/// The variable the override is read from.
const VARIABLE: &str = "LIVE_EDIT_TEST_ONLY_KEY_PREFIX";

const CFG_TEST_SUPPORT: &str = r#"#[cfg(feature = "test-support")]"#;
const CFG_NOT_TEST_SUPPORT: &str = r#"#[cfg(not(feature = "test-support"))]"#;

/// `src`'s lines with their indices, `//` comment lines left out.
fn code_lines(src: &str) -> impl Iterator<Item = (usize, &str)> {
    src.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
}

/// The attribute lines directly above the item at `item` (contiguous `#[…]`
/// and `///` lines), trimmed.
fn attributes_above(lines: &[&str], item: usize) -> Vec<String> {
    lines[..item]
        .iter()
        .rev()
        .map(|l| l.trim())
        .take_while(|l| l.starts_with("#[") || l.starts_with("///"))
        .filter(|l| l.starts_with("#["))
        .map(str::to_owned)
        .collect()
}

/// The last line of the item that starts at `item`: the line itself if it
/// ends with `;`, else the first later line that closes it at its own
/// indentation.
fn item_end(lines: &[&str], item: usize) -> usize {
    let line = lines[item];
    if line.trim_end().ends_with(';') {
        return item;
    }
    let close = format!("{}}}", &line[..line.len() - line.trim_start().len()]);
    (item + 1..lines.len())
        .find(|&i| lines[i] == close)
        .unwrap_or_else(|| panic!("the item at line {} closes", item + 1))
}

/// Whether the attribute line `attr` gates its item on `test`.
fn is_test_cfg(attr: &str) -> bool {
    attr == "#[cfg(test)]" || attr.starts_with("#[cfg(all(test,")
}

/// At `i`, the opening of a raw string literal (`r"`, `r#"`, `br##"`, …):
/// its length, prefix, `#`s and `"` included, and its count of `#`s.
fn raw_string_opening(chars: &[char], i: usize) -> Option<(usize, usize)> {
    let mut j = i;
    if chars.get(j) == Some(&'b') {
        j += 1;
    }
    if chars.get(j) != Some(&'r') {
        return None;
    }
    j += 1;
    let mut hashes = 0;
    while chars.get(j) == Some(&'#') {
        hashes += 1;
        j += 1;
    }
    (chars.get(j) == Some(&'"')).then_some((j + 1 - i, hashes))
}

/// The `{` and `}` of each line of `src` that are code: not inside a
/// comment, a string, byte-string or raw-string literal (which may span
/// lines), or a char literal. A lifetime (`'a`) is not a char literal.
/// Shared with `hold_unhold_scan`.
pub(crate) fn code_braces_per_line(src: &str) -> Vec<(i64, i64)> {
    let chars: Vec<char> = src.chars().collect();
    let mut per_line = vec![(0i64, 0i64)];
    let mut i = 0;
    let at = |i: usize| chars.get(i).copied();
    let is_ident = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
    // Advance over `n` chars, starting a new line entry at each newline.
    let skip = |i: &mut usize, n: usize, per_line: &mut Vec<(i64, i64)>| {
        for _ in 0..n {
            if at(*i) == Some('\n') {
                per_line.push((0, 0));
            }
            *i += 1;
        }
    };
    while i < chars.len() {
        let c = chars[i];
        let prev_ident = i > 0 && is_ident(Some(chars[i - 1]));
        match c {
            '\n' => skip(&mut i, 1, &mut per_line),
            '/' if at(i + 1) == Some('/') => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '/' if at(i + 1) == Some('*') => {
                let mut depth = 0;
                loop {
                    if i >= chars.len() {
                        break;
                    }
                    if at(i) == Some('/') && at(i + 1) == Some('*') {
                        depth += 1;
                        skip(&mut i, 2, &mut per_line);
                    } else if at(i) == Some('*') && at(i + 1) == Some('/') {
                        depth -= 1;
                        skip(&mut i, 2, &mut per_line);
                        if depth == 0 {
                            break;
                        }
                    } else {
                        skip(&mut i, 1, &mut per_line);
                    }
                }
            }
            'r' | 'b' if !prev_ident && raw_string_opening(&chars, i).is_some() => {
                // A raw string: up to the `"` followed by as many `#`s as it
                // opened with.
                let (opening, hashes) = raw_string_opening(&chars, i).expect("matched above");
                skip(&mut i, opening, &mut per_line);
                loop {
                    if i >= chars.len() {
                        break;
                    }
                    if at(i) == Some('"') && (1..=hashes).all(|k| at(i + k) == Some('#')) {
                        skip(&mut i, 1 + hashes, &mut per_line);
                        break;
                    }
                    skip(&mut i, 1, &mut per_line);
                }
            }
            '"' => {
                skip(&mut i, 1, &mut per_line);
                while i < chars.len() && chars[i] != '"' {
                    let n = if chars[i] == '\\' { 2 } else { 1 };
                    skip(&mut i, n, &mut per_line);
                }
                skip(&mut i, 1, &mut per_line);
            }
            '\'' if at(i + 1) == Some('\\') => {
                // An escaped char literal (`'\n'`, `'\''`, `'\u{7}'`): past the
                // escaped char, up to the closing quote.
                let mut j = i + 3;
                while at(j).is_some_and(|c| c != '\'') {
                    j += 1;
                }
                i = j + 1;
            }
            '\'' if at(i + 2) == Some('\'') => i += 3,
            '{' => {
                per_line.last_mut().expect("a line").0 += 1;
                i += 1;
            }
            '}' => {
                per_line.last_mut().expect("a line").1 += 1;
                i += 1;
            }
            _ => i += 1,
        }
    }
    per_line
}

/// `src`'s code lines (no `//` comment lines) with every item gated on `test`
/// left out. A test item is found by its code braces
/// ([`code_braces_per_line`]) from its first line; the depth must return to
/// zero on the line that closes the item at its own indentation, or this
/// panics rather than hide code after a miscount. Shared with
/// `hold_unhold_scan`.
pub(crate) fn production_lines(src: &str, file: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = src.lines().collect();
    if lines.iter().any(|l| l.trim() == "#![cfg(test)]") {
        return Vec::new();
    }
    let braces = code_braces_per_line(src);
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if !is_test_cfg(trimmed) {
            if !trimmed.starts_with("//") {
                out.push((i, lines[i].to_owned()));
            }
            i += 1;
            continue;
        }
        // Skip the attribute, any further attributes and doc lines, and the
        // item they gate.
        let mut item = i + 1;
        while item < lines.len() && {
            let t = lines[item].trim();
            t.starts_with("#[") || t.starts_with("///") || t.starts_with("//")
        } {
            item += 1;
        }
        assert!(
            item < lines.len(),
            "{file}: a test cfg at line {} gates nothing",
            i + 1
        );
        // An item with no braced body — a `use`, `mod x;`, a statement —
        // ends at its first line ending `;`, and a field at its own line
        // ending `,`; any other item at the line that closes its first `{`.
        let mut open = item;
        let mut end = None;
        while open < lines.len() && braces[open].0 == 0 {
            let t = lines[open].trim_end();
            if t.ends_with(';') || (open == item && t.ends_with(',')) {
                end = Some(open);
                break;
            }
            open += 1;
        }
        if let Some(end) = end {
            i = end + 1;
            continue;
        }
        let indent = lines[item].len() - lines[item].trim_start().len();
        let mut depth = 0i64;
        for (j, line) in lines.iter().enumerate().skip(open) {
            let (opens, closes) = braces[j];
            depth += opens - closes;
            if depth == 0 && closes > 0 {
                assert!(
                    j == open
                        || line.trim_start().starts_with('}')
                            && line.len() - line.trim_start().len() == indent,
                    "{file}: the test item at line {} closes off its indentation at line {} — \
                     the brace count lost track",
                    item + 1,
                    j + 1
                );
                end = Some(j);
                break;
            }
            assert!(depth >= 0, "{file}: negative brace depth at line {}", j + 1);
        }
        let end = end.unwrap_or_else(|| {
            panic!(
                "{file}: the test item at line {} never closes — the brace count lost track",
                item + 1
            )
        });
        i = end + 1;
    }
    out
}

/// Every `.rs` file under `dir`, recursively, sorted.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut dirs = vec![dir.to_path_buf()];
    while let Some(d) = dirs.pop() {
        for entry in std::fs::read_dir(&d).expect("read a source directory") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// Every source file of the crate, by its path from the crate root
/// (`src/…`), with its text.
fn crate_sources() -> Vec<(String, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    rust_files(&root.join("src"))
        .into_iter()
        .map(|path| {
            let rel = path
                .strip_prefix(root)
                .expect("under the crate root")
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(&path).expect("read a source file");
            (rel, text)
        })
        .collect()
}

/// The files of the test modules `file` declares out of line
/// (`#[cfg(test)] mod name;`): `<dir>/name.rs`, and the prefix
/// `<dir>/name/` of its own files and submodules, where `<dir>` is the
/// directory `file`'s child modules live in.
fn out_of_line_test_modules(file: &str, text: &str) -> Vec<String> {
    let (parent, name) = file.rsplit_once('/').unwrap_or(("", file));
    let dir = match name {
        "mod.rs" | "lib.rs" | "main.rs" => parent.to_owned(),
        _ => format!("{parent}/{}", name.trim_end_matches(".rs")),
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !is_test_cfg(line.trim()) {
            continue;
        }
        let Some(item) = lines[i + 1..]
            .iter()
            .map(|l| l.trim())
            .find(|l| !(l.starts_with("#[") || l.starts_with("//")))
        else {
            continue;
        };
        let decl = item
            .strip_prefix("pub(crate) ")
            .or_else(|| item.strip_prefix("pub "))
            .unwrap_or(item);
        if let Some(module) = decl.strip_prefix("mod ").and_then(|m| m.strip_suffix(';')) {
            out.push(format!("{dir}/{module}.rs"));
            out.push(format!("{dir}/{module}/"));
        }
    }
    out
}

/// The crate's source files that are not test modules, with their text.
/// Shared with `hold_unhold_scan`.
pub(crate) fn production_sources() -> Vec<(String, String)> {
    let sources = crate_sources();
    let excluded: Vec<String> = sources
        .iter()
        .flat_map(|(file, text)| out_of_line_test_modules(file, text))
        .collect();
    sources
        .into_iter()
        .filter(|(file, _)| {
            !excluded
                .iter()
                .any(|x| file == x || (x.ends_with('/') && file.starts_with(x.as_str())))
        })
        .collect()
}

/// The non-test code lines of the crate containing `needle`, as
/// `(file, 1-based line)`. Shared with `hold_unhold_scan`.
pub(crate) fn production_hits(needle: &str) -> Vec<(String, usize)> {
    production_sources()
        .iter()
        .flat_map(|(file, text)| {
            production_lines(text, file)
                .into_iter()
                .filter(|(_, line)| line.contains(needle))
                .map(|(i, _)| (file.clone(), i + 1))
                .collect::<Vec<_>>()
        })
        .collect()
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// The production-line filter leaves out exactly the items gated on `test`
/// — a module, a function, a `use`, and a module in its own file — and keeps
/// the code around them, whatever braces the test code's literals and
/// comments hold, so the crate-wide scans below do not read test code and do
/// not skip production code.
#[test]
fn the_production_line_filter_leaves_out_test_items_and_keeps_the_rest() {
    // The test module's braces in literals and comments are unbalanced on
    // purpose: only code braces may count.
    let src = r##"fn production() {
    let keys = KeySchema::default();
}

pub struct Kept {
    kept: u8,
    #[cfg(test)]
    hook: Option<KeySchema>,
}

#[cfg(test)]
use crate::KeySchema;

#[cfg(test)]
fn helper<'a>(_s: &'a str) -> KeySchema {
    KeySchema::new("t:")
}

#[cfg(test)]
fn long_signature(
    a: u8,
) -> KeySchema {
    let x = KeySchema::default(
    );
    x
}

// KeySchema::default() in a comment
#[cfg(all(test, feature = "test-support"))]
mod tests {
    /* a { in a block comment */
    #[test]
    fn a() {
        let c = '{';
        let q = '\'';
        let u = '\u{7}';
        let s = format!("{{x");
        let b = b"{ not valid json";
        let r = r#"{"multi
line": {"#;
        let k = KeySchema::default();
    }
}

fn after_the_tests() {
    KeySchema::new("after:");
}
"##;
    let kept: Vec<String> = production_lines(src, "fixture")
        .into_iter()
        .map(|(_, l)| l.trim().to_owned())
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(
        kept,
        [
            "fn production() {",
            "let keys = KeySchema::default();",
            "}",
            "pub struct Kept {",
            "kept: u8,",
            "}",
            "fn after_the_tests() {",
            "KeySchema::new(\"after:\");",
            "}",
        ],
        "only the non-test code, and none of the comment"
    );

    // A test module in its own file is left out with its submodules, from
    // both a `name.rs` module and a `mod.rs` one; an ungated one is kept.
    let decls = "#[cfg(test)]\nmod calibration;\nmod kept;\n#[cfg(test)]\npub(crate) mod more;\n";
    assert_eq!(
        out_of_line_test_modules("src/allocator.rs", decls),
        [
            "src/allocator/calibration.rs",
            "src/allocator/calibration/",
            "src/allocator/more.rs",
            "src/allocator/more/",
        ]
    );
    assert_eq!(
        out_of_line_test_modules("src/storage/mod.rs", decls)[..2],
        [
            "src/storage/calibration.rs".to_owned(),
            "src/storage/calibration/".to_owned()
        ]
    );
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// A release build — any build without `test-support` — cannot read
/// `LIVE_EDIT_TEST_ONLY_KEY_PREFIX`: the name's one quoted literal is the
/// gated const; every code line naming the variable is inside a gated item;
/// the ungated `binary_key_schema_from_env` is the only one in that build and
/// returns the production schema whatever the environment holds; and no other
/// file names the variable outside a comment. Fails if a gate is removed, the
/// literal is copied into ungated code, or the ungated function reads
/// anything.
#[test]
fn the_key_prefix_override_is_read_only_by_a_test_support_build() {
    let lines: Vec<&str> = KEY_PREFIX.lines().collect();
    let quoted = format!("\"{VARIABLE}\"");

    // The quoted literal: once, in the gated const.
    let literals: Vec<usize> = code_lines(KEY_PREFIX)
        .filter(|(_, l)| l.contains(&quoted))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        literals.len(),
        1,
        "{quoted} is written once in key_prefix.rs, at lines {:?}",
        literals.iter().map(|i| i + 1).collect::<Vec<_>>()
    );
    let literal = literals[0];
    assert!(
        lines[literal]
            .trim_start()
            .starts_with("pub const TEST_ONLY_KEY_PREFIX_ENV: &str ="),
        "the literal is the const's value (line {}): {}",
        literal + 1,
        lines[literal]
    );
    assert_eq!(
        lines[literal - 1].trim(),
        CFG_TEST_SUPPORT,
        "the const is gated on test-support, on the line above it"
    );

    // The gated items: each starts at a line whose attributes include the
    // test-support gate.
    let gated: Vec<(usize, usize)> = (0..lines.len())
        .filter(|&i| {
            let t = lines[i].trim_start();
            !t.starts_with("#[")
                && !t.starts_with("//")
                && !t.is_empty()
                && attributes_above(&lines, i)
                    .iter()
                    .any(|a| a == CFG_TEST_SUPPORT)
        })
        .map(|i| (i, item_end(&lines, i)))
        .collect();
    for (idx, line) in code_lines(KEY_PREFIX) {
        if line.contains(VARIABLE) || line.contains("TEST_ONLY_KEY_PREFIX_ENV") {
            assert!(
                gated.iter().any(|(s, e)| (*s..=*e).contains(&idx)),
                "key_prefix.rs line {} names the variable outside an item gated on \
                 test-support: {line}",
                idx + 1
            );
        }
    }

    // `binary_key_schema_from_env`: one per build, the ungated one reading
    // nothing. `test_only_key_schema`: test-support only.
    let fns = |name: &str| -> Vec<usize> {
        code_lines(KEY_PREFIX)
            .filter(|(_, l)| l.trim_start().starts_with(&format!("pub fn {name}(")))
            .map(|(i, _)| i)
            .collect()
    };
    let binary = fns("binary_key_schema_from_env");
    let gates: Vec<Vec<String>> = binary
        .iter()
        .map(|&i| attributes_above(&lines, i))
        .collect();
    assert_eq!(
        binary.len(),
        2,
        "binary_key_schema_from_env is defined once for each build"
    );
    let not_gated: Vec<usize> = binary
        .iter()
        .zip(&gates)
        .filter(|(_, g)| g.iter().any(|a| a == CFG_NOT_TEST_SUPPORT))
        .map(|(i, _)| *i)
        .collect();
    let gated_fn = binary
        .iter()
        .zip(&gates)
        .filter(|(_, g)| g.iter().any(|a| a == CFG_TEST_SUPPORT))
        .count();
    assert_eq!(
        (not_gated.len(), gated_fn),
        (1, 1),
        "one binary_key_schema_from_env under each of {CFG_NOT_TEST_SUPPORT} and \
         {CFG_TEST_SUPPORT}: {gates:?}"
    );
    let release = not_gated[0];
    let body: Vec<&str> = lines[release + 1..item_end(&lines, release)]
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with("//"))
        .collect();
    assert_eq!(
        body,
        ["Ok(KeySchema::default())"],
        "without test-support the binary's schema is the production one, read from nothing"
    );
    let validator = fns("test_only_key_schema");
    assert_eq!(validator.len(), 1, "test_only_key_schema is defined once");
    assert!(
        attributes_above(&lines, validator[0])
            .iter()
            .any(|a| a == CFG_TEST_SUPPORT),
        "test_only_key_schema is compiled only with test-support"
    );

    // No other file names the variable outside a comment.
    for (file, text) in crate_sources() {
        if file == "src/config/key_prefix.rs" {
            continue;
        }
        for (idx, line) in code_lines(&text) {
            assert!(
                !line.contains(VARIABLE),
                "{file}:{} names {VARIABLE} outside a comment; only the gated code in \
                 src/config/key_prefix.rs may: {line}",
                idx + 1
            );
        }
    }
}

/// The text of the parenthesized list that opens at the first `(` at or
/// after byte `from` of `src`, without its parentheses, and the byte offset
/// just past its `)`.
fn paren_list(src: &str, from: usize) -> (&str, usize) {
    let open = from + src[from..].find('(').expect("a parenthesized list follows");
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return (&src[open + 1..open + i], open + i + 1);
                }
            }
            _ => {}
        }
    }
    panic!("the list opening at byte {open} closes");
}

/// `list` split at its top-level commas, each item trimmed, empty items
/// (a trailing comma's) left out.
fn top_level_items(list: &str) -> Vec<String> {
    let (mut items, mut depth, mut item) = (Vec::new(), 0i32, String::new());
    for c in list.chars() {
        match c {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth -= 1,
            ',' if depth == 0 => {
                items.push(std::mem::take(&mut item).trim().to_owned());
                continue;
            }
            _ => {}
        }
        item.push(c);
    }
    items.push(item.trim().to_owned());
    items.retain(|i| !i.is_empty());
    items
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// `main.rs` builds no key schema of its own: it calls
/// `binary_key_schema_from_env()` once, binds it to `key_schema`, and hands
/// that to `run`, called once, after the schema is built, which takes it as
/// its `key_schema: KeySchema` parameter: the call passes `key_schema` in
/// that parameter's position. The call and the signature are read whatever
/// their line breaks and however many other arguments they carry. Fails if
/// `KeySchema::default` or `KeySchema::new` appears in `main.rs`'s code (the
/// self-test's and `run`'s own schema were once built that way), if the
/// schema is built twice or bound to another name, if `run` is called more
/// than once or before the schema exists, or if `run` stops taking the schema
/// or is handed another value in its place.
#[test]
fn main_builds_no_key_schema_and_hands_run_the_one_from_binary_key_schema_from_env() {
    for needle in ["KeySchema::default", "KeySchema::new"] {
        let hits: Vec<usize> = code_lines(MAIN)
            .filter(|(_, l)| l.contains(needle))
            .map(|(i, _)| i + 1)
            .collect();
        assert!(
            hits.is_empty(),
            "main.rs builds a key schema with `{needle}` at lines {hits:?}: every key the binary \
             builds must be under the schema binary_key_schema_from_env returns"
        );
    }
    let calls: Vec<(usize, &str)> = code_lines(MAIN)
        .filter(|(_, l)| l.contains("binary_key_schema_from_env()"))
        .collect();
    assert_eq!(
        calls.len(),
        1,
        "main.rs builds its key schema once: {calls:?}"
    );
    let (call, call_line) = calls[0];
    assert!(
        call_line
            .trim_start()
            .starts_with("let key_schema = binary_key_schema_from_env()"),
        "the schema is bound to `key_schema` (line {}): {call_line}",
        call + 1
    );

    // The code with its comment lines blanked, so byte offsets map back to
    // line numbers and a commented-out call or signature is not read.
    let code: String = MAIN
        .lines()
        .map(|l| {
            if l.trim_start().starts_with("//") {
                ""
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let line_of = |offset: usize| code[..offset].matches('\n').count();

    // `run`'s signature: one `async fn run(`, taking `key_schema: KeySchema`.
    let defs: Vec<usize> = code
        .match_indices("async fn run(")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(defs.len(), 1, "one `async fn run(`: at bytes {defs:?}");
    let (params, _) = paren_list(&code, defs[0]);
    let params = top_level_items(params);
    let schema_param: Vec<usize> = params
        .iter()
        .enumerate()
        .filter(|(_, p)| p.as_str() == "key_schema: KeySchema")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        schema_param.len(),
        1,
        "run takes the schema as a `key_schema: KeySchema` parameter: {params:?}"
    );

    // Every call of `run`: `run(` not part of a longer name, not a method
    // call, and not its definition.
    let runs: Vec<usize> = code
        .match_indices("run(")
        .map(|(i, _)| i)
        .filter(|&i| {
            let before = code[..i].chars().next_back();
            !before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.')
                && !code[..i].ends_with("fn ")
        })
        .collect();
    assert_eq!(
        runs.len(),
        1,
        "main calls run once: at lines {:?}",
        runs.iter().map(|&i| line_of(i) + 1).collect::<Vec<_>>()
    );
    let run_call = runs[0];
    assert!(
        call < line_of(run_call),
        "the schema (line {}) is built before run is called (line {})",
        call + 1,
        line_of(run_call) + 1
    );
    let (args, _) = paren_list(&code, run_call);
    let args = top_level_items(args);
    assert_eq!(
        args.len(),
        params.len(),
        "run is called with one argument per parameter: {args:?} for {params:?}"
    );
    assert_eq!(
        args[schema_param[0]], "key_schema",
        "run is handed the schema binary_key_schema_from_env built, in its `key_schema` \
         parameter's position: {args:?} for {params:?}"
    );
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// Outside test code, only `src/config/key_prefix.rs` calls
/// `KeySchema::default`, and only it and the `test-support` fixtures
/// (`src/storage/test_support.rs`) call `KeySchema::new`: no other code on the
/// binary's path builds a schema of its own, which would name keys outside the
/// binary's prefix (a stream reader's placeholder schema once did). A new
/// call site fails here; a site that is truly not on the binary's path is
/// gated on `test`, or listed here with why. It reads every source file of
/// the crate: about 2 s in a debug build.
#[test]
fn only_the_key_prefix_module_builds_a_key_schema_outside_test_code() {
    for (needle, allowed) in [
        ("KeySchema::default", &["src/config/key_prefix.rs"][..]),
        (
            "KeySchema::new",
            &["src/config/key_prefix.rs", "src/storage/test_support.rs"][..],
        ),
    ] {
        let hits = production_hits(needle);
        let outside: Vec<&(String, usize)> = hits
            .iter()
            .filter(|(file, _)| !allowed.contains(&file.as_str()))
            .collect();
        assert!(
            outside.is_empty(),
            "`{needle}` outside test code, where only {allowed:?} may call it: {outside:?}"
        );
        assert!(
            hits.iter()
                .any(|(file, _)| file == "src/config/key_prefix.rs"),
            "non-vacuity: the scan sees key_prefix.rs's own `{needle}`: {hits:?}"
        );
    }
}

/// The values `test_only_key_schema` refuses — empty, a hash-tag brace, a
/// `SCAN MATCH` glob character, whitespace or a control character — each
/// with an error naming the variable, and the `TestValkey` fixture's prefix
/// shape it accepts, building every key under it.
#[cfg(feature = "test-support")]
#[test]
fn test_only_key_schema_refuses_unsafe_prefixes_and_accepts_the_fixture_shape() {
    use live_edit::config::key_prefix::{test_only_key_schema, TEST_ONLY_KEY_PREFIX_ENV};

    assert_eq!(TEST_ONLY_KEY_PREFIX_ENV, VARIABLE);
    for bad in [
        "",
        "test_{a}:",
        "test_a}:",
        "test_*:",
        "test_?:",
        "test_[a]:",
        "test_\\:",
        "test a:",
        "test_a:\n",
        "test_\u{7}:",
    ] {
        let err = test_only_key_schema(bad)
            .err()
            .unwrap_or_else(|| panic!("{bad:?} must be refused"));
        assert!(
            err.contains(VARIABLE),
            "{bad:?}: the refusal names the variable: {err}"
        );
    }
    let prefix = "test_0123456789abcdef0123456789abcdef:";
    let schema = test_only_key_schema(prefix).expect("the fixture's prefix shape is accepted");
    assert_eq!(schema.prefix(), prefix);
    assert!(
        schema.dr_hold_marker().starts_with(prefix),
        "the hold marker is built under the prefix: {}",
        schema.dr_hold_marker()
    );
}
