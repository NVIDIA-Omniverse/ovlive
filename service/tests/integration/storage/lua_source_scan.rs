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

//! Shared scanners over the service's Lua scripts, for the source guards that
//! read them: `b_1jt_rollout_tests` (who may write the GC `pending` set and
//! `inflight` scores), `ext_blob_reader_guard_test` (no script reads blob
//! bytes) and `script_migration_guard_test` (no writing script that can tear
//! runs while its slot migrates).
//!
//! Static: reads source files, needs no Valkey and no feature. The scanners'
//! own self-tests are in `script_migration_guard_test`.
//!
//! **What the Lua model covers.** `--` line comments and single- and
//! double-quoted strings with backslash escapes. Not long brackets (`[[ … ]]`,
//! `--[[ … ]]`, `[=[ … ]=]`): no script uses one, and [`long_bracket_lines`]
//! finds any that appears, so a guard can fail on it rather than read a long
//! comment as code or a long string as commands.

use std::collections::BTreeSet;
use std::path::Path;

use sha2::{Digest, Sha256};

/// Where the scripts live, relative to the crate root.
pub(crate) const SCRIPTS_DIR: &str = "src/storage/scripts";

/// Every Lua script the service compiles in (`include_str!("…/scripts/….lua")`
/// under `src/`, test-only inclusions too, the path on the macro's line or
/// the next), by file name, with its text. A script left on disk that no code
/// includes is not run, so it is not here; [`script_files`] lists what is on
/// disk.
pub(crate) fn compiled_scripts() -> Vec<(String, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut referenced = BTreeSet::new();
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read src dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).expect("read source");
                for (i, _) in text.match_indices("include_str!(") {
                    let rest = text[i + "include_str!(".len()..].trim_start();
                    let Some(rest) = rest.strip_prefix('"') else {
                        continue;
                    };
                    let included = &rest[..rest.find('"').expect("closing quote")];
                    if let Some(name) = included
                        .strip_suffix(".lua")
                        .and_then(|stem| stem.rsplit_once("scripts/"))
                        .map(|(_, stem)| format!("{stem}.lua"))
                    {
                        referenced.insert(name);
                    }
                }
            }
        }
    }
    referenced
        .into_iter()
        .map(|name| {
            let text = std::fs::read_to_string(root.join(SCRIPTS_DIR).join(&name))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            (name, text)
        })
        .collect()
}

/// The file name of every `.lua` file in [`SCRIPTS_DIR`], sorted. The
/// directory is flat; a subdirectory fails here, so a script cannot hide in
/// one.
pub(crate) fn script_files() -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(SCRIPTS_DIR);
    let mut names = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read the scripts dir") {
        let path = entry.expect("dir entry").path();
        assert!(
            !path.is_dir(),
            "{} is a directory; the scripts directory is flat",
            path.display()
        );
        if path.extension().is_some_and(|e| e == "lua") {
            names.push(
                path.file_name()
                    .expect("a file name")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    names.sort();
    names
}

/// `source` (a Lua script) without its `--` comments, line by line; a `--`
/// inside a quoted string is kept.
pub(crate) fn lua_code(source: &str) -> Vec<String> {
    source
        .lines()
        .map(|line| {
            let chars: Vec<char> = line.chars().collect();
            let mut quote: Option<char> = None;
            let mut code = String::new();
            let mut i = 0;
            while i < chars.len() {
                let c = chars[i];
                match quote {
                    Some(q) => {
                        if c == '\\' {
                            code.push(c);
                            i += 1;
                            if let Some(&escaped) = chars.get(i) {
                                code.push(escaped);
                            }
                        } else {
                            if c == q {
                                quote = None;
                            }
                            code.push(c);
                        }
                    }
                    None => {
                        if c == '-' && chars.get(i + 1) == Some(&'-') {
                            break;
                        }
                        if c == '"' || c == '\'' {
                            quote = Some(c);
                        }
                        code.push(c);
                    }
                }
                i += 1;
            }
            code
        })
        .collect()
}

/// The 1-based lines of `source` that open a long bracket — a `--[[` /
/// `--[=…[` block comment, or a `[[` / `[=…[` long string in code — which
/// [`lua_code`] does not model.
pub(crate) fn long_bracket_lines(source: &str) -> Vec<usize> {
    let opens_long_bracket = |text: &str| {
        let bytes = text.as_bytes();
        (0..bytes.len()).any(|i| {
            bytes[i] == b'['
                && bytes[i + 1..]
                    .iter()
                    .position(|b| *b != b'=')
                    .is_some_and(|run| bytes.get(i + 1 + run) == Some(&b'['))
        })
    };
    source
        .lines()
        .zip(lua_code(source))
        .enumerate()
        .filter(|(_, (raw, code))| {
            let comment = &raw[code.len()..];
            opens_long_bracket(code)
                || comment
                    .strip_prefix("--")
                    .is_some_and(|rest| rest.starts_with('[') && opens_long_bracket(rest))
        })
        .map(|(i, _)| i + 1)
        .collect()
}

/// The functions a script sends commands through.
pub(crate) const CALL_FUNCTIONS: [&str; 4] =
    ["redis.call", "redis.pcall", "server.call", "server.pcall"];

/// Every use of a [`CALL_FUNCTIONS`] function in `code` (from [`lua_code`]),
/// in order: the upper-cased command when the call's first argument is a
/// string literal, `None` when it is anything else — a variable, an
/// expression, or the function taken as a value without a call
/// (`local f = redis.call`), all of which hide the command from a static
/// scan.
pub(crate) fn script_calls(code: &[String]) -> Vec<Option<String>> {
    let joined = code.join("\n");
    let bytes = joined.as_bytes();
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut found: Vec<(usize, Option<String>)> = Vec::new();
    for function in CALL_FUNCTIONS {
        for (at, _) in joined.match_indices(function) {
            let end = at + function.len();
            if (at > 0 && (is_ident(bytes[at - 1]) || bytes[at - 1] == b'.'))
                || bytes.get(end).is_some_and(|b| is_ident(*b))
            {
                continue;
            }
            let mut i = end;
            while bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
                i += 1;
            }
            if bytes.get(i) != Some(&b'(') {
                found.push((at, None));
                continue;
            }
            i += 1;
            while bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
                i += 1;
            }
            let command = match bytes.get(i) {
                Some(&q @ (b'"' | b'\'')) => joined[i + 1..]
                    .find(q as char)
                    .map(|len| joined[i + 1..i + 1 + len].to_ascii_uppercase()),
                _ => None,
            };
            found.push((at, command));
        }
    }
    found.sort_by_key(|(at, _)| *at);
    found.into_iter().map(|(_, command)| command).collect()
}

/// `source` reduced to what a pinned hash should cover: its code without
/// comments ([`lua_code`]), each line's whitespace collapsed to single spaces,
/// blank lines dropped. A comment or layout edit leaves it unchanged; any
/// code edit changes it.
pub(crate) fn normalized_code(source: &str) -> String {
    lua_code(source)
        .iter()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The lowercase hex SHA-256 of `text`.
pub(crate) fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
