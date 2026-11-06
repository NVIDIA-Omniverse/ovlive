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

//! Source scans of the hold marker's single deleter (SDD SR-4.2; ADD 9,
//! PD-2 (3)): the un-hold is the only code that removes the marker, and it
//! runs only from the Job's finish, after a completed recount.
//!
//! The type-state receipt (`storage::dr_hold::unhold` takes the recount's
//! receipt) and the finish's run-time check (no write unit since the
//! recount) are pinned at run time in `tests/integration/dr/`
//! (`restore_finish_test`); what no run-time test can show is that no other
//! path deletes the marker. These scans pin it:
//!
//! - `scripts/dr_unhold.lua` is compiled in once, as `DR_UNHOLD_SCRIPT`;
//! - `DR_UNHOLD_SCRIPT` is run from one place, `storage::dr_hold::unhold`;
//! - `storage::dr_hold::unhold` is called from one place,
//!   `dr::job::finish::unhold`; nothing imports it by name, and no `use`
//!   imports `dr_hold` by glob or under another name;
//! - every production line naming the marker key (`dr_hold_marker()`) is a
//!   listed site, and the only one whose function body sends a delete or an
//!   expiry (`del`, `unlink`, `getdel`, an `expire` or `pexpire` method, in
//!   any turbofish form, or a custom command of one of those names) is the
//!   test-only `storage::test_support::remove_hold_marker`, in a module
//!   compiled only for tests and the `test-support` feature;
//! - of the two scripts handed the marker, `dr_hold_poll.lua` writes nothing
//!   and `dr_unhold.lua` deletes it in one `DEL`, the command names read in
//!   any case and either quote style.
//!
//! The scans read the crate's production code (`test_only_key_prefix`'s
//! filter: no `//` comment line, no item gated on `test`, no out-of-line
//! test module) and the scripts' code lines (no `--` comment line). They do
//! not see a marker key built by hand from its text, a script sent by text
//! rather than through `storage::lua`, a delete issued by a function the
//! listed site calls, nor a script command whose name is held in a
//! variable. The delete detector's positive control
//! ([`the_delete_detector_sees_every_delete_form_and_the_test_only_remover`])
//! keeps it from passing vacuously.

use crate::test_only_key_prefix::{
    code_braces_per_line, production_hits, production_lines, production_sources,
};

const LUA: &str = include_str!("../../src/storage/lua.rs");
const STORAGE_MOD: &str = include_str!("../../src/storage/mod.rs");
const DR_HOLD_POLL_LUA: &str = include_str!("../../src/storage/scripts/dr_hold_poll.lua");
const DR_UNHOLD_LUA: &str = include_str!("../../src/storage/scripts/dr_unhold.lua");

/// Every production site naming the marker key, by file and enclosing
/// function, with why it names it.
const MARKER_SITES: &[(&str, &str, &str)] = &[
    (
        "src/storage/dr_hold.rs",
        "read_hold_marker",
        "the hold poll's read (dr_hold_poll.lua)",
    ),
    (
        "src/storage/dr_hold.rs",
        "unhold",
        "the un-hold (dr_unhold.lua), the one deleter",
    ),
    (
        "src/storage/dr_job.rs",
        "store_time_ms",
        "pins TIME to the marker's slot; sends no command naming the key",
    ),
    (
        "src/storage/test_support.rs",
        "write_hold_marker",
        "test-only: how a test holds the fleet",
    ),
    (
        "src/storage/test_support.rs",
        "remove_hold_marker",
        "test-only: how a test lifts a hold",
    ),
];

/// The one site allowed to send a `DEL` or `UNLINK` naming the marker.
const TEST_ONLY_REMOVER: (&str, &str) = ("src/storage/test_support.rs", "remove_hold_marker");

/// The function whose body holds line `line` (0-based) of `text`: the
/// nearest code line at or above it declaring `fn <name>`, as its name and
/// that line's index.
fn enclosing_fn_at(text: &str, line: usize) -> (String, usize) {
    let lines: Vec<&str> = text.lines().collect();
    (0..=line)
        .rev()
        .filter(|&i| !lines[i].trim_start().starts_with("//"))
        .find_map(|i| {
            let (_, after) = lines[i].trim_start().split_once("fn ")?;
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            (!name.is_empty()).then_some((name, i))
        })
        .unwrap_or_else(|| panic!("no function encloses line {}", line + 1))
}

/// The name of the function whose body holds line `line` (0-based) of
/// `text` ([`enclosing_fn_at`]).
fn enclosing_fn(text: &str, line: usize) -> String {
    enclosing_fn_at(text, line).0
}

/// The whole text of the function declared at line `decl` (0-based) of
/// `text`, from its declaration to the line that closes its body, by code
/// braces (`test_only_key_prefix::code_braces_per_line`: braces in
/// comments and literals do not count).
fn fn_text(text: &str, decl: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let braces = code_braces_per_line(text);
    let mut depth = 0i64;
    let mut opened = false;
    for (j, &(opens, closes)) in braces.iter().enumerate().skip(decl) {
        depth += opens - closes;
        opened |= opens > 0;
        if opened && depth == 0 {
            return lines[decl..=j].join("\n");
        }
    }
    panic!("the function declared at line {} never closes", decl + 1)
}

/// The whole text of the function enclosing line `line` (0-based) of
/// `text`.
fn enclosing_fn_text(text: &str, line: usize) -> String {
    fn_text(text, enclosing_fn_at(text, line).1)
}

/// The Valkey commands that delete a key or set it to expire.
const DELETE_COMMANDS: [&str; 7] = [
    "del",
    "unlink",
    "getdel",
    "expire",
    "pexpire",
    "expireat",
    "pexpireat",
];

/// The first delete `code` sends, if any: a method call `.del(`,
/// `.unlink(`, `.getdel(`, or one whose name starts `expire` or `pexpire`
/// (`expire_at`, …), each also in turbofish form (`.del::<(), _>(`); or a
/// custom command named by a string literal of a [`DELETE_COMMANDS`] name,
/// in any case.
fn first_delete(code: &str) -> Option<String> {
    let chars: Vec<char> = code.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if *c != '.' {
            continue;
        }
        let name: String = chars[i + 1..]
            .iter()
            .take_while(|c| c.is_alphanumeric() || **c == '_')
            .collect();
        let after: String = chars[i + 1 + name.len()..].iter().take(3).collect();
        let is_call = after.starts_with('(') || after.starts_with("::<");
        let is_delete = matches!(name.as_str(), "del" | "unlink" | "getdel")
            || name.starts_with("expire")
            || name.starts_with("pexpire");
        if is_call && is_delete {
            return Some(format!(".{name}{after}"));
        }
    }
    let lower = code.to_lowercase();
    DELETE_COMMANDS
        .iter()
        .map(|command| format!("\"{command}\""))
        .find(|literal| lower.contains(literal.as_str()))
}

/// A script line names `command` as a Valkey command: the name quoted with
/// `"` or `'`, in any case.
fn lua_names(line: &str, command: &str) -> bool {
    let lower = line.to_lowercase();
    let command = command.to_lowercase();
    lower.contains(&format!("\"{command}\"")) || lower.contains(&format!("'{command}'"))
}

/// The text of `file` among the production sources.
fn source(sources: &[(String, String)], file: &str) -> String {
    sources
        .iter()
        .find(|(f, _)| f == file)
        .map(|(_, text)| text.clone())
        .unwrap_or_else(|| panic!("{file} is a production source"))
}

/// The production hits of `needle`, each with its enclosing function.
fn hits_in_fns(needle: &str) -> Vec<(String, String)> {
    let sources = production_sources();
    production_hits(needle)
        .into_iter()
        .map(|(file, line)| {
            let text = source(&sources, &file);
            let name = enclosing_fn(&text, line - 1);
            (file, name)
        })
        .collect()
}

/// A script's code lines: no `--` comment line.
fn lua_code(src: &str) -> Vec<&str> {
    src.lines()
        .filter(|l| !l.trim_start().starts_with("--"))
        .collect()
}

// Guard test: this module is pinned in scripts/guard-tests.sh; move or remove it there too.
/// `scripts/dr_unhold.lua` is compiled into the crate once, as
/// `DR_UNHOLD_SCRIPT` in `src/storage/lua.rs`: no second `Script` of it can
/// run the un-hold around the one wrapper.
#[test]
fn the_unhold_script_is_compiled_in_once_as_dr_unhold_script() {
    let included: Vec<(String, usize)> = production_hits("dr_unhold.lua\")")
        .into_iter()
        .filter(|(file, line)| {
            let sources = production_sources();
            let text = source(&sources, file);
            text.lines()
                .nth(line - 1)
                .is_some_and(|l| l.contains("include_str!("))
        })
        .collect();
    assert_eq!(
        included.len(),
        1,
        "dr_unhold.lua is included once in production code: {included:?}"
    );
    let (file, line) = (&included[0].0, included[0].1);
    assert_eq!(file, "src/storage/lua.rs");
    let lines: Vec<&str> = LUA.lines().collect();
    assert!(
        lines[line - 2..line]
            .iter()
            .any(|l| l.contains("static DR_UNHOLD_SCRIPT")),
        "as the body of `DR_UNHOLD_SCRIPT`: lua.rs:{line}"
    );
}

// Guard test: this module is pinned in scripts/guard-tests.sh; move or remove it there too.
/// `DR_UNHOLD_SCRIPT` runs from one place: `storage::dr_hold::unhold`, which
/// takes the recount's receipt. Its definition and its imports are not uses.
#[test]
fn the_unhold_script_is_run_only_by_storage_dr_hold_unhold() {
    let sources = production_sources();
    let uses: Vec<(String, String)> = production_hits("DR_UNHOLD_SCRIPT")
        .into_iter()
        .filter(|(file, line)| {
            let text = source(&sources, file);
            let l = text.lines().nth(line - 1).unwrap_or("").trim_start();
            !l.starts_with("use ") && !l.contains("static DR_UNHOLD_SCRIPT")
        })
        .map(|(file, line)| {
            let name = enclosing_fn(&source(&sources, &file), line - 1);
            (file, name)
        })
        .collect();
    assert_eq!(
        uses,
        vec![("src/storage/dr_hold.rs".to_owned(), "unhold".to_owned())],
        "the un-hold script runs only inside storage::dr_hold::unhold"
    );
}

// Guard test: this module is pinned in scripts/guard-tests.sh; move or remove it there too.
/// `storage::dr_hold::unhold` is called from one place, the Job's
/// `dr::job::finish::unhold`, which refuses after a write since the recount;
/// and no production `use` imports it by name, nor imports `dr_hold` by glob
/// or under another name, so neither a bare `unhold(` nor an alias's
/// `unhold(` can call it from elsewhere.
#[test]
fn storage_dr_hold_unhold_is_called_only_by_the_jobs_finish() {
    assert_eq!(
        hits_in_fns("dr_hold::unhold("),
        vec![("src/dr/job/finish.rs".to_owned(), "unhold".to_owned())],
        "storage::dr_hold::unhold is called only by dr::job::finish::unhold"
    );
    for (file, text) in production_sources() {
        for (line, statement) in use_statements(&production_lines(&text, &file)) {
            if let Some(why) = forbidden_dr_hold_import(&statement) {
                panic!("{file}:{line}: {why}: {statement}");
            }
        }
    }
}

/// The `use` statements among `lines` (a file's production lines), each
/// with its first line (1-based) and its whole text on one line, however
/// many lines it spans.
fn use_statements(lines: &[(usize, String)]) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut open: Option<(usize, String)> = None;
    for (i, line) in lines {
        let l = line.trim();
        if open.is_none() {
            let is_use = ["use ", "pub use ", "pub(crate) use ", "pub(super) use "]
                .iter()
                .any(|p| l.starts_with(p));
            if !is_use {
                continue;
            }
            open = Some((i + 1, String::new()));
        }
        let (_, text) = open.as_mut().expect("a statement is open");
        text.push_str(l);
        text.push(' ');
        if l.contains(';') {
            let (first, text) = open.take().expect("a statement is open");
            out.push((first, text.trim_end().to_owned()));
        }
    }
    out
}

/// Why `statement`, a whole `use` statement, is a forbidden import of
/// `storage::dr_hold`: it names `unhold`, globs the module, or names the
/// module (or `self` in its group) under another name.
fn forbidden_dr_hold_import(statement: &str) -> Option<&'static str> {
    if !statement.contains("dr_hold") {
        return None;
    }
    let tokens: Vec<&str> = statement
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '*'))
        .filter(|t| !t.is_empty())
        .collect();
    let compact: String = statement.chars().filter(|c| !c.is_whitespace()).collect();
    if tokens.contains(&"unhold") {
        return Some("nothing imports storage::dr_hold::unhold by name");
    }
    if compact.contains("dr_hold::*") || compact.contains("dr_hold::{*") {
        return Some("nothing imports storage::dr_hold by glob");
    }
    let renamed = tokens
        .windows(3)
        .any(|w| (w[0] == "dr_hold" || w[0] == "self") && w[1] == "as");
    if renamed {
        return Some("nothing imports storage::dr_hold under another name");
    }
    None
}

// Guard test: this module is pinned in scripts/guard-tests.sh; move or remove it there too.
/// The import check's positive control: a `use` naming `unhold` (directly,
/// in a group, across lines, under an alias), globbing `dr_hold`, or naming
/// `dr_hold` or its `self` under another name is refused; the imports the
/// crate makes today pass.
#[test]
fn the_import_check_refuses_every_way_round_the_call_scan() {
    let statements = |src: &str| {
        let lines: Vec<(usize, String)> = src
            .lines()
            .enumerate()
            .map(|(i, l)| (i, l.to_owned()))
            .collect();
        use_statements(&lines)
    };
    for refused in [
        "use crate::storage::dr_hold::unhold;",
        "use crate::storage::dr_hold::{read_hold_marker, unhold as clear};",
        "use crate::storage::dr_hold::{\n    UnholdOutcome,\n    unhold,\n};",
        "use crate::storage::dr_hold::*;",
        "pub(crate) use crate::storage::{dr_hold::*, keys};",
        "use crate::storage::dr_hold as hold;",
        "use crate::storage::{\n    dr_hold as h,\n    keys,\n};",
        "use crate::storage::dr_hold::{self as h, UnholdOutcome};",
    ] {
        let parsed = statements(refused);
        assert_eq!(parsed.len(), 1, "one statement: {refused}");
        assert!(
            forbidden_dr_hold_import(&parsed[0].1).is_some(),
            "refused: {refused}"
        );
    }
    for allowed in [
        "use crate::storage::dr_hold::{self, UnholdOutcome};",
        "use crate::storage::dr_hold::read_hold_marker;",
        "use crate::storage::dr_hold::{\n    MARKER_FIELD_INCIDENT, MARKER_FIELD_REASON,\n};",
        "use crate::storage::keys::KeySchema;",
    ] {
        let parsed = statements(allowed);
        assert_eq!(parsed.len(), 1, "one statement: {allowed}");
        assert_eq!(
            forbidden_dr_hold_import(&parsed[0].1),
            None,
            "allowed: {allowed}"
        );
    }
}

// Guard test: this module is pinned in scripts/guard-tests.sh; move or remove it there too.
/// Every production site naming the hold marker's key is listed in
/// [`MARKER_SITES`], and the only one whose function body sends a delete or
/// an expiry ([`first_delete`]) is the test-only remover, in
/// `storage::test_support`, which the crate compiles only for tests and the
/// `test-support` feature. So no service code deletes the marker but
/// through `dr_unhold.lua`.
#[test]
fn no_production_code_deletes_the_hold_marker_but_the_unhold_script() {
    let sources = production_sources();
    let mut sites: Vec<(String, String)> = Vec::new();
    for (file, line) in production_hits("dr_hold_marker()") {
        let text = source(&sources, &file);
        let name = enclosing_fn(&text, line - 1);
        assert!(
            MARKER_SITES
                .iter()
                .any(|(f, n, _)| *f == file && *n == name),
            "{file}:{line} (fn {name}) names the hold marker's key and is not a listed site; \
             if it deletes the marker, it must go through dr_unhold.lua instead"
        );
        if let Some(delete) = first_delete(&enclosing_fn_text(&text, line - 1)) {
            assert_eq!(
                (file.as_str(), name.as_str()),
                TEST_ONLY_REMOVER,
                "{file}:{line}: fn {name} names the hold marker and sends `{delete}` outside the \
                 un-hold script"
            );
        }
        sites.push((file, name));
    }
    for (file, name, why) in MARKER_SITES {
        assert!(
            sites.iter().any(|(f, n)| f == file && n == name),
            "listed site {file} fn {name} ({why}) no longer names the marker: update the list"
        );
    }
    assert!(
        STORAGE_MOD
            .contains("#[cfg(any(test, feature = \"test-support\"))]\npub mod test_support;"),
        "storage::test_support, home of the test-only remover, is compiled only for tests and \
         the test-support feature"
    );
}

// Guard test: this module is pinned in scripts/guard-tests.sh; move or remove it there too.
/// Of the two scripts the listed sites hand the marker key, the hold poll
/// (`dr_hold_poll.lua`) sends no write command, and the un-hold
/// (`dr_unhold.lua`) deletes the marker in exactly one `DEL`, and sends no
/// other delete, expiry or rename. Command names are read in any case and
/// either quote style ([`lua_names`]).
#[test]
fn of_the_scripts_handed_the_marker_only_dr_unhold_deletes_it() {
    const WRITES: [&str; 14] = [
        "DEL",
        "UNLINK",
        "GETDEL",
        "HDEL",
        "HSET",
        "HSETNX",
        "HMSET",
        "SET",
        "EXPIRE",
        "PEXPIRE",
        "EXPIREAT",
        "PEXPIREAT",
        "RENAME",
        "RENAMENX",
    ];
    for line in lua_code(DR_HOLD_POLL_LUA) {
        for write in WRITES {
            assert!(
                !lua_names(line, write),
                "dr_hold_poll.lua sends no write command: {line}"
            );
        }
    }
    let unhold = lua_code(DR_UNHOLD_LUA);
    let deletes: Vec<&&str> = unhold
        .iter()
        .filter(|l| {
            DELETE_COMMANDS
                .iter()
                .chain(&["rename", "renamenx"])
                .any(|command| lua_names(l, command))
        })
        .collect();
    assert_eq!(
        deletes.len(),
        1,
        "one delete, expiry or rename in dr_unhold.lua: {deletes:?}"
    );
    assert!(
        lua_names(deletes[0], "DEL") && deletes[0].contains("marker_key"),
        "and it is the marker's DEL: {}",
        deletes[0]
    );
}

// Guard test: this module is pinned in scripts/guard-tests.sh; move or remove it there too.
/// The delete detectors' positive control, so the scans above cannot pass
/// vacuously: [`first_delete`] sees every Rust delete form — a plain and a
/// turbofish method call, `getdel`, the `expire` family, a custom command in
/// any case — and the body of the test-only remover
/// (`storage::test_support::remove_hold_marker`) as a delete, and sees none
/// in a read or a field delete; [`lua_names`] sees a command in either
/// quote style and any case.
#[test]
fn the_delete_detector_sees_every_delete_form_and_the_test_only_remover() {
    for deleting in [
        "let _: i64 = client.pool.del(key).await?;",
        "let _: () = pool.del::<(), _>(keys.dr_hold_marker()).await?;",
        "conn.unlink::<i64, _>(key)",
        "conn.getdel(key)",
        "conn.expire(key, 1, None)",
        "conn.pexpire_at::<(), _>(key, 1, None)",
        "conn.custom(CustomCommand::new_static(\"UNLINK\", ClusterHash::FirstKey, false), args)",
        "conn.custom(CustomCommand::new_static(\"Del\", ClusterHash::FirstKey, false), args)",
    ] {
        assert!(first_delete(deleting).is_some(), "a delete: {deleting}");
    }
    for reading in [
        "conn.get(key)",
        "conn.hdel(key, \"incident\")",
        "conn.delta(key)",
        "conn.hgetall(keys.dr_hold_marker())",
        "let deleted = removed == 1;",
    ] {
        assert_eq!(first_delete(reading), None, "not a delete: {reading}");
    }

    let sources = production_sources();
    let (file, function) = TEST_ONLY_REMOVER;
    let text = source(&sources, file);
    let decl = text
        .lines()
        .position(|l| {
            l.trim_start()
                .starts_with(&format!("pub async fn {function}("))
        })
        .unwrap_or_else(|| panic!("{file} declares {function}"));
    assert!(
        first_delete(&fn_text(&text, decl)).is_some(),
        "the test-only remover's body is seen as a delete"
    );

    for (line, command) in [
        ("redis.call(\"DEL\", marker_key)", "DEL"),
        ("redis.call('del', marker_key)", "DEL"),
        ("redis.call('Unlink', k)", "UNLINK"),
    ] {
        assert!(lua_names(line, command), "{command} in {line}");
    }
    assert!(!lua_names("redis.call(\"HDEL\", k, f)", "DEL"));
}
