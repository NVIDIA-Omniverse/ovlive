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

//! b.2tu — who writes a `CreateSession`'s metadata, and who removes a
//! session's index members (SDD §7.10, §7.13).
//!
//! The ListSessions ghost walk may remove a names member only after vetoing
//! its id while the metadata is absent (or finding the id carrying the
//! session deletion marker), and `CreateSession` may write metadata only
//! through `session_meta_write.lua`, which refuses a vetoed or a marked id. Both
//! halves hold only while nothing goes around the scripts, so this guard pins
//! the source shape, over the production code (comments stripped, the
//! `#[cfg(test)] mod tests` block cut off) of `src/grpc/session.rs` and
//! `src/storage/session.rs`:
//!
//! - **`src/grpc/session.rs` sends no `SREM`**: the walk removes members only
//!   through `SessionStore::retire_ghost_member`.
//! - **`src/storage/session.rs` writes no hash directly** (`HSET` / `HMSET` /
//!   `HSETNX` in any call spelling: the method, its turbofish, or a `"HSET"`
//!   command literal): the metadata goes only through
//!   `session_meta_write.lua`, which only `write_session_meta` runs and only
//!   `finish_create` calls.
//! - **Names members are removed only by `SessionStore::delete` and the
//!   release script's callers**: the only direct `SREM`s are `delete`'s two;
//!   `session_index_release.lua` is run only by `release_index_members`, whose
//!   callers are exactly the walk's retirement and `CreateSession`'s three
//!   clean-ups; and the retirement sends its veto before its release.
//!
//! Scoped to those two files on purpose: the DR restore's
//! `dr_write::remove_path_members` (two `SREM`s, `path_index` first) is another
//! remover, outside this rule.
//!
//! Static: reads source files, needs no Valkey and no feature.

// Guard tests: this module is pinned in scripts/guard-tests.sh `MODULES`; a
// test added here is a guard, and renaming or removing the module means
// editing that list too.

use std::collections::BTreeSet;

use super::script_migration_guard_test::{enclosing_fn_at, production_sources};

const GRPC_SESSION: &str = "src/grpc/session.rs";
const STORAGE_SESSION: &str = "src/storage/session.rs";

/// The production code of `file` (repository-relative), comments stripped and
/// the inline test module cut off.
fn production(file: &str) -> String {
    production_sources()
        .into_iter()
        .find(|(path, _)| path.to_string_lossy().replace('\\', "/") == file)
        .map(|(_, code)| code)
        .unwrap_or_else(|| panic!("{file} must be among the production sources"))
}

/// Byte offsets in `code` of every match of any of `needles`.
fn sites(code: &str, needles: &[&str]) -> Vec<usize> {
    let mut at: Vec<usize> = needles
        .iter()
        .flat_map(|needle| code.match_indices(needle).map(|(i, _)| i))
        .collect();
    at.sort_unstable();
    at.dedup();
    at
}

/// The names of the `fn`s holding each of `offsets`.
fn enclosing_fns(code: &str, offsets: &[usize]) -> Vec<String> {
    offsets
        .iter()
        .map(|&at| enclosing_fn_at(code, at).1)
        .collect()
}

/// The `fn`s that name `ident`, one entry per mention, leaving out mentions
/// before the file's first `fn` (its `use` declarations).
fn fns_naming(code: &str, ident: &str) -> Vec<String> {
    let at: Vec<usize> = code.match_indices(ident).map(|(at, _)| at).collect();
    enclosing_fns(code, &at)
        .into_iter()
        .filter(|name| name != "<file scope>")
        .collect()
}

/// Every call of `name(` in `code` that is not its definition.
fn calls_of(code: &str, name: &str) -> Vec<usize> {
    let needle = format!("{name}(");
    code.match_indices(&needle)
        .map(|(at, _)| at)
        .filter(|&at| {
            let prev = code[..at].chars().next_back();
            !prev.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
                && !code[..at].trim_end().ends_with("fn")
        })
        .collect()
}

/// The needles of a direct `SREM` from Rust.
const SREM_NEEDLES: [&str; 3] = [".srem(", "srem::<", "\"SREM\""];

/// The needles of a direct hash write from Rust.
const HSET_NEEDLES: [&str; 7] = [
    ".hset(",
    "hset::<",
    ".hmset(",
    "hmset::<",
    ".hsetnx(",
    "hsetnx::<",
    "\"HSET\"",
];

#[test]
fn the_write_needles_match_every_call_spelling_and_nothing_else() {
    for srem in [
        "pool.srem::<i64, _, _>(key, member)",
        "pool.srem(key, member)",
        "pool.custom(cmd!(\"SREM\"), args)",
    ] {
        assert!(
            !sites(srem, &SREM_NEEDLES).is_empty(),
            "SREM_NEEDLES must match {srem:?}"
        );
    }
    for hset in [
        ".hset::<(), _, _>(key, fields)",
        "client.hset(key, fields)",
        "client.hmset(key, fields)",
        "client.hmset::<(), _, _>(key, fields)",
        "client.hsetnx(key, field, value)",
        "client.hsetnx::<bool, _, _, _>(key, field, value)",
        "pool.custom(cmd!(\"HSET\"), args)",
    ] {
        assert!(
            !sites(hset, &HSET_NEEDLES).is_empty(),
            "HSET_NEEDLES must match {hset:?}"
        );
    }
    for innocent in [
        "std::collections::HashSet<String>",
        "let hset_result = 1;",
        "smembers(key)",
    ] {
        assert!(
            sites(innocent, &SREM_NEEDLES).is_empty() && sites(innocent, &HSET_NEEDLES).is_empty(),
            "no needle may match {innocent:?}"
        );
    }
}

/// The ghost walk removes nothing itself: `src/grpc/session.rs` sends no
/// `SREM` and does not import the trait that sends one.
#[test]
fn grpc_session_production_code_sends_no_srem() {
    let code = production(GRPC_SESSION);
    let found = sites(&code, &SREM_NEEDLES);
    assert!(
        found.is_empty(),
        "{GRPC_SESSION} sends an SREM in {:?}. The ListSessions walk removes a ghost's members \
         only through `SessionStore::retire_ghost_member`, which vetoes the id first (SDD §7.13)",
        enclosing_fns(&code, &found)
    );
    assert!(
        !code.contains("SetsInterface"),
        "{GRPC_SESSION} imports fred's SetsInterface; the walk sends no set command of its own"
    );
}

/// `CreateSession`'s metadata is written only through
/// `session_meta_write.lua`: `src/storage/session.rs` writes no hash directly,
/// the script handle is used only in `write_session_meta`, and that is called
/// only from `finish_create`.
#[test]
fn storage_session_writes_metadata_only_through_the_metadata_write_script() {
    let code = production(STORAGE_SESSION);
    let hset = sites(&code, &HSET_NEEDLES);
    assert!(
        hset.is_empty(),
        "{STORAGE_SESSION} writes a hash directly in {:?}. A CreateSession's metadata must go \
         through `session_meta_write.lua`, which refuses an id a ListSessions walk vetoed \
         (SDD §7.10); a direct HSET could complete a vetoed id",
        enclosing_fns(&code, &hset)
    );

    assert_eq!(
        fns_naming(&code, "SESSION_META_WRITE_SCRIPT"),
        vec!["write_session_meta".to_owned()],
        "SESSION_META_WRITE_SCRIPT must be run in exactly one place, `write_session_meta`"
    );
    assert_eq!(
        enclosing_fns(&code, &calls_of(&code, "write_session_meta")),
        vec!["finish_create".to_owned()],
        "`write_session_meta` must be called only by `finish_create`"
    );
}

/// Names members (and their `path_index` members) are removed only by
/// `SessionStore::delete`'s two `SREM`s and by `session_index_release.lua`,
/// run only by `release_index_members`, whose callers are exactly the walk's
/// retirement and `CreateSession`'s three clean-ups. The retirement vetoes
/// before it releases.
#[test]
fn session_index_members_are_removed_only_by_delete_and_the_release_scripts_callers() {
    let storage = production(STORAGE_SESSION);

    let srems = sites(&storage, &SREM_NEEDLES);
    assert_eq!(
        enclosing_fns(&storage, &srems),
        vec!["delete".to_owned(), "delete".to_owned()],
        "the only direct SREMs in {STORAGE_SESSION} are DeleteSession's two (path_index, then \
         the names member). Every other removal goes through `session_index_release.lua`, which \
         removes both members at once"
    );

    assert_eq!(
        fns_naming(&storage, "SESSION_INDEX_RELEASE_SCRIPT"),
        vec!["release_index_members".to_owned()],
        "SESSION_INDEX_RELEASE_SCRIPT must be run in exactly one place, `release_index_members`"
    );

    let callers: BTreeSet<String> =
        enclosing_fns(&storage, &calls_of(&storage, "release_index_members"))
            .into_iter()
            .collect();
    let want: BTreeSet<String> = [
        "retire_ghost_member",
        "release_abandoned_attempt",
        "release_in_doubt_attempt",
        "compensate_unwatched_create",
    ]
    .map(str::to_owned)
    .into_iter()
    .collect();
    assert_eq!(
        callers, want,
        "`release_index_members` is called only by the ListSessions retirement and \
         CreateSession's three clean-ups; a new remover of index members needs its own case \
         here and in SDD §7.13"
    );

    for (file, code) in production_sources() {
        let file = file.to_string_lossy().replace('\\', "/");
        if file == STORAGE_SESSION || file == "src/storage/lua.rs" {
            continue;
        }
        assert!(
            !code.contains("SESSION_INDEX_RELEASE_SCRIPT")
                && !code.contains("session_index_release.lua\""),
            "{file} runs the session index release; only `SessionStore` may"
        );
    }

    // The retirement's veto precedes its release, so the walk removes members
    // only of an id no metadata write can complete.
    let retire = function_body(&storage, "pub async fn retire_ghost_member(");
    let veto = retire
        .find("veto_session_meta(")
        .expect("retire_ghost_member must veto");
    let release = retire
        .find("release_index_members(")
        .expect("retire_ghost_member must release");
    let vetoed_arm = retire
        .find("MetaVeto::Vetoed")
        .expect("retire_ghost_member must branch on the veto's verdict");
    assert!(
        veto < vetoed_arm && vetoed_arm < release,
        "retire_ghost_member must send its veto first and release only in the `Vetoed` / \
         `Deleted` arm"
    );
    assert_eq!(
        calls_of(retire, "release_index_members").len(),
        1,
        "retire_ghost_member releases in one place, the `Vetoed` / `Deleted` arm"
    );
}

/// The text of the `fn` whose signature starts with `signature`, up to the
/// next `fn` at the same or a shallower indent (good enough for a method body
/// in this file, which holds no nested `fn`).
fn function_body<'a>(code: &'a str, signature: &str) -> &'a str {
    let start = code.find(signature).unwrap_or_else(|| {
        panic!("`{signature}` not found — signature changed? update this guard")
    });
    let rest = &code[start + signature.len()..];
    let end = [
        "\n    pub async fn ",
        "\n    async fn ",
        "\n    pub fn ",
        "\n    fn ",
    ]
    .iter()
    .filter_map(|next| rest.find(next))
    .min()
    .unwrap_or(rest.len());
    &code[start..start + signature.len() + end]
}
