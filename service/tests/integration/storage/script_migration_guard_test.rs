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

//! b.cyu — the guard behind SDD §18.1's rule: **no writing script that can
//! tear runs while its slot is migrating or importing.**
//!
//! During a legacy cluster slot migration every command a script sends is
//! checked against the slot's state again, and on the migrating source one
//! that names a key absent at that moment is refused, aborting the script with
//! its earlier writes kept — a *tear*. A script that can tear declares its
//! slot's never-written migration sentinel (`KeySchema::migration_sentinel`),
//! so the store refuses it before it starts for the slot's whole move: the
//! [`SENTINEL`] list. Every other writing script cannot tear, for one stated
//! reason ([`Reason`]): the [`ALLOWLIST`]. This guard keeps the two lists
//! complete and the sentinel wired:
//!
//! - **Inventory.** Every script in `src/storage/scripts/` is compiled in
//!   (`include_str!` under `src/`, test-only inclusions too), and every
//!   compiled script is a file there.
//! - **Closed-world command scan.** Every `redis.call` / `redis.pcall` /
//!   `server.call` / `server.pcall` in a script names its command as a string
//!   literal in [`WRITE_COMMANDS`], [`READ_COMMANDS`] or [`KEYLESS_COMMANDS`]
//!   (`PUBLISH` is keyless, not a key write). A literal on no list fails, so a
//!   new command is classified before it ships. A computed command name hides
//!   what a script sends, so it is allowed only in a script on
//!   [`READ_ONLY_COMPUTED`], whose source is hash-pinned. A long-bracket
//!   comment or string, which the scanner does not model, fails.
//! - **Every writing script is on exactly one list, and no read-only script is
//!   on either.** A script writes when it sends a [`WRITE_COMMANDS`] command.
//! - **Hash pins (decision 11).** Each [`ALLOWLIST`] and [`READ_ONLY_COMPUTED`]
//!   script's source, comments stripped and whitespace collapsed
//!   (`lua_source_scan::normalized_code`), hashes to its pinned SHA-256. An
//!   edit to one fails until its reason is re-confirmed and the hash updated;
//!   the failure prints the new row.
//! - **The sentinel is wired.** Each [`SENTINEL`] script declares its slot and
//!   checks it before any read or write; no command's argument list, read
//!   whole from its `(` to the matching `)` across lines, names the slot by
//!   `KEYS[<n>]`, the `KEY_MIGRATION_SENTINEL` constant, the check's
//!   `sentinel` local, or a variable assigned from one of those (a key
//!   reached any other way, such as a computed index, is beyond the scan;
//!   a run-time check covers it instead, asserting the sentinel still absent
//!   after a landed run: the GC, recount and signed-zero merge controls in
//!   `b_cyu_script_sentinel_test`, the commit's fast mode in
//!   `update_layer_lua_test`'s
//!   `update_layer_fast_script_direct_invocation_commits_like_the_call_site`,
//!   and its structural mode in `b_cyu_slot_migration_test`'s
//!   `b_qtk_children_rebuild_staging_runs_and_names_its_emptied_list_after_its_xadd`);
//!   `migration-sentinel` is spelled in code only by the builder in
//!   `src/storage/keys.rs` and the five scripts' checks; the builder is called
//!   only at the five `KEYS`-assembly sites; and each of those scripts'
//!   `EVALSHA` keys carry it.
//! - **SDD §18.1 names every script on every list**, as the bare name in
//!   backticks.
//!
//! What it does not do: decide that an allowlisted script cannot tear. Branchy
//! Lua defeats a textual "nothing refusable after the first write" test; the
//! reason on each row, re-confirmed whenever its hash moves, is the review.
//! The runtime half — each sentinel script refusing a call without its
//! sentinel on a standalone store — is in `b_cyu_script_sentinel_test` and
//! `update_layer_lua_test`; the refusal on a real migrating slot in the
//! cluster-gated `b_cyu_slot_migration_test`. The Rust-side rule that every
//! fred `Transaction` re-syncs its routing after a redirect is a separate
//! guard, `txn_exec_routing_guard_test`, which shares this module's Rust
//! source helpers ([`production_sources`], [`enclosing_fn_at`]).
//!
//! Static: reads source files, needs no Valkey and no feature.

// Guard tests: this module is pinned in scripts/guard-tests.sh `MODULES`; a
// test added here is a guard, and renaming or removing the module means
// editing that list too.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::b_289_tests::{call_arguments, code_lines, rust_sources_under};
use super::lua_source_scan::{
    compiled_scripts, long_bracket_lines, lua_code, normalized_code, script_calls, script_files,
    sha256_hex, CALL_FUNCTIONS,
};

// ─────────────────────────────────────────────────────────────────────────────
// The lists
// ─────────────────────────────────────────────────────────────────────────────

/// Why an allowlisted writing script cannot tear (SDD §18.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reason {
    /// Nothing refusable between the first and last write: every call after
    /// the first write names a key present then (declared, and not emptied
    /// earlier in the run).
    A1,
    /// Only the reply can be lost: every refusable call follows the last
    /// write, or is a no-op on a key the run already emptied.
    A2,
    /// Writes only when a declared key is absent: on a migrating source the
    /// script runs only with every declared key present, and then writes
    /// nothing.
    A3,
    /// A named store invariant keeps the emptied key non-empty.
    A4,
    /// Write-first: the script writes its own field before pruning others.
    A5,
}

/// One allowlisted writing script: its name (no `.lua`), its reason, a note
/// on where the reason holds, and the SHA-256 of its normalized source.
struct Allowed {
    script: &'static str,
    reason: Reason,
    note: &'static str,
    sha256: &'static str,
}

const fn allowed(
    script: &'static str,
    reason: Reason,
    note: &'static str,
    sha256: &'static str,
) -> Allowed {
    Allowed {
        script,
        reason,
        note,
        sha256,
    }
}

/// How a sentinel script declares and checks its slot.
struct Sentinel {
    script: &'static str,
    /// The `KEYS` index of the sentinel.
    slot: usize,
    /// Code the script's check must contain, comments stripped.
    check: &'static [&'static str],
}

/// The five writing scripts that can tear and declare their slot's
/// migration sentinel (SDD §18.1).
const SENTINEL: [Sentinel; 5] = [
    Sentinel {
        script: "update_layer_fast",
        slot: live_edit::storage::lua::UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL,
        check: &[
            "local KEY_MIGRATION_SENTINEL = 10",
            "local sentinel = KEYS[KEY_MIGRATION_SENTINEL]",
            "local suffix = \":migration-sentinel\"",
            "or string.sub(sentinel, -string.len(suffix)) ~= suffix",
            "or hash_tag(sentinel) ~= hash_tag(KEYS[KEY_HEADER]) then",
        ],
    },
    Sentinel {
        script: "gc_drop_foreign_claims",
        slot: 7,
        check: &[
            "local KEY_MIGRATION_SENTINEL = 7",
            "local sentinel = KEYS[KEY_MIGRATION_SENTINEL]",
            "local suffix = \":migration-sentinel\"",
            "if #KEYS ~= KEY_MIGRATION_SENTINEL",
            "or string.sub(sentinel, -string.len(suffix)) ~= suffix",
            "or hash_tag(sentinel) ~= hash_tag(KEYS[1]) then",
        ],
    },
    Sentinel {
        script: "gc_ledger_audit",
        slot: 7,
        check: &[
            "local KEY_MIGRATION_SENTINEL = 7",
            "local sentinel = KEYS[KEY_MIGRATION_SENTINEL]",
            "local suffix = \":migration-sentinel\"",
            "if #KEYS ~= KEY_MIGRATION_SENTINEL",
            "or string.sub(sentinel, -string.len(suffix)) ~= suffix",
            "or hash_tag(sentinel) ~= hash_tag(KEYS[1]) then",
        ],
    },
    Sentinel {
        script: "ext_index_recount",
        slot: 5,
        check: &[
            "if #KEYS ~= 5 or KEYS[5] ~= ARGV[1] .. '{' .. ARGV[2] .. '}:migration-sentinel' then",
        ],
    },
    Sentinel {
        script: "time_sample_signed_zero_merge",
        slot: 4,
        check: &[
            "if #KEYS < 4 or KEYS[4] ~= ARGV[1] .. '{' .. ARGV[2] .. '}:migration-sentinel' then",
        ],
    },
];

/// The writing scripts that cannot tear, with their reason (SDD §18.1) and
/// their pinned source hash (decision 11). A script whose hash no longer
/// matches fails [`every_pinned_script_matches_its_hash`] until its reason is
/// re-confirmed against the edit and the hash updated.
const ALLOWLIST: [Allowed; 41] = [
    allowed(
        "alloc_object_id",
        Reason::A1,
        "`EXISTS`, then one `INCR`, last; untagged counter",
        "2f79e16b76a4324f126e92cd01e7e8ccd9a8ea80fec8b2f7479dce3673ae05f4",
    ),
    allowed(
        "commit_record_retract",
        Reason::A1,
        "one `HDEL`, last",
        "3677afa8d0229a98b5392f931fd0f12f7e3e125c051627a37c80da0dd2e87efd",
    ),
    allowed(
        "create_layer_commit",
        Reason::A1,
        "no command empties a key; the deletion marker (and the commit claim, read before the \
         first write) is normally absent, so already refused",
        "22552692c03640b340b382593da1446851b56581b36d3a4b5016f13ba9537b6b",
    ),
    allowed(
        "create_layer_veto",
        Reason::A1,
        "a no-op `PEXPIRE` opener then one `SET`, last, both on the claim key just read absent, \
         so a migrating source refuses the opener before any write; the deletion marker is \
         normally absent, so already refused",
        "6a8f25ee86f901be691c9a7ddc4d4d076266458380547211692571484359b4b7",
    ),
    allowed(
        "ext_backup_confirm",
        Reason::A1,
        "one key; the opening `EXISTS` names it, so a meta that moved is refused before any \
         write; then `HGET`, the keyless `TIME` and one `HSET`, last",
        "676d7e5db88d12eae1e784bbc8f50135edf7823a8c8cb9366088e838233d074e",
    ),
    allowed(
        "ext_enqueue_attempt_resolve",
        Reason::A1,
        "additive",
        "bc7c9568df2d4203d34ccf84c10e54c8d83274e129868a8753ac962c51a1d4fd",
    ),
    allowed(
        "ext_release_enqueue",
        Reason::A1,
        "additive",
        "f9a057371b068ef3d57ae9fca19665690545410d9689d4e53666854762ebd634",
    ),
    allowed(
        "ext_retire_clear",
        Reason::A1,
        "one `DEL`, last",
        "7e3479c12f17f6538d4ccd780c5588a877680f9f93450c05e0007fdc17fcee53",
    ),
    allowed(
        "ext_retire_retry_claim",
        Reason::A1,
        "`HSETNX`, then reads of a non-empty carrier",
        "7b5545eec8079038ecb4cd40f962b2a87e492f917a6d8ff9ba358ac8c361ea42",
    ),
    allowed(
        "ext_retire_settle",
        Reason::A1,
        "`DEL`, then `HDEL` of the header, last",
        "c819ec723bc6afb3a32db6f393385c7d772b3fb8675bb9b32ecae0b57d53c105",
    ),
    allowed(
        "ext_store_value",
        Reason::A1,
        "the heal is a plain `SET`, never `DEL` then `SET` (decision 9)",
        "b3ca486e85d65ddeeac40d1701b22d6c67e024a976c1fc0bcb7c3f846e782e72",
    ),
    allowed(
        "gc_claim_pending",
        Reason::A1,
        "an emptied key is never named again",
        "d678b1d0dd65f3e38060ef9a94e4bba07e783ae84a9bf8922582ef8f99e262e8",
    ),
    allowed(
        "gc_complete",
        Reason::A1,
        "an emptied key is never named again",
        "2979213d3a4e5f35c62f6e938e7583ec736280daf41c0757dc796528806c07ab",
    ),
    allowed(
        "gc_decrement_and_collect",
        Reason::A1,
        "the out-of-memory opener (`ZREM` of a never-written member) is the first write; the \
         `DEL` is the last call, the `STRLEN` before it a `pcall`",
        "75be08e8e2cec4fcba0655c23c790269d6781c4d45b2bebd4821198f81d01de7",
    ),
    allowed(
        "gc_leader_acquire",
        Reason::A1,
        "per branch a `PEXPIRE` opener, then a `SET` of the same key, last; untagged lock",
        "2efbc72d2f1dc91e4e2b5ce4bcb186087bd3b298cd5f5a4eac121d0971ddc99f",
    ),
    allowed(
        "gc_oom_signal",
        Reason::A1,
        "a `PEXPIRE` opener, then a `SET` of the same key, last",
        "1ca621f19a8ebb8c858bc0f2edcbc557d406ae59b5eac2dfd72965e5eef5c5ea",
    ),
    allowed(
        "gc_reclaim_stale",
        Reason::A1,
        "one `ZADD XX`, last",
        "221525adfd079d126b58c2637d81f2e2921d190824fbde5d3ffde5b30fc8234b",
    ),
    allowed(
        "gc_stamp_unaged",
        Reason::A1,
        "additive",
        "2d2af903bef28e8d95fd86e8b7adc0b39fa95b12c808e3ff4b76353a079a048f",
    ),
    allowed(
        "gc_unmark",
        Reason::A1,
        "one `ZREM`, last",
        "05434477021ae9aeac5857888282cd0c855df4d6625ef580be4b7efb9f1258af",
    ),
    allowed(
        "leader_acquire_or_refresh",
        Reason::A1,
        "one write per branch; untagged lock",
        "a42d2b1da1614846d5729c7afbf050f7cc3666fe763e042cf83a48979e0c2782",
    ),
    allowed(
        "leader_refresh_if_held",
        Reason::A1,
        "one write per branch; untagged lock",
        "b1210516d5fc19b6603ea016535cc54c7aacb0dd8ba217cd15b5b467c9d0dd30",
    ),
    allowed(
        "presence_refresh",
        Reason::A1,
        "`HSET` of the info hash after `SET` of the lease, the info hash proven present",
        "adeb1aa7125d98144a6e6ee3e6894b5a0d2c9d7a7a2389091c078c04e196a898",
    ),
    allowed(
        "recon_claim_expired",
        Reason::A1,
        "`ZADD NX`, then a read; caller-chosen tag",
        "a949a741dc22ccfe250df56a59b47c9c61037a9b1bbd74c8c783a7a15f3c9cc1",
    ),
    allowed(
        "retired_object_wake",
        Reason::A1,
        "one key; the opening `PEXPIRE` names it, so a stream that moved is refused before any \
         write, and every later command names a present key",
        "9177194207451f84524d97467e79b7c3d37bc36c4e9159fbd78519c871de126b",
    ),
    allowed(
        "snapshot_fence_renew",
        Reason::A1,
        "one write per run",
        "70c2cb609499b3d603bb4392aae2a249cf0aa72fb1106a4cc8884ea7d24219b4",
    ),
    allowed(
        "snapshot_intent_renew",
        Reason::A1,
        "one write per run",
        "a8bea4db03148f20d6bf7cd183fcc2283b7490d502df4d6a989b3e5709b60800",
    ),
    allowed(
        "stamp_cursor_if_live",
        Reason::A1,
        "one `HSET`, last",
        "66785f0aca181929483fdc6736fb0355aa16afdb8ad21ab1ef680dfd1deab2c5",
    ),
    allowed(
        "stream_mux_wake",
        Reason::A1,
        "`XADD`, then `PEXPIRE` of the same key; caller-chosen tag",
        "37ec19b141f65d20dec1988811b8f596dc6aa7b8370a1539a449e36a41103df7",
    ),
    allowed(
        "gc_enqueue_attempt_audit",
        Reason::A2,
        "cannot empty while page entries remain; forget mode's later calls are no-ops",
        "20b6ef7c02ab456c148854a704751a9813ff3ba97f06416320864e8cdba1d2b7",
    ),
    allowed(
        "gc_settled_retire",
        Reason::A2,
        "after emptying, later calls are no-ops",
        "d42907d24c3b3f8e856f24e93201cc22261e6e126bbb93824ebbf5223b8e4793",
    ),
    allowed(
        "presence_test_evict",
        Reason::A2,
        "test-only; trailing `SCARD`",
        "f5a7900decd4c560efeaf08aa9ba889632d78b7b612e02a0bb0d590509e58ca9",
    ),
    allowed(
        "presence_unregister",
        Reason::A2,
        "trailing `SCARD`; never runs on a migrating source",
        "6df04ce908c1c3e6e5690d723fc93040f95bfc59d7aca4f0f14de6c508d3fa20",
    ),
    allowed(
        "session_checkpoint_publish",
        Reason::A2,
        "trailing `HGET`",
        "d27b52e5524aeb120246dd81d80024e37852dc5232bf7be8c771169271469c52",
    ),
    allowed(
        "session_checkpoint_read",
        Reason::A2,
        "its writes precede the refusable undeclared mapping reads",
        "7a147454910f8e03b9266012fb7fcfadc5940b5a8f6be2022ea5792b14029366",
    ),
    allowed(
        "ext_retire_begin",
        Reason::A3,
        "writes only with the carrier absent",
        "2fb6c22f4d8905be3846db7ff48368e10de8c8c14461f6eee60cf276b7ee481d",
    ),
    allowed(
        "presence_register",
        Reason::A3,
        "on a migrating source runs only with the teardown marker present, then returns",
        "08a77ce41a3b9c9338dcf471b3221a4d46efb9156b574f6a09666d89e93a1be3",
    ),
    allowed(
        "presence_sweep_batch",
        Reason::A3,
        "writes only for an absent lease",
        "f893c0aa90cde493146ae636b610775ceed9cb33ffac46f15f3c576ca7fdce0a",
    ),
    allowed(
        "snapshot_fence_acquire",
        Reason::A4,
        "a fence always holds `size_e` / `size_b`",
        "aabcfed738b0fc0a6bd988d7f642412fddab2418ac5f3859cb48b46d0fe05eba",
    ),
    allowed(
        "snapshot_fence_release",
        Reason::A4,
        "each later write runs only while `epoch_start`, `gap_until` or another live lease keeps \
         the fence non-empty",
        "69fe23e4ca3de01da6343cbf764814005a293f56da3c257eac1b40da3a3316dc",
    ),
    allowed(
        "ext_read_hold",
        Reason::A5,
        "after the out-of-memory opener (`HDEL` of a never-written field), `take` writes its own \
         field before pruning others (decision 9)",
        "c5f865d516ade7b544b6b380138b1a4dcb96eddab448dc872a80f06198c2009f",
    ),
    allowed(
        "snapshot_intent_register",
        Reason::A5,
        "writes its own field before pruning others (decision 9)",
        "03f13e029791b894043680c78f05bb9d72478772b0777b3d194dee48ddf476e1",
    ),
];

/// Read-only scripts whose command names are computed (a wrapper that
/// `pcall`s its argument), so the scan cannot see what they send. Each is
/// pinned by hash: an edit fails until someone re-confirms it still sends only
/// reads.
const READ_ONLY_COMPUTED: [(&str, &str); 3] = [
    (
        "fanout_window_read",
        "a0f97c02fe8fc30a493ab95d7dff49b41ff77b5e81336cd9e004a4cb80638b14",
    ),
    (
        "object_snapshot_read",
        "da0c6554e64b6b88b2ccbfeedb111074701db06d987addc0d5d8f4092899ae15",
    ),
    (
        "structural_cache_sync",
        "f0c51bf5a369d066c49c53582d3a7f8127b3f60cf191b6e42e036328a92be6cb",
    ),
];

/// Commands that write a key.
const WRITE_COMMANDS: [&str; 19] = [
    "SET", "DEL", "UNLINK", "HSET", "HSETNX", "HDEL", "HINCRBY", "INCR", "INCRBY", "DECRBY",
    "SADD", "SREM", "ZADD", "ZREM", "RPUSH", "PEXPIRE", "XADD", "XTRIM", "RENAME",
];

/// Commands that only read a key.
const READ_COMMANDS: [&str; 24] = [
    "GET",
    "EXISTS",
    "TYPE",
    "STRLEN",
    "PTTL",
    "HGET",
    "HMGET",
    "HGETALL",
    "HKEYS",
    "HLEN",
    "HEXISTS",
    "HSCAN",
    "SMEMBERS",
    "SISMEMBER",
    "SCARD",
    "SRANDMEMBER",
    "ZSCORE",
    "ZCARD",
    "ZRANGE",
    "ZRANGEBYSCORE",
    "LRANGE",
    "XRANGE",
    "XINFO",
    "XLEN",
];

/// Commands that name no key: `TIME`, and `PUBLISH`, whose channel is not a
/// key a migration moves.
const KEYLESS_COMMANDS: [&str; 2] = ["TIME", "PUBLISH"];

/// The scripts on every list, and the read-only ones on none, as of b.cyu:
/// a floor on what the inventory must see.
const SCRIPTS_WHEN_WRITTEN: usize = 49;

// ─────────────────────────────────────────────────────────────────────────────
// Classification
// ─────────────────────────────────────────────────────────────────────────────

/// What the closed-world scan makes of one script.
#[derive(Debug, Default, PartialEq, Eq)]
struct Scan {
    /// Literal commands on no list.
    unclassified: Vec<String>,
    /// How many calls name their command by anything but a literal.
    computed: usize,
    /// Whether any literal command is a [`WRITE_COMMANDS`] one.
    writes: bool,
    /// The 1-based lines opening a long bracket.
    long_brackets: Vec<usize>,
}

fn scan(source: &str) -> Scan {
    let mut out = Scan {
        long_brackets: long_bracket_lines(source),
        ..Scan::default()
    };
    for call in script_calls(&lua_code(source)) {
        match call {
            None => out.computed += 1,
            Some(command) if WRITE_COMMANDS.contains(&command.as_str()) => out.writes = true,
            Some(command)
                if READ_COMMANDS.contains(&command.as_str())
                    || KEYLESS_COMMANDS.contains(&command.as_str()) => {}
            Some(command) => out.unclassified.push(command),
        }
    }
    out
}

/// `name.lua` → `name`.
fn stem(file: &str) -> &str {
    file.strip_suffix(".lua").unwrap_or(file)
}

/// Every compiled script, by stem, with its source.
fn scripts() -> Vec<(String, String)> {
    compiled_scripts()
        .into_iter()
        .map(|(file, text)| (stem(&file).to_owned(), text))
        .collect()
}

fn source_of<'a>(scripts: &'a [(String, String)], name: &str) -> &'a str {
    scripts
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, text)| text.as_str())
        .unwrap_or_else(|| panic!("`{name}` is on a list but is not a compiled script"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Rust sources
// ─────────────────────────────────────────────────────────────────────────────

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.rs` file under `src/`, relative to the crate root, with its
/// production code: comments stripped (string literals kept) and the
/// `#[cfg(test)] mod tests` block cut, joined by newlines.
pub(crate) fn production_sources() -> Vec<(PathBuf, String)> {
    let root = crate_root();
    let mut files = Vec::new();
    rust_sources_under(&root, &root.join("src"), &mut files);
    files.sort();
    files
        .into_iter()
        .map(|file| {
            let text = std::fs::read_to_string(root.join(&file))
                .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
            let production = text
                .split("#[cfg(test)]\nmod tests")
                .next()
                .unwrap_or("")
                .to_owned();
            (file, code_lines(&production).join("\n"))
        })
        .collect()
}

/// The name of the `fn` whose body holds byte `at` of `code`: the nearest
/// `fn <name>` before it.
fn enclosing_fn(code: &str, at: usize) -> String {
    enclosing_fn_at(code, at).1
}

/// The byte offset and name of the nearest `fn <name>` before byte `at` of
/// `code` — `(0, "<file scope>")` when there is none. The offset tells two
/// `fn`s of one name apart.
pub(crate) fn enclosing_fn_at(code: &str, at: usize) -> (usize, String) {
    let before = &code[..at];
    let mut from = before.len();
    while let Some(found) = before[..from].rfind("fn ") {
        let boundary = found == 0
            || !before.as_bytes()[found - 1].is_ascii_alphanumeric()
                && before.as_bytes()[found - 1] != b'_';
        if boundary {
            let name = before[found + 3..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            return (found, name);
        }
        from = found;
    }
    (0, "<file scope>".to_owned())
}

/// Every call of `name(` in `code` that is not its definition (`fn name(`),
/// as the byte offset of the name.
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

// ─────────────────────────────────────────────────────────────────────────────
// The guards
// ─────────────────────────────────────────────────────────────────────────────

/// Every script on disk is compiled in, and every compiled script is on disk;
/// the inventory sees at least as many scripts as when this guard was written,
/// and its landmarks.
#[test]
fn every_script_on_disk_is_compiled_in_and_every_compiled_script_is_on_disk() {
    let on_disk: Vec<String> = script_files();
    let compiled: Vec<String> = compiled_scripts().into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        on_disk, compiled,
        "the scripts in src/storage/scripts/ and the scripts `include_str!`ed under src/ must be \
         the same set: a script on disk that nothing compiles in is unreviewed dead code, and one \
         compiled from elsewhere escapes this guard"
    );
    assert!(
        on_disk.len() >= SCRIPTS_WHEN_WRITTEN,
        "the inventory found {} scripts, fewer than the {SCRIPTS_WHEN_WRITTEN} this guard was \
         written against: the walk has stopped reaching them",
        on_disk.len()
    );
    for landmark in [
        "update_layer_fast.lua",
        "gc_drop_foreign_claims.lua",
        "presence_test_evict.lua",
        "object_snapshot_read.lua",
    ] {
        assert!(
            on_disk.iter().any(|n| n == landmark),
            "the inventory must see {landmark}"
        );
    }
}

/// Every command every script sends is named by a literal on a list, except in
/// a pinned read-only script with computed names; no script opens a long
/// bracket.
#[test]
fn every_script_command_is_classified_and_only_pinned_read_only_scripts_compute_one() {
    let mut problems = Vec::new();
    for (name, source) in scripts() {
        let found = scan(&source);
        if !found.unclassified.is_empty() {
            problems.push(format!(
                "{name}: unclassified command(s) {:?} — add each to WRITE_COMMANDS or \
                 READ_COMMANDS (or KEYLESS_COMMANDS) in this guard",
                found.unclassified
            ));
        }
        let computed_allowed = READ_ONLY_COMPUTED.iter().any(|(n, _)| *n == name);
        if found.computed > 0 && !computed_allowed {
            problems.push(format!(
                "{name}: {} call(s) whose command is not a string literal; the scan cannot see \
                 what they send. Name the command literally, or, for a read-only script, add a \
                 READ_ONLY_COMPUTED row with its hash",
                found.computed
            ));
        }
        if !found.long_brackets.is_empty() {
            problems.push(format!(
                "{name}: long bracket(s) at line(s) {:?}, which the scanner does not model",
                found.long_brackets
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// Every writing script is on exactly one of [`SENTINEL`] and [`ALLOWLIST`],
/// every script on either list writes, and every [`READ_ONLY_COMPUTED`]
/// script writes nothing it names.
#[test]
fn every_writing_script_is_on_exactly_one_list_and_no_read_only_script_is_on_either() {
    let scripts = scripts();
    let writing: BTreeSet<String> = scripts
        .iter()
        .filter(|(_, source)| scan(source).writes)
        .map(|(name, _)| name.clone())
        .collect();
    let sentinel: BTreeSet<String> = SENTINEL.iter().map(|s| s.script.to_owned()).collect();
    let allowlisted: BTreeSet<String> = ALLOWLIST.iter().map(|a| a.script.to_owned()).collect();
    assert_eq!(
        sentinel.len() + allowlisted.len(),
        SENTINEL.len() + ALLOWLIST.len(),
        "a script is listed twice within one list"
    );
    let on_both: Vec<_> = sentinel.intersection(&allowlisted).collect();
    assert!(on_both.is_empty(), "on both lists: {on_both:?}");
    let listed: BTreeSet<String> = sentinel.union(&allowlisted).cloned().collect();
    let unlisted: Vec<_> = writing.difference(&listed).collect();
    assert!(
        unlisted.is_empty(),
        "writing script(s) on neither list: {unlisted:?}. A script that can stop part-way during \
         a slot migration (a command naming a key the run emptied, or an undeclared key, after \
         its first write) declares its slot's migration sentinel and joins SENTINEL; one that \
         cannot joins ALLOWLIST with its reason and hash. See SDD §18.1"
    );
    let not_writing: Vec<_> = listed.difference(&writing).collect();
    assert!(
        not_writing.is_empty(),
        "listed script(s) that send no write command: {not_writing:?}; a read-only script is on \
         neither list"
    );
    for (name, _) in READ_ONLY_COMPUTED {
        assert!(
            !writing.contains(name),
            "{name} is on READ_ONLY_COMPUTED but sends a write command"
        );
    }
    let read_only = scripts.len() - writing.len();
    assert_eq!(
        (sentinel.len(), allowlisted.len(), read_only),
        (5, 41, 8),
        "the lists as SDD §18.1 states them: 5 sentinel, 41 allowlisted, 8 read-only. A change \
         here is a change to the section"
    );
}

/// Each [`ALLOWLIST`] and [`READ_ONLY_COMPUTED`] script's normalized source
/// hashes to its pin (decision 11).
#[test]
fn every_pinned_script_matches_its_hash() {
    let scripts = scripts();
    let pins = ALLOWLIST
        .iter()
        .map(|a| (a.script, a.sha256, format!("{:?}: {}", a.reason, a.note)))
        .chain(
            READ_ONLY_COMPUTED
                .iter()
                .map(|(n, h)| (*n, *h, "read-only, computed command names".to_owned())),
        );
    let mut moved = Vec::new();
    for (name, pinned, reason) in pins {
        let actual = sha256_hex(&normalized_code(source_of(&scripts, name)));
        if actual != pinned {
            moved.push(format!(
                "  {name} ({reason})\n    pinned: {pinned:?}\n    now:    \"{actual}\""
            ));
        }
    }
    assert!(
        moved.is_empty(),
        "these pinned scripts' code changed:\n{}\n\nFor each, re-read the script against its \
         reason — can it now stop part-way during a slot migration, a command naming a key the \
         run emptied or never declared after its first write? (SDD §18.1) — then update its \
         hash here, or move it to SENTINEL. Comment and whitespace edits do not move a hash.",
        moved.join("\n")
    );
}

/// Each [`SENTINEL`] script declares its slot and checks it before any
/// command, and no command's argument list names the slot as
/// [`sentinel_named_by_a_command`] reads it.
#[test]
fn every_sentinel_script_checks_its_sentinel_before_any_command_and_names_it_in_none() {
    let scripts = scripts();
    for sentinel in &SENTINEL {
        let code = lua_code(source_of(&scripts, sentinel.script));
        let normalized: Vec<String> = code
            .iter()
            .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
            .collect();
        let mut check_ends = 0;
        for needle in sentinel.check {
            let line = normalized
                .iter()
                .position(|line| line.contains(needle))
                .unwrap_or_else(|| {
                    panic!(
                        "{}: the sentinel check must contain `{needle}`",
                        sentinel.script
                    )
                });
            check_ends = check_ends.max(line);
        }
        let first_command = code
            .iter()
            .position(|line| !script_calls(std::slice::from_ref(line)).is_empty())
            .expect("a sentinel script sends commands");
        assert!(
            check_ends < first_command,
            "{}: the sentinel check (to line {}) must come before the first command (line {})",
            sentinel.script,
            check_ends + 1,
            first_command + 1
        );
        let forged = sentinel_named_by_a_command(&code, sentinel.slot);
        assert!(
            forged.is_empty(),
            "{}: these command calls' argument lists name the sentinel slot; the sentinel is \
             never read or written: {forged:?}",
            sentinel.script
        );
    }
}

/// The command calls in `code` (from `lua_code`) whose argument list names
/// the sentinel, each as its argument list with whitespace collapsed.
///
/// Each call's whole argument list is read, from its `(` to the matching `)`
/// across lines, quoted strings skipped. It names the sentinel when it holds
/// `KEYS[<slot>]` (any spacing), or, outside a string, a word in
/// [`sentinel_names`]: the `KEY_MIGRATION_SENTINEL` constant, the check's
/// `sentinel` local, or a variable assigned from one of those. A key reached
/// any other way — a computed index, a function parameter — is beyond this
/// scan; a write to it is caught at run time, where each sentinel script's
/// landed control asserts the sentinel still absent.
fn sentinel_named_by_a_command(code: &[String], slot: usize) -> Vec<String> {
    let slot_literal = format!("KEYS[{slot}]");
    let names = sentinel_names(code, slot);
    call_argument_lists(&code.join("\n"))
        .into_iter()
        .filter(|args| names_the_sentinel(args, &slot_literal, &names))
        .map(|args| args.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

/// Whether the Lua expression `text` names the sentinel: `slot_literal`
/// (`KEYS[<slot>]`, whitespace ignored), or a word of `names` outside a
/// quoted string.
fn names_the_sentinel(text: &str, slot_literal: &str, names: &BTreeSet<String>) -> bool {
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    compact.contains(slot_literal)
        || without_strings(text)
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|word| names.contains(word))
}

/// The words that name the sentinel in `code`: `KEY_MIGRATION_SENTINEL`,
/// `sentinel`, and every variable assigned — `local a = …`, `a = …`,
/// `a.f = …` (which taints `a`) — from an expression that names the sentinel
/// by `KEYS[<slot>]` or by a word already in the set, to a fixed point. An
/// assignment's right-hand side is the rest of its line, or the next line
/// when nothing follows the `=`.
fn sentinel_names(code: &[String], slot: usize) -> BTreeSet<String> {
    let slot_literal = format!("KEYS[{slot}]");
    let mut names: BTreeSet<String> = ["KEY_MIGRATION_SENTINEL", "sentinel"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    let assignments: Vec<(Vec<String>, String)> = code
        .iter()
        .enumerate()
        .filter_map(|(i, line)| {
            let (lhs, rhs) = split_assignment(line)?;
            let rhs = if rhs.trim().is_empty() {
                code[i + 1..]
                    .iter()
                    .find(|l| !l.trim().is_empty())
                    .cloned()
                    .unwrap_or_default()
            } else {
                rhs
            };
            Some((lhs, rhs))
        })
        .collect();
    loop {
        let before = names.len();
        for (lhs, rhs) in &assignments {
            if names_the_sentinel(rhs, &slot_literal, &names) {
                names.extend(lhs.iter().cloned());
            }
        }
        if names.len() == before {
            return names;
        }
    }
}

/// `line` split at its first assignment `=` (not `==`, `~=`, `<=`, `>=`, nor
/// one inside a string) into the assigned names — for each comma-separated
/// target, the leading identifier of its last word, so `local a`, `for a`,
/// `then a` and `a.f` / `a[i]` all give `a` — and the text after the `=`;
/// `None` for a line that assigns nothing.
fn split_assignment(line: &str) -> Option<(Vec<String>, String)> {
    let blanked = without_strings(line);
    let bytes = blanked.as_bytes();
    let at = (0..bytes.len()).find(|&i| {
        bytes[i] == b'='
            && bytes.get(i + 1) != Some(&b'=')
            && (i == 0 || !matches!(bytes[i - 1], b'=' | b'~' | b'<' | b'>'))
    })?;
    let names: Vec<String> = blanked[..at]
        .split(',')
        .filter_map(|target| target.split_whitespace().last())
        .map(|target| {
            target
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect::<String>()
        })
        .filter(|n| !n.is_empty())
        .collect();
    (!names.is_empty()).then(|| (names, line[at + 1..].to_owned()))
}

/// `text` with the contents of each quoted string blanked to spaces (the
/// quotes kept), so a word inside a string is not read as code. Byte
/// offsets are preserved.
fn without_strings(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for c in text.chars() {
        match quote {
            Some(q) => {
                if !escaped && c == q {
                    quote = None;
                    out.push(c);
                } else {
                    escaped = !escaped && c == '\\';
                    out.extend(std::iter::repeat_n(' ', c.len_utf8()));
                }
            }
            None => {
                if c == '"' || c == '\'' {
                    quote = Some(c);
                }
                out.push(c);
            }
        }
    }
    out
}

/// The argument list of every [`CALL_FUNCTIONS`] call in `joined` (Lua code,
/// comments stripped, lines joined), from just after its `(` to just before
/// the matching `)` — across lines, parentheses inside quoted strings
/// ignored. A call left unclosed yields the rest of the text. A call
/// function taken as a value (no `(`) has no argument list.
fn call_argument_lists(joined: &str) -> Vec<String> {
    let blanked = without_strings(joined);
    let bytes = blanked.as_bytes();
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut found: Vec<(usize, String)> = Vec::new();
    for function in CALL_FUNCTIONS {
        for (at, _) in blanked.match_indices(function) {
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
                continue;
            }
            let open = i + 1;
            let mut depth = 1usize;
            let mut close = bytes.len();
            for (j, b) in bytes.iter().enumerate().skip(open) {
                match b {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            close = j;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            found.push((at, joined[open..close].to_owned()));
        }
    }
    found.sort_by_key(|(at, _)| *at);
    found.into_iter().map(|(_, args)| args).collect()
}

/// `migration-sentinel` is spelled in code only by its builder in
/// `src/storage/keys.rs` and the five sentinel scripts' checks: no other
/// script names the key, and no Rust code builds it by hand.
#[test]
fn the_sentinel_key_is_spelled_only_by_its_builder_and_the_sentinel_scripts() {
    const SPELLING: &str = "migration-sentinel";
    let mut rust: Vec<String> = production_sources()
        .into_iter()
        .filter(|(_, code)| code.contains(SPELLING))
        .map(|(file, code)| format!("{} ×{}", file.display(), code.matches(SPELLING).count()))
        .collect();
    rust.sort();
    assert_eq!(
        rust,
        vec!["src/storage/keys.rs ×1".to_owned()],
        "only `KeySchema::migration_sentinel` may spell the sentinel key in Rust code"
    );
    let mut lua: Vec<String> = scripts()
        .into_iter()
        .filter(|(_, source)| lua_code(source).iter().any(|l| l.contains(SPELLING)))
        .map(|(name, _)| name)
        .collect();
    lua.sort();
    let mut expected: Vec<String> = SENTINEL.iter().map(|s| s.script.to_owned()).collect();
    expected.sort();
    assert_eq!(
        lua, expected,
        "only the sentinel scripts' checks may spell the sentinel key in Lua code"
    );
}

/// `KeySchema::migration_sentinel` is called only at the five `KEYS`-assembly
/// sites, one per sentinel script's caller.
#[test]
fn the_sentinel_builder_is_called_only_at_the_sentinel_scripts_keys_assembly_sites() {
    let mut sites: Vec<String> = Vec::new();
    for (file, code) in production_sources() {
        for at in calls_of(&code, "migration_sentinel") {
            sites.push(format!("{}::{}", file.display(), enclosing_fn(&code, at)));
        }
    }
    sites.sort();
    assert_eq!(
        sites,
        vec![
            "src/storage/ext_index_audit.rs::run_script".to_owned(),
            "src/storage/external_values.rs::drop_foreign_claims".to_owned(),
            "src/storage/external_values.rs::gc_audit_page".to_owned(),
            "src/storage/lua.rs::update_script_shared_keys".to_owned(),
            "src/storage/time_sample_merge.rs::apply_signed_zero_merge".to_owned(),
        ],
        "the migration sentinel is built only where a sentinel script's KEYS are assembled; a \
         new site either serves a new sentinel script (add it to SENTINEL) or misuses the key"
    );
}

/// Each sentinel script's production `EVALSHA` passes the sentinel among its
/// keys: inline, or in the `let` binding of the keys variable it passes. The
/// commit's keys come from `lua::update_script_shared_keys`, whose sentinel
/// slot is checked here; that the production `EVALSHA` carries it at
/// `KEYS[10]` is pinned on the wire by `update_layer_lua_test`.
#[test]
fn each_sentinel_scripts_evalsha_keys_carry_the_sentinel() {
    let sources = production_sources();
    let mut checked = Vec::new();
    for script_static in [
        "GC_DROP_FOREIGN_CLAIMS_SCRIPT",
        "GC_LEDGER_AUDIT_SCRIPT",
        "EXT_INDEX_RECOUNT_SCRIPT",
        "TIME_SAMPLE_SIGNED_ZERO_MERGE_SCRIPT",
    ] {
        let mut sites = 0;
        for (file, code) in &sources {
            for (at, _) in code.match_indices(script_static) {
                let rest = &code[at + script_static.len()..];
                let trimmed = rest.trim_start();
                let Some(after) = trimmed.strip_prefix(".evalsha_with_reload") else {
                    continue;
                };
                let after_name = code.len() - after.len();
                let args = call_arguments(code, after_name).unwrap_or_else(|| {
                    panic!("{}: cannot parse {script_static}'s call", file.display())
                });
                let keys = args.get(1).unwrap_or_else(|| {
                    panic!(
                        "{}: {script_static}'s call has no keys argument",
                        file.display()
                    )
                });
                assert!(
                    keys_expression_carries_the_sentinel(code, at, keys),
                    "{}: {script_static}'s EVALSHA keys `{keys}` do not carry the migration \
                     sentinel",
                    file.display()
                );
                sites += 1;
            }
        }
        assert!(
            sites > 0,
            "found no `{script_static}.evalsha_with_reload` call; the scan no longer sees the \
             script's caller"
        );
        checked.push(script_static);
    }

    let (_, lua) = sources
        .iter()
        .find(|(file, _)| file == Path::new("src/storage/lua.rs"))
        .expect("src/storage/lua.rs");
    let builder_at = lua
        .find("pub fn update_script_shared_keys(")
        .expect("lua.rs defines update_script_shared_keys");
    let body = &lua[builder_at..builder_at + lua[builder_at..].find("\n}").expect("its body ends")];
    assert!(
        body.contains(
            "fixed[UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL - 1] = keys.migration_sentinel(&oid_str);"
        ),
        "the commit's shared KEYS builder must fill the sentinel slot with the object's sentinel"
    );
    assert_eq!(checked.len(), 4);
}

/// Whether the keys expression `keys` of a call at `at` in `code` carries the
/// sentinel: inline, or through the `let` binding of the identifier it names
/// in the same `fn`.
fn keys_expression_carries_the_sentinel(code: &str, at: usize, keys: &str) -> bool {
    if keys.contains("migration_sentinel(") {
        return true;
    }
    let ident = keys.trim_start_matches('&');
    if !ident.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return false;
    }
    let fn_start = code[..at].rfind("fn ").unwrap_or(0);
    let body = &code[fn_start..at];
    let Some(binding) = [format!("let {ident} ="), format!("let mut {ident} =")]
        .iter()
        .filter_map(|needle| body.rfind(needle.as_str()))
        .max()
    else {
        return false;
    };
    let expression = &body[binding..];
    let end = expression.find(';').unwrap_or(expression.len());
    expression[..end].contains("migration_sentinel(")
}

/// SDD §18.1 names every script on every list as its bare name in backticks.
#[test]
fn sdd_18_1_names_every_sentinel_allowlisted_and_read_only_script() {
    let sdd = std::fs::read_to_string(crate_root().join("docs-internal/sdd.md"))
        .expect("docs-internal/sdd.md must be readable");
    let start = sdd.find("\n### 18.1 ").expect("the SDD has §18.1");
    let end = start + 1 + sdd[start + 1..].find("\n### ").expect("§18.1 ends");
    let section = &sdd[start..end];
    let scripts = scripts();
    let writing: BTreeSet<&str> = scripts
        .iter()
        .filter(|(_, s)| scan(s).writes)
        .map(|(n, _)| n.as_str())
        .collect();
    let names = SENTINEL
        .iter()
        .map(|s| s.script)
        .chain(ALLOWLIST.iter().map(|a| a.script))
        .chain(
            scripts
                .iter()
                .map(|(n, _)| n.as_str())
                .filter(|n| !writing.contains(n)),
        );
    let missing: Vec<&str> = names
        .filter(|name| !section.contains(&format!("`{name}`")))
        .collect();
    assert!(
        missing.is_empty(),
        "SDD §18.1 must name every script on the lists (and every read-only one) as `name`: \
         missing {missing:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Scanner self-tests
// ─────────────────────────────────────────────────────────────────────────────

/// The command scan skips comments (a `--` inside a string is not one), reads
/// both quote styles and every call function, sees a computed command and a
/// call function taken as a value, and a literal on no list is unclassified.
#[test]
fn the_command_scanner_skips_comments_and_sees_computed_and_unknown_commands() {
    let source = r#"
local t = redis.call("TIME")
redis.call('HSET', KEYS[1], "f", "a -- not a comment")
-- redis.call("DEL", KEYS[1])
x = 1 -- redis.call("UNLINK", KEYS[1])
redis.pcall(
  "GET", KEYS[1])
server.call("EXISTS", KEYS[1])
local r = redis.pcall(cmd, key)
local f = redis.call
redis.call("FLUSHALL")
"#;
    let calls = script_calls(&lua_code(source));
    assert_eq!(
        calls,
        vec![
            Some("TIME".to_owned()),
            Some("HSET".to_owned()),
            Some("GET".to_owned()),
            Some("EXISTS".to_owned()),
            None,
            None,
            Some("FLUSHALL".to_owned()),
        ]
    );
    let found = scan(source);
    assert_eq!(found.unclassified, vec!["FLUSHALL".to_owned()]);
    assert_eq!(found.computed, 2);
    assert!(
        found.writes,
        "the HSET writes; the commented DEL / UNLINK do not count"
    );
    assert!(
        !scan("redis.call('GET', KEYS[1])\n-- redis.call('DEL', KEYS[1])").writes,
        "a commented-out write is not a write"
    );
}

/// A long-bracket comment or string is reported, and ordinary indexing is not.
#[test]
fn the_long_bracket_finder_reports_long_brackets_only() {
    assert_eq!(
        long_bracket_lines("local a = KEYS[KEY_HEADER]\nlocal b = t[x[1]]\n--[[ block\n]]\nlocal s = [==[ long ]==]\n-- [not long]\n"),
        vec![3, 5]
    );
}

/// The normalized hash ignores comments and whitespace and sees code.
#[test]
fn the_pinned_hash_ignores_comments_and_layout_but_not_code() {
    let base = "local a = KEYS[1]\nredis.call(\"DEL\", a)\n";
    let relaid = "-- a header\nlocal  a = KEYS[1]   -- the key\n\n\tredis.call(\"DEL\", a)\n";
    let edited = "local a = KEYS[1]\nredis.call(\"UNLINK\", a)\n";
    assert_eq!(
        sha256_hex(&normalized_code(base)),
        sha256_hex(&normalized_code(relaid))
    );
    assert_ne!(
        sha256_hex(&normalized_code(base)),
        sha256_hex(&normalized_code(edited))
    );
}

/// A command naming the sentinel — by its literal slot, by its constant, by
/// the check's local, or by a local assigned from one of them, on the call's
/// line or on a later line of a call that spans several — is caught. The
/// check itself, which names it outside any command, is not; nor is a call
/// whose only mention of the sentinel is inside a string, a call naming
/// another slot, or a local assigned from the suffix string.
#[test]
fn a_command_naming_the_sentinel_is_caught() {
    let check = "local KEY_MIGRATION_SENTINEL = 7\nlocal sentinel = KEYS[KEY_MIGRATION_SENTINEL]\n\
                 local suffix = ':migration-sentinel'\nif #KEYS ~= KEY_MIGRATION_SENTINEL\n    or \
                 string.sub(sentinel, -string.len(suffix)) ~= suffix\n    or hash_tag(sentinel) \
                 ~= hash_tag(KEYS[1]) then\n  return redis.error_reply('ENCODING: KEYS[' .. \
                 KEY_MIGRATION_SENTINEL .. '] is not the sentinel')\nend\n";
    let clean = format!(
        "{check}redis.call('GET', KEYS[1])\nredis.call('SET', KEYS[17], 'sentinel')\n\
         redis.pcall(\n  'HSET',\n  KEYS[2], 'f', suffix)\nlocal n = #KEYS\n"
    );
    assert_eq!(
        sentinel_named_by_a_command(&lua_code(&clean), 7),
        Vec::<String>::new(),
        "the check, a string mentioning the sentinel, another slot and the suffix local are not \
         a command naming it"
    );
    for forged in [
        "redis.call('EXISTS', KEYS[7])",
        "redis.call('EXISTS', KEYS[ 7 ])",
        "redis.call('EXISTS', KEYS[KEY_MIGRATION_SENTINEL])",
        "redis.call('EXISTS', sentinel)",
        // Multi-line: the key on a later line than the call.
        "redis.pcall(\n  'XADD',\n  KEYS[KEY_MIGRATION_SENTINEL], '*', 'f', 'v')",
        "redis.call('EXISTS',\n    KEYS[7])",
        "server.call(\n  'SET',\n  sentinel, -- a comment\n  '1')",
        // A nested call's parentheses do not end the argument list early.
        "redis.call('HSET', tostring(KEYS[1]), 'f',\n  sentinel)",
        // Local aliases, direct, chained and assigned on the next line.
        "local s = sentinel\nredis.call('SET', s, '1')",
        "local k = KEYS[7]\nredis.call('DEL', k)",
        "local a = KEYS[ 7 ]\nlocal b = a\nredis.call('GET',\n  b)",
        "local k =\n  KEYS[KEY_MIGRATION_SENTINEL]\nredis.call('DEL', k)",
        "local t = {}\nt.key = sentinel\nredis.call('DEL', t.key)",
    ] {
        let source = format!("{check}{forged}\n");
        assert_eq!(
            sentinel_named_by_a_command(&lua_code(&source), 7).len(),
            1,
            "{forged:?} names the sentinel"
        );
    }
}

/// The Rust helpers: a call's enclosing `fn`, a call that is not the
/// definition, and a keys expression carrying the sentinel inline or through
/// its binding.
#[test]
fn the_rust_scanners_find_the_enclosing_fn_and_the_keys_binding() {
    let code = "fn migration_sentinel(&self) {}\nasync fn page(&self) {\n    let keys = \
                vec![a(), self.keys.migration_sentinel(\"gc\")];\n    S.evalsha_with_reload(p, \
                keys, argv);\n    S.evalsha_with_reload(p, vec![a()], argv);\n}\n";
    let calls = calls_of(code, "migration_sentinel");
    assert_eq!(calls.len(), 1, "the definition is not a call");
    assert_eq!(enclosing_fn(code, calls[0]), "page");
    let first = code.find("S.evalsha").expect("first call");
    assert!(keys_expression_carries_the_sentinel(code, first, "keys"));
    let second = code.rfind("S.evalsha").expect("second call");
    assert!(!keys_expression_carries_the_sentinel(
        code,
        second,
        "vec![a()]"
    ));
    assert!(keys_expression_carries_the_sentinel(
        code,
        second,
        "vec![a(), k.migration_sentinel(\"gc\")]"
    ));
}
