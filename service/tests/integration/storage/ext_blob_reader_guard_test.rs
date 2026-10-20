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

//! b.bnn — a source guard over every production site that can read
//! external-value blob bytes (`le:ext:data:{hash}`; SDD §11.2, §11.3).
//!
//! The GC may collect a released blob 5 s after its release
//! (`GcPolicy::PRODUCTION`), and at any age during an out-of-memory
//! collection (`GcPolicy::OOM_COLLECTION`, b.pa1). That is safe only because
//! every reader of blob
//! bytes treats a blob collected under its snapshot as a stale read: it probes,
//! and on a miss takes a `{gc}` read hold, re-reads the layer and rebuilds
//! (the miss protocol). A new reader written without the protocol passes every
//! behavioural test that does not stage a GC between its snapshot and its
//! read, and can hand a client a delta whose blob is gone. This guard fails
//! when such a site appears: every production site that can reach blob bytes is
//! listed in [`ALLOWED_SITES`] with the reason it is safe, and a site the list
//! does not name fails until it is justified there.
//!
//! **What counts as a site.** Code under `src/` as the feature-free release
//! build compiles it (comments, string literal contents, and items under
//! `test` or a test feature such as `test-support` removed, by the
//! `ProductionTree::load_release` view shared with
//! `refusal_and_shutdown_guards_test.rs`), searched for:
//!
//! - the key builder `KeySchema::ext_data` ([`KEY_BUILDER`]), called or named
//!   by path. Every Valkey read of a blob needs its key, and this is the only
//!   builder of it;
//! - the `ExternalValueStore` methods that return blob bytes
//!   ([`BLOB_READ_METHODS`]), called or named by path. The list is checked
//!   against the trait: every trait method whose return type carries bytes
//!   ([`BYTES_RETURN_TYPES`]: `Vec<u8>`, `[u8]` as in `Arc<[u8]>`, `Bytes`,
//!   or `ExtValueProbe`) must be in it, so a new read method joins the scan;
//! - the key's literal spelling [`KEY_LITERAL`] in code or a string, which
//!   only the builder's own `format!` may hold: a hand-built key would bypass
//!   the first search.
//!
//! Matching is by identifier, not by type: an unrelated method that happens to
//! share a scanned name (a `load_values` on another type) is reported too, and
//! needs a row saying it reads no external-value blob. Each hit is attributed
//! to the item it sits in (`Type::method`, or a free function's name), so a
//! row survives line moves and a second site in the same function still fails,
//! through the row's count.
//!
//! **Lua scripts.** What keeps a script from reaching a blob key on its own is
//! the key's name, not its slot: in standalone Valkey a script can address any
//! key, and even in Cluster mode `ext:meta` and `ext:applied` share the blob's
//! `{hash}` slot, so a script handed either could name the blob beside it. A
//! script that does not spell the name takes the key from `KEYS`, and every
//! blob key handed over is built by a Rust site this scan lists. The Rust call
//! site therefore stands for its script, and a blob key handed to a script
//! from a new site fails here as a new `ext_data` site. Two things the Rust
//! scan cannot see are checked over the scripts themselves
//! ([`every_script_handed_a_blob_key_never_reads_its_bytes`]): no script
//! spells [`KEY_LITERAL`] in code, and each script an allowlisted site hands a
//! blob key to ([`SCRIPTS_HANDED_A_BLOB_KEY`]) reaches the key only through
//! its one binding and runs only the commands its row lists on it, each named
//! by a literal, none of which returns the bytes. That keeps those rows'
//! "never reads blob bytes" reason checked rather than asserted.
//!
//! **What it cannot see** (a partial guard, by design; review and the SDD
//! rule cover the rest):
//!
//! - the protocol itself. A new caller of an allowlisted helper — say a new
//!   caller of `diff_read::resolve_external_values`, or a new catch-up build —
//!   adds no site here. The catch-up side is pinned by
//!   `grpc::b_yuo_catch_up_miss_test`'s source scans (every builder call inside
//!   the miss protocol; the fetch-on-miss mode only from a live hold's proof);
//! - a listed site whose body changes what it does with the key (a `STRLEN`
//!   turned into a `GET`) without changing its count;
//! - a key reached by a wider pattern (a `SCAN` over `ext:*`), an alias of a
//!   script's key variable, a computed `KEYS` index (`KEYS[i]`), or a macro
//!   that assembles an identifier;
//! - a blob key derived from a sibling key by a string edit (Lua
//!   `string.gsub(meta_key, ":meta:", ":data:")`, Rust
//!   `keys.ext_meta(h).replace("meta", "data")`): neither spells
//!   [`KEY_LITERAL`] or calls the builder;
//! - a new read method whose bytes travel in a type of its own (a new struct,
//!   as `ExtValueProbe` is) that [`BYTES_RETURN_TYPES`] does not name;
//! - the checkpoint adapter's mirror copy (`CheckpointStore::get_ext_blob`),
//!   deliberately: it is not a Valkey key, the GC never deletes it
//!   (checkpoint retention does), so the GC's collection does not reach it. The
//!   store itself reads it only from Recover's miss arms, behind the
//!   genuine-miss check.
//!
//! Static: reads source files, needs no Valkey and no feature.

// Guard tests: every test in this file is pinned by name in
// scripts/guard-tests.sh; rename or remove one there too.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::lua_source_scan::lua_code;
use super::refusal_and_shutdown_guards_test::{
    is_ident, production_code, FlatCode, ProductionCode, ProductionTree,
};
use super::txn_checkout_test::needle_hits;

/// The blob key builder's name.
const KEY_BUILDER: &str = "ext_data";

/// Where it is defined, and the item it is, as [`item_spans`] names it.
const KEY_BUILDER_FILE: &str = "src/storage/keys.rs";
const KEY_BUILDER_ITEM: &str = "KeySchema::ext_data";

/// The blob key's spelling, `le:ext:data:{hash_hex}` without its prefix.
const KEY_LITERAL: &str = "ext:data";

/// The external value store's module, and its trait.
const STORE_FILE: &str = "src/storage/external_values.rs";
const STORE_TRAIT_HEADER: &str = "pub trait ExternalValueStore";

/// The catch-up build's blob read.
const DIFF_READ_FILE: &str = "src/storage/diff_read.rs";

/// The backup confirmations' module, home of the backup repair's reads of a
/// stored value (b.wsx).
const EXT_BACKUP_FILE: &str = "src/storage/ext_backup.rs";

/// The `ExternalValueStore` methods that hand blob bytes to their caller,
/// sorted; checked against the trait by [`trait_blob_read_methods`].
const BLOB_READ_METHODS: [&str; 4] = [
    "load_values",
    "probe_values",
    "read_batch",
    "read_batch_with_recovery_counter",
];

/// Return-type spellings that carry blob bytes, for [`trait_blob_read_methods`],
/// each matched as a whole token: an owned buffer, a byte slice behind any
/// pointer (`Arc<[u8]>`, `Box<[u8]>`), `bytes::Bytes`, and the probe's answer.
const BYTES_RETURN_TYPES: [&str; 4] = ["Vec<u8>", "[u8]", "Bytes", "ExtValueProbe"];

/// One allowlisted site: `count` hits of `needle` inside `item` of `file`.
struct AllowedSite {
    file: &'static str,
    item: &'static str,
    needle: &'static str,
    count: usize,
    why: &'static str,
}

/// Every production site that can reach external-value blob bytes, exactly.
/// A reason is one of six: the site follows the miss protocol; it never reads
/// blob bytes (a write, a delete, an existence or size probe); it runs under
/// the GC's own leader lock and so cannot race the GC; it is a store's own
/// implementation of a read method, or a delegation between those methods (a
/// wrapping implementation's included), which hands the bytes only to the
/// method's callers, each a row of its own; it serves no snapshot, so a miss
/// only means the value was collected and has nothing left to serve (the
/// backup copy's reader); or (none today) it is an unrelated method that only
/// shares a scanned name.
const ALLOWED_SITES: [AllowedSite; 13] = [
    // ── The key builder ────────────────────────────────────────────────────
    AllowedSite {
        file: STORE_FILE,
        item: "read_release_blob_size",
        needle: KEY_BUILDER,
        count: 1,
        why: "never reads blob bytes: `STRLEN` only, the release ledger's byte accounting \
              (`release_entry_argv` issues it, at most `RELEASE_SIZE_READS_IN_FLIGHT` at a \
              time)",
    },
    AllowedSite {
        file: STORE_FILE,
        item: "ValkeyExternalValueStore::store_values_with",
        needle: KEY_BUILDER,
        count: 1,
        why: "never reads blob bytes: the write, `KEYS[2]` of `ext_store_value.lua` \
              (`SET NX`, and the verified heal's `TYPE` / `STRLEN` / `SET`)",
    },
    AllowedSite {
        file: STORE_FILE,
        item: "ValkeyExternalValueStore::load_values",
        needle: KEY_BUILDER,
        count: 1,
        why: "the store's own `GET` behind `load_values` and the probe; its callers are \
              the rows for those methods",
    },
    AllowedSite {
        file: STORE_FILE,
        item: "ValkeyExternalValueStore::read_batch_with_recovery_counter",
        needle: KEY_BUILDER,
        count: 1,
        why: "the store's own `GET` behind Recover (`read_batch`); its callers are the rows \
              for those methods",
    },
    AllowedSite {
        file: STORE_FILE,
        item: "ValkeyExternalValueStore::gc_member",
        needle: KEY_BUILDER,
        count: 1,
        why: "runs under the GC's own leader lock and never reads blob bytes: `KEYS[2]` of \
              `gc_decrement_and_collect.lua`, which deletes the blob at zero",
    },
    // ── The backup copy's reads (b.wsx) ────────────────────────────────────
    AllowedSite {
        file: EXT_BACKUP_FILE,
        item: "stored_ext_value_len",
        needle: KEY_BUILDER,
        count: 1,
        why: "never reads blob bytes: `STRLEN` only, the backup repair's size probe before it \
              reserves the value's charge",
    },
    AllowedSite {
        file: EXT_BACKUP_FILE,
        item: "read_ext_value_bytes",
        needle: KEY_BUILDER,
        count: 1,
        why: "serves no snapshot: the backup repair's ranged read (`GETRANGE`s, then a closing \
              `STRLEN`) of a value it copies to the backup store, never to a layer read or a \
              client. A miss means the GC collected the value, \
              so no live object references it and there is nothing to back up; the caller drops \
              the hash. The clean-up fence's heal (b.g65) is to reuse it",
    },
    // ── The store's read methods ───────────────────────────────────────────
    AllowedSite {
        file: STORE_FILE,
        item: "ExternalValueStore::read_batch_with_recovery_counter",
        needle: "read_batch",
        count: 1,
        why: "the trait default delegating to the same read; its callers are the rows for \
              `read_batch_with_recovery_counter`",
    },
    AllowedSite {
        file: STORE_FILE,
        item: "ValkeyExternalValueStore::read_batch",
        needle: "read_batch_with_recovery_counter",
        count: 1,
        why: "delegation with no counter; its callers are the rows for `read_batch`",
    },
    AllowedSite {
        file: STORE_FILE,
        item: "ValkeyExternalValueStore::probe_values",
        needle: "load_values",
        count: 1,
        why: "the probe's read is `load_values`, a miss answered as missing; its callers are \
              the rows for `probe_values`",
    },
    AllowedSite {
        file: STORE_FILE,
        item: "MockExternalValueStore::read_batch",
        needle: "read_batch_with_recovery_counter",
        count: 1,
        why: "the in-memory test double's delegation; it reads no Valkey key, so the Valkey \
              GC cannot collect under it",
    },
    // ── The catch-up build: the one reader of blob bytes ───────────────────
    AllowedSite {
        file: DIFF_READ_FILE,
        item: "resolve_external_values",
        needle: "probe_values",
        count: 1,
        why: "follows the miss protocol: Probe, every catch-up build's first read; a miss \
              answers `Built::Missing` and the caller re-reads under a read hold",
    },
    AllowedSite {
        file: DIFF_READ_FILE,
        item: "resolve_external_values",
        needle: "read_batch_with_recovery_counter",
        count: 1,
        why: "follows the miss protocol: Recover, built only from the `LiveHoldProof` of a \
              genuine-miss check under a live read hold (pinned by \
              `grpc::b_yuo_catch_up_miss_test`)",
    },
];

/// A script an allowlisted site hands a blob key to: the site's `item` in
/// [`STORE_FILE`], the `static` it runs the script through, the script, the
/// local the script binds the key to, its `KEYS` index, and the exact set of
/// commands the script runs on it, sorted.
struct ScriptHandedBlobKey {
    item: &'static str,
    script_static: &'static str,
    script: &'static str,
    key_var: &'static str,
    key_index: usize,
    commands: &'static [&'static str],
}

/// Every script an [`ALLOWED_SITES`] row hands a blob key to.
const SCRIPTS_HANDED_A_BLOB_KEY: [ScriptHandedBlobKey; 2] = [
    ScriptHandedBlobKey {
        item: "ValkeyExternalValueStore::store_values_with",
        script_static: "EXT_STORE_VALUE_SCRIPT",
        script: "src/storage/scripts/ext_store_value.lua",
        key_var: "data_key",
        key_index: 2,
        // No `DEL` since b.cyu: the verified heal replaces a wrong-typed blob
        // with one plain `SET`.
        commands: &["SET", "STRLEN", "TYPE"],
    },
    ScriptHandedBlobKey {
        item: "ValkeyExternalValueStore::gc_member",
        script_static: "GC_DECREMENT_AND_COLLECT_SCRIPT",
        script: "src/storage/scripts/gc_decrement_and_collect.lua",
        key_var: "data_key",
        key_index: 2,
        // `STRLEN` since b.pa1: the deleting branch answers the blob's length
        // as the bytes it freed. It returns a length, never the bytes.
        commands: &["DEL", "STRLEN"],
    },
];

/// Commands, and `SET`'s `GET` option, that return a string key's bytes. None
/// may be run on a blob key a script is handed.
const BYTE_READING_COMMANDS: [&str; 8] = [
    "DUMP", "GET", "GETDEL", "GETEX", "GETRANGE", "GETSET", "MGET", "SUBSTR",
];

/// The rule a failure restates.
const RULE: &str = "A new reader of external-value blob bytes must follow the GC miss \
                    protocol (probe -> miss -> take a `{gc}` read hold -> re-read the layer -> \
                    rebuild; SDD §11.2, §11.3): the GC may collect a released blob 5 s after \
                    its release, and at any age while the store is out of memory, so a direct \
                    read can hand a client a delta whose blob is gone. \
                    If the site follows the protocol, never reads blob bytes (a write, delete, \
                    existence or size probe), runs under the GC's own leader lock, is a store \
                    implementation's read method or its delegation to another store's (a \
                    wrapping implementation), whose own callers are rows, serves no snapshot \
                    (a miss only means the value was collected and has nothing left to serve), \
                    or is an unrelated \
                    method that only shares a scanned name (matching is by identifier), add a \
                    row for it to ALLOWED_SITES in \
                    tests/integration/storage/ext_blob_reader_guard_test.rs with that reason";

// ─────────────────────────────────────────────────────────────────────────────
// The scanners
// ─────────────────────────────────────────────────────────────────────────────

/// An item with a body: a `fn`, an `impl` block (named by its self type), a
/// `trait` or a `mod`, as the `::`-joined path of the items around it, and
/// the indices of its opening and closing braces in the flattened code.
struct ItemSpan {
    path: String,
    start: usize,
    end: usize,
}

/// The identifier starting at `text[at]`.
fn ident_at(text: &[char], at: usize) -> String {
    text[at..].iter().take_while(|c| is_ident(**c)).collect()
}

/// The identifier after `text[from..]`'s leading spaces, if one starts there.
fn next_ident(text: &[char], from: usize) -> Option<String> {
    let at = (from..text.len()).find(|&i| text[i] != ' ')?;
    (is_ident(text[at]) && !text[at].is_ascii_digit()).then(|| ident_at(text, at))
}

/// The self type an `impl` header names (`impl<T> Trait for Self<T> where …`
/// → `Self`), without generics or a path.
fn impl_label(header: &str) -> String {
    let mut rest = header.trim();
    if rest.starts_with('<') {
        let mut depth = 0i32;
        let mut end = rest.len();
        let chars: Vec<(usize, char)> = rest.char_indices().collect();
        for (k, &(at, c)) in chars.iter().enumerate() {
            match c {
                '<' => depth += 1,
                '>' if k == 0 || chars[k - 1].1 != '-' => {
                    depth -= 1;
                    if depth == 0 {
                        end = at + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        rest = rest[end..].trim_start();
    }
    let rest = rest.split(" where ").next().unwrap_or("");
    let self_type = rest.rsplit(" for ").next().unwrap_or("").trim();
    let self_type = self_type
        .trim_start_matches('&')
        .trim_start_matches("dyn ")
        .trim();
    let base = self_type.split('<').next().unwrap_or("");
    base.rsplit("::").next().unwrap_or("").trim().to_owned()
}

/// Every item with a body in `text` (a [`FlatCode`] text), in order of its
/// opening brace. A `fn` without a body (a trait method's declaration) ends at
/// its `;`; `impl` inside a function signature is `impl Trait`, not a block.
fn item_spans(text: &[char]) -> Vec<ItemSpan> {
    let mut spans: Vec<ItemSpan> = Vec::new();
    // (brace depth outside the item, index into `spans`)
    let mut open: Vec<(usize, usize)> = Vec::new();
    let mut pending: Option<String> = None;
    let mut pending_fn = false;
    // `(` / `[` depth since `pending` was set: a `;` inside `[u8; 32]` does
    // not end a signature.
    let mut signature_depth = 0i32;
    let mut depth = 0usize;
    let mut i = 0;
    while i < text.len() {
        let c = text[i];
        if is_ident(c) && (i == 0 || !is_ident(text[i - 1])) {
            let word = ident_at(text, i);
            let after = i + word.chars().count();
            match word.as_str() {
                "fn" => {
                    if let Some(name) = next_ident(text, after) {
                        pending = Some(name);
                        pending_fn = true;
                        signature_depth = 0;
                    }
                }
                "impl" if !(pending.is_some() && pending_fn) => {
                    let header: String = text[after..]
                        .iter()
                        .take_while(|c| **c != '{' && **c != ';')
                        .collect();
                    pending = Some(impl_label(&header));
                    pending_fn = false;
                    signature_depth = 0;
                }
                "trait" | "mod" if pending.is_none() => {
                    if let Some(name) = next_ident(text, after) {
                        pending = Some(name);
                        pending_fn = false;
                        signature_depth = 0;
                    }
                }
                _ => {}
            }
            i = after;
            continue;
        }
        match c {
            '(' | '[' if pending.is_some() => signature_depth += 1,
            ')' | ']' if pending.is_some() => signature_depth -= 1,
            ';' if pending.is_some() && signature_depth <= 0 => pending = None,
            '{' => {
                if let Some(name) = pending.take() {
                    let path = match open.last() {
                        Some((_, parent)) => format!("{}::{name}", spans[*parent].path),
                        None => name,
                    };
                    spans.push(ItemSpan {
                        path,
                        start: i,
                        end: text.len(),
                    });
                    open.push((depth, spans.len() - 1));
                }
                depth += 1;
            }
            '}' => {
                depth = depth.saturating_sub(1);
                if open.last().is_some_and(|(d, _)| *d == depth) {
                    let (_, at) = open.pop().expect("checked above");
                    spans[at].end = i;
                }
            }
            _ => {}
        }
        i += 1;
    }
    spans
}

/// The path of the innermost item around `at`, or `<file scope>`.
fn enclosing_item(spans: &[ItemSpan], at: usize) -> String {
    spans
        .iter()
        .filter(|s| s.start <= at && at <= s.end)
        .max_by_key(|s| s.start)
        .map(|s| s.path.clone())
        .unwrap_or_else(|| "<file scope>".to_owned())
}

/// Does `text` hold `needle` at `at`?
fn starts_with_at(text: &[char], at: usize, needle: &str) -> bool {
    let mut i = at;
    for c in needle.chars() {
        if text.get(i) != Some(&c) {
            return false;
        }
        i += 1;
    }
    true
}

/// One use of a scanned name: the name, the item it sits in, its line.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Hit {
    needle: String,
    item: String,
    line: usize,
}

/// Every use of a name in `needles` in `code`: a method call (`.name`), a path
/// (`Type::name`) or a call (`name(`), as a whole identifier, and not its
/// definition (`fn name`). A local binding or a bare word of the same name is
/// not a use.
fn uses(code: &ProductionCode, needles: &[&str]) -> Vec<Hit> {
    let flat = FlatCode::new(code);
    let text = &flat.text;
    let spans = item_spans(text);
    let mut hits = Vec::new();
    for at in 0..text.len() {
        if at > 0 && is_ident(text[at - 1]) {
            continue;
        }
        for needle in needles {
            if !starts_with_at(text, at, needle) {
                continue;
            }
            let next = text.get(at + needle.chars().count()).copied();
            if next.is_some_and(is_ident) {
                continue;
            }
            let method = at > 0 && text[at - 1] == '.';
            let path = at > 1 && text[at - 1] == ':' && text[at - 2] == ':';
            let call = next == Some('(');
            let definition = at > 2 && starts_with_at(text, at - 3, "fn ");
            if (method || path || call) && !definition {
                hits.push(Hit {
                    needle: (*needle).to_owned(),
                    item: enclosing_item(&spans, at),
                    line: flat.line_of[at],
                });
            }
        }
    }
    hits
}

/// Every production line of `source` (`code` is its production view) whose
/// code or string literals spell [`KEY_LITERAL`], with the item it sits in.
fn key_literal_hits(source: &str, code: &ProductionCode) -> Vec<Hit> {
    let flat = FlatCode::new(code);
    let spans = item_spans(&flat.text);
    needle_hits(source, KEY_LITERAL)
        .into_iter()
        .filter(|(line, _)| {
            code.lines
                .get(line - 1)
                .is_some_and(|(_, kept)| !kept.is_empty())
        })
        .map(|(line, _)| {
            // The line's last character of code: on a one-line item that is
            // its closing brace, which the item's span includes, where the
            // first character (`fn`) is still outside it.
            let end = flat.line_of.partition_point(|l| *l <= line);
            let at = (0..end)
                .rev()
                .take_while(|&i| flat.line_of[i] == line)
                .find(|&i| flat.text[i] != ' ')
                .unwrap_or(end.saturating_sub(1));
            Hit {
                needle: KEY_LITERAL.to_owned(),
                item: enclosing_item(&spans, at),
                line,
            }
        })
        .collect()
}

/// Does a return type name one of [`BYTES_RETURN_TYPES`] as a whole token
/// (`Bytes`, not `PendingBytes`)?
fn carries_bytes(returns: &str) -> bool {
    BYTES_RETURN_TYPES.iter().any(|spelling| {
        returns.match_indices(spelling).any(|(at, _)| {
            let before = returns[..at].chars().next_back();
            let after = returns[at + spelling.len()..].chars().next();
            !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
        })
    })
}

/// The methods of the `ExternalValueStore` trait in `code` (the store
/// module's production view) whose return type carries bytes
/// ([`BYTES_RETURN_TYPES`]), sorted.
fn trait_blob_read_methods(code: &ProductionCode) -> Vec<String> {
    let flat = FlatCode::new(code);
    let text = &flat.text;
    let header = (0..text.len())
        .find(|&at| starts_with_at(text, at, STORE_TRAIT_HEADER))
        .unwrap_or_else(|| panic!("{STORE_FILE} declares `{STORE_TRAIT_HEADER}`"));
    let open = (header..text.len())
        .find(|&at| text[at] == '{')
        .expect("the trait has a body");
    let mut found = BTreeSet::new();
    let mut depth = 0i32;
    let mut at = open + 1;
    while at < text.len() {
        match text[at] {
            '{' => depth += 1,
            '}' if depth == 0 => break,
            '}' => depth -= 1,
            _ if depth == 0
                && starts_with_at(text, at, "fn ")
                && (at == 0 || !is_ident(text[at - 1])) =>
            {
                let name = next_ident(text, at + 3).expect("a method name");
                let mut nesting = 0i32;
                let mut end = at;
                while end < text.len() {
                    match text[end] {
                        '(' | '[' => nesting += 1,
                        ')' | ']' => nesting -= 1,
                        ';' | '{' if nesting == 0 => break,
                        _ => {}
                    }
                    end += 1;
                }
                let signature: String = text[at..end].iter().collect();
                let returns = signature.split_once("->").map_or("", |(_, r)| r);
                if carries_bytes(returns) {
                    found.insert(name);
                }
                at = end;
                continue;
            }
            _ => {}
        }
        at += 1;
    }
    found.into_iter().collect()
}

/// One `redis.call(…)` / `redis.pcall(…)` naming a script's blob key.
#[derive(Debug, PartialEq, Eq)]
struct LuaCall {
    /// Whether the first argument, the command, is a string literal. A
    /// command held in a variable (`redis.call(op, data_key)`) cannot be read
    /// by this scan.
    literal_command: bool,
    /// The upper-cased string literals among the arguments, in order: the
    /// command first when `literal_command`.
    strings: Vec<String>,
}

/// Every `redis.call(…)` / `redis.pcall(…)` in `code` (from [`lua_code`])
/// whose arguments name `key_var` as a whole identifier.
fn lua_calls_on(code: &[String], key_var: &str) -> Vec<LuaCall> {
    let joined: Vec<char> = code.join(" ").chars().collect();
    let mut calls = Vec::new();
    for opener in ["redis.call(", "redis.pcall("] {
        for at in 0..joined.len() {
            if !starts_with_at(&joined, at, opener) {
                continue;
            }
            let start = at + opener.len();
            let mut depth = 1i32;
            let mut quote: Option<char> = None;
            let mut end = start;
            while end < joined.len() {
                let c = joined[end];
                match quote {
                    Some(q) if c == q => quote = None,
                    Some(_) => {}
                    None => match c {
                        '"' | '\'' => quote = Some(c),
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    },
                }
                end += 1;
            }
            let args = &joined[start..end];
            let names_key = (0..args.len()).any(|i| {
                starts_with_at(args, i, key_var)
                    && (i == 0 || !is_ident(args[i - 1]))
                    && !args.get(i + key_var.len()).copied().is_some_and(is_ident)
            });
            if !names_key {
                continue;
            }
            let args: String = args.iter().collect();
            let literal_command = args.trim_start().starts_with(['"', '\'']);
            let strings: Vec<String> = args
                .split(['"', '\''])
                .skip(1)
                .step_by(2)
                .map(str::to_uppercase)
                .collect();
            calls.push(LuaCall {
                literal_command,
                strings,
            });
        }
    }
    calls
}

/// How many times `code` (from [`lua_code`]) names `KEYS[index]`, spaces
/// inside the brackets ignored.
fn keys_index_uses(code: &[String], index: usize) -> usize {
    let needle = format!("KEYS[{index}]");
    code.iter()
        .map(|line| {
            line.chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
        })
        .map(|line| {
            line.match_indices(&needle)
                .filter(|(at, _)| !line[..*at].chars().next_back().is_some_and(is_ident))
                .count()
        })
        .sum()
}

/// Every `.lua` file under `dir`, relative to `root`, sorted.
fn lua_sources_under(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {} must succeed: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry must be readable").path();
        if path.is_dir() {
            lua_sources_under(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "lua") {
            out.push(
                path.strip_prefix(root)
                    .expect("scanned paths are under the manifest dir")
                    .to_path_buf(),
            );
        }
    }
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ─────────────────────────────────────────────────────────────────────────────
// The guards
// ─────────────────────────────────────────────────────────────────────────────

/// Every production site under `src/` that can reach external-value blob bytes
/// is a row of [`ALLOWED_SITES`], at its exact count, and every row still
/// names a site; [`KEY_LITERAL`] is spelled only by the key builder; and
/// [`BLOB_READ_METHODS`] is exactly the trait's methods that return bytes.
#[test]
fn every_external_value_blob_read_site_under_src_is_allowlisted() {
    let tree = ProductionTree::load_release();
    tree.assert_visits(&[KEY_BUILDER_FILE, STORE_FILE, DIFF_READ_FILE]);

    // The read methods are the trait's own.
    let derived = trait_blob_read_methods(&tree.file(STORE_FILE).code);
    assert_eq!(
        derived, BLOB_READ_METHODS,
        "the `ExternalValueStore` methods returning blob bytes (a return type with {:?}) must \
         be exactly BLOB_READ_METHODS, so this guard scans every caller of each; add a new \
         one there (and its callers to ALLOWED_SITES). {RULE}",
        BYTES_RETURN_TYPES
    );

    // Every site, grouped by (file, item, needle).
    let mut needles = vec![KEY_BUILDER];
    needles.extend(BLOB_READ_METHODS);
    let mut found: BTreeMap<(String, String, String), Vec<usize>> = BTreeMap::new();
    let mut literals = Vec::new();
    for file in &tree.files {
        let path = file.path.to_string_lossy().into_owned();
        for hit in uses(&file.code, &needles) {
            found
                .entry((path.clone(), hit.item, hit.needle))
                .or_default()
                .push(hit.line);
        }
        for hit in key_literal_hits(&file.source, &file.code) {
            literals.push((path.clone(), hit.item, hit.line));
        }
    }

    let mut problems = Vec::new();
    let mut listed: BTreeSet<(String, String, String)> = BTreeSet::new();
    for row in &ALLOWED_SITES {
        let key = (
            row.file.to_owned(),
            row.item.to_owned(),
            row.needle.to_owned(),
        );
        assert!(
            listed.insert(key.clone()),
            "ALLOWED_SITES lists {key:?} twice; merge the rows"
        );
        let lines = found.get(&key).cloned().unwrap_or_default();
        if lines.len() < row.count {
            problems.push(format!(
                "stale allowlist row: {} `{}` uses `{}` {} time(s) (lines {lines:?}) but its \
                 row allows {}. Remove or update the row, so the list stays the exact set of \
                 blob-read sites (reason on the row: {})",
                row.file,
                row.item,
                row.needle,
                lines.len(),
                row.count,
                row.why
            ));
        } else if lines.len() > row.count {
            problems.push(format!(
                "new blob-read site: {} `{}` uses `{}` {} time(s) (lines {lines:?}) but its \
                 row allows {}. {RULE}",
                row.file,
                row.item,
                row.needle,
                lines.len(),
                row.count
            ));
        }
    }
    for ((file, item, needle), lines) in &found {
        if !listed.contains(&(file.clone(), item.clone(), needle.clone())) {
            problems.push(format!(
                "new blob-read site: {file} `{item}` uses `{needle}` at line(s) {lines:?}, \
                 and ALLOWED_SITES has no row for it. {RULE}"
            ));
        }
    }

    // The key is spelled only where it is built.
    let builder_spellings = literals
        .iter()
        .filter(|(file, item, _)| file == KEY_BUILDER_FILE && item == KEY_BUILDER_ITEM)
        .count();
    if builder_spellings != 1 {
        problems.push(format!(
            "non-vacuity: `{KEY_BUILDER_ITEM}` in {KEY_BUILDER_FILE} spells `{KEY_LITERAL}` \
             once; found {builder_spellings} (the literal scan, or the builder, moved)"
        ));
    }
    for (file, item, line) in &literals {
        if !(file == KEY_BUILDER_FILE && item == KEY_BUILDER_ITEM) {
            problems.push(format!(
                "new blob-read site: {file}:{line} (`{item}`) spells the blob key \
                 `{KEY_LITERAL}` by hand instead of calling `{KEY_BUILDER_ITEM}`. Build the key \
                 with the builder, so this guard sees the site. {RULE}"
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "external-value blob readers under src/ (b.bnn):\n- {}",
        problems.join("\n- ")
    );
}

/// The scripts an allowlisted site hands a blob key to read none of its bytes,
/// and no script builds a blob key itself (module docs, *Lua scripts*).
///
/// For each [`SCRIPTS_HANDED_A_BLOB_KEY`] row: the site's item runs the
/// row's `static`, which is the row's script; the script binds the key from
/// `KEYS[key_index]`, and names `KEYS[key_index]` nowhere else, so every use
/// of the key goes through that variable; every call naming the variable names
/// its command with a string literal; and the commands it runs on the key are
/// exactly the row's, with no [`BYTE_READING_COMMANDS`] word among any such
/// call's string arguments (which also refuses `SET … GET`).
#[test]
fn every_script_handed_a_blob_key_never_reads_its_bytes() {
    let root = crate_root();
    let tree = ProductionTree::load_release();
    let store = tree.file(STORE_FILE);
    let flat = FlatCode::new(&store.code);
    let spans = item_spans(&flat.text);

    let mut scripts = Vec::new();
    lua_sources_under(&root, &root.join("src"), &mut scripts);
    scripts.sort();
    for row in &SCRIPTS_HANDED_A_BLOB_KEY {
        assert!(
            scripts.iter().any(|s| s == Path::new(row.script)),
            "non-vacuity: the Lua walk reaches {}; it read {} scripts",
            row.script,
            scripts.len()
        );
    }

    let mut problems = Vec::new();
    for script in &scripts {
        let source = std::fs::read_to_string(root.join(script))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", script.display()));
        for (index, line) in lua_code(&source).iter().enumerate() {
            if line.contains(KEY_LITERAL) {
                problems.push(format!(
                    "{}:{} spells the blob key `{KEY_LITERAL}` in code: a script must take a \
                     blob key from `KEYS`, handed over by a Rust site this guard lists. {RULE}",
                    script.display(),
                    index + 1
                ));
            }
        }
    }

    for row in &SCRIPTS_HANDED_A_BLOB_KEY {
        assert!(
            row.commands
                .iter()
                .all(|c| !BYTE_READING_COMMANDS.contains(c)),
            "SCRIPTS_HANDED_A_BLOB_KEY's {} row allows a byte-reading command: {:?}",
            row.script,
            row.commands
        );
        assert!(
            ALLOWED_SITES
                .iter()
                .any(|s| s.file == STORE_FILE && s.item == row.item && s.needle == KEY_BUILDER),
            "{} is an ALLOWED_SITES `{KEY_BUILDER}` row",
            row.item
        );

        // The site runs the row's static, and the static is the row's script.
        let span = spans
            .iter()
            .find(|s| s.path == row.item)
            .unwrap_or_else(|| panic!("{STORE_FILE} defines `{}`", row.item));
        let body: Vec<char> = flat.text[span.start..=span.end].to_vec();
        let runs_static = (0..body.len()).any(|i| {
            starts_with_at(&body, i, row.script_static)
                && (i == 0 || !is_ident(body[i - 1]))
                && !body
                    .get(i + row.script_static.len())
                    .copied()
                    .is_some_and(is_ident)
        });
        if !runs_static {
            problems.push(format!(
                "`{}` no longer runs `{}`: re-check which script it hands the blob key to, \
                 and update SCRIPTS_HANDED_A_BLOB_KEY",
                row.item, row.script_static
            ));
        }
        let basename = Path::new(row.script)
            .file_name()
            .and_then(|n| n.to_str())
            .expect("a script file name");
        let declaration = format!("static {}:", row.script_static);
        let statement = store
            .source
            .split_once(&declaration)
            .and_then(|(_, rest)| rest.split_once(';'))
            .map(|(statement, _)| statement)
            .unwrap_or_else(|| panic!("{STORE_FILE} declares `{declaration}`"));
        if !statement.contains(&format!("include_str!(\"scripts/{basename}\")")) {
            problems.push(format!(
                "`{}` is no longer `{}`: update SCRIPTS_HANDED_A_BLOB_KEY",
                row.script_static, row.script
            ));
        }

        // The script binds the key and runs only the listed commands on it.
        let source = std::fs::read_to_string(root.join(row.script))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", row.script));
        let code = lua_code(&source);
        let binding = format!("local {} = KEYS[{}]", row.key_var, row.key_index);
        let bindings = code
            .iter()
            .filter(|line| line.split_whitespace().collect::<Vec<_>>().join(" ") == binding)
            .count();
        if bindings != 1 {
            problems.push(format!(
                "{} binds `{binding}` {bindings} time(s), not once: the row's key variable or \
                 index moved; update SCRIPTS_HANDED_A_BLOB_KEY",
                row.script
            ));
        }
        let index_uses = keys_index_uses(&code, row.key_index);
        if index_uses != 1 {
            problems.push(format!(
                "{} names `KEYS[{}]` {index_uses} time(s); only its binding to `{}` may, so \
                 every use of the blob key goes through the variable this check reads. {RULE}",
                row.script, row.key_index, row.key_var
            ));
        }
        let calls = lua_calls_on(&code, row.key_var);
        for call in calls.iter().filter(|call| !call.literal_command) {
            problems.push(format!(
                "{} runs a command on its blob key `{}` whose name is not a string literal \
                 ({:?}): this check cannot tell whether it returns the bytes. Name the command \
                 literally. {RULE}",
                row.script, row.key_var, call.strings
            ));
        }
        let commands: BTreeSet<String> = calls
            .iter()
            .filter(|call| call.literal_command)
            .filter_map(|call| call.strings.first().cloned())
            .collect();
        let expected: BTreeSet<String> = row.commands.iter().map(|c| (*c).to_owned()).collect();
        if commands != expected {
            problems.push(format!(
                "{} runs {commands:?} on its blob key `{}`; its row lists {expected:?}. A \
                 command that returns the bytes makes the script a blob reader. {RULE}",
                row.script, row.key_var
            ));
        }
        for call in &calls {
            if let Some(word) = call
                .strings
                .iter()
                .find(|s| BYTE_READING_COMMANDS.contains(&s.as_str()))
            {
                problems.push(format!(
                    "{} sends `{word}` with its blob key `{}` ({:?}): that returns the blob's \
                     bytes. {RULE}",
                    row.script, row.key_var, call.strings
                ));
            }
        }
    }

    assert!(
        problems.is_empty(),
        "scripts handed an external-value blob key (b.bnn):\n- {}",
        problems.join("\n- ")
    );
}

/// The scanners on a fixture: what is a use and where it is attributed, the
/// trait's byte-returning methods and the return types that carry bytes, the
/// literal scan, and the Lua command extraction and `KEYS` index count.
#[test]
fn the_blob_reader_scanners_attribute_uses_and_skip_definitions_tests_comments_and_strings() {
    let rust = concat!(
        "pub struct KeySchema;\n",                                          // 1
        "impl KeySchema {\n",                                               // 2
        "    pub fn ext_data(&self, h: &str) -> String {\n",                // 3
        "        format!(\"{}ext:data:{{{h}}}\", self.prefix)\n",           // 4
        "    }\n",                                                          // 5
        "}\n",                                                              // 6
        "fn direct(keys: &KeySchema) {\n",                                  // 7
        "    let k = keys.ext_data(\"h\");\n",                              // 8
        "    let ext_data = 1;\n",                                          // 9
        "    let _ = (k, ext_data);\n",                                     // 10
        "}\n",                                                              // 11
        "impl<S: Send> ExternalValueStore for Store<S> where S: Sync {\n",  // 12
        "    async fn read_batch(&self, f: impl Fn() -> u8) -> X {\n",      // 13
        "        self.read_batch_with_recovery_counter(h, None).await\n",   // 14
        "    }\n",                                                          // 15
        "}\n",                                                              // 16
        "fn path_call(s: &dyn ExternalValueStore) {\n",                     // 17
        "    ExternalValueStore::probe_values(s, &[]);\n",                  // 18
        "    s\n",                                                          // 19
        "        .load_values(&[]);\n",                                     // 20
        "    let f = |k: &KeySchema| { KeySchema::ext_data(k, \"h\") };\n", // 21
        "}\n",                                                              // 22
        "pub trait ExternalValueStore: Send {\n",                           // 23
        "    async fn probe_values(&self, h: &[[u8; 32]]) -> Result<ExtValueProbe, E>;\n", // 24
        "    async fn load_values(&self) -> Result<Vec<([u8; 32], Vec<u8>)>, E>;\n", // 25
        "    async fn store_values(&self, v: &[(&[u8; 32], &[u8])]) -> Result<(), E>;\n", // 26
        "    async fn read_batch_with_recovery_counter(&self) -> Result<Vec<u8>, E> {\n", // 27
        "        self.read_batch(h).await\n",                               // 28
        "    }\n",                                                          // 29
        "}\n",                                                              // 30
        "#[cfg(test)]\n",                                                   // 31
        "mod tests {\n",                                                    // 32
        "    fn t(keys: &KeySchema) { keys.ext_data(\"ext:data\"); }\n",    // 33
        "}\n",                                                              // 34
        "// keys.ext_data(h) and ext:data in a comment\n",                  // 35
        "fn in_string() { let s = \"keys.ext_data(h)\"; let t = \"le:ext:data:x\"; }\n", // 36
    );
    let code = production_code(rust);
    let mut needles = vec![KEY_BUILDER];
    needles.extend(BLOB_READ_METHODS);
    let mut hits = uses(&code, &needles);
    hits.sort();
    let hit = |needle: &str, item: &str, line: usize| Hit {
        needle: needle.to_owned(),
        item: item.to_owned(),
        line,
    };
    let mut expected = vec![
        hit("ext_data", "direct", 8),
        hit("read_batch_with_recovery_counter", "Store::read_batch", 14),
        hit("probe_values", "path_call", 18),
        hit("load_values", "path_call", 20),
        hit("ext_data", "path_call", 21),
        hit(
            "read_batch",
            "ExternalValueStore::read_batch_with_recovery_counter",
            28,
        ),
    ];
    expected.sort();
    assert_eq!(
        hits, expected,
        "a method call, a path and a call each count once, in the item around them; a \
         definition, a local binding, a test module, a comment and a string do not"
    );

    assert_eq!(
        trait_blob_read_methods(&code),
        [
            "load_values",
            "probe_values",
            "read_batch_with_recovery_counter"
        ],
        "the trait's methods returning `Vec<u8>` or `ExtValueProbe`, declared or with a body"
    );
    for returns in [
        " Result<Option<bytes::Bytes>, E>",
        " Result<Arc<[u8]>, E>",
        " Box<[u8]>",
        " Result<HashMap<[u8; 32], Vec<u8>>, E>",
    ] {
        assert!(carries_bytes(returns), "`{returns}` carries bytes");
    }
    for returns in [
        " Result<PendingBytes, E>",
        " Result<[u8; 32], E>",
        " Result<(), E>",
        " ExtValueVerify<'_>",
    ] {
        assert!(!carries_bytes(returns), "`{returns}` carries no blob bytes");
    }

    let literals: Vec<(String, usize)> = key_literal_hits(rust, &code)
        .into_iter()
        .map(|h| (h.item, h.line))
        .collect();
    assert_eq!(
        literals,
        [
            ("KeySchema::ext_data".to_owned(), 4),
            ("in_string".to_owned(), 36)
        ],
        "the literal scan reads code and strings, not comments or test modules"
    );

    let lua = concat!(
        "-- redis.call(\"GET\", data_key) in a comment\n",
        "local data_key = KEYS[2]\n",
        "local s = \"--not a comment\" redis.call(\"TYPE\", data_key)\n",
        "redis.call(\"SET\", data_key, ARGV[1], \"NX\")\n",
        "if redis.pcall('strlen', data_key) > 0 then redis.call(\"DEL\", meta_key, data_key) end\n",
        "redis.call(\"GET\", other_data_key)\n",
        "redis.call(\"SET\",\n  data_key, ARGV[1], \"GET\")\n",
        "redis.call(op, data_key)\n",
        "-- KEYS[2] in a comment\n",
        "redis.call(\"GET\", KEYS[2])\n",
        "local spaced = KEYS[ 2 ]\n",
        "local other = KEYS[20]\n",
    );
    let code = lua_code(lua);
    assert_eq!(code[1], "local data_key = KEYS[2]");
    let calls = lua_calls_on(&code, "data_key");
    let commands: BTreeSet<&str> = calls
        .iter()
        .filter(|c| c.literal_command)
        .filter_map(|c| c.strings.first().map(String::as_str))
        .collect();
    assert_eq!(
        commands,
        BTreeSet::from(["DEL", "SET", "STRLEN", "TYPE"]),
        "every call naming the key, `redis.pcall` and a call split across lines included; \
         not a comment, and not another key whose name contains it"
    );
    assert!(
        calls
            .iter()
            .any(|c| c.literal_command && c.strings == ["SET", "GET"].map(str::to_owned).to_vec()),
        "a call's string arguments are all read, so `SET … GET` is seen: {calls:?}"
    );
    assert_eq!(
        calls
            .iter()
            .filter(|c| !c.literal_command)
            .collect::<Vec<_>>(),
        [&LuaCall {
            literal_command: false,
            strings: Vec::new(),
        }],
        "`redis.call(op, data_key)` is seen as a call whose command is not a literal"
    );
    assert_eq!(
        keys_index_uses(&code, 2),
        3,
        "`KEYS[2]` is counted at the binding, inside a call and with spaces in the brackets; \
         not in a comment, and not as `KEYS[20]`"
    );
}
