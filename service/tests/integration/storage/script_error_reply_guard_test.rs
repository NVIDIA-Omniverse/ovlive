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

//! b.myh — a source guard over the error text every service Lua script can
//! return (SDD §15.3, vendored fred change 10).
//!
//! fred takes an error reply for a cluster `MOVED` / `ASK` redirect when its
//! text starts `MOVED` or `ASK` (case-sensitive) and has exactly three
//! space-separated tokens, and change 8 then follows an `ASK` by connecting,
//! with the store's credentials, to the host the text names. Change 10 keeps a
//! data reply from steering that connection; an error reply still can. The
//! service's scripts are the one place the service writes error text of its
//! own onto a reply, so the rule this guard holds them to is: **a script's
//! error text never begins with data the script read or was passed.** A
//! `redis.error_reply(ARGV[1])`, or a relay of a value read from a key, would
//! let a stored or client-supplied `ASK <slot> <host:port>` reopen the
//! credentialed connect.
//!
//! **What it checks.** Every `.lua` file under `src/storage/scripts`, with
//! comments removed (`lua_source_scan::lua_code`), for every place error text
//! is made:
//!
//! - `error_reply` named anywhere in code (`redis.error_reply`,
//!   `server.error_reply`, any spacing around the `.`, an alias such as
//!   `local f = redis.error_reply`): its argument. A string literal reading
//!   `error_reply` (`redis['error_reply'](…)`) fails whatever its argument;
//! - a call of Lua's `error`: its argument (a table raised with an `err`
//!   field becomes the reply's error text);
//! - a write of an `err` field (`{err = …}`, `{…, err = …}`, `{…; err = …}`,
//!   `t.err = …`, `t["err"] = …`): the value written, since a returned or
//!   raised table with an `err` field is an error reply.
//!
//! Each must begin with a fixed lead: a non-empty string literal whose first
//! token, after one leading `-`, does not start with `ASK` or `MOVED`, and
//! which either is the whole text or contains a space (so the text's first
//! token is fixed by the script, whatever data follows). A literal followed by
//! `..`, on its line or the next, is not the whole text. This is stricter than
//! "starts with a literal": `"" .. ARGV[1]`, `"AS" .. ARGV[1]` and
//! `'-ASK ' .. ARGV[1]` fail.
//!
//! Anything else fails unless it is one of [`RELAYS`]: a named script, form and
//! expression with the reason it can never begin with data. Each relay
//! carries [`Pin`]s over the assignments its reason depends on, as `writes`
//! reads them: a plain or dotted name on the left of `=` (`xadd_err = …`,
//! `REFUSAL.msg = …`), alone or in a multiple assignment. A pin holds the
//! relayed name assigned exactly once and as the reason says, or every such
//! assignment fixed-led (`REFUSAL.msg`) or taken from `redis.pcall`
//! (`xadd_result`), and the script's one `return nil, …` (a `count_arg`
//! helper's refusal) fixed-led, or that a given line of the script is still
//! present (the `REFUSAL` catch line). Each relay must also match exactly one site
//! in its script, so a second site making the same text from the same name
//! fails, as does a relay no longer used
//! ([`every_relay_is_still_used_and_its_pins_hold`]): the list cannot rot.
//! Any long bracket (`[[ … ]]`, `--[[ … ]]`) fails, since `lua_code` does not
//! model one.
//!
//! **Lua built in Rust.** [`no_rust_source_builds_lua_error_text`] reads every
//! string literal under `src/` (comments, char literals and raw strings
//! handled) that contains `redis.call(`, `redis.pcall(`, `server.call(` or
//! `server.pcall(` as Lua and fails on any error-text site in it; there is no
//! relay list for Rust-built Lua, so such a site must move to a script or gain
//! one. It also fails on `error_reply` spelled anywhere in Rust outside a `//`
//! comment, which catches Lua that calls no command.
//!
//! **What it cannot see.**
//!
//! - A `redis.pcall` reply returned as it came (`return r`): the server's own
//!   error text, which begins with the server's error code. The guard sees
//!   only text the script builds, names or raises (raising one with
//!   `error(r)` is a site, and needs a relay entry).
//! - Error text reaching a reply some other way than the three forms above,
//!   for example a table given an `err` key by a computed name (`t[k] = …`).
//!   No script does this; review covers it.
//! - A second binding of a pinned name that is not an assignment: a function
//!   parameter or a `for` variable of the same name in the same script. A
//!   [`Pin`] counts assignments, not other bindings. (A relay site added
//!   inside such a function still fails, as a second site of the relay; one
//!   moved there need not.)
//! - A write of a pinned name that `writes` does not read as one: an indexed
//!   write (`REFUSAL["msg"] = …`), a write through an alias
//!   (`local R = REFUSAL; R.msg = …`), or `rawset(REFUSAL, "msg", …)`.
//! - A second value returned by `count_arg` other than through
//!   `return nil, …` (`return v, ARGV[i]`): [`Pin::OneReturnNil`] reads only
//!   that form.
//! - Rust-built Lua that contains none of the call functions and does not
//!   spell `error_reply`, or a Lua text split across several Rust literals
//!   (`concat!`, `format!` pieces), which is read one literal at a time.
//!
//! **Shown to fail.** With `src/storage/scripts/snapshot_intent_renew.lua`'s
//! `redis.error_reply("ERR snapshot_intent_renew: empty token")` changed in
//! place to `redis.error_reply(ARGV[1])`, the main guard fails naming that
//! script, its line and `ARGV[1]`. With
//! `local function f(REFUSAL) return redis.error_reply(REFUSAL.msg) end` added
//! to `fanout_window_read.lua`, the relay test fails on its two sites; with
//! [`COUNTERS_LUA`]'s text mangled, the Rust scan fails its precondition. Each
//! file was then restored byte-identical. The scanner's own cases are pinned
//! by [`the_scanner_reads_every_form_of_error_text`], the pins' by
//! [`a_relay_pin_fails_on_a_second_data_led_write`], the one-site count's by
//! [`a_relay_matching_a_second_site_fails`], and the Rust literal reader's by
//! [`the_rust_literal_reader_finds_lua_in_every_string_form`].
//!
//! Static: reads source files, needs no Valkey and no feature. Pinned in
//! `scripts/guard-tests.sh` `MODULES`.

use std::path::{Path, PathBuf};

use crate::storage::lua_source_scan::{long_bracket_lines, lua_code, script_files, SCRIPTS_DIR};

// ═════════════════════════════════════════════════════════════════════════════
// The scanner
// ═════════════════════════════════════════════════════════════════════════════

/// How a script makes error text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sink {
    /// `error_reply` (`redis.error_reply` / `server.error_reply`).
    ErrorReply,
    /// Lua's `error(…)`.
    Error,
    /// A write of an `err` field.
    ErrField,
}

/// How a piece of error text begins.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Lead {
    /// A string literal (its raw contents); `whole` when it is all of the text.
    Literal { lead: String, whole: bool },
    /// Anything else: the expression, whitespace collapsed.
    Expr(String),
    /// `error_reply` named but not called with an argument list or a string
    /// (taken as a value, called with a table, or spelled as a string).
    NotCalled,
}

/// One place a script makes error text.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Site {
    sink: Sink,
    /// 1-based line in the script.
    line: usize,
    lead: Lead,
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The first index at or after `at` that is not whitespace.
fn skip_ws(b: &[u8], mut at: usize) -> usize {
    while b.get(at).is_some_and(|c| c.is_ascii_whitespace()) {
        at += 1;
    }
    at
}

/// The last character before `at` that is not whitespace.
fn prev_non_ws(b: &[u8], at: usize) -> Option<u8> {
    b[..at]
        .iter()
        .rev()
        .copied()
        .find(|c| !c.is_ascii_whitespace())
}

/// The 1-based line of byte `at` in `code`.
fn line_of(code: &str, at: usize) -> usize {
    code[..at].matches('\n').count() + 1
}

/// The string literal opening at `at`: the index after its closing quote,
/// and its raw contents. An unterminated one ends at the line's end.
fn string_at(code: &str, at: usize) -> (usize, &str) {
    let b = code.as_bytes();
    let quote = b[at];
    let mut i = at + 1;
    while i < b.len() && b[i] != quote && b[i] != b'\n' {
        i += if b[i] == b'\\' { 2 } else { 1 };
    }
    let end = i.min(b.len());
    (
        (end + 1).min(b.len()),
        &code[at + 1..end.min(b.len()).max(at + 1)],
    )
}

/// Whether `at` holds an assignment `=` (not `==`).
fn is_assign(b: &[u8], at: usize) -> bool {
    b.get(at) == Some(&b'=') && b.get(at + 1) != Some(&b'=')
}

/// The expression starting at `at`, up to the first `,` or closer at depth 0
/// (or, with `stop_at_newline`, a newline at depth 0), whitespace collapsed.
fn expression(code: &str, at: usize, stop_at_newline: bool) -> String {
    let b = code.as_bytes();
    let mut depth = 0i32;
    let mut i = at;
    while i < b.len() {
        match b[i] {
            b'"' | b'\'' => {
                i = string_at(code, i).0;
                continue;
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' if depth == 0 => break,
            b')' | b']' | b'}' => depth -= 1,
            b',' | b';' if depth == 0 => break,
            b'\n' if depth == 0 && stop_at_newline => break,
            _ => {}
        }
        i += 1;
    }
    code[at..i].split_whitespace().collect::<Vec<_>>().join(" ")
}

/// How the text starting at `at` begins. A literal is the whole text when a
/// `)`, `,`, `}`, `;` or the end follows it (or, with `stop_at_newline`, as
/// for a field's value, the line's end), unless the next line goes on with
/// `..`.
fn lead_at(code: &str, at: usize, stop_at_newline: bool) -> Lead {
    let b = code.as_bytes();
    let at = skip_ws(b, at);
    match b.get(at).copied() {
        Some(b'"' | b'\'') => {
            let (end, lead) = string_at(code, at);
            let mut next = end;
            while b.get(next).is_some_and(|c| *c == b' ' || *c == b'\t') {
                next += 1;
            }
            let whole = match b.get(next).copied() {
                None => true,
                Some(b'\n') => {
                    let after = skip_ws(b, next);
                    !code[after..].starts_with("..")
                        && (stop_at_newline
                            || matches!(
                                b.get(after).copied(),
                                None | Some(b')' | b',' | b'}' | b';')
                            ))
                }
                Some(c) => matches!(c, b')' | b',' | b'}' | b';'),
            };
            Lead::Literal {
                lead: lead.to_owned(),
                whole,
            }
        }
        _ => Lead::Expr(expression(code, at, stop_at_newline)),
    }
}

/// Every place `code` (a script's code, comments removed, lines joined by
/// `\n`) makes error text, in order.
fn sites(code: &str) -> Vec<Site> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let mut braces = 0i32;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'"' || c == b'\'' {
            let (end, contents) = string_at(code, i);
            // `redis['error_reply'](…)`: the function named through a string.
            if contents == "error_reply" {
                out.push(Site {
                    sink: Sink::ErrorReply,
                    line: line_of(code, i),
                    lead: Lead::NotCalled,
                });
            }
            // `t["err"] = …`
            if contents == "err" && prev_non_ws(b, i) == Some(b'[') {
                let close = skip_ws(b, end);
                if b.get(close) == Some(&b']') {
                    let eq = skip_ws(b, close + 1);
                    if is_assign(b, eq) {
                        out.push(Site {
                            sink: Sink::ErrField,
                            line: line_of(code, i),
                            lead: lead_at(code, eq + 1, true),
                        });
                    }
                }
            }
            i = end;
            continue;
        }
        match c {
            b'{' => braces += 1,
            b'}' => braces -= 1,
            _ => {}
        }
        if is_ident(c) && (i == 0 || !is_ident(b[i - 1])) {
            let start = i;
            while i < b.len() && is_ident(b[i]) {
                i += 1;
            }
            let after = skip_ws(b, i);
            let prev = prev_non_ws(b, start);
            match &code[start..i] {
                "error_reply" => {
                    let lead = match b.get(after).copied() {
                        Some(b'(') => lead_at(code, after + 1, false),
                        Some(b'"' | b'\'') => {
                            let (_, lead) = string_at(code, after);
                            Lead::Literal {
                                lead: lead.to_owned(),
                                whole: true,
                            }
                        }
                        _ => Lead::NotCalled,
                    };
                    out.push(Site {
                        sink: Sink::ErrorReply,
                        line: line_of(code, start),
                        lead,
                    });
                }
                "error" if !matches!(prev, Some(b'.' | b':')) => match b.get(after).copied() {
                    Some(b'(') => out.push(Site {
                        sink: Sink::Error,
                        line: line_of(code, start),
                        lead: lead_at(code, after + 1, false),
                    }),
                    Some(b'"' | b'\'') => {
                        let (_, lead) = string_at(code, after);
                        out.push(Site {
                            sink: Sink::Error,
                            line: line_of(code, start),
                            lead: Lead::Literal {
                                lead: lead.to_owned(),
                                whole: true,
                            },
                        });
                    }
                    _ => {}
                },
                "err" => {
                    let field = prev == Some(b'.')
                        || (braces > 0 && matches!(prev, Some(b'{' | b',' | b';')));
                    if field && is_assign(b, after) {
                        out.push(Site {
                            sink: Sink::ErrField,
                            line: line_of(code, start),
                            lead: lead_at(code, after + 1, true),
                        });
                    }
                }
                _ => {}
            }
            continue;
        }
        i += 1;
    }
    out
}

/// Whether `lead` fixes the text's first token to one that is not a redirect:
/// a non-empty literal whose first token, after one leading `-`, does not
/// start with `ASK` or `MOVED`, and which is the whole text or contains a
/// space.
fn fixed_lead(lead: &Lead) -> bool {
    match lead {
        Lead::Literal { lead, whole } => {
            let first = lead.split(' ').next().unwrap_or("");
            let first = first.strip_prefix('-').unwrap_or(first);
            !lead.is_empty()
                && !first.starts_with("ASK")
                && !first.starts_with("MOVED")
                && (*whole || lead.contains(' '))
        }
        Lead::Expr(_) | Lead::NotCalled => false,
    }
}

/// `source` (a script) reduced to code, lines joined.
fn code_of(source: &str) -> String {
    lua_code(source).join("\n")
}

// ═════════════════════════════════════════════════════════════════════════════
// Writes of a name
// ═════════════════════════════════════════════════════════════════════════════

/// One assignment to a name.
#[derive(Clone, Debug)]
struct Write {
    /// 1-based line.
    line: usize,
    /// The whole line it is on, whitespace collapsed.
    stmt: String,
    /// How the assigned value begins.
    value: Lead,
    /// The assigned value, whitespace collapsed.
    rhs: String,
}

/// Every assignment to `name` (a plain or dotted name: `xadd_err`,
/// `REFUSAL.msg`) in `code`, alone or in a multiple assignment
/// (`local a, name = …`). A table field of the same name (`{name = …}`) and a
/// field of a longer name (`name.x = …`, `t.name = …`) are not writes of it.
fn writes(code: &str, name: &str) -> Vec<Write> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let mut braces = 0i32;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'"' || c == b'\'' {
            i = string_at(code, i).0;
            continue;
        }
        match c {
            b'{' => braces += 1,
            b'}' => braces -= 1,
            _ => {}
        }
        if !(is_ident(c) && (i == 0 || !is_ident(b[i - 1]))) {
            i += 1;
            continue;
        }
        let start = i;
        let end = start + name.len();
        let named = code[start..].starts_with(name)
            && !b
                .get(end)
                .is_some_and(|n| is_ident(*n) || matches!(n, b'.' | b'[' | b':'));
        let prev = prev_non_ws(b, start);
        let field_of_other = matches!(prev, Some(b'.' | b':'));
        let table_key = braces > 0 && matches!(prev, Some(b'{' | b',' | b';'));
        if named && !field_of_other && !table_key {
            let mut j = skip_ws(b, end);
            while b.get(j) == Some(&b',') {
                j = skip_ws(b, j + 1);
                while b.get(j).is_some_and(|n| is_ident(*n) || *n == b'.') {
                    j += 1;
                }
                j = skip_ws(b, j);
            }
            if is_assign(b, j) {
                let line_start = code[..start].rfind('\n').map_or(0, |n| n + 1);
                let line_end = code[start..].find('\n').map_or(code.len(), |n| start + n);
                out.push(Write {
                    line: line_of(code, start),
                    stmt: code[line_start..line_end]
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" "),
                    value: lead_at(code, j + 1, true),
                    rhs: expression(code, skip_ws(b, j + 1), true),
                });
            }
        }
        while i < b.len() && is_ident(b[i]) {
            i += 1;
        }
    }
    out
}

/// Every `return nil, …` in `code`: its line and how the text after the comma
/// begins.
fn returns_nil(code: &str) -> Vec<(usize, Lead)> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'"' || c == b'\'' {
            i = string_at(code, i).0;
            continue;
        }
        if is_ident(c) && (i == 0 || !is_ident(b[i - 1])) {
            let start = i;
            while i < b.len() && is_ident(b[i]) {
                i += 1;
            }
            if &code[start..i] == "return" {
                let nil = skip_ws(b, i);
                let nil_end = nil + "nil".len();
                if code[nil..].starts_with("nil") && !b.get(nil_end).is_some_and(|n| is_ident(*n)) {
                    let comma = skip_ws(b, nil_end);
                    if b.get(comma) == Some(&b',') {
                        out.push((line_of(code, start), lead_at(code, comma + 1, true)));
                    }
                }
            }
            continue;
        }
        i += 1;
    }
    out
}

// ═════════════════════════════════════════════════════════════════════════════
// The relays
// ═════════════════════════════════════════════════════════════════════════════

/// What a relay's reason depends on, held over every write in its script.
#[derive(Clone, Copy, Debug)]
enum Pin {
    /// The name is assigned exactly once, on a line containing the text.
    One(&'static str, &'static str),
    /// The name is assigned at least once, and every value begins with a fixed
    /// lead.
    EachFixed(&'static str),
    /// The name is assigned at least once, and every value begins with the
    /// text.
    EachStarts(&'static str, &'static str),
    /// The script has exactly one `return nil, …`, and its text begins with a
    /// fixed lead (the `count_arg` helper's refusal).
    OneReturnNil,
    /// The script's code contains the text.
    Has(&'static str),
}

/// What is wrong with `pin` in `code`, if anything.
fn pin_failures(pin: Pin, code: &str) -> Vec<String> {
    let mut wrong = Vec::new();
    match pin {
        Pin::One(name, text) => {
            let found = writes(code, name);
            if found.len() != 1 || !found[0].stmt.contains(text) {
                wrong.push(format!(
                    "`{name}` must be assigned exactly once, as `{text}…`; its writes: {found:?}"
                ));
            }
        }
        Pin::EachFixed(name) => {
            let found = writes(code, name);
            if found.is_empty() {
                wrong.push(format!("`{name}` is never assigned"));
            }
            for write in found.iter().filter(|w| !fixed_lead(&w.value)) {
                wrong.push(format!(
                    "line {}: `{name}` is assigned a value without a fixed lead: {:?}",
                    write.line, write.value
                ));
            }
        }
        Pin::EachStarts(name, text) => {
            let found = writes(code, name);
            if found.is_empty() {
                wrong.push(format!("`{name}` is never assigned"));
            }
            for write in found.iter().filter(|w| !w.rhs.starts_with(text)) {
                wrong.push(format!(
                    "line {}: `{name}` is assigned `{}`, not `{text}…`",
                    write.line, write.rhs
                ));
            }
        }
        Pin::OneReturnNil => {
            let found = returns_nil(code);
            if found.len() != 1 || !fixed_lead(&found[0].1) {
                wrong.push(format!(
                    "the script must have exactly one `return nil, …`, with a fixed lead; it has \
                     {found:?}"
                ));
            }
        }
        Pin::Has(text) => {
            if !code.contains(text) {
                wrong.push(format!("`{text}` is no longer in the script"));
            }
        }
    }
    wrong
}

/// A piece of error text that does not begin with a fixed lead and is allowed
/// anyway: what it is, why it can never begin with data, and the pins that
/// hold the reason.
struct Relay {
    script: &'static str,
    sink: Sink,
    /// The expression, whitespace collapsed, as [`sites`] reports it.
    expr: &'static str,
    pins: &'static [Pin],
    reason: &'static str,
}

const COUNT_ARG_REASON: &str = "the second value `count_arg` returns, built only by its one \
     `return nil, \"ERR <script>: \" .. name .. …`: a fixed `ERR` lead";

const PCALL_ERR_REASON: &str = "the `err` of a `redis.pcall` reply: the server's own error \
     text, which begins with the server's error code, relayed so the caller classifies it as itself";

const REFUSAL_MSG_REASON: &str = "`REFUSAL.msg`, every write of which is a fixed tag literal \
     followed by what was read and the server's error, so the text begins with the tag";

const REFUSAL_RAISE_REASON: &str = "`REFUSAL`, a table the script declares empty once and gives \
     only a `msg` field (an `err` field would be a site of its own), raised to unwind to the \
     script's own `pcall(read)`, whose caller answers `redis.error_reply(REFUSAL.msg)`";

const RETHROW_REASON: &str = "rethrows, untouched, whatever `pcall(read)` caught other than \
     `REFUSAL`: a Lua runtime error (a string carrying Lua's own position prefix) or an error a \
     `redis.call` raised (the server's own error text). Every other raise inside `read` is a \
     site of its own";

/// The pins of a `count_arg` relay of `name`, written by `local <x>, <name> =`.
macro_rules! count_arg_pins {
    ($stmt:literal, $name:literal) => {
        &[Pin::One($name, $stmt), Pin::OneReturnNil]
    };
}

/// The pins of a `REFUSAL.msg` relay.
const REFUSAL_MSG_PINS: &[Pin] = &[
    Pin::EachFixed("REFUSAL.msg"),
    Pin::One("REFUSAL", "local REFUSAL = {}"),
];

/// The pins of an `error(REFUSAL)` relay.
const REFUSAL_RAISE_PINS: &[Pin] = &[
    Pin::One("REFUSAL", "local REFUSAL = {}"),
    Pin::EachFixed("REFUSAL.msg"),
    Pin::Has("if reply == REFUSAL then return redis.error_reply(REFUSAL.msg) end"),
];

/// The pins of an `error(reply)` relay.
const RETHROW_PINS: &[Pin] = &[
    Pin::One("reply", "local ok, reply = pcall(read)"),
    Pin::Has("if reply == REFUSAL then return redis.error_reply(REFUSAL.msg) end"),
];

/// Every allowed piece of error text without a fixed lead.
const RELAYS: &[Relay] = &[
    Relay {
        script: "snapshot_fence_acquire.lua",
        sink: Sink::ErrorReply,
        expr: "err_e",
        pins: count_arg_pins!("local est_e, err_e = count_arg(", "err_e"),
        reason: COUNT_ARG_REASON,
    },
    Relay {
        script: "snapshot_fence_acquire.lua",
        sink: Sink::ErrorReply,
        expr: "err_b",
        pins: count_arg_pins!("local est_b, err_b = count_arg(", "err_b"),
        reason: COUNT_ARG_REASON,
    },
    Relay {
        script: "snapshot_fence_release.lua",
        sink: Sink::ErrorReply,
        expr: "err_e",
        pins: count_arg_pins!("local measured_e, err_e = count_arg(", "err_e"),
        reason: COUNT_ARG_REASON,
    },
    Relay {
        script: "snapshot_fence_release.lua",
        sink: Sink::ErrorReply,
        expr: "err_b",
        pins: count_arg_pins!("local measured_b, err_b = count_arg(", "err_b"),
        reason: COUNT_ARG_REASON,
    },
    Relay {
        script: "snapshot_fence_renew.lua",
        sink: Sink::ErrorReply,
        expr: "err_e",
        pins: count_arg_pins!("local est_e, err_e = count_arg(", "err_e"),
        reason: COUNT_ARG_REASON,
    },
    Relay {
        script: "snapshot_fence_renew.lua",
        sink: Sink::ErrorReply,
        expr: "err_b",
        pins: count_arg_pins!("local est_b, err_b = count_arg(", "err_b"),
        reason: COUNT_ARG_REASON,
    },
    Relay {
        script: "snapshot_intent_register.lua",
        sink: Sink::ErrorReply,
        expr: "err_e",
        pins: count_arg_pins!("local est_e, err_e = count_arg(", "err_e"),
        reason: COUNT_ARG_REASON,
    },
    Relay {
        script: "snapshot_intent_register.lua",
        sink: Sink::ErrorReply,
        expr: "err_b",
        pins: count_arg_pins!("local est_b, err_b = count_arg(", "err_b"),
        reason: COUNT_ARG_REASON,
    },
    Relay {
        script: "stream_collision_probe.lua",
        sink: Sink::ErrorReply,
        expr: "entries.err",
        pins: &[Pin::One("entries", "local entries = redis.pcall('XRANGE',")],
        reason: PCALL_ERR_REASON,
    },
    Relay {
        script: "update_layer_fast.lua",
        sink: Sink::ErrorReply,
        expr: "xadd_err",
        pins: &[
            Pin::One("xadd_err", "local xadd_err = xadd_result.err"),
            Pin::EachStarts("xadd_result", "redis.pcall("),
        ],
        reason: PCALL_ERR_REASON,
    },
    Relay {
        script: "object_snapshot_read.lua",
        sink: Sink::ErrorReply,
        expr: "REFUSAL.msg",
        pins: REFUSAL_MSG_PINS,
        reason: REFUSAL_MSG_REASON,
    },
    Relay {
        script: "object_snapshot_read.lua",
        sink: Sink::Error,
        expr: "REFUSAL",
        pins: REFUSAL_RAISE_PINS,
        reason: REFUSAL_RAISE_REASON,
    },
    Relay {
        script: "object_snapshot_read.lua",
        sink: Sink::Error,
        expr: "reply",
        pins: RETHROW_PINS,
        reason: RETHROW_REASON,
    },
    Relay {
        script: "fanout_window_read.lua",
        sink: Sink::ErrorReply,
        expr: "REFUSAL.msg",
        pins: REFUSAL_MSG_PINS,
        reason: REFUSAL_MSG_REASON,
    },
    Relay {
        script: "fanout_window_read.lua",
        sink: Sink::Error,
        expr: "REFUSAL",
        pins: REFUSAL_RAISE_PINS,
        reason: REFUSAL_RAISE_REASON,
    },
    Relay {
        script: "fanout_window_read.lua",
        sink: Sink::Error,
        expr: "reply",
        pins: RETHROW_PINS,
        reason: RETHROW_REASON,
    },
    Relay {
        script: "structural_cache_sync.lua",
        sink: Sink::ErrorReply,
        expr: "REFUSAL.msg",
        pins: REFUSAL_MSG_PINS,
        reason: REFUSAL_MSG_REASON,
    },
    Relay {
        script: "structural_cache_sync.lua",
        sink: Sink::Error,
        expr: "REFUSAL",
        pins: REFUSAL_RAISE_PINS,
        reason: REFUSAL_RAISE_REASON,
    },
    Relay {
        script: "structural_cache_sync.lua",
        sink: Sink::Error,
        expr: "reply",
        pins: RETHROW_PINS,
        reason: RETHROW_REASON,
    },
];

fn relay_for(script: &str, site: &Site) -> Option<&'static Relay> {
    let Lead::Expr(expr) = &site.lead else {
        return None;
    };
    RELAYS
        .iter()
        .find(|r| r.script == script && r.sink == site.sink && r.expr == expr)
}

/// Every script on disk, by file name, with its text.
fn scripts() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(SCRIPTS_DIR);
    script_files()
        .into_iter()
        .map(|name| {
            let text =
                std::fs::read_to_string(dir.join(&name)).unwrap_or_else(|e| panic!("{name}: {e}"));
            (name, text)
        })
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// Rust string literals
// ═════════════════════════════════════════════════════════════════════════════

/// Every string literal in `text` (Rust source), with the 1-based line it
/// opens on and its contents: `"…"` and `b"…"` with their escapes decoded
/// (a `\` line continuation drops the newline and the next line's leading
/// whitespace), and `r"…"`, `r#"…"#`, `br"…"` as written. Comments and char
/// literals (`'"'`) are skipped, lifetimes are not mistaken for either.
fn rust_string_literals(text: &str) -> Vec<(usize, String)> {
    let b = text.as_bytes();
    let newlines: Vec<usize> = text.match_indices('\n').map(|(at, _)| at).collect();
    let line_of = |at: usize| newlines.partition_point(|&n| n < at) + 1;
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'/' && b.get(i + 1) == Some(&b'/') {
            i = text[i..].find('\n').map_or(b.len(), |n| i + n);
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'*') {
            i = text[i + 2..].find("*/").map_or(b.len(), |n| i + 2 + n + 2);
            continue;
        }
        if c == b'\'' {
            if b.get(i + 1) == Some(&b'\\') {
                // `'\''`, `'\n'`, `'\u{…}'`: the escaped character, then the
                // closing quote.
                let from = (i + 3).min(b.len());
                let close = text[from..].find('\'').map_or(b.len(), |n| from + n);
                i = close + 1;
                continue;
            }
            let width = text[i + 1..].chars().next().map_or(1, char::len_utf8);
            if b.get(i + 1 + width) == Some(&b'\'') {
                i += width + 2;
                continue;
            }
            i += 1;
            continue;
        }
        let starts_word = i == 0 || !is_ident(b[i - 1]);
        // A raw string: `r`, or `br`, then `#`s, then `"`.
        let raw_at = if starts_word && c == b'r' {
            Some(i + 1)
        } else if starts_word && c == b'b' && b.get(i + 1) == Some(&b'r') {
            Some(i + 2)
        } else {
            None
        };
        if let Some(mut j) = raw_at {
            let hashes_from = j;
            while b.get(j) == Some(&b'#') {
                j += 1;
            }
            if b.get(j) == Some(&b'"') {
                let closer = format!("\"{}", "#".repeat(j - hashes_from));
                let body = j + 1;
                let end = text[body..].find(&closer).map_or(b.len(), |n| body + n);
                out.push((line_of(i), text[body..end].to_owned()));
                i = (end + closer.len()).min(b.len());
                continue;
            }
        }
        if c == b'"' {
            let line = line_of(i);
            let mut contents = String::new();
            let mut chars = text[i + 1..].char_indices();
            let mut end = b.len();
            while let Some((at, ch)) = chars.next() {
                match ch {
                    '"' => {
                        end = i + 1 + at + 1;
                        break;
                    }
                    '\\' => match chars.next() {
                        Some((_, 'n')) => contents.push('\n'),
                        Some((_, 't')) => contents.push('\t'),
                        Some((_, 'r')) => contents.push('\r'),
                        Some((_, '0')) => contents.push('\0'),
                        Some((_, '\n')) => {
                            let rest = chars.as_str();
                            let skipped = rest.len() - rest.trim_start().len();
                            for _ in 0..rest[..skipped].chars().count() {
                                chars.next();
                            }
                        }
                        Some((_, other)) => contents.push(other),
                        None => {}
                    },
                    other => contents.push(other),
                }
            }
            out.push((line, contents));
            i = end;
            continue;
        }
        i += 1;
    }
    out
}

/// The functions whose presence marks a Rust string literal as Lua.
const LUA_CALLS: [&str; 4] = [
    "redis.call(",
    "redis.pcall(",
    "server.call(",
    "server.pcall(",
];

/// The one inline Lua script the service builds in Rust, as (file, text its
/// literal holds): the Rust scan must read it, so it cannot go vacuous on the
/// one real case while other literals still count.
const COUNTERS_LUA: (&str, &str) = (
    "src/storage/counters.rs",
    "tonumber(redis.call('GET', KEYS[1])",
);

/// Every `.rs` file under `src/`.
fn rust_sources() -> Vec<PathBuf> {
    let mut stack = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read src dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

// ═════════════════════════════════════════════════════════════════════════════
// The guards
// ═════════════════════════════════════════════════════════════════════════════

/// Every piece of error text in every script begins with a fixed,
/// non-redirect lead, or is a [`RELAYS`] entry; and no script opens a long
/// bracket the scanner would misread.
#[test]
fn every_script_error_text_begins_with_a_fixed_lead_or_is_a_listed_relay() {
    let scripts = scripts();
    assert!(
        scripts.len() > 20,
        "precondition: the scripts directory was read; found {}",
        scripts.len()
    );
    let mut sites_seen = 0;
    let mut wrong = Vec::new();
    for (name, source) in &scripts {
        for line in long_bracket_lines(source) {
            wrong.push(format!(
                "{name}:{line}: opens a long bracket, which this scan does not read"
            ));
        }
        for site in sites(&code_of(source)) {
            sites_seen += 1;
            if fixed_lead(&site.lead) || relay_for(name, &site).is_some() {
                continue;
            }
            wrong.push(format!(
                "{name}:{}: {:?} text begins with {:?}",
                site.line, site.sink, site.lead
            ));
        }
    }
    assert!(
        sites_seen > 100,
        "precondition: the scan found the scripts' error replies; found {sites_seen}"
    );
    assert!(
        wrong.is_empty(),
        "a script's error text must begin with a non-empty string literal whose first token is \
         not ASK or MOVED and which is the whole text or contains a space, so no data the script \
         read or was passed can make it a cluster redirect (SDD §15.3, vendored fred change 10). \
         Build the text from a literal tag, or, for a relay that can never begin with data, add a \
         RELAYS entry with its reason and pins. Offending sites: {wrong:#?}"
    );
}

/// What is wrong with the number of sites `relay` matches in `code` (its
/// script's code): it must match exactly one.
fn relay_site_failures(relay: &'static Relay, code: &str) -> Vec<String> {
    let matched = sites(code)
        .iter()
        .filter(|site| relay_for(relay.script, site).is_some_and(|r| std::ptr::eq(r, relay)))
        .count();
    match matched {
        1 => Vec::new(),
        0 => vec![format!(
            "{} no longer makes {:?} text from `{}`: remove the entry",
            relay.script, relay.sink, relay.expr
        )],
        n => vec![format!(
            "{} makes {:?} text from `{}` at {n} sites; a relay covers exactly one",
            relay.script, relay.sink, relay.expr
        )],
    }
}

/// Every [`RELAYS`] entry matches exactly one site of its script, and every
/// [`Pin`] of it holds over the script's assignments.
#[test]
fn every_relay_is_still_used_and_its_pins_hold() {
    let scripts = scripts();
    let mut wrong = Vec::new();
    for relay in RELAYS {
        assert!(!relay.reason.is_empty(), "every relay states its reason");
        assert!(!relay.pins.is_empty(), "every relay pins its reason");
        let Some((_, source)) = scripts.iter().find(|(name, _)| name == relay.script) else {
            wrong.push(format!("{}: no such script", relay.script));
            continue;
        };
        let code = code_of(source);
        wrong.extend(relay_site_failures(relay, &code));
        for pin in relay.pins {
            for failure in pin_failures(*pin, &code) {
                wrong.push(format!(
                    "{}: `{}`'s reason no longer holds: {failure}. Re-check the reason ({})",
                    relay.script, relay.expr, relay.reason
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "the relay list is out of step: {wrong:#?}"
    );
}

/// No Lua built in Rust makes error text: no string literal under `src/` that
/// calls a command ([`LUA_CALLS`]) holds an error-text site, and no Rust
/// source spells `error_reply` outside a `//` comment.
#[test]
fn no_rust_source_builds_lua_error_text() {
    let files = rust_sources();
    assert!(
        files.len() > 50,
        "precondition: src/ was read; found {} files",
        files.len()
    );
    let mut lua_literals = 0;
    let mut counters_scanned = false;
    let mut wrong = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path).expect("read source");
        for (n, line) in text.lines().enumerate() {
            if let Some(at) = line.find("error_reply") {
                if !line[..at].contains("//") {
                    wrong.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                }
            }
        }
        for (line, literal) in rust_string_literals(&text) {
            if !LUA_CALLS.iter().any(|call| literal.contains(call)) {
                continue;
            }
            lua_literals += 1;
            if path.ends_with(COUNTERS_LUA.0) && literal.contains(COUNTERS_LUA.1) {
                counters_scanned = true;
            }
            for bracket in long_bracket_lines(&literal) {
                wrong.push(format!(
                    "{}:{line}: Lua opening a long bracket on its line {bracket}, which this scan \
                     does not read",
                    path.display()
                ));
            }
            for site in sites(&code_of(&literal)) {
                wrong.push(format!(
                    "{}:{line}: Lua with {:?} text {:?} on its line {}",
                    path.display(),
                    site.sink,
                    site.lead,
                    site.line
                ));
            }
        }
    }
    assert!(
        lua_literals > 0 && counters_scanned,
        "precondition: the Rust-built Lua was read; {lua_literals} literals, and the inline script \
         in {} (a literal containing `{}`) {}",
        COUNTERS_LUA.0,
        COUNTERS_LUA.1,
        if counters_scanned {
            "was scanned"
        } else {
            "was not found: the literal reader lost its place, or the script moved (update \
             COUNTERS_LUA)"
        }
    );
    assert!(
        wrong.is_empty(),
        "Lua error text belongs in a script under src/storage/scripts, where its lead is checked \
         and a relay can be listed with its reason: {wrong:#?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The scanner's own cases
// ═════════════════════════════════════════════════════════════════════════════

/// The scanner finds every form of error text and judges its lead: a
/// literal-led text passes; a variable, `ARGV`, an empty, space-free or
/// dash-redirect literal before data, a redirect-shaped literal, an alias, a
/// table call, a name spelled as a string and a literal continued on the next
/// line fail; a comment, a string's contents and a plain `local ok, err = …`
/// are not sites.
#[test]
fn the_scanner_reads_every_form_of_error_text() {
    let script = [
        "return redis.error_reply('ENCODING: bad args')",
        "return redis.error_reply(\"STALE \" .. hdr['seq'])",
        "return redis.error_reply(",
        "  'NOT_FOUND: ' .. key)",
        "return redis.error_reply(ARGV[1])",
        "return redis . error_reply ( err_e )",
        "return server.error_reply(\"\" .. ARGV[1])",
        "return redis.error_reply('AS' .. ARGV[1])",
        "return redis.error_reply('ASK 1234 10.0.0.9:6379')",
        "return redis.error_reply('MOVED_TAG: ' .. x)",
        "local f = redis.error_reply",
        "return redis.error_reply{ 'x' }",
        "error({err = ARGV[1]})",
        "return {ok = 1, err = 'BUSY: try later'}",
        "t.err = value",
        "t[\"err\"] = 'TAG: ' .. value",
        "error(REFUSAL)",
        "error('plain message')",
        "-- return redis.error_reply(ARGV[1])",
        "local s = 'redis.error_reply(ARGV[1])'",
        "local ok, err = pcall(f)",
        "if r.err == nil then return end",
        "return {ok = 1; err = ARGV[1]}",
        "return redis['error_reply'](ARGV[1])",
        "return redis.error_reply('-ASK ' .. ARGV[1])",
        "t.err = 'AS'",
        "  .. ARGV[1]",
    ]
    .join("\n");
    let code = code_of(&script);
    let found: Vec<(usize, Sink, bool)> = sites(&code)
        .iter()
        .map(|site| (site.line, site.sink, fixed_lead(&site.lead)))
        .collect();
    assert_eq!(
        found,
        vec![
            (1, Sink::ErrorReply, true),
            (2, Sink::ErrorReply, true),
            (3, Sink::ErrorReply, true),
            (5, Sink::ErrorReply, false),
            (6, Sink::ErrorReply, false),
            (7, Sink::ErrorReply, false),
            (8, Sink::ErrorReply, false),
            (9, Sink::ErrorReply, false),
            (10, Sink::ErrorReply, false),
            (11, Sink::ErrorReply, false),
            (12, Sink::ErrorReply, false),
            (13, Sink::Error, false),
            (13, Sink::ErrField, false),
            (14, Sink::ErrField, true),
            (15, Sink::ErrField, false),
            (16, Sink::ErrField, true),
            (17, Sink::Error, false),
            (18, Sink::Error, true),
            (23, Sink::ErrField, false),
            (24, Sink::ErrorReply, false),
            (25, Sink::ErrorReply, false),
            (26, Sink::ErrField, false),
        ],
        "each line's sites and verdicts, in order; the scan saw {:#?}",
        sites(&code)
    );
    let spaced = sites(&code)
        .into_iter()
        .find(|site| site.line == 6)
        .expect("the spaced call is a site");
    assert_eq!(
        spaced.lead,
        Lead::Expr("err_e".to_owned()),
        "a call spelled with spaces is read by its argument"
    );
}

/// Each [`Pin`] holds over a script whose relayed names are written as the
/// reasons say, and fails once a second, data-led write is added: a
/// `REFUSAL.msg = r.err`, a second `return nil, ARGV[i]`, a reassigned
/// `err_e` / `xadd_err`, and an `xadd_result` not from `redis.pcall`.
#[test]
fn a_relay_pin_fails_on_a_second_data_led_write() {
    let clean = [
        "local REFUSAL = {}",
        "REFUSAL.msg = 'TAG: ' .. key",
        "local function count_arg(i)",
        "  if bad then return nil, \"ERR x: bad \" .. i end",
        "  return v",
        "end",
        "local est_e, err_e = count_arg(2)",
        "local xadd_result",
        "xadd_result = redis.pcall(",
        "  'XADD', k)",
        "local xadd_err = xadd_result.err",
        "local t = {REFUSAL = 1, err_e = 2}",
        "local u = t.err_e",
    ];
    let pins = [
        Pin::EachFixed("REFUSAL.msg"),
        Pin::One("REFUSAL", "local REFUSAL = {}"),
        Pin::OneReturnNil,
        Pin::One("err_e", "local est_e, err_e = count_arg("),
        Pin::EachStarts("xadd_result", "redis.pcall("),
        Pin::One("xadd_err", "local xadd_err = xadd_result.err"),
    ];
    let clean_code = code_of(&clean.join("\n"));
    for pin in pins {
        assert_eq!(
            pin_failures(pin, &clean_code),
            Vec::<String>::new(),
            "{pin:?} holds over the clean script"
        );
    }
    let second_writes = [
        (Pin::EachFixed("REFUSAL.msg"), "REFUSAL.msg = r.err"),
        (
            Pin::OneReturnNil,
            "local function g(i) return nil, ARGV[i] end",
        ),
        (
            Pin::One("err_e", "local est_e, err_e = count_arg("),
            "err_e = ARGV[1]",
        ),
        (
            Pin::One("err_e", "local est_e, err_e = count_arg("),
            "local a, err_e = 1, ARGV[1]",
        ),
        (
            Pin::One("REFUSAL", "local REFUSAL = {}"),
            "REFUSAL = {err = 'X: y'}",
        ),
        (
            Pin::EachStarts("xadd_result", "redis.pcall("),
            "xadd_result = {err = ARGV[2]}",
        ),
        (
            Pin::One("xadd_err", "local xadd_err = xadd_result.err"),
            "xadd_err = ARGV[1]",
        ),
    ];
    for (pin, extra) in second_writes {
        let mut dirty = clean.to_vec();
        dirty.push(extra);
        assert!(
            !pin_failures(pin, &code_of(&dirty.join("\n"))).is_empty(),
            "{pin:?} must fail once `{extra}` is added"
        );
    }
}

/// A relay matches exactly one site: its script's one `error_reply(REFUSAL.msg)`
/// passes, and a second site of the same text (here behind a parameter that
/// shadows `REFUSAL`, which no [`Pin`] reads) fails, as does none.
#[test]
fn a_relay_matching_a_second_site_fails() {
    let relay = RELAYS
        .iter()
        .find(|r| r.script == "fanout_window_read.lua" && r.expr == "REFUSAL.msg")
        .expect("the relay is listed");
    let one = [
        "local REFUSAL = {}",
        "local ok, reply = pcall(read)",
        "if reply == REFUSAL then return redis.error_reply(REFUSAL.msg) end",
    ];
    assert_eq!(
        relay_site_failures(relay, &code_of(&one.join("\n"))),
        Vec::<String>::new(),
        "one site passes"
    );
    let mut two = one.to_vec();
    two.push("local function f(REFUSAL) return redis.error_reply(REFUSAL.msg) end");
    two.push("return f({msg = ARGV[1]})");
    assert_eq!(
        relay_site_failures(relay, &code_of(&two.join("\n"))).len(),
        1,
        "a second site of the relay fails"
    );
    assert_eq!(
        relay_site_failures(relay, &code_of(&one[..2].join("\n"))).len(),
        1,
        "no site fails"
    );
}

/// The Rust literal reader finds the Lua in a plain string with escapes and a
/// line continuation, a raw string with hashes and a byte string, skips
/// comments and char literals, and is not thrown by a lifetime.
#[test]
fn the_rust_literal_reader_finds_lua_in_every_string_form() {
    let source = [
        "// \"return redis.error_reply(ARGV[1])\" in a comment",
        "/* \"error({err = ARGV[1]})\" in a block comment */",
        "fn f<'a>(x: &'a str) -> char { let _q = '\"'; '\\'' }",
        "const A: &str = \"if redis.call('GET', KEYS[1]) then\\n  \\",
        "    return {err = ARGV[1]} end\";",
        "const B: &str = r#\"return redis.pcall(\"GET\", KEYS[1])\"#;",
        "const C: &[u8] = b\"t.err = 'X: y'; redis.call('SET', KEYS[1], 1)\";",
        "const D: &str = \"redis.call('GET', KEYS[1])\";",
    ]
    .join("\n");
    let literals = rust_string_literals(&source);
    assert_eq!(
        literals,
        vec![
            (
                4,
                "if redis.call('GET', KEYS[1]) then\n  return {err = ARGV[1]} end".to_owned()
            ),
            (6, "return redis.pcall(\"GET\", KEYS[1])".to_owned()),
            (
                7,
                "t.err = 'X: y'; redis.call('SET', KEYS[1], 1)".to_owned()
            ),
            (8, "redis.call('GET', KEYS[1])".to_owned()),
        ],
        "every string literal, decoded, with the line it opens on"
    );
    let verdicts: Vec<Vec<bool>> = literals
        .iter()
        .map(|(_, lua)| {
            sites(&code_of(lua))
                .iter()
                .map(|site| fixed_lead(&site.lead))
                .collect()
        })
        .collect();
    assert_eq!(
        verdicts,
        vec![vec![false], vec![], vec![true], vec![]],
        "the guard reads each literal's sites (and fails on any of them, fixed-led or not)"
    );
}
