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

//! b.kgb — two source-shape guards over the production code under `src/`.
//!
//! Both rules are easy to break without any behavioural test noticing, because
//! the code that breaks them works on every path a test drives:
//!
//! - **Streaming RPCs take their shutdown token only from
//!   `GracefulShutdown::enroll_stream`** (SDD §5.6). A stream holding a bare
//!   child of the service shutdown token still ends at shutdown, but outside
//!   the paced ending: it is in no snapshot, so it ends with the last batch
//!   (a child of the parent) or all at once at t0 (a child of `begun`), and a
//!   stream opened after t0 is not refused. The reconnect storm b.kgb fixed
//!   comes back, on one handler, with every test green.
//! - **Every gRPC status detail is built by `crate::error`**
//!   (`RefusalDetail` / `refusal_status`), so the `ErrorInfo` / `RetryInfo`
//!   wire shape clients parse cannot drift between refusals.
//!
//! Both scans read **production** code only: comments and string-literal
//! contents are removed by the scanner shared with `txn_checkout_test.rs`
//! ([`code_lines_outside_strings`]), and every item under a `cfg` that holds
//! only in test builds (`#[cfg(test)]`, `#[cfg(all(test, …))]`) is removed by
//! [`production_code`], including out-of-line modules a `#[cfg(test)] mod
//! name;` declaration makes test-only. The removal is tested on a fixture by
//! [`guard_scanners_read_only_production_code_and_tell_observation_from_escape`].

// Guard tests: scripts/guard-tests.sh runs this whole module by name; rename it there too.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::txn_checkout_test::{code_lines_outside_strings, needle_hits, rust_sources_under};

/// The module that owns the service shutdown token, and the only production
/// file allowed to take a child of it.
const GRACEFUL_SHUTDOWN_MODULE: &str = "src/grpc/graceful_shutdown.rs";

/// The one builder of status details.
const ERROR_MODULE: &str = "src/error.rs";

/// The upload waits' module, whose `shutdown` parameters are `begun()`, and
/// whose commit charge (`UploadCommitCharge`) keeps the graceful-shutdown
/// handle itself as its `shutdown` field.
const UPLOAD_ADMISSION_MODULE: &str = "src/grpc/upload_admission.rs";

/// The external-value release budget's module (b.yuo), whose upload wait
/// `ExtReleaseGate::admit` also takes `begun()` as its `shutdown` parameter.
const EXT_RELEASE_GATE_MODULE: &str = "src/storage/ext_release_gate.rs";

/// The upload waits each upload-wait module defines taking a bare
/// `&CancellationToken`, sorted: every name in [`UPLOAD_WAITS`] is defined so
/// only in these modules.
const UPLOAD_WAIT_MODULES: [(&str, &[&str]); 2] = [
    (UPLOAD_ADMISSION_MODULE, &["admit", "reserve_processing"]),
    (EXT_RELEASE_GATE_MODULE, &["admit"]),
];

/// The streaming RPC handlers, each of which must enroll exactly once.
const STREAMING_HANDLERS: [&str; 2] = ["src/grpc/subscribe.rs", "src/grpc/presence.rs"];

/// How a streaming handler takes its token: the service's `shutdown` field
/// (a `GracefulShutdown`), enrolled.
const ENROLL_CALL: &str = ".shutdown.enroll_stream()";

/// The `GracefulShutdown` methods that hand out one of its shared tokens by
/// reference. The handle's only field is private, so in a build without the
/// `test-support` feature these are the only ways out of it (the guard pins
/// that, [`GRACEFUL_SHUTDOWN_RELEASE_API`]). Both names are defined nowhere
/// else in `src/` (also pinned), so they are matched on any receiver.
const SHARED_TOKEN_ACCESSORS: [&str; 2] = [".begun()", ".service_shutdown_token()"];

/// The public methods of `GracefulShutdown` a build without `test-support`
/// compiles. `is_cancelled()` / `cancelled()` read the parent token too; a
/// stream selecting on them would be a bare child of it, so they exist only in
/// test builds (`#[cfg(any(test, feature = "test-support"))]`), and the
/// feature-free release build refuses a production call.
const GRACEFUL_SHUTDOWN_RELEASE_API: [&str; 6] = [
    "begun",
    "cancel",
    "enroll_stream",
    "enrolled",
    "new",
    "service_shutdown_token",
];

/// Ways to read a shared token without keeping it or deriving one from it.
const OBSERVATIONS: [&str; 2] = [".cancelled()", ".is_cancelled()"];

/// The upload waits that take `begun()` by reference and race it
/// ([`UPLOAD_WAIT_MODULES`], where the parameter is named `shutdown`).
/// Passing the token to any other call hands it to code this guard does not
/// read.
const UPLOAD_WAITS: [&str; 2] = ["admit", "reserve_processing"];

/// Every production read of the handle's shared tokens, exactly: `(file, kind,
/// count, why it is not a stream's ending)`. An observation is harmless only
/// where it is not a stream's ending — a stream that selects on
/// `begun().cancelled()` ends at t0 with every other such stream, the storm
/// b.kgb removed — and the scan cannot tell a refusal check from an ending, so
/// each one is listed here, and a new one fails the guard until it is
/// justified in this list. Every file not listed must have none.
const ALLOWED_OBSERVATIONS: [(&str, &str, usize, &str); 8] = [
    (
        "src/grpc/presence.rs",
        "begun().is_cancelled()",
        1,
        "JoinSession refuses a join that arrives after t0; a check at admission, not an ending",
    ),
    (
        "src/grpc/transport.rs",
        "begun().cancelled()",
        1,
        "the bound after GOAWAY starts its clock at t0; no stream waits on it",
    ),
    (
        "src/grpc/transport.rs",
        "service_shutdown_token().cancelled()",
        1,
        "the bound after GOAWAY waits for the paced ending to finish; no stream waits on it",
    ),
    (
        "src/grpc/layer.rs",
        "begun() -> admit",
        2,
        "an upload waiting in admission, or at the external-value release budget \
         (`ExtReleaseGate::admit`, b.yuo), is refused at t0 with nothing applied; uploads are not \
         paced streams",
    ),
    (
        "src/grpc/layer.rs",
        "begun() -> reserve_processing",
        1,
        "an upload waiting for processing room is refused at t0 with nothing applied",
    ),
    (
        "src/grpc/layer.rs",
        "begun().cancelled()",
        1,
        "a chained UpdateLayer waiting in the chain gate for its predecessor is refused at t0 \
         with nothing applied; the race ends a wait, not a stream",
    ),
    (
        "src/grpc/upload_admission.rs",
        "begun() -> reserve_processing",
        1,
        "a commit's recharge between attempts (`UploadCommitCharge::hold`) waits for processing \
         room and is refused at t0; an upload is not a paced stream",
    ),
    (
        "src/main.rs",
        "begun().is_cancelled()",
        1,
        "the replica autoscaler's `shutting_down` probe reports a replica that has begun to shut \
         down; not a stream",
    ),
];

/// Continuations of an accessor, or of an upload wait's `shutdown` parameter,
/// that keep or derive a token.
const ESCAPES: [&str; 3] = [".child_token()", ".clone()", ".to_owned()"];

/// The status-detail types of `tonic_types`. Naming one is fine (a parse
/// annotates `Option<ErrorInfo>`); constructing one, or renaming it with a
/// `type` alias or `use … as`, is not.
const DETAIL_TYPES: [&str; 4] = ["ErrorDetails", "ErrorDetail", "ErrorInfo", "RetryInfo"];

/// Status-detail constructors and setters, matched as identifier PREFIXES, so
/// every variant is caught: `with_error_details` covers tonic-types'
/// `with_error_details_vec`, `with_error_details_and_metadata` and
/// `with_error_details_vec_and_metadata`; `with_details` covers tonic's
/// raw-bytes `Status::with_details` and `with_details_and_metadata`. No other
/// identifier in `src/` starts with one of these.
const DETAIL_BUILDERS: [&str; 4] = [
    "with_error_details",
    "with_details",
    "set_error_info",
    "set_retry_info",
];

/// The trailer a status detail travels in.
const DETAILS_HEADER: &str = "grpc-status-details-bin";

/// Reads of [`DETAILS_HEADER`]. Any other code naming it — an `insert`, an
/// `append`, a `from_static`, a constant that hides it from this scan — is
/// treated as a write.
const DETAILS_HEADER_READS: [&str; 3] = ["get(", "get_all(", "contains_key("];

// ─────────────────────────────────────────────────────────────────────────────
// The production-code view
// ─────────────────────────────────────────────────────────────────────────────
//
// `pub(super)` where `ext_blob_reader_guard_test.rs` (b.bnn) reads `src/`
// through the same views: [`ProductionTree`] (its release-build walk,
// `ProductionTree::load_release`), [`ProductionCode`], [`production_code`],
// [`FlatCode`] and [`is_ident`]. One set of views, tested once below, rather
// than two that can drift.

/// One file's code, minus comments, string-literal contents and test-only
/// items.
pub(super) struct ProductionCode {
    /// `(1-based line number, code)` for every line of the file; a line wholly
    /// inside a test-only item has empty code.
    pub(super) lines: Vec<(usize, String)>,
    /// Names from `#[cfg(test)] mod name;` declarations: out-of-line modules
    /// whose files are test-only.
    test_only_modules: Vec<String>,
}

impl ProductionCode {
    fn code_at(&self, line: usize) -> &str {
        self.lines
            .get(line - 1)
            .map(|(_, code)| code.as_str())
            .unwrap_or("")
    }
}

/// Does a `cfg` predicate hold only in test builds?
///
/// `test`, an `all(…)` with such an argument, and an `any(…)` whose every
/// argument is one. `any(test, feature = "test-support")` is NOT test-only —
/// a `test-support` build compiles it — so it stays in the scan.
fn cfg_requires_test(predicate: &str) -> bool {
    cfg_holds_only_when(predicate, &|leaf| leaf == "test")
}

/// Does a `cfg` predicate hold only in a test build or a build with a test
/// feature (`test-support`, or any other `test-*` feature)? What the
/// feature-free release build compiles is everything this is false for.
///
/// The predicate comes from the production view, whose string literals are
/// emptied (`feature = ""`), so a feature leaf is read from `raw_line`, the
/// attribute's line as written: it counts as a test feature only when every
/// feature that line names is a `test-*` one. A line that also names another
/// feature keeps its item in the release view (the loud direction).
fn cfg_requires_test_or_test_feature(predicate: &str, raw_line: &str) -> bool {
    let features: Vec<&str> = raw_line
        .split("feature")
        .skip(1)
        .filter_map(|rest| rest.trim_start().strip_prefix('='))
        .filter_map(|rest| rest.trim_start().strip_prefix('"'))
        .filter_map(|rest| rest.split_once('"').map(|(name, _)| name))
        .collect();
    let only_test_features =
        !features.is_empty() && features.iter().all(|name| name.starts_with("test-"));
    cfg_holds_only_when(predicate, &|leaf| {
        leaf == "test"
            || (only_test_features
                && leaf
                    .strip_prefix("feature")
                    .is_some_and(|rest| rest.trim_start().starts_with('=')))
    })
}

/// Does `predicate` hold only when some `leaf` holds: a matching leaf, an
/// `all(…)` with such an argument, or an `any(…)` whose every argument is one.
fn cfg_holds_only_when(predicate: &str, leaf: &dyn Fn(&str) -> bool) -> bool {
    let predicate = predicate.trim();
    if leaf(predicate) {
        return true;
    }
    if let Some(args) = predicate
        .strip_prefix("all(")
        .and_then(|rest| rest.strip_suffix(')'))
    {
        return top_level_args(args)
            .into_iter()
            .any(|arg| cfg_holds_only_when(arg, leaf));
    }
    if let Some(args) = predicate
        .strip_prefix("any(")
        .and_then(|rest| rest.strip_suffix(')'))
    {
        let args = top_level_args(args);
        return !args.is_empty() && args.into_iter().all(|arg| cfg_holds_only_when(arg, leaf));
    }
    false
}

/// `args` split at its top-level commas.
fn top_level_args(args: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (at, c) in args.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&args[start..at]);
                start = at + 1;
            }
            _ => {}
        }
    }
    out.push(&args[start..]);
    out.into_iter()
        .filter(|arg| !arg.trim().is_empty())
        .collect()
}

/// A test-only `cfg` attribute starting at `chars[at]`: `Some((length,
/// inner))` for `#[cfg(<test-only>)]` (`inner` false) or `#![cfg(…)]` (`inner`
/// true). The whole attribute must sit on one line; a multi-line one is not
/// recognised, which leaves its item in the scan (the loud direction).
fn test_only_attribute_at(
    chars: &[char],
    at: usize,
    excluded: &dyn Fn(&str) -> bool,
) -> Option<(usize, bool)> {
    if chars.get(at) != Some(&'#') {
        return None;
    }
    let inner = chars.get(at + 1) == Some(&'!');
    let open = at + 1 + usize::from(inner);
    if chars.get(open) != Some(&'[') {
        return None;
    }
    let mut depth = 0i32;
    let mut close = None;
    for (offset, c) in chars[open..].iter().enumerate() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let body: String = chars[open + 1..close].iter().collect();
    let predicate = body
        .trim()
        .strip_prefix("cfg")?
        .trim_start()
        .strip_prefix('(')?
        .strip_suffix(')')?;
    excluded(predicate).then_some((close + 1 - at, inner))
}

/// The name a `mod name;` item declares, if `item` is one.
fn out_of_line_module_name(item: &str) -> Option<String> {
    let item = item.trim();
    let item = ["pub(crate) ", "pub(super) ", "pub "]
        .iter()
        .find_map(|vis| item.strip_prefix(vis))
        .unwrap_or(item);
    let name = item.strip_prefix("mod ")?.strip_suffix(';')?.trim();
    (!name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_'))
        .then(|| name.to_string())
}

/// The length of the inner attribute (`#![…]`, on one line) starting at
/// `chars[at]`, if one does.
fn inner_attribute_len_at(chars: &[char], at: usize) -> Option<usize> {
    if chars.get(at) != Some(&'#')
        || chars.get(at + 1) != Some(&'!')
        || chars.get(at + 2) != Some(&'[')
    {
        return None;
    }
    let mut depth = 0i32;
    for (offset, c) in chars[at + 2..].iter().enumerate() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(offset + 3);
                }
            }
            _ => {}
        }
    }
    None
}

/// The item a test-only attribute applies to, being skipped.
struct SkippedItem {
    depth: i32,
    text: String,
    /// Skipping the rest of the enclosing block (a test-only inner attribute
    /// after the file's first item): only the `}` that closes that block, or
    /// the end of the file, ends it.
    to_block_end: bool,
}

/// `source` as production code: [`code_lines_outside_strings`] with every
/// item under a test-only `cfg` removed.
///
/// The skipped item runs from the attribute to the first `;` or `,` at its own
/// nesting level (a `use`, a `mod name;`, a struct field), to the `}` that
/// closes its first top-level brace (a `mod`, `fn` or `impl` body), or to the
/// `}` that closes the block it sits in (a last field without a trailing
/// comma). Generic angle brackets are not tracked, so a field like
/// `HashMap<u8, u8>` ends at its inner comma and its tail stays in the scan —
/// again the loud direction. A test-only inner attribute (`#![cfg(test)]`)
/// before the file's first item (other inner attributes aside) removes the
/// whole file; anywhere else it removes the rest of the block it sits in (an
/// inline `mod name { #![cfg(test)] … }`), and only that.
pub(super) fn production_code(source: &str) -> ProductionCode {
    code_without(source, |predicate, _| cfg_requires_test(predicate))
}

/// `source` as the feature-free release build compiles it: like
/// [`production_code`], but also without items under a test feature
/// (`#[cfg(any(test, feature = "test-support"))]`).
fn release_code(source: &str) -> ProductionCode {
    code_without(source, cfg_requires_test_or_test_feature)
}

/// [`code_lines_outside_strings`] with every item under a `cfg` attribute
/// whose predicate `excluded` accepts removed ([`production_code`]).
/// `excluded` is handed the predicate and the attribute's line as written.
fn code_without(source: &str, excluded: fn(&str, &str) -> bool) -> ProductionCode {
    let raw_lines: Vec<&str> = source.lines().collect();
    let mut lines = Vec::new();
    let mut test_only_modules = Vec::new();
    let mut skipping: Option<SkippedItem> = None;
    let mut whole_file = false;
    // Whether any code other than an inner attribute has been kept yet.
    let mut seen_item = false;
    for (line_no, code) in code_lines_outside_strings(source) {
        let raw_line = raw_lines.get(line_no - 1).copied().unwrap_or("");
        let excluded_here = |predicate: &str| excluded(predicate, raw_line);
        let chars: Vec<char> = code.chars().collect();
        let mut kept = String::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if let Some(item) = skipping.as_mut() {
                item.text.push(c);
                i += 1;
                let ended = match c {
                    '(' | '[' | '{' => {
                        item.depth += 1;
                        false
                    }
                    ')' | ']' | '}' => {
                        item.depth -= 1;
                        item.depth < 0 || (item.depth == 0 && c == '}' && !item.to_block_end)
                    }
                    ';' | ',' => item.depth == 0 && !item.to_block_end,
                    _ => false,
                };
                if ended {
                    if let Some(name) = skipping
                        .take()
                        .and_then(|item| out_of_line_module_name(&item.text))
                    {
                        test_only_modules.push(name);
                    }
                }
                continue;
            }
            if let Some((len, inner)) = test_only_attribute_at(&chars, i, &excluded_here) {
                if inner && !seen_item {
                    whole_file = true;
                } else {
                    skipping = Some(SkippedItem {
                        depth: 0,
                        text: String::new(),
                        to_block_end: inner,
                    });
                }
                i += len;
                continue;
            }
            if let Some(len) = inner_attribute_len_at(&chars, i) {
                kept.extend(&chars[i..i + len]);
                i += len;
                continue;
            }
            seen_item |= !c.is_whitespace();
            kept.push(c);
            i += 1;
        }
        if let Some(item) = skipping.as_mut() {
            item.text.push(' ');
        }
        lines.push((line_no, kept.trim().to_string()));
    }
    if whole_file {
        for (_, code) in &mut lines {
            code.clear();
        }
    }
    ProductionCode {
        lines,
        test_only_modules,
    }
}

/// Where the out-of-line module `name`, declared in `file`, lives: its
/// `name.rs` file and its `name/` directory.
fn test_only_module_paths(file: &Path, name: &str) -> (PathBuf, PathBuf) {
    let parent = file.parent().expect("a source file has a parent directory");
    let stem = file
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("a source file has a UTF-8 stem");
    let dir = if matches!(stem, "mod" | "lib" | "main") {
        parent.to_path_buf()
    } else {
        parent.join(stem)
    };
    (dir.join(format!("{name}.rs")), dir.join(name))
}

/// Production code flattened to one character stream, whitespace runs
/// collapsed and the whitespace around every `.` removed, so a method chain
/// rustfmt splits across lines (`svc.shutdown\n    .begun()`) reads as one.
/// Each character remembers its source line.
pub(super) struct FlatCode {
    pub(super) text: Vec<char>,
    pub(super) line_of: Vec<usize>,
}

impl FlatCode {
    pub(super) fn new(code: &ProductionCode) -> Self {
        let mut text: Vec<char> = Vec::new();
        let mut line_of = Vec::new();
        for (line_no, code) in &code.lines {
            for c in code.chars().chain(std::iter::once(' ')) {
                let c = if c.is_whitespace() { ' ' } else { c };
                if c == ' ' && matches!(text.last(), None | Some(' ' | '.')) {
                    continue;
                }
                if c == '.' && text.last() == Some(&' ') {
                    text.pop();
                    line_of.pop();
                }
                text.push(c);
                line_of.push(*line_no);
            }
        }
        Self { text, line_of }
    }

    fn string(&self) -> String {
        self.text.iter().collect()
    }

    fn starts_with_at(&self, at: usize, needle: &str) -> bool {
        let mut i = at;
        for c in needle.chars() {
            if self.text.get(i) != Some(&c) {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Every index `needle` starts at.
    fn find_all(&self, needle: &str) -> Vec<usize> {
        (0..self.text.len())
            .filter(|&at| self.starts_with_at(at, needle))
            .collect()
    }

    /// Every index `needle` starts at as a whole identifier: not preceded by
    /// an identifier character (`MyErrorInfo` is not `ErrorInfo`).
    fn find_standalone(&self, needle: &str) -> Vec<usize> {
        self.find_all(needle)
            .into_iter()
            .filter(|&at| at == 0 || !is_ident(self.text[at - 1]))
            .collect()
    }

    /// The identifier immediately before `at` (`shutdown` in
    /// `svc.shutdown.begun()` when `at` is the `.` of `.begun()`); empty when
    /// the receiver is a call or index result.
    fn receiver_ident(&self, at: usize) -> String {
        let start = (0..at)
            .rev()
            .take_while(|&i| is_ident(self.text[i]))
            .last()
            .unwrap_or(at);
        self.text[start..at].iter().collect()
    }

    /// The name of the call whose argument list encloses `at`, if any:
    /// `admit` for a token at `.admit(rpc, id, svc.shutdown.begun())`. `None`
    /// when `at` is in a block, an index, or a statement of its own.
    fn enclosing_call(&self, at: usize) -> Option<String> {
        let mut depth = 0i32;
        for i in (0..at).rev() {
            match self.text[i] {
                ')' | ']' | '}' => depth += 1,
                '(' | '[' | '{' if depth > 0 => depth -= 1,
                '(' => {
                    let name = self.receiver_ident(i);
                    return (!name.is_empty()).then_some(name);
                }
                '[' | '{' => return None,
                ';' if depth == 0 => return None,
                _ => {}
            }
        }
        None
    }
}

pub(super) fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Every `.rs` file under `src/` that is production code, read once.
pub(super) struct ProductionTree {
    pub(super) files: Vec<ProductionFile>,
}

pub(super) struct ProductionFile {
    pub(super) path: PathBuf,
    pub(super) source: String,
    pub(super) code: ProductionCode,
}

impl ProductionFile {
    /// Line `line` (1-based) as written, for a failure message: the
    /// production view empties string literals, which would hide a
    /// `"grpc-status-details-bin"` behind `""`.
    fn source_line(&self, line: usize) -> &str {
        self.source.lines().nth(line - 1).unwrap_or("").trim()
    }
}

impl ProductionTree {
    /// Walk `src/` and drop every file a `#[cfg(test)] mod name;` declaration
    /// makes test-only (today `src/allocator/structural_cache_calibration.rs`).
    pub(super) fn load() -> Self {
        Self::load_with(production_code)
    }

    /// [`Self::load`] as the feature-free release build compiles `src/`
    /// ([`release_code`]): items and out-of-line modules under a test feature
    /// (`#[cfg(any(test, feature = "test-support"))]`) are dropped too.
    pub(super) fn load_release() -> Self {
        Self::load_with(release_code)
    }

    /// The walk behind [`Self::load`] and [`Self::load_release`], each file
    /// read through `view`.
    fn load_with(view: fn(&str) -> ProductionCode) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut walked = Vec::new();
        rust_sources_under(&root, &root.join("src"), &mut walked);
        walked.sort();
        let mut files: Vec<ProductionFile> = walked
            .into_iter()
            .map(|path| {
                let source = std::fs::read_to_string(root.join(&path))
                    .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", path.display()));
                let code = view(&source);
                ProductionFile { path, source, code }
            })
            .collect();
        let test_only: Vec<(PathBuf, PathBuf)> = files
            .iter()
            .flat_map(|file| {
                file.code
                    .test_only_modules
                    .iter()
                    .map(|name| test_only_module_paths(&file.path, name))
            })
            .collect();
        files.retain(|file| {
            !test_only.iter().any(|(module_file, module_dir)| {
                file.path == *module_file || file.path.starts_with(module_dir)
            })
        });
        Self { files }
    }

    /// Non-vacuity: the production walk reaches every file in `expected`.
    pub(super) fn assert_visits(&self, expected: &[&str]) {
        for expected in expected {
            assert!(
                self.files.iter().any(|f| f.path == Path::new(expected)),
                "the production scan did not visit `{expected}`; it read {} files. Either the \
                 file moved (update this list) or the walk is not reaching where it should",
                self.files.len()
            );
        }
    }

    pub(super) fn file(&self, path: &str) -> &ProductionFile {
        self.files
            .iter()
            .find(|f| f.path == Path::new(path))
            .unwrap_or_else(|| panic!("`{path}` is not in the production scan"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Guard 1 — shutdown tokens
// ─────────────────────────────────────────────────────────────────────────────

/// What [`scan_shutdown_tokens`] found in one file.
#[derive(Default)]
struct ShutdownTokenScan {
    /// `(line, why)` for every site that keeps or derives one of the
    /// handle's shared tokens.
    escapes: Vec<(usize, String)>,
    /// Sites that read a shared token without keeping it, counted by kind: a
    /// `.cancelled()` / `.is_cancelled()` observation (`begun().cancelled()`),
    /// or `begun()` handed to an upload wait (`begun() -> admit`).
    observations: BTreeMap<String, usize>,
    /// [`ENROLL_CALL`] sites.
    enrollments: usize,
}

/// Classify every use of the `GracefulShutdown` handle's shared tokens in
/// `source` (a file other than [`GRACEFUL_SHUTDOWN_MODULE`]);
/// `upload_waits` is true for the modules of [`UPLOAD_WAIT_MODULES`].
///
/// - `.begun()` and `.service_shutdown_token()` ([`SHARED_TOKEN_ACCESSORS`],
///   names nothing else in `src/` defines), on any receiver, must be observed
///   on the spot ([`OBSERVATIONS`]) or, for `begun()` only, passed straight to
///   an upload wait ([`UPLOAD_WAITS`]). Binding one to a local, storing it, or
///   passing it anywhere else is an escape: the next line can take a child of
///   it out of this scan's sight. So is either accessor followed by
///   [`ESCAPES`]. The observations are counted by kind; which are allowed,
///   and where, is [`ALLOWED_OBSERVATIONS`]'s business.
/// - Either accessor's name anywhere else — a fully qualified call
///   (`GracefulShutdown::begun(&handle)`) or a path used as a function
///   pointer — is an escape: the method-call scan above cannot follow it.
/// - In the upload waits' modules, `shutdown` — the name their waits give
///   the `begun()` they are handed, and the commit charge
///   (`UploadCommitCharge`) its graceful-shutdown handle field — followed by
///   [`ESCAPES`] is an escape. In every other file a receiver named `shutdown` is a stream's own enrolled
///   token (`subscribe.rs`, `presence.rs`), or another token entirely, and a
///   child of it is not this guard's business.
fn scan_shutdown_tokens(source: &str, upload_waits: bool) -> ShutdownTokenScan {
    let code = production_code(source);
    let flat = FlatCode::new(&code);
    let mut scan = ShutdownTokenScan::default();
    for accessor in SHARED_TOKEN_ACCESSORS {
        for at in flat.find_all(accessor) {
            let after = at + accessor.len();
            let line = flat.line_of[at];
            if let Some(escape) = ESCAPES.iter().find(|e| flat.starts_with_at(after, e)) {
                scan.escapes.push((
                    line,
                    format!("`{accessor}{escape}` keeps or derives the token"),
                ));
                continue;
            }
            if let Some(observation) = OBSERVATIONS.iter().find(|o| flat.starts_with_at(after, o)) {
                *scan
                    .observations
                    .entry(format!("{}{observation}", &accessor[1..]))
                    .or_default() += 1;
                continue;
            }
            let as_argument = flat.starts_with_at(after, ")") || flat.starts_with_at(after, ",");
            let call = flat.enclosing_call(at);
            if accessor == ".begun()"
                && as_argument
                && call
                    .as_deref()
                    .is_some_and(|call| UPLOAD_WAITS.contains(&call))
            {
                *scan
                    .observations
                    .entry(format!(
                        "begun() -> {}",
                        call.as_deref().unwrap_or_default()
                    ))
                    .or_default() += 1;
                continue;
            }
            scan.escapes.push((
                line,
                format!(
                    "`{accessor}` is neither observed on the spot nor handed to an upload wait \
                     (enclosing call: {call:?})"
                ),
            ));
        }
    }
    for accessor in SHARED_TOKEN_ACCESSORS {
        let name = &accessor[1..accessor.len() - 2];
        for at in flat.find_standalone(name) {
            let after = at + name.len();
            if flat.text.get(after).is_some_and(|c| is_ident(*c)) {
                continue;
            }
            if at > 0 && flat.text[at - 1] == '.' && flat.starts_with_at(after, "()") {
                // A method call: classified above.
                continue;
            }
            scan.escapes.push((
                flat.line_of[at],
                format!(
                    "`{name}` named other than as a method call (a fully qualified call or a \
                     function pointer), which this scan cannot follow"
                ),
            ));
        }
    }
    if upload_waits {
        for escape in ESCAPES {
            for at in flat.find_all(escape) {
                if flat.receiver_ident(at) == "shutdown" {
                    scan.escapes.push((
                        flat.line_of[at],
                        format!(
                            "`shutdown{escape}` keeps or derives the `begun()` an upload wait was \
                             handed"
                        ),
                    ));
                }
            }
        }
    }
    scan.enrollments = flat.find_all(ENROLL_CALL).len();
    scan.escapes.sort();
    scan
}

/// The public methods of `GracefulShutdown` (`pub` or `pub(crate)`, from every
/// `impl GracefulShutdown` block) that the feature-free release build
/// compiles, sorted.
fn release_methods_of_graceful_shutdown(source: &str) -> Vec<String> {
    let flat = FlatCode::new(&release_code(source));
    let text = flat.string();
    let mut names = Vec::new();
    for (start, _) in text.match_indices("impl GracefulShutdown {") {
        let body_start = start + "impl GracefulShutdown {".len();
        let mut depth = 1i32;
        let mut body_end = text.len();
        for (offset, c) in text[body_start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        body_end = body_start + offset;
                        break;
                    }
                }
                _ => {}
            }
        }
        let body = &text[body_start..body_end];
        for (at, _) in body.match_indices("fn ") {
            let head = body[..at].trim_end();
            let head = head.strip_suffix("async").unwrap_or(head).trim_end();
            if head.ends_with("pub") || head.ends_with("pub(crate)") {
                names.push(
                    body[at + 3..]
                        .chars()
                        .take_while(|c| is_ident(*c))
                        .collect(),
                );
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

/// The names of the functions in `flat` whose signature mentions
/// `CancellationToken` — the ways a module can hand one out.
fn functions_handing_out_tokens(flat: &str) -> Vec<String> {
    let mut names: Vec<String> = flat
        .match_indices("fn ")
        .filter(|(at, _)| *at == 0 || !flat[..*at].ends_with(is_ident))
        .filter_map(|(at, _)| {
            let rest = &flat[at + 3..];
            let signature_end = rest.find(['{', ';']).unwrap_or(rest.len());
            rest[..signature_end]
                .contains("CancellationToken")
                .then(|| rest.chars().take_while(|c| is_ident(*c)).collect())
        })
        .collect();
    names.sort();
    names
}

/// The names of the functions in `flat` that take a bare `&CancellationToken`
/// (not an `Option<&CancellationToken>`) — the shape of an upload wait.
fn functions_taking_a_bare_token(flat: &str) -> Vec<String> {
    let mut names: Vec<String> = flat
        .match_indices("fn ")
        .filter(|(at, _)| *at == 0 || !flat[..*at].ends_with(is_ident))
        .filter_map(|(at, _)| {
            let rest = &flat[at + 3..];
            let signature_end = rest.find(['{', ';']).unwrap_or(rest.len());
            rest[..signature_end]
                .replace("Option<&CancellationToken>", "")
                .contains("&CancellationToken")
                .then(|| rest.chars().take_while(|c| is_ident(*c)).collect())
        })
        .collect();
    names.sort();
    names
}

/// **Streaming RPC handlers take their shutdown token only from
/// `GracefulShutdown::enroll_stream`**, and nothing outside
/// `src/grpc/graceful_shutdown.rs` takes a child of, or keeps, the service
/// shutdown token (SDD §5.6).
///
/// A bare child token works: the stream still ends at shutdown, and every
/// behavioural test of that handler passes. What it loses is the paced ending.
/// It is in no snapshot, so it ends either with the last batch (a child of the
/// parent, after the server has stopped waiting for it) or all at once at t0
/// (a child of `begun`, the step b.kgb replaced with a ramp), and a stream
/// opened after t0 is not refused, because only a closed registry refuses.
///
/// # What is allowed, by name
///
/// - **`src/grpc/graceful_shutdown.rs`** — the handle's own internals: the
///   parent's two `child_token()` calls (`begun`, and each enrollment).
/// - **Observations, exactly as listed** ([`ALLOWED_OBSERVATIONS`], per file
///   and kind): JoinSession's t0 refusal, a chained `UpdateLayer`'s
///   chain-gate wait in `layer.rs`, the bound after GOAWAY in
///   `transport.rs`, the replica autoscaler's `shutting_down` probe in
///   `main.rs`. `.cancelled()` / `.is_cancelled()` on a shared token is not
///   harmless in itself — a stream that selects on `begun().cancelled()` ends
///   at t0 with every other such stream — so any other one fails the guard.
/// - **The upload waits** — `begun()` handed straight to `admit` /
///   `reserve_processing`, which race it (`layer.rs`, also listed; and
///   `upload_admission.rs`, also listed, where `UploadCommitCharge::hold`
///   reads it from the handle it keeps and hands it straight to
///   `reserve_processing`), and which only observe it (`upload_admission.rs`,
///   and the external-value release budget's `admit` in
///   `ext_release_gate.rs`). Those names take a bare
///   `&CancellationToken` only there (pinned), so the call name identifies
///   the wait.
/// - **Other tokens** — `child_token()` of a stream's own enrolled token, a
///   session token, the router supervisor's or a background task's is not
///   this guard's business.
///
/// The handle's only field is private. In a build without the `test-support`
/// feature, `begun()` and `service_shutdown_token()` are the only ways to its
/// shared tokens: its `is_cancelled()` / `cancelled()` read the parent too,
/// but are compiled only with `test-support` (the feature-free release build
/// refuses a production call). The guard pins that public surface, the set
/// of the module's functions that hand out a token, and that no other file
/// defines either accessor name, so a new way out has to come past this
/// guard.
#[test]
fn streaming_handlers_take_their_shutdown_token_only_from_enroll_stream() {
    let tree = ProductionTree::load();
    tree.assert_visits(&[
        GRACEFUL_SHUTDOWN_MODULE,
        STREAMING_HANDLERS[0],
        STREAMING_HANDLERS[1],
        "src/grpc/transport.rs",
        "src/grpc/layer.rs",
        "src/main.rs",
        "src/storage/router_supervisor.rs",
        EXT_RELEASE_GATE_MODULE,
    ]);

    // The handle's encapsulation is what makes the accessor scan complete:
    // with a public field, or a new method returning or awaiting a token, a
    // stream could reach the parent without naming an accessor.
    let handle = FlatCode::new(&tree.file(GRACEFUL_SHUTDOWN_MODULE).code).string();
    for anchor in [
        "pub struct GracefulShutdown { shared: Arc<Shared>, }",
        "pub fn enroll_stream(&self) -> (CancellationToken, StreamEnrollment)",
        "pub fn begun(&self) -> &CancellationToken",
        "pub(crate) fn service_shutdown_token(&self) -> &CancellationToken",
    ] {
        assert!(
            handle.contains(anchor),
            "`{anchor}` is no longer in {GRACEFUL_SHUTDOWN_MODULE}'s production code. This guard \
             reads the service shutdown token's only exits by those names; if the handle's shape \
             changed, update the anchors and the scan rather than deleting the guard"
        );
    }
    assert_eq!(
        release_methods_of_graceful_shutdown(&tree.file(GRACEFUL_SHUTDOWN_MODULE).source),
        GRACEFUL_SHUTDOWN_RELEASE_API,
        "the public methods of `GracefulShutdown` compiled without `test-support` changed. A \
         method that awaits or reads the parent token (like `cancelled()` / `is_cancelled()`) \
         lets a stream behave as a bare child of it; keep such a method under \
         `#[cfg(any(test, feature = \"test-support\"))]`, or police it in \
         `scan_shutdown_tokens` before adding it here"
    );
    for file in &tree.files {
        if file.path == Path::new(GRACEFUL_SHUTDOWN_MODULE) {
            continue;
        }
        let flat = FlatCode::new(&file.code).string();
        for accessor in ["fn begun", "fn service_shutdown_token"] {
            assert!(
                !flat.contains(&format!("{accessor}(")) && !flat.contains(&format!("{accessor}<")),
                "`{}` defines `{accessor}`, a name this guard reserves for \
                 `GracefulShutdown`'s token accessors and matches on any receiver. Rename it",
                file.path.display()
            );
        }
    }
    assert_eq!(
        functions_handing_out_tokens(&handle),
        ["begin", "begun", "enroll_stream", "service_shutdown_token"],
        "the set of functions in {GRACEFUL_SHUTDOWN_MODULE} whose signature carries a \
         `CancellationToken` changed. A new one is a new way out of the handle that this guard \
         does not police: decide whether it may hand the token to a stream, and extend \
         `scan_shutdown_tokens` before adding it here"
    );

    let mut escapes = Vec::new();
    let mut observations: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut enrollments: BTreeMap<String, usize> = BTreeMap::new();
    for file in &tree.files {
        if file.path == Path::new(GRACEFUL_SHUTDOWN_MODULE) {
            continue;
        }
        let scan = scan_shutdown_tokens(
            &file.source,
            UPLOAD_WAIT_MODULES
                .iter()
                .any(|(module, _)| file.path == Path::new(module)),
        );
        let path = file.path.display().to_string();
        for (line, why) in &scan.escapes {
            escapes.push(format!(
                "  {path}:{line}: {why}\n    {}",
                file.source_line(*line)
            ));
        }
        if !scan.observations.is_empty() {
            observations.insert(path.clone(), scan.observations);
        }
        enrollments.insert(path, scan.enrollments);
    }

    // Exactly the allowed observations, per file and kind. This is also the
    // non-vacuity check: a rename of the field or the accessors would leave
    // the escape scan below matching nothing, and these sites with it.
    let allowed: BTreeMap<(String, String), usize> = ALLOWED_OBSERVATIONS
        .iter()
        .map(|(path, kind, count, _)| (((*path).to_string(), (*kind).to_string()), *count))
        .collect();
    let found: BTreeMap<(String, String), usize> = observations
        .iter()
        .flat_map(|(path, kinds)| {
            kinds
                .iter()
                .map(move |(kind, count)| ((path.clone(), kind.clone()), *count))
        })
        .collect();
    assert_eq!(
        found, allowed,
        "the production reads of the shutdown handle's shared tokens changed (left: found, \
         right: ALLOWED_OBSERVATIONS). A `.cancelled()` / `.is_cancelled()` on `begun()` or \
         `service_shutdown_token()` is NOT harmless in itself: a stream that selects on \
         `begun().cancelled()` ends at t0 with every other such stream, outside the paced \
         ending — a stream takes its ending ONLY from `enroll_stream()`. If the new read is not \
         a stream's ending (a refusal check at admission, a shutdown-sequence clock), add it to \
         `ALLOWED_OBSERVATIONS` with the reason; if a listed site went away, remove its entry"
    );

    // The upload waits are recognised by call name, so those names must take
    // a bare token only in the upload waits' modules (`subscriber_admission`'s
    // `admit` takes an `Option`, `structural_cache`'s none).
    for file in &tree.files {
        let flat = FlatCode::new(&file.code).string();
        let takers = functions_taking_a_bare_token(&flat);
        let waits: Vec<&str> = takers
            .iter()
            .map(String::as_str)
            .filter(|name| UPLOAD_WAITS.contains(name))
            .collect();
        if let Some((module, expected)) = UPLOAD_WAIT_MODULES
            .iter()
            .find(|(module, _)| file.path == Path::new(module))
        {
            assert_eq!(
                waits, *expected,
                "`{module}` no longer defines exactly its upload waits taking a bare \
                 `&CancellationToken`; update `UPLOAD_WAIT_MODULES` and `UPLOAD_WAITS`"
            );
        } else {
            assert!(
                waits.is_empty(),
                "`{}` defines {waits:?} taking a bare `&CancellationToken`. This guard accepts \
                 `begun()` handed to a call of that name as an upload wait; rename the function",
                file.path.display()
            );
        }
    }

    assert!(
        escapes.is_empty(),
        "production code outside `{GRACEFUL_SHUTDOWN_MODULE}` keeps or derives the service \
         shutdown token:\n\n{}\n\nA streaming RPC takes its token ONLY from \
         `service.shutdown.enroll_stream()`, which enters it in the paced ending's registry and \
         refuses it after t0. A child of `service_shutdown_token()` ends after the paced ending and is not \
         refused; a child of `begun()` ends at t0 with every other such stream — the reconnect \
         storm b.kgb removed. Read `begun()` / `service_shutdown_token()` only by method call, \
         on the spot, with `.cancelled()` / `.is_cancelled()` (each such read must also be \
         listed in `ALLOWED_OBSERVATIONS` with why it is not a stream's ending), or hand \
         `begun()` straight to an upload wait (`admit`, `reserve_processing`).",
        escapes.join("\n")
    );

    for handler in STREAMING_HANDLERS {
        assert_eq!(
            enrollments.get(handler).copied(),
            Some(1),
            "`{handler}` must take its shutdown token from exactly one `{ENROLL_CALL}` call in \
             production code. Without it the stream is in no paced-ending snapshot"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Guard 2 — status details
// ─────────────────────────────────────────────────────────────────────────────

/// `(line, what)` for every construction of a status detail in `source`.
///
/// - A [`DETAIL_TYPES`] name followed by `::` (an associated constructor) or
///   by `{` (a struct literal; `-> ErrorInfo {` is a return type and is
///   skipped), or renamed — by `use … as`, or as the right-hand side of a
///   `type` alias — which would hide it from this scan.
/// - An identifier starting with a [`DETAIL_BUILDERS`] name.
/// - [`DETAILS_HEADER`] named in code other than a read
///   ([`DETAILS_HEADER_READS`]): tonic writes that trailer from the status's
///   details, so code writing it by hand is building a detail.
///
/// Patterns that destructure a detail (`ErrorInfo { reason, .. } = …`) read
/// as struct literals; the failure message says to allow-list such a parse by
/// name.
fn scan_status_details(source: &str) -> Vec<(usize, String)> {
    let code = production_code(source);
    let flat = FlatCode::new(&code);
    let mut hits = Vec::new();
    for ty in DETAIL_TYPES {
        for at in flat.find_standalone(ty) {
            let after = at + ty.len();
            if flat.text.get(after).is_some_and(|c| is_ident(*c)) {
                continue;
            }
            let brace_at = if flat.starts_with_at(after, " {") {
                Some(after + 1)
            } else if flat.starts_with_at(after, "{") {
                Some(after)
            } else {
                None
            };
            let before: String = flat.text[..at].iter().collect();
            let what = if flat.starts_with_at(after, "::") {
                format!("`{ty}::…` constructs a detail")
            } else if brace_at.is_some() && !before.trim_end().ends_with("->") {
                format!("`{ty} {{ … }}` constructs a detail")
            } else if flat.starts_with_at(after, " as ") {
                format!("`{ty} as …` renames a detail type out of this scan's sight")
            } else if is_type_alias_target(&before) {
                format!("`type … = {ty}` renames a detail type out of this scan's sight")
            } else {
                continue;
            };
            hits.push((flat.line_of[at], what));
        }
    }
    for builder in DETAIL_BUILDERS {
        for at in flat.find_standalone(builder) {
            let name: String = flat.text[at..]
                .iter()
                .take_while(|c| is_ident(**c))
                .collect();
            hits.push((flat.line_of[at], format!("`{name}` builds a detail")));
        }
    }
    // The header is a string literal, which the production view empties; read
    // its lines with strings kept, restricted to production lines that carry a
    // literal (so a doc comment the strings-kept scanner misreads cannot count).
    let quoted = format!("\"{DETAILS_HEADER}\"");
    for (line, text) in needle_hits(source, DETAILS_HEADER) {
        if !code.code_at(line).contains('"') {
            continue;
        }
        let is_read = DETAILS_HEADER_READS
            .iter()
            .any(|read| text.contains(&format!("{read}{quoted}")));
        if !is_read {
            hits.push((
                line,
                format!("`{DETAILS_HEADER}` named in code other than a read"),
            ));
        }
    }
    hits.sort();
    hits.dedup();
    hits
}

/// Is a detail type named right after `before` the target of a `type` alias:
/// does the statement `before` ends in read `[pub[(…)]] type Name[<…>] =`,
/// with at most a path (`tonic_types::`) between the `=` and the type?
fn is_type_alias_target(before: &str) -> bool {
    let head = before.trim_end_matches(|c: char| is_ident(c) || c == ':');
    let Some(head) = head.trim_end().strip_suffix('=') else {
        return false;
    };
    let statement_start = head.rfind([';', '{', '}']).map_or(0, |at| at + 1);
    let statement = head[statement_start..].trim_start();
    let statement = statement
        .strip_prefix("pub")
        .map(|rest| {
            rest.trim_start()
                .strip_prefix('(')
                .and_then(|rest| rest.split_once(')').map(|(_, rest)| rest))
                .unwrap_or(rest)
        })
        .unwrap_or(statement);
    statement.trim_start().starts_with("type ")
}

/// **No production code outside `src/error.rs` builds a gRPC status detail.**
///
/// `crate::error::refusal_status` (driven by `RefusalDetail`) is the one
/// builder of the `google.rpc.ErrorInfo` / `google.rpc.RetryInfo` details the
/// service's refusals carry, so the domain, reason vocabulary, metadata keys
/// and retry hint clients parse have one definition. A second builder works
/// — its status reaches the wire, and a test of its own RPC passes — while
/// the contract quietly forks.
///
/// Parsing is allowed: naming a detail type (`Option<ErrorInfo>`), importing
/// `StatusExt` to call `get_error_details`, and reading the
/// `grpc-status-details-bin` trailer with `get` / `get_all` /
/// `contains_key`. The `MetricsLayer` backstop (`src/grpc/metrics_layer.rs`)
/// reads only `grpc-status` and `grpc-message`, and names the details trailer
/// only in comments and tests, so it needs no allow-list entry today; if a
/// parse site ever reads as a construction (a destructuring pattern), add it
/// here by file and line text, with the reason.
#[test]
fn no_status_detail_is_hand_built_outside_crate_error() {
    let tree = ProductionTree::load();
    tree.assert_visits(&[
        ERROR_MODULE,
        "src/grpc/metrics_layer.rs",
        "src/grpc/subscribe.rs",
        "src/grpc/transport.rs",
        "src/grpc/upload_admission.rs",
        "src/storage/client.rs",
    ]);

    // Non-vacuity: the scan still sees the real builder. `refusal_status`
    // constructs with `ErrorDetails::with_error_info` and attaches with
    // `Status::with_error_details`; if either needle stopped matching there,
    // it would match nothing anywhere else either.
    let builder = scan_status_details(&tree.file(ERROR_MODULE).source);
    for expected in ["`ErrorDetails::…`", "`with_error_details`"] {
        assert!(
            builder.iter().any(|(_, what)| what.starts_with(expected)),
            "the status-detail scan no longer finds {expected} in `{ERROR_MODULE}`'s production \
             code, so it would find nothing anywhere: the builder was renamed or the scan \
             broke. Found: {builder:?}"
        );
    }

    let mut violations = Vec::new();
    for file in &tree.files {
        if file.path == Path::new(ERROR_MODULE) {
            continue;
        }
        for (line, what) in scan_status_details(&file.source) {
            violations.push(format!(
                "  {}:{line}: {what}\n    {}",
                file.path.display(),
                file.source_line(line)
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "production code outside `{ERROR_MODULE}` builds a gRPC status detail:\n\n{}\n\nBuild \
         every refusal detail through `crate::error::RefusalDetail` / `refusal_status`, the one \
         builder of the `ErrorInfo` / `RetryInfo` wire shape (SDD §18.1's client contract), so \
         the domain, reasons, metadata keys and retry hint cannot drift between refusals. If \
         this site only PARSES a detail and the scan misreads it, allow-list it by name in this \
         test with the reason.",
        violations.join("\n")
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The scanners, on fixtures
// ─────────────────────────────────────────────────────────────────────────────

/// Both guards' scanners, on fixtures carrying every shape they must see
/// through and every shape they must not trip on.
///
/// The guards are filters over the tree, and a filter that silently stopped
/// filtering (or started over-filtering) changes what a green guard means
/// without changing whether it is green. The tree holds only some of these
/// shapes at any one time.
#[test]
fn guard_scanners_read_only_production_code_and_tell_observation_from_escape() {
    // ── Test-only items are removed, everything else is kept ────────────────
    let fixture = concat!(
        "fn live() { let a = 1; }\n",                      // 1
        "#[cfg(test)]\n",                                  // 2
        "mod tests {\n",                                   // 3
        "    fn t() { let b = \"}\"; let c = '}'; }\n",    // 4
        "}\n",                                             // 5
        "#[cfg(all(test, feature = \"x\"))]\n",            // 6
        "fn gated() { ErrorInfo::new(); }\n",              // 7
        "#[cfg(not(test))]\n",                             // 8
        "fn kept_not_test() {}\n",                         // 9
        "#[cfg(any(test, feature = \"test-support\"))]\n", // 10
        "fn kept_any() {}\n",                              // 11
        "struct S {\n",                                    // 12
        "    #[cfg(test)] hook: u8,\n",                    // 13
        "    after_field: u8,\n",                          // 14
        "}\n",                                             // 15
        "#[cfg(test)]\n",                                  // 16
        "mod calibration;\n",                              // 17
        "impl S {\n",                                      // 18
        "    #[cfg(test)]\n",                              // 19
        "    #[allow(dead_code)]\n",                       // 20
        "    fn seam(&self) { RetryInfo::new(); }\n",      // 21
        "    fn after_seam(&self) {}\n",                   // 22
        "}\n",                                             // 23
    );
    let code = production_code(fixture);
    let kept: Vec<usize> = code
        .lines
        .iter()
        .filter(|(_, code)| !code.is_empty())
        .map(|(line, _)| *line)
        .collect();
    assert_eq!(
        kept,
        [1, 8, 9, 10, 11, 12, 14, 15, 18, 22, 23],
        "production_code must drop exactly the items under a test-only cfg — the `mod tests` \
         block (whose string and char literals hold a `}}` that must not end it early), the \
         `all(test, …)` fn, the `#[cfg(test)]` field, the `mod calibration;` declaration and \
         the `#[cfg(test)] #[allow(…)]` method — and keep `not(test)` and `any(test, \
         feature)` items. Lines kept: {:?}",
        code.lines
    );
    assert_eq!(code.test_only_modules, ["calibration"]);
    assert_eq!(
        test_only_module_paths(Path::new("src/allocator.rs"), "calibration"),
        (
            PathBuf::from("src/allocator/calibration.rs"),
            PathBuf::from("src/allocator/calibration")
        )
    );
    assert_eq!(
        test_only_module_paths(Path::new("src/grpc/mod.rs"), "calibration"),
        (
            PathBuf::from("src/grpc/calibration.rs"),
            PathBuf::from("src/grpc/calibration")
        )
    );
    let inner =
        production_code("#![allow(dead_code)]\n#![cfg(test)]\nfn helper() { ErrorInfo::new(); }\n");
    assert!(
        inner.lines.iter().all(|(_, code)| code.is_empty()),
        "a `#![cfg(test)]` before the file's first item (other inner attributes aside) makes \
         the file wholly test-only: {:?}",
        inner.lines
    );
    let nested = production_code(concat!(
        "fn live() {}\n",                     // 1
        "mod m {\n",                          // 2
        "    #![cfg(test)]\n",                // 3
        "    fn t() { ErrorInfo::new(); }\n", // 4
        "}\n",                                // 5
        "fn after() { let a = 1; }\n",        // 6
    ));
    let kept: Vec<usize> = nested
        .lines
        .iter()
        .filter(|(_, code)| !code.is_empty())
        .map(|(line, _)| *line)
        .collect();
    assert_eq!(
        kept,
        [1, 2, 6],
        "a `#![cfg(test)]` after the file's first item removes the rest of its own block, not \
         the file: {:?}",
        nested.lines
    );

    // ── The release view drops test-feature items too ───────────────────────
    let release = concat!(
        "impl GracefulShutdown {\n",
        "    pub fn begun(&self) -> &CancellationToken { &self.shared.begun }\n",
        "    #[cfg(any(test, feature = \"test-support\"))]\n",
        "    pub async fn cancelled(&self) { self.shared.parent.cancelled().await; }\n",
        "    #[cfg(feature = \"test-fault-injection\")]\n",
        "    pub fn seam(&self) {}\n",
        "    #[cfg(not(test))]\n",
        "    pub(crate) fn kept(&self) {}\n",
        "    #[cfg(any(feature = \"test-support\", feature = \"valkey-tls\"))]\n",
        "    pub fn mixed(&self) {}\n",
        "    fn private(&self) {}\n",
        "}\n",
        "impl Default for GracefulShutdown { fn default() -> Self { Self::new() } }\n",
        "impl GracefulShutdown {\n",
        "    pub async fn awaited(&self) {}\n",
        "}\n",
    );
    assert_eq!(
        release_methods_of_graceful_shutdown(release),
        ["awaited", "begun", "kept", "mixed"],
        "the release view keeps `pub` / `pub(crate)` methods outside test-only and test-feature \
         `cfg`s (one that also names a non-test feature stays), from every `impl \
         GracefulShutdown` block, and no private or trait method"
    );

    // ── Shutdown tokens: observation vs escape ──────────────────────────────
    let fixture = concat!(
        "async fn a(svc: &Svc, shutdown_service_handle: &GracefulShutdown, path: &Path) {\n", // 1
        "    let (shutdown, _enrollment) = svc.shutdown.enroll_stream();\n",                  // 2
        "    if svc.shutdown.begun().is_cancelled() { return; }\n",                           // 3
        "    svc.shutdown.begun().cancelled().await;\n",                                      // 4
        "    shutdown_service_handle.service_shutdown_token().cancelled().await;\n",          // 5
        "    admission.admit(rpc, session_id, svc.shutdown.begun()).await;\n",                // 6
        "    admission.reserve_processing(\n",                                                // 7
        "        target,\n",                                                                  // 8
        "        svc.shutdown\n",                                                             // 9
        "            .begun(),\n",                                                            // 10
        "    );\n",                                                                           // 11
        "    let up = path.parent();\n",                                                      // 12
        "    let bg = bootstrap_shutdown.child_token();\n",                                   // 13
        "    let mine = shutdown.child_token();\n",                                           // 14
        "    let e1 = svc.shutdown.service_shutdown_token().child_token();\n",                // 15
        "    let e2 = svc.shutdown.begun().clone();\n",                                       // 16
        "    let e3 = shutdown_service_handle.service_shutdown_token().to_owned();\n",        // 17
        "    let e5 = svc.shutdown.begun();\n",                                               // 18
        "    spawn_stream(svc.shutdown.begun());\n",                                          // 19
        "    let e7 = gs.service_shutdown_token().clone();\n",                                // 20
        "    let e8 = handle.begun().child_token();\n",                                       // 21
        "    let gs = &svc.shutdown; let p = gs.service_shutdown_token(); p.child_token();\n", // 22
        "    let e9 = GracefulShutdown::begun(&gs).clone();\n",                               // 23
        "    let e10 = GracefulShutdown::service_shutdown_token;\n",                          // 24
        "    let not_it = begun_at.elapsed();\n",                                             // 25
        "    // svc.shutdown.service_shutdown_token().child_token() in a comment\n",          // 26
        "    let s = \"svc.shutdown.service_shutdown_token().child_token()\";\n",             // 27
        "}\n",                                                                                // 28
        "#[cfg(test)]\n",                                                                     // 29
        "mod tests { fn t() { let c = svc.shutdown.begun().child_token(); } }\n",             // 30
    );
    let scan = scan_shutdown_tokens(fixture, false);
    let escape_lines: Vec<usize> = scan.escapes.iter().map(|(line, _)| *line).collect();
    assert_eq!(
        escape_lines,
        [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        "escapes are the derived, cloned, bound and foreign-argument uses (lines 15-21), the \
         accessor bound through an alias of the handle (line 22), whatever the receiver is \
         called, and the accessors named as a fully qualified call or a function pointer \
         (lines 23-24) — not the observations, the upload-wait arguments (one split across \
         lines 9-10), `Path::parent`, a background task's token, a child of the stream's own \
         enrolled token, a longer identifier, a comment, a string, or a `cfg(test)` module. \
         Got {:?}",
        scan.escapes
    );
    let expected: BTreeMap<String, usize> = [
        ("begun().is_cancelled()", 1),
        ("begun().cancelled()", 1),
        ("service_shutdown_token().cancelled()", 1),
        ("begun() -> admit", 1),
        ("begun() -> reserve_processing", 1),
    ]
    .into_iter()
    .map(|(kind, count)| (kind.to_string(), count))
    .collect();
    assert_eq!(
        scan.observations, expected,
        "lines 3, 4, 5, 6 and 10 are observations, counted by kind"
    );
    assert_eq!(
        functions_taking_a_bare_token(concat!(
            "pub async fn admit(&self, rpc: UploadRpc, shutdown: &CancellationToken) -> R { } ",
            "pub async fn admit_maybe(&self, cancel: Option<&CancellationToken>) -> R { } ",
            "fn owned(token: CancellationToken) { } ",
        )),
        ["admit"],
        "a bare `&CancellationToken` parameter, not an `Option` of one or an owned token"
    );
    assert_eq!(scan.enrollments, 1, "line 2 enrolls");

    let upload_waits = concat!(
        "async fn admit(&self, shutdown: &CancellationToken) {\n", // 1
        "    if shutdown.is_cancelled() { return; }\n",            // 2
        "    tokio::select! { () = shutdown.cancelled() => {} }\n", // 3
        "    let kept = shutdown.clone();\n",                      // 4
        "    let child = shutdown\n",                              // 5
        "        .child_token();\n",                               // 6
        "    let other = session_shutdown.child_token();\n",       // 7
        "}\n",                                                     // 8
    );
    let escape_lines: Vec<usize> = scan_shutdown_tokens(upload_waits, true)
        .escapes
        .iter()
        .map(|(line, _)| *line)
        .collect();
    assert_eq!(
        escape_lines,
        [4, 6],
        "in the upload waits' module, keeping or deriving the `shutdown` parameter (the \
         `begun()` handed in) is an escape; observing it, or another token, is not"
    );
    assert!(
        scan_shutdown_tokens(upload_waits, false).escapes.is_empty(),
        "outside the upload waits' module a receiver named `shutdown` is a stream's own token"
    );

    // The commit charge (`UploadCommitCharge`) keeps the graceful-shutdown
    // handle as its `shutdown` field and reads `begun()` on the spot in
    // `hold`, handing it straight to `reserve_processing`. Only that shape is
    // an observation; keeping the token, deriving from it, copying the
    // handle, or handing `begun()` to a constructor that stores it (an
    // earlier design's `UploadCommitCharge::new(…, svc.shutdown.begun(), …)`,
    // whose enclosing call is `new`, never an upload wait) is an escape.
    let commit_charge = concat!(
        "async fn hold(&mut self, charge: u64) -> R {\n",                            // 1
        "    match self\n",                                                          // 2
        "        .reservation\n",                                                    // 3
        "        .reserve_processing(charge, self.shutdown.begun())\n",              // 4
        "        .await\n",                                                          // 5
        "    { Ok(()) => {} Err(e) => {} }\n",                                       // 6
        "    let kept = self.shutdown.begun();\n",                                   // 7
        "    let child = self.shutdown.begun().child_token();\n",                    // 8
        "    let handle = self.shutdown.clone();\n",                                 // 9
        "    let charge = UploadCommitCharge::new(&mut r, base, svc.shutdown.begun(), sid, layer);\n", // 10
        "}\n",                                                                       // 11
    );
    let scan = scan_shutdown_tokens(commit_charge, true);
    let escape_lines: Vec<usize> = scan.escapes.iter().map(|(line, _)| *line).collect();
    assert_eq!(
        escape_lines,
        [7, 8, 9, 10],
        "in the upload waits' module, `begun()` bound to a local (line 7), a child of it (line \
         8), a copy of the commit charge's `shutdown` handle (line 9) and `begun()` handed to a \
         storing constructor (line 10, enclosing call `new`) are escapes; `hold`'s rustfmt-split \
         `reserve_processing(charge, self.shutdown.begun())` (lines 2-5) is not. Got {:?}",
        scan.escapes
    );
    let expected: BTreeMap<String, usize> = [("begun() -> reserve_processing", 1)]
        .into_iter()
        .map(|(kind, count)| (kind.to_string(), count))
        .collect();
    assert_eq!(
        scan.observations, expected,
        "`hold`'s rustfmt-split read (lines 2-5) is one `begun() -> reserve_processing` \
         observation, and nothing else is"
    );

    // ── Status details: construction vs parse ───────────────────────────────
    let fixture = concat!(
        "use tonic_types::{ErrorDetails, StatusExt};\n", // 1
        "fn parse(status: &Status) -> Option<ErrorInfo> {\n", // 2
        "    let details: ErrorDetails = status.get_error_details();\n", // 3
        "    let raw = headers.get(\"grpc-status-details-bin\");\n", // 4
        "    details.error_info().cloned()\n",           // 5
        "}\n",                                           // 6
        "fn info() -> ErrorInfo {\n",                    // 7
        "    todo!()\n",                                 // 8
        "}\n",                                           // 9
        "fn build() -> Status {\n",                      // 10
        "    let d = ErrorDetails::with_error_info(\"r\", \"d\", HashMap::new());\n", // 11
        "    let i = ErrorInfo { reason: String::new(), metadata: HashMap::new() };\n", // 12
        "    let r = tonic_types::RetryInfo::new(Some(Duration::ZERO));\n", // 13
        "    headers.insert(\"grpc-status-details-bin\", value);\n", // 14
        "    Status::with_details(Code::Unavailable, \"m\", bytes);\n", // 15
        "    Status::with_error_details(Code::Unavailable, \"m\", d)\n", // 16
        "}\n",                                           // 17
        "// ErrorInfo::new() and with_error_details in a comment\n", // 18
        "const NOTE: &str = \"ErrorInfo::new() with_error_details\";\n", // 19
        "struct MyErrorInfo; fn f() { MyErrorInfo::default(); my_with_details(); }\n", // 20
        "use tonic_types::ErrorInfo as Info;\n",         // 21
        "const HEADER: &str = \"grpc-status-details-bin\";\n", // 22
        "    Status::with_error_details_and_metadata(c, \"m\", D::with_retry_info(None), md);\n", // 23
        "    s.with_error_details_vec(c, \"m\", vec![ErrorDetail::ErrorInfo(i)]);\n", // 24
        "    Status::with_details_and_metadata(c, \"m\", bytes, md);\n",              // 25
        "type D = ErrorDetails;\n",                                                   // 26
        "pub(crate) type Info2 = tonic_types::ErrorInfo;\n",                          // 27
        "fn takes(one: ErrorDetail, all: &ErrorDetails) -> Option<ErrorDetail> { None }\n", // 28
        "#[cfg(test)]\n",                                                             // 29
        "mod tests { fn t() { ErrorDetails::new(); h.insert(\"grpc-status-details-bin\", v); } }\n", // 30
    );
    let mut hits: Vec<usize> = scan_status_details(fixture)
        .into_iter()
        .map(|(line, _)| line)
        .collect();
    hits.dedup();
    assert_eq!(
        hits,
        [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27],
        "constructions are the `::` constructors, the struct literal, the header write, every \
         `with_*details*` variant (`_vec`, `_and_metadata`), the `ErrorDetail::` variant, the \
         `as` rename, the `type` aliases and the header hidden in a constant — not the import, \
         the type annotations, the return type, the header read, a comment, a string, an \
         identifier merely ending in a detail name, or a `cfg(test)` module. Got {:?}",
        scan_status_details(fixture)
    );
}
