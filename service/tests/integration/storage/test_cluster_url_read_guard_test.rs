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

//! Source fence for the rule: **a test reads `LIVE_EDIT_TEST_CLUSTER_URL` only
//! through `live_edit::storage::test_support::test_cluster_url`.**
//!
//! CI's `test-cluster:` job runs only the tests its name filterset selects
//! (`test_support::CLUSTER_GATED_FILTERSET`). `test_cluster_url` checks the
//! calling test's name against that rule and fails the test when it misses,
//! which CI's `test:` job, running every test, turns into a red pipeline. A
//! read that goes around the helper is not checked, so a test named outside
//! the rule could read the variable, be left out of `test-cluster:`, and only
//! ever take its `SKIP:` branch.
//!
//! The fence walks every `.rs` file under the crate's `src/`, `tests/`,
//! `benches/` and `examples/`, and the other workspace members and `tools/`,
//! and finds, outside `src/storage/test_support.rs`:
//!
//! - the variable's name as a whole string literal, the way to alias it to a
//!   local constant the helper does not see;
//! - an `env::var` or `env::var_os` call whose argument names a `CLUSTER_URL`
//!   constant, the way to read it through `test_support::TEST_CLUSTER_URL_VAR`
//!   or an alias of it, read across lines when rustfmt wrapped the call.
//!
//! A string that merely mentions the variable inside a longer message is not a
//! site. Not caught: a read through a local variable bound to an alias
//! (`let v = TEST_CLUSTER_URL_VAR; env::var(v)`), or through an alias whose
//! name lacks `CLUSTER_URL` (`const URL_ENV: &str = TEST_CLUSTER_URL_VAR;`).
//! The scan excludes this file, whose synthetic inputs spell the sites it
//! looks for.
//!
//! A second fence holds every file that calls the helper to the targets CI's
//! `test-cluster:` job builds ([`CLUSTER_JOB_TARGETS`]).

use std::path::{Path, PathBuf};

use live_edit::storage::test_support::TEST_CLUSTER_URL_VAR;

/// Directories under the crate root the fence walks.
const SCANNED_ROOTS: &[&str] = &["src", "tests", "benches", "examples"];

/// The read scan also walks the other workspace members and `tools/`: their
/// tests cannot call the helper (`conformance_tests` is fenced off from
/// depending on `live-edit`), so any read of the variable there is one around it.
const OTHER_READ_ROOTS: &[&str] = &[
    "scaler",
    "oodf-ffi",
    "capabilities-proto",
    "conformance_tests",
    "tools",
];

/// The one file allowed to spell and read the variable.
const HELPER_FILE: &str = "src/storage/test_support.rs";

/// This file, relative to the crate root. Excluded from the read scan: its
/// synthetic inputs spell the very sites it looks for.
const THIS_FILE: &str = "tests/integration/storage/test_cluster_url_read_guard_test.rs";

/// The test targets CI's `test-cluster:` job builds, each with the source
/// tree whose tests it compiles. The job builds only these, so a
/// cluster-gated test anywhere else would never be built there; the
/// CI-wiring fence in `b_289_tests.rs` holds the job's command line to the
/// flags.
pub(crate) const CLUSTER_JOB_TARGETS: &[(&str, &str)] = &[
    ("--lib", "src/"),
    ("--test integration", "tests/integration/"),
    (
        "--test upload_memory_envelope",
        "tests/upload_memory_envelope/",
    ),
];

/// Under `src/` but compiled into the `live-edit` binary target, not `--lib`
/// (`src/main.rs` declares these).
const BIN_ONLY_SOURCES: &[&str] = &["src/main.rs", "src/allocator.rs", "src/allocator/"];

/// The sites in `source`: one `line: text` entry per offending line. An
/// `env::var(` or `env::var_os(` call's argument is read to its matching `)`
/// across lines, so a call rustfmt wrapped is seen whole; the site is the line
/// the call starts on.
fn read_sites(source: &str) -> Vec<String> {
    let literal = format!("\"{TEST_CLUSTER_URL_VAR}\"");
    let lines: Vec<&str> = source.lines().collect();
    let code: Vec<&str> = lines.iter().map(|line| code_of(line)).collect();
    let joined = code.join("\n");
    let mut flagged: Vec<usize> = code
        .iter()
        .enumerate()
        .filter(|(_, line)| line.contains(&literal))
        .map(|(index, _)| index)
        .collect();
    for call in ["env::var", "env::var_os"] {
        for (at, _) in joined.match_indices(call) {
            let after = joined[at + call.len()..].trim_start();
            let Some(args) = after.strip_prefix('(') else {
                continue;
            };
            if call_argument(args).contains("CLUSTER_URL") {
                flagged.push(joined[..at].matches('\n').count());
            }
        }
    }
    flagged.sort_unstable();
    flagged.dedup();
    flagged
        .into_iter()
        .map(|index| format!("{}: {}", index + 1, lines[index].trim()))
        .collect()
}

/// `line` up to a `//` comment that starts outside a string literal.
fn code_of(line: &str) -> &str {
    let mut in_string = false;
    let mut escaped = false;
    let mut previous_slash = false;
    for (at, c) in line.char_indices() {
        if in_string {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '/' if previous_slash => return &line[..at - 1],
            _ => {}
        }
        previous_slash = c == '/';
    }
    line
}

/// The text of a call's arguments, `args` starting just after its `(`, up to
/// the matching `)` (or the end, unbalanced).
fn call_argument(args: &str) -> &str {
    let mut depth = 0usize;
    for (at, c) in args.char_indices() {
        match c {
            '(' => depth += 1,
            ')' if depth == 0 => return &args[..at],
            ')' => depth -= 1,
            _ => {}
        }
    }
    args
}

/// Whether CI's `test-cluster:` job builds the target that compiles the
/// crate-relative source file `path` ([`CLUSTER_JOB_TARGETS`]).
fn built_by_cluster_job(path: &str) -> bool {
    CLUSTER_JOB_TARGETS
        .iter()
        .any(|(_, tree)| path.starts_with(tree))
        && !BIN_ONLY_SOURCES.iter().any(|bin| path.starts_with(bin))
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Generated, untracked directories the walk skips: build output of the
/// members built outside the workspace, and the fuzz harness's inputs and
/// reports (`oodf-ffi/fuzz/`), which can hold tens of thousands of files.
const SKIPPED_DIRS: &[&str] = &["target", "corpus", "artifacts", "coverage"];

/// Every `.rs` file under `dir`, recursively, outside [`SKIPPED_DIRS`].
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("the fence must be able to list {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("a directory entry must be readable").path();
        if path.is_dir()
            && path
                .file_name()
                .is_some_and(|name| SKIPPED_DIRS.iter().any(|skipped| name == *skipped))
        {
            continue;
        }
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

/// No file but `test_support.rs` spells or reads the variable, and that file
/// still does, so the fence is not passing over a helper that moved.
#[test]
fn the_cluster_url_is_read_only_through_test_cluster_url() {
    let mut files = Vec::new();
    for root in SCANNED_ROOTS.iter().chain(OTHER_READ_ROOTS) {
        let dir = crate_root().join(root);
        if dir.is_dir() {
            rust_files(&dir, &mut files);
        }
    }
    assert!(
        files.len() > 100,
        "the walk found only {} file(s); it is not reading the tree",
        files.len()
    );

    let mut offending = Vec::new();
    let mut helper_sites = 0;
    for file in &files {
        let source = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("the fence must be able to read {}: {e}", file.display()));
        let name = relative(file);
        if name == THIS_FILE {
            continue;
        }
        let sites = read_sites(&source);
        if name == HELPER_FILE {
            helper_sites = sites.len();
        } else {
            offending.extend(sites.into_iter().map(|site| format!("{name}:{site}")));
        }
    }
    assert!(
        offending.is_empty(),
        "these lines spell or read {TEST_CLUSTER_URL_VAR} outside {HELPER_FILE}. Read it with \
         `test_support::test_cluster_url()`, which holds the test's name to CI's \
         `test-cluster:` filterset, and name the variable in messages through \
         `test_support::TEST_CLUSTER_URL_VAR`:\n{}",
        offending.join("\n")
    );
    assert!(
        helper_sites >= 2,
        "{HELPER_FILE} should spell {TEST_CLUSTER_URL_VAR} once and read it in \
         `test_cluster_url`; found {helper_sites} site(s), so the helper moved and this fence \
         must follow it"
    );
}

/// Every file that calls `test_cluster_url` is compiled into a target CI's
/// `test-cluster:` job builds ([`CLUSTER_JOB_TARGETS`]).
#[test]
fn every_cluster_url_reader_is_in_a_target_the_cluster_job_builds() {
    let mut files = Vec::new();
    for root in SCANNED_ROOTS {
        let dir = crate_root().join(root);
        if dir.is_dir() {
            rust_files(&dir, &mut files);
        }
    }
    let mut readers = Vec::new();
    let mut outside = Vec::new();
    for file in &files {
        let name = relative(file);
        let source = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("the fence must be able to read {}: {e}", file.display()));
        let calls = source.lines().any(|line| {
            line.split("//")
                .next()
                .unwrap_or_default()
                .contains("test_cluster_url()")
        });
        if name == HELPER_FILE || !calls {
            continue;
        }
        if !built_by_cluster_job(&name) {
            outside.push(name.clone());
        }
        readers.push(name);
    }
    assert!(
        readers.len() > 20,
        "found only {} file(s) calling `test_cluster_url()`; the scan is not seeing the \
         cluster-gated tests",
        readers.len()
    );
    assert!(
        outside.is_empty(),
        "these files call `test_cluster_url()` but are not compiled into any target CI's \
         `test-cluster:` job builds ({CLUSTER_JOB_TARGETS:?}), so their cluster-gated tests \
         would never run on a cluster. Move the tests, or add the target to the job and to \
         CLUSTER_JOB_TARGETS:\n{}",
        outside.join("\n")
    );
}

/// The scanner, on synthetic sources: each way around the helper is a site,
/// and a message or a comment that names the variable is not.
#[test]
fn the_cluster_url_read_scanner_finds_each_way_around_the_helper() {
    let var = TEST_CLUSTER_URL_VAR;
    let cases = [
        (format!("let Ok(u) = std::env::var(\"{var}\") else {{"), 1),
        (format!("const CLUSTER_URL_VAR: &str = \"{var}\";"), 1),
        ("std::env::var(CLUSTER_URL_VAR).ok()".to_string(), 1),
        ("env::var_os(TEST_CLUSTER_URL_VAR)".to_string(), 1),
        (format!("eprintln!(\"SKIP: {var} not set\");"), 0),
        (format!("// std::env::var(\"{var}\")"), 0),
        ("std::env::var(\"TEST_VALKEY_URL\")".to_string(), 0),
        (
            "let Some(u) = test_support::test_cluster_url() else {".to_string(),
            0,
        ),
        // rustfmt's wrapped call, and an unrelated read beside it.
        (
            "let a = std::env::var(\"TEST_VALKEY_URL\");\nlet u = std::env::var(\n    CLUSTER_URL_VAR,\n)\n.ok();".to_string(),
            1,
        ),
        // Not calls: an import of `env::var`, then the sanctioned import of
        // the name for messages, before any `)`.
        (
            "use std::env::var;\nuse live_edit::storage::test_support::TEST_CLUSTER_URL_VAR;\nfn f() {}".to_string(),
            0,
        ),
        // A `//` inside a string literal is not a comment: the call's own `)`
        // still ends it, so the later name is not read as its argument.
        (
            "let u = std::env::var(\"redis://x\").ok(); let m = (CLUSTER_URL_VAR);".to_string(),
            0,
        ),
        // A nested call inside the argument.
        ("std::env::var(name(CLUSTER_URL_VAR)).ok()".to_string(), 1),
    ];
    for (source, expected) in cases {
        assert_eq!(
            read_sites(&source).len(),
            expected,
            "sites found in {source:?}"
        );
    }
    assert_eq!(
        read_sites("fn f() {}\nlet u = std::env::var(\n    CLUSTER_URL_VAR,\n);"),
        vec!["2: let u = std::env::var(".to_string()],
        "a wrapped call is reported on the line it starts on"
    );
}

/// The target classification: the library's sources and the two test targets'
/// trees count; nothing else does — the binary-only modules, other test
/// targets, examples, and `tests/common/`, which the two targets include
/// (`#[path]`) but so do targets the job does not build, so a read there
/// would also be compiled where no cluster is.
#[test]
fn the_cluster_job_target_classification_follows_the_built_targets() {
    for built in [
        "src/storage/counters.rs",
        "tests/integration/storage/gc_test.rs",
        "tests/upload_memory_envelope/main.rs",
    ] {
        assert!(built_by_cluster_job(built), "{built} is built by the job");
    }
    for not_built in [
        "src/main.rs",
        "src/allocator.rs",
        "src/allocator/structural_cache_calibration.rs",
        "tests/proptests.rs",
        "tests/common/mod.rs",
        "tests/presence_memory/main.rs",
        "examples/some_example.rs",
    ] {
        assert!(
            !built_by_cluster_job(not_built),
            "{not_built} is not counted as built by the job alone"
        );
    }
}
