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

//! b.4t3 — the checkpoint lane's source-shape fence on `txn_pool` checkouts.
//!
//! # What this pins, and why it is not the storage lane's fence
//!
//! `tests/integration/storage/txn_checkout_test.rs::no_unpinned_acquire_txn_call_site_survives_under_src`
//! is a **negative** guard over all of `src/`: no module may take a `txn_pool`
//! slot through the unpinned `ValkeyClient::acquire_txn()` shim. It cannot tell
//! "migrated to `acquire_txn_for`" from "the transaction was deleted": both
//! shapes read as an absence, and the absence is what makes it pass.
//!
//! This file is the **positive** half, scoped to the lane that owns the
//! DR-restore applier. It asserts that `src/checkpoint/session.rs` still holds
//! exactly two `acquire_txn_for(` call sites — Step A's `{hex(base_path)}` pin
//! and Step B's `{session_id}` pin — so that a future edit which drops a pin,
//! collapses the two transactions into one, or reintroduces a third unpinned
//! checkout fails *here*, naming the file, rather than surviving as a green
//! cluster-only routing defect.
//!
//! # Why a source-shape guard rather than a behavioural one
//!
//! The damage is invisible on a single-node fixture and misattributed on a
//! cluster. `acquire_txn()` still probes; what it loses is the *pin*, so
//! `DISCARD` / `UNWATCH` carry no key, fred's `ClusterHash::FirstKey` policy
//! finds nothing to hash, and the router falls through to an arbitrary primary
//! (`writers.values_mut().next()`). The checkout then returns a verdict about a
//! connection the DR-restore transaction never touches: a genuinely dirty slot
//! passes its checkout and the failure lands later, on some unrelated request.
//! CI has no cluster with a fronting proxy, so nothing behavioural in this suite
//! can see the difference — the same reasoning the storage-lane fence records.
//!
//! Two things this file deliberately does **not** re-assert, because they are
//! owned elsewhere and duplicating a decision is how the copies drift:
//!
//! - *that the pins name the right slot* — `src/checkpoint/session.rs`'s
//!   `apply_session_checkpoint_step_pins_collapse_every_touched_key_to_one_slot`
//!   computes the actual `ClusterHash` of every key each step touches;
//! - *that the probe runs at all on the applier's checkouts* — same module's
//!   `apply_session_checkpoint_dirty_slot_on_each_step_is_counted_and_repaired`
//!   drives it against a live Valkey through the probe fault seam.
//!
//! No Valkey, no features: this is a filesystem read and two counts, plus two
//! fixture-only tests pinning the filters those counts run through.

use std::path::{Path, PathBuf};

/// The applier under guard, relative to the crate root.
const APPLIER_FILE: &str = "src/checkpoint/session.rs";

/// Pinned-checkout call-site form. Matches `acquire_txn_for(` — the definition
/// (`pub async fn acquire_txn_for(&self, hash_key: &str)`) also contains this
/// substring, but it lives in `src/storage/client.rs`, not in the scanned lane.
const PINNED_ACQUIRE_NEEDLE: &str = "acquire_txn_for(";

/// Unpinned-shim call-site form. Restated from the storage-lane fence rather
/// than shared: the two files are separate integration modules with separate
/// remits, and a needle that has to be edited in two places on a rename is a
/// feature here — each fence's own non-vacuity anchor catches it.
const BARE_ACQUIRE_TXN_NEEDLE: &str = ".acquire_txn()";

/// How many pinned checkouts `apply_session_checkpoint` performs.
///
/// Step A (path-index, `{hex(base_path)}`) and Step B (session-scoped,
/// `{session_id}`) are two independently-atomic transactions per SR-2.7, each
/// taking and releasing its own `txn_pool` slot. Two is therefore not a
/// magic number but the SR-2.7 step count; a change to it is a change to the
/// restore's atomicity contract and should fail this assertion loudly.
const EXPECTED_PINNED_CHECKOUTS: usize = 2;

/// First line of the applier's in-crate unit-test module.
///
/// [`EXPECTED_PINNED_CHECKOUTS`] counts the applier's PRODUCTION checkouts, and
/// `src/checkpoint/session.rs` carries ~2 500 lines of `#[cfg(test)] mod tests`
/// after them. A future in-crate test that legitimately takes its own pinned
/// checkout — driving the probe seam, say — would otherwise trip this fence
/// with a message accusing production of having grown a third transaction.
/// [`production_half`] truncates the scan here so the count means what the
/// assertion says it means.
const TEST_MODULE_MARKER: [&str; 2] = ["#[cfg(test)]", "mod tests {"];

/// Every `.rs` file under `dir`, as paths relative to `root`.
///
/// Same walk shape as the storage-lane fence (and `b_ji8_tests.rs`'s UNWATCH
/// fence before it): a NEW module in the lane that reaches for the unpinned
/// shim is exactly the regression worth catching, and a hand-maintained file
/// list would have to be remembered to see it.
fn rust_sources_under(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {} must succeed: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry must be readable").path();
        if path.is_dir() {
            rust_sources_under(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(
                path.strip_prefix(root)
                    .expect("scanned paths are under the manifest dir")
                    .to_path_buf(),
            );
        }
    }
}

/// Strip Rust comments from one line, carrying `/* … */` state across lines.
///
/// Restated from `tests/integration/storage/txn_checkout_test.rs` (whose
/// doc-comment carries the full rationale and the enumerated limits) for the
/// same reason the needles are restated: two fences with separate remits, each
/// with its own non-vacuity anchor. In short — line comments (`//`, `///`,
/// `//!`), trailing comments and `/* … */` spans are removed; string literals
/// are deliberately kept, so a needle inside a `"…"` fails loud rather than
/// passing silently; nested block comments and `r#"…"#` raw strings are not
/// modelled, both erring toward more text surviving into the match.
fn strip_comments(line: &str, in_block: &mut bool) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut code = String::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if *in_block {
            if c == '*' && chars.get(i + 1) == Some(&'/') {
                *in_block = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if in_string {
            code.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            code.push(c);
            i += 1;
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            break;
        }
        if c == '/' && chars.get(i + 1) == Some(&'*') {
            *in_block = true;
            i += 2;
            continue;
        }
        code.push(c);
        i += 1;
    }
    code
}

/// Lines of `source` whose **code** contains `needle`, as `line: text` records.
///
/// Comments are stripped for the same reason the storage-lane fence strips
/// them: the applier's own doc comment spells both forms out in prose, and
/// prose is not a call site. All three shapes go — whole-line, trailing, and
/// `/* … */` spans — because a fence that fires on a reworded comment is a red
/// pipeline with no defect behind it.
fn call_sites(source: &str, needle: &str) -> Vec<String> {
    let mut in_block = false;
    let mut sites = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        let code = strip_comments(line, &mut in_block);
        let trimmed = code.trim();
        if trimmed.contains(needle) {
            sites.push(format!("{}: {trimmed}", idx + 1));
        }
    }
    sites
}

/// The part of `source` before its `#[cfg(test)] mod tests {` block.
///
/// Fails loudly when the marker is absent rather than falling back to the whole
/// file: a silent fallback would restore exactly the over-counting this exists
/// to prevent, and would do it at the moment somebody restructured the module —
/// the moment the count is least likely to be re-derived by hand.
fn production_half<'a>(source: &'a str, file: &str) -> &'a str {
    // `split_inclusive` keeps the terminator, so the running offset is exact
    // whatever the line endings are.
    let raw: Vec<&str> = source.split_inclusive('\n').collect();
    let mut offset = 0usize;
    for (idx, line) in raw.iter().enumerate() {
        if line.trim() == TEST_MODULE_MARKER[0]
            && raw
                .get(idx + 1)
                .is_some_and(|next| next.trim().starts_with(TEST_MODULE_MARKER[1]))
        {
            return &source[..offset];
        }
        offset += line.len();
    }
    panic!(
        "`{file}` no longer contains the `{}` / `{}` marker pair this fence truncates its scan \
         at. Either the in-crate test module was renamed or removed, or the file was \
         restructured. Update TEST_MODULE_MARKER — do NOT let the scan silently cover the test \
         module, which would make the {EXPECTED_PINNED_CHECKOUTS}-checkout assertion below count \
         test-only checkouts as production ones and fail with a message blaming the applier.",
        TEST_MODULE_MARKER[0], TEST_MODULE_MARKER[1],
    )
}

/// Every `txn_pool` checkout in `src/checkpoint/` is slot-pinned, and the two
/// DR-restore pins are still there.
#[test]
fn checkpoint_lane_txn_checkouts_are_all_slot_pinned() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lane = root.join("src").join("checkpoint");
    let mut files = Vec::new();
    rust_sources_under(&root, &lane, &mut files);
    files.sort();

    // Non-vacuity, part 1: the walk reaches the applier. A lane reorganisation
    // that moved or renamed it would otherwise leave every assertion below
    // trivially satisfied by an empty scan.
    assert!(
        files.iter().any(|f| f == Path::new(APPLIER_FILE)),
        "the lane scan did not visit `{APPLIER_FILE}`; it walked {} files ({files:?}). Either \
         the DR-restore applier moved (update APPLIER_FILE) or the walk is not reaching where it \
         should — and in the latter case every assertion in this test is vacuous.",
        files.len()
    );

    // (1) The negative half, lane-scoped: no unpinned checkout anywhere under
    //     src/checkpoint/. Redundant with the storage-lane fence by design —
    //     that one is absolute over src/ — but it is the assertion that makes
    //     this file readable on its own, and a lane that loses its fence should
    //     not depend on a sibling lane's test to notice.
    let mut unpinned: Vec<String> = Vec::new();
    for file in &files {
        let source = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
        for site in call_sites(&source, BARE_ACQUIRE_TXN_NEEDLE) {
            unpinned.push(format!("  {}:{site}", file.display()));
        }
    }
    assert!(
        unpinned.is_empty(),
        "unpinned `{BARE_ACQUIRE_TXN_NEEDLE}` checkout(s) in the checkpoint lane:\n\n{}\n\nUse \
         `acquire_txn_for(&key)` with any key the transaction touches. The shim still probes, but \
         `DISCARD` / `UNWATCH` carry no key, so on a cluster fred routes them to an arbitrary \
         primary and the verdict describes a connection the transaction never runs on (b.4t3, \
         the call-site half of b.7td).",
        unpinned.join("\n")
    );

    // (2) The positive half: both DR-restore pins are still present. This is
    //     what (1) alone cannot say — deleting a checkout satisfies (1) exactly
    //     as well as pinning it does.
    //     Counted over the PRODUCTION half only — see `production_half`.
    let applier = std::fs::read_to_string(root.join(APPLIER_FILE))
        .unwrap_or_else(|e| panic!("reading {APPLIER_FILE} must succeed: {e}"));
    let pinned = call_sites(
        production_half(&applier, APPLIER_FILE),
        PINNED_ACQUIRE_NEEDLE,
    );
    assert_eq!(
        pinned.len(),
        EXPECTED_PINNED_CHECKOUTS,
        "the PRODUCTION half of `{APPLIER_FILE}` (everything before its `#[cfg(test)] mod tests`) \
         must hold exactly {EXPECTED_PINNED_CHECKOUTS} `{PINNED_ACQUIRE_NEEDLE}` \
         call sites — Step A pinned on `keys.path_index(&checkpoint.base_path)` \
         (`{{hex(base_path)}}`) and Step B on `keys.session_meta(session_id)` (`{{session_id}}`). \
         Found {found}:\n{sites}\n\nFewer means a step lost its pin or was deleted (SR-2.7 \
         requires two independently-atomic transactions); more means a third transactional \
         checkout was added and needs its own slot argument reviewed — and its own entry in the \
         `ClusterHash` collapse test in `src/checkpoint/session.rs`.",
        found = pinned.len(),
        sites = pinned.join("\n")
    );
}

/// The fence's own two filters, on a fixture carrying every shape they exist
/// for.
///
/// Both are the sort of helper whose failure is silent: a scanner that stopped
/// stripping comments turns a reworded doc comment into a red pipeline, and a
/// truncation that stopped truncating turns any future in-crate test checkout
/// into a failure that accuses production of growing a third transaction.
/// Neither shows up in the tree today, which is exactly why they are pinned on
/// a fixture.
#[test]
fn scanner_strips_comment_shapes_and_truncation_stops_at_the_test_module() {
    let fixture = r#"pub async fn step_a() {
    let guard = client.acquire_txn_for(&key).await;
}
// a whole-line comment mentioning acquire_txn_for(
/// a doc comment mentioning acquire_txn_for(
/* a block-comment span mentioning
   acquire_txn_for( and closing here */
pub async fn step_b() {
    let guard = client.acquire_txn_for(&key).await; // trailing acquire_txn_for( prose
}

#[cfg(test)]
mod tests {
    async fn an_in_crate_test_taking_its_own_pinned_checkout() {
        let guard = client.acquire_txn_for(&key).await;
    }
}
"#;

    let production = production_half(fixture, "fixture.rs");
    assert!(
        !production.contains("an_in_crate_test_taking_its_own_pinned_checkout"),
        "the truncation must cut at the `#[cfg(test)] mod tests` marker; kept: {production:?}"
    );
    let production_sites = call_sites(production, PINNED_ACQUIRE_NEEDLE);
    assert_eq!(
        production_sites.len(),
        2,
        "the production half holds two real call sites; the four comment lines between them (a \
         line comment, a doc comment, a two-line block span, and one trailing comment on an \
         otherwise-real call site) must contribute nothing. Got: {production_sites:?}"
    );

    // Without the truncation the very same fixture reads as three — the
    // over-count `production_half` exists to prevent, and the one that would
    // have fired a message blaming the applier for a test-only checkout.
    assert_eq!(
        call_sites(fixture, PINNED_ACQUIRE_NEEDLE).len(),
        3,
        "fixture precondition: the untruncated scan must see the test module's checkout too, or \
         this test is not exercising the truncation at all"
    );
}

/// A missing marker fails loudly rather than silently scanning the whole file.
#[test]
#[should_panic(expected = "no longer contains")]
fn production_half_without_the_test_module_marker_panics() {
    let _ = production_half("pub async fn step_a() {}\n", "fixture.rs");
}
