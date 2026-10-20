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

// Guard tests: scripts/guard-tests.sh runs this whole binary by name; rename it there too.

//! Local mirror of the GitLab `schema:drift-check` CI job. Catches drift between
//! the vendored `OmniObjectDelta.fbs` and the record in `SOURCE.md` at
//! `cargo test` time, so contributors see a mismatch before pushing. Both
//! check the same three things: the whole file hashes to `**SHA256:**`; its
//! body, every line after the first `**Notice lines:**` lines (the copy's
//! license notice), hashes to `**Body SHA256:**`, upstream's body; and
//! `**Upstream SHA256:**`, the unmodified upstream file's hash, is recorded.

use std::fs;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

fn schema_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("schema")
}

/// The backtick-quoted value of SOURCE.md's one `- **<name>:** `<value>``
/// line. Strict: exactly one line may carry the `**<name>:**` marker, with a
/// single space between the marker and the opening backtick and the value
/// enclosed in a pair of ASCII backticks, so format drift fails loudly with a
/// specific complaint rather than silently picking up the wrong value. The
/// marker includes the leading `**`, so `**SHA256:**` never matches the
/// `**Upstream SHA256:**` or `**Body SHA256:**` line.
fn parse_field<'a>(source_md: &'a str, name: &str) -> &'a str {
    let marker = format!("**{name}:**");
    let mut lines = source_md.lines().filter_map(|line| {
        line.split_once(marker.as_str())
            .map(|(_, tail)| (line, tail))
    });
    let Some((line, tail)) = lines.next() else {
        panic!(
            "SOURCE.md does not contain a `{marker}` line. \
             Expected format: `- {marker} \\`<value>\\``"
        );
    };
    if let Some((other, _)) = lines.next() {
        panic!("SOURCE.md has more than one `{marker}` line: {line:?} and {other:?}");
    }
    // Expect exactly one space and then a backtick-wrapped value.
    let tail = tail.strip_prefix(' ').unwrap_or_else(|| {
        panic!(
            "SOURCE.md `{marker}` line is missing the single separator space. \
             Expected `- {marker} \\`<value>\\`` but saw: {line:?}"
        )
    });
    let tail = tail.strip_prefix('`').unwrap_or_else(|| {
        panic!(
            "SOURCE.md `{marker}` line must enclose its value in backticks. \
             Expected `- {marker} \\`<value>\\`` but saw: {line:?}"
        )
    });
    let Some(end) = tail.find('`') else {
        panic!("SOURCE.md `{marker}` line has an unterminated backtick-quoted value: {line:?}");
    };
    &tail[..end]
}

/// The SHA-256 digest of the `**<name>:**` field: exactly 64 hex characters,
/// normalized to lowercase. The previous implementation filtered all non-hex
/// chars from the trailing substring, which would have silently concatenated
/// adjacent hex runs and accepted unquoted digests.
fn parse_sha_field(source_md: &str, name: &str) -> String {
    let hex = parse_field(source_md, name);
    if hex.len() != 64 {
        panic!(
            "SOURCE.md `**{name}:**` digest must be exactly 64 hex chars, got \
             {len}: {hex:?}",
            len = hex.len()
        );
    }
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        panic!("SOURCE.md `**{name}:**` digest contained a non-hex character: {hex:?}");
    }
    hex.to_ascii_lowercase()
}

/// `**SHA256:**`: the hash of the vendored file as committed.
fn parse_declared_sha(source_md: &str) -> String {
    parse_sha_field(source_md, "SHA256")
}

/// `**Notice lines:**`: how many leading lines of the vendored file hold its
/// license notice; the body starts on the line after them. A positive decimal
/// count with no leading zero, as the CI job accepts: its `$((...))` would read
/// `014` as octal.
fn parse_notice_lines(source_md: &str) -> usize {
    let value = parse_field(source_md, "Notice lines");
    if value.is_empty() || value.starts_with('0') || !value.chars().all(|c| c.is_ascii_digit()) {
        panic!(
            "SOURCE.md `**Notice lines:**` value is not a line count (a positive decimal \
             with no leading zero): {value:?}"
        );
    }
    value
        .parse()
        .unwrap_or_else(|e| panic!("SOURCE.md `**Notice lines:**` value {value:?}: {e}"))
}

/// The bytes after the first `notice_lines` lines, as
/// `tail -n +<notice_lines + 1>` prints them (the CI job's form): empty when
/// the file has no more lines.
fn body_after_notice(bytes: &[u8], notice_lines: usize) -> &[u8] {
    let mut rest = bytes;
    for _ in 0..notice_lines {
        match rest.iter().position(|&b| b == b'\n') {
            Some(newline) => rest = &rest[newline + 1..],
            None => return &[],
        }
    }
    rest
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Panics unless the body of `schema_bytes` (after `**Notice lines:**`)
/// hashes to `**Body SHA256:**`.
fn verify_body(schema_bytes: &[u8], source_md: &str) {
    let notice_lines = parse_notice_lines(source_md);
    let declared = parse_sha_field(source_md, "Body SHA256");
    let actual = sha256_hex(body_after_notice(schema_bytes, notice_lines));
    assert_eq!(
        actual, declared,
        "\nThe vendored OmniObjectDelta.fbs body (after its {notice_lines} notice lines) does \
         not match the Body SHA256 declared in SOURCE.md.\n  \
         declared (SOURCE.md): {declared}\n  \
         actual   (file):      {actual}\n\
         Only the leading notice may differ from upstream; if upstream's schema changed, \
         follow the re-vendoring steps in oodf-ffi/schema/SOURCE.md.\n"
    );
}

// ── parse_declared_sha strictness (b.8uk) ────────────────────────────────
//
// The strict parser replaces a permissive "filter non-hex chars from the
// trailing substring" implementation. Each test below synthesizes a minimal
// SOURCE.md content that trips one of the strict-parser's guards and
// asserts a panic with a distinguishing message substring. `catch_unwind`
// is necessary because `parse_declared_sha` returns `String` on success
// and panics on failure — preserving the "format drift fails the test
// loudly" invariant called out in the docstring.

fn expect_panic_substring<F: FnOnce() + std::panic::UnwindSafe>(f: F, needle: &str) {
    let prev_hook = std::panic::take_hook();
    // Silence panic stderr spam from the expected panics.
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(prev_hook);
    let payload = result.expect_err("expected the strict parser to panic on malformed input");
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&'static str>()
                .map(|s| s.to_string())
        })
        .unwrap_or_default();
    assert!(
        msg.contains(needle),
        "expected panic message to contain {needle:?}, got {msg:?}"
    );
}

#[test]
fn parse_declared_sha_accepts_canonical_line() {
    let input =
        "- **SHA256:** `abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789`\n";
    let sha = parse_declared_sha(input);
    assert_eq!(sha.len(), 64);
    assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
    // Lowercase normalization is part of the contract — upstream allows
    // uppercase hex, we normalize to lower so comparisons against
    // Sha256::digest-formatted output are case-insensitive-safe.
    assert!(
        sha.chars().all(|c| !c.is_ascii_uppercase()),
        "declared SHA must be normalized to lowercase, got {sha:?}"
    );
}

#[test]
fn parse_declared_sha_normalizes_uppercase_to_lowercase() {
    let input =
        "- **SHA256:** `ABCDEF0123456789abcdef0123456789ABCDEF0123456789abcdef0123456789`\n";
    let sha = parse_declared_sha(input);
    assert_eq!(
        sha,
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    );
}

#[test]
fn parse_declared_sha_rejects_missing_separator_space() {
    // No space between `**SHA256:**` and the opening backtick — the strict
    // parser demands exactly one separator space.
    let input = "- **SHA256:**`abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789`\n";
    expect_panic_substring(
        || {
            parse_declared_sha(input);
        },
        "separator space",
    );
}

#[test]
fn parse_declared_sha_rejects_missing_opening_backtick() {
    // Space is present but the backtick is missing — digest is bare.
    let input = "- **SHA256:** abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789\n";
    expect_panic_substring(
        || {
            parse_declared_sha(input);
        },
        "backticks",
    );
}

#[test]
fn parse_declared_sha_rejects_unterminated_backtick() {
    // Opening backtick but no closing one. The strict parser cannot locate
    // the end of the digest so it panics with an "unterminated" message.
    let input = "- **SHA256:** `abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789\n";
    expect_panic_substring(
        || {
            parse_declared_sha(input);
        },
        "unterminated",
    );
}

#[test]
fn parse_declared_sha_rejects_too_short_digest() {
    // 63 hex chars instead of 64 — fail with the length complaint.
    let input = "- **SHA256:** `abcdef0123456789abcdef0123456789abcdef0123456789abcdef012345678`\n";
    expect_panic_substring(
        || {
            parse_declared_sha(input);
        },
        "64 hex chars",
    );
}

#[test]
fn parse_declared_sha_rejects_too_long_digest() {
    // 65 hex chars instead of 64 — same length-check path.
    let input =
        "- **SHA256:** `abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567890`\n";
    expect_panic_substring(
        || {
            parse_declared_sha(input);
        },
        "64 hex chars",
    );
}

#[test]
fn parse_declared_sha_rejects_non_hex_characters() {
    // 64 chars but the `z` is not a hex digit. The earlier permissive
    // parser would have tried to filter this out; the strict parser panics.
    let input =
        "- **SHA256:** `zbcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789`\n";
    expect_panic_substring(
        || {
            parse_declared_sha(input);
        },
        "non-hex character",
    );
}

#[test]
fn parse_declared_sha_rejects_source_md_with_no_sha_line() {
    // A SOURCE.md without any `**SHA256:**` line at all — `parse_field` finds
    // no marker line and panics with the "does not contain" complaint.
    let input = "# Not a SHA source\n\nNothing to see here.\n";
    expect_panic_substring(
        || {
            parse_declared_sha(input);
        },
        "does not contain",
    );
}

#[test]
fn parse_declared_sha_finds_line_among_other_content() {
    let input = "# Header\n\nsome prose\n\n- **SHA256:** `cafef00d0000000000000000000000000000000000000000000000000000beef`\n\nmore prose\n";
    let sha = parse_declared_sha(input);
    assert_eq!(
        sha,
        "cafef00d0000000000000000000000000000000000000000000000000000beef"
    );
}

// ── the adaptation fields: Upstream SHA256, Notice lines, Body SHA256 ────

const HEX_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HEX_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

/// A SOURCE.md Pinned Version block with the given field lines.
fn record(lines: &[&str]) -> String {
    format!(
        "## Pinned Version\n\n- **Vendored file:** `OmniObjectDelta.fbs`\n{}\n- **Vendor date:** 2026-04-16\n",
        lines.join("\n")
    )
}

#[test]
fn whole_file_sha_is_not_confused_with_the_upstream_or_body_sha() {
    // `**SHA256:**` must stay first and unique; the other two digests carry a
    // word before `SHA256`, so they never satisfy its marker.
    let both = record(&[
        &format!("- **Upstream SHA256:** `{HEX_A}`"),
        &format!("- **SHA256:** `{HEX_B}`"),
        &format!("- **Body SHA256:** `{HEX_A}`"),
    ]);
    assert_eq!(parse_declared_sha(&both), HEX_B);
    assert_eq!(parse_sha_field(&both, "Upstream SHA256"), HEX_A);
    let without = record(&[
        &format!("- **Upstream SHA256:** `{HEX_A}`"),
        &format!("- **Body SHA256:** `{HEX_A}`"),
    ]);
    expect_panic_substring(
        || {
            parse_declared_sha(&without);
        },
        "does not contain",
    );
}

#[test]
fn a_duplicated_field_fails() {
    let input = record(&[
        &format!("- **SHA256:** `{HEX_A}`"),
        &format!("- **SHA256:** `{HEX_B}`"),
    ]);
    expect_panic_substring(
        || {
            parse_declared_sha(&input);
        },
        "more than one",
    );
}

#[test]
fn a_missing_upstream_sha_fails() {
    let input = record(&[
        &format!("- **SHA256:** `{HEX_A}`"),
        "- **Notice lines:** `14`",
        &format!("- **Body SHA256:** `{HEX_B}`"),
    ]);
    expect_panic_substring(
        || {
            parse_sha_field(&input, "Upstream SHA256");
        },
        "does not contain a `**Upstream SHA256:**` line",
    );
}

#[test]
fn a_missing_body_sha_or_notice_lines_fails() {
    let no_body = record(&["- **Notice lines:** `14`"]);
    expect_panic_substring(
        || verify_body(b"", &no_body),
        "does not contain a `**Body SHA256:**` line",
    );
    let no_notice = record(&[&format!("- **Body SHA256:** `{HEX_B}`")]);
    expect_panic_substring(
        || verify_body(b"", &no_notice),
        "does not contain a `**Notice lines:**` line",
    );
}

#[test]
fn malformed_upstream_and_body_digests_fail() {
    let short = &HEX_A[..63];
    let non_hex = format!("z{}", &HEX_A[1..]);
    for (name, value, needle) in [
        ("Upstream SHA256", short, "64 hex chars"),
        ("Upstream SHA256", non_hex.as_str(), "non-hex character"),
        ("Body SHA256", short, "64 hex chars"),
        ("Body SHA256", non_hex.as_str(), "non-hex character"),
    ] {
        let input = record(&[&format!("- **{name}:** `{value}`")]);
        expect_panic_substring(
            || {
                parse_sha_field(&input, name);
            },
            needle,
        );
    }
    let unquoted = record(&[&format!("- **Upstream SHA256:** {HEX_A}")]);
    expect_panic_substring(
        || {
            parse_sha_field(&unquoted, "Upstream SHA256");
        },
        "backticks",
    );
}

#[test]
fn a_notice_lines_value_that_is_not_a_count_fails() {
    for value in ["", "fourteen", "-1", "14 lines"] {
        let input = record(&[&format!("- **Notice lines:** `{value}`")]);
        expect_panic_substring(
            || {
                parse_notice_lines(&input);
            },
            "not a line count",
        );
    }
    assert_eq!(
        parse_notice_lines(&record(&["- **Notice lines:** `14`"])),
        14
    );
}

#[test]
fn a_notice_lines_value_with_a_leading_zero_fails() {
    // The CI job's shell arithmetic reads `014` as octal 12 and fails on `08`,
    // so both checks reject a leading zero (and zero itself) rather than
    // disagree about where the body starts.
    for value in ["014", "08", "0"] {
        let input = record(&[&format!("- **Notice lines:** `{value}`")]);
        expect_panic_substring(
            || {
                parse_notice_lines(&input);
            },
            "no leading zero",
        );
    }
    assert_eq!(
        parse_notice_lines(&record(&["- **Notice lines:** `10`"])),
        10
    );
}

#[test]
fn body_after_notice_matches_tail_from_the_next_line() {
    let file = b"n1\nn2\n\nbody 1\nbody 2\n";
    assert_eq!(body_after_notice(file, 0), file);
    assert_eq!(body_after_notice(file, 2), b"\nbody 1\nbody 2\n");
    assert_eq!(body_after_notice(file, 5), b"");
    assert_eq!(body_after_notice(file, 9), b"");
    // A last line without a newline is still a line.
    assert_eq!(body_after_notice(b"n1\nlast", 1), b"last");
    assert_eq!(body_after_notice(b"n1\nlast", 2), b"");
}

#[test]
fn a_wrong_body_hash_fails() {
    let file = b"// notice\n//\n\ntable T {}\n";
    let body = sha256_hex(b"\ntable T {}\n");
    let good = record(&[
        "- **Notice lines:** `2`",
        &format!("- **Body SHA256:** `{body}`"),
    ]);
    verify_body(file, &good);
    // A changed body under the same record.
    expect_panic_substring(
        || verify_body(b"// notice\n//\n\ntable U {}\n", &good),
        "does not match the Body SHA256",
    );
    // The right body, but the notice miscounted.
    let miscounted = record(&[
        "- **Notice lines:** `1`",
        &format!("- **Body SHA256:** `{body}`"),
    ]);
    expect_panic_substring(
        || verify_body(file, &miscounted),
        "does not match the Body SHA256",
    );
}

// ── the vendored schema against its record ──────────────────────────────

fn read_vendored() -> (Vec<u8>, String) {
    let schema_path = schema_dir().join("OmniObjectDelta.fbs");
    let source_path = schema_dir().join("SOURCE.md");
    let schema_bytes = fs::read(&schema_path)
        .unwrap_or_else(|e| panic!("failed to read vendored schema at {:?}: {e}", schema_path));
    let source_md = fs::read_to_string(&source_path)
        .unwrap_or_else(|e| panic!("failed to read SOURCE.md at {:?}: {e}", source_path));
    (schema_bytes, source_md)
}

#[test]
fn vendored_schema_body_matches_declared_body_sha() {
    let (schema_bytes, source_md) = read_vendored();
    verify_body(&schema_bytes, &source_md);
}

#[test]
fn source_md_records_the_upstream_sha() {
    // The provenance record is never erased: the unmodified upstream file's
    // hash stays recorded beside the adapted copy's.
    let (_, source_md) = read_vendored();
    parse_sha_field(&source_md, "Upstream SHA256");
}

#[test]
fn vendored_schema_matches_declared_sha() {
    let (schema_bytes, source_md) = read_vendored();

    let declared_sha = parse_declared_sha(&source_md);
    let actual_sha = sha256_hex(&schema_bytes);

    assert_eq!(
        actual_sha, declared_sha,
        "\nVendored OmniObjectDelta.fbs does not match the SHA256 declared in SOURCE.md.\n  \
         declared (SOURCE.md): {declared_sha}\n  \
         actual   (file):      {actual_sha}\n\
         If you intentionally updated the schema, follow the re-vendoring steps in \
         oodf-ffi/schema/SOURCE.md and update the SHA256 there atomically.\n"
    );
}
