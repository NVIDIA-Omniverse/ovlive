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

// Canonicalization of caller-supplied session base paths (b.uoc).
//
// `CreateSession` / `ListSessions` are the only two RPCs that take a base path
// off the wire; every other RPC reads the stored value back out of
// `session_meta` (directly, or via `SessionBasePathCache`). Canonicalizing at
// those two ingress points is therefore sufficient to make the key schema,
// `session_meta.base_path`, the session checkpoint, and the permission probe
// all agree on one spelling.
//
// A sibling of `validation.rs` rather than an addition to it: this is a
// transform (string in, string out) where the rest of that module is pure
// predicate checking, and `validation.rs` is already large.

use percent_encoding::{percent_decode_str, utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};

use crate::domain::validation::{validate_base_path, MAX_BASE_PATH_BYTES};
use crate::error::ServiceError;

/// Percent-encoding set for the *path component* of a canonical base path.
///
/// Byte-for-byte the storage API's own identity-canonicalization set
/// (`storage-core-util/src/url_path.rs`, storage-core 0.10.0 / commit
/// `2f49107`, "canonicalize resource identity URL paths"): everything outside
/// the RFC 3986 unreserved set (`A-Z a-z 0-9 - _ . ~`) plus the path separator
/// `/` is escaped as uppercase `%XX`. Matching the authority's set is the point
/// — the storage API, the metadata store, and the permission service all
/// resolve identities under this rule, so live-edit's stored spelling is the
/// same string they compare against.
///
/// **Deliberately distinct from [`crate::auth::rest::encode_resource_address`]'s
/// `PATH_ENCODE_SET`.** That set escapes `/` as well, because it packs an entire
/// resource URL into a *single* URL path segment. This one preserves `/` because
/// it is encoding a path, not a segment. The two must not be merged.
const BASE_PATH_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~')
    .remove(b'/');

/// The WHATWG URL Standard's **special schemes** — the exhaustive list from
/// §"special scheme" (`ftp`, `file`, `http`, `https`, `ws`, `wss`), which is
/// closed by the standard rather than extensible.
///
/// This list gates the host case fold in [`canonicalize_prefix`], because it is
/// the same list `url::Url` gates *its* host case fold on. `url` runs the host
/// parser (which ASCII-lowercases, and IDNA-maps, the host) only for a special
/// scheme; every other scheme goes through the **opaque-host** parser, which
/// preserves host case verbatim — `omniverse://Server.Example.COM/Path`
/// round-trips unchanged through `Url::parse` → `to_string()`, as do `omni`,
/// `s3`, `azure`, `blob` and `nucleus`. Folding host case for those schemes
/// would merge two identities the storage API and the permission service keep
/// apart, and the merge would land on the authorization probe.
///
/// Scheme-gating covers the host *case* fold and nothing else. It is not a
/// claim that a non-special authority is untouched upstream: `url` branches on
/// scheme **first** (`Parser::parse_host`, `url-2.5.8` `src/parser.rs:1038`),
/// but *both* arms then open with the same `[…]` test — `Host::parse_cow`
/// (`src/host.rs:92`) and `Host::parse_opaque_cow` (`src/host.rs:123`) alike —
/// so an IPv6 literal is parsed and re-serialized compressed and lowercase
/// under *every* scheme. See
/// [`canonicalize_base_path`] step 2 for that and the other normalizations
/// upstream applies beyond the case fold.
///
/// Hardcoded rather than derived from the `url` crate: `url` is already a direct
/// dependency, but its `SchemeType` (the type that carries this distinction) is
/// private, so there is no public API to ask. The standard's list is closed, so
/// a hardcoded copy cannot drift.
const WHATWG_SPECIAL_SCHEMES: &[&str] = &["ftp", "file", "http", "https", "ws", "wss"];

/// Canonicalize a caller-supplied `base_path` into the one spelling live-edit
/// stores, keys, hashes, and authorizes against (b.uoc).
///
/// Two RFC 3986-equivalent spellings of the same storage object (Kit's literal
/// `@` vs. an ordinary URL encoder's `%40`, `%20` vs. a raw space, `%2f` vs.
/// `%2F` vs. `/`) otherwise produce two unrelated Valkey keys and two unrelated
/// session sets, so a non-Kit client silently sees a disjoint session universe
/// on the same file.
///
/// Steps, in order:
///
/// 1. [`validate_base_path`] on the **raw** wire string. This runs first so the
///    4096-byte cap bounds the input before any decode, and so the NUL /
///    control-character class is rejected on the form the caller actually sent.
/// 2. Split into `scheme://authority` prefix, path, and `?query` / `#fragment`
///    suffix by RFC 3986 §3 rules (see [`split_prefix`] / [`split_suffix`]).
///    Two things are then normalized, and only two: the **path component**
///    (steps 3–6, strict re-encoding) and the **case of the scheme and — under
///    a WHATWG special scheme only — the host** inside the prefix
///    ([`canonicalize_prefix`], ASCII-lowercased). Userinfo, port, query and
///    fragment are recomposed exactly as received, and so is the host under any
///    non-special scheme.
///
///    That pair sits inside what the storage API's identity rule normalizes,
///    which is why the boundary sits here. `canonicalize_url_path` takes an
///    **already-parsed** `url::Url` and re-serializes it with `to_string()`,
///    and the permission service's `encode_resource_url` is likewise
///    `Url::parse` → `to_string()`. The `url` crate lowercases the **scheme**
///    while parsing, for every scheme; it lowercases the **host** only for the
///    six WHATWG special schemes ([`WHATWG_SPECIAL_SCHEMES`]), since only those
///    run the host parser — every other scheme takes the opaque-host parser,
///    which preserves host *case* byte-for-byte (case only; the invariant below
///    covers what it still rewrites). So upstream `HTTPS://Host/x` and
///    `https://host/x` are **one** identity, while `omniverse://Server/x` and
///    `omniverse://server/x` are **two**. Live-edit's fold is conditioned on
///    exactly that list: keying `HTTPS://Host/x` and `https://host/x` as two
///    session sets would be the b.uoc bug in miniature, and folding
///    `omniverse://Server/x` down to `omniverse://server/x` would be the
///    opposite and worse error — a merge of two upstream identities, landing on
///    the authorization probe.
///
///    **The invariant, not a list.** Outside the path component, everything
///    live-edit does to the scheme, the authority, and the suffix is a *strict
///    subset* of what upstream does: the scheme fold matches upstream's
///    exactly; the host fold is ASCII-case-only, applies outside `%XX` escapes,
///    and is gated to the six special schemes upstream gates its own host fold
///    on; and userinfo, port, query and fragment are passed through untouched.
///    Live-edit therefore never applies a normalization upstream does not also
///    apply, so any two spellings live-edit maps to **one** string are already
///    one identity upstream. Every remaining divergence — named below or not —
///    can consequently only **split** one upstream identity across two session
///    sets, never **merge** two upstream identities into one. That asymmetry is
///    the whole safety argument: a split costs a caller sessions it cannot see,
///    while a merge would hand one caller the session set of a path it may not
///    be authorized on, landing on the authorization probe. Reasoning from the
///    subset property rather than from an enumeration is deliberate — the
///    enumeration is not closed, and a reader who treats it as closed will read
///    a divergence this file has not named as a bug.
///
///    Known divergences, as **illustrative examples rather than a closed
///    list**: query-parameter order (`?a=1&b=2` vs. `?b=2&a=1`);
///    percent-encoding inside userinfo, the query, or the fragment
///    (`https://us er@host/x` vs. `https://us%20er@host/x` — upstream
///    serializes both as `https://us%20er@host/x`, one identity, while
///    live-edit keeps the userinfo bytes as received and keys two; `?q=%20`
///    vs. a raw-space query); **default-port elision** (`https://host:443/x`, which
///    upstream serializes as `https://host/x`, while live-edit keeps the port
///    exactly as received); **IPv4-literal normalization** under a special
///    scheme (`http://0x7f.1/x`, which upstream reserializes as
///    `http://127.0.0.1/x`); **IPv6-literal compression and lowercasing** under
///    *every* scheme, special or not (`omniverse://[2001:0DB8::1]/x`, which
///    upstream reserializes as `omniverse://[2001:db8::1]/x`, while live-edit
///    passes the literal through verbatim — see [`canonicalize_prefix`]);
///    **`file://` host elision** (`file://localhost/x` and `file:///x` are one
///    identity upstream and two here); and Unicode host case-folding / IDNA
///    (`canonicalize_prefix`, which also records why `url::Host::parse` is
///    available to close it and deliberately unused). All of these key
///    **distinct** sessions, and all of them split. storage-core keeps the
///    query and fragment in the identity it serializes, while the permission
///    layer discards both before its ACL check (`ResourceUrl::parse`), so
///    preserving them byte-for-byte matches storage-core's identity exactly and
///    is at worst finer-grained.
/// 3. Percent-decode each of the three components once; invalid UTF-8 is
///    rejected. Only the path decode feeds step 6 — the prefix and suffix
///    decodes exist purely to give step 4 the whole recomposed string.
/// 4. Reject NUL / control characters in the **decoded** form, over the whole
///    decoded string — `scheme://authority` prefix, path, *and* `?query` /
///    `#fragment` suffix alike. `Product%00Pound` and `%1B%5B31m` pass step 1
///    but decode into a NUL / an ANSI introducer that downstream systems then
///    name a blob with; the prefix and the suffix need the same bar because they
///    are stored in `session_meta.base_path`, hex-encoded into the path-index
///    key, and logged exactly like the path. This is `validate_base_path`'s
///    threat model carried one hop further. The prefix and suffix are decoded
///    for this screen **only** — neither is ever re-encoded (step 6) — so a host
///    or query whose percent-escapes are not valid UTF-8 (`https://ho%FFst/x`,
///    `?x=%FF`) is rejected rather than mangled.
/// 5. Reject `.` and `..` path segments. Decode-whole-then-split is what makes
///    this sound: `..%2Fkey` decodes to `../key` and is caught. Empty segments
///    are **not** collapsed — `a//b` is a legal, distinct Azure blob name.
/// 6. Re-encode the decoded path with [`BASE_PATH_ENCODE_SET`] and recompose,
///    splicing in the case-folded prefix from step 2 and the suffix
///    byte-for-byte as received.
/// 7. Re-check the byte cap, so it also applies to the form that actually gets
///    stored — canonicalization can inflate a string ~3×. The overflow is
///    reported against the *canonicalized* form by name, so a caller whose own
///    string was under the cap is not told otherwise.
///
/// The rule is idempotent (re-canonicalizing a canonical path is a no-op) and
/// injective on decoded input apart from scheme case, and host case under a
/// WHATWG special scheme — both of which it folds on purpose, because upstream
/// folds both as well. Upstream folds *more* than that (step 2), which is the
/// safe direction: live-edit's fold is a subset of upstream's, never a
/// superset.
///
/// Errors map to `ServiceError::InvalidArgument` → gRPC `INVALID_ARGUMENT`.
pub fn canonicalize_base_path(path: &str) -> Result<String, ServiceError> {
    // Step 1: bound and screen the raw wire form first.
    validate_base_path(path)?;

    // Step 2: peel off the prefix and the suffix, then fold the prefix's scheme
    // case (always) and its host case (special schemes only) — the one dimension
    // outside the path component that the storage API's own identity rule
    // normalizes (`Url::parse` → `to_string()` lowercases the scheme for every
    // scheme and the host only for a WHATWG special one). Everything else in the
    // prefix, and all of the suffix, rides through as received.
    let (prefix, rest) = split_prefix(path);
    let (raw_path, suffix) = split_suffix(rest);
    let canonical_prefix = canonicalize_prefix(prefix);

    // Step 3: decode the path component exactly once. The prefix and the suffix
    // are decoded too, but for *validation only* — step 6 recomposes the prefix
    // from `canonical_prefix` (the received bytes with only ASCII case folded)
    // and the suffix from `suffix`, never from these decodes. Screening the
    // received spelling rather than the folded one is equivalent: case folding
    // touches no percent-escape and can neither introduce nor remove a NUL or a
    // control character.
    let decoded = percent_decode_str(raw_path).decode_utf8().map_err(|_| {
        ServiceError::InvalidArgument(
            "base_path percent-encoding does not decode to valid UTF-8".to_owned(),
        )
    })?;
    let decoded_prefix = percent_decode_str(prefix).decode_utf8().map_err(|_| {
        ServiceError::InvalidArgument(
            "base_path percent-encoding does not decode to valid UTF-8".to_owned(),
        )
    })?;
    let decoded_suffix = percent_decode_str(suffix).decode_utf8().map_err(|_| {
        ServiceError::InvalidArgument(
            "base_path percent-encoding does not decode to valid UTF-8".to_owned(),
        )
    })?;

    // Step 4: the decoded form must clear the same NUL / control-character bar
    // the raw form did — across the whole string, prefix and suffix included.
    // All three components are stored, keyed on, and logged alike, so
    // `https://ho%00st/x` and `?x=%1B%5B31m` are the same threat as their
    // in-path spellings.
    if decoded.contains('\0') || decoded_prefix.contains('\0') || decoded_suffix.contains('\0') {
        return Err(ServiceError::InvalidArgument(
            "base_path must not decode to a value containing null bytes".to_owned(),
        ));
    }
    if decoded
        .chars()
        .chain(decoded_prefix.chars())
        .chain(decoded_suffix.chars())
        .any(char::is_control)
    {
        return Err(ServiceError::InvalidArgument(
            "base_path must not decode to a value containing control characters".to_owned(),
        ));
    }

    // Step 5: no dot segments, on the decoded form (so `..%2Fkey` is caught).
    if decoded
        .split('/')
        .any(|segment| segment == "." || segment == "..")
    {
        return Err(ServiceError::InvalidArgument(
            "base_path must not contain '.' or '..' path segments".to_owned(),
        ));
    }

    // Step 6: re-encode the path and recompose. `canonical_prefix` differs from
    // the received prefix only in ASCII case, and `suffix` is the original
    // slice, so nothing outside the path component is re-encoded or reordered.
    let encoded_path = utf8_percent_encode(&decoded, BASE_PATH_ENCODE_SET).to_string();
    let mut canonical =
        String::with_capacity(canonical_prefix.len() + encoded_path.len() + suffix.len());
    canonical.push_str(&canonical_prefix);
    canonical.push_str(&encoded_path);
    canonical.push_str(suffix);

    // Step 7: the cap applies to the stored form too. Checked here rather than
    // left to `validate_base_path` so the message names the *canonicalized*
    // form: a 2000-byte path of reserved characters is under the cap as sent and
    // over it once encoded, and "base_path exceeds 4096 byte limit" would be
    // reporting bytes the caller never wrote. The remaining `validate_base_path`
    // arms (empty / NUL / control) are unreachable on the canonical form —
    // percent-encoding removes no bytes and introduces neither — but the call
    // stays as the single authority on the rule set.
    if canonical.len() > MAX_BASE_PATH_BYTES {
        return Err(ServiceError::InvalidArgument(format!(
            "canonicalized base_path exceeds {MAX_BASE_PATH_BYTES} byte limit ({} bytes after percent-encoding)",
            canonical.len()
        )));
    }
    validate_base_path(&canonical)?;

    Ok(canonical)
}

/// Split a `scheme://authority` prefix off the front of `path`, per RFC 3986 §3.
///
/// This function is a pure **split**: it returns the prefix exactly as received
/// and normalizes nothing. The case fold that [`canonicalize_base_path`] step 2
/// applies — the scheme always, the host only under a WHATWG special scheme —
/// lives in [`canonicalize_prefix`], which runs on this return value.
///
/// Nothing else in the prefix is normalized anywhere — no default-port elision,
/// no percent-encoding rewrite — so prefixes differing only in percent-encoding
/// (`https://us er@host/x` vs. `https://us%20er@host/x`, which upstream
/// serializes alike as `https://us%20er@host/x`) key two session sets.
/// Passing those bytes through, rather than reproducing every normalization
/// upstream applies to them, is what keeps live-edit's authority handling a
/// subset of upstream's: it can leave one upstream identity split across two
/// session sets, but it can never merge two. See [`canonicalize_base_path`]
/// step 2 for the invariant and the known splits.
///
/// The prefix is percent-decoded in [`canonicalize_base_path`] step 3 solely to
/// run the NUL / control-character screen over it, never re-encoded — the
/// recomposed prefix is these exact bytes with only ASCII case folded.
///
/// Returns `(prefix, rest)`; `prefix` is empty when `path` does not match
/// `^[A-Za-z][A-Za-z0-9+.-]*://[^/?#]*` (the schemeless `/stage/...` form Kit
/// and every existing caller use). The authority runs from the `//` to the
/// first `/`, `?`, or `#`.
///
/// Hand-rolled rather than delegating to `url::Url`, deliberately: `Url::parse`
/// reinterprets `#`, `?` and `\` and silently drops TAB/LF/CR (upstream
/// storage-core hit exactly this and says so in commit `2f49107`), it rejects
/// the schemeless form outright, it applies WHATWG dot-segment *removal* before
/// this module could reject those segments, and `cannot_be_a_base` inputs like
/// `c:x` have no `set_path` at all.
fn split_prefix(path: &str) -> (&str, &str) {
    // A scheme cannot contain ':', so the first ':' is the only candidate
    // delimiter.
    let Some(colon) = path.find(':') else {
        return ("", path);
    };
    if !path[colon..].starts_with("://") {
        return ("", path);
    }

    let scheme = &path[..colon];
    let mut scheme_bytes = scheme.bytes();
    let valid_scheme = match scheme_bytes.next() {
        Some(first) => {
            first.is_ascii_alphabetic()
                && scheme_bytes
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
        }
        None => false,
    };
    if !valid_scheme {
        return ("", path);
    }

    let authority_start = colon + "://".len();
    let authority_len = path[authority_start..]
        .find(['/', '?', '#'])
        .unwrap_or(path.len() - authority_start);
    let split_at = authority_start + authority_len;
    (&path[..split_at], &path[split_at..])
}

/// ASCII-lowercase the scheme of a `scheme://authority` prefix — and, when that
/// scheme is one of the six WHATWG special schemes, its host as well — leaving
/// every other byte of the prefix exactly as received.
///
/// This is the whole of live-edit's authority normalization, and it is a
/// deliberate **strict subset** of *upstream's*: storage-core's
/// `canonicalize_url_path` runs on a parsed `url::Url` and re-serializes it, and
/// the permission service's `encode_resource_url` is `Url::parse` →
/// `to_string()`. The `url` crate lowercases the scheme while parsing, for every
/// scheme, so the scheme fold here is likewise unconditional. It lowercases the
/// **host** only for a special scheme ([`WHATWG_SPECIAL_SCHEMES`]), where the
/// host parser runs `idna::domain_to_ascii`; under any other scheme the
/// opaque-host parser keeps host case verbatim. So `HTTPS://Host/x` and
/// `https://host/x` are one identity on both of those services, while
/// `omniverse://Server/x` and `omniverse://server/x` are two — and this function
/// gates the host fold on the same list so it reproduces both halves of that.
///
/// Upstream normalizes the authority in further ways this function does not
/// reproduce; the bullets below are examples of that, **not a closed list**.
/// What makes each of them safe is not its presence on a list but the subset
/// property: this function applies no authority normalization upstream does not
/// also apply, so it can only ever leave one upstream identity **split** across
/// two session sets, never **merge** two upstream identities into one. The
/// merge is the direction that would land on the authorization probe. See
/// [`canonicalize_base_path`] step 2 for the full statement.
///
/// What is *not* folded, with the reason it is safe:
///
/// - **The host under a non-special scheme** (`omniverse`, `omni`, `s3`,
///   `azure`, `blob`, `nucleus`, anything else). Upstream preserves its case, so
///   folding it here would *merge* two upstream identities into one session set
///   — the one direction of divergence that is unsafe, because the merged
///   spelling is what the authorization probe is built from. Case is the only
///   thing upstream's opaque-host parser preserves unconditionally, though — a
///   bracketed IPv6 literal is still rewritten under a non-special scheme (see
///   the IPv6-literal bullet below), so "non-special ⇒ upstream leaves the
///   authority alone" would be the wrong reading.
/// - **userinfo** and **port** ride through byte-for-byte. Userinfo is
///   case-sensitive credential material, and a port is digits. A default port is
///   therefore kept, not elided — `https://host:443/x` stays as received where
///   `Url::to_string()` upstream would drop the `:443` (see
///   [`canonicalize_base_path`] step 2; safe-direction divergence, it can only
///   split).
/// - **Percent-escapes inside the host** keep their received spelling, hex case
///   included (`https://ho%4Ast/x` stays `%4A`). The prefix is decoded for the
///   NUL / control screen only and is never re-encoded, so rewriting an escape
///   here would be the one place this module invents authority bytes.
/// - **Non-ASCII host bytes.** The fold is ASCII-only, so a Unicode host is
///   neither case-folded nor punycoded — `HTTPS://BÜCHER.DE/x` becomes
///   `https://bÜcher.de/x`, where `idna::domain_to_ascii` upstream would reach
///   `xn--bcher-kva.de`. That leaves live-edit *finer*-grained than upstream on
///   internationalized hosts (several spellings, several session sets, for one
///   upstream identity), which is the same safe direction as the query and
///   fragment: it can split, never merge. Closing it needs the full IDNA
///   mapping rather than a case fold — which is in fact available here: `url` is
///   already a direct dependency and exposes that mapping through
///   `url::Host::parse`. It is deliberately unused. No caller sends an
///   internationalized host today, and a divergence that can only split is
///   documented and left alone rather than paid for with new parsing surface on
///   the ingress path.
/// - **IPv6-literal spelling — under every scheme, not just the special ones.**
///   `url` tests for a `[…]` host *before* it branches on scheme type:
///   `Host::parse_cow` and `Host::parse_opaque_cow` open with the same check
///   and both hand the contents to `parse_ipv6addr` (`url-2.5.8`
///   `src/host.rs:92` / `:123`), and `Display` serializes through `write_ipv6`
///   (`:214`), which emits maximal `::` compression in lowercase hex. So
///   upstream rewrites the literal on a non-special scheme too, where the case
///   fold does not run at all. Live-edit's divergence is therefore *wider* here
///   than for host case: under a special scheme it ASCII-lowercases the literal
///   and stops (`[2001:0DB8:0000:0000:0000:0000:0000:0001]` →
///   `[2001:0db8:0000:0000:0000:0000:0000:0001]`, upstream `[2001:db8::1]`),
///   and under a non-special scheme it passes the literal through verbatim
///   (`omniverse://[2001:0DB8::1]/x` stays as sent, upstream
///   `omniverse://[2001:db8::1]/x`). Both are splits, so both are safe, and
///   `url::Host::parse` would close either.
/// - **IPv4-literal spelling under a special scheme.** `Host::parse_cow` sends
///   a host that `ends_in_a_number` through `parse_ipv4addr` and reserializes
///   it in dotted-decimal, so upstream folds `http://0x7f.1/x` and
///   `http://127.0.0.1/x` into one identity where live-edit keeps two. Unlike
///   the IPv6 case this one really is special-scheme-only — `parse_ipv4addr`
///   has no counterpart in the opaque-host arm.
/// - **`file://` host elision.** Upstream drops a `localhost` host from a
///   `file` URL (`url-2.5.8` `src/parser.rs:1068`), making `file://localhost/x`
///   and `file:///x` one identity; live-edit keeps the authority as received
///   and keys two. Another split.
///
/// The userinfo delimiter is the **last** `@` in the authority (RFC 3986 §3.2.1
/// forbids a raw `@` in a host, so any earlier one belongs to the userinfo). The
/// port is what follows the closing `]` of an IPv6 literal, else what follows the
/// last `:`; a `[` with no closing `]` is malformed, and the whole of it is then
/// treated as host. That host/port split runs only on the special-scheme arm —
/// under any other scheme the whole `host:port` is copied through verbatim, so
/// there is nothing to split.
///
/// Returns `prefix` unchanged when it is empty (the schemeless form) — and is
/// idempotent, since folding ASCII case moves none of the delimiters it splits
/// on.
fn canonicalize_prefix(prefix: &str) -> String {
    // `split_prefix` yields a non-empty prefix only after matching `://`, so
    // the `else` arm covers just the schemeless case.
    let Some((scheme, authority)) = prefix.split_once("://") else {
        return prefix.to_owned();
    };

    let (userinfo, host_port) = match authority.rfind('@') {
        Some(at) => authority.split_at(at + 1),
        None => ("", authority),
    };

    // The scheme fold is unconditional: `url::Url` lowercases the scheme of
    // every scheme it parses, special or not.
    let scheme = scheme.to_ascii_lowercase();

    let mut out = String::with_capacity(prefix.len());
    out.push_str(&scheme);
    out.push_str("://");
    out.push_str(userinfo);
    if WHATWG_SPECIAL_SCHEMES.contains(&scheme.as_str()) {
        // Special scheme: `url` runs the host parser, which lowercases the host
        // and leaves the port alone. Split so the fold lands on the host only.
        let host_len = if host_port.starts_with('[') {
            match host_port.find(']') {
                // Port, if any, starts at the `:` after the closing bracket.
                Some(close) => host_port[close + 1..]
                    .find(':')
                    .map_or(host_port.len(), |offset| close + 1 + offset),
                // Unterminated IPv6 literal: not a shape with a port to find.
                None => host_port.len(),
            }
        } else {
            host_port.rfind(':').unwrap_or(host_port.len())
        };
        let (host, port) = host_port.split_at(host_len);
        out.push_str(&lowercase_host_outside_escapes(host));
        out.push_str(port);
    } else {
        // Non-special scheme: `url` runs the opaque-host parser, which preserves
        // host case. Folding here would merge identities upstream keeps apart.
        out.push_str(host_port);
    }
    out
}

/// ASCII-lowercase `host`, stepping over `%XX` triplets so their hex digits keep
/// the case they were sent with (see [`canonicalize_prefix`], whose
/// special-scheme arm is the only caller — a non-special scheme's host never
/// reaches this function).
///
/// A `%` not followed by two hex digits is not an escape introducer; it is
/// copied through and the scan resumes on the byte after it. Non-ASCII bytes are
/// untouched — `str::to_ascii_lowercase` leaves everything above `0x7F` alone —
/// so an internationalized host survives intact.
fn lowercase_host_outside_escapes(host: &str) -> String {
    let mut out = String::with_capacity(host.len());
    let mut rest = host;
    while let Some(pct) = rest.find('%') {
        let (before, tail) = rest.split_at(pct);
        out.push_str(&before.to_ascii_lowercase());
        let bytes = tail.as_bytes();
        let escape_len =
            if bytes.len() >= 3 && bytes[1].is_ascii_hexdigit() && bytes[2].is_ascii_hexdigit() {
                3
            } else {
                1
            };
        out.push_str(&tail[..escape_len]);
        rest = &tail[escape_len..];
    }
    out.push_str(&rest.to_ascii_lowercase());
    out
}

/// Split the `?query` / `#fragment` suffix off the end of `rest`.
///
/// Returns `(path_component, suffix)`, cut at the first **raw** `?` or `#`
/// (percent-encoded `%3F` / `%23` are path bytes, not delimiters). `suffix`
/// includes the delimiter and is empty when there is none.
///
/// The suffix is returned for recomposition unaltered: it is percent-decoded in
/// [`canonicalize_base_path`] step 3 solely to run the NUL / control-character
/// screen over it, never re-encoded, reordered, or otherwise normalized.
fn split_suffix(rest: &str) -> (&str, &str) {
    match rest.find(['?', '#']) {
        Some(idx) => rest.split_at(idx),
        None => (rest, ""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Assert `input` canonicalizes to `expected`, and that re-canonicalizing
    /// that output is a no-op.
    ///
    /// Idempotency is asserted on *every* accepted case rather than in one
    /// dedicated test: the fixed-point property is what lets a canonical path
    /// be re-validated on any later hop (`ListSessions` re-canonicalizes a
    /// spelling `CreateSession` already stored), so a future encode-set change
    /// that breaks it should go red at the case that broke it.
    #[track_caller]
    fn assert_canonical(input: &str, expected: &str) {
        let got = canonicalize_base_path(input)
            .unwrap_or_else(|e| panic!("{input:?} must be accepted, got error: {e:?}"));
        assert_eq!(got, expected, "canonicalize_base_path({input:?})");

        let again = canonicalize_base_path(&got).unwrap_or_else(|e| {
            panic!("canonical form {got:?} must re-canonicalize, got error: {e:?}")
        });
        assert_eq!(
            again, got,
            "canonicalization must be idempotent (input {input:?})"
        );
    }

    /// Assert `input` is rejected as `InvalidArgument` with a message
    /// containing `fragment`.
    ///
    /// The variant is the contract — it maps to gRPC `INVALID_ARGUMENT` (see
    /// [`canonicalize_rejection_maps_to_grpc_invalid_argument`]). The fragment
    /// pins only *which* rule fired, so a rewrite of one rule cannot silently
    /// start satisfying another rule's test.
    #[track_caller]
    fn assert_rejected(input: &str, fragment: &str) {
        let err = canonicalize_base_path(input).expect_err(&format!("{input:?} must be rejected"));
        match err {
            ServiceError::InvalidArgument(msg) => assert!(
                msg.contains(fragment),
                "{input:?}: expected message containing {fragment:?}, got {msg:?}"
            ),
            other => panic!("{input:?}: expected InvalidArgument, got {other:?}"),
        }
    }

    // ── The b.uoc failure mode: equivalent spellings must converge ──────────

    /// The bug as observed in dev: Kit sends a literal `@`, while
    /// `encodeURIComponent` / Python `quote` / Postman's "URL encode" send
    /// `%40`. Pre-fix the two spellings keyed two unrelated path indices on
    /// the same file, so a Postman caller's `ListSessions` returned `[]` for a
    /// path Kit had sessions on. Both must land on `%40`.
    #[test]
    fn canonicalize_at_sign_and_percent_40_converge_on_percent_40() {
        assert_canonical(
            "/main/Playground/Jdoe@example.com/new_file0908.usd",
            "/main/Playground/Jdoe%40example.com/new_file0908.usd",
        );
        assert_canonical(
            "/main/Playground/Jdoe%40example.com/new_file0908.usd",
            "/main/Playground/Jdoe%40example.com/new_file0908.usd",
        );
    }

    #[test]
    fn canonicalize_raw_space_and_percent_20_converge_on_percent_20() {
        assert_canonical("/main/Product Pound/f.usd", "/main/Product%20Pound/f.usd");
        assert_canonical("/main/Product%20Pound/f.usd", "/main/Product%20Pound/f.usd");
    }

    /// `%2f`, `%2F` and a raw `/` all name the same separator: storage-core
    /// resolves an encoded slash to a real one when it names the blob, so
    /// live-edit must collapse it too or the spellings key different sessions
    /// for one object.
    #[test]
    fn canonicalize_encoded_slash_any_hex_case_collapses_to_separator() {
        assert_canonical("/a%2fb/c", "/a/b/c");
        assert_canonical("/a%2Fb/c", "/a/b/c");
        assert_canonical("/a/b/c", "/a/b/c");
    }

    /// The storage API's own conformance vector for the escape set
    /// (storage-core-util `url_path.rs`, `URL_PATH_ESCAPE_SET`, storage-core
    /// 0.10.0 / OMPE-104009). Live-edit's canonical form must be byte-identical
    /// to the authority's, or the two sides derive different identities for one
    /// object — the divergence this fix exists to close.
    #[test]
    fn canonicalize_matches_storage_core_escape_set_conformance_vector() {
        const LOOSE: &str = "/stage/!$%&'()*+,:;=@[]^|";
        const STRICT: &str = "/stage/%21%24%25%26%27%28%29%2A%2B%2C%3A%3B%3D%40%5B%5D%5E%7C";
        assert_canonical(LOOSE, STRICT);
        assert_canonical(STRICT, STRICT);
    }

    // ── Encode set ──────────────────────────────────────────────────────────

    /// A literal `%` is not an escape introducer once decoding has run over
    /// it, so it must come back as `%25` — otherwise `/stage/100%` and
    /// `/stage/100%25` are two identities for one folder.
    #[test]
    fn canonicalize_literal_percent_is_escaped_to_percent_25() {
        // Trailing bare `%`: no two hex digits follow, so it decodes to itself.
        assert_canonical("/stage/100%", "/stage/100%25");
        // `%zz` is not a valid escape either.
        assert_canonical("/stage/%zz", "/stage/%25zz");
    }

    /// The RFC 3986 unreserved set (`A-Z a-z 0-9 - _ . ~`) plus the separator
    /// `/` survive unescaped. `~` is the one most encoders get wrong.
    #[test]
    fn canonicalize_unreserved_characters_are_preserved() {
        assert_canonical("/stage/a-b_c.d~e/AZaz09", "/stage/a-b_c.d~e/AZaz09");
    }

    #[test]
    fn canonicalize_non_ascii_round_trips_through_utf8_escapes() {
        assert_canonical(
            "/stage/ステージ",
            "/stage/%E3%82%B9%E3%83%86%E3%83%BC%E3%82%B8",
        );
        // The pre-escaped spelling is the same identity — that is the point.
        assert_canonical(
            "/stage/%E3%82%B9%E3%83%86%E3%83%BC%E3%82%B8",
            "/stage/%E3%82%B9%E3%83%86%E3%83%BC%E3%82%B8",
        );
    }

    /// Escapes are re-emitted in uppercase hex, so `%2f` and `%2F` (and any
    /// other mixed-case pair) cannot key two sessions.
    #[test]
    fn canonicalize_emits_uppercase_hex_escapes() {
        assert_canonical("/stage/a%3fb", "/stage/a%3Fb");
        assert_canonical("/stage/a%3Fb", "/stage/a%3Fb");
    }

    /// A percent-encoded `?` / `#` is a path byte, not a delimiter: the
    /// prefix/suffix split cuts on *raw* delimiters only, so neither escapes
    /// into the suffix nor decays into a raw delimiter on the way back out.
    #[test]
    fn canonicalize_encoded_delimiters_stay_in_the_path_component() {
        assert_canonical("/stage/a%3Fb%23c", "/stage/a%3Fb%23c");
    }

    /// Empty segments are legal, distinct Azure blob names, so `a//b` is not
    /// collapsed to `a/b` and a trailing `/` is not trimmed.
    #[test]
    fn canonicalize_does_not_collapse_empty_segments() {
        assert_canonical("/a//b", "/a//b");
        assert_canonical("/a/b/", "/a/b/");
    }

    // ── Prefix / suffix boundary ────────────────────────────────────────────

    #[test]
    fn canonicalize_preserves_scheme_and_authority_and_encodes_only_the_path() {
        assert_canonical(
            "https://host.example.com/main/Jdoe@example.com/f.usd",
            "https://host.example.com/main/Jdoe%40example.com/f.usd",
        );
    }

    // ── Scheme and host case folding ────────────────────────────────────────
    //
    // Live-edit folds exactly two things outside the path component: the
    // scheme (always) and the host (only under a WHATWG special scheme). Both
    // reproduce a fold `Url::parse` → `to_string()` already applies upstream,
    // which is what keeps live-edit's authority handling a strict *subset* of
    // upstream's — able to split one upstream identity across two session
    // sets, never to merge two into one. The tests below pin both halves: the
    // folds that must happen (or `HTTPS://Host/x` and `https://host/x` key two
    // session sets — b.uoc in miniature) and the folds that must NOT happen
    // (or `omniverse://Server/x` merges into `omniverse://server/x`, the
    // unsafe direction, landing on the authorization probe).

    /// The scheme fold is unconditional — `url::Url` lowercases the scheme of
    /// every scheme it parses, special or not, so live-edit does too.
    #[test]
    fn canonicalize_lowercases_the_scheme_under_every_scheme() {
        // Special scheme: scheme *and* host fold.
        assert_canonical("HTTPS://host/x", "https://host/x");
        // Non-special: the scheme still folds, on its own.
        assert_canonical("OMNIVERSE://server/x", "omniverse://server/x");
        assert_canonical("S3://bucket/key", "s3://bucket/key");
        // Mixed case, and a scheme carrying the `+`/`-`/`.` characters RFC
        // 3986 allows.
        assert_canonical("OmNi://server/x", "omni://server/x");
        assert_canonical("Git+SSH://server/x", "git+ssh://server/x");
    }

    /// The host fold runs under each of the six WHATWG special schemes,
    /// because those are exactly the schemes `url` runs its host parser for.
    #[test]
    fn canonicalize_lowercases_the_host_under_every_whatwg_special_scheme() {
        for scheme in ["FTP", "FILE", "HTTP", "HTTPS", "WS", "WSS"] {
            let lower = scheme.to_ascii_lowercase();
            assert_canonical(
                &format!("{scheme}://Host.Example.COM/x"),
                &format!("{lower}://host.example.com/x"),
            );
        }
    }

    /// The complement: under any other scheme `url` takes the opaque-host
    /// parser, which preserves host case — so folding here would *merge* two
    /// upstream identities into one session set. `ftps` is the deliberate
    /// near-miss: one character off a special scheme, and not on the list.
    #[test]
    fn canonicalize_preserves_host_case_under_a_non_special_scheme() {
        assert_canonical(
            "OMNIVERSE://Server.Example.COM/Path",
            "omniverse://Server.Example.COM/Path",
        );
        assert_canonical("S3://Bucket/key", "s3://Bucket/key");
        assert_canonical("OMNI://Server/x", "omni://Server/x");
        assert_canonical("AZURE://Account.Blob.CORE/x", "azure://Account.Blob.CORE/x");
        assert_canonical("FTPS://Host/x", "ftps://Host/x");
    }

    /// Two spellings differing only in scheme/host case are one identity
    /// upstream, so they must be one identity here — and the path component
    /// converges alongside them. This is the b.uoc convergence assertion
    /// carried across the authority boundary.
    #[test]
    fn canonicalize_cross_case_authority_spellings_converge_with_the_path() {
        const CANONICAL: &str = "https://host.example.com/Stage/Jdoe%40example.com";
        assert_canonical("HTTPS://HOST.EXAMPLE.COM/Stage/Jdoe@example.com", CANONICAL);
        assert_canonical(
            "https://host.example.com/Stage/Jdoe%40example.com",
            CANONICAL,
        );
    }

    /// The list gates the whole host fold, so an edit to it silently changes
    /// which identities merge. The WHATWG standard closes this list, so any
    /// change is a bug rather than an update — pinned as a set so a reorder is
    /// not a false failure but an addition or removal is a red test.
    #[test]
    fn whatwg_special_schemes_is_exactly_the_standard_six() {
        let mut got = WHATWG_SPECIAL_SCHEMES.to_vec();
        got.sort_unstable();
        assert_eq!(
            got,
            ["file", "ftp", "http", "https", "ws", "wss"],
            "the WHATWG special-scheme list is closed by the standard; adding a \
             scheme here starts merging two upstream identities into one session \
             set, and removing one starts splitting one identity into two"
        );
    }

    /// Userinfo is case-sensitive credential material and a port is digits, so
    /// both ride through byte-for-byte while the host beside them folds. A
    /// default port is kept, not elided.
    #[test]
    fn canonicalize_leaves_userinfo_and_port_untouched_while_folding_the_host() {
        assert_canonical("HTTPS://User@Host:443/x", "https://User@host:443/x");
        assert_canonical("HTTPS://Host:8443/x", "https://host:8443/x");
    }

    /// RFC 3986 §3.2.1 forbids a raw `@` in a host, so the userinfo delimiter
    /// is the **last** `@` — an earlier one belongs to the userinfo and must
    /// not drag part of the host into the untouched region.
    #[test]
    fn canonicalize_splits_userinfo_at_the_last_at_sign() {
        assert_canonical("HTTPS://U@S@Host:99/x", "https://U@S@host:99/x");
    }

    /// An IPv6 literal is a host, so it folds under a special scheme and is
    /// preserved under a non-special one — and the port after the closing `]`
    /// is not mistaken for part of the literal.
    #[test]
    fn canonicalize_folds_an_ipv6_literal_only_under_a_special_scheme() {
        assert_canonical("HTTP://[2001:DB8::1]:80/x", "http://[2001:db8::1]:80/x");
        assert_canonical(
            "OMNIVERSE://[2001:DB8::1]:80/x",
            "omniverse://[2001:DB8::1]:80/x",
        );
    }

    /// The bracketed-host arm of [`canonicalize_prefix`] picks the host/port
    /// cut three different ways — port present, no port, unterminated `[` —
    /// and only the first is exercised above. The no-port form is the common
    /// one on the wire and is the shape that lands on the `map_or` *default*,
    /// where a wrong bound would truncate the literal (or panic) rather than
    /// fold it; the unterminated form is the malformed shape, which has no
    /// port to find and must still fold whole rather than being cut at the
    /// missing bracket. Both fold, and neither loses a byte.
    #[test]
    fn canonicalize_folds_a_bracketed_host_with_no_port_and_when_unterminated() {
        assert_canonical("HTTPS://[2001:DB8::1]/x", "https://[2001:db8::1]/x");
        assert_canonical("HTTPS://[2001:DB8::1/x", "https://[2001:db8::1/x");
    }

    /// The prefix is decoded for the NUL / control screen only and is never
    /// re-encoded, so a `%XX` triplet in the host keeps its hex spelling —
    /// case included — while the bytes around it fold.
    #[test]
    fn canonicalize_leaves_percent_escape_hex_in_the_host_untouched() {
        assert_canonical("https://ho%4Ast/x", "https://ho%4Ast/x");
        assert_canonical("HTTPS://HO%4AST/x", "https://ho%4Ast/x");
        assert_canonical("HTTPS://Host%2FMore/x", "https://host%2Fmore/x");
    }

    /// The host fold is **ASCII-only**: no Unicode case folding and no IDNA
    /// mapping, so an internationalized host is left as sent while the ASCII
    /// bytes beside it fold. `HTTPS://BÜCHER.DE/x` therefore canonicalizes to
    /// `https://bÜcher.de/x`, where `idna::domain_to_ascii` upstream would
    /// reach `xn--bcher-kva.de` — live-edit ends up *finer*-grained than
    /// upstream, a split rather than a merge, which is the safe direction and
    /// is accepted deliberately (see [`canonicalize_prefix`]'s non-ASCII-host
    /// bullet, which records that `url::Host::parse` would close it and is
    /// deliberately unused). Pinned so that reaching for the full IDNA mapping
    /// later is a visible decision rather than a silent one.
    ///
    /// The second case additionally pins **char-boundary safety** in
    /// [`lowercase_host_outside_escapes`]: that scan slices `host` by *byte*
    /// index around each `%XX`, and here the two-byte `Ü` sits immediately
    /// before the `%`, so an off-by-one in the split would panic on a
    /// non-char-boundary rather than merely return a wrong string. The escape
    /// hex stays `%4A` while `BÜ`/`CHER.DE` around it fold.
    #[test]
    fn canonicalize_folds_a_non_ascii_host_ascii_only_without_idna() {
        assert_canonical("HTTPS://BÜCHER.DE/x", "https://bÜcher.de/x");
        assert_canonical("HTTPS://BÜ%4ACHER.DE/x", "https://bÜ%4Acher.de/x");
    }

    /// A `%` not followed by two hex digits is not an escape introducer: the
    /// scan resumes on the byte after it, so the letters that follow still
    /// fold rather than being skipped as if they were hex.
    #[test]
    fn canonicalize_bare_percent_in_the_host_is_not_an_escape_introducer() {
        assert_canonical("https://ho%zzST/x", "https://ho%zzst/x");
        assert_canonical("https://host%/x", "https://host%/x");
    }

    /// Percent-encoding *inside the authority* is left exactly as received —
    /// the fold is ASCII case only. So two authority spellings that upstream
    /// serializes alike stay two identities here: upstream renders both
    /// `https://us er@host/x` and `https://us%20er@host/x` as the latter,
    /// while live-edit keys two session sets. That is a *split*, the safe
    /// direction (module doc step 2), and it is accepted deliberately.
    ///
    /// (`user@host` vs. `user%40host` is NOT the example to use: upstream
    /// cannot parse `https://user%40host/x` at all — the `%40` decodes into
    /// the host and `idna` rejects the `@` — so the pair has no upstream
    /// identity to be split from.)
    #[test]
    fn canonicalize_leaves_authority_percent_encoding_untouched() {
        assert_canonical("https://us er@host/stage", "https://us er@host/stage");
        assert_canonical("https://us%20er@host/stage", "https://us%20er@host/stage");
        assert_ne!(
            canonicalize_base_path("https://us er@host/stage").expect("must be accepted"),
            canonicalize_base_path("https://us%20er@host/stage").expect("must be accepted"),
            "authority percent-encoding is deliberately NOT normalized; if this \
             now converges, the authority boundary moved and the SDD must move \
             with it"
        );
    }

    #[test]
    fn canonicalize_schemeless_path_is_all_path_component() {
        assert_canonical("/stage/foo", "/stage/foo");
    }

    /// `//host/path` carries no scheme, so no `scheme://authority` prefix is
    /// peeled — the whole string is the path component and its leading empty
    /// segments survive rather than being read as an authority. No scheme also
    /// means no case fold: `//Host/Path` is all path, and a path is
    /// case-sensitive.
    #[test]
    fn canonicalize_protocol_relative_form_is_treated_as_path() {
        assert_canonical("//host/path", "//host/path");
        assert_canonical("//Host/Path", "//Host/Path");
    }

    /// `c:x` is RFC 3986's `cannot_be_a_base` shape: the `:` is not followed
    /// by `//`, so no prefix is peeled and the colon is an ordinary path byte
    /// — which means the leading `C` is path, not a scheme, and does not fold.
    #[test]
    fn canonicalize_colon_without_double_slash_is_a_path_byte() {
        assert_canonical("c:x", "c%3Ax");
        assert_canonical("C:X", "C%3AX");
    }

    #[test]
    fn canonicalize_authority_only_url_with_no_path_is_preserved() {
        assert_canonical("https://host", "https://host");
    }

    #[test]
    fn canonicalize_query_before_any_slash_leaves_an_empty_path() {
        assert_canonical("https://host?q=1", "https://host?q=1");
        assert_canonical("stage?q=1", "stage?q=1");
    }

    /// Query and fragment ride through byte-for-byte while the path is
    /// re-encoded around them; the cut is at the first raw `?` or `#`, so a
    /// `#` inside a query stays inside the suffix.
    #[test]
    fn canonicalize_query_and_fragment_are_preserved_verbatim() {
        assert_canonical(
            "https://host/x y?query=1&b=2#frag",
            "https://host/x%20y?query=1&b=2#frag",
        );
        assert_canonical("https://host/x?q=%20", "https://host/x?q=%20");
        assert_canonical("https://host/x#frag", "https://host/x#frag");
    }

    /// The accepted consequence of the path-only boundary, pinned so changing
    /// it is a deliberate act: two spellings differing *only inside the query*
    /// stay distinct identities.
    ///
    /// This is a *split*, and splits are the safe direction. Upstream's
    /// permission layer discards the query outright before its ACL check
    /// (`ResourceUrl::parse`) while storage-core keeps it in the identity it
    /// serializes, so live-edit passing the query through byte-for-byte
    /// matches storage-core exactly and is at worst finer-grained than the
    /// permission layer — two session sets for what that layer would call one
    /// identity, never one session set spanning two.
    #[test]
    fn canonicalize_query_spellings_deliberately_do_not_converge() {
        let encoded = canonicalize_base_path("https://host/x?q=%20").expect("must be accepted");
        let raw = canonicalize_base_path("https://host/x?q= ").expect("must be accepted");
        assert_ne!(
            encoded, raw,
            "query-component spellings are deliberately NOT normalized \
             (canonicalize_base_path doc step 2); if this now converges, the \
             path-only boundary moved and the SDD must move with it"
        );
    }

    // ── Rejections ──────────────────────────────────────────────────────────

    #[test]
    fn canonicalize_invalid_utf8_escape_is_rejected() {
        assert_rejected("/stage/%FF", "does not decode to valid UTF-8");
        // A truncated two-byte sequence: `C3` with a non-continuation byte.
        assert_rejected("/stage/%C3%28", "does not decode to valid UTF-8");
        // The prefix and the suffix are decoded for screening too, so invalid
        // UTF-8 there is rejected rather than silently spliced back through.
        assert_rejected("https://ho%FFst/x", "does not decode to valid UTF-8");
        assert_rejected("/stage/x?q=%FF", "does not decode to valid UTF-8");
    }

    /// Decode-whole-then-split is what makes this sound: an encoded slash
    /// would otherwise smuggle `..` past a per-segment check.
    #[test]
    fn canonicalize_dot_segments_are_rejected_including_encoded_slashes() {
        assert_rejected("..%2Fkey", "'.' or '..' path segments");
        assert_rejected(".%2Fkey", "'.' or '..' path segments");
        assert_rejected("%2E%2E%2Fkey", "'.' or '..' path segments");
        assert_rejected("https://host/a/../b", "'.' or '..' path segments");
        assert_rejected("/a/./b", "'.' or '..' path segments");
    }

    /// A NUL that only appears *after* decoding clears `validate_base_path`'s
    /// raw screen, so the bar is re-applied to the decoded form — over the
    /// authority and the query/fragment as well as the path, since all three
    /// are stored in `session_meta.base_path`, hex-encoded into the path-index
    /// key, and logged alike.
    #[test]
    fn canonicalize_encoded_nul_is_rejected_in_every_component() {
        assert_rejected(
            "/stage/a%00b",
            "must not decode to a value containing null bytes",
        );
        assert_rejected(
            "/stage/x?q=%00",
            "must not decode to a value containing null bytes",
        );
        assert_rejected(
            "/stage/x#%00",
            "must not decode to a value containing null bytes",
        );
        assert_rejected(
            "https://ho%00st/x",
            "must not decode to a value containing null bytes",
        );
        // The screen is scheme-independent: it runs before the fold and does
        // not care whether the host would have been case-folded.
        assert_rejected(
            "OMNIVERSE://Ho%00st/x",
            "must not decode to a value containing null bytes",
        );
    }

    /// `%1B%5B31m` decodes to an ANSI colour introducer — the terminal-escape
    /// injection `validate_base_path` already blocks in raw form, carried one
    /// hop past the decode.
    #[test]
    fn canonicalize_encoded_control_characters_are_rejected_in_every_component() {
        assert_rejected(
            "/stage/%1B%5B31m",
            "must not decode to a value containing control characters",
        );
        assert_rejected(
            "/stage/x?q=%1B%5B31m",
            "must not decode to a value containing control characters",
        );
        assert_rejected(
            "/stage/x#%1B%5B31m",
            "must not decode to a value containing control characters",
        );
        assert_rejected(
            "https://ho%1B%5B31mst/x",
            "must not decode to a value containing control characters",
        );
    }

    /// Step 1 still runs on the raw wire form, and its messages are the
    /// un-prefixed ones — distinct from the post-decode wording above, so a
    /// reader can tell which screen fired.
    #[test]
    fn canonicalize_raw_nul_and_control_characters_are_rejected_before_decoding() {
        assert_rejected("/stage/a\u{0}b", "base_path must not contain null bytes");
        assert_rejected(
            "/stage/a\u{1b}b",
            "base_path must not contain control characters",
        );
    }

    #[test]
    fn canonicalize_empty_path_is_rejected() {
        assert_rejected("", "base_path must not be empty");
    }

    /// Over-cap input is rejected against the raw form, before any decode, and
    /// the message must NOT claim a percent-encoding overflow.
    #[test]
    fn canonicalize_oversize_raw_input_is_rejected_with_the_raw_message() {
        let raw = "a".repeat(MAX_BASE_PATH_BYTES + 1);
        let err = canonicalize_base_path(&raw).expect_err("over-cap input must be rejected");
        match err {
            ServiceError::InvalidArgument(msg) => {
                assert_eq!(
                    msg,
                    format!("base_path exceeds {MAX_BASE_PATH_BYTES} byte limit")
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// Canonicalization can inflate a string ~3×, so the cap must also apply
    /// to the form that actually gets stored — and the message must name the
    /// *canonicalized* length, or a caller whose own 2000-byte string was well
    /// under the cap is told otherwise.
    #[test]
    fn canonicalize_post_encoding_overflow_reports_the_canonicalized_length() {
        let raw = " ".repeat(2000);
        assert!(
            raw.len() <= MAX_BASE_PATH_BYTES,
            "premise: the raw input is under the cap, only the canonical form is over"
        );

        let err = canonicalize_base_path(&raw).expect_err("must overflow after encoding");
        match err {
            ServiceError::InvalidArgument(msg) => {
                assert_eq!(
                    msg,
                    format!(
                        "canonicalized base_path exceeds {MAX_BASE_PATH_BYTES} byte limit \
                         (6000 bytes after percent-encoding)"
                    ),
                    "the overflow must be reported against the canonicalized form"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// Every rejection above is one gRPC status: `INVALID_ARGUMENT`. Asserted
    /// once at the boundary rather than in each case.
    #[test]
    fn canonicalize_rejection_maps_to_grpc_invalid_argument() {
        for input in [
            "",
            "/a/../b",
            "/stage/%FF",
            "/stage/a%00b",
            "/stage/%1B%5B31m",
            &" ".repeat(2000),
        ] {
            let err =
                canonicalize_base_path(input).expect_err(&format!("{input:?} must be rejected"));
            let status = tonic::Status::from(err);
            assert_eq!(
                status.code(),
                tonic::Code::InvalidArgument,
                "{input:?} must surface as INVALID_ARGUMENT"
            );
        }
    }
}
