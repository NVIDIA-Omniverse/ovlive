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

// SECURITY: This module decodes JWT *payloads* for display-identity extraction
// only. It does NOT verify signatures, issuers, audiences, or expiry — token
// *validity* is delegated upstream to the user permissions service (SDD §14.2,
// §14.3), which is contacted before this client is consulted. `JwtUserinfoClient`
// is therefore invoked only after the permissions service has already accepted
// the bearer token; the decoded claims here are used purely for display
// (`UserInfo.display_name`, `id`, `email`, `username`) and MUST NOT be consumed
// for authorization decisions. Re-verifying signatures locally would duplicate
// the permissions service's contract and create a second, divergent trust
// anchor; see SDD §14.3 for the rejected alternative.
//
// That delegation holds only while `Config::authz_enabled` is true —
// `Config::from_env` rejects `userinfo off + authz off` at startup for exactly
// this reason. The one exception is the test-only hatch
// `LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS`
// (`Config::unvalidated_tokens_in_effect`): under it this client runs with NO
// upstream validator, the permissions client is the unconditional no-op, and
// any three-part JWT-shaped bearer whose payload carries a display-name claim
// is accepted as a genuine caller — and on RPCs that never resolve identity,
// any non-empty bearer string at all (the interceptor's presence check is the
// only gate left). That shape exists for the standalone conformance suite and
// nothing else.
//
// The client runs in two places: as the userinfo client itself when the
// userinfo service is disabled, and inside `UserinfoFallbackClient` when the
// userinfo service is on, authorization is on, and that service answers that
// it holds no user for the bearer (`AuthError::UserNotFound`). Both uses rest
// on the same delegation; the fallback is built only with authorization on
// (`build_userinfo_client`) for exactly this reason.
//
// The label this client produces grants nothing. It is computed only after the
// interceptor's bearer-presence check and the per-path permission check, flows
// only into stored or broadcast name fields (session creator, layer
// `created_by`, presence `user_name`), and nothing compares it for access.
// Every label the app tier builds ends in ` (app)`, so the name an app's owner
// chose cannot pass for a person's name; a human-name claim wins over it, and
// an app-only token with no app-tier claim falls through to `sub`. The owner's
// name is used only trimmed, at most 64 characters long, and only when no
// character in it can split, hide or reorder the marker; otherwise the label
// falls back to the client ID. See `app_only_label`.
//
// The `Option<String>` typing on every `JwtClaims` field is load-bearing: it
// converts a non-string claim (e.g. numeric `sub`) into `None` via serde rather
// than panicking, so the fallback chain below treats it as missing.

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::{Deserialize, Deserializer};

use crate::auth::userinfo::UserinfoClient;
use crate::auth::{AuthError, UserInfo};

/// Deserialize a JWT claim as `Option<String>`, collapsing any non-string JSON
/// value (number, bool, array, object) to `None` instead of failing the parse.
///
/// `Option<String>` alone is not enough: with the default `serde` derive, a
/// JSON `null` collapses to `None` as expected, but a number / array / object
/// fails the entire deserialization and surfaces as SR-2.4 — which would
/// reject otherwise-decodable JWTs whose `sub` or `name` happens to be the
/// wrong JSON shape. The fallback chain in step 4 expects per-field
/// "treat as missing" semantics on shape mismatches, so we route every claim
/// through this helper.
///
/// `pub(crate)` because it is the shared substrate for the *other* in-process
/// JWT-payload projection in this crate:
/// [`crate::auth::service_token::TokenClaims`], the non-secret claim set the
/// Valkey `accessTokenFile` auth diagnostic reports (b.cbe). Both want the same
/// per-field "wrong JSON shape means missing" semantics, and one copy is what
/// keeps them from drifting.
pub(crate) fn deserialize_optional_string<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    // Deserialize into a free-form `Value` first; reject the parse only if the
    // input is not valid JSON at all (which is caught by the caller's `?` on
    // the surrounding `from_slice` call). For valid JSON, project to a string
    // when the value is a JSON string; everything else (number, bool, array,
    // object, null) becomes `None`.
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::String(s) => Some(s),
        _ => None,
    })
}

/// Subset of standard / OIDC JWT claims consulted for display identity.
///
/// Every field is `Option<String>` so that a missing claim, a `null` claim, or
/// a non-string claim (number / bool / array / object — routed through
/// [`deserialize_optional_string`] so it collapses to `None` rather than
/// failing the whole parse) all converge on the same "treat as missing" path
/// through the fallback chain.
#[derive(Debug, Deserialize)]
struct JwtClaims {
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    name: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    preferred_username: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    sub: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    oid: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    email: Option<String>,
    /// Entra optional claim: `"app"` on app-only access tokens, `"user"` on
    /// user tokens when `include_user_token` is set.
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    idtyp: Option<String>,
    /// Application (client) ID of the token's client, v2.0 tokens.
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    azp: Option<String>,
    /// Application (client) ID of the token's client, v1.0 tokens.
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    appid: Option<String>,
    /// The client application's display name. Undocumented in Entra's claims
    /// reference and chosen by the app's owner, so it is only a preferred
    /// label, read on app-only tokens alone (see [`app_only_label`]).
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    app_displayname: Option<String>,
}

/// Non-empty claim value, or `None` — a present-but-empty claim counts as
/// missing everywhere in the label chain.
fn non_empty(claim: &Option<String>) -> Option<&str> {
    claim.as_deref().filter(|s| !s.is_empty())
}

/// Whether the token is an app-only (client-credentials) token.
///
/// Either signal is enough: `idtyp` equal to `app` (case-insensitive), or
/// `oid` and `sub` both present, non-empty and equal — Microsoft's documented
/// app-only check (`isAppOnly = oid != null && sub != null && oid == sub`).
/// `idtyp` is optional and comes back `"user"` on user tokens, so it is
/// compared exactly rather than tested for presence. `oid` is Entra-specific,
/// so the second test cannot fire for other issuers' tokens: a user token
/// from another issuer that carries `azp` and no `scp` keeps its `sub` label.
fn is_app_only(claims: &JwtClaims) -> bool {
    let idtyp_app = claims
        .idtyp
        .as_deref()
        .is_some_and(|t| t.eq_ignore_ascii_case("app"));
    let oid_is_sub = match (non_empty(&claims.oid), non_empty(&claims.sub)) {
        (Some(oid), Some(sub)) => oid == sub,
        _ => false,
    };
    idtyp_app || oid_is_sub
}

/// Unicode general category Cf (format characters), as ranges, listed from
/// the Unicode Character Database 15.1.0. Hand-listed because no Unicode
/// category crate is a dependency and the set is short and changes rarely; a
/// Cf character added in a later Unicode version and missing here would only
/// let that character into a cosmetic label, which still ends in ` (app)`.
const FORMAT_CHARACTER_RANGES: &[(char, char)] = &[
    ('\u{00AD}', '\u{00AD}'),
    ('\u{0600}', '\u{0605}'),
    ('\u{061C}', '\u{061C}'),
    ('\u{06DD}', '\u{06DD}'),
    ('\u{070F}', '\u{070F}'),
    ('\u{0890}', '\u{0891}'),
    ('\u{08E2}', '\u{08E2}'),
    ('\u{180E}', '\u{180E}'),
    ('\u{200B}', '\u{200F}'),
    ('\u{202A}', '\u{202E}'),
    ('\u{2060}', '\u{2064}'),
    ('\u{2066}', '\u{206F}'),
    ('\u{FEFF}', '\u{FEFF}'),
    ('\u{FFF9}', '\u{FFFB}'),
    ('\u{110BD}', '\u{110BD}'),
    ('\u{110CD}', '\u{110CD}'),
    ('\u{13430}', '\u{1343F}'),
    ('\u{1BCA0}', '\u{1BCA3}'),
    ('\u{1D173}', '\u{1D17A}'),
    ('\u{E0001}', '\u{E0001}'),
    ('\u{E0020}', '\u{E007F}'),
];

/// Whether `c` may not appear in an `app_displayname` used as a label: any
/// control character (general category Cc: C0, DEL and C1), any format
/// character (Cf, which holds the bidi controls), and the line and paragraph
/// separators U+2028 / U+2029.
fn is_unusable_in_app_name(c: char) -> bool {
    c.is_control()
        || c == '\u{2028}'
        || c == '\u{2029}'
        || FORMAT_CHARACTER_RANGES
            .iter()
            .any(|&(low, high)| (low..=high).contains(&c))
}

/// Longest `app_displayname`, in Unicode scalar values after trimming, used
/// as a label.
const MAX_APP_DISPLAYNAME_CHARS: usize = 64;

/// The `app_displayname` claim as a label, trimmed of surrounding
/// whitespace; `None` when it is absent, holds an unusable character
/// ([`is_unusable_in_app_name`]), or after trimming is empty or longer than
/// [`MAX_APP_DISPLAYNAME_CHARS`].
fn usable_app_displayname(claims: &JwtClaims) -> Option<&str> {
    let name = claims.app_displayname.as_deref()?;
    if name.chars().any(is_unusable_in_app_name) {
        return None;
    }
    let trimmed = name.trim();
    if trimmed.is_empty() || trimmed.chars().count() > MAX_APP_DISPLAYNAME_CHARS {
        return None;
    }
    Some(trimmed)
}

/// Label for an app-only token with no human-name claim; always ends in
/// ` (app)`, so the name an app's owner chose cannot pass for a person's
/// name.
///
/// Prefers `app_displayname`, then the client ID (`azp`, else `appid`), then
/// `oid`. `app_displayname` is the app owner's free choice, so it is used
/// only trimmed of surrounding whitespace, and only when the trimmed name is
/// non-empty, at most [`MAX_APP_DISPLAYNAME_CHARS`] (64) Unicode scalar
/// values long, and no character in it is unusable
/// ([`is_unusable_in_app_name`]): control characters (C0, DEL, C1), format
/// characters (Unicode Cf, which includes the bidi controls U+200E/U+200F,
/// U+202A–U+202E and U+2066–U+2069) and U+2028 / U+2029 can each hide,
/// reorder or break the line before the marker, so a client could render the
/// owner's name as a person's with the marker out of sight.
///
/// Otherwise the label falls back to the client ID. Client IDs and `oid` are
/// issuer-assigned GUIDs. `None` when none of the claims is usable.
fn app_only_label(claims: &JwtClaims) -> Option<String> {
    usable_app_displayname(claims)
        .or_else(|| non_empty(&claims.azp))
        .or_else(|| non_empty(&claims.appid))
        .or_else(|| non_empty(&claims.oid))
        .map(|label| format!("{label} (app)"))
}

/// Local JWT-payload decoder used when the external userinfo service is
/// disabled (`LIVE_EDIT_USERINFO_SERVICE_ENABLED=false`), and by
/// [`crate::auth::UserinfoFallbackClient`] when that service, with
/// authorization on, holds no user for the caller.
///
/// See the module-level `// SECURITY:` comment for the trust-delegation model
/// (decode only, no verification — validity is the permissions service's job).
pub struct JwtUserinfoClient;

impl JwtUserinfoClient {
    pub fn new() -> Self {
        Self
    }
}

impl Default for JwtUserinfoClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl UserinfoClient for JwtUserinfoClient {
    async fn get_current_user(&self, token: &str) -> Result<UserInfo, AuthError> {
        // SR-2.4 — opaque-token error string. Pinned verbatim by Task 4
        // integration tests; do not reword.
        let opaque_token_msg: String = "opaque bearer token requires the userinfo service; deploy the userinfo service or use JWT-format bearer tokens".to_owned();

        // Step 1: split on '.' and require exactly three parts (header.payload.signature).
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(AuthError::InvalidResponse(opaque_token_msg));
        }

        // Step 2: base64url-decode the middle (payload) segment.
        let decoded = URL_SAFE_NO_PAD
            .decode(parts[1])
            .map_err(|_| AuthError::InvalidResponse(opaque_token_msg.clone()))?;

        // Step 3: parse the decoded bytes as JSON into our typed claim subset.
        // Bytes that are not valid JSON surface as a serde error here and
        // collapse to the opaque-token message. Individual claims whose JSON
        // value is the wrong shape (number / bool / array / object) are
        // routed through `deserialize_optional_string` and collapse to `None`
        // per-field, so the fallback chain below treats them as missing
        // rather than failing the whole parse.
        //
        // The payload must be a JSON *object* (RFC 7519 §7.2). Type the value
        // only after that check: serde's derived impl also accepts a JSON
        // *array* and fills the claim fields POSITIONALLY (name,
        // preferred_username, sub, oid, email, idtyp, azp, appid,
        // app_displayname), so an attacker-supplied
        // `["Administrator", …]` payload would mint a display name and an `id`
        // out of an array with no claim names in it at all. This token is
        // untrusted inbound bearer material, so the guard fails closed to the
        // same opaque-token message every other undecodable shape uses.
        let payload: serde_json::Value = serde_json::from_slice(&decoded)
            .map_err(|_| AuthError::InvalidResponse(opaque_token_msg.clone()))?;
        if !payload.is_object() {
            return Err(AuthError::InvalidResponse(opaque_token_msg));
        }
        let claims: JwtClaims = serde_json::from_value(payload)
            .map_err(|_| AuthError::InvalidResponse(opaque_token_msg.clone()))?;

        // Step 4: walk the fallback chain, taking the first claim whose value
        // is `Some(s)` AND non-empty (a present-but-empty string falls
        // through, matching the missing-claim path):
        //   - human claims first: `name → preferred_username`;
        //   - else, on an app-only token carrying an app-tier claim, the
        //     ` (app)` label (`app_only_label`);
        //   - else `sub → oid`, so a user token, another issuer's token, or
        //     an app-only token with no app-tier claim keeps the label it
        //     always had.
        let display_name: Option<String> = non_empty(&claims.name)
            .or_else(|| non_empty(&claims.preferred_username))
            .map(str::to_owned)
            .or_else(|| {
                if is_app_only(&claims) {
                    app_only_label(&claims)
                } else {
                    None
                }
            })
            .or_else(|| {
                non_empty(&claims.sub)
                    .or_else(|| non_empty(&claims.oid))
                    .map(str::to_owned)
            });

        // Step 5: if the fallback chain produced nothing usable, surface the
        // missing-claim error (SR-2.5). Pinned verbatim by Task 4 integration
        // tests; do not reword.
        let display_name: String = display_name.ok_or_else(|| {
            AuthError::InvalidResponse("JWT payload contains no display-name claim (expected name / preferred_username / sub / oid)".to_owned())
        })?;

        // SR-2.3 — populate the returned UserInfo. `id` prefers `oid` (Azure AD
        // immutable user identifier) and falls back to `sub` (standard JWT
        // subject); empty string when neither is present. `email` is forwarded
        // as-is (Option<String>); `username` defaults to empty when
        // `preferred_username` is absent.
        Ok(UserInfo {
            id: claims.oid.or(claims.sub).unwrap_or_default(),
            display_name,
            email: claims.email,
            username: claims.preferred_username.unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Verbatim SR-2.4 / SR-2.5 strings asserted against the production
    // decoder. Task 4 integration tests pin the same wording; any drift here
    // must be reconciled with that surface as well.
    const SR_2_4_OPAQUE_TOKEN_MSG: &str = "opaque bearer token requires the userinfo service; deploy the userinfo service or use JWT-format bearer tokens";
    const SR_2_5_MISSING_CLAIM_MSG: &str = "JWT payload contains no display-name claim (expected name / preferred_username / sub / oid)";

    /// Build a JWT-shaped string with the given claims as the payload.
    ///
    /// The header is a fixed `{"alg":"none"}`, the signature is empty; both
    /// segments are base64url-encoded with `URL_SAFE_NO_PAD` to match the
    /// production decoder's expectations.
    fn make_jwt(claims: &serde_json::Value) -> String {
        let header = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"none\"}");
        let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(claims).unwrap());
        let signature = URL_SAFE_NO_PAD.encode(b"");
        format!("{header}.{payload}.{signature}")
    }

    // ── Happy-path tests: full fallback chain `name → preferred_username → sub → oid` ──

    #[tokio::test]
    async fn get_current_user_name_claim_present_returns_name_as_display_and_populates_userinfo() {
        // `name` is the highest-priority claim; with `oid` + `sub` + `email` +
        // `preferred_username` absent, SR-2.3 says `id` and `username` collapse
        // to empty strings while `email` stays `None`.
        let jwt = make_jwt(&json!({ "name": "Alice Example" }));

        let info = JwtUserinfoClient::new()
            .get_current_user(&jwt)
            .await
            .expect("name-only JWT should decode successfully");

        assert_eq!(info.display_name, "Alice Example");
        assert_eq!(info.id, "");
        assert_eq!(info.email, None);
        assert_eq!(info.username, "");
    }

    #[tokio::test]
    async fn get_current_user_preferred_username_fallback_returns_preferred_username_as_display() {
        // `name` absent (chain falls through one rung) → `preferred_username` wins.
        let jwt = make_jwt(&json!({
            "preferred_username": "alice@example.com",
            "sub": "should-not-win",
            "oid": "should-not-win-either",
        }));

        let info = JwtUserinfoClient::new()
            .get_current_user(&jwt)
            .await
            .expect("preferred_username-source JWT should decode successfully");

        assert_eq!(info.display_name, "alice@example.com");
    }

    #[tokio::test]
    async fn get_current_user_sub_fallback_returns_sub_as_display() {
        // `name` and `preferred_username` both empty strings → chain falls
        // through twice and lands on `sub`.
        let jwt = make_jwt(&json!({
            "name": "",
            "preferred_username": "",
            "sub": "user-1234",
            "oid": "should-not-win",
        }));

        let info = JwtUserinfoClient::new()
            .get_current_user(&jwt)
            .await
            .expect("sub-source JWT should decode successfully");

        assert_eq!(info.display_name, "user-1234");
    }

    #[tokio::test]
    async fn get_current_user_oid_fallback_returns_oid_as_display() {
        // `name` is a JSON number (non-string → None via serde), `preferred_username`
        // absent, `sub` empty → chain falls through three rungs and lands on `oid`.
        let jwt = make_jwt(&json!({
            "name": 12345,
            "sub": "",
            "oid": "00000000-0000-0000-0000-000000000001",
        }));

        let info = JwtUserinfoClient::new()
            .get_current_user(&jwt)
            .await
            .expect("oid-source JWT should decode successfully");

        assert_eq!(info.display_name, "00000000-0000-0000-0000-000000000001");
    }

    // ── Error-path tests: 1/2/4-segment, bad base64, bad JSON, missing claim ──

    #[tokio::test]
    async fn get_current_user_one_segment_token_returns_opaque_token_error() {
        let err = JwtUserinfoClient::new()
            .get_current_user("opaque-token")
            .await
            .expect_err("single-segment input must be rejected as opaque");

        assert!(
            matches!(err, AuthError::InvalidResponse(ref msg) if msg == SR_2_4_OPAQUE_TOKEN_MSG),
            "expected SR-2.4 opaque-token message, got {err:?}",
        );
    }

    #[tokio::test]
    async fn get_current_user_two_segment_token_returns_opaque_token_error() {
        let err = JwtUserinfoClient::new()
            .get_current_user("header.payload")
            .await
            .expect_err("two-segment input must be rejected as opaque");

        assert!(
            matches!(err, AuthError::InvalidResponse(ref msg) if msg == SR_2_4_OPAQUE_TOKEN_MSG),
            "expected SR-2.4 opaque-token message, got {err:?}",
        );
    }

    #[tokio::test]
    async fn get_current_user_four_segment_token_returns_opaque_token_error() {
        let err = JwtUserinfoClient::new()
            .get_current_user("a.b.c.d")
            .await
            .expect_err("four-segment input must be rejected as opaque");

        assert!(
            matches!(err, AuthError::InvalidResponse(ref msg) if msg == SR_2_4_OPAQUE_TOKEN_MSG),
            "expected SR-2.4 opaque-token message, got {err:?}",
        );
    }

    #[tokio::test]
    async fn get_current_user_payload_not_base64url_returns_opaque_token_error() {
        // `=` is invalid under `URL_SAFE_NO_PAD` (padding rejected); `!` is also
        // not part of the URL-safe alphabet. Either guarantees the base64
        // decode step fails.
        let token = "header.bad!=payload.signature";

        let err = JwtUserinfoClient::new()
            .get_current_user(token)
            .await
            .expect_err("non-base64url payload must be rejected as opaque");

        assert!(
            matches!(err, AuthError::InvalidResponse(ref msg) if msg == SR_2_4_OPAQUE_TOKEN_MSG),
            "expected SR-2.4 opaque-token message, got {err:?}",
        );
    }

    #[tokio::test]
    async fn get_current_user_payload_not_json_returns_opaque_token_error() {
        // Middle segment decodes cleanly to `not json bytes` but is not valid
        // JSON; serde_json::from_slice must surface as the opaque-token error.
        let header = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"none\"}");
        let payload = URL_SAFE_NO_PAD.encode(b"not json bytes");
        let signature = URL_SAFE_NO_PAD.encode(b"");
        let token = format!("{header}.{payload}.{signature}");

        let err = JwtUserinfoClient::new()
            .get_current_user(&token)
            .await
            .expect_err("non-JSON payload must be rejected as opaque");

        assert!(
            matches!(err, AuthError::InvalidResponse(ref msg) if msg == SR_2_4_OPAQUE_TOKEN_MSG),
            "expected SR-2.4 opaque-token message, got {err:?}",
        );
    }

    #[tokio::test]
    async fn get_current_user_json_array_payload_returns_opaque_token_error() {
        // Valid JSON, wrong top-level shape — and the ONE wrong shape that does
        // not fail on its own. RFC 7519 §7.2 makes the payload a JSON object;
        // without the `payload.is_object()` guard, serde's derived impl also
        // accepts a *sequence* and fills `JwtClaims` POSITIONALLY (`name`,
        // `preferred_username`, `sub`, `oid`, `email`, then the app-label
        // claims), so this payload — which
        // carries no claim NAMES at all — would mint `display_name =
        // "Administrator"` and `id = "victim-oid"` for an attacker-supplied
        // token. Every other undecodable shape already fails closed to the
        // SR-2.4 message, and this must too.
        //
        // If this test reds, the fix is to restore the object guard, NOT to
        // accept the positional mapping.
        let jwt = make_jwt(&json!([
            "Administrator",
            "admin@example.com",
            "victim-sub",
            "victim-oid",
            "admin@example.com"
        ]));

        let err = JwtUserinfoClient::new()
            .get_current_user(&jwt)
            .await
            .expect_err("a JSON array payload must never mint a display identity");

        assert!(
            matches!(err, AuthError::InvalidResponse(ref msg) if msg == SR_2_4_OPAQUE_TOKEN_MSG),
            "expected SR-2.4 opaque-token message, got {err:?}",
        );
    }

    #[tokio::test]
    async fn get_current_user_json_scalar_payload_returns_opaque_token_error() {
        // The sibling non-object shapes. A scalar has no struct interpretation
        // at all, so it would fail even without the guard — pinned alongside
        // the array case so "the payload must be a JSON object" is asserted as
        // one rule rather than one accident.
        for payload in [json!("just-a-string"), json!(42), json!(null)] {
            let jwt = make_jwt(&payload);

            let err = JwtUserinfoClient::new()
                .get_current_user(&jwt)
                .await
                .expect_err("a non-object JWT payload must be rejected as opaque");

            assert!(
                matches!(err, AuthError::InvalidResponse(ref msg) if msg == SR_2_4_OPAQUE_TOKEN_MSG),
                "expected SR-2.4 opaque-token message for payload {payload}, got {err:?}",
            );
        }
    }

    #[tokio::test]
    async fn get_current_user_valid_json_with_no_display_claim_returns_missing_claim_error() {
        // Valid JSON, but none of name/preferred_username/sub/oid is present.
        let jwt = make_jwt(&json!({ "email": "alice@example.com", "aud": "live-edit" }));

        let err = JwtUserinfoClient::new()
            .get_current_user(&jwt)
            .await
            .expect_err("JWT with no display-name claim must surface SR-2.5");

        assert!(
            matches!(err, AuthError::InvalidResponse(ref msg) if msg == SR_2_5_MISSING_CLAIM_MSG),
            "expected SR-2.5 missing-claim message, got {err:?}",
        );
    }

    #[tokio::test]
    async fn get_current_user_name_empty_string_falls_through_to_preferred_username() {
        // `name` present but empty → fallback chain treats as missing and
        // falls through to `preferred_username`.
        let jwt = make_jwt(&json!({
            "name": "",
            "preferred_username": "alice@example.com",
        }));

        let info = JwtUserinfoClient::new()
            .get_current_user(&jwt)
            .await
            .expect("empty `name` should fall through to preferred_username");

        assert_eq!(info.display_name, "alice@example.com");
    }

    #[tokio::test]
    async fn get_current_user_name_non_string_falls_through_to_preferred_username() {
        // `name` is a JSON number → serde collapses the field to None for the
        // `Option<String>` typing, and the fallback chain falls through.
        let jwt = make_jwt(&json!({
            "name": 42,
            "preferred_username": "alice@example.com",
        }));

        let info = JwtUserinfoClient::new()
            .get_current_user(&jwt)
            .await
            .expect("non-string `name` should fall through to preferred_username");

        assert_eq!(info.display_name, "alice@example.com");
    }

    // ── App-only tier: `name → preferred_username → <app label> → sub → oid` ──
    //
    // A token is app-only when `idtyp` is `app` (case-insensitive) or `oid` and
    // `sub` are equal and non-empty. With no human claim, such a token's label
    // is `app_displayname`, else the client ID (`azp`, else `appid`), else
    // `oid` — always followed by ` (app)`. Every other token keeps `sub → oid`.

    /// Entra object id of the app's service principal (invented).
    const APP_OID: &str = "11111111-2222-3333-4444-555555555555";
    /// Application (client) ID of the app registration (invented).
    const CLIENT_ID: &str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
    /// A user's `sub` — pairwise and distinct from any `oid`.
    const USER_SUB: &str = "user-sub-pairwise";

    async fn display_name_of(claims: serde_json::Value) -> String {
        JwtUserinfoClient::new()
            .get_current_user(&make_jwt(&claims))
            .await
            .unwrap_or_else(|e| panic!("{claims} must decode, got {e:?}"))
            .display_name
    }

    #[tokio::test]
    async fn get_current_user_idtyp_app_with_app_displayname_returns_display_name_with_app_marker()
    {
        // `oid != sub` so only `idtyp` marks the token app-only.
        let name = display_name_of(json!({
            "idtyp": "app",
            "app_displayname": "Asset Agent",
            "azp": CLIENT_ID,
            "oid": APP_OID,
            "sub": USER_SUB,
        }))
        .await;

        assert_eq!(name, "Asset Agent (app)");
    }

    #[tokio::test]
    async fn get_current_user_idtyp_app_is_matched_case_insensitively() {
        let name = display_name_of(json!({
            "idtyp": "APP",
            "azp": CLIENT_ID,
            "oid": APP_OID,
            "sub": USER_SUB,
        }))
        .await;

        assert_eq!(name, format!("{CLIENT_ID} (app)"));
    }

    #[tokio::test]
    async fn get_current_user_oid_equal_to_sub_without_idtyp_returns_client_id_with_app_marker() {
        // Microsoft's documented app-only check, for tokens issued without the
        // optional `idtyp` claim: v2.0 tokens carry `azp`, v1.0 tokens `appid`.
        for (version, client_claim) in [("v2.0", "azp"), ("v1.0", "appid")] {
            let name = display_name_of(json!({
                client_claim: CLIENT_ID,
                "oid": APP_OID,
                "sub": APP_OID,
            }))
            .await;

            assert_eq!(name, format!("{CLIENT_ID} (app)"), "{version} token");
        }
    }

    #[tokio::test]
    async fn get_current_user_app_only_without_client_id_returns_oid_with_app_marker() {
        let name = display_name_of(json!({ "oid": APP_OID, "sub": APP_OID })).await;

        assert_eq!(name, format!("{APP_OID} (app)"));
    }

    #[tokio::test]
    async fn get_current_user_app_only_with_no_label_claim_falls_back_to_sub() {
        // `idtyp=app` but none of `app_displayname` / `azp` / `appid` / `oid`:
        // the app tier yields nothing and the chain carries on to `sub`.
        let name = display_name_of(json!({ "idtyp": "app", "sub": USER_SUB })).await;

        assert_eq!(name, USER_SUB);
    }

    #[tokio::test]
    async fn get_current_user_absent_or_empty_oid_and_sub_are_not_app_only() {
        // The `oid == sub` test needs both claims present and non-empty:
        // absent (`None == None`) or empty (`"" == ""`) pairs must not mark the
        // token app-only, or an owner-chosen `app_displayname` or a bare
        // client ID would be shown on a token that is not one. With no human,
        // `sub` or `oid` claim left, the chain fails closed with SR-2.5.
        for (case, claims) in [
            ("oid and sub absent", json!({ "azp": CLIENT_ID })),
            (
                "oid and sub empty",
                json!({ "oid": "", "sub": "", "app_displayname": "Asset Agent" }),
            ),
        ] {
            let result = JwtUserinfoClient::new()
                .get_current_user(&make_jwt(&claims))
                .await;

            assert!(
                matches!(result, Err(AuthError::InvalidResponse(ref msg)) if msg == SR_2_5_MISSING_CLAIM_MSG),
                "{case}: expected the SR-2.5 missing-claim error, got {result:?}",
            );
        }
    }

    #[tokio::test]
    async fn get_current_user_idtyp_user_returns_sub_not_app_label() {
        // `idtyp` comes back `"user"` on user tokens; its presence alone must
        // not mark the token app-only.
        let name = display_name_of(json!({
            "idtyp": "user",
            "azp": CLIENT_ID,
            "oid": APP_OID,
            "sub": USER_SUB,
        }))
        .await;

        assert_eq!(name, USER_SUB);
    }

    #[tokio::test]
    async fn get_current_user_delegated_token_never_uses_app_displayname() {
        // On a user-delegated token `app_displayname` names the client app the
        // user signed in with, not the person.
        let name = display_name_of(json!({
            "scp": "user_impersonation",
            "app_displayname": "Some CLI",
            "azp": CLIENT_ID,
            "oid": APP_OID,
            "sub": USER_SUB,
        }))
        .await;

        assert_eq!(name, USER_SUB);
    }

    #[tokio::test]
    async fn get_current_user_non_entra_token_with_azp_and_no_scp_returns_sub() {
        // Another issuer's user token: `azp` present, no `scp` (it uses
        // `scope`), no `oid`. Only Entra's `oid == sub` or `idtyp=app` marks a
        // token app-only, so this keeps its `sub` label.
        let name = display_name_of(json!({
            "sub": "auth0|1",
            "azp": "c",
            "scope": "openid",
        }))
        .await;

        assert_eq!(name, "auth0|1");
    }

    #[tokio::test]
    async fn get_current_user_human_claim_wins_over_app_label() {
        let name = display_name_of(json!({
            "name": "Ada Lovelace",
            "idtyp": "app",
            "app_displayname": "Asset Agent",
            "azp": CLIENT_ID,
            "oid": APP_OID,
            "sub": APP_OID,
        }))
        .await;

        assert_eq!(name, "Ada Lovelace");
    }

    #[tokio::test]
    async fn get_current_user_sub_only_token_is_unchanged() {
        let name = display_name_of(json!({ "sub": "user-1234" })).await;

        assert_eq!(name, "user-1234");
    }

    #[tokio::test]
    async fn get_current_user_unusable_app_displayname_falls_back_to_client_id() {
        // A control character (here C0 newline and ESC) can break the line
        // before the ` (app)` marker or hide it; empty counts as missing. The
        // rest of the rule (C1 and DEL, format characters, U+2028 / U+2029,
        // trimming, the 64-scalar cap) and the characters it keeps, such as
        // `|`, are covered row by row in
        // `get_current_user_app_displayname_label_rule_by_class`.
        for app_displayname in ["Asset\nAgent", "Asset\u{1b}Agent", ""] {
            let name = display_name_of(json!({
                "idtyp": "app",
                "app_displayname": app_displayname,
                "azp": CLIENT_ID,
                "oid": APP_OID,
                "sub": APP_OID,
            }))
            .await;

            assert_eq!(
                name,
                format!("{CLIENT_ID} (app)"),
                "app_displayname {app_displayname:?} must not be used"
            );
        }
    }

    #[tokio::test]
    async fn get_current_user_app_displayname_label_rule_by_class() {
        // Rows for each class of the `app_displayname` rule on an app-only
        // token: a rejected name falls back to the client ID, and an accepted
        // name is trimmed and gets ` (app)`. U+0085, U+2028 and U+2029 are
        // also Unicode whitespace, which `trim` strips from the ends, so each
        // character row puts its character in the middle of the name, where
        // only the character check can reject it.
        let client_fallback = format!("{CLIENT_ID} (app)");
        let e64 = "\u{e9}".repeat(64);
        let rows: Vec<(&str, String, String)> = vec![
            (
                "C1 control U+0085",
                "Asset\u{85}Agent".into(),
                client_fallback.clone(),
            ),
            (
                "DEL U+007F",
                "Asset\u{7f}Agent".into(),
                client_fallback.clone(),
            ),
            (
                "Cf bidi override U+202E",
                "Asset\u{202e}Agent".into(),
                client_fallback.clone(),
            ),
            (
                "Cf zero-width space U+200B",
                "Asset\u{200b}Agent".into(),
                client_fallback.clone(),
            ),
            (
                "Cf byte-order mark U+FEFF",
                "Asset\u{feff}Agent".into(),
                client_fallback.clone(),
            ),
            (
                "line separator U+2028",
                "Asset\u{2028}Agent".into(),
                client_fallback.clone(),
            ),
            (
                "paragraph separator U+2029",
                "Asset\u{2029}Agent".into(),
                client_fallback.clone(),
            ),
            (
                "whitespace only: spaces",
                "   ".into(),
                client_fallback.clone(),
            ),
            (
                "whitespace only: U+3000",
                "\u{3000}".into(),
                client_fallback.clone(),
            ),
            (
                "length cap: 65 two-byte scalars",
                "\u{e9}".repeat(65),
                client_fallback.clone(),
            ),
            // 64 scalars but 128 bytes, and 68 scalars before trimming: kept,
            // so the cap counts scalar values (not bytes) after trimming.
            (
                "length cap boundary: 64 scalars after trim",
                format!("  {e64}  "),
                format!("{e64} (app)"),
            ),
            (
                "surrounding whitespace trimmed",
                "  Asset Agent  ".into(),
                "Asset Agent (app)".into(),
            ),
            (
                "interior spaces kept",
                "Asset  Render Agent".into(),
                "Asset  Render Agent (app)".into(),
            ),
            // `|` is an ordinary character in a label: the presence wire
            // escapes it, so the name and its marker reach every peer whole.
            (
                "pipe kept",
                "Asset|Agent".into(),
                "Asset|Agent (app)".into(),
            ),
        ];

        // Every row runs before the assertion, so a failure lists each row
        // that broke, not only the first.
        let mut failures = Vec::new();
        for (class, app_displayname, expected) in rows {
            let name = display_name_of(json!({
                "idtyp": "app",
                "app_displayname": app_displayname,
                "azp": CLIENT_ID,
                "oid": APP_OID,
                "sub": APP_OID,
            }))
            .await;

            if name != expected {
                failures.push(format!(
                    "{class}: app_displayname {app_displayname:?} gave {name:?}, expected {expected:?}"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "app_displayname label rule broken for:\n{}",
            failures.join("\n")
        );
    }

    #[tokio::test]
    async fn get_current_user_app_only_token_keeps_oid_as_id() {
        // The ` (app)` label is the display name only; `id` keeps its rule.
        let info = JwtUserinfoClient::new()
            .get_current_user(&make_jwt(&json!({
                "idtyp": "app",
                "app_displayname": "Asset Agent",
                "azp": CLIENT_ID,
                "oid": APP_OID,
                "sub": APP_OID,
            })))
            .await
            .expect("app-only JWT must decode");

        assert_eq!(info.id, APP_OID);
        assert_eq!(info.display_name, "Asset Agent (app)");
    }
}
