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

//! The key schema the `live-edit` binary builds every Valkey key under, and
//! its test-only prefix override, `LIVE_EDIT_TEST_ONLY_KEY_PREFIX`.
//!
//! `main.rs` calls [`binary_key_schema_from_env`] once, after configuration
//! loads and before any dependency step, and hands the result to every part of
//! the process that names a key; nothing on the binary's path builds a
//! `KeySchema` of its own.
//!
//! The override exists so a test can run the real binary against a shared
//! test store under its own prefix (the `TestValkey` fixture's
//! `test_<uuid>:`), rather than under the production `le:`, where the binary
//! would read and write the fleet-wide keys every other process on that
//! store shares, the disaster-recovery hold marker among them. It is compiled
//! only into a build with the test-only `test-support` feature: a build
//! without it, the release build among them, carries neither the variable's
//! name nor any code that reads it, so it always builds the production
//! schema whatever its environment holds. Like
//! `LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS`, it is not in
//! the Helm chart or the operator configuration table.

use crate::storage::keys::KeySchema;

/// The test-only key-prefix variable. Compiled only with `test-support`.
#[cfg(feature = "test-support")]
pub const TEST_ONLY_KEY_PREFIX_ENV: &str = "LIVE_EDIT_TEST_ONLY_KEY_PREFIX";

/// The key schema the binary builds every key under: always the production
/// schema (`le:`). This build has no `test-support` feature, so it reads no
/// variable.
#[cfg(not(feature = "test-support"))]
pub fn binary_key_schema_from_env() -> Result<KeySchema, String> {
    Ok(KeySchema::default())
}

/// The key schema the binary builds every key under: the production schema
/// (`le:`) when [`TEST_ONLY_KEY_PREFIX_ENV`] is unset, or a schema under the
/// prefix it names ([`test_only_key_schema`]), with one WARN saying so.
///
/// Fails closed with a message naming the variable when the value is not
/// UTF-8 or [`test_only_key_schema`] refuses it; `main.rs` reports that as a
/// configuration error and exits before any dependency step (1, or 2 in a DR
/// Job).
#[cfg(feature = "test-support")]
pub fn binary_key_schema_from_env() -> Result<KeySchema, String> {
    match std::env::var(TEST_ONLY_KEY_PREFIX_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(KeySchema::default()),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "invalid value for {TEST_ONLY_KEY_PREFIX_ENV}: it is not valid UTF-8"
        )),
        Ok(prefix) => {
            let schema = test_only_key_schema(&prefix)?;
            tracing::warn!(
                prefix = schema.prefix(),
                "LIVE_EDIT_TEST_ONLY_KEY_PREFIX is set: this process builds every Valkey key \
                 under this prefix instead of le:. A test-only setting, read only by a build \
                 with the test-support feature"
            );
            Ok(schema)
        }
    }
}

/// A key schema under `prefix`, the value of [`TEST_ONLY_KEY_PREFIX_ENV`], or
/// an error naming the variable and the reason when `prefix` is:
///
/// - empty: every key would lose its namespace;
/// - holding `{` or `}`: the braces would become, or cut short, the keys'
///   Valkey Cluster hash tags. The hold marker's tag is the whole object-id
///   counter key, prefix included (`KeySchema::dr_hold_marker`), so a brace
///   there would move the marker off the counter's slot;
/// - holding a glob character (`*`, `?`, `[`, `]`, `\`): the service's
///   `SCAN MATCH` patterns start with the prefix, so the orphan sweep and the
///   other key walks would also match keys under other prefixes;
/// - holding whitespace or a control character: a stray newline or space from
///   a shell would silently become part of every key.
///
/// The `TestValkey` fixture's `test_<uuid>:` passes.
#[cfg(feature = "test-support")]
pub fn test_only_key_schema(prefix: &str) -> Result<KeySchema, String> {
    if prefix.is_empty() {
        return Err(format!(
            "invalid value for {TEST_ONLY_KEY_PREFIX_ENV}: it is empty; unset it to build keys \
             under the production prefix"
        ));
    }
    if let Some(c) = prefix
        .chars()
        .find(|c| matches!(c, '{' | '}' | '*' | '?' | '[' | ']' | '\\'))
    {
        return Err(format!(
            "invalid value for {TEST_ONLY_KEY_PREFIX_ENV}: {prefix:?} contains {c:?}, which \
             would break the keys' hash tags or the key walks' SCAN MATCH patterns"
        ));
    }
    if let Some(c) = prefix.chars().find(|c| c.is_whitespace() || c.is_control()) {
        return Err(format!(
            "invalid value for {TEST_ONLY_KEY_PREFIX_ENV}: {prefix:?} contains the whitespace or \
             control character {c:?}"
        ));
    }
    Ok(KeySchema::new(prefix))
}
