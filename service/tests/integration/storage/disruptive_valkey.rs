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

//! The store a disruptive `-OOM` test runs against, and the `maxmemory` it
//! sets there: shared by the `store_write_refusal_disruptive_*` tests of
//! `store_write_refusal_test.rs` (b.yvh) and `oom_collection_test.rs`
//! (b.pa1).
//!
//! Such a test changes the server's `maxmemory`, a server-wide setting, so it
//! never runs against the shared `TEST_VALKEY_URL` server: only against the
//! Valkey named by `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL` (CI's
//! `test-cluster-disruptive:` job sets it to that job's `valkey-disruptive`
//! service). Every such test is `#[ignore]`d and named with the
//! `store_write_refusal_disruptive_` prefix, which the job's `-E` filter and
//! the `disruptive-valkey-config` nextest test group (`.config/nextest.toml`,
//! one at a time) both match. [`RestoreMaxmemory`] puts the setting back when
//! the test ends, on a failure too. This file is the one place the
//! configuration change is made (`shared_valkey_config_fence_test`'s
//! allowlist names it).

use std::collections::HashMap;

use fred::interfaces::{ClientLike, ConfigInterface};

/// The variable naming the disruptive store.
pub(crate) const DISRUPTIVE_VALKEY_URL: &str = "LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL";

/// The disruptive store's URL; `None`, after a `SKIP:` line, when the
/// variable is unset outside CI. Unset under `GITLAB_CI` it is broken job
/// wiring, and the test fails.
pub(crate) fn disruptive_valkey_url() -> Option<String> {
    match std::env::var(DISRUPTIVE_VALKEY_URL) {
        Ok(url) => Some(url),
        Err(_) => {
            assert!(
                std::env::var_os("GITLAB_CI").is_none(),
                "{DISRUPTIVE_VALKEY_URL} is unset in a CI job that runs the disruptive tests; the \
                 job must point it at its own `valkey-disruptive` service"
            );
            eprintln!("SKIP: {DISRUPTIVE_VALKEY_URL} not set");
            None
        }
    }
}

/// Assert the store runs the required `noeviction` policy: a full store must
/// refuse writes `-OOM`, not evict.
pub(crate) async fn assert_noeviction(client: &fred::clients::Client) {
    let policy: HashMap<String, String> = client
        .config_get("maxmemory-policy")
        .await
        .expect("CONFIG GET maxmemory-policy must succeed");
    assert_eq!(
        policy.get("maxmemory-policy").map(String::as_str),
        Some("noeviction"),
        "fixture precondition: the disruptive store must run the required noeviction policy"
    );
}

/// The store's `used_memory` (`INFO memory`), in bytes.
pub(crate) async fn used_memory(client: &fred::clients::Client) -> u64 {
    let info: String = client
        .info(Some(fred::types::InfoKind::Memory))
        .await
        .expect("INFO memory must succeed");
    info.lines()
        .find_map(|line| line.strip_prefix("used_memory:"))
        .expect("INFO memory reports used_memory")
        .trim()
        .parse()
        .expect("used_memory is a byte count")
}

/// Puts the server's `maxmemory` back when dropped — on a panic too.
pub(crate) struct RestoreMaxmemory {
    client: Option<fred::clients::Client>,
    original: String,
}

impl RestoreMaxmemory {
    /// Set `maxmemory` to 1 byte, below anything the server holds: under
    /// `noeviction` every write that can grow memory is refused `-OOM`, while
    /// reads, `DEL`, `UNLINK` and `CONFIG` still run.
    pub(crate) async fn set_full(client: &fred::clients::Client) -> Self {
        Self::set(client, 1).await
    }

    /// Read the server's `maxmemory` through `client`, then set it to
    /// `bytes`.
    ///
    /// A limit at or below what the server holds when this reads it can only
    /// be left over by an earlier run killed before it restored — a test
    /// store is never run full — so the server default (0, no limit) is what
    /// gets restored then.
    pub(crate) async fn set(client: &fred::clients::Client, bytes: u64) -> Self {
        let current: HashMap<String, String> = client
            .config_get("maxmemory")
            .await
            .expect("CONFIG GET maxmemory must succeed");
        let current = current
            .get("maxmemory")
            .cloned()
            .expect("CONFIG GET names maxmemory");
        let limit: u64 = current.parse().expect("maxmemory is a byte count");
        let original = if limit != 0 && limit <= used_memory(client).await {
            "0".to_owned()
        } else {
            current
        };
        client
            .config_set("maxmemory", bytes)
            .await
            .expect("CONFIG SET maxmemory must succeed");
        Self {
            client: Some(client.clone()),
            original,
        }
    }

    /// Put `maxmemory` back now.
    pub(crate) async fn restore(mut self) {
        let client = self.client.take().expect("restored once");
        client
            .config_set("maxmemory", self.original.as_str())
            .await
            .expect("restoring the server's maxmemory must succeed");
    }
}

impl Drop for RestoreMaxmemory {
    fn drop(&mut self) {
        let Some(client) = self.client.take() else {
            return;
        };
        // Only reached when the test failed first: restore without asserting,
        // since a second panic while unwinding would abort the run.
        let original = std::mem::take(&mut self.original);
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    client.config_set::<_, _>("maxmemory", original.as_str()),
                )
                .await;
            });
        });
    }
}
