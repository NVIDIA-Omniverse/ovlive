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

//! A Valkey ACL user that can run everything but one denied command (or
//! command family), for the tests that need a client the store refuses
//! that command to: `b_bhb_tests` (denied the `client` family) and
//! `scan_guard_test` (denied `cluster|nodes`, or `info`). The user's `ACL SETUSER`
//! arguments, the per-node `ACL` command, the nodes to provision, the URL
//! that authenticates as the user, and the guard that deletes the user on
//! every exit. Each test builds its own fixture from these, with its own
//! SKIP conditions and its own check that the denial is real.

use fred::clients::Client;
use fred::interfaces::{ClientLike, ClusterInterface};
use fred::types::config::{Options, Server};
use fred::types::{CustomCommand, Value};

use live_edit::storage::test_support::TestValkey;

/// Password for an ACL fixture's user. Fixed rather than generated: the user
/// is unique per run and is deleted before the test returns, so the secret
/// carries no meaning — the *username* is what must not collide.
pub(crate) const ACL_FIXTURE_PASSWORD: &str = "acl-fixture-pw";

/// `ACL SETUSER` arguments for `user`, able to do everything **except**
/// `denied` (an ACL rule such as `-client` or `-cluster|nodes`). `~*` `&*`
/// `+@all` leave the user indistinguishable from `default` on every other
/// axis, so a failure in the scenario cannot be blamed on an over-tight ACL.
pub(crate) fn acl_setuser_args(user: &str, denied: &str) -> Vec<String> {
    vec![
        "SETUSER".to_owned(),
        user.to_owned(),
        "on".to_owned(),
        format!(">{ACL_FIXTURE_PASSWORD}"),
        "~*".to_owned(),
        "&*".to_owned(),
        "+@all".to_owned(),
        denied.to_owned(),
    ]
}

/// Send one `ACL` subcommand, optionally pinned to a single cluster node.
///
/// Pinning is mandatory on a cluster: ACL users are **per-node** state, not
/// replicated by the cluster bus, so a `SETUSER` that lands on an arbitrary
/// primary leaves the others rejecting the credential and the fixture's client
/// fails to connect for a reason that has nothing to do with the mechanism
/// under test.
pub(crate) async fn acl_command(
    client: &Client,
    node: Option<&Server>,
    args: Vec<String>,
) -> Result<Value, fred::error::Error> {
    let command = CustomCommand::new_static("ACL", None::<u16>, false);
    match node {
        Some(node) => {
            // One attempt, as `b_bhb_tests`' `pinned_to`: an `ACL` command
            // whose connection fails is reported, not retried.
            let pinned = Options {
                cluster_node: Some(node.clone()),
                max_attempts: Some(1),
                ..Default::default()
            };
            client
                .with_options(&pinned)
                .custom::<Value, String>(command, args)
                .await
        }
        None => client.custom::<Value, String>(command, args).await,
    }
}

/// Every node a fixture must provision its ACL user on: all primaries of a
/// clustered client, or `[None]` (unpinned) for a centralized one.
pub(crate) fn acl_target_nodes(client: &Client) -> Vec<Option<Server>> {
    if !client.is_clustered() {
        return vec![None];
    }
    client
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .unique_primary_nodes()
        .into_iter()
        .map(Some)
        .collect()
}

/// Deletes the fixture's ACL user on the way out — **including on a panic**,
/// which is the whole reason this is a `Drop` impl rather than a call at the
/// end of the test body. A leaked user would survive on the shared store and,
/// since it carries a password, would quietly change what a later run's
/// `ACL SETUSER` is doing.
///
/// Holds the *control* fixture (the default-user connection that created the
/// user) so the delete has a connection to run on regardless of what happened
/// to the restricted one. Build it before checking whether every `SETUSER`
/// succeeded: one that landed on node 1 and failed on node 2 has still created
/// the user. Bind it before the restricted fixture, so the restricted one (and
/// its key clean-up) drops first: `ACL DELUSER` disconnects every client
/// authenticated as the user.
pub(crate) struct AclUserGuard {
    pub(crate) control: TestValkey,
    pub(crate) nodes: Vec<Option<Server>>,
    pub(crate) user: String,
}

impl Drop for AclUserGuard {
    fn drop(&mut self) {
        let client = self.control.client().pool.next().clone();
        let nodes = self.nodes.clone();
        let user = self.user.clone();
        // Same two guards `TestValkey::drop` takes, for the same reasons: no
        // ambient runtime, or a current-thread one, makes `block_in_place`
        // unusable. Leaking the user is the lesser evil to panicking in a drop.
        let Ok(rt_handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        if rt_handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::MultiThread {
            return;
        }
        tokio::task::block_in_place(|| {
            rt_handle.block_on(async move {
                for node in &nodes {
                    let _ = acl_command(
                        &client,
                        node.as_ref(),
                        vec!["DELUSER".to_owned(), user.clone()],
                    )
                    .await;
                }
            });
        });
    }
}

/// Rewrite `raw` so it authenticates as `user`.
///
/// `None` means the ambient URL cannot carry an injected credential and the
/// caller must SKIP: either the scheme is not the one expected, or the URL
/// already carries credentials — rewriting those would silently change which
/// identity the control half of the fixture runs as, and the fixture's whole
/// claim is that the two halves differ in exactly one respect.
pub(crate) fn credentialed_url(raw: &str, scheme: &str, user: &str) -> Option<String> {
    let rest = raw.strip_prefix(scheme)?;
    if rest.contains('@') {
        return None;
    }
    Some(format!("{scheme}{user}:{ACL_FIXTURE_PASSWORD}@{rest}"))
}
