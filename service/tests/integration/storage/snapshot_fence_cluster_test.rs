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

//! The six snapshot-fence scripts against a real multi-shard Valkey Cluster
//! (b.y3m Epic 3 Task 2; Spec SDD SR-5.1–SR-5.4, PM review F4).
//!
//! Every fence script declares every key it touches, and all of them carry the
//! object's `{oid}` hash tag. Objects spread across every shard, so the scripts
//! must answer on each one, and each must touch no key outside its object's
//! tag. One test proves both per primary: for an object on that primary it runs
//! the whole lifecycle through the `storage::snapshot_fence` wrappers — acquire,
//! renew, probe, release, intent register, intent renew — checks each verdict,
//! and reads the keys every script touched from the primary's `MONITOR` record.
//!
//! Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` (`bash scripts/dev-valkey.sh
//! cluster-up` → `redis-cluster://127.0.0.1:7000`) and prints a `SKIP:` line
//! when it is unset. nextest records that skip as a pass, so the test also
//! prints `CLUSTER-GATED RAN: <test> shards=<n>` when it actually runs, and it
//! is in the `ci` profile's success-output filter in `.config/nextest.toml`:
//! CI's release-gating `test-cluster:` job stores its output in the JUnit
//! report, which is the proof it ran.
//!
//! Containment is asserted on the `MONITOR` record, never on a refusal: the
//! local cluster and AMR accept an undeclared other-slot key that lands on the
//! same node, so a refusal cannot prove it.

use std::collections::{BTreeMap, BTreeSet};

use fred::interfaces::{ClientLike, ClusterInterface, HashesInterface, KeysInterface};
use fred::types::config::Server;

use live_edit::storage::snapshot_fence::{
    self as fence, AcquireVerdict, FenceProbe, GrantKind, IntentRenewVerdict, IntentVerdict,
    LayerSize, ReleaseVerdict, RenewVerdict,
};
use live_edit::storage::test_support::TestValkey;

use crate::storage::snapshot_read_support::{monitor_fence, MonitorTap, MonitoredCommand};

/// Upper bound on object ids tried while looking for one per primary. Ids hash
/// over the whole slot space, so a handful suffice for three primaries; the
/// bound turns an impossible search into a failure instead of a hang.
const MAX_OIDS_PER_SEARCH: u64 = 10_000;

/// The size every script computes its limits from: 4× the shipped ceiling.
const SIZE: LayerSize = LayerSize {
    items: 150_216,
    bytes: 1_799_752,
};

async fn cluster() -> Option<TestValkey> {
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); run \
             `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster:` job run this"
        );
        return None;
    };
    let tv = TestValkey::setup_with_url(url, 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (`redis-cluster://...`); \
         a single node smuggled in through this gate would report per-shard coverage the suite \
         does not have"
    );
    Some(tv)
}

fn primaries(tv: &TestValkey) -> Vec<Server> {
    tv.client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .unique_primary_nodes()
}

/// The primary serving `key`'s slot, from the routing table fred dispatches by.
fn owner_of(tv: &TestValkey, key: &str) -> String {
    let slot = fred::util::redis_keyslot(key.as_bytes());
    let server = tv
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .get_server(slot)
        .unwrap_or_else(|| panic!("no primary serves slot {slot}"))
        .clone();
    format!("{}:{}", server.host, server.port)
}

/// Every command a script issued after a client `EVALSHA` naming a key under
/// `tag`: the script's own commands follow its `EVALSHA` in the feed, before
/// any other client's (a script runs atomically). An `EVALSHA` answered
/// `NOSCRIPT` is followed by the client's `SCRIPT LOAD`, so it contributes
/// nothing.
fn script_commands_for(seen: &[MonitoredCommand], tag: &str) -> Vec<MonitoredCommand> {
    let mut out = Vec::new();
    for (i, c) in seen.iter().enumerate() {
        let names_tag = !c.issued_by_script()
            && c.name() == "EVALSHA"
            && c.args.iter().skip(3).any(|a| a.starts_with(tag));
        if names_tag {
            out.extend(
                seen[i + 1..]
                    .iter()
                    .take_while(|s| s.issued_by_script())
                    .cloned(),
            );
        }
    }
    out
}

/// Run the whole fence and intent lifecycle on `oid`, checking every verdict.
async fn run_every_fence_script(tv: &TestValkey, oid: u64, shard: &str) {
    let client = tv.client();
    let keys = tv.key_schema();

    let lease = match fence::acquire_fence(client, keys, oid, SIZE).await {
        Ok(AcquireVerdict::Granted(lease)) => lease,
        other => panic!("{shard}: acquire must grant; got {other:?}"),
    };
    assert_eq!(lease.kind(), GrantKind::StartedEpoch, "{shard}");
    assert_eq!(lease.cut(), 1, "{shard}: the cut is the seeded header seq");
    assert_eq!(
        lease.renew().await.expect("renew must answer"),
        RenewVerdict::Renewed,
        "{shard}"
    );
    let probe = fence::probe_fence(client, keys, oid)
        .await
        .expect("probe must answer");
    assert!(
        matches!(
            probe,
            FenceProbe::Present {
                live_retry_after: Some(_),
                stored_size: Some(SIZE),
                ..
            }
        ),
        "{shard}: the probe must see the live lease and the stored size; got {probe:?}"
    );
    assert_eq!(
        lease
            .release(Some(SIZE))
            .await
            .expect("release must answer"),
        ReleaseVerdict::Verified,
        "{shard}"
    );
    let intent = match fence::register_intent(client, keys, oid, SIZE).await {
        Ok(IntentVerdict::Registered(intent)) => intent,
        other => panic!("{shard}: register_intent must register; got {other:?}"),
    };
    assert_eq!(
        intent.renew().await.expect("intent renew must answer"),
        IntentRenewVerdict::Renewed,
        "{shard}"
    );
    intent.clear().await.expect("clear must succeed");
}

/// For every primary, an object whose `{oid}` slot it serves: each of the six
/// scripts answers there with the verdict its state calls for, and the
/// primary's `MONITOR` record shows every key the scripts touched carrying the
/// object's hash tag — including both keys they write, on that shard.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_fence_scripts_run_for_an_object_on_every_shard() {
    let Some(tv) = cluster().await else { return };
    let primaries = primaries(&tv);

    let mut per_shard: BTreeMap<String, u64> = BTreeMap::new();
    for oid in 1..=MAX_OIDS_PER_SEARCH {
        if per_shard.len() == primaries.len() {
            break;
        }
        let owner = owner_of(&tv, &tv.key_schema().obj_header(&oid.to_string()));
        per_shard.entry(owner).or_insert(oid);
    }
    assert_eq!(
        per_shard.len(),
        primaries.len(),
        "the search must find an object id on every primary within {MAX_OIDS_PER_SEARCH} ids"
    );

    for (shard, oid) in &per_shard {
        let o = oid.to_string();
        let keys = tv.key_schema();
        let _: () = tv
            .client()
            .pool
            .hset(keys.obj_header(&o), [("seq", "1")])
            .await
            .expect("seeding the header must succeed");
        let tag = format!("{}{{{oid}}}:", tv.prefix());

        let mut tap = MonitorTap::start(shard).await;
        run_every_fence_script(&tv, *oid, shard).await;
        // The fence must land on the same primary as the scripts: give it the tag.
        let feed_fence = monitor_fence(&tag);
        let _: i64 = tv
            .client()
            .pool
            .exists(&feed_fence)
            .await
            .expect("the feed fence EXISTS must succeed");
        let touched = script_commands_for(&tap.drain_until(&feed_fence).await, &tag);

        for c in &touched {
            if c.name() == "TIME" {
                continue;
            }
            let key = c.args.get(1).map(String::as_str).unwrap_or_default();
            assert!(
                key.starts_with(&tag),
                "{shard}: every key a fence script touches must carry the object's hash tag \
                 ({tag}…); it touched {key:?} with {c:?}"
            );
        }
        let writes: BTreeSet<(String, String)> = touched
            .iter()
            .filter(|c| ["HSET", "PEXPIRE"].contains(&c.name().as_str()))
            .map(|c| (c.name(), c.args.get(1).cloned().unwrap_or_default()))
            .collect();
        for key in [keys.obj_snapshot_fence(&o), keys.obj_snapshot_intent(&o)] {
            for command in ["HSET", "PEXPIRE"] {
                assert!(
                    writes.contains(&(command.to_owned(), key.clone())),
                    "{shard}: the record must show the scripts' {command} on {key}, or the \
                     containment check above is vacuous; script writes seen: {writes:?}"
                );
            }
        }
    }

    println!(
        "CLUSTER-GATED RAN: storage::snapshot_fence_cluster_test::snapshot_fence_scripts_run_for_an_object_on_every_shard shards={}",
        per_shard.len()
    );
}
