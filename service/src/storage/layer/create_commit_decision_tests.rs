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

//! [`decide_create_commit`], pure: the one decision table a `CreateLayer`
//! call's commit reads, for a commit script's reply and for the check after a
//! failed run alike. Its order is the commit script's: the operation record
//! first, then the commit claim, then the mapping.
//!
//! The record rows return `mapping_now: None` — a record does not show the
//! mapping. On the check after a failed run the call itself replaces that with
//! the mapping its `MGET` read; that override is a call-site rule, pinned
//! through the call in `tests/integration/storage/b_ua9_tests.rs`, not here.

use super::*;

/// The call's own new object.
const OWN: &str = "910";
/// The mapping value the call planned against.
const EXPECTED: &str = "900";
/// A third object another create mapped.
const THIRD: &str = "920";

fn record(object_id: u64) -> CreateOperationRecord {
    CreateOperationRecord {
        object_id,
        previous_object_id: 900,
        digest: "d".repeat(64),
    }
}

fn decide(
    record: Option<&CreateOperationRecord>,
    claim: Option<&str>,
    mapping: Option<&str>,
) -> CreateCommitVerdict {
    decide_create_commit(record, claim, mapping, Some(EXPECTED), OWN)
}

/// A record naming the call's own object decides `Committed` even over a
/// `vetoed` claim: the record is read first, and only a landing run writes
/// it.
#[test]
fn decide_create_commit_own_record_over_a_vetoed_claim_is_committed() {
    assert_eq!(
        decide(
            Some(&record(910)),
            Some(keys::CREATE_LAYER_CLAIM_VETOED),
            Some(EXPECTED)
        ),
        CreateCommitVerdict::Committed { mapping_now: None }
    );
}

/// A record naming the call's own object decides `Committed` whatever the
/// mapping holds now: another create may have replaced the object since, and
/// that is not a re-plan.
#[test]
fn decide_create_commit_own_record_with_a_moved_mapping_is_committed() {
    for mapping in [Some(THIRD), None] {
        assert_eq!(
            decide(Some(&record(910)), None, mapping),
            CreateCommitVerdict::Committed { mapping_now: None },
            "mapping {mapping:?}"
        );
    }
}

/// A record naming another object is a concurrent call with the same
/// operation id that committed first, even when this call's object is
/// claimed `committed`: the call is answered from that record and re-plans
/// nothing.
#[test]
fn decide_create_commit_foreign_record_over_a_committed_claim_is_recorded_by_another() {
    let foreign = record(920);
    assert_eq!(
        decide(
            Some(&foreign),
            Some(keys::CREATE_LAYER_CLAIM_COMMITTED),
            Some(OWN)
        ),
        CreateCommitVerdict::RecordedByAnother(foreign)
    );
}

/// Without a record, the claim decides, then the mapping.
#[test]
fn decide_create_commit_without_a_record_reads_the_claim_then_the_mapping() {
    let committed = Some(keys::CREATE_LAYER_CLAIM_COMMITTED);
    let vetoed = Some(keys::CREATE_LAYER_CLAIM_VETOED);
    let rows: [(Option<&str>, Option<&str>, CreateCommitVerdict); 7] = [
        // A `committed` claim: a run landed, and the mapping is reported as
        // read, whatever it holds.
        (
            committed,
            Some(THIRD),
            CreateCommitVerdict::Committed {
                mapping_now: Some(Some(THIRD.to_owned())),
            },
        ),
        (
            committed,
            None,
            CreateCommitVerdict::Committed {
                mapping_now: Some(None),
            },
        ),
        // Any other claim value: no run ever will land.
        (vetoed, Some(OWN), CreateCommitVerdict::Vetoed),
        (Some("unknown"), Some(EXPECTED), CreateCommitVerdict::Vetoed),
        // No claim: the mapping naming the call's object reads as landed, the
        // expected value as unchanged, anything else as moved.
        (
            None,
            Some(OWN),
            CreateCommitVerdict::Committed {
                mapping_now: Some(Some(OWN.to_owned())),
            },
        ),
        (None, Some(EXPECTED), CreateCommitVerdict::Unchanged),
        (
            None,
            Some(THIRD),
            CreateCommitVerdict::Moved(Some(THIRD.to_owned())),
        ),
    ];
    for (claim, mapping, verdict) in rows {
        assert_eq!(
            decide(None, claim, mapping),
            verdict,
            "claim {claim:?}, mapping {mapping:?}"
        );
    }
    // A deleted layer under an overwrite is moved to absent; under a new
    // layer, whose expected value is an absent mapping, it is unchanged.
    assert_eq!(decide(None, None, None), CreateCommitVerdict::Moved(None));
    assert_eq!(
        decide_create_commit(None, None, None, None, OWN),
        CreateCommitVerdict::Unchanged
    );
}
