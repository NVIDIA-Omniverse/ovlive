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

//! The store side of the fleet hold (SDD SR-2.2, SR-4.1, SR-4.2): the hold
//! marker's field names, the hold poll's one read, and the un-hold.
//!
//! The marker is the HASH at [`KeySchema::dr_hold_marker`], in the object-id
//! counter's slot. [`read_hold_marker`] reads it in one `EVALSHA` of
//! `scripts/dr_hold_poll.lua` and returns its fields, or `None` when it does
//! not exist. Turning that into a verdict, the per-call timeout, the poll
//! loop and the staleness clock live in `crate::dr::hold`.
//!
//! [`unhold`] is the marker's only deleter (SR-4.2): one `EVALSHA` of
//! `scripts/dr_unhold.lua`, which checks the Job's lease and incident, raises
//! the object-id counter and deletes the marker atomically. It takes the
//! restore or accept-loss Job's [`RecountReceipt`], so no un-hold without a
//! completed authoritative recount compiles; its one caller is
//! `dr::job::finish`.
//! Nothing here writes the marker; tests write it through
//! `storage::test_support::write_hold_marker`, and remove it through the
//! test-only `storage::test_support::remove_hold_marker`.

use fred::types::Value;

use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use crate::storage::ext_reconcile::RecountReceipt;
use crate::storage::keys::KeySchema;
use crate::storage::lua::{DR_HOLD_POLL_SCRIPT, DR_UNHOLD_SCRIPT};

/// The marker field naming the incident an operator Job must match.
pub const MARKER_FIELD_INCIDENT: &str = "incident";
/// The marker field naming why the fleet is held.
pub const MARKER_FIELD_REASON: &str = "reason";
/// The marker field describing what the detector saw.
pub const MARKER_FIELD_EVIDENCE: &str = "evidence";
/// The marker field holding the store's clock, in milliseconds, when the
/// hold was written.
pub const MARKER_FIELD_DETECTED_AT_MS: &str = "detected_at_ms";
/// The marker field naming the process or detector that wrote it.
pub const MARKER_FIELD_DETECTED_BY: &str = "detected_by";

/// Every marker field, in the order `scripts/dr_hold_poll.lua` returns them.
pub const MARKER_FIELDS: [&str; 5] = [
    MARKER_FIELD_INCIDENT,
    MARKER_FIELD_REASON,
    MARKER_FIELD_EVIDENCE,
    MARKER_FIELD_DETECTED_AT_MS,
    MARKER_FIELD_DETECTED_BY,
];

/// An existing hold marker's fields as stored, each `None` when the field is
/// absent. A marker missing fields still holds the fleet: the predicate is
/// that the marker exists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HoldMarkerFields {
    /// `incident`.
    pub incident: Option<String>,
    /// `reason`.
    pub reason: Option<String>,
    /// `evidence`.
    pub evidence: Option<String>,
    /// `detected_at_ms`, unparsed.
    pub detected_at_ms: Option<String>,
    /// `detected_by`.
    pub detected_by: Option<String>,
}

/// Read the hold marker: `Ok(Some(fields))` when it exists, `Ok(None)` when it
/// does not.
///
/// One `EVALSHA` of [`DR_HOLD_POLL_SCRIPT`] on the data pool, keyed by the
/// marker, so it runs on the object-id counter's slot. The data pool is
/// multiplexed, so a poll never waits for a checkout behind request traffic;
/// the txn pool's exclusive checkout could stall it under load and age the
/// staleness clock for a reason that has nothing to do with the store.
///
/// Carries no timeout of its own beyond fred's command timeout: the caller
/// bounds each call (`crate::dr::hold`). Any error, a script error from a
/// marker of the wrong type included, is returned as it is; a reply outside
/// the script's contract is a `Protocol` error.
///
/// fred's own command timeout is counted on
/// `valkey_errors_total{pool="data", kind="command_timeout"}`, but only when it
/// fires before the caller's bound. The hold poll's bound
/// (`crate::dr::hold::poll_call_timeout`) is the smaller of the command
/// timeout and the poll interval, so at the defaults (2 s against 5 s) the
/// caller drops this future first and nothing is counted. A poll stuck on a
/// dead connection then shows only in the poll's own log, with class
/// `call_timeout`, never on that metric.
pub async fn read_hold_marker(
    client: &ValkeyClient,
    keys: &KeySchema,
) -> Result<Option<HoldMarkerFields>, fred::error::Error> {
    let reply: Value = DR_HOLD_POLL_SCRIPT
        .evalsha_with_reload::<Value, _, _>(
            client.pool.next(),
            vec![keys.dr_hold_marker()],
            Vec::<String>::new(),
        )
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(e))?;
    decode_hold_poll_reply(reply)
}

/// Decode `scripts/dr_hold_poll.lua`'s reply: `[present, incident, reason,
/// evidence, detected_at_ms, detected_by]`, `present` 0 or 1 and each field a
/// string or nil (a Lua `false`, which a RESP3 connection may carry as a
/// boolean).
fn decode_hold_poll_reply(reply: Value) -> Result<Option<HoldMarkerFields>, fred::error::Error> {
    let Value::Array(items) = reply else {
        return Err(unexpected_reply(format!("{reply:?}")));
    };
    if items.len() != 1 + MARKER_FIELDS.len() {
        return Err(unexpected_reply(format!("{items:?}")));
    }
    let present = match &items[0] {
        Value::Integer(0) => false,
        Value::Integer(1) => true,
        other => return Err(unexpected_reply(format!("present = {other:?}"))),
    };
    let mut fields = Vec::with_capacity(MARKER_FIELDS.len());
    for item in &items[1..] {
        fields.push(match item {
            Value::Null | Value::Boolean(false) => None,
            Value::String(s) => Some(s.to_string()),
            Value::Bytes(b) => Some(String::from_utf8_lossy(b).into_owned()),
            Value::Integer(n) => Some(n.to_string()),
            other => return Err(unexpected_reply(format!("field = {other:?}"))),
        });
    }
    if !present {
        // A missing hash has no fields; anything else is outside the contract.
        if fields.iter().any(Option::is_some) {
            return Err(unexpected_reply(format!("{items:?}")));
        }
        return Ok(None);
    }
    let mut fields = fields.into_iter();
    Ok(Some(HoldMarkerFields {
        incident: fields.next().flatten(),
        reason: fields.next().flatten(),
        evidence: fields.next().flatten(),
        detected_at_ms: fields.next().flatten(),
        detected_by: fields.next().flatten(),
    }))
}

/// The script replied outside its contract. `Protocol` because the store
/// answered, but not in the agreed shape; not a timeout, so no origin counts
/// it.
fn unexpected_reply(reply: String) -> fred::error::Error {
    fred::error::Error::new(
        fred::error::ErrorKind::Protocol,
        format!("dr_hold_poll returned an unexpected reply: {reply}"),
    )
}

/// How an un-hold ([`unhold`]) ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnholdOutcome {
    /// The marker is deleted; the object-id counter was `old` and is `new`.
    Unheld {
        /// The counter before (0 when absent).
        old: u64,
        /// The counter after: `max(old, floor)`.
        new: u64,
    },
    /// The marker was already absent; the counter was raised all the same.
    /// Usually the caller's own earlier attempt whose reply was lost; a
    /// marker someone removed by hand answers the same.
    AlreadyUnheld {
        /// The counter before.
        old: u64,
        /// The counter after.
        new: u64,
    },
    /// The object-id counter holds a string that is not a decimal integer:
    /// nothing written.
    CounterUnreadable {
        /// The counter's value, as stored.
        value: String,
    },
    /// The marker names another incident: nothing written.
    OtherIncident {
        /// The incident the marker names (empty when its field is absent).
        held: String,
    },
    /// The Job lease is not the caller's: nothing written.
    LeaseLost {
        /// The lease's value (empty when absent).
        holder: String,
    },
}

/// The un-hold (SR-4.2, SR-9.9): one `EVALSHA` of `scripts/dr_unhold.lua` on
/// the object-id counter's slot. In one atomic step it checks that the Job
/// lease still holds `lease_value` and that the marker, when present, names
/// `incident`; then it raises the object-id counter to `floor` when that is
/// higher, and deletes the marker. Every check comes before any write, so a
/// refusal ([`UnholdOutcome::LeaseLost`],
/// [`UnholdOutcome::CounterUnreadable`], [`UnholdOutcome::OtherIncident`])
/// writes nothing.
///
/// The hold marker's only deleter. `receipt` is the proof that the restore
/// or accept-loss Job's authoritative recount completed
/// (`storage::ext_reconcile`); the
/// caller (`dr::job::finish`) also checks that no write ran since.
///
/// An error — a store fault, the server's `WRONGTYPE` for a lease, counter or
/// marker of the wrong type, a reply outside the script's contract — is
/// returned as it is. The caller retries a transient one in place (a retry of
/// an un-hold whose reply was lost answers [`UnholdOutcome::AlreadyUnheld`])
/// and does not retry the others; a `WRONGTYPE` fails the script before any
/// write.
pub(crate) async fn unhold(
    client: &ValkeyClient,
    keys: &KeySchema,
    receipt: &RecountReceipt,
    incident: &str,
    lease_value: &str,
    floor: u64,
) -> Result<UnholdOutcome, fred::error::Error> {
    // The receipt is the gate; its report is the caller's to log.
    let _ = receipt.report();
    let reply: Value = DR_UNHOLD_SCRIPT
        .evalsha_with_reload::<Value, _, _>(
            client.pool.next(),
            vec![
                keys.dr_hold_marker(),
                keys.dr_job_lease(),
                keys.counter_object_id(),
            ],
            vec![
                incident.to_owned(),
                lease_value.to_owned(),
                floor.to_string(),
            ],
        )
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(e))?;
    decode_unhold_reply(reply)
}

/// Decode `scripts/dr_unhold.lua`'s reply: `[outcome, a, b]`, each a string.
fn decode_unhold_reply(reply: Value) -> Result<UnholdOutcome, fred::error::Error> {
    let unexpected = |what: String| {
        fred::error::Error::new(
            fred::error::ErrorKind::Protocol,
            format!("dr_unhold returned an unexpected reply: {what}"),
        )
    };
    let Value::Array(items) = reply else {
        return Err(unexpected(format!("{reply:?}")));
    };
    let text = |item: &Value| -> Option<String> {
        match item {
            Value::String(s) => Some(s.to_string()),
            Value::Bytes(b) => Some(String::from_utf8_lossy(b).into_owned()),
            Value::Integer(n) => Some(n.to_string()),
            _ => None,
        }
    };
    let fields: Option<Vec<String>> = items.iter().map(text).collect();
    let Some(fields) = fields.filter(|f| f.len() == 3) else {
        return Err(unexpected(format!("{items:?}")));
    };
    let number = |s: &str| {
        s.parse::<u64>()
            .map_err(|_| unexpected(format!("{items:?}")))
    };
    match fields[0].as_str() {
        "unheld" => Ok(UnholdOutcome::Unheld {
            old: number(&fields[1])?,
            new: number(&fields[2])?,
        }),
        "already_unheld" => Ok(UnholdOutcome::AlreadyUnheld {
            old: number(&fields[1])?,
            new: number(&fields[2])?,
        }),
        "counter_unreadable" => Ok(UnholdOutcome::CounterUnreadable {
            value: fields[1].clone(),
        }),
        "other_incident" => Ok(UnholdOutcome::OtherIncident {
            held: fields[1].clone(),
        }),
        "lease_lost" => Ok(UnholdOutcome::LeaseLost {
            holder: fields[1].clone(),
        }),
        _ => Err(unexpected(format!("{items:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The script's reply: `present`, then the five fields in
    /// [`MARKER_FIELDS`] order.
    fn reply(present: i64, fields: [Value; 5]) -> Value {
        let mut items = vec![Value::Integer(present)];
        items.extend(fields);
        Value::Array(items)
    }

    fn nils() -> [Value; 5] {
        [
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
        ]
    }

    fn assert_out_of_contract(reply: Value, what: &str) {
        let err = decode_hold_poll_reply(reply)
            .expect_err(&format!("{what} is outside the script's contract"));
        assert_eq!(
            *err.kind(),
            fred::error::ErrorKind::Protocol,
            "{what}: an out-of-contract reply is a Protocol error, got {err:?}"
        );
        assert!(
            err.details().contains("dr_hold_poll"),
            "{what}: the error names the script, got {err:?}"
        );
    }

    #[test]
    fn decode_hold_poll_reply_absent_marker_is_none() {
        assert_eq!(decode_hold_poll_reply(reply(0, nils())).unwrap(), None);
    }

    /// A RESP3 connection may carry Lua's `false` (a nil `HMGET` slot) as a
    /// boolean: it reads as an absent field, exactly as `Null` does.
    #[test]
    fn decode_hold_poll_reply_boolean_false_fields_read_as_absent() {
        let falses = [
            Value::Boolean(false),
            Value::Boolean(false),
            Value::Boolean(false),
            Value::Boolean(false),
            Value::Boolean(false),
        ];
        assert_eq!(
            decode_hold_poll_reply(reply(0, falses.clone())).unwrap(),
            None
        );
        assert_eq!(
            decode_hold_poll_reply(reply(1, falses)).unwrap(),
            Some(HoldMarkerFields::default())
        );
    }

    /// Each field lands in its own slot: five distinct values, five distinct
    /// fields, in the script's order.
    #[test]
    fn decode_hold_poll_reply_present_marker_maps_each_field_in_order() {
        let decoded = decode_hold_poll_reply(reply(
            1,
            [
                Value::from("inc-7"),
                Value::from("counter lost"),
                Value::from("counter absent"),
                Value::from("1760000000123"),
                Value::from("pod-a"),
            ],
        ))
        .unwrap();
        assert_eq!(
            decoded,
            Some(HoldMarkerFields {
                incident: Some("inc-7".to_owned()),
                reason: Some("counter lost".to_owned()),
                evidence: Some("counter absent".to_owned()),
                detected_at_ms: Some("1760000000123".to_owned()),
                detected_by: Some("pod-a".to_owned()),
            })
        );
    }

    /// The predicate is existence: a marker missing fields decodes as present,
    /// with only the missing ones `None`.
    #[test]
    fn decode_hold_poll_reply_present_marker_missing_fields_is_still_present() {
        let decoded = decode_hold_poll_reply(reply(
            1,
            [
                Value::from("inc-7"),
                Value::Null,
                Value::Null,
                Value::Null,
                Value::Null,
            ],
        ))
        .unwrap();
        assert_eq!(
            decoded,
            Some(HoldMarkerFields {
                incident: Some("inc-7".to_owned()),
                ..HoldMarkerFields::default()
            })
        );
        assert_eq!(
            decode_hold_poll_reply(reply(1, nils())).unwrap(),
            Some(HoldMarkerFields::default()),
            "a marker with no fields at all is still present"
        );
    }

    /// A field may arrive as bytes or as an integer, depending on the
    /// connection's protocol and how fred types it; both read as text.
    #[test]
    fn decode_hold_poll_reply_bytes_and_integer_fields_read_as_text() {
        let decoded = decode_hold_poll_reply(reply(
            1,
            [
                Value::from(&b"inc-bytes"[..]),
                Value::Null,
                Value::Null,
                Value::Integer(1_760_000_000_123),
                Value::Null,
            ],
        ))
        .unwrap()
        .expect("present");
        assert_eq!(decoded.incident.as_deref(), Some("inc-bytes"));
        assert_eq!(decoded.detected_at_ms.as_deref(), Some("1760000000123"));
    }

    /// `present` 0 with any field set cannot come from a missing hash: the
    /// reply is refused, never read as "not held".
    #[test]
    fn decode_hold_poll_reply_absent_with_a_field_is_an_error() {
        for slot in 0..5 {
            let mut fields = nils();
            fields[slot] = Value::from("stray");
            assert_out_of_contract(reply(0, fields), &format!("present 0 with field {slot}"));
        }
    }

    #[test]
    fn decode_hold_poll_reply_wrong_length_is_an_error() {
        assert_out_of_contract(Value::Array(vec![]), "an empty array");
        assert_out_of_contract(Value::Array(vec![Value::Integer(0); 5]), "five elements");
        assert_out_of_contract(Value::Array(vec![Value::Integer(0); 7]), "seven elements");
    }

    #[test]
    fn decode_hold_poll_reply_non_array_is_an_error() {
        assert_out_of_contract(Value::Null, "a nil reply");
        assert_out_of_contract(Value::Integer(0), "an integer reply");
        assert_out_of_contract(Value::from("held"), "a string reply");
    }

    #[test]
    fn decode_hold_poll_reply_present_outside_zero_and_one_is_an_error() {
        let mut items = vec![Value::Integer(2)];
        items.extend(nils());
        assert_out_of_contract(Value::Array(items), "present = 2");
        let mut items = vec![Value::from("1")];
        items.extend(nils());
        assert_out_of_contract(Value::Array(items), "present as a string");
        let mut items = vec![Value::Null];
        items.extend(nils());
        assert_out_of_contract(Value::Array(items), "present as nil");
    }

    #[test]
    fn decode_hold_poll_reply_field_of_another_type_is_an_error() {
        for (what, field) in [
            ("an array field", Value::Array(vec![Value::from("x")])),
            ("a double field", Value::Double(1.5)),
            ("a true field", Value::Boolean(true)),
        ] {
            let mut fields = nils();
            fields[0] = field;
            assert_out_of_contract(reply(1, fields), what);
        }
    }

    /// `scripts/dr_unhold.lua`'s reply `[outcome, a, b]`, as three strings.
    fn unhold_reply(outcome: &str, a: &str, b: &str) -> Value {
        Value::Array(vec![Value::from(outcome), Value::from(a), Value::from(b)])
    }

    /// Each outcome the script names decodes to its variant, carrying `a`
    /// (and `b`) as the contract says: the counter before and after for the
    /// two that clear, the stored text for the three refusals.
    #[test]
    fn decode_unhold_reply_maps_each_outcome_to_its_variant() {
        for (reply, expected) in [
            (
                unhold_reply("unheld", "7", "1101"),
                UnholdOutcome::Unheld { old: 7, new: 1101 },
            ),
            (
                unhold_reply("already_unheld", "1101", "1101"),
                UnholdOutcome::AlreadyUnheld {
                    old: 1101,
                    new: 1101,
                },
            ),
            (
                unhold_reply("counter_unreadable", "abc", ""),
                UnholdOutcome::CounterUnreadable {
                    value: "abc".to_owned(),
                },
            ),
            (
                unhold_reply("other_incident", "inc-8", ""),
                UnholdOutcome::OtherIncident {
                    held: "inc-8".to_owned(),
                },
            ),
            (
                unhold_reply("lease_lost", "", ""),
                UnholdOutcome::LeaseLost {
                    holder: String::new(),
                },
            ),
        ] {
            assert_eq!(decode_unhold_reply(reply).unwrap(), expected);
        }
    }

    /// A reply outside the contract is a `Protocol` error naming the
    /// script, never an outcome: an unknown outcome, a wrong length, a
    /// counter of a clearing outcome that is not a decimal, a non-array.
    #[test]
    fn decode_unhold_reply_outside_the_contract_is_a_protocol_error() {
        for (what, reply) in [
            ("an unknown outcome", unhold_reply("held", "1", "1")),
            (
                "two elements",
                Value::Array(vec![Value::from("unheld"), Value::from("1")]),
            ),
            (
                "a non-decimal old counter",
                unhold_reply("unheld", "x", "1"),
            ),
            (
                "a non-decimal new counter",
                unhold_reply("already_unheld", "1", "-1"),
            ),
            (
                "an array element",
                Value::Array(vec![
                    Value::from("unheld"),
                    Value::Array(vec![]),
                    Value::from("1"),
                ]),
            ),
            ("a string reply", Value::from("unheld")),
        ] {
            let err = decode_unhold_reply(reply).expect_err(what);
            assert_eq!(*err.kind(), fred::error::ErrorKind::Protocol, "{what}");
            assert!(err.details().contains("dr_unhold"), "{what}: {err:?}");
        }
    }

    /// [`unhold`] against a real store (SDD SR-4.2, SR-9.9): the counter it
    /// keeps or raises, the outcome of each refusal, and that a refusal or a
    /// `WRONGTYPE` writes nothing — the marker, the counter and the lease
    /// byte-identical (`DUMP`) after it. The restore Job's end-to-end exits
    /// are pinned in `tests/integration/dr/restore_finish_test.rs`.
    ///
    /// Fail-on-old (in-place perturbation of `dr_unhold.lua`, red then green
    /// once restored): with the unreadable-counter arm replaced by the
    /// `redis.error_reply` it had before, the non-decimal counter case fails
    /// with a script error instead of `CounterUnreadable`.
    mod against_the_store {
        use fred::interfaces::{KeysInterface, ListInterface};

        use super::*;
        use crate::storage::ext_reconcile::{recount_authoritative, RecountReceipt};
        use crate::storage::test_support::{HoldMarkerSeed, TestValkey};

        const INCIDENT: &str = "unhold-test-incident";
        const LEASE: &str = "unhold-test-job";

        /// A receipt from a recount of the fixture's keyspace, which must
        /// complete: the un-hold's gate.
        async fn receipt(tv: &TestValkey) -> RecountReceipt {
            recount_authoritative(tv.client(), tv.key_schema(), || async { Ok::<(), ()>(()) })
                .await
                .expect("the recount of the fixture's keyspace completes")
        }

        /// The store held under [`INCIDENT`], the lease [`LEASE`]'s, and the
        /// counter at `counter` (absent when `None`).
        async fn held(tv: &TestValkey, counter: Option<&str>) {
            tv.write_hold_marker(&HoldMarkerSeed::default().with_incident(INCIDENT))
                .await;
            set(tv, tv.key_schema().dr_job_lease(), LEASE).await;
            if let Some(counter) = counter {
                set(tv, tv.key_schema().counter_object_id(), counter).await;
            }
        }

        async fn set(tv: &TestValkey, key: String, value: &str) {
            let _: () = tv
                .client()
                .pool
                .set(key, value, None, None, false)
                .await
                .expect("SET must succeed");
        }

        async fn counter(tv: &TestValkey) -> Option<String> {
            tv.client()
                .pool
                .get(tv.key_schema().counter_object_id())
                .await
                .expect("GET the counter must succeed")
        }

        async fn marker_exists(tv: &TestValkey) -> bool {
            tv.client()
                .pool
                .exists::<bool, _>(tv.key_schema().dr_hold_marker())
                .await
                .expect("EXISTS the marker must succeed")
        }

        /// The marker, the lease and the counter, each as `DUMP` serializes
        /// it (`None` when absent): what "nothing written" compares.
        async fn the_three_keys(tv: &TestValkey) -> Vec<Option<Vec<u8>>> {
            let keys = tv.key_schema();
            let mut dumps = Vec::new();
            for key in [
                keys.dr_hold_marker(),
                keys.dr_job_lease(),
                keys.counter_object_id(),
            ] {
                let dump: Option<Vec<u8>> =
                    tv.client().pool.dump(key).await.expect("DUMP must succeed");
                dumps.push(dump);
            }
            dumps
        }

        async fn unhold_at(
            tv: &TestValkey,
            floor: u64,
        ) -> Result<UnholdOutcome, fred::error::Error> {
            let receipt = receipt(tv).await;
            unhold(
                tv.client(),
                tv.key_schema(),
                &receipt,
                INCIDENT,
                LEASE,
                floor,
            )
            .await
        }

        /// A counter above the floor is kept: `Unheld { old: N, new: N }`,
        /// the counter still N, never lowered; the marker is deleted.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn unhold_keeps_a_counter_above_the_floor_and_deletes_the_marker() {
            let tv = TestValkey::setup().await;
            held(&tv, Some("5000")).await;

            let outcome = unhold_at(&tv, 1_101).await.expect("the un-hold answers");

            assert_eq!(
                outcome,
                UnholdOutcome::Unheld {
                    old: 5_000,
                    new: 5_000
                }
            );
            assert_eq!(counter(&tv).await.as_deref(), Some("5000"), "never lowered");
            assert!(!marker_exists(&tv).await, "the marker is deleted");
        }

        /// A counter with fewer digits than the floor is below it and is
        /// raised (999 against 1101): the script compares decimal texts by
        /// length first, where a plain string compare reads "999" above
        /// "1101".
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn unhold_raises_a_counter_with_fewer_digits_than_the_floor() {
            let tv = TestValkey::setup().await;
            held(&tv, Some("999")).await;

            let outcome = unhold_at(&tv, 1_101).await.expect("the un-hold answers");

            assert_eq!(
                outcome,
                UnholdOutcome::Unheld {
                    old: 999,
                    new: 1_101
                }
            );
            assert_eq!(counter(&tv).await.as_deref(), Some("1101"));
            assert!(!marker_exists(&tv).await);
        }

        /// A lease another Job holds: `LeaseLost` naming the holder, and the
        /// marker, the lease and the counter byte-identical.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn unhold_with_the_lease_held_by_another_answers_lease_lost_and_writes_nothing() {
            let tv = TestValkey::setup().await;
            held(&tv, Some("7")).await;
            set(&tv, tv.key_schema().dr_job_lease(), "another-job").await;
            let before = the_three_keys(&tv).await;

            let outcome = unhold_at(&tv, 1_101).await.expect("the un-hold answers");

            assert_eq!(
                outcome,
                UnholdOutcome::LeaseLost {
                    holder: "another-job".to_owned()
                }
            );
            assert_eq!(the_three_keys(&tv).await, before, "nothing written");
        }

        /// The marker already gone (the Job's own earlier attempt whose
        /// reply was lost): `AlreadyUnheld`, with the counter raised all the
        /// same.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn unhold_with_the_marker_gone_answers_already_unheld_and_raises_the_counter() {
            let tv = TestValkey::setup().await;
            held(&tv, Some("40")).await;
            assert!(tv.remove_hold_marker().await, "fixture: the marker existed");

            let outcome = unhold_at(&tv, 1_101).await.expect("the un-hold answers");

            assert_eq!(
                outcome,
                UnholdOutcome::AlreadyUnheld {
                    old: 40,
                    new: 1_101
                }
            );
            assert_eq!(counter(&tv).await.as_deref(), Some("1101"));
        }

        /// A counter that is not a decimal integer: `CounterUnreadable`
        /// carrying it, and nothing written — the marker stays, the counter
        /// is not overwritten with the floor.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn unhold_on_a_non_decimal_counter_answers_counter_unreadable_and_writes_nothing() {
            let tv = TestValkey::setup().await;
            held(&tv, Some("abc")).await;
            let before = the_three_keys(&tv).await;

            let outcome = unhold_at(&tv, 1_101)
                .await
                .expect("the un-hold answers, not a script error");

            assert_eq!(
                outcome,
                UnholdOutcome::CounterUnreadable {
                    value: "abc".to_owned()
                }
            );
            assert_eq!(the_three_keys(&tv).await, before, "nothing written");
        }

        /// A counter, or a marker, of the wrong type fails the script with
        /// the server's `WRONGTYPE` before any write.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn unhold_on_a_counter_or_marker_of_the_wrong_type_fails_wrongtype_and_writes_nothing(
        ) {
            for wrong in ["counter", "marker"] {
                let tv = TestValkey::setup().await;
                held(&tv, Some("7")).await;
                let keys = tv.key_schema();
                let key = match wrong {
                    "counter" => keys.counter_object_id(),
                    _ => keys.dr_hold_marker(),
                };
                let _: i64 = tv.client().pool.del(&key).await.expect("DEL must succeed");
                let _: i64 = tv
                    .client()
                    .pool
                    .rpush(&key, "x")
                    .await
                    .expect("RPUSH must succeed");
                let before = the_three_keys(&tv).await;

                let err = unhold_at(&tv, 1_101)
                    .await
                    .expect_err("a key of the wrong type fails the script");

                assert!(
                    crate::storage::client::is_wrongtype_reply(&err),
                    "{wrong}: the server's WRONGTYPE, got {err:?}"
                );
                assert_eq!(
                    the_three_keys(&tv).await,
                    before,
                    "{wrong}: nothing written"
                );
            }
        }
    }
}
