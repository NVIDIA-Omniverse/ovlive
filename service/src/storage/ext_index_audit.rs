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

//! Atomic per-object recount of the external-value reference index
//! (`le:{oid}:obj:ext`, [`KeySchema::obj_ext_index`]).
//!
//! One primitive, [`recount_ext_index`], over `scripts/ext_index_recount.lua`:
//! count the object's value rows per external hash by the decoder's rule (a row
//! holds hash h iff it has no `value` column and its `hash` column is h),
//! compare against the stored index, and — only if they differ — rewrite the
//! differing fields with absolute counts in the same atomic unit. While any
//! row's `hash` column is malformed (not countable here, though the Rust
//! decoder may still count it), it applies only raises and withholds every
//! lowering ([`RecountOutcome::Withheld`]), so its own writes never
//! under-count what the object serves. It never
//! releases a reference: an over-count it removes leaves a global `ref_count`
//! surplus for the reference-count reconciler to drain, and an under-count it
//! raises changes no global count. The script header carries the full contract
//! (liveness, the header stamp that cancels in-flight watched txn commits, the row
//! ceiling, the reply shapes).
//!
//! Two callers, each a `site` label on the counters this records:
//!
//! * `site="reconcile"` — `storage::ext_reconcile`'s object
//!   walk, when an object's index disagrees with the raw row counts it already
//!   read. Outcomes there drive the reconciler's drain gate.
//! * `site="torn_commit"` — `storage::layer`'s txn commit
//!   path after a torn `EXEC`, whose queued absolute index counts may have
//!   landed while other frames of the same commit did not.
//!
//! Kept out of `layer.rs` and `ext_reconcile.rs` so both callers stay a single
//! call.

use std::sync::LazyLock;

use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
use fred::types::scripts::Script;

use crate::observability::metrics::{
    Metrics, EXT_INDEX_DRIFT_DIRECTION_OVER, EXT_INDEX_DRIFT_DIRECTION_UNDER,
    EXT_INDEX_RECOUNT_OUTCOME_CLEAN, EXT_INDEX_RECOUNT_OUTCOME_FAILED,
    EXT_INDEX_RECOUNT_OUTCOME_NOT_LIVE, EXT_INDEX_RECOUNT_OUTCOME_REPAIRED,
    EXT_INDEX_RECOUNT_OUTCOME_TOO_LARGE, EXT_INDEX_RECOUNT_OUTCOME_WITHHELD,
};
use crate::storage::client::{FredErrOrigin, ValkeyClient};
use crate::storage::keys::KeySchema;

/// The compiled per-object recount script. Same lazy-`SHA-1` /
/// `NOSCRIPT`-reload story as `storage::lua::UPDATE_LAYER_FAST_SCRIPT`.
pub(crate) static EXT_INDEX_RECOUNT_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/ext_index_recount.lua")));

/// The `obj:hdr` column a repair stamps with the Valkey-side millisecond time
/// of the repair.
///
/// Written only by `scripts/ext_index_recount.lua`, as its first write, and
/// only when it repairs something. Nothing reads it: its purpose is the write
/// itself, which fires the header `WATCH` of every in-flight **watched** txn
/// commit on the object, so none of them can `EXEC` index counts planned from
/// the pre-repair index. A commit whose `WATCH` was void is not cancelled: it
/// commits unconditionally and can overwrite the repair, and its all-OK `EXEC`
/// triggers no torn-commit recount, so that drift waits for the next
/// reconciler audit. It is removed with the header (retirement, the restore
/// residue wipe) and ignored by the snapshot decoder, so it never reaches a
/// checkpoint. It does not advance `seq`.
///
/// `pub` so integration tests in the external test crate can read it by name;
/// the script spells it literally and the two must match.
pub const HEADER_FIELD_EXT_INDEX_REPAIRED_MS: &str = "ext_index_repaired_ms";

/// One corrected index field, as the script reported it — for the repair log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecountExemplar {
    /// The external hash, lowercase hex.
    pub hash_hex: String,
    /// The stored field value before the repair, truncated by the script;
    /// `None` when the field was absent (or stored the empty string).
    pub stored: Option<String>,
    /// The count the repair wrote; `0` means the field was removed.
    pub counted: u64,
}

/// What one recount found and did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecountOutcome {
    /// The index already matched the rows. Nothing written.
    Clean {
        /// Rows with no `value` column whose `hash` column was not 64
        /// lowercase hex — not counted, and not copied into the index.
        malformed_rows: u64,
    },
    /// The index differed and was rewritten atomically.
    Repaired {
        /// Fields corrected upward (stored below the rows; absent reads 0).
        under: u64,
        /// Fields corrected downward or removed (stored above the rows).
        over: u64,
        /// As in [`RecountOutcome::Clean`].
        malformed_rows: u64,
        /// A bounded sample of the corrected fields.
        exemplars: Vec<RecountExemplar>,
    },
    /// Some value row's `hash` column was malformed, and at least one field
    /// needed lowering or removing. The script applied only the raises (if
    /// any) and withheld every lowering: a malformed row is not counted here,
    /// but the Rust decoder may still count it (uppercase hex, say), so
    /// lowering on this count could under-count what the object serves — the
    /// loss direction. The index still disagrees with the rows on the
    /// withheld fields, so a caller must treat the object as NOT repaired;
    /// the reconciler holds its mismatched hashes from surplus drain.
    Withheld {
        /// Fields raised (as [`RecountOutcome::Repaired`]'s `under`); the
        /// header was stamped iff this is non-zero.
        raised: u64,
        /// Fields that needed lowering or removing and were left as they are.
        withheld: u64,
        /// As in [`RecountOutcome::Clean`]; always non-zero here.
        malformed_rows: u64,
        /// A bounded sample of the raised fields.
        exemplars: Vec<RecountExemplar>,
    },
    /// The object has more value rows than one call reads. Nothing written.
    TooLarge {
        /// `ZCARD obj:flds + ZCARD obj:tss` as the script read it.
        rows: u64,
    },
    /// The object was not live (header absent or partial, `superseded`, or
    /// `ext_gc_scheduled`). Nothing written.
    NotLive,
}

/// Recount `object_id`'s reference index against its rows and repair any
/// drift, atomically, on `conn`.
///
/// `client` and `pool_label` name where `conn` came from, so a command timeout
/// is counted on `valkey_errors_total{pool}` here — callers must not count the
/// returned error again. The row ceiling is half the atomic snapshot read's
/// item limit (two reads per row; a conservative proxy for the same
/// script-duration budget, not a measurement).
///
/// Records `live_edit_ext_index_recount_total{site, outcome}` for every call
/// (`failed` on an error) and `live_edit_ext_index_drift_total{site,
/// direction}` on a repair, and logs a repair at WARN. The WARN is the whole
/// operator response: the repair already happened, so there is nothing to do
/// beyond reading the `direction="under"` trend.
///
/// Any answer that reports malformed rows — [`RecountOutcome::Withheld`], or a
/// `Clean` / `Repaired` one that needed no lowering — records
/// `outcome="withheld"`, at both sites: the recount saw a row it could not
/// count, so its count is not authoritative. Raises it applied still count as
/// `direction="under"`, and the malformed-row WARN (plus, for `Withheld`, the
/// withheld-lowerings WARN) names the layer. That WARN does call for an
/// operator: the malformed rows are keyspace corruption, and until they are
/// corrected the object's index is not repaired. The reconciler gates on the
/// same malformed count.
///
/// Returns the error rather than a classification: a script refusal
/// (`WRONGTYPE_KEY`, `RECOUNT_ARGS`, an OOM on the first write, a slot
/// migration's refusal) and a transport failure all mean "nothing was
/// written", and both callers treat them alike. A slot migration cannot stop
/// the script part-way: its `KEYS` carry the object's migration sentinel
/// ([`KeySchema::migration_sentinel`]), so the store refuses it before it
/// starts while the slot is moving.
pub(crate) async fn recount_ext_index(
    client: &ValkeyClient,
    conn: &fred::clients::Client,
    pool_label: &'static str,
    keys: &KeySchema,
    object_id: &str,
    site: &'static str,
    metrics: Option<&Metrics>,
) -> Result<RecountOutcome, FredError> {
    let result = run_script(client, conn, pool_label, keys, object_id).await;
    match &result {
        Ok(outcome) => record_outcome(object_id, site, outcome, metrics),
        Err(e) => {
            if let Some(m) = metrics {
                m.inc_ext_index_recount(site, EXT_INDEX_RECOUNT_OUTCOME_FAILED);
            }
            tracing::warn!(
                object_id,
                site,
                error = %e,
                "external-value reference index recount failed; nothing was written, and any \
                 per-object drift on this layer is left for the next reference-count reconciler \
                 pass's audit (which cannot repair it either while a key it reads is wrong-typed, \
                 and does not run when the reconciler is disabled); until it is repaired, a \
                 commit releasing on the drifted count can over-release"
            );
        }
    }
    result
}

async fn run_script(
    client: &ValkeyClient,
    conn: &fred::clients::Client,
    pool_label: &'static str,
    keys: &KeySchema,
    object_id: &str,
) -> Result<RecountOutcome, FredError> {
    let row_ceiling = client.snapshot_atomic_read_max_items / 2;
    let script_keys = vec![
        keys.obj_header(object_id),
        keys.obj_ext_index(object_id),
        keys.obj_fields_index(object_id),
        keys.obj_time_samples_index(object_id),
        // Never written: makes the store refuse the script while the object's
        // slot migrates, so it cannot stop after its first write (see the
        // script's "Migration sentinel").
        keys.migration_sentinel(object_id),
    ];
    let argv = vec![
        keys.prefix().to_owned(),
        object_id.to_owned(),
        row_ceiling.to_string(),
    ];
    let reply: Vec<String> = EXT_INDEX_RECOUNT_SCRIPT
        .evalsha_with_reload::<Vec<String>, _, _>(conn, script_keys, argv)
        .await
        .inspect_err(|e| FredErrOrigin::client(client, pool_label).observe_error(e))?;
    parse_reply(&reply)
}

/// Decode the script's flat all-string reply. An unrecognised shape is a
/// protocol fault (a stale cached script body, say) and comes back as an
/// error, so the caller treats it exactly like a failed call.
fn parse_reply(reply: &[String]) -> Result<RecountOutcome, FredError> {
    let bad = || {
        FredError::new(
            FredErrorKind::Parse,
            format!("ext_index_recount returned an unrecognised reply: {reply:?}"),
        )
    };
    let num = |i: usize| -> Result<u64, FredError> {
        reply
            .get(i)
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(bad)
    };
    match reply.first().map(String::as_str) {
        Some("NOT_LIVE") if reply.len() == 1 => Ok(RecountOutcome::NotLive),
        Some("TOO_LARGE") if reply.len() == 2 => Ok(RecountOutcome::TooLarge { rows: num(1)? }),
        Some("CLEAN") if reply.len() == 2 => Ok(RecountOutcome::Clean {
            malformed_rows: num(1)?,
        }),
        Some(tag @ ("REPAIRED" | "WITHHELD"))
            if reply.len() >= 4 && (reply.len() - 4).is_multiple_of(3) =>
        {
            let exemplars = reply[4..]
                .chunks_exact(3)
                .map(|t| {
                    Ok(RecountExemplar {
                        hash_hex: t[0].clone(),
                        stored: (!t[1].is_empty()).then(|| t[1].clone()),
                        counted: t[2].parse::<u64>().map_err(|_| bad())?,
                    })
                })
                .collect::<Result<Vec<_>, FredError>>()?;
            if tag == "REPAIRED" {
                Ok(RecountOutcome::Repaired {
                    under: num(1)?,
                    over: num(2)?,
                    malformed_rows: num(3)?,
                    exemplars,
                })
            } else {
                Ok(RecountOutcome::Withheld {
                    raised: num(1)?,
                    withheld: num(2)?,
                    malformed_rows: num(3)?,
                    exemplars,
                })
            }
        }
        _ => Err(bad()),
    }
}

/// `hash:stored->counted` strings for a repair log line.
fn exemplar_strings(exemplars: &[RecountExemplar]) -> Vec<String> {
    exemplars
        .iter()
        .map(|e| {
            format!(
                "{}:{}->{}",
                e.hash_hex,
                e.stored.as_deref().unwrap_or("absent"),
                e.counted
            )
        })
        .collect()
}

fn record_outcome(
    object_id: &str,
    site: &'static str,
    outcome: &RecountOutcome,
    metrics: Option<&Metrics>,
) {
    let (cell, malformed_rows) = match outcome {
        RecountOutcome::Clean { malformed_rows } => {
            (EXT_INDEX_RECOUNT_OUTCOME_CLEAN, *malformed_rows)
        }
        RecountOutcome::Repaired {
            under,
            over,
            malformed_rows,
            exemplars,
        } => {
            if let Some(m) = metrics {
                m.inc_ext_index_drift(site, EXT_INDEX_DRIFT_DIRECTION_UNDER, *under);
                m.inc_ext_index_drift(site, EXT_INDEX_DRIFT_DIRECTION_OVER, *over);
            }
            let exemplars = exemplar_strings(exemplars);
            tracing::warn!(
                object_id,
                site,
                under,
                over,
                exemplars = ?exemplars,
                "repaired a layer's external-value reference index that disagreed with its value \
                 rows (exemplars are hash:stored->counted); no reference was released, and any \
                 global ref_count surplus left behind is left for the reference-count reconciler \
                 to drain"
            );
            (EXT_INDEX_RECOUNT_OUTCOME_REPAIRED, *malformed_rows)
        }
        // Recorded as `withheld` whatever it raised (as is any reply with
        // malformed rows; see below): the index still disagrees with the rows.
        // The raises it did apply still count as `direction="under"` drift.
        RecountOutcome::Withheld {
            raised,
            withheld,
            malformed_rows,
            exemplars,
        } => {
            if let Some(m) = metrics {
                m.inc_ext_index_drift(site, EXT_INDEX_DRIFT_DIRECTION_UNDER, *raised);
            }
            let exemplars = exemplar_strings(exemplars);
            tracing::warn!(
                object_id,
                site,
                raised,
                withheld,
                malformed_rows,
                exemplars = ?exemplars,
                "external-value reference index recount withheld every lowering because the layer \
                 has value rows whose hash column it cannot count; it applied only the raises \
                 (exemplars are hash:stored->counted). The index still disagrees with the rows \
                 on the withheld fields; the reconciler's audit holds those hashes from surplus \
                 drain on every pass until the malformed rows are corrected or the layer retires, \
                 but a commit releasing on a count below the decoder's can still over-release \
                 until then"
            );
            (EXT_INDEX_RECOUNT_OUTCOME_WITHHELD, *malformed_rows)
        }
        RecountOutcome::TooLarge { rows } => {
            tracing::debug!(
                object_id,
                site,
                rows,
                "external-value reference index recount refused: the layer has more value rows \
                 than one atomic recount reads; nothing was written"
            );
            (EXT_INDEX_RECOUNT_OUTCOME_TOO_LARGE, 0)
        }
        RecountOutcome::NotLive => {
            tracing::debug!(
                object_id,
                site,
                "external-value reference index recount skipped: the layer is not live; nothing \
                 was written"
            );
            (EXT_INDEX_RECOUNT_OUTCOME_NOT_LIVE, 0)
        }
    };
    // One rule at both sites: a recount that saw a malformed row is not
    // authoritative — the decoder may count a row it skipped — so it is
    // recorded as `withheld` whatever its tag. A `clean` or `repaired` there
    // would read as benign while the reconciler holds the layer's hashes.
    let cell = if malformed_rows > 0 {
        EXT_INDEX_RECOUNT_OUTCOME_WITHHELD
    } else {
        cell
    };
    if let Some(m) = metrics {
        m.inc_ext_index_recount(site, cell);
    }
    if malformed_rows > 0 {
        tracing::warn!(
            object_id,
            site,
            malformed_rows,
            "external-value reference index recount found value rows whose hash column is not \
             64 lowercase hex digits; they were not counted, and no index field was lowered on \
             this count. Nothing in the service writes such a column, so this is keyspace \
             corruption; at site=reconcile the layer's mismatched hashes are held from surplus \
             drain on every pass until the rows are corrected or the layer retires"
        );
    }
}

#[cfg(test)]
mod tests {
    // Valkey-backed and in-crate: `recount_ext_index` is `pub(crate)`, and the
    // WATCH-cancellation arm needs `LayerStore`'s pre-dispatch hook, which the
    // crate's own test build compiles in without a feature.
    //
    // The script arms stage the keyspace by hand — a header, value rows with
    // their index members, and an `obj:ext` — because every arm is about a
    // shape the recount must answer for (a dual-shape row, a malformed column,
    // a retiring header) that no single commit produces on demand.

    use super::*;
    use crate::storage::test_support::relay::RelayedUpdates;

    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface};
    use futures::FutureExt;

    use crate::config::UpdateLayerBackend;
    use crate::domain::types::ObjectState;
    use crate::observability::metrics::metrics_scrape_only;
    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::{
        EXT_INDEX_RECOUNT_SITE_RECONCILE, EXT_INDEX_RECOUNT_SITE_TORN_COMMIT,
    };
    use crate::storage::client::POOL_LABEL_DATA;
    use crate::storage::external_values::{ExternalValueStore, ValkeyExternalValueStore};
    use crate::storage::keys::sha256_to_hex;
    use crate::storage::layer::LayerStore;
    use crate::storage::session::SessionStore;
    use crate::storage::test_support::TestValkey;
    use crate::storage::watched_txn::WatchWindowHook;
    use oodf_ffi::{OwnedDelta, SetField};

    /// The object every script arm stages. Each arm has its own `TestValkey`
    /// prefix, so one id is enough.
    const OID: &str = "960001";

    fn hex_of(byte: u8) -> String {
        sha256_to_hex(&[byte; 32])
    }

    /// Which index family a staged row belongs to.
    #[derive(Clone, Copy)]
    enum Family {
        Field,
        TimeSample,
    }

    /// Stage one value row with `columns`, and its index member. A
    /// time-sample row's `name` is its time.
    async fn seed_row(tv: &TestValkey, family: Family, name: &str, columns: &[(&str, &str)]) {
        let keys = tv.key_schema();
        let (index, member, row) = match family {
            Family::Field => {
                let member = KeySchema::field_member("1", name);
                let row = keys.obj_field_from_member(OID, &member);
                (keys.obj_fields_index(OID), member, row)
            }
            Family::TimeSample => {
                let time: f64 = name.parse().expect("a time-sample name is its time");
                let member = KeySchema::time_sample_member("1", time);
                let row = keys.obj_ts_from_member(OID, &member);
                (keys.obj_time_samples_index(OID), member, row)
            }
        };
        let _: () = tv
            .client()
            .pool
            .hset(row, columns.to_vec())
            .await
            .expect("seeding a value row must succeed");
        let _: i64 = tv
            .client()
            .pool
            .zadd(index, None, None, false, false, (1.0, member.as_str()))
            .await
            .expect("seeding a row index member must succeed");
    }

    async fn seed_header(tv: &TestValkey, fields: &[(&str, &str)]) {
        let _: () = tv
            .client()
            .pool
            .hset(tv.key_schema().obj_header(OID), fields.to_vec())
            .await
            .expect("seeding obj:hdr must succeed");
    }

    async fn seed_index(tv: &TestValkey, fields: &[(&str, &str)]) {
        let _: () = tv
            .client()
            .pool
            .hset(tv.key_schema().obj_ext_index(OID), fields.to_vec())
            .await
            .expect("seeding obj:ext must succeed");
    }

    async fn header(tv: &TestValkey) -> HashMap<String, String> {
        tv.client()
            .pool
            .hgetall(tv.key_schema().obj_header(OID))
            .await
            .expect("HGETALL obj:hdr must succeed")
    }

    async fn index(tv: &TestValkey, oid: &str) -> HashMap<String, String> {
        tv.client()
            .pool
            .hgetall(tv.key_schema().obj_ext_index(oid))
            .await
            .expect("HGETALL obj:ext must succeed")
    }

    /// One recount of [`OID`] on `client` (whose item limit sets the row
    /// ceiling), recorded on `metrics` at `site="reconcile"`.
    async fn recount(
        client: &ValkeyClient,
        tv: &TestValkey,
        metrics: &Metrics,
    ) -> Result<RecountOutcome, FredError> {
        recount_ext_index(
            client,
            client.pool.next(),
            POOL_LABEL_DATA,
            tv.key_schema(),
            OID,
            EXT_INDEX_RECOUNT_SITE_RECONCILE,
            Some(metrics),
        )
        .await
    }

    /// `live_edit_ext_index_recount_total{site="reconcile", outcome}`.
    fn recounts(metrics: &Metrics, outcome: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_ext_index_recount_total",
            &[("site", "reconcile"), ("outcome", outcome)],
        )
        .unwrap_or_else(|| panic!("recount cell outcome={outcome} is zero-seeded")) as u64
    }

    // ── Reply decoding (no I/O) ───────────────────────────────────────────

    fn reply(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|p| (*p).to_owned()).collect()
    }

    /// Every reply shape the script does not produce is refused as an
    /// error, so a stale cached script body or a protocol fault is treated
    /// exactly like a failed call — never decoded into an outcome the
    /// callers act on.
    ///
    /// Covers an unknown tag, an empty reply, each tag at the wrong arity, a
    /// non-numeric count, and a `REPAIRED` or `WITHHELD` whose exemplar list
    /// ends in an incomplete triple (which `chunks_exact` would otherwise drop
    /// silently). Every refused shape is collected before asserting, so a
    /// failure lists all of them. The well-formed shapes are decoded too, so
    /// the refusals cannot be a blanket `Err` — and `WITHHELD` must decode as
    /// its own variant, never as `Repaired`: the walk gates on the difference
    /// and the third element means "withheld", not "over".
    #[test]
    fn parse_reply_refuses_every_shape_the_script_does_not_produce() {
        let hash = hex_of(0x76);
        let malformed = [
            reply(&[]),
            reply(&["BOGUS"]),
            reply(&["NOT_LIVE", "extra"]),
            reply(&["TOO_LARGE"]),
            reply(&["TOO_LARGE", "many"]),
            reply(&["CLEAN"]),
            reply(&["CLEAN", "0", "extra"]),
            reply(&["REPAIRED", "1", "0"]),
            reply(&["REPAIRED", "1", "0", "0", &hash, ""]),
            reply(&["REPAIRED", "1", "0", "0", &hash, "", "two"]),
            reply(&["WITHHELD", "1", "2"]),
            reply(&["WITHHELD", "one", "2", "1"]),
            reply(&["WITHHELD", "1", "2", "1", &hash, ""]),
            reply(&["WITHHELD", "1", "2", "1", &hash, "", "two"]),
        ];
        let accepted: Vec<(&Vec<String>, RecountOutcome)> = malformed
            .iter()
            .filter_map(|r| parse_reply(r).ok().map(|o| (r, o)))
            .collect();
        assert!(
            accepted.is_empty(),
            "these replies must be refused, not decoded: {accepted:?}"
        );

        assert_eq!(
            parse_reply(&reply(&["NOT_LIVE"])).ok(),
            Some(RecountOutcome::NotLive)
        );
        assert_eq!(
            parse_reply(&reply(&["TOO_LARGE", "7"])).ok(),
            Some(RecountOutcome::TooLarge { rows: 7 })
        );
        assert_eq!(
            parse_reply(&reply(&["CLEAN", "2"])).ok(),
            Some(RecountOutcome::Clean { malformed_rows: 2 })
        );
        assert_eq!(
            parse_reply(&reply(&["REPAIRED", "1", "0", "0", &hash, "", "1"])).ok(),
            Some(RecountOutcome::Repaired {
                under: 1,
                over: 0,
                malformed_rows: 0,
                exemplars: vec![RecountExemplar {
                    hash_hex: hash.clone(),
                    stored: None,
                    counted: 1,
                }],
            })
        );
        assert_eq!(
            parse_reply(&reply(&["WITHHELD", "1", "2", "3", &hash, "", "1"])).ok(),
            Some(RecountOutcome::Withheld {
                raised: 1,
                withheld: 2,
                malformed_rows: 3,
                exemplars: vec![RecountExemplar {
                    hash_hex: hash.clone(),
                    stored: None,
                    counted: 1,
                }],
            })
        );
        assert_eq!(
            parse_reply(&reply(&["WITHHELD", "0", "1", "1"])).ok(),
            Some(RecountOutcome::Withheld {
                raised: 0,
                withheld: 1,
                malformed_rows: 1,
                exemplars: Vec::new(),
            }),
            "every fix withheld: no raise, no exemplar"
        );
    }

    /// Any recount that saw a malformed row records `outcome="withheld"` —
    /// never `clean` or `repaired` — and this holds at the torn-commit site
    /// as well as the reconcile site.
    ///
    /// Pure: `record_outcome` is the one place both sites record through,
    /// so driving it with each answer shape at `site="torn_commit"` pins
    /// the site that is expensive to reach end to end (it takes a torn
    /// `EXEC` over an object carrying a malformed row). The reconcile site
    /// is pinned end to end by
    /// [`recount_over_a_malformed_row_records_withheld_whatever_it_answers`].
    #[test]
    fn a_reply_reporting_malformed_rows_records_withheld_at_the_torn_commit_site() {
        let metrics = metrics_scrape_only("nvf-recount-withheld-torn");
        let answers = [
            RecountOutcome::Clean { malformed_rows: 1 },
            RecountOutcome::Repaired {
                under: 1,
                over: 0,
                malformed_rows: 1,
                exemplars: Vec::new(),
            },
            RecountOutcome::Withheld {
                raised: 0,
                withheld: 1,
                malformed_rows: 1,
                exemplars: Vec::new(),
            },
        ];
        for answer in &answers {
            record_outcome(
                OID,
                EXT_INDEX_RECOUNT_SITE_TORN_COMMIT,
                answer,
                Some(&metrics),
            );
        }
        let text = render(&metrics);
        let cell = |outcome: &str| {
            sample_value(
                &text,
                "live_edit_ext_index_recount_total",
                &[("site", "torn_commit"), ("outcome", outcome)],
            )
        };
        assert_eq!(
            (cell("withheld"), cell("clean"), cell("repaired")),
            (Some(3.0), Some(0.0), Some(0.0)),
            "all three answers saw a malformed row, so all three are `withheld`"
        );
    }

    // ── Liveness ──────────────────────────────────────────────────────────

    /// The not-live shapes the script must refuse.
    #[derive(Clone, Copy, Debug)]
    enum NotLive {
        /// No `obj:hdr` at all — the recount must not create one.
        NoHeader,
        /// A header with no numeric `seq` — residue, not a materialized
        /// object.
        PartialHeader,
        /// A CreateLayer overwrite (or the orphan sweep) has retired it.
        Superseded,
        /// Its retirement releases are being scheduled.
        GcScheduled,
    }

    /// The recount answers `not_live` and writes NOTHING — no index field, no
    /// repair stamp, no phantom header — on an object that is not live, even
    /// though its index visibly disagrees with its rows.
    ///
    /// Each shape is a window another component owns: a retirement walking
    /// the object's keys, or a restore between its residue wipe and its
    /// rebuild. A repair landing there would race that owner, and a stamp
    /// written to an absent header would mint a header for an object that
    /// does not exist.
    async fn recount_of_a_not_live_object_writes_nothing(shape: NotLive) {
        let tv = TestValkey::setup().await;
        let metrics = metrics_scrape_only("nvf-recount-not-live");
        // An under-count the recount WOULD repair on a live object: one row
        // holds the hash and the index has no field for it.
        let hash = hex_of(0x71);
        seed_row(&tv, Family::Field, "a", &[("hash", &hash), ("size", "1")]).await;
        match shape {
            NotLive::NoHeader => {}
            NotLive::PartialHeader => seed_header(&tv, &[("source_format", "usda")]).await,
            NotLive::Superseded => seed_header(&tv, &[("seq", "1"), ("superseded", "1")]).await,
            NotLive::GcScheduled => {
                seed_header(&tv, &[("seq", "1"), ("ext_gc_scheduled", "1")]).await;
            }
        }
        let header_before = header(&tv).await;

        let outcome = recount(tv.client(), &tv, &metrics)
            .await
            .unwrap_or_else(|e| panic!("{shape:?}: the recount must answer, not fail: {e}"));

        assert_eq!(outcome, RecountOutcome::NotLive, "{shape:?}");
        assert!(
            index(&tv, OID).await.is_empty(),
            "{shape:?}: nothing may be written to obj:ext"
        );
        assert_eq!(
            header(&tv).await,
            header_before,
            "{shape:?}: the header is untouched — no `ext_index_repaired_ms`"
        );
        if matches!(shape, NotLive::NoHeader) {
            let exists: u64 = tv
                .client()
                .pool
                .exists(tv.key_schema().obj_header(OID))
                .await
                .expect("EXISTS obj:hdr must succeed");
            assert_eq!(exists, 0, "the recount must never create a phantom header");
        }
        assert_eq!(recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_NOT_LIVE), 1);
    }

    #[tokio::test]
    async fn recount_of_an_object_with_no_header_writes_nothing() {
        recount_of_a_not_live_object_writes_nothing(NotLive::NoHeader).await;
    }

    #[tokio::test]
    async fn recount_of_an_object_with_a_partial_header_writes_nothing() {
        recount_of_a_not_live_object_writes_nothing(NotLive::PartialHeader).await;
    }

    #[tokio::test]
    async fn recount_of_a_superseded_object_writes_nothing() {
        recount_of_a_not_live_object_writes_nothing(NotLive::Superseded).await;
    }

    #[tokio::test]
    async fn recount_of_an_object_scheduled_for_gc_writes_nothing() {
        recount_of_a_not_live_object_writes_nothing(NotLive::GcScheduled).await;
    }

    // ── The row ceiling ───────────────────────────────────────────────────

    /// The recount reads at most half the atomic snapshot read's item limit
    /// in rows (two reads per row), inclusive; above it, it answers
    /// `too_large` and writes nothing.
    ///
    /// The limit is 3 first, so the ceiling is the integer half, 1, and the
    /// two-row object is refused; then 4, a ceiling of exactly 2, and the
    /// same object is repaired — the boundary is inclusive.
    #[tokio::test]
    async fn recount_above_its_row_ceiling_answers_too_large_and_writes_nothing() {
        let tv = TestValkey::setup().await;
        let metrics = metrics_scrape_only("nvf-recount-ceiling");
        let hash = hex_of(0x72);
        seed_header(&tv, &[("seq", "1")]).await;
        seed_row(&tv, Family::Field, "a", &[("hash", &hash), ("size", "1")]).await;
        seed_row(
            &tv,
            Family::TimeSample,
            "2.5",
            &[("hash", &hash), ("size", "1")],
        )
        .await;
        seed_index(&tv, &[(&hash, "1")]).await;

        let mut client = tv.client().clone();
        client.snapshot_atomic_read_max_items = 3;
        let outcome = recount(&client, &tv, &metrics)
            .await
            .expect("the recount must answer");
        assert_eq!(
            outcome,
            RecountOutcome::TooLarge { rows: 2 },
            "fields and time samples both count toward the ceiling"
        );
        assert_eq!(
            index(&tv, OID).await,
            HashMap::from([(hash.clone(), "1".to_owned())]),
            "a refused recount writes nothing to the index"
        );
        assert!(
            !header(&tv)
                .await
                .contains_key(HEADER_FIELD_EXT_INDEX_REPAIRED_MS),
            "and nothing to the header"
        );
        assert_eq!(recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_TOO_LARGE), 1);

        client.snapshot_atomic_read_max_items = 4;
        let outcome = recount(&client, &tv, &metrics)
            .await
            .expect("the recount must answer");
        assert!(
            matches!(
                outcome,
                RecountOutcome::Repaired {
                    under: 1,
                    over: 0,
                    malformed_rows: 0,
                    ..
                }
            ),
            "at a ceiling equal to the row count the object is recounted; got {outcome:?}"
        );
        assert_eq!(
            index(&tv, OID).await,
            HashMap::from([(hash, "2".to_owned())])
        );
    }

    // ── The counting rule ─────────────────────────────────────────────────

    /// A dual-shape row — an inline `value` beside a stale `hash` column —
    /// holds no hash, so the recount neither counts it nor writes a field for
    /// it.
    ///
    /// Two objects. The first holds the hash on a field row and a time-sample
    /// row and carries a dual-shape field row as well, with an index of 3 —
    /// the RAW count of the `hash` column. The rule's answer is 2: an
    /// over-count by one, repaired downward, with the time sample counted.
    /// The second holds ONLY a dual-shape row and no index: raw reading says
    /// one reference, the rule says none, so the recount is `clean` and
    /// writes nothing — the answer the reconciler's raw-triggered audit gets
    /// for every legacy row.
    #[tokio::test]
    async fn recount_counts_a_dual_shape_row_as_holding_no_hash() {
        let tv = TestValkey::setup().await;
        let metrics = metrics_scrape_only("nvf-recount-dual-shape");
        let hash = hex_of(0x73);
        seed_header(&tv, &[("seq", "1")]).await;
        seed_row(&tv, Family::Field, "a", &[("hash", &hash), ("size", "1")]).await;
        seed_row(
            &tv,
            Family::TimeSample,
            "2.5",
            &[("hash", &hash), ("size", "1")],
        )
        .await;
        seed_row(
            &tv,
            Family::Field,
            "b",
            &[("value", "inline"), ("hash", &hash), ("size", "1")],
        )
        .await;
        seed_index(&tv, &[(&hash, "3")]).await;

        let outcome = recount(tv.client(), &tv, &metrics)
            .await
            .expect("the recount must answer");
        assert_eq!(
            outcome,
            RecountOutcome::Repaired {
                under: 0,
                over: 1,
                malformed_rows: 0,
                exemplars: vec![RecountExemplar {
                    hash_hex: hash.clone(),
                    stored: Some("3".to_owned()),
                    counted: 2,
                }],
            },
            "the dual-shape row is not a reference: 2, not the raw 3"
        );
        assert_eq!(
            index(&tv, OID).await,
            HashMap::from([(hash.clone(), "2".to_owned())])
        );
        let stamp = header(&tv)
            .await
            .remove(HEADER_FIELD_EXT_INDEX_REPAIRED_MS)
            .expect("a repair stamps the header");
        assert!(
            stamp.parse::<u64>().is_ok(),
            "the stamp is a millisecond time; got {stamp:?}"
        );

        // ── An object holding only a dual-shape row ──────────────────────────
        let tv = TestValkey::setup().await;
        seed_header(&tv, &[("seq", "1")]).await;
        seed_row(
            &tv,
            Family::Field,
            "a",
            &[("value", "inline"), ("hash", &hash), ("size", "1")],
        )
        .await;
        let outcome = recount(tv.client(), &tv, &metrics)
            .await
            .expect("the recount must answer");
        assert_eq!(outcome, RecountOutcome::Clean { malformed_rows: 0 });
        assert!(index(&tv, OID).await.is_empty());
        assert!(
            !header(&tv)
                .await
                .contains_key(HEADER_FIELD_EXT_INDEX_REPAIRED_MS),
            "a clean recount writes nothing, the stamp included"
        );
    }

    /// A `hash` column that is not 64 lowercase hex digits is reported as
    /// malformed and never copied into the index, and an existing index
    /// field whose NAME is malformed is left exactly as it is.
    ///
    /// Copying a malformed column would mint an index field naming no blob;
    /// removing a malformed field would be a write the recount could not
    /// have counted. Both directions are pinned against one valid row, which
    /// is what makes the recount repair at all. Its only fix is a raise, so
    /// the script answers `REPAIRED` — but a recount that saw malformed rows
    /// is not authoritative, so it is RECORDED as `withheld`, not `repaired`.
    #[tokio::test]
    async fn recount_never_writes_a_malformed_hash_column_as_an_index_field() {
        let tv = TestValkey::setup().await;
        let metrics = metrics_scrape_only("nvf-recount-malformed");
        let valid = hex_of(0xAB);
        let uppercase = valid.to_uppercase();
        seed_header(&tv, &[("seq", "1")]).await;
        seed_row(&tv, Family::Field, "a", &[("hash", &valid), ("size", "1")]).await;
        seed_row(
            &tv,
            Family::Field,
            "b",
            &[("hash", "not-a-hash"), ("size", "1")],
        )
        .await;
        seed_row(
            &tv,
            Family::Field,
            "c",
            &[("hash", &uppercase), ("size", "1")],
        )
        .await;
        seed_index(&tv, &[("not-a-hex-field", "7")]).await;

        let outcome = recount(tv.client(), &tv, &metrics)
            .await
            .expect("the recount must answer");
        assert_eq!(
            outcome,
            RecountOutcome::Repaired {
                under: 1,
                over: 0,
                malformed_rows: 2,
                exemplars: vec![RecountExemplar {
                    hash_hex: valid.clone(),
                    stored: None,
                    counted: 1,
                }],
            },
            "the short and the uppercase column are both malformed; only the valid row counts"
        );
        assert_eq!(
            index(&tv, OID).await,
            HashMap::from([
                (valid, "1".to_owned()),
                ("not-a-hex-field".to_owned(), "7".to_owned()),
            ]),
            "no field for either malformed column, and the malformed field name untouched"
        );
        assert_eq!(
            (
                recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_WITHHELD),
                recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_REPAIRED)
            ),
            (1, 0),
            "a recount that saw malformed rows is recorded as `withheld`, whatever it answered"
        );
    }

    // ── Raises only, over a malformed row ────────────────────────────────

    /// H for the raises-only arms, and its uppercase spelling — a `hash`
    /// column the Rust decoder accepts (`hash_from_hex`) and the script
    /// reports as malformed. The byte is chosen so the hex has letters, or
    /// uppercasing would change nothing.
    const RAISES_ONLY_H: u8 = 0xC1;

    /// Over a malformed row, a recount whose every fix is a LOWERING writes
    /// nothing at all — no index field, and no header stamp — and answers
    /// `WITHHELD`.
    ///
    /// The index here is exactly what the decoder counts: H is held by two
    /// lowercase rows and one UPPERCASE row (3), and K only by an uppercase
    /// row (1). The script counts the lowercase rows alone, so it sees H
    /// over-counted (3 against 2) and K stored for a hash it counts no row
    /// for. Lowering H to 2 or removing K would under-count what the layer
    /// serves — the loss direction — so both are withheld. With nothing
    /// applied there is no write to stamp the header for either.
    #[tokio::test]
    async fn recount_over_a_malformed_row_withholds_every_lowering_and_writes_nothing() {
        let tv = TestValkey::setup().await;
        let metrics = metrics_scrape_only("nvf-recount-withhold-all");
        let h = hex_of(RAISES_ONLY_H);
        let k = hex_of(0xD4);
        seed_header(&tv, &[("seq", "1")]).await;
        for name in ["a", "b"] {
            seed_row(&tv, Family::Field, name, &[("hash", &h), ("size", "1")]).await;
        }
        seed_row(
            &tv,
            Family::Field,
            "c",
            &[("hash", &h.to_uppercase()), ("size", "1")],
        )
        .await;
        seed_row(
            &tv,
            Family::Field,
            "d",
            &[("hash", &k.to_uppercase()), ("size", "1")],
        )
        .await;
        seed_index(&tv, &[(&h, "3"), (&k, "1")]).await;
        let header_before = header(&tv).await;

        let outcome = recount(tv.client(), &tv, &metrics)
            .await
            .expect("the recount must answer");

        assert_eq!(
            outcome,
            RecountOutcome::Withheld {
                raised: 0,
                withheld: 2,
                malformed_rows: 2,
                exemplars: Vec::new(),
            },
            "both fixes are lowerings (H 3 → 2, K removed) and both are withheld"
        );
        assert_eq!(
            index(&tv, OID).await,
            HashMap::from([(h, "3".to_owned()), (k, "1".to_owned())]),
            "nothing lowered and nothing removed"
        );
        assert_eq!(
            header(&tv).await,
            header_before,
            "with every fix withheld there is no write at all — no `ext_index_repaired_ms`"
        );
        assert_eq!(recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_WITHHELD), 1);
    }

    /// Over a malformed row, a recount with one RAISE and one LOWERING
    /// applies the raise, stamps the header for it, withholds the lowering
    /// and answers `WITHHELD`.
    ///
    /// H is held by two lowercase rows and one uppercase row and stored at
    /// 3 (the script sees 3 against 2: a lowering, withheld). J is held by
    /// one lowercase row and has no field (a raise from 0 to 1, applied —
    /// a raise can only move the index toward what every reader counts).
    #[tokio::test]
    async fn recount_over_a_malformed_row_applies_raises_and_withholds_the_lowering() {
        let tv = TestValkey::setup().await;
        let metrics = metrics_scrape_only("nvf-recount-withhold-some");
        let h = hex_of(RAISES_ONLY_H);
        let j = hex_of(0xD5);
        seed_header(&tv, &[("seq", "1")]).await;
        for name in ["a", "b"] {
            seed_row(&tv, Family::Field, name, &[("hash", &h), ("size", "1")]).await;
        }
        seed_row(
            &tv,
            Family::Field,
            "c",
            &[("hash", &h.to_uppercase()), ("size", "1")],
        )
        .await;
        seed_row(&tv, Family::Field, "d", &[("hash", &j), ("size", "1")]).await;
        seed_index(&tv, &[(&h, "3")]).await;

        let outcome = recount(tv.client(), &tv, &metrics)
            .await
            .expect("the recount must answer");

        assert_eq!(
            outcome,
            RecountOutcome::Withheld {
                raised: 1,
                withheld: 1,
                malformed_rows: 1,
                exemplars: vec![RecountExemplar {
                    hash_hex: j.clone(),
                    stored: None,
                    counted: 1,
                }],
            },
            "J raised (the only exemplar), H's lowering withheld"
        );
        assert_eq!(
            index(&tv, OID).await,
            HashMap::from([(h, "3".to_owned()), (j, "1".to_owned())]),
            "the raise is applied; the lowering is not"
        );
        assert!(
            header(&tv)
                .await
                .contains_key(HEADER_FIELD_EXT_INDEX_REPAIRED_MS),
            "an applied raise stamps the header, which is what cancels an in-flight txn commit \
             planned from the old count"
        );
        assert_eq!(
            sample_value(
                &render(&metrics),
                "live_edit_ext_index_drift_total",
                &[("site", "reconcile"), ("direction", "under")],
            ),
            Some(1.0),
            "the applied raise still counts as upward drift"
        );
    }

    /// At the reconcile site, a recount that saw a malformed row records
    /// `withheld` even when the script answered `CLEAN` or `REPAIRED` (no
    /// lowering was needed, so nothing was withheld).
    ///
    /// H is held by two lowercase rows and one uppercase row. Stored at 2,
    /// the script's count matches and it answers `CLEAN { malformed: 1 }`;
    /// stored at 1, it raises to 2 and answers `REPAIRED { malformed: 1 }`.
    /// Neither count includes the uppercase row the decoder counts, so
    /// neither is a repair an operator may read as benign.
    #[tokio::test]
    async fn recount_over_a_malformed_row_records_withheld_whatever_it_answers() {
        let tv = TestValkey::setup().await;
        let metrics = metrics_scrape_only("nvf-recount-withheld-metric");
        let h = hex_of(RAISES_ONLY_H);
        seed_header(&tv, &[("seq", "1")]).await;
        for name in ["a", "b"] {
            seed_row(&tv, Family::Field, name, &[("hash", &h), ("size", "1")]).await;
        }
        seed_row(
            &tv,
            Family::Field,
            "c",
            &[("hash", &h.to_uppercase()), ("size", "1")],
        )
        .await;

        seed_index(&tv, &[(&h, "2")]).await;
        let clean = recount(tv.client(), &tv, &metrics)
            .await
            .expect("the recount must answer");
        seed_index(&tv, &[(&h, "1")]).await;
        let repaired = recount(tv.client(), &tv, &metrics)
            .await
            .expect("the recount must answer");

        assert_eq!(
            (&clean, &repaired),
            (
                &RecountOutcome::Clean { malformed_rows: 1 },
                &RecountOutcome::Repaired {
                    under: 1,
                    over: 0,
                    malformed_rows: 1,
                    exemplars: vec![RecountExemplar {
                        hash_hex: h.clone(),
                        stored: Some("1".to_owned()),
                        counted: 2,
                    }],
                }
            ),
            "fixture precondition: the two answers this test is about"
        );
        assert_eq!(
            (
                recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_WITHHELD),
                recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_CLEAN),
                recounts(&metrics, EXT_INDEX_RECOUNT_OUTCOME_REPAIRED)
            ),
            (2, 0, 0),
            "both are recorded as `withheld`, neither as the benign cell its tag names"
        );
    }

    // ── The header write cancels an in-flight txn commit ─────────────────

    /// One deadline for the whole disturbed commit, retry included. The
    /// healthy path is a handful of round trips.
    const COMMIT_BUDGET: Duration = Duration::from_secs(45);

    fn external_field(key: &str, hash_byte: u8, set_order: u64) -> OwnedDelta {
        OwnedDelta {
            set_fields: vec![SetField {
                section_id: 1,
                key_name: key.to_owned(),
                value_or_ext_hash: vec![hash_byte; 32],
                ext_value_size: 1_024,
                ext_value_index: 1,
                set_order,
            }],
            ..Default::default()
        }
    }

    async fn pre_store(ext: &ValkeyExternalValueStore, hash_byte: u8) {
        ext.store_values(&[(&[hash_byte; 32], b"nvf-payload".as_slice())])
            .await
            .expect("the pre-flight store must succeed");
    }

    /// A repair landing between a txn commit's index read and its `EXEC`
    /// cancels that `EXEC`, so the commit retries and plans from the repaired
    /// index instead of overwriting it with counts — and a release — planned
    /// from the old one.
    ///
    /// The layer is built by the real txn commit path with an under-counted
    /// index: R1 := H, `obj:ext` deleted (the pre-index shape), R2 := H, so
    /// H reads 1 while two rows hold it. The commit under test rewrites
    /// R2 := K on the non-snapshot branch, which reads `obj:ext` before
    /// `MULTI` and plans H 1 → 0: an `HDEL` and a `rewrite` release of H.
    /// The pre-dispatch hook runs the recount in exactly that window.
    ///
    /// The txn path WATCHes the header, stream and snapshot fence only — not `obj:ext` — so
    /// what cancels the `EXEC` is the repair's header stamp. Without it the
    /// `EXEC` commits, deletes H's freshly repaired count and releases H
    /// while R1 still holds it.
    ///
    /// The hook counts its invocations: it fires once per attempt that
    /// reaches dispatch, so two means the first attempt's `EXEC` was
    /// cancelled and retried, and one means it committed over the repair.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_repair_cancels_an_in_flight_txn_commit_planned_from_the_old_index() {
        const LAYER: &str = "nvf-recount-watch";
        const H: u8 = 0x74;
        const K: u8 = 0x75;
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only(LAYER));
        let ext = Arc::new(ValkeyExternalValueStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(LAYER, &format!("/{LAYER}"), "nvf-user")
            .await
            .expect("session creation must succeed");
        let plain = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Txn)
            .with_metrics(metrics.clone());
        let state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..Default::default()
        };
        let oid = plain
            .create_layer(&session_id, LAYER, "nvf-user", &state, ext.clone())
            .await
            .expect("create_layer must succeed");
        let oid_str = oid.to_string();

        pre_store(&ext, H).await;
        plain
            .update_layer_relayed(
                &session_id,
                LAYER,
                oid,
                Some(external_field("color", H, 0)),
                "nvf-writer",
                1,
            )
            .await
            .expect("R1 := H must commit");
        let _: u64 = tv
            .client()
            .pool
            .del(tv.key_schema().obj_ext_index(&oid_str))
            .await
            .expect("deleting obj:ext must succeed");
        pre_store(&ext, H).await;
        plain
            .update_layer_relayed(
                &session_id,
                LAYER,
                oid,
                Some(external_field("roughness", H, 1)),
                "nvf-writer",
                2,
            )
            .await
            .expect("R2 := H must commit");
        assert_eq!(
            index(&tv, &oid_str).await,
            HashMap::from([(hex_of(H), "1".to_owned())]),
            "fixture precondition: H under-counted at 1 while two rows hold it"
        );

        let dispatches = Arc::new(AtomicU32::new(0));
        let hook: WatchWindowHook = {
            let dispatches = dispatches.clone();
            let client = tv.client().clone();
            let keys = tv.key_schema().clone();
            let oid_str = oid_str.clone();
            Arc::new(move |_guarded: &fred::clients::Client| {
                let dispatches = dispatches.clone();
                let client = client.clone();
                let keys = keys.clone();
                let oid_str = oid_str.clone();
                async move {
                    if dispatches.fetch_add(1, Ordering::SeqCst) > 0 {
                        return;
                    }
                    let outcome = recount_ext_index(
                        &client,
                        client.pool.next(),
                        POOL_LABEL_DATA,
                        &keys,
                        &oid_str,
                        EXT_INDEX_RECOUNT_SITE_RECONCILE,
                        None,
                    )
                    .await
                    .expect("the in-window recount must answer");
                    assert!(
                        matches!(outcome, RecountOutcome::Repaired { under: 1, .. }),
                        "the in-window recount must repair H upward; got {outcome:?}"
                    );
                }
                .boxed()
            })
        };
        let disturbed = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Txn)
            .with_metrics(metrics.clone())
            .with_pre_dispatch_hook_for_test(hook);

        pre_store(&ext, K).await;
        tokio::time::timeout(
            COMMIT_BUDGET,
            disturbed.update_layer_relayed(
                &session_id,
                LAYER,
                oid,
                Some(external_field("roughness", K, 2)),
                "nvf-writer",
                3,
            ),
        )
        .await
        .expect("the disturbed commit did not settle within its budget")
        .expect("R2 := K must commit on its retry");

        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            2,
            "the first attempt's EXEC must be CANCELLED by the repair's header write and \
             retried. One dispatch means it committed counts planned from the pre-repair index"
        );
        assert_eq!(
            index(&tv, &oid_str).await,
            HashMap::from([(hex_of(H), "1".to_owned()), (hex_of(K), "1".to_owned())]),
            "the retry planned from the repaired count: H 2 → 1 (R1 still holds it), K 0 → 1"
        );
        let pending: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().ext_pending())
            .await
            .expect("SMEMBERS pending must succeed");
        assert!(
            pending.is_empty(),
            "no release may be owed: R1 still holds H, so a `rewrite` member here is the \
             premature release the cancelled attempt had planned; got {pending:?}"
        );
        assert_eq!(
            sample_value(
                &render(&metrics),
                "live_edit_ext_ref_released_total",
                &[("reason", "rewrite")],
            ),
            Some(0.0)
        );
    }
}
