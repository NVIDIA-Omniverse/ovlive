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

//! Reading a per-object stream's high-water: the `last-generated-id` a
//! restore folds into its sequence floor (SDD SR-2.5, §22.6 step 4, b.4et).
//!
//! The pieces both restore paths share:
//!
//! - [`STREAM_HIGH_WATER_PROBE_SCRIPT`] and [`probe_once`]: one `EVALSHA` of
//!   `stream_high_water_probe.lua`, which projects `XINFO STREAM … FULL COUNT
//!   1` down to `nil` or the bare `last-generated-id` server-side, so the
//!   reply is O(1) in layer size (the script header has the cost argument);
//! - [`key_type_once`]: one `TYPE` of the stream key, which un-conflates the
//!   script's single `nil` into no key, a stream, or a foreign value;
//! - [`parse_stream_id_seq`], [`MAX_PLAUSIBLE_HIGH_WATER_GAP`] and the pure
//!   [`classify`]: an entry id's sequence and whether it is plausible above a
//!   checkpoint's `snap_seq`.
//!
//! The on-access recovery hook (`checkpoint::recovery`) calls these from its
//! own seams, retry wrappers and adjudication, which stay there with its
//! verified stale-stream `UNLINK`. The restore Job calls [`read_stream_floor`]:
//! its own adjudication, which removes nothing — the Job's restore writer
//! `UNLINK`s the stream only inside its header transaction
//! (`storage::layer::dr_restore`), so a Job killed before the header still
//! finds the stream and its high-water.
//!
//! Every function here reads only.

use std::future::Future;
use std::sync::LazyLock;

use fred::interfaces::KeysInterface;
use fred::types::scripts::Script;
use futures::FutureExt;

use crate::error::ServiceError;
use crate::storage::client::{map_snapshot_load_err, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};

/// Compiled Lua script backing [`probe_once`]: returns the stream key's
/// `last-generated-id`, or `nil` for a missing key, a wrong-type key, or a
/// reply carrying no id.
///
/// The script lives alongside its siblings in `src/storage/scripts/` (repo
/// convention: that directory is the home for every Lua script). The
/// plausibility gate that interprets the id lives on the Rust side
/// ([`classify`]).
///
/// `Script::from_lua` is infallible — it just hashes the source. The first
/// `evalsha_with_reload` call sends `SCRIPT LOAD` on `NOSCRIPT` before
/// retrying.
pub(crate) static STREAM_HIGH_WATER_PROBE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/stream_high_water_probe.lua")));

/// Largest gap above a checkpoint's `snap_seq` that a restore will accept as
/// a genuine sequence number rather than a foreign ID.
///
/// A legitimate gap is the commit count that landed between the last
/// checkpoint and the crash, on **one object**. Even at a wildly generous
/// 10,000 commits/sec sustained for a full 24 hours of checkpointer outage
/// — orders of magnitude beyond what a per-object serialized OMPF-parse +
/// Lua-EVAL pipeline can do — that is under 9e8. One billion is therefore
/// generous for the true signal.
///
/// It is also unreachable by the noise it exists to reject. Valkey mints an
/// auto-ID (`*`) from unix-milliseconds, which has been above 1e12 since
/// 2001 and only rises, so a timestamp-shaped ID clears this bound by three
/// orders of magnitude and always will. The two populations do not overlap
/// and cannot start overlapping with time.
pub(crate) const MAX_PLAUSIBLE_HIGH_WATER_GAP: u64 = 1_000_000_000;

/// Parse the sequence component (the `ms` field, which this service uses as
/// the commit sequence number) out of a Valkey Stream entry ID.
///
/// Entry IDs are `<ms>-<seq>`; both commit paths `XADD` at `{sequence_num}-0`,
/// so the leading component IS the layer's sequence number. Returns `None`
/// for any ID that does not parse as `<u64>-<...>`, including a bare `<u64>`
/// with no separator (server replies always carry both components, so a
/// missing separator means the reply shape is not what we assumed).
///
/// Local to the restore paths by design: `src/storage/notifications.rs`
/// owns the subscriber-side cursor handling and its internals are not part
/// of this module's contract surface.
pub(crate) fn parse_stream_id_seq(entry_id: &str) -> Option<u64> {
    entry_id.split_once('-')?.0.parse::<u64>().ok()
}

/// What one probed `last-generated-id` says about a restore at `snap_seq`
/// ([`classify`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProbeReading {
    /// A sequence at most [`MAX_PLAUSIBLE_HIGH_WATER_GAP`] above `snap_seq`:
    /// a genuine high-water.
    Plausible(u64),
    /// A sequence further above `snap_seq`: overwhelmingly an auto-ID
    /// wall-clock timestamp from a `deleted` tombstone. Not a high-water.
    Implausible(u64),
    /// An id that does not parse as `<u64>-<...>`.
    Unparseable,
}

/// Classify a probed `last-generated-id` for a restore at `snap_seq`: the
/// plausibility gate, cause-agnostic by design.
pub(crate) fn classify(raw_id: &str, snap_seq: u64) -> ProbeReading {
    match parse_stream_id_seq(raw_id) {
        Some(seq) if seq <= snap_seq.saturating_add(MAX_PLAUSIBLE_HIGH_WATER_GAP) => {
            ProbeReading::Plausible(seq)
        }
        Some(seq) => ProbeReading::Implausible(seq),
        None => ProbeReading::Unparseable,
    }
}

/// What a `TYPE le:{oid}:obj:stream` round-trip says about the key the
/// high-water probe could not read a usable sequence out of (b.4et).
///
/// The three arms are the whole point: the probe script's single `nil` covers
/// all of them, and they demand opposite actions. Folding any two back
/// together reintroduces the defect.
#[derive(Debug)]
pub(crate) enum StreamKeyType {
    /// `"none"` — no key at all. The stream did not survive the state loss;
    /// the restore floor is `snap_seq` alone. The dominant production shape.
    Absent,
    /// `"stream"` — the key is there and is ours, and we could not read its
    /// top. Evidence unknown; the restore must not proceed.
    Stream,
    /// Any other Valkey type — a foreign value on the `le:` keyspace. Carries
    /// the reported type so the WARN can name it.
    Foreign(String),
}

/// One high-water probe attempt: the `EVALSHA` on the data pool, classified
/// for a retry driver by [`map_snapshot_load_err`] — transient kinds become
/// [`ServiceError::Aborted`] (retriable), everything else `Internal`
/// (terminal).
///
/// Single-key `EVALSHA`. The script returns `nil` for a missing key, a
/// wrong-type key, or a reply without `last-generated-id`; otherwise the ID
/// string. `evalsha_with_reload` handles `NOSCRIPT` transparently.
///
/// Not an `async fn`, nor is [`key_type_once`]: the on-access restore awaits
/// both inside every `UpdateLayer` call's future, where an `async fn`'s own
/// copy of its arguments would be held beside the command's.
pub(crate) fn probe_once<'a>(
    client: &'a ValkeyClient,
    stream_key: &str,
) -> impl Future<Output = Result<Option<String>, ServiceError>> + 'a {
    STREAM_HIGH_WATER_PROBE_SCRIPT
        .evalsha_with_reload(
            client.pool.next(),
            vec![stream_key.to_string()],
            Vec::<String>::new(),
        )
        .map(move |reply| {
            reply.map_err(|e| {
                map_snapshot_load_err(
                    FredErrOrigin::client(client, POOL_LABEL_DATA),
                    "EVALSHA stream_high_water_probe",
                    e,
                )
            })
        })
}

/// One `TYPE` attempt on the stream key, classified as [`probe_once`] is.
///
/// Deliberately NOT `storage::lua::KEYSPACE_TYPE_PROBE_SCRIPT`: that
/// script answers "is this key the type I am about to write", which folds
/// `none` and the expected type into one "fine" verdict. Here `none` and
/// `stream` are the two arms that must NOT be folded — one proceeds, the
/// other refuses.
pub(crate) fn key_type_once<'a>(
    client: &'a ValkeyClient,
    stream_key: &'a str,
) -> impl Future<Output = Result<StreamKeyType, ServiceError>> + 'a {
    client.pool.r#type::<String, _>(stream_key).map(move |ty| {
        let ty = ty.map_err(|e| {
            map_snapshot_load_err(
                FredErrOrigin::client(client, POOL_LABEL_DATA),
                "TYPE obj:stream",
                e,
            )
        })?;
        Ok(match ty.as_str() {
            "none" => StreamKeyType::Absent,
            "stream" => StreamKeyType::Stream,
            other => StreamKeyType::Foreign(other.to_string()),
        })
    })
}

/// What the restore Job found at an object's stream key ([`read_stream_floor`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamFloor {
    /// A stream whose `last-generated-id` is a plausible sequence above or at
    /// the restore's `snap_seq` — the floor is `max(snap_seq, high_water)` —
    /// or below it.
    Plausible {
        /// The stream's high-water.
        high_water: u64,
    },
    /// A stream whose `last-generated-id` is implausibly far above `snap_seq`
    /// (a `deleted` tombstone's auto-ID): the floor is `snap_seq`.
    Implausible {
        /// The id's sequence component, for the log.
        high_water: u64,
    },
    /// No key: the floor is `snap_seq`.
    Absent,
    /// A non-stream value of this type at the stream key: the floor is
    /// `snap_seq` (PD-3: the Job's header transaction removes it).
    Foreign(String),
}

impl StreamFloor {
    /// The floor this reading gives a restore at `snap_seq`.
    pub fn floor(&self, snap_seq: u64) -> u64 {
        match self {
            Self::Plausible { high_water } => snap_seq.max(*high_water),
            Self::Implausible { .. } | Self::Absent | Self::Foreign(_) => snap_seq,
        }
    }

    /// The plausible high-water, when there is one.
    pub fn high_water(&self) -> Option<u64> {
        match self {
            Self::Plausible { high_water } => Some(*high_water),
            _ => None,
        }
    }

    /// Whether a key is at the stream key at all.
    pub fn key_present(&self) -> bool {
        !matches!(self, Self::Absent)
    }
}

/// The restore Job's reading of object `object_id`'s stream key for a
/// restore at `snap_seq`: one [`probe_once`], then, when it gives no
/// plausible or implausible sequence, one [`key_type_once`]:
///
/// | Probe | `TYPE` | Reading |
/// |---|---|---|
/// | a plausible id | — | [`StreamFloor::Plausible`] |
/// | an implausible id | — | [`StreamFloor::Implausible`] |
/// | `nil`, unparseable or an error | `none` | [`StreamFloor::Absent`] |
/// | `nil`, unparseable or an error | `stream` | `Err` (unreadable) |
/// | `nil`, unparseable or an error | another type | [`StreamFloor::Foreign`] |
///
/// One attempt, no retry: the Job retries the step itself. An unreadable
/// stream is [`ServiceError::Aborted`], so the caller retries it; a failed
/// `TYPE` is its own error. Writes nothing.
pub async fn read_stream_floor(
    client: &ValkeyClient,
    stream_key: &str,
    snap_seq: u64,
) -> Result<StreamFloor, ServiceError> {
    let unreadable = match probe_once(client, stream_key).await {
        Ok(Some(raw_id)) => match classify(&raw_id, snap_seq) {
            ProbeReading::Plausible(high_water) => {
                return Ok(StreamFloor::Plausible { high_water });
            }
            ProbeReading::Implausible(high_water) => {
                return Ok(StreamFloor::Implausible { high_water });
            }
            ProbeReading::Unparseable => format!("unparseable last-generated-id `{raw_id}`"),
        },
        Ok(None) => "the probe script returned nil".to_owned(),
        Err(e) => format!("EVALSHA stream_high_water_probe failed: {e}"),
    };
    match key_type_once(client, stream_key).await? {
        StreamKeyType::Absent => Ok(StreamFloor::Absent),
        StreamKeyType::Foreign(actual) => Ok(StreamFloor::Foreign(actual)),
        StreamKeyType::Stream => Err(ServiceError::Aborted(format!(
            "the stream key {stream_key} is a stream whose last-generated-id could not be read: \
             {unreadable}"
        ))),
    }
}

/// Unit tests for [`parse_stream_id_seq`], the one piece of reply decoding
/// the high-water probe still does on the client side.
///
/// # Why only this function
///
/// The probe used to walk a client-side `XINFO STREAM` reply with three
/// helpers (`find_member` / `last_generated_id` / `last_entry_payload`),
/// each with RESP2, RESP3 and malformed-reply arms that no integration test
/// could reach — hence a large unit-test block here. That walking now
/// happens **server-side** inside [`STREAM_HIGH_WATER_PROBE_SCRIPT`], which
/// projects the reply down to `nil` or the bare `last-generated-id` string
/// before it crosses the wire. Wire-shape decoding is no longer this
/// module's concern at all: scripts run in RESP2 conversion mode regardless
/// of the connection's negotiated protocol, so the RESP3 divergence the old
/// tests existed to pin cannot occur, and fred decodes the scalar reply into
/// `Option<String>` directly.
///
/// The behaviour those tests covered is now pinned end-to-end by the gate
/// tests in `tests/integration/checkpoint/e2e_object_id_preservation_test.rs`
/// (`seq_floor_folds_in_surviving_stream_high_water`,
/// `seq_floor_ignores_tombstoned_stream_with_auto_id`,
/// `seq_floor_ignores_implausible_stream_high_water`), which drive a real
/// Valkey and therefore exercise the script's own extraction rather than a
/// hand-built reply value.
///
/// # One decode nuance deliberately left untested
///
/// fred maps **both** `Value::Null` and a zero-length bulk string to `None`
/// when decoding into `Option<String>`, so the probe cannot distinguish "the
/// script returned `nil`" from "the script returned an empty string". That is
/// inert: the script's only non-`nil` return is a server-minted
/// `last-generated-id`, which is never empty, and both fold to the same caller
/// behaviour anyway — since b.4et that is "hand the case to the `TYPE`
/// round-trip", which then decides between falling the restore floor back to
/// `snap_seq` (`none` / a foreign type) and refusing the restore outright
/// (`stream`); the ambiguity is invisible either way, because the verdict
/// turns on the key's state and not on which flavour of unreadable produced
/// it. Pinning it would test fred rather than this module. Noted here so a
/// future script edit that can return an empty value knows the ambiguity
/// exists.
#[cfg(test)]
mod stream_id_parsing_tests {
    use super::*;

    #[test]
    fn parse_stream_id_seq_well_formed_id_returns_ms_component() {
        assert_eq!(parse_stream_id_seq("42-0"), Some(42));
        assert_eq!(parse_stream_id_seq("42-7"), Some(42));
        // Both commit paths XADD at `{seq}-0`, so the ms component IS the
        // sequence number — including at the f64 ceiling apply guards on.
        assert_eq!(
            parse_stream_id_seq("9007199254740991-0"),
            Some(9_007_199_254_740_991)
        );
        // The wall-clock auto-ID shape the plausibility gate exists to
        // reject still parses — rejection is the gate's job, not the
        // parser's.
        assert_eq!(
            parse_stream_id_seq("1790000000000-0"),
            Some(1_790_000_000_000)
        );
    }

    #[test]
    fn parse_stream_id_seq_bare_id_without_separator_returns_none() {
        // Server replies always carry both components, so a missing
        // separator means the reply shape is not what the probe assumed —
        // treat it as unparseable rather than guessing.
        assert_eq!(parse_stream_id_seq("42"), None);
    }

    #[test]
    fn parse_stream_id_seq_non_numeric_ms_returns_none() {
        assert_eq!(parse_stream_id_seq("abc-0"), None);
        assert_eq!(parse_stream_id_seq("4 2-0"), None);
        // A leading `-` splits into an empty ms component.
        assert_eq!(parse_stream_id_seq("-1-0"), None);
    }

    #[test]
    fn parse_stream_id_seq_accepts_rusts_leading_plus_sign() {
        // Not a shape any server emits, pinned because it is the one
        // "obviously malformed" input the parser does NOT reject:
        // `u64::from_str` accepts a leading `+`. The parser is therefore not
        // the defense against a weird-looking ID — the plausibility gate,
        // [`classify`] in this module, is: the on-access probe and the restore
        // Job both run it on the parsed value, however it was spelled.
        assert_eq!(parse_stream_id_seq("+42-0"), Some(42));
    }

    #[test]
    fn parse_stream_id_seq_empty_string_returns_none() {
        assert_eq!(parse_stream_id_seq(""), None);
        assert_eq!(parse_stream_id_seq("-"), None);
    }
}

/// U4: the plausibility gate both restore paths share ([`classify`]) and the
/// floor the restore Job takes from each reading ([`StreamFloor::floor`]).
/// The readings themselves, off a real stream key (none, a stream, a foreign
/// value), are the restore Job's integration tests
/// (`tests/integration/dr/restore_write_test.rs`, W3).
#[cfg(test)]
mod classify_tests {
    use super::*;

    /// A sequence at most a billion above `snap_seq` is plausible, below it
    /// included; one further above is a `deleted` tombstone's auto-ID.
    #[test]
    fn classify_is_plausible_up_to_the_gap_and_implausible_beyond_it() {
        assert_eq!(classify("40-0", 100), ProbeReading::Plausible(40));
        assert_eq!(classify("140-0", 100), ProbeReading::Plausible(140));
        let edge = 100 + MAX_PLAUSIBLE_HIGH_WATER_GAP;
        assert_eq!(
            classify(&format!("{edge}-0"), 100),
            ProbeReading::Plausible(edge)
        );
        assert_eq!(
            classify(&format!("{}-0", edge + 1), 100),
            ProbeReading::Implausible(edge + 1)
        );
        assert_eq!(
            classify("1790000000000-0", 100),
            ProbeReading::Implausible(1_790_000_000_000),
            "a wall-clock auto-ID"
        );
    }

    /// An id that does not parse is neither: the caller asks `TYPE`.
    #[test]
    fn classify_an_unparseable_id_is_unparseable() {
        for raw in ["", "42", "abc-0", "-1-0"] {
            assert_eq!(classify(raw, 100), ProbeReading::Unparseable, "{raw:?}");
        }
    }

    /// The floor a reading gives: `max(snap_seq, high_water)` for a
    /// plausible high-water, `snap_seq` for every other reading.
    #[test]
    fn stream_floor_raises_snap_seq_only_by_a_plausible_high_water() {
        let plausible = |high_water| StreamFloor::Plausible { high_water };
        assert_eq!(plausible(140).floor(100), 140);
        assert_eq!(plausible(60).floor(100), 100);
        assert_eq!(plausible(140).high_water(), Some(140));
        let implausible = StreamFloor::Implausible {
            high_water: 1_790_000_000_000,
        };
        assert_eq!(implausible.floor(100), 100);
        assert_eq!(implausible.high_water(), None);
        assert_eq!(StreamFloor::Absent.floor(100), 100);
        assert!(!StreamFloor::Absent.key_present());
        let foreign = StreamFloor::Foreign("string".to_owned());
        assert_eq!(foreign.floor(100), 100);
        assert!(foreign.key_present());
    }
}
