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

//! The refusal detail a status carries, decoded as a client decodes it, and
//! the assertions every refusal test shares.
//!
//! A refusal that applied nothing — a load-shedding or shutdown refusal, or a
//! `CreateLayer` / `UpdateLayer` refused before its commit
//! (`UPSTREAM_UNAVAILABLE`) — is `UNAVAILABLE` with its usual message plus, in
//! `grpc-status-details-bin`, a `google.rpc.ErrorInfo` {reason, domain
//! `nvidia.omniverse.live.v1alpha`, metadata} and a `google.rpc.RetryInfo`
//! {retry_delay} drawn from the reason's range. Every other `UNAVAILABLE`
//! carries no detail at all. The
//! assertions here decode the wire form with `tonic_types::StatusExt`, the way
//! a gRPC client reads it, rather than comparing the service's own builder
//! output with itself.
//!
//! The size refusals of `CreateLayer` / `UpdateLayer` — a delta of more than
//! 99,999 entries (`DELTA_TOO_LARGE`), a `ReorderChildren` list of more than
//! 100,000 ids (`CHILDREN_LIST_TOO_LONG`) — are `INVALID_ARGUMENT` with an
//! `ErrorInfo` and no `RetryInfo`. Their assertions
//! ([`assert_delta_too_large`], [`assert_children_list_too_long`]) decode
//! `grpc-status-details-bin` by hand, as the C++ client library does (what it
//! reads is recorded in `docs-internal/sdd.md` §18.1 *Size refusals*), so the
//! wire shape that client depends on is what is checked.

use std::time::Duration;

use live_edit::error::{
    refusal_metadata, refusal_reason, RefusalDetail, RetryDelayRange, ServiceError,
    CAPACITY_RETRY_DELAY, REFUSAL_DETAIL_DOMAIN, SHUTDOWN_RETRY_DELAY,
    UPSTREAM_UNAVAILABLE_RETRY_DELAY,
};
use tonic::{Code, Status};
use tonic_types::StatusExt;

/// A status's refusal detail as a client decodes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireRefusal {
    pub reason: String,
    pub domain: String,
    /// `ErrorInfo.metadata`, sorted by key.
    pub metadata: Vec<(String, String)>,
    /// `RetryInfo.retry_delay`; `None` when the status carries no `RetryInfo`.
    pub retry_delay: Option<Duration>,
}

/// Decode `status`'s refusal detail; `None` when it carries no `ErrorInfo`.
pub fn decode(status: &Status) -> Option<WireRefusal> {
    let info = status.get_details_error_info()?;
    let mut metadata: Vec<(String, String)> = info.metadata.into_iter().collect();
    metadata.sort();
    Some(WireRefusal {
        reason: info.reason,
        domain: info.domain,
        metadata,
        retry_delay: status.get_details_retry_info().and_then(|r| r.retry_delay),
    })
}

fn owned(metadata: &[(&str, &str)]) -> Vec<(String, String)> {
    let mut owned: Vec<(String, String)> = metadata
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    owned.sort();
    owned
}

/// Assert `status` is `UNAVAILABLE` carrying the refusal detail `reason` in
/// the service's domain, with exactly `metadata`, and a `retry_delay` inside
/// `range`.
#[track_caller]
pub fn assert_refusal_detail(
    status: &Status,
    reason: &str,
    metadata: &[(&str, &str)],
    range: RetryDelayRange,
) {
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    let wire = decode(status).unwrap_or_else(|| {
        panic!("the refusal must carry an ErrorInfo ({reason}) in grpc-status-details-bin; got {status:?}")
    });
    assert_eq!(
        (wire.reason.as_str(), wire.domain.as_str(), &wire.metadata),
        (reason, REFUSAL_DETAIL_DOMAIN, &owned(metadata)),
        "the refusal detail's reason, domain and metadata; status {status:?}"
    );
    let delay = wire
        .retry_delay
        .unwrap_or_else(|| panic!("the refusal must carry a RetryInfo delay; got {wire:?}"));
    assert!(
        range.min <= delay && delay < range.max,
        "retry_delay {delay:?} is outside [{:?}, {:?}) for {reason}",
        range.min,
        range.max
    );
}

/// Assert `status` is the shutdown ending or refusal: `UNAVAILABLE "server
/// shutting down"` with `SERVER_SHUTTING_DOWN`, no metadata, and a
/// `retry_delay` under 100 ms.
#[track_caller]
pub fn assert_shutdown_status(status: &Status) {
    assert_eq!(status.message(), "server shutting down", "{status:?}");
    assert_refusal_detail(
        status,
        refusal_reason::SERVER_SHUTTING_DOWN,
        &[],
        SHUTDOWN_RETRY_DELAY,
    );
}

/// Assert `status` is a `CreateLayer` / `UpdateLayer` refused before its
/// commit for a dependency outage: `UNAVAILABLE` with `UPSTREAM_UNAVAILABLE`,
/// no metadata, and a `retry_delay` in `[1 s, 3 s)`.
#[track_caller]
pub fn assert_upstream_unavailable(status: &Status) {
    assert_refusal_detail(
        status,
        refusal_reason::UPSTREAM_UNAVAILABLE,
        &[],
        UPSTREAM_UNAVAILABLE_RETRY_DELAY,
    );
}

/// Assert `status` carries no refusal detail: no `ErrorInfo`, no `RetryInfo`,
/// no details at all.
#[track_caller]
pub fn assert_no_refusal_detail(status: &Status) {
    assert!(
        status.details().is_empty() && decode(status).is_none(),
        "this ending is not a refusal that applied nothing (a load-shedding or shutdown \
         refusal, or an upload refused before its commit) and must carry no refusal detail; \
         got {:?}",
        decode(status)
    );
}

/// The message and detail of an `UnavailableWithDetail`, panicking on any
/// other error.
#[track_caller]
pub fn detail_of(err: &ServiceError) -> (&str, &RefusalDetail) {
    match err {
        ServiceError::UnavailableWithDetail { message, detail } => (message.as_str(), detail),
        other => panic!("expected UNAVAILABLE with a refusal detail; got {other:?}"),
    }
}

/// Assert `err` is `UnavailableWithDetail` with `reason`, exactly `metadata`
/// and `range`, and that its wire form decodes to the same; return its
/// message.
#[track_caller]
pub fn assert_refusal_error<'a>(
    err: &'a ServiceError,
    reason: &str,
    metadata: &[(&str, &str)],
    range: RetryDelayRange,
) -> &'a str {
    let (message, detail) = detail_of(err);
    let mut got: Vec<(String, String)> = detail
        .metadata
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    got.sort();
    assert_eq!(
        (detail.reason, &got, detail.retry_delay),
        (reason, &owned(metadata), Some(range)),
        "the refusal detail of {err:?}"
    );
    assert_refusal_detail(&detail.unavailable(message), reason, metadata, range);
    message
}

/// The message of a stream-multiplexer ending (`ConsumerTerminalExit`'s
/// `error`), asserting its shape: the multiplexer's shutdown ending (`reason`
/// = `shutdown`, at the pool quit) is a shutdown ending like every other and
/// carries `SERVER_SHUTTING_DOWN`; every other reader-side ending — a stream
/// gap, a slow reader, a watchdog, a reader error — is a plain retryable
/// `UNAVAILABLE` with no refusal detail.
#[track_caller]
pub fn mux_ending_message(error: ServiceError, reason: &str) -> String {
    if reason == live_edit::observability::metrics::STREAM_EXIT_SHUTDOWN {
        let message = assert_refusal_error(
            &error,
            refusal_reason::SERVER_SHUTTING_DOWN,
            &[],
            SHUTDOWN_RETRY_DELAY,
        )
        .to_owned();
        assert_eq!(message, "server shutting down");
        return message;
    }
    // The plain variant is the check: only `UnavailableWithDetail` carries a
    // refusal detail.
    match error {
        ServiceError::Unavailable(message) => message,
        other => panic!(
            "every reader-side `{reason}` ending is a plain retryable UNAVAILABLE, with no \
             refusal detail; got {other:?}"
        ),
    }
}

/// The retry hint of a subscription whose catch-up was refused its slot:
/// `CATCHUP_CAPACITY`, `wait` = `slot`, a delay in `[500 ms, 1500 ms)`.
#[track_caller]
pub fn assert_catchup_slot_refusal(status: &Status) {
    assert_refusal_detail(
        status,
        refusal_reason::CATCHUP_CAPACITY,
        &[(refusal_metadata::WAIT, refusal_metadata::SLOT)],
        CAPACITY_RETRY_DELAY,
    );
}

/// The retry hint of a subscription whose catch-up was refused its snapshot
/// read permit: `CATCHUP_CAPACITY`, `wait` = `read_permit`.
#[track_caller]
pub fn assert_catchup_read_permit_refusal(status: &Status) {
    assert_refusal_detail(
        status,
        refusal_reason::CATCHUP_CAPACITY,
        &[(refusal_metadata::WAIT, refusal_metadata::READ_PERMIT)],
        CAPACITY_RETRY_DELAY,
    );
}

// ── The size refusals, decoded as the C++ client library decodes them ───────

/// The `google.rpc.Status` details of `status`, decoded straight from
/// `grpc-status-details-bin` the way the C++ client library reads a refusal:
/// its `Any`s, each a `(type_url, value)`; no `tonic_types` convenience in
/// between, so the wire shape itself is checked.
pub fn wire_details(status: &Status) -> Vec<prost_types::Any> {
    use prost::Message;
    tonic_types::pb::Status::decode(status.details())
        .unwrap_or_else(|e| panic!("grpc-status-details-bin must decode as google.rpc.Status: {e}"))
        .details
}

/// Assert `status` carries exactly one `ErrorInfo` — an `Any` whose type URL
/// ends `/google.rpc.ErrorInfo` — of reason `reason` in the service's domain,
/// whose metadata is exactly `metadata`, every value a decimal `u64` with no
/// sign (the C++ client library parses each with an unsigned decimal parse),
/// and no `RetryInfo`. This is the refusal contract the C++ client library
/// consumes (`docs-internal/sdd.md` §18.1 *Size refusals*); the other
/// assertions here decode through `tonic_types`, which would accept shapes
/// that client does not.
#[track_caller]
pub fn assert_ovlive_reads_size_refusal(status: &Status, reason: &str, metadata: &[(&str, u64)]) {
    use prost::Message;
    let details = wire_details(status);
    let infos: Vec<&prost_types::Any> = details
        .iter()
        .filter(|any| any.type_url.ends_with("/google.rpc.ErrorInfo"))
        .collect();
    assert_eq!(
        infos.len(),
        1,
        "exactly one ErrorInfo; got the details {:?}",
        details.iter().map(|a| &a.type_url).collect::<Vec<_>>()
    );
    assert!(
        !details
            .iter()
            .any(|any| any.type_url.ends_with("/google.rpc.RetryInfo")),
        "a size refusal carries no RetryInfo: no re-send of the same delta can succeed"
    );
    let info = tonic_types::pb::ErrorInfo::decode(infos[0].value.as_slice())
        .expect("the ErrorInfo Any decodes as google.rpc.ErrorInfo");
    assert_eq!(
        (info.reason.as_str(), info.domain.as_str()),
        (reason, REFUSAL_DETAIL_DOMAIN),
        "{status:?}"
    );
    let mut got: Vec<(String, u64)> = info
        .metadata
        .iter()
        .map(|(key, value)| {
            assert!(
                !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()),
                "metadata {key}={value:?} must be a decimal integer with no sign"
            );
            let parsed: u64 = value
                .parse()
                .unwrap_or_else(|e| panic!("metadata {key}={value:?} must fit a u64: {e}"));
            (key.clone(), parsed)
        })
        .collect();
    got.sort();
    let mut want: Vec<(String, u64)> = metadata
        .iter()
        .map(|(k, v)| ((*k).to_owned(), *v))
        .collect();
    want.sort();
    assert_eq!(got, want, "the refusal's metadata");
}

/// Assert `status` is the refusal of a delta of more entries than one delta
/// may hold: `INVALID_ARGUMENT`, a message naming the three counts, their
/// total, the 99,999 limit and the remedy ("split"), and, as the C++ client
/// library reads it, `DELTA_TOO_LARGE` with exactly `struct_commands`, `set_fields`,
/// `time_samples` and `max_entries` = 99,999, and no `RetryInfo`.
#[track_caller]
pub fn assert_delta_too_large(
    status: &Status,
    struct_commands: u64,
    set_fields: u64,
    time_samples: u64,
) {
    let max_entries = live_edit::domain::limits::MAX_DELTA_ENTRIES;
    assert_eq!(
        max_entries, 99_999,
        "the limit the C++ client library defaults to"
    );
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    let total = struct_commands + set_fields + time_samples;
    assert_eq!(
        status.message(),
        format!(
            "delta too large: {total} entries ({struct_commands} struct commands, {set_fields} \
             set fields, {time_samples} time samples); one delta may carry at most \
             {max_entries} entries in total; split it into smaller deltas"
        ),
        "the message names the counts, the limit and the remedy"
    );
    assert_ovlive_reads_size_refusal(
        status,
        refusal_reason::DELTA_TOO_LARGE,
        &[
            (refusal_metadata::STRUCT_COMMANDS, struct_commands),
            (refusal_metadata::SET_FIELDS, set_fields),
            (refusal_metadata::TIME_SAMPLES, time_samples),
            (refusal_metadata::MAX_ENTRIES, max_entries),
        ],
    );
}

/// Assert `status` is the refusal of a `ReorderChildren` list of `actual`
/// ids: `INVALID_ARGUMENT` with the domain bound's message, unchanged from
/// before the refusal carried a reason, and, as the C++ client library reads
/// it, `CHILDREN_LIST_TOO_LONG` with `limit` = 100,000 and `actual`, and no
/// `RetryInfo`.
#[track_caller]
pub fn assert_children_list_too_long(status: &Status, actual: u64) {
    let limit = live_edit::domain::limits::MAX_CHILDREN_LIST_LEN as u64;
    assert_eq!(
        limit, 100_000,
        "the limit the C++ client library defaults to"
    );
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    assert_eq!(
        status.message(),
        format!(
            "delta error: delta bounds exceeded: ReorderChildren.children_list.len \
             limit={limit}, actual={actual}"
        )
    );
    assert_ovlive_reads_size_refusal(
        status,
        refusal_reason::CHILDREN_LIST_TOO_LONG,
        &[
            (refusal_metadata::LIMIT, limit),
            (refusal_metadata::ACTUAL, actual),
        ],
    );
}

/// The retry hint of a subscription refused a stream reader:
/// `READER_CAPACITY`, no metadata.
#[track_caller]
pub fn assert_reader_capacity_refusal(status: &Status) {
    assert_refusal_detail(
        status,
        refusal_reason::READER_CAPACITY,
        &[],
        CAPACITY_RETRY_DELAY,
    );
}
