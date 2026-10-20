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

//! Shared JSON log capture for the completion-logging tests (SDD §17.2.1).
//!
//! The completion line *is* the deliverable of those tests, so they assert on
//! it structurally — level, message, every field — rather than on a substring.
//! JSON is what makes an absent field distinguishable from an empty one:
//! `tracing` records nothing for a `None`, so an identifier the handler never
//! validated is a missing key, not a `null`.
//!
//! Two capture scopes, because completion lines come from two kinds of place:
//!
//! * [`capture`] — a `WithSubscriber` scope around one future. Right for a line
//!   emitted inside the awaited handler (a setup rejection), on a multi-thread
//!   runtime: the dispatcher follows the future across worker threads.
//! * [`ThreadCapture`] — a thread-local `set_default` scope. Required for a
//!   line emitted from a task the handler *spawned* (a streaming tail, a
//!   tonic connection task behind a real server): a spawned task inherits no
//!   dispatcher, so `WithSubscriber` on the handler future captures nothing
//!   there and the assertion would read "the line was never emitted". On a
//!   **current-thread** runtime every spawned task is polled on the thread
//!   holding the guard, so it is in scope. Tests using it must run on
//!   `#[tokio::test]` (current-thread) — see `b_muu_tests.rs` for the idiom.
//!
//! Both record at `with_max_level(INFO)`, which covers every level a
//! completion line can have (INFO, WARN, and the server-fault ERROR) while
//! keeping DEBUG backend chatter out.
//!
//! Two more, for evidence that is not a completion line: [`capture_debug`] is
//! [`capture`]'s scope recording DEBUG too. It serves the pool-quit assertion
//! ([`assert_quit_was_not_held_by_a_blocked_read`]), whose proof that a
//! server-side read was really woken is a DEBUG line, and the snapshot-fence
//! tests' measurement events emitted inside the awaited call (`update_layer
//! fence wait`, `update_layer exec conflict`, `snapshot fence lease`). And
//! [`ThreadCapture::start_debug`] is the thread-local scope recording DEBUG,
//! for a DEBUG line a spawned task emits (a stream reader's backoff, or the
//! `snapshot fence lease` event of a dropped lease's spawned release).

use std::sync::{Arc, Mutex};

use live_edit::observability::trace_export::test_support::keep_a_second_dispatcher_registered;
use serde_json::Value;

/// The message of every non-fault completion line. Duplicated from
/// `rpc_log::RPC_FAILURE_MSG` (which is `pub(crate)`) on purpose: the string is
/// what operators grep for, so a rename must break a test.
pub(crate) const RPC_FAILURE_MSG: &str = "RPC failed";

/// The message of the server-fault ERROR line (`error::INTERNAL_ERROR_MSG`).
pub(crate) const INTERNAL_ERROR_MSG: &str = "internal error";

/// The message of the metrics layer's drop-guard line
/// (`rpc_log::RPC_ABANDONED_MSG`).
pub(crate) const RPC_ABANDONED_MSG: &str = "RPC abandoned before completion";

/// Every `grpc_code` slug the metrics layer records on
/// `live_endpoint_calls_total` — and every completion line carries — so a
/// per-endpoint total is a sum over all of them. Mirrors
/// `observability::metrics::GRPC_CODES`, which is crate-private.
pub(crate) const GRPC_CODE_SLUGS: &[&str] = &[
    "OK",
    "CANCELLED",
    "UNKNOWN",
    "INVALID_ARGUMENT",
    "DEADLINE_EXCEEDED",
    "NOT_FOUND",
    "ALREADY_EXISTS",
    "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED",
    "FAILED_PRECONDITION",
    "ABORTED",
    "OUT_OF_RANGE",
    "UNIMPLEMENTED",
    "INTERNAL",
    "UNAVAILABLE",
    "DATA_LOSS",
    "UNAUTHENTICATED",
];

#[derive(Clone)]
struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
    type Writer = CapturingWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl std::io::Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn json_dispatch(buf: &Arc<Mutex<Vec<u8>>>) -> tracing::Dispatch {
    json_dispatch_at(buf, tracing::Level::INFO)
}

/// A JSON dispatch recording up to `max_level` into `buf`. Registers the
/// process-wide second dispatcher first, so a capture is never the only
/// dispatcher and never loses a line to a callsite's interest cached from
/// another thread — see [`keep_a_second_dispatcher_registered`].
fn json_dispatch_at(buf: &Arc<Mutex<Vec<u8>>>, max_level: tracing::Level) -> tracing::Dispatch {
    keep_a_second_dispatcher_registered();
    tracing::Dispatch::new(
        tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(Arc::clone(buf)))
            .with_max_level(max_level)
            .with_ansi(false)
            .finish(),
    )
}

fn parse(buf: &Arc<Mutex<Vec<u8>>>) -> Captured {
    let raw = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
    let text = String::from_utf8(raw).expect("captured tracing output must be UTF-8");
    let events = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("captured line is not JSON: {e}\n{line}"))
        })
        .collect();
    Captured { events, text }
}

/// Everything captured, parsed and raw.
pub(crate) struct Captured {
    pub(crate) events: Vec<Value>,
    /// The raw buffer, for sweeps that must cover fields no test names.
    pub(crate) text: String,
}

impl Captured {
    /// The events whose message is `msg`.
    pub(crate) fn with_message(&self, msg: &str) -> Vec<&Value> {
        self.events
            .iter()
            .filter(|e| e["fields"]["message"] == msg)
            .collect()
    }

    /// The single event carrying `msg`, asserting there is exactly one — so no
    /// test can pass by reading the first of two duplicated lines.
    pub(crate) fn only(&self, msg: &str) -> &Value {
        let hits = self.with_message(msg);
        assert_eq!(
            hits.len(),
            1,
            "expected exactly one `{msg}` event; got {} — full capture:\n{}",
            hits.len(),
            self.text
        );
        hits[0]
    }

    /// Every completion line in the capture: `RPC failed`, the server-fault
    /// `internal error`, and the abandoned-call line.
    pub(crate) fn completion_lines(&self) -> Vec<&Value> {
        self.events
            .iter()
            .filter(|e| {
                let m = &e["fields"]["message"];
                m == RPC_FAILURE_MSG || m == INTERNAL_ERROR_MSG || m == RPC_ABANDONED_MSG
            })
            .collect()
    }

    /// Assert the capture holds no completion line. A subscription has no other
    /// request-level line to check for: the only ending that logs none is a
    /// client disconnect, and every other ending logs its completion line.
    pub(crate) fn assert_no_completion_line(&self, ctx: &str) {
        let lines = self.completion_lines();
        assert!(
            lines.is_empty(),
            "{ctx}: expected no completion line; full capture:\n{}",
            self.text
        );
    }

    /// Assert `needle` appears nowhere — verbatim or JSON-escaped. The capture
    /// is JSON, so a control byte is only ever written escaped; a raw-only
    /// sweep for one could never fire.
    pub(crate) fn assert_absent_everywhere(&self, needle: &str) {
        assert!(
            !self.text.contains(needle),
            "`{needle}` appears verbatim in the log output:\n{}",
            self.text
        );
        let encoded = serde_json::to_string(needle).expect("a &str always serializes");
        let escaped = &encoded[1..encoded.len() - 1];
        if escaped != needle {
            assert!(
                !self.text.contains(escaped),
                "`{needle}` appears JSON-escaped (`{escaped}`) in the log output:\n{}",
                self.text
            );
        }
    }
}

/// Assert an event carries `field` with `want`.
pub(crate) fn assert_field(event: &Value, field: &str, want: Value) {
    assert_eq!(
        event["fields"][field], want,
        "field `{field}` must be {want}; full event: {event}"
    );
}

/// Assert `field` is ABSENT — not present-and-null, not present-and-empty.
pub(crate) fn assert_field_absent(event: &Value, field: &str, why: &str) {
    assert!(
        event["fields"].get(field).is_none(),
        "field `{field}` must be absent ({why}); full event: {event}"
    );
}

/// Assert the always-present fields and the level.
pub(crate) fn assert_common_fields(
    event: &Value,
    level: &str,
    endpoint: &str,
    stage: &str,
    grpc_code: &str,
) {
    assert_eq!(event["level"], level, "wrong level; full event: {event}");
    assert_field(event, "endpoint", endpoint.into());
    assert_field(event, "stage", stage.into());
    assert_field(event, "grpc_code", grpc_code.into());
}

/// Drive `future` under a scoped JSON capture. See the module doc for when
/// this is the right scope.
pub(crate) async fn capture<F, T>(future: F) -> (T, Captured)
where
    F: std::future::Future<Output = T>,
{
    use tracing::instrument::WithSubscriber;
    let buf = Arc::new(Mutex::new(Vec::new()));
    let value = future.with_subscriber(json_dispatch(&buf)).await;
    (value, parse(&buf))
}

/// As [`capture`], but keeping DEBUG events too — for a test whose evidence
/// is a DEBUG line (for example, that a step actually did work rather than
/// finding nothing to do).
pub(crate) async fn capture_debug<F, T>(future: F) -> (T, Captured)
where
    F: std::future::Future<Output = T>,
{
    use tracing::instrument::WithSubscriber;
    let buf = Arc::new(Mutex::new(Vec::new()));
    let value = future
        .with_subscriber(json_dispatch_at(&buf, tracing::Level::DEBUG))
        .await;
    (value, parse(&buf))
}

/// Assert that a `ValkeyClient::quit` run under [`capture_debug`] was not held
/// up by a server-side blocked read. A `QUIT` queued behind a reader's blocked
/// `XREAD` overruns the per-pool `QUIT_TIMEOUT`, and the quit then logs the
/// stream reader connections' quit-timeout WARN; so the quit must have logged
/// neither that WARN nor the multiplexer's reader-grace WARN (matched on each
/// message's fixed prefix), and must have logged the stream multiplexer's
/// shutdown DEBUG line counting exactly `blocked_reads` woken reads — the
/// evidence that there were blocked reads to get past at all. No wall-clock
/// time is asserted: the absent WARN is what rules out the overrun.
///
/// That `woken` count is also the only evidence that every shutdown wake
/// landed. A reader whose shutdown wake fails or times out counts as not
/// woken, so the count comes up short. Its WARN ("stream multiplexer shutdown
/// wake failed…") is logged from the reader's spawned task, which
/// [`capture_debug`]'s `WithSubscriber` scope does not reach, so it is not
/// matched here.
pub(crate) fn assert_quit_was_not_held_by_a_blocked_read(captured: &Captured, blocked_reads: u64) {
    for warning in [
        "stream reader connection quit timed out",
        "stream multiplexer readers did not stop within the shutdown grace",
    ] {
        assert!(
            !captured.events.iter().any(|e| e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with(warning))),
            "the pool quit logged `{warning}…`; full capture:\n{}",
            captured.text
        );
    }
    let woke = captured.with_message("woke blocked stream reads before the reader pool QUIT");
    assert_eq!(
        woke.len(),
        1,
        "the stream multiplexer's shutdown must report the reads it woke — the evidence that a \
         read was blocked at all; full capture:\n{}",
        captured.text
    );
    assert_eq!(
        woke[0]["fields"]["woken"], blocked_reads,
        "every reader's blocked read must have been woken"
    );
}

/// A thread-local capture, live until dropped. See the module doc: only
/// meaningful on a current-thread runtime.
pub(crate) struct ThreadCapture {
    buf: Arc<Mutex<Vec<u8>>>,
    _guard: tracing::subscriber::DefaultGuard,
}

impl ThreadCapture {
    pub(crate) fn start() -> Self {
        Self::start_at(tracing::Level::INFO)
    }

    /// As [`Self::start`], keeping DEBUG events too — for evidence a spawned
    /// task logs at DEBUG (a stream reader's backoff under a fault-injection
    /// seam, or a dropped snapshot-fence lease's spawned release and its
    /// `snapshot fence lease` event).
    pub(crate) fn start_debug() -> Self {
        Self::start_at(tracing::Level::DEBUG)
    }

    fn start_at(max_level: tracing::Level) -> Self {
        assert_eq!(
            tokio::runtime::Handle::current().runtime_flavor(),
            tokio::runtime::RuntimeFlavor::CurrentThread,
            "ThreadCapture only sees spawned tasks on a current-thread runtime; on a \
             multi-thread one it would silently miss every line a worker thread emits"
        );
        let buf = Arc::new(Mutex::new(Vec::new()));
        let guard = tracing::dispatcher::set_default(&json_dispatch_at(&buf, max_level));
        Self { buf, _guard: guard }
    }

    /// Everything captured so far.
    pub(crate) fn snapshot(&self) -> Captured {
        parse(&self.buf)
    }
}

/// Delete every key under the fixture's prefix.
///
/// `TestValkey`'s `Drop` cleanup needs `block_in_place`, which a current-thread
/// runtime cannot provide, so it skips there. Tests that run on one call this
/// at the end instead — so a current-thread test that fails before its end
/// leaves its keys behind. They are all under its own `test_<uuid>:` prefix,
/// so no other test reads them; only a red run leaks, and a flush of the test
/// database clears it.
pub(crate) async fn purge_keys(tv: &live_edit::storage::test_support::TestValkey) {
    use fred::interfaces::KeysInterface;
    use futures::StreamExt;

    let pattern = format!("{}*", tv.prefix());
    let pool = &tv.client().pool;
    let mut keys = Vec::new();
    {
        let mut scan = pool.next().scan_buffered(&pattern, Some(100), None);
        while let Some(Ok(key)) = scan.next().await {
            keys.push(key);
        }
    }
    for chunk in keys.chunks(100) {
        let _: Result<(), _> = pool.unlink(chunk.to_vec()).await;
    }
}
