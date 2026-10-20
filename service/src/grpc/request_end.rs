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

//! How each request stream actually ended, recorded from the transport, so an
//! upload commits only when its client finished it (SDD §5.4); and when its
//! body last carried bytes, which the JoinSession client keepalive reads (SDD
//! §13.2; see the last two paragraphs below).
//!
//! An upload (`CreateLayer`, `UpdateLayer`) is applied only if its request
//! stream ended with HTTP/2 END_STREAM. The handler cannot learn that from
//! `tonic::Streaming`: `message()` answers `Ok(None)` both for END_STREAM and
//! for several endings where the client never finished sending, because two
//! layers below it each turn an abandoned stream into a clean end:
//!
//! - **hyper** (1.11.0) ends the body with `None`, not an error, on a
//!   `RST_STREAM(NO_ERROR)` and on the service's own keepalive `GOAWAY(NO_ERROR)`
//!   (`body/incoming.rs:250-255`, and `:273-278` for trailers). Envoy v1.36.x
//!   forwards a downstream client's cancel upstream as exactly that
//!   `RST_STREAM(NO_ERROR)`.
//! - **tonic** (0.14.6) turns a request-body error whose status is `Cancelled`
//!   — what a `RST_STREAM(CANCEL)` maps to — into `Ok(None)`
//!   (`codec/decode.rs:260-263`), and keeps no trace of it.
//!
//! The one signal both swallows leave is h2's (0.4.19) end-of-stream state:
//! hyper's `Incoming::is_end_stream()` (`body/incoming.rs:297`) returns h2's
//! `RecvStream::is_end_stream()`, which is `is_recv_end_stream() &&
//! pending_recv.is_empty()` (`proto/streams/recv.rs:645-651`). Once the body
//! has yielded `None`, that is true exactly when END_STREAM was received —
//! states `HalfClosedRemote`, `Closed(EndStream)` and
//! `Closed(ErrorAfterEndStream)` (`proto/streams/state.rs:424-430`) — and
//! false for every reset, connection loss or GOAWAY before it. Every wrapper
//! between hyper and the handler forwards `is_end_stream`: tonic's
//! `body::Body`, http-body-util's `MapErr` / `MapFrame` / boxed bodies, and
//! axum-core's `Body`.
//!
//! The record reads h2's end-of-stream state, so it holds for HTTP/2 only. An
//! HTTP/1.1 body reports its end only at a zero remaining content length
//! (`body/incoming.rs:295`), which a chunked body never reaches, so every
//! chunked upload would be refused `CANCELLED`. The service serves neither
//! HTTP/1.1 nor gRPC-web; enabling either needs its own end signal here.
//!
//! So this module records it, positively:
//!
//! - [`EndAttestingBody`] wraps a request body and passes every frame, error
//!   and size hint through unchanged. It records the body's [`StreamEnding`]
//!   once, the first time the ending is known: an error is
//!   [`StreamEnding::ResetBeforeEnd`]; the end of the body (`None`, or a
//!   trailers frame, after which tonic stops polling, `codec/decode.rs:283-290`)
//!   is [`StreamEnding::Complete`] when the inner body reports
//!   `is_end_stream()`, else [`StreamEnding::EndedWithoutEndStream`]. An inner
//!   `is_end_stream()` of `true` asked at any time records `Complete` too —
//!   tonic's `body::Body::new` drops an already-ended body unpolled
//!   (`body.rs:38-40`), which is how a HEADERS-only request arrives. The first
//!   record wins and is never changed.
//! - [`RequestStreamEnd`] is the shared record, carried in the request's
//!   extensions. Only [`attest_request`] and [`attest_request_end`] make one,
//!   paired with the body that writes it, so no caller can fabricate a
//!   "complete" record.
//! - [`AttestRequestEnd`] is the service wrapper that does this for every
//!   request to the service it wraps. The LiveService is built only through
//!   [`live_service_server`](crate::grpc::transport::live_service_server), which
//!   puts it outermost, so `main.rs` and every test server carry the record.
//!
//! The upload ingest (`grpc::streaming::UploadStream`) reads the record at
//! each `Ok(None)` and refuses every ending but `Complete` with `CANCELLED`,
//! before anything is stored or committed. A request with no record — a stack
//! built without the wrapper — fails closed with `INTERNAL`.
//!
//! The record errs only conservatively: it never reads `Complete` for a stream
//! whose END_STREAM did not arrive. A connection lost after END_STREAM but
//! before the handler drained the body reads as truncated (h2's `recv_eof`,
//! `proto/streams/state.rs:309-322`); that client is gone anyway. A reset
//! *after* END_STREAM reads `Complete`: the upload is whole, and only the call
//! was cancelled (hyper then drops the handler at its next `.await`,
//! `proto/h2/server.rs:453-465`).
//!
//! Every request is wrapped, not only the uploads, so the wrapper needs no
//! path matching that could drift from the proto; the cost is one more body
//! layer per request. Only the two upload handlers read the record.
//!
//! **A second record: when the request body last carried bytes.** The same
//! body also stamps a [`RequestBodyProgress`] with `tokio::time::Instant::now()`
//! on every non-empty DATA frame it yields — a trailers frame ends the stream
//! and an empty DATA frame carries no bytes, so neither stamps. It is a
//! sibling record, not part of [`RequestStreamEnd`], whose contract is
//! "written once". [`attest_request`] makes both, and [`AttestRequestEnd`]
//! inserts both into every request's extensions. `JoinSession` reads the
//! progress record to end a join whose client opted in to keepalives and then
//! sent no data for three intervals (SDD §13.2): any bytes of its request
//! stream count, so a large message still arriving keeps the join alive.
//!
//! The stamp is taken when the frame is *polled*, which tonic does only from
//! `Streaming::message` (tonic 0.14.6 `codec/decode.rs:257-259, 401-418`):
//! never earlier than the bytes arrived. The record therefore errs only
//! toward a client that sent more recently than it did, never less, and a
//! reader that polls the stream before it checks the record has every byte
//! that had arrived stamped. The clock is tokio's, which is the std clock in a
//! production build and the paused clock under a paused test runtime, the
//! same clock as `tokio::time::Sleep`. Every request is stamped, again with no
//! path matching; the cost is one clock read and one relaxed store per DATA
//! frame.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http_body::{Body, Frame, SizeHint};
use tonic::server::NamedService;
use tower::Service;

/// How a request stream ended, as the transport saw it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamEnding {
    /// The client finished the stream: HTTP/2 END_STREAM was received.
    Complete,
    /// The body ended with an error before END_STREAM — a reset with a code
    /// hyper surfaces, or a lost connection. Carries the error's `Display`.
    ResetBeforeEnd(String),
    /// The body ended cleanly but END_STREAM was never received — hyper's
    /// shape for a `RST_STREAM(NO_ERROR)` and for the service's own keepalive
    /// `GOAWAY`.
    EndedWithoutEndStream,
}

/// The end-of-stream record of one request, shared between the
/// [`EndAttestingBody`] that writes it and the handler that reads it.
///
/// Carried in the request's extensions (it is `Clone + Send + Sync`, so
/// `request.extensions_mut().insert(end)` places it). Made only by
/// [`attest_request`] and [`attest_request_end`], together with the body that
/// writes it, so no caller can fabricate a "complete" record. Written once;
/// never changed after.
#[derive(Clone, Debug)]
pub struct RequestStreamEnd(Arc<OnceLock<StreamEnding>>);

impl RequestStreamEnd {
    /// How the stream ended, or `None` while its ending is not yet known.
    pub fn ending(&self) -> Option<StreamEnding> {
        self.0.get().cloned()
    }

    /// Record `ending` unless an ending is already recorded: the first wins.
    fn record(&self, ending: StreamEnding) {
        let _ = self.0.set(ending);
    }
}

/// When one request's body last yielded a non-empty DATA frame, shared
/// between the [`EndAttestingBody`] that stamps it and the handler that reads
/// it (see the module doc).
///
/// Carried in the request's extensions beside the [`RequestStreamEnd`]. Made
/// only by [`attest_request`], together with the body that stamps it.
#[derive(Clone, Debug)]
pub struct RequestBodyProgress(Arc<BodyProgress>);

#[derive(Debug)]
struct BodyProgress {
    /// When the record was made; the stamp is an offset from it.
    base: tokio::time::Instant,
    /// Nanoseconds from `base` to the latest stamp, plus one; `0` while no
    /// DATA frame has been stamped. Written only by the body's poller.
    last_data_nanos: AtomicU64,
}

impl RequestBodyProgress {
    fn new() -> Self {
        Self(Arc::new(BodyProgress {
            base: tokio::time::Instant::now(),
            last_data_nanos: AtomicU64::new(0),
        }))
    }

    /// When the body last yielded a non-empty DATA frame, on tokio's clock,
    /// or `None` if it has yielded none.
    pub fn last_data_at(&self) -> Option<tokio::time::Instant> {
        match self.0.last_data_nanos.load(Ordering::Relaxed) {
            0 => None,
            stamped => Some(self.0.base + Duration::from_nanos(stamped - 1)),
        }
    }

    /// Stamp now as the body's latest DATA frame.
    fn stamp(&self) {
        let since_base = tokio::time::Instant::now().saturating_duration_since(self.0.base);
        let nanos = u64::try_from(since_base.as_nanos()).unwrap_or(u64::MAX - 1);
        self.0
            .last_data_nanos
            .store(nanos.saturating_add(1), Ordering::Relaxed);
    }
}

/// Wrap `body` so it records how it ends and when it last carried bytes, and
/// return both records.
///
/// The one way to make a [`RequestBodyProgress`], and, with
/// [`attest_request_end`], a [`RequestStreamEnd`]. [`AttestRequestEnd`] calls
/// it for every request; a handler-level test calls it on its own fake body
/// and inserts the records it needs into the request's extensions. A fake
/// body must report `is_end_stream()` truthfully (see
/// [`attest_request_end`]).
pub fn attest_request<B>(body: B) -> (EndAttestingBody<B>, RequestStreamEnd, RequestBodyProgress) {
    let end = RequestStreamEnd(Arc::new(OnceLock::new()));
    let progress = RequestBodyProgress::new();
    (
        EndAttestingBody {
            inner: body,
            end: end.clone(),
            progress: progress.clone(),
        },
        end,
        progress,
    )
}

/// Wrap `body` so it records how it ends, and return the record:
/// [`attest_request`] without the progress record.
///
/// [`attest_request`] and this are the only ways to make a
/// [`RequestStreamEnd`]. A handler-level test calls this on its own fake body
/// and inserts the record into the request's extensions. A fake body must
/// report `is_end_stream()` truthfully — `true` once it has ended the way
/// END_STREAM does, never before — or every clean upload over it reads as
/// truncated (`http_body_util::StreamBody`, for one, always reports `false`).
pub fn attest_request_end<B>(body: B) -> (EndAttestingBody<B>, RequestStreamEnd) {
    let (body, end, _progress) = attest_request(body);
    (body, end)
}

/// A request body that records its [`StreamEnding`] into a
/// [`RequestStreamEnd`] and stamps each non-empty DATA frame into a
/// [`RequestBodyProgress`], passing every frame, error and size hint through
/// unchanged. Made by [`attest_request`] and [`attest_request_end`].
///
/// The inner body must be `Unpin` (hyper's `Incoming`, tonic's `body::Body`
/// and the boxed bodies are); pin a body that is not.
#[derive(Debug)]
pub struct EndAttestingBody<B> {
    inner: B,
    end: RequestStreamEnd,
    progress: RequestBodyProgress,
}

impl<B> EndAttestingBody<B>
where
    B: Body,
{
    /// The ending a body that has just ended reached: END_STREAM if the inner
    /// body says so.
    fn ended(&self) -> StreamEnding {
        if self.inner.is_end_stream() {
            StreamEnding::Complete
        } else {
            StreamEnding::EndedWithoutEndStream
        }
    }
}

impl<B> Body for EndAttestingBody<B>
where
    B: Body<Data = Bytes> + Unpin,
    B::Error: std::fmt::Display,
{
    type Data = Bytes;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // `EndAttestingBody<B>` is `Unpin` because `B` is and both records
        // are `Arc`s, so `get_mut` is sound.
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(Some(Err(e))) => {
                this.end.record(StreamEnding::ResetBeforeEnd(e.to_string()));
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                this.end.record(this.ended());
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(frame))) => {
                // tonic stops polling a request body after a trailers frame,
                // so the body has ended here as far as the upload is
                // concerned.
                if frame.is_trailers() {
                    this.end.record(this.ended());
                }
                // Bytes from the client: the progress stamp. An empty DATA
                // frame carries none, and a trailers frame is not data.
                if frame.data_ref().is_some_and(|data| !data.is_empty()) {
                    this.progress.stamp();
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        let ended = self.inner.is_end_stream();
        if ended {
            // tonic's `body::Body::new` drops an already-ended body without
            // polling it, so this may be the only look the record gets.
            self.end.record(StreamEnding::Complete);
        }
        ended
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

/// A service wrapper that, for every request, wraps the body in an
/// [`EndAttestingBody`], inserts its [`RequestStreamEnd`] and
/// [`RequestBodyProgress`] into the request's extensions, and calls the inner
/// service. Takes the inner service's
/// [`NamedService::NAME`], so it routes exactly as the service it wraps.
///
/// Built by [`live_service_server`](crate::grpc::transport::live_service_server).
#[derive(Clone, Debug)]
pub struct AttestRequestEnd<S> {
    inner: S,
}

impl<S> AttestRequestEnd<S> {
    /// Wrap `inner`.
    pub fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S> Service<http::Request<tonic::body::Body>> for AttestRequestEnd<S>
where
    S: Service<http::Request<tonic::body::Body>, Error = Infallible>,
{
    type Response = S::Response;
    type Error = Infallible;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: http::Request<tonic::body::Body>) -> Self::Future {
        let (mut parts, body) = request.into_parts();
        let (body, end, progress) = attest_request(body);
        parts.extensions.insert(end);
        parts.extensions.insert(progress);
        self.inner.call(http::Request::from_parts(
            parts,
            tonic::body::Body::new(body),
        ))
    }
}

impl<S> NamedService for AttestRequestEnd<S>
where
    S: NamedService,
{
    const NAME: &'static str = S::NAME;
}

#[cfg(test)]
mod tests {
    //! The recording rules of [`EndAttestingBody`] over fake inner bodies —
    //! the end record and the progress stamp — and [`AttestRequestEnd`]'s
    //! wiring of both records. What hyper and h2 actually report for each way
    //! a real HTTP/2 stream ends, and that a real DATA frame stamps the
    //! progress record, are pinned against a real server in
    //! `tests/integration/grpc/request_end_wire_test.rs`.

    use std::collections::VecDeque;
    use std::task::Poll;

    use http_body_util::BodyExt;

    use super::*;

    /// One poll of a [`FakeBody`]'s script.
    type Polled = Poll<Option<Result<Frame<Bytes>, FakeError>>>;

    /// A fake request body: yields `polls` in order (an exhausted script
    /// stays `Pending`), and reports `end_stream` as its `is_end_stream()`,
    /// which the test sets as it likes.
    struct FakeBody {
        polls: VecDeque<Polled>,
        end_stream: bool,
    }

    #[derive(Debug)]
    struct FakeError(&'static str);

    impl std::fmt::Display for FakeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl Body for FakeBody {
        type Data = Bytes;
        type Error = FakeError;

        fn poll_frame(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Polled {
            self.polls.pop_front().unwrap_or(Poll::Pending)
        }

        fn is_end_stream(&self) -> bool {
            self.end_stream
        }

        fn size_hint(&self) -> SizeHint {
            SizeHint::with_exact(42)
        }
    }

    fn fake(
        polls: impl IntoIterator<Item = Polled>,
        end_stream: bool,
    ) -> (EndAttestingBody<FakeBody>, RequestStreamEnd) {
        attest_request_end(FakeBody {
            polls: polls.into_iter().collect(),
            end_stream,
        })
    }

    fn data(bytes: &'static [u8]) -> Polled {
        Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(bytes)))))
    }

    fn trailers() -> Polled {
        Poll::Ready(Some(Ok(Frame::trailers(http::HeaderMap::new()))))
    }

    fn error(msg: &'static str) -> Polled {
        Poll::Ready(Some(Err(FakeError(msg))))
    }

    const END: Polled = Poll::Ready(None);

    /// Poll `body` once, with a no-op waker.
    fn poll_once(body: &mut EndAttestingBody<FakeBody>) -> Polled {
        let mut cx = Context::from_waker(std::task::Waker::noop());
        Pin::new(body).poll_frame(&mut cx)
    }

    /// Data, `Pending` and the size hint pass through unchanged, and none of
    /// them records an ending.
    #[test]
    fn frames_pending_and_size_hint_pass_through_without_recording() {
        let (mut body, end) = fake([data(b"abc"), Poll::Pending, data(b"de")], false);

        let first = poll_once(&mut body);
        assert!(
            matches!(&first, Poll::Ready(Some(Ok(f))) if f.data_ref() == Some(&Bytes::from_static(b"abc"))),
            "got {first:?}"
        );
        assert!(poll_once(&mut body).is_pending());
        let second = poll_once(&mut body);
        assert!(
            matches!(&second, Poll::Ready(Some(Ok(f))) if f.data_ref() == Some(&Bytes::from_static(b"de"))),
            "got {second:?}"
        );
        assert_eq!(body.size_hint().exact(), Some(42));
        assert_eq!(end.ending(), None, "no ending is known while data flows");
    }

    /// The end of a body whose inner reports END_STREAM is `Complete`.
    #[test]
    fn end_with_end_stream_records_complete() {
        let (mut body, end) = fake([data(b"abc"), END], true);
        let _ = poll_once(&mut body);
        assert!(matches!(poll_once(&mut body), Poll::Ready(None)));
        assert_eq!(end.ending(), Some(StreamEnding::Complete));
    }

    /// The end of a body whose inner does NOT report END_STREAM — hyper's
    /// clean end for a `RST_STREAM(NO_ERROR)` or a keepalive GOAWAY — is
    /// `EndedWithoutEndStream`, never `Complete`.
    #[test]
    fn end_without_end_stream_records_ended_without_end_stream() {
        let (mut body, end) = fake([data(b"abc"), END], false);
        let _ = poll_once(&mut body);
        assert!(matches!(poll_once(&mut body), Poll::Ready(None)));
        assert_eq!(end.ending(), Some(StreamEnding::EndedWithoutEndStream));
    }

    /// A body error is recorded as `ResetBeforeEnd` with the error's
    /// `Display`, and the error itself is forwarded unchanged.
    #[test]
    fn error_records_reset_before_end_and_is_forwarded() {
        let (mut body, end) = fake([data(b"abc"), error("stream reset: CANCEL")], false);
        let _ = poll_once(&mut body);
        let polled = poll_once(&mut body);
        assert!(
            matches!(
                &polled,
                Poll::Ready(Some(Err(FakeError("stream reset: CANCEL"))))
            ),
            "got {polled:?}"
        );
        assert_eq!(
            end.ending(),
            Some(StreamEnding::ResetBeforeEnd(
                "stream reset: CANCEL".to_owned()
            ))
        );
    }

    /// An error is never overwritten by a later end of the body, even one the
    /// inner body by then reports as END_STREAM.
    #[test]
    fn a_later_end_never_overwrites_an_error() {
        let (mut body, end) = fake([error("connection lost"), END], false);
        let _ = poll_once(&mut body);
        body.inner.end_stream = true;
        assert!(matches!(poll_once(&mut body), Poll::Ready(None)));
        assert!(body.is_end_stream());
        assert_eq!(
            end.ending(),
            Some(StreamEnding::ResetBeforeEnd("connection lost".to_owned()))
        );
    }

    /// The first record wins: an ending without END_STREAM is not upgraded
    /// to `Complete` by a later `is_end_stream()` of `true`.
    #[test]
    fn the_first_record_wins() {
        let (mut body, end) = fake([END], false);
        assert!(matches!(poll_once(&mut body), Poll::Ready(None)));
        body.inner.end_stream = true;
        assert!(body.is_end_stream());
        assert_eq!(end.ending(), Some(StreamEnding::EndedWithoutEndStream));
    }

    /// A trailers frame ends the body as far as tonic is concerned, so it is
    /// recorded by the END_STREAM rule — `Complete` if the inner body reports
    /// it — and forwarded.
    #[test]
    fn trailers_with_end_stream_record_complete() {
        let (mut body, end) = fake([trailers()], true);
        let polled = poll_once(&mut body);
        assert!(
            matches!(&polled, Poll::Ready(Some(Ok(f))) if f.is_trailers()),
            "got {polled:?}"
        );
        assert_eq!(end.ending(), Some(StreamEnding::Complete));
    }

    /// A trailers frame from a body that does not report END_STREAM is
    /// `EndedWithoutEndStream`.
    #[test]
    fn trailers_without_end_stream_record_ended_without_end_stream() {
        let (mut body, end) = fake([trailers()], false);
        let _ = poll_once(&mut body);
        assert_eq!(end.ending(), Some(StreamEnding::EndedWithoutEndStream));
    }

    /// An inner body already at END_STREAM when asked records `Complete` from
    /// `is_end_stream()` alone, unpolled: tonic's `body::Body::new` drops such
    /// a body without polling it, which is how a HEADERS-only request arrives.
    #[test]
    fn is_end_stream_true_records_complete_unpolled() {
        let (body, end) = fake([], true);
        assert!(body.is_end_stream());
        assert_eq!(end.ending(), Some(StreamEnding::Complete));
    }

    /// Nothing is recorded `Complete` while the inner body reports it has not
    /// ended — however often it is asked, and with data frames pulled.
    #[test]
    fn nothing_is_complete_while_the_inner_body_has_not_ended() {
        let (mut body, end) = fake([data(b"abc"), data(b"de")], false);
        for _ in 0..3 {
            assert!(!body.is_end_stream());
        }
        let _ = poll_once(&mut body);
        let _ = poll_once(&mut body);
        assert!(!body.is_end_stream());
        assert_eq!(end.ending(), None);
    }

    /// A `NamedService` the wrapper is tested around: drains the body it is
    /// handed and answers with the record it found in the extensions.
    #[derive(Clone)]
    struct Probe;

    impl NamedService for Probe {
        const NAME: &'static str = "test.Probe";
    }

    impl Service<http::Request<tonic::body::Body>> for Probe {
        type Response = Option<StreamEnding>;
        type Error = Infallible;
        type Future = Pin<
            Box<dyn std::future::Future<Output = Result<Option<StreamEnding>, Infallible>> + Send>,
        >;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, request: http::Request<tonic::body::Body>) -> Self::Future {
            Box::pin(async move {
                let end = request.extensions().get::<RequestStreamEnd>().cloned();
                let mut body = request.into_body();
                while let Some(Ok(_)) = body.frame().await {}
                Ok(end.and_then(|end| end.ending()))
            })
        }
    }

    /// [`AttestRequestEnd`] routes as the service it wraps, and hands every
    /// request to it with the record of the body it re-boxed: a body that
    /// ended with END_STREAM reads `Complete`, one that ended without it
    /// `EndedWithoutEndStream`.
    #[tokio::test]
    async fn attest_request_end_inserts_the_record_of_the_body_it_wraps() {
        use tower::ServiceExt;

        assert_eq!(
            <AttestRequestEnd<Probe> as NamedService>::NAME,
            "test.Probe"
        );

        let complete = AttestRequestEnd::new(Probe)
            .oneshot(http::Request::new(tonic::body::Body::new(
                http_body_util::Full::new(Bytes::from_static(b"payload")),
            )))
            .await
            .expect("infallible");
        assert_eq!(complete, Some(StreamEnding::Complete));

        let truncated = AttestRequestEnd::new(Probe)
            .oneshot(http::Request::new(tonic::body::Body::new(
                http_body_util::StreamBody::new(futures::stream::iter([Ok::<_, Infallible>(
                    Frame::data(Bytes::from_static(b"payload")),
                )])),
            )))
            .await
            .expect("infallible");
        assert_eq!(truncated, Some(StreamEnding::EndedWithoutEndStream));
    }

    // ── The body-progress record ─────────────────────────────────────────

    /// A [`FakeBody`] over `polls`, with its progress record.
    fn fake_with_progress(
        polls: impl IntoIterator<Item = Polled>,
    ) -> (EndAttestingBody<FakeBody>, RequestBodyProgress) {
        let (body, _end, progress) = attest_request(FakeBody {
            polls: polls.into_iter().collect(),
            end_stream: false,
        });
        (body, progress)
    }

    /// Advance the paused clock by one second.
    async fn a_second_later() {
        tokio::time::advance(std::time::Duration::from_secs(1)).await;
    }

    /// Each non-empty DATA frame stamps the record with tokio's `now` (the
    /// paused clock here) at the moment it is polled. `Pending`, an empty DATA
    /// frame, a trailers frame and the body's end carry no bytes from the
    /// client and leave the stamp where it was.
    #[tokio::test(start_paused = true)]
    async fn progress_stamps_each_non_empty_data_frame_at_tokio_now() {
        let (mut body, progress) = fake_with_progress([
            data(b"abc"),
            Poll::Pending,
            data(b""),
            data(b"de"),
            trailers(),
            END,
        ]);
        assert_eq!(progress.last_data_at(), None, "nothing is stamped unpolled");

        a_second_later().await;
        let first = tokio::time::Instant::now();
        assert!(poll_once(&mut body).is_ready());
        assert_eq!(progress.last_data_at(), Some(first), "a DATA frame stamps");

        for what in ["Pending", "an empty DATA frame"] {
            a_second_later().await;
            let _ = poll_once(&mut body);
            assert_eq!(
                progress.last_data_at(),
                Some(first),
                "{what} must not stamp"
            );
        }

        a_second_later().await;
        let second = tokio::time::Instant::now();
        assert!(poll_once(&mut body).is_ready());
        assert_eq!(
            progress.last_data_at(),
            Some(second),
            "a later DATA frame moves the stamp"
        );

        for what in ["a trailers frame", "the body's end"] {
            a_second_later().await;
            let _ = poll_once(&mut body);
            assert_eq!(
                progress.last_data_at(),
                Some(second),
                "{what} must not stamp"
            );
        }
    }

    /// A body error carries no bytes and does not stamp.
    #[tokio::test(start_paused = true)]
    async fn progress_is_not_stamped_by_a_body_error() {
        let (mut body, progress) = fake_with_progress([error("stream reset: CANCEL")]);
        let _ = poll_once(&mut body);
        assert_eq!(progress.last_data_at(), None);
    }

    /// A DATA frame polled at the very instant the record was made reads back
    /// as that instant, not as "never stamped": the record keeps "no stamp"
    /// apart from a stamp at offset zero.
    #[tokio::test(start_paused = true)]
    async fn progress_stamped_at_the_records_own_instant_reads_back_as_stamped() {
        let made = tokio::time::Instant::now();
        let (mut body, progress) = fake_with_progress([data(b"abc")]);
        assert!(poll_once(&mut body).is_ready());
        assert_eq!(progress.last_data_at(), Some(made));
    }

    /// The end record and the progress record [`attest_request`] returns are
    /// the ones its body writes: a body that ends with END_STREAM after one
    /// DATA frame reads `Complete` and stamped.
    #[tokio::test(start_paused = true)]
    async fn attest_request_returns_the_records_its_body_writes() {
        let (mut body, end, progress) = attest_request(FakeBody {
            polls: [data(b"abc"), END].into_iter().collect(),
            end_stream: true,
        });
        a_second_later().await;
        let _ = poll_once(&mut body);
        let _ = poll_once(&mut body);
        assert_eq!(end.ending(), Some(StreamEnding::Complete));
        assert_eq!(progress.last_data_at(), Some(tokio::time::Instant::now()));
    }

    /// A `NamedService` that drains the body it is handed and answers with
    /// the progress record it found in the extensions, read once the body is
    /// drained: `None` if there was no record, `Some(None)` if nothing
    /// stamped it.
    #[derive(Clone)]
    struct ProgressProbe;

    impl NamedService for ProgressProbe {
        const NAME: &'static str = "test.ProgressProbe";
    }

    impl Service<http::Request<tonic::body::Body>> for ProgressProbe {
        type Response = Option<Option<tokio::time::Instant>>;
        type Error = Infallible;
        type Future =
            Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Infallible>> + Send>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, request: http::Request<tonic::body::Body>) -> Self::Future {
            Box::pin(async move {
                let progress = request.extensions().get::<RequestBodyProgress>().cloned();
                let mut body = request.into_body();
                while let Some(Ok(_)) = body.frame().await {}
                Ok(progress.map(|p| p.last_data_at()))
            })
        }
    }

    /// [`AttestRequestEnd`] inserts the progress record of the body it
    /// re-boxed into every request, stamped or not: a body with bytes reads
    /// stamped at the paused `now` once drained, and an empty one carries a
    /// record nothing stamped.
    #[tokio::test(start_paused = true)]
    async fn attest_request_end_inserts_the_progress_record_of_the_body_it_wraps() {
        use tower::ServiceExt;

        a_second_later().await;
        let with_bytes = AttestRequestEnd::new(ProgressProbe)
            .oneshot(http::Request::new(tonic::body::Body::new(
                http_body_util::Full::new(Bytes::from_static(b"payload")),
            )))
            .await
            .expect("infallible");
        assert_eq!(with_bytes, Some(Some(tokio::time::Instant::now())));

        let empty = AttestRequestEnd::new(ProgressProbe)
            .oneshot(http::Request::new(tonic::body::Body::new(
                http_body_util::Full::new(Bytes::new()),
            )))
            .await
            .expect("infallible");
        assert_eq!(
            empty,
            Some(None),
            "an empty body still carries its (unstamped) record"
        );
    }
}
