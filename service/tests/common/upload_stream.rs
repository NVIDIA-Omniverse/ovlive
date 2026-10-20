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

//! Handler-level requests that carry the records the LiveService stack
//! attaches to every request body (SDD §5.4, §13.2).
//!
//! `CreateLayer` and `UpdateLayer` apply an upload only if its request stream
//! ended with END_STREAM, which they read from the `RequestStreamEnd` in the
//! request's extensions (`live_edit::grpc::request_end`). `JoinSession` arms
//! the client keepalive deadline only for a request carrying its body's
//! `RequestBodyProgress`, which records when the body last carried bytes. A
//! real server's `live_service_server` stack inserts both; a test that calls
//! a handler directly must make them from its own body through
//! `attest_request`, or the upload handler refuses the upload `INTERNAL` and
//! an opted-in join is never armed (it is told interval 0). This module is
//! that seam:
//!
//! - [`AttestedStreaming`]: a request's message stream with both records, as
//!   the stack pairs them. [`IntoHandlerRequest`] turns it, or a bare
//!   `Streaming` (a JoinSession body that never opts in to the client
//!   keepalive, which needs neither record), into the handler's request.
//! - [`EndsWithEndStream`]: for a body whose `is_end_stream()` never turns
//!   true, such as `http_body_util::StreamBody`. It reports END_STREAM once
//!   the inner body has ended, as h2 does for a client that finished its
//!   stream. Without it every clean upload over such a body reads as
//!   truncated.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::{BufMut, Bytes};
use http_body::{Body, Frame, SizeHint};
use live_edit::grpc::request_end::{attest_request, RequestBodyProgress, RequestStreamEnd};
use prost::Message;
use tonic::codec::{BufferSettings, Streaming};
use tonic::Request;
use tonic_prost::ProstCodec;

/// A request stream paired with its end-of-stream record and its
/// body-progress record, as the LiveService stack pairs them.
pub struct AttestedStreaming<T> {
    stream: Streaming<T>,
    end: RequestStreamEnd,
    progress: RequestBodyProgress,
}

impl<T: Message + Default + Send + 'static> AttestedStreaming<T> {
    /// Decode `body` as a request stream of `T`, recording how `body` ends
    /// and when it last carried bytes.
    ///
    /// `body` must report `is_end_stream()` truthfully (see the module doc);
    /// a JoinSession body, whose handler never reads the end record, need not.
    pub fn new<B>(body: B) -> Self
    where
        B: Body<Data = Bytes> + Unpin + Send + 'static,
        B::Error: std::fmt::Display + Into<tonic::codegen::StdError>,
    {
        let (body, end, progress) = attest_request(body);
        let decoder = ProstCodec::<(), T>::raw_decoder(BufferSettings::default());
        Self {
            stream: Streaming::new_request(decoder, body, None, None),
            end,
            progress,
        }
    }

    /// `messages`, each as its own gRPC frame, in one body that ends with
    /// END_STREAM: a client that sent them all and finished its stream.
    pub fn from_messages(messages: &[T]) -> Self {
        Self::new(http_body_util::Full::new(grpc_frames(messages)))
    }
}

impl<T> AttestedStreaming<T> {
    /// The stream's record, for a test that asserts how its body ended.
    pub fn end(&self) -> RequestStreamEnd {
        self.end.clone()
    }
}

/// Each message as a gRPC frame — compression flag 0, big-endian length,
/// protobuf bytes — back to back.
pub fn grpc_frames<T: Message>(messages: &[T]) -> Bytes {
    let mut buf = bytes::BytesMut::new();
    for msg in messages {
        let encoded = msg.encode_to_vec();
        buf.put_u8(0);
        buf.put_u32(encoded.len() as u32);
        buf.extend_from_slice(&encoded);
    }
    buf.freeze()
}

/// What a handler-level test hands a handler: a request stream, as a request.
pub trait IntoHandlerRequest<T> {
    /// The request, carrying the stream's end-of-stream and body-progress
    /// records if it has them.
    fn into_handler_request(self) -> Request<Streaming<T>>;
}

/// A bare stream: no end-of-stream record and no body-progress record. Right
/// for a JoinSession that does not opt in to the client keepalive; one that
/// does is never armed without the progress record, and is told interval 0.
/// An upload handler refuses a bare stream `INTERNAL`.
impl<T> IntoHandlerRequest<T> for Streaming<T> {
    fn into_handler_request(self) -> Request<Streaming<T>> {
        Request::new(self)
    }
}

impl<T> IntoHandlerRequest<T> for AttestedStreaming<T> {
    fn into_handler_request(self) -> Request<Streaming<T>> {
        let mut request = Request::new(self.stream);
        request.extensions_mut().insert(self.end);
        request.extensions_mut().insert(self.progress);
        request
    }
}

/// A body that reports END_STREAM once its inner body has ended (yielded
/// `None`), and never before: the truthful `is_end_stream()` of a client that
/// finished its stream, for an inner body that never reports one.
pub struct EndsWithEndStream<B> {
    inner: B,
    ended: bool,
}

impl<B> EndsWithEndStream<B> {
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            ended: false,
        }
    }
}

impl<B> Body for EndsWithEndStream<B>
where
    B: Body + Unpin,
{
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        if let Poll::Ready(None) = polled {
            this.ended = true;
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.ended || self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}
