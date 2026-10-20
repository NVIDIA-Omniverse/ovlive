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

// Upload ingest helpers for client-streaming RPCs (CreateLayer, UpdateLayer)
// and OMPF chunking for server-streaming RPCs (SubscribeReadLayer).
//
// Client-streaming (SDD §5.4), in two calls so the handler can authorize and
// admit between them, over an `UploadStream` — the request's message stream
// paired with its end-of-stream record (`grpc::request_end`):
//   1. `read_upload_header` reads the first message, which carries the header
//      (session_id, layer_name, etc.), and validates it. No chunk is read.
//   2. `read_upload_body` reads the remaining messages — OMPF byte chunks —
//      under the upload's admission reservation, enforcing the size and
//      chunk-count caps.
// An upload is applied only if its stream ended with END_STREAM: wherever the
// stream ends, the record decides, and every other ending — a reset, a lost
// connection, a GOAWAY — is `CANCELLED` before anything is applied.
//
// Server-streaming (SDD §12.7):
//   Large OMPF blobs are chunked at ~1 MB boundaries. First message includes
//   the header (object_id, sequence_num, total_size, timestamps). Subsequent
//   messages contain only chunk data.
//
// The `StreamingRequest` trait abstracts the header-or-chunk message shape so
// one implementation serves both CreateLayer and UpdateLayer.

use bytes::Bytes;
use tonic::{Request, Streaming};

use crate::error::ServiceError;
use crate::grpc::proto::{
    create_layer_request, subscribe_read_layer_response, update_layer_request,
    ChainedUpdateLayerHeader, CreateLayerHeader, CreateLayerRequest, SubscribeReadLayerChunk,
    SubscribeReadLayerHeader, SubscribeReadLayerResponse, UpdateLayerHeader, UpdateLayerRequest,
};
use crate::grpc::request_end::{RequestStreamEnd, StreamEnding};
use crate::grpc::upload_admission::{
    UploadRefusal, UploadReservation, FIXED_CHARGE_BYTES, RECEIVE_CHARGE_FACTOR,
};

/// Chunk size for server-streaming OMPF delivery (1 MiB, SDD §12.7).
const OMPF_CHUNK_SIZE: usize = 1_048_576;

/// Header fields for an OMPF server-streaming delivery.
pub struct OmpfStreamHeader {
    pub object_id: u64,
    pub sequence_num: u64,
    pub server_timestamp: u64,
    pub client_timestamp: u64,
}

/// Build a sequence of `SubscribeReadLayerResponse` messages from OMPF bytes.
///
/// The first message contains the header (object_id, sequence_num, total_size,
/// timestamps). Subsequent messages contain data chunks of up to 1 MiB each.
///
/// Returns an iterator so each proto message is allocated just in time and
/// the caller never has to hold a `Vec` of all chunks in memory alongside
/// the source buffer (b.jr8 lows). The only copy of each byte that lives
/// past `next()` is the one inside the yielded `SubscribeReadLayerChunk`.
pub fn chunk_ompf_for_stream<'a>(
    ompf_bytes: &'a [u8],
    header_fields: &OmpfStreamHeader,
) -> impl Iterator<Item = SubscribeReadLayerResponse> + 'a {
    let total_size = ompf_bytes.len() as u64;
    let header_msg = SubscribeReadLayerResponse {
        response_type: Some(subscribe_read_layer_response::ResponseType::Header(
            SubscribeReadLayerHeader {
                object_id: header_fields.object_id,
                sequence_num: header_fields.sequence_num,
                total_size,
                client_timestamp_microseconds_since_epoch: header_fields.client_timestamp,
                server_timestamp_microseconds_since_epoch: header_fields.server_timestamp,
            },
        )),
    };

    std::iter::once(header_msg).chain(ompf_bytes.chunks(OMPF_CHUNK_SIZE).map(|chunk| {
        SubscribeReadLayerResponse {
            response_type: Some(subscribe_read_layer_response::ResponseType::Chunk(
                SubscribeReadLayerChunk {
                    data: chunk.to_vec(),
                },
            )),
        }
    }))
}

/// Trait abstracting the header-or-data oneof pattern shared by
/// `CreateLayerRequest` and `UpdateLayerRequest`.
pub trait StreamingRequest: Send + 'static {
    type Header;

    /// Extract the header variant from this message, if present.
    fn take_header(self) -> Result<HeaderOrData<Self::Header>, ServiceError>;
}

/// Discriminant returned by `StreamingRequest::take_header`.
pub enum HeaderOrData<H> {
    Header(H),
    Data(Vec<u8>),
    Empty,
}

// ── CreateLayerRequest ──────────────────────────────────────────────────────

impl StreamingRequest for CreateLayerRequest {
    type Header = CreateLayerHeader;

    fn take_header(self) -> Result<HeaderOrData<Self::Header>, ServiceError> {
        match self.message_type {
            Some(create_layer_request::MessageType::Header(h)) => Ok(HeaderOrData::Header(h)),
            Some(create_layer_request::MessageType::Chunk(c)) => Ok(HeaderOrData::Data(c.data)),
            None => Ok(HeaderOrData::Empty),
        }
    }
}

// ── UpdateLayerRequest ──────────────────────────────────────────────────────

/// The header an `UpdateLayer` stream opens with: the plain `header` arm, or
/// the `chained_header` arm of a chained call (SDD §23.1). Either is the
/// stream's one header: a second message of either arm is the second-header
/// `INVALID_ARGUMENT` of [`read_upload_body`].
#[derive(Debug, Clone)]
pub enum UpdateLayerFirstHeader {
    /// `UpdateLayerRequest.header`.
    Plain(UpdateLayerHeader),
    /// `UpdateLayerRequest.chained_header`, as received: the handler's
    /// header validation refuses one without its `header`.
    Chained(ChainedUpdateLayerHeader),
}

impl StreamingRequest for UpdateLayerRequest {
    type Header = UpdateLayerFirstHeader;

    fn take_header(self) -> Result<HeaderOrData<Self::Header>, ServiceError> {
        match self.message_type {
            Some(update_layer_request::MessageType::Header(h)) => {
                Ok(HeaderOrData::Header(UpdateLayerFirstHeader::Plain(h)))
            }
            Some(update_layer_request::MessageType::ChainedHeader(h)) => {
                Ok(HeaderOrData::Header(UpdateLayerFirstHeader::Chained(h)))
            }
            Some(update_layer_request::MessageType::Chunk(c)) => Ok(HeaderOrData::Data(c.data)),
            None => Ok(HeaderOrData::Empty),
        }
    }
}

/// Minimum bytes-per-chunk implied by the chunk-count cap (b.jr8 H-grpc-4).
/// The cap is derived as `max_size / MIN_AVG_CHUNK_BYTES` — an attacker
/// sending 1-byte chunks at the 100 MB max would otherwise pass 100M frames
/// through prost's decoder and exhaust CPU without ever tripping the
/// byte-count cap.
const MIN_AVG_CHUNK_BYTES: usize = 1024;

/// Compute the chunk-count cap for a given byte budget. Exposed as a
/// `const fn` so callers can reason about the exact ceiling in tests.
/// Guarantees at least 1 so `max_size < MIN_AVG_CHUNK_BYTES` still admits
/// a single tiny chunk.
pub const fn chunk_count_cap(max_size: usize) -> usize {
    let cap = max_size / MIN_AVG_CHUNK_BYTES;
    if cap == 0 {
        1
    } else {
        cap
    }
}

/// `INTERNAL` detail for an upload request with no end-of-stream record: the
/// LiveService was built without [`AttestRequestEnd`](crate::grpc::request_end::AttestRequestEnd).
const NO_END_RECORD: &str = "upload request carries no end-of-stream record; build the \
                             LiveService with grpc::transport::live_service_server";

/// `INTERNAL` detail for an upload stream that ended with its record unset.
/// Cannot happen with tonic 0.14.6, whose decoder ends a request stream only
/// after the body has ended or failed; a fail-closed backstop against a tonic
/// change.
const END_NOT_RECORDED: &str = "upload stream ended with no end-of-stream record";

/// An upload's request stream paired with its end-of-stream record, so no
/// read of it can skip the record: the upload readers take this, never a bare
/// `Streaming`.
///
/// Wherever the stream yields `Ok(None)`, the record decides what that ending
/// was (see [`crate::grpc::request_end`]): only END_STREAM is an end the
/// upload may be applied on.
pub struct UploadStream<R> {
    messages: Streaming<R>,
    end: RequestStreamEnd,
}

impl<R> UploadStream<R> {
    /// Take the request's message stream and its [`RequestStreamEnd`].
    ///
    /// The record is a request extension that
    /// [`AttestRequestEnd`](crate::grpc::request_end::AttestRequestEnd) — part
    /// of the stack [`live_service_server`](crate::grpc::transport::live_service_server)
    /// builds — inserts into every request; a handler-level test makes one
    /// with [`attest_request_end`](crate::grpc::request_end::attest_request_end)
    /// over its own body and inserts it. A request without one is refused
    /// `INTERNAL` here, before any message is read: without the record a reset
    /// upload would read as a complete one.
    pub fn from_request(request: Request<Streaming<R>>) -> Result<Self, ServiceError> {
        let end = request
            .extensions()
            .get::<RequestStreamEnd>()
            .cloned()
            .ok_or_else(|| ServiceError::Internal(NO_END_RECORD.to_owned()))?;
        Ok(Self {
            messages: request.into_inner(),
            end,
        })
    }

    /// What the stream's `Ok(None)` means: `Ok` only if the stream ended with
    /// END_STREAM. A reset, a lost connection or an ending without END_STREAM
    /// is `CANCELLED`; an unset record is `INTERNAL`.
    fn ended_with_end_stream(&self) -> Result<(), ServiceError> {
        match self.end.ending() {
            Some(StreamEnding::Complete) => Ok(()),
            Some(StreamEnding::ResetBeforeEnd(cause)) => Err(ServiceError::Cancelled(format!(
                "stream receive error: the upload stream was reset before it ended ({cause})"
            ))),
            Some(StreamEnding::EndedWithoutEndStream) => Err(ServiceError::Cancelled(
                "stream receive error: the upload stream ended without END_STREAM".to_owned(),
            )),
            None => Err(ServiceError::Internal(END_NOT_RECORDED.to_owned())),
        }
    }
}

/// Reads an upload's first message and validates its header. Reads no chunk.
///
/// - Reads the first message to extract the header.
/// - Invokes `validate_header` on the parsed header before returning; a
///   validation failure returns immediately. Nothing past the first message
///   is read here: the caller authorizes and admits the upload from the
///   header, and only then reads the content with [`read_upload_body`]. So
///   neither a bogus header nor an unauthorized caller makes the server
///   buffer content.
/// - A stream that ends before its first message is INVALID_ARGUMENT if it
///   ended with END_STREAM (the client sent nothing), and CANCELLED if it
///   ended any other way (reset, lost connection, GOAWAY). A receive error is
///   CANCELLED too.
/// - Returns INVALID_ARGUMENT if the first message is a chunk or carries
///   neither.
pub async fn read_upload_header<R, F>(
    stream: &mut UploadStream<R>,
    validate_header: F,
) -> Result<R::Header, ServiceError>
where
    R: StreamingRequest,
    F: FnOnce(&R::Header) -> Result<(), ServiceError>,
{
    // b.avd item 3: a transport error reading the *header* frame is classified
    // exactly like a transport error reading a *chunk* frame (see
    // `read_upload_body`) — both mean the client/transport went away after the
    // RPC was accepted but before we received a complete request. Map both to
    // `Cancelled` so metrics and alerting bucket them consistently and so
    // `Internal` stays reserved for genuine service faults. `Ok(None)` before
    // any header is `InvalidArgument` only when the client ended the stream
    // with END_STREAM — a malformed request shape. Any other ending is the
    // client or transport going away, `Cancelled` like a receive error.
    let Some(first) = stream
        .messages
        .message()
        .await
        .map_err(|e| ServiceError::Cancelled(format!("stream receive error: {e}")))?
    else {
        stream.ended_with_end_stream()?;
        return Err(ServiceError::InvalidArgument(
            "stream closed without sending any messages".to_owned(),
        ));
    };

    let header = match first.take_header()? {
        HeaderOrData::Header(h) => h,
        HeaderOrData::Data(_) => {
            return Err(ServiceError::InvalidArgument(
                "first message must be a header, got data chunk".to_owned(),
            ));
        }
        HeaderOrData::Empty => {
            return Err(ServiceError::InvalidArgument(
                "first message has no header or data".to_owned(),
            ));
        }
    };

    validate_header(&header)?;
    Ok(header)
}

/// Reads an admitted upload's content: every message after the header.
///
/// - Enforces `max_size`: INVALID_ARGUMENT if exceeded. Deliberately not
///   `RESOURCE_EXHAUSTED`, which clients retry: an upload over the cap can
///   never succeed, and every retry would re-send and re-buffer up to
///   `max_size` bytes before being refused again.
/// - Enforces a chunk-count cap of `max_size / 1024` (b.jr8 H-grpc-4):
///   INVALID_ARGUMENT if exceeded. Bounds frame-decode CPU when an attacker
///   dribbles sub-KiB chunks that individually stay under the byte cap.
/// - A second header is INVALID_ARGUMENT; a receive error is CANCELLED.
/// - At the end of the stream, applies the content only if the stream ended
///   with END_STREAM. Any other ending — a reset, a lost connection, a GOAWAY
///   — is CANCELLED, returned before the chunks are concatenated or the
///   reservation leaves its receive phase, so a truncated upload is never
///   handed on. The reservation is left for the caller's drop to release.
/// - After each chunk passes the caps, grows `reservation` by
///   [`RECEIVE_CHARGE_FACTOR`] × its length, in budget at once or not at all
///   ([`UploadReservation::try_grow_in_budget`]): a refusal answers its
///   [`UploadRefusal::into_service_error`], `UNAVAILABLE` — the budget is full
///   now. Never for size: the budget is floored so that the receive phase of
///   an upload under `max_size` fits it on its own.
/// - Keeps the chunks as received and concatenates them once, at an
///   END_STREAM end of the stream, into an exact-size buffer (no doubling
///   reallocation; a single chunk is handed on without a copy). Then drops the
///   stream — freeing tonic's receive buffer — ends the reservation's receive
///   phase and shrinks it to `FIXED_CHARGE_BYTES` + the body length.
/// - Empty content (header only, no data chunks) is valid when the stream
///   ended with END_STREAM — returns empty `Bytes`.
pub async fn read_upload_body<R>(
    mut stream: UploadStream<R>,
    max_size: usize,
    reservation: &mut UploadReservation,
) -> Result<Bytes, ServiceError>
where
    R: StreamingRequest,
{
    // What admission reserved (`F + A`); every receive charge is on top.
    let receive_base = reservation.reserved();
    let max_chunks = chunk_count_cap(max_size);
    let mut chunk_count: usize = 0;
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut received: usize = 0;

    while let Some(msg) = stream
        .messages
        .message()
        .await
        // Mid-upload transport error — the client sent a header, began
        // streaming chunks, and then the connection dropped (or a peer
        // aborted the RPC). Map to `Cancelled` (b.jr8 mediums): this is not
        // an internal service fault, it's the client/transport going away
        // after we'd already started accepting the upload. Reserving
        // `Internal` for genuine service faults keeps metrics and alerting
        // usable.
        .map_err(|e| ServiceError::Cancelled(format!("stream receive error: {e}")))?
    {
        // b.avd item 2: count every post-header frame against `max_chunks`,
        // not just Data frames. Pre-fix the `HeaderOrData::Empty` arm
        // skipped the increment, letting an attacker stream unbounded
        // `message_type: None` frames forever — never tripping any cap
        // while burning prost-decode CPU. Counting all frames here closes
        // that DoS shape and keeps `max_chunks` the single source of truth
        // for "how many frames are we willing to accept past the header".
        chunk_count += 1;
        if chunk_count > max_chunks {
            return Err(ServiceError::InvalidArgument(format!(
                "streaming chunk count exceeded {max_chunks}"
            )));
        }

        match msg.take_header()? {
            HeaderOrData::Data(chunk) => {
                if received + chunk.len() > max_size {
                    return Err(ServiceError::InvalidArgument(format!(
                        "layer content exceeds maximum size of {max_size} bytes"
                    )));
                }
                received += chunk.len();
                let received_bytes = u64::try_from(received).unwrap_or(u64::MAX);
                reservation
                    .try_grow_in_budget(
                        receive_base
                            .saturating_add(RECEIVE_CHARGE_FACTOR.saturating_mul(received_bytes)),
                    )
                    .map_err(UploadRefusal::into_service_error)?;
                if !chunk.is_empty() {
                    chunks.push(chunk);
                }
            }
            HeaderOrData::Header(_) => {
                return Err(ServiceError::InvalidArgument(
                    "unexpected second header message in stream".to_owned(),
                ));
            }
            HeaderOrData::Empty => {
                // Skip empty messages (no oneof set); the chunk-count tick
                // above still applies so empty-frame floods are bounded.
            }
        }
    }

    // End of stream. Only END_STREAM ends an upload: a stream reset, lost or
    // ended without it is truncated, and is refused here, before the content
    // is assembled or anything downstream sees it.
    stream.ended_with_end_stream()?;

    // Drop the stream first: that frees tonic's receive buffer, which the
    // receive allowance released below was covering.
    drop(stream);
    let body = concatenate(chunks, received);
    reservation.end_receive();
    reservation.shrink_to(
        FIXED_CHARGE_BYTES.saturating_add(u64::try_from(body.len()).unwrap_or(u64::MAX)),
    );
    Ok(body)
}

/// Join the chunks into one exact-size buffer of `total` bytes, freeing each
/// chunk as it is copied. A single chunk is handed on without a copy.
fn concatenate(mut chunks: Vec<Vec<u8>>, total: usize) -> Bytes {
    if chunks.len() == 1 {
        return Bytes::from(chunks.swap_remove(0));
    }
    let mut body = Vec::with_capacity(total);
    for chunk in chunks {
        body.extend_from_slice(&chunk);
    }
    Bytes::from(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::BufMut;
    use prost::Message;
    // tonic 0.14 split the prost codec out into the `tonic-prost` crate.
    use tonic::codec::{BufferSettings, Streaming};
    use tonic_prost::ProstCodec;

    /// Decode `body` as a request stream of `T`, paired with its end-of-stream
    /// record as the LiveService stack pairs them.
    fn upload_stream<T, B>(body: B) -> UploadStream<T>
    where
        T: Message + Default + Send + 'static,
        B: http_body::Body<Data = Bytes> + Unpin + Send + 'static,
        B::Error: std::fmt::Display + Into<tonic::codegen::StdError>,
    {
        let (body, end) = crate::grpc::request_end::attest_request_end(body);
        let decoder = ProstCodec::<(), T>::raw_decoder(BufferSettings::default());
        let mut request = Request::new(Streaming::new_request(decoder, body, None, None));
        request.extensions_mut().insert(end);
        UploadStream::from_request(request).expect("the request carries its record")
    }

    /// Encode protobuf messages into a gRPC-framed body that ends with
    /// END_STREAM, decoded as an [`UploadStream`] with its record. Each
    /// message is framed as:
    ///   [0u8 compression_flag] [4-byte big-endian length] [protobuf bytes]
    fn streaming_from_messages<T: Message + Default + Send + 'static>(
        messages: Vec<T>,
    ) -> UploadStream<T> {
        let mut buf = bytes::BytesMut::new();
        for msg in &messages {
            let encoded = msg.encode_to_vec();
            buf.put_u8(0); // no compression
            buf.put_u32(encoded.len() as u32);
            buf.extend_from_slice(&encoded);
        }
        upload_stream(http_body_util::Full::new(buf.freeze()))
    }

    /// Build an empty body (no messages at all) that ended with END_STREAM.
    fn streaming_empty<T: Message + Default + Send + 'static>() -> UploadStream<T> {
        upload_stream(http_body_util::Full::new(Bytes::new()))
    }

    fn make_header() -> CreateLayerRequest {
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Header(
                CreateLayerHeader {
                    session_id: "sess-1".to_owned(),
                    layer_name: "layer-1".to_owned(),
                    operation_id: String::new(),
                },
            )),
        }
    }

    fn make_data(payload: Vec<u8>) -> CreateLayerRequest {
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Chunk(
                crate::grpc::proto::CreateLayerChunk { data: payload },
            )),
        }
    }

    /// The handlers' two-step ingest — [`read_upload_header`], then
    /// [`read_upload_body`] under a reservation from a gate at the production
    /// defaults, which every fixture here fits.
    async fn ingest<R, F>(
        mut stream: UploadStream<R>,
        max_size: usize,
        validate_header: F,
    ) -> Result<(R::Header, Bytes), ServiceError>
    where
        R: StreamingRequest,
        F: FnOnce(&R::Header) -> Result<(), ServiceError>,
    {
        use crate::grpc::upload_admission::{
            UploadAdmissionConfig, UploadAdmissionGate, UploadRpc,
        };

        let header = read_upload_header(&mut stream, validate_header).await?;
        let gate = std::sync::Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig::default(),
            None,
        ));
        let mut reservation = gate
            .admit(
                UploadRpc::CreateLayer,
                "sess-1",
                &tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("an idle gate at the default budget admits at once");
        let body = read_upload_body(stream, max_size, &mut reservation).await?;
        Ok((header, body))
    }

    // ── Empty stream ────────────────────────────────────────────────

    #[tokio::test]
    async fn ingest_empty_stream_returns_invalid_argument() {
        let stream = streaming_empty::<CreateLayerRequest>();
        let err = ingest(stream, 1024, |_| Ok(())).await.unwrap_err();
        assert!(
            matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("without sending")),
            "expected InvalidArgument about empty stream, got: {err}"
        );
    }

    // ── First message is data (no header) ───────────────────────────

    #[tokio::test]
    async fn ingest_first_message_data_returns_invalid_argument() {
        let stream = streaming_from_messages(vec![make_data(vec![1, 2, 3])]);
        let err = ingest(stream, 1024, |_| Ok(())).await.unwrap_err();
        assert!(
            matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("header")),
            "expected InvalidArgument about missing header, got: {err}"
        );
    }

    // ── Header only, no data chunks ─────────────────────────────────

    #[tokio::test]
    async fn ingest_header_only_returns_empty_bytes() {
        let stream = streaming_from_messages(vec![make_header()]);
        let (header, body) = ingest(stream, 1024, |_| Ok(())).await.unwrap();
        assert_eq!(header.session_id, "sess-1");
        assert_eq!(header.layer_name, "layer-1");
        assert!(body.is_empty());
    }

    // ── Header + single chunk ───────────────────────────────────────

    #[tokio::test]
    async fn ingest_header_plus_single_chunk_returns_chunk_bytes() {
        let payload = vec![0xAA; 100];
        let stream = streaming_from_messages(vec![make_header(), make_data(payload.clone())]);
        let (header, body) = ingest(stream, 1024, |_| Ok(())).await.unwrap();
        assert_eq!(header.session_id, "sess-1");
        assert_eq!(body.as_ref(), &payload);
    }

    // ── Header + multiple chunks → concatenated ─────────────────────

    #[tokio::test]
    async fn ingest_header_plus_multiple_chunks_returns_concatenated() {
        let chunk_a = vec![0x01; 50];
        let chunk_b = vec![0x02; 75];
        let chunk_c = vec![0x03; 25];
        let stream = streaming_from_messages(vec![
            make_header(),
            make_data(chunk_a.clone()),
            make_data(chunk_b.clone()),
            make_data(chunk_c.clone()),
        ]);
        // max_size chosen so chunk_count_cap(max_size) >= 3 — otherwise
        // the b.jr8 H-grpc-4 chunk-count cap rejects before reaching the
        // byte-limit check this test exercises.
        let max_size = 4096;
        assert!(chunk_count_cap(max_size) >= 3);
        let (_, body) = ingest(stream, max_size, |_| Ok(())).await.unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&chunk_a);
        expected.extend_from_slice(&chunk_b);
        expected.extend_from_slice(&chunk_c);
        assert_eq!(body.as_ref(), &expected);
    }

    // ── Exactly at size limit → succeeds ────────────────────────────

    #[tokio::test]
    async fn ingest_exactly_at_limit_succeeds() {
        let max_size = 1000;
        let payload = vec![0xFF; max_size];
        let stream = streaming_from_messages(vec![make_header(), make_data(payload.clone())]);
        let (_, body) = ingest(stream, max_size, |_| Ok(())).await.unwrap();
        assert_eq!(body.len(), max_size);
    }

    // ── One byte over limit → INVALID_ARGUMENT ──────────────────────

    #[tokio::test]
    async fn ingest_one_byte_over_limit_returns_invalid_argument() {
        let max_size = 1000;
        let payload = vec![0xFF; max_size + 1];
        let stream = streaming_from_messages(vec![make_header(), make_data(payload)]);
        let err = ingest(stream, max_size, |_| Ok(())).await.unwrap_err();
        assert!(
            matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("exceeds")),
            "expected InvalidArgument, got: {err}"
        );
    }

    // ── Multiple chunks that together exceed limit ──────────────────

    #[tokio::test]
    async fn ingest_multiple_chunks_exceeding_limit_returns_invalid_argument() {
        // Use max_size large enough that chunk_count_cap admits both chunks,
        // so the chunk-count cap (b.jr8 H-grpc-4) does not fire first — we
        // specifically want the byte-limit cap to be what trips.
        let max_size = 4096;
        assert!(chunk_count_cap(max_size) >= 2);
        let stream = streaming_from_messages(vec![
            make_header(),
            make_data(vec![0x01; 3000]),
            make_data(vec![0x02; 3000]), // total 6000 > 4096
        ]);
        let err = ingest(stream, max_size, |_| Ok(())).await.unwrap_err();
        assert!(
            matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("exceeds")),
            "expected InvalidArgument, got: {err}"
        );
    }

    // ── Second header after first → error ───────────────────────────

    #[tokio::test]
    async fn ingest_second_header_returns_invalid_argument() {
        // Use a large enough `max_size` that the chunk-count cap
        // (`max_size / MIN_AVG_CHUNK_BYTES`) accommodates the two
        // post-header frames in this fixture; otherwise the second-header
        // arm we want to exercise hides behind the chunk-count cap added
        // in b.avd item 2.
        let stream = streaming_from_messages(vec![
            make_header(),
            make_data(vec![1, 2, 3]),
            make_header(), // unexpected second header
        ]);
        let err = ingest(stream, 16_384, |_| Ok(())).await.unwrap_err();
        assert!(
            matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("second header")),
            "expected InvalidArgument about second header, got: {err}"
        );
    }

    // ── First message with no oneof set → error ─────────────────────

    #[tokio::test]
    async fn ingest_first_message_empty_oneof_returns_invalid_argument() {
        let empty_msg = CreateLayerRequest { message_type: None };
        let stream = streaming_from_messages(vec![empty_msg]);
        let err = ingest(stream, 1024, |_| Ok(())).await.unwrap_err();
        assert!(
            matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("no header")),
            "expected InvalidArgument about no header, got: {err}"
        );
    }

    // ── b.avd item 2: empty-frame flood is bounded by chunk_count cap ───
    //
    // Pre-fix, `HeaderOrData::Empty` skipped without ticking `chunk_count`,
    // so an attacker could stream unbounded `message_type: None` frames
    // forever — never tripping the byte cap and never tripping any chunk
    // cap, while burning prost-decode CPU. Post-fix, every post-header
    // frame counts against `max_chunks`, so even an empty-frame flood is
    // bounded by the same DoS contract as a data flood.

    #[tokio::test]
    async fn ingest_empty_frame_flood_hits_chunk_count_cap() {
        // `max_size = 4096` → `max_chunks = 4`. Stream a header followed
        // by 1000 empty frames; only 4 (chunk_count > max_chunks fires on
        // the 5th frame) should be consumed before the cap fires.
        let max_size = 4096;
        let cap = chunk_count_cap(max_size);
        assert_eq!(cap, 4, "test assumes chunk_count_cap(4096) == 4");

        let mut messages = vec![make_header()];
        for _ in 0..1_000 {
            messages.push(CreateLayerRequest { message_type: None });
        }
        let stream = streaming_from_messages(messages);
        let err = ingest(stream, max_size, |_| Ok(())).await.unwrap_err();
        assert!(
            matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("chunk count exceeded")),
            "empty-frame flood must trip the chunk-count cap (b.avd item 2), got: {err}"
        );
    }

    #[tokio::test]
    async fn ingest_mixed_empty_and_data_frames_share_the_chunk_cap() {
        // A mix of two empty frames + two data frames at `max_size = 4096`
        // (cap = 4) commits cleanly because total frames == cap. Three
        // empty frames + two data frames at the same cap must fail —
        // empties no longer escape the count.
        let max_size = 4096;
        let cap = chunk_count_cap(max_size);
        assert_eq!(cap, 4);

        // Just-at-cap path → success.
        let messages_ok = vec![
            make_header(),
            CreateLayerRequest { message_type: None },
            CreateLayerRequest { message_type: None },
            make_data(vec![1, 2, 3]),
            make_data(vec![4, 5, 6]),
        ];
        let stream = streaming_from_messages(messages_ok);
        let (header, body) = ingest(stream, max_size, |_| Ok(()))
            .await
            .expect("mixed cap-exact stream must succeed");
        assert_eq!(header.session_id, "sess-1");
        assert_eq!(body.len(), 6); // 3 + 3 bytes

        // One-over-cap path → InvalidArgument.
        let messages_over = vec![
            make_header(),
            CreateLayerRequest { message_type: None },
            CreateLayerRequest { message_type: None },
            CreateLayerRequest { message_type: None },
            make_data(vec![1, 2, 3]),
            make_data(vec![4, 5, 6]),
        ];
        let stream = streaming_from_messages(messages_over);
        let err = ingest(stream, max_size, |_| Ok(())).await.unwrap_err();
        assert!(
            matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("chunk count exceeded")),
            "mixed-empty-and-data flood must share the cap (b.avd item 2), got: {err}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // The split: what each half reads, and the reservation the body grows
    // ═══════════════════════════════════════════════════════════════════

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use http_body::Frame;

    use crate::grpc::upload_admission::{
        admission_charge, UploadAdmissionConfig, UploadAdmissionGate, UploadRpc,
        RECEIVE_ALLOWANCE_BYTES,
    };

    /// One message as its own gRPC frame.
    fn frame<T: Message>(msg: &T) -> Bytes {
        let encoded = msg.encode_to_vec();
        let mut buf = bytes::BytesMut::with_capacity(5 + encoded.len());
        buf.put_u8(0);
        buf.put_u32(encoded.len() as u32);
        buf.extend_from_slice(&encoded);
        buf.freeze()
    }

    type BoxError = Box<dyn std::error::Error + Send + Sync>;

    /// How a [`ScriptBody`] ends once its data frames are pulled.
    enum Ending {
        /// END_STREAM: `is_end_stream()` turns true once the data is pulled.
        EndStream,
        /// The body's error, as a reset or a lost connection produces it.
        Error(BoxError),
        /// The end of the body without END_STREAM: hyper's shape for a
        /// `RST_STREAM(NO_ERROR)` and for the server's keepalive `GOAWAY`.
        WithoutEndStream,
    }

    /// A request body yielding one HTTP/2 DATA frame per entry of `frames`,
    /// then its [`Ending`], reporting `is_end_stream()` as h2 does — true
    /// exactly when END_STREAM arrived and no data is left — and counting the
    /// data frames pulled out of it.
    struct ScriptBody {
        frames: std::collections::VecDeque<Bytes>,
        ending: Option<Ending>,
        pulled: Arc<AtomicUsize>,
    }

    impl http_body::Body for ScriptBody {
        type Data = Bytes;
        type Error = BoxError;

        fn poll_frame(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Result<Frame<Bytes>, BoxError>>> {
            if let Some(data) = self.frames.pop_front() {
                self.pulled.fetch_add(1, Ordering::SeqCst);
                return std::task::Poll::Ready(Some(Ok(Frame::data(data))));
            }
            match self.ending.take() {
                Some(Ending::Error(e)) => std::task::Poll::Ready(Some(Err(e))),
                Some(Ending::EndStream) => {
                    // Keep reporting END_STREAM after it has been read.
                    self.ending = Some(Ending::EndStream);
                    std::task::Poll::Ready(None)
                }
                Some(Ending::WithoutEndStream) | None => std::task::Poll::Ready(None),
            }
        }

        fn is_end_stream(&self) -> bool {
            self.frames.is_empty() && matches!(self.ending, Some(Ending::EndStream))
        }
    }

    /// A [`ScriptBody`] over `messages` ending as `ending`, decoded as an
    /// upload stream with its record, and the count of data frames pulled.
    fn counted_stream(
        messages: Vec<CreateLayerRequest>,
        ending: Ending,
    ) -> (UploadStream<CreateLayerRequest>, Arc<AtomicUsize>) {
        let pulled = Arc::new(AtomicUsize::new(0));
        let body = ScriptBody {
            frames: messages.iter().map(frame).collect(),
            ending: Some(ending),
            pulled: Arc::clone(&pulled),
        };
        (upload_stream(body), pulled)
    }

    /// The body error a client's `RST_STREAM(CANCEL)` produces, which tonic's
    /// request decoder turns into a clean end of stream (`Ok(None)`).
    fn h2_cancel() -> Ending {
        Ending::Error(Box::new(h2::Error::from(h2::Reason::CANCEL)))
    }

    fn gate(budget_bytes: u64) -> Arc<UploadAdmissionGate> {
        Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig {
                budget_bytes,
                // No size-cap floor, so these fixtures can use small budgets;
                // `read_upload_body` enforces its own `max_size` argument.
                max_upload_bytes: 0,
                ..UploadAdmissionConfig::default()
            },
            None,
        ))
    }

    async fn admitted(gate: &Arc<UploadAdmissionGate>, session: &str) -> UploadReservation {
        gate.admit(
            UploadRpc::CreateLayer,
            session,
            &tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("the gate has room")
    }

    /// The header read pulls exactly the header's frame: the content stays in
    /// the transport until the caller has authorized and admitted the upload.
    #[tokio::test]
    async fn read_upload_header_pulls_only_the_header_frame() {
        let (mut stream, pulled) = counted_stream(
            vec![
                make_header(),
                make_data(vec![1; 64]),
                make_data(vec![2; 64]),
            ],
            Ending::EndStream,
        );
        let header = read_upload_header(&mut stream, |_| Ok(()))
            .await
            .expect("a valid header");
        assert_eq!(header.session_id, "sess-1");
        assert_eq!(
            pulled.load(Ordering::SeqCst),
            1,
            "reading the header must not pull any content frame"
        );
    }

    /// A header the validator refuses is refused with only its own frame
    /// pulled.
    #[tokio::test]
    async fn read_upload_header_refused_by_the_validator_pulls_only_the_header_frame() {
        let (mut stream, pulled) = counted_stream(
            vec![make_header(), make_data(vec![1; 64])],
            Ending::EndStream,
        );
        let err = read_upload_header(&mut stream, |_| {
            Err(ServiceError::InvalidArgument("refused".to_owned()))
        })
        .await
        .expect_err("the validator refuses");
        assert!(matches!(err, ServiceError::InvalidArgument(ref m) if m == "refused"));
        assert_eq!(pulled.load(Ordering::SeqCst), 1);
    }

    /// While the content is read the reservation grows by twice each chunk
    /// (the chunk and its share of the concatenation); at the end it gives
    /// back the receive allowance and the session slot and shrinks to `F` +
    /// the body length.
    #[tokio::test]
    async fn read_upload_body_grows_twice_each_chunk_then_shrinks_to_the_body() {
        let g = gate(UploadAdmissionConfig::default().budget_bytes);
        let mut reservation = admitted(&g, "sess-1").await;
        let (mut stream, _) = counted_stream(
            vec![
                make_header(),
                make_data(vec![1; 100]),
                make_data(vec![2; 200]),
            ],
            Ending::EndStream,
        );
        read_upload_header(&mut stream, |_| Ok(()))
            .await
            .expect("a valid header");

        let body = read_upload_body(stream, 4096, &mut reservation)
            .await
            .expect("the content fits");

        assert_eq!(body.len(), 300);
        assert_eq!(
            g.high_water_bytes(),
            admission_charge() + RECEIVE_CHARGE_FACTOR * 300,
            "the receive phase charges F + A + 2 × the bytes received"
        );
        assert!(!reservation.is_receiving());
        assert_eq!(g.receiving_uploads("sess-1"), 0);
        assert_eq!(reservation.reserved(), FIXED_CHARGE_BYTES + 300);
        assert_eq!(g.reserved_bytes(), FIXED_CHARGE_BYTES + 300);
    }

    /// A chunk whose charge the budget cannot hold right now is refused
    /// `UNAVAILABLE` — nothing applied, retry — at once, with the retry hint
    /// `UPLOAD_CAPACITY` (`cause` = `growth`).
    #[tokio::test]
    async fn read_upload_body_growth_the_budget_cannot_hold_now_is_unavailable() {
        const CHUNK: usize = 100;
        let budget = 4 * RECEIVE_ALLOWANCE_BYTES;
        let g = gate(budget);
        let mut reservation = admitted(&g, "sess-1").await;
        let mut other = admitted(&g, "other").await;
        // Leave less room than the chunk's charge.
        other
            .try_grow_in_budget(budget - admission_charge() - CHUNK as u64)
            .expect("fill the budget");
        let (mut stream, _) = counted_stream(
            vec![make_header(), make_data(vec![1; CHUNK])],
            Ending::EndStream,
        );
        read_upload_header(&mut stream, |_| Ok(()))
            .await
            .expect("a valid header");

        let err = read_upload_body(stream, 4096, &mut reservation)
            .await
            .expect_err("the budget cannot hold the chunk now");
        match &err {
            ServiceError::UnavailableWithDetail { detail, .. } => {
                assert_eq!(
                    detail.reason,
                    crate::error::refusal_reason::UPLOAD_CAPACITY,
                    "{err:?}"
                );
                assert_eq!(
                    detail.metadata,
                    vec![(
                        crate::error::refusal_metadata::CAUSE,
                        crate::error::refusal_metadata::GROWTH.to_owned()
                    )],
                    "{err:?}"
                );
            }
            other => panic!(
                "a growth refusal must be UNAVAILABLE with the upload-capacity retry hint; got \
                 {other:?}"
            ),
        }
        assert_eq!(
            reservation.reserved(),
            admission_charge(),
            "a refused growth reserves nothing more"
        );
    }

    /// A transport error mid-content is `CANCELLED`, and the reservation is
    /// left for the caller's drop to release.
    #[tokio::test]
    async fn read_upload_body_receive_error_is_cancelled() {
        let g = gate(UploadAdmissionConfig::default().budget_bytes);
        let mut reservation = admitted(&g, "sess-1").await;
        let (mut stream, _) = counted_stream(
            vec![make_header(), make_data(vec![1; 64])],
            Ending::Error(Box::new(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "peer reset",
            ))),
        );
        read_upload_header(&mut stream, |_| Ok(()))
            .await
            .expect("a valid header");

        let err = read_upload_body(stream, 4096, &mut reservation)
            .await
            .expect_err("the transport failed mid-content");
        assert!(
            matches!(err, ServiceError::Cancelled(ref m) if m.contains("stream receive error")),
            "got {err:?}"
        );
        drop(reservation);
        assert_eq!(g.reserved_bytes(), 0);
        assert_eq!(g.receiving_uploads("sess-1"), 0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // What each `Ok(None)` means: the end-of-stream record decides
    // ═══════════════════════════════════════════════════════════════════
    //
    // tonic answers `Ok(None)` for END_STREAM and for endings that are not
    // one (a `RST_STREAM(CANCEL)`, which its decoder swallows; hyper's clean
    // end for `RST_STREAM(NO_ERROR)` and a keepalive GOAWAY). Only the record
    // tells them apart: END_STREAM proceeds, every other ending is
    // `CANCELLED`, and an unset record is `INTERNAL`.

    /// The `CANCELLED` detail for a stream reset before it ended.
    const RESET_BEFORE_END: &str =
        "stream receive error: the upload stream was reset before it ended";

    /// The `CANCELLED` detail for a stream ended without END_STREAM.
    const ENDED_WITHOUT_END_STREAM: &str =
        "stream receive error: the upload stream ended without END_STREAM";

    /// An upload stream over `messages` (then END_STREAM) whose request
    /// carries a record that is never written: the record of another body,
    /// never polled. The shape of a stream that ends with its record unset.
    fn stream_with_an_unset_record(
        messages: Vec<CreateLayerRequest>,
    ) -> UploadStream<CreateLayerRequest> {
        let (_unpolled, end) =
            crate::grpc::request_end::attest_request_end(http_body_util::Full::new(Bytes::new()));
        let mut buf = bytes::BytesMut::new();
        for msg in &messages {
            buf.extend_from_slice(&frame(msg));
        }
        let decoder = ProstCodec::<(), CreateLayerRequest>::raw_decoder(BufferSettings::default());
        let mut request = Request::new(Streaming::new_request(
            decoder,
            http_body_util::Full::new(buf.freeze()),
            None,
            None,
        ));
        request.extensions_mut().insert(end);
        UploadStream::from_request(request).expect("the request carries a record")
    }

    /// A request with no end-of-stream record — a LiveService stack built
    /// without `AttestRequestEnd` — is refused `INTERNAL` when the stream is
    /// taken, before any message is read.
    #[test]
    fn upload_stream_from_a_request_without_a_record_is_internal() {
        let decoder = ProstCodec::<(), CreateLayerRequest>::raw_decoder(BufferSettings::default());
        let request = Request::new(Streaming::new_request(
            decoder,
            http_body_util::Full::new(frame(&make_header())),
            None,
            None,
        ));
        let err = UploadStream::from_request(request)
            .err()
            .expect("a request without its record must be refused");
        assert!(
            matches!(err, ServiceError::Internal(ref m) if m == NO_END_RECORD),
            "got {err:?}"
        );
    }

    /// Before the first message, a stream reset by its client (tonic swallows
    /// the `CANCEL`, `Ok(None)`) is `CANCELLED`, not the `INVALID_ARGUMENT` a
    /// client that sent nothing and ended its stream gets.
    #[tokio::test]
    async fn read_upload_header_reset_before_any_message_is_cancelled() {
        let (mut stream, _) = counted_stream(Vec::new(), h2_cancel());
        let err = read_upload_header(&mut stream, |_| Ok(()))
            .await
            .expect_err("a reset stream has no header");
        assert!(
            matches!(err, ServiceError::Cancelled(ref m) if m.starts_with(RESET_BEFORE_END)),
            "got {err:?}"
        );
    }

    /// Before the first message, a stream that ended without END_STREAM is
    /// `CANCELLED`.
    #[tokio::test]
    async fn read_upload_header_ended_without_end_stream_before_any_message_is_cancelled() {
        let (mut stream, _) = counted_stream(Vec::new(), Ending::WithoutEndStream);
        let err = read_upload_header(&mut stream, |_| Ok(()))
            .await
            .expect_err("a truncated stream has no header");
        assert!(
            matches!(err, ServiceError::Cancelled(ref m) if m == ENDED_WITHOUT_END_STREAM),
            "got {err:?}"
        );
    }

    /// Before the first message, a client that sent nothing and ended its
    /// stream with END_STREAM sent a malformed request: `INVALID_ARGUMENT`.
    #[tokio::test]
    async fn read_upload_header_end_stream_before_any_message_is_invalid_argument() {
        let (mut stream, _) = counted_stream(Vec::new(), Ending::EndStream);
        let err = read_upload_header(&mut stream, |_| Ok(()))
            .await
            .expect_err("an empty stream has no header");
        assert!(
            matches!(err, ServiceError::InvalidArgument(ref m) if m == "stream closed without sending any messages"),
            "got {err:?}"
        );
    }

    /// Before the first message, a stream whose record was never written is
    /// `INTERNAL`: fail closed rather than guess what the ending was.
    #[tokio::test]
    async fn read_upload_header_with_an_unset_record_is_internal() {
        let mut stream = stream_with_an_unset_record(Vec::new());
        let err = read_upload_header(&mut stream, |_| Ok(()))
            .await
            .expect_err("an unset record must not read as any ending");
        assert!(
            matches!(err, ServiceError::Internal(ref m) if m == END_NOT_RECORDED),
            "got {err:?}"
        );
    }

    /// Read the header of a [`counted_stream`] over a header and one 64-byte
    /// chunk ending as `ending`, then its body under a fresh reservation.
    /// Returns the body's result and the reservation as the read left it.
    async fn body_ending(
        g: &Arc<UploadAdmissionGate>,
        ending: Ending,
    ) -> (Result<Bytes, ServiceError>, UploadReservation) {
        let mut reservation = admitted(g, "sess-1").await;
        let (mut stream, _) = counted_stream(vec![make_header(), make_data(vec![1; 64])], ending);
        read_upload_header(&mut stream, |_| Ok(()))
            .await
            .expect("a valid header");
        let result = read_upload_body(stream, 4096, &mut reservation).await;
        (result, reservation)
    }

    /// What a truncated body leaves behind: the error came back before the
    /// content was handed on — the reservation still in its receive phase,
    /// never shrunk to the body — and dropping the reservation releases it.
    fn assert_refused_before_the_body_was_handed_on(
        g: &Arc<UploadAdmissionGate>,
        reservation: UploadReservation,
    ) {
        assert!(
            reservation.is_receiving(),
            "the refusal must come before the reservation leaves its receive phase"
        );
        assert_eq!(
            reservation.reserved(),
            admission_charge() + RECEIVE_CHARGE_FACTOR * 64,
            "the reservation still holds F + A and the received chunk's growth: never shrunk to \
             the body"
        );
        drop(reservation);
        assert_eq!(g.reserved_bytes(), 0);
        assert_eq!(g.receiving_uploads("sess-1"), 0);
    }

    /// After content, a stream reset by its client (tonic swallows the
    /// `CANCEL`, `Ok(None)`) is `CANCELLED`, returned before the content is
    /// handed on.
    #[tokio::test]
    async fn read_upload_body_reset_after_content_is_cancelled_before_the_body_is_handed_on() {
        let g = gate(UploadAdmissionConfig::default().budget_bytes);
        let (result, reservation) = body_ending(&g, h2_cancel()).await;
        let err = result.expect_err("a reset upload must not be handed on");
        assert!(
            matches!(err, ServiceError::Cancelled(ref m) if m.starts_with(RESET_BEFORE_END)),
            "got {err:?}"
        );
        assert_refused_before_the_body_was_handed_on(&g, reservation);
    }

    /// After content, a stream that ended without END_STREAM is `CANCELLED`,
    /// returned before the content is handed on.
    #[tokio::test]
    async fn read_upload_body_ended_without_end_stream_is_cancelled_before_the_body_is_handed_on() {
        let g = gate(UploadAdmissionConfig::default().budget_bytes);
        let (result, reservation) = body_ending(&g, Ending::WithoutEndStream).await;
        let err = result.expect_err("a truncated upload must not be handed on");
        assert!(
            matches!(err, ServiceError::Cancelled(ref m) if m == ENDED_WITHOUT_END_STREAM),
            "got {err:?}"
        );
        assert_refused_before_the_body_was_handed_on(&g, reservation);
    }

    /// After content, END_STREAM hands the content on: the counterpart of the
    /// two refusals above over the same body.
    #[tokio::test]
    async fn read_upload_body_end_stream_after_content_hands_the_body_on() {
        let g = gate(UploadAdmissionConfig::default().budget_bytes);
        let (result, reservation) = body_ending(&g, Ending::EndStream).await;
        assert_eq!(
            result.expect("a finished upload is handed on").as_ref(),
            &[1u8; 64][..]
        );
        assert!(!reservation.is_receiving());
        assert_eq!(reservation.reserved(), FIXED_CHARGE_BYTES + 64);
    }

    /// After content, a stream whose record was never written is `INTERNAL`.
    #[tokio::test]
    async fn read_upload_body_with_an_unset_record_is_internal() {
        let g = gate(UploadAdmissionConfig::default().budget_bytes);
        let mut reservation = admitted(&g, "sess-1").await;
        let mut stream = stream_with_an_unset_record(vec![make_header(), make_data(vec![1; 64])]);
        read_upload_header(&mut stream, |_| Ok(()))
            .await
            .expect("a valid header");
        let err = read_upload_body(stream, 4096, &mut reservation)
            .await
            .expect_err("an unset record must not read as END_STREAM");
        assert!(
            matches!(err, ServiceError::Internal(ref m) if m == END_NOT_RECORDED),
            "got {err:?}"
        );
        assert!(reservation.is_receiving());
    }

    // ═══════════════════════════════════════════════════════════════════
    // chunk_ompf_for_stream tests
    // ═══════════════════════════════════════════════════════════════════

    fn test_stream_header() -> OmpfStreamHeader {
        OmpfStreamHeader {
            object_id: 42,
            sequence_num: 7,
            server_timestamp: 1_000_000,
            client_timestamp: 2_000_000,
        }
    }

    /// Extract the header from the first message; panic if missing.
    fn extract_header(msg: &SubscribeReadLayerResponse) -> &SubscribeReadLayerHeader {
        match &msg.response_type {
            Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
            other => panic!("expected Header, got: {other:?}"),
        }
    }

    /// Extract the data bytes from a message; panic if not data.
    fn extract_data(msg: &SubscribeReadLayerResponse) -> &[u8] {
        match &msg.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => &c.data,
            other => panic!("expected Chunk, got: {other:?}"),
        }
    }

    /// Reassemble all data chunks from messages (skip the header).
    fn reassemble_data(messages: &[SubscribeReadLayerResponse]) -> Vec<u8> {
        let mut out = Vec::new();
        for msg in &messages[1..] {
            out.extend_from_slice(extract_data(msg));
        }
        out
    }

    // ── 0 bytes → header only, no data chunks ─────────────────────────

    #[test]
    fn chunk_ompf_zero_bytes_returns_header_only() {
        let messages: Vec<_> = chunk_ompf_for_stream(&[], &test_stream_header()).collect();
        assert_eq!(messages.len(), 1, "0 bytes should produce header only");

        let header = extract_header(&messages[0]);
        assert_eq!(header.total_size, 0);
        assert_eq!(header.object_id, 42);
        assert_eq!(header.sequence_num, 7);
        assert_eq!(header.server_timestamp_microseconds_since_epoch, 1_000_000);
        assert_eq!(header.client_timestamp_microseconds_since_epoch, 2_000_000);
    }

    // ── 500 KB → 1 header + 1 data chunk ──────────────────────────────

    #[test]
    fn chunk_ompf_500kb_returns_header_plus_one_data() {
        let data = vec![0xAB; 500 * 1024]; // 500 KB < 1 MB chunk size
        let messages: Vec<_> = chunk_ompf_for_stream(&data, &test_stream_header()).collect();

        assert_eq!(
            messages.len(),
            2,
            "500 KB should produce 1 header + 1 data chunk"
        );

        let header = extract_header(&messages[0]);
        assert_eq!(header.total_size, data.len() as u64);

        let reassembled = reassemble_data(&messages);
        assert_eq!(reassembled, data, "reassembled bytes must match original");
    }

    // ── exactly 1 MB → 1 header + 1 data chunk ───────────────────────

    #[test]
    fn chunk_ompf_1mb_exact_returns_header_plus_one_data() {
        let data = vec![0xCD; OMPF_CHUNK_SIZE]; // exactly 1 MiB
        let messages: Vec<_> = chunk_ompf_for_stream(&data, &test_stream_header()).collect();

        assert_eq!(
            messages.len(),
            2,
            "exactly 1 MB should produce 1 header + 1 data chunk"
        );

        let header = extract_header(&messages[0]);
        assert_eq!(header.total_size, OMPF_CHUNK_SIZE as u64);

        let reassembled = reassemble_data(&messages);
        assert_eq!(reassembled, data, "reassembled bytes must match original");
    }

    // ── 1 MB + 1 byte → 1 header + 2 data chunks ─────────────────────

    #[test]
    fn chunk_ompf_1mb_plus_one_returns_header_plus_two_data() {
        let data = vec![0xEF; OMPF_CHUNK_SIZE + 1]; // 1 MiB + 1
        let messages: Vec<_> = chunk_ompf_for_stream(&data, &test_stream_header()).collect();

        assert_eq!(
            messages.len(),
            3,
            "1 MB + 1 byte should produce 1 header + 2 data chunks"
        );

        let header = extract_header(&messages[0]);
        assert_eq!(header.total_size, (OMPF_CHUNK_SIZE + 1) as u64);

        // First data chunk should be exactly 1 MB.
        assert_eq!(extract_data(&messages[1]).len(), OMPF_CHUNK_SIZE);
        // Second data chunk should be 1 byte.
        assert_eq!(extract_data(&messages[2]).len(), 1);

        let reassembled = reassemble_data(&messages);
        assert_eq!(reassembled, data, "reassembled bytes must match original");
    }

    // ── 5 MB → 1 header + 5 data chunks ──────────────────────────────

    #[test]
    fn chunk_ompf_5mb_returns_header_plus_five_data() {
        let data = vec![0x99; 5 * OMPF_CHUNK_SIZE]; // exactly 5 MiB
        let messages: Vec<_> = chunk_ompf_for_stream(&data, &test_stream_header()).collect();

        assert_eq!(
            messages.len(),
            6,
            "5 MB should produce 1 header + 5 data chunks"
        );

        let header = extract_header(&messages[0]);
        assert_eq!(header.total_size, (5 * OMPF_CHUNK_SIZE) as u64);

        // Each data chunk should be exactly 1 MB.
        for msg in &messages[1..] {
            assert_eq!(
                extract_data(msg).len(),
                OMPF_CHUNK_SIZE,
                "each chunk in a 5 MB payload should be exactly 1 MB"
            );
        }

        let reassembled = reassemble_data(&messages);
        assert_eq!(reassembled, data, "reassembled bytes must match original");
    }
}
