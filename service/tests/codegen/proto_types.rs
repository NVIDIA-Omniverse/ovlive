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

// Compile-time existence checks for proto-generated types.
//
// Compilation is the assertion: if tonic-build failed to generate a type, this file
// will not compile. Each test constructs a zero-value instance to confirm the type
// and its fields are accessible as expected.

tonic::include_proto!("nvidia.omniverse.live.v1alpha");

// Re-export the server module items needed for the trait and server checks.
use live_service_server::{LiveService, LiveServiceServer};
use tonic::{async_trait, Request, Response, Status, Streaming};

// ── Type construction tests ────────────────────────────────────────────────────

#[test]
fn ping_request_constructible() {
    let req = PingRequest {
        client_timestamp_microseconds_since_epoch: 0,
    };
    assert_eq!(req.client_timestamp_microseconds_since_epoch, 0);
}

#[test]
fn ping_response_constructible() {
    let resp = PingResponse {
        client_timestamp_microseconds_since_epoch: 0,
        server_timestamp_microseconds_since_epoch: 0,
        operation_ids: false,
        chained_writes: false,
    };
    assert_eq!(resp.server_timestamp_microseconds_since_epoch, 0);
}

#[test]
fn create_session_request_constructible() {
    let req = CreateSessionRequest {
        path: String::new(),
        name: String::new(),
    };
    assert!(req.path.is_empty());
}

#[test]
fn delete_session_request_constructible() {
    let req = DeleteSessionRequest {
        session_id: String::new(),
    };
    assert!(req.session_id.is_empty());
}

#[test]
fn list_sessions_request_constructible() {
    let req = ListSessionsRequest {
        path: String::new(),
    };
    assert!(req.path.is_empty());
}

#[test]
fn create_layer_request_constructible() {
    let req = CreateLayerRequest { message_type: None };
    assert!(req.message_type.is_none());
}

#[test]
fn update_layer_request_constructible() {
    let req = UpdateLayerRequest { message_type: None };
    assert!(req.message_type.is_none());
}

#[test]
fn delete_layer_request_constructible() {
    let req = DeleteLayerRequest {
        session_id: String::new(),
        layer_name: String::new(),
    };
    assert!(req.layer_name.is_empty());
}

#[test]
fn stat_layer_request_constructible() {
    let req = StatLayerRequest {
        session_id: String::new(),
        layer_name: String::new(),
    };
    assert!(req.layer_name.is_empty());
}

#[test]
fn subscribe_read_layer_request_constructible() {
    let req = SubscribeReadLayerRequest {
        session_id: String::new(),
        layer_name: String::new(),
        object_id: 0,
        sequence_num: 0,
        client_id: String::new(),
    };
    assert!(req.client_id.is_empty());
}

#[test]
fn join_session_request_constructible() {
    let msg = JoinSessionRequest { request_type: None };
    assert!(msg.request_type.is_none());
}

// ── LiveService trait and server check ────────────────────────────────────────

// A minimal stub that satisfies the LiveService trait. Compilation verifies that:
//   1. The trait exists with the correct method signatures.
//   2. All associated streaming types are defined.
//   3. LiveServiceServer::new() accepts a LiveService impl.
struct StubService;

#[async_trait]
impl LiveService for StubService {
    async fn create_session(
        &self,
        _request: Request<CreateSessionRequest>,
    ) -> Result<Response<CreateSessionResponse>, Status> {
        Err(Status::unimplemented("stub"))
    }

    async fn delete_session(
        &self,
        _request: Request<DeleteSessionRequest>,
    ) -> Result<Response<DeleteSessionResponse>, Status> {
        Err(Status::unimplemented("stub"))
    }

    async fn list_sessions(
        &self,
        _request: Request<ListSessionsRequest>,
    ) -> Result<Response<ListSessionsResponse>, Status> {
        Err(Status::unimplemented("stub"))
    }

    type JoinSessionStream = futures::stream::Empty<Result<JoinSessionResponse, Status>>;

    async fn join_session(
        &self,
        _request: Request<Streaming<JoinSessionRequest>>,
    ) -> Result<Response<Self::JoinSessionStream>, Status> {
        Err(Status::unimplemented("stub"))
    }

    async fn create_layer(
        &self,
        _request: Request<Streaming<CreateLayerRequest>>,
    ) -> Result<Response<CreateLayerResponse>, Status> {
        Err(Status::unimplemented("stub"))
    }

    async fn update_layer(
        &self,
        _request: Request<Streaming<UpdateLayerRequest>>,
    ) -> Result<Response<UpdateLayerResponse>, Status> {
        Err(Status::unimplemented("stub"))
    }

    async fn delete_layer(
        &self,
        _request: Request<DeleteLayerRequest>,
    ) -> Result<Response<DeleteLayerResponse>, Status> {
        Err(Status::unimplemented("stub"))
    }

    async fn stat_layer(
        &self,
        _request: Request<StatLayerRequest>,
    ) -> Result<Response<StatLayerResponse>, Status> {
        Err(Status::unimplemented("stub"))
    }

    type SubscribeReadLayerStream =
        futures::stream::Empty<Result<SubscribeReadLayerResponse, Status>>;

    async fn subscribe_read_layer(
        &self,
        _request: Request<SubscribeReadLayerRequest>,
    ) -> Result<Response<Self::SubscribeReadLayerStream>, Status> {
        Err(Status::unimplemented("stub"))
    }

    async fn ping(&self, _request: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        Err(Status::unimplemented("stub"))
    }
}

#[test]
fn live_service_server_accepts_impl() {
    // Compilation verifies LiveServiceServer::new accepts a LiveService impl.
    let _server = LiveServiceServer::new(StubService);
}

// ── Regression: wire-compatible field names (old C++ service) ────────────────
// These tests verify that the proto definitions use field names matching the
// original C++ live-edit service. If anyone accidentally reverts to the
// intermediate Rust-only field names, these tests will fail to compile.

#[test]
fn regression_create_session_uses_path_and_name_fields() {
    let req = CreateSessionRequest {
        path: "/stage/scene.usd".to_owned(),
        name: "my-session".to_owned(),
    };
    assert_eq!(req.path, "/stage/scene.usd");
    assert_eq!(req.name, "my-session");
}

#[test]
fn regression_session_uses_wire_compatible_field_names() {
    let session = Session {
        session_id: "sid-1".to_owned(),
        name: "session-name".to_owned(),
        created_by_user_name: "alice".to_owned(),
        created_at: 1_000_000,
    };
    assert_eq!(session.name, "session-name");
    assert_eq!(session.created_by_user_name, "alice");
    assert_eq!(session.created_at, 1_000_000);
}

#[test]
fn regression_list_sessions_uses_path_field() {
    let req = ListSessionsRequest {
        path: "/stage".to_owned(),
    };
    assert_eq!(req.path, "/stage");
}

#[test]
fn regression_update_layer_header_uses_timestamp_field() {
    let header = UpdateLayerHeader {
        session_id: "sid".to_owned(),
        layer_name: "layer".to_owned(),
        object_id: 1,
        timestamp_microseconds_since_epoch: 42,
        client_id: "c1".to_owned(),
        operation_id: String::new(),
    };
    assert_eq!(header.timestamp_microseconds_since_epoch, 42);
}

#[test]
fn regression_stat_layer_response_uses_created_at_and_last_modified_at() {
    let resp = StatLayerResponse {
        created_by_user_name: "bob".to_owned(),
        created_at: 100,
        last_modified_at: 200,
    };
    assert_eq!(resp.created_at, 100);
    assert_eq!(resp.last_modified_at, 200);
}

#[test]
fn regression_create_layer_request_uses_message_type_oneof_with_chunk() {
    use prost::Message;

    // Verify the oneof is named `message_type` (not `message`) and the data
    // variant uses the `CreateLayerChunk` wrapper (not bare `bytes`).
    let req = CreateLayerRequest {
        message_type: Some(create_layer_request::MessageType::Chunk(CreateLayerChunk {
            data: vec![0xAB, 0xCD],
        })),
    };
    // Round-trip through protobuf encoding to confirm serialization works.
    let encoded = req.encode_to_vec();
    let decoded = CreateLayerRequest::decode(encoded.as_slice()).unwrap();
    match decoded.message_type {
        Some(create_layer_request::MessageType::Chunk(chunk)) => {
            assert_eq!(chunk.data, vec![0xAB, 0xCD]);
        }
        other => panic!("expected Chunk variant, got: {other:?}"),
    }
}

#[test]
fn regression_update_layer_request_uses_message_type_oneof_with_chunk() {
    use prost::Message;

    // Verify the oneof is named `message_type` and the data variant uses
    // the `UpdateLayerChunk` wrapper (mirrors CreateLayerRequest pattern).
    let req = UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Chunk(UpdateLayerChunk {
            data: vec![0xDE, 0xAD],
        })),
    };
    let encoded = req.encode_to_vec();
    let decoded = UpdateLayerRequest::decode(encoded.as_slice()).unwrap();
    match decoded.message_type {
        Some(update_layer_request::MessageType::Chunk(chunk)) => {
            assert_eq!(chunk.data, vec![0xDE, 0xAD]);
        }
        other => panic!("expected Chunk variant, got: {other:?}"),
    }
}

#[test]
fn regression_subscribe_read_layer_response_uses_response_type_oneof_with_chunk() {
    use prost::Message;

    // Verify the oneof is named `response_type` and the data variant uses
    // the `SubscribeReadLayerChunk` wrapper.
    let resp = SubscribeReadLayerResponse {
        response_type: Some(subscribe_read_layer_response::ResponseType::Chunk(
            SubscribeReadLayerChunk {
                data: vec![0xBE, 0xEF],
            },
        )),
    };
    let encoded = resp.encode_to_vec();
    let decoded = SubscribeReadLayerResponse::decode(encoded.as_slice()).unwrap();
    match decoded.response_type {
        Some(subscribe_read_layer_response::ResponseType::Chunk(chunk)) => {
            assert_eq!(chunk.data, vec![0xBE, 0xEF]);
        }
        other => panic!("expected Chunk variant, got: {other:?}"),
    }
}

#[test]
fn regression_join_session_uses_request_and_response_types() {
    // Verify JoinSessionRequest (not JoinSessionClientMessage) with request_type oneof.
    let req = JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Header(
            JoinSessionHeader {
                session_id: "sid".to_owned(),
                user_application: "app".to_owned(),
                client_keepalive: false,
            },
        )),
    };
    assert!(req.request_type.is_some());

    // Verify JoinSessionResponse (not JoinSessionServerMessage) with response_type oneof.
    let resp = JoinSessionResponse {
        response_type: Some(join_session_response::ResponseType::Connected(
            SessionConnected {
                connection_id: "conn-1".to_owned(),
                keepalive_interval_ms: 0,
            },
        )),
    };
    assert!(resp.response_type.is_some());
}

// ── JoinSession client keepalive: field numbers are client contract ──────────
//
// The three keepalive fields are additive to a published proto, and clients
// built from other copies of it (and other server implementations) agree on
// them by number alone. Each is pinned by its wire bytes, so a renumbering is a
// red test rather than a silent wire break.

/// `JoinSessionRequest.keepalive` is field 3 of the `request_type` oneof, an
/// empty `JoinSessionKeepalive`: tag `0x1a` (field 3, length-delimited) and
/// length 0.
#[test]
fn join_session_keepalive_is_request_field_3_on_the_wire() {
    use prost::Message;
    let keepalive = JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Keepalive(
            JoinSessionKeepalive {},
        )),
    };
    assert_eq!(keepalive.encode_to_vec(), vec![0x1a, 0x00]);
    let decoded = JoinSessionRequest::decode([0x1a, 0x00].as_slice())
        .expect("a keepalive's two bytes decode");
    assert_eq!(decoded, keepalive);
}

/// `JoinSessionHeader.client_keepalive` is field 3, a varint `bool`: tag
/// `0x18` and value 1; inside a request it is the header (field 1) of length 2.
#[test]
fn join_session_header_client_keepalive_is_field_3_on_the_wire() {
    use prost::Message;
    let header = JoinSessionHeader {
        session_id: String::new(),
        user_application: String::new(),
        client_keepalive: true,
    };
    assert_eq!(header.encode_to_vec(), vec![0x18, 0x01]);
    let request = JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Header(header)),
    };
    assert_eq!(request.encode_to_vec(), vec![0x0a, 0x02, 0x18, 0x01]);
}

/// `SessionConnected.keepalive_interval_ms` is field 2, a varint `uint64`:
/// tag `0x10`, then 15000 as the varint `98 75`.
#[test]
fn session_connected_keepalive_interval_ms_is_field_2_on_the_wire() {
    use prost::Message;
    let connected = SessionConnected {
        connection_id: String::new(),
        keepalive_interval_ms: 15_000,
    };
    assert_eq!(connected.encode_to_vec(), vec![0x10, 0x98, 0x75]);
    let decoded = SessionConnected::decode([0x10, 0x98, 0x75].as_slice())
        .expect("the interval's bytes decode");
    assert_eq!(decoded.keepalive_interval_ms, 15_000);
}

// ── Chained writes (b.pav): field numbers are client contract ────────────────
//
// `UpdateLayerRequest.chained_header` (oneof field 3), the three fields of
// `ChainedUpdateLayerHeader` and `PingResponse`'s two capability bits (fields
// 3 and 4) are additive to a published proto, agreed with the client on their
// numbers alone, so each is pinned by its wire bytes.

/// `UpdateLayerRequest.chained_header` is field 3 of the `message_type`
/// oneof: tag `0x1a` (field 3, length-delimited). An empty header encodes as
/// its tag and length 0, and decodes back to the arm.
#[test]
fn update_layer_chained_header_is_request_field_3_on_the_wire() {
    use prost::Message;
    let chained = UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::ChainedHeader(
            ChainedUpdateLayerHeader::default(),
        )),
    };
    assert_eq!(chained.encode_to_vec(), vec![0x1a, 0x00]);
    let decoded =
        UpdateLayerRequest::decode([0x1a, 0x00].as_slice()).expect("the arm's two bytes decode");
    assert_eq!(decoded, chained);
}

/// `ChainedUpdateLayerHeader` is `header` = 1 (an `UpdateLayerHeader`),
/// `previous_operation_id` = 2 (a string) and `no_wait` = 3 (a varint
/// `bool`).
#[test]
fn chained_update_layer_header_fields_are_1_2_and_3_on_the_wire() {
    use prost::Message;
    let header = ChainedUpdateLayerHeader {
        header: Some(UpdateLayerHeader {
            session_id: String::new(),
            layer_name: String::new(),
            object_id: 0,
            timestamp_microseconds_since_epoch: 0,
            client_id: String::new(),
            operation_id: "o".to_owned(),
        }),
        previous_operation_id: "p".to_owned(),
        no_wait: true,
    };
    // header: field 1, length 3, holding operation_id (field 6) "o";
    // previous_operation_id: field 2, "p"; no_wait: field 3, 1.
    assert_eq!(
        header.encode_to_vec(),
        vec![0x0a, 0x03, 0x32, 0x01, b'o', 0x12, 0x01, b'p', 0x18, 0x01]
    );
}

/// `PingResponse.operation_ids` is field 3 and `chained_writes` field 4,
/// both varint `bool`s: tags `0x18` and `0x20`.
#[test]
fn ping_response_capability_bits_are_fields_3_and_4_on_the_wire() {
    use prost::Message;
    let reply = PingResponse {
        client_timestamp_microseconds_since_epoch: 0,
        server_timestamp_microseconds_since_epoch: 0,
        operation_ids: true,
        chained_writes: true,
    };
    assert_eq!(reply.encode_to_vec(), vec![0x18, 0x01, 0x20, 0x01]);
}

/// A replica built before chained writes, holding the `UpdateLayerRequest`
/// oneof of `header` = 1 and `chunk` = 2 only, decodes a chained call's first
/// frame as a message with neither arm set: field 3 is unknown to it and
/// skipped. That is the fail-closed half of the contract: such a replica
/// refuses the frame `INVALID_ARGUMENT` "first message has no header or data"
/// with no `ErrorInfo`, before any chunk (pinned through the handler in
/// `integration::grpc::chained_writes_test`).
#[test]
fn pre_chaining_update_layer_request_decodes_a_chained_header_as_neither_arm() {
    use prost::Message;

    /// `UpdateLayerRequest` as a pre-chaining replica compiled it.
    #[derive(Clone, PartialEq, prost::Message)]
    struct PreChainingUpdateLayerRequest {
        #[prost(oneof = "PreChainingMessageType", tags = "1, 2")]
        message_type: Option<PreChainingMessageType>,
    }

    #[derive(Clone, PartialEq, prost::Oneof)]
    enum PreChainingMessageType {
        #[prost(message, tag = "1")]
        Header(UpdateLayerHeader),
        #[prost(message, tag = "2")]
        Chunk(UpdateLayerChunk),
    }

    let chained = UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::ChainedHeader(
            ChainedUpdateLayerHeader {
                header: Some(UpdateLayerHeader {
                    session_id: "sid".to_owned(),
                    layer_name: "layer".to_owned(),
                    object_id: 7,
                    timestamp_microseconds_since_epoch: 1,
                    client_id: "c".to_owned(),
                    operation_id: "op-2".to_owned(),
                }),
                previous_operation_id: "op-1".to_owned(),
                no_wait: true,
            },
        )),
    }
    .encode_to_vec();
    let as_old = PreChainingUpdateLayerRequest::decode(chained.as_slice())
        .expect("an unknown field never fails a decode");
    assert_eq!(
        as_old.message_type, None,
        "a pre-chaining replica must see neither a header nor a chunk"
    );
}

/// A server that does not implement keepalives sends no field 2, which a
/// client reads as interval 0: send no keepalives on that join.
#[test]
fn session_connected_without_the_interval_field_reads_as_zero() {
    use prost::Message;
    // `connection_id` "c" only: field 1, length 1.
    let decoded = SessionConnected::decode([0x0a, 0x01, b'c'].as_slice())
        .expect("a pre-keepalive SessionConnected decodes");
    assert_eq!(decoded.connection_id, "c");
    assert_eq!(decoded.keepalive_interval_ms, 0);
}
