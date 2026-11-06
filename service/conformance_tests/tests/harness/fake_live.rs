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

//! A scripted LiveService for the harness tests: a real gRPC server over
//! h2c on 127.0.0.1, written by hand on tonic's server pieces (the crate
//! generates client code only).
//!
//! It answers `CreateSession`, `DeleteSession`, `Ping`, `CreateLayer`,
//! `UpdateLayer`, `SubscribeReadLayer` and the `grpc.health.v1` `Check` as
//! the test's script says ([`Answer`]); every other method ends
//! `UNIMPLEMENTED`. Every request's method, `grpc-timeout` and
//! `authorization` headers are recorded ([`Received`]), as is each session a
//! `DeleteSession` names.
//!
//! tonic's server enforces a request's `grpc-timeout` on the wait for its
//! answer: an [`Answer::Hang`] still ends, `CANCELLED`, once the request's
//! deadline has passed on the server. The server's timer starts when the
//! request arrives, after the client's own, so the client has already ended
//! the call by then; and with a long `CONFORMANCE_TIMEOUT` a hang outlasts
//! any bound a test measures. A test of how the client itself ends a call no
//! server answers uses `support::SilentServer` or `support::silent_listener`
//! instead. An answered `SubscribeReadLayer` stream, which tonic does not
//! bound, stays open.

use std::collections::HashMap;
use std::convert::Infallible;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::StreamExt as _;
use live_edit_conformance::proto;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, BoxStream, Service};
use tonic::server::{ClientStreamingService, Grpc, ServerStreamingService, UnaryService};
use tonic::{Request, Response, Status, Streaming};
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::{HealthCheckRequest, HealthCheckResponse};
use tonic_prost::ProstCodec;

/// What the server does with one call.
#[derive(Clone, Debug)]
pub enum Answer {
    /// The method's success response (see [`FakeLive::start`]); `SERVING`
    /// for `Check`.
    Ok,
    /// `Check`: the response `NOT_SERVING`. Any other method: as [`Answer::Ok`].
    NotServing,
    /// End the call with this status.
    Fail(Status),
    /// Wait this long, then answer as the inner answer says.
    After(Duration, Box<Answer>),
    /// Never answer (see the module doc for how the call still ends).
    Hang,
}

impl Answer {
    /// The answer once every [`Answer::After`] delay has passed.
    fn last(&self) -> &Answer {
        match self {
            Answer::After(_, then) => then.last(),
            other => other,
        }
    }

    /// Waits out the answer's delays, then answers with `response` or fails
    /// the call.
    async fn deliver<T>(self, response: T) -> Result<Response<T>, Status> {
        let mut answer = self;
        loop {
            match answer {
                Answer::After(delay, then) => {
                    tokio::time::sleep(delay).await;
                    answer = *then;
                }
                Answer::Ok | Answer::NotServing => return Ok(Response::new(response)),
                Answer::Fail(status) => return Err(status),
                Answer::Hang => return futures::future::pending().await,
            }
        }
    }
}

/// One request the server received.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Received {
    /// The method name, e.g. `CreateSession`, or `Check` for a health check.
    pub method: String,
    /// The request's `grpc-timeout` header.
    pub grpc_timeout: Option<String>,
    /// The request's `authorization` header.
    pub authorization: Option<String>,
}

type Script = dyn Fn(&str, usize) -> Answer + Send + Sync;

struct Inner {
    script: Box<Script>,
    received: Mutex<Vec<Received>>,
    deleted: Mutex<Vec<String>>,
    calls: Mutex<HashMap<&'static str, usize>>,
}

impl Inner {
    /// The 1-based number of this call of `method`, and the script's answer
    /// to it.
    fn answer(&self, method: &'static str) -> (usize, Answer) {
        let mut calls = self.calls.lock().unwrap();
        let n = calls.entry(method).or_default();
        *n += 1;
        (*n, (self.script)(method, *n))
    }
}

/// A running scripted LiveService; stopped when dropped.
pub struct FakeLive {
    /// `http://127.0.0.1:<port>`.
    pub endpoint: String,
    inner: Arc<Inner>,
    server: tokio::task::JoinHandle<()>,
}

impl FakeLive {
    /// Starts the server. `script(method, n)` answers the `n`th call of
    /// `method`, counting from one, and is called as the call arrives. The
    /// success responses: `CreateSession` the session `session-<n>`;
    /// `CreateLayer` object 1; `UpdateLayer` sequence `n + 1`;
    /// `DeleteSession` and `Ping` empty; `SubscribeReadLayer` one empty
    /// header and then an open, silent stream; `Check` `SERVING` (or
    /// `NOT_SERVING`, [`Answer::NotServing`]). Uploads are read to their end
    /// before the answer.
    pub async fn start(script: impl Fn(&str, usize) -> Answer + Send + Sync + 'static) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let inner = Arc::new(Inner {
            script: Box::new(script),
            received: Mutex::new(Vec::new()),
            deleted: Mutex::new(Vec::new()),
            calls: Mutex::new(HashMap::new()),
        });
        let router = Router {
            inner: inner.clone(),
        };
        // The listener is bound before the server task starts, so a client
        // connecting at once waits in the backlog rather than failing.
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .serve_with_incoming(
                    router,
                    tonic::transport::server::TcpIncoming::from(listener),
                )
                .await
                .expect("the fake LiveService serves");
        });
        FakeLive {
            endpoint,
            inner,
            server,
        }
    }

    /// A server answering every scripted call with success.
    pub async fn ok() -> Self {
        Self::start(|_, _| Answer::Ok).await
    }

    /// Every request received, in order.
    pub fn received(&self) -> Vec<Received> {
        self.inner.received.lock().unwrap().clone()
    }

    /// The method names of [`FakeLive::received`].
    pub fn methods(&self) -> Vec<String> {
        self.received().into_iter().map(|r| r.method).collect()
    }

    /// The sessions `DeleteSession` calls named, in order.
    pub fn deleted(&self) -> Vec<String> {
        self.inner.deleted.lock().unwrap().clone()
    }
}

impl Drop for FakeLive {
    fn drop(&mut self) {
        self.server.abort();
    }
}

#[derive(Clone)]
struct Router {
    inner: Arc<Inner>,
}

fn header(request: &http::Request<Body>, name: &str) -> Option<String> {
    request
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

impl Service<http::Request<Body>> for Router {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let method = request
            .uri()
            .path()
            .rsplit('/')
            .next()
            .unwrap_or_default()
            .to_string();
        self.inner.received.lock().unwrap().push(Received {
            method: method.clone(),
            grpc_timeout: header(&request, "grpc-timeout"),
            authorization: header(&request, "authorization"),
        });
        let inner = self.inner.clone();
        Box::pin(async move {
            let response = match method.as_str() {
                "CreateSession" => {
                    Grpc::new(ProstCodec::default())
                        .unary(
                            Unary::new(
                                inner,
                                "CreateSession",
                                |_: &Inner, _: &proto::CreateSessionRequest, n, _| {
                                    proto::CreateSessionResponse {
                                        session_id: format!("session-{n}"),
                                    }
                                },
                            ),
                            request,
                        )
                        .await
                }
                "DeleteSession" => {
                    Grpc::new(ProstCodec::default())
                        .unary(
                            Unary::new(
                                inner,
                                "DeleteSession",
                                |inner: &Inner, message: &proto::DeleteSessionRequest, _, _| {
                                    inner
                                        .deleted
                                        .lock()
                                        .unwrap()
                                        .push(message.session_id.clone());
                                    proto::DeleteSessionResponse::default()
                                },
                            ),
                            request,
                        )
                        .await
                }
                "Ping" => {
                    Grpc::new(ProstCodec::default())
                        .unary(
                            Unary::new(inner, "Ping", |_: &Inner, _: &proto::PingRequest, _, _| {
                                proto::PingResponse::default()
                            }),
                            request,
                        )
                        .await
                }
                "Check" => {
                    Grpc::new(ProstCodec::default())
                        .unary(
                            Unary::new(
                                inner,
                                "Check",
                                |_: &Inner, _: &HealthCheckRequest, _, answer: &Answer| {
                                    let status = match answer {
                                        Answer::NotServing => ServingStatus::NotServing,
                                        _ => ServingStatus::Serving,
                                    };
                                    HealthCheckResponse {
                                        status: status as i32,
                                    }
                                },
                            ),
                            request,
                        )
                        .await
                }
                "CreateLayer" => {
                    Grpc::new(ProstCodec::default())
                        .client_streaming(
                            Upload::<proto::CreateLayerRequest, _>::new(
                                inner,
                                "CreateLayer",
                                |_| proto::CreateLayerResponse {
                                    object_id: 1,
                                    ..Default::default()
                                },
                            ),
                            request,
                        )
                        .await
                }
                "UpdateLayer" => {
                    Grpc::new(ProstCodec::default())
                        .client_streaming(
                            Upload::<proto::UpdateLayerRequest, _>::new(
                                inner,
                                "UpdateLayer",
                                |n| proto::UpdateLayerResponse {
                                    sequence_num: n as u64 + 1,
                                    ..Default::default()
                                },
                            ),
                            request,
                        )
                        .await
                }
                "SubscribeReadLayer" => {
                    Grpc::new(ProstCodec::default())
                        .server_streaming(Subscribe { inner }, request)
                        .await
                }
                _ => Status::unimplemented("not scripted").into_http(),
            };
            Ok(response)
        })
    }
}

/// The success response of a unary method to the `n`th call, given the
/// answer it will be sent with once any delay has passed; it may record the
/// request in the server's state.
type Respond<Req, Resp> = fn(&Inner, &Req, usize, &Answer) -> Resp;

/// A unary method.
struct Unary<Req, Resp> {
    inner: Arc<Inner>,
    method: &'static str,
    respond: Respond<Req, Resp>,
}

impl<Req, Resp> Unary<Req, Resp> {
    fn new(inner: Arc<Inner>, method: &'static str, respond: Respond<Req, Resp>) -> Self {
        Self {
            inner,
            method,
            respond,
        }
    }
}

impl<Req, Resp> UnaryService<Req> for Unary<Req, Resp>
where
    Req: Send + 'static,
    Resp: Send + 'static,
{
    type Response = Resp;
    type Future = BoxFuture<Response<Resp>, Status>;

    fn call(&mut self, request: Request<Req>) -> Self::Future {
        let message = request.into_inner();
        let (n, answer) = self.inner.answer(self.method);
        let response = (self.respond)(&self.inner, &message, n, answer.last());
        Box::pin(answer.deliver(response))
    }
}

/// A client-streaming method: reads the upload to its end, then answers.
struct Upload<Req, Resp> {
    inner: Arc<Inner>,
    method: &'static str,
    respond: fn(usize) -> Resp,
    request: PhantomData<fn(Req)>,
}

impl<Req, Resp> Upload<Req, Resp> {
    fn new(inner: Arc<Inner>, method: &'static str, respond: fn(usize) -> Resp) -> Self {
        Self {
            inner,
            method,
            respond,
            request: PhantomData,
        }
    }
}

impl<Req, Resp> ClientStreamingService<Req> for Upload<Req, Resp>
where
    Req: Send + 'static,
    Resp: Send + 'static,
{
    type Response = Resp;
    type Future = BoxFuture<Response<Resp>, Status>;

    fn call(&mut self, request: Request<Streaming<Req>>) -> Self::Future {
        let (n, answer) = self.inner.answer(self.method);
        let response = (self.respond)(n);
        Box::pin(async move {
            let mut upload = request.into_inner();
            while upload.message().await?.is_some() {}
            answer.deliver(response).await
        })
    }
}

/// `SubscribeReadLayer`: one empty header, then a stream that never sends
/// again and never ends.
struct Subscribe {
    inner: Arc<Inner>,
}

impl ServerStreamingService<proto::SubscribeReadLayerRequest> for Subscribe {
    type Response = proto::SubscribeReadLayerResponse;
    type ResponseStream = BoxStream<proto::SubscribeReadLayerResponse>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, _request: Request<proto::SubscribeReadLayerRequest>) -> Self::Future {
        let (_, answer) = self.inner.answer("SubscribeReadLayer");
        let header = proto::SubscribeReadLayerResponse {
            response_type: Some(proto::subscribe_read_layer_response::ResponseType::Header(
                proto::SubscribeReadLayerHeader::default(),
            )),
        };
        let messages = futures::stream::once(futures::future::ready(Ok(header)))
            .chain(futures::stream::pending());
        Box::pin(answer.deliver(Box::pin(messages) as Self::ResponseStream))
    }
}
