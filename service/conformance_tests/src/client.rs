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

//! The suite's LiveService client: typed, bounded RPC helpers and ownership of
//! everything a case opens.
//!
//! - **Channel.** One lazily connected channel per [`Client`]: `https`
//!   endpoints use TLS with [`Settings::ca_pem`] or else the system trust
//!   store; responses may be up to [`MAX_RESPONSE_BYTES`].
//! - **Deadlines.** Every unary and client-streaming call runs under
//!   [`with_deadline`] with [`Settings::timeout`]; `SubscribeReadLayer` and
//!   `JoinSession` under [`STREAM_DEADLINE`]. The deadline is sent as the
//!   `grpc-timeout` header *and* enforced on the client by the suite's own
//!   timer: tonic's channel enforces the header only until the response
//!   headers arrive, and a case such as the chained-write deadline refusal
//!   relies on the client ending the call itself. A call the client ends at
//!   its deadline fails `DEADLINE_EXCEEDED`, whichever timer fired: tonic's
//!   own expiry, which it reports as `CANCELLED` "Timeout expired", is
//!   reported as `DEADLINE_EXCEEDED` too.
//! - **Ownership.** The client records every session id a `CreateSession`
//!   response carries (even one a negative case did not expect to be
//!   accepted), and [`Client::close`] deletes them in reverse order,
//!   tolerating `NOT_FOUND`. Streams and pending calls the client started are
//!   cancelled at close too.
//! - **Errors.** A call that does not succeed returns [`CallError`]: the
//!   status the server (or the transport) ended it with, or a local failure
//!   that never reached the server. Cases match codes with
//!   `crate::check::expect_code!`.

use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use futures::{future, stream, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{AbortHandle, JoinError, JoinHandle};
use tokio::time::Instant;
use tonic::metadata::AsciiMetadataValue;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;

use crate::check::{CaseError, CaseResult};
use crate::config::{target, ConfigError, Settings, Target};
use crate::proto;
use crate::proto::live_service_client::LiveServiceClient;
use crate::{check, check_eq};

/// How long [`Client::close`] waits for the streams and calls it cancelled to
/// end.
const CANCEL_GRACE: Duration = Duration::from_secs(2);

/// The deadline of a `SubscribeReadLayer` or `JoinSession` call.
pub const STREAM_DEADLINE: Duration = Duration::from_secs(110);

/// The largest response message the client decodes, and the largest
/// `total_size` a subscribe delivery may declare: 128 MiB.
pub const MAX_RESPONSE_BYTES: usize = 128 * 1024 * 1024;

/// How many stream messages a [`Stream`]'s reader buffers before it stops
/// reading.
pub const STREAM_QUEUE_DEPTH: usize = 4096;

/// The chunk size of an upload unless a case sets another.
pub const DEFAULT_CHUNK_SIZE: usize = 65536;

/// The service name the readiness and `deployment` health checks ask about.
pub const LIVE_SERVICE_NAME: &str = "nvidia.omniverse.live.v1alpha.LiveService";

/// Why a call did not succeed.
#[derive(Debug)]
pub enum CallError {
    /// The call ended with this gRPC status: the server's answer, or the
    /// transport's (`UNAVAILABLE`), or the client's own deadline
    /// (`DEADLINE_EXCEEDED`, from [`with_deadline`] or [`STREAM_DEADLINE`]).
    Status(tonic::Status),
    /// A failure on the client that is no server answer: an unreadable token
    /// file, a stream that ended without a status, a timeout waiting for a
    /// stream message, a cancelled [`Pending`] call. Never matches an
    /// expected code.
    Local(String),
}

impl CallError {
    /// The status code, for [`CallError::Status`].
    pub fn code(&self) -> Option<tonic::Code> {
        match self {
            Self::Status(status) => Some(status.code()),
            Self::Local(_) => None,
        }
    }

    /// The status, for [`CallError::Status`].
    pub fn status(&self) -> Option<&tonic::Status> {
        match self {
            Self::Status(status) => Some(status),
            Self::Local(_) => None,
        }
    }
}

impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Status(status) => write!(
                f,
                "call ended with {:?}: {}",
                status.code(),
                status.message()
            ),
            Self::Local(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for CallError {}

impl From<tonic::Status> for CallError {
    fn from(status: tonic::Status) -> Self {
        Self::Status(status)
    }
}

/// A token that could not be read for a call fails that call locally.
impl From<ConfigError> for CallError {
    fn from(error: ConfigError) -> Self {
        Self::Local(error.to_string())
    }
}

/// The metadata a call carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Auth {
    /// `authorization: Bearer <token>` from [`Settings::authorization`].
    #[default]
    Bearer,
    /// No `authorization` header: the anonymous-request case.
    Anonymous,
}

/// Runs one call under a client-side deadline: the call's result, or
/// `CallError::Status(DEADLINE_EXCEEDED)` once `deadline` has passed, the
/// call then being dropped (cancelled). The request the call sends must carry
/// the same `deadline` in its `grpc-timeout` header (`tonic::Request::set_timeout`);
/// the client's helpers do both.
///
/// tonic's channel enforces that header too, with its own timer set to expire
/// at the same instant, and the call is polled before this timer, so an
/// unanswered call usually ends with tonic's `CANCELLED` "Timeout expired"
/// rather than this timer's expiry. That status is reported as the
/// `DEADLINE_EXCEEDED` it is (`own_deadline_as_exceeded`); any other
/// status, a server's `CANCELLED` included, is passed through.
pub async fn with_deadline<T, F>(deadline: Duration, call: F) -> Result<T, CallError>
where
    F: Future<Output = Result<tonic::Response<T>, tonic::Status>>,
{
    match tokio::time::timeout(deadline, call).await {
        Ok(Ok(response)) => Ok(response.into_inner()),
        Ok(Err(status)) => Err(CallError::Status(own_deadline_as_exceeded(status))),
        Err(_) => Err(CallError::Status(deadline_exceeded())),
    }
}

/// The status of a call the client ended at its deadline.
fn deadline_exceeded() -> tonic::Status {
    tonic::Status::deadline_exceeded("Deadline Exceeded")
}

/// `status`, or `DEADLINE_EXCEEDED` ([`deadline_exceeded`]) when it is
/// tonic's report that the call's own `grpc-timeout` expired: a `CANCELLED`
/// whose source chain holds `tonic::TimeoutExpired`.
///
/// That attribution is unambiguous. Only tonic's client-side `grpc-timeout`
/// layer creates a `TimeoutExpired`, from the header the client set (the
/// channel has no `Endpoint::timeout` of its own, `channel_to`), and a status
/// the server sent arrives decoded from the response, with no source at all.
fn own_deadline_as_exceeded(status: tonic::Status) -> tonic::Status {
    let expired_here = status.code() == tonic::Code::Cancelled
        && std::iter::successors(std::error::Error::source(&status), |error| error.source())
            .any(|error| error.is::<tonic::TimeoutExpired>());
    if expired_here {
        deadline_exceeded()
    } else {
        status
    }
}

/// Options of a `CreateLayer` upload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CreateOptions {
    /// Default `layer`.
    pub layer: String,
    /// Bytes per chunk message, default [`DEFAULT_CHUNK_SIZE`].
    pub chunk_size: usize,
    /// Default empty (none).
    pub operation_id: String,
}

impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            layer: "layer".into(),
            chunk_size: DEFAULT_CHUNK_SIZE,
            operation_id: String::new(),
        }
    }
}

/// Options of an unchained `UpdateLayer` upload (chunks of
/// [`DEFAULT_CHUNK_SIZE`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpdateOptions {
    /// Default `layer`.
    pub layer: String,
    /// Default `writer`.
    pub client_id: String,
    /// The header's `timestamp_microseconds_since_epoch`, default 123456.
    pub timestamp: u64,
    /// Default empty (none).
    pub operation_id: String,
}

impl Default for UpdateOptions {
    fn default() -> Self {
        Self {
            layer: "layer".into(),
            client_id: "writer".into(),
            timestamp: 123_456,
            operation_id: String::new(),
        }
    }
}

/// Options of a chained `UpdateLayer` (its `chained_header`; chunks of
/// [`DEFAULT_CHUNK_SIZE`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainOptions {
    /// Default `layer`.
    pub layer: String,
    /// Default `writer`.
    pub client_id: String,
    /// The header's `timestamp_microseconds_since_epoch`, default 123456.
    pub timestamp: u64,
    /// The chained header's `no_wait`, default false.
    pub no_wait: bool,
    /// The call's deadline; default `None`, [`Settings::timeout`].
    pub timeout: Option<Duration>,
}

impl Default for ChainOptions {
    fn default() -> Self {
        Self {
            layer: "layer".into(),
            client_id: "writer".into(),
            timestamp: 123_456,
            no_wait: false,
            timeout: None,
        }
    }
}

/// Options of a `SubscribeReadLayer`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubscribeOptions {
    /// Default `layer`.
    pub layer: String,
    /// Default `reader`.
    pub client_id: String,
}

impl Default for SubscribeOptions {
    fn default() -> Self {
        Self {
            layer: "layer".into(),
            client_id: "reader".into(),
        }
    }
}

/// A LiveService client for one endpoint, owning what its calls create (see
/// the module doc). Every method takes `&self`, so a case can hold streams and
/// pending calls while it makes more calls. Not closed on drop: the runner
/// calls [`Client::close`].
pub struct Client {
    settings: Arc<Settings>,
    channel: Channel,
    /// Session ids from `CreateSession` responses, in creation order.
    sessions: Mutex<Vec<String>>,
    /// The tasks behind every stream reader and pending call this client
    /// started, cancelled at close.
    tasks: Mutex<Vec<AbortHandle>>,
}

/// Locks `mutex`, recovering the data from a panic in another holder (the
/// lists it guards stay valid whatever happened).
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The host of a `host:port` address, without IPv6 brackets: the name the
/// server's certificate is checked against.
fn host_of(address: &str) -> &str {
    let host = address.rsplit_once(':').map_or(address, |(host, _)| host);
    host.strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host)
}

/// The lazily connected channel to `target`: h2c for `http`; for `https`,
/// TLS trusting the CA file when one is set and the system store otherwise.
/// It sets no `Endpoint::timeout`: the only client-side `grpc-timeout` is the
/// one each request carries, which [`own_deadline_as_exceeded`] relies on.
fn channel_to(settings: &Settings, target: &Target) -> Result<Channel, ConfigError> {
    let scheme = if target.tls { "https" } else { "http" };
    let mut endpoint = Endpoint::from_shared(format!("{scheme}://{}", target.address))
        .map_err(|_| ConfigError::Invalid("endpoint is not a valid URI".to_string()))?;
    if target.tls {
        let tls = ClientTlsConfig::new().domain_name(host_of(&target.address));
        let tls = match settings.ca_pem()? {
            Some(pem) => tls.ca_certificate(Certificate::from_pem(pem)),
            None => tls.with_native_roots(),
        };
        endpoint = endpoint.tls_config(tls).map_err(|error| {
            let cause = std::error::Error::source(&error)
                .map(|source| format!(": {source}"))
                .unwrap_or_default();
            ConfigError::Invalid(format!("TLS configuration failed: {error}{cause}"))
        })?;
    }
    Ok(endpoint.connect_lazy())
}

impl Client {
    /// A client for `endpoint` (default [`Settings::endpoint`]). Connects
    /// lazily: no I/O happens here. Fails only on configuration: an invalid
    /// endpoint or an unreadable CA file. Must be called inside the runtime.
    pub fn new(settings: Arc<Settings>, endpoint: Option<&str>) -> Result<Self, ConfigError> {
        let target = target(endpoint.unwrap_or(&settings.endpoint))?;
        let channel = channel_to(&settings, &target)?;
        Ok(Client {
            settings,
            channel,
            sessions: Mutex::new(Vec::new()),
            tasks: Mutex::new(Vec::new()),
        })
    }

    /// The settings this client was made with.
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// The session ids this client owns, in creation order.
    pub fn owned_sessions(&self) -> Vec<String> {
        lock(&self.sessions).clone()
    }

    /// A LiveService stub on the channel (cheap: the channel is shared).
    fn live(&self) -> LiveServiceClient<Channel> {
        LiveServiceClient::new(self.channel.clone()).max_decoding_message_size(MAX_RESPONSE_BYTES)
    }

    /// `message` as a request carrying `deadline` in its `grpc-timeout`
    /// header and, for [`Auth::Bearer`], the token as read now.
    fn request<M>(
        &self,
        message: M,
        auth: Auth,
        deadline: Duration,
    ) -> Result<tonic::Request<M>, CallError> {
        let mut request = tonic::Request::new(message);
        request.set_timeout(deadline);
        if auth == Auth::Bearer {
            let value: AsciiMetadataValue =
                self.settings.authorization()?.parse().map_err(|_| {
                    CallError::Local("the bearer token is not a valid header value".to_string())
                })?;
            request.metadata_mut().insert("authorization", value);
        }
        Ok(request)
    }

    /// Records the session a `CreateSession` response carries as owned.
    fn own_session(&self, response: &proto::CreateSessionResponse) {
        if !response.session_id.is_empty() {
            lock(&self.sessions).push(response.session_id.clone());
        }
    }

    /// Records a task to cancel at close.
    fn track(&self, task: AbortHandle) {
        lock(&self.tasks).push(task);
    }

    /// Runs `call` as a task the client cancels at close.
    fn spawn_call<T, F>(&self, call: F) -> Pending<T>
    where
        T: Send + 'static,
        F: Future<Output = Result<T, CallError>> + Send + 'static,
    {
        let handle = tokio::spawn(call);
        self.track(handle.abort_handle());
        Pending::new(handle)
    }

    /// Starts an `UpdateLayer` sending `requests` (a stream that may never
    /// end) under `deadline`.
    fn spawn_update<S>(
        &self,
        requests: S,
        deadline: Duration,
    ) -> Pending<proto::UpdateLayerResponse>
    where
        S: futures::Stream<Item = proto::UpdateLayerRequest> + Send + 'static,
    {
        match self.request(requests, Auth::Bearer, deadline) {
            Ok(request) => {
                let mut live = self.live();
                self.spawn_call(
                    async move { with_deadline(deadline, live.update_layer(request)).await },
                )
            }
            Err(error) => self.spawn_call(async move { Err(error) }),
        }
    }

    // --- Unary calls. Each runs under `Settings::timeout` with `Auth::Bearer`.

    /// `Ping`.
    pub async fn ping(
        &self,
        request: proto::PingRequest,
    ) -> Result<proto::PingResponse, CallError> {
        self.ping_within(request, self.settings.timeout).await
    }

    /// `Ping` under `deadline` instead of [`Settings::timeout`]: the readiness
    /// fallback.
    pub async fn ping_within(
        &self,
        request: proto::PingRequest,
        deadline: Duration,
    ) -> Result<proto::PingResponse, CallError> {
        let request = self.request(request, Auth::Bearer, deadline)?;
        let mut live = self.live();
        with_deadline(deadline, live.ping(request)).await
    }

    /// `CreateSession`; the returned session is owned by the client.
    pub async fn create_session(
        &self,
        request: proto::CreateSessionRequest,
    ) -> Result<proto::CreateSessionResponse, CallError> {
        self.create_session_as(request, Auth::Bearer).await
    }

    /// `CreateSession` with `auth`; a session in the response is owned by the
    /// client even when the call was expected to be refused.
    pub async fn create_session_as(
        &self,
        request: proto::CreateSessionRequest,
        auth: Auth,
    ) -> Result<proto::CreateSessionResponse, CallError> {
        let deadline = self.settings.timeout;
        let request = self.request(request, auth, deadline)?;
        let mut live = self.live();
        let response = with_deadline(deadline, live.create_session(request)).await?;
        self.own_session(&response);
        Ok(response)
    }

    /// `DeleteSession`.
    pub async fn delete_session(
        &self,
        request: proto::DeleteSessionRequest,
    ) -> Result<proto::DeleteSessionResponse, CallError> {
        let deadline = self.settings.timeout;
        let request = self.request(request, Auth::Bearer, deadline)?;
        let mut live = self.live();
        with_deadline(deadline, live.delete_session(request)).await
    }

    /// `ListSessions`.
    pub async fn list_sessions(
        &self,
        request: proto::ListSessionsRequest,
    ) -> Result<proto::ListSessionsResponse, CallError> {
        let deadline = self.settings.timeout;
        let request = self.request(request, Auth::Bearer, deadline)?;
        let mut live = self.live();
        with_deadline(deadline, live.list_sessions(request)).await
    }

    /// `StatLayer`.
    pub async fn stat_layer(
        &self,
        request: proto::StatLayerRequest,
    ) -> Result<proto::StatLayerResponse, CallError> {
        let deadline = self.settings.timeout;
        let request = self.request(request, Auth::Bearer, deadline)?;
        let mut live = self.live();
        with_deadline(deadline, live.stat_layer(request)).await
    }

    /// `DeleteLayer`.
    pub async fn delete_layer(
        &self,
        request: proto::DeleteLayerRequest,
    ) -> Result<proto::DeleteLayerResponse, CallError> {
        let deadline = self.settings.timeout;
        let request = self.request(request, Auth::Bearer, deadline)?;
        let mut live = self.live();
        with_deadline(deadline, live.delete_layer(request)).await
    }

    /// The `grpc.health.v1` check of `service`, without credentials, under
    /// `deadline`.
    pub async fn health(
        &self,
        service: &str,
        deadline: Duration,
    ) -> Result<tonic_health::pb::HealthCheckResponse, CallError> {
        let request = self.request(
            HealthCheckRequest {
                service: service.to_string(),
            },
            Auth::Anonymous,
            deadline,
        )?;
        let mut health =
            HealthClient::new(self.channel.clone()).max_decoding_message_size(MAX_RESPONSE_BYTES);
        with_deadline(deadline, health.check(request)).await
    }

    // --- Sessions.

    /// A new session on [`Settings::base_path`] named `conformance-<uuid>`.
    /// A response without a session id is a [`CallError::Local`].
    pub async fn session(&self) -> Result<String, CallError> {
        let name = format!("conformance-{}", uuid::Uuid::new_v4().simple());
        self.session_named(&name).await
    }

    /// A new session on [`Settings::base_path`] named `name`.
    pub async fn session_named(&self, name: &str) -> Result<String, CallError> {
        let response = self
            .create_session(proto::CreateSessionRequest {
                path: self.settings.base_path.clone(),
                name: name.to_string(),
            })
            .await?;
        if response.session_id.is_empty() {
            return Err(CallError::Local(
                "CreateSession returned no session id".to_string(),
            ));
        }
        Ok(response.session_id)
    }

    // --- Uploads.

    /// `CreateLayer` of `payload` with the default [`CreateOptions`].
    pub async fn create(
        &self,
        session: &str,
        payload: &[u8],
    ) -> Result<proto::CreateLayerResponse, CallError> {
        self.create_with(session, payload, &CreateOptions::default())
            .await
    }

    /// `CreateLayer` of `payload`: the header, then `payload` in
    /// `options.chunk_size` chunks (none for an empty payload).
    pub async fn create_with(
        &self,
        session: &str,
        payload: &[u8],
        options: &CreateOptions,
    ) -> Result<proto::CreateLayerResponse, CallError> {
        use proto::create_layer_request::MessageType;
        if options.chunk_size == 0 {
            return Err(CallError::Local("chunk_size must be positive".to_string()));
        }
        let header = proto::CreateLayerRequest {
            message_type: Some(MessageType::Header(proto::CreateLayerHeader {
                session_id: session.to_string(),
                layer_name: options.layer.clone(),
                operation_id: options.operation_id.clone(),
            })),
        };
        let chunks = payload
            .chunks(options.chunk_size)
            .map(|chunk| proto::CreateLayerRequest {
                message_type: Some(MessageType::Chunk(proto::CreateLayerChunk {
                    data: chunk.to_vec(),
                })),
            });
        self.create_layer_raw(std::iter::once(header).chain(chunks).collect())
            .await
    }

    /// `CreateLayer` sending exactly `requests`, for malformed sequences.
    pub async fn create_layer_raw(
        &self,
        requests: Vec<proto::CreateLayerRequest>,
    ) -> Result<proto::CreateLayerResponse, CallError> {
        let deadline = self.settings.timeout;
        let request = self.request(stream::iter(requests), Auth::Bearer, deadline)?;
        let mut live = self.live();
        with_deadline(deadline, live.create_layer(request)).await
    }

    /// `UpdateLayer` of `payload` with the default [`UpdateOptions`].
    pub async fn update(
        &self,
        session: &str,
        object_id: u64,
        payload: &[u8],
    ) -> Result<proto::UpdateLayerResponse, CallError> {
        self.update_with(session, object_id, payload, &UpdateOptions::default())
            .await
    }

    /// `UpdateLayer` of `payload`: the header, then `payload` in chunks.
    pub async fn update_with(
        &self,
        session: &str,
        object_id: u64,
        payload: &[u8],
        options: &UpdateOptions,
    ) -> Result<proto::UpdateLayerResponse, CallError> {
        let header = proto::UpdateLayerRequest {
            message_type: Some(proto::update_layer_request::MessageType::Header(
                proto::UpdateLayerHeader {
                    session_id: session.to_string(),
                    layer_name: options.layer.clone(),
                    object_id,
                    timestamp_microseconds_since_epoch: options.timestamp,
                    client_id: options.client_id.clone(),
                    operation_id: options.operation_id.clone(),
                },
            )),
        };
        self.update_layer_raw(
            std::iter::once(header)
                .chain(update_chunks(payload))
                .collect(),
        )
        .await
    }

    /// `UpdateLayer` sending exactly `requests`, for malformed sequences.
    pub async fn update_layer_raw(
        &self,
        requests: Vec<proto::UpdateLayerRequest>,
    ) -> Result<proto::UpdateLayerResponse, CallError> {
        let deadline = self.settings.timeout;
        let request = self.request(stream::iter(requests), Auth::Bearer, deadline)?;
        let mut live = self.live();
        with_deadline(deadline, live.update_layer(request)).await
    }

    // --- Chained writes.

    /// The messages of a chained `UpdateLayer`: the `chained_header` naming
    /// `previous_operation_id`, then `payload` in chunks. Sends nothing.
    /// (`options.timeout` does not apply here.)
    pub fn chained_requests(
        &self,
        session: &str,
        object_id: u64,
        payload: &[u8],
        operation_id: &str,
        previous_operation_id: &str,
        options: &ChainOptions,
    ) -> Vec<proto::UpdateLayerRequest> {
        let header = proto::UpdateLayerRequest {
            message_type: Some(proto::update_layer_request::MessageType::ChainedHeader(
                proto::ChainedUpdateLayerHeader {
                    header: Some(proto::UpdateLayerHeader {
                        session_id: session.to_string(),
                        layer_name: options.layer.clone(),
                        object_id,
                        timestamp_microseconds_since_epoch: options.timestamp,
                        client_id: options.client_id.clone(),
                        operation_id: operation_id.to_string(),
                    }),
                    previous_operation_id: previous_operation_id.to_string(),
                    no_wait: options.no_wait,
                },
            )),
        };
        std::iter::once(header)
            .chain(update_chunks(payload))
            .collect()
    }

    /// A chained `UpdateLayer`, waited for, with the default [`ChainOptions`].
    pub async fn chained_update(
        &self,
        session: &str,
        object_id: u64,
        payload: &[u8],
        operation_id: &str,
        previous_operation_id: &str,
    ) -> Result<proto::UpdateLayerResponse, CallError> {
        self.chained_update_with(
            session,
            object_id,
            payload,
            operation_id,
            previous_operation_id,
            &ChainOptions::default(),
        )
        .await
    }

    /// A chained `UpdateLayer`, waited for, under `options.timeout` or
    /// [`Settings::timeout`].
    pub async fn chained_update_with(
        &self,
        session: &str,
        object_id: u64,
        payload: &[u8],
        operation_id: &str,
        previous_operation_id: &str,
        options: &ChainOptions,
    ) -> Result<proto::UpdateLayerResponse, CallError> {
        let requests = self.chained_requests(
            session,
            object_id,
            payload,
            operation_id,
            previous_operation_id,
            options,
        );
        let deadline = options.timeout.unwrap_or(self.settings.timeout);
        let request = self.request(stream::iter(requests), Auth::Bearer, deadline)?;
        let mut live = self.live();
        with_deadline(deadline, live.update_layer(request)).await
    }

    /// Starts a chained `UpdateLayer` without waiting for it, with the default
    /// [`ChainOptions`]: see [`Pending`].
    pub async fn chained_update_async(
        &self,
        session: &str,
        object_id: u64,
        payload: &[u8],
        operation_id: &str,
        previous_operation_id: &str,
    ) -> Pending<proto::UpdateLayerResponse> {
        self.chained_update_async_with(
            session,
            object_id,
            payload,
            operation_id,
            previous_operation_id,
            &ChainOptions::default(),
        )
        .await
    }

    /// Starts a chained `UpdateLayer` without waiting for it, under
    /// `options.timeout` or [`Settings::timeout`]. A failure to start it (an
    /// unreadable token) is the [`Pending`]'s result.
    pub async fn chained_update_async_with(
        &self,
        session: &str,
        object_id: u64,
        payload: &[u8],
        operation_id: &str,
        previous_operation_id: &str,
        options: &ChainOptions,
    ) -> Pending<proto::UpdateLayerResponse> {
        let requests = self.chained_requests(
            session,
            object_id,
            payload,
            operation_id,
            previous_operation_id,
            options,
        );
        let deadline = options.timeout.unwrap_or(self.settings.timeout);
        self.spawn_update(stream::iter(requests), deadline)
    }

    /// Starts an `UpdateLayer` that sends `requests` and then never ends its
    /// request stream, under [`Settings::timeout`]. Cancelling the returned
    /// [`Pending`] resets the call (`RST_STREAM`) before any end of stream
    /// the server could commit on, as a lost connection would; the client
    /// cancels it at close otherwise.
    pub async fn unended_update_async(
        &self,
        requests: Vec<proto::UpdateLayerRequest>,
    ) -> Pending<proto::UpdateLayerResponse> {
        // `pending()` never yields and never ends, so no END_STREAM is ever
        // sent: dropping the call (its task) resets the HTTP/2 stream.
        let requests = stream::iter(requests).chain(stream::pending());
        self.spawn_update(requests, self.settings.timeout)
    }

    // --- Streams.

    /// `SubscribeReadLayer` from (`object_id`, `sequence`) with the default
    /// [`SubscribeOptions`]. See [`Client::subscribe_with`].
    pub async fn subscribe(
        &self,
        session: &str,
        object_id: u64,
        sequence: u64,
    ) -> Result<Stream<proto::SubscribeReadLayerResponse>, CallError> {
        self.subscribe_with(session, object_id, sequence, &SubscribeOptions::default())
            .await
    }

    /// `SubscribeReadLayer` under [`STREAM_DEADLINE`]. Returns once the call
    /// is started; the server's answer, including a refusal such as
    /// `NOT_FOUND`, arrives through [`Stream::next`]. `Err` only for a local
    /// failure to start it.
    pub async fn subscribe_with(
        &self,
        session: &str,
        object_id: u64,
        sequence: u64,
        options: &SubscribeOptions,
    ) -> Result<Stream<proto::SubscribeReadLayerResponse>, CallError> {
        let request = self.request(
            proto::SubscribeReadLayerRequest {
                session_id: session.to_string(),
                layer_name: options.layer.clone(),
                object_id,
                sequence_num: sequence,
                client_id: options.client_id.clone(),
            },
            Auth::Bearer,
            STREAM_DEADLINE,
        )?;
        let mut live = self.live();
        let source = call_stream(
            async move { live.subscribe_read_layer(request).await },
            STREAM_DEADLINE,
            (),
        );
        let stream = Stream::spawn(source, self.settings.timeout);
        self.track(stream.reader.clone());
        Ok(stream)
    }

    /// `JoinSession` under [`STREAM_DEADLINE`], with `user_application`
    /// `conformance` and the header's `client_keepalive` set to `keepalive`.
    /// Returns the response stream and the sender of the request stream.
    /// Nothing but the header is sent unless the case sends it, so a case
    /// controls when a join is silent. As for [`Client::subscribe_with`], the
    /// server's answer arrives through the stream.
    pub async fn join(
        &self,
        session: &str,
        keepalive: bool,
    ) -> Result<(Stream<proto::JoinSessionResponse>, JoinSender), CallError> {
        let header = proto::JoinSessionRequest {
            request_type: Some(proto::join_session_request::RequestType::Header(
                proto::JoinSessionHeader {
                    session_id: session.to_string(),
                    user_application: "conformance".to_string(),
                    client_keepalive: keepalive,
                },
            )),
        };
        let (sender, outgoing) = mpsc::unbounded_channel();
        let (closed_tx, closed) = oneshot::channel::<()>();
        let requests = stream::once(future::ready(header)).chain(join_requests(outgoing, closed));
        let request = self.request(requests, Auth::Bearer, STREAM_DEADLINE)?;
        let mut live = self.live();
        // The response side holds `closed_tx`: when it is dropped (the stream
        // closed, cancelled or ended), the request stream ends too.
        let source = call_stream(
            async move { live.join_session(request).await },
            STREAM_DEADLINE,
            closed_tx,
        );
        let stream = Stream::spawn(source, self.settings.timeout);
        self.track(stream.reader.clone());
        Ok((stream, JoinSender { sender }))
    }

    /// Cancels every stream and pending call this client started, then
    /// deletes its owned sessions in reverse creation order (`NOT_FOUND`
    /// tolerated). Any other failure is collected and returned as one error
    /// after every step has been tried: the runner reports it as the case's
    /// `<error>`.
    pub async fn close(self) -> CaseResult<()> {
        let mut failures = Vec::new();

        let tasks = std::mem::take(&mut *lock(&self.tasks));
        for task in &tasks {
            task.abort();
        }
        let grace_end = Instant::now() + CANCEL_GRACE;
        while tasks.iter().any(|task| !task.is_finished()) {
            if Instant::now() >= grace_end {
                failures.push(
                    "Stream reader or pending call failed to terminate after cancellation"
                        .to_string(),
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        for session in self.owned_sessions().iter().rev() {
            let deleted = self
                .delete_session(proto::DeleteSessionRequest {
                    session_id: session.clone(),
                })
                .await;
            match deleted {
                Ok(_) => {}
                Err(CallError::Status(status)) if status.code() == tonic::Code::NotFound => {}
                Err(CallError::Status(status)) => {
                    failures.push(format!("DeleteSession({session}): {:?}", status.code()))
                }
                Err(CallError::Local(message)) => {
                    failures.push(format!("DeleteSession({session}): {message}"))
                }
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(CaseError::new(format!(
                "Resource cleanup failed: {}",
                failures.join("; ")
            )))
        }
    }
}

/// The chunk messages of an `UpdateLayer` upload of `payload`
/// ([`DEFAULT_CHUNK_SIZE`] bytes each; none for an empty payload).
fn update_chunks(payload: &[u8]) -> impl Iterator<Item = proto::UpdateLayerRequest> + '_ {
    payload
        .chunks(DEFAULT_CHUNK_SIZE)
        .map(|chunk| proto::UpdateLayerRequest {
            message_type: Some(proto::update_layer_request::MessageType::Chunk(
                proto::UpdateLayerChunk {
                    data: chunk.to_vec(),
                },
            )),
        })
}

/// A join's request stream after its header: what the case sends through its
/// [`JoinSender`], until `closed` resolves (its sender, held by the response
/// side, is dropped). A dropped [`JoinSender`] does not end it: the join stays
/// open, silent, as a case that ignores its sender expects.
fn join_requests(
    outgoing: mpsc::UnboundedReceiver<proto::JoinSessionRequest>,
    closed: oneshot::Receiver<()>,
) -> impl futures::Stream<Item = proto::JoinSessionRequest> + Send + 'static {
    stream::unfold(
        (outgoing, closed),
        |(mut outgoing, mut closed)| async move {
            let next = tokio::select! {
                biased;
                _ = &mut closed => None,
                message = outgoing.recv() => Some(message),
            };
            match next {
                None => None,
                Some(Some(message)) => Some((message, (outgoing, closed))),
                Some(None) => {
                    let _ = closed.await;
                    None
                }
            }
        },
    )
}

/// The response stream of a server-streaming or bidirectional call started by
/// `call`: the call's start (its refusal, if any, as the only item), then its
/// messages, ended with `DEADLINE_EXCEEDED` once `deadline` has passed since
/// now, the call then being dropped (cancelled). `guard` is held for as long
/// as the call is, and dropped with it.
///
/// Before the response headers arrive, tonic's own `grpc-timeout` timer can
/// end the call first, as `CANCELLED` "Timeout expired": that ending is
/// reported as `DEADLINE_EXCEEDED` too ([`own_deadline_as_exceeded`]). After
/// the headers only this stream's deadline applies.
fn call_stream<T, F, G>(
    call: F,
    deadline: Duration,
    guard: G,
) -> impl futures::Stream<Item = Result<T, tonic::Status>> + Send + 'static
where
    T: Send + 'static,
    F: Future<Output = Result<tonic::Response<tonic::Streaming<T>>, tonic::Status>>
        + Send
        + 'static,
    G: Send + 'static,
{
    let end = Instant::now() + deadline;
    let body = stream::once(call)
        .map(|started| match started {
            Ok(response) => response.into_inner().left_stream(),
            Err(status) => stream::iter([Err(status)]).right_stream(),
        })
        .flatten();
    stream::unfold(Some((Box::pin(body), guard)), move |state| async move {
        let (mut body, guard) = state?;
        match tokio::time::timeout_at(end, body.next()).await {
            Ok(Some(item)) => Some((item.map_err(own_deadline_as_exceeded), Some((body, guard)))),
            Ok(None) => None,
            Err(_) => Some((Err(deadline_exceeded()), None)),
        }
    })
}

/// The response side of a server-streaming or bidirectional call: a reader
/// task drains the call into a queue of [`STREAM_QUEUE_DEPTH`] messages, and
/// the case takes them with [`Stream::next`]. The call's terminal status, or
/// its end without one, is the last item. Dropping or closing the stream
/// cancels the call: the reader task, and with it the call, is dropped, which
/// resets the HTTP/2 stream (a join's request stream ends first; see
/// [`Client::join`]).
pub struct Stream<T> {
    items: mpsc::Receiver<Result<T, CallError>>,
    timeout: Duration,
    reader: AbortHandle,
}

impl<T> Drop for Stream<T> {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

impl<T: Send + 'static> Stream<T> {
    /// A stream reading `source` (a call's `tonic::Streaming`, or a test's
    /// channel), whose [`Stream::next`] waits up to `timeout`. Must be called
    /// inside the runtime.
    pub fn spawn<S>(source: S, timeout: Duration) -> Self
    where
        S: futures::Stream<Item = Result<T, tonic::Status>> + Send + 'static,
    {
        let (queue, items) = mpsc::channel(STREAM_QUEUE_DEPTH);
        let reader = tokio::spawn(async move {
            let mut source = std::pin::pin!(source);
            loop {
                let item = match source.next().await {
                    Some(Ok(message)) => Ok(message),
                    Some(Err(status)) => Err(CallError::Status(status)),
                    None => Err(CallError::Local("Stream ended unexpectedly".to_string())),
                };
                let terminal = item.is_err();
                // A full queue stops the reading until the case takes more.
                if queue.send(item).await.is_err() || terminal {
                    return;
                }
            }
        })
        .abort_handle();
        Stream {
            items,
            timeout,
            reader,
        }
    }

    /// The next message, waiting up to the stream's timeout. `Err` with the
    /// call's status when it ended with one; [`CallError::Local`] when it
    /// ended without a status or nothing arrived in time.
    pub async fn next(&mut self) -> Result<T, CallError> {
        let deadline = Instant::now() + self.timeout;
        self.next_until(deadline).await
    }

    /// As [`Stream::next`], waiting until `deadline` instead. A message
    /// already queued is returned even when `deadline` has passed.
    pub async fn next_until(&mut self, deadline: tokio::time::Instant) -> Result<T, CallError> {
        match tokio::time::timeout_at(deadline, self.items.recv()).await {
            Ok(Some(item)) => item,
            Ok(None) => Err(CallError::Local(
                "The stream already ended: no further message can arrive".to_string(),
            )),
            Err(_) => Err(CallError::Local(
                "Timed out waiting for a required stream message".to_string(),
            )),
        }
    }

    /// Cancels the call now. (A join's peers then see it leave.)
    pub fn close(self) {
        self.reader.abort();
    }
}

impl Stream<proto::SubscribeReadLayerResponse> {
    /// One delivery: a header, then chunks until exactly its `total_size`
    /// bytes have arrived, all within one stream timeout. Fails unless the
    /// header comes first, every chunk is non-empty, no header interrupts the
    /// chunks, the chunks do not overrun `total_size`, and `total_size` is at
    /// most [`MAX_RESPONSE_BYTES`].
    pub async fn delivery(&mut self) -> CaseResult<(proto::SubscribeReadLayerHeader, Vec<u8>)> {
        use proto::subscribe_read_layer_response::ResponseType;
        let deadline = Instant::now() + self.timeout;
        let first = self.next_until(deadline).await?;
        let Some(ResponseType::Header(header)) = first.response_type else {
            return Err(CaseError::new("Expected header before chunks"));
        };
        check!(
            header.total_size <= MAX_RESPONSE_BYTES as u64,
            "Unbounded response size"
        );
        let mut data = Vec::new();
        while (data.len() as u64) < header.total_size {
            let message = self.next_until(deadline).await?;
            let Some(ResponseType::Chunk(chunk)) = message.response_type else {
                return Err(CaseError::new("Header interrupted content"));
            };
            check!(!chunk.data.is_empty(), "Empty content chunk");
            data.extend_from_slice(&chunk.data);
            check!(
                data.len() as u64 <= header.total_size,
                "Chunks exceed declared total_size"
            );
        }
        Ok((header, data))
    }

    /// The caught-up acknowledgement: a delivery with no content whose header
    /// names (`object_id`, `sequence`) and carries zero client and server
    /// timestamps. A commit's header is not one.
    pub async fn caught_up(&mut self, object_id: u64, sequence: u64) -> CaseResult<()> {
        let (header, data) = self.delivery().await?;
        check!(
            data.is_empty() && header.total_size == 0,
            "A caught-up acknowledgement carries no content"
        );
        check_eq!(
            (header.object_id, header.sequence_num),
            (object_id, sequence)
        );
        check_eq!(header.client_timestamp_microseconds_since_epoch, 0);
        check_eq!(header.server_timestamp_microseconds_since_epoch, 0);
        Ok(())
    }
}

/// The request side of a `JoinSession` call. Dropping it does not end the
/// call's request stream: only closing the [`Stream`] (or the client) does.
pub struct JoinSender {
    sender: mpsc::UnboundedSender<proto::JoinSessionRequest>,
}

impl JoinSender {
    /// Sends a `SessionMessageRequest` carrying `data`.
    pub fn message(&self, data: &[u8]) -> CaseResult<()> {
        self.send(proto::join_session_request::RequestType::Message(
            proto::SessionMessageRequest {
                data: data.to_vec(),
            },
        ))
    }

    /// Sends a `JoinSessionKeepalive`.
    pub fn keepalive(&self) -> CaseResult<()> {
        self.send(proto::join_session_request::RequestType::Keepalive(
            proto::JoinSessionKeepalive {},
        ))
    }

    /// Queues one request; fails once the join's request stream has ended.
    fn send(&self, request: proto::join_session_request::RequestType) -> CaseResult<()> {
        self.sender
            .send(proto::JoinSessionRequest {
                request_type: Some(request),
            })
            .map_err(|_| CaseError::new("The join's request stream has already ended"))
    }
}

/// A call started and not yet waited for, like a gRPC future. The call runs
/// as its own task, so it proceeds whether or not anyone waits for it.
pub struct Pending<T> {
    state: tokio::sync::Mutex<PendingState<T>>,
    task: AbortHandle,
}

/// A [`Pending`] call's task, or its outcome once [`Pending::cancel`] has
/// collected it.
enum PendingState<T> {
    Running(JoinHandle<Result<T, CallError>>),
    Ended(Result<T, CallError>),
}

/// A call task's outcome as the call's result.
fn call_outcome<T>(joined: Result<Result<T, CallError>, JoinError>) -> Result<T, CallError> {
    match joined {
        Ok(result) => result,
        Err(error) if error.is_cancelled() => {
            Err(CallError::Local("The call was cancelled".to_string()))
        }
        Err(_) => Err(CallError::Local("The call's task panicked".to_string())),
    }
}

impl<T: Send + 'static> Pending<T> {
    fn new(handle: JoinHandle<Result<T, CallError>>) -> Self {
        Pending {
            task: handle.abort_handle(),
            state: tokio::sync::Mutex::new(PendingState::Running(handle)),
        }
    }

    /// Whether the call has ended (answered, refused, failed or cancelled).
    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }

    /// Cancels the call if it has not ended: `true` when it was still running
    /// and is now cancelled, `false` when it had already ended. Returns once
    /// the call has been dropped, so its reset is queued on the connection
    /// before the case's next call.
    pub async fn cancel(&self) -> bool {
        let mut state = self.state.lock().await;
        let PendingState::Running(handle) = &mut *state else {
            return false;
        };
        if handle.is_finished() {
            return false;
        }
        handle.abort();
        // A task's future is dropped before its handle reports it cancelled.
        let joined = handle.await;
        let cancelled = matches!(&joined, Err(error) if error.is_cancelled());
        *state = PendingState::Ended(call_outcome(joined));
        cancelled
    }

    /// Waits for the call's result. A cancelled call's result is a
    /// [`CallError::Local`].
    pub async fn result(self) -> Result<T, CallError> {
        match self.state.into_inner() {
            PendingState::Running(handle) => call_outcome(handle.await),
            PendingState::Ended(result) => result,
        }
    }
}

// Cases hold `&Client` and `&Pending` across awaits in `Send` futures, and
// move streams and senders into them.
const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    send_sync::<Client>();
    send_sync::<Pending<proto::UpdateLayerResponse>>();
    send::<Stream<proto::SubscribeReadLayerResponse>>();
    send::<Stream<proto::JoinSessionResponse>>();
    send::<JoinSender>();
};

#[cfg(test)]
mod tests {
    use futures::channel::mpsc as channel;
    use proto::subscribe_read_layer_response::ResponseType;

    use super::*;

    type Item = Result<proto::SubscribeReadLayerResponse, tonic::Status>;

    /// The stream timeout of the tests: what a missing message waits for.
    const TIMEOUT: Duration = Duration::from_secs(5);

    fn header(total_size: u64) -> proto::SubscribeReadLayerResponse {
        header_of(proto::SubscribeReadLayerHeader {
            total_size,
            ..Default::default()
        })
    }

    fn header_of(header: proto::SubscribeReadLayerHeader) -> proto::SubscribeReadLayerResponse {
        proto::SubscribeReadLayerResponse {
            response_type: Some(ResponseType::Header(header)),
        }
    }

    fn chunk(data: &[u8]) -> proto::SubscribeReadLayerResponse {
        proto::SubscribeReadLayerResponse {
            response_type: Some(ResponseType::Chunk(proto::SubscribeReadLayerChunk {
                data: data.to_vec(),
            })),
        }
    }

    /// A [`Stream`] fed `messages` through a channel that then stays open, as
    /// a call that sends nothing more (the sender is returned to keep it
    /// open).
    fn open_stream(
        messages: Vec<proto::SubscribeReadLayerResponse>,
    ) -> (
        Stream<proto::SubscribeReadLayerResponse>,
        channel::UnboundedSender<Item>,
    ) {
        let (sender, receiver) = channel::unbounded();
        for message in messages {
            sender.unbounded_send(Ok(message)).unwrap();
        }
        (Stream::spawn(receiver, TIMEOUT), sender)
    }

    /// The failure message of a delivery read from `messages`.
    async fn broken_delivery(messages: Vec<proto::SubscribeReadLayerResponse>) -> String {
        let (mut stream, _open) = open_stream(messages);
        match stream.delivery().await {
            Err(error) => error.message().to_string(),
            Ok(delivery) => panic!("a broken stream passed as {delivery:?}"),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_well_formed_header_and_chunks_is_returned_whole() {
        let (mut stream, _open) = open_stream(vec![header(4), chunk(b"ab"), chunk(b"cd")]);
        let (header, data) = stream.delivery().await.expect("a well-formed delivery");
        assert_eq!(header.total_size, 4);
        assert_eq!(data, b"abcd");
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_chunk_before_any_header_fails() {
        let message = broken_delivery(vec![chunk(b"bad")]).await;
        assert!(
            message.contains("Expected header before chunks"),
            "{message}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_chunks_past_the_declared_total_size_fail() {
        let message = broken_delivery(vec![header(2), chunk(b"long")]).await;
        assert!(
            message.contains("Chunks exceed declared total_size"),
            "{message}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_header_interrupting_the_content_fails() {
        let message = broken_delivery(vec![header(1), header(0)]).await;
        assert!(message.contains("Header interrupted content"), "{message}");
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_from_a_stream_that_sends_nothing_fails_at_the_timeout() {
        let started = tokio::time::Instant::now();
        let message = broken_delivery(Vec::new()).await;
        assert!(
            message.contains("Timed out waiting for a required stream message"),
            "{message}"
        );
        assert!(started.elapsed() >= TIMEOUT);
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_empty_content_chunk_fails() {
        let message = broken_delivery(vec![header(1), chunk(b"")]).await;
        assert!(message.contains("Empty content chunk"), "{message}");
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_total_size_over_the_response_bound_fails() {
        let message = broken_delivery(vec![header(MAX_RESPONSE_BYTES as u64 + 1)]).await;
        assert!(message.contains("Unbounded response size"), "{message}");
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_missing_chunks_one_deadline_covers_the_whole_delivery() {
        // Each chunk arrives in time on its own, but the delivery as a whole
        // does not: one stream timeout bounds it.
        let (sender, receiver) = channel::unbounded::<Item>();
        let mut stream = Stream::spawn(receiver, TIMEOUT);
        sender.unbounded_send(Ok(header(3))).unwrap();
        // One chunk every half timeout: the third would arrive at 1.5
        // timeouts. The sender is handed back so the stream never ends.
        let feeder = tokio::spawn(async move {
            for _ in 0..3 {
                tokio::time::sleep(TIMEOUT / 2).await;
                if sender.unbounded_send(Ok(chunk(b"x"))).is_err() {
                    break;
                }
            }
            sender
        });
        let started = tokio::time::Instant::now();
        let error = stream.delivery().await.unwrap_err();
        assert!(
            error.message().contains("Timed out waiting"),
            "{}",
            error.message()
        );
        assert!(started.elapsed() < TIMEOUT + TIMEOUT / 2);
        let _open = feeder.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_stream_ending_without_a_status_fails() {
        let (mut stream, open) = open_stream(vec![header(3), chunk(b"ab")]);
        drop(open);
        let message = stream.delivery().await.unwrap_err().message().to_string();
        assert!(message.contains("Stream ended unexpectedly"), "{message}");
    }

    #[tokio::test(start_paused = true)]
    async fn caught_up_acknowledgement_with_zero_timestamps_passes() {
        let ack = header_of(proto::SubscribeReadLayerHeader {
            object_id: 1,
            sequence_num: 2,
            ..Default::default()
        });
        let (mut stream, _open) = open_stream(vec![ack]);
        assert_eq!(stream.caught_up(1, 2).await, Ok(()));
    }

    #[tokio::test(start_paused = true)]
    async fn caught_up_commit_header_is_not_an_acknowledgement() {
        let commit = header_of(proto::SubscribeReadLayerHeader {
            object_id: 1,
            sequence_num: 2,
            server_timestamp_microseconds_since_epoch: 42,
            ..Default::default()
        });
        let (mut stream, _open) = open_stream(vec![commit]);
        assert!(stream.caught_up(1, 2).await.is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn caught_up_wrong_position_client_timestamp_or_content_is_refused() {
        let refused = [
            // Another position.
            proto::SubscribeReadLayerHeader {
                object_id: 1,
                sequence_num: 3,
                ..Default::default()
            },
            // A client timestamp: a commit, not an acknowledgement.
            proto::SubscribeReadLayerHeader {
                object_id: 1,
                sequence_num: 2,
                client_timestamp_microseconds_since_epoch: 7,
                ..Default::default()
            },
        ];
        for header in refused {
            let (mut stream, _open) = open_stream(vec![header_of(header)]);
            assert!(stream.caught_up(1, 2).await.is_err(), "{header:?}");
        }
        let (mut stream, _open) = open_stream(vec![
            header_of(proto::SubscribeReadLayerHeader {
                object_id: 1,
                sequence_num: 2,
                total_size: 1,
                ..Default::default()
            }),
            chunk(b"x"),
        ]);
        assert!(stream.caught_up(1, 2).await.is_err(), "content");
    }

    #[tokio::test(start_paused = true)]
    async fn next_returns_the_calls_status_then_reports_the_stream_ended() {
        let (sender, receiver) = channel::unbounded::<Item>();
        sender.unbounded_send(Ok(header(0))).unwrap();
        sender
            .unbounded_send(Err(tonic::Status::not_found("session deleted")))
            .unwrap();
        let mut stream = Stream::spawn(receiver, TIMEOUT);

        assert!(stream.next().await.is_ok());
        assert_eq!(
            stream.next().await.unwrap_err().code(),
            Some(tonic::Code::NotFound)
        );
        // The status ended the stream: nothing further can arrive.
        let ended = stream.next().await.unwrap_err();
        assert!(
            matches!(&ended, CallError::Local(message) if message.contains("already ended")),
            "{ended:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn with_deadline_unanswered_call_is_deadline_exceeded_at_the_deadline() {
        let deadline = Duration::from_secs(3);
        let started = tokio::time::Instant::now();
        let result = with_deadline(
            deadline,
            future::pending::<Result<tonic::Response<()>, tonic::Status>>(),
        )
        .await;
        assert_eq!(
            result.unwrap_err().code(),
            Some(tonic::Code::DeadlineExceeded)
        );
        // Paused time: the timer fires exactly at the deadline, give or take
        // the timer wheel's millisecond.
        let elapsed = started.elapsed();
        assert!(
            elapsed >= deadline && elapsed <= deadline + Duration::from_millis(1),
            "{elapsed:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn with_deadline_answer_before_the_deadline_is_returned() {
        let answer = async {
            tokio::time::sleep(Duration::from_secs(2)).await;
            Ok(tonic::Response::new(7u32))
        };
        assert_eq!(
            with_deadline(Duration::from_secs(3), answer).await.unwrap(),
            7
        );
    }

    #[tokio::test(start_paused = true)]
    async fn with_deadline_server_status_is_passed_through() {
        let refused = async {
            Err::<tonic::Response<()>, _>(tonic::Status::aborted("predecessor not committed"))
        };
        assert_eq!(
            with_deadline(Duration::from_secs(3), refused)
                .await
                .unwrap_err()
                .code(),
            Some(tonic::Code::Aborted)
        );
    }

    /// A status tonic builds from its `grpc-timeout` timer's
    /// `tonic::TimeoutExpired`: `CANCELLED` "Timeout expired", the expiry
    /// as its source.
    fn tonic_timeout_expired() -> tonic::Status {
        let status = tonic::Status::from_error(Box::new(tonic::TimeoutExpired(())));
        assert_eq!(status.code(), tonic::Code::Cancelled, "{status:?}");
        assert_eq!(status.message(), "Timeout expired");
        status
    }

    /// A failure whose source is `tonic::TimeoutExpired`. A call through
    /// tonic's channel ends this way: its status's source is a
    /// `tonic::transport::Error` wrapping the expiry, one level down.
    #[derive(Debug)]
    struct Wrapped(tonic::TimeoutExpired);

    impl fmt::Display for Wrapped {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("wrapped")
        }
    }

    impl std::error::Error for Wrapped {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }

    /// Statuses the client's own deadline produced: tonic's report of its
    /// `grpc-timeout` expiry, with `TimeoutExpired` as the source or, as the
    /// channel reports it, deeper in the source chain.
    fn own_deadline_statuses() -> [tonic::Status; 2] {
        let wrapped = tonic::Status::from_error(Box::new(Wrapped(tonic::TimeoutExpired(()))));
        assert_eq!(wrapped.code(), tonic::Code::Cancelled, "{wrapped:?}");
        [tonic_timeout_expired(), wrapped]
    }

    /// Statuses that are not the client's own deadline, each with its code:
    /// `CANCELLED` as a server sends it (decoded from the response, so with
    /// no source), even worded as tonic words its timer's; `CANCELLED` with
    /// another source; and another code with `TimeoutExpired` as its source.
    fn other_statuses() -> Vec<(tonic::Status, tonic::Code)> {
        let mut reset = tonic::Status::cancelled("stream reset");
        reset.set_source(Arc::new(std::io::Error::other("connection reset")));
        let mut unavailable = tonic::Status::unavailable("no connection");
        unavailable.set_source(Arc::new(tonic::TimeoutExpired(())));
        vec![
            (
                tonic::Status::cancelled("Timeout expired"),
                tonic::Code::Cancelled,
            ),
            (reset, tonic::Code::Cancelled),
            (unavailable, tonic::Code::Unavailable),
        ]
    }

    #[tokio::test(start_paused = true)]
    async fn with_deadline_tonics_own_timeout_expired_is_deadline_exceeded() {
        for status in own_deadline_statuses() {
            let shown = format!("{status:?}");
            let result = with_deadline(
                Duration::from_secs(3),
                future::ready(Err::<tonic::Response<()>, _>(status)),
            )
            .await;
            assert_eq!(
                result.unwrap_err().code(),
                Some(tonic::Code::DeadlineExceeded),
                "{shown}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn with_deadline_cancelled_without_tonics_timeout_expired_is_passed_through() {
        for (status, code) in other_statuses() {
            let shown = format!("{status:?}");
            let result = with_deadline(
                Duration::from_secs(3),
                future::ready(Err::<tonic::Response<()>, _>(status)),
            )
            .await;
            assert_eq!(result.unwrap_err().code(), Some(code), "{shown}");
        }
    }

    /// The first item of a stream whose call ended at its start with
    /// `status`.
    async fn stream_start_ended_with(status: tonic::Status) -> CallError {
        let call = future::ready(Err::<
            tonic::Response<tonic::Streaming<proto::SubscribeReadLayerResponse>>,
            _,
        >(status));
        let mut stream = Stream::spawn(call_stream(call, STREAM_DEADLINE, ()), TIMEOUT);
        stream.next().await.unwrap_err()
    }

    #[tokio::test(start_paused = true)]
    async fn call_stream_tonics_own_timeout_expired_before_the_headers_is_deadline_exceeded() {
        for status in own_deadline_statuses() {
            let shown = format!("{status:?}");
            let ended = stream_start_ended_with(status).await;
            assert_eq!(ended.code(), Some(tonic::Code::DeadlineExceeded), "{shown}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn call_stream_cancelled_without_tonics_timeout_expired_is_passed_through() {
        for (status, code) in other_statuses() {
            let shown = format!("{status:?}");
            let ended = stream_start_ended_with(status).await;
            assert_eq!(ended.code(), Some(code), "{shown}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn call_stream_unanswered_call_ends_deadline_exceeded_at_the_stream_deadline() {
        let call = future::pending::<
            Result<
                tonic::Response<tonic::Streaming<proto::SubscribeReadLayerResponse>>,
                tonic::Status,
            >,
        >();
        let started = tokio::time::Instant::now();
        let mut stream = Stream::spawn(call_stream(call, STREAM_DEADLINE, ()), STREAM_DEADLINE * 2);

        // Nothing before the deadline...
        let early = stream
            .next_until(started + STREAM_DEADLINE - Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(matches!(early, CallError::Local(_)), "{early:?}");
        // ...then DEADLINE_EXCEEDED at it.
        let ended = stream.next().await.unwrap_err();
        assert_eq!(ended.code(), Some(tonic::Code::DeadlineExceeded));
        let elapsed = started.elapsed();
        assert!(
            elapsed >= STREAM_DEADLINE && elapsed <= STREAM_DEADLINE + Duration::from_millis(1),
            "{elapsed:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn call_stream_refused_call_is_its_only_item() {
        let call = future::ready(Err::<
            tonic::Response<tonic::Streaming<proto::SubscribeReadLayerResponse>>,
            _,
        >(tonic::Status::not_found("no such layer")));
        let mut stream = Stream::spawn(call_stream(call, STREAM_DEADLINE, ()), TIMEOUT);
        assert_eq!(
            stream.next().await.unwrap_err().code(),
            Some(tonic::Code::NotFound)
        );
        assert!(matches!(stream.next().await, Err(CallError::Local(_))));
    }

    #[test]
    fn host_of_strips_the_port_and_ipv6_brackets() {
        assert_eq!(host_of("service.example.com:443"), "service.example.com");
        assert_eq!(host_of("[::1]:443"), "::1");
    }
}
