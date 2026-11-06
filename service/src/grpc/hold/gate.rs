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

//! The hold gate: the `Service` wrapper in front of every LiveService RPC that
//! refuses them all while this replica is held (SDD SR-6.1), or while its hold
//! verdict is stale (SDD SR-6.5).

use std::convert::Infallible;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::future::{Either, MapOk, Ready};
use futures::TryFutureExt;
use tokio::sync::watch;
use tonic::server::NamedService;
use tonic::Status;
use tower::Service;

use crate::dr::hold::HoldObservation;
use crate::error::RefusalDetail;
use crate::grpc::rpc_log::{log_rpc_failure, RpcLogContext};

/// Whether the gate lets LiveService RPCs through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldGateState {
    /// Every RPC passes to the auth interceptor and its handler.
    Open,
    /// The fleet is held: every RPC is refused `UNAVAILABLE` `SERVICE_HELD`
    /// with the incident of the latest verdict. Never reopened: a held
    /// replica exits when the hold clears (SDD SR-6.4).
    Held,
    /// This replica's hold verdict went stale (SDD SR-6.5): it cannot tell
    /// whether the fleet is held, so every RPC is refused `UNAVAILABLE`
    /// `UPSTREAM_UNAVAILABLE`, never `SERVICE_HELD`, because no fleet hold is
    /// known and there is no incident. Never reopened: the replica exits on
    /// its next successful hold read, or non-zero once the read has failed
    /// for long enough.
    Stale,
}

const OPEN: u8 = 0;
const HELD: u8 = 1;
const STALE: u8 = 2;

struct GateShared {
    state: AtomicU8,
    /// The hold poll's verdict channel, read for the incident each refusal
    /// names. `None` for [`HoldGate::always_open`].
    verdicts: Option<watch::Receiver<HoldObservation>>,
}

/// The gate in front of the LiveService (SDD SR-6.1, SR-6.5). Cloning shares
/// it.
///
/// [`live_service_server`](crate::grpc::transport::live_service_server)
/// wraps the auth-intercepted service in it ([`HoldGateService`]), so a
/// refused call never reaches the auth interceptor or a handler, and health,
/// reflection and the CapabilitiesService, which are not behind it, keep
/// answering. The hold entry routine ([`enter_hold`](super::enter_hold))
/// closes it for a hold, and the stale-verdict entry routine
/// ([`enter_stale_verdict`](super::enter_stale_verdict)) for a stale verdict.
/// Held mode's refuse-all stub
/// ([`held_live_service`](super::held_live_service)) is one closed for a hold
/// before it serves.
///
/// A hold's refusal names the incident of the hold poll's **latest** verdict,
/// read at the refusal, never a copy taken when the gate closed: an incident
/// can be replaced while held. When the latest verdict names none — the
/// marker has no incident, the hold has just cleared and the replica is about
/// to exit, or the gate has no verdict channel — the refusal says `incident
/// unknown`. A stale verdict's refusal names no incident.
#[derive(Clone)]
pub struct HoldGate {
    shared: Arc<GateShared>,
}

impl std::fmt::Debug for HoldGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoldGate")
            .field("state", &self.state())
            .finish_non_exhaustive()
    }
}

impl HoldGate {
    /// An open gate whose refusals read their incident from `verdicts`, the
    /// hold poll's channel (`dr::hold::spawn_hold_poll`).
    pub fn new(verdicts: watch::Receiver<HoldObservation>) -> Self {
        Self::with_verdicts(Some(verdicts))
    }

    /// An open gate with no verdict channel, for a server that runs no hold
    /// poll: the test servers. Nothing closes it unless the caller does;
    /// closed, its refusals say `incident unknown`.
    pub fn always_open() -> Self {
        Self::with_verdicts(None)
    }

    fn with_verdicts(verdicts: Option<watch::Receiver<HoldObservation>>) -> Self {
        Self {
            shared: Arc::new(GateShared {
                state: AtomicU8::new(OPEN),
                verdicts,
            }),
        }
    }

    /// The gate's state now.
    pub fn state(&self) -> HoldGateState {
        match self.shared.state.load(Ordering::Acquire) {
            OPEN => HoldGateState::Open,
            HELD => HoldGateState::Held,
            _ => HoldGateState::Stale,
        }
    }

    /// Close the gate for a fleet hold: from now on every LiveService RPC is
    /// refused `SERVICE_HELD`. Idempotent; never undone. A hold replaces a
    /// stale verdict's closing: the fleet hold is the more specific answer.
    pub fn close_held(&self) {
        self.shared.state.store(HELD, Ordering::Release);
    }

    /// Close the gate for a stale verdict (SDD SR-6.5): from now on every
    /// LiveService RPC is refused `UPSTREAM_UNAVAILABLE`. Only an open gate
    /// changes: a gate already closed for a hold keeps refusing
    /// `SERVICE_HELD`, because a held replica whose verdict goes stale stays
    /// held. Idempotent; never undone.
    pub fn close_stale(&self) {
        let _ =
            self.shared
                .state
                .compare_exchange(OPEN, STALE, Ordering::AcqRel, Ordering::Acquire);
    }

    /// The status a call is refused with now, or `None` while the gate is
    /// open.
    pub fn refusal(&self) -> Option<Status> {
        self.refusal_and_state().map(|(status, _)| status)
    }

    /// [`Self::refusal`] with the closed state it was built for.
    fn refusal_and_state(&self) -> Option<(Status, HoldGateState)> {
        match self.state() {
            HoldGateState::Open => None,
            HoldGateState::Held => {
                let incident =
                    self.shared.verdicts.as_ref().and_then(|verdicts| {
                        verdicts.borrow().verdict.incident().map(str::to_owned)
                    });
                Some((
                    RefusalDetail::service_held_status(incident.as_deref()),
                    HoldGateState::Held,
                ))
            }
            HoldGateState::Stale => Some((
                crate::grpc::transport::hold_verdict_stale(),
                HoldGateState::Stale,
            )),
        }
    }

    /// Wrap `inner` behind this gate.
    pub fn wrap<S>(&self, inner: S) -> HoldGateService<S> {
        HoldGateService {
            inner,
            gate: self.clone(),
        }
    }
}

/// The response every [`HoldGateService`] answers with.
pub type HoldGateResponse = http::Response<tonic::body::Body>;

/// The future of a [`HoldGateService`] call over an inner service answering
/// `http::Response<B>`: the inner service's future, its body boxed into a
/// [`tonic::body::Body`], for a call the gate lets through; or the refusal,
/// ready at once. Its own type because a refusal never calls the inner
/// service, so it has no inner future to return.
pub type HoldGateFuture<F, B> = Either<
    MapOk<F, fn(http::Response<B>) -> HoldGateResponse>,
    Ready<Result<HoldGateResponse, Infallible>>,
>;

/// Box an inner response's body into a [`tonic::body::Body`]: the interceptor's
/// response body cannot be built outside tonic, so the gate's responses, the
/// refusal's included, share tonic's body type instead.
fn into_tonic_body<B>(response: http::Response<B>) -> HoldGateResponse
where
    B: http_body::Body<Data = bytes::Bytes> + Send + 'static,
    B::Error: Into<tonic::codegen::StdError>,
{
    response.map(tonic::body::Body::new)
}

/// A service behind a [`HoldGate`]: built by [`HoldGate::wrap`], inside
/// [`live_service_server`](crate::grpc::transport::live_service_server).
///
/// While the gate is open every call goes to `inner` unchanged, and only its
/// response body is boxed into a [`tonic::body::Body`] ([`HoldGateResponse`]).
/// While it is closed every call is answered with the gate's refusal as a
/// trailers-only
/// response, and `inner` — the auth interceptor and the handler — is never
/// called. Each refusal is logged once through the completion recipe
/// (`grpc::rpc_log`): a hold's at INFO (marked hold-caused), a stale
/// verdict's at WARN like any other dependency outage. The line marks the
/// request's completion ledger reported, so the metrics layer's backstop logs
/// nothing more for it. Takes
/// the inner service's [`NamedService::NAME`], so it routes exactly as the
/// service it wraps.
#[derive(Clone)]
pub struct HoldGateService<S> {
    inner: S,
    gate: HoldGate,
}

impl<S> std::fmt::Debug for HoldGateService<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoldGateService")
            .field("gate", &self.gate)
            .finish_non_exhaustive()
    }
}

impl<S, B> Service<http::Request<tonic::body::Body>> for HoldGateService<S>
where
    S: Service<http::Request<tonic::body::Body>, Response = http::Response<B>, Error = Infallible>,
    B: http_body::Body<Data = bytes::Bytes> + Send + 'static,
    B::Error: Into<tonic::codegen::StdError>,
{
    type Response = HoldGateResponse;
    type Error = Infallible;
    type Future = HoldGateFuture<S::Future, B>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: http::Request<tonic::body::Body>) -> Self::Future {
        let Some((status, state)) = self.gate.refusal_and_state() else {
            return Either::Left(
                self.inner
                    .call(request)
                    .map_ok(into_tonic_body::<B> as fn(http::Response<B>) -> HoldGateResponse),
            );
        };
        let mut ctx = RpcLogContext::for_http_request(&request);
        if state == HoldGateState::Held {
            ctx.hold_caused();
        }
        log_rpc_failure(&ctx, &status);
        Either::Right(futures::future::ready(Ok(status.into_http())))
    }
}

impl<S> NamedService for HoldGateService<S>
where
    S: NamedService,
{
    const NAME: &'static str = S::NAME;
}
