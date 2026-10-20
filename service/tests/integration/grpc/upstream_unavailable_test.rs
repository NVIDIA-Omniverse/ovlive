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

//! b.xd1 — the `UPSTREAM_UNAVAILABLE` hint (SDD §18.1), through the real
//! `CreateLayer` / `UpdateLayer` handlers.
//!
//! A plain `UNAVAILABLE` an upload is refused with before its commit marker
//! — a dependency the call needed was down, and no step before the marker
//! writes client-visible state — is answered `UNAVAILABLE` carrying
//! `UPSTREAM_UNAVAILABLE` in the service's domain, no metadata, and a
//! `RetryInfo` delay in `[1 s, 3 s)`: the client's "this call applied
//! nothing" signal. Every `UNAVAILABLE` from the marker on carries no detail,
//! because the call's commit may have been sent.
//!
//! Store faults are staged with [`ScriptReplyProxy`]: the fixture's Valkey
//! client — and so the service built on it — connects through the proxy, and
//! the proxy answers the one command the armed matcher names with the
//! store's own reply (`-LOADING`, `-BUSY`, `-OOM`, `-TRYAGAIN`, or a plain
//! `-ERR` for `CreateLayer`'s check after a failed commit run), so fred's
//! reply parsing and the service's classification run as in production.
//! Each proxy serves only its own test, and each test arms its matcher
//! around one call with nothing else in flight, then asserts the matcher was
//! hit (the vacuity guard). Auth outages are staged with hand-written auth
//! doubles; the call-start read, `CreateLayer`'s commit runs and the
//! overwrite's wake-up, and the commit loop use the layer store's
//! `test-fault-injection` seams, so those tests sit in the
//! [`fault_injected`] module.
//!
//! | test | site | answer |
//! |---|---|---|
//! | `update_layer_base_path_read_refused_loading_*` | `UpdateLayer` session base-path `HGET` | hinted |
//! | `update_layer_permission_check_timed_out_*` | `UpdateLayer` permission check | hinted |
//! | `update_layer_mapping_read_refused_loading_*` (unchained, chained) | `UpdateLayer` layer-mapping `GET` | hinted |
//! | `update_layer_ext_pre_store_refused_busy_*` | `UpdateLayer` external-value pre-store | hinted |
//! | `update_layer_ext_pre_store_refused_tryagain_*` | `UpdateLayer` external-value pre-store, a multi-key script refused `-TRYAGAIN` during a slot migration | hinted |
//! | `fault_injected::update_layer_call_start_read_refused_loading_*` | `UpdateLayer` call-start record read | hinted |
//! | `create_layer_base_path_read_refused_loading_*` | `CreateLayer` session base-path `HGET` | hinted |
//! | `create_layer_operation_record_read_refused_loading_*` | `CreateLayer` operation-record `GET` | hinted |
//! | `create_layer_userinfo_unreachable_*` | `CreateLayer` userinfo | hinted |
//! | `create_layer_ext_pre_store_refused_busy_*` | `CreateLayer` external-value pre-store | hinted |
//! | `create_layer_object_id_allocation_refused_loading_*` | `CreateLayer` object-id allocation | plain |
//! | `fault_injected::create_layer_overwrite_whose_commit_runs_failed_answers_plain_unavailable` | `CreateLayer` overwrite commit, both runs failed with a transient (`IO`) error with the old object fenced (the replacement can still complete after the answer) | plain, carrying the commit's error |
//! | `fault_injected::create_layer_{overwrite, new_layer}_whose_commit_runs_failed_permanently_*` | `CreateLayer` commit, both runs failed with a permanent-class error (one the mapper alone answers `INTERNAL`) and the check still naming the expected value | plain, carrying the commit's error |
//! | `fault_injected::create_layer_{overwrite, new_layer}_whose_permanent_commit_failure_check_failed_too_*` | `CreateLayer` commit, a run failed with a permanent-class error and the check after it failed too | plain, carrying the commit's error |
//! | `fault_injected::create_layer_overwrite_wake_up_failed_*` | `CreateLayer` overwrite wake-up, after the commit point | OK — not a refusal |
//! | `create_layer_commit_refused_out_of_memory_*` | `CreateLayer` new-layer commit script refused `-OOM` | plain |
//! | `fault_injected::update_layer_loading_past_the_retry_budget_*` | `UpdateLayer` commit loop exhausted by `-LOADING` | plain |
//! | `fault_injected::update_layer_tryagain_on_the_watch_past_the_retry_budget_*` | `UpdateLayer` commit loop (`WATCH` route) exhausted by `-TRYAGAIN` | plain |
//! | `fault_injected::update_layer_tryagain_on_the_commit_script_past_the_retry_budget_*` | `UpdateLayer` commit loop (`lua` route) exhausted by `-TRYAGAIN` | plain |
//! | `fault_injected::update_layer_recovery_reissue_pre_store_refused_busy_*` | `UpdateLayer` recovery re-issue's pre-store, after a two-attempt first call | plain |
//!
//! Every hinted refusal also asserts the call applied nothing: on
//! `UpdateLayer` the layer's `seq` and stream length are unchanged, on
//! `CreateLayer` no mapping was created and the object-id counter did not
//! move. The `CreateLayer` commit give-up rows (`IO` and permanent-class,
//! budget spent and check failed) share one scenario,
//! `fault_injected::commit_failure_gave_up`, parameterised on the injected
//! fault and on where the call gives up; each also asserts no run landed.

use std::future::Future;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};
use tonic::{Code, Status};

use live_edit::auth::permissions::PermissionsClient;
use live_edit::auth::userinfo::UserinfoClient;
use live_edit::auth::{AuthError, AuthResult, Permission, UserInfo};
use live_edit::domain::types::ObjectState;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    create_layer_request, update_layer_request, ChainedUpdateLayerHeader, CreateLayerHeader,
    CreateLayerRequest, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::storage::client::{STORE_NOT_SERVING_SHAPES, TRYAGAIN_REPLY};
use live_edit::storage::external_values::{
    ext_content_hash, MockExternalValueStore, ValkeyExternalValueStore, EXT_STORE_VALUE_SCRIPT,
};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::CREATE_LAYER_COMMIT_SCRIPT;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::{
    build_service, build_service_with_userinfo, make_create_data, make_request, make_update_data,
    streaming_create, streaming_update, AllowAllPermissions,
};
use crate::common::owned_delta::{
    delta_to_ompf_bytes, delta_with_set_field_external, delta_with_set_field_inline,
};
use crate::common::refusal_detail::{assert_no_refusal_detail, assert_upstream_unavailable};
use crate::storage::snapshot_read_support::{
    error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// Replies and constants
// ═════════════════════════════════════════════════════════════════════════════

/// The store's reply while it replays its dataset after a restart.
const LOADING_REPLY: &str = STORE_NOT_SERVING_SHAPES[0];
/// The store's reply while a long script runs.
const BUSY_REPLY: &str = STORE_NOT_SERVING_SHAPES[2];
/// The store's reply when it is out of memory under `noeviction`.
const OOM_REPLY: &str = "OOM command not allowed when used memory > 'maxmemory'.";

const POOL_SIZE: u32 = 4;
const CREATED_BY: &str = "upstream-unavailable-user";
/// The layer every fixture creates, and every `UpdateLayer` targets.
const LAYER: &str = "upstream-layer";
/// The layer a `CreateLayer` positive tries to create.
const NEW_LAYER: &str = "upstream-new-layer";
const CLIENT_ID: &str = "upstream-writer";
const OPERATION_ID: &str = "6f1c2a3b-4d5e-4f60-8a7b-9c0d1e2f3a4b";

// ═════════════════════════════════════════════════════════════════════════════
// Staged store replies
// ═════════════════════════════════════════════════════════════════════════════

type Matcher = Box<dyn Fn(&[Vec<u8>]) -> bool + Send + Sync>;

/// The proxy's one staged reply: while armed, every command `matcher` accepts
/// is answered `-<reply>`, as the server's own error reply; disarmed, every
/// command passes through.
#[derive(Clone, Default)]
struct Stage(Arc<Mutex<Option<(Matcher, &'static str)>>>);

impl Stage {
    fn rule(&self) -> ProxyRule {
        let staged = Arc::clone(&self.0);
        Arc::new(move |args: &[Vec<u8>]| {
            let staged = staged.lock().unwrap();
            let (matcher, reply) = staged.as_ref()?;
            matcher(args).then(|| ProxyAction::Eval(error_reply_script(reply)))
        })
    }

    fn arm(
        &self,
        matcher: impl Fn(&[Vec<u8>]) -> bool + Send + Sync + 'static,
        reply: &'static str,
    ) {
        *self.0.lock().unwrap() = Some((Box::new(matcher), reply));
    }

    fn disarm(&self) {
        *self.0.lock().unwrap() = None;
    }
}

fn is_command(args: &[Vec<u8>], name: &str) -> bool {
    args.first()
        .is_some_and(|a| a.eq_ignore_ascii_case(name.as_bytes()))
}

fn arg_is(args: &[Vec<u8>], index: usize, wanted: &str) -> bool {
    args.get(index)
        .is_some_and(|a| a.as_slice() == wanted.as_bytes())
}

/// `<command> <key> …`
fn command_on(command: &'static str, key: String) -> impl Fn(&[Vec<u8>]) -> bool {
    move |args: &[Vec<u8>]| is_command(args, command) && arg_is(args, 1, &key)
}

/// `EVALSHA <sha> …` of the script whose SHA is `sha`. The scripts staged here
/// are only ever sent as `EVALSHA` (fred's `evalsha_with_reload`, or
/// `storage::lua::evalsha_once_with_reload` for the update script, answers a
/// `NOSCRIPT` with `SCRIPT LOAD` and the same `EVALSHA` again); an `EVAL`
/// would carry the script body, not its SHA. An `EVALSHA` this file refuses
/// gets the staged error, never `NOSCRIPT`, so it is not sent again.
fn is_script(args: &[Vec<u8>], sha: &str) -> bool {
    is_command(args, "EVALSHA")
        && args
            .get(1)
            .is_some_and(|a| String::from_utf8_lossy(a).eq_ignore_ascii_case(sha))
}

fn script_call(sha: String) -> impl Fn(&[Vec<u8>]) -> bool {
    move |args: &[Vec<u8>]| is_script(args, &sha)
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// A `TestValkey` whose every connection goes through a [`ScriptReplyProxy`]
/// running the fixture's [`Stage`], a session, and one layer ([`LAYER`]).
struct Fx {
    // Declared before `proxy` so it drops (and cleans up through it) first.
    tv: TestValkey,
    proxy: ScriptReplyProxy,
    stage: Stage,
    session_id: String,
    object_id: u64,
}

impl Fx {
    async fn new(name: &str) -> Self {
        let stage = Stage::default();
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), stage.rule()).await;
        let tv = TestValkey::setup_with_url(proxy.url(), POOL_SIZE).await;
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(name, &format!("/{name}"), CREATED_BY)
            .await
            .expect("session creation must succeed");
        let state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..Default::default()
        };
        let object_id = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .create_layer(
                &session_id,
                LAYER,
                CREATED_BY,
                &state,
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("create_layer must succeed");
        Self {
            tv,
            proxy,
            stage,
            session_id,
            object_id,
        }
    }

    /// A service over the fixture's (proxied) client, with an
    /// external-value store that reaches Valkey through it too.
    fn service_with_valkey_ext(&self, base: LiveEditService) -> LiveEditService {
        LiveEditService {
            ext_value_store: Arc::new(ValkeyExternalValueStore::new(
                self.tv.client().clone(),
                self.tv.key_schema().clone(),
            )),
            ..base
        }
    }

    /// What an `UpdateLayer` commit changes on [`LAYER`]'s object: its
    /// header's `seq` and its stream's length.
    async fn update_surface(&self) -> (Option<String>, u64) {
        let pool = &self.tv.client().pool;
        let seq: Option<String> = pool
            .hget(
                self.tv.key_schema().obj_header(&self.object_id.to_string()),
                "seq",
            )
            .await
            .expect("HGET seq must succeed");
        let xlen: u64 = pool
            .xlen(self.tv.key_schema().layer_stream(self.object_id))
            .await
            .expect("XLEN must succeed");
        (seq, xlen)
    }

    /// What a `CreateLayer` of [`NEW_LAYER`] changes before its commit can:
    /// the new layer's mapping and the object-id counter.
    async fn create_surface(&self) -> (Option<String>, Option<String>) {
        let pool = &self.tv.client().pool;
        let mapping: Option<String> = pool
            .get(
                self.tv
                    .key_schema()
                    .layer_mapping(&self.session_id, NEW_LAYER),
            )
            .await
            .expect("GET mapping must succeed");
        let counter: Option<String> = pool
            .get(self.tv.key_schema().counter_object_id())
            .await
            .expect("GET counter must succeed");
        (mapping, counter)
    }

    fn base_path_read(&self) -> impl Fn(&[Vec<u8>]) -> bool {
        let key = self.tv.key_schema().session_meta(&self.session_id);
        move |args: &[Vec<u8>]| {
            is_command(args, "HGET") && arg_is(args, 1, &key) && arg_is(args, 2, "base_path")
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Requests
// ═════════════════════════════════════════════════════════════════════════════

fn update_header(fx: &Fx, operation_id: &str) -> UpdateLayerHeader {
    UpdateLayerHeader {
        session_id: fx.session_id.clone(),
        layer_name: LAYER.to_owned(),
        object_id: fx.object_id,
        timestamp_microseconds_since_epoch: 1_000,
        client_id: CLIENT_ID.to_owned(),
        operation_id: operation_id.to_owned(),
    }
}

fn plain_update(fx: &Fx, operation_id: &str) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(update_header(
            fx,
            operation_id,
        ))),
    }
}

/// A chain's first write: `previous_operation_id` empty.
fn chained_update(fx: &Fx, operation_id: &str) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::ChainedHeader(
            ChainedUpdateLayerHeader {
                header: Some(update_header(fx, operation_id)),
                previous_operation_id: String::new(),
                no_wait: false,
            },
        )),
    }
}

/// A `CreateLayer` of [`NEW_LAYER`].
fn create_header(fx: &Fx, operation_id: &str) -> CreateLayerRequest {
    create_header_for(fx, NEW_LAYER, operation_id)
}

/// A `CreateLayer` of `layer`: [`LAYER`] for an overwrite, [`NEW_LAYER`] for a
/// new layer.
fn create_header_for(fx: &Fx, layer: &str, operation_id: &str) -> CreateLayerRequest {
    CreateLayerRequest {
        message_type: Some(create_layer_request::MessageType::Header(
            CreateLayerHeader {
                session_id: fx.session_id.clone(),
                layer_name: layer.to_owned(),
                operation_id: operation_id.to_owned(),
            },
        )),
    }
}

/// One inline field.
fn field_payload() -> Vec<u8> {
    delta_to_ompf_bytes(
        &delta_with_set_field_inline(1, "upstream-field", vec![0x5E], 0),
        &[],
    )
}

/// One field whose value is an external value, carried in the container.
fn external_payload() -> Vec<u8> {
    let blob: &[u8] = b"upstream-unavailable external value";
    delta_to_ompf_bytes(
        &delta_with_set_field_external(
            1,
            "upstream-external",
            ext_content_hash(blob).to_vec(),
            blob.len() as u64,
            1,
            0,
        ),
        &[blob],
    )
}

async fn update_status(
    svc: &LiveEditService,
    header: UpdateLayerRequest,
    payload: Vec<u8>,
) -> Status {
    handle_update_layer(
        svc,
        make_request(streaming_update(vec![header, make_update_data(payload)])),
    )
    .await
    .expect_err("the UpdateLayer must be refused")
}

async fn create_status(
    svc: &LiveEditService,
    header: CreateLayerRequest,
    payload: Vec<u8>,
) -> Status {
    match handle_create_layer(
        svc,
        make_request(streaming_create(vec![header, make_create_data(payload)])),
    )
    .await
    {
        Err(status) => status,
        Ok(response) => panic!("the CreateLayer must be refused; got {response:?}"),
    }
}

/// One staged store reply for a hinted positive: the command `matcher`
/// accepts is refused `reply`. `what` names the refused step in the hit-count
/// message; `message_prefix`, when given, is how the store's plain message
/// must start (its operation label).
struct Staged<M> {
    matcher: M,
    reply: &'static str,
    what: &'static str,
    message_prefix: Option<&'static str>,
}

/// Arm `staged`, run `call`, disarm; then assert the staged command was
/// refused exactly once, and the answer is the pre-commit hint carrying the
/// store's plain message unchanged.
async fn refused<M>(fx: &Fx, staged: Staged<M>, call: impl Future<Output = Status>)
where
    M: Fn(&[Vec<u8>]) -> bool + Send + Sync + 'static,
{
    let Staged {
        matcher,
        reply,
        what,
        message_prefix,
    } = staged;
    // `call` is lazy: nothing of it runs before it is awaited, after arming.
    fx.stage.arm(matcher, reply);
    let status = call.await;
    fx.stage.disarm();

    assert_eq!(fx.proxy.hits(), 1, "the {what} was refused once");
    assert_upstream_unavailable(&status);
    if let Some(prefix) = message_prefix {
        assert!(
            status.message().starts_with(prefix),
            "the store's plain message, unchanged, naming {prefix:?}; got {status:?}"
        );
    }
    assert!(
        status.message().contains(reply),
        "the store's plain message, carrying its reply; got {status:?}"
    );
}

/// A hinted `UpdateLayer` positive: [`refused`], and the layer's `seq` and
/// stream length unchanged.
async fn refused_update<M>(
    fx: &Fx,
    svc: &LiveEditService,
    staged: Staged<M>,
    request: UpdateLayerRequest,
    payload: Vec<u8>,
) where
    M: Fn(&[Vec<u8>]) -> bool + Send + Sync + 'static,
{
    let before = fx.update_surface().await;
    refused(fx, staged, update_status(svc, request, payload)).await;
    assert_eq!(fx.update_surface().await, before, "nothing applied");
}

/// A hinted `CreateLayer` positive: [`refused`], no mapping created and the
/// object-id counter unchanged.
async fn refused_create<M>(
    fx: &Fx,
    svc: &LiveEditService,
    staged: Staged<M>,
    request: CreateLayerRequest,
    payload: Vec<u8>,
) where
    M: Fn(&[Vec<u8>]) -> bool + Send + Sync + 'static,
{
    let before = fx.create_surface().await;
    refused(fx, staged, create_status(svc, request, payload)).await;
    assert_eq!(
        fx.create_surface().await,
        before,
        "no mapping created, no object id allocated"
    );
}

/// `status` is a plain `UNAVAILABLE`: no refusal detail at all.
#[track_caller]
fn assert_plain_unavailable(status: &Status) {
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_no_refusal_detail(status);
}

// ── Auth doubles ────────────────────────────────────────────────────────────

/// The permission service timing out.
struct TimedOutPermissions;

#[async_trait]
impl PermissionsClient for TimedOutPermissions {
    async fn check_permission(
        &self,
        _token: &str,
        _path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        Err(AuthError::Timeout)
    }
}

/// The userinfo service unreachable.
struct UnreachableUserinfo;

#[async_trait]
impl UserinfoClient for UnreachableUserinfo {
    async fn get_current_user(&self, _token: &str) -> Result<UserInfo, AuthError> {
        Err(AuthError::Unreachable)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// UpdateLayer: refused before its commit → UPSTREAM_UNAVAILABLE
// ═════════════════════════════════════════════════════════════════════════════

/// The session base-path read refused `-LOADING` (the service's cache is
/// cold, so the read reaches the store): the hint, the store's message, and
/// nothing applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_base_path_read_refused_loading_answers_upstream_unavailable() {
    let fx = Fx::new("upstream-update-base-path").await;
    let svc = build_service(&fx.tv, Arc::new(AllowAllPermissions));

    let staged = Staged {
        matcher: fx.base_path_read(),
        reply: LOADING_REPLY,
        what: "base-path read",
        message_prefix: Some("HGET session base_path failed"),
    };
    refused_update(&fx, &svc, staged, plain_update(&fx, ""), field_payload()).await;
}

/// The permission check timed out: the hint, the auth backend's message, and
/// nothing applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_permission_check_timed_out_answers_upstream_unavailable() {
    let fx = Fx::new("upstream-update-permission").await;
    let svc = build_service(&fx.tv, Arc::new(TimedOutPermissions));
    let before = fx.update_surface().await;

    let status = update_status(&svc, plain_update(&fx, ""), field_payload()).await;

    assert_upstream_unavailable(&status);
    assert_eq!(status.message(), "auth backend: auth service timed out");
    assert_eq!(fx.update_surface().await, before, "nothing applied");
}

/// The layer-mapping read refused `-LOADING`, on an unchained call and on a
/// chain's first write: the hint, the store's message, nothing applied.
async fn mapping_read_refused_loading(name: &str, header: fn(&Fx, &str) -> UpdateLayerRequest) {
    let fx = Fx::new(name).await;
    let svc = build_service(&fx.tv, Arc::new(AllowAllPermissions));

    let staged = Staged {
        matcher: command_on(
            "GET",
            fx.tv.key_schema().layer_mapping(&fx.session_id, LAYER),
        ),
        reply: LOADING_REPLY,
        what: "mapping read",
        message_prefix: Some("GET layer mapping failed"),
    };
    refused_update(
        &fx,
        &svc,
        staged,
        header(&fx, OPERATION_ID),
        field_payload(),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_mapping_read_refused_loading_answers_upstream_unavailable() {
    mapping_read_refused_loading("upstream-update-mapping", plain_update).await;
}

/// The chained twin: the `INVALID_UPDATE` choke point passes the
/// `UNAVAILABLE` through, and the hint is added after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_mapping_read_refused_loading_on_a_chained_call_answers_upstream_unavailable()
{
    mapping_read_refused_loading("upstream-update-mapping-chained", chained_update).await;
}

/// The external-value pre-store refused `-BUSY` — a not-serving reply, so the
/// mapper's plain `UNAVAILABLE`, not the `STORE_WRITE_REFUSED` refusal: the
/// hint, and no commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_ext_pre_store_refused_busy_answers_upstream_unavailable() {
    let fx = Fx::new("upstream-update-pre-store").await;
    let svc = fx.service_with_valkey_ext(build_service(&fx.tv, Arc::new(AllowAllPermissions)));

    let staged = Staged {
        matcher: script_call(EXT_STORE_VALUE_SCRIPT.sha1().to_string()),
        reply: BUSY_REPLY,
        what: "pre-store",
        message_prefix: Some("EVALSHA ext store_values failed"),
    };
    refused_update(&fx, &svc, staged, plain_update(&fx, ""), external_payload()).await;
}

/// The external-value pre-store refused `-TRYAGAIN` — a cluster node
/// refusing the multi-key store script while its slot migrates, before the
/// script ran, which the shared classifier calls transient (b.vn4): the
/// mapper's plain `UNAVAILABLE`, so the hint, and no commit. The store's
/// reply verbatim, as Valkey 9.0.3 words it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_ext_pre_store_refused_tryagain_answers_upstream_unavailable() {
    let fx = Fx::new("upstream-update-pre-store-tryagain").await;
    let svc = fx.service_with_valkey_ext(build_service(&fx.tv, Arc::new(AllowAllPermissions)));

    let staged = Staged {
        matcher: script_call(EXT_STORE_VALUE_SCRIPT.sha1().to_string()),
        reply: TRYAGAIN_REPLY,
        what: "pre-store",
        message_prefix: Some("EVALSHA ext store_values failed"),
    };
    refused_update(&fx, &svc, staged, plain_update(&fx, ""), external_payload()).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// CreateLayer: refused before its commit → UPSTREAM_UNAVAILABLE
// ═════════════════════════════════════════════════════════════════════════════

/// The session base-path read refused `-LOADING`: the hint, no mapping, no id
/// allocated.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_base_path_read_refused_loading_answers_upstream_unavailable() {
    let fx = Fx::new("upstream-create-base-path").await;
    let svc = build_service(&fx.tv, Arc::new(AllowAllPermissions));

    let staged = Staged {
        matcher: fx.base_path_read(),
        reply: LOADING_REPLY,
        what: "base-path read",
        message_prefix: None,
    };
    refused_create(&fx, &svc, staged, create_header(&fx, ""), field_payload()).await;
}

/// A re-send's operation-record read refused `-LOADING`: the hint, no
/// mapping, no id allocated.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_operation_record_read_refused_loading_answers_upstream_unavailable() {
    let fx = Fx::new("upstream-create-op-record").await;
    let svc = build_service(&fx.tv, Arc::new(AllowAllPermissions));

    let staged = Staged {
        matcher: command_on(
            "GET",
            fx.tv
                .key_schema()
                .create_layer_operation(&fx.session_id, OPERATION_ID),
        ),
        reply: LOADING_REPLY,
        what: "operation-record read",
        message_prefix: None,
    };
    refused_create(
        &fx,
        &svc,
        staged,
        create_header(&fx, OPERATION_ID),
        field_payload(),
    )
    .await;
}

/// The userinfo service unreachable: the hint, the auth backend's message, no
/// mapping, no id allocated.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_userinfo_unreachable_answers_upstream_unavailable() {
    let fx = Fx::new("upstream-create-userinfo").await;
    let svc = build_service_with_userinfo(
        &fx.tv,
        Arc::new(AllowAllPermissions),
        Arc::new(UnreachableUserinfo),
    );
    let before = fx.create_surface().await;

    let status = create_status(&svc, create_header(&fx, ""), field_payload()).await;

    assert_upstream_unavailable(&status);
    assert_eq!(status.message(), "auth backend: auth service unreachable");
    assert_eq!(
        fx.create_surface().await,
        before,
        "no mapping created, no object id allocated"
    );
}

/// The external-value pre-store refused `-BUSY`: the hint, no mapping, no id
/// allocated.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_ext_pre_store_refused_busy_answers_upstream_unavailable() {
    let fx = Fx::new("upstream-create-pre-store").await;
    let svc = fx.service_with_valkey_ext(build_service(&fx.tv, Arc::new(AllowAllPermissions)));

    let staged = Staged {
        matcher: script_call(EXT_STORE_VALUE_SCRIPT.sha1().to_string()),
        reply: BUSY_REPLY,
        what: "pre-store",
        message_prefix: None,
    };
    refused_create(
        &fx,
        &svc,
        staged,
        create_header(&fx, ""),
        external_payload(),
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Safety negatives: refused from the commit on → no refusal detail
// ═════════════════════════════════════════════════════════════════════════════

/// The object-id allocation — the first step of `CreateLayer`'s commit —
/// refused `-LOADING`: a plain `UNAVAILABLE`, no detail. The allocation is
/// past the commit marker, and a commit-stage failure never claims it applied
/// nothing (SDD §18.1).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_object_id_allocation_refused_loading_answers_plain_unavailable() {
    let fx = Fx::new("upstream-create-allocation").await;
    let svc = build_service(&fx.tv, Arc::new(AllowAllPermissions));
    let counter_key = fx.tv.key_schema().counter_object_id();

    // The allocation is the only script the call sends that names the
    // object-id counter key (the counter exists, so no repair runs first).
    fx.stage.arm(
        move |args: &[Vec<u8>]| {
            is_command(args, "EVALSHA")
                && args.iter().any(|a| a.as_slice() == counter_key.as_bytes())
        },
        LOADING_REPLY,
    );
    let status = create_status(&svc, create_header(&fx, ""), field_payload()).await;
    fx.stage.disarm();

    assert_eq!(
        fx.proxy.hits(),
        1,
        "the allocation was refused once, and not retried"
    );
    assert_plain_unavailable(&status);
    assert!(
        status.message().starts_with("EVAL alloc_object_id failed")
            && status.message().contains(LOADING_REPLY),
        "the allocator's plain message; got {status:?}"
    );
    assert_eq!(fx.create_surface().await.0, None, "no mapping created");
}

/// The new-layer commit script refused out of memory — the handler twin of
/// `storage::store_write_refusal_test`'s `create_commit` sites: the
/// `CreateLayer` commit stage's plain `UNAVAILABLE` "… ; retry", no detail
/// (b.yvh: a `CreateLayer` commit-stage refusal is never hinted).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_commit_refused_out_of_memory_answers_plain_unavailable() {
    let fx = Fx::new("upstream-create-commit").await;
    let svc = build_service(&fx.tv, Arc::new(AllowAllPermissions));

    fx.stage.arm(
        script_call(CREATE_LAYER_COMMIT_SCRIPT.sha1().to_string()),
        OOM_REPLY,
    );
    let status = create_status(&svc, create_header(&fx, ""), field_payload()).await;
    fx.stage.disarm();

    assert_eq!(fx.proxy.hits(), 1, "the commit script was refused once");
    assert_plain_unavailable(&status);
    assert_eq!(
        status.message(),
        "the backing store refused a write this request needed because it is out of memory; \
         retry"
    );
    assert_eq!(fx.create_surface().await.0, None, "no mapping created");
}

// ═════════════════════════════════════════════════════════════════════════════
// The layer store's fault seams
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod fault_injected {
    //! The sites only a layer-store seam reaches: the call-start record read,
    //! `CreateLayer`'s commit runs and an overwrite's wake-up after the
    //! commit point, the commit loop's attempts, and the recovery re-issue.
    //! (`test-fault-injection` compiles the seams and
    //! `build_service_with_layer_store`.)

    use super::*;

    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};

    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::{CheckpointStore, RecoveryHook};
    use live_edit::config::UpdateLayerBackend;
    use live_edit::storage::lua::UPDATE_LAYER_FAST_SCRIPT;

    use crate::grpc::helpers::build_service_with_layer_store;

    type FredQueue = Arc<Mutex<VecDeque<fred::error::Error>>>;

    fn fred_queue(errors: impl IntoIterator<Item = fred::error::Error>) -> FredQueue {
        Arc::new(Mutex::new(errors.into_iter().collect()))
    }

    fn loading() -> fred::error::Error {
        fred::error::Error::new(fred::error::ErrorKind::Unknown, LOADING_REPLY)
    }

    /// A slot migration's `-TRYAGAIN`, as fred hands it over (b.vn4).
    fn tryagain() -> fred::error::Error {
        fred::error::Error::new(fred::error::ErrorKind::Unknown, TRYAGAIN_REPLY)
    }

    /// The commit loop's exhaustion, as the client sees it: a plain
    /// `UNAVAILABLE` (no detail — an earlier attempt's commit may have been
    /// sent) naming the spent budget and the `-TRYAGAIN` that spent it, with
    /// every injected failure taken and nothing landed.
    async fn assert_tryagain_exhausted(
        fx: &Fx,
        failures: &FredQueue,
        status: &Status,
        max_attempts: u32,
        before: (Option<String>, u64),
    ) {
        assert!(
            failures.lock().unwrap().is_empty(),
            "every attempt took one injected -TRYAGAIN"
        );
        assert_plain_unavailable(status);
        assert!(
            status
                .message()
                .contains(&format!("{max_attempts} attempts exhausted"))
                && status.message().contains("TRYAGAIN"),
            "the loop's exhaustion, naming the -TRYAGAIN; got {status:?}"
        );
        assert_eq!(fx.update_surface().await, before, "nothing landed");
    }

    fn store(fx: &Fx) -> LayerStore {
        LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
    }

    /// The call-start read of an unchained operation-id call's start record
    /// refused `-LOADING`: it ends the call just before its first commit
    /// attempt, before the commit marker — the hint, nothing applied.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_call_start_read_refused_loading_answers_upstream_unavailable() {
        let fx = Fx::new("upstream-update-start-record").await;
        let failures = fred_queue([loading()]);
        let svc = build_service_with_layer_store(
            &fx.tv,
            Arc::new(
                store(&fx)
                    .with_start_record_read_fred_error_injector_for_test(Arc::clone(&failures)),
            ),
        );
        let before = fx.update_surface().await;

        let status = update_status(&svc, plain_update(&fx, OPERATION_ID), field_payload()).await;

        assert!(
            failures.lock().unwrap().is_empty(),
            "the start-record read took the injected failure"
        );
        assert_upstream_unavailable(&status);
        assert!(
            status.message().contains(LOADING_REPLY),
            "the store's plain message; got {status:?}"
        );
        assert_eq!(fx.update_surface().await, before, "nothing applied");
    }

    /// The commit failure a give-up row injects on each of its call's runs,
    /// before the script runs.
    #[derive(Clone, Copy)]
    enum CommitFault {
        /// A fred `IO` error: transient, so the mapper alone answers it
        /// `UNAVAILABLE`.
        Io,
        /// A fred `Protocol` error, which is neither transient nor a store
        /// write refusal nor a wrong-type refusal, so the mapper alone would
        /// answer it `INTERNAL`.
        Permanent,
    }

    impl CommitFault {
        /// The injected error's details, which the answer must carry.
        fn details(self) -> &'static str {
            match self {
                Self::Io => "upstream-unavailable: connection reset",
                Self::Permanent => "upstream-unavailable: protocol violation",
            }
        }

        fn error(self) -> fred::error::Error {
            let kind = match self {
                Self::Io => fred::error::ErrorKind::IO,
                Self::Permanent => fred::error::ErrorKind::Protocol,
            };
            fred::error::Error::new(kind, self.details())
        }
    }

    /// The opening of the message the mapper gives a failed commit run,
    /// whatever the error's class.
    const COMMIT_ERROR_PREFIX: &str = "EVALSHA create_layer_commit failed";

    /// The reply the proxy refuses the post-failure check with in the
    /// check-failed rows.
    const CHECK_REFUSED_REPLY: &str = "ERR upstream-unavailable: the commit check failed";

    /// Where a give-up row's call stops: with its run budget spent and the
    /// check still naming the expected value, or at a check that failed too.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum GiveUp {
        BudgetSpent,
        CheckFailed,
    }

    /// `layer`'s mapping, read directly.
    async fn mapping_of(fx: &Fx, layer: &str) -> Option<String> {
        fx.tv
            .client()
            .pool
            .get(fx.tv.key_schema().layer_mapping(&fx.session_id, layer))
            .await
            .expect("GET mapping must succeed")
    }

    /// A `CreateLayer` of `layer` whose every commit run fails with `fault`
    /// before its script runs, until the call gives up at `exit`; the call's
    /// finisher, when it spawns one, has no budget, so nothing lands. Asserts
    /// every injected fault was taken, that the check was refused exactly
    /// when `exit` says so, and that no run landed (`layer`'s mapping did not
    /// move). Returns the call's status.
    async fn commit_failure_gave_up(
        name: &str,
        layer: &str,
        fault: CommitFault,
        exit: GiveUp,
    ) -> Status {
        use live_edit::storage::layer::CreateCommitFault;

        let fx = Fx::new(name).await;
        let runs = match exit {
            GiveUp::BudgetSpent => 2,
            GiveUp::CheckFailed => 1,
        };
        let faults: Arc<Mutex<VecDeque<CreateCommitFault>>> = Arc::new(Mutex::new(
            (0..runs)
                .map(|_| CreateCommitFault {
                    error: fault.error(),
                    script_ran: false,
                })
                .collect(),
        ));
        let svc = build_service_with_layer_store(
            &fx.tv,
            Arc::new(
                store(&fx)
                    .with_create_layer_commit_fault_injector_for_test(Arc::clone(&faults))
                    .with_create_commit_finisher_timing_for_test(
                        std::time::Duration::from_millis(1),
                        std::time::Duration::from_millis(1),
                        std::time::Duration::ZERO,
                    ),
            ),
        );
        let mapped_before = mapping_of(&fx, layer).await;
        let request = create_header_for(&fx, layer, "");
        if exit == GiveUp::CheckFailed {
            // The check is the only `MGET` naming a commit claim.
            fx.stage.arm(
                |args: &[Vec<u8>]| {
                    is_command(args, "MGET")
                        && args
                            .iter()
                            .any(|a| a.windows(13).any(|w| w == b":createclaim:"))
                },
                CHECK_REFUSED_REPLY,
            );
        }

        let status = create_status(&svc, request, field_payload()).await;
        fx.stage.disarm();

        assert!(
            faults.lock().unwrap().is_empty(),
            "precondition: each of the call's {runs} commit run(s) took an injected failure"
        );
        assert_eq!(
            fx.proxy.hits(),
            usize::from(exit == GiveUp::CheckFailed),
            "precondition: the check after the failed run was refused exactly in the \
             check-failed rows"
        );
        assert_eq!(
            mapping_of(&fx, layer).await,
            mapped_before,
            "precondition: no run landed, so the mapping did not move"
        );
        status
    }

    /// A give-up row's answer: a plain `UNAVAILABLE` — no refusal detail, so
    /// no "applied nothing" claim — carrying the commit's failure, `fault`'s
    /// details included. For a permanent-class fault that is not the
    /// redacted `INTERNAL` the mapper alone gives it, which would claim the
    /// content had not taken effect.
    #[track_caller]
    fn assert_gave_up_plain_unavailable(status: &Status, fault: CommitFault) {
        assert_plain_unavailable(status);
        assert!(
            status.message().starts_with(COMMIT_ERROR_PREFIX)
                && status.message().contains(fault.details()),
            "the commit's own failure, its message kept; got {status:?}"
        );
    }

    /// An overwrite whose commit runs both failed before their script ran,
    /// with the old object still mapped: a plain `UNAVAILABLE`, no detail.
    /// The call fenced the old object, and its finisher (here given no budget,
    /// so nothing moves during the test) or the orphan sweep may still
    /// complete the replacement after the answer, so "this call applied
    /// nothing" would be false.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_whose_commit_runs_failed_answers_plain_unavailable() {
        let status = commit_failure_gave_up(
            "upstream-create-commit-runs",
            LAYER,
            CommitFault::Io,
            GiveUp::BudgetSpent,
        )
        .await;

        assert_gave_up_plain_unavailable(&status, CommitFault::Io);
    }

    /// The same overwrite with a permanent-class error on both runs, the
    /// check still naming the old object, gives up with a plain `UNAVAILABLE`
    /// carrying that error.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_whose_commit_runs_failed_permanently_answers_plain_unavailable()
    {
        let status = commit_failure_gave_up(
            "upstream-create-perm-runs",
            LAYER,
            CommitFault::Permanent,
            GiveUp::BudgetSpent,
        )
        .await;

        assert_gave_up_plain_unavailable(&status, CommitFault::Permanent);
    }

    /// A new layer whose two commit runs fail with a permanent-class error
    /// gives up with the same plain `UNAVAILABLE`: a run of it may still be
    /// on the wire and map the layer after the answer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_new_layer_whose_commit_runs_failed_permanently_answers_plain_unavailable()
    {
        let status = commit_failure_gave_up(
            "upstream-create-new-perm-runs",
            NEW_LAYER,
            CommitFault::Permanent,
            GiveUp::BudgetSpent,
        )
        .await;

        assert_gave_up_plain_unavailable(&status, CommitFault::Permanent);
    }

    /// An overwrite whose commit run fails with a permanent-class error and
    /// whose check after it fails too — whether the commit landed is unknown
    /// — gives up with a plain `UNAVAILABLE` carrying the commit's error.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_whose_permanent_commit_failure_check_failed_too_answers_plain_unavailable(
    ) {
        let status = commit_failure_gave_up(
            "upstream-create-perm-check",
            LAYER,
            CommitFault::Permanent,
            GiveUp::CheckFailed,
        )
        .await;

        assert_gave_up_plain_unavailable(&status, CommitFault::Permanent);
    }

    /// A new layer whose commit run fails with a permanent-class error and
    /// whose check after it fails too gives up with the same plain
    /// `UNAVAILABLE`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_new_layer_whose_permanent_commit_failure_check_failed_too_answers_plain_unavailable(
    ) {
        let status = commit_failure_gave_up(
            "upstream-create-new-perm-check",
            NEW_LAYER,
            CommitFault::Permanent,
            GiveUp::CheckFailed,
        )
        .await;

        assert_gave_up_plain_unavailable(&status, CommitFault::Permanent);
    }

    /// An overwrite that passed `CreateLayer`'s commit point — the mapping
    /// now names the new object — and then failed its old-stream wake-up
    /// transiently answers OK, not `UNAVAILABLE`: the layer WAS replaced, and
    /// the overwrite's cleanup retries the wake-up.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_wake_up_failed_after_the_commit_point_answers_ok() {
        let fx = Fx::new("upstream-create-wake-up").await;
        let failures = fred_queue([fred::error::Error::new(
            fred::error::ErrorKind::IO,
            "upstream-unavailable: the overwrite's wake-up was lost",
        )]);
        let svc = build_service_with_layer_store(
            &fx.tv,
            Arc::new(
                store(&fx).with_create_layer_overwrite_xadd_fault_injector_for_test(Arc::clone(
                    &failures,
                )),
            ),
        );
        let overwrite = CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Header(
                CreateLayerHeader {
                    session_id: fx.session_id.clone(),
                    layer_name: LAYER.to_owned(),
                    operation_id: String::new(),
                },
            )),
        };

        let answered = handle_create_layer(
            &svc,
            make_request(streaming_create(vec![
                overwrite,
                make_create_data(field_payload()),
            ])),
        )
        .await
        .expect("an overwrite whose wake-up failed after the commit point answers OK")
        .into_inner();

        assert!(
            failures.lock().unwrap().is_empty(),
            "the overwrite's wake-up took the injected failure"
        );
        let mapped: Option<String> = fx
            .tv
            .client()
            .pool
            .get(fx.tv.key_schema().layer_mapping(&fx.session_id, LAYER))
            .await
            .expect("GET mapping must succeed");
        assert_ne!(
            answered.object_id, fx.object_id,
            "the overwrite's NEW object"
        );
        assert_eq!(
            mapped,
            Some(answered.object_id.to_string()),
            "the commit point flipped the mapping to the object the call answers"
        );
    }

    /// A `-LOADING` burst that outlasts the commit loop's budget — the handler
    /// twin of `storage::update_layer_retry_test`'s
    /// `update_layer_loading_past_the_retry_budget_answers_unavailable_exhaustion_not_internal`:
    /// the loop's exhaustion reaches the client as a plain `UNAVAILABLE`, no
    /// detail — an earlier attempt's commit may have been sent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_loading_past_the_retry_budget_answers_plain_unavailable() {
        const MAX_ATTEMPTS: u32 = 4;
        let fx = Fx::new("upstream-update-exhausted").await;
        let failures = fred_queue((0..MAX_ATTEMPTS).map(|_| loading()));
        let svc = build_service_with_layer_store(
            &fx.tv,
            Arc::new(
                store(&fx)
                    .with_update_layer_backend(UpdateLayerBackend::Txn)
                    .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
                    .with_fred_error_injector_for_test(Arc::clone(&failures)),
            ),
        );
        let before = fx.update_surface().await;

        let status = update_status(&svc, plain_update(&fx, ""), field_payload()).await;

        assert!(
            failures.lock().unwrap().is_empty(),
            "every attempt took one injected -LOADING"
        );
        assert_plain_unavailable(&status);
        assert!(
            status
                .message()
                .contains(&format!("{MAX_ATTEMPTS} attempts exhausted")),
            "the loop's exhaustion; got {status:?}"
        );
        assert_eq!(fx.update_surface().await, before, "nothing landed");
    }

    /// The `WATCH` route's attempts each refused `-TRYAGAIN` (a slot that
    /// stays mid-migration), past the commit loop's budget: the commit marker
    /// is set before any `WATCH`-route attempt, so the exhaustion is a plain
    /// `UNAVAILABLE`, no detail. Before b.vn4 the reply was permanent and the
    /// call ended `INTERNAL` at attempt 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_tryagain_on_the_watch_past_the_retry_budget_answers_plain_unavailable() {
        const MAX_ATTEMPTS: u32 = 4;
        let fx = Fx::new("upstream-update-tryagain-watch").await;
        let failures = fred_queue((0..MAX_ATTEMPTS).map(|_| tryagain()));
        let svc = build_service_with_layer_store(
            &fx.tv,
            Arc::new(
                store(&fx)
                    .with_update_layer_backend(UpdateLayerBackend::Txn)
                    .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
                    .with_fred_error_injector_for_test(Arc::clone(&failures)),
            ),
        );
        let before = fx.update_surface().await;

        let status = update_status(&svc, plain_update(&fx, ""), field_payload()).await;

        assert_tryagain_exhausted(&fx, &failures, &status, MAX_ATTEMPTS, before).await;
    }

    /// The `lua` route's commit `EVALSHA` refused `-TRYAGAIN` on every
    /// attempt, past the loop's budget: a plain `UNAVAILABLE`, no detail.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_tryagain_on_the_commit_script_past_the_retry_budget_answers_plain_unavailable(
    ) {
        const MAX_ATTEMPTS: u32 = 4;
        let fx = Fx::new("upstream-update-tryagain-script").await;
        let failures = fred_queue((0..MAX_ATTEMPTS).map(|_| tryagain()));
        let svc = build_service_with_layer_store(
            &fx.tv,
            Arc::new(
                store(&fx)
                    .with_update_layer_backend(UpdateLayerBackend::Lua)
                    .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
                    .with_lua_fred_error_injector_for_test(Arc::clone(&failures)),
            ),
        );
        let before = fx.update_surface().await;

        let status = update_status(&svc, plain_update(&fx, ""), field_payload()).await;

        assert_tryagain_exhausted(&fx, &failures, &status, MAX_ATTEMPTS, before).await;
    }

    /// The recovery re-issue's own pre-store refused `-BUSY` after a first
    /// call of two attempts (attempt 1's `EVALSHA` timed out — a lost reply
    /// whose commit may still land — and attempt 2 found the object gone, so
    /// the checkpoint restore ran): a plain `UNAVAILABLE`, no detail. The
    /// refusal itself applied nothing, but the call's first attempt may have.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_recovery_reissue_pre_store_refused_busy_after_a_two_attempt_first_call_answers_plain_unavailable(
    ) {
        let fx = Fx::new("upstream-update-reissue").await;
        let timed_out = fred_queue([fred::error::Error::new(
            fred::error::ErrorKind::Timeout,
            "upstream-unavailable: the first attempt's reply was lost",
        )]);
        let mut svc = fx.service_with_valkey_ext(build_service_with_layer_store(
            &fx.tv,
            Arc::new(
                store(&fx)
                    .with_update_layer_backend(UpdateLayerBackend::Lua)
                    .with_update_layer_retry_max_attempts(2)
                    .with_lua_fred_error_injector_for_test(Arc::clone(&timed_out)),
            ),
        ));
        let metrics = crate::common::replace_metrics(&mut svc);
        let checkpoints = InMemoryCheckpointStore::new();
        crate::common::seed_layer_checkpoint(&checkpoints, fx.object_id, &fx.session_id, LAYER, 3)
            .await;
        let mut config = crate::common::test_config();
        config.checkpoint_enabled = true;
        config.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
        svc.recovery = Arc::new(RecoveryHook::new(
            Arc::new(checkpoints) as Arc<dyn CheckpointStore>,
            Arc::clone(&svc.layer_store),
            Arc::new(config),
            Arc::clone(&svc.metrics),
        ));
        crate::common::wipe_object_keys(&fx.tv, fx.object_id).await;

        // Only a pre-store sent after an update-script `EVALSHA` is refused:
        // the first call's pre-store precedes every commit attempt, the
        // re-issue's follows the first call's attempt 2.
        let committed = Arc::new(AtomicBool::new(false));
        let update_sha = UPDATE_LAYER_FAST_SCRIPT.sha1().to_string();
        let ext_sha = EXT_STORE_VALUE_SCRIPT.sha1().to_string();
        fx.stage.arm(
            move |args: &[Vec<u8>]| {
                if is_script(args, &update_sha) {
                    committed.store(true, Ordering::SeqCst);
                    return false;
                }
                committed.load(Ordering::SeqCst) && is_script(args, &ext_sha)
            },
            BUSY_REPLY,
        );
        let status = update_status(&svc, plain_update(&fx, ""), external_payload()).await;
        fx.stage.disarm();

        assert!(
            timed_out.lock().unwrap().is_empty(),
            "precondition: the first call's attempt 1 took the injected timeout"
        );
        assert_eq!(
            crate::common::recovery_metric(&metrics, "restored"),
            1,
            "precondition: the first call ended NOT_FOUND and the restore landed"
        );
        assert_eq!(
            fx.proxy.hits(),
            1,
            "the re-issue's pre-store, and only it, was refused"
        );
        assert_plain_unavailable(&status);
        assert!(
            status
                .message()
                .starts_with("EVALSHA ext store_values failed")
                && status.message().contains(BUSY_REPLY),
            "the re-pre-store's plain message; got {status:?}"
        );
    }
}
