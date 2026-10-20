# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable
from asyncio import Future
from pxr import Sdf

class Result(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    INVALID_HANDLE = 2
    NOT_INITIALIZED = 3
    NOT_FOUND = 4
    ALREADY_EXISTS = 5
    UNAUTHENTICATED = 6
    PERMISSION_DENIED = 7
    UNAVAILABLE = 8
    TIMEOUT = 9
    CANCELLED = 10
    MALFORMED_STREAM = 11
    UNSUPPORTED = 12
    RESOURCE_EXHAUSTED = 13
    BUSY = 14
    STALE_TOKEN_GENERATION = 15
    OBJECT_REPLACED = 16
    # 17 is reserved and never assigned.
    BUFFER_TOO_SMALL = 18
    OUTCOME_UNKNOWN = 19
    INTERNAL_ERROR = 20
    IO_ERROR = 21

class TransportSecurity(IntEnum):
    DEFAULT = 0
    TLS = 1
    PLAINTEXT = 2

class State(IntEnum):
    OPENING = 1
    READY = 2
    RECONNECTING = 3
    AUTH_WAIT = 4
    BLOCKED = 5
    # 6 is reserved and never assigned.
    FAILED = 7
    CLOSING = 8
    CLOSED = 9

class SessionEventKind(IntEnum):
    CONNECTION = 1
    JOINED = 2
    LEFT = 3
    MESSAGE = 4

class Severity(IntEnum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4

class DiagnosticFieldKind(IntEnum):
    UINT64 = 1
    INT64 = 2
    STRING = 3
    BOOLEAN = 4

@dataclass(frozen=True)
class AuthRequired:
    endpoint: int
    rejected_generation: int
    challenge_id: int
    reason: int
    grpc_status_present: bool
    grpc_status: int

@dataclass(frozen=True)
class StatusEvent:
    endpoint: int
    session: int
    uri: str
    status: Status

@dataclass(frozen=True)
class DiagnosticField:
    name: str
    kind: DiagnosticFieldKind
    value: int | bool | str

@dataclass(frozen=True)
class DiagnosticRecord:
    schema_version: int
    event_code: int
    event_name: str
    severity: Severity
    timestamp_ms: int
    elapsed_ms: int
    runtime_instance_id: int
    sequence: int
    endpoint: int
    subject_id: int
    incident_id: int
    fields: tuple[DiagnosticField, ...]

@dataclass(frozen=True)
class SessionEvent:
    kind: SessionEventKind
    result: Result
    connection_id: str | None
    user_name: str | None
    app_name: str | None
    bytes: bytes

@dataclass(frozen=True)
class CreatedSession:
    session_id: str
    uri: str

@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    uri: str
    name: str
    created_by: str
    created_time_ms: int

@dataclass(frozen=True)
class LayerInfo:
    created_time_ms: int
    modified_time_ms: int

@dataclass(frozen=True)
class AuthState:
    generation: int
    expires_at_ms: int
    challenge_id: int
    state: int
    reason: int

@dataclass(frozen=True)
class Capabilities:
    delegate_version: int
    diagnostic_schema_version: int
    autonomous_progress: bool

@dataclass(frozen=True)
class UriParts:
    authority: str
    session_id: str
    layer_name: str
    canonical_uri: str
    is_layer: bool

@dataclass(frozen=True)
class RequestError:
    result: Result
    operation: int = 0
    grpc_status_present: bool = False
    grpc_status: int = 0
    request_id: int = 0
    attempt_id: int = 0
    incident_id: int = 0
    detail: str = ''

@dataclass(frozen=True)
class Status:
    subject_kind: int
    state: State
    subject_id: int
    revision: int
    changed_at_ms: int
    result: Result
    operation: int
    grpc_status_present: bool
    grpc_status: int
    recovery_action: int
    progress_known: bool
    incident_id: int
    pending_write_count: int
    pending_bytes: int
    has_unconfirmed_edits: bool
    blocked_layers: int
    failed_layers: int
    blocked_sessions: int
    failed_sessions: int
    buffered_edit_count: int = 0
    read_sequence: int = 0
    applied_sequence: int = 0

class LiveError(RuntimeError):
    result: Result
    def __init__(self, result: int) -> None: ...

class RequestTimeout(LiveError):
    request: Request
    def __init__(self, request: Request) -> None: ...

class Request:
    @property
    def native_status(self) -> Result: ...
    @property
    def status(self) -> Result: ...
    @property
    def error(self) -> RequestError: ...
    @property
    def outcome(self) -> Result | tuple[Result, CreatedSession | LayerInfo | list[SessionInfo] | None]: ...
    def wait(self, timeout_ms: int = 30000) -> Result: ...
    def cancel(self, timeout_ms: int = 30000) -> Result: ...
    def close(self, timeout_ms: int = 30000) -> Result: ...
    def __enter__(self) -> Request: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

class Endpoint:
    @property
    def handle(self) -> int: ...
    def close(self, timeout_ms: int = 30000) -> Result: ...
    def __enter__(self) -> Endpoint: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

class Runtime:
    def close(self, timeout_ms: int = 30000) -> Result: ...
    def register_endpoint(self, authority: str, **options: Any) -> Endpoint: ...
    def join_session(self, uri: str, app_name: str, callback: Callable[[SessionEvent], Any], *, max_events: int = 256, max_bytes: int = 1048576) -> Session: ...
    def __enter__(self) -> Runtime: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

class Session:
    @property
    def handle(self) -> int: ...
    @property
    def status(self) -> Status: ...
    @property
    def delivery_status(self) -> Result: ...
    def leave(self, timeout_ms: int = 30000) -> Result: ...
    def close(self, timeout_ms: int = 30000) -> Result: ...
    def send_message(self, data: bytes, *, timeout_ms: int = 30000) -> Result: ...
    def send_message_async(self, data: bytes) -> Future[Result]: ...
    def send_message_with_callback(self, data: bytes, callback: Callable[[Result], Any]) -> Request: ...
    def __enter__(self) -> Session: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

class Subscription:
    @property
    def handle(self) -> int: ...
    @property
    def delivery_status(self) -> Result: ...
    @property
    def dropped_records(self) -> int: ...
    def close(self, timeout_ms: int = 30000) -> Result: ...
    def unregister(self, timeout_ms: int = 30000) -> Result: ...
    def __enter__(self) -> Subscription: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

def register_auth_required_callback(endpoint: Endpoint, callback: Callable[[AuthRequired], Any], *, on_error: Callable[[Result], Any] | None = None) -> Subscription: ...
def register_status_callback(endpoint: Endpoint | None, callback: Callable[[StatusEvent], Any], *, on_error: Callable[[Result], Any] | None = None) -> Subscription: ...
def register_diagnostic_callback(callback: Callable[[DiagnosticRecord], Any], *, endpoint: Endpoint | None = None, min_severity: int = 2, flags: int = 0, event_codes: tuple[int, ...] = (), max_records: int = 256, max_bytes: int = 1048576, on_error: Callable[[Result], Any] | None = None) -> Subscription: ...

def initialize(*, python_workers: int = 4, python_max_pending: int = 256, python_max_sessions: int = 64, python_max_subscriptions: int = 64, python_max_host_contexts: int = 8, **limits: Any) -> Runtime: ...
def shutdown(timeout_ms: int = 30000) -> Result: ...
def register_endpoint(authority: str, **options: Any) -> Endpoint: ...
def join_session(uri: str, app_name: str, callback: Callable[[SessionEvent], Any], *, max_events: int = 256, max_bytes: int = 1048576) -> Session: ...
def unregister_endpoint(endpoint: Endpoint, timeout_ms: int = 30000) -> Result: ...
def get_auth_state(endpoint: Endpoint) -> AuthState: ...
def set_access_token(endpoint: Endpoint, expected_generation: int, token: str, expires_at_ms: int = 0) -> tuple[Result, int]: ...
def fail_auth_refresh(endpoint: Endpoint, challenge_id: int) -> Result: ...
def get_endpoint_status(endpoint: Endpoint) -> Status: ...
def get_layer_status(uri: str) -> Status: ...
def get_session_status(session: Session) -> Status: ...
def list_sessions(endpoint: Endpoint, base_layer_key: str, *, timeout_ms: int = 30000) -> tuple[Result, list[SessionInfo]]: ...
def list_sessions_async(endpoint: Endpoint, base_layer_key: str) -> Future[tuple[Result, list[SessionInfo]]]: ...
def list_sessions_with_callback(endpoint: Endpoint, base_layer_key: str, callback: Callable[[Result, list[SessionInfo]], Any]) -> Request: ...
def create_session(endpoint: Endpoint, base_layer_key: str, session_name: str, *, timeout_ms: int = 30000) -> tuple[Result, CreatedSession | None]: ...
def create_session_async(endpoint: Endpoint, base_layer_key: str, session_name: str) -> Future[tuple[Result, CreatedSession | None]]: ...
def create_session_with_callback(endpoint: Endpoint, base_layer_key: str, session_name: str, callback: Callable[[Result, CreatedSession | None], Any]) -> Request: ...
def stat_layer(uri: str, *, timeout_ms: int = 30000) -> tuple[Result, LayerInfo | None]: ...
def stat_layer_async(uri: str) -> Future[tuple[Result, LayerInfo | None]]: ...
def stat_layer_with_callback(uri: str, callback: Callable[[Result, LayerInfo | None], Any]) -> Request: ...
def delete_layer(uri: str, *, timeout_ms: int = 30000) -> Result: ...
def delete_layer_async(uri: str) -> Future[Result]: ...
def delete_layer_with_callback(uri: str, callback: Callable[[Result], Any]) -> Request: ...
def delete_session(uri: str, *, timeout_ms: int = 30000) -> Result: ...
def delete_session_async(uri: str) -> Future[Result]: ...
def delete_session_with_callback(uri: str, callback: Callable[[Result], Any]) -> Request: ...
def get_abi_version() -> int: ...
def get_capabilities() -> Capabilities: ...
def get_result_string(result: int) -> str: ...
def make_session_uri(authority: str, session_id: str) -> str: ...
def make_layer_uri(session_uri: str, layer_name: str) -> str: ...
def combine_with_base_uri(base_uri: str, relative_layer_name: str) -> str: ...
def parse_uri(uri: str) -> UriParts: ...
def copy_diagnostics(endpoint: Endpoint | None = None, **options: Any) -> dict[str, Any]: ...
def export_diagnostics_json(endpoint: Endpoint | None = None, **options: Any) -> str: ...
def configure_jitter_reduction(endpoint: Endpoint, delay_constant_ms: int, delay_multiple: int, delay_maximum_ms: int) -> Result: ...
def wait_for_pending_updates(timeout_ms: int = 30000) -> Result: ...

class PublicationContext:
    @property
    def delivery_status(self) -> Result: ...
    @property
    def creations(self) -> tuple[Creation, ...]: ...
    def prepare_creation(self, seed: Sdf.Layer, live_uri: str) -> Creation: ...
    def prepare_retained_creation(self, seed: Sdf.Layer, live_uri: str) -> int: ...
    def acquire_retained_creation(self, subject_id: int) -> Creation: ...
    def close(self, timeout_ms: int = 30000) -> Result: ...
    def __enter__(self) -> PublicationContext: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

def set_publication_context(dispatch: Callable[[Callable[[], Result]], Any], *, max_pending: int = 64, on_error: Callable[[Result], Any] | None = None) -> PublicationContext: ...
def prepare_creation(seed: Sdf.Layer, live_uri: str, *, context: PublicationContext) -> Creation: ...
def save_layer(layer: Sdf.Layer, timeout_ms: int = 30000) -> Result: ...

class CreationPhase(IntEnum):
    PREPARED = 0
    CREATING = 1
    CONFIRMED = 2
    UNISSUED_FAILURE = 3
    REJECTED = 4
    OUTCOME_UNKNOWN = 5

@dataclass(frozen=True)
class CreationOutcome:
    phase: CreationPhase
    result: Result
    error: RequestError
    subject_id: int
    started: bool
    closing: bool
    quiescent: bool

class Creation:
    def start(self) -> Result: ...
    def wait(self, timeout_ms: int = 30000) -> Result: ...
    @property
    def outcome(self) -> CreationOutcome: ...
    def wait_closed(self, timeout_ms: int = 30000) -> Result: ...
    def prepare_layer(self, timeout_ms: int = 30000) -> Result: ...
    def activate(self) -> Result: ...
    @property
    def layer(self) -> Sdf.Layer: ...
    def close(self) -> Result: ...

@dataclass(frozen=True)
class RetainedCreationInfo:
    subject_id: int
    live_uri: str

def next_retained_creation(after_subject: int = 0) -> RetainedCreationInfo | None: ...
def release_retained_creation(subject_id: int) -> Result: ...
def prepare_retained_creation(seed: Sdf.Layer, live_uri: str, *, context: PublicationContext) -> int: ...
def acquire_retained_creation(subject_id: int, *, context: PublicationContext) -> Creation: ...
