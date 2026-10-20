# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent test LiveService. Opaque payload history is not a USD scene oracle.

Only the wire schema is shared with the client. Test controls are Python objects,
not RPCs, metadata magic, or a listener in the production plugin.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import re
import time
from typing import Callable

import grpc
from nvidia.omniverse.live.v1alpha import live_pb2 as pb, live_pb2_grpc as rpc


ERROR_DOMAIN = "nvidia.omniverse.live.v1alpha"
# The UNAVAILABLE message of an identified, unchained UpdateLayer call whose writer's operation record moved after the
# call was received. It carries no ErrorInfo; clients match its leading clause.
OPERATION_OVERTAKEN_MESSAGE = ("this attempt applied nothing, but an earlier attempt of this call may have committed "
                               "and the server can no longer tell: this client_id's commit record on this object moved")


def _rpc_status_types():
    """google.rpc.Status, ErrorInfo and RetryInfo, built from descriptors.

    The reference interpreter pins grpcio and protobuf only, without the
    googleapis common protos, so the messages are declared here with their
    published field numbers.
    """
    from google.protobuf import any_pb2, descriptor_pb2, descriptor_pool, duration_pb2, message_factory
    field = descriptor_pb2.FieldDescriptorProto
    pool = descriptor_pool.DescriptorPool()
    for module in (any_pb2, duration_pb2):
        well_known = descriptor_pb2.FileDescriptorProto()
        module.DESCRIPTOR.CopyToProto(well_known)
        pool.Add(well_known)
    rpc = descriptor_pb2.FileDescriptorProto(name="reference/google_rpc.proto", package="google.rpc", syntax="proto3",
                                             dependency=["google/protobuf/any.proto",
                                                         "google/protobuf/duration.proto"])
    status = rpc.message_type.add(name="Status")
    status.field.add(name="code", number=1, type=field.TYPE_INT32, label=field.LABEL_OPTIONAL)
    status.field.add(name="message", number=2, type=field.TYPE_STRING, label=field.LABEL_OPTIONAL)
    status.field.add(name="details", number=3, type=field.TYPE_MESSAGE, label=field.LABEL_REPEATED,
                     type_name=".google.protobuf.Any")
    info = rpc.message_type.add(name="ErrorInfo")
    info.field.add(name="reason", number=1, type=field.TYPE_STRING, label=field.LABEL_OPTIONAL)
    info.field.add(name="domain", number=2, type=field.TYPE_STRING, label=field.LABEL_OPTIONAL)
    entry = info.nested_type.add(name="MetadataEntry")
    entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=field.TYPE_STRING, label=field.LABEL_OPTIONAL)
    entry.field.add(name="value", number=2, type=field.TYPE_STRING, label=field.LABEL_OPTIONAL)
    info.field.add(name="metadata", number=3, type=field.TYPE_MESSAGE, label=field.LABEL_REPEATED,
                   type_name=".google.rpc.ErrorInfo.MetadataEntry")
    retry = rpc.message_type.add(name="RetryInfo")
    retry.field.add(name="retry_delay", number=1, type=field.TYPE_MESSAGE, label=field.LABEL_OPTIONAL,
                    type_name=".google.protobuf.Duration")
    pool.Add(rpc)
    return tuple(message_factory.GetMessageClass(pool.FindMessageTypeByName("google.rpc." + name))
                 for name in ("Status", "ErrorInfo", "RetryInfo"))


RpcStatus, ErrorInfo, RetryInfo = _rpc_status_types()


def status_details(code, message, reason, metadata, retry_delay_ms=None, *, domain=ERROR_DOMAIN):
    """A grpc-status-details-bin value: google.rpc.Status with one ErrorInfo, of the service's domain unless
    `domain` names another, and an optional RetryInfo."""
    status = RpcStatus(code=code.value[0], message=message)
    info = ErrorInfo(reason=reason, domain=domain, metadata={key: str(value) for key, value in metadata.items()})
    status.details.add(type_url="type.googleapis.com/google.rpc.ErrorInfo", value=info.SerializeToString())
    if retry_delay_ms is not None:
        retry = RetryInfo()
        retry.retry_delay.seconds, retry.retry_delay.nanos = divmod(retry_delay_ms * 1_000_000, 1_000_000_000)
        status.details.add(type_url="type.googleapis.com/google.rpc.RetryInfo", value=retry.SerializeToString())
    return status.SerializeToString()


# The reason of the service's ErrorInfo when it ends a join whose client sent
# nothing for three keepalive intervals.
KEEPALIVE_TIMEOUT_REASON = "CLIENT_KEEPALIVE_TIMEOUT"


def keepalive_timeout_status():
    """The grpc-status-details-bin of the UNAVAILABLE that ends a join for a client keepalive timeout."""
    status = RpcStatus(code=grpc.StatusCode.UNAVAILABLE.value[0], message="client keepalive timeout")
    info = ErrorInfo(reason=KEEPALIVE_TIMEOUT_REASON, domain=ERROR_DOMAIN)
    status.details.add(type_url="type.googleapis.com/google.rpc.ErrorInfo", value=info.SerializeToString())
    return status.SerializeToString()


@dataclass(frozen=True)
class Principal:
    name: str
    write: bool = True
    expires_us: int | None = None  # State.clock() microseconds; None never expires.
    expired: bool = False  # Set by State.expire(); independent of the clock.


@dataclass(frozen=True)
class Commit:
    index: int
    operation: str
    session: str
    layer: str = ""
    object_id: int = 0
    sequence: int = 0
    payload: bytes = b""
    writer: str = ""
    timestamp_us: int = 0
    client_timestamp_us: int = 0


@dataclass(frozen=True)
class Operation:
    """A committed write's operation record: its ID, content digest and the reply it earned."""
    operation_id: str
    digest: bytes
    reply: object


# The operation_id alphabet and length bound: 1-64 bytes of ASCII A-Z a-z 0-9 - . _ ~.
OPERATION_ID = re.compile(r"[A-Za-z0-9._~-]{1,64}")


def operation_key(method, session_id, header):
    """The key a re-send matches under, as the service keys it.

    A create matches its (session, operation_id) record, which the service keeps for 24 hours and which survives a
    later replace or delete of the layer. An update matches the last write of its writer to the same object:
    (session, layer, object_id, client_id); that writer's next write to the object replaces the record."""
    if method == "CreateLayer":
        return method, session_id, header.operation_id
    return method, session_id, header.layer_name, header.object_id, header.client_id


class Chained:
    """The UpdateLayerHeader of a chained call, with its ChainedUpdateLayerHeader's previous_operation_id and
    no_wait."""

    def __init__(self, chained):
        self.header, self.previous_operation_id = chained.header, chained.previous_operation_id
        self.no_wait = chained.no_wait

    def __getattr__(self, name):
        return getattr(self.header, name)


def predecessor(header):
    """The predecessor a chained update names, or None for an unchained one."""
    return header.previous_operation_id if isinstance(header, Chained) else None


def content_digest(header, payload):
    """The service's "same content": SHA-256 over layer_name, one 0x00 byte and the concatenated chunk bytes.

    No other header field counts, and chunk boundaries do not matter."""
    return hashlib.sha256(header.layer_name.encode() + b"\0" + payload).digest()


@dataclass
class Layer:
    object_id: int
    creator: str
    commits: list[Commit]


@dataclass
class Session:
    id: str
    path: str
    name: str
    creator: str
    created_us: int
    writers: set[str]
    layers: dict[str, Layer] = field(default_factory=dict)
    peers: dict[str, "Peer"] = field(default_factory=dict)


@dataclass
class Peer:
    id: str
    name: str
    application: str
    events: asyncio.Queue
    failed: bool = False
    close_code: grpc.StatusCode | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    input_closed: asyncio.Event = field(default_factory=asyncio.Event)
    # The keepalive interval the join's Connected reported, zero when it reported none, and the keepalive clock's
    # time of the join's last client message.
    keepalive_ms: int = 0
    last_activity_ms: int = 0
    close_details: bytes | None = None


@dataclass
class Fault:
    method: str
    attempt: int
    boundary: str
    code: grpc.StatusCode | None = None
    pause: bool = False
    reached: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    fired: int = 0
    details: bytes | None = None  # The grpc-status-details-bin of the scripted status, if any.


WRITE_ACTIONS = {"refuse", "hold", "stale", "drop_reply", "hold_reply", "delay_start", "stale_start"}


@dataclass
class WriteFault(Fault):
    """A scripted fault on one UpdateLayer write, chosen by what the write is rather than by RPC attempt.

    It fires once, on the first write that reaches its commit step and matches every given selector: `position`, the
    write's 1-based index among its writer's applied writes to the object (the writer's Nth write), `operation_id` and
    `writer` (client_id). Faults are tried in plan order. At the commit step a chained write's chain check has passed.
    A delay_start or stale_start fault fires when an identified unchained call has been received, before it reads
    its start record, and its position counts the writer's applied writes at that moment.

    - refuse: apply nothing and fail UNAVAILABLE with ErrorInfo `reason` (metadata nothing_applied=true) and,
      with `retry_delay_ms`, RetryInfo.
    - hold: set `reached`, wait for `release` without blocking any other call, then run the commit step again.
    - stale: fail the call UNAVAILABLE at once with a plain status, so the client cannot know the outcome, and keep
      its commit step pending in the service until `release`; `outcome` (APPLIED, ALREADY_COMMITTED or a status
      name) and `done` record what that late commit step did.
    - drop_reply: commit, then fail the reply UNAVAILABLE with a plain status.
    - hold_reply: commit, set `reached`, and hold the reply until `release`.
    - delay_start: set `reached` and wait for `release` before the call reads its start record, as a handler held at
      upload admission with its body buffered does; with `code` set before the release, fail the call with it instead,
      as the service drops a handler still waiting at its deadline. `done` is set once the call has read its start
      record or failed.
    - stale_start: fail the call UNAVAILABLE and retain its work before the start-record read until `release`;
      `code` set before release drops that work, otherwise it reads the record and runs its commit step. `done`
      and `outcome` record the retained work's completion, as for stale.
    """
    action: str = ""
    position: int | None = None
    operation_id: str | None = None
    writer: str | None = None
    reason: str = "UPLOAD_ADMISSION_SATURATED"
    retry_delay_ms: int | None = None
    outcome: str | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def matches(self, header, position):
        return (not self.fired and (self.position is None or self.position == position) and
                (self.operation_id is None or self.operation_id == header.operation_id) and
                (self.writer is None or self.writer == header.client_id))


@dataclass
class Chain:
    """Bookkeeping for one (session, layer, object_id, client_id), the key of the writer's operation record."""
    writes: int = 0  # Applied UpdateLayer calls by the writer to the object.
    # What the writer's latest write left, as a chain refusal's last_record names it: `absent` before any write
    # and after a lost record, `operation` while the record holds its operation ID, `no_operation_id` after a
    # write without one.
    last_record: str = "absent"


@dataclass
class Call:
    """One running update: its operation key once its identified header arrived, and whether it waits."""
    key: tuple | None = None
    header: object = None
    waiting: bool = False
    # The monotonic time, ahead of the call's deadline, after which a waiting call is refused rather than
    # committed, so the client receives the refusal before its deadline; None without a deadline.
    cutoff: float | None = None
    # An identified unchained call's start record (State.start_record), read once when the whole call has been
    # received; None for any other call.
    start: tuple | None = None


class ChainedCallContext:
    """A call's context that marks every INVALID_ARGUMENT a chaining replica gives a chained call with an ErrorInfo
    INVALID_UPDATE of the service's domain, unless the refusal names its own details."""

    def __init__(self, context, call):
        self.context, self.call = context, call

    def __getattr__(self, name):
        return getattr(self.context, name)

    async def abort(self, code, details="", trailing_metadata=()):
        if (code == grpc.StatusCode.INVALID_ARGUMENT and isinstance(self.call.header, Chained) and
                not any(key == "grpc-status-details-bin" for key, _ in trailing_metadata)):
            trailing_metadata = tuple(trailing_metadata) + (
                ("grpc-status-details-bin", status_details(code, details, "INVALID_UPDATE", {})),)
        await self.context.abort(code, details, trailing_metadata=trailing_metadata)


class DetachedAbort(Exception):
    """A status the service gives a commit step whose client already has its answer."""

    def __init__(self, code):
        super().__init__(code.name)
        self.code = code


class DetachedContext:
    """The context of a commit step that runs after its call has been answered."""

    @staticmethod
    def time_remaining():
        return None

    @staticmethod
    async def abort(code, details="", trailing_metadata=()):
        raise DetachedAbort(code)


class State:
    def __init__(self, seed: int, *, clock: Callable[[], int] | None = None,
                 max_payload=8 * 1024 * 1024, max_commits=10000,
                 max_retained_bytes=64 * 1024 * 1024, peer_queue=256):
        self.namespace = hashlib.sha256(str(seed).encode()).hexdigest()[:12]
        self.clock = clock or (lambda: time.time_ns() // 1000)
        self.max_payload = max_payload
        self.max_commits = max_commits
        self.max_retained_bytes = max_retained_bytes
        self.peer_queue = peer_queue
        self.changed = asyncio.Condition()
        self.tokens: dict[str, Principal] = {}
        self.sessions: dict[str, Session] = {}
        self.incarnations: dict[tuple[str, str], int] = {}
        self.ledger: list[Commit] = []
        self.retained_bytes = 0
        self.session_counter = self.connection_counter = 0
        self.attempts: dict[str, int] = {}
        self.faults: list[Fault] = []
        self.fault_log: list[dict] = []
        self.auth_rejections: list[dict] = []
        # False models a server older than operation IDs: it ignores operation_id, applies every request and
        # replies UNSPECIFIED.
        self.operation_ids = True
        # False models a server with operation IDs but without chained writes: its Ping reports chained_writes
        # false and it does not know UpdateLayerRequest.chained_header. pre_chaining models one replica of a
        # chaining service during a rolling upgrade or rollback: the Ping, answered by another replica, still
        # reports chained writes, but UpdateLayer refuses a chained header INVALID_ARGUMENT, with no ErrorInfo,
        # before reading any chunk. A chained call whose operation ID is in pre_chaining_operations reaches such a
        # replica once: the ID leaves the set when the call is refused.
        self.chained_writes = True
        self.pre_chaining = False
        self.pre_chaining_operations: set[str] = set()
        # False models a server that honours operation IDs but whose Ping reports no capabilities, as one released
        # before the Ping reported them: a client neither re-sends nor chains its writes.
        self.advertise_capabilities = True
        # Operation records by operation_key; the reference keeps create records for its whole lifetime, inside the
        # service's 24-hour window.
        self.operations: dict[tuple, Operation] = {}
        self.operation_log: list[dict] = []
        # Chained-write bookkeeping by operation key; (client_id, operation_id) of calls waiting for their
        # predecessor; header arrivals by (client_id, operation_id); the chain events in order; and each identified
        # update header's arrival: its operation, whether it is chained and no_wait, and its monotonic arrival time
        # and cutoff in seconds (None without a deadline).
        self.chains: dict[tuple, Chain] = {}
        self.waiting: set[tuple[str, str]] = set()
        self.arrivals: Counter = Counter()
        self.chain_log: list[dict] = []
        self.arrival_log: list[dict] = []
        # None models a service without the client keepalive: it ignores the header's client_keepalive, returns no
        # interval and refuses a JoinSessionKeepalive frame INVALID_ARGUMENT. An interval is returned in
        # SessionConnected to a join that opts in, and such a join that sends nothing for three intervals of
        # keepalive_clock_ms ends UNAVAILABLE with the service's ErrorInfo. The keepalive clock moves only through
        # advance_keepalive_clock, so a timeout happens exactly when a test advances it.
        self.keepalive_interval_ms: int | None = None
        self.keepalive_clock_ms = 0
        self.joins: list[dict] = []
        self.keepalives: dict[str, int] = {}
        self.keepalive_timeouts: list[str] = []
        self.tasks: set[asyncio.Task] = set()

    def plan(self, method, boundary, *, attempt=None, code=None, pause=False, details=None):
        if boundary not in {"before_auth", "before", "after_commit", "before_reply"}:
            raise ValueError("unknown fault boundary")
        if method not in pb.DESCRIPTOR.services_by_name["LiveService"].methods_by_name:
            raise ValueError("unknown RPC")
        attempt = attempt or self.attempts.get(method, 0) + 1
        if any((f.method, f.attempt, f.boundary) == (method, attempt, boundary) for f in self.faults):
            raise ValueError("duplicate fault target")
        fault = Fault(method, attempt, boundary, code, pause, details=details)
        self.faults.append(fault)
        return fault

    def plan_unknown(self, method, boundary="after_commit", *, attempt=None, code=grpc.StatusCode.UNAVAILABLE,
                     pause=False):
        """Loses the reply to one `method` request and fails the client's operation-ID re-send of it before the
        service evaluates it, so the write's outcome stays unknown. Returns the first request's fault."""
        fault = self.plan(method, boundary, attempt=attempt, code=code, pause=pause)
        self.plan(method, "before", attempt=fault.attempt + 1, code=grpc.StatusCode.UNAVAILABLE)
        return fault

    def plan_write(self, action, *, position=None, operation_id=None, writer=None,
                   reason="UPLOAD_ADMISSION_SATURATED", retry_delay_ms=None):
        """Plans a WriteFault; see its documentation for the actions and selectors."""
        if action not in WRITE_ACTIONS:
            raise ValueError("unknown write fault action")
        if position is None and operation_id is None:
            raise ValueError("a write fault needs a position or an operation_id")
        if position is not None and (type(position) is not int or position <= 0):
            raise ValueError("write position must be a positive integer")
        if retry_delay_ms is not None and (type(retry_delay_ms) is not int or retry_delay_ms < 0):
            raise ValueError("retry delay must be a nonnegative integer of milliseconds")
        code = grpc.StatusCode.UNAVAILABLE if action in {"refuse", "stale", "drop_reply"} else None
        boundary = f"write:{action}:{position or ''}:{operation_id or ''}:{writer or ''}"
        paused = action in {"hold", "stale", "hold_reply", "delay_start", "stale_start"}
        fault = WriteFault("UpdateLayer", 0, boundary, code, paused,
                           action=action, position=position, operation_id=operation_id, writer=writer,
                           reason=reason, retry_delay_ms=retry_delay_ms)
        self.faults.append(fault)
        return fault

    def write_fault(self, key, header, *, start=False):
        """The first unfired WriteFault matching the write about to commit under `key`, which this marks fired; with
        `start`, the first unfired delay_start or stale_start fault matching the call about to read its start record."""
        position = self.chain(key).writes + 1
        for fault in self.faults:
            if (isinstance(fault, WriteFault) and (fault.action in {"delay_start", "stale_start"}) == start and
                    fault.matches(header, position)):
                fault.fired += 1
                self.fault_log.append(dict(method="UpdateLayer", attempt=0, boundary=fault.boundary,
                                           position=position, operation_id=header.operation_id))
                return fault
        return None

    def chain(self, key):
        return self.chains.setdefault(key, Chain())

    def start_record(self, key):
        """The writer's operation record under `key` as a call's start record: the writer's applied writes to the
        object and the record itself, or None when it is absent."""
        return self.chain(key).writes, self.operations.get(key)

    def overtaken(self, key, start):
        """Whether the writer's record under `key` moved since `start`: a later write by the writer to the object, or
        a restore_checkpoint that lost the record."""
        writes, record = self.start_record(key)
        return writes != start[0] or record is not start[1]

    def knows_chained_header(self, chained):
        """Whether the replica a call reaches knows UpdateLayerRequest.chained_header."""
        return (self.operation_ids and self.chained_writes and not self.pre_chaining and
                chained.header.operation_id not in self.pre_chaining_operations)

    def chain_event(self, event, header, **fields):
        """Records a chain event and wakes observers; the state lock is held."""
        previous = predecessor(header)
        self.chain_log.append(dict(event=event, client_id=header.client_id, operation_id=header.operation_id,
                                   previous_operation_id=previous, **fields))
        self.changed.notify_all()

    def restore_checkpoint(self):
        """Models a backing-store loss and a restore from a saved snapshot: every UpdateLayer operation record is lost.

        Each writer's record is then absent, and a call waiting for a lost operation keeps waiting until its deadline
        or a cancellation. Returns how many records were lost."""
        lost = [key for key in self.operations if key[0] == "UpdateLayer"]
        for key in lost:
            self.operations.pop(key)
        for chain in self.chains.values():
            chain.last_record = "absent"
        self.chain_log.append(dict(event="restore", lost=len(lost)))
        self.wake()
        return len(lost)

    def wake(self):
        """Wakes every waiter on `changed` from code that may not hold it."""
        async def notify():
            async with self.changed:
                self.changed.notify_all()
        self.spawn(notify())

    def spawn(self, coroutine):
        task = asyncio.get_running_loop().create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def observed(self, predicate):
        """Returns once `predicate()` holds; it is checked after every state change."""
        async with self.changed:
            await self.changed.wait_for(predicate)

    async def settle(self):
        """Waits for the service's own background work, such as stale commit steps, to finish."""
        while self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)

    def issue(self, token, name, *, write=True, expires_us=None):
        """Registers a synthetic bearer; expires_us=0 issues it already expired."""
        principal = Principal(name, write, expires_us)
        self.tokens[token] = principal
        return principal

    def expire(self, token):
        """The service stops accepting an issued token from the next RPC start, whatever the clock reads later."""
        self.tokens[token] = replace(self.tokens[token], expired=True)

    def advance_keepalive_clock(self, milliseconds):
        """Moves the keepalive clock and ends every opted-in join whose last client message is three intervals old.

        Each ended join closes UNAVAILABLE with the service's ErrorInfo, and its peers receive one PeerLeft when its
        handler returns. Returns the connection IDs it ended."""
        if milliseconds < 0:
            raise ValueError("the keepalive clock never moves back")
        self.keepalive_clock_ms += milliseconds
        ended = []
        for session in self.sessions.values():
            for peer in session.peers.values():
                if (peer.keepalive_ms and peer.close_code is None and
                        self.keepalive_clock_ms - peer.last_activity_ms >= 3 * peer.keepalive_ms):
                    LiveService.fail_peer(peer, grpc.StatusCode.UNAVAILABLE)
                    peer.close_details = keepalive_timeout_status()
                    self.keepalive_timeouts.append(peer.id)
                    ended.append(peer.id)
        return ended

    async def boundary(self, method, attempt, boundary, context, backend):
        for fault in self.faults:
            if (fault.method, fault.attempt, fault.boundary) == (method, attempt, boundary):
                fault.fired += 1
                self.fault_log.append(dict(method=method, attempt=attempt, boundary=boundary, backend=backend))
                fault.reached.set()
                if fault.pause:
                    try:
                        await fault.release.wait()
                    except asyncio.CancelledError:
                        fault.cancelled.set()
                        raise
                if fault.code is not None:
                    await context.abort(fault.code, "scripted reference-service fault", trailing_metadata=(
                        (("grpc-status-details-bin", fault.details),) if fault.details is not None else ()))

    def assert_faults_fired(self):
        unused = [(f.method, f.attempt, f.boundary, f.fired) for f in self.faults if f.fired != 1]
        if unused:
            raise AssertionError(f"faults must fire exactly once: {unused}")

    async def capacity(self, context, size=0):
        if len(self.ledger) >= self.max_commits or self.retained_bytes + size > self.max_retained_bytes:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "reference ledger capacity")

    def commit(self, operation, session, **fields):
        entry = Commit(len(self.ledger) + 1, operation, session, timestamp_us=self.clock(), **fields)
        self.ledger.append(entry)
        self.retained_bytes += len(entry.payload)
        self.changed.notify_all()  # Caller holds changed.
        return entry

    def snapshot(self):
        # Payloads remain in memory for independent assertions; artifacts contain
        # hashes/counts and synthetic resource identities, never bearer tokens.
        return {"schema_version": 1, "boundary": "opaque protocol ledger, not USD semantics",
                "namespace": self.namespace, "commits": [
                    {"index": c.index, "operation": c.operation, "session": c.session,
                     "layer": c.layer, "object_id": c.object_id, "sequence": c.sequence,
                     "size": len(c.payload), "sha256": hashlib.sha256(c.payload).hexdigest(),
                     "writer": c.writer, "timestamp_us": c.timestamp_us,
                     "client_timestamp_us": c.client_timestamp_us} for c in self.ledger], "faults": list(self.fault_log),
                "fault_plan": [{"method": f.method, "attempt": f.attempt, "boundary": f.boundary,
                                "code": f.code.name if f.code else None, "pause": f.pause,
                                "fired": f.fired, "cancelled": f.cancelled.is_set()} for f in self.faults],
                "auth_rejections": list(self.auth_rejections), "operation_ids": self.operation_ids,
                "operations": list(self.operation_log), "chain": list(self.chain_log),
                "keepalive_interval_ms": self.keepalive_interval_ms,
                "joins": list(self.joins), "keepalives": dict(self.keepalives),
                "keepalive_timeouts": list(self.keepalive_timeouts)}


class LiveService(rpc.LiveServiceServicer):
    def __init__(self, state: State, backend: str):
        self.state, self.backend = state, backend

    async def begin(self, method, context):
        attempt = self.state.attempts.get(method, 0) + 1
        self.state.attempts[method] = attempt
        values = [value for key, value in context.invocation_metadata() if key == "authorization"]
        token = values[0][7:] if len(values) == 1 and values[0].startswith("Bearer ") else ""
        await self.state.boundary(method, attempt, "before_auth", context, self.backend)
        principal = self.state.tokens.get(token)
        # Checked when an RPC starts, as the real service does: an open stream is
        # never re-authenticated, and Ping checks no permission or expiry.
        reason = "unknown" if principal is None else "expired" if method != "Ping" and (principal.expired or (
            principal.expires_us is not None and principal.expires_us <= self.state.clock())) else None
        if reason:
            self.state.auth_rejections.append(dict(method=method, attempt=attempt, backend=self.backend,
                token_sha256=hashlib.sha256(token.encode()).hexdigest(), reason=reason))
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "")
        await self.state.boundary(method, attempt, "before", context, self.backend)
        return principal, attempt

    async def end(self, method, attempt, context, *, committed=False):
        if committed:
            await self.state.boundary(method, attempt, "after_commit", context, self.backend)
        await self.state.boundary(method, attempt, "before_reply", context, self.backend)

    async def session(self, identifier, principal, context, *, write=False):
        session = self.state.sessions.get(identifier)
        if session is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "session not found")
        if write and (not principal.write or principal.name not in session.writers):
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "write permission required")
        return session

    async def committed(self, method, session_id, header, payload, context):
        """The earlier commit a re-send of header.operation_id matches, or None when this request applies.

        Called with the state lock held, after the identity and incarnation checks and before any commit."""
        if not header.operation_id or not self.state.operation_ids:
            return None
        if not OPERATION_ID.fullmatch(header.operation_id):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "operation_id must be 1-64 bytes of A-Z a-z 0-9 - . _ ~")
        record = self.state.operations.get(operation_key(method, session_id, header))
        if record is None or record.operation_id != header.operation_id:
            return None
        if record.digest != content_digest(header, payload):
            self.state.operation_log.append(dict(method=method, outcome="ALREADY_EXISTS"))
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, "operation_id was committed with other content")
        self.state.operation_log.append(dict(method=method, outcome="ALREADY_COMMITTED"))
        reply = type(record.reply)()
        reply.CopyFrom(record.reply)
        reply.operation_outcome = pb.OPERATION_OUTCOME_ALREADY_COMMITTED
        return reply

    def applied(self, method, session_id, header, payload, reply):
        """Records a committed write under its operation key and marks its reply; the state lock is held."""
        key = operation_key(method, session_id, header)
        if header.operation_id and self.state.operation_ids:
            reply.operation_outcome = pb.OPERATION_OUTCOME_APPLIED
            self.state.operations[key] = Operation(header.operation_id, content_digest(header, payload), reply)
            self.state.operation_log.append(dict(method=method, outcome="APPLIED"))
        elif method == "UpdateLayer":
            # Any applied write by the writer to the object, identified or not, ends its last operation's match.
            self.state.operations.pop(key, None)
        if method == "UpdateLayer":
            chain = self.state.chain(key)
            chain.writes += 1
            chain.last_record = "operation" if key in self.state.operations else "no_operation_id"
        return reply

    async def layer(self, session_id, name, principal, context, *, write=False):
        session = await self.session(session_id, principal, context, write=write)
        layer = session.layers.get(name)
        if layer is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "layer not found")
        return session, layer

    async def CreateSession(self, request, context):
        principal, attempt = await self.begin("CreateSession", context)
        if not principal.write:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "write permission required")
        if not request.path or not request.name:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "path and name required")
        async with self.state.changed:
            await self.state.capacity(context)
            self.state.session_counter += 1
            identifier = f"{self.state.namespace}/session-{self.state.session_counter}"
            commit = self.state.commit("CreateSession", identifier)
            self.state.sessions[identifier] = Session(identifier, request.path, request.name,
                                                     principal.name, commit.timestamp_us, {principal.name})
        await self.end("CreateSession", attempt, context, committed=True)
        return pb.CreateSessionResponse(session_id=identifier)

    async def ListSessions(self, request, context):
        _, attempt = await self.begin("ListSessions", context)
        async with self.state.changed:
            result = pb.ListSessionsResponse(sessions=[pb.Session(session_id=s.id, name=s.name,
                created_by_user_name=s.creator, created_at=s.created_us)
                for s in self.state.sessions.values() if s.path == request.path])
        await self.end("ListSessions", attempt, context)
        return result

    async def DeleteSession(self, request, context):
        principal, attempt = await self.begin("DeleteSession", context)
        async with self.state.changed:
            session = await self.session(request.session_id, principal, context, write=True)
            await self.state.capacity(context)
            del self.state.sessions[session.id]
            for peer in session.peers.values():
                self.fail_peer(peer, grpc.StatusCode.NOT_FOUND)
            self.state.commit("DeleteSession", session.id)
        await self.end("DeleteSession", attempt, context, committed=True)
        return pb.DeleteSessionResponse()

    async def chunks(self, stream, context):
        header = None
        payload = bytearray()
        count = 0
        async for message in stream:
            count += 1
            if count > 8192:
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "too many chunks")
            kind = message.WhichOneof("message_type")
            if header is None and kind == "chained_header":
                if not self.state.knows_chained_header(message.chained_header):
                    # A server that does not know the arm sees a request without a header.
                    self.state.pre_chaining_operations.discard(message.chained_header.header.operation_id)
                    async with self.state.changed:
                        self.state.chain_event("unsupported", Chained(message.chained_header))
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "one header must precede chunks")
                header = Chained(message.chained_header)
                if not header.session_id or not header.layer_name:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "layer identity required")
            elif header is None and kind == "header":
                header = message.header
                if not header.session_id or not header.layer_name:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "layer identity required")
            elif header is not None and kind == "chunk":
                if len(payload) + len(message.chunk.data) > self.state.max_payload:
                    await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "payload too large")
                payload.extend(message.chunk.data)
            else:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "one header must precede chunks")
        if header is None:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "header required")
        return header, bytes(payload)

    async def CreateLayer(self, stream, context):
        principal, attempt = await self.begin("CreateLayer", context)
        header, payload = await self.chunks(stream, context)
        async with self.state.changed:
            session = await self.session(header.session_id, principal, context, write=True)
            key = (session.id, header.layer_name)
            earlier = await self.committed("CreateLayer", session.id, header, payload, context)
            if earlier is None:
                await self.state.capacity(context, len(payload))
                object_id = self.state.incarnations.get(key, 0) + 1
                self.state.incarnations[key] = object_id
                commit = self.state.commit("CreateLayer", session.id, layer=header.layer_name,
                                           object_id=object_id, sequence=1, payload=payload)
                session.layers[header.layer_name] = Layer(object_id, principal.name, [commit])
                reply = self.applied("CreateLayer", session.id, header, payload,
                                     pb.CreateLayerResponse(object_id=object_id))
        if earlier is not None:
            await self.end("CreateLayer", attempt, context)
            return earlier
        await self.end("CreateLayer", attempt, context, committed=True)
        return reply

    async def target(self, method, attempt, header, principal, context, payload):
        """The session and layer an update writes, after its identity and incarnation checks; the lock is held."""
        session, layer = await self.layer(header.session_id, header.layer_name, principal, context, write=True)
        if not header.object_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "object_id must be nonzero")
        if header.object_id != layer.object_id:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "object incarnation changed")
        return session, layer

    async def apply_update(self, method, attempt, session, layer, header, payload, context):
        """Commits one update and returns its reply; the lock is held."""
        await self.state.capacity(context, len(payload))
        commit = self.state.commit(method, session.id, layer=header.layer_name,
            object_id=layer.object_id, sequence=layer.commits[-1].sequence + 1, payload=payload,
            writer=header.client_id, client_timestamp_us=header.timestamp_microseconds_since_epoch)
        layer.commits.append(commit)
        return self.applied(method, session.id, header, payload, pb.UpdateLayerResponse(
            sequence_num=commit.sequence, server_timestamp_microseconds_since_epoch=commit.timestamp_us))

    @staticmethod
    def chained(header):
        """Whether an update arrived with a chained header."""
        return predecessor(header) is not None

    def track(self, call, header, context=None):
        """Marks an identified update's operation in flight from the arrival of its header; a client's call, with
        its `context`, is logged in arrival_log."""
        if not (self.state.operation_ids and header.operation_id):
            return
        call.key, call.header = operation_key("UpdateLayer", header.session_id, header), header
        self.state.arrivals[header.client_id, header.operation_id] += 1
        if context is not None:
            now, remaining = time.monotonic(), context.time_remaining()
            call.cutoff = None if remaining is None else now + remaining - min(0.5, remaining / 2)
            self.state.arrival_log.append(dict(
                client_id=header.client_id, operation_id=header.operation_id, chained=self.chained(header),
                no_wait=bool(getattr(header, "no_wait", False)), at=now, cutoff=call.cutoff))
        self.state.wake()

    def untrack(self, call):
        """Ends a call's operation in flight."""
        if call.key is None:
            return
        self.stop_waiting(call)
        call.key = None
        self.state.wake()

    def stop_waiting(self, call):
        if call.waiting:
            call.waiting = False
            self.state.waiting.discard((call.header.client_id, call.header.operation_id))

    async def chain_refusal(self, context, key):
        """Refuses a chained update ABORTED with the writer's latest write as last_operation_id and last_record."""
        record = self.state.operations.get(key)
        message = "previous_operation_id is not the writer's latest committed operation"
        await context.abort(grpc.StatusCode.ABORTED, message, trailing_metadata=(("grpc-status-details-bin",
            status_details(grpc.StatusCode.ABORTED, message, "PREDECESSOR_NOT_COMMITTED",
                           dict(last_operation_id=record.operation_id if record else "",
                                last_record=self.state.chain(key).last_record))),))

    async def chain_ready(self, key, header, context, call):
        """Whether an update may commit at this step under its chain; the lock is held.

        An unchained update always may. A chained one may when its writer's latest write is its predecessor: the
        record holds that operation ID, or for the empty predecessor the record is absent. Otherwise a no_wait call,
        and a commit step that runs after its call was answered, is refused ABORTED at once; any other waits for a
        state change and returns False, and is refused at its deadline. A cancellation drops a waiting call with
        nothing applied."""
        if not self.chained(header):
            return True
        record, chain = self.state.operations.get(key), self.state.chain(key)
        previous = header.previous_operation_id
        if (record.operation_id == previous) if record else (not previous and chain.last_record == "absent"):
            self.stop_waiting(call)
            return True
        last = record.operation_id if record else ""
        if header.no_wait or isinstance(context, DetachedContext):
            self.stop_waiting(call)
            self.state.chain_event("refused", header, last_operation_id=last, last_record=chain.last_record)
            await self.chain_refusal(context, key)
        if not call.waiting:
            call.waiting = True
            self.state.waiting.add((header.client_id, header.operation_id))
            self.state.chain_event("wait", header, last_operation_id=last)
        try:
            if call.cutoff is None:
                await self.state.changed.wait()
            else:
                await asyncio.wait_for(self.state.changed.wait(), max(0.0, call.cutoff - time.monotonic()))
        except TimeoutError:
            record = self.state.operations.get(key)
            self.stop_waiting(call)
            self.state.chain_event("expired", header, last_operation_id=record.operation_id if record else "",
                                   last_record=chain.last_record)
            await self.chain_refusal(context, key)
        except asyncio.CancelledError:
            self.stop_waiting(call)
            self.state.chain_event("dropped", header)
            raise
        return False

    async def hold(self, fault):
        """Waits for a write fault's release without holding the state lock; the lock is held again after."""
        fault.reached.set()
        self.state.changed.release()
        try:
            await fault.release.wait()
        except asyncio.CancelledError:
            fault.cancelled.set()
            raise
        finally:
            await self.state.changed.acquire()

    def stale(self, principal, attempt, header, payload, fault, start, *, delayed_start=False):
        """Keeps an answered call's work pending until release, preserving or delaying its start-record read."""
        call = Call(start=start)
        self.track(call, header)
        fault.reached.set()

        async def late():
            try:
                await fault.release.wait()
                if delayed_start:
                    if fault.code is not None:
                        raise DetachedAbort(fault.code)
                    call.start = self.state.start_record(operation_key("UpdateLayer", header.session_id, header))
                reply, _, _ = await self.update(principal, attempt, header, payload, DetachedContext(), call)
                fault.outcome = pb.OperationOutcome.Name(reply.operation_outcome).removeprefix("OPERATION_OUTCOME_")
            except DetachedAbort as error:
                fault.outcome = error.code.name
            finally:
                self.untrack(call)
                fault.done.set()
        self.state.spawn(late())

    async def read_start(self, call, header, context, principal, attempt, payload):
        """Reads an identified unchained call's start record once the whole call has been received, after any
        delay_start or stale_start fault. A chained call's predecessor check keeps it from applying after its writer's next write,
        so it reads none."""
        if not (self.state.operation_ids and header.operation_id) or self.chained(header):
            return
        key = operation_key("UpdateLayer", header.session_id, header)
        fault = self.state.write_fault(key, header, start=True)
        if fault is None:
            call.start = self.state.start_record(key)
            return
        if fault.action == "stale_start":
            self.stale(principal, attempt, header, payload, fault, None, delayed_start=True)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "scripted reference-service fault")
        fault.reached.set()
        try:
            await fault.release.wait()
            if fault.code is not None:
                await context.abort(fault.code, "scripted reference-service fault")
            call.start = self.state.start_record(key)
        except asyncio.CancelledError:
            fault.cancelled.set()
            raise
        finally:
            fault.done.set()

    async def update(self, principal, attempt, header, payload, context, call):
        """The commit step of one update: (reply, the write fault to apply after the commit, committed)."""
        method = "UpdateLayer"
        async with self.state.changed:
            while True:
                session, layer = await self.target(method, attempt, header, principal, context, payload)
                earlier = await self.committed(method, session.id, header, payload, context)
                if earlier is not None:
                    self.stop_waiting(call)
                    if self.chained(header):
                        self.state.chain_event("already_committed", header)
                    return earlier, None, False
                key = operation_key(method, session.id, header)
                if call.start is not None and self.state.overtaken(key, call.start):
                    # The start-record guard, after the own-key match: no attempt of the call applies once its
                    # writer's record moved, whatever earlier attempt of it may have committed.
                    self.state.operation_log.append(dict(method=method, outcome="OVERTAKEN"))
                    await context.abort(grpc.StatusCode.UNAVAILABLE, OPERATION_OVERTAKEN_MESSAGE)
                if not await self.chain_ready(key, header, context, call):
                    continue
                fault = self.state.write_fault(key, header)
                if fault and fault.action == "refuse":
                    message = "the write was refused and nothing was applied"
                    await context.abort(grpc.StatusCode.UNAVAILABLE, message, trailing_metadata=(
                        ("grpc-status-details-bin", status_details(grpc.StatusCode.UNAVAILABLE, message, fault.reason,
                                                                   dict(nothing_applied="true"),
                                                                   fault.retry_delay_ms)),))
                if fault and fault.action == "hold":
                    await self.hold(fault)
                    continue
                if fault and fault.action == "stale":
                    self.stale(principal, attempt, header, payload, fault, call.start)
                    await context.abort(grpc.StatusCode.UNAVAILABLE, "scripted reference-service fault")
                reply = await self.apply_update(method, attempt, session, layer, header, payload, context)
                if self.chained(header):
                    self.state.chain_event("commit", header, sequence=reply.sequence_num)
                return reply, fault, True

    async def UpdateLayer(self, stream, context):
        method = "UpdateLayer"
        principal, attempt = await self.begin(method, context)
        call = Call()
        context = ChainedCallContext(context, call)

        async def arrived(header):
            if self.chained(header):
                if not header.operation_id or not header.client_id:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                        "a chained write requires operation_id and client_id")
                for value in (header.operation_id, header.previous_operation_id):
                    if value and not OPERATION_ID.fullmatch(value):
                        await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                            "operation IDs must be 1-64 bytes of A-Z a-z 0-9 - . _ ~")
            self.track(call, header, context)

        async def observed():
            # The first header is checked and tracked before chunks() asks for the next message.
            async for message in stream:
                kind = message.WhichOneof("message_type")
                if call.header is None and kind == "header":
                    call.header = message.header
                    await arrived(message.header)
                elif (call.header is None and kind == "chained_header" and
                      self.state.knows_chained_header(message.chained_header)):
                    call.header = Chained(message.chained_header)
                    await arrived(call.header)
                yield message
        try:
            header, payload = await self.chunks(observed(), context)
            await self.read_start(call, header, context, principal, attempt, payload)
            reply, after, committed = await self.update(principal, attempt, header, payload, context, call)
        finally:
            self.untrack(call)
        if not committed:
            await self.end(method, attempt, context)
            return reply
        if after and after.action == "drop_reply":
            await context.abort(grpc.StatusCode.UNAVAILABLE, "scripted reference-service fault")
        if after and after.action == "hold_reply":
            after.reached.set()
            try:
                await after.release.wait()
            except asyncio.CancelledError:
                after.cancelled.set()
                raise
        await self.end(method, attempt, context, committed=True)
        return reply

    async def DeleteLayer(self, request, context):
        principal, attempt = await self.begin("DeleteLayer", context)
        async with self.state.changed:
            session, layer = await self.layer(request.session_id, request.layer_name, principal, context, write=True)
            await self.state.capacity(context)
            del session.layers[request.layer_name]
            self.state.commit("DeleteLayer", session.id, layer=request.layer_name, object_id=layer.object_id)
        await self.end("DeleteLayer", attempt, context, committed=True)
        return pb.DeleteLayerResponse()

    async def StatLayer(self, request, context):
        principal, attempt = await self.begin("StatLayer", context)
        async with self.state.changed:
            _, layer = await self.layer(request.session_id, request.layer_name, principal, context)
            result = pb.StatLayerResponse(created_by_user_name=layer.creator,
                created_at=layer.commits[0].timestamp_us, last_modified_at=layer.commits[-1].timestamp_us)
        await self.end("StatLayer", attempt, context)
        return result

    @staticmethod
    def frames(commit):
        yield pb.SubscribeReadLayerResponse(header=pb.SubscribeReadLayerHeader(
            object_id=commit.object_id, sequence_num=commit.sequence, total_size=len(commit.payload),
            client_timestamp_microseconds_since_epoch=commit.client_timestamp_us,
            server_timestamp_microseconds_since_epoch=commit.timestamp_us))
        for offset in range(0, len(commit.payload), 65536):
            yield pb.SubscribeReadLayerResponse(chunk=pb.SubscribeReadLayerChunk(data=commit.payload[offset:offset+65536]))

    @staticmethod
    def caught_up(object_id, sequence):
        """The Live Edit service's catch-up terminator: header only, no timestamps."""
        return pb.SubscribeReadLayerResponse(header=pb.SubscribeReadLayerHeader(
            object_id=object_id, sequence_num=sequence, total_size=0))

    async def SubscribeReadLayer(self, request, context):
        principal, attempt = await self.begin("SubscribeReadLayer", context)
        object_id, cursor = request.object_id, request.sequence_num
        initial = True
        # Opaque payloads cannot be coalesced, so catch-up follows the Live Edit
        # service's per-commit mode: a fresh subscriber gets every update from
        # the base CreateLayer payload, a resumer every update after its cursor,
        # and a cursor at or ahead of the latest sequence nothing. Catch-up never
        # suppresses the subscriber's own updates and always ends with one
        # caught-up header at the latest sequence; live delivery then suppresses
        # them. A replacement incarnation re-bootstraps the subscriber the same
        # way. Opaque history is never compacted.
        while True:
            async with self.state.changed:
                _, layer = await self.layer(request.session_id, request.layer_name, principal, context)
                fresh = object_id != layer.object_id or cursor == 0
                if fresh:
                    object_id, cursor = layer.object_id, 0
                catching_up = initial or fresh
                entries = [c for c in layer.commits if c.sequence > cursor]
                latest = layer.commits[-1]
                if not entries and not initial:
                    await self.state.changed.wait()
                    continue
                outgoing = [c for c in entries if catching_up or not request.client_id or c.writer != request.client_id]
                cursor = latest.sequence
            if initial:
                await self.end("SubscribeReadLayer", attempt, context)
                initial = False
            for commit in outgoing:
                for frame in self.frames(commit):
                    yield frame
            if catching_up:
                yield self.caught_up(object_id, cursor)

    @staticmethod
    def fail_peer(peer, code=grpc.StatusCode.RESOURCE_EXHAUSTED):
        peer.failed = True
        peer.close_code = code
        while not peer.events.empty():
            peer.events.get_nowait()
        peer.ready.set()

    @staticmethod
    async def close_peer(peer, context):
        if peer.close_details is not None:
            await context.abort(peer.close_code, "client keepalive timeout",
                                trailing_metadata=(("grpc-status-details-bin", peer.close_details),))
        await context.abort(peer.close_code, "peer event queue closed")

    def emit(self, peer, event):
        if peer.failed or peer.input_closed.is_set():
            return
        try:
            peer.events.put_nowait(event)
            peer.ready.set()
        except asyncio.QueueFull:
            self.fail_peer(peer)

    async def JoinSession(self, stream, context):
        principal, attempt = await self.begin("JoinSession", context)
        first = await anext(stream, None)
        if first is None or first.WhichOneof("request_type") != "header" or not first.header.user_application:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "join header and application required")
        session_id = first.header.session_id
        async with self.state.changed:
            session = await self.session(session_id, principal, context)
            if len(session.peers) >= 128:
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "presence capacity")
            self.state.connection_counter += 1
            identifier = f"{self.state.namespace}/connection-{self.state.connection_counter}"
            interval = self.state.keepalive_interval_ms if first.header.client_keepalive else None
            peer = Peer(identifier, principal.name, first.header.user_application, asyncio.Queue(self.state.peer_queue),
                        keepalive_ms=interval or 0, last_activity_ms=self.state.keepalive_clock_ms)
            self.state.joins.append(dict(connection=identifier, client_keepalive=first.header.client_keepalive,
                                         keepalive_interval_ms=peer.keepalive_ms))
            self.emit(peer, pb.JoinSessionResponse(connected=pb.SessionConnected(
                connection_id=identifier, keepalive_interval_ms=peer.keepalive_ms)))
            for other in session.peers.values():
                self.emit(peer, pb.JoinSessionResponse(peer_joined=pb.PeerJoined(
                    connection_id=other.id, user_name=other.name, user_application=other.application)))
                self.emit(other, pb.JoinSessionResponse(peer_joined=pb.PeerJoined(
                    connection_id=peer.id, user_name=peer.name, user_application=peer.application)))
            session.peers[identifier] = peer

        async def receive():
            try:
                async for request in stream:
                    # Each request frame read restarts the keepalive timeout. The service restarts it on any bytes
                    # of the stream; the Python gRPC server exposes nothing below a whole frame.
                    peer.last_activity_ms = self.state.keepalive_clock_ms
                    kind = request.WhichOneof("request_type")
                    # A service without the client keepalive knows no JoinSessionKeepalive frame.
                    if kind == "keepalive" and self.state.keepalive_interval_ms is not None:
                        async with self.state.changed:
                            self.state.keepalives[peer.id] = self.state.keepalives.get(peer.id, 0) + 1
                            self.state.changed.notify_all()
                        continue
                    if kind != "message":
                        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "message required after join header")
                    if len(request.message.data) > self.state.max_payload:
                        await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "message too large")
                    async with self.state.changed:
                        current = await self.session(session_id, principal, context)
                        for other in current.peers.values():
                            if other.id != peer.id:
                                self.emit(other, pb.JoinSessionResponse(peer_message=pb.PeerMessage(
                                    connection_id=peer.id, data=request.message.data)))
            finally:
                # EOF is control state, not a queued data item. It cannot turn
                # a full but valid response queue into overflow or drop events.
                peer.input_closed.set()
                peer.ready.set()

        reader = asyncio.create_task(receive())
        try:
            await self.end("JoinSession", attempt, context)
            while True:
                await peer.ready.wait()
                if peer.close_code is not None:
                    await self.close_peer(peer, context)
                if not peer.events.empty():
                    event = peer.events.get_nowait()
                    if peer.events.empty() and not peer.input_closed.is_set():
                        peer.ready.clear()
                    yield event
                elif peer.input_closed.is_set():
                    await reader  # Propagate protocol errors from the receive task.
                    if peer.close_code is not None:
                        await self.close_peer(peer, context)
                    return
                else:
                    peer.ready.clear()
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            async with self.state.changed:
                current = self.state.sessions.get(session_id)
                if current and current.peers.pop(peer.id, None):
                    for other in current.peers.values():
                        self.emit(other, pb.JoinSessionResponse(peer_left=pb.PeerLeft(connection_id=peer.id)))

    async def Ping(self, request, context):
        _, attempt = await self.begin("Ping", context)
        result = pb.PingResponse(client_timestamp_microseconds_since_epoch=request.client_timestamp_microseconds_since_epoch,
                                server_timestamp_microseconds_since_epoch=self.state.clock(),
                                operation_ids=self.state.advertise_capabilities and self.state.operation_ids,
                                chained_writes=(self.state.advertise_capabilities and self.state.operation_ids and
                                                self.state.chained_writes))
        await self.end("Ping", attempt, context)
        return result


class ReferenceServer:
    def __init__(self, state, backend="A", *, credentials=None, service_factory=LiveService):
        self.state, self.backend, self.credentials = state, backend, credentials
        self.server = grpc.aio.server(options=[("grpc.max_receive_message_length", state.max_payload + 65536),
                                               ("grpc.max_send_message_length", state.max_payload + 65536)],
                                      maximum_concurrent_rpcs=256)
        rpc.add_LiveServiceServicer_to_server(service_factory(state, backend), self.server)
        self.port = 0

    async def start(self):
        if self.credentials:
            self.port = self.server.add_secure_port("127.0.0.1:0", self.credentials)
        else:
            self.port = self.server.add_insecure_port("127.0.0.1:0")
        if not self.port:
            raise RuntimeError("reference service could not bind loopback")
        await self.server.start()
        return f"127.0.0.1:{self.port}"

    async def stop(self):
        await self.server.stop(0)
        await self.server.wait_for_termination()
