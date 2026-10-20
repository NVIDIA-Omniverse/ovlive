# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded graph-aware reference policy, not a deployed service guarantee.

Graph conflicts follow scene_graph_model's service rules: a Create of a live
identity is INVALID_ARGUMENT with the service's status text, retired identities
may be created again, and occupied slots are replaced. CreateLayer alone
explicitly replaces an incarnation. Captured production
structural authoring remains gated independently of this test service.

Reads follow the Live Edit service's default catch-up: a fresh subscriber gets
one full snapshot, a resumer behind the latest sequence one coalesced diff based
at its cursor, which includes the resumer's own acknowledged writes, and a
cursor at or ahead of the latest sequence no content. Both cover only sections
reachable from the root through stored children lists; a section a Reorder left
unlisted is absent from a snapshot. Reference policy: a
resume diff retracts an unlisted section the resumer held, where the service
neither retracts nor restates it. Every
catch-up ends with one caught-up header (total_size=0) at the latest sequence.
Each layer retains its newest `retained_updates` updates for live delivery
(the service's stream MAXLEN, 4096 by default); coalesced diffs come from
per-row versions, so a cursor older than that window still gets one diff.
"""
import copy
from dataclasses import dataclass, replace
import hashlib
import time

import grpc

from service import (State, LiveService, ReferenceServer, Layer, Commit, Fault, ERROR_DOMAIN, ErrorInfo,  # noqa: F401
                     RetryInfo, RpcStatus, pb, status_details)
import scene_graph_model as graph
import scene_graph_wire as wire
from scene_graph_snapshot import snapshot, coalesced, Versions

MAX_USED_IDS = 65536
MAX_OBSERVATIONS = 1024
# The live edit service's limits on one delta: its struct commands, set fields
# and time samples together, and the length of one child reorder's list.
MAX_DELTA_ENTRIES = 99_999
MAX_CHILDREN_LIST_ENTRIES = 100_000


def entry_limit_status(reason, metadata, message):
    """The grpc-status-details-bin of the service's INVALID_ARGUMENT entry-limit refusal."""
    return status_details(grpc.StatusCode.INVALID_ARGUMENT, message, reason, metadata)


# The reason the reference gives its UNAVAILABLE refusals. The service's
# reasons for a refusal that applied nothing are not final; the client keys
# on the domain alone.
UNAVAILABLE_REASON = "UPLOAD_NOT_ADMITTED"


def unavailable_status(*, error_info=True, domain=ERROR_DOMAIN, reason=UNAVAILABLE_REASON, retry_delay_ms=None):
    """The grpc-status-details-bin of an UNAVAILABLE refusal, or None for a plain UNAVAILABLE.

    The service attaches an ErrorInfo of its domain and a RetryInfo to every
    UNAVAILABLE that applied nothing; either can be left out to model a
    refusal without it.
    """
    if not error_info and retry_delay_ms is None:
        return None
    status = RpcStatus(code=grpc.StatusCode.UNAVAILABLE.value[0], message="service unavailable")
    if error_info:
        info = ErrorInfo(reason=reason, domain=domain)
        status.details.add(type_url="type.googleapis.com/google.rpc.ErrorInfo", value=info.SerializeToString())
    if retry_delay_ms is not None:
        retry = RetryInfo()
        retry.retry_delay.seconds, milliseconds = divmod(retry_delay_ms, 1000)
        retry.retry_delay.nanos = milliseconds * 1_000_000
        status.details.add(type_url="type.googleapis.com/google.rpc.RetryInfo", value=retry.SerializeToString())
    return status.SerializeToString()


@dataclass
class GraphLayer(Layer):
    model: graph.Model
    used_ids: frozenset[int]
    snapshot_bytes: bytes
    versions: Versions
    # (section, key) fields an update erased, which a fresh subscriber's snapshot keeps as ERASED fields, as the
    # service's compacted snapshot keeps them.
    erased: frozenset = frozenset()


class GraphState(State):
    def __init__(self, seed, *, retained_updates=graph.MAX_COMMITS, delta_entry_limit=MAX_DELTA_ENTRIES,
                 children_list_limit=MAX_CHILDREN_LIST_ENTRIES, **kwargs):
        kwargs["max_payload"] = min(kwargs.get("max_payload", wire.MAX_BYTES), wire.MAX_BYTES)
        kwargs["max_retained_bytes"] = min(kwargs.get("max_retained_bytes", graph.MAX_HISTORY_BYTES),
                                           graph.MAX_HISTORY_BYTES)
        super().__init__(seed, **kwargs)
        self.retained_updates = retained_updates
        # The service's entry limits, which a test can lower below this
        # profile's own record bound to reach the refusal.
        self.delta_entry_limit = delta_entry_limit
        self.children_list_limit = children_list_limit
        # How a committed Reorder is stored: "verbatim", as the live edit service's 0.1 line stores it, the
        # list replacing the parent's whole children list and a non-child or repeat rejecting the update; or
        # "per_list", as its later line orders each child list the list names and ignores those IDs.
        self.reorder_rule = "verbatim"
        # How many newest commits a fresh snapshot is acked below and delivered again after it: the service's
        # catch-up acks its pin when it cannot prove a stable cut (live_edit_service sdd.md §12.3). A read filters
        # the reader's own commits only above its open position, the latest commit when the request is handled, as
        # the service's self-echo filter does (live_edit_service subscribe.rs is_self_echo). The service reads its
        # pin after the request arrives, so it delivers again only commits made after the read opened; the cut
        # reaches below that, so a reader applies held commits again, its own among them.
        self.unverified_cut = 0
        # How a subscriber receives an empty commit: "diff", as the service sends it, an empty diff based at the
        # previous sequence (live_edit_service diff_read.rs empty_commit_payload); "header", a header-only frame;
        # or "silent", nothing.
        self.empty_commit_form = "diff"
        self.rejections = []
        self.rejections_dropped = 0
        self.reads = []
        self.reads_dropped = 0
        # Resumed reads whose catch-up nets to nothing, so only the caught-up header follows the cursor.
        self.net_empty_catch_ups = []
        self.net_empty_catch_ups_dropped = 0
        self.partial_reads = {}
        # Planned UNAVAILABLE refusals of the next UpdateLayer and CreateLayer
        # uploads, in order per method, and each refused upload's identity and
        # bytes.
        self.planned_refusals = []
        self.refusals = []
        self.refusals_dropped = 0
        # Each CreateLayer upload that reached the service: its attempt,
        # operation ID, payload digest and monotonic arrival time.
        self.creates = []
        self.creates_dropped = 0

    def refuse_updates(self, count, *, error_info=True, domain=ERROR_DOMAIN, retry_delay_ms=None,
                       method="UpdateLayer"):
        """Refuse the next `count` uploads of `method` UNAVAILABLE before anything applies.

        Each refusal reads the whole upload, then applies nothing; its
        grpc-status-details-bin is unavailable_status's. Every planned refusal
        must be used, as every planned fault must fire.
        """
        if method not in ("UpdateLayer", "CreateLayer"):
            raise ValueError('only UpdateLayer and CreateLayer uploads are refused')
        if type(count) is not int or count <= 0 or len(self.planned_refusals) + count > MAX_OBSERVATIONS:
            raise ValueError('refusal count must be positive and bounded')
        details = unavailable_status(error_info=error_info, domain=domain, retry_delay_ms=retry_delay_ms)
        self.planned_refusals += [dict(method=method, details=details, error_info=error_info, domain=domain,
                                       retry_delay_ms=retry_delay_ms)] * count

    def refuse_creates(self, count, **kwargs):
        """Refuse the next `count` CreateLayer uploads UNAVAILABLE before anything applies, as refuse_updates."""
        self.refuse_updates(count, method="CreateLayer", **kwargs)

    def assert_faults_fired(self):
        super().assert_faults_fired()
        if self.planned_refusals:
            raise AssertionError(f"planned refusals must all be used: {len(self.planned_refusals)} left")

    def plan_partial_read(self, attempt, sequence, prefix_bytes):
        """Yield a valid header and incomplete first chunk, then wait for control."""
        if (type(attempt) is not int or type(sequence) is not int or attempt <= 0 or sequence <= 0 or
                type(prefix_bytes) is not int or not 0 < prefix_bytes < 65536):
            raise ValueError('partial-read gate needs positive attempt/sequence and a bounded prefix')
        boundary = f'after_partial:{sequence}'
        if len(self.faults) >= MAX_OBSERVATIONS or (attempt, sequence) in self.partial_reads:
            raise ValueError('duplicate or over-capacity partial-read gate')
        fault = Fault('SubscribeReadLayer', attempt, boundary, None, True)
        self.faults.append(fault)
        self.partial_reads[attempt, sequence] = prefix_bytes
        return fault

    def plan_read_frame(self, attempt, sequence, *, include_suppressed=False):
        """Pause an exact entry outside locks, optionally before own-writer suppression."""
        if type(attempt) is not int or type(sequence) is not int or attempt <= 0 or sequence <= 0:
            raise ValueError('read-frame gate needs positive attempt and sequence')
        boundary = f'{"before_entry" if include_suppressed else "before_frame"}:{sequence}'
        if len(self.faults) >= MAX_OBSERVATIONS or any(
                (f.method, f.attempt, f.boundary) == ('SubscribeReadLayer', attempt, boundary) for f in self.faults):
            raise ValueError('duplicate or over-capacity read-frame gate')
        fault = Fault('SubscribeReadLayer', attempt, boundary, None, True)
        self.faults.append(fault)
        return fault

    def observe(self, target, entry):
        records = getattr(self, target)
        if len(records) < MAX_OBSERVATIONS:
            records.append(entry)
        else:
            name = target + "_dropped"
            setattr(self, name, getattr(self, name) + 1)

    def snapshot(self):
        result = super().snapshot()
        result.update(boundary="independent bounded graph reference policy; not deployed structural authorization",
                      graph_rejections=list(self.rejections), graph_rejections_dropped=self.rejections_dropped,
                      graph_reads=list(self.reads), graph_reads_dropped=self.reads_dropped,
                      graph_net_empty_catch_ups=list(self.net_empty_catch_ups),
                      graph_net_empty_catch_ups_dropped=self.net_empty_catch_ups_dropped,
                      graph_refusals=list(self.refusals), graph_refusals_dropped=self.refusals_dropped,
                      graph_creates=list(self.creates), graph_creates_dropped=self.creates_dropped,
                      graph_partial_reads=[dict(attempt=a, sequence=s, prefix_bytes=n)
                                           for (a, s), n in self.partial_reads.items()])
        return result


def _retained_snapshot(model):
    """Half the wire budgets reserve room for any before/after diff union."""
    state = model.state
    data = snapshot(state, source_format=model.source_format)
    records = len(state)
    strings = len(model.source_format.encode("utf-8"))
    budget = wire._Budget()
    for node in state:
        strings += len(node["name"].encode("utf-8"))
        records += len(node["fields"])
        for key, value in node["fields"].items():
            strings += len(key.encode("utf-8"))
            wire._value(value, budget)
    records += wire.MAX_RECORDS - budget.records
    strings += wire.MAX_STRING_BYTES - budget.strings
    if (len(data) > wire.MAX_BYTES // 2 or records > wire.MAX_RECORDS // 2 or
            strings > wire.MAX_STRING_BYTES // 2):
        raise graph.GraphError("snapshot/resume union capacity", "limit")
    return data


class GraphService(LiveService):
    async def reject(self, method, attempt, context, kind, *, code=None, status_message=None, details=None):
        # FAILED_PRECONDITION means only an object/incarnation change, as on the
        # service. A graph outside this reference profile is a definitive
        # INVALID_ARGUMENT rejection (reference policy: the service may accept it).
        code = code or {"malformed": grpc.StatusCode.INVALID_ARGUMENT,
                       "unsupported": grpc.StatusCode.INVALID_ARGUMENT,
                       "incarnation": grpc.StatusCode.FAILED_PRECONDITION,
                       "limit": grpc.StatusCode.RESOURCE_EXHAUSTED}[kind]
        self.state.observe("rejections", dict(method=method, attempt=attempt, backend=self.backend,
                                             kind=kind, code=code.value[0]))
        if self.state.changed.locked():
            self.state.changed.notify_all()  # Tests synchronize on the recorded rejection.
        # No raw payload, token, or exception text is exposed; a reproduced
        # service status message carries only what the service itself returns,
        # and `details` its google.rpc.Status with ErrorInfo.
        await context.abort(code, status_message or "graph reference policy: " + kind,
                            trailing_metadata=(("grpc-status-details-bin", details),) if details else ())

    async def check_entry_limits(self, method, attempt, context, payload):
        """Refuse a delta over the service's entry limits as the service does.

        The service counts struct commands, set fields and time samples
        together against max_entries, then each child reorder's list against
        its own limit, before it processes the upload; an empty upload holds
        no entries. A payload this profile cannot frame is left to decode().
        """
        counts = wire.entry_counts(payload) if payload else (0, 0, 0, ())
        if counts is None:
            return
        commands, fields, samples, children = counts
        limit = self.state.delta_entry_limit
        if commands + fields + samples > limit:
            await self.reject(method, attempt, context, "delta_too_large", code=grpc.StatusCode.INVALID_ARGUMENT,
                              status_message="delta too large",
                              details=entry_limit_status("DELTA_TOO_LARGE", dict(
                                  struct_commands=commands, set_fields=fields, time_samples=samples, max_entries=limit),
                                  "delta too large"))
        limit = self.state.children_list_limit
        for length in children:
            if length > limit:
                await self.reject(method, attempt, context, "children_list_too_long",
                                  code=grpc.StatusCode.INVALID_ARGUMENT, status_message="children list too long",
                                  details=entry_limit_status("CHILDREN_LIST_TOO_LONG", dict(limit=limit, actual=length),
                                                             "children list too long"))

    async def refuse_planned(self, method, attempt, context, header, payload):
        """Refuse this upload UNAVAILABLE, applying nothing, when a refusal is planned.

        The refusal records the upload's operation ID, writer (empty for a
        CreateLayer, whose header carries none) and payload digest, so a test
        can compare the uploads a client sends again.
        """
        index = next((i for i, planned in enumerate(self.state.planned_refusals) if planned["method"] == method), None)
        if index is None:
            return
        planned = self.state.planned_refusals.pop(index)
        self.state.observe("refusals", dict(
            method=method, attempt=attempt, backend=self.backend, operation_id=header.operation_id,
            client_id=getattr(header, "client_id", ""), sha256=hashlib.sha256(payload).hexdigest(), monotonic_ns=time.monotonic_ns(),
            error_info=planned["error_info"], domain=planned["domain"], retry_delay_ms=planned["retry_delay_ms"]))
        self.state.changed.notify_all()  # Tests synchronize on the recorded refusal.
        details = planned["details"]
        await context.abort(grpc.StatusCode.UNAVAILABLE, "service unavailable",
                            trailing_metadata=(("grpc-status-details-bin", details),) if details else ())

    def staged(self, payload, sequence, old=None):
        retained = self.state.retained_updates
        if type(retained) is not int or not 0 < retained <= graph.MAX_COMMITS:
            raise ValueError("retained update count must be within the model history bound")
        if old is not None and not payload:
            # An empty upload commits an empty delta that consumes a sequence (service SDD §8.4).
            return old.model, old.used_ids, old.snapshot_bytes, old.versions, old.erased
        delta = wire.decode(payload)
        if old is not None and delta.is_diff:
            raise graph.GraphError("updates require ordinary commands", "unsupported")
        if old is None and delta.is_diff and delta.base_version:
            raise graph.GraphError("initial diff requires reset base", "unsupported")
        created = {command.section for command in delta.commands if command.tag in (1, 5)}
        used = frozenset({1} | created | (set(old.used_ids) if old else set()))
        if len(used) > MAX_USED_IDS:
            raise graph.GraphError("incarnation identity history bound", "limit")
        model = copy.deepcopy(old.model) if old else graph.Model()
        before, before_format = (old.model.state, old.model.source_format) if old else ([], "")
        model.trim_history(retained - 1)  # Room for this update within the retained window.
        model.apply(payload, sequence, per_list=self.state.reorder_rule == "per_list")
        versions = (old.versions if old else Versions()).advanced(
            before, model.state, sequence, before_source_format=before_format,
            after_source_format=model.source_format)
        erased = set(old.erased) if old else set()
        for field in delta.fields:
            (erased.add if field.value is None else erased.discard)((field.section, field.key))
        fields = {node["id"]: node["fields"] for node in model.state}
        erased = frozenset((section, key) for section, key in erased if section in fields and key not in fields[section])
        return model, used, _retained_snapshot(model), versions, erased

    async def CreateLayer(self, stream, context):
        method = "CreateLayer"
        principal, attempt = await self.begin(method, context)
        header, payload = await self.chunks(stream, context)
        async with self.state.changed:
            session = await self.session(header.session_id, principal, context, write=True)
            key = (session.id, header.layer_name)
            self.state.observe("creates", dict(attempt=attempt, operation_id=header.operation_id,
                                               sha256=hashlib.sha256(payload).hexdigest(),
                                               monotonic_ns=time.monotonic_ns()))
            await self.refuse_planned(method, attempt, context, header, payload)
            earlier = await self.committed(method, session.id, header, payload, context)
            if earlier is None:
                await self.check_entry_limits(method, attempt, context, payload)
                await self.state.capacity(context, len(payload))
                try:
                    model, used, full, versions, erased = self.staged(payload, 1)
                    object_id = self.state.incarnations.get(key, 0) + 1
                    if object_id >= 1 << 64:
                        raise graph.GraphError("incarnation capacity", "limit")
                    commit = Commit(len(self.state.ledger) + 1, method, session.id,
                                    layer=header.layer_name, object_id=object_id, sequence=1,
                                    payload=payload, timestamp_us=self.state.clock())
                    layer = GraphLayer(object_id, principal.name, [commit], model, used, full, versions, erased)
                    ledger = self.state.ledger + [commit]
                    layers = dict(session.layers)
                    layers[header.layer_name] = layer
                    incarnations = dict(self.state.incarnations)
                    incarnations[key] = object_id
                    retained_bytes = self.state.retained_bytes + len(payload)
                    reply = pb.CreateLayerResponse(object_id=object_id)
                except (graph.GraphError, wire.GraphWireError) as error:
                    await self.reject(method, attempt, context, error.kind,
                                      status_message=getattr(error, "status_message", None))
                except MemoryError:
                    await self.reject(method, attempt, context, "limit")
                self.state.ledger, self.state.incarnations = ledger, incarnations
                session.layers, self.state.retained_bytes = layers, retained_bytes
                reply = self.applied(method, session.id, header, payload, reply)
                self.state.changed.notify_all()
        if earlier is not None:
            await self.end(method, attempt, context)
            return earlier
        await self.end(method, attempt, context, committed=True)
        return reply

    async def target(self, method, attempt, header, principal, context, payload):
        session, old = await self.layer(header.session_id, header.layer_name, principal, context, write=True)
        if not header.object_id:
            await self.reject(method, attempt, context, "malformed")  # As the service: object_id 0.
        if header.object_id != old.object_id:
            await self.reject(method, attempt, context, "incarnation")
        await self.refuse_planned(method, attempt, context, header, payload)
        return session, old

    async def apply_update(self, method, attempt, session, old, header, payload, context):
        await self.check_entry_limits(method, attempt, context, payload)
        await self.state.capacity(context, len(payload))
        try:
            sequence = old.commits[-1].sequence + 1
            model, used, full, versions, erased = self.staged(payload, sequence, old)
            commit = Commit(len(self.state.ledger) + 1, method, session.id,
                            layer=header.layer_name, object_id=old.object_id, sequence=sequence,
                            payload=payload, writer=header.client_id, timestamp_us=self.state.clock(),
                            client_timestamp_us=header.timestamp_microseconds_since_epoch)
            retained = (old.commits + [commit])[-self.state.retained_updates:]
            layer = GraphLayer(old.object_id, old.creator, retained, model, used, full, versions, erased)
            ledger = self.state.ledger + [commit]
            layers = dict(session.layers)
            layers[header.layer_name] = layer
            retained_bytes = self.state.retained_bytes + len(payload)
            reply = pb.UpdateLayerResponse(sequence_num=sequence,
                                            server_timestamp_microseconds_since_epoch=commit.timestamp_us)
        except (graph.GraphError, wire.GraphWireError) as error:
            await self.reject(method, attempt, context, error.kind,
                              status_message=getattr(error, "status_message", None))
        except MemoryError:
            await self.reject(method, attempt, context, "limit")
        self.state.ledger, session.layers = ledger, layers
        self.state.retained_bytes = retained_bytes
        reply = self.applied(method, session.id, header, payload, reply)
        self.state.changed.notify_all()
        return reply

    def catch_up(self, layer, cursor):
        """Return the coalesced diff from `cursor` to the layer's latest state."""
        payload = coalesced(layer.model.state, layer.versions, cursor, source_format=layer.model.source_format)
        if layer.commits[0].sequence == 1:
            # With the whole incarnation retained, check the diff against the
            # exact state at the cursor; a mismatch is a reference defect.
            service = graph.Model()
            for entry in layer.commits:
                if entry.sequence > cursor:
                    break
                service.apply(entry.payload, entry.sequence, per_list=self.state.reorder_rule == "per_list")
            # The subscriber holds only the reachable state at its cursor.
            before = graph.Model()
            before.apply(snapshot(service.state, source_format=service.source_format))
            held = {node["id"] for node in before.state}
            # Clients conflict on a retraction of a section they never held.
            if any(command.tag == 6 and command.section not in held for command in wire.decode(payload).commands):
                raise graph.GraphError("coalesced diff retracts a section the subscriber never held", "unsupported")
            before.apply(payload)
            if snapshot(before.state, source_format=before.source_format) != layer.snapshot_bytes:
                raise graph.GraphError("coalesced diff does not reproduce the latest graph", "unsupported")
        return payload

    async def SubscribeReadLayer(self, request, context):
        method = "SubscribeReadLayer"
        principal, attempt = await self.begin(method, context)
        object_id, cursor = request.object_id, request.sequence_num
        initial = True
        opened = None  # The latest commit when the request is handled; own commits above it are filtered.
        while True:
            terminator = None
            async with self.state.changed:
                _, layer = await self.layer(request.session_id, request.layer_name, principal, context)
                latest = layer.commits[-1]
                if initial:
                    opened = latest.sequence
                if not initial and object_id != layer.object_id:
                    await self.reject(method, attempt, context, "incarnation")
                fresh = object_id != layer.object_id or cursor == 0
                if initial:
                    self.state.observe("reads", dict(backend=self.backend, attempt=attempt,
                                                     object=request.object_id, cursor=cursor))
                    try:
                        if fresh:
                            payload = (snapshot(layer.model.state, source_format=layer.model.source_format,
                                                erased=layer.erased) if layer.erased else layer.snapshot_bytes)
                        elif cursor >= latest.sequence:
                            payload = None  # At or ahead of the latest: only the caught-up header.
                        else:
                            payload = self.catch_up(layer, cursor)
                            if wire.decode(payload) == wire.Delta(True, cursor):
                                payload = None  # Nothing owed: only the caught-up header above the cursor.
                                self.state.observe("net_empty_catch_ups", dict(backend=self.backend, attempt=attempt,
                                                                               cursor=cursor, latest=latest.sequence))
                    except (graph.GraphError, wire.GraphWireError) as error:
                        await self.reject(method, attempt, context, error.kind)
                    except MemoryError:
                        await self.reject(method, attempt, context, "limit")
                    acked = latest
                    if fresh and self.state.unverified_cut:
                        retained = [entry for entry in layer.commits if entry.sequence < latest.sequence]
                        acked = retained[max(0, len(retained) - self.state.unverified_cut)] if retained else latest
                    outgoing = [] if payload is None else [
                        Commit(0, method, request.session_id, object_id=layer.object_id, sequence=acked.sequence,
                               payload=payload, timestamp_us=acked.timestamp_us,
                               client_timestamp_us=acked.client_timestamp_us)]
                    terminator = self.caught_up(layer.object_id, acked.sequence)
                    latest = acked
                else:
                    entries = [entry for entry in layer.commits if entry.sequence > cursor]
                    if not entries:
                        await self.state.changed.wait()
                        continue
                    if entries[0].sequence != cursor + 1:
                        # Reference policy, not a service rule: a reader whose next
                        # update was trimmed would silently miss it, so it is ended
                        # UNAVAILABLE and resubscribes from its cursor, the same
                        # status the service gives a reader more than 5 s behind.
                        await context.abort(grpc.StatusCode.UNAVAILABLE, "reader fell behind retained updates")
                    outgoing = entries
                object_id, cursor = layer.object_id, latest.sequence
            if initial:
                await self.end(method, attempt, context)
                initial = False
            for entry in outgoing:
                await self.state.boundary(method, attempt, f'before_entry:{entry.sequence}', context, self.backend)
                if request.client_id and entry.writer == request.client_id and entry.sequence > opened:
                    continue
                if not entry.payload:
                    if self.state.empty_commit_form == "silent":
                        continue
                    if self.state.empty_commit_form == "diff":
                        entry = replace(entry, payload=wire.encode(wire.Delta(True, entry.sequence - 1)))
                await self.state.boundary(method, attempt, f'before_frame:{entry.sequence}', context, self.backend)
                prefix = self.state.partial_reads.get((attempt, entry.sequence))
                if prefix is None:
                    for frame in self.frames(entry):
                        yield frame
                else:
                    if prefix >= len(entry.payload):
                        raise ValueError('partial-read prefix must leave payload bytes unissued')
                    yield next(self.frames(entry))  # Original full-size header.
                    yield pb.SubscribeReadLayerResponse(chunk=pb.SubscribeReadLayerChunk(data=entry.payload[:prefix]))
                    await self.state.boundary(method, attempt, f'after_partial:{entry.sequence}', context, self.backend)
                    for offset in range(prefix, len(entry.payload), 65536):
                        yield pb.SubscribeReadLayerResponse(chunk=pb.SubscribeReadLayerChunk(data=entry.payload[offset:offset+65536]))
            if terminator is not None:
                yield terminator


class GraphServer(ReferenceServer):
    def __init__(self, state, backend="A", *, credentials=None):
        super().__init__(state, backend, credentials=credentials, service_factory=GraphService)
