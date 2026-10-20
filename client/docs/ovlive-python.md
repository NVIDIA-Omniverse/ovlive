<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Python guide and API reference

`omni.usd.live` is the Python package for ovlive: it registers Live Services endpoints, manages sessions, reports typed status and connects a Python USD host to the plugin that backs `.live` layers. It is for engineers and agents who write a Python host or tool, and ends with a table of every public name.

## Setup

The staged plugin package at `<target-dir>/usd/omniverse/live` holds the native library `lib/libomni_usd_live.so`, the USD plugin registration `resources/plugInfo.json` and the Python package `python/omni/usd/live`, typed by `__init__.pyi`. The [build guide](ovlive-build.md) gives the `PYTHONPATH`, bytecode and library-path settings; the staged package is immutable.

Import with `from omni.usd import live` under the matching USD SDK's CPython 3.12, in the main interpreter; a subinterpreter import is rejected. Unless the host's plugin path already includes the package, register `resources/plugInfo.json` with `Plug.Registry().RegisterPlugins`, as the packaged sample does. Live layers are read and written with plain `pxr.Sdf` and `pxr.Usd` calls; the package has no layer I/O or edit function.

## Concepts

- **Runtime**: `initialize` returns a `Runtime` owner; module-level calls use the most recent open one.
- **Endpoint**: `register_endpoint` returns an `Endpoint` owner for one service authority and its credentials.
- **Live layer**: an `Sdf.Layer` with a `live://` identifier; the plugin captures and publishes its edits.
- **Publication context**: `set_publication_context` makes the calling thread the serial USD host on which opens, creations and plugin jobs run.
- **Flush job**: the plugin job that turns one host turn's edits into one update, run after the enclosing USD call returns.
- **Creation**: a `Creation` owns the single `CreateLayer` attempt that makes a live layer from an ordinary seed layer.
- **Status**: `Status` records are the operational signal; diagnostics only explain them.
- **Owners**: `close` or a context manager releases them; `BUSY` and `TIMEOUT` keep ownership, and dropping a reference cancels nothing.
- **Callbacks**: request, session and subscription callbacks run on package threads, never on the USD host thread.

## Quick start

The example needs a Live Services endpoint you can reach: `localhost:50051` stands for its authority, because the repository's reference service runs only inside the test suites. `layer_uri` names an existing live layer such as `live://localhost:50051/<session-id>/scene.live`, which [Creating layers](#creating-layers) shows how to make; `package_root` is a `pathlib.Path` to the staged package `<target-dir>/usd/omniverse/live`; and `token` is an access token the service accepts.

```python
import logging, queue, threading, types
from concurrent.futures import Future
from pxr import Plug, Sdf
from omni.usd import live

class Host:
    """The serial thread that owns USD access."""
    def __init__(self):
        self.jobs = queue.SimpleQueue()
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while (job := self.jobs.get()) is not None:
            try:
                job()
            except Exception:
                logging.exception("host job failed")  # typed status reports the failure
            del job                                  # release its captures on the host
    def dispatch(self, job):                         # enqueue and return; never run inline
        self.jobs.put(job)
    def call(self, fn):                              # run fn on the host and return its result
        future = Future()
        def invoke():
            try:
                future.set_result(fn())
            except BaseException as error:
                future.set_exception(error)
        self.jobs.put(invoke)
        return future.result(35)

Plug.Registry().RegisterPlugins(str(package_root / "resources/plugInfo.json"))
host, scene = Host(), types.SimpleNamespace()
runtime = live.initialize()
endpoint = runtime.register_endpoint("localhost:50051", access_token=token)
context = host.call(lambda: live.set_publication_context(host.dispatch))

def open_and_author():
    scene.layer = Sdf.Layer.FindOrOpen(layer_uri)
    with Sdf.ChangeBlock():                          # one host turn, one update
        prim = Sdf.CreatePrimInLayer(scene.layer, "/Example")   # reuses an existing spec
        attribute = scene.layer.GetAttributeAtPath("/Example.value") or \
            Sdf.AttributeSpec(prim, "value", Sdf.ValueTypeNames.Int)
        attribute.default = 42
host.call(open_and_author)

# Off the host, save waits for confirmation; BUSY means the flush job has not run yet.
while live.save_layer(scene.layer, 30000) == live.Result.BUSY:
    host.call(lambda: None)

host.call(lambda: setattr(scene, "layer", None))     # release the layer on the host
for owner in (context, endpoint, runtime):
    while owner.close(30000) == live.Result.BUSY:    # BUSY keeps ownership for a retry
        host.call(lambda: None)
host.jobs.put(None)
```

The packaged sample ([samples guide](ovlive-samples.md)) instead retries a save when the status callback reports the flush.

## Authoring rules

[Design section 8](ovlive-design.md#8-authoring-and-reconciliation-model) owns these rules; for a Python host they mean:

- **Open on the host, then compose.** Live layers are in-memory only. Open each one with `Sdf.Layer.FindOrOpen`, or create it, on the host, then insert it into a stage at runtime, for example into the session layer's `subLayerPaths`. Keep the returned layer referenced until the stage holds it: USD releases a layer nothing references, and composition then opens the URI again as an unopened live URI. Never save a live URI into a file's sublayers, references or payloads, and never make a live layer reference another. Composing a live URI that the host has not opened is unsupported: USD opens sublayers on its worker threads, where the open fails with `BUSY`, and USD's `Could not load sublayer` warning says to open live layers on the host before composing them ([design section 1](ovlive-design.md#1-purpose-audience-and-scope)).
- **Author on the host.** Edit and read live layers only on the thread that called `set_publication_context`, with ordinary `Sdf` and `Usd` calls.
- **One host turn is one update.** The first edit schedules a flush job that runs after the enclosing USD call and any `Sdf.ChangeBlock` return, so group related edits in one change block or host job. There is no draft, commit or snapshot call.
- **Off-host edits fail the layer.** An edit from another thread, or from a handler for a notice a plugin job sent, moves the layer to `FAILED` with `UNSUPPORTED`; the edit stays in the layer, which stays exportable.
- **The later update wins one field.** Two writers that set one field converge on the update with the later sequence; each sees its own pending value until its update is confirmed, and neither fails.
- **Dictionary keys are whole-field replacements.** Setting one key sends the whole field, which can supersede a peer's concurrent edit to another key.
- **Whole-content operations are rejected.** `CopyFrom`, `TransferContent` and `Import` into a live layer fail with `UNSUPPORTED` and change nothing; `Clear` fails the layer and discards its content.
- **Refusals raise before the change**, for example `BUSY` for an overlapping read; see the [authoring limits](ovlive-reference.md#authoring-limits).

A creation layer is read-only until activation. Copied handles to a live layer, including `Creation.layer`, follow the host rule and are released on the host.

## Publication context and max_pending

Call `set_publication_context(dispatch, *, max_pending=64, on_error=None)` on the host thread and keep the returned `PublicationContext`. The package passes each plugin job to `dispatch` from its own thread; `dispatch` enqueues it on the host and returns, and the host runs it later, outside any USD call or change block. A job returns `Result.OK`, returns `Result.CANCELLED` when its context was disarmed, or raises `LiveError`. If `dispatch` raises, the context is abandoned, latching `delivery_status` and calling `on_error`. See [design section 7](ovlive-design.md#7-host-concurrency-and-scheduling-contract).

`max_pending`, from 1 to 256, separately bounds the jobs a context holds and the creations it retains. A layer's own publication job is uncharged; the runtime allows one per layer. Flush jobs and peer-publication jobs are charged, including the job that applies the service's confirmation of the host's own edit and a flush requested by an off-host save while another flush job runs, so a live layer can hold about three charged jobs at once: size `max_pending` at three per live layer. When every slot is held, the context refuses a charged job with `RESOURCE_EXHAUSTED` and latches `delivery_status`. A refused flush job keeps the layer's edits buffered and the layer `BLOCKED` with `RESOURCE_EXHAUSTED`; its next record, on-host or off-host save, publication job, or pending-byte release on its endpoint requests the flush again. A refused publication job fails its layer, which recovers through export and reopen.

A runtime allows `python_max_host_contexts` contexts (default 8, at most 64). A new context applies to future opens; existing layers and creations keep theirs. `PublicationContext.close(timeout_ms=30000)` returns `BUSY` while a layer, creation or job still uses the context, which keeps dispatching until then.

## Creating layers

`FindOrOpen` never creates a layer. A creation makes one from an ordinary seed layer with a single `CreateLayer` attempt that replaces any content at the destination; [design section 11](ovlive-design.md#11-layer-creation-save-and-replacement) explains why it is never repeated. From the packaged sample:

```python
def prepare():                                       # on the host
    seed = Sdf.Layer.CreateAnonymous("seed.usda")
    prim = Sdf.PrimSpec(seed, "Example", Sdf.SpecifierDef, "Xform")
    Sdf.AttributeSpec(prim, "value", Sdf.ValueTypeNames.Int).default = 0
    scene.seed = seed
    scene.creation = context.prepare_creation(seed, layer_uri)  # freezes the seed; no RPC
    return scene.creation.start()                   # the single CreateLayer attempt

assert host.call(prepare) == live.Result.OK
assert scene.creation.wait(30000) == live.Result.OK  # off the host; OK means the request retired
if scene.creation.outcome.result == live.Result.OK:
    assert host.call(lambda: scene.creation.prepare_layer(30000)) == live.Result.OK
    assert host.call(scene.creation.activate) == live.Result.OK
    scene.layer = host.call(lambda: scene.creation.layer)
```

`wait` and `outcome` are thread-neutral; `start`, `prepare_layer`, `activate`, `layer` and `close` run on the original host, and other threads get `BUSY`. In `outcome.phase`, `REJECTED` is definitive and `OUTCOME_UNKNOWN` may have committed, so inspect the destination before creating there again. A creation that fails before activation loses nothing: the seed is the caller's layer.

`context.creations` retains every creation until a successful `creation.close()` on the host, so export what you need first; on `BUSY`, call `creation.wait_closed(timeout_ms=30000)` off the host and close again on the host.

**Retained creations.** `context.prepare_retained_creation(seed, live_uri)` keeps its workflow in a process registry of 64 slots, and so do `Sdf.Layer.CreateNew` of a live URI and `Sdf.Layer.Export` to another live URI until their layer activates, or when they fail. After a successful activation a host job hands the layer its attachment and frees the slot; a refused host job keeps the workflow retained. The layer closes when its last reference goes, and an `Export` destination that nothing references closes after that job. A reopen with `Sdf.Layer.FindOrOpen` waits for the closing destination; only while the host still holds a queued publication of it (a peer update or its own admitted edits) does the reopen fail at once with `ALREADY_EXISTS`, until its status reports `CLOSED`. Enumerate it with `next_retained_creation(after_subject=0)`, passing each returned `subject_id` back until it returns `None`; no runtime or context is needed. On the workflow's original host, `context.acquire_retained_creation(subject_id)` transfers one into a `Creation`, and `release_retained_creation(subject_id)` cancels and removes it. Export first: even a `BUSY` release starts cancellation, and `BUSY` keeps the entry for a retry.

## Status, waits and errors

**Status.** The `get_*_status` queries return a `Status` without an RPC or wait. `register_status_callback(endpoint_or_None, callback)` delivers each subject's initial state and changes as `StatusEvent` records; `None` observes every endpoint of the current runtime. Transient revisions may coalesce; terminal failures and `CLOSED` never do. [Design section 17](ovlive-design.md#17-operational-status) defines each state and recovery action.

**Lag.** A layer's `Status` carries `read_sequence`, the highest sequence its read delivered, and `applied_sequence`, the highest it applied; how far the host trails the stream is `status.read_sequence - status.applied_sequence` when that is positive, and zero otherwise. The plain difference can be negative for a healthy layer: the read never delivers the layer's own writes, so an acknowledged local write puts `applied_sequence` ahead of `read_sequence`. The layer stays `READY` while it lags, and the sequences refresh without a new revision or callback, so poll `get_layer_status` to follow them. `initialize(apply_lag_warning_threshold=256)` sets the lag past which the runtime records an `apply_lag_entered` warning, and `copy_diagnostics()` reports it with each update's receive-to-apply time under `metrics.apply_lag` ([diagnostic catalog](ovlive-diagnostic-catalog.md#apply-lag)).

**Waits.** A blocking call returns a `Result`, or a `(Result, value)` pair for `list_sessions`, `create_session` and `stat_layer`. If its local wait expires it raises `RequestTimeout`, whose `request` stays owned and uncancelled: wait again, read `request.outcome` or cancel. The `_async` forms need a running `asyncio` loop. From inside any package callback, including a plugin job, blocking calls, `Request.wait` and runtime, session, subscription and context closes return `BUSY` instead of waiting. Queries such as `get_layer_status` raise `LiveError` on failure.

**Results.** The [result codes](ovlive-reference.md#result-codes) table defines every `Result`. `OUTCOME_UNKNOWN` means a mutation may have committed: never retry it. A `create_session` request with `native_status` `OK` and status `OUTCOME_UNKNOWN` created a session whose identity was lost; reconcile with `list_sessions`. The package never retries.

**Saving.** `save_layer(layer, timeout_ms=30000)` flushes the layer's host edit buffer and waits for its updates to be confirmed. On the host it polls and returns `BUSY` while work is pending; off the host it waits within the timeout, but returns `BUSY` after scheduling a flush when host edits are unflushed. `wait_for_pending_updates(timeout_ms=30000)` drains admitted writes process-wide without flushing. See [status and drain](ovlive-reference.md#status-and-drain).

**Credentials.** Call `register_auth_required_callback(endpoint, callback)` right after registering the endpoint. Acquire a token off the callback, then call `set_access_token(endpoint, expected_generation, token)` with the event's `rejected_generation`, or `fail_auth_refresh(endpoint, challenge_id)`. A stale generation returns `STALE_TOKEN_GENERATION`.

**Failed layers.** A definitive rejection, an uncertain mutation, an unsupported flush, an off-host edit, permission denial, object replacement or malformed content moves a layer to `FAILED`, with the cause, incident ID, pending intent count and buffered edit count in its status. Nothing more is sent: peer updates stop, the visible scene freezes at the last installed state plus the host's own edits, and saves return the cause.

**Recovery.** Recovery is ordinary USD. On the host, stop offering the layer for collaborative authoring, export it, release every reference, wait for its subject to close and reopen:

The sketch below runs as a coroutine on an `asyncio` loop that drives the host thread, as Kit's main loop does, and the caller passes in its last reference to the failed layer. `subject_closed` is an application-supplied coroutine that completes when the status subscription delivers a `StatusEvent` for the given subject with state `live.State.CLOSED`.

```python
from pxr import Sdf
from omni.usd import live

async def recover(stage, layer, layer_uri, local_recovery_path, subject_closed):
    status = live.get_layer_status(layer_uri)
    if status.state != live.State.FAILED:
        return layer
    failed_subject = status.subject_id
    layer.Export(local_recovery_path)   # the frozen visible scene; a failed write raises Tf.ErrorException
    # Detach the failed layer and drop every reference, or FindOrOpen returns the cached failed layer.
    stage.GetRootLayer().subLayerPaths.remove(layer.identifier)
    del layer
    # While this coroutine waits, the loop keeps running dispatched jobs, which retire the subject.
    await subject_closed(failed_subject)
    # A new subject ID and the authoritative state. Compare the export with it and
    # reapply chosen changes explicitly.
    return Sdf.Layer.FindOrOpen(layer_uri)
```

The export keeps every local edit, whether admitted, buffered, rejected or of unknown outcome, except edits a committed peer delete or replacing create superseded before the failure and pending updates the client predicted the service to reject, such as a create of an identity a peer created first ([design section 10](ovlive-design.md#unresolved-conflicts-fail-the-layer)). Exporting cannot show whether an uncertain write committed, so it never authorizes a replay. A cached layer or `Reload` is not a fresh observation. The exported USD file is the whole recovery artifact ([design section 13](ovlive-design.md#recovery-export-then-reopen)).

## Sessions and messages

`create_session(endpoint, base_layer_key, session_name)` returns `(Result, CreatedSession)` with only the session ID and URI; names, creators and times come from `list_sessions(endpoint, base_layer_key)`.

`runtime.join_session(uri, app_name, callback, max_events=256, max_bytes=1048576)`, or `join_session` on the current runtime, returns a `Session`. Only the application name is sent; user names come from the service. The callback receives `SessionEvent` records in order on the session's own thread: `CONNECTION` per connection attempt (a reconnect can bring a new `connection_id`), `JOINED` and `LEFT` for peers, and `MESSAGE` with the payload in `bytes`. A non-OK `CONNECTION` event alone is not a stop; read `session.status`. A dropped join rejoins, as a layer read reconnects, on the backoff that `initialize(reconnect_initial_ms=1000, reconnect_maximum_ms=60000, reconnect_max_attempts=10)` configures, apart from the `retry_*` options of finite requests ([runtime options](ovlive-reference.md#runtime-options)).

`Session.send_message` succeeds when the local queue accepts the bytes, not on delivery, and a message that fails in flight is resent, so peers can receive it twice; carry your own message IDs for exactly-once handling. If the event queue (`max_events` up to 4096, `max_bytes` up to 16 MiB) overflows or an event cannot be converted, the session delivers one terminal `CONNECTION` event with `RESOURCE_EXHAUSTED` or `INTERNAL_ERROR`, latches `delivery_status`, refuses sends and stops; join again. `Session.leave(timeout_ms=30000)` stops and waits for a running callback. See [design section 15](ovlive-design.md#15-sessions-presence-and-messages).

## Dispatch, bounds and teardown

Request callbacks and future settlement run on `python_workers` threads (default 4), with at most `python_max_pending` obligations (default 256) queued or running. Each session and subscription has its own thread, up to `python_max_sessions` and `python_max_subscriptions` (default 64, at most 256). A capacity refusal never issues a request.

Shut down in order: stop authoring; save or export each live layer; release layers and close creations on the host; then `Runtime.close(timeout_ms=30000)` leaves sessions and closes subscriptions, contexts, endpoints and the runtime. Save or drain before releasing a layer: releasing its last reference discards host edits no flush has admitted, and its `CLOSED` status keeps their `buffered_edit_count` and the diagnostic history records `host_edits_discarded`; a `FAILED` layer's buffered edits survive only in its export. `BUSY` and `TIMEOUT` keep the owner for a retry; an endpoint or runtime stays `BUSY` until the runtime releases retired layers.

At interpreter exit the package disarms Python delivery, abandons publication contexts, leaves sessions and closes subscriptions, then waits for running callbacks and host jobs under a 30 s failure-only watchdog: a callback that never returns is abandoned and the interpreter exits. A native release pass then closes the remaining native sessions, subscriptions and endpoints with zero timeouts and the native runtime under its own failure-only bound, calling no Python. Owners that did not retire are listed by kind in one ERROR record on the `omni.usd.live` logger and remain owned for the rest of the process. Close explicitly before exit to observe nonquiescence.

## API reference

[`__init__.pyi`](../source/bindings/python/omni/usd/live/__init__.pyi) gives signatures and types. Finite operations are blocking (keyword `timeout_ms`, default 30000), `_async` (an `asyncio` future) or `_with_callback` (a `Request`).

### Functions

| Function | Description |
| --- | --- |
| `initialize` | Start the runtime; takes Python bounds and the [runtime options](ovlive-reference.md#runtime-options) as keywords |
| `shutdown` | Close the most recent `Runtime` |
| `get_abi_version`, `get_capabilities` | Native ABI version; `Capabilities` |
| `register_endpoint` | Register an authority with the [endpoint options](ovlive-reference.md#endpoint-options) |
| `unregister_endpoint` | Close an `Endpoint` |
| `get_auth_state` | Endpoint `AuthState` |
| `set_access_token` | Replace the token if the generation matches |
| `fail_auth_refresh` | Report failed token acquisition for a challenge |
| `register_auth_required_callback` | Subscribe to `AuthRequired` challenges |
| `configure_jitter_reduction` | Set the endpoint's peer-update holdback |
| `get_endpoint_status`, `get_session_status`, `get_layer_status` | `Status` of an endpoint, joined session, or layer or creation |
| `register_status_callback` | Subscribe to `StatusEvent` changes |
| `wait_for_pending_updates` | Drain admitted local writes |
| `list_sessions`, `list_sessions_async`, `list_sessions_with_callback` | List a base layer's sessions as `SessionInfo` |
| `create_session`, `create_session_async`, `create_session_with_callback` | Create a session; `CreatedSession` |
| `delete_session`, `delete_session_async`, `delete_session_with_callback` | Delete a session |
| `stat_layer`, `stat_layer_async`, `stat_layer_with_callback` | Service `LayerInfo` of a layer |
| `delete_layer`, `delete_layer_async`, `delete_layer_with_callback` | Delete a layer |
| `join_session` | Join on the current runtime; `Session` |
| `make_session_uri`, `make_layer_uri` | Build session and layer URIs |
| `combine_with_base_uri` | Resolve a relative `.live` name against a live URI |
| `parse_uri` | Canonicalize and split a URI into `UriParts` |
| `get_result_string` | Text of a result code |
| `register_diagnostic_callback` | Subscribe to filtered `DiagnosticRecord` events |
| `copy_diagnostics`, `export_diagnostics_json` | Diagnostic snapshot as a dictionary or JSON text; keyword options `flags`, `max_records` and `max_output_bytes` are the [export options](ovlive-diagnostic-schema.md#taking-a-snapshot). Omitting `max_records` returns every retained history record, as NULL C options do; `max_records=0` omits records |
| `set_publication_context` | Make the calling thread the USD host; `PublicationContext` |
| `save_layer` | Flush and drain a live layer |
| `prepare_creation`, `prepare_retained_creation`, `acquire_retained_creation` | The `PublicationContext` methods, with the context as a keyword |
| `next_retained_creation` | Next `RetainedCreationInfo` after a subject, or `None` |
| `release_retained_creation` | Cancel and remove a retained creation |

### Owner classes

| Class | Members |
| --- | --- |
| `Runtime` | `register_endpoint`, `join_session`, `close` |
| `Endpoint` | `handle` (not an owner), `close` |
| `Session` | `handle`, `status`, `delivery_status`, `send_message`, `send_message_async`, `send_message_with_callback`, `leave`, `close` (same as `leave`) |
| `Subscription` | `handle`, `delivery_status`, `dropped_records` (diagnostic bridge drops), `close`, `unregister` (same as `close`) |
| `Request` | `status` and `error` (effective result), `native_status` (native result), `outcome`, `wait`, `cancel`, `close` |
| `PublicationContext` | `delivery_status`, `creations`, `prepare_creation`, `prepare_retained_creation` (returns a subject ID), `acquire_retained_creation`, `close` |
| `Creation` | `start`, `wait`, `outcome`, `prepare_layer`, `activate`, `layer`, `close` (host, polls), `wait_closed` (off the host) |

### Records and exceptions

| Name | Contents |
| --- | --- |
| `Status` | State, result, subject and incident IDs, recovery action, pending and buffered counts, child counts, read and applied sequences |
| `StatusEvent` | A `Status` with endpoint, session and URI |
| `AuthRequired` | Rejected generation, challenge ID, reason, raw gRPC status |
| `AuthState` | Generation, expiry, challenge, state, reason |
| `SessionEvent` | Kind, result, connection ID, user and application names, payload |
| `CreatedSession` | Session ID and URI |
| `SessionInfo` | Session ID, URI, name, creator, creation time |
| `LayerInfo` | Creation and modification times |
| `UriParts` | Authority, session ID, layer name, canonical URI, layer flag |
| `Capabilities` | Delegate and diagnostic schema versions, autonomous progress |
| `RequestError` | Result, operation, gRPC status, request, attempt and incident IDs, detail |
| `CreationOutcome` | Phase, result, `RequestError`, subject ID, lifecycle flags |
| `RetainedCreationInfo` | Subject ID and live URI |
| `DiagnosticRecord` | Event code and name, severity, times, correlation IDs, fields |
| `DiagnosticField` | Name, kind, value |
| `LiveError` | `RuntimeError` carrying a `result` |
| `RequestTimeout` | `LiveError` for an expired local wait; `request` stays owned |

### Enums

| Enum | Values |
| --- | --- |
| `Result` | `OK` (0) through `IO_ERROR` (21); 17 is never assigned |
| `State` | `OPENING`, `READY`, `RECONNECTING`, `AUTH_WAIT`, `BLOCKED`, `FAILED`, `CLOSING`, `CLOSED` |
| `CreationPhase` | `PREPARED`, `CREATING`, `CONFIRMED`, `UNISSUED_FAILURE`, `REJECTED`, `OUTCOME_UNKNOWN` |
| `SessionEventKind` | `CONNECTION`, `JOINED`, `LEFT`, `MESSAGE` |
| `TransportSecurity` | `DEFAULT` (plaintext to loopback, else TLS), `TLS`, `PLAINTEXT` |
| `Severity` | `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `DiagnosticFieldKind` | `UINT64`, `INT64`, `STRING`, `BOOLEAN` |

## Kit hosts

In Kit, the `omni.kit.live_services` extension registers Kit's main thread as the publication context before any open, so Kit code authors live layers through `omni.usd`. The extension declares `omni.usd.live` as a public Python module and imports it only after its startup checks pass, so an extension that depends on it imports `omni.usd.live` as usual ([Kit guide](ovlive-kit.md#python-modules-and-startup-order)). The [Kit guide](ovlive-kit.md) describes the extension and what a public Kit build supports.
