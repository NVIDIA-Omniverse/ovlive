#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer and reader processes of the interactive edit benchmark (tools/live/edit_benchmark.py).

Each process speaks JSON lines: it prints `{"type": "ready"}` once connected, waits for `go` (writers) or
`stop` (readers) on stdin, writes its timeline to the JSON file named on its command line and prints
`{"type": "done"}`. Times are `time.monotonic()` seconds, comparable between processes on one host.

- `ovlive-writer` drives ovlive through its Python bindings as a USD host: the main thread is the host and
  models a Kit frame per turn, authoring the turn in one `Sdf.ChangeBlock` and then running host jobs until
  the flush job that turn requested has run (bounded by `--frame-wait-ms`). With `--batch` every turn is
  authored inside one frame. A refused USD call is recorded as a rejected turn.
- `client-library-writer` models omni-spectree's default live layer format on Omniverse Client Library:
  edits since the last tick coalesce into one update, sent at the tick only while fewer than 16 updates
  are unacknowledged (omni-spectree `PENDING_UPDATES_STOP_COUNT`); otherwise they stay buffered.
- `client-library-reader` is a peer reading through Omniverse Client Library with its default jitter
  reduction, calling `live_process` every millisecond.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
CLIENT_LIBRARY_WINDOW = 16
REGISTRATIONS = []


def turn_marker(data):
    """The highest turn number an update's attribute defaults carry, or None for an update without one.

    A coalesced update can exceed the reference profile's per-update record and byte bounds, which guard
    the service, so the readers decode with bounds sized for any update the benchmark sends.
    """
    sys.path.insert(0, str(ROOT / 'tests/reference_service'))
    import scene_graph_wire
    scene_graph_wire.MAX_BYTES = scene_graph_wire.MAX_RECORDS = scene_graph_wire.MAX_STRING_BYTES = 1 << 30
    return max((f.value for f in scene_graph_wire.decode(data).fields
                if f.key == 'default' and isinstance(f.value, float)), default=None)


def scene(prims, attributes):
    """(prim name, attribute names) of the benchmark scene: prims /P<i> with double attributes a<j>, as
    edit_benchmark.layout seeds the live layer."""
    return [(f'P{prim}', [f'a{attribute}' for attribute in range(attributes)]) for prim in range(prims)]


def emit(kind, **fields):
    print(json.dumps(dict(type=kind, **fields)), flush=True)


def wait_line(expected):
    for line in sys.stdin:
        if line.strip() == expected:
            return
    raise SystemExit(f'stdin closed before {expected}')


def turn_schedule(args):
    """Yield (turn, due) pairs; due is None for a burst, whose turns are authored back to back."""
    period = 1.0 / args.rate_hz if args.rate_hz else None
    start = time.monotonic()
    for turn in range(args.turns):
        yield turn, None if period is None else start + turn * period


def ovlive_writer(args):
    from pxr import Plug, Sdf
    from omni.usd import live

    Plug.Registry().RegisterPlugins(str(Path(args.package) / 'resources/plugInfo.json'))
    live.initialize()
    endpoint = live.register_endpoint(args.authority, access_token=args.token)
    jobs = queue.SimpleQueue()
    arrivals = threading.Condition()
    dispatched = [0]

    def dispatch(job):
        jobs.put(job)
        with arrivals:
            dispatched[0] += 1
            arrivals.notify_all()
    context = live.set_publication_context(dispatch, max_pending=64)
    uri = live.make_layer_uri(live.make_session_uri(args.authority, args.session), args.layer)
    layer = Sdf.Layer.FindOrOpen(uri)
    assert layer and layer.permissionToEdit, 'live layer did not open for editing'

    def run_jobs(timeout):
        """Run dispatched jobs until `timeout` seconds pass with none queued; a zero timeout drains once."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                job = jobs.get(timeout=max(0.0, deadline - time.monotonic())) if timeout else jobs.get_nowait()
            except queue.Empty:
                return
            job()

    def run_until(predicate, limit):
        deadline = time.monotonic() + limit
        while not predicate():
            if time.monotonic() > deadline:
                return False
            run_jobs(0.001)
        return True

    assert run_until(lambda: live.get_layer_status(uri).state == live.State.READY, 30), 'layer never became ready'
    specs = [layer.GetAttributeAtPath(path) for path in args.attributes.split(',')]
    assert all(specs), 'seeded attribute missing from the opened layer'
    go = threading.Event()
    threading.Thread(target=lambda: (wait_line('go'), go.set()), daemon=True).start()
    emit('ready')
    while not go.is_set():
        run_jobs(0.005)

    edits, samples = [], []
    peak = dict(pending_write_count=0, buffered_edit_count=0)
    blocked_turns = 0

    def sample(turn):
        nonlocal blocked_turns
        status = live.get_layer_status(uri)
        for key in peak:
            peak[key] = max(peak[key], getattr(status, key))
        blocked_turns += status.state == live.State.BLOCKED
        samples.append((turn, time.monotonic(), status.pending_write_count, status.buffered_edit_count,
                        int(status.state)))

    rejected, frame_timeouts = [], 0

    def end_frame(before):
        """A frame ends after the host runs the flush job its turn requested and every other queued job.

        The job reaches the host queue from the runtime's own thread, so the host waits for a dispatch
        after `before`, bounded by --frame-wait-ms, then runs everything queued.
        """
        nonlocal frame_timeouts
        with arrivals:
            if not arrivals.wait_for(lambda: dispatched[0] > before, args.frame_wait_ms / 1000):
                frame_timeouts += 1
        run_jobs(0)

    for turn, due in turn_schedule(args):
        if due is not None:
            while time.monotonic() < due:
                run_jobs(min(0.001, max(0.0, due - time.monotonic())))
        value = float(turn)
        with arrivals:
            before = dispatched[0]
        authored = time.monotonic()
        try:
            with Sdf.ChangeBlock():
                for spec in specs:
                    spec.default = value
        except Exception as error:  # A refused USD call, such as the host edit buffer quota.
            rejected.append((turn, authored, type(error).__name__, str(error)[:200]))
        else:
            edits.append((turn, authored))
        if not args.batch:
            end_frame(before)
        sample(turn)
    if args.batch:
        # The batch is one frame: its jobs run once, after every turn is authored.
        end_frame(0)

    quiet_since = [None]

    def settled():
        # A status can be read before it reports the last turn's record, so the layer must stay idle for
        # a quiet period while the host keeps running dispatched jobs.
        status = live.get_layer_status(uri)
        idle = status.state != live.State.BLOCKED and not status.pending_write_count and not status.buffered_edit_count
        now = time.monotonic()
        quiet_since[0] = (quiet_since[0] or now) if idle else None
        return idle and now - quiet_since[0] >= 0.25
    drained = run_until(lambda: live.get_layer_status(uri).state == live.State.FAILED or settled(),
                        args.drain_timeout_s)
    status = live.get_layer_status(uri)
    drained = drained and status.state != live.State.FAILED
    diagnostics = live.copy_diagnostics(endpoint, max_records=64)
    result = dict(client='ovlive', edits=edits, rejected=rejected, frame_timeouts=frame_timeouts,
                  samples=samples, drained=drained, blocked_turns=blocked_turns,
                  peak=peak, final=dict(state=int(status.state), result=int(status.result),
                                        pending_write_count=status.pending_write_count,
                                        buffered_edit_count=status.buffered_edit_count),
                  delivery_status=int(context.delivery_status),
                  metrics=diagnostics.get('metrics', {}) if isinstance(diagnostics, dict) else {},
                  recent_records=diagnostics.get('recent_records', []) if isinstance(diagnostics, dict) else [])
    Path(args.output).write_text(json.dumps(result))
    emit('done')
    wait_line('exit')
    layer = None
    context.close()


def client_library_start(args):
    import omni.client as client
    client.initialize()
    if args.log_level:
        client.set_log_level(getattr(client.LogLevel, args.log_level.upper()))
    registration = client.register_storage_direct(f'http://{args.authority}',
                                                  headers={'authorization': f'Bearer {args.token}'})
    assert registration and registration.registered, f'storage registration failed: {registration}'
    # The registration lasts while its object lives, so the module keeps it for the process lifetime.
    REGISTRATIONS.append(registration)
    # The first registered provider has live provider key 1.
    return client, f'live://1/{args.session}/{args.layer}'


def client_library_writer(args):
    sys.path.insert(0, str(ROOT / 'tests/reference_service'))
    from scene_graph_wire import Delta, Field, encode
    client, url = client_library_start(args)
    sections = [int(section) for section in args.sections.split(',')]
    lock = threading.Lock()
    state = dict(in_flight=0, failures=[], acks=[])

    def acknowledgment(sent):
        # The bindings take a callback of exactly three positional parameters.
        def acknowledged(result, sequence, _request):
            with lock:
                state['in_flight'] -= 1
                state['acks'].append((sent, time.monotonic(), int(sequence)))
                if result != client.Result.OK:
                    state['failures'].append(str(result))
        return acknowledged

    notices = [0]
    if args.baseline_model == 'usd':
        # omni-spectree's host cost: the host authors USD into the layer, change notices record which specs
        # changed, and each tick builds the delta by reading those specs' current values from the layer.
        # The layer holds the live layer's whole seeded scene before timing starts.
        from pxr import Sdf, Tf
        layer = Sdf.Layer.CreateAnonymous('baseline.usda')
        with Sdf.ChangeBlock():
            for prim_name, attributes in scene(args.scene_prims, args.scene_attributes):
                prim = Sdf.PrimSpec(layer.pseudoRoot, prim_name, Sdf.SpecifierDef)
                for attribute in attributes:
                    Sdf.AttributeSpec(prim, attribute, Sdf.ValueTypeNames.Double).default = -1.0
        specs = [layer.GetAttributeAtPath(path) for path in args.attributes.split(',')]
        assert all(specs), 'edited attribute missing from the baseline scene'
        # The listener lasts while its registration lives, so the module keeps it for the process lifetime.
        REGISTRATIONS.append(Tf.Notice.Register(Sdf.Notice.LayersDidChangeSentPerLayer,
                                                lambda notice, sender: notices.__setitem__(0, notices[0] + 1), layer))

    go = threading.Event()
    threading.Thread(target=lambda: (wait_line('go'), go.set()), daemon=True).start()
    emit('ready')
    go.wait()

    edits, updates, requests = [], [], []
    buffered, buffered_turns = {}, 0
    peak = dict(in_flight=0, buffered_turns=0)

    def author(turn):
        value = float(turn)
        if args.baseline_model == 'usd':
            with Sdf.ChangeBlock():
                for spec in specs:
                    spec.default = value
            buffered.update((index, None) for index in range(len(sections)))
        else:
            buffered.update((index, value) for index in range(len(sections)))

    def delta_fields():
        if args.baseline_model == 'usd':
            return tuple(Field(sections[i], 'default', specs[i].default) for i in sorted(buffered))
        return tuple(Field(sections[i], 'default', v) for i, v in sorted(buffered.items()))

    def tick():
        """omni-spectree SendDelta at a live process tick: send the coalesced buffer if a slot is free."""
        nonlocal buffered, buffered_turns
        with lock:
            free = state['in_flight'] < CLIENT_LIBRARY_WINDOW
            if free and buffered:
                state['in_flight'] += 1
                peak['in_flight'] = max(peak['in_flight'], state['in_flight'])
        if free and buffered:
            payload = encode(Delta(fields=delta_fields()))
            index = len(updates)
            updates.append((time.monotonic(), buffered_turns, len(payload)))
            # Dropping a Python request object stops its request, so every request is kept.
            requests.append(client.live_update_with_callback(url, args.object_id, payload, acknowledgment(index)))
            buffered, buffered_turns = {}, 0
        client.live_process()

    for turn, due in turn_schedule(args):
        if due is not None:
            time.sleep(max(0.0, due - time.monotonic()))
        edits.append((turn, time.monotonic()))
        author(turn)
        buffered_turns += 1
        peak['buffered_turns'] = max(peak['buffered_turns'], buffered_turns)
        if not args.batch:
            tick()
    if args.batch:
        tick()  # The batch is one frame: one live-process tick after every turn is authored.

    period = 1.0 / args.rate_hz if args.rate_hz else 0.001
    deadline = time.monotonic() + args.drain_timeout_s
    while time.monotonic() < deadline:
        with lock:
            idle = not state['in_flight']
        if idle and not buffered:
            break
        time.sleep(period)
        tick()
    with lock:
        result = dict(client='client-library', edits=edits, updates=updates, acks=state['acks'],
                      failures=state['failures'], drained=not state['in_flight'] and not buffered, peak=peak,
                      window=CLIENT_LIBRARY_WINDOW, version=client.get_version(), model=args.baseline_model,
                      change_notices=notices[0])
    Path(args.output).write_text(json.dumps(result))
    emit('done')
    wait_line('exit')
    client.shutdown()


def client_library_reader(args):
    client, url = client_library_start(args)
    lock = threading.Lock()
    received, first = [], threading.Event()

    def on_read(result, _object, sequence, content):
        now = time.monotonic()
        if result != client.Result.OK:
            return
        # Decoding waits until the read ends, so the reader keeps pace with any writer.
        with lock:
            received.append((now, int(sequence), bytes(memoryview(content))))
        first.set()

    request = client.live_read_with_callback(url, 0, 0, on_read, process_immediately=False)
    deadline = time.monotonic() + 30
    while not first.is_set() and time.monotonic() < deadline:
        client.live_process()
        time.sleep(0.001)
    assert first.is_set(), 'client-library read delivered no snapshot'
    stop = threading.Event()
    threading.Thread(target=lambda: (wait_line('stop'), stop.set()), daemon=True).start()
    emit('ready')
    while not stop.is_set():
        client.live_process()
        time.sleep(0.001)
    request.stop()
    with lock:
        for index, (now, sequence, data) in enumerate(received):
            received[index] = (now, sequence, turn_marker(data), len(data))
        Path(args.output).write_text(json.dumps(dict(client='client-library-reader', received=received,
                                                     jitter_reduction='client-library default',
                                                     version=client.get_version())))
    emit('done')
    client.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('role', choices=('ovlive-writer', 'client-library-writer', 'client-library-reader'))
    parser.add_argument('--authority', required=True)
    parser.add_argument('--token', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--layer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--turns', type=int, default=0)
    parser.add_argument('--rate-hz', type=float, default=0, help='turn rate; 0 authors turns back to back')
    parser.add_argument('--drain-timeout-s', type=float, default=120)
    parser.add_argument('--package', help='ovlive plugin package root (ovlive-writer)')
    parser.add_argument('--attributes', help='comma-separated attribute paths (ovlive-writer)')
    parser.add_argument('--sections', help='comma-separated attribute section IDs (client-library-writer)')
    parser.add_argument('--object-id', type=int, default=0, help='layer object ID (client-library-writer)')
    parser.add_argument('--scene-prims', type=int, default=0, help='prims in the seeded scene (client-library-writer)')
    parser.add_argument('--scene-attributes', type=int, default=0,
                        help='attributes per prim in the seeded scene (client-library-writer)')
    parser.add_argument('--log-level', help='Omniverse Client Library log level, such as debug')
    parser.add_argument('--batch', action='store_true', help='author every turn inside one frame (writers)')
    parser.add_argument('--baseline-model', choices=('usd', 'record'), default='usd',
                        help='client-library-writer host cost: USD authoring with a per-tick delta, or values only')
    parser.add_argument('--frame-wait-ms', type=float, default=2.0,
                        help='longest wait at a frame end for the flush job the turn requested (ovlive-writer)')
    args = parser.parse_args()
    roles = {'ovlive-writer': ovlive_writer, 'client-library-writer': client_library_writer,
             'client-library-reader': client_library_reader}
    roles[args.role](args)


if __name__ == '__main__':
    main()
