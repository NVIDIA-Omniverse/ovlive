# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify USD creation ownership and deferred work through explicit hook events."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


CASES = {
    'create-deferred-work': ('queued-anonymous-work', 'outer-create-returned', 'before-deferred-anonymous', 'after-deferred-anonymous'),
    'create-deferred-save-notice': ('save-notice-queued', 'queued-anonymous-work', 'outer-create-returned', 'before-deferred-anonymous', 'after-deferred-anonymous'),
    'create-release-failure': ('outer-create-returned', 'failed-target-destroyed', 'before-find-after-failure', 'after-find-after-failure'),
    'create-freeze-failure': ('after-initial-data-swap', 'outer-create-returned', 'failed-target-destroyed', 'frozen-backing-survived-target', 'before-find-after-failure', 'after-find-after-failure'),
    'create-install': ('before-initial-data-swap', 'after-initial-data-swap', 'outer-create-returned'),
    'export-nested': ('before-nested-anonymous', 'after-nested-anonymous'),
}
MAX_OUTPUT = 1024 * 1024


def supervise(command, environment):
    started = time.monotonic()
    deadline = started + 10  # Failure watchdog only; no passing test waits for it.
    markers, output, pending = [], bytearray(), bytearray()
    timed_out, output_limited = False, False
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               bufsize=0, env=environment)
    # A reader thread hands the output over a queue: select() waits on pipes only on POSIX.
    chunks = queue.Queue()
    def read():
        while chunk := os.read(process.stdout.fileno(), 65536):
            chunks.put(chunk)
        chunks.put(b'')
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                chunk = chunks.get(timeout=remaining)
            except queue.Empty:
                continue
            if not chunk:
                break
            if len(output) + len(chunk) > MAX_OUTPUT:
                output.extend(chunk[:MAX_OUTPUT - len(output)])
                output_limited = True
                break
            output.extend(chunk)
            pending.extend(chunk)
            while b'\n' in pending:
                line, _, pending = pending.partition(b'\n')
                if line.startswith(b'{'):
                    value = json.loads(line)
                    if 'marker' in value:
                        markers.append(value['marker'])
        observed = time.monotonic()
        if timed_out or output_limited:
            process.kill()
        process.wait(timeout=5)
        reader.join(timeout=5)
        tail = bytearray()
        while not chunks.empty():
            tail.extend(chunks.get_nowait())
        if len(output) + len(tail) > MAX_OUTPUT:
            output_limited = True
        output.extend(tail[:max(0, MAX_OUTPUT - len(output))])
        result = dict(returncode=process.returncode, timed_out=timed_out, output_limited=output_limited,
                      markers=markers, elapsed_ms=int((observed - started) * 1000))
        return bytes(output), result
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()


def run(driver, plugin, directory):
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, boundary='usd_creation_hook_contract', usd_version=2511,
                  service=False, production_plugin=False, passed=False, cases=[])
    environment = dict(os.environ, PXR_PLUGINPATH_NAME='', PXR_AR_DEFAULT_SEARCH_PATH='', AR_DEFAULT_RESOLVER='')
    try:
        for name, required_markers in CASES.items():
            output, result = supervise([str(driver), str(plugin), str(directory / (name + '.trace.json')), name],
                                       environment)
            (directory / (name + '.log')).write_bytes(output)
            result.update(name=name, passed=False)
            report['cases'].append(result)
            markers = result['markers']
            assert not result['output_limited'], (name, 'child output limit reached')
            assert 'anonymous-control-passed' in markers, (name, 'startup/control did not complete')
            assert 'content-notice-control-passed' in markers, (name, 'content notice control did not complete')
            assert not result['timed_out'] and result['returncode'] == 0 and 'completed' in markers, (name, result)
            positions = [markers.index(marker) for marker in required_markers]
            assert positions == sorted(set(positions)), (name, 'hook events out of order', result)
            trace = json.loads((directory / (name + '.trace.json')).read_text())
            assert trace['passed'] and trace['steps'][-1]['name'] == name, (name, trace)
            result['hooks'] = [event['hook'] for event in trace['steps'][-1]['events']]
            assert ('WriteToFile' if name == 'export-nested' else 'SaveToFile') in result['hooks'], (name, trace)
            result['passed'] = True
        report['passed'] = True
    finally:
        (directory / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('driver', type=Path)
    parser.add_argument('plugin', type=Path)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--with-baseline', action='store_true')
    args = parser.parse_args()
    driver, plugin, directory = args.driver.resolve(), args.plugin.resolve(), args.directory.resolve()
    if args.with_baseline:
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(driver), str(plugin), str(directory / 'trace.json')], check=True, timeout=30)
    run(driver, plugin, directory / 'creation' if args.with_baseline else directory)


if __name__ == '__main__':
    main()
