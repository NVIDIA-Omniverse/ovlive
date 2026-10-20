<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Samples

The package ships three samples: a Python and a native participant that create a session and a live layer, edit it with plain USD, save, recover and leave, and a read-only diagnostics sample that lists sessions and prints the runtime's diagnostic records. This guide is for engineers and agents who want a working end-to-end example of the Python API, the C ABI or the diagnostic output, and for operators who drive a participant by hand.

All three run from the staged package; set up the library and Python paths as the [build guide](ovlive-build.md#using-the-package) describes. Each needs a Live Services endpoint (`HOST:PORT`) and an access token the service accepts. The participants read the token from `LIVE_SAMPLE_ACCESS_TOKEN`; the diagnostics sample reads `LIVE_SERVICES_TOKEN` unless `--token-env` names another variable. The [test suites](ovlive-testing.md) start the repository's independent reference service on loopback and drive the samples against it: `plugin-reference` runs the native participant and diagnostics samples, and `python-reference` runs the Python participant. The reference service has no standalone server command. The [support boundary](ovlive-support-boundary.md) states the policy for service-backed testing.

Each sample accepts `--ca-bundle PEM_FILE`, `--tls-server-name DNS_NAME` and `--transport-security default|tls|plaintext`. The CA file supplies PEM trust roots and is limited to 1 MiB; the TLS server name identifies a certificate's DNS name when the authority is an IP literal. The default transport uses plaintext for loopback authorities and TLS with platform trust roots otherwise; `tls` and `plaintext` override that selection ([endpoint options](ovlive-reference.md#endpoint-options)). For a private CA behind a TLS proxy on an IP literal, pass all three options with `--transport-security tls`. Plaintext to a non-loopback authority requires `--transport-security plaintext`; use it only when the service or a trusted network provides the intended protection.

## Participant samples

The participants create a fresh session and a `sample.live` layer, list the session's creator and time metadata, join presence, edit `/Example.value`, refresh credentials from a file on request, record status and diagnostic events, export a failed layer and leave. Each runs a serial USD host thread that progresses on its own, so no frame pump is needed.

### Running the Python participant

`omni.usd.live.sample` uses the public Python API and registers the package's `resources/plugInfo.json` itself. Supply the token in `LIVE_SAMPLE_ACCESS_TOKEN`; the sample removes it from its environment after reading it.

```sh
python3.12 -B -m omni.usd.live.sample --authority localhost:50051 --output live-participant
```

`--output` names a new directory for its logs, snapshots and exports. `--help` prints usage.

For example, add `--ca-bundle private-ca.pem --tls-server-name live.example.test --transport-security tls` when connecting through a TLS proxy at an IP-literal authority.

### Running the native participant

`live_participant_sample` uses the public C ABI and the C++ USD adapter in `LiveUsd.h`; C++ is required for ordinary `SdfLayer` authoring. It links the shared plugin and the matching USD SDK and embeds no second runtime. It takes the same token variable and options plus the plugin registration:

```sh
live_participant_sample --authority localhost:50051 --output native-participant \
  --plug-info <target-dir>/usd/omniverse/live/resources/plugInfo.json
```

### Commands

Standard input takes one JSON object per line with an `operation` field; standard output prints `{"type":"ready"}` and then one `reply` object per command. Both samples accept the same commands and field names:

| Command | Arguments | Reply fields besides `result` |
| --- | --- | --- |
| `create` | `base_layer_key`, optional `name` | `session` (created ID and URI) and `metadata` (listed sessions with creator and time), separately; `export_required` if the layer did not activate |
| `edit` | `value`, a 32-bit integer | none; authors `/Example.value` on the host with plain USD |
| `inspect` | none | `value` read from the live layer on the host |
| `save` | none | `status`; when the layer has failed, the failed-layer result code, with `export_required` in the Python sample |
| `status` | none | `status` (the layer's operational status), `exported` |
| `wait-status` | `state`, a numeric layer state (1–5 or 7–9) | `status` once the layer reports that state; times out after 30 seconds |
| `refresh` | `token_file`, a file holding the replacement token | `generation` of the new credential |
| `snapshot` | optional `name`, a file name in the output directory | writes the diagnostic snapshot (default `diagnostics.json`) |
| `export` | `path` of a new file | `SdfLayer::Export` of the live layer, or of the seed when the layer never activated |
| `resume` | none | `export_required` when the creation failed; continues a timed-out layer creation with the same owner |
| `session-outcome` | none | `session`; waits on a timed-out session creation, and the native sample adds the request's numeric error, attempt and gRPC status |
| `close` | none | `closed` and `logging` (bytes written, records dropped, error class); when work still needs export, the failed-layer result code, with `retained` in the Python sample |

For example:

```json
{"operation":"create","base_layer_key":"headless-demo","name":"Demo"}
{"operation":"edit","value":42}
{"operation":"save"}
{"operation":"inspect"}
{"operation":"refresh","token_file":"/private/new-access-token"}
{"operation":"snapshot"}
{"operation":"close"}
```

`result` is a numeric result code from the [reference](ovlive-reference.md#result-codes), including `OUTCOME_UNKNOWN`. When a command fails outside the live API, the Python sample replies -1 with only the exception class name in `error`, and the native sample replies `INTERNAL_ERROR`. Replies never echo credentials, command input or server text. The native sample rejects command lines over 64 KiB.

### Saving, closing and credentials

`save` reports that the captured local work has retired; it does not mean that every peer has consumed it. An off-host save returns `BUSY` while host edits are buffered, and the sample retries it after the status callback reports them admitted. A save is a drain, not a mutation, so a timed-out save may be repeated. `close` leaves the session and releases the local owners; the created session and layer stay on the service for other participants. The samples never delete or replace an existing session.

The token file is read only on an explicit `refresh`; the new token replaces the current credential generation. Token acquisition and renewal policy belong to the host: the samples log auth challenges so an operator or controlling agent can supply a replacement, and implement no login.

`events.jsonl` in the output directory records structured diagnostics, operational status, auth challenges and session connection events; peer message bodies are omitted. Log copies go through a separate writer with a 1,024-record, 1 MiB pending limit and an 8 MiB file limit. A full queue, an oversized record or the file limit drops the log copy; status delivery continues. The `close` reply reports what was written and dropped. Snapshots are separate from this budget and never replace an existing file.

### Recovery

A local wait timeout does not show whether the service finished. After an uncertain `create`, inspect the reply and `status`; do not repeat the create:

- After a session-creation timeout, `session-outcome` waits on the same retained request and records its eventual identity or error. The sample never issues a second session create; start another participant only after reconciling the first outcome.
- After a layer creation timeout, `resume` continues the same owner without restarting the layer creation.
- After a terminal failure, export the layer to a new file, then close:

```json
{"operation":"export","path":"live-participant/failed-layer.usda"}
{"operation":"close"}
```

`close` refuses to discard retained work. `export` refuses an existing destination with `ALREADY_EXISTS` and leaves it unchanged, so a retry never replaces an earlier export. If export fails, the sample keeps its creation, layer and host alive and accepts another export to a new path.

End of input, a lost standard output or an interrupted command first attempts a safe close. If work needs export, the sample writes `eof-export.usda` in the output directory. If preservation or cleanup fails, it prints a `retained` reply naming the directory and stays alive; fix the local storage problem and send `SIGUSR1` to retry. No mutation is retried. Killing the process ends its in-memory ownership, as with any client, so prefer explicit save, export and close. The native sample does not treat `SIGINT` as a shutdown request: send `close` or end the input and wait. An unread standard output pipe can block replies while the host keeps working.

## Diagnostics sample

`live_diagnostics_sample` is a read-only consumer of the public C ABI in `Live.h`. It registers an endpoint and diagnostic and status observers, performs one session-list query, prints the result and cleans up. It does not create or join a session, author USD, refresh credentials or export a layer. The executable is `<target-dir>/usd/omniverse/live/bin/live_diagnostics_sample`; the [event catalog](ovlive-diagnostic-catalog.md) defines the events it prints and the [snapshot schema](ovlive-diagnostic-schema.md) the snapshot it writes.

### Running the diagnostics sample

Put the token in `LIVE_SERVICES_TOKEN` (never on the command line), set `LIVE_STAGE` to `<target-dir>/usd/omniverse/live`, then run:

```sh
"$LIVE_STAGE/bin/live_diagnostics_sample" \
  --authority "$LIVE_AUTHORITY" \
  --base-layer-key "$LIVE_BASE_LAYER_KEY" \
  --format jsonl --debug --snapshot --timeout-ms 5000
```

| Option | Meaning |
| --- | --- |
| `--authority HOST:PORT` | Required service authority |
| `--base-layer-key KEY` | Required key for the session-list query |
| `--token-env NAME` | Environment variable holding the token; default `LIVE_SERVICES_TOKEN` |
| `--ca-bundle PEM_FILE` | PEM trust roots file, at most 1 MiB |
| `--tls-server-name DNS_NAME` | Certificate DNS name for an IP-literal TLS authority |
| `--transport-security default\|tls\|plaintext` | Endpoint transport override; default `default` |
| `--format jsonl\|text` | Record format; default `jsonl` |
| `--debug` | Include DEBUG records, such as successful RPC diagnostics; the default minimum severity is INFO |
| `--snapshot` | Append a diagnostic JSON snapshot after the diagnostic callback is unregistered |
| `--timeout-ms N` | Operation, diagnostic and cleanup budget, 1 to 60000 milliseconds; default 5000 |
| `--help` | Print usage; must be used alone |

Unknown or duplicate options, missing values and an absent or empty token variable fail with fixed messages that never echo arguments or the token.

### Output

Diagnostic records go to standard output. The query result and count, typed endpoint status, status callback counts, sink result and cleanup result go to standard error. The query outcome is separate from diagnostic delivery: a log failure does not change whether the query succeeded. The callback output is a delivered subset, not a complete history: the sample unregisters its sink after at least one record and one status notification, and the runtime bounds its queues and suppresses repeated warnings. Never infer a missing failure from a missing line.

`--snapshot` copies at most 128 recent records and 256 KiB of output after the callback is quiescent, so the JSON document never interleaves with the records. With `--format text --snapshot`, standard output holds text records followed by one JSON document.

Each record and the snapshot are flushed as they are written, and the sample flushes both streams before it returns, so no output waits for the C runtime's flush at process exit. On Windows that flush runs in the C runtime's own process detach, after `ExitProcess` has ended every other thread, and the process can end before it with its exit code: when another thread, such as one of gRPC's after shutdown, is still exiting, or when a library's detach enters a lock that a terminated thread held. The native participant writes its replies directly to the standard output descriptor and flushes its usage text.

### Record encoding

Each record is one line of at most 65,536 bytes including its newline. Event names and field keys are at most 64 bytes, string values at most 512 bytes, and a record has at most 16 fields; invalid or oversized records are rejected, never truncated.

JSON records use snake_case keys for the schema, event, severity, timestamps, runtime, sequence, endpoint, subject and incident IDs and the field count. `fields` maps each name to `{"kind": KIND, "value": VALUE}`, where kinds 1, 2, 3 and 4 are UINT64, INT64, STRING and BOOLEAN. Numbers are exact decimal integers, so parse them with an integer-capable JSON parser. Text records print a severity label, the quoted event name, the correlation values and each field as `"key"[kind]=value`.

Both formats declare `text_encoding=escaped_bytes`: quotes and backslashes are escaped and control or high bytes appear as `\u00XX`. After JSON decoding, code points 0–255 are the original bytes; decode them as UTF-8 only if the source is known to be UTF-8. Embedded newlines or terminal control bytes cannot inject output lines.

### Exit codes

A watchdog bounds the whole run at `3 * timeout_ms + 1000` milliseconds, including a stalled output reader. If cleanup cannot confirm that callbacks have stopped, the sample exits with code 3 rather than freeing live captures; a real host keeps its owners and retries instead.

| Exit code | Meaning |
| --- | --- |
| 0 | The query, output and cleanup succeeded |
| 1 | The query, diagnostic or snapshot output, or the final flush failed |
| 2 | Invalid arguments or an initialization failure |
| 3 | Cleanup could not stop the callbacks, or the watchdog expired |
