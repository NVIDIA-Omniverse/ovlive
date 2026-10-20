<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Live Edit conformance tests

This standalone suite tests `nvidia.omniverse.live.v1alpha.LiveService` through public gRPC calls.
It creates UUID-named sessions/layers and deletes only those resources through the API.
It works with NVIDIA Live Edit or another implementation of the documented contract.

## Run

```sh
docker run --rm \
  -e CONFORMANCE_GRPC_ENDPOINT=https://live.example.com:443 \
  -e CONFORMANCE_BASE_PATH=https://storage.example.com/test/base.usda \
  -e CONFORMANCE_AUTH_TOKEN_FILE=/auth/token \
  -e CONFORMANCE_REPORT_STDOUT=true \
  -v "$PWD/token:/auth/token:ro" \
  nvcr.io/<org>/<team>/live-edit-conformance-tests:<release>
```

The image includes bindings, fixtures, dependencies and the native OODF decoder.
Running it requires no source checkout or compiler.
Runtime asset permissions are set explicitly by the image build, independent of checkout file modes. The build verifies the decoder, a loopback identity/authentication fixture check, and the CLI entrypoint as UID/GID 65532.
The base layer must exist and the principal needs read/write permission on it; the suite never deletes that file.
For files instead of encoded log reports, mount a directory writable by UID/GID 65532 at `/reports`.
Any failed test, cleanup failure, missing prerequisite, skipped selected test or empty selection returns nonzero.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CONFORMANCE_GRPC_ENDPOINT` | `http://localhost:50001` | Explicit HTTP/h2c or HTTPS/TLS endpoint. |
| `CONFORMANCE_BASE_PATH` | required | Existing authorized base layer. |
| `CONFORMANCE_AUTH_TOKEN` | unset | Bearer token, mutually exclusive with token file. |
| `CONFORMANCE_AUTH_TOKEN_FILE` | unset | Plain token or JSON with `access_token`; reread per RPC. |
| `CONFORMANCE_CA_FILE` | system roots | Optional CA bundle; HTTPS verification is always enabled. |
| `CONFORMANCE_TIMEOUT` | `15` | Seconds per RPC or complete streamed delivery. |
| `CONFORMANCE_READY_TIMEOUT` | `120` | Readiness/token-file wait in seconds. |
| `CONFORMANCE_DECODER` | `decode_ompf` | Protocol decoder executable; missing decoder is fatal. |
| `CONFORMANCE_REPORT_DIR` | `reports`, `/reports` in image | JUnit XML and JSON summary directory. |
| `CONFORMANCE_REPORT_STDOUT` | unset | `true` includes base64-encoded reports in logs. |
| `CONFORMANCE_PEER_ENDPOINT` | unset | Distinct replica endpoint for the `multi_replica` profile. |
| `CONFORMANCE_DENIED_PATH` | unset | Existing denied layer for the `permissions` profile. |

The runner never loads `.env` files.
It ignores ambient pytest configuration, selection options and third-party plugins.
Readiness checks LiveService health, falling back to an authenticated Ping only if health is unimplemented.
Live Edit still requires a bearer header. Only the isolated local/Compose stack enables `LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS=true`, disables external userinfo and authorization, and supplies the public unsigned `local-test-token.jwt` for in-process display-identity decoding.
There is no identity helper or external auth dependency in those runs. The expected startup ERROR warns that no backend validates tokens; never use this mode outside isolated tests.
The packaged runner has no default token, and neither Helm chart nor GitOps deployment validation enables the override. Deployed-service runs still require real credentials.

## Coverage

Core tests always run; profiles are additive, for example `--profile deployment --profile permissions`.
Missing prerequisites for a selected profile fail the run.

| Contract | Coverage |
|---|---|
| Ping | Timestamp echo and server time. |
| Sessions | CRUD, duplicate names, invalid identifiers, session isolation, deletion cascade. |
| Layers | CRUD, overwrite generations, sequence progression, stale generation rejection. |
| Uploads | Header ordering, malformed OMPF, multiple chunks, empty commits. |
| Operation ids | `UpdateLayer` and `CreateLayer` re-sends answered `ALREADY_COMMITTED` with the original sequence and timestamp or `object_id` and nothing applied, no id answered `UNSPECIFIED`, an id reused for a different request refused `ALREADY_EXISTS`, over-long or out-of-charset ids refused `INVALID_ARGUMENT` on both RPCs, an `UpdateLayer` `operation_id` with an empty `client_id` refused `INVALID_ARGUMENT` (an empty `client_id` without an id still accepted). |
| Chained writes | `Ping` reports `operation_ids` and `chained_writes`; chained `UpdateLayer` calls sent in order, and sent out of order (the later calls observed still waiting), commit in chain order; a re-send of the latest write is answered `ALREADY_COMMITTED` with the original sequence and applies nothing; a waiting call whose predecessor never commits is refused `ABORTED` `PREDECESSOR_NOT_COMMITTED` (`last_record` `operation`, no `RetryInfo`) before the client's own 3 s deadline, which the client times itself, so a refusal sent too late fails as the client's `DEADLINE_EXCEEDED`; a `no_wait` call whose predecessor was passed is refused at once naming the latest write, or `last_record` `absent` under a `client_id` with no record; a cancelled waiting call applies nothing and its re-send applies; a predecessor whose stream never ended, cancelled and re-sent, releases the call waiting behind it; a chained `INVALID_ARGUMENT` carries `ErrorInfo` `INVALID_UPDATE` and an unchained one none, except a size refusal, which carries its own reason (`DELTA_TOO_LARGE`, `CHILDREN_LIST_TOO_LONG`) on a chained call too. |
| Size limits | A delta of more than 99,999 entries (struct commands, set fields and time samples summed; 60,000 + 60,000, and 100,000 set fields alone) refused `INVALID_ARGUMENT` by `CreateLayer` and `UpdateLayer` with `ErrorInfo` `DELTA_TOO_LARGE` (`struct_commands`, `set_fields`, `time_samples`, `max_entries` = 99999, each a decimal integer), no `RetryInfo` and a message saying to split it, nothing applied; exactly 99,999 entries never refused `DELTA_TOO_LARGE`; a `ReorderChildren` list of 100,001 or 1,000,001 ids refused `CHILDREN_LIST_TOO_LONG` (`limit` = 100000, `actual`). The oversized deltas are built by the suite, not shipped as fixtures. |
| Content | Fields, time samples, external-value indices/hashes/sizes, section order, source format, 2 MiB inline value. |
| Subscribe | Full/diff envelope, unique field-order slots, caught-up header, live relay, self-echo suppression, overwrite/deletion, reconnect reconstruction. |
| Structural edits | Create, move, delete and reorder, with independently pinned expected state. |
| Captured traffic | Kit input replay and reconnect fold compared with the pinned C++ reference scene. |
| Presence | Connected, peer join, message and leave. |
| Presence keepalive | An opted-in join is told a non-zero `keepalive_interval_ms` and any other join 0; keepalives are never delivered to peers; a silent opted-in join ends `UNAVAILABLE` no sooner than three intervals and at most 10 s later, and its peer sees one `PeerLeft`. The silent case waits three intervals, about 50 s at the default 15 s. A deployment with the keepalive off (0) fails both cases. One with an interval above about 35 s fails the silent case: three intervals outlast the join call's own 110 s gRPC deadline, so the call ends `DEADLINE_EXCEEDED` rather than the asserted `UNAVAILABLE`. Live Edit refuses intervals of 1–999 ms at startup, so the shortest interval a run can use is 1000 ms. |
| Bearer contract | Missing bearer rejection and requests carrying an explicit token. Isolated CI does not validate signatures, issuers, audiences, expiry, or real permissions. |
| `deployment` | Unauthenticated LiveService health. |
| `permissions` | PERMISSION_DENIED on a configured denied base path. |
| `multi_replica` | Updates delivered across distinct replica endpoints; chained writes sent out of order across two replica endpoints commit in chain order. |

This does not certify every error combination, discovery, forced history trimming, performance, checkpoint durability, storage internals or identity-provider implementation.
The captured traffic has no time samples; separate synthetic cases cover those.
The decoder uses only `oodf-ffi`; expected state is not computed by the service's domain/diff engine.
Negative controls check that dropped structure, fields, samples and retractions break the comparison.

## Source development

```sh
uv sync --project conformance_tests --frozen
cargo build --locked -p oodf-ffi --example decode_ompf
CONFORMANCE_DECODER="$PWD/target/debug/examples/decode_ompf" \
  uv run --project conformance_tests run-conformance-tests
```

In an API archive, build the helper with `cargo build --locked --release --manifest-path native-codec/Cargo.toml -p oodf-ffi --example decode_ompf`.
Set `CONFORMANCE_DECODER` to its `native-codec/target/release/examples/decode_ompf` output.
The native build requires Rust 1.94.1, a C++17 compiler and FlatBuffers 24.3.25 (`flatc` and headers installed under the same prefix).
On Linux, `python3 scripts/test-conformance-native.py` (from this repository) stages an isolated API archive, builds its release decoder with GNU ld, checks runtime dependencies, and decodes every synthetic fixture. It leaves the scratch build path in its output for inspection. The container build also exercises GNU linking and runs a decoder smoke check in the final non-root image.

From this repository, test a production binary using `TEST_VALKEY_URL`, a temporary local-disk checkpoint directory and the bundled unsigned test JWT:

```sh
cargo build --locked -p live-edit
uv run --project conformance_tests python scripts/run-conformance-local.py
```

The launcher does not change Valkey configuration or flush its contents.
The local service listener binds all interfaces, even though the runner connects over loopback; run it only in an isolated development environment and do not expose its ports to untrusted clients.
Add `--replicas 2` to exercise the cross-replica profile against two production processes.
Container CI uses its own Valkey with AOF enabled:

```sh
docker build -t live-edit-service:local .
docker build -f conformance_tests/Dockerfile -t live-edit-conformance-tests:local .
SERVICE_IMAGE=live-edit-service:local CONFORMANCE_IMAGE=live-edit-conformance-tests:local \
  bash scripts/run-conformance-container.sh
```

Rebuild the service image too: images predating the test-only override reject this configuration at startup.

Regenerate assets with `uv run --project conformance_tests python scripts/regenerate-conformance-assets.py`.
`build.json` records proto/schema/fixture hashes independently of release tags.
`native-Cargo.lock` locks the independent codec workspace. When changing its dependencies, stage an API archive, update that workspace's lockfile, copy it back and verify the extracted build with `--locked`.
Run harness tests with `uv run --project conformance_tests pytest conformance_tests/unit_tests`.

## Kubernetes and releases

The separate `helm/live-edit-conformance-tests` chart creates a bounded Job without retries.
Configure the endpoint, base path and bearer Secret or token-file sidecar.
Use a fresh release name per run, wait with `--wait --wait-for-jobs`, and collect the encoded reports from the `conformance` container log before cleanup.

The optional `python -m live_edit_conformance.deployment` adapter provisions a UUID-named file through Storage API when `CONFORMANCE_STORAGE_ENDPOINT` is set.
In this mode the base path is an authorized directory; only that newly created file is deleted afterward.
Alternative implementations do not need this adapter.

CI tests immutable image digests and gates RC/DEV-MR/stable/external publication on conformance success.
The service build and conformance run share an explicit pipeline-specific image tag; the run resolves its digest and records `tested-service.env` only after successful execution.
Images and charts publish to the same dev/staging NGC teams as the service; external distribution follows the service's staging-to-public release process.
Test images/charts use matching service release tags; the API archive retains its independent version scheme and includes source and native helper build inputs.
In-cluster execution runs through your GitOps process; do not install or modify live cluster Jobs imperatively.
