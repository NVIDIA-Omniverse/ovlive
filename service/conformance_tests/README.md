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

The image is a single executable, `/app/live-edit-conformance`, with its fixtures and OODF decoder built in, on the same NVIDIA distroless base as the service image.
Running it requires no source checkout or compiler, and the image has no shell.
The base layer must exist and the principal needs read/write permission on it; the suite never deletes that file.
The image runs as UID/GID 65532, and its `/reports` directory is writable by that user.
For report files instead of encoded log lines, mount a directory writable by UID/GID 65532 at `/reports`.

Arguments: `--profile <name>` adds a profile's cases and may be repeated (see Coverage); `--help` prints the usage, every variable and the exit codes.
Any other argument stops the run with exit code 2.

| Exit code | Meaning |
|---|---|
| `0` | Every selected case passed. |
| `1` | A case failed, or its setup or cleanup did, or the reports could not be written. |
| `2` | The arguments, the preflight (configuration, token file, profile prerequisites, fixture check, readiness), base-layer provisioning or its cleanup failed, or the run was stopped before or during its cases. |

The image carries its third-party notices in `/licenses/` and the source of the third-party code built into it in `/vendor.tar.gz`; that code is a subset of the service image's.
To check them, run `sh tools/check-container-oss.sh <image> live-edit-conformance` from this repository.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CONFORMANCE_GRPC_ENDPOINT` | `http://localhost:50001` | Explicit HTTP/h2c or HTTPS/TLS endpoint. |
| `CONFORMANCE_BASE_PATH` | required | Existing authorized base layer; with `CONFORMANCE_STORAGE_ENDPOINT`, a writable directory. |
| `CONFORMANCE_AUTH_TOKEN` | one of the two required | Bearer token; set exactly one of this and the token file. |
| `CONFORMANCE_AUTH_TOKEN_FILE` | one of the two required | Plain token or JSON with `access_token`; reread per RPC. Set exactly one of this and the token. |
| `CONFORMANCE_CA_FILE` | unset | PEM CA bundle that replaces the default roots of every TLS connection; HTTPS verification is always enabled. Without it, gRPC endpoints trust the image's system certificate store and the Storage API a built-in Mozilla CA list. Refused when none of the gRPC, peer and Storage API endpoints uses `https://`. |
| `CONFORMANCE_TIMEOUT` | `15` | Seconds per RPC or complete streamed delivery; at most 86400 (one day). |
| `CONFORMANCE_READY_TIMEOUT` | `120` | Readiness/token-file wait in seconds; at most 86400 (one day). |
| `CONFORMANCE_STORAGE_ENDPOINT` | unset | Storage API URL. When set, the run creates its own base layer under `CONFORMANCE_BASE_PATH` and deletes it afterward (see Kubernetes and releases). |
| `CONFORMANCE_REPORT_DIR` | `reports`, `/reports` in image | JUnit XML and JSON summary directory. |
| `CONFORMANCE_REPORT_STDOUT` | unset | `true` includes base64-encoded reports in logs. |
| `CONFORMANCE_PEER_ENDPOINT` | unset | Distinct replica endpoint for the `multi_replica` profile. |
| `CONFORMANCE_DENIED_PATH` | unset | Existing denied layer for the `permissions` profile. |

A variable that is empty or only whitespace counts as unset and takes its default, except `CONFORMANCE_GRPC_ENDPOINT`, `CONFORMANCE_TIMEOUT` and `CONFORMANCE_READY_TIMEOUT`.
Set but empty or only whitespace, one of those three stops the run with exit code 2 and `<NAME> is set but empty; unset it to use the default`.
So the chart's `service.endpoint` set to `""` stops the run before any case.
A timeout above 86400 stops the run the same way, with `<NAME> must be at most 86400 seconds (one day)`.
With `CONFORMANCE_STORAGE_ENDPOINT` set, a refused configuration is reported as `Provisioning or cleanup failed (Config): <message>`, such as `Provisioning or cleanup failed (Config): CONFORMANCE_TIMEOUT must be finite and positive`.

The runner never loads `.env` files.
Readiness checks LiveService health, falling back to an authenticated Ping only if health is unimplemented.
While an endpoint is unreachable, not yet serving or not answering, the health check is retried until `CONFORMANCE_READY_TIMEOUT`.
So an endpoint that accepts connections but never answers fails the readiness check only after that whole wait, 120 s by default.
Live Edit still requires a bearer header. Only the isolated local/Compose stack enables `LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS=true`, disables external userinfo and authorization, and supplies the public unsigned `local-test-token.jwt` for in-process display-identity decoding.
There is no identity helper or external auth dependency in those runs. The expected startup ERROR warns that no backend validates tokens; never use this mode outside isolated tests.
The packaged runner has no default token, and neither Helm chart nor GitOps deployment validation enables the override. Deployed-service runs still require real credentials.

## Reports

Each run writes `conformance-results.xml` (JUnit) to `CONFORMANCE_REPORT_DIR`.
It also writes `summary.json` there, unless the run ends at a preflight failure.
A preflight failure leaves in place any `summary.json` an earlier run wrote to that directory, which can show that run's `exit_code`, such as `0`.
So mount an empty report directory for each run, or treat a JUnit file holding a `conformance-preflight` suite as the run's result.
With `CONFORMANCE_REPORT_STDOUT=true` it also prints `conformance-results.xml`, and `summary.json` whenever the directory holds one, an earlier run's included, each as one `CONFORMANCE_REPORT <name> <base64>` line.

- The JUnit suite `live-edit-conformance` holds one test case per selected case, named `<module>::<test>[<parameter>]`, such as `layers::test_content_round_trip[large]`, with the module as its class name.
- A case that runs longer than 120 s fails, and the run goes on with the next case.
- A run stopped during its cases records an error on the case it was running, and the remaining cases do not run.
- A failure before the cases run, a stop included, is the JUnit suite `conformance-preflight`, and no `summary.json` is written for it.
  With `CONFORMANCE_STORAGE_ENDPOINT` set, a refused configuration, a failure or stop before the base layer is created, and a failure to create or delete it are the suite `deployment-fixture` instead.
  Its error names the kind of failure, as in `Provisioning or cleanup failed (Transport)`, and only a refused configuration (`Config`) adds its message, which never includes a value.
  A `deployment-fixture` failure always writes `summary.json`, even after a preflight failure.
- `summary.json` holds `profiles`, `executed`, `skipped` (always 0), `unexpected_deselections` (always 0), `exit_code` and `build`.
  `build` names the contract and records the SHA-256 of the protocol file, the OODF schema, the captured traffic and its expected result (`proto`, `oodf_schema`, `captured`, `captured_golden`).
  `failure` names the JUnit suite holding the error that set exit code 2: `live-edit-conformance` for a run stopped during its cases, or `deployment-fixture`, which takes precedence.
  A `summary.json` for a run that never reached its cases holds only `exit_code` and `failure`.

## Coverage

Core tests always run; profiles are additive, for example `--profile deployment --profile permissions`.
Missing prerequisites for a selected profile fail the run.

| Contract | Coverage |
|---|---|
| Ping | Timestamp echo and server time. |
| Sessions | CRUD, duplicate names, invalid identifiers, session isolation, deletion cascade. |
| Layers | CRUD, overwrite generations, sequence progression, stale generation rejection. |
| Uploads | Header ordering, malformed OMPF, multiple chunks, empty commits, a header-only `CreateLayer` holding exactly its root section, an `UpdateLayer` removing a layer's last root section refused `INVALID_ARGUMENT` with nothing applied. |
| Operation ids | `UpdateLayer` and `CreateLayer` re-sends answered `ALREADY_COMMITTED` with the original sequence and timestamp or `object_id` and nothing applied, no id answered `UNSPECIFIED`, an id reused for a different request refused `ALREADY_EXISTS`, over-long or out-of-charset ids refused `INVALID_ARGUMENT` on both RPCs, an `UpdateLayer` `operation_id` with an empty `client_id` refused `INVALID_ARGUMENT` (an empty `client_id` without an id still accepted). |
| Chained writes | `Ping` reports `operation_ids` and `chained_writes`; chained `UpdateLayer` calls sent in order, and sent out of order (the later calls observed still waiting), commit in chain order; a re-send of the latest write is answered `ALREADY_COMMITTED` with the original sequence and applies nothing; a waiting call whose predecessor never commits is refused `ABORTED` `PREDECESSOR_NOT_COMMITTED` (`last_record` `operation`, no `RetryInfo`) before the client's own 3 s deadline, which the client times itself, so a refusal sent too late fails as the client's `DEADLINE_EXCEEDED`; a `no_wait` call whose predecessor was passed is refused at once naming the latest write, or `last_record` `absent` under a `client_id` with no record; a cancelled waiting call applies nothing and its re-send applies; a predecessor whose stream never ended, cancelled and re-sent, releases the call waiting behind it; a chained `INVALID_ARGUMENT` carries `ErrorInfo` `INVALID_UPDATE` and an unchained one none, except a size refusal, which carries its own reason (`DELTA_TOO_LARGE`, `CHILDREN_LIST_TOO_LONG`) on a chained call too. |
| Size limits | A delta of more than 99,999 entries (struct commands, set fields and time samples summed; 60,000 + 60,000, and 100,000 set fields alone) refused `INVALID_ARGUMENT` by `CreateLayer` and `UpdateLayer` with `ErrorInfo` `DELTA_TOO_LARGE` (`struct_commands`, `set_fields`, `time_samples`, `max_entries` = 99999, each a decimal integer), no `RetryInfo` and a message saying to split it, nothing applied; exactly 99,999 entries never refused `DELTA_TOO_LARGE`; a `ReorderChildren` list of 100,001 or 1,000,001 ids refused `CHILDREN_LIST_TOO_LONG` (`limit` = 100000, `actual`). The oversized deltas are built by the suite, not shipped as fixtures. |
| Content | Fields, time samples, external-value indices/hashes/sizes, section order, source format, 2 MiB inline value. |
| Subscribe | Full/diff envelope, unique field-order slots, caught-up header, live relay, self-echo suppression, overwrite/deletion, reconnect reconstruction. |
| Structural edits | Create, move, delete and reorder, and the sibling-name rules (a same-kind rename deletes the sibling it lands on, a prim and a property can share a name, an attribute and a relationship cannot), with independently pinned expected state. |
| Captured traffic | Kit input replay and reconnect fold compared with the pinned C++ reference scene. |
| Presence | Connected, peer join, message and leave. |
| Presence keepalive | An opted-in join is told a non-zero `keepalive_interval_ms` and any other join 0; keepalives are never delivered to peers; a silent opted-in join ends `UNAVAILABLE` no sooner than three intervals and at most 10 s later, and its peer sees one `PeerLeft`. The silent case waits three intervals, about 50 s at the default 15 s. A deployment with the keepalive off (0) fails both cases. One with an interval above about 35 s fails the silent case: three intervals outlast the join call's own 110 s gRPC deadline, so the call ends `DEADLINE_EXCEEDED` rather than the asserted `UNAVAILABLE`. Live Edit refuses intervals of 1–999 ms at startup, so the shortest interval a run can use is 1000 ms. |
| Bearer contract | Missing bearer rejection and requests carrying an explicit token. Isolated CI does not validate signatures, issuers, audiences, expiry, or real permissions. |
| `deployment` | Unauthenticated LiveService health. |
| `permissions` | PERMISSION_DENIED on a configured denied base path. |
| `multi_replica` | Updates delivered across distinct replica endpoints; chained writes sent out of order across two replica endpoints commit in chain order. |

This does not certify every error combination, discovery, forced history trimming, performance, checkpoint durability, storage internals or identity-provider implementation.
The captured traffic has no time samples; separate synthetic cases cover those.
Expected state is computed independently of the service's own code.
Negative controls check that dropped structure, fields, samples and retractions break the comparison.

## Source development

From this repository, test a production binary using `TEST_VALKEY_URL`, a temporary local-disk checkpoint directory and the bundled unsigned test JWT:

```sh
cargo build --locked -p live-edit -p live-edit-conformance
python3 scripts/run-conformance-local.py
```

The launcher starts `target/debug/live-edit` and runs `target/debug/live-edit-conformance --profile deployment` against it; `--binary` and `--suite` name other builds.
The launcher does not change Valkey configuration or flush its contents.
The local service listener binds all interfaces, even though the runner connects over loopback; run it only in an isolated development environment and do not expose its ports to untrusted clients.
Add `--replicas 2` to exercise the cross-replica profile against two production processes.
The container stack runs its own Valkey with AOF enabled:

```sh
docker build -t live-edit-service:local .
docker build -f Dockerfile.conformance -t live-edit-conformance-tests:local .
SERVICE_IMAGE=live-edit-service:local CONFORMANCE_IMAGE=live-edit-conformance-tests:local \
  bash scripts/run-conformance-container.sh
```

Rebuild the service image too: images predating the test-only override reject this configuration at startup.

The synthetic fixtures are built when the suite runs and checked against pinned SHA-256 hashes before any case starts.
`fixtures/` holds the captured Kit traffic, its expected C++ reference result, and `build.json`, which records the protocol, schema and capture hashes independently of release tags.
Run the suite's own tests with `cargo test -p live-edit-conformance`.

## Kubernetes and releases

The separate `helm/live-edit-conformance-tests` chart creates a bounded Job without retries, which runs `/app/live-edit-conformance` from the conformance-test image.
Configure the endpoint, base path and bearer Secret or token-file sidecar.
Use a fresh release name per run, wait with `--wait --wait-for-jobs`, and collect the encoded reports from the `conformance` container log before cleanup.

Use the chart and the image of the same release: the chart's image tag defaults to its own version.
A chart and an image from different releases can run different commands.
When the commands differ, as with an `image.tag` naming another release, the Job fails without running any case: the container cannot start, and the Job does not retry, so such a run never reports a pass.

When `CONFORMANCE_STORAGE_ENDPOINT` is set (the chart's `service.storageEndpoint`), the suite provisions a UUID-named base layer, `conformance-<uuid>.usda`, through the Storage API before the cases run.
In this mode the base path is an authorized directory; only that newly created file is deleted afterward.
Alternative implementations do not need this.

When the Job is stopped, for example at `activeDeadlineSeconds`, the suite spends up to 10 s closing the running case's sessions, deletes any provisioned base layer, logs its reports and exits 2.
Once the stop has arrived, the deletion waits at most 10 s for its answer, so the reports are logged within the 30 s Kubernetes allows the pod after the stop, whatever `timeouts.request` is.
A deletion with no answer by then fails the run as `deployment-fixture` (`Provisioning or cleanup failed (Transport)`) with exit code 2, and can leave the `conformance-<uuid>.usda` file behind; delete it through the Storage API.
A stop that arrives during the deletion, after every case has passed, limits the deletion's wait to 10 s in the same way.
If the deletion is answered in time, the run keeps its cases' exit code, 0 when every case passed.

Test images and charts use the service's release tags; the API archive keeps its own version scheme and carries the protocol file, not the suite.
