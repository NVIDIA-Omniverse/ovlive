<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ovlive reference

The macOS arm64 stock USD source build uses the same public C and Python interfaces; its platform-specific artifact is `libomni_usd_live.dylib`. A hosted Apple Silicon source-build job runs direct component and package checks; supervised release qualification remains separate. SDK and qualification requirements are described in the [build guide](ovlive-build.md#macos-source-sdks).

This reference states the semantics, limits and defaults of ovlive's public interfaces: the C ABI in [Live.h](../include/omni/usd/live/Live.h), the C++ host API in [LiveUsd.h](../include/omni/usd/live/LiveUsd.h), and the `.live` file format and `live://` resolver inside USD. It is for engineers and agents who write a host, binding or test against those interfaces and need exact values. The [design](ovlive-design.md), linked from each section, explains why each rule exists; the headers are normative for signatures, and the [Python guide](ovlive-python.md) covers `omni.usd.live`, whose names are the C names in snake case.

## Supported platform and service boundary

The implemented profile is Linux x86-64 with USD 25.11 and Python 3.12, in the stock and Kit USD builds of the [build guide](ovlive-build.md). The runtime talks to any endpoint that implements the Live Services `v1alpha` protocol and satisfies the [service assumptions](#service-assumptions). The [support boundary](ovlive-support-boundary.md) states how the plugin is tested and validated against services, and what is open.

## Live URIs

A live URI is `live://<authority>/<session-id>` for a session or `live://<authority>/<session-id>/<layer-name>` for a layer. The helpers `omniUsdLiveMakeSessionUri`, `MakeLayerUri`, `ParseUri` and `CombineWithBaseUri` are pure functions that need no runtime, endpoint or network, and the endpoint registry and resolver use the same codec. [Design section 5](ovlive-design.md#5-live-uris-and-portable-identity) explains why URIs carry no token or handle.

**Authority.** A host is an ASCII DNS name (internationalized names in caller-supplied punycode), a strict IPv4 literal or a bracketed IPv6 literal. DNS names are lowercased and lose one terminal dot; labels are 1 to 63 letters, digits or inner hyphens, and the name is at most 253 bytes. A host of only digits and dots must be valid IPv4. IPv6 is written in lowercase hexadecimal, compressing the first longest run of at least two zero words. A port is 1 to 65535; 443 is omitted and other ports have no leading zeros. Credentials, zone-qualified IPv6 literals, backslashes and whitespace are rejected. Parsing never resolves DNS, so two aliases for one server are two authorities. `localhost`, `127.0.0.1` and `[::1]` are loopback.

**Session ID and layer name.** A session ID is an opaque, nonempty UTF-8 string other than `.` or `..`, encoded as one URI component: every byte outside the unreserved set (letters, digits, `-`, `.`, `_`, `~`) is percent-encoded in uppercase hexadecimal, so an encoded slash is session data. A layer name is one or more components separated by literal `/`, each encoded the same way, ending in `.live`; an encoded slash in a layer component is rejected, as are empty, `.` and `..` components. Both reject invalid UTF-8, C0 and C1 controls and backslashes, and neither is Unicode-normalized.

**Syntax.** URI input is ASCII, with every byte outside the unreserved set escaped; escapes of either case are accepted and written uppercase. Queries and fragments are rejected, while a `?` or `#` passed to a builder as name data stays encoded name data. Round trips preserve the session ID and layer name exactly.

**Helpers.** `MakeLayerUri` takes a session URI, not a layer URI, and a decoded layer name. `CombineWithBaseUri` anchors a decoded relative layer name at the base layer's directory, or the session root, and cannot escape the session. `ParseUri` writes the decoded authority, session ID, layer name (empty for a session) and canonical URI as NUL-terminated strings into one caller buffer and returns their offsets, valid only on `OK` and until the buffer changes.

**Errors.** An input or result over 65,536 bytes returns `RESOURCE_EXHAUSTED`, and invalid syntax `INVALID_ARGUMENT`. Sizes include every terminating NUL; a null buffer is a successful size query, and a short buffer returns `BUFFER_TOO_SMALL` and writes nothing.

## ABI conventions

These rules apply to every function in [Live.h](../include/omni/usd/live/Live.h); [design section 16](ovlive-design.md#16-public-c-abi) lists the 40 entry points by area.

**Versioning.** Every configuration and output structure starts with `struct_size` and `abi_version`, which the caller initializes; other input fields start at zero. `OMNI_USD_LIVE_ABI_VERSION` is `0x00010000` (1.0), with the major version in the high 16 bits. An unknown major version returns `UNSUPPORTED`; an input shorter than its structure returns `INVALID_ARGUMENT` before any later field is read, and a larger same-major input tail is ignored. Fields appended within 1.0 accept the layout that ends before them, and a size that splits a field returns `INVALID_ARGUMENT`. `OmniUsdLiveRuntimeOptions` appends `apply_lag_warning_threshold` and then, together, `reconnect_initial_ms`, `reconnect_maximum_ms` and `reconnect_max_attempts`: options whose `struct_size` ends before either select the defaults of the fields they omit, and a size inside the reconnect group returns `INVALID_ARGUMENT`. Outputs are written only up to the caller's `struct_size`; `OmniUsdLiveStatus` appends `buffered_edit_count` and then `read_sequence` and `applied_sequence`, a status output that ends before either receives the fields before it, and an `OmniUsdLiveStatusObservation` that ends before either inline field holds that status prefix, which its `status.struct_size` reports. Nested extensible records are reached through pointers and record arrays are pointer arrays, so extension never changes a consumer's stride.

**Types and ownership.** Handles are opaque 64-bit values from separate registries; zero is invalid except where endpoint zero means "all endpoints". Enum and boolean fields are `uint32_t`, and enum values have no ordering. Durations are milliseconds and absolute times Unix-epoch milliseconds. Inputs, including strings and arrays, are copied before return; callback payloads are borrowed for the callback only, and the caller never frees plugin memory. Callbacks run on runtime callback workers, never under runtime locks, in FIFO order per session and per request. A function that takes a callback returns `OK` when submission succeeds, and the callback may run before it returns; a failed submission zeroes its output handle and schedules no callback.

**Wait, cancel and release.** `omniUsdLiveWait` returns `OK` once the operation and its callback have completed or been suppressed, even when the operation failed; `GetRequestStatus` and `GetRequestError` report the outcome and return `BUSY` while it is pending. A wait timeout never cancels. `omniUsdLiveCancel` suppresses a callback that has not started, lets a running one finish, and returns `OK` only after callback quiescence; it never turns a committed or uncertain result into `CANCELLED` and cannot undo a service mutation. Suppressed payloads retire on a callback worker, so `Cancel` with timeout zero can return `TIMEOUT` while that cleanup is queued. `omniUsdLiveReleaseRequest` starts cancellation and returns `BUSY` until quiescence; the owner calls `Cancel` with a bounded timeout and retries. `LeaveSession` and `UnregisterCallback` follow the same rule, and on `TIMEOUT` or `BUSY` the caller keeps ownership, including user data.

**Zero timeouts.** A zero timeout polls without blocking, and cancellation and teardown still start. A blocking wait from a callback worker on unfinished work, and a callback that cancels its own owner, return `BUSY` instead of deadlocking.

**Initialization.** `omniUsdLiveInitialize` is reference-counted; a later call must pass options equal to the first after defaults are applied, or it returns `INVALID_ARGUMENT`. `omniUsdLiveShutdown` releases one reference; the final release returns `BUSY` while endpoints, joined sessions, global status observers or diagnostic sinks remain, and otherwise drains transport and callbacks within its timeout. `TIMEOUT` keeps the runtime for retry, and `Initialize` returns `BUSY` until shutdown finishes. On Windows, gRPC's own threads can still be exiting for a few milliseconds after the final `Shutdown` returns, and a process that exits then can end before the C runtime's exit-time flush of buffered standard I/O, so a host flushes the output it needs before it exits ([testing guide](ovlive-testing.md#known-test-environment-issues)).

## Results

Every function returns an `OmniUsdLiveResult`; there is no thread-local error state. A status query's `OK` describes the query, and the subject's own result is a field of the output. `omniUsdLiveGetResultString` returns a static name for any value.

### Result codes

| Value | Name | Meaning |
| --- | --- | --- |
| 0 | `OK` | The call, or the reported operation, succeeded |
| 1 | `INVALID_ARGUMENT` | A malformed argument, URI or option, a stale status acknowledgment, or gRPC `INVALID_ARGUMENT` or `OUT_OF_RANGE` |
| 2 | `INVALID_HANDLE` | An unknown, released, closed or wrong-kind handle |
| 3 | `NOT_INITIALIZED` | No runtime is initialized, or a live layer is opened without a configured publication context |
| 4 | `NOT_FOUND` | No such session or layer on the service, no registered endpoint for the authority, or no such status subject or retained creation |
| 5 | `ALREADY_EXISTS` | A second backing for one resource, a conflicting endpoint re-registration, a repeated `Creation::start`, or gRPC `ALREADY_EXISTS` |
| 6 | `UNAUTHENTICATED` | No usable credential within the auth wait, a failed refresh, or a rejected current token |
| 7 | `PERMISSION_DENIED` | The service denied the principal; never triggers a refresh |
| 8 | `UNAVAILABLE` | The service is unreachable, or retries are exhausted |
| 9 | `TIMEOUT` | A local deadline passed, or a read hit gRPC `DEADLINE_EXCEEDED`; ownership is kept |
| 10 | `CANCELLED` | The caller cancelled, a stream was reset, or a layer closed while opening |
| 11 | `MALFORMED_STREAM` | The service violated the protocol, such as bad framing or a sequence regression |
| 12 | `UNSUPPORTED` | An unknown ABI major version, an unsupported USD operation or value, authoring off the host context, or a pull operation on a callback observer |
| 13 | `RESOURCE_EXHAUSTED` | A local [limit](#options-defaults-and-limits), an allocation failure, or a service quota |
| 14 | `BUSY` | The call would block or dependents remain; retry later |
| 15 | `STALE_TOKEN_GENERATION` | `SetAccessToken` named a generation other than the current one |
| 16 | `OBJECT_REPLACED` | A `CreateLayer` replaced the layer's object, or a mutation got gRPC `FAILED_PRECONDITION` |
| 17 | (reserved) | Never assigned |
| 18 | `BUFFER_TOO_SMALL` | A caller buffer is too short; only the required size is written |
| 19 | `OUTCOME_UNKNOWN` | An issued mutation may or may not have committed |
| 20 | `INTERNAL_ERROR` | A caught implementation failure, or a read that ended with an unclassified gRPC status |
| 21 | `IO_ERROR` | A file I/O failure in a host tool or sample; the runtime never returns it |

Neither `INTERNAL_ERROR` nor `IO_ERROR` proves that an issued mutation failed before execution. `OmniUsdLiveRequestError` carries the result, operation, request, attempt and incident IDs and any raw gRPC status; its `detail` is sanitized and empty by default. Incoming scene failures are `MALFORMED_STREAM` for malformed batches, `UNSUPPORTED` for graph cases the client cannot apply, and `RESOURCE_EXHAUSTED` for quotas.

### Mutation classification

A mutation (`CreateLayer`, `UpdateLayer`, `CreateSession`, `DeleteSession`, `DeleteLayer`) reports a typed result for a definitive gRPC rejection and `OUTCOME_UNKNOWN` for every other failure, keeping the raw gRPC code; [design section 13](ovlive-design.md#mutation-classification) gives the table. An `UpdateLayer` or `CreateLayer` refused `UNAVAILABLE` with the live edit service's `google.rpc.ErrorInfo` (domain `nvidia.omniverse.live.v1alpha`) applied nothing and reports `UNAVAILABLE`; any other `UNAVAILABLE` is `OUTCOME_UNKNOWN`. Issued mutations are never retried, except that an `UpdateLayer` rejected with `UNAUTHENTICATED` is issued once more after the host installs a new token generation ([design section 13](ovlive-design.md#the-single-unauthenticated-reissue)), an `UpdateLayer` refused with nothing applied is issued again unchanged after a backoff: without a `google.rpc.RetryInfo` delay, that attempt's draw in the reconnect window of `reconnect_initial_ms` and `reconnect_maximum_ms`; with one, the longer of the delay and the draw, with no jitter added to the delay, capped at `reconnect_maximum_ms`, until `reconnect_max_attempts` consecutive refusals fail the layer `UNAVAILABLE`, a `CreateLayer` refused with nothing applied is issued again unchanged after the same backoff while it ends within the initial-open deadline, until `reconnect_max_attempts` consecutive refusals fail the creation `UNAVAILABLE` with its seed kept ([design section 13](ovlive-design.md#the-reissue-after-a-refusal-that-applied-nothing)), and, on an endpoint whose `Ping` reports operation IDs and chained writes, a layer chains its writes: several are in flight, each names its predecessor in a `ChainedUpdateLayerHeader`, and a write that did not commit (outcome unknown, refused `ABORTED` with `PREDECESSOR_NOT_COMMITTED`, or refused with nothing applied) cancels the chained calls behind it and is re-sent unchanged at once, the first re-send with `no_wait`, which commits it exactly once; a chained write a replica refuses `INVALID_ARGUMENT` without an `ErrorInfo` of the service's domain, for not knowing the chained header, applied nothing, and the endpoint's writes are sent unchained from then on, each once every chained call of it or an earlier write is settled, writes whose chained calls ended unsettled first finished through the chain with `no_wait`, while an `INVALID_ARGUMENT` with that `ErrorInfo` (`INVALID_UPDATE`) is definitive ([design section 13](ovlive-design.md#chained-writes)). An unchained `UpdateLayer` or a `CreateLayer` whose outcome is unknown is re-sent once with the same bytes and operation ID on an endpoint whose `Ping` reports operation-ID support. An `OK` acknowledges an attempt that follows one of unknown outcome only with `OPERATION_OUTCOME_APPLIED` or `OPERATION_OUTCOME_ALREADY_COMMITTED`, and an `OPERATION_OUTCOME_UNSPECIFIED` reply leaves `OUTCOME_UNKNOWN` ([design section 13](ovlive-design.md#the-single-operation-id-re-send)). The issuance window and its bounds are internal, not options.

### Read retry policy

`ListSessions` and `StatLayer` retry only `UNAVAILABLE` and `UNAUTHENTICATED`. After `UNAVAILABLE` the next attempt waits a backoff whose base starts at `retry_initial_ms` and doubles up to `retry_maximum_ms`, drawn from the upper half of the base. After `UNAUTHENTICATED` the request parks until a new token generation, bounded by the endpoint's auth wait. At most `retry_max_attempts` attempts are issued within one deadline of `request_timeout_ms`; every other status is final, and a passed deadline ends the request with `TIMEOUT`, or `UNAUTHENTICATED` when no usable credential was found. A read maps gRPC `INVALID_ARGUMENT` and `OUT_OF_RANGE` to `INVALID_ARGUMENT`; `NOT_FOUND`, `ALREADY_EXISTS`, `PERMISSION_DENIED`, `UNAUTHENTICATED`, `RESOURCE_EXHAUSTED`, `UNAVAILABLE` and `CANCELLED` to the results of the same name; `DEADLINE_EXCEEDED` to `TIMEOUT`; `UNIMPLEMENTED` to `UNSUPPORTED`; and any other code, including `FAILED_PRECONDITION`, to `INTERNAL_ERROR`. gRPC's own retry and resolver-supplied service configuration are disabled. Layer streams and joins reconnect with their own backoff, the `reconnect_*` options below ([design section 12](ovlive-design.md#reconnect)); `Ping` is never retried.

## Options, defaults and limits

### Runtime options

`OmniUsdLiveRuntimeOptions` configures the process-wide runtime. A zero numeric field selects its default, except the severity and flag fields, where zero is a value; a null options pointer selects every default, including severity `INFO`.

| Option | Default | Meaning |
| --- | --- | --- |
| `default_auth_wait_timeout_ms` | 30000 | Auth wait for an endpoint whose own `auth_wait_timeout_ms` is zero |
| `max_rejected_token_generations` | 2 | Replacement generations the service may reject after the initial one before refresh fails |
| `auth_expiry_lead_ms` | 30000 | How long before token expiry an `EXPIRING` challenge is raised |
| `request_timeout_ms` | 35000 | Overall deadline of a finite request and of each unary and layer-write RPC; above the Live Edit service's 30 s request timeout, so a layer write's deadline outlasts the service's internal retries. After the operation-ID re-send of an unchained layer write, the layer issues no later write until this deadline of the first attempt, plus 5 s, has passed ([design section 13](ovlive-design.md#the-single-operation-id-re-send)) |
| `initial_open_timeout_ms` | 30000 | Deadline for opening or creating a layer, including its metadata queries. Also the time each layer-read attempt has for its first complete response: an attempt without one ends `TIMEOUT` and counts toward `reconnect_max_attempts`; a read past its first response has no deadline |
| `retry_initial_ms` | 100 | First backoff base of a finite read; must not exceed `retry_maximum_ms` |
| `retry_maximum_ms` | 10000 | Finite-read backoff base ceiling |
| `retry_max_attempts` | 8 | Attempts per finite read |
| `peer_reconciliation_grace_ms` | 10000 | Time an absent peer has after a reconnect before a Left event |
| `max_update_bytes` | 104857600 (100 MiB, the live edit service's default upload limit) | Largest single update the layer sends and largest session message; at most `INT_MAX`. A session message is also capped at 4,128,768 bytes, the service's 4 MiB stream message limit less framing, and a larger one is refused locally with `RESOURCE_EXHAUSTED`. A read update is bounded by `max_pending_bytes_per_endpoint` instead, since a catch-up snapshot of a large layer exceeds any single upload |
| `max_pending_bytes_per_endpoint` | 1073741824 (1 GiB) | Bytes an endpoint may reserve for payloads, reads, messages and responses; at least `max_update_bytes`; shared by every layer, read, intent, message and response on the endpoint, so one layer's held read frames reduce what another can reserve, and a read frame it cannot reserve fails that layer. It also bounds, as a separate budget, every layer's buffered host edits on the endpoint together ([design section 8](ovlive-design.md#quotas)); transport reservations never count buffered bytes, so an endpoint can retain about twice this value plus one started host turn per layer: buffered host edits up to it plus each layer's turn in progress, and transport up to it. Size memory for both budgets |
| `max_pending_messages_per_session` | 1024 | Queued outgoing messages per joined session, including the one in flight; also bounds tracked peers per session |
| `max_synthetic_gap` | 65536 | Largest coalesced sequence jump; a larger one fails the layer with `RESOURCE_EXHAUSTED` |
| `max_status_subjects` | 4096 | Status subjects; each endpoint, joined session and live layer needs one |
| `max_status_observers` | 64 | Status callbacks and pull observers together |
| `max_diagnostic_records` | 2048 | Recent diagnostic records retained in history; a snapshot with NULL export options, or a Python `copy_diagnostics` or `export_diagnostics_json` call without `max_records`, returns all of them |
| `max_diagnostic_bytes` | 8388608 (8 MiB) | Byte bound on that history; the history holds whichever bound is smaller |
| `max_diagnostic_sink_records` | 512 | Records queued per diagnostic sink |
| `max_diagnostic_sinks` | 8 | Registered diagnostic callbacks |
| `max_closed_incidents` | 128 | Closed incidents retained for diagnostics |
| `diagnostic_min_severity` | `INFO` with null options; otherwise the given value | Lowest severity recorded; `TRACE` is 0 and `ERROR` 4 |
| `diagnostic_allowed_flags` | 0 | Which diagnostic opt-ins (`RESOURCE_NAMES`, `SERVER_DETAIL`) a sink may request |
| `flags` | 0 | `DISABLE_DIAGNOSTIC_HISTORY` turns off record history |
| `default_jitter` | 10 ms constant, multiple 2, 1000 ms maximum | Jitter holdback for new endpoints; see below |
| `max_requests` | 4096 | Unreleased requests, counting completed ones; one internal timing slot is extra. Also each layer's incoming bound: its read pauses when queued frames plus unapplied peer coverage reach `max_requests − max_requests/8` and resumes at `max_requests/2`. Retained peer and local work share the cap: a peer frame that finds it full fails the layer with `RESOURCE_EXHAUSTED`, and a flush that finds it full leaves the layer `BLOCKED` |
| `callback_worker_count` | 2 | Callback worker threads; at most 64 |
| `max_auth_callbacks` | 64 | Auth-required callback registrations per runtime |
| `reconnect_initial_ms` | 1000 | Nominal first wait of a layer-read or session-join reconnect, or of a layer write's or creation's reissue after a refusal that applied nothing and carried no retry delay; attempt *n* waits uniformly in [0, 2 × min(`reconnect_maximum_ms`, `reconnect_initial_ms` × 2^(*n*−1))), drawn per layer or session. Must not exceed `reconnect_maximum_ms` |
| `reconnect_maximum_ms` | 60000 | Nominal reconnect wait ceiling; the longest single wait is twice it. A write or creation reissue after a refusal with a retry delay waits the longer of that delay and its reconnect draw, capped at this maximum |
| `reconnect_max_attempts` | 10 | Consecutive failed reconnect attempts, counting a failed first read or join and a read attempt without its first complete response within `initial_open_timeout_ms`, after which a layer or session fails with `UNAVAILABLE`; recovery readiness or a validated Connection restarts the count. With the defaults the mean total wait before giving up is about 303 s. Also the consecutive refusals with nothing applied after which a layer write fails its layer, or a creation fails, `UNAVAILABLE` |
| `apply_lag_warning_threshold` | 256 | Sequence positions a `READY` layer's applied sequence may trail its read sequence, beyond the read generation's catch-up target, before an `apply_lag_entered` warning; the matching `apply_lag_exited` follows once the lag falls to half of it. The default is four full 64-update batches. The layer stays `READY`; `UINT64_MAX` disables the warning |

Options that break a rule in the table, or name an unknown severity or flag, make `Initialize` return `INVALID_ARGUMENT`.

**Jitter.** Peer updates are held back by `delay_constant_ms + delay_multiple × estimated one-way latency`, capped at `delay_maximum_ms`; without an estimate the constant applies, and a zero maximum disables holdback and the timing probe. `Ping` samples the latency every 10 seconds with a one-second deadline, and a failed or skipped probe never fails an edit. `omniUsdLiveConfigureJitterReduction` sets one endpoint's parameters, or with endpoint zero the default for later endpoints, and applies to updates already held.

### Endpoint options

`OmniUsdLiveEndpointOptions` configures one registration. Everything except the token and expiry is immutable for the endpoint's lifetime.

| Field | Default | Meaning |
| --- | --- | --- |
| `authority` | required | `host[:port]` as in [live URIs](#live-uris); at most 65,536 bytes |
| `tls_server_name` | the DNS host | Required for TLS to an IP literal; must be a DNS name without a port, and for a DNS authority must equal its host; at most 253 bytes |
| `ca_pem` | platform trust roots | PEM trust roots, at most 1 MiB |
| `access_token` | none | Initial bearer token, printable ASCII without spaces, at most 65,536 bytes; without one the endpoint starts in `AUTH_WAIT` |
| `expires_at_ms` | 0 (no expiry) | Initial token expiry, Unix-epoch milliseconds |
| `transport_security` | `DEFAULT` | `DEFAULT` is plaintext to a loopback authority and TLS otherwise; `TLS` and `PLAINTEXT` override it |
| `auth_wait_timeout_ms` | the runtime default | How long work waits for a usable credential before failing with `UNAUTHENTICATED` |

### Fixed limits

| Limit | Value | Effect |
| --- | --- | --- |
| URI and authority length | 65,536 bytes | Longer input returns `RESOURCE_EXHAUSTED` |
| Layer transfer chunk | 2 MiB | `CreateLayer` and `UpdateLayer` send one header and chunks of at most this size |
| Read frame | `min(max_update_bytes, 2 MiB) + 64` bytes | Largest single read-stream message; each active reader reserves one frame from the endpoint budget |
| Channel message | `max(max_update_bytes + 65,600, 197,632)` bytes, at most `INT_MAX` | Send and receive cap on other methods |
| `ListSessions` response | `min(channel message limit, 1 MiB)` | Reserved from the endpoint budget before issuance |
| `CreateSession` response | 66,560 bytes | Reserved before issuance |
| `StatLayer` response | 4,096 bytes | Reserved before issuance; a creator name longer than about 4 KB fails the stat with `RESOURCE_EXHAUSTED` |
| `DeleteSession`, `DeleteLayer`, `Ping` response | 1,024 bytes | Reserved before issuance, except `Ping`, which reserves nothing |
| Retained creation slots | 64 per process | Explicit retained creations, and implicit creations until they activate or after they fail, beyond this return `RESOURCE_EXHAUSTED`; no slot is evicted |
| Host edit buffer | Every layer's buffer on an endpoint shares `max_pending_bytes_per_endpoint` bytes, separate from transport reservations, and each layer's buffer holds at most 1 GiB; no record count | A mutation that would start a new host turn once the endpoint's buffered bytes reach the budget is refused with `RESOURCE_EXHAUSTED`, and a USD runtime error names the buffered bytes and `max_pending_bytes_per_endpoint`; a turn in progress always records whole ([design section 8](ovlive-design.md#quotas)) |
| Pending intents per layer | 4,096 intents, 256 MiB of what the intents retain (their bytes and operation logs) | Admission beyond either returns `RESOURCE_EXHAUSTED`; accepted work is never evicted ([design section 8](ovlive-design.md#quotas)) |
| Replay per admission | 512 MiB of bytes; 512 MiB of work plus 32 retained charges of the scene | Charged across the whole transaction |
| Peer and join text fields | 65,536 bytes | Connection IDs, user names and application names, including a join's `app_name`; longer values return `RESOURCE_EXHAUSTED` |
| Diagnostics export | 16 MiB | `CopyDiagnostics` output; zero selects it and larger values clamp to it |
| Transport keepalive | 10 s interval, 10 s timeout | Also sent without active calls |

**Budget floor for opening a layer.** A finite request reserves its response allowance from `max_pending_bytes_per_endpoint` before issuance, and fails locally with `RESOURCE_EXHAUSTED`, sending no RPC, when it does not fit. Opening a layer needs one read frame, the assembled catch-up content plus its header and 64 bytes, and the 4,096-byte `StatLayer` allowance at once; `Initialize` accepts a smaller budget, and the open then fails with `RESOURCE_EXHAUSTED`.

## Endpoints and authentication

**Registration.** `omniUsdLiveRegisterEndpoint` returns a handle with one ownership reference. Registering the same authority with equal immutable configuration adds a reference to the same handle without replacing credentials; a different configuration returns `ALREADY_EXISTS`. Each owner pairs its successful registration with exactly one successful `omniUsdLiveUnregisterEndpoint`; the final one returns `BUSY` at once while any layer, joined session, unreleased request or endpoint-scoped callback depends on the endpoint, never returns `TIMEOUT`, and otherwise publishes `CLOSED` and invalidates the handle. [Design section 4](ovlive-design.md#4-endpoints-tls-and-credential-contexts) covers TLS, credential contexts and aliases.

**Tokens and challenges.** `omniUsdLiveSetAccessToken` succeeds only when `expected_generation` is current and then increments it; otherwise it returns `STALE_TOKEN_GENERATION` with a zero output. An empty token requests a refresh, and a new token never clears a failed layer's incident. A challenge has reason `MISSING`, `EXPIRING` (within `auth_expiry_lead_ms`) or `REJECTED` (gRPC `UNAUTHENTICATED` for the current generation); `PERMISSION_DENIED` never starts one. The host answers with `SetAccessToken`, passing the event's `rejected_generation`, or `omniUsdLiveFailAuthRefresh(challenge_id)`. Refresh fails after `auth_wait_timeout_ms`, on `FailAuthRefresh`, or after more than `max_rejected_token_generations` rejected replacements; parked work then ends with `UNAUTHENTICATED`. [Design section 14](ovlive-design.md#14-authentication-and-token-refresh) gives the full sequence.

## Status and drain

**Queries.** `omniUsdLiveGetEndpointStatus`, `GetSessionStatus` and `GetLayerStatus` copy a local snapshot without an RPC, callback, USD access or wait. `GetLayerStatus` returns `NOT_FOUND` for a URI with neither a reserved creation nor an attached layer, and a reopened layer gets a new `subject_id`.

**Fields.** For a layer, `pending_write_count` and `pending_bytes` count admitted intents not yet applied in order, `buffered_edit_count` counts unflushed host edits, and `has_unconfirmed_edits` is nonzero while either is. `read_sequence` is the highest sequence the layer's read delivered and `applied_sequence` the highest applied in order; the apply lag is `read_sequence − applied_sequence` when `read_sequence` is greater and zero otherwise, and it leaves the state `READY`. An acknowledged local write advances `applied_sequence` past `read_sequence`, since the read never delivers the layer's own writes, so a C consumer compares the two before subtracting: the unsigned difference otherwise wraps to a value near `UINT64_MAX`. They refresh without advancing the revision, so poll `GetLayerStatus` to follow them, and they are zero for endpoints and sessions. For an endpoint, `blocked_layers` and `blocked_sessions` count children in `AUTH_WAIT` or `BLOCKED`, and `failed_layers` and `failed_sessions` children in `FAILED`. The state values are `OPENING` (1), `READY` (2), `RECONNECTING` (3), `AUTH_WAIT` (4), `BLOCKED` (5), `FAILED` (7), `CLOSING` (8) and `CLOSED` (9). [Design section 17](ovlive-design.md#17-operational-status) gives each state's meaning, ranking, latching and callback delivery.

**Observers.** A status callback or pull observer counts against `max_status_observers`, and a global one keeps final `Shutdown` at `BUSY` until unregistered. `ReadStatus` copies one selected observation and its URI without acknowledging it; only `AcknowledgeStatus` of that exact subject and revision releases it, and a stale acknowledgment returns `INVALID_ARGUMENT`. An empty read returns `TIMEOUT` at timeout zero, and pull operations on a callback registration return `UNSUPPORTED`.

**Drain and save.** `omniUsdLiveWaitForPendingUpdates` waits for already admitted work under one deadline, as [design section 12](ovlive-design.md#12-stream-ordering-coverage-and-reconnect) describes. Saving an existing layer, with `SdfLayer::Save` or `host::save(layer, timeoutMs)`, flushes its host edit buffer and waits for its admitted updates; where `isCurrent()` holds it polls and returns `BUSY` while work is pending, elsewhere it waits within the timeout, `TIMEOUT` keeps the work, and a `FAILED` layer returns its cause. [Design section 11](ovlive-design.md#11-layer-creation-save-and-replacement) gives the save rules, and the [Python guide](ovlive-python.md) the export-and-reopen procedure.

## USD integration

One library registers the `.live` file format and the `live` Ar resolver. [Design section 6](ovlive-design.md#6-usd-integration-file-format-resolver-one-backing-per-resource-bootstrap) explains both, and [section 11](ovlive-design.md#11-layer-creation-save-and-replacement) explains creation and save.

### Live layers and the `.live` format

A live layer is an `SdfLayer` whose content the Live Service holds and the runtime keeps current; it has no file on disk. Its identifier is a [live URI](#live-uris) whose layer name ends in `.live`, such as `live://live.example.internal/<escaped-session-id>/root.live`: `.live` ends the layer name inside the URI and is not the extension of a stored file. Live layers are in-memory only. A host opens or creates each one on its publication context and inserts it into a stage, and no sublayer, reference or payload in loaded or saved content names a live URI.

The package's `resources/plugInfo.json`, generated from [plugInfo.json.in](../source/live/scene/plugin/plugInfo.json.in), registers both types from the `omni_usd_live` library:

| Type | Base | Registration |
| --- | --- | --- |
| `OmniUsdLiveNativeFileFormat` | `SdfFileFormat` | Format ID `live`, extension `live`, display name `Live Services`, primary format for the extension, target `usd` |
| `OmniUsdLiveResolver` | `ArResolver` | URI scheme `live` |

- `CanRead` accepts only a `live://` URI that the URI codec parses and whose layer name ends in `.live`. A local path or another scheme's URL ending in `.live` is not a live layer, and this format does not read it.
- A live layer has no byte form: opening it as an Ar asset, passing file-format arguments, or a metadata-only read fails with `UNSUPPORTED`.
- `SdfLayer::Save` of an attached layer and `SdfLayer::Export` to a different live URI write to the service: a save drains the layer's updates and an export creates the destination layer ([opening and creating layers](#opening-and-creating-layers), [design section 11](ovlive-design.md#11-layer-creation-save-and-replacement)). `SdfLayer::Export` to an ordinary path, such as a `.usda` file, writes that path's format and is how a host keeps a failed layer's edits ([design section 13](ovlive-design.md#recovery-export-then-reopen)).
- `SdfLayer::ExportToString` and spec text output use the `usda` text format.

**Nucleus live layers are a different format.** Omniverse Nucleus live layers, which `omni.client` and Kit's stock `omni.usd_resolver` serve for `omniverse://` paths ending in `.live`, use their own format and protocol. ovlive neither reads nor writes them, and Nucleus compatibility is outside its scope ([design section 1](ovlive-design.md#1-purpose-audience-and-scope)). A USD process registers one handler for the `.live` extension, so ovlive cannot load beside a plugin that registers the Nucleus `.live` format; the Kit extension rejects startup in that case ([Kit guide](ovlive-kit.md#what-works-with-a-public-kit-build)).

### Resolver

- A `live://` identifier is canonicalized with the URI codec.
- A relative path ending in `.live` (case-sensitive) anchors in the anchoring live layer's directory. Such a path over the 65,536-byte URI limit, or one whose combined URI would exceed it, resolves to `RESOURCE_EXHAUSTED`. An absolute `.live` path, a path containing a colon, or an empty anchor resolves to `INVALID_ARGUMENT`.
- Any other path anchored in a live layer gets an empty identifier with no diagnostic and no RPC, so USD reports it unresolved; hosts author ordinary assets relative to a non-live layer.
- `Resolve` of an unattached layer needs a registered endpoint (`NOT_FOUND` otherwise, reported as a configuration failure) and a successful `StatLayer`, run under the initial-open deadline and the read retry policy; from a callback worker it returns `BUSY` without an RPC.
- `Resolve` of a layer attached in this process, whose status subject exists and is not `CLOSING` or `CLOSED`, uses the attachment as existence evidence and issues no RPC. The first open still stats, and a closing backing is not attached.
- New-asset resolution and write eligibility check only the endpoint registration and create nothing.
- Modification times keep the service's microseconds; zero is the Unix epoch, and clock skew is never clamped.
- The modification time of an attached layer is the one its last successful `StatLayer` returned, with no RPC. A forced reload of an attached layer is a no-op, so an older time only suppresses a reload that does nothing. The cache holds at most `max_status_subjects` URIs and drops a URI's time when a lookup finds no attached layer for it.
- Opening a live layer as a byte asset fails with `UNSUPPORTED`.

Each failure posts a USD error with the typed result and a fixed, credential-free message.

### Opening and creating layers

`SdfLayer::FindOrOpen` runs `Read` on the configured publication context; without one it fails with `NOT_INITIALIZED`, and elsewhere with `BUSY`. Live layers are in-memory only: the host opens each one with `FindOrOpen` or `CreateNew` on its publication context and then inserts it into a stage, and no sublayer, reference or payload in loaded or saved content names a live URI ([design section 1](ovlive-design.md#1-purpose-audience-and-scope)). Composing a live URI the host has not opened is unsupported: USD opens sublayers on its worker threads, where `Read` fails with `BUSY`, and that error's message says to open live layers on the host before composing them. File-format arguments and metadata-only reads fail with `UNSUPPORTED`, a second backing for one resource fails with `ALREADY_EXISTS`, an open of a resource whose layer was released while its backing closes, or whose prior open failed, waits for that backing within the same deadline ([design section 6](ovlive-design.md#6-usd-integration-file-format-resolver-one-backing-per-resource-bootstrap)), and opening fails after `initial_open_timeout_ms`. `SdfLayer::CreateNew`, the first `Save` of an `SdfLayer::New` target and `SdfLayer::Export` to a different live URI each make one `CreateLayer` attempt, retained in a process slot; nonempty comments and format arguments are rejected first, and no failure retries the create.

### Save-notice handlers

USD can deliver `SdfNotice::LayerDidSaveLayerToFile` during `CreateNew` under its layer-registry lock, before the target is initialized. Handlers enqueue work on the serial host for after the outer call returns and capture identifiers or weak handles, never a strong reference to the target; inside the notice they never find, open or create a layer or wait for host work.

### Publication context

`host::setPublicationContext(dispatch, isCurrent)`, or an owned `createPublicationContext`, configures the serial host context for future opens; [design section 7](ovlive-design.md#7-host-concurrency-and-scheduling-contract) states the contract. `dispatchingLocalPublication()` is true while the runtime dispatches a layer's own publication job, bounded to one per layer, so a metering dispatcher can accept it without charging its quota. `PublicationContext::close()` returns `BUSY` until no layer, creation or job holds its callbacks. A dispatcher that meters host jobs sizes its queue at about three charged jobs per live layer; a flush job it refuses for capacity leaves the layer `BLOCKED` with its edits buffered, and a publication job it refuses fails the layer.

### Authoring limits

[Design section 8](ovlive-design.md#8-authoring-and-reconciliation-model) owns the authoring rules; a host observes these limits:

- A mutation where `isCurrent()` is false, or from a handler of a notice a publication job sent, fails the layer with `UNSUPPORTED`.
- A mutation that begins while a read or another call holds the layer's storage is refused before any change with `BUSY`, as an exception derived from `std::runtime_error`.
- A mutation that would start a host turn while every layer's buffered host edits on the endpoint hold `max_pending_bytes_per_endpoint` bytes is refused with `RESOURCE_EXHAUSTED`, a USD runtime error names that bound, and the layer is `BLOCKED` until a flush admits or, with an empty buffer of its own, until another layer's buffer releases bytes. A frame of many change blocks records whole however many it holds, and the flush that follows sends it as several updates, each holding the most whole change blocks the compile limits allow.
- A host turn whose update exceeds the live edit service's entry limits, 99,999 struct commands, set fields and time samples together or 100,000 children in one child reorder, is never sent and never split, like a turn over the compiler's 65,536 operations, which count repeated writes to one field or sample once, or over its 1,048,576 operations counting every write: the layer fails with `RESOURCE_EXHAUSTED`, a USD runtime error names the counts in the service's terms (`DELTA_TOO_LARGE struct_commands=... set_fields=... time_samples=... max_entries=...` or `CHILDREN_LIST_TOO_LONG limit=... actual=...`), and the host content keeps every edit for `SdfLayer::Export`. An on-host `SdfLayer::Save` ends the turn, so a host that authors in bulk saves between chunks. A creation seed over the limits fails before any `CreateLayer` with the same error, so a layer larger than the limit is created in parts: a smaller seed, then saves of further host turns.
- A NaN time-sample key is refused with `INVALID_ARGUMENT`; infinite keys are valid.
- An edit to one dictionary key is sent as a replacement of the whole field, so it can supersede a concurrent peer edit to another key.
- `CopyFrom` and `TransferContent` throw with `UNSUPPORTED` and `Import` returns false, leaving content and status unchanged; `SdfLayer::Clear` fails the layer and its content is lost.
- A creation seed field holding an empty `VtValue` fails with `UNSUPPORTED`; a nested empty value, a non-identity payload layer offset or a type outside the wire's version-zero table fails the flush with `UNSUPPORTED`. Asset paths are sent as authored.
- Creation and flush reject explicitly authored empty derived child lists or time-sample maps with `UNSUPPORTED`, preserving source or host content. Outgoing sample times must be finite. Whole-map replacement removes observed keys through individual sample tombstones; unseen concurrent peer keys survive. Creation preserves child membership and order through section records.
- A dictionary-key edit of `timeSamples`, and a namespace change that identity-preserving Create, Delete and Move commands cannot realize, such as a mapper with a relative target, fail the flush with `UNSUPPORTED`.

A definitive flush failure moves the layer to `FAILED` with host content intact; a transient one leaves it `BLOCKED` with the cause; `buffered_edit_count` is nonzero unless the refusal came through the shared endpoint budget or a capacity refusal of a flush requested while the buffer was empty.

## C++ host API

[LiveUsd.h](../include/omni/usd/live/LiveUsd.h) exposes `omni::usd::live::host` for explicit creation, save and host configuration; it requires the plugin's exact USD, compiler and STL build. Every function is `noexcept`, returns a result and leaves outputs unchanged on failure.

**Creation.** `host::prepareCreation(seed, liveUri, &creation)` freezes an ordinary `SdfLayer` seed on the serial host, outside change blocks, and reserves the destination, endpoint bytes and a status subject without an RPC. `*creation` must be null, and the caller keeps the returned `Creation` for the created layer's whole lifetime. `PublicationContext::prepareCreation` does the same for one exact context and fails once it is closed or replaced.

- `start()` runs on the original host and authorizes the single `CreateLayer`, replacing any existing destination. A repeated `start` returns `ALREADY_EXISTS`; `OK` means submitted.
- `wait(timeoutMs)` and `getOutcome(&outcome)` may run off the host. A successful wait means the request retired, not that creation succeeded; `CreationOutcome` carries the phase (`Prepared`, `Creating`, `Confirmed`, `UnissuedFailure`, `Rejected`, `OutcomeUnknown`), the result, the request error and the status subject.
- `prepareLayer(timeoutMs)` opens the confirmed object once and waits for its bootstrap; `activate()` installs the content and canonical identifier, grants edit permission and starts publication; `getLayer(&layer)` returns the live layer and is `BUSY` before activation. All three run on the original host. Until activation the created layer is read-only.
- `close()` is poll-only cancellation on the original host; on `BUSY`, let queued work retire and call it again. `waitForClose(timeoutMs)` waits off the host for cleanup once `close` began. Destroy the `Creation` on the original host.

```cpp
namespace host = omni::usd::live::host;
host::Creation* raw = nullptr;                    // On the serial host, outside change blocks:
if (host::prepareCreation(*seed, uri.c_str(), &raw) != OMNI_USD_LIVE_OK) return;
std::unique_ptr<host::Creation> creation(raw);
if (creation->start() != OMNI_USD_LIVE_OK) return; // The single CreateLayer attempt.
if (creation->wait(30000) != OMNI_USD_LIVE_OK) return; // Off the host; TIMEOUT keeps the work.
host::CreationOutcome outcome;
creation->getOutcome(&outcome);
if (outcome.phase != host::CreationPhase::Confirmed) return; // Inspect outcome.result.
pxr::SdfLayerRefPtr layer;                        // Back on the serial host:
if (creation->prepareLayer(30000) == OMNI_USD_LIVE_OK && creation->activate() == OMNI_USD_LIVE_OK)
    creation->getLayer(&layer);                   // Author `layer` with plain USD.
```

**Retained creations.** Explicit workflows from `host::prepareRetainedCreation(seed, liveUri, &subjectId)`, and implicit `CreateNew` and live `Export` workflows until activation succeeds, are owned by a process registry of 64 slots; a full registry returns `RESOURCE_EXHAUSTED` and nothing is evicted. A host job after the outer USD call removes a successfully activated implicit workflow: the layer's binding owns its attachment, and the last layer reference closes it. A failed or uncertain implicit workflow stays retained, and so does one whose host job is refused. An `Export` destination that no host reference holds closes after that job; a reopen waits for the closing destination, and fails at once with `ALREADY_EXISTS` only while the host still holds a queued publication of it (a peer update or its own admitted edits), until it reports `CLOSED`. `nextRetainedCreation(afterSubject, &info)` returns the smallest retained subject above the cursor and its canonical URI; start at zero, and `NOT_FOUND` ends the scan. `acquireRetainedCreation(subjectId, &creation)` runs on the original host and transfers the workflow to a new `Creation`; it returns `BUSY` while a hook holds the workflow. `releaseRetainedCreation(subjectId)` is destructive cleanup on the original host: export needed content with `SdfLayer::Export` first, because even `BUSY` starts cancellation; `BUSY` keeps the slot for retry.

`host::save` follows the [save](#status-and-drain) rules, and the publication functions the [publication context](#publication-context) rules.

## Sessions, presence and messages

[Design section 15](ovlive-design.md#15-sessions-presence-and-messages) describes sessions, presence and messages; these are the interface facts.

- `omniUsdLiveCreateSession` returns only the session ID and URI; `ListSessions` returns ID, URI, name, creator and creation time. A malformed successful creation reply is `OUTCOME_UNKNOWN`.
- `omniUsdLiveJoinSession` requires a nonempty `app_name` of at most 65,536 bytes; `OK` means admitted, and each attempt reports a Connection event. An absent peer is reported Left after `peer_reconciliation_grace_ms`.
- `omniUsdLiveSendMessage` completes on local queue acceptance, never on peer delivery. A message is at most `max_update_bytes`, and at most `max_pending_messages_per_session` wait. A write that failed in flight is sent again after reconnect, so a peer can receive a message twice; `LeaveSession` or a terminal failure discards the queue.
- `LeaveSession` returns `OK` once callbacks and transport are quiescent; send requests are released separately.
- Every join opts in to the service's client keepalive. When the join's Connected reports a non-zero keepalive interval, decided on every join, the session's own worker sends an empty keepalive after a full interval in which no write on the stream has finished, and none while messages flow; with a zero interval, as from a service that has the keepalive disabled or predates it, the stream carries only the header and messages. A keepalive takes no request handle, callback, queue slot or pending bytes. A join the service ends `UNAVAILABLE` for a keepalive timeout rejoins with the reconnect backoff, and its `stream_disconnected` record carries `reason` 1.

## Service assumptions

The client relies on these service behaviors beyond the wire protocol. When it detects a broken assumption, the affected layer fails with its edits kept; it never compensates by recreating a layer or resending a mutation. An aggregate that omits the effect of the reader's own acknowledged write is not detectable: the client retires that intent when the aggregate applies, and the edit is lost from the layer. [Design section 20](ovlive-design.md#20-service-assumptions-and-support-boundaries) states them as design constraints, with the service's conflict rules.

| Concern | Client behavior and assumption |
| --- | --- |
| Session creation | Only the session ID is returned; metadata comes from listing, and a failed listing never repeats a create |
| Session names | Not unique and not an idempotency key |
| Base-layer identity | The host's stable, unsigned canonical URL is passed verbatim |
| User identity | A join sends only the application name; user names and connection IDs are service-reported |
| Client keepalive | A service that reports a non-zero keepalive interval ends a join that sent nothing for three intervals `UNAVAILABLE` with `google.rpc.ErrorInfo` reason `CLIENT_KEEPALIVE_TIMEOUT` and one `PeerLeft`; its timer restarts on any bytes of the request stream arriving and does not run while the service is not reading the stream |
| Authentication | Authentication and permission checks precede commit, so `UNAUTHENTICATED` proves an update did not execute |
| Object replacement | `CreateLayer` produces a new object at sequence 1; saving an existing layer drains updates and never recreates |
| Sequence resume | A reconnect resumes only from contiguous retained coverage; a self-write acknowledgment never covers an unseen peer position |
| Self-filtering | Echoes are filtered by the endpoint lifetime's writer ID; other endpoints, including DNS aliases of one server, see each other's writes |
| Coalescing | An aggregate update is the service's state at its last sequence: it covers every peer effect in the span it skips and the effect of every own write acknowledged in that span after its first sequence, which retires when the aggregate applies; an aggregate that starts on an own acknowledged sequence fails the layer with `UNSUPPORTED` |
| Catch-up | The first complete response of a subscription attempt is its catch-up target; an empty caught-up header establishes readiness |
| Backend rotation | Backends behind one authority share objects, ordering and resumable history, or routing preserves those guarantees |
| Clocks | `Ping` server times from different backends are close enough for jitter estimation |
| Durability | A `FAILED` layer keeps every local edit in host content for `SdfLayer::Export`; unacknowledged work does not survive a process crash |

## Known limitations

The [support boundary](ovlive-support-boundary.md#known-limitations) lists the limitations a host meets, each with its open item. This reference adds only the rules a host builds on:

- **Lost responses.** A lost mutation response is `OUTCOME_UNKNOWN`: the layer fails with the intent kept, and any resend is a deliberate user decision after export and reopen ([mutation classification](#mutation-classification)). A definitive rejection is final for issuance but does not prove the update is absent from the service.
- **Same field.** Two writers that set one field converge on the update with the later sequence; each sees its own pending value until its update is confirmed, and neither fails.
- **Multi-editor cases.** [Design section 10](ovlive-design.md#known-limitations) explains concurrent first creation, dictionary-key and child-list replacement, and the conflicts that fail a layer.
- **USD shapes.** The [authoring limits](#authoring-limits) list the unsupported USD operations and values. A flush they fail leaves the layer `FAILED` with the typed cause `UNSUPPORTED` and no detail in status; a flush that fails for one of the [`flush_unsupported`](ovlive-diagnostic-catalog.md#unsupported-flushes) reasons (value type, payload offset, identity type change, multi-update, move cycle, time-sample dictionary key, identity realization) posts a USD runtime error on the host that names the layer and reason, plus the path, field and value type when a single operation causes it, and emits that diagnostic with only the reason code. A compiled edit that conflicts with the current scene when it is replayed onto it fails the flush with the typed status only.
- **Sessions.** Session messages may be delivered twice after a reconnect, and presence departure is a timed heuristic.
- **Resources.** Resource bounds are logical charges on the runtime's own storage, not bounds on process memory or USD caches, and the plugin cannot be unloaded.
