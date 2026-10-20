// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef OMNI_USD_LIVE_LIVE_H
#define OMNI_USD_LIVE_LIVE_H

#include <stdint.h>

#if defined(_WIN32)
#    define OMNI_USD_LIVE_CALL __cdecl
#    if defined(OMNIUSDLIVE_EXPORTS)
#        define OMNI_USD_LIVE_API __declspec(dllexport)
#    else
#        define OMNI_USD_LIVE_API __declspec(dllimport)
#    endif
#else
#    define OMNI_USD_LIVE_CALL
#    define OMNI_USD_LIVE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OMNI_USD_LIVE_ABI_VERSION UINT32_C(0x00010000)
#define OMNI_USD_LIVE_DIAGNOSTIC_SCHEMA_VERSION UINT32_C(1)

/* All handles are opaque; zero is invalid except documented all-endpoint queries.
 * Inputs are copied before return. Callback payloads are borrowed for the call.
 * Every configuration/output structure starts with struct_size and abi_version.
 * Caller initializes those two fields; other input fields start at zero.
 * Unknown major versions fail; documented trailing extensions are ignored.
 * Arrays of extensible records are arrays of pointers, never sizeof-strided
 * records. Nested records use pointers so their growth cannot shift outer fields.
 * uint32_t boolean fields take 0 or 1. Enum values have no ordering semantics.
 * All durations are milliseconds, absolute times Unix epoch milliseconds. */
typedef uint64_t OmniUsdLiveEndpoint;
typedef uint64_t OmniUsdLiveSession;
typedef uint64_t OmniUsdLiveRequest;
typedef uint64_t OmniUsdLiveCallbackHandle;
typedef uint32_t OmniUsdLiveResult;

enum {
    OMNI_USD_LIVE_OK = 0,
    OMNI_USD_LIVE_INVALID_ARGUMENT = 1,
    OMNI_USD_LIVE_INVALID_HANDLE = 2,
    OMNI_USD_LIVE_NOT_INITIALIZED = 3,
    OMNI_USD_LIVE_NOT_FOUND = 4,
    OMNI_USD_LIVE_ALREADY_EXISTS = 5,
    OMNI_USD_LIVE_UNAUTHENTICATED = 6,
    OMNI_USD_LIVE_PERMISSION_DENIED = 7,
    OMNI_USD_LIVE_UNAVAILABLE = 8,
    OMNI_USD_LIVE_TIMEOUT = 9,
    OMNI_USD_LIVE_CANCELLED = 10,
    OMNI_USD_LIVE_MALFORMED_STREAM = 11,
    OMNI_USD_LIVE_UNSUPPORTED = 12,
    OMNI_USD_LIVE_RESOURCE_EXHAUSTED = 13,
    OMNI_USD_LIVE_BUSY = 14,
    OMNI_USD_LIVE_STALE_TOKEN_GENERATION = 15,
    OMNI_USD_LIVE_OBJECT_REPLACED = 16,
    /* 17 is reserved and never assigned. */
    OMNI_USD_LIVE_BUFFER_TOO_SMALL = 18,
    OMNI_USD_LIVE_OUTCOME_UNKNOWN = 19,
    OMNI_USD_LIVE_INTERNAL_ERROR = 20,
    OMNI_USD_LIVE_IO_ERROR = 21
};
enum { OMNI_USD_LIVE_TRANSPORT_DEFAULT = 0, OMNI_USD_LIVE_TRANSPORT_TLS = 1, OMNI_USD_LIVE_TRANSPORT_PLAINTEXT = 2 };
enum { OMNI_USD_LIVE_SUBJECT_ENDPOINT = 1, OMNI_USD_LIVE_SUBJECT_SESSION = 2, OMNI_USD_LIVE_SUBJECT_LAYER = 3 };
enum {
    OMNI_USD_LIVE_STATE_OPENING = 1,
    OMNI_USD_LIVE_STATE_READY = 2,
    OMNI_USD_LIVE_STATE_RECONNECTING = 3,
    OMNI_USD_LIVE_STATE_AUTH_WAIT = 4,
    OMNI_USD_LIVE_STATE_BLOCKED = 5,
    /* 6 is reserved and never assigned. */
    OMNI_USD_LIVE_STATE_FAILED = 7,
    OMNI_USD_LIVE_STATE_CLOSING = 8,
    OMNI_USD_LIVE_STATE_CLOSED = 9
};
enum {
    OMNI_USD_LIVE_OPERATION_NONE = 0,
    OMNI_USD_LIVE_OPERATION_OPEN = 1,
    OMNI_USD_LIVE_OPERATION_CREATE = 2,
    OMNI_USD_LIVE_OPERATION_READ = 3,
    OMNI_USD_LIVE_OPERATION_UPDATE = 4,
    OMNI_USD_LIVE_OPERATION_SAVE = 5,
    OMNI_USD_LIVE_OPERATION_JOIN = 6,
    OMNI_USD_LIVE_OPERATION_MESSAGE = 7,
    OMNI_USD_LIVE_OPERATION_AUTH = 8,
    OMNI_USD_LIVE_OPERATION_DRAIN = 9,
    OMNI_USD_LIVE_OPERATION_LIST = 10,
    OMNI_USD_LIVE_OPERATION_DELETE = 11,
    OMNI_USD_LIVE_OPERATION_STAT = 12,
    OMNI_USD_LIVE_OPERATION_PING = 13,
    OMNI_USD_LIVE_OPERATION_SHUTDOWN = 14
    /* 15 is reserved and never assigned. */
};
enum {
    OMNI_USD_LIVE_RECOVERY_NONE = 0,
    OMNI_USD_LIVE_RECOVERY_WAIT_FOR_RETRY = 1,
    OMNI_USD_LIVE_RECOVERY_REFRESH_AUTH = 2,
    OMNI_USD_LIVE_RECOVERY_RELIEVE_BACKPRESSURE = 3,
    OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN = 4,
    OMNI_USD_LIVE_RECOVERY_RECONFIGURE = 5
};
enum { OMNI_USD_LIVE_AUTH_MISSING = 1, OMNI_USD_LIVE_AUTH_EXPIRING = 2, OMNI_USD_LIVE_AUTH_REJECTED = 3 };
enum {
    OMNI_USD_LIVE_EVENT_CONNECTION = 1,
    OMNI_USD_LIVE_EVENT_JOINED = 2,
    OMNI_USD_LIVE_EVENT_LEFT = 3,
    OMNI_USD_LIVE_EVENT_MESSAGE = 4
};
enum {
    OMNI_USD_LIVE_SEVERITY_TRACE = 0,
    OMNI_USD_LIVE_SEVERITY_DEBUG = 1,
    OMNI_USD_LIVE_SEVERITY_INFO = 2,
    OMNI_USD_LIVE_SEVERITY_WARN = 3,
    OMNI_USD_LIVE_SEVERITY_ERROR = 4
};
enum { OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES = 1, OMNI_USD_LIVE_DIAGNOSTIC_SERVER_DETAIL = 2 };
enum { OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY = 1 };
enum {
    OMNI_USD_LIVE_FIELD_UINT64 = 1,
    OMNI_USD_LIVE_FIELD_INT64 = 2,
    OMNI_USD_LIVE_FIELD_STRING = 3,
    OMNI_USD_LIVE_FIELD_BOOLEAN = 4
};
enum {
    OMNI_USD_LIVE_RECORD_ENDPOINT_REGISTERED = 1,
    OMNI_USD_LIVE_RECORD_ENDPOINT_CLOSED = 2,
    OMNI_USD_LIVE_RECORD_RPC_COMPLETED = 3,
    OMNI_USD_LIVE_RECORD_RPC_FAILED = 4,
    OMNI_USD_LIVE_RECORD_AUTH_CHALLENGE = 5,
    OMNI_USD_LIVE_RECORD_AUTH_REPLACED = 6,
    OMNI_USD_LIVE_RECORD_AUTH_FAILED = 7,
    OMNI_USD_LIVE_RECORD_STREAM_DISCONNECTED = 8,
    OMNI_USD_LIVE_RECORD_STREAM_RESUMED = 9,
    OMNI_USD_LIVE_RECORD_LAYER_READY = 10,
    OMNI_USD_LIVE_RECORD_OBJECT_REPLACED = 11,
    OMNI_USD_LIVE_RECORD_RECOVERY_GATE_CLOSED = 12,
    OMNI_USD_LIVE_RECORD_RECOVERY_GATE_OPENED = 13,
    OMNI_USD_LIVE_RECORD_OUTCOME_UNKNOWN = 14,
    OMNI_USD_LIVE_RECORD_BACKPRESSURE_ENTERED = 15,
    OMNI_USD_LIVE_RECORD_BACKPRESSURE_EXITED = 16,
    OMNI_USD_LIVE_RECORD_SAVE_COMPLETED = 17,
    OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED = 18,
    OMNI_USD_LIVE_RECORD_DRAIN_TIMEOUT = 19,
    OMNI_USD_LIVE_RECORD_QUIESCENCE_FAILED = 20,
    OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED = 21,
    OMNI_USD_LIVE_RECORD_RECORDS_SUPPRESSED = 22,
    /* 23 is reserved and never assigned. */
    /* Message activity is local queue/transport evidence, never peer delivery. */
    OMNI_USD_LIVE_RECORD_MESSAGE_ACCEPTED = 24,
    OMNI_USD_LIVE_RECORD_MESSAGE_WRITE_COMPLETED = 25,
    OMNI_USD_LIVE_RECORD_MESSAGE_RECEIVED = 26,
    OMNI_USD_LIVE_RECORD_MESSAGE_DISCARDED = 27,
    OMNI_USD_LIVE_RECORD_MESSAGE_REJECTED = 28,
    /* Host callback exception; never substitutes for an operation result. */
    OMNI_USD_LIVE_RECORD_CALLBACK_FAILED = 29,
    /* A host edit flush failed UNSUPPORTED; a reason code, never scene names. */
    OMNI_USD_LIVE_RECORD_FLUSH_UNSUPPORTED = 30,
    /* A layer closed with unadmitted host edits; a count, never scene names. */
    OMNI_USD_LIVE_RECORD_HOST_EDITS_DISCARDED = 31,
    /* A layer's read paused at its incoming high-water mark; a queue depth, never scene names. */
    OMNI_USD_LIVE_RECORD_READ_PAUSED = 32,
    /* A paused read resumed at its low-water mark. */
    OMNI_USD_LIVE_RECORD_READ_RESUMED = 33,
    /* A layer's applied sequence trails its read sequence by more than
     * apply_lag_warning_threshold positions; sequences, never scene names. */
    OMNI_USD_LIVE_RECORD_APPLY_LAG_ENTERED = 34,
    /* A lagging layer's applied sequence came within half the threshold. */
    OMNI_USD_LIVE_RECORD_APPLY_LAG_EXITED = 35
};

typedef struct OmniUsdLiveCapabilities {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t delegate_version; /* Always 2; no selector or fallback. */
    uint32_t diagnostic_schema_version;
    uint32_t autonomous_progress;
} OmniUsdLiveCapabilities;

typedef struct OmniUsdLiveJitterParams {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t delay_constant_ms;
    uint32_t delay_multiple;
    uint32_t delay_maximum_ms; /* Zero means no holdback, not a default. */
} OmniUsdLiveJitterParams;

/* NULL Initialize options selects defaults. Zero numeric limits also select
 * defaults, except severity (TRACE=0) and flags. docs/ovlive-reference.md lists
 * the defaults. NULL default_jitter selects default jitter. */
typedef struct OmniUsdLiveRuntimeOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t default_auth_wait_timeout_ms;
    uint32_t max_rejected_token_generations;
    uint32_t auth_expiry_lead_ms;
    uint32_t request_timeout_ms;
    uint32_t initial_open_timeout_ms;
    uint32_t retry_initial_ms;
    uint32_t retry_maximum_ms;
    uint32_t retry_max_attempts;
    uint32_t peer_reconciliation_grace_ms; /* Zero selects default 10000 ms. */
    uint32_t max_update_bytes;
    uint64_t max_pending_bytes_per_endpoint;
    uint32_t max_pending_messages_per_session;
    uint32_t max_synthetic_gap;
    uint32_t max_status_subjects;
    uint32_t max_status_observers;
    uint32_t max_diagnostic_records;
    uint64_t max_diagnostic_bytes;
    uint32_t max_diagnostic_sink_records;
    uint32_t max_diagnostic_sinks;
    uint32_t max_closed_incidents;
    uint32_t diagnostic_min_severity;
    uint32_t diagnostic_allowed_flags;
    uint32_t flags;
    const OmniUsdLiveJitterParams* default_jitter;
    uint32_t max_requests; /* Zero selects 4096 ordinary unreleased requests.
                             One additional internal timing-probe slot is reserved. */
    uint32_t callback_worker_count; /* Zero selects 2; maximum 64. */
    uint32_t max_auth_callbacks; /* Zero selects 64 registrations per runtime. */
    /* Sequence positions a layer's applied sequence may trail its read
     * sequence before an apply_lag_entered record; zero selects 256. Appended
     * in 1.0: inputs accept struct_size equal to the offset of this field and
     * then use the default. */
    uint64_t apply_lag_warning_threshold;
    /* Backoff of layer-read and session-join reconnects, separate from the
     * retry_* options of finite requests: attempt n waits uniformly in
     * [0, 2 x min(reconnect_maximum_ms, reconnect_initial_ms x 2^(n-1))), and
     * after reconnect_max_attempts consecutive failed attempts the layer or
     * session fails UNAVAILABLE. A layer write or creation the service
     * refuses with nothing applied is reissued after that attempt's draw;
     * when the refusal carries a retry delay, the wait is the longer of the
     * delay and the draw, capped at reconnect_maximum_ms. The layer or
     * creation fails UNAVAILABLE after reconnect_max_attempts refusals, and a
     * creation also when its next reissue would pass its initial-open
     * deadline.
     * Zero selects 1000 ms, 60000 ms and 10;
     * reconnect_initial_ms must not exceed reconnect_maximum_ms. Appended
     * together in 1.0: inputs accept struct_size equal to the offset of
     * reconnect_initial_ms and then use the defaults. */
    uint32_t reconnect_initial_ms;
    uint32_t reconnect_maximum_ms;
    uint32_t reconnect_max_attempts;
} OmniUsdLiveRuntimeOptions;

typedef struct OmniUsdLiveEndpointOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* authority;
    const char* tls_server_name;
    const char* ca_pem;
    const char* access_token;
    uint64_t expires_at_ms;
    uint32_t transport_security;
    uint32_t auth_wait_timeout_ms;
} OmniUsdLiveEndpointOptions;

typedef struct OmniUsdLiveAuthState {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t generation;
    uint64_t expires_at_ms;
    uint64_t challenge_id;
    uint32_t state;
    uint32_t reason; /* Auth reason; zero when no active challenge. */
} OmniUsdLiveAuthState;

typedef struct OmniUsdLiveAuthRequired {
    uint32_t struct_size;
    uint32_t abi_version;
    OmniUsdLiveEndpoint endpoint; /* Borrowed, not a registration reference. */
    uint64_t rejected_generation;
    uint64_t challenge_id;
    uint32_t reason;
    uint32_t grpc_status_present;
    int32_t grpc_status;
} OmniUsdLiveAuthRequired;

typedef struct OmniUsdLiveJoinOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* app_name; /* Required; no user-name override on the wire. */
} OmniUsdLiveJoinOptions;

typedef struct OmniUsdLiveCreatedSession {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* session_id;
    const char* uri;
} OmniUsdLiveCreatedSession;

typedef struct OmniUsdLiveSessionInfo {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* session_id;
    const char* uri;
    const char* name;
    const char* created_by;
    uint64_t created_time_ms;
} OmniUsdLiveSessionInfo;

typedef struct OmniUsdLiveLayerInfo {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t created_time_ms;
    uint64_t modified_time_ms;
} OmniUsdLiveLayerInfo;

typedef struct OmniUsdLiveSessionEvent {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t kind;
    OmniUsdLiveResult result;
    const char* connection_id;
    const char* user_name; /* Service-reported; NULL when not applicable. */
    const char* app_name;
    const void* bytes;
    uint64_t size;
} OmniUsdLiveSessionEvent;

typedef struct OmniUsdLiveStatus {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t subject_kind;
    uint32_t state;
    uint64_t subject_id;
    uint64_t revision;
    uint64_t changed_at_ms;
    OmniUsdLiveResult result;
    uint32_t operation;
    uint32_t grpc_status_present;
    int32_t grpc_status;
    uint32_t recovery_action;
    uint32_t progress_known;
    uint64_t incident_id;
    uint64_t pending_write_count;
    uint64_t pending_bytes;
    uint32_t has_unconfirmed_edits;
    uint32_t blocked_layers;
    uint32_t failed_layers;
    uint32_t blocked_sessions;
    uint32_t failed_sessions;
    /* Records in the host edit buffer that no flush has admitted. Appended in
     * 1.0: outputs accept struct_size equal to the offset of this field and then
     * write only the fields before it. */
    uint64_t buffered_edit_count;
    /* Layer progress: the highest sequence the read delivered and the highest
     * sequence applied in order. The apply lag is read_sequence minus
     * applied_sequence when read_sequence is greater, and zero otherwise: an
     * acknowledged local write advances applied_sequence past read_sequence,
     * since the read never delivers the layer's own writes. Compare before
     * subtracting; the unsigned difference otherwise wraps.
     * Zero for endpoints and sessions. These values never advance revision.
     * Appended in 1.0: outputs also accept struct_size equal to the offset of
     * read_sequence and then write only the fields before it. */
    uint64_t read_sequence;
    uint64_t applied_sequence;
} OmniUsdLiveStatus;

typedef struct OmniUsdLiveStatusEvent {
    uint32_t struct_size;
    uint32_t abi_version;
    OmniUsdLiveEndpoint endpoint;
    OmniUsdLiveSession session;
    const char* uri;
    const OmniUsdLiveStatus* status;
} OmniUsdLiveStatusEvent;

/* Pull-observer output. Initialize the outer header only. The inline status is
 * the final tail record; no future outer fields may be added after it. URI bytes
 * are returned separately, so this value has no borrowed pointer lifetime.
 * struct_size may end at the inline status.buffered_edit_count or
 * status.read_sequence; the output then holds the status prefix before it, and
 * status.struct_size reports that prefix. */
typedef struct OmniUsdLiveStatusObservation {
    uint32_t struct_size;
    uint32_t abi_version;
    OmniUsdLiveEndpoint endpoint;
    OmniUsdLiveSession session;
    OmniUsdLiveStatus status;
} OmniUsdLiveStatusObservation;

typedef struct OmniUsdLiveRequestError {
    uint32_t struct_size;
    uint32_t abi_version;
    OmniUsdLiveResult result;
    uint32_t operation;
    uint32_t grpc_status_present;
    int32_t grpc_status;
    uint64_t request_id;
    uint64_t attempt_id;
    uint64_t incident_id;
    char detail[512]; /* Always terminated, sanitized; empty by default. */
} OmniUsdLiveRequestError;

typedef struct OmniUsdLiveDiagnosticField {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* name; /* Stable catalog key, never an arbitrary metric label. */
    uint32_t kind;
    uint64_t unsigned_value; /* Also stores BOOLEAN 0/1. */
    int64_t signed_value;
    const char* string_value;
} OmniUsdLiveDiagnosticField;

typedef struct OmniUsdLiveDiagnosticRecord {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t schema_version;
    uint32_t event_code;
    const char* event_name;
    uint32_t severity;
    uint64_t timestamp_ms;
    uint64_t elapsed_ms;
    uint64_t runtime_instance_id;
    uint64_t sequence;
    OmniUsdLiveEndpoint endpoint;
    uint64_t subject_id;
    uint64_t incident_id;
    uint32_t field_count;
    const OmniUsdLiveDiagnosticField* const* fields;
} OmniUsdLiveDiagnosticRecord;

typedef struct OmniUsdLiveDiagnosticOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    OmniUsdLiveEndpoint endpoint; /* Zero observes all. */
    uint32_t min_severity;
    uint32_t flags; /* Resource/detail opt-in, restricted by runtime options. */
    uint32_t event_code_count; /* Zero observes all event codes. */
    const uint32_t* event_codes;
} OmniUsdLiveDiagnosticOptions;

typedef struct OmniUsdLiveDiagnosticExportOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t max_records; /* Zero omits recent records; status/metrics remain. */
    uint64_t max_output_bytes; /* Zero selects 16 MiB; larger values clamp to that hard bound. */
} OmniUsdLiveDiagnosticExportOptions;

/* ParseUri writes NUL-terminated decoded strings into the caller's buffer.
 * Offsets index that buffer; session URIs contain an empty layer-name string.
 * Output offsets are valid only on OK and until the caller changes the buffer. */
typedef struct OmniUsdLiveUriParts {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t authority_offset;
    uint64_t session_id_offset;
    uint64_t layer_name_offset;
    uint64_t canonical_uri_offset;
    uint32_t is_layer;
} OmniUsdLiveUriParts;

typedef void(OMNI_USD_LIVE_CALL* OmniUsdLiveAuthRequiredCallback)(void*, const OmniUsdLiveAuthRequired*);
typedef void(OMNI_USD_LIVE_CALL* OmniUsdLiveResultCallback)(void*, OmniUsdLiveResult);
typedef void(OMNI_USD_LIVE_CALL* OmniUsdLiveListSessionsCallback)(void*, OmniUsdLiveResult, uint64_t,
    const OmniUsdLiveSessionInfo* const*);
typedef void(
    OMNI_USD_LIVE_CALL* OmniUsdLiveCreateSessionCallback)(void*, OmniUsdLiveResult, const OmniUsdLiveCreatedSession*);
typedef void(OMNI_USD_LIVE_CALL* OmniUsdLiveStatLayerCallback)(void*, OmniUsdLiveResult, const OmniUsdLiveLayerInfo*);
typedef void(OMNI_USD_LIVE_CALL* OmniUsdLiveSessionCallback)(void*, const OmniUsdLiveSessionEvent*);
typedef void(OMNI_USD_LIVE_CALL* OmniUsdLiveStatusCallback)(void*, const OmniUsdLiveStatusEvent*);
typedef void(OMNI_USD_LIVE_CALL* OmniUsdLiveDiagnosticCallback)(void*, const OmniUsdLiveDiagnosticRecord*);

// clang-format off
/* Successful submission may invoke its callback before returning. All failed
 * submissions set output handles to zero and never schedule a callback.
 * Status-query OK describes the query, not necessarily the subject's result. */
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetAbiVersion(uint32_t* version);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetCapabilities(OmniUsdLiveCapabilities* capabilities);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveInitialize(const OmniUsdLiveRuntimeOptions* options);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveShutdown(uint32_t timeout_ms);

/* Each successful registration acquires a reference, even for the same handle.
 * Nonfinal unregister only releases that reference. Final unregister returns
 * BUSY at once, without waiting and keeping the reference for retry, while any
 * layer, session, request or endpoint-scoped callback registration depends on
 * the endpoint. timeout_ms bounds only draining endpoint-owned internal work,
 * and the endpoint owns none beyond its dependents. Never unregister a
 * borrowed handle from an event. */
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveRegisterEndpoint(
    const OmniUsdLiveEndpointOptions* options, OmniUsdLiveEndpoint* endpoint);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveUnregisterEndpoint(
    OmniUsdLiveEndpoint endpoint, uint32_t timeout_ms);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetAuthState(
    OmniUsdLiveEndpoint endpoint, OmniUsdLiveAuthState* state);
/* Replacement copies credentials and succeeds only for expected_generation.
 * Empty tokens revoke local eligibility and request refresh. Failure leaves
 * new_generation zero. A new token alone never establishes service readiness. */
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveSetAccessToken(
    OmniUsdLiveEndpoint endpoint, uint64_t expected_generation, const char* token,
    uint64_t expires_at_ms, uint64_t* new_generation);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveRegisterAuthRequiredCallback(
    OmniUsdLiveEndpoint endpoint, void* user_data, OmniUsdLiveAuthRequiredCallback callback,
    OmniUsdLiveCallbackHandle* handle);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveFailAuthRefresh(
    OmniUsdLiveEndpoint endpoint, uint64_t challenge_id);

OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveListSessions(
    OmniUsdLiveEndpoint endpoint, const char* base_layer_key, void* user_data,
    OmniUsdLiveListSessionsCallback callback, OmniUsdLiveRequest* request);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveCreateSession(
    OmniUsdLiveEndpoint endpoint, const char* base_layer_key, const char* name, void* user_data,
    OmniUsdLiveCreateSessionCallback callback, OmniUsdLiveRequest* request);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveDeleteSession(
    const char* session_uri, void* user_data, OmniUsdLiveResultCallback callback, OmniUsdLiveRequest* request);
/* Each join owns an independent participant and retains its endpoint. OK means
 * admitted; Connection reports readiness/failure asynchronously. Event strings
 * and bytes are borrowed only for the callback. */
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveJoinSession(
    const char* session_uri, const OmniUsdLiveJoinOptions* options, void* user_data,
    OmniUsdLiveSessionCallback callback, OmniUsdLiveSession* session);
/* OK releases the handle after join callbacks and transport are quiescent.
 * BUSY/TIMEOUT retains ownership for retry. Send-request callbacks have separate
 * request ownership and must be waited/cancelled/released independently. */
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveLeaveSession(
    OmniUsdLiveSession session, uint32_t timeout_ms);
/* Copies bytes before returning. Success and its request callback mean local
 * queue acceptance only, never peer delivery. Messages are nondurable: failed
 * in-flight replay may duplicate delivery; Leave/terminal failure discards the
 * pending queue. Cancelling/releasing the receipt cannot retract accepted bytes. */
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveSendMessage(
    OmniUsdLiveSession session, const void* bytes, uint64_t size, void* user_data,
    OmniUsdLiveResultCallback callback, OmniUsdLiveRequest* request);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveStatLayer(
    const char* layer_uri, void* user_data, OmniUsdLiveStatLayerCallback callback, OmniUsdLiveRequest* request);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveDeleteLayer(
    const char* layer_uri, void* user_data, OmniUsdLiveResultCallback callback, OmniUsdLiveRequest* request);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetEndpointStatus(
    OmniUsdLiveEndpoint endpoint, OmniUsdLiveStatus* status);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetSessionStatus(
    OmniUsdLiveSession session, OmniUsdLiveStatus* status);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetLayerStatus(
    const char* layer_uri, OmniUsdLiveStatus* status);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveRegisterStatusCallback(
    OmniUsdLiveEndpoint endpoint, void* user_data, OmniUsdLiveStatusCallback callback,
    OmniUsdLiveCallbackHandle* handle);

/* Retained pull observers share callback-observer limits and endpoint leases.
 * Endpoint zero observes all subjects. UnregisterCallback cancels/wakes reads
 * and releases retained observations after quiescence. Callback observers cannot
 * be used with ReadStatus/AcknowledgeStatus (UNSUPPORTED).
 * Read selects one immutable observation until its exact subject/revision is
 * acknowledged. Newer changes coalesce independently. NULL uri is a successful
 * size query; size includes NUL. Size queries/BUFFER_TOO_SMALL set uri_size only,
 * leave observation/buffer unchanged, and never acknowledge. A complete copy
 * returns OK and still retains ownership until explicit acknowledgment.
 * Empty reads return TIMEOUT at timeout zero; cancellation returns CANCELLED.
 * A callback worker cannot block on an empty observer (BUSY). A mismatched or
 * stale acknowledgment is INVALID_ARGUMENT and does not release current state.
 * In particular CLOSED subjects retain capacity until ACK or unsubscribe. */
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveCreateStatusObserver(
    OmniUsdLiveEndpoint endpoint, OmniUsdLiveCallbackHandle* observer);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveReadStatus(
    OmniUsdLiveCallbackHandle observer, uint32_t timeout_ms, OmniUsdLiveStatusObservation* observation,
    char* uri, uint64_t* uri_size);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveAcknowledgeStatus(
    OmniUsdLiveCallbackHandle observer, uint64_t subject_id, uint64_t revision);

OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveWaitForPendingUpdates(uint32_t timeout_ms);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveConfigureJitterReduction(
    OmniUsdLiveEndpoint endpoint, const OmniUsdLiveJitterParams* params);

/* Wait returns OK when the finite operation and its callback have completed,
 * even when the operation failed; GetRequestStatus supplies that outcome.
 * Wait timeout only stops waiting, never cancels the operation. timeout_ms==0
 * polls without waiting for Wait/drain/cancel/leave/unregister/shutdown; cancel
 * and teardown still initiate the requested action. Deadlines bound elapsed time,
 * not just one attempt. Self-dependent Wait/drain returns BUSY without blocking.
 *
 * Cancellation/leave/unregister establish callback quiescence on OK. TIMEOUT
 * retains ownership for retry. Self-cancellation returns BUSY until callback
 * exit. ReleaseRequest requests cancellation and returns BUSY if it cannot yet
 * establish quiescence; use Cancel with a bounded deadline, then retry release.
 * Blocking waits from a runtime callback worker on any unfinished request
 * return BUSY to prevent callback-pool starvation. Quiescence includes callback
 * payload cleanup on the executor, which can keep Cancel pending after delivery
 * has been suppressed. GetRequestStatus/GetRequestError return BUSY while
 * pending and write output only on OK. */
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveWait(
    OmniUsdLiveRequest request, uint32_t timeout_ms);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveCancel(
    OmniUsdLiveRequest request, uint32_t timeout_ms);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveReleaseRequest(OmniUsdLiveRequest request);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetRequestStatus(
    OmniUsdLiveRequest request, OmniUsdLiveResult* result);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetRequestError(
    OmniUsdLiveRequest request, OmniUsdLiveRequestError* info);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveUnregisterCallback(
    OmniUsdLiveCallbackHandle handle, uint32_t timeout_ms);

/* All buffer sizes count bytes INCLUDING the terminating NUL(s). NULL buffer is
 * a successful size query. Insufficient buffer returns BUFFER_TOO_SMALL, updates
 * size, and writes no partial output. Strings are UTF-8. URI helpers are pure and
 * require no registered endpoint or initialized runtime. MakeLayerUri takes a
 * session URI and decoded layer name; never concatenate an unescaped name. */
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveMakeSessionUri(
    const char* authority, const char* session_id, char* buffer, uint64_t* size);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveMakeLayerUri(
    const char* session_uri, const char* layer_name, char* buffer, uint64_t* size);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveParseUri(
    const char* uri, OmniUsdLiveUriParts* parts, char* buffer, uint64_t* size);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveCombineWithBaseUri(
    const char* base_uri, const char* relative_layer_name, char* buffer, uint64_t* size);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetResultString(
    OmniUsdLiveResult result, const char** text);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveRegisterDiagnosticCallback(
    const OmniUsdLiveDiagnosticOptions* options, void* user_data, OmniUsdLiveDiagnosticCallback callback,
    OmniUsdLiveCallbackHandle* handle);
OMNI_USD_LIVE_API OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveCopyDiagnostics(
    OmniUsdLiveEndpoint endpoint, const OmniUsdLiveDiagnosticExportOptions* options,
    char* buffer, uint64_t* size);
// clang-format on

#ifdef __cplusplus
}
#endif
#endif /* OMNI_USD_LIVE_LIVE_H */
