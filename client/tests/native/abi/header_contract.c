/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
#include <omni/usd/live/Live.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>

#if defined(__cplusplus)
#    define STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#    define STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif
STATIC_ASSERT(sizeof(OmniUsdLiveResult) == 4, "result must be fixed width");
STATIC_ASSERT(sizeof(OmniUsdLiveEndpoint) == 8, "handle must be fixed width");
STATIC_ASSERT(offsetof(OmniUsdLiveStatus, struct_size) == 0, "size prefix");
STATIC_ASSERT(offsetof(OmniUsdLiveStatus, abi_version) == 4, "version prefix");
STATIC_ASSERT(offsetof(OmniUsdLiveStatus, buffered_edit_count) == 112, "buffered_edit_count appends to the 1.0 layout");
STATIC_ASSERT(offsetof(OmniUsdLiveStatus, read_sequence) == 120, "read_sequence appends after buffered_edit_count");
STATIC_ASSERT(offsetof(OmniUsdLiveStatus, applied_sequence) == 128, "applied_sequence follows read_sequence");
STATIC_ASSERT(sizeof(OmniUsdLiveStatus) == 136, "status layout ends after applied_sequence");
STATIC_ASSERT(offsetof(OmniUsdLiveStatusObservation, status) + offsetof(OmniUsdLiveStatus, buffered_edit_count) == 136,
              "observation prefix before buffered_edit_count");
STATIC_ASSERT(offsetof(OmniUsdLiveStatusObservation, status) + offsetof(OmniUsdLiveStatus, read_sequence) == 144,
              "observation prefix before read_sequence");
STATIC_ASSERT(offsetof(OmniUsdLiveRuntimeOptions, apply_lag_warning_threshold) == 136,
              "apply_lag_warning_threshold appends to the 1.0 runtime options");
STATIC_ASSERT(offsetof(OmniUsdLiveRuntimeOptions, reconnect_initial_ms) == 144,
              "the reconnect options append after apply_lag_warning_threshold");
STATIC_ASSERT(offsetof(OmniUsdLiveRuntimeOptions, reconnect_maximum_ms) == 148, "reconnect_maximum_ms follows");
STATIC_ASSERT(offsetof(OmniUsdLiveRuntimeOptions, reconnect_max_attempts) == 152, "reconnect_max_attempts follows");
STATIC_ASSERT(sizeof(OmniUsdLiveRuntimeOptions) == 160, "runtime options end after reconnect_max_attempts");
STATIC_ASSERT(offsetof(OmniUsdLiveCapabilities, autonomous_progress) == 16, "capabilities carry no recovery schema field");
STATIC_ASSERT(sizeof(OmniUsdLiveCapabilities) == 20, "capabilities layout ends after autonomous_progress");
STATIC_ASSERT(offsetof(OmniUsdLiveJoinOptions, app_name) == 8, "join ABI prefix");
STATIC_ASSERT(offsetof(OmniUsdLiveAuthRequired, grpc_status) == 40, "auth challenge ends with grpc_status");
STATIC_ASSERT(sizeof(OmniUsdLiveAuthRequired) == 48, "auth challenge size is padded to 8 bytes");
/* Retired values stay reserved gaps: result 17, state 6, operation 15 and record 23.
 * Their neighbors keep their numbers. */
STATIC_ASSERT(OMNI_USD_LIVE_OBJECT_REPLACED == 16 && OMNI_USD_LIVE_BUFFER_TOO_SMALL == 18, "result 17 is reserved");
STATIC_ASSERT(OMNI_USD_LIVE_IO_ERROR == 21, "last result");
STATIC_ASSERT(OMNI_USD_LIVE_STATE_BLOCKED == 5 && OMNI_USD_LIVE_STATE_FAILED == 7, "state 6 is reserved");
STATIC_ASSERT(OMNI_USD_LIVE_STATE_CLOSED == 9, "last state");
STATIC_ASSERT(OMNI_USD_LIVE_OPERATION_SHUTDOWN == 14, "operation 15 is reserved");
STATIC_ASSERT(OMNI_USD_LIVE_RECORD_RECORDS_SUPPRESSED == 22 && OMNI_USD_LIVE_RECORD_MESSAGE_ACCEPTED == 24,
              "record 23 is reserved");
STATIC_ASSERT(OMNI_USD_LIVE_RECORD_APPLY_LAG_EXITED == 35, "last record");

/* Model a later same-major provider extending each record independently.
 * Consumer sees the old prefix; pointer arrays preserve element addressing. */
struct ExtendedSessionInfo {
    OmniUsdLiveSessionInfo prefix;
    uint64_t future_field[7];
};
struct ExtendedDiagnosticField {
    OmniUsdLiveDiagnosticField prefix;
    uint64_t future_field[11];
};

static void OMNI_USD_LIVE_CALL consume_sessions(
    void* user_data, OmniUsdLiveResult result, uint64_t count, const OmniUsdLiveSessionInfo* const* infos)
{
    int* called = (int*)user_data;
    assert(result == OMNI_USD_LIVE_OK);
    assert(count == 2);
    assert(strcmp(infos[0]->session_id, "first") == 0);
    assert(strcmp(infos[1]->session_id, "second") == 0);
    assert(infos[1]->created_time_ms == 123);
    *called = 1;
}

int main(void)
{
    struct ExtendedSessionInfo first;
    struct ExtendedSessionInfo second;
    struct ExtendedDiagnosticField first_field;
    struct ExtendedDiagnosticField second_field;
    OmniUsdLiveDiagnosticRecord record;
    const OmniUsdLiveSessionInfo* sessions[2];
    const OmniUsdLiveDiagnosticField* fields[2];
    OmniUsdLiveListSessionsCallback callback = consume_sessions;
    int called = 0;
    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));
    memset(&first_field, 0, sizeof(first_field));
    memset(&second_field, 0, sizeof(second_field));
    memset(&record, 0, sizeof(record));
    first.prefix.struct_size = sizeof(first);
    second.prefix.struct_size = sizeof(second);
    first.prefix.session_id = "first";
    second.prefix.session_id = "second";
    second.prefix.created_time_ms = 123;
    sessions[0] = &first.prefix;
    sessions[1] = &second.prefix;
    callback(&called, OMNI_USD_LIVE_OK, 2, sessions);
    assert(called);
    first_field.prefix.struct_size = sizeof(first_field);
    second_field.prefix.struct_size = sizeof(second_field);
    first_field.prefix.name = "first_field";
    second_field.prefix.name = "second_field";
    second_field.prefix.unsigned_value = 456;
    fields[0] = &first_field.prefix;
    fields[1] = &second_field.prefix;
    record.fields = fields;
    record.field_count = 2;
    assert(strcmp(record.fields[1]->name, "second_field") == 0);
    assert(record.fields[1]->unsigned_value == 456);
    return 0;
}
