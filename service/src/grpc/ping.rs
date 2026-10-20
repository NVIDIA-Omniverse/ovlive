// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::domain::time::now_micros;

use super::proto::{PingRequest, PingResponse};

/// `PingResponse.operation_ids`: this build honours
/// `CreateLayerHeader.operation_id` and `UpdateLayerHeader.operation_id`.
///
/// A build constant, never configuration: it describes the answering
/// replica's code, which any caller with a well-formed bearer may read (Ping
/// checks no permission) and anyone can read from the served schema anyway.
pub const PING_OPERATION_IDS: bool = true;

/// `PingResponse.chained_writes`: this build accepts
/// `UpdateLayerRequest.chained_header`, `ChainedUpdateLayerHeader.no_wait`
/// included. A build constant, as [`PING_OPERATION_IDS`].
pub const PING_CHAINED_WRITES: bool = true;

/// Handle a Ping RPC.
///
/// Echoes `client_timestamp_microseconds_since_epoch` unchanged, sets
/// `server_timestamp_microseconds_since_epoch` to the current wall-clock time,
/// and reports this build's capability bits ([`PING_OPERATION_IDS`],
/// [`PING_CHAINED_WRITES`]). Pure computation — no auth, no Valkey.
pub fn handle_ping(request: PingRequest) -> PingResponse {
    // b.k1r medium: consolidated to `domain::time::now_micros` so the
    // Ping reply uses the same saturating helper as session / layer
    // writes instead of a local `SystemTime::now()...unwrap_or_default()`.
    let server_ts = now_micros();

    PingResponse {
        client_timestamp_microseconds_since_epoch: request
            .client_timestamp_microseconds_since_epoch,
        server_timestamp_microseconds_since_epoch: server_ts,
        operation_ids: PING_OPERATION_IDS,
        chained_writes: PING_CHAINED_WRITES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn handle_ping_echoes_client_timestamp_exactly() {
        let client_ts: u64 = 1_700_000_000_000_000;
        let req = PingRequest {
            client_timestamp_microseconds_since_epoch: client_ts,
        };
        let resp = handle_ping(req);
        assert_eq!(
            resp.client_timestamp_microseconds_since_epoch, client_ts,
            "client_timestamp must be echoed exactly"
        );
    }

    #[test]
    fn handle_ping_server_timestamp_is_plausible() {
        let req = PingRequest {
            client_timestamp_microseconds_since_epoch: 0,
        };
        // Capture timestamps tightly around the call
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        let resp = handle_ping(req);
        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        assert!(
            resp.server_timestamp_microseconds_since_epoch > 0,
            "server_timestamp must be > 0"
        );
        assert!(
            resp.server_timestamp_microseconds_since_epoch >= before,
            "server_timestamp must be >= time before call"
        );
        assert!(
            resp.server_timestamp_microseconds_since_epoch <= after,
            "server_timestamp must be <= time after call"
        );
    }

    #[test]
    fn handle_ping_client_timestamp_zero_is_valid() {
        let req = PingRequest {
            client_timestamp_microseconds_since_epoch: 0,
        };
        let resp = handle_ping(req);
        assert_eq!(
            resp.client_timestamp_microseconds_since_epoch, 0,
            "client_ts=0 must be echoed as 0"
        );
        assert!(
            resp.server_timestamp_microseconds_since_epoch > 0,
            "server_timestamp must still be set when client_ts=0"
        );
    }

    #[test]
    fn handle_ping_client_timestamp_u64_max_echoed_exactly() {
        let client_ts = u64::MAX;
        let req = PingRequest {
            client_timestamp_microseconds_since_epoch: client_ts,
        };
        let resp = handle_ping(req);
        assert_eq!(
            resp.client_timestamp_microseconds_since_epoch, client_ts,
            "u64::MAX must be echoed without truncation"
        );
    }

    /// b.pav: this build honours operation ids and accepts chained writes, so
    /// every Ping reply reports both bits true — a client reads them to decide
    /// whether to chain against the replica that answered.
    #[test]
    fn handle_ping_reports_operation_ids_and_chained_writes() {
        let resp = handle_ping(PingRequest {
            client_timestamp_microseconds_since_epoch: 1,
        });
        assert_eq!(
            (resp.operation_ids, resp.chained_writes),
            (true, true),
            "both capability bits are build constants of this replica: true"
        );
    }
}
