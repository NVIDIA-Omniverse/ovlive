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

//! A reply is a cluster redirect only when it is an error frame.
//!
//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
//!
//! A cluster node refuses a command it does not serve with an error reply, `-MOVED <slot> <host:port>` or
//! `-ASK <slot> <host:port>`. Stock `clustered::process_response_frame` (`src/router/clustered.rs:322`) asked
//! redis-protocol 6.0.0's `Resp3Frame::is_redirection` (`src/resp3/types.rs:894-897`). That reads the frame's text
//! through `as_str` (`:1878-1889`) and accepts any text that starts `MOVED` or `ASK` and has exactly three
//! space-separated tokens (`src/utils.rs:66-73`). `as_str` answers for data frames too (`SimpleString`, `BlobString`,
//! `BigNumber`, `VerbatimString`, `ChunkedString`), so a data reply whose value reads `ASK 1234 10.0.0.9:6379` was
//! routed as a redirect and never reached its caller. Stock fred then re-synced the cluster and re-sent the command
//! (`MOVED`), or sent `ASKING` to the named node when connected to it and otherwise slept and failed (`ASK`). With
//! this patch's [`super::ask::follow`], a command with write attempts left opens a connection to the named host, with
//! the client's store credentials, when the client has none.
//!
//! The RESP type, not the text, is the protocol's signal that a frame is a server refusal and not data. So
//! [`is_redirection`] requires an error frame and then applies stock's text rule to it unchanged (a case-sensitive
//! `MOVED` or `ASK` prefix and exactly three tokens): every real `-MOVED` and `-ASK` is handled exactly as before,
//! and a data reply reaches its caller as data, whatever it reads.
//!
//! The types cover both protocols. Under RESP2 a server error (`-…`) arrives as `SimpleError`, a status reply (`+…`)
//! as `SimpleString` and a bulk string (`$…`) as `BlobString`, both through this patch's conversion for a connection
//! that never subscribed (`liveness::resp2_as_data`) and through redis-protocol's `into_resp3`
//! (`src/resp2/types.rs:679-682`). Under RESP3 a server error arrives as `SimpleError` or `BlobError`.
//!
//! fred's other redirect paths already decided by type and are unchanged. The transaction router converts a reply
//! with `protocol::utils::frame_to_error` (`src/protocol/utils.rs:493-502`, error frames only) and then asks
//! `Error::is_moved` / `Error::is_ask` (`src/error.rs:418-426`; `src/router/transactions.rs:77-79`, `:203`, `:222`).
//! The cluster-wide reply parser uses `frame_to_error` too (`src/router/clustered.rs:85`). The centralized and
//! sentinel routers handle no redirects.
//!
//! To revert: put `frame.is_redirection()` back in `clustered.rs:322`'s condition, then delete this module, its
//! `pub mod` line and its bullet in `live_edit/mod.rs`.

use redis_protocol::resp3::types::{BytesFrame as Resp3Frame, Resp3Frame as _};

/// Whether `frame`, a reply read from a cluster node, is a `MOVED` or `ASK` redirect: an error frame (`SimpleError` or
/// `BlobError`) whose text redis-protocol's redirect rule accepts. Any other frame is data, whatever its text.
pub fn is_redirection(frame: &Resp3Frame) -> bool {
  matches!(frame, Resp3Frame::SimpleError { .. } | Resp3Frame::BlobError { .. }) && frame.is_redirection()
}
