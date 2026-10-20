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

pub mod chain_gate;
pub mod checkout;
pub mod client;
pub mod commit_chain;
pub mod commit_charge;
pub mod connection_buffers;
pub mod counters;
pub mod diff_read;
pub mod ext_backup;
pub mod ext_index_audit;
pub mod ext_reconcile;
pub mod ext_recovery_singleflight;
pub mod ext_release_gate;
pub mod external_values;
pub mod fanout_thread;
pub mod iam_credentials;
pub mod keys;
pub mod layer;
pub mod leader_lease;
pub mod log_gate;
pub mod lua;
pub mod member_replacement;
pub mod memory_budget;
pub mod no_refresh_credentials;
pub mod notifications;
pub mod orphan_sweep;
pub mod presence;
pub mod relay_retention;
pub mod retirement;
pub mod retry;
pub mod router_supervisor;
pub mod session;
pub mod snapshot_fence;
pub mod snapshot_flights;
pub mod store_memory;
pub mod store_oom;
pub mod stream_mux;
pub mod structural;
pub mod structural_cache;
pub mod structural_cache_audit;
pub mod structural_cache_cold_start;
pub mod structural_cache_fenced;
pub mod structural_cache_read;
pub mod structural_cache_repair;
pub mod structural_cache_sync;
pub mod structural_size_gate;
pub mod subscriber_admission;
pub mod tail_slice;
pub mod time_sample_merge;
pub mod token_file_credentials;
pub mod txn_checkout;
pub mod watch_guard;
pub mod watched_txn;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

pub use checkout::{CheckedOutClient, ExclusiveCheckout};
pub use ext_recovery_singleflight::ExtRecoverySingleflight;
pub use subscriber_admission::{AdmissionRefusal, SubscriberAdmission, SubscriberAdmissionGate};

// Re-export notification types for handler use.
pub use notifications::{
    consume_layer_stream, reserve_layer_stream, ConsumeError, LayerNotification,
    LayerStreamConsumer, NotificationParseError, ReservedStream, StreamCursor,
};
