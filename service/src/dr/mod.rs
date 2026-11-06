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

//! Disaster recovery for a lost backing store: the fleet hold, and the
//! operator restore and accept-loss Jobs.
//!
//! When the store loses state, every replica holds: it refuses LiveService
//! RPCs and writes nothing until an operator Job restores the store, or
//! accepts the loss, and removes the hold marker. Design and rationale: the
//! hold and Job sections of SDD §22.
//!
//! # Layout
//!
//! - [`hold`] — the hold's data and its store reads: the verdict a replica
//!   holds on, the poll that keeps it current, the one-shot startup
//!   evaluation, the shared timing constants, the bounds on a held replica's
//!   last store write after entry (`hold::hold_write_bound` and
//!   `hold::join_teardown_write_bound`), and the inventory of
//!   the store- and backup-writing tasks a hold aborts.
//! - [`job`] — the restore and accept-loss Jobs (`live-edit dr restore`,
//!   `live-edit dr accept-loss`): their command line, lease, precondition,
//!   quiet period, write permit and absence rule, around the
//!   restore and accept-loss algorithms, the restore plan with its
//!   coverage check and accept-loss's lost set (`job::plan`), and the
//!   restore writer and accept-loss's discards (`job::execute`), whose store
//!   writes live in `storage` (see *Dependency edges*). Its module docs give
//!   the Job flow.
//!
//! The server's reactions to a verdict (the gate in front of the LiveService
//! RPCs, the stream endings, the abort of the writer inventory, held-mode
//! startup) live with the server, in `crate::grpc` and `main.rs`, and read
//! this module's verdict.
//!
//! # Dependency edges
//!
//! `grpc → dr`, and `dr → {storage, checkpoint, observability, config,
//! domain}`. `dr` never depends on `grpc`: the server consumes the verdict,
//! never the other way round. A bound the server defines that a Job's quiet
//! period counts (graceful shutdown's) is read through `crate::config`
//! (`Config::minimum_termination_grace`), never from `grpc`. The
//! marker key, its field names and the hold-poll script live in `storage`
//! (`storage::keys::KeySchema::dr_hold_marker`, `storage::dr_hold`,
//! `scripts/dr_hold_poll.lua`), and so do the Jobs' store reads
//! (`storage::dr_job`), their lease scripts (`storage::external_values`),
//! the restore writer's and accept-loss's store writes (the mapping
//! compare-and-sets and the path-index members in `storage::dr_write`, the
//! object writes and the value removal in `storage::layer::dr_restore`) and
//! the stream high-water probe they share
//! with the on-access restore (`storage::stream_high_water`). So every
//! `EVALSHA` stays under the storage lane's source fences.

pub mod hold;
pub mod job;
