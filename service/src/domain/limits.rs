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

//! Hard caps on delta-payload dimensions enforced by `apply_delta`.
//!
//! The OODF FlatBuffers verifier and the OMPF `max_layer_size` cap (100 MB by
//! default) bound the container size, but nothing inside the delta-application
//! engine caps per-primitive counts or per-name byte lengths. Without these
//! limits a single 100 MB layer can carry one 99 MB `key_name` or a
//! `ReorderChildren` command with tens of millions of ids — both of which cause
//! pathological work inside the engine (O(N) rebuilds, HashMap blow-up) even
//! though the wire format is legal.
//!
//! These constants are enforced by [`validate_delta_bounds`](crate::domain::validation::validate_delta_bounds):
//! first by the `CreateLayer` / `UpdateLayer` handlers right after the parse,
//! before the upload's processing charge, then again at the top of
//! `apply_delta` and of `LayerStore::update_layer_with_operation` as the
//! backstop for every other caller. They are intentionally generous relative
//! to realistic workloads (a mid-sized scene carries thousands of
//! commands/fields/samples per update, not millions). The one apply that skips
//! the entry-count cap is a checkpoint base decode (`ApplyMode::RestoreBase`),
//! which rebuilds a whole layer; every other rule here still runs there.
//! Documented in SDD §9.

/// Max entries — struct commands + set fields + time samples, summed — one
/// delta may carry: the OODF verifier's table cap
/// ([`oodf_ffi::MAX_DELTA_TABLES`]) less the delta's root table, since the
/// schema has no nested tables. Counted by
/// [`own_element_count`](crate::domain::delta::own_element_count), the one
/// definition of a delta's entries.
///
/// It bounds one inbound delta, never a layer and never a catch-up: a layer is
/// built from as many deltas as its writers send, and a catch-up is one delta
/// of whatever size the layer has. The parser refuses an over-limit delta
/// before its verifier walks it (`oodf_ffi::OodfError::TooManyTables`), and
/// [`validate_delta_bounds`](crate::domain::validation::validate_delta_bounds)
/// refuses it again for any delta that did not come through the parser. A
/// checkpoint base over it is written as several chunks of at most this many
/// entries each (`checkpoint::chunked_base`), so it stays decodable; on-access
/// restore refuses such a base (SR-14.7).
pub const MAX_DELTA_ENTRIES: u64 = oodf_ffi::MAX_DELTA_TABLES as u64 - 1;

/// Max number of section-id entries in a `CommandReorderChildren::children_list`.
pub const MAX_CHILDREN_LIST_LEN: usize = 100_000;

/// Max byte length of any name string inside a delta (`section_name`,
/// `key_name`, `new_name`).
///
/// Byte length is checked — not `chars().count()` — because Valkey keys,
/// Rust HashMap shard cost, and downstream storage all treat names as bytes.
pub const MAX_NAME_BYTES: usize = 4096;

/// Maximum `delta.set_order` value carried by a `SetField` primitive.
///
/// Equal to `2^53 - 1`, the Lua f64 integer-precision ceiling enforced by
/// the `update_layer_fast.lua` fast path. Bounding the per-SetField input
/// here keeps a single malicious delta from blowing past the ceiling on
/// one shot; the runtime ceiling check in `apply_set_field` still catches
/// the cumulative case where a long-lived object accumulates set_order
/// across many writes (b.qw6 item 2).
///
/// `(2^53 - 1)` rather than `u64::MAX` because the fast path stores
/// `set_order` through Lua's `number` (IEEE 754 f64) and silently rounds
/// integers past 2^53; the slow path is pure Rust u64 but we cap both
/// backends at the same ceiling for byte-identical parity.
pub const MAX_DELTA_SET_ORDER: u64 = (1u64 << 53) - 1;

/// Hard cap on section-tree depth traversed by
/// [`DiffEngine::sections_in_dfs_order`](crate::domain::diff::DiffEngine).
///
/// A linear parent → child chain longer than this produces
/// [`DiffError::SectionTreeTooDeep`](crate::domain::diff::DiffError::SectionTreeTooDeep)
/// rather than blowing the stack. The recursive walker it replaced would
/// crash the process on chains of ~10K–80K sections (all trivially within
/// the 100 MB layer cap). The iterative walker has no inherent depth bound,
/// so this cap primarily guards against producer bugs / malicious input that
/// would otherwise drive an unbounded visited-set allocation.
pub const MAX_SECTION_TREE_DEPTH: usize = 10_000;
