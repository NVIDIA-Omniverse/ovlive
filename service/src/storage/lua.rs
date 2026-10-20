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

// Lua EVAL fast path for `update_layer` (b.67u).
//
// Collapses the legacy WATCH/HGETALL/MULTI/EXEC client-side OCC loop to a
// single atomic server-side `EVALSHA` for deltas that touch only fields and
// time samples (the overwhelming majority of updates under sustained
// collaborative editing). Structural commands (reset, non-ignored
// `StructCommand`s) take the same script's structural mode on the structural
// route, or the WATCH path — see `is_fast_path` for the precise
// classification and "Structural mode" below.
//
// See `src/storage/scripts/update_layer_fast.lua` for the script itself and
// its argument contract.
//
// ── TOCTOU-free sequence allocation (why the script owns `new_seq`) ─────
//
// An earlier draft of this path had the Rust side do an `HMGET seq
// next_set_order` before dispatching `EVALSHA`, compute `new_seq` /
// per-field `version` / per-field `set_order` in a Rust-resident planner,
// and pass the pre-computed values to the script via ARGV. That design
// has a fatal TOCTOU race: concurrent writers each read `seq=N` in their
// own HMGET, each compute `new_seq=N+1`, each call EVAL with the same
// `new_seq`, and the second writer's `XADD <new_seq>-0` fails with
// `ERR The ID specified in XADD is equal or smaller than the target
// stream top item`. Lua atomicity protects the script body from
// interleaving but does NOT freshen a seq that was read outside the
// atomic window.
//
// Fix: every sequence-derived value (`new_seq`, per-entry `version`,
// per-SetField `stored_set_order`, header `next_set_order`) is computed
// INSIDE the Lua script from the header read within the same atomic
// block. The Rust side carries through only TOCTOU-free inputs — the
// OODF delta-level `set_order` offsets, value / hash / size hash-field
// pairs, ZADD / ZREM member strings — and the script does the
// arithmetic. See the script's own `-- The TOCTOU rule` block for the
// matching server-side invariant.
//
// This also makes `apply_delta` unnecessary on the Rust side of the
// fast path: the encoder walks `OwnedDelta` directly and emits the
// tagged ARGV records the script consumes. Net RTT count is still 1
// (one `EVALSHA`; no pre-EVAL HMGET), which is the best case the b.67u
// ticket called out.
//
// ── Structural mode (`encode_mutations`) ─────────────────────────────────
//
// The structural commit (`storage::structural`) is the deliberate exception
// to the rule above: it encodes the `StorageMutation`s `apply_delta` planned
// on a cached state, with Rust-computed absolute values. That is safe there
// because the script's structural mode is fenced (SR-6.2 guard 6): it refuses
// the commit unless the header's `seq` and `commit_chain` still equal the cut
// the plan was made at, so the values cannot be stale when they land. See the
// script's "Structural-mode records" for the record vocabulary.
//
// ── One script, two modes, one shared prefix ─────────────────────────────
//
// Both modes take the same leading KEYS and ARGV
// ([`UPDATE_SCRIPT_SHARED_KEYS`], [`UPDATE_SCRIPT_SHARED_ARGV`]), built by
// one function ([`UpdateScriptPrefix::build`], whose KEYS half is
// [`update_script_shared_keys`]) and followed by the mode slot
// ([`UpdateScriptMode`]); everything mode-specific comes after them. Every
// slot of the prefix has one name per language — the script's step-0
// `local KEY_<X>` / `local ARG_<X>`, and the `UPDATE_SCRIPT_KEY_*` /
// `UPDATE_SCRIPT_ARG_*` constants here — and a compile-time check holds the
// two to the same numbers. The prefix carries plan b.y3m's snapshot-fence
// and writer-intent keys (`KEYS[7]` / `KEYS[8]`), so both modes refuse a
// commit under a live fence at the same step (the script's step 1d), with
// the same `{"FENCED", "<retry_after_ms>"}` reply, b.yuo's commit release
// record (`KEYS[9]`), which both modes write at step 7a, with its TTL
// (`ARGV[16]`), and the object's migration sentinel (`KEYS[10]`, b.qtk), a
// never-written key that makes the store refuse the commit before it starts
// while the object's slot is mid-migration.

use std::sync::LazyLock;

use bytes::Bytes;
use fred::types::scripts::Script;

use crate::domain::delta::StorageMutation;
use crate::domain::types::FieldValue;
use crate::error::ServiceError;
use crate::observability::metrics;
use crate::storage::keys::KeySchema;
use crate::storage::layer::{self, hex_encode_bytes, section_hash_fields, CommitIdentity};
use crate::storage::relay_retention::RelayRetention;
use crate::storage::structural::{PreImage, PreImageDigests};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;

/// The compiled `update_layer` commit script, `update_layer_fast.lua` (named
/// for its first mode): the field / time-sample fast path in its `fast` mode
/// and the fenced structural commit in its `structural` mode
/// ([`UpdateScriptMode`]). Loaded lazily on first use so the process start-up
/// cost is paid only when a commit actually runs.
///
/// `Script::from_lua` is infallible — it just computes the SHA-1 of the Lua
/// source and stores both. The commit sends it through
/// [`evalsha_once_with_reload`], which on the first `NOSCRIPT` sends
/// `SCRIPT LOAD` to the server and then retries the `EVALSHA` once.
pub static UPDATE_LAYER_FAST_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(UPDATE_LAYER_FAST_SOURCE));

/// The source of `scripts/update_layer_fast.lua`, shared by
/// [`UPDATE_LAYER_FAST_SCRIPT`] and the compile-time check below.
const UPDATE_LAYER_FAST_SOURCE: &str = include_str!("scripts/update_layer_fast.lua");

/// Whether `h[at..]` starts with `prefix`.
const fn starts_with_at(h: &[u8], at: usize, prefix: &[u8]) -> bool {
    if at + prefix.len() > h.len() {
        return false;
    }
    let mut i = 0;
    while i < prefix.len() {
        if h[at + i] != prefix[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The byte offsets just past each `\nlocal ` that opens a line of the
/// script, in order, and how many there are. One pass over the source, so the
/// compile-time checks below stay cheap (const evaluation is slow, and the
/// script is over 100 KB). The script has far fewer top-level locals than the
/// array holds; the assertion below says so if that ever changes.
const UPDATE_SCRIPT_LOCALS: ([usize; 512], usize) = {
    let h = UPDATE_LAYER_FAST_SOURCE.as_bytes();
    let lead = b"\nlocal ";
    let mut out = [0usize; 512];
    let mut n = 0;
    let mut at = 0;
    while at < h.len() {
        if h[at] == b'\n' && starts_with_at(h, at, lead) {
            assert!(
                n < out.len(),
                "scripts/update_layer_fast.lua has more top-level locals than the compile-time \
                 check holds; raise UPDATE_SCRIPT_LOCALS' capacity"
            );
            out[n] = at + lead.len();
            n += 1;
        }
        at += 1;
    }
    (out, n)
};

/// The value of the script's `local <name> = <n>` declaration: the decimal
/// after the first line that starts `local <name>` followed by optional
/// spaces and `=`, or `None` when no such line exists or no digit follows the
/// `=`. A line declaring a longer name (`local <name>_X = …`) does not match.
/// `const` so the script/Rust agreement on the shared prefix is a build
/// failure rather than a test.
const fn lua_local(name: &str) -> Option<usize> {
    let (h, n) = (UPDATE_LAYER_FAST_SOURCE.as_bytes(), name.as_bytes());
    let (starts, count) = UPDATE_SCRIPT_LOCALS;
    let mut k = 0;
    while k < count {
        let start = starts[k];
        if starts_with_at(h, start, n) {
            let mut at = start + n.len();
            while at < h.len() && h[at] == b' ' {
                at += 1;
            }
            if at < h.len() && h[at] == b'=' {
                at += 1;
                while at < h.len() && h[at] == b' ' {
                    at += 1;
                }
                let mut value = 0usize;
                let mut digits = 0;
                while at < h.len() && h[at].is_ascii_digit() {
                    value = value * 10 + (h[at] - b'0') as usize;
                    at += 1;
                    digits += 1;
                }
                return if digits == 0 { None } else { Some(value) };
            }
        }
        k += 1;
    }
    None
}

/// The number of lines of the script that start `local <prefix>`.
const fn lua_locals_named(prefix: &str) -> usize {
    let (h, p) = (UPDATE_LAYER_FAST_SOURCE.as_bytes(), prefix.as_bytes());
    let (starts, count) = UPDATE_SCRIPT_LOCALS;
    let mut found = 0;
    let mut k = 0;
    while k < count {
        if starts_with_at(h, starts[k], p) {
            found += 1;
        }
        k += 1;
    }
    found
}

/// Assert at compile time that the script declares `local <name> = <value>`:
/// a shared-prefix slot or length against its `UPDATE_SCRIPT_*` twin, or a
/// commit release ring constant against its `storage::layer` twin.
const fn assert_lua_local(name: &str, value: usize) {
    match lua_local(name) {
        Some(n) => assert!(
            n == value,
            "scripts/update_layer_fast.lua declares a shared-prefix slot or length, or a commit \
             release ring constant, with another value than its Rust twin \
             (storage::lua::UPDATE_SCRIPT_*, storage::layer::COMMIT_RELEASE_RING_*): the Rust \
             caller would place keys or arguments where the script does not read them, or the \
             two backends would write the ring differently. Change both together."
        ),
        None => panic!(
            "scripts/update_layer_fast.lua no longer declares a shared-prefix slot or length, or \
             a commit release ring constant, that its Rust twin is checked against"
        ),
    }
}

/// Every shared-prefix name the script declares (its step 0) and its Rust
/// twin. The compile-time check below holds the two sides to this list.
const UPDATE_SCRIPT_KEY_SLOTS: [(&str, usize); 10] = [
    ("KEY_HEADER", UPDATE_SCRIPT_KEY_HEADER),
    ("KEY_FIELDS_INDEX", UPDATE_SCRIPT_KEY_FIELDS_INDEX),
    ("KEY_TS_INDEX", UPDATE_SCRIPT_KEY_TS_INDEX),
    ("KEY_STREAM", UPDATE_SCRIPT_KEY_STREAM),
    ("KEY_EXT_INDEX", UPDATE_SCRIPT_KEY_EXT_INDEX),
    ("KEY_COMMIT_TOKEN", UPDATE_SCRIPT_KEY_COMMIT_TOKEN),
    ("KEY_FENCE", UPDATE_SCRIPT_KEY_FENCE),
    ("KEY_INTENT", UPDATE_SCRIPT_KEY_INTENT),
    ("KEY_COMMIT_RELEASE", UPDATE_SCRIPT_KEY_COMMIT_RELEASE),
    (
        "KEY_MIGRATION_SENTINEL",
        UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL,
    ),
];
const UPDATE_SCRIPT_ARG_SLOTS: [(&str, usize); 16] = [
    ("ARG_SERVER_TS", UPDATE_SCRIPT_ARG_SERVER_TS),
    ("ARG_CLIENT_ID", UPDATE_SCRIPT_ARG_CLIENT_ID),
    ("ARG_CLIENT_TS", UPDATE_SCRIPT_ARG_CLIENT_TS),
    ("ARG_STREAM_MAXLEN", UPDATE_SCRIPT_ARG_STREAM_MAXLEN),
    ("ARG_OBJECT_ID", UPDATE_SCRIPT_ARG_OBJECT_ID),
    ("ARG_NUM_MUTATIONS", UPDATE_SCRIPT_ARG_NUM_MUTATIONS),
    ("ARG_RELAY_OMPF", UPDATE_SCRIPT_ARG_RELAY_OMPF),
    ("ARG_COMMIT_KEY", UPDATE_SCRIPT_ARG_COMMIT_KEY),
    ("ARG_RECORD_SUFFIX", UPDATE_SCRIPT_ARG_RECORD_SUFFIX),
    ("ARG_WRITER_GUARD", UPDATE_SCRIPT_ARG_WRITER_GUARD),
    (
        "ARG_RELAY_BUDGET_BYTES",
        UPDATE_SCRIPT_ARG_RELAY_BUDGET_BYTES,
    ),
    (
        "ARG_RELAY_GENERATION_BYTES",
        UPDATE_SCRIPT_ARG_RELAY_GENERATION_BYTES,
    ),
    ("ARG_RELAY_GENERATIONS", UPDATE_SCRIPT_ARG_RELAY_GENERATIONS),
    ("ARG_RELAY_MIN_ENTRIES", UPDATE_SCRIPT_ARG_RELAY_MIN_ENTRIES),
    ("ARG_CALL_TOKEN", UPDATE_SCRIPT_ARG_CALL_TOKEN),
    ("ARG_RING_TTL_MS", UPDATE_SCRIPT_ARG_RING_TTL_MS),
];

// The script declares every slot of the shared prefix as
// `local KEY_<X> = <n>` / `local ARG_<X> = <n>` and both lengths as
// `local SHARED_<X> = <n>`, and locates everything mode-specific from the
// lengths. These assertions read every one of those declarations out of the
// script source at compile time, so the Rust builder
// ([`UpdateScriptPrefix::build`]) and the script cannot drift: a slot moved on
// one side, a slot the script adds that Rust does not name (the counts), or a
// slot Rust names that the script dropped, fails the build. The slots must
// also be exactly `1..=length`, each once, so no slot is shared or skipped.
const _: () = {
    let mut seen_keys = [false; UPDATE_SCRIPT_SHARED_KEYS];
    let mut i = 0;
    while i < UPDATE_SCRIPT_KEY_SLOTS.len() {
        let (name, slot) = UPDATE_SCRIPT_KEY_SLOTS[i];
        assert_lua_local(name, slot);
        assert!(
            slot >= 1 && slot <= UPDATE_SCRIPT_SHARED_KEYS && !seen_keys[slot - 1],
            "the UPDATE_SCRIPT_KEY_* slots must be 1..=UPDATE_SCRIPT_SHARED_KEYS, each once"
        );
        seen_keys[slot - 1] = true;
        i += 1;
    }
    let mut seen_args = [false; UPDATE_SCRIPT_SHARED_ARGV];
    let mut i = 0;
    while i < UPDATE_SCRIPT_ARG_SLOTS.len() {
        let (name, slot) = UPDATE_SCRIPT_ARG_SLOTS[i];
        assert_lua_local(name, slot);
        assert!(
            slot >= 1 && slot <= UPDATE_SCRIPT_SHARED_ARGV && !seen_args[slot - 1],
            "the UPDATE_SCRIPT_ARG_* slots must be 1..=UPDATE_SCRIPT_SHARED_ARGV, each once"
        );
        seen_args[slot - 1] = true;
        i += 1;
    }
    assert!(
        UPDATE_SCRIPT_KEY_SLOTS.len() == UPDATE_SCRIPT_SHARED_KEYS
            && lua_locals_named("KEY_") == UPDATE_SCRIPT_KEY_SLOTS.len(),
        "scripts/update_layer_fast.lua declares a `local KEY_*` slot storage::lua does not name \
         (or the other way round): add its UPDATE_SCRIPT_KEY_* twin and place it in \
         UpdateScriptPrefix::build"
    );
    assert!(
        UPDATE_SCRIPT_ARG_SLOTS.len() == UPDATE_SCRIPT_SHARED_ARGV
            && lua_locals_named("ARG_") == UPDATE_SCRIPT_ARG_SLOTS.len(),
        "scripts/update_layer_fast.lua declares a `local ARG_*` slot storage::lua does not name \
         (or the other way round): add its UPDATE_SCRIPT_ARG_* twin and place it in \
         UpdateScriptPrefix::build"
    );
    assert_lua_local("SHARED_KEYS", UPDATE_SCRIPT_SHARED_KEYS);
    assert_lua_local("SHARED_ARGV", UPDATE_SCRIPT_SHARED_ARGV);
    // The commit release ring's shape (`storage::layer`, "The commit release
    // ring"): step 7a and the WATCH path's `MULTI` run one rule, so both must
    // size it alike.
    assert_lua_local(
        "COMMIT_RELEASE_RING_SLOTS",
        layer::COMMIT_RELEASE_RING_SLOTS,
    );
    assert_lua_local(
        "COMMIT_RELEASE_RING_SLOT_RELEASES",
        layer::COMMIT_RELEASE_RING_SLOT_RELEASES,
    );
};

/// The attempt budget of the `update_layer` commit `EVALSHA`, in both script
/// modes — the fast path's and `structural::commit_structural`'s: fred writes
/// it to the wire at most once.
///
/// The data pool otherwise keeps fred's default of three attempts, and fred
/// re-writes an in-flight command after a reconnect, filtered by the command's
/// own give-up flag: a command whose caller gave up — its timeout fired or its
/// future was dropped — is not replayed (the vendored fred's change 9), but one
/// whose caller still waits is. For the commit that replay is a second
/// execution of an attempt that may already have landed with its reply lost,
/// and it can land after that `client_id`'s next commit. With one attempt, a
/// dropped connection fails the
/// attempt transiently instead, and so does a cluster redirect, which the one
/// write leaves no attempt to follow: a `MOVED` fails it as `Unknown` "Too many
/// failed write attempts." and an `-ASK` as `Routing` "Max attempts reached."
/// (both transient, `storage::client::is_router_retry_budget_exhausted_error`).
/// `update_layer`'s own retry loop decides what happens next — its next attempt
/// starts with the commit-token check, which reports a commit that did land.
/// The `-ASK` leaves the client's router running only because of the vendored
/// fred patch (`vendor/fred-10.1.0.SOURCE.md`); stock fred 10.1.0 ended it.
///
/// "At most once" is about the number of writes; the call's own timeout bounds
/// their timing. A commit still queued in fred's channel while the client's
/// router reconnects — or while `storage::router_supervisor` restarts a router
/// that ended with an error — is written when the router next reads the
/// channel only if its caller is still waiting: once the caller gives up, its
/// timeout fired or its future dropped, fred never writes it (the vendored
/// fred's change 9). (A router that panicked or was aborted takes its channel
/// with it: the restart builds a new one, and the queued commit fails instead
/// of being written.) A commit fred wrote before its caller gave up is not
/// recalled and can still land after the caller was answered. For an
/// unchained call with an `operation_id` the start-record check refuses it as
/// overtaken if the client has committed since; for a chained call the
/// predecessor check refuses it (`PREDECESSOR`) unless the record still names
/// its predecessor, so it lands only directly after it; without an
/// `operation_id`, it is one late execution of the attempt.
pub const UPDATE_LAYER_COMMIT_MAX_ATTEMPTS: u32 = 1;

/// Send `script` as `EVALSHA` over `client` with an attempt budget of
/// [`UPDATE_LAYER_COMMIT_MAX_ATTEMPTS`], loading it and trying once more on
/// `NOSCRIPT` — the same reload `Script::evalsha_with_reload` performs, which
/// cannot carry options. Used for the commit. The GC decrement does not use it:
/// its script answers a re-send with `replayed`, so it keeps fred's retries
/// (`external_values`' `gc_member`, b.yuo H8).
///
/// Dispatched through `WithOptions` + `ClientLike::custom`, never
/// `with_options(..).evalsha(..)`: fred 10.1.0's `WithOptions` implements no
/// `LuaInterface`, so that call would resolve through `Deref` to the plain
/// client and silently drop the options. `custom` goes through `WithOptions`'
/// own `ClientLike::change_command`, which applies them. The command is hashed
/// to `keys[0]`'s Cluster slot, as fred's `evalsha` hashes a single-slot key set.
///
/// The `SCRIPT LOAD` itself keeps fred's default retry: it is idempotent.
pub(crate) async fn evalsha_once_with_reload<R: fred::types::FromValue>(
    client: &fred::clients::Client,
    script: &Script,
    keys: Vec<String>,
    args: Vec<fred::types::Value>,
) -> Result<R, fred::error::Error> {
    use fred::interfaces::ClientLike;

    let options = fred::types::config::Options {
        max_attempts: Some(UPDATE_LAYER_COMMIT_MAX_ATTEMPTS),
        ..Default::default()
    };
    let once = client.with_options(&options);
    let cluster_hash = keys
        .first()
        .map(|key| fred::types::ClusterHash::from(key.as_str()))
        .unwrap_or_default();
    let mut command_args: Vec<fred::types::Value> = Vec::with_capacity(2 + keys.len() + args.len());
    command_args.push(script.sha1().clone().into());
    command_args.push((keys.len() as i64).into());
    command_args.extend(keys.into_iter().map(fred::types::Value::from));
    command_args.extend(args);
    let command = fred::types::CustomCommand::new_static("EVALSHA", cluster_hash, false);

    match once
        .custom::<R, _>(command.clone(), command_args.clone())
        .await
    {
        Err(error) if error.details().starts_with("NOSCRIPT") => {
            script.load(client).await?;
            once.custom::<R, _>(command, command_args).await
        }
        result => result,
    }
}

/// The compiled stream-ID collision probe used by the `update_layer`
/// WATCH/MULTI/EXEC slow path and by nothing else (b.ph9).
///
/// Answers "does the per-object stream already hold an entry at or above
/// the ID this commit intends to write?" with a server-side projection to
/// the colliding entry's ID plus a one-word classification (`deleted` when
/// that entry is a layer-retirement tombstone), so a multi-megabyte OMPF
/// payload never crosses the wire. See `scripts/stream_collision_probe.lua`
/// for the classification's purpose, and for why the slow path
/// needs a pre-`MULTI` probe rather than the fast path's reorder-the-XADD
/// fix, and why `XRANGE` beats `XINFO STREAM` / `XREVRANGE` here.
///
/// Same lazy-`SHA-1` / `NOSCRIPT`-reload story as
/// [`UPDATE_LAYER_FAST_SCRIPT`].
pub static STREAM_COLLISION_PROBE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/stream_collision_probe.lua")));

/// The compiled keyspace type probe used by the `update_layer`
/// WATCH/MULTI/EXEC slow path and by nothing else (b.459).
///
/// Answers "does any key this commit is about to mutate already hold the
/// wrong Valkey type?" in one round trip, so the slow path refuses before
/// `MULTI` rather than half-committing on a WRONGTYPE frame inside `EXEC`.
/// See `scripts/keyspace_type_probe.lua` for why `TYPE` needs no `pcall`
/// and why the fast path's equivalent guard lives inside its own script
/// instead.
///
/// Same lazy-`SHA-1` / `NOSCRIPT`-reload story as
/// [`UPDATE_LAYER_FAST_SCRIPT`].
pub static KEYSPACE_TYPE_PROBE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/keyspace_type_probe.lua")));

/// The compiled conditional cursor-stamp script (b.vn6) — the ONE primitive
/// every checkpoint-cursor stamp site goes through.
///
/// See `scripts/stamp_cursor_if_live.lua` for the race it closes, why the
/// predicate is `HEXISTS seq` rather than `EXISTS`, and why it is Lua rather
/// than WATCH/MULTI/EXEC. Callers go through [`stamp_obj_header_cursor`]
/// (the checkpoint loop) or [`stamp_obj_header_cursor_below`] (the base
/// bootstrap, with its cursor-regression floor) rather than driving this
/// directly.
///
/// Same lazy-`SHA-1` / `NOSCRIPT`-reload story as [`UPDATE_LAYER_FAST_SCRIPT`].
pub static STAMP_CURSOR_IF_LIVE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/stamp_cursor_if_live.lua")));

/// The compiled atomic object-snapshot read (Spec SDD SR-1), driven only by
/// [`crate::storage::diff_read`]: one `EVALSHA` per atomic read, and once in
/// self-test mode before readiness.
///
/// Read-only and stateless, so there is nothing to tear down. See
/// `scripts/object_snapshot_read.lua` for its `KEYS` / `ARGV` contract and
/// reply shapes.
///
/// Same lazy-`SHA-1` / `NOSCRIPT`-reload story as [`UPDATE_LAYER_FAST_SCRIPT`].
pub static OBJECT_SNAPSHOT_READ_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/object_snapshot_read.lua")));

/// The compiled per-window read of the structural WATCH read (SDD §10.5),
/// driven only by [`crate::storage::diff_read`]: one `EVALSHA` per window of
/// the bounded-reply fan-out's rounds 2 and 3 (`FanoutReplies::Bounded`),
/// reading that window's value rows, section and tombstone hashes and
/// children lists in one call. Two kinds of round-2/3 window stay plain and
/// do not come here: a lone children list over τ, and every window after a
/// script fallback (`storage::diff_read`'s bounded fan-out names them).
///
/// Not [`OBJECT_SNAPSHOT_READ_SCRIPT`]: that script reads a whole object and
/// blocks its shard for the whole read; this one reads one window, sized
/// before it is sent so that its modelled shard time stays within
/// `config::FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS`. Read-only and stateless,
/// so there is nothing to tear down. See `scripts/fanout_window_read.lua` for
/// its `KEYS` / `ARGV` contract and reply shapes.
///
/// Sent through [`evalsha_reload_rebuilding`], which loads it on every
/// primary on `NOSCRIPT` (a `SCRIPT FLUSH`, a store restart, a failover) and
/// sends the window once more.
pub static FANOUT_WINDOW_READ_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/fanout_window_read.lua")));

/// Send `script` as `EVALSHA` over `client`, the arguments built by
/// `build_args`, and on `NOSCRIPT` load the script (`Script::load`, fred's
/// `SCRIPT LOAD` on every primary) and send it once more with the arguments
/// built afresh. Used by the structural WATCH read's script windows
/// ([`FANOUT_WINDOW_READ_SCRIPT`]).
///
/// The same reload `Script::evalsha_with_reload` performs, except that the
/// arguments are rebuilt for the second send instead of cloned: that method
/// clones its keys and arguments and holds the clone across the first send,
/// which would double a window's request memory for the whole call.
///
/// The command declares exactly one key, `key`, sent as `KEYS[1]` with
/// `numkeys` 1: the signature admits no other key count, so the command
/// always has a key to hash. It is hashed to `key`'s Cluster slot, as fred's
/// `evalsha` hashes a single-slot key set. fred's default attempt budget is
/// kept (the call is read-only and idempotent, so a re-send after a
/// reconnect is harmless). Dispatched through `ClientLike::custom`, as
/// [`evalsha_once_with_reload`] is.
pub(crate) async fn evalsha_reload_rebuilding<R: fred::types::FromValue>(
    client: &fred::clients::Client,
    script: &Script,
    key: &str,
    build_args: impl Fn() -> Vec<fred::types::Value>,
) -> Result<R, fred::error::Error> {
    use fred::interfaces::ClientLike;

    let cluster_hash = fred::types::ClusterHash::from(key);
    let command_args = || {
        let args = build_args();
        let mut command_args: Vec<fred::types::Value> = Vec::with_capacity(3 + args.len());
        command_args.push(script.sha1().clone().into());
        command_args.push(1_i64.into());
        command_args.push(key.into());
        command_args.extend(args);
        command_args
    };
    let command = fred::types::CustomCommand::new_static("EVALSHA", cluster_hash, false);

    match client.custom::<R, _>(command.clone(), command_args()).await {
        Err(error) if error.details().starts_with("NOSCRIPT") => {
            script.load(client).await?;
            client.custom::<R, _>(command, command_args()).await
        }
        result => result,
    }
}

// ── The shared KEYS / ARGV prefix of `update_layer_fast.lua` ─────────────
//
// One-based slots of the prefix both script modes share, the Rust twins of
// the script's step-0 `local KEY_<X>` / `local ARG_<X>` and `SHARED_KEYS` /
// `SHARED_ARGV`. [`UpdateScriptPrefix::build`] places every slot through
// these, and the compile-time check above reads each name's value out of the
// script and refuses to build on any disagreement, or when either side names
// a slot the other does not.

/// `KEYS` slot of the object header, `le:{oid}:obj:hdr`.
pub const UPDATE_SCRIPT_KEY_HEADER: usize = 1;
/// `KEYS` slot of the fields index, `le:{oid}:obj:flds`.
pub const UPDATE_SCRIPT_KEY_FIELDS_INDEX: usize = 2;
/// `KEYS` slot of the time-samples index, `le:{oid}:obj:tss`.
pub const UPDATE_SCRIPT_KEY_TS_INDEX: usize = 3;
/// `KEYS` slot of the per-object stream.
pub const UPDATE_SCRIPT_KEY_STREAM: usize = 4;
/// `KEYS` slot of the external-value reference index, `le:{oid}:obj:ext`.
pub const UPDATE_SCRIPT_KEY_EXT_INDEX: usize = 5;
/// `KEYS` slot of the commit-token record, `le:{oid}:obj:wtok`.
pub const UPDATE_SCRIPT_KEY_COMMIT_TOKEN: usize = 6;
/// `KEYS` slot of the snapshot fence, `le:{oid}:obj:snapfence` (Spec SDD
/// SR-5.4): any live reader lease refuses the commit `FENCED`, in both modes
/// (the script's step 1d).
pub const UPDATE_SCRIPT_KEY_FENCE: usize = 7;
/// `KEYS` slot of the writer-intent key, `le:{oid}:obj:snapintent`. Neither
/// script mode registers an intent (only the WATCH path does); the script only
/// `TYPE`-probes it, so a wrong-typed intent key is refused the same way in
/// both modes and on the WATCH path (SDD §7.5).
pub const UPDATE_SCRIPT_KEY_INTENT: usize = 8;
/// `KEYS` slot of the commit release ring, `le:{oid}:obj:wrel`: a commit that
/// owes external-value releases records them there, under the call token
/// (`ARGV[15]`), in its own atomic unit (the script's step 7a), so an
/// own-call `ALREADY_LANDED` verdict, or a call that ends without proof of
/// its outcome, can enqueue the releases a lost reply carried, and sets its
/// TTL (`ARGV[16]`). `TYPE`-probed at step 1e, right after the fence check.
pub const UPDATE_SCRIPT_KEY_COMMIT_RELEASE: usize = 9;
/// `KEYS` slot of the object's migration sentinel
/// ([`KeySchema::migration_sentinel`] of the object id, b.qtk / b.cyu): a key
/// that is never written, so that during a legacy cluster slot
/// migration the store refuses the commit before it starts rather than
/// letting an in-script refusal tear it after its first write. The script
/// checks the slot's name before anything else (refusing `ENCODING`) and
/// sends no command on it.
pub const UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL: usize = 10;

/// The number of leading `KEYS` both modes of [`UPDATE_LAYER_FAST_SCRIPT`]
/// share: the `UPDATE_SCRIPT_KEY_*` slots above. The Rust twin of the
/// script's `SHARED_KEYS`; the two change together, checked at compile time.
/// Every mode-specific key follows them and is located from this length: fast
/// mode's record `i` (one-based) is `KEYS[UPDATE_SCRIPT_SHARED_KEYS + i]`;
/// structural mode's sections index is `KEYS[UPDATE_SCRIPT_SHARED_KEYS + 1]`,
/// then the deleted-sections and children indexes, then its records.
pub const UPDATE_SCRIPT_SHARED_KEYS: usize = UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL;

/// `ARGV` slot of the server timestamp (µs).
pub const UPDATE_SCRIPT_ARG_SERVER_TS: usize = 1;
/// `ARGV` slot of the writer's `client_id`.
pub const UPDATE_SCRIPT_ARG_CLIENT_ID: usize = 2;
/// `ARGV` slot of the writer's client timestamp.
pub const UPDATE_SCRIPT_ARG_CLIENT_TS: usize = 3;
/// `ARGV` slot of the stream `MAXLEN` (`0` = untrimmed).
pub const UPDATE_SCRIPT_ARG_STREAM_MAXLEN: usize = 4;
/// `ARGV` slot of the object id.
pub const UPDATE_SCRIPT_ARG_OBJECT_ID: usize = 5;
/// `ARGV` slot of the mutation-record count.
pub const UPDATE_SCRIPT_ARG_NUM_MUTATIONS: usize = 6;
/// `ARGV` slot of the writer's raw OMPF for verbatim relay (empty = none).
pub const UPDATE_SCRIPT_ARG_RELAY_OMPF: usize = 7;
/// `ARGV` slot of the commit match key (never empty).
pub const UPDATE_SCRIPT_ARG_COMMIT_KEY: usize = 8;
/// `ARGV` slot of the commit record suffix (`""` without an operation id).
pub const UPDATE_SCRIPT_ARG_RECORD_SUFFIX: usize = 9;
/// `ARGV` slot of the writer guard (`""` without an operation id; `"=…"`, an
/// unchained call's start record; `"^"` or `"^op.<previous>"`, a chained
/// call's predecessor — `CommitIdentity::guard_arg`).
pub const UPDATE_SCRIPT_ARG_WRITER_GUARD: usize = 10;
/// `ARGV` slot of the relay byte budget B ([`RelayRetention`]; `0` = off:
/// the script neither folds `relay_gens` nor trims by bytes).
pub const UPDATE_SCRIPT_ARG_RELAY_BUDGET_BYTES: usize = 11;
/// `ARGV` slot of the relay generation size T.
pub const UPDATE_SCRIPT_ARG_RELAY_GENERATION_BYTES: usize = 12;
/// `ARGV` slot of the relay generation count K.
pub const UPDATE_SCRIPT_ARG_RELAY_GENERATIONS: usize = 13;
/// `ARGV` slot of the fewest entries a byte trim leaves, MIN.
pub const UPDATE_SCRIPT_ARG_RELAY_MIN_ENTRIES: usize = 14;
/// `ARGV` slot of the calling call's commit token in its recorded form
/// (`CommitIdentity::token`; 32 lowercase hex digits in production), with or
/// without an operation id: the key of the call's entry in the commit release
/// ring, which the script's step 7a writes when the commit owes releases. The
/// script refuses an empty one `ENCODING` and checks nothing else about it.
pub const UPDATE_SCRIPT_ARG_CALL_TOKEN: usize = 15;
/// `ARGV` slot of the commit release ring's TTL in ms
/// (`storage::keys::commit_release_ring_ttl`, the store's
/// `LayerStore::with_commit_release_ring_ttl`): the script's step 7a sets it
/// on `le:{oid}:obj:wrel` with `PEXPIRE` after every ring write. Canonical
/// decimal, 1 to 12 digits with no leading zero; anything else is refused
/// `ENCODING` on every run that reaches step 1b, whether or not the commit
/// owes releases, before the commit-token record is read.
pub const UPDATE_SCRIPT_ARG_RING_TTL_MS: usize = 16;

/// The number of leading `ARGV` slots both modes of
/// [`UPDATE_LAYER_FAST_SCRIPT`] share: the `UPDATE_SCRIPT_ARG_*` slots above.
/// The Rust twin of the script's `SHARED_ARGV`, checked at compile time. The
/// mode slot is `ARGV[UPDATE_SCRIPT_SHARED_ARGV + 1]` (`ARGV[17]`), and the
/// mode's own slots follow it.
pub const UPDATE_SCRIPT_SHARED_ARGV: usize = UPDATE_SCRIPT_ARG_RING_TTL_MS;

/// The mode slot of [`UPDATE_LAYER_FAST_SCRIPT`] (Spec SDD SR-6.1): which
/// record vocabulary and guards follow the shared prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateScriptMode {
    /// The field / time-sample fast path (`encode_delta`'s `SF` / `ST`
    /// records), run by `LayerStore::update_layer`'s Lua backend.
    Fast,
    /// The fenced structural commit (`encode_mutations`' records), run by
    /// `structural::commit_structural`.
    Structural,
}

impl UpdateScriptMode {
    /// The mode slot's value, as the script spells it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Structural => "structural",
        }
    }

    /// The `mode` label value this mode's refusals carry on the
    /// `{backend="lua", site="update"}` cells of
    /// `live_edit_keyspace_type_error_total` and
    /// `live_edit_stream_id_collision_total` — a metric vocabulary owned by
    /// `observability::metrics` ([`metrics::UPDATE_SCRIPT_MODES`]), kept apart
    /// from [`Self::as_str`], the script's own wire spelling, even where the
    /// two strings agree.
    pub const fn metric_label(self) -> &'static str {
        match self {
            Self::Fast => metrics::UPDATE_SCRIPT_MODE_FAST,
            Self::Structural => metrics::UPDATE_SCRIPT_MODE_STRUCTURAL,
        }
    }
}

/// The shared `KEYS` prefix of an [`UPDATE_LAYER_FAST_SCRIPT`] call on
/// `object_id`, in either mode: every `UPDATE_SCRIPT_KEY_*` slot, in slot
/// order, in a vector with room for `extra_keys` more (the mode's own keys,
/// which the caller appends). The one place the prefix's keys are built:
/// [`UpdateScriptPrefix::build`] uses it, and so does any test that hand-builds
/// a script call.
///
/// Every key carries the `{object_id}` hash tag, so the script runs on one
/// slot.
pub fn update_script_shared_keys(
    keys: &KeySchema,
    object_id: u64,
    extra_keys: usize,
) -> Vec<String> {
    let oid_str = object_id.to_string();
    // Every slot is placed through its constant, so the order below is not
    // what locates it; a slot left unfilled is a builder bug, caught here.
    let mut fixed: [String; UPDATE_SCRIPT_SHARED_KEYS] = Default::default();
    fixed[UPDATE_SCRIPT_KEY_HEADER - 1] = keys.obj_header(&oid_str);
    fixed[UPDATE_SCRIPT_KEY_FIELDS_INDEX - 1] = keys.obj_fields_index(&oid_str);
    fixed[UPDATE_SCRIPT_KEY_TS_INDEX - 1] = keys.obj_time_samples_index(&oid_str);
    fixed[UPDATE_SCRIPT_KEY_STREAM - 1] = keys.layer_stream(object_id);
    fixed[UPDATE_SCRIPT_KEY_EXT_INDEX - 1] = keys.obj_ext_index(&oid_str);
    fixed[UPDATE_SCRIPT_KEY_COMMIT_TOKEN - 1] = keys.obj_write_token(&oid_str);
    fixed[UPDATE_SCRIPT_KEY_FENCE - 1] = keys.obj_snapshot_fence(&oid_str);
    fixed[UPDATE_SCRIPT_KEY_INTENT - 1] = keys.obj_snapshot_intent(&oid_str);
    fixed[UPDATE_SCRIPT_KEY_COMMIT_RELEASE - 1] = keys.obj_write_releases(&oid_str);
    fixed[UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL - 1] = keys.migration_sentinel(&oid_str);
    debug_assert!(
        fixed.iter().all(|k| !k.is_empty()),
        "every UPDATE_SCRIPT_KEY_* slot must be filled"
    );
    let mut out = Vec::with_capacity(UPDATE_SCRIPT_SHARED_KEYS + extra_keys);
    out.extend(fixed);
    out
}

/// The values of the shared `KEYS` / `ARGV` prefix of one
/// [`UPDATE_LAYER_FAST_SCRIPT`] call, in either mode.
pub(crate) struct UpdateScriptPrefix<'a> {
    /// The object the commit lands on.
    pub(crate) object_id: u64,
    /// The commit's server timestamp, microseconds.
    pub(crate) server_ts: u64,
    /// The writing client, recorded in the stream payload, the header and
    /// the field of the commit-token record.
    pub(crate) client_id: &'a str,
    /// The client's timestamp.
    pub(crate) client_ts: u64,
    /// The stream's approximate `MAXLEN`; `0` means no count trim.
    pub(crate) stream_maxlen: u64,
    /// The relay byte budget's parameters (`ARGV[11..14]`): the script folds
    /// the header's `relay_gens` with them and trims the stream head when a
    /// generation is dropped. A budget of `0` writes neither.
    pub(crate) relay_retention: RelayRetention,
    /// The commit release ring's TTL in ms (`ARGV[16]`), from the store
    /// (`LayerStore::with_commit_release_ring_ttl`, already clamped to
    /// `[1, 999_999_999_999]`): what step 7a's `PEXPIRE` sets on the ring
    /// after writing it.
    pub(crate) ring_ttl_ms: u64,
    /// The number of mutation records after the mode's own slots.
    pub(crate) num_mutations: u64,
    /// The writer's raw inbound OMPF, `XADD`ed as the entry's `ompf` field;
    /// `None` sends the empty string, which the script reads as "no relay
    /// bytes".
    pub(crate) relay_ompf: Option<Bytes>,
    /// The call's identity on the commit-token record: its per-call token
    /// and, when the client sent one, its operation. Supplies the commit match
    /// key (`ARGV[8]`, never empty — the script refuses an empty one), the
    /// commit record suffix (`ARGV[9]`), the writer guard (`ARGV[10]`) and the
    /// call token (`ARGV[15]`, the key of the call's commit release ring
    /// entry; never empty either).
    pub(crate) identity: &'a CommitIdentity<'a>,
}

impl UpdateScriptPrefix<'_> {
    /// Build the shared `KEYS` and `ARGV` prefix, then push the mode slot.
    /// The caller appends the mode's own keys and slots; `extra_keys` /
    /// `extra_argv` only size the vectors for them.
    ///
    /// `ARGV` is `Vec<fred::types::Value>` because the OMPF is the one binary
    /// element (`Value::Bytes`); every other element is a UTF-8 string.
    pub(crate) fn build(
        self,
        keys: &KeySchema,
        mode: UpdateScriptMode,
        extra_keys: usize,
        extra_argv: usize,
    ) -> (Vec<String>, Vec<fred::types::Value>) {
        let oid_str = self.object_id.to_string();
        let script_keys = update_script_shared_keys(keys, self.object_id, extra_keys);

        // Every slot is placed through its `UPDATE_SCRIPT_ARG_*` constant, so
        // the order below is not what locates it; a slot left unfilled is a
        // builder bug, caught here.
        let mut argv: Vec<fred::types::Value> =
            vec![fred::types::Value::Null; UPDATE_SCRIPT_SHARED_ARGV];
        argv.reserve(1 + extra_argv);
        argv[UPDATE_SCRIPT_ARG_SERVER_TS - 1] = self.server_ts.to_string().into();
        argv[UPDATE_SCRIPT_ARG_CLIENT_ID - 1] = self.client_id.to_owned().into();
        argv[UPDATE_SCRIPT_ARG_CLIENT_TS - 1] = self.client_ts.to_string().into();
        argv[UPDATE_SCRIPT_ARG_STREAM_MAXLEN - 1] = self.stream_maxlen.to_string().into();
        argv[UPDATE_SCRIPT_ARG_OBJECT_ID - 1] = oid_str.into();
        argv[UPDATE_SCRIPT_ARG_NUM_MUTATIONS - 1] = self.num_mutations.to_string().into();
        argv[UPDATE_SCRIPT_ARG_RELAY_OMPF - 1] = match self.relay_ompf {
            Some(bytes) => fred::types::Value::Bytes(bytes),
            None => fred::types::Value::from(""),
        };
        argv[UPDATE_SCRIPT_ARG_COMMIT_KEY - 1] = self.identity.match_key().to_owned().into();
        argv[UPDATE_SCRIPT_ARG_RECORD_SUFFIX - 1] = self.identity.record_suffix().into();
        argv[UPDATE_SCRIPT_ARG_WRITER_GUARD - 1] = self.identity.guard_arg().into();
        argv[UPDATE_SCRIPT_ARG_RELAY_BUDGET_BYTES - 1] =
            self.relay_retention.budget_bytes.to_string().into();
        argv[UPDATE_SCRIPT_ARG_RELAY_GENERATION_BYTES - 1] =
            self.relay_retention.generation_bytes.to_string().into();
        argv[UPDATE_SCRIPT_ARG_RELAY_GENERATIONS - 1] =
            self.relay_retention.generations.to_string().into();
        argv[UPDATE_SCRIPT_ARG_RELAY_MIN_ENTRIES - 1] =
            self.relay_retention.min_entries.to_string().into();
        argv[UPDATE_SCRIPT_ARG_CALL_TOKEN - 1] = self.identity.token().to_owned().into();
        argv[UPDATE_SCRIPT_ARG_RING_TTL_MS - 1] = self.ring_ttl_ms.to_string().into();
        debug_assert!(
            argv.iter().all(|slot| !slot.is_null()),
            "every UPDATE_SCRIPT_ARG_* slot must be filled"
        );
        argv.push(mode.as_str().into());

        (script_keys, argv)
    }
}

/// The size of one script call in bytes: every `KEYS` string plus every
/// `ARGV` element (the OMPF's raw bytes included). The one definition of the
/// commit size reported for an [`UPDATE_LAYER_FAST_SCRIPT`] call, in either
/// mode: `LayerStore::update_layer_once_lua` and
/// `structural::commit_structural` both size the exact `KEYS` and `ARGV` they
/// send with it, so their recorded commit sizes compare like for like.
pub(crate) fn script_call_bytes(keys: &[String], argv: &[fred::types::Value]) -> u64 {
    let key_bytes: usize = keys.iter().map(String::len).sum();
    let argv_bytes: usize = argv
        .iter()
        .map(|value| value.as_bytes().map_or(0, <[u8]>::len))
        .sum();
    (key_bytes + argv_bytes) as u64
}

/// The compiled structural-cache repair read (Spec SDD SR-4.1), driven only by
/// [`crate::storage::structural_cache_sync`]: the header facts, the call's
/// commit-token record and a bounded chunk of the layer stream, in one
/// `EVALSHA` on the data pool.
///
/// Read-only. See `scripts/structural_cache_sync.lua` for its `KEYS` / `ARGV`
/// contract and reply shapes.
///
/// Same lazy-`SHA-1` / `NOSCRIPT`-reload story as [`UPDATE_LAYER_FAST_SCRIPT`].
pub static STRUCTURAL_CACHE_SYNC_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/structural_cache_sync.lua")));

/// The compiled `CreateLayer` commit point: refuses a session whose deletion
/// marker exists, then — only when the new object has no commit claim
/// (`le:{sid}:createclaim:{oid}`: never mapped before, never vetoed by the
/// orphan sweep) and the layer mapping still holds the value the caller
/// planned against — flips the layer mapping, lists the layer in its
/// session's layers index, claims the object `committed` and, for a call
/// carrying a client `operation_id`, checks and writes the operation record,
/// all in one atomic step on the `{session_id}` slot.
/// Driven only through `LayerStore::dispatch_create_layer_commit`, on a
/// `txn_pool` connection: by `LayerStore::create_layer_with_operation`, its
/// in-process finisher, and the orphan sweep's completion of a stranded
/// overwrite.
///
/// See `scripts/create_layer_commit.lua` for its `KEYS` / `ARGV` contract and
/// reply shapes.
///
/// Same lazy-`SHA-1` / `NOSCRIPT`-reload story as [`UPDATE_LAYER_FAST_SCRIPT`].
pub static CREATE_LAYER_COMMIT_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/create_layer_commit.lua")));

/// The compiled orphan-sweep veto on an unmapped object: unless the layer
/// mapping already names the object, the session's deletion marker exists, or
/// the object already has a commit claim (kept as it is), claims the object
/// `vetoed` (`le:{sid}:createclaim:{object_id}`) with its TTL, so
/// [`CREATE_LAYER_COMMIT_SCRIPT`] refuses any later commit of that object
/// (`VETOED`). Driven only by `storage::orphan_sweep`, before it retires an
/// unmapped, unfenced object. Its `SET` follows a no-op `PEXPIRE` opener on
/// the claim key, so it runs on a full store (the opener rule, SDD §7.5.1).
///
/// See `scripts/create_layer_veto.lua` for its `KEYS` / `ARGV` contract and
/// reply shapes.
///
/// Same lazy-`SHA-1` / `NOSCRIPT`-reload story as [`UPDATE_LAYER_FAST_SCRIPT`].
pub static CREATE_LAYER_VETO_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/create_layer_veto.lua")));

/// The compiled wake-up of a retired object's stream: `PEXPIRE`, then
/// `XADD * payload deleted` (uncapped; it re-creates a stream that is gone),
/// then `PEXPIRE` again, in one atomic step on the `{object_id}` slot. The
/// opening `PEXPIRE` is its opener, so it runs on a full store (the opener
/// rule, SDD §7.5.1). Driven only through `layer::wake_retired_object_stream`,
/// the one wake-up of every retirement route (the overwrite tail, the orphan
/// sweep, `DeleteLayer`).
///
/// See `scripts/retired_object_wake.lua` for its `KEYS` / `ARGV` contract and
/// why it runs on a full store.
///
/// Same lazy-`SHA-1` / `NOSCRIPT`-reload story as [`UPDATE_LAYER_FAST_SCRIPT`].
pub static RETIRED_OBJECT_WAKE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/retired_object_wake.lua")));

/// The compiled compare-and-delete of one writer's field in an object's
/// `UpdateLayer` commit-token record (`le:{oid}:obj:wtok`). Driven only by the
/// `update_layer` slow path, to retract the record a torn commit carrying a
/// client `operation_id` may have left behind.
///
/// See `scripts/commit_record_retract.lua` for its contract.
///
/// Same lazy-`SHA-1` / `NOSCRIPT`-reload story as [`UPDATE_LAYER_FAST_SCRIPT`].
pub static COMMIT_RECORD_RETRACT_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/commit_record_retract.lua")));

/// Stamp `pairs` onto a layer's `obj:hdr` **only if that header is still
/// live**, atomically (b.vn6).
///
/// Returns `Ok(true)` when the header carried a `seq` field and every pair was
/// written, `Ok(false)` when it did not and nothing was written — the layer
/// was deleted or overwritten under the caller.
///
/// # Every cursor stamp site in the service goes through this script
///
/// There are four, and they were all bare `HSET`s before this existed:
/// `checkpoint::base_bootstrap`'s post-PUT stamp (through
/// [`stamp_obj_header_cursor_below`], which adds a cursor-regression floor),
/// and the checkpoint loop's periodic-snapshot, tail-capture and chain-collapse
/// re-base arms (through this function). A bare `HSET` on a deleted header
/// CREATES a cursor-only phantom (see the script's header for the full
/// failure), so guarding one site and not the others would only move the bug.
/// One script is what makes "all four" checkable by grepping for the bare
/// `hset` on a header key rather than by reading four call sites.
///
/// # `false` is not an error, and callers must not treat it as one
///
/// The layer is gone. Nothing is wrong with Valkey, nothing is wrong with the
/// adapter, and no retry can change the answer. What a caller owes on `false`
/// is argued at its own site: the loop's three arms simply decline to record a
/// cursor for a layer that no longer exists and skip the retention prune.
///
/// `client` is a single fred connection — `maint_pool.next()` at the loop's
/// three arms, matching what each site's `HSET` used before.
///
/// Sends the plain, even-length `ARGV` ([`stamp_cursor_argv`] with no floor),
/// byte-identical to what this function sent before the floor existed.
pub async fn stamp_obj_header_cursor(
    client: &fred::clients::Client,
    header_key: &str,
    pairs: &[(&str, String)],
) -> Result<bool, fred::error::Error> {
    let argv = stamp_cursor_argv(pairs, None);
    let stamped: i64 = STAMP_CURSOR_IF_LIVE_SCRIPT
        .evalsha_with_reload::<i64, _, _>(client, vec![header_key.to_owned()], argv)
        .await?;
    Ok(stamped == 1)
}

/// What [`stamp_obj_header_cursor_below`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorStamp {
    /// The header was live, no generation at or above the floor was stamped,
    /// and every pair was written.
    Stamped,
    /// The header carries no `seq` — the layer is gone — and nothing was
    /// written. The same answer as [`stamp_obj_header_cursor`]'s `false`.
    HeaderAbsent,
    /// The header's `last_checkpoint_seq` is already at or above the floor
    /// (or present but not a plain decimal), and nothing was written: a
    /// generation that covers this one was already stamped.
    CursorAtOrAbove,
}

/// [`stamp_obj_header_cursor`] with a cursor-regression floor: additionally
/// refuse — atomically, in the same script step as the write — when
/// `obj:hdr last_checkpoint_seq` is already at or above `refuse_at_or_above`.
///
/// The base bootstrap's caller only (`checkpoint::base_bootstrap`), with its
/// target `snap_seq`. That turns the seq half of its guard (b) — otherwise a
/// read, then this write — into one atomic check, which is what keeps two
/// overlapping bootstrap runs for one object (the creating call's, and a client
/// re-send's re-spawn) from both stamping and both counting a persisted
/// generation. The checkpoint loop's three stamp sites keep calling
/// [`stamp_obj_header_cursor`], whose `ARGV` carries no floor: their equality
/// semantics differ (a chain-collapse re-base re-stamps the `to_seq` its own
/// capture already stamped) and this floor would refuse them. See
/// `scripts/stamp_cursor_if_live.lua`'s "optional floor" section.
///
/// [`CursorStamp::HeaderAbsent`] is this function's "the layer is gone"
/// answer — the same verdict as [`stamp_obj_header_cursor`]'s `false`, and
/// likewise not an error. The bootstrap owes the blob it just wrote a reap on
/// it, argued at its site. [`CursorStamp::CursorAtOrAbove`] is not an error
/// either: a generation covering this one was already stamped.
///
/// `client` is a single fred connection — `pool.next()` on the data pool at
/// the bootstrap, matching what its `HSET` used before.
pub async fn stamp_obj_header_cursor_below(
    client: &fred::clients::Client,
    header_key: &str,
    pairs: &[(&str, String)],
    refuse_at_or_above: u64,
) -> Result<CursorStamp, fred::error::Error> {
    let argv = stamp_cursor_argv(pairs, Some(refuse_at_or_above));
    let stamped: i64 = STAMP_CURSOR_IF_LIVE_SCRIPT
        .evalsha_with_reload::<i64, _, _>(client, vec![header_key.to_owned()], argv)
        .await?;
    Ok(match stamped {
        1 => CursorStamp::Stamped,
        2 => CursorStamp::CursorAtOrAbove,
        _ => CursorStamp::HeaderAbsent,
    })
}

/// The `ARGV` `scripts/stamp_cursor_if_live.lua` receives for `pairs`, with an
/// optional cursor-regression floor.
///
/// - `None` — the flat `field, value, …` list, EVEN length. Exactly what every
///   stamp site sent before the floor existed, and what the checkpoint loop's
///   three arms still send ([`stamp_obj_header_cursor`]).
/// - `Some(floor)` — the floor as a leading decimal, then the same list: ODD
///   length, which is how the script tells the two shapes apart
///   ([`stamp_obj_header_cursor_below`]).
///
/// `pub` so the shape each call site sends can be pinned directly.
pub fn stamp_cursor_argv(pairs: &[(&str, String)], refuse_at_or_above: Option<u64>) -> Vec<String> {
    debug_assert!(
        !pairs.is_empty(),
        "a cursor stamp with no pairs would run `HSET key` with no field/value arguments, which \
         Valkey rejects with a wrong-number-of-arguments error"
    );
    let mut argv: Vec<String> = Vec::with_capacity(pairs.len() * 2 + 1);
    if let Some(floor) = refuse_at_or_above {
        argv.push(floor.to_string());
    }
    for (field, value) in pairs {
        argv.push((*field).to_owned());
        argv.push(value.clone());
    }
    argv
}

/// Classify an `update_layer` delta as fast-path (Lua-safe) vs slow-path
/// (requires loading the full object snapshot).
///
/// The fast path handles only deltas whose Rust-side `apply_delta` does not
/// need section / children state to compute the mutation list:
///
/// - `None` — empty delta, touches only the header.
/// - Pure field-and-time-sample deltas (no structural commands, not a
///   reset).
/// - Deltas whose only structural commands are `DiffDeleteSection`, which
///   is silently ignored on ingest (SDD §9.1.4f).
///
/// The inverse — any reset (`is_diff && base_version == 0`) or any non-
/// ignored `StructCommand` — must fall through to the snapshot-loading
/// legacy path. This mirrors the `needs_snapshot` branch previously inlined
/// into `update_layer_once`.
pub fn is_fast_path(delta: &Option<oodf_ffi::OwnedDelta>) -> bool {
    match delta {
        None => true,
        Some(d) => {
            if d.is_diff && d.base_version == 0 {
                return false;
            }
            // Any non-`DiffDeleteSection` struct command reads section
            // state, so it needs the snapshot.
            !d.struct_commands.iter().any(|c| match c {
                oodf_ffi::StructCommand::DiffDeleteSection(_) => false, // silently ignored on ingest (SDD §9.1.4f)
                oodf_ffi::StructCommand::CreateSection(_)
                | oodf_ffi::StructCommand::DeleteSection(_)
                | oodf_ffi::StructCommand::MoveSection(_)
                | oodf_ffi::StructCommand::ReorderChildren(_)
                | oodf_ffi::StructCommand::DiffChangeSection(_) => true,
            })
        }
    }
}

/// Output of [`encode_delta`]: the parallel `KEYS` / `ARGV` vectors to pass
/// to `UPDATE_LAYER_FAST_SCRIPT`.
pub struct EncodedMutations {
    /// Per-mutation keys in the order the Lua script will consume them
    /// (see `scripts/update_layer_fast.lua`'s KEYS contract). Does NOT
    /// include the [`UPDATE_SCRIPT_SHARED_KEYS`] shared-prefix keys (the
    /// `UPDATE_SCRIPT_KEY_*` slots) — the caller prepends those
    /// ([`update_script_shared_keys`]).
    pub mutation_keys: Vec<String>,
    /// Per-mutation argv records. Does NOT include the fixed-prefix ARGV
    /// block or the mode slot — the caller prepends them
    /// ([`UpdateScriptPrefix::build`]).
    pub mutation_argv: Vec<String>,
    /// Number of mutation records in `mutation_argv`. Passed in the
    /// fixed-prefix ARGV block so the Lua script knows how many records
    /// to walk.
    pub num_mutations: u64,
}

/// Encode the fast-path portion of an `OwnedDelta` into the Lua script's
/// tagged-ARGV format.
///
/// Walks the delta directly (not the `StorageMutation` list) because every
/// sequence-derived field (`version`, `set_order`, `next_set_order`) is
/// computed server-side inside the atomic Lua block — see the TOCTOU
/// rationale at the top of this module. Only TOCTOU-free inputs — OODF
/// delta-level `set_order` offsets, value/hash/size hash-field pairs,
/// ZADD / ZREM member strings — flow through ARGV.
///
/// The caller guarantees `is_fast_path(delta) == true` before calling;
/// structural commands that reach this point (other than the silently-
/// ignored `DiffDeleteSection`) indicate caller misuse and are skipped
/// with a debug-build assertion. The caller also guarantees the delta has
/// passed `validate_delta_bounds`; an external hash is encoded verbatim on
/// that basis (see `value_hash_fields`).
pub fn encode_delta(
    delta: &Option<oodf_ffi::OwnedDelta>,
    keys: &KeySchema,
    oid_str: &str,
) -> EncodedMutations {
    let mut mutation_keys: Vec<String> = Vec::new();
    let mut mutation_argv: Vec<String> = Vec::new();
    let mut num_mutations: u64 = 0;

    let Some(delta) = delta else {
        // Empty delta: no mutations, the script still HSETs the header
        // and XADDs a stream entry.
        return EncodedMutations {
            mutation_keys,
            mutation_argv,
            num_mutations,
        };
    };

    // Guard against caller misuse: the fast path must not see any struct
    // command that emits storage mutations. `DiffDeleteSection` is
    // silently ignored on ingest (SDD §9.1.4f) so it's allowed through.
    for c in &delta.struct_commands {
        debug_assert!(
            matches!(c, oodf_ffi::StructCommand::DiffDeleteSection(_)),
            "encode_delta called with non-fast-path struct command {c:?}; caller \
             must filter via `is_fast_path` before invoking the Lua fast path"
        );
    }

    // Field assignments. Emitted in delta order; the Lua script advances
    // its `next_set_order` cursor per record using the OODF
    // `delta_set_order` offset we pass through.
    for sf in &delta.set_fields {
        let sid = sf.section_id.to_string();
        mutation_keys.push(keys.obj_field(oid_str, &sid, &sf.key_name));

        let value_hfields =
            value_hash_fields(sf.ext_value_size, sf.ext_value_index, &sf.value_or_ext_hash);
        mutation_argv.push("SF".to_owned());
        mutation_argv.push(value_hfields.len().to_string());
        for (name, val) in value_hfields {
            mutation_argv.push(name);
            mutation_argv.push(val);
        }
        mutation_argv.push(sf.set_order.to_string());
        mutation_argv.push(KeySchema::field_member(&sid, &sf.key_name));
        num_mutations += 1;
    }

    // Time sample assignments.
    for ts in &delta.time_samples {
        let sid = ts.section_id.to_string();
        let time_f64 = ts.time;
        mutation_keys.push(keys.obj_time_sample(oid_str, &sid, time_f64));

        let value_hfields =
            value_hash_fields(ts.ext_value_size, ts.ext_value_index, &ts.value_or_ext_hash);
        mutation_argv.push("ST".to_owned());
        mutation_argv.push(value_hfields.len().to_string());
        for (name, val) in value_hfields {
            mutation_argv.push(name);
            mutation_argv.push(val);
        }
        mutation_argv.push(KeySchema::time_sample_member(&sid, time_f64));
        num_mutations += 1;
    }

    EncodedMutations {
        mutation_keys,
        mutation_argv,
        num_mutations,
    }
}

/// Digest token: this record carries no pre-image check. Either an earlier
/// record of the same commit carries its key's check, or the record is a
/// children list under a dangling parent, which is never checked.
pub const DIGEST_NONE: &str = "none";

/// Digest token: the cache believed the record's key absent before the plan,
/// so the store's key must be absent. For a children list this is also the
/// absent marker [`children_list_digest`] returns, which cannot collide with a
/// 40-hex digest.
pub const DIGEST_ABSENT: &str = "absent";

/// Digest token: the six stored columns of a section row follow.
pub const DIGEST_ROW: &str = "row";

/// Digest token: the cache believed the record's row exists before the plan.
pub const DIGEST_EXISTS: &str = "exists";

/// The children-list digest the structural commit checks a stored list
/// against (SR-6.3): the lowercase 40-hex SHA-1 of the list's ids as decimal
/// strings joined by `,`, byte-identical to `update_layer_fast.lua`'s
/// `redis.sha1hex(table.concat(LRANGE(ch:{p}, 0, -1), ","))` over a list
/// stored by the same writers; or [`DIGEST_ABSENT`] for a missing or empty
/// list, since an empty list has no stored form.
pub fn children_list_digest(children: Option<&[u64]>) -> String {
    use sha1::{Digest, Sha1};

    let children = match children {
        Some(children) if !children.is_empty() => children,
        _ => return DIGEST_ABSENT.to_owned(),
    };
    let mut hasher = Sha1::new();
    for (i, child) in children.iter().enumerate() {
        if i > 0 {
            hasher.update(b",");
        }
        hasher.update(child.to_string().as_bytes());
    }
    hex_encode_bytes(&hasher.finalize())
}

/// Output of [`encode_mutations`]: the structural-mode untouched-row count
/// block, the records and their per-record `KEYS`, to place after the
/// script's fixed `KEYS` / `ARGV` prefixes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedStructuralRecords {
    /// The untouched-row count block (`PreImageDigests`' untouched-row
    /// counts): `K`, then `hash_hex_1, n_1, …, hash_hex_K, n_K`, each hash 64
    /// lowercase hex and strictly ascending, each count decimal. The caller
    /// places it right after `expected_commit_chain` and before
    /// [`Self::mutation_argv`] (the script header's structural-mode layout),
    /// so the records still run to the end of ARGV.
    pub untouched_ext_block: Vec<String>,
    /// One key per record, in record order: the key the record writes (see
    /// the record vocabulary in `scripts/update_layer_fast.lua`). The fixed
    /// keys, including the object-level indexes the records also write, are
    /// not here; the caller places them.
    pub mutation_keys: Vec<String>,
    /// The records, back to back, each a tag followed by its slots. The fixed
    /// ARGV prefix is not here; the caller places it.
    pub mutation_argv: Vec<String>,
    /// The number of records in [`Self::mutation_argv`], `HDR` included: the
    /// count the script walks.
    pub num_mutations: u64,
    /// The number of records excluding `HDR`: the commit's size in records,
    /// as `CommitObserver::commit_size` reports it.
    pub num_size_records: u64,
}

/// Encode a structural plan into `update_layer_fast.lua`'s structural-mode
/// records (SR-6.3).
///
/// The record vocabulary — tags, slot order, the meaning of each slot and the
/// digest each record carries — is defined once, in the header comment of
/// `scripts/update_layer_fast.lua` ("Structural-mode records"); this function
/// produces exactly that. One record per mutation, 1:1 and in `mutations`
/// order, each with one per-record key. Every value is the string the WATCH
/// path's `execute_mutation` writes for the same mutation, and `SFA` / `STA`
/// carry Rust's absolute `set_order` / `authoring_set_order` / `version`, so
/// the script does no arithmetic on them. Stateless.
///
/// `digests` must be [`PreImageDigests::from_plan`] of these same
/// `mutations`; `next_set_order` is the plan's post-apply
/// `ObjectState::next_set_order`, which the `HDR` record carries because
/// `SetObjectHeader` has no such field (`StructuralCommit::next_set_order`).
/// The digests' untouched-row counts are rendered separately, as
/// [`EncodedStructuralRecords::untouched_ext_block`], in the hash order they
/// are held in.
///
/// # Errors
///
/// `ServiceError::Internal` when `digests` were not built from `mutations`:
/// their plan fingerprint differs (see "Bound to its plan" on
/// [`PreImageDigests`]), or, defensively, a digest is of the wrong kind for
/// its record. Mismatched digests would check the wrong keys' beliefs, so the
/// encoder refuses rather than send them.
///
/// [`PreImageDigests::from_plan`]: crate::storage::structural::PreImageDigests::from_plan
pub fn encode_mutations(
    mutations: &[StorageMutation],
    digests: &PreImageDigests,
    next_set_order: u64,
    keys: &KeySchema,
    oid_str: &str,
) -> Result<EncodedStructuralRecords, ServiceError> {
    if !digests.built_from(mutations) {
        return Err(ServiceError::Internal(format!(
            "structural encode: the pre-image digests were not built from this \
             {}-mutation plan",
            mutations.len()
        )));
    }

    let mut mutation_keys: Vec<String> = Vec::with_capacity(mutations.len());
    let mut mutation_argv: Vec<String> = Vec::new();
    let mut num_size_records: u64 = 0;

    for (index, mutation) in mutations.iter().enumerate() {
        let digest = digests.for_record(index);
        let argv = &mut mutation_argv;
        match mutation {
            StorageMutation::SetSection {
                section,
                previous: _,
            } => {
                let sid = section.section_id.to_string();
                mutation_keys.push(keys.obj_section(oid_str, &sid));
                argv.push("SEC".to_owned());
                argv.push(sid);
                argv.extend(section_hash_fields(section).map(|(_, value)| value));
                push_section_digest(argv, digest, index)?;
            }
            StorageMutation::DeleteSection {
                section_id,
                parent_id: _,
                name: _,
            } => {
                let sid = section_id.to_string();
                mutation_keys.push(keys.obj_section(oid_str, &sid));
                argv.push("DSEC".to_owned());
                argv.push(sid);
                push_section_digest(argv, digest, index)?;
            }
            StorageMutation::CreateTombstone {
                section_id,
                first_version,
                last_version,
            } => {
                let sid = section_id.to_string();
                mutation_keys.push(keys.obj_deleted_section(oid_str, &sid));
                argv.push("TOMB".to_owned());
                argv.push(sid);
                argv.push(first_version.to_string());
                argv.push(last_version.to_string());
                expect_no_digest(digest, index, "TOMB")?;
            }
            StorageMutation::SetChildren {
                section_id,
                children,
            } => {
                let pid = section_id.to_string();
                mutation_keys.push(keys.obj_children(oid_str, &pid));
                argv.push("CHL".to_owned());
                argv.push(pid);
                argv.push(children.len().to_string());
                argv.extend(children.iter().map(u64::to_string));
                push_list_digest(argv, digest, index)?;
            }
            StorageMutation::DeleteChildrenList { section_id } => {
                let pid = section_id.to_string();
                mutation_keys.push(keys.obj_children(oid_str, &pid));
                argv.push("DCHL".to_owned());
                argv.push(pid);
                push_list_digest(argv, digest, index)?;
            }
            StorageMutation::SetField { field } => {
                let sid = field.section_id.to_string();
                mutation_keys.push(keys.obj_field(oid_str, &sid, &field.key_name));
                argv.push("SFA".to_owned());
                push_value_hash_fields(argv, &field.value);
                argv.push(field.set_order.to_string());
                argv.push(field.authoring_set_order.to_string());
                argv.push(field.version.to_string());
                argv.push(KeySchema::field_member(&sid, &field.key_name));
                expect_no_digest(digest, index, "SFA")?;
            }
            StorageMutation::DeleteField {
                section_id,
                key_name,
            } => {
                let sid = section_id.to_string();
                mutation_keys.push(keys.obj_field(oid_str, &sid, key_name));
                argv.push("DF".to_owned());
                argv.push(KeySchema::field_member(&sid, key_name));
                push_row_digest(argv, digest, index)?;
            }
            StorageMutation::SetTimeSample { sample } => {
                let sid = sample.section_id.to_string();
                let time = sample.time.into_inner();
                mutation_keys.push(keys.obj_time_sample(oid_str, &sid, time));
                argv.push("STA".to_owned());
                push_value_hash_fields(argv, &sample.value);
                argv.push(sample.version.to_string());
                argv.push(KeySchema::time_sample_member(&sid, time));
                expect_no_digest(digest, index, "STA")?;
            }
            StorageMutation::DeleteTimeSample { section_id, time } => {
                let sid = section_id.to_string();
                let time = time.into_inner();
                mutation_keys.push(keys.obj_time_sample(oid_str, &sid, time));
                argv.push("DT".to_owned());
                argv.push(KeySchema::time_sample_member(&sid, time));
                push_row_digest(argv, digest, index)?;
            }
            StorageMutation::SetObjectHeader {
                object_id: _,
                sequence_num,
                first_version,
                source_format,
            } => {
                mutation_keys.push(keys.obj_header(oid_str));
                argv.push("HDR".to_owned());
                argv.push(sequence_num.to_string());
                argv.push(first_version.to_string());
                argv.push(next_set_order.to_string());
                match source_format {
                    Some(src_fmt) => {
                        argv.push("1".to_owned());
                        argv.push(src_fmt.clone());
                    }
                    None => argv.push("0".to_owned()),
                }
                expect_no_digest(digest, index, "HDR")?;
                // The header bump is not counted in the commit's size.
                continue;
            }
        }
        num_size_records += 1;
    }

    let counts = digests.untouched_ext_counts();
    let mut untouched_ext_block: Vec<String> = Vec::with_capacity(1 + 2 * counts.len());
    untouched_ext_block.push(counts.len().to_string());
    for (hash, n) in counts {
        untouched_ext_block.push(hex_encode_bytes(hash));
        untouched_ext_block.push(n.to_string());
    }

    Ok(EncodedStructuralRecords {
        untouched_ext_block,
        mutation_keys,
        mutation_argv,
        num_mutations: mutations.len() as u64,
        num_size_records,
    })
}

/// Append a value's hash-field pairs as `count, name1, value1, …`, in
/// `execute_mutation`'s string forms ([`layer::encode_value_fields`]).
fn push_value_hash_fields(argv: &mut Vec<String>, value: &FieldValue) {
    let mut pairs: Vec<(String, String)> = Vec::with_capacity(2);
    layer::encode_value_fields(value, &mut pairs);
    argv.push(pairs.len().to_string());
    for (name, value) in pairs {
        argv.push(name);
        argv.push(value);
    }
}

/// Append a `SEC` / `DSEC` record's digest slots.
fn push_section_digest(
    argv: &mut Vec<String>,
    digest: Option<&PreImage>,
    index: usize,
) -> Result<(), ServiceError> {
    match digest {
        None => argv.push(DIGEST_NONE.to_owned()),
        Some(PreImage::Section(None)) => argv.push(DIGEST_ABSENT.to_owned()),
        Some(PreImage::Section(Some(columns))) => {
            argv.push(DIGEST_ROW.to_owned());
            argv.extend(columns.iter().cloned());
        }
        Some(other) => return Err(digest_kind_mismatch(index, "section row", other)),
    }
    Ok(())
}

/// Append a `CHL` / `DCHL` record's digest slot.
fn push_list_digest(
    argv: &mut Vec<String>,
    digest: Option<&PreImage>,
    index: usize,
) -> Result<(), ServiceError> {
    match digest {
        None => argv.push(DIGEST_NONE.to_owned()),
        Some(PreImage::ChildrenList(list_digest)) => argv.push(list_digest.clone()),
        Some(other) => return Err(digest_kind_mismatch(index, "children list", other)),
    }
    Ok(())
}

/// Append a `DF` / `DT` record's digest slot.
fn push_row_digest(
    argv: &mut Vec<String>,
    digest: Option<&PreImage>,
    index: usize,
) -> Result<(), ServiceError> {
    match digest {
        None => argv.push(DIGEST_NONE.to_owned()),
        Some(PreImage::RowExists(true)) => argv.push(DIGEST_EXISTS.to_owned()),
        Some(PreImage::RowExists(false)) => argv.push(DIGEST_ABSENT.to_owned()),
        Some(other) => return Err(digest_kind_mismatch(index, "row existence", other)),
    }
    Ok(())
}

/// Refuse a digest on a record kind that carries none.
fn expect_no_digest(
    digest: Option<&PreImage>,
    index: usize,
    tag: &str,
) -> Result<(), ServiceError> {
    match digest {
        None => Ok(()),
        Some(other) => Err(ServiceError::Internal(format!(
            "structural encode: record {index} ({tag}) carries no digest, got {other:?}"
        ))),
    }
}

fn digest_kind_mismatch(index: usize, expected: &str, got: &PreImage) -> ServiceError {
    ServiceError::Internal(format!(
        "structural encode: record {index} expects a {expected} digest, got {got:?}"
    ))
}

/// Encode the value-portion hash fields for a field or time sample. Matches
/// the `value` / `hash` + `size` ordering written by
/// `layer::encode_value_fields`, but only the value portion — the Lua
/// script appends `set_order` + `version` (SetField) or `version`
/// (SetTimeSample) itself using sequence-derived values it computes
/// atomically.
///
/// An external hash is written verbatim: `validate_delta_bounds`, which
/// `LayerStore::update_layer` runs ahead of the backend dispatch, has already
/// refused any `ext_value_size > 0` entry whose hash is not exactly 32 bytes,
/// so the hex here is the same one the txn path's `field_value_from_oodf`
/// stores.
fn value_hash_fields(
    ext_value_size: u64,
    _ext_value_index: u32,
    value_or_ext_hash: &[u8],
) -> Vec<(String, String)> {
    if ext_value_size > 0 {
        debug_assert_eq!(
            value_or_ext_hash.len(),
            32,
            "encode_delta called with an unvalidated external hash; `validate_delta_bounds` \
             must run before the Lua fast path"
        );
        vec![
            ("hash".to_owned(), hex_encode_bytes(value_or_ext_hash)),
            ("size".to_owned(), ext_value_size.to_string()),
        ]
    } else {
        vec![("value".to_owned(), BASE64.encode(value_or_ext_hash))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oodf_ffi::{CommandDiffDeleteSection, CommandReorderChildren, OwnedDelta, StructCommand};

    #[test]
    fn is_fast_path_none_is_fast() {
        assert!(is_fast_path(&None));
    }

    #[test]
    fn is_fast_path_reset_is_slow() {
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            ..Default::default()
        };
        assert!(!is_fast_path(&Some(delta)));
    }

    #[test]
    fn is_fast_path_pure_diff_is_fast() {
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 7,
            ..Default::default()
        };
        assert!(is_fast_path(&Some(delta)));
    }

    #[test]
    fn is_fast_path_non_diff_fields_only_is_fast() {
        let delta = OwnedDelta {
            is_diff: false,
            base_version: 0,
            ..Default::default()
        };
        assert!(is_fast_path(&Some(delta)));
    }

    #[test]
    fn is_fast_path_diff_delete_section_only_is_fast() {
        // DiffDeleteSection is silently ignored on ingest (SDD §9.1.4f) —
        // its presence alone does not require the snapshot.
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 7,
            struct_commands: vec![StructCommand::DiffDeleteSection(CommandDiffDeleteSection {
                section_id: 42,
            })],
            ..Default::default()
        };
        assert!(is_fast_path(&Some(delta)));
    }

    #[test]
    fn is_fast_path_any_other_struct_command_is_slow() {
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 7,
            struct_commands: vec![StructCommand::ReorderChildren(CommandReorderChildren {
                section_id: 1,
                children_list_id: 0,
                children_list: vec![2, 3],
            })],
            ..Default::default()
        };
        assert!(!is_fast_path(&Some(delta)));
    }

    // ── `is_fast_path` equals the domain's replay-skip class ───────────────
    //
    // `domain::delta`'s `ApplyMode::Replay` skips the post-mutation depth
    // walk for a delta that cannot change the section tree, and documents
    // that class as the one this function admits. The predicate itself is
    // private to `domain::delta`, so the equivalence is observed through the
    // public `apply_delta_into_mode`: on a stored tree one level deeper than
    // `MAX_SECTION_TREE_DEPTH`, a delta that walks — and does not shorten the
    // chain — is refused `SectionTreeTooDeep`, so whether it walked is
    // visible from the result alone.

    /// One structural command of each `StructCommand` variant, none of which
    /// shortens [`too_deep_chain_state`]'s chain or fails on its own. The
    /// `match` is exhaustive with no wildcard, so a new variant does not
    /// compile until it has a case here.
    fn one_command_per_variant() -> Vec<StructCommand> {
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;
        use oodf_ffi::{
            CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection,
            CommandMoveSection,
        };

        let leaf = MAX_SECTION_TREE_DEPTH as u64 + 1;
        let commands = vec![
            StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: 900_000,
                section_name: "New".to_owned(),
                section_type: 1,
            }),
            // An absent id: nothing is removed.
            StructCommand::DeleteSection(CommandDeleteSection {
                parent_id: 0,
                section_id: 900_000,
            }),
            // A rename in place.
            StructCommand::MoveSection(CommandMoveSection {
                old_parent_id: leaf - 1,
                new_parent_id: leaf - 1,
                section_id: leaf,
                new_name: "Renamed".to_owned(),
            }),
            StructCommand::ReorderChildren(CommandReorderChildren {
                section_id: 0,
                children_list_id: 0,
                children_list: vec![1],
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: 900_000,
                section_name: "New".to_owned(),
                section_type: 1,
                section_order: 0,
            }),
            StructCommand::DiffDeleteSection(CommandDiffDeleteSection { section_id: 1 }),
        ];
        let mut seen = std::collections::HashSet::new();
        for command in &commands {
            seen.insert(match command {
                StructCommand::CreateSection(_) => "CreateSection",
                StructCommand::DeleteSection(_) => "DeleteSection",
                StructCommand::MoveSection(_) => "MoveSection",
                StructCommand::ReorderChildren(_) => "ReorderChildren",
                StructCommand::DiffChangeSection(_) => "DiffChangeSection",
                StructCommand::DiffDeleteSection(_) => "DiffDeleteSection",
            });
        }
        assert_eq!(seen.len(), 6, "one command of every variant");
        commands
    }

    /// A linear chain `1 → 2 → … → cap + 1` under the root: one level deeper
    /// than `MAX_SECTION_TREE_DEPTH`. A copy of the `domain::delta` test
    /// module's helper of the same name, which that module keeps private;
    /// change the two together.
    fn too_deep_chain_state() -> crate::domain::types::ObjectState {
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;
        use crate::domain::types::{ObjectState, Section};

        let mut state = ObjectState {
            object_id: 1,
            ..Default::default()
        };
        for id in 1..=(MAX_SECTION_TREE_DEPTH as u64 + 1) {
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: id - 1,
                    name: format!("S{id}"),
                    section_type: 1,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
            state.children.entry(id - 1).or_default().push(id);
        }
        state
    }

    /// Whether a `Replay` apply of `delta` over [`too_deep_chain_state`] ran
    /// the depth walk: `true` when it was refused `SectionTreeTooDeep`,
    /// `false` when it applied. Any other error is a fixture defect.
    fn replay_walks_the_depth_check(delta: &Option<OwnedDelta>, label: &str) -> bool {
        use crate::domain::delta::{
            apply_delta_into_mode, ApplyDeltaStats, ApplyMode, DeltaError, MutationSink,
        };

        let mut state = too_deep_chain_state();
        match apply_delta_into_mode(
            &mut state,
            delta.clone(),
            &mut MutationSink::Discard,
            &mut ApplyDeltaStats::default(),
            ApplyMode::Replay,
        ) {
            Ok(()) => false,
            Err(DeltaError::SectionTreeTooDeep { .. }) => true,
            Err(other) => panic!("{label}: fixture delta must apply or walk; got {other:?}"),
        }
    }

    /// Across `None`, an empty delta, a field-only delta, one delta per
    /// `StructCommand` variant, and a `DiffDeleteSection` beside a variant
    /// that walks, the deltas `Replay` lets skip the depth walk are exactly the
    /// deltas `is_fast_path` admits.
    ///
    /// A reset is the one input the walk cannot discriminate: it clears the
    /// tree, and with no structural command after it the walk runs over an
    /// empty tree and cannot fail. So for the reset this pins only what is
    /// observable: `Replay` admits exactly what `Commit` (which always walks)
    /// admits. That `is_fast_path` refuses a reset is
    /// `is_fast_path_reset_is_slow`'s.
    #[test]
    fn is_fast_path_equals_the_domain_replay_depth_walk_skip_class() {
        use crate::domain::delta::{
            apply_delta_into_mode, ApplyDeltaStats, ApplyMode, MutationSink,
        };

        let incremental = |struct_commands: Vec<StructCommand>| OwnedDelta {
            is_diff: true,
            base_version: 7,
            struct_commands,
            ..Default::default()
        };
        let mut cases: Vec<(String, Option<OwnedDelta>)> = vec![
            ("None (an empty commit)".to_owned(), None),
            ("empty delta".to_owned(), Some(OwnedDelta::default())),
            (
                "field only".to_owned(),
                Some(OwnedDelta {
                    set_fields: vec![oodf_ffi::SetField {
                        section_id: 1,
                        key_name: "k".to_owned(),
                        value_or_ext_hash: vec![3; 4],
                        ext_value_size: 0,
                        ext_value_index: 0,
                        set_order: 1,
                    }],
                    ..Default::default()
                }),
            ),
        ];
        for command in one_command_per_variant() {
            cases.push((format!("{command:?}"), Some(incremental(vec![command]))));
        }
        let mut beside = one_command_per_variant();
        beside.retain(|c| matches!(c, StructCommand::CreateSection(_)));
        beside.insert(
            0,
            StructCommand::DiffDeleteSection(CommandDiffDeleteSection { section_id: 1 }),
        );
        cases.push((
            "DiffDeleteSection beside a CreateSection".to_owned(),
            Some(incremental(beside)),
        ));

        for (label, delta) in &cases {
            assert_eq!(
                !replay_walks_the_depth_check(delta, label),
                is_fast_path(delta),
                "{label}: Replay skips the depth walk exactly for the deltas the Lua fast path \
                 commits"
            );
        }

        let reset = Some(OwnedDelta {
            is_diff: true,
            base_version: 0,
            ..Default::default()
        });
        let outcome = |mode| {
            let mut state = too_deep_chain_state();
            apply_delta_into_mode(
                &mut state,
                reset.clone(),
                &mut MutationSink::Discard,
                &mut ApplyDeltaStats::default(),
                mode,
            )
            .map(|()| (state.sections.len(), state.sequence_num))
            .map_err(|e| e.to_string())
        };
        assert_eq!(
            outcome(ApplyMode::Replay),
            outcome(ApplyMode::Commit),
            "a replayed reset admits exactly what a committed one does"
        );
    }

    #[test]
    fn script_static_compiles_and_has_sha1() {
        // The SHA-1 hash is derived from the Lua source; pin that it is a
        // 40-char hex string so a corruption of the embedded script is
        // caught at unit-test time rather than at first `evalsha_with_reload`.
        let hash = UPDATE_LAYER_FAST_SCRIPT.sha1();
        assert_eq!(
            hash.len(),
            40,
            "fred Script::from_lua should produce a 40-char SHA-1 hex"
        );
        assert!(
            hash.chars().all(|c| c.is_ascii_hexdigit()),
            "SHA-1 must be lowercase hex; got {hash}"
        );
    }

    #[test]
    fn encode_delta_none_emits_no_mutations() {
        let keys = KeySchema::default();
        let enc = encode_delta(&None, &keys, "42");
        assert_eq!(enc.num_mutations, 0);
        assert!(enc.mutation_keys.is_empty());
        assert!(enc.mutation_argv.is_empty());
    }

    #[test]
    fn encode_delta_set_field_emits_sf_record_without_version_or_set_order() {
        // The script computes version + set_order inside the atomic block,
        // so the Rust encoder must not emit them. This pins the new
        // TOCTOU-free ARGV contract.
        use oodf_ffi::SetField;

        let keys = KeySchema::default();
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 7,
            set_fields: vec![SetField {
                section_id: 11,
                key_name: "thing".to_owned(),
                value_or_ext_hash: b"payload".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 3,
            }],
            ..Default::default()
        };

        let enc = encode_delta(&Some(delta), &keys, "42");

        assert_eq!(enc.num_mutations, 1);
        assert_eq!(enc.mutation_keys.len(), 1);
        assert_eq!(enc.mutation_argv[0], "SF");
        // num_value_hfields = 1 (inline "value" pair).
        assert_eq!(enc.mutation_argv[1], "1");
        assert_eq!(enc.mutation_argv[2], "value");
        // ARGV[3] is the base64 of "payload".
        assert_eq!(enc.mutation_argv[3], BASE64.encode(b"payload"));
        // delta_set_order — passed through untouched.
        assert_eq!(enc.mutation_argv[4], "3");
        // ZADD member.
        assert_eq!(enc.mutation_argv[5], KeySchema::field_member("11", "thing"));
        // No further ARGV entries — version + set_order are NOT here.
        assert_eq!(enc.mutation_argv.len(), 6);
    }

    #[test]
    fn encode_delta_time_sample_emits_st_record_without_version() {
        // Symmetric to SetField: `version` is not in ARGV on the ST path
        // either — the script computes it atomically.
        use oodf_ffi::TimeSample;

        let keys = KeySchema::default();
        let delta = OwnedDelta {
            is_diff: false,
            base_version: 0,
            time_samples: vec![TimeSample {
                section_id: 5,
                time: 1.5,
                value_or_ext_hash: b"tick".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..Default::default()
        };

        let enc = encode_delta(&Some(delta), &keys, "7");
        assert_eq!(enc.num_mutations, 1);
        assert_eq!(enc.mutation_argv[0], "ST");
        assert_eq!(enc.mutation_argv[1], "1");
        assert_eq!(enc.mutation_argv[2], "value");
        assert_eq!(enc.mutation_argv[3], BASE64.encode(b"tick"));
        // ZADD member for time 1.5.
        assert_eq!(
            enc.mutation_argv[4],
            KeySchema::time_sample_member("5", 1.5)
        );
        assert_eq!(enc.mutation_argv.len(), 5);
    }

    #[test]
    fn encode_delta_external_value_emits_hash_and_size_pairs() {
        // External-value path — the value hash fields switch from a
        // single "value" pair to ("hash", ..) + ("size", ..). This pins
        // the wire format and the verbatim hex of the 32-byte hash.
        use oodf_ffi::SetField;

        let keys = KeySchema::default();
        let hash_blob: Vec<u8> = (0..32).collect();
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 7,
            set_fields: vec![SetField {
                section_id: 1,
                key_name: "ext".to_owned(),
                value_or_ext_hash: hash_blob.clone(),
                ext_value_size: 4096,
                ext_value_index: 2,
                set_order: 0,
            }],
            ..Default::default()
        };

        let enc = encode_delta(&Some(delta), &keys, "1");
        assert_eq!(enc.num_mutations, 1);
        assert_eq!(enc.mutation_argv[0], "SF");
        // Two value hash-field pairs: "hash" + "size".
        assert_eq!(enc.mutation_argv[1], "2");
        assert_eq!(enc.mutation_argv[2], "hash");
        assert_eq!(enc.mutation_argv[3], hex_encode_bytes(&hash_blob));
        assert_eq!(enc.mutation_argv[4], "size");
        assert_eq!(enc.mutation_argv[5], "4096");
        // delta_set_order then zadd_member.
        assert_eq!(enc.mutation_argv[6], "0");
        assert_eq!(enc.mutation_argv[7], KeySchema::field_member("1", "ext"));
    }

    #[test]
    fn encode_delta_diff_delete_section_ignored_emits_no_record() {
        // DiffDeleteSection is silently ignored on ingest (SDD §9.1.4f).
        // It passes `is_fast_path` but emits nothing through the encoder.
        let keys = KeySchema::default();
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 7,
            struct_commands: vec![StructCommand::DiffDeleteSection(CommandDiffDeleteSection {
                section_id: 99,
            })],
            ..Default::default()
        };

        let enc = encode_delta(&Some(delta), &keys, "1");
        assert_eq!(enc.num_mutations, 0);
        assert!(enc.mutation_argv.is_empty());
    }
}

#[cfg(test)]
mod encode_mutations_tests {
    //! [`encode_mutations`] and [`children_list_digest`]: the structural-mode
    //! record contract defined in `scripts/update_layer_fast.lua`'s header
    //! ("Structural-mode records"), which Task 2's script parses.
    //!
    //! One test per tag pins that tag's exact slots and its one per-record
    //! key. Ids are chosen so their decimal form is unmistakable: [`SID`]'s
    //! hex form differs, and [`ID_ABOVE_F64_EXACT`] would read differently
    //! had it passed through a float. The plan-level tests pin 1:1 emission
    //! order, the reset plan's wipe and rebuild with `HDR` last, the size
    //! counts, a fixed list-digest vector for the cross-language pin, and the
    //! refusal of digests built from another plan. Two tests pin the
    //! untouched-row count block: `K`, then each hash's 64 lowercase hex and
    //! decimal count ascending by hash, and `K = 0` alone for a plan naming no
    //! external hash.
    //!
    //! Which digest a record carries, and which counts the block holds, is
    //! `structural::PreImageDigests`'s rule, pinned in that module's tests;
    //! here only how each renders.

    use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField as OodfSetField, StructCommand};
    use ordered_float::OrderedFloat;

    use super::*;
    use crate::domain::delta::apply_delta;
    use crate::domain::types::{Field, ObjectState, Section, SectionKey, TimeSample};

    /// The object id every record's key is built under.
    const OID: &str = "42";
    /// `0x1234`: an id whose decimal and hex forms differ.
    const SID: u64 = 4660;
    /// `2^53 + 1`: the smallest id an `f64` cannot hold exactly.
    const ID_ABOVE_F64_EXACT: u64 = (1 << 53) + 1;
    /// A field key name.
    const KEY: &str = "xformOp";
    /// `hex("xformOp")`: [`KEY`]'s form in a field key and member.
    const KEY_HEX: &str = "78666f726d4f70";

    /// The fixed list-digest vector: ids `[5, 12, 1000]` and the lowercase
    /// SHA-1 of the bytes `5,12,1000`, computed outside Rust (`sha1sum`) so
    /// the pin is not the implementation checking itself. Task 2 pins the
    /// same pair against Valkey's `redis.sha1hex`.
    const LIST_VECTOR: [u64; 3] = [5, 12, 1000];
    const LIST_VECTOR_SHA1: &str = "ed3e4ac759f1f0b121ae29a57350ea82a6dbb0c3";

    fn section(
        section_id: u64,
        parent_id: u64,
        name: &str,
        section_type: u8,
        section_order: u32,
        first_version: u64,
        last_version: u64,
    ) -> Section {
        Section {
            section_id,
            parent_id,
            name: name.to_owned(),
            section_type,
            section_order,
            first_version,
            last_version,
        }
    }

    /// A pre-state holding exactly `sections`.
    fn pre_with(sections: &[Section]) -> ObjectState {
        let mut state = ObjectState::default();
        for s in sections {
            state.sections.insert(s.section_id, s.clone());
        }
        state
    }

    /// Encode `mutations` with the digests [`PreImageDigests::from_plan`]
    /// builds from `pre`, as a structural commit pairs them.
    fn encode(
        mutations: &[StorageMutation],
        pre: &ObjectState,
        next_set_order: u64,
    ) -> EncodedStructuralRecords {
        let digests = PreImageDigests::from_plan(mutations, pre);
        encode_mutations(
            mutations,
            &digests,
            next_set_order,
            &KeySchema::default(),
            OID,
        )
        .expect("digests built from the plan they encode must be accepted")
    }

    fn strings(slots: &[&str]) -> Vec<String> {
        slots.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Split `argv` into its records by the vocabulary in the script header:
    /// the walk the script's structural mode does. Panics on an unknown tag.
    fn split_records(argv: &[String]) -> Vec<Vec<String>> {
        // A `SEC` / `DSEC` digest is one slot, or `row` plus six.
        let section_digest_len = |at: usize| if argv[at] == DIGEST_ROW { 7 } else { 1 };
        let count = |at: usize| -> usize { argv[at].parse().expect("a count slot") };
        let mut records = Vec::new();
        let mut at = 0;
        while at < argv.len() {
            let len = match argv[at].as_str() {
                "SEC" => 8 + section_digest_len(at + 8),
                "DSEC" => 2 + section_digest_len(at + 2),
                "TOMB" => 4,
                "CHL" => 3 + count(at + 2) + 1,
                "DCHL" | "DF" | "DT" => 3,
                "SFA" => 2 + 2 * count(at + 1) + 4,
                "STA" => 2 + 2 * count(at + 1) + 2,
                "HDR" => 5 + usize::from(argv[at + 4] == "1"),
                other => panic!("unknown tag {other:?} at slot {at} of {argv:?}"),
            };
            records.push(argv[at..at + len].to_vec());
            at += len;
        }
        records
    }

    fn tag_of(mutation: &StorageMutation) -> &'static str {
        match mutation {
            StorageMutation::SetSection { .. } => "SEC",
            StorageMutation::DeleteSection { .. } => "DSEC",
            StorageMutation::CreateTombstone { .. } => "TOMB",
            StorageMutation::DeleteChildrenList { .. } => "DCHL",
            StorageMutation::SetChildren { .. } => "CHL",
            StorageMutation::SetField { .. } => "SFA",
            StorageMutation::DeleteField { .. } => "DF",
            StorageMutation::SetTimeSample { .. } => "STA",
            StorageMutation::DeleteTimeSample { .. } => "DT",
            StorageMutation::SetObjectHeader { .. } => "HDR",
        }
    }

    // ── One test per tag ─────────────────────────────────────────────────────

    #[test]
    fn encode_mutations_set_section_new_row_emits_sec_columns_then_absent() {
        let mutations = vec![StorageMutation::SetSection {
            section: section(SID, ID_ABOVE_F64_EXACT, "Lamp", 3, 17, 5, 9),
            previous: None,
        }];

        let enc = encode(&mutations, &ObjectState::default(), 0);

        assert_eq!(
            (enc.mutation_keys, enc.mutation_argv),
            (
                strings(&["le:{42}:obj:sec:4660"]),
                strings(&[
                    "SEC",
                    "4660",
                    "9007199254740993",
                    "Lamp",
                    "3",
                    "17",
                    "5",
                    "9",
                    "absent"
                ])
            )
        );
    }

    #[test]
    fn encode_mutations_set_section_present_row_emits_sec_columns_then_pre_state_row() {
        // The digest's six slots are the pre-state row's, not the new row's.
        let old = section(SID, ID_ABOVE_F64_EXACT, "Old", 1, 2, 3, 4);
        let mutations = vec![StorageMutation::SetSection {
            section: section(SID, 0, "Lamp", 3, 17, 3, 9),
            previous: Some(SectionKey {
                parent_id: ID_ABOVE_F64_EXACT,
                name: "Old".to_owned(),
            }),
        }];

        let enc = encode(&mutations, &pre_with(&[old]), 0);

        assert_eq!(
            enc.mutation_argv,
            strings(&[
                "SEC",
                "4660",
                "0",
                "Lamp",
                "3",
                "17",
                "3",
                "9",
                "row",
                "9007199254740993",
                "Old",
                "1",
                "2",
                "3",
                "4"
            ])
        );
    }

    #[test]
    fn encode_mutations_delete_section_emits_dsec_sid_then_pre_state_row() {
        let old = section(SID, ID_ABOVE_F64_EXACT, "Old", 1, 2, 3, 4);
        let mutations = vec![StorageMutation::DeleteSection {
            section_id: SID,
            parent_id: ID_ABOVE_F64_EXACT,
            name: "Old".to_owned(),
        }];

        let enc = encode(&mutations, &pre_with(&[old]), 0);

        assert_eq!(
            (enc.mutation_keys, enc.mutation_argv),
            (
                strings(&["le:{42}:obj:sec:4660"]),
                strings(&[
                    "DSEC",
                    "4660",
                    "row",
                    "9007199254740993",
                    "Old",
                    "1",
                    "2",
                    "3",
                    "4"
                ])
            )
        );
    }

    #[test]
    fn encode_mutations_delete_section_of_an_absent_row_emits_dsec_sid_then_absent() {
        let mutations = vec![StorageMutation::DeleteSection {
            section_id: SID,
            parent_id: 0,
            name: "Gone".to_owned(),
        }];

        let enc = encode(&mutations, &ObjectState::default(), 0);

        assert_eq!(enc.mutation_argv, strings(&["DSEC", "4660", "absent"]));
    }

    #[test]
    fn encode_mutations_create_tombstone_emits_tomb_versions_and_no_digest() {
        let mutations = vec![StorageMutation::CreateTombstone {
            section_id: SID,
            first_version: 5,
            last_version: ID_ABOVE_F64_EXACT,
        }];

        let enc = encode(&mutations, &ObjectState::default(), 0);

        assert_eq!(
            (enc.mutation_keys, enc.mutation_argv),
            (
                strings(&["le:{42}:obj:del:4660"]),
                strings(&["TOMB", "4660", "5", "9007199254740993"])
            )
        );
    }

    #[test]
    fn encode_mutations_set_children_emits_chl_count_ids_then_pre_state_list_digest() {
        // The pre-state list is the fixed vector, so the digest slot is
        // `LIST_VECTOR_SHA1`; the written list is a different one.
        let mut pre = pre_with(&[section(SID, 0, "P", 0, 0, 1, 1)]);
        pre.children.insert(SID, LIST_VECTOR.to_vec());
        let mutations = vec![StorageMutation::SetChildren {
            section_id: SID,
            children: vec![1000, ID_ABOVE_F64_EXACT, 5],
        }];

        let enc = encode(&mutations, &pre, 0);

        assert_eq!(
            (enc.mutation_keys, enc.mutation_argv),
            (
                strings(&["le:{42}:obj:ch:4660"]),
                strings(&[
                    "CHL",
                    "4660",
                    "3",
                    "1000",
                    "9007199254740993",
                    "5",
                    LIST_VECTOR_SHA1
                ])
            )
        );
    }

    #[test]
    fn encode_mutations_set_children_empty_list_emits_chl_zero_count_then_digest() {
        let pre = pre_with(&[section(SID, 0, "P", 0, 0, 1, 1)]);
        let mutations = vec![StorageMutation::SetChildren {
            section_id: SID,
            children: Vec::new(),
        }];

        let enc = encode(&mutations, &pre, 0);

        assert_eq!(enc.mutation_argv, strings(&["CHL", "4660", "0", "absent"]));
    }

    #[test]
    fn encode_mutations_delete_children_list_emits_dchl_parent_then_digest() {
        // The root's list, by its decimal id `0`, against the fixed vector.
        let mut pre = ObjectState::default();
        pre.children.insert(0, LIST_VECTOR.to_vec());
        let mutations = vec![StorageMutation::DeleteChildrenList { section_id: 0 }];

        let enc = encode(&mutations, &pre, 0);

        assert_eq!(
            (enc.mutation_keys, enc.mutation_argv),
            (
                strings(&["le:{42}:obj:ch:0"]),
                strings(&["DCHL", "0", LIST_VECTOR_SHA1])
            )
        );
    }

    #[test]
    fn encode_mutations_children_list_of_a_dangling_parent_emits_none() {
        // `SID` has no row in the pre-state, so its list is never checked.
        let mutations = vec![StorageMutation::DeleteChildrenList { section_id: SID }];

        let enc = encode(&mutations, &ObjectState::default(), 0);

        assert_eq!(enc.mutation_argv, strings(&["DCHL", "4660", "none"]));
    }

    #[test]
    fn encode_mutations_set_field_inline_emits_sfa_value_pair_absolute_orders_and_member() {
        let mutations = vec![StorageMutation::SetField {
            field: Field {
                section_id: SID,
                key_name: KEY.to_owned(),
                value: FieldValue::Inline(b"abc".to_vec()),
                set_order: 70,
                authoring_set_order: 3,
                version: ID_ABOVE_F64_EXACT,
            },
        }];

        let enc = encode(&mutations, &ObjectState::default(), 0);

        assert_eq!(
            (enc.mutation_keys, enc.mutation_argv),
            (
                vec![format!("le:{{42}}:obj:fld:4660:{KEY_HEX}")],
                vec![
                    "SFA".to_owned(),
                    "1".to_owned(),
                    "value".to_owned(),
                    // base64("abc")
                    "YWJj".to_owned(),
                    "70".to_owned(),
                    "3".to_owned(),
                    "9007199254740993".to_owned(),
                    format!("4660:{KEY_HEX}"),
                ]
            )
        );
    }

    #[test]
    fn encode_mutations_set_field_external_emits_sfa_hash_and_size_pairs() {
        let mutations = vec![StorageMutation::SetField {
            field: Field {
                section_id: SID,
                key_name: KEY.to_owned(),
                value: FieldValue::External {
                    hash: [0xab; 32],
                    size: 4096,
                    ext_value_index: 2,
                },
                set_order: 70,
                authoring_set_order: 3,
                version: 12,
            },
        }];

        let enc = encode(&mutations, &ObjectState::default(), 0);

        assert_eq!(
            enc.mutation_argv,
            vec![
                "SFA".to_owned(),
                "2".to_owned(),
                "hash".to_owned(),
                "ab".repeat(32),
                "size".to_owned(),
                "4096".to_owned(),
                "70".to_owned(),
                "3".to_owned(),
                "12".to_owned(),
                format!("4660:{KEY_HEX}"),
            ]
        );
    }

    #[test]
    fn encode_mutations_set_time_sample_emits_sta_value_pair_version_and_member() {
        let mutations = vec![StorageMutation::SetTimeSample {
            sample: TimeSample {
                section_id: SID,
                time: OrderedFloat(1.5),
                value: FieldValue::Inline(b"tick".to_vec()),
                version: 12,
            },
        }];

        let enc = encode(&mutations, &ObjectState::default(), 0);

        // 1.5's bits are 0x3ff8000000000000; base64("tick") is "dGljaw==".
        assert_eq!(
            (enc.mutation_keys, enc.mutation_argv),
            (
                strings(&["le:{42}:obj:ts:4660:3ff8000000000000"]),
                strings(&[
                    "STA",
                    "1",
                    "value",
                    "dGljaw==",
                    "12",
                    "4660:3ff8000000000000"
                ])
            )
        );
    }

    #[test]
    fn encode_mutations_delete_field_emits_df_member_then_existence() {
        let mut pre = ObjectState::default();
        pre.fields.insert(
            (SID, KEY.to_owned()),
            Field {
                section_id: SID,
                key_name: KEY.to_owned(),
                value: FieldValue::Inline(vec![1]),
                set_order: 1,
                authoring_set_order: 1,
                version: 1,
            },
        );
        let mutations = vec![
            StorageMutation::DeleteField {
                section_id: SID,
                key_name: KEY.to_owned(),
            },
            StorageMutation::DeleteField {
                section_id: SID,
                key_name: "x".to_owned(),
            },
        ];

        let enc = encode(&mutations, &pre, 0);

        assert_eq!(
            (enc.mutation_keys, enc.mutation_argv),
            (
                vec![
                    format!("le:{{42}}:obj:fld:4660:{KEY_HEX}"),
                    // hex("x")
                    "le:{42}:obj:fld:4660:78".to_owned(),
                ],
                vec![
                    "DF".to_owned(),
                    format!("4660:{KEY_HEX}"),
                    "exists".to_owned(),
                    "DF".to_owned(),
                    "4660:78".to_owned(),
                    "absent".to_owned(),
                ]
            )
        );
    }

    #[test]
    fn encode_mutations_delete_time_sample_zero_and_negative_zero_emit_distinct_dt_records() {
        // The pre-state holds a sample at `0.0` only. `-0.0` is its own key
        // (sign bit set) and its own row, which does not exist.
        let mut pre = ObjectState::default();
        pre.time_samples.insert(
            (SID, OrderedFloat(0.0)),
            TimeSample {
                section_id: SID,
                time: OrderedFloat(0.0),
                value: FieldValue::Inline(vec![1]),
                version: 1,
            },
        );
        let mutations = vec![
            StorageMutation::DeleteTimeSample {
                section_id: SID,
                time: OrderedFloat(-0.0),
            },
            StorageMutation::DeleteTimeSample {
                section_id: SID,
                time: OrderedFloat(0.0),
            },
        ];

        let enc = encode(&mutations, &pre, 0);

        assert_eq!(
            (enc.mutation_keys, enc.mutation_argv),
            (
                strings(&[
                    "le:{42}:obj:ts:4660:8000000000000000",
                    "le:{42}:obj:ts:4660:0000000000000000"
                ]),
                strings(&[
                    "DT",
                    "4660:8000000000000000",
                    "absent",
                    "DT",
                    "4660:0000000000000000",
                    "exists"
                ])
            )
        );
    }

    fn header(source_format: Option<&str>) -> StorageMutation {
        StorageMutation::SetObjectHeader {
            object_id: 42,
            sequence_num: 8,
            first_version: 3,
            source_format: source_format.map(str::to_owned),
        }
    }

    #[test]
    fn encode_mutations_set_object_header_with_source_format_emits_hdr_flag_one_and_format() {
        let enc = encode(&[header(Some("usda"))], &ObjectState::default(), 77);

        assert_eq!(
            (enc.mutation_keys, enc.mutation_argv),
            (
                strings(&["le:{42}:obj:hdr"]),
                strings(&["HDR", "8", "3", "77", "1", "usda"])
            )
        );
    }

    #[test]
    fn encode_mutations_set_object_header_without_source_format_emits_hdr_flag_zero() {
        let enc = encode(&[header(None)], &ObjectState::default(), 77);

        assert_eq!(enc.mutation_argv, strings(&["HDR", "8", "3", "77", "0"]));
    }

    #[test]
    fn encode_mutations_set_object_header_empty_source_format_emits_hdr_flag_one_and_empty() {
        // An empty format is present, not absent: the flag tells them apart.
        let enc = encode(&[header(Some(""))], &ObjectState::default(), 77);

        assert_eq!(
            enc.mutation_argv,
            strings(&["HDR", "8", "3", "77", "1", ""])
        );
    }

    // ── Digest rendering ─────────────────────────────────────────────────────

    #[test]
    fn encode_mutations_second_touch_of_a_section_emits_none() {
        let mutations = vec![
            StorageMutation::SetSection {
                section: section(SID, 0, "a", 0, 0, 1, 1),
                previous: None,
            },
            StorageMutation::SetSection {
                section: section(SID, 0, "b", 0, 0, 1, 1),
                previous: Some(SectionKey {
                    parent_id: 0,
                    name: "a".to_owned(),
                }),
            },
        ];

        let enc = encode(&mutations, &ObjectState::default(), 0);

        let digests: Vec<String> = split_records(&enc.mutation_argv)
            .into_iter()
            .map(|record| record[8..].join(" "))
            .collect();
        assert_eq!(digests, strings(&["absent", "none"]));
    }

    // ── Plan-level shape ─────────────────────────────────────────────────────

    #[test]
    fn encode_mutations_counts_every_record_and_sizes_all_but_hdr() {
        let mutations = vec![
            StorageMutation::CreateTombstone {
                section_id: SID,
                first_version: 1,
                last_version: 2,
            },
            StorageMutation::DeleteChildrenList { section_id: SID },
            header(None),
        ];

        let enc = encode(&mutations, &ObjectState::default(), 0);

        assert_eq!((enc.num_mutations, enc.num_size_records), (3, 2));
    }

    #[test]
    fn encode_mutations_header_only_plan_has_zero_size_records() {
        let enc = encode(&[header(None)], &ObjectState::default(), 0);

        assert_eq!((enc.num_mutations, enc.num_size_records), (1, 0));
    }

    // ── The untouched-row count block ────────────────────────────────────────

    /// An external field row `(SID, key_name)` holding `[tag; 32]`.
    fn external_field(key_name: &str, tag: u8) -> Field {
        Field {
            section_id: SID,
            key_name: key_name.to_owned(),
            value: FieldValue::External {
                hash: [tag; 32],
                size: 64,
                ext_value_index: 0,
            },
            set_order: 1,
            authoring_set_order: 1,
            version: 1,
        }
    }

    #[test]
    fn encode_mutations_untouched_block_emits_k_then_hash_count_pairs_ascending_by_hash() {
        // Untouched rows: two hold `0x22…`, one holds `0xcc…`. The plan writes
        // `0xcc…`, `0x88…`, `0x22…` in that order; the block lists them by
        // hash, `0x88…` with its zero count. The records are not affected.
        let mut pre = ObjectState::default();
        for (key, tag) in [("u1", 0x22), ("u2", 0x22), ("u3", 0xcc)] {
            pre.fields
                .insert((SID, key.to_owned()), external_field(key, tag));
        }
        let mutations: Vec<StorageMutation> = [("w1", 0xcc), ("w2", 0x88), ("w3", 0x22)]
            .into_iter()
            .map(|(key, tag)| StorageMutation::SetField {
                field: external_field(key, tag),
            })
            .collect();

        let enc = encode(&mutations, &pre, 0);

        assert_eq!(
            enc.untouched_ext_block,
            vec![
                "3".to_owned(),
                "22".repeat(32),
                "2".to_owned(),
                "88".repeat(32),
                "0".to_owned(),
                "cc".repeat(32),
                "1".to_owned(),
            ]
        );
        assert_eq!(
            split_records(&enc.mutation_argv)
                .iter()
                .map(|record| record[0].as_str())
                .collect::<Vec<_>>(),
            vec!["SFA", "SFA", "SFA"],
            "the block is not part of the records"
        );
    }

    #[test]
    fn encode_mutations_untouched_block_of_a_plan_naming_no_external_hash_is_k_zero() {
        let enc = encode(
            &[
                StorageMutation::DeleteChildrenList { section_id: SID },
                header(None),
            ],
            &ObjectState::default(),
            0,
        );

        assert_eq!(enc.untouched_ext_block, vec!["0".to_owned()]);
    }

    /// `A` (1) under the root with child `B` (2); `B` holds field `k` and a
    /// sample at `1.5`. The seed the plan-level tests run `apply_delta` on.
    fn seeded_state() -> ObjectState {
        let mut state = pre_with(&[
            section(1, 0, "A", 1, 0, 1, 3),
            section(2, 1, "B", 1, 0, 2, 3),
        ]);
        state.object_id = 42;
        state.sequence_num = 3;
        state.first_version = 1;
        state.next_set_order = 4;
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);
        state.fields.insert(
            (2, "k".to_owned()),
            Field {
                section_id: 2,
                key_name: "k".to_owned(),
                value: FieldValue::Inline(vec![1]),
                set_order: 3,
                authoring_set_order: 0,
                version: 3,
            },
        );
        state.time_samples.insert(
            (2, OrderedFloat(1.5)),
            TimeSample {
                section_id: 2,
                time: OrderedFloat(1.5),
                value: FieldValue::Inline(vec![2]),
                version: 3,
            },
        );
        state
    }

    fn set_field_cmd(section_id: u64, key_name: &str) -> OodfSetField {
        OodfSetField {
            section_id,
            key_name: key_name.to_owned(),
            value_or_ext_hash: b"v".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }
    }

    /// `apply_delta(delta)` on [`seeded_state`]: the plan, the pre-state and
    /// the post-apply `next_set_order`.
    fn planned(delta: OwnedDelta) -> (Vec<StorageMutation>, ObjectState, u64) {
        let pre = seeded_state();
        let mut post = pre.clone();
        let mutations = apply_delta(&mut post, Some(delta)).expect("the fixture delta must apply");
        (mutations, pre, post.next_set_order)
    }

    #[test]
    fn encode_mutations_mixed_plan_emits_one_record_and_one_key_per_mutation_in_order() {
        // Move, reorder, create, field set and time-sample set in one delta.
        use oodf_ffi::{CommandMoveSection, CommandReorderChildren, TimeSample as OodfTimeSample};
        let (mutations, pre, next_set_order) = planned(OwnedDelta {
            is_diff: false,
            struct_commands: vec![
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 1,
                    section_id: 3,
                    section_name: "C".to_owned(),
                    section_type: 1,
                }),
                StructCommand::MoveSection(CommandMoveSection {
                    old_parent_id: 1,
                    new_parent_id: 0,
                    section_id: 2,
                    new_name: "B2".to_owned(),
                }),
                StructCommand::ReorderChildren(CommandReorderChildren {
                    section_id: 0,
                    children_list_id: 0,
                    children_list: vec![2, 1],
                }),
            ],
            set_fields: vec![set_field_cmd(3, "f")],
            time_samples: vec![OodfTimeSample {
                section_id: 2,
                time: 4.0,
                value_or_ext_hash: b"t".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..Default::default()
        });
        for kind in ["SEC", "CHL", "SFA", "STA", "HDR"] {
            assert!(
                mutations.iter().any(|m| tag_of(m) == kind),
                "fixture precondition: the plan must carry a {kind} record; got {mutations:?}"
            );
        }

        let enc = encode(&mutations, &pre, next_set_order);

        let tags: Vec<String> = split_records(&enc.mutation_argv)
            .into_iter()
            .map(|record| record[0].clone())
            .collect();
        let expected_tags: Vec<String> = mutations.iter().map(|m| tag_of(m).to_owned()).collect();
        let expected_keys: Vec<String> = mutations.iter().map(key_of).collect();
        assert_eq!((tags, enc.mutation_keys), (expected_tags, expected_keys));
    }

    /// The one key a record writes, per the vocabulary's `KEYS:` column.
    fn key_of(mutation: &StorageMutation) -> String {
        let keys = KeySchema::default();
        match mutation {
            StorageMutation::SetSection { section, .. } => {
                keys.obj_section(OID, &section.section_id.to_string())
            }
            StorageMutation::DeleteSection { section_id, .. } => {
                keys.obj_section(OID, &section_id.to_string())
            }
            StorageMutation::CreateTombstone { section_id, .. } => {
                keys.obj_deleted_section(OID, &section_id.to_string())
            }
            StorageMutation::DeleteChildrenList { section_id }
            | StorageMutation::SetChildren { section_id, .. } => {
                keys.obj_children(OID, &section_id.to_string())
            }
            StorageMutation::SetField { field } => {
                keys.obj_field(OID, &field.section_id.to_string(), &field.key_name)
            }
            StorageMutation::DeleteField {
                section_id,
                key_name,
            } => keys.obj_field(OID, &section_id.to_string(), key_name),
            StorageMutation::SetTimeSample { sample } => keys.obj_time_sample(
                OID,
                &sample.section_id.to_string(),
                sample.time.into_inner(),
            ),
            StorageMutation::DeleteTimeSample { section_id, time } => {
                keys.obj_time_sample(OID, &section_id.to_string(), time.into_inner())
            }
            StorageMutation::SetObjectHeader { .. } => keys.obj_header(OID),
        }
    }

    #[test]
    fn encode_mutations_reset_plan_encodes_the_whole_wipe_then_the_rebuild_with_hdr_last() {
        // Reset the seed and rebuild `A` (1) under the root with a field.
        let (mutations, pre, next_set_order) = planned(OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: "usda".to_owned(),
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: 1,
                section_name: "A".to_owned(),
                section_type: 1,
            })],
            set_fields: vec![set_field_cmd(1, "f")],
            ..Default::default()
        });

        let enc = encode(&mutations, &pre, next_set_order);
        let records = split_records(&enc.mutation_argv);

        // 1:1, one key each.
        assert_eq!(
            (records.len(), enc.mutation_keys.len()),
            (mutations.len(), mutations.len()),
            "one record and one key per mutation"
        );
        let tagged = |tag: &str| -> Vec<Vec<String>> {
            records
                .iter()
                .filter(|record| record[0] == tag)
                .cloned()
                .collect()
        };
        // The wipe: every pre-state row is deleted, each against its pre-state
        // value, and each section leaves a tombstone.
        assert_eq!(
            {
                let mut dsec = tagged("DSEC");
                dsec.sort();
                dsec
            },
            vec![
                strings(&["DSEC", "1", "row", "0", "A", "1", "0", "1", "3"]),
                strings(&["DSEC", "2", "row", "1", "B", "1", "0", "2", "3"]),
            ],
            "the wipe deletes every pre-state section against its row"
        );
        assert_eq!(
            {
                let mut tomb: Vec<String> =
                    tagged("TOMB").into_iter().map(|r| r[1].clone()).collect();
                tomb.sort();
                tomb
            },
            strings(&["1", "2"]),
            "the wipe tombstones every pre-state section"
        );
        assert_eq!(
            (tagged("DF"), tagged("DT")),
            (
                vec![strings(&["DF", "2:6b", "exists"])],
                vec![strings(&["DT", "2:3ff8000000000000", "exists"])]
            ),
            "the wipe deletes every pre-state field and sample row"
        );
        // Every pre-state list (the root's and `A`'s) is rewritten or deleted.
        for parent in ["0", "1"] {
            assert!(
                records
                    .iter()
                    .any(|r| (r[0] == "DCHL" || r[0] == "CHL") && r[1] == parent),
                "the plan must rewrite or delete list {parent}; got {records:?}"
            );
        }
        // The rebuild: `A` again, after its delete, carrying no second check
        // of `sec:1` (the delete carried it), and its field.
        let dsec_a = records
            .iter()
            .position(|r| r[0] == "DSEC" && r[1] == "1")
            .expect("the wipe deletes A");
        let sec_a = records
            .iter()
            .position(|r| r[0] == "SEC" && r[1] == "1")
            .expect("the rebuild sets A");
        assert!(dsec_a < sec_a, "the wipe precedes the rebuild: {records:?}");
        assert_eq!(
            records[sec_a],
            strings(&["SEC", "1", "0", "A", "1", "0", "4", "4", "none"])
        );
        assert_eq!(tagged("SFA").len(), 1, "the rebuild sets the field");
        // `HDR` last, once, with the reset's format.
        assert_eq!(
            (
                tagged("HDR").len(),
                records.last().cloned(),
                enc.mutation_keys.last().cloned()
            ),
            (
                1,
                Some(strings(&[
                    "HDR",
                    "4",
                    "4",
                    &next_set_order.to_string(),
                    "1",
                    "usda"
                ])),
                Some("le:{42}:obj:hdr".to_owned())
            )
        );
        assert_eq!(
            (enc.num_mutations, enc.num_size_records),
            (mutations.len() as u64, mutations.len() as u64 - 1)
        );
    }

    // ── The list digest ──────────────────────────────────────────────────────

    #[test]
    fn children_list_digest_fixed_vector_returns_sha1_hex_of_comma_joined_decimal_ids() {
        assert_eq!(children_list_digest(Some(&LIST_VECTOR)), LIST_VECTOR_SHA1);
    }

    #[test]
    fn children_list_digest_u64_max_id_hashes_its_full_decimal_form() {
        // sha1sum of the bytes `18446744073709551615`.
        assert_eq!(
            children_list_digest(Some(&[u64::MAX])),
            "234524f46607504594696f875bd0ca86fe0ee671"
        );
    }

    #[test]
    fn children_list_digest_missing_or_empty_list_returns_the_absent_marker() {
        assert_eq!(
            (children_list_digest(None), children_list_digest(Some(&[]))),
            (DIGEST_ABSENT.to_owned(), DIGEST_ABSENT.to_owned())
        );
    }

    // ── Digests bound to their plan ──────────────────────────────────────────

    /// A one-field plan; `byte` varies the field's value only.
    fn field_plan(byte: u8) -> Vec<StorageMutation> {
        vec![
            StorageMutation::SetField {
                field: Field {
                    section_id: SID,
                    key_name: KEY.to_owned(),
                    value: FieldValue::Inline(vec![byte]),
                    set_order: 1,
                    authoring_set_order: 0,
                    version: 2,
                },
            },
            header(None),
        ]
    }

    fn encode_with(
        mutations: &[StorageMutation],
        digests: &PreImageDigests,
    ) -> Result<EncodedStructuralRecords, ServiceError> {
        encode_mutations(mutations, digests, 0, &KeySchema::default(), OID)
    }

    #[test]
    fn encode_mutations_digests_from_a_different_plan_of_the_same_length_returns_internal() {
        // The two plans differ in one value byte only: same length, same kinds,
        // same keys, and neither carries a digest, so only the plan
        // fingerprint can tell them apart.
        let pre = ObjectState::default();
        let other = PreImageDigests::from_plan(&field_plan(1), &pre);

        let result = encode_with(&field_plan(2), &other);

        assert!(
            matches!(result, Err(ServiceError::Internal(_))),
            "a digests value built from another plan must be refused; got {result:?}"
        );
    }

    #[test]
    fn encode_mutations_digests_from_a_plan_differing_in_key_or_list_return_internal() {
        // Each row is two one-record plans of one kind that differ in one
        // thing: the stored key the record names (and its digest is checked
        // against), or, in the order row, the list value written to the same
        // key. Pairing either plan with the other's digests must be refused:
        // the plan fingerprint covers every field, so two plans that differ
        // anywhere encode differently. A fingerprint blind to a key would send
        // a digest out against the wrong key. The rows not refused are
        // collected, so a failure names each.
        let dt = |time: f64| StorageMutation::DeleteTimeSample {
            section_id: SID,
            time: OrderedFloat(time),
        };
        let df = |key_name: &str| StorageMutation::DeleteField {
            section_id: SID,
            key_name: key_name.to_owned(),
        };
        let dsec = |section_id: u64| StorageMutation::DeleteSection {
            section_id,
            parent_id: 0,
            name: "n".to_owned(),
        };
        let chl = |section_id: u64, children: Vec<u64>| StorageMutation::SetChildren {
            section_id,
            children,
        };
        let dchl = |section_id: u64| StorageMutation::DeleteChildrenList { section_id };
        let rows: Vec<(&str, StorageMutation, StorageMutation)> = vec![
            ("time sample at 0.0 vs -0.0", dt(0.0), dt(-0.0)),
            (
                "SetChildren under two parents",
                chl(1, vec![5]),
                chl(2, vec![5]),
            ),
            ("DeleteChildrenList under two parents", dchl(1), dchl(2)),
            ("DeleteField of two key names", df("a"), df("b")),
            ("DeleteSection of two ids", dsec(1), dsec(2)),
            (
                "SetChildren of one list in two orders",
                chl(1, vec![5, 6]),
                chl(1, vec![6, 5]),
            ),
        ];
        let pre = ObjectState::default();

        let mut not_refused = Vec::new();
        for (name, a, b) in rows {
            let (a, b) = (vec![a], vec![b]);
            let digests_a = PreImageDigests::from_plan(&a, &pre);
            let digests_b = PreImageDigests::from_plan(&b, &pre);
            for (direction, plan, digests) in [
                ("a's digests on b", &b, &digests_a),
                ("b's digests on a", &a, &digests_b),
            ] {
                if !matches!(encode_with(plan, digests), Err(ServiceError::Internal(_))) {
                    not_refused.push(format!("{name}: {direction}"));
                }
            }
        }

        assert_eq!(
            not_refused,
            Vec::<String>::new(),
            "every swapped digests/plan pair must be refused with Internal"
        );
    }

    #[test]
    fn encode_mutations_digests_from_the_empty_plan_returns_internal_for_a_non_empty_plan() {
        let pre = ObjectState::default();
        let empty = PreImageDigests::from_plan(&[], &pre);

        let result = encode_with(&field_plan(1), &empty);

        assert!(
            matches!(result, Err(ServiceError::Internal(_))),
            "the empty plan's digests must be refused for a non-empty plan; got {result:?}"
        );
    }

    #[test]
    fn encode_mutations_digests_from_the_same_plan_encodes() {
        let pre = ObjectState::default();
        let digests = PreImageDigests::from_plan(&field_plan(1), &pre);

        let result = encode_with(&field_plan(1), &digests);

        assert!(
            matches!(&result, Ok(enc) if enc.num_mutations == 2),
            "the matching pair must encode; got {result:?}"
        );
    }
}

#[cfg(test)]
mod stamp_cursor_tests {
    //! b.vn6 — the conditional cursor-stamp primitive
    //! ([`stamp_obj_header_cursor`] over `scripts/stamp_cursor_if_live.lua`).
    //!
    //! ## Why these live in `src/` against a live Valkey
    //!
    //! The primitive's whole content is a *server-side* atomicity claim: the
    //! `HEXISTS seq` test and the `HSET` are one indivisible act, and the
    //! refusal writes NOTHING. Neither half is observable without Valkey — a
    //! fake would only be asserting that the fake refuses — and the helper sits
    //! inside `pub(crate) mod lua`, so a test in `tests/` cannot name it. Same
    //! shape and same reasoning as `counters.rs`'s in-`src`
    //! `seed_object_id_if_below` cases.
    //!
    //! ## The three states, and why each is a separate case
    //!
    //! `HEXISTS key 'seq'` partitions the world into three, not two, and the
    //! third is the one the ticket's field evidence produced:
    //!
    //! - **live** (`seq` present) — the header a real layer has. Stamp.
    //! - **absent** (no key at all) — the header a `DeleteLayer` UNLINKed
    //!   mid-flight. Refusing is not enough on its own: a bare `HSET` on an
    //!   absent key CREATES it, so the assertion that matters is
    //!   `EXISTS == false` AFTER the call, not merely the `false` return.
    //! - **seq-less phantom** (key present, `seq` absent) — precisely what the
    //!   pre-fix bare `HSET` left behind in dev, and the reason the predicate
    //!   is `HEXISTS seq` rather than `EXISTS`. The primitive must refuse over
    //!   one and leave its fields byte-identical, which is what makes the fix
    //!   self-healing rather than phantom-perpetuating.
    //!
    //! **Namespace discipline (shared Valkey).** Every case touches only
    //! `tv.key_schema().obj_header(..)`, which carries the fixture's per-test
    //! prefix; `TestValkey`'s Drop SCAN+UNLINKs it. Nothing here flushes.

    use std::collections::HashMap;

    use fred::interfaces::{HashesInterface, KeysInterface};

    use super::stamp_obj_header_cursor;
    use crate::storage::test_support::TestValkey;

    /// The cursor columns the periodic-snapshot arm stamps — the real
    /// three-pair shape rather than a synthetic one, so the cases cover the
    /// widest ARGV the helper ever builds.
    fn cursor_pairs() -> Vec<(&'static str, String)> {
        vec![
            ("last_checkpoint_seq", "42".to_owned()),
            ("last_checkpoint_ms", "1700000000000".to_owned()),
            ("restore_seq_bump", "0".to_owned()),
        ]
    }

    async fn header_map(tv: &TestValkey, key: &str) -> HashMap<String, String> {
        tv.client()
            .pool
            .hgetall(key)
            .await
            .expect("HGETALL obj:hdr must succeed")
    }

    async fn header_exists(tv: &TestValkey, key: &str) -> bool {
        tv.client()
            .pool
            .exists::<bool, _>(key)
            .await
            .expect("EXISTS obj:hdr must succeed")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stamp_obj_header_cursor_on_a_live_header_writes_every_pair_and_returns_true() {
        let tv = TestValkey::setup().await;
        let key = tv.key_schema().obj_header("9001");

        // A live header is one that carries `seq`. The two identity fields
        // beside it are what a real `create_layer` writes in the same
        // MULTI/EXEC, and they are here so the "untouched" half of the claim
        // has something to be about.
        let _: u64 = tv
            .client()
            .pool
            .hset(
                &key,
                vec![
                    ("seq", "7"),
                    ("layer_name", "stamp-live"),
                    ("created_by", "stamp-test-user"),
                ],
            )
            .await
            .expect("seeding a live header must succeed");

        let stamped = stamp_obj_header_cursor(tv.client().pool.next(), &key, &cursor_pairs())
            .await
            .expect("the conditional stamp must not error against a healthy Valkey");

        assert!(
            stamped,
            "a header carrying `seq` is LIVE, so the stamp must be applied and reported as \
             applied — a `false` here would make every producer believe its layer had died"
        );

        let after = header_map(&tv, &key).await;
        assert_eq!(
            after.get("last_checkpoint_seq").map(String::as_str),
            Some("42")
        );
        assert_eq!(
            after.get("last_checkpoint_ms").map(String::as_str),
            Some("1700000000000")
        );
        assert_eq!(after.get("restore_seq_bump").map(String::as_str), Some("0"));
        assert_eq!(
            after.get("seq").map(String::as_str),
            Some("7"),
            "the stamp writes the pairs it was handed and nothing else — the commit sequence is \
             the writer's field, and clobbering it would rewrite the layer's version"
        );
        assert_eq!(
            after.get("layer_name").map(String::as_str),
            Some("stamp-live"),
            "identity fields outside the handed pairs must survive untouched"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stamp_obj_header_cursor_on_an_absent_key_writes_nothing_and_creates_no_phantom() {
        let tv = TestValkey::setup().await;
        let key = tv.key_schema().obj_header("9002");

        assert!(
            !header_exists(&tv, &key).await,
            "precondition: the header must actually be absent — this case is about the key \
             `DeleteLayer` UNLINKed, and a stray one would make the assertion below vacuous"
        );

        let stamped = stamp_obj_header_cursor(tv.client().pool.next(), &key, &cursor_pairs())
            .await
            .expect("the conditional stamp must not error against a healthy Valkey");

        assert!(
            !stamped,
            "an absent header is a dead layer and the caller must be told so — `false` is what \
             makes the bootstrap reap its blob and the loop's arms decline to record a cursor"
        );
        assert!(
            !header_exists(&tv, &key).await,
            "THE assertion of this module. A bare `HSET` on a missing key CREATES it, and what \
             it creates is a cursor-only phantom with no `seq` — the exact artifact b.vn6 was \
             filed for, which then makes the checkpoint loop WARN on every tick and `diff_read` \
             default fields until the Valkey orphan sweep reaps it minutes later. The refusal has \
             to leave the keyspace byte-identical, not merely return `false`"
        );
    }

    /// The COUNTERFACTUAL: a bare `HSET` on an absent header key CREATES the
    /// cursor-only phantom this whole primitive exists to prevent.
    ///
    /// # Why a test of Valkey's own semantics earns its place here
    ///
    /// Ordinarily "does the backing store behave as documented?" is not this
    /// suite's business. This one is, because every b.vn6 regression assertion
    /// in the repo is a claim of ABSENCE — `EXISTS obj:hdr == false` after the
    /// refusal — and an absence assertion is only as sharp as the hazard it
    /// rules out. Without this case a reader has to take on faith that the
    /// pre-fix code would have failed those assertions; with it, the pre-fix
    /// behaviour is executable right beside the fixed one, and the shape it
    /// produces (a hash carrying cursor columns and no `seq`) is pinned rather
    /// than described. It is also what makes the `HEXISTS seq` predicate
    /// legible: the phantom this creates is exactly the input the third case
    /// below refuses.
    ///
    /// This asserts nothing about the production code and will not change when
    /// the production code does — it is a fixture for the argument, and if it
    /// ever goes red the conclusion is that the hazard changed shape, which is
    /// news worth having.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_bare_hset_on_an_absent_header_creates_the_phantom_the_script_exists_to_prevent() {
        let tv = TestValkey::setup().await;
        let key = tv.key_schema().obj_header("9004");

        assert!(!header_exists(&tv, &key).await, "precondition: absent");

        // The pre-b.vn6 stamp, verbatim in shape: an unconditional HSET of the
        // cursor columns onto a key `DeleteLayer` had already UNLINKed.
        let _: u64 = tv
            .client()
            .pool
            .hset(&key, cursor_pairs())
            .await
            .expect("a bare HSET must succeed");

        let created = header_map(&tv, &key).await;
        assert!(
            !created.is_empty(),
            "the key was RESURRECTED — this is the bug: `HSET` on a missing key creates it, so \
             the pre-fix stamp wrote a header for a layer that had just been deleted"
        );
        assert!(
            !created.contains_key("seq"),
            "and what it created is a PHANTOM: cursor columns with no `seq`, none of the twelve \
             fields `create_layer` writes. The checkpoint loop WARNs \"missing/unparseable \
             obj:hdr seq\" on every tick over one of these, `diff_read` defaults fields around \
             it, and the Valkey orphan sweep is what eventually reaps it — minutes later. Got \
             {created:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stamp_obj_header_cursor_on_a_seq_less_phantom_refuses_and_leaves_it_unchanged() {
        let tv = TestValkey::setup().await;
        let key = tv.key_schema().obj_header("9003");

        // Exactly what a pre-fix bare `HSET` left behind: cursor columns and
        // nothing else. This is why the predicate is `HEXISTS seq` and not
        // `EXISTS` — under `EXISTS` this key reads as live, and every later
        // stamp would keep re-writing the garbage instead of clearing it.
        let _: u64 = tv
            .client()
            .pool
            .hset(
                &key,
                vec![
                    ("last_checkpoint_seq", "5"),
                    ("last_checkpoint_ms", "1600000000000"),
                ],
            )
            .await
            .expect("seeding a phantom header must succeed");

        let stamped = stamp_obj_header_cursor(tv.client().pool.next(), &key, &cursor_pairs())
            .await
            .expect("the conditional stamp must not error against a healthy Valkey");

        assert!(
            !stamped,
            "a header with no `seq` is not a live layer under any reading — the commit paths fail \
             closed on one and the assembler drops it as NotFound — so the stamp must refuse. \
             `EXISTS` as the predicate would answer `true` here and perpetuate the phantom"
        );
        assert_eq!(
            header_map(&tv, &key).await,
            HashMap::from([
                ("last_checkpoint_seq".to_owned(), "5".to_owned()),
                ("last_checkpoint_ms".to_owned(), "1600000000000".to_owned()),
            ]),
            "and it must leave the phantom's fields byte-identical: refusing but still advancing \
             `last_checkpoint_seq` would claim durable coverage for a layer that does not exist"
        );
    }
}
