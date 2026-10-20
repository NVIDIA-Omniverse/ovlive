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

//! Startup-time durability-guard validator (SDD SR-5.5, Epic 6 Task 6).
//!
//! Refuses to start the service when no durability layer is configured.
//! Two unsafe configurations are detected:
//!
//! 1. **Case 1 — both layers off.** `LIVE_EDIT_CHECKPOINT_ENABLED=false` and
//!    Valkey reports `appendonly=no`. The validator probes
//!    `CONFIG GET appendonly` against the maintenance pool to determine the
//!    AOF state. If the probe cannot answer confidently (timeout, the command
//!    or the `appendonly` parameter unavailable on a managed platform, any
//!    other fred error) it falls back to a startup `WARN` and returns
//!    `Ok(())` — AWS MemoryDB and ElastiCache block the `CONFIG` command
//!    family, and Azure Managed Redis exposes `CONFIG GET` but not the
//!    `appendonly` parameter, so refusing every managed deployment is worse
//!    than the softer signal.
//! 2. **Case 2 — checkpoint loop enabled with the no-op adapter.**
//!    `LIVE_EDIT_CHECKPOINT_ENABLED=true` combined with
//!    `LIVE_EDIT_CHECKPOINT_ADAPTER=Disabled`. The loop spawns but writes go
//!    to a no-op store, masquerading as durability. Fully detectable from
//!    `Config` alone — no Valkey probe needed.
//!
//! The two refusals share one error variant and one error string. The text
//! already names both remedies ("enable Valkey AOF, enable
//! LIVE_EDIT_CHECKPOINT_ENABLED with a configured adapter, or both"), so a
//! single grep against operator logs catches both cases regardless of which
//! one fired.
//!
//! Case 2 is checked before Case 1 — case 2 is pure config inspection while
//! case 1 dispatches a Valkey command, and an operator who tripped both
//! predicates simultaneously should not pay the I/O cost.

use std::collections::HashMap;
use std::time::Duration;

use fred::interfaces::ConfigInterface;

use crate::checkpoint::CheckpointAdapter;
use crate::config::Config;
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_MAINTENANCE};

/// Three-bucket classification of the `CONFIG GET appendonly` probe.
///
/// `Yes` and `No` are the two confident answers Valkey can return. `Unknown`
/// covers every other outcome — timeout, fred protocol error, the `CONFIG`
/// command or the `appendonly` parameter being unavailable on a managed
/// platform (AWS MemoryDB and ElastiCache block `CONFIG`; Azure Managed Redis
/// does not expose the `appendonly` parameter), missing key in the returned
/// map, etc. The validator treats `Unknown` as "operator knows what they are
/// doing" and emits a `WARN` rather than refusing startup; refusing every
/// managed deployment is a worse failure mode than the softer signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AofProbeOutcome {
    /// Valkey reported `appendonly=yes`. AOF is on; durability covered.
    Yes,
    /// Valkey reported `appendonly=no`. AOF is off; combined with
    /// `checkpoint_enabled=false` this triggers the SR-5.5 refusal.
    No,
    /// Could not determine AOF state — `CONFIG` (or the `appendonly`
    /// parameter) unavailable on a managed platform, probe timed out, or
    /// fred returned any error. Best-effort; warn and proceed.
    Unknown,
}

/// Errors returned by [`validate_durability_mode`].
///
/// One variant covers both refusal cases (case 1 = AOF off + checkpoints
/// off; case 2 = checkpoints enabled with the Disabled adapter). The
/// operator-facing message names both remedies, so operators grep one
/// string regardless of which case fired.
#[derive(Debug, thiserror::Error)]
pub enum DurabilityGuardError {
    /// SR-5.5 refusal — no durability layer configured.
    ///
    /// The `Display` text is the verbatim SR-5.5 contract string, pinned
    /// byte-for-byte by the test below; `docs/service/docs/durability.md`
    /// § The startup durability check and the Helm `## @param checkpoint.enabled`
    /// comment quote it (SDD §5.5 step 3 points here).
    #[error("live-edit refuses to start with no durability configured: enable Valkey AOF, enable LIVE_EDIT_CHECKPOINT_ENABLED with a configured adapter, or both")]
    NoDurabilityConfigured,
}

/// Startup-time durability-guard entry point.
///
/// Runs once after `Config::from_env` and `ValkeyClient::connect`, before
/// the gRPC server binds. Returns `Ok(())` for every safe configuration
/// (case 1 AOF on, case 1 unknown probe outcome, case 1 checkpoints on
/// with a real adapter) and `Err(NoDurabilityConfigured)` for the two
/// detectable unsafe configurations.
///
/// See SDD §5.5 / §20.4 for the full rationale and the operational
/// guardrails this fits into.
pub async fn validate_durability_mode(
    config: &Config,
    client: &ValkeyClient,
) -> Result<(), DurabilityGuardError> {
    // Case 2 (cheap, no I/O): checkpoint loop enabled but the adapter is the
    // no-op Disabled variant. Catch this before the Valkey probe so an
    // operator who tripped both predicates simultaneously does not pay the
    // I/O cost. Log the specific predicate at ERROR before returning so the
    // operator sees both the specific misconfiguration AND the SR-5.5
    // string that main.rs's error-log shim re-emits.
    if config.checkpoint_enabled && config.checkpoint_adapter == CheckpointAdapter::Disabled {
        tracing::error!(
            adapter = ?config.checkpoint_adapter,
            "checkpoint_enabled=true but adapter=Disabled — no durability layer configured"
        );
        return Err(DurabilityGuardError::NoDurabilityConfigured);
    }

    // Case 1a (cheap): checkpoint loop enabled AND adapter is non-Disabled
    // (verified above). The checkpoint path is the durability layer; no
    // need to probe Valkey AOF.
    if config.checkpoint_enabled {
        return Ok(());
    }

    // Case 1b (probe required): checkpoint loop disabled. Probe Valkey to
    // find out whether AOF is the active durability layer.
    match probe_aof_state(client, config.valkey_command_timeout_ms).await {
        AofProbeOutcome::Yes => Ok(()),
        AofProbeOutcome::No => {
            tracing::error!(
                "checkpoint_enabled=false and Valkey AOF=no — no durability layer configured"
            );
            Err(DurabilityGuardError::NoDurabilityConfigured)
        }
        AofProbeOutcome::Unknown => {
            // Managed Valkey commonly can't answer this probe (AWS MemoryDB
            // and ElastiCache block `CONFIG`; Azure Managed Redis does not
            // expose the `appendonly` parameter) — refusing every managed
            // deployment is a worse failure mode than this softer signal. The
            // message substring `"could not verify Valkey AOF state"` is
            // pinned by tests and by SDD §20.4's operational-guardrail
            // documentation.
            tracing::warn!(
                target: "live_edit::durability",
                "could not verify Valkey AOF state — assuming operator knows what they are doing (managed Valkey cannot answer CONFIG GET appendonly)"
            );
            Ok(())
        }
    }
}

/// Issue `CONFIG GET appendonly` against the maintenance pool and classify
/// the result into the three-bucket [`AofProbeOutcome`] taxonomy.
///
/// Uses the maintenance pool so the probe cannot queue behind data-pool
/// writers (mirrors `external_values` GC and `checkpointer` SCAN). The
/// `acquire_maintenance` token bumps `valkey_pool_in_use{pool="maintenance"}`
/// for the duration of the probe so operators see the saturation signal.
///
/// Timeout binding: `Duration::from_millis(timeout_ms)` derived from
/// `config.valkey_command_timeout_ms` — startup must not block indefinitely
/// behind a stuck Valkey. Timeout → `Unknown` (best-effort).
///
/// Any fred `ErrorKind` (Unknown / InvalidCommand / Protocol / NotFound /
/// Timeout / IO / Canceled / Cluster / Backpressure / Auth / Config / etc.)
/// folds to `Unknown` — the validator's job is "confidently-no → refuse",
/// not "ambiguous → refuse".
async fn probe_aof_state(client: &ValkeyClient, timeout_ms: u64) -> AofProbeOutcome {
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let timeout = Duration::from_millis(timeout_ms);

    let probe = maint_pool.config_get::<HashMap<String, String>, _>("appendonly");
    match tokio::time::timeout(timeout, probe).await {
        Ok(Ok(map)) => match map.get("appendonly").map(|s| s.as_str()) {
            Some("yes") => AofProbeOutcome::Yes,
            Some("no") => AofProbeOutcome::No,
            _ => AofProbeOutcome::Unknown,
        },
        // Any fred error (command blocked, protocol mismatch, IO, cluster,
        // auth, etc.) → Unknown. We do not classify by `ErrorKind` here:
        // the validator only acts on confident answers, and every error
        // path is equally "we cannot say".
        //
        // This is the arm a stalled probe takes, so the count is reachable:
        // fred arms its own command timeout (same duration) on the probe's
        // first poll, microseconds after the outer one, so both land on the
        // same timer tick and `tokio::time::timeout` polls the inner future
        // before its own deadline — fred's `Timeout` wins the tie.
        Ok(Err(e)) => {
            FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE).observe_error(&e);
            AofProbeOutcome::Unknown
        }
        // tokio::time::timeout elapsed before fred returned: a zero (disabled)
        // command timeout makes this deadline zero, or fred's deadline landed
        // on a later tick than this one.
        Err(_) => AofProbeOutcome::Unknown,
    }
}

#[cfg(test)]
mod tests {
    //! Inline tests pin the exact `Display` string of the SR-5.5 error
    //! variant. The case-1a, case-1b, and case-2 branches of
    //! `validate_durability_mode` require a live `ValkeyClient` and are
    //! exercised by the integration-test sibling
    //! (`tests/integration/storage/durability_guard_test.rs`). It never
    //! changes the server's `appendonly` setting: each branch test routes
    //! its client through a `ScriptReplyProxy` that stages the reply to
    //! `CONFIG GET appendonly` (`yes`, `no`, an empty reply as Azure Managed
    //! Redis returns, an error reply as MemoryDB and ElastiCache return, and
    //! a delay past the command timeout), and counts on the wire whether the
    //! probe was sent at all, so the case-1a and case-2 paths are shown to
    //! skip it. One read-only test probes the real server unproxied.

    use super::*;

    /// The `Display` rendering of `NoDurabilityConfigured` IS the SR-5.5
    /// contract string, pinned byte-for-byte here.
    /// `docs/service/docs/durability.md` and the Helm
    /// `## @param checkpoint.enabled` comment quote it (SDD §5.5 step 3 points
    /// here).
    #[test]
    fn display_of_no_durability_configured_matches_sr_5_5_verbatim() {
        let rendered = format!("{}", DurabilityGuardError::NoDurabilityConfigured);
        assert_eq!(
            rendered,
            "live-edit refuses to start with no durability configured: enable Valkey AOF, enable LIVE_EDIT_CHECKPOINT_ENABLED with a configured adapter, or both"
        );
    }
}
