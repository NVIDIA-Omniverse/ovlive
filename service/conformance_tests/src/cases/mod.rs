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

//! The conformance cases, one module per Python test file
//! (`test_<file>.py` → `cases/<file>.rs`), each a 1:1 port of its cases.
//!
//! Each module exports `cases() -> Vec<Case>`, its cases in the order the
//! Python file defines them (parametrizations in pytest's order). [`all`]
//! concatenates the modules in pytest's file order, which is the run order
//! and the registry's pinned order.

pub mod chained;
pub mod layers;
pub mod profiles;
pub mod sessions;
pub mod streams;

use crate::runner::Case;

/// Every case, in run order.
pub fn all() -> Vec<Case> {
    let mut cases = chained::cases();
    cases.extend(layers::cases());
    cases.extend(profiles::cases());
    cases.extend(sessions::cases());
    cases.extend(streams::cases());
    cases
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use super::*;
    use crate::runner::Profile::{self, Core, Deployment, MultiReplica, Permissions};

    /// Every case the retired Python suite collected, in pytest's file and
    /// definition order: (id, profile, whether the Python test took the
    /// per-test `session` fixture), and each case added since, at its place
    /// in its module. A parametrized case's id carries its parameter as the
    /// committed node-id map names it.
    const REGISTRY: [(&str, Profile, bool); 62] = [
        // test_chained.py
        ("chained::test_ping_reports_operation_ids_and_chained_writes", Core, false),
        ("chained::test_chained_writes_sent_in_order_commit_in_chain_order", Core, true),
        ("chained::test_chained_writes_sent_out_of_order_commit_in_chain_order", Core, true),
        ("chained::test_chained_resend_of_the_latest_write_is_answered_already_committed", Core, true),
        ("chained::test_waiting_chained_call_is_refused_aborted_before_its_deadline", Core, true),
        ("chained::test_no_wait_call_whose_predecessor_was_passed_is_refused_at_once", Core, true),
        ("chained::test_cancelled_waiting_chained_call_applies_nothing", Core, true),
        ("chained::test_predecessor_lost_before_its_stream_ended_and_resent_releases_its_successor", Core, true),
        ("chained::test_invalid_chained_call_carries_invalid_update", Core, true),
        ("chained::test_size_refused_chained_call_carries_its_size_reason_in_place_of_invalid_update", Core, true),
        ("chained::test_chained_writes_across_replica_endpoints_commit_in_chain_order", MultiReplica, true),
        // test_layers.py
        ("layers::test_layer_lifecycle_and_generation_precondition", Core, true),
        ("layers::test_malformed_content_is_rejected_without_creating_layer[not-ompf]", Core, true),
        ("layers::test_malformed_content_is_rejected_without_creating_layer[OMPF-count-1-truncated]", Core, true),
        ("layers::test_malformed_content_is_rejected_without_creating_layer[OMPF-ff-x40]", Core, true),
        ("layers::test_delta_over_the_entry_limit_is_refused_delta_too_large_by_both_rpcs", Core, true),
        ("layers::test_delta_one_entry_over_the_limit_is_refused_and_one_at_it_never_delta_too_large", Core, true),
        ("layers::test_children_list_over_the_limit_is_refused_children_list_too_long_by_both_rpcs[100001]", Core, true),
        ("layers::test_children_list_over_the_limit_is_refused_children_list_too_long_by_both_rpcs[1000001]", Core, true),
        ("layers::test_chunk_before_header_is_invalid[CreateLayer]", Core, true),
        ("layers::test_chunk_before_header_is_invalid[UpdateLayer]", Core, true),
        ("layers::test_content_round_trip[initial]", Core, true),
        ("layers::test_content_round_trip[time_sample]", Core, true),
        ("layers::test_content_round_trip[external]", Core, true),
        ("layers::test_content_round_trip[large]", Core, true),
        ("layers::test_header_only_create_holds_exactly_the_layers_root_section", Core, true),
        ("layers::test_update_removing_the_layers_last_top_level_section_is_invalid_argument", Core, true),
        ("layers::test_missing_layer_subscription_returns_not_found", Core, true),
        ("layers::test_requests_without_operation_id_are_answered_unspecified", Core, true),
        ("layers::test_update_resend_with_operation_id_is_answered_already_committed", Core, true),
        ("layers::test_create_resend_with_operation_id_is_answered_already_committed", Core, true),
        ("layers::test_operation_id_reused_for_a_different_request_is_already_exists", Core, true),
        ("layers::test_update_operation_id_without_client_id_is_invalid_argument", Core, true),
        // Stacked parametrizations: `operation` (nearest the function) outer.
        ("layers::test_malformed_operation_id_is_invalid_argument[CreateLayer-a-x65]", Core, true),
        ("layers::test_malformed_operation_id_is_invalid_argument[CreateLayer-op:1]", Core, true),
        ("layers::test_malformed_operation_id_is_invalid_argument[CreateLayer-op 1]", Core, true),
        ("layers::test_malformed_operation_id_is_invalid_argument[UpdateLayer-a-x65]", Core, true),
        ("layers::test_malformed_operation_id_is_invalid_argument[UpdateLayer-op:1]", Core, true),
        ("layers::test_malformed_operation_id_is_invalid_argument[UpdateLayer-op 1]", Core, true),
        // test_profiles.py
        ("profiles::test_live_service_health_is_serving_without_auth", Deployment, false),
        ("profiles::test_denied_base_path_returns_permission_denied", Permissions, false),
        ("profiles::test_update_across_replica_endpoints", MultiReplica, true),
        // test_sessions.py
        ("sessions::test_ping_echoes_timestamp", Core, false),
        ("sessions::test_session_create_list_delete_and_duplicate_name", Core, false),
        ("sessions::test_empty_identifiers_are_invalid[CreateSession]", Core, false),
        ("sessions::test_empty_identifiers_are_invalid[ListSessions]", Core, false),
        ("sessions::test_empty_identifiers_are_invalid[DeleteSession]", Core, false),
        ("sessions::test_empty_identifiers_are_invalid[StatLayer]", Core, false),
        ("sessions::test_empty_identifiers_are_invalid[DeleteLayer]", Core, false),
        ("sessions::test_anonymous_request_is_rejected", Core, false),
        ("sessions::test_sessions_isolate_layer_names", Core, true),
        ("sessions::test_session_delete_cascades_to_layer_and_subscription", Core, true),
        // test_streams.py
        ("streams::test_live_delivery_and_self_echo_suppression", Core, true),
        ("streams::test_empty_update_is_delivered_with_commit_timestamp", Core, true),
        ("streams::test_overwrite_rebootstraps_subscription", Core, true),
        ("streams::test_reconnect_folds_fields_and_time_samples", Core, true),
        ("streams::test_captured_traffic_catchup_preserves_client_state", Core, true),
        ("streams::test_raw_structural_edits_have_independent_expected_state", Core, true),
        // Added after the port: no Python counterpart.
        ("streams::test_sibling_name_rules_have_independent_expected_state", Core, true),
        ("streams::test_presence_join_message_leave", Core, true),
        ("streams::test_presence_keepalive_is_advertised_on_opt_in_and_never_delivered", Core, true),
        ("streams::test_presence_silent_keepalive_join_ends_unavailable_after_leaving", Core, true),
    ];

    #[test]
    fn all_registers_the_python_suites_cases_in_pytest_order_and_the_62_in_all() {
        let registered: Vec<(String, Profile, bool)> = all()
            .iter()
            .map(|case| (case.id(), case.profile, case.needs_session))
            .collect();
        let expected: Vec<(String, Profile, bool)> = REGISTRY
            .iter()
            .map(|(id, profile, session)| (id.to_string(), *profile, *session))
            .collect();
        assert_eq!(registered, expected);
    }

    #[test]
    fn all_case_ids_are_unique() {
        let cases = all();
        let ids: HashSet<String> = cases.iter().map(|case| case.id()).collect();
        assert_eq!(ids.len(), cases.len());
    }

    #[test]
    fn all_splits_58_core_1_deployment_1_permissions_2_multi_replica() {
        let mut counts: BTreeMap<Profile, usize> = BTreeMap::new();
        for case in all() {
            *counts.entry(case.profile).or_default() += 1;
        }
        assert_eq!(
            counts,
            BTreeMap::from([
                (Core, 58),
                (Deployment, 1),
                (Permissions, 1),
                (MultiReplica, 2)
            ])
        );
    }

    #[test]
    fn all_with_the_ci_deployment_profile_selects_59_cases() {
        assert_eq!(crate::runner::selected(all(), &[Deployment]).len(), 59);
    }

    #[test]
    fn each_case_module_names_its_own_file() {
        for (file, cases) in [
            ("chained", chained::cases()),
            ("layers", layers::cases()),
            ("profiles", profiles::cases()),
            ("sessions", sessions::cases()),
            ("streams", streams::cases()),
        ] {
            assert!(cases.iter().all(|case| case.file == file), "{file}");
        }
    }
}
