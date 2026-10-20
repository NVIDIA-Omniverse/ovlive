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

//! b.azh — stream readers connect lazily, and on a cluster to their bound
//! slot's primary alone.
//!
//! A reader holds no connection until it first serves a subscription, so a new
//! replica opens none at startup. On a cluster its connection is a single-node
//! client to the primary that owns its bound hash slot, which it replaces (a
//! retarget) when it binds a slot another primary owns, or when a `-MOVED`
//! shows its slot's owner moved. Its router is supervised from the first
//! connect and across every replacement; its metric hooks follow it.
//!
//! # The two-node cluster facade
//!
//! The cluster tests run here, against the one standalone test store, through
//! a [`ClusterFacade`]: two [`ScriptReplyProxy`]s, A and B, in front of the
//! same Valkey, each answering `CLUSTER SLOTS` with a two-node map — slots
//! below a movable split are A's, the rest B's (or, for a test that needs one
//! node serving two ranges, A's again from a second split up) — and `CLUSTER
//! INFO` with `cluster_state:ok`. A client connected with a `redis-cluster://`
//! URL runs fred's clustered router across both, and a reader's single-node
//! connection goes to one proxy or the other: which one is the observable
//! "node". Every
//! other frame is forwarded, so every command still runs against the real
//! store (in `TEST_VALKEY_URL`'s database: the proxy selects it). Proxy A can
//! also answer the next `XREAD` naming one stream key with a real-shaped
//! `-MOVED <slot> <B>`, stall the next `ECHO` carrying one marker, or answer
//! the next `ECHO` carrying one marker and then stop reading that connection.
//! Both count the `AUTH` and `CLIENT UNBLOCK` frames they relay. The facade has no
//! real slot ownership: a command sent to the "wrong" proxy still runs. So these
//! tests assert where a reader connects, which node's replies a stalled
//! connection is judged by, or how a failed connection is closed, never that a
//! misdirected command fails. The
//! real-cluster counterparts, gated on `LIVE_EDIT_TEST_CLUSTER_URL`, are in
//! `cluster_subscriber_test.rs`.
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | [`readers_hold_no_connection_until_they_first_serve_a_subscription`] | T3 | startup opens no reader connection; the first subscription connects exactly the reader that serves it, supervised and hooked |
//! | [`the_connected_gauge_counts_each_held_connection_once_and_never_goes_negative`] | metrics | a reader connected before the metrics existed is counted when its hooks are armed; each connection once; each retire subtracts what it added |
//! | [`quit_retires_every_connected_reader_and_leaves_no_router_watcher`] | T9 | `quit()` ends every reader's supervision, hooks and connection |
//! | [`a_reader_connected_before_the_hooks_counts_one_reconnect_per_kill`] | T8 | a reader connected before the hooks existed is armed by their install, with one pair: one kill, one reconnect |
//! | [`a_bound_cluster_reader_connects_to_its_slot_owner_alone`] | T4, T15 | one connection, to the routing view's `Server` for the slot, no database (the TLS-name comparison bites only against a TLS cluster; the taken-whole proof is the `client.rs` source guard) |
//! | [`a_cluster_reader_bound_to_a_slot_on_another_node_retargets_its_connection`] | T5, T7, T8 | the client now targets B, A's connection is gone, `cause="retarget"` ticks, delivery holds; supervision and hooks follow the new client, nothing is restarted |
//! | [`a_retargeted_readers_dead_router_is_restarted_and_counted`] | T7 | the replacement client's router, ended, is restarted, counted and logged once |
//! | [`after_repeated_retargets_one_hook_pair_follows_the_reader_and_a_kill_counts_once`] | T8, b.zt3 T16 | after N retargets one hook pair per connected reader, and one reconnect per kill; the registry also holds exactly one pair per data, maintenance and snapshot member and per `txn` slot, each under its client id |
//! | [`a_moved_xread_retargets_the_reader_to_the_slots_new_owner`] | T6 | a real `-MOVED` on the XREAD re-syncs the routing view (one readers' sync) and retargets, without a budget charge or a lost update |
//! | [`after_data_member_0_is_replaced_the_readers_route_through_the_fresh_client`] | b.zt3 T14 | after data member 0 is replaced, a `-MOVED` re-syncs the readers' view through the fresh client and the retarget finds the new owner in its table |
//! | [`a_connection_silent_on_one_node_is_not_closed_by_a_fast_verdict_while_another_node_answers`] | b.zt3 T3 | node B's replies are no evidence for a stalled connection to A: only the slow verdict (a 4 s test bound) closes it |
//! | [`a_command_sent_to_a_named_node_whose_slot_another_node_owns_is_not_judged_by_that_slots_replies`] | b.zt3 T-K2a | an `ECHO` sent to A by name, hashing into B's range: B's replies to that very slot are no evidence for it; only the slow verdict closes A's connection, at no less than the 4 s bound |
//! | [`a_reply_from_a_named_node_to_a_slot_another_node_owns_is_no_evidence_for_that_slot`] | b.zt3 T-K2b | `ECHO`s sent to B by name, hashing into the slot of a command stalled on A, are stamped as B's: only the slow verdict closes A's connection |
//! | [`a_command_routed_by_its_slot_is_judged_by_its_own_range_not_every_range_on_its_node`] | b.zt3 T-K3 (rule pin, not a regression test) | A owns two ranges: replies from its second range are no evidence for a command stalled in its first; only the slow verdict closes the connection |
//! | [`a_stalled_command_whose_range_answers_another_connection_on_its_node_gets_the_fast_verdict`] | b.zt3 T-Kpos | the positive control: a reply from the stalled command's range, in another slot, gives the fast verdict within fast bound + scan interval + 1 s |
//! | [`a_connection_whose_flush_expired_is_not_flushed_again_before_the_router_waits`] | b.zt3 guard on expired connections (`flush`), close of a connection judged dead (rev 4) | a 16 MiB `SET` fed to a connection A stopped reading: its flush expires at the 4 s bound, the router's re-flush before it waits idle fails at once (its DEBUG line, once), and the read loop closes A with the slow verdict, through a close that waits on nothing, within the bound + 2 s; one `(expired flush)` WARN; rider 1: the close returns the in-flight commands (the one-attempt `SET` fails with the verdict, a multi-attempt `GET` is re-sent and answered) |
//! | [`a_command_written_to_a_connection_whose_flush_expired_fails_at_once_without_being_fed`] | b.zt3 guard on expired connections (`feed`), close of a connection judged dead (rev 4) | the same, with the `SET`, a custom `ECHO` for B and a `GET` for A in one pipeline: the `GET`, fed to A in the router step its flush expired in, fails at once with the slow verdict (its DEBUG line, once), which closes A, through a close that waits on nothing, within the bound + 2 s; one WARN; rider 1 as above |
//! | [`a_free_reader_already_on_the_owner_takes_a_new_binding_without_a_retarget`] | T1 | node affinity end to end |
//! | [`a_command_queued_behind_a_cluster_readers_blocked_read_interrupts_it_and_the_read_is_replanned`] | T13 | an outside command queued behind a clustered reader's live blocked read (no production path sends one since b.19i) is interrupted by fred's `Blocking::Interrupt` (`CLIENT UNBLOCK`): it does not wait out the BLOCK, and the reader re-plans on `-UNBLOCKED` |
//! | [`a_client_unblock_on_a_single_node_cluster_reader_is_replanned_without_loss`] | T13 | the `CLIENT UNBLOCK` re-plan on the single-node reader |
//! | [`a_cluster_readers_due_reauth_wakes_its_read_and_auths_between_reads`] | b.19i | a clustered reader's own re-AUTH: a credential wake, one `AUTH` between reads, no `CLIENT UNBLOCK`, no `-UNBLOCKED`, no fred refresh scheduled on the reader's client |
//! | [`a_cluster_clients_auth_broadcast_behind_a_blocked_read_fails_at_the_slow_bound_and_a_drain_inside_it_flags_no_connection`] | b.zt3 T8 (inverts the b.19i pin) | a cluster client's `AUTH` broadcast drains behind a node's blocked read: a drain that ends inside the slow bound leaves no connection declared unresponsive; one that outlasts it fails the `AUTH` with the slow-verdict error at the bound, closes that node's connection, and a command for the other node is answered |
//! | [`an_idle_cluster_readers_keepalive_consumes_one_fault_per_tick`] | T12 | `test-fault-injection` only: one `PING` per tick on the one connection |
//! | [`a_moved_xread_whose_owner_has_not_moved_backs_off_and_reads_the_same_node_again`] | `slot_moved` | a `-MOVED` whose owner, after one view sync, is unchanged: transient, no retarget |
//! | [`a_moved_xread_retargets_without_a_sync_when_the_view_already_names_the_new_owner`] | `slot_moved` | the view already names the new owner: retarget, no `CLUSTER SLOTS` |
//! | `faults::quit_while_a_reader_is_connecting_cancels_the_connect_and_counts_it_neither_ok_nor_failed` | T9 | a reader whose connect never reports back (`Hang`) is cancelled by `quit()`: its task stops inside the shutdown grace, the connect counts neither `ok` nor `failed`, no watcher is left |
//! | `faults::a_subscription_that_leaves_while_its_reader_connects_brings_the_reader_gauges_to_zero` | b.9y2 | a subscription dropped while its reader's connect is held (`Hold`) brings `live_edit_stream_mux_streams` and `live_edit_valkey_pool_in_use{pool="subscriber_pool"}` back to 0: the reader publishes its gauges after its idle drain; no read issued, the held connect completes `bind`/`ok` |
//! | `faults::a_transient_reader_connect_failure_keeps_the_subscription_and_the_retry_serves_it` | connect failure | transient: one WARN naming the failure, budget and backoff, subscription kept, the retry serves it; the gauge counts only the success |
//! | `faults::a_permanent_reader_connect_failure_ends_the_subscriptions_and_a_resubscribe_connects_again` | connect failure | `Auth` (a `WRONGPASS` setup): subscriptions end `UNAVAILABLE` naming the kind; a re-subscribe connects |
//! | `faults::a_reader_connect_stalled_mid_setup_is_cut_at_its_connect_bound_and_retried` | connect bound | a setup that never reports back is cut by `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` (the WARN's `Timeout` names it); the retry serves the subscription |
//! | `faults::a_reader_connect_refused_wrongpass_logs_one_steady_state_hint_and_noauth_none` | connect failure | the credential hint: once for `WRONGPASS`, `pool="subscriber_pool"`; never for `NOAUTH` |
//! | `faults::a_transient_reader_connect_failure_past_the_budget_ends_the_subscriptions` | connect failure | budget exhaustion ends the subscriptions |
//! | `faults::a_failed_retarget_connect_and_its_retry_both_count_as_retargets` | `retarget_due` | a failed retarget counts `retarget`/`failed`, and its retry, with no connection held, `retarget`/`ok` |
//! | `faults::a_forced_reconnect_after_the_slots_owner_moved_retargets_instead_of_reconnecting_the_old_node` | `force_reconnect` | a wedge verdict that finds the owner moved retargets |
//! | `faults::an_idle_reader_whose_node_is_no_longer_a_primary_retires_without_retargeting` | `force_reconnect` | keepalive phase: retire, no retarget; the next connect is a `bind` |
//! | `faults::a_forced_reconnect_whose_slot_owner_has_not_moved_reconnects_the_same_client` | `force_reconnect` | a wedge verdict with the owner unchanged rebuilds the connection in place |
//! | `faults::a_failed_keepalive_on_a_node_still_a_primary_reconnects_instead_of_retiring` | `force_reconnect` | keepalive phase, node still a primary: reconnect in place, no retire |
//! | `faults::a_moved_probe_retargets_the_reader_without_charging_its_budget` | `-MOVED` | on the idle→active probe |
//! | `faults::a_second_moved_retarget_before_any_read_is_charged_to_the_budget` | `-MOVED` streak | only the first retarget since a successful read is free; the second is charged and, past the budget, ends the subscription |
//! | `faults::a_moved_head_check_retargets_the_reader_and_the_stream_is_re_read_there` | `-MOVED` | on a head check |
//! | `faults::a_moved_probe_whose_slot_owner_has_not_moved_is_charged_as_transient` | `-MOVED` | on the probe, owner unchanged: a transient charge, no retarget |
//! | `faults::a_moved_head_check_whose_slot_owner_has_not_moved_is_charged_as_transient` | `-MOVED` | on a head check, owner unchanged: a transient charge, no retarget |
//! | `faults::a_failed_retarget_that_ends_the_subscriptions_makes_the_next_connect_a_bind` | `retarget_due` | a failed retarget that ends the subscriptions leaves nothing to retarget: the next connect is a `bind` |
//! | `faults::a_retarget_whose_quit_overruns_aborts_the_old_router_and_restarts_nothing` | T7, T9 | a stalled retire: aborted, not restarted, not counted, not logged |
//! | `faults::a_retarget_whose_quit_fails_inside_its_bound_still_ends_the_old_router` | retire | a `QUIT` that fails inside `QUIT_TIMEOUT` still ends the old router |
//!
//! The `faults` cases (and the `test-fault-injection` knob in the `-MOVED`
//! retarget test) drive the multiplexer's fault sites and knobs and are
//! compiled out without `test-fault-injection`.
//!
//! # A failed reader connect
//!
//! Every failed, stalled or held reader connect here is staged with the
//! multiplexer's `MuxFaultSite::Connect` seam, so those tests are `faults`
//! cases. The next reader connect attempt, a `bind` or a `retarget`, takes
//! the queued fault, which replaces, or (a `Hold`) delays, only what its new
//! connection's setup reports. From there the production path runs: the
//! cancellation race, the connect bound, the router's abort, the connects
//! counter, the credential hint, and the reader's classification and WARN.
//! No pooled or pub/sub connection takes one. A TCP-proxy fault cannot be
//! aimed like that, because the proxy cannot tell one client's frames from
//! another's: a fault armed for the reader's setup `PING` was sometimes taken
//! by another connection's (a pool reconnect's), and the reader then
//! connected cleanly. Each test pushes
//! its fault after the reader's earlier connects have finished, and asserts
//! the queue is empty once the attempt is over, so the fault was taken.
//!
//! Two things the seam cannot show:
//!
//! - The socket. That a connect cut at its bound, or cancelled, mid-setup
//!   closes its connection is pinned in `src/storage/client.rs` by
//!   `a_reader_connect_cut_at_its_bound_mid_setup_closes_its_connection` and
//!   `a_reader_connect_cancelled_mid_setup_closes_its_connection`, against a
//!   listener that only the reader's connection reaches.
//! - What fred makes of a real refused or dropped setup. The seam's errors
//!   are built the way fred builds them: an `Auth` error whose details are a
//!   `-WRONGPASS` or `-NOAUTH` reply, an `IO` error for a dropped connection.
//!   That fred does parse a server's reply that way, and fails a reader's
//!   first connect with it at once under the readers' production template
//!   rather than retrying it until the connect bound cuts it, is pinned in
//!   `src/storage/client.rs` by
//!   `a_reader_connect_refused_wrongpass_fails_at_once_as_a_server_auth_rejection`,
//!   `a_reader_connect_refused_noauth_fails_at_once_as_a_server_auth_rejection`
//!   and `a_reader_connect_whose_setup_connection_is_dropped_fails_at_once_as_transient`,
//!   against a listener that only the reader's connection reaches.
//!
//! # Wait budget
//!
//! Every test's worst case — every wait at its full bound: [`CONNECT_BUDGET`]
//! (30 s) per client it connects, one [`SETTLE`] (10 s) per admission,
//! delivery, polled condition, ended subscription and closing `quit()`,
//! fred's 5 s command timeout per one-shot command (an `XADD`, a `DEL`, a
//! `CLIENT KILL`, a poll's last `CLIENT LIST`), and the key purge — a
//! `TestValkey`'s drop on a multi-thread runtime, or `purge_keys` on a
//! current-thread one (a `SCAN` page and an `UNLINK`, 5 s each) — is under
//! 180 s. That is the kill deadline `.config/nextest.toml` sets for this whole
//! module (60 s × 3), in place of the global 120 s hang-killer, which kills a
//! test silently. These are deadlines, not sleeps: the healthy path takes a
//! few seconds at most, mostly where a test waits out a bound by construction
//! (`QUIT_TIMEOUT`, the reader connect bound, the fred pin's quiet window
//! and slow bound — about six seconds). A [`serve_once`] is 25 s
//! (subscription 10, `XADD` 5, delivery 10).
//!
//! | Test | Worst case |
//! |---|---|
//! | `readers_hold_no_connection_until_they_first_serve_a_subscription` | 30 + 10 + 5 + 10 + purge 10 = 65 s |
//! | `the_connected_gauge_counts_each_held_connection_once_and_never_goes_negative` | 30 + (10 + 10) × 2 + 10 + quit 10 = 90 s |
//! | `quit_retires_every_connected_reader_and_leaves_no_router_watcher` | 30 + 10 × 2 + 10 + quit 10 + 10 = 80 s |
//! | `a_reader_connected_before_the_hooks_counts_one_reconnect_per_kill` | 30 + 10 + 10 + kill 5 + 10 + 0.5 + quit 10 = 75.5 s |
//! | `a_bound_cluster_reader_connects_to_its_slot_owner_alone` | 30 + 10 + 5 + 10 + purge 10 = 65 s |
//! | `a_cluster_reader_bound_to_a_slot_on_another_node_retargets_its_connection` | 30 + 25 + 10 + 5 + 10 + close poll 15 + purge 10 = 105 s |
//! | `a_retargeted_readers_dead_router_is_restarted_and_counted` | 30 + 25 × 2 + 10 + 25 + purge 10 = 125 s |
//! | `after_repeated_retargets_one_hook_pair_follows_the_reader_and_a_kill_counts_once` | 30 + 25 × 3 + kill 5 + 10 + 0.5 + purge 10 = 130.5 s |
//! | `a_moved_xread_retargets_the_reader_to_the_slots_new_owner` | 30 + 10 + (5 + 10) × 3 + 10 + purge 10 = 105 s |
//! | `after_data_member_0_is_replaced_the_readers_route_through_the_fresh_client` | 30 + 10 + 10 + (5 + 10) × 3 + 10 + purge 10 = 115 s |
//! | `a_connection_silent_on_one_node_is_not_closed_by_a_fast_verdict_while_another_node_answers` | 30 + 10 + 5 + purge 10 = 55 s |
//! | `a_command_sent_to_a_named_node_whose_slot_another_node_owns_is_not_judged_by_that_slots_replies` | 30 + 5 + 10 + 5 + purge 10 = 60 s |
//! | `a_reply_from_a_named_node_to_a_slot_another_node_owns_is_no_evidence_for_that_slot` | 30 + 5 + 10 + 5 + purge 10 = 60 s |
//! | `a_command_routed_by_its_slot_is_judged_by_its_own_range_not_every_range_on_its_node` | 30 + 5 + 10 + 5 + purge 10 = 60 s |
//! | `a_stalled_command_whose_range_answers_another_connection_on_its_node_gets_the_fast_verdict` | 30 + 10 + 5 + purge 10 = 55 s |
//! | `a_connection_whose_flush_expired_is_not_flushed_again_before_the_router_waits` | 30 + 5 + close 25 + purge 10 = 70 s |
//! | `a_command_written_to_a_connection_whose_flush_expired_fails_at_once_without_being_fed` | 30 + 5 + close 25 + purge 10 = 70 s |
//! | `a_free_reader_already_on_the_owner_takes_a_new_binding_without_a_retarget` | 30 + (10 + 5 + 10) × 2 + 10 + 25 + purge 10 = 125 s |
//! | `a_command_queued_behind_a_cluster_readers_blocked_read_interrupts_it_and_the_read_is_replanned` | 30 + 25 + blocked poll 15 + `PING` 10 + 10 + purge 10 = 100 s |
//! | `a_cluster_readers_due_reauth_wakes_its_read_and_auths_between_reads` | 30 + 25 + 15 + 10 × 2 + 5 + 10 + purge 10 = 115 s |
//! | `a_cluster_clients_auth_broadcast_behind_a_blocked_read_fails_at_the_slow_bound_and_a_drain_inside_it_flags_no_connection` | 172.25 s (its own docs) |
//! | `a_client_unblock_on_a_single_node_cluster_reader_is_replanned_without_loss` | 30 + 25 + unblock poll 15 + 5 + 10 + purge 10 = 95 s |
//! | `an_idle_cluster_readers_keepalive_consumes_one_fault_per_tick` | 30 + 25 + 10 + 10 + purge 10 = 85 s |
//! | `a_moved_xread_whose_owner_has_not_moved_backs_off_and_reads_the_same_node_again` | 30 + 25 + (5 + 10) × 2 + 10 + purge 10 = 105 s |
//! | `a_moved_xread_retargets_without_a_sync_when_the_view_already_names_the_new_owner` | 30 + 25 + sync 10 + (5 + 10) × 2 + 10 + purge 10 = 115 s |
//! | `faults::quit_while_a_reader_is_connecting_cancels_the_connect_and_counts_it_neither_ok_nor_failed` | 30 + 10 + 10 + quit 10 + 10 = 70 s |
//! | `faults::a_subscription_that_leaves_while_its_reader_connects_brings_the_reader_gauges_to_zero` | 30 + 10 + 10 + gauges 10 + quit 10 = 70 s |
//! | `faults::a_transient_reader_connect_failure_keeps_the_subscription_and_the_retry_serves_it` | 30 + 10 + 5 + 10 + `DEL` 5 + quit 10 = 70 s |
//! | `faults::a_permanent_reader_connect_failure_ends_the_subscriptions_and_a_resubscribe_connects_again` | 30 + 10 + 10 + 10 + 5 + 10 + `DEL` 5 + quit 10 = 90 s |
//! | `faults::a_reader_connect_stalled_mid_setup_is_cut_at_its_connect_bound_and_retried` | 30 + 10 + WARN 10 + 5 + 10 + `DEL` 5 + quit 10 = 80 s |
//! | `faults::a_reader_connect_refused_wrongpass_logs_one_steady_state_hint_and_noauth_none` | 30 + (10 + 10) × 2 + quit 10 = 80 s |
//! | `faults::a_transient_reader_connect_failure_past_the_budget_ends_the_subscriptions` | 30 + 10 + 10 + `DEL` 5 + quit 10 = 65 s |
//! | `faults::a_failed_retarget_connect_and_its_retry_both_count_as_retargets` | 30 + 25 + 10 + 5 + 10 + purge 10 = 90 s |
//! | `faults::a_forced_reconnect_after_the_slots_owner_moved_retargets_instead_of_reconnecting_the_old_node` | 30 + 25 + sync 10 + (5 + 10) × 2 + purge 10 = 105 s |
//! | `faults::an_idle_reader_whose_node_is_no_longer_a_primary_retires_without_retargeting` | 30 + 25 + 10 + sync 10 + 10 + 25 + purge 10 = 120 s |
//! | `faults::a_moved_probe_retargets_the_reader_without_charging_its_budget` | 30 + 25 + 10 + 10 + 5 + 10 + purge 10 = 100 s |
//! | `faults::a_second_moved_retarget_before_any_read_is_charged_to_the_budget` | 30 + 25 + 10 + 10 + 10 + purge 10 = 95 s |
//! | `faults::a_moved_head_check_retargets_the_reader_and_the_stream_is_re_read_there` | 30 + 5 × 5 + 10 + 10 + 5 + 10 + 0.3 + purge 10 = 100.3 s |
//! | `faults::a_moved_probe_whose_slot_owner_has_not_moved_is_charged_as_transient` | 30 + 25 + 10 + 10 + 10 + purge 10 = 95 s |
//! | `faults::a_moved_head_check_whose_slot_owner_has_not_moved_is_charged_as_transient` | 30 + 5 × 5 + 10 + 10 + 5 + 10 + purge 10 = 100 s |
//! | `faults::a_forced_reconnect_whose_slot_owner_has_not_moved_reconnects_the_same_client` | 30 + 10 + (5 + 10) × 3 + purge 10 = 95 s |
//! | `faults::a_failed_keepalive_on_a_node_still_a_primary_reconnects_instead_of_retiring` | 30 + 25 + 10 + 10 + 10 + purge 10 = 95 s |
//! | `faults::a_failed_retarget_that_ends_the_subscriptions_makes_the_next_connect_a_bind` | 30 + 25 + 10 + 10 + 10 + 5 + 10 + purge 10 = 110 s |
//! | `faults::a_retarget_whose_quit_overruns_aborts_the_old_router_and_restarts_nothing`, `…_fails_inside_its_bound_…` | 30 + 25 + blocked poll 15 + 10 + 25 + close poll 15 + 0.5 + `DEL` 5 × 2 + quit 10 = 140.5 s |

use std::collections::HashSet;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use fred::clients::Client;
use fred::interfaces::{
    ClientInterface, ClientLike, ClusterInterface, EventInterface, KeysInterface,
};
use fred::types::config::{Server, ServerConfig};
use fred::types::ClientUnblockFlag;
use tokio::time::Instant;

use live_edit::config::Config;
use live_edit::observability::metrics::{
    Metrics, READER_CONNECT_CAUSE_BIND, READER_CONNECT_CAUSE_RETARGET,
    READER_CONNECT_RESULT_FAILED, READER_CONNECT_RESULT_OK,
};
use live_edit::storage::client::{ValkeyClient, POOL_LABEL_DATA};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::stream_mux::router::Routing;
use live_edit::storage::test_support::{connect_test_client, TestValkey};
use live_edit::storage::{consume_layer_stream, LayerStreamConsumer, StreamCursor};

use super::mux_test_helpers::{
    assert_no_fred_refresh_on_reader, await_mux, await_reader_blocked_in_xread, install_metrics,
    read_series, read_unblocked_retries, reauths, recv_seqs, wakes, xadd_update, ScriptedProvider,
    SETTLE,
};
use super::snapshot_read_support::{
    error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};
use crate::grpc::log_capture::{assert_field, purge_keys, ThreadCapture};

/// Bound on connecting one client (the fixture's own connect budget).
const CONNECT_BUDGET: Duration = Duration::from_secs(30);

/// `live_edit_stream_mux_reader_connects_total{cause, result}`.
const READER_CONNECTS: &str = "live_edit_stream_mux_reader_connects_total";

/// `live_edit_stream_mux_readers_connected`.
const READERS_CONNECTED: &str = "live_edit_stream_mux_readers_connected";

/// `live_edit_valkey_router_restarts_total{pool}`.
const ROUTER_RESTARTS: &str = "live_edit_valkey_router_restarts_total";

/// `live_edit_valkey_reconnects_total{pool}`.
const RECONNECTS: &str = "live_edit_valkey_reconnects_total";

/// The stream readers' `pool` label.
const READER_POOL: &str = "subscriber_pool";

/// The ERROR line a router restart logs (restated: the published metrics
/// reference tells operators to report it).
const RESTART_MSG: &str = "Valkey client router task ended; restarting it";

fn reader_connects(metrics: &Metrics, cause: &str, result: &str) -> i64 {
    read_series(
        metrics,
        READER_CONNECTS,
        &[("cause", cause), ("result", result)],
    )
}

/// Every `reader_connects_total` cell, as `[bind ok, bind failed, retarget ok,
/// retarget failed]`.
fn reader_connect_row(metrics: &Metrics) -> [i64; 4] {
    [
        reader_connects(metrics, READER_CONNECT_CAUSE_BIND, READER_CONNECT_RESULT_OK),
        reader_connects(
            metrics,
            READER_CONNECT_CAUSE_BIND,
            READER_CONNECT_RESULT_FAILED,
        ),
        reader_connects(
            metrics,
            READER_CONNECT_CAUSE_RETARGET,
            READER_CONNECT_RESULT_OK,
        ),
        reader_connects(
            metrics,
            READER_CONNECT_CAUSE_RETARGET,
            READER_CONNECT_RESULT_FAILED,
        ),
    ]
}

fn readers_connected(metrics: &Metrics) -> i64 {
    read_series(metrics, READERS_CONNECTED, &[])
}

fn reader_restarts(metrics: &Metrics) -> i64 {
    read_series(metrics, ROUTER_RESTARTS, &[("pool", READER_POOL)])
}

fn reader_reconnects(metrics: &Metrics) -> i64 {
    read_series(metrics, RECONNECTS, &[("pool", READER_POOL)])
}

/// Subscribes `client` to `object_id`'s layer stream from `0-0`.
async fn subscribe_on(
    client: &ValkeyClient,
    schema: &KeySchema,
    object_id: u64,
) -> LayerStreamConsumer {
    consume_layer_stream(
        client,
        schema,
        object_id,
        StreamCursor::From("0-0".into()),
        Instant::now() + SETTLE,
    )
    .await
    .expect("the subscription must start")
}

async fn subscribe(tv: &TestValkey, object_id: u64) -> LayerStreamConsumer {
    subscribe_on(tv.client(), tv.key_schema(), object_id).await
}

/// Subscribes to `object_id`, proves a read served it — an update written
/// after the subscription (`seq`) is delivered — and drops the subscription.
/// The reader stays connected; its read may still be in flight.
async fn serve_once(tv: &TestValkey, object_id: u64, seq: u64) {
    let c = subscribe(tv, object_id).await;
    xadd_update(tv, object_id, seq).await;
    assert_eq!(recv_seqs(&c, 1).await, [seq]);
    assert!(c.take_terminal_exit().is_none());
}

/// The fred client id (`ClientLike::id`) of reader `reader`'s connection.
fn reader_client_id(client: &ValkeyClient, reader: usize) -> String {
    ClientLike::id(
        &client
            .stream_mux
            .reader_client_for_test(reader)
            .expect("the reader holds a connection"),
    )
    .to_owned()
}

/// Polls until `cond` holds, by `deadline`.
async fn await_until(what: &str, deadline: Instant, cond: impl Fn() -> bool) {
    while !cond() {
        assert!(Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// The ids among `ids` the server still lists (`CLIENT LIST ID`), over
/// `observer`.
async fn listed(observer: &Client, ids: &[i64]) -> Vec<i64> {
    let wanted: Vec<String> = ids.iter().map(ToString::to_string).collect();
    let listing: String = observer
        .client_list::<String, String>(None, Some(wanted))
        .await
        .expect("CLIENT LIST ID must succeed");
    listing
        .lines()
        .filter_map(|line| {
            line.split_whitespace()
                .find_map(|kv| kv.strip_prefix("id="))
                .and_then(|id| id.parse().ok())
        })
        .collect()
}

/// Polls until the server no longer lists connection `id`.
async fn await_connection_closed(observer: &Client, id: i64, what: &str) {
    let deadline = Instant::now() + SETTLE;
    loop {
        if listed(observer, &[id]).await.is_empty() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: the server still lists connection {id}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The one server-side connection id fred cached for a single-node client.
fn sole_connection_id(client: &Client) -> i64 {
    let ids = client.connection_ids();
    assert_eq!(
        ids.len(),
        1,
        "a reader client holds one connection; got {ids:?}"
    );
    *ids.values().next().expect("checked above")
}

/// A standalone client this test owns, with `readers` stream readers — for the
/// tests that `quit()` it, which a `TestValkey` cannot survive (its drop-time
/// cleanup needs the data pool).
async fn own_client(readers: u32) -> ValkeyClient {
    let mut config: Config = crate::common::test_config();
    config.subscriber_pool_size = readers;
    own_client_with(config).await
}

/// [`own_client`], from `config` (its reader count included).
async fn own_client_with(config: Config) -> ValkeyClient {
    tokio::time::timeout(CONNECT_BUDGET, connect_test_client(&config))
        .await
        .expect("connect must settle")
        .expect("connect must succeed")
}

fn fresh_schema() -> KeySchema {
    KeySchema::new(&format!("test_{}:", uuid::Uuid::new_v4()))
}

// ═════════════════════════════════════════════════════════════════════════════
// Standalone: laziness, the gauge, shutdown, hooks
// ═════════════════════════════════════════════════════════════════════════════

/// T3: right after `connect` no reader holds a connection — the gauge reads 0,
/// no reader client exists, none is supervised, no connect was counted. The
/// first subscription connects exactly the reader that serves it (standalone
/// routing: `object_id % readers`), `cause="bind"`, and from then on that one
/// reader's client is the supervised one and has its hook pair armed.
///
/// Before b.azh every reader was connected (and PINGed) at startup, so the
/// opening assertions fail on that code.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn readers_hold_no_connection_until_they_first_serve_a_subscription() {
    const READERS: usize = 4;
    const OID: u64 = 6;
    let tv = tokio::time::timeout(
        CONNECT_BUDGET,
        TestValkey::setup_with_readers(READERS as u32),
    )
    .await
    .expect("the fixture must connect");
    let metrics = install_metrics(&tv).await;
    let client = tv.client();
    let mux = &client.stream_mux;

    assert_eq!(
        mux.connected_readers(),
        0,
        "startup opens no reader connection"
    );
    assert!(mux.reader_clients_for_test().is_empty());
    assert_eq!(mux.snapshot().connected_readers(), 0);
    assert_eq!(readers_connected(&metrics), 0);
    assert_eq!(reader_connect_row(&metrics), [0; 4]);
    assert!(
        client.reader_watches_for_test().is_empty(),
        "no reader client exists to supervise"
    );

    let c = subscribe(&tv, OID).await;
    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1], "the first read served");

    let serving = (OID % READERS as u64) as usize;
    let snap = mux.snapshot();
    assert_eq!(
        snap.readers.iter().map(|r| r.connected).collect::<Vec<_>>(),
        (0..READERS).map(|r| r == serving).collect::<Vec<_>>(),
        "exactly the reader serving the subscription is connected"
    );
    assert_eq!(
        snap.readers[serving].node, None,
        "off a cluster a reader's connection names no node"
    );
    assert_eq!(mux.connected_readers(), 1);
    assert_eq!(readers_connected(&metrics), 1);
    assert_eq!(
        reader_connect_row(&metrics),
        [1, 0, 0, 0],
        "one connect, cause=\"bind\", result=\"ok\""
    );
    let id = reader_client_id(client, serving);
    assert_eq!(
        client.reader_watches_for_test(),
        std::collections::HashMap::from([(serving, id.clone())]),
        "the connected reader's client is the supervised one"
    );
    assert!(
        client.armed_hook_client_ids_for_test().contains(&id),
        "the connected reader's hook pair is armed"
    );
}

/// The connected gauge counts every held reader connection once, whenever the
/// metrics arrived: a reader connected before `install_observability_hooks`
/// (none is, in production, where the hooks are installed before serving) is
/// counted when that install arms its hooks, and a reader connected
/// afterwards is counted by its connect — and not again when it arms its own
/// hooks just after publishing. Each retire at `quit()` subtracts what its
/// connection added, so the gauge reads exactly the readers held, back to 0
/// and never below it. The connects counter still counts only connects the
/// metrics saw.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_connected_gauge_counts_each_held_connection_once_and_never_goes_negative() {
    let client = own_client(2).await;
    let schema = fresh_schema();

    // Reader 0 connects with no metrics installed.
    let early = subscribe_on(&client, &schema, 2).await;
    await_mux(&client, "reader 0 connected", |s| s.readers[0].connected).await;

    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    assert_eq!(
        readers_connected(&metrics),
        1,
        "a reader connected before the metrics existed is counted when its hooks are armed"
    );
    assert_eq!(
        reader_connect_row(&metrics),
        [0; 4],
        "its connect itself, before the metrics, is not on the connects counter"
    );

    // Reader 1 connects with the metrics installed: its connect counts it,
    // and the arm right after its publish finds it counted.
    let late = subscribe_on(&client, &schema, 3).await;
    await_mux(&client, "reader 1 connected", |s| s.readers[1].connected).await;
    let reader_1 = reader_client_id(&client, 1);
    let deadline = Instant::now() + SETTLE;
    await_until(
        "reader 1's hooks must be armed after its publish, so the arm's count has run too",
        deadline,
        || client.armed_hook_client_ids_for_test().contains(&reader_1),
    )
    .await;
    assert_eq!(
        readers_connected(&metrics),
        2,
        "two readers held, each counted once: neither the arm after reader 1's connect nor \
         the install's arm of reader 0 counted a connection twice"
    );
    assert_eq!(reader_connect_row(&metrics), [1, 0, 0, 0]);

    drop((early, late));
    tokio::time::timeout(SETTLE, client.quit())
        .await
        .expect("quit must settle");
    assert_eq!(
        readers_connected(&metrics),
        0,
        "both retires subtracted exactly what each connection added: back to 0, not below it"
    );
}

/// T9: `quit()` with connected readers retires every one: no router watcher
/// is left (readers' included), no reader client is held or supervised, no
/// hook pair stays armed, the gauge is back to 0, and nothing was restarted.
///
/// A reader that finishes connecting between the supervisor's shutdown and the
/// multiplexer's inside `quit` registers a watch whose watcher exits at once,
/// so the watcher count is polled down rather than read once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quit_retires_every_connected_reader_and_leaves_no_router_watcher() {
    let client = own_client(2).await;
    let schema = fresh_schema();
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let subs = (
        subscribe_on(&client, &schema, 4).await,
        subscribe_on(&client, &schema, 5).await,
    );
    await_mux(&client, "both readers connected", |s| {
        s.connected_readers() == 2
    })
    .await;
    assert_eq!(
        client.reader_watches_for_test().len(),
        2,
        "precondition: both readers' clients are supervised"
    );
    let watchers = client.router_watchers_for_test();
    assert!(watchers >= 2);

    drop(subs);
    tokio::time::timeout(SETTLE, client.quit())
        .await
        .expect("quit must settle");

    let deadline = Instant::now() + SETTLE;
    await_until(
        "quit must leave no router watcher running, the readers' included",
        deadline,
        || client.router_watchers_for_test() == 0,
    )
    .await;
    assert!(client.reader_watches_for_test().is_empty());
    assert_eq!(client.stream_mux.connected_readers(), 0);
    assert!(client.stream_mux.reader_clients_for_test().is_empty());
    assert!(
        client.armed_hook_client_ids_for_test().is_empty(),
        "quit drains every armed hook pair, the readers' included"
    );
    assert_eq!(readers_connected(&metrics), 0);
    assert_eq!(
        reader_restarts(&metrics),
        0,
        "no reader router ended by quit is restarted"
    );
}

/// T8: a reader that connected before the hooks were installed (its own arm
/// at publish found no metrics, so armed nothing) is armed by
/// `install_observability_hooks`, with one pair: a `CLIENT KILL` of its
/// connection counts exactly one reconnect under `pool="subscriber_pool"`,
/// even after a quiet window. Left unarmed it would count none; armed twice,
/// two. (That a retarget leaves one pair per reader is pinned by
/// [`after_repeated_retargets_one_hook_pair_follows_the_reader_and_a_kill_counts_once`].)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reader_connected_before_the_hooks_counts_one_reconnect_per_kill() {
    let client = own_client(1).await;
    let schema = fresh_schema();
    let sub = subscribe_on(&client, &schema, 7).await;
    await_mux(&client, "the reader connected", |s| s.readers[0].connected).await;

    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let reader = client
        .stream_mux
        .reader_client_for_test(0)
        .expect("the reader holds a connection");
    let id = ClientLike::id(&reader).to_owned();
    assert_eq!(
        client
            .armed_hook_client_ids_for_test()
            .iter()
            .filter(|armed| **armed == id)
            .count(),
        1,
        "install_observability_hooks arms the reader connected before it"
    );

    let before = reader_reconnects(&metrics);
    let killed: i64 = client
        .pool
        .next()
        .custom(
            fred::types::CustomCommand::new_static("CLIENT KILL", None::<u16>, false),
            vec!["ID".to_owned(), sole_connection_id(&reader).to_string()],
        )
        .await
        .expect("CLIENT KILL must succeed");
    assert_eq!(killed, 1, "the kill closes exactly the reader's connection");
    let deadline = Instant::now() + SETTLE;
    await_until("the reconnect must be counted", deadline, || {
        reader_reconnects(&metrics) > before
    })
    .await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        reader_reconnects(&metrics),
        before + 1,
        "one hook pair per reader client: one kill, one reconnect"
    );

    drop(sub);
    tokio::time::timeout(SETTLE, client.quit())
        .await
        .expect("quit must settle");
}

// ═════════════════════════════════════════════════════════════════════════════
// The two-node cluster facade
// ═════════════════════════════════════════════════════════════════════════════

/// One of the facade's two primaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Node {
    A,
    B,
}

/// Where the facade first splits the slots: below it A's, from it B's.
const INITIAL_SPLIT: u16 = 8192;

/// The node ids the facade's `CLUSTER SLOTS` reply names.
const NODE_A_ID: &str = "b0a2h00000000000000000000000000000000aaa";
const NODE_B_ID: &str = "b0a2h00000000000000000000000000000000bbb";

/// See the module docs.
struct ClusterFacade {
    _a: ScriptReplyProxy,
    _b: ScriptReplyProxy,
    port_a: u16,
    port_b: u16,
    /// Slots below this are A's, the rest B's: 0 makes every slot B's.
    split: Arc<AtomicU16>,
    /// When not 0, slots from this one up are A's again: A owns two ranges,
    /// `[0, split)` and `[a_again, 16384)`, and B the slots between.
    a_again: Arc<AtomicU16>,
    /// Splits the next `CLUSTER SLOTS` frames take, one each, in order,
    /// before they answer: a map that changes between two syncs.
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    split_plan: Arc<Mutex<std::collections::VecDeque<u16>>>,
    /// A stream key whose next `XREAD` on A is answered `-MOVED <slot> <B>`.
    moved_trap: Arc<Mutex<Option<String>>>,
    moved_fired: Arc<AtomicUsize>,
    /// `CLUSTER SLOTS` frames answered, on either node: each is a routing
    /// sync that reached the wire.
    slots_served: Arc<AtomicUsize>,
    /// An `ECHO` argument whose next `ECHO` on A is stalled: nothing more is
    /// forwarded on that connection, so the `ECHO` and everything behind it
    /// never gets a reply.
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    echo_stall: Arc<Mutex<Option<String>>>,
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    echo_stalled: Arc<AtomicUsize>,
    /// An `ECHO` argument whose next `ECHO` on A is forwarded and answered,
    /// after which A reads nothing more from that connection
    /// ([`ProxyAction::StopReading`]).
    echo_stop_reading: Arc<Mutex<Option<String>>>,
    /// The arguments of each `AUTH` frame relayed, on either node, in order.
    auths: Arc<Mutex<Vec<Vec<String>>>>,
    /// `CLIENT UNBLOCK` frames relayed, on either node: a backchannel
    /// interrupting a blocked read.
    client_unblocks: Arc<AtomicUsize>,
}

impl ClusterFacade {
    async fn start() -> Self {
        let auths: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let client_unblocks = Arc::new(AtomicUsize::new(0));
        let ports: Arc<OnceLock<(u16, u16)>> = Arc::new(OnceLock::new());
        let split = Arc::new(AtomicU16::new(INITIAL_SPLIT));
        let a_again = Arc::new(AtomicU16::new(0));
        let split_plan: Arc<Mutex<std::collections::VecDeque<u16>>> =
            Arc::new(Mutex::new(std::collections::VecDeque::new()));
        let moved_trap: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let moved_fired = Arc::new(AtomicUsize::new(0));
        let slots_served = Arc::new(AtomicUsize::new(0));
        let echo_stall: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let echo_stalled = Arc::new(AtomicUsize::new(0));
        let echo_stop_reading: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let rule = |me: Node| -> ProxyRule {
            let (ports, split, trap, fired, served, stall, stalled) = (
                Arc::clone(&ports),
                Arc::clone(&split),
                Arc::clone(&moved_trap),
                Arc::clone(&moved_fired),
                Arc::clone(&slots_served),
                Arc::clone(&echo_stall),
                Arc::clone(&echo_stalled),
            );
            let (auths, client_unblocks) = (Arc::clone(&auths), Arc::clone(&client_unblocks));
            let plan = Arc::clone(&split_plan);
            let a_again = Arc::clone(&a_again);
            let stop_reading = Arc::clone(&echo_stop_reading);
            Arc::new(move |args: &[Vec<u8>]| {
                let (port_a, port_b) = *ports
                    .get()
                    .expect("the ports are recorded before any client connects");
                let name = args.first()?.to_ascii_uppercase();
                if name == b"AUTH" {
                    auths.lock().expect("never poisoned").push(
                        args[1..]
                            .iter()
                            .map(|a| String::from_utf8_lossy(a).into_owned())
                            .collect(),
                    );
                }
                if name == b"CLIENT"
                    && args
                        .get(1)
                        .is_some_and(|sub| sub.eq_ignore_ascii_case(b"UNBLOCK"))
                {
                    client_unblocks.fetch_add(1, Ordering::SeqCst);
                }
                if name == b"CLUSTER" {
                    return match args.get(1)?.to_ascii_uppercase().as_slice() {
                        b"SLOTS" => {
                            served.fetch_add(1, Ordering::SeqCst);
                            if let Some(next) = plan.lock().expect("never poisoned").pop_front() {
                                split.store(next, Ordering::SeqCst);
                            }
                            Some(ProxyAction::Eval(slots_script(
                                split.load(Ordering::SeqCst),
                                a_again.load(Ordering::SeqCst),
                                port_a,
                                port_b,
                            )))
                        }
                        b"INFO" => Some(ProxyAction::Eval(
                            "return 'cluster_state:ok\\r\\ncluster_slots_assigned:16384\\r\\n\
                             cluster_known_nodes:2\\r\\ncluster_size:2\\r\\n'"
                                .to_owned(),
                        )),
                        _ => None,
                    };
                }
                if me == Node::A && name == b"XREAD" {
                    let mut trap = trap.lock().expect("the trap lock is never poisoned");
                    let hit = trap
                        .as_ref()
                        .is_some_and(|key| args.iter().any(|arg| arg == key.as_bytes()));
                    if hit {
                        let key = trap.take().expect("checked above");
                        fired.fetch_add(1, Ordering::SeqCst);
                        let slot = fred::util::redis_keyslot(key.as_bytes());
                        return Some(ProxyAction::Eval(error_reply_script(&format!(
                            "MOVED {slot} 127.0.0.1:{port_b}"
                        ))));
                    }
                }
                if me == Node::A && name == b"ECHO" {
                    let mut stop = stop_reading.lock().expect("never poisoned");
                    let hit = stop
                        .as_ref()
                        .is_some_and(|marker| args.get(1).is_some_and(|a| a == marker.as_bytes()));
                    if hit {
                        stop.take();
                        return Some(ProxyAction::StopReading);
                    }
                }
                if me == Node::A && name == b"ECHO" {
                    let mut stall = stall.lock().expect("the stall lock is never poisoned");
                    let hit = stall
                        .as_ref()
                        .is_some_and(|marker| args.get(1).is_some_and(|a| a == marker.as_bytes()));
                    if hit {
                        stall.take();
                        stalled.fetch_add(1, Ordering::SeqCst);
                        return Some(ProxyAction::Stall);
                    }
                }
                None
            })
        };
        let a = ScriptReplyProxy::start(test_valkey_addr(), rule(Node::A)).await;
        let b = ScriptReplyProxy::start(test_valkey_addr(), rule(Node::B)).await;
        let (port_a, port_b) = (a.port(), b.port());
        ports
            .set((port_a, port_b))
            .expect("the ports are recorded once");
        Self {
            _a: a,
            _b: b,
            port_a,
            port_b,
            split,
            a_again,
            split_plan,
            moved_trap,
            moved_fired,
            slots_served,
            echo_stall,
            echo_stalled,
            echo_stop_reading,
            auths,
            client_unblocks,
        }
    }

    /// `AUTH` frames relayed so far, on either node.
    fn auths(&self) -> usize {
        self.auths.lock().expect("never poisoned").len()
    }

    /// The arguments of every `AUTH` relayed after the first `from`.
    fn auth_args_since(&self, from: usize) -> Vec<Vec<String>> {
        self.auths.lock().expect("never poisoned")[from..].to_vec()
    }

    /// `CLIENT UNBLOCK` frames relayed so far, on either node.
    fn client_unblocks(&self) -> usize {
        self.client_unblocks.load(Ordering::SeqCst)
    }

    fn port(&self, node: Node) -> u16 {
        match node {
            Node::A => self.port_a,
            Node::B => self.port_b,
        }
    }

    /// `node` as a `Server`, and as the `host:port` a reader snapshot names.
    fn server(&self, node: Node) -> Server {
        Server::new("127.0.0.1", self.port(node))
    }

    fn name(&self, node: Node) -> String {
        format!("127.0.0.1:{}", self.port(node))
    }

    /// The node that owns `slot` in the facade's current map.
    fn owner(&self, slot: u16) -> Node {
        let a_again = self.a_again.load(Ordering::SeqCst);
        if slot < self.split.load(Ordering::SeqCst) || (a_again != 0 && slot >= a_again) {
            Node::A
        } else {
            Node::B
        }
    }

    /// From the next `CLUSTER SLOTS` on, A owns two slot ranges, `[0,
    /// b_from)` and `[a_again_from, 16384)`, and B the slots between: one node
    /// serving two ranges. Call it before the client connects, which is
    /// when the client reads the map.
    fn give_a_two_ranges(&self, b_from: u16, a_again_from: u16) {
        assert!(
            0 < b_from && b_from < a_again_from,
            "A keeps slot 0 and B keeps at least one slot"
        );
        self.split.store(b_from, Ordering::SeqCst);
        self.a_again.store(a_again_from, Ordering::SeqCst);
    }

    /// From the next `CLUSTER SLOTS` on, `slot` (A's today) and every slot
    /// above it are B's. Nothing re-syncs on its own: a client sees the move
    /// only when it syncs.
    fn move_to_b_from(&self, slot: u16) {
        assert_eq!(self.owner(slot), Node::A, "the slot moves from A to B");
        self.split.store(slot, Ordering::SeqCst);
    }

    /// From the next `CLUSTER SLOTS` on, B owns every slot and A is no
    /// primary at all (a failed-over node).
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    fn move_all_to_b(&self) {
        self.split.store(0, Ordering::SeqCst);
    }

    /// The next `CLUSTER SLOTS` frames each take the next of `splits` as the
    /// map's split before answering, one per sync, in order; once they are
    /// used up the map stays at the last one.
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    fn plan_splits(&self, splits: impl IntoIterator<Item = u16>) {
        self.split_plan
            .lock()
            .expect("never poisoned")
            .extend(splits);
    }

    /// Answer the next `XREAD` on A naming `key` with `-MOVED <slot> <B>`.
    fn trap_next_xread_on_a(&self, key: String) {
        *self.moved_trap.lock().expect("never poisoned") = Some(key);
    }

    /// Stall the next `ECHO <marker>` on A, and everything behind it on that
    /// connection.
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    fn stall_next_echo_on_a(&self, marker: String) {
        *self.echo_stall.lock().expect("never poisoned") = Some(marker);
    }

    /// Forward and answer the next `ECHO <marker>` on A, then read nothing
    /// more from that connection: whatever the client writes on it after
    /// backs up in the socket buffers.
    fn stop_reading_after_echo_on_a(&self, marker: String) {
        *self.echo_stop_reading.lock().expect("never poisoned") = Some(marker);
    }

    /// `CLUSTER SLOTS` frames answered so far.
    fn slots_served(&self) -> usize {
        self.slots_served.load(Ordering::SeqCst)
    }

    fn cluster_url(&self) -> String {
        format!("redis-cluster://127.0.0.1:{}", self.port_a)
    }

    /// A `TestValkey` connected through A in cluster mode, with `readers`
    /// stream readers.
    async fn test_valkey(&self, readers: u32) -> TestValkey {
        let tv = tokio::time::timeout(
            CONNECT_BUDGET,
            TestValkey::setup_with_url_and_readers(self.cluster_url(), 4, readers),
        )
        .await
        .expect("a client must connect through the cluster facade");
        Self::self_check(tv)
    }

    /// [`Self::test_valkey`] with `provider` installed on every connection
    /// (b.19i): the pools as cluster clients, the readers as single-node ones.
    async fn test_valkey_with_provider(
        &self,
        readers: u32,
        provider: Arc<dyn fred::types::config::CredentialProvider>,
    ) -> TestValkey {
        let tv = tokio::time::timeout(
            CONNECT_BUDGET,
            TestValkey::setup_with_url_readers_and_credential_provider(
                self.cluster_url(),
                4,
                readers,
                provider,
            ),
        )
        .await
        .expect("a client must connect through the cluster facade");
        Self::self_check(tv)
    }

    fn self_check(tv: TestValkey) -> TestValkey {
        assert!(tv.client().pool.next().is_clustered());
        assert_eq!(
            tv.client().stream_mux.snapshot().routing,
            Routing::SlotAffinity,
            "fixture self-check: the facade is read as a cluster"
        );
        tv
    }

    /// The first object id above `after` whose layer stream's slot `node`
    /// owns, in a slot no id in `avoid` uses.
    fn object_on(&self, tv: &TestValkey, node: Node, after: u64, avoid: &[u64]) -> u64 {
        self.object_on_schema(tv.key_schema(), node, after, avoid)
    }

    /// [`Self::object_on`] for streams named by `schema`.
    fn object_on_schema(&self, schema: &KeySchema, node: Node, after: u64, avoid: &[u64]) -> u64 {
        let slot_of = |oid: u64| fred::util::redis_keyslot(schema.layer_stream(oid).as_bytes());
        let taken: HashSet<u16> = avoid.iter().map(|&oid| slot_of(oid)).collect();
        (after + 1..)
            .find(|&oid| {
                let slot = slot_of(oid);
                self.owner(slot) == node && !taken.contains(&slot)
            })
            .expect("some object id hashes to each node")
    }
}

/// The Lua body answering `CLUSTER SLOTS` for the map split at `split`: slots
/// below it A's, the rest B's. With `split` 0, B owns every slot and the reply
/// does not name A. With `a_again` not 0, the slots from `a_again` up are A's
/// again ([`ClusterFacade::give_a_two_ranges`]).
fn slots_script(split: u16, a_again: u16, port_a: u16, port_b: u16) -> String {
    let range = |start: u16, end: u16, port: u16, id: &str| {
        format!("{{{start}, {end}, {{'127.0.0.1', {port}, '{id}'}}}}")
    };
    let b_end = if a_again == 0 { 16383 } else { a_again - 1 };
    let mut ranges = Vec::with_capacity(3);
    if split > 0 {
        ranges.push(range(0, split - 1, port_a, NODE_A_ID));
    }
    ranges.push(range(split, b_end, port_b, NODE_B_ID));
    if a_again != 0 {
        ranges.push(range(a_again, 16383, port_a, NODE_A_ID));
    }
    format!("return {{{}}}", ranges.join(", "))
}

/// Re-syncs the readers' routing view — the pinned data-pool client's cached
/// cluster map — from the facade, as a reader does after a `-MOVED`.
async fn sync_readers_view(tv: &TestValkey) {
    tokio::time::timeout(SETTLE, tv.client().pool.clients()[0].sync_cluster())
        .await
        .expect("the routing sync must settle")
        .expect("the routing sync must succeed");
}

fn stream_slot(tv: &TestValkey, object_id: u64) -> u16 {
    fred::util::redis_keyslot(tv.key_schema().layer_stream(object_id).as_bytes())
}

/// T4 and T15: a bound reader's client is a single-node client connected to
/// its slot's owner alone — one connection, to the `Server` the readers'
/// routing view (the pinned data-pool client's) names for the slot — with no
/// database, which a cluster URL never selects. The snapshot names that node,
/// and the router records it.
///
/// The `tls_server_name` comparison is spelled out because fred's `Server`
/// equality ignores that field, but against this plaintext facade both sides
/// are `None`: it means something only against a TLS cluster. The proof that
/// the `Server` is taken whole, never rebuilt from host and port, is the source
/// guard `reader_connector_diverges_only_the_server_and_database_of_its_template`
/// in `src/storage/client.rs`.
///
/// Before b.azh a reader was a full cluster client: one connection to every
/// primary, so `active_connections` named both.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bound_cluster_reader_connects_to_its_slot_owner_alone() {
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let oid = facade.object_on(&tv, Node::A, 0, &[]);
    let slot = stream_slot(&tv, oid);

    let c = subscribe(&tv, oid).await;
    xadd_update(&tv, oid, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);

    let reader = tv
        .client()
        .stream_mux
        .reader_client_for_test(0)
        .expect("the bound reader holds a connection");
    assert!(
        !reader.is_clustered(),
        "a reader client is a single-node client"
    );
    assert_eq!(
        reader.active_connections(),
        vec![facade.server(Node::A)],
        "one connection, to the slot's owner alone"
    );
    let routing_view = tv.client().pool.clients()[0]
        .cached_cluster_state()
        .expect("the routing view is a cluster map");
    let owner = routing_view
        .get_server(slot)
        .cloned()
        .expect("the routing view names the slot's owner");
    let config = reader.client_config();
    let server = match &config.server {
        ServerConfig::Centralized { server } => server.clone(),
        other => panic!("a reader's config targets one server; got {other:?}"),
    };
    assert_eq!(server, owner, "the routing view's Server for the slot");
    #[cfg(feature = "valkey-tls")]
    assert_eq!(
        server.tls_server_name, owner.tls_server_name,
        "the TLS server name the routing view carries rides along (Server's equality ignores it)"
    );
    assert_eq!(
        config.database, None,
        "a cluster URL's database is never selected"
    );
    assert_eq!(
        tv.client().stream_mux.snapshot().readers[0].node,
        Some(facade.name(Node::A))
    );
    assert_eq!(reader_connect_row(&metrics), [1, 0, 0, 0]);
}

/// T5 (with T7 and T8 at the retarget): a one-reader multiplexer serves a slot
/// on A, then a slot on B. The reader retargets: its client now targets B
/// alone, its connection to A is gone from the server, `cause="retarget"`
/// ticks once, and the subscription on B is served. Supervision and hooks
/// follow the reader to its new client — the old client is neither supervised
/// nor armed, the watcher count is unchanged — and nothing was restarted.
/// That last check does not prove the old watch was retired before the
/// `QUIT`: a `QUIT` ends its router cleanly, and a clean end is never
/// restarted either way. The stalled-`QUIT` retarget tests below, whose old
/// router is aborted rather than ended cleanly, are what pin the retire.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cluster_reader_bound_to_a_slot_on_another_node_retargets_its_connection() {
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let client = tv.client();
    let on_a = facade.object_on(&tv, Node::A, 0, &[]);
    let on_b = facade.object_on(&tv, Node::B, 0, &[]);

    serve_once(&tv, on_a, 1).await;
    let old = client
        .stream_mux
        .reader_client_for_test(0)
        .expect("connected to A");
    let old_id = ClientLike::id(&old).to_owned();
    let old_connection = sole_connection_id(&old);
    assert_eq!(old.active_connections(), vec![facade.server(Node::A)]);
    let watchers = client.router_watchers_for_test();

    let c = subscribe(&tv, on_b).await;
    xadd_update(&tv, on_b, 1).await;
    assert_eq!(
        recv_seqs(&c, 1).await,
        [1],
        "delivery holds across the retarget"
    );
    assert!(c.take_terminal_exit().is_none());

    let new = client
        .stream_mux
        .reader_client_for_test(0)
        .expect("connected to B");
    let new_id = ClientLike::id(&new).to_owned();
    assert_ne!(new_id, old_id, "the reader replaced its client");
    assert_eq!(
        new.active_connections(),
        vec![facade.server(Node::B)],
        "the new client targets B alone"
    );
    assert_eq!(
        client.stream_mux.snapshot().readers[0].node,
        Some(facade.name(Node::B))
    );
    assert_eq!(
        reader_connect_row(&metrics),
        [1, 0, 1, 0],
        "one bind, then one retarget"
    );
    assert_eq!(readers_connected(&metrics), 1, "one reader, one connection");
    await_connection_closed(
        tv.client().pool.next(),
        old_connection,
        "the retarget must close the reader's connection to A",
    )
    .await;

    assert_eq!(
        client.reader_watches_for_test(),
        std::collections::HashMap::from([(0, new_id.clone())]),
        "supervision follows the reader to its new client"
    );
    let armed = client.armed_hook_client_ids_for_test();
    assert!(armed.contains(&new_id), "the new client's hooks are armed");
    assert!(
        !armed.contains(&old_id),
        "the old client's hook pair was disarmed at its retire"
    );
    assert_eq!(
        client.router_watchers_for_test(),
        watchers,
        "the old client's watcher exited as the new one's started"
    );
    assert_eq!(
        reader_restarts(&metrics),
        0,
        "nothing was restarted across the retarget"
    );
}

/// T7, decider clarification 3: a retargeted reader's router is supervised
/// like any other — ended, it is restarted, counted once under
/// `pool="subscriber_pool"` and logged once at ERROR naming the reader — and
/// the reader serves on the restarted router: a new subscription on B is
/// delivered on the same client, with no new connect.
///
/// The router is ended with no subscription left on the reader: a router that
/// dies fails the read in flight on it, and a subscription that read served
/// would end retryably — the client's re-subscribe is the recovery there, so
/// a new subscription is what this test serves.
///
/// Current-thread, so the [`ThreadCapture`] sees the line the supervisor's
/// spawned watcher logs; keys are purged explicitly, since a `TestValkey`
/// cleans up only on a multi-thread runtime.
#[tokio::test]
async fn a_retargeted_readers_dead_router_is_restarted_and_counted() {
    let capture = ThreadCapture::start();
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let client = tv.client();
    let on_a = facade.object_on(&tv, Node::A, 0, &[]);
    let on_b = facade.object_on(&tv, Node::B, 0, &[]);
    let also_on_b = facade.object_on(&tv, Node::B, on_b, &[on_b]);
    serve_once(&tv, on_a, 1).await;
    serve_once(&tv, on_b, 1).await;
    assert_eq!(
        reader_connect_row(&metrics),
        [1, 0, 1, 0],
        "precondition: the reader retargeted to B"
    );
    let retargeted = reader_client_id(client, 0);

    assert!(
        client.end_router_for_test(READER_POOL, 0),
        "the retargeted client's router is the supervised one"
    );
    let deadline = Instant::now() + SETTLE;
    await_until("the restart must be counted", deadline, || {
        reader_restarts(&metrics) >= 1
    })
    .await;
    let captured = capture.snapshot();
    let line = captured.only(RESTART_MSG);
    assert_eq!(line["level"], "ERROR");
    assert_field(line, "pool", READER_POOL.into());
    assert_field(line, "member", 0.into());
    assert_eq!(reader_restarts(&metrics), 1);

    serve_once(&tv, also_on_b, 1).await;
    assert_eq!(
        reader_client_id(client, 0),
        retargeted,
        "the reader serves on its restarted router: same client"
    );
    assert_eq!(
        reader_connect_row(&metrics),
        [1, 0, 1, 0],
        "a restart is not a connect"
    );
    assert_eq!(reader_restarts(&metrics), 1);
    purge_keys(&tv).await;
}

/// T8: after a reader retargets there and back (two retargets), exactly one
/// hook pair is armed per connected reader — the current client's — beside
/// the `txn` slots' pairs, and a `CLIENT KILL` of the reader's connection
/// counts one reconnect, not one per client the reader ever held.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn after_repeated_retargets_one_hook_pair_follows_the_reader_and_a_kill_counts_once() {
    const RETARGETS: usize = 2;
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let client = tv.client();
    let mut objects: Vec<u64> = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    for hop in 0..=RETARGETS {
        let node = if hop % 2 == 0 { Node::A } else { Node::B };
        let oid = facade.object_on(&tv, node, 0, &objects);
        objects.push(oid);
        serve_once(&tv, oid, 1).await;
        ids.push(reader_client_id(client, 0));
    }
    assert_eq!(
        reader_connect_row(&metrics),
        [1, 0, RETARGETS as i64, 0],
        "precondition: one bind, then a retarget per hop"
    );

    let current = ids.last().expect("at least one hop").clone();
    let armed = client.armed_hook_client_ids_for_test();
    let txn_slots = client.txn_pool.slots();
    let members = client.pool.size() + client.maintenance_pool.size() + client.snapshot_pool.size();
    assert_eq!(
        armed.len(),
        members + txn_slots + client.stream_mux.connected_readers(),
        "one hook pair per data / maintenance / snapshot member, per txn slot and per \
         connected reader: {armed:?}"
    );
    assert!(armed.contains(&current));
    for member in client
        .pool
        .clients()
        .iter()
        .chain(&client.maintenance_pool.clients())
        .chain(&client.snapshot_pool.clients())
    {
        let id = ClientLike::id(member).to_owned();
        assert!(
            armed.contains(&id),
            "pool member {id}'s pair is armed: {armed:?}"
        );
    }
    for id in client.txn_slot_client_ids_for_test() {
        assert!(
            armed.contains(&id),
            "txn slot {id}'s pair is armed: {armed:?}"
        );
    }
    for old in &ids[..RETARGETS] {
        assert!(
            !armed.contains(old),
            "a retired reader client's pair was disarmed"
        );
    }

    let reader = client
        .stream_mux
        .reader_client_for_test(0)
        .expect("connected");
    let before = reader_reconnects(&metrics);
    let killed: i64 = client
        .pool
        .next()
        .custom(
            fred::types::CustomCommand::new_static("CLIENT KILL", None::<u16>, false),
            vec!["ID".to_owned(), sole_connection_id(&reader).to_string()],
        )
        .await
        .expect("CLIENT KILL must succeed");
    assert_eq!(killed, 1);
    let deadline = Instant::now() + SETTLE;
    await_until("the reconnect must be counted", deadline, || {
        reader_reconnects(&metrics) > before
    })
    .await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        reader_reconnects(&metrics),
        before + 1,
        "one kill of the current connection counts one reconnect"
    );
}

/// T6: the reader's `XREAD` is answered with a real-shaped `-MOVED <slot> <B>`
/// after the slot moved to B. The reader's routing view still names A, so it
/// re-syncs that view once (`CLUSTER SLOTS`, which now says B) and retargets
/// to B instead of backing off and retrying A; the subscription survives with
/// nothing lost. Under `test-fault-injection` the transient budget is set to
/// zero, so a `-MOVED` charged to it would end the subscription: surviving
/// shows no charge.
///
/// Treated as an ordinary transient failure instead, the retry would land on
/// A, which still serves this facade's store, so delivery alone could not tell
/// the two apart: the retarget counter and the reader's node are what the test
/// asserts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_moved_xread_retargets_the_reader_to_the_slots_new_owner() {
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let oid = facade.object_on(&tv, Node::A, 0, &[]);
    let slot = stream_slot(&tv, oid);
    // With no transient failure allowed, a `-MOVED` charged to the budget
    // would end the subscription: the terminal-exit check below catches it.
    #[cfg(feature = "test-fault-injection")]
    tv.client()
        .stream_mux
        .set_knobs_for_test(live_edit::storage::stream_mux::StreamMuxKnobs {
            max_transient_failures: 0,
            ..tv.client().stream_mux.knobs()
        });

    let c = subscribe(&tv, oid).await;
    xadd_update(&tv, oid, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    assert_eq!(
        tv.client().stream_mux.snapshot().readers[0].node,
        Some(facade.name(Node::A))
    );

    // The slot moves to B; the next XREAD on A — the one issued after the
    // update below returns the read in flight — is answered -MOVED.
    let syncs = facade.slots_served();
    let reader_syncs = tv.client().stream_mux.routing_syncs_finished_for_test();
    facade.move_to_b_from(slot);
    facade.trap_next_xread_on_a(tv.key_schema().layer_stream(oid));
    xadd_update(&tv, oid, 2).await;
    assert_eq!(recv_seqs(&c, 1).await, [2]);
    let deadline = Instant::now() + SETTLE;
    await_until("the -MOVED must reach the reader", deadline, || {
        facade.moved_fired.load(Ordering::SeqCst) == 1
    })
    .await;

    xadd_update(&tv, oid, 3).await;
    assert_eq!(
        recv_seqs(&c, 1).await,
        [3],
        "the subscription survives the redirect with nothing lost"
    );
    assert!(c.take_terminal_exit().is_none());
    assert_eq!(
        reader_connect_row(&metrics),
        [1, 0, 1, 0],
        "the -MOVED retargeted the reader once"
    );
    assert!(
        facade.slots_served() > syncs,
        "the view still named A, so the reader re-synced it (`CLUSTER SLOTS`) to find the owner"
    );
    assert_eq!(
        tv.client().stream_mux.routing_syncs_finished_for_test() - reader_syncs,
        1,
        "one -MOVED, one readers' routing sync"
    );
    assert_eq!(
        tv.client().stream_mux.snapshot().readers[0].node,
        Some(facade.name(Node::B)),
        "the reader now reads from the slot's new owner"
    );
    let reader = tv
        .client()
        .stream_mux
        .reader_client_for_test(0)
        .expect("connected to B");
    assert_eq!(reader.active_connections(), vec![facade.server(Node::B)]);
}

/// b.zt3 T14: after data member 0 — the readers' routing view — is replaced
/// by a fresh client, the readers route through the fresh client: a `-MOVED`
/// on the reader's `XREAD` re-syncs the view through it (`sync_routing`) and
/// the retarget finds the slot's new owner in its table (`slot_owner`). The
/// retired client's command channel is closed, so a sync or lookup through it
/// would fail and leave the reader on A.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn after_data_member_0_is_replaced_the_readers_route_through_the_fresh_client() {
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let client = tv.client();
    client.set_member_replacement_knobs_for_test(
        live_edit::storage::member_replacement::ReplaceKnobs {
            grace: Duration::ZERO,
            ..live_edit::storage::member_replacement::ReplaceKnobs::production(Duration::ZERO)
        },
    );
    let old = ClientLike::id(client.pool.member(0)).to_owned();
    for _ in 0..live_edit::storage::member_replacement::MEMBER_REPLACE_AFTER_FAST_VERDICTS {
        client.fast_verdict_for_test(POOL_LABEL_DATA, 0);
    }
    let deadline = Instant::now() + SETTLE;
    // Replaced, and its task finished: the old client's retirement included.
    await_until("data member 0 must be replaced", deadline, || {
        ClientLike::id(client.pool.member(0)) != old
            && client.member_replacement_tasks_for_test() == 0
    })
    .await;
    let fresh = ClientLike::id(client.pool.member(0)).to_owned();
    assert_eq!(client.stream_mux.routing_client_id_for_test(), fresh);

    let oid = facade.object_on(&tv, Node::A, 0, &[]);
    let slot = stream_slot(&tv, oid);
    let c = subscribe(&tv, oid).await;
    xadd_update(&tv, oid, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    assert_eq!(
        client.stream_mux.snapshot().readers[0].node,
        Some(facade.name(Node::A))
    );
    let reader_syncs = client.stream_mux.routing_syncs_finished_for_test();
    facade.move_to_b_from(slot);
    facade.trap_next_xread_on_a(tv.key_schema().layer_stream(oid));
    xadd_update(&tv, oid, 2).await;
    assert_eq!(recv_seqs(&c, 1).await, [2]);
    let deadline = Instant::now() + SETTLE;
    await_until("the -MOVED must reach the reader", deadline, || {
        facade.moved_fired.load(Ordering::SeqCst) == 1
    })
    .await;
    xadd_update(&tv, oid, 3).await;
    assert_eq!(recv_seqs(&c, 1).await, [3]);

    assert!(c.take_terminal_exit().is_none());
    assert_eq!(
        client.stream_mux.routing_syncs_finished_for_test() - reader_syncs,
        1,
        "one -MOVED, one readers' routing sync"
    );
    assert_eq!(
        reader_connect_row(&metrics),
        [1, 0, 1, 0],
        "the reader retargeted once, to the owner the fresh client's table names"
    );
    assert_eq!(
        client.stream_mux.snapshot().readers[0].node,
        Some(facade.name(Node::B))
    );
}

/// b.zt3 T3, the two-node half: a connection to node A carrying a stalled
/// command in A's slot range (an `ECHO` sent to A, whose argument fred hashes
/// into A's range) is not closed by the fast verdict while node B keeps
/// answering another member's reads in B's range — another node's replies are
/// no evidence for A. With nothing from A on any connection, only the slow
/// verdict (here a 4 s test bound) closes it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_connection_silent_on_one_node_is_not_closed_by_a_fast_verdict_while_another_node_answers(
) {
    let facade = ClusterFacade::start().await;
    let tv = evidence_test_valkey(&facade).await;
    let oid_on_b = facade.object_on(&tv, Node::B, 0, &[]);
    let _reads_on_b = super::b_zt3_support::PeerReads::start(
        tv.client().pool.member(1).clone(),
        tv.key_schema().layer_stream(oid_on_b),
    );
    // fred hashes `ECHO`'s argument like a key, and the verdict judges a
    // command by its slot's range: a marker in A's range, so the stall is
    // judged by A's replies, which nothing sends.
    let marker = hashing_where("zt3-echo", |slot| facade.owner(slot) == Node::A);

    let closed = close_a_stalled_echo(&facade, &tv, marker, true).await;

    assert_closed_by_the_slow_verdict(&closed, "B's replies must not count as evidence for A");
}

// ── b.zt3 rev 3: which replies are evidence for a stalled command ────────────
//
// A command is judged by its hash slot's range when the client's cluster
// routing places that slot on the connection's node, whether the slot comes
// from a key, a `ClusterHash::Custom` pin or an argument fred hashes (`ECHO`'s
// is one); any other command is judged by every reply from its connection's
// node. A reply is stamped by the same rule. Each test below stages T3's
// fixture — the slow bound at 4 s, an `ECHO <marker>` stalled on A through data
// member 0, data member 1 the only other connection sending commands (one
// every 200 ms), no reader connected — and varies only which node and slot
// answer member 1.

/// The slow bound the evidence tests run under: past the fast bound and its
/// evidence lag, so the fast rule has had its chance before it.
const EVIDENCE_TEST_SLOW_BOUND: Duration = Duration::from_secs(4);

/// The latest the fast verdict may close the stalled connection, from the
/// `ECHO`'s send: the fast bound, one scan interval, and 1 s of slack (3.5 s at
/// the production values), under [`EVIDENCE_TEST_SLOW_BOUND`].
const EVIDENCE_TEST_FAST_DEADLINE: Duration =
    live_edit::storage::test_support::UNRESPONSIVE_FAST_TIMEOUT
        .saturating_add(live_edit::storage::test_support::UNRESPONSIVE_CHECK_INTERVAL)
        .saturating_add(Duration::from_secs(1));

/// A `TestValkey` connected through the facade with one stream reader and the
/// slow bound at [`EVIDENCE_TEST_SLOW_BOUND`] (the fast bound at its
/// production value).
async fn evidence_test_valkey(facade: &ClusterFacade) -> TestValkey {
    live_edit::storage::client::with_liveness_overrides_for_test(
        live_edit::storage::client::LivenessOverrides {
            slow_timeout: Some(EVIDENCE_TEST_SLOW_BOUND),
            ..Default::default()
        },
        facade.test_valkey(1),
    )
    .await
}

/// A fresh string starting `tag` whose hash slot satisfies `wanted`.
fn hashing_where(tag: &str, wanted: impl Fn(u16) -> bool) -> String {
    (0..)
        .map(|n| format!("{tag}-{n}-{}", uuid::Uuid::new_v4()))
        .find(|s| wanted(fred::util::redis_keyslot(s.as_bytes())))
        .expect("some string hashes to every slot range")
}

/// How the stalled connection to A was closed.
struct StalledEchoClosed {
    error: fred::error::Error,
    /// From the `ECHO`'s send to the error reaching data member 0's error
    /// stream.
    after: Duration,
}

/// Stall `ECHO <marker>` on A through data member 0 — sent to A by name when
/// `named_a`, otherwise routed by the slot fred hashes `marker` to — and wait
/// for the connection's close on the member's error stream.
async fn close_a_stalled_echo(
    facade: &ClusterFacade,
    tv: &TestValkey,
    marker: String,
    named_a: bool,
) -> StalledEchoClosed {
    let member = tv.client().pool.member(0).clone();
    let mut errors = member.error_rx();
    facade.stall_next_echo_on_a(marker.clone());
    let sent = Instant::now();
    let echo = async {
        if named_a {
            member
                .with_cluster_node(facade.server(Node::A))
                .echo::<String, _>(marker)
                .await
        } else {
            member.echo::<String, _>(marker).await
        }
    };
    let closed = async {
        let closed = tokio::time::timeout(SETTLE, errors.recv()).await;
        (closed, sent.elapsed())
    };
    // The ECHO's own outcome (re-sent after the verdict) is not the subject.
    let (_echo, (closed, after)) = tokio::join!(echo, closed);
    let (error, server) = closed
        .expect("the silent connection to A must be closed within the bound")
        .expect("the error stream stays open");
    assert_eq!(
        facade.echo_stalled.load(Ordering::SeqCst),
        1,
        "the marker ECHO must have been stalled on A"
    );
    assert_eq!(server, Some(facade.server(Node::A)));
    StalledEchoClosed { error, after }
}

/// Asserts the stalled connection was closed by the slow verdict, at no less
/// than the slow bound: the replies member 1 got were no evidence for it.
fn assert_closed_by_the_slow_verdict(closed: &StalledEchoClosed, why: &str) {
    assert_eq!(
        fred::live_edit::liveness::verdict_of(&closed.error),
        Some(fred::live_edit::liveness::Verdict::Slow),
        "{why}: {:?}",
        closed.error
    );
    assert!(
        closed.after >= EVIDENCE_TEST_SLOW_BOUND,
        "the slow verdict cannot come before the slow bound: closed after {:?}",
        closed.after
    );
}

/// b.zt3 rev 3 T-K2a, the judging side (regression): a command sent to node A
/// by name whose argument hashes into B's range is judged by A's replies, not
/// by its slot's. A's connection carries `ECHO <marker>` sent with
/// `with_cluster_node(A)`, `marker` hashing into B's range, stalled; member 1
/// reads keys in B's range, one of them in the marker's own slot. B's replies
/// are no evidence for A, so only the slow verdict closes the connection.
/// Judged by the marker's raw slot, the stall would see B's replies to that
/// slot and get the fast verdict at about 2.5 s.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_command_sent_to_a_named_node_whose_slot_another_node_owns_is_not_judged_by_that_slots_replies(
) {
    let facade = ClusterFacade::start().await;
    let tv = evidence_test_valkey(&facade).await;
    let marker = hashing_where("zt3-k2a", |slot| facade.owner(slot) == Node::B);
    let marker_slot = fred::util::redis_keyslot(marker.as_bytes());
    let in_marker_slot = format!("{{{marker}}}:peer");
    assert_eq!(
        fred::util::redis_keyslot(in_marker_slot.as_bytes()),
        marker_slot,
        "fixture self-check: the peer key shares the marker's slot"
    );
    let peer = tv.client().pool.member(1).clone();
    let _: Option<String> = peer
        .get(&in_marker_slot)
        .await
        .expect("fixture self-check: B answers member 1's read in the marker's slot");
    let _reads_in_marker_slot =
        super::b_zt3_support::PeerReads::start(peer.clone(), in_marker_slot);
    let _reads_on_b = super::b_zt3_support::PeerReads::start(
        peer,
        tv.key_schema()
            .layer_stream(facade.object_on(&tv, Node::B, 0, &[])),
    );

    let closed = close_a_stalled_echo(&facade, &tv, marker, true).await;

    assert_closed_by_the_slow_verdict(
        &closed,
        "B's replies to the marker's slot must not count as evidence for a command sent to A",
    );
}

/// b.zt3 rev 3 T-K2b, the stamping side (regression): a reply to a command
/// sent to node B by name whose argument hashes into A's range is stamped as
/// B's, not into that slot's range. A's connection carries `ECHO <marker>`,
/// `marker` hashing into A's range and routed there by its slot, stalled;
/// member 1 sends `ECHO <m2>` to B by name every 200 ms, `m2` hashing into the
/// marker's slot, and B answers. Those replies are no evidence for A, so only
/// the slow verdict closes the connection. Stamped into the marker's slot, they
/// would give the stall a fast verdict at about 2.5 s.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reply_from_a_named_node_to_a_slot_another_node_owns_is_no_evidence_for_that_slot() {
    let facade = ClusterFacade::start().await;
    let tv = evidence_test_valkey(&facade).await;
    let marker = hashing_where("zt3-k2b", |slot| facade.owner(slot) == Node::A);
    let marker_slot = fred::util::redis_keyslot(marker.as_bytes());
    let m2 = format!("{{{marker}}}:peer");
    assert_eq!(
        fred::util::redis_keyslot(m2.as_bytes()),
        marker_slot,
        "fixture self-check: m2 hashes into the marker's slot"
    );
    let peer = tv.client().pool.member(1).clone();
    let echoed: String = peer
        .with_cluster_node(facade.server(Node::B))
        .echo(m2.as_str())
        .await
        .expect("fixture self-check: B answers member 1's ECHO sent to it by name");
    assert_eq!(echoed, m2);
    let _echoes_on_b = super::b_zt3_support::PeerReads::echoes_on(peer, facade.server(Node::B), m2);

    let closed = close_a_stalled_echo(&facade, &tv, marker, false).await;

    assert_closed_by_the_slow_verdict(
        &closed,
        "B's replies to commands hashing into A's range must not count as evidence for A",
    );
}

/// b.zt3 rev 3 T-K3, the rule pin — **not** a regression test: it passes on
/// the code before rev 3 too. It pins the chosen rule (a command routed by a
/// hashed argument is judged by its own slot's range) against judging such a
/// command by every range on its node (the rejected options B and C), which
/// would let another shard on the same node close a healthy connection behind
/// its shard's long script. A owns two slot ranges, `[0, 4096)` and `[12288,
/// 16384)`; A's connection carries `ECHO <marker>` hashing into the first,
/// routed there by its slot, stalled; member 1 reads only keys in the second,
/// which A answers. Those replies are no evidence for the first range, so only
/// the slow verdict closes the connection.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_command_routed_by_its_slot_is_judged_by_its_own_range_not_every_range_on_its_node() {
    const B_FROM: u16 = 4096;
    const A_AGAIN_FROM: u16 = 12288;
    let facade = ClusterFacade::start().await;
    facade.give_a_two_ranges(B_FROM, A_AGAIN_FROM);
    let tv = evidence_test_valkey(&facade).await;
    let routing = tv
        .client()
        .pool
        .member(0)
        .cached_cluster_state()
        .expect("a cluster client holds its routing");
    let a_ranges: Vec<(u16, u16)> = routing
        .slots()
        .iter()
        .filter(|range| range.primary == facade.server(Node::A))
        .map(|range| (range.start, range.end))
        .collect();
    assert_eq!(
        a_ranges,
        [(0, B_FROM - 1), (A_AGAIN_FROM, 16383)],
        "fixture self-check: the client's routing places two ranges on A"
    );
    let marker = hashing_where("zt3-k3", |slot| slot < B_FROM);
    let second_range_key = hashing_where(&format!("{}zt3-k3-peer", tv.prefix()), |slot| {
        slot >= A_AGAIN_FROM
    });
    let peer = tv.client().pool.member(1).clone();
    let _: Option<String> = peer
        .get(&second_range_key)
        .await
        .expect("fixture self-check: A answers member 1's read in its second range");
    let _reads_second_range = super::b_zt3_support::PeerReads::start(peer, second_range_key);

    let closed = close_a_stalled_echo(&facade, &tv, marker, false).await;

    assert_closed_by_the_slow_verdict(
        &closed,
        "replies from A's other range must not count as evidence for a command in its first",
    );
}

/// b.zt3 rev 3 T-Kpos, the positive control for T-K2a, T-K2b and T-K3: with
/// member 1 reading a key in the stalled command's range on A — another slot of
/// it, so the evidence is the range's, not the slot's — the fast verdict
/// closes A's connection within the fast bound, one scan interval and 1 s.
/// Evidence does reach the registry through the facade, so the slow verdicts
/// in those tests are the rule's, not a fixture that stamps nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stalled_command_whose_range_answers_another_connection_on_its_node_gets_the_fast_verdict(
) {
    let facade = ClusterFacade::start().await;
    let tv = evidence_test_valkey(&facade).await;
    let marker = hashing_where("zt3-kpos", |slot| facade.owner(slot) == Node::A);
    let marker_slot = fred::util::redis_keyslot(marker.as_bytes());
    let same_range_key = hashing_where(&format!("{}zt3-kpos-peer", tv.prefix()), |slot| {
        facade.owner(slot) == Node::A && slot != marker_slot
    });
    let _reads_same_range =
        super::b_zt3_support::PeerReads::start(tv.client().pool.member(1).clone(), same_range_key);

    let closed = close_a_stalled_echo(&facade, &tv, marker, false).await;

    assert_eq!(
        fred::live_edit::liveness::verdict_of(&closed.error),
        Some(fred::live_edit::liveness::Verdict::Fast),
        "A answering member 1 in the stalled command's range is evidence: {:?}",
        closed.error
    );
    assert!(
        closed.after <= EVIDENCE_TEST_FAST_DEADLINE,
        "the fast verdict must close the connection within {EVIDENCE_TEST_FAST_DEADLINE:?}: \
         closed after {:?}",
        closed.after
    );
}

// ── b.zt3: nothing waits again on a connection whose wait expired ───────────
//
// The guard on expired connections (rev 3) at the two sites a cluster client
// reaches after a flush expired, and the close of the failed connection
// (rev 4), which waits on nothing either. Staging: A stops reading data
// member 0's connection; a multi-attempt `GET` and then a 16 MiB one-attempt
// `SET`, both for keys in A's range, are fed to it, because another command is
// queued behind each; a command for B is then sent, and before that write,
// which can reach B's socket, fred flushes every connection holding fed frames
// (b.bo2): A's flush cannot complete, expires at the 4 s slow bound, and fred
// ignores the error. The write to B is then flushed, and fred's router-wide
// flush after it (`Connections::flush`, which flushes every connection holding
// fed frames) reaches A first: its `flush` guard fails at once, logging
// nothing itself, and the caller of that flush logs fred's generic DEBUG
// line, `Failed to flush connections: …`, which neither test's reach marker
// counts. After that:
//
// - with nothing more for A, the router's own re-flush before it waits idle
//   reaches A's `flush` guard again, and logs its DEBUG line (the `flush`
//   site);
// - with a `GET` for A next in the same pipeline, its write reaches A in the
//   same router step, before the read loop can close A (the `feed` site, since
//   a command is still queued behind the pipeline).
//
// Either way A is then closed, and its close returns the in-flight `GET` and
// `SET`. fred's stock close would first try to flush the bytes still buffered,
// which cannot complete, under its own 5 s bound.
//
// Current-thread runtime: every command is queued before the router runs, and
// the thread-local capture sees fred's log lines, DEBUG included.

/// A value larger than every socket buffer between the client and a proxy that
/// stopped reading (`tcp_wmem` and the receive window together hold a few MiB
/// on loopback), so a flush of it cannot complete.
const UNSENDABLE_VALUE_LEN: usize = 16 * 1024 * 1024;

/// The latest A's connection may be closed, from the staged commands' send:
/// one slow bound and 2 s of slack. A second expired wait would add another
/// 4 s bound, and a close that flushed would add fred's 5 s close bound.
const ONE_EXPIRED_WAIT_DEADLINE: Duration =
    EVIDENCE_TEST_SLOW_BOUND.saturating_add(Duration::from_secs(2));

/// Bound on the close reaching data member 0's error stream, and the staged
/// commands' own timeout. It outlasts the slowest regression — two expired
/// bounds and fred's 5 s close, about 13 s — so a regression fails on
/// [`ONE_EXPIRED_WAIT_DEADLINE`] and the WARN count, not on a timeout here, and
/// the staged commands end with the close, not with a command timeout.
const EXPIRED_CLOSE_WAIT: Duration = Duration::from_secs(25);

/// How the staged expired flush ended.
struct ExpiredFlush {
    /// A's `host:port`, as fred's log lines name it.
    a: String,
    /// The error data member 0's error stream got for its connection to A.
    error: fred::error::Error,
    /// From the staged commands' send to that error.
    after: Duration,
    /// Every fred WARN naming data member 0's client and an unresponsive
    /// connection.
    warns: Vec<String>,
    /// Every captured fred line, DEBUG included.
    lines: Vec<String>,
    /// The one-attempt pipeline's outcome (`try_all`): fred answers a
    /// pipeline whose command fails — a write or connection error, as here —
    /// at once with that error, as a single-element result; a server's error
    /// reply is instead buffered as that command's element, and the pipeline
    /// answered once every element has arrived.
    pipeline: Vec<Result<fred::types::Value, fred::error::Error>>,
    /// The multi-attempt `GET` fed to A ahead of the `SET`.
    resent: Result<Option<String>, fred::error::Error>,
}

/// Stage the expired flush described above on data member 0's connection to
/// A. Without `then_feed_a`: the multi-attempt `GET`, the `SET`, then an
/// `ECHO` for B. With it: the `GET`, then one pipeline of the `SET`, an `ECHO`
/// for B sent as a custom command (which fred always flushes) and a `GET` for
/// A, with an `ECHO` for B queued behind it.
async fn expire_a_flush_on_a(
    facade: &ClusterFacade,
    tv: &TestValkey,
    then_feed_a: bool,
) -> ExpiredFlush {
    super::b_zt3_support::bridge_fred_log();
    let capture = ThreadCapture::start_debug();
    let member = tv.client().pool.member(0).clone();
    let id = ClientLike::id(&member).to_owned();
    let mut errors = member.error_rx();
    let on = |node: Node, tag: &str| {
        hashing_where(&format!("{}{tag}", tv.prefix()), |slot| {
            facade.owner(slot) == node
        })
    };
    let trigger = on(Node::A, "zt3-stop-reading");
    facade.stop_reading_after_echo_on_a(trigger.clone());
    let echoed: String = member
        .echo(trigger.as_str())
        .await
        .expect("fixture self-check: A answers the ECHO after which it stops reading");
    assert_eq!(echoed, trigger);
    let resent_key = on(Node::A, "zt3-resent-get");
    let unsendable_key = on(Node::A, "zt3-unsendable");
    let fed_get_key = on(Node::A, "zt3-fed-get");
    let flushed_on_b = on(Node::B, "zt3-flushed-on-b");
    let to_b = on(Node::B, "zt3-to-b");
    let one_attempt =
        fred::clients::Pipeline::from(member.with_options(&fred::types::config::Options {
            max_attempts: Some(1),
            timeout: Some(EXPIRED_CLOSE_WAIT),
            ..Default::default()
        }));
    let value = "x".repeat(UNSENDABLE_VALUE_LEN);
    let _: () = one_attempt
        .set(&unsendable_key, value.as_str(), None, None, false)
        .await
        .expect("queueing into a pipeline does no I/O");
    if then_feed_a {
        let _: () = one_attempt
            .custom(
                fred::types::CustomCommand::new_static(
                    "ECHO",
                    fred::types::ClusterHash::FirstKey,
                    false,
                ),
                vec![flushed_on_b.as_str()],
            )
            .await
            .expect("queueing into a pipeline does no I/O");
        let _: () = one_attempt
            .get(&fed_get_key)
            .await
            .expect("queueing into a pipeline does no I/O");
    }
    // The client's default attempts, with the long timeout.
    let patient = member.with_options(&fred::types::config::Options {
        timeout: Some(EXPIRED_CLOSE_WAIT),
        ..Default::default()
    });

    let sent = Instant::now();
    let closed = async {
        let closed = tokio::time::timeout(EXPIRED_CLOSE_WAIT, errors.recv()).await;
        (closed, sent.elapsed())
    };
    // Polled in this order, so queued in it: the multi-attempt `GET` (fed to
    // A first, while A's buffer is still small), the pipeline, then the `ECHO`
    // for B. Without `then_feed_a` that `ECHO`, sent with nothing behind it,
    // is the write before which fred flushes every connection holding fed
    // frames; with it, the pipeline's custom `ECHO` is, and this one only
    // keeps a command queued behind the pipeline. Its own outcome is not the
    // subject.
    let (resent, pipeline, _echo_b, (closed, after)) = tokio::join!(
        patient.get::<Option<String>, _>(&resent_key),
        one_attempt.try_all::<fred::types::Value>(),
        patient.echo::<String, _>(to_b.as_str()),
        closed,
    );
    let (error, server) = closed
        .expect("A's connection must be closed within the bound")
        .expect("the error stream stays open");
    assert_eq!(server, Some(facade.server(Node::A)));
    let captured = capture.snapshot();
    let lines = super::b_zt3_support::messages_containing(&captured, "");
    let warns = super::b_zt3_support::messages_containing(&captured, "Unresponsive connection to ")
        .into_iter()
        .filter(|line| line.starts_with(&format!("{id}: ")))
        .collect();
    ExpiredFlush {
        a: facade.name(Node::A),
        error,
        after,
        warns,
        lines,
        pipeline,
        resent,
    }
}

impl ExpiredFlush {
    /// How many captured lines contain `needle`.
    fn count(&self, needle: &str) -> usize {
        self.lines
            .iter()
            .filter(|line| line.contains(needle))
            .count()
    }

    /// fred's DEBUG line for `before_wait`'s re-flush of A failing.
    fn reflush_failed(&self) -> String {
        format!("Failed to flush fed frames to {} before waiting", self.a)
    }

    /// fred's DEBUG line for a write to A failing (`router/utils.rs`'s
    /// `write_command`).
    fn write_failed(&self) -> String {
        format!("{}: Error sending frame to socket", self.a)
    }
}

/// Asserts what both expired-flush tests share:
///
/// - one WARN, for the expired flush;
/// - A's connection closed by the slow verdict by
///   [`ONE_EXPIRED_WAIT_DEADLINE`]: one bound, with no second wait and a close
///   that waits on nothing;
/// - b.zt3 rev 4 decider rider 1: the close of the connection judged dead
///   returns the same in-flight commands as a stock close. Both staged
///   commands were in flight on A, and each ends as a returned command does:
///   the one-attempt `SET` (in the pipeline) fails with the slow verdict's
///   error, and the multi-attempt `GET` is re-sent on the rebuilt connection
///   and answered. A command the close did not return would be dropped with
///   the connection: its caller would fail at once with fred's dropped-channel
///   error (`Unknown`, "channel closed"), not the verdict's, and the `GET`
///   would not be re-sent. The 16 MiB value never reaches the store.
fn assert_one_expired_flush(expired: &ExpiredFlush) {
    assert_eq!(
        expired.warns.len(),
        1,
        "one expired wait, one WARN; nothing waits on the failed connection again: {:#?}",
        expired.warns
    );
    assert!(
        expired.warns[0].contains("(expired flush)"),
        "the one WARN is the expired flush's: {}",
        expired.warns[0]
    );
    assert!(
        expired.after <= ONE_EXPIRED_WAIT_DEADLINE,
        "A's connection must be closed one slow bound after the commands were sent, with no \
         second wait and a close that waits on nothing: closed after {:?}",
        expired.after
    );
    assert_eq!(
        fred::live_edit::liveness::verdict_of(&expired.error),
        Some(fred::live_edit::liveness::Verdict::Slow),
        "{:?}",
        expired.error
    );
    // Rider 1: the close returned both in-flight commands.
    let [Err(pipeline_error)] = expired.pipeline.as_slice() else {
        panic!(
            "the one-attempt pipeline must fail with its connection: {:?}",
            expired.pipeline
        );
    };
    assert_eq!(
        fred::live_edit::liveness::verdict_of(pipeline_error),
        Some(fred::live_edit::liveness::Verdict::Slow),
        "the in-flight one-attempt SET the close returned fails with the verdict's error, not \
         fred's dropped-channel error: {pipeline_error:?}"
    );
    assert_eq!(
        expired.resent.as_ref().unwrap_or_else(|e| panic!(
            "the in-flight multi-attempt GET the close returned must be re-sent and \
                 answered: {e:?}"
        )),
        &None,
        "the GET's key holds nothing"
    );
}

/// b.zt3 rev 3, the guard on expired connections at the re-flush, and rev 4's
/// close of a connection judged dead (regression for both): after A's flush
/// expired, with nothing queued, fred's router flushes every connection still
/// holding unsent bytes before it waits idle. That flush of A must fail at
/// once — fred logs it at DEBUG, once, and writes nothing more to A — so the
/// router reaches its read loop, which closes A with the slow verdict, and the
/// close waits on nothing: A is closed within one bound + 2 s. (fred's flush
/// of every connection holding fed frames after the write to B runs before it
/// and fails on A at once too; its caller logs fred's generic DEBUG line,
/// `Failed to flush connections: …`, not the reach marker's.) Without the
/// guard on `flush`,
/// the first of those re-flushes waits out a second bound and logs a second
/// `(expired flush)` WARN; with a close that flushed, the close adds fred's
/// 5 s close bound.
#[tokio::test]
async fn a_connection_whose_flush_expired_is_not_flushed_again_before_the_router_waits() {
    let facade = ClusterFacade::start().await;
    let tv = evidence_test_valkey(&facade).await;

    let expired = expire_a_flush_on_a(&facade, &tv, false).await;

    assert_one_expired_flush(&expired);
    assert_eq!(
        expired.count(&expired.reflush_failed()),
        1,
        "the reach marker: before_wait's re-flush of A ran once and failed"
    );
    assert_eq!(
        expired.count(&expired.write_failed()),
        0,
        "nothing more was written to A"
    );
    purge_keys(&tv).await;
}

/// b.zt3 rev 3, the guard on expired connections at a fed write, and rev 4's
/// close of a connection judged dead (regression for both): the pipelined
/// one-attempt `GET` for A, written in the same router step as A's expired
/// flush, is fed (a command is still queued behind the pipeline). On the
/// failed connection that feed must fail at once — fred logs the failed write
/// at DEBUG, once — which drops the connection with the slow verdict, and the
/// close waits on nothing: A is closed within one bound + 2 s. Without the
/// guard on `feed`, the feed first flushes the 16 MiB still buffered, waits out
/// a second bound and logs an `(expired write)` WARN (fred's label for an
/// expired feed or send); with a close that flushed, the close adds fred's 5 s
/// close bound.
#[tokio::test]
async fn a_command_written_to_a_connection_whose_flush_expired_fails_at_once_without_being_fed() {
    let facade = ClusterFacade::start().await;
    let tv = evidence_test_valkey(&facade).await;

    let expired = expire_a_flush_on_a(&facade, &tv, true).await;

    assert_one_expired_flush(&expired);
    assert_eq!(
        expired.count(&expired.write_failed()),
        1,
        "the reach marker: the GET's write to A failed"
    );
    assert_eq!(
        expired.count(&expired.reflush_failed()),
        0,
        "A was dropped before the router could re-flush it"
    );
    purge_keys(&tv).await;
}

/// T1 end to end: node affinity. Reader 0 serves a slot on A and reader 1 a
/// slot on B, then both go free. A new subscription to another slot on B goes
/// to reader 1, already connected to B, not to the lower-index reader 0 —
/// so no retarget is needed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_free_reader_already_on_the_owner_takes_a_new_binding_without_a_retarget() {
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(2).await;
    let metrics = install_metrics(&tv).await;
    let on_a = facade.object_on(&tv, Node::A, 0, &[]);
    let on_b = facade.object_on(&tv, Node::B, 0, &[]);
    let also_on_b = facade.object_on(&tv, Node::B, on_b, &[on_b]);

    let first = subscribe(&tv, on_a).await;
    let second = subscribe(&tv, on_b).await;
    xadd_update(&tv, on_a, 1).await;
    xadd_update(&tv, on_b, 1).await;
    assert_eq!(recv_seqs(&first, 1).await, [1]);
    assert_eq!(recv_seqs(&second, 1).await, [1]);
    let snap = tv.client().stream_mux.snapshot();
    assert_eq!(
        (snap.readers[0].node.clone(), snap.readers[1].node.clone()),
        (Some(facade.name(Node::A)), Some(facade.name(Node::B))),
        "precondition: reader 0 is on A, reader 1 on B"
    );
    drop((first, second));
    await_mux(tv.client(), "both readers free", |s| s.subscriptions() == 0).await;

    let c = subscribe(&tv, also_on_b).await;
    xadd_update(&tv, also_on_b, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    let snap = tv.client().stream_mux.snapshot();
    assert_eq!(
        (snap.readers[0].subscriptions, snap.readers[1].subscriptions),
        (0, 1),
        "the new binding goes to the free reader already connected to its owner"
    );
    assert_eq!(
        reader_connect_row(&metrics),
        [2, 0, 0, 0],
        "two binds and no retarget"
    );
}

/// T13: a command from outside the reader loop, queued behind a clustered
/// reader's *live* blocked `XREAD`, interrupts the read instead of waiting it
/// out: fred's `Blocking::Interrupt` sends `CLIENT UNBLOCK` first, over a
/// backchannel, so the command is answered at once rather than after the
/// 30 s BLOCK, and the reader takes the interrupted read's `-UNBLOCKED` as a
/// re-plan (counted). The test sends that command itself — a `PING` on the
/// reader's client — to pin the interrupt and the reader's re-plan. It does
/// not tell a single-node reader client from a cluster one: a `PING` is
/// interrupted the same way on either.
///
/// No production path sends such a command since b.19i: a reader sends its
/// own other commands — probes, head checks, keepalive and its credential
/// re-AUTH — only between reads, and fred schedules no credential refresh on a
/// reader (the re-AUTH's own path is
/// [`a_cluster_readers_due_reauth_wakes_its_read_and_auths_between_reads`]).
/// What `Blocking::Interrupt` still serves in production is the residual
/// case, and it looks different from this test: the reader's next command
/// after a client-side `XREAD` timeout that did not force a reconnect, while
/// that timed-out read is still blocked on the server. There the interrupted
/// read is the reader's own abandoned one, its `-UNBLOCKED` goes to a caller
/// that has already dropped it, and nothing is re-planned or counted. (A pin
/// of that residual path is deferred.)
///
/// The subscription is dropped once the read is blocked and before the
/// `PING`. A subscription leaving does not wake the read, and the reader
/// applies the removal before it plans another read, so the interrupted
/// reader has nothing to re-issue. With the subscription still live, its
/// re-issued XREAD could be written ahead of the `PING`, which would then
/// wait behind the new BLOCK: a race in fred's interrupt (it decides when the
/// command is sent), which is why the readers no longer leave their re-AUTH
/// to fred. Re-planning with nothing lost is pinned by
/// [`a_client_unblock_on_a_single_node_cluster_reader_is_replanned_without_loss`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_command_queued_behind_a_cluster_readers_blocked_read_interrupts_it_and_the_read_is_replanned(
) {
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let oid = facade.object_on(&tv, Node::A, 0, &[]);
    let c = subscribe(&tv, oid).await;
    xadd_update(&tv, oid, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    let reader = tv
        .client()
        .stream_mux
        .reader_client_for_test(0)
        .expect("connected");
    await_reader_blocked_in_xread(tv.client().pool.next(), &reader).await;
    drop(c);
    let unblocks_before = facade.client_unblocks();

    tokio::time::timeout(SETTLE, reader.ping::<()>(None))
        .await
        .expect("a command behind the blocked XREAD must interrupt it, not wait out the 30 s BLOCK")
        .expect("the PING is answered");
    let deadline = Instant::now() + SETTLE;
    await_until(
        "the interrupted read's -UNBLOCKED must reach the reader",
        deadline,
        || read_unblocked_retries(&metrics) >= 1,
    )
    .await;
    assert!(
        facade.client_unblocks() > unblocks_before,
        "the interrupt is fred's CLIENT UNBLOCK, relayed by the facade"
    );
}

/// b.19i on a cluster reader (design item 4): the busy-reader case of
/// `stream_mux_reauth_test`, through the two-node facade. The pools are
/// cluster clients; the reader is a single-node client to node A. Under a
/// [`ScriptedProvider`] the pools start no fred refresh while their routers
/// keep running.
///
/// A credential granted while the reader's read is blocked on a quiet layer
/// wakes that read; the reader sends one `AUTH` on its own connection between
/// reads, answered well inside the 30 s BLOCK (the wait is one [`SETTLE`]).
/// No `CLIENT UNBLOCK` crosses either node and no read is re-planned on
/// `-UNBLOCKED`; fred schedules no refresh on the reader's client
/// ([`assert_no_fred_refresh_on_reader`]); its connection is not reconnected;
/// the next update is delivered.
/// Before b.19i fred's refresh task sent that `AUTH` itself, through
/// `Blocking::Interrupt` — the path
/// [`a_command_queued_behind_a_cluster_readers_blocked_read_interrupts_it_and_the_read_is_replanned`]
/// shows.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cluster_readers_due_reauth_wakes_its_read_and_auths_between_reads() {
    use live_edit::observability::metrics::{
        REAUTH_RESULT_FAILED, REAUTH_RESULT_OK, WAKE_OUTCOME_OK,
    };

    let facade = ClusterFacade::start().await;
    let provider = ScriptedProvider::new("b19i-cluster-reader-password");
    let tv = facade.test_valkey_with_provider(1, provider.clone()).await;
    provider.start_ticking();
    let metrics = install_metrics(&tv).await;
    let client = tv.client();
    let oid = facade.object_on(&tv, Node::A, 0, &[]);
    let c = subscribe(&tv, oid).await;
    xadd_update(&tv, oid, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    let reader = client
        .stream_mux
        .reader_client_for_test(0)
        .expect("connected");
    assert_eq!(reader.active_connections(), vec![facade.server(Node::A)]);
    await_reader_blocked_in_xread(client.pool.next(), &reader).await;
    let auths_before = facade.auths();
    let unblocks_before = facade.client_unblocks();
    let wakes_before = wakes(&metrics, WAKE_OUTCOME_OK);

    provider.grant(1);
    let deadline = Instant::now() + SETTLE;
    await_until(
        "the granted credential's AUTH must be answered well inside the 30 s BLOCK",
        deadline,
        || reauths(&metrics, REAUTH_RESULT_OK) == 1,
    )
    .await;
    let ticks = client.stream_mux.reauth_for_test(0).ticks;
    let deadline = Instant::now() + SETTLE;
    await_until(
        "the reader must take one more tick, with nothing pending",
        deadline,
        || {
            let snap = client.stream_mux.reauth_for_test(0);
            snap.ticks > ticks && !snap.pending
        },
    )
    .await;

    assert_eq!(reauths(&metrics, REAUTH_RESULT_FAILED), 0);
    assert_eq!(
        facade.auth_args_since(auths_before),
        vec![vec![
            "default".to_owned(),
            "b19i-cluster-reader-password".to_owned()
        ]],
        "exactly the reader's one AUTH crossed the facade — no backchannel handshake, no \
         cluster-wide broadcast — carrying the credential its tick fetched"
    );
    assert_eq!(wakes(&metrics, WAKE_OUTCOME_OK) - wakes_before, 1);
    assert_eq!(
        facade.client_unblocks(),
        unblocks_before,
        "no CLIENT UNBLOCK on either node"
    );
    assert_eq!(read_unblocked_retries(&metrics), 0);
    assert_no_fred_refresh_on_reader(&reader, &provider);
    assert_eq!(
        reader_reconnects(&metrics),
        0,
        "the re-AUTH reconnected nothing"
    );
    xadd_update(&tv, oid, 2).await;
    assert_eq!(recv_seqs(&c, 1).await, [2], "nothing lost across the wake");
    assert!(c.take_terminal_exit().is_none());
}

/// b.zt3 T8, inverting b.19i's pin of stock fred 10.1.0: what a *cluster*
/// client's credential refresh does to a node connection with a long command
/// in flight, now that the vendored fred bounds every router wait on one
/// connection's I/O by the slow bound (`UnresponsiveConfig::max_timeout`) and
/// measures a stall from the oldest outstanding command rather than from a
/// `last_write` stamp. This is the pooled clients' path: a pool runs fred's
/// refresh as `_AuthAllCluster`, which drains every connection before it
/// broadcasts.
///
/// A raw fred cluster client through the two-node facade, default
/// `Blocking::Block`, a 2 s slow bound, an `XREAD BLOCK` in flight on node A,
/// and an `AUTH` queued behind it:
///
/// 1. **A drain that ends inside the bound flags nothing.** While the read
///    blocks, a probe for B is held behind the `AUTH` (the router is in the
///    drain). The read is then answered by an `XADD` from another client, and
///    the broadcast reaches both nodes. With the client idle for longer than
///    the slow bound afterwards, no connection is declared unresponsive and
///    both keep their sockets. Stock fred's drain left A's `last_write` stale,
///    which b.19i's pin showed declaring A unresponsive and rebuilding it.
/// 2. **A drain that outlasts the bound fails at it.** The read's BLOCK
///    outlasts every deadline in this test, so only the bound can end the
///    drain. The `AUTH` fails with the slow-verdict error and its broadcast is
///    never written. The read fails instead of waiting out its BLOCK, and A's
///    blocked server connection is closed. A command for B queued behind the
///    broadcast is answered, and the rebuilt client serves A again. Stock
///    fred's drain waited out the BLOCK, holding the router (and B with it)
///    that long.
///
/// The `AUTH`s and the reads run with one attempt, so in phase 2 the drain's
/// error reaches their callers instead of a re-send.
///
/// Waits: phase 1's quiet window (the slow bound plus 1 s) is waited out by
/// construction, and phase 2's drain takes the slow bound; everything else is a
/// deadline. The healthy path takes about six seconds. Worst case: the
/// facade's fixture and the raw client connect (30 s each), the `XADD` on B
/// (5 s); phase 1's blocked poll (10 s plus its last 5 s `CLIENT LIST`), probe
/// (0.25 s), `XADD` (5 s), `AUTH` (7 s), read (5 s), quiet window (3 s) and
/// `XLEN` (5 s); phase 2's blocked poll (15 s), `AUTH` (7 s; the command for B,
/// sent with it, ends inside its own 7 s), read (5 s), close poll (10 s plus
/// 5 s) and `XLEN` (5 s); `quit()` (10 s) and the drop-time purge (10 s):
/// 172.25 s, under the module's 180 s kill deadline.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cluster_clients_auth_broadcast_behind_a_blocked_read_fails_at_the_slow_bound_and_a_drain_inside_it_flags_no_connection(
) {
    use fred::interfaces::{AuthInterface, StreamsInterface};
    use fred::live_edit::liveness::{verdict_of, Verdict};
    use fred::prelude::{Builder, ReconnectPolicy};
    use fred::types::config::{Config as FredConfig, Options, UnresponsiveConfig};
    use fred::types::Value;

    /// The slow bound: fred's unresponsive bound and, with b.zt3, the bound on
    /// every router wait on one connection's I/O, the drain's included.
    const MAX_TIMEOUT: Duration = Duration::from_millis(2_000);
    /// fred's unresponsive scan interval.
    const INTERVAL: Duration = Duration::from_millis(100);
    /// Each read's server-side BLOCK: far past every deadline here, so only an
    /// answer (phase 1) or the bound (phase 2) ends a drain behind it.
    const BLOCK_MS: u64 = 60_000;
    /// How long phase 1's probe for B waits. A free router answers it at once;
    /// one held in the drain cannot.
    const PROBE_TIMEOUT: Duration = Duration::from_millis(250);
    /// Phase 1's quiet window after the drain: the slow bound plus ten scans,
    /// past where stock fred's stale-`last_write` verdict fell (the slow bound
    /// after the read was written).
    const QUIET: Duration = MAX_TIMEOUT.saturating_add(Duration::from_secs(1));
    /// The deadline on a one-shot command, and on a read already ended.
    const ONE_SHOT: Duration = Duration::from_secs(5);
    /// The deadline on anything that waits out the slow bound.
    const BOUNDED: Duration = MAX_TIMEOUT.saturating_add(ONE_SHOT);
    const PASSWORD: &str = "b-zt3-pin-any-password";

    let facade = ClusterFacade::start().await;
    // An observer on the store (through the facade's pools), one stream on
    // each node.
    let tv = facade.test_valkey(1).await;
    let oid_a = facade.object_on(&tv, Node::A, 1_000_000, &[]);
    let oid_b = facade.object_on(&tv, Node::B, 1_000_000, &[oid_a]);
    let (key_on_a, key_on_b) = (
        tv.key_schema().layer_stream(oid_a),
        tv.key_schema().layer_stream(oid_b),
    );
    xadd_update(&tv, oid_b, 1).await;

    let mut builder = Builder::from_config(
        FredConfig::from_url(&facade.cluster_url()).expect("the facade URL parses"),
    );
    builder
        .with_connection_config(|c| {
            c.unresponsive = UnresponsiveConfig {
                max_timeout: Some(MAX_TIMEOUT),
                interval: INTERVAL,
            };
        })
        .set_policy(ReconnectPolicy::new_constant(0, 50));
    let pin = builder.build().expect("the raw cluster client builds");
    tokio::time::timeout(CONNECT_BUDGET, pin.init())
        .await
        .expect("the raw cluster client must connect")
        .expect("the raw cluster client connects through the facade");
    assert!(pin.is_clustered());
    let before = pin.connection_ids();
    let (a, b) = (facade.server(Node::A), facade.server(Node::B));
    let a_id = *before.get(&a).expect("a connection to A");
    assert!(before.contains_key(&b), "a connection to B: {before:?}");
    // A verdict surfaces as the connection's read error, broadcast on
    // `error_rx` with the node.
    let mut errors = pin.error_rx();

    let one_attempt = |timeout: Duration| Options {
        timeout: Some(timeout),
        max_attempts: Some(1),
        ..Default::default()
    };
    let read_on_a = || {
        let pin = pin.clone();
        let key = key_on_a.clone();
        let options = one_attempt(Duration::from_millis(BLOCK_MS) + SETTLE);
        tokio::spawn(async move {
            pin.with_options(&options)
                .xread::<Value, _, _>(None, Some(BLOCK_MS), key, "$")
                .await
        })
    };
    // The AUTH's client: its caller waits out the slow bound, and its one
    // attempt takes the drain's error instead of a re-send.
    let auth_client = pin.with_options(&one_attempt(BOUNDED));
    // Whether the server shows A's connection blocked in the XREAD
    // (`cmd=xread`, flag `b`), and the `CLIENT LIST` line it read.
    let observer = tv.client().pool.next();
    let a_blocked_in_xread = || async {
        let listing: String = observer
            .client_list::<String, String>(None, Some(vec![a_id.to_string()]))
            .await
            .expect("CLIENT LIST ID must succeed");
        let blocked = listing.split_whitespace().any(|kv| kv == "cmd=xread")
            && listing
                .split_whitespace()
                .any(|kv| kv.starts_with("flags=") && kv.contains('b'));
        (blocked, listing)
    };
    let await_a_blocked = || async {
        let deadline = Instant::now() + SETTLE;
        loop {
            let (blocked, listing) = a_blocked_in_xread().await;
            if blocked {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "A's connection never blocked in the XREAD; last CLIENT LIST:\n{listing}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };

    // ── Phase 1: the drain ends inside the bound ──────────────────────────
    let read = read_on_a();
    await_a_blocked().await;
    let auths_before = facade.auths();
    // One poll sends the AUTH to the router, so what follows queues behind it.
    let mut auth_ok = std::pin::pin!(auth_client.auth(Some("default".to_owned()), PASSWORD));
    assert!(futures::poll!(auth_ok.as_mut()).is_pending());
    // Queued behind the AUTH, a command for B is held while the router drains:
    // the drain, waiting on A's read, is what holds it.
    let probe = pin
        .with_options(&one_attempt(PROBE_TIMEOUT))
        .xlen::<u64, _>(key_on_b.clone())
        .await;
    let held = probe.expect_err(
        "a command for B must be held behind the AUTH while the drain waits on A's blocked read",
    );
    assert_eq!(
        *held.kind(),
        fred::error::ErrorKind::Timeout,
        "the held probe's caller gives up at its own timeout: {held:?}"
    );
    // Answer the read from another client: the drain takes the reply and the
    // broadcast goes out.
    xadd_update(&tv, oid_a, 1).await;
    tokio::time::timeout(BOUNDED, auth_ok)
        .await
        .expect("the AUTH must be answered once the drain ends")
        .expect(
            "the AUTH succeeds: the drain ended inside the slow bound, so the broadcast was \
             written (a fixture stall past the bound fails it with the slow-verdict error)",
        );
    assert_eq!(
        facade.auth_args_since(auths_before),
        vec![vec!["default".to_owned(), PASSWORD.to_owned()]; 2],
        "the broadcast reached both nodes once the drain ended"
    );
    // The drain took the read's reply before the broadcast went out.
    let read = tokio::time::timeout(ONE_SHOT, read)
        .await
        .expect("the answered read must have ended")
        .expect("the read task must not panic");
    assert!(
        matches!(read, Ok(ref v) if *v != Value::Null),
        "the read was answered by the XADD, not ended by its BLOCK or the bound; got {read:?}"
    );

    // Nothing is written to the client for longer than the slow bound. A
    // connection with nothing outstanding has no stall, so no verdict.
    let flagged = tokio::time::timeout(QUIET, async {
        loop {
            match errors.recv().await {
                Ok((err, server)) if verdict_of(&err).is_some() => return (err, server),
                Ok(_) => continue,
                Err(e) => panic!("the error channel must stay open: {e:?}"),
            }
        }
    })
    .await;
    assert!(
        flagged.is_err(),
        "no connection may be declared unresponsive after a drain that ended inside the bound \
         (stock fred's stale last_write flagged A here); got {flagged:?}"
    );
    assert_eq!(
        pin.connection_ids(),
        before,
        "neither node's socket was rebuilt"
    );
    let on_a: u64 = tokio::time::timeout(ONE_SHOT, pin.xlen(key_on_a.clone()))
        .await
        .expect("XLEN on A must settle")
        .expect("A keeps serving on its original connection");
    assert_eq!(on_a, 1);
    assert_eq!(pin.connection_ids(), before);

    // ── Phase 2: the drain outlasts the bound ─────────────────────────────
    let read = read_on_a();
    await_a_blocked().await;
    let auths_before = facade.auths();
    let mut auth_failed = std::pin::pin!(auth_client.auth(Some("default".to_owned()), PASSWORD));
    assert!(futures::poll!(auth_failed.as_mut()).is_pending());
    // A command for B, queued behind the broadcast.
    let served_b = {
        let pin = pin.clone();
        let key = key_on_b.clone();
        tokio::spawn(async move {
            pin.with_options(&Options {
                timeout: Some(BOUNDED),
                ..Default::default()
            })
            .xlen::<u64, _>(key)
            .await
        })
    };
    let err = tokio::time::timeout(BOUNDED, auth_failed)
        .await
        .expect(
            "the AUTH must end at the slow bound: its drain behind A's read is bounded, not \
             waited out to the read's BLOCK",
        )
        .expect_err("the AUTH fails: the drain's wait on A expired before the broadcast");
    assert_eq!(
        verdict_of(&err),
        Some(Verdict::Slow),
        "an expired drain fails with the slow-verdict error: {err:?}"
    );
    assert_eq!(
        facade.auth_args_since(auths_before),
        Vec::<Vec<String>>::new(),
        "the broadcast was never written"
    );
    let on_b = tokio::time::timeout(BOUNDED, served_b)
        .await
        .expect("the command for B must be answered once the drain fails")
        .expect("the B task must not panic")
        .expect("B keeps serving: its command, queued behind the broadcast, is answered");
    assert_eq!(on_b, 1, "B's stream holds its one entry");
    // The rebuild re-sends the read ahead of the command for B, so it has
    // ended by now.
    let read = tokio::time::timeout(ONE_SHOT, read)
        .await
        .expect("the read must end with the failed drain, not wait out its BLOCK")
        .expect("the read task must not panic");
    // The failed connection's in-flight read is handed back for a re-send,
    // which its one attempt refuses: it was neither answered nor ended by its
    // BLOCK.
    assert!(
        matches!(read, Err(ref e) if e.details() == "Too many failed write attempts."),
        "the read fails with its connection, refused a re-send; got {read:?}"
    );
    await_connection_closed(
        observer,
        a_id,
        "A's blocked connection, failed at the bound",
    )
    .await;
    let on_a: u64 = tokio::time::timeout(ONE_SHOT, pin.xlen(key_on_a.clone()))
        .await
        .expect("XLEN on A must settle")
        .expect("the rebuilt client serves A");
    assert_eq!(on_a, 1);
    assert!(
        pin.connection_ids().get(&a).is_some_and(|id| *id != a_id),
        "A's socket is a new one: {:?}",
        pin.connection_ids()
    );
    let _ = tokio::time::timeout(SETTLE, pin.quit()).await;
}

/// T13: an external `CLIENT UNBLOCK <id> ERROR` on a cluster reader's one
/// connection fails its `XREAD` with `-UNBLOCKED`; the reader re-plans from
/// its cursors and the next update is delivered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_unblock_on_a_single_node_cluster_reader_is_replanned_without_loss() {
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let oid = facade.object_on(&tv, Node::B, 0, &[]);
    let c = subscribe(&tv, oid).await;
    xadd_update(&tv, oid, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    let reader = tv
        .client()
        .stream_mux
        .reader_client_for_test(0)
        .expect("connected");
    let id = sole_connection_id(&reader);

    let deadline = Instant::now() + SETTLE;
    loop {
        let unblocked: i64 = tv
            .client()
            .pool
            .next()
            .client_unblock(id, Some(ClientUnblockFlag::Error))
            .await
            .expect("CLIENT UNBLOCK must succeed");
        if unblocked == 1 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the reader's connection {id} never blocked"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    xadd_update(&tv, oid, 2).await;
    assert_eq!(recv_seqs(&c, 1).await, [2]);
    assert!(read_unblocked_retries(&metrics) >= 1);
    assert!(c.take_terminal_exit().is_none());
}

/// T12: an idle, connected cluster reader's keepalive tick is one `PING` on
/// its one connection, so it consumes exactly one `Keepalive` fault. With
/// `[Pass, Err]` queued, the first tick answers (`ok`) and the second fails;
/// a tick that pinged every primary of this two-node facade — as a cluster
/// reader did before b.azh — would take both faults in its first tick and
/// count `failed` with no `ok` before it.
///
/// Needs `test-fault-injection` (the fault sites and knobs); compiled out
/// without it.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_idle_cluster_readers_keepalive_consumes_one_fault_per_tick() {
    use live_edit::observability::metrics::{KEEPALIVE_RESULT_FAILED, KEEPALIVE_RESULT_OK};
    use live_edit::storage::stream_mux::{MuxFault, MuxFaultSite, StreamMuxKnobs};

    const KEEPALIVES: &str = "live_edit_stream_mux_keepalives_total";
    let keepalives =
        |metrics: &Metrics, result: &str| read_series(metrics, KEEPALIVES, &[("result", result)]);

    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let mux = &tv.client().stream_mux;
    // A short BLOCK, so the reader goes idle soon after its subscription
    // leaves; the keepalive stays at its production interval until the faults
    // are queued.
    mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: 100,
        ..mux.knobs()
    });
    let oid = facade.object_on(&tv, Node::A, 0, &[]);
    serve_once(&tv, oid, 1).await;
    await_mux(tv.client(), "the subscription left", |s| {
        s.subscriptions() == 0
    })
    .await;

    mux.push_faults_for_test(
        MuxFaultSite::Keepalive,
        [
            MuxFault::Pass,
            MuxFault::Err(fred::error::Error::new(
                fred::error::ErrorKind::IO,
                "synthetic connection reset",
            )),
        ],
    );
    // Ticks far enough apart that the tick after the failed one — on the
    // reconnected connection — cannot answer before `ok` is read below: with
    // two PINGs per tick, the first tick takes both faults, and only a later
    // answered tick could make `ok >= 1` hold.
    mux.set_knobs_for_test(StreamMuxKnobs {
        keepalive_interval: Some(Duration::from_millis(400)),
        probe_timeout: Duration::from_millis(100),
        ..mux.knobs()
    });

    let deadline = Instant::now() + SETTLE;
    await_until("the second fault's tick must fail", deadline, || {
        keepalives(&metrics, KEEPALIVE_RESULT_FAILED) == 1
    })
    .await;
    assert!(
        keepalives(&metrics, KEEPALIVE_RESULT_OK) >= 1,
        "the first tick took only the Pass and answered: one PING per tick"
    );
    assert_eq!(mux.pending_faults_for_test(MuxFaultSite::Keepalive), 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// The two other outcomes of a `-MOVED` (`slot_moved`)
// ═════════════════════════════════════════════════════════════════════════════

/// A `-MOVED` on the reader's `XREAD` whose slot has NOT moved — the facade
/// keeps the slot on A. The reader's routing view names A, the reader's own
/// node, so it re-syncs the view once (`CLUSTER SLOTS` reaches the wire),
/// finds A still the owner, and treats the redirect as an ordinary transient
/// failure: it backs off and reads again from A. No retarget, and nothing lost.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_moved_xread_whose_owner_has_not_moved_backs_off_and_reads_the_same_node_again() {
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let oid = facade.object_on(&tv, Node::A, 0, &[]);
    let c = subscribe(&tv, oid).await;
    xadd_update(&tv, oid, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);

    let syncs = facade.slots_served();
    let reader_syncs = tv.client().stream_mux.routing_syncs_finished_for_test();
    facade.trap_next_xread_on_a(tv.key_schema().layer_stream(oid));
    xadd_update(&tv, oid, 2).await;
    assert_eq!(recv_seqs(&c, 1).await, [2]);
    let deadline = Instant::now() + SETTLE;
    await_until("the -MOVED must reach the reader", deadline, || {
        facade.moved_fired.load(Ordering::SeqCst) == 1
    })
    .await;

    xadd_update(&tv, oid, 3).await;
    assert_eq!(recv_seqs(&c, 1).await, [3], "nothing is lost");
    assert!(c.take_terminal_exit().is_none());
    assert!(
        facade.slots_served() > syncs,
        "the view named the reader's own node, so the reader re-synced it before deciding"
    );
    assert_eq!(
        tv.client().stream_mux.routing_syncs_finished_for_test() - reader_syncs,
        1,
        "one readers' routing sync for the one -MOVED"
    );
    assert_eq!(
        reader_connect_row(&metrics),
        [1, 0, 0, 0],
        "the owner did not move: no retarget"
    );
    assert_eq!(
        tv.client().stream_mux.snapshot().readers[0].node,
        Some(facade.name(Node::A)),
        "the reader reads from A again"
    );
}

/// A `-MOVED` on the reader's `XREAD` when the readers' routing view already
/// names the slot's new owner (another client synced it): the reader retargets
/// at once, without a `CLUSTER SLOTS` sync of its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_moved_xread_retargets_without_a_sync_when_the_view_already_names_the_new_owner() {
    let facade = ClusterFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let metrics = install_metrics(&tv).await;
    let oid = facade.object_on(&tv, Node::A, 0, &[]);
    let c = subscribe(&tv, oid).await;
    xadd_update(&tv, oid, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);

    facade.move_to_b_from(stream_slot(&tv, oid));
    sync_readers_view(&tv).await;
    let syncs = facade.slots_served();
    let reader_syncs = tv.client().stream_mux.routing_syncs_finished_for_test();
    facade.trap_next_xread_on_a(tv.key_schema().layer_stream(oid));
    xadd_update(&tv, oid, 2).await;
    assert_eq!(recv_seqs(&c, 1).await, [2]);
    let deadline = Instant::now() + SETTLE;
    await_until("the -MOVED must reach the reader", deadline, || {
        facade.moved_fired.load(Ordering::SeqCst) == 1
    })
    .await;

    xadd_update(&tv, oid, 3).await;
    assert_eq!(recv_seqs(&c, 1).await, [3], "nothing is lost");
    assert!(c.take_terminal_exit().is_none());
    assert_eq!(reader_connect_row(&metrics), [1, 0, 1, 0], "one retarget");
    assert_eq!(
        tv.client().stream_mux.snapshot().readers[0].node,
        Some(facade.name(Node::B))
    );
    assert_eq!(
        facade.slots_served(),
        syncs,
        "the view already named B: the reader retargeted without a sync"
    );
    assert_eq!(
        tv.client().stream_mux.routing_syncs_finished_for_test(),
        reader_syncs,
        "and the readers' connector issued none"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Needs `test-fault-injection` (fault sites and knobs)
// ═════════════════════════════════════════════════════════════════════════════

/// The cases that drive the multiplexer's fault sites or knobs; compiled out
/// without `test-fault-injection`.
#[cfg(feature = "test-fault-injection")]
mod faults {
    use super::*;
    use fred::types::{CustomCommand, Value};
    use live_edit::observability::metrics::{
        CONSUMER_PHASE_KEEPALIVE, CONSUMER_PHASE_XREAD, KEEPALIVE_RESULT_FAILED,
    };
    use live_edit::storage::stream_mux::{MuxFault, MuxFaultSite, StreamMuxKnobs};

    use super::super::mux_test_helpers::{
        fast_backoff_knobs, forced_reconnect_row, forced_reconnects_only_on, push_timeout_faults,
        QUIET, STREAMS,
    };

    /// The WARN a reader logs when a retired connection's `QUIT` overran.
    const QUIT_OVERRAN_MSG: &str = "a retired stream reader connection did not answer its QUIT \
                                    in time; its router was aborted";

    /// A real-shaped `-MOVED` for `slot`, pointing at B.
    fn moved(facade: &ClusterFacade, slot: u16) -> MuxFault {
        MuxFault::Err(fred::error::Error::new(
            fred::error::ErrorKind::Cluster,
            format!("MOVED {slot} {}", facade.name(Node::B)),
        ))
    }

    // ── A cancelled, held or failed reader connect (`handle_connect_failure`) ─
    //
    // Staged with the `MuxFaultSite::Connect` seam (module docs, "A failed
    // reader connect"), on a one-reader client each test owns: a connect
    // cancelled by `quit()`, one held and then completed while its
    // subscription leaves, and failed ones the reader classifies. Its reader
    // holds no connection until the test's first subscription, so the first
    // fault a test pushes is taken by that subscription's connect.

    /// The WARN a reader logs for a transient connect failure it retries.
    const CONNECT_RETRIED_MSG: &str = "stream reader could not connect to the backing store; its \
                                       subscriptions are kept and the connect is retried after a \
                                       backoff";

    /// The WARN the multiplexer's shutdown logs when a reader task outlives
    /// its grace.
    const READERS_OUTLIVED_GRACE_MSG: &str =
        "stream multiplexer readers did not stop within the shutdown grace";

    /// The DEBUG line the multiplexer's shutdown logs once every reader task
    /// has stopped inside its grace.
    const READERS_STOPPED_MSG: &str = "woke blocked stream reads before the reader pool QUIT";

    /// The details of [`dropped_setup`]'s error.
    const DROPPED_SETUP: &str = "connection closed during setup";

    /// The server's verbatim reply refusing a credential it has and rejects.
    const WRONGPASS_REPLY: &str = "WRONGPASS invalid username-password pair or user is disabled.";

    /// The server's verbatim reply refusing a connection that has not
    /// authenticated.
    const NOAUTH_REPLY: &str = "NOAUTH Authentication required.";

    /// A connect whose new connection the store dropped mid-setup: an `IO`
    /// error, which a reader treats as transient.
    fn dropped_setup() -> MuxFault {
        MuxFault::Err(fred::error::Error::new(
            fred::error::ErrorKind::IO,
            DROPPED_SETUP,
        ))
    }

    /// A connect whose setup the store refused with `reply`, in the shape fred
    /// gives a server's `-WRONGPASS` or `-NOAUTH`: an `Auth` error whose
    /// details are the reply. A reader treats it as permanent.
    fn refused_setup(reply: &'static str) -> MuxFault {
        MuxFault::Err(fred::error::Error::new(fred::error::ErrorKind::Auth, reply))
    }

    /// The vacuity guard: no `Connect` fault is left queued on `client`, so a
    /// reader connect took each one and the outcome asserted is the staged
    /// one's.
    fn assert_connect_faults_taken(client: &ValkeyClient, what: &str) {
        assert_eq!(
            client
                .stream_mux
                .pending_faults_for_test(MuxFaultSite::Connect),
            0,
            "{what}: a reader connect must take every queued Connect fault"
        );
    }

    /// Fresh metrics, installed on `client`'s hooks.
    async fn install_own_metrics(client: &ValkeyClient) -> Arc<Metrics> {
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        metrics
    }

    /// XADDs a well-formed `update` entry `{seq}-0` through `client`.
    async fn xadd_on(client: &ValkeyClient, schema: &KeySchema, object_id: u64, seq: u64) {
        use fred::interfaces::StreamsInterface;
        let _: String = client
            .pool
            .xadd(
                schema.layer_stream(object_id),
                false,
                None,
                format!("{seq}-0").as_str(),
                vec![("payload", format!("update|{object_id}|{seq}|probe|0|0"))],
            )
            .await
            .expect("XADD must succeed");
    }

    /// Deletes `object_id`'s layer stream and quits `client`.
    async fn drop_stream_and_quit(client: ValkeyClient, schema: &KeySchema, object_id: u64) {
        use fred::interfaces::KeysInterface;
        let _: i64 = client
            .pool
            .del(schema.layer_stream(object_id))
            .await
            .expect("DEL must succeed");
        tokio::time::timeout(SETTLE, client.quit())
            .await
            .expect("quit must settle");
    }

    /// The subscription ended with a retryable `Unavailable`; its message.
    async fn ended_unavailable(consumer: &LayerStreamConsumer) -> String {
        assert!(
            tokio::time::timeout(SETTLE, consumer.recv())
                .await
                .expect("the subscription must end")
                .is_none(),
            "the subscription must end, not deliver"
        );
        let exit = consumer
            .take_terminal_exit()
            .expect("a reader-side ending is never a clean close");
        assert_eq!(
            exit.reason,
            live_edit::observability::metrics::STREAM_EXIT_ERROR
        );
        match exit.error {
            live_edit::error::ServiceError::Unavailable(message) => message,
            other => {
                panic!("a failed reader connect ends its subscriptions UNAVAILABLE; got {other:?}")
            }
        }
    }

    /// The steady-state credential hint lines (`phase="steady-state"`)
    /// captured.
    fn steady_state_hints(
        captured: &crate::grpc::log_capture::Captured,
    ) -> Vec<&serde_json::Value> {
        captured
            .events
            .iter()
            .filter(|e| e["fields"]["phase"] == "steady-state")
            .collect()
    }

    /// T9: a reader connect still in progress when `quit()` runs is
    /// cancelled: its task stops inside the multiplexer's shutdown grace, the
    /// connect is counted neither `ok` nor `failed`, and no router watcher is
    /// left. The connect is held by a `Hang` — its setup never reports back —
    /// and its connect bound is raised far past this test's waits, so nothing
    /// but the cancellation ends it while the test runs. Without the
    /// cancellation the reader task outlives the grace: the shutdown then
    /// logs [`READERS_OUTLIVED_GRACE_MSG`] instead of [`READERS_STOPPED_MSG`].
    ///
    /// That the cancelled connect also aborts the router it started, closing
    /// its connection, is the socket half the seam cannot show; it is pinned
    /// by `a_reader_connect_cancelled_mid_setup_closes_its_connection` in
    /// `src/storage/client.rs` (this file's module docs, "A failed reader
    /// connect").
    ///
    /// Current-thread, so the DEBUG [`ThreadCapture`] sees the shutdown's
    /// lines.
    #[tokio::test]
    async fn quit_while_a_reader_is_connecting_cancels_the_connect_and_counts_it_neither_ok_nor_failed(
    ) {
        let capture = ThreadCapture::start_debug();
        let mut config: Config = crate::common::test_config();
        config.subscriber_pool_size = 1;
        // The reader's connect bound: far past every wait below.
        config.valkey_connect_timeout_secs = 120;
        let client = own_client_with(config).await;
        let metrics = install_own_metrics(&client).await;

        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Connect, [MuxFault::Hang]);
        let sub = subscribe_on(&client, &fresh_schema(), 1).await;
        let deadline = Instant::now() + SETTLE;
        await_until("the reader's connect must take the Hang", deadline, || {
            client
                .stream_mux
                .pending_faults_for_test(MuxFaultSite::Connect)
                == 0
        })
        .await;
        assert_eq!(
            client.stream_mux.connected_readers(),
            0,
            "precondition: the reader is mid-connect"
        );

        drop(sub);
        tokio::time::timeout(SETTLE, client.quit())
            .await
            .expect("quit must settle");

        let captured = capture.snapshot();
        assert!(
            captured.with_message(READERS_OUTLIVED_GRACE_MSG).is_empty(),
            "quit must cancel the connect, so the reader task stops inside the shutdown grace; \
             full capture:\n{}",
            captured.text
        );
        let stopped = captured.only(READERS_STOPPED_MSG);
        assert_field(stopped, "readers", 1.into());
        let deadline = Instant::now() + SETTLE;
        await_until("no router watcher may be left", deadline, || {
            client.router_watchers_for_test() == 0
        })
        .await;
        assert!(client.reader_watches_for_test().is_empty());
        assert_eq!(
            reader_connect_row(&metrics),
            [0; 4],
            "a cancelled connect is neither an ok nor a failed connect"
        );
        assert_eq!(readers_connected(&metrics), 0);
    }

    /// `live_edit_valkey_pool_in_use{pool}`.
    const POOL_IN_USE: &str = "live_edit_valkey_pool_in_use";

    /// The reader's two gauges, as `(live_edit_stream_mux_streams,
    /// live_edit_valkey_pool_in_use{pool="subscriber_pool"})`.
    fn reader_gauges(metrics: &Metrics) -> (i64, i64) {
        (
            read_series(metrics, STREAMS, &[]),
            read_series(metrics, POOL_IN_USE, &[("pool", READER_POOL)]),
        )
    }

    /// b.9y2: a subscription that leaves while its reader is connecting
    /// brings both reader gauges, `live_edit_stream_mux_streams` and
    /// `live_edit_valkey_pool_in_use{pool="subscriber_pool"}`, back to 0.
    ///
    /// The subscription's `Add` reaches the reader while it is idle, so the
    /// gauges read 1 when it starts its connect. The connect is held by a
    /// `Hold` (its connect bound raised far past this test's waits, so the
    /// hold cannot be cut), the subscription is dropped meanwhile — its
    /// `Remove` queues behind the connect — and the hold is released. The
    /// connect completes, the probe runs, and the reader, with no read in
    /// flight, applies the `Remove` in its idle branch's drain and waits.
    /// The gauges must be published after that drain; a reader that published
    /// them only at the top of its loop waits with them stuck at 1.
    ///
    /// The keepalive is turned off before the subscription, so nothing wakes
    /// the idle reader to republish them: a keepalive tick, or a knob change
    /// made later, would send it round its loop and hide the stale gauges.
    /// The self-checks before the drop prove the `Add` was applied (and
    /// published) before the connect, so the test cannot pass without the
    /// race it exists for; those after the wait prove the `Remove` took the
    /// idle drain (no read was ever issued) and the held connect completed
    /// rather than being cut.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_subscription_that_leaves_while_its_reader_connects_brings_the_reader_gauges_to_zero()
    {
        let mut config: Config = crate::common::test_config();
        config.subscriber_pool_size = 1;
        // The reader's connect bound: far past every wait below.
        config.valkey_connect_timeout_secs = 120;
        let client = own_client_with(config).await;
        let metrics = install_own_metrics(&client).await;
        client.stream_mux.set_knobs_for_test(StreamMuxKnobs {
            keepalive_interval: None,
            ..client.stream_mux.knobs()
        });
        let (release, held) = tokio::sync::oneshot::channel::<()>();
        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Connect, [MuxFault::Hold(held)]);

        let sub = subscribe_on(&client, &fresh_schema(), 1).await;
        let deadline = Instant::now() + SETTLE;
        await_until("the reader's connect must take the Hold", deadline, || {
            client
                .stream_mux
                .pending_faults_for_test(MuxFaultSite::Connect)
                == 0
        })
        .await;
        assert_eq!(
            client.stream_mux.connected_readers(),
            0,
            "precondition: the reader is mid-connect"
        );
        assert_eq!(
            reader_gauges(&metrics),
            (1, 1),
            "precondition: the subscription's stream was applied and published before the connect"
        );

        drop(sub);
        release
            .send(())
            .expect("the held connect must still be waiting for its release");

        let deadline = Instant::now() + SETTLE;
        loop {
            let gauges = reader_gauges(&metrics);
            if gauges == (0, 0) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the reader's gauges must return to 0 once its only subscription has left; \
                 (streams, subscriber_pool in use) stuck at {gauges:?}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        assert_eq!(
            client.stream_mux.snapshot().xreads(),
            0,
            "the Remove was applied by the idle drain, not by a read's control arm"
        );
        assert_eq!(
            client.stream_mux.connected_readers(),
            1,
            "the held connect completed rather than being cut"
        );
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 0, 0, 0],
            "the held connect counts once, bind/ok"
        );

        tokio::time::timeout(SETTLE, client.quit())
            .await
            .expect("quit must settle");
    }

    /// A transient connect failure — the store drops the reader's new
    /// connection mid-setup, an `IO` failure — keeps the reader's
    /// subscription: one WARN naming the cause and the failure, the budget
    /// charge (the first consecutive failure) and the backoff window drawn,
    /// and the retried connect, which takes no fault, serves it. Counted
    /// `cause="bind", result="failed"` then `result="ok"`, and the connected
    /// gauge moved only for the connect that succeeded.
    ///
    /// Current-thread, so the [`ThreadCapture`] sees the reader task's WARN.
    #[tokio::test]
    async fn a_transient_reader_connect_failure_keeps_the_subscription_and_the_retry_serves_it() {
        const OID: u64 = 11;
        let capture = ThreadCapture::start();
        let client = own_client(1).await;
        let metrics = install_own_metrics(&client).await;
        let schema = fresh_schema();

        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Connect, [dropped_setup()]);
        let c = subscribe_on(&client, &schema, OID).await;
        xadd_on(&client, &schema, OID, 1).await;
        assert_eq!(recv_seqs(&c, 1).await, [1], "the retried connect serves it");
        assert!(c.take_terminal_exit().is_none());
        assert_connect_faults_taken(&client, "the reader's first connect");

        assert_eq!(
            reader_connect_row(&metrics),
            [1, 1, 0, 0],
            "one failed connect, then one that succeeded"
        );
        assert_eq!(
            readers_connected(&metrics),
            1,
            "the gauge counts the connect that succeeded, not the one that failed"
        );
        let captured = capture.snapshot();
        let warn = captured.only(CONNECT_RETRIED_MSG);
        assert_eq!(warn["level"], "WARN");
        assert_field(warn, "cause", READER_CONNECT_CAUSE_BIND.into());
        assert_field(warn, "kind", fred::error::ErrorKind::IO.to_str().into());
        assert!(
            warn["fields"]["error"]
                .as_str()
                .is_some_and(|error| error.contains(DROPPED_SETUP)),
            "the WARN names the connect's failure; got {warn}"
        );
        assert_field(warn, "consecutive_failures", 1.into());
        assert!(
            warn["fields"]["backoff_window_ms"]
                .as_u64()
                .is_some_and(|window| window > 0),
            "the retry is backed off: the WARN names a non-empty backoff window; got {warn}"
        );

        drop(c);
        drop(capture);
        drop_stream_and_quit(client, &schema, OID).await;
    }

    /// A connect the store refuses with an error no retry fixes — a
    /// `WRONGPASS` setup, an `Auth` failure — ends the reader's subscriptions
    /// with a retryable `UNAVAILABLE` naming the kind, and is counted
    /// `result="failed"` with the gauge untouched. The client's re-subscribe
    /// connects again, taking no fault, and is served.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_permanent_reader_connect_failure_ends_the_subscriptions_and_a_resubscribe_connects_again(
    ) {
        const OID: u64 = 12;
        let client = own_client(1).await;
        let metrics = install_own_metrics(&client).await;
        let schema = fresh_schema();

        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Connect, [refused_setup(WRONGPASS_REPLY)]);
        let c = subscribe_on(&client, &schema, OID).await;
        let message = ended_unavailable(&c).await;
        assert_connect_faults_taken(&client, "the reader's first connect");
        let auth = fred::error::ErrorKind::Auth.to_str();
        assert!(
            message.contains(&format!(
                "could not connect to the configured endpoint (kind: {auth})"
            )),
            "the ending names the connect failure and its kind; got {message:?}"
        );
        assert_eq!(reader_connect_row(&metrics), [0, 1, 0, 0]);
        assert_eq!(
            readers_connected(&metrics),
            0,
            "a failed connect moves no gauge"
        );
        assert_eq!(client.stream_mux.connected_readers(), 0);

        let again = subscribe_on(&client, &schema, OID).await;
        xadd_on(&client, &schema, OID, 1).await;
        assert_eq!(
            recv_seqs(&again, 1).await,
            [1],
            "the re-subscribe connects and is served"
        );
        assert_eq!(reader_connect_row(&metrics), [1, 1, 0, 0]);
        assert_eq!(readers_connected(&metrics), 1);

        drop((c, again));
        drop_stream_and_quit(client, &schema, OID).await;
    }

    /// `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` in
    /// [`a_reader_connect_stalled_mid_setup_is_cut_at_its_connect_bound_and_retried`]:
    /// short, so the cut comes soon, and long enough for the client's own
    /// startup phases, which share the setting.
    const READER_CONNECT_BOUND_SECS: u64 = 2;

    /// A reader connect whose setup never reports back (a `Hang`) is cut at
    /// the reader's own connect bound (`LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`,
    /// here [`READER_CONNECT_BOUND_SECS`]): the transient WARN reports the
    /// bound's `Timeout`, naming the setting. Nothing else ends this connect —
    /// under the `Hang` the real connection's setup completes, so fred's own
    /// setup bound never fires. Counted `cause="bind", result="failed"`; the
    /// retry takes no fault, connects and serves the subscription
    /// (`result="ok"`).
    ///
    /// No duration is asserted: which bound fired is read from the error.
    /// That the cut also aborts the connect's router, closing its connection,
    /// is the socket half the seam cannot show; it is pinned by
    /// `a_reader_connect_cut_at_its_bound_mid_setup_closes_its_connection` in
    /// `src/storage/client.rs` (this file's module docs, "A failed reader
    /// connect").
    ///
    /// Current-thread, so the [`ThreadCapture`] sees the reader task's WARN.
    #[tokio::test]
    async fn a_reader_connect_stalled_mid_setup_is_cut_at_its_connect_bound_and_retried() {
        const OID: u64 = 14;
        let capture = ThreadCapture::start();
        let mut config: Config = crate::common::test_config();
        config.subscriber_pool_size = 1;
        config.valkey_connect_timeout_secs = READER_CONNECT_BOUND_SECS;
        let client = own_client_with(config).await;
        let metrics = install_own_metrics(&client).await;
        let schema = fresh_schema();

        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Connect, [MuxFault::Hang]);
        let c = subscribe_on(&client, &schema, OID).await;
        let deadline = Instant::now() + SETTLE;
        await_until("the cut connect must log its WARN", deadline, || {
            !capture
                .snapshot()
                .with_message(CONNECT_RETRIED_MSG)
                .is_empty()
        })
        .await;
        assert_connect_faults_taken(&client, "the reader's first connect");
        let captured = capture.snapshot();
        let warn = captured.only(CONNECT_RETRIED_MSG);
        assert_eq!(warn["level"], "WARN");
        assert_field(warn, "cause", READER_CONNECT_CAUSE_BIND.into());
        assert_field(
            warn,
            "kind",
            fred::error::ErrorKind::Timeout.to_str().into(),
        );
        let error = warn["fields"]["error"].as_str().unwrap_or_default();
        assert!(
            error.contains(&format!(
                "not up within {READER_CONNECT_BOUND_SECS} s \
                 (LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS)"
            )),
            "the connect was cut by the reader's own bound; got {error:?}"
        );

        xadd_on(&client, &schema, OID, 1).await;
        assert_eq!(recv_seqs(&c, 1).await, [1], "the retried connect serves it");
        assert!(c.take_terminal_exit().is_none());
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 1, 0, 0],
            "the cut connect counts bind/failed, the retry bind/ok"
        );
        assert_eq!(readers_connected(&metrics), 1);

        drop(c);
        drop(capture);
        drop_stream_and_quit(client, &schema, OID).await;
    }

    /// A running process's reader connect refused `WRONGPASS` (a credential
    /// the server has and rejects — the misconfiguration the operator
    /// checklist diagnoses) logs the steady-state credential hint exactly
    /// once, naming `pool="subscriber_pool"`: no hook watches a reader
    /// connection that never came up, so the connect path is the only place
    /// this rejection can be explained. A `NOAUTH` — the session reset
    /// `reconnect_on_auth_error` absorbs — logs none. The `NOAUTH` connect
    /// runs first, so the hint gate it would have consumed is still open for
    /// the `WRONGPASS` one.
    ///
    /// Both refusals are seam errors in the shape fred gives a server's reply
    /// (`Auth`, the reply as details). That fred gives a real `-WRONGPASS` or
    /// `-NOAUTH` setup reply that shape, failing the connect at once, is pinned
    /// by the rejection tests in `src/storage/client.rs` (this file's module
    /// docs, "A failed reader connect").
    ///
    /// Current-thread, so the [`ThreadCapture`] sees the reader task's lines.
    #[tokio::test]
    async fn a_reader_connect_refused_wrongpass_logs_one_steady_state_hint_and_noauth_none() {
        const NOAUTH_OID: u64 = 15;
        const WRONGPASS_OID: u64 = 16;
        let capture = ThreadCapture::start();
        let client = own_client(1).await;
        let schema = fresh_schema();

        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Connect, [refused_setup(NOAUTH_REPLY)]);
        let noauth = subscribe_on(&client, &schema, NOAUTH_OID).await;
        let message = ended_unavailable(&noauth).await;
        assert_connect_faults_taken(&client, "the NOAUTH connect");
        assert!(
            message.contains(&format!(
                "(kind: {})",
                fred::error::ErrorKind::Auth.to_str()
            )),
            "precondition: the NOAUTH connect failed as an Auth error; got {message:?}"
        );
        assert!(
            steady_state_hints(&capture.snapshot()).is_empty(),
            "a NOAUTH on a reader connect logs no steady-state hint; full capture:\n{}",
            capture.snapshot().text
        );

        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Connect, [refused_setup(WRONGPASS_REPLY)]);
        let wrongpass = subscribe_on(&client, &schema, WRONGPASS_OID).await;
        ended_unavailable(&wrongpass).await;
        assert_connect_faults_taken(&client, "the WRONGPASS connect");
        let captured = capture.snapshot();
        let hints = steady_state_hints(&captured);
        assert_eq!(
            hints.len(),
            1,
            "one steady-state hint for the WRONGPASS reader connect; full capture:\n{}",
            captured.text
        );
        assert_eq!(hints[0]["level"], "ERROR");
        assert_field(hints[0], "pool", READER_POOL.into());

        drop((noauth, wrongpass));
        drop(capture);
        tokio::time::timeout(SETTLE, client.quit())
            .await
            .expect("quit must settle");
    }

    /// With no transient failure allowed, one transient connect failure
    /// exhausts the budget: the subscription ends `UNAVAILABLE` naming the
    /// connect, counted `result="failed"`, and no connection is held.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_transient_reader_connect_failure_past_the_budget_ends_the_subscriptions() {
        const OID: u64 = 13;
        let client = own_client(1).await;
        let metrics = install_own_metrics(&client).await;
        client.stream_mux.set_knobs_for_test(StreamMuxKnobs {
            max_transient_failures: 0,
            ..client.stream_mux.knobs()
        });
        let schema = fresh_schema();

        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Connect, [dropped_setup()]);
        let c = subscribe_on(&client, &schema, OID).await;
        let message = ended_unavailable(&c).await;
        assert_connect_faults_taken(&client, "the reader's connect");
        assert!(
            message.contains(
                "gave up after 1 consecutive transient failures (last: it could not \
                              connect to the configured endpoint"
            ),
            "the ending names the exhausted budget and the connect; got {message:?}"
        );
        assert_eq!(reader_connect_row(&metrics), [0, 1, 0, 0]);
        assert_eq!(readers_connected(&metrics), 0);
        assert_eq!(client.stream_mux.connected_readers(), 0);

        drop(c);
        drop_stream_and_quit(client, &schema, OID).await;
    }

    // ── A forced reconnect that finds the owner moved (`force_reconnect`) ───

    /// The slot's owner moved (a failover) and the readers' routing view knows
    /// it, but nothing has told the reader: its next XREAD times out (an
    /// injected `Timeout`), the wedge verdict forces a reconnect under
    /// `phase="xread"`, and that reconnect finds the bound slot owned by B —
    /// so it retires the connection to A and retargets (`cause="retarget"`)
    /// instead of rebuilding a connection to the old node. Nothing is lost.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_forced_reconnect_after_the_slots_owner_moved_retargets_instead_of_reconnecting_the_old_node(
    ) {
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        tv.client()
            .stream_mux
            .set_knobs_for_test(fast_backoff_knobs(tv.client()));
        let oid = facade.object_on(&tv, Node::A, 0, &[]);
        let c = subscribe(&tv, oid).await;
        xadd_update(&tv, oid, 1).await;
        assert_eq!(recv_seqs(&c, 1).await, [1]);

        facade.move_to_b_from(stream_slot(&tv, oid));
        sync_readers_view(&tv).await;
        // Taken by the read issued once update 2 returns the one in flight.
        push_timeout_faults(tv.client(), 1);
        xadd_update(&tv, oid, 2).await;
        assert_eq!(recv_seqs(&c, 1).await, [2]);
        xadd_update(&tv, oid, 3).await;
        assert_eq!(
            recv_seqs(&c, 1).await,
            [3],
            "the retargeted reader serves the slot"
        );
        assert!(c.take_terminal_exit().is_none());

        assert_eq!(
            forced_reconnect_row(&metrics),
            forced_reconnects_only_on(CONSUMER_PHASE_XREAD, 1),
            "the verdict is counted whichever remedy follows"
        );
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 0, 1, 0],
            "the forced reconnect retargeted"
        );
        assert_eq!(
            tv.client().stream_mux.snapshot().readers[0].node,
            Some(facade.name(Node::B))
        );
    }

    /// An idle reader whose node stops being a primary — the facade hands
    /// every slot to B, and the readers' view is synced — retires its
    /// connection when its keepalive fails, rather than reconnecting a node
    /// that no longer serves anything; it does not retarget, since it serves
    /// no slot. Its next subscription connects afresh, as a `bind`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_idle_reader_whose_node_is_no_longer_a_primary_retires_without_retargeting() {
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        let mux = &tv.client().stream_mux;
        mux.set_knobs_for_test(StreamMuxKnobs {
            block_ms: 100,
            ..mux.knobs()
        });
        let first = facade.object_on(&tv, Node::A, 0, &[]);
        serve_once(&tv, first, 1).await;
        await_mux(tv.client(), "the reader is connected and free", |s| {
            s.subscriptions() == 0 && s.readers[0].connected
        })
        .await;

        facade.move_all_to_b();
        sync_readers_view(&tv).await;
        mux.push_faults_for_test(
            MuxFaultSite::Keepalive,
            [MuxFault::Err(fred::error::Error::new(
                fred::error::ErrorKind::IO,
                "synthetic connection reset",
            ))],
        );
        mux.set_knobs_for_test(StreamMuxKnobs {
            keepalive_interval: Some(Duration::from_millis(50)),
            probe_timeout: Duration::from_millis(100),
            ..mux.knobs()
        });
        let snap = await_mux(tv.client(), "the reader retires its connection", |s| {
            s.connected_readers() == 0
        })
        .await;

        assert_eq!(
            snap.readers[0].node, None,
            "the router's node record is cleared"
        );
        assert_eq!(
            read_series(
                &metrics,
                "live_edit_stream_mux_keepalives_total",
                &[("result", KEEPALIVE_RESULT_FAILED)]
            ),
            1
        );
        assert_eq!(
            forced_reconnect_row(&metrics),
            forced_reconnects_only_on(CONSUMER_PHASE_KEEPALIVE, 1)
        );
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 0, 0, 0],
            "an idle reader retires without retargeting"
        );
        assert_eq!(readers_connected(&metrics), 0);

        let next = facade.object_on(&tv, Node::B, first, &[first]);
        serve_once(&tv, next, 1).await;
        assert_eq!(
            reader_connect_row(&metrics),
            [2, 0, 0, 0],
            "the next subscription connects afresh, as a bind"
        );
        assert_eq!(
            tv.client().stream_mux.snapshot().readers[0].node,
            Some(facade.name(Node::B))
        );
    }

    // ── A `-MOVED` on the probe, and on a head check ────────────────────────

    /// An idle→active probe answered `-MOVED` (a `PING` names no key, so only
    /// an injected fault produces one) whose slot has moved: the reader
    /// re-syncs its view, retargets before reading, and charges nothing — with
    /// no transient failure allowed, a charge would end the subscription.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_moved_probe_retargets_the_reader_without_charging_its_budget() {
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        let mux = &tv.client().stream_mux;
        mux.set_knobs_for_test(StreamMuxKnobs {
            block_ms: 100,
            max_transient_failures: 0,
            ..mux.knobs()
        });
        let first = facade.object_on(&tv, Node::A, 0, &[]);
        let second = facade.object_on(&tv, Node::A, first, &[first]);
        serve_once(&tv, first, 1).await;
        await_mux(tv.client(), "the reader is connected and free", |s| {
            s.subscriptions() == 0 && s.readers[0].connected
        })
        .await;

        let slot = stream_slot(&tv, second);
        facade.move_to_b_from(slot);
        mux.push_faults_for_test(MuxFaultSite::Probe, [moved(&facade, slot)]);
        let c = subscribe(&tv, second).await;
        xadd_update(&tv, second, 1).await;
        assert_eq!(recv_seqs(&c, 1).await, [1]);
        assert!(
            c.take_terminal_exit().is_none(),
            "the -MOVED was not charged"
        );
        assert_eq!(mux.pending_faults_for_test(MuxFaultSite::Probe), 0);
        assert_eq!(reader_connect_row(&metrics), [1, 0, 1, 0]);
        assert_eq!(mux.snapshot().readers[0].node, Some(facade.name(Node::B)));
    }

    /// The slot's owner moves twice before the reader reads anything: only
    /// the first `-MOVED` retarget of the streak is free. Two probe faults
    /// answer `-MOVED`, and the facade's map moves the slot to B at the first
    /// view sync and back to A at the second. The first probe (on A) syncs,
    /// finds B and retargets there for free; the probe on B syncs, finds A,
    /// and that second retarget with no read between is charged — with no
    /// transient failure allowed, it ends the subscription `UNAVAILABLE`
    /// naming the repeated redirect. Left free like the first, the reader
    /// would retarget back to A and the subscription would not end.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_second_moved_retarget_before_any_read_is_charged_to_the_budget() {
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        let mux = &tv.client().stream_mux;
        mux.set_knobs_for_test(StreamMuxKnobs {
            block_ms: 100,
            max_transient_failures: 0,
            ..mux.knobs()
        });
        let first = facade.object_on(&tv, Node::A, 0, &[]);
        let second = facade.object_on(&tv, Node::A, first, &[first]);
        serve_once(&tv, first, 1).await;
        await_mux(tv.client(), "the reader is connected and free", |s| {
            s.subscriptions() == 0 && s.readers[0].connected
        })
        .await;

        let slot = stream_slot(&tv, second);
        assert_eq!(facade.owner(slot), Node::A, "fixture self-check");
        let syncs = facade.slots_served();
        let reader_syncs = mux.routing_syncs_finished_for_test();
        // First sync: the slot is B's. Second: A's again.
        facade.plan_splits([slot, INITIAL_SPLIT]);
        mux.push_faults_for_test(
            MuxFaultSite::Probe,
            [moved(&facade, slot), moved(&facade, slot)],
        );
        let c = subscribe(&tv, second).await;
        let message = ended_unavailable(&c).await;
        assert!(
            message.contains(
                "gave up after 1 consecutive transient failures (last: a connection probe was \
                 redirected (MOVED) to a new hash slot owner again before any read succeeded)"
            ),
            "the second retarget of the streak is charged, and past the budget ends the \
             subscription; got {message:?}"
        );
        assert_eq!(mux.pending_faults_for_test(MuxFaultSite::Probe), 0);
        assert_eq!(
            (
                facade.slots_served() - syncs,
                mux.routing_syncs_finished_for_test() - reader_syncs
            ),
            (2, 2),
            "precondition: each -MOVED synced the readers' view once, and the map moved at each"
        );
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 0, 1, 0],
            "one bind, then the first (free) retarget to B; the charged one ended the \
             subscription before it retargeted"
        );
    }

    /// A head check answered `-MOVED` (`check_heads`' `any_moved`) whose slot
    /// has moved: the reply's keys are held back, the reader retargets with no
    /// budget charge, and the stream is re-read and checked on the new owner,
    /// so the entry is delivered once. A reply's first entry past the entry
    /// after the read position (here `1005-0` after `5-0`) is what makes the
    /// reader check the head.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_moved_head_check_retargets_the_reader_and_the_stream_is_re_read_there() {
        const RESTORED: u64 = 1_005;
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        let mux = &tv.client().stream_mux;
        mux.set_knobs_for_test(StreamMuxKnobs {
            max_transient_failures: 0,
            ..mux.knobs()
        });
        let oid = facade.object_on(&tv, Node::A, 0, &[]);
        for seq in 1..=5 {
            xadd_update(&tv, oid, seq).await;
        }
        let c = consume_layer_stream(
            tv.client(),
            tv.key_schema(),
            oid,
            StreamCursor::From("5-0".into()),
            Instant::now() + SETTLE,
        )
        .await
        .expect("the subscription must start");
        await_mux(tv.client(), "the read is in flight", |s| s.xreads() >= 1).await;

        let slot = stream_slot(&tv, oid);
        facade.move_to_b_from(slot);
        mux.push_faults_for_test(MuxFaultSite::HeadCheck, [moved(&facade, slot)]);
        xadd_update(&tv, oid, RESTORED).await;
        assert_eq!(recv_seqs(&c, 1).await, [RESTORED]);
        assert!(
            c.take_terminal_exit().is_none(),
            "the -MOVED was not charged"
        );
        assert_eq!(mux.pending_faults_for_test(MuxFaultSite::HeadCheck), 0);
        assert_eq!(reader_connect_row(&metrics), [1, 0, 1, 0]);
        assert_eq!(mux.snapshot().readers[0].node, Some(facade.name(Node::B)));
        super::super::mux_test_helpers::assert_quiet(&c, Duration::from_millis(300), "once").await;
    }

    // ── A `-MOVED` whose slot owner has not moved ───────────────────────────
    //
    // `slot_moved` re-syncs the view and finds the reader's own node still
    // the owner: the redirect is an ordinary transient failure, charged to the
    // budget. With no transient failure allowed, that charge ends the
    // subscription — which a `-MOVED` taken as a retarget would not.

    /// On the idle→active probe: the `-MOVED` is charged as a transient probe
    /// failure (the subscription ends `UNAVAILABLE` naming it), after one view
    /// sync, and nothing retargets.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_moved_probe_whose_slot_owner_has_not_moved_is_charged_as_transient() {
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        let mux = &tv.client().stream_mux;
        mux.set_knobs_for_test(StreamMuxKnobs {
            block_ms: 100,
            max_transient_failures: 0,
            ..mux.knobs()
        });
        let first = facade.object_on(&tv, Node::A, 0, &[]);
        let second = facade.object_on(&tv, Node::A, first, &[first]);
        serve_once(&tv, first, 1).await;
        await_mux(tv.client(), "the reader is connected and free", |s| {
            s.subscriptions() == 0 && s.readers[0].connected
        })
        .await;

        let syncs = facade.slots_served();
        mux.push_faults_for_test(
            MuxFaultSite::Probe,
            [moved(&facade, stream_slot(&tv, second))],
        );
        let c = subscribe(&tv, second).await;
        let message = ended_unavailable(&c).await;
        assert!(
            message.contains(&format!(
                "(last: its connection probe failed transiently (kind: {}))",
                fred::error::ErrorKind::Cluster.to_str()
            )),
            "the unchanged-owner -MOVED is charged as a transient probe failure; got {message:?}"
        );
        assert_eq!(mux.pending_faults_for_test(MuxFaultSite::Probe), 0);
        assert!(
            facade.slots_served() > syncs,
            "the view named the reader's own node, so the reader re-synced it before deciding"
        );
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 0, 0, 0],
            "the owner did not move: no retarget"
        );
        assert_eq!(mux.snapshot().readers[0].node, Some(facade.name(Node::A)));
    }

    /// On a head check: the failed check is charged as a transient head-check
    /// failure (the subscription ends `UNAVAILABLE` naming it), after one view
    /// sync, and nothing retargets.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_moved_head_check_whose_slot_owner_has_not_moved_is_charged_as_transient() {
        const RESTORED: u64 = 1_005;
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        let mux = &tv.client().stream_mux;
        mux.set_knobs_for_test(StreamMuxKnobs {
            max_transient_failures: 0,
            ..mux.knobs()
        });
        let oid = facade.object_on(&tv, Node::A, 0, &[]);
        for seq in 1..=5 {
            xadd_update(&tv, oid, seq).await;
        }
        let c = consume_layer_stream(
            tv.client(),
            tv.key_schema(),
            oid,
            StreamCursor::From("5-0".into()),
            Instant::now() + SETTLE,
        )
        .await
        .expect("the subscription must start");
        await_mux(tv.client(), "the read is in flight", |s| s.xreads() >= 1).await;

        let syncs = facade.slots_served();
        mux.push_faults_for_test(
            MuxFaultSite::HeadCheck,
            [moved(&facade, stream_slot(&tv, oid))],
        );
        xadd_update(&tv, oid, RESTORED).await;
        let message = ended_unavailable(&c).await;
        assert!(
            message.contains("(last: a stream head check failed)"),
            "the unchanged-owner -MOVED is charged as a failed head check; got {message:?}"
        );
        assert_eq!(mux.pending_faults_for_test(MuxFaultSite::HeadCheck), 0);
        assert!(
            facade.slots_served() > syncs,
            "the view named the reader's own node, so the reader re-synced it before deciding"
        );
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 0, 0, 0],
            "the owner did not move: no retarget"
        );
        assert_eq!(mux.snapshot().readers[0].node, Some(facade.name(Node::A)));
    }

    // ── A forced reconnect whose node still serves the reader ───────────────

    /// A wedge verdict on the XREAD (an injected `Timeout`) whose bound slot
    /// is still A's: the reader rebuilds its connection to A in place
    /// (`force_reconnection`) — the same client, no retire, no retarget — and
    /// keeps serving.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_forced_reconnect_whose_slot_owner_has_not_moved_reconnects_the_same_client() {
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        tv.client()
            .stream_mux
            .set_knobs_for_test(fast_backoff_knobs(tv.client()));
        let oid = facade.object_on(&tv, Node::A, 0, &[]);
        let c = subscribe(&tv, oid).await;
        xadd_update(&tv, oid, 1).await;
        assert_eq!(recv_seqs(&c, 1).await, [1]);
        let id = reader_client_id(tv.client(), 0);

        // Taken by the read issued once update 2 returns the one in flight;
        // update 3 is read after the forced reconnect.
        push_timeout_faults(tv.client(), 1);
        xadd_update(&tv, oid, 2).await;
        assert_eq!(recv_seqs(&c, 1).await, [2]);
        xadd_update(&tv, oid, 3).await;
        assert_eq!(recv_seqs(&c, 1).await, [3]);
        assert!(c.take_terminal_exit().is_none());

        assert_eq!(
            forced_reconnect_row(&metrics),
            forced_reconnects_only_on(CONSUMER_PHASE_XREAD, 1),
            "precondition: the wedge verdict forced a reconnect"
        );
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 0, 0, 0],
            "the owner did not move: no retire, no retarget"
        );
        assert_eq!(
            reader_client_id(tv.client(), 0),
            id,
            "the connection was rebuilt in place, on the same client"
        );
        assert_eq!(
            tv.client().stream_mux.snapshot().readers[0].node,
            Some(facade.name(Node::A))
        );
    }

    /// A failed keepalive on an idle reader whose node is still a primary:
    /// the reader rebuilds its connection in place rather than retiring it —
    /// it stays connected, on the same client — and its keepalive answers
    /// again.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_keepalive_on_a_node_still_a_primary_reconnects_instead_of_retiring() {
        use live_edit::observability::metrics::KEEPALIVE_RESULT_OK;
        let keepalives = |metrics: &Metrics, result: &str| {
            read_series(
                metrics,
                "live_edit_stream_mux_keepalives_total",
                &[("result", result)],
            )
        };
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        let mux = &tv.client().stream_mux;
        mux.set_knobs_for_test(StreamMuxKnobs {
            block_ms: 100,
            ..mux.knobs()
        });
        let oid = facade.object_on(&tv, Node::A, 0, &[]);
        serve_once(&tv, oid, 1).await;
        await_mux(tv.client(), "the reader is connected and free", |s| {
            s.subscriptions() == 0 && s.readers[0].connected
        })
        .await;
        let id = reader_client_id(tv.client(), 0);

        mux.push_faults_for_test(
            MuxFaultSite::Keepalive,
            [MuxFault::Err(fred::error::Error::new(
                fred::error::ErrorKind::IO,
                "synthetic connection reset",
            ))],
        );
        mux.set_knobs_for_test(StreamMuxKnobs {
            keepalive_interval: Some(Duration::from_millis(50)),
            probe_timeout: Duration::from_millis(100),
            ..mux.knobs()
        });
        let deadline = Instant::now() + SETTLE;
        await_until("the faulted keepalive tick must fail", deadline, || {
            keepalives(&metrics, KEEPALIVE_RESULT_FAILED) == 1
        })
        .await;
        // A keepalive tick that answers after the failed one ran after that
        // tick's forced reconnect returned: the reader's loop is sequential.
        // A retired reader holds no connection to tick on.
        let answered = keepalives(&metrics, KEEPALIVE_RESULT_OK);
        let deadline = Instant::now() + SETTLE;
        await_until(
            "the reader must answer a keepalive again, or retire",
            deadline,
            || keepalives(&metrics, KEEPALIVE_RESULT_OK) > answered || mux.connected_readers() == 0,
        )
        .await;

        assert_eq!(
            mux.connected_readers(),
            1,
            "a node still a primary is reconnected, not retired"
        );
        assert_eq!(
            reader_client_id(tv.client(), 0),
            id,
            "the connection was rebuilt in place, on the same client"
        );
        assert_eq!(
            forced_reconnect_row(&metrics),
            forced_reconnects_only_on(CONSUMER_PHASE_KEEPALIVE, 1)
        );
        assert_eq!(reader_connect_row(&metrics), [1, 0, 0, 0]);
        assert_eq!(readers_connected(&metrics), 1);
    }

    // ── A failed retarget (`retarget_due`) ──────────────────────────────────
    //
    // The facade's one reader is connected to A by a served stream there; a
    // `Connect` fault pushed only then is taken by its next connect attempt:
    // the retarget to B a subscription on B makes it do (the facade names B,
    // so the attempt reaches its setup and takes the fault there).

    /// A retarget whose connect fails transiently (`IO`) is counted
    /// `cause="retarget", result="failed"`, and its retry, made while the
    /// reader holds no connection at all, is still a retarget
    /// (`result="ok"`): the reader remembers it retired its connection to A
    /// to follow its slot. The subscription is kept and served from B.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_retarget_connect_and_its_retry_both_count_as_retargets() {
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        let mux = &tv.client().stream_mux;
        let on_a = facade.object_on(&tv, Node::A, 0, &[]);
        let on_b = facade.object_on(&tv, Node::B, 0, &[]);
        serve_once(&tv, on_a, 1).await;
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 0, 0, 0],
            "precondition: one bind"
        );
        assert_eq!(
            mux.snapshot().readers[0].node,
            Some(facade.name(Node::A)),
            "precondition: the reader is on A"
        );

        mux.push_faults_for_test(MuxFaultSite::Connect, [dropped_setup()]);
        let c = subscribe(&tv, on_b).await;
        xadd_update(&tv, on_b, 1).await;
        assert_eq!(
            recv_seqs(&c, 1).await,
            [1],
            "the retried retarget serves it"
        );
        assert!(c.take_terminal_exit().is_none());
        assert_connect_faults_taken(tv.client(), "the retarget's first attempt");
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 0, 1, 1],
            "the failed retarget, and its retry, both counted as retargets"
        );
        assert_eq!(mux.snapshot().readers[0].node, Some(facade.name(Node::B)));
        assert_eq!(readers_connected(&metrics), 1);
    }

    /// A retarget whose connect fails with no transient failure allowed ends
    /// the reader's subscriptions (counted `cause="retarget",
    /// result="failed"`). The reader is then retargeting nothing, so its next
    /// connect — for a re-subscribe — is a `bind`, not a retarget.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_retarget_that_ends_the_subscriptions_makes_the_next_connect_a_bind() {
        let facade = ClusterFacade::start().await;
        let tv = facade.test_valkey(1).await;
        let metrics = install_metrics(&tv).await;
        let mux = &tv.client().stream_mux;
        mux.set_knobs_for_test(StreamMuxKnobs {
            max_transient_failures: 0,
            ..mux.knobs()
        });
        let on_a = facade.object_on(&tv, Node::A, 0, &[]);
        let on_b = facade.object_on(&tv, Node::B, 0, &[]);
        serve_once(&tv, on_a, 1).await;
        assert_eq!(
            mux.snapshot().readers[0].node,
            Some(facade.name(Node::A)),
            "precondition: the reader is on A"
        );

        mux.push_faults_for_test(MuxFaultSite::Connect, [dropped_setup()]);
        let c = subscribe(&tv, on_b).await;
        let message = ended_unavailable(&c).await;
        assert_connect_faults_taken(tv.client(), "the retarget's connect");
        assert!(
            message.contains(&format!(
                "gave up after 1 consecutive transient failures (last: it could not connect to {}",
                facade.name(Node::B)
            )),
            "the failed retarget to B ends the subscription; got {message:?}"
        );
        assert_eq!(
            reader_connect_row(&metrics),
            [1, 0, 0, 1],
            "one bind, one failed retarget, and no retry"
        );
        assert_eq!(mux.connected_readers(), 0);

        let again = subscribe(&tv, on_b).await;
        xadd_update(&tv, on_b, 1).await;
        assert_eq!(
            recv_seqs(&again, 1).await,
            [1],
            "the re-subscribe is served"
        );
        assert_eq!(
            reader_connect_row(&metrics),
            [2, 0, 0, 1],
            "the re-subscribe's connect is a bind: the failed retarget ended with its \
             subscriptions"
        );
        assert_eq!(mux.snapshot().readers[0].node, Some(facade.name(Node::B)));
    }

    // ── A retarget whose `QUIT` does not finish cleanly ─────────────────────
    //
    // fred never waits for `QUIT`'s reply: it first drains the replies of the
    // commands in flight on the connection, then writes `QUIT` and ends the
    // router (`vendor/fred-10.1.0/src/router/commands.rs:122-123`,
    // `router/utils.rs:99-114`). So a `QUIT` does not finish when a command
    // ahead of it is never answered: here an `ECHO` the facade stalls on A,
    // sent on the reader's client while its subscription is gone and its read
    // in flight (the `ECHO` interrupts that read, `Blocking::Interrupt`, and
    // the reader goes idle). The reader's next binding is on B, so it retires
    // its connection to A and its `QUIT` hangs in that drain:
    //
    // - with the 5 s command timeout, past `QUIT_TIMEOUT` (2 s): an overrun;
    // - with a 500 ms command timeout, fred fails the `QUIT` itself inside
    //   `QUIT_TIMEOUT`: an error, not an overrun.
    //
    // Either way the old router is left stuck in the drain unless the retire
    // aborts it. It is the only thing that can close the stalled connection to
    // A, and a router ended by an abort (unlike one ended by a clean `QUIT`) is
    // one a still-live watch would restart and count — so these tests are what
    // show the watch was retired first.
    //
    // The keepalive is turned off, so no tick lands on the stalled connection;
    // current-thread, so the [`ThreadCapture`] sees the reader task's lines.

    /// How a stalled-`QUIT` retarget ended, for the two tests to assert on.
    struct StalledQuitRetarget {
        restarts: i64,
        restart_lines: usize,
        overran_lines: usize,
        watchers_before: usize,
        watchers_after: usize,
        connects: [i64; 4],
    }

    async fn retarget_with_a_stalled_quit(command_timeout_ms: u64) -> StalledQuitRetarget {
        let capture = ThreadCapture::start();
        let facade = ClusterFacade::start().await;
        let mut config: Config = crate::common::test_config();
        config.valkey_url = facade.cluster_url();
        config.subscriber_pool_size = 1;
        config.valkey_command_timeout_ms = command_timeout_ms;
        let client = tokio::time::timeout(CONNECT_BUDGET, connect_test_client(&config))
            .await
            .expect("connect must settle")
            .expect("a client must connect through the cluster facade");
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        client.stream_mux.set_knobs_for_test(StreamMuxKnobs {
            keepalive_interval: None,
            ..client.stream_mux.knobs()
        });
        let schema = fresh_schema();
        let on_a = facade.object_on_schema(&schema, Node::A, 0, &[]);
        let on_b = facade.object_on_schema(&schema, Node::B, 0, &[]);

        let first = subscribe_on(&client, &schema, on_a).await;
        xadd_on(&client, &schema, on_a, 1).await;
        assert_eq!(recv_seqs(&first, 1).await, [1]);
        let old = client
            .stream_mux
            .reader_client_for_test(0)
            .expect("connected to A");
        let old_connection = sole_connection_id(&old);
        // The `ECHO` below must meet a read the server shows blocked, so it
        // interrupts that read rather than queueing behind its BLOCK; a
        // subscription leaving does not wake the read, so it stays blocked.
        await_reader_blocked_in_xread(client.pool.next(), &old).await;
        drop(first);
        // A subscription's drop releases its router reference and queues the
        // reader's removal synchronously, so this holds at once; no wait.
        assert_eq!(client.stream_mux.snapshot().subscriptions(), 0);
        let watchers_before = client.router_watchers_for_test();

        let marker = format!("b-azh-stall-{}", uuid::Uuid::new_v4());
        facade.stall_next_echo_on_a(marker.clone());
        let echo = tokio::spawn(async move {
            old.custom::<Value, String>(
                CustomCommand::new_static("ECHO", None::<u16>, false),
                vec![marker],
            )
            .await
        });
        let deadline = Instant::now() + SETTLE;
        await_until(
            "the ECHO must be stalled on A and the reader's read interrupted",
            deadline,
            || {
                facade.echo_stalled.load(Ordering::SeqCst) == 1
                    && read_unblocked_retries(&metrics) >= 1
            },
        )
        .await;

        let second = subscribe_on(&client, &schema, on_b).await;
        xadd_on(&client, &schema, on_b, 1).await;
        assert_eq!(
            recv_seqs(&second, 1).await,
            [1],
            "the retarget to B serves the new subscription"
        );
        await_connection_closed(
            client.pool.next(),
            old_connection,
            "the old router, stuck draining ahead of its QUIT, must be aborted: nothing else \
             closes the connection to A",
        )
        .await;
        echo.abort();
        // A wrongful restart of the aborted old router (its watch still live)
        // follows the abort at once — a first restart waits nothing — but on
        // the watcher's own task; a bounded quiet window lets it land, so the
        // counts below are not read before it could.
        tokio::time::sleep(QUIET).await;

        let captured = capture.snapshot();
        let outcome = StalledQuitRetarget {
            restarts: reader_restarts(&metrics),
            restart_lines: captured.with_message(RESTART_MSG).len(),
            overran_lines: captured.with_message(QUIT_OVERRAN_MSG).len(),
            watchers_before,
            watchers_after: client.router_watchers_for_test(),
            connects: reader_connect_row(&metrics),
        };
        drop(second);
        drop(capture);
        for oid in [on_a, on_b] {
            use fred::interfaces::KeysInterface;
            let _: i64 = client
                .pool
                .del(schema.layer_stream(oid))
                .await
                .expect("DEL must succeed");
        }
        tokio::time::timeout(SETTLE, client.quit())
            .await
            .expect("quit must settle");
        outcome
    }

    fn assert_retired_cleanly(outcome: &StalledQuitRetarget) {
        assert_eq!(outcome.connects, [1, 0, 1, 0], "one bind, one retarget");
        assert_eq!(
            (outcome.restarts, outcome.restart_lines),
            (0, 0),
            "the aborted old router is neither restarted, counted nor logged: its watch was \
             retired before its QUIT"
        );
        assert_eq!(
            outcome.watchers_after, outcome.watchers_before,
            "the old client's watcher exited as the new one's started"
        );
    }

    /// A retarget whose `QUIT` overruns `QUIT_TIMEOUT`: the old router is
    /// aborted (A's connection closes), nothing is restarted, counted or
    /// logged at ERROR, the watcher count is unchanged, and the overrun WARN
    /// is logged once.
    #[tokio::test]
    async fn a_retarget_whose_quit_overruns_aborts_the_old_router_and_restarts_nothing() {
        let outcome = retarget_with_a_stalled_quit(5_000).await;
        assert_retired_cleanly(&outcome);
        assert_eq!(outcome.overran_lines, 1, "the QUIT overran its bound");
    }

    /// A retarget whose `QUIT` fails inside `QUIT_TIMEOUT` — not an overrun —
    /// still ends the old router: A's connection closes, and nothing is
    /// restarted, counted or logged. Aborting only on an overrun would leave
    /// the old router stuck in its drain, unsupervised, holding A's
    /// connection open.
    #[tokio::test]
    async fn a_retarget_whose_quit_fails_inside_its_bound_still_ends_the_old_router() {
        let outcome = retarget_with_a_stalled_quit(500).await;
        assert_retired_cleanly(&outcome);
        assert_eq!(
            outcome.overran_lines, 0,
            "precondition: the QUIT failed inside its bound rather than overrunning it"
        );
    }
}
