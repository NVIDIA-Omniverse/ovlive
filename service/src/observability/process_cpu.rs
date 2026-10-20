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

//! Process CPU time as the kernel accounts it: the `cpu_seconds_total` field of
//! `/scaling/v1`, from which the autoscaler derives each replica's CPU use
//! between polls. Read from the service process itself, so it needs no
//! cluster metrics pipeline and leaves out any sidecar in the pod.

/// Clock ticks per second of the `utime` / `stime` fields in `/proc/<pid>/stat`
/// (`USER_HZ`, what `sysconf(_SC_CLK_TCK)` reports). Linux fixes it at 100 on
/// x86-64 and arm64, the architectures the service ships for, independent of
/// the kernel's internal tick rate (`getconf CLK_TCK` prints 100 on both). A
/// constant rather than a `sysconf` call so this module needs no `libc`
/// dependency in the production build.
const USER_HZ: f64 = 100.0;

/// CPU seconds (user plus system, every thread) this process has used since it
/// started, from `/proc/self/stat`. `None` off Linux, or when the file cannot
/// be read or parsed.
pub fn cpu_seconds_total() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
        parse_cpu_seconds(&stat)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// `(utime + stime) / USER_HZ` from a `/proc/<pid>/stat` line. The second field
/// is the command name in parentheses, which may itself contain spaces and
/// parentheses, so the fields are counted from after its last `)`: `utime` and
/// `stime` are the 14th and 15th fields of the line, the 12th and 13th after
/// the name. `None` when the line is too short or either field is not an
/// unsigned integer.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn parse_cpu_seconds(stat: &str) -> Option<f64> {
    let after_name = &stat[stat.rfind(')')? + 1..];
    let mut fields = after_name.split_whitespace().skip(11);
    let utime: u64 = fields.next()?.parse().ok()?;
    let stime: u64 = fields.next()?.parse().ok()?;
    Some((utime.saturating_add(stime)) as f64 / USER_HZ)
}

#[cfg(test)]
mod tests {
    use super::parse_cpu_seconds;

    /// A `/proc/<pid>/stat` line: pid, the command name in parentheses, then
    /// fields 3 onward, of which 14 (`utime`) and 15 (`stime`) are 1234 and
    /// 566 ticks. The fields either side of them (`cmajflt` = 777, `cutime` =
    /// 888) are distinct, so an off-by-one reads a different total.
    fn stat_line(comm: &str) -> String {
        format!(
            "4242 ({comm}) S 1 4242 4242 0 -1 4194560 9000 0 0 777 1234 566 888 0 20 0 \
             12 0 5000 123456789 2048 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 3 \
             0 0 0 0 0"
        )
    }

    #[test]
    fn parse_cpu_seconds_sums_utime_and_stime_in_clock_ticks() {
        assert_eq!(parse_cpu_seconds(&stat_line("live-edit")), Some(18.0));
    }

    /// The command name may itself contain spaces and parentheses, so the
    /// fields are counted from after its last `)`.
    #[test]
    fn parse_cpu_seconds_counts_fields_after_the_last_parenthesis_of_the_name() {
        assert_eq!(parse_cpu_seconds(&stat_line("a) b (c) d")), Some(18.0));
    }

    #[test]
    fn parse_cpu_seconds_refuses_a_short_or_malformed_line() {
        assert_eq!(parse_cpu_seconds("4242 (live-edit) S 1 4242"), None);
        assert_eq!(parse_cpu_seconds("no name here"), None);
        let malformed = stat_line("live-edit").replace(" 1234 ", " x ");
        assert_eq!(parse_cpu_seconds(&malformed), None);
    }

    /// On Linux the live reading parses, and the process's CPU time never
    /// goes backwards between two readings.
    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_seconds_total_reads_this_process_and_never_decreases() {
        let first = super::cpu_seconds_total().expect("/proc/self/stat parses on Linux");
        let mut x = 0u64;
        for i in 0..1_000_000u64 {
            x = x.wrapping_mul(31).wrapping_add(i);
        }
        std::hint::black_box(x);
        let second = super::cpu_seconds_total().expect("/proc/self/stat parses on Linux");
        assert!(first >= 0.0 && second >= first, "{first} then {second}");
    }
}
