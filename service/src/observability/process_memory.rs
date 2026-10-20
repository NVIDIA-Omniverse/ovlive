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

//! Process memory as the kernel accounts it, independent of which allocator
//! the binary uses. Source for the `process_resident_memory_bytes` gauge
//! (b.u5k): the resident set is what the pod's working set and a memory-based
//! autoscaler see, so it is the number that shows whether memory returns to
//! the OS after load.

/// The process's resident set size in bytes, read from `VmRSS` in
/// `/proc/self/status`. `None` off Linux, or when the file cannot be read or
/// parsed.
pub fn resident_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        parse_vm_rss_bytes(&status)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Parse the `VmRSS:` line of a `/proc/<pid>/status` document into bytes.
/// The kernel reports the value in kB (KiB), e.g. `VmRSS:     123456 kB`.
/// `None` when the line is missing, the unit is not `kB`, or the value is not
/// an unsigned integer.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn parse_vm_rss_bytes(status: &str) -> Option<u64> {
    let value = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))?;
    let mut fields = value.split_whitespace();
    let kib: u64 = fields.next()?.parse().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() {
        return None;
    }
    kib.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `/proc/self/status` excerpt in the kernel's own layout (tab after the
    /// colon, right-aligned value). The neighbouring `Vm*` lines carry
    /// different values so a parser that read the wrong line would be caught.
    const STATUS: &str = "Name:\tlive-edit\n\
                          State:\tS (sleeping)\n\
                          VmPeak:\t  900000 kB\n\
                          VmSize:\t  800000 kB\n\
                          VmHWM:\t   300000 kB\n\
                          VmRSS:\t   123456 kB\n\
                          RssAnon:\t   100000 kB\n\
                          Threads:\t17\n";

    #[test]
    fn parse_vm_rss_bytes_kernel_layout_converts_kib_to_bytes() {
        assert_eq!(parse_vm_rss_bytes(STATUS), Some(123_456 * 1024));
    }

    #[test]
    fn parse_vm_rss_bytes_missing_line_returns_none() {
        let without = STATUS.replace("VmRSS:\t   123456 kB\n", "");
        assert_eq!(parse_vm_rss_bytes(&without), None);
    }

    #[test]
    fn parse_vm_rss_bytes_malformed_line_returns_none() {
        for line in [
            "VmRSS:\n",                     // no value
            "VmRSS:\t   123456\n",          // no unit
            "VmRSS:\t   123456 MB\n",       // unit other than kB
            "VmRSS:\t   -123456 kB\n",      // negative
            "VmRSS:\t   12.5 kB\n",         // not an integer
            "VmRSS:\t   lots kB\n",         // not a number
            "VmRSS:\t   123456 kB extra\n", // trailing field
        ] {
            assert_eq!(parse_vm_rss_bytes(line), None, "line {line:?}");
        }
    }

    #[test]
    fn parse_vm_rss_bytes_value_overflowing_u64_bytes_returns_none() {
        // A kB count whose byte value does not fit in u64 must not wrap into
        // a small, plausible-looking resident size.
        let line = format!("VmRSS:\t{} kB\n", u64::MAX / 1024 + 1);
        assert_eq!(parse_vm_rss_bytes(&line), None);
    }

    /// Reads this test process's real `/proc/self/status`: the one I/O check
    /// that the kernel's actual format still parses.
    #[cfg(target_os = "linux")]
    #[test]
    fn resident_memory_bytes_live_read_on_linux_is_positive() {
        let bytes = resident_memory_bytes().expect("VmRSS must parse from /proc/self/status");
        assert!(bytes > 0, "a running process has a non-zero resident set");
    }
}
