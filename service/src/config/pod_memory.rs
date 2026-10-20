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

//! The pod memory limit every pod-memory budget is derived from
//! ([`super::memory_plan`]), and where it is read.
//!
//! [`Config::from_env`](super::Config::from_env), the production entry,
//! passes [`PodMemorySource::Cgroup`]: the limit of the cgroup the process
//! runs in, read once at startup ([`read_pod_memory_limit`]). Tests pass
//! [`PodMemorySource::Fixed`] or [`PodMemorySource::Reference`], so the
//! memory limit of the machine that runs them (a CI runner's own cgroup, a
//! developer laptop) never changes a test's outcome.
//!
//! # The read
//!
//! - **cgroup v2.** The process's cgroup is the `0::<path>` line of
//!   `/proc/self/cgroup`; the `cgroup2` mount comes from
//!   `/proc/self/mountinfo`. `memory.max` is read in the process's cgroup
//!   directory and in each ancestor visible under the mount, and the smallest
//!   number wins; `max` means no limit at that level, and a level without the
//!   file (the mount root has none) is skipped. In a private cgroup namespace,
//!   the usual container case, that is just `/sys/fs/cgroup/memory.max`.
//! - **cgroup v1.** Used when the `memory` controller is on a v1 hierarchy (a
//!   hybrid host's memory hierarchy wins over its v2 one): the `memory` line
//!   of `/proc/self/cgroup` and the `cgroup` mount carrying the `memory`
//!   option. `memory.stat`'s `hierarchical_memory_limit`, which already
//!   accounts for the ancestors, is preferred, then `memory.limit_in_bytes`.
//!   A value of 2^62 or more is the kernel's "unlimited".
//! - **Host clamp.** A cgroup limit above the host's physical memory
//!   (`MemTotal` in `/proc/meminfo`) is sized to the host instead
//!   ([`PodMemoryLimitSource::HostMemory`]). Without a readable `MemTotal` the
//!   cgroup limit stands.
//!
//! `memory.high` is not read: the budgets bound the limit the kernel's OOM
//! killer enforces, and `memory.high` only throttles.
//!
//! # No readable limit
//!
//! An unlimited cgroup, no cgroup memory files at all (macOS, other
//! non-Linux hosts, no cgroupfs) and a read or parse error all size the pod at
//! [`REFERENCE_POD_MEMORY_LIMIT_BYTES`], the chart's default limit
//! ([`PodMemoryLimitSource::Reference`], with the [`ReferenceReason`]). That is
//! the one derivation path every budget follows, so a pod whose limit cannot be
//! read runs at exactly the budgets a pod at the chart's default limit gets.
//!
//! The limit is read once. A pod-spec change rolls the pods, and on
//! Kubernetes 1.27 and later, where a pod's memory can be resized in place,
//! the chart restarts the container on such a resize, so a running process
//! under the chart never needs to read it again.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

/// The pod size used when no limit can be read, in bytes: 5 GiB, the chart's
/// default `resources.limits.memory`.
pub const REFERENCE_POD_MEMORY_LIMIT_BYTES: u64 = REFERENCE_POD_MEMORY_LIMIT_MIB * 1024 * 1024;

/// [`REFERENCE_POD_MEMORY_LIMIT_BYTES`] in MiB: 5,120.
pub const REFERENCE_POD_MEMORY_LIMIT_MIB: u64 = 5_120;

/// A cgroup v1 limit at or above this is the kernel's "unlimited" sentinel
/// (`PAGE_COUNTER_MAX` pages, rounded to the page size, reads back just under
/// 2^63).
pub const CGROUP_V1_UNLIMITED_BYTES: u64 = 1 << 62;

/// The source of the pod memory limit a [`Config`](super::Config) is built
/// for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PodMemorySource {
    /// Read the limit of the cgroup the running process belongs to
    /// ([`read_pod_memory_limit`]). What
    /// [`Config::from_env`](super::Config::from_env) passes.
    Cgroup,
    /// A limit of exactly this many bytes, as if read from a cgroup v2
    /// `memory.max` (source [`PodMemoryLimitSource::CgroupV2`], no host
    /// clamp).
    Fixed(u64),
    /// The reference pod size, [`REFERENCE_POD_MEMORY_LIMIT_BYTES`]: the
    /// limit used when no limit is readable.
    Reference,
}

impl PodMemorySource {
    /// The limit this source gives.
    ///
    /// [`Self::Cgroup`] reads under the filesystem root, or, in a build with
    /// the test-only `test-support` feature, under the directory named by the
    /// `MEASUREMENT_POD_MEMORY_CGROUP_ROOT` environment variable when it is
    /// set, so a test can start the binary on a fixture cgroup tree.
    pub fn resolve(&self) -> PodMemoryLimit {
        match self {
            Self::Cgroup => read_pod_memory_limit(&cgroup_root()),
            Self::Fixed(bytes) => PodMemoryLimit {
                bytes: *bytes,
                source: PodMemoryLimitSource::CgroupV2,
                reason: None,
            },
            Self::Reference => PodMemoryLimit::reference(None),
        }
    }
}

/// The directory [`PodMemorySource::Cgroup`] reads `proc/` and the cgroup
/// mounts under: `/`, or the test-only override.
fn cgroup_root() -> PathBuf {
    #[cfg(feature = "test-support")]
    {
        if let Some(root) = std::env::var_os("MEASUREMENT_POD_MEMORY_CGROUP_ROOT") {
            return PathBuf::from(root);
        }
    }
    PathBuf::from("/")
}

/// Where a [`PodMemoryLimit`] came from: the `source` label of
/// `live_edit_pod_memory_limit_bytes` and of the startup line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PodMemoryLimitSource {
    /// A cgroup v2 `memory.max`.
    CgroupV2,
    /// A cgroup v1 `hierarchical_memory_limit` or `memory.limit_in_bytes`.
    CgroupV1,
    /// The host's physical memory (`MemTotal`), which was below the cgroup
    /// limit.
    HostMemory,
    /// No readable limit: the reference size,
    /// [`REFERENCE_POD_MEMORY_LIMIT_BYTES`].
    Reference,
}

impl PodMemoryLimitSource {
    /// Every source, in label order.
    pub const ALL: [Self; 4] = [
        Self::CgroupV2,
        Self::CgroupV1,
        Self::HostMemory,
        Self::Reference,
    ];

    /// The label value: `cgroup_v2`, `cgroup_v1`, `host_memory` or
    /// `reference`.
    pub const fn label(self) -> &'static str {
        match self {
            Self::CgroupV2 => "cgroup_v2",
            Self::CgroupV1 => "cgroup_v1",
            Self::HostMemory => "host_memory",
            Self::Reference => "reference",
        }
    }

    /// The source in words, for operator-facing text.
    pub const fn describe(self) -> &'static str {
        match self {
            Self::CgroupV2 => "cgroup v2",
            Self::CgroupV1 => "cgroup v1",
            Self::HostMemory => "the host's physical memory",
            Self::Reference => "no readable limit; the 5,120 MiB reference size",
        }
    }
}

/// Why a limit is the reference size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceReason {
    /// The cgroup sets no memory limit (`max`, or the v1 sentinel).
    Unlimited,
    /// No cgroup memory files: not Linux, no cgroupfs, or no memory
    /// controller.
    NotFound,
    /// A file could not be read or parsed.
    Unreadable(String),
}

impl fmt::Display for ReferenceReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unlimited => f.write_str("unlimited"),
            Self::NotFound => f.write_str("not_found"),
            Self::Unreadable(detail) => write!(f, "unreadable: {detail}"),
        }
    }
}

/// The pod memory limit the budgets are derived from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PodMemoryLimit {
    /// The limit, in bytes.
    pub bytes: u64,
    /// Where it came from.
    pub source: PodMemoryLimitSource,
    /// Why it is the reference size, for [`PodMemoryLimitSource::Reference`]
    /// read from a cgroup; `None` otherwise, and for an explicit
    /// [`PodMemorySource::Reference`].
    pub reason: Option<ReferenceReason>,
}

impl PodMemoryLimit {
    /// The reference size, for `reason`.
    pub fn reference(reason: Option<ReferenceReason>) -> Self {
        Self {
            bytes: REFERENCE_POD_MEMORY_LIMIT_BYTES,
            source: PodMemoryLimitSource::Reference,
            reason,
        }
    }

    /// [`Self::bytes`] in whole MiB, rounded down.
    pub const fn mib(&self) -> u64 {
        self.bytes / (1024 * 1024)
    }

    /// [`Self::reason`] as text, `none` when there is none.
    pub fn reason_text(&self) -> String {
        self.reason
            .as_ref()
            .map_or_else(|| "none".to_owned(), ToString::to_string)
    }
}

/// What one cgroup read found, before the host clamp.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CgroupLimit {
    /// A limit in bytes, from the hierarchy named.
    Limited(u64, PodMemoryLimitSource),
    /// No limit, or none readable.
    Reference(ReferenceReason),
}

/// Read the pod memory limit under `root` (`/` in production): the cgroup
/// limit (module docs, *The read*), clamped to the host's `MemTotal`, or the
/// reference size with its reason. Never fails.
pub fn read_pod_memory_limit(root: &Path) -> PodMemoryLimit {
    match read_cgroup_limit(root) {
        CgroupLimit::Reference(reason) => PodMemoryLimit::reference(Some(reason)),
        CgroupLimit::Limited(bytes, source) => match read_host_memory(root) {
            Some(host) if host < bytes => PodMemoryLimit {
                bytes: host,
                source: PodMemoryLimitSource::HostMemory,
                reason: None,
            },
            _ => PodMemoryLimit {
                bytes,
                source,
                reason: None,
            },
        },
    }
}

/// `path` under `root`: an absolute path read below the root instead of `/`.
fn under(root: &Path, path: &str) -> PathBuf {
    root.join(path.trim_start_matches('/'))
}

/// Read a file to a string; `Ok(None)` when it does not exist.
fn read_optional(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

fn read_cgroup_limit(root: &Path) -> CgroupLimit {
    match try_read_cgroup_limit(root) {
        Ok(limit) => limit,
        Err(detail) => CgroupLimit::Reference(ReferenceReason::Unreadable(detail)),
    }
}

fn try_read_cgroup_limit(root: &Path) -> Result<CgroupLimit, String> {
    let Some(cgroups) = read_optional(&under(root, "/proc/self/cgroup"))? else {
        return Ok(CgroupLimit::Reference(ReferenceReason::NotFound));
    };
    let Some(mountinfo) = read_optional(&under(root, "/proc/self/mountinfo"))? else {
        return Ok(CgroupLimit::Reference(ReferenceReason::NotFound));
    };
    let memberships = parse_proc_cgroup(&cgroups);
    let mounts = parse_mountinfo(&mountinfo);

    // A hybrid host: the memory controller's own hierarchy wins.
    if let Some(path) = memberships.v1_memory.as_deref() {
        if let Some(mount) = mounts.iter().find(|m| m.is_v1_memory()) {
            let dir = under(root, &mount.cgroup_dir(path));
            return read_v1_limit(&dir);
        }
    }
    if let Some(path) = memberships.v2.as_deref() {
        if let Some(mount) = mounts.iter().find(|m| m.fstype == "cgroup2") {
            let top = under(root, &mount.mount_point);
            let dir = under(root, &mount.cgroup_dir(path));
            return read_v2_limit(&top, &dir);
        }
    }
    Ok(CgroupLimit::Reference(ReferenceReason::NotFound))
}

/// The cgroup v2 limit: the smallest `memory.max` from `dir` up to `top`, the
/// mount point, inclusive.
fn read_v2_limit(top: &Path, dir: &Path) -> Result<CgroupLimit, String> {
    let mut smallest: Option<u64> = None;
    let mut found_any = false;
    let mut level = Some(dir);
    while let Some(current) = level {
        let file = current.join("memory.max");
        if let Some(text) = read_optional(&file)? {
            found_any = true;
            if let Some(bytes) =
                parse_memory_max(&text).map_err(|e| format!("{}: {e}", file.display()))?
            {
                smallest = Some(smallest.map_or(bytes, |s| s.min(bytes)));
            }
        }
        if current == top {
            break;
        }
        level = current.parent().filter(|parent| parent.starts_with(top));
    }
    Ok(match (smallest, found_any) {
        (Some(bytes), _) => CgroupLimit::Limited(bytes, PodMemoryLimitSource::CgroupV2),
        (None, true) => CgroupLimit::Reference(ReferenceReason::Unlimited),
        (None, false) => CgroupLimit::Reference(ReferenceReason::NotFound),
    })
}

/// The cgroup v1 limit in `dir`: `memory.stat`'s `hierarchical_memory_limit`,
/// else `memory.limit_in_bytes`.
fn read_v1_limit(dir: &Path) -> Result<CgroupLimit, String> {
    let stat_file = dir.join("memory.stat");
    let from_stat = match read_optional(&stat_file)? {
        Some(text) => parse_hierarchical_memory_limit(&text)
            .map_err(|e| format!("{}: {e}", stat_file.display()))?,
        None => None,
    };
    let bytes = match from_stat {
        Some(bytes) => bytes,
        None => {
            let file = dir.join("memory.limit_in_bytes");
            match read_optional(&file)? {
                Some(text) => {
                    parse_bytes(text.trim()).map_err(|e| format!("{}: {e}", file.display()))?
                }
                None => return Ok(CgroupLimit::Reference(ReferenceReason::NotFound)),
            }
        }
    };
    Ok(if bytes >= CGROUP_V1_UNLIMITED_BYTES {
        CgroupLimit::Reference(ReferenceReason::Unlimited)
    } else {
        CgroupLimit::Limited(bytes, PodMemoryLimitSource::CgroupV1)
    })
}

/// The host's physical memory in bytes, from `/proc/meminfo`'s `MemTotal`
/// (in kB); `None` when it cannot be read.
fn read_host_memory(root: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(under(root, "/proc/meminfo")).ok()?;
    parse_meminfo_total(&text)
}

/// The process's cgroup memberships, from `/proc/self/cgroup`.
#[derive(Debug, Default, PartialEq, Eq)]
struct Memberships {
    /// The path of the unified (`0::`) hierarchy.
    v2: Option<String>,
    /// The path of the v1 hierarchy carrying the `memory` controller.
    v1_memory: Option<String>,
}

/// Parse `/proc/self/cgroup`: `hierarchy-id:controllers:path` per line.
fn parse_proc_cgroup(text: &str) -> Memberships {
    let mut memberships = Memberships::default();
    for line in text.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if id == "0" && controllers.is_empty() {
            memberships.v2 = Some(path.to_owned());
        } else if controllers.split(',').any(|c| c == "memory") {
            memberships.v1_memory = Some(path.to_owned());
        }
    }
    memberships
}

/// One `/proc/self/mountinfo` mount of a cgroup filesystem.
#[derive(Debug, PartialEq, Eq)]
struct CgroupMount {
    /// The path inside the cgroup filesystem that is mounted (field 4).
    root: String,
    /// Where it is mounted (field 5).
    mount_point: String,
    /// `cgroup2` or `cgroup`.
    fstype: String,
    /// The super options (the last field), which name a v1 hierarchy's
    /// controllers.
    super_options: String,
}

impl CgroupMount {
    fn is_v1_memory(&self) -> bool {
        self.fstype == "cgroup" && self.super_options.split(',').any(|o| o == "memory")
    }

    /// The directory of cgroup `path` under this mount, as an absolute path
    /// (before [`under`]). A path outside the mounted root, or one climbing
    /// out of it (`..`, as a cgroup namespace can show), maps to the mount
    /// point itself.
    fn cgroup_dir(&self, path: &str) -> String {
        let mount_root = self.root.trim_end_matches('/');
        let relative = if mount_root.is_empty() {
            Some(path)
        } else {
            path.strip_prefix(mount_root)
                .filter(|rest| rest.is_empty() || rest.starts_with('/'))
        };
        let relative = relative
            .filter(|rest| !rest.split('/').any(|part| part == ".."))
            .unwrap_or("")
            .trim_matches('/');
        if relative.is_empty() {
            self.mount_point.clone()
        } else {
            format!("{}/{relative}", self.mount_point.trim_end_matches('/'))
        }
    }
}

/// Parse the cgroup mounts out of `/proc/self/mountinfo`.
fn parse_mountinfo(text: &str) -> Vec<CgroupMount> {
    text.lines()
        .filter_map(|line| {
            let (head, tail) = line.split_once(" - ")?;
            let head: Vec<&str> = head.split(' ').collect();
            let mut tail = tail.split(' ');
            let fstype = tail.next()?;
            let _source = tail.next()?;
            let super_options = tail.next().unwrap_or("");
            if fstype != "cgroup2" && fstype != "cgroup" {
                return None;
            }
            Some(CgroupMount {
                root: unescape_mount_field(head.get(3)?),
                mount_point: unescape_mount_field(head.get(4)?),
                fstype: fstype.to_owned(),
                super_options: super_options.to_owned(),
            })
        })
        .collect()
}

/// Undo mountinfo's octal escapes (`\040` for a space, and so on).
fn unescape_mount_field(field: &str) -> String {
    let bytes = field.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 4 <= bytes.len() {
            let digits = &bytes[i + 1..i + 4];
            if digits.iter().all(|b| (b'0'..=b'7').contains(b)) {
                let value = digits
                    .iter()
                    .fold(0u32, |acc, b| acc * 8 + u32::from(b - b'0'));
                if let Ok(value) = u8::try_from(value) {
                    out.push(value);
                    i += 4;
                    continue;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A whole number of bytes.
fn parse_bytes(text: &str) -> Result<u64, String> {
    text.parse::<u64>()
        .map_err(|e| format!("{text:?} is not a byte count: {e}"))
}

/// A cgroup v2 `memory.max`: `Ok(None)` for `max` (no limit at that level),
/// `Ok(Some(bytes))` for a number.
fn parse_memory_max(text: &str) -> Result<Option<u64>, String> {
    let value = text.trim();
    if value == "max" {
        Ok(None)
    } else {
        parse_bytes(value).map(Some)
    }
}

/// `hierarchical_memory_limit` from a cgroup v1 `memory.stat`; `Ok(None)`
/// when the file has no such line.
fn parse_hierarchical_memory_limit(text: &str) -> Result<Option<u64>, String> {
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("hierarchical_memory_limit ") {
            return parse_bytes(value.trim()).map(Some);
        }
    }
    Ok(None)
}

/// `MemTotal` from `/proc/meminfo`, in bytes.
fn parse_meminfo_total(text: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
    let mut parts = line["MemTotal:".len()..].split_whitespace();
    let kib: u64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some("kB") | None => kib.checked_mul(1024),
        Some(_) => None,
    }
}

/// The cgroup reader on fixture trees: each test writes the `/proc` files and
/// the cgroup mount's files a container, host or hybrid host shows under a
/// temporary root and reads its limit through [`read_pod_memory_limit`], the
/// production entry, with that root in place of `/`. One test reads the
/// machine's own cgroup, so the parse is exercised on a real kernel's files
/// too. The reference fallback's plan (every reason gives the reference
/// pod's budgets) is checked here as well, since the fixtures live here.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::memory_plan::{PlanInputs, PodMemoryPlan};

    const GIB: u64 = 1024 * 1024 * 1024;

    /// A `0::/` membership: the process at the root of its cgroup namespace.
    const V2_ROOT_MEMBERSHIP: &str = "0::/\n";

    /// A `cgroup2` mount of the hierarchy's root at `/sys/fs/cgroup`, the
    /// usual container view.
    const V2_MOUNT: &str =
        "30 23 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup2 rw\n";

    /// A host with 64 GiB of memory, above every cgroup limit below that is
    /// not about the clamp.
    const MEMINFO_64_GIB: &str = "MemTotal:       67108864 kB\nMemFree:        1024 kB\n";

    /// A v1 `memory` hierarchy mounted at `/sys/fs/cgroup/memory`, beside a
    /// `cpu,cpuacct` one that must not be taken for it.
    const V1_MOUNTS: &str = "\
40 30 0:35 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid,nodev,noexec,relatime - cgroup cgroup rw,cpu,cpuacct
41 30 0:36 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime - cgroup cgroup rw,memory
";

    /// The v1 memberships of a container under `/docker/abc`.
    const V1_MEMBERSHIPS: &str = "4:cpu,cpuacct:/docker/abc\n5:memory:/docker/abc\n";

    /// A temporary root holding `files` (`path relative to /`, contents).
    fn tree(files: &[(&str, &str)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("create the fixture root");
        for (path, text) in files {
            let path = root.path().join(path);
            std::fs::create_dir_all(path.parent().expect("a fixture file has a parent"))
                .expect("create a fixture directory");
            std::fs::write(&path, text).expect("write a fixture file");
        }
        root
    }

    /// A cgroup v2 container at its namespace root with `memory_max` in
    /// `/sys/fs/cgroup/memory.max` and a 64 GiB host.
    fn v2_container(memory_max: &str) -> tempfile::TempDir {
        tree(&[
            ("proc/self/cgroup", V2_ROOT_MEMBERSHIP),
            ("proc/self/mountinfo", V2_MOUNT),
            ("proc/meminfo", MEMINFO_64_GIB),
            ("sys/fs/cgroup/memory.max", memory_max),
        ])
    }

    fn read(root: &tempfile::TempDir) -> PodMemoryLimit {
        read_pod_memory_limit(root.path())
    }

    fn limited(bytes: u64, source: PodMemoryLimitSource) -> PodMemoryLimit {
        PodMemoryLimit {
            bytes,
            source,
            reason: None,
        }
    }

    /// The `Unreadable` detail of `limit`, failing on any other reading.
    fn unreadable_detail(limit: &PodMemoryLimit) -> &str {
        assert_eq!(limit.source, PodMemoryLimitSource::Reference, "{limit:?}");
        assert_eq!(limit.bytes, REFERENCE_POD_MEMORY_LIMIT_BYTES, "{limit:?}");
        match &limit.reason {
            Some(ReferenceReason::Unreadable(detail)) => detail,
            other => panic!("expected an unreadable reason, got {other:?}"),
        }
    }

    #[test]
    fn read_pod_memory_limit_v2_number_at_the_mount_root_is_the_limit() {
        let root = v2_container("5368709120\n");
        assert_eq!(
            read(&root),
            limited(5 * GIB, PodMemoryLimitSource::CgroupV2)
        );
    }

    #[test]
    fn read_pod_memory_limit_v2_max_is_unlimited_and_sizes_the_reference_pod() {
        let root = v2_container("max\n");
        assert_eq!(
            read(&root),
            PodMemoryLimit::reference(Some(ReferenceReason::Unlimited))
        );
    }

    /// A nested cgroup: the smallest `memory.max` from the process's cgroup up
    /// to the mount wins, `max` at a level sets nothing there, and the mount
    /// root, which has no `memory.max`, is skipped rather than read as an
    /// error. The smallest sits in the middle, so neither the leaf alone nor
    /// the top level alone gives it.
    #[test]
    fn read_pod_memory_limit_v2_nested_takes_the_smallest_ancestor_and_skips_a_missing_root() {
        let root = tree(&[
            ("proc/self/cgroup", "0::/kubepods/pod1/ctr\n"),
            ("proc/self/mountinfo", V2_MOUNT),
            ("proc/meminfo", MEMINFO_64_GIB),
            ("sys/fs/cgroup/kubepods/memory.max", "8589934592\n"),
            ("sys/fs/cgroup/kubepods/pod1/memory.max", "6442450944\n"),
            ("sys/fs/cgroup/kubepods/pod1/ctr/memory.max", "max\n"),
        ]);
        assert!(
            !root.path().join("sys/fs/cgroup/memory.max").exists(),
            "fixture self-check: the mount root has no memory.max"
        );
        assert_eq!(
            read(&root),
            limited(6 * GIB, PodMemoryLimitSource::CgroupV2)
        );
    }

    /// A mount of a sub-tree (a cgroup namespace's view): the membership path
    /// is resolved below the mounted root, and one climbing out of it (`..`)
    /// maps to the mount point.
    #[test]
    fn read_pod_memory_limit_v2_resolves_the_membership_below_a_mounted_sub_tree() {
        let sub_tree_mount =
            "30 23 0:26 /kubepods/pod1 /sys/fs/cgroup rw,nosuid - cgroup2 cgroup2 rw\n";
        let root = tree(&[
            ("proc/self/cgroup", "0::/kubepods/pod1/ctr\n"),
            ("proc/self/mountinfo", sub_tree_mount),
            ("proc/meminfo", MEMINFO_64_GIB),
            ("sys/fs/cgroup/memory.max", "4294967296\n"),
            ("sys/fs/cgroup/ctr/memory.max", "3221225472\n"),
        ]);
        assert_eq!(
            read(&root),
            limited(3 * GIB, PodMemoryLimitSource::CgroupV2)
        );

        let escaped = tree(&[
            ("proc/self/cgroup", "0::/../../other\n"),
            ("proc/self/mountinfo", sub_tree_mount),
            ("proc/meminfo", MEMINFO_64_GIB),
            ("sys/fs/cgroup/memory.max", "4294967296\n"),
        ]);
        assert_eq!(
            read(&escaped),
            limited(4 * GIB, PodMemoryLimitSource::CgroupV2)
        );
    }

    #[test]
    fn read_pod_memory_limit_v1_limit_in_bytes_is_the_limit() {
        let root = tree(&[
            ("proc/self/cgroup", V1_MEMBERSHIPS),
            ("proc/self/mountinfo", V1_MOUNTS),
            ("proc/meminfo", MEMINFO_64_GIB),
            (
                "sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
                "4294967296\n",
            ),
        ]);
        assert_eq!(
            read(&root),
            limited(4 * GIB, PodMemoryLimitSource::CgroupV1)
        );
    }

    /// The kernel's v1 "unlimited" reads back just under 2^63; anything at or
    /// above 2^62 is that sentinel, never a real limit.
    #[test]
    fn read_pod_memory_limit_v1_sentinel_at_or_above_2_pow_62_is_unlimited() {
        for sentinel in ["9223372036854771712", "4611686018427387904"] {
            let root = tree(&[
                ("proc/self/cgroup", V1_MEMBERSHIPS),
                ("proc/self/mountinfo", V1_MOUNTS),
                ("proc/meminfo", MEMINFO_64_GIB),
                (
                    "sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
                    sentinel,
                ),
            ]);
            assert_eq!(
                read(&root),
                PodMemoryLimit::reference(Some(ReferenceReason::Unlimited)),
                "{sentinel}"
            );
        }
    }

    /// `memory.stat`'s `hierarchical_memory_limit`, which counts the
    /// ancestors' limits, wins over the cgroup's own `memory.limit_in_bytes`
    /// (here the unlimited sentinel, so only the hierarchical value can give a
    /// limit at all).
    #[test]
    fn read_pod_memory_limit_v1_prefers_the_hierarchical_memory_limit() {
        let root = tree(&[
            ("proc/self/cgroup", V1_MEMBERSHIPS),
            ("proc/self/mountinfo", V1_MOUNTS),
            ("proc/meminfo", MEMINFO_64_GIB),
            (
                "sys/fs/cgroup/memory/docker/abc/memory.stat",
                "cache 0\nrss 1024\nhierarchical_memory_limit 3221225472\nhierarchical_memsw_limit 9223372036854771712\n",
            ),
            (
                "sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
                "9223372036854771712\n",
            ),
        ]);
        assert_eq!(
            read(&root),
            limited(3 * GIB, PodMemoryLimitSource::CgroupV1)
        );
    }

    /// A v1 `memory.stat` without a `hierarchical_memory_limit` line (older
    /// kernels, or no hierarchy accounting) falls back to the cgroup's own
    /// `memory.limit_in_bytes`.
    #[test]
    fn read_pod_memory_limit_v1_stat_without_the_hierarchical_limit_falls_back_to_limit_in_bytes() {
        let root = tree(&[
            ("proc/self/cgroup", V1_MEMBERSHIPS),
            ("proc/self/mountinfo", V1_MOUNTS),
            ("proc/meminfo", MEMINFO_64_GIB),
            (
                "sys/fs/cgroup/memory/docker/abc/memory.stat",
                "cache 0\nrss 1024\nhierarchical_memsw_limit 9223372036854771712\n",
            ),
            (
                "sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
                "2147483648\n",
            ),
        ]);
        assert_eq!(
            read(&root),
            limited(2 * GIB, PodMemoryLimitSource::CgroupV1)
        );
    }

    /// A v1 memory cgroup with neither `memory.stat` nor
    /// `memory.limit_in_bytes` is `not_found`, not a limit or an error.
    #[test]
    fn read_pod_memory_limit_v1_cgroup_without_either_file_is_not_found() {
        let root = tree(&[
            ("proc/self/cgroup", V1_MEMBERSHIPS),
            ("proc/self/mountinfo", V1_MOUNTS),
            ("proc/meminfo", MEMINFO_64_GIB),
            ("sys/fs/cgroup/memory/docker/abc/cgroup.procs", "1\n"),
        ]);
        assert_eq!(
            read(&root),
            PodMemoryLimit::reference(Some(ReferenceReason::NotFound))
        );
    }

    /// A hybrid host mounts both: the memory controller lives on the v1
    /// hierarchy, so its limit is the one the kernel enforces, whatever the
    /// unified hierarchy's `memory.max` file says.
    #[test]
    fn read_pod_memory_limit_hybrid_host_reads_the_memory_controllers_v1_hierarchy() {
        let mounts = format!(
            "{V1_MOUNTS}29 23 0:26 / /sys/fs/cgroup/unified rw,nosuid - cgroup2 cgroup2 rw\n"
        );
        let root = tree(&[
            ("proc/self/cgroup", "0::/docker/abc\n5:memory:/docker/abc\n"),
            ("proc/self/mountinfo", &mounts),
            ("proc/meminfo", MEMINFO_64_GIB),
            (
                "sys/fs/cgroup/unified/docker/abc/memory.max",
                "1073741824\n",
            ),
            (
                "sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
                "4294967296\n",
            ),
        ]);
        assert_eq!(
            read(&root),
            limited(4 * GIB, PodMemoryLimitSource::CgroupV1)
        );
    }

    /// No `/proc` files (not Linux, no procfs), and a cgroup mount without a
    /// single memory file (no memory controller), are both `not_found`.
    #[test]
    fn read_pod_memory_limit_missing_files_are_not_found() {
        let empty = tree(&[]);
        assert_eq!(
            read(&empty),
            PodMemoryLimit::reference(Some(ReferenceReason::NotFound))
        );
        let no_mountinfo = tree(&[("proc/self/cgroup", V2_ROOT_MEMBERSHIP)]);
        assert_eq!(
            read(&no_mountinfo),
            PodMemoryLimit::reference(Some(ReferenceReason::NotFound))
        );
        let no_memory_files = tree(&[
            ("proc/self/cgroup", "0::/a/b\n"),
            ("proc/self/mountinfo", V2_MOUNT),
            ("sys/fs/cgroup/a/b/cpu.max", "max 100000\n"),
        ]);
        assert_eq!(
            read(&no_memory_files),
            PodMemoryLimit::reference(Some(ReferenceReason::NotFound))
        );
        let no_cgroup_mount = tree(&[
            ("proc/self/cgroup", V2_ROOT_MEMBERSHIP),
            (
                "proc/self/mountinfo",
                "22 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n",
            ),
        ]);
        assert_eq!(
            read(&no_cgroup_mount),
            PodMemoryLimit::reference(Some(ReferenceReason::NotFound))
        );
    }

    /// A malformed limit file, and one that cannot be read (a directory in
    /// its place), are `unreadable`, naming the file: never a guessed number.
    #[test]
    fn read_pod_memory_limit_malformed_or_unreadable_file_is_unreadable_naming_it() {
        for (path, text) in [
            ("sys/fs/cgroup/memory.max", "lots\n"),
            ("sys/fs/cgroup/memory.max", "-1\n"),
        ] {
            let root = tree(&[
                ("proc/self/cgroup", V2_ROOT_MEMBERSHIP),
                ("proc/self/mountinfo", V2_MOUNT),
                (path, text),
            ]);
            let limit = read(&root);
            assert!(
                unreadable_detail(&limit).contains("memory.max"),
                "{text:?}: {limit:?}"
            );
        }
        let v1_malformed = tree(&[
            ("proc/self/cgroup", V1_MEMBERSHIPS),
            ("proc/self/mountinfo", V1_MOUNTS),
            (
                "sys/fs/cgroup/memory/docker/abc/memory.stat",
                "hierarchical_memory_limit twelve\n",
            ),
        ]);
        let limit = read(&v1_malformed);
        assert!(
            unreadable_detail(&limit).contains("memory.stat"),
            "{limit:?}"
        );

        let directory = tree(&[
            ("proc/self/cgroup", V2_ROOT_MEMBERSHIP),
            ("proc/self/mountinfo", V2_MOUNT),
        ]);
        std::fs::create_dir_all(directory.path().join("sys/fs/cgroup/memory.max"))
            .expect("a directory where memory.max should be");
        let limit = read(&directory);
        assert!(
            unreadable_detail(&limit).contains("memory.max"),
            "{limit:?}"
        );
    }

    /// A limit file the process may not read (EACCES) is `unreadable`. Skipped
    /// when the test runs as a user whom the permission bits do not stop (root):
    /// the directory case above covers the same read-error branch everywhere.
    #[cfg(unix)]
    #[test]
    fn read_pod_memory_limit_permission_denied_is_unreadable() {
        use std::os::unix::fs::PermissionsExt;
        let root = v2_container("5368709120\n");
        let file = root.path().join("sys/fs/cgroup/memory.max");
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o000))
            .expect("make memory.max unreadable");
        if std::fs::read(&file).is_ok() {
            eprintln!("skipped: the permission bits do not stop this user (root)");
            return;
        }
        let limit = read(&root);
        let detail = unreadable_detail(&limit).to_owned();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644))
            .expect("restore memory.max for cleanup");
        assert!(detail.contains("memory.max"), "{detail}");
    }

    /// A cgroup limit above the host's memory is sized to the host; a host
    /// whose `MemTotal` cannot be read leaves the cgroup limit standing.
    #[test]
    fn read_pod_memory_limit_clamps_to_the_hosts_memory_below_the_cgroup_limit() {
        let root = tree(&[
            ("proc/self/cgroup", V2_ROOT_MEMBERSHIP),
            ("proc/self/mountinfo", V2_MOUNT),
            ("proc/meminfo", "MemTotal:       16777216 kB\n"),
            ("sys/fs/cgroup/memory.max", "68719476736\n"),
        ]);
        assert_eq!(
            read(&root),
            limited(16 * GIB, PodMemoryLimitSource::HostMemory)
        );

        let no_meminfo = tree(&[
            ("proc/self/cgroup", V2_ROOT_MEMBERSHIP),
            ("proc/self/mountinfo", V2_MOUNT),
            ("sys/fs/cgroup/memory.max", "68719476736\n"),
        ]);
        assert_eq!(
            read(&no_meminfo),
            limited(64 * GIB, PodMemoryLimitSource::CgroupV2)
        );
    }

    /// The machine's own cgroup, read as production reads it: whatever this
    /// runner's limit is, the read classifies it without panicking, a limit
    /// carries no reason, and the reference size carries one.
    #[test]
    fn read_pod_memory_limit_of_this_machine_classifies_without_panicking() {
        let limit = read_pod_memory_limit(Path::new("/"));
        assert!(limit.bytes > 0, "{limit:?}");
        match limit.source {
            PodMemoryLimitSource::Reference => {
                assert_eq!(limit.bytes, REFERENCE_POD_MEMORY_LIMIT_BYTES);
                assert!(limit.reason.is_some(), "{limit:?}");
            }
            _ => assert_eq!(limit.reason, None, "{limit:?}"),
        }
        if !cfg!(target_os = "linux") {
            assert_eq!(
                limit,
                PodMemoryLimit::reference(Some(ReferenceReason::NotFound))
            );
        }
    }

    #[test]
    fn parse_mountinfo_unescapes_octal_escapes_in_the_mount_point() {
        let mounts = parse_mountinfo(
            "30 23 0:26 / /sys/fs/cgroup\\040x rw - cgroup2 cgroup2 rw\n\
             31 23 0:27 / /proc rw - proc proc rw\n",
        );
        assert_eq!(mounts.len(), 1, "{mounts:?}");
        assert_eq!(mounts[0].mount_point, "/sys/fs/cgroup x");
    }

    /// W7: a pod whose limit cannot be read — unlimited, no cgroup files, or
    /// an unreadable file — gets exactly the reference pod's plan: every
    /// budget and every term of the line and the margin, and the smallest
    /// limit the configuration closes at, with the source `reference` and the
    /// reason that put it there.
    #[test]
    fn reference_fallback_for_every_reason_gives_the_reference_pods_plan() {
        let reference = PodMemoryPlan::reference();
        let directory = tree(&[
            ("proc/self/cgroup", V2_ROOT_MEMBERSHIP),
            ("proc/self/mountinfo", V2_MOUNT),
        ]);
        std::fs::create_dir_all(directory.path().join("sys/fs/cgroup/memory.max"))
            .expect("a directory where memory.max should be");
        let cases = [
            ("unlimited", v2_container("max\n")),
            ("not_found", tree(&[])),
            ("unreadable", directory),
        ];
        for (reason, root) in &cases {
            let limit = read(root);
            assert_eq!(limit.source, PodMemoryLimitSource::Reference, "{reason}");
            assert!(
                limit.reason_text().starts_with(reason),
                "{reason}: {}",
                limit.reason_text()
            );
            let plan = PodMemoryPlan::derive(limit, &PlanInputs::reference())
                .expect("the reference pod's plan closes");
            assert_eq!(plan.limit().bytes, REFERENCE_POD_MEMORY_LIMIT_BYTES);
            assert_eq!(
                (
                    plan.max_subscriptions(),
                    plan.upload_budget_mib(),
                    plan.catchup_budget_mib(),
                    plan.catchup_step_centi(),
                    plan.structural_cache_budget_mib(),
                    plan.retirement_budget_bytes(),
                    plan.minimum_limit_mib(),
                ),
                (
                    reference.max_subscriptions(),
                    reference.upload_budget_mib(),
                    reference.catchup_budget_mib(),
                    reference.catchup_step_centi(),
                    reference.structural_cache_budget_mib(),
                    reference.retirement_budget_bytes(),
                    reference.minimum_limit_mib(),
                ),
                "{reason}: the budgets"
            );
            assert_eq!(plan.line_terms(), reference.line_terms(), "{reason}");
            assert_eq!(plan.margin_terms(), reference.margin_terms(), "{reason}");
        }
        assert_eq!(
            (
                reference.max_subscriptions(),
                reference.upload_budget_mib(),
                reference.catchup_budget_mib(),
                reference.structural_cache_budget_mib(),
            ),
            (4_096, 384, 744, 256),
            "the reference pod's budgets are the chart's 5 GiB pod's"
        );
    }
}
