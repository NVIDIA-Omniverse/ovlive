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

// Guard tests: scripts/guard-tests.sh runs this whole binary by name; rename it there too.

//! The dependency fence between the conformance suite and the service.
//!
//! The suite ships as its own image, built on the service's builder, and the
//! service image's release build (`cargo build --release --workspace`)
//! compiles the suite's crate beside the service. So the suite must stay
//! independent of the service in both directions, must not bring a
//! third-party crate into its image that the service image does not already
//! ship, and must not change what the service build resolves. Four rules:
//!
//! - **A1:** the workspace members the suite reaches are exactly
//!   `live-edit-conformance` and `oodf-ffi`: no service code.
//! - **A2:** `live-edit`'s tree does not contain `live-edit-conformance`:
//!   the service binary does not carry the suite. Dev-dependency edges are
//!   not counted, because they do not ship.
//! - **B:** every third-party package (name and version) the suite reaches
//!   is in `live-edit`'s tree, the set the service image ships. "Third-party"
//!   is anything not a workspace member, the patched `fred` path package
//!   included.
//! - **C:** the suite asks for no feature the service build does not
//!   already resolve. Every normal or build dependency it declares, on a
//!   workspace member as on a third-party crate, is declared by `live-edit`
//!   or `live-edit-scaler` with the same kind, the same `default-features`
//!   and a superset of its features; failing that, the suite declares it
//!   with `default-features = false` and no features (and Rules A1 and B
//!   hold it to the service's set). So `oodf-ffi` is held to `live-edit`'s
//!   declaration: default features, none added, never `internal-bridge`.
//!   The suite's own `[features]` table is empty, since a feature there (a
//!   `default` one above all) could switch on a dependency's feature that no
//!   declaration shows.
//!
//! Inputs, read from the checkout with the toolchain's own cargo, offline:
//! `cargo tree -e normal,build --prefix none -f '{p}'` of each of the two
//! packages on the release image's linux-gnu targets (the targets
//! `scripts/check-measurement-features.sh` uses, for the same reason), and
//! `cargo metadata --no-deps` for the workspace members and the declared
//! dependencies.
//!
//! Each rule has a negative control: the real input with one planted line or
//! declaration must fail that rule, naming the planted package. A rule
//! that stops matching its input then fails here instead of passing
//! silently.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde_json::Value;

/// The suite's package.
const SUITE: &str = "live-edit-conformance";
/// The service's package: its tree is what the service image ships.
const SERVICE: &str = "live-edit";
/// The workspace members the suite may reach (Rule A1), itself included.
const SUITE_MEMBERS: [&str; 2] = [SUITE, "oodf-ffi"];
/// The members whose declarations can vouch for a suite declaration (Rule C).
const SERVICE_DECLARERS: [&str; 2] = [SERVICE, "live-edit-scaler"];
/// The release image's targets: its `rust:1.94-slim` builder builds them
/// natively, and a Linux build fetches every package they need.
const RELEASE_TARGETS: [&str; 2] = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"];

/// One package of a `cargo tree` listing.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Package {
    name: String,
    /// As `cargo tree` prints it, `v` included.
    version: String,
}

impl std::fmt::Display for Package {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.name, self.version)
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate is one directory below the workspace root")
        .to_path_buf()
}

/// Runs the toolchain's cargo in the workspace root and returns its stdout;
/// any failure fails the test, since the fence could not run.
fn cargo(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO"))
        .args(args)
        .current_dir(workspace_root())
        .output()
        .unwrap_or_else(|error| panic!("cannot run cargo {args:?}: {error}"));
    assert!(
        output.status.success(),
        "cargo {args:?} failed ({}), so the fence could not run. Is Cargo.lock \
         current, and has a Linux build fetched the packages (it runs offline)?\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("cargo prints UTF-8")
}

/// `cargo tree` of `package`'s normal and build edges on the release
/// targets, one package per line.
fn tree_output(package: &str) -> String {
    let mut args = vec!["tree", "--locked", "--offline", "-e", "normal,build"];
    for target in RELEASE_TARGETS {
        args.extend(["--target", target]);
    }
    args.extend(["-p", package, "--prefix", "none", "-f", "{p}"]);
    cargo(&args)
}

fn suite_tree_output() -> &'static str {
    static OUTPUT: OnceLock<String> = OnceLock::new();
    OUTPUT.get_or_init(|| tree_output(SUITE))
}

fn service_tree_output() -> &'static str {
    static OUTPUT: OnceLock<String> = OnceLock::new();
    OUTPUT.get_or_init(|| tree_output(SERVICE))
}

fn metadata() -> &'static Value {
    static METADATA: OnceLock<Value> = OnceLock::new();
    METADATA.get_or_init(|| {
        let output = cargo(&[
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--offline",
        ]);
        serde_json::from_str(&output).expect("cargo metadata prints JSON")
    })
}

/// The packages of a `cargo tree --prefix none -f '{p}'` listing whose root
/// is `root`. Each package line is `<name> v<version>` and may carry a
/// source in parentheses and a `(*)` or `(proc-macro)` marker after it;
/// `[build-dependencies]` headers and blank lines are skipped. A line of any
/// other shape, or a listing without `root`, is an error: the fence's
/// reading no longer matches what cargo prints.
fn parse_tree(output: &str, root: &str) -> Result<BTreeSet<Package>, String> {
    let mut packages = BTreeSet::new();
    for line in output.lines().map(str::trim) {
        if line.is_empty() || (line.starts_with('[') && line.ends_with(']')) {
            continue;
        }
        let mut words = line.split_whitespace();
        match (words.next(), words.next()) {
            (Some(name), Some(version)) if version.starts_with('v') => {
                packages.insert(Package {
                    name: name.to_string(),
                    version: version.to_string(),
                });
            }
            _ => return Err(format!("unexpected cargo tree line {line:?}")),
        }
    }
    if !packages.iter().any(|package| package.name == root) {
        return Err(format!(
            "the cargo tree listing does not contain its root {root}, so the \
             fence's reading no longer matches cargo tree's output"
        ));
    }
    Ok(packages)
}

/// The workspace members' package names, from `cargo metadata --no-deps`.
fn workspace_members(metadata: &Value) -> BTreeSet<String> {
    let ids: BTreeSet<&str> = metadata["workspace_members"]
        .as_array()
        .expect("metadata lists workspace_members")
        .iter()
        .map(|id| id.as_str().expect("a member id is a string"))
        .collect();
    let members: BTreeSet<String> = packages(metadata)
        .iter()
        .filter(|package| ids.contains(package["id"].as_str().unwrap_or_default()))
        .map(|package| {
            package["name"]
                .as_str()
                .expect("a package has a name")
                .to_string()
        })
        .collect();
    for name in [SUITE, SERVICE] {
        assert!(
            members.contains(name),
            "cargo metadata does not list {name} as a workspace member"
        );
    }
    members
}

fn packages(metadata: &Value) -> &Vec<Value> {
    metadata["packages"]
        .as_array()
        .expect("metadata lists packages")
}

fn package<'a>(metadata: &'a Value, name: &str) -> &'a Value {
    packages(metadata)
        .iter()
        .find(|package| package["name"] == name)
        .unwrap_or_else(|| panic!("cargo metadata has no package {name}"))
}

/// Rule A1: the workspace members in the suite's tree are exactly
/// [`SUITE_MEMBERS`].
fn rule_a1(members: &BTreeSet<String>, suite_tree: &BTreeSet<Package>) -> Result<(), String> {
    let reached: BTreeSet<&str> = suite_tree
        .iter()
        .filter(|package| members.contains(&package.name))
        .map(|package| package.name.as_str())
        .collect();
    let allowed: BTreeSet<&str> = SUITE_MEMBERS.into_iter().collect();
    if reached == allowed {
        return Ok(());
    }
    Err(format!(
        "Rule A1: {SUITE} must reach exactly the workspace members {allowed:?}, \
         and reaches {reached:?}. The suite checks the service from outside: it \
         may use oodf-ffi's public API and its own generated proto code, never \
         service code."
    ))
}

/// Rule A2: the service's tree does not contain the suite.
fn rule_a2(service_tree: &BTreeSet<Package>) -> Result<(), String> {
    let carried: Vec<String> = service_tree
        .iter()
        .filter(|package| package.name == SUITE)
        .map(Package::to_string)
        .collect();
    if carried.is_empty() {
        return Ok(());
    }
    Err(format!(
        "Rule A2: {SERVICE}'s normal and build tree contains {carried:?}. The \
         service binary must not carry the suite; it ships as its own image."
    ))
}

/// Rule B: every non-member package in the suite's tree is in the
/// service's tree, at the same version.
fn rule_b(
    members: &BTreeSet<String>,
    suite_tree: &BTreeSet<Package>,
    service_tree: &BTreeSet<Package>,
) -> Result<(), String> {
    let missing: Vec<String> = suite_tree
        .iter()
        .filter(|package| !members.contains(&package.name))
        .filter(|package| !service_tree.contains(package))
        .map(Package::to_string)
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "Rule B: {SUITE} reaches third-party packages that {SERVICE}'s tree does \
         not contain: {missing:?}. The suite's image may ship no third-party \
         component the service image does not already ship."
    ))
}

/// One declared dependency, from `cargo metadata`.
#[derive(Debug)]
struct Declaration {
    name: String,
    /// `None` for a normal dependency, else `build` or `dev`.
    kind: Option<String>,
    default_features: bool,
    features: BTreeSet<String>,
}

fn declarations(package: &Value) -> Vec<Declaration> {
    package["dependencies"]
        .as_array()
        .expect("a package lists its dependencies")
        .iter()
        .map(|dependency| Declaration {
            name: dependency["name"]
                .as_str()
                .expect("a dependency has a name")
                .to_string(),
            kind: dependency["kind"].as_str().map(str::to_string),
            default_features: dependency["uses_default_features"]
                .as_bool()
                .expect("a dependency states uses_default_features"),
            features: dependency["features"]
                .as_array()
                .expect("a dependency lists its features")
                .iter()
                .map(|feature| feature.as_str().expect("a feature is a string").to_string())
                .collect(),
        })
        .collect()
}

/// Rule C: the suite's own `[features]` table is empty, and each normal or
/// build dependency it declares, on a workspace member (`oodf-ffi`) as on a
/// third-party crate, is vouched for by a service declaration or asks for
/// nothing.
fn rule_c(metadata: &Value) -> Result<(), String> {
    let mut violations = Vec::new();
    let suite = package(metadata, SUITE);
    let features: BTreeSet<&String> = suite["features"]
        .as_object()
        .expect("a package lists its features")
        .keys()
        .collect();
    if !features.is_empty() {
        violations.push(format!(
            "{SUITE} declares the features {features:?}; it must declare none"
        ));
    }
    let service: Vec<Declaration> = SERVICE_DECLARERS
        .iter()
        .flat_map(|name| declarations(package(metadata, name)))
        .collect();
    for declared in declarations(suite) {
        // Members are held too: a feature on `oodf-ffi` (`internal-bridge`)
        // would be unified into the service binary just like a third-party one.
        let shipped_kind = matches!(declared.kind.as_deref(), None | Some("build"));
        if !shipped_kind {
            continue;
        }
        let vouched = service.iter().any(|candidate| {
            candidate.name == declared.name
                && candidate.kind == declared.kind
                && candidate.default_features == declared.default_features
                && candidate.features.is_superset(&declared.features)
        });
        let asks_nothing = !declared.default_features && declared.features.is_empty();
        if !vouched && !asks_nothing {
            violations.push(format!(
                "{} ({} dependency, default-features {}, features {:?}) is not \
                 declared by {SERVICE_DECLARERS:?} with the same kind and \
                 default-features and at least these features",
                declared.name,
                declared.kind.as_deref().unwrap_or("normal"),
                declared.default_features,
                declared.features,
            ));
        }
    }
    if violations.is_empty() {
        return Ok(());
    }
    Err(format!(
        "Rule C: the service's `--workspace` release build unifies features \
         with {SUITE}, so a feature only the suite asks for would change the \
         shipped service binary. Violations:\n- {}",
        violations.join("\n- ")
    ))
}

fn suite_tree() -> BTreeSet<Package> {
    parse_tree(suite_tree_output(), SUITE).unwrap_or_else(|error| panic!("{error}"))
}

fn service_tree() -> BTreeSet<Package> {
    parse_tree(service_tree_output(), SERVICE).unwrap_or_else(|error| panic!("{error}"))
}

/// `output` with `line` added as one more package line.
fn planted(output: &str, line: &str) -> String {
    format!("{output}{line}\n")
}

/// Asserts `result` is a failure whose message names `needle`.
fn assert_fails_naming(result: Result<(), String>, needle: &str) {
    match result {
        Ok(()) => panic!("the rule passed input planted with {needle}"),
        Err(message) => assert!(
            message.contains(needle),
            "the rule failed, but its message does not name {needle}: {message}"
        ),
    }
}

#[test]
fn suite_reaches_no_workspace_member_but_oodf_ffi() {
    let members = workspace_members(metadata());
    rule_a1(&members, &suite_tree()).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn service_tree_does_not_contain_the_suite() {
    rule_a2(&service_tree()).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn every_third_party_package_of_the_suite_is_in_the_service_tree() {
    let members = workspace_members(metadata());
    let suite = suite_tree();
    // The rule must have something to compare: the suite reaches
    // third-party packages (tonic, tokio and the rest).
    assert!(
        suite.iter().any(|package| !members.contains(&package.name)),
        "the suite's tree holds no third-party package, so Rule B compared nothing"
    );
    rule_b(&members, &suite, &service_tree()).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn suite_declares_no_feature_the_service_build_does_not_resolve() {
    rule_c(metadata()).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn rule_a1_fails_when_the_suite_reaches_the_service() {
    let members = workspace_members(metadata());
    let tree = parse_tree(
        &planted(suite_tree_output(), "live-edit v0.1.1 (/build)"),
        SUITE,
    )
    .expect("the planted listing parses");
    assert_fails_naming(rule_a1(&members, &tree), "\"live-edit\"");
}

#[test]
fn rule_a2_fails_when_the_service_tree_contains_the_suite() {
    let tree = parse_tree(
        &planted(
            service_tree_output(),
            "live-edit-conformance v0.1.0 (/build/conformance_tests)",
        ),
        SERVICE,
    )
    .expect("the planted listing parses");
    assert_fails_naming(rule_a2(&tree), "live-edit-conformance v0.1.0");
}

#[test]
fn rule_b_fails_on_a_third_party_package_the_service_does_not_ship() {
    let members = workspace_members(metadata());
    let tree = parse_tree(&planted(suite_tree_output(), "clap v4.5.0"), SUITE)
        .expect("the planted listing parses");
    assert_fails_naming(rule_b(&members, &tree, &service_tree()), "clap v4.5.0");
}

/// The real metadata with `edit` applied to the suite's package.
fn edited_metadata(edit: impl FnOnce(&mut Value)) -> Value {
    let mut metadata = metadata().clone();
    let suite = metadata["packages"]
        .as_array_mut()
        .expect("metadata lists packages")
        .iter_mut()
        .find(|package| package["name"] == SUITE)
        .expect("metadata has the suite's package");
    edit(suite);
    metadata
}

/// The real metadata with `feature` added to the suite's normal declaration
/// of `dependency`.
fn metadata_with_dependency_feature(dependency: &str, feature: &str) -> Value {
    edited_metadata(|suite| {
        let declared = suite["dependencies"]
            .as_array_mut()
            .expect("the suite lists its dependencies")
            .iter_mut()
            .find(|declared| declared["name"] == dependency && declared["kind"].is_null())
            .unwrap_or_else(|| panic!("the suite declares {dependency}"));
        declared["features"]
            .as_array_mut()
            .unwrap_or_else(|| panic!("{dependency} lists its features"))
            .push(Value::from(feature));
    })
}

#[test]
fn rule_c_fails_on_a_dependency_feature_the_service_does_not_declare() {
    let metadata = metadata_with_dependency_feature("reqwest", "blocking");
    assert_fails_naming(rule_c(&metadata), "\"blocking\"");
}

#[test]
fn rule_c_fails_on_a_feature_of_a_workspace_member_dependency() {
    let metadata = metadata_with_dependency_feature("oodf-ffi", "internal-bridge");
    assert_fails_naming(rule_c(&metadata), "\"internal-bridge\"");
}

#[test]
fn rule_c_fails_on_a_feature_of_the_suite_itself() {
    let metadata = edited_metadata(|suite| {
        suite["features"]["default"] = Value::from(vec!["reqwest/blocking"]);
    });
    assert_fails_naming(rule_c(&metadata), "\"default\"");
}
