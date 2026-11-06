#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline disaster-recovery chart regressions. Requires Helm and PyYAML; never
contacts a cluster.

Setting `dr.restore.incident` or `dr.acceptLoss.incident` renders that
action's Job. These checks pin what such a render does, and what it leaves
alone:

* the Job's settings live outside `dr.*`: `checkpoint.restoreParallelism`
  renders into the settings ConfigMap as `LIVE_EDIT_DR_RESTORE_PARALLELISM`,
  with the binary's default, and a value that is not a whole number fails the
  render naming the key;
* a DR action, with or without the Job's other `dr.*` settings, changes
  nothing else: every other manifest renders byte for byte as without it,
  except the smoke-test hooks, which do not render while
  an action is set. So the settings ConfigMap and the server Deployment's
  `checksum/config` stay the same, and the fleet does not restart;
* the Job itself: its name, which always fits 63 characters; its command
  line, with the subcommands and confirmations read from the binary's source;
  its pod failure policy, with the exit codes it fails at once read from the
  binary's source too; the pod spec it shares with the server and the
  server-only fields it leaves out; its labels, which no server or scaler
  selector matches; its resources, retries and retention; and every
  render-time refusal of a `dr.*` value, each by its message;
* values from a release that predates the `dr` block, as
  `helm upgrade --reuse-values` passes them.

DR renders target Kubernetes 1.29 (`--kube-version 1.29.0`), the Jobs'
minimum. `--service-chart PATH` renders another copy of the chart.
"""

import argparse
import copy
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SERVICE_CHART = ROOT / "helm/live-edit-service"
CONFIG_RS = ROOT / "src/config.rs"
JOB_RS = ROOT / "src/dr/job/mod.rs"
ARGS_RS = ROOT / "src/dr/job/args.rs"
RELEASE = "live-edit-service"
# The chart's name: the main container's name, and the prefix of every
# template's `# Source:` path.
CHART_NAME = "live-edit-service"
NAMESPACE = "live-edit"
KUBE_VERSION = "1.29.0"
INCIDENT = "20261020t0000-abcd"
NAME = "app.kubernetes.io/name"
INSTANCE = "app.kubernetes.io/instance"
COMPONENT = "app.kubernetes.io/component"

DR_TEMPLATES = {
    "restore": f"{CHART_NAME}/templates/dr-restore-job.yaml",
    "acceptLoss": f"{CHART_NAME}/templates/dr-accept-loss-job.yaml",
}
HOOK_TEMPLATES = (
    f"{CHART_NAME}/templates/tests/smoke-test.yaml",
    f"{CHART_NAME}/templates/tests/smoke-test-configmap.yaml",
    f"{CHART_NAME}/templates/tests/smoke-test-serviceaccount.yaml",
)
DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject a duplicate mapping key instead of silently keeping the last."""


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def helm_template(values, args, chart, release, kube):
    command = ["helm", "template", release, str(chart or SERVICE_CHART),
               "--namespace", NAMESPACE]
    if kube is not None:
        command += ["--kube-version", kube]
    command += ["-f", "-", *args]
    return subprocess.run(command, input=yaml.safe_dump(values or {}),
                          capture_output=True, text=True, check=False)


def render_text(values=None, args=(), *, chart=None, release=RELEASE, kube=KUBE_VERSION):
    """`helm template`'s stdout. `kube=None` leaves Helm's built-in version."""
    result = helm_template(values, args, chart, release, kube)
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout


def render(values=None, args=(), **kwargs):
    """The rendered manifests as (template path, parsed document) pairs."""
    parsed = [(source, yaml.load(text, Loader=UniqueKeyLoader))
              for source, text in split_documents(render_text(values, args, **kwargs))]
    return [(source, doc) for source, doc in parsed if doc]


def render_error(values=None, args=(), *, fragment, chart=None, release=RELEASE,
                 kube=KUBE_VERSION):
    """Assert the render fails with `fragment` in its message; the stderr."""
    result = helm_template(values, args, chart, release, kube)
    if result.returncode == 0:
        raise AssertionError(f"Expected the render to fail naming {fragment!r}; it rendered")
    if fragment not in result.stderr:
        raise AssertionError(f"Expected {fragment!r} in the render's failure: {result.stderr}")
    return result.stderr


def split_documents(stdout):
    """The rendered manifests as (template path, raw text) pairs, in Helm's
    order. Each starts with its `# Source:` line."""
    pairs = []
    for chunk in re.split(r"^---\n(?=# Source: )", stdout, flags=re.M):
        if not chunk.strip():
            continue
        first, _, _ = chunk.partition("\n")
        assert first.startswith("# Source: "), first
        pairs.append((first[len("# Source: "):], chunk))
    return pairs


def merged(*parts):
    """A deep merge of value dicts, later parts winning."""
    result = {}
    for part in parts:
        for key, value in part.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merged(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
    return result


def restore(incident=INCIDENT, **settings):
    return {"dr": {"restore": {"incident": incident, **settings}}}


def accept_loss(incident=INCIDENT, confirm="discard-lost-data", **settings):
    return {"dr": {"acceptLoss": {"incident": incident, "confirm": confirm, **settings}}}


# Every way a DR action is set: (label, values, the action that renders).
ACTION_VARIANTS = (
    ("restore", restore(), "restore"),
    ("restore with an attempt", restore(attempt=2), "restore"),
    ("accept-loss", accept_loss(), "acceptLoss"),
    ("accept-loss with both confirms",
     accept_loss(confirmUnconfirmed="discard-unconfirmed-data"), "acceptLoss"),
    ("accept-loss with an attempt", accept_loss(attempt="b"), "acceptLoss"),
    # The Job's other dr.* settings too, none at its default: they reach the
    # Job only, so every check over these variants holds with them set.
    ("restore with the Job's settings", merged(restore(attempt=3), {"dr": {
        "resources": {"limits": {"memory": "8Gi"}, "requests": {"cpu": "500m"}},
        "backoffLimit": 2, "ttlSecondsAfterFinished": 60}}), "restore"),
)

SIDECAR = {"serviceIdentity": {"accessTokenFile": {"enabled": True, "provisionerSidecar": {
    "container": {"image": "example.com/token-provisioner:1"}}}}}
STORAGE_API = merged(SIDECAR, {"checkpoint": {
    "adapter": "StorageApi", "storageApiEndpoint": "https://storage.example.com/",
    "basePath": "/container/live-edit"}})
SMOKE_TESTS = {"tests": {"enabled": True,
                         "storageApiResourceBase": "https://storage.example.com/container/"}}
NODE_AFFINITY = {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
    {"matchExpressions": [{"key": "example.com/pool", "operator": "In", "values": ["live-edit"]}]}]}}
# A rule written for server pods: never two on one node.
POD_ANTI_AFFINITY = {"requiredDuringSchedulingIgnoredDuringExecution": [{
    "labelSelector": {"matchLabels": {NAME: CHART_NAME}},
    "topologyKey": "kubernetes.io/hostname"}]}
PLACEMENT = {
    "nodeSelector": {"example.com/pool": "live-edit"},
    "tolerations": [{"key": "example.com/dedicated", "operator": "Equal",
                     "value": "live-edit", "effect": "NoSchedule"}],
    "affinity": {"nodeAffinity": NODE_AFFINITY, "podAntiAffinity": POD_ANTI_AFFINITY},
}
# The value sets a DR action is laid over in the "changes nothing else" check.
BASE_VALUES = (
    ("defaults", {}),
    ("smoke tests", SMOKE_TESTS),
    ("token provisioner and StorageApi backups", STORAGE_API),
    ("placement", PLACEMENT),
)


def dr_jobs(manifests):
    """The DR Job documents, by the template they came from."""
    return [doc for source, doc in manifests if source in DR_TEMPLATES.values()]


def dr_job(manifests):
    jobs = dr_jobs(manifests)
    assert len(jobs) == 1, len(jobs)
    return jobs[0]


def job_pod(manifests):
    return dr_job(manifests)["spec"]["template"]


def main_container(pod_spec):
    containers = pod_spec["containers"]
    assert containers[0]["name"] == CHART_NAME, containers[0]["name"]
    return containers[0]


def server_deployment(manifests):
    """The server's Deployment, by its pods' component=server label: the
    default render also has the scaler's."""
    matches = [doc for _, doc in manifests if doc["kind"] == "Deployment"
               and doc["spec"]["template"]["metadata"]["labels"].get(COMPONENT) == "server"]
    assert len(matches) == 1, len(matches)
    return matches[0]


def settings_configmap(manifests):
    """The ConfigMap that carries the service's env vars."""
    maps = [doc for _, doc in manifests if doc["kind"] == "ConfigMap"
            and "LIVE_EDIT_CHECKPOINT_ENABLED" in (doc.get("data") or {})]
    assert len(maps) == 1, len(maps)
    return maps[0]


def config_checksum(manifests):
    """The server Deployment's `checksum/config` pod annotation."""
    return server_deployment(manifests)["spec"]["template"]["metadata"]["annotations"][
        "checksum/config"]


def source_match(path, pattern, what):
    match = re.search(pattern, path.read_text())
    assert match, f"{what} not found in {path}"
    return match


def binary_parallelism_default():
    """`DR_RESTORE_PARALLELISM_DEFAULT` as `src/config.rs` declares it."""
    return int(source_match(
        CONFIG_RS,
        r"pub const DR_RESTORE_PARALLELISM_DEFAULT: NonZeroUsize = match NonZeroUsize::new\((\d+)\)",
        "DR_RESTORE_PARALLELISM_DEFAULT").group(1))


def binary_fail_at_once_exit_codes():
    """The exit codes a retry cannot change, as `src/dr/job/mod.rs` declares
    them: a refused command line or configuration (`EXIT_USAGE`), the service
    staying held, and a hold naming another incident."""
    return [
        int(source_match(JOB_RS, r"pub const EXIT_USAGE: i32 = (\d+);", "EXIT_USAGE").group(1)),
        int(source_match(JOB_RS, r"Self::StaysHeld => (\d+),", "StaysHeld").group(1)),
        int(source_match(JOB_RS, r"Self::DifferentIncident => (\d+),", "DifferentIncident").group(1)),
    ]


def binary_confirmations():
    """`accept-loss`'s two `--confirm` values, as `src/dr/job/args.rs`
    declares them."""
    return tuple(source_match(ARGS_RS, rf'pub const {name}: &str = "([^"]+)";', name).group(1)
                 for name in ("CONFIRM_DISCARD_LOST_DATA", "CONFIRM_DISCARD_UNCONFIRMED_DATA"))


def binary_subcommand(variant):
    """`restore` or `accept-loss`, as `JobMode::name` in `src/dr/job/args.rs`
    spells the subcommand."""
    return source_match(ARGS_RS, rf'Self::{variant} => "([^"]+)",', variant).group(1)


def expected_job_name(fullname, action, incident, attempt=""):
    """The Job name rule: `<fullname>-dr-<action>-<incident>[-<attempt>]`
    while that fits 63 characters; otherwise the fullname cut to leave room
    for the first 8 hex characters of its sha256 (trailing hyphens dropped),
    then that hash, then the same suffix, which is never cut."""
    suffix = f"-dr-{'restore' if action == 'restore' else 'accept-loss'}-{incident}"
    if attempt:
        suffix += f"-{attempt}"
    if len(fullname) + len(suffix) <= 63:
        return fullname + suffix
    digest = hashlib.sha256(fullname.encode()).hexdigest()[:8]
    cut = fullname[:max(0, 54 - len(suffix))].rstrip("-")
    return f"{cut}-{digest}{suffix}" if cut else digest + suffix


def matches(selector, labels):
    return all(labels.get(key) == value for key, value in selector.items())


def pod_selectors(manifests):
    """Every selector in the release that picks pods: the Deployments' (the
    server's and the scaler's), the Service's, the PodDisruptionBudget's and
    the NetworkPolicy's, as (kind, labels) pairs."""
    found = []
    for _, doc in manifests:
        spec = doc.get("spec") or {}
        if doc["kind"] in ("Deployment", "PodDisruptionBudget"):
            found.append((doc["kind"], spec["selector"]["matchLabels"]))
        elif doc["kind"] == "Service" and "selector" in spec:
            found.append(("Service", spec["selector"]))
        elif doc["kind"] == "NetworkPolicy":
            found.append(("NetworkPolicy", spec["podSelector"]["matchLabels"]))
    return found


class DrSettingsTests(unittest.TestCase):
    """The Job's settings outside `dr.*`, and the ConfigMap that carries them."""

    def test_restore_parallelism_renders_from_checkpoint_with_the_binarys_default(self):
        data = settings_configmap(render())["data"]
        self.assertEqual(data["LIVE_EDIT_DR_RESTORE_PARALLELISM"],
                         str(binary_parallelism_default()))
        data = settings_configmap(
            render(values={"checkpoint": {"restoreParallelism": 3}}))["data"]
        self.assertEqual(data["LIVE_EDIT_DR_RESTORE_PARALLELISM"], "3")

    def test_restore_parallelism_that_is_not_a_whole_number_fails_the_render(self):
        render_error({"checkpoint": {"restoreParallelism": "fast"}},
                     fragment="checkpoint.restoreParallelism")

    def test_no_restore_memory_setting_renders(self):
        data = settings_configmap(render())["data"]
        self.assertNotIn("LIVE_EDIT_DR_RESTORE_MEMORY_FRACTION", data)

    def test_every_dr_action_leaves_the_settings_configmap_and_the_checksum_unchanged(self):
        plain = render()
        for label, values, _ in ACTION_VARIANTS:
            with self.subTest(label):
                with_action = render(values)
                self.assertEqual(settings_configmap(with_action), settings_configmap(plain))
                self.assertEqual(config_checksum(with_action), config_checksum(plain))

    def test_the_configmap_template_reads_no_dr_value(self):
        # The checksum is the sha256 of configmap.yaml's render, so a template
        # that read no dr value cannot change it whatever dr holds.
        text = (SERVICE_CHART / "templates/configmap.yaml").read_text()
        self.assertIsNone(re.search(r"Values\.dr\b|\"dr\"", text))


class DrActionChangesNothingElseTests(unittest.TestCase):
    def test_a_dr_action_adds_one_job_removes_the_hooks_and_leaves_every_other_manifest_byte_identical(self):
        # Raw text, document by document: a DR action may only add its Job
        # and take the smoke-test hooks away, never change another manifest.
        for base_label, base in BASE_VALUES:
            without = split_documents(render_text(base))
            hooks = [source for source, _ in without if source in HOOK_TEMPLATES]
            self.assertEqual(sorted(hooks), sorted(HOOK_TEMPLATES) if "tests" in base else [],
                             f"fixture: {base_label} renders the hooks only with tests.enabled")
            for label, values, action in ACTION_VARIANTS:
                with self.subTest(base=base_label, action=label):
                    with_action = split_documents(render_text(merged(base, values)))
                    added = [source for source, _ in with_action
                             if source in DR_TEMPLATES.values()]
                    self.assertEqual(added, [DR_TEMPLATES[action]])
                    self.assertEqual(
                        [pair for pair in with_action if pair[0] not in DR_TEMPLATES.values()],
                        [pair for pair in without if pair[0] not in HOOK_TEMPLATES])


class DrJobRenderTests(unittest.TestCase):
    def test_no_action_renders_no_job_and_keeps_the_smoke_test_hooks(self):
        for label, values in (("defaults", {}),
                              ("an attempt without an incident",
                               {"dr": {"restore": {"attempt": 2}, "acceptLoss": {"attempt": 3}}}),
                              ("an empty incident", restore(incident=""))):
            with self.subTest(label):
                manifests = render(merged(SMOKE_TESTS, values))
                self.assertEqual(dr_jobs(manifests), [])
                self.assertEqual(sorted(source for source, _ in manifests
                                        if source in HOOK_TEMPLATES), sorted(HOOK_TEMPLATES))

    def test_restore_renders_one_job_named_for_the_incident_that_runs_dr_restore(self):
        job = dr_job(render(restore()))
        self.assertEqual((job["apiVersion"], job["kind"]), ("batch/v1", "Job"))
        self.assertEqual(job["metadata"]["name"], f"{RELEASE}-dr-restore-{INCIDENT}")
        self.assertEqual(main_container(job["spec"]["template"]["spec"])["args"],
                         ["dr", binary_subcommand("Restore"), INCIDENT])

    def test_a_whole_number_incident_reads_as_its_digits(self):
        # A values-file number decodes as a float; unquoted, 20261013 would
        # otherwise print as 2.0261013e+07.
        job = dr_job(render(restore(incident=20261013)))
        self.assertEqual(job["metadata"]["name"], f"{RELEASE}-dr-restore-20261013")
        self.assertEqual(main_container(job["spec"]["template"]["spec"])["args"],
                         ["dr", "restore", "20261013"])

    def test_an_attempt_suffixes_the_job_name_for_both_actions(self):
        for values, name in (
                (restore(attempt=2), f"{RELEASE}-dr-restore-{INCIDENT}-2"),
                (restore(attempt=0), f"{RELEASE}-dr-restore-{INCIDENT}-0"),
                (restore(attempt="abcdefgh"), f"{RELEASE}-dr-restore-{INCIDENT}-abcdefgh"),
                (restore(attempt=""), f"{RELEASE}-dr-restore-{INCIDENT}"),
                (accept_loss(attempt="b"), f"{RELEASE}-dr-accept-loss-{INCIDENT}-b"),
                (accept_loss(attempt=2), f"{RELEASE}-dr-accept-loss-{INCIDENT}-2")):
            with self.subTest(values=values):
                job = dr_job(render(values))
                self.assertEqual(job["metadata"]["name"], name)
                # The attempt names the Job only; the command line is the same.
                self.assertEqual(main_container(job["spec"]["template"]["spec"])["args"][2],
                                 INCIDENT)

    def test_accept_loss_runs_with_exactly_the_first_confirmation(self):
        lost, _ = binary_confirmations()
        job = dr_job(render(accept_loss(confirm=lost)))
        self.assertEqual(job["metadata"]["name"], f"{RELEASE}-dr-accept-loss-{INCIDENT}")
        self.assertEqual(main_container(job["spec"]["template"]["spec"])["args"],
                         ["dr", binary_subcommand("AcceptLoss"), INCIDENT, "--confirm", lost])

    def test_accept_loss_with_confirm_unconfirmed_runs_with_both_confirmations(self):
        lost, unconfirmed = binary_confirmations()
        job = dr_job(render(accept_loss(confirm=lost, confirmUnconfirmed=unconfirmed)))
        self.assertEqual(main_container(job["spec"]["template"]["spec"])["args"],
                         ["dr", "accept-loss", INCIDENT, "--confirm", lost,
                          "--confirm", unconfirmed])

    def test_the_job_is_a_plain_job_on_the_servers_image_selected_by_args(self):
        for label, values, _ in ACTION_VARIANTS:
            with self.subTest(label):
                manifests = render(merged(SIDECAR, values))
                job = dr_job(manifests)
                # Not a Helm hook: --timeout and --atomic must not bind its run.
                self.assertFalse(any(key.startswith("helm.sh/")
                                     for key in job["metadata"].get("annotations") or {}))
                self.assertNotIn("selector", job["spec"])
                self.assertNotIn("activeDeadlineSeconds", job["spec"])
                pod = job["spec"]["template"]["spec"]
                server = main_container(server_deployment(manifests)["spec"]["template"]["spec"])
                self.assertEqual(main_container(pod)["image"], server["image"])
                # The image's entrypoint is the binary: args select the mode.
                for container in pod["containers"] + pod["initContainers"]:
                    self.assertNotIn("command", container, container["name"])
                # Fixture: a variant's dr.* settings reach the Job.
                dr = values["dr"]
                self.assertEqual((job["spec"]["backoffLimit"],
                                  job["spec"]["ttlSecondsAfterFinished"]),
                                 (dr.get("backoffLimit", 6),
                                  dr.get("ttlSecondsAfterFinished", 604800)))
                resources = main_container(pod)["resources"]
                for section, entries in dr.get("resources", {}).items():
                    for key, value in entries.items():
                        self.assertEqual(resources[section][key], value, (section, key))

    def test_the_pod_failure_policy_fails_the_job_at_once_on_the_binarys_exit_codes(self):
        codes = binary_fail_at_once_exit_codes()
        self.assertEqual(len(set(codes)), 3, codes)
        for label, values, _ in ACTION_VARIANTS:
            with self.subTest(label):
                job = dr_job(render(merged(SIDECAR, values)))
                self.assertEqual(job["spec"]["podFailurePolicy"], {"rules": [
                    {"action": "FailJob", "onExitCodes": {
                        "containerName": CHART_NAME, "operator": "In",
                        "values": sorted(codes)}},
                    {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]},
                ]})
                pod = job["spec"]["template"]["spec"]
                # The rule names the main container, never the provisioner,
                # whose exit must not fail the Job.
                self.assertEqual(main_container(pod)["name"], CHART_NAME)
                self.assertNotIn(CHART_NAME, [c["name"] for c in pod["initContainers"]])

    def test_backoff_limit_and_ttl_render_their_defaults_overrides_and_omissions(self):
        for dr, backoff, ttl in (
                ({}, 6, 604800),
                ({"backoffLimit": 0, "ttlSecondsAfterFinished": 3600}, 0, 3600),
                ({"backoffLimit": "3", "ttlSecondsAfterFinished": "+120"}, 3, 120),
                ({"backoffLimit": None, "ttlSecondsAfterFinished": None}, None, None),
                ({"backoffLimit": "", "ttlSecondsAfterFinished": ""}, None, None),
                # 0 would mean "delete at once" to Kubernetes.
                ({"ttlSecondsAfterFinished": 0}, 6, None)):
            with self.subTest(dr=dr):
                job = dr_job(render(merged(restore(), {"dr": dr})))
                self.assertEqual(job["spec"].get("backoffLimit"), backoff)
                self.assertEqual(job["spec"].get("ttlSecondsAfterFinished"), ttl)


class DrJobPodSpecTests(unittest.TestCase):
    """The Job's pod spec against the server's, under non-default values."""

    VALUES = merged(STORAGE_API, PLACEMENT, {
        "podSecurityContext": {"runAsNonRoot": True, "runAsUser": 10001, "fsGroup": 10001},
        "securityContext": {"readOnlyRootFilesystem": True, "allowPrivilegeEscalation": False},
        "image": {"pullSecrets": [{"name": "image-pull"}]},
        "imagePullSecrets": [{"name": "global-pull"}],
        "serviceAccount": {"automount": True, "annotations": {
            "azure.workload.identity/client-id": "00000000-0000-0000-0000-000000000000"}},
        "podLabels": {"example.com/team": "editing"},
        "podAnnotations": {"example.com/scrape": "true"},
        "volumes": [{"name": "scratch", "emptyDir": {}}],
        "volumeMounts": [{"name": "scratch", "mountPath": "/scratch"}],
        "valkey": {"url": {"value": "redis://valkey.example.com:6379"}},
        "serviceIdentity": {"clientCredentials": {"enabled": True}},
        "shareProcessNamespace": True,
    })

    POD_FIELDS = ("imagePullSecrets", "serviceAccountName", "automountServiceAccountToken",
                  "securityContext", "volumes", "nodeSelector", "tolerations")
    CONTAINER_FIELDS = ("name", "image", "imagePullPolicy", "securityContext", "envFrom", "env",
                        "volumeMounts")

    @classmethod
    def setUpClass(cls):
        cls.pods = {}
        for label, values, _ in ACTION_VARIANTS:
            manifests = render(merged(cls.VALUES, values))
            cls.pods[label] = (server_deployment(manifests)["spec"]["template"],
                               job_pod(manifests))

    def test_the_job_shares_the_servers_identity_image_settings_volumes_and_placement(self):
        for label, (server, job) in self.pods.items():
            with self.subTest(label):
                for field in self.POD_FIELDS:
                    self.assertIn(field, server["spec"], f"fixture: the server sets {field}")
                    self.assertEqual(job["spec"][field], server["spec"][field], field)
                server_main, job_main = main_container(server["spec"]), main_container(job["spec"])
                for field in self.CONTAINER_FIELDS:
                    self.assertIn(field, server_main, f"fixture: the server sets {field}")
                    self.assertEqual(job_main[field], server_main[field], field)
                metrics = [port for port in server_main["ports"] if port["name"] == "metrics"]
                self.assertEqual(job_main["ports"], metrics)

    def test_the_job_runs_the_servers_token_provisioner_as_a_native_sidecar(self):
        for label, (server, job) in self.pods.items():
            with self.subTest(label):
                provisioner = server["spec"]["containers"][1]
                self.assertEqual(provisioner["name"], f"{CHART_NAME}-token-provisioner")
                self.assertEqual(job["spec"]["initContainers"],
                                 [{**provisioner, "restartPolicy": "Always"}])
                self.assertEqual(len(job["spec"]["containers"]), 1)

    def test_a_values_restart_policy_on_the_provisioner_is_replaced_in_the_job(self):
        values = merged(SIDECAR, restore(), {"serviceIdentity": {"accessTokenFile": {
            "provisionerSidecar": {"container": {"restartPolicy": "OnFailure"}}}}})
        self.assertEqual(job_pod(render(values))["spec"]["initContainers"][0]["restartPolicy"],
                         "Always")

    def test_the_job_takes_node_affinity_only(self):
        for label, (server, job) in self.pods.items():
            with self.subTest(label):
                self.assertEqual(server["spec"]["affinity"], PLACEMENT["affinity"])
                self.assertEqual(job["spec"]["affinity"], {"nodeAffinity": NODE_AFFINITY})
        # Pod (anti-)affinity alone leaves the Job no affinity at all.
        values = merged(restore(), {"affinity": {"podAntiAffinity": POD_ANTI_AFFINITY}})
        self.assertNotIn("affinity", job_pod(render(values))["spec"])

    def test_the_job_carries_the_pod_labels_and_annotations_without_the_checksum(self):
        for label, (server, job) in self.pods.items():
            with self.subTest(label):
                self.assertEqual(job["metadata"]["annotations"], {"example.com/scrape": "true"})
                labels = job["metadata"]["labels"]
                self.assertEqual(labels["example.com/team"], "editing")
                self.assertEqual(labels["azure.workload.identity/use"], "true")
                self.assertEqual(labels[COMPONENT], "dr-job")
                self.assertNotIn(NAME, labels)

    def test_the_job_leaves_out_the_server_only_fields(self):
        for label, (server, job) in self.pods.items():
            with self.subTest(label):
                server_main, job_main = main_container(server["spec"]), main_container(job["spec"])
                for probe in ("startupProbe", "livenessProbe", "readinessProbe"):
                    self.assertIn(probe, server_main, f"fixture: the server has a {probe}")
                    self.assertNotIn(probe, job_main)
                self.assertIn("resizePolicy", server_main, "fixture: Kubernetes 1.29")
                self.assertNotIn("resizePolicy", job_main)
                self.assertIs(server["spec"]["shareProcessNamespace"], True)
                self.assertNotIn("shareProcessNamespace", job["spec"])
                self.assertIn("terminationGracePeriodSeconds", server["spec"])
                self.assertNotIn("terminationGracePeriodSeconds", job["spec"])
                self.assertEqual(job["spec"]["restartPolicy"], "Never")
                self.assertNotIn("restartPolicy", server["spec"])


class DrJobResourcesTests(unittest.TestCase):
    def job_resources(self, values):
        manifests = render(merged(restore(), values))
        return (main_container(job_pod(manifests)["spec"])["resources"],
                main_container(server_deployment(manifests)["spec"]["template"]["spec"])[
                    "resources"])

    def test_unset_dr_resources_give_the_job_the_servers_resources(self):
        for values in ({}, {"resources": {"limits": {"memory": "6Gi"},
                                          "requests": {"memory": "4Gi"}}}):
            with self.subTest(values=values):
                job, server = self.job_resources(values)
                self.assertEqual(job, server)

    def test_dr_resources_are_laid_over_the_servers_entry_by_entry(self):
        job, server = self.job_resources({"dr": {"resources": {"limits": {"memory": "8Gi"}}}})
        # 70 % of 8Gi, rounded down to whole Mi; the CPU stays the server's.
        self.assertEqual(job, {"requests": {"cpu": 1, "memory": "5734Mi"},
                               "limits": {"cpu": 2, "memory": "8Gi"}})
        self.assertEqual(server, {"requests": {"cpu": 1, "memory": "3584Mi"},
                                  "limits": {"cpu": 2, "memory": "5Gi"}})
        job, _ = self.job_resources({"dr": {"resources": {"requests": {"memory": "4Gi"}}}})
        self.assertEqual(job["requests"], {"cpu": 1, "memory": "4Gi"})
        job, _ = self.job_resources({"dr": {"resources": {"limits": {"cpu": None}}}})
        self.assertEqual(job["limits"], {"memory": "5Gi"})


class DrJobSelectionTests(unittest.TestCase):
    def test_no_server_or_scaler_selector_selects_a_job_or_its_pod(self):
        for name in ("live-edit-service", "custom"):
            for label, values, _ in ACTION_VARIANTS:
                with self.subTest(name=name, action=label):
                    # Pod labels copying every server selector label.
                    manifests = render(merged(values, {
                        "nameOverride": name, "networkPolicy": {"enabled": True},
                        "podLabels": {NAME: name, INSTANCE: RELEASE, COMPONENT: "server"}}))
                    selectors = pod_selectors(manifests)
                    self.assertEqual(sorted(kind for kind, _ in selectors), [
                        "Deployment", "Deployment", "NetworkPolicy", "PodDisruptionBudget",
                        "Service"], "fixture: every pod selector in the release")
                    server_labels = server_deployment(manifests)["spec"]["template"][
                        "metadata"]["labels"]
                    self.assertTrue(any(matches(selector, server_labels)
                                        for _, selector in selectors),
                                    "fixture: the matcher selects the server's pods")
                    job = dr_job(manifests)
                    for labels in (job["metadata"]["labels"],
                                   job["spec"]["template"]["metadata"]["labels"]):
                        self.assertNotIn(NAME, labels)
                        self.assertEqual(labels[COMPONENT], "dr-job")
                        for kind, selector in selectors:
                            self.assertFalse(matches(selector, labels), (kind, selector))

    def test_the_runbooks_selector_finds_the_job_and_its_pod(self):
        # The operator lists the Jobs with
        # `-l app.kubernetes.io/component=dr-job,app.kubernetes.io/instance=<release>`.
        selector = {COMPONENT: "dr-job", INSTANCE: RELEASE}
        for label, values, _ in ACTION_VARIANTS:
            with self.subTest(label):
                job = dr_job(render(values))
                self.assertTrue(matches(selector, job["metadata"]["labels"]),
                                job["metadata"]["labels"])
                self.assertTrue(matches(selector, job["spec"]["template"]["metadata"]["labels"]),
                                job["spec"]["template"]["metadata"]["labels"])


class DrSmokeTestHookTests(unittest.TestCase):
    def test_no_smoke_test_hook_renders_while_an_action_is_set(self):
        # The hooks run after every upgrade, and a held service cannot pass
        # them: they would fail the upgrade that starts the Job.
        for label, values, _ in ACTION_VARIANTS:
            with self.subTest(label):
                manifests = render(merged(SMOKE_TESTS, values))
                for hook in HOOK_TEMPLATES:
                    self.assertNotIn(hook, [source for source, _ in manifests])


class DrJobNameTests(unittest.TestCase):
    """The Job's name always fits the 63 characters a label value allows (the
    name is one on the Job's pod), and the incident and attempt are never
    cut: the fullname is cut instead and an 8-character hash of the whole
    fullname follows it."""

    # 40 characters: its fullname, with "-live-edit-service", is 58, under
    # the 63 the fullname itself is cut to.
    LONG_RELEASE = "a-forty-character-release-name-for-tests"
    INCIDENT = "20261013t1708-a3f9"

    def job_name(self, values, **kwargs):
        return dr_job(render(values, **kwargs))["metadata"]["name"]

    def test_the_short_default_fullname_is_kept_whole(self):
        self.assertEqual(self.job_name(restore(incident=self.INCIDENT)),
                         f"{RELEASE}-dr-restore-{self.INCIDENT}")
        self.assertEqual(self.job_name(accept_loss(incident=self.INCIDENT, attempt="abcdefgh")),
                         f"{RELEASE}-dr-accept-loss-{self.INCIDENT}-abcdefgh")

    def test_a_name_of_exactly_63_characters_is_kept_whole_and_64_is_cut(self):
        # A 33-character fullname, then "-dr-restore-" and the 18-character
        # incident (30): 63. One more fullname character makes 64.
        kept = "abcdefghij" * 3 + "abc"
        self.assertEqual(self.job_name(merged(restore(incident=self.INCIDENT),
                                              {"fullnameOverride": kept})),
                         f"abcdefghijabcdefghijabcdefghijabc-dr-restore-{self.INCIDENT}")
        self.assertEqual(len(kept) + len(f"-dr-restore-{self.INCIDENT}"), 63)
        cut = kept + "d"
        name = self.job_name(merged(restore(incident=self.INCIDENT), {"fullnameOverride": cut}))
        digest = hashlib.sha256(cut.encode()).hexdigest()[:8]
        # Cut to 54 - 30 = 24 characters, then the hash.
        self.assertEqual(name, f"{cut[:24]}-{digest}-dr-restore-{self.INCIDENT}")
        self.assertEqual(len(name), 63)

    def test_a_cut_name_ends_with_the_whole_incident_and_attempt(self):
        # Worked example: the fullname test-release-live-edit-service is cut
        # to 20 characters, followed by the first 8 of its sha256.
        self.assertEqual(self.job_name(accept_loss(incident=self.INCIDENT), release="test-release"),
                         f"test-release-live-ed-7071b642-dr-accept-loss-{self.INCIDENT}")
        fullname = f"{self.LONG_RELEASE}-{CHART_NAME}"
        self.assertEqual((len(self.LONG_RELEASE), len(fullname)), (40, 58))
        for values, action, attempt in (
                (restore(incident=self.INCIDENT, attempt="abcdefgh"), "restore", "abcdefgh"),
                (accept_loss(incident=self.INCIDENT, attempt="abcdefgh"), "acceptLoss",
                 "abcdefgh"),
                (restore(incident=self.INCIDENT), "restore", "")):
            with self.subTest(action=action, attempt=attempt):
                name = self.job_name(values, release=self.LONG_RELEASE)
                self.assertLessEqual(len(name), 63)
                self.assertRegex(name, DNS_LABEL)
                self.assertEqual(name, expected_job_name(fullname, action, self.INCIDENT, attempt))
                self.assertTrue(name.endswith(
                    f"-{self.INCIDENT}-{attempt}" if attempt else f"-{self.INCIDENT}"), name)
                self.assertEqual(name, self.job_name(values, release=self.LONG_RELEASE),
                                 "the same values give the same name")

    def test_fullnames_sharing_the_kept_prefix_get_different_names(self):
        names = {}
        for override in ("x" * 40 + "-alpha", "x" * 40 + "-bravo"):
            names[override] = self.job_name(
                merged(restore(incident=self.INCIDENT), {"fullnameOverride": override}))
        first, second = names.values()
        suffix = f"-dr-restore-{self.INCIDENT}"
        # Both keep the same cut fullname; only the hash tells them apart.
        self.assertEqual(first[:24], second[:24])
        self.assertTrue(first.endswith(suffix) and second.endswith(suffix))
        self.assertNotEqual(first, second)

    def test_trailing_hyphens_of_the_cut_fullname_are_dropped(self):
        # The restore suffix here is 30 characters, so the fullname is cut
        # to 24, which ends in hyphens.
        for override in ("a" * 23 + "-" + "b" * 30, "a" * 22 + "--" + "b" * 30):
            with self.subTest(override=override):
                name = self.job_name(merged(restore(incident=self.INCIDENT),
                                            {"fullnameOverride": override}))
                digest = hashlib.sha256(override.encode()).hexdigest()[:8]
                self.assertEqual(name, f"{override.rstrip('b').rstrip('-')}-{digest}"
                                       f"-dr-restore-{self.INCIDENT}")
                self.assertEqual(name, expected_job_name(override, "restore", self.INCIDENT))
                self.assertRegex(name, DNS_LABEL)

    def test_an_incident_that_leaves_no_room_for_the_fullname_keeps_only_the_hash(self):
        digest = hashlib.sha256(f"{RELEASE}".encode()).hexdigest()[:8]
        for values, suffix in (
                # "-dr-accept-loss-" and a 39-character incident: 55.
                (accept_loss(incident="a" * 39), "-dr-accept-loss-" + "a" * 39),
                # "-dr-restore-", 34 characters and an 8-character attempt: 55.
                (restore(incident="b" * 34, attempt="abcdefgh"),
                 "-dr-restore-" + "b" * 34 + "-abcdefgh")):
            with self.subTest(suffix=suffix):
                self.assertEqual(len(suffix), 55)
                self.assertEqual(self.job_name(values), digest + suffix)

    def test_an_incident_and_attempt_too_long_for_the_hash_fail_the_render(self):
        # 56 characters after the fullname leave no room for its hash.
        for values, action, incident, attempt in (
                (accept_loss(incident="a" * 40), "acceptLoss", "a" * 40, ""),
                (restore(incident="b" * 35, attempt="abcdefgh"), "restore", "b" * 35,
                 "abcdefgh")):
            with self.subTest(action=action):
                render_error(values, fragment=(
                    f'dr.{action}.incident ("{incident}", {len(incident)} characters) and '
                    f'dr.{action}.attempt ("{attempt}") are too long for the disaster-recovery '
                    f"Job's name"))


class DrRenderRefusalTests(unittest.TestCase):
    """Every `dr.*` value the render refuses, by its message. The checks run
    on every render, with or without an action set."""

    def assert_refused(self, cases):
        for label, values, fragment in cases:
            with self.subTest(label):
                render_error(values, fragment=fragment)

    def test_one_action_at_a_time(self):
        self.assert_refused([("both actions", merged(restore(incident="a1"),
                                                     accept_loss(incident="b2")),
                              'dr.restore.incident ("a1") and dr.acceptLoss.incident ("b2") are '
                              "both set: run one disaster-recovery action at a time")])

    def test_accept_loss_confirmations(self):
        self.assert_refused([
            ("no confirm", accept_loss(confirm=""),
             f'dr.acceptLoss.incident is set ("{INCIDENT}") but dr.acceptLoss.confirm is not'),
            ("a wrong confirm", accept_loss(confirm="yes"),
             'dr.acceptLoss.confirm must be "discard-lost-data" (got "yes")'),
            ("a wrong second confirm", accept_loss(confirmUnconfirmed="yes"),
             'dr.acceptLoss.confirmUnconfirmed must be "discard-unconfirmed-data" or unset '
             '(got "yes")'),
            ("the second confirm alone",
             accept_loss(confirm="", confirmUnconfirmed="discard-unconfirmed-data"),
             "dr.acceptLoss.confirmUnconfirmed is set but dr.acceptLoss.confirm is not "
             '"discard-lost-data"'),
            ("a confirm left without its incident", accept_loss(incident=""),
             "dr.acceptLoss.confirm is set but dr.acceptLoss.incident is not"),
            ("a second confirm left without its incident",
             accept_loss(incident="", confirm="", confirmUnconfirmed="discard-unconfirmed-data"),
             "dr.acceptLoss.confirmUnconfirmed is set but dr.acceptLoss.incident is not"),
        ])

    def test_incident_format(self):
        cases = []
        for incident in ("20261020T0000-ABCD", "2026_1020", "-20261020", "20261020-",
                         "2026 1020", 1.5):
            for action, values in (("restore", restore(incident=incident)),
                                   ("acceptLoss", accept_loss(incident=incident))):
                cases.append((f"{action} {incident!r}", values,
                              f"dr.{action}.incident must be the incident id the hold names: "
                              f'lowercase letters, digits and hyphens, starting and ending with '
                              f'a letter or digit (got "{incident}")'))
        self.assert_refused(cases)

    def test_attempt_format(self):
        cases = []
        for attempt in ("-1", -1, "A", "123456789", 1.5, "a-"):
            for action, values in (("restore", restore(attempt=attempt)),
                                   ("acceptLoss", accept_loss(attempt=attempt))):
                cases.append((f"{action} {attempt!r}", values,
                              f"dr.{action}.attempt must be up to 8 lowercase letters, digits and "
                              f'hyphens, starting and ending with a letter or digit, such as 2 '
                              f'(got "{attempt}")'))
        self.assert_refused(cases)

    def test_kubernetes_1_29_whenever_an_action_is_set(self):
        for action, values in (("restore", restore()), ("acceptLoss", accept_loss())):
            with self.subTest(action):
                render_error(values, kube="1.28.15", fragment=(
                    f"dr.{action}.incident is set, but the disaster-recovery Jobs need Kubernetes "
                    f"1.29 or later and this render targets v1.28.15"))
                dr_job(render(values, kube="1.29.0"))
        # Without an action the chart still renders for older clusters.
        render(kube="1.28.15")

    def test_backoff_limit_and_ttl_must_be_whole_numbers(self):
        for dr, fragment in (
                ({"backoffLimit": "fast"},
                 'dr.backoffLimit must be a non-negative whole number (retries), got "fast"'),
                ({"ttlSecondsAfterFinished": "1d"},
                 'dr.ttlSecondsAfterFinished must be a non-negative whole number (seconds), '
                 'got "1d"')):
            for label, action in (("no action", {}), ("restore", restore())):
                with self.subTest(dr=dr, action=label):
                    render_error(merged(action, {"dr": dr}), fragment=fragment)

    def test_dr_values_must_be_maps(self):
        self.assert_refused([
            ("dr", {"dr": "restore"}, 'dr must be a map of settings (got "restore")'),
            ("dr.restore", {"dr": {"restore": INCIDENT}},
             f'dr.restore must be a map of settings (got "{INCIDENT}")'),
            ("dr.acceptLoss", {"dr": {"acceptLoss": ["x"]}},
             "dr.acceptLoss must be a map of settings"),
            ("dr.resources", {"dr": {"resources": "big"}},
             "dr.resources must be a map of resource settings"),
            ("dr.resources.limits", merged(restore(), {"dr": {"resources": {"limits": "8Gi"}}}),
             "dr.resources.limits must be a map of resource quantities"),
            ("dr.resources.requests",
             merged(restore(), {"dr": {"resources": {"requests": "4Gi"}}}),
             "dr.resources.requests must be a map of resource quantities"),
        ])

    def test_dr_resources_memory(self):
        def resources(**settings):
            return merged(restore(), {"dr": {"resources": settings}})
        self.assert_refused([
            ("a zero limit", resources(limits={"memory": "0"}),
             'dr.resources.limits.memory is "0", which is zero bytes'),
            ("a zero limit, no action", {"dr": {"resources": {"limits": {"memory": "0"}}}},
             'dr.resources.limits.memory is "0", which is zero bytes'),
            ("an unreadable limit", resources(limits={"memory": "5GB"}),
             'dr.resources.limits.memory is "5GB", which is not a Kubernetes memory quantity'),
            ("a request above its limit",
             resources(limits={"memory": "8Gi"}, requests={"memory": "9Gi"}),
             'dr.resources.requests.memory ("9Gi") is above dr.resources.limits.memory ("8Gi") '
             "for the disaster-recovery Job"),
            ("a request above the inherited limit", resources(requests={"memory": "6Gi"}),
             'dr.resources.requests.memory ("6Gi") is above resources.limits.memory ("5Gi") '
             "for the disaster-recovery Job"),
        ])
        render_error(restore(), ["--set", "dr.resources.limits.memory=null"],
                     fragment="dr.resources.limits.memory must be set")

    def test_an_inherited_bad_value_fails_with_the_services_message(self):
        stderr = render_error(merged(restore(), {"resources": {"limits": {"memory": "5GB"}}}),
                              fragment='resources.limits.memory is "5GB", which is not a '
                                       "Kubernetes memory quantity")
        self.assertNotIn("dr.resources", stderr)


class ReusedValuesTests(unittest.TestCase):
    """The reuse chart is a copy whose values.yaml has no dr block, as values
    from a release before it: `helm upgrade --reuse-values` passes those."""

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.reuse_chart = Path(tmp.name) / "live-edit-service"
        shutil.copytree(SERVICE_CHART, cls.reuse_chart)
        values_path = cls.reuse_chart / "values.yaml"
        values = yaml.safe_load(values_path.read_text())
        del values["dr"]
        values_path.write_text(yaml.safe_dump(values, sort_keys=False))

    def test_values_without_dr_render_as_the_chart_without_an_action(self):
        self.assertEqual(render_text(chart=self.reuse_chart), render_text())

    def test_an_action_set_on_values_without_dr_renders_with_kubernetes_defaults(self):
        manifests = render(chart=self.reuse_chart, args=["--set", f"dr.restore.incident={INCIDENT}"])
        job = dr_job(manifests)
        self.assertEqual(job["metadata"]["name"], f"{RELEASE}-dr-restore-{INCIDENT}")
        self.assertNotIn("backoffLimit", job["spec"])
        self.assertNotIn("ttlSecondsAfterFinished", job["spec"])
        self.assertEqual(
            main_container(job["spec"]["template"]["spec"])["resources"],
            main_container(server_deployment(manifests)["spec"]["template"]["spec"])["resources"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-chart", type=Path, default=SERVICE_CHART)
    args, rest = parser.parse_known_args()
    SERVICE_CHART = args.service_chart
    unittest.main(argv=[sys.argv[0], *rest])
