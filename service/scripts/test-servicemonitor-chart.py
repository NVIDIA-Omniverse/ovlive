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

"""Offline ServiceMonitor label regressions. Requires Helm and PyYAML; never
contacts a cluster.

The service labels its own series with `endpoint` (the RPC name) and
`service` (the replica). A Prometheus Operator ServiceMonitor scrape adds
target labels with the same names, and with honorLabels off Prometheus keeps
the target's values and moves the service's to `exported_endpoint` and
`exported_service`. The chart's ServiceMonitor carries metric relabelings
that put the service's values back. These tests check, through a small model
of Prometheus's label handling:

* a series keeps the service's own `endpoint` and `service` wherever the
  service emits them, takes the operator's values elsewhere, and carries no
  `exported_*` label;
* the chart's rules with honorLabels off give the same labels as honorLabels
  on with no rules;
* serviceMonitor.metricRelabelings is appended, verbatim, after the chart's
  rules, so its rules see the restored labels; null, an empty string or a
  release whose values lack the key adds no rules, any other value that is
  not a list fails the render naming it, and the value is inert while the
  ServiceMonitor is disabled;
* of the names the operator or Prometheus may set on a target, the service
  emits only `endpoint` and `service`, each kept through the chart's
  ServiceMonitor rules, and the replica autoscaler emits none.

The model follows these upstream sources, at the revisions named:

* Prometheus v3.5.0 sample-label merge, `mutateSampleLabels` and
  `resolveConflictingExposedLabels`:
  https://github.com/prometheus/prometheus/blob/v3.5.0/scrape/scrape.go#L670-L719
* Prometheus v3.5.0 relabeling: defaults, `Validate` and `relabel`:
  https://github.com/prometheus/prometheus/blob/v3.5.0/model/relabel/relabel.go#L35-L40
  https://github.com/prometheus/prometheus/blob/v3.5.0/model/relabel/relabel.go#L118-L171
  https://github.com/prometheus/prometheus/blob/v3.5.0/model/relabel/relabel.go#L280-L364
* Prometheus v3.5.0 `instance` defaulting to the target address:
  https://github.com/prometheus/prometheus/blob/v3.5.0/scrape/target.go#L518-L519
* Prometheus Operator target labels for a ServiceMonitor endpoint, and the
  `metricRelabelings` to `metric_relabel_configs` conversion:
  https://github.com/prometheus-operator/prometheus-operator/blob/a4f958df0454225026318c53bb4d6c807294358b/pkg/prometheus/promcfg.go#L2159-L2247
  https://github.com/prometheus-operator/prometheus-operator/blob/a4f958df0454225026318c53bb4d6c807294358b/pkg/prometheus/promcfg.go#L2273-L2279
  https://github.com/prometheus-operator/prometheus-operator/blob/a4f958df0454225026318c53bb4d6c807294358b/pkg/prometheus/promcfg.go#L2438-L2475
* The RelabelConfig CRD's action default:
  https://github.com/prometheus-operator/prometheus-operator/blob/a4f958df0454225026318c53bb4d6c807294358b/pkg/apis/monitoring/v1/prometheus_types.go#L2257-L2260

The model fails closed: a relabel action, regex construct or replacement
template it does not model raises `Unmodelled` instead of guessing.
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SERVICE_CHART = ROOT / "helm/live-edit-service"
SERVICE_SRC = ROOT / "src"
SCALER_SRC = ROOT / "scaler/src"
RELEASE = "live-edit-service"
NAMESPACE = "live-edit"
# A server pod's name and address, as Kubernetes would give them. The
# service's `service` label is its pod name by default, so the scraped
# `service` and the target's `pod` carry the same value.
POD = "live-edit-service-6f7c9d8b4-x2k9q"
POD_IP = "10.0.0.12"
# Labels the Prometheus Operator (a ServiceMonitor or a PodMonitor, each a
# subset) or Prometheus may set on a target (promcfg.go, target.go above).
RESERVED = {"namespace", "service", "pod", "container", "node", "job", "endpoint",
            "instance"}


# ── The Prometheus model ────────────────────────────────────────────────────


class Unmodelled(Exception):
    """A rule or template this model does not cover; never guess its effect."""


class InvalidRule(Exception):
    """A rule Prometheus would refuse to load (`Config.Validate`)."""


# relabel.go `DefaultRelabelConfig`.
PROMETHEUS_DEFAULTS = {"separator": ";", "regex": "(.*)", "replacement": "$1",
                       "action": "replace"}
MODELLED_ACTIONS = {"replace", "keep", "drop", "labeldrop", "labelkeep"}
CRD_FIELDS = {"sourceLabels", "separator", "targetLabel", "regex", "modulus",
              "replacement", "action"}
LEGACY_LABEL_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
# Regex syntax on which Python's `re` and RE2 agree for the constructs used
# here: no escapes, no `(?` groups (flags, names, lookaround), no `{n,m}`
# counts and no `[:` (RE2 reads `[[:digit:]]` as a POSIX class, Python as a
# different set). Anything else is unmodelled.
UNMODELLED_REGEX = re.compile(r"[\\{}]|\(\?|\[:")


def set_label(labels, name, value):
    """`labels.Builder.Set`: an empty value deletes the label."""
    if value == "":
        labels.pop(name, None)
    else:
        labels[name] = value


def merge(scraped, target, honor):
    """The scraped series' labels after the target's are attached, before
    metric relabeling: `mutateSampleLabels` / `resolveConflictingExposedLabels`.

    With `honor`, a target label is added only where the series has no label
    of that name. Without it, every target label is set; a scraped label it
    displaces, if non-empty, moves to `exported_<name>` (with `exported_`
    prepended again until the name is free), shortest names first."""
    labels = dict(scraped)
    if honor:
        for name, value in sorted(target.items()):
            if name not in scraped:
                set_label(labels, name, value)
        return labels
    conflicts = []
    for name, value in sorted(target.items()):
        existing = scraped.get(name, "")
        if existing != "":
            conflicts.append((name, existing))
        set_label(labels, name, value)
    conflicts.sort(key=lambda label: len(label[0]))  # stable, as SortStableFunc
    for name, value in conflicts:
        new_name = name
        while True:
            new_name = "exported_" + new_name
            if labels.get(new_name, "") == "":
                labels[new_name] = value
                break
    return labels


def crd_to_prometheus(rule):
    """A ServiceMonitor RelabelConfig as the operator writes it into
    Prometheus's configuration (`generateRelabelConfig`): only the fields set,
    the action lower-cased. An absent action is the CRD's default, `replace`,
    which the API server fills in."""
    unknown = set(rule) - CRD_FIELDS
    if unknown:
        raise Unmodelled(f"unknown RelabelConfig fields {sorted(unknown)} in {rule}")
    config = {}
    if rule.get("sourceLabels"):
        config["source_labels"] = list(rule["sourceLabels"])
    if rule.get("separator") is not None:
        config["separator"] = rule["separator"]
    if rule.get("targetLabel"):
        config["target_label"] = rule["targetLabel"]
    if rule.get("regex"):
        config["regex"] = rule["regex"]
    if rule.get("modulus"):
        config["modulus"] = rule["modulus"]
    if rule.get("replacement") is not None:
        config["replacement"] = rule["replacement"]
    config["action"] = str(rule.get("action") or "replace").lower()
    return config


def load_rule(config):
    """A Prometheus `relabel_config` with the defaults filled in, checked as
    `Config.Validate` checks the actions modelled here."""
    action = config.get("action", PROMETHEUS_DEFAULTS["action"])
    if action not in MODELLED_ACTIONS:
        raise Unmodelled(f"relabel action {action!r} is not modelled")
    rule = {**PROMETHEUS_DEFAULTS, "source_labels": [], "target_label": "",
            "modulus": 0, **config}
    # The replace fast path applies only to the default regex object, which a
    # rule has only when its configuration names no regex at all.
    rule["default_regex"] = "regex" not in config
    if UNMODELLED_REGEX.search(rule["regex"]):
        raise Unmodelled(f"regex {rule['regex']!r} is not modelled")
    rule["compiled"] = re.compile(rule["regex"], re.S)
    if action == "replace":
        if rule["target_label"] == "":
            raise InvalidRule(f"relabel configuration for {action} action requires "
                              f"'target_label' value: {config}")
        if "$" in rule["target_label"]:
            raise Unmodelled(f"target_label template {rule['target_label']!r} "
                             "is not modelled")
        if not LEGACY_LABEL_NAME.fullmatch(rule["target_label"]):
            raise Unmodelled(f"target_label {rule['target_label']!r} is not a "
                             "legacy label name")
    if action in ("labeldrop", "labelkeep") and (
            rule["source_labels"] or rule["target_label"] or rule["modulus"]
            or rule["separator"] != PROMETHEUS_DEFAULTS["separator"]
            or rule["replacement"] != PROMETHEUS_DEFAULTS["replacement"]):
        raise InvalidRule(f"{action} action requires only 'regex', and no other "
                          f"fields: {config}")
    return rule


# A group reference as Go's `regexp.extract` reads one (go1.24 regexp.go
# l.975-1020): an ASCII number without a leading zero, of at most 8 digits,
# braced or not followed by a letter, digit or `_`. Go reads any longer run
# of letters, digits (Unicode ones included) and `_` as a group name. `\w`
# here is Unicode-aware and a superset of Go's name characters, so every
# reference Go reads as a name fails to match and is unmodelled.
TEMPLATE_TOKEN = re.compile(r"\$(?:\{(0|[1-9][0-9]{0,7})\}|(0|[1-9][0-9]{0,7})(?!\w))")


def expand(template, match):
    """`Regexp.ExpandString` for `$N` and `${N}`; any other `$` (a name, `$$`,
    `$1x` or `$1é`, which Go reads as a name, `$01`, a number with a leading
    zero, which Go also reads as a name, a non-ASCII digit) is unmodelled."""
    out = []
    position = 0
    for token in TEMPLATE_TOKEN.finditer(template):
        literal = template[position:token.start()]
        if "$" in literal:
            raise Unmodelled(f"replacement template {template!r} is not modelled")
        out.append(literal)
        index = int(token.group(1) or token.group(2))
        group = match.group(index) if index <= match.re.groups else None
        out.append(group or "")
        position = token.end()
    rest = template[position:]
    if "$" in rest:
        raise Unmodelled(f"replacement template {template!r} is not modelled")
    out.append(rest)
    return "".join(out)


def apply_rule(rule, labels):
    """`relabel()` for one rule: False when the series is dropped."""
    value = rule["separator"].join(labels.get(name, "") for name in rule["source_labels"])
    action = rule["action"]
    if action == "drop":
        return not rule["compiled"].fullmatch(value)
    if action == "keep":
        return bool(rule["compiled"].fullmatch(value))
    if action == "replace":
        if (value == "" and rule["default_regex"]
                and "$" not in rule["target_label"] and "$" not in rule["replacement"]):
            set_label(labels, rule["target_label"], rule["replacement"])
            return True
        match = rule["compiled"].fullmatch(value)
        if match is None:
            return True
        result = expand(rule["replacement"], match)
        if result == "":
            labels.pop(rule["target_label"], None)
        else:
            labels[rule["target_label"]] = result
        return True
    keep_matching = action == "labelkeep"
    for name in list(labels):
        if bool(rule["compiled"].fullmatch(name)) != keep_matching:
            del labels[name]
    return True


def relabel(labels, rules):
    """The series' labels after the ServiceMonitor `metricRelabelings`
    `rules`, in order, or None when a rule drops it (`relabel.Process`)."""
    labels = dict(labels)
    for rule in rules:
        if not apply_rule(load_rule(crd_to_prometheus(rule)), labels):
            return None
    return labels


def scrape(series, target, *, honor, rules):
    """A scraped series as Prometheus stores it."""
    return relabel(merge(series, target, honor), rules)


# ── The chart ───────────────────────────────────────────────────────────────


def monitor_values(extra=None):
    return {"serviceMonitor": {"enabled": True, **(extra or {})}}


def render(values=None, *, chart=None, error=None):
    """`helm template` with `values`: the parsed documents, or with `error`,
    the failure's stderr, asserting the render failed naming `error`."""
    command = ["helm", "template", RELEASE, str(chart or SERVICE_CHART),
               "--namespace", NAMESPACE, "-f", "-"]
    result = subprocess.run(command, input=yaml.safe_dump(values or {}),
                            capture_output=True, text=True, check=False)
    if error is not None:
        if result.returncode == 0 or error not in result.stderr:
            raise AssertionError(f"Expected render failure naming {error!r}: {result.stderr}")
        return result.stderr
    if result.returncode:
        raise AssertionError(result.stderr)
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def only(documents, predicate, what):
    matches = [doc for doc in documents if predicate(doc)]
    assert len(matches) == 1, f"{len(matches)} {what} in the render"
    return matches[0]


def monitor_endpoint(documents):
    monitor = only(documents, lambda doc: doc["kind"] == "ServiceMonitor", "ServiceMonitors")
    endpoints = monitor["spec"]["endpoints"]
    assert len(endpoints) == 1, endpoints
    return monitor, endpoints[0]


def metric_relabelings(documents):
    """The endpoint's metricRelabelings; none when the field is absent."""
    return monitor_endpoint(documents)[1].get("metricRelabelings") or []


def target_labels(documents):
    """The labels the operator and Prometheus put on the server pod's target,
    read from the render: `service` and `job` from the selected Service's
    name, `endpoint` from the endpoint's port name, `container` from the
    server container's name, and `namespace`, `pod` and `instance`
    (promcfg.go l.2159-2247, target.go l.518-519)."""
    monitor, endpoint = monitor_endpoint(documents)
    selector = monitor["spec"]["selector"]["matchLabels"]
    service = only(
        documents,
        lambda doc: doc["kind"] == "Service"
        and selector.items() <= doc["metadata"].get("labels", {}).items(),
        "Services the ServiceMonitor selects")
    port = only(service["spec"]["ports"], lambda p: p["name"] == endpoint["port"],
                "Service ports of that name")
    deployment = only(
        documents,
        lambda doc: doc["kind"] == "Deployment"
        and service["spec"]["selector"].items()
        <= doc["spec"]["template"]["metadata"]["labels"].items(),
        "Deployments the Service selects")
    containers = deployment["spec"]["template"]["spec"]["containers"]
    container = only(containers, lambda c: any(
        p.get("name") == port["targetPort"] for p in c.get("ports", [])),
        "containers serving the metrics port")
    container_port = only(container["ports"], lambda p: p.get("name") == port["targetPort"],
                          "container ports of that name")["containerPort"]
    name = service["metadata"]["name"]
    return {"namespace": NAMESPACE, "service": name, "job": name,
            "endpoint": endpoint["port"], "pod": POD, "container": container["name"],
            "instance": f"{POD_IP}:{container_port}"}


def restored_labels(rules):
    """The labels the rules restore: the target of each replace rule that
    reads `exported_<name>` and writes `<name>`."""
    return {rule["targetLabel"] for rule in rules
            if str(rule.get("action") or "replace").lower() == "replace"
            and rule.get("sourceLabels") == [f"exported_{rule.get('targetLabel')}"]}


# Series as the service's `/metrics` exposes them (the metrics reference).
CALLS = {"__name__": "live_endpoint_calls_total", "endpoint": "UpdateLayer",
         "grpc_code": "OK", "service": POD}
CALLS_OTHER = {**CALLS, "endpoint": "other"}
ACTIVE_SESSIONS = {"__name__": "live_active_sessions", "service": POD}
ADMITTED = {"__name__": "live_edit_subscriber_admitted"}
SERIES = {"calls": CALLS, "calls_other": CALLS_OTHER,
          "active_sessions": ACTIVE_SESSIONS, "admitted": ADMITTED}


# ── Label scan of the binaries' metric sources ──────────────────────────────


KEY_VALUE_SITE = re.compile(r"KeyValue::new\(")
KEY_VALUE_LITERAL = re.compile(r'KeyValue::new\(\s*"([^"\\]*)"')


def emitted_label_names(root):
    """Every label name a `.rs` file under `root` passes to `KeyValue::new`,
    which is how the service and the scaler name their metric labels. Every
    file is read, not only today's metrics modules, and each call must name
    its key with a string literal, so the scan misses none."""
    names = set()
    unreadable = []
    for path in sorted(root.rglob("*.rs")):
        source = path.read_text()
        for site in KEY_VALUE_SITE.finditer(source):
            literal = KEY_VALUE_LITERAL.match(source, site.start())
            if literal is None:
                line = source.count("\n", 0, site.start()) + 1
                unreadable.append(f"{path.relative_to(ROOT)}:{line}")
            else:
                names.add(literal.group(1))
    assert not unreadable, (
        f"KeyValue::new without a string-literal key at {unreadable}; name the key "
        "with a literal so this scan can read it")
    return names


# ── Tests ───────────────────────────────────────────────────────────────────


class PrometheusModelTests(unittest.TestCase):
    """The model against hand-written expected outputs, independent of the
    chart."""

    TARGET = {"endpoint": "metrics", "service": "live-edit-service", "job": "live-edit-service"}

    def test_merge_without_honor_moves_clashing_scraped_labels_to_exported(self):
        merged = merge({"__name__": "m", "endpoint": "UpdateLayer", "service": "pod-a"},
                       self.TARGET, honor=False)
        self.assertEqual(merged, {"__name__": "m", "endpoint": "metrics",
                                  "service": "live-edit-service", "job": "live-edit-service",
                                  "exported_endpoint": "UpdateLayer",
                                  "exported_service": "pod-a"})

    def test_merge_without_honor_prepends_exported_until_the_name_is_free(self):
        merged = merge({"job": "a", "exported_job": "b"}, {"job": "j"}, honor=False)
        self.assertEqual(merged, {"job": "j", "exported_job": "b",
                                  "exported_exported_job": "a"})

    def test_merge_with_honor_keeps_scraped_labels_and_adds_the_rest(self):
        merged = merge({"endpoint": "UpdateLayer"}, self.TARGET, honor=True)
        self.assertEqual(merged, {"endpoint": "UpdateLayer", "service": "live-edit-service",
                                  "job": "live-edit-service"})

    def test_a_match_anything_replace_from_an_absent_source_deletes_the_target(self):
        # Why the chart's rules match `(.+)`: with `(.*)` a series that has no
        # exported_endpoint would lose the operator's endpoint label.
        rule = {"sourceLabels": ["exported_endpoint"], "regex": "(.*)",
                "targetLabel": "endpoint", "replacement": "$1"}
        self.assertEqual(relabel({"endpoint": "metrics"}, [rule]), {})
        self.assertEqual(relabel({"endpoint": "metrics"}, [{**rule, "regex": "(.+)"}]),
                         {"endpoint": "metrics"})

    def test_replace_expands_both_group_forms_and_copies_the_value(self):
        for replacement in ("$1", "${1}"):
            with self.subTest(replacement):
                rule = {"sourceLabels": ["exported_endpoint"], "regex": "(.+)",
                        "targetLabel": "endpoint", "replacement": replacement}
                self.assertEqual(relabel({"endpoint": "metrics",
                                          "exported_endpoint": "UpdateLayer"}, [rule]),
                                 {"endpoint": "UpdateLayer", "exported_endpoint": "UpdateLayer"})

    def test_replace_without_a_regex_sets_the_literal_on_an_empty_source(self):
        # The fast path: the default regex object, no template in either field.
        self.assertEqual(relabel({}, [{"targetLabel": "tier", "replacement": "edge"}]),
                         {"tier": "edge"})

    def test_labeldrop_drop_and_keep(self):
        labels = {"endpoint": "other", "exported_endpoint": "x", "exported_service": "y"}
        self.assertEqual(relabel(labels, [{"regex": "exported_(endpoint|service)",
                                           "action": "labeldrop"}]),
                         {"endpoint": "other"})
        self.assertIsNone(relabel(labels, [{"sourceLabels": ["endpoint"], "regex": "other",
                                            "action": "drop"}]))
        self.assertIsNone(relabel(labels, [{"sourceLabels": ["endpoint"], "regex": "metrics",
                                            "action": "Keep"}]))

    def test_the_model_fails_closed(self):
        def template(replacement):
            return {"sourceLabels": ["a"], "regex": "(.+)", "targetLabel": "b",
                    "replacement": replacement}

        cases = {
            "an unmodelled action": ({"sourceLabels": ["a"], "targetLabel": "b",
                                      "modulus": 2, "action": "hashmod"}, Unmodelled),
            "a named template": (template("$name"), Unmodelled),
            "a group followed by a word character": (template("$1x"), Unmodelled),
            # Go reads each of these as a group name, not group 1.
            "a group number with a leading zero": (template("$01"), Unmodelled),
            "a braced group number with a leading zero": (template("${01}"), Unmodelled),
            "a group followed by a non-ASCII letter": (template("$1é"), Unmodelled),
            "a non-ASCII digit": (template("$١"), Unmodelled),
            "an escape in the regex": ({"sourceLabels": ["a"], "regex": r"\d+",
                                        "targetLabel": "b"}, Unmodelled),
            "a POSIX class in the regex": ({"sourceLabels": ["a"], "regex": "[[:alpha:]]+",
                                            "targetLabel": "b"}, Unmodelled),
            "an unknown field": ({"targetLabel": "b", "replacment": "x"}, Unmodelled),
            "replace without a target": ({"sourceLabels": ["a"]}, InvalidRule),
            "labeldrop with a source": ({"sourceLabels": ["a"], "regex": "a",
                                         "action": "labeldrop"}, InvalidRule),
        }
        for name, (rule, error) in cases.items():
            with self.subTest(name), self.assertRaises(error):
                relabel({"a": "v"}, [rule])


class ServiceMonitorLabelTests(unittest.TestCase):
    """The chart's rendered ServiceMonitor through the model."""

    @classmethod
    def setUpClass(cls):
        cls.docs = render(monitor_values())
        cls.target = target_labels(cls.docs)
        cls.rules = metric_relabelings(cls.docs)

    def stored(self, series, *, honor=False, rules=None):
        return scrape(series, self.target, honor=honor,
                      rules=self.rules if rules is None else rules)

    def test_the_target_labels_are_the_operators(self):
        # A self-check of the render reading: the clash exists only because
        # the operator's values differ from the service's.
        self.assertEqual(self.target["endpoint"], "metrics")
        self.assertEqual(self.target["service"], RELEASE)
        self.assertEqual(set(self.target), RESERVED - {"node"})

    def test_a_series_keeps_the_services_endpoint_and_service(self):
        # calls_other keeps endpoint="other", so `endpoint!="other"` still filters.
        for name in ("calls", "calls_other"):
            with self.subTest(name):
                series = SERIES[name]
                self.assertEqual(self.stored(series), {**self.target, **series})

    def test_a_series_with_service_only_takes_the_operators_endpoint(self):
        stored = self.stored(ACTIVE_SESSIONS)
        self.assertEqual(stored, {**self.target, **ACTIVE_SESSIONS})
        self.assertEqual((stored["service"], stored["endpoint"]), (POD, "metrics"))

    def test_a_label_less_series_takes_the_operators_endpoint_and_service(self):
        stored = self.stored(ADMITTED)
        self.assertEqual(stored, {**self.target, **ADMITTED})
        self.assertEqual((stored["service"], stored["endpoint"]), (RELEASE, "metrics"))

    def test_no_series_keeps_an_exported_label(self):
        for name, series in SERIES.items():
            with self.subTest(name):
                stored = self.stored(series)
                self.assertEqual([label for label in stored if label.startswith("exported_")],
                                 [], stored)

    def test_the_rules_without_honor_equal_honor_with_no_rules(self):
        # Why the chart does not set honorLabels: its rules give the same
        # series, and keep working where the Prometheus resource forces
        # honorLabels off.
        for name, series in SERIES.items():
            with self.subTest(name):
                self.assertEqual(self.stored(series),
                                 self.stored(series, honor=True, rules=[]))


class MetricRelabelingsValueTests(unittest.TestCase):
    """serviceMonitor.metricRelabelings: appended after the chart's rules,
    and type-checked. The reuse chart is a copy whose values.yaml lacks the
    key, as values from a release before it: `helm upgrade --reuse-values`
    passes those."""

    USER_RULE = {"sourceLabels": ["endpoint"], "regex": "other", "action": "drop"}

    @classmethod
    def setUpClass(cls):
        cls.chart_rules = metric_relabelings(render(monitor_values()))
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.reuse_chart = Path(tmp.name) / "live-edit-service"
        shutil.copytree(SERVICE_CHART, cls.reuse_chart)
        values_path = cls.reuse_chart / "values.yaml"
        values = yaml.safe_load(values_path.read_text())
        del values["serviceMonitor"]["metricRelabelings"]
        values_path.write_text(yaml.safe_dump(values, sort_keys=False))

    def test_the_chart_renders_three_rules_of_its_own(self):
        self.assertEqual(len(self.chart_rules), 3, self.chart_rules)
        self.assertEqual(restored_labels(self.chart_rules), {"endpoint", "service"})

    def test_user_rules_follow_the_charts_verbatim(self):
        docs = render(monitor_values({"metricRelabelings": [self.USER_RULE]}))
        self.assertEqual(metric_relabelings(docs), self.chart_rules + [self.USER_RULE])

    def test_user_rules_see_the_restored_labels(self):
        # Run first, the rule would see endpoint="metrics" on every series
        # and drop nothing.
        docs = render(monitor_values({"metricRelabelings": [self.USER_RULE]}))
        target, rules = target_labels(docs), metric_relabelings(docs)
        self.assertIsNone(scrape(CALLS_OTHER, target, honor=False, rules=rules))
        self.assertEqual(scrape(CALLS, target, honor=False, rules=rules), {**target, **CALLS})

    def test_null_absent_and_empty_render_only_the_charts_rules(self):
        cases = {
            "empty": render(monitor_values({"metricRelabelings": []})),
            "null": render(monitor_values({"metricRelabelings": None})),
            "the empty string": render(monitor_values({"metricRelabelings": ""})),
            "absent from reused values": render(monitor_values(), chart=self.reuse_chart),
            "null over reused values": render(monitor_values({"metricRelabelings": None}),
                                              chart=self.reuse_chart),
        }
        for name, docs in cases.items():
            with self.subTest(name):
                self.assertEqual(metric_relabelings(docs), self.chart_rules)

    def test_a_value_that_is_not_a_list_fails_the_render_naming_it(self):
        # The empty non-lists are falsy in a template, so a guard that only
        # tests truthiness lets them through with no rules added.
        cases = (
            ("a map", self.USER_RULE,
             '(got map[string]interface {}{"action":"drop", "regex":"other", '
             '"sourceLabels":[]interface {}{"endpoint"}})'),
            ("a string", "drop", '(got "drop")'),
            ("a number", 5, "(got 5)"),
            ("an empty map", {}, "(got map[string]interface {}{})"),
            ("false", False, "(got false)"),
            ("zero", 0, "(got 0)"),
        )
        for name, value, got in cases:
            with self.subTest(name):
                render(monitor_values({"metricRelabelings": value}),
                       error=f"serviceMonitor.metricRelabelings must be a list of "
                             f"relabeling rules {got}")

    def test_the_value_is_inert_while_the_service_monitor_is_disabled(self):
        for name, value in (("a list", [self.USER_RULE]), ("a map", self.USER_RULE),
                            ("a string", "drop")):
            with self.subTest(name):
                docs = render({"serviceMonitor": {"enabled": False,
                                                  "metricRelabelings": value}})
                self.assertEqual([doc for doc in docs if doc["kind"] == "ServiceMonitor"], [])


class EmittedLabelNameTests(unittest.TestCase):
    """The forward guard: a label the service or the scaler emits under a
    name the operator or Prometheus sets on targets is renamed to
    `exported_<name>` by any honorLabels-off scrape that sets that name as a
    target label. The chart's rules restore only the service's `endpoint`
    and `service`, and only for the chart's own ServiceMonitor; any other
    scrape (a PodMonitor, an annotation-driven scrape config) restores
    nothing, so it renames whichever of them it sets as a target label. So
    the service may emit those two reserved names and no other, and the
    scaler none."""

    # The two deliberate exceptions. Adding one is a design change (SDD
    # §17.1, the metric-label rules), not a test change.
    SERVICE_EXCEPTIONS = {"endpoint", "service"}

    def test_the_service_emits_no_reserved_name_but_its_two_exceptions(self):
        names = emitted_label_names(SERVICE_SRC)
        self.assertIn("grpc_code", names, "scan self-check: the service's labels were read")
        self.assertLessEqual(
            names & RESERVED, self.SERVICE_EXCEPTIONS,
            "the service emits a label under a name the operator or Prometheus sets on "
            "targets, so scrapes rename it to exported_<name>: rename the label. Making "
            "it a further exception to the endpoint/service rule is a design change, "
            "and needs a restore rule in templates/servicemonitor.yaml and its labeldrop "
            "as well")

    def test_every_reserved_name_the_service_emits_survives_the_scrape(self):
        names = emitted_label_names(SERVICE_SRC)
        self.assertIn("grpc_code", names, "scan self-check: the service's labels were read")
        docs = render(monitor_values())
        target, rules = target_labels(docs), metric_relabelings(docs)
        for name in sorted(names & RESERVED):
            with self.subTest(name):
                stored = scrape({"__name__": "m", name: "own"}, target, honor=False,
                                rules=rules)
                self.assertEqual(
                    (stored.get(name),
                     sorted(label for label in stored if label.startswith("exported_"))),
                    ("own", []),
                    f"the service emits `{name}`, which the operator or Prometheus also "
                    f"sets on targets, and the stored series {stored} does not keep the "
                    "service's value alone: rename the label, unless it is one of the "
                    "deliberate exceptions, whose restore rule or labeldrop in "
                    "templates/servicemonitor.yaml no longer covers it")

    def test_the_scaler_emits_no_reserved_name(self):
        names = emitted_label_names(SCALER_SRC)
        self.assertIn("scope", names, "scan self-check: the scaler's labels were read")
        self.assertEqual(
            names & RESERVED, set(),
            "the scaler's metrics are scraped by a PodMonitor or scrape config the "
            "chart does not render, so no rule restores a clashing label: rename it")


if __name__ == "__main__":
    unittest.main(verbosity=2)
