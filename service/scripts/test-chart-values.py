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

"""Offline chart value regressions: value naming, value checks, the server's
memory request and the checkpoint tick's maximum. Requires Helm and PyYAML;
never contacts a cluster.

* Every value in the service chart's values.yaml is camelCase: no key holds
  an underscore.
* Every Helm value the configuration page (docs/service/docs/configuration.md)
  names exists in values.yaml, so the page cannot name a value the chart
  does not read.
* The renamed-value check (`live-edit-service.renamedValueChecks` in
  _helpers.tpl) maps each old snake_case name to its camelCase name, one
  capital per underscore, and every new name is a value the chart has. Any
  old name, from `--set` or a values file, fails the render naming the new
  name, even beside a value that fails another check in a template Helm
  renders before configmap.yaml (deployment.yaml's termination grace); several
  fail in one message; and a `helm upgrade --reuse-values` from a chart with
  the old names fails the same way, with the `--reset-then-reuse-values`
  remedy. That case is simulated by rendering this chart's templates over
  today's values.yaml with the 45 names put back: it models the renamed
  values, not every other difference of an actual earlier chart.
* `valkey.commandTimeoutMs` of 0, however it is written, fails the render
  naming the default to set instead, as the binary refuses 0 at startup; a
  positive value renders as set.
* `checkpoint.notFoundCacheSecs` (read only by the DR Jobs) renders 900 by
  default and any whole number of seconds as set, 0 included; a negative,
  fractional or non-numeric value fails the render naming it.
* The server container's memory request. Unset (absent, null or empty), it
  renders 70% of resources.limits.memory, rounded down to whole Mi, for every
  quantity form Kubernetes accepts: binary and decimal units, plain bytes,
  decimal fractions, decimal exponents (with Kubernetes' 32-bit narrowing of
  the exponent) and values-file or `--set` numbers. A set request renders
  exactly as written, an older chart's 1Gi included (what `helm upgrade
  --reuse-values` carries over), and one above the limit fails the render,
  compared at Kubernetes' nanobyte precision. A limit or request in a form
  Kubernetes refuses, with the m, u or n suffix, with a decimal exponent below
  -1000 as Kubernetes narrows it (the chart's own bound, named in the
  message), negative, or above 8Pi fails naming the value, as do a zero limit
  and a resources.requests that is not a map; an unset limit keeps its "must
  be set" message. The CPU request, extra resources and the scaler's CPU
  request are untouched. Expected figures come from an exact rational oracle
  in this file (`quantity_bytes`), never from the chart's own output.
* checkpoint.tick: 19 and the default 10 render, and 20 or more fails the
  render naming the value, its maximum and the setting to change, as the
  binary refuses a tick of 20 s or more (two thirds of the 30 s checkpoint
  leader-lock TTL).
* The integer rows (`IntegerRowTests`): sixteen ConfigMap rows,
  minReadySeconds and tests.ttlSecondsAfterFinished go through the
  nonNegativeInteger helper. Text, a fraction, a negative number, "5.0" and
  "0x10" fail the render naming the key; surrounding space and a leading "+"
  are accepted; a values-file million renders as digits; each row's default
  digits are pinned. Null and "" keep each row's meaning (refused, left out,
  derived, or `required`'s "must be set"), as do the 0 sentinels: 0 renders
  where it means "off" or "never wait", and leaves the variable or field out
  where it means "unset", however it is written. A tracing batch setting of 0
  fails as below its minimum of 1. A `"` or `\\` in a text value fails naming
  templates/configmap.yaml.
"""

from fractions import Fraction
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import NamedTuple
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SERVICE_CHART = ROOT / "helm/live-edit-service"
VALUES = SERVICE_CHART / "values.yaml"
HELPERS = SERVICE_CHART / "templates/_helpers.tpl"
CONFIG_PAGE = ROOT / "docs/service/docs/configuration.md"
RELEASE = "live-edit-service"
REMEDY = "--reset-then-reuse-values"


def render(chart=SERVICE_CHART, *args):
    return subprocess.run(["helm", "template", RELEASE, str(chart), *args],
                          capture_output=True, text=True, check=False)


def camel(name):
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def renamed():
    """(section, old, new) triples from the renamed-value check's list."""
    text = HELPERS.read_text()
    block = text[text.index('define "live-edit-service.renamedValueChecks"'):]
    block = block[:block.index("$found := list")]
    return re.findall(r'"([A-Za-z]+) ([a-z0-9_]+) ([A-Za-z0-9]+)"', block)


def key_paths(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path + (str(key),)
            yield from key_paths(value, path + (str(key),))


def has_path(values, dotted):
    node = values
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


LIMIT = "resources.limits.memory"
REQUEST = "resources.requests.memory"
MIB = 1024 ** 2
# Kubernetes' quantity grammar for the forms the tests derive a figure from:
# a sign, a decimal number with at least one digit, and at most one suffix, a
# binary or decimal unit or a decimal exponent.
QUANTITY = re.compile(
    r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))(Ki|Mi|Gi|Ti|Pi|Ei|[eE][+-]?[0-9]+|[kMGTPE])?")
BINARY_UNITS = {"Ki": 1, "Mi": 2, "Gi": 3, "Ti": 4, "Pi": 5, "Ei": 6}
DECIMAL_UNITS = {"": 0, "k": 3, "M": 6, "G": 9, "T": 12, "P": 15, "E": 18}


def quantity_bytes(text):
    """The oracle: a quantity's value in whole bytes, computed exactly with
    rationals and rounded up, as Kubernetes rounds a quantity's value. A
    decimal exponent keeps only its low 32 bits, as a signed number, as
    Kubernetes reads it."""
    number, suffix = QUANTITY.fullmatch(str(text).strip()).groups()
    suffix = suffix or ""
    mantissa = Fraction(number)
    if suffix in BINARY_UNITS:
        value = mantissa * 1024 ** BINARY_UNITS[suffix]
    elif suffix in DECIMAL_UNITS:
        value = mantissa * 10 ** DECIMAL_UNITS[suffix]
    else:
        exponent = (int(suffix[1:]) + 2 ** 31) % 2 ** 32 - 2 ** 31
        value = mantissa * Fraction(10) ** exponent
    return math.ceil(value)


def derived_request(limit):
    """The request the chart derives from `limit`: 70% of it, rounded down to
    whole Mi."""
    return f"{quantity_bytes(limit) * 7 // 10 // MIB}Mi"


class Raw(str):
    """A values-file scalar written as is, so YAML reads it as a number
    (5e9, 5368709120), not a string."""


UNSET = object()


def memory_values(limit=UNSET, request=UNSET, extra_requests=None, extra_limits=None):
    """A values file setting the server's memory limit and request. A Raw
    value is written as is; any other value as JSON, which is YAML: a string
    quoted, None as null, True as true."""
    def scalar(value):
        return str(value) if isinstance(value, Raw) else json.dumps(value)
    lines = ["resources:"]
    for section, value, extra in (("limits", limit, extra_limits),
                                  ("requests", request, extra_requests)):
        entries = [] if value is UNSET else [f"    memory: {scalar(value)}"]
        entries += [f"    {key}: {scalar(item)}" for key, item in (extra or {}).items()]
        if entries:
            lines += [f"  {section}:", *entries]
    return "\n".join(lines) + "\n"


def render_server(*args, values=None, template="templates/deployment.yaml"):
    """`helm template` of one template, with `values` (a values-file text)
    and `args`. Every template still renders, so a failing check anywhere in
    the chart fails this render too."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        if values is not None:
            f.write(values)
            f.flush()
            args = ("-f", f.name, *args)
        return render(SERVICE_CHART, "--show-only", template, *args)


def server_resources(result):
    """The server container's resources from a deployment.yaml render."""
    deployment = yaml.safe_load(result.stdout)
    return deployment["spec"]["template"]["spec"]["containers"][0]["resources"]


class ChartValueNamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = yaml.safe_load(VALUES.read_text())
        cls.renamed = renamed()

    def test_no_value_name_holds_an_underscore(self):
        snake = [".".join(p) for p in key_paths(self.values) if "_" in p[-1]]
        self.assertEqual(snake, [], "values.yaml names must be camelCase")

    def test_every_value_on_the_configuration_page_exists(self):
        missing = []
        for line in CONFIG_PAGE.read_text().splitlines():
            match = re.match(r"\|\s*`([a-z][A-Za-z0-9.]*)`\s*\|", line)
            if match and not has_path(self.values, match.group(1)):
                missing.append(match.group(1))
        self.assertEqual(missing, [], f"{CONFIG_PAGE.name} names values the chart lacks")

    def test_the_renamed_list_maps_each_old_name_to_an_existing_value(self):
        self.assertEqual(len(self.renamed), 45)
        for section, old, new in self.renamed:
            with self.subTest(f"{section}.{old}"):
                self.assertEqual(new, camel(old))
                self.assertTrue(has_path(self.values, f"{section}.{new}"),
                                f"{section}.{new} is not in values.yaml")

    def test_each_old_name_fails_the_render_naming_its_new_name(self):
        for section, old, new in self.renamed:
            with self.subTest(f"{section}.{old}"):
                result = render(SERVICE_CHART, "--set", f"{section}.{old}=1")
                self.assertNotEqual(result.returncode, 0, "the render succeeded")
                self.assertIn(f"{section}.{old} is now {section}.{new}", result.stderr)
                self.assertIn(REMEDY, result.stderr)

    def test_an_old_name_wins_over_a_check_rendered_before_configmap(self):
        # deployment.yaml renders before configmap.yaml and its grace check
        # reads renamed values: an old-name values file with a grace that was
        # valid for it must get the rename message, not a grace error.
        result = render(SERVICE_CHART, "--set", "shutdown.overall_budget_secs=20",
                        "--set", "terminationGracePeriodSeconds=70")
        self.assertNotEqual(result.returncode, 0, "the render succeeded")
        self.assertIn("shutdown.overall_budget_secs is now shutdown.overallBudgetSecs",
                      result.stderr)
        self.assertNotIn("terminationGracePeriodSeconds (70)", result.stderr)

    def test_several_old_names_fail_once_listing_all(self):
        overlay = {"valkey": {"memory_bytes": 12000000000, "pool_size": 64},
                   "rpc": {"deadline_ms": 30000}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            yaml.safe_dump(overlay, f)
            f.flush()
            result = render(SERVICE_CHART, "-f", f.name)
        self.assertNotEqual(result.returncode, 0, "the render succeeded")
        for name in ("valkey.memory_bytes is now valkey.memoryBytes",
                     "valkey.pool_size is now valkey.poolSize",
                     "rpc.deadline_ms is now rpc.deadlineMs"):
            self.assertIn(name, result.stderr)
        self.assertEqual(result.stderr.count("renamed to camelCase"), 1)

    def test_reuse_values_from_a_chart_with_the_old_names_fails_with_the_remedy(self):
        # `--reuse-values` renders the new templates over the previous
        # chart's values: every old name, none of the new ones.
        old_values = yaml.safe_load(VALUES.read_text())
        for section, old, new in self.renamed:
            old_values[section][old] = old_values[section].pop(new)
        with tempfile.TemporaryDirectory() as tmp:
            chart = Path(tmp) / "live-edit-service"
            shutil.copytree(SERVICE_CHART, chart)
            (chart / "values.yaml").write_text(yaml.safe_dump(old_values))
            result = render(chart)
        self.assertNotEqual(result.returncode, 0, "the render succeeded")
        self.assertIn("renamed to camelCase", result.stderr)
        self.assertIn("valkey.memory_bytes is now valkey.memoryBytes", result.stderr)
        self.assertIn(REMEDY, result.stderr)

    def test_the_default_render_passes(self):
        result = render()
        self.assertEqual(result.returncode, 0, result.stderr)


COMMAND_TIMEOUT_ZERO =("valkey.commandTimeoutMs is 0, which is no longer accepted: 0 meant "
                        "no timeout, and the service refuses it at startup. Set a timeout in "
                        "milliseconds, such as the default 5000")


class ChartValueCheckTests(unittest.TestCase):
    def test_a_zero_command_timeout_fails_the_render_naming_the_default(self):
        # The binary refuses LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS=0 at startup;
        # the chart refuses it first, however 0 is written.
        for flag, value in (("--set", "0"), ("--set", "00"), ("--set-string", "0"),
                            ("--set-string", "00"), ("--set-string", "+0")):
            with self.subTest(f"{flag} {value}"):
                result = render(SERVICE_CHART, flag, f"valkey.commandTimeoutMs={value}")
                self.assertNotEqual(result.returncode, 0, "the render succeeded")
                self.assertIn(COMMAND_TIMEOUT_ZERO, result.stderr)

    def test_a_positive_command_timeout_renders_as_set(self):
        result = render(SERVICE_CHART, "--set", "valkey.commandTimeoutMs=1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS: "1"', result.stdout)
        self.assertNotIn(COMMAND_TIMEOUT_ZERO, result.stderr)

    def test_the_not_found_cache_renders_its_default_and_any_whole_number_of_seconds(self):
        # checkpoint.notFoundCacheSecs, read only by the DR Jobs: 900 by
        # default, 0 allowed, as the binary takes any whole number.
        result = render()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('LIVE_EDIT_CHECKPOINT_STORAGE_API_NOT_FOUND_CACHE_SECS: "900"',
                      result.stdout)
        for value in ("0", "3600"):
            with self.subTest(value):
                result = render(SERVICE_CHART, "--set", f"checkpoint.notFoundCacheSecs={value}")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f'LIVE_EDIT_CHECKPOINT_STORAGE_API_NOT_FOUND_CACHE_SECS: "{value}"',
                    result.stdout)

    def test_a_not_found_cache_that_is_not_a_whole_number_fails_the_render_naming_it(self):
        # The binary refuses these at startup; the chart refuses them first.
        for value in ("-1", "1.5", "x"):
            with self.subTest(value):
                result = render(SERVICE_CHART, "--set-string",
                                f"checkpoint.notFoundCacheSecs={value}")
                self.assertNotEqual(result.returncode, 0, "the render succeeded")
                self.assertIn("checkpoint.notFoundCacheSecs must be a non-negative whole number "
                              "(seconds)", result.stderr)


class ServerMemoryRequestTests(unittest.TestCase):
    """The server container's memory request against its memory limit."""

    MUST_BE_SET = f"{LIMIT} must be set:"
    UNREADABLE = "which is not a Kubernetes memory quantity"
    SUB_BYTE = "almost certainly a typo for a pod memory value"
    NEGATIVE = "a negative quantity"
    TOO_LARGE = "above 8Pi (9007199254740992 bytes)"
    ZERO = "which is zero bytes"
    ABOVE_LIMIT = f"is above {LIMIT}"

    def rendered_resources(self, *args, values=None):
        result = render_server(*args, values=values)
        self.assertEqual(result.returncode, 0, result.stderr)
        return server_resources(result)

    def assert_derived(self, limit, *args, values=None):
        """The render derives the request from `limit`, at or below it."""
        resources = self.rendered_resources(*args, values=values)
        self.assertEqual(resources["requests"]["memory"], derived_request(limit))
        self.assertLessEqual(quantity_bytes(resources["requests"]["memory"]),
                             quantity_bytes(limit))
        return resources

    def assert_refused(self, message, *args, values=None):
        result = render_server(*args, values=values)
        self.assertNotEqual(result.returncode, 0, "the render succeeded")
        self.assertIn(message, result.stderr)
        return result.stderr

    def test_the_oracle_gives_the_documented_figures(self):
        # Hand-checked figures, so the oracle every test below uses cannot
        # drift with the chart.
        for limit, request in (("5Gi", "3584Mi"), ("2709Mi", "1896Mi"), ("4Gi", "2867Mi"),
                               ("8Gi", "5734Mi"), ("5G", "3337Mi"), ("1Ti", "734003Mi"),
                               ("1T", "667572Mi"), ("2.5Gi", "1792Mi"), ("4.5G", "3004Mi"),
                               ("010Gi", "7168Mi"), ("5e9", "3337Mi"), ("5.5e9", "3671Mi"),
                               ("5368709120000e-3", "3584Mi"), ("5e4294967305", "3337Mi"),
                               ("2709.5", "0Mi"), ("3e0", "0Mi"), ("0.5", "0Mi"),
                               ("5e-9223372036854775808", "0Mi"), ("1e-1000", "0Mi"),
                               ("1e4294966296", "0Mi"),
                               # 1497965.1 bytes rounds up to 1497966, which is
                               # the smallest limit giving 1Mi.
                               ("1497965.1", "1Mi"), ("1497965.0000000001", "1Mi"),
                               ("1497965", "0Mi")):
            with self.subTest(limit=limit):
                self.assertEqual(derived_request(limit), request)

    def test_the_default_request_is_70_percent_of_the_default_limit(self):
        resources = self.rendered_resources()
        self.assertEqual(resources, {"requests": {"cpu": 1, "memory": "3584Mi"},
                                     "limits": {"cpu": 2, "memory": "5Gi"}})
        self.assertEqual(derived_request("5Gi"), "3584Mi")

    def test_an_unset_request_follows_each_form_of_the_limit(self):
        # 2709Mi rules out a fixed default; the decimal and fractional forms
        # rule out a binary-only reader; 010Gi rules out reading a leading 0
        # as octal (8Gi, 5734Mi); 1497965.1 rules out rounding the limit to
        # the nearest byte or down (0Mi) instead of up, and 1497965.0000000001
        # rules out dropping the digits below a nanobyte instead of rounding
        # them up; 2709.5, 3e0 and 0.5 (half a byte, so not zero) are limits
        # too small for a whole Mi; 8Pi is the largest accepted.
        for limit in ("2709Mi", "5Gi", "5120Mi", "5242880Ki", "1Ti", "4Gi", "8Gi",
                      "5G", "5000M", "5000000k", "1T", "5368709120", "2.5Gi", "4.5G",
                      "+5Gi", " 5Gi ", "010Gi", "1497965.1", "1497965.0000000001",
                      "2709.5", "3e0", "0.5", "8Pi", "9007199254740992"):
            with self.subTest(limit=limit):
                self.assert_derived(limit, values=memory_values(limit))

    def test_an_unset_request_follows_a_limit_with_a_decimal_exponent(self):
        # 5e4294967305 is 5e9: Kubernetes keeps the exponent's low 32 bits.
        # 5e-9223372036854775808 has the lowest 64-bit exponent, whose low
        # 32 bits are 0, so it is 5 bytes. 1e-1000 and 1e4294966296 (which
        # narrows to e-1000) sit on the chart's lowest exponent.
        for limit in ("5e9", "5E9", "5E+9", "5e+09", "5.5e9", "5368709120000e-3",
                      "5.36870912e9", "5e4294967305", "5e-9223372036854775808",
                      "1e-1000", "1e4294966296"):
            with self.subTest(limit=limit):
                self.assert_derived(limit, values=memory_values(limit))

    def test_an_unset_request_follows_a_numeric_limit(self):
        # A values-file number is a float (5368709120 renders through
        # 5.36870912e+09); a --set number is an integer.
        for limit in ("5368709120", "5e9", "5.0e+9", "2.5", "2709.5"):
            with self.subTest(values_file=limit):
                resources = self.assert_derived(limit, values=memory_values(Raw(limit)))
                self.assertNotIsInstance(resources["limits"]["memory"], str)
        with self.subTest(set="5368709120"):
            resources = self.assert_derived("5368709120", "--set", f"{LIMIT}=5368709120")
            self.assertEqual(resources["limits"]["memory"], 5368709120)

    def test_a_null_or_empty_request_is_unset(self):
        for args, values in ((("--set", f"{REQUEST}=null"), None),
                             (("--set-string", f"{REQUEST}="), None),
                             ((), memory_values(request=None)),
                             ((), memory_values(request=""))):
            with self.subTest(args=args, values=values):
                self.assert_derived("5Gi", *args, values=values)

    def test_a_set_request_renders_exactly_as_written(self):
        # 1Gi is the request an older chart's values carry into `helm
        # upgrade --reuse-values`: it is a set request, and kept. 5Gi and
        # 5120Mi equal the 5Gi limit; 900Mi is a whole digit shorter in
        # nanobytes. 4e9 and 1e-1000 keep their exponents as written. 0, -0
        # and Ki read as zero, which Kubernetes accepts for a request.
        for request in ("2Gi", "1Gi", "5Gi", "5120Mi", "900Mi", "5368709120", "4e9",
                        "1e-1000", "0", "-0", "Ki"):
            with self.subTest(request=request):
                resources = self.rendered_resources(values=memory_values(request=request))
                self.assertEqual(resources["requests"]["memory"], request)
        with self.subTest(set=2147483648):
            resources = self.rendered_resources("--set", f"{REQUEST}=2147483648")
            self.assertEqual(resources["requests"]["memory"], 2147483648)
        with self.subTest(values_file=2147483648):
            resources = self.rendered_resources(values=memory_values(request=Raw("2147483648")))
            self.assertEqual(resources["requests"]["memory"], 2147483648)
        with self.subTest(set=0):
            resources = self.rendered_resources("--set", f"{REQUEST}=0")
            self.assertEqual(resources["requests"]["memory"], 0)

    def test_a_request_above_the_limit_fails_naming_both(self):
        # 5121Mi, 5.5G and 5.5e9 are above 5Gi only when units are converted;
        # 10Gi is a whole digit longer in nanobytes.
        for request in ("6Gi", "10Gi", "5121Mi", "5.5G", "5.5e9", "5368709121"):
            with self.subTest(request=request):
                stderr = self.assert_refused(self.ABOVE_LIMIT,
                                             values=memory_values("5Gi", request))
                self.assertIn(f'{REQUEST} ("{request}")', stderr)
                self.assertIn('("5Gi")', stderr)
                self.assertIn("3584Mi", stderr)

    def test_request_and_limit_compare_at_nanobyte_precision(self):
        # Both round up to 5368709121 whole bytes, so a whole-byte
        # comparison would pass the first; Kubernetes compares to the
        # nanobyte and refuses it.
        self.assert_refused(self.ABOVE_LIMIT,
                            values=memory_values("5368709120.4", "5368709120.5"))
        resources = self.rendered_resources(
            values=memory_values("5368709120.5", "5368709120.4"))
        self.assertEqual(resources["requests"]["memory"], "5368709120.4")
        # Below a nanobyte Kubernetes rounds up: a tenth of a nanobyte over
        # 5Gi is a whole nanobyte over it, and a tenth under 5368709121 rounds
        # (carrying through every 9) to exactly it, as 0.9999999999 does to 1.
        self.assert_refused(self.ABOVE_LIMIT,
                            values=memory_values("5Gi", "5368709120.0000000001"))
        for limit, request in (("5368709121", "5368709120.9999999999"),
                               ("1", "0.9999999999")):
            with self.subTest(limit=limit, request=request):
                resources = self.rendered_resources(values=memory_values(limit, request))
                self.assertEqual(resources["requests"]["memory"], request)

    # Each refused form, with the message that names why.
    REFUSED = (
        ("512m", SUB_BYTE), ("512u", SUB_BYTE), ("512n", SUB_BYTE),
        ("5GB", UNREADABLE), ("5gi", UNREADABLE), ("5K", UNREADABLE),
        ("5e", UNREADABLE), ("5e3Mi", UNREADABLE), ("abc", UNREADABLE),
        ("   ", UNREADABLE), (True, UNREADABLE),
        # No digit, where Kubernetes refuses that rather than read zero: with
        # Pi or Ei, or with an exponent below -9.
        ("Pi", UNREADABLE), ("e-10", UNREADABLE),
        # Exponents beyond 64 bits. -2^63 itself is accepted (it narrows to
        # 0); one below it is not.
        ("1e99999999999999999999", UNREADABLE), ("5e9223372036854775808", UNREADABLE),
        ("5e-9223372036854775809", UNREADABLE),
        ("-1Gi", NEGATIVE),
        ("9Pi", TOO_LARGE), ("1Ei", TOO_LARGE), ("1E", TOO_LARGE),
        ("9007199254740993", TOO_LARGE),
    )

    def test_an_unreadable_limit_fails_naming_it(self):
        for limit, message in self.REFUSED:
            with self.subTest(limit=limit):
                stderr = self.assert_refused(message, values=memory_values(limit))
                self.assertIn(f"{LIMIT} is ", stderr)

    def test_an_unreadable_request_fails_naming_it(self):
        for request, message in self.REFUSED:
            with self.subTest(request=request):
                stderr = self.assert_refused(message, values=memory_values(request=request))
                self.assertIn(f"{REQUEST} is ", stderr)

    # Exponents below the chart's lowest, -1000, each with the exponent as
    # Kubernetes narrows it to 32 bits: 5e2147483648 and 5e-2147483648 both
    # narrow to -2^31, and 1e4294966295 to -1001.
    BELOW_BOUND = (("1e-1001", -1001), ("1e4294966295", -1001), ("1E-1001", -1001),
                   ("5e2147483648", -2147483648), ("5e-2147483648", -2147483648))

    def test_a_decimal_exponent_below_minus_1000_fails_naming_the_bound(self):
        for key, make in ((LIMIT, lambda v: memory_values(v)),
                          (REQUEST, lambda v: memory_values(request=v))):
            for value, narrowed in self.BELOW_BOUND:
                with self.subTest(key=key, value=value):
                    stderr = self.assert_refused(
                        f"({narrowed}, as Kubernetes reads it) is below -1000, "
                        "the lowest this chart accepts", values=make(value))
                    self.assertIn(f'{key} is "{value}"', stderr)
                    self.assertNotIn(self.UNREADABLE, stderr)

    def test_a_zero_limit_fails_naming_it(self):
        for limit in ("0", "0Gi", "-0", "0.0", "0e9", "Ki"):
            with self.subTest(limit=limit):
                stderr = self.assert_refused(self.ZERO, values=memory_values(limit))
                self.assertIn(f"{LIMIT} is ", stderr)

    def test_an_unset_limit_keeps_its_message(self):
        # The quantity reader runs after the "must be set" check: an unset
        # limit (null, empty, or a --set 0, an integer zero that check reads
        # as empty) gets that message and none of the reader's.
        for args, values in ((("--set", f"{LIMIT}=null"), None),
                             (("--set-string", f"{LIMIT}="), None),
                             (("--set", f"{LIMIT}=0"), None),
                             ((), memory_values(None))):
            with self.subTest(args=args, values=values):
                stderr = self.assert_refused(self.MUST_BE_SET, *args, values=values)
                for other in (self.UNREADABLE, self.ZERO, self.NEGATIVE, self.TOO_LARGE):
                    self.assertNotIn(other, stderr)

    def test_requests_that_are_not_a_map_fail(self):
        # Autoscaling off: the autoscaler's own reads of resources.requests
        # would fail first.
        self.assert_refused("resources.requests must be a map",
                            "--set-string", "resources.requests=abc",
                            "--set", "autoscaling.enabled=false")

    def test_cpu_and_other_resources_pass_through(self):
        values = memory_values(extra_requests={"cpu": "500m", "ephemeral-storage": "1Gi"},
                               extra_limits={"ephemeral-storage": "2Gi"})
        resources = self.rendered_resources(values=values)
        self.assertEqual(resources, {
            "requests": {"cpu": "500m", "memory": "3584Mi", "ephemeral-storage": "1Gi"},
            "limits": {"cpu": 2, "memory": "5Gi", "ephemeral-storage": "2Gi"}})
        # The scaler reads the CPU request as the deployer set it.
        for request_values, cpu in ((None, "1"), (values, "500m")):
            with self.subTest(cpu=cpu):
                result = render_server(values=request_values,
                                       template="templates/scaler-deployment.yaml")
                self.assertEqual(result.returncode, 0, result.stderr)
                container = yaml.safe_load(result.stdout)["spec"]["template"]["spec"]["containers"][0]
                env = {item["name"]: item.get("value") for item in container["env"]}
                self.assertEqual(env["LIVE_EDIT_SCALER_CPU_REQUEST"], cpu)


class CheckpointTickTests(unittest.TestCase):
    """checkpoint.tick at most 19: the binary refuses a tick of 20 s or more
    at startup (two thirds of the 30 s checkpoint leader-lock TTL), and the
    chart refuses to render one, naming the value, the maximum and the
    setting to change."""

    def configmap_tick(self, tick):
        result = render_server("--set", f"checkpoint.tick={tick}",
                               template="templates/configmap.yaml")
        self.assertEqual(result.returncode, 0, result.stderr)
        return yaml.safe_load(result.stdout)["data"]["LIVE_EDIT_CHECKPOINT_TICK_SECS"]

    def test_a_tick_of_19_renders(self):
        self.assertEqual(self.configmap_tick(19), "19")

    def test_the_default_tick_renders(self):
        result = render_server(template="templates/configmap.yaml")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            yaml.safe_load(result.stdout)["data"]["LIVE_EDIT_CHECKPOINT_TICK_SECS"], "10")

    def test_a_tick_of_20_or_more_fails_naming_the_value(self):
        for tick in (20, 59):
            with self.subTest(tick=tick):
                result = render_server("--set", f"checkpoint.tick={tick}")
                self.assertNotEqual(result.returncode, 0, "the render succeeded")
                self.assertIn(f"checkpoint.tick is {tick}", result.stderr)
                self.assertIn("must be 19 or less", result.stderr)
                self.assertIn("Set checkpoint.tick to at most 19", result.stderr)


CONFIGMAP = "templates/configmap.yaml"
DEPLOYMENT = "templates/deployment.yaml"
SMOKE_TEST = "templates/tests/smoke-test.yaml"
# What a row renders when its variable or field is left out.
ABSENT = "<absent>"
# The row fails the render with the nonNegativeInteger helper's message.
REFUSED = "<refused>"
# The row fails the render with its `required` message ("<key> must be set").
MUST_BE_SET = "<must be set>"

TRACING_ON = {"observability": {"tracing": {
    "enabled": True, "endpoint": {"grpc": "http://otel.example.com:4317"}}}}
SMOKE_TEST_ON = {"tests": {"enabled": True,
                           "storageApiResourceBase": "https://storage.example.com/container/"}}
# A grace that covers a stream-ending window of a million seconds, so the
# termination-grace check passes whatever window a case sets.
LONG_GRACE = {"terminationGracePeriodSeconds": 2000000}


def env_row(var):
    return lambda doc: doc["data"].get(var, ABSENT)


def spec_field(name):
    return lambda doc: doc["spec"].get(name, ABSENT)


class IntegerRow(NamedTuple):
    key: str
    template: str
    read: object
    # The rendered form of a digit string: a ConfigMap value is a string, a
    # Kubernetes field an integer.
    form: object
    default: object
    on_null: object
    on_empty: object
    extra: dict = {}


def configmap_row(key, var, default, on_null=REFUSED, on_empty=REFUSED, extra=None):
    return IntegerRow(key, CONFIGMAP, env_row(var), str, default, on_null, on_empty,
                      extra or {})


# Each row with its default and what null and "" render. Shapes: refused
# both ways (the ceilings, the tail caps, the tracing batch settings), left
# out both ways (memoryBytes, extReleaseMaxBytes, the subscriber waiter cap,
# the smoke-test retention), left out or derived on null but refused on ""
# (uploadAdmissionWaitMs, minReadySeconds), and `required` (the two paced
# stream-ending settings).
INTEGER_ROWS = (
    configmap_row("valkey.snapshotAtomicReadMaxItems", "LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS",
                  "37554"),
    configmap_row("valkey.snapshotAtomicReadMaxBytes", "LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES",
                  "604619"),
    configmap_row("layer.maxSize", "LIVE_EDIT_MAX_LAYER_SIZE", "104857600"),
    configmap_row("layer.uploadAdmissionMaxWaiters", "LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS",
                  "64"),
    configmap_row("checkpoint.tail.maxChain", "LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN", "8"),
    configmap_row("checkpoint.tail.maxEntries", "LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", "4096"),
    configmap_row("checkpoint.tail.maxBytes", "LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES", "67108864"),
    configmap_row("valkey.memoryBytes", "LIVE_EDIT_VALKEY_MEMORY_BYTES", ABSENT,
                  on_null=ABSENT, on_empty=ABSENT),
    configmap_row("valkey.extReleaseMaxBytes", "LIVE_EDIT_EXT_RELEASE_MAX_BYTES", ABSENT,
                  on_null=ABSENT, on_empty=ABSENT),
    configmap_row("layer.uploadAdmissionWaitMs", "LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS", ABSENT,
                  on_null=ABSENT),
    configmap_row("valkey.subscriberAdmissionMaxWaiters",
                  "LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS", ABSENT,
                  on_null=ABSENT, on_empty=ABSENT),
    configmap_row("shutdown.streamEndRate", "LIVE_EDIT_SHUTDOWN_STREAM_END_RATE", "25",
                  on_null=MUST_BE_SET, on_empty=MUST_BE_SET),
    configmap_row("shutdown.streamEndMaxWindowSecs",
                  "LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS", "20",
                  on_null=MUST_BE_SET, on_empty=MUST_BE_SET, extra=LONG_GRACE),
    configmap_row("observability.tracing.batch.maxQueueSize", "OTEL_BSP_MAX_QUEUE_SIZE", "2048",
                  extra=TRACING_ON),
    configmap_row("observability.tracing.batch.maxExportBatchSize",
                  "OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "512", extra=TRACING_ON),
    configmap_row("observability.tracing.batch.scheduleDelayMs", "OTEL_BSP_SCHEDULE_DELAY",
                  "5000", extra=TRACING_ON),
    # Null derives timeoutSecs + streamEndMaxWindowSecs + 5 (test-shutdown-chart.py).
    IntegerRow("minReadySeconds", DEPLOYMENT, spec_field("minReadySeconds"), int, 30,
               on_null=30, on_empty=REFUSED),
    IntegerRow("tests.ttlSecondsAfterFinished", SMOKE_TEST,
               spec_field("ttlSecondsAfterFinished"), int, 900,
               on_null=ABSENT, on_empty=ABSENT, extra=SMOKE_TEST_ON),
)
ROWS = {row.key: row for row in INTEGER_ROWS}


def merged(base, overlay):
    """`base` with `overlay` merged in key by key (a deep copy)."""
    result = json.loads(json.dumps(base))
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merged(result[key], value)
        else:
            result[key] = value
    return result


def at_path(dotted, value):
    """A values tree holding `value` at the dotted path."""
    for part in reversed(dotted.split(".")):
        value = {part: value}
    return value


class IntegerRowTests(unittest.TestCase):
    """Every integer row goes through the nonNegativeInteger helper, keeping
    the meaning of null, "" and 0 that its @param documents. Before, most of
    them went through a bare `| int64`, which turned text or null into 0,
    "0x10" into 16 and a fraction into its whole part, so a typo could switch
    a limit off unseen."""

    UNSET = object()

    def render_row(self, row, value=UNSET, *args):
        """The render of `row`'s template with `value` at its key in a values
        file (written as YAML, so a number stays a number) and `args`."""
        values = row.extra
        if value is not self.UNSET:
            values = merged(values, at_path(row.key, value))
        return render_server(*args, values=yaml.safe_dump(values), template=row.template)

    def rendered(self, row, value=UNSET, *args):
        """What `row` renders: its value, or ABSENT."""
        result = self.render_row(row, value, *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return row.read(yaml.safe_load(result.stdout))

    def assert_refused(self, row, value=UNSET, *args):
        result = self.render_row(row, value, *args)
        self.assertNotEqual(result.returncode, 0, "the render succeeded")
        self.assertIn(f"{row.key} must be a non-negative whole number (", result.stderr)

    def assert_outcome(self, row, expected, value):
        if expected == REFUSED:
            self.assert_refused(row, value)
        elif expected == MUST_BE_SET:
            result = self.render_row(row, value)
            self.assertNotEqual(result.returncode, 0, "the render succeeded")
            self.assertIn(f"{row.key} must be set:", result.stderr)
        else:
            self.assertEqual(self.rendered(row, value), expected)

    def test_each_row_renders_its_default_digits(self):
        self.assertEqual(len(INTEGER_ROWS), 18)
        for row in INTEGER_ROWS:
            with self.subTest(row.key):
                self.assertEqual(self.rendered(row), row.default)

    def test_a_value_that_is_not_a_whole_number_fails_naming_the_key(self):
        # On most rows the old chart rendered these 0, 1, -1, 5 and 16.
        for row in INTEGER_ROWS:
            for value in ("fast", 1.5, -1):
                with self.subTest(row.key, values_file=value):
                    self.assert_refused(row, value)
            for value in ("5.0", "0x10"):
                with self.subTest(row.key, set_string=value):
                    self.assert_refused(row, self.UNSET, "--set-string", f"{row.key}={value}")

    def test_surrounding_space_and_a_leading_plus_render_the_digits(self):
        # The binary's integer parse accepts both; the old chart read " 64"
        # as 0.
        for row in INTEGER_ROWS:
            for value in (" 64", "+64"):
                with self.subTest(row.key, value=value):
                    self.assertEqual(self.rendered(row, value), row.form("64"))

    def test_a_million_from_a_values_file_renders_as_digits(self):
        # Helm decodes a values-file number as float64; rendered as is, a
        # million reads 1e+06, which the binary's integer parse refuses.
        for row in INTEGER_ROWS:
            with self.subTest(row.key):
                self.assertEqual(self.rendered(row, 1000000), row.form("1000000"))

    def test_null_and_empty_keep_each_rows_meaning(self):
        for row in INTEGER_ROWS:
            for value, expected in ((None, row.on_null), ("", row.on_empty)):
                with self.subTest(row.key, value=value):
                    self.assert_outcome(row, expected, value)

    def test_zero_renders_where_it_is_a_setting(self):
        # 0 turns the release budget off, never waits for an upload slot, or
        # ends every stream at once; minReadySeconds 0 makes a pod available
        # once Ready.
        for key in ("valkey.extReleaseMaxBytes", "layer.uploadAdmissionWaitMs",
                    "shutdown.streamEndRate", "shutdown.streamEndMaxWindowSecs",
                    "minReadySeconds"):
            row = ROWS[key]
            with self.subTest(key, values_file=0):
                self.assertEqual(self.rendered(row, 0), row.form("0"))
            with self.subTest(key, set_string="0"):
                self.assertEqual(self.rendered(row, self.UNSET, "--set-string", f"{key}=0"),
                                 row.form("0"))

    def test_a_zero_subscriber_waiter_cap_however_written_is_left_out(self):
        # 0 means "follow the subscription ceiling", and the binary refuses an
        # explicit 0, so the variable is left out. The old chart rendered a
        # quoted "0" (truthy) as "0".
        row = ROWS["valkey.subscriberAdmissionMaxWaiters"]
        for value in (0, "0", "00", "+0", " 0"):
            with self.subTest(value=value):
                self.assertEqual(self.rendered(row, value), ABSENT)
        self.assertEqual(self.rendered(row, 64), "64")

    def test_a_zero_smoke_test_retention_renders_no_field(self):
        # To Kubernetes a ttlSecondsAfterFinished of 0 deletes the Job at once;
        # the chart's 0 keeps it.
        row = ROWS["tests.ttlSecondsAfterFinished"]
        for value in (0, "0"):
            with self.subTest(value=value):
                self.assertEqual(self.rendered(row, value), ABSENT)

    def test_a_zero_tracing_batch_setting_fails_naming_its_minimum(self):
        for key in ("maxQueueSize", "maxExportBatchSize", "scheduleDelayMs"):
            row = ROWS[f"observability.tracing.batch.{key}"]
            for value in (0, "00", " 0"):
                with self.subTest(key, value=value):
                    result = self.render_row(row, value)
                    self.assertNotEqual(result.returncode, 0, "the render succeeded")
                    self.assertIn(f"{row.key} must be at least 1", result.stderr)
            with self.subTest(key, value=" 100"):
                self.assertEqual(self.rendered(row, " 100"), "100")

    def test_a_quote_or_backslash_in_a_text_value_fails_naming_the_configmap(self):
        # A `"` or an unknown `\` escape in a double-quoted ConfigMap row stops
        # the ConfigMap parsing as YAML; deployment.yaml reads the rendered
        # ConfigMap and names the template rather than failing on its absence.
        # A values file, as `--set` drops a backslash.
        for level in ('in"fo', "info\\q"):
            with self.subTest(level=level):
                result = render_server(values=yaml.safe_dump(
                    {"observability": {"logging": {"level": level}}}))
                self.assertNotEqual(result.returncode, 0, "the render succeeded")
                self.assertIn("templates/configmap.yaml does not render as valid YAML",
                              result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
