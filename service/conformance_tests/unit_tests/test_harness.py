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
#
import copy
import hashlib
import json
from pathlib import Path
import threading

import pytest

from live_edit_conformance.client import Stream
from live_edit_conformance.config import Settings, target
from live_edit_conformance.protocol import (
    ERROR_INFO_TYPE,
    apply_snapshot,
    content_hash,
    error_info,
    fixture,
    status_details,
    unpack_ompf,
)
from live_edit_conformance.runner import Coverage, main
from nvidia.omniverse.live.v1alpha import live_service_pb2 as pb


@pytest.fixture
def configured(monkeypatch, tmp_path):
    for name in list(__import__("os").environ):
        if name.startswith("CONFORMANCE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("CONFORMANCE_AUTH_TOKEN", "local-token")
    monkeypatch.setenv("CONFORMANCE_BASE_PATH", "https://example.test/base.usda")
    monkeypatch.setenv("CONFORMANCE_REPORT_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    "url",
    [
        "localhost:50001",
        "ftp://host",
        "https://user:pass@host",
        "http://host/path",
        "https://host?token=x",
        "http://host:99999",
    ],
)
def test_bad_endpoints_fail(url):
    with pytest.raises(ValueError):
        target(url)


def test_ipv6_and_tls_targets():
    assert target("https://[::1]:443") == ("[::1]:443", True)
    assert target("http://service:50001") == ("service:50001", False)


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_timeouts_fail(configured, monkeypatch, timeout):
    monkeypatch.setenv("CONFORMANCE_TIMEOUT", timeout)
    with pytest.raises(ValueError):
        Settings.from_env()


def test_rotating_token_file_is_read_each_call(configured, monkeypatch):
    token = configured / "token"
    token.write_text("first")
    monkeypatch.delenv("CONFORMANCE_AUTH_TOKEN")
    monkeypatch.setenv("CONFORMANCE_AUTH_TOKEN_FILE", str(token))
    settings = Settings.from_env()
    assert settings.metadata() == [("authorization", "Bearer first")]
    token.write_text("second")
    assert settings.metadata() == [("authorization", "Bearer second")]
    token.write_text("bad\nheader")
    with pytest.raises(ValueError):
        settings.metadata()


def test_conflicting_credentials_fail(configured, monkeypatch):
    monkeypatch.setenv("CONFORMANCE_AUTH_TOKEN_FILE", "/unused")
    with pytest.raises(ValueError, match="exactly one"):
        Settings.from_env()


def test_preflight_failure_is_junit_error(configured, monkeypatch):
    monkeypatch.setattr("sys.argv", ["run-conformance-tests"])
    monkeypatch.setenv("CONFORMANCE_DECODER", "/nonexistent/decoder")
    assert main() == 2
    assert 'errors="1"' in (configured / "conformance-results.xml").read_text()


class Call:
    def __init__(self, messages):
        self.messages = messages
        self.cancelled = threading.Event()

    def __iter__(self):
        yield from self.messages
        self.cancelled.wait()

    def cancel(self):
        self.cancelled.set()


@pytest.mark.parametrize(
    "messages",
    [
        [pb.SubscribeReadLayerResponse(chunk=pb.SubscribeReadLayerChunk(data=b"bad"))],
        [
            pb.SubscribeReadLayerResponse(header=pb.SubscribeReadLayerHeader(total_size=2)),
            pb.SubscribeReadLayerResponse(chunk=pb.SubscribeReadLayerChunk(data=b"long")),
        ],
        [
            pb.SubscribeReadLayerResponse(header=pb.SubscribeReadLayerHeader(total_size=1)),
            pb.SubscribeReadLayerResponse(header=pb.SubscribeReadLayerHeader()),
        ],
        [],
    ],
)
def test_broken_streams_cannot_pass(messages):
    stream = Stream(Call(messages), timeout=0.05)
    try:
        with pytest.raises(AssertionError):
            stream.delivery()
    finally:
        stream.close()


def test_commit_header_is_not_a_caught_up_ack():
    header = pb.SubscribeReadLayerHeader(
        object_id=1, sequence_num=2, server_timestamp_microseconds_since_epoch=42
    )
    stream = Stream(Call([pb.SubscribeReadLayerResponse(header=header)]), timeout=0.1)
    try:
        with pytest.raises(AssertionError):
            stream.caught_up(1, 2)
    finally:
        stream.close()


def test_fixture_provenance_hashes_match_committed_inputs():
    root = Path(__file__).resolve().parents[2]
    package = root / "conformance_tests/src/live_edit_conformance"
    manifest = json.loads((package / "build.json").read_text())["sha256"]
    for key, path in {
        "proto": root / "proto/nvidia/omniverse/live/v1alpha/live_service.proto",
        "oodf_schema": root / "oodf-ffi/schema/OmniObjectDelta.fbs",
        "synthetic": package / "fixtures/synthetic.json",
        "captured": package / "fixtures/captured.json",
        "captured_golden": package / "fixtures/captured-golden.json",
    }.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key]
    assert (package / "fixtures/captured.json").read_bytes() == (
        root / "tools/fixtures/gze_inbound_deltas.json"
    ).read_bytes()
    for name in ("initial", "external", "large", "update", "update_time", "time_sample"):
        assert unpack_ompf(fixture(name))[0]


def test_content_hash_matches_client_vectors_and_keys_the_external_fixture():
    # Vectors over bytes(i % 251) from the clients' own suites (the C++ and the
    # Rust client libraries) and hashlib: the 1 MiB block hash, not plain
    # SHA-256 of the bytes.
    for length, expected in {
        0: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        1: "1406e05881e299367766d313e26c05564ec91bf721d31726bd6e46e60689539a",
        1048577: "a9c574ce937d2371daf87cdd0e75396b096d7f74a66c764e4c656bd12b2e7bf9",
    }.items():
        assert content_hash(bytes(i % 251 for i in range(length))) == expected
    external = unpack_ompf(fixture("external"))[1:]
    assert [content_hash(part) for part in external] == [
        "368a4df03f4e059ae6f4392ba381e03c261d21f064fa1caa48e72e6b14c9e84d"
    ]


def test_error_info_reads_the_refusal_detail_of_grpc_status_details_bin():
    def varint(value):
        out = bytearray()
        while True:
            byte, value = value & 0x7F, value >> 7
            out.append(byte | (0x80 if value else 0))
            if not value:
                return bytes(out)

    def field(number, payload):
        """One length-delimited protobuf field."""
        return varint(number << 3 | 2) + varint(len(payload)) + payload

    error_info_bytes = (
        field(1, b"PREDECESSOR_NOT_COMMITTED")
        + field(2, b"nvidia.omniverse.live.v1alpha")
        + field(3, field(1, b"last_record") + field(2, b"operation"))
        + field(3, field(1, b"last_operation_id") + field(2, b"op-1"))
    )
    detail = field(1, ERROR_INFO_TYPE.encode()) + field(2, error_info_bytes)
    # google.rpc.Status: code 10 (ABORTED, a varint), a message, one detail.
    status = bytes([0x08, 10]) + field(2, b"refused") + field(3, detail)

    class Refused:
        def __init__(self, metadata):
            self.metadata = metadata

        def trailing_metadata(self):
            return self.metadata

    refused = Refused((("grpc-status-details-bin", status),))
    assert error_info(refused) == (
        "PREDECESSOR_NOT_COMMITTED",
        "nvidia.omniverse.live.v1alpha",
        {"last_record": "operation", "last_operation_id": "op-1"},
    )
    assert list(status_details(refused)) == [ERROR_INFO_TYPE]
    assert error_info(Refused(())) is None
    assert error_info(Refused(None)) is None


def test_fold_negative_controls_detect_lost_content():
    full = {
        "structCommands": [
            {
                "type": "CommandDiffChangeSection",
                "sectionId": 5,
                "parentId": 1,
                "sectionName": "probe",
                "sectionType": 0,
                "sectionOrder": 0,
            }
        ],
        "setFields": [
            {"sectionId": 5, "keyName": "value", "valueOrExtHash": "42", "extValueSize": 0}
        ],
        "timeSamples": [{"sectionId": 5, "time": 1.5, "valueOrExtHash": "43", "extValueSize": 0}],
    }
    expected = apply_snapshot(full)
    for key in full:
        broken = copy.deepcopy(full)
        broken[key] = []
        assert apply_snapshot(broken) != expected
    for key, field in [
        ("structCommands", "parentId"),
        ("structCommands", "sectionOrder"),
        ("setFields", "valueOrExtHash"),
        ("timeSamples", "time"),
    ]:
        broken = copy.deepcopy(full)
        broken[key][0][field] = "corrupt"
        assert apply_snapshot(broken) != expected
    deleted = {
        "structCommands": [{"type": "CommandDiffDeleteSection", "sectionId": 5}],
        "setFields": [],
        "timeSamples": [],
    }
    assert apply_snapshot(deleted, expected) == ({}, {}, {})


def test_skips_and_empty_runs_are_visible_to_runner():
    from types import SimpleNamespace

    coverage = Coverage(set())
    coverage.pytest_runtest_logreport(SimpleNamespace(skipped=True, when="setup"))
    assert coverage.skipped == 1 and coverage.executed == 0


def test_json_token_file(configured, monkeypatch):
    token = configured / "token"
    token.write_text('{"access_token":"rotating"}')
    monkeypatch.delenv("CONFORMANCE_AUTH_TOKEN")
    monkeypatch.setenv("CONFORMANCE_AUTH_TOKEN_FILE", str(token))
    assert Settings.from_env().metadata() == [("authorization", "Bearer rotating")]


def test_unexpected_deselection_cannot_certify():
    from types import SimpleNamespace

    coverage = Coverage(set())
    coverage.pytest_deselected([SimpleNamespace(nodeid="test_required")])
    assert coverage.unexpected_deselections == 1


def test_runner_ignores_ambient_pytest_selection(configured, monkeypatch):
    import os
    from live_edit_conformance import runner

    monkeypatch.setattr("sys.argv", ["run-conformance-tests"])
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k test_ping")
    monkeypatch.setenv("PYTEST_PLUGINS", "unrelated_plugin")
    monkeypatch.setattr(runner, "decode", lambda *args: {})
    monkeypatch.setattr(runner, "wait_ready", lambda *args: None)

    def fake_pytest(args, plugins):
        assert "PYTEST_ADDOPTS" not in os.environ and "PYTEST_PLUGINS" not in os.environ
        assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert args[args.index("-c") + 1] == os.devnull
        plugins[0].executed = 1
        return 0

    monkeypatch.setattr(runner.pytest, "main", fake_pytest)
    assert runner.run() == 0


def test_deployment_failures_emit_failed_junit(configured, monkeypatch, capsys):
    from live_edit_conformance import deployment

    monkeypatch.setenv("CONFORMANCE_REPORT_STDOUT", "true")

    def failure():
        raise RuntimeError("secret-must-not-be-logged")

    monkeypatch.setattr(deployment, "run", failure)
    assert deployment.main() == 2
    output = capsys.readouterr()
    assert "secret-must-not-be-logged" not in output.out + output.err
    assert "CONFORMANCE_REPORT conformance-results.xml" in output.out
    assert 'errors="1"' in (configured / "conformance-results.xml").read_text()
    assert json.loads((configured / "summary.json").read_text())["exit_code"] == 2


@pytest.mark.parametrize("mutation", ["isDiff", "baseVersion", "duplicateOrder"])
def test_catchup_envelope_and_order_negative_controls(monkeypatch, mutation):
    from live_edit_conformance import protocol

    signature = copy.deepcopy(protocol.capture_golden())
    if mutation == "duplicateOrder":
        for field in signature["setFields"]:
            field["setOrder"] = 0
    elif mutation == "isDiff":
        signature["isDiff"] = False
    else:
        signature["baseVersion"] = 42
    monkeypatch.setattr(protocol, "decode", lambda *args: signature)
    with pytest.raises(AssertionError):
        protocol.snapshot(None, b"")


def test_relative_field_order_detects_reordering():
    from live_edit_conformance.protocol import capture_golden, field_order

    original = capture_golden()
    broken = copy.deepcopy(original)
    broken["setFields"][0]["setOrder"], broken["setFields"][1]["setOrder"] = (
        broken["setFields"][1]["setOrder"],
        broken["setFields"][0]["setOrder"],
    )
    assert field_order(original) != field_order(broken)


def test_unexpectedly_accepted_anonymous_session_is_owned(configured):
    from live_edit_conformance.client import Client
    from types import SimpleNamespace

    client = Client(Settings.from_env())
    try:
        client.stub = SimpleNamespace(
            CreateSession=lambda *a, **kw: pb.CreateSessionResponse(session_id="unexpected")
        )
        client.call("CreateSession", pb.CreateSessionRequest(), metadata=[])
        assert client.resources == ["unexpected"]
    finally:
        client.channel.close()


@pytest.mark.parametrize("failure", [None, "PUT", "DELETE"])
def test_storage_fixture_provisioning_owns_only_one_uuid_file(configured, monkeypatch, failure):
    import os
    from urllib.parse import unquote, urlsplit
    from live_edit_conformance import deployment

    monkeypatch.setenv("CONFORMANCE_STORAGE_ENDPOINT", "https://storage.test")
    original = os.environ["CONFORMANCE_BASE_PATH"]
    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            assert request.get_header("Authorization") == "Bearer local-token"
            assert timeout > 0
            if request.method == failure:
                raise OSError("lost response")
            return Response()

    monkeypatch.setattr(deployment.urllib.request, "build_opener", lambda *a: Opener())
    monkeypatch.setattr(deployment, "run_tests", lambda: 0)
    assert deployment.main() == (2 if failure else 0)
    assert [request.method for request in calls] == ["PUT", "DELETE"]
    put_path = unquote(urlsplit(calls[0].full_url).path)
    delete_path = unquote(urlsplit(calls[1].full_url).path)
    assert put_path == delete_path
    assert put_path.startswith("/v1alpha/fileobject/by-address/" + original + "/conformance-")
    assert put_path.endswith(".usda")
    assert calls[0].data == b"#usda 1.0\n"


def _delta_vectors(delta):
    """(identifier, {vtable slot: (length word position, length)}) of an OODF
    delta, read as the FlatBuffers layout defines it: the root offset at 0,
    the root table's signed offset back to its vtable, each vector field's
    unsigned offset to its length word."""
    import struct

    root = struct.unpack_from("<I", delta, 0)[0]
    vtable = root - struct.unpack_from("<i", delta, root)[0]
    vtable_size = struct.unpack_from("<H", delta, vtable)[0]
    vectors = {}
    for slot in range(8, vtable_size, 2):
        field = struct.unpack_from("<H", delta, vtable + slot)[0]
        if field:
            at = root + field
            start = at + struct.unpack_from("<I", delta, at)[0]
            vectors[slot] = (start, struct.unpack_from("<I", delta, start)[0])
    return delta[4:8], vectors


def test_oversized_delta_builders_write_their_counts_where_a_reader_finds_them():
    from live_edit_conformance.protocol import (
        delta_of_children_list,
        delta_of_entries,
        pack_ompf,
    )

    # structCommands_type (slot 8), structCommands (10), setFields (12).
    identifier, vectors = _delta_vectors(delta_of_entries(60_000, 59_000))
    assert identifier == b"OODF"
    assert {slot: n for slot, (_, n) in vectors.items()} == {8: 60_000, 10: 60_000, 12: 59_000}

    delta = delta_of_children_list(7)
    identifier, vectors = _delta_vectors(delta)
    assert identifier == b"OODF"
    assert {slot: n for slot, (_, n) in vectors.items()} == {8: 1, 10: 1}
    # The list's u64 elements follow an 8-aligned length word + 4: the last
    # 7 × 8 bytes are the ids 1..7, little-endian.
    assert delta[-56:] == b"".join(i.to_bytes(8, "little") for i in range(1, 8))
    assert (len(delta) - 56) % 8 == 0

    assert unpack_ompf(pack_ompf(delta)) == [delta]


def test_storage_fixture_never_follows_redirects():
    from live_edit_conformance.deployment import NoRedirect

    with pytest.raises(RuntimeError, match="must not redirect"):
        NoRedirect().redirect_request(None, None, 302, None, None, "https://untrusted.test")
