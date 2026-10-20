# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.live.test import native_identity


class NativeIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.patch = patch("tools.live.test.ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.artifact = self.root / "probe.so"
        self.artifact.write_bytes(b"synthetic artifact for identity-verifier unit tests")
        paths = ["CMakeLists.txt", "include/omni/usd/live/Live.h", "source/live/runtime/LiveUri.h",
                 "source/live/runtime/LiveUri.cpp", "source/live/runtime/LiveUriApi.cpp", "source/live/runtime/Live.exports"]
        self.manifest = {"schema_version": 1, "boundary": "uri_component", "sanitizers": False,
                         "artifact_sha256": hashlib.sha256(self.artifact.read_bytes()).hexdigest(), "inputs": []}
        for name in paths:
            source = self.root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("original " + name)
            self.manifest["inputs"].append({"path": name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        self.write_manifest()

    def write_manifest(self):
        Path(str(self.artifact) + ".manifest.json").write_text(json.dumps(self.manifest))

    def test_matching_identity_accepted_and_changed_binary_rejected(self):
        self.assertEqual(native_identity(self.artifact), self.manifest)
        self.artifact.write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            native_identity(self.artifact)

    def test_changed_source_or_missing_required_input_rejected(self):
        source = self.root / "source/live/runtime/LiveUri.cpp"
        original = source.read_bytes()
        source.write_bytes(original + b"\nmodified")
        with self.assertRaisesRegex(ValueError, "stale source"):
            native_identity(self.artifact)
        source.write_bytes(original)
        self.manifest["inputs"].pop()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "missing required"):
            native_identity(self.artifact)

    def test_sanitized_artifact_cannot_be_loaded_by_python_suite(self):
        self.manifest["sanitizers"] = True
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "native sanitizer driver"):
            native_identity(self.artifact)

    def test_runtime_requires_and_verifies_linked_dependency_identity(self):
        self.manifest["boundary"] = "runtime_foundation"
        for name in ("source/live/runtime/LiveRuntime.h", "source/live/runtime/LiveRuntime.cpp", "source/live/support/LiveTsan.h",
                     "source/live/runtime/LiveRequest.h", "source/live/runtime/LiveRequest.cpp",
                     "source/live/runtime/LiveUnary.h", "source/live/runtime/LiveUnary.cpp",
                     "source/live/runtime/LiveDispatch.h", "source/live/runtime/LiveDispatch.cpp",
                     "source/live/runtime/LiveMetricDuration.h", "source/live/runtime/LiveDeadline.h", "source/live/runtime/LiveExceptionResult.h",
                     "source/live/runtime/LiveAuth.h", "source/live/runtime/LiveAuth.cpp",
                     "source/live/runtime/LiveFinite.cpp", "source/live/runtime/LiveMetadata.h",
                     "source/live/runtime/LiveLayerWrite.h", "source/live/runtime/LiveLayerWrite.cpp",
                     "source/live/runtime/LiveLayerRead.h", "source/live/runtime/LiveLayerRead.cpp",
                     "source/live/runtime/LiveLayerProgress.h", "source/live/runtime/LiveLayerProgress.cpp",
                     "source/live/runtime/LiveLayerActor.h", "source/live/runtime/LiveLayerActor.cpp",
                     "source/live/runtime/LiveJitter.h", "source/live/runtime/LiveJitter.cpp",
                     "source/live/runtime/LiveDrain.h", "source/live/runtime/LiveDrain.cpp",
                     "source/live/runtime/LiveStatus.h", "source/live/runtime/LiveStatus.cpp",
                     "source/live/runtime/LiveDiagnostics.h", "source/live/runtime/LiveDiagnostics.cpp",
                     "source/live/runtime/LiveDiagnosticsApi.cpp", "cmake/LiveBuildProvenance.h.in",
                     "source/live/runtime/LiveSession.h", "source/live/runtime/LiveSession.cpp",
                     "source/live/runtime/LiveSessionPeers.h", "source/live/runtime/LiveSessionPeers.cpp",
                     "source/live/runtime/LiveSessionApi.cpp",
                     "source/live/runtime/LiveLayerAdmission.h", "source/live/runtime/LiveLayerAdmission.cpp",
                     "source/live/runtime/LiveLayerStatus.h", "source/live/runtime/LiveLayerStatus.cpp",
                     "source/live/runtime/LiveRuntimeApi.cpp", "cmake/LiveGrpcDependencies.cmake",
                     "source/live/protos/nvidia/omniverse/live/v1alpha/live.proto",
                     "tests/native/runtime/runtime_foundation.cpp", "tests/native/runtime/request_lifecycle.cpp",
                     "tests/contracts/diagnostic-catalog.json", "tools/live/diagnostic_catalog.py",
                     "tests/native/runtime/auth_controller.cpp", "tests/native/runtime/layer_write_lifecycle.cpp", "tests/native/runtime/layer_read_lifecycle.cpp", "tests/native/runtime/layer_progress.cpp", "tests/native/runtime/jitter.cpp", "tests/native/runtime/drain.cpp", "tests/native/runtime/status.cpp", "tests/native/runtime/diagnostics.cpp", "tests/native/runtime/diagnostics_api.cpp", "tests/native/runtime/session_peers.cpp", "tests/native/runtime/session_api.cpp", "tests/native/runtime/layer_admission.cpp", "tests/native/runtime/layer_status.cpp",
                     "cmake/LiveTestSupport.cmake", "tests/native/support/test_tls.h"):
            source = self.root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("original " + name)
            self.manifest["inputs"].append({"path": name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "missing native dependency"):
            native_identity(self.artifact, "runtime_foundation")
        dependency = self.root / "dependency.a"
        dependency.write_bytes(b"synthetic archive identity")
        self.manifest["dependencies"] = [{"path": str(dependency),
                                          "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest()}]
        self.write_manifest()
        self.assertEqual(native_identity(self.artifact, "runtime_foundation"), self.manifest)
        for name in ("tests/contracts/diagnostic-catalog.json", "tools/live/diagnostic_catalog.py",
                     "tests/native/support/test_tls.h"):
            with self.subTest(catalog_input=name):
                original_inputs = self.manifest["inputs"]
                self.manifest["inputs"] = [entry for entry in original_inputs if entry["path"] != name]
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "missing required source identity"):
                    native_identity(self.artifact, "runtime_foundation")
                self.manifest["inputs"] = original_inputs
                self.write_manifest()
                source = self.root / name
                original = source.read_bytes()
                source.write_bytes(b"changed catalog input")
                with self.assertRaisesRegex(ValueError, "stale source inputs"):
                    native_identity(self.artifact, "runtime_foundation")
                source.write_bytes(original)
        self.manifest["sanitizers"] = True
        self.write_manifest()
        self.assertEqual(native_identity(self.artifact, "runtime_foundation"), self.manifest)
        dependency.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "stale dependency"):
            native_identity(self.artifact, "runtime_foundation")

    def test_scene_identity_detects_sdk_header_change_and_removal(self):
        from tools.live.scene_identity import inputs, tree_hash
        scene_source = self.root / "source/live/scene/SdfSceneData.cpp"
        scene_source.parent.mkdir(parents=True)
        scene_source.write_text("synthetic adapter source")
        extra_test = self.root / "tests/native/usd/synthetic_extra.cpp"
        extra_test.parent.mkdir(parents=True)
        extra_test.write_text("synthetic extra test")
        self.assertIn("tests/native/usd/synthetic_extra.cpp", inputs(self.root))
        self.manifest["boundary"] = "sdf_scene"
        self.manifest["inputs"] = []
        for name in inputs(self.root):
            source = self.root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("scene input " + name)
            self.manifest["inputs"].append({"path": name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        dependency = self.root / "python.so"
        dependency.write_bytes(b"synthetic python library")
        self.manifest["dependencies"] = [{"path": str(dependency), "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest()}]
        self.manifest["dependency_trees"] = []
        for i in range(3):
            tree = self.root / f"sdk-{i}"
            tree.mkdir()
            (tree / "header.h").write_bytes(b"synthetic SDK header")
            self.manifest["dependency_trees"].append({"path": str(tree), "sha256": tree_hash(tree)})
        self.write_manifest()
        self.assertEqual(native_identity(self.artifact, "sdf_scene"), self.manifest)
        helper = self.root / "tools/live/sdk_tree_identity.py"
        original_helper = helper.read_bytes()
        helper.write_bytes(original_helper + b" changed hash helper")
        with self.assertRaisesRegex(ValueError, "stale source"):
            native_identity(self.artifact, "sdf_scene")
        helper.write_bytes(original_helper)
        original_test = extra_test.read_bytes()
        extra_test.write_bytes(original_test + b"changed")
        with self.assertRaisesRegex(ValueError, "stale source"):
            native_identity(self.artifact, "sdf_scene")
        extra_test.write_bytes(original_test)
        with self.assertRaisesRegex(ValueError, "unexpected native artifact"):
            native_identity(self.artifact, "scene_benchmark")
        original = scene_source.read_bytes()
        scene_source.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "stale source"):
            native_identity(self.artifact, "sdf_scene")
        scene_source.write_bytes(original)
        self.manifest["boundary"] = "sdf_scene"
        self.write_manifest()
        header = self.root / "sdk-0/header.h"
        header.write_bytes(b"changed SDK header")
        with self.assertRaisesRegex(ValueError, "scene SDK differs"):
            native_identity(self.artifact, "sdf_scene")
        header.unlink()
        with self.assertRaisesRegex(ValueError, "scene SDK differs"):
            native_identity(self.artifact, "sdf_scene")
        header.write_bytes(b"synthetic SDK header")
        self.manifest["boundary"] = "scene_benchmark"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "compile-command provenance"):
            native_identity(self.artifact, "scene_benchmark")
        commands = self.root / "compile_commands.json"
        commands.write_text("[]")
        self.manifest["dependencies"].append({"path": str(commands), "sha256": hashlib.sha256(commands.read_bytes()).hexdigest()})
        self.manifest["compile_commands"] = [{"source": "synthetic"}]
        alias = self.root / "selected-sdk"
        alias.symlink_to(self.root / "sdk-0", target_is_directory=True)
        self.manifest["dependency_trees"][0]["path"] = str(alias)
        self.write_manifest()
        self.assertEqual(native_identity(self.artifact, "scene_benchmark"), self.manifest)
        replacement = self.root / "replacement-sdk"
        replacement.mkdir(); (replacement / "header.h").write_bytes(b"incompatible replacement")
        alias.unlink(); alias.symlink_to(replacement, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "scene SDK differs"):
            native_identity(self.artifact, "scene_benchmark")
        from tools.live.scene_identity import sdk_paths
        selected, python_library = sdk_paths(alias, alias)
        self.assertEqual(selected[0], alias / "include")
        self.assertEqual(python_library, alias / ("libs/python312.lib" if os.name == "nt" else
                                                "lib/libpython3.12.dylib" if sys.platform == "darwin" else
                                                "lib/libpython3.12.so"))


if __name__ == "__main__":
    unittest.main()
