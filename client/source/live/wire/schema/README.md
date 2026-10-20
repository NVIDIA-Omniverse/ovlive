<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OMPF wire schema provenance

These four source files are copied from the source-only package `omniverse.nucleus-objects.schema.linux-x86_64`, version `111.0-dev+gitlab.23672466.8b7c8579`. `provenance.json` records both the current file hashes and original upstream hashes and the matching FlatBuffers header pin, and nothing in the repository fetches the package. Only the schema, generated header, builder, multipart-layout declaration and license are retained; package metadata is not redistributed. No Nucleus implementation is included. The matching FlatBuffers runtime headers are vendored unchanged in [../third_party/flatbuffers](../third_party/flatbuffers/README.md).

The generated verifier checks structural FlatBuffer bounds only and accepts unknown union tags, so it is not by itself a safe decoder: network admission also validates OMPF framing, semantic fields and commands, external-value hashes and bounded value decoding before any scene mutation.

The NVIDIA-owned copies carry Apache-2.0 SPDX notices. The local modifications affect license notices only; the source and schema bodies are unchanged. NVIDIA owns the upstream sources and authorizes their Apache-2.0 licensing. The FlatBuffers headers retain Google's original Apache-2.0 notices.
