<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OMPF wire schema

`OmniObjectDelta.fbs` is the OMPF FlatBuffers schema the live edit service also speaks. `OmniObjectDelta_generated.h` is its generated C++ code, `ObjectDeltaBuilder.h` builds deltas and `ObjectMultiPartFormat.h` declares the multipart layout. No Nucleus implementation is included.

The generated verifier checks structural FlatBuffer bounds only and accepts unknown union tags, so it is not by itself a safe decoder: network admission also validates OMPF framing, semantic fields and commands, external-value hashes and bounded value decoding before any scene mutation.

The schema files are NVIDIA's and carry Apache-2.0 SPDX notices. The FlatBuffers headers retain Google's original Apache-2.0 notices.
