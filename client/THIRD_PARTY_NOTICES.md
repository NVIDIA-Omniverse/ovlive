<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Client notices and source attribution

This document covers the ovlive client. NVIDIA-owned client code and documentation use Apache-2.0, including the Live Services implementation and the SpecTree-derived support code in `source/live/support`. The license text is in the repository root’s `LICENSE`, installed beside this file in a native package. [NOTICE](NOTICE) records the client license scope. Paths to source and provenance records below are relative to the repository’s `client/` directory; they identify source records, not native package members. `PROVENANCE.json` records the import snapshot and notice updates.

The LiveService protocol (`source/live/protos/PROVENANCE.md`) and OMPF schema (`source/live/wire/schema/README.md`) carry Apache-2.0 notices. The LiveService proto is copied unchanged from the live edit service and carries that file's own notice. The OMPF schema's provenance record identifies both original upstream and current repository hashes; only its license notices are adapted. NVIDIA owns both upstream sources and authorizes their Apache-2.0 licensing here.

The FlatBuffers runtime headers (`source/live/wire/third_party/flatbuffers/README.md`) in `source/live/wire/third_party/flatbuffers` are unmodified Google FlatBuffers 1.12.0 files under Apache-2.0, with the package license beside them; they receive no NVIDIA grant.

OpenUSD, TBB, FlatBuffers, gRPC, Protobuf, Abseil, RE2, c-ares, OpenSSL, zlib, Python, pybind11 and separately provisioned NVIDIA SDKs retain their applicable licenses. The package installs [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt), the license texts of the statically linked gRPC, Protobuf, Abseil, RE2, c-ares, OpenSSL and zlib packages and of the FlatBuffers and pybind11 headers, and [BOM.json](BOM.json), which lists each component with its pinned version, SPDX license, upstream project and linkage; `tools/release/third_party_notices.py` generates both from the pinned packages' license files.

## SHA-256 implementation (MIT and BSD-2-Clause)

`source/live/support/SHA256.cpp`, compiled into the plugin, derives from Zilong Tan's SHA-256 implementation, itself derived from Allan Saddi's code. Both the MIT and BSD-2-Clause notices apply; NVIDIA modifications are Apache-2.0. Allan Saddi's original notice comes from `sha-1.1/sha256.c` in [sha-1.1.tar.gz](https://www.saddi.com/software/sha/dist/sha-1.1.tar.gz), SHA-256 `786159140fbb6647a8fbb5d909c721ff58e27c705333457566985777e3d6d730`. The notices are reproduced below and the BSD text is also in `LICENSES/SHA256-BSD-2-Clause.txt` in the client source tree:

```text
The MIT License

Copyright (C) 2011 Zilong Tan (eric.zltan@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```text
Copyright (c) 2001-2003 Allan Saddi <allan@saddi.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
```
