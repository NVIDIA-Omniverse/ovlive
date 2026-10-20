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

FROM rust:1.94.1-slim-bookworm AS codec
RUN apt-get update && apt-get install -y --no-install-recommends clang cmake g++ git make ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v24.3.25 https://github.com/google/flatbuffers.git /tmp/flatbuffers \
    && cmake -S /tmp/flatbuffers -B /tmp/flatbuffers/build -DFLATBUFFERS_BUILD_TESTS=OFF \
    && cmake --build /tmp/flatbuffers/build --target flatc -j4 \
    && cp /tmp/flatbuffers/build/flatc /usr/local/bin/flatc \
    && cp -r /tmp/flatbuffers/include/flatbuffers /usr/local/include/
WORKDIR /build
COPY conformance_tests/native-workspace.toml Cargo.toml
COPY conformance_tests/native-Cargo.lock Cargo.lock
COPY LICENSE.txt LICENSE.txt
COPY oodf-ffi/ oodf-ffi/
# Exercise GNU archive ordering on both ARM64 and x86_64 (not mold/LLD).
RUN RUSTFLAGS="-C linker=cc -C link-arg=-fuse-ld=bfd" \
    cargo build --locked --release -p oodf-ffi --example decode_ompf -j4
RUN set -eu; \
    deps="$(ldd target/release/examples/decode_ompf)"; \
    if printf '%s\n' "$deps" | grep -E 'libstdc\+\+|not found'; then exit 1; fi

FROM python:3.13.7-slim-bookworm AS package
RUN pip install --no-cache-dir uv==0.8.22
WORKDIR /suite
ENV UV_PROJECT_ENVIRONMENT=/opt/conformance
COPY conformance_tests/pyproject.toml conformance_tests/uv.lock ./
COPY conformance_tests/src/ src/
RUN uv sync --frozen --no-dev --no-editable \
    && chmod -R a+rX /opt/conformance

FROM python:3.13.7-slim-bookworm
COPY --from=package /opt/conformance /opt/conformance
# Host umasks must not determine whether the non-root runtime can read/execute assets.
COPY --from=codec --chmod=0755 /build/target/release/examples/decode_ompf /usr/local/bin/decode_ompf
# Public unsigned fixture; no default credential is configured in this image.
COPY --chmod=0644 conformance_tests/local-test-token.jwt /opt/local-test-token.jwt
COPY --chmod=0644 LICENSE.txt NOTICE ATTRIBUTIONS.md /usr/share/doc/live-edit-conformance-tests/
ENV PATH="/opt/conformance/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFORMANCE_REPORT_DIR=/reports
RUN mkdir /reports && chown 65532:65532 /reports
WORKDIR /reports
USER 65532:65532
RUN python -c "from types import SimpleNamespace; from live_edit_conformance.protocol import decode, fixture; decode(SimpleNamespace(decoder='decode_ompf', timeout=15), fixture('external'))"
RUN CONFORMANCE_AUTH_TOKEN_FILE=/opt/local-test-token.jwt CONFORMANCE_BASE_PATH=https://example.test/conformance/base.usda \
    python -c "from live_edit_conformance.config import Settings; Settings.from_env().metadata()"
RUN run-conformance-tests --help
ENTRYPOINT ["run-conformance-tests"]
