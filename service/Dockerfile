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

# Stage 1: Build
FROM rust:1.94-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       clang \
       curl \
       cmake \
       g++ \
       git \
       make \
       mold \
       python3 \
       protobuf-compiler \
       libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Build flatc 24.x from source — Debian's packaged flatbuffers-compiler is too
# old for the flatbuffers Rust crate v24, which marked Table::get as unsafe.
# We also install the C++ headers to /usr/local/include/flatbuffers/ because
# oodf-ffi/build.rs derives the include path from the flatc binary's prefix
# (<prefix>/bin/flatc → <prefix>/include). The cxx-build step in oodf-ffi
# requires flatbuffers/flatbuffers.h to be resolvable.
RUN git clone --depth 1 --branch v24.3.25 https://github.com/google/flatbuffers.git /opt/flatbuffers-source \
    && cmake -S /opt/flatbuffers-source -B /tmp/flatbuffers-build -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_TESTS=OFF \
    && cmake --build /tmp/flatbuffers-build --target flatc -j4 \
    && cp /tmp/flatbuffers-build/flatc /usr/local/bin/flatc \
    && cp -r /opt/flatbuffers-source/include/flatbuffers /usr/local/include/

WORKDIR /build
COPY . .
# Build the entire workspace (live-edit bin + oodf-ffi lib). oodf-ffi's build.rs
# statically links libstdc++ on Linux (cargo:rustc-link-lib=static=stdc++) so the
# final binary carries no libstdc++ dependency, regardless of what the
# distroless/cc runtime image ships — this avoids libstdc++ version skew against
# the base image. oodf-ffi's Cargo.toml suppresses link-cplusplus's competing
# dynamic -lstdc++ on Linux (the `nothing` feature) so the static link is the
# sole stdc++ reference. The ldd guard below enforces this. Works under any
# linker (the clang+mold config in .cargo/config.toml does not affect it).
RUN cargo build --locked --release --workspace

# Package source from the same Cargo lockfile, FlatBuffers checkout and compiler
# runtimes used above. Any missing source or checksum mismatch fails the build.
RUN rustup component add rust-src \
    && python3 tools/package_oss.py --flatbuffers /opt/flatbuffers-source \
       --output /oss-output --work /oss-work

# Verify the binary is not dynamically linked against libstdc++. libstdc++ is
# static-linked regardless of what the base image ships (current distroless/cc
# releases do ship a libstdc++.so, older ones did not) to avoid libstdc++
# version skew against the runtime image; this guard enforces it.
# oodf-ffi/build.rs emits the static-link directives on Linux to ensure this.
# We use `ldd` here because glibc is still allowed to be dynamic (the
# distroless/cc image ships glibc). Only libstdc++ is the concern.
# We capture `ldd`'s output into a variable first so its exit status is the last
# command in its statement (under `set -eu`) and a failure of `ldd` itself —
# missing binary, unreadable ELF, etc. — aborts the RUN. A piped form like
# `ldd … | grep -q libstdc++` would mask that: the pipeline's exit status is
# grep's, the default Dockerfile `/bin/sh` is dash (no `set -o pipefail`), and
# an `ldd` failure feeds empty stdin to grep, which returns 1 (no match) and
# the check silently passes. The if/fi shape (b.bm8) over the prior
# `cmd && (...) || true` shape correctly disambiguates grep's match (failure)
# from no-match (success) — that's the real win — but it does NOT propagate
# ldd-itself errors without capturing output first.
RUN set -eu; \
    LDD_OUT=$(ldd /build/target/release/live-edit); \
    if echo "$LDD_OUT" | grep -q libstdc++; then \
        echo "BUILD FAIL: live-edit is dynamically linked against libstdc++"; \
        exit 1; \
    fi; \
    echo "ldd libstdc++ check OK"

# Runtime smoke test: prove the binary loads and runs past static init. Without
# any LIVE_EDIT_* env vars, main() rejects the missing required env var and
# exits 1 (see src/main.rs / config::from_env). We assert on the exit code
# (b.bm8) rather than grepping the stderr message — the previous grep-based
# check was fragile against JSON log reformatting or an error-message refactor,
# either of which would silently bypass the smoke test. If the binary can't
# load (missing libstdc++ symbols, wrong glibc, etc.) it segfaults or aborts
# with a different exit code, which the check below rejects.
# This test runs in the builder stage because the distroless/cc runtime image
# has no shell and cannot execute pipelined RUN commands. Since the binary is
# statically linked to libstdc++/libgcc (independent of whatever libstdc++ the
# base image ships), the same binary behavior is verified here as would be in
# the runtime image.
RUN env -i /build/target/release/live-edit; rc=$?; \
    [ "$rc" = "1" ] || (echo "smoke-test FAIL: expected exit 1 from missing-config startup, got $rc" && exit 1)

# Scaler-mode smoke test: the chart's scaler Deployment runs this same binary
# with the `scaler` argument, which must dispatch to the autoscaler before the
# service configuration loads. With an empty environment the scaler refuses
# its own missing configuration and exits 1. The exit code alone cannot tell
# that apart from the service refusing ITS missing configuration (also exit
# 1), which is what a broken dispatch would produce, so the captured stderr
# must name a scaler variable as well. Same capture-then-check shape as the
# gates around it.
RUN set -u; \
    SCALER_ERR=$(env -i /build/target/release/live-edit scaler 2>&1 >/dev/null); rc=$?; \
    if [ "$rc" != "1" ]; then \
        echo "scaler smoke-test FAIL: expected exit 1 from scaler mode with no configuration, got $rc"; \
        echo "$SCALER_ERR"; \
        exit 1; \
    fi; \
    case "$SCALER_ERR" in \
        *LIVE_EDIT_SCALER_NAMESPACE*) echo "scaler smoke-test OK" ;; \
        *) echo "scaler smoke-test FAIL: stderr does not name LIVE_EDIT_SCALER_NAMESPACE, so the scaler argument did not reach the scaler"; echo "$SCALER_ERR"; exit 1 ;; \
    esac

# Allocator gate (b.u5k): prove the shipped artifact allocates through jemalloc
# with its retention options compiled in, and without the heap profiler.
# `confirm_conf:true` makes jemalloc print every option it applies to stderr as
# it initialises, before main() runs, so the same missing-config startup as the
# smoke test above (exit 1) is enough. The three lines asserted:
#   - `Set conf value: background_thread:true` and `Set conf value: thp:never`
#     — the options the binary defines in `src/allocator.rs` were read and
#     applied. A binary on the system allocator prints no `<jemalloc>:` lines.
#   - `Invalid conf pair: prof:false` — jemalloc rejects the `prof` key, which
#     it only does when the profiler is NOT compiled in (no
#     `jemalloc-profiling` feature). A profiling build prints
#     `Set conf value: prof:false` instead and fails here.
# `env -i` keeps an inherited `_RJEM_MALLOC_CONF` from changing the result.
# Same capture-then-check shape as the ldd guard (b.bm8): stderr is captured
# into a variable (stdout discarded) and the exit status read from the
# assignment, so a crash or a missing binary cannot pass through a pipeline
# whose status is grep's. `case` matches the captured text with no pipe at all.
RUN set -u; \
    ALLOC_ERR=$(env -i _RJEM_MALLOC_CONF=confirm_conf:true,prof:false /build/target/release/live-edit 2>&1 >/dev/null); rc=$?; \
    if [ "$rc" != "1" ]; then \
        echo "allocator gate FAIL: expected exit 1 from missing-config startup, got $rc"; \
        echo "$ALLOC_ERR"; \
        exit 1; \
    fi; \
    for want in '<jemalloc>: -- Set conf value: background_thread:true' \
                '<jemalloc>: -- Set conf value: thp:never' \
                '<jemalloc>: Invalid conf pair: prof:false'; do \
        case "$ALLOC_ERR" in \
            *"$want"*) ;; \
            *) echo "allocator gate FAIL: stderr lacks: $want"; echo "$ALLOC_ERR"; exit 1 ;; \
        esac; \
    done; \
    echo "allocator gate OK"

# Measurement-feature gate: the shipped artifact carries no measurement-only
# code. The test-only `test-support` cargo feature (Cargo.toml) makes the
# binary read the snapshot fan-out arena's decay from
# MEASUREMENT_FANOUT_ARENA_DIRTY_DECAY_MS at startup (`src/allocator.rs`) and
# the pod memory limit's cgroup root from MEASUREMENT_POD_MEMORY_CGROUP_ROOT
# (`src/config/pod_memory.rs`); a build without it carries no reference to
# either name, so the names' absence from the binary shows the feature is
# off. The control: each source file declares its name exactly once, as a
# string literal, so a renamed variable fails here rather than leaving the
# search looking for a name nothing carries.
# `scripts/check-measurement-features.sh` runs the same search (`--binary`),
# checks that a `test-support` build does carry the names
# (`--control-binary`, run by hand on such a build) and, in CI, fences the
# feature's manifest and resolved edges; `scripts/` is not in the build
# context, so the names are repeated here.
# grep's status is read directly, never through a pipeline: 0 is found, 1 not
# found, and anything else (an unreadable binary) fails rather than passing as
# not found.
RUN set -u; \
    for PAIR in MEASUREMENT_FANOUT_ARENA_DIRTY_DECAY_MS:src/allocator.rs MEASUREMENT_POD_MEMORY_CGROUP_ROOT:src/config/pod_memory.rs; do \
        ENV_NAME=${PAIR%%:*}; \
        ENV_SOURCE=${PAIR#*:}; \
        HITS=$(grep -c -F "\"$ENV_NAME\"" "$ENV_SOURCE"); \
        if [ "$HITS" != "1" ]; then \
            echo "measurement-feature gate FAIL: $ENV_SOURCE declares \"$ENV_NAME\" ${HITS:-0} time(s), not once; update this gate to the variable's current name"; \
            exit 1; \
        fi; \
        grep -q -a -F -- "$ENV_NAME" /build/target/release/live-edit; rc=$?; \
        case "$rc" in \
            0) echo "measurement-feature gate FAIL: the release binary contains $ENV_NAME, so it was built with test-support"; exit 1 ;; \
            1) echo "measurement-feature gate OK: no $ENV_NAME" ;; \
            *) echo "measurement-feature gate FAIL: grep could not read the release binary (exit $rc)"; exit 1 ;; \
        esac; \
    done

# Stage 2: Runtime
FROM nvcr.io/nvidia/distroless/cc:v4.1.4

COPY --from=builder /build/target/release/live-edit /app/live-edit
COPY --from=builder --chmod=0644 /oss-output/vendor.tar.gz /vendor.tar.gz
COPY --from=builder /oss-output/licenses /licenses

EXPOSE 50001 9090

# UID/GID 1000 is the `nvs` user in the NVIDIA hardened distroless base.
# Numeric form (not `nvs:nvs`) is required so Kubernetes `runAsNonRoot`
# admission can verify the user is non-root without reading /etc/passwd.
USER 1000:1000

ENTRYPOINT ["/app/live-edit"]
