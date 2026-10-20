<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Omniverse Live Edit Service

A gRPC service for real-time collaborative editing of NVIDIA Omniverse scene layers.

## Overview

Live Edit lets several users edit the same Omniverse scene layers at once, in a shared live session.
Clients send their edits as incremental deltas over the `nvidia.omniverse.live.v1alpha.LiveService` gRPC API.
The service applies each delta to its copy of the layer and streams the change to every other participant.
It also tracks who is in a session and relays messages between participants.

- **Stateful.** The service applies every delta, so it can rebuild a layer's current state for a client that joins late.
- **Valkey-backed.** Live state is in Valkey, a Redis-compatible store that is authoritative, not a cache. Optional checkpoints to a separate backup store protect against losing it.
- **No session affinity.** Any replica serves any call, so replicas scale out behind a load balancer.
- **Authenticated.** Every call carries a bearer token, and by default access to a session is checked against the storage service that holds its base layer.

Clients need Live Edit protocol support: deploying the service does not by itself make an arbitrary USD application collaborative.

## Documentation

- [Operator documentation](docs/service/docs/index.md): deploy with the Helm chart, configure, choose and size a backing store, monitor, upgrade and troubleshoot.
- [Client API documentation](docs/api/docs/index.md): the RPCs, error codes, retries and limits that client developers rely on.
- [`live_service.proto`](proto/nvidia/omniverse/live/v1alpha/live_service.proto): the gRPC API definition.
- [Conformance suite](conformance_tests/README.md): a container that checks a running service through its public gRPC API.
- [`CHANGELOG.md`](CHANGELOG.md): release notes.

The pages are written for the NVIDIA documentation site; on GitHub, their links to each other show as raw `{ref}` text, and every page they link is in the same directory.

## Architecture

```
          clients (gRPC over HTTP/2)
                     |
     ingress or load balancer (terminates TLS)
                     |
      +--------------+--------------+
      |              |              |
  live-edit      live-edit      live-edit    --> storage service (authorization)
  replica        replica        replica      --> userinfo service (optional)
      |              |              |        --> backup store (optional checkpoints)
      +--------------+--------------+
                     |
          Valkey (live state, pub/sub)
```

Each replica keeps only presence, subscription routing and disposable caches in memory; all durable data is in Valkey.
Replicas serve plaintext HTTP/2 on port 50001, so client TLS ends at the ingress.
Outbound connections use TLS when the target's address asks for it: a `rediss://` Valkey URL, or an `https://` service endpoint.
Prometheus metrics are served on port 9090.

## Build from source

### Prerequisites

| Tool | Version | Used for |
|---|---|---|
| Rust | 1.94 (`rust-toolchain.toml` selects it; `rustup` installs it) | the service |
| `protoc` | 3.x or later | gRPC code generation |
| `flatc` | 24.3.25 (majors 23 to 25 accepted), with its C++ headers | the C++ library for the delta format |
| C++17 compiler | Apple Clang, `g++` or `clang++` | the C++ library for the delta format |
| `make`, and a C compiler run as `cc` | Linux only | the jemalloc memory allocator |
| Docker | any | optional: a local Valkey and the container image |
| `grpcurl` | any | optional: checking a running service |

The build fails when `flatc`'s major version is outside 23 to 25, and warns when it is not 24.
It looks for the FlatBuffers headers at `<prefix>/include/flatbuffers/`, beside `<prefix>/bin/flatc`.

- **macOS:** install the Xcode Command Line Tools (`xcode-select --install`), then `brew install protobuf flatbuffers`.
- **Linux (Debian or Ubuntu):** run `sudo apt install build-essential protobuf-compiler cmake git`, then build `flatc` 24.3.25 and install it with its headers:

```bash
git clone --depth 1 --branch v24.3.25 https://github.com/google/flatbuffers.git
cmake -S flatbuffers -B flatbuffers/build -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_TESTS=OFF
cmake --build flatbuffers/build --target flatc -j4
sudo cp flatbuffers/build/flatc /usr/local/bin/flatc
sudo cp -r flatbuffers/include/flatbuffers /usr/local/include/
```

### Build

```bash
cargo build --release --locked --workspace
```

The binary is `target/release/live-edit`.
The same binary runs the Helm chart's replica autoscaler as `live-edit scaler`; you do not start that yourself.

## Run locally

The service needs a Valkey with `maxmemory-policy noeviction`, and it refuses to start without a durability layer: Valkey append-only-file persistence (AOF), checkpoints, or both.
Start a throwaway Valkey with AOF on, on a Docker network the container image can join later:

```bash
docker network create live-edit
docker run -d --rm --name live-edit-valkey --network live-edit -p 6379:6379 \
  valkey/valkey:9.0.3-alpine \
  valkey-server --appendonly yes --appendfsync everysec --maxmemory 1gb --maxmemory-policy noeviction
```

Then start the service:

```bash
# The Valkey to use (default redis://localhost:6379).
export LIVE_EDIT_VALKEY_URL="redis://127.0.0.1:6379"
# Required: the storage service that authorizes access to each session's base layer.
# If it is unreachable the service still starts, but every call that checks permission fails.
export LIVE_EDIT_STORAGE_SERVICE_ENDPOINT="https://{STORAGE_SERVICE_HOST}"
cargo run --release --locked
```

The service reaches the storage service over REST by default (`LIVE_EDIT_AUTH_PROTOCOL=grpc` switches to gRPC).
With the userinfo service off, the default, it reads each user's identity from the bearer token, which must then be a JWT.
Every other setting has a default; the [configuration reference](docs/service/docs/configuration.md) lists them all.

The service logs JSON lines to standard output.
It is ready when the log shows `readiness raised to SERVING`.
Then check the gRPC port and the metrics port:

```bash
grpcurl -plaintext localhost:50001 list    # lists nvidia.omniverse.live.v1alpha.LiveService
curl -s localhost:9090/metrics | head
```

If the service exits with `live-edit refuses to start with no durability configured`, the Valkey it reached has AOF off: start one as shown above.
To clean up, stop the service, then run `docker stop live-edit-valkey` and `docker network rm live-edit`.

## Container image

Build the image on the CPU architecture it will run on, and run it against the Valkey from [Run locally](#run-locally):

```bash
docker build -t live-edit-service .
docker run --rm --network live-edit -p 50001:50001 -p 9090:9090 --memory 5g \
  -e LIVE_EDIT_VALKEY_URL="redis://live-edit-valkey:6379" \
  -e LIVE_EDIT_STORAGE_SERVICE_ENDPOINT="https://{STORAGE_SERVICE_HOST}" \
  live-edit-service
```

The service sizes its memory use from the container's memory limit (`--memory`), and exits naming the size it needs if the limit is too small; without a limit it logs a warning and sizes itself as a 5 GiB container.
Check it as in [Run locally](#run-locally): the log shows `readiness raised to SERVING`, `grpcurl -plaintext localhost:50001 list` lists the `LiveService`, and `curl -s localhost:9090/metrics` answers.

NVIDIA publishes release images as `nvcr.io/nvidia/omniverse/live-edit-service:{VERSION}`.
The image runs as the non-root user `1000:1000` on a distroless base, so it has no shell.

### Open-source notices and sources

Each image carries:

- `/licenses/`: the service's license, `LICENSE.txt`, with NVIDIA's attribution in `NOTICE`; `ATTRIBUTIONS.md` with the license texts of the Rust crates compiled into the service, the other third-party notices, `SOURCES.md` describing what is shipped, and `SOURCE-MANIFEST.json` recording the service version, compiler versions and binary hashes.
- `/vendor.tar.gz`: the source of the third-party code built into the binary, including the locked Rust crates, FlatBuffers, the Rust standard library and the GCC runtime.

Copy them out without starting the service:

```bash
container=$(docker create nvcr.io/nvidia/omniverse/live-edit-service:{VERSION})
docker cp "$container:/licenses" ./live-edit-licenses
docker cp "$container:/vendor.tar.gz" ./live-edit-oss-source.tar.gz
docker rm "$container"
```

To check an image's source archive, notices and binary checksums, run `sh tools/check-container-oss.sh {IMAGE}`.

## Deploy on Kubernetes

Production deployments use the Helm chart in `helm/live-edit-service/`, which NVIDIA also publishes with each release.
Follow the [deployment guide](docs/service/docs/deployment.md): with the chart's defaults the service refuses to start until you choose a durability mode.
The [configuration reference](docs/service/docs/configuration.md) lists the settings by Helm value, and `helm/live-edit-service/values.yaml` documents each one in place.

## License

This repository is licensed under the Apache License 2.0 (`LICENSE.txt`), unless `NOTICE`, a directory's own license file or a file's license notice says otherwise.
In `vendor/fred-10.1.0/`, a patched copy of the fred Valkey client crate, fred's own files and NVIDIA's edits to them are under the MIT License, and NVIDIA's added files under `vendor/fred-10.1.0/src/live_edit/` are under the Apache License 2.0.
`NOTICE` holds NVIDIA's copyright attribution and the fred crate's licenses.
`oss/SOURCE-INVENTORY.md` records how each file and release artifact is licensed, the third-party code and dependencies, and where each copied source comes from.

To report a security vulnerability, follow [`oss/SECURITY.md`](oss/SECURITY.md).
