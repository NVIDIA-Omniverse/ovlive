<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Storage benchmarks

`tools/live/benchmark.py` characterizes the exact Sdf scene storage (`SdfSceneData`) that backs live layers, and `tools/live/paired_benchmark.py` compares two revisions of it. This guide is for contributors who change that storage and want to measure the effect. The tools measure a storage component in an unsanitized Release build; they are not a product performance gate and the report states `qualification.state` `unverified`.

Commands in this guide run from the repository's `client` directory, and the paths they name are relative to it; build output and evidence go to `../.cache`, the `.cache` directory at the repository root.

## Running a characterization

Build `live_scene_benchmark` in Release with the USD and Python SDKs of the [build guide](ovlive-build.md#direct-cmake), then run the characterization:

```sh
cmake -S . -B ../.cache/live-benchmark-release -DCMAKE_BUILD_TYPE=Release \
  -DLIVE_BUILD_SCENE_CORE=ON \
  -DLIVE_USD_ROOT="$LIVE_BENCH_USD_ROOT" \
  -DLIVE_USD_PYTHON_ROOT="$LIVE_BENCH_PYTHON_ROOT"
cmake --build ../.cache/live-benchmark-release --target live_scene_benchmark -j3
python3 -B tools/live/benchmark.py \
  --driver ../.cache/live-benchmark-release/live_scene_benchmark --seed 42 --trials 7
```

`--trials` takes 2 to 100 (default 7) and `--seed` defaults to 42. Debug and sanitized binaries are rejected. Each trial runs every case in a fresh child process, in an order shuffled per round, with identical seeded work across trials. After timing, stock `SdfData` regenerates the inputs from the seed and checks the final scene and read checksum exactly; any mismatch, crash or timeout fails the run.

## Cases

`tests/benchmarks/scene-workloads.json` declares the cases:

| Case | One timed operation |
| --- | --- |
| Scalar, small scene | One field Set of a prebuilt scalar over 32 specs |
| Scalar, larger scene | The same operation over 4096 attribute specs |
| Array installation | Set of a prebuilt 4096-element copy-on-write array |
| Time samples | One SetTimeSample of a prebuilt scalar at a generated key |
| Namespace move pair | Two one-spec MoveSpec calls, each checked by HasSpec and GetSpecType |
| Field reads | One Get, type check and checksum accumulation over 4096 specs |
| Mixed, two readers | One writer installs a revision and payload pair under the write guard while each reader checks a coherent pair under the read guard |
| Mixed, four readers | The same transactions with four reader threads |

Each mixed case runs 8192 writer transactions and 8192 reads per reader over 128 attribute specs, and reports the writer and each reader separately.

## Reports and replay

Each run writes `report.json`, `workloads.json`, `seeds.json` and bounded child logs to a new `../.cache/live-benchmarks/run-*` directory. The report keeps every raw trial with wall and process CPU time, per-operation p50, p95 and p99 latency, throughput, candidate destruction time and process peak RSS, plus the source fingerprint, compiler flags, binary and SDK identities, CPU model, OS and Python environment. Summary percentiles are the median, minimum and maximum of the per-trial percentiles.

Replay reuses the recorded seed, trial count and workloads and refuses to run if the binary, source, SDK, interpreter or machine identity changed:

```sh
python3 -B tools/live/benchmark.py --replay ../.cache/live-benchmarks/run-EXAMPLE/report.json
```

## Comparing two revisions

Build the candidate and the reference revision in separate source and build trees with the same SDKs, compiler and Release configuration, and keep both unchanged until the run and any replay finish. The benchmark driver source must match between them.

```sh
python3 -B tools/live/paired_benchmark.py \
  --candidate-driver ../.cache/live-benchmark-release/live_scene_benchmark \
  --reference-root "$LIVE_BENCH_REFERENCE_SOURCE" \
  --reference-driver "$LIVE_BENCH_REFERENCE_BUILD/live_scene_benchmark" \
  --seed 42 --trials 20
```

`--trials` takes 20 to 100 pairs. Each pair runs both revisions adjacently in fresh processes, with case order randomized per trial and the first revision balanced. For every case the report gives the median candidate/reference ratio of elapsed time, writer (or operation) p99 and each reader's p99, with an exact order-statistic confidence interval Bonferroni-adjusted across all comparisons for 95% family coverage. An interval wholly below 0.99 is labeled `improvement`, wholly above 1.01 `regression`, and otherwise `inconclusive`; too few pairs give `insufficient_pairs` and a zero timer gives `unmeasurable`. A regression label does not change the exit code. Results go to `../.cache/live-paired-benchmarks/run-*`; replay with:

```sh
python3 -B tools/live/paired_benchmark.py \
  --replay ../.cache/live-paired-benchmarks/run-EXAMPLE/report.json
```
