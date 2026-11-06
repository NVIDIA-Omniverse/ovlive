<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Storage benchmarks

`tools/live/benchmark.py` characterizes the exact Sdf scene storage (`SdfSceneData`) that backs live layers, and `tools/live/paired_benchmark.py` compares two revisions of it. This guide is for contributors who change that storage and want to measure the effect. The tools measure a storage component in an unsanitized Release build; their wall-clock results are informational, reports for comparing two revisions on one machine, never a gate, and the report states `qualification.state` `unverified`. The gate is the [work budgets](#work-budgets), which count work instead of timing it. The guide also [records the interactive edit benchmark run](#recorded-edit-benchmark-results) that the support boundary cites.

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

## Work budgets

`tests/native/scene/work_budgets.cpp` freezes a budget for each case below in the units of the scene work meter, `source/live/scene/SceneWork.h`: the elements scene code visits or copies, the key comparisons of its ordered indexes, and the atomic updates of the owner pins and reentry-depth guards that host storage and its projection take. The counts depend only on the source, the seeded workload and the toolchain, and the section IDs that a turn's created specs receive come from a seeded generator, so a case counts the same on every machine and run with the same toolchain and STL checking mode. Under MSVC the host reparent, reorder and subtree-delete turns make a different number of index comparisons than under GCC, with the same elements and atomic updates, so `kFrozenMsvc` holds their MSVC comparison counts, which replace the frozen ones on Windows. Its checked-STL column applies only at `_ITERATOR_DEBUG_LEVEL=2`: MSVC validates comparator ordering with extra calls, and the sibling-sort comparator performs scene-index lookups that the meter counts. The Windows debug stock profile repeatedly measured 78,509/169,905/85,431 comparisons for these three turns; their release counts remain 69,409/148,233/72,411. Element and atomic counts and the ten-percent allowance are unchanged. Each count's budget is its frozen value plus a tenth, rounded up: the counts are exact, so the headroom is not for noise; it lets an incidental extra visit pass, while a change that does a tenth more work fails. The check fails when any count of any case exceeds its budget, and when the measured cases and the frozen table differ.

| Cases | What each runs once |
| --- | --- |
| `storage.*` | Each workload of `tests/benchmarks/scene-workloads.json` on `SdfSceneData`, with the benchmark driver's seeded history; a mixed case runs each reader's reads after the writer's transaction on one thread, since a read's counts do not depend on the threads around it |
| `turn.*` | Host turns on a layer of 4,000 unedited prims, each through capture, compilation, admission, the acknowledgment's application and publication: four field turns, a sample turn, a create, rename, reparent, reorder and subtree delete, a turn of 150 field writes and one that rewrites them 20 times, and a one-field write to a spec of 8,192 fields |
| `publication.*` | A peer's field update and its create, rename and delete of a child prim on that layer, and two peer updates that write and then change the 8,192 fields of one spec |
| `install.*` | Copying a scene with an 8,192-field spec into empty host storage and over storage that holds it |
| `flush.capture` | The capture of host storage that each flush compiles against, on the 4,000-prim layer |
| `read.*` | 12,000 reads through the host projection, one per attribute of the layer, as USD composition makes them: a value, a typed value, a spec type and a time sample |

The scene journal driver checks the budgets after its other cases, so the `scene-journal` suite runs them, and `--work-budgets` runs only them, printing each case's counts and budgets:

```sh
../.cache/local/build/live_scene_journal_test --work-budgets
```

The `scene-journal` suite checks the budgets on both `stock-25.11` and `kit-25.11`; both profiles count the same.

To change a budget, run `--work-budgets`, copy the printed counts of the cases whose work changed on purpose into `kFrozen` (and, for a case `kFrozenMsvc` lists, its comparisons from a Windows run into `kFrozenMsvc`), and state the reason in the pull request.

Two costs measured here are accepted rather than reduced:

- **The capture per flush.** A flush compiles against a capture of the whole host storage. The capture shares the storage's immutable spec records and index structure, so on the 4,000-prim layer it visits and copies no element and makes no comparison; its 18 atomic updates are the pins and guards of the copy, independent of the layer's size. Its budget is zero elements and zero comparisons.
- **Atomic updates per composition read.** Each read through the host projection pins the projection and its storage and counts itself in the projection's reentry depth (six updates), and host storage pins itself and the spec's record (four more): a value or time-sample read makes 10 atomic updates, a spec-type read 8, and a typed-value read, which also takes the storage's reentry guard, 14. They are uncontended reference-count and counter updates on the reading thread, constant per read whatever the layer's size, and the `read.*` budgets hold them at those counts.

## Recorded edit benchmark results

The [interactive edit benchmark](ovlive-testing.md#interactive-edit-benchmark) is a characterization; this section records the run that the [support boundary](ovlive-support-boundary.md#qualification-boundary) cites for the comparison with the previous Live Services client and for the cost of diagnostics. Both clients ran on the same machine in the same session, so the comparison needs no dedicated worker; a rerun on other hardware gives other absolute times.

- **Date and revision.** 2026-10-28, ovlive `b2955181` built with `./build.sh` on the `stock-25.11` profile (Release), against Omniverse Client Library `2.74.0-release.7316+gl.ec99a64b` for `manylinux_2_35_x86_64`.
- **Machine.** Intel Core Ultra 9 285K, 24 cores (one thread each), 31 GiB of memory visible to WSL 2, Ubuntu 22.04.5 on WSL 2 kernel 6.6.87.2-microsoft-standard-WSL2. The machine was not otherwise idle: other development work ran beside the benchmark, at a load average of about 4 to 12 on its 24 cores.
- **Command.** The testing guide's acceptance-bar command at seed 42 with scenarios `single,multi,burst,multi-burst,batch-in-one-frame` and round trips 20, 50 and 100 ms, the opaque reference service and the default 2 ms proxy jitter; the diagnostics runs add `--clients ovlive` and `--diagnostics default` or `--diagnostics reduced`.

**Comparison with Client Library 2.74.0.** Each cell is ovlive/baseline, as the summary's comparison table prints it; latencies are milliseconds at the raw peer. Every case delivered every turn on both sides. ovlive meets the bar in 4 of 15 rows.

| Scenario | RTT ms | p50 | p99 | Published turns/s | Meets the bar |
| --- | --- | --- | --- | --- | --- |
| single | 20 | 26.0/25.7 (worse) | 28.2/27.8 (worse) | 59.946/59.948 (worse) | no |
| single | 50 | 56.2/55.9 (worse) | 57.8/62.2 | 59.766/59.772 (worse) | no |
| single | 100 | 106.8/106.1 (worse) | 110.5/109.3 (worse) | 59.469/59.464 | no |
| multi | 20 | 28.3/27.7 (worse) | 33.3/32.0 (worse) | 59.921/59.929 (worse) | no |
| multi | 50 | 58.5/57.4 (worse) | 63.9/60.9 (worse) | 59.752/59.764 (worse) | no |
| multi | 100 | 107.3/107.4 | 110.2/110.8 | 59.451/59.463 (worse) | no |
| burst | 20 | 34.0/48.0 | 48.9/49.8 | 4735/19223 (info) | yes |
| burst | 50 | 84.1/109.8 | 113.1/111.7 (worse) | 4333/8838 (info) | no |
| burst | 100 | 170.9/211.0 | 210.3/213.9 | 3110/4640 (info) | yes |
| multi-burst | 20 | 28.6/51.2 | 37.8/89.9 | 764/6041 (info) | yes |
| multi-burst | 50 | 63.7/99.3 | 101.7/134.5 | 1134/6543 (info) | yes |
| multi-burst | 100 | 165.7/155.8 (worse) | 212.6/193.6 (worse) | 1745/4701 (info) | no |
| batch-in-one-frame | 20 | 164.8/64.9 (worse) | 271.7/100.2 (worse) | 3652/9900 (info) | no |
| batch-in-one-frame | 50 | 212.8/99.8 (worse) | 314.9/135.6 (worse) | 3154/7330 (info) | no |
| batch-in-one-frame | 100 | 261.7/145.1 (worse) | 387.3/183.6 (worse) | 2566/5422 (info) | no |

The sustained rows miss by under 1.1 ms at p50 and 3.1 ms at p99, and by under 0.02 turns/s at the 60 Hz authoring rate, within what repeated runs vary; the bar counts any shortfall. `batch-in-one-frame` misses at every round trip, with p50 2.5 times the baseline's at 20 ms and 1.8 times at 100 ms, and `multi-burst` misses at 100 ms by about 6% at p50 and 10% at p99. The writer processes' peak RSS was 98 to 126 MiB for ovlive and 113 to 135 MiB for the baseline.

**Diagnostics at their defaults and reduced.** The same scenarios with the ovlive writer only, once with the runtime's default diagnostics and once with diagnostic history disabled (`DISABLE_DIAGNOSTIC_HISTORY`) and the minimum severity at `ERROR`. Each cell is default/reduced; CPU is the writer process's user and system time over its whole life, including interpreter and USD startup, and RSS its peak resident set (`ru_maxrss`). Every case delivered every turn in both modes.

| Scenario | RTT ms | p50 ms | p99 ms | Published turns/s | Writer CPU s | Writer peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| single | 20 | 26.1/26.3 | 28.5/28.5 | 59.9/59.9 | 1.14/1.24 | 99/97 |
| single | 50 | 56.3/56.6 | 58.0/59.5 | 59.8/59.8 | 1.18/1.36 | 99/98 |
| single | 100 | 106.0/105.9 | 108.1/108.1 | 59.5/59.5 | 1.19/1.19 | 99/98 |
| multi | 20 | 27.3/27.2 | 29.6/29.3 | 59.9/59.9 | 2.09/2.00 | 100/98 |
| multi | 50 | 57.6/57.3 | 59.6/59.0 | 59.8/59.8 | 2.21/1.95 | 100/98 |
| multi | 100 | 107.8/106.9 | 118.6/109.1 | 59.5/59.5 | 2.39/1.83 | 101/98 |
| burst | 20 | 37.0/32.7 | 49.3/45.2 | 6408/6614 | 0.41/0.41 | 100/97 |
| burst | 50 | 85.8/87.8 | 109.5/109.1 | 4412/5959 | 0.42/0.34 | 99/97 |
| burst | 100 | 162.5/169.0 | 208.0/208.2 | 3127/4715 | 0.43/0.35 | 99/97 |
| multi-burst | 20 | 27.3/26.8 | 41.3/39.0 | 1111/1269 | 1.29/1.15 | 102/101 |
| multi-burst | 50 | 76.2/78.1 | 103.9/103.8 | 1853/2076 | 0.79/0.72 | 107/103 |
| multi-burst | 100 | 159.3/161.8 | 206.0/212.8 | 1751/2126 | 0.76/0.67 | 113/110 |
| batch-in-one-frame | 20 | 161.0/157.4 | 263.6/255.7 | 3763/3879 | 0.51/0.48 | 126/124 |
| batch-in-one-frame | 50 | 206.4/192.2 | 318.1/292.0 | 3118/3400 | 0.55/0.49 | 127/124 |
| batch-in-one-frame | 100 | 254.7/240.3 | 365.0/336.2 | 2724/2955 | 0.53/0.45 | 127/124 |

Over the 15 cases the writer used 15.89 s of CPU with default diagnostics and 14.61 s reduced, 8.8% more, and its peak RSS was 1 to 4 MiB higher; latency and sustained throughput differ by no more than the run-to-run variation the comparison above shows. A case emitted at most 606 records; at the defaults all but 3 per case fell below the `INFO` minimum, and reduced suppressed every one. Peak RSS stayed between 97 and 127 MiB in every scenario, highest in `batch-in-one-frame`, which buffers 1,000 change blocks of 150 attributes before its one flush.
