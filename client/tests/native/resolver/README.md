<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# USD resolver hook probe

This test-only plugin and executable check the Sdf/Ar call boundary on the Linux USD 25.11 SDK. The plugin owns only the `live` URI handler and `live` file extension in its isolated test process; it implements no service, transport, production file format or storage, and neither target is installed. Do not load it into Kit or into a process with another live URI resolver.

## Build and run

Configure the scene SDK inputs with `LIVE_BUILD_SCENE_CORE=ON`, `BUILD_TESTING=ON` and `LIVE_BUILD_RESOLVER_PROBE=ON`, build `live_resolver_probe`, then run either test, or the `resolver-probe` suite of `tools/live/test.py`, which runs both:

```bash
ctest --test-dir BUILD -R '^live_resolver_hook_probe$' --output-on-failure
ctest --test-dir BUILD -R '^live_creation_hook_contract$' --output-on-failure
```

`BUILD/resolver-probe/trace.json` records each hook in invocation order with its path and detail, and a failure tail if an assertion fails. The creation cases keep per-child logs and `summary.json` in `BUILD/resolver-probe/creation`. Missing or incomplete contract evidence makes the suite an infrastructure failure even if the subprocess wrapper reports success.

## Cases

- `live_resolver_hook_probe` distinguishes canonical duplicate opens from argument-bearing Sdf identities, new-asset from existing-asset resolution, and Save from Export; an ordinary USDA export and open checks that the supplied scene data is usable by stock USD.
- `live_creation_hook_contract` runs six subprocess cases: anonymous layer creation queued from `SaveToFile` and from the `LayerDidSaveLayerToFile` notice after the outer `CreateNew` returns; failed-target destruction and fresh open; independently retained failed-host backing; initial `_SetLayerData` installation without content notices; and nested anonymous creation inside Export's `WriteToFile`.

The cases cover the hook and ownership boundary only. New tests follow the [determinism rules](../../../docs/ovlive-testing.md#determinism-rules).
