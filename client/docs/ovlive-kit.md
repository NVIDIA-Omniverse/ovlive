<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kit guide

This guide explains how to add the `omni.kit.live_services` extension to a Kit application so that Kit opens, authors and saves ovlive live layers through `omni.usd`. It is for engineers who have a Kit build and want to load the extension, connect it to a Live Services endpoint and author live layers from Kit code. The extension ID is `omni_usd_live`, and it declares two Python modules: `omni.kit.live_services`, the bootstrap that starts the extension, and `omni.usd.live`, the public API that other extensions import. The [Python guide](ovlive-python.md) documents the `omni.usd.live` API that the extension hosts.

Commands in this guide run from the repository's `client` directory, and the paths they name are relative to it; build output and evidence go to `../.cache`, the `.cache` directory at the repository root.

## What works with a public Kit build

The extension does not start on stock public Kit 110.3. There are two reasons, and each stops startup before any live layer opens:

- **USD profile.** The extension's bootstrap accepts only a `kit-25.11` Live Services package, and that package is built against the `ov`-namespace USD of Kit's internal builds. Public Kit 110.3 ships stock OpenUSD 25.11 in the `pxr` namespace, so a `kit-25.11` package does not bind to its USD, and the bootstrap rejects a stock-profile package with `Kit requires the explicit kit-25.11 Live Services package`.
- **Competing `.live` handler.** The stock `omni.usd_resolver` extension registers its own `.live` file format. The bootstrap rejects startup when another plugin declares the `live` extension or URI scheme (`A competing live file format or URI resolver is registered`), because only one `.live` handler may be registered in a process. The stock resolver loads before the extension, and USD cannot unregister a plugin.

An internal Kit build with the companion extensions removes both obstacles; neither is published. The internal Kit build supplies the `ov`-namespace USD that a `kit-25.11` package binds to, and the companion `omni.usd_resolver` omits its own `.live` handler; the companion extensions alone do not make public Kit work. The companion extensions also carry a fix for a Kit bug in the TBB global-control plugin (`omni.tbb.globalcontrol`, part of `omni.usd.core`): the stock plugin leaves its thread-naming observer running after the profiler shuts down, so a Kit process can crash with SIGSEGV at exit after its work has finished. With that build, the extension loads, registers Kit's main thread as the host for live-layer work, and Kit opens, authors, saves and closes live layers through `omni.usd`. The headless reference lane described below exercises that workflow with rendering and Fabric scene delegation off.

The stock `omni.client` extension ships a frame-pump plugin instead of the lifecycle plugin that the Kit reference test fixture requires. That requirement belongs to the fixture, not to the extension: the extension neither loads nor calls the frame pump. The [support boundary's Kit rows](ovlive-support-boundary.md#qualification-boundary) and the [`kit-companion` open item](ovlive-support-boundary.md#open-items) track delivery of these changes in a published Kit build, and the [`kit-rerun` open item](ovlive-support-boundary.md#open-items) tracks the outstanding manual Kit rerun on the current sources.

## Installation

Build the package with `--usd-profile kit-25.11` and the Kit USD SDK named in the [build guide](ovlive-build.md). The build stages the package at `<target-dir>/usd/omniverse/live`, and that package directory is the Kit extension folder. It holds `config/` (`extension.toml` and `live-sdk.json`, the package's USD profile), `lib/libomni_usd_live.so`, the USD plugin registration under `resources/`, the Python packages `omni.kit.live_services` and `omni.usd.live` under `python/`, the C headers under `include/`, the native samples under `bin/`, license and provenance files under `share/`, and `package-manifest.json`. Kit derives the extension ID from the folder name and the extension is enabled as `omni_usd_live`, so the folder must be named `omni_usd_live`: consume a release's packman package with a link folder of that name ([release procedure](ovlive-release.md#publish-to-packman)) or unpack the package, whose root is the extension folder, into a directory of that name, or copy the staged package to a directory of that name, or create a symlink of that name that points to it.

Point Kit at the folder with `--ext-path` and enable the extension along with the companion extensions:

```sh
kit my-app.kit --ext-path /path/to/omni_usd_live \
  --ext-path /path/to/companion/omni.usd_resolver \
  --enable omni.usd --enable omni_usd_live
```

`extension.toml` declares the dependencies `omni.usd`, `omni.usd.libs` and `omni.kit.async_engine`, marks the extension non-reloadable and declares no settings. Endpoints, tokens and TLS policy are configured in code (see [Usage](#usage)); nothing about a service lives in Kit settings. The Kit app settings that the tested configuration requires are listed in the [configuration reference](#configuration-reference).

At startup the extension checks the package before it imports any native Live Services code. Startup fails, and the extension stays disabled, if:

- the extension has already started once in this Kit process ([Python modules and startup order](#python-modules-and-startup-order));
- `config/live-sdk.json` names a profile other than `kit-25.11`;
- the package's Python extension module does not match the running interpreter's ABI suffix;
- the package is incomplete, or another Live Services package, `.live` file format or `live` URI resolver is registered (checked again after importing Kit's `omni.usd`);
- `omni.usd` resolves to a namespace package, which means Kit's `omni.usd` extension has not loaded first;
- another `omni.usd.live` package is loaded or on the search path.

### Python modules and startup order

`extension.toml` declares two `[[python.module]]` entries, both under `python/`, in this order:

```toml
[[python.module]]
name = "omni.kit.live_services"
path = "python"

[[python.module]]
name = "omni.usd.live"
path = "python"
searchExt = false
```

Kit imports an extension's declared modules in declaration order and starts the `omni.ext.IExt` classes it finds in a module before it imports the next one, and Kit's API checker (`repo checkapi`) treats each declared module name as a public module of the extension. The order is what keeps the native binding behind the startup checks:

1. Kit imports `omni.kit.live_services`, which imports no native Live Services code, and runs its extension's `on_startup`.
2. `on_startup` runs the checks listed above. Only when they pass does it register the USD plugin, import `omni.usd.live` and its native `_live` module, and start the host.
3. Kit then reaches the `omni.usd.live` entry, finds the module that `on_startup` imported and uses it as is; `searchExt = false` tells Kit that the module holds no extension class to start. If a check failed, the failure ends the extension's startup before Kit reaches this entry, so the native binding is never imported.

Import `omni.usd.live` (`from omni.usd import live`) from an extension that depends on `omni_usd_live`, at module scope or later; the dependency makes Kit start this extension, and so import the module, first. `omni.kit.live_services` is public too, for `get_host()` and `close_host()`. The underscore-prefixed submodules of both packages, including the native `omni.usd.live._live`, are internal. Outside Kit, `omni.usd.live` is an ordinary package, and importing it loads the native binding directly ([Python guide](ovlive-python.md#setup)).

The extension is non-reloadable and starts once per Kit process. USD cannot unregister the plugin, and when Kit enables a disabled extension again it reloads the extension's modules, which would re-execute `omni.usd.live` over a running host. A second start in the same process therefore fails before any import with `RuntimeError: Live Services starts once per Kit process; restart Kit to enable the extension again`, and Kit never reaches the `omni.usd.live` entry; restart Kit to enable the extension again.

## Usage

**Register an endpoint.** On the Kit main thread, obtain the host owner and register each service authority before opening or creating a live layer:

```python
import omni.kit.live_services
from omni.usd import live

host = omni.kit.live_services.get_host()
registration = host.register_endpoint(
    "live.example.com:443",
    on_auth_required=on_auth_required,  # runs on an API consumer thread
    on_status=on_status,                # likewise; marshal USD and UI work to the main thread
    access_token=token,
)
```

The keyword options are the endpoint options in the [reference](ovlive-reference.md#endpoint-options); the extension supplies no endpoint defaults or credentials. Both callbacks are installed before `register_endpoint` returns. They run on bounded API consumer threads, must return promptly, and must marshal any USD or UI action to the main thread. A host admits 64 endpoint registrations. For a runtime-wide status observer, call `host.observe_status(callback, on_error=...)` and close the returned subscription when the workflow ends.

**Open a live layer.** Open the layer with `Sdf.Layer.FindOrOpen(uri)` on the Kit main thread, then insert it: add its identifier to the session layer's sublayers, or open a stage with it as the root and attach that stage through `omni.usd`'s `UsdContext`. Keep the returned layer referenced until the stage holds it: USD releases a layer nothing references, and composition then opens the URI again as an unopened live URI. Live layers are in-memory only: never save a live URI into a file's sublayers, references or payloads, and remove a live layer from a file-backed layer's sublayers before saving that layer. Composing a live URI that has not been opened is unsupported; USD opens sublayers on its worker threads, so such an open fails with `BUSY` and Kit logs `Could not load sublayer` with a message to open live layers on the host before composing them ([design section 1](ovlive-design.md#1-purpose-audience-and-scope)). The extension has already registered Kit's main thread as the publication context, so plugin jobs run on that thread after the enclosing USD call returns; there is no per-frame live processing. A peer's edit reaches the stage as per-path USD notices, so Kit resyncs only the prims an edit creates, removes or moves and sees a field edit as an info-only change; only the layer's first installation replaces its whole content ([design section 7](ovlive-design.md#7-host-concurrency-and-scheduling-contract)).

**Author.** Set the live layer as the stage's edit target and edit it with ordinary `Sdf` and `Usd` calls on the Kit main thread. The plugin turns each host turn's edits into one update. Edits made off the host, or from inside a handler for a notice that a plugin job sent, move the layer to `FAILED`; [design section 8](ovlive-design.md#8-authoring-and-reconciliation-model) defines the authoring rules.

**Save.** `live.save_layer(layer)` flushes the layer's edits and waits for the service to confirm them. On the Kit main thread it returns `BUSY` while work is pending, so poll it across frames rather than blocking the main thread; the [Python guide](ovlive-python.md#status-waits-and-errors) describes save and drain.

**Sessions.** `host.session_catalog(registration, base_layer_key, current_user=user)` returns a catalogue for one base layer. `catalog.set_active(True)` lists sessions once and then polls every two seconds while active; `catalog.rows` holds the listing. `catalog.create(name)` and `catalog.delete(session_id)` return an operation; `await operation.wait()` yields `(result, payload)`, and `catalog.acknowledge(operation)` releases it once handled. Mutations are never retried; an operation with an unknown outcome keeps the catalogue and the host from closing until it is reconciled. To create the session's root layer, pass the completed create operation and an ordinary USD seed layer to `host.prepare_session_root(catalog, operation, seed)`, then call `root.start()`, `await root.wait()` and `root.activate()`; `root.layer` is the activated live layer. Close each root with `root.close()` when you are done with it. A host admits 64 catalogues and 64 unfinished roots, and each catalogue retains 1,024 rows and 64 unacknowledged operations.

**Close.** Stop authoring, save or export each live layer, detach layers and close roots, then call `omni.kit.live_services.close_host(timeout_ms)` on the Kit main thread. The default timeout of zero polls. `BUSY` or `TIMEOUT` keeps every owner for a later call; closing never saves or discards edits. The extension's shutdown hook only polls and logs an error if cleanup is incomplete, so finish `close_host` before disabling the extension or exiting Kit. The [Python guide](ovlive-python.md#dispatch-bounds-and-teardown) describes what happens to owners left open at interpreter exit.

**Where status shows.** Typed status is the operational signal: the registration's `on_status` callback, `host.observe_status` observers, and the `live.get_layer_status`, `live.get_endpoint_status` and `live.get_session_status` queries. Drive Kit UI from these, never from logs.

## Configuration reference

`extension.toml` declares no settings. The Kit app settings below are the values the tested configuration requires; set them in the application's `.kit` file or on the command line before the USD context initializes.

| Setting | Required value | Why |
| --- | --- | --- |
| `/exts/omni.usd/locking/earlyRelease` | `true` | Kit releases the USD lock after each update, so the stage close worker can acquire it between frames. |
| `/exts/omni.usd/locking/onClose` | `true` | Kit keeps close-time USD locking on; disabling it is not a substitute for early release. |
| `/app/useFabricSceneDelegate` | `false` | The tested configuration runs without Fabric scene delegation. |

Endpoint, token and TLS configuration is passed to `host.register_endpoint(**options)`; the [endpoint options table](ovlive-reference.md#endpoint-options) lists each option and its default. Runtime options and fixed limits are in the [reference](ovlive-reference.md#options-defaults-and-limits).

## Troubleshooting

**Extension fails to start.** Kit's log shows the bootstrap's `RuntimeError`. The message names the failed check from [Installation](#installation); on stock public Kit it is one of the two described in [What works with a public Kit build](#what-works-with-a-public-kit-build).

**A layer stops updating.** Query `live.get_layer_status(uri)`. A `FAILED` layer reports its cause, incident ID and counts of pending and buffered edits; it sends nothing more and saves return the cause. Recover by exporting the layer and reopening it, as the [Python guide](ovlive-python.md#status-waits-and-errors) describes. [Design section 17](ovlive-design.md#17-operational-status) defines each state and its recovery action.

**Diagnostics.** Python-side failures, including incomplete teardown, are logged on the `omni.usd.live` Python logger, and the extension's shutdown hook reports incomplete cleanup through Kit's log. Kit routes Python logging records to its log, whose file `/log/file` selects. Runtime diagnostic records stay in the runtime's history until the host reads them with `live.copy_diagnostics()` or `live.export_diagnostics_json()`, or subscribes with `live.register_diagnostic_callback`; they are not forwarded to Kit's log. The [diagnostic snapshot schema](ovlive-diagnostic-schema.md) describes the records.

### Known Kit issues

Kit's frame-rate limiter can request multi-second sleeps after the realtime clock jumps relative to the monotonic clock, which stalls a Kit process with pending work; run Kit on a host with a stable realtime clock. A Kit process can crash with SIGSEGV at exit because the TBB thread-naming observer in stock `omni.tbb.globalcontrol` outlives the profiler; the companion `omni.usd.core` overlay fixes it. The testing guide lists both under [known test-environment issues](ovlive-testing.md#known-test-environment-issues).

## Running the Kit reference lane

The `kit-reference` suite of `tools/live/test.py` launches one headless Kit process per case, and two for the convergence case (`--no-window --portable`, extension registry off), against the independent reference service, overlays the package and the companion extensions with `--ext-path`, and enables only `omni.kit.loop`, `omni.usd` and `omni_usd_live`. Its cases attach a live sublayer or a live stage root through `UsdContext`, author with plain USD on the Kit main thread, converge two stages, resume after a backend rotation, and preserve edits after a lost write response. The profile is `headless-overlay-reference-v2`. The [support boundary](ovlive-support-boundary.md#supported-configuration) states the policy for service-backed testing.

The lane takes the Kit executable, the package and the companion `omni.client` and `omni.usd_resolver` extensions as `--kit-executable`, `--kit-package`, `--kit-client-extension` and `--kit-resolver-extension`, with optional `--kit-core-extension` and `--kit-library-dir`, and `--kit-release-package`, which selects the [release-package input lane](#validating-a-released-package). The [testing guide](ovlive-testing.md) describes the lane's inputs, checks and replay.

### Validating a released package

The lane qualifies one of two kinds of package, and its inputs select the kind; a missing identity sidecar never switches lanes. Without `--kit-release-package`, `--kit-package` is a build-tree package, whose Python adapter's `_live*.so.manifest.json` sidecar must bind it to the current sources, the plugin driver, the SDK trees and the build interpreter; a folder holding `release-manifest.json` is rejected with a pointer to `--kit-release-package`. With `--kit-release-package <zip>`, `--kit-package` is the folder that released packman package zip was unpacked into, which must be named `omni_usd_live`. The lane then verifies the zip against its `.sha256` checksum file and the release manifest it carries, as `tools/release/package.py` checks a package; requires the manifest to name a clean `kit-25.11` package of the host's platform whose Python extension carries the CPython tag of the manifest's Python component; and requires the folder to hold exactly that manifest and its files, each with its recorded SHA-256 and size, so the binaries and `config/` files Kit loads are the archive's bytes. Only package metadata (`PACKAGE-INFO.yaml`) is ignored, unread.

Every Kit participant reports its interpreter's extension-module suffix and the path, SHA-256 and ABI version of the `_live` module it loaded. Both lanes require the selected adapter and the interpreter ABI its file name carries; the release-package lane also requires the manifest's ABI version. The selected-input identity in `replay.json` and `manifest.json`, `kit-reference-result.json` (`kit_lane`, `release_package_sha256`) and the suite result (`kit_lane`, `kit_release_package_sha256`) record the lane, `build-tree` or `release-package`, and the SHA-256 of the zip a release-package run validated. The harness, the reference service and the Kit driver come from the checkout that runs the lane, whose source fingerprint is recorded apart from the package manifest's `git_head` and `source_fingerprint`: a release-package result qualifies the downloaded archive's bytes under that harness, and a build-tree result never stands for a downloaded archive.

To validate a release candidate's Kit package, check the downloaded candidate artifact with `sha256sum -c SHA256SUMS`, unpack the target's package into a folder named `omni_usd_live`, and run the lane from a checkout that has the release-package lane:

```sh
unzip -q <candidate>/omni_usd_live_ov_openusd_0.25.11_py_3.12--<version>.zip -d <dir>/omni_usd_live
./tools/live/test.sh --suite kit-reference --seed 42 --reference-python <reference python> --reference-openssl <openssl> \
  --kit-executable <kit>/kit --kit-package <dir>/omni_usd_live \
  --kit-release-package <candidate>/omni_usd_live_ov_openusd_0.25.11_py_3.12--<version>.zip \
  --kit-client-extension <kit>/exts/omni.client --kit-resolver-extension <kit>/exts/omni.usd_resolver \
  --kit-core-extension <kit>/exts/omni.usd.core
```

`<version>` is the package version, such as `0.1.0.manylinux_2_35_x86_64.release`, and the `.sha256` file must stay beside the zip. Follow the run with `--replay` of its `replay.json`, which verifies the same zip again before Kit starts.
