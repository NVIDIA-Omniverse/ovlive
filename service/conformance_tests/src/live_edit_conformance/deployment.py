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
#
"""Optional Storage API fixture provisioning around the public-wire suite.

Only a UUID-named file created by this invocation is removed. The test suite
itself only calls LiveService, so alternative implementations need no Storage API.
"""

import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .config import Settings
from .runner import run as run_tests, emit_reports, record_failure


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Storage API fixture endpoint must not redirect bearer credentials")


def run():
    endpoint = os.environ.get("CONFORMANCE_STORAGE_ENDPOINT", "").rstrip("/")
    if not endpoint:
        return run_tests()
    base = os.environ.get("CONFORMANCE_BASE_PATH", "").rstrip("/")
    if not base:
        raise ValueError(
            "CONFORMANCE_BASE_PATH must be a writable resource directory for provisioning"
        )
    path = f"{base}/conformance-{uuid.uuid4().hex}.usda"
    os.environ["CONFORMANCE_BASE_PATH"] = path
    settings = Settings.from_env()
    # Sidecar-written token may arrive after this container starts.
    import time

    deadline = time.monotonic() + settings.ready_timeout
    while True:
        try:
            settings.metadata()
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
    url = endpoint + "/v1alpha/fileobject/by-address/" + urllib.parse.quote(path, safe="")
    context = ssl.create_default_context(cafile=settings.ca_file or None)
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=context))

    def request(method, body=None):
        headers = dict(settings.metadata())
        headers["Content-Type"] = "application/octet-stream"
        address = url + (f"?data_object_size={len(body)}" if body is not None else "")
        with opener.open(
            urllib.request.Request(address, data=body, headers=headers, method=method),
            timeout=settings.timeout,
        ) as response:
            if response.status not in (200, 201, 204):
                raise RuntimeError(f"Fixture {method} failed with HTTP {response.status}")

    # Attempt cleanup even if a successful PUT response was lost in transit.
    try:
        request("PUT", b"#usda 1.0\n")
        return run_tests()
    finally:
        try:
            request("DELETE")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise


def main():
    try:
        return run()
    except Exception as exc:
        # Do not print HTTP/RPC exception text, which can include credentials.
        message = f"Provisioning or cleanup failed ({type(exc).__name__})"
        record_failure("deployment-fixture", message)
        print(message, file=sys.stderr)
        return 2
    finally:
        emit_reports()


if __name__ == "__main__":
    sys.exit(main())
