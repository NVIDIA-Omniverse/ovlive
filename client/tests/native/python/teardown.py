# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interpreter exit while a Python auth callback never returns (design section 19)."""
import json
import logging
import sys
import threading

from omni.usd import live


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(name)s %(levelname)s %(message)s")
    live._teardown_watchdog_s = float(sys.argv[2])
    runtime = live.initialize()
    endpoint = runtime.register_endpoint(sys.argv[1], access_token="")  # Missing token: challenge at once.
    entered = threading.Event()

    def never_returns(event):
        entered.set()
        threading.Event().wait()

    live.register_auth_required_callback(endpoint, never_returns)
    if not entered.wait(10):  # Failure-only watchdog.
        raise SystemExit("auth challenge was not delivered")
    print(json.dumps(dict(type="blocked")), flush=True)
    live._teardown()  # atexit runs it again; teardown is idempotent.
    print(json.dumps(dict(type="native", runtime=int(runtime._native.close(0)),
                          endpoint=int(endpoint._native.close(0)))), flush=True)


if __name__ == "__main__":
    main()
