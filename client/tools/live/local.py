#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parameter-free build, test and clean on a machine whose worker configuration is installed.

`build.sh`, `test.sh` and `clean.sh` in the client root run this entry point. It reads the same
worker configuration as tools/live/ci.py from OVLIVE_WORKER_CONFIG or `$XDG_CONFIG_HOME/ovlive/worker.json`
(`$HOME/.config/ovlive/worker.json` by default), runs the presubmit profile's configure, build and contracts
steps in the reusable `.cache/local` directory of the repository root, and removes that directory on clean. It
downloads nothing, discovers no SDK and never touches `.cache/ci`, whose runs stay fresh per invocation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import ci

CONFIG_ENV = 'OVLIVE_WORKER_CONFIG'
OUTPUT = ROOT.parent / '.cache/local'
PROFILE = 'presubmit'
SEED = 42
STEPS = {'build': ('configure', 'build'), 'test': ('contracts',)}


def default_config_path(environ=os.environ):
    base = environ.get('XDG_CONFIG_HOME') or str(Path.home() / '.config')
    return Path(base) / 'ovlive' / 'worker.json'


def config_path(environ=os.environ):
    return Path(environ[CONFIG_ENV]) if environ.get(CONFIG_ENV) else default_config_path(environ)


def parallel(cpu_count=os.cpu_count):
    return max(1, min(8, cpu_count() or 2))


def steps(config, command, output=OUTPUT, workers=None):
    """The (name, command) CI steps one local command runs, planned against the reusable output directory."""
    record = ci.plan(config, output, PROFILE, SEED, workers or parallel(), fresh=False)
    wanted = STEPS[command]
    return [(step['name'], step['command']) for step in record['steps'] if step['name'] in wanted]


def run_steps(planned, environment, runner=subprocess.run):
    for name, command in planned:
        print('Step: ' + name, flush=True)
        code = runner(command, cwd=ROOT, env=environment).returncode
        if code != 0:
            print('Step failed: ' + name, file=sys.stderr, flush=True)
            return code
    return 0


def clean(output=OUTPUT):
    if output.is_symlink() or output.exists():
        shutil.rmtree(output)
    return 0


def main(argv=None, environ=os.environ, runner=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'test', 'clean'))
    args = parser.parse_args(argv)
    if args.command == 'clean':
        return clean(OUTPUT)
    path = config_path(environ)
    if not path.is_file():
        parser.error(f'No worker configuration at {path}; install one as docs/ovlive-ci.md describes '
                     f'or name it in {CONFIG_ENV}')
    if args.command == 'test' and not (OUTPUT / 'build').is_dir():
        parser.error('No local build; run ./build.sh first')
    try:
        planned = steps(ci.read_config(path), args.command, output=OUTPUT)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    return run_steps(planned, ci.clean_environment(), runner)


if __name__ == '__main__':
    raise SystemExit(main())
