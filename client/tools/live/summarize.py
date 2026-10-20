# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print the evidence a qualification record needs from one CI or test run.

The directory is either a `tools/live/ci.py` output directory (it holds
`summary.json` and a `tests` directory) or a `tools/live/test.py` artifact
directory (it holds `run-*` directories). The tool only reads; it never
changes the directory.

For a CI output directory it prints the CI result, CI profile, USD profile,
master seed, source fingerprint, checkout state, release archive name and
SHA-256 when the profile builds one, and the SHA-256 of
`support-identity.json`. For every `run-*` directory, in name order, which is
not chronological, it prints the run result, the Git head and source
fingerprint the run recorded, the master and writer seeds, one line per suite
with its outcome and test count (`-` when the suite reports none), and, when
the run holds a writer campaign, its result, case count and campaign seeds. A
summary or manifest that is missing or unreadable prints one line naming it
and counts as an incomplete result; a run interrupted before its first
manifest leaves such a directory.

The exit status and the fresh-directory rule are in `EPILOG`, which `--help`
prints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

RAN = re.compile(r'^Ran (\d+) tests? in ', re.M)
EPILOG = ('Exit status: 0 when every result passed and completed; 1 when any did not pass or is incomplete; '
          '3 when the directory holds no CI summary and no run-* directory; 2 for a usage error. '
          'Use a fresh artifact directory for each recorded test.py run: an artifact directory keeps '
          'earlier runs, and every run it holds counts toward the exit status.')


def _read_json(path):
    """Return the JSON object at `path`, or None when it is missing or unreadable."""
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _usd_profile(directory):
    plan = _read_json(directory / 'plan.json') or {}
    build_plan = plan.get('build_plan')
    if isinstance(build_plan, dict) and build_plan.get('usd_profile'):
        return build_plan['usd_profile']
    support = _read_json(directory / 'support-identity.json') or {}
    return support.get('usd_profile', '-')


def _passed(record):
    return record.get('passed') is True and record.get('complete') is True


def _tests_run(run, index, result):
    if 'tests_run' in result:
        return str(result['tests_run'])
    log = run / f'{index}.log'
    if log.is_file():
        counts = RAN.findall(log.read_text(encoding='utf-8', errors='replace'))
        if counts:
            return counts[-1]
    return '-'


def summarize(directory):
    """Return the evidence lines for `directory` and whether every result passed."""
    directory = Path(directory)
    lines = []
    passed = True
    found = False
    summary_path = directory / 'summary.json'
    if summary_path.is_file():
        found = True
        summary = _read_json(summary_path)
        if summary is None:
            passed = False
            lines.append('ci summary.json unreadable')
            summary = {}
        passed = passed and _passed(summary)
        lines.append(f"ci passed {summary.get('passed')} complete {summary.get('complete')} "
                     f"reason {summary.get('reason', '-')}")
        lines.append(f"ci profile {summary.get('profile')} usd_profile {_usd_profile(directory)} seed {summary.get('seed')} "
                     f"fingerprint {summary.get('source_fingerprint')}")
        checkout = summary.get('checkout')
        if isinstance(checkout, dict):
            lines.append(f"checkout {checkout.get('state')} changed_files {checkout.get('changed_files')}")
        archive = summary.get('release_archive')
        if isinstance(archive, dict):
            lines.append(f"release archive {archive.get('name')} sha256 {archive.get('sha256')} "
                         f"git_head {archive.get('git_head')} dirty {archive.get('dirty')}")
    tests = directory / 'tests' if (directory / 'tests').is_dir() else directory
    runs = sorted((path for path in tests.iterdir() if path.is_dir() and path.name.startswith('run-')),
                  key=lambda path: path.name)
    for run in runs:
        found = True
        manifest = _read_json(run / 'manifest.json')
        if manifest is None:
            passed = False
            lines.append(f"run {run.name} manifest missing or unreadable")
            continue
        passed = passed and _passed(manifest)
        lines.append(f"run {run.name} passed {manifest.get('passed')} complete {manifest.get('complete')} "
                     f"git_head {manifest.get('git_head')} fingerprint {manifest.get('source_fingerprint')}")
        seed = manifest.get('seed')
        if isinstance(seed, dict):
            writer_seeds = ' '.join(str(value) for value in seed.get('writer_seeds', [])) or '-'
            lines.append(f"  seed {seed.get('master_seed')} writer_seeds {writer_seeds}")
        for index, result in enumerate(manifest.get('results', [])):
            lines.append(f"  suite {result.get('suite')} {result.get('outcome')} tests {_tests_run(run, index, result)}")
        campaign_path = run / 'writer-fuzz' / 'summary.json'
        if campaign_path.is_file():
            campaign = _read_json(campaign_path)
            if campaign is None:
                passed = False
                lines.append('  writer-fuzz summary.json unreadable')
                continue
            passed = passed and campaign.get('passed') is True
            plan = campaign.get('plan', {})
            seeds = ' '.join(str(value) for value in plan.get('seeds', [])) or '-'
            lines.append(f"  writer-fuzz passed {campaign.get('passed')} cases {len(plan.get('cases', []))} "
                         f"results {len(campaign.get('results', []))} seeds {seeds}")
    support = directory / 'support-identity.json'
    if support.is_file():
        lines.append(f"support identity sha256 {hashlib.sha256(support.read_bytes()).hexdigest()}")
    if not found:
        return lines, None
    return lines, passed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n', 1)[0], epilog=EPILOG)
    parser.add_argument('directory', type=Path,
                        help='a tools/live/ci.py --output directory or a tools/live/test.py --artifact-dir directory')
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        parser.error(f'not a directory: {args.directory}')
    lines, passed = summarize(args.directory)
    for line in lines:
        print(line)
    if passed is None:
        print(f'no CI summary or run-* directory in {args.directory}', file=sys.stderr)
        return 3
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
