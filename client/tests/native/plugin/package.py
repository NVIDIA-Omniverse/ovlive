#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect the actual native closure (ELF, Mach-O or PE) and load a relocated package in fresh processes."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def dumpbin(option, path):
    """The dumpbin report of one PE file; the MSVC environment provides dumpbin."""
    return subprocess.check_output(['dumpbin', '/nologo', option, str(path)], text=True)


def dependents(path):
    """The DLL names a PE file imports."""
    report = dumpbin('/DEPENDENTS', path)
    return {line.strip().lower() for line in report.splitlines() if line.strip().lower().endswith('.dll')}


# The C runtime and Windows system DLLs a sample may import besides the plugin.
WINDOWS_SYSTEM = re.compile(r'(kernel32|user32|advapi32|ws2_32|bcrypt|crypt32|msvcp140(_\w+)?|vcruntime140(_\w+)?|'
                            r'ucrtbase|api-ms-win-[\w-]+|ntdll|shell32|ole32|dbghelp)\.dll')


def check_pe(package, probe, root, declared):
    """The PE counterpart of the ELF checks: exports, import closure and relocated runs with bin on PATH."""
    library = package / 'bin/omni_usd_live.dll'
    sample = package / 'bin/live_diagnostics_sample.exe'
    participant = package / 'bin/live_participant_sample.exe'
    for path in (library, sample, participant):
        assert path.is_file(), path
    report = dumpbin('/EXPORTS', library)
    exported = set(re.findall(r'^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)', report, re.M))
    assert {name for name in exported if name.startswith('omniUsdLive')} == declared, exported
    # Every other export is the C++ host facade, which MSVC decorates with its namespace.
    assert all(name.startswith('omniUsdLive') or '@host@live@usd@omni@@' in name for name in exported), exported
    for artifact in (library, probe, sample, participant):
        imports = dependents(artifact)
        assert not any('omniclient' in name or 'resolver_runtime_component' in name for name in imports), imports
    assert 'omni_usd_live.dll' not in dependents(probe)
    for executable in (sample, participant):
        assert 'omni_usd_live.dll' in dependents(executable), executable
    # The diagnostics sample is a C ABI consumer with no other dependency; the participant sample also links USD, as
    # on Linux.
    imports = dependents(sample)
    assert all(name == 'omni_usd_live.dll' or WINDOWS_SYSTEM.fullmatch(name) for name in imports), imports
    environment = dict(os.environ, PXR_PLUGINPATH_NAME='', PXR_AR_DEFAULT_SEARCH_PATH='', AR_DEFAULT_RESOLVER='')
    with tempfile.TemporaryDirectory(prefix='live-plugin-relocated-') as temporary:
        copied = Path(temporary).resolve() / 'package'
        shutil.copytree(package, copied)
        meta = copied/'resources/plugInfo.json'
        info = json.loads(meta.read_text())['Plugins'][0]
        assert (meta.parent/info['Root']/info['LibraryPath']).resolve() == (copied/'bin/omni_usd_live.dll').resolve()
        run_environment = dict(environment, PATH=str(copied / 'bin') + os.pathsep + environment.get('PATH', ''))
        help_result = subprocess.run([str(copied / 'bin/live_diagnostics_sample.exe'), '--help'],
                                     env=run_environment, check=True, timeout=20,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert help_result.stdout.startswith('Usage: live_diagnostics_sample '), help_result.stdout
        participant_help = subprocess.run([str(copied / 'bin/live_participant_sample.exe'), '--help'],
                                          env=run_environment, check=True, timeout=20,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert participant_help.stdout.startswith('live_participant_sample --authority ')
        for order in ('api-first', 'plugin-first'):
            subprocess.run([str(probe), str(copied/'bin/omni_usd_live.dll'), str(meta), order],
                           env=environment, check=True, timeout=20)
    print('public exports, sample DLL import closure, relocated help, relative resources and load orders passed')


def symbols(path, *, undefined=False, dynamic=False):
    if sys.platform == 'darwin':
        options = ['-j', '-u' if undefined else '-U']
        if dynamic:
            options.append('-g')
        report = subprocess.check_output(['nm', *options, str(path)], text=True)
        return {name.removeprefix('_') for name in report.splitlines() if name}
    options = ['--undefined-only' if undefined else '--defined-only']
    if dynamic:
        options.append('-D')
    report = subprocess.check_output(['nm', *options, str(path)], text=True)
    return {line.split()[-1].split('@', 1)[0] for line in report.splitlines()}


def imports(path):
    if sys.platform == 'darwin':
        report = subprocess.check_output(['otool', '-L', str(path)], text=True)
        return {line.strip().split(' (compatibility version', 1)[0] for line in report.splitlines()[1:]}
    report = subprocess.check_output(['readelf', '-d', str(path)], text=True)
    return set(re.findall(r'\(NEEDED\).*\[([^\]]+)\]', report))


def search_paths(path):
    if sys.platform == 'darwin':
        report = subprocess.check_output(['otool', '-l', str(path)], text=True)
        return re.findall(r'cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset', report)
    report = subprocess.check_output(['readelf', '-d', str(path)], text=True)
    rows = re.findall(r'\((?:RUNPATH|RPATH)\).*\[([^\]]*)\]', report)
    assert len(rows) == 1, report
    return rows[0].split(':')


def check_dependency_metadata(path):
    """Screen the full ELF dynamic section and Mach-O imports for competing runtimes."""
    if sys.platform == 'darwin':
        report = '\n'.join(sorted(imports(path)))
    else:
        report = subprocess.check_output(['readelf', '-d', str(path)], text=True)
    assert not any(name in report.lower() for name in ('omniclient', 'resolver_runtime_component')), report


def main():
    package, probe, root = (Path(p).resolve() for p in sys.argv[1:])
    sys.path.insert(0, str(root))
    from tools.live import platforms
    if platforms.host().binary_format == 'pe':
        header = (root/'include/omni/usd/live/Live.h').read_text()
        declared = set(re.findall(r'OMNI_USD_LIVE_CALL (omniUsdLive\w+)\(', header))
        return check_pe(package, probe, root, declared)
    # A build.py stage and an unpacked release archive are installed layouts; a CI install tree is not.
    installed = (package / 'package-manifest.json').is_file() or (package.parent / 'release-manifest.json').is_file()
    macos = sys.platform == 'darwin'
    library_name = 'libomni_usd_live.dylib' if macos else 'libomni_usd_live.so'
    library = package / 'lib' / library_name
    origin = '@loader_path' if macos else '$ORIGIN'
    sample = package / 'bin/live_diagnostics_sample'
    assert sample.is_file() and os.access(sample, os.X_OK), sample
    participant = package / 'bin/live_participant_sample'
    assert participant.is_file() and os.access(participant, os.X_OK), participant
    exported = symbols(library, dynamic=True)
    declared = set(re.findall(r'OMNI_USD_LIVE_CALL (omniUsdLive\w+)\(', (root/'include/omni/usd/live/Live.h').read_text()))
    assert {s for s in exported if s.startswith('omniUsdLive')} == declared
    extra = exported - declared
    # symbols() removes the Mach-O prefix; the C++ mangling underscore remains.
    demangled = subprocess.check_output(['c++filt', '-n', *sorted(extra)], text=True).splitlines()
    allowed = ('omni::usd::live::host::setPublicationContext(', 'omni::usd::live::host::save(',
               'omni::usd::live::host::createPublicationContext(',
               'omni::usd::live::host::dispatchingLocalPublication(',
               'omni::usd::live::host::PublicationContext::~PublicationContext(',
               'omni::usd::live::host::PublicationContext::close(',
               'omni::usd::live::host::PublicationContext::prepareCreation(',
               'omni::usd::live::host::PublicationContext::prepareRetainedCreation(',
               'omni::usd::live::host::prepareCreation(',
               'omni::usd::live::host::prepareRetainedCreation(',
               'omni::usd::live::host::nextRetainedCreation(',
               'omni::usd::live::host::acquireRetainedCreation(',
               'omni::usd::live::host::releaseRetainedCreation(',
               'omni::usd::live::host::Creation::~Creation(',
               'omni::usd::live::host::Creation::start(',
               'omni::usd::live::host::Creation::wait(',
               'omni::usd::live::host::Creation::waitForClose(',
               'omni::usd::live::host::Creation::getOutcome(',
               'omni::usd::live::host::Creation::prepareLayer(',
               'omni::usd::live::host::Creation::activate(',
               'omni::usd::live::host::Creation::getLayer(',
               'omni::usd::live::host::Creation::close(')
    assert demangled and all(name.startswith(allowed) for name in demangled), demangled
    assert all(any(name.startswith(prefix) for name in demangled) for prefix in allowed), demangled
    for artifact in (library, probe, sample, participant):
        check_dependency_metadata(artifact)
        dependencies = imports(artifact)
        if macos and artifact in (library, participant):
            assert '@rpath/libpython3.12.dylib' in dependencies, dependencies
            assert not any('Python.framework/' in name for name in dependencies), dependencies
    # Probe is deliberately not linked to the plugin or any private runtime.
    assert not any('libomni_usd_live' in name for name in imports(probe))
    # The diagnostics executable consumes the shared C ABI and imports only the plugin and system libraries.
    needed = imports(sample)
    plugin_import = '@rpath/' + library_name if macos else library_name
    system_libraries = ({'/usr/lib/libc++.1.dylib', '/usr/lib/libSystem.B.dylib'} if macos else
                        {'libstdc++.so.6', 'libgcc_s.so.1', 'libc.so.6', 'libm.so.6',
                         'libpthread.so.0', 'libdl.so.2', 'librt.so.1', platforms.host().dynamic_loader})
    sanitizer_libraries = {name for name in needed if re.fullmatch(r'lib(?:asan|ubsan|tsan)\.so\.\d+', name)}
    assert plugin_import in needed, needed
    assert not needed - system_libraries - sanitizer_libraries - {plugin_import}, needed
    defined = symbols(sample)
    assert defined, 'sample symbol table unavailable for runtime-duplication audit'
    assert not any(name.startswith('omniUsdLive') for name in defined), defined
    undefined = symbols(sample, undefined=True, dynamic=True)
    imported_api = {name for name in undefined if name.startswith('omniUsdLive')}
    assert {'omniUsdLiveInitialize', 'omniUsdLiveCopyDiagnostics',
            'omniUsdLiveRegisterDiagnosticCallback', 'omniUsdLiveRegisterStatusCallback'} <= imported_api
    assert imported_api <= declared, imported_api - declared
    assert plugin_import in imports(participant)
    participant_defined = symbols(participant)
    assert participant_defined
    assert not any(name.startswith('omniUsdLive') for name in participant_defined)
    participant_paths = search_paths(participant)
    assert participant_paths[0] == origin + '/../lib', participant_paths
    if installed:
        assert participant_paths == [origin + '/../lib', origin + '/../lib/omni_usd_live/lib']
    cpp_imports = subprocess.check_output(['c++filt', '-n', *sorted(undefined)], text=True)
    assert 'omni::usd::live::' not in cpp_imports, cpp_imports
    sample_paths = search_paths(sample)
    assert sample_paths[0] == origin + '/../lib', sample_paths
    if installed:
        assert sample_paths == [origin + '/../lib', origin + '/../lib/omni_usd_live/lib'], sample_paths
    environment = dict(os.environ, PXR_PLUGINPATH_NAME='', PXR_AR_DEFAULT_SEARCH_PATH='', AR_DEFAULT_RESOLVER='')
    with tempfile.TemporaryDirectory(prefix='live-plugin-relocated-') as temporary:
        copied = Path(temporary).resolve() / 'package'
        shutil.copytree(package, copied)
        for name in ('Live.h', 'LiveUsd.h'):
            assert (copied/'include/omni/usd/live'/name).read_bytes() == (root/'include/omni/usd/live'/name).read_bytes()
        meta = copied/'resources/plugInfo.json'
        info = json.loads(meta.read_text())['Plugins'][0]
        assert (meta.parent/info['Root']/info['LibraryPath']).resolve() == copied/'lib'/library_name
        sample_environment = dict(environment)
        library_path = 'DYLD_LIBRARY_PATH' if macos else 'LD_LIBRARY_PATH'
        sdk_paths = sample_environment.get(library_path, '')
        sample_environment[library_path] = str(copied / 'lib') + (':' + sdk_paths if sdk_paths else '')
        help_result = subprocess.run([str(copied / 'bin/live_diagnostics_sample'), '--help'],
                                    env=sample_environment, check=True, timeout=20,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert help_result.stdout.startswith('Usage: live_diagnostics_sample '), help_result.stdout
        assert '--format jsonl|text' in help_result.stdout and '--snapshot' in help_result.stdout, help_result.stdout
        assert not help_result.stderr, help_result.stderr
        participant_help = subprocess.run([str(copied / 'bin/live_participant_sample'), '--help'],
                                         env=sample_environment, check=True, timeout=20,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert participant_help.stdout.startswith('live_participant_sample --authority ')
        assert '--plug-info' in participant_help.stdout and not participant_help.stderr
        for order in ('api-first', 'plugin-first'):
            subprocess.run([str(probe), str(copied/'lib'/library_name), str(meta), order],
                           env=environment, check=True, timeout=20)
    print('public exports, sample shared ABI closure/relative RPATH/relocated help, relative resources and load orders passed')


if __name__ == '__main__':
    main()
