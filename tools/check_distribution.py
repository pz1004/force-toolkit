"""Check current release artifacts and optionally test the extracted sdist."""
from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ("all_estimators.py", "dependence_diagnostics.py", "common_input.py")
NOTICES = ("LICENSE", "Licence.txt", "licenses/force.txt", "licenses/mad-force.txt", "licenses/dual-bound-force.txt")


def public_files(root):
    """MANIFEST.in explicitly enumerates the reviewed public Git files."""
    entries = set()
    for line in (root / "MANIFEST.in").read_text().splitlines():
        if line.startswith("#") or not line:
            continue
        directive, name = line.split()
        require(directive == "include" and name not in entries, "Expected a unique explicit include")
        entries.add(name)
    check_paths(entries)
    return entries


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def project_metadata(root=ROOT):
    with (Path(root) / 'pyproject.toml').open('rb') as stream:
        project = tomllib.load(stream)['project']
    return re.sub(r'[-_.]+', '_', project['name']), project['version']


def select_artifact(kind, root=ROOT):
    """Ignore retained historical releases; reject missing/ambiguous current ones."""
    root = Path(root)
    name, version = project_metadata(root)
    if kind == 'wheel':
        candidates = sorted((root / 'dist').glob(f'{name}-{version}-*.whl'))
    elif kind == 'source':
        candidates = sorted((root / 'dist').glob(f'{name}-{version}.tar.gz'))
    else:
        raise ValueError('kind must be wheel or source')
    require(len(candidates) == 1,
            f'Expected exactly one {kind} for {name} {version}; found {len(candidates)}')
    return candidates[0]


def check_paths(names):
    forbidden = {
        'research', '__pycache__', '.pytest_cache', '.git', '.mypy_cache',
        '.ruff_cache', '.tox', '.nox', 'build', 'dist', 'release-envs', 'cold-cache',
        '.comparator-cache', '.numba-cache',
    }
    for name in names:
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts
                and '\\' not in name and ':' not in name,
                f'Unsafe archive path: {name}')
        require(not forbidden.intersection(path.parts)
                and not any(part.startswith('.venv') or part.endswith('-venv')
                            for part in path.parts), f'Build/environment leakage: {name}')
        require(not name.endswith(('.nbc', '.nbi', '.pyc', '.pyo')),
                f'Compiled cache leakage: {name}')
        require('benchmarks/pilot' not in path.as_posix(), f'Pilot leakage: {name}')


def verify_distribution(root=ROOT):
    root = Path(root)
    public = public_files(root)
    name, version = project_metadata(root)
    wheel, source = select_artifact('wheel', root), select_artifact('source', root)
    dist_info = f'{name}-{version}.dist-info'
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        check_paths(names)
        require(len(names) == len(set(names)), 'Duplicate wheel members')
        for required in ('force_toolkit/api.py', 'force_toolkit/_validity.py',
                         'force_toolkit/_mad/stationary.py', 'force_toolkit/_mad/_controller.py',
                         'force_toolkit/_mad/_state.py', 'force_toolkit/_mad/_detector.py',
                         f'{dist_info}/METADATA'):
            require(required in names, f'Missing wheel file: {required}')
        package_files = {n.removeprefix("src/"): n for n in public if n.startswith("src/force_toolkit/")}
        expected = set(package_files) | {f"{dist_info}/licenses/{n}" for n in NOTICES}
        expected.update(f"{dist_info}/{n}" for n in ("METADATA", "WHEEL", "top_level.txt", "RECORD"))
        require(set(names) == expected, f"Unexpected wheel inventory: {expected.symmetric_difference(names)}")
        for member, path in package_files.items():
            require(archive.read(member) == (root / path).read_bytes(), f"Changed runtime: {member}")
        for notice in NOTICES:
            require(archive.read(f"{dist_info}/licenses/{notice}") == (root / notice).read_bytes(),
                    f"Changed license: {notice}")
        require((root / "LICENSE").read_bytes() == (root / "Licence.txt").read_bytes(), "MIT texts differ")
        metadata = Parser().parsestr(archive.read(f'{dist_info}/METADATA').decode('utf-8'))
        require(metadata['Version'] == version, 'Wheel metadata version mismatch')
        require(re.sub(r'[-_.]+', '_', metadata['Name'] or '') == name,
                'Wheel metadata name mismatch')
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        check_paths(names)
        require(len(names) == len(set(names)), 'Duplicate source members')
        require(all(member.isfile() or member.isdir() for member in members),
                'Source archive contains links or special files')
        require({PurePosixPath(n).parts[0] for n in names} == {f'{name}-{version}'},
                'Source archive root does not match the current release')
        files = {'/'.join(PurePosixPath(m.name).parts[1:]) for m in members if m.isfile()}
        generated = {"PKG-INFO", "setup.cfg"} | {
            f"src/{name}.egg-info/{n}" for n in
            ("PKG-INFO", "SOURCES.txt", "dependency_links.txt", "requires.txt", "top_level.txt")
        }
        expected = public | generated
        require(files == expected, f"Unexpected source inventory: {expected.symmetric_difference(files)}")
        for path in public:
            data = archive.extractfile(f"{name}-{version}/{path}").read()
            require(data == (root / path).read_bytes(), f"Source archive bytes differ: {path}")
        metadata_file = archive.extractfile(f'{name}-{version}/PKG-INFO')
        require(metadata_file is not None, 'Source archive has no package metadata')
        metadata = Parser().parsestr(metadata_file.read().decode('utf-8'))
        require(metadata['Version'] == version, 'Source metadata version mismatch')
        require(re.sub(r'[-_.]+', '_', metadata['Name'] or '') == name,
                'Source metadata name mismatch')
    print(f'Verified {wheel.name} and {source.name}.', flush=True)
    return source


def test_source_archive(source, root=ROOT):
    """Run archive-owned tests/examples against the installed wheel, outside checkout."""
    name, version = project_metadata(root)
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory(prefix='force-toolkit-sdist-') as temporary:
        with tarfile.open(source) as archive:
            # verify_distribution already rejects traversal, links and special files.
            if hasattr(tarfile, 'data_filter'):
                archive.extractall(temporary, filter='data')
            else:  # Early Python 3.11 patch releases lack extraction filters.
                archive.extractall(temporary)
        extracted = Path(temporary) / f'{name}-{version}'
        check_install = (
            'import importlib.metadata as m; from pathlib import Path; '
            'import force_toolkit as f; import sys; '
            "d=m.distribution('force-toolkit'); "
            "expected=Path(d.locate_file('force_toolkit/__init__.py')).resolve(); "
            'assert d.version == sys.argv[1], (d.version, sys.argv[1]); '
            'assert Path(f.__file__).resolve() == expected and expected.is_file(), '
            '"Tests must import an installed wheel, not editable checkout source"; '
            'print("Testing installed wheel:", expected)'
        )
        subprocess.run([sys.executable, '-c', check_install, version],
                       cwd=extracted, env=environment, check=True)
        for example in EXAMPLES:
            subprocess.run([sys.executable, str(Path("examples") / example)],
                           cwd=extracted, env=environment, check=True)
        subprocess.run([sys.executable, "tools/report_020.py"], cwd=extracted, env=environment, check=True)
        subprocess.run([sys.executable, '-m', 'pytest', '-q'],
                       cwd=extracted, env=environment, check=True)


def stage_release(destination, root=ROOT):
    """Assemble the three downloadable artifacts and checksum their staged bytes."""
    name, version = project_metadata(root)
    evidence = Path(root) / 'dist' / f"{name.replace('_', '-')}-evidence-{version}.zip"
    assets = (select_artifact('wheel', root), select_artifact('source', root), evidence)
    destination = Path(destination)
    require(not destination.exists(), f'Release destination already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / 'release'
        staged.mkdir()
        checksums = []
        for asset in assets:
            copy = staged / asset.name
            shutil.copyfile(asset, copy)
            with copy.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            checksums.append(f'{checksum}  {asset.name}\n')
        (staged / 'SHA256SUMS').write_bytes(''.join(checksums).encode('ascii'))
        staged.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-sdist', action='store_true',
                        help='also run extracted tests/examples against the installed wheel')
    parser.add_argument('--stage-release', type=Path, metavar='DIR',
                        help='copy the wheel, source and evidence into a new directory with SHA256SUMS')
    args = parser.parse_args()
    source = verify_distribution()
    if args.test_sdist:
        test_source_archive(source)
    if args.stage_release:
        stage_release(args.stage_release)


if __name__ == '__main__':
    main()
