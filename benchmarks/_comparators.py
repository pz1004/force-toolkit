"""Identify and load three isolated implementation comparators."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import tomllib
import zipfile

VARIANTS = ('baseline', 'optimized_010', 'accelerated')
WHEEL_RELATIVE = 'benchmarks/artifacts/force_toolkit-0.1.0-py3-none-any.whl'
WHEEL_SHA256 = '1e4827136c856931e0202b4f29a0124c51797d17fbd99cce3b6af32a161fc1b6'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tree_hash(files):
    return digest(json.dumps(files, sort_keys=True, separators=(',', ':')).encode())


def python_files(directory):
    return {p.relative_to(directory).as_posix(): digest(p.read_bytes())
            for p in sorted(Path(directory).rglob('*.py'))}


def wheel_files(path):
    from benchmarks._protocol import EvidenceError
    if digest(Path(path).read_bytes()) != WHEEL_SHA256:
        raise EvidenceError('Optimized 0.1.0 wheel differs from its preserved release checksum')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise EvidenceError('Duplicate prior-wheel members')
        result = {}
        for name in names:
            parts = PurePosixPath(name).parts
            if (not parts or name.startswith('/') or '..' in parts or '\\' in name
                    or ':' in name or not name.startswith(('force_toolkit/', 'force_toolkit-0.1.0.dist-info/'))):
                raise EvidenceError('Unsafe prior-wheel member')
            if not name.endswith('/'):
                result[name] = digest(archive.read(name))
    return result


def comparator_identities(root):
    from benchmarks._protocol import EvidenceError
    root = Path(root)
    with (root / 'pyproject.toml').open('rb') as stream:
        version = tomllib.load(stream)['project']['version']
    result = {}
    for variant, relative, declared_version in (
            ('baseline', 'tests/reference/force_toolkit_baseline', '0.1.0'),
            ('accelerated', 'src/force_toolkit', version)):
        files = python_files(root / relative)
        if '__init__.py' not in files:
            raise EvidenceError(f'Missing {variant} package')
        result[variant] = dict(kind='source_tree', version=declared_version,
                               relative_package=relative, python_sha256=files,
                               source_tree_sha256=tree_hash(files))
    members = wheel_files(root / WHEEL_RELATIVE)
    files = {name.removeprefix('force_toolkit/'): value for name, value in members.items()
             if name.startswith('force_toolkit/') and name.endswith('.py')}
    result['optimized_010'] = dict(kind='preserved_wheel', version='0.1.0',
                                  relative_wheel=WHEEL_RELATIVE, wheel_sha256=WHEEL_SHA256,
                                  wheel_members_sha256=members, python_sha256=files,
                                  source_tree_sha256=tree_hash(files))
    return result


def prior_package(directory):
    return Path(directory).resolve() / '.comparator-cache' / WHEEL_SHA256 / 'force_toolkit'


def ensure_prior_extracted(directory, manifest):
    """Extract once into a stable path; never trust existing extracted code."""
    from benchmarks._protocol import EvidenceError
    artifact = Path(manifest['project_root']) / WHEEL_RELATIVE
    expected = manifest['comparators']['optimized_010']['wheel_members_sha256']
    if wheel_files(artifact) != expected:
        raise EvidenceError('Prior-wheel manifest members differ')
    destination = prior_package(directory).parent
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.extract-', dir=destination.parent))
        try:
            with zipfile.ZipFile(artifact) as archive:
                archive.extractall(temporary)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    actual = {p.relative_to(destination).as_posix(): digest(p.read_bytes())
              for p in destination.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.nbc', '.nbi')}
    if actual != expected:
        raise EvidenceError('Extracted prior-wheel code or metadata changed')
    return destination / 'force_toolkit'


def expected_package(directory, manifest, variant):
    if variant == 'optimized_010':
        return prior_package(directory)
    return Path(manifest['project_root']) / manifest['comparators'][variant]['relative_package']


def runtime_identity(module, directory, manifest, variant):
    """Verify the imported origins and bytes, including already loaded submodules."""
    from benchmarks._protocol import EvidenceError
    expected = manifest['comparators'][variant]
    package = expected_package(directory, manifest, variant).resolve()
    if Path(module.__file__).resolve() != package / '__init__.py':
        raise EvidenceError(f'{variant} imported from the wrong implementation path')
    if module.__version__ != expected['version']:
        raise EvidenceError(f'{variant} imported the wrong runtime version')
    files = python_files(package)
    if files != expected['python_sha256']:
        raise EvidenceError(f'{variant} imported modified implementation bytes')
    loaded = {}
    for name, value in list(sys.modules.items()):
        if name == module.__name__ or name.startswith(module.__name__ + '.'):
            path = getattr(value, '__file__', None)
            if path is None:
                continue
            path = Path(path).resolve()
            if not path.is_relative_to(package) or path.suffix != '.py':
                raise EvidenceError(f'{variant} submodule origin escaped its package: {name}')
            relative = path.relative_to(package).as_posix()
            value_hash = digest(path.read_bytes())
            if files.get(relative) != value_hash:
                raise EvidenceError(f'{variant} submodule source mismatch: {name}')
            loaded[name] = dict(path=str(path), sha256=value_hash)
    return dict(variant=variant, kind=expected['kind'], version=module.__version__,
                package_root=str(package), entrypoint=str(package / '__init__.py'),
                source_tree_sha256=tree_hash(files), loaded_modules=loaded)


def import_comparator(directory, manifest, variant):
    from benchmarks._protocol import EvidenceError
    if variant not in VARIANTS:
        raise EvidenceError('Unknown implementation comparator')
    if variant == 'baseline':
        sys.path.insert(0, manifest['project_root'])
        module_name = 'tests.reference.force_toolkit_baseline'
    else:
        if variant == 'optimized_010':
            package = ensure_prior_extracted(directory, manifest)
        else:
            package = expected_package(directory, manifest, variant)
        sys.path.insert(0, str(package.parent))
        module_name = 'force_toolkit'
    if module_name in sys.modules:
        raise EvidenceError('Comparator import must occur in a fresh process')
    module = importlib.import_module(module_name)
    return module, runtime_identity(module, directory, manifest, variant)
