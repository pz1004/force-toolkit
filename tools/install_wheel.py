"""Portable wheel-only, no-extra install of the exact pyproject release for CI."""
import subprocess
import sys

if __package__:
    from .check_distribution import select_artifact
else:
    from check_distribution import select_artifact


def main():
    wheel = select_artifact('wheel')
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--force-reinstall', str(wheel),
        'numpy==2.3.5', 'numba==0.65.1',
    ], check=True)


if __name__ == '__main__':
    main()
