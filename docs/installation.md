# Installation

Use CPython 3.11–3.13. From an empty directory, clone version 0.2.1, activate a virtual environment and run the examples.

```sh
git clone --branch v0.2.1 --depth 1 https://github.com/pz1004/force-toolkit.git
cd force-toolkit
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/dependence_diagnostics.py
python examples/common_input.py
python tools/report_020.py
```

On Windows PowerShell activate with `.\.venv\Scripts\Activate.ps1`. For a downloaded wheel, install `python -m pip install ./force_toolkit-0.2.1-py3-none-any.whl`, extract `force_toolkit-0.2.1.tar.gz` and run the same scripts from its root. For a checkout-free example, see [Using the wheel](../README.md#using-the-wheel). [Release downloads](https://github.com/pz1004/force-toolkit/releases/tag/v0.2.1)

Confirm the interpreter is using version 0.2.1 with `python -c "import force_toolkit as ft; print(ft.__version__, ft.__file__)"`.

NumPy and Numba supply numerical arrays and compiled kernels. Numba installs llvmlite and can reuse its disk cache after first use. SciPy supplies an independent Gaussian quantile reference in the tests. Use the optional test and benchmark extras for development, or the locked dependency list for the reported scientific environment. [Version constraints](../pyproject.toml) · [Tested dependency pins](../benchmarks/requirements-lock.txt)

```sh
python -m pip install '.[test,benchmark]'
python -m build
python tools/install_wheel.py
python tools/check_distribution.py --test-sdist
```

The default build produces a source archive and then builds its wheel from the extracted archive. The checker runs the archive’s examples and tests against the installed distribution in a temporary directory. [CI matrix](../.github/workflows/ci.yml)
