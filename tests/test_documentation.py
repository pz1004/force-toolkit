"""Documented Python snippets and example files must remain executable."""
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[1]


def test_readme_examples():
    for block in re.findall(r'```python\n(.*?)```', (ROOT/'README.md').read_text(), re.S):
        exec(compile(block, 'README.md', 'exec'), {})


def test_complete_example():
    runpy.run_path(str(ROOT/'examples/all_estimators.py'), run_name='__main__')


def test_dependence_diagnostics_example():
    runpy.run_path(str(ROOT / 'examples/dependence_diagnostics.py'), run_name='__main__')


def test_common_input_example():
    runpy.run_path(str(ROOT / 'examples/common_input.py'), run_name='__main__')
