"""Only explicit release assets belong in the download directory."""
import hashlib

import pytest

from tools.check_distribution import stage_release


def test_staging_selects_only_release_assets_and_hashes_the_copies(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "force-toolkit"\nversion = "0.2.1"\n')
    dist = tmp_path / "dist"
    dist.mkdir()
    names = ["force_toolkit-0.2.1-py3-none-any.whl", "force_toolkit-0.2.1.tar.gz",
             "force-toolkit-evidence-0.2.1.zip"]
    output = tmp_path / "release"
    for name in [*names[:2], "scratch.txt", "SHA256SUMS"]:
        (dist / name).write_bytes(name.encode())
    with pytest.raises(FileNotFoundError):
        stage_release(output, tmp_path)
    assert not output.exists()
    (dist / names[2]).write_bytes(names[2].encode())
    stage_release(output, tmp_path)
    assert {p.name for p in output.iterdir()} == {*names, "SHA256SUMS"}
    for name in names:
        assert (output / name).read_bytes() == (dist / name).read_bytes()
    assert (output / "SHA256SUMS").read_text().splitlines() == [
        f"{hashlib.sha256((dist / name).read_bytes()).hexdigest()}  {name}" for name in names
    ]
    with pytest.raises(RuntimeError, match="already exists"):
        stage_release(output, tmp_path)
