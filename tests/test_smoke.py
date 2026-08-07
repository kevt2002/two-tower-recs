"""Smoke tests: the package installs and imports."""

import two_tower


def test_version_is_set() -> None:
    assert isinstance(two_tower.__version__, str)
    assert two_tower.__version__
