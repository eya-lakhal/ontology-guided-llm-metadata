"""Smoke test for the project skeleton."""

import src


def test_package_importable() -> None:
    assert src.__version__
