"""Separate portable CPU tests from local-artifact integration checks."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "tiny_backbone_path" in item.fixturenames:
            item.add_marker(pytest.mark.integration)
