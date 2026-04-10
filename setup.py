"""Minimal setup.py shim to install the .pth bootstrap file.

The main package metadata lives in pyproject.toml (Poetry).
This shim exists solely to place ants_platform_crewai.pth into
site-packages root, which Poetry's build backend cannot do.

When building with `pip install .`, setuptools reads [project] from
pyproject.toml for metadata and uses this setup.py for data_files.
"""
from setuptools import setup

setup(
    data_files=[(".", ["ants_platform_crewai.pth"])],
)
