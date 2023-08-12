"""Thesis Work package."""
from pathlib import Path

import pkg_resources  # type: ignore

# Fetches the version of the package as defined in pyproject.toml
__version__ = pkg_resources.get_distribution("thesis_work").version

from .utils import ignore_warnings

ignore_warnings()

LIBRARY_ROOT_PATH = Path(__file__).parent.parent
