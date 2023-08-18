"""Thesis Work package."""
import pkg_resources  # type: ignore

# Fetches the version of the package as defined in pyproject.toml
__version__ = pkg_resources.get_distribution("thesis_work").version


from pathlib import Path

from thesis_work.utils.utils import ignore_warnings, initialize_logger

initialize_logger()
ignore_warnings()

LIBRARY_ROOT_PATH = Path(__file__).parent.parent
