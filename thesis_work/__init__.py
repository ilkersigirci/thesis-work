"""Thesis Work package."""
import pkg_resources  # type: ignore

# Fetches the version of the package as defined in pyproject.toml
__version__ = pkg_resources.get_distribution("thesis_work").version
