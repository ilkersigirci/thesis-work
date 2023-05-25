from dagster import load_assets_from_package_module

from . import chemberta

chemberta_assets = load_assets_from_package_module(
    package_module=chemberta, key_prefix="chemberta", group_name="ChemBERTaGroup"
)
