"""Product Configurator services — the domain's business rules.

Layering, per CLAUDE.md:

    routes  ->  services (here)  ->  app/database/configurator_repo.py  ->  models

Services own transaction boundaries and every rule; repositories only read and
write. Nothing here should be imported by a repository.

  material_service  resolves a product's current GLB into glb_version, material
                    count and per-material method suggestions
  part_service      Product Part rules: ownership, material validation, sibling
                    overlap under a product row lock, glb_version pinning
  option_service    Part Option rules: ownership via option -> part -> product,
                    recipe semantics, image_url namespace check, recipe_hash,
                    default-option rules

Baking is not here. It arrives in a later phase behind ``bake_runner.enqueue()``
so its execution backend can move to a worker without touching this layer.

Import the singletons from their own modules, not from this package::

    from app.services.configurator.part_service import part_service
    from app.services.configurator.option_service import option_service

Re-exporting them here would bind the name ``part_service`` on this package to
the INSTANCE, shadowing the submodule of the same name — so
``app.services.configurator.part_service`` would stop resolving to the module
and any ``monkeypatch.setattr`` or ``importlib`` reference against it would
break. Only classes and pure helpers are re-exported.
"""

from app.services.configurator.material_service import MaterialService, MeshContext
from app.services.configurator.option_service import OptionService
from app.services.configurator.part_service import PartService
from app.services.configurator.recipe import (
    CONFIGURATOR_BAKER_VERSION,
    compute_recipe_hash,
    validate_image_url_ownership,
)

__all__ = [
    "CONFIGURATOR_BAKER_VERSION",
    "MaterialService",
    "MeshContext",
    "OptionService",
    "PartService",
    "compute_recipe_hash",
    "validate_image_url_ownership",
]
