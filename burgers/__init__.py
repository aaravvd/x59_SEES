"""Beta CFD-signature propagation and perceived-level toolkit."""

from .augmented_burgers import AugmentedBurgersSolver, SolverResult, SolverSettings
from .core import SourceMetadata, load_and_construct_source
from .pipeline import AuditedPropagationConfig, propagate_audited_beta
from .pldb import (
    PLdBResult,
    stevens_mark_vii_pldb,
    stevens_mark_vii_pldb_nasa_2025,
)

__all__ = [
    "AuditedPropagationConfig",
    "AugmentedBurgersSolver",
    "PLdBResult",
    "SolverResult",
    "SolverSettings",
    "SourceMetadata",
    "load_and_construct_source",
    "propagate_audited_beta",
    "stevens_mark_vii_pldb",
    "stevens_mark_vii_pldb_nasa_2025",
]

__version__ = "0.1.0"
