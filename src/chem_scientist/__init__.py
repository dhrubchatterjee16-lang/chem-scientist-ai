"""
ChemScientist AI: SE(3)-Equivariant Continuous-Filter SchNet & Conformal Molecular Intelligence
"""

from .schnet import SchNet3DEmbedder, GaussianRBFExpansion
from .conformal import (
    ConformalPredictionInterval,
    ApplicabilityDomainDetector,
    ConformalMolecularEngine
)

__version__ = "1.0.0"
__all__ = [
    "SchNet3DEmbedder",
    "GaussianRBFExpansion",
    "ConformalPredictionInterval",
    "ApplicabilityDomainDetector",
    "ConformalMolecularEngine",
    "__version__"
]
