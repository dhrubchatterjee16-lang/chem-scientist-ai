"""
FastAPI Microservice for ChemScientist AI
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .schnet import SchNet3DEmbedder
from .conformal import ConformalMolecularEngine

if HAS_FASTAPI:
    app = FastAPI(
        title="ChemScientist AI API",
        description="SE(3)-Equivariant Continuous-Filter SchNet & Conformal UQ Service",
        version="1.0.0"
    )

    class MolecularRequest(BaseModel):
        smiles: str = Field(..., description="SMILES chemical representation")
        embed_dim: int = Field(64, ge=16, le=512)

    @app.get("/")
    def root():
        return {
            "service": "ChemScientist AI",
            "version": "1.0.0",
            "status": "operational"
        }

    @app.post("/schnet/predict")
    def api_predict(req: MolecularRequest):
        embedder = SchNet3DEmbedder(embed_dim=req.embed_dim)
        res = embedder.compute_energy_and_forces_proxy(req.smiles)
        if "error" in res:
            raise HTTPException(status_code=400, detail=res["error"])

        latent = np.array(res["latent_vector"])
        uq = ConformalMolecularEngine().predict_conformal_interval(
            point_prediction=res["predicted_3d_potential_energy_eV"],
            latent_features=latent
        )

        return {
            "smiles": req.smiles,
            "num_atoms": res["num_atoms"],
            "predicted_energy_eV": res["predicted_3d_potential_energy_eV"],
            "interatomic_geometry": {
                "max_distance_A": res["max_interatomic_distance_A"],
                "mean_bond_distance_A": res["mean_bond_distance_A"]
            },
            "conformal_uq": {
                "bounds_90": [uq.lower_bound_90, uq.upper_bound_90],
                "bounds_95": [uq.lower_bound_95, uq.upper_bound_95],
                "margin_90": uq.margin_90,
                "margin_95": uq.margin_95
            },
            "applicability_domain": {
                "mahalanobis_distance": uq.mahalanobis_distance,
                "is_inside": uq.inside_applicability_domain
            }
        }
else:
    app = None
