"""
Command-Line Interface for ChemScientist AI
"""

import argparse
import sys
import numpy as np
from .schnet import SchNet3DEmbedder
from .conformal import ConformalMolecularEngine

def main():
    parser = argparse.ArgumentParser(
        prog="chem-scientist",
        description="ChemScientist AI: SE(3)-Equivariant Continuous-Filter SchNet & Conformal Molecular Intelligence"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    pred_parser = subparsers.add_parser("predict", help="Predict 3D molecular potential energy with conformal UQ")
    pred_parser.add_argument("--smiles", type=str, default="CC(=O)Oc1ccccc1C(=O)O", help="SMILES string (e.g., Aspirin)")
    pred_parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")

    args = parser.parse_args()

    if args.command == "predict":
        embedder = SchNet3DEmbedder(embed_dim=args.embed_dim)
        res = embedder.compute_energy_and_forces_proxy(args.smiles)
        
        latent = np.array(res["latent_vector"])
        uq_engine = ConformalMolecularEngine()
        interval = uq_engine.predict_conformal_interval(
            point_prediction=res["predicted_3d_potential_energy_eV"],
            latent_features=latent
        )
        
        print("\n--- ChemScientist AI SchNet Prediction ---")
        print(f"SMILES:               {args.smiles}")
        print(f"Atoms:                {res['num_atoms']}")
        print(f"Max Interatomic Dist: {res['max_interatomic_distance_A']} Å")
        print(f"Mean Bond Dist:       {res['mean_bond_distance_A']} Å")
        print(f"Predicted 3D Energy:  {res['predicted_3d_potential_energy_eV']} eV")
        print(f"90% Conformal Bounds: [{interval.lower_bound_90}, {interval.upper_bound_90}] eV")
        print(f"95% Conformal Bounds: [{interval.lower_bound_95}, {interval.upper_bound_95}] eV")
        print(f"Mahalanobis Distance: {interval.mahalanobis_distance}")
        print(f"Applicability Domain: {'INSIDE' if interval.inside_applicability_domain else 'OOD (Out-of-Domain)'}\n")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
