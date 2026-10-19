"""
Tests for SchNet Continuous-Filter Convolutions & RBF Distance Expansion
"""

import unittest
import numpy as np
from chem_scientist.schnet import GaussianRBFExpansion, SchNet3DEmbedder

class TestSchNet(unittest.TestCase):
    def test_rbf_expansion_dimensions(self):
        rbf = GaussianRBFExpansion(d_min=0.0, d_max=5.0, n_gaussians=20)
        dists = np.array([1.0, 2.5, 4.0])
        expanded = rbf.expand(dists)
        self.assertEqual(expanded.shape, (3, 20))
        # RBF values must be in [0, 1]
        self.assertTrue(np.all(expanded >= 0.0))
        self.assertTrue(np.all(expanded <= 1.0))

    def test_schnet_embedder_geometry(self):
        embedder = SchNet3DEmbedder(embed_dim=32, n_rbf=15)
        res = embedder.compute_energy_and_forces_proxy("CCO") # Ethanol
        self.assertNotIn("error", res)
        self.assertGreater(res["num_atoms"], 0)
        self.assertGreater(res["max_interatomic_distance_A"], 0.0)
        self.assertGreater(res["mean_bond_distance_A"], 0.0)
        self.assertIsInstance(res["predicted_3d_potential_energy_eV"], float)

if __name__ == "__main__":
    unittest.main()
