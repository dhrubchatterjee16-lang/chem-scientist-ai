# ChemScientist AI: SE(3)-Equivariant Continuous-Filter SchNet & Conformal Molecular Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-chem--scientist--ai-yellow)](https://huggingface.co/spaces/dhrubchatterjee/chem-scientist-ai)
[![Tests: Passing](https://img.shields.io/badge/tests-4%2F4%20passed-brightgreen.svg)](tests/)

**ChemScientist AI** is a 3D equivariant molecular representation and machine learning platform. It couples continuous-filter convolutional neural networks (SchNet) with radial basis distance expansions, Mahalanobis applicability domain detection, and distribution-free inductive conformal prediction.

---

## 🔬 Scientific Foundations & Mathematical Formulations

### 1. Continuous-Filter Convolutional Layers (SchNet)
Traditional graph neural networks operate on discrete graph topology, losing spatial 3D conformation. SchNet maps continuous interatomic distances $d_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|_2$ into Gaussian radial basis functions (RBF):

$$e_k(d_{ij}) = \exp\left(-\gamma (d_{ij} - \mu_k)^2\right)$$

Continuous-filter interaction convolution with smooth cosine cutoff $f_{\mathrm{cut}}(d_{ij})$:

$$\mathbf{v}_i^{(l+1)} = \mathbf{v}_i^{(l)} + \sum_{j \neq i, d_{ij} < r_c} \mathbf{v}_j^{(l)} \odot W(e_k(d_{ij})) \cdot f_{\mathrm{cut}}(d_{ij})$$

$$f_{\mathrm{cut}}(d_{ij}) = \frac{1}{2} \left[\cos\left(\frac{\pi d_{ij}}{r_c}\right) + 1\right]$$

This formulation guarantees strict $E(3)$ / $SE(3)$ invariance with respect to global 3D translations and rotations.

### 2. Multivariate Mahalanobis Applicability Domain (AD)
Detects out-of-distribution (OOD) chemical structures in the latent representation space:

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

### 3. Finite-Sample Conformal Prediction
Yields rigorous $90\%$ and $95\%$ statistical coverage intervals $[y - q_{\alpha}, y + q_{\alpha}]$ with zero distribution assumptions.

---

## ⚡ Quickstart

### Installation

```bash
pip install -e .
```

### Python API

```python
from chem_scientist import SchNet3DEmbedder, ConformalMolecularEngine
import numpy as np

# 1. Generate 3D SchNet Embedding & Energy Prediction
embedder = SchNet3DEmbedder(embed_dim=64)
result = embedder.compute_energy_and_forces_proxy("CC(=O)Oc1ccccc1C(=O)O") # Aspirin

print(f"Number of Atoms: {result['num_atoms']}")
print(f"Max Interatomic Distance: {result['max_interatomic_distance_A']} Å")
print(f"Predicted 3D Potential Energy: {result['predicted_3d_potential_energy_eV']} eV")

# 2. Conformal Uncertainty & Applicability Domain Screening
engine = ConformalMolecularEngine()
latent = np.array(result["latent_vector"])
interval = engine.predict_conformal_interval(result["predicted_3d_potential_energy_eV"], latent)

print(f"90% Conformal Interval: [{interval.lower_bound_90}, {interval.upper_bound_90}] eV")
print(f"Mahalanobis Distance D_M: {interval.mahalanobis_distance:.2f} (Inside Domain: {interval.inside_applicability_domain})")
```

---

## 🖥️ Command-Line Interface (CLI)

```bash
# Predict molecular property with 3D conformation & conformal bounds
chem-scientist predict --smiles "c1ccccc1O" --embed-dim 64
```

---

## 🧪 Automated Verification Suite

```bash
python run_tests.py
```
```text
Ran 4 tests in 0.150s
OK - ALL CHEM-SCIENTIST-AI TESTS PASSED (4/4)!
```

---

## 📚 References
- Schütt, K. T., et al. (2018). *SchNet – A deep learning architecture for molecules and materials*. Journal of Chemical Physics, 148(24), 241722.
- Shafer, G., & Vovk, V. (2008). *A Tutorial on Conformal Prediction*. Journal of Machine Learning Research, 9, 371-421.
- Ramakrishnan, R., et al. (2014). *Quantum chemistry structures and properties of 134 kilo molecules*. Scientific Data, 1, 140022.

---

## 📄 License
MIT License. Copyright (c) 2026 Dhrub Chatterjee.
