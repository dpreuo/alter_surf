# AI Coding Agent Instructions for alter_surf

## Project Overview
Research codebase for studying altermagnetic materials using mean-field theory and tight-binding models. Core focus: surface states, spectral densities, and spin-dependent transport in DLKK (d-wave antiferromagnetic) lattice models with layered geometries.

## Architecture & Core Dependencies

### External Library: blochK
**Critical**: This project heavily relies on the [`blochK`](https://github.com/valeeb/blochK.git) library (installed from git). All Hamiltonian construction, diagonalization, and observables computation flow through blochK's API:
- `blochK.Hamiltonian2D` / `Hamiltonian3D` - wrapper objects for tight-binding models
- `blochK.observable` - compute conductivity, spectral functions, DOS
- `blochK.plotting` - band structures, Fermi surfaces with coloring operators

**Never** implement band diagonalization or BZ sampling from scratch - always use blochK's methods.

### Module Structure
- `src/alter_surf/hamiltonian_*.py` - Model definitions (DLKK, CuMnAs, 2-orbital)
  - Functions return Hamiltonians **as functions** of momentum: `H_DLKK_2D_fct(kx, ky, **params) -> ndarray`
  - Wrapped by `create_H_DLKK_3D(param=dict)` → returns blochK.Hamiltonian2D object
- `src/alter_surf/mean_field.py` - Self-consistent Hartree-Fock solver
- `src/alter_surf/utils3D.py` - Layer projectors, custom colormaps (`cmap_bkr`, `two_param_color`)
- `src/alter_surf/utils_DLKK.py` - Layer-resolved conductivity calculations

## Critical Conventions

### Parameter Dictionaries
Model parameters are **always** passed as dictionaries to Hamiltonian constructors:
```python
Hparam = dict(len_z=50, delta=1, tp=0.3, tz=1, mAF=3, mu=3.1, PBC=False, U=4, filling=0.5)
H = create_H_DLKK_3D_MF(param=Hparam)
```
**Never** use positional arguments for physics parameters.

### Layer Indexing Convention
- Slabs use `len_z` layers (not `len_z-1`)
- Layer pairs counted: (0,1), (2,3), ..., meaning `layer//2` gives unit cell index
- AFM order alternates by layer: `mAF_z = m * cos(Q_z * z)` where `Q_z=π` typical

### Basis Ordering in Hamiltonians
```python
# DLKK 2D: [A↑, B↑, A↓, B↓] where A,B are sublattices
# DLKK 3D slab: [z=0: A↑,B↑,A↓,B↓, z=1: A↑,B↑,A↓,B↓, ..., z=len_z-1]
```
Operator construction uses `np.kron(layer_projector, spin/sublattice_operator)`.

### Data Persistence
Precomputed results stored as **pickle files** in `analysis/` directory:
```python
with open('MF_solution_DLKK_n=0.54_U=4_L=50.pkl', 'rb') as f:
    Hparam = pickle.load(f)
```
Naming convention: `{type}_DLKK_n={filling}_U={U}_L={len_z}.pkl`

## Typical Workflows

### Mean-Field Calculation
1. Define initial parameters with `len_z`, `U`, `filling`
2. Set initial guesses: `initial_mAF`, `initial_mF`, `initial_n` (arrays of length `len_z`)
3. Run `hartree_fock(H, initial_parameters, n_steps=100, Lq=50, mixing_proportion=0.2)`
4. Check convergence by plotting magnetization/density evolution
5. Save converged params: `pickle.dump(H.param, file)`

### Visualization (Notebooks)
**Standard imports** in all analysis notebooks:
```python
import numpy as np
from numpy import pi, cos, sin, exp
import matplotlib.pyplot as plt
import blochK
from blochK.plotting import plot_FS, plot_bandstruc
from blochK.plotting.publication import set_size, revtex_columnwidth
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from alter_surf.hamiltonian_DLKK import create_H_DLKK_3D_MF
from alter_surf.utils3D import projector2layer, two_param_color
import pickle
%load_ext autoreload
%autoreload 2
```

**Typical plot structure**:
- Spectral density: Use `two_param_color(spin_spectral, spectral_density)` for combined color+alpha
- Spin splitter angle: `2*arctan(spin_cond/cond) * 180/π`
- Apply power laws for visibility: `colors**0.7`, `alphas**0.8`

### Testing
Run with pytest from project root:
```bash
pytest tests/
```
Tests check Hamiltonian shapes, mean-field convergence, observable calculations. All tests import from `alter_surf` package (path set in `tests/conftest.py`).

## Common Pitfalls

1. **Layer projectors**: Use `projector2layer(layer, len_z)` from `utils3D.py`, not manual zero-arrays
2. **Conductivity computation**: For layer-resolved, use `get_conductivity_layer_resolved()` which handles sublattice/spin contraction
3. **Spectral broadening**: Lorentzian `Gamma/(E^2 + Gamma^2)` applied after diagonalization, typical `Gamma ~ 0.05`
4. **BZ sampling**: Always use `H.BZ.sample(Lk)` to get momentum grid, never manual `np.linspace`
5. **Plotting extent**: Use `extent(ks)` from blochK.plotting.utils for correct imshow axes

## File Organization
- Top-level notebooks (`MF_surface_DLKK.ipynb`, etc.): Computation scripts that **save** data
- `analysis/*.ipynb`: Load pickled data and generate figures
- `figures_for_the_paper/`: Publication-ready plotting with specific styles
- `benchmarks/`: Performance and validation checks

## Questions to Address
When unclear about implementation:
1. Check if similar calculation exists in `analysis/Figures_DLKK.ipynb` (most comprehensive example)
2. Verify parameter names match Hamiltonian function signatures in `hamiltonian_DLKK.py`
3. For observables, check blochK documentation - don't reimplement
