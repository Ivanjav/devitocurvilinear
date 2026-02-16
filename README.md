# devitocurvilinear

Finite-difference elastodynamic simulations in curvilinear coordinates, focused on complex topography.

This repository contains:
- a Python package (`devitocurvilinear`) for curvilinear grid generation/mapping and model setup,
- Devito-based example notebooks,
- analytical reference examples (2D elastic half-space), and
- SPECFEM2D comparison cases.

## Main capabilities

- Build curvilinear meshes from topography with `meshgrid_from_topo`.
- Compute high-order finite-difference derivatives (`deriv1_8th`, `deriv1_4th`).
- Map fields between Cartesian and curvilinear grids (`CurviMap`, `mapping_velocity`).
- Define seismic models with curvilinear metric factors (`SeismicModel`, aliases `Model`, `ModelElastic`, etc.).

## Repository layout

- `devitocurvilinear/`: core Python package.
- `devito_examples/`: notebooks for Devito simulations (`homotop`, `homotilted`, `seam`).
- `analytical_examples/`: analytical/Fortran reference case (`homoflat`).
- `specfem2d_examples/`: SPECFEM2D setups and outputs for comparison.
- `benchmark/`: notebook-based comparison/analysis workflows.

## Requirements

Core Python package (`devitocurvilinear`):
- `numpy` (declared in `pyproject.toml`)
- `scipy`
- `sympy`
- `matplotlib`

Example-specific requirements:
- `devito_examples/`: requires **Devito** (official repository: https://github.com/devitocodes/devito)
- `analytical_examples/`: requires **EX2DVAEL** (https://www.quest-itn.org/library/software/ex2ddir-ex2delel-ex2dvael.html)
- `specfem2d_examples/`: requires **SPECFEM2D** (official repository: https://github.com/SPECFEM/specfem2d)

For notebook workflows, also install:
- `jupyter` / `notebook`

## Installation

```bash
git clone <your-repo-url>
cd devitocurvilinear
pip install -e .
```

Install extra runtime dependencies as needed for notebooks, for example:

```bash
pip install scipy sympy matplotlib jupyter devito
```


## Running included examples

### 1) Devito notebook examples

Requirement: Devito (official repository: https://github.com/devitocodes/devito)

Open and run:
- `devito_examples/homotop/example_homotop.ipynb`
- `devito_examples/homotilted/example_homotilted.ipynb`
- `devito_examples/seam/example_seam.ipynb`


### 2) Analytical reference example (Fortran)

Requirement: EX2DVAEL (https://www.quest-itn.org/library/software/ex2ddir-ex2delel-ex2dvael.html)

Inside `analytical_examples/homoflat/`:

```bash
cd analytical_examples/homoflat
gfortran explosion.f -O2 -o explosion
./explosion < INDATA_EXPLO
```

This generates analytical traces used for comparison.

### 3) SPECFEM2D comparison runs

Requirement: SPECFEM2D (official repository: https://github.com/SPECFEM/specfem2d)

Each case has a `run_this_example.sh` script (e.g., `specfem2d_examples/homotop/`).

Before running, set the correct SPECFEM2D path in the script (it currently exports `SPECFEM2D=/home/ivan/Software/specfem2d`).

```bash
cd specfem2d_examples/homotop
bash run_this_example.sh
```

## Notes

- This is an actively developed research codebase; notebooks and example outputs are included in-repo.
- The git tree may contain generated outputs from numerical runs (`OUTPUT_FILES`, logs, traces).
- Examples based on the **SEAM Foothills Model Phase II** may not be fully runnable without licensed model access/data. More information: https://seg.org/seam/