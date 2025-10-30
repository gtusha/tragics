# TRAGICS: TRajectory Analysis and Gauging In Chemical Space

A Python package for analyzing molecular dynamics trajectories, perform NEB calculations and more.

## Features

- **SOAP Analysis**: Calculate SOAP descriptors and kernel similarities
- **Geometric Analysis**: Radius of gyration, atomic distances, RDF  
- **NEB Calculations**: Nudged Elastic Band with climbing image support
- **Frame Selection**: Sequential similarity selection with SOAP from trajectories
- **Visualization**: Automated plotting of (some) analysis results
- **Trajectory I/O**: Filter and write trajectory subsets

## Installation

```bash
# Install dependencies
pip install numpy matplotlib MDAnalysis ase dscribe

# For MACE support (optional)
pip install mace-torch
```

Place `tragics` folder in your working directory.

## Quick Start

### NEB Calculation

```python
from tragics import TRAGICS
from mace.calculators import MACECalculator

traj = TRAGICS('trajectory.xyz', 'analysis.log')

# Basic NEB from trajectory frames
images, energies = traj.calculate_neb(
    calculator=MACECalculator(model_paths='model.model', device='cpu'),
    initial_frame=0,
    final_frame=100,
    n_images=11,
    fmax=0.05
)

# Advanced: climbing image + endpoint optimization + TS guess
images, energies = traj.calculate_neb(
    calculator=calculator,
    initial_file='initial.xyz',      # Or use initial_frame=0
    final_file='final.xyz',           # Or use final_frame=100
    ts_guess_file='guess.xyz',        # Optional TS guess
    n_images=15,
    optimize_endpoints=True,          # Pre-optimize endpoints
    use_climbing_image=True,          # Accurate TS location
    optimizer='FIRE',                 # Or 'BFGS'
    spring_constant=0.3               # Or None for auto-scale
)

print(f"Barrier: {max(energies):.3f} eV")
```

**Key parameters:**
- `calculator`: Any ASE calculator (MACE, EMT, etc.)
- Input: `(initial_frame, final_frame)` OR `(initial_file, final_file)`
- `n_images=7`: Images including endpoints
- `fmax=0.05`: Convergence criterion (eV/Å)
- `use_climbing_image=False`: Enable for accurate TS
- `optimize_endpoints=False`: Pre-optimize reactant/product

**Outputs:** XYZ path, ASE trajectory, energy plot/CSV, detailed log

**Parallel:** Automatic MPI support via ASE: `mpirun -np 4 python script.py`

## Other TRAGICS Features

### Geometric Analysis
```python
frames, rg = traj.calculate_radius_of_gyration()
distances = traj.calculate_distance(atom1_idx=0, atom2_idx=1)
```

### Trajectory Filtering
```python
traj.filter_trajectory(
    output_file='subset.xyz',
    frames_to_write=[0, 10, 20, 30]
)
```

## Version

Current version: 0.2.0 (with NEB support)

## Authors

Gers

## License

MIT License
