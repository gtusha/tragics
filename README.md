# TRAGICS: TRajectory Analysis and Gauging In Chemical Space

A Python package for analyzing molecular dynamics trajectories with integrated NEB calculations.

## Features

- **SOAP Analysis**: Calculate SOAP descriptors and kernel similarities
- **Geometric Analysis**: Radius of gyration, atomic distances, RDF  
- **NEB Calculations**: Nudged Elastic Band with climbing image support
- **Frame Selection**: Sequential similarity selection for trajectory sampling
- **Visualization**: Automated plotting of analysis results
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

### Basic NEB Calculation

```python
from tragics import TRAGICS
from ase.calculators.emt import EMT

# Initialize
traj = TRAGICS('trajectory.xyz', 'analysis.log')

# Run NEB
images, energies = traj.calculate_neb(
    calculator=EMT(),
    initial_frame=0,
    final_frame=100,
    n_images=11,
    fmax=0.05,
    output_file='neb_result.xyz'
)

print(f"Barrier: {max(energies):.3f} eV")
```

### NEB with MACE

```python
from mace.calculators import MACECalculator

calculator = MACECalculator(
    model_paths='your_model.model',
    device='cpu'
)

images, energies = traj.calculate_neb(
    calculator=calculator,
    initial_frame=0,
    final_frame=100,
    n_images=15,
    fmax=0.03,
    optimize_endpoints=True,
    use_climbing_image=True,
    output_file='neb_mace.xyz'
)
```

### NEB from Separate Files

```python
images, energies = traj.calculate_neb(
    calculator=calculator,
    initial_file='initial.xyz',
    final_file='final.xyz',
    ts_guess_file='ts_guess.xyz',  # Optional
    n_images=11,
    output_file='neb_result.xyz'
)
```

## NEB Parameters

### Required
- `calculator`: ASE calculator (MACE, EMT, etc.)
- Either `(initial_frame, final_frame)` or `(initial_file, final_file)`

### Optional
- `n_images=7`: Number of images (including endpoints)
- `fmax=0.05`: Force convergence (eV/Å)
- `optimizer='FIRE'`: Optimizer ('FIRE' or 'BFGS')
- `optimize_endpoints=False`: Optimize initial/final structures
- `ts_guess_frame=None`: TS guess frame index
- `ts_guess_file=None`: TS guess XYZ file
- `use_climbing_image=False`: Enable climbing image NEB
- `ci_steps=None`: Climbing image steps (None = converge)
- `spring_constant=None`: Manual spring constant (None = auto-scale)
- `output_file='neb_result.xyz'`: Output path

## NEB Features

### Climbing Image NEB
Accurately locates transition states:
```python
images, energies = traj.calculate_neb(
    calculator=calculator,
    initial_frame=0,
    final_frame=100,
    use_climbing_image=True,
    ci_steps=500  # Or None for convergence
)
```

### Endpoint Optimization
Pre-optimize reactant/product:
```python
images, energies = traj.calculate_neb(
    calculator=calculator,
    initial_frame=0,
    final_frame=100,
    optimize_endpoints=True
)
```

### Parallel Computation
Automatic MPI support with ASE:
```bash
mpirun -np 4 python neb_script.py
```

### Outputs
- `neb_result.xyz`: Final NEB path
- `neb_trajectory.traj`: ASE trajectory
- `trajectory_neb_barrier.pdf`: Energy profile plot
- `trajectory_neb_barrier.csv`: Energy data
- `neb_analysis.log`: Detailed log

## Other TRAGICS Features

### SOAP Analysis
```python
soap_vectors = traj.calculate_soap()
kernel_matrix = traj.soap_kernel_matrix()

selected_frames, scores = traj.sequential_similarity_selection(
    output_file='selected.xyz',
    threshold=0.99
)
```

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

Gers & Claude

## License

MIT License
