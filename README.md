# TRAGICS: TRajectory Analysis and Gauging In Chemical Space

A Python package for analyzing molecular dynamics trajectories.
This package draws some ideas from duartegroup/mlp-train (https://github.com/duartegroup/mlp-train). Thanks to the authors for their excellent work and open-source contributions.

## Features

- **SOAP Analysis**: Calculate SOAP descriptors and kernel similarities between molecular structures
- **Geometric Analysis**: Radius of gyration, atomic distances, radial distribution functions (RDF)  
- **Frame Selection**: Sequential similarity selection for trajectory sampling
- **Visualization**: Automated plotting of (some) analysis results
- **Trajectory I/O**: Filter and write trajectory subsets

## Installation

### Step 1: Download TRAGICS
Download or clone this repository and place the `tragics` folder in your working directory.

### Step 2: Install Dependencies
Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` (≥1.20.0)
- `matplotlib` (≥3.3.0) 
- `MDAnalysis` (≥2.0.0)
- `dscribe` (≥1.2.0) 
- `ase` (≥3.20.0)

### Step 3: Verify Installation
Test that everything works:

```python
from tragics import TRAGICS
print("TRAGICS successfully imported!")
```

## Quick Start

### Basic Usage

```python
from tragics import TRAGICS

# Initialize with your trajectory file
trajectory = TRAGICS('your_trajectory.xyz', 'analysis.log')

# Calculate SOAP descriptors for all frames
soap_vectors = trajectory.calculate_soap()

# Calculate radius of gyration over time
frames, rg_values = trajectory.calculate_radius_of_gyration()

# Select representative frames based on similarity
selected_frames, scores = trajectory.sequential_similarity_selection(
    output_file='selected_frames.xyz',
    threshold=0.999
)
```

### SOAP Analysis Example

```python
# Calculate SOAP kernel matrix for similarity analysis
kernel_matrix = trajectory.soap_kernel_matrix()

# Get similarity of frame 0 to all other frames  
similarity_vector = trajectory.soap_kernel_vector(frame_idx=0)

# Select frames with custom SOAP parameters
selected_frames, scores = trajectory.sequential_similarity_selection(
    output_file='representative_frames.xyz',
    threshold=0.99,
    r_cut=6.0,      # SOAP cutoff radius (Angstrom)
    nl_max=8        # SOAP basis set size
)
```

### Geometric Analysis Example

```python
# Calculate distance between atoms 0 and 1 over trajectory
distances = trajectory.calculate_distance(atom1_idx=0, atom2_idx=1)

# Calculate oxygen-oxygen RDF with periodic boundary conditions
distances, rdf_values = trajectory.calculate_rdf(
    selection1='O',                           # First selection (element)
    selection2='O',                           # Second selection  
    box_dimensions=[25.0, 25.0, 25.0],       # Box size in Angstrom
    max_dist=12.0                             # Maximum distance for RDF
)
```

### Trajectory Filtering Example

```python
# Write every 10th frame to a new file
trajectory.filter_trajectory(
    output_file='subsampled.xyz',
    initial_frame=0,
    final_frame=1000,
    step=10
)

# Write specific frames only
trajectory.filter_trajectory(
    output_file='selected.xyz',
    frames_to_write=[0, 50, 100, 200]
)

# Write subset of atoms (first 20 atoms)
trajectory.filter_trajectory(
    output_file='subset.xyz',
    subset_atoms=list(range(20))
)
```

## Output Files

TRAGICS automatically generates several output files:

- **Log files**: `your_logfile.log` - Detailed analysis information and timing
- **Plots**: 
  - `trajectory_name_radius_of_gyration.pdf` - Rg vs time
  - `trajectory_name-kernel_matrix.pdf` - SOAP similarity heatmap
  - `trajectory_name_element1_element2_rdf.pdf` - RDF plots
- **Data**: 
  - `trajectory_name_element1_element2_rdf.csv` - RDF data in CSV format
- **Trajectories**: User-specified XYZ files from filtering operations

## API Reference

### Core Class
- `TRAGICS(trajectory_path, logfile)` - Main analysis class

### SOAP Methods
- `calculate_soap(...)` - Calculate SOAP descriptors
- `soap_kernel_vector(frame_idx, ...)` - Frame-to-all similarity
- `soap_kernel_matrix(...)` - All-pairs similarity matrix  
- `sequential_similarity_selection(...)` - Representative frame selection

### Geometry Methods
- `calculate_radius_of_gyration(...)` - Molecular compactness over time
- `calculate_distance(atom1_idx, atom2_idx, ...)` - Inter-atomic distances
- `calculate_rdf(selection1, selection2, box_dimensions, ...)` - Radial distribution functions

### I/O Methods  
- `filter_trajectory(...)` - Write trajectory subsets


## Contributing

This package is under active development. For bug reports or feature requests, please check the issues section.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Gers&Claude

## Version

Current version: 0.1.0
