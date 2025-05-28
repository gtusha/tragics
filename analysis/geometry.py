"""Geometric analysis methods."""

from typing import Optional, Tuple, Union, List
import numpy as np
from ase import Atoms

class GeometryAnalyzer:
    """Handles geometric analysis of molecular structures."""
    
    def __init__(self, tragics_instance):
        """Initialize with reference to parent TRAGICS instance."""
        self.tragics = tragics_instance
    
    def calculate_radius_of_gyration(self,
                                   initial_frame: Optional[int] = None,
                                   final_frame: Optional[int] = None,
                                   step: int = 1,
                                   plot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate radius of gyration for all frames or a subset of frames."""
        # Validate parameters
        if step <= 0:
            raise ValueError("step must be positive")
        
        # Define frame range
        start = initial_frame if initial_frame is not None else 0
        end = final_frame if final_frame is not None else self.tragics.n_frames
        
        # Validate frame range
        if start < 0 or start >= self.tragics.n_frames:
            raise ValueError(f"initial_frame must be between 0 and {self.tragics.n_frames-1}")
        if end <= 0 or end > self.tragics.n_frames:
            raise ValueError(f"final_frame must be between 1 and {self.tragics.n_frames}")
        if start >= end:
            raise ValueError("initial_frame must be less than final_frame")
        
        frames = range(start, end, step)
        
        # Initialize arrays for results
        rg_values = []
        frame_nums = []
        
        # Get masses once - reuse for all frames
        masses = self.tragics.universe.atoms.masses
        if masses is None:
            masses = np.ones(self.tragics.n_atoms)
        total_mass = np.sum(masses)
            
        # Calculate Rg for each frame
        for frame in frames:
            # Set the current frame
            self.tragics.universe.trajectory[frame]
            
            # Get positions for current frame
            positions = self.tragics.universe.atoms.positions
            
            # Calculate center of mass
            com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
            
            # Calculate radius of gyration
            rg_sq = np.sum(masses * np.sum((positions - com) ** 2, axis=1)) / total_mass
            rg = np.sqrt(rg_sq)
            
            rg_values.append(rg)
            frame_nums.append(frame)
        
        frame_nums = np.array(frame_nums)
        rg_values = np.array(rg_values)
        
        # Calculate and log statistics
        self.tragics.logger.info(
            f"\nRadius of Gyration Statistics:"
            f"\nMean: {np.mean(rg_values):.2f} Å"
            f"\nStandard Deviation: {np.std(rg_values):.2f} Å"
            f"\nMin: {np.min(rg_values):.2f} Å"
            f"\nMax: {np.max(rg_values):.2f} Å"
        )
        
        if plot:
            self.tragics.plotter.plot_radius_of_gyration(frame_nums, rg_values)
            
        return frame_nums, rg_values
    
    def calculate_distance(self,
                         atom1_idx: int,
                         atom2_idx: int,
                         initial_frame: Optional[int] = None,
                         final_frame: Optional[int] = None,
                         step: int = 1) -> np.ndarray:
        """Calculate distance between two atoms across trajectory frames."""
        # Validate atom indices
        if not isinstance(atom1_idx, int) or not isinstance(atom2_idx, int):
            raise ValueError("Atom indices must be integers")
        if not (0 <= atom1_idx < self.tragics.n_atoms and 0 <= atom2_idx < self.tragics.n_atoms):
            raise ValueError(f"Atom indices must be between 0 and {self.tragics.n_atoms-1}")
        if atom1_idx == atom2_idx:
            raise ValueError("Cannot calculate distance between the same atom")
        
        # Validate step
        if step <= 0:
            raise ValueError("step must be positive")
    
        # Define frame range
        start = initial_frame if initial_frame is not None else 0
        end = final_frame if final_frame is not None else self.tragics.n_frames
        
        # Validate frame range
        if start < 0 or start >= self.tragics.n_frames:
            raise ValueError(f"initial_frame must be between 0 and {self.tragics.n_frames-1}")
        if end <= 0 or end > self.tragics.n_frames:
            raise ValueError(f"final_frame must be between 1 and {self.tragics.n_frames}")
        if start >= end:
            raise ValueError("initial_frame must be less than final_frame")
        
        frames = range(start, end, step)
    
        distances = []
        for frame in frames:
            self.tragics.universe.trajectory[frame]
            pos1 = self.tragics.universe.atoms[atom1_idx].position
            pos2 = self.tragics.universe.atoms[atom2_idx].position
            distance = np.linalg.norm(pos1 - pos2)
            distances.append(distance)
            
        return np.array(distances)

    def calculate_rdf(self,
                    selection1: Union[List[int], int, str],
                    selection2: Union[List[int], int, str],
                    box_dimensions: List[float],
                    max_dist: float = 10.0,
                    n_bins: int = 100,
                    initial_frame: Optional[int] = None,
                    final_frame: Optional[int] = None,
                    step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the radial distribution function between two groups of atoms.
    
        Args:
            selection1: First selection (atom indices or element symbol)
            selection2: Second selection (atom indices or element symbol)
            box_dimensions: List of [x, y, z] dimensions of simulation box in Angstrom
            max_dist: Maximum distance for RDF calculation (in Angstrom)
            n_bins: Number of bins in the RDF histogram
            initial_frame: First frame to include (default: 0)
            final_frame: Last frame to include (default: all frames)
            step: Frame step size (default: 1)
        
        Returns:
            Tuple containing:
                - Array of radial distances (bin centers)
                - Array of RDF values
        
        Raises:
            ValueError: If parameters are invalid or selections are empty
            TypeError: If box_dimensions is not a list/array of numbers
        
        Examples:
            # RDF between elements with box dimensions
            rdf = calculate_rdf('O', 'O', box_dimensions=[30.0, 30.0, 30.0])
            rdf = calculate_rdf('O', 'H', box_dimensions=[25.0, 25.0, 25.0])
        
            # RDF between specific atoms
            rdf = calculate_rdf([0, 1], [2, 3], box_dimensions=[20.0, 20.0, 20.0])
        """
        # Validate box dimensions
        if not isinstance(box_dimensions, (list, tuple, np.ndarray)):
            raise TypeError("box_dimensions must be a list, tuple, or numpy array")
        if len(box_dimensions) != 3:
            raise ValueError("box_dimensions must contain exactly 3 values [x, y, z]")
        
        try:
            box_dimensions = np.array(box_dimensions, dtype=float)
        except (ValueError, TypeError):
            raise ValueError("box_dimensions must contain numeric values")
        
        if not np.all(box_dimensions > 0):
            raise ValueError("All box dimensions must be positive")
        
        # Validate RDF parameters
        if max_dist <= 0:
            raise ValueError("max_dist must be positive")
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if step <= 0:
            raise ValueError("step must be positive")
        
        # Check if max_dist is reasonable compared to box size
        min_box_size = np.min(box_dimensions)
        if max_dist > min_box_size / 2:
            raise ValueError(f"max_dist ({max_dist}) should be <= half the smallest box dimension ({min_box_size/2:.2f})")
        
        # Validate frame range
        start = initial_frame if initial_frame is not None else 0
        end = final_frame if final_frame is not None else self.tragics.n_frames
        
        if start < 0 or start >= self.tragics.n_frames:
            raise ValueError(f"initial_frame must be between 0 and {self.tragics.n_frames-1}")
        if end <= 0 or end > self.tragics.n_frames:
            raise ValueError(f"final_frame must be between 1 and {self.tragics.n_frames}")
        if start >= end:
            raise ValueError("initial_frame must be less than final_frame")
        
        # Validate and process selections
        def validate_and_get_indices(selection, name):
            if isinstance(selection, str):
                # Check if element exists in trajectory
                available_elements = set(atom.name for atom in self.tragics.universe.atoms)
                if selection not in available_elements:
                    raise ValueError(f"Element '{selection}' not found in trajectory. Available: {available_elements}")
                atom_indices = [i for i, atom in enumerate(self.tragics.universe.atoms) 
                              if atom.name == selection]
                if not atom_indices:
                    raise ValueError(f"No atoms found for element '{selection}'")
                return atom_indices
            elif isinstance(selection, int):
                if not (0 <= selection < self.tragics.n_atoms):
                    raise ValueError(f"{name} index must be between 0 and {self.tragics.n_atoms-1}")
                return [selection]
            elif isinstance(selection, (list, tuple)):
                if len(selection) == 0:
                    raise ValueError(f"{name} cannot be empty")
                if not all(isinstance(idx, int) for idx in selection):
                    raise ValueError(f"All indices in {name} must be integers")
                if not all(0 <= idx < self.tragics.n_atoms for idx in selection):
                    raise ValueError(f"All indices in {name} must be between 0 and {self.tragics.n_atoms-1}")
                return list(selection)
            else:
                raise ValueError(f"{name} must be an element symbol (str), atom index (int), or list of indices")
        
        atom_indices1 = validate_and_get_indices(selection1, "selection1")
        atom_indices2 = validate_and_get_indices(selection2, "selection2")
        
        box_volume = np.prod(box_dimensions)
        
        # Define frame range
        frames = range(start, end, step)
    
        # Initialize histogram
        bins = np.linspace(0, max_dist, n_bins + 1)
        hist = np.zeros(n_bins)
    
        # Calculate volume elements for each bin
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bin_volumes = 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
    
        # Count pairs for each frame
        n_frames = 0
    
        for frame in frames:
            self.tragics.universe.trajectory[frame]
            positions = self.tragics.universe.atoms.positions
        
            # Get positions for both groups
            pos1 = positions[atom_indices1]
            pos2 = positions[atom_indices2]
        
            # Calculate all pairwise distances, avoiding self-pairs if same selection
            same_selection = selection1 == selection2
            for i, p1 in enumerate(pos1):
                # If same selection, start from i+1 to avoid double counting
                start_j = i + 1 if same_selection else 0
                for j, p2 in enumerate(pos2[start_j:], start=start_j):
                    # Skip self-pairs
                    if atom_indices1[i] == atom_indices2[j]:
                        continue
                
                    # Calculate minimum image distance considering PBC
                    dr = p1 - p2
                    # Apply minimum image convention
                    for dim in range(3):
                        dr[dim] = dr[dim] - box_dimensions[dim] * round(dr[dim] / box_dimensions[dim])
                    dist = np.linalg.norm(dr)
                
                    if dist < max_dist:
                        # Find the bin index for this distance
                        bin_idx = np.digitize(dist, bins) - 1
                        if 0 <= bin_idx < n_bins:
                            hist[bin_idx] += 1
        
            n_frames += 1
    
        # Normalize histogram
        if n_frames > 0:
            # Volume normalization
            hist = hist / bin_volumes
        
            # Normalize by number of frames and particle densities
            n1 = len(atom_indices1)
            n2 = len(atom_indices2)
        
            # For same element RDF, correct the normalization
            if same_selection:
                # For N particles, number of unique pairs is N*(N-1)/2
                pair_norm = (n1 * (n1 - 1)) / 2
            else:
                pair_norm = n1 * n2
            
            density_norm = pair_norm / box_volume
            hist = hist / (n_frames * density_norm)
    
        # Create descriptive name for output files
        if isinstance(selection1, str) and isinstance(selection2, str):
            rdf_name = f"{selection1}_{selection2}_rdf"
        else:
            rdf_name = "rdf"
        
        # Log some statistics
        self.tragics.logger.info(
            f"\nRDF Statistics for {selection1}-{selection2}:"
            f"\nNumber of frames analyzed: {n_frames}"
            f"\nNumber of particles: {len(atom_indices1)} {selection1}, {len(atom_indices2)} {selection2}"
            f"\nBox dimensions: {box_dimensions[0]:.2f} × {box_dimensions[1]:.2f} × {box_dimensions[2]:.2f} Å³"
            f"\nBox volume: {box_volume:.2f} Å³"
            f"\nMaximum g(r) value: {np.max(hist):.2f}"
            f"\nFirst peak position: {bin_centers[np.argmax(hist)]:.2f} Å"
        )
    
        # Create plot if available
        if hasattr(self.tragics, 'plotter'):
            self.tragics.logger.info(f"\nPlotting {selection1}-{selection2} RDF...")
            self.tragics.plotter.plot_rdf(bin_centers, hist, rdf_name=rdf_name)
    
        return bin_centers, hist