"""Geometric analysis methods - simplified for NEB integration."""

from typing import Optional, Tuple
import numpy as np

class GeometryAnalyzer:
    """Handles geometric analysis of molecular structures."""
    
    def __init__(self, tragics_instance):
        self.tragics = tragics_instance
    
    def calculate_radius_of_gyration(self,
                                   initial_frame: Optional[int] = None,
                                   final_frame: Optional[int] = None,
                                   step: int = 1,
                                   plot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate radius of gyration for all frames."""
        if step <= 0:
            raise ValueError("step must be positive")
        
        start = initial_frame if initial_frame is not None else 0
        end = final_frame if final_frame is not None else self.tragics.n_frames
        
        if start < 0 or start >= self.tragics.n_frames:
            raise ValueError(f"initial_frame must be between 0 and {self.tragics.n_frames-1}")
        if end <= 0 or end > self.tragics.n_frames:
            raise ValueError(f"final_frame must be between 1 and {self.tragics.n_frames}")
        if start >= end:
            raise ValueError("initial_frame must be less than final_frame")
        
        frames = range(start, end, step)
        
        rg_values = []
        frame_nums = []
        
        masses = self.tragics.universe.atoms.masses
        if masses is None:
            masses = np.ones(self.tragics.n_atoms)
        total_mass = np.sum(masses)
            
        for frame in frames:
            self.tragics.universe.trajectory[frame]
            positions = self.tragics.universe.atoms.positions
            com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
            rg_sq = np.sum(masses * np.sum((positions - com) ** 2, axis=1)) / total_mass
            rg = np.sqrt(rg_sq)
            
            rg_values.append(rg)
            frame_nums.append(frame)
        
        frame_nums = np.array(frame_nums)
        rg_values = np.array(rg_values)
        
        self.tragics.logger.info(
            f"\nRadius of Gyration Statistics:"
            f"\nMean: {np.mean(rg_values):.2f} Å"
            f"\nStd: {np.std(rg_values):.2f} Å"
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
        """Calculate distance between two atoms."""
        if not isinstance(atom1_idx, int) or not isinstance(atom2_idx, int):
            raise ValueError("Atom indices must be integers")
        if not (0 <= atom1_idx < self.tragics.n_atoms and 0 <= atom2_idx < self.tragics.n_atoms):
            raise ValueError(f"Atom indices must be between 0 and {self.tragics.n_atoms-1}")
        if atom1_idx == atom2_idx:
            raise ValueError("Cannot calculate distance between the same atom")
        
        if step <= 0:
            raise ValueError("step must be positive")
    
        start = initial_frame if initial_frame is not None else 0
        end = final_frame if final_frame is not None else self.tragics.n_frames
        
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
    
    def calculate_rdf(self, *args, **kwargs):
        """RDF calculation - see full TRAGICS implementation."""
        raise NotImplementedError("RDF implementation omitted for brevity. See full package.")
