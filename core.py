"""Core module containing the main TRAGICS class."""

from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np
import MDAnalysis as mda

from .analysis.soap import SOAPCalculator
from .analysis.geometry import GeometryAnalyzer
from .analysis.neb import NEBCalculator
from .analysis.plotting import Plotter
from .utils import setup_logging, timer, TrajectoryWriter

class TRAGICS:
    """Class for analyzing molecular dynamics trajectories and more."""
    
    def __init__(self, trajectory_path: Union[str, Path], logfile: str):
        """Initialize trajectory analysis class.
        
        Args:
            trajectory_path: Path to the trajectory file
            logfile: Path to the log file
            
        Raises:
            FileNotFoundError: If trajectory file doesn't exist
            ValueError: If trajectory file is empty or invalid format
        """
        # Validate trajectory file exists
        trajectory_path = Path(trajectory_path)
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
        
        # Validate log file path
        logfile_path = Path(logfile)
        try:
            # Test if we can create the log file
            logfile_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create log file at {logfile}: {e}")
        
        try:
            self.universe = mda.Universe(str(trajectory_path), format='xyz')
            self.n_frames = len(self.universe.trajectory)
            self.n_atoms = len(self.universe.atoms)
            
            # Validate trajectory is not empty
            if self.n_frames == 0:
                raise ValueError("Trajectory file contains no frames")
            if self.n_atoms == 0:
                raise ValueError("Trajectory file contains no atoms")
                
        except Exception as e:
            raise ValueError(f"Failed to load trajectory file {trajectory_path}: {e}")
        
        self.timings = {}
        self.logger = setup_logging(str(logfile_path))
        
        # Initialize submodules
        self.name = trajectory_path.stem
        self.soap = SOAPCalculator(self)
        self.geometry = GeometryAnalyzer(self)
        self.neb = NEBCalculator(self)
        self.plotter = Plotter(self)
        self.writer = TrajectoryWriter(self)
    
    @timer
    def calculate_soap(self, *args, **kwargs):
        """Delegate to SOAP calculator."""
        return self.soap.calculate_soap(*args, **kwargs)
    
    @timer
    def soap_kernel_vector(self, *args, **kwargs):
        """Delegate to SOAP calculator."""
        return self.soap.soap_kernel_vector(*args, **kwargs)
    
    @timer
    def soap_kernel_matrix(self, *args, **kwargs):
        """Delegate to SOAP calculator."""
        return self.soap.soap_kernel_matrix(*args, **kwargs)
    
    @timer
    def sequential_similarity_selection(self, *args, **kwargs):
        """Delegate to SOAP calculator."""
        return self.soap.sequential_similarity_selection(*args, **kwargs)
    
    @timer
    def calculate_radius_of_gyration(self, *args, **kwargs):
        """Delegate to geometry analyzer."""
        return self.geometry.calculate_radius_of_gyration(*args, **kwargs)
    
    @timer
    def calculate_distance(self, *args, **kwargs):
        """Delegate to geometry analyzer."""
        return self.geometry.calculate_distance(*args, **kwargs)

    @timer
    def calculate_rdf(self, *args, **kwargs):
        """Delegate to geometry analyzer."""
        return self.geometry.calculate_rdf(*args, **kwargs)
    
    @timer
    def calculate_neb(self, *args, **kwargs):
        """Delegate to NEB calculator."""
        return self.neb.calculate_neb(*args, **kwargs)

    @timer
    def filter_trajectory(self, *args, **kwargs):
        """Delegate to trajectory writer."""
        return self.writer.filter_trajectory(*args, **kwargs)
