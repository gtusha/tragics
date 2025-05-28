"""Utility functions and classes for the TRAGICS package."""

import time
import logging
import numpy as np
from functools import wraps
from pathlib import Path
from typing import Optional, Union, List

def timer(func):
    """Decorator to measure execution time of methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Start timing
        start_time = time.time()
        start_cpu = time.process_time()
        
        # Execute the function
        result = func(self, *args, **kwargs)
        
        # End timing
        end_time = time.time()
        end_cpu = time.process_time()
        
        # Calculate durations
        wall_time = end_time - start_time
        cpu_time = end_cpu - start_cpu
        
        # Store timing information
        self.timings[func.__name__] = {
            'wall_time': wall_time,
            'cpu_time': cpu_time,
        }

        # Print timing information
        self.logger.info(
            f"\nTiming for {func.__name__}:"
            f"\nWall clock time: {wall_time:.2f} seconds"
            f"\nCPU time: {cpu_time:.2f} seconds"
        )

        return result
    return wrapper

def setup_logging(log_file: str) -> logging.Logger:
    """Initialize logging to file only.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Configured logger instance
    """
    # Create a new logger with a unique name
    logger = logging.getLogger(log_file)
    
    # Reset any existing handlers
    logger.handlers = []
    
    # Set the logging level
    logger.setLevel(logging.INFO)
    
    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False
    
    # Create and add the file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('- %(asctime)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

class TrajectoryWriter:
    """Handles trajectory file I/O operations."""
    
    def __init__(self, tragics_instance):
        """Initialize with reference to parent TRAGICS instance."""
        self.tragics = tragics_instance
    
    def filter_trajectory(self,
                         output_file: str,
                         initial_frame: Optional[int] = None,
                         final_frame: Optional[int] = None,
                         step: int = 1,
                         append: str = 'no',
                         frames_to_write: Optional[list] = None,
                         subset_atoms: Optional[Union[str, List[int]]] = None) -> List[int]:
        """Write selected frames to a new xyz file, or append them to an existing file.
        
        Args:
            output_file: Output file path
            initial_frame: First frame to include
            final_frame: Last frame to include
            step: Frame step size
            append: If 'yes', append to existing file instead of overwriting
            frames_to_write: Specific frames to write (overrides other frame selection parameters)
            subset_atoms: Atoms to include in output
            
        Returns:
            List of frames that were written
            
        Raises:
            ValueError: If parameters are invalid
            OSError: If output file cannot be created
        """
        # Validate output file
        if not isinstance(output_file, str) or not output_file.strip():
            raise ValueError("output_file must be a non-empty string")
        
        # Check if we can create the output file
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise OSError(f"Cannot create output file at {output_file}: {e}")
        
        # Validate step
        if step <= 0:
            raise ValueError("step must be positive")
        
        # Validate append parameter
        if not isinstance(append, str) or append.lower() not in ['yes', 'no']:
            raise ValueError("append must be 'yes' or 'no'")
        
        # Validate frames_to_write if provided
        if frames_to_write is not None:
            if not isinstance(frames_to_write, (list, tuple)):
                raise ValueError("frames_to_write must be a list or tuple")
            if len(frames_to_write) == 0:
                raise ValueError("frames_to_write cannot be empty")
            if not all(isinstance(f, int) for f in frames_to_write):
                raise ValueError("All frame indices must be integers")
            if not all(0 <= f < self.tragics.n_frames for f in frames_to_write):
                raise ValueError(f"All frame indices must be between 0 and {self.tragics.n_frames-1}")
        else:
            # Validate frame range
            start = initial_frame if initial_frame is not None else 0
            end = final_frame if final_frame is not None else self.tragics.n_frames
            
            if start < 0 or start >= self.tragics.n_frames:
                raise ValueError(f"initial_frame must be between 0 and {self.tragics.n_frames-1}")
            if end <= 0 or end > self.tragics.n_frames:
                raise ValueError(f"final_frame must be between 1 and {self.tragics.n_frames}")
            if start >= end:
                raise ValueError("initial_frame must be less than final_frame")
        
        # Process subset_atoms parameter
        if subset_atoms is not None:
            if isinstance(subset_atoms, str):
                if '-' not in subset_atoms:
                    raise ValueError("subset_atoms string must be in format 'start-end'")
                try:
                    start_atom, end_atom = map(int, subset_atoms.split('-'))
                    atom_indices = list(range(start_atom, end_atom + 1))
                except ValueError:
                    raise ValueError("subset_atoms string must contain valid integers")
            elif isinstance(subset_atoms, (list, tuple)):
                atom_indices = list(subset_atoms)
                if not all(isinstance(idx, int) for idx in atom_indices):
                    raise ValueError("All atom indices must be integers")
            else:
                raise ValueError("subset_atoms must be None, a list of indices, or a range string")

            if not all(0 <= idx < self.tragics.n_atoms for idx in atom_indices):
                raise ValueError(f"Atom indices must be between 0 and {self.tragics.n_atoms-1}")
            
            if len(atom_indices) == 0:
                raise ValueError("subset_atoms cannot be empty")
        else:
            atom_indices = list(range(self.tragics.n_atoms))

        # Determine frames to write
        if frames_to_write is not None:
            frames = frames_to_write
        else:
            start = initial_frame if initial_frame is not None else 0
            end = final_frame if final_frame is not None else self.tragics.n_frames
            frames = list(range(start, end, step))

        mode = 'a' if append.lower() == 'yes' else 'w'

        try:
            with open(output_file, mode) as f:
                for frame in frames:
                    # Go to specific frame
                    ts = self.tragics.universe.trajectory[frame]

                    # Write number of atoms
                    f.write(f"{len(atom_indices)}\n")
                    f.write(f"Frame {frame}\n")

                    # Write atomic positions for selected atoms
                    for idx in atom_indices:
                        atom = self.tragics.universe.atoms[idx]
                        f.write(f"{atom.name} {atom.position[0]:.6f} "
                               f"{atom.position[1]:.6f} {atom.position[2]:.6f}\n")
        except (OSError, IOError) as e:
            raise OSError(f"Error writing to file {output_file}: {e}")

        return frames

    @staticmethod
    def load_kernel_matrix(file_path: str) -> np.ndarray:
        """Load a previously saved kernel matrix.
        
        Args:
            file_path: Path to the .npy file containing the kernel matrix
            
        Returns:
            Loaded kernel matrix
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be loaded or is not a valid numpy array
        """
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Kernel matrix file not found: {file_path}")
        
        try:
            kernel_matrix = np.load(file_path)
            if not isinstance(kernel_matrix, np.ndarray):
                raise ValueError("Loaded file does not contain a numpy array")
            if kernel_matrix.ndim != 2:
                raise ValueError("Kernel matrix must be a 2D array")
            if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
                raise ValueError("Kernel matrix must be square")
            return kernel_matrix
        except Exception as e:
            raise ValueError(f"Failed to load kernel matrix from {file_path}: {e}")