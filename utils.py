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
        start_time = time.time()
        start_cpu = time.process_time()
        
        result = func(self, *args, **kwargs)
        
        wall_time = time.time() - start_time
        cpu_time = time.process_time() - start_cpu
        
        self.timings[func.__name__] = {
            'wall_time': wall_time,
            'cpu_time': cpu_time,
        }

        self.logger.info(
            f"\nTiming for {func.__name__}:"
            f"\nWall clock time: {wall_time:.2f} seconds"
            f"\nCPU time: {cpu_time:.2f} seconds"
        )

        return result
    return wrapper

def setup_logging(log_file: str) -> logging.Logger:
    """Initialize logging to file only."""
    logger = logging.getLogger(log_file)
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
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
        """Write selected frames to a new xyz file."""
        if not isinstance(output_file, str) or not output_file.strip():
            raise ValueError("output_file must be a non-empty string")
        
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise OSError(f"Cannot create output file at {output_file}: {e}")
        
        if step <= 0:
            raise ValueError("step must be positive")
        
        if not isinstance(append, str) or append.lower() not in ['yes', 'no']:
            raise ValueError("append must be 'yes' or 'no'")
        
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
            start = initial_frame if initial_frame is not None else 0
            end = final_frame if final_frame is not None else self.tragics.n_frames
            
            if start < 0 or start >= self.tragics.n_frames:
                raise ValueError(f"initial_frame must be between 0 and {self.tragics.n_frames-1}")
            if end <= 0 or end > self.tragics.n_frames:
                raise ValueError(f"final_frame must be between 1 and {self.tragics.n_frames}")
            if start >= end:
                raise ValueError("initial_frame must be less than final_frame")
        
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
                    ts = self.tragics.universe.trajectory[frame]
                    f.write(f"{len(atom_indices)}\n")
                    f.write(f"Frame {frame}\n")
                    for idx in atom_indices:
                        atom = self.tragics.universe.atoms[idx]
                        f.write(f"{atom.name} {atom.position[0]:.6f} "
                               f"{atom.position[1]:.6f} {atom.position[2]:.6f}\n")
        except (OSError, IOError) as e:
            raise OSError(f"Error writing to file {output_file}: {e}")

        return frames
