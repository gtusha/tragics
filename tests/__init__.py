"""Test suite for TRAGICS package."""

import os
import tempfile
import numpy as np

def create_test_trajectory(n_frames: int = 10, n_atoms: int = 5) -> str:
    """Create a temporary xyz trajectory file for testing.
    
    Args:
        n_frames: Number of frames to generate
        n_atoms: Number of atoms per frame
        
    Returns:
        Path to the temporary trajectory file
    """
    # Create temporary file
    fd, path = tempfile.mkstemp(suffix='.xyz')
    
    with os.fdopen(fd, 'w') as f:
        # Generate random trajectory
        for _ in range(n_frames):
            f.write(f"{n_atoms}\n")
            f.write("Random frame\n")
            
            # Generate random atomic positions
            for _ in range(n_atoms):
                x = np.random.uniform(-5, 5)
                y = np.random.uniform(-5, 5)
                z = np.random.uniform(-5, 5)
                f.write(f"C {x:.6f} {y:.6f} {z:.6f}\n")
    
    return path
