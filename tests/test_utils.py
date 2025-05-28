"""Tests for utility functions and classes."""

import pytest
import os
import tempfile
import numpy as np
from tragics import TRAGICS
from . import create_test_trajectory

class TestTrajectoryWriter:
    """Test suite for trajectory writing functionality."""
    
    @pytest.fixture
    def tragics_instance(self):
        """Create a TRAGICS instance with a test trajectory."""
        trajectory_path = create_test_trajectory(n_frames=5, n_atoms=3)
        logfile = tempfile.mktemp(suffix='.log')
        tragics = TRAGICS(trajectory_path, logfile)
        yield tragics
        # Cleanup
        os.remove(trajectory_path)
        os.remove(logfile)
    
    def test_filter_trajectory(self, tragics_instance):
        """Test trajectory filtering and writing."""
        output_file = tempfile.mktemp(suffix='.xyz')
        
        # Write specific frames
        frames = tragics_instance.filter_trajectory(
            output_file=output_file,
            frames_to_write=[0, 2, 4]
        )
        
        # Check output
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            content = f.readlines()
            
        # Each frame has n_atoms + 2 lines (count and comment)
        assert len(content) == (3 * 5)  # 3 frames * (3 atoms + 2 header lines)
        
        os.remove(output_file)
    
    def test_subset_atoms(self, tragics_instance):
        """Test writing subset of atoms."""
        output_file = tempfile.mktemp(suffix='.xyz')
        
        # Write only first two atoms
        frames = tragics_instance.filter_trajectory(
            output_file=output_file,
            subset_atoms=[0, 1]
        )
        
        # Check output
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            content = f.readlines()
            
        # Each frame should have 2 atoms + 2 header lines
        assert len(content) == (5 * 4)  # 5 frames * (2 atoms + 2 header lines)
        
        os.remove(output_file)
    
    def test_append_mode(self, tragics_instance):
        """Test appending to existing trajectory file."""
        output_file = tempfile.mktemp(suffix='.xyz')
        
        # Write first frame
        tragics_instance.filter_trajectory(
            output_file=output_file,
            frames_to_write=[0]
        )
        
        # Append second frame
        tragics_instance.filter_trajectory(
            output_file=output_file,
            frames_to_write=[1],
            append='yes'
        )
        
        # Check output
        with open(output_file, 'r') as f:
            content = f.readlines()
            
        # Should have 2 frames * (3 atoms + 2 header lines)
        assert len(content) == (2 * 5)
        
        os.remove(output_file)
    
    def test_invalid_atom_indices(self, tragics_instance):
        """Test error handling for invalid atom subset specification."""
        with pytest.raises(ValueError):
            tragics_instance.filter_trajectory(
                output_file="test.xyz",
                subset_atoms=[100]  # Invalid index
            )
        
        with pytest.raises(ValueError):
            tragics_instance.filter_trajectory(
                output_file="test.xyz",
                subset_atoms="invalid-format"  # Invalid format
            )
