"""Tests for geometric analysis methods."""

import pytest
import numpy as np
import os
import tempfile
from tragics import TRAGICS
from . import create_test_trajectory

class TestGeometryAnalyzer:
    """Test suite for geometry calculations."""
    
    @pytest.fixture
    def tragics_instance(self):
        """Create a TRAGICS instance with a test trajectory."""
        trajectory_path = create_test_trajectory(n_frames=5, n_atoms=4)
        logfile = tempfile.mktemp(suffix='.log')
        tragics = TRAGICS(trajectory_path, logfile)
        yield tragics
        # Cleanup
        os.remove(trajectory_path)
        os.remove(logfile)
    
    def test_calculate_radius_of_gyration(self, tragics_instance):
        """Test radius of gyration calculation."""
        frames, rg_values = tragics_instance.calculate_radius_of_gyration()
        
        # Check outputs
        assert isinstance(frames, np.ndarray)
        assert isinstance(rg_values, np.ndarray)
        assert len(frames) == len(rg_values) == 5  # n_frames
        assert np.all(rg_values >= 0)  # Non-negative values
        
        # Check plot file creation
        plot_file = f"{tragics_instance.name}_radius_of_gyration.pdf"
        assert os.path.exists(plot_file)
        os.remove(plot_file)
    
    def test_calculate_distance(self, tragics_instance):
        """Test atomic distance calculation."""
        distances = tragics_instance.calculate_distance(0, 1)
        
        # Check output
        assert isinstance(distances, np.ndarray)
        assert len(distances) == 5  # n_frames
        assert np.all(distances >= 0)  # Non-negative distances
    
    def test_invalid_atom_indices(self, tragics_instance):
        """Test error handling for invalid atom indices."""
        with pytest.raises(ValueError):
            tragics_instance.calculate_distance(-1, 1)
        
        with pytest.raises(ValueError):
            tragics_instance.calculate_distance(0, 100)  # Beyond n_atoms
