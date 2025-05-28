"""Tests for SOAP-based analysis methods."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from tragics import TRAGICS
from . import create_test_trajectory

class TestSOAPCalculator:
    """Test suite for SOAP calculations."""
    
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
    
    def test_calculate_soap(self, tragics_instance):
        """Test basic SOAP vector calculation."""
        soap_vectors = tragics_instance.calculate_soap()
        
        # Check shape and content
        assert soap_vectors is not None
        assert isinstance(soap_vectors, np.ndarray)
        assert len(soap_vectors) == 5  # n_frames
        assert all(isinstance(v, np.ndarray) for v in soap_vectors)
    
    def test_soap_kernel_vector(self, tragics_instance):
        """Test SOAP kernel vector calculation."""
        kernel_vector = tragics_instance.soap_kernel_vector(frame_idx=0)
        
        # Check properties
        assert kernel_vector is not None
        assert isinstance(kernel_vector, np.ndarray)
        assert len(kernel_vector) == 5  # n_frames
        assert np.all(kernel_vector <= 1.0)  # Normalized
        assert np.all(kernel_vector >= 0.0)  # Non-negative
    
    def test_sequential_similarity_selection(self, tragics_instance):
        """Test frame selection based on similarity."""
        output_file = tempfile.mktemp(suffix='.xyz')
        
        selected_frames, scores = tragics_instance.sequential_similarity_selection(
            output_file=output_file,
            threshold=0.9
        )
        
        # Check results
        assert isinstance(selected_frames, np.ndarray)
        assert isinstance(scores, list)
        assert len(selected_frames) > 0
        assert all(0 <= idx < 5 for idx in selected_frames)  # Valid frame indices
        
        # Check output file
        assert os.path.exists(output_file)
        os.remove(output_file)
    
    def test_invalid_threshold(self, tragics_instance):
        """Test error handling for invalid threshold values."""
        with pytest.raises(ValueError):
            tragics_instance.sequential_similarity_selection(
                output_file="test.xyz",
                threshold=0.05  # Too low
            )
        
        with pytest.raises(ValueError):
            tragics_instance.sequential_similarity_selection(
                output_file="test.xyz",
                threshold=1.1  # Too high
            )
