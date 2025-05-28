"""SOAP-based analysis methods."""

from typing import Optional, Union, Tuple, List
import numpy as np
from dscribe.descriptors import SOAP
from ase import Atoms
import logging
import os

class SOAPCalculator:
    """Handles all SOAP-related calculations."""
    
    def __init__(self, tragics_instance):
        """Initialize with reference to parent TRAGICS instance."""
        self.tragics = tragics_instance
        self.soap_vec_frames = None
    
    def calculate_soap(self,
                      initial_frame: Optional[int] = None,
                      final_frame: Optional[int] = None,
                      step: int = 1,
                      r_cut: float = 5.0,
                      nl_max: int = 6,
                      subset_atoms: Optional[Union[str, List[int]]] = None) -> np.ndarray:
        """Calculate SOAP descriptor for all atoms or a subset.
        
        Args:
            initial_frame: First frame to include
            final_frame: Last frame to include  
            step: Frame step size
            r_cut: SOAP cutoff radius in Angstrom (must be positive)
            nl_max: SOAP basis set size (must be positive)
            subset_atoms: Atom indices or range string
            
        Returns:
            Array of SOAP vectors for each frame
            
        Raises:
            ValueError: If parameters are invalid or out of bounds
        """
        # Validate numeric parameters
        if step <= 0:
            raise ValueError("step must be positive")
        if r_cut <= 0:
            raise ValueError("r_cut must be positive")
        if nl_max <= 0:
            raise ValueError("nl_max must be positive")
        if nl_max > 20:  # Reasonable upper limit
            raise ValueError("nl_max should be <= 20 for computational efficiency")
        
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

            subset_file = f"{self.tragics.name}_subset.xyz"
            with open(subset_file, 'w') as f:
                frames = range(start, end, step)

                for frame in frames:
                    self.tragics.universe.trajectory[frame]
                    f.write(f"{len(atom_indices)}\n")
                    f.write(f"Frame {frame}\n")
                    for idx in atom_indices:
                        atom = self.tragics.universe.atoms[idx]
                        f.write(f"{atom.name} {atom.position[0]:.6f} "
                               f"{atom.position[1]:.6f} {atom.position[2]:.6f}\n")

            subset_universe = self.tragics.universe.__class__(subset_file, format='xyz')
            
            # Clean up temporary file
            try:
                os.remove(subset_file)
            except OSError:
                pass  # File cleanup is not critical
        else:
            subset_universe = self.tragics.universe

        # Calculate SOAP vectors
        frames = range(start, end, step)

        elements = list(set(atom.name for atom in subset_universe.atoms))

        soap_desc = SOAP(
            species=elements,
            r_cut=r_cut,
            n_max=nl_max,
            l_max=nl_max,
            average='inner'
        )

        soap_vectors = []
        for frame in frames:
            subset_universe.trajectory[frame]
            atoms_obj = Atoms(
                symbols=[atom.name for atom in subset_universe.atoms],
                positions=subset_universe.atoms.positions
            )
            soap_vec = soap_desc.create(atoms_obj)
            soap_vectors.append(soap_vec)

        self.soap_vec_frames = np.array(soap_vectors)
        return self.soap_vec_frames

    def soap_kernel_vector(self,
                         frame_idx: int,
                         initial_frame: Optional[int] = None,
                         final_frame: Optional[int] = None,
                         step: int = 1,
                         zeta: int = 4) -> np.ndarray:
        """Calculate SOAP kernel vector between one frame and all frames.
        
        Args:
            frame_idx: Reference frame index
            initial_frame: First frame to include 
            final_frame: Last frame to include
            step: Frame step size
            zeta: Kernel exponent (must be positive)
            
        Returns:
            Kernel vector of similarities
            
        Raises:
            ValueError: If frame_idx is out of bounds or parameters are invalid
        """
        # Validate frame_idx
        if not isinstance(frame_idx, int):
            raise ValueError("frame_idx must be an integer")
        if frame_idx < 0 or frame_idx >= self.tragics.n_frames:
            raise ValueError(f"frame_idx must be between 0 and {self.tragics.n_frames-1}")
        
        # Validate other parameters
        if step <= 0:
            raise ValueError("step must be positive")
        if zeta <= 0:
            raise ValueError("zeta must be positive")
            
        # Validate frame range
        start = initial_frame if initial_frame is not None else 0
        end = final_frame if final_frame is not None else self.tragics.n_frames
        
        if start < 0 or start >= self.tragics.n_frames:
            raise ValueError(f"initial_frame must be between 0 and {self.tragics.n_frames-1}")
        if end <= 0 or end > self.tragics.n_frames:
            raise ValueError(f"final_frame must be between 1 and {self.tragics.n_frames}")
        if start >= end:
            raise ValueError("initial_frame must be less than final_frame")
        
        # Calculate SOAP vectors if not already calculated
        if self.soap_vec_frames is None:
            soap_vectors = self.calculate_soap(initial_frame, final_frame, step)
        else:
            soap_vectors = self.soap_vec_frames
        
        # Validate frame_idx is within the calculated range
        frames = range(start, end, step)
        if frame_idx not in frames:
            raise ValueError(f"frame_idx {frame_idx} not in calculated frame range {list(frames)}")
        
        # Convert frame_idx to index in soap_vectors array
        soap_idx = list(frames).index(frame_idx)
        
        # Normalize reference frame SOAP vector
        ref_soap = soap_vectors[soap_idx]
        ref_soap_norm = ref_soap / np.linalg.norm(ref_soap)
        
        # Normalize all SOAP vectors
        soap_vectors_norm = soap_vectors / np.linalg.norm(soap_vectors, axis=1).reshape(-1, 1)
        
        # Calculate kernel vector
        kernel_vector = np.power(np.dot(soap_vectors_norm, ref_soap_norm), zeta)
        
        return kernel_vector

    def soap_kernel_matrix(self,
                          initial_frame: Optional[int] = None,
                          final_frame: Optional[int] = None,
                          step: int = 1,
                          zeta: int = 4,
                          output_file: Optional[str] = None) -> np.ndarray:
        """Calculate SOAP kernel matrix between all pairs of frames."""
        # Validate parameters
        if step <= 0:
            raise ValueError("step must be positive")
        if zeta <= 0:
            raise ValueError("zeta must be positive")
            
        # Validate frame range
        start = initial_frame if initial_frame is not None else 0
        end = final_frame if final_frame is not None else self.tragics.n_frames
        
        if start < 0 or start >= self.tragics.n_frames:
            raise ValueError(f"initial_frame must be between 0 and {self.tragics.n_frames-1}")
        if end <= 0 or end > self.tragics.n_frames:
            raise ValueError(f"final_frame must be between 1 and {self.tragics.n_frames}")
        if start >= end:
            raise ValueError("initial_frame must be less than final_frame")
            
        # Define frame range
        frames = range(start, end, step)
        n_frames = len(frames)
        
        # Initialize kernel matrix
        kernel_matrix = np.zeros((n_frames, n_frames))
        
        # Calculate kernel vectors for each frame as reference
        for i, frame in enumerate(frames):
            kernel_matrix[i] = self.soap_kernel_vector(frame, initial_frame, final_frame, step, zeta)

        # Save matrix if output file is specified
        if output_file:
            np.save(output_file, kernel_matrix)
            
        self.tragics.plotter.plot_matrix_scatter(kernel_matrix)
        return kernel_matrix

    def sequential_similarity_selection(self,
                                     output_file: str,
                                     threshold: float = 0.999,
                                     zeta: int = 4,
                                     r_cut: float = 5.0,
                                     nl_max: int = 6,
                                     subset_atoms: Optional[Union[str, List[int]]] = None) -> Tuple[np.ndarray, list]:
        """Select frames sequentially based on SOAP kernel similarity threshold."""
        # Validate parameters
        if not isinstance(output_file, str) or not output_file.strip():
            raise ValueError("output_file must be a non-empty string")
        if threshold < 0.1 or threshold >= 1.0:
            raise ValueError('Threshold must be in range [0.1, 1.0)')
        if zeta <= 0:
            raise ValueError("zeta must be positive")
        if r_cut <= 0:
            raise ValueError("r_cut must be positive")
        if nl_max <= 0:
            raise ValueError("nl_max must be positive")

        # Pre-compute all SOAP vectors
        if subset_atoms is not None:
            soap_all_frames = self.calculate_soap(subset_atoms=subset_atoms, r_cut=r_cut, nl_max=nl_max)
        else:
            soap_all_frames = self.calculate_soap(r_cut=r_cut, nl_max=nl_max)

        # Initialize lists for selected frames and their SOAP vectors
        selected_frames = []
        soap_selected_frames = []
        similarity_scores = []
        
        # Process all frames
        for i, frame_soap in enumerate(soap_all_frames):
            if i == 0:
                # Always select first frame
                selected_frames.append(i)
                soap_selected_frames.append(frame_soap / np.linalg.norm(frame_soap))
            else:
                # Normalize current frame SOAP vector
                frame_soap_norm = frame_soap / np.linalg.norm(frame_soap)

                if len(soap_selected_frames) > 0:
                    selected_soaps_array = np.array(soap_selected_frames)
                    similarities = np.power(np.dot(selected_soaps_array, frame_soap_norm), zeta)
                    max_similarity = np.max(similarities)

                    if max_similarity < threshold:
                        selected_frames.append(i)
                        soap_selected_frames.append(frame_soap_norm)
                        similarity_scores.append(max_similarity)

        # Write selected frames to output file
        self.tragics.writer.filter_trajectory(
            output_file=output_file,
            frames_to_write=selected_frames
        )

        self.tragics.logger.info(
            f"\nSelected {len(selected_frames)} frames out of {self.tragics.n_frames} total frames"
            f"\nThe similarity scores are: {similarity_scores}"
            f"\nSelection ratio: {len(selected_frames)/self.tragics.n_frames:.2%}"
        )

        self.tragics.logger.info(
            f"\nsequential_similarity_selection run for zeta={zeta}, "
            f"threshold={threshold}, nl_max={nl_max}, r_cut={r_cut}\n"
        )
        
        return np.array(selected_frames), similarity_scores