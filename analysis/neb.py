"""Nudged Elastic Band (NEB) calculations."""

from typing import Optional, Union, Tuple
import numpy as np
from pathlib import Path
from ase.io import read, write, Trajectory
from ase.neb import NEB
from ase.optimize import FIRE, BFGS
from ase.parallel import world
from ase.calculators.calculator import Calculator

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS
}


class NEBCalculator:
    """Handles NEB calculations with ASE."""
    
    def __init__(self, tragics_instance):
        """Initialize with reference to parent TRAGICS instance."""
        self.tragics = tragics_instance
    
    def _calculate_spring_constant(self, n_images: int, base_k: float = 0.1, 
                                   reference_images: int = 10) -> float:
        """Scale spring constant proportionally to number of images."""
        k = base_k * (n_images / reference_images)
        self.tragics.logger.info(f"Spring constant: k={k:.3f} eV/Å² for {n_images} images")
        return k
    
    def _optimize_endpoint(self, atoms, calculator: Calculator, fmax: float, 
                          optimizer_name: str, label: str):
        """Optimize an endpoint structure."""
        self.tragics.logger.info(f"Optimizing {label} structure...")
        
        atoms = atoms.copy()
        atoms.calc = calculator
        
        optimizer = OPTIMIZERS[optimizer_name](atoms)
        traj = Trajectory(f"optimized_{label}.traj", 'w', atoms)
        optimizer.attach(traj)
        optimizer.run(fmax=fmax)
        
        write(f"optimized_{label}.xyz", atoms)
        return atoms
    
    def _create_images(self, initial, final, n_images: int, ts_guess=None):
        """Create NEB images with optional TS guess."""
        images = [initial]
        
        if ts_guess is not None:
            if len(ts_guess) != len(initial):
                raise ValueError("TS guess must have same number of atoms as endpoints")
            
            n_left = (n_images - 3) // 2
            n_right = n_images - 3 - n_left
            
            for _ in range(n_left):
                images.append(initial.copy())
            images.append(ts_guess)
            for _ in range(n_right):
                images.append(final.copy())
        else:
            for _ in range(n_images - 2):
                images.append(initial.copy())
        
        images.append(final)
        return images
    
    def _setup_calculators(self, images, calculator: Calculator):
        """Set up calculators for NEB images with parallel support."""
        n_proc = world.size
        
        if n_proc > 1:
            self.tragics.logger.info(f"Parallel mode: {n_proc} processes")
            for i in range(1, len(images) - 1):
                if (i - 1) % n_proc == world.rank:
                    images[i].calc = calculator
        else:
            for i in range(1, len(images) - 1):
                images[i].calc = calculator
    
    def calculate_neb(self,
                     calculator: Calculator,
                     initial_frame: Optional[int] = None,
                     final_frame: Optional[int] = None,
                     initial_file: Optional[str] = None,
                     final_file: Optional[str] = None,
                     n_images: int = 7,
                     fmax: float = 0.05,
                     optimizer: str = 'FIRE',
                     optimize_endpoints: bool = False,
                     ts_guess_frame: Optional[int] = None,
                     ts_guess_file: Optional[str] = None,
                     use_climbing_image: bool = False,
                     ci_steps: Optional[int] = None,
                     spring_constant: Optional[float] = None,
                     output_file: str = 'neb_result.xyz') -> Tuple[list, np.ndarray]:
        """Run NEB calculation.
        
        Args:
            calculator: ASE calculator for energy/force calculations
            initial_frame: Frame index for initial structure (from trajectory)
            final_frame: Frame index for final structure (from trajectory)
            initial_file: Path to initial structure XYZ file
            final_file: Path to final structure XYZ file
            n_images: Number of images including endpoints
            fmax: Force convergence criterion (eV/Å)
            optimizer: Optimizer name ('FIRE' or 'BFGS')
            optimize_endpoints: Whether to optimize endpoints first
            ts_guess_frame: Frame index for TS guess (from trajectory)
            ts_guess_file: Path to TS guess XYZ file
            use_climbing_image: Enable climbing image NEB
            ci_steps: Steps for climbing image phase (None = converge)
            spring_constant: Manual spring constant (None = auto-scale)
            output_file: Output file for final NEB path
            
        Returns:
            Tuple of (images list, energy array)
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {list(OPTIMIZERS.keys())}")
        if n_images < 3:
            raise ValueError("n_images must be at least 3")
        if fmax <= 0:
            raise ValueError("fmax must be positive")
        
        # Load initial/final structures
        if initial_file and final_file:
            initial = read(initial_file)
            final = read(final_file)
            self.tragics.logger.info(f"Loaded structures from {initial_file}, {final_file}")
        elif initial_frame is not None and final_frame is not None:
            if not (0 <= initial_frame < self.tragics.n_frames):
                raise ValueError(f"initial_frame must be 0-{self.tragics.n_frames-1}")
            if not (0 <= final_frame < self.tragics.n_frames):
                raise ValueError(f"final_frame must be 0-{self.tragics.n_frames-1}")
            
            self.tragics.universe.trajectory[initial_frame]
            initial = self._trajectory_to_atoms()
            self.tragics.universe.trajectory[final_frame]
            final = self._trajectory_to_atoms()
            self.tragics.logger.info(f"Loaded frames {initial_frame}, {final_frame}")
        else:
            raise ValueError("Must provide either (initial_file, final_file) or (initial_frame, final_frame)")
        
        if len(initial) != len(final):
            raise ValueError("Initial and final must have same number of atoms")
        
        # Load TS guess if provided
        ts_guess = None
        if ts_guess_file:
            ts_guess = read(ts_guess_file)
            self.tragics.logger.info(f"Loaded TS guess from {ts_guess_file}")
        elif ts_guess_frame is not None:
            if not (0 <= ts_guess_frame < self.tragics.n_frames):
                raise ValueError(f"ts_guess_frame must be 0-{self.tragics.n_frames-1}")
            self.tragics.universe.trajectory[ts_guess_frame]
            ts_guess = self._trajectory_to_atoms()
            self.tragics.logger.info(f"Loaded TS guess from frame {ts_guess_frame}")
        
        # Optimize endpoints if requested
        if optimize_endpoints:
            initial = self._optimize_endpoint(initial, calculator, fmax, optimizer, "initial")
            final = self._optimize_endpoint(final, calculator, fmax, optimizer, "final")
        
        # Create images
        images = self._create_images(initial, final, n_images, ts_guess)
        
        # Set up NEB
        k = spring_constant if spring_constant else self._calculate_spring_constant(n_images)
        neb = NEB(images, k=k, climb=False, parallel=True)
        neb.interpolate()
        
        # Try IDPP interpolation
        try:
            neb.interpolate(method='idpp')
            self.tragics.logger.info("IDPP interpolation successful")
        except Exception as e:
            self.tragics.logger.info(f"IDPP failed, using linear: {e}")
        
        # Set up calculators
        self._setup_calculators(images, calculator)
        
        # Optimize
        self.tragics.logger.info(f"Starting NEB: {optimizer}, fmax={fmax} eV/Å")
        opt = OPTIMIZERS[optimizer](neb)
        traj = Trajectory("neb_trajectory.traj", 'w', neb)
        opt.attach(traj)
        
        if use_climbing_image:
            # Initial optimization with looser convergence
            opt.run(fmax=fmax*1.5)
            
            # Find highest energy image
            energies = np.array([img.get_potential_energy() for img in images])
            max_idx = np.argmax(energies[1:-1]) + 1
            
            self.tragics.logger.info(f"Enabling climbing image at index {max_idx}")
            neb = NEB(images, k=k, climb=True, parallel=True)
            neb.ci = max_idx - 1
            
            opt = OPTIMIZERS[optimizer](neb)
            ci_traj = Trajectory("neb_climbing_image.traj", 'w', neb)
            opt.attach(ci_traj)
            
            if ci_steps:
                opt.run(steps=ci_steps)
            else:
                opt.run(fmax=fmax)
        else:
            opt.run(fmax=fmax)
        
        # Calculate energies
        energies = np.array([img.get_potential_energy() for img in images])
        energies -= energies[0]
        
        barrier = max(energies)
        barrier_idx = np.argmax(energies)
        
        self.tragics.logger.info(
            f"\nNEB Results:\n"
            f"Energy barrier: {barrier:.4f} eV\n"
            f"Barrier at image: {barrier_idx}\n"
            f"Reaction energy: {energies[-1]:.4f} eV"
        )
        
        # Save results
        write(output_file, images)
        
        # Plot if available
        if hasattr(self.tragics, 'plotter'):
            self.tragics.plotter.plot_neb_barrier(np.arange(len(images)), energies)
        
        return images, energies
    
    def _trajectory_to_atoms(self):
        """Convert current MDAnalysis frame to ASE Atoms object."""
        from ase import Atoms
        return Atoms(
            symbols=[atom.name for atom in self.tragics.universe.atoms],
            positions=self.tragics.universe.atoms.positions
        )
