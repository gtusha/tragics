"""Plotting functionality for trajectory analysis."""

import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    """Handles all plotting functionality."""
    
    def __init__(self, tragics_instance):
        self.tragics = tragics_instance
    
    def plot_matrix_scatter(self, matrix: np.ndarray) -> None:
        """Plot matrix values as a dense scatter plot."""
        rows, cols = matrix.shape
        x, y = np.meshgrid(np.arange(rows), np.arange(cols))

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x.flatten(), y.flatten(),
                            c=matrix.flatten(),
                            cmap='hot',
                            s=0.75,
                            marker='s',
                            edgecolors='none')

        plt.colorbar(scatter)
        plt.xlabel('Frame')
        plt.ylabel('Frame')
        plt.title('SOAP kernel: similarity function')
        plt.savefig(f'{self.tragics.name}-kernel_matrix.pdf',
                   bbox_inches='tight',
                   dpi=300)
        plt.close()
    
    def plot_radius_of_gyration(self,
                               frames: np.ndarray,
                               rg_values: np.ndarray) -> None:
        """Plot radius of gyration over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(frames, rg_values, '-b', label='Radius of Gyration')
        plt.xlabel('Frame')
        plt.ylabel('Radius of Gyration (Å)')
        plt.title('Radius of Gyration vs Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"{self.tragics.name}_radius_of_gyration.pdf",
                   dpi=300,
                   bbox_inches='tight')
        plt.close()

    def plot_rdf(self,
                distances: np.ndarray,
                rdf_values: np.ndarray,
                rdf_name: str = "rdf",
                save_csv: bool = True) -> None:
        """Plot radial distribution function."""
        plt.figure(figsize=(10, 6))
        plt.plot(distances, rdf_values, '-b', label='RDF')
        plt.xlabel('Distance (Å)')
        plt.ylabel('g(r)')
        plt.title('Radial Distribution Function')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"{self.tragics.name}_{rdf_name}.pdf",
                dpi=300,
                bbox_inches='tight')
        plt.close()
    
        if save_csv:
            csv_filename = f"{self.tragics.name}_{rdf_name}.csv"
            header = "Distance(Angstrom),g(r)"
            data = np.column_stack((distances, rdf_values))
            np.savetxt(csv_filename, data, delimiter=',', header=header, comments='')
    
    def plot_neb_barrier(self,
                        image_indices: np.ndarray,
                        energies: np.ndarray,
                        save_csv: bool = True) -> None:
        """Plot NEB energy barrier."""
        plt.figure(figsize=(10, 6))
        plt.plot(image_indices, energies, '-o', color='blue', 
                markersize=8, linewidth=2, label='NEB Path')
        
        max_idx = np.argmax(energies)
        plt.plot(image_indices[max_idx], energies[max_idx], 'r*', 
                markersize=15, label=f'Barrier: {energies[max_idx]:.3f} eV')
        
        plt.xlabel('Image Index')
        plt.ylabel('Relative Energy (eV)')
        plt.title('NEB Energy Profile')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"{self.tragics.name}_neb_barrier.pdf",
                   dpi=300,
                   bbox_inches='tight')
        plt.close()
        
        if save_csv:
            csv_filename = f"{self.tragics.name}_neb_barrier.csv"
            header = "ImageIndex,RelativeEnergy(eV)"
            data = np.column_stack((image_indices, energies))
            np.savetxt(csv_filename, data, delimiter=',', header=header, comments='')
