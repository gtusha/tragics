"""Plotting functionality for trajectory analysis."""

import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    """Handles all plotting functionality."""
    
    def __init__(self, tragics_instance):
        """Initialize with reference to parent TRAGICS instance."""
        self.tragics = tragics_instance
    
    def plot_matrix_scatter(self, matrix: np.ndarray) -> None:
        """Plot matrix values as a very dense scatter plot.
        
        Args:
            matrix: 2D numpy array containing the matrix to plot
        """
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
        """Create and save a plot of radius of gyration over time.
        
        Args:
            frames: Array of frame numbers
            rg_values: Array of radius of gyration values
        """
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
        """Create and save a plot of the radial distribution function.
    
        Args:
            distances: Array of radial distances (bin centers)
            rdf_values: Array of RDF values
            rdf_name: Base name for output files (will be appended with extensions)
            save_csv: If True, save the RDF data to a CSV file
        """
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
    
        # Save data to CSV if requested
        if save_csv:
            csv_filename = f"{self.tragics.name}_{rdf_name}.csv"
            header = "Distance(Angstrom),g(r)"
            data = np.column_stack((distances, rdf_values))
            np.savetxt(csv_filename, data, delimiter=',', header=header, comments='')
