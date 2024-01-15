import logging
from multiprocessing import shared_memory
from typing import List, Tuple

import numpy as np
from IPython.core.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


class SimulationEnvironment:
    def __init__(self, grid_size, molecules, diffusion_rate, dt):
        self.environment_grid = EnvironmentGrid(grid_size=grid_size, molecules=molecules, diffusion_rate=diffusion_rate,
                                                dt=dt)
        self.astrocytes = []
        self.glutamate_release_manager = GlutamateReleaseManager(self.environment_grid)
        self.spatial_index = RtreeSpatialIndex()  # Optional
    
    def run_simulation_step(self):
        pass
    
    def add_astrocyte(self, astrocyte):
        pass
    
    def remove_astrocyte(self, astrocyte):
        pass


class EnvironmentGrid:
    def __init__(self, grid_size: Tuple[int, int], molecules: List[str], diffusion_rate: float, dt: float,
                 dtype: str = 'float16'):
        """
        Initialize the environment grid with shared numpy arrays for each molecule.

        Args:
            grid_size: A tuple representing the dimensions of the grid (NxM).
            molecules: A list of molecule names to be tracked.
            diffusion_rate: Diffusion rate.
            dt: Time step of the simulation.
            dtype: The data type of the arrays.
            
        """
        self.grid_size = grid_size
        self.molecules = molecules
        self.diffusion_rate = diffusion_rate
        self.dt = dt
        
        self.shared_arrays = self._create_shared_arrays(grid_size, molecules, dtype=dtype)
        
        self._check_cfl_condition(diffusion_rate=diffusion_rate, dt=dt)
    
    @staticmethod
    def _create_shared_arrays(grid_size: Tuple[int, int], molecules: List[str], dtype='float16'):
        """
        Create shared numpy arrays for each molecule.

        Args:
            grid_size: Dimensions of the grid.
            molecules: List of molecule names.
            dtype: The data type of the arrays.

        Returns:
            Dictionary of shared numpy arrays.
        """
        shared_arrays = {}
        for molecule in molecules:
            # Create a new shared memory block
            shm = shared_memory.SharedMemory(create=True, size=np.prod(grid_size) * np.dtype(dtype).itemsize)
            # Create a numpy array using the shared memory
            shared_array = np.ndarray(grid_size, dtype=dtype, buffer=shm.buf)
            shared_arrays[molecule] = (shared_array, shm)
        return shared_arrays
    
    def _update_concentrations(self):
        """
        Update the concentrations in the grid using a vectorized approach with
        Dirichlet boundary conditions (edges as sinks).
        """
        for molecule, (shared_array, _) in self.shared_arrays.items():
            # Creating a copy of the current state of the grid
            current_concentration = np.copy(shared_array)
            
            # Applying Dirichlet boundary conditions (setting edges to zero)
            current_concentration[0, :] = current_concentration[-1, :] = 0
            current_concentration[:, 0] = current_concentration[:, -1] = 0
            
            # Vectorized diffusion update
            # Using slicing for interior cells and avoiding loop calculations
            change = self.diffusion_rate * self.dt * (
                    current_concentration[:-2, 1:-1] -  # concentration from upper row
                    2 * current_concentration[1:-1, 1:-1] +  # current cell concentration (doubled for laplacian)
                    current_concentration[2:, 1:-1] +  # concentration from lower row
                    current_concentration[1:-1, :-2] -  # concentration from left column
                    2 * current_concentration[1:-1, 1:-1] +  # current cell concentration (doubled for laplacian)
                    current_concentration[1:-1, 2:])  # concentration from right column
            
            shared_array[1:-1, 1:-1] = current_concentration[1:-1, 1:-1] + change
    
    def step(self, time_step: int = 1):
        """
        Advance the simulation by a specified number of time steps.

        This method updates the molecular concentrations in the grid
        based on the diffusion rate, using the vectorized update approach.
        It applies the updates iteratively for the given number of time steps.

        Args:
            time_step: An integer representing the number of time steps
                       to advance the simulation. Defaults to 1.
        """
        for t in range(time_step):
            self._update_concentrations()
    
    def get_concentration_at(self, location: Tuple[int, int], molecule: str):
        """
        Get the concentration of a specific molecule at a given location.

        Args:
            location: A tuple (x, y) representing the grid coordinates.
            molecule: Name of the molecule.

        Returns:
            The concentration of the specified molecule at the given location.
        """
        x, y = location
        return self.shared_arrays[molecule][0][x][y]
    
    def set_random_starting_concentration(self, molecule: str, n_spots: int = 10,
                                          concentration_boundaries: Tuple[int, int] = (75, 150),
                                          border: int = 3):
        
        if molecule not in self.molecules:
            logging.error(f"Molecule {molecule} not found in the grid.")
        
        arr, _ = self.shared_arrays[molecule]
        
        locations = []
        for n in range(n_spots):
            x, y = (np.random.randint(border, self.grid_size[0] - border),
                    np.random.randint(border, self.grid_size[1] - border))
            concentration = np.random.randint(*concentration_boundaries)
            
            arr[x, y] = concentration
            locations.append((x, y, concentration))
        
        return locations
    
    @staticmethod
    def _check_cfl_condition(diffusion_rate: float, dt: float, dx=1):
        """
        Check if the CFL condition is met for given parameters.

        Args:
            diffusion_rate: Diffusion rate.
            dt: Time step of the simulation.
            dx: Grid spacing
        """
        cfl_value = diffusion_rate * dt / (dx ** 2)
        
        if cfl_value > 0.5:
            logging.warning("Warning: CFL condition not met. Consider reducing the time step or diffusion rate.")
    
    def close(self):
        """
        Close and unlink all shared memory blocks.
        """
        for _, (array, shm) in self.shared_arrays.items():
            shm.close()
            shm.unlink()
    
    def __del__(self):
        """
        Destructor to ensure proper cleanup of shared memory resources.
        This method is automatically called when the object is garbage collected.
        """
        self.close()
    
    def plot_concentration(self, molecule: str, figsize: tuple = (10, 8), cmap: str = 'inferno', ax: plt.axis = None):
        """
        Plot the concentration of a specified molecule across the grid using Matplotlib.

        Args:
            molecule: Name of the molecule to plot.
            figsize: Tuple representing the figure size (width, height).
            cmap: Colormap for the heatmap.
            ax: Axes object to plot the heatmap.
        """
        if molecule not in self.molecules:
            logging.error(f"Molecule {molecule} not found in the grid.")
        
        # Retrieve the shared array for the specified molecule
        concentration_array, _ = self.shared_arrays[molecule]
        
        # Create a figure and axis for the heatmap
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            # ax.get_figure().clear()
            ax.clear()
        
        # Plotting the concentration heatmap
        heatmap = ax.imshow(concentration_array, cmap=cmap, interpolation='nearest')
        
        # Adding a colorbar and setting titles and labels
        # plt.colorbar(heatmap, ax=ax)
        ax.set_title(f"Concentration of {molecule}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Display the plot
        return ax
    
    def interactive_plot(self, molecule: str, figsize: tuple = (5, 5), cmap: str = 'inferno', frames: int = 200,
                         time_steps: int = 1):
        """
        Create an interactive plot for Jupyter Notebooks to visualize the concentration changes
        of a specified molecule over time.

        Args:
            molecule: Name of the molecule to plot.
            figsize: Tuple representing the figure size (width, height).
            cmap: Colormap for the heatmap.
            frames: Number of frames.
            time_steps: Number of time steps per frame
        """
        if molecule not in self.molecules:
            logging.error(f"Molecule {molecule} not found in the grid.")
            return None
        
        # Create a figure and axis for the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initialize the heatmap with the initial state of the grid
        concentration_array, _ = self.shared_arrays[molecule]
        heatmap = ax.imshow(concentration_array, cmap=cmap, interpolation='nearest')
        
        # Function to update the heatmap
        def update(frame):
            self.step(time_step=time_steps)  # Advance the simulation by one time step
            _concentration_array, _ = self.shared_arrays[molecule]
            heatmap.set_data(_concentration_array)
            
            # Update the color scale based on the new data
            new_min = np.min(concentration_array)
            new_max = np.max(concentration_array)
            heatmap.set_clim(new_min, new_max)
            
            return [heatmap]
        
        # Create the animation
        anim = FuncAnimation(fig, update, frames=frames, )
        
        # Display the animation
        return HTML(anim.to_html5_video())


class Astrocyte:
    def __init__(self):
        self.branches = []
    
    def manage_growth(self):
        pass
    
    def interact_with_environment(self, environment_grid):
        pass


class AstrocyteBranch:
    def __init__(self):
        self.nodes = []
        self.status = "active"  # or "pruned"
    
    def grow(self):
        pass
    
    def branch(self):
        pass
    
    def prune(self):
        pass


class AstrocyteNode:
    def __init__(self, position, diameter):
        self.position = position
        self.diameter = diameter
    
    def update_position(self, new_position):
        pass
    
    def change_diameter(self, new_diameter):
        pass


class GlutamateReleaseManager:
    def __init__(self, environment_grid):
        self.environment_grid = environment_grid
    
    def stochastic_release(self):
        pass
    
    def signal_based_release(self, signal):
        pass
    
    def update_environment(self):
        pass


class RtreeSpatialIndex:
    def __init__(self):
        self.rtree = None  # Placeholder for the R-tree structure
    
    def insert_branch(self, branch):
        pass
    
    def remove_branch(self, branch):
        pass
    
    def search(self, region):
        pass


class DataLogger:
    def __init__(self):
        self.logged_data = []
    
    def log_data(self, data):
        pass
    
    def retrieve_historical_data(self):
        pass
    
    def export_data(self):
        pass
