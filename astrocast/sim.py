import logging
from multiprocessing import shared_memory
from typing import Callable, List, Tuple, Union

import numpy as np
import seaborn as sns
from IPython.core.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


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
    
    def set_concentration_at(self, location: Tuple[int, int], molecule: str, concentration: float):
        """
        Set the concentration of a specific molecule at a given location.

        Args:
            location: A tuple (x, y) representing the grid coordinates.
            molecule: Name of the molecule.
            concentration: New concentration value to set.
        """
        x, y = location
        if molecule in self.shared_arrays:
            self.shared_arrays[molecule][0][x, y] = concentration
        else:
            logging.error(f"Molecule {molecule} not found in the grid.")
    
    def update_concentration_at(self, location: Tuple[int, int], molecule: str, concentration_change: float):
        """
        Update the concentration of a specific molecule at a given location.

        Args:
            location: A tuple (x, y) representing the grid coordinates.
            molecule: Name of the molecule.
            concentration_change: Amount to increment the concentration by.
        """
        x, y = location
        if molecule in self.shared_arrays:
            self.shared_arrays[molecule][0][x, y] += concentration_change
        else:
            print(f"Molecule {molecule} not found in the grid.")
    
    def set_random_starting_concentration(self, molecule: str, n_spots: int = 10,
                                          concentration_boundaries: Tuple[int, int] = (75, 150),
                                          border: int = 3):
        
        if molecule not in self.molecules:
            logging.error(f"Molecule {molecule} not found in the grid.")
        
        locations = []
        for n in range(n_spots):
            x, y = (np.random.randint(border, self.grid_size[0] - border),
                    np.random.randint(border, self.grid_size[1] - border))
            concentration = np.random.randint(*concentration_boundaries)
            
            self.set_concentration_at((x, y), molecule, concentration)
            
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
        ax.imshow(concentration_array, cmap=cmap, interpolation='nearest')
        
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


class GlutamateReleaseManager:
    def __init__(self, environment_grid: EnvironmentGrid, num_branches: int = 10, num_hotspots: int = 16,
                 z_thickness: float = 3.0, border: int = 3, jitter: float = 0.1,
                 release_amplitude: float = 1,
                 stochastic_probability: float = 0, signal_function: Union[Callable, np.array] = lambda x: 0):
        """
        Initialize the GlutamateReleaseManager with dendritic branches intersecting the imaging volume.

        Args:
            environment_grid: Instance of the EnvironmentGrid class.
            num_branches: Number of dendritic branches to simulate.
            z_thickness: Thickness of the imaging plane in Z-dimension.
            jitter: Amount of jitter in the hotspot placement, default is 0.1.
            stochastic_probability: Probability of release at each hotspot.
            signal_function: Function to generate signal-based probabilities.
        """
        self.environment_grid = environment_grid
        self.num_branches = num_branches
        self.num_hotspots = num_hotspots
        self.z_thickness = z_thickness
        self.border = border
        self.jitter = jitter
        self.release_amplitude = release_amplitude
        self.stochastic_probability = stochastic_probability
        self.signal_function = signal_function
        self.lines, self.hotspots = self._generate_hotspots()
        self.time_step = 0
    
    def _generate_hotspots(self) -> Tuple[np.array, np.array]:
        """
        Generate hotspots based on dendritic branches intersecting the imaging volume.

        Returns:
            Tuple of line and hotspot coordinates as np.array.
        """
        hotspots = []
        lines = []
        for branch_id in range(self.num_branches):
            # Generate a random line (dendritic branch) within the volume
            line = self._generate_random_line()
            lines.extend([line])
            
            # Place hotspots along the line with some jitter
            hotspots.extend(self._place_hotspots_on_line(line, self.num_hotspots, branch_id))
        
        return np.array(lines), np.array(hotspots)
    
    def _generate_random_line(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        # Generate random start and end points for the line within the grid boundaries
        start_point = (np.random.uniform(self.border, self.environment_grid.grid_size[0] - self.border),
                       np.random.uniform(self.border, self.environment_grid.grid_size[1] - self.border),
                       np.random.uniform(0, self.z_thickness))
        
        end_point = (np.random.uniform(self.border, self.environment_grid.grid_size[0] - self.border),
                     np.random.uniform(self.border, self.environment_grid.grid_size[1] - self.border),
                     np.random.uniform(0, self.z_thickness))
        
        return start_point, end_point
    
    def _place_hotspots_on_line(self, line: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
                                num_hotspots: Union[Tuple[int, int], int] = 16,
                                branch_id: int = 0) -> List[Tuple[int, int, int]]:
        start_point, end_point = line
        hotspots = []
        
        # Define the number of hotspots to place along the line
        if isinstance(num_hotspots, int):
            min_num_hotspots, max_num_hotspots = 2, num_hotspots
        elif isinstance(num_hotspots, type):
            min_num_hotspots, max_num_hotspots = num_hotspots
        else:
            raise ValueError(f"Invalid input for num_hotspots: {num_hotspots}. Please provide an integer or a tuple.")
        
        num_hotspots = np.random.randint(low=max(2, min_num_hotspots), high=max_num_hotspots)
        
        # Randomly distribute hotspot locations
        for i in range(num_hotspots):
            # Interpolate along the line to get the hotspot position
            t = i / float(num_hotspots - 1)
            x = start_point[0] + t * (end_point[0] - start_point[0])
            y = start_point[1] + t * (end_point[1] - start_point[1])
            
            # Apply jitter
            x_jitter = x + np.random.uniform(-self.jitter, self.jitter)
            y_jitter = y + np.random.uniform(-self.jitter, self.jitter)
            
            # Ensure the hotspot is still within the grid boundaries
            x_jitter = max(0, min(x_jitter, self.environment_grid.grid_size[0] - 1))
            y_jitter = max(0, min(y_jitter, self.environment_grid.grid_size[1] - 1))
            
            # ensure type int
            x_jitter, y_jitter = int(x_jitter), int(y_jitter)
            
            hotspots.append((x_jitter, y_jitter, branch_id))
        
        return hotspots
    
    def step(self, time_step: int = 1):
        """
        Advance the glutamate release simulation by one step.

        Arg:
            time_step: How many steps to advance the glutamate release simulation.

        """
        
        for i in range(time_step):
            stochastic_vector = self._stochastic_release(self.stochastic_probability)
            signal_vector = self._signal_based_release(self.signal_function)
            
            # Sum the probabilities and decide on release
            combined_prob = stochastic_vector + signal_vector
            random_vector = np.random.uniform(0, 1, size=combined_prob.shape)
            release_decision = random_vector < combined_prob
            release_difference = combined_prob - release_decision
            
            # Update the grid for each hotspot where release is decided
            for (x, y, _), release, difference in zip(self.hotspots, release_decision, release_difference):
                if release:
                    release_amount = difference * self.release_amplitude
                    self.environment_grid.update_concentration_at((int(x), int(y)), "glutamate", release_amount)
            
            self.time_step += 1
    
    def _stochastic_release(self, stochastic_probability: float) -> np.array:
        """
        Generate a probability vector for stochastic release.

        Args:
            stochastic_probability: Probability of release at each hotspot.

        Returns:
            A numpy array of release probabilities for each hotspot.
        """
        
        stoch_probability = np.zeros(len(self.hotspots))
        stoch_probability[:] = stochastic_probability
        
        return stoch_probability
    
    def _signal_based_release(self, signal_function: Union[Callable, np.array] = lambda x: 0) -> np.array:
        """
        Generate a probability vector for signal-based release.

        Args:
            signal_function: Function to generate signal-based probabilities.

        Returns:
            A numpy array of signal-based release probabilities for each hotspot.
        """
        
        signal_probability = np.zeros(len(self.hotspots))
        if isinstance(signal_function, Callable):
            signal_probability[:] = signal_function(self.time_step)
        elif isinstance(signal_function, np.ndarray):
            idx = self.time_step % len(signal_function)
            signal_probability[:] = signal_function[idx]
        else:
            raise ValueError("signal_function must be Callable or np.ndarray")
        
        return signal_probability
    
    def plot(self, line_thickness: int = 1, line_alpha: float = 0.7,
             marker: str = 'x', hotspot_alpha: float = 0.8, title: str = 'Dendrites and Hotspots',
             figsize: Tuple[int, int] = (5, 5), ax: plt.axis = None):
        """
        Plot dendritic lines and hotspots in a 2D image.

        Args:
            line_thickness: Thickness of the dendritic lines.
            line_alpha: Alpha (transparency) of the dendritic lines.
            marker: Marker style for hotspots.
            hotspot_alpha: Alpha (transparency) of the hotspots.
            title: Title of the figure.
            figsize: Size of the figure.
            ax: Matplotlib axis to plot the dendritic lines and hotspots in the 2D image.
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        color_palette = sns.color_palette("husl", n_colors=self.num_branches)
        for branch_id in range(self.num_branches):
            line = self.lines[branch_id][:, :-1]
            hotspots = self.hotspots
            
            # Extracting x, y coordinates for line and hotspots
            x_hotspots, y_hotspots = zip(*[(x, y) for x, y, b_id in hotspots if b_id == branch_id])
            
            # Plot the line
            px0, px1 = line[:, 0], line[:, 1]
            ax.plot(px0, px1, color=color_palette[branch_id], linewidth=line_thickness, alpha=line_alpha)
            
            # Plot the hotspots
            ax.scatter(x_hotspots, y_hotspots, marker=marker, color=color_palette[branch_id], alpha=hotspot_alpha)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        
        X, Y = self.environment_grid.grid_size
        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        
        return ax


class Simulation:
    def __init__(self, environment_grid: EnvironmentGrid, glutamate_release_manager: GlutamateReleaseManager = None):
        self.environment_grid = environment_grid
        self.astrocytes = []
        self.glutamate_release_manager = glutamate_release_manager
        self.spatial_index = RtreeSpatialIndex()  # Optional
    
    def run_simulation_step(self, time_step=1):
        
        if self.glutamate_release_manager is not None:
            self.glutamate_release_manager.step(time_step=time_step)
        
        self.environment_grid.step(time_step=time_step)
    
    def add_astrocyte(self, astrocyte):
        pass
    
    def remove_astrocyte(self, astrocyte):
        pass
    
    def plot(self, molecule="glutamate", figsize=(10, 5)):
        
        fig, axx = plt.subplot_mosaic("AB", figsize=figsize)
        
        self.environment_grid.plot_concentration(molecule=molecule, ax=axx["A"])
        if self.glutamate_release_manager is not None:
            self.glutamate_release_manager.plot(ax=axx["B"])
        
        return fig


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
