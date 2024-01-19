from __future__ import annotations

import logging
from collections import deque
from multiprocessing import shared_memory
from typing import Callable, List, Tuple, Union

import numpy as np
import seaborn as sns
from IPython.core.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from rtree import index


class EnvironmentGrid:
    def __init__(self, grid_size: Tuple[int, int], diffusion_rate: float, dt: float, pixel_volume: float = 1.0,
                 molecules: List[str] = ("glutamate", "calcium", "repellent"),
                 degrades: List[str] = ("repellent",), degradation_factor: float = 0.75,
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
        self.degrades = degrades
        self.degradation_factor = degradation_factor
        self.pixel_volume = pixel_volume
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
    
    def _degrade_molecules(self):
        for molecule in self.degrades:
            self.shared_arrays[molecule] *= self.degradation_factor
    
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
            self._degrade_molecules()
    
    def get_tracked_molecules(self) -> List[str]:
        return list(self.shared_arrays.keys())
    
    def get_concentration_at(self, location: Union[Tuple[int, int], List[Tuple[int, int]]],
                             molecule: str) -> Union[float, List[float]]:
        """
        Get the concentration of a specific molecule at a given location.

        Args:
            location: A tuple (x, y) representing the grid coordinates.
            molecule: Name of the molecule.

        Returns:
            The concentration of the specified molecule at the given location.
        """
        
        if isinstance(location, tuple):
            location = [location]
        
        concentrations = []
        for x, y in location:
            concentrations.append(self.shared_arrays[molecule][0][x][y])
        
        if len(concentrations) < 2:
            return concentrations[0]
        else:
            return concentrations
    
    def get_amount_at(self, location: Union[Tuple[int, int], List[Tuple[int, int]]],
                      molecule: str) -> Union[float, List[float]]:
        
        concentrations = self.get_concentration_at(location, molecule)
        return [concentration / self.pixel_volume for concentration in concentrations]
    
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
    
    def update_concentration_at(self, location: Union[Tuple[int, int], List[Tuple[int, int]]],
                                molecule: str,
                                amount: float):
        """
        Update the concentration of a specific molecule at a given location or multiple locations.

        Args:
            location: A single tuple (x, y) representing the grid coordinates or a list of such tuples.
            molecule: Name of the molecule.
            amount: Amount to increment in mol.
        """
        if molecule not in self.shared_arrays:
            raise ValueError(f"Molecule {molecule} not found in the grid.")
        
        # If a single location is provided, wrap it in a list.
        if isinstance(location, tuple):
            location = [location]
        
        # evenly distribute concentration across pixels
        amount /= len(location)
        concentration_change = amount / self.pixel_volume
        
        # Update the concentration at each location by the specified amount.
        actually_removed = 0
        for x, y in location:
            
            self.shared_arrays[molecule][0][x, y] += concentration_change
            
            if self.shared_arrays[molecule][0][x, y] < 0:
                actually_removed += concentration_change - self.shared_arrays[molecule][0][x, y]
                self.shared_arrays[molecule][0][x, y] = 0
            else:
                actually_removed += concentration_change
        
        return actually_removed
    
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
            release_amplitude: Maximum release amplitude of a glutamate event in mol.
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
    
    children: List[AstrocyteBranch] = []
    branches: List[AstrocyteBranch] = []
    
    def __init__(self, position: Tuple[int, int], radius: int, num_branches: int,
                 max_branch_radius: float, start_spawn_radius: float, spawn_length: int,
                 repellent_concentration: float,
                 environment_grid: EnvironmentGrid, spatial_index: RtreeSpatialIndex,
                 max_history: int = 100, molecules: dict = None):
        
        self.max_history = max_history
        self.environment_grid = environment_grid
        self.spatial_index = spatial_index
        
        self.spawn_initial_branches(num_branches=num_branches, max_branch_radius=max_branch_radius,
                                    spawn_radius=start_spawn_radius, spawn_length=spawn_length)
        
        self.x, self.y = position
        self.radius = radius
        self.pixels = self.get_pixels_within_cell()
        
        # Establish initial concentrations
        self.molecules = dict(glutamate=0, calcium=1e-6, ATP=10e-6) if molecules is None else molecules
        self.repellent_concentration = repellent_concentration
    
    def step(self, time_step=1):
        for i in range(time_step):
            
            for branch in self.branches:
                branch.step()
            
            self.remove_molecules_from_cell_body()
            self.release_repellent()
    
    def get_pixels_within_cell(self) -> List[Tuple[int, int]]:
        """
        Get the grid cells (pixels) that are within the astrocyte's cell body.

        Returns:
            A list of tuples, where each tuple represents the coordinates (x, y) of a grid cell within the cell body.
        """
        pixels = []
        # Define the bounding box around the astrocyte
        x_min = max(0, self.x - self.radius)
        x_max = min(self.environment_grid.grid_size[0], self.x + self.radius)
        y_min = max(0, self.y - self.radius)
        y_max = min(self.environment_grid.grid_size[1], self.y + self.radius)
        
        # Check each pixel within the bounding box to see if it's within the astrocyte's radius
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                # Calculate the distance from the center of the astrocyte to this pixel
                distance = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
                # If the distance is less than the radius, the pixel is within the astrocyte
                if distance <= self.radius:
                    pixels.append((x, y))
        
        return pixels
    
    def remove_molecules_from_cell_body(self):
        
        for pixel in self.pixels:
            for molecule in self.environment_grid.get_tracked_molecules():
                self.environment_grid.set_concentration_at(pixel, molecule=molecule, concentration=0.0)
    
    def release_repellent(self):
        for pixel in self.pixels:
            self.environment_grid.set_concentration_at(pixel, molecule="repellent",
                                                       concentration=self.repellent_concentration)
    
    def get_concentration(self, molecule: str):
        return self.molecules[molecule]
    
    def update_concentration(self, molecule: str, amount: float):
        # cell body acts as infinite sink
        pass
    
    def spawn_initial_branches(self, num_branches: int, max_branch_radius: float, spawn_radius: float,
                               spawn_length: float):
        """
        Spawn initial branches for the astrocyte.

        Args:
            num_branches: Number of branches to spawn.
            max_branch_radius: Maximum radius for a branch.
            spawn_radius: The radius at which the branches spawn from the cell body.
            spawn_length: The length of the branch from the starting point.
        """
        # Calculate the angle between each branch
        angle_increment = 2 * np.pi / num_branches
        
        for i in range(num_branches):
            angle = i * angle_increment
            
            # Choose a random location on the boundary (x, y, radius)
            start_x = self.x + np.cos(angle) * self.radius
            start_y = self.y + np.sin(angle) * self.radius
            start = AstrocyteNode(start_x, start_y, max_branch_radius)
            
            # Set the end point perpendicular to the center of the astrocyte with 'spawn_length' and 'spawn_radius'
            end_x = start_x + np.cos(angle) * spawn_length
            end_y = start_y + np.sin(angle) * spawn_length
            end = AstrocyteNode(end_x, end_y, spawn_radius)
            
            # Create the new branch
            new_branch = AstrocyteBranch(parent=self, nucleus=self, start=start, end=end, max_history=self.max_history)
            self.children.append(new_branch)
            self.branches.append(new_branch)
            
            # Add the new branch to the spatial index
            self.spatial_index.insert(new_branch)
    
    def plot(self, figsize=(5, 5), ax: plt.Axes = None):
        
        # create figure
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # plot cell body
        cell_body = Circle((self.x, self.y), self.radius, color='blue', fill=False)
        ax.add_patch(cell_body)
        
        # plot branches
        for branch in self.branches:
            x0, y0 = branch.start.x, branch.start.y
            x1, y1 = branch.end.x, branch.end.y
            
            ax.plot([x0, x1], [y0, y1], color="black")


class AstrocyteNode:
    
    def __init__(self, x, y, radius):
        self.radius = radius
        self.x = int(x)
        self.y = int(y)
    
    def copy(self):
        return AstrocyteNode(self.x, self.y, self.radius)


class RtreeSpatialIndex:
    def __init__(self):
        # Create an R-tree index
        self.rtree = index.Index()
        self.branch_counter = 0
    
    def search(self, region: Union[Tuple[int, int, int, int], AstrocyteBranch]):
        """
        Search for branches intersecting with the given region.

        Args:
            region: The region to search in (xmin, ymin, xmax, ymax).

        Returns:
            A list of branch IDs that intersect with the region.
        """
        
        if isinstance(region, AstrocyteBranch):
            region = self._get_bbox(region)
        
        return list(self.rtree.intersection(region))
    
    def insert(self, branch):
        """
        Insert a new branch into the R-tree.

        Args:
            branch: The branch to insert.
        """
        bbox = self._get_bbox(branch)
        branch.id = self.branch_counter
        self.rtree.insert(branch.id, bbox)
        
        self.branch_counter += 1
    
    def remove(self, branch):
        """
        Remove a branch from the R-tree.

        Args:
            branch: The branch to remove.
        """
        self.rtree.delete(branch.id, self._get_bbox(branch))
    
    def update(self, branch):
        """
        Update a branch in the R-tree.

        Args:
            branch: The branch to update.
        """
        self.remove(branch)
        self.insert(branch)
    
    @staticmethod
    def _get_bbox(branch):
        """
        Get the bounding box of a branch.

        Args:
            branch: The branch to get the bounding box for.

        Returns:
            The bounding box (xmin, ymin, xmax, ymax).
        """
        xmin = min(branch.start.x, branch.end.x)
        ymin = min(branch.start.y, branch.end.y)
        xmax = max(branch.start.x, branch.end.x)
        ymax = max(branch.start.y, branch.end.y)
        return xmin, ymin, xmax, ymax
    
    @staticmethod
    def plot_line_low(x0, y0, x1, y1):
        """
        Helper function for Bresenham's algorithm for lines with absolute slope less than 1.
        """
        points = []
        dx = x1 - x0
        dy = y1 - y0
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        D = (2 * dy) - dx
        y = y0
        
        for x in range(x0, x1 + 1):
            points.append((x, y))
            if D > 0:
                y += yi
                D += (2 * (dy - dx))
            else:
                D += 2 * dy
        return points
    
    @staticmethod
    def plot_line_high(x0, y0, x1, y1):
        """
        Helper function for Bresenham's algorithm for lines with absolute slope greater than 1.
        """
        points = []
        dx = x1 - x0
        dy = y1 - y0
        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        D = (2 * dx) - dy
        x = x0
        
        for y in range(y0, y1 + 1):
            points.append((x, y))
            if D > 0:
                x += xi
                D += (2 * (dx - dy))
            else:
                D += 2 * dx
        return points
    
    def rasterize_line(self, x0, y0, x1, y1):
        """
        Rasterize a line using Bresenham's line algorithm.

        Args:
            x0, y0: The starting point of the line.
            x1, y1: The ending point of the line.

        Returns:
            A list of grid cells (x, y) that the line intersects.
        """
        if abs(y1 - y0) < abs(x1 - x0):
            if x0 > x1:
                return self.plot_line_low(x1, y1, x0, y0)
            else:
                return self.plot_line_low(x0, y0, x1, y1)
        else:
            if y0 > y1:
                return self.plot_line_high(x1, y1, x0, y0)
            else:
                return self.plot_line_high(x0, y0, x1, y1)


class AstrocyteBranch:
    
    interacting_pixels = None
    volume = None
    surface_area = None
    repellent_release = None
    environment = None
    diffusion_rate = None
    id = 0
    
    def __init__(self, parent, nucleus: Astrocyte, start: Union[Tuple[int, int, int], AstrocyteNode],
                 end: Union[Tuple[int, int, int], AstrocyteNode], max_history=100):
        
        self.min_trend_amplitude = None  # minimum trend in ATP and glutamate to grow or shrink
        self.min_steepness = None  # minimum steepness for spawning
        self.atp_cost_per_glutamate = None  # mol ATP / mol Glutamate
        self.diffusion_coefficient = None
        self.glutamate_uptake_rate = None  # mol/m² --> same as surface factor?
        self.direction_threshold = None
        self.spatial_index = None
        self.spawn_length = None  # m
        self.spawn_radius_factor = None  # Relative proportion of end point compared to start point
        self.min_radius = None  # m
        self.atp_cost_per_unit_surface = None  # mol/m²
        self.growth_factor = None
        self.volume_factor = None  # mol/m³
        self.surface_factor = None  # mol/m²
        
        self.parent: AstrocyteBranch = parent
        self.nucleus = nucleus
        self.children: List[AstrocyteBranch] = []
        self.start: AstrocyteNode = AstrocyteNode(*start) if isinstance(start, tuple) else start
        self.end: AstrocyteNode = AstrocyteNode(*end) if isinstance(end, tuple) else end
        
        # Establish initial concentrations
        self.get_environment()
        self.molecules = {"glutamate": 0.0, "calcium": 0.0, "ATP": 0.0}
        
        self.intracellular_history = {molecule: deque(maxlen=max_history) for molecule in self.molecules}
        self.extracellular_history = {molecule: deque(maxlen=max_history) for molecule in self.environment}
        
        self.update_physical_properties()
    
    def step(self):
        
        # update environment
        self.get_environment()
        
        # simulate flow of molecules
        self._simulate_calcium()
        self._simulate_glutamate()
        self._simulate_repellent()
        
        # run through actions
        self._act()
        
        # simulate molecule diffusion
        self.diffuse_molecules()
        
        # save new state
        self._update_history()
    
    def _update_history(self):
        """
        Update the intracellular and extracellular history of molecule concentrations.
        Each history dictionary keeps track of the concentrations in a rolling fashion,
        storing at most 'max_history' values and dropping the oldest values.
        """
        # Update intracellular history
        for molecule, concentration in self.molecules.items():
            self.intracellular_history[molecule].append(concentration)
        
        # Update extracellular history
        for molecule, concentration in self.environment.items():
            self.extracellular_history[molecule].append(concentration)
    
    def get_trend(self, molecule: str, intra: bool) -> float:
        """
        Perform linear regression on the history of a molecule's concentration.

        Args:
            molecule: Name of the molecule.
            intra: True for intracellular history, False for extracellular history.

        Returns:
            Slope (m) of the linear regression line, representing the trend.
        """
        history = self.intracellular_history[molecule] if intra else self.extracellular_history[molecule]
        if not history:
            return 0.0  # No trend if history is empty
        history = np.array(history)
        
        # Create an array of time points (assuming equal time intervals)
        x = np.arange(len(history))
        
        # Perform linear regression
        slope, _ = np.polyfit(x, history, 1)
        
        return slope
    
    def update_concentration(self, molecule, amount):
        if molecule not in self.molecules:
            raise ValueError(f"Unknown molecule {molecule}")
        
        self.molecules[molecule] += amount
    
    def get_concentration(self, molecule):
        if molecule not in self.molecules:
            raise ValueError(f"Unknown molecule {molecule}")
        
        return self.molecules[molecule]
    
    def get_amount(self, molecule):
        return self.volume * self.get_concentration(molecule)
    
    def set_concentration(self, molecule, concentration):
        if molecule not in self.molecules:
            raise ValueError(f"Unknown molecule {molecule}")
        
        self.molecules[molecule] = concentration
    
    def get_interacting_pixels(self) -> List[Tuple[int, int]]:
        """
        Get the grid cells (pixels) that are intersected by the current branch.

        Returns:
            A list of grid cells (x, y) that the branch intersects.
        """
        # Use the rasterize_line method from the RtreeSpatialIndex to get the pixels
        interacting_pixels = self.nucleus.spatial_index.rasterize_line(self.start.x, self.start.y, self.end.x,
                                                                       self.end.y)
        return interacting_pixels
    
    def get_environment(self):
        
        env_grid = self.nucleus.environment_grid
        
        total_amount = {}
        for molecule in env_grid.get_tracked_molecules():
            total_amount[molecule] = np.sum(env_grid.get_amount_at(self.interacting_pixels, molecule))
        
        self.environment = total_amount
    
    def update_environment(self, molecule, amount):
        return self.nucleus.environment_grid.update_concentration_at(self.interacting_pixels, molecule, amount)
    
    def _calculate_diffusion_rate(self) -> float:
        """
        Calculate the diffusion rate of ions along the branch.

        Returns:
            The diffusion rate for the branch.
        """
        # Calculate the average radius of the branch
        average_radius = (self.start.radius + self.end.radius) / 2
        
        # Calculate the length of the branch
        length_of_branch = np.sqrt((self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2)
        
        # Apply the formula for diffusion rate
        diffusion_rate = self.diffusion_coefficient * (1 / length_of_branch) * (1 / (np.pi * average_radius ** 2))
        
        return diffusion_rate
    
    def diffuse_molecules(self):
        
        for molecule, concentration in self.molecules.items():
            for target in [self.parent] + self.children:
                
                # Calculate the concentration difference between the branch and the target
                concentration_difference = abs(self.get_concentration(molecule) - target.get_concentration(molecule))
                
                if concentration_difference == 0:
                    continue
                
                # calculate ions/molecules to move
                dt = 1  # one time step
                ions_to_move = self.diffusion_rate * concentration_difference * dt
                
                # update new concentrations
                self.update_concentration(molecule, - ions_to_move)
                target.update_concentration(molecule, ions_to_move)
    
    def _simulate_calcium(self):
        # todo: implement
        pass
    
    def _simulate_glutamate(self):
        
        glutamate_removal_capacity = self.calculate_removal_capacity(self.glutamate_uptake_rate)
        
        # remove from environment
        actually_removed = self.update_environment("glutamate", glutamate_removal_capacity)
        
        # increase intracellular concentration with actually removed concentration
        self.update_concentration("glutamate", actually_removed)
        
        # convert glutamate using up ATP
        atp_needed = self.get_amount("glutamate") * self.atp_cost_per_glutamate
        atp_used = min(self.get_amount("ATP"), atp_needed)
        self.update_concentration("ATP", -atp_used)
        self.update_concentration("glutamate", int(atp_used / self.atp_cost_per_glutamate))
    
    def _simulate_repellent(self):
        # Release repellent into environment
        self.update_environment("repellent", self.repellent_release)
    
    def calculate_branch_volume(self):
        # Calculate the Euclidean distance between the start and end nodes to get the height of the truncated cone
        h = np.sqrt((self.end.x - self.start.x) ** 2 +
                    (self.end.y - self.start.y) ** 2)
        # Use the radii of the start and end nodes
        r1, r2 = self.start.radius, self.end.radius
        
        # Volume of a truncated cone
        volume = (1 / 3) * np.pi * h * (r1 ** 2 + r1 * r2 + r2 ** 2)
        return volume
    
    def calculate_branch_surface(self, start: AstrocyteNode = None, end: AstrocyteNode = None):
        
        start = self.start if start is None else start
        end = self.end if end is None else end
        
        # Calculate the Euclidean distance between the start and end nodes to get the slant height of the truncated cone
        h = np.sqrt((end.x - start.x) ** 2 +
                    (end.y - start.y) ** 2)
        
        # Use the radii of the start and end nodes
        r1, r2 = start.radius, end.radius
        
        # Lateral surface area of a truncated cone
        # This calculation assumes that 'h' is the slant height of the cone's lateral surface
        slant_height = np.sqrt((r2 - r1) ** 2 + h ** 2)
        surface = np.pi * (r1 + r2) * slant_height
        return surface
    
    def update_physical_properties(self):
        self.interacting_pixels = self.get_interacting_pixels()
        self.volume = self.calculate_branch_volume()
        self.surface_area = self.calculate_branch_surface()
        self.diffusion_rate = self._calculate_diffusion_rate()
        self.repellent_release = self.calculate_repellent_release(self.surface_factor, self.volume_factor)
    
    def calculate_removal_capacity(self, uptake_rate: float) -> float:
        """
        Calculate the glutamate removal capacity of the branch.

        Args:
            uptake_rate: The rate of uptake per unit surface area, representing channel density or efficiency.

        Returns:
            The glutamate removal capacity of the branch.
        """
        
        # Calculate the removal capacity as the product of surface area and uptake rate
        removal_capacity = self.surface_area * uptake_rate
        
        return removal_capacity
    
    def calculate_repellent_release(self, surface_factor: float, volume_factor: float) -> float:
        """
        Calculate the repellent release of the branch based on its geometry.

        Args:
            surface_factor: The factor that determines how much the surface area contributes to repellent release.
            volume_factor: The factor that determines how much the volume contributes to repellent release.

        Returns:
            The repellent release of the branch.
        """
        
        # Calculate the repellent release as a weighted sum of surface area and volume contributions
        repellent_release = (self.surface_area * surface_factor) + (self.volume * volume_factor)
        
        return repellent_release
    
    def _act(self):
        
        # we prune automatically if the end radius drops below min_radius
        self._action_grow_or_shrink(self.growth_factor, self.min_trend_amplitude,
                                    self.atp_cost_per_unit_surface, self.min_radius)
        
        self._action_spawn_or_move(self.spawn_radius_factor, self.spawn_length, self.min_steepness,
                                   self.direction_threshold)
    
    def _action_grow_or_shrink(self, growth_factor: float, min_trend_amplitude: float,
                               atp_cost_per_unit_surface: float, min_radius: float):
        """
        Grow or shrink the branch by adjusting the radius of the end node.

        .. note::
        
            | Glutamate vs ATP | Low | Medium | High |
            |-----------------|-----|--------|------|
            | High            | MIX | GROW    | GROW  |
            | Medium          |  SHRINK  | -    | GROW  |
            | Low             | SHRINK | SHRINK | MIX   |

        Args:
            growth_factor: The factor determining how much the node grows or shrinks.
            atp_cost_per_unit_surface: The ATP cost (or gain, if negative) for each unit of surface area change.
            min_radius: The minimum allowed radius of the end node to prevent over-shrinkage.
        """
        
        atp_trend = self.get_trend("ATP", intra=True)
        glutamate_trend = self.get_trend("glutamate", intra=False)
        combined_trend = glutamate_trend - atp_trend
        
        if abs(combined_trend) < min_trend_amplitude:
            return
        
        if combined_trend < 0:
            growth_factor *= -1
        
        # Calculate the new surface area after growth/shrinkage
        new_end = self.end.copy()
        new_end.radius *= growth_factor
        
        if new_end.radius < min_radius:
            self._action_prune()
            return
        
        new_surface_area = self.calculate_branch_surface(start=self.start, end=new_end)
        
        # Ensure the end node radius is not less than the start node radius after growth/shrinkage
        if new_end.radius <= self.start.radius:
            logging.info("Growth constrained by start node size.")
            return
        
        # Calculate the required ATP based on the change in surface area
        delta_surface = new_surface_area - self.surface_area
        required_atp = delta_surface * atp_cost_per_unit_surface
        
        # If shrinking, ATP is released (required_atp will be negative)
        available_atp = self.get_amount("ATP")
        
        # Check if there's enough ATP to support the growth, or if ATP needs to be added back for shrinkage
        if growth_factor > 0 and available_atp < required_atp:
            logging.info("Not enough ATP to support growth.")
            return
        
        # Update new end node
        self.end = new_end
        
        # Update the ATP concentration in the branch
        if growth_factor > 0:
            self.update_concentration("ATP", -required_atp)
        
        # Update the physical properties of the branch (e.g., recalculate volume, surface area)
        self.update_physical_properties()
    
    def _action_spawn_or_move(self, spawn_radius_factor: float, spawn_length: float, min_steepness: float,
                              direction_threshold: float):
        """
        Spawn a new branch or move the current branch based on the environmental factors.

        If the direction of growth does not vary too much from the current direction and the branch has no children,
        the branch will move. Otherwise, a new branch will be spawned.

        Args:
            spawn_radius_factor: The radius factor for the new branch.
            spawn_length: The length of the new branch or movement.
            min_steepness: Minimum steepness of the combined gradient (glutamate and repellent) that triggers
                spawning of a new branch or movement.
            direction_threshold: The threshold for how much the new direction can vary from the current direction.
        """
        
        # todo: collision control
        
        # Calculate the direction of the new branch based on glutamate and repellent gradients
        direction, steepness = self._calculate_spawn_direction()
        
        if direction is not None and steepness > min_steepness:
            if len(self.children) > 0 or np.linalg.norm(direction) > direction_threshold:
                # If direction varies too much or branch has children, spawn a new branch
                self._spawn_new_branch(spawn_radius_factor, spawn_length, direction, self.spatial_index)
            else:
                # If direction is similar and branch has no children, move the branch
                self._move_branch(spawn_length, direction)
    
    def _action_prune(self):
        """
        Prune the branch if it has no children.
        """
        # Ensure no children exist; else skip
        if self.children:
            logging.warning("Branch has children, cannot prune.")
            return
        
        # Remove self from spatialIndexTree
        self.spatial_index.remove(self)
        
        # Delete self from parent
        self.parent.children.remove(self)
        self.nucleus.branches.remove(self)
        
        # Additional cleanup if needed (e.g., freeing resources or nullifying references)
        self.children = []
        self.start = None
        self.end = None
        
        logging.info("Branch pruned successfully.")
    
    def _spawn_new_branch(self, radius_factor: float, length_factor: float, direction: Tuple[float, float],
                          spatial_index: RtreeSpatialIndex, atp_cost_per_unit_surface: float = None):
        """
        Spawn a new branch from the current branch.

        Args:
            radius_factor: The radius factor for the new branch.
            length_factor: The length of the new branch.
            direction: The direction for the new branch.
            spatial_index: The spatial index for managing branches.
        """
        # Determine the starting point and end point of the new branch
        start_point = self.end  # New branch starts where the current branch ends
        end_point = AstrocyteNode(
                x=int(start_point.x + direction[0] * length_factor),
                y=int(start_point.y + direction[1] * length_factor),
                radius=start_point.radius * radius_factor
                )
        
        # Create the new branch
        new_branch = AstrocyteBranch(parent=self, start=start_point, end=end_point, nucleus=self.nucleus)
        
        # calculate cost
        atp_cost = atp_cost_per_unit_surface * new_branch.surface_area
        if atp_cost <= self.get_amount("ATP"):
            
            # Save the new branch to the list of children
            self.children.append(new_branch)
            self.nucleus.branches.append(new_branch)
            
            # Update the spatial index with the new branch
            spatial_index.insert(new_branch)
            
            # remove atp
            self.update_concentration("ATP", atp_cost)
        
        else:
            logging.info(f"Insufficient ATP available.")
    
    def _move_branch(self, spawn_length: float, direction: Tuple[float, float]):
        """
        Move the current branch in a specified direction.

        Args:
            spawn_length: The length of the movement.
            direction: The direction for the movement.
        """
        # Calculate new end point
        new_end_position = (
            int(self.start.x + direction[0] * spawn_length),
            int(self.start.y + direction[1] * spawn_length),
            self.start.radius  # Assuming radius remains constant during movement
            )
        
        # Update the end node
        self.end.x, self.end.y = new_end_position[:2]
        
        # Update the spatial index before changing the position
        self.nucleus.spatial_index.update(self)
        
        # Update the physical properties of the branch
        self.update_physical_properties()
    
    def _calculate_spawn_direction(self, repellent_name='repellent'):
        """
        Calculate the direction for spawning a new branch based on environmental factors.

        Args:
            repellent_name: The name of the repellent substance in the environment grid.

        Returns:
            A tuple representing the direction vector (dx, dy).
        """
        environment_grid = self.nucleus.environment_grid
        
        # Get the position of the current branch's end node
        x, y = self.end.x, self.end.y
        
        # Define the range to look around the end node for gradient calculation
        range_x = range(max(0, x - 1), min(self.nucleus.environment_grid.grid_size[0], x + 2))
        range_y = range(max(0, y - 1), min(self.nucleus.environment_grid.grid_size[1], y + 2))
        
        # Get the concentrations of glutamate and repellent around the end node
        glutamate_concentration = environment_grid.get_concentration_at((x, y), 'glutamate')
        repellent_concentration = environment_grid.get_concentration_at((x, y), repellent_name)
        
        # Initialize variables to store gradient sums
        gradient_sum_glutamate = np.array([0.0, 0.0])
        gradient_sum_repellent = np.array([0.0, 0.0])
        
        # Calculate the sum of gradients for glutamate and repellent
        for i in range_x:
            for j in range_y:
                if (i, j) != (x, y):
                    direction = np.array([i - x, j - y])
                    distance = np.linalg.norm(direction)
                    direction_normalized = direction / distance
                    
                    diff_glu = environment_grid.get_concentration_at((i, j), 'glutamate') - glutamate_concentration
                    diff_rep = environment_grid.get_concentration_at((i, j), repellent_name) - repellent_concentration
                    
                    gradient_sum_glutamate += direction_normalized * diff_glu
                    gradient_sum_repellent += direction_normalized * diff_rep
        
        # Combine gradients: attract towards glutamate, repel from repellent
        combined_gradient = gradient_sum_glutamate - gradient_sum_repellent
        
        # Normalize the combined gradient to get a unit direction vector
        if np.linalg.norm(combined_gradient) > 0:
            steepness = np.linalg.norm(combined_gradient)
            direction_vector = combined_gradient / steepness
            return tuple(direction_vector), steepness
        else:
            # If the gradient is zero (no preference) return None
            return None


class DataLogger:
    def __init__(self):
        self.logged_data = []
    
    def log_data(self, data):
        pass
    
    def retrieve_historical_data(self):
        pass
    
    def export_data(self):
        pass
