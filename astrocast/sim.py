from __future__ import annotations

import logging
from multiprocessing import shared_memory
from typing import Callable, List, Tuple, Union

import numpy as np
import seaborn as sns
from IPython.core.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from rtree import index


class EnvironmentGrid:
    def __init__(self, grid_size: Tuple[int, int], diffusion_rate: float, dt: float,
                 molecules: List[str] = ["glutamate", "calcium", "repellent"], dtype: str = 'float16'):
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
                                concentration_change: float):
        """
        Update the concentration of a specific molecule at a given location or multiple locations.

        Args:
            location: A single tuple (x, y) representing the grid coordinates or a list of such tuples.
            molecule: Name of the molecule.
            concentration_change: Amount to increment the concentration by at each location.
        """
        if molecule not in self.shared_arrays:
            raise ValueError(f"Molecule {molecule} not found in the grid.")
        
        # If a single location is provided, wrap it in a list.
        if isinstance(location, tuple):
            location = [location]
        
        # evenly distribute concentration across pixels
        concentration_change /= len(location)
        
        # Update the concentration at each location by the specified amount.
        for x, y in location:
            
            self.shared_arrays[molecule][0][x, y] += concentration_change
            
            if self.shared_arrays[molecule][0][x, y] < 0:
                self.shared_arrays[molecule][0][x, y] = 0
    
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
    
    def __init__(self, environment_grid: EnvironmentGrid):
        self.environment_grid = environment_grid
        self.spatial_index = RtreeSpatialIndex()
        self.branches = []
    
    def manage_growth(self):
        pass
    
    def interact_with_environment(self, environment_grid):
        pass


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
    history = None
    
    def __init__(self, parent, nucleus: Astrocyte, start: Union[Tuple[int, int, int], AstrocyteNode],
                 end: Union[Tuple[int, int, int], AstrocyteNode], branch_id: int = 0):
        
        self.direction_threshold = None
        self.spatial_index = None
        self.spawn_length = None
        self.spawn_radius = None
        self.min_radius = None
        self.atp_cost_per_unit_surface = None
        self.growth_factor = None
        self.volume_factor = None
        self.surface_factor = None
        
        self.parent: AstrocyteBranch = parent
        self.nucleus = nucleus
        self.id = branch_id
        self.children: List[AstrocyteBranch] = []
        self.start: AstrocyteNode = AstrocyteNode(*start) if isinstance(start, tuple) else start
        self.end: AstrocyteNode = AstrocyteNode(*end) if isinstance(end, tuple) else end
        
        self.molecules = {"glutamate": 0.0, "calcium": 0.0, "ATP": 0.0}
        
        self.update_physical_properties()
    
    def step(self):
        
        # simulate flow of molecules
        self._simulate_calcium()
        self._simulate_atp()
        self._simulate_glutamate()
        self._simulate_repellent()
        
        # run through actions
        self._act()
        
        # save new state
        self.update_history()
    
    def update_history(self):
        # todo: implement
        
        pass
    
    def update_concentration(self, molecule, concentration):
        if molecule not in self.molecules:
            raise ValueError(f"Unknown molecule {molecule}")
        
        self.molecules[molecule] += concentration
    
    def get_concentration(self, molecule):
        if molecule not in self.molecules:
            raise ValueError(f"Unknown molecule {molecule}")
        
        return self.molecules[molecule]
    
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
        
        total_concentration = {}
        for molecule in env_grid.get_tracked_molecules():
            total_concentration[molecule] = np.sum(env_grid.get_concentration_at(self.interacting_pixels, molecule))
        
        return total_concentration
    
    def update_environment(self, molecule, concentration):
        self.nucleus.environment_grid.update_concentration_at(self.interacting_pixels, molecule, concentration)
    
    def _simulate_calcium(self):
        # todo: implement
        pass
    
    def _simulate_atp(self):
        # Get ATP concentration from the parent and child, if they exist
        parent_atp = self.parent.get_concentration("ATP") if self.parent else 0
        child_atp = min(child.get_concentration("ATP") for child in self.children) if self.children else 0
        # todo: Move ATP based on concentration gradient
        # This is a placeholder; the actual movement logic needs to be implemented based on your model
        # self.set_concentration("ATP", (parent_atp + child_atp) / 2)
    
    def _simulate_glutamate(self):
        # todo: Calculate the capacity for glutamate removal based on the branch's volume
        # This is a placeholder for the capacity calculation
        glutamate_removal_capacity = self.calculate_removal_capacity()
        
        # todo: Remove glutamate from the external environment
        # This is a placeholder; the actual removal logic needs to be implemented based on your model
        environmental_glutamate = self.get_environment()["glutamate"]
        glutamate_to_remove = min(glutamate_removal_capacity, environmental_glutamate)
        self.update_environment("glutamate", -glutamate_to_remove)
        
        # todo: decide whether or not to keep track of internal glutamate
        
        # todo: add ATP cost of converting glutamate
    
    def _simulate_repellent(self):
        # Release repellent into environment, assuming a fixed concentration for demonstration purposes
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
        self._action_grow_or_shrink(self.growth_factor, self.atp_cost_per_unit_surface, self.min_radius)
        
        self._action_spawn_or_move(self.spawn_radius, self.spawn_length, self.nucleus.environment_grid,
                                   self.spatial_index, self.direction_threshold)
    
    def _action_grow_or_shrink(self, growth_factor: float, atp_cost_per_unit_surface: float, min_radius: float):
        """
        Grow or shrink the branch by adjusting the radius of the end node.

        Args:
            growth_factor: The factor determining how much the node grows or shrinks.
            atp_cost_per_unit_surface: The ATP cost (or gain, if negative) for each unit of surface area change.
            min_radius: The minimum allowed radius of the end node to prevent over-shrinkage.
        """
        
        # todo: when would we want to grow?
        #  - glutamate cannot be removed efficiently
        #  - more ATP required (eg, history of ATP in children is decreasing
        #  - what if converting glutamate takes ATP?
        # todo: when would we want to shrink the branch?
        #  - not using enough ATP?
        
        # Calculate the new surface area after growth/shrinkage
        new_end = self.end.copy()
        new_end.radius += growth_factor
        
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
        available_atp = self.get_concentration("ATP")
        
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
    
    def _action_spawn_or_move(self, spawn_radius: float, spawn_length: float, environment_grid: EnvironmentGrid,
                              spatial_index: RtreeSpatialIndex, direction_threshold: float):
        """
        Spawn a new branch or move the current branch based on the environmental factors.

        If the direction of growth does not vary too much from the current direction and the branch has no children,
        the branch will move. Otherwise, a new branch will be spawned.

        Args:
            spawn_radius: The radius factor for the new branch.
            spawn_length: The length of the new branch or movement.
            environment_grid: The grid that represents the environment.
            spatial_index: The spatial index for managing branches.
            direction_threshold: The threshold for how much the new direction can vary from the current direction.
        """
        
        # todo: when would we want to move or spawn?
        #  - calculate the gradient for glutamate and repellent
        #  - if only one direction, that is consistent with current direction, move further along
        #  - if competing directions, spawn new child
        #  - there should be a cutoff in gradient steepness; only adjust if steep enough
        
        # Calculate the direction of the new branch based on glutamate and repellent gradients
        direction = self._calculate_spawn_direction(environment_grid)
        
        if len(self.children) > 0 or np.linalg.norm(direction) > direction_threshold:
            # If direction varies too much or branch has children, spawn a new branch
            self._spawn_new_branch(spawn_radius, spawn_length, direction, spatial_index)
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
        if self.parent:
            self.parent.children.remove(self)
        
        # Additional cleanup if needed (e.g., freeing resources or nullifying references)
        self.parent = None
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
        if atp_cost <= self.get_concentration("ATP"):
            
            # Save the new branch to the list of children
            self.children.append(new_branch)
            
            # Update the spatial index with the new branch
            spatial_index.insert(new_branch)
            
            # remove atp
            self.update_concentration("ATP", atp_cost)
    
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
    
    def _calculate_spawn_direction(self, environment_grid: EnvironmentGrid) -> Tuple[float, float]:
        """
        Calculate the direction for spawning a new branch based on environmental factors.

        Args:
            environment_grid: The grid that represents the environment.

        Returns:
            A tuple representing the direction vector (dx, dy).
        """
        # todo: Placeholder for actual direction calculation based on glutamate and repellent gradients
        # This will require accessing the environment grid and potentially performing calculations
        # to determine the gradient direction
        # For now, we return a random direction for demonstration purposes
        dx = np.random.uniform(-1, 1)
        dy = np.random.uniform(-1, 1)
        norm = np.sqrt(dx ** 2 + dy ** 2)
        return dx / norm, dy / norm


class DataLogger:
    def __init__(self):
        self.logged_data = []
    
    def log_data(self, data):
        pass
    
    def retrieve_historical_data(self):
        pass
    
    def export_data(self):
        pass
