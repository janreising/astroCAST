from __future__ import annotations

import logging
import shutil
import time
import uuid
from datetime import datetime
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable, List, Literal, Tuple, Union

import dill as pickle
import humanize
import numpy as np
import seaborn as sns
import xxhash
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from rtree import index
from scipy.ndimage import convolve, gaussian_filter
from tqdm import tqdm


class Loggable:
    def __init__(self, data_logger, message_logger=None, settings: dict = None):
        self.id = uuid.uuid4()
        self.steps = 0
        
        # register with data logger
        self.data_logger = data_logger
        self.data_logger.register(self, settings)
        
        # register with message_logger
        self.message_logger = message_logger
    
    @staticmethod
    def extract_settings(locals_dict, exclude_types=None):
        # Create a copy of the dict to avoid modifying the original
        settings = dict(locals_dict)
        
        # Remove 'self' and any other keys that are not settings
        settings.pop('self', None)
        
        # Exclude parameters that are instances of specific types
        for key, value in list(settings.items()):
            
            exclude = False
            if isinstance(value, (DataLogger, Simulation)) or callable(value) or "func" in key or "_param" in key:
                exclude = True
            
            if exclude_types is not None:
                if isinstance(value, exclude_types):
                    exclude = True
            
            if exclude:
                settings.pop(key)
        
        return settings
    
    def get_hex_id(self):
        return self.id.hex
    
    def get_short_id(self):
        return xxhash.xxh32_hexdigest(self.id.hex)
    
    def log(self, msg: str, level: int = logging.INFO, tag: str = "default"):
        
        if self.message_logger is not None:
            self.message_logger.log(msg=msg, tag=tag, level=level, caller_id=self.id)
        else:
            msg = f"{self.id.hex}:{tag}:{msg}"
            print(msg)
    
    def log_state(self):
        """Override this method in the subclass to return the specific state of the object."""
        raise NotImplementedError("The log_state method must be implemented by the subclass.")


class Visualization:
    
    def __init__(self, simulation: Simulation, dpi: int = 160,
                 display_interval: int = 1, save_interval: int = 1,
                 img_folder: Path = Path("./imgs"), override=False):
        
        self.sim = simulation
        self.X, self.Y = simulation.grid_size
        self.dpi = dpi
        self.display_interval = display_interval
        self.save_interval = save_interval
        
        self.figures = {}
        
        # sanity check for image folder
        if img_folder is not None and img_folder.exists():
            
            if override:
                shutil.rmtree(img_folder)
            else:
                img_folder = img_folder.joinpath(self.sim.get_short_id())
                logging.warning(f"img_folder exists and override is False. Choosing {img_folder}")
        
        self.img_folder = img_folder
        
        self.steps = 0
    
    def step(self, param: dict = None):
        
        if self.steps == 0:
            self.plot_initial_state()
        
        if self.steps % self.display_interval == 0 or self.steps % self.save_interval == 0:
            self.plot(params=param)
        
        self.steps += 1
    
    def save(self, fig, prefix="", force=False):
        
        if self.img_folder is not None and (self.steps % self.save_interval == 0 or force):
            
            if not self.img_folder.exists():
                self.img_folder.mkdir()
            
            save_path = self.img_folder.joinpath(f"{prefix}_{self.steps}.png")
            fig.savefig(save_path.as_posix(), dpi=(self.dpi))
    
    def plot_environment_grid_concentration(self, molecule: str, cmap: str = 'inferno',
                                            figsize: tuple = (5, 5), ax: plt.axis = None):
        """
        Plot the concentration of a specified molecule across the grid using Matplotlib.

        Args:
            molecule: Name of the molecule to plot.
            figsize: Tuple representing the figure size (width, height).
            cmap: Colormap for the heatmap.
            ax: Axes object to plot the heatmap.
        """
        
        env_grid = self.sim.environment_grid
        
        if molecule not in env_grid.molecules:
            logging.error(f"Molecule {molecule} not found in the grid.")
        
        # Retrieve the shared array for the specified molecule
        concentration_array, _ = env_grid.shared_arrays[molecule]
        
        # Create a figure and axis for the heatmap
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            # ax.get_figure().clear()
            ax.clear()
        
        # Plotting the concentration heatmap
        sns.heatmap(concentration_array.transpose(), vmin=0, cmap=cmap, cbar=False, robust=True, ax=ax)
        
        # Adding a colorbar and setting titles and labels
        # plt.colorbar(heatmap, ax=ax)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(f"{molecule} "
                     f"({humanize.metric(np.sum(concentration_array) * env_grid.pixel_volume, 'mol')})")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Display the plot
        return ax
    
    def plot_environment_grid_history(self, molecule, figsize=(5, 5), ax=None):
        
        env_grid = self.sim.environment_grid
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        history = env_grid.history[molecule]
        ax.plot(history, label=molecule)
        ax.legend()
        ax.set_title(f"Total concentration {molecule}")
    
    def plot_dendrites(self, line_thickness: int = 1, line_alpha: float = 0.7,
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
        
        glu_manager = self.sim.glutamate_release_manager
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        color_palette = sns.color_palette("husl", n_colors=glu_manager.num_dendrites)
        for branch_id in range(glu_manager.num_dendrites):
            line = glu_manager.lines[branch_id][:, :-1]
            hotspots = glu_manager.hotspots
            
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
        
        ax.set_xlim(0, self.X)
        ax.set_ylim(0, self.Y)
        
        return ax
    
    def get_bodies_and_branches(self) -> Tuple[List[Astrocyte], List[AstrocyteBranch]]:
        cell_bodies: List[Astrocyte] = []
        branches: List[AstrocyteBranch] = []
        for ast in self.sim.astrocytes:
            cell_bodies.append(ast)
            branches += ast.branches
        
        return cell_bodies, branches
    
    def plot_astrocyte_by_line(self, line_thickness=(0.2, 1.5), line_scaling: Literal['log', 'sqrt'] = 'log',
                               figsize=(5, 5), ax: plt.Axes = None, molecule: str = None,
                               normalize: Literal['max'] = None,
                               cmap=plt.cm.viridis):
        """
        Plot the astrocyte with options to color-code by molecule concentration and simulate imaging blur.

        The function plots the cell body and branches of a neuron. Branch thickness is scaled
        and coloring can be applied based on the concentration of a specified molecule.
        Additionally, a convolution can be applied to simulate the blur effect of a microscope.

        Args:
            line_thickness: Base thickness of the plotted lines.
            line_scaling: Method to scale the line thickness ('log' or 'sqrt').
            figsize: Size of the figure.
            ax: Matplotlib Axes object to plot on. New axes are created if None.
            molecule: Name of the molecule to color-code by its concentration. If None, no color-coding is applied.
            normalize: Value to normalize the molecule concentrations. Can be 'max' or a specific float. If None, no normalization is applied.

        Raises:
            ValueError: If 'normalize' is set to a value other than 'max' or a non-negative number.

        Example:
            neuron.plot(line_thickness=2, line_scaling='log', molecule='serotonin', normalize='max', convolve=True)
        """
        # create figure
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # collect bodies and branches
        cell_bodies, branches = self.get_bodies_and_branches()
        
        # normalize
        if normalize == "max" and molecule is not None:
            
            cb_concentration = np.array([cb.get_concentration(molecule) for cb in cell_bodies])
            b_concentration = np.array([b.get_concentration(molecule) for b in branches])
            
            max_concentration = max(np.max(cb_concentration), np.max(b_concentration))
            cb_concentration /= max_concentration
            b_concentration /= max_concentration
        
        # plot cell bodies
        for i, cb in enumerate(cell_bodies):
            # plot cell body
            cell_body = Circle((cb.x, cb.y), cb.radius, color='blue', fill=False)
            ax.add_patch(cell_body)
        
        # plot branches
        for i, branch in enumerate(branches):
            x0, y0 = branch.start.x, branch.start.y
            x1, y1 = branch.end.x, branch.end.y
            
            width = self._scale_branch_thickness(max_branch_radius=cell_bodies[0].max_branch_radius,
                                                 radius=branch.end.radius,
                                                 line_thickness=line_thickness,
                                                 line_scaling=line_scaling)
            
            # Determine color based on molecule concentration, if applicable
            color = 'black'
            if normalize == "max" and molecule is not None:
                color = cmap(b_concentration[i])
            
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=width)
    
    def plot_astrocyte_by_grid(self, figsize=(5, 5), ax: plt.Axes = None, molecule: str = None,
                               blur_sigma: float = None, blur_radius: int = None,
                               robust: bool = False, cmap='magma'):
        """
        Plot the neuron with options to color-code by molecule concentration and simulate imaging blur.

        The function plots the cell body and branches of a neuron. Branch thickness is scaled
        and coloring can be applied based on the concentration of a specified molecule.
        Additionally, a convolution can be applied to simulate the blur effect of a microscope.

        Args:
            figsize: Size of the figure.
            ax: Matplotlib Axes object to plot on. New axes are created if None.
            molecule: Name of the molecule to color-code by its concentration. If None, no color-coding is applied.
            robust: seaborn flag for robust plotting
            blur_sigma: sigma kernel for blurring
            blur_radius: radius of blurring
            cmap: Color map for heatmap.
        Raises:
            ValueError: If 'normalize' is set to a value other than 'max' or a non-negative number.

        Example:
            neuron.plot(line_thickness=2, line_scaling='log', molecule='serotonin', normalize='max', convolve=True)
        """
        # create figure
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.invert_yaxis()
        
        arr = np.zeros((self.X, self.Y), dtype=float)
        
        # get cell bodies and branches
        cell_bodies, branches = self.get_bodies_and_branches()
        
        # plot cell_body
        for cb in cell_bodies:
            coordinates = cb.get_pixels_within_cell()
            x_coords, y_coords = coordinates[0, :], coordinates[1, :]
            arr[x_coords, y_coords] += cb.get_concentration(molecule)
        
        # plot branches
        for branch in branches:
            coordinates = branch.interacting_pixels
            x_coords, y_coords = coordinates[0, :], coordinates[1, :]
            arr[x_coords, y_coords] += branch.get_concentration(molecule)
        
        # transpose arr
        arr = arr.transpose()
        
        # blur
        if blur_sigma is not None:
            arr = gaussian_filter(arr, sigma=blur_sigma, radius=blur_radius)
        
        # heatmap
        sns.heatmap(arr, robust=robust, norm=LogNorm(), cmap=cmap, cbar=False, square=True,
                    ax=ax)
        
        # title
        ax.set_title(f"[{molecule}] "
                     f"{humanize.metric(np.sum(arr), 'mol')}")
    
    def _scale_branch_thickness(self, radius, max_branch_radius: float,
                                line_thickness: Union[float, Tuple[float, float]],
                                line_scaling: Union[Literal['log', 'sqrt'], None] = 'log'):
        
        # define min and max thickness
        min_width = 0.1
        if isinstance(line_thickness, (tuple, list)):
            min_width, max_width = line_thickness
        else:
            max_width = line_thickness
        
        # scale width
        if line_scaling == 'log':
            
            ast0 = self.sim.astrocytes[0]
            radius_min, radius_max = ast0.min_radius, ast0.max_branch_radius
            
            # Check if the ranges are valid
            if radius_min <= 0 or radius_max <= 0 or min_width < 0 or max_width < 0:
                raise ValueError("Minimum values of the ranges must be positive.")
            
            if not (radius_min <= radius <= radius_max):
                raise ValueError(f"The value {radius} is not within the logarithmic range {(radius_min, radius_max)}.")
            
            # Normalize the value in the logarithmic scale
            log_normalized = (np.log(radius) - np.log(radius_min)) / (np.log(radius_max) - np.log(radius_min))
            
            # Map the normalized value to the linear range
            linear_value = log_normalized * (max_width - min_width) + min_width
            
            return linear_value
        
        elif line_scaling == 'sqrt':
            width = np.sqrt(radius) / np.sqrt(max_branch_radius) * line_thickness
            return width
        
        elif line_scaling is None:
            width = line_thickness
            return width
        
        else:
            raise ValueError('Unknown line scaling use one of ["log", "sqrt"]')
    
    def plot_branch_history(self, branch_id: uuid.UUID):
        
        if isinstance(branch_id, uuid.UUID):
            branch_id = [branch_id]
        
        _, branches = self.get_bodies_and_branches()
        selected_branches = [branch for branch in branches if branch.id in branch_id]
        
        if len(selected_branches) != len(branch_id):
            raise ValueError(f"Incorrect number of branches found ({len(selected_branches)}), "
                             f"expected {len(branch_id)}.")
        
        M = selected_branches[0].intracellular_history.keys()
        fig, axx = plt.subplots(len(M), 1, sharex=True)
        
        for ax in axx:
            ax.set_yscale('log')
        
        colors = sns.color_palette("husl", len(selected_branches))
        for m, branch in enumerate(selected_branches):
            history = branch.intracellular_history
            for i, (molecule, values) in enumerate(history.items()):
                axx[i].plot(values, color=colors[m])
        
        for i, m in enumerate(M):
            axx[i].set_title(m)
    
    def plot_initial_state(self, figsize=(10, 5), force=False):
        
        X, Y = self.X, self.Y
        
        fig, axx = plt.subplot_mosaic("AB", figsize=figsize,
                                      gridspec_kw={
                                          "wspace": 0.2,
                                          "hspace": 0.2}
                                      )
        
        ax = axx['A']
        self.plot_astrocyte_by_line(ax=ax)
        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_aspect('equal')
        ax.set_title('Astrocytes')
        
        ax = axx['B']
        self.plot_dendrites(ax=ax)
        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_aspect('equal')
        
        self.save(fig, prefix="initial_", force=force)
    
    def plot(self, figsize=(15, 10), params=None, axx=None, force=False):
        
        X, Y = self.X, self.Y
        
        if axx is not None:
            fig = axx['A'].get_figure()
        
        elif 'plot' in self.figures:
            
            fig, axx = self.figures['plot']
            for k, ax in axx.items():
                ax.clear()
        
        else:
            fig, axx = plt.subplot_mosaic("AABB\nCDEF", figsize=figsize,
                                          gridspec_kw={
                                              "wspace": 0.2,
                                              "hspace": 0.2}
                                          )
        
        if params is None:
            params = {k: {} for k in axx.keys()}
        else:
            params = {k: params[k] if k in params else {} for k in axx.keys()}
        
        k = 'A'
        self.plot_environment_grid_concentration(molecule="glutamate", ax=axx[k], **params[k])
        
        k = 'B'
        self.plot_environment_grid_history("glutamate", ax=axx[k], **params[k])
        
        for k, mol, grid in [('C', None, False),
                             ('D', 'glutamate', True), ('E', 'ATP', True), ('F', 'CaCyt', True)]:
            ax = axx[k]
            
            if grid:
                self.plot_astrocyte_by_grid(molecule=mol, ax=ax, **params[k])
                
                ax.invert_yaxis()
            else:
                self.plot_astrocyte_by_line(ax=ax, **params[k])
                
                ax.set_xlim(0, X)
                ax.set_ylim(0, Y)
                ax.set_aspect('equal')
        
        fig.suptitle(f"Simulation step {self.steps}")
        
        self.figures['plot'] = fig, axx
        
        self.save(fig, prefix="overview_", force=force)


class RtreeSpatialIndex:
    def __init__(self, simulation):
        # Create an R-tree index
        self.rtree = index.Index()
        self.branch_counter = 0
        self.simulation = simulation
    
    def search(self, region: Union[Tuple[int, int, int, int], AstrocyteBranch]):
        """
        Search for branches intersecting with the given region.

        Args:
            region: The region to search in (xmin, ymin, xmax, ymax).

        Returns:
            A list of branch IDs that intersect with the region.
        """
        
        if isinstance(region, (AstrocyteBranch, Astrocyte)):
            region = region.get_bbox()
        
        return list(self.rtree.intersection(region))
    
    def insert(self, obj):
        """
        Insert a new branch into the R-tree.

        Args:
            obj: The branch to insert.
        """
        bbox = obj.get_bbox()
        id_ = obj.id.int
        self.rtree.insert(id_, bbox)
        
        self.branch_counter += 1
    
    def remove(self, obj):
        """
        Remove a branch from the R-tree.

        Args:
            obj: The branch to remove.
        """
        self.rtree.delete(obj.id.int, obj.get_bbox())
    
    def update(self, obj):
        """
        Update a branch in the R-tree.

        Args:
            obj: The branch to update.
        """
        self.remove(obj)
        self.insert(obj)
    
    def check_collision(self, obj: Union[AstrocyteBranch, Astrocyte, Tuple[int, int, int, int]], border=3):
        
        if not isinstance(obj, tuple):
            bbox = obj.get_bbox()
        else:
            bbox = obj
        
        # check collision with border
        x0, y0, x1, y1 = bbox
        
        X, Y = self.simulation.grid_size
        for x in [x0, x1]:
            if x < border or x > X - border:
                return True
        for y in [y0, y1]:
            if y < border or y > Y - border:
                return True
        
        # check collision with other objects
        collisions = self.search(bbox)
        
        return len(collisions) > 0
    
    @staticmethod
    def plot_line_low(x0, y0, x1, y1):
        """
        Helper function for Bresenham's algorithm for lines with absolute slope less than 1.
        """
        points_x, points_y = [], []
        dx = x1 - x0
        dy = y1 - y0
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        D = (2 * dy) - dx
        y = y0
        
        for x in range(x0, x1 + 1):
            points_x.append(x)
            points_y.append(y)
            
            if D > 0:
                y += yi
                D += (2 * (dy - dx))
            else:
                D += 2 * dy
        return np.array([points_x, points_y])
    
    @staticmethod
    def plot_line_high(x0, y0, x1, y1):
        """
        Helper function for Bresenham's algorithm for lines with absolute slope greater than 1.
        """
        points_x, points_y = [], []
        dx = x1 - x0
        dy = y1 - y0
        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        D = (2 * dx) - dy
        x = x0
        
        for y in range(y0, y1 + 1):
            points_x.append(x)
            points_y.append(y)
            if D > 0:
                x += xi
                D += (2 * (dx - dy))
            else:
                D += 2 * dx
        return np.array([points_x, points_y])
    
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


class EnvironmentGrid(Loggable):
    def __init__(self, simulation: Simulation, grid_size: Tuple[int, int], diffusion_rate: float, dt: float,
                 pixel_volume: float = 1.0, molecules: Union[List[str], str] = "glutamate",
                 degrades: Union[str, List[str]] = None, degradation_factor: float = 0.75, dtype: str = np.float32):
        """
        Initialize the environment grid with shared numpy arrays for each molecule.

        Args:
            grid_size: A tuple representing the dimensions of the grid (NxM).
            molecules: A list of molecule names to be tracked.
            diffusion_rate: Diffusion rate.
            dt: Time step of the simulation.
            dtype: The data type of the arrays.

        """
        
        super().__init__(simulation.data_logger, settings=Loggable.extract_settings(locals()))
        
        if isinstance(molecules, list):
            self.molecules = molecules
        else:
            self.molecules = [molecules]
        
        self.grid_size = grid_size
        self.degrades = degrades
        self.degradation_factor = degradation_factor
        self.pixel_volume = pixel_volume
        self.diffusion_rate = diffusion_rate
        self.dt = dt
        self.history = {molecule: [] for molecule in molecules}
        
        self.shared_arrays = self._create_shared_arrays(grid_size, molecules, dtype=dtype)
        
        self._check_cfl_condition(diffusion_rate=diffusion_rate, dt=dt)
    
    def log_state(self):
        
        state = {
            "shared_arrays":         {molecule: arr[0] for molecule, arr in self.shared_arrays.items()},
            "average_concentration": {molecule: np.mean(arr[0]) for molecule, arr in self.shared_arrays.items()}
            }
        return state
    
    @staticmethod
    def _create_shared_arrays(grid_size: Tuple[int, int], molecules: List[str], dtype=np.float32, parallel=False):
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
            
            if parallel:
                # Create a new shared memory block
                shm = shared_memory.SharedMemory(create=True, size=np.prod(grid_size) * np.dtype(dtype).itemsize)
                # Create a numpy array using the shared memory
                shared_array = np.ndarray(grid_size, dtype=dtype, buffer=shm.buf)
                shared_arrays[molecule] = (shared_array, shm)
            
            else:
                shared_array = np.zeros(grid_size, dtype=dtype)
                shared_arrays[molecule] = (shared_array, None)
        
        return shared_arrays
    
    def _update_concentrations(self):
        """
        Update the concentrations in the grid using a convolution approach with
        Dirichlet boundary conditions (edges as sinks).
        """
        # Define the diffusion kernel
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=float)
        
        for molecule, (shared_array, _) in self.shared_arrays.items():
            for _ in range(int(self.dt)):
                # Apply Dirichlet boundary conditions
                shared_array[0, :] = shared_array[-1, :] = 0
                shared_array[:, 0] = shared_array[:, -1] = 0
                
                # Compute changes using convolution
                change = convolve(shared_array, kernel, mode='constant', cval=0.0)
                
                # Update the shared_array with the changes
                shared_array += self.diffusion_rate * change
                
                if np.min(shared_array) < 0:
                    logging.warning(f"Concentration({molecule}) < 0: {np.min(shared_array)}."
                                    f"Diffusion simulation is unstable, reduce 'diffusion_rate'.")
    
    def _degrade_molecules(self):
        if self.degrades is not None:
            
            if not isinstance(self.degrades, list):
                molecules = [self.degrades]
            else:
                molecules = self.degrades
            
            for molecule in molecules:
                arr, shm = self.shared_arrays[molecule]
                arr *= self.degradation_factor
                self.shared_arrays[molecule] = arr, shm
    
    def step(self):
        """
        Advance the simulation by one time steps.

        This method updates the molecular concentrations in the grid
        based on the diffusion rate, using the vectorized update approach.
        It applies the updates iteratively for the given number of time steps.

        """
        self._update_concentrations()
        self._degrade_molecules()
        
        self.update_history()
        self.steps += 1
    
    def update_history(self):
        total_amounts = self.get_total_amount()
        for molecule in self.molecules:
            self.history[molecule].append(total_amounts[molecule])
    
    def get_total_amount(self):
        
        total_amounts = {}
        for molecule in self.molecules:
            amount = np.sum(self.shared_arrays[molecule][0]) * self.pixel_volume
            total_amounts[molecule] = amount
        
        return total_amounts
    
    def get_tracked_molecules(self) -> List[str]:
        return self.molecules
    
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
            
            if shm is not None:
                shm.close()
                shm.unlink()
    
    def __del__(self):
        """
        Destructor to ensure proper cleanup of shared memory resources.
        This method is automatically called when the object is garbage collected.
        """
        self.close()


class GlutamateReleaseManager(Loggable):
    def __init__(self, simulation: Simulation,
                 num_dendrites: int = 10, num_hotspots: int = 16,
                 z_thickness: float = 3.0, border: int = 3, jitter: float = 0.1,
                 release_amplitude: float = 1,
                 stochastic_probability: float = 0, signal_function: Union[Callable, np.array, float] = lambda x: 0):
        """
        Initialize the GlutamateReleaseManager with dendritic branches intersecting the imaging volume.

        Args:
            simulation: Instance of the Simulation class.
            num_dendrites: Number of dendritic branches to simulate.
            z_thickness: Thickness of the imaging plane in Z-dimension.
            jitter: Amount of jitter in the hotspot placement, default is 0.1.
            release_amplitude: Maximum release amplitude of a glutamate event in mol.
            stochastic_probability: Probability of release at each hotspot.
            signal_function: Function to generate signal-based probabilities.
        """
        
        settings = Loggable.extract_settings(locals())
        super().__init__(simulation.data_logger, message_logger=simulation.message_logger, settings=settings)
        
        self.simulation = simulation
        self.environment_grid = simulation.environment_grid
        self.num_dendrites = num_dendrites
        self.num_hotspots = num_hotspots
        self.z_thickness = z_thickness
        self.border = border
        self.jitter = jitter
        self.release_amplitude = release_amplitude
        self.stochastic_probability = stochastic_probability
        self.signal_function = signal_function
        self.lines, self.hotspots = self._generate_hotspots()
    
    def log_state(self):
        
        if self.steps < 1:
            state = {
                "lines":    self.lines,
                "hotspots": self.hotspots
                }
            return state
        return None
    
    def _generate_hotspots(self) -> Tuple[np.array, np.array]:
        """
        Generate hotspots based on dendritic branches intersecting the imaging volume.

        Returns:
            Tuple of line and hotspot coordinates as np.array.
        """
        hotspots = []
        lines = []
        tries = 0
        branch_id = 0
        while tries < self.simulation.max_tries and len(lines) < self.num_dendrites:
            # Generate a random line (dendritic branch) within the volume
            line = self._generate_random_line()
            
            # Place hotspots along the line with some jitter
            spots = self._place_hotspots_on_line(line, self.num_hotspots, branch_id)
            
            if spots is None:
                tries += 1
            else:
                lines.extend([line])
                hotspots.extend(spots)
                branch_id += 1
                tries = 0
        
        return np.array(lines), np.array(hotspots)
    
    def _generate_random_line(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        # Generate random start and end points for the line within the grid boundaries
        start_point = (np.random.uniform(-self.border, self.environment_grid.grid_size[0] + self.border),
                       np.random.uniform(-self.border, self.environment_grid.grid_size[1] + self.border),
                       np.random.uniform(0, self.z_thickness))
        
        end_point = (np.random.uniform(-self.border, self.environment_grid.grid_size[0] + self.border),
                     np.random.uniform(-self.border, self.environment_grid.grid_size[1] + self.border),
                     np.random.uniform(0, self.z_thickness))
        
        return start_point, end_point
    
    def _place_hotspots_on_line(self, line: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
                                num_hotspots: Union[Tuple[int, int], int] = 16,
                                branch_id: int = 0) -> Union[None, List[Tuple[int, int, int]]]:
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
            
            X, Y = self.environment_grid.grid_size
            if x < self.border or x > X - self.border or y < self.border or y > Y:
                return None
            
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
    
    def step(self):
        """
        Advance the glutamate release simulation by one step.

        Arg:
            time_step: How many steps to advance the glutamate release simulation.

        """
        
        # collect probabilities and combine
        stochastic_vector = self._stochastic_release(self.stochastic_probability)
        signal_vector = self._signal_based_release(self.signal_function)
        combined_prob = stochastic_vector + signal_vector
        
        # calculate release amount
        release_amount = self.release_amplitude * combined_prob
        
        # compare to random vector and set release 0 for failed hotspots
        random_vector = np.random.uniform(0, 1, size=combined_prob.shape)
        release_amount[random_vector > combined_prob] = 0
        
        # update environment
        location = np.array([self.hotspots[:, 0], self.hotspots[:, 1]])
        self.environment_grid.update_amount_at(location=location, molecule="glutamate",
                                               amount=release_amount)
        
        self.steps += 1
    
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
    
    def _signal_based_release(self, signal_function: Union[Callable, np.array, float] = lambda x: 0) -> np.array:
        """
        Generate a probability vector for signal-based release.

        Args:
            signal_function: Function to generate signal-based probabilities.

        Returns:
            A numpy array of signal-based release probabilities for each hotspot.
        """
        
        signal_probability = np.zeros(len(self.hotspots))
        if isinstance(signal_function, Callable):
            signal_probability[:] = signal_function(int(self.steps))
        elif isinstance(signal_function, np.ndarray):
            idx = self.steps % len(signal_function)
            signal_probability[:] = signal_function[idx]
        elif isinstance(signal_function, float):
            signal_probability[:] = signal_function
        else:
            raise ValueError("signal_function must be Callable or np.ndarray")
        
        return signal_probability


class Simulation(Loggable):
    
    def __init__(self, data_logger: DataLogger, num_astrocytes=1, grid_size=(100, 100), border=10,
                 max_tries=5,
                 environment_param: dict = None, glutamate_release_param: dict = None, astrocyte_param: dict = None,
                 ion_flow_param: dict = None, messenger_param: dict = {}, vis_param: dict = None):
        
        super().__init__(data_logger, message_logger=MessageLogger(**messenger_param),
                         settings=Loggable.extract_settings(locals()))
        
        self.max_tries = max_tries
        self.grid_size = grid_size
        self.border = border
        self.axx = None
        self.fig = None
        
        environment_param = {} if environment_param is None else environment_param
        glutamate_release_param = {} if glutamate_release_param is None else glutamate_release_param
        astrocyte_param = {} if astrocyte_param is None else astrocyte_param
        self.ion_flow_param = {} if ion_flow_param is None else ion_flow_param
        vis_param = {} if vis_param is None else vis_param
        
        self.spatial_index = RtreeSpatialIndex(simulation=self)
        self.environment_grid = EnvironmentGrid(simulation=self, grid_size=grid_size, **environment_param)
        self.glutamate_release_manager = GlutamateReleaseManager(simulation=self, **glutamate_release_param)
        self.vis = Visualization(self, **vis_param)
        
        self.astrocytes = []
        tries = 0
        while tries < self.max_tries and len(self.astrocytes) < num_astrocytes:
            
            ast = self.add_astrocyte(astrocyte_param=astrocyte_param, ion_flow_param=ion_flow_param)
            
            if ast is None:
                tries += 1
                continue
            
            self.astrocytes.append(ast)
            tries = 0
        
        # log initial state
        self.data_logger.log_state()
    
    def log_state(self):
        state = {
            "astrocytes": [ast.id for ast in self.astrocytes]
            }
        return state
    
    def get_num_branches(self):
        return np.sum([len(ast.branches) for ast in self.astrocytes])
    
    @staticmethod
    def _timedelta(t0):
        return humanize.naturaldelta(time.time() - t0)
    
    def run_simulation_step(self, time_step=1, plot_param: dict = None):
        
        t_total = time.time()
        
        for _ in (pbar := tqdm(range(time_step))):
            
            t0 = time.time()
            
            self.glutamate_release_manager.step()
            self.environment_grid.step()
            
            for astrocyte in self.astrocytes:
                astrocyte.step()
            
            self.message_logger.step()
            self.data_logger.step()
            self.vis.step(param=plot_param)
            
            self.log(msg=f"runtime {self._timedelta(t0)} for {humanize.intword(self.get_num_branches())} branches")
            
            self.steps += 1
            pbar.set_description(f"#branches {self.get_num_branches()}")
        
        self.vis.plot(force=True, params=plot_param)
        self.log(msg=f"Total runtime ({time_step} steps): {self._timedelta(t_total)}")
    
    def add_astrocyte(self, x=None, y=None, radius=3, astrocyte_param: dict = None, ion_flow_param: dict = None):
        
        # set parameters
        if astrocyte_param is None:
            astrocyte_param = {}
        
        # get location
        X, Y = self.grid_size
        if x is None:
            x = np.random.randint(self.border, X - self.border)
        
        if y is None:
            y = np.random.randint(self.border, Y - self.border)
        
        # get radius
        if "radius" in astrocyte_param:
            radius = astrocyte_param["radius"]
        
        # check validity of location
        x0, x1 = x - radius, x + radius
        y0, y1 = y - radius, y + radius
        valid_placement = not self.spatial_index.check_collision((x0, y0, x1, y1))
        
        # place astrocyte
        if valid_placement:
            
            ast = Astrocyte(simulation=self, position=(x, y), ion_flow_param=ion_flow_param, **astrocyte_param)
            
            self.spatial_index.insert(ast)
            return ast
        
        else:
            return None
    
    def remove_astrocyte(self, astrocyte):
        pass


class Astrocyte(Loggable):
    
    def __init__(self, simulation: Simulation, position: Tuple[int, int], radius: int, num_branches: int,
                 max_branch_radius: float, start_spawn_radius: float,
                 repellent_name: str = None, repellent_concentration: float = 1,
                 numerical_tolerance=1e-6, allow_pruning=True,
                 min_trend_amplitude=0.5, min_steepness=0.05, spawn_angle_threshold=5,
                 diffusion_coefficient=50, glutamate_uptake_rate=100,
                 spawn_length=3, spawn_radius_factor=0.1, min_radius=0.001,
                 growth_factor=0.01, steps_till_death=50,
                 glu_v_max=100, glu_k_m=0.5, trend_history=30,
                 repellent_volume_factor=0.00001, repellent_surface_factor=0.00001,
                 atp_cost_per_glutamate=- 18, atp_cost_per_unit_surface=1,
                 max_history: int = 1000, max_tries=5,
                 ion_flow_param: dict = None,
                 molecules: dict = None):
        
        settings = Loggable.extract_settings(locals(), exclude_types=Simulation)
        super().__init__(simulation.data_logger, message_logger=simulation.message_logger, settings=settings)
        
        self.simulation = simulation
        self.environment_grid = simulation.environment_grid
        self.spatial_index = simulation.spatial_index
        self.data_logger = simulation.data_logger
        self.max_branch_radius = max_branch_radius
        self.children = []
        self.branches = []
        self.death_counter = 0
        self.num_branches = num_branches
        self.steps_till_death = steps_till_death
        self.ion_flow_param = ion_flow_param
        
        # computational parameters
        self.max_history = max_history
        self.max_tries = max_tries
        self.numerical_tolerance = numerical_tolerance
        
        # decision parameters
        self.allow_pruning = allow_pruning  # True
        self.min_trend_amplitude = min_trend_amplitude  # minimum trend in ATP and glutamate to grow or shrink
        self.min_steepness = min_steepness  # minimum steepness for spawning
        self.spawn_angle_threshold = spawn_angle_threshold  # °
        self.trend_history = trend_history
        
        # ion flow parameters
        self.diffusion_coefficient = diffusion_coefficient
        self.glutamate_uptake_rate = glutamate_uptake_rate  # mol/m² --> same as surface factor?
        self.repellent_name = repellent_name
        
        # cost parameters
        self.atp_cost_per_glutamate = atp_cost_per_glutamate  # mol ATP / mol Glutamate  # 1/18
        self.atp_cost_per_unit_surface = atp_cost_per_unit_surface  # mol/m²
        self.glu_V_max = glu_v_max
        self.glu_K_m = glu_k_m
        
        # physical properties
        self.spawn_length = spawn_length  # m
        self.spawn_radius_factor = spawn_radius_factor  # Relative proportion of end point compared to start point
        self.min_radius = min_radius  # m
        self.growth_factor = growth_factor
        self.repellent_volume_factor = repellent_volume_factor  # mol/m³
        self.repellent_surface_factor = repellent_surface_factor  # mol/m²
        
        self.x, self.y = position
        self.radius = radius
        self.volume = 4 / 3 * np.pi * radius ** 3
        self.pixels = self.get_pixels_within_cell()
        self.bbox = self.get_bbox()
        
        # create compartments
        self.cytosol = Compartment(volume=self.volume, start_concentration=cytosol_concentration_0)
        self.ER = Compartment(volume=er_volume_ratio * self.volume, start_concentration=er_concentration_0)
        self.extracellular_space = ExtracellularSpace(pixel=self.pixels, environment_grid=self.environment_grid)
        
        # todo: step compartments
        # todo: update compartment physical properties
        
        # spawn mother branches
        self.spawn_initial_branches(num_branches=num_branches, max_branch_radius=max_branch_radius,
                                    spawn_radius=start_spawn_radius, spawn_length=spawn_length)
        
        self.repellent_concentration = repellent_concentration
    
    def log_state(self):
        state = {
            "cytosol":       self.cytosol.get_all_concentrations(),
            "ER":            self.ER.get_all_concentrations(),
            "branches":      [branch.id for branch in self.branches],
            "children":      [child.id for child in self.children],
            "death_counter": self.death_counter
            }
        return state
    
    def get_bbox(self):
        x0, x1 = self.x - self.radius, self.x + self.radius
        y0, y1 = self.y - self.radius, self.y + self.radius
        
        xmin = min(x0, x1)
        ymin = min(y0, y1)
        xmax = max(x0, x1)
        ymax = max(y0, y1)
        return xmin, ymin, xmax, ymax
    
    def step(self):
        
        if self.death_counter > self.steps_till_death:
            self.die()
        
        self.simulate_glutamate()
        self.diffuse_molecules()
        
        for branch in self.branches:
            branch.step()
        
        self.release_repellent()
        
        if len(self.branches) <= self.num_branches:
            self.death_counter += 1
        else:
            self.death_counter = 0
        
        self.steps += 1
    
    def die(self):
        
        if self.death_counter / 100 < np.random.random():
            
            for child in self.children:
                child.action_prune()
            
            self.spatial_index.remove(self)
            self.simulation.astrocytes.remove(self)
            self.simulation.data_logger.unregister(self.id)
    
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
        pixels_x, pixels_y = [], []
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                # Calculate the distance from the center of the astrocyte to this pixel
                distance = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
                # If the distance is less than the radius, the pixel is within the astrocyte
                if distance <= self.radius:
                    pixels_x.append(x)
                    pixels_y.append(y)
        
        return np.array([pixels_x, pixels_y])
    
    def release_repellent(self):
        if self.repellent_name is not None:
            self.extracellular_space.update_concentration(molecule=self.repellent_name,
                                                          concentration=self.repellent_concentration)
    
    def simulate_glutamate(self):
        
        # todo: should there really be no rate for conversion?
        
        intra_glutamate = self.cytosol.get_amount("glutamate")
        
        if self.atp_cost_per_glutamate > 0:
            self.cytosol.update_amount("glutamate", -intra_glutamate)
        else:
            atp_change = intra_glutamate * self.atp_cost_per_glutamate
            self.cytosol.update_amount('ATP', abs(atp_change))
            self.cytosol.update_amount("glutamate", -intra_glutamate)
    
    def diffuse_molecules(self, dt=1):
        for molecule in self.cytosol.get_tracked_molecules():
            for target in self.children:
                
                # todo: maybe move to Compartment class? OR to IonFlowClass?
                
                # define source and target
                source = self.cytosol
                target = target.cytosol
                
                # Calculate the concentration difference between the branch and the target
                source_concentration = source.get_concentration(molecule)
                target_concentration = target.get_concentration(molecule)
                concentration_difference = source_concentration - target_concentration
                
                if concentration_difference <= 0:
                    continue
                
                # set radius
                radius = target.start.radius
                
                # calculate ions/molecules to move
                ions_to_move = self._calculate_diffusion_rate(radius) * concentration_difference * dt
                source.move_amount(target=target, molecule=molecule, amount=ions_to_move)
    
    def _calculate_diffusion_rate(self, radius):
        return self.diffusion_coefficient / (np.pi * radius ** 2)
    
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
        
        tries = 0
        while len(self.children) < num_branches and tries < self.max_tries:
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Choose a random location on the boundary (x, y, radius)
            start_x = self.x + np.cos(angle) * self.radius
            start_y = self.y + np.sin(angle) * self.radius
            start = AstrocyteNode(start_x, start_y, max_branch_radius)
            
            # Set the end point perpendicular to the center of the astrocyte with 'spawn_length' and 'spawn_radius'
            end_x = start_x + np.cos(angle) * spawn_length
            end_y = start_y + np.sin(angle) * spawn_length
            end = AstrocyteNode(end_x, end_y, spawn_radius)
            
            # Create the new branch
            new_branch = AstrocyteBranch(parent=self, nucleus=self, start=start, end=end,
                                         ion_flow_param=self.ion_flow_param)
            
            if not self.spatial_index.check_collision(new_branch):
                
                self.children.append(new_branch)
                self.branches.append(new_branch)
                
                # Add the new branch to the spatial index
                self.spatial_index.insert(new_branch)
                
                tries = 0
            
            else:
                tries += 1
        
        if tries >= self.max_tries:
            self.log(f"Maximum astrocyte tries exceeded: {tries}", level=logging.WARNING,
                     tag="warning,tries,astrocyte")


class AstrocyteBranch(Loggable):
    
    def __init__(self, parent, nucleus: Astrocyte, start: Union[Tuple[int, int, int], AstrocyteNode],
                 end: Union[Tuple[int, int, int], AstrocyteNode], ion_flow_param: dict = None):
        
        super().__init__(nucleus.simulation.data_logger, message_logger=nucleus.simulation.message_logger,
                         settings=Loggable.extract_settings(locals()))
        
        # Instances
        self.parent: AstrocyteBranch = parent
        self.nucleus = nucleus
        self.spatial_index = nucleus.spatial_index
        
        self.start: AstrocyteNode = start if isinstance(start, AstrocyteNode) else AstrocyteNode(*start)
        self.end: AstrocyteNode = end if isinstance(end, AstrocyteNode) else AstrocyteNode(*end)
        
        self.children: List[AstrocyteBranch] = []
        
        self.interacting_pixels = None
        self.volume = None
        self.surface_area = None
        self.repellent_release = None
        self.glutamate_uptake_capacity = None
        self.pruned = False
        self.counter_failed_spawn = 0
        
        # update physical properties
        self.update_physical_properties()
        
        # compartments
        # create compartments
        self.cytosol = Compartment(volume=self.volume, start_concentration=cytosol_concentration_0)
        self.ER = Compartment(volume=er_volume_ratio * self.volume, start_concentration=er_concentration_0)
        self.extracellular_space = ExtracellularSpace(pixel=self.interacting_pixels,
                                                      environment_grid=self.nucleus.environment_grid)
        
        # todo: step compartments
        # todo: update compartment physical properties
        
        # Create calcium flow model
        ion_flow_param = {} if ion_flow_param is None else ion_flow_param
        self.ion_flow_model = IonFlowModel(self, **ion_flow_param)
    
    def log_state(self):
        state = {
            "environment":          self.environment,
            "molecules":            self.molecules,
            "children":             [child.id for child in self.children],
            "parent":               self.parent.id,
            "nucleus":              self.nucleus.id,
            "counter_failed_spawn": self.counter_failed_spawn,
            "start":                self.start.get_dimension(),
            "end":                  self.end.get_dimension(),
            "pruned":               self.pruned,
            }
        return state
    
    def get_bbox(self):
        
        x0, y0 = self.start.x, self.start.y
        x1, y1 = self.end.x, self.end.y
        
        xmin = min(x0, x1)
        ymin = min(y0, y1)
        xmax = max(x0, x1)
        ymax = max(y0, y1)
        return xmin, ymin, xmax, ymax
    
    def step(self):
        
        # update environment
        self.get_environment()
        
        # simulate flow of molecules
        self._simulate_calcium()
        self._simulate_glutamate()
        self._simulate_repellent()
        
        # run through actions
        self._act()
        if self.pruned:
            self.log("I was pruned. Aborting simulation step.", tag="branch")
            return
        
        # simulate molecule diffusion
        self.diffuse_molecules()
        
        # save new state
        self._update_history()
        self.steps += 1
    
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
        if not history or len(history) < 2:
            return 0.0  # No trend if history is empty
        history = np.array(history)[:self.nucleus.trend_history].astype(float)
        
        # Create an array of time points (assuming equal time intervals)
        x = np.arange(len(history))
        
        # Perform linear regression
        slope, _ = np.polyfit(x, history, 1)
        
        return slope
    
    def get_interacting_pixels(self) -> List[Tuple[int, int]]:
        """
        Get the grid cells (pixels) that are intersected by the current branch.

        Returns:
            A list of grid cells (x, y) that the branch intersects.
        """
        # Use the rasterize_line method from the RtreeSpatialIndex to get the pixels
        interacting_pixels = self.nucleus.spatial_index.rasterize_line(self.start.x, self.start.y, self.end.x,
                                                                       self.end.y)
        
        # todo convert to numpy array?
        
        return interacting_pixels
    
    def _calculate_diffusion_rate(self, radius: float) -> float:
        """
        Calculate the diffusion rate of ions along the branch.

        Returns:
            The diffusion rate for the branch.
        """
        
        # Calculate the length of the branch
        length_of_branch = np.sqrt((self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2)
        
        # Apply the formula for diffusion rate
        diffusion_rate = self.nucleus.diffusion_coefficient * (1 / length_of_branch) * (
                1 / (np.pi * radius ** 2))
        
        return diffusion_rate
    
    def diffuse_molecules(self):
        
        for molecule in self.molecules.items():
            for target in self.children + [self.parent]:
                
                # Calculate the concentration difference between the branch and the target
                concentration_difference = self.get_concentration(molecule) - target.get_concentration(molecule)
                
                if concentration_difference <= 0:
                    continue
                
                # set radius
                if target == self.parent:
                    radius = self.start.radius
                else:
                    radius = self.end.radius
                
                # calculate ions/molecules to move
                dt = 1  # one time step
                ions_to_move = self._calculate_diffusion_rate(radius) * concentration_difference * dt
                ions_to_move = min(ions_to_move, self.get_amount(molecule))
                
                # update new concentrations
                self.update_amount(molecule, -ions_to_move)
                target.update_amount(molecule, ions_to_move)
    
    def _simulate_calcium(self, dt: int = 1):
        self.ion_flow_model.update_concentrations(dt=dt, use_physical_properties=True)
    
    def _simulate_glutamate(self):
        
        # remove from environment
        self.extracellular_space.move_amount(self.cytosol, molecule="glutamate", amount=self.glutamate_uptake_capacity)
        
        # convert glutamate using up ATP
        self.cytosol.convert(source_molecule="glutamate", target_molecule="ATP",
                             conversion_factor=self.nucleus.atp_cost_per_glutamate,
                             v_max=self.nucleus.glu_V_max, k_m=self.nucleus.glu_K_m)
    
    def _simulate_repellent(self):
        # Release repellent into environment
        if self.nucleus.repellent_name is not None and self.repellent_release is not None:
            self.extracellular_space.update_amount(molecule=self.nucleus.repellent_name, amount=self.repellent_release)
    
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
    
    def update_physical_properties(self, location_updated=True, radius_updated=True):
        
        if location_updated:
            self.interacting_pixels = self.get_interacting_pixels()
        
        if location_updated or radius_updated:
            
            self.volume = self.calculate_branch_volume()
            self.surface_area = self.calculate_branch_surface()
        
        if location_updated or radius_updated:
            self.glutamate_uptake_capacity = self.calculate_removal_capacity(self.nucleus.glutamate_uptake_rate)
        
        if location_updated or radius_updated:
            if self.nucleus.repellent_name is not None and self.repellent_release is not None:
                self.repellent_release = self.calculate_repellent_release(self.nucleus.repellent_surface_factor,
                                                                          self.nucleus.repellent_volume_factor)
    
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
        
        spawn_probability = 1 if len(self.children) == 0 else 1 / len(self.children)
        if spawn_probability > np.random.random():
            self._action_spawn_or_move(self.nucleus.min_steepness, self.nucleus.spawn_angle_threshold)
        
        # we prune automatically if the end radius drops below min_radius
        self._action_grow_or_shrink(self.nucleus.growth_factor, self.nucleus.min_trend_amplitude,
                                    self.nucleus.atp_cost_per_unit_surface, self.nucleus.min_radius)
    
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
        
        # atp_trend = self.get_trend("ATP", intra=True)
        glutamate_trend = self.get_trend("glutamate", intra=False)
        # combined_trend = glutamate_trend - atp_trend
        combined_trend = glutamate_trend
        
        if abs(combined_trend) < min_trend_amplitude:
            
            if combined_trend > self.nucleus.numerical_tolerance:
                self.log(f"Growth trend too low: {combined_trend:.1E} "
                         f"({combined_trend / min_trend_amplitude * 100:.1f}%)",
                         tag="branch,growth,act")
            
            return
        
        if combined_trend < 0:
            growth = 1 - growth_factor
        else:
            growth = 1 + growth_factor
        
        # Calculate the new surface area after growth/shrinkage
        new_end = self.end.copy()
        new_end.radius *= growth
        
        if new_end.radius < min_radius:
            self.log(f"Growth prune: {humanize.metric(new_end.radius, 'm')} ({growth})", tag="branch,growth,prune,act")
            self.action_prune()
            return
        
        new_surface_area = self.calculate_branch_surface(start=self.start, end=new_end)
        
        # Ensure the end node radius is not less than the start node radius after growth/shrinkage
        if new_end.radius >= self.start.radius:
            self.log(f"Growth constrained by start node size: "
                     f"{humanize.metric(new_end.radius, 'm')} vs. "
                     f"{humanize.metric(self.start.radius, 'm')}",
                     tag="branch,growth,act")
            return
        
        # Calculate the required ATP based on the change in surface area
        delta_surface = new_surface_area - self.surface_area
        required_atp = delta_surface * atp_cost_per_unit_surface
        
        # If shrinking, ATP is released (required_atp will be negative)
        available_atp = self.get_amount("ATP")
        
        # Check if there's enough ATP to support the growth, or if ATP needs to be added back for shrinkage
        if growth_factor > 1 and available_atp < required_atp:
            self.log(f"Growth not supported by ATP: ({available_atp / required_atp * 100:.1f}%).",
                     tag="branch,growth,act")
            return
        
        # Update new end node
        self.end = new_end
        
        # Update the ATP concentration in the branch
        if growth_factor > 0:
            self.update_amount("ATP", -required_atp)
        
        # Update the physical properties of the branch (e.g., recalculate volume, surface area)
        self.update_physical_properties(location_updated=False)
        
        self.log(f"Growth adjusted to "
                 f"{humanize.metric(new_end.radius, 'm')} for "
                 f"{humanize.metric(abs(required_atp), 'mol')} ATP",
                 tag="branch,growth,act")
    
    def _action_spawn_or_move(self, min_steepness: float,
                              angle_threshold: float):
        """
        Spawn a new branch or move the current branch based on the environmental factors.

        If the direction of growth does not vary too much from the current direction and the branch has no children,
        the branch will move. Otherwise, a new branch will be spawned.

        Args:
            min_steepness: Minimum steepness of the combined gradient (glutamate and repellent) that triggers
                spawning of a new branch or movement.
            angle_threshold: The threshold for how much the new direction can vary from the current direction.
        """
        
        # Calculate the direction of the new branch based on glutamate and repellent gradients
        spawn_locations = self._find_best_spawn_location()
        
        if spawn_locations is None:
            self.counter_failed_spawn += 1
            self.log(f"No branch found.", tag="branch,act,spawn,fail")
            return
        
        best_branch, steepness = spawn_locations
        
        self.counter_failed_spawn = 0
        if steepness > min_steepness:
            
            # determine angle between new and current branch
            angle = self._get_angle_between_branches(best_branch, self)
            
            if len(self.children) > 0 or angle > angle_threshold:
                # If direction varies too much or branch has children, spawn a new branch
                self._spawn_new_branch(best_branch)
            else:
                # If direction is similar and branch has no children, move the branch
                self._move_branch(best_branch)
                self.log(f"Moving branch: {angle} < {angle_threshold}", tag="branch,move,act")
        else:
            if steepness > self.nucleus.numerical_tolerance:
                
                self.log(
                        f"Branch spawning steepness low: {steepness:.1E} !> {min_steepness:.1E} "
                        f"({steepness / min_steepness * 100:.1f}%)",
                        tag="branch,spawn,act,fail")
    
    @staticmethod
    def _get_angle_between_branches(branch1, branch2):
        
        # get normalized direction
        direction_1 = branch1.get_norm_direction()
        direction_2 = branch2.get_norm_direction()
        
        # Calculate the dot product of the two direction vectors
        dot_product = np.dot(direction_1, direction_2)
        
        # Ensure the dot product does not exceed 1 due to numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate the angle using the arccosine of the dot product
        angle = np.arccos(dot_product)
        
        # convert to degrees
        angle = np.degrees(angle)
        
        return abs(angle)
    
    def action_prune(self):
        """
        Prune the branch if it has no children.
        """
        if self.nucleus.allow_pruning:
            
            # Ensure no children exist; else skip
            if self.children:
                self.log("Branch has children, cannot prune.", tag="branch,act,prune,fail")
                return
            
            # Remove self from spatialIndexTree
            self.spatial_index.remove(self)
            
            # Delete self from parent
            self.parent.children.remove(self)
            self.nucleus.branches.remove(self)
            self.nucleus.simulation.data_logger.unregister(self.id)
            
            # Additional cleanup if needed (e.g., freeing resources or nullifying references)
            self.pruned = True
            self.log("Branch pruned successfully.", tag="branch,act,prune")
    
    def _spawn_new_branch(self, branch: AstrocyteBranch):
        """
        Spawn a new branch from the current branch.

        Args:
            branch: The branch that will be spawned.
        """
        
        # calculate cost
        atp_cost = self.nucleus.atp_cost_per_unit_surface * branch.surface_area
        if atp_cost <= self.cytosol.get_amount("ATP"):
            
            # Save the new branch to the list of children
            self.children.append(branch)
            self.nucleus.branches.append(branch)
            
            # Update the spatial index with the new branch
            self.nucleus.spatial_index.insert(branch)
            
            # remove atp
            self.cytosol.update_amount("ATP", -atp_cost)
            
            self.log(f"Spawned new branch at: {(branch.end.x, branch.end.y)} for "
                     f"{humanize.metric(atp_cost, 'mol')} ATP",
                     tag="branch,act,spawn")
        
        else:
            self.log(
                    f"Insufficient ATP available for branch spawning ({atp_cost:.2f} !< "
                    f"{self.cytosol.get_amount('ATP'):.2f}).",
                    tag="branch,act,spawn,fail")
    
    def _move_branch(self, new_branch: AstrocyteBranch):
        """
        Move the current branch in a specified direction.

        Args:
            new_branch: The location of the new branch.
        """
        
        # Update the end node
        surface_area_difference = new_branch.surface_area - self.surface_area
        
        atp_cost = self.nucleus.atp_cost_per_unit_surface * surface_area_difference
        
        if atp_cost <= self.cytosol.get_amount("ATP"):
            self.end.x, self.end.y = new_branch.end.x, new_branch.end.y
            
            # Update the spatial index before changing the position
            self.nucleus.spatial_index.update(self)
            
            # remove atp
            self.cytosol.update_amount("ATP", -atp_cost)
            
            # Update the physical properties of the branch
            self.update_physical_properties()
            
            self.log(f"Moved branch to {(self.end.x, self.end.y)}", tag="branch,act,spawn,move")
        
        else:
            self.log(
                    f"Insufficient ATP to move branch ({atp_cost:.2f} !< {self.cytosol.get_amount('ATP'):.2f}).",
                    tag="branch,act,spawn,move,fail")
    
    def _find_best_spawn_location(self, num_candidates=8) -> Union[Tuple[AstrocyteBranch, float], None]:
        
        # check if too many fails
        spawn_probability = 1 if self.counter_failed_spawn == 0 else 1 / self.counter_failed_spawn
        spawn_probability = max(0.01, spawn_probability)
        if spawn_probability <= np.random.random():
            self.log(f"Spawn failed too many times: {self.counter_failed_spawn} ({spawn_probability * 100:.1f}%)",
                     tag="branch,act,spawn,fail")
            return None
        
        # New branch radius
        new_branch_radius = self.end.radius * self.nucleus.spawn_radius_factor
        if new_branch_radius < self.nucleus.min_radius:
            self.log(f"Spawned branch would be below minimum radius "
                     f"{humanize.metric(new_branch_radius, 'm')} !> "
                     f"{humanize.metric(self.nucleus.min_radius, 'm')}",
                     tag="branch,act,spawn,fail")
            return None
        
        # Generate a set of candidate directions
        candidate_branches = []
        angle_increment = 2 * np.pi / num_candidates
        
        for i in range(num_candidates):
            angle = i * angle_increment + np.random.random() * angle_increment
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # End coordinates of the candidate branch
            end_x = self.end.x + dx * self.nucleus.spawn_length
            end_y = self.end.y + dy * self.nucleus.spawn_length
            
            X, Y = self.nucleus.environment_grid.grid_size
            if end_x > X or end_x < 0 or end_y > Y or end_y < 0:
                continue
            
            # create candidate branch
            new_end_node = AstrocyteNode(x=end_x, y=end_y, radius=new_branch_radius)
            candidate = AstrocyteBranch(parent=self, nucleus=self.nucleus, start=self.end, end=new_end_node)
            
            candidate_branches.append(candidate)
        
        # Choose best candidate
        best_candidate = None
        max_steepness = -np.inf
        
        for candidate in candidate_branches:
            # Check for collision with existing branches
            end_collision_zone = (candidate.end.x - 1, candidate.end.y - 1,
                                  candidate.end.x + 1, candidate.end.y + 1)
            if not self.nucleus.spatial_index.check_collision(end_collision_zone):
                # Calculate the combined gradient along the theoretical branch
                steepness_glutamate = self._calculate_gradient_along_branch(candidate, molecule="glutamate")
                
                if self.nucleus.repellent_name is not None and self.repellent_release is not None:
                    steepness_repellent = self._calculate_gradient_along_branch(candidate, molecule="repellent")
                else:
                    steepness_repellent = 0
                
                combined_steepness = steepness_glutamate - steepness_repellent
                
                if combined_steepness > max_steepness:
                    max_steepness = combined_steepness
                    best_candidate = candidate
        
        if best_candidate is not None:
            # Normalize the best direction vector
            return best_candidate, max_steepness
        else:
            # If no valid direction is found or if the gradient is zero, return None
            return None
    
    def _calculate_gradient_along_branch(self, candidate: AstrocyteBranch, molecule: str) -> float:
        """
        Calculate the gradient of a specified molecule along a branch.

        Args:
            candidate: The candidate branch (AstrocyteBranch object) along which to calculate the gradient.
            molecule: The name of the molecule for which to calculate the gradient.

        Returns:
            The gradient of the specified molecule along the branch.
        """
        # Get the molecule concentration at the start and end of the branch
        
        concentration_start = self.extracellular_space.get_concentration_at(candidate.start.get_position(), molecule)
        concentration_end = self.extracellular_space.get_concentration_at(candidate.end.get_position(), molecule)
        
        # Calculate the concentration difference
        concentration_difference = concentration_end - concentration_start
        
        # Calculate the length of the branch
        length_of_branch = np.sqrt(
                (candidate.end.x - candidate.start.x) ** 2 + (candidate.end.y - candidate.start.y) ** 2)
        
        # Avoid division by zero
        if length_of_branch == 0:
            return 0
        
        # Calculate the gradient
        gradient = concentration_difference / length_of_branch
        
        return gradient
    
    def get_norm_direction(self):
        """
        Calculate the normalized direction vector (unit vector) from the start to the end of the branch.

        Returns:
            A tuple representing the normalized direction vector (dx, dy).
        """
        # Direction vector from start to end
        direction_vector = np.array([self.end.x - self.start.x, self.end.y - self.start.y])
        
        # Calculate the norm (magnitude) of the direction vector
        norm = np.linalg.norm(direction_vector)
        
        # Normalize the direction vector to get a unit vector
        if norm > 0:
            norm_direction = direction_vector / norm
        else:
            # If the norm is zero (start and end are the same), return a zero vector
            norm_direction = np.array([0, 0])
        
        return tuple(norm_direction)


class Compartment:
    
    def __init__(self, volume: float = None, start_amount: dict = None, start_concentration: dict = None):
        
        self.volume = volume
        self.concentration = {}
        
        if start_amount is not None and start_concentration is not None:
            raise ValueError(f"Choose either start amount or concentration, not both.")
        
        elif start_amount is not None:
            for molecule in start_amount.keys():
                self.update_amount(molecule, start_amount[molecule])
        
        elif start_concentration is not None:
            for molecule in start_concentration.keys():
                self.set_concentration(molecule, start_concentration[molecule])
        
        else:
            pass
        
        # todo: implement history
    
    def get_tracked_molecules(self):
        return self.concentration.keys()
    
    def set_concentration(self, molecule: str, concentration: float):
        
        if concentration < 0:
            raise ValueError(f"Concentration must be positive.")
        
        self.concentration[molecule] = concentration
    
    def set_amount(self, molecule: str, amount: float):
        self.set_concentration(molecule=molecule, concentration=amount / self.volume)
    
    def get_all_concentrations(self):
        return self.concentration
    
    def get_concentration(self, molecule: str):
        if molecule not in self.concentration:
            self.set_concentration(molecule=molecule, concentration=0)
        
        return self.concentration[molecule]
    
    def get_amount(self, molecule: str):
        return self.get_concentration(molecule=molecule) * self.volume
    
    def update_concentration(self, molecule: str, concentration: float) -> float:
        
        if concentration < 0:
            concentration = - min(self.get_concentration(molecule), abs(concentration))
        
        new_concentration = self.get_concentration(molecule) + concentration
        
        self.set_concentration(molecule=molecule, concentration=new_concentration)
        
        return concentration
    
    def update_amount(self, molecule: str, amount: float) -> float:
        
        updated_concentration = self.update_concentration(molecule=molecule, concentration=amount / self.volume)
        return updated_concentration * self.volume
    
    def move_concentration(self, target: Compartment, molecule: str, concentration: float):
        if concentration > 0:
            raise ValueError(f"Concentration must be negative.")
        
        self.move_amount(target=target, molecule=molecule, amount=concentration * self.volume)
    
    def move_amount(self, target: Compartment, molecule: str, amount: float):
        
        if amount > 0:
            raise ValueError(f"Amount must be negative.")
        
        moved_amount = self.update_amount(molecule=molecule, amount=amount)
        target.update_amount(molecule=molecule, amount=abs(moved_amount))
        
        self.log(f"Moved "
                 f"{humanize.metric(moved_amount, 'mol')} {molecule} "
                 f"({moved_amount / (self.get_amount(molecule) + moved_amount) * 100:.1f}%). "
                 f"{moved_amount / amount * 100:.1f}% of capacity.",
                 tag="branch,ion,move")
    
    def convert(self, source_molecule: str, target_molecule: str, conversion_factor: float,
                v_max: float, k_m: float):
        
        if conversion_factor < 0:
            temp = source_molecule
            source_molecule = target_molecule
            target_molecule = temp
            
            conversion_factor *= -1
        
        # amounts and concentration
        source_concentration = self.get_concentration(source_molecule)
        source_amount = self.get_amount(source_molecule)
        
        target_concentration = self.get_concentration(target_molecule)
        target_amount = self.get_amount(target_molecule)
        
        # calculate rate
        rate = (v_max * source_concentration) / (k_m + source_concentration)
        actual_rate = min(rate, source_amount)
        
        # calculate amount
        converted_target = abs(actual_rate * conversion_factor)
        
        # update values
        self.update_amount(molecule=source_molecule, amount=actual_rate)
        self.update_amount(molecule=target_molecule, amount=converted_target)
        
        # logging
        self.log(f"Converted "
                 f"{humanize.metric(actual_rate, 'mol')} {source_molecule} "
                 f"(- {actual_rate / source_amount * 100:.1f}%)"
                 f" to"
                 f"{humanize.metric(converted_target, 'mol')} {target_molecule}"
                 f"(+ {100 - converted_target / target_amount * 100:.1f}%, ",
                 tag="conversion,branch,ion")


class ExtracellularSpace(Compartment):
    def __init__(self, pixel: np.ndarray, environment_grid: EnvironmentGrid):
        self.environment_grid = environment_grid
        self.pixel = pixel
        
        super().__init__(volume=None, start_amount=None)
        
        self.update_pixel(pixel)
    
    def update_pixel(self, pixel: np.ndarray):
        
        if not (isinstance(pixel, np.ndarray) and pixel.ndim == 2 and pixel.shape[0] == 2):
            raise ValueError(f"location must be a 2D numpy array with shape (num_locations, 2). Found {pixel.shape}")
        
        self.pixel = pixel
        self.volume = pixel.shape[1] * self.environment_grid.pixel_volume
    
    def get_xy_coordinates(self):
        return self.pixel[0, :], self.pixel[1, :]
    
    def molecule_exists(self, molecule: str):
        if molecule not in self.environment_grid.shared_arrays:
            raise ValueError(f"Molecule {molecule} not found in the grid.")
    
    def set_concentration(self, molecule: str, concentration: float):
        
        if concentration < 0:
            raise ValueError(f"Concentration must be positive.")
        
        self.molecule_exists(molecule)
        
        x, y = self.get_xy_coordinates()
        self.environment_grid.shared_arrays[molecule][0][x, y] = concentration
    
    def get_concentration(self, molecule: str):
        
        self.molecule_exists(molecule=molecule)
        
        x, y = self.get_xy_coordinates()
        return np.sum(self.environment_grid.shared_arrays[molecule][0][x, y])
    
    def update_concentration(self, molecule: str, concentration: float) -> float:
        
        self.molecule_exists(molecule=molecule)
        
        current_concentration = self.get_concentration(molecule)
        if concentration < 0:
            concentration = - min(current_concentration, abs(concentration))
        
        new_concentration = current_concentration + concentration
        
        self.set_concentration(molecule=molecule, concentration=new_concentration)
        
        return concentration


class IonFlowModel:
    def __init__(self, branch: AstrocyteBranch, total_cytosolic_ca_concentration: float = 2.0,
                 er_cytosol_volume_ratio: float = 0.185, max_ca_channel_flux: float = 6.0,
                 ca_leak_flux_constant: float = 0.11, max_ca_uptake: float = 2.2,
                 max_ip3_production_rate: float = 0.3, ca_extrusion_rate_constant: float = 0.5,
                 atp_ca_pump_activation_constant: float = 0.1, ip3_dissociation_constant: float = 0.13,
                 ca_activation_constant: float = 82.0, ca_leak_rate_plasma_membrane: float = 0.025,
                 max_activation_dependent_ca_influx: float = 0.2, half_saturation_constant_ca_entry: float = 1.0,
                 ip3_coupling_coefficient: float = 0.8,
                 ca_stimulation_ip3_production_dissociation_constant: float = 1.1,
                 ip3_production_rate_glutamate: float = 0.062,
                 glutamate_stimulation_ip3_production_dissociation_constant: float = 0.78,
                 steady_state_ip3_concentration: float = 0.16, ip3_loss_rate_constant: float = 0.14):
        """
        Initialize the IonFlowModel which simulates the ion flows related to calcium ions and IP3 in astrocytes.

        The model parameters are based on the differential equations provided in the referenced papers.

        Args:
            branch (AstrocyteBranch): The astrocyte branch for which the ion flows are being modeled.
            total_cytosolic_ca_concentration (float): Total [Ca2+] in terms of cytosolic volume.
            er_cytosol_volume_ratio (float): ER volume / cytosolic volume ratio.
            max_ca_channel_flux (float): Maximum Ca2+ channel flux.
            ca_leak_flux_constant (float): Ca2+ leak flux constant.
            max_ca_uptake (float): Maximum Ca2+ uptake by the ATP-dependent pump.
            max_ip3_production_rate (float): Maximum rate of IP3 production.
            ca_extrusion_rate_constant (float): Rate constant of calcium extrusion.
            atp_ca_pump_activation_constant (float): Activation constant for the ATP-Ca2+ pump.
            ip3_dissociation_constant (float): Dissociation constant for IP3.
            ca_activation_constant (float): Ca2+ activation constant.
            ca_leak_rate_plasma_membrane (float): Rate of calcium leak across the plasma membrane.
            max_activation_dependent_ca_influx (float): Maximal rate of activation-dependent calcium influx.
            half_saturation_constant_ca_entry (float): Half-saturation constant for agonist-dependent calcium entry.
            ip3_coupling_coefficient (float): Coupling coefficient for IP3.
            ca_stimulation_ip3_production_dissociation_constant (float): Dissociation constant for Ca2+ stimulation of IP3 production.
            ip3_production_rate_glutamate (float): Rate of IP3 production through glutamate.
            glutamate_stimulation_ip3_production_dissociation_constant (float): Dissociation constant for glutamate stimulation of IP3 production.
            steady_state_ip3_concentration (float): Steady state concentration of IP3.
            ip3_loss_rate_constant (float): Rate constant for the loss of IP3.

        Note:
            Verisokin Andrey Yu., Verveyko Darya V., Postnov Dmitry E., Brazhe Alexey R., Modeling of Astrocyte Networks: Toward Realistic Topology and Dynamics, 2021, 10.3389/fncel.2021.645068
            Ghanim Ullah, Peter Jung, A.H. Cornell-Bell, Anti-phase calcium oscillations in astrocytes via inositol (1, 4, 5)-trisphosphate regeneration, Cell Calcium, https://doi.org/10.1016/j.ceca.2005.10.009.
        """
        
        # Assign parameters to class properties
        self.branch = branch
        self.total_cytosolic_ca_concentration = total_cytosolic_ca_concentration
        self.er_cytosol_volume_ratio = er_cytosol_volume_ratio
        self.max_ca_channel_flux = max_ca_channel_flux
        self.ca_leak_flux_constant = ca_leak_flux_constant
        self.max_ca_uptake = max_ca_uptake
        self.max_ip3_production_rate = max_ip3_production_rate
        self.ca_extrusion_rate_constant = ca_extrusion_rate_constant
        self.atp_ca_pump_activation_constant = atp_ca_pump_activation_constant
        self.ip3_dissociation_constant = ip3_dissociation_constant
        self.ca_activation_constant = ca_activation_constant
        self.ca_leak_rate_plasma_membrane = ca_leak_rate_plasma_membrane
        self.max_activation_dependent_ca_influx = max_activation_dependent_ca_influx
        self.half_saturation_constant_ca_entry = half_saturation_constant_ca_entry
        self.ip3_coupling_coefficient = ip3_coupling_coefficient
        self.ca_stimulation_ip3_production_dissociation_constant = ca_stimulation_ip3_production_dissociation_constant
        self.ip3_production_rate_glutamate = ip3_production_rate_glutamate
        self.glutamate_stimulation_ip3_production_dissociation_constant = glutamate_stimulation_ip3_production_dissociation_constant
        self.steady_state_ip3_concentration = steady_state_ip3_concentration
        self.ip3_loss_rate_constant = ip3_loss_rate_constant
        
        # Initialize variables
        self.branch = branch
        self.concentration = branch.molecules
        
        for req in ['CaCyt', 'CaER', 'IP3', 'glutamate']:
            if req not in self.concentration:
                raise KeyError(f"Molecule array does not contain {req}: {self.concentration.keys()}")
    
    def __getitem__(self, item):
        return self.concentration[item]
    
    def ca_flow_er_ip3(self):
        """
        Calculate the flux of calcium ions from the endoplasmic reticulum (ER) to the cytosol through IP3 receptors (JIP3).

        The flux is determined by the IP3 receptor activation (m_inf and n_inf), the maximum Ca2+ channel flux (v1),
        and the difference in calcium concentration between the ER and the cytosol.

        Returns:
            The flux of calcium ions through IP3 receptors from the ER to the cytosol.
        """
        
        # Steady-state value of the gating variable m
        m_inf = self["IP3"] / (self["IP3"] + self.ip3_dissociation_constant)
        
        # Steady-state value of the gating variable n
        n_inf = self["CaCyt"] / (self["CaCyt"] + self.ca_activation_constant)
        return self.er_cytosol_volume_ratio * self.max_ca_channel_flux * m_inf ** 3 * n_inf ** 3 * (
                self["CaER"] - self["CaCyt"])
    
    def ca_flow_er_leak(self):
        """
        Calculate the leakage flux of calcium ions from the endoplasmic reticulum (ER) to the cytosol (JLeak).

        The leakage is proportional to the difference in calcium concentration between the ER and the cytosol and
        is controlled by the leak flux constant (v2).

        Returns:
            The leakage flux of calcium ions from the ER to the cytosol.
        """
        return self.er_cytosol_volume_ratio * self.ca_leak_flux_constant * (self["CaER"] - self["CaCyt"])
    
    def ca_flow_er_pump(self):
        """
        Calculate the pump flux of calcium ions from the cytosol back into the endoplasmic reticulum (ER) through ATP-dependent pumps (JPump).

        The pump flux depends on the cytosolic calcium concentration, the maximum Ca2+ uptake (v3), and the activation
        constant for the ATP-Ca2+ pump (k3).

        Returns:
            The pump flux of calcium ions from the cytosol to the ER.
        """
        return (self.max_ca_uptake * self["CaCyt"] ** 2) / (
                self["CaCyt"] ** 2 + self.atp_ca_pump_activation_constant ** 2)
    
    def ca_flow_pm_in(self):
        """
        Calculates the influx (Jin) of calcium ions through the plasma membrane.
        """
        
        flow = 0
        flow += self.ca_leak_rate_plasma_membrane
        flow += self.max_activation_dependent_ca_influx * (
                self["IP3"] ** 2 / (self.half_saturation_constant_ca_entry ** 2 + self["IP3"] ** 2))
        
        return flow
    
    def ca_flow_pm_glu(self):
        """
        Calculates the glutamate-induced influx (JGlu) of calcium ions through the plasma membrane.
        """
        return self.ip3_production_rate_glutamate * self["glutamate"] / (
                self.glutamate_stimulation_ip3_production_dissociation_constant + self["glutamate"])
    
    def ca_flow_pm_out(self):
        """
        Calculates the extrusion (Jout) of calcium ions by the plasma membrane pump.
        """
        return self.ca_extrusion_rate_constant * self["CaCyt"]
    
    def ip3_eq(self):
        """
        Calculate the equilibration of IP3 to a resting level.
        """
        return (self["IP3"] - self.steady_state_ip3_concentration) / self.ip3_loss_rate_constant
    
    def ip3_glu(self):
        """
        Calculate the glutamate-driven IP3 production.
        """
        return self.ip3_production_rate_glutamate * self["glutamate"] / (
                self.glutamate_stimulation_ip3_production_dissociation_constant + self["glutamate"])
    
    def ip3_ca(self):
        """
        Calculate the Ca2+-stimulated IP3 production.
        """
        return self.max_ip3_production_rate * (self["CaCyt"] + (
                1 - self.ip3_coupling_coefficient) * self.ca_stimulation_ip3_production_dissociation_constant) / (
                self["CaCyt"] + self.ca_stimulation_ip3_production_dissociation_constant)
    
    @staticmethod
    def move_ions(source: Union[AstrocyteBranch, Astrocyte], target: Union[AstrocyteBranch, Astrocyte],
                  molecule_source: str, molecule_target: str, amount):
        
        if amount == 0:
            return
        
        if amount < 0:
            # switch source and target
            temp = source
            source = target
            target = temp
            
            temp = molecule_source
            molecule_source = molecule_target
            molecule_target = temp
            
            amount *= -1
        
        available_amount = source.get_amount(molecule=molecule_source)
        if amount > available_amount:
            amount = available_amount
        
        source.update_amount(molecule=molecule_source, amount=-amount)
        target.update_amount(molecule=molecule_target, amount=amount)
    
    def move_concentration(self, source: Union[AstrocyteBranch, Astrocyte], target: Union[AstrocyteBranch, Astrocyte],
                           molecule_source: str, molecule_target: str, concentration):
        
        amount = source.volume * concentration
        self.move_ions(source=source, target=target, molecule_source=molecule_source, molecule_target=molecule_target,
                       amount=amount)
    
    def update_concentrations(self, dt: float, use_physical_properties: bool = True):
        """
        Update the concentrations of calcium and IP3 within the astrocyte branch based on the calculated flows and diffusion terms.

        Args:
            dt (float): The time step for the simulation.
            use_physical_properties (bool): Flag to determine whether to use physical properties for SVR calculation.
                                           If True, use the branch's surface area and volume to calculate SVR.
                                           If False, calculate SVR based on AVF as per Verisokin et al. 2021.

        Returns:
            None: The function updates the concentrations within the branch directly.
        """
        
        if use_physical_properties:
            # Calculate Surface-to-Volume Ratio based on branch's physical properties
            SVR = self.branch.surface_area / self.branch.volume
        else:
            # Calculate Surface-to-Volume Ratio based on AVF as per Verisokin et al. 2021
            external_space = len(self.branch.interacting_pixels) * self.branch.nucleus.environment_grid.pixel_volume
            occupied_volume = self.branch.volume
            
            # Warning if occupied volume exceeds the external space
            if occupied_volume > external_space:
                logging.warning(
                        f"Occupied volume ({occupied_volume}) is greater than external space ({external_space}).")
            
            AVF = occupied_volume / external_space
            SVR = 1 / (1 - np.exp(0.1 * (AVF - 0.5)))  # Calculate SVR based on AVF
        
        # Calculate the total flow of calcium ions in exchange between the cytosol and endoplasmic reticulum
        J_ER = self.ca_flow_er_ip3() + self.ca_flow_er_leak() - self.ca_flow_er_pump()
        # self.move_ions(source=self.branch, target=self.branch,
        #                molecule_source="CaCyt", molecule_target="CaER",
        #                amount=(1 - SVR) * J_ER * dt)
        #
        # self.move_ions(source=self.branch, target=self.branch,
        #                molecule_source="CaER", molecule_target="CaCyt",
        #                amount=- ((1 - SVR) / self.er_cytosol_volume_ratio) * J_ER * dt)
        
        self.move_ions(source=self.branch, target=self.branch,
                       molecule_source="CaCyt", molecule_target="CaER",
                       # combined moved ions
                       amount=(1 - SVR) * J_ER * dt * (1 - 1 / self.er_cytosol_volume_ratio))
        
        # Calculate the total flow of calcium ions through the plasma membrane
        J_pm = self.ca_flow_pm_in() + self.ca_flow_pm_glu() - self.ca_flow_pm_out()
        self.move_ions(source=self.branch, target=self.branch.extracellular_space,
                       molecule_source="CaCyt", molecule_target="CaER",
                       amount=SVR * J_pm * dt)
        
        # Calculate the total flow of IP3
        I_total_IP3 = self.ip3_glu() + self.ip3_ca() - self.ip3_eq()
        self.move_ions(source=self.branch, target=self.branch,
                       molecule_source="IP3", molecule_target="IP3",
                       amount=SVR * I_total_IP3 * dt)
        
        # Update concentrations based on the model equations
        
        # Calculate the change in concentration over the time step
        
        # Update concentration in the branch


class AstrocyteNode:
    
    def __init__(self, x, y, radius):
        self.radius = radius
        self.x = int(x)
        self.y = int(y)
    
    def get_position(self):
        return np.array([[self.x], [self.y]])
    
    def copy(self):
        return AstrocyteNode(self.x, self.y, self.radius)
    
    def get_dimension(self):
        return self.x, self.y, self.radius


class MessageLogger:
    def __init__(self, print_messages: bool = True, log_path: Path = None, save_log_every: int = None,
                 timestamp_format="%d-%m-%Y %H:%M:%S") -> None:
        self.messages = []
        self.steps = 0
        self.print_messages = print_messages
        self.timestamp_format = timestamp_format
        self.log_path = log_path
        self.save_log_every = save_log_every
    
    def step(self):
        
        if self.steps % self.save_log_every == 0:
            self.save_messages()
        
        self.steps += 1
    
    def log(self, msg: str, tag: str = 'default', level: int = logging.INFO,
            caller_id: [str, int, uuid.UUID] = 'unknown'):
        timestamp = datetime.now().strftime(self.timestamp_format)
        message = {
            "timestamp": timestamp,
            "step":      self.steps,
            "tag":       tag,
            "level":     level,
            "caller_id": caller_id.hex if isinstance(caller_id, uuid.UUID) else caller_id,
            "message":   msg
            }
        self.messages.append(message)
        
        if self.print_messages:
            pr_msg = f"{caller_id} at step {self.steps}: {msg}"
            # logging.log(level, pr_msg)
            print(pr_msg)
    
    def get_messages(self, filter_criteria: dict = None, n=None):
        """Retrieve filtered messages based on specified criteria.

        Args:
            filter_criteria: A dictionary where each key-value pair specifies a filter for the corresponding message
                attribute.
            n: The number of latest messages to retrieve. If None, all messages are retrieved.

        Returns:
            A list of filtered messages, each message is a dictionary.

        Raises:
            KeyError: If a filter_criteria key does not correspond to any message attribute.
            ValueError: If 'n' is not a non-negative integer or None.
        """
        
        if n is not None and (not isinstance(n, int) or n < 0):
            raise ValueError("'n' should be a non-negative integer or None.")
        
        if self.messages is None or len(self.messages) < 1:
            return []
        
        filtered_messages = self.messages
        possible_keys = filtered_messages[0].keys()
        
        if filter_criteria:
            for key, item in filter_criteria.items():
                if key not in possible_keys:
                    raise KeyError(f"Please choose a filter from {possible_keys}")
                
                item = [item] if not isinstance(item, (list, tuple)) else item
                filtered_messages = [msg for msg in filtered_messages if msg[key] in item]
        
        return filtered_messages[-n:]
    
    def save_messages(self):
        if self.log_path is not None:
            with open(self.log_path, "w") as txt:
                for msg in self.messages:
                    txt.write(
                            f"{msg['timestamp']} - {msg['step']} - {msg['level']} - {msg['caller_id']}: {msg['message']}\n")


class DataLogger:
    def __init__(self, save_path: Union[Path, None], overwrite=False,
                 log_interval: int = 1, save_checkpoint_interval: Union[int, None] = 1):
        
        self.messages = []
        self.steps = 0
        self.last_checkpoint_save = 0
        self.save_path = Path(save_path)
        
        if save_path.is_dir():
            if overwrite:
                shutil.rmtree(save_path)
            else:
                raise FileExistsError(f"{save_path} already exists. Use overwrite to ignore existing files.")
        
        self.tracked_objects = {}
        self.log_settings = {}
        self.log_data = {}
        self.log_interval = log_interval
        self.save_checkpoint_interval = save_checkpoint_interval
    
    def step(self):
        self.steps += 1
        self.log_state()
    
    def register(self, obj, settings: dict = None):
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("Settings should be a dictionary or None.")
        
        obj_id = obj.id
        self.tracked_objects[obj_id] = obj
        
        # Store settings as a copy to prevent accidental modifications
        self.log_settings[obj_id] = settings.copy() if settings is not None else {}
        
        # Initialize log_data for the object
        self.log_data[obj_id] = {}
    
    def unregister(self, obj_id):
        del self.tracked_objects[obj_id]
    
    def log_state(self):
        for obj_id, obj in self.tracked_objects.items():
            if self.steps % self.log_interval == 0:
                state = obj.log_state()
                if state is not None and len(state) > 0:
                    self.log_data[obj_id][self.steps] = state
        
        self.save_checkpoint()
    
    def save_checkpoint(self):
        
        # Define the base directory and file name pattern
        if not self.save_path.exists():
            self.save_path.mkdir()
        
        if self.save_checkpoint_interval is not None and self.save_path is not None:
            
            if self.steps == 0:
                file_path = self.save_path.joinpath(f"settings_{self.steps}.p")
                
                # Prepare the data to be saved
                checkpoint_data = {
                    "log_settings": self.log_settings,
                    }
                
                # Save the settings to a pickle file
                with open(file_path.as_posix(), "wb") as file:
                    pickle.dump(checkpoint_data, file)
                
                # Optionally, log that the checkpoint has been saved
                print(f"Checkpoint saved at step {self.steps} to {file_path}")
            
            if self.steps % self.save_checkpoint_interval == 0:
                
                file_path = self.save_path.joinpath(f"checkpoint_{self.steps}.p")
                
                # steps since last save
                steps_to_save = range(self.last_checkpoint_save, self.steps)
                
                # Prepare the data to be saved
                checkpoint_data = {"steps": steps_to_save, "log_data": {}}
                
                # Extract only the data since the last checkpoint
                for obj_id, data in self.log_data.items():
                    
                    selected = {}
                    for step in steps_to_save:
                        if step in data:
                            selected[step] = data[step]
                    
                    checkpoint_data["log_data"][obj_id] = selected
                
                # Save the data to a JSON file
                with open(file_path.as_posix(), "wb") as file:
                    pickle.dump(checkpoint_data, file)
                
                # Update the last checkpoint save step
                self.last_checkpoint_save = self.steps
                
                # Optionally, log that the checkpoint has been saved
                print(f"Checkpoint saved at step {self.steps} to {file_path}")
    
    def load_checkpoint(self):
        # Implement loading logic here
        pass
