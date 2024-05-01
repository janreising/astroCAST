import logging
from datetime import time
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display_functions import clear_output, display
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from scipy.stats import poisson
from tifffile import imsave
from tqdm.auto import tqdm

from astrocast.helper import SignalGenerator


class OptionalProgressBar:
    """
    A class to manage an optional progress bar using tqdm.

    This class creates a progress bar that can be optionally displayed and directly manipulated
    if needed. It supports operations like manually setting progress or refreshing the bar.

    Args:
        range_obj: An iterable object over which to iterate.
        show_progress: A boolean flag to control the display of the progress bar.

    Example:
        with OptionalProgressBar(range(100), True) as pb:
            for i in pb:
                process(i)
                pb.pbar.n = 50  # Manually set the progress
                pb.refresh()   # Refresh the progress bar display
    """
    
    def __init__(self, range_obj: Iterable, show_progress: bool = False):
        self.range_obj = range_obj
        self.show_progress = show_progress
        self.pbar = None
    
    def __enter__(self):
        if self.show_progress:
            self.pbar = tqdm(self.range_obj)
        else:
            self.pbar = self.range_obj
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.show_progress:
            self.pbar.close()
    
    def __iter__(self):
        return iter(self.pbar)
    
    def refresh(self):
        if self.show_progress:
            self.pbar.refresh()
    
    def set_progress(self, n):
        if self.show_progress:
            self.pbar.n = n


class SimData:
    
    def __init__(self, frames, X, Y, n_blobs=10, noise_amplitude=0.2, blob_amplitude=5,
                 max_drift=(0.01, 0.1), max_jitter=1, background_noise=1.0, shot_noise=0.2):
        self.astrocytes = None
        self.frames = frames
        self.X = X
        self.Y = Y
        self.n_blobs = n_blobs
        self.noise_amplitude = noise_amplitude
        self.blob_amplitude = blob_amplitude
        self.data = None
        self.shifts = None
        self.max_jitter = max_jitter
        self.background_noise = background_noise
        self.shot_noise = shot_noise
        
        # save keeping
        self.vor = None
        
        # Determine the type of drift (linear or quadratic) and randomize parameters
        if isinstance(max_drift, tuple):
            self.drift_type = 'quadratic'
            self.a_x = np.random.uniform(-max_drift[0], max_drift[0])
            self.a_y = np.random.uniform(-max_drift[0], max_drift[0])
            self.b_x = np.random.uniform(-max_drift[1], max_drift[1])
            self.b_y = np.random.uniform(-max_drift[1], max_drift[1])
        else:
            self.drift_type = 'linear'
            self.m_x = np.random.uniform(-max_drift, max_drift)
            self.m_y = np.random.uniform(-max_drift, max_drift)
    
    def generate_gaussian_blob(self, x, y, x0, y0, sigma, amplitude):
        return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    def generate_base_image(self, padding=0):
        X_padded = self.X + 2 * padding
        Y_padded = self.Y + 2 * padding
        noise_floor = self.background_noise * np.random.randn(X_padded, Y_padded)
        x = np.arange(X_padded)
        y = np.arange(Y_padded)
        x, y = np.meshgrid(x, y)
        for _ in range(self.n_blobs):
            x0 = np.random.randint(X_padded)
            y0 = np.random.randint(Y_padded)
            sigma = np.random.uniform(5, 15)
            if np.random.rand() < 0.5:
                amplitude = np.random.uniform(1, self.blob_amplitude)
            else:
                amplitude = np.random.uniform(-1, -self.blob_amplitude)
            noise_floor += self.generate_gaussian_blob(x, y, x0, y0, sigma, amplitude)
        return noise_floor
    
    def calculate_padding(self):
        if self.drift_type == 'linear':
            max_drift_x = self.m_x * self.frames
            max_drift_y = self.m_y * self.frames
        else:
            max_drift_x = self.a_x * self.frames ** 2 + self.b_x * self.frames
            max_drift_y = self.a_y * self.frames ** 2 + self.b_y * self.frames
        total_max_shift_x = max_drift_x + self.frames * self.max_jitter
        total_max_shift_y = max_drift_y + self.frames * self.max_jitter
        return int(max(total_max_shift_x, total_max_shift_y))
    
    def simulate(self):
        padding = self.calculate_padding()
        large_base_image = self.generate_base_image(padding)
        self.data = np.zeros((self.frames, self.X, self.Y))
        shifts = []
        for t in range(self.frames):
            jitter_x = np.random.uniform(-self.max_jitter, self.max_jitter)
            jitter_y = np.random.uniform(-self.max_jitter, self.max_jitter)
            
            if self.drift_type == 'linear':
                drift_x = self.m_x * t
                drift_y = self.m_y * t
            else:
                drift_x = self.a_x * t ** 2 + self.b_x * t
                drift_y = self.a_y * t ** 2 + self.b_y * t
            
            shift_x = jitter_x + drift_x
            shift_y = jitter_y + drift_y
            shifts.append((shift_x, shift_y))
            
            # Apply affine transformation
            transformation_matrix = np.array([[1, 0, -shift_x], [0, 1, -shift_y], [0, 0, 1]])
            shifted_image = affine_transform(large_base_image, transformation_matrix[:2, :2],
                                             offset=transformation_matrix[:2, 2], mode='nearest', order=1)
            
            shifted_image += self.shot_noise * np.random.randn(self.X + 2 * padding, self.Y + 2 * padding)
            start_x = padding
            end_x = start_x + self.X
            start_y = padding
            end_y = start_y + self.Y
            self.data[t] = shifted_image[start_x:end_x, start_y:end_y]
        self.shifts = shifts
        return self.data, shifts
    
    def save(self, path):
        path = Path(path)
        imsave(path, self.data.astype(np.float32))
    
    def plot_overview(self):
        shifts_x = [s[0] for s in self.shifts]
        shifts_y = [s[1] for s in self.shifts]
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        axs[0].imshow(self.data[0], cmap='gray')
        axs[0].set_title('First Frame')
        
        axs[1].imshow(self.data[self.frames // 2], cmap='gray')
        axs[1].set_title('Middle Frame')
        
        axs[2].imshow(self.data[-1], cmap='gray')
        axs[2].set_title('Last Frame')
        
        axs[3].plot(shifts_x, label='Shift X')
        axs[3].plot(shifts_y, label='Shift Y')
        axs[3].set_title('Shifts over Time')
        axs[3].set_xlabel('Frame')
        axs[3].legend()
        
        for ax in axs[:3]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_voronoi(self, num_cells):
        """
        Generate a Voronoi diagram based on random seed points.

        Parameters:
        - num_cells: Number of Voronoi cells (astrocytes) to create.

        Returns:
        - Voronoi object containing the segmentation information.
        """
        
        # Generate random seed points
        seeds = np.column_stack((np.random.randint(0, self.X, num_cells),
                                 np.random.randint(0, self.Y, num_cells)))
        
        # Compute the Voronoi segmentation
        vor = Voronoi(seeds)
        self.vor = vor
        
        return vor
    
    def plot_voronoi(self, figsize=(8, 8)):
        
        # Plot the Voronoi diagram
        fig, ax = plt.subplots(figsize=figsize)
        voronoi_plot_2d(self.vor, ax=ax, show_points=True, show_vertices=False)
        ax.set_xlim([0, self.X])
        ax.set_ylim([0, self.Y])
        ax.set_title("Voronoi Segmentation for Astrocytes")
        plt.show()
    
    def plot_single_voronoi_cell(self, region_index=0, figsize=(8, 8), ax=None):
        
        # Get the vertices of the specified region
        # region_vertices = vor.regions[region_index]
        region_vertices, polygon, center = self.get_voronoi(region_index)
        
        # If the region is valid (non-empty and doesn't contain -1), plot it
        if len(region_vertices) > 0 and -1 not in region_vertices:
            polygon = [self.vor.vertices[i] for i in region_vertices]
            polygon = np.array(polygon)
            
            # Plot the Voronoi cell
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            
            ax.fill(polygon[:, 0], polygon[:, 1], edgecolor='k', facecolor='none')
        
        else:
            print(f"Region {region_index} is not a valid region")
    
    def get_voronoi(self, index):
        
        vor = self.vor
        
        region = vor.regions[vor.point_region[index]]
        polygon = vor.vertices[region]
        center = vor.points[index].tolist()
        
        return region, polygon, center
    
    def generator_voronoi(self):
        
        i = 0
        while i < len(self.vor.point_region):
            yield self.get_voronoi(i)
            i += 1
    
    def populate_astrocytes(self, branch_length=2, kill_range=2.25, attraction_range=8, num_leaves=200,
                            max_iterations=100, max_astrocytes: int = None, show_progress=False, plot=False):
        
        astrocytes = []
        
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        else:
            ax = None
        
        for r in tqdm(self.generator_voronoi(), total=len(self.vor.point_region)):
            
            region, polygon, center = r
            
            if len(region) > 0 and -1 not in region and np.min(polygon) > 0 and np.max(polygon) < max(self.X, self.Y):
                
                ast = Astrocyte(voronoi_cell=polygon, cell_center=center, ax=None,
                                num_leaves=num_leaves, max_iterations=max_iterations, plot_updates=False,
                                branch_length=branch_length, attraction_range=attraction_range, kill_range=kill_range,
                                show_progress=show_progress)
                
                astrocytes.append(ast)
                
                if plot:
                    ax = ast.plot(ax=ax,
                                  line_color='green', linewidth=1, line_alpha=1, thickness_multiplier=0.04,
                                  scatter_color="gray", scatter_size=1, scatter_marker="x", scatter_alpha=0.3,
                                  plot_original_scatter=False)
                    
                    ax.set_xlim(0, self.X)
                    ax.set_ylim(0, self.Y)
                    
                    clear_output(wait=True)
                    display(ax.get_figure())
                
                if max_astrocytes is not None and len(astrocytes) >= max_astrocytes:
                    break
        
        self.astrocytes = astrocytes
        return astrocytes
    
    def plot(self, figsize=(5, 5), ax=None):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        for ast in self.astrocytes:
            ax = ast.plot(ax=ax,
                          line_color='green', linewidth=1, line_alpha=1, thickness_multiplier=0.04,
                          scatter_color="gray", scatter_size=1, scatter_marker="x", scatter_alpha=0.3,
                          plot_original_scatter=False)
        
        ax.set_xlim(0, self.X)
        ax.set_ylim(0, self.Y)
    
    def generate_activity(self, signal_generator: Union[SignalGenerator, List[SignalGenerator]] = None,
                          event_probability=0.5,
                          noise: Tuple[float, float] = None, signal_noise: Tuple[float, float] = None,
                          blur: Tuple[float, int] = None):
        
        if noise is None:
            activity = np.zeros((self.frames, self.X, self.Y))
        else:
            loc, scale = noise
            activity = np.random.normal(loc=loc, scale=scale, size=(self.frames, self.X, self.Y))
        
        for i, ast in tqdm(enumerate(self.astrocytes)):
            
            signal = ast.generate_signal(num_frames=self.frames, signal_generator=signal_generator,
                                         event_probability=event_probability, noise=signal_noise)
            ast_activity = ast.generate_activity(signal=signal, grid=None, blur=None)
            
            min_x, min_y = np.min(ast.voronoi_cell, axis=0)
            min_x, min_y = int(min_x), int(min_y)
            
            max_x, max_y = min_x + ast_activity.shape[2], min_y + ast_activity.shape[1]
            
            try:
                activity[:, min_y:max_y, min_x:max_x] += ast_activity
            except ValueError as err:
                logging.warning(f"encountered error in astrocyte {i}: {err}")
        
        if blur is not None:
            sigma, radius = blur
            activity = gaussian_filter(activity, sigma=sigma, radius=radius)
        
        return activity
    
    def get_events(self, save_path: Union[str, Path] = None):
        
        df = []
        for i, ast in enumerate(self.astrocytes):
            
            df_ = ast.get_events()
            if df_ is not None:
                df_['ast_id'] = i
                df.append(df_)
        
        df = pd.concat(df, axis=0)
        
        if save_path is not None:
            df.to_pickle(save_path)
        
        return df
    
    @staticmethod
    def load_events(save_path: Union[str, Path]):
        return pd.read_pickle(save_path)


class Branch:
    def __init__(self, start, end, direction, parent=None, size=1.0):
        self.start = start
        self.end = end
        self.direction = direction
        self.parent = parent
        self.children = []
        self.attractors = []
        self.distance_from_root = 0 if parent is None else parent.distance_from_root + 1
        self.num_children = 0
        self.size = size
        self.grown = False
    
    def add_child(self, child):
        self.children.append(child)
        self.increment_num_children()
    
    def increment_num_children(self):
        self.num_children += 1
        
        if self.parent is not None:
            self.parent.increment_num_children()
    
    def grow(self, length):
        """Grow the branch in its current direction by a specified length."""
        self.end = (self.end[0] + length * self.direction[0], self.end[1] + length * self.direction[1])
    
    def update_direction(self, randomness_factor=0.2):
        """Update the growth direction based on the attractors."""
        if self.attractors:
            avg_direction = [0, 0]
            for attractor in self.attractors:
                dir_to_attractor = (
                    attractor[0] - self.end[0],
                    attractor[1] - self.end[1]
                    )
                magnitude = np.sqrt(dir_to_attractor[0] ** 2 + dir_to_attractor[1] ** 2)
                avg_direction[0] += dir_to_attractor[0] / magnitude
                avg_direction[1] += dir_to_attractor[1] / magnitude
            
            avg_direction = [avg_direction[0] / len(self.attractors), avg_direction[1] / len(self.attractors)]
            
            # Adding randomness to the growth direction
            random_vector = self.random_growth_vector()
            random_magnitude = np.sqrt(random_vector[0] ** 2 + random_vector[1] ** 2)
            avg_direction[0] += randomness_factor * random_vector[0] / random_magnitude
            avg_direction[1] += randomness_factor * random_vector[1] / random_magnitude
            
            self.direction = avg_direction
    
    def random_growth_vector(self, magnitude=0.1):
        """Generate a random growth vector."""
        theta = np.random.uniform(0, 2 * np.pi)
        x = magnitude * np.cos(theta)
        y = magnitude * np.sin(theta)
        return (x, y)


class Astrocyte:
    
    def __init__(self, voronoi_cell, cell_center, num_leaves=50,
                 branch_length=1, attraction_range=2, kill_range=0.3,
                 randomness_factor=0.2, max_iterations=100,
                 show_progress: bool = False, plot_updates=False, plot_sleep=0.2, ax=None):
        
        self.mask = None
        self.grid = None
        self.events = {}
        self.ax = ax
        
        self.cell_center = cell_center
        self.voronoi_cell = voronoi_cell
        self.delaunay = Delaunay(self.voronoi_cell)
        
        self.leaves = self.generate_leaves(num_leaves)
        self.leaves_x = [leaf[0] for leaf in self.leaves]
        self.leaves_y = [leaf[1] for leaf in self.leaves]
        
        self.root_branch = Branch(cell_center, cell_center, (0, 1))
        self.branches = [self.root_branch]
        self.extremities = [self.root_branch]
        self.grow_tree(branch_length=branch_length,
                       attraction_range=attraction_range,
                       kill_range=kill_range,
                       randomness_factor=randomness_factor,
                       max_iterations=max_iterations, plot=plot_updates, plot_sleep=plot_sleep,
                       show_progress=show_progress)
    
    def generate_leaves(self, num_leaves):
        
        leaves = []
        min_x, min_y = self.voronoi_cell.min(axis=0)
        max_x, max_y = self.voronoi_cell.max(axis=0)
        
        while len(leaves) < num_leaves:
            point = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
            if self.delaunay.find_simplex(point) >= 0:  # Check if the point is within the Voronoi cell
                leaves.append(point)
        
        return leaves
    
    def random_growth_vector(self, magnitude=1):
        alpha = np.random.uniform(0, np.pi)
        theta = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(theta) * np.sin(alpha), np.sin(theta) * np.sin(alpha)]) * magnitude
    
    def grow_tree(self, branch_length=0.2, attraction_range=1.0, kill_range=0.1,
                  randomness_factor=0.2, max_iterations=100, plot=True, plot_sleep=0.2, show_progress: bool = False):
        
        if plot:
            ax = self.plot() if self.ax is None else self.ax
        
        leaves = self.leaves.copy()
        n_leaves = len(leaves)
        
        with OptionalProgressBar(range(max_iterations), show_progress) as pbar:
            for iteration in pbar:
                if not leaves:
                    break
                
                # New growth control structure
                for branch in self.extremities:
                    branch.grown = True
                
                # Remove attractors in kill range
                leaves = [leaf for leaf in leaves if all(
                        np.sqrt((leaf[0] - branch.end[0]) ** 2 + (leaf[1] - branch.end[1]) ** 2) > kill_range for branch
                        in
                        self.branches)]
                if len(leaves) < 1:
                    break
                
                # clear active attractors
                active_attractors = []
                for b in self.branches:
                    b.attractors = []
                
                # Each attractor is associated to its closest branch, if in attraction range
                for leaf in leaves:
                    min_distance = np.inf
                    closest_branch = None
                    for branch in self.branches:
                        distance = np.sqrt((leaf[0] - branch.end[0]) ** 2 + (leaf[1] - branch.end[1]) ** 2)
                        if distance < attraction_range and distance < min_distance:
                            min_distance = distance
                            closest_branch = branch
                    
                    if closest_branch is not None:
                        closest_branch.attractors.append(leaf)
                        active_attractors.append(leaf)
                
                # add branches
                if len(active_attractors) > 0:
                    
                    self.extremities = []
                    new_branches = []
                    
                    for i, branch in enumerate(self.branches):
                        
                        if len(branch.attractors) > 0:
                            
                            # Compute the new growth direction
                            direction = np.array([0.0, 0.0])
                            for attractor in branch.attractors:
                                direction += (np.array(attractor) - np.array(branch.end)).astype(float)
                            direction /= len(branch.attractors)
                            direction += self.random_growth_vector(randomness_factor)
                            direction /= np.linalg.norm(direction)
                            
                            # Create a new branch growing in the updated direction
                            new_end_point = (branch.end[0] + branch_length * direction[0],
                                             branch.end[1] + branch_length * direction[1])
                            
                            if self.delaunay.find_simplex(new_end_point) >= 0:
                                new_branch = Branch(branch.end, new_end_point, direction, parent=branch)
                                new_branches.append(new_branch)
                                self.extremities.append(new_branch)
                                
                                # add new branch to parent branch
                                branch.add_child(new_branch)
                        
                        else:
                            if len(branch.children) < 1:
                                self.extremities.append(branch)
                    
                    self.branches.extend(new_branches)
                
                else:
                    
                    # Grow the extremities of the tree
                    for i, extremity in enumerate(self.extremities):
                        
                        if not extremity.grown:
                            continue
                        
                        start = extremity.end
                        direction = extremity.direction + self.random_growth_vector(randomness_factor)
                        end = (extremity.end[0] + branch_length * direction[0],
                               extremity.end[1] + branch_length * direction[1])
                        
                        if self.delaunay.find_simplex(end) >= 0:
                            new_branch = Branch(start, end, direction, parent=extremity)
                            extremity.children.append(new_branch)
                            self.branches.append(new_branch)
                            self.extremities[i] = new_branch
                
                if show_progress and plot:
                    clear_output(wait=True)
                    ax.clear()
                    ax = self.plot(branches=self.branches, leaves=leaves, ax=ax)
                    display(ax.get_figure())
                    time.sleep(plot_sleep)
                
                if show_progress and not plot:
                    pbar.set_progress(n_leaves - len(leaves))
                    pbar.refresh()
        
        if plot:
            clear_output(wait=True)
    
    def plot(self, branches=None, leaves=None,
             line_color='green', linewidth=1, line_alpha=0.3, thickness_multiplier=0.04,
             scatter_color="red", scatter_size=2, scatter_marker="x", scatter_alpha=1,
             plot_original_scatter=True,
             figsize=(10, 10), ax=None):
        
        if branches is None:
            branches = self.branches
        
        if leaves is None:
            leaves = self.leaves
        
        leaves_x = [leaf[0] for leaf in leaves]
        leaves_y = [leaf[1] for leaf in leaves]
        leaves_x_o, leaves_y_o = self.leaves_x, self.leaves_y
        
        # Extract branch coordinates for visualization and count the number of branches generated
        branch_coords = [((branch.start, branch.end), branch.num_children) for branch in branches]
        
        # Visualize the resulting tree structure with the new branch coordinates
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        self.plot_voronoi_cell(ax=ax)
        
        for coord, num_children in branch_coords:
            ax.plot([coord[0][0], coord[1][0]], [coord[0][1], coord[1][1]],
                    color=line_color, linewidth=linewidth + thickness_multiplier * num_children, alpha=line_alpha)
        
        if plot_original_scatter:
            ax.scatter(leaves_x_o, leaves_y_o, color='gray', s=2, alpha=0.5)
        
        ax.scatter(leaves_x, leaves_y, color=scatter_color, s=scatter_size, marker=scatter_marker, alpha=scatter_alpha)
        
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Resulting Tree Structure')
        
        return ax
    
    def plot_voronoi_cell(self, figsize=(8, 8), ax=None):
        
        # Get the Voronoi object
        polygon = np.array(self.voronoi_cell)
        
        # Plot the Voronoi cell
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        ax.fill(polygon[:, 0], polygon[:, 1], edgecolor='k', facecolor='none')
    
    def generate_signal(self, num_frames: int, signal_generator: Union[SignalGenerator, List[SignalGenerator]] = None,
                        event_probability=0.5, noise: Tuple[float, float] = None):
        
        events = {}
        
        if signal_generator is None:
            signal_generator = [SignalGenerator()]
        elif isinstance(signal_generator, SignalGenerator):
            signal_generator = [signal_generator]
        
        if noise is None:
            signal = np.zeros(num_frames)
        else:
            loc, scale = noise
            signal = np.random.normal(loc=loc, scale=scale, size=num_frames)
        
        z = 0
        e = 0
        while z < num_frames:
            
            if np.random.random() < event_probability:
                gen = np.random.choice(signal_generator)
                event = gen.generate_signal()
                event_len = len(event)
                
                z0 = z
                z1 = min(num_frames, z + event_len)
                signal[z0:z1] += event[:z1 - z0]
                z += z1 - z0
                
                events[e] = {
                    "eid":   e,
                    "trace": event,
                    "z0":    z0,
                    "z1":    z1,
                    "dz":    z1 - z0
                    }
                e += 1
            
            z += 1
        
        self.events = events
        return signal
    
    def generate_activity(self, signal: np.ndarray, grid=None, blur: Tuple[float, int] = None):
        
        if grid is None:
            grid = AstrocyteGrid(self, grid_resolution=1).get_grid(blur=blur)
        
        Z = len(signal)
        X, Y = grid.shape
        self.grid = grid
        self.mask = grid > 0
        
        activity = np.zeros((Z, X, Y))
        for z in range(Z):
            activity[z, :, :] = signal[z] * grid
        
        return activity
    
    def get_events(self):
        
        events = self.events
        mask = self.mask
        grid = self.grid
        
        min_x, min_y = np.min(self.voronoi_cell, axis=0)
        
        if grid is not None:
            max_x = min_x + grid.shape[0]
            max_y = min_y + grid.shape[1]
        else:
            max_x, max_y = np.min(self.voronoi_cell, axis=0)
        
        df = {k: [] for k in ['eid', 'trace', 'mask', 'mask_intensity', 'x0', 'y0', 'x1', 'y1', 'z0', 'z1',
                              'dz', 'dx', 'dy']}
        
        for k in events.keys():
            
            ev = events[k]
            df['eid'].append(ev["eid"])
            df['trace'].append(ev["trace"])
            df['z0'].append(ev["z0"])
            df['z1'].append(ev["z1"])
            df['x0'].append(min_x)
            df['x1'].append(max_x)
            df['y0'].append(min_y)
            df['y1'].append(max_y)
            df['dz'].append(ev["dz"])
            df['dx'].append(max_x - min_x)
            df['dy'].append(max_y - min_y)
            df['mask'].append(mask.flatten() if mask is not None else np.array([]))
            df['mask_intensity'].append(grid.flatten() if grid is not None else np.array([]))
        
        df = pd.DataFrame(df)
        return df


class AstrocyteGrid:
    def __init__(self, astrocyte, grid_resolution, grid_size=None):
        self.astrocyte = astrocyte
        self.grid_resolution = grid_resolution
        
        if grid_size is None:
            self.grid_size, (self.x_offset, self.y_offset) = self.calculate_grid_size()
        else:
            self.grid_size = grid_size
            self.x_offset, self.y_offset = 0, 0
        
        self.grid = np.zeros(self.grid_size)
        self.add_branches_to_grid()
    
    def calculate_grid_size(self):
        # Ensure voronoi_cell is a property of astrocyte and is a structure that supports min and max methods
        if hasattr(self.astrocyte, 'voronoi_cell') and isinstance(self.astrocyte.voronoi_cell, np.ndarray):
            min_x, min_y = np.min(self.astrocyte.voronoi_cell, axis=0)
            max_x, max_y = np.max(self.astrocyte.voronoi_cell, axis=0)
            size_x = int((max_x - min_x) / self.grid_resolution) + 2
            size_y = int((max_y - min_y) / self.grid_resolution) + 2
            return (size_y, size_x), (min_x, min_y)
        else:
            raise AttributeError(
                    f"Astrocyte instance does not have a 'voronoi_cell' attribute or it's not a numpy array."
                    f" {type(self.astrocyte.voronoi_cell)}")
    
    def add_branches_to_grid(self):
        for branch in self.astrocyte.branches:
            self.rasterize_branch(branch)
    
    def rasterize_branch(self, branch, thickness_multiplier=0.04):
        start_idx = list(self.point_to_grid_index(branch.start))
        end_idx = list(self.point_to_grid_index(branch.end))
        
        dx = abs(end_idx[0] - start_idx[0])
        dy = abs(end_idx[1] - start_idx[1])
        sx = -1 if start_idx[0] > end_idx[0] else 1
        sy = -1 if start_idx[1] > end_idx[1] else 1
        err = dx - dy
        
        # Calculate thickness and ensure it is an integer
        thickness = int(1 + thickness_multiplier * branch.num_children)
        
        while True:
            self.mark_grid_cell(start_idx[0], start_idx[1], thickness)  # Mark cell with thickness
            
            if start_idx[0] == end_idx[0] and start_idx[1] == end_idx[1]:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                start_idx[0] += sx
            if e2 < dx:
                err += dx
                start_idx[1] += sy
    
    def mark_grid_cell(self, x, y, thickness):
        
        # x_offset = int(self.x_offset)
        # y_offset = int(self.y_offset)
        x_offset = 0
        y_offset = 0
        
        # Convert thickness to an integer, if it's not already
        half_thickness = int(thickness // 2)
        for i in range(-half_thickness, half_thickness + 1):
            for j in range(-half_thickness, half_thickness + 1):
                # Ensure x+i and y+j are within the grid bounds before marking the cell
                if 0 <= y + j < self.grid_size[0] and 0 <= x + i < self.grid_size[1]:
                    try:
                        self.grid[y + j, x + i] = 1  # Mark the cell
                    except IndexError:
                        print(f"error at {y + j}, {x + i}")
    
    def point_to_grid_index(self, point):
        # Convert a point in space to a grid index
        return (
            int((point[0] - self.astrocyte.voronoi_cell.min(axis=0)[0]) / self.grid_resolution),
            int((point[1] - self.astrocyte.voronoi_cell.min(axis=0)[1]) / self.grid_resolution)
            )
    
    def get_grid(self, blur: Tuple[float, int] = None):
        grid = self.grid
        
        if blur is not None:
            sigma, radius = blur
            grid = gaussian_filter(grid, sigma=sigma, radius=radius)
        
        return grid
    
    def plot_grid(self, blur: Tuple[float, int] = None):
        
        grid = self.get_grid(blur=blur)
        
        plt.imshow(grid, cmap='Greys', interpolation='nearest', origin='lower')
        plt.title('Astrocyte Branch Structure on Grid')
        plt.xlabel('X-axis (grid units)')
        plt.ylabel('Y-axis (grid units)')
        # plt.gca().invert_yaxis()  # This inverts the y-axis to match the astrocyte plot's orientation
        plt.show()


def astrocyte_dynamics(y, t,
                       A, psyn, D_Ca, delta_x,  # check parameters
                       r=0.5,
                       gamma=1.0, IP3_0=1.0e-6, tau_IP3=1.0,  # check values
                       c0=2.0e-6, c1=0.185, v1=6.0, v2=0.11, v3=2.2e-6,
                       v5=0.025e-6, v6=0.2e-6, k1=0.5, k2=1.0e-6, k3=0.1e-6,
                       a2=0.14e-6, d1=0.13e-6, d2=1.049e-6, d3=943.4e-9, d5=82.0e-9,
                       alpha=0.8, v4=0.25e-6, v_g=0.062e-6, k4=1.1e-6, k_g=0.78e-6, n=2):
    """
    Astrocyte dynamics function that models calcium dynamics in astrocytes.

    Parameters:
    - y: List of state variables [Ca_c, Ca_ER, IP3, Glu, h]
    - t: Time variable
    - r: Parameter representing local AVF
    - IP3_0: Steady-state concentration of IP3 (μM); resting level
    - tau_IP3: Time constant for IP3 equilibration (s)
    - c0: Total [Ca2+] in terms of cytosolic vol (μM)
    - c1: (ER vol)/(cytosolic vol)
    - v1: Max Ca2+ channel flux (s^-1)
    - v2: Ca2+ leak flux constant (s^-1)
    - v3: Max Ca2+ uptake (μM s^-1)
    - v5: Rate of calcium leak across the plasma membrane (μM s^-1)
    - v6: Maximal rate of activation dependent calcium influx (μM s^-1)
    - k1: Rate constant of calcium extrusion (s^-1)
    - k2: Half-saturation constant for agonist-dependent calcium entry (μM)
    - k3: Activation constant for ATP-Ca2+ pump (μM)
    - a2: Ca2+ inhibition constant (μM^-1 s^-1)
    - d1: Dissociation constant for IP3 (μM)
    - d2: Dissociation constant for Ca2+ inhibition (μM)
    - d3: Receptor dissociation constant for IP3 (μM)
    - d5: Ca2+ activation constant (μM)
    - alpha: Alpha constant for J_delta
    - v4: Max rate of IP3 production (μM s^-1)
    - v_g: Rate of IP3 production through glutamate (μM s^-1)
    - k4: Dissociation constant for Ca2+ stimulation of IP3 production (μM)
    - k_g: Dissociation constant for glutamate stimulation of IP3 production (μM)
    - n: Exponent for Hill equation in J_glutamate

    Returns:
    - dydt: List of derivatives [dCa_c_dt, dCa_ER_dt, dIP3_dt, dGlu_dt, dh_dt]
    """
    
    Ca_c, Ca_ER, IP3, Glu, h = y
    
    # Compute linked parameter s based on r
    s = 1 / (1 - np.exp(0.1 * (r - 0.5)))
    
    # Definitions for J_IP3
    m_inf = IP3 / (IP3 + d1)
    n_inf = Ca_c / (Ca_c + d5)
    Q2 = d2 * ((IP3 + d1) / (IP3 + d3))
    h_inf = Q2 / (Q2 + Ca_c)
    tau_h = 1 / (a2 * (Q2 + Ca_c))
    dh_dt = (h_inf - h) / tau_h  # We'll need to return this as part of dydt
    
    # definitions based on Ullah et al. (2006)
    
    # Stochastic glutamate source based on a Poisson process
    xi_p_t = A * poisson.rvs(psyn)  # This is a simplification
    
    # IP3 turnover
    J_ip3_delta = v4 * ((Ca_c + (1 - alpha) * k4) / (Ca_c + k4))
    J_ip3_glutamate = (v_g * Glu ** n) / (k_g ** n + Glu ** n)
    I_diff = 0  # Placeholder for IP3 diffusion term (I_diff)
    # ?
    I_ip3_Ca = J_ip3_delta  # Ca2+-stimulated IP3 production
    I_ip3_Glu = J_ip3_glutamate  # Glutamate-driven IP3 production
    I_ip3_eq = (IP3 - IP3_0) / tau_IP3
    
    # Total flows across ER
    J_leak = c1 * v1 * (Ca_ER - Ca_c)  # leak from ER
    J_pump = (v3 * Ca_c ** 2) / (Ca_c ** 2 + k3 ** 2)  # retrieval of Ca2+ into ER
    J_IP3 = c1 * v1 * m_inf ** 3 * n_inf ** 3 * h ** 3 * (Ca_ER - Ca_c)
    J_ER = J_IP3 + J_leak - J_pump  # total flow of Ca2+ (cytosol > ER; ER membrane)
    
    # Total flow across PM
    J_in = v5 + v6 * (IP3 ** 2 / (k2 ** 2 + IP3 ** 2))  # Constant Ca2+ influx + IP3 stimulated influx
    J_out = k1 * Ca_c  # Extrusion current
    J_Glu = gamma * Glu  # direct effect of Glutamate on Ca2+
    J_pm = J_in + J_Glu - J_out  # total flow of C2+ (cytosol > extracellular space; plasma membrane)
    
    # Diffusion??
    D_star_Ca = r ** 2 * D_Ca / delta_x ** 2
    sum_neighbors_Ca = 0  # Placeholder for sum over nearest neighbors; assume single astrocyte for now
    J_diff = D_star_Ca * (sum_neighbors_Ca - Ca_c)
    
    # Define the differential equations
    dCa_c_dt = (1 - s) * J_ER + s * J_pm + J_diff
    dCa_ER_dt = -1 - s * J_ER
    dIP3_dt = s * (I_ip3_Glu + I_ip3_Ca) - I_ip3_eq + I_diff
    dGlu_dt = Glu_amb - Glu / tau_Glu + xi_p_t + G_diff  # G_diff still to be defined
    
    dydt = [dCa_c_dt, dCa_ER_dt, dIP3_dt, dGlu_dt]
    return dydt
