import numpy as np
import pytest

from astrocast.sim import EnvironmentGrid


class TestEnvironmentGrid:
    
    def test_creation(self, grid_size=(100, 100), molecules=("glutamate", "calcium"),
                      diffusion_rate=0.1, dt=1):
        eg = EnvironmentGrid(grid_size=grid_size, molecules=molecules, diffusion_rate=diffusion_rate, dt=dt,
                             dtype='float16')
        
        eg.step()
    
    def test_random_starting_concentration(self, n_spots=10, grid_size=(100, 100), molecules=("glutamate", "calcium"),
                                           diffusion_rate=0.1, dt=1):
        eg = EnvironmentGrid(grid_size=grid_size, molecules=molecules, diffusion_rate=diffusion_rate, dt=dt,
                             dtype='float16')
        
        locations = eg.set_random_starting_concentration(molecule="glutamate", n_spots=n_spots)
        assert len(locations) == n_spots, f"Number of spots: {len(locations)} does not match number of spots: {n_spots}"
        
        for x, y, concentration in locations:
            assert np.allclose(concentration, eg.get_concentration_at((x, y), molecule="glutamate"))
        
        eg.step()
        for x, y, concentration in locations:
            assert concentration > eg.get_concentration_at((x, y), molecule="glutamate")
    
    def test_borders(self, n_spots=1, grid_size=(16, 16), molecules=("glutamate", "calcium"),
                     diffusion_rate=1, dt=10):
        eg = EnvironmentGrid(grid_size=grid_size, molecules=molecules, diffusion_rate=diffusion_rate, dt=dt,
                             dtype='float16')
        
        eg.set_random_starting_concentration(molecule="glutamate", n_spots=n_spots)
        eg.step(int(1e4))
        
        arr, _ = eg.shared_arrays["glutamate"]
        
        border_sum = np.sum(arr[0, :]) + np.sum(arr[-1, :]) + np.sum(arr[:, 0]) + np.sum(arr[:, -1])
        assert border_sum == 0
    
    def test_set_concentration(self):
        raise NotImplementedError
    
    @pytest.mark.parametrize("concentration", [-1, 1])
    def test_update_concentration(self, concentration):
        raise NotImplementedError


class TestGlutamateReleaseManager:
    
    def test_vanilla(self):
        raise NotImplementedError
