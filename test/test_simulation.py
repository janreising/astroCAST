from astroCAST.simulation import *

class Test_simulation:

    def test_default_parameters(self, frames=100, X=25, Y=25):

        sim = SimData(frames=frames, X=X, Y=Y)

        data_new, shifts_new = sim.simulate()
        assert data_new.shape == (frames, X, Y), f"unexpected video shape: sim > {data_new.shape} vs. ({frames}, {X}, {Y})"
