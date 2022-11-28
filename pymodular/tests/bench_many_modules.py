import unittest
import pymodular as pym
import numpy as np
import time

class AddModule(pym.Module):
    def _response(self, x):
        return x*0.99+1
    def _sensitivity(self, dy: float):
        return dy*0.99

class TestComplex(unittest.TestCase):
    def test_many_modules(self):
        N = 1000
        start = time.perf_counter()
        fn = pym.Network()
        s = pym.Signal('s_start', 1.0)
        for i in range(N):
            s = fn.append(AddModule([s],[pym.Signal(f's{i}')]))
        elapsed = time.perf_counter() - start
        print(f"Function with {len(fn.mods)} modules, SETUP -- Elapsed time: {elapsed:0.4f} seconds")

        start = time.perf_counter()
        fn.response()
        elapsed = time.perf_counter() - start
        print(f"Response result = {s.state} -- Elapsed time: {elapsed:0.4f} seconds")

        s.sensitivity = 1.0
        start = time.perf_counter()
        fn.sensitivity()
        elapsed = time.perf_counter() - start
        print(f"Sensitivity -- Elapsed time: {elapsed:0.4f} seconds")


if __name__=="__main__":
    unittest.main()
