import unittest
from pathlib import Path

from lammps_util import Dump
import lammps_util

from test_lammps_util import TestLAMMPSUtil

DUMP_PATH = Path("./dump.surface")


class TestSurfaceMethods(TestLAMMPSUtil):
    def test_calc_surface(self):
        dump = Dump(DUMP_PATH)
        run_dir = Path("/tmp/surface")
        if run_dir.exists() == False:
            run_dir.mkdir()
        sigma = lammps_util.calc_surface(dump, run_dir, 5.43, 5.43 * 5, 5)
        self.assertAlmostEqual(sigma, 5.43 / 2, 3)


if __name__ == "__main__":
    unittest.main()
