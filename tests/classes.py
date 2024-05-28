import unittest
import numpy as np
from pathlib import Path

from lammps_util import Atom, Cluster, Dump

from test_lammps_util import TestLAMMPSUtil

DUMP_PATH = Path("./dump.during")


class TestClasses(TestLAMMPSUtil):

    def test_Dump(self):
        dump = Dump(DUMP_PATH)
        self.assertEqual(dump.name, str(DUMP_PATH))

        self.assertEqual(
            dump.keys,
            ["id", "type", "x", "y", "z", "vx", "vy", "vz", "c_atom_ke"],
        )
        self.assertEqual(len(dump["id"]), 60)

        with open(DUMP_PATH, "r") as file:
            count_lines = len(file.readlines())
        n_steps = 8100 / 25 + 1

        self.assertEqual(len(dump.timesteps), n_steps + 1)
        self.assertEqual(dump.timestep_i, 0)
        self.assertEqual(dump.timesteps[0], (0, 0))
        self.assertEqual(
            dump.timesteps[100], (100 * 25, 100 * count_lines / n_steps)
        )
        self.assertEqual(dump.timesteps[-1], (-1, count_lines))
        self.assertEqual(dump.name, str(DUMP_PATH))

        dump = Dump(DUMP_PATH, 100 * 25)
        self.assertEqual(dump.timestep_i, 100)
        self.assertEqual(len(dump["id"]), 60)

    def test_Atom(self):
        props = (-1, 2, -3, -4, 5, -6, 28, 10, 123)
        atom = Atom(*props)
        self.check_atom(atom, *props)
        self.assertEqual(str(atom), " ".join(map(str, props)))

    # @unittest.skip
    def test_Cluster(self):
        atoms = list()
        atoms.append(
            Atom(
                -6.47066,
                27.2587,
                144.554,
                -16.1056,
                20.2075,
                16.8818,
                28.0855,
                1.0,
                384761.0,
            )
        )
        atoms.append(
            Atom(
                -7.93474,
                26.4849,
                146.689,
                9.73458,
                -13.4722,
                17.0874,
                28.0855,
                1.0,
                384432.0,
            )
        )
        atoms.append(
            Atom(
                -7.62823,
                28.0589,
                147.199,
                -2.68143,
                15.6842,
                21.9035,
                12.011,
                2.0,
                403227.0,
            )
        )
        atoms.append(
            Atom(
                -6.57531,
                29.1637,
                145.904,
                0.150277,
                0.592103,
                9.96577,
                28.0855,
                1.0,
                384755.0,
            )
        )
        cluster = Cluster(atoms, 1)

        self.assertEqual(cluster.count_si, 3)
        self.assertEqual(cluster.count_c, 1)
        self.assertEqual(cluster.mass, 28.0855 * 3 + 12.011)
        self.assertEqual(cluster.mx, -206.9193332565)
        self.assertEqual(cluster.my, 394.1767031564999)
        self.assertEqual(cluster.mz, 1497.018538435)
        self.assertAlmostEqual(cluster.angle, 16.5615, 3)
        self.assertAlmostEqual(cluster.ek, 1.314, 3)


if __name__ == "__main__":
    unittest.main()
