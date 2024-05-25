import unittest

from lammps_util import Atom, Cluster, Dump

from test_lammps_util import TestLAMMPSUtil

class TestClasses(TestLAMMPSUtil):

    def test_Atom(self):
        props = (-1, 2, -3, -4, 5, -6, 28, 10, 123)
        atom = Atom(*props)
        self.check_atom(atom, *props)
        self.assertEqual(str(atom), " ".join(map(str, props)))

    def test_Cluster(self):
        atoms = list()
        atoms.append(Atom(-6.47066, 27.2587, 144.554, -16.1056, 20.2075, 16.8818, 28.0855, 1.0, 384761.0))
        atoms.append(Atom(-7.93474, 26.4849, 146.689, 9.73458, -13.4722, 17.0874, 28.0855, 1.0, 384432.0))
        atoms.append(Atom(-7.62823, 28.0589, 147.199, -2.68143, 15.6842, 21.9035, 12.011, 2.0, 403227.0))
        atoms.append(Atom(-6.57531, 29.1637, 145.904, 0.150277, 0.592103, 9.96577, 28.0855, 1.0, 384755.0))
        cluster = Cluster(atoms, 1)

        self.assertEqual(cluster.count_si, 3)
        self.assertEqual(cluster.count_c, 1)
        self.assertEqual(cluster.mass, 28.0855 * 3 + 12.011)
        self.assertEqual(


if __name__ == "__main__":
    unittest.main()
