import unittest
from pathlib import Path

from lammps_util import Dump
import lammps_util

DUMP_PATH = Path("./dump.final")
OUTPUT_PATH = Path("/tmp/tmp.dump")


class TestSputterMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dump = Dump(DUMP_PATH)

    def setUp(cls):
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()

    @unittest.skip
    def test_create_clusters_dump(self):
        self.assertFalse(OUTPUT_PATH.is_file())
        lammps_util.create_clusters_dump(
            DUMP_PATH, self._dump.timesteps[0][0], OUTPUT_PATH
        )
        self.assertTrue(OUTPUT_PATH.is_file())

        clusters = Dump(OUTPUT_PATH)
        self.assertEqual(len(clusters["id"]), len(self._dump["id"]))

        try:
            clusters["c_clusters"]
        except ValueError as e:
            self.fail(e)

    def check_atom(self, atom, id_, x, y, z, vx, vy, vz, type_, mass):
        self.assertEqual(atom.id, id_)
        self.assertEqual(atom.x, x)
        self.assertEqual(atom.y, y)
        self.assertEqual(atom.z, z)
        self.assertEqual(atom.vx, vx)
        self.assertEqual(atom.vy, vy)
        self.assertEqual(atom.vz, vz)
        self.assertEqual(atom.type, type_)
        self.assertEqual(atom.mass, mass)

    def test_get_cluster_atoms_dict(self):
        lammps_util.create_clusters_dump(
            DUMP_PATH, self._dump.timesteps[0][0], OUTPUT_PATH
        )
        clusters = Dump(OUTPUT_PATH)
        cluster_atoms_dict, _ = lammps_util.get_cluster_atoms_dict(clusters)
        self.assertEqual(len(cluster_atoms_dict.keys()), 31)
        self.assertEqual(sum(map(len, cluster_atoms_dict.values())), 46)

        self.assertIn(396856, cluster_atoms_dict.keys())
        self.assertEqual(len(cluster_atoms_dict[396856]), 1)
        self.check_atom(
            cluster_atoms_dict[396856][0],
            396856,
            -89.628,
            -28.8952,
            197.315,
            -61.4057,
            -51.0242,
            23.3924,
            1,
            28.0855,
        )

        self.assertIn(403231, cluster_atoms_dict.keys())
        self.assertEqual(len(cluster_atoms_dict[403231]), 1)
        self.check_atom(
            cluster_atoms_dict[403231][0],
            403231,
            -105.942,
            -42.2043,
            208.304,
            23.609,
            -9.31622,
            28.1357,
            2,
            12.011,
        )

        ids = [384761, 384432, 384755, 403227]
        self.assertTrue(
            ids[0] in cluster_atoms_dict.keys()
            or ids[1] in cluster_atoms_dict.keys()
            or ids[2] in cluster_atoms_dict.keys()
            or ids[3] in cluster_atoms_dict.keys()
        )
        count_found = 0
        key = -1
        for id_ in ids:
            if id_ in cluster_atoms_dict.keys():
                count_found += 1
                key = id_
        self.assertEqual(count_found, 1)
        self.assertEqual(len(cluster_atoms_dict[key]), 4)


if __name__ == "__main__":
    unittest.main()
