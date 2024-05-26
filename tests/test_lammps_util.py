import unittest


class TestLAMMPSUtil(unittest.TestCase):
    def check_atom(self, atom, x, y, z, vx, vy, vz, mass, type_, id_):
        self.assertEqual(atom.id, id_)
        self.assertEqual(atom.x, x)
        self.assertEqual(atom.y, y)
        self.assertEqual(atom.z, z)
        self.assertEqual(atom.vx, vx)
        self.assertEqual(atom.vy, vy)
        self.assertEqual(atom.vz, vz)
        self.assertEqual(atom.type, type_)
        self.assertEqual(atom.mass, mass)
