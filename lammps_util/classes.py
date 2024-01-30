""" lammps_util.classes """

from pathlib import Path
from typing import List
import numpy as np


class Dump:
    """Dump"""

    def __init__(self, dump_path: Path) -> None:
        self.data: np.ndarray = np.loadtxt(dump_path, ndmin=2, skiprows=9)

        with open(dump_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            self.keys = lines[8].strip().split()
            self.keys = self.keys[2:]

        self.name = str(dump_path)

        if len(set(self.keys)) != len(self.keys):
            raise ValueError("dump keys must be unique")

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.keys:
            raise ValueError(f"no such key: {key}")

        if len(self.data) == 0:
            return np.empty(0)

        return self.data[:, self.keys.index(key)]


class Atom:
    """Atom"""

    def __init__(self, x=0, y=0, z=0, vx=0, vy=0, vz=0, mass=0, type=0, id=0):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass
        self.type = type
        self.id = id


class Cluster:
    """Cluster"""

    def __init__(self, clusters: List[Atom], si_atom_type):
        self.mass = 0
        self.count_si = 0
        self.count_c = 0
        self.mx = 0
        self.my = 0
        self.mz = 0

        for cluster in clusters:
            self.mx += cluster.vx * cluster.mass
            self.my += cluster.vy * cluster.mass
            self.mz += cluster.vz * cluster.mass
            self.mass += cluster.mass

            if cluster.type == si_atom_type:
                self.count_si += 1
            else:
                self.count_c += 1

        self.ek = (
            2
            * 5.1875
            * 1e-5
            * (self.mx**2 + self.my**2 + self.mz**2)
            / (2 * self.mass)
        )

        self.angle = np.arctan(self.mz / np.sqrt(self.mx**2 + self.my**2))
        self.angle = 90 - self.angle * 180 / np.pi
