""" lammps_util.classes """

from pathlib import Path
from typing import List
import numpy as np
import numpy.typing as npt


class Dump:
    """Dump"""

    def __init__(self, dump_path: Path, timestemp: int | None = None) -> None:
        self.dump_path = dump_path
        self.timestep = timestemp
        self._data: npt.NDArray[np.double] | None = None
        self._keys: list[str] | None = None

    def __getitem__(self, key: str) -> npt.NDArray[np.double]:
        if key not in self.keys:
            raise ValueError(f"no such key: {key}")

        if len(self.data) == 0:
            return np.empty(0)

        return self.data[:, self.keys.index(key)]

    @property
    def name(self) -> str:
        return str(self.dump_path)

    @property
    def timesteps(self) -> list[tuple[int, int]]:
        timesteps: list[tuple[int, int]] = list()

        with open(self.dump_path, "r") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if line.strip() == "ITEM: TIMESTEP":
                    timestep = int(lines[i + 1].strip())
                    timesteps.append((timestep, i))

        if len(timesteps) == 0:
            raise ValueError(f"dump: {self.dump_path} - is empty")

        timesteps.append((-1, len(lines)))

        return timesteps

    @property
    def read_window(self) -> tuple[int, int]:
        start = self.timesteps[self.timestep_i][1]
        end = self.timesteps[self.timestep_i + 1][1]
        return (start, end)

    @property
    def timestep_i(self) -> int:
        if self.timestep is None:
            return 0

        for i, item in enumerate(self.timesteps):
            if item[0] == self.timestep:
                return i

        raise ValueError(f"no timestep found: {self.timestep}")

    @property
    def data(self) -> npt.NDArray[np.double]:
        if self._data is not None:
            return self._data

        start, end = self.read_window
        start += 9
        self._data = np.loadtxt(
            self.dump_path, ndmin=2, skiprows=start, max_rows=(end - start)
        )
        return self._data

    @property
    def keys(self) -> list[str]:
        if self._keys is not None:
            return self._keys

        start, _ = self.read_window
        with open(self.dump_path, "r") as file:
            lines = file.readlines()
            keys = lines[start + 8].strip().split()
            self._keys = keys[2:]

        return self._keys


#    def select(self, mask: npt.NDArray[np.double]) -> Dump:
#        dump = Dump(self.dump_path, self.timestep)
#        dump._keys = self._keys
#        dump._data = self._data[mask, :]
#        return Dump
#
#    def save(self, dump_path: Path):
#        pass


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

    def __str__(self) -> str:
        return f"{self.x} {self.y} {self.z} {self.vx} {self.vy} {self.vz} {self.mass} {self.type} {self.id}"


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
            2 * 5.1875 * 1e-5 * (self.mx**2 + self.my**2 + self.mz**2) / (2 * self.mass)
        )

        self.angle = np.arctan(self.mz / np.sqrt(self.mx**2 + self.my**2))
        self.angle = 90 - self.angle * 180 / np.pi
