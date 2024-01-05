""" lammps_util """

from pathlib import Path
from typing import List, Tuple
import sys
import subprocess
import time
import numpy as np


class Dump:
    """Dump"""

    def __init__(self, dump_path: Path):
        self.data = np.loadtxt(dump_path, ndmin=2, skiprows=9)

        with open(dump_path, "r", encoding="utf-8") as file:
            self.keys = file.readlines()[8].split()
            self.keys = self.keys[2:]
        print(self.keys)

        self.name = str(dump_path)

        if len(set(self.keys)) != len(self.keys):
            raise ValueError("dump keys must be unique")

    def __getitem__(self, key: str):
        if key not in self.keys:
            raise ValueError(f"no such key: {key}")

        if len(self.data) == 0:
            return []

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

            if cluster.atype == si_atom_type:
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


def lammps_run(
    in_file: Path,
    in_vars: List[Tuple[str, str]] = None,
    omp_threads: int = 4,
    mpi_cores: int = 3,
    log_file: Path = Path("./log.lammps"),
) -> int:
    """lammps_run"""

    # fmt: off
    args = [
        "mpirun", "-np", str(mpi_cores),
        "lmp", "-in", str(in_file)
    ]

    if omp_threads <= 0:
        args += [
            "-sf", "gpu",
            "-pk", "gpu", "0",
        ]
    else:
        args += [
            "-sf", "omp",
            "-pk", "omp", str(omp_threads),
        ]
    # fmt: on

    for var in in_vars:
        args += ["-var", var[0], var[1]]

    args += ["-log", str(log_file)]
    print("lammps_run:", args)

    with subprocess.Popen(args, encoding="utf-8") as process:
        while process.poll() is None:
            time.sleep(0.1)

        if process.returncode != 0:
            print("lammps_run: FAIL")
            return 1

    return 0


def save_table(filename, table, header="", dtype="f", precision=5, mode="w"):
    """save_table"""

    fmt_str = ""

    if dtype == "d":
        fmt_str = "%d"
    elif dtype == "f":
        fmt_str = f"%.{precision}f"

    with open(filename, f"{mode}b", encoding="utf8") as file:
        np.savetxt(file, table, delimiter="\t", fmt=fmt_str, header=header)
