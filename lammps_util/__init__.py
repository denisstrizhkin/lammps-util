from pathlib import Path
from typing import List, Tuple
import sys
import subprocess
import time
import numpy as np


def lammps_run(
    in_file: Path,
    in_vars: List[Tuple[str, str]] = None,
    omp_threads: int = 4,
    mpi_cores: int = 3,
    log_file: Path = Path("./log.lammps"),
) -> int:
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
    fmt_str = ""

    if dtype == "d":
        fmt_str = "%d"
    elif dtype == "f":
        fmt_str = f"%.{precision}f"

    with open(filename, f"{mode}b", encoding="utf8") as file:
        np.savetxt(file, table, delimiter="\t", fmt=fmt_str, header=header)
