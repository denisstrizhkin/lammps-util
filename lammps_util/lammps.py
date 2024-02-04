""" lammps_util.lammps """

import logging
import subprocess
import time
from pathlib import Path


def lammps_run(
    in_file: Path,
    in_vars: dict[str, str] | None = None,
    omp_threads: int = 4,
    mpi_cores: int = 3,
    log_file: Path = Path("./log.lammps"),
) -> int:
    """lammps_run"""

    if in_vars is None:
        in_vars = {}

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

    for key, value in in_vars.items():
        args += ["-var", key, value]

    args += ["-log", str(log_file)]
    logging.info(" ".join(args))

    with subprocess.Popen(args, encoding="utf-8") as process:
        while process.poll() is None:
            time.sleep(0.1)

        if process.returncode != 0:
            logging.error("FAIL")
            return 1

    return 0
