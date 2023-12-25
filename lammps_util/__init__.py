from pathlib import Path
from typing import List, Tuple
import numpy as np
import sys
import os
import subprocess
import time

from .hello import hi_there


__all__ = [ "hi_there" ]


def lammps_run(
        in_file: Path,
        vars: List[Tuple[str, str]]=[],
        omp_threads: int=4,
        mpi_cores: int=3,
        log_file: Path=Path('./log.lammps')
    ) -> None:
    mpirun_base = [
        'mpirun', '-np', str(mpi_cores),
        'lmp', '-in', str(in_file)
    ]

    if (omp_threads <= 0):
        args = mpirun_base + [
          '-sf', 'gpu',
          '-pk', 'gpu', '0',
        ]
    else:
        args = mpirun_base + [
          '-sf', 'omp',
          '-pk', 'omp', str(omp_threads),
        ]

    vars_list = [ ]      
    for var in vars:
        vars_list.append('-var')
        vars_list.append(var[0])
        vars_list.append(var[1])
                  
    print('lammps_run:', args)
    run_args = args + vars_list + [ '-log', f'{log_file}' ]
    print('lammps_run:', run_args)

    process = subprocess.Popen(run_args, encoding='utf-8')
    while process.poll() is None:
        time.sleep(1)

    if process.returncode != 0:
        print("lammps_run: FAIL")
        sys.exit()


def save_table(filename, table, header="", dtype="f", precision=5, mode='w'):
    fmt_str = ""

    if dtype == "d":
        fmt_str = "%d"
    elif dtype == "f":
        fmt_str = f"%.{precision}f"

    with open(filename, f"{mode}b") as file:
        np.savetxt(file, table, delimiter="\t", fmt=fmt_str, header=header)
