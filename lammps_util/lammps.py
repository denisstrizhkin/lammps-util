""" lammps_util.lammps """

import logging
import subprocess
import time
import tempfile
from pathlib import Path

from lammps import lammps

from .classes import Dump


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


def lammps_script_init() -> str:
    return """
    units metal
    dimension 3
    boundary p p m
    atom_style atomic
    atom_modify map yes
    """


def lammps_script_potential() -> str:
    return """
    pair_style tersoff/zbl
    pair_coeff * * SiC.tersoff.zbl Si C
    neigh_modify every 1 delay 0 check no
    neigh_modify binsize 0.0
    neigh_modify one 4000
    """


def create_clusters_dump(
    dump_path: Path, timestep: int, out_path: Path
) -> None:
    lmp = lammps()
    init = f"""
    {lammps_script_init()}

    region r block -1 1 -1 1 -1 1
    create_box 2 r

    read_dump {dump_path} {timestep} x y z vx vy vz box yes add keep

    mass 1 28.08553
    mass 2 12.011

    {lammps_script_potential()}

    compute atom_ke all ke/atom
    compute clusters all cluster/atom 3
    compute mass all property/atom mass
    dump clusters all custom 1 {out_path} id x y z vx vy vz type c_mass c_clusters c_atom_ke
    run 0
    """
    lmp.commands_string(init)


def create_dump_from_input(input: Path, output: Path):
    lmp = lammps()
    script = f"""
    read_data {input}
    write_dump all custom {output} id x y z vx vy vz type 
    """
    lmp.commands_string(script)


def create_crater_dump(
    dump_crater_path, dump_final, input_path, offset_x=0, offset_y=0
):
    lmp = lammps()
    with tempfile.NamedTemporaryFile(mode="r") as f:
        dump_input_path = f.name
        create_dump_from_input(input_path, dump_input_path)
        dump_input = Dump(dump_input_path)
        script = f"""
        {lammps_script_init()}
        
        read_data {input_path}
        displace_atoms all move {offset_x} {offset_y} 0 units box

        {lammps_script_potential()}

        group Si type 1

        compute voro_occupation Si voronoi/atom occupation only_group
        variable is_vacancy atom "c_voro_occupation[1]==0"

        run 0
        group vac1 variable is_vacancy

        read_dump {dump_final.name} {dump_final.timesteps[0][0]} x y z add keep replace yes

        run 0
        group vac2 variable is_vacancy
        group C type 2
        group vac3 subtract vac2 C

        read_dump {dump_input.name} {dump_input.timesteps[0][0]} x y z add keep replace yes
        displace_atoms all move {offset_x} {offset_y} 0 units box

        compute voro_vol vac3 voronoi/atom only_group
        compute clusters vac3 cluster/atom 3
        dump clusters vac3 custom 1 {dump_crater_path} id x y z type c_clusters c_voro_vol[1]
        run 0
        """
    lmp.commands_string(script)
