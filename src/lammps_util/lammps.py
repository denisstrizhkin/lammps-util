""" lammps_util.lammps """

import uuid
import tempfile
from pathlib import Path
from lammps_mpi4py import LammpsMPI

from .classes import Dump


def lammps_script_init() -> str:
    return """
    clear
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
    lmp: LammpsMPI, dump_path: Path, timestep: int, out_path: Path
) -> None:
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


def create_dump_from_input(lmp: LammpsMPI, input: Path, output: Path) -> None:
    script = f"""
    {lammps_script_init()}

    read_data {input}
    write_dump all custom {output} id x y z vx vy vz type 
    """
    lmp.commands_string(script)


def create_crater_dump(
    lmp: LammpsMPI,
    dump_crater_path: Path,
    dump_final: Dump,
    input_path: Path,
    offset_x: float = 0,
    offset_y: float = 0,
) -> None:
    dump_input_path = Path(f"{tempfile.gettempdir()}/{uuid.uuid4()}")
    create_dump_from_input(lmp, input_path, dump_input_path)
    dump_input = Dump(dump_input_path)
    script = f"""
    {lammps_script_init()}
    
    read_data {input_path}
    displace_atoms all move {offset_x} {offset_y} 0 units box

    {lammps_script_potential()}

    group Si type 1

    compute voro_occupation Si voronoi/atom occupation only_group
    compute voro_vol Si voronoi/atom only_group
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

    compute clusters vac3 cluster/atom 3
    dump clusters vac3 custom 1 {dump_crater_path} id x y z type c_clusters c_voro_vol[1]
    run 0
    """
    lmp.commands_string(script)
    dump_input_path.unlink()
