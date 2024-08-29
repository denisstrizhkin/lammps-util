""" lammps_util.filesystem """

import logging
import subprocess
from pathlib import Path
from typing import List
import numpy as np


def create_archive(dir_path: Path) -> None:
    """create_archive"""

    # fmt: off
    result = subprocess.run(
        [
            "tar",
            "-C", str(dir_path.parent),
            "-czvf", str(dir_path.with_suffix(".tar.gz")),
            str(dir_path.name),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # fmt: on

    logging.info(result.stdout)


def file_without_suffix(file_path: Path) -> str:
    """file_without_suffix"""

    return str(file_path).removesuffix(file_get_suffix(file_path))


def file_get_suffix(file_path: Path) -> str:
    """file_get_suffix"""

    return "".join(file_path.suffixes)


def save_table(filename, table, header="", dtype="f", precision=5, mode="w"):
    """save_table"""

    fmt_str = ""

    if dtype == "d":
        fmt_str = "%d"
    elif dtype == "f":
        fmt_str = f"%.{precision}f"

    with open(filename, f"{mode}b") as file:
        np.savetxt(file, table, delimiter="\t", fmt=fmt_str, header=header)


def dump_delete_atoms(
    in_path: Path, out_path: Path, ids_to_delete: List[int]
) -> None:
    """dump_delete_atoms"""

    with (
        open(in_path, "r", encoding="utf-8") as f_in,
        open(out_path, "w", encoding="utf-8") as f_out,
    ):
        cnt = 0
        for line in f_in:
            cnt += 1

            if cnt == 4:
                num_atoms = int(line) - len(ids_to_delete)
                f_out.write(f"{num_atoms}\n")
            elif (cnt < 10) or (
                int(line.split(" ", 1)[0]) not in ids_to_delete
            ):
                f_out.write(line)


def input_delete_atoms(
    in_path: Path, out_path: Path, ids_to_delete: List[int]
) -> None:
    """input_delete_atoms"""

    with (
        open(in_path, "r", encoding="utf-8") as f_in,
        open(out_path, "w", encoding="utf-8") as f_out,
    ):
        cnt = 0
        for line in f_in:
            cnt += 1

            tokens = line.split(" ", 1)
            if cnt == 3:
                num_atoms = int(tokens[0]) - len(ids_to_delete)
                f_out.write(f"{num_atoms} atoms\n")
            elif (
                (cnt < 17)
                or len(tokens) == 1
                or int(tokens[0]) not in ids_to_delete
            ):
                f_out.write(line)
