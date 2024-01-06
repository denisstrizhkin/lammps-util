""" lammps_util.filesystem """

import logging
import subprocess
from pathlib import Path
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
