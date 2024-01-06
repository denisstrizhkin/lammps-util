""" lammps_util.run """

import logging
from pathlib import Path


def setup_root_logger(log_file_path: Path) -> None:
    """setup_root_logger"""

    log_formatter = logging.Formatter(
        "%(asctime)s %(funcName)s [%(levelname)s]: %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
