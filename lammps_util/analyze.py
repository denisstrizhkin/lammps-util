""" lammps_util.analyze """

from pathlib import Path
import tempfile
import logging
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from .classes import Dump
from .lammps import lammps_run


def calc_zero_lvl(input_file: Path, in_path: Path) -> float:
    """calc_zero_lvl"""

    tmp_dir = Path(tempfile.gettempdir())

    dump_path = tmp_dir / "dump.temp"
    dump_str = "x y z"

    lammps_run(
        in_path,
        [
            ("input_file", str(input_file)),
            ("dump_path", str(dump_path)),
            ("dump_str", dump_str),
        ],
    )

    dump = Dump(dump_path)

    return dump["z"][:].max()


def calc_surface_values(
    data: Dump, lattice: float, coeff: float, square: float
) -> np.ndarray:
    """calc_surface_values"""

    def get_linspace(left, right):
        return np.linspace(left, right, round((right - left) / square) + 1)

    X = get_linspace(-lattice * coeff, lattice * coeff)
    Y = get_linspace(-lattice * coeff, lattice * coeff)
    Z = np.zeros((len(X) - 1, len(Y) - 1))
    Z[:] = np.nan

    for i in range(len(X) - 1):
        for j in range(len(Y) - 1):
            Z_vals = data["z"][
                np.where(
                    (data["x"] >= X[i])
                    & (data["x"] < X[i + 1])
                    & (data["y"] >= Y[j])
                    & (data["y"] < Y[j + 1])
                )
            ]
            if len(Z_vals) != 0:
                Z[i, j] = Z_vals.max()

    logging.info(f"NaN: {np.count_nonzero(np.isnan(Z))}")

    def check_value(i, j):
        if i < 0 or j < 0 or i >= len(X) - 1 or j >= len(Y) - 1:
            return np.nan
        return Z[i, j]

    for i in range(len(X) - 1):
        for j in range(len(Y) - 1):
            if Z[i, j] == 0 or Z[i, j] == np.nan:
                neighs = [
                    check_value(i - 1, j - 1),
                    check_value(i - 1, j),
                    check_value(i - 1, j + 1),
                    check_value(i + 1, j - 1),
                    check_value(i + 1, j),
                    check_value(i + 1, j + 1),
                    check_value(i, j - 1),
                    check_value(i, j + 1),
                ]
                Z[i, j] = np.nanmean(neighs)

    return Z


def calc_surface(data: Dump, run_dir: Path, lattice: float, zero_lvl: float):
    """calc_surface"""

    SQUARE = lattice / 2
    COEFF = 5
    VMIN = -20
    VMAX = 10

    def plotting(square, run_dir):
        fig, ax = plt.subplots()

        width = len(square) + 1
        x = np.linspace(0, COEFF * lattice * 2, width)
        y = np.linspace(0, COEFF * lattice * 2, width)
        x, y = np.meshgrid(x, y)

        ax.set_aspect("equal")
        plt.pcolor(x, y, square, vmin=VMIN, vmax=VMAX, cmap=cm.viridis)
        plt.colorbar()
        plt.savefig(f"{run_dir / 'surface_2d.png'}")

    def histogram(data, run_dir):
        data = data.flatten()
        desired_bin_size = 5
        num_bins = compute_histogram_bins(data, desired_bin_size)
        fig, ax = plt.subplots()
        n, bins, patches = plt.hist(data, num_bins, facecolor="green", alpha=1)
        plt.xlabel("Z coordinate (Å)")
        plt.ylabel("Count")
        plt.title("Surface atoms depth distribution")
        plt.grid(True)
        plt.savefig(f"{run_dir / 'surface_hist.png'}")

    def compute_histogram_bins(data, desired_bin_size):
        min_val = np.min(data)
        max_val = np.max(data)
        min_boundary = min_val - min_val % desired_bin_size
        max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
        n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
        num_bins = np.linspace(min_boundary, max_boundary, n_bins)
        return num_bins

    Z = calc_surface_values(data, lattice, COEFF, SQUARE) - zero_lvl

    n_X = Z.shape[0]
    X = np.linspace(0, n_X - 1, n_X, dtype=int)

    n_Y = Z.shape[1]
    Y = np.linspace(0, n_Y - 1, n_Y, dtype=int)

    def f_Z(i, j):
        return Z[i, j]

    z_all = Z.flatten()
    sigma = np.std(z_all)
    logging.info(f"D: {sigma}")

    plotting(Z, run_dir)
    histogram(Z, run_dir)

    Xs, Ys = np.meshgrid(X, Y)
    Z = f_Z(Xs, Ys)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        Xs * SQUARE, Ys * SQUARE, Z, vmin=VMIN, vmax=VMAX, cmap=cm.viridis
    )
    ax.set_zlim3d(-60, 15)
    plt.savefig(f"{run_dir / 'surface_3d.png'}")

    return sigma
