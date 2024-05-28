""" lammps_util.analyze """

from pathlib import Path
import tempfile
import logging
import operator

import matplotlib.pyplot as plt
import numpy as np

from .classes import Dump, Atom
from .filesystem import save_table, file_get_suffix, file_without_suffix


def calc_surface_values(
    data: Dump, lattice: float, coeff: float, square: float
) -> np.ndarray:
    """calc_surface_values"""

    def get_linspace(left, right):
        return np.linspace(left, right, round((right - left) / square) + 1)

    coords_x = get_linspace(-lattice * coeff, lattice * coeff)
    coords_y = get_linspace(-lattice * coeff, lattice * coeff)
    coords_z = np.zeros((len(coords_x) - 1, len(coords_y) - 1))
    coords_z[:] = np.nan

    for i in range(len(coords_x) - 1):
        for j in range(len(coords_y) - 1):
            z_vals = data["z"][
                np.where(
                    (data["x"] >= coords_x[i])
                    & (data["x"] < coords_x[i + 1])
                    & (data["y"] >= coords_y[j])
                    & (data["y"] < coords_y[j + 1])
                )
            ]
            if len(z_vals) != 0:
                coords_z[i, j] = z_vals.max()

    logging.info("NaN: %s", str(np.count_nonzero(np.isnan(coords_z))))

    def check_value(i, j):
        if i < 0 or j < 0 or i >= len(coords_x) - 1 or j >= len(coords_y) - 1:
            return np.nan
        return coords_z[i, j]

    for i in range(len(coords_x) - 1):
        for j in range(len(coords_y) - 1):
            if coords_z[i, j] == 0 or coords_z[i, j] == np.nan:
                neighs = np.array(
                    [
                        check_value(i - 1, j - 1),
                        check_value(i - 1, j),
                        check_value(i - 1, j + 1),
                        check_value(i + 1, j - 1),
                        check_value(i + 1, j),
                        check_value(i + 1, j + 1),
                        check_value(i, j - 1),
                        check_value(i, j + 1),
                    ]
                )
                coords_z[i, j] = np.nanmean(neighs)

    return coords_z


def calc_surface(
    data: Dump, run_dir: Path, lattice: float, zero_lvl: float, c60_width: int
):
    """calc_surface"""

    square_width = lattice / 2
    color_min = -20
    color_max = 10

    def plotting(square, run_dir):
        _, ax = plt.subplots()

        width = len(square) + 1
        x = np.linspace(0, c60_width * lattice * 2, width)
        y = np.linspace(0, c60_width * lattice * 2, width)
        x, y = np.meshgrid(x, y)

        ax.set_aspect("equal")
        plt.pcolor(x, y, square, vmin=color_min, vmax=color_max)
        plt.colorbar()
        plt.viridis()
        plt.savefig(f"{run_dir / 'surface_2d.png'}")
        plt.close()

    def histogram(data, run_dir):
        data = data.flatten()
        desired_bin_size = 5
        num_bins = compute_histogram_bins(data, desired_bin_size)
        plt.hist(data, num_bins, facecolor="green", alpha=1)
        plt.xlabel("Z coordinate (Å)")
        plt.ylabel("Count")
        plt.title("Surface atoms depth distribution")
        plt.grid(True)
        plt.savefig(f"{run_dir / 'surface_hist.png'}")
        plt.close()

    def compute_histogram_bins(data, desired_bin_size):
        min_val = np.min(data)
        max_val = np.max(data)
        min_boundary = min_val - min_val % desired_bin_size
        max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
        n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
        num_bins = np.linspace(min_boundary, max_boundary, n_bins)
        return num_bins

    coord_z = (
        calc_surface_values(data, lattice, c60_width, square_width) - zero_lvl
    )

    len_x = coord_z.shape[0]
    coord_x = np.linspace(0, len_x - 1, len_x, dtype=int)

    len_y = coord_z.shape[1]
    coord_y = np.linspace(0, len_y - 1, len_y, dtype=int)

    z_all = coord_z.flatten()
    sigma = np.std(z_all)
    logging.info("D: %s", str(sigma))

    plotting(coord_z, run_dir)
    histogram(coord_z, run_dir)

    mesh_x, mesh_y = np.meshgrid(coord_x, coord_y)
    coord_z = coord_z[mesh_x, mesh_y]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        mesh_x * square_width,
        mesh_y * square_width,
        coord_z,
        vmin=color_min,
        vmax=color_max,
        cmap=plt.cm.viridis,
    )
    ax.set_zlim3d(-60, 15)
    plt.savefig(f"{run_dir / 'surface_3d.png'}")
    plt.close("all")

    return sigma


def get_parsed_file_path(file_path: Path, suffix: str = ""):
    """get_parsed_file_path"""
    return (
        file_without_suffix(file_path)
        + "_parsed"
        + suffix
        + file_get_suffix(file_path)
    )


def carbon_dist_parse(file_path: Path):
    """carbon_dist_parse"""
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines.pop(0)

    lines_dic: dict[int, list[tuple[float, ...]]] = {}
    sim_num: int
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) == 0:
            continue

        if tokens[0] == "#":
            sim_num = int(tokens[1])
            lines_dic[sim_num] = []
        else:
            lines_dic[sim_num].append(tuple(map(float, tokens)))

    z_min: float = float("inf")
    z_max: float = float("-inf")
    for distribution in lines_dic.values():
        z_min = min(distribution[0][0], z_min)
        z_max = max(distribution[len(distribution) - 1][0], z_max)

    bins = np.linspace(z_min, z_max, int(z_max - z_min) + 1)
    table = np.zeros((len(lines_dic) + 1, len(bins) + 1))

    for sim_num, distribution in lines_dic.items():
        table[sim_num][0] = sim_num
        for pair in distribution:
            index = int(pair[0] - z_min)
            table[sim_num][index + 1] = pair[1]

    for i, pin in enumerate(bins):
        table[0][i + 1] = pin

    header_str = "simN " + " ".join(list(map(str, bins)))
    output_path = get_parsed_file_path(file_path)
    save_table(output_path, table.T, header_str)


def clusters_parse_angle_dist(file_path: Path, n_runs: int):
    """carbon_dist_parse"""

    clusters = np.loadtxt(file_path, ndmin=2, skiprows=1)

    clusters_sim_num_n = clusters[:, :2]
    clusters_sim_num_n[:, 1] = clusters[:, 1] + clusters[:, 2]

    clusters_enrg_ang = clusters[:, -2:]
    clusters_enrg_ang[:, 0] /= clusters_sim_num_n[:, 1]
    print(clusters_enrg_ang)

    num_bins = (85 - 5) // 10 + 1
    num_sims = n_runs + 1

    number_table = np.zeros((num_bins, num_sims))
    energy_table = np.zeros((num_bins, num_sims))

    number_table[:, 0] = np.linspace(5, 85, 9)
    energy_table[:, 0] = np.linspace(5, 85, 9)

    for i in range(0, len(clusters)):
        if np.isnan(clusters_enrg_ang[i, 1]):
            continue

        angle_index = int(np.floor(clusters_enrg_ang[i, 1])) // 10
        sim_index = int(clusters_sim_num_n[i, 0])

        if angle_index >= num_bins:
            continue

        number_table[angle_index, sim_index] += clusters_sim_num_n[i, 1]
        energy_table[angle_index, sim_index] += clusters_enrg_ang[i, 1]

    print(number_table[:, :10])

    header_str_number = "angle N1 N2 N3 ... N50"
    output_path_number = get_parsed_file_path(file_path, "_number_dist")
    save_table(output_path_number, number_table, header_str_number)

    header_str_energy = "angle E1 E2 E3 ... E50"
    output_path_energy = get_parsed_file_path(file_path, "_energy_dist")
    save_table(output_path_energy, energy_table, header_str_energy)


def clusters_parse(file_path: Path, n_runs: int):
    """clusters_parse"""
    clusters = np.loadtxt(file_path, ndmin=2, skiprows=1)
    clusters = clusters[:, :3]

    clusters_dic: dict[str, dict[int, int]] = {}
    for cluster in clusters:
        cluster_str = "Si" + str(int(cluster[1])) + "C" + str(int(cluster[2]))
        if cluster_str not in clusters_dic:
            clusters_dic[cluster_str] = {}

        sim_num = int(cluster[0])
        if not cluster[0] in clusters_dic[cluster_str]:
            clusters_dic[cluster_str][sim_num] = 0

        clusters_dic[cluster_str][sim_num] += 1

    total_sims = n_runs
    total_clusters = len(clusters_dic.keys())

    table = np.zeros((total_sims, total_clusters + 1))
    cluster_index = 0
    for cluster_str, sim_count in clusters_dic.items():
        for sim_num, cluster_count in sim_count.items():
            table[sim_num - 1, 0] = sim_num
            table[sim_num - 1][cluster_index + 1] = cluster_count
        cluster_index += 1

    header_str = "simN\t" + "\t".join(clusters_dic.keys())
    output_path = get_parsed_file_path(file_path)
    save_table(output_path, table, header_str, dtype="d")


def clusters_parse_sum(file_path: Path, n_runs: int):
    """clusters_parse_sum"""
    clusters = np.loadtxt(file_path, ndmin=2, skiprows=1)

    clusters = clusters[:, :3]

    clusters_dic: dict[int, dict[str, int]] = {}
    for cluster in clusters:
        sim_num = int(cluster[0])
        if sim_num not in clusters_dic:
            clusters_dic[sim_num] = {}
            clusters_dic[sim_num]["Si"] = 0
            clusters_dic[sim_num]["C"] = 0
        clusters_dic[sim_num]["Si"] += cluster[1]
        clusters_dic[sim_num]["C"] += cluster[2]
    table = np.zeros((n_runs, 4))

    for sim_num, atom_count in clusters_dic.items():
        table[sim_num - 1][0] = sim_num
        table[sim_num - 1][1] = atom_count["Si"]
        table[sim_num - 1][2] = atom_count["C"]
        table[sim_num - 1][3] = atom_count["Si"] + atom_count["C"]

    header_str = "simN Si C"
    output_path = get_parsed_file_path(file_path, "_sum")
    save_table(output_path, table, header_str, dtype="d")


def get_sputtered_ids(
    dump: Dump,
) -> list[int]:
    cluster_id = dump["c_clusters"]
    unique, counts = np.unique(cluster_id, return_counts=True)
    cluster_count = dict(zip(unique, counts))
    cluster_count = dict(filter(lambda x: x[1] > 1000, cluster_count.items()))
    clusters_to_delete = list(cluster_count.keys())
    return dump["id"][np.logical_not(np.isin(cluster_id, clusters_to_delete))]


def get_cluster_atoms_dict(
    cluster_dump: Dump,
) -> tuple[dict[int, list[Atom]], list[Atom]]:
    """get_cluster_atoms_dict"""
    cluster_id = cluster_dump["c_clusters"]

    unique, counts = np.unique(cluster_id, return_counts=True)
    cluster_count = dict(zip(unique, counts))

    cluster_to_delete = dict(
        filter(lambda x: x[1] > 1000, cluster_count.items())
    )
    rim_id = max(cluster_to_delete.items(), key=operator.itemgetter(1))[0]

    cluster_dict: dict[int, list[Atom]] = {}
    for cid in np.unique(cluster_id):
        cluster_dict[cid] = []

    x = cluster_dump["x"]
    y = cluster_dump["y"]
    z = cluster_dump["z"]
    vx = cluster_dump["vx"]
    vy = cluster_dump["vy"]
    vz = cluster_dump["vz"]
    mass = cluster_dump["c_mass"]
    type = cluster_dump["type"]
    id = cluster_dump["id"]

    for i, cid in enumerate(cluster_id):
        atom = Atom(
            x=x[i],
            y=y[i],
            z=z[i],
            vx=vx[i],
            vy=vy[i],
            vz=vz[i],
            mass=mass[i],
            type=type[i],
            id=id[i],
        )
        cluster_dict[cid].append(atom)

    rim_atoms = cluster_dict[rim_id]
    for cid in cluster_to_delete.keys():
        logging.info(
            f"deleteing cluster {cid} with {len(cluster_dict[cid])} atoms"
        )
        cluster_dict.pop(cid)

    return cluster_dict, rim_atoms


def calc_dump_zero_lvl(dump: Dump) -> float:
    return dump["z"][:].max()


def calc_input_zero_lvl(input_file: Path) -> float:
    tmp_dir = Path(tempfile.gettempdir())
    dump_path = tmp_dir / "dump.temp"
    create_dump_from_input(input_file, dump_path)
    dump = Dump(dump_path)
    return calc_dump_zero_lvl(dump)


def get_crater_info(
    dump_crater: Dump, sim_num: int, zero_lvl: float
) -> np.ndarray:
    id = dump_crater["id"]
    z = dump_crater["z"]
    clusters = dump_crater["c_clusters"]

    crater_id = np.bincount(clusters.astype(int)).argmax()
    atoms = []
    for i in range(0, len(id)):
        if clusters[i] == crater_id:
            atoms.append(Atom(z=z[i], id=id[i]))

    cell_vol = float(np.median(dump_crater["c_voro_vol[1]"]))
    crater_vol = cell_vol * len(atoms)

    surface_count = 0
    z = []
    for atom in atoms:
        z.append(atom.z)
        if atom.z > -2.4 * 0.707 + zero_lvl:
            surface_count += 1
    z = np.array(z)

    cell_surface = 7.3712
    surface_area = cell_surface * surface_count

    return np.array(
        [
            [
                sim_num,
                len(atoms),
                crater_vol,
                surface_area,
                z.mean() - zero_lvl,
                z.min() - zero_lvl,
            ]
        ]
    )
