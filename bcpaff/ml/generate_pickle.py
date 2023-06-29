"""
© 2023, ETH Zurich
"""

import argparse
import glob
import os
import pickle
from typing import Dict, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from rdkit import Chem
from scipy.spatial.distance import cdist
from tqdm import tqdm

from bcpaff.qtaim.qtaim_reader import QtaimProps, QtaimPropsCritic2
from bcpaff.utils import ATOM_NEIGHBOR_IDS, DATA_PATH, DEFAULT_PROPS, OTHER

MAX_INTERACTION_DISTANCE = 3  # Angstrom

COL_NAMES = {
    "pdbbind": {"id_col": "pdb_id", "aff_col": "aff"},
    "pde10a": {"id_col": "docking_folder", "aff_col": "pic50"},
}
NULL_COORDS = torch.FloatTensor([0.0] * 3)
ESP_NAMES = ["esp", "esp_nuc", "esp_ele"]


def get_structure_id(folder: str, dataset: str) -> int:
    return os.path.basename(folder)
    # if dataset == "pdbbind" or data:
    # elif dataset == "pde10a":
    #     return int(os.path.basename(folder).split("_")[-1])


def get_bcp_graph_from_cp_prop_file(df: pd.DataFrame, cpprop_file: str, dataset: str, **kwargs) -> Union[Dict, None]:
    """Same as get_graph_from_cp_prop_file_intermolecular, but for BCP-based graphs. Uses only intermolecular BCPs

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with critical-point information
    cpprop_file : str
        path to CPprop.txt file from Multiwfn output
    dataset : str
        type of dataset, either "pdbbind" or "pde10a"
    only_key_interactions: bool
        only for PDE10A dataset, use only interactions with Y693 & Q726
    only_short_interactions: bool
        use only interactions (BCPs) where the corresponding atoms are <= MAX_INTERACTION_DISTANCE apart
    no_c_c_interactions: bool
        remove BCPs between two carbon atoms

    Raises
    ------
    ValueError
        in case of strange assigments of atom neighbors
    """
    folder = os.path.dirname(cpprop_file)
    structure_id = get_structure_id(folder, dataset)
    docking_folder = os.path.basename(folder)
    try:
        ligand_sdf = os.path.join(folder, f"{os.path.basename(folder)}_ligand_with_hydrogens.sdf")
        pl_complex_xyz = os.path.join(folder, "pl_complex.xyz")
        cp_file = os.path.join(folder, "CPs.txt")
        paths_file = os.path.join(folder, "paths.txt")
        qtaim_props = QtaimProps(
            cp_file=cp_file,
            cpprop_file=cpprop_file,
            paths_file=paths_file,
            ligand_sdf=ligand_sdf,
            pl_complex_xyz=pl_complex_xyz,
            identifier=structure_id,
        )
        include_esp = "esp" in qtaim_props.critical_points[0].props
        prop_list = DEFAULT_PROPS + ESP_NAMES if include_esp else DEFAULT_PROPS

        target = df[df[COL_NAMES[dataset]["id_col"]] == structure_id][COL_NAMES[dataset]["aff_col"]].values[0]

        G = nx.Graph(target=target, include_esp=include_esp)

        bcps = [cp for cp in qtaim_props.critical_points if cp.name == "bond_critical_point" and cp.intermolecular]

        if kwargs["only_key_interactions"]:  # remove bcps that are not key interactions
            if dataset != "pde10a":
                raise ValueError(f"Choice only_key_interactions is not available for {dataset}")

            pdb_file = os.path.join(
                DATA_PATH, "pde10a", "pde-10_pdb_bind_format_blinded", docking_folder, f"{docking_folder}_protein.pdb"
            )
            protein = Chem.rdmolfiles.MolFromPDBFile(pdb_file, sanitize=False)
            all_protein_positions = protein.GetConformer().GetPositions()
            protein_positions = []
            # find contact atom in the key protein residues, e.g. 693 --> take position
            for a in protein.GetAtoms():
                if a.GetPDBResidueInfo().GetResidueNumber() == 693 or a.GetPDBResidueInfo().GetResidueNumber() == 726:
                    protein_positions.append(all_protein_positions[a.GetIdx()])

            # using this position, find the corresponding atom in qtaim_props
            distance_matrix = cdist(np.vstack(protein_positions), qtaim_props.pl_complex_coords)
            pl_complex_idxs = distance_matrix.argmin(axis=1)[
                distance_matrix.min() < 0.01
            ]  # not all atoms of those residues might have been included (cutting between backbone and sidechain)
            pl_complex_idxs = set(pl_complex_idxs.flatten().tolist())

            # get the corresponding bcp and verify that it is connected to the correct ligand atom
            bcps = [bcp for bcp in bcps if len(pl_complex_idxs.intersection(set(bcp.atom_neighbors)))]

        # loop over the intermolecular BCPs and pull out their neighboring NCPs
        for bcp in bcps:
            path_length = np.linalg.norm(bcp.path_positions[0] - bcp.path_positions[-1])
            if kwargs["only_short_interactions"] and path_length > MAX_INTERACTION_DISTANCE:
                continue  # skip

            if len(bcp.atom_neighbors) > 2:
                raise ValueError(f"More than two neighbors: {structure_id}")
            # neighbors = [ncp for ncp in ncps if ncp.corresponding_atom_id in bcp.atom_neighbors]
            atom_neighbors_symbol = bcp.atom_neighbors_symbol
            if len(atom_neighbors_symbol) == 2:  # BCP with two neighbors
                if kwargs["no_c_c_interactions"] and atom_neighbors_symbol == ["C", "C"]:
                    continue  # skip this interaction
            elif len(atom_neighbors_symbol) == 1:  # BCP where only one neighbor was found
                atom_neighbors_symbol = atom_neighbors_symbol + ["*"]  # append dummy
            elif len(atom_neighbors_symbol) == 0:  # NCP
                atom_neighbors_symbol = ["*", "*"]  # append 2x dummy
            atom_neighbors_type_id = [ATOM_NEIGHBOR_IDS.get(n, OTHER) for n in atom_neighbors_symbol]

            props = [bcp.props[prop] for prop in prop_list]
            coords = bcp.position
            G.add_node(
                bcp.idx,
                node_props=torch.FloatTensor(props),
                atom_type_id=atom_neighbors_type_id,
                node_coords=torch.FloatTensor(coords),
            )
        return {structure_id: G}
    except KeyboardInterrupt:
        raise ValueError
    except:
        print(f"\n{docking_folder}\n", flush=True)
        return None


def get_graph_from_cp_prop_file_intramolecular(
    df: pd.DataFrame, cpprop_file: str, dataset: str, **kwargs
) -> Union[Dict, None]:
    "kwargs unused, just for compatibility with get_bcp_graph_from_cp_prop_file"
    folder = os.path.dirname(cpprop_file)
    structure_id = get_structure_id(folder, dataset)
    try:
        ligand_sdf = os.path.join(folder, f"{os.path.basename(folder)}_ligand_with_hydrogens.sdf")
        pl_complex_xyz = os.path.join(folder, "pl_complex.xyz")
        cp_file = os.path.join(folder, "CPs.txt")
        paths_file = os.path.join(folder, "paths.txt")
        qtaim_props = QtaimProps(
            cp_file=cp_file,
            cpprop_file=cpprop_file,
            paths_file=paths_file,
            ligand_sdf=ligand_sdf,
            pl_complex_xyz=pl_complex_xyz,
            identifier=structure_id,
        )
        include_esp = "esp" in qtaim_props.critical_points[0].props
        prop_list = DEFAULT_PROPS + ESP_NAMES if include_esp else DEFAULT_PROPS
        null_props = torch.FloatTensor([0.0] * len(prop_list))
        target = df[df[COL_NAMES[dataset]["id_col"]] == structure_id][COL_NAMES[dataset]["aff_col"]].values[0]

        G = nx.Graph(target=target)

        bcps = [cp for cp in qtaim_props.critical_points if cp.name == "bond_critical_point"]
        ncps = [cp for cp in qtaim_props.critical_points if cp.name == "nucleus_critical_point"]
        atom_id_to_ncp = {ncp.corresponding_atom_id: ncp for ncp in ncps}

        # loop over the intermolecular BCPs and pull out their neighboring NCPs
        for bcp in bcps:
            if len(bcp.atom_neighbors) < 2:
                continue  # incomplete path
            elif len(bcp.atom_neighbors) > 2:
                raise ValueError(f"More than two neighbors: {structure_id}")
            if not all([neighbor_id in atom_id_to_ncp for neighbor_id in bcp.atom_neighbors]):
                continue
                # in rare cases, NCPs do not map 1:1 to atoms, e.g. because two atoms are too close together
                # --> not included in graph construction; example: PDE10A, 5sfj_1153
            ncp_neighbors = atom_id_to_ncp[bcp.atom_neighbors[0]], atom_id_to_ncp[bcp.atom_neighbors[1]]
            intra_ligand = all([idx < qtaim_props.natoms_ligand for idx in bcp.atom_neighbors])
            if not (bcp.intermolecular or intra_ligand):
                continue  # only saving intermolecular BCPs and those for BCPs within the ligand (covalent & non-covalent)
            for ncp in ncp_neighbors:
                props = [ncp.props[prop] for prop in prop_list]
                atom_type_id = ATOM_NEIGHBOR_IDS.get(ncp.corresponding_atom_symbol, OTHER)
                coords = ncp.position
                is_ligand = ncp.corresponding_atom_id < qtaim_props.natoms_ligand
                G.add_node(
                    ncp.idx,
                    node_props=torch.FloatTensor(props),
                    atom_type_id=atom_type_id,
                    node_coords=torch.FloatTensor(coords),
                    is_ligand=is_ligand,
                )

            props = [bcp.props[prop] for prop in prop_list]
            coords = bcp.position
            distance = np.linalg.norm(ncp_neighbors[0].position - ncp_neighbors[1].position)
            G.add_edge(
                *(ncp.idx for ncp in ncp_neighbors),
                edge_props=torch.FloatTensor(props),
                edge_coords=torch.FloatTensor(coords),
                distance=distance,
            )

        # add intra-ligand edges
        coords = nx.get_node_attributes(G, "node_coords")
        for n1 in G.nodes:
            for n2 in G.nodes:
                if G.nodes[n1]["is_ligand"] and G.nodes[n2]["is_ligand"]:
                    if G.get_edge_data(n1, n2) is None:  # don't overwrite existing ligand edges characterized by BCPs
                        distance = np.linalg.norm(coords[n1] - coords[n2])
                        if distance > 0:  # no self-loops
                            G.add_edge(
                                n1,
                                n2,
                                edge_props=null_props,
                                edge_coords=NULL_COORDS,
                                distance=distance,
                            )
                            #  need same edge attributes for all edges so we can convert from networkx to torch_geometric
        return {structure_id: G}
    except:
        print(f"\n{structure_id}\n", flush=True)
        return None


def get_graph_from_cp_prop_file_intramolecular_critic2(*args, **kwargs):
    raise NotImplementedError


def get_bcp_graph_from_cp_prop_file_critic2(
    df: pd.DataFrame, output_cri: str, dataset: str, **kwargs
) -> Union[Dict, None]:
    """Same as get_graph_from_cp_prop_file_intermolecular, but for BCP-based graphs. Uses only intermolecular BCPs and critic2.
    kwargs unused, just for compatibility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with critical-point information
    output_cri : str
        path to output.cri file from critic2 output
    dataset : str
        type of dataset, either "pdbbind" or "pde10a"

    Raises
    ------
    ValueError
        in case of strange assigments of atom neighbors
    """
    try:
        folder = os.path.dirname(output_cri)
        structure_id = get_structure_id(folder, dataset)
        ligand_sdf = os.path.join(folder, f"{os.path.basename(folder)}_ligand_with_hydrogens.sdf")
        pl_complex_xyz = os.path.join(folder, "pl_complex.xyz")
        qtaim_props = QtaimPropsCritic2(
            basepath=folder, output_cri=output_cri, pl_complex_xyz=pl_complex_xyz, ligand_sdf=ligand_sdf
        )
        prop_list = ["density", "laplacian", "gradient_norm"]  # critic2

        target = df[df[COL_NAMES[dataset]["id_col"]] == structure_id][COL_NAMES[dataset]["aff_col"]].values[0]

        G = nx.Graph(target=target, include_esp=False)  # no esp in critic2?

        bcps = [cp for cp in qtaim_props.critical_points if cp.name == "bond_critical_point" and cp.intermolecular]

        # loop over the intermolecular BCPs and pull out their neighboring NCPs
        for bcp in bcps:
            if len(bcp.atom_neighbors) > 2:
                raise ValueError(f"More than two neighbors: {structure_id}")
            # neighbors = [ncp for ncp in ncps if ncp.corresponding_atom_id in bcp.atom_neighbors]
            atom_neighbors_symbol = bcp.atom_neighbors_symbol
            if len(atom_neighbors_symbol) == 2:  # BCP with two neighbors
                pass
            elif len(atom_neighbors_symbol) == 1:  # BCP where only one neighbor was found
                atom_neighbors_symbol = atom_neighbors_symbol + ["*"]  # append dummy
            elif len(atom_neighbors_symbol) == 0:  # NCP
                atom_neighbors_symbol = ["*", "*"]  # append 2x dummy
            atom_neighbors_type_id = [ATOM_NEIGHBOR_IDS.get(n, OTHER) for n in atom_neighbors_symbol]

            props = [bcp.props[prop] for prop in prop_list]
            coords = bcp.position
            G.add_node(
                bcp.idx,
                node_props=torch.FloatTensor(props),
                atom_type_id=atom_neighbors_type_id,
                node_coords=torch.FloatTensor(coords),
            )
        return {structure_id: G}
    except:
        return None


def generate_pickle(
    search_path: str,
    affinity_data: str,
    save_path: str,
    qm_method: str = "xtb",
    dataset: str = "pdbbind",
    nucleus_critical_points=False,
    **kwargs,
):
    if qm_method == "xtb" or qm_method == "psi4":
        graph_fun = (
            get_graph_from_cp_prop_file_intramolecular if nucleus_critical_points else get_bcp_graph_from_cp_prop_file
        )
        cpprop_files = sorted(glob.glob(os.path.join(search_path, "*", "CPprop.txt")))
    elif qm_method.startswith("dftb"):
        graph_fun = (
            get_graph_from_cp_prop_file_intramolecular_critic2
            if nucleus_critical_points
            else get_bcp_graph_from_cp_prop_file_critic2
        )
        cpprop_files = sorted(glob.glob(os.path.join(search_path, "*", "output.cri")))
    else:
        raise ValueError("Unknown qm_method")

    df = pd.read_csv(affinity_data)

    res = {}

    res_par = Parallel(n_jobs=-1)(
        delayed(graph_fun)(df, cpprop_file, dataset, **kwargs) for cpprop_file in tqdm(cpprop_files)
    )
    # res_par = []
    # for cpprop_file in tqdm(cpprop_files):
    #    res_par.append(graph_fun(df, cpprop_file, dataset, **kwargs))

    for r in res_par:
        if r is not None:
            res.update(r)

    with open(save_path, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved pickle to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_path", type=str, required=True)
    parser.add_argument("--affinity_data", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--only_key_interactions", action="store_true", dest="only_key_interactions", default=False)
    parser.add_argument(
        "--only_short_interactions", action="store_true", dest="only_short_interactions", default=False
    )
    parser.add_argument("--no_c_c_interactions", action="store_true", dest="no_c_c_interactions", default=False)
    parser.add_argument("--dataset", type=str, default="pdbbind")
    parser.add_argument(
        "--nucleus_critical_points", action="store_true", dest="nucleus_critical_points", default=False
    )
    parser.add_argument("--qm_method", type=str, default="xtb")

    args = parser.parse_args()

    generate_pickle(
        args.search_path,
        args.affinity_data,
        args.save_path,
        qm_method=args.qm_method,
        dataset=args.dataset,
        nucleus_critical_points=args.nucleus_critical_points,
        only_key_interactions=args.only_key_interactions,
        only_short_interactions=args.only_short_interactions,
        no_c_c_interactions=args.no_c_c_interactions,
    )
