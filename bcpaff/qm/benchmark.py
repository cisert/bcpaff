"""
© 2023, ETH Zurich
"""

import argparse
import glob
import os
import pickle
import shutil
import subprocess
import time
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import cdist
from tqdm import tqdm

from bcpaff.qtaim.multiwfn_tools import run_multiwfn_analysis
from bcpaff.qtaim.qtaim_reader import QtaimProps
from bcpaff.utils import ANALYSIS_PATH, DATA_PATH, PROCESSED_DATA_PATH

XTB_VALIDATION_BASEPATH = os.path.join(ANALYSIS_PATH, "xtb_validation")
os.makedirs(XTB_VALIDATION_BASEPATH, exist_ok=True)

NUM_BENCHMARK_STRUCTURES = 3
THRESHOLD = 0.1
NO_CORRES_PLACEHOLDER = -9999
ABBREVIATIONS = {
    "nucleus_critical_point": "NCP",
    "bond_critical_point": "BCP",
    "ring_critical_point": "RCP",
    "cage_critical_point": "CCP",
}


def submit_benchmark_job(structure_basepath: str, level_of_theory: str, cluster_options: Optional[str] = None) -> str:
    conda_env = "bcpaff" if level_of_theory == "xtb" else "bcpaff_psi4"
    cmd_str = f"""source ~/.bashrc; source activate {conda_env}; python -c 'from bcpaff.qm.benchmark import run_benchmark; run_benchmark(\\\"{structure_basepath}\\\", \\\"{level_of_theory}\\\")' """
    cluster_options = "" if cluster_options is None else cluster_options
    if cluster_options != "no_cluster":
        pdb_id = os.path.basename(structure_basepath)
        slurm_output_file = os.path.join(
            XTB_VALIDATION_BASEPATH,
            pdb_id,
            level_of_theory,
            "slurm_files",
            "out_files",
            f"{level_of_theory}_out_%A.out",
        )
        os.makedirs(os.path.dirname(slurm_output_file), exist_ok=True)
        cmd_str = f"""sbatch -n 4 --mem-per-cpu=16000 --tmp=100000 --time=48:00:00 --parsable {cluster_options} --output={slurm_output_file} --wrap "{cmd_str}" """
    completed_process = subprocess.run(
        cmd_str,
        shell=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
    )
    job_id = completed_process.stdout.rstrip("\n")
    return job_id


def run_benchmark(structure_basepath: str, level_of_theory: str):
    pdb_id = os.path.basename(structure_basepath)
    dest_basepath = os.path.join(XTB_VALIDATION_BASEPATH, pdb_id, level_of_theory, pdb_id)  # naming_convention...
    os.makedirs(dest_basepath, exist_ok=True)

    filenames_to_copy = [
        f"{pdb_id}_ligand_with_hydrogens.sdf",
        f"{pdb_id}_pocket_with_hydrogens.sdf",
        f"{pdb_id}_pocket_with_hydrogens.xyz",
        "pl_complex.sdf",
        "pl_complex.xyz",
        "psi4_input.pkl",
        "chrg_uhf.json",
    ]
    for filename in filenames_to_copy:
        src = os.path.join(structure_basepath, filename)
        dest = os.path.join(dest_basepath, filename)
        shutil.copy(src, dest)

    results = {}

    # compute wfn
    t0 = time.time()
    if level_of_theory == "dft":
        import psi4  # do the imports here since environments for psi4 and xtb are incompatible

        from bcpaff.qm.compute_wfn_psi4 import compute_wfn_psi4

        psi4.core.be_quiet()
        psi4_input_pickle = os.path.join(dest_basepath, "psi4_input.pkl")
        wfn_file = compute_wfn_psi4(
            psi4_input_pickle,
            memory=8,
            num_cores=1,
            level_of_theory="wb97x-d",
            basis_set="def2-qzvp",
        )
    elif level_of_theory == "xtb":
        from bcpaff.qm.compute_wfn_xtb import (
            compute_wfn_xtb,
        )  # do the imports here since environments for psi4 and xtb are incompatible

        wfn_file = compute_wfn_xtb(os.path.join(dest_basepath, "pl_complex.xyz"))
    else:
        raise ValueError("Unknown level of theory")
    t1 = time.time()
    time_needed = t1 - t0

    # run multiwfn analysis
    ligand_sdf = os.path.join(dest_basepath, f"{pdb_id}_ligand_with_hydrogens.sdf")
    num_ligand_atoms = next(Chem.SDMolSupplier(ligand_sdf, removeHs=False)).GetNumAtoms()
    cp_file, cpprop_file, paths_file = run_multiwfn_analysis(
        wfn_file, only_intermolecular=False, only_bcps=False, num_ligand_atoms=num_ligand_atoms, include_esp=False
    )
    qtaim_props = QtaimProps(basepath=dest_basepath)

    results[f"{level_of_theory}"] = (
        qtaim_props,
        time_needed,
        cp_file,
        cpprop_file,
        paths_file,
        wfn_file,
    )

    results_savepath = os.path.join(dest_basepath, "results.pkl")
    with open(results_savepath, "wb") as f:
        pickle.dump(results, f)


def get_equivalent_points(results):
    equivalent_points = {}
    qtaim_props_ref = results["dft"]
    other_methods = ["xtb"]
    other_qtaim_props = [results[method] for method in other_methods]

    for method, oqp in zip(other_methods, other_qtaim_props):
        other_points = oqp.cp_positions
        distance_matrix = cdist(qtaim_props_ref.cp_positions, other_points)
        corresponding_idx = np.argmin(distance_matrix, axis=1)
        no_corresponding = np.min(distance_matrix, axis=1) > THRESHOLD
        corresponding_idx[no_corresponding] = NO_CORRES_PLACEHOLDER
        equivalent_points[method] = corresponding_idx
    df_equiv = pd.DataFrame.from_dict(equivalent_points)
    df_equiv.loc[:, "point_name"] = [ABBREVIATIONS[cp.name] for cp in qtaim_props_ref.critical_points]
    # check that the point names (types) match
    for _, row in df_equiv.iterrows():
        for method in other_methods:
            method_idx = row[method]
            if method_idx == NO_CORRES_PLACEHOLDER:
                continue
            assert row.point_name == ABBREVIATIONS[results[method].critical_points[method_idx].name]
    return df_equiv


def summarize_benchmark_results(computed_results_benchmark, force_recompute=False):
    folders = glob.glob(os.path.join(computed_results_benchmark, "*"))
    pdb_ids = [folder for folder in folders if not os.path.isfile(folder)]
    pdb_ids = sorted([os.path.basename(pdb_id) for pdb_id in pdb_ids])
    all_savepaths = []
    for pdb_id in pdb_ids:
        results_savepath = os.path.join(computed_results_benchmark, pdb_id, "benchmark_results.pkl")
        all_savepaths.append(results_savepath)
        if os.path.exists(results_savepath) and not force_recompute:
            continue
        # load results from xTB and DFT
        results = {}
        for level_of_theory in ["dft", "xtb"]:
            qtaim_props = QtaimProps(
                basepath=os.path.join(computed_results_benchmark, pdb_id, level_of_theory, pdb_id)
            )
            results[level_of_theory] = qtaim_props

        reference_name = "dft"
        df_equiv = get_equivalent_points(results)
        properties = list(results[reference_name].critical_points[0].props.keys())
        iterables = [results.keys(), properties + ["x", "y", "z", "intermolecular"]]
        multiindex = pd.MultiIndex.from_product(iterables, names=["method", "property"])
        df = pd.DataFrame(columns=multiindex)
        for method in results.keys():
            qtaim_props = results[method]
            for prop in properties:
                for atom_idx_ref, row in df_equiv.iterrows():
                    atom_idx = row[method] if method != reference_name else atom_idx_ref
                    if atom_idx == NO_CORRES_PLACEHOLDER:
                        continue
                    cp = qtaim_props.critical_points[atom_idx]
                    val = cp.props[prop]
                    df.loc[atom_idx_ref, (method, prop)] = val
            for atom_idx_ref, row in df_equiv.iterrows():  # add the coordinates (only needs one iteration)
                atom_idx = row[method] if method != reference_name else atom_idx_ref
                if atom_idx == NO_CORRES_PLACEHOLDER:
                    continue
                cp = qtaim_props.critical_points[atom_idx]
                pos = cp.position
                df.loc[atom_idx_ref, (method, "x")] = pos[0]
                df.loc[atom_idx_ref, (method, "y")] = pos[1]
                df.loc[atom_idx_ref, (method, "z")] = pos[2]
                df.loc[atom_idx_ref, (method, "intermolecular")] = cp.intermolecular
        df.loc[:, "point_name"] = pd.Series(df.index).apply(lambda x: df_equiv.loc[x, "point_name"])
        with open(results_savepath, "wb") as f:
            pickle.dump((df, df_equiv), f)
    return all_savepaths


def plot_benchmark_results(all_savepaths):
    dfs, dfs_equiv = [], []
    for savepath in all_savepaths:
        with open(savepath, "rb") as f:
            df, df_equiv = pickle.load(f)
        dfs.append(df)
        dfs_equiv.append(df_equiv)
    df = pd.concat(dfs, axis=0)
    df = df.reset_index()


def select_benchmark_structures() -> pd.DataFrame:
    structure_basepath = os.path.join(PROCESSED_DATA_PATH, "prepared_structures", "pdbbind")
    pl_complex_xyzs = glob.glob(os.path.join(structure_basepath, "*", "pl_complex.xyz"))
    pdb_ids = [os.path.basename(os.path.dirname(x)) for x in pl_complex_xyzs]
    num_atoms = []
    for pl_complex_xyz in tqdm(pl_complex_xyzs):
        with open(pl_complex_xyz, "r") as f:
            num_atoms.append(int(f.readlines()[0]))
    df = pd.DataFrame({"pdb_id": pdb_ids, "num_atoms": num_atoms, "pl_complex_xyz": pl_complex_xyzs})
    df_sample = df.sort_values(by="num_atoms").iloc[:NUM_BENCHMARK_STRUCTURES]

    csv_savepath = os.path.join(XTB_VALIDATION_BASEPATH, "df_sample.csv")
    df_sample.to_csv(csv_savepath, index=False)
    print(f"Wrote df_sample to {csv_savepath}", flush=True)
    return df_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_options", type=str, default="no_cluster")
    args = parser.parse_args()

    computed_results_basepath = os.path.join(XTB_VALIDATION_BASEPATH, "computed_on_euler")
    all_savepaths = summarize_benchmark_results(computed_results_basepath, force_recompute=True)
    plot_benchmark_results(all_savepaths)
    # df_sample = select_benchmark_structures()
    # job_ids = []
    # for _, row in df_sample.iterrows():
    #     structure_basepath = os.path.dirname(row.pl_complex_xyz)
    #     for level_of_theory in ["dft", "xtb"]:
    #         job_id = submit_benchmark_job(structure_basepath, level_of_theory)
