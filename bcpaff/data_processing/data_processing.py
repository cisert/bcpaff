"""
© 2023, ETH Zurich
"""

import argparse
import glob
import os
import pickle
import subprocess
from typing import List, Optional, Tuple

import pandas as pd

from bcpaff.ml.cluster_tools import submit_job
from bcpaff.utils import DATA_PATH, PROCESSED_DATA_PATH, REPORT_PATH

DATASETS = ["pdbbind", "pde10a"]
AFFINITY_DATA = {
    "pdbbind": os.path.join(DATA_PATH, "pdbbind", "pdbbind2019_affinity.csv"),
    "pde10a": os.path.join(DATA_PATH, "pde10a", "10822_2022_478_MOESM2_ESM.csv"),
}
INPUT_FOLDERS = {
    "pdbbind": os.path.join(DATA_PATH, "pdbbind", "dataset"),
    "pde10a": os.path.join(DATA_PATH, "pde10a", "pde-10_pdb_bind_format_blinded"),
    "d2dr": os.path.join(DATA_PATH, "d2dr", "D2DR_complexes_prepared"),
}


def generate_jobscript_structure_prep_and_qm(
    dataset: str,
    base_input_dir: str,
    base_output_dir: str,
    test_run: bool = True,
    esp: bool = False,
    qm_method: str = "xtb",
    cutoff: int = 6,
    keep_wfn: bool = True,
) -> Tuple[str, int]:
    """Generate file with python -m bcpaff.data_processing.structure_prep_and_qm commands for compounds from specified dataset.

    Parameters
    ----------
    dataset : str
        name of dataset, either pdbbind or pde10a
    base_input_dir : str
        path of input data
    base_output_dir : str
        path to output folder
    test_run : bool, optional
        whether to only use the first 5 compounds for quick testing, by default True

    Returns
    -------
    tuple (str, str)
        path to cmds file, number of lines in cmds file
    """

    cmds_file_path = os.path.join(base_output_dir, "slurm_files", "cmds.txt")
    os.makedirs(os.path.dirname(cmds_file_path), exist_ok=True)
    folders = sorted(glob.glob(os.path.join(base_input_dir, "*")))
    folders = [folder for folder in folders if os.path.isdir(folder)]

    structure_ids = [os.path.basename(folder) for folder in folders]

    if test_run:
        if dataset == "pdbbind":
            structure_ids = ["3zzf", "1w8l", "5eb2", "2r58", "3ao4"]  # something from train/val/test set
        elif dataset == "pde10a":
            structure_ids = ["5sf8_0", "5sfr_1", "5sf4_2", "5se7_11"]  # something from train/val/test set

    lines = []
    for structure_id in structure_ids:
        line = f"python -m bcpaff.data_processing.structure_prep_and_qm {structure_id}"
        line += f" --dataset {dataset}"
        line += f" --basepath {base_input_dir}"
        line += f" --output_basepath {base_output_dir}"
        if qm_method.startswith("dftb"):
            line += f" --implicit_solvent water"
        elif qm_method == "xtb":
            line += f" --implicit_solvent alpb_water"
        line += f" --cutoff {cutoff}"
        line += f" --qm_method {qm_method}"
        if esp:
            line += f" --esp"
        if keep_wfn:
            line += f" --keep_wfn"
        lines.append(line)
    with open(cmds_file_path, "w") as f:
        f.write("\n".join(lines))
    return cmds_file_path, len(lines)


def generate_bash_script_structure_prep_and_qm(
    cmds_file_path: str, time: str = "04:00:00", num_cores: int = 4, memory: int = 16, qm_method: str = "xtb"
) -> str:
    """Generate the bash script which runs the job array for run_all function.

    Parameters
    ----------
    cmds_file_path : str
        path to cmds file for the job array
    time : str
        Slurm-formatted time specifier, by default 04:00:00 (increase for ESP)
    num_cores : int
        number of cores, by default 4
    memory : int
        total memory in GB (not MB/core; is being converted)

    Returns
    -------
    str
        path to script file that runs the job array
    """

    script_path = os.path.join(os.path.dirname(cmds_file_path), "jobscript.sh")
    out_files_folder = os.path.join(os.path.dirname(script_path), "out_files")
    os.makedirs(out_files_folder, exist_ok=True)
    conda_env = "bcpaff_psi4" if qm_method == "psi4" else "bcpaff"
    conda_env = "bcpaff"
    with open(script_path, "w") as f:
        f.write(
            f"""#!/bin/bash

#SBATCH --job-name=bcpaff                                                           # Job name 
#SBATCH -n {num_cores}                                                              # Number of CPU cores
#SBATCH --mem-per-cpu={int(memory/num_cores*1024)}                                  # Memory per CPU in MB
#SBATCH --time={time}                                                               # Maximum execution time (HH:MM:SS)
#SBATCH --tmp=4000                                                                  # Total scratch for job in MB
#SBATCH --output {os.path.join(out_files_folder, "structprep_qm_out_%A_%a.out")}    # Standard output
#SBATCH --error {os.path.join(out_files_folder, "structprep_qm_out_%A_%a.out")}     # Standard error

source ~/.bashrc; 
eval "$(conda shell.bash hook)"; conda activate {conda_env}; 
export CMDS_FILE_PATH={cmds_file_path}
export cmd=$(head -$SLURM_ARRAY_TASK_ID $CMDS_FILE_PATH|tail -1)
echo "=========SLURM_COMMAND========="
echo $cmd
echo "=========SLURM_COMMAND========="
eval $cmd
"""
        )
    return script_path


def submit_job_generate_pickle(
    search_path: str,
    affinity_data: str,
    dataset: str,
    qm_method: str,
    cluster_options: Optional[str],
    save_path: Optional[str] = None,
    nucleus_critical_points: bool = False,
):
    """Submit an sbatch job to generate pickle file based on input arguments.

    Parameters
    ----------
    search_path : str
        search path for pre-processed structures
    affinity_data : str
        path to affinity data for PDBbind or PDE10A datasets
    save_path : str, optional
        path where pickle file will be saved, by default None
    nucleus_critical_points : bool, optional
        whether or not to use BCP-centric (False) or NCP-centric (True) graph, by default False
    """
    job_id = None
    if save_path is None:
        ncp_bcp = "ncp" if nucleus_critical_points else "bcp"
        save_path = os.path.join(search_path, f"qtaim_props_{ncp_bcp}.pkl")
    cmd_str = "python -m bcpaff.ml.generate_pickle"
    cmd_str += f" --search_path {search_path}"
    cmd_str += f" --dataset {dataset}"
    cmd_str += f" --qm_method {qm_method}"
    cmd_str += f" --affinity_data {affinity_data}"
    cmd_str += f" --save_path {save_path}"
    if nucleus_critical_points:
        cmd_str += " --nucleus_critical_points"
    if cluster_options == "no_cluster":
        completed_process = subprocess.run(cmd_str, shell=True)
    else:
        slurm_output_file = os.path.join(search_path, "slurm_files", "out_files", "generate_pickle_out_%A.out")
        if cluster_options is None:
            cluster_options = ""
        completed_process = subprocess.run(
            f'sbatch --job-name=bcpaff_generate_pickle -n 8 --time=04:00:00 --mem-per-cpu=8192 {cluster_options} --parsable --output={slurm_output_file} --wrap "{cmd_str}"',
            shell=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
        )
        job_id = completed_process.stdout.rstrip("\n")
    if completed_process.returncode != 0:
        print(completed_process.returncode)
    return job_id


def homogenize_pickles(dataset: str, output_basepath: str = None):
    # only keep complexes which are present in all pickle files (NCP/BCP etc.)
    if output_basepath is None:
        output_basepath = os.path.join(PROCESSED_DATA_PATH, "prepared_structures", dataset)
    savepath_bcp_pickle = os.path.join(output_basepath, "qtaim_props_bcp.pkl")
    savepath_ncp_pickle = os.path.join(output_basepath, "qtaim_props_ncp.pkl")
    pickle_files = [savepath_bcp_pickle, savepath_ncp_pickle]
    keys_per_pickle_file = []
    all_data = {}
    for pickle_file in pickle_files:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        all_data[pickle_file] = data
        keys_per_pickle_file.append(set(data.keys()))
    keys_in_all_pickle_files = set.intersection(*keys_per_pickle_file)

    for pickle_file in pickle_files:
        keys_to_remove = sorted(list(set(all_data[pickle_file].keys()) - keys_in_all_pickle_files))
        savepath_report = os.path.join(
            REPORT_PATH, "structure_prep", dataset, f"removed_ids_{os.path.basename(pickle_file)}.txt"
        )
        os.makedirs(os.path.dirname(savepath_report), exist_ok=True)
        with open(savepath_report, "w") as f:
            f.write(
                f"These IDs were removed from {pickle_file} because they didn't exist in all of the following pickle_files:\n"
            )
            f.write("\n".join([f"    - {p}" for p in pickle_files]) + "\n")
            f.write("\n".join(keys_to_remove))
        new_data = {key: all_data[pickle_file][key] for key in keys_in_all_pickle_files}
        with open(pickle_file, "wb") as f:
            pickle.dump(new_data, f)
        print(f"Wrote homogenized data to {pickle_file}", flush=True)


def report_data_processing_outcome(search_path: str, dataset: str, affinity_data: str):
    """Generate a report on outcome of data processing (number of successfully cleaned structures,
    missing structures, radicals etc.)

    Parameters
    ----------
    search_path : str
        path to output folder from structure preparation (contains cleaned files)
    dataset : str
        which dataset, "pdbbind" or "pde10a"
    affinity_data : str
        path to csv file with affinity data, needed to retrieve the total list of compounds we start with
    """

    # analysis for structure cleaning
    col_name = "pdb_id" if dataset == "pdbbind" else "docking_folder"
    df = pd.read_csv(affinity_data, dtype={col_name: "str"}, parse_dates=False)
    paths_files = sorted(glob.glob(os.path.join(search_path, "*", "paths.pdb")))
    completed_ids = [os.path.basename(os.path.dirname(x)) for x in paths_files]
    uhf_error_ids = [x for x in completed_ids if os.path.exists(os.path.join(search_path, x, "uhf_error.txt"))]
    all_ids = sorted(df[col_name].unique().astype(str).tolist())
    missing_ids = sorted(set(all_ids) - set(completed_ids))
    base_report_dir = os.path.join(REPORT_PATH, "structure_prep", dataset)
    os.makedirs(base_report_dir, exist_ok=True)
    with open(os.path.join(base_report_dir, "completed_ids.txt"), "w") as f:
        f.write("\n".join(completed_ids))
    with open(os.path.join(base_report_dir, "missing_ids.txt"), "w") as f:
        f.write("\n".join(missing_ids))
    with open(os.path.join(base_report_dir, "uhf_error_ids.txt"), "w") as f:
        f.write("\n".join(uhf_error_ids))
    with open(os.path.join(base_report_dir, "overview.txt"), "w") as f:
        f.write(f"{len(completed_ids)} compounds for which structure preparation was successful.\n\n")
        f.write(f"Of those, {len(uhf_error_ids)} compounds had uhf_error. Those were NOT removed.\n\n")
        f.write(f"{len(missing_ids)} compounds are missing, {len(all_ids)} compounds exist in total.\n\n")

    # analysis for pickle generation
    for ncp_bcp in ["bcp", "ncp"]:
        ncp_bcp_base_report_dir = os.path.join(base_report_dir, ncp_bcp)
        os.makedirs(ncp_bcp_base_report_dir, exist_ok=True)
        pickle_file = os.path.join(search_path, f"qtaim_props_{ncp_bcp}.pkl")
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        pickle_ids = [str(x) for x in sorted(list(data.keys()))]
        missing_pickle_ids = sorted(set(completed_ids) - set(pickle_ids))
        with open(os.path.join(ncp_bcp_base_report_dir, f"pickle_missing_{ncp_bcp}.txt"), "w") as f:
            f.write("\n".join(missing_pickle_ids))
        with open(os.path.join(ncp_bcp_base_report_dir, "overview.txt"), "w") as f:
            f.write(f"{pickle_file}\n")
            f.write(
                f"{len(missing_pickle_ids)} compounds for which structure preparation was successful didn't end up in the pickle file.\n\n"
            )


def submit_structure_prep_and_qm_jobs(args: argparse.Namespace, dataset: str, cluster_options=None) -> Optional[str]:
    base_input_dir = INPUT_FOLDERS[dataset]
    if args.output_basepath is None:
        base_output_dir = os.path.join(PROCESSED_DATA_PATH, "prepared_structures", dataset)
    else:
        base_output_dir = args.output_basepath
    cmds_file_path, num_lines = generate_jobscript_structure_prep_and_qm(
        dataset,
        base_input_dir,
        base_output_dir,
        test_run=args.test_run,
        esp=args.esp,
        qm_method=args.qm_method,
        cutoff=args.cutoff,
        keep_wfn=args.keep_wfn,
    )
    script_path = generate_bash_script_structure_prep_and_qm(
        cmds_file_path, time="04:00:00", num_cores=1, memory=8, qm_method=args.qm_method
    )
    job_id = submit_job(script_path, num_lines=num_lines, cluster_options=cluster_options)
    return job_id


def submit_pickle_generation_jobs(
    dataset: str, cluster_options: Optional[str] = None, output_basepath: Optional[str] = None, qm_method: str = "xtb"
) -> Optional[str]:
    # generate pickle
    if output_basepath is None:
        search_path = os.path.join(PROCESSED_DATA_PATH, "prepared_structures", dataset)
    else:
        search_path = output_basepath
    print("=====================================")
    print(search_path)
    print("=====================================")

    job_ids = []
    affinity_data = AFFINITY_DATA[dataset]
    savepath = os.path.join(search_path, "qtaim_props_bcp.pkl")
    job_id = submit_job_generate_pickle(
        search_path,
        affinity_data,
        dataset,
        qm_method=qm_method,
        nucleus_critical_points=False,
        cluster_options=cluster_options,
        save_path=savepath,
    )
    job_ids.append(job_id)
    savepath = os.path.join(search_path, "qtaim_props_ncp.pkl")
    job_id = submit_job_generate_pickle(
        search_path,
        affinity_data,
        dataset,
        qm_method=qm_method,
        nucleus_critical_points=True,
        cluster_options=cluster_options,
        save_path=savepath,
    )
    job_ids.append(job_id)
    if not all([j is None for j in job_ids]):
        return ",".join(job_ids)


def submit_homogenize_pickle_jobs(
    dataset: str, cluster_options: Optional[str] = None, output_basepath: Optional[str] = None
):
    if output_basepath is None:
        search_path = os.path.join(PROCESSED_DATA_PATH, "prepared_structures", dataset)
    else:
        search_path = output_basepath

    cmd_str = f"""python -c 'from bcpaff.data_processing.data_processing import homogenize_pickles; homogenize_pickles(\"{dataset}\", \"{search_path}\")' """
    cluster_options = "" if cluster_options is None else cluster_options
    if cluster_options != "no_cluster":
        slurm_output_file = os.path.join(
            PROCESSED_DATA_PATH,
            "prepared_structures",
            dataset,
            "slurm_files",
            "out_files",
            "homogenize_pickles_out_%A.out",
        )
        cmd_str = f"""sbatch --parsable {cluster_options} --output={slurm_output_file} --wrap "{cmd_str}" """
    completed_process = subprocess.run(cmd_str, shell=True, universal_newlines=True, stdout=subprocess.PIPE,)
    job_id = completed_process.stdout.rstrip("\n")
    return job_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="all")
    parser.add_argument("--test_run", action="store_true", default=False, dest="test_run")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--cutoff", type=float, default=6)
    parser.add_argument("--qm_method", type=str, default="xtb")
    parser.add_argument("--output_basepath", type=str, default=None)
    parser.add_argument("--esp", action="store_true", default=False, dest="esp")
    parser.add_argument("--cluster_options", type=str, default=None)
    parser.add_argument("--keep_wfn", action="store_true", default=False, dest="keep_wfn")

    args = parser.parse_args()

    datasets_to_run = DATASETS if args.dataset is None else [args.dataset]

    if args.action == "structure_prep_and_qm":
        for dataset in datasets_to_run:
            job_ids = submit_structure_prep_and_qm_jobs(args, dataset, cluster_options=args.cluster_options)

    elif args.action == "generate_pickle":
        for dataset in datasets_to_run:
            job_ids = submit_pickle_generation_jobs(
                dataset,
                cluster_options=args.cluster_options,
                output_basepath=args.output_basepath,
                qm_method=args.qm_method,
            )

    elif args.action == "homogenize_pickles":
        # make sure that both NCP- and BCP-based pickle-files contain the same data
        for dataset in datasets_to_run:
            submit_homogenize_pickle_jobs(
                dataset, cluster_options=args.cluster_options, output_basepath=args.output_basepath
            )

    elif args.action == "report":
        # do some reporting on the outcome of the data preparation
        # number of radicals, number of failed structures etc.
        # (only xtb, no dftb+)
        for dataset, affinity_data in AFFINITY_DATA.items():
            search_path = os.path.join(PROCESSED_DATA_PATH, "prepared_structures", dataset)
            report_data_processing_outcome(search_path, dataset, affinity_data)

    elif args.action == "all":
        for dataset in datasets_to_run:
            job_ids = submit_structure_prep_and_qm_jobs(args, dataset, cluster_options=args.cluster_options)

            cluster_options = (
                f"--dependency=afterok:{job_ids}" if args.cluster_options is None else args.cluster_options
            )  # respect no_cluster
            job_ids = submit_pickle_generation_jobs(dataset, cluster_options=cluster_options)
            cluster_options = (
                f"--dependency=afterok:{job_ids}" if args.cluster_options is None else args.cluster_options
            )  # respect no_cluster
        submit_homogenize_pickle_jobs(dataset, cluster_options=cluster_options)

    else:
        print("Unknown action!")
