"""
© 2023, ETH Zurich
"""

import argparse
import os
import subprocess
from typing import Optional

from bcpaff.data_processing.data_processing import DATASETS
from bcpaff.ml.hparam_screen import run_hparam_screen
from bcpaff.utils import DATASETS_AND_SPLITS, PROCESSED_DATA_PATH, ROOT_PATH


def submit_job_collect_results(
    base_output_dir: str, hparam_file: str, cluster_options: Optional[str] = None, last_epoch: bool = False
):
    cmd_str = f"""python -c 'from bcpaff.ml.hparam_screen import collect_results; collect_results(\\\"{base_output_dir}\\\", hparam_file=\\\"{hparam_file}\\\", last_epoch={last_epoch})' """
    cluster_options = "" if cluster_options is None else cluster_options
    if cluster_options != "no_cluster":
        slurm_output_file = os.path.join(base_output_dir, "slurm_files", "out_files", "collect_results_out_%A.out")
        cmd_str = f"""sbatch --parsable {cluster_options} --output={slurm_output_file} --wrap "{cmd_str}" """
    completed_process = subprocess.run(
        cmd_str,
        shell=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
    )
    job_id = completed_process.stdout.rstrip("\n")
    return job_id


def submit_job_run_test(
    pickle_file: str,
    base_output_dir: str,
    dataset: str,
    split_type: str,
    cluster_options: Optional[str] = None,
    last_epoch: bool = False,
):
    cmd_str = f"""python -c 'from bcpaff.ml.test import run_test; run_test(\\\"{pickle_file}\\\", \\\"{base_output_dir}\\\", \\\"{dataset}\\\", \\\"{split_type}\\\", last_epoch={last_epoch})' """
    cluster_options = "" if cluster_options is None else cluster_options
    if cluster_options != "no_cluster":
        slurm_output_file = os.path.join(base_output_dir, "slurm_files", "out_files", "test_out_%A.out")
        cmd_str = f"""sbatch -n 4 --mem-per-cpu=16000 --parsable {cluster_options} --output={slurm_output_file} --wrap "{cmd_str}" """
    completed_process = subprocess.run(
        cmd_str,
        shell=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
    )
    job_id = completed_process.stdout.rstrip("\n")
    return job_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_run", action="store_true", dest="test_run", default=False)
    parser.add_argument("--cluster_options", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--last_epoch", action="store_true", default=False)
    args = parser.parse_args()

    datasets_to_run = DATASETS if args.dataset is None else [args.dataset]
    t = 0
    for dataset in datasets_to_run:
        for split_type in DATASETS_AND_SPLITS[dataset]:
            for ncp_bcp in ["ncp", "bcp"]:
                for atom_ids in ["props", "atom_ids", "atom_ids_and_props"]:
                    base_output_dir = os.path.join(
                        PROCESSED_DATA_PATH, "model_runs_1000", ncp_bcp, dataset, split_type, atom_ids
                    )
                    print(t)
                    t += 1
                    pickle_file = os.path.join(
                        PROCESSED_DATA_PATH,
                        "prepared_structures",
                        dataset,
                        f"qtaim_props_{ncp_bcp}.pkl",
                    )
                    if args.test_run:
                        hparam_file = os.path.join(ROOT_PATH, "hparam_files", f"hparams_{ncp_bcp}_{atom_ids}_mini.csv")
                        num_epochs = 20
                    else:
                        hparam_file = os.path.join(ROOT_PATH, "hparam_files", f"hparams_{ncp_bcp}_{atom_ids}.csv")
                        num_epochs = 1000
                    job_id = run_hparam_screen(
                        pickle_file=pickle_file,
                        hparam_file=hparam_file,
                        base_output_dir=base_output_dir,
                        dataset=dataset,
                        split_type=split_type,
                        overwrite=True,
                        cluster_options=args.cluster_options,
                        num_epochs=num_epochs,
                    )
                    cluster_options = (
                        f"--dependency=afterok:{job_id}" if args.cluster_options is None else args.cluster_options
                    )
                    job_id = submit_job_collect_results(
                        base_output_dir, hparam_file, cluster_options=cluster_options, last_epoch=args.last_epoch
                    )
                    cluster_options = (
                        f"--dependency=afterok:{job_id}" if args.cluster_options is None else args.cluster_options
                    )
                    submit_job_run_test(
                        pickle_file,
                        base_output_dir,
                        dataset,
                        split_type,
                        cluster_options=cluster_options,
                        last_epoch=args.last_epoch,
                    )
