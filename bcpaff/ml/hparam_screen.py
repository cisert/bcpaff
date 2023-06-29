"""
© 2023, ETH Zurich
"""

import argparse
import glob
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from bcpaff.ml import statsig
from bcpaff.ml.cluster_tools import generate_bash_script, generate_cmds_file, submit_job
from bcpaff.ml.ml_utils import run_id_to_hparams
from bcpaff.ml.test import HPARAM_KEYS
from bcpaff.utils import HPARAMS, ROOT_PATH


def generate_all_config_files(base_output_dir: str, hparam_file: Optional[str] = None) -> List[str]:
    config_files = []
    if hparam_file is not None:
        hparam_df = pd.read_csv(hparam_file)
    else:
        hparam_df = HPARAMS
    for _, row in hparam_df.iterrows():
        hparams = row[HPARAM_KEYS].to_dict()
        config_file = os.path.join(base_output_dir, row.run_id, "hparams.json")
        dirname = os.path.dirname(config_file)
        checkpoint_path = os.path.join(dirname, "checkpoint.pt")
        if os.path.exists(checkpoint_path):  # already trained
            continue
        os.makedirs(dirname, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(hparams, f)
        config_files.append(config_file)
    return config_files


def run_hparam_screen(
    pickle_file: str,
    hparam_file: str,
    base_output_dir: Optional[str] = None,
    dataset: str = "pdbbind",
    split_type: str = "random",
    overwrite: bool = False,
    cluster_options: Optional[str] = None,
    num_epochs: int = 300,
) -> Optional[int]:
    config_files = generate_all_config_files(base_output_dir=base_output_dir, hparam_file=hparam_file)
    cmds_file_path = os.path.join(base_output_dir, "slurm_files", "cmds.txt")
    script_path = os.path.join(base_output_dir, "slurm_files", "jobscript.sh")
    os.makedirs(os.path.dirname(cmds_file_path), exist_ok=True)
    num_lines = generate_cmds_file(
        cmds_file_path,
        config_files,
        pickle_file,
        dataset=dataset,
        split_type=split_type,
        base_output_dir=base_output_dir,
        overwrite=overwrite,
        hparam_file=hparam_file,
        num_epochs=num_epochs,
    )
    time = "04:00:00" if dataset == "pde10a" else "24:00:00"
    generate_bash_script(script_path, cmds_file_path, time=time, num_cores=4, memory=64)

    job_id = submit_job(script_path, num_lines=num_lines, cluster_options=cluster_options)
    return job_id
    # a = 2

    # for config_file in config_files:
    #     dirname = os.path.dirname(config_file)
    #     checkpoint_path = os.path.join(dirname, "checkpoint.pt")
    #     if os.path.exists(checkpoint_path):  # already trained
    #         continue
    #     else:
    #         script_path = os.path.join(dirname, "train.sh")
    #         generate_bash_script(
    #             script_path,
    #             pickle_file,
    #             config_file,
    #             dataset=dataset,
    #             split_type=split_type,
    #             num_cores=4,
    #             memory=16,
    #             base_output_dir=base_output_dir,
    #             overwrite=overwrite,
    #         )
    #         submit_job(script_path, no_cluster=no_cluster)

    #         submit_job(
    #             pickle_file,
    #             config_file,
    #             dataset=dataset,
    #             split_type=split_type,
    #             base_output_dir=base_output_dir,
    #             overwrite=False,
    #             num_cores=4,
    #             memory=16,
    #             no_cluster=no_cluster,
    #         )


def collect_results(basepath, force_recompute=True, last_epoch=False, hparam_file=None):
    if last_epoch:
        hparam_results_csv = os.path.join(basepath, "hparam_results_last_epoch.csv")
    else:
        hparam_results_csv = os.path.join(basepath, "hparam_results.csv")
    have_existing = False
    if os.path.exists(hparam_results_csv) and not force_recompute:
        df = pd.read_csv(hparam_results_csv)
        have_existing = True

    folders = sorted(glob.glob(os.path.join(basepath, "run_*")))
    keys = ["eval_mae", "eval_rmse", "train_mae", "train_rmse"]
    all_res = []
    for folder in tqdm(folders):
        run_id = os.path.basename(folder)
        if last_epoch:
            checkpoint_file = os.path.join(folder, "last_epoch_checkpoint.pt")
        else:
            checkpoint_file = os.path.join(folder, "checkpoint.pt")
        if not os.path.exists(checkpoint_file):
            continue
        with open(os.path.join(folder, "hparams.json"), "r") as f:
            hparams = json.load(f)
        try:
            assert run_id_to_hparams(run_id, hparam_file=hparam_file) == hparams  # sanity check
        except:
            print(run_id)
        if have_existing and len(df[df.run_id == run_id]):
            continue  # already in the df
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        res = {"run_id": run_id}
        res.update({k: checkpoint[k] for k in keys})
        rmse, le, ue = statsig.rmse(checkpoint["y_eval"], checkpoint["yhat_eval"])
        res["rmse_le"] = le
        res["rmse_ue"] = ue
        assert np.isclose(rmse, res["eval_rmse"])
        hparams.update(res)
        all_res.append(hparams)
    df_update = pd.DataFrame(all_res)
    df = pd.concat([df, df_update]) if have_existing else df_update
    if len(df) == 0:
        print(f"No completed experiments, couldn't write results from hparam screen", flush=True)
    else:
        df = df[["run_id"] + [col for col in df.columns if col != "run_id"]]  #  move run_id column to front
        df.to_csv(hparam_results_csv, index=False)
        print(f"Wrote results from hparam screen to {hparam_results_csv}", flush=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickle_file",
        type=str,
    )
    parser.add_argument("--dataset", type=str, default="pdbbind")
    parser.add_argument("--split_type", type=str, default="random")
    parser.add_argument("--base_output_dir", type=str, required=True)
    parser.add_argument("--collect_results", action="store_true", default=False, dest="collect_results")
    parser.add_argument("--force_recompute", action="store_true", default=False, dest="force_recompute")
    parser.add_argument("--cluster_options", type=str, default=None)
    parser.add_argument("--hparam_file", type=str, default=os.path.join(ROOT_PATH, "new_hparams.csv"))
    parser.add_argument("--last_epoch", action="store_true", default=False, dest="last_epoch")
    parser.add_argument("--scaler_name", type=str, default="qtaim_scaler")
    parser.add_argument("--num_epochs", type=int, default=1000)
    args = parser.parse_args()

    if args.collect_results:
        df = collect_results(
            args.base_output_dir,
            force_recompute=args.force_recompute,
            last_epoch=args.last_epoch,
            hparam_file=args.hparam_file,
        )
    else:
        job_id = run_hparam_screen(
            args.pickle_file,
            hparam_file=args.hparam_file,
            base_output_dir=args.base_output_dir,
            dataset=args.dataset,
            split_type=args.split_type,
            overwrite=False,
            cluster_options=args.cluster_options,
            num_epochs=args.num_epochs,
        )
        if job_id is not None:
            print(job_id)
