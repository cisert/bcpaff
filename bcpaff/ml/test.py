"""
© 2023, ETH Zurich
"""

import argparse
import os
import pickle
import random

import pandas as pd
import torch

from bcpaff.ml.ml_utils import get_data_loader, load_checkpoint
from bcpaff.ml.net_utils import QtaimScaler
from bcpaff.ml.test import run_test
from bcpaff.ml.train import eval_loop, get_split_idxs
from bcpaff.utils import SEED

random.seed(SEED)
torch.manual_seed(SEED)
HPARAM_KEYS = [
    "batch_size",
    "kernel_dim",
    "mlp_dim",
    "cutoff",
    "baseline_atom_ids",
    "aggr",
    "pool",
    "properties",
    "n_kernels",
    "ncp_graph",
]


def get_best_hparams(base_output_dir, last_epoch=False, quiet=False):
    if last_epoch:
        hparam_file = os.path.join(base_output_dir, "hparam_results_last_epoch.csv")
    else:
        hparam_file = os.path.join(base_output_dir, "hparam_results.csv")
    if not os.path.exists(hparam_file):
        raise ValueError(f"hparam_file not found: {hparam_file}")
    df = pd.read_csv(hparam_file).sort_values(by="eval_rmse")
    if not quiet:
        print(f"Found {len(df)} results from hparam optimization and picking the one with best eval_rmse")
    best_hparams = df.iloc[0][HPARAM_KEYS].to_dict()
    best_run_id = df.iloc[0].run_id
    dir_best_eval_rmse = os.path.join(base_output_dir, best_run_id)
    if last_epoch:
        checkpoint_savepath = os.path.join(dir_best_eval_rmse, "last_epoch_checkpoint.pt")
    else:
        checkpoint_savepath = os.path.join(dir_best_eval_rmse, "checkpoint.pt")
    if not quiet:
        print(f"Using model {checkpoint_savepath}")
    return best_hparams, checkpoint_savepath


def run_test(pickle_file, base_output_dir, dataset, split_type, last_epoch=False):
    hparams, checkpoint_savepath = get_best_hparams(base_output_dir, last_epoch=last_epoch)
    model = load_checkpoint(hparams, checkpoint_savepath)
    train_idxs, _, core_idxs, test_idxs = get_split_idxs(dataset, split_type)

    scaler = QtaimScaler(
        pickle_file, train_idxs, ncp_graph=hparams["ncp_graph"]
    )  # still use train set normalization values

    test_loader = get_data_loader(
        pickle_file, hparams, scaler, test_idxs, shuffle=False, pickle_data=scaler.pickle_data
    )

    loaders = {"test": test_loader}

    if dataset == "pdbbind":  # core set only for pdbbind
        core_loader = get_data_loader(
            pickle_file, hparams, scaler, core_idxs, shuffle=False, pickle_data=scaler.pickle_data
        )
        loaders["core"] = core_loader

    for key, loader in loaders.items():
        mae, rmse, loss, y, yhat = eval_loop(model, loader, torch.nn.MSELoss())
        results = {
            f"{key}_mae": mae,
            f"{key}_rmse": rmse,
            f"{key}_loss": loss,
            f"y_{key}": y,
            f"yhat_{key}": yhat,
            f"{key}_idxs": loader.dataset.pdb_ids,
            "hparams": hparams,
        }

        if last_epoch:
            results_savepath = os.path.join(base_output_dir, f"{key}_results_last_epoch.pkl")
        else:
            results_savepath = os.path.join(base_output_dir, f"{key}_results.pkl")
        with open(results_savepath, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {key} set results to {results_savepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file", type=str, required=True)
    parser.add_argument("--base_output_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="pdbbind")
    parser.add_argument("--split_type", type=str, default="random")
    parser.add_argument("--last_epoch", action="store_true", default=False)
    args = parser.parse_args()

    run_test(
        args.pickle_file,
        args.base_output_dir,
        args.dataset,
        args.split_type,
        args.last_epoch,
    )
