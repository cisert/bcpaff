"""
© 2023, ETH Zurich
"""

import argparse
import os
import pickle
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bcpaff.ml import statsig
from bcpaff.ml.ml_utils import get_data_loader, load_checkpoint
from bcpaff.ml.net_utils import QtaimScaler
from bcpaff.ml.test import DATASETS_AND_SPLITS, get_best_hparams, run_test
from bcpaff.ml.train import eval_loop, get_split_idxs
from bcpaff.utils import ROOT_PATH, SEED

random.seed(SEED)
torch.manual_seed(SEED)


def get_rmse_bounds_for_one_exp(test_results_savepath):
    with open(test_results_savepath, "rb") as f:
        res = pickle.load(f)
    rmse, le, ue = statsig.rmse(res["y_test"], res["yhat_test"])
    assert np.isclose(rmse, res["test_rmse"])
    return rmse, le, ue


def run_scrambling_experiments(pickle_file, dataset, base_output_dir):
    last_epoch = False

    for split_type in DATASETS_AND_SPLITS[dataset]:
        this_base_output_dir = os.path.join(base_output_dir, dataset, split_type)

        hparams, checkpoint_savepath = get_best_hparams(this_base_output_dir, last_epoch=last_epoch, quiet=True)
        try:
            model = load_checkpoint(hparams, checkpoint_savepath)
        except:
            print(
                f"scp -r euler:/cluster/project/schneider/cisert/bcpaff/processed_data/model_runs_esp/bcp/pde10a/{split_type}/{os.path.basename(os.path.dirname(checkpoint_savepath))} ./{split_type}"
            )
            continue

        train_idxs, _, _, test_idxs = get_split_idxs(dataset, split_type)

        scaler = QtaimScaler(
            pickle_file, train_idxs, ncp_graph=hparams["ncp_graph"]
        )  # still use train set normalization values
        test_loader = get_data_loader(pickle_file, hparams, scaler, test_idxs, shuffle=False)

        # y_scrambling
        test_mae, test_rmse, test_loss, y_test, yhat_test = eval_loop(
            model, test_loader, torch.nn.MSELoss(), y_scrambling=True, input_scrambling=False
        )
        results = {
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_loss": test_loss,
            "y_test": y_test,
            "yhat_test": yhat_test,
            "hparams": hparams,
        }
        results_savepath = os.path.join(this_base_output_dir, "test_results_y_scrambling.pkl")
        with open(results_savepath, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved test set results to {results_savepath}")

        # input_scrambling
        test_mae, test_rmse, test_loss, y_test, yhat_test = eval_loop(
            model, test_loader, torch.nn.MSELoss(), y_scrambling=False, input_scrambling=True
        )
        results = {
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_loss": test_loss,
            "y_test": y_test,
            "yhat_test": yhat_test,
            "hparams": hparams,
        }
        results_savepath = os.path.join(this_base_output_dir, "test_results_input_scrambling.pkl")
        with open(results_savepath, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved test set results to {results_savepath}")


def collect_scrambling_results(base_output_dir, dataset):
    df_res = defaultdict(dict)
    mad_benchmark = pd.read_csv(os.path.join(ROOT_PATH, "notebooks", "collect_results", "mad_benchmark.csv"))
    for split_type in DATASETS_AND_SPLITS[dataset]:
        experiment = f"{dataset}_{split_type}"
        rmse, le, ue = get_rmse_bounds_for_one_exp(
            os.path.join(base_output_dir, dataset, split_type, "test_results.pkl")
        )
        rmse_y, le_y, ue_y = get_rmse_bounds_for_one_exp(
            os.path.join(base_output_dir, dataset, split_type, "test_results_y_scrambling.pkl")
        )
        rmse_input, le_input, ue_input = get_rmse_bounds_for_one_exp(
            os.path.join(base_output_dir, dataset, split_type, "test_results_input_scrambling.pkl")
        )

        mad_benchmark_rmse = mad_benchmark[
            (mad_benchmark.dataset == dataset) & (mad_benchmark.split_type == split_type)
        ].rmse.values[0]
        mad_benchmark_le = mad_benchmark[
            (mad_benchmark.dataset == dataset) & (mad_benchmark.split_type == split_type)
        ]["le"].values[0]
        mad_benchmark_ue = mad_benchmark[
            (mad_benchmark.dataset == dataset) & (mad_benchmark.split_type == split_type)
        ].ue.values[0]

        df_res[experiment] = {
            "rmse": rmse,
            "le": le,
            "ue": ue,
            "rmse_y": rmse_y,
            "le_y": le_y,
            "ue_y": ue_y,
            "rmse_input": rmse_input,
            "le_input": le_input,
            "ue_input": ue_input,
            "rmse_mad": mad_benchmark_rmse,
            "le_mad": mad_benchmark_le,
            "ue_mad": mad_benchmark_ue,
        }
    df_res = pd.DataFrame(df_res).T
    return df_res


def plot_scrambling_results(df_res, dataset, savepath):
    names = DATASETS_AND_SPLITS[dataset]
    names_replace = {
        "random": "Random",
        "temporal_2011": "Temp.\n2011",
        "temporal_2012": "Temp.\n2012",
        "temporal_2013": "Temp.\n2013",
        "aminohetaryl_c1_amide": "Binding\nmode 1",
        "c1_hetaryl_alkyl_c2_hetaryl": "Binding\nmode 2",
        "aryl_c1_amide_c2_hetaryl": "Binding\nmode 3",
    }
    names = [names_replace[n] for n in names]

    heights_bcpaff = df_res["rmse"]
    errorbars_bcpaff = np.array([df_res["le"].tolist(), df_res["ue"].tolist()], dtype="float")

    heights_y = df_res["rmse_y"]
    errorbars_y = np.array([df_res["le_y"].tolist(), df_res["ue_y"].tolist()], dtype="float")

    heights_input = df_res["rmse_input"]
    errorbars_input = np.array([df_res["le_input"].tolist(), df_res["ue_input"].tolist()], dtype="float")

    heights_mad = df_res["rmse_mad"]
    errorbars_mad = np.array([df_res["le_mad"].tolist(), df_res["ue_mad"].tolist()], dtype="float")

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    w = 0.15
    x = np.array(range(len(names)))
    ax.bar(x - 1.5 * w, heights_bcpaff, width=w, label="Normal network", color="orange", edgecolor="black")
    ax.errorbar(x - 1.5 * w, heights_bcpaff, yerr=errorbars_bcpaff, linestyle="", color="black")

    ax.bar(x - 0.5 * w, heights_y, width=w, label="Y scrambling", color="green", edgecolor="black")
    ax.errorbar(x - 0.5 * w, heights_y, yerr=errorbars_y, linestyle="", color="black")

    ax.bar(x + 0.5 * w, heights_input, width=w, label="Input scrambling", color="gray", edgecolor="black")
    ax.errorbar(x + 0.5 * w, heights_input, yerr=errorbars_input, linestyle="", color="black")

    ax.bar(x + 1.5 * w, heights_mad, width=w, label="MAD", color="white", edgecolor="black")
    ax.errorbar(x + 1.5 * w, heights_mad, yerr=errorbars_mad, linestyle="", color="black")

    ax.set_ylabel("Test set RMSE", fontsize=12)
    ax.set_xticks(x)
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.set_xticklabels(names, rotation=0, ha="center", fontsize=12)
    ax.legend(fontsize=12, ncol=4)
    ax.set_ylim([0, 2.0])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {savepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="pdbbind")
    parser.add_argument("--base_output_dir", type=str, default=None)
    parser.add_argument("--plot_savepath", type=str, default=None)
    args = parser.parse_args()

    run_scrambling_experiments(args.pickle_file, args.dataset, args.base_output_dir)
    df_res = collect_scrambling_results(args.base_output_dir, args.dataset)
    if args.plot_savepath is None:
        plot_savepath = os.path.join(args.base_output_dir, args.dataset, "scrambling_results.pdf")
    else:
        plot_savepath = args.plot_savepath
    plot_scrambling_results(df_res, args.dataset, plot_savepath)
