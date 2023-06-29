"""
© 2023, ETH Zurich
"""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from bcpaff.ml import statsig
from bcpaff.ml.generate_pickle import ESP_NAMES
from bcpaff.ml.test import datasets_and_splits
from bcpaff.utils import DATA_PATH, DEFAULT_PROPS, PROCESSED_DATA_PATH

ANALYSIS_SAVEPATH = os.path.join(PROCESSED_DATA_PATH, "analysis")


def get_mad_val(dataset: str, split_type: str) -> Tuple[float, float, float]:
    if dataset == "pdbbind":
        split_assignment_df = pd.read_csv(os.path.join(DATA_PATH, "pdbbind", "pdbbind2019_affinity.csv"))
        y_mad_val = split_assignment_df[split_assignment_df["split"] == "validation_set"].aff.values
        yhat_mad_val = split_assignment_df[split_assignment_df["split"] == "training_set"].aff.mean() * np.ones(
            y_mad_val.shape
        )
    elif dataset == "pde10a":
        split_assignment_df = pd.read_csv(os.path.join(DATA_PATH, "pde10a", "10822_2022_478_MOESM2_ESM.csv"))
        split_type = split_type + "_split"
        y_mad_val = split_assignment_df[split_assignment_df[split_type] == "val"].pic50.values
        yhat_mad_val = split_assignment_df[split_assignment_df[split_type] == "train"].pic50.mean() * np.ones(
            y_mad_val.shape
        )
    rmse_mad_val, le_mad_val, ue_mad_val = statsig.rmse(y_mad_val, yhat_mad_val)
    return rmse_mad_val, le_mad_val, ue_mad_val


def plot_hparam_results(
    base_output_dir: str,
    dataset: str,
    split_type: str,
    savepath: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    top_n: int = 50,
):
    if df is None:
        df = pd.read_csv(os.path.join(base_output_dir, "hparam_results.csv"))

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    sub_df = df.sort_values(by="eval_rmse")[:top_n]
    colors = []
    for _, row in sub_df.iterrows():
        if not row.baseline_atom_ids:
            colors.append("white")
        else:
            if row.properties == "n" * (len(DEFAULT_PROPS) + len(ESP_NAMES)):
                colors.append("green")
            else:
                colors.append("orange")
    x = range(len(sub_df))
    heights = sub_df.eval_rmse.tolist()
    ax.scatter(x, heights, color=colors, marker="D", s=100, zorder=5, edgecolor="black")
    errorbars = np.array([sub_df.rmse_le.tolist(), sub_df.rmse_ue.tolist()], dtype="float")
    ax.errorbar(x, heights, yerr=errorbars, linestyle="", color="black", zorder=4)
    ax.set_ylim(0.7, 1.8)
    ax.set_xlim(-1, len(sub_df))
    ax.set_xlabel("Experiments ranked by validation set RMSE", fontsize=16)
    ax.set_ylabel("Validation set RMSE", fontsize=16)
    ax.set_xticks([])
    ax.tick_params(axis="both", which="major", labelsize=14)
    custom_lines = [
        Line2D([], [], color="green", lw=4, marker="D", markersize=10, markeredgecolor="black", linestyle="None"),
        Line2D([], [], color="white", lw=4, marker="D", markersize=10, markeredgecolor="black", linestyle="None"),
        Line2D([], [], color="orange", lw=4, marker="D", markersize=10, markeredgecolor="black", linestyle="None"),
        Line2D([0], [0], color="black", linestyle="--", lw=1),
    ]
    labels = ["Atom-IDs", "BCPs", "Atom-IDs & \nBCPs", "MAD"]
    rmse_mad_val, le_mad_val, ue_mad_val = get_mad_val(dataset, split_type)
    ax.plot([min(x) - 10, max(x) + 10], [rmse_mad_val, rmse_mad_val], color="black", linestyle="--")
    ax.fill_between(
        [min(x) - 10, max(x) + 10],
        [rmse_mad_val - le_mad_val, rmse_mad_val - le_mad_val],
        [rmse_mad_val + ue_mad_val, rmse_mad_val + ue_mad_val],
        color="gray",
        alpha=0.2,
    )
    ax.legend(
        custom_lines,
        labels,
        fontsize=16,
        ncol=1,
        bbox_to_anchor=(1, 0.5),
        bbox_transform=ax.transAxes,
        loc="center left",
    )
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")


def plot_hparam_results_esp(
    base_output_dir: str,
    dataset: str,
    split_type: str,
    savepath: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    top_n: int = 50,
):
    if df is None:
        df = pd.read_csv(os.path.join(base_output_dir, "hparam_results.csv"))

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    sub_df = df.sort_values(by="eval_rmse")[:top_n]
    colors = []
    h_esp = []
    for _, row in sub_df.iterrows():
        if not row.baseline_atom_ids:  # only QM props
            colors.append("white")
        else:  # baseline_atom_ids + QM props
            if row.properties == "n" * (len(DEFAULT_PROPS) + len(ESP_NAMES)):  # no props
                colors.append("green")
            else:  # some QM props
                colors.append("orange")

        h_esp.append(True) if "y" in row.properties[-3:] else h_esp.append(False)

    x = range(len(sub_df))
    heights = sub_df.eval_rmse.tolist()
    for this_x, height, color, esp in zip(x, heights, colors, h_esp):
        ax.scatter(this_x, height, color=color, marker="D", s=100, zorder=5, edgecolor="black")
        if esp:
            ax.scatter(this_x, 1.75, color="black", marker="*", s=30)
    errorbars = np.array([sub_df.rmse_le.tolist(), sub_df.rmse_ue.tolist()], dtype="float")
    ax.errorbar(x, heights, yerr=errorbars, linestyle="", color="black", zorder=4)
    ax.set_ylim(0.7, 1.8)
    ax.set_xlim(-1, len(sub_df))
    ax.set_xlabel("Experiments ranked by validation set RMSE", fontsize=16)
    ax.set_ylabel("Validation set RMSE", fontsize=16)
    ax.set_xticks([])
    ax.tick_params(axis="both", which="major", labelsize=14)
    custom_lines = [
        Line2D([], [], color="green", lw=4, marker="D", markersize=10, markeredgecolor="black", linestyle="None"),
        Line2D([], [], color="white", lw=4, marker="D", markersize=10, markeredgecolor="black", linestyle="None"),
        Line2D([], [], color="orange", lw=4, marker="D", markersize=10, markeredgecolor="black", linestyle="None"),
        Line2D([0], [0], color="black", linestyle="--", lw=1),
    ]
    labels = ["Atom-IDs", "BCPs", "Atom-IDs & \nBCPs", "MAD"]
    rmse_mad_val, le_mad_val, ue_mad_val = get_mad_val(dataset, split_type)
    ax.plot([min(x) - 10, max(x) + 10], [rmse_mad_val, rmse_mad_val], color="black", linestyle="--")
    ax.fill_between(
        [min(x) - 10, max(x) + 10],
        [rmse_mad_val - le_mad_val, rmse_mad_val - le_mad_val],
        [rmse_mad_val + ue_mad_val, rmse_mad_val + ue_mad_val],
        color="gray",
        alpha=0.2,
    )
    ax.legend(
        custom_lines,
        labels,
        fontsize=16,
        ncol=1,
        bbox_to_anchor=(1, 0.5),
        bbox_transform=ax.transAxes,
        loc="center left",
    )
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # hparam visualization
    dataset = "pde10a"
    for ncp_bcp in ["bcp", "ncp"]:
        for split_type in datasets_and_splits[dataset]:
            base_output_dir = os.path.join(PROCESSED_DATA_PATH, "model_runs_esp", ncp_bcp, dataset, split_type)
            savepath = os.path.join(
                ANALYSIS_SAVEPATH, "hparam_visualization_esp", f"{dataset}_{split_type}_{ncp_bcp}.png"
            )
            plot_hparam_results_esp(
                base_output_dir, dataset, split_type, savepath=savepath, df=None, top_n=99999
            )  # plot all
