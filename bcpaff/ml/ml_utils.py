"""
© 2023, ETH Zurich
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from bcpaff.ml.net import EGNN, EGNN_NCP
from bcpaff.ml.net_utils import QtaimDataBCP, QtaimDataNCP, QtaimScaler
from bcpaff.utils import BASE_OUTPUT_DIR, HPARAMS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def run_id_to_hparams(run_id: str, hparam_file: Optional[str] = None):
    if hparam_file is not None:
        df = pd.read_csv(hparam_file)
    else:
        df = HPARAMS
    sub_df = df[df.run_id == run_id][HPARAM_KEYS]
    if len(sub_df) == 0:
        return None
    elif len(sub_df) == 1:
        return sub_df.iloc[0].to_dict()
    elif len(sub_df) > 1:
        raise ValueError(f">1 entry for {run_id}")


def hparams_to_run_id(hparams: dict, hparam_file: Optional[str] = None):
    if hparam_file is not None:
        df = pd.read_csv(hparam_file)
    else:
        df = HPARAMS
    sub_df = df.loc[(df[list(hparams)] == pd.Series(hparams)).all(axis=1)]
    if len(sub_df) == 0:
        return None
    elif len(sub_df) == 1:
        return sub_df.run_id.values[0]
    elif len(sub_df) > 1:
        raise ValueError(f">1 entry for {hparams}")


def get_output_dir(hparams: dict, base_output_dir: Optional[str] = None):
    if base_output_dir is None:
        base_output_dir = BASE_OUTPUT_DIR
    output_dir = os.path.join(base_output_dir, "_".join([f"{key}{val}" for key, val in sorted(hparams.items())]))
    return output_dir


def save_checkpoint(
    model: EGNN, optimizer: torch.optim.Adam, epoch: int, savepath: str, datapoints: Optional[str] = None
):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    if datapoints is not None:
        checkpoint.update(datapoints)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    torch.save(checkpoint, savepath)


def load_checkpoint(hparams: dict, checkpoint_savepath: str):
    model = EGNN_NCP if hparams["ncp_graph"] else EGNN
    model = model(
        n_kernels=hparams["n_kernels"],
        aggr=hparams["aggr"],
        pool=hparams["pool"],
        mlp_dim=hparams["mlp_dim"],
        kernel_dim=hparams["kernel_dim"],
        baseline_atom_ids=hparams["baseline_atom_ids"],
        properties=hparams["properties"],
    ).to(DEVICE)
    checkpoint = torch.load(checkpoint_savepath, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    return model


def generate_scatter_plot(y_train: np.array, yhat_train: np.array, y_eval: np.array, yhat_eval: np.array):
    global_min = np.min(np.concatenate([y_train, yhat_train, y_eval, yhat_eval]))
    global_max = np.max(np.concatenate([y_train, yhat_train, y_eval, yhat_eval]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot([global_min, global_max], [global_min, global_max], color="black", linestyle="--", zorder=10)
    ax.scatter(y_train, yhat_train, color="blue", zorder=3)
    ax.scatter(y_eval, yhat_eval, color="orange", zorder=5)

    return fig


def get_data_loader(
    pickle_file: str,
    hparams: dict,
    scaler: QtaimScaler,
    idxs: List,
    shuffle: bool = True,
    pickle_data: Optional[dict] = None,
) -> DataLoader:
    data = QtaimDataNCP if hparams["ncp_graph"] else QtaimDataBCP
    data = data(
        pickle_file,
        scaler=scaler,
        idxs=idxs,
        cutoff=hparams["cutoff"],
        baseline_atom_ids=hparams["baseline_atom_ids"],
        properties=hparams["properties"],
        pickle_data=pickle_data,
    )
    loader = DataLoader(data, batch_size=hparams["batch_size"], num_workers=0, shuffle=shuffle)
    return loader
