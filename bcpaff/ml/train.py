"""
© 2023, ETH Zurich
"""

import argparse
import json
import os
import random
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from bcpaff.ml.ml_utils import get_data_loader, hparams_to_run_id, save_checkpoint
from bcpaff.ml.net import EGNN, EGNN_NCP, EGNNAtt
from bcpaff.ml.net_utils import QtaimScaler
from bcpaff.utils import DATA_PATH, SEED

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(
    model: Union[EGNN, EGNN_NCP, EGNNAtt], loader: DataLoader, optimizer: torch.optim.Adam, criterion: torch.nn.MSELoss
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.train()
    training_loss = []
    targets = []
    predictions = []

    # for g_batch in tqdm(loader, total=len(loader)):
    for g_batch in loader:
        g_batch = g_batch.to(DEVICE)
        optimizer.zero_grad()

        prediction = model(g_batch).squeeze(axis=1)
        target = g_batch.target

        loss = criterion(prediction, target.float())
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())
        predictions.append(prediction.detach().cpu().numpy())

        targets.append(target.detach().cpu().numpy())

    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    train_mae = np.mean(np.abs(targets - predictions))
    train_rmse = mean_squared_error(targets, predictions, squared=False)

    return (
        train_mae,
        train_rmse,
        np.mean(training_loss, axis=0),
        targets,
        predictions,
    )


def eval_loop(
    model: Union[EGNN, EGNN_NCP, EGNNAtt],
    loader: DataLoader,
    criterion: torch.nn.MSELoss,
    y_scrambling: bool = False,
    input_scrambling: bool = False,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    eval_loss = []
    targets = []
    predictions = []

    with torch.no_grad():
        # for g_batch in tqdm(loader, total=len(loader)):
        for g_batch in loader:
            g_batch = g_batch.to(DEVICE)

            prediction = model(g_batch, input_scrambling=input_scrambling).squeeze(axis=1)
            target = g_batch.target
            if y_scrambling:
                target = target[torch.randperm(target.shape[0])]
            loss = criterion(prediction, target.float())

            eval_loss.append(loss.item())
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())

    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    eval_mae = np.mean(np.abs(targets - predictions))
    eval_rmse = mean_squared_error(targets, predictions, squared=False)

    return (
        eval_mae,
        eval_rmse,
        np.mean(eval_loss, axis=0),
        targets,
        predictions,
    )


def get_split_idxs(dataset: str, split_type: str) -> Tuple[List[str],]:
    split_col = f"{split_type}_split"
    if dataset.startswith("pdbbind"):  # allowed split types: random, carbonic_anhydrase_2 (core_set not available)
        split_assignment_df = pd.read_csv(os.path.join(DATA_PATH, "pdbbind", "pdbbind2019_affinity.csv"))
        train_idxs = split_assignment_df[split_assignment_df[split_type] == "training_set"].pdb_id.tolist()
        eval_idxs = split_assignment_df[split_assignment_df[split_type] == "validation_set"].pdb_id.tolist()
        core_idxs = split_assignment_df[split_assignment_df[split_type] == "core_set"].pdb_id.tolist()
        test_idxs = split_assignment_df[split_assignment_df[split_type] == "hold_out_set"].pdb_id.tolist()
    elif dataset == "pde10a":
        split_assignment_df = pd.read_csv(os.path.join(DATA_PATH, "pde10a", "10822_2022_478_MOESM2_ESM.csv"))
        train_idxs = split_assignment_df[split_assignment_df[split_col] == "train"].docking_folder.tolist()
        eval_idxs = split_assignment_df[split_assignment_df[split_col] == "val"].docking_folder.tolist()
        test_idxs = split_assignment_df[split_assignment_df[split_col] == "test"].docking_folder.tolist()
        core_idxs = None  # not applicable for this dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train_idxs, eval_idxs, core_idxs, test_idxs


def run_training(
    hparams: dict,
    hparam_file: str,
    pickle_file: str,
    dataset: str,
    split_type: str,
    base_output_dir: str,
    overwrite: bool = False,
    no_lr_decay: bool = False,
    y_scrambling: bool = False,
    num_epochs: int = 300,
):
    if not os.path.exists(pickle_file):
        raise ValueError(f"pickle_file {pickle_file} missing")

    run_id = hparams_to_run_id(hparams, hparam_file=hparam_file)

    output_dir = os.path.join(base_output_dir, str(run_id))

    checkpoint_savepath = os.path.join(output_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_savepath) and not overwrite:
        raise ValueError(f"Checkpoint {checkpoint_savepath} already exists and overwrite = False.")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    print(output_dir)
    print("=================================")
    print(f"DEVICE = {DEVICE}")
    print(f"tensorboard --logdir {output_dir}")
    print("=================================", flush=True)

    train_idxs, eval_idxs, _, _ = get_split_idxs(dataset, split_type)

    scaler = QtaimScaler(pickle_file, train_idxs, ncp_graph=hparams["ncp_graph"])

    train_loader = get_data_loader(
        pickle_file, hparams, scaler, train_idxs, shuffle=True, pickle_data=scaler.pickle_data
    )
    print("Got train_loader", flush=True)
    eval_loader = get_data_loader(
        pickle_file, hparams, scaler, eval_idxs, shuffle=False, pickle_data=scaler.pickle_data
    )
    print("Got eval_loader", flush=True)

    if hparams["ncp_graph"]:
        model = EGNN_NCP
    else:
        if hparams["pool"].startswith("att"):
            model = EGNNAtt
        else:
            model = EGNN
    model = model(
        n_kernels=hparams["n_kernels"],
        aggr=hparams["aggr"],
        pool=hparams["pool"],
        mlp_dim=hparams["mlp_dim"],
        kernel_dim=hparams["kernel_dim"],
        baseline_atom_ids=hparams["baseline_atom_ids"],
        properties=hparams["properties"],
    )
    print("Got model", flush=True)
    model = model.to(DEVICE)
    print("Put model in device", flush=True)

    print(model, flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-10)

    if not no_lr_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=20, verbose=True
        )

    min_mae = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}", flush=True)

        train_mae, train_rmse, train_loss, y_train, yhat_train = train_loop(
            model, train_loader, optimizer, torch.nn.MSELoss()
        )
        writer.add_scalars("loss", {"train": train_loss}, global_step=epoch)
        writer.add_scalars("mae", {"train": train_mae}, global_step=epoch)
        writer.add_scalars("rmse", {"train": train_rmse}, global_step=epoch)
        if epoch % 1 == 0:
            eval_mae, eval_rmse, eval_loss, y_eval, yhat_eval = eval_loop(
                model, eval_loader, torch.nn.MSELoss(), y_scrambling=y_scrambling
            )
            # fig = generate_scatter_plot(y_train, yhat_train, y_eval, yhat_eval)
            # writer.add_figure("scatter_plot", fig, global_step=epoch)
            writer.add_scalars("loss", {"eval": eval_loss}, global_step=epoch)
            writer.add_scalars("mae", {"eval": eval_mae}, global_step=epoch)
            writer.add_scalars("rmse", {"eval": eval_rmse}, global_step=epoch)
            datapoints = {
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "y_train": y_train,
                "yhat_train": yhat_train,
                "y_eval": y_eval,
                "yhat_eval": yhat_eval,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "eval_mae": eval_mae,
                "eval_rmse": eval_rmse,
            }
            if not no_lr_decay:
                scheduler.step(eval_mae)

            if eval_mae < min_mae:
                min_mae = eval_mae
                print(f"New min eval_mae in epoch {epoch}: {eval_mae:.6f}", flush=True)
                save_checkpoint(model, optimizer, epoch, checkpoint_savepath, datapoints=datapoints)
            if epoch == num_epochs - 1:
                print(f"Final epoch {epoch} eval_mae: {eval_mae:.6f}", flush=True)
                checkpoint_savepath = os.path.join(output_dir, "last_epoch_checkpoint.pt")
                save_checkpoint(model, optimizer, epoch, checkpoint_savepath, datapoints=datapoints)
    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file", type=str, required=True)
    parser.add_argument("--hparam_file", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="pdbbind")
    parser.add_argument("--split_type", type=str, default="random")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--base_output_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--no_lr_decay", action="store_true", default=False)
    parser.add_argument("--y_scrambling", action="store_true", default=False)
    parser.add_argument("--num_epochs", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        hparams = json.load(f)

    run_training(
        hparams,
        args.hparam_file,
        args.pickle_file,
        args.dataset,
        args.split_type,
        args.base_output_dir,
        overwrite=args.overwrite,
        no_lr_decay=args.no_lr_decay,
        y_scrambling=args.y_scrambling,
        num_epochs=args.num_epochs,
    )
