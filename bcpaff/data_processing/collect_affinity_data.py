"""
© 2023, ETH Zurich
"""

import os

import pandas as pd

from bcpaff.utils import DATA_PATH

SPLIT_ASSIGNMENTS_BASEPATH = os.path.join(DATA_PATH, "pdbbind", "pdb_ids")


def collect_affinity_as_dataframe(affinity_file: str) -> pd.DataFrame:
    with open(affinity_file, "r") as f:
        lines = f.readlines()[6:]
    info = []
    for line in lines:
        tokens = line.split()
        pdb_id = tokens[0]
        act = float(tokens[3])
        info.append([pdb_id, act])
    df = pd.DataFrame(info, columns=["pdb_id", "aff"])
    return df


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    split_names = ["training_set", "core_set", "validation_set", "hold_out_set"]
    split_assignments = {}
    for split_name in split_names:
        split_df = pd.read_csv(os.path.join(SPLIT_ASSIGNMENTS_BASEPATH, f"{split_name}.csv"), names=["pdb_id"])
        for pdb_id in split_df.pdb_id.tolist():
            assert pdb_id not in split_assignments
            split_assignments[pdb_id] = split_name

    df.loc[:, "random"] = df.pdb_id.apply(
        lambda x: split_assignments[x] if x in split_assignments else "no_assignment"
    )
    df = df[df.random != "no_assignment"].reset_index(drop=True)
    return df


def collect_pdb_affinity_data():
    pdbbind_structure_path = os.path.join(DATA_PATH, "pdbbind")
    pdbbind_csv = os.path.join(
        pdbbind_structure_path, "PDBbind_v2019_plain_text_index/plain-text-index/index/INDEX_general_PL_data.2019"
    )
    df = collect_affinity_as_dataframe(pdbbind_csv)
    df = assign_splits(df)
    df.to_csv(os.path.join(pdbbind_structure_path, "pdbbind2019_affinity.csv"), index=False)


if __name__ == "__main__":
    collect_pdb_affinity_data()
