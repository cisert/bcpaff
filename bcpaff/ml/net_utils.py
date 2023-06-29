"""
© 2023, ETH Zurich
"""

import pickle

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Dataset
from torch_geometric.transforms import RadiusGraph
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from bcpaff.ml.generate_pickle import ESP_NAMES
from bcpaff.utils import ATOM_NEIGHBOR_IDS, DEFAULT_PROPS, DEFAULT_PROPS_CRITIC2, OTHER

EPS = 0.001  # small epsilon to avoid undefined logarithms

MIN_PERCENTILE = 1
MAX_PERCENTILE = 99
ELLIPTICITY_OFFSET = np.zeros(len(DEFAULT_PROPS))
ELLIPTICITY_OFFSET[DEFAULT_PROPS.index("ellipticity")] = 1e-5  # needed to scale ellipticity for NCPs
ATOM_TYPE_ID_PLACEHOLDER = -9999  # placeholder


class QtaimDataBCP(Dataset):
    def __init__(
        self,
        pickle_file,
        scaler,
        idxs=None,
        cutoff=6,
        baseline_atom_ids=False,
        properties="yyyyyyyyy",
        pickle_data=None,
    ):
        if pickle_data is None:
            with open(pickle_file, "rb") as handle:
                self.all_props = pickle.load(handle)
        else:
            self.all_props = pickle_data
        self.idxs = idxs
        if idxs is None:  # use all
            self.pdb_ids = sorted(self.all_props.keys())
        else:
            self.pdb_ids = [key for key in self.all_props.keys() if key in idxs]
        self.cutoff = cutoff
        self.radius_graph = RadiusGraph(r=cutoff)
        self.baseline_atom_ids = baseline_atom_ids  # not used here, just keeping track (filtering in network)
        self.properties = [x == "y" for x in properties]  # boolean mask for which properties to use (translate y/n)
        self.scaler = scaler
        graph_data = []
        print("Preparing graph data...", flush=True)
        successfully_processed_pdb_ids = []
        for pdb_id in tqdm(self.pdb_ids):  # precompute graph data
            this_graph_data = self._prepare_graph_data(pdb_id)
            if this_graph_data is not None:
                graph_data.append(this_graph_data)
                successfully_processed_pdb_ids.append(pdb_id)
        print(f"Successfully processed {len(graph_data)}/{len(self.pdb_ids)} graphs.", flush=True)
        self.pdb_ids = successfully_processed_pdb_ids
        self.graph_data = graph_data

    def _prepare_graph_data(self, pdb_id):
        G = self.all_props[pdb_id]

        # add edges based on self.cutoff
        bcp_coords = list(nx.get_node_attributes(G, "node_coords").values())
        if len(bcp_coords):
            bcp_coords = torch.stack(list(nx.get_node_attributes(G, "node_coords").values()))
        else:  # no BCPs that fulfil the filters (e.g., only short interactions etc.)
            return None
        distance_matrix = squareform(pdist(bcp_coords))
        edge_idxs = np.stack(np.where(distance_matrix < self.cutoff))
        self_loops = edge_idxs[0, :] == edge_idxs[1, :]
        edge_idxs = edge_idxs[:, ~self_loops]
        G.add_edges_from(
            np.array(G.nodes)[edge_idxs].T
        )  # edge_idxs where only in terms of 0 ... n, but node ids in the graph are different

        untransformed_node_props = torch.stack(list(nx.get_node_attributes(G, "node_props").values()))

        transformed_node_props = self.scaler.transform(untransformed_node_props, point_name="bond_critical_point")[
            :, self.properties
        ]
        nx.set_node_attributes(
            G, {key: val for key, val in zip(G.nodes, torch.FloatTensor(transformed_node_props))}, name="node_props"
        )
        graph_data = from_networkx(G)
        if sum(self.properties) == 1:  # single property
            graph_data["node_props"] = graph_data["node_props"].unsqueeze(dim=1)
        else:
            graph_data["node_props"] = torch.stack(graph_data["node_props"])
        graph_data["node_coords"] = torch.stack(graph_data["node_coords"])
        graph_data["target"] = G.graph["target"]

        return graph_data

    def __getitem__(self, idx):
        return self.graph_data[idx]

    def __len__(self):
        return len(self.pdb_ids)


class QtaimDataNCP(Dataset):
    def __init__(
        self,
        pickle_file,
        scaler,
        idxs=None,
        cutoff=6,
        baseline_atom_ids=False,
        properties="yyyyyyyyy",
        pickle_data=None,
    ):
        if pickle_data is None:
            with open(pickle_file, "rb") as handle:
                self.all_props = pickle.load(handle)
        else:
            self.all_props = pickle_data
        self.idxs = idxs
        if idxs is None:  # use all
            self.pdb_ids = sorted(self.all_props.keys())
        else:
            self.pdb_ids = [key for key in self.all_props.keys() if key in idxs]
        self.cutoff = cutoff
        self.radius_graph = RadiusGraph(r=cutoff)
        self.baseline_atom_ids = baseline_atom_ids  # not used here, just keeping track (filtering in network)
        self.properties = [x == "y" for x in properties]  # boolean mask for which properties to use (translate y/n)
        self.include_esp = any(self.properties[-3:])
        self.prop_list = DEFAULT_PROPS + ESP_NAMES if self.include_esp else DEFAULT_PROPS
        self.null_props = torch.FloatTensor([0.0] * len(self.properties))

        self.scaler = scaler
        graph_data = []
        print("Preparing graph data...")
        for pdb_id in tqdm(self.pdb_ids):  # precompute graph data
            graph_data.append(self._prepare_graph_data(pdb_id))
        self.graph_data = graph_data

    def _prepare_graph_data(self, pdb_id):
        G = self.all_props[pdb_id]

        # remove edges with NULL_PROPS & distance > self.cutoff
        edge_distances = torch.FloatTensor(list(nx.get_edge_attributes(G, "distance").values()))
        edge_props_for_checking = torch.stack(list(nx.get_edge_attributes(G, "edge_props").values()))
        null_props_mask = (edge_props_for_checking == self.null_props).all(axis=1)
        remove_edges = null_props_mask & (edge_distances > self.cutoff)
        G.remove_edges_from(np.asarray(list(G.edges))[remove_edges])

        untransformed_edge_props = torch.stack(list(nx.get_edge_attributes(G, "edge_props").values()))
        untransformed_node_props = torch.stack(list(nx.get_node_attributes(G, "node_props").values()))

        transformed_node_props = self.scaler.transform(untransformed_node_props, point_name="nucleus_critical_point")[
            :, self.properties
        ]
        transformed_edge_props = self.scaler.transform(untransformed_edge_props, point_name="bond_critical_point")[
            :, self.properties
        ]
        nx.set_node_attributes(
            G, {key: val for key, val in zip(G.nodes, torch.FloatTensor(transformed_node_props))}, name="node_props"
        )
        nx.set_edge_attributes(
            G, {key: val for key, val in zip(G.edges, torch.FloatTensor(transformed_edge_props))}, name="edge_props"
        )
        graph_data = from_networkx(G)
        if sum(self.properties) == 1:  # maintain same dimension if single QM property is used
            graph_data["edge_props"] = torch.stack(list(graph_data["edge_props"])).unsqueeze(dim=1)
            graph_data["node_props"] = torch.stack(list(graph_data["node_props"])).unsqueeze(dim=1)
        else:
            graph_data["edge_props"] = torch.stack(list(graph_data["edge_props"]))
            graph_data["node_props"] = torch.stack(list(graph_data["node_props"]))
        graph_data["node_coords"] = torch.stack(graph_data["node_coords"])
        graph_data["edge_coords"] = torch.stack(graph_data["edge_coords"])
        graph_data["target"] = G.graph["target"]

        return graph_data

    def __getitem__(self, idx):
        return self.graph_data[idx]

    def __len__(self):
        return len(self.pdb_ids)


def pickle_to_df_ncp(pickle_file):
    with open(pickle_file, "rb") as handle:
        pickle_data = pickle.load(handle)
    all_ids, all_coords, all_props, all_targets, all_point_names = ([], [], [], [], [])
    all_atom_type_ids, all_is_ligand = [], []
    for pdb_id, G in pickle_data.items():
        # NCP data
        ncp_coords = list(nx.get_node_attributes(G, "node_coords").values())
        ncp_features = nx.get_node_attributes(G, "node_props").values()
        ncp_atom_type_ids = nx.get_node_attributes(G, "atom_type_id").values()

        # BCP data
        bcp_coords = nx.get_edge_attributes(G, "edge_coords").values()
        bcp_features = nx.get_edge_attributes(G, "edge_props").values()
        bcp_atom_type_ids = [ATOM_TYPE_ID_PLACEHOLDER] * len(bcp_features)  # dummy

        target = G.graph["target"]
        cp_names = len(ncp_coords) * ["nucleus_critical_point"] + len(bcp_coords) * ["bond_critical_point"]
        num_points = len(ncp_coords) + len(bcp_coords)
        all_ids.extend([pdb_id] * num_points)
        all_coords.extend(ncp_coords)
        all_coords.extend(bcp_coords)
        all_props.extend(ncp_features)
        all_props.extend(bcp_features)
        all_atom_type_ids.extend(ncp_atom_type_ids)
        all_atom_type_ids.extend(bcp_atom_type_ids)
        all_targets.extend([target] * num_points)
        all_point_names.extend(cp_names)
        all_is_ligand.extend(nx.get_node_attributes(G, "is_ligand").values())
        all_is_ligand.extend([False] * len(bcp_coords))  # only labelling NCPs as belonging to ligand
    all_ids = np.array(all_ids)
    all_targets = np.array(all_targets)
    all_coords = np.stack(all_coords)
    all_props = np.stack(all_props)
    all_atom_type_ids = np.stack(all_atom_type_ids).astype(int)
    if all_props.shape[1] == len(DEFAULT_PROPS):
        prop_list = DEFAULT_PROPS
    elif all_props.shape[1] == len(DEFAULT_PROPS) + len(ESP_NAMES):
        prop_list = DEFAULT_PROPS + ESP_NAMES
    elif all_props.shape[1] == len(DEFAULT_PROPS_CRITIC2):
        prop_list = DEFAULT_PROPS_CRITIC2
    dict_for_df = {
        "pdb_id": all_ids,
        "cp_name": all_point_names,
        "atom_type_id": all_atom_type_ids,
        "is_ligand": all_is_ligand,
        "target": all_targets,
        "x": all_coords[:, 0],
        "y": all_coords[:, 1],
        "z": all_coords[:, 2],
    }
    update_dict = {prop_name: all_props[:, i] for i, prop_name in enumerate(prop_list)}
    dict_for_df.update(update_dict)
    if len(set([len(x) for x in dict_for_df.values()])) != 1:
        print("You probably supplied the BCP pickle to the NCP function...")  # catching common error
    df = pd.DataFrame(dict_for_df)
    return df, pickle_data


def pickle_to_df_bcp(pickle_file):
    with open(pickle_file, "rb") as handle:
        pickle_data = pickle.load(handle)
    all_ids, all_coords, all_props, all_targets, all_point_names = ([], [], [], [], [])
    all_atom_type_ids, all_is_ligand = [], []
    for pdb_id, G in pickle_data.items():
        # BCP data
        bcp_coords = list(nx.get_node_attributes(G, "node_coords").values())
        bcp_features = nx.get_node_attributes(G, "node_props").values()
        bcp_atom_type_ids = nx.get_node_attributes(G, "atom_type_id").values()

        target = G.graph["target"]
        cp_names = len(bcp_coords) * ["bond_critical_point"]  # only doing BCPs in this function
        num_points = len(bcp_coords)
        all_ids.extend([pdb_id] * num_points)
        all_coords.extend(bcp_coords)
        all_props.extend(bcp_features)
        all_atom_type_ids.extend(bcp_atom_type_ids)
        all_targets.extend([target] * num_points)
        all_point_names.extend(cp_names)
    all_ids = np.array(all_ids)
    all_targets = np.array(all_targets)
    all_coords = np.stack(all_coords)
    all_props = np.stack(all_props)
    dict_for_df = {
        "pdb_id": all_ids,
        "cp_name": all_point_names,
        "atom_type_id": all_atom_type_ids,
        "target": all_targets,
        "x": all_coords[:, 0],
        "y": all_coords[:, 1],
        "z": all_coords[:, 2],
    }
    if all_props.shape[1] == len(DEFAULT_PROPS):
        prop_list = DEFAULT_PROPS
    elif all_props.shape[1] == len(DEFAULT_PROPS) + len(ESP_NAMES):
        prop_list = DEFAULT_PROPS + ESP_NAMES
    elif all_props.shape[1] == len(DEFAULT_PROPS_CRITIC2):
        prop_list = DEFAULT_PROPS_CRITIC2
    update_dict = {prop_name: all_props[:, i] for i, prop_name in enumerate(prop_list)}
    dict_for_df.update(update_dict)
    df = pd.DataFrame(dict_for_df)
    return df, pickle_data


def pickle_to_df(pickle_file, ncp_graph=False):
    if ncp_graph:
        return pickle_to_df_ncp(pickle_file)
    else:
        return pickle_to_df_bcp(pickle_file)


class QtaimScaler(object):
    def __init__(self, pickle_file, train_idxs, ncp_graph=False):
        self.pickle_file = pickle_file
        self.train_idxs = train_idxs
        self.ncp_graph = ncp_graph
        df, self.pickle_data = pickle_to_df(pickle_file, ncp_graph=self.ncp_graph)
        self.df = df[df.pdb_id.isin(train_idxs)]  # only fit scaler to training data
        self.prop_list = self._get_prop_list()
        self.ellipticity_offset = self._get_ellipticity_offset()
        self.min_percentiles = {}
        self.max_percentiles = {}
        self.scaled_abs_min_val = {}
        self.minmax_scaler = {}
        self.fit()

    def _get_ellipticity_offset(self):
        if self.prop_list == DEFAULT_PROPS_CRITIC2:
            return 0
        else:
            if "esp" in self.prop_list:
                return np.hstack([ELLIPTICITY_OFFSET, [0.0] * len(ESP_NAMES)])
            else:
                return ELLIPTICITY_OFFSET

    def _get_prop_list(self):
        prop_list = [x for x in DEFAULT_PROPS + ESP_NAMES if x in self.df.keys()]
        return prop_list

    def transform(self, props_unscaled, point_name):
        # clip anything above/below 1st/99th percentile
        props_scaled = np.clip(props_unscaled, self.min_percentiles[point_name], self.max_percentiles[point_name])
        # move minimum to zero and add small amount (EPS) so all positive (needed for log)
        props_scaled = (
            props_scaled + ((1 + EPS) * self.scaled_abs_min_val[point_name]).to_numpy() + self.ellipticity_offset
        )
        # take log10
        props_scaled = np.log10(props_scaled)
        # scale from 0 to 1
        props_scaled = self.minmax_scaler[point_name].transform(props_scaled)
        return props_scaled

    def fit(self):
        cp_names = ["bond_critical_point", "nucleus_critical_point"]
        for cp_name in cp_names:
            sub_df = self.df[self.df.cp_name == cp_name]  #  separate scaling for BCPs and NCPs
            if len(sub_df) == 0:  # don't have any points of this type --> no need to scale
                pass
            else:
                if cp_name == "bond_critical_point":
                    sub_df = sub_df[(sub_df[self.prop_list] != 0).all(axis=1)]  # don't scale for NULL_PROPS
                # find 1st/99st percentiles for each property
                self.min_percentiles[cp_name] = np.percentile(sub_df[self.prop_list], MIN_PERCENTILE, axis=0)
                self.max_percentiles[cp_name] = np.percentile(sub_df[self.prop_list], MAX_PERCENTILE, axis=0)
                # clip anything above/below 1st/99th percentile
                props_scaled = np.clip(
                    sub_df[self.prop_list], self.min_percentiles[cp_name], self.max_percentiles[cp_name]
                )
                # move minimum to zero and add small amount (EPS) so all positive (needed for log)
                self.scaled_abs_min_val[cp_name] = abs(props_scaled.min())

                props_scaled = props_scaled + (1 + EPS) * self.scaled_abs_min_val[cp_name] + self.ellipticity_offset
                # take log10
                props_scaled = np.log10(props_scaled)
                # scale to 0 mean and unit variance
                self.minmax_scaler[cp_name] = MinMaxScaler().fit(props_scaled.values)
