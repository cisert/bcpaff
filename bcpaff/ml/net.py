"""
© 2023, ETH Zurich
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.glob.gmt import GraphMultisetTransformer
from torch_geometric.typing import Adj, Size, Tensor
from torch_scatter import scatter_mean, scatter_sum

from bcpaff.utils import ATOM_NEIGHBOR_IDS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_NEIGHBORS = 2


class Embed(nn.Module):
    def __init__(self, baseline_atom_ids, properties, kernel_dim, input_dim):
        super(Embed, self).__init__()
        self.baseline_atom_ids = baseline_atom_ids
        self.properties = properties
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim
        self.atom_embedding = nn.Embedding(num_embeddings=len(ATOM_NEIGHBOR_IDS) + 1, embedding_dim=self.kernel_dim)
        if self.input_dim > 0:  # not needed if we don't process QM features
            self.prop_embedding = nn.Linear(self.input_dim, self.kernel_dim)
            self.compress = nn.Linear(self.kernel_dim * 2, self.kernel_dim)
            self.weight = torch.cat([self.atom_embedding.weight, self.prop_embedding.weight.T])
        else:
            self.weight = self.atom_embedding.weight

    def forward(self, node_props, atom_ids):
        if self.baseline_atom_ids and any(self.properties):
            atom_out = self.atom_embedding(atom_ids[:, 0]) + self.atom_embedding(atom_ids[:, 1])
            # sum up embeddings of both neighbors
            prop_out = self.prop_embedding(node_props)
            concatenated = torch.cat([atom_out, prop_out], dim=1)  # concat along feature dimension
            compressed = self.compress(concatenated)  # compress to get to kernel_dim
            return compressed
        elif self.baseline_atom_ids:
            atom_out = self.atom_embedding(atom_ids.to(torch.long))
            atom_out = atom_out.sum(dim=1)  # sum up embeddings of both neighbors
            return atom_out
        elif any(self.properties):
            prop_out = self.prop_embedding(node_props)
            return prop_out
        else:
            raise ValueError("Neither atom_ids nor properties specified")


class EmbedEdge(nn.Module):
    def __init__(self, properties, kernel_dim, input_dim):
        super(EmbedEdge, self).__init__()
        self.properties = properties
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim
        if self.input_dim > 0:
            self.prop_embedding = nn.Linear(self.input_dim, self.kernel_dim)
            self.weight = self.prop_embedding.weight

    def forward(self, input):
        if self.input_dim > 0:
            return self.prop_embedding(input)
        else:
            return input


class EmbedNCP(nn.Module):
    def __init__(self, baseline_atom_ids, properties, kernel_dim, input_dim):
        super(EmbedNCP, self).__init__()
        self.baseline_atom_ids = baseline_atom_ids
        self.properties = properties
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim
        self.atom_embedding = nn.Embedding(num_embeddings=len(ATOM_NEIGHBOR_IDS) + 1, embedding_dim=self.kernel_dim)
        if self.input_dim > 0:
            self.prop_embedding = nn.Linear(self.input_dim, self.kernel_dim)
            self.compress = nn.Linear(self.kernel_dim * 2, self.kernel_dim)
            self.weight = torch.cat([self.atom_embedding.weight, self.prop_embedding.weight.T])
        else:
            self.weight = self.atom_embedding.weight

    def forward(self, props, atom_type_ids):
        if self.baseline_atom_ids and any(self.properties):
            atom_out = self.atom_embedding(atom_type_ids)
            prop_out = self.prop_embedding(props)
            concatenated = torch.cat([atom_out, prop_out], dim=1)  # concat along feature dimension
            compressed = self.compress(concatenated)  # compress to get to kernel_dim
            return compressed
        elif self.baseline_atom_ids:
            atom_out = self.atom_embedding(atom_type_ids)
            # don't need to sum up neighbor embeddings because we're using the NCPs as nodes
            return atom_out
        elif any(self.properties):
            prop_out = self.prop_embedding(torch.Tensor.float(props))
            return prop_out
        else:
            raise ValueError("Neither atom_ids nor properties specified")


class EGNN_NCP(nn.Module):
    def __init__(
        self,
        n_kernels=5,
        n_mlp=3,
        mlp_dim=256,
        n_outputs=1,
        m_dim=32,
        initialize_weights=True,
        fourier_features=32,
        aggr="mean",
        pool="mean",
        kernel_dim=32,
        baseline_atom_ids=False,
        properties="yyyyyyyyy",
        save_latent_space=False,
    ):
        """Main Equivariant Graph Neural Network class.

        Parameters
        ----------
        embedding_dim : int, optional
            Embedding dimension, by default 128
        n_kernels : int, optional
            Number of message-passing rounds, by default 5
        n_mlp : int, optional
            Number of node-level and global-level MLPs, by default 3
        mlp_dim : int, optional
            Hidden size of the node-level and global-level MLPs, by default 256
        m_dim : int, optional
            Node-level hidden size, by default 32
        initialize_weights : bool, optional
            Whether to use Xavier init. for learnable weights, by default True
        fourier_features : int, optional
            Number of Fourier features to use, by default 32
        aggr : str, optional
            Node update function, by default "mean"
        pool: str, optional
            How to pool features for global tasks, by default "mean". Other option: "sum"
        baseline_atom_ids: bool, optional
            Whether to just use an embedding of the adjacent atom ids
        """
        super(EGNN_NCP, self).__init__()
        self.properties = [x == "y" for x in properties]  # boolean mask for which properties to use
        self.input_dim = sum(self.properties)
        self.pos_dim = 3  # XYZ coordinates
        self.m_dim = m_dim
        self.n_kernels = n_kernels
        self.n_mlp = n_mlp
        self.mlp_dim = mlp_dim
        self.n_outputs = n_outputs
        self.initialize_weights = initialize_weights
        self.fourier_features = fourier_features
        self.aggr = aggr  # message aggregation
        self.pool = pool
        self.kernel_dim = kernel_dim  # message passing size
        self.baseline_atom_ids = baseline_atom_ids
        self.save_latent_space = save_latent_space
        self.embedding = EmbedNCP(
            baseline_atom_ids=self.baseline_atom_ids,
            properties=self.properties,
            kernel_dim=self.kernel_dim,
            input_dim=self.input_dim,
        )
        if self.input_dim > 0:
            self.edge_embedding = EmbedEdge(
                properties=self.properties,
                kernel_dim=self.kernel_dim,
                input_dim=self.input_dim,
            )

        self.kernels = nn.ModuleList()
        edge_attr_dim = self.kernel_dim if self.input_dim > 0 else 0
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=self.kernel_dim,
                    pos_dim=self.pos_dim,
                    m_dim=self.m_dim,
                    fourier_features=self.fourier_features,
                    aggr=self.aggr,
                    edge_attr_dim=edge_attr_dim,
                )
            )

        # MLP 1
        self.fnn = nn.ModuleList()
        input_fnn = self.kernel_dim * (self.n_kernels + 1)
        self.fnn.append(nn.Linear(input_fnn, mlp_dim))
        for _ in range(self.n_mlp - 1):
            self.fnn.append(nn.Linear(self.mlp_dim, self.mlp_dim))
        self.fnn.append(nn.Linear(self.mlp_dim, self.n_outputs))

        # Initialize weights
        if self.initialize_weights:
            self.kernels.apply(weights_init)
            self.fnn.apply(weights_init)
            nn.init.xavier_uniform_(self.embedding.weight)

    def _set_save_latent_space(self, save_latent_space):
        self.save_latent_space = save_latent_space

    def forward(self, g_batch, input_scrambling=False):
        if input_scrambling:
            raise NotImplementedError
        # Embedding
        features = self.embedding(
            g_batch.node_props, g_batch.atom_type_id
        )  # num_nodes x kernel_dim (embedding_dim = kernel_dim)
        if self.input_dim > 0:
            edge_features = self.edge_embedding(torch.Tensor.float(g_batch.edge_props))
        else:
            edge_features = None
        coords = torch.Tensor.float(g_batch.node_coords)
        features = torch.cat((coords, features), dim=1)  # num_nodes x (3 + kernel_dim) (3 = XYZ)

        # Kernel
        feature_list = []
        feature_list.append(features[:, self.pos_dim :])

        for kernel in self.kernels:
            features = kernel(
                x=features, edge_index=g_batch.edge_index, edge_features=edge_features
            )  # num_nodes x (3 + kernel_dim)
            feature_list.append(features[:, self.pos_dim :])

        # Concat
        features = F.silu(
            torch.cat(feature_list, dim=1)
        )  # num_nodes x (self.n_kernels+1)*kernel_dim (+1 = initial features)

        # MLP 1 per node
        for mlp in self.fnn[:-1]:
            features = F.silu(mlp(features))  # num_nodes x mlp_dim

        if self.pool == "mean":
            features = scatter_mean(features, g_batch.batch, dim=0)
        elif self.pool == "sum":
            features = scatter_sum(features, g_batch.batch, dim=0)
        else:
            raise ValueError("Unknown pooling operation")
        # batch_size x mlp_dim
        if self.save_latent_space:
            return {"structure_ids": g_batch.pdb_id, "features": features}
        features = self.fnn[-1](features)  # batch_size x 1 (1 = output_dim)
        return features


class EGNN(nn.Module):
    def __init__(
        self,
        n_kernels=5,
        n_mlp=3,
        mlp_dim=256,
        n_outputs=1,
        m_dim=32,
        initialize_weights=True,
        fourier_features=32,
        aggr="mean",
        pool="mean",
        kernel_dim=32,
        baseline_atom_ids=False,
        properties="yyyyyyyyy",
        save_latent_space=False,
    ):
        """Main Equivariant Graph Neural Network class.

        Parameters
        ----------
        embedding_dim : int, optional
            Embedding dimension, by default 128
        n_kernels : int, optional
            Number of message-passing rounds, by default 5
        n_mlp : int, optional
            Number of node-level and global-level MLPs, by default 3
        mlp_dim : int, optional
            Hidden size of the node-level and global-level MLPs, by default 256
        m_dim : int, optional
            Node-level hidden size, by default 32
        initialize_weights : bool, optional
            Whether to use Xavier init. for learnable weights, by default True
        fourier_features : int, optional
            Number of Fourier features to use, by default 32
        aggr : str, optional
            Node update function, by default "mean"
        pool: str, optional
            How to pool features for global tasks, by default "mean". Other option: "sum"
        baseline_atom_ids: bool, optional
            Whether to just use an embedding of the adjacent atom ids
        """
        super(EGNN, self).__init__()
        self.properties = [x == "y" for x in properties]  # boolean mask for which properties to use
        self.input_dim = sum(self.properties)
        self.pos_dim = 3  # XYZ coordinates
        self.m_dim = m_dim
        self.n_kernels = n_kernels
        self.n_mlp = n_mlp
        self.mlp_dim = mlp_dim
        self.n_outputs = n_outputs
        self.initialize_weights = initialize_weights
        self.fourier_features = fourier_features
        self.aggr = aggr  # message aggregation
        self.pool = pool
        self.kernel_dim = kernel_dim  # message passing size
        self.baseline_atom_ids = baseline_atom_ids
        self.save_latent_space = save_latent_space
        self.embedding = Embed(
            baseline_atom_ids=self.baseline_atom_ids,
            properties=self.properties,
            kernel_dim=self.kernel_dim,
            input_dim=self.input_dim,
        )

        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=self.kernel_dim,
                    pos_dim=self.pos_dim,
                    m_dim=self.m_dim,
                    fourier_features=self.fourier_features,
                    aggr=self.aggr,
                )
            )

        # MLP 1
        self.fnn = nn.ModuleList()
        input_fnn = self.kernel_dim * (self.n_kernels + 1)
        self.fnn.append(nn.Linear(input_fnn, mlp_dim))
        for _ in range(self.n_mlp - 1):
            self.fnn.append(nn.Linear(self.mlp_dim, self.mlp_dim))
        self.fnn.append(nn.Linear(self.mlp_dim, self.n_outputs))

        # Initialize weights
        if self.initialize_weights:
            self.kernels.apply(weights_init)
            self.fnn.apply(weights_init)
            nn.init.xavier_uniform_(self.embedding.weight)

    def _set_save_latent_space(self, save_latent_space):
        self.save_latent_space = save_latent_space

    def forward(self, g_batch, input_scrambling=False):
        # Embedding
        if input_scrambling:
            g_batch.node_props = g_batch.node_props[torch.randperm(g_batch.node_props.shape[0])]
        features = self.embedding(
            g_batch.node_props, g_batch.atom_type_id
        )  # num_nodes x kernel_dim (embedding_dim = kernel_dim)
        coords = g_batch.node_coords
        features = torch.cat((coords, features), dim=1)  # num_nodes x (3 + kernel_dim) (3 = XYZ)

        # Kernel
        feature_list = []
        feature_list.append(features[:, self.pos_dim :])

        for kernel in self.kernels:
            features = kernel(x=features, edge_index=g_batch.edge_index)  # num_nodes x (3 + kernel_dim)
            feature_list.append(features[:, self.pos_dim :])

        # Concat
        features = F.silu(
            torch.cat(feature_list, dim=1)
        )  # num_nodes x (self.n_kernels+1)*kernel_dim (+1 = initial features)

        # MLP 1 per node
        for mlp in self.fnn[:-1]:
            features = F.silu(mlp(features))  # num_nodes x mlp_dim

        if self.pool == "mean":
            features = scatter_mean(features, g_batch.batch, dim=0)
        elif self.pool == "sum":
            features = scatter_sum(features, g_batch.batch, dim=0)
        else:
            raise ValueError("Unknown pooling operation")
        # batch_size x mlp_dim
        if self.save_latent_space:
            return {"structure_ids": g_batch.pdb_id, "features": features}
        features = self.fnn[-1](features)  # batch_size x 1 (1 = output_dim)
        return features


class EGNNAtt(nn.Module):
    def __init__(
        self,
        n_kernels=5,
        n_mlp=3,
        mlp_dim=256,
        n_outputs=1,
        m_dim=32,
        initialize_weights=True,
        fourier_features=32,
        aggr="mean",
        pool="mean",
        kernel_dim=32,
        baseline_atom_ids=False,
        properties="yyyyyyyyy",
        save_latent_space=False,
    ):
        """Main Equivariant Graph Neural Network class.

        Parameters
        ----------
        embedding_dim : int, optional
            Embedding dimension, by default 128
        n_kernels : int, optional
            Number of message-passing rounds, by default 5
        n_mlp : int, optional
            Number of node-level and global-level MLPs, by default 3
        mlp_dim : int, optional
            Hidden size of the node-level and global-level MLPs, by default 256
        m_dim : int, optional
            Node-level hidden size, by default 32
        initialize_weights : bool, optional
            Whether to use Xavier init. for learnable weights, by default True
        fourier_features : int, optional
            Number of Fourier features to use, by default 32
        aggr : str, optional
            Node update function, by default "mean". If attention, includes the number of attention heads (e.g., "att_4")
        pool: str, optional
            How to pool features for global tasks, by default "mean". Other option: "sum"
        baseline_atom_ids: bool, optional
            Whether to just use an embedding of the adjacent atom ids
        """
        super(EGNNAtt, self).__init__()
        self.properties = [x == "y" for x in properties]  # boolean mask for which properties to use
        self.input_dim = sum(self.properties)
        self.pos_dim = 3  # XYZ coordinates
        self.m_dim = m_dim
        self.n_kernels = n_kernels
        self.n_mlp = n_mlp
        self.mlp_dim = mlp_dim
        self.n_outputs = n_outputs
        self.initialize_weights = initialize_weights
        self.fourier_features = fourier_features
        self.aggr = aggr  # message aggregation
        self.pool = pool
        self.kernel_dim = kernel_dim  # message passing size
        self.baseline_atom_ids = baseline_atom_ids
        self.attention_heads = int(self.pool.split("_")[1])
        self.save_latent_space = save_latent_space
        self.embedding = Embed(
            baseline_atom_ids=self.baseline_atom_ids,
            properties=self.properties,
            kernel_dim=self.kernel_dim,
            input_dim=self.input_dim,
        )

        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=self.kernel_dim,
                    pos_dim=self.pos_dim,
                    m_dim=self.m_dim,
                    fourier_features=self.fourier_features,
                    aggr=self.aggr,
                )
            )

        # post-message-passing MLP
        self.post_message_passing_mlp = nn.ModuleList()
        input_fnn = self.kernel_dim * (self.n_kernels + 1)
        self.post_message_passing_mlp.append(nn.Linear(input_fnn, self.mlp_dim))
        for _ in range(self.n_mlp - 1):
            self.post_message_passing_mlp.append(nn.Linear(self.mlp_dim, self.mlp_dim))

        # attention-based pooling
        self.transformer = GraphMultisetTransformer(
            in_channels=self.mlp_dim,
            hidden_channels=self.mlp_dim,
            out_channels=self.mlp_dim,
            num_heads=self.attention_heads,
            layer_norm=True,
        )

        # post-attention MLP
        self.post_attention_mlp = nn.ModuleList()
        for _ in range(self.n_mlp):
            self.post_attention_mlp.append(nn.Linear(self.mlp_dim, self.mlp_dim))
        self.post_attention_mlp.append(nn.Linear(self.mlp_dim, self.n_outputs))

        # Initialize weights
        if self.initialize_weights:
            self.kernels.apply(weights_init)
            self.post_message_passing_mlp.apply(weights_init)
            self.embedding.apply(weights_init)
            self.transformer.apply(weights_init)

    def _set_save_latent_space(self, save_latent_space):
        self.save_latent_space = save_latent_space

    def forward(self, g_batch, input_scrambling=False):
        # Embedding
        if input_scrambling:
            g_batch.node_props = g_batch.node_props[torch.randperm(g_batch.node_props.shape[0])]
        features = self.embedding(
            g_batch.node_props, g_batch.atom_type_id
        )  # num_nodes x kernel_dim (embedding_dim = kernel_dim)
        coords = g_batch.node_coords
        features = torch.cat((coords, features), dim=1)  # num_nodes x (3 + kernel_dim) (3 = XYZ)

        # Kernel
        feature_list = []
        feature_list.append(features[:, self.pos_dim :])

        for kernel in self.kernels:
            features = kernel(x=features, edge_index=g_batch.edge_index)  # num_nodes x (3 + kernel_dim)
            feature_list.append(features[:, self.pos_dim :])

        # Concat
        features = F.silu(
            torch.cat(feature_list, dim=1)
        )  # num_nodes x (self.n_kernels+1)*kernel_dim (+1 = initial features)

        # post-message-passing MLP
        for mlp in self.post_message_passing_mlp:
            features = F.silu(mlp(features))  # num_nodes x mlp_dim

        # Attention-based pooling
        features = self.transformer(
            features, edge_index=g_batch.edge_index, batch=g_batch.batch
        )  # batch_size x mlp_dim

        if self.save_latent_space:
            return {"structure_ids": g_batch.pdb_id, "features": features}

        # post-attention MLP
        for mlp in self.post_attention_mlp:
            features = mlp(features)  # batch_size x 1 (1 = output_dim)
        return features


class EGNN_IntGrads(nn.Module):
    def __init__(
        self,
        n_kernels=5,
        n_mlp=3,
        mlp_dim=256,
        n_outputs=1,
        m_dim=32,
        initialize_weights=True,
        fourier_features=32,
        aggr="mean",
        pool="mean",
        kernel_dim=32,
        baseline_atom_ids=False,
        properties="yyyyyyyyy",
        save_latent_space=False,
    ):
        """Main Equivariant Graph Neural Network class.

        Parameters
        ----------
        embedding_dim : int, optional
            Embedding dimension, by default 128
        n_kernels : int, optional
            Number of message-passing rounds, by default 5
        n_mlp : int, optional
            Number of node-level and global-level MLPs, by default 3
        mlp_dim : int, optional
            Hidden size of the node-level and global-level MLPs, by default 256
        m_dim : int, optional
            Node-level hidden size, by default 32
        initialize_weights : bool, optional
            Whether to use Xavier init. for learnable weights, by default True
        fourier_features : int, optional
            Number of Fourier features to use, by default 32
        aggr : str, optional
            Node update function, by default "mean"
        pool: str, optional
            How to pool features for global tasks, by default "mean". Other option: "sum"
        baseline_atom_ids: bool, optional
            Whether to just use an embedding of the adjacent atom ids
        """
        super(EGNN_IntGrads, self).__init__()
        self.properties = [x == "y" for x in properties]  # boolean mask for which properties to use
        self.input_dim = sum(self.properties)
        self.pos_dim = 3  # XYZ coordinates
        self.m_dim = m_dim
        self.n_kernels = n_kernels
        self.n_mlp = n_mlp
        self.mlp_dim = mlp_dim
        self.n_outputs = n_outputs
        self.initialize_weights = initialize_weights
        self.fourier_features = fourier_features
        self.aggr = aggr  # message aggregation
        self.pool = pool
        self.kernel_dim = kernel_dim  # message passing size
        self.baseline_atom_ids = baseline_atom_ids
        self.save_latent_space = save_latent_space
        self.embedding = Embed(
            baseline_atom_ids=self.baseline_atom_ids,
            properties=self.properties,
            kernel_dim=self.kernel_dim,
            input_dim=self.input_dim,
        )

        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=self.kernel_dim,
                    pos_dim=self.pos_dim,
                    m_dim=self.m_dim,
                    fourier_features=self.fourier_features,
                    aggr=self.aggr,
                )
            )

        # MLP 1
        self.fnn = nn.ModuleList()
        input_fnn = self.kernel_dim * (self.n_kernels + 1)
        self.fnn.append(nn.Linear(input_fnn, mlp_dim))
        for _ in range(self.n_mlp - 1):
            self.fnn.append(nn.Linear(self.mlp_dim, self.mlp_dim))
        self.fnn.append(nn.Linear(self.mlp_dim, self.n_outputs))

        # Initialize weights
        if self.initialize_weights:
            self.kernels.apply(weights_init)
            self.fnn.apply(weights_init)
            nn.init.xavier_uniform_(self.embedding.weight)

    def _set_save_latent_space(self, save_latent_space):
        self.save_latent_space = save_latent_space

    def forward(self, node_weights, g_batch):
        # Embedding
        features = self.embedding(
            g_batch.node_props, g_batch.atom_type_id
        )  # num_nodes x kernel_dim (embedding_dim = kernel_dim)
        features = features * node_weights  # for integrated gradients
        coords = g_batch.node_coords
        features = torch.cat((coords, features), dim=1)  # num_nodes x (3 + kernel_dim) (3 = XYZ)

        # Kernel
        feature_list = []
        feature_list.append(features[:, self.pos_dim :])

        for kernel in self.kernels:
            features = kernel(x=features, edge_index=g_batch.edge_index)  # num_nodes x (3 + kernel_dim)
            feature_list.append(features[:, self.pos_dim :])

        # Concat
        features = F.silu(
            torch.cat(feature_list, dim=1)
        )  # num_nodes x (self.n_kernels+1)*kernel_dim (+1 = initial features)

        # MLP 1 per node
        for mlp in self.fnn[:-1]:
            features = F.silu(mlp(features))  # num_nodes x mlp_dim

        if self.pool == "mean":
            features = scatter_mean(features, g_batch.batch, dim=0)
        elif self.pool == "sum":
            features = scatter_sum(features, g_batch.batch, dim=0)
        else:
            raise ValueError("Unknown pooling operation")
        # batch_size x mlp_dim
        if self.save_latent_space:
            return {"structure_ids": g_batch.pdb_id, "features": features}
        features = self.fnn[-1](features)  # batch_size x 1 (1 = output_dim)
        return features


def weights_init(m):
    """Xavier uniform weight initialization

    Parameters
    ----------
    m : [torch.nn.modules.linear.Linear]
        A list of learnable linear PyTorch modules.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


class EGNN_sparse(MessagePassing):
    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim=0,
        m_dim=32,
        dropout=0.1,
        fourier_features=32,
        aggr="mean",
        **kwargs,
    ):
        """Base torch geometric EGNN message-passing layer

        Parameters
        ----------
        feats_dim : int
            Dimension of the node-level features
        pos_dim : int, optional
            Dimensions of the positional features (e.g. cartesian coordinates), by default 3
        edge_attr_dim : int, optional
            Dimension of the edge-level features, by default 0
        m_dim : int, optional
            Hidden node/edge layer size, by default 32
        dropout : float, optional
            Whether to use dropout, by default 0.1
        fourier_features : int, optional
            Number of Fourier features, by default 32
        aggr : str, optional
            Node update function, by default "mean"
        """
        valid_aggrs = {
            "add",
            "max",
            "mean",
        }
        assert aggr in valid_aggrs, f"pool method must be one of {valid_aggrs}"

        kwargs.setdefault("aggr", aggr)
        super(EGNN_sparse, self).__init__(**kwargs)

        # Model parameters
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.fourier_features = fourier_features
        self.edge_attr_dim = edge_attr_dim

        self.edge_input_dim = (self.fourier_features * 2) + self.edge_attr_dim + 1 + (self.feats_dim * 2)
        # feats_dim*2: features of both adjacent nodes
        # edge_attr_dim: features of edge itself
        # fourier_features*2 --> fourier encoded di-atomic distance, 2x for sin & cos
        # +1 -> real distance

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Edge layers
        self.edge_norm1 = nn.LayerNorm(m_dim)
        self.edge_norm2 = nn.LayerNorm(m_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            nn.SiLU(),
        )

        # Node layers
        self.node_norm1 = nn.LayerNorm(feats_dim)
        self.node_norm2 = nn.LayerNorm(feats_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )

        # Initialization
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_features=None):
        coords, feats = x[:, : self.pos_dim], x[:, self.pos_dim :]
        rel_coords = coords[edge_index[0]] - coords[edge_index[1]]
        rel_dist = (rel_coords**2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, "n () d -> n d")

        if edge_features is not None:
            edge_attr = torch.cat([rel_dist, edge_features], dim=1)
        else:
            edge_attr = rel_dist
        hidden_out = self.propagate(
            edge_index,
            x=feats,
            edge_attr=edge_attr,
            coors=coords,
            rel_coors=rel_coords,
        )

        return torch.cat([coords, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr):
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        # get input tensors
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", coll_dict)
        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
        update_kwargs = self.inspector.distribute("update", coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)
        m_ij = self.edge_norm1(m_ij)

        # aggregate messages
        m_i = self.aggregate(m_ij, **aggr_kwargs)
        m_i = self.edge_norm1(m_i)

        # get updated node features
        hidden_feats = self.node_norm1(kwargs["x"])
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = self.node_norm2(hidden_out)
        hidden_out = kwargs["x"] + hidden_out

        return self.update(hidden_out, **update_kwargs)
