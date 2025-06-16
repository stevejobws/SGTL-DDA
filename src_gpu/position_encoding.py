import os
import pickle
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
from torch_scatter import scatter_add
import torch_geometric.utils as utils
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj

class RPositionEncoding(object):
    def __init__(self, savepath=None, zero_diag=False):
        self.savepath = savepath
        self.zero_diag = zero_diag

    def apply_to(self, dataset, split='train'):
        saved_pos_enc = self.load(split)
        all_pe = []
        dataset.re_pe_list = []
        for i, g in enumerate(dataset):
            if saved_pos_enc is None:
                pe = self.compute_pe(g)
                all_pe.append(pe)
            else:
                pe = saved_pos_enc[i]
            if self.zero_diag:
                pe = pe.clone()
                pe.diagonal()[:] = 0
            dataset.re_pe_list.append(pe)

        self.save(all_pe, split)

        return dataset

    def save(self, pos_enc, split):
        if self.savepath is None:
            return
        directory = os.path.dirname(self.savepath + "." + split)
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.isfile(self.savepath + "." + split):
            with open(self.savepath + "." + split, 'wb') as handle:
                pickle.dump(pos_enc, handle)

    def load(self, split):
        if self.savepath is None:
            return None
        if not os.path.isfile(self.savepath + "." + split):
            return None
        with open(self.savepath + "." + split, 'rb') as handle:
            pos_enc = pickle.load(handle)
        return pos_enc

    def compute_pe(self, graph):
        pass


class DiffusionEncoding(RPositionEncoding):
    def __init__(self, savepath, beta=1., use_edge_attr=False, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
                graph.edge_index, edge_attr, normalization=self.normalization,
                num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        L = expm(-self.beta * L)
        return torch.from_numpy(L.toarray())


class PStepRWEncoding(RPositionEncoding):
    def __init__(self, savepath, p=1, beta=0.5, use_edge_attr=False, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.p = p
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_attr = edge_attr.argmax(dim=1) + 1
        edge_attr = edge_attr.to(torch.float)
        edge_index = graph.edge_index.to(torch.int64)
        edge_index, edge_attr = get_laplacian(
            edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        L = sp.identity(L.shape[0], dtype=L.dtype) - self.beta * L
        tmp = L
        for _ in range(self.p - 1):
            tmp = tmp.dot(L)
        # add identity matrix       
        pe = torch.from_numpy(tmp.toarray())
        identity_matrix = torch.eye(pe.size(0))
        pe_with_identity = pe + identity_matrix        
        return pe_with_identity

class PositionEncoding(object):
    def apply_to(self, dataset):
        dataset.abs_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)         
            dataset.abs_pe_list.append(pe)
        return dataset

class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = utils.get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:,idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()


class RWEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        W0 = normalize_adj(graph.edge_index, num_nodes=graph.num_nodes).tocsc()
        W = W0
        vector = torch.zeros((graph.num_nodes, self.pos_enc_dim))
        vector[:, 0] = torch.from_numpy(W0.diagonal())
        for i in range(self.pos_enc_dim - 1):
            W = W.dot(W0)
            vector[:, i + 1] = torch.from_numpy(W.diagonal())
        return vector.float()


def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 device=edge_index.device)
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)


POSENCODINGS = {
    'lap': LapEncoding,
    'rw': RWEncoding,
    "diffusion": DiffusionEncoding,
    "pstep": PStepRWEncoding,
}
