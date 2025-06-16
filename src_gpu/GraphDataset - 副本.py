# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.data import Data
import numpy as np
import os
import random

def my_inc(self, key, value, *args, **kwargs):
    if key == 'subgraph_edge_index':
        return self.num_subgraph_nodes 
    if key == 'subgraph_node_idx':
        return self.num_nodes 
    if key == 'subgraph_indicator':
        return self.num_nodes 
    elif 'index' in key:
        return self.num_nodes
    else:
        return 0
def flexible_metapath_random_walk(data, start_node, metapath, walk_length):
    """
    Perform a random walk on a PyTorch Geometric Data object where the next node type 
    can be any subsequent type in the metapath after the current node's type.

    Args:
    data (torch_geometric.data.Data): Input graph data
    start_node (int): Start node index for the random walk
    metapath (str): Metapath schema (e.g., "DPTDP")
    walk_length (int): Length of the random walk

    Returns:
    list: A sequence of nodes visited during the random walk
    list: A list of tuples representing the edges traversed during the walk
    """
    walk = [start_node]
    edges_traversed = []

    cur_node = start_node

    while len(walk) < walk_length:
        cur_type = data.node_type[cur_node]
        valid_next_types = metapath[metapath.index(cur_type) + 1:] + metapath[:metapath.index(cur_type)]
        
        neighbors = (data.edge_index[0] == cur_node).nonzero(as_tuple=True)[0]
        neighbors = data.edge_index[1][neighbors].tolist()

        next_candidates = [n for n in neighbors if data.node_type[n] in valid_next_types]

        if not next_candidates:
            break

        next_node = random.choice(next_candidates)
        edges_traversed.append((cur_node, next_node))
        walk.append(next_node)
        cur_node = next_node

    return walk, edges_traversed

def generate_subgraph(data, start_node, metapath, walk_length):
    """
    Generate a subgraph from a PyTorch Geometric Data object based on a random walk.

    Args:
    data (torch_geometric.data.Data): Input graph data
    start_node (int): Start node index for the random walk
    metapath (str): Metapath schema (e.g., "DPTDP")
    walk_length (int): Length of the random walk

    Returns:
    torch.Tensor: Node indices of the subgraph
    torch.Tensor: Edge indices of the subgraph
    torch.Tensor: Edge mask indicating which edges are in the subgraph
    """
    # Perform a random walk starting from the start node
    walk, sub_edge_index = flexible_metapath_random_walk(data, start_node, metapath, walk_length)
    
    sub_nodes = torch.tensor(walk) 
    sub_edge_index = torch.tensor(sub_edge_index) 
    
    # Extract the unique nodes from the walk
    unique_nodes = torch.unique(torch.tensor(walk)) 

    # Create a mapping from original node indices to new indices
    node_mapping = {node.item(): i for i, node in enumerate(unique_nodes)}

    # Create edge indices for the subgraph and an edge mask
    subgraph_edges = []
    edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
    for i in range(len(walk) - 1):
        src, dest = walk[i], walk[i + 1]
        mapped_src = node_mapping[src]
        mapped_dest = node_mapping[dest]
        subgraph_edges.append([mapped_src, mapped_dest])

        # Update the edge mask
        edge_indices = (data.edge_index[0] == src) & (data.edge_index[1] == dest)
        edge_mask |= edge_indices
        
    return sub_nodes, sub_edge_index, edge_mask

class GraphDataset(object):
    def __init__(self, dataset, degree=False, metapath=None, walk_length=10, num_walks=20, use_subgraph_edge_attr=False, cache_path=None, return_complete_index=False):
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.degree = degree
        self.compute_degree()
        self.abs_pe_list = None
        self.re_pe_list = None
        self.return_complete_index = return_complete_index
        self.metapath = metapath
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.use_subgraph_edge_attr = use_subgraph_edge_attr
        self.cache_path = cache_path
        Data.__inc__ = my_inc
        self.extract_subgraphs()
 
    def compute_degree(self):
        if not self.degree:
            self.degree_list = None
            return
        self.degree_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)

    def extract_subgraphs(self):
        print("Extracting subgraphs based on metapath {}".format(self.metapath))
        # indicate which node in a graph it is; for each graph, the
        # indices will range from (0, num_nodes). PyTorch will then
        # increment this according to the batch size
        self.subgraph_node_index = []

        # Each graph will become a block diagonal adjacency matrix of
        # all the k-hop subgraphs centered around each node. The edge
        # indices get augumented within a given graph to make this
        # happen (and later are augmented for proper batching)
        self.subgraph_edge_index = []

        # This identifies which indices correspond to which subgraph
        # (i.e. which node in a graph)
        self.subgraph_indicator_index = []

        # This gets the edge attributes for the new indices
        if self.use_subgraph_edge_attr:
            self.subgraph_edge_attr = []

        for i in range(self.num_walks):
            if self.cache_path is not None:
                filepath = "{}_{}.pt".format(self.cache_path, i)
                if os.path.exists(filepath):
                    continue
            graph = self.dataset[0]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicators = []
            # edge_index_start = 0
            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, edge_mask = generate_subgraph(graph, node_idx, metapath=self.metapath, walk_length=self.walk_length) 
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index)
                # edge_indices.append(sub_edge_index + edge_index_start)
                indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask]) # CHECK THIS DIDN"T BREAK ANYTHING
                # edge_index_start += len(sub_nodes)

            if self.cache_path is not None:
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    subgraph_edge_attr = torch.cat(edge_attributes)
                else:
                    subgraph_edge_attr = None
                torch.save({
                    'subgraph_node_index': torch.cat(node_indices),
                    'subgraph_edge_index': torch.cat(edge_indices).t(),
                    'subgraph_indicator_index': torch.cat(indicators).type(torch.LongTensor),
                    'subgraph_edge_attr': subgraph_edge_attr
                }, filepath)
            else:
                self.subgraph_node_index.append(torch.cat(node_indices))
                self.subgraph_edge_index.append(torch.cat(edge_indices).t())
                self.subgraph_indicator_index.append(torch.cat(indicators))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    self.subgraph_edge_attr.append(torch.cat(edge_attributes))
            print("Done!")

    def __len__(self):
        return len(self.dataset) * self.num_walks

    def __getitem__(self, index):
        if isinstance(index, int):
            # Handle single item
            walk_index = index % self.num_walks
            # Rest of your logic
        elif isinstance(index, slice):
            # Handle slice
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")
        # walk_index = index % self.num_walks
        dataset_index = index // self.num_walks    
        data = self.dataset[dataset_index]
        
        if data.x is not None:
            data.x = data.x.squeeze(-1)
        n = data.num_nodes
        s = torch.arange(n)
        if self.return_complete_index:
            data.complete_edge_index = torch.vstack((s.repeat_interleave(n), s.repeat(n)))
        data.degree = None
        if self.degree:
            data.degree = self.degree_list[dataset_index]
        data.abs_pe = None
        if self.abs_pe_list is not None and len(self.abs_pe_list) == len(self.dataset) * self.num_walks:
            data.abs_pe = self.abs_pe_list[dataset_index]
        data.re_pe = None    
        if self.re_pe_list is not None and len(self.re_pe_list) == len(self.dataset) * self.num_walks:
            data.re_pe = self.re_pe_list[index]

        # add subgraphs and relevant meta data
        if self.cache_path is not None:
            cache_file = torch.load("{}_{}.pt".format(self.cache_path, walk_index))
            data.subgraph_edge_index = cache_file['subgraph_edge_index']
            data.num_subgraph_nodes = len(cache_file['subgraph_node_index'])
            data.subgraph_node_idx = cache_file['subgraph_node_index']
            data.subgraph_edge_attr = cache_file['subgraph_edge_attr']
            data.subgraph_indicator = cache_file['subgraph_indicator_index']
        else:
            data.subgraph_edge_index = self.subgraph_edge_index[walk_index]
            data.num_subgraph_nodes = len(self.subgraph_node_index[walk_index])
            data.subgraph_node_idx = self.subgraph_node_index[walk_index]
            if self.use_subgraph_edge_attr and data.edge_attr is not None:
                data.subgraph_edge_attr = self.subgraph_edge_attr[walk_index]
            data.subgraph_indicator = self.subgraph_indicator_index[walk_index].type(torch.LongTensor)

        return data

