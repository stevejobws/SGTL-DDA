# -*- coding: utf-8 -*-
from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse import csr_matrix,lil_matrix
import numpy as np
import pandas as pd

import psutil
def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory Used: {memory_info.rss / 1024 ** 2:.2f} MB")

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def dense_to_sparse_tensor(matrix):
    rows, columns = torch.where(matrix > 0)
    values = torch.ones(rows.shape)
    indices = torch.from_numpy(np.vstack((rows,
                                          columns))).long()
    shape = torch.Size(matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                         data.edge_index[0],
                         dim=0,
                         dim_size=data.num_nodes,
                         reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data


def pad_batch(x, ptr, return_mask=False):
    bsz = len(ptr) - 1
    # num_nodes = torch.diff(ptr)
    max_num_nodes = torch.diff(ptr).max().item()

    all_num_nodes = ptr[-1].item()
    cls_tokens = False
    x_size = len(x[0]) if isinstance(x, (list, tuple)) else len(x)
    if x_size > all_num_nodes:
        cls_tokens = True
        max_num_nodes += 1
    if isinstance(x, (list, tuple)):
        new_x = [xi.new_zeros(bsz, max_num_nodes, xi.shape[-1]) for xi in x]
        if return_mask:
            padding_mask = x[0].new_zeros(bsz, max_num_nodes).bool()
    else:
        new_x = x.new_zeros(bsz, max_num_nodes, x.shape[-1])
        if return_mask:
            padding_mask = x.new_zeros(bsz, max_num_nodes).bool()

    for i in range(bsz):
        num_node = ptr[i + 1] - ptr[i]
        if isinstance(x, (list, tuple)):
            for j in range(len(x)):
                new_x[j][i][:num_node] = x[j][ptr[i]:ptr[i + 1]]
                if cls_tokens:
                    new_x[j][i][-1] = x[j][all_num_nodes + i]
        else:
            new_x[i][:num_node] = x[ptr[i]:ptr[i + 1]]
            if cls_tokens:
                new_x[i][-1] = x[all_num_nodes + i]
        if return_mask:
            padding_mask[i][num_node:] = True
            if cls_tokens:
                padding_mask[i][-1] = False
    if return_mask:
        return new_x, padding_mask
    return new_x

def unpad_batch(x, ptr):
    bsz, n, d = x.shape
    max_num_nodes = torch.diff(ptr).max().item()
    num_nodes = ptr[-1].item()
    all_num_nodes = num_nodes
    cls_tokens = False
    if n > max_num_nodes:
        cls_tokens = True
        all_num_nodes += bsz
    new_x = x.new_zeros(all_num_nodes, d)
    for i in range(bsz):
        new_x[ptr[i]:ptr[i + 1]] = x[i][:ptr[i + 1] - ptr[i]]
        if cls_tokens:
            new_x[num_nodes + i] = x[i][-1]
    return new_x

def generate_negative_samples1(drdi, num_samples=18416):
    # 假设 drdi 的第一列为节点类型A的编号，第二列为节点类型B的编号
    # 节点类型A的数量为第一列的最大值加1（因为是从0开始编号）
    num_A = drdi.iloc[:, 0].max() + 1
    # 节点类型B的数量为第二列的最大值减去第一列的最大值
    num_B = drdi.iloc[:, 1].max() - drdi.iloc[:, 0].max()
    # 所有可能的正样本数
    all_possible_samples = num_A * num_B
    # 所需的负样本数
    num_negative_samples = all_possible_samples - len(drdi)
    # 如果需要的负样本数超过了总的可能的负样本数，抛出异常
    if num_samples > num_negative_samples:
        raise ValueError(f"Requested number of negative samples ({num_samples}) exceeds the total possible number of negative samples ({num_negative_samples}).")
    # 生成所有正样本的集合
    positive_samples_set = set(tuple(x) for x in drdi.values)
    # 初始化负样本集合
    negative_samples_set = set()
    # 随机生成负样本，直到达到所需数量
    while len(negative_samples_set) < num_samples:
        # 随机选择节点A和节点B的编号
        a = np.random.randint(0, num_A)
        b = np.random.randint(num_A, num_A + num_B)
        # 如果不是正样本，加入负样本集合
        if (a, b) not in positive_samples_set:
            negative_samples_set.add((a, b))
    # 将负样本集合转换为DataFrame
    negative_samples = pd.DataFrame(list(negative_samples_set))
    return negative_samples
    
    
def generate_negative_samples(positive_samples, ratio='1'):
    # 假设 positive_samples 的第一列为节点类型A的编号，第二列为节点类型B的编号
    num_A = positive_samples.iloc[:, 0].max() + 1
    num_B = positive_samples.iloc[:, 1].max() - positive_samples.iloc[:, 0].max()
    # 所有可能的正样本数
    all_possible_samples = num_A * num_B
    # 正样本数
    num_positive_samples = len(positive_samples)
    # 所有负样本数
    num_negative_samples = all_possible_samples - num_positive_samples
    
    if ratio == '1':
        num_samples = num_positive_samples
    elif ratio == '10':
        num_samples = num_positive_samples * 10
    elif ratio == 'all':
        num_samples = num_negative_samples
    else:
        raise ValueError(f"Invalid ratio value: {ratio}. It should be '1', '10', or 'all'.")
    
    # 如果需要的负样本数超过了总的可能的负样本数，抛出异常
    if num_samples > num_negative_samples:
        raise ValueError(f"Requested number of negative samples ({num_samples}) exceeds the total possible number of negative samples ({num_negative_samples}).")
    
    # 生成所有正样本的集合
    positive_samples_set = set(tuple(x) for x in positive_samples.values)
    # 初始化负样本集合
    negative_samples_set = set()
    # 随机生成负样本，直到达到所需数量
    while len(negative_samples_set) < num_samples:
        # 随机选择节点A和节点B的编号
        a = np.random.randint(0, num_A)
        b = np.random.randint(num_A, num_A + num_B)
        # 如果不是正样本，加入负样本集合
        if (a, b) not in positive_samples_set:
            negative_samples_set.add((a, b))
    
    # 将负样本集合转换为DataFrame
    negative_samples = pd.DataFrame(list(negative_samples_set))
    return negative_samples

import subprocess
import re

def get_free_gpus():
    free_gpus = []
    try:
        # 使用 nvidia-smi 命令获取 GPU 状态
        smi_output = subprocess.check_output('nvidia-smi', encoding='utf-8')

        # 使用正则表达式解析输出
        gpu_usage = re.findall(r'\d+MiB / \d+MiB', smi_output)
        for idx, usage in enumerate(gpu_usage):
            used, total = map(int, re.findall(r'\d+', usage))
            if used < total * 0.1:  # 假设少于10%的使用被认为是空闲
                free_gpus.append(idx)
    except Exception as e:
        print(f"Error fetching free GPUs: {e}")

    return free_gpus
