import os
import copy
import torch
import logging
import pandas as pd
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
from timeit import default_timer as timer
from collections import defaultdict
from torch.optim import Adam
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from src_gpu.utils import count_parameters, generate_negative_samples
from src_gpu.metric import accuracy_SBM
from src_gpu.position_encoding import POSENCODINGS
from src_gpu.GraphTransformer import GraphTransformer
from src_gpu.GraphDataset import GraphDataset
from src_gpu.Attention import *
from src_gpu.BinaryClassifier import *
from src_gpu.utils import print_memory_usage, get_free_gpus
from src_gpu.FocalLoss import combined_loss

# 获取空闲的GPU
free_gpus = get_free_gpus()
print("Free GPUs:", free_gpus)
if free_gpus:
    device_ids = free_gpus[:1] 
else:
    device_ids = [1]  # 如果没有空闲的 GPU，则手动定义GPU
primary_device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

scaler = GradScaler() # 添加梯度缩放

# 数据加载和预处理
def load_data(dataset):
    drdi = pd.read_csv('./data/' + dataset + '/DrDiNum.csv', header=None)
    drpr = pd.read_csv('./data/' + dataset + '/DrPrNum.csv', header=None)
    dipr = pd.read_csv('./data/' + dataset + '/DiPrNum.csv', header=None)
    max_node = max(max(drdi[0]), max(drdi[1]), max(drpr[0]), max(drpr[1]), max(dipr[0]), max(dipr[1]))
    positive_samples = drdi
    positive_samples[2] = 1
    negative_samples = generate_negative_samples(drdi, "1") # {"1","10","all"}
    negative_samples[2] = 0
    result = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
    return result, drdi, drpr, dipr, max_node

# 生成元路径数据集
def generate_metapath_datasets(dataset, metapath, drdi, drpr, dipr, max_node):
    G = nx.Graph()
    G.add_nodes_from(range(max(drdi[0]) + 1), node_type='D')
    G.add_nodes_from(range(max(drdi[0]) + 1, max(drdi[1]) + 1), node_type='P')
    G.add_nodes_from(range(max(drdi[1]) + 1, max_node + 1), node_type='T')

    edge_attrs = {'drdi': [1, 0, 0], 'drpr': [0, 1, 0], 'dipr': [0, 0, 1]}
    drdipr = pd.concat([drdi, drpr, dipr])

    for _, row in drdipr.iterrows():
        node1, node2 = row[0], row[1]
        if node1 in G and node2 in G:
            if G.nodes[node1]['node_type'] == 'D' and G.nodes[node2]['node_type'] == 'P':
                attr = edge_attrs['drdi']
            elif G.nodes[node1]['node_type'] == 'D' and G.nodes[node2]['node_type'] == 'T':
                attr = edge_attrs['drpr']
            elif G.nodes[node1]['node_type'] == 'P' and G.nodes[node2]['node_type'] == 'T':
                attr = edge_attrs['dipr']
            G.add_edge(node1, node2, edge_attr=attr)
            G.add_edge(node2, node1, edge_attr=attr)

    if dataset == 'B-Dataset':
        attr_path = './data/' + dataset + '/AllNodeAttribute.csv'
    else:
        attr_path = './data/' + dataset + '/AllNodeAttribute_ChemS_PhS.csv'

    node_attrs = pd.read_csv(attr_path, header=None).iloc[:, 1:]
    node_attrs.insert(0, 0, node_attrs.index)
    node_attrs_dict = node_attrs.set_index(0).T.to_dict('list')

    for node in G.nodes():
        G.nodes[node]['x'] = node_attrs_dict[node]

    pyg_data = from_networkx(G, group_node_attrs=['x'])

    # Ensure the cache path exists
    cache_path = f'./results/cache/{dataset}/custom_dataset_metapath/{metapath}'
    os.makedirs(cache_path, exist_ok=True)

    return GraphDataset([pyg_data], degree=True, metapath=metapath, walk_length=10, num_walks=10, use_subgraph_edge_attr=True, cache_path=cache_path)

# 定义训练和评估函数
def train_epoch(graph_model, attn_model, cls_model, loader, train_data, criterion, optimizer, use_cuda=False):
    graph_model.train()
    attn_model.train()
    cls_model.train()
    
    running_loss = 0.0
    n_sample = 0
    tic = timer()
    optimizer.zero_grad()
    
    all_outputs = []
    for i, data in enumerate(loader):
        if use_cuda:
            data = data.to(primary_device)
        with autocast():
            output = graph_model(data)
        all_outputs.append(output)     
    all_outputs = torch.stack(all_outputs)
    with autocast():
        attn_features, beta = attn_model(all_outputs)    
    combined_features = torch.stack([torch.cat((attn_features[node1], attn_features[node2]), dim=0) for node1, node2 in train_data[:,:2]])
    
    if use_cuda:
        combined_features = combined_features.to(primary_device)
    with autocast():
        output = cls_model(combined_features)
    
    target = train_data[:, 2].float()
    if use_cuda:
        target = target.to(primary_device)
    with autocast():
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    running_loss += loss.item() * len(train_data)
    n_sample += len(train_data)

    del all_outputs, attn_features, combined_features, output, target, loss
    torch.cuda.empty_cache()

    toc = timer()
    epoch_loss = running_loss / n_sample

    return toc - tic, epoch_loss, beta

def eval_epoch(graph_model, attn_model, cls_model, loader, eval_data, criterion, use_cuda=False, split='Val'):
    graph_model.eval()
    attn_model.eval()
    cls_model.eval()

    running_loss = 0.0
    all_predictions = []
    all_labels = []
    n_sample = 0
    tic = timer()

    all_outputs = []
    for i, data in enumerate(loader):
        if use_cuda:
            data = data.to(primary_device)
        with autocast():
            output = graph_model(data)
        all_outputs.append(output)
    
    all_outputs = torch.stack(all_outputs)
    with autocast():
        attn_features, beta = attn_model(all_outputs)    
    combined_features = torch.stack([torch.cat((attn_features[node1], attn_features[node2]), dim=0) for node1, node2 in eval_data[:,:2]])
    
    if use_cuda:
        combined_features = combined_features.to(primary_device)
    with autocast():
        output = cls_model(combined_features)
    
    target = eval_data[:, 2].float()
    if use_cuda:
        target = target.to(primary_device)
    with autocast():
        loss = criterion(output, target)
    
    running_loss += loss.item() * len(eval_data)
    n_sample += len(eval_data)

    all_predictions.extend(output.detach().cpu().numpy())
    all_labels.extend(target.detach().cpu().numpy())

    del all_outputs, attn_features, combined_features, output, target, loss
    torch.cuda.empty_cache()

    toc = timer()
    average_loss = running_loss / n_sample
    all_predictions = np.array(all_predictions)
    if np.any(np.isnan(all_predictions)) or np.any(np.isinf(all_predictions)):
        all_predictions = np.nan_to_num(all_predictions, nan=0.0, posinf=1.0, neginf=0.0)
    auc = roc_auc_score(all_labels, all_predictions)
    acc = accuracy_score(all_labels, (all_predictions > 0.5).astype(np.float32))

    return auc, average_loss, all_labels, all_predictions

def main(input_dataset):
    # Load data and initialize datasets and loaders
    result, drdi, drpr, dipr, max_node = load_data(input_dataset)
    metapath = "DTP"  # You can change this to the desired metapath
    dataset = generate_metapath_datasets(input_dataset, metapath, drdi, drpr, dipr, max_node)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize model parameters
    node_dim = 64 
    input_size = dataset[0].x.shape[1]  # Extract input size from the first sample
    num_class = 3
    dim_hidden = 64
    dropout = 0.2
    num_heads = 8
    num_layers = 6
    batch_norm = False
    abs_pe_dim = 3
    re_pe_dim = 3
    num_edge_features = 3

    # Initialize absolute and relative position encodings
    abs_pe = 'rw'
    abs_pe_encoder = None
    if abs_pe and abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[abs_pe]
        abs_pe_encoder = abs_pe_method(abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(dataset)

    re_pos_enc = 'pstep'
    beta = 1
    p = 1
    normalization = 'sym'
    zero_diag = False
    use_edge_attr = True
    pos_encoding_params_str = "{}_{}".format(p, beta)
    pos_cache_path = './results/cache/re_pe/{}/{}_{}_{}.pkl'.format(
        input_dataset, re_pos_enc, normalization, pos_encoding_params_str)
    re_pos_encoder = POSENCODINGS[re_pos_enc](
        pos_cache_path, normalization=normalization, zero_diag=zero_diag, use_edge_attr=use_edge_attr,
        beta=beta, p=p
    )
    re_pos_encoder.apply_to(dataset, split='all')

    use_cuda = torch.cuda.is_available()

    # Cross-validation
    all_best_scores = []

    # Split data into train and test sets using StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds = [(train_index, test_index) for train_index, test_index in skf.split(result.iloc[:, [0, 1]], result.iloc[:, 2])]

    for fold, (train_index, test_index) in enumerate(folds):
        train_data = torch.tensor(result.iloc[train_index].values)
        test_data = torch.tensor(result.iloc[test_index].values)

        if torch.cuda.is_available():
            train_data = train_data.to(primary_device)
            test_data = test_data.to(primary_device)

        # Initialize models on primary_device
        graph_model = GraphTransformer(
            in_size=input_size, num_class=num_class, d_model=dim_hidden, dim_feedforward=2*dim_hidden,
            dropout=dropout, num_heads=num_heads, num_layers=num_layers, batch_norm=batch_norm,
            abs_pe=True, abs_pe_dim=abs_pe_dim, re_pe=re_pos_enc, re_pe_dim=re_pe_dim, gnn_type='graph',
            use_edge_attr=False, num_edge_features=num_edge_features, in_embed=False, se='khopgnn', 
            use_global_pool=False, global_pool='cls'
        ).to(primary_device)

        attn_model = Attention(
            in_dim=node_dim*2, attn_vec_dim=dim_hidden*2
        ).to(primary_device)

        cls_model = BinaryClassifier(
            input_dim=node_dim*2
        ).to(primary_device)

        # Wrap models with DataParallel
        if len(device_ids) > 1:
            print(f"使用多个 GPU: {device_ids}")
            graph_model = nn.DataParallel(graph_model, device_ids=device_ids)
            attn_model = nn.DataParallel(attn_model, device_ids=device_ids)
            cls_model = nn.DataParallel(cls_model, device_ids=device_ids)

        # Initialize loss function, optimizer, and learning rate scheduler
        criterion = nn.BCEWithLogitsLoss() # combined_loss
        optimizer = torch.optim.AdamW(list(graph_model.parameters()) + list(attn_model.parameters()) + list(cls_model.parameters()), lr=5e-4, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        # Ensure the directory exists
        save_dir = f'./results/model/{input_dataset}/fold_{fold}/'
        os.makedirs(save_dir, exist_ok=True)

        # Initialize logs, best scores and best weights
        logs = {'train_loss': [], 'test_score': [], 'test_loss': []}
        best_test_score = float('-inf')
        best_test_loss = float('inf')
        best_epoch = 0
        best_weights = {}
        best_beta = None
        best_labels = None
        best_predictions = None
        epochs = 500

        # Training loop
        start_time = timer()
        for epoch in range(epochs):
            logging.info("Epoch {}/{}, LR {:.6f}".format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
            train_time, train_loss, beta = train_epoch(graph_model, attn_model, cls_model, loader, train_data, criterion, optimizer, use_cuda)
            test_score, test_loss, test_labels, test_predictions = eval_epoch(graph_model, attn_model, cls_model, loader, test_data, criterion, use_cuda)

            lr_scheduler.step()

            logs['train_loss'].append(train_loss)
            logs['test_score'].append(test_score)
            logs['test_loss'].append(test_loss)
            logging.info("Train Loss: {:.4f}, Test Score: {:.4f}, Test Loss: {:.4f}, Time: {:.2f}s".format(train_loss, test_score, test_loss, train_time))

            if test_score > best_test_score:
                best_test_score = test_score
                best_test_loss = test_loss
                best_epoch = epoch
                best_weights = {
                    'graph_model': copy.deepcopy(graph_model.state_dict()),
                    'attn_model': copy.deepcopy(attn_model.state_dict()),
                    'cls_model': copy.deepcopy(cls_model.state_dict())
                }
                best_beta = beta.clone().detach().cpu().numpy()
                best_labels = test_labels
                best_predictions = test_predictions

            # Clear unnecessary variables to free memory
            del train_loss, test_score, test_loss
            torch.cuda.empty_cache()  # Clear CUDA memory if using GPU

        total_time = timer() - start_time
        logging.info("Fold {} Best epoch: {} Best test score: {:.4f} Best test loss: {:.4f} Time: {:.2f}s".format(fold, best_epoch, best_test_score, best_test_loss, total_time))

        # Save the best model weights and beta for this fold
        torch.save(best_weights['graph_model'], os.path.join(save_dir, 'graph_model.pth'))
        torch.save(best_weights['attn_model'], os.path.join(save_dir, 'attn_model.pth'))
        torch.save(best_weights['cls_model'], os.path.join(save_dir, 'cls_model.pth'))
        np.save(os.path.join(save_dir, 'best_beta.npy'), best_beta)

        # Save the best labels and predictions for this fold
        df = pd.DataFrame({'label': best_labels, 'prediction': best_predictions})
        df.to_csv(os.path.join(save_dir, 'best_labels_predictions.csv'), index=False)

        # Store best score for this fold
        all_best_scores.append(best_test_score)

    # Output the best beta shapes for all folds and the scores
    for fold, score in enumerate(all_best_scores):
        logging.info(f"Fold {fold} obtained best score is: {score:.4f}")

    logging.info(f"10 fold CV mean best score is: {np.mean(all_best_scores):.4f}")
        
if __name__ == "__main__":
    # 设置日志记录
    input_dataset = 'F-Dataset'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(input_dataset + "log.txt"), logging.StreamHandler()])
    main(input_dataset)
