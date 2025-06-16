import torch
from torch import nn
import torch_geometric.nn as gnn
from .layers import TransformerEncoderLayer
from einops import repeat
import torch.nn.functional as F
from .layers import DiffTransformerEncoderLayer
from src_gpu.utils import print_memory_usage

class DiffTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, pe, batch_num, degree=None, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, pe=pe, batch_num=batch_num, degree=degree, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index, subgraph_node_index=None, subgraph_edge_index=None,
                subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
                ptr=None, return_attn=False):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index, edge_attr=edge_attr, degree=degree,
                         subgraph_node_index=subgraph_node_index, subgraph_edge_index=subgraph_edge_index,
                         subgraph_indicator_index=subgraph_indicator_index, subgraph_edge_attr=subgraph_edge_attr,
                         ptr=ptr, return_attn=return_attn)
        if self.norm is not None:
            output = self.norm(output)
        return output

class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8, dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0, re_pe=False, re_pe_dim=0, gnn_type="graph", se="gnn",
                 use_edge_attr=False, num_edge_features=4, in_embed=True, edge_embed=True, use_global_pool=True,
                 max_seq_len=None, global_pool='mean', **kwargs):
        super(GraphTransformer, self).__init__()

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        self.re_pe = re_pe
        self.re_pe_dim = re_pe_dim
        if re_pe and re_pe_dim > 0:
            self.embedding_re_pe = nn.Linear(re_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size, out_features=d_model, bias=False)

        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features, out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        ab_encoder_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
                                                   gnn_type=gnn_type, se=se, **kwargs)
        self.ab_encoder = GraphTransformerEncoder(ab_encoder_layer, num_layers)

        re_encoder_layer = DiffTransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        self.re_encoder = DiffTransformerEncoder(re_encoder_layer, num_layers)

        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

        self.combined_dim = 128 # 64 # 128
        self.intermediate_dim = 64 # 32 # 64
        self.W_h = nn.Parameter(torch.randn(self.combined_dim, self.intermediate_dim))
        self.b_h = nn.Parameter(torch.randn(self.intermediate_dim))
        self.W_w = nn.Parameter(torch.randn(self.intermediate_dim, self.intermediate_dim))
        self.b_w = nn.Parameter(torch.randn(self.intermediate_dim))

    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # device = x.device     
        # Print the device information for debugging
        # print(f"Device: {device}")
        # print(f"Embedding weight device: {self.embedding.weight.device}")
        # if hasattr(self.embedding, 'bias') and self.embedding.bias is not None:
            # print(f"Embedding bias device: {self.embedding.bias.device}")

        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        re_pe = data.re_pe if hasattr(data, 're_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None
        
        output = self.embedding(x)
        # print(f"Output after embedding call device: {output.device}")
        # if hasattr(self.embedding, 'bias') and self.embedding.bias is not None:
            # print(f"Embedding bias device after call: {self.embedding.bias.device}")
        
        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        ab_output = self.ab_encoder(
            output,
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )

        batch_num = data.batch.size()[0]
        masks = None
        re_output = self.embedding(x).unsqueeze(1)
        re_output = self.re_encoder(re_output, re_pe.unsqueeze(0), batch_num=batch_num, degree=degree, src_key_padding_mask=masks)
        re_output = re_output.squeeze(1)

        combined_features = torch.cat((ab_output, re_output), dim=1)
        # combined_features = (ab_output + re_output)/2
        
        # primary_device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        # attention_layer = torch.nn.Linear(ab_output.shape[1], re_output.shape[1]).to(primary_device)
        # attention_scores = attention_layer(ab_output)
        # attention_weights = torch.softmax(attention_scores, dim=-1)
        # combined_features = attention_weights * ab_output + (1 - attention_weights) * re_output
        
        # combined_features = ab_output * re_output
        
        return combined_features

