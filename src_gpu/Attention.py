import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, attn_vec_dim):
        super(MultiHeadAttention, self).__init__()
        self.fc1 = nn.Linear(in_dim, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(in_dim)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, metapath_outs):
        # metapath_outs shape: (num_paths, 2065, in_dim)
        num_paths, seq_len, in_dim = metapath_outs.shape
        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))  # shape: (2065, attn_vec_dim)
            fc2 = self.fc2(fc1)  # shape: (2065, 1)
            beta.append(fc2)

        beta = torch.stack(beta, dim=0)  # shape: (num_paths, 2065, 1)
        beta = F.softmax(beta, dim=0)  # shape: (num_paths, 2065, 1)

        # No need to stack again, use directly
        h = torch.sum(beta * metapath_outs, dim=0, keepdim=True)  # shape: (1, 2065, in_dim)
        h = self.layer_norm(h + metapath_outs.mean(dim=0, keepdim=True))  # residual connection and layer norm
        return h, beta

class Attention(nn.Module):
    def __init__(self, in_dim, attn_vec_dim):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(in_dim, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, metapath_outs):
        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc2 = self.fc2(fc1)
            beta.append(fc2)
        
        beta = torch.stack(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        
        h = torch.sum(beta * metapath_outs, dim=0)
        return h, beta.squeeze(-1)


        