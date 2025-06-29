import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class TemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, lstm_hidden, output_dim, dropout=0.2):
        super(TemporalGCN, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_dim))
        self.norms.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for gcn, norm in zip(self.gcn_layers, self.norms):
            x = gcn(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        x_pooled = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        x_pooled = x_pooled.unsqueeze(1)       # [batch, seq, feat]
        _, (hn, _) = self.lstm(x_pooled)
        hn = hn[-1]
        out = self.out(hn)
        return out

def build_correlation_graph(batch_time_series, threshold=0.3, self_loops=True, max_edges_per_node=None):
    """
    Improved graph builder:
    - Uses absolute correlation (|r|).
    - Can limit edges per node.
    - Optionally adds self-loops for GCN stability.
    """
    from torch_geometric.data import Data

    data_list = []
    batch, channels, timesteps = batch_time_series.shape

    for i in range(batch):
        x = batch_time_series[i].T  # [timesteps, channels]
        x = x.float()
        # Compute correlations
        corr = torch.corrcoef(x.T).abs()  # [channels, channels]
        edge_index = (corr > threshold).nonzero(as_tuple=False).T  # [2, num_edges]
        # Remove self-loops unless requested
        if not self_loops:
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
        # Optionally limit the number of edges per node for sparsity
        if max_edges_per_node is not None:
            # Limit edges per node (keep top-k highest correlations per node)
            topk_edges = []
            for node in range(channels):
                node_corr = corr[node]
                node_corr[node] = 0  # Remove self correlation
                topk = torch.topk(node_corr, k=min(max_edges_per_node, channels-1)).indices
                for dst in topk:
                    topk_edges.append([node, dst.item()])
            if topk_edges:
                edge_index = torch.tensor(topk_edges, dtype=torch.long).T
        # Node features: mean over time (could also try std or raw window)
        node_features = x.mean(dim=0).unsqueeze(0)  # [1, channels]
        node_features = node_features.repeat(channels, 1)  # [channels, channels]
        data = Data(x=node_features, edge_index=edge_index)
        data_list.append(data)
    return data_list
