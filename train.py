import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm import tqdm
import shap

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
        self.hidden_dim = hidden_dim

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
    """Improved graph builder with correlation-based edges"""
    data_list = []
    batch, channels, timesteps = batch_time_series.shape

    for i in range(batch):
        x = batch_time_series[i].T  # [timesteps, channels]
        x = x.float()
        corr = torch.corrcoef(x.T).abs()  # [channels, channels]
        edge_index = (corr > threshold).nonzero(as_tuple=False).T
        
        if not self_loops:
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            
        if max_edges_per_node is not None:
            topk_edges = []
            for node in range(channels):
                node_corr = corr[node]
                node_corr[node] = 0
                topk = torch.topk(node_corr, k=min(max_edges_per_node, channels-1)).indices
                for dst in topk:
                    topk_edges.append([node, dst.item()])
            if topk_edges:
                edge_index = torch.tensor(topk_edges, dtype=torch.long).T
                
        node_features = x.mean(dim=0).unsqueeze(0)
        node_features = node_features.repeat(channels, 1)
        data = Data(x=node_features, edge_index=edge_index)
        data_list.append(data)
    return data_list

class GNNExplainerWrapper(nn.Module):
    """Wrapper for SHAP explanations that preserves graph structure"""
    def __init__(self, model, original_data_sample):
        super().__init__()
        self.model = model
        self.original_data = original_data_sample
        
    def forward(self, x_flat_list):
        device = next(self.model.parameters()).device
        data_list = []
        
        for x_flat in x_flat_list:
            x = torch.FloatTensor(x_flat.reshape(self.original_data.x.shape)).to(device)
            edge_index = self.original_data.edge_index.clone().to(device)
            data = Data(
                x=x,
                edge_index=edge_index,
                batch=torch.zeros(x.size(0), dtype=torch.long).to(device)
            data_list.append(data)
            
        batch = Batch.from_data_list(data_list)
        return self.model(batch)

def explain_gnn_predictions(model, data_loader, device, sample_count=10, nsamples=100):
    """Explain model predictions using SHAP values"""
    model.eval()
    
    # Get background samples
    background_batch = next(iter(data_loader)).to(device)
    background_data = background_batch.to_data_list()[:sample_count]
    explain_data = background_data[0]
    
    # Create wrapper model
    wrapped_model = GNNExplainerWrapper(model, explain_data).to(device)
    
    # Prepare background data
    background_shap = [data.x.cpu().numpy().flatten() for data in background_data]
    background_shap = np.array(background_shap)
    
    # Create explainer
    explainer = shap.KernelExplainer(
        model=lambda x: wrapped_model(x).detach().cpu().numpy(),
        data=background_shap,
        keep_index=True
    )
    
    # Calculate SHAP values
    sample_shap = explain_data.x.cpu().numpy().flatten()[np.newaxis, :]
    shap_values = explainer.shap_values(
        X=sample_shap,
        nsamples=nsamples,
        silent=True
    )
    
    # Reshape results
    shap_values = [v.reshape(explain_data.x.shape) for v in shap_values]
    
    return shap_values, explain_data

def train_model(args):
    """Main training function with integrated explanation capability"""
    # Your existing training setup
    model = TemporalGCN(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lstm_hidden=args.lstm_hidden,
        output_dim=args.output_dim
    ).to(args.device)
    
    # Training loop
    for epoch in range(args.epochs):
        # Your existing training code
        
        # After training, explain predictions if requested
        if args.use_shap and epoch == args.epochs - 1:
            print("\n[INFO] Generating SHAP explanations...")
            shap_values, explained_data = explain_gnn_predictions(
                model=model,
                data_loader=valid_loader,
                device=args.device,
                sample_count=10
            )
            
            # You can add visualization here
            print("SHAP values computed for final model")
    
    return model

if __name__ == "__main__":
    # Your argument parsing and main execution
    args = parse_args()
    model = train_model(args)
