import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import numpy as np
import shap
from tqdm import tqdm

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

class GNNExplainerWrapper(nn.Module):
    """Wrapper for SHAP explanations that preserves graph structure"""
    def __init__(self, model, original_data_sample=None):
        super().__init__()
        self.model = model
        self.original_data = original_data_sample
        
    def forward(self, inputs):
        """Handle both direct GNN calls and SHAP explanations"""
        if isinstance(inputs, (list, np.ndarray)):
            # SHAP explanation case
            return self._explain_forward(inputs)
        # Normal GNN forward pass
        return self.model(inputs)
        
    def _explain_forward(self, x_flat_list):
        """Convert SHAP inputs to graph data"""
        if self.original_data is None:
            raise ValueError("Original data sample required for SHAP explanations")
            
        device = next(self.model.parameters()).device
        data_list = []
        
        for x_flat in x_flat_list:
            # Reshape flattened features to original dimensions
            x = torch.FloatTensor(x_flat.reshape(self.original_data.x.shape)).to(device)
            
            # Create new Data object preserving original structure
            data = Data(
                x=x,
                edge_index=self.original_data.edge_index.clone().to(device),
                batch=torch.zeros(x.size(0), dtype=torch.long).to(device)
            )
            data_list.append(data)
            
        batch = Batch.from_data_list(data_list)
        return self.model(batch)

def build_correlation_graph(batch_time_series, threshold=0.3, self_loops=True, max_edges_per_node=None, device='cpu'):
    """
    Improved graph builder:
    - Uses absolute correlation (|r|)
    - Can limit edges per node
    - Optionally adds self-loops for GCN stability
    """
    data_list = []
    batch_size, channels, timesteps = batch_time_series.shape

    for i in range(batch_size):
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
                node_corr[node] = 0  # Remove self correlation
                topk = torch.topk(node_corr, k=min(max_edges_per_node, channels-1)).indices
                for dst in topk:
                    topk_edges.append([node, dst.item()])
            if topk_edges:
                edge_index = torch.tensor(topk_edges, dtype=torch.long).to(device)
        
        node_features = x.mean(dim=0).unsqueeze(0)
        node_features = node_features.repeat(channels, 1)
        data = Data(
            x=node_features.to(device),
            edge_index=edge_index.to(device),
            batch=torch.zeros(channels, dtype=torch.long).to(device)
        )
        data_list.append(data)
    
    return data_list

def explain_gnn_predictions(model, data_loader, device, sample_count=5, nsamples=50):
    """Explain model predictions using SHAP values"""
    model.eval()
    
    # Get background samples
    try:
        background_batch = next(iter(data_loader)).to(device)
        background_data = background_batch.to_data_list()[:sample_count]
        explain_data = background_data[0]
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None
    
    # Create wrapper model
    wrapped_model = GNNExplainerWrapper(model, explain_data).to(device)
    
    # Prepare background data
    background_shap = [data.x.cpu().numpy().flatten() for data in background_data]
    background_shap = np.array(background_shap)
    
    # Create explainer
    explainer = shap.KernelExplainer(
        model=wrapped_model,
        data=background_shap,
        keep_index=True
    )
    
    # Calculate SHAP values
    try:
        sample_shap = explain_data.x.cpu().numpy().flatten()[np.newaxis, :]
        shap_values = explainer.shap_values(sample_shap, nsamples=nsamples)
        shap_values = shap_values[0].reshape(explain_data.x.shape)
        return shap_values, explain_data
    except Exception as e:
        print(f"SHAP calculation failed: {e}")
        return None, None

def train(args, model, train_loader, valid_loader):
    """Main training function"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.max_epoch):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.max_epoch}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Run SHAP explanation if requested
        if args.use_shap and epoch == args.max_epoch - 1:
            print("\nGenerating SHAP explanations...")
            shap_values, explained_data = explain_gnn_predictions(
                model=model,
                data_loader=valid_loader,
                device=args.device,
                sample_count=3,
                nsamples=20
            )
            
            if shap_values is not None:
                print("SHAP values computed successfully")
                # Add visualization here if needed

def main(args):
    # Initialize model
    model = TemporalGCN(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lstm_hidden=args.lstm_hidden,
        output_dim=args.output_dim
    ).to(args.device)
    
    # Training
    train(args, model, train_loader, valid_loader)
    
    return model

if __name__ == "__main__":
    # Parse arguments (you'll need to implement this)
    args = parse_args()
    
    # Create data loaders (you'll need to implement this)
    train_loader, valid_loader = create_data_loaders(args)
    
    # Run training
    model = main(args)
