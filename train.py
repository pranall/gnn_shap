import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import numpy as np
import shap
from tqdm import tqdm
import argparse

# Define the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train Diversify+GNN with optional SHAP")

    # Diversify / dataset flags
    parser.add_argument('--dataset',           type=str,   default='emg',
                        help="Which dataset to run (e.g. emg, dsads)")
    parser.add_argument('--data_dir',          type=str,   default='./data/',
                        help="Path to your ./data/ folder")
    parser.add_argument('--test_envs',         type=int,   nargs='+', default=[0],
                        help="Which environment(s) to hold out as test")
    parser.add_argument('--algorithm',         type=str,   default='diversify',
                        help="Domain generalization algorithm")
    parser.add_argument('--use_gnn',           type=int,   choices=[0,1], default=1,
                        help="1 to enable GNN, 0 to disable")
    parser.add_argument('--latent_domain_num', type=int,   default=5,
                        help="Number of latent domains")
    parser.add_argument('--alpha1',            type=float, default=1.0,
                        help="Diversify hyperparam α1")
    parser.add_argument('--alpha',             type=float, default=10.0,
                        help="Diversify hyperparam α")
    parser.add_argument('--lam',               type=float, default=0.0,
                        help="Diversify hyperparam λ")
    parser.add_argument('--local_epoch',       type=int,   default=2,
                        help="Local adaptation epochs")

    # Model / training flags
    parser.add_argument('--in_channels',       type=int,   default=8,
                        help="Number of input channels")
    parser.add_argument('--hidden_dim',        type=int,   default=64,
                        help="Hidden dimension of GNN")
    parser.add_argument('--num_layers',        type=int,   default=2,
                        help="Number of GCN layers")
    parser.add_argument('--lstm_hidden',       type=int,   default=64,
                        help="Hidden size of LSTM")
    parser.add_argument('--output_dim',        type=int,   default=6,
                        help="Number of classes / output dim")
    parser.add_argument('--lr',                type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument('--weight_decay',      type=float, default=0.0005,
                        help="Weight decay")
    parser.add_argument('--batch_size',        type=int,   default=32,
                        help="Batch size")
    parser.add_argument('--max_epoch',         type=int,   default=1,
                        help="Number of epochs")
    parser.add_argument('--device',            type=str,   choices=['cpu','cuda'], default='cuda',
                        help="Device to train on")

    # SHAP flags
    parser.add_argument('--use_shap',          type=int, choices=[0,1], default=1,
                        help="1 to run SHAP during validation")
    parser.add_argument('--shap_freq',         type=int, default=1,
                        help="Run SHAP every N epochs")

    return args


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


def validate(args, model, valid_loader, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(args.device)
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%, Loss: {total_loss/len(valid_loader):.4f}')
    
    # Run SHAP explanation during validation
    if args.use_shap and epoch % args.shap_freq == 0:
        print("\n[SHAP] Generating explanations...")
        shap_values, explained_data = explain_gnn_with_shap(
            model,
            valid_loader,
            device=args.device
        )
        
        if shap_values is not None:
            print("SHAP values computed successfully")
            # Add visualization/saving here
            visualize_shap_values(shap_values, explained_data)
    
    return accuracy


def train(args, model, train_loader, valid_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(args.max_epoch):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.max_epoch}, Loss: {train_loss/len(train_loader):.4f}")
        
        # Validation phase with SHAP
        val_acc = validate(args, model, valid_loader, epoch)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')


def main(args):
    # Initialize model
    model = TemporalGCN(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lstm_hidden=args.lstm_hidden,
        output_dim=args.output_dim
    ).to(args.device)
    
    # Run training
    train(args, model, train_loader, valid_loader)
    
    return model


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Create data loaders (you'll need to implement this)
    train_loader, valid_loader = create_data_loaders(args)
    
    # Run training
    model = main(args)
