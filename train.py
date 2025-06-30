import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from datautil.getdataloader_single import get_act_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Train Diversify+GNN with SHAP")
    parser.add_argument('--dataset',           type=str,   default='emg')
    parser.add_argument('--data_dir',          type=str,   default='./data/')
    parser.add_argument('--test_envs',         type=int,   nargs='+', default=[0])
    parser.add_argument('--algorithm',         type=str,   default='diversify')
    parser.add_argument('--use_gnn',           type=int,   choices=[0,1], default=1)
    parser.add_argument('--task',              type=str,   default='cross_people')
    parser.add_argument('--latent_domain_num', type=int,   default=5)
    parser.add_argument('--alpha1',            type=float, default=1.0)
    parser.add_argument('--alpha',             type=float, default=10.0)
    parser.add_argument('--lam',               type=float, default=0.0)
    parser.add_argument('--local_epoch',       type=int,   default=2)
    parser.add_argument('--in_channels',       type=int,   default=8)
    parser.add_argument('--hidden_dim',        type=int,   default=64)
    parser.add_argument('--num_layers',        type=int,   default=2)
    parser.add_argument('--lstm_hidden',       type=int,   default=64)
    parser.add_argument('--output_dim',        type=int,   default=6)
    parser.add_argument('--lr',                type=float, default=0.01)
    parser.add_argument('--weight_decay',      type=float, default=0.0005)
    parser.add_argument('--batch_size',        type=int,   default=32)
    parser.add_argument('--max_epoch',         type=int,   default=1)
    parser.add_argument('--device',            type=str,   choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--seed',              type=int,   default=0)
    parser.add_argument('--N_WORKERS',         type=int,   default=4)
    parser.add_argument('--steps_per_epoch',   type=int,   default=int(1e9))
    parser.add_argument('--use_shap',          type=int,   choices=[0,1], default=1)
    parser.add_argument('--shap_freq',         type=int,   default=1)
    parser.add_argument('--output',           type=str,   default='./output')
    return parser.parse_args()

def create_data_loaders(args):
    args.act_people = {
        'emg':   [list(range(0,9)), list(range(9,18)), list(range(18,27)), list(range(27,36))],
        'dsads':[list(range(i*8, i*8+8)) for i in range(8)]
    }
    train_loader, _, valid_loader, _, _, _, _ = get_act_dataloader(args)
    return train_loader, valid_loader

def build_correlation_graph(batch_time_series, threshold=0.3, self_loops=True,
                            max_edges_per_node=None, device='cpu'):
    if batch_time_series.dim() == 4:
        batch_time_series = batch_time_series.squeeze(2)
    B, C, T = batch_time_series.shape
    data_list = []
    for i in range(B):
        x_ts = batch_time_series[i].t().float()
        corr = torch.corrcoef(x_ts.T).abs()
        edge_index = (corr > threshold).nonzero(as_tuple=False).T.to(device)
        if not self_loops:
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
        if max_edges_per_node is not None:
            topk_edges = []
            for node in range(C):
                row = corr[node].clone()
                row[node] = 0
                topk = torch.topk(row, k=min(max_edges_per_node, C-1)).indices
                for dst in topk:
                    topk_edges.append([node, dst.item()])
            if topk_edges:
                edge_index = torch.tensor(topk_edges, dtype=torch.long).T.to(device)
        feat = x_ts.mean(dim=0).unsqueeze(0).repeat(C, 1).to(device)
        data_list.append(Data(x=feat, edge_index=edge_index))
    return data_list

def batch_to_graph(batch, device):
    if isinstance(batch, (list, tuple)):
        x_ts, y = batch[0], batch[1]
    else:
        raise ValueError("Expected batch as tuple/list")
    y = y.to(device).long()
    graphs = build_correlation_graph(x_ts, device=device)
    for g, label in zip(graphs, y):
        g.y = label
    return Batch.from_data_list(graphs)

class TemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, lstm_hidden, output_dim, dropout=0.2):
        super().__init__()
        self.gcn_layers = nn.ModuleList()
        self.norms      = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_dim))
        self.norms.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))
        self.lstm    = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.out     = nn.Linear(lstm_hidden, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data):
        x  = data.x.float()
        ei = data.edge_index
        b  = data.batch
        for gcn, norm in zip(self.gcn_layers, self.norms):
            x = F.relu(self.dropout(norm(gcn(x, ei))))
        pooled = global_mean_pool(x, b)
        seq    = pooled.unsqueeze(1)
        _, (hn, _) = self.lstm(seq)
        return self.out(hn[-1])

class GNNExplainerWrapper(nn.Module):
    def __init__(self, model, sample_data: Data):
        super().__init__()
        self.model  = model
        self.sample = sample_data

    def forward(self, *inputs):
        # Handle both direct calls and SHAP explanations
        if len(inputs) == 1 and isinstance(inputs[0], (np.ndarray, list)):
            # SHAP explanation case
            return self._explain_forward(inputs[0])
        # Normal GNN forward pass
        return self.model(*inputs)

    def _explain_forward(self, x_flat_list):
        """Convert SHAP inputs to graph data"""
        if self.sample is None:
            raise ValueError("Original data sample required for SHAP explanations")
            
        device = next(self.model.parameters()).device
        shape  = self.sample.x.shape
        data_list = []
        
        for x_flat in x_flat_list:
            if isinstance(x_flat, np.ndarray):
                x_flat = torch.FloatTensor(x_flat)
            x = x_flat.reshape(shape).to(device)
            
            data = Data(
                x=x,
                edge_index=self.sample.edge_index.clone().to(device),
                batch=torch.zeros(shape[0], dtype=torch.long, device=device)
            )
            data_list.append(data)
            
        batch = Batch.from_data_list(data_list)
        with torch.no_grad():
            return self.model(batch).cpu().numpy()

def disable_inplace_relu(model):
    """Disable inplace operations in the model for SHAP compatibility"""
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

def safe_compute_shap_values(model, background, X_eval):
    """Safely compute SHAP values with proper error handling"""
    try:
        disable_inplace_relu(model)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_eval)
        return shap_values
    except Exception as e:
        print(f"Error computing SHAP values: {str(e)}")
        return None

def run_shap_analysis(args, model, valid_loader):
    print("\n📊 Running SHAP explainability...")
    try:
        # Prepare background and evaluation data
        background_list = []
        for data in valid_loader:
            background_list.append(batch_to_graph(data, args.device))
            if len(background_list) * args.batch_size >= 64:
                break
        
        if not background_list:
            raise ValueError("No validation data available for SHAP analysis")
            
        background = background_list[0]  # Use first batch
        X_eval = Batch.from_data_list([g for g in background.to_data_list()[:10]])  # First 10 samples
        
        # Create wrapper model for SHAP
        sample_data = background[0] if isinstance(background, Batch) else background
        wrapped_model = GNNExplainerWrapper(model, sample_data).to(args.device)
        
        # Compute SHAP values
        shap_values = safe_compute_shap_values(wrapped_model, background, X_eval)
        
        if shap_values is None:
            raise RuntimeError("Failed to compute SHAP values")
            
        # Convert to numpy array if needed
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values)
            
        print(f"SHAP values shape: {shap_values.shape}")
        
        # Prepare evaluation data for visualization
        X_eval_np = torch.cat([g.x for g in X_eval.to_data_list()]).cpu().numpy()
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Save SHAP values
        np.save(os.path.join(args.output, 'shap_values.npy'), shap_values)
        
        # Visualize SHAP values
        visualize_shap_values(shap_values[0], X_eval_np[0], args.output)  # Visualize first sample
        
        # Confusion matrix
        true_labels, pred_labels = [], []
        for data in valid_loader:
            g = batch_to_graph(data, args.device)
            with torch.no_grad():
                preds = model(g).cpu()
                true_labels.extend(g.y.cpu().numpy())
                pred_labels.extend(torch.argmax(preds, dim=1).detach().cpu().numpy())
        
        cm = confusion_matrix(true_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Validation Set)")
        plt.savefig(os.path.join(args.output, "confusion_matrix.png"), dpi=300)
        plt.close()
        
        print("✅ SHAP analysis completed successfully")
        
    except Exception as e:
        print(f"[ERROR] SHAP analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_shap_values(shap_values, explained_data, output_path):
    """Enhanced visualization for SHAP values"""
    # Feature importance plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(shap_values.T, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='SHAP Value')
    plt.xlabel('Node Index')
    plt.ylabel('Feature Index')
    plt.title('Node Feature Importance')
    
    # Global feature importance
    plt.subplot(1, 2, 2)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    plt.bar(range(mean_abs_shap.shape[1]), mean_abs_shap.mean(axis=0))
    plt.xlabel('Feature Index')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Global Feature Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'shap_analysis.png'))
    plt.close()

def validate(args, model, valid_loader, epoch):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for raw in valid_loader:
            g   = batch_to_graph(raw, args.device)
            out = model(g)
            loss_sum += crit(out, g.y).item()
            preds = out.argmax(dim=1)
            total += g.y.size(0)
            correct += (preds == g.y).sum().item()
    acc = 100 * correct / total
    print(f"[Val] Epoch {epoch} — Acc {acc:.2f}%  Loss {loss_sum/len(valid_loader):.4f}")
    
    # Run SHAP analysis at specified frequency
    if args.use_shap and epoch % args.shap_freq == 0:
        run_shap_analysis(args, model, valid_loader)
    
    return acc

def train(args, model, train_loader, valid_loader):
    opt      = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_acc = 0.0
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        run_loss = 0.0
        crit     = nn.CrossEntropyLoss()
        for raw in train_loader:
            g   = batch_to_graph(raw, args.device)
            opt.zero_grad()
            out = model(g)
            loss= crit(out, g.y)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        print(f"[Train] Epoch {epoch} — Loss {run_loss/len(train_loader):.4f}")
        val_acc = validate(args, model, valid_loader, epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
            print(f"[Save] Best model @ Epoch {epoch} — Acc {best_acc:.2f}%")

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    train_loader, valid_loader = create_data_loaders(args)
    model = TemporalGCN(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lstm_hidden=args.lstm_hidden,
        output_dim=args.output_dim
    ).to(args.device)
    train(args, model, train_loader, valid_loader)

if __name__ == '__main__':
    main()
