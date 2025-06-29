import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from datautil.getdataloader_single import get_act_dataloader
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Diversify+GNN with SHAP")

    # Diversify / dataset flags
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

    # Model / training flags
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

    # Repro & loader tuning
    parser.add_argument('--seed',              type=int,   default=0)
    parser.add_argument('--N_WORKERS',         type=int,   default=4)
    parser.add_argument('--steps_per_epoch',   type=int,   default=int(1e9))

    # SHAP flags
    parser.add_argument('--use_shap',          type=int,   choices=[0,1], default=1)
    parser.add_argument('--shap_freq',         type=int,   default=1)

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------
def create_data_loaders(args):
    # Diversify expects this mapping for ActList
    args.act_people = {
        'emg':   [list(range(0,9)), list(range(9,18)), list(range(18,27)), list(range(27,36))],
        'dsads':[list(range(i*8, i*8+8)) for i in range(8)]
    }
    train_loader, _, valid_loader, _, _, _, _ = get_act_dataloader(args)
    return train_loader, valid_loader

# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# **FIXED**: batch → PyG Batch with correct dtypes
# -----------------------------------------------------------------------------
def batch_to_graph(batch, device):
    # Expect batch as (x_ts, y, …)
    if isinstance(batch, (list, tuple)):
        x_ts, y = batch[0], batch[1]
    else:
        raise ValueError("Expected batch as tuple/list")

    # Move and cast target to LongTensor on device
    y = y.to(device).long()

    # Build graph list
    graphs = build_correlation_graph(x_ts, device=device)
    for g, label in zip(graphs, y):
        g.y = label  # now a LongTensor

    return Batch.from_data_list(graphs)

# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------
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
        x = data.x.float()  # ensure float32
        ei = data.edge_index
        b  = data.batch

        for gcn, norm in zip(self.gcn_layers, self.norms):
            x = F.relu(self.dropout(norm(gcn(x, ei))))

        pooled = global_mean_pool(x, b)
        seq    = pooled.unsqueeze(1)
        _, (hn, _) = self.lstm(seq)
        return self.out(hn[-1])

class GNNExplainerWrapper(nn.Module):
    """Wraps TemporalGCN for SHAP: flattens & rebuilds graphs."""
    def __init__(self, model, sample_data: Data):
        super().__init__()
        self.model  = model
        self.sample = sample_data

    def forward(self, flat_arrays):
        device = next(self.model.parameters()).device
        shape  = self.sample.x.shape
        data_list = []
        for arr in flat_arrays:
            x = torch.tensor(arr, dtype=torch.float32, device=device).reshape(shape)
            data_list.append(Data(
                x=x,
                edge_index=self.sample.edge_index.to(device),
                batch=torch.zeros(shape[0], dtype=torch.long, device=device)
            ))
        return self.model(Batch.from_data_list(data_list)).detach().cpu().numpy()

# -----------------------------------------------------------------------------
# SHAP & validation
# -----------------------------------------------------------------------------
def visualize_shap_values(shap_vals, data):
    # TODO: your plotting/saving code here
    pass


def explain_gnn_with_shap(model, valid_loader, device, sample_count=5, nsamples=50):
    model.eval()

    # 1) grab one raw batch and turn it into a PyG Batch
    raw = next(iter(valid_loader))
    g   = batch_to_graph(raw, device)

    # 2) get individual graphs & pick the first as our “anchor”
    data_list = g.to_data_list()
    anchor    = data_list[0]

    # 3) wrap the model so SHAP can call it on flat arrays
    wrapper = GNNExplainerWrapper(model, anchor).to(device)

    # 4) build the background matrix of flattened features
    background = np.stack([
        d.x.cpu().numpy().flatten()
        for d in data_list[:sample_count]
    ], axis=0)

    # 5) instantiate SHAP with only the wrapper & background
    explainer = shap.KernelExplainer(wrapper, background)

    # 6) flatten the same anchor to get its SHAP values
    target = anchor.x.cpu().numpy().flatten()[None, :]

    # 7) get SHAP values (this might return a list *or* a single big array)
    shap_out = explainer.shap_values(target, nsamples=nsamples)

    # 8) Determine which class we care about (we’ll pick the model’s top prediction)
    with torch.no_grad():
        pred = model(Batch.from_data_list([anchor]).to(device)).argmax(dim=1).item()

    # 9) Extract & reshape the correct slice
    if isinstance(shap_out, list):
        # shap_out[class] is shape (1, num_features)
        flat_shap = shap_out[pred][0]
    else:
        # shap_out is an array of shape (1, num_features * num_classes)
        flat_arr = shap_out[0]
        M = anchor.x.numel()              # num_features (e.g. 64)
        K = flat_arr.size // M            # num_classes (e.g. 6)
        reshaped = flat_arr.reshape(K, M) # [num_classes, num_features]
        flat_shap = reshaped[pred]        # pick the row for our class

    # 10) finally, restore the (nodes, features) layout
    shap_vals = flat_shap.reshape(anchor.x.shape)
    return shap_vals, anchor


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

    if args.use_shap and epoch % args.shap_freq == 0:
        #print("[SHAP] computing explanations…")
        shap_vals, anchor = explain_gnn_with_shap(model, valid_loader, args.device)
        #print(f"[SHAP] values shape: {shap_vals.shape}")
        visualize_shap_values(shap_vals, anchor)

    return acc

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
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
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"[Save] Best model @ Epoch {epoch} — Acc {best_acc:.2f}%")

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

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
