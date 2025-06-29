import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.loader import DataLoader as GeoDataLoader
import numpy as np
import shap

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Diversify+GNN with optional SHAP and Diversify DataLoader")

    # Diversify / dataset flags
    parser.add_argument('--dataset',           type=str,   default='emg',
                        help="Which dataset to run (e.g. emg, dsads)")
    parser.add_argument('--data_dir',          type=str,   default='./data/',
                        help="Path to your data folder")
    parser.add_argument('--test_envs',         type=int,   nargs='+', default=[0],
                        help="Which env(s) to hold out as test")
    parser.add_argument('--algorithm',         type=str,   default='diversify',
                        help="Domain generalization algorithm")
    parser.add_argument('--use_gnn',           type=int,   choices=[0,1], default=1,
                        help="1 to enable GNN, 0 to disable")
    parser.add_argument('--task',              type=str,   default='cross_people',
                        help="Diversify task for loader")
    parser.add_argument('--latent_domain_num', type=int,   default=5,
                        help="Number of latent domains")
    parser.add_argument('--alpha1',            type=float, default=1.0,
                        help="Diversify α1")
    parser.add_argument('--alpha',             type=float, default=10.0,
                        help="Diversify α")
    parser.add_argument('--lam',               type=float, default=0.0,
                        help="Diversify λ")
    parser.add_argument('--local_epoch',       type=int,   default=2,
                        help="Local adaptation epochs")

    # Model/training flags
    parser.add_argument('--in_channels',       type=int,   default=8,
                        help="Input channels")
    parser.add_argument('--hidden_dim',        type=int,   default=64,
                        help="GCN hidden dim")
    parser.add_argument('--num_layers',        type=int,   default=2,
                        help="Number of GCN layers")
    parser.add_argument('--lstm_hidden',       type=int,   default=64,
                        help="LSTM hidden dim")
    parser.add_argument('--output_dim',        type=int,   default=6,
                        help="Output classes")
    parser.add_argument('--lr',                type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument('--weight_decay',      type=float, default=0.0005,
                        help="Weight decay")
    parser.add_argument('--batch_size',        type=int,   default=32,
                        help="Batch size")
    parser.add_argument('--max_epoch',         type=int,   default=1,
                        help="Epochs")
    parser.add_argument('--device',            type=str,   choices=['cpu','cuda'], default='cuda',
                        help="Device")
    parser.add_argument('--seed',              type=int,   default=0,
                        help="Random seed")
    parser.add_argument('--N_WORKERS',         type=int,   default=2,
                        help="DataLoader workers")
    parser.add_argument('--steps_per_epoch',   type=int,   default=int(1e9),
                        help="Cap on steps-per-epoch for Diversify loader")

    # SHAP flags
    parser.add_argument('--use_shap',          type=int, choices=[0,1], default=1,
                        help="Enable SHAP in validation")
    parser.add_argument('--shap_freq',         type=int, default=1,
                        help="SHAP freq")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Build DataLoaders
# -----------------------------------------------------------------------------
def create_data_loaders(args):
    args.act_people = {
        'emg': [list(range(0,9)), list(range(9,18)), list(range(18,27)), list(range(27,36))],
        'dsads': [list(range(i*8, i*8+8)) for i in range(8)]
    }
    from datautil.getdataloader_single import get_act_dataloader
    tr, tr_noshuf, val, tar, ds_tr, ds_val, ds_tar = get_act_dataloader(args)
    return tr, val

# -----------------------------------------------------------------------------
# Graph Builder
# -----------------------------------------------------------------------------
def build_correlation_graph(batch_time_series, threshold=0.3, self_loops=True, device='cpu'):
    data_list = []
    batch_size, channels, timesteps = batch_time_series.shape
    for i in range(batch_size):
        ts = batch_time_series[i].T.float()  # [timesteps, channels]
        corr = torch.corrcoef(ts.T).abs()
        edges = (corr > threshold).nonzero(as_tuple=False).T
        if not self_loops:
            mask = edges[0] != edges[1]
            edges = edges[:, mask]
        # node features: mean over time
        x = ts.mean(dim=0).unsqueeze(0).repeat(channels, 1)
        data = Data(x=x.to(device), edge_index=edges.to(device))
        data_list.append(data)
    return data_list

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class TemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, lstm_hidden, output_dim, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.norms.append(BatchNorm(hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(self.drop(norm(conv(x, edge_index))))
        pooled = global_mean_pool(x, batch)
        out, (hn, _) = self.lstm(pooled.unsqueeze(1))
        return self.fc(hn[-1])

# -----------------------------------------------------------------------------
# Batch to Graph helper
# -----------------------------------------------------------------------------
def batch_to_graph(batch, device):
    if isinstance(batch, (list, tuple)):
        x, y = batch[0], batch[1]
    else:
        return batch.to(device)
    x, y = x.to(device), y.to(device)
    graphs = build_correlation_graph(x, device=device)
    for g, label in zip(graphs, y):
        g.y = label.to(g.x.device)
    batch_graph = Batch.from_data_list(graphs)
    return batch_graph

# -----------------------------------------------------------------------------
# SHAP & Validation
# -----------------------------------------------------------------------------
def visualize_shap_values(vals, anchor):
    # implement matplotlib overlay
    pass


def explain_gnn_with_shap(model, loader, device, sample_count=5, nsamples=50):
    model.eval()
    batch = next(iter(loader))
    graphs = batch_to_graph(batch, device)
    data_list = graphs.to_data_list()
    anchor = data_list[0]
    wrapper = GNNExplainerWrapper(model, anchor).to(device)
    bg = np.stack([d.x.cpu().numpy().flatten() for d in data_list[:sample_count]])
    expl = shap.KernelExplainer(wrapper, bg, keep_index=True)
    tgt = anchor.x.cpu().numpy().flatten()[None,:]
    vals = expl.shap_values(tgt, nsamples=nsamples)
    return vals[0].reshape(anchor.x.shape), anchor


def validate(args, model, loader, epoch):
    model.eval()
    tot, corr, ls = 0,0,0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            bg = batch_to_graph(batch, args.device)
            out = model(bg)
            loss = crit(out, bg.y)
            ls += loss.item()
            pred = out.argmax(dim=1)
            corr += (pred==bg.y).sum().item()
            tot += bg.y.size(0)
    acc = 100*corr/tot
    print(f"[Val] Epoch {epoch} Acc: {acc:.2f}% Loss: {ls/len(loader):.4f}")
    if args.use_shap and epoch%args.shap_freq==0:
        vals, anc = explain_gnn_with_shap(model, loader, args.device)
        print(f"SHAP shape: {vals.shape}")
        visualize_shap_values(vals, anc)
    return acc

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(args, model, train_loader, valid_loader):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best=0.0
    for ep in range(1, args.max_epoch+1):
        model.train(); rl=0.0; crit=nn.CrossEntropyLoss()
        for batch in train_loader:
            bg=batch_to_graph(batch,args.device)
            opt.zero_grad()
            out=model(bg)
            loss=crit(out,bg.y)
            loss.backward(); opt.step(); rl+=loss.item()
        print(f"[Train] Ep {ep} Loss: {rl/len(train_loader):.4f}")
        va=validate(args,model,valid_loader,ep)
        if va>best:
            best=va; torch.save(model.state_dict(), 'best_model.pth'); print(f"Saved Ep{ep} Acc{best:.2f}%")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args=parse_args(); torch.manual_seed(args.seed)
    tr,va = create_data_loaders(args)
    model=TemporalGCN(args.in_channels,args.hidden_dim,args.num_layers,args.lstm_hidden,args.output_dim).to(args.device)
    train(args,model,tr,va)

if __name__=='__main__': main()
