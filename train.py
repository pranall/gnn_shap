import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from datautil.getdataloader_single import get_act_dataloader
from shap_utils import (
    get_background_batch, disable_inplace_relu, transform_for_gnn,
    safe_compute_shap_values, plot_summary, plot_emg_shap_4d,
    overlay_signal_with_shap, plot_shap_heatmap, evaluate_shap_impact,
    save_shap_numpy, compute_flip_rate, compute_confidence_change,
    compute_aopc, evaluate_advanced_shap_metrics, _get_shap_array,
    compute_jaccard_topk, compute_kendall_tau, cosine_similarity_shap,
    plot_4d_shap_surface
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Diversify+GNN with SHAP")
    parser.add_argument('--dataset',     type=str, default='emg')
    parser.add_argument('--data_dir',    type=str, default='./data/')
    parser.add_argument('--output',      type=str, default='./output')
    parser.add_argument('--test_envs',   type=int, nargs='+', default=[0])
    parser.add_argument('--algorithm',   type=str, default='diversify')
    parser.add_argument('--use_gnn',     type=int, choices=[0,1], default=1)
    parser.add_argument('--enable_shap', action='store_true',help='Run SHAP explainability')
    parser.add_argument('--task',        type=str, default='cross_people')
    parser.add_argument('--latent_domain_num', type=int, default=5)
    parser.add_argument('--alpha1',      type=float, default=1.0)
    parser.add_argument('--alpha',       type=float, default=10.0)
    parser.add_argument('--lam',         type=float, default=0.0)
    parser.add_argument('--local_epoch', type=int, default=2)
    parser.add_argument('--in_channels', type=int, default=8)
    parser.add_argument('--hidden_dim',  type=int, default=64)
    parser.add_argument('--num_layers',  type=int, default=2)
    parser.add_argument('--lstm_hidden', type=int, default=64)
    parser.add_argument('--output_dim',  type=int, default=6)
    parser.add_argument('--lr',          type=float, default=0.01)
    parser.add_argument('--weight_decay',type=float, default=0.0005)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--max_epoch',   type=int, default=1)
    parser.add_argument('--device',      type=str, choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--seed',        type=int, default=0)
    parser.add_argument('--N_WORKERS',   type=int, default=4)
    parser.add_argument('--steps_per_epoch', type=int, default=int(1e9))
    parser.add_argument('--use_shap',    type=int, choices=[0,1], default=1)
    parser.add_argument('--shap_freq',   type=int, default=1)
    return parser.parse_args()

def create_data_loaders(args):
    args.act_people = {
        'emg':   [list(range(0,9)), list(range(9,18)), list(range(18,27)), list(range(27,36))],
        'dsads':[list(range(i*8, i*8+8)) for i in range(8)]
    }
    train_loader, _, valid_loader, _, _, _, _ = get_act_dataloader(args)
    return train_loader, valid_loader

def build_correlation_graph(batch_ts, threshold=0.3, self_loops=True,
                            max_edges_per_node=None, device='cpu'):
    if batch_ts.dim() == 4:
        batch_ts = batch_ts.squeeze(2)
    B, C, T = batch_ts.shape
    data_list = []
    for i in range(B):
        x_ts = batch_ts[i].t().float()
        corr = torch.corrcoef(x_ts.T).abs()
        ei = (corr > threshold).nonzero(as_tuple=False).T.to(device)
        if not self_loops:
            mask = ei[0] != ei[1]
            ei = ei[:, mask]
        if max_edges_per_node is not None:
            topk_edges = []
            for node in range(C):
                row = corr[node].clone(); row[node]=0
                topk = torch.topk(row, k=min(max_edges_per_node, C-1)).indices
                for dst in topk: topk_edges.append([node, dst.item()])
            if topk_edges:
                ei = torch.tensor(topk_edges, dtype=torch.long).T.to(device)
        feat = x_ts.mean(dim=0).unsqueeze(0).repeat(C,1).to(device)
        data_list.append(Data(x=feat, edge_index=ei))
    return data_list

def batch_to_graph(batch, device):
    if isinstance(batch, (list,tuple)):
        x_ts, y = batch[0], batch[1]
    else:
        raise ValueError("Expected batch as tuple/list")
    y = y.to(device).long()
    graphs = build_correlation_graph(x_ts, device=device)
    for g,label in zip(graphs,y):
        g.y = label
    return Batch.from_data_list(graphs)

class TemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, lstm_hidden, output_dim, dropout=0.2):
        super().__init__()
        self.gcn_layers = nn.ModuleList(); self.norms = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_dim)); self.norms.append(BatchNorm(hidden_dim))
        for _ in range(num_layers-1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim)); self.norms.append(BatchNorm(hidden_dim))
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.out  = nn.Linear(lstm_hidden, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, data):
        x, ei, b = data.x.float(), data.edge_index, data.batch
        for gcn,norm in zip(self.gcn_layers,self.norms):
            x = F.relu(self.dropout(norm(gcn(x,ei))))
        pooled = global_mean_pool(x,b)
        seq = pooled.unsqueeze(1)
        _,(hn,_) = self.lstm(seq)
        return self.out(hn[-1])

class GNNExplainerWrapper(nn.Module):
    def __init__(self, model, sample_data:Data):
        super().__init__()
        self.model = model; self.sample = sample_data
    def forward(self, flat_arrays):
        device = next(self.model.parameters()).device; shape=self.sample.x.shape
        dl = []
        for arr in flat_arrays:
            x = torch.tensor(arr, dtype=torch.float32, device=device).reshape(shape)
            dl.append(Data(x=x, edge_index=self.sample.edge_index.to(device),
                           batch=torch.zeros(shape[0],dtype=torch.long,device=device)))
        return self.model(Batch.from_data_list(dl)).detach().cpu().numpy()

def visualize_shap_values(shap_values, explained_data):
    plt.figure(figsize=(10,6))
    plt.imshow(shap_values.T, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='SHAP Value')
    plt.xlabel('Node Index'); plt.ylabel('Feature Index')
    plt.title('Node Feature Importance')
    plt.savefig('shap_values.png'); plt.close()
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    plt.figure(figsize=(10,4))
    plt.bar(range(mean_abs.shape[0]), mean_abs)
    plt.xlabel('Feature Index'); plt.ylabel('Mean |SHAP|')
    plt.title('Global Feature Importance')
    plt.savefig('global_shap.png'); plt.close()

def explain_gnn_with_shap(model, valid_loader, device, sample_count=5, nsamples=50):
    model.eval()
    raw = next(iter(valid_loader)); g = batch_to_graph(raw,device)
    dl = g.to_data_list(); anchor = dl[0]
    wrap = GNNExplainerWrapper(model,anchor).to(device)
    bg = np.stack([d.x.cpu().numpy().flatten() for d in dl[:sample_count]],axis=0)
    expl=shap.KernelExplainer(wrap,bg)
    tgt = anchor.x.cpu().numpy().flatten()[None,:]
    out=expl.shap_values(tgt,nsamples=nsamples)
    with torch.no_grad():
        pred = model(Batch.from_data_list([anchor]).to(device)).argmax(dim=1).item()
    if isinstance(out,list):
        flat=out[pred][0]
    else:
        arr=out[0]; M=anchor.x.numel(); K=arr.size//M
        flat=arr.reshape(K,M)[pred]
    return flat.reshape(anchor.x.shape), anchor

def validate(args, model, valid_loader, epoch):
    model.eval(); crit=nn.CrossEntropyLoss(); tot,cor,ls=0,0,0.0
    with torch.no_grad():
        for raw in valid_loader:
            g=batch_to_graph(raw,args.device); o=model(g); ls+=crit(o,g.y).item()
            p=o.argmax(dim=1); tot+=g.y.size(0); cor+=(p==g.y).sum().item()
    acc=100*cor/tot
    print(f"[Val] Epoch {epoch} — Acc {acc:.2f}%  Loss {ls/len(valid_loader):.4f}")
    return acc

def train(args, model, train_loader, valid_loader):
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best_acc=0.0; GNN_AVAILABLE=True
    for epoch in range(1,args.max_epoch+1):
        model.train(); rl=0.0; crit=nn.CrossEntropyLoss()
        for raw in train_loader:
            g=batch_to_graph(raw,args.device)
            opt.zero_grad(); out=model(g)
            loss=crit(out,g.y); loss.backward(); opt.step()
            rl+=loss.item()
        print(f"[Train] Epoch {epoch} — Loss {rl/len(train_loader):.4f}")
        val_acc = validate(args, model, valid_loader, epoch)
        if val_acc>best_acc:
            best_acc=val_acc; torch.save(model.state_dict(),'best_model.pth')
            print(f"[Save] Best model @ Epoch {epoch} — Acc {best_acc:.2f}%")

        # ======================= SHAP EXPLAINABILITY =======================
        if getattr(args, 'enable_shap', False):
            print("\n📊 Running SHAP explainability...")
            try:
                if args.use_gnn and GNN_AVAILABLE:
                    background_list=[]
                    for data in valid_loader:
                        background_list.append(data)
                        if len(background_list)*args.batch_size>=64: break
                    background=background_list[0]
                    X_eval=background[:10]
                else:
                    background=get_background_batch(valid_loader,size=64).cuda()
                    X_eval=background[:10]
                disable_inplace_relu(model)
                transform_fn = transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
                if transform_fn is not None:
                    background=transform_fn(background)
                    X_eval=transform_fn(X_eval)
                shap_explanation = safe_compute_shap_values(model, background, X_eval)
                shap_vals = shap_explanation.values
                print(f"SHAP values shape: {shap_vals.shape}")
                X_eval_np = X_eval.detach().cpu().numpy()
                if args.use_gnn and GNN_AVAILABLE:
                    print(f"Original SHAP values shape: {shap_vals.shape}")
                    print(f"Original X_eval shape: {X_eval_np.shape}")
                    if shap_vals.ndim==4:
                        shap_vals=np.abs(shap_vals).sum(axis=-1)
                        print(f"SHAP values after class sum: {shap_vals.shape}")
                    if shap_vals.ndim==3:
                        shap_vals=np.transpose(shap_vals,(0,2,1)); shap_vals=np.expand_dims(shap_vals,axis=2)
                        X_eval_np=np.transpose(X_eval_np,(0,2,1)); X_eval_np=np.expand_dims(X_eval_np,axis=2)
                    else:
                        print(f"⚠️ Unexpected SHAP values dimension: {shap_vals.ndim}")
                        print("Skipping visualization-specific reshaping")
                os.makedirs(args.output,exist_ok=True)
                try:
                    plot_summary(shap_vals, X_eval_np, output_path=os.path.join(args.output,"shap_summary.png"))
                except IndexError as e:
                    print(f"SHAP summary plot dimension error: {e}")
                    print("Using fallback 3D visualization instead")
                    plot_emg_shap_4d(X_eval, shap_vals, output_path=os.path.join(args.output,"shap_3d_fallback.html"))
                overlay_signal_with_shap(X_eval_np[0], shap_vals, output_path=os.path.join(args.output,"shap_overlay.png"))
                plot_shap_heatmap(shap_vals, output_path=os.path.join(args.output,"shap_heatmap.png"))
                base_preds, masked_preds, acc_drop = evaluate_shap_impact(model, X_eval, shap_vals)
                save_path=os.path.join(args.output,"shap_values.npy")
                save_shap_numpy(shap_vals,save_path=save_path)
                print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
                print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds,masked_preds):.4f}")
                print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds,masked_preds):.4f}")
                print(f"[SHAP] AOPC: {compute_aopc(model, X_eval, shap_vals):.4f}")
                metrics = evaluate_advanced_shap_metrics(shap_vals, X_eval)
                print(f"[SHAP] Entropy: {metrics.get('shap_entropy',0):.4f}")
                print(f"[SHAP] Coherence: {metrics.get('feature_coherence',0):.4f}")
                print(f"[SHAP] Channel Variance: {metrics.get('channel_variance',0):.4f}")
                print(f"[SHAP] Temporal Entropy: {metrics.get('temporal_entropy',0):.4f}")
                print(f"[SHAP] Mutual Info: {metrics.get('mutual_info',0):.4f}")
                print(f"[SHAP] PCA Alignment: {metrics.get('pca_alignment',0):.4f}")
                shap_array=_get_shap_array(shap_vals)
                if len(shap_array)>=2:
                    sample1, sample2 = shap_array[0], shap_array[1]
                    print(f"[SHAP] Jaccard (top-10): {compute_jaccard_topk(sample1, sample2, k=10):.4f}")
                    print(f"[SHAP] Kendall's Tau: {compute_kendall_tau(sample1, sample2):.4f}")
                    print(f"[SHAP] Cosine Similarity: {cosine_similarity_shap(sample1, sample2):.4f}")
                else:
                    print("[SHAP] Not enough samples for similarity metrics")
                plot_emg_shap_4d(X_eval, shap_vals, output_path=os.path.join(args.output,"shap_4d_scatter.html"))
                plot_4d_shap_surface(shap_vals, output_path=os.path.join(args.output,"shap_4d_surface.html"))
                true_labels, pred_labels = [], []
                for data in valid_loader:
                    if args.use_gnn and GNN_AVAILABLE:
                        x = data[0].to(args.device); y = data[1]
                    else:
                        x = data[0].to(args.device).float(); y = data[1]
                    with torch.no_grad():
                        if args.use_gnn and GNN_AVAILABLE:
                            x = transform_for_gnn(x)
                        preds = model(x).cpu()
                        true_labels.extend(y.cpu().numpy())
                        pred_labels.extend(torch.argmax(preds,dim=1).cpu().numpy())
                cm=confusion_matrix(true_labels,pred_labels)
                disp=ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap="Blues")
                plt.title("Confusion Matrix (Validation Set)")
                plt.savefig(os.path.join(args.output,"confusion_matrix.png"),dpi=300)
                plt.close()
                print("✅ SHAP analysis completed successfully")
            except Exception as e:
                print(f"[ERROR] SHAP analysis failed: {e}")
                import traceback; traceback.print_exc()
        # ======================= END SHAP SECTION =======================

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
