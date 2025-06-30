import os
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap
import matplotlib.pyplot as plt

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from datautil.getdataloader_single import get_act_dataloader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# placeholder imports for utility functions used in SHAP block
from your_utils import (
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
    parser.add_argument('--dataset',           type=str,   default='emg')
    parser.add_argument('--data_dir',          type=str,   default='./data/')
    parser.add_argument('--output',            type=str,   default='./output')
    parser.add_argument('--test_envs',         type=int,   nargs='+', default=[0])
    parser.add_argument('--algorithm',         type=str,   default='diversify')
    parser.add_argument('--use_gnn',           type=int,   choices=[0,1], default=1)
    parser.add_argument('--enable_shap',       action='store_true')
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
                row = corr[node].clone(); row[node]=0
                topk = torch.topk(row, k=min(max_edges_per_node, C-1)).indices
                for dst in topk: topk_edges.append([node,dst.item()])
            if topk_edges:
                edge_index = torch.tensor(topk_edges,dtype=torch.long).T.to(device)
        feat = x_ts.mean(dim=0).unsqueeze(0).repeat(C,1).to(device)
        data_list.append(Data(x=feat, edge_index=edge_index))
    return data_list

def batch_to_graph(batch, device):
    if isinstance(batch, (list, tuple)):
        x_ts, y = batch[0], batch[1]
    else:
        raise ValueError("Expected batch as tuple/list")
    y = y.to(device).long()
    graphs = build_correlation_graph(x_ts, device=device)
    for g,label in zip(graphs,y): g.y=label
    return Batch.from_data_list(graphs)

class TemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, lstm_hidden, output_dim, dropout=0.2):
        super().__init__()
        self.gcn_layers, self.norms = nn.ModuleList(), nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_dim)); self.norms.append(BatchNorm(hidden_dim))
        for _ in range(num_layers-1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim)); self.norms.append(BatchNorm(hidden_dim))
        self.lstm=nn.LSTM(hidden_dim,lstm_hidden,batch_first=True)
        self.out=nn.Linear(lstm_hidden,output_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self, data):
        x,ei,b = data.x.float(),data.edge_index,data.batch
        for gcn,norm in zip(self.gcn_layers,self.norms):
            x=F.relu(self.dropout(norm(gcn(x,ei))))
        pooled=global_mean_pool(x,b)
        seq=pooled.unsqueeze(1)
        _,(hn,_) = self.lstm(seq)
        return self.out(hn[-1])

class GNNExplainerWrapper(nn.Module):
    def __init__(self, model, sample_data:Data):
        super().__init__()
        self.model, self.sample = model, sample_data
    def forward(self, flat_arrays):
        device=next(self.model.parameters()).device; shape=self.sample.x.shape
        dl=[]
        for arr in flat_arrays:
            x=torch.tensor(arr,dtype=torch.float32,device=device).reshape(shape)
            dl.append(Data(x=x,edge_index=self.sample.edge_index.to(device),
                           batch=torch.zeros(shape[0],dtype=torch.long,device=device)))
        return self.model(Batch.from_data_list(dl)).detach().cpu().numpy()

def visualize_shap_values(shap_values, explained_data):
    plt.figure(figsize=(10,6))
    plt.imshow(shap_values.T,cmap='coolwarm',aspect='auto')
    plt.colorbar(label='SHAP Value')
    plt.xlabel('Node Index'); plt.ylabel('Feature Index'); plt.title('Node Feature Importance')
    plt.savefig('shap_values.png'); plt.close()
    mean_abs_shap=np.mean(np.abs(shap_values),axis=0)
    plt.figure(figsize=(10,4))
    plt.bar(range(mean_abs_shap.shape[0]),mean_abs_shap)
    plt.xlabel('Feature Index'); plt.ylabel('Mean |SHAP|')
    plt.title('Global Feature Importance')
    plt.savefig('global_shap.png'); plt.close()

def explain_gnn_with_shap(model, valid_loader, device, sample_count=5, nsamples=50):
    model.eval()
    raw=next(iter(valid_loader)); g=batch_to_graph(raw,device)
    dl=g.to_data_list(); anchor=dl[0]
    wrapper=GNNExplainerWrapper(model,anchor).to(device)
    bg=np.stack([d.x.cpu().numpy().flatten() for d in dl[:sample_count]],axis=0)
    expl=shap.KernelExplainer(wrapper,bg)
    tgt=anchor.x.cpu().numpy().flatten()[None,:]
    so=expl.shap_values(tgt,nsamples=nsamples)
    with torch.no_grad():
        pred=model(Batch.from_data_list([anchor]).to(device)).argmax(dim=1).item()
    if isinstance(so,list):
        flat=so[pred][0]
    else:
        ar=so[0]; M=anchor.x.numel(); K=ar.size//M
        flat=ar.reshape(K,M)[pred]
    return flat.reshape(anchor.x.shape),anchor

def validate(args,model,valid_loader,epoch):
    model.eval(); crit=nn.CrossEntropyLoss()
    tot,cor,ls=0,0,0.0
    with torch.no_grad():
        for raw in valid_loader:
            g=batch_to_graph(raw,args.device)
            out=model(g); ls+=crit(out,g.y).item()
            p=out.argmax(dim=1)
            tot+=g.y.size(0); cor+=(p==g.y).sum().item()
    acc=100*cor/tot
    print(f"[Val] Epoch {epoch} — Acc {acc:.2f}%  Loss {ls/len(valid_loader):.4f}")
    return acc

def train(args,model,train_loader,valid_loader):
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=0.0
    GNN_AVAILABLE=True
    for epoch in range(1,args.max_epoch+1):
        model.train(); rl=0.0; crit=nn.CrossEntropyLoss()
        for raw in train_loader:
            g=batch_to_graph(raw,args.device)
            opt.zero_grad(); out=model(g)
            loss=crit(out,g.y); loss.backward(); opt.step()
            rl+=loss.item()
        print(f"[Train] Epoch {epoch} — Loss {rl/len(train_loader):.4f}")
        acc=validate(args,model,valid_loader,epoch)
        if acc>best:
            best=acc; torch.save(model.state_dict(),'best_model.pth')
            print(f"[Save] Best model @ Epoch {epoch} — Acc {best:.2f}%")

        # ======================= SHAP EXPLAINABILITY =======================
        if args.enable_shap:
            print("\n📊 Running SHAP explainability...")
            try:
                if args.use_gnn and GNN_AVAILABLE:
                    bl=[] 
                    for data in valid_loader:
                        bl.append(data)
                        if len(bl)*args.batch_size>=64: break
                    background=bl[0]; X_eval=background[:10]
                else:
                    background=get_background_batch(valid_loader,size=64).cuda()
                    X_eval=background[:10]
                disable_inplace_relu(model)
                tf=transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
                if tf is not None:
                    background=tf(background); X_eval=tf(X_eval)
                se=safe_compute_shap_values(model,background,X_eval)
                shap_vals=se.values
                X_np=X_eval.detach().cpu().numpy()
                if args.use_gnn and GNN_AVAILABLE:
                    if shap_vals.ndim==4:
                        shap_vals=np.abs(shap_vals).sum(axis=-1)
                    if shap_vals.ndim==3:
                        shap_vals=np.transpose(shap_vals,(0,2,1))[...,None]
                        X_np=np.transpose(X_np,(0,2,1))[...,None]
                os.makedirs(args.output,exist_ok=True)
                try:
                    plot_summary(shap_vals,X_np,output_path=os.path.join(args.output,"shap_summary.png"))
                except IndexError:
                    plot_emg_shap_4d(X_eval,shap_vals,output_path=os.path.join(args.output,"shap_3d_fallback.html"))
                overlay_signal_with_shap(X_np[0],shap_vals,output_path=os.path.join(args.output,"shap_overlay.png"))
                plot_shap_heatmap(shap_vals,output_path=os.path.join(args.output,"shap_heatmap.png"))
                bp,mp,ad=evaluate_shap_impact(model,X_eval,shap_vals)
                save_shap_numpy(shap_vals,save_path=os.path.join(args.output,"shap_values.npy"))
                print(f"[SHAP] Accuracy Drop: {ad:.4f}")
                print(f"[SHAP] Flip Rate: {compute_flip_rate(bp,mp):.4f}")
                print(f"[SHAP] Confidence Δ: {compute_confidence_change(bp,mp):.4f}")
                print(f"[SHAP] AOPC: {compute_aopc(model,X_eval,shap_vals):.4f}")
                mets=evaluate_advanced_shap_metrics(shap_vals,X_eval)
                print(f"[SHAP] Entropy: {mets.get('shap_entropy',0):.4f}")
                print(f"[SHAP] Coherence: {mets.get('feature_coherence',0):.4f}")
                print(f"[SHAP] Channel Variance: {mets.get('channel_variance',0):.4f}")
                print(f"[SHAP] Temporal Entropy: {mets.get('temporal_entropy',0):.4f}")
                print(f"[SHAP] Mutual Info: {mets.get('mutual_info',0):.4f}")
                print(f"[SHAP] PCA Alignment: {mets.get('pca_alignment',0):.4f}")
                sa=_get_shap_array(shap_vals)
                if len(sa)>=2:
                    s1,s2=sa[0],sa[1]
                    print(f"[SHAP] Jaccard (top-10): {compute_jaccard_topk(s1,s2,k=10):.4f}")
                    print(f"[SHAP] Kendall's Tau: {compute_kendall_tau(s1,s2):.4f}")
                    print(f"[SHAP] Cosine Similarity: {cosine_similarity_shap(s1,s2):.4f}")
                true_l,pred_l=[],[]
                for data in valid_loader:
                    if args.use_gnn and GNN_AVAILABLE:
                        x,y=data[0].to(args.device),data[1]
                    else:
                        x,y=data[0].to(args.device).float(),data[1]
                    with torch.no_grad():
                        if args.use_gnn and GNN_AVAILABLE:
                            x=transform_for_gnn(x)
                        pr=model(x).cpu()
                        true_l.extend(y.cpu().numpy())
                        pred_l.extend(torch.argmax(pr,dim=1).cpu().numpy())
                cm=confusion_matrix(true_l,pred_l)
                disp=ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap="Blues")
                plt.title("Confusion Matrix")
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
