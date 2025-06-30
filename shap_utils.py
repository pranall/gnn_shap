# shap_utils.py

import os
import shap
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from datautil.getdataloader_single import get_act_dataloader  # if needed elsewhere

class GNNWrapper(torch.nn.Module):
    def __init__(self, model, original_data_sample=None):
        super().__init__()
        self.model = model
        self.original_data = original_data_sample

    def forward(self, inputs):
        if isinstance(inputs, (list, np.ndarray)):
            return self._explain_forward(inputs)
        return self.model(inputs)

    def _explain_forward(self, x_flat_list):
        if self.original_data is None:
            raise ValueError("Original data sample required for SHAP explanations")
        device = next(self.model.parameters()).device
        data_list = []
        for x_flat in x_flat_list:
            x = torch.FloatTensor(x_flat.reshape(self.original_data.x.shape)).to(device)
            data = Data(
                x=x,
                edge_index=self.original_data.edge_index.clone().to(device),
                batch=torch.zeros(x.size(0), dtype=torch.long).to(device)
            )
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        return self.model(batch)

def explain_gnn_with_shap(gnn_model, data_loader, device='cuda', sample_count=5):
    gnn_model.eval()
    try:
        sample_batch = next(iter(data_loader)).to(device)
        if isinstance(sample_batch, Batch):
            sample_data = sample_batch[0]
            background = [d.x.cpu().numpy().flatten() for d in sample_batch.to_data_list()[:sample_count]]
        else:
            sample_data = sample_batch
            background = [sample_batch.x.cpu().numpy().flatten()]
        wrapped_model = GNNWrapper(gnn_model, sample_data).to(device)
        explainer = shap.KernelExplainer(model=wrapped_model, data=np.array(background))
        sample_to_explain = background[0]
        shap_values = explainer.shap_values(sample_to_explain, nsamples=50)
        return shap_values.reshape(sample_data.x.shape), sample_batch
    except Exception as e:
        print(f"[SHAP Error] {e}")
        return None, None

def run_shap_explainability(args, algorithm, valid_loader, GNN_AVAILABLE):
    if not getattr(args, 'enable_shap', False):
        return
    try:
        if args.use_gnn and GNN_AVAILABLE:
            background_list = []
            for data in valid_loader:
                background_list.append(data)
                if len(background_list) * args.batch_size >= 64:
                    break
            background = background_list[0]
            X_eval = background[:10]
        else:
            from your_utils import get_background_batch
            background = get_background_batch(valid_loader, size=64).cuda()
            X_eval = background[:10]
        from your_utils import disable_inplace_relu, transform_for_gnn, safe_compute_shap_values
        disable_inplace_relu(algorithm)
        transform_fn = transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
        if transform_fn is not None:
            background = transform_fn(background)
            X_eval = transform_fn(X_eval)
        shap_explanation = safe_compute_shap_values(algorithm, background, X_eval)
        shap_vals = shap_explanation.values
        X_eval_np = X_eval.detach().cpu().numpy()
        if args.use_gnn and GNN_AVAILABLE:
            if shap_vals.ndim == 4:
                shap_vals = np.abs(shap_vals).sum(axis=-1)
            if shap_vals.ndim == 3:
                shap_vals = np.transpose(shap_vals, (0,2,1))
                shap_vals = np.expand_dims(shap_vals, axis=2)
                X_eval_np = np.transpose(X_eval_np, (0,2,1))
                X_eval_np = np.expand_dims(X_eval_np, axis=2)
        from your_utils import plot_summary, plot_emg_shap_4d, overlay_signal_with_shap, plot_shap_heatmap
        os.makedirs(args.output, exist_ok=True)
        try:
            plot_summary(shap_vals, X_eval_np, output_path=os.path.join(args.output, "shap_summary.png"))
        except IndexError:
            plot_emg_shap_4d(X_eval, shap_vals, output_path=os.path.join(args.output, "shap_3d_fallback.html"))
        overlay_signal_with_shap(X_eval_np[0], shap_vals, output_path=os.path.join(args.output, "shap_overlay.png"))
        plot_shap_heatmap(shap_vals, output_path=os.path.join(args.output, "shap_heatmap.png"))
        from your_utils import evaluate_shap_impact, save_shap_numpy, compute_flip_rate, compute_confidence_change, compute_aopc, evaluate_advanced_shap_metrics, _get_shap_array, compute_jaccard_topk, compute_kendall_tau, cosine_similarity_shap, plot_4d_shap_surface
        base_preds, masked_preds, acc_drop = evaluate_shap_impact(algorithm, X_eval, shap_vals)
        save_shap_numpy(shap_vals, save_path=os.path.join(args.output, "shap_values.npy"))
        print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
        print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds, masked_preds):.4f}")
        print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds, masked_preds):.4f}")
        print(f"[SHAP] AOPC: {compute_aopc(algorithm, X_eval, shap_vals):.4f}")
        metrics = evaluate_advanced_shap_metrics(shap_vals, X_eval)
        print(f"[SHAP] Entropy: {metrics.get('shap_entropy',0):.4f}")
        print(f"[SHAP] Coherence: {metrics.get('feature_coherence',0):.4f}")
        print(f"[SHAP] Channel Variance: {metrics.get('channel_variance',0):.4f}")
        print(f"[SHAP] Temporal Entropy: {metrics.get('temporal_entropy',0):.4f}")
        print(f"[SHAP] Mutual Info: {metrics.get('mutual_info',0):.4f}")
        print(f"[SHAP] PCA Alignment: {metrics.get('pca_alignment',0):.4f}")
        shap_array = _get_shap_array(shap_vals)
        if len(shap_array) >= 2:
            s1, s2 = shap_array[0], shap_array[1]
            print(f"[SHAP] Jaccard (top-10): {compute_jaccard_topk(s1,s2,k=10):.4f}")
            print(f"[SHAP] Kendall's Tau: {compute_kendall_tau(s1,s2):.4f}")
            print(f"[SHAP] Cosine Similarity: {cosine_similarity_shap(s1,s2):.4f}")
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        true_labels, pred_labels = [], []
        for data in valid_loader:
            if args.use_gnn and GNN_AVAILABLE:
                x, y = data[0].to(args.device), data[1]
            else:
                x, y = data[0].to(args.device).float(), data[1]
            with torch.no_grad():
                if args.use_gnn and GNN_AVAILABLE:
                    from your_utils import transform_for_gnn
                    x = transform_for_gnn(x)
                preds = algorithm.predict(x).cpu()
                true_labels.extend(y.cpu().numpy())
                pred_labels.extend(torch.argmax(preds,dim=1).cpu().numpy())
        cm = confusion_matrix(true_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.savefig(os.path.join(args.output, "confusion_matrix.png"), dpi=300)
        plt.close()
        print("✅ SHAP analysis completed successfully")
    except Exception as e:
        print(f"[ERROR] SHAP analysis failed: {e}")
        import traceback; traceback.print_exc()
