import shap
import torch

def explain_gnn_with_shap(model, data_loader, device, sample_count=10):
    model.eval()
    batch = next(iter(data_loader))
    if isinstance(batch, (list, tuple)):
        batch_data = batch[0]
    else:
        batch_data = batch
    batch_data = batch_data.to(device)
    background = batch_data[:min(sample_count, batch_data.shape[0])]
    # Use GradientExplainer instead of DeepExplainer
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(batch_data)
    return shap_values, batch_data

def plot_shap_summary(shap_values, data, feature_names=None):
    shap.summary_plot(shap_values, data.x.cpu().numpy(), feature_names=feature_names)
