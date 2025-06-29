import shap
import torch

def explain_gnn_with_shap(model, data_loader, device, sample_count=10):
    model.eval()
    # Select a batch for explanation
    batch_data, batch_labels = next(iter(data_loader))
    batch_data = batch_data.to(device)
    
    # DeepExplainer expects a callable; wrap your model
    def gnn_forward(input_data):
        # input_data: batch of graph objects on CPU, so move to device
        return model(input_data.to(device), input_data.batch.to(device)).detach().cpu().numpy()
    
    # SHAP expects background samples; sample from data_loader
    background = [batch_data[i] for i in range(min(sample_count, len(batch_data)))]
    
    explainer = shap.DeepExplainer(gnn_forward, background)
    shap_values = explainer.shap_values(batch_data)
    return shap_values, batch_data, batch_labels

def plot_shap_summary(shap_values, data, feature_names=None):
    # Summarize feature attributions (works for tabular-like or node-level)
    shap.summary_plot(shap_values, data.x.cpu().numpy(), feature_names=feature_names)
