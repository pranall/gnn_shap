import shap
import torch

class GNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, batch):
        return self.model(batch)

def explain_gnn_with_shap(gnn_model, data_loader, device, sample_count=10):
    gnn_model.eval()
    batch = next(iter(data_loader))
    if isinstance(batch, (list, tuple)):
        batch_data = batch[0]
    else:
        batch_data = batch
    batch_data = batch_data.to(device)
    background = batch_data
    wrapped_model = GNNWrapper(gnn_model)   # <--- use 'gnn' not 'model'
    explainer = shap.GradientExplainer(wrapped_model, (background,))
    shap_values = explainer.shap_values((batch_data,))
    return shap_values, batch_data
