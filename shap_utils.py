import shap
import torch

class GNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, batch):
        # batch is a single torch_geometric.data.Batch
        return self.model(batch)

def explain_gnn_with_shap(model, data_loader, device, sample_count=10):
    model.eval()
    # Get one *batched* graph (PyG Batch) from data_loader
    batch = next(iter(data_loader))
    if isinstance(batch, (list, tuple)):
        batch_data = batch[0]  # (batch, labels) tuple
    else:
        batch_data = batch
    batch_data = batch_data.to(device)
    # For background, use a single Batch object (do NOT slice)
    background = batch_data
    wrapped_model = GNNWrapper(model)
    explainer = shap.GradientExplainer(wrapped_model, background)
    shap_values = explainer.shap_values(batch_data)
    return shap_values, batch_data
