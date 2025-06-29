import shap
import torch
from torch_geometric.data import Batch

class GNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, batch):
        # Unpack tuple if necessary (from SHAP)
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        # Convert list of Data to Batch
        if isinstance(batch, list):
            batch = Batch.from_data_list(batch)
        return self.model(batch)

def explain_gnn_with_shap(gnn, data_loader, device, sample_count=10):
    gnn.eval()
    batch = next(iter(data_loader))
    if isinstance(batch, (list, tuple)):
        batch_data = batch[0]
    else:
        batch_data = batch
    batch_data = batch_data.to(device)
    background = batch_data
    wrapped_model = GNNWrapper(gnn)
    explainer = shap.GradientExplainer(wrapped_model, (background,))
    shap_values = explainer.shap_values((batch_data,))
    return shap_values, batch_data
