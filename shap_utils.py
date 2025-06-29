import shap
import torch
from torch_geometric.data import Batch

class GNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, batch):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        if isinstance(batch, list):
            batch = Batch.from_data_list(batch)
        return self.model(batch)

def gnn_predict(batch_list):
    # batch_list: list of arrays (from SHAP); need to convert to Batch objects
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = Batch.from_data_list([torch_geometric.data.Data(x=torch.tensor(arr).float().to(device)) for arr in batch_list])
    return wrapped_model(batch).detach().cpu().numpy()

def explain_gnn_with_shap(gnn, data_loader, device, sample_count=10):
    gnn.eval()
    batch = next(iter(data_loader))
    if isinstance(batch, (list, tuple)):
        batch_data = batch[0]
    else:
        batch_data = batch
    batch_data = batch_data.to(device)
    wrapped_model = GNNWrapper(gnn)
    # Use only a few samples for KernelExplainer (as it's slow)
    background = [d for d in batch_data.to_data_list()[:sample_count]]
    explainer = shap.KernelExplainer(lambda arrs: wrapped_model(Batch.from_data_list([d.to(device) for d in arrs])).detach().cpu().numpy(),
                                     background)
    shap_values = explainer.shap_values([d for d in batch_data.to_data_list()])
    return shap_values, batch_data
