import shap
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from tqdm import tqdm

class GNNWrapper(torch.nn.Module):
    def __init__(self, model, original_data_sample):
        super().__init__()
        self.model = model
        self.original_data = original_data_sample  # Used to reconstruct graph structure
        
    def forward(self, x_flat_list):
        """Convert flattened SHAP inputs back to graph data"""
        device = next(self.model.parameters()).device
        data_list = []
        
        for x_flat in x_flat_list:
            # Reconstruct node features
            x = torch.FloatTensor(x_flat.reshape(self.original_data.x.shape)).to(device)
            
            # Use original edge structure
            edge_index = self.original_data.edge_index.clone().to(device)
            
            # Create new Data object preserving original structure
            data = Data(
                x=x,
                edge_index=edge_index,
                batch=torch.zeros(x.size(0), dtype=torch.long).to(device)
            )
            data_list.append(data)
            
        batch = Batch.from_data_list(data_list)
        return self.model(batch)

def explain_gnn_with_shap(gnn, data_loader, device, sample_count=10, nsamples=100):
    """
    Explain GNN predictions using SHAP
    
    Args:
        gnn: Your trained GNN model
        data_loader: DataLoader containing graph data
        device: 'cuda' or 'cpu'
        sample_count: Number of background samples to use
        nsamples: Number of SHAP samples to generate
        
    Returns:
        shap_values: SHAP values for node features
        explained_data: The data sample being explained
    """
    gnn.eval()
    
    # Get background samples and sample to explain
    background_batch = next(iter(data_loader)).to(device)
    background_data = background_batch.to_data_list()[:sample_count]
    explain_data = background_data[0]  # Explain first sample
    
    # Create wrapper model with original data structure reference
    wrapped_model = GNNWrapper(gnn, explain_data).to(device)
    
    # Prepare background data (flatten node features)
    background_shap = [data.x.cpu().numpy().flatten() for data in background_data]
    background_shap = np.array(background_shap)
    
    # Create explainer
    explainer = shap.KernelExplainer(
        model=lambda x: wrapped_model(x).detach().cpu().numpy(),
        data=background_shap,
        keep_index=True
    )
    
    # Prepare sample to explain
    sample_shap = explain_data.x.cpu().numpy().flatten()[np.newaxis, :]
    
    # Calculate SHAP values with progress bar
    with tqdm(total=nsamples, desc="Calculating SHAP values") as pbar:
        shap_values = explainer.shap_values(
            X=sample_shap,
            nsamples=nsamples,
            silent=True,
            progress_callback=lambda x: pbar.update(1)
        )
    
    # Reshape SHAP values back to original feature dimensions
    shap_values = [v.reshape(explain_data.x.shape) for v in shap_values]
    
    return shap_values, explain_data

def visualize_shap_values(shap_values, feature_names=None):
    """Visualize SHAP values for node features"""
    import matplotlib.pyplot as plt
    
    # Assuming single graph explanation
    shap_val = shap_values[0] if isinstance(shap_values, list) else shap_values
    
    plt.figure(figsize=(10, 6))
    plt.imshow(shap_val.T, aspect='auto', cmap='coolwarm')
    plt.colorbar(label='SHAP value')
    
    if feature_names:
        plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Node index')
    plt.ylabel('Feature')
    plt.title('Node Feature Importance')
    plt.show()
