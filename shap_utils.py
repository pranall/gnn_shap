import shap
import torch
import numpy as np
from torch_geometric.data import Batch

class GNNWrapper(torch.nn.Module):
    def __init__(self, model, original_data_sample=None):
        super().__init__()
        self.model = model
        self.original_data = original_data_sample
        
    def forward(self, x_list):
        """Handle both direct GNN calls and SHAP explanations"""
        if isinstance(x_list, (list, np.ndarray)):
            # SHAP explanation case
            return self._explain_forward(x_list)
        # Normal GNN forward pass
        return self.model(x_list)
        
    def _explain_forward(self, x_flat_list):
        """Convert SHAP inputs to graph data"""
        if self.original_data is None:
            raise ValueError("Original data sample required for SHAP explanations")
            
        device = next(self.model.parameters()).device
        data_list = []
        
        for x_flat in x_flat_list:
            # Reshape flattened features to original dimensions
            x = torch.FloatTensor(x_flat.reshape(self.original_data.x.shape)).to(device)
            
            # Create new Data object preserving original structure
            data = self.original_data.__class__(
                x=x,
                edge_index=self.original_data.edge_index.clone().to(device),
                batch=torch.zeros(x.size(0), dtype=torch.long).to(device)
            data_list.append(data)
            
        batch = Batch.from_data_list(data_list)
        return self.model(batch)

def explain_gnn_with_shap(gnn_model, data_loader, device='cuda', sample_count=5):
    """Safe SHAP explanation for GNNs"""
    gnn_model.eval()
    
    # Get a sample batch
    try:
        sample_batch = next(iter(data_loader)).to(device)
    except StopIteration:
        raise ValueError("DataLoader is empty")
        
    # Use first graph as reference
    sample_data = sample_batch[0] if isinstance(sample_batch, Batch) else sample_batch
    
    # Create wrapper with reference sample
    wrapped_model = GNNWrapper(gnn_model, sample_data).to(device)
    
    # Prepare background data (small sample for efficiency)
    background = [sample_batch[i].x.cpu().numpy().flatten() 
                 for i in range(min(sample_count, len(sample_batch)))]
    
    # Create explainer
    explainer = shap.KernelExplainer(
        model=wrapped_model,
        data=np.array(background),
        keep_index=True
    )
    
    # Explain first sample
    sample_to_explain = background[0]
    shap_values = explainer.shap_values(sample_to_explain, nsamples=50)
    
    return shap_values.reshape(sample_data.x.shape), sample_batch
