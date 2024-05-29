import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Model,GCNNModel

class AnchorModel(nn.Module):
    def __init__(self, hidden_size,model_type='GCNN'):
        super().__init__()
        # self.extractor = Model(hidden_size)
        if(model_type=='GCNN'):
            print('Using GCNN')
            self.extractor = GCNNModel(hidden_size)
        else:
            print('Using baseline')
            self.extractor = Model(hidden_size)
    
    def forward(self, 
                inputs, 
                masks):
        protein_embeddings = self.extractor(inputs, masks)
        protein_with_protein_distances = torch.norm(protein_embeddings.unsqueeze(1) -
                                                    protein_embeddings.unsqueeze(0), p=2, dim=2)
        
        return protein_with_protein_distances
    
    @torch.no_grad()
    def forward_features(self, 
                         inputs, 
                         masks):
        protein_embeddings = self.extractor(inputs, masks) # anchor_embeddings

        return protein_embeddings