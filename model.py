import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Model,GCNNModel,CNN_Attention
from loguru import logger
class AnchorModel(nn.Module):
    def __init__(self, hidden_size,model_type='GCNN'):
        super().__init__()
        # self.extractor = Model(hidden_size,encoder_type=model_type)
        if(model_type=='GCNN'):
            logger.info('Using GCNN')
            self.extractor = GCNNModel(hidden_size)
        if(model_type=='Attn'):
            logger.info('Using Attn')
            self.extractor = CNN_Attention(hidden_size)
        else:
            logger.info('Using baseline')
            self.extractor = Model(hidden_size)
        #输出模型参数量
        logger.info('Model has {} parameters'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
    
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