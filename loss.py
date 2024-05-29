import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class TripletMarginLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.lossfunc = nn.SoftMarginLoss

    def get_pos_neg_index(self, target_distances):
        # target_distances shape: (batch_size, batch_size)
        batch_size = target_distances.shape[0]
        pos_index = torch.zeros(batch_size).cuda()
        neg_index = torch.zeros(batch_size).cuda()
        for i in range(batch_size):
            pi, pj = random.sample(range(batch_size), 2)
            while pi == i or pj == i:
                pi, pj = random.sample(range(batch_size), 2)
            # if target_distances[i][pi] > target_distances[i][pj]: 写反了
            if target_distances[i][pi] < target_distances[i][pj]:
                pos_index[i] = pi
                neg_index[i] = pj
            else:
                pos_index[i] = pj
                neg_index[i] = pi
        return pos_index, neg_index

    def forward(self, predict_distances, target_distances):
        pos_index, neg_index = self.get_pos_neg_index(target_distances)
        pos_distances = torch.gather(predict_distances, 1, pos_index.unsqueeze(1).long())
        neg_distances = torch.gather(predict_distances, 1, neg_index.unsqueeze(1).long())
        pos_target_distances = torch.gather(target_distances, 1, pos_index.unsqueeze(1).long())
        neg_target_distances = torch.gather(target_distances, 1, neg_index.unsqueeze(1).long())
        threshold = neg_target_distances - pos_target_distances
        loss = torch.relu(pos_distances - neg_distances + threshold).mean()
        return loss

class LossFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression_loss = nn.MSELoss()
        self.triplet = TripletMarginLoss(margin=1.0)
    
    def forward(self, 
                protein_with_protein_targets, 
                protein_with_protein_predicts, 
                alpha=1.0,
                beta=1.0):
        # calculate the protein loss
        protein_loss = self.regression_loss(protein_with_protein_predicts, protein_with_protein_targets)
        # calculate the triplet loss
        triplet_loss = self.triplet(protein_with_protein_predicts, protein_with_protein_targets)
        
        total_loss = alpha * protein_loss + beta * triplet_loss 
        return total_loss