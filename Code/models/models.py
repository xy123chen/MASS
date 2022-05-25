import torch.nn as nn
import numpy as np
from loss import *

from .SpatialTransformation import SpatialTransformation
from .network import network

from tools import NCC

class m_net(nn.Module):
    def __init__(self, data_size):
        super(m_net, self).__init__()
        self.size = data_size

        self.model = network(self.size).cuda()

        self.warp = SpatialTransformation(size = self.size).cuda()
        self.warp_valid = SpatialTransformation(size = self.size,mode = 'nearest')
        

    def forward(self, atlas_sim, atlas_dif, target, modality,G = True, valid = False, dice = False):
        
        if modality == 'CT':
            m1 = 'CT-CT';
            m2 = 'CT-MR';
            m3 = 'MR-CT';
            m4 = 'MR-MR';
            m5 = 'CT';

        else :
            m1 = 'MR-MR';
            m2 = 'MR-CT';
            m3 = 'CT-MR';
            m4 = 'CT-CT';
            m5 = 'MR';
            
        if G:
            if valid:
                flow_sim,feature_sim,feature_target_sim = self.model(atlas_sim, target,m1)
                rec_sim = self.warp(atlas_sim, flow_sim)
                weight_sim = NCC(rec_sim, target).cuda()
                return rec_sim,flow_sim,weight_sim
                
            
            flow_sim,feature_sim,feature_target_sim = self.model(atlas_sim, target,m1) 
            rec_sim = self.warp(atlas_sim, flow_sim)
            weight_sim = NCC(rec_sim, target).cuda()

            flow_dif,feature_dif,feature_target_dif = self.model(atlas_dif, target,m2) 
            rec_dif = self.warp(atlas_dif, flow_dif)

        
            if dice:
                flow,_,_ = self.model(rec_sim, target,m5) 
                rec = self.warp(rec_sim, flow)
                weight = NCC(rec, target).cuda()
                
                A = rec_sim.detach().cpu().numpy().reshape(1,-1).squeeze()
                B = rec_dif.detach().cpu().numpy().reshape(1,-1).squeeze()
                
                contrastive_loss = self.model(rec_sim ,rec_dif, m1,G = False,contrastive = True)

                return rec_sim,flow_sim,weight_sim,rec_dif,flow_dif,flow,rec,weight,feature_target_sim, feature_target_dif,contrastive_loss
            
        else:
            feature_sim,feature_target_sim = self.model(atlas_sim, target,m1,G)
            feature_dif,feature_target_dif = self.model(atlas_dif, target,m2,G)
            
            return feature_sim,feature_dif,feature_target_sim,feature_target_dif
    