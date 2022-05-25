from tools import *

import torch
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage import zoom

def eval_model_single(model,warp_model_valid,dataloader_valid,data_num):

    labels = [1,    # spleen
              2,    # right kidney
              3,    # left kidney
              6]    # liver
    model.cuda()
    warp_model_valid.cuda()
    model.eval()
    
    dice_valid = 0.0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_valid):
            target = sample_batched['target_img'].cuda() 
            modality = sample_batched['target_ID'][0].split('_')[0]
            if modality == 'CT':
                atlas_all = sample_batched['atlas_img_ct']
                atlas_seg = sample_batched['atlas_label_ct']
            
            if modality == 'MR':
                atlas_all = sample_batched['atlas_img_mr']
                atlas_seg = sample_batched['atlas_label_mr']
            atlas_num = len(atlas_all)

            flow = [0] * atlas_num
            weight = [0] * atlas_num
            recons = [0] * atlas_num

            registered = torch.Tensor([]).cuda()
            flow_weight = torch.Tensor([]).cuda()

            for i in range(atlas_num):
                recons[i],flow[i],weight[i] = model(atlas_all[i].cuda(),atlas_all[i].cuda(), target, modality,valid=True)
                
                registered = torch.cat((registered,recons[i]),1) 
                flow_weight = torch.cat((flow_weight,weight[i]),1)
            flow_weight = F.softmax(flow_weight,1)

            registered_img = torch.sum(registered*flow_weight,1).unsqueeze(1)
            
            recons_seg = [0] * atlas_num
            registered_seg = torch.Tensor([]).cuda()
            
            for i in range(atlas_num):
                atlas_seg[i][atlas_seg[i] > 7] = 0
                recons_seg[i] = warp_model_valid(atlas_seg[i].cuda().float(),flow[i])
                
                registered_seg = torch.cat((registered_seg,recons_seg[i]),1)

            registered_seg = max_one_hot_weight(registered_seg, flow_weight)
            
            registered_seg = registered_seg.squeeze().cpu().numpy()
            registered_img = registered_img.squeeze().cpu().numpy()
            ID = sample_batched['target_ID'][0].split('_')[1]
            
            if modality == 'MR':
                target_segs = os.listdir('../Dataset/CHAOS/data_process/label/' + ID +'/T2SPIR/Ground')
                target_segs.sort()
                for i in range(len(target_segs)):
                    if i == 0:
                        target_seg = mpimg.imread('../Dataset/CHAOS/data_process/label/' + ID +'/T2SPIR/Ground/'+target_segs[i])[...,np.newaxis]
                    else:
                        mask = mpimg.imread('../Dataset/CHAOS/data_process/label/' + ID +'/T2SPIR/Ground/'+target_segs[i])[...,np.newaxis]
                        target_seg = np.concatenate((target_seg,mask),axis=2)
                target_seg = np.rot90(target_seg,-1)      
                target_seg[target_seg == 0.9882353] = 1
                target_seg[target_seg == 0.49411765] = 2
                target_seg[target_seg == 0.7411765] = 3
                target_seg[target_seg == 0.24705882] = 6
                target_seg[target_seg == 0] = 0
            
            if modality == 'CT':
                target_seg,_,_,_ = read_nii('../Dataset/BTCV/data/label_'+str(int(ID) + 43)+'.nii.gz')

            resample_size = registered_seg.shape
            origin_shape = target_seg.shape
            resize_factor = 1.0 *  np.asarray(origin_shape) / np.asarray(resample_size)
            registered_seg = zoom(registered_seg,resize_factor, order=0)
            
            
            dicem = Dice(registered_seg, target_seg, labels)
            dice_valid += np.mean(dicem) * len(sample_batched['target_ID'])

    dice_valid = dice_valid / data_num
    return dice_valid