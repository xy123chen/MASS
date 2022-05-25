from tools import *
from models.models import SpatialTransformation,m_net
from loss import * 
from dataset import Dataset
#from dataset_crop import Dataset

import torch
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import os
import nibabel as nib
from scipy.ndimage import zoom
import pandas as pd
import matplotlib.image as mpimg
import pandas as pd


labels = [ # background
              1,    # spleen
              2,    # right kidney
              3,    # left kidney
              6]    # liver

os.environ["CUDA_VISIBLE_DEVICES"] = '7' # Set your GPU
path ='../Dataset/CT_MR/' # Dataset path

weight_path = "../Weight/" # Your weight path

#initial envirement
np.random.seed(6)
torch.manual_seed(6)
torch.cuda.manual_seed_all(6)

batch_size = 1

atlas_filename = ['MR_15','MR_19','MR_33','MR_37','MR_39','CT_16','CT_24','CT_25','CT_26','CT_29']

test_filename = list(set(x for x in 
                    os.listdir(path+'valid')))

test_filename.sort()

# Generators
dataset_valid = Dataset(atlas_filename,test_filename,state = 'valid',
                        path = path,modality = 'multi')
dataloader_valid = data.DataLoader(dataset_valid, batch_size=batch_size, 
                                   shuffle=False , num_workers=4)

print(len(dataset_valid))

model = m_net((256,256,32))

print('# generator parameters:', sum(param.numel() for param in model.parameters()))
model = load_weights(model, weight_path)
warp_model_valid = SpatialTransformation(size=(256,256,32),mode = 'nearest').cuda()

def save_nii( data, save_path, header, affine):
    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, save_path)
    
def read_nii( volume_path):
    nii = nib.load(volume_path)
    data = nii.get_fdata()
    header = nii.header
    affine = nii.affine
    spacing = list(nii.header['pixdim'][1:4])
    return data, header, affine, spacing

def create_path(pathlist):
    for path in pathlist:
        if not os.path.exists(path):
            os.makedirs(path)
model.cuda()
warp_model_valid.cuda()
model.eval()
dicem_all = list([[0,0,0,0]])
dicem_ct = list([[0,0,0,0]])
dicem_mr = list([[0,0,0,0]])
dice_valid_all = 0.0
dice_valid_ct = 0.0
dice_valid_mr = 0.0
num = 0

result = pd.DataFrame(columns=['id','dice','dstd','spleen','sstd','right kidney','rstd','left kidney','lstd','liver','l-std'])

with torch.no_grad():
    for i_batch, sample_batched in enumerate(dataloader_valid):
        print(i_batch)
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
        flow2 = [0] * atlas_num
        weight2 = [0] * atlas_num
        recons2 = [0] * atlas_num

        flow_weight = torch.Tensor([]).cuda()
        flow_weight2 = torch.Tensor([]).cuda()

        for i in range(atlas_num):

            recons[i],flow[i],weight[i],_,_,flow2[i],recons2[i],weight2[i],_,_,_ = model(atlas_all[i].cuda(),atlas_all[i].cuda(), target, modality,dice=True)

            flow_weight2 = torch.cat((flow_weight2,weight2[i]),1) 
        flow_weight2 = F.softmax(flow_weight2,1)

        recons_seg = [0] * atlas_num
        registered_seg = torch.Tensor([]).cuda()
        recons_seg2 = [0] * atlas_num

        for i in range(atlas_num):
            atlas_seg[i][atlas_seg[i] > 7] = 0
            recons_seg[i] = warp_model_valid(atlas_seg[i].cuda().float(),flow[i])
            recons_seg2[i] = warp_model_valid(recons_seg[i].cuda().float(),flow2[i])

            registered_seg = torch.cat((registered_seg,recons_seg2[i]),1)

        registered_seg = max_one_hot_weight(registered_seg, flow_weight2)

        registered_seg = registered_seg.squeeze().cpu().numpy()
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
        target_ID = str(sample_batched['target_ID'][0])
        new = pd.DataFrame({'id':target_ID,'dice':np.mean(dicem[0]),'dstd':0,'spleen':dicem[0][0],'sstd':0,'right kidney':dicem[0][1],'rstd':0,
                            'left kidney':dicem[0][2],'lstd':0,'liver':dicem[0][3],'l-std':0},index=[1])
        result = result.append(new)
        dicem_all += dicem
        dice_valid_all += np.mean(dicem[0])
        if modality == 'MR':
            dicem_mr += dicem
            dice_valid_mr += np.mean(dicem[0])
        if modality == 'CT':
            dicem_ct += dicem
            dice_valid_ct += np.mean(dicem[0])
        num += 1

dice_valid_all = dice_valid_all / num
dice_valid_ct = dice_valid_ct / (num / 2)
dice_valid_mr = dice_valid_mr / (num / 2)

dicem_all = dicem_all / num
dicem_ct = dicem_ct / (num / 2)
dicem_mr = dicem_mr / (num / 2)

dice_std = np.std(result['dice'][:num]).round(2)
spleen_std = np.std(result['spleen'][:num]).round(2)
r_kidney_std = np.std(result['right kidney'][:num]).round(2)
l_kidney_std = np.std(result['left kidney'][:num]).round(2)
liver_std = np.std(result['liver'][:num]).round(2)

all_average = pd.DataFrame({'id':'ALL_average','dice':dice_valid_all,'dstd':dice_std,'spleen':dicem_all[0][0],'sstd':spleen_std,'right kidney':dicem_all[0][1],'rstd':r_kidney_std,
                            'left kidney':dicem_all[0][2],'lstd':l_kidney_std,'liver':dicem_all[0][3],'l-std':liver_std},index=[1])

result = result.append(all_average.round(4))

dice_std = np.std(result['dice'][:int(num/2)]).round(2)
spleen_std = np.std(result['spleen'][:int(num/2)]).round(2)
r_kidney_std = np.std(result['right kidney'][:int(num/2)]).round(2)
l_kidney_std = np.std(result['left kidney'][:int(num/2)]).round(2)
liver_std = np.std(result['liver'][:int(num/2)]).round(2)

ct_average = pd.DataFrame({'id':'ct_average','dice':dice_valid_ct,'dstd':dice_std,'spleen':dicem_ct[0][0],'sstd':spleen_std,'right kidney':dicem_ct[0][1],'rstd':r_kidney_std,
                            'left kidney':dicem_ct[0][2],'lstd':l_kidney_std,'liver':dicem_ct[0][3],'l-std':liver_std},index=[1])

result = result.append(ct_average.round(4))

dice_std = np.std(result['dice'][int(num/2):num]).round(2)
spleen_std = np.std(result['spleen'][int(num/2):num]).round(2)
r_kidney_std = np.std(result['right kidney'][int(num/2):num]).round(2)
l_kidney_std = np.std(result['left kidney'][int(num/2):num]).round(2)
liver_std = np.std(result['liver'][int(num/2):num]).round(2)
mr_average = pd.DataFrame({'id':'mr_average','dice':dice_valid_mr,'dstd':dice_std,'spleen':dicem_mr[0][0],'sstd':spleen_std,'right kidney':dicem_mr[0][1],'rstd':r_kidney_std,
                            'left kidney':dicem_mr[0][2],'lstd':l_kidney_std,'liver':dicem_mr[0][3],'l-std':liver_std},index=[1])

result = result.append(mr_average.round(4))

print(result)
result.to_csv('result.csv',index = 0)