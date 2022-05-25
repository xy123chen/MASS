#!/usr/bin/env python
# coding: utf-8

from models.models import  SpatialTransformation,m_net
from loss import * 
from dataset import Dataset
from tools import *
from eval_model_single import eval_model_single

import torch
import torch.optim as optim
from torch.utils import data

import numpy as np
import os
import time
import copy
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='few-shot segmentation')

parser.add_argument('--path', type=str, default='../Dataset/CHAOS/CHAOS_256_256_32/')
parser.add_argument('--save_path', type=str, default='../Weight/')
parser.add_argument('--weight_path', type=str)
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--data_size', type=tuple, default=(256,256,32))
parser.add_argument('--reg_param', type=float, default=1)

parser.add_argument('--check', type=bool, default=False)

args = parser.parse_args()

def train_model(model,warp_train, warp_valid, optimizer, args, labels):
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_dice = 0.0
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    # write logs
    record_path = args.save_path + '/record.log'
    rec = open(record_path,'w')
    rec.write('network:'+str(model))
    rec.writelines('\n')
    rec.writelines('# generator parameters:'+str(sum(param.numel() for param in model.parameters())))
    rec.writelines('\n')

    for epoch in range(args.epochs):
        
        s1 = time.time()
        print('Epoch {}/{}'.format(epoch, args.epochs-1))
        print('-' * 60)

        for phase in ['train', 'valid']:

            if phase == 'train':
                model.cuda()
                model.train()
                train_loss = 0.0
                dice_loss_all = 0.0
                contrastive_loss_all = 0.0
                train_t = enumerate(dataloader_train)
                
                for i_batch, sample_batched in tqdm(train_t):
                    loss_G = 0
                    target = sample_batched['target_img'].cuda()
                    num = len(sample_batched['atlas_img_ct'])

                    ct = random.randint(0,num-1)
                    atlas_ct = sample_batched['atlas_img_ct'][ct]
                    atlas_ct_label = sample_batched['atlas_label_ct'][ct]
                    
                    mr = random.randint(0,num-1)
                    atlas_mr = sample_batched['atlas_img_mr'][mr]
                    atlas_mr_label = sample_batched['atlas_label_mr'][mr]
                    
                    optimizer.zero_grad()
                    modality = sample_batched['target_ID'][0].split('_')[0]
                    
                    if modality == 'CT':
                        rec_sim,flow_sim,weight_sim,_,flow_dif,flow,_,_,_,_,contrastive_loss = model(atlas_ct.cuda(), atlas_mr.cuda(), target, modality,dice=True)
                        seg_sim = warp_train(atlas_ct_label.cuda(), flow_sim)
                        seg_dif = warp_train(atlas_mr_label.cuda(), flow_dif)
                    
                    if modality == 'MR':
                        rec_sim,flow_sim,weight_sim,_,flow_dif,flow,_,_,_,_,contrastive_loss = model(atlas_mr.cuda(), atlas_ct.cuda(), target, modality,dice=True)
                        seg_sim = warp_train(atlas_mr_label.cuda(), flow_sim)
                        seg_dif = warp_train(atlas_ct_label.cuda(), flow_dif)
                        
                    diffs = torch.add(target, -rec_sim)
                    n = torch.numel(diffs.data)
                    atlas_loss_msd = torch.sum(diffs.pow(2)) / n
                    atlas_loss = 1 - torch.mean(weight_sim)
                    smooth_loss =  gradient_loss(flow_sim)  + gradient_loss(flow)+ gradient_loss(flow_dif)
                    dice_loss = Dice(seg_sim.cpu().detach().numpy(), seg_dif.cpu().detach().numpy(), labels.numpy())
                    dice_loss = 1 - torch.mean(torch.from_numpy(dice_loss).cuda())

                    loss_G += smooth_loss + atlas_loss + dice_loss + atlas_loss_msd + contrastive_loss
            
                    loss_G.backward()
                    optimizer.step()
                    dice_loss_all += dice_loss * target.size(0)
                    contrastive_loss_all +=  contrastive_loss * target.size(0)
                    train_loss += loss_G * target.size(0)
                    
                if i_batch > 0:
                    s2 = time.time()
                    train_loss = train_loss / len(dataset_train)  
                    print('Training complete in %.0f m %.0f s'%((s2 - s1) // 60, (s2 - s1) % 60))
                    print('%s Loss: %.4f'%(phase, train_loss)) 
                    rec.writelines('train epoch:{},loss:{:f},dice loss:{:f},contrastive_loss2:{:f}'.format(epoch,train_loss, dice_loss_all / len(dataset_train) ,contrastive_loss_all / len(dataset_train) ))
                    rec.writelines('\n')
                    
            #valid
            else:
                warp_valid.cuda()
                model.cuda()
                model.eval()
                
                # Verification per five epochs
                if epoch >= 0 and epoch%5 == 0:
                    dice = eval_model_single(model,warp_valid,dataloader_valid,len(dataset_valid))
                    print('valid epoch:%d, dice loss:%.4f'%((epoch + 1), dice/ len(dataset_valid)))
                    rec = open(record_path,'a')  
                    rec.writelines('valid epoch:%d, dice loss:%.4f'%((epoch + 1), (dice / len(dataset_valid))))
                    if dice > best_dice:
                        best_dice = dice
                        torch.save(model.state_dict(), args.save_path + '/epoch_%d_dice_%.4f_.pth'%(epoch+1,dice))
                rec = open(record_path,'a')        
                rec.writelines('\n')        
                
    time_elapsed = time.time() - since
    print('Training complete in %.0f m %.0f s'%(time_elapsed // 60, time_elapsed % 60))
    print('Best dice: %.4f'%(best_dice))

if __name__ == '__main__':
    
    require_labels = torch.tensor([0,    # background
                      1,    # spleen
                      2,    # right kidney
                      3,    # left kidney
                      6])    # liver

    #initial envirement
    np.random.seed(6)
    torch.manual_seed(6)
    torch.cuda.manual_seed_all(6)
    
    # Select atlas number
    atlas_filename = ['MR_15','MR_19','MR_33','MR_37','MR_39','CT_16','CT_24','CT_25','CT_26','CT_29']

    train_filename = list(set(x for x in 
                        os.listdir(args.path+'train') if x not in atlas_filename))
        #print(train_filename)
    test_filename = list(set(x for x in os.listdir(args.path+'valid'))) # 20 patients
    test_filename.sort()

    
    print('atlas_filename:',atlas_filename)
    print('train_filename',train_filename)
    print('test_filename',test_filename)
    
    # Generators
    dataset_train = Dataset(atlas_filename,train_filename,state = 'train',
                            path = args.path ,modality = 'multi')
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, 
                                       shuffle=True , num_workers=32)

    dataset_valid = Dataset(atlas_filename,test_filename,state = 'valid',
                            path = args.path ,modality = 'multi')
    dataloader_valid = data.DataLoader(dataset_valid, batch_size=1, 
                                       shuffle=False , num_workers=32)

    print('dataset_train:',len(dataset_train))
    print('dataset_valid:',len(dataset_valid))
    
    model = m_net(args.data_size)
    
    if args.pretrain:
        print('weight_path', args.weight_path)
        model = load_weights(model, args.weight_path)
        
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    
    warp_train = SpatialTransformation(size = args.data_size)
    warp_valid = SpatialTransformation(size = args.data_size,mode = 'nearest')
    optimizer = optim.Adam(model.parameters(), lr = 3e-4)
    mseloss = MSE().cuda()
    
    #train 
    train_model(model,warp_train, warp_valid, optimizer, args, require_labels)
    