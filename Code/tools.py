 #coding=utf-8

import torch
import torch.nn.functional as F
import numpy as np
import math
import nibabel as nib
from medpy.metric import binary

def to_one_hot(v):
    batch, c, w, h, d = v.shape
    recons = v[:, 0, ...].unsqueeze(4)
    v_one_hot = torch.zeros((batch, w, h, d, 7)).cuda()
    v_one_hot = v_one_hot.scatter_(4, recons.long(), 1)
    return v_one_hot

def read_nii( volume_path):
    nii = nib.load(volume_path)
    data = nii.get_fdata()
    header = nii.header
    affine = nii.affine
    spacing = list(nii.header['pixdim'][1:4])
    return data, header, affine, spacing

def normalize(x,norm):    
    if norm == 'win_maxmin':
        x[x > 1000] = 1000
        y = (x - x.min()) / (x.max() - x.min())
    
    if norm == 'win':
        x[x > 250] = 250
        x[x < -150] = -150
        y = (x - x.min()) / (x.max() - x.min())
    
    return y

def Dice(vol1, vol2, labels, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''

    if len(vol1.shape) != 4:
        vol1 = vol1[np.newaxis,:]
        vol2 = vol2[np.newaxis,:]
        
    dicem = np.zeros((vol1.shape[0],len(labels)))
    
    for i in range(len(dicem)):
        for idx, lab in enumerate(labels):
            top = 2 * np.sum(np.logical_and(vol1[i] == lab, vol2[i] == lab))
            bottom = np.sum(vol1[i] == lab) + np.sum(vol2[i] == lab)
            bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
            dicem[i][idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path)
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    return model


def max_one_hot_weight(seg, weight):
    '''
    input
    seg [batch,5,w,h,d]
    weight [batch,5,w,h,d]
    
    output
    seg [batch,1,w,h,d]
    '''
    class_num = 7 #随便写一个比类总数大的数
    batch, c, w, h, d = seg.shape #batch,channel,width,height,depth
    
    weight_one_hot = torch.zeros((batch, w, h, d,class_num)).cuda()
    for i in range(c):
        
        recons = seg[:, i, ...].unsqueeze(4)
        weight_atlas = weight[:, i, ...].unsqueeze(4)
        seg_one_hot = torch.zeros((batch, w, h, d, class_num)).cuda()
        seg_one_hot = seg_one_hot.scatter_(4, recons.long(), 1)
        weight_one_hot += seg_one_hot * weight_atlas

    seg = torch.argmax(weight_one_hot,4).unsqueeze(1)

    return seg

def NCC(I, J, wins = 9,ndims=3):

    win = [wins] * ndims
    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I*I
    J2 = J*J
    IJ = I*J
    
    filt = torch.ones([1,1, (*win)]).cuda()
    
    pad_no = math.floor(win[0]/2)
    stride = (1,1,1)
    padding = (pad_no, pad_no, pad_no)

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    
    cc = cross*cross / (I_var*J_var + 1e-5)
    return cc

def HD_distence(vol1, vol2, labels):
    if len(vol1.shape) != 4:
        vol1 = vol1[np.newaxis,:]
        vol2 = vol2[np.newaxis,:]
        
    hdm = np.zeros((vol1.shape[0],len(labels)))
    
    for i in range(len(hdm)):
        for idx, lab in enumerate(labels):
            seg1 = (vol1 == lab)*1.0
            seg2 = (vol2 == lab)*1.0
            hd = binary.hd95(seg1, seg2)
            hdm[i][idx] = hd

    return hdm



def ASD_distence(vol1, vol2, labels):
    if len(vol1.shape) != 4:
        vol1 = vol1[np.newaxis,:]
        vol2 = vol2[np.newaxis,:]
        
    asdm = np.zeros((vol1.shape[0],len(labels)))
    
    for i in range(len(asdm)):
        for idx, lab in enumerate(labels):
            seg1 = (vol1 == lab)*1.0
            seg2 = (vol2 == lab)*1.0
            hd = binary.assd(seg1, seg2)
            asdm[i][idx] = hd

    return asdm
