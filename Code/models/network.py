import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .SpatialTransformation import SpatialTransformation

# Cost volume layer -------------------------------------
def get_cost(feature1, feature2, shift):
    """
    Calculate cost volume for specific shift
    - inputs
    feature1 (batch, h, w, nch): feature maps at time slice 0
    feature2 (batch, h, w, nch): feature maps at time slice 0 warped from 1
    shift (2): spatial (vertical and horizontal) shift to be considered
    - output
    cost (batch, h, w): cost volume map for the given shift
    """

    v, h, d = shift  # vertical/horizontal element
    
    vt, vb, hl, hr, ds, df= max(v, 0), abs(min(v, 0)), max(h, 0), abs(min(h, 0)), max(d, 0), abs(min(d, 0))  # top/bottom left/right
    feature1_pad = F.pad(feature1,pad = [ ds, df, hl, hr,vt, vb],mode = 'constant',value= 0)
    feature2_pad = F.pad(feature2,pad = [ df, ds, hr, hl,vb, vt],mode = 'constant',value= 0)
    cost_pad = feature1_pad * feature2_pad
    size = cost_pad.shape[2:]
    
    crop_cost = cost_pad[:,:,vt:size[0]-vb,hl:size[1]-hr,ds:size[2]-df]
    return torch.mean(crop_cost, axis=1)

def CostVolumeLayer(feature1,feature2,rang = 2):
    search_range = rang
    cost_length = (2 * search_range + 1) ** 3
    cv = [0] * cost_length
    depth = 0
    for v in range(-search_range, search_range + 1):
        for h in range(-search_range, search_range + 1):
            for d in range(-search_range, search_range+1):
                cv[depth] = get_cost(feature1, feature2, shift=[v, h, d])
                depth += 1
    cv = torch.stack(cv, dim=1)
    
    return cv

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.InstanceNorm3d(num_features=out_planes),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv3d(in_planes,3,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class base_encoder(nn.Module):
    def __init__(self,num_classes,channel):
        super(base_encoder,self).__init__()
        self.channel = channel
        self.conv = conv(32,channel,kernel_size=3, stride=2)
        
        self.fc1 = MLP(16*16*2,512,512)
        self.fc2 = MLP(512,num_classes,256)
        
    def forward(self,x):
        x = self.conv(x)
        x = x.squeeze().view(self.channel,-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x # 32*128

class network(nn.Module):
    def __init__(self,data_size):
        super(network,self).__init__()
        self.data_size = data_size
        self.downsample_size_1 = tuple(np.array(self.data_size) / 2)
        self.downsample_size_2 = tuple(np.array(self.data_size) / 4)

        self.channel = 64
        self.encoder = base_encoder(num_classes=128, channel = self.channel) # final output channel * 128
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.conv_ct = conv(1, 4, kernel_size=3, stride=2)
        self.conv_mr = conv(1, 4, kernel_size=3, stride=2)

        self.conv1a_ct  = conv(4,   16, kernel_size=3, stride=2)
        self.conv1aa_ct = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b_ct  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a_ct  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa_ct = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b_ct  = conv(32,  32, kernel_size=3, stride=1)
        
        self.conv1a_mr  = conv(4,   16, kernel_size=3, stride=2)
        self.conv1aa_mr = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b_mr  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a_mr  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa_mr = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b_mr  = conv(32,  32, kernel_size=3, stride=1)
              
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        od = 125
        self.conv2_0 = conv(od,      32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(32) 
        self.deconv2 = deconv(3, 3, kernel_size=4, stride=2, padding=1)
        self.upfeat2 = deconv(32, 3, kernel_size=4, stride=2, padding=1) 
        self.stn2 = SpatialTransformation(size = self.downsample_size_2) 

        od = 147
        self.conv1_0 = conv(od,      32, kernel_size=3, stride=1)
        self.predict_flow1 = predict_flow(32) 
        self.deconv1 = deconv(3, 3, kernel_size=4, stride=2, padding=1)
        self.upfeat1 = deconv(32, 64, kernel_size=4, stride=2, padding=1) 
        self.stn1 = SpatialTransformation(size = self.downsample_size_1) 

        od = 196
        self.conv0_0 = conv(od, 32, kernel_size=3, stride=1)
        self.predict_flow0 = predict_flow(32) 
        self.deconv0 = deconv(3, 3, kernel_size=4, stride=2, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,atlas, target,modality,G = True, contrastive = False):
        im1 = target
        im2 = atlas

        
        if modality == 'CT-CT' or modality == 'CT':   
            im2 = self.conv_ct(im2)
            
            im1 = self.conv_ct(im1)
            c11 = self.conv1b_ct(self.conv1aa_ct(self.conv1a_ct(im1))) 
            c21 = self.conv1b_ct(self.conv1aa_ct(self.conv1a_ct(im2))) 
            c12 = self.conv2b_ct(self.conv2aa_ct(self.conv2a_ct(c11))) 
            c22 = self.conv2b_ct(self.conv2aa_ct(self.conv2a_ct(c21)))

        if modality == 'CT-MR' :
            im1 = self.conv_ct(im1)
            im2 = self.conv_mr(im2)
            c11 = self.conv1b_ct(self.conv1aa_ct(self.conv1a_ct(im1))) 
            c21 = self.conv1b_mr(self.conv1aa_mr(self.conv1a_mr(im2))) 
            c12 = self.conv2b_ct(self.conv2aa_ct(self.conv2a_ct(c11))) 
            c22 = self.conv2b_mr(self.conv2aa_mr(self.conv2a_mr(c21))) 

        if modality == 'MR-MR' or modality == 'MR':
            im2 = self.conv_mr(im2)
            im1 = self.conv_mr(im1)
            c11 = self.conv1b_mr(self.conv1aa_mr(self.conv1a_mr(im1))) 
            c21 = self.conv1b_mr(self.conv1aa_mr(self.conv1a_mr(im2))) 
            c12 = self.conv2b_mr(self.conv2aa_mr(self.conv2a_mr(c11))) 
            c22 = self.conv2b_mr(self.conv2aa_mr(self.conv2a_mr(c21)))

        if modality == 'MR-CT':  
            im1 = self.conv_mr(im1)
            im2 = self.conv_ct(im2)
            c11 = self.conv1b_mr(self.conv1aa_mr(self.conv1a_mr(im1))) 
            c21 = self.conv1b_ct(self.conv1aa_ct(self.conv1a_ct(im2))) 
            c12 = self.conv2b_mr(self.conv2aa_mr(self.conv2a_mr(c11))) 
            c22 = self.conv2b_ct(self.conv2aa_ct(self.conv2a_ct(c21)))
        
        if contrastive:
            c1 = self.encoder(c12)
            c2 = self.encoder(c22)
            c1 = F.normalize(c1,dim=0,p=2) #l2 normalization
            c2 = F.normalize(c2,dim=0,p=2)
            labels = torch.arange(start=0,end=self.channel, step=1).long().cuda()
            logits = torch.mm(c1,c2.T) 
            loss = self.criterion(logits, labels)
            return loss
        
        if G:
            
            x = CostVolumeLayer(c12,c22) 
            x = self.leakyRELU(x) 
            x = self.conv2_0(x) 
            flow2 = self.predict_flow2(x) 
            up_flow2 = self.deconv2(flow2)
            up_feat2 = self.upfeat2(x) 
            warp1 = self.stn2(c21, up_flow2) 
            
            corr1 = CostVolumeLayer(c11,warp1) 
            corr1 = self.leakyRELU(corr1)
            x = torch.cat((corr1, c11, up_flow2, up_feat2), 1)
            x = self.conv1_0(x)
            flow1 = self.predict_flow1(x) 
            up_flow1 = self.deconv1(flow1) 
            up_feat1 = self.upfeat1(x) 
            warp0 = self.stn1(im2, up_flow1) 
            
            corr0 = CostVolumeLayer(im1,warp0) 
            corr0 = self.leakyRELU(corr0)
            x = torch.cat((corr0, im1, up_flow1, up_feat1), 1) 
            x = self.conv0_0(x) 
            flow0 = self.predict_flow0(x) 
            flow = self.deconv0(flow0)

            return flow,c22,c12
        
        else:
            return c22,c12