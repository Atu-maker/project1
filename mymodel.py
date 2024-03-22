import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm.auto import  tqdm
import numpy as np
import random
import torch.nn.functional as F
from model import LSKblock,SKAttention,RCAB,FFA
from transformers import ConvNextModel, ConvNextConfig,AutoModel
from torchvision.models import vgg16

from torchvision.utils import make_grid
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import gc

import cv2
#from swin_transformer_v2 import SwinTransformerV2
from Block import OutlookAttention
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR

# class MyNet(nn.Module):
#     def __init__(self):
#         super(MyNet, self).__init__()

#         self.net=nn.Sequential(

#                     nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#                     nn.BatchNorm2d(32),
#                     nn.ReLU(),

#                     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#                     nn.BatchNorm2d(64),
#                     nn.ReLU(),

#                     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#                     nn.BatchNorm2d(128),
#                     nn.ReLU(),
#                     LSKblock(32),

#                     nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
#                     nn.BatchNorm2d(3),
#                     nn.ReLU(),
                                        
#                     )
        
#     def forward(self, x):
        
#         # x=self.relu(self.conv1(x))
#         # x=self.relu(self.conv2(x))
#         # x=self.relu(self.conv3(x))
#         # x=self.conv4(x)
#         x=x+self.net(x)
        
#         return x



#https://zhuanlan.zhihu.com/p/489225535
# 增强器---金字塔池化模块
# （PPM）

# 金字塔池化模块（Pyramid Pooling Module，PPM）出自论文《Pyramid Scene Parsing Network
# 》，也就是PSPNet。它可以聚合不同区域的上下文信息，提高网络获取全局信息的能力。
# 在现有深度网络方法中，一个操作的感受野直接决定了这个操作可以获得多少上下文信息，所以提升感受野可以为网络引入更多的上下文信息。

class Enhance(nn.Module):
    def __init__(self,inchannles=3,outchannles=3):
        super(Enhance, self).__init__()
        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.tanh=nn.Tanh()
        self.refine1= nn.Conv2d(inchannles, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.refine3= nn.Conv2d(20+4, outchannles, kernel_size=3,stride=1,padding=1)
        self.upsample = F.upsample_nearest
        self.batch1 = nn.InstanceNorm2d(100, affine=True)
    
    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(dehaze, 32)
        x102 = F.avg_pool2d(dehaze, 16)
        x103 = F.avg_pool2d(dehaze, 8)
        x104 = F.avg_pool2d(dehaze, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))
        return dehaze

import torch
import torch.nn as nn

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        self.w = nn.Parameter(torch.tensor(m, dtype=torch.float32), requires_grad=True)
        self.exp = torch.exp
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=bias)
class PALayer(nn.Module):
    def __init__(self, channel,re):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // re, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // re, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel,re):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // re, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel //re, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim,8)
        self.palayer = PALayer(dim,8)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x

        return res


# class CNet(nn.Module):
#     def __init__(self):
#         super(CNet, self).__init__()
#         self.model_pre_path='ConvNext-tiny'
#         self.config=ConvNextConfig().from_pretrained(self.model_pre_path)
#         self.encoder=ConvNextModel(self.config).from_pretrained(self.model_pre_path)
#         self.s=nn.PixelShuffle(2)
#         self.c1=nn.Conv2d(45, 12, kernel_size=3, stride=1, padding=1)
#         self.feat=DehazeBlock(default_conv, 768, 3)

        
        
#         self.relu = nn.LeakyReLU(0.1)


#     def forward(self, x2):
#         x2=self.encoder(x2,output_hidden_states=True)
#         f1,f2,f3,f4,x2=x2['hidden_states']


        
#         x2=self.s(x2)
        
#         x2=torch.cat((x2,f4),1)
#         x2=self.s(x2)
   

#         x2=torch.cat((x2,f3),1)
#         x2=self.s(x2)

   
#         x2=torch.cat((x2,f2),1)
#         x2=self.s(x2)#(45)

        
        

        
#         return x2



        
        
        
        
        return x

from model import DoubleConv,Down,Up,OutConv
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class RCBlock(nn.Module):
    def __init__(self,ins,out):
        super(RCBlock, self).__init__()
        self.rcbblock = nn.Sequential(     
                    #nn.LeakyReLU(0.1),
                    nn.Conv2d(ins, 64, kernel_size=3, padding=1),
                    #nn.BatchNorm2d(64),
                    RCAB(64,64),
                    RCAB(64,64),
                    RCAB(64,64),
                    # RCAB(64,64),
                    nn.Conv2d(64, out, kernel_size=3, padding=1),
                    )

    def forward(self, x):
        return x+self.rcbblock(x)





class CBlock(nn.Module):
    def __init__(self,ins,out):
        super(CBlock, self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(ins, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(),
            )
    

    def forward(self,x):
        return self.net(x)+x



class RC_CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(RC_CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self,mid):
        super(RCAB, self).__init__()

        self.body=nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1,bias=True),
            #nn.BatchNorm2d(mid),
            nn.ReLU(True),
            nn.Conv2d(mid, mid, kernel_size=3,  padding=1,bias=True),
            #nn.BatchNorm2d(mid),
        )
        self.tail=RC_CALayer(mid,mid//2)


    def forward(self, x):

        y=self.body(x)
        y=self.tail(y)

        
        return x + y

class RABGroup(nn.Module):
    def __init__(self,n,mid):
        super(RABGroup, self).__init__()
        body=[RCAB(mid) for _ in range(n)]
        body.append(nn.Conv2d(mid, mid, kernel_size=3,  padding=1,bias=True))
        self.net=nn.Sequential(*body)

    def forward(self,x):
        res=self.net(x)
        res=res+x
        return res
        
class MyNet(nn.Module):
    def __init__(self,n,n_feat):
        super(MyNet, self).__init__()
        self.n_feat=n_feat
        self.ins=nn.Conv2d(3, n_feat, kernel_size=3, bias=True, padding=1)
        self.trans1=nn.Conv2d(n_feat, n_feat//4, kernel_size=3, padding=1,bias=True)

        self.nets=nn.Sequential(
                    RABGroup(min(n*2,10),n_feat),
                    RABGroup(min(n*2,10),n_feat),
                    # RABGroup(min(n*2,10),n_feat),
                    # RABGroup(min(n*2,10),n_feat),
      
                    #nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1,bias=True),
                    )
        self.mid=self.n_feat//4
        self.net1=nn.Sequential(
                    RABGroup(n,self.mid),
                    RABGroup(n,self.mid),
                    #RABGroup(n,self.mid),

                    #nn.Conv2d(self.mid, self.mid, kernel_size=3, padding=1,bias=True),
                    )
        self.net2=nn.Sequential(
                    RABGroup(n,self.mid),
                    RABGroup(n,self.mid),
                    #RABGroup(n,self.mid),

                   # nn.Conv2d(self.mid,self.mid, kernel_size=3, padding=1,bias=True),
                    )
        self.net3=nn.Sequential(
                    RABGroup(n,self.mid),
                    RABGroup(n,self.mid),
                    #RABGroup(n,self.mid),

                    #nn.Conv2d(self.mid, self.mid, kernel_size=3, padding=1,bias=True),
                    )
        self.net4=nn.Sequential(
                    RABGroup(n,self.mid),
                    RABGroup(n,self.mid),
                    #RABGroup(n,self.mid),

                   # nn.Conv2d(self.mid, self.mid, kernel_size=3, padding=1,bias=True),
                    )
        self.nete=nn.Sequential(
                    RABGroup(min(n*2,10),self.n_feat),
                    RABGroup(min(n*2,10),self.n_feat),

                    nn.Conv2d(self.n_feat, self.n_feat, kernel_size=3, padding=1,bias=True),
                    )
        #self.enhance=Enhance(self.n_feat,self.n_feat)

        
        

        
    def forward(self, x):
        x=self.ins(x)
        #单通道提取
        
        xd=self.nets(x)

        x1,x2,x3,x4=torch.split(xd, split_size_or_sections=self.n_feat//4, dim=1)
        x1=self.net1(x1)+x1
        x2=self.net2(x2)+x2
        x3=self.net3(x3)+x3
        x4=self.net4(x4)+x4
        xs=torch.cat((x1,x2,x3,x4),1)
 
        

        x=self.nete(xs+x+xd)+x
        
        return x #64
        
class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.model_pre_path='ConvNext-tiny'
        self.config=ConvNextConfig().from_pretrained(self.model_pre_path)
        self.encoder=ConvNextModel(self.config).from_pretrained(self.model_pre_path)
        self.s=nn.PixelShuffle(2)
        self.c1=nn.Conv2d(45, 60, kernel_size=3, stride=1, padding=1)
        self.feat=DehazeBlock(default_conv, 768, 3)

        self.relu = nn.LeakyReLU(0.1)


    def forward(self, x2):
        x2=self.encoder(x2,output_hidden_states=True)
        f1,f2,f3,f4,x2=x2['hidden_states']

        x2=self.s(x2)

        x2=torch.cat((x2,f4),1)
        x2=self.s(x2)
   
        x2=torch.cat((x2,f3),1)
        x2=self.s(x2)

        x2=torch.cat((x2,f2),1)
        x2=self.s(x2)#(60)
        x2=self.c1(x2)
        x2=self.s(x2)#(15)

        return x2


class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        self.model_pre_path="./Swingv2"
        self.encoder=AutoModel.from_pretrained(self.model_pre_path)
        self.s=nn.PixelShuffle(2)

        self.relu = nn.LeakyReLU(0.1)
        self.att1=DehazeBlock(default_conv,256,3)
        self.att2=DehazeBlock(default_conv,192,3)
        self.att3=DehazeBlock(default_conv,112,3)
        self.conv= nn.Conv2d(90, 92, kernel_size=3, padding=1,bias=True)
        self.enhance=Enhance(15,15)

    def forward(self, input):
        a=self.encoder(input,output_hidden_states=True)
        x,layer_feature=a['hidden_states'][-1],a['hidden_states'][:-1]
        x, feature1, feature2, feature3 = x.transpose(1,2), layer_feature[0].transpose(1,2), layer_feature[1].transpose(1,2), layer_feature[2].transpose(1,2)
        x = torch.reshape(x, (x.shape[0], x.shape[1], int(np.sqrt(x.shape[2])), int(np.sqrt(x.shape[2]))))
        feature1 = torch.reshape(feature1, (feature1.shape[0], feature1.shape[1], int(np.sqrt(feature1.shape[2])), int(np.sqrt(feature1.shape[2]))))
        feature2 = torch.reshape(feature2, (feature2.shape[0], feature2.shape[1], int(np.sqrt(feature2.shape[2])), int(np.sqrt(feature2.shape[2]))))
        feature3 = torch.reshape(feature3, (feature3.shape[0], feature3.shape[1], int(np.sqrt(feature3.shape[2])), int(np.sqrt(feature3.shape[2]))))
        
        x = self.s(x)                   # [8, 256, 16, 16] 
        x=self.att1(x)
        
        x = torch.cat((x, feature3), 1) 
        x = self.s(x)                       # [8, 192, 32, 32]
        x=self.att2(x)
        
        x = torch.cat((x, feature2), 1) 
        x = self.s(x)                       # [8, 112, 64, 64]
        x=self.att3(x)
        
        x = torch.cat((x, feature1), 1)             # [8, 240, 64, 64] = 112+128
        x = self.s(x)  # [8, 60, 128, 128]
        #x=self.conv(x)
        
        x = self.s(x)
        x=self.enhance(x)

        return x  #23large 15base

##ALLNEXt
class AllNet(nn.Module):
    def __init__(self):
        super(AllNet, self).__init__()

        self.cnet=SNet()
        self.mynet=MyNet(10,32)#CNet()#FFA(3,10)#MyNet(10,32)
    
        self.relu=nn.LeakyReLU(0.1)
        self.trans1=nn.Conv2d(15,32, kernel_size=3, stride=1, padding=1)
        self.lc1=LSKblock(15)
        self.lc2=LSKblock(32)


        self.tail=nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(47, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, x):


        x1=self.cnet(x)#15
        x2=self.mynet(x)#32

        # x1=self.lc1(x1)
        # x2=self.lc2(x2)
        
        x11=self.trans1(x1)




        x3=torch.cat((x1,x2),1)
        #x3=x11+x12

        #x=self.rcb(x3)
        x=self.tail(x3)
        
        return x,F.smooth_l1_loss(x11,x2)
##MyUnet
class MyUNet(nn.Module):
    def __init__(self,factor=1, bilinear=False):
        super(MyUNet, self).__init__()
        ####UNet####
        in_channels=8
        mid_channels=4
        out_channels=6
        self.model_pre_path='Swingv2'
        self.encoder=AutoModel.from_pretrained(self.model_pre_path)

        self.trans=nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)

        self.pix=nn.PixelShuffle(2)
        self.pix2=nn.PixelShuffle(2)

        self.trans=nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)
        
        #self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        #self.rab=RABGroup(4,1024)
        

        self.enhance=Enhance(8,8)#nn.Conv2d(64, 3, kernel_size=1)


        ####mynet#######
        self.mynet=MyNet()
        self.tail=nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(72, 3, kernel_size=7, padding=0), nn.Tanh())
        # self.rcb=RCBlock()


    def forward(self, x):
        x2=self.mynet(x)
        a=self.encoder(x,output_hidden_states=True)
        x,layer_feature=a['hidden_states'][-1],a['hidden_states'][:-1]
        x, feature1, feature2, feature3 = x.transpose(1,2), layer_feature[0].transpose(1,2), layer_feature[1].transpose(1,2), layer_feature[2].transpose(1,2)
        x = torch.reshape(x, (x.shape[0], x.shape[1], int(np.sqrt(x.shape[2])), int(np.sqrt(x.shape[2]))))
        f1 = torch.reshape(feature1, (feature1.shape[0], feature1.shape[1], int(np.sqrt(feature1.shape[2])), int(np.sqrt(feature1.shape[2]))))
        f2 = torch.reshape(feature2, (feature2.shape[0], feature2.shape[1], int(np.sqrt(feature2.shape[2])), int(np.sqrt(feature2.shape[2]))))
        f3 = torch.reshape(feature3, (feature3.shape[0], feature3.shape[1], int(np.sqrt(feature3.shape[2])), int(np.sqrt(feature3.shape[2]))))

        #x=self.rab(x)
        #上采样恢复
        x=self.up1(x,f3)
        
        x=self.up2(x,f2)
        
        x=self.up3(x,f1)
        
        x=self.pix(x)
        x=self.pix2(x)
        
        x=self.enhance(x)
        
        x=torch.cat((x,x2),1)
        x=self.tail(x)
        

        #x=self.trans(x)

        #x=self.out(self.alpha*x+(1-self.alpha)*x2)


        return x

#我的cunet
class MyCUNet(nn.Module):
    def __init__(self,factor=1, bilinear=False):
        super(MyCUNet, self).__init__()
        ####UNet####
        in_channels=8
        mid_channels=4
        out_channels=6
        self.model_pre_path='ConvNext-base'
        self.config=ConvNextConfig().from_pretrained(self.model_pre_path)
        self.encoder=ConvNextModel(self.config).from_pretrained(self.model_pre_path)

        self.trans=nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 3, bilinear)

        self.ct=nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4)
        self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))

        self.out=Enhance()#nn.Conv2d(64, 3, kernel_size=1)

        #self.att1=DehazeBlock(default_conv,512,3)
        #self.att2=DehazeBlock(default_conv,256,3)
        #self.att3=DehazeBlock(default_conv,128,3)
        self.att4=DehazeBlock(default_conv,3,3)


        ####mynet#######
        self.mynet=MyNet()
        #self.rcb=RCBlock(3,3)


    def forward(self, x):
        x=self.trans(x)
        #x_c=x.clone()
        x2=self.mynet(x)
        x=self.encoder(x,output_hidden_states=True)
        f1,f2,f3,f4,x1=x['hidden_states']
        #上采样恢复
        x=self.up1(x1,f4)

        
        x=self.up2(x,f3)


        
        x=self.up3(x,f2)


        f1=self.ct(f1)


        x=self.up4(x,f1)
        x=self.att4(x)

        x=self.out(self.alpha*x+(1-self.alpha)*x2)
        x=self.out(x)
        #x=self.rcb(x)

        return x
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)  # ,

            # nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
