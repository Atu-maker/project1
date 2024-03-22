#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from model import U_Net,RDB,GridDehazeNet,FFA,LSKblock
from transformers import ConvNextModel, ConvNextConfig
from torchvision.models import vgg16

from torchvision.utils import make_grid
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import gc

import cv2
#from swin_transformer_v2 import SwinTransformerV2
from Block import OutlookAttention
import logging
#from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.scheduler.cosine_lr import CosineLRScheduler
from mymodel import AllNet,UNet,MyUNet,Discriminator
from pytorch_msssim import msssim


# In[2]:


# 创建一个过滤器，只允许INFO级别的日志通过
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO
# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期格式
    filename='train.log',  # 日志文件名
    filemode='a'  # 追加模式，如果你想要覆盖旧的日志文件，可以使用 'w'
)
# 添加过滤器到根日志记录器
logging.getLogger().addFilter(InfoFilter())


# In[3]:


#####种子####
def seed_all(seed): 
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed=42
seed_all(seed)
device = torch.device("cuda:0" ) if torch.cuda.is_available() else torch.device("cpu")
model_save_path = 'model.pth'
#device='cpu'
device


# In[4]:


from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF
import random
import os
from utils_test import cropping, cropping_ohaze,image_stick, image_stick_ohaze
def augment(hazy, clean):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
    rotate_degree = random.choice([90, 180, 270])
    '''Rotate'''
    if augmentation_method == 0:
        hazy = transforms.functional.rotate(hazy, rotate_degree)
        clean = transforms.functional.rotate(clean, rotate_degree)
        return hazy, clean
    '''Vertical'''
    if augmentation_method == 1:
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
        hazy = vertical_flip(hazy)
        clean = vertical_flip(clean)
        return hazy, clean
    '''Horizontal'''
    if augmentation_method == 2:
        horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        hazy = horizontal_flip(hazy)
        clean = horizontal_flip(clean)
        return hazy, clean
    '''no change'''
    if augmentation_method == 3 or augmentation_method == 4 or augmentation_method == 5:
        return hazy, clean
# 定义转换函数
def image_loader(image_pathsize,mode):
    process=transforms.Compose([transforms.ToTensor()])
    if mode=='train' or mode=='valid':
        haze_image_path=image_pathsize[0]
        clean_image_path=image_pathsize[1]
        # 打开并加载图片
        hzimg = Image.open(haze_image_path)
        cleanimg = Image.open(clean_image_path)
        return hzimg,cleanimg
    if mode=='test':
        haze_image_path=image_pathsize
        # 打开并加载图片
        hzimg = Image.open(haze_image_path)
        return hzimg
        
def get_data_list(train_path='./',train_label_path=None,mode='train',size=(512,512)):
    train_path = train_path
    # 获取文件夹内容
    file_list = os.listdir(train_path)
    # 过滤出文件，排除子文件夹
    png_name = [f for f in file_list if os.path.isfile(os.path.join(train_path, f))]
    png_name.sort()
    if mode=='train' or mode=='valid':
        train_label_path = train_label_path
        # 获取文件夹内容
        file_list = os.listdir(train_label_path)
        # 过滤出文件，排除子文件夹
        label_name = [f for f in file_list if os.path.isfile(os.path.join(train_path, f))]
        label_name.sort()
        #训练集数据
        train_datas=[]
        label_datas=[]
        lists=list(zip(png_name,label_name))
        lists=[list(a) for a in lists]
        for imgname in tqdm(lists):
            imgname[0]=train_path+imgname[0]
            imgname[1]=train_label_path+imgname[1]
            # 调用函数读取图像数据
            input_data =image_loader(imgname,mode)#read_img(train_path+'/'+imgname) * 2 - 1 #image_loader(train_path+'/'+imgname,'train')
            train_datas.append(input_data[0])
            label_datas.append(input_data[1])
        return train_datas,label_datas
    if mode=='test':
        datas=[]
        lists=png_name
        for imgname in tqdm(lists):
            imgname=train_path+imgname
            # 调用函数读取图像数据
            input_data =image_loader(imgname,mode)#read_img(train_path+'/'+imgname) * 2 - 1 #image_loader(train_path+'/'+imgname,'train')
            datas.append(input_data)
        return datas


# In[5]:


train_data,train_label=get_data_list(train_path="./train/hzImage/",train_label_path="./train/lbImage/",mode='train',size=(512,512))


# In[6]:


eval_data,eval_label=get_data_list(train_path="./eval/hzImage/",train_label_path="./eval/lbImage/",mode='train',size=(512,512))


# In[7]:


# train_data=train_data[:40]
# train_label=train_label[:40]


# In[8]:


class MyLoader:
    def __init__(self, data,y=None,mode='train',crop_size=6,size=(512,512)):
        self.data=data
        self.y=y
        self.mode=mode
        self.crop_size=crop_size
        self.cropping=cropping
        self.process=transforms.Compose([transforms.ToTensor()])
        self.process2=transforms.Compose([transforms.Resize(size),transforms.ToTensor()])
        self.size=size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seed=torch.randn(5)
        if self.mode=='train':
            data=self.data[idx]
            y=self.y[idx]
            #crop a patch
            i,j,h,w = transforms.RandomCrop.get_params(data, output_size = self.size)
            data = TF.crop(data, i, j, h, w)
            y = TF.crop(y, i, j, h, w)
            #data argumentation
            data, y = augment(data, y)#放入datasets
      
        
            data = self.process(data)
            y =self.process(y)
            return {'data':data.to(device) , 'label':y.to(device)}
        if self.mode=='eval':
            data=self.data[idx]
            y=self.y[idx]
            #data,_=self.cropping(data,self.crop_size)
            #data=[a.to(device) for a in data]
            #y,_=self.cropping(y,self.crop_size)
            #y=[a.to(device) for a in y]
            data=self.process2(data).to(device)
            y=self.process2(y).to(device)

            
            return {'data':data , 'label':y}

        if self.mode=='test':
            data=self.data[idx]
            data = self.process(data)
            data=data[:3,:,:]
            if data.size(1)==6000:
                data=data.permute(0,2,1)
            data,_=self.cropping(data,self.crop_size)
            data=[a.to(device) for a in data]
            return {'data':data }


# In[9]:


seed_all(seed)
train_dataset=MyLoader(train_data,train_label,mode='train',size=(256,256))
train_loader=DataLoader(train_dataset,batch_size=8,shuffle=True)

eval_dataset=MyLoader(eval_data,eval_label,mode='eval',size=(512,512))
eval_loader=DataLoader(eval_dataset,batch_size=1,shuffle=False)

# test_dataset=MyLoader(test_data,mode='test')
# test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)


# In[10]:


import torch
import torch.nn.functional as F
from math import log10
import cv2
import numpy as np
import torchvision
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms


def to_psnr(frame_out, gt):
    mse = F.mse_loss(frame_out, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True,win_size=11,channel_axis=2) for ind in range(len(dehaze_list))]

    return ssim_list


# In[11]:


# 实例化模型、损失函数和优化器
#model = DehazeNet().to(device)
model=AllNet().to(device)
#model=GridDehazeNet().to(device)
#model=UNet(3,3).to(device)
#model=MyUNet().to(device)
#model=AODnet().to(device)
#model=FFA(gps=3,blocks=6).to(device)
loss_fn =nn.L1Loss().to(device)#nn.L1Loss() #nn.MSELoss()
optimizer =torch.optim.AdamW(params=model.parameters(), eps=1e-8, 
                                betas=(0.9, 0.999), lr=0.00004, weight_decay=1e-8) #optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-8)


split=False#训练时是否切块
vgg=True


# In[12]:


from utils_test import cats
import gc
def eval(model,eval_loader,split=False):
    #model.eval()
    psnrs=[]
    ssims=[]
    s=0
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            if split:
                for c,y in tqdm(list(zip(batch['data'],batch['label']))):
                    output,_ = model(c)#.detach().cpu().numpy()
                    psnrs.extend(to_psnr(output,y))
                    ssims.extend(to_ssim_skimage(output,y))
            else:
                output,_ = model(batch['data'])#.detach().cpu().numpy()
                psnrs.extend(to_psnr(output,batch['label']))
                ssims.extend(to_ssim_skimage(output,batch['label']))
    print(f"eval_ssim:{sum(ssims)/len(ssims)}")  
    print(f"eval_psnr:{sum(psnrs)/len(psnrs)}")
    logging.info(f"eval_ssim:{sum(ssims)/len(ssims)},eval_psnr:{sum(psnrs)/len(psnrs)}")
    #model.train()
    return sum(ssims)/len(ssims)
        


# In[13]:


# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)


# In[14]:


if vgg:
    vgg_model = vgg16(weights=True)
    #weights_path = "vgg_model/vgg16-397923af.pth"
    # 更新模型的权重
    #vgg_model.load_state_dict(torch.load(weights_path))
    vgg_model = vgg_model.features[:16].to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()


# In[15]:


#GAN训练方式
DNet=Discriminator().to(device)
D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[5000,7000,8000], gamma=0.5)
msssim_loss = msssim


# In[ ]:


best_score=0
num_epochs=12000

scheduler =CosineLRScheduler(optimizer,t_initial=num_epochs, lr_min=2.5e-6) #CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
for epoch in tqdm(range(num_epochs)):
    scheduler.step(epoch=epoch)
    scheduler_D.step()
    DNet.train()
    model.train()
    s=0
    psnrs=0
    for batch in train_loader:
        s+=1
        inputs=batch['data']
        targets=batch['label']
        
        outputs,simLoss = model(inputs)
        #####鉴别器####
        DNet.zero_grad()
        real_out = DNet(targets).mean()
        fake_out = DNet(outputs).mean()
        D_loss = 1 - real_out + fake_out
        # no more forward
        D_loss.backward(retain_graph=True)
        #####生成器#####
        model.zero_grad()
        loss =F.smooth_l1_loss(outputs, targets) #F.smooth_l1_loss(outputs, targets)#loss_fn(outputs, targets)
        adversarial_loss = torch.mean(1 - fake_out)
        msssim_loss_ = -msssim_loss(outputs, targets, normalize=True)
        if vgg:
            perceptual_loss = loss_network(outputs, targets)
        else:
            perceptual_loss =0
            
        # if epoch>=1:
        #     loss=loss+0.0005 * adversarial_loss+0.01*perceptual_loss+0.5*msssim_loss_
        # else:
        total_loss=loss+0.0005 * adversarial_loss+0.01*perceptual_loss+0.5*msssim_loss_+0.4*simLoss
        total_loss.backward()

        ##########
        D_optim.step()
        optimizer.step()

 
    if epoch%25==0 and epoch!=0:
        logging.info(f"EPOCH:{epoch}")
        score=eval(model,eval_loader,False)
        if score>best_score:
            
            torch.save(model.state_dict(), model_save_path)
            best_score=score
            print(f"best_score:{best_score},save!")
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()},  D_Loss:{D_loss.item()}')
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}')



# In[ ]:


model_save_path2="model2.pth"
torch.save(model.state_dict(), model_save_path2)


# # In[16]:


# #测试集
# model_save_path2="model2.pth"
# test_data=get_data_list(train_path="./test_b/",mode='test',size=None)
# test_dataset=MyLoader(test_data,mode='test',crop_size=4)
# test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)

# testmodel=AllNet().to(device)#DehazeNet().to(device)#DehazeNet().to(device)#GridDehazeNet().to(device)#DehazeNet().to(device) #DehazeNet().to(device)#GridDehazeNet()#DehazeNet()#U_Net()
# testmodel.load_state_dict(torch.load(model_save_path2))


# # In[17]:


# from utils_test import cats
# from torchvision.utils import save_image as imwrite
# from utils_test import image_stick
# def test(test_loader, model):
#     torch.cuda.empty_cache()
#     cnt=0
#     res=[]
#     for batch in tqdm(test_loader):
#         indices=batch['data']
#         img_list=[]
#         for i in tqdm(range(len(indices))):
#             with torch.no_grad():
#                 out= model(indices[i])[0].detach().cpu().numpy()
#                 img_list.append(out)
            
#         img_list=[torch.tensor(a,device='cpu') for a in img_list]
        
#         #img_list=cats(img_list,vi=False)
#         img_list=image_stick(img_list,False)
#         if cnt%2==0:
#             img_list=img_list.permute(0,1,3,2)
#         one_t = torch.ones_like(img_list[:,0,:,:])
#         one_t = one_t[:, None, :, :]
#         img_t = torch.cat((img_list, one_t) , 1)
#         cnt+=1
#         res.append(img_t)
#     for index,i in enumerate(tqdm(res)):
#         imwrite(i.squeeze(),f'./result/{index}.png')
                
    
  


# # In[19]:


# test(test_loader, testmodel)


# # In[ ]:




