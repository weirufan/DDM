from __future__ import print_function, division
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torchsummary
from Data_Loader import *
from demixing_diffusion_pytorch import Unet
from PIL import Image
#from perceptual_loss import Perceptual_loss134
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from focal_frequency_loss import FocalFrequencyLoss
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
#from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DIVICES"] = '1'


num_workers = 0
pin_memory = False
batch_size=4
valid_size = 0.1
T=1000
#n_epoch=200
#Lr=1e-4
t_data = 'F:\\MLdata\\data_2019-2020.9\\DATA6.1(graycell)\\OUTORI\\'
l_data = 'F:\\MLdata\\data_2019-2020.9\\DATA6.1(graycell)\\IN\\'


Model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=1,
    with_time_emb=True,
    residual=True)


train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")


if train_on_gpu:
    pin_memory = True

Training_Data = Images_Dataset_folder(t_data, l_data)
num_train = len(Training_Data)
indices = list(range(int(np.floor(valid_size * num_train))))
valid_idx = [i*int(np.floor(1/valid_size))+5 for i in indices]
train_idx = list(range(num_train))
train_idx = [i for i in train_idx if i not in valid_idx]
train_sampler = train_idx
valid_sampler = valid_idx

if torch.cuda.device_count() > 1:
    Model = nn.DataParallel(Model.to(device))
else:
    Model.to(device)


#torchsummary.summary(Gen_model, input_size=(1, 256, 256))

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory)

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory)
#G_criterion = torch.nn.SmoothL1Loss()
class NLPCC(nn.Module):
    def __init__(self):
        super(NLPCC,self).__init__()
        
    def forward(self, X, Y):
        X=X-torch.mean(X,dim=(2,3),keepdim=True)
        Y=Y-torch.mean(Y,dim=(2,3),keepdim=True)
        CovXY=torch.sum(torch.mul(X,Y),dim=(2,3))
        CovX=torch.sqrt(torch.sum(torch.pow(X,2),dim=(2,3)))
        CovY=torch.sqrt(torch.sum(torch.pow(Y,2),dim=(2,3)))
        PCC=torch.div(CovXY,torch.mul(CovX,CovY))
        #loss=torch.mean(-torch.log((1+torch.mean(PCC,dim=1))/2),dim=0)
        loss=torch.mean(1-torch.mean(PCC,dim=1),dim=0)
        return loss
Loss_NLPCC=NLPCC()

class NSSIM(nn.Module):
    def __init__(self):
        super(NSSIM,self).__init__()

    def forward(self, X, Y):
        m1=torch.mean(X,dim=(2,3),keepdim=True)
        m2=torch.mean(Y,dim=(2,3),keepdim=True)
        sig1=torch.mean(torch.pow(X-m1,2),dim=(2,3),keepdim=True)
        sig2=torch.mean(torch.pow(Y-m2,2),dim=(2,3),keepdim=True)
        sig1_2=torch.mean(torch.mul((X-m1),(Y-m2)),dim=(2,3),keepdim=True)
        t1=2*torch.mul(m1,m2)+0.0001
        t2=2*sig1_2+0.0009
        t3=torch.pow(m1,2)+torch.pow(m2,2)+0.0001
        t4=sig1+sig2+0.0009
        SSIM=torch.div(torch.mul(t1,t2),torch.mul(t3,t4))
        temp=torch.mean(SSIM,dim=1)
        loss=torch.mean(1-temp,dim=0)
        return loss
Loss_NSSIM=NSSIM()
Loss_L2=nn.MSELoss()
def pre_Gen(Model, train_loader, valid_loader, T):
    #Model.load_state_dict(torch.load('./DATA7.25(CIFAR_LED_noise)/model/Unet_' + 'Model' + '.pth'))
    Model.load_state_dict(torch.load('./model/Unet_' + 'Model_maxstep' + '.pth'))
    valid_loss = 0.0
    Model.eval()
    with torch.no_grad(): #to increase the validation process uses less memory
        for i, data in enumerate(valid_loader,0):
            x, y = data
            x, y = x.to(device), y.to(device)
            yt=x
            for t in range(T):
                s=(T-t)*torch.ones(x.size(0),1,1,1).to(device)
                y0_pred = Model(yt,torch.squeeze(s))
                Ds=(s/T*x+(1-s/T)*y0_pred)
                Ds_1=((s-1)/T*x+(1-(s-1)/T)*y0_pred)
                yt=yt-Ds+Ds_1
            
            for k in range(x.size(0)):
                temp_y_pred=torch.unsqueeze(yt[k,:,:,:],dim=0)
                torchvision.utils.save_image(temp_y_pred, os.path.join('predicts', (str(valid_sampler[batch_size*i+k]+1) + '.png')),normalize=True)
                pcc=1-Loss_NLPCC(torch.unsqueeze(y[k,:,:,:],dim=0),temp_y_pred)
                vs=str(valid_sampler[batch_size*i+k]+1)+' '+str(pcc.item())+'\n'
                with open('pcc.txt','a') as f:
                    f.write(vs)
                ssim=1-Loss_NSSIM(torch.unsqueeze(y[k,:,:,:],dim=0),temp_y_pred)
                vs=str(valid_sampler[batch_size*i+k]+1)+' '+str(ssim.item())+'\n'
                with open('ssim.txt','a') as f:
                    f.write(vs)
                L2=Loss_L2(torch.unsqueeze(y[k,:,:,:],dim=0),temp_y_pred)
                vs=str(valid_sampler[batch_size*i+k]+1)+' '+str(L2.item())+'\n'
                with open('L2.txt','a') as f:
                    f.write(vs)
            del x, y, yt, s, Ds, Ds_1, y0_pred, temp_y_pred
            torch.cuda.empty_cache()
    print('Finished')


if __name__ == '__main__':
    
    pre_Gen(Model, train_loader, valid_loader, T)
