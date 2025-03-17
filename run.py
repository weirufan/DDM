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
batch_size=12
valid_size = 0.1
T=1000
n_epoch=200
Lr=1e-4
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
train_sampler = SubsetRandomSampler(train_idx)
#valid_sampler = SubsetRandomSampler(valid_idx)
valid_sampler = valid_idx

if torch.cuda.device_count() > 1:
    Model = nn.DataParallel(Model.to(device))
else:
    Model.to(device)


def init_weight(layer):
    if type(layer)==nn.Conv2d:
        nn.init.normal_(layer.weight,mean=0,std=0.5)
        #torch.nn.init.orthogonal_(layer.weight, gain=1)
    elif type(layer)==nn.Linear:
        nn.init.uniform_(layer.weight,a=-1,b=0.1)
        nn.init.constant_(layer.bias,0.1)
#Model.apply(init_weight)

#torchsummary.summary(Model, input_size=(1, 256, 256))

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory)

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory)


New_folder = './model'

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)

#######################################################
#checking if the model exists and if true then delete
#######################################################

read_model_path = './predicts'

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

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
        #loss=torch.mean(torch.mean(-torch.log((1+PCC)/2),dim=1),dim=0)
        #loss=torch.mean(-torch.log((1+torch.mean(PCC,dim=1))/2),dim=0)
        loss=torch.mean(1-torch.mean(PCC,dim=1),dim=0)
        return loss
Loss_NLPCC=NLPCC()
Loss_NSSIM=NSSIM()
Loss_MSLL=MS_SSIM_L1_LOSS(data_range=2,compensation=0.1)
Loss_FFL=FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)

def train_Net(Model, Criterion, Optimizer, Explr, n_epoch, batch_size, train_loader, valid_loader, T):
    Model.load_state_dict(torch.load('./model/Unet_' + 'Model_maxstep' + '.pth'))
    comp=1e6
    for epoch in range(n_epoch):

        train_loss = 0.0
        valid_loss = 0.0
        Model.train()
        for i, data in enumerate(train_loader, 0):
            x, y = data
            x, y = x.to(device), y.to(device)
            Optimizer.zero_grad()
            t=torch.ceil(T*(1-torch.rand(x.size(0),1,1,1))).to(device)
            yt=t/T*x+(1-t/T)*y
            y_pred = Model(yt,torch.squeeze(t))
            Loss=Criterion(y_pred,y)
            train_loss += Loss.item() * x.size(0)
            Loss.backward()
            Optimizer.step()
            print('Epoch: {} \tStep: {} \tTraining Loss: {:.6f}'.format(epoch+1, (i+1), train_loss/(i+1)/batch_size))
            del x, y, t, yt, y_pred, Loss
            torch.cuda.empty_cache()
        Explr.step() 
        torch.save(Model.state_dict(),'./model/Unet_' + 'Model_maxstep' + '.pth')
        Model.eval()
        with torch.no_grad(): #to increase the validation process uses less memory
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                yt=x
                for t in range(T):
                    s=(T-t)*torch.ones(x.size(0),1,1,1).to(device)
                    y0_pred = Model(yt,torch.squeeze(s))
                    Ds=(s/T*x+(1-s/T)*y0_pred)
                    Ds_1=((s-1)/T*x+(1-(s-1)/T)*y0_pred)
                    yt=yt-Ds+Ds_1
                Loss=1-Loss_NLPCC(yt,y)
                grid = torchvision.utils.make_grid(yt, nrow=4, padding=2, pad_value=0, normalize=True, range=None, scale_each=False)      
                torchvision.utils.save_image(grid, os.path.join('predicts', 'temp.png'))
                valid_loss += Loss.item() * x.size(0)
                del x, y, yt, s, Ds, Ds_1, y0_pred, Loss
                torch.cuda.empty_cache()
                break
                
        train_loss = train_loss / len(train_idx)
        valid_loss = valid_loss / len(valid_idx)
        vs=str(epoch+1)+' Train: '+str(train_loss)+' Valid: '+str(valid_loss)+'\n'
        with open('train.txt','a') as f:
            f.write(vs)
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 1, n_epoch, train_loss, valid_loss))
        if comp>valid_loss:
            comp=valid_loss
            torch.save(Model.state_dict(),'./model/Unet_' + 'Model_beststep' + '.pth')
        
    print('Finished Training')
   
   
if __name__ == '__main__':
    Criterion = torch.nn.L1Loss()
    #Criterion = Perceptual_loss134()
    #Criterion = torch.nn.SmoothL1Loss()
    #Optimizer = optim.Adam(Model.parameters(), lr=Lr, betas=(0.9, 0.999))
    Optimizer = optim.AdamW(Model.parameters(), lr=Lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5, amsgrad=True)
    #Optimizer = optim.Adadelta(Model.parameters(), lr=1, rho=0.9, eps=1e-06, weight_decay=5e-3)
    #Explr = optim.lr_scheduler.OneCycleLR(Optimizer, max_lr=Lr, steps_per_epoch=len(train_loader), epochs=15)
    #Explr=torch.optim.lr_scheduler.StepLR(Optimizer, step_size=5, gamma=0.6, last_epoch=-1)
    Explr=optim.lr_scheduler.ExponentialLR(Optimizer, gamma=0.98, last_epoch=-1)  
    train_Net(Model, Criterion, Optimizer, Explr, n_epoch, batch_size, train_loader, valid_loader, T)