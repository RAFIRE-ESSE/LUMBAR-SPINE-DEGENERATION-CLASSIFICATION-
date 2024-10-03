import argparse
import os
import random,numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

workers = 2
batch_size = 1
nz = 100
num_epochs = 5
lr = 0.001
beta1 = 0.5
ngpu=1
ngf,nc = 3,3
ndf = 64

#transforms.Resize(size=(config.INPUT_HEIGHT,config.INPUT_WIDTH))
transform = transforms.Compose(
    [transforms.Resize(256),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root=f'./train_wolf/subarticular_stenosis',transform=transform)
dataloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
"""testset_ = torchvision.datasets.ImageFolder(root=f'./train_wolf/right_neural_foraminal_narrowing',transform=transform)
dataloader_ = torch.utils.data.DataLoader(testset_,batch_size=1,shuffle=True,num_workers=2)"""
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print(dataloader.dataset.classes)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.devil=nn.Sequential(
			nn.Conv2d(3, 4,3),
			nn.BatchNorm2d(4),
			nn.ReLU(),
			nn.AvgPool2d(2, 2),

			nn.Conv2d(4, 8, 3),
			nn.BatchNorm2d(8),
			nn.AvgPool2d(2, 2),

			nn.Conv2d(8, 16, 3),
			nn.BatchNorm2d(16),
			nn.AvgPool2d(2, 2),

			nn.Conv2d(16, 32, 3),
			nn.BatchNorm2d(32),
			nn.AvgPool2d(2, 2),

			nn.Conv2d(32, 64, 3),
			nn.BatchNorm2d(64),
			nn.AvgPool2d(2, 2),

			nn.Conv2d(64, 128, 3),
			nn.BatchNorm2d(128),
			nn.AvgPool2d(2, 2),

			nn.Flatten(),

			nn.Linear(512, 64),
			nn.Linear(64, 32),
			nn.Linear(32, 16),
			nn.Linear(16, 3)
			)

	def forward(self,x):
		return self.devil(x)


netD = Discriminator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)

#netD.load_state_dict(torch.load(f"./D_.pth"))

criterion,img_devil = nn.CrossEntropyLoss(),0
fixed_noise = torch.randn(1, nz, 1, 1, device=device)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
schedulerD=torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.86)

if __name__=="__main__":
	z_w_=[]
	while(True):
		z_,z,z_w=0,0,{"normal": 0,"Moderate": 0,"Severe": 0}
		for i, data in enumerate(dataloader, 0):
			optimizerD.zero_grad()
			real_cpu=data[0].to(device)
			label=data[1].to(device)
			output = netD(real_cpu)
			errD_real = criterion(output,label)
			errD_real.backward()
			#optimizerD.zero_grad()
			optimizerD.step()
			wolf_z=torch.argmax(netD(real_cpu),dim=1).view(-1)
			#print(wolf_z,label)
			for i,j in zip(wolf_z,label):
				if i==j:
					if dataloader.dataset.classes[i]=="normal":
						z_w["normal"]+=1
					elif dataloader.dataset.classes[i]=="Moderate":
						z_w["Moderate"]+=1
					elif dataloader.dataset.classes[i]=="Severe":
						z_w["Severe"]+=1
					z+=1
				z_+=1
		z_w_.append(z)
		if len(z_w_)>=3:
			if len([True for i in range(1,3) if z_w_[len(z_w_)-i]<=z_w_[len(z_w_)-3]+2 and z_w_[len(z_w_)-i]>=z_w_[len(z_w_)-4]-3])==2:
				z_w_=[]
				print(optimizerD.param_groups[0]["lr"])
				schedulerD.step()
				print(optimizerD.param_groups[0]["lr"])

		print(z_,z,z_w)	
			



