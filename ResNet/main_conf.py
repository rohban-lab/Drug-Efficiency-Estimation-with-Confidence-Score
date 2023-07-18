import pandas as pd
import math
import sys, os, glob, shutil
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch  
import torch.nn as nn
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from IPython import display
from model import Conv,Conv_confidence
from  dataset import *
from helper import *
from torch.nn.modules.batchnorm import _BatchNorm
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

fcs=[2]
model=Conv_confidence((3,224,224),8,'resnet18',fcs,fc_drop_rate=0.5)
model.to(DEVICE)
TRAIN=True
path="Put image path here"
store="Put store path here"
transforms = [ torchvision.transforms.Resize([224,224],interpolation=torchvision.transforms.InterpolationMode.NEAREST),]
transforms = torchvision.transforms.Compose(transforms)
loss=nn.MSELoss()

#You can change training strategy or metric to evaluate by using
#different function from helper and dataset files

if TRAIN:
  train_dataset=DrugdatasetNew(path,transforms)
  train_loader = DataLoader(train_dataset,batch_size=1536, shuffle=True)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)
  train_confidence(model, train_loader, train_loader, optimizer,loss,mode, num_epochs=100)
  #confidence_val(model, train_loader, train_loader, optimizer,loss,mode, num_epochs=100)
else:
   test_dataset=DrugdatasetNew_test(path,transforms)
   test_loader = DataLoader(test_dataset,batch_size=512, shuffle=True)
   mode='res18_Score_confTOTAL_CONF3.pth'
   model.load_state_dict(torch.load(store+mode))
   calibration_eval(model, test_loader, test_loader, optimizer,loss, num_epochs=200)
   #test_conf(model, test_loader, test_loader, optimizer,loss, num_epochs=200)
   #test_query_confidence(model, test_dataset,"GS-441524","3.0","HRCE-2","27","P39",[107,106,108,102])












